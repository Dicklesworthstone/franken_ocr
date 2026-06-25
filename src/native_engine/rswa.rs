//! R-SWA — `SlidingWindowLlamaAttention`, the load-bearing ring-buffer kernel
//! ([SPEC-090..096], PROPOSED_ARCHITECTURE.md §6.8; OQ-1/2/3/13).
//!
//! "R-SWA" = **R**etain-prefix **S**liding-**W**indow **A**ttention: the entire
//! prefill (BOS + visual + prompt) is the **reference** block — read-only, never
//! evicted — and only the *generated* tail occupies a fixed `W = 128`-slot ring
//! buffer. Plain MHA: `num_heads = 10`, `num_kv_heads = 10`
//! (`num_key_value_groups = 1`, `repeat_kv` is a no-op), `head_dim = 128`, scale
//! `1/sqrt(128)`, no QKV bias.
//!
//! Three regimes (per layer, per forward):
//!  1. **True prefill** — record the whole prefill as the reference block and set
//!     `prefill_len`; the causal prefill attention itself is the SDPA path in
//!     [`super::nn::sdpa`], so this module only *captures* the prefill K/V here.
//!  2. **Warm-up decode** — the first `W = 128` decode steps **append** K/V into
//!     the ring (no eviction); each query attends over the full reference block +
//!     all decoded tokens so far. `ring_len` grows `0 -> W`.
//!  3. **Steady-state ring** — step `W+1` onward: overwrite slot
//!     `prefill_len + ring_pos` in place, `ring_pos = (ring_pos + 1) % W`; the
//!     cache no longer grows.
//!
//! Decode applies **NO causal mask** — the window is enforced physically by the
//! ring overwrite. Scores are `Q · (referenceK ++ ringK)`, softmax over the
//! union, weighted sum of `(referenceV ++ ringV)`. The reference block is large
//! (worst-case `m ~= 32768 - 128`), so its contribution is folded with an
//! **online / streaming softmax** that never materializes the full score row.
//!
//! [SPEC-095 PORT INVARIANT]: RoPE uses the TRUE absolute `position_ids`, not the
//! ring slot — the physical slot is decoupled from the logical position. RoPE is
//! applied by the caller (decoder) *before* K reaches this kernel; we keep
//! `position_ids` only as provenance / a sanity hook, never re-deriving phase
//! from the ring slot.

use super::tensor::Mat;
use crate::error::{FocrError, FocrResult};

/// Query heads (== KV heads; plain MHA, `num_key_value_groups = 1`). [SPEC-090]
pub const NUM_HEADS: usize = 10;
/// Per-head dimension for Q/K/V. [SPEC-090] (`head_dim = 1280 / 10 = 128`).
pub const HEAD_DIM: usize = 128;
/// Ring window `W` — slots reserved for the generated tail. [SPEC-094]
pub const RING_WINDOW: usize = 128;

/// Attention scale `1/sqrt(head_dim)`. [SPEC-090]
#[inline]
#[must_use]
fn scale() -> f32 {
    1.0 / (HEAD_DIM as f32).sqrt()
}

/// Per-layer ring KV cache.
///
/// Two contiguous, head-major regions per K and per V:
///  * the **reference** block — `[NUM_HEADS][ref_capacity * HEAD_DIM]`, holding
///    the permanent prefill prefix; only the first `prefill_len` rows are live.
///  * the **ring** — `[NUM_HEADS][RING_WINDOW * HEAD_DIM]`, the sliding tail.
///
/// Both are pre-allocated for the worst case at [`RingCache::new`] so the decode
/// hot loop never reallocates. A K/V row for head `h` at row `r` lives at
/// `region[h][r * HEAD_DIM .. (r + 1) * HEAD_DIM]`.
#[derive(Debug, Clone)]
pub struct RingCache {
    /// Worst-case reference rows this cache can hold (e.g. `32768 - 128`).
    ref_capacity: usize,
    /// Reference K, one `Vec` per head (each `ref_capacity * HEAD_DIM`).
    ref_k: Vec<Vec<f32>>,
    /// Reference V, one `Vec` per head.
    ref_v: Vec<Vec<f32>>,
    /// Ring K, one `Vec` per head (each `RING_WINDOW * HEAD_DIM`).
    ring_k: Vec<Vec<f32>>,
    /// Ring V, one `Vec` per head.
    ring_v: Vec<Vec<f32>>,
    /// Reference-block length: number of live prefill rows. `None` until prefill
    /// has been recorded.
    prefill_len: Option<usize>,
    /// Number of live ring rows (`0..=RING_WINDOW`). During warm-up this grows;
    /// at steady state it saturates at `RING_WINDOW`.
    ring_len: usize,
    /// Next ring slot to write (`0..RING_WINDOW`). Only advances modulo `W` at
    /// steady state; during warm-up it tracks `ring_len`.
    ring_pos: usize,
}

impl RingCache {
    /// Allocate a ring cache for one layer sized for the worst-case prefill.
    ///
    /// `prefill_capacity` is the maximum reference-block length (worst-case `m`,
    /// e.g. `32768 - 128`); the `RING_WINDOW` ring slots are allocated on top.
    /// Everything is zero-filled up front so the decode loop is allocation-free.
    #[must_use]
    pub fn new(prefill_capacity: usize) -> Self {
        let ref_elems = prefill_capacity * HEAD_DIM;
        let ring_elems = RING_WINDOW * HEAD_DIM;
        Self {
            ref_capacity: prefill_capacity,
            ref_k: (0..NUM_HEADS).map(|_| vec![0.0f32; ref_elems]).collect(),
            ref_v: (0..NUM_HEADS).map(|_| vec![0.0f32; ref_elems]).collect(),
            ring_k: (0..NUM_HEADS).map(|_| vec![0.0f32; ring_elems]).collect(),
            ring_v: (0..NUM_HEADS).map(|_| vec![0.0f32; ring_elems]).collect(),
            prefill_len: None,
            ring_len: 0,
            ring_pos: 0,
        }
    }

    /// Worst-case reference capacity (rows) this cache was sized for.
    #[must_use]
    pub fn ref_capacity(&self) -> usize {
        self.ref_capacity
    }

    /// Reference-block length once prefill is recorded (`None` before).
    #[must_use]
    pub fn prefill_len(&self) -> Option<usize> {
        self.prefill_len
    }

    /// Number of live ring rows (`0..=RING_WINDOW`).
    #[must_use]
    pub fn ring_len(&self) -> usize {
        self.ring_len
    }

    /// Next ring write position (`0..RING_WINDOW`).
    #[must_use]
    pub fn ring_pos(&self) -> usize {
        self.ring_pos
    }

    /// `true` once the ring has filled (`ring_len == RING_WINDOW`) — i.e. decode
    /// has entered the steady-state overwrite regime.
    #[must_use]
    pub fn is_warm(&self) -> bool {
        self.ring_len >= RING_WINDOW
    }

    /// Effective number of keys a decode query attends over:
    /// `prefill_len + ring_len`. [SPEC-094]
    #[must_use]
    pub fn effective_len(&self) -> usize {
        self.prefill_len.unwrap_or(0) + self.ring_len
    }

    /// Record the prefill K/V as the permanent reference block and set
    /// `prefill_len` (regime 1: true prefill, [SPEC-091]).
    ///
    /// `k`/`v` are head-major flat `[NUM_HEADS, seq, HEAD_DIM]` — exactly the
    /// layout fed to [`super::nn::sdpa`] (`num_bh = NUM_HEADS` for batch 1). This
    /// is the *reference* set `m` = the ENTIRE prefill (BOS + visual + prompt),
    /// per OQ-1/OQ-13; it is never evicted.
    ///
    /// # Errors
    /// [`FocrError::Other`] if `seq` exceeds [`RingCache::ref_capacity`] or the
    /// `k`/`v` lengths don't match `NUM_HEADS * seq * HEAD_DIM`.
    pub fn record_prefill(&mut self, k: &[f32], v: &[f32], seq: usize) -> FocrResult<()> {
        if seq > self.ref_capacity {
            return Err(FocrError::Other(anyhow::anyhow!(
                "rswa: prefill seq {seq} exceeds ref_capacity {}",
                self.ref_capacity
            )));
        }
        let expect = NUM_HEADS * seq * HEAD_DIM;
        if k.len() != expect || v.len() != expect {
            return Err(FocrError::Other(anyhow::anyhow!(
                "rswa: prefill k/v len {}/{} != NUM_HEADS*seq*HEAD_DIM {}",
                k.len(),
                v.len(),
                expect
            )));
        }
        let stride = seq * HEAD_DIM;
        for h in 0..NUM_HEADS {
            let src = &k[h * stride..(h + 1) * stride];
            self.ref_k[h][..stride].copy_from_slice(src);
            let src = &v[h * stride..(h + 1) * stride];
            self.ref_v[h][..stride].copy_from_slice(src);
        }
        self.prefill_len = Some(seq);
        self.ring_len = 0;
        self.ring_pos = 0;
        Ok(())
    }

    /// Write one decode token's K/V for every head into the ring, advancing the
    /// ring state (regimes 2 & 3, [SPEC-091/094]).
    ///
    /// * **Warm-up** (`ring_len < W`): append at row `ring_len`, grow `ring_len`,
    ///   keep `ring_pos == ring_len` (so the next steady-state overwrite starts at
    ///   slot 0). No eviction in the first `W` steps.
    /// * **Steady state** (`ring_len == W`): overwrite slot `ring_pos`, then
    ///   `ring_pos = (ring_pos + 1) % W`.
    ///
    /// `k_step`/`v_step` are `[NUM_HEADS, HEAD_DIM]` flat (the one new token's K/V
    /// across heads, already RoPE'd by the caller at the TRUE absolute position —
    /// [SPEC-095]). Returns the physical slot index written (`0..W`).
    ///
    /// # Errors
    /// [`FocrError::Other`] if prefill was never recorded or the slice lengths
    /// are wrong.
    pub fn write_decode_step(&mut self, k_step: &[f32], v_step: &[f32]) -> FocrResult<usize> {
        if self.prefill_len.is_none() {
            return Err(FocrError::Other(anyhow::anyhow!(
                "rswa: write_decode_step before record_prefill"
            )));
        }
        let expect = NUM_HEADS * HEAD_DIM;
        if k_step.len() != expect || v_step.len() != expect {
            return Err(FocrError::Other(anyhow::anyhow!(
                "rswa: decode step k/v len {}/{} != NUM_HEADS*HEAD_DIM {}",
                k_step.len(),
                v_step.len(),
                expect
            )));
        }

        let slot = if self.ring_len < RING_WINDOW {
            // Warm-up: append, no eviction.
            let slot = self.ring_len;
            for h in 0..NUM_HEADS {
                let off = slot * HEAD_DIM;
                let src = &k_step[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                self.ring_k[h][off..off + HEAD_DIM].copy_from_slice(src);
                let src = &v_step[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                self.ring_v[h][off..off + HEAD_DIM].copy_from_slice(src);
            }
            self.ring_len += 1;
            // Keep ring_pos aligned with the fill cursor so steady state begins
            // overwriting the oldest slot (slot 0) once full.
            self.ring_pos = self.ring_len % RING_WINDOW;
            slot
        } else {
            // Steady state: in-place overwrite at ring_pos, then advance mod W.
            let slot = self.ring_pos;
            for h in 0..NUM_HEADS {
                let off = slot * HEAD_DIM;
                let src = &k_step[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                self.ring_k[h][off..off + HEAD_DIM].copy_from_slice(src);
                let src = &v_step[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                self.ring_v[h][off..off + HEAD_DIM].copy_from_slice(src);
            }
            self.ring_pos = (self.ring_pos + 1) % RING_WINDOW;
            slot
        };
        Ok(slot)
    }

    /// Reference-K row `r` for head `h` (`r < prefill_len`).
    #[inline]
    fn ref_k_row(&self, h: usize, r: usize) -> &[f32] {
        let off = r * HEAD_DIM;
        &self.ref_k[h][off..off + HEAD_DIM]
    }

    /// Reference-V row `r` for head `h`.
    #[inline]
    fn ref_v_row(&self, h: usize, r: usize) -> &[f32] {
        let off = r * HEAD_DIM;
        &self.ref_v[h][off..off + HEAD_DIM]
    }

    /// Ring-K row `r` for head `h` (`r < ring_len`).
    #[inline]
    fn ring_k_row(&self, h: usize, r: usize) -> &[f32] {
        let off = r * HEAD_DIM;
        &self.ring_k[h][off..off + HEAD_DIM]
    }

    /// Ring-V row `r` for head `h`.
    #[inline]
    fn ring_v_row(&self, h: usize, r: usize) -> &[f32] {
        let off = r * HEAD_DIM;
        &self.ring_v[h][off..off + HEAD_DIM]
    }
}

/// Dot product of two equal-length `HEAD_DIM` slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..HEAD_DIM {
        acc += a[i] * b[i];
    }
    acc
}

/// One R-SWA **decode** attention step (`q_len == 1`) over the layer's
/// [`RingCache`].
///
/// `q` is the single decode query, head-major flat `[NUM_HEADS, HEAD_DIM]`
/// (already RoPE'd by the caller — [SPEC-095]). The cache must already hold this
/// step's K/V (call [`RingCache::write_decode_step`] first), because the query
/// attends to *itself* as the most-recent ring token, exactly as the reference
/// does (the new K/V is written into the cache before the mask-free softmax —
/// OQ-3).
///
/// Per head the attention is, with **no causal mask** and scale `1/sqrt(128)`:
/// ```text
///   scores = q . (referenceK[0..prefill_len] ++ ringK[0..ring_len]) * scale
///   weights = softmax(scores)               (over the union)
///   out     = sum_j weights[j] * (referenceV ++ ringV)[j]
/// ```
/// The reference block is folded with an **online (streaming) softmax** so the
/// large `m` score row is never materialized: we accumulate a running max `m*`,
/// a running denominator `l`, and a running weighted-V accumulator, rescaling on
/// each new running-max. The ring (`<= 128` keys) is folded into the same
/// accumulators.
///
/// Returns the decode context as a `[1, NUM_HEADS * HEAD_DIM]` [`Mat`] (the
/// concatenated per-head outputs, ready for `o_proj`).
///
/// # Errors
/// [`FocrError::Other`] if prefill was never recorded, `q` has the wrong length,
/// or the effective key set is empty.
pub fn decode_attention(cache: &RingCache, q: &[f32]) -> FocrResult<Mat> {
    let Some(prefill_len) = cache.prefill_len else {
        return Err(FocrError::Other(anyhow::anyhow!(
            "rswa: decode_attention before record_prefill"
        )));
    };
    let expect = NUM_HEADS * HEAD_DIM;
    if q.len() != expect {
        return Err(FocrError::Other(anyhow::anyhow!(
            "rswa: query len {} != NUM_HEADS*HEAD_DIM {}",
            q.len(),
            expect
        )));
    }
    let ring_len = cache.ring_len;
    if prefill_len + ring_len == 0 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "rswa: empty attention key set (prefill_len=0, ring_len=0)"
        )));
    }

    let s = scale();
    let mut out = vec![0.0f32; NUM_HEADS * HEAD_DIM];

    for h in 0..NUM_HEADS {
        let qh = &q[h * HEAD_DIM..(h + 1) * HEAD_DIM];

        // Online softmax accumulators over the union (reference ++ ring).
        let mut run_max = f32::NEG_INFINITY;
        let mut run_den = 0.0f32;
        let mut acc = [0.0f32; HEAD_DIM];

        // --- streaming fold of the (large) reference block ---
        for r in 0..prefill_len {
            let score = dot(qh, cache.ref_k_row(h, r)) * s;
            fold(
                &mut run_max,
                &mut run_den,
                &mut acc,
                score,
                cache.ref_v_row(h, r),
            );
        }
        // --- fold of the ring tail (<= W keys) ---
        for r in 0..ring_len {
            let score = dot(qh, cache.ring_k_row(h, r)) * s;
            fold(
                &mut run_max,
                &mut run_den,
                &mut acc,
                score,
                cache.ring_v_row(h, r),
            );
        }

        // Normalize: out_h = acc / run_den.
        let inv = if run_den > 0.0 { 1.0 / run_den } else { 0.0 };
        let dst = &mut out[h * HEAD_DIM..(h + 1) * HEAD_DIM];
        for i in 0..HEAD_DIM {
            dst[i] = acc[i] * inv;
        }
    }

    Ok(Mat::from_vec(1, NUM_HEADS * HEAD_DIM, out))
}

/// One step of the streaming (online) softmax recurrence: fold key `(score, v)`
/// into the running max / denominator / weighted-V accumulators.
#[inline]
fn fold(run_max: &mut f32, run_den: &mut f32, acc: &mut [f32; HEAD_DIM], score: f32, v: &[f32]) {
    if score > *run_max {
        // New running max: rescale the existing accumulators down.
        let correction = if run_max.is_finite() {
            (*run_max - score).exp()
        } else {
            0.0
        };
        *run_den *= correction;
        for a in acc.iter_mut() {
            *a *= correction;
        }
        *run_max = score;
    }
    let w = (score - *run_max).exp();
    *run_den += w;
    for i in 0..HEAD_DIM {
        acc[i] += w * v[i];
    }
}

/// Full convenience step: write this token's K/V into the ring, then run the
/// mask-free decode attention over `reference ++ ring`.
///
/// `q/k_step/v_step` are each `[NUM_HEADS, HEAD_DIM]` head-major flat (the one
/// new token, RoPE applied at its TRUE absolute position by the caller —
/// [SPEC-095]). `position_ids` is the single logical position of this decode
/// token; it is *not* used to re-derive RoPE phase (the physical ring slot is
/// decoupled from it), only carried for provenance / validation.
///
/// # Errors
/// Propagates [`RingCache::write_decode_step`] / [`decode_attention`] errors.
pub fn attention(
    cache: &mut RingCache,
    q: &Mat,
    k: &Mat,
    v: &Mat,
    position_ids: &[usize],
) -> FocrResult<Mat> {
    let expect = NUM_HEADS * HEAD_DIM;
    if q.rows != 1 || q.cols != expect {
        return Err(FocrError::Other(anyhow::anyhow!(
            "rswa: attention expects q [1, {expect}] (decode q_len=1), got [{},{}]",
            q.rows,
            q.cols
        )));
    }
    if k.data.len() != expect || v.data.len() != expect {
        return Err(FocrError::Other(anyhow::anyhow!(
            "rswa: attention expects k/v [1, {expect}], got k {} v {}",
            k.data.len(),
            v.data.len()
        )));
    }
    if position_ids.len() != 1 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "rswa: attention expects a single decode position_id, got {}",
            position_ids.len()
        )));
    }
    cache.write_decode_step(&k.data, &v.data)?;
    decode_attention(cache, &q.data)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reference V row whose entries are all `val`, for head `h`, row `r`.
    fn fill_head_major(seq: usize, f: impl Fn(usize, usize) -> f32) -> Vec<f32> {
        // [NUM_HEADS, seq, HEAD_DIM]
        let mut out = vec![0.0f32; NUM_HEADS * seq * HEAD_DIM];
        for h in 0..NUM_HEADS {
            for r in 0..seq {
                for d in 0..HEAD_DIM {
                    out[(h * seq + r) * HEAD_DIM + d] = f(r, d);
                }
            }
        }
        out
    }

    fn one_token(f: impl Fn(usize, usize) -> f32) -> Vec<f32> {
        // [NUM_HEADS, HEAD_DIM]
        let mut out = vec![0.0f32; NUM_HEADS * HEAD_DIM];
        for h in 0..NUM_HEADS {
            for d in 0..HEAD_DIM {
                out[h * HEAD_DIM + d] = f(h, d);
            }
        }
        out
    }

    #[test]
    fn constants_match_spec() {
        assert_eq!(NUM_HEADS, 10);
        assert_eq!(HEAD_DIM, 128);
        assert_eq!(RING_WINDOW, 128);
        assert!((scale() - 1.0 / (128.0f32).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn new_allocates_worst_case() {
        let cache = RingCache::new(32768 - 128);
        assert_eq!(cache.ref_capacity(), 32768 - 128);
        assert_eq!(cache.prefill_len(), None);
        assert_eq!(cache.ring_len(), 0);
        assert_eq!(cache.ring_pos(), 0);
        assert!(!cache.is_warm());
        // Each head's ring is exactly W * HEAD_DIM.
        assert_eq!(cache.ring_k.len(), NUM_HEADS);
        assert_eq!(cache.ring_k[0].len(), RING_WINDOW * HEAD_DIM);
    }

    #[test]
    fn record_prefill_sets_boundary() {
        let mut cache = RingCache::new(64);
        let k = fill_head_major(8, |r, _| r as f32);
        let v = fill_head_major(8, |r, _| (r * 2) as f32);
        cache.record_prefill(&k, &v, 8).unwrap();
        assert_eq!(cache.prefill_len(), Some(8));
        assert_eq!(cache.effective_len(), 8);
        // Row 3 of head 0 round-trips.
        assert_eq!(cache.ref_k_row(0, 3)[0], 3.0);
        assert_eq!(cache.ref_v_row(0, 3)[0], 6.0);
    }

    #[test]
    fn record_prefill_rejects_overflow() {
        let mut cache = RingCache::new(4);
        let k = fill_head_major(8, |_, _| 1.0);
        let v = fill_head_major(8, |_, _| 1.0);
        assert!(cache.record_prefill(&k, &v, 8).is_err());
    }

    /// Single-key attention (prefill_len=1, no ring): softmax over one key is
    /// weight 1, so the output is exactly that key's V row.
    #[test]
    fn decode_single_key_returns_value() {
        let mut cache = RingCache::new(8);
        let k = fill_head_major(1, |_, d| if d == 0 { 1.0 } else { 0.0 });
        let v = fill_head_major(1, |_, _| 7.0);
        cache.record_prefill(&k, &v, 1).unwrap();
        let q = one_token(|_, d| if d == 0 { 5.0 } else { 0.0 });
        let out = decode_attention(&cache, &q).unwrap();
        assert_eq!(out.shape(), (1, NUM_HEADS * HEAD_DIM));
        // Every output element == the single V (7.0).
        for &x in &out.data {
            assert!((x - 7.0).abs() < 1e-5);
        }
    }

    /// Two reference keys with EQUAL scores -> uniform 0.5/0.5 weights -> output
    /// is the average of the two V rows. Validates the online softmax denominator
    /// and rescale path across two folds.
    #[test]
    fn decode_equal_scores_averages_values() {
        let mut cache = RingCache::new(8);
        // Both keys identical -> identical scores for any q.
        let k = fill_head_major(2, |_, d| if d == 0 { 1.0 } else { 0.0 });
        // V row 0 all 2.0, V row 1 all 4.0 -> average 3.0.
        let v = fill_head_major(2, |r, _| if r == 0 { 2.0 } else { 4.0 });
        cache.record_prefill(&k, &v, 2).unwrap();
        let q = one_token(|_, d| if d == 0 { 3.0 } else { 0.0 });
        let out = decode_attention(&cache, &q).unwrap();
        for &x in &out.data {
            assert!((x - 3.0).abs() < 1e-5, "got {x}");
        }
    }

    /// Online softmax must match a naive (materialized) softmax over the union.
    #[test]
    fn online_matches_naive_softmax() {
        let mut cache = RingCache::new(16);
        let m = 5usize;
        // Distinct keys/values so weights are non-uniform.
        let k = fill_head_major(m, |r, d| {
            ((r + 1) as f32) * (if d == 0 { 1.0 } else { 0.0 })
        });
        let v = fill_head_major(m, |r, d| (r as f32) + (d as f32) * 0.01);
        cache.record_prefill(&k, &v, m).unwrap();
        let q = one_token(|_, d| if d == 0 { 0.5 } else { 0.0 });
        let out = decode_attention(&cache, &q).unwrap();

        // Naive reference for head 0.
        let s = scale();
        let mut scores = vec![0.0f32; m];
        for (r, sc) in scores.iter_mut().enumerate() {
            let mut d0 = 0.0f32;
            for d in 0..HEAD_DIM {
                d0 += q[d] * cache.ref_k_row(0, r)[d];
            }
            *sc = d0 * s;
        }
        let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&x| (x - mx).exp()).collect();
        let den: f32 = exps.iter().sum();
        // Reconstruct expected output dim 0 for head 0.
        let mut expect0 = 0.0f32;
        for r in 0..m {
            expect0 += (exps[r] / den) * cache.ref_v_row(0, r)[0];
        }
        assert!(
            (out.data[0] - expect0).abs() < 1e-4,
            "online {} naive {}",
            out.data[0],
            expect0
        );
    }

    /// Warm-up: the first W decode steps append without eviction; ring_len grows,
    /// ring_pos tracks it, the cache stays un-warm until exactly W.
    #[test]
    fn warmup_appends_without_eviction() {
        let mut cache = RingCache::new(8);
        let k = fill_head_major(4, |_, _| 1.0);
        let v = fill_head_major(4, |_, _| 1.0);
        cache.record_prefill(&k, &v, 4).unwrap();

        for step in 0..RING_WINDOW {
            let kt = one_token(|_, _| step as f32);
            let vt = one_token(|_, _| step as f32);
            let slot = cache.write_decode_step(&kt, &vt).unwrap();
            assert_eq!(slot, step, "warm-up writes are append-in-order");
            assert_eq!(cache.ring_len(), step + 1);
            assert!(cache.effective_len() == 4 + step + 1);
        }
        assert!(cache.is_warm());
        assert_eq!(cache.ring_len(), RING_WINDOW);
        // ring_pos wrapped back to 0 once full, ready to overwrite the oldest.
        assert_eq!(cache.ring_pos(), 0);
    }

    /// Steady state: after warm-up the ring overwrites slot `ring_pos` in place
    /// and cycles modulo W; ring_len stays saturated.
    #[test]
    fn steady_state_overwrites_modulo_w() {
        let mut cache = RingCache::new(4);
        let k = fill_head_major(2, |_, _| 0.0);
        let v = fill_head_major(2, |_, _| 0.0);
        cache.record_prefill(&k, &v, 2).unwrap();

        // Fill the ring (warm-up).
        for _ in 0..RING_WINDOW {
            let t = one_token(|_, _| 0.0);
            cache.write_decode_step(&t, &t).unwrap();
        }
        assert!(cache.is_warm());

        // Two more steady-state writes overwrite slots 0 then 1.
        let kt = one_token(|_, _| 99.0);
        let slot0 = cache.write_decode_step(&kt, &kt).unwrap();
        assert_eq!(slot0, 0);
        assert_eq!(cache.ring_pos(), 1);
        let slot1 = cache.write_decode_step(&kt, &kt).unwrap();
        assert_eq!(slot1, 1);
        assert_eq!(cache.ring_pos(), 2);
        // ring_len never exceeds W.
        assert_eq!(cache.ring_len(), RING_WINDOW);
        // The overwritten slot 0 now holds 99.0.
        assert_eq!(cache.ring_k_row(0, 0)[0], 99.0);
    }

    /// ring_pos wraps from W-1 back to 0 at steady state.
    #[test]
    fn ring_pos_wraps_modulo_w() {
        let mut cache = RingCache::new(2);
        let k = fill_head_major(1, |_, _| 0.0);
        cache.record_prefill(&k, &k, 1).unwrap();
        for _ in 0..RING_WINDOW {
            let t = one_token(|_, _| 0.0);
            cache.write_decode_step(&t, &t).unwrap();
        }
        // Do W more steady-state writes: ring_pos cycles 0..W and back to 0.
        for expected_slot in 0..RING_WINDOW {
            let t = one_token(|_, _| 0.0);
            let slot = cache.write_decode_step(&t, &t).unwrap();
            assert_eq!(slot, expected_slot);
        }
        assert_eq!(cache.ring_pos(), 0);
    }

    /// `attention` (the Mat convenience entry) writes the step then attends; with
    /// only the new ring token present (prefill_len=0 is rejected, so use a
    /// 1-token reference) it routes through both paths and yields the right shape.
    #[test]
    fn attention_entry_writes_then_attends() {
        let mut cache = RingCache::new(8);
        // 1-token reference whose K is orthogonal to q so the new ring token,
        // matching q, dominates.
        let k = fill_head_major(1, |_, d| if d == 1 { 1.0 } else { 0.0 });
        let v = fill_head_major(1, |_, _| 1.0);
        cache.record_prefill(&k, &v, 1).unwrap();

        let q = Mat::from_vec(
            1,
            NUM_HEADS * HEAD_DIM,
            one_token(|_, d| if d == 0 { 10.0 } else { 0.0 }),
        );
        let kt = Mat::from_vec(
            1,
            NUM_HEADS * HEAD_DIM,
            one_token(|_, d| if d == 0 { 1.0 } else { 0.0 }),
        );
        let vt = Mat::from_vec(1, NUM_HEADS * HEAD_DIM, one_token(|_, _| 5.0));
        let out = attention(&mut cache, &q, &kt, &vt, &[42]).unwrap();
        assert_eq!(out.shape(), (1, NUM_HEADS * HEAD_DIM));
        assert_eq!(cache.ring_len(), 1);
        // q aligns with the ring token (score 10/sqrt(128) ~= 0.8839) and is
        // orthogonal to the reference (score 0). softmax([0.8839, 0]) puts
        // ~0.7077 on the ring V (5.0) and ~0.2923 on the reference V (1.0), so
        // the output is ~0.7077*5 + 0.2923*1 = 3.831; the ring V dominates the
        // reference V even though the softmax temperature is mild.
        let ring_w = (10.0f32 / (HEAD_DIM as f32).sqrt()).exp();
        let expect = (ring_w * 5.0 + 1.0) / (ring_w + 1.0);
        for &x in &out.data {
            assert!((x - expect).abs() < 1e-4, "got {x}, expected {expect}");
            assert!(x > 3.0, "ring token should dominate the reference, got {x}");
        }
    }

    #[test]
    fn decode_before_prefill_errors() {
        let cache = RingCache::new(4);
        let q = one_token(|_, _| 1.0);
        assert!(decode_attention(&cache, &q).is_err());
    }

    #[test]
    fn write_step_before_prefill_errors() {
        let mut cache = RingCache::new(4);
        let t = one_token(|_, _| 1.0);
        assert!(cache.write_decode_step(&t, &t).is_err());
    }

    #[test]
    fn attention_rejects_multi_row_query() {
        let mut cache = RingCache::new(4);
        let k = fill_head_major(1, |_, _| 0.0);
        cache.record_prefill(&k, &k, 1).unwrap();
        let q = Mat::zeros(2, NUM_HEADS * HEAD_DIM);
        let kt = Mat::zeros(1, NUM_HEADS * HEAD_DIM);
        let vt = Mat::zeros(1, NUM_HEADS * HEAD_DIM);
        assert!(attention(&mut cache, &q, &kt, &vt, &[0]).is_err());
    }
}
