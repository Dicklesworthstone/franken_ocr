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

// Numerical kernel file: hot attention loops index parallel stride-arrays by range
// (`v[r*HEAD_DIM..]`, `scores[r]`, `scale[r]`), where `clippy::needless_range_loop`
// is a false positive — the index is genuinely needed across several arrays.
#![allow(clippy::needless_range_loop)]

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

// ── decode-attention kill-switches (AGENTS.md doctrine: default path is the
//    bit-exact scalar online-softmax loop; both levers are opt-in via env). ───

/// Env kill-switch: reformulate the decode-attention `QK^T` / `probs@V` as a
/// batched GEMV over the contiguous reference/ring key/value blocks (one pass
/// over all keys per head, with the `exp` lifted out of the dot loop) instead of
/// the per-key interleaved online-softmax fold. f32 accumulation reorders vs the
/// scalar path, so it is **not** bit-exact — gated, parity-checked at `2e-6`.
const ATTN_GEMM_ENV: &str = "FOCR_ATTN_GEMM";

/// Env kill-switch (ACCURACY-RISKY, needs a measured-CER gate): additionally
/// store the reference/ring K/V as per-row symmetric int8 and run the `QK` dot
/// in int8 (`simd::igemm_s8s8` / SDOT), dequantizing `V` per row in the `PV`
/// pass. Implies (and overrides) the f32 GEMM path. Default keeps f32.
const INT8_KV_ENV: &str = "FOCR_INT8_KV";

/// Read `FOCR_ATTN_GEMM` ONCE into a process-global bool (doctrine: not per
/// token). The int8-KV path subsumes the GEMM path, so it is dispatched ahead of
/// this flag in [`decode_attention`].
fn attn_gemm_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os(ATTN_GEMM_ENV).is_some())
}

/// Read `FOCR_INT8_KV` ONCE into a process-global bool. Also gates whether
/// [`RingCache::new`] allocates the int8 K/V mirror buffers (so the default path
/// pays zero extra memory).
fn int8_kv_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os(INT8_KV_ENV).is_some())
}

fn checked_cache_region_elems(rows: usize) -> usize {
    rows.checked_mul(HEAD_DIM)
        .expect("rswa: cache rows*HEAD_DIM overflow")
}

fn checked_head_major_layout(seq: usize, label: &str) -> FocrResult<(usize, usize)> {
    let stride = seq.checked_mul(HEAD_DIM).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "rswa: {label} seq*HEAD_DIM overflow (seq={seq}, HEAD_DIM={HEAD_DIM})"
        ))
    })?;
    let expect = NUM_HEADS.checked_mul(stride).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "rswa: {label} NUM_HEADS*seq*HEAD_DIM overflow \
             (NUM_HEADS={NUM_HEADS}, seq={seq}, HEAD_DIM={HEAD_DIM})"
        ))
    })?;
    Ok((stride, expect))
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
    /// Per-row symmetric int8 mirror of the K/V regions — `Some` iff
    /// [`FOCR_INT8_KV`](INT8_KV_ENV) was set when this cache was built (so the
    /// default path allocates nothing). Populated in lock-step with the f32 K/V
    /// by [`RingCache::record_prefill`] / [`RingCache::write_decode_step`].
    int8: Option<Int8Kv>,
}

/// Per-row symmetric int8 mirror of one layer's K/V regions (built only under
/// [`FOCR_INT8_KV`](INT8_KV_ENV)). Each `_i8` buffer mirrors the corresponding
/// f32 region byte-for-byte in shape (`rows * HEAD_DIM` per head); each `_scale`
/// holds one f32 `max(|row|)/127` per row. The f32 regions remain the source of
/// truth and the parity oracle; these are a bandwidth-reduced read path for the
/// hot `QK`/`PV` dots.
#[derive(Debug, Clone)]
struct Int8Kv {
    /// Reference K int8, one `Vec` per head (each `ref_capacity * HEAD_DIM`).
    ref_k: Vec<Vec<i8>>,
    /// Reference K per-row scales, one `Vec` per head (each `ref_capacity`).
    ref_k_scale: Vec<Vec<f32>>,
    /// Reference V int8, one `Vec` per head.
    ref_v: Vec<Vec<i8>>,
    /// Reference V per-row scales, one `Vec` per head.
    ref_v_scale: Vec<Vec<f32>>,
    /// Ring K int8, one `Vec` per head (each `RING_WINDOW * HEAD_DIM`).
    ring_k: Vec<Vec<i8>>,
    /// Ring K per-row scales, one `Vec` per head (each `RING_WINDOW`).
    ring_k_scale: Vec<Vec<f32>>,
    /// Ring V int8, one `Vec` per head.
    ring_v: Vec<Vec<i8>>,
    /// Ring V per-row scales, one `Vec` per head.
    ring_v_scale: Vec<Vec<f32>>,
}

impl Int8Kv {
    /// Allocate zeroed int8 mirrors sized exactly like the f32 K/V regions.
    fn new(ref_capacity: usize) -> Self {
        let ref_elems = checked_cache_region_elems(ref_capacity);
        let ring_elems = checked_cache_region_elems(RING_WINDOW);
        let i8_ref = || (0..NUM_HEADS).map(|_| vec![0i8; ref_elems]).collect();
        let sc_ref = || (0..NUM_HEADS).map(|_| vec![0.0f32; ref_capacity]).collect();
        let i8_ring = || (0..NUM_HEADS).map(|_| vec![0i8; ring_elems]).collect();
        let sc_ring = || (0..NUM_HEADS).map(|_| vec![0.0f32; RING_WINDOW]).collect();
        Self {
            ref_k: i8_ref(),
            ref_k_scale: sc_ref(),
            ref_v: i8_ref(),
            ref_v_scale: sc_ref(),
            ring_k: i8_ring(),
            ring_k_scale: sc_ring(),
            ring_v: i8_ring(),
            ring_v_scale: sc_ring(),
        }
    }
}

/// Per-row symmetric int8 quantization of a `HEAD_DIM`-length f32 row into `q`,
/// returning the row scale `max(|row|)/127` (or `1.0` for an all-zero row).
///
/// Values are rounded then clamped to `[-127, 127]`, so the int8 `QK` dot
/// accumulates at most `127 * 127 * HEAD_DIM = 2_064_512` in i32 — three orders
/// of magnitude under `i32::MAX`, regardless of head_dim/key count (the
/// contraction is fixed at `HEAD_DIM`). See `int8_qk_i32_accumulation_cannot_overflow`.
fn quantize_row_i8(row: &[f32], q: &mut [i8]) -> f32 {
    let maxabs = row.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let scale = if maxabs > 0.0 { maxabs / 127.0 } else { 1.0 };
    let inv = 1.0 / scale;
    for (qd, &x) in q.iter_mut().zip(row.iter()) {
        *qd = (x * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

impl RingCache {
    /// Allocate a ring cache for one layer sized for the worst-case prefill.
    ///
    /// `prefill_capacity` is the maximum reference-block length (worst-case `m`,
    /// e.g. `32768 - 128`); the `RING_WINDOW` ring slots are allocated on top.
    /// Everything is zero-filled up front so the decode loop is allocation-free.
    #[must_use]
    pub fn new(prefill_capacity: usize) -> Self {
        Self::new_inner(prefill_capacity, int8_kv_enabled())
    }

    /// [`RingCache::new`] with the int8-mirror decision made explicit (so the
    /// unit tests can build an int8-enabled cache without setting the process
    /// env). `with_int8` allocates the [`Int8Kv`] side buffers.
    #[must_use]
    fn new_inner(prefill_capacity: usize, with_int8: bool) -> Self {
        let ref_elems = checked_cache_region_elems(prefill_capacity);
        let ring_elems = checked_cache_region_elems(RING_WINDOW);
        Self {
            ref_capacity: prefill_capacity,
            ref_k: (0..NUM_HEADS).map(|_| vec![0.0f32; ref_elems]).collect(),
            ref_v: (0..NUM_HEADS).map(|_| vec![0.0f32; ref_elems]).collect(),
            ring_k: (0..NUM_HEADS).map(|_| vec![0.0f32; ring_elems]).collect(),
            ring_v: (0..NUM_HEADS).map(|_| vec![0.0f32; ring_elems]).collect(),
            prefill_len: None,
            ring_len: 0,
            ring_pos: 0,
            int8: with_int8.then(|| Int8Kv::new(prefill_capacity)),
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
        let (stride, expect) = checked_head_major_layout(seq, "prefill")?;
        if k.len() != expect || v.len() != expect {
            return Err(FocrError::Other(anyhow::anyhow!(
                "rswa: prefill k/v len {}/{} != NUM_HEADS*seq*HEAD_DIM {}",
                k.len(),
                v.len(),
                expect
            )));
        }
        for h in 0..NUM_HEADS {
            let src = &k[h * stride..(h + 1) * stride];
            self.ref_k[h][..stride].copy_from_slice(src);
            let src = &v[h * stride..(h + 1) * stride];
            self.ref_v[h][..stride].copy_from_slice(src);
        }
        // Mirror the whole reference block into int8 once (amortized over the
        // entire decode) when the int8-KV lever is active.
        if let Some(i8kv) = self.int8.as_mut() {
            for h in 0..NUM_HEADS {
                for r in 0..seq {
                    let off = r * HEAD_DIM;
                    i8kv.ref_k_scale[h][r] = quantize_row_i8(
                        &self.ref_k[h][off..off + HEAD_DIM],
                        &mut i8kv.ref_k[h][off..off + HEAD_DIM],
                    );
                    i8kv.ref_v_scale[h][r] = quantize_row_i8(
                        &self.ref_v[h][off..off + HEAD_DIM],
                        &mut i8kv.ref_v[h][off..off + HEAD_DIM],
                    );
                }
            }
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
        // Mirror the freshly-written ring row into int8 at the same physical slot.
        if let Some(i8kv) = self.int8.as_mut() {
            let off = slot * HEAD_DIM;
            for h in 0..NUM_HEADS {
                i8kv.ring_k_scale[h][slot] = quantize_row_i8(
                    &self.ring_k[h][off..off + HEAD_DIM],
                    &mut i8kv.ring_k[h][off..off + HEAD_DIM],
                );
                i8kv.ring_v_scale[h][slot] = quantize_row_i8(
                    &self.ring_v[h][off..off + HEAD_DIM],
                    &mut i8kv.ring_v[h][off..off + HEAD_DIM],
                );
            }
        }
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
///
/// Dispatch (env read ONCE into a bool — doctrine):
///  * default — [`decode_attention_scalar`], the bit-exact online-softmax fold;
///  * [`FOCR_INT8_KV`](INT8_KV_ENV) — [`decode_attention_int8`] (int8 `QK` SDOT +
///    int8 `V` dequant), overriding the f32 GEMM path;
///  * [`FOCR_ATTN_GEMM`](ATTN_GEMM_ENV) — [`decode_attention_gemm`], the f32
///    batched-GEMV path (`exp` lifted out of the dot loop).
///
/// Returns the decode context as a `[1, NUM_HEADS * HEAD_DIM]` [`Mat`] (the
/// concatenated per-head outputs, ready for `o_proj`).
///
/// # Errors
/// [`FocrError::Other`] if prefill was never recorded, `q` has the wrong length,
/// or the effective key set is empty.
pub fn decode_attention(cache: &RingCache, q: &[f32]) -> FocrResult<Mat> {
    if int8_kv_enabled() && cache.int8.is_some() {
        decode_attention_int8(cache, q)
    } else if attn_gemm_enabled() {
        decode_attention_gemm(cache, q)
    } else {
        decode_attention_scalar(cache, q)
    }
}

/// Shared validation prologue for the three decode-attention paths: returns
/// `(prefill_len, ring_len)` or the same errors the original kernel raised.
fn decode_dims(cache: &RingCache, q: &[f32]) -> FocrResult<(usize, usize)> {
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
    Ok((prefill_len, ring_len))
}

/// Default (bit-exact) decode attention: per-head online (streaming) softmax
/// fold over the union `reference ++ ring`, never materializing the score row.
/// This is the parity oracle for the GEMM / int8-KV levers.
fn decode_attention_scalar(cache: &RingCache, q: &[f32]) -> FocrResult<Mat> {
    let (prefill_len, ring_len) = decode_dims(cache, q)?;
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

/// `scores[r] = scale * (q · keys[r])` for `r in 0..rows`, where `keys` is the
/// contiguous `[rows, HEAD_DIM]` row-major key block. The inner dot runs in the
/// SAME order (`d = 0..HEAD_DIM`) as [`dot`], so each per-key score is
/// bit-identical to the scalar path; only the *softmax* and *PV* accumulation
/// reorder. A clean, branch-free, autovectorizable loop (doctrine #3 — no
/// hand-rolled wide SIMD over an autovectorizable scalar loop).
#[inline]
fn block_scores(q: &[f32], keys: &[f32], rows: usize, scale: f32, out: &mut [f32]) {
    for r in 0..rows {
        let krow = &keys[r * HEAD_DIM..(r + 1) * HEAD_DIM];
        let mut acc = 0.0f32;
        for d in 0..HEAD_DIM {
            acc += q[d] * krow[d];
        }
        out[r] = acc * scale;
    }
}

/// `acc[d] += sum_r probs[r] * vals[r*HEAD_DIM + d]` over the contiguous
/// `[rows, HEAD_DIM]` value block — the `probs @ V` GEMV, broadcast-scalar inner.
#[inline]
fn block_accumulate(probs: &[f32], vals: &[f32], rows: usize, acc: &mut [f32; HEAD_DIM]) {
    for r in 0..rows {
        let vrow = &vals[r * HEAD_DIM..(r + 1) * HEAD_DIM];
        let p = probs[r];
        for d in 0..HEAD_DIM {
            acc[d] += p * vrow[d];
        }
    }
}

/// Numerically-stable softmax over the first `total` entries of `scores`,
/// IN PLACE: subtract the row max, exponentiate, and return the denominator
/// (the un-normalized weights stay in `scores`, mirroring how the scalar path
/// defers the `1/run_den` normalization to the output).
#[inline]
fn softmax_inplace(scores: &mut [f32]) -> f32 {
    let mut mx = f32::NEG_INFINITY;
    for &sc in scores.iter() {
        if sc > mx {
            mx = sc;
        }
    }
    let mut den = 0.0f32;
    for sc in scores.iter_mut() {
        let w = (*sc - mx).exp();
        *sc = w;
        den += w;
    }
    den
}

/// f32 batched-GEMV decode attention (`FOCR_ATTN_GEMM`): per head, compute ALL
/// scores over the contiguous reference + ring key blocks in one branch-free
/// pass ([`block_scores`]), softmax the materialized row, then `probs @ V` in one
/// pass over the value blocks ([`block_accumulate`]). Same R-SWA semantics as
/// [`decode_attention_scalar`]; the per-key dots are bit-identical, only the
/// softmax/PV accumulation reorders (f32, so NOT bit-exact — parity-checked at
/// `2e-6`).
fn decode_attention_gemm(cache: &RingCache, q: &[f32]) -> FocrResult<Mat> {
    let (prefill_len, ring_len) = decode_dims(cache, q)?;
    let total = prefill_len + ring_len;
    let s = scale();
    let mut out = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    let mut scores = vec![0.0f32; total];

    for h in 0..NUM_HEADS {
        let qh = &q[h * HEAD_DIM..(h + 1) * HEAD_DIM];

        // QK^T over the two contiguous key blocks → materialized score row.
        block_scores(
            qh,
            &cache.ref_k[h][..prefill_len * HEAD_DIM],
            prefill_len,
            s,
            &mut scores[..prefill_len],
        );
        block_scores(
            qh,
            &cache.ring_k[h][..ring_len * HEAD_DIM],
            ring_len,
            s,
            &mut scores[prefill_len..total],
        );

        let den = softmax_inplace(&mut scores[..total]);

        // probs @ V over the two contiguous value blocks.
        let mut acc = [0.0f32; HEAD_DIM];
        block_accumulate(
            &scores[..prefill_len],
            &cache.ref_v[h][..prefill_len * HEAD_DIM],
            prefill_len,
            &mut acc,
        );
        block_accumulate(
            &scores[prefill_len..total],
            &cache.ring_v[h][..ring_len * HEAD_DIM],
            ring_len,
            &mut acc,
        );

        let inv = if den > 0.0 { 1.0 / den } else { 0.0 };
        let dst = &mut out[h * HEAD_DIM..(h + 1) * HEAD_DIM];
        for i in 0..HEAD_DIM {
            dst[i] = acc[i] * inv;
        }
    }

    Ok(Mat::from_vec(1, NUM_HEADS * HEAD_DIM, out))
}

/// int8-KV decode attention (`FOCR_INT8_KV`, ACCURACY-RISKY — needs a measured
/// CER gate). The query is dynamically quantized per head; the `QK` dot runs in
/// int8 (`simd::igemm_s8s8` / SDOT) over the int8 K mirror with an i32
/// accumulator (worst case `127*127*HEAD_DIM = 2_064_512`, far under `i32::MAX`),
/// dequantized by `qscale * k_scale[r]`. Softmax is identical f32. `PV` reads the
/// int8 V mirror (4× less bandwidth) and dequantizes per row. R-SWA semantics
/// unchanged; f32 stays the default + the parity oracle.
fn decode_attention_int8(cache: &RingCache, q: &[f32]) -> FocrResult<Mat> {
    let (prefill_len, ring_len) = decode_dims(cache, q)?;
    let Some(i8kv) = cache.int8.as_ref() else {
        return Err(FocrError::Other(anyhow::anyhow!(
            "rswa: decode_attention_int8 without an int8 KV mirror (FOCR_INT8_KV)"
        )));
    };
    let total = prefill_len + ring_len;
    let s = scale();
    let mut out = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    let mut scores = vec![0.0f32; total];
    let mut qi8 = [0i8; HEAD_DIM];
    let mut acc_i32 = vec![0i32; total];

    for h in 0..NUM_HEADS {
        let qh = &q[h * HEAD_DIM..(h + 1) * HEAD_DIM];
        let qscale = quantize_row_i8(qh, &mut qi8);

        // int8 QK over the reference K mirror.
        if prefill_len > 0 {
            let dst = &mut acc_i32[..prefill_len];
            dst.fill(0);
            crate::simd::igemm_s8s8(
                &qi8,
                &i8kv.ref_k[h][..prefill_len * HEAD_DIM],
                1,
                HEAD_DIM,
                prefill_len,
                dst,
            );
            let k_scale = &i8kv.ref_k_scale[h];
            for r in 0..prefill_len {
                scores[r] = acc_i32[r] as f32 * qscale * k_scale[r] * s;
            }
        }
        // int8 QK over the ring K mirror.
        if ring_len > 0 {
            let dst = &mut acc_i32[..ring_len];
            dst.fill(0);
            crate::simd::igemm_s8s8(
                &qi8,
                &i8kv.ring_k[h][..ring_len * HEAD_DIM],
                1,
                HEAD_DIM,
                ring_len,
                dst,
            );
            let k_scale = &i8kv.ring_k_scale[h];
            for r in 0..ring_len {
                scores[prefill_len + r] = acc_i32[r] as f32 * qscale * k_scale[r] * s;
            }
        }

        let den = softmax_inplace(&mut scores[..total]);

        // PV: read int8 V (1 B/elem), dequantize per row.
        let mut acc = [0.0f32; HEAD_DIM];
        for r in 0..prefill_len {
            let vrow = &i8kv.ref_v[h][r * HEAD_DIM..(r + 1) * HEAD_DIM];
            let pw = scores[r] * i8kv.ref_v_scale[h][r];
            for d in 0..HEAD_DIM {
                acc[d] += pw * f32::from(vrow[d]);
            }
        }
        for r in 0..ring_len {
            let vrow = &i8kv.ring_v[h][r * HEAD_DIM..(r + 1) * HEAD_DIM];
            let pw = scores[prefill_len + r] * i8kv.ring_v_scale[h][r];
            for d in 0..HEAD_DIM {
                acc[d] += pw * f32::from(vrow[d]);
            }
        }

        let inv = if den > 0.0 { 1.0 / den } else { 0.0 };
        let dst = &mut out[h * HEAD_DIM..(h + 1) * HEAD_DIM];
        for i in 0..HEAD_DIM {
            dst[i] = acc[i] * inv;
        }
    }

    Ok(Mat::from_vec(1, NUM_HEADS * HEAD_DIM, out))
}

fn validate_decode_step_mat(label: &str, mat: &Mat, expect: usize) -> FocrResult<()> {
    if mat.rows != 1 || mat.cols != expect {
        return Err(FocrError::Other(anyhow::anyhow!(
            "rswa: attention expects {label} [1, {expect}], got [{},{}]",
            mat.rows,
            mat.cols
        )));
    }
    if mat.data.len() != expect {
        return Err(FocrError::Other(anyhow::anyhow!(
            "rswa: attention {label} data len {} != NUM_HEADS*HEAD_DIM {}",
            mat.data.len(),
            expect
        )));
    }
    Ok(())
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
    validate_decode_step_mat("q", q, expect)?;
    validate_decode_step_mat("k", k, expect)?;
    validate_decode_step_mat("v", v, expect)?;
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

    fn assert_err_contains<T>(res: FocrResult<T>, needle: &str) {
        let message = match res {
            Ok(_) => String::from("<ok>"),
            Err(err) => err.to_string(),
        };
        assert!(
            message.contains(needle),
            "error {message:?} did not contain {needle:?}"
        );
    }

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
    #[should_panic(expected = "cache rows*HEAD_DIM overflow")]
    fn new_rejects_ref_capacity_shape_overflow_before_allocating() {
        let _ = RingCache::new(usize::MAX / HEAD_DIM + 1);
    }

    #[test]
    fn head_major_layout_rejects_stride_overflow() {
        let err = checked_head_major_layout(usize::MAX / HEAD_DIM + 1, "test")
            .expect_err("seq*HEAD_DIM should overflow");
        assert!(err.to_string().contains("seq*HEAD_DIM overflow"));
    }

    #[test]
    fn head_major_layout_rejects_total_overflow() {
        let seq = (usize::MAX / HEAD_DIM) / NUM_HEADS + 1;
        let err = checked_head_major_layout(seq, "test")
            .expect_err("NUM_HEADS*seq*HEAD_DIM should overflow");
        assert!(err.to_string().contains("NUM_HEADS*seq*HEAD_DIM overflow"));
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
            // `d` indexes both `q` and the borrowed `ref_k_row` slice (two buffers).
            #[allow(clippy::needless_range_loop)]
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
        // `r` indexes `exps` and is also passed to `ref_v_row` (row selector).
        #[allow(clippy::needless_range_loop)]
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

    #[test]
    fn attention_rejects_malformed_query_without_mutating_cache() {
        let mut cache = RingCache::new(4);
        let k = fill_head_major(1, |_, _| 0.0);
        cache.record_prefill(&k, &k, 1).unwrap();

        let q = Mat {
            rows: 1,
            cols: NUM_HEADS * HEAD_DIM,
            data: vec![0.0; NUM_HEADS * HEAD_DIM - 1],
        };
        let kt = Mat::from_vec(1, NUM_HEADS * HEAD_DIM, one_token(|_, _| 1.0));
        let vt = Mat::from_vec(1, NUM_HEADS * HEAD_DIM, one_token(|_, _| 2.0));

        assert_err_contains(
            attention(&mut cache, &q, &kt, &vt, &[0]),
            "attention q data len",
        );
        assert_eq!(cache.ring_len(), 0, "malformed q must not write K/V");
    }

    #[test]
    fn attention_rejects_kv_logical_shape_mismatch_before_mutating_cache() {
        let mut cache = RingCache::new(4);
        let prefill = fill_head_major(1, |_, _| 0.0);
        cache.record_prefill(&prefill, &prefill, 1).unwrap();

        let q = Mat::from_vec(1, NUM_HEADS * HEAD_DIM, one_token(|_, _| 1.0));
        let kt = Mat {
            rows: 2,
            cols: (NUM_HEADS * HEAD_DIM) / 2,
            data: one_token(|_, _| 1.0),
        };
        let vt = Mat::from_vec(1, NUM_HEADS * HEAD_DIM, one_token(|_, _| 2.0));

        assert_err_contains(
            attention(&mut cache, &q, &kt, &vt, &[0]),
            "attention expects k [1",
        );
        assert_eq!(cache.ring_len(), 0, "malformed k must not write K/V");
    }

    // ── decode-attention lever parity (FOCR_ATTN_GEMM / FOCR_INT8_KV) ──────────

    fn max_abs_diff(a: &Mat, b: &Mat) -> f32 {
        assert_eq!(a.shape(), b.shape(), "shape mismatch in max_abs_diff");
        a.data
            .iter()
            .zip(b.data.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// Build a cache (optionally int8-mirrored) with a `pf`-row reference block
    /// and `ring` decode steps, filling K/V via the supplied closures.
    fn build_cache(
        pf: usize,
        ring: usize,
        int8: bool,
        kf: impl Fn(usize, usize) -> f32,
        vf: impl Fn(usize, usize) -> f32,
        rk: impl Fn(usize, usize, usize) -> f32,
        rv: impl Fn(usize, usize, usize) -> f32,
    ) -> RingCache {
        let mut cache = RingCache::new_inner(pf + ring + 8, int8);
        let k = fill_head_major(pf, &kf);
        let v = fill_head_major(pf, &vf);
        cache.record_prefill(&k, &v, pf).unwrap();
        for step in 0..ring {
            let kt = one_token(|h, d| rk(step, h, d));
            let vt = one_token(|h, d| rv(step, h, d));
            cache.write_decode_step(&kt, &vt).unwrap();
        }
        cache
    }

    /// The f32 batched-GEMV path (`FOCR_ATTN_GEMM`) tracks the scalar online-
    /// softmax oracle within the SAM-style `2e-6` reorder tolerance over a
    /// reference block + ring tail with non-uniform, non-degenerate scores.
    #[test]
    fn gemm_attention_matches_scalar_reference() {
        let cache = build_cache(
            7,
            5,
            false,
            |r, d| ((r * 13 + d * 7) % 17) as f32 * 0.11 - 0.9,
            |r, d| ((r * 5 + d * 3) % 11) as f32 * 0.07 - 0.3,
            |s, h, d| ((h * 3 + d * 2 + s) % 13) as f32 * 0.05 - 0.31,
            |s, h, d| ((h + d * 4 + s) % 9) as f32 * 0.06 - 0.2,
        );
        let q = one_token(|h, d| ((h * 2 + d) % 7) as f32 * 0.2 - 0.6);
        let gemm = decode_attention_gemm(&cache, &q).unwrap();
        let scalar = decode_attention_scalar(&cache, &q).unwrap();
        let max_abs = max_abs_diff(&gemm, &scalar);
        assert!(max_abs <= 2.0e-6, "gemm vs scalar max_abs={max_abs}");
    }

    /// The public dispatcher with no env set IS the scalar path (default,
    /// bit-exact). Guards against a future dispatch regression.
    #[test]
    fn default_dispatch_is_bit_exact_scalar() {
        let cache = build_cache(
            4,
            3,
            false,
            |r, d| ((r + d) % 5) as f32 * 0.13 - 0.3,
            |r, d| ((r * 2 + d) % 6) as f32 * 0.09 - 0.2,
            |s, h, d| ((h + d + s) % 7) as f32 * 0.04 - 0.1,
            |s, h, d| ((h * 2 + d + s) % 5) as f32 * 0.05 - 0.1,
        );
        let q = one_token(|h, d| ((h + d) % 9) as f32 * 0.1 - 0.4);
        let public = decode_attention(&cache, &q).unwrap();
        let scalar = decode_attention_scalar(&cache, &q).unwrap();
        // Default (no FOCR_* env): dispatcher routes to the scalar oracle exactly.
        assert_eq!(public.data, scalar.data);
    }

    /// int8 i32-accumulation overflow proof (doctrine #2). The `QK` dot contracts
    /// over `HEAD_DIM` with both operands clamped to `[-127, 127]`, so the i32
    /// accumulator tops out three orders of magnitude under `i32::MAX`.
    #[test]
    fn int8_qk_i32_accumulation_cannot_overflow() {
        let worst = 127i64 * 127 * HEAD_DIM as i64;
        assert_eq!(worst, 2_064_512);
        assert!(
            worst <= i64::from(i32::MAX),
            "worst-case int8 QK accumulation {worst} overflows i32"
        );
    }

    /// The int8 mirror buffers are allocated iff requested (the env gate keeps the
    /// default path at zero extra memory).
    #[test]
    fn int8_mirror_allocated_only_when_enabled() {
        assert!(RingCache::new_inner(8, false).int8.is_none());
        let c = RingCache::new_inner(8, true);
        let i8kv = c.int8.as_ref().expect("int8 mirror present");
        assert_eq!(i8kv.ref_k.len(), NUM_HEADS);
        assert_eq!(i8kv.ring_k[0].len(), RING_WINDOW * HEAD_DIM);
        assert_eq!(i8kv.ref_k_scale[0].len(), 8);
    }

    /// With **lossless-quantizable** K/V/q (integer entries, per-row max-abs
    /// pinned to 127 so `scale == 1`), the int8-KV path reproduces the f32 GEMM
    /// path to f32 precision — isolating the int8 QK-SDOT + V-dequant *kernel*
    /// correctness from the quantization error itself (which the 20-page CER gate
    /// measures). The d=0 anchor (127) pins the scale; the small d>=1 integers
    /// keep the softmax non-degenerate.
    #[test]
    fn int8_kv_attention_matches_gemm_when_losslessly_quantizable() {
        // Per-row: entry 0 = 127 (pins max-abs => scale 1), rest small integers.
        let anchor = |base: usize| {
            move |r: usize, d: usize| -> f32 {
                if d == 0 {
                    127.0
                } else {
                    (((r * base + d * 3) % 7) as i32 - 3) as f32
                }
            }
        };
        let ranchor = |base: usize| {
            move |s: usize, h: usize, d: usize| -> f32 {
                if d == 0 {
                    127.0
                } else {
                    (((h * base + d * 5 + s * 2) % 7) as i32 - 3) as f32
                }
            }
        };
        let cache = build_cache(6, 4, true, anchor(31), anchor(17), ranchor(13), ranchor(11));
        let q = one_token(|_h, d| {
            if d == 0 {
                127.0
            } else {
                ((d % 7) as i32 - 3) as f32
            }
        });

        let int8 = decode_attention_int8(&cache, &q).unwrap();
        let gemm = decode_attention_gemm(&cache, &q).unwrap();
        let scalar = decode_attention_scalar(&cache, &q).unwrap();

        // Lossless quantization => int8 reproduces the GEMM path essentially to
        // the bit (only f32 multiply by the unit scales intervenes).
        let i8_vs_gemm = max_abs_diff(&int8, &gemm);
        assert!(i8_vs_gemm <= 1.0e-6, "int8 vs gemm max_abs={i8_vs_gemm}");
        // And the whole int8 pipeline tracks the scalar oracle within the
        // online-vs-batched softmax reorder tolerance. The lossless construction
        // pins V's max-abs at 127, so the outputs are O(127); use a
        // MAGNITUDE-RELATIVE 2e-6 bound (1 ULP at mag 127 is ~1.5e-5 absolute).
        let i8_vs_scalar = max_abs_diff(&int8, &scalar);
        let out_mag = scalar.data.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let rel = i8_vs_scalar / out_mag.max(1.0);
        assert!(
            rel <= 2.0e-6,
            "int8 vs scalar rel={rel} (abs={i8_vs_scalar}, out_mag={out_mag})"
        );
    }

    /// int8-KV degrades GRACEFULLY (no panic, finite output, right shape) on
    /// arbitrary lossy f32 K/V — the realistic regime the CER gate evaluates.
    #[test]
    fn int8_kv_attention_runs_on_lossy_inputs() {
        let cache = build_cache(
            5,
            3,
            true,
            |r, d| ((r * 7 + d * 3) % 19) as f32 * 0.013 - 0.12,
            |r, d| ((r * 11 + d) % 23) as f32 * 0.009 - 0.1,
            |s, h, d| ((h * 5 + d * 2 + s) % 17) as f32 * 0.011 - 0.09,
            |s, h, d| ((h + d * 3 + s) % 13) as f32 * 0.012 - 0.07,
        );
        let q = one_token(|h, d| ((h * 3 + d) % 11) as f32 * 0.02 - 0.1);
        let int8 = decode_attention_int8(&cache, &q).unwrap();
        assert_eq!(int8.shape(), (1, NUM_HEADS * HEAD_DIM));
        assert!(int8.data.iter().all(|x| x.is_finite()));
    }

    /// int8 dispatch without an int8 mirror surfaces a clear error rather than
    /// panicking (defends the dispatcher precondition).
    #[test]
    fn int8_path_without_mirror_errors() {
        let cache = build_cache(
            3,
            0,
            false, // no int8 mirror
            |_, _| 0.5,
            |_, _| 0.5,
            |_, _, _| 0.0,
            |_, _, _| 0.0,
        );
        let q = one_token(|_, _| 0.5);
        assert_err_contains(
            decode_attention_int8(&cache, &q),
            "without an int8 KV mirror",
        );
    }
}
