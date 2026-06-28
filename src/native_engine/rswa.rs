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
    /// Monotonic count of [`RingCache::write_decode_step`] calls since the last
    /// [`RingCache::record_prefill`] (reset to `0` there). Not part of the
    /// attention math — it exists so [`RingCache::checkpoint`] /
    /// [`RingCache::rollback_to`] (bd-1azu.31) can count the discarded
    /// speculative steps and enforce the lossless roll-back contract.
    decode_writes: usize,
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

/// A lightweight, copy-able snapshot of a [`RingCache`]'s *mutable generated-tail
/// write state* — the cursors needed to roll the ring back to a prior decode
/// point WITHOUT copying any K/V buffer (bd-1azu.31).
///
/// Captured by [`RingCache::checkpoint`] and replayed by
/// [`RingCache::rollback_to`] to discard speculative decode steps (e.g. the
/// rejected suffix of a speculative-decoding draft block): checkpoint before the
/// draft, write the draft tokens, then roll back to erase the rejected ones. Only
/// the generated tail moves — the reference (prefill) block is frozen and never
/// rolls back — so the snapshot is just the prefill boundary (recorded as a
/// provenance guard), the ring `len`/`pos` cursors, and the monotonic
/// decode-write count.
///
/// ## LOSSLESS contract
/// Rollback restores the EXACT byte-for-byte readable ring state — the discarded
/// speculative steps leave no trace — PROVIDED those steps only *appended* fresh
/// ring slots and never wrapped around to overwrite a slot that was already live
/// at the checkpoint. That holds whenever
/// `checkpoint.ring_len + (#discarded steps) <= RING_WINDOW`, which is the
/// speculative-decode regime (a short draft block appended to the generated
/// tail). If a discarded write evicted a checkpoint-live slot (a steady-state
/// in-place overwrite), that slot's prior K/V is physically gone and cannot be
/// reconstructed from cursors alone; [`RingCache::rollback_to`] `debug_assert`s
/// against exactly that case rather than silently returning a corrupted cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingCheckpoint {
    /// Reference-block boundary at checkpoint time (frozen; a provenance guard so
    /// a checkpoint can't be replayed onto a differently-prefilled cache).
    prefill_len: Option<usize>,
    /// Live ring rows at checkpoint (`0..=RING_WINDOW`).
    ring_len: usize,
    /// Next ring write slot at checkpoint (`0..RING_WINDOW`).
    ring_pos: usize,
    /// [`RingCache::write_decode_step`] count at checkpoint time — lets
    /// [`RingCache::rollback_to`] tally the discarded steps and enforce the
    /// lossless contract.
    decode_writes: usize,
}

impl RingCheckpoint {
    /// Reference-block length captured at checkpoint (`None` if not yet prefilled).
    #[must_use]
    pub fn prefill_len(&self) -> Option<usize> {
        self.prefill_len
    }

    /// Live ring rows captured at checkpoint (`0..=RING_WINDOW`).
    #[must_use]
    pub fn ring_len(&self) -> usize {
        self.ring_len
    }

    /// Next ring write slot captured at checkpoint (`0..RING_WINDOW`).
    #[must_use]
    pub fn ring_pos(&self) -> usize {
        self.ring_pos
    }

    /// Effective key count (`prefill_len + ring_len`) captured at checkpoint —
    /// the number of keys a decode query attended over at that point.
    #[must_use]
    pub fn effective_len(&self) -> usize {
        self.prefill_len.unwrap_or(0) + self.ring_len
    }
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
            decode_writes: 0,
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
        self.decode_writes = 0;
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
        // Count this write so checkpoint/rollback (bd-1azu.31) can tally the
        // discarded speculative steps; not used by the attention math.
        self.decode_writes += 1;
        Ok(slot)
    }

    /// Capture the current generated-tail write cursors as a [`RingCheckpoint`]
    /// for a later [`RingCache::rollback_to`] (bd-1azu.31 — speculative-decode
    /// roll-back). `O(1)`, copies NO K/V buffer: only the ring `len`/`pos`, the
    /// (frozen) prefill boundary, and the decode-write counter. Take this BEFORE
    /// writing speculative decode steps; if they are rejected, [`rollback_to`]
    /// erases them so they leave no trace.
    ///
    /// [`rollback_to`]: RingCache::rollback_to
    #[must_use]
    pub fn checkpoint(&self) -> RingCheckpoint {
        RingCheckpoint {
            prefill_len: self.prefill_len,
            ring_len: self.ring_len,
            ring_pos: self.ring_pos,
            decode_writes: self.decode_writes,
        }
    }

    /// Roll the ring back to a prior [`RingCheckpoint`], discarding every decode
    /// step written since it was taken so subsequent reads/writes behave as if
    /// those speculative steps never happened (bd-1azu.31). Restores only the
    /// generated-tail cursors (`ring_len`/`ring_pos`/`decode_writes`); the
    /// reference (prefill) block is frozen and untouched. The stale K/V bytes left
    /// in the now-dead ring slots (`>= ring_len`) are never read — the decode read
    /// paths scan exactly `0..ring_len` — so no buffer needs clearing or copying,
    /// which is what makes the roll-back both `O(1)` and byte-for-byte lossless.
    ///
    /// # Panics (debug only)
    /// `debug_assert`s the [LOSSLESS contract](RingCheckpoint): the checkpoint
    /// must belong to this cache (same `prefill_len`), must not be from the future
    /// (its `decode_writes` not above the cache's), and the discarded steps must
    /// not have wrapped the ring past the checkpoint's live rows
    /// (`checkpoint.ring_len + discarded <= RING_WINDOW`) — otherwise an evicted
    /// slot's prior K/V is physically gone and the roll-back would NOT be lossless.
    pub fn rollback_to(&mut self, cp: &RingCheckpoint) {
        // Contract guard (debug builds only — compiled out of release, where the
        // local would otherwise be unused). `discarded` lives entirely inside the
        // gated block so there is no release-build dead-code/unused-var warning.
        #[cfg(debug_assertions)]
        {
            assert_eq!(
                cp.prefill_len, self.prefill_len,
                "rswa: rollback_to checkpoint prefill_len {:?} != cache {:?}",
                cp.prefill_len, self.prefill_len
            );
            assert!(
                cp.decode_writes <= self.decode_writes,
                "rswa: rollback_to checkpoint is from the future \
                 (cp.decode_writes={} > cache={})",
                cp.decode_writes,
                self.decode_writes
            );
            let discarded = self.decode_writes - cp.decode_writes;
            assert!(
                cp.ring_len + discarded <= RING_WINDOW,
                "rswa: rollback_to is not lossless — {discarded} discarded steps \
                 wrapped the ring past checkpoint.ring_len={} (W={RING_WINDOW}); an \
                 evicted slot's prior K/V cannot be restored from cursors alone",
                cp.ring_len
            );
        }
        self.ring_len = cp.ring_len;
        self.ring_pos = cp.ring_pos;
        self.decode_writes = cp.decode_writes;
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

/// Batched R-SWA KV cache: `B` in-flight page-streams, each owning its own
/// `n_layers` per-layer [`RingCache`]s (bd-1azu.4 — the Phase-6 continuous-batch
/// decode spine, bd-1azu).
///
/// Every `(stream, layer)` is an INDEPENDENT [`RingCache`], byte-for-byte the
/// single-page structure, so a stream's KV state is bit-exact to running that
/// page ALONE — the lossless default. The batched decode step consumes the `B`
/// streams' K/V from the batched GEMM (batched-decode-gemm) and writes each into
/// ITS OWN ring via [`BatchedRingCache::write_decode_step`]; the
/// score/softmax/value attention stays per-stream (per [`decode_attention`],
/// bd-1waa-safe), NEVER cross-stream key-batching. Streams that share a prefix
/// can later be made to point at ONE shared reference block — that is the
/// prefix-kv-share seam (bd-1azu.12), OFF by default here.
///
/// **KV working-set arithmetic** (the figure the L3-residency budget in
/// `l3-layer-sync-engine` / `ccd-expert-service` depends on). For the default
/// f32 path, per stream per layer the live K+V bytes are
/// `2 * NUM_HEADS * (ref_capacity + RING_WINDOW) * HEAD_DIM * size_of::<f32>()`,
/// so the whole batched working set is that summed over the `B` streams' layers
/// — see [`BatchedRingCache::kv_f32_bytes`]. At the int8 baseline ~12 MB/page the
/// f32 default is 4× that (~48 MB/page → ~12 GB for 256 streams, trivial in
/// 499 GB), NOT 3 GB. With the int8-KV lever (`FOCR_INT8_KV`, bd-1waa,
/// gated/lossy) each stream additionally carries an int8 mirror; the default path
/// allocates none.
#[derive(Debug)]
pub struct BatchedRingCache {
    /// `streams[s]` holds page-stream `s`'s `n_layers` per-layer caches.
    streams: Vec<Vec<RingCache>>,
    /// Layers per stream (every stream has the same count).
    n_layers: usize,
}

impl BatchedRingCache {
    /// Build `B = prefill_caps.len()` page-streams, each with `n_layers` per-layer
    /// [`RingCache`]s sized to that stream's worst-case prefill (`prefill_caps[s]`,
    /// the reference-block capacity `m`). Pre-allocated and allocation-free
    /// thereafter, exactly like the single-page path.
    ///
    /// # Panics
    /// Panics if `n_layers == 0` or `prefill_caps` is empty (a batched forward
    /// always has ≥1 layer and ≥1 stream — a programming error otherwise).
    #[must_use]
    pub fn new(prefill_caps: &[usize], n_layers: usize) -> Self {
        assert!(n_layers > 0, "BatchedRingCache: n_layers must be > 0");
        assert!(
            !prefill_caps.is_empty(),
            "BatchedRingCache: needs at least one stream"
        );
        let streams = prefill_caps
            .iter()
            .map(|&cap| (0..n_layers).map(|_| RingCache::new(cap)).collect())
            .collect();
        Self { streams, n_layers }
    }

    /// Adopt `B` pre-built, already-prefilled per-stream [`RingCache`] vectors as
    /// the batched cache — the bd-1azu.11 spine seam. Each `streams[s]` is page
    /// `s`'s full `Vec<RingCache>` straight out of
    /// [`super::decoder::prefill_with_cache_i8`] (one [`RingCache`] per decoder
    /// layer, its reference block already recorded), so stream `s` decodes
    /// byte-for-byte as if it had run alone. Stream slot `s` here is the cache
    /// index the scheduler's [`super::batch_scheduler::DecoderBatchStep`] addresses.
    ///
    /// # Errors
    /// [`FocrError::Other`] if `streams` is empty, the first stream has zero
    /// layers, or any stream's layer count differs from the first (every stream
    /// must carry the same number of layers for a batched forward).
    pub fn from_streams(streams: Vec<Vec<RingCache>>) -> FocrResult<Self> {
        let Some(first) = streams.first() else {
            return Err(FocrError::Other(anyhow::anyhow!(
                "rswa: BatchedRingCache::from_streams needs at least one stream"
            )));
        };
        let n_layers = first.len();
        if n_layers == 0 {
            return Err(FocrError::Other(anyhow::anyhow!(
                "rswa: BatchedRingCache::from_streams stream 0 has zero layers"
            )));
        }
        for (s, layers) in streams.iter().enumerate() {
            if layers.len() != n_layers {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "rswa: BatchedRingCache::from_streams stream {s} has {} layers != {n_layers}",
                    layers.len()
                )));
            }
        }
        Ok(Self { streams, n_layers })
    }

    /// Number of in-flight page-streams `B`.
    #[must_use]
    pub fn num_streams(&self) -> usize {
        self.streams.len()
    }

    /// Layers per stream.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.n_layers
    }

    /// Stream `s`'s per-layer caches (read-only).
    ///
    /// # Panics
    /// Panics if `s >= num_streams()`.
    #[must_use]
    pub fn stream(&self, s: usize) -> &[RingCache] {
        &self.streams[s]
    }

    /// One `(stream, layer)` cache (read-only) — e.g. for [`decode_attention`].
    ///
    /// # Panics
    /// Panics if `s >= num_streams()` or `layer >= num_layers()`.
    #[must_use]
    pub fn layer(&self, s: usize, layer: usize) -> &RingCache {
        &self.streams[s][layer]
    }

    /// One `(stream, layer)` cache (mutable) — for prefill / decode writes.
    ///
    /// # Panics
    /// Panics if `s >= num_streams()` or `layer >= num_layers()`.
    pub fn layer_mut(&mut self, s: usize, layer: usize) -> &mut RingCache {
        &mut self.streams[s][layer]
    }

    /// Record stream `s` layer `layer`'s prefill reference block. Per-stream
    /// independent; delegates to [`RingCache::record_prefill`].
    ///
    /// # Errors
    /// Propagates [`RingCache::record_prefill`]'s errors.
    pub fn record_prefill(
        &mut self,
        s: usize,
        layer: usize,
        k: &[f32],
        v: &[f32],
        seq: usize,
    ) -> FocrResult<()> {
        self.streams[s][layer].record_prefill(k, v, seq)
    }

    /// Write one decode token's K/V for stream `s` layer `layer` into ITS ring.
    /// Returns the physical ring slot written. Delegates to
    /// [`RingCache::write_decode_step`].
    ///
    /// # Errors
    /// Propagates [`RingCache::write_decode_step`]'s errors.
    pub fn write_decode_step(
        &mut self,
        s: usize,
        layer: usize,
        k_step: &[f32],
        v_step: &[f32],
    ) -> FocrResult<usize> {
        self.streams[s][layer].write_decode_step(k_step, v_step)
    }

    /// Capture stream `s` layer `layer`'s ring write cursors as a
    /// [`RingCheckpoint`] for a later [`BatchedRingCache::rollback_to`]
    /// (bd-1azu.31 — per-stream speculative-decode roll-back). Per-stream
    /// independent; delegates to [`RingCache::checkpoint`].
    ///
    /// # Panics
    /// Panics if `s >= num_streams()` or `layer >= num_layers()`.
    #[must_use]
    pub fn checkpoint(&self, s: usize, layer: usize) -> RingCheckpoint {
        self.streams[s][layer].checkpoint()
    }

    /// Roll stream `s` layer `layer`'s ring back to a prior [`RingCheckpoint`],
    /// discarding the speculative decode steps written into THAT stream's ring so
    /// it behaves as if they never happened. Per-stream independent (a roll-back
    /// on one stream never touches another); delegates to
    /// [`RingCache::rollback_to`] under the identical lossless contract / debug
    /// asserts.
    ///
    /// # Panics
    /// Panics if `s >= num_streams()` or `layer >= num_layers()`; in debug builds
    /// also on a checkpoint that violates the lossless contract (see
    /// [`RingCache::rollback_to`]).
    pub fn rollback_to(&mut self, s: usize, layer: usize, cp: &RingCheckpoint) {
        self.streams[s][layer].rollback_to(cp);
    }

    /// Total default-path (f32) KV working-set bytes across all streams/layers —
    /// the auditable figure for the L3-residency budget. Sums the live K+V
    /// reference + ring regions: `Σ_s Σ_layer 2 * NUM_HEADS * (ref_capacity_s +
    /// RING_WINDOW) * HEAD_DIM * size_of::<f32>()`. The int8 mirror, when present,
    /// is extra and intentionally not counted here (this is the f32 baseline).
    #[must_use]
    pub fn kv_f32_bytes(&self) -> usize {
        const F32: usize = core::mem::size_of::<f32>();
        let mut bytes = 0usize;
        for stream in &self.streams {
            for cache in stream {
                let rows = cache.ref_capacity() + RING_WINDOW;
                bytes += 2 * NUM_HEADS * rows * HEAD_DIM * F32;
            }
        }
        bytes
    }
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

/// Batched R-SWA **decode** attention stage (bd-1azu.5 — the Phase-6
/// continuous-batch attention stage, bd-1azu): run all `B` in-flight
/// page-streams' decode queries for ONE `layer` and return the `B` per-stream
/// decode contexts.
///
/// `queries` is the `B` streams' decode queries STACKED head-major flat as
/// `[B, NUM_HEADS * HEAD_DIM]` — one `q` row per stream in `cache` order (row `s`
/// is stream `s`'s `[NUM_HEADS, HEAD_DIM]` query, already RoPE'd by the caller at
/// its TRUE absolute position, [SPEC-095]). Each stream's K/V for this step must
/// already be in its ring (call [`BatchedRingCache::write_decode_step`] for every
/// stream first, exactly as the single-page path writes before it attends —
/// OQ-3). Returns one `[1, NUM_HEADS * HEAD_DIM]` context [`Mat`] per stream, in
/// stream order, ready for the batched `o_proj`.
///
/// **LOSSLESS by construction** (Doctrine #1). `result[s]` is byte-for-byte the
/// single-page [`decode_attention`] applied to `cache.layer(s, layer)` with
/// stream `s`'s query: the attention is run PER STREAM over its OWN
/// `reference ++ ring` union — never key-batching across streams, because each
/// stream's key set, effective length, and ring regime (warm-up vs steady-state)
/// are independent. This preserves both the per-key f32 reduction order
/// (bd-1waa) AND the env-gated per-stream dispatch (`FOCR_ATTN_GEMM` /
/// `FOCR_INT8_KV`) identically to the single-page call. The future bit-exact
/// cross-stream QUERY-dimension batching lands at the clearly-marked SEAM in the
/// body (OFF here — Wave-1 ships the correct per-stream loop).
///
/// # Errors
/// [`FocrError::Other`] if `layer >= num_layers()` or `queries.len() != B *
/// NUM_HEADS * HEAD_DIM`; propagates [`decode_attention`]'s per-stream errors
/// (e.g. a stream whose prefill was never recorded, or an empty key set).
pub fn batched_decode_attention(
    cache: &BatchedRingCache,
    layer: usize,
    queries: &[f32],
) -> FocrResult<Vec<Mat>> {
    if layer >= cache.num_layers() {
        return Err(FocrError::Other(anyhow::anyhow!(
            "rswa: batched_decode_attention layer {layer} >= num_layers {}",
            cache.num_layers()
        )));
    }
    let b = cache.num_streams();
    let per_stream = NUM_HEADS * HEAD_DIM;
    let expect = b * per_stream;
    if queries.len() != expect {
        return Err(FocrError::Other(anyhow::anyhow!(
            "rswa: batched_decode_attention queries len {} != B*NUM_HEADS*HEAD_DIM {} \
             (B={b})",
            queries.len(),
            expect
        )));
    }

    // ── SEAM (bd-1waa-safe — bit-exact QUERY-dimension batching; OFF by default) ─
    // The future optimization regroups the OUTER (stream × head) query/key-row
    // iteration of the loop below into fewer kernel launches WITHOUT touching the
    // inner per-key math. Because every (stream, head) shares the same `HEAD_DIM`
    // contraction and `scale`:
    //   * streams pointing at ONE shared reference block (the prefix-kv-share seam,
    //     bd-1azu.12) collapse their `QK^T` over that block into a single
    //     `M = Σ q-rows` GEMM (the bd-1azu.2 batched-M int8 gate already proves
    //     M=B GEMM == m=1 GEMV bit-for-bit); and
    //   * even without sharing, the per-(stream,head) score rows can be tiled into
    //     one batched-GEMV launch.
    // The hard invariant that keeps this lossless: the regrouping may ONLY reorder
    // the outer query/key-row loop — NEVER the inner `HEAD_DIM` reduction or the
    // per-key online-softmax fold order that `decode_attention` (and its `FOCR_*`
    // levers) fix. Until that lands proven-bit-exact, this stage is the correct
    // per-stream loop over the existing kernel.
    let mut contexts = Vec::with_capacity(b);
    for s in 0..b {
        let q_s = &queries[s * per_stream..(s + 1) * per_stream];
        // Per-stream, bit-exact to the single-page path: stream `s` attends over
        // ITS OWN reference ++ ring union via the existing kernel + dispatch.
        contexts.push(decode_attention(cache.layer(s, layer), q_s)?);
    }
    Ok(contexts)
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

    // ── checkpoint / rollback (bd-1azu.31) — direct private-row byte-identity ───

    /// Roll-back restores the cursors AND leaves every accepted (live) K/V row
    /// byte-for-byte identical to a control cache that never wrote the
    /// speculative steps — proved here by reading the private `ring_*`/`ref_*`
    /// rows directly (the integration test proves the same through the public
    /// `decode_attention` output). Also shows the design is O(1): the dead ring
    /// slots are NOT cleared (they still hold the discarded draft) yet are never
    /// read, since the cursors confine all reads to `0..ring_len`.
    #[test]
    fn checkpoint_rollback_restores_ring_rows_byte_identical() {
        let pf = 5usize;
        let accept = 9usize;
        let accept_tok =
            |t: usize| one_token(|h, d| ((h * 7 + d * 3 + t * 5) % 13) as f32 * 0.1 - 0.6);
        let build = || {
            let mut c = RingCache::new(64);
            let k = fill_head_major(pf, |r, d| ((r * 3 + d) % 11) as f32 * 0.07 - 0.3);
            let v = fill_head_major(pf, |r, d| ((r + d * 2) % 9) as f32 * 0.05 - 0.2);
            c.record_prefill(&k, &v, pf).unwrap();
            for t in 0..accept {
                let kt = accept_tok(t);
                c.write_decode_step(&kt, &kt).unwrap();
            }
            c
        };
        let control = build();
        let mut test = build();

        let cp = test.checkpoint();
        assert_eq!(cp.ring_len(), accept);
        assert_eq!(cp.ring_pos(), accept);
        assert_eq!(cp.prefill_len(), Some(pf));
        assert_eq!(cp.effective_len(), pf + accept);

        // Speculate 6 distinct (clearly non-zero) draft steps, then roll back.
        for t in 0..6 {
            let kt = one_token(|h, d| ((h + d + t) % 5) as f32 * 0.3 + 0.9);
            test.write_decode_step(&kt, &kt).unwrap();
        }
        assert_eq!(test.ring_len(), accept + 6);
        test.rollback_to(&cp);

        // Cursors (including the private decode-write counter) restored exactly.
        assert_eq!(test.ring_len(), accept);
        assert_eq!(test.ring_pos(), accept);
        assert_eq!(test.decode_writes, accept);
        assert_eq!(test.effective_len(), pf + accept);
        assert!(!test.is_warm());

        // Every LIVE K/V row is byte-identical to the never-speculated control.
        for h in 0..NUM_HEADS {
            for r in 0..test.ring_len() {
                assert_eq!(
                    test.ring_k_row(h, r),
                    control.ring_k_row(h, r),
                    "ring_k h{h} r{r}"
                );
                assert_eq!(
                    test.ring_v_row(h, r),
                    control.ring_v_row(h, r),
                    "ring_v h{h} r{r}"
                );
            }
            for r in 0..pf {
                assert_eq!(
                    test.ref_k_row(h, r),
                    control.ref_k_row(h, r),
                    "ref_k h{h} r{r}"
                );
                assert_eq!(
                    test.ref_v_row(h, r),
                    control.ref_v_row(h, r),
                    "ref_v h{h} r{r}"
                );
            }
        }
        // The first dead slot still holds the discarded draft (not cleared) — the
        // O(1) roll-back relies on it simply never being read past `ring_len`.
        assert_ne!(
            test.ring_k_row(0, accept),
            control.ring_k_row(0, accept),
            "dead slot is intentionally left stale (never read)"
        );

        // A fresh accepted write lands in the checkpoint's slot, as if the
        // speculation never happened.
        let kt = accept_tok(accept);
        assert_eq!(test.write_decode_step(&kt, &kt).unwrap(), accept);
        assert_eq!(test.ring_len(), accept + 1);
    }

    /// The contract guard refuses a roll-back that would have to resurrect an
    /// evicted (steady-state-overwritten) slot — indices alone cannot restore it.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "not lossless")]
    fn rollback_rejects_eviction_in_steady_state() {
        let mut c = RingCache::new(8);
        let k = fill_head_major(3, |_, _| 0.25);
        c.record_prefill(&k, &k, 3).unwrap();
        for _ in 0..RING_WINDOW {
            let t = one_token(|_, _| 0.5);
            c.write_decode_step(&t, &t).unwrap();
        }
        assert!(c.is_warm());
        let cp = c.checkpoint();
        // One steady-state write overwrites slot 0 (live at the checkpoint).
        let t = one_token(|_, _| 1.0);
        c.write_decode_step(&t, &t).unwrap();
        c.rollback_to(&cp); // must panic: not lossless.
    }

    /// A no-op roll-back to a fresh checkpoint (no writes in between) leaves the
    /// cache bit-identical and is always allowed.
    #[test]
    fn rollback_noop_is_identity() {
        let mut c = RingCache::new(16);
        let k = fill_head_major(4, |r, d| ((r + d) % 7) as f32 * 0.1);
        c.record_prefill(&k, &k, 4).unwrap();
        let t = one_token(|_, _| 0.3);
        c.write_decode_step(&t, &t).unwrap();
        let cp = c.checkpoint();
        c.rollback_to(&cp);
        assert_eq!(c.ring_len(), 1);
        assert_eq!(c.ring_pos(), 1);
        assert_eq!(c.decode_writes, 1);
    }
}

/// Batched R-SWA KV cache tests (bd-1azu.4): per-stream bit-exactness vs the
/// single-page path, bounded-KV / reference-unwritten invariants at `B` up to
/// 256, and the f32 KV-budget arithmetic the L3-residency beads depend on.
#[cfg(test)]
mod batched_ring_tests {
    use super::{BatchedRingCache, HEAD_DIM, NUM_HEADS, RING_WINDOW, RingCache, decode_attention};

    /// Deterministic value in `[-1, 1)` per `(stream, layer, head, row, dim)` so
    /// every stream's K/V is distinct yet reproducible without a dependency.
    fn val(stream: usize, layer: usize, h: usize, r: usize, d: usize) -> f32 {
        let mut x = (stream as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (layer as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ (h as u64).wrapping_mul(0x1656_67B1_9E37_79F9)
            ^ (r as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
            ^ (d as u64).wrapping_mul(0x27D4_EB2F_1656_67C5);
        x ^= x >> 33;
        x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        x ^= x >> 33;
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// Head-major `[NUM_HEADS, seq, HEAD_DIM]` prefill K and V (V offset by one
    /// dim so K != V).
    fn build_prefill(stream: usize, layer: usize, seq: usize) -> (Vec<f32>, Vec<f32>) {
        let mut k = vec![0.0f32; NUM_HEADS * seq * HEAD_DIM];
        let mut v = vec![0.0f32; NUM_HEADS * seq * HEAD_DIM];
        for h in 0..NUM_HEADS {
            for r in 0..seq {
                for d in 0..HEAD_DIM {
                    let idx = h * seq * HEAD_DIM + r * HEAD_DIM + d;
                    k[idx] = val(stream, layer, h, r, d);
                    v[idx] = val(stream, layer, h, r, d + 1);
                }
            }
        }
        (k, v)
    }

    /// One decode token's `[NUM_HEADS, HEAD_DIM]` K/V at logical step `t` (row
    /// space offset so it never collides with prefill rows).
    fn build_step(stream: usize, layer: usize, t: usize) -> (Vec<f32>, Vec<f32>) {
        let mut k = vec![0.0f32; NUM_HEADS * HEAD_DIM];
        let mut v = vec![0.0f32; NUM_HEADS * HEAD_DIM];
        for h in 0..NUM_HEADS {
            for d in 0..HEAD_DIM {
                let idx = h * HEAD_DIM + d;
                k[idx] = val(stream, layer, h, 1_000 + t, d);
                v[idx] = val(stream, layer, h, 1_000 + t, d + 1);
            }
        }
        (k, v)
    }

    /// One decode query `[NUM_HEADS, HEAD_DIM]`.
    fn build_q(stream: usize, layer: usize, t: usize) -> Vec<f32> {
        let mut q = vec![0.0f32; NUM_HEADS * HEAD_DIM];
        for h in 0..NUM_HEADS {
            for d in 0..HEAD_DIM {
                q[h * HEAD_DIM + d] = val(stream, layer, h, 2_000 + t, d);
            }
        }
        q
    }

    /// THE invariant the spine rests on: stream `s`'s `(layer)` KV state is
    /// byte-for-byte identical to driving that page alone through a standalone
    /// `Vec<RingCache>` — including the decode-attention output. Covers warm-up
    /// (`< RING_WINDOW`) and steady-state (`>= RING_WINDOW`) regimes.
    #[test]
    fn per_stream_independent_and_bit_exact() {
        let n_layers = 2usize;
        let caps = [5usize, 9, 16, 4, 7, 20, 3, 11]; // B=8, distinct prefill caps
        let mut bc = BatchedRingCache::new(&caps, n_layers);
        assert_eq!(bc.num_streams(), caps.len());
        assert_eq!(bc.num_layers(), n_layers);

        for (s, &cap) in caps.iter().enumerate() {
            for l in 0..n_layers {
                let (k, v) = build_prefill(s, l, cap);
                bc.record_prefill(s, l, &k, &v, cap).expect("prefill");
            }
        }

        // Standalone mirror of one chosen stream, driven with identical inputs.
        let st = 5usize;
        let mut standalone: Vec<RingCache> =
            (0..n_layers).map(|_| RingCache::new(caps[st])).collect();
        for (l, cache) in standalone.iter_mut().enumerate() {
            let (k, v) = build_prefill(st, l, caps[st]);
            cache
                .record_prefill(&k, &v, caps[st])
                .expect("standalone prefill");
        }

        let steps = 200usize; // crosses RING_WINDOW=128 into steady state
        for t in 0..steps {
            for (s, _) in caps.iter().enumerate() {
                for l in 0..n_layers {
                    let (k, v) = build_step(s, l, t);
                    bc.write_decode_step(s, l, &k, &v).expect("batched step");
                }
            }
            for (l, cache) in standalone.iter_mut().enumerate() {
                let (k, v) = build_step(st, l, t);
                cache.write_decode_step(&k, &v).expect("standalone step");
            }

            if t == 0 || t == RING_WINDOW - 1 || t == RING_WINDOW || t == steps - 1 {
                for l in 0..n_layers {
                    let bcl = bc.layer(st, l);
                    let sal = &standalone[l];
                    assert_eq!(
                        bcl.prefill_len(),
                        sal.prefill_len(),
                        "prefill_len l{l} t{t}"
                    );
                    assert_eq!(bcl.ring_len(), sal.ring_len(), "ring_len l{l} t{t}");
                    assert_eq!(bcl.ring_pos(), sal.ring_pos(), "ring_pos l{l} t{t}");
                    assert_eq!(
                        bcl.effective_len(),
                        sal.effective_len(),
                        "eff_len l{l} t{t}"
                    );
                    assert_eq!(bcl.is_warm(), sal.is_warm(), "is_warm l{l} t{t}");
                    let q = build_q(st, l, t);
                    let a = decode_attention(bcl, &q).expect("batched attn");
                    let b = decode_attention(sal, &q).expect("standalone attn");
                    assert_eq!(a.rows, b.rows);
                    assert_eq!(a.cols, b.cols);
                    assert_eq!(
                        a.data, b.data,
                        "stream {st} layer {l} step {t}: attention differs"
                    );
                }
            }
        }
    }

    /// At `B = 256` in-flight streams: every stream's KV stays bounded
    /// (`effective_len <= prefill_len + RING_WINDOW`), the ring saturates, and the
    /// reference block is NEVER written by decode (`prefill_len` constant).
    #[test]
    fn large_batch_invariants_bounded_and_reference_unwritten() {
        let b = 256usize;
        let seq = 4usize;
        let caps = vec![seq; b];
        let mut bc = BatchedRingCache::new(&caps, 1);
        assert_eq!(bc.num_streams(), b);

        for s in 0..b {
            let (k, v) = build_prefill(s, 0, seq);
            bc.record_prefill(s, 0, &k, &v, seq).expect("prefill");
        }
        let steps = 200usize; // > RING_WINDOW → steady state
        for t in 0..steps {
            for s in 0..b {
                let (k, v) = build_step(s, 0, t);
                bc.write_decode_step(s, 0, &k, &v).expect("step");
            }
        }
        for s in 0..b {
            let c = bc.layer(s, 0);
            assert_eq!(
                c.prefill_len(),
                Some(seq),
                "stream {s}: reference block must be untouched by decode"
            );
            assert_eq!(c.ring_len(), RING_WINDOW, "stream {s}: ring saturates");
            assert!(c.is_warm(), "stream {s}: warm after {steps} steps");
            assert_eq!(c.effective_len(), seq + RING_WINDOW, "stream {s}");
            assert!(
                c.effective_len() <= c.prefill_len().expect("prefilled") + RING_WINDOW,
                "stream {s}: generated-token KV is bounded"
            );
        }
    }

    /// The f32 KV working-set figure (the L3-residency budget) equals the
    /// per-stream-per-layer formula summed over streams/layers.
    #[test]
    fn kv_f32_bytes_matches_budget_arithmetic() {
        let caps = [10usize, 20, 30];
        let n_layers = 12usize;
        let bc = BatchedRingCache::new(&caps, n_layers);
        let f32sz = core::mem::size_of::<f32>();
        let expect: usize = caps
            .iter()
            .map(|&cap| n_layers * 2 * NUM_HEADS * (cap + RING_WINDOW) * HEAD_DIM * f32sz)
            .sum();
        assert_eq!(bc.kv_f32_bytes(), expect);
        assert!(bc.kv_f32_bytes() > 0);
    }

    /// `from_streams` must ADOPT pre-built, already-prefilled per-stream rings so
    /// the batched cache decodes byte-for-byte as the standalone `Vec<RingCache>`
    /// it was handed — the bd-1azu.11 prefill→spine handoff. Build standalone
    /// streams, move them into a [`BatchedRingCache`], and verify the adopted slot
    /// stays bit-exact to an independent mirror across decode steps.
    #[test]
    fn from_streams_adopts_prebuilt_rings_bit_exact() {
        let n_layers = 2usize;
        let caps = [6usize, 13];
        // Each stream's standalone Vec<RingCache> with prefill recorded.
        let mut built: Vec<Vec<RingCache>> = Vec::new();
        for (s, &cap) in caps.iter().enumerate() {
            let mut layers: Vec<RingCache> = (0..n_layers).map(|_| RingCache::new(cap)).collect();
            for (l, cache) in layers.iter_mut().enumerate() {
                let (k, v) = build_prefill(s, l, cap);
                cache.record_prefill(&k, &v, cap).expect("prefill");
            }
            built.push(layers);
        }
        // Independent mirror of stream 1 to compare against after adoption.
        let st = 1usize;
        let mut standalone: Vec<RingCache> =
            (0..n_layers).map(|_| RingCache::new(caps[st])).collect();
        for (l, cache) in standalone.iter_mut().enumerate() {
            let (k, v) = build_prefill(st, l, caps[st]);
            cache
                .record_prefill(&k, &v, caps[st])
                .expect("mirror prefill");
        }

        let mut bc = BatchedRingCache::from_streams(built).expect("adopt");
        assert_eq!(bc.num_streams(), caps.len());
        assert_eq!(bc.num_layers(), n_layers);

        for t in 0..5usize {
            for (l, cache) in standalone.iter_mut().enumerate() {
                let (k, v) = build_step(st, l, t);
                cache.write_decode_step(&k, &v).expect("mirror step");
            }
            for l in 0..n_layers {
                let (k, v) = build_step(st, l, t);
                bc.write_decode_step(st, l, &k, &v).expect("adopted step");
            }
            for l in 0..n_layers {
                let q = build_q(st, l, t);
                let a = decode_attention(bc.layer(st, l), &q).expect("adopted attn");
                let b = decode_attention(&standalone[l], &q).expect("mirror attn");
                assert_eq!(
                    a.data, b.data,
                    "adopted stream {st} layer {l} step {t} differs"
                );
            }
        }
    }

    /// `from_streams` rejects an empty batch and ragged per-stream layer counts.
    #[test]
    fn from_streams_rejects_empty_and_ragged() {
        assert!(BatchedRingCache::from_streams(Vec::new()).is_err());
        let s0: Vec<RingCache> = vec![RingCache::new(4), RingCache::new(4)];
        let s1: Vec<RingCache> = vec![RingCache::new(4)]; // ragged: 1 layer vs 2
        assert!(BatchedRingCache::from_streams(vec![s0, s1]).is_err());
    }
}
