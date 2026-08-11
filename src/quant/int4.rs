//! Group-quantized int4 packing + the in-register unpack to int8 (the Phase-4
//! decode-bandwidth wedge, AGENTS.md doctrine #4).
//!
//! ## Why int4, and how it reaches the int8 GEMM
//!
//! No CPU has an int4 multiply-accumulate. The win is **bandwidth**, not a new
//! MAC: the expert weights dominate the decode working set, so storing them at 4
//! bits halves the bytes the GEMM must stream. The packed nibbles are
//! **unpacked to int8 in-register** ([`unpack_int4_to_i8`]) and fed to the exact
//! same int8 kernel (`igemm_s8s8`). This module owns the *packing*; the unpack
//! must reproduce the precise int8 values the GEMM consumes, bit-for-bit.
//!
//! ## Layout (matches the committed reader `Weights::qint4`, `docs/focrq-format.md`)
//!
//! * Logical weight is row-major `[n, k]` (`n` = output channels, `k` =
//!   contraction). `k` is even.
//! * **Per-group symmetric scales** along the K dimension within each output row:
//!   `scale[row, g] = max(|w[row, g·G .. (g+1)·G]|) / 7`, one f32 per group.
//!   Scale count is `n · (k / G)`. Dequant is `f32(q4) · scale[row, group]`.
//! * **Signed two's-complement int4 in `[-8, 7]`**; `q = clamp(round_ties_even(
//!   w / scale), -8, 7)`. An all-zero group stores `scale = 1.0`, all-zero
//!   nibbles (no NaN), exactly as the int8 path handles all-zero rows.
//! * **Packing: two nibbles per byte, low nibble first then high nibble.** For a
//!   row of `k` values the byte at index `j` holds value `2j` in its low nibble
//!   and value `2j+1` in its high nibble. Each nibble is the int4 two's
//!   complement: `(q as u8) & 0x0F`. The packed payload is `n · (k / 2)` bytes.
//!
//! ## Group-size choice (`docs/focrq-format.md` §QInt4PerGroup)
//!
//! `group_size ∈ {16, 32}` only (tiers `Int4G16` / `Int4G32`). The committed
//! reader requires `group_size` to **divide `k` exactly** (it rejects a
//! `group_size` that does not divide `k`), and the quantized decoder GEMM
//! contraction dims are all multiples of 16 and 32 (hidden 1280, expert
//! intermediate 896, dense intermediate 6848, projector 2048), so exact division
//! holds for every real tensor. We default to **16** (`Int4G16`): finer groups
//! track the per-channel weight range more tightly (less quantization error per
//! group) at the cost of `k/16` vs `k/32` scales — a few KB more per tensor,
//! negligible against the int4 bandwidth win. 32 is offered for tensors where
//! the allocator trades that accuracy for the smaller scale table.

use half::bf16;

/// Default int4 group size (elements per group along K). 16 = tier `Int4G16`
/// (`docs/focrq-format.md`). Finer than 32 ⇒ tighter per-group range ⇒ lower
/// quant error, at a small extra scale-table cost.
pub const DEFAULT_GROUP_SIZE: usize = 16;

/// The two valid int4 group sizes (`docs/focrq-format.md` §QInt4PerGroup).
pub const VALID_GROUP_SIZES: [usize; 2] = [16, 32];

/// int4 signed range: two's complement in `[-8, 7]`.
pub const Q4_MIN: i32 = -8;
/// int4 signed range upper bound.
pub const Q4_MAX: i32 = 7;

/// A group-quantized int4 weight, ready to write to a `.focrq` `QInt4PerGroup`
/// record (payload = `packed`, inline scales = `scales`).
///
/// Mirrors [`crate::native_engine::tensor::QInt4`]: `packed` is `n · k/2` bytes
/// (two nibbles each, low-then-high), `scales` is `n · (k / group_size)` f32.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedInt4 {
    /// Two signed int4 nibbles per byte, row-major `[n, k/2]` (low nibble =
    /// even index, high nibble = odd index).
    pub packed: Vec<u8>,
    /// Per-group scales, length `n · (k / group_size)`.
    pub scales: Vec<f32>,
    /// Output channels (rows).
    pub n: usize,
    /// Contraction length (columns); even, and a multiple of `group_size`.
    pub k: usize,
    /// Elements per quantization group along K (16 or 32).
    pub group_size: usize,
}

impl QuantizedInt4 {
    /// The raw packed payload bytes (already the on-disk form).
    #[must_use]
    pub fn packed_bytes(&self) -> Vec<u8> {
        self.packed.clone()
    }

    /// The inline scale bytes (`n · k/group_size` little-endian f32).
    #[must_use]
    pub fn scale_bytes(&self) -> Vec<u8> {
        self.scales.iter().flat_map(|&s| s.to_le_bytes()).collect()
    }

    /// Number of groups per output row (`k / group_size`).
    #[must_use]
    pub fn groups_per_row(&self) -> usize {
        assert!(
            self.group_size != 0 && self.k.is_multiple_of(self.group_size),
            "QuantizedInt4::groups_per_row: group_size {} must divide k {}",
            self.group_size,
            self.k
        );
        self.k / self.group_size
    }
}

/// Round to nearest, ties to even — the pinned converter rounding (shared rule
/// with the int8 path, keeps the quant a pure function of the input).
#[inline]
#[must_use]
fn round_ties_even_f32(x: f32) -> f32 {
    x.round_ties_even()
}

/// Pack a single signed int4 value (`-8..=7`) into a nibble (`0x0..=0xF`).
#[inline]
#[must_use]
fn nibble_of(q: i8) -> u8 {
    (q as u8) & 0x0F
}

/// Sign-extend a nibble (`0x0..=0xF`) back to a signed int4 value (`-8..=7`) as
/// i8. This is the inverse of [`nibble_of`] and the exact value the GEMM sees.
#[inline]
#[must_use]
fn i8_of_nibble(nib: u8) -> i8 {
    let n = nib & 0x0F;
    // Bit 3 is the sign bit; if set, subtract 16 to sign-extend.
    if n & 0x08 != 0 {
        (n as i32 - 16) as i8
    } else {
        n as i8
    }
}

/// Quantize a row-major `[n, k]` f32 weight matrix to group-quantized int4.
///
/// `group_size` must be 16 or 32 and divide `k` (the committed reader requires
/// exact division; every real quantized tensor satisfies it). Symmetric per-group
/// scales (`max|group| / 7`); an all-zero group stores `scale = 1.0`.
///
/// # Panics
/// Panics if `weights.len() != n * k`, if `group_size` is not 16/32, if `k` is
/// odd, or if `group_size` does not divide `k` — all caller/shape contract
/// violations, surfaced early.
#[must_use]
pub fn pack_int4_f32(weights: &[f32], n: usize, k: usize, group_size: usize) -> QuantizedInt4 {
    let len = super::int8::checked_len("pack_int4_f32", n, k, "n*k");
    assert_eq!(
        weights.len(),
        len,
        "pack_int4_f32: weights len {} != n*k {}",
        weights.len(),
        len
    );
    assert!(
        VALID_GROUP_SIZES.contains(&group_size),
        "pack_int4_f32: group_size {group_size} must be 16 or 32"
    );
    assert!(k.is_multiple_of(2), "pack_int4_f32: k {k} must be even");
    assert!(
        k.is_multiple_of(group_size),
        "pack_int4_f32: group_size {group_size} must divide k {k} (reader requires exact division)"
    );

    let groups_per_row = k / group_size;
    let packed_len = super::int8::checked_len("pack_int4_f32", n, k / 2, "n*k/2");
    let scales_len =
        super::int8::checked_len("pack_int4_f32", n, groups_per_row, "n*(k/group_size)");
    let mut packed = vec![0u8; packed_len];
    let mut scales = Vec::with_capacity(scales_len);

    for o in 0..n {
        let row = &weights[o * k..(o + 1) * k];
        // First pass per group: compute the scale.
        let mut row_q = vec![0i8; k];
        for g in 0..groups_per_row {
            let grp = &row[g * group_size..(g + 1) * group_size];
            let max_abs = grp.iter().fold(0.0f32, |m, &w| m.max(w.abs()));
            let scale = if max_abs == 0.0 {
                1.0
            } else {
                max_abs / Q4_MAX as f32
            };
            scales.push(scale);
            for (i, &w) in grp.iter().enumerate() {
                // True division — documented contract; reciprocal-multiply diverges
                // by a ULP on non-power-of-two scales (audit rank 3).
                let r = round_ties_even_f32(w / scale);
                let qv = r.clamp(Q4_MIN as f32, Q4_MAX as f32) as i32 as i8;
                row_q[g * group_size + i] = qv;
            }
        }
        // Pack two nibbles per byte: low = even col, high = odd col.
        let row_base = o * (k / 2);
        for j in 0..(k / 2) {
            let lo = nibble_of(row_q[2 * j]);
            let hi = nibble_of(row_q[2 * j + 1]);
            packed[row_base + j] = lo | (hi << 4);
        }
    }

    QuantizedInt4 {
        packed,
        scales,
        n,
        k,
        group_size,
    }
}

// ── Importance-weighted clip-scale search (bd-50wo stage B) ─────────────────
//
// Plain RTN spends the whole 4-bit code range covering a group's single largest
// magnitude, so ONE outlier weight coarsens the step size for the other 15. The
// search keeps the format bit-identical (per-group f32 scale, signed nibbles in
// [-8, 7]) and only chooses a BETTER scale: it clips the group at a fraction of
// max|w| when doing so lowers the ACTIVATION-WEIGHTED reconstruction error
//
//     Σ_i  E[x_i²] · (w_i − dequant(quant(w_i)))²
//
// over the group's input channels `i` (`E[x_i²]` from the stage-A calibration;
// uniform weights when the tensor has no statistics). Weighting by E[x²] is what
// makes this calibration-AWARE rather than a plain MSE tweak: an input channel
// the model never excites contributes nothing to the output error no matter how
// badly its weight column is rounded, while a hot channel deserves the precision.
//
// The candidate list ALWAYS begins at 1.0 (exactly RTN) and a candidate replaces
// the incumbent only on a STRICT improvement, so the search is never worse than
// RTN on the stated objective — for any importance vector, not merely uniform
// (`search_never_loses_to_rtn_*`). All arithmetic is the pinned quantize rule
// (true division, ties-to-even, clamp), so the chosen scale reproduces exactly
// the codes the packer writes.

/// Number of clip-scale candidates evaluated per group/channel, spanning
/// [`CLIP_SEARCH_MIN_FRAC`, 1.0] inclusive.
pub const CLIP_SEARCH_STEPS: usize = 32;

/// Smallest clip fraction of `max|w|` the search will consider. Below ~0.65 the
/// clipping error on the group's largest weights dominates any step-size gain.
pub const CLIP_SEARCH_MIN_FRAC: f32 = 0.65;

/// The `i`-th clip fraction of the search grid: index 0 is exactly 1.0 (RTN, the
/// incumbent), indices 1.. sweep [`CLIP_SEARCH_MIN_FRAC`, 1.0] ascending.
///
/// # Panics
/// Panics if `i >= CLIP_SEARCH_STEPS + 1`.
#[must_use]
pub fn clip_fraction(i: usize) -> f32 {
    assert!(
        i <= CLIP_SEARCH_STEPS,
        "clip_fraction: index {i} exceeds the {CLIP_SEARCH_STEPS}-step grid"
    );
    if i == 0 {
        return 1.0;
    }
    let t = (i - 1) as f32 / (CLIP_SEARCH_STEPS - 1) as f32;
    CLIP_SEARCH_MIN_FRAC + t * (1.0 - CLIP_SEARCH_MIN_FRAC)
}

/// Quantize one contiguous group with `scale` under the pinned rule and return
/// the importance-weighted squared reconstruction error.
///
/// `importance[i]` weights column `i` of the group (`None` ⇒ uniform 1.0).
#[inline]
fn group_objective(group: &[f32], scale: f32, importance: Option<&[f64]>) -> f64 {
    let mut acc = 0.0f64;
    for (i, &w) in group.iter().enumerate() {
        let q = round_ties_even_f32(w / scale).clamp(Q4_MIN as f32, Q4_MAX as f32);
        let d = f64::from(w - q * scale);
        let weight = importance.map_or(1.0, |imp| imp[i]);
        acc += weight * d * d;
    }
    acc
}

/// The scale minimizing [`group_objective`] over the clip-fraction grid, and the
/// objective it achieves. Ties keep the EARLIER candidate, and index 0 is RTN,
/// so the result is never worse than RTN.
#[inline]
fn best_group_scale(group: &[f32], importance: Option<&[f64]>) -> (f32, f64) {
    let max_abs = group.iter().fold(0.0f32, |m, &w| m.max(w.abs()));
    if max_abs == 0.0 {
        // All-zero group: unit scale, zero error (the pinned no-NaN convention).
        return (1.0, 0.0);
    }
    let mut best_scale = 0.0f32;
    let mut best_err = f64::INFINITY;
    for i in 0..=CLIP_SEARCH_STEPS {
        let scale = clip_fraction(i) * max_abs / Q4_MAX as f32;
        if scale <= 0.0 || !scale.is_finite() {
            continue;
        }
        let err = group_objective(group, scale, importance);
        if err < best_err {
            best_err = err;
            best_scale = scale;
        }
    }
    (best_scale, best_err)
}

/// A calibration-aware int4 quantization: the packed tensor plus the total
/// importance-weighted reconstruction error it achieved (the quantity the AWQ
/// per-layer α selection minimizes — [`super::convert`] stage C).
#[derive(Debug, Clone, PartialEq)]
pub struct SearchedInt4 {
    /// The packed result — byte-layout-identical to [`pack_int4_f32`]'s.
    pub q: QuantizedInt4,
    /// `Σ_groups Σ_i E[x_i²]·(w_i − dequant(quant(w_i)))²` over the whole tensor.
    pub objective: f64,
}

/// [`pack_int4_f32`] with the importance-weighted clip-scale search in place of
/// the fixed `max|w|/7` scale.
///
/// `importance` is `E[x_i²]` for the tensor's `k` INPUT channels (stage-A
/// calibration); `None` means uniform weights. Output rows are independent, so
/// the row fan-out is bit-identical serial-vs-parallel (asserted by
/// `search_is_deterministic_and_order_independent`).
///
/// # Panics
/// As [`pack_int4_f32`], plus if `importance` is `Some` with a length other than
/// `k` (a mis-keyed calibration vector must never silently mis-weight columns).
#[must_use]
pub fn pack_int4_f32_searched(
    weights: &[f32],
    n: usize,
    k: usize,
    group_size: usize,
    importance: Option<&[f64]>,
) -> SearchedInt4 {
    let len = super::int8::checked_len("pack_int4_f32_searched", n, k, "n*k");
    assert_eq!(
        weights.len(),
        len,
        "pack_int4_f32_searched: weights len {} != n*k {}",
        weights.len(),
        len
    );
    assert!(
        VALID_GROUP_SIZES.contains(&group_size),
        "pack_int4_f32_searched: group_size {group_size} must be 16 or 32"
    );
    assert!(
        k.is_multiple_of(2),
        "pack_int4_f32_searched: k {k} must be even"
    );
    assert!(
        k.is_multiple_of(group_size),
        "pack_int4_f32_searched: group_size {group_size} must divide k {k} (reader requires \
         exact division)"
    );
    if let Some(imp) = importance {
        assert_eq!(
            imp.len(),
            k,
            "pack_int4_f32_searched: importance len {} != k {k}",
            imp.len()
        );
    }

    let groups_per_row = k / group_size;
    let packed_len = super::int8::checked_len("pack_int4_f32_searched", n, k / 2, "n*k/2");
    let scales_len = super::int8::checked_len(
        "pack_int4_f32_searched",
        n,
        groups_per_row,
        "n*(k/group_size)",
    );
    let mut packed = vec![0u8; packed_len];
    let mut scales = vec![0.0f32; scales_len];
    let mut errors = vec![0.0f64; n];

    {
        use rayon::prelude::*;
        packed
            .par_chunks_mut(k / 2)
            .zip(scales.par_chunks_mut(groups_per_row))
            .zip(errors.par_iter_mut())
            .zip(weights.par_chunks(k))
            .for_each(|(((packed_row, scale_row), err_slot), row)| {
                let mut row_q = vec![0i8; k];
                let mut row_err = 0.0f64;
                for (g, scale_slot) in scale_row.iter_mut().enumerate() {
                    let lo = g * group_size;
                    let grp = &row[lo..lo + group_size];
                    let imp = importance.map(|imp| &imp[lo..lo + group_size]);
                    let (scale, err) = best_group_scale(grp, imp);
                    *scale_slot = scale;
                    row_err += err;
                    for (i, &w) in grp.iter().enumerate() {
                        let r = round_ties_even_f32(w / scale);
                        row_q[lo + i] = r.clamp(Q4_MIN as f32, Q4_MAX as f32) as i32 as i8;
                    }
                }
                *err_slot = row_err;
                for (j, byte) in packed_row.iter_mut().enumerate() {
                    *byte = nibble_of(row_q[2 * j]) | (nibble_of(row_q[2 * j + 1]) << 4);
                }
            });
    }

    // Fixed left-to-right fold over the per-row errors: order-independent result
    // regardless of how rayon scheduled the rows.
    let objective = errors.iter().sum::<f64>();
    SearchedInt4 {
        q: QuantizedInt4 {
            packed,
            scales,
            n,
            k,
            group_size,
        },
        objective,
    }
}

/// Quantize a row-major `[n, k]` **bf16** weight matrix to group-quantized int4
/// (widen-then-quantize, exact bf16→f32 — see [`super::int8::quantize_int8_bf16`]).
///
/// # Panics
/// As [`pack_int4_f32`].
#[must_use]
pub fn pack_int4_bf16(weights: &[bf16], n: usize, k: usize, group_size: usize) -> QuantizedInt4 {
    let len = super::int8::checked_len("pack_int4_bf16", n, k, "n*k");
    assert_eq!(
        weights.len(),
        len,
        "pack_int4_bf16: weights len {} != n*k {}",
        weights.len(),
        len
    );
    let widened: Vec<f32> = weights.iter().map(|&w| w.to_f32()).collect();
    pack_int4_f32(&widened, n, k, group_size)
}

/// Unpack a [`QuantizedInt4`] to the exact int8 values the int8 GEMM consumes —
/// the in-register scheme (doctrine #4), here as an owned `Vec<i8>` of length
/// `n · k` in OUTPUT-CHANNEL-major `[n, k]` order.
///
/// Each byte yields two signed int4 values: low nibble (even column) then high
/// nibble (odd column), each sign-extended to `[-8, 7]` as i8. These are the
/// *unscaled* int4 codes promoted to i8 — exactly what `igemm_s8s8` multiplies;
/// the per-group scale is applied after the integer accumulation (as in the int8
/// path), never folded into the unpacked codes.
///
/// # Panics
/// Panics if `k` is odd or `packed.len() != n · k/2`.
#[must_use]
pub fn unpack_int4_to_i8(q: &QuantizedInt4) -> Vec<i8> {
    assert!(
        q.k.is_multiple_of(2),
        "unpack_int4_to_i8: k {} must be even",
        q.k
    );
    let packed_len = super::int8::checked_len("unpack_int4_to_i8", q.n, q.k / 2, "n*k/2");
    let out_len = super::int8::checked_len("unpack_int4_to_i8", q.n, q.k, "n*k");
    assert_eq!(
        q.packed.len(),
        packed_len,
        "unpack_int4_to_i8: packed len {} != n*k/2 {}",
        q.packed.len(),
        packed_len
    );
    let mut out = vec![0i8; out_len];
    for o in 0..q.n {
        let row_base = o * (q.k / 2);
        let out_base = o * q.k;
        for j in 0..(q.k / 2) {
            let byte = q.packed[row_base + j];
            out[out_base + 2 * j] = i8_of_nibble(byte & 0x0F);
            out[out_base + 2 * j + 1] = i8_of_nibble(byte >> 4);
        }
    }
    out
}

/// Dequantize a [`QuantizedInt4`] to f32 (`scale[row, group] · f32(q4)`).
///
/// The exact logical-value reconstruction; used by the round-trip / error tests
/// and any consumer measuring int4 quant error.
#[must_use]
pub fn dequantize_int4(q: &QuantizedInt4) -> Vec<f32> {
    let codes = unpack_int4_to_i8(q);
    let groups_per_row = q.groups_per_row();
    let scales_len =
        super::int8::checked_len("dequantize_int4", q.n, groups_per_row, "n*(k/group_size)");
    assert_eq!(
        q.scales.len(),
        scales_len,
        "dequantize_int4: scales len {} != n*(k/group_size) {}",
        q.scales.len(),
        scales_len
    );
    let out_len = super::int8::checked_len("dequantize_int4", q.n, q.k, "n*k");
    let mut out = Vec::with_capacity(out_len);
    for o in 0..q.n {
        for col in 0..q.k {
            let g = col / q.group_size;
            let scale = q.scales[o * groups_per_row + g];
            out.push(scale * f32::from(codes[o * q.k + col]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── nibble pack/sign-extend identity ────────────────────────────────────

    #[test]
    fn nibble_roundtrips_full_int4_range() {
        for v in Q4_MIN..=Q4_MAX {
            let nib = nibble_of(v as i8);
            assert!(nib <= 0x0F);
            assert_eq!(i8_of_nibble(nib), v as i8, "value {v} must round-trip");
        }
    }

    #[test]
    fn sign_extension_is_correct_for_all_16_nibbles() {
        // 0..=7 stay positive; 8..=15 map to -8..=-1.
        let expected: [i8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1];
        for (nib, &exp) in expected.iter().enumerate() {
            assert_eq!(i8_of_nibble(nib as u8), exp, "nibble {nib}");
        }
    }

    // ── pack layout (low nibble = even col) ─────────────────────────────────

    #[test]
    fn pack_places_even_col_in_low_nibble() {
        // One group of 16 with max_abs 7 -> scale 1.0, q == values.
        // values 0..16 clamped into [-8,7]: 0..7 then 8..15 clamp to 7.
        let mut w = vec![0.0f32; 16];
        w[0] = 1.0; // col 0 -> low nibble of byte 0
        w[1] = 2.0; // col 1 -> high nibble of byte 0
        w[2] = 7.0; // col 2 -> low nibble of byte 1
        w[3] = -8.0; // col 3 -> high nibble of byte 1
        let q = pack_int4_f32(&w, 1, 16, 16);
        // scale: max_abs over the group is 8.0 -> scale 8/7. Re-quantize to know
        // exact codes. Easier: just unpack and check ordering semantics hold.
        let codes = unpack_int4_to_i8(&q);
        assert_eq!(codes.len(), 16);
        // byte 0 low nibble is codes[0], high nibble codes[1]; check packing
        // order by reconstructing the first byte.
        let b0 = q.packed[0];
        assert_eq!(i8_of_nibble(b0 & 0x0F), codes[0]);
        assert_eq!(i8_of_nibble(b0 >> 4), codes[1]);
    }

    #[test]
    fn exact_int4_values_roundtrip_with_unit_scale() {
        // Group of 16 with max_abs 7 -> scale exactly 1.0, every value exact.
        // (No -8 here: -8 would make max_abs 8 and scale 8/7, breaking unit
        // scale; the full -8..=7 range is exercised by `unpack_is_exact_inverse_
        // of_pack_nibbles` and `sign_extension_is_correct_for_all_16_nibbles`.)
        let vals: Vec<f32> = vec![
            7.0, -7.0, 0.0, 1.0, -1.0, 3.0, -3.0, 6.0, -6.0, 2.0, -2.0, 4.0, -4.0, 5.0, -5.0, 7.0,
        ];
        let q = pack_int4_f32(&vals, 1, 16, 16);
        assert_eq!(q.scales.len(), 1);
        assert!((q.scales[0] - 1.0).abs() < 1e-9);
        let codes = unpack_int4_to_i8(&q);
        let exp: Vec<i8> = vals.iter().map(|&v| v as i8).collect();
        assert_eq!(codes, exp);
        // dequant is exact at unit scale.
        let d = dequantize_int4(&q);
        assert_eq!(d, vals);
    }

    #[test]
    fn packed_and_scale_counts_match_layout() {
        // n=2, k=32, group 16 -> 2 groups/row, 4 scales; 2*16=32 packed bytes.
        let w = vec![1.0f32; 2 * 32];
        let q = pack_int4_f32(&w, 2, 32, 16);
        assert_eq!(q.packed.len(), 2 * (32 / 2));
        assert_eq!(q.scales.len(), 2 * (32 / 16));
        assert_eq!(q.groups_per_row(), 2);
        // group_size 32 -> 1 group/row, 2 scales.
        let q32 = pack_int4_f32(&w, 2, 32, 32);
        assert_eq!(q32.scales.len(), 2);
    }

    #[test]
    fn per_group_scales_are_independent() {
        // Row of 32: group 0 (cols 0..16) max_abs 7 (scale 1), group 1
        // (cols 16..32) max_abs 14 (scale 2).
        let mut w = vec![0.0f32; 32];
        w[0] = 7.0;
        w[16] = 14.0;
        w[17] = -7.0;
        let q = pack_int4_f32(&w, 1, 32, 16);
        assert!((q.scales[0] - 1.0).abs() < 1e-9);
        assert!((q.scales[1] - 2.0).abs() < 1e-9);
        let codes = unpack_int4_to_i8(&q);
        assert_eq!(codes[0], 7); // 7/1
        assert_eq!(codes[16], 7); // 14/2
        assert_eq!(codes[17], -4); // -7/2 = -3.5 -> ties-even -> -4
    }

    #[test]
    fn all_zero_group_unit_scale_no_nan() {
        let w = vec![0.0f32; 16];
        let q = pack_int4_f32(&w, 1, 16, 16);
        assert_eq!(q.scales, vec![1.0]);
        assert!(q.scales[0].is_finite());
        assert_eq!(unpack_int4_to_i8(&q), vec![0i8; 16]);
    }

    #[test]
    fn clamps_to_int4_range_never_overflows() {
        // scale derived from the max; a value at the group max maps to 7, a huge
        // negative clamps to -8. Force it: group [16, -16, ...rest 0].
        let mut w = vec![0.0f32; 16];
        w[0] = 16.0;
        w[1] = -16.0;
        let q = pack_int4_f32(&w, 1, 16, 16);
        let codes = unpack_int4_to_i8(&q);
        for &c in &codes {
            assert!((Q4_MIN as i8..=Q4_MAX as i8).contains(&c));
        }
        assert_eq!(codes[0], 7); // 16/scale where scale=16/7 -> 7
        assert_eq!(codes[1], -7); // -16/(16/7) = -7
    }

    #[test]
    fn bf16_path_matches_f32_path() {
        let vals: Vec<f32> = (0..32).map(|i| (i as f32) - 16.0).collect();
        let bf: Vec<bf16> = vals.iter().map(|&v| bf16::from_f32(v)).collect();
        let qf = pack_int4_f32(&vals, 1, 32, 16);
        let qb = pack_int4_bf16(&bf, 1, 32, 16);
        assert_eq!(qf, qb);
    }

    #[test]
    fn unpack_is_exact_inverse_of_pack_nibbles() {
        // Adversarial: every nibble code present. Build q codes directly via a
        // pack with unit scale, then assert unpack reproduces them and the
        // packed bytes carry low-then-high.
        let vals: Vec<f32> = vec![
            -8.0, 7.0, -1.0, 1.0, -8.0, -8.0, 7.0, 7.0, 0.0, 0.0, -4.0, 4.0, -2.0, 2.0, -6.0, 6.0,
        ];
        let q = pack_int4_f32(&vals, 1, 16, 16); // max_abs 8 -> scale 8/7
        let codes = unpack_int4_to_i8(&q);
        // Reconstruct each byte from the codes and compare to the packed bytes.
        for j in 0..(16 / 2) {
            let lo = nibble_of(codes[2 * j]);
            let hi = nibble_of(codes[2 * j + 1]);
            assert_eq!(q.packed[j], lo | (hi << 4), "byte {j}");
        }
    }

    #[test]
    fn multi_row_packing_indexes_rows_correctly() {
        // n=2, k=16: row 0 all 7s (scale 1), row 1 all -8s (scale 8/7).
        let mut w = vec![0.0f32; 2 * 16];
        for elem in w[0..16].iter_mut() {
            *elem = 7.0;
        }
        for elem in w[16..32].iter_mut() {
            *elem = -8.0;
        }
        let q = pack_int4_f32(&w, 2, 16, 16);
        let codes = unpack_int4_to_i8(&q);
        assert!(codes[0..16].iter().all(|&c| c == 7));
        assert!(codes[16..32].iter().all(|&c| c == -7)); // -8/(8/7) = -7
    }

    // ── importance-weighted clip-scale search (bd-50wo stage B) ─────────────

    /// The whole-tensor weighted objective of an ARBITRARY packing — the
    /// independent yardstick the search claims to minimize. Deliberately
    /// recomputed from the dequantized values rather than reusing the search's
    /// own accumulator, so a bug in the accumulator cannot hide here.
    fn objective_of(q: &QuantizedInt4, w: &[f32], importance: Option<&[f64]>) -> f64 {
        let deq = dequantize_int4(q);
        let mut acc = 0.0f64;
        for o in 0..q.n {
            for c in 0..q.k {
                let d = f64::from(w[o * q.k + c] - deq[o * q.k + c]);
                acc += importance.map_or(1.0, |imp| imp[c]) * d * d;
            }
        }
        acc
    }

    /// Deterministic pseudo-random weights (no dev-dependency on a RNG crate).
    fn lcg_weights(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    #[test]
    fn clip_grid_starts_at_rtn_and_spans_the_documented_range() {
        assert_eq!(clip_fraction(0), 1.0, "candidate 0 must be exactly RTN");
        assert!((clip_fraction(1) - CLIP_SEARCH_MIN_FRAC).abs() < 1e-6);
        assert!((clip_fraction(CLIP_SEARCH_STEPS) - 1.0).abs() < 1e-6);
        for i in 1..CLIP_SEARCH_STEPS {
            assert!(
                clip_fraction(i) < clip_fraction(i + 1),
                "grid must be strictly ascending at {i}"
            );
            assert!((CLIP_SEARCH_MIN_FRAC..=1.0).contains(&clip_fraction(i)));
        }
    }

    #[test]
    fn search_never_loses_to_rtn_with_uniform_weights() {
        // Property (1): over many shapes/seeds the searched packing's UNIFORM
        // objective is <= the RTN packing's. Same format, same group sizes.
        for (n, k, gs, seed) in [
            (4usize, 64usize, 16usize, 7u64),
            (3, 32, 32, 11),
            (8, 128, 16, 23),
            (1, 16, 16, 99),
        ] {
            let w = lcg_weights(n * k, seed);
            let rtn = pack_int4_f32(&w, n, k, gs);
            let searched = pack_int4_f32_searched(&w, n, k, gs, None);
            let rtn_obj = objective_of(&rtn, &w, None);
            let new_obj = objective_of(&searched.q, &w, None);
            assert!(
                new_obj <= rtn_obj * (1.0 + 1e-12),
                "shape [{n},{k}] g{gs}: searched {new_obj} must not exceed RTN {rtn_obj}"
            );
            // The reported objective is the one actually achieved.
            assert!(
                (searched.objective - new_obj).abs() <= new_obj.abs() * 1e-9 + 1e-12,
                "reported objective {} != recomputed {new_obj}",
                searched.objective
            );
            // The packed layout is unchanged by the search.
            assert_eq!(searched.q.packed.len(), rtn.packed.len());
            assert_eq!(searched.q.scales.len(), rtn.scales.len());
            assert_eq!(searched.q.group_size, gs);
        }
    }

    #[test]
    fn search_never_loses_to_rtn_with_nonuniform_importance() {
        // The guarantee is not special to uniform weights: candidate 0 IS RTN
        // and replacement is strict, so it holds for any importance vector.
        let (n, k, gs) = (4usize, 64usize, 16usize);
        let w = lcg_weights(n * k, 4242);
        let importance: Vec<f64> = (0..k)
            .map(|i| ((i % 7) as f64).mul_add(3.0, 0.01))
            .collect();
        let rtn = pack_int4_f32(&w, n, k, gs);
        let searched = pack_int4_f32_searched(&w, n, k, gs, Some(&importance));
        assert!(
            objective_of(&searched.q, &w, Some(&importance))
                <= objective_of(&rtn, &w, Some(&importance)) * (1.0 + 1e-12)
        );
    }

    #[test]
    fn search_beats_min_max_on_an_outlier_channel_group() {
        // Property (2): a group whose LAST column carries a large weight that no
        // activation ever excites (importance ~0), while the other 15 columns are
        // hot and sit exactly on RTN's rounding midpoints — the worst case for a
        // step size sized off the outlier. Clipping the outlier away halves the
        // step and lands the hot columns near-exactly.
        let k = 16usize;
        let outlier = 2.2f32;
        let rtn_scale = outlier / Q4_MAX as f32;
        let mut w = vec![0.0f32; k];
        for (i, slot) in w.iter_mut().enumerate().take(k - 1) {
            // 3.5 * rtn_scale: exactly between codes 3 and 4 under RTN.
            *slot = 3.5 * rtn_scale * if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        w[k - 1] = outlier;
        let mut importance = vec![1.0f64; k];
        importance[k - 1] = 1.0e-6;

        let rtn = pack_int4_f32(&w, 1, k, 16);
        let searched = pack_int4_f32_searched(&w, 1, k, 16, Some(&importance));
        let rtn_obj = objective_of(&rtn, &w, Some(&importance));
        let new_obj = objective_of(&searched.q, &w, Some(&importance));
        assert!(
            new_obj < rtn_obj * 0.5,
            "the outlier case must improve substantially: searched {new_obj} vs RTN {rtn_obj}"
        );
        // It really did clip: the chosen scale is strictly smaller than min-max.
        assert!(
            searched.q.scales[0] < rtn.scales[0],
            "searched scale {} must clip below the min-max scale {}",
            searched.q.scales[0],
            rtn.scales[0]
        );
        // And the format contract still holds — every code in [-8, 7].
        for &c in &unpack_int4_to_i8(&searched.q) {
            assert!((Q4_MIN as i8..=Q4_MAX as i8).contains(&c));
        }
    }

    #[test]
    fn search_is_deterministic_and_order_independent() {
        // Property (3): repeated runs (rayon may schedule rows differently) are
        // byte-identical, and each output row depends only on its own weights.
        let (n, k, gs) = (6usize, 64usize, 16usize);
        let w = lcg_weights(n * k, 31337);
        let importance: Vec<f64> = (0..k).map(|i| 1.0 + (i as f64) * 0.01).collect();
        let first = pack_int4_f32_searched(&w, n, k, gs, Some(&importance));
        for _ in 0..4 {
            let again = pack_int4_f32_searched(&w, n, k, gs, Some(&importance));
            assert_eq!(first.q, again.q, "packing must be deterministic");
            assert_eq!(
                first.objective.to_bits(),
                again.objective.to_bits(),
                "objective must be deterministic"
            );
        }
        // Row independence: quantizing row 2 alone reproduces its slice.
        let row2 = &w[2 * k..3 * k];
        let alone = pack_int4_f32_searched(row2, 1, k, gs, Some(&importance));
        assert_eq!(
            &first.q.packed[2 * (k / 2)..3 * (k / 2)],
            &alone.q.packed[..]
        );
    }

    #[test]
    fn all_zero_group_still_gets_the_no_nan_unit_scale() {
        let searched = pack_int4_f32_searched(&[0.0f32; 32], 1, 32, 16, None);
        assert_eq!(searched.q.scales, vec![1.0, 1.0]);
        assert_eq!(searched.objective, 0.0);
        assert_eq!(unpack_int4_to_i8(&searched.q), vec![0i8; 32]);
    }

    #[test]
    #[should_panic(expected = "importance len")]
    fn search_rejects_a_mismatched_importance_vector() {
        let _ = pack_int4_f32_searched(&[0.0; 16], 1, 16, 16, Some(&[1.0; 8]));
    }

    // ── panics on bad inputs ────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "must be 16 or 32")]
    fn rejects_bad_group_size() {
        let _ = pack_int4_f32(&[0.0; 16], 1, 16, 8);
    }

    #[test]
    #[should_panic(expected = "must divide k")]
    fn rejects_group_size_not_dividing_k() {
        // k=16 with group 32 does not divide.
        let _ = pack_int4_f32(&[0.0; 16], 1, 16, 32);
    }

    #[test]
    #[should_panic(expected = "weights len")]
    fn rejects_shape_mismatch() {
        let _ = pack_int4_f32(&[0.0; 10], 1, 16, 16);
    }

    #[test]
    #[should_panic(expected = "pack_int4_f32: n*k overflow")]
    fn pack_int4_f32_rejects_shape_product_overflow() {
        let _ = pack_int4_f32(&[], usize::MAX, 16, 16);
    }

    #[test]
    #[should_panic(expected = "pack_int4_bf16: n*k overflow")]
    fn pack_int4_bf16_rejects_shape_product_overflow() {
        let _ = pack_int4_bf16(&[], usize::MAX, 16, 16);
    }

    #[test]
    #[should_panic(expected = "unpack_int4_to_i8: k 3 must be even")]
    fn unpack_rejects_odd_k() {
        let q = QuantizedInt4 {
            packed: vec![0],
            scales: vec![1.0],
            n: 1,
            k: 3,
            group_size: 1,
        };
        let _ = unpack_int4_to_i8(&q);
    }

    #[test]
    #[should_panic(expected = "unpack_int4_to_i8: n*k/2 overflow")]
    fn unpack_rejects_packed_shape_product_overflow() {
        let q = QuantizedInt4 {
            packed: Vec::new(),
            scales: Vec::new(),
            n: usize::MAX,
            k: 4,
            group_size: 2,
        };
        let _ = unpack_int4_to_i8(&q);
    }

    #[test]
    #[should_panic(expected = "QuantizedInt4::groups_per_row")]
    fn groups_per_row_rejects_zero_group_size() {
        let q = QuantizedInt4 {
            packed: Vec::new(),
            scales: Vec::new(),
            n: 1,
            k: 16,
            group_size: 0,
        };
        let _ = q.groups_per_row();
    }
}
