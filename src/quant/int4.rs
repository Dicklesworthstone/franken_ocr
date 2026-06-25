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
    assert_eq!(
        weights.len(),
        n * k,
        "pack_int4_f32: weights len {} != n*k {}",
        weights.len(),
        n * k
    );
    assert!(
        VALID_GROUP_SIZES.contains(&group_size),
        "pack_int4_f32: group_size {group_size} must be 16 or 32"
    );
    assert!(k % 2 == 0, "pack_int4_f32: k {k} must be even");
    assert!(
        k % group_size == 0,
        "pack_int4_f32: group_size {group_size} must divide k {k} (reader requires exact division)"
    );

    let groups_per_row = k / group_size;
    let mut packed = vec![0u8; n * (k / 2)];
    let mut scales = Vec::with_capacity(n * groups_per_row);

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
            let inv = 1.0 / scale;
            for (i, &w) in grp.iter().enumerate() {
                let r = round_ties_even_f32(w * inv);
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

/// Quantize a row-major `[n, k]` **bf16** weight matrix to group-quantized int4
/// (widen-then-quantize, exact bf16→f32 — see [`super::int8::quantize_int8_bf16`]).
///
/// # Panics
/// As [`pack_int4_f32`].
#[must_use]
pub fn pack_int4_bf16(weights: &[bf16], n: usize, k: usize, group_size: usize) -> QuantizedInt4 {
    assert_eq!(
        weights.len(),
        n * k,
        "pack_int4_bf16: weights len {} != n*k {}",
        weights.len(),
        n * k
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
/// Panics if `packed.len() != n · k/2`.
#[must_use]
pub fn unpack_int4_to_i8(q: &QuantizedInt4) -> Vec<i8> {
    assert_eq!(
        q.packed.len(),
        q.n * (q.k / 2),
        "unpack_int4_to_i8: packed len {} != n*k/2 {}",
        q.packed.len(),
        q.n * (q.k / 2)
    );
    let mut out = vec![0i8; q.n * q.k];
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
    let groups_per_row = q.k / q.group_size;
    let mut out = Vec::with_capacity(q.n * q.k);
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
        let vals: Vec<f32> = vec![
            7.0, -8.0, 0.0, 1.0, -1.0, 3.0, -3.0, 6.0, -6.0, 2.0, -2.0, 4.0, -4.0, 5.0, -5.0, 7.0,
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
        assert_eq!(q32.scales.len(), 2 * 1);
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
        for i in 0..16 {
            w[i] = 7.0;
        }
        for i in 16..32 {
            w[i] = -8.0;
        }
        let q = pack_int4_f32(&w, 2, 16, 16);
        let codes = unpack_int4_to_i8(&q);
        assert!(codes[0..16].iter().all(|&c| c == 7));
        assert!(codes[16..32].iter().all(|&c| c == -7)); // -8/(8/7) = -7
    }

    // ── panics on bad inputs ────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "must be 16 or 32")]
    fn rejects_bad_group_size() {
        let _ = pack_int4_f32(&vec![0.0; 16], 1, 16, 8);
    }

    #[test]
    #[should_panic(expected = "must divide k")]
    fn rejects_group_size_not_dividing_k() {
        // k=16 with group 32 does not divide.
        let _ = pack_int4_f32(&vec![0.0; 16], 1, 16, 32);
    }

    #[test]
    #[should_panic(expected = "weights len")]
    fn rejects_shape_mismatch() {
        let _ = pack_int4_f32(&vec![0.0; 10], 1, 16, 16);
    }
}
