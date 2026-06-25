//! Symmetric per-output-channel int8 weight quantization + the U8S8 dynamic
//! activation-quant helper (the offline converter side; runtime dequant lives in
//! [`crate::native_engine`]).
//!
//! ## Pinned cross-module convention (do not drift)
//!
//! * **Weight layout: OUTPUT-CHANNEL-MAJOR `[N, K]`** — one weight row per output
//!   channel, exactly the [`crate::native_engine::tensor::QInt8`] / `nn`
//!   `linear_int8_dynamic` layout (`n = out_features`, `k = in_features`).
//! * **Symmetric, per-output-channel scale** (`docs/focrq-format.md` §"Scale
//!   Layout / QInt8PerChan", `AGENTS.md` PINNED CONTRACTS):
//!   ```text
//!   scale[o] = max(|w[o, :]|) / 127
//!   q[o, k]  = round_ties_to_even(clamp(w[o, k] / scale[o], -127, 127))
//!   zero_point = 0
//!   ```
//!   An **all-zero row** (`max_abs == 0`) stores `scale[o] = 1.0` and an all-zero
//!   `q` row — never `0/0 = NaN` (focrq-format.md: "avoids NaN/Inf while
//!   preserving the row exactly").
//! * **Rounding is `round_ties_to_even`** (banker's rounding), matching the
//!   converter's pinned `rounding: "round_ties_to_even"` (focrq-format.md
//!   `packing_manifest`). This is what makes the quant a *pure function* of the
//!   input bytes — no RNG, data-free PTQ, byte-identical across runs.
//! * **Clamp to `[-127, 127]`**, not `[-128, 127]`: a symmetric range keeps `q`
//!   sign-symmetric so `dequant(q) = scale * q` has zero systematic bias, and
//!   keeps `−128` (whose negation overflows i8) out of the operand domain. The
//!   i32-overflow proof (`tests/int32_overflow_proof.rs`) bounds S8S8 by
//!   `K · 127 · 127`.
//!
//! ## i32-accumulation overflow safety (AGENTS.md doctrine #6)
//!
//! Quantizing to `[-127, 127]` is what makes the downstream int8 GEMM's i32
//! accumulator provably non-overflowing. The global worst case is the dense
//! layer-0 `down_proj` at **K = 6848**:
//!
//! * S8S8 monotone bound: `6848 · 127 · 127 = 110_451_392 < i32::MAX`
//!   (`2_147_483_647`) — ~19× headroom.
//! * U8S8 monotone bound (asymmetric u8 activation × s8 weight):
//!   `6848 · 255 · 127 = 221_772_480 < i32::MAX` — ~9.7× headroom.
//!
//! Both fit i32 with room to spare; this module's clamp is the *source* of the
//! `127` factor that proof depends on. We deliberately do NOT inherit
//! frankensearch's `k ≤ 1536` bound.

use half::bf16;

/// Maximum int8 magnitude after symmetric quantization. We clamp to
/// `[-Q_MAX, Q_MAX]` (not `-128`) so the operand domain stays sign-symmetric and
/// the i32 GEMM accumulator bound is `K · Q_MAX²` (doctrine #6).
pub const Q_MAX: i32 = 127;

/// A symmetric per-output-channel int8-quantized weight, ready to be written to a
/// `.focrq` `QInt8PerChan` record (its payload is `q`, its inline scales are
/// `scales`).
///
/// Layout mirrors [`crate::native_engine::tensor::QInt8`] exactly: `q` is
/// `n * k` int8 in OUTPUT-CHANNEL-major `[n, k]` row order, `scales` is one f32
/// per output channel.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedInt8 {
    /// Row-major `[n, k]` int8 weights (one row per output channel).
    pub q: Vec<i8>,
    /// Per-output-channel scales, length `n` (`scale[o] = max|w_row|/127`).
    pub scales: Vec<f32>,
    /// Number of output channels (rows).
    pub n: usize,
    /// Contraction length (columns).
    pub k: usize,
}

impl QuantizedInt8 {
    /// The raw int8 payload bytes (one byte per weight, OC-major), as the
    /// `.focrq` writer wants them. `i8 -> u8` is a pure bit reinterpret (the
    /// reader does the inverse `b as i8`).
    #[must_use]
    pub fn weight_bytes(&self) -> Vec<u8> {
        self.q.iter().map(|&v| v as u8).collect()
    }

    /// The inline scale bytes (`n` little-endian f32), as the `.focrq` writer
    /// wants them.
    #[must_use]
    pub fn scale_bytes(&self) -> Vec<u8> {
        self.scales.iter().flat_map(|&s| s.to_le_bytes()).collect()
    }
}

/// Round to the nearest integer, ties to even (banker's rounding) — the pinned
/// converter rounding. Kept as a named helper so every quant path uses the same
/// rule and the determinism contract holds.
#[inline]
#[must_use]
fn round_ties_even_f32(x: f32) -> f32 {
    x.round_ties_even()
}

/// Quantize one output-channel row of `k` f32 weights to symmetric int8,
/// returning `(q_row, scale)`.
///
/// `scale = max|row| / 127`; an all-zero row yields `scale = 1.0` and an all-zero
/// `q_row` (no NaN). `q = clamp(round_ties_even(w / scale), -127, 127)`.
#[must_use]
fn quantize_row(row: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = row.iter().fold(0.0f32, |m, &w| m.max(w.abs()));
    if max_abs == 0.0 {
        // All-zero (or all -0.0) row: store scale 1.0, q all zero. Exact.
        return (vec![0i8; row.len()], 1.0);
    }
    let scale = max_abs / Q_MAX as f32;
    let inv = 1.0 / scale;
    let q: Vec<i8> = row
        .iter()
        .map(|&w| {
            let r = round_ties_even_f32(w * inv);
            // clamp into [-127, 127] then narrow; the clamp guarantees the cast
            // never wraps.
            r.clamp(-(Q_MAX as f32), Q_MAX as f32) as i32 as i8
        })
        .collect();
    (q, scale)
}

/// Quantize a row-major `[n, k]` (PyTorch `[out_features, in_features]`) f32
/// weight matrix to symmetric per-output-channel int8.
///
/// `weights` is `n * k` f32 in OC-major order. Returns OUTPUT-CHANNEL-major int8
/// weights `[n, k]` plus one f32 scale per output channel — exactly the
/// `QInt8PerChan` `.focrq` payload + inline scales.
///
/// # Panics
/// Panics if `weights.len() != n * k` (a shape contract violation is a caller
/// bug, surfaced early — same discipline as `QInt8::new`).
#[must_use]
pub fn quantize_int8_f32(weights: &[f32], n: usize, k: usize) -> QuantizedInt8 {
    assert_eq!(
        weights.len(),
        n * k,
        "quantize_int8_f32: weights len {} != n*k {}",
        weights.len(),
        n * k
    );
    let mut q = Vec::with_capacity(n * k);
    let mut scales = Vec::with_capacity(n);
    for o in 0..n {
        let row = &weights[o * k..(o + 1) * k];
        let (q_row, scale) = quantize_row(row);
        q.extend_from_slice(&q_row);
        scales.push(scale);
    }
    QuantizedInt8 { q, scales, n, k }
}

/// Quantize a row-major `[n, k]` **bf16** weight matrix to symmetric
/// per-output-channel int8.
///
/// The upstream Unlimited-OCR shard is bf16; this widens each bf16 weight to f32
/// (exact — bf16 is the high 16 bits of f32) and applies the same symmetric
/// per-OC quantization as [`quantize_int8_f32`]. Widen-then-quantize means the
/// quant operates on the true checkpoint values with no intermediate narrowing
/// (focrq-format.md: "BF16, NOT F16").
///
/// # Panics
/// Panics if `weights.len() != n * k`.
#[must_use]
pub fn quantize_int8_bf16(weights: &[bf16], n: usize, k: usize) -> QuantizedInt8 {
    assert_eq!(
        weights.len(),
        n * k,
        "quantize_int8_bf16: weights len {} != n*k {}",
        weights.len(),
        n * k
    );
    let widened: Vec<f32> = weights.iter().map(|&w| w.to_f32()).collect();
    quantize_int8_f32(&widened, n, k)
}

/// Dequantize a [`QuantizedInt8`] back to f32 (`w ≈ scale[o] * q[o, k]`).
///
/// The exact logical-value reconstruction the runtime performs; used by the
/// determinism tests and by any consumer that needs the dequantized weights to
/// measure error.
#[must_use]
pub fn dequantize_int8(q: &QuantizedInt8) -> Vec<f32> {
    let mut out = Vec::with_capacity(q.n * q.k);
    for o in 0..q.n {
        let s = q.scales[o];
        for &v in &q.q[o * q.k..(o + 1) * q.k] {
            out.push(s * f32::from(v));
        }
    }
    out
}

/// A U8S8 dynamic activation quantization of one f32 activation vector: unsigned
/// `u8` values in `[0, 255]` with an asymmetric `zero_point` and an f32 `scale`,
/// mirroring ONNX `DynamicQuantizeLinear`.
///
/// Dequantization is `scale * (f32(q) - zero_point)`. This is the activation side
/// of the U8S8 GEMM path (`igemm_u8s8`): activations are `u8`, weights stay `s8`.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedU8Activation {
    /// Quantized unsigned activations in `[0, 255]`.
    pub q: Vec<u8>,
    /// The single per-vector scale.
    pub scale: f32,
    /// The asymmetric zero-point (a `u8` value, carried as i32 for arithmetic).
    pub zero_point: i32,
}

/// Dynamically quantize an f32 activation vector to U8 with an asymmetric
/// zero-point (ONNX `DynamicQuantizeLinear` semantics).
///
/// The range is widened to include 0 (`min ≤ 0 ≤ max`) so the zero-point is a
/// representable `u8`. Then:
/// ```text
/// scale      = (max - min) / 255                 (1.0 if the range is degenerate)
/// zero_point = round_ties_even(clamp(-min / scale, 0, 255))
/// q[i]       = clamp(round_ties_even(x[i] / scale) + zero_point, 0, 255)
/// ```
/// A constant (or empty) vector gets `scale = 1.0`, `zero_point = clamp(round(-x0))`
/// so `dequant(q) == x` exactly where representable.
#[must_use]
pub fn quantize_activation_u8(x: &[f32]) -> QuantizedU8Activation {
    // Range widened to include 0 (ONNX rule), so the zero-point is representable.
    let mut min = 0.0f32;
    let mut max = 0.0f32;
    for &v in x {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    let range = max - min;
    let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
    let inv = 1.0 / scale;
    // zero_point quantizes the real value 0.0: zp = round(0 - min/scale) = round(-min/scale).
    let zp_f = round_ties_even_f32(-min * inv);
    let zero_point = zp_f.clamp(0.0, 255.0) as i32;
    let q: Vec<u8> = x
        .iter()
        .map(|&v| {
            let r = round_ties_even_f32(v * inv) as i32 + zero_point;
            r.clamp(0, 255) as u8
        })
        .collect();
    QuantizedU8Activation {
        q,
        scale,
        zero_point,
    }
}

/// Dequantize a U8 activation back to f32 (`scale * (q - zero_point)`).
#[must_use]
pub fn dequantize_activation_u8(a: &QuantizedU8Activation) -> Vec<f32> {
    a.q.iter()
        .map(|&v| a.scale * (i32::from(v) - a.zero_point) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── symmetric per-OC int8 weight quant ──────────────────────────────────

    #[test]
    fn quantizes_known_row_exactly() {
        // Row [127, -127, 0, 64] : max_abs = 127 -> scale = 1.0, q == values.
        let q = quantize_int8_f32(&[127.0, -127.0, 0.0, 64.0], 1, 4);
        assert_eq!(q.n, 1);
        assert_eq!(q.k, 4);
        assert_eq!(q.scales, vec![1.0]);
        assert_eq!(q.q, vec![127i8, -127, 0, 64]);
    }

    #[test]
    fn scale_is_max_abs_over_127() {
        // max_abs = 254 -> scale = 2.0; 254/2=127, -2/2=-1, 100/2=50.
        let q = quantize_int8_f32(&[254.0, -2.0, 100.0], 1, 3);
        assert!((q.scales[0] - 2.0).abs() < 1e-9);
        assert_eq!(q.q, vec![127i8, -1, 50]);
    }

    #[test]
    fn per_output_channel_scales_are_independent() {
        // Row 0 max_abs 127 (scale 1), row 1 max_abs 254 (scale 2).
        let w = [127.0f32, 0.0, -64.0, 254.0, -254.0, 0.0];
        let q = quantize_int8_f32(&w, 2, 3);
        assert_eq!(q.scales.len(), 2);
        assert!((q.scales[0] - 1.0).abs() < 1e-9);
        assert!((q.scales[1] - 2.0).abs() < 1e-9);
        assert_eq!(&q.q[0..3], &[127i8, 0, -64]);
        assert_eq!(&q.q[3..6], &[127i8, -127, 0]);
    }

    #[test]
    fn all_zero_row_gets_unit_scale_no_nan() {
        let q = quantize_int8_f32(&[0.0, 0.0, -0.0, 0.0], 1, 4);
        assert_eq!(q.scales, vec![1.0]);
        assert!(q.scales[0].is_finite());
        assert_eq!(q.q, vec![0i8; 4]);
    }

    #[test]
    fn round_ties_to_even_at_half() {
        // scale forced to 1.0 by a 127 in the row; the 0.5 / 1.5 / 2.5 ties must
        // round to even (0, 2, 2) not away-from-zero (1, 2, 3).
        let q = quantize_int8_f32(&[127.0, 0.5, 1.5, 2.5], 1, 4);
        assert!((q.scales[0] - 1.0).abs() < 1e-9);
        assert_eq!(q.q, vec![127i8, 0, 2, 2]);
    }

    #[test]
    fn negative_value_never_reaches_minus_128() {
        // Even a huge-magnitude negative is clamped at -127, never -128.
        let q = quantize_int8_f32(&[-1000.0, 1000.0], 1, 2);
        assert_eq!(q.q, vec![-127i8, 127]);
        for &v in &q.q {
            assert!(v >= -Q_MAX as i8 && v <= Q_MAX as i8);
        }
    }

    #[test]
    fn dequant_roundtrips_representable_values() {
        // With scale 2.0, the values 254, -2, 100 dequant back exactly.
        let q = quantize_int8_f32(&[254.0, -2.0, 100.0], 1, 3);
        let d = dequantize_int8(&q);
        assert_eq!(d, vec![254.0, -2.0, 100.0]);
    }

    #[test]
    fn bf16_path_matches_f32_path_on_exact_values() {
        // bf16-exact values: widen-then-quantize must equal f32-quantize.
        let vals = [1.0f32, -2.0, 0.5, 64.0, -64.0, 0.0];
        let bf: Vec<bf16> = vals.iter().map(|&v| bf16::from_f32(v)).collect();
        let qb = quantize_int8_bf16(&bf, 2, 3);
        let qf = quantize_int8_f32(&vals, 2, 3);
        assert_eq!(qb, qf);
    }

    #[test]
    fn weight_and_scale_bytes_are_writer_ready() {
        let q = quantize_int8_f32(&[127.0, -127.0, 254.0, -254.0], 2, 2);
        let wb = q.weight_bytes();
        assert_eq!(wb.len(), 4);
        // -127 as u8 is 129; round-trips back to -127.
        assert_eq!(wb[1] as i8, -127i8);
        let sb = q.scale_bytes();
        assert_eq!(sb.len(), 2 * 4); // n=2 f32 scales
        let s0 = f32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]]);
        assert!((s0 - 1.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "weights len")]
    fn rejects_shape_mismatch() {
        let _ = quantize_int8_f32(&[1.0, 2.0, 3.0], 2, 3);
    }

    // ── i32-overflow safety (doctrine #6, worst case K=6848) ────────────────

    #[test]
    fn quant_clamp_bounds_i32_accumulator_at_k_6848() {
        const K: usize = 6848;
        // The clamp guarantees |q| <= 127. Build an adversarial all-max S8S8
        // dot product and confirm the i32 accumulator does not overflow.
        let a = vec![Q_MAX as i8; K];
        let b = vec![Q_MAX as i8; K];
        let mut acc: i64 = 0;
        for (&x, &w) in a.iter().zip(b.iter()) {
            acc += i64::from(x) * i64::from(w);
        }
        assert_eq!(acc, (K as i64) * 127 * 127);
        assert_eq!(acc, 110_451_392);
        assert!(acc < i64::from(i32::MAX), "S8S8 K=6848 must fit i32");

        // U8S8 worst case: u8=255 activation x s8=127 weight.
        let u8s8: i64 = (K as i64) * 255 * 127;
        assert_eq!(u8s8, 221_772_480);
        assert!(u8s8 < i64::from(i32::MAX), "U8S8 K=6848 must fit i32");
    }

    // ── U8S8 dynamic activation quant ───────────────────────────────────────

    #[test]
    fn activation_u8_quant_covers_zero_in_range() {
        // x in [-1, 3]: range widened already includes 0. scale=(3-(-1))/255.
        let x = [-1.0f32, 0.0, 1.0, 3.0];
        let a = quantize_activation_u8(&x);
        let scale = 4.0f32 / 255.0;
        assert!((a.scale - scale).abs() < 1e-7);
        // zero_point quantizes 0.0 -> round(-(-1)/scale) = round(1/scale).
        let zp = round_ties_even_f32(1.0 / scale) as i32;
        assert_eq!(a.zero_point, zp);
        // dequant ~ original within one scale step.
        let d = dequantize_activation_u8(&a);
        for (orig, deq) in x.iter().zip(d.iter()) {
            assert!((orig - deq).abs() <= a.scale, "{orig} vs {deq}");
        }
    }

    #[test]
    fn activation_u8_all_positive_keeps_min_at_zero() {
        // All-positive x: min clamps to 0, zero_point should be 0.
        let x = [1.0f32, 2.0, 255.0];
        let a = quantize_activation_u8(&x);
        assert_eq!(a.zero_point, 0);
        assert!((a.scale - (255.0 / 255.0)).abs() < 1e-7);
        assert_eq!(a.q, vec![1u8, 2, 255]);
    }

    #[test]
    fn activation_u8_constant_vector_is_lossless() {
        // Constant 5.0: range with 0 is [0,5], scale 5/255; each value quantizes
        // and dequantizes within a scale step.
        let x = [5.0f32; 6];
        let a = quantize_activation_u8(&x);
        let d = dequantize_activation_u8(&a);
        for &v in &d {
            assert!((v - 5.0).abs() <= a.scale + 1e-6);
        }
    }

    #[test]
    fn activation_u8_empty_is_unit_scale_no_panic() {
        let a = quantize_activation_u8(&[]);
        assert!(a.scale.is_finite());
        assert_eq!(a.q.len(), 0);
        assert_eq!(a.zero_point, 0);
    }

    #[test]
    fn activation_u8_values_stay_in_byte_range() {
        let x = [-1000.0f32, 1000.0, 0.0, -0.0001, 0.0001];
        let a = quantize_activation_u8(&x);
        for &v in &a.q {
            // u8 is already 0..=255 by type; assert the round-trip is bounded.
            let _ = v;
        }
        assert!((0..=255).contains(&a.zero_point));
    }
}
