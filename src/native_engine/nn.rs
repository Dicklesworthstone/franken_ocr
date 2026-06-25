//! The frankentorch facade — the single boundary between the hand-written
//! model and `ft-kernel-cpu` (PROPOSED_ARCHITECTURE.md §5).
//!
//! Every kernel call goes through here; nothing else in `native_engine/*`
//! touches frankentorch directly. This is where the op -> kernel map (plan §4.3)
//! is realized. Three categories:
//!
//! * **REUSE as-is** (§5.1): [`matmul`] (`matmul_tensor_contiguous_f32`),
//!   [`linear_int8_dynamic`] (`linear_int8_dynamic_f32` +
//!   `quantize_per_output_channel_i8`), [`conv2d`] (`conv2d_forward_f32`),
//!   [`sdpa`] (`sdpa_forward_f32`), [`rms_norm`] (`rms_norm_forward_f32`),
//!   [`layer_norm`] (`layer_norm_forward_f32`), [`softmax_rows`]
//!   (`softmax_dim_tensor_contiguous_f32`).
//! * **BUILD — model-specific glue** (§5.2): [`silu`], [`gelu`], and the
//!   CLIP-specific [`quick_gelu`] `x·σ(1.702x)` ([SPEC-049]). Per the
//!   frankensearch lesson (plan §9, doctrine #3) these are tight *scalar* loops
//!   that LLVM autovectorizes — NOT hand-wide SIMD.
//! * **BUILD — the perf wedge** (§5.3): the int8/int4 register-blocked GEMM
//!   tiers land later behind a runtime ISA dispatch; [`linear_int8_dynamic`] is
//!   the entrypoint they slot under (the dispatch already lives inside
//!   `ft-kernel-cpu`).
//!
//! The contiguous `[rows, cols]` [`Mat`] currency matches a
//! `TensorMeta::from_shape(vec![rows, cols], DType::F32, Device::Cpu)` exactly,
//! so building the kernel metas is a thin wrapper ([`meta_2d`]).

use ft_core::{DType, Device, TensorMeta};

use super::tensor::{Mat, QInt8};
use crate::error::{FocrError, FocrResult};

/// Build a contiguous 2-D f32 CPU `TensorMeta` for a `[rows, cols]` tensor.
///
/// `from_shape` fills in row-major strides and a zero storage offset — exactly
/// the layout of our row-major [`Mat`].
fn meta_2d(rows: usize, cols: usize) -> TensorMeta {
    TensorMeta::from_shape(vec![rows, cols], DType::F32, Device::Cpu)
}

/// Map a FrankenTorch `KernelError` into [`FocrError`].
///
/// Kernel failures here are almost always shape/contract violations from our
/// own callers (mismatched dimensions), so [`FocrError::Other`] carrying the
/// kernel's `Display` is the right bucket for the skeleton; a dedicated variant
/// can be added once call sites need to branch on it.
fn kernel_err(e: ft_kernel_cpu::KernelError) -> FocrError {
    FocrError::Other(anyhow::anyhow!("ft-kernel-cpu: {e}"))
}

// ── §5.1 REUSE ────────────────────────────────────────────────────────────

/// `[m,k] x [k,n] -> [m,n]`, delegating to FrankenTorch's parallel sgemm.
///
/// # Errors
/// Returns [`FocrError::Other`] if the inner dimensions disagree
/// (`a.cols != b.rows`) or the kernel rejects the shapes.
pub fn matmul(a: &Mat, b: &Mat) -> FocrResult<Mat> {
    if a.cols != b.rows {
        return Err(FocrError::Other(anyhow::anyhow!(
            "matmul inner dim mismatch: [{},{}] x [{},{}]",
            a.rows,
            a.cols,
            b.rows,
            b.cols
        )));
    }
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let lhs_meta = meta_2d(m, k);
    let rhs_meta = meta_2d(k, n);
    let data = ft_kernel_cpu::matmul_tensor_contiguous_f32(&a.data, &b.data, &lhs_meta, &rhs_meta)
        .map_err(kernel_err)?;
    Ok(Mat::from_vec(m, n, data))
}

/// Quantize a row-major `[out, in]` (PyTorch `[out_features, in_features]`)
/// weight matrix to symmetric per-output-channel int8, returning a [`QInt8`].
///
/// Thin wrapper over `ft_kernel_cpu::quantize_per_output_channel_i8`:
/// `scale[o] = max(|w[o,:]|)/127` (or `1.0` for an all-zero row), zero-point 0.
/// The result is ready to feed [`linear_int8_dynamic`].
///
/// # Panics
/// Panics if `w.len() != out * in_` (propagated from the kernel's assertion).
#[must_use]
pub fn quantize_int8(w: &[f32], out: usize, in_: usize) -> QInt8 {
    let (qw, scales) = ft_kernel_cpu::quantize_per_output_channel_i8(w, out, in_);
    QInt8::new(qw, scales, out, in_)
}

/// Int8 dynamic-quantized linear: `y[m,n] = dequant(quant_rows(x) @ w_i8^T) +
/// bias`, the crown decoder GEMM (plan §5.1).
///
/// `x` is a row-major `[m, k]` f32 activation [`Mat`]; `w` is a pre-quantized
/// [`QInt8`] in `[n, k]` layout (`n` output channels, `k` contraction).
/// Activations are dynamically quantized per row inside the kernel; the matmul
/// accumulates in i32 and dequantizes via `a_scale[s] * w_scale[o]`. Mirrors
/// ONNX `DynamicQuantizeLinear` + `MatMulInteger`.
///
/// # Errors
/// Returns [`FocrError::Other`] on a dimension mismatch (`x.cols != w.k`, or a
/// `bias` whose length isn't `w.n`).
pub fn linear_int8_dynamic(x: &Mat, w: &QInt8, bias: Option<&[f32]>) -> FocrResult<Mat> {
    if x.cols != w.k {
        return Err(FocrError::Other(anyhow::anyhow!(
            "linear_int8_dynamic: x.cols {} != w.k {}",
            x.cols,
            w.k
        )));
    }
    if let Some(b) = bias {
        if b.len() != w.n {
            return Err(FocrError::Other(anyhow::anyhow!(
                "linear_int8_dynamic: bias len {} != n {}",
                b.len(),
                w.n
            )));
        }
    }
    let (m, k, n) = (x.rows, x.cols, w.n);
    let data = ft_kernel_cpu::linear_int8_dynamic_f32(&x.data, m, k, &w.w, &w.scales, n, bias);
    Ok(Mat::from_vec(m, n, data))
}

/// 2-D convolution forward over a pre-padded NCHW input (`conv2d_forward_f32`).
///
/// `input` is the already-zero-padded tensor laid out `[batch, in_ch, ph, pw]`
/// (the caller pads to satisfy the SAM/CLIP patch-embed + neck shapes — plan
/// §5.1, [SPEC-041/046]); `weight` is `[out_ch, in_ch, kh, kw]` row-major;
/// `bias` is optional length `out_ch`. `(sh, sw)` are the strides; `(oh, ow)`
/// the output spatial dims. Returns the `[batch, out_ch, oh, ow]` feature map
/// as a flat `Vec<f32>` (the caller reshapes into the appropriate [`Mat`]).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn conv2d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    in_ch: usize,
    ph: usize,
    pw: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    sh: usize,
    sw: usize,
    out_ch: usize,
) -> Vec<f32> {
    ft_kernel_cpu::conv2d_forward_f32(
        input, weight, bias, batch, in_ch, ph, pw, kh, kw, oh, ow, sh, sw, out_ch,
    )
}

/// Scaled dot-product attention forward (`sdpa_forward_f32`).
///
/// `q/k/v` are flat, head-major: `q` is `[num_bh, seq_q, d_k]`, `k` is
/// `[num_bh, seq_k, d_k]`, `v` is `[num_bh, seq_k, d_v]`, where `num_bh =
/// batch * num_heads`. `scale` is the softmax temperature (`1/sqrt(d_k)` for the
/// SAM-global / CLIP / R-SWA-prefill basis); `causal` selects the lower-triangular
/// mask. Returns the `[num_bh, seq_q, d_v]` context flat.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_bh: usize,
    seq_q: usize,
    seq_k: usize,
    d_k: usize,
    d_v: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    ft_kernel_cpu::sdpa_forward_f32(q, k, v, num_bh, seq_q, seq_k, d_k, d_v, scale, causal)
}

/// Row-wise RMSNorm (`rms_norm_forward_f32`) — the decoder norm ([SPEC-071]).
///
/// Normalizes each row of the `[batch, norm_size]` matrix `x` by
/// `x * rsqrt(mean(x^2) + eps)` then scales by the optional per-feature
/// `weight` (length `cols`). `eps = rms_norm_eps = 1e-6` for the decoder.
/// Returns a fresh [`Mat`] (the kernel does not mutate in place).
///
/// # Errors
/// Returns [`FocrError::Other`] if `weight` is present but its length isn't
/// `x.cols`.
pub fn rms_norm(x: &Mat, weight: Option<&[f32]>, eps: f32) -> FocrResult<Mat> {
    if let Some(w) = weight {
        if w.len() != x.cols {
            return Err(FocrError::Other(anyhow::anyhow!(
                "rms_norm: weight len {} != cols {}",
                w.len(),
                x.cols
            )));
        }
    }
    let data = ft_kernel_cpu::rms_norm_forward_f32(&x.data, weight, x.rows, x.cols, eps);
    Ok(Mat::from_vec(x.rows, x.cols, data))
}

/// Row-wise LayerNorm (`layer_norm_forward_f32`) — vision norms, the
/// `LayerNorm2d` thin wrapper ([SPEC-046]).
///
/// `(weight, bias)` are optional per-feature affine params (length `cols`).
/// Returns a fresh [`Mat`].
///
/// # Errors
/// Returns [`FocrError::Other`] if a present `weight`/`bias` has the wrong
/// length.
pub fn layer_norm(
    x: &Mat,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) -> FocrResult<Mat> {
    if let Some(w) = weight {
        if w.len() != x.cols {
            return Err(FocrError::Other(anyhow::anyhow!(
                "layer_norm: weight len {} != cols {}",
                w.len(),
                x.cols
            )));
        }
    }
    if let Some(b) = bias {
        if b.len() != x.cols {
            return Err(FocrError::Other(anyhow::anyhow!(
                "layer_norm: bias len {} != cols {}",
                b.len(),
                x.cols
            )));
        }
    }
    let data = ft_kernel_cpu::layer_norm_forward_f32(&x.data, weight, bias, x.rows, x.cols, eps);
    Ok(Mat::from_vec(x.rows, x.cols, data))
}

/// In-place numerically-stable per-row softmax over the last dim
/// (`softmax_dim_tensor_contiguous_f32`, dim = 1).
///
/// Each row is softmaxed independently (max-subtract for overflow safety). The
/// kernel returns a fresh buffer; we copy it back into `x` so call sites can
/// keep operating in place over their `Mat`.
///
/// # Errors
/// Returns [`FocrError::Other`] if the kernel rejects the shape.
pub fn softmax_rows(x: &mut Mat) -> FocrResult<()> {
    if x.cols == 0 || x.rows == 0 {
        return Ok(());
    }
    let meta = meta_2d(x.rows, x.cols);
    let out =
        ft_kernel_cpu::softmax_dim_tensor_contiguous_f32(&x.data, &meta, 1).map_err(kernel_err)?;
    x.data.copy_from_slice(&out);
    Ok(())
}

// ── §5.2 BUILD: elementwise activation glue (tight scalar, LLVM-autovec) ────

/// In-place SiLU / swish `x · σ(x)` over every element.
///
/// The decoder MLP / MoE-expert gate activation ([SPEC-075/076], `hidden_act =
/// "silu"`). Tight scalar loop — LLVM autovectorizes; hand-wide SIMD measured
/// slower (plan §9, doctrine #3).
pub fn silu(x: &mut Mat) {
    for v in &mut x.data {
        let s = *v;
        *v = s / (1.0 + (-s).exp());
    }
}

/// In-place exact (erf-based) GELU over every element — the SAM MLP activation
/// ([SPEC-049] distinguishes SAM GELU from CLIP `quick_gelu` and the LLM SiLU).
///
/// `0.5 * x * (1 + erf(x / sqrt(2)))`. Uses [`libm::erff`] via `f32::erf`?
/// Rust has no stable `f32::erf`, so we compute through the tanh-free exact form
/// available from the standard library's `f64` math by widening per element —
/// kept scalar so LLVM can vectorize the surrounding arithmetic. (A poly-erf
/// fast path is a later, parity-gated lever; plan §9.)
pub fn gelu(x: &mut Mat) {
    const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    for v in &mut x.data {
        let xf = f64::from(*v);
        *v = (0.5 * xf * (1.0 + erf_f64(xf * INV_SQRT2))) as f32;
    }
}

/// In-place CLIP `quick_gelu` `x · σ(1.702 x)` ([SPEC-049]).
///
/// The CLIP-L vision MLP activation (`NoTPFeedForward`,
/// `fc2(quick_gelu(fc1(x)))`). Distinct from the SAM erf-GELU and the decoder
/// SiLU — a frequent source of silent vision divergence, so it is its own named
/// op. Tight scalar loop (plan §5.2, bd-1gv.9).
pub fn quick_gelu(x: &mut Mat) {
    for v in &mut x.data {
        let s = *v;
        *v = s / (1.0 + (-1.702 * s).exp());
    }
}

/// `quick_gelu` on a single scalar — the hand-verifiable kernel of [`quick_gelu`].
#[inline]
#[must_use]
pub fn quick_gelu_scalar(x: f32) -> f32 {
    x / (1.0 + (-1.702 * x).exp())
}

/// Error function on f64 — an Abramowitz & Stegun 7.1.26 rational-poly
/// approximation (max abs error ~1.5e-7), used by the exact-form [`gelu`].
///
/// This stays well inside the f32 activation tolerance and avoids a libm
/// dependency for the skeleton; a measured-parity swap is a later lever.
#[inline]
fn erf_f64(x: f64) -> f64 {
    // Constants for A&S 7.1.26.
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-ax * ax).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed 2x3 @ 3x2 product:
    ///   A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
    ///   AB = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    ///      = [[58, 64], [139, 154]]
    #[test]
    fn matmul_matches_hand_computed() {
        let a = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Mat::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), (2, 2));
        assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_rejects_inner_mismatch() {
        let a = Mat::zeros(2, 3);
        let b = Mat::zeros(4, 2); // 3 != 4
        assert!(matmul(&a, &b).is_err());
    }

    /// RMSNorm of a single row [3,4] with no weight, eps=0:
    ///   mean(x^2) = (9+16)/2 = 12.5; rstd = 1/sqrt(12.5) = 0.282842712...
    ///   out = [3*rstd, 4*rstd] = [0.848528137, 1.131370849]
    #[test]
    fn rms_norm_matches_hand_computed() {
        let x = Mat::from_vec(1, 2, vec![3.0, 4.0]);
        let y = rms_norm(&x, None, 0.0).unwrap();
        let rstd = 1.0f32 / 12.5f32.sqrt();
        assert!((y.data[0] - 3.0 * rstd).abs() < 1e-6);
        assert!((y.data[1] - 4.0 * rstd).abs() < 1e-6);
    }

    #[test]
    fn rms_norm_applies_weight() {
        let x = Mat::from_vec(1, 2, vec![3.0, 4.0]);
        let y = rms_norm(&x, Some(&[2.0, 0.5]), 0.0).unwrap();
        let rstd = 1.0f32 / 12.5f32.sqrt();
        assert!((y.data[0] - 3.0 * rstd * 2.0).abs() < 1e-6);
        assert!((y.data[1] - 4.0 * rstd * 0.5).abs() < 1e-6);
    }

    /// quick_gelu(0) = 0; quick_gelu(1) = 1/(1+e^-1.702) = 0.845855...;
    /// quick_gelu(-1) = -1/(1+e^1.702) = -0.154144...
    #[test]
    fn quick_gelu_matches_hand_computed() {
        assert!((quick_gelu_scalar(0.0)).abs() < 1e-7);
        let q1 = 1.0f32 / (1.0 + (-1.702f32).exp());
        assert!((quick_gelu_scalar(1.0) - q1).abs() < 1e-6);
        assert!((quick_gelu_scalar(1.0) - 0.845_855).abs() < 1e-4);
        let qm1 = -1.0f32 / (1.0 + (1.702f32).exp());
        assert!((quick_gelu_scalar(-1.0) - qm1).abs() < 1e-6);
        assert!((quick_gelu_scalar(-1.0) - (-0.154_144)).abs() < 1e-4);
    }

    #[test]
    fn quick_gelu_mat_matches_scalar() {
        let mut m = Mat::from_vec(1, 3, vec![-1.0, 0.0, 1.0]);
        quick_gelu(&mut m);
        assert!((m.data[0] - quick_gelu_scalar(-1.0)).abs() < 1e-7);
        assert!((m.data[1] - quick_gelu_scalar(0.0)).abs() < 1e-7);
        assert!((m.data[2] - quick_gelu_scalar(1.0)).abs() < 1e-7);
    }

    #[test]
    fn silu_matches_hand_computed() {
        // silu(0) = 0; silu(1) = 1*sigmoid(1) = 0.7310586; silu(-1) = -0.2689414
        let mut m = Mat::from_vec(1, 3, vec![0.0, 1.0, -1.0]);
        silu(&mut m);
        assert!(m.data[0].abs() < 1e-7);
        assert!((m.data[1] - 0.731_058_6).abs() < 1e-5);
        assert!((m.data[2] - (-0.268_941_4)).abs() < 1e-5);
    }

    #[test]
    fn gelu_matches_hand_computed() {
        // gelu(0)=0; gelu(1)=0.8413447; gelu(-1)=-0.1586553 (exact erf form)
        let mut m = Mat::from_vec(1, 3, vec![0.0, 1.0, -1.0]);
        gelu(&mut m);
        assert!(m.data[0].abs() < 1e-6);
        assert!((m.data[1] - 0.841_344_7).abs() < 1e-4);
        assert!((m.data[2] - (-0.158_655_3)).abs() < 1e-4);
    }

    #[test]
    fn softmax_rows_sums_to_one() {
        let mut m = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0]);
        softmax_rows(&mut m).unwrap();
        let r0: f32 = m.row(0).iter().sum();
        let r1: f32 = m.row(1).iter().sum();
        assert!((r0 - 1.0).abs() < 1e-6);
        assert!((r1 - 1.0).abs() < 1e-6);
        // uniform row -> 1/3 each
        for &v in m.row(1) {
            assert!((v - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    /// int8 dynamic linear should approximate the f32 product. With small
    /// integer-valued weights/activations the symmetric quant is near-lossless.
    /// x=[[1,2,3]] (1x3), W=[[1,0,1],[0,1,0]] (n=2,k=3) => y=[1+3, 2]=[4,2].
    #[test]
    fn linear_int8_dynamic_approximates_f32() {
        let x = Mat::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let w = quantize_int8(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 2, 3);
        let y = linear_int8_dynamic(&x, &w, None).unwrap();
        assert_eq!(y.shape(), (1, 2));
        assert!((y.data[0] - 4.0).abs() < 0.1);
        assert!((y.data[1] - 2.0).abs() < 0.1);
    }
}
