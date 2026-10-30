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
use crate::quant::recipe::switch_on;

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

fn checked_mat_len(context: &str, x: &Mat) -> FocrResult<usize> {
    let expected = x.rows.checked_mul(x.cols).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "{context}: rows*cols overflow ({} * {})",
            x.rows,
            x.cols
        ))
    })?;
    if x.data.len() != expected {
        return Err(FocrError::Other(anyhow::anyhow!(
            "{context}: data len {} != rows*cols {} for shape [{}, {}]",
            x.data.len(),
            expected,
            x.rows,
            x.cols
        )));
    }
    Ok(expected)
}

fn checked_qint8_len(context: &str, w: &QInt8) -> FocrResult<usize> {
    let expected = w.n.checked_mul(w.k).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "{context}: n*k overflow ({} * {})",
            w.n,
            w.k
        ))
    })?;
    if w.w.len() != expected {
        return Err(FocrError::Other(anyhow::anyhow!(
            "{context}: weight len {} != n*k {} for shape [{}, {}]",
            w.w.len(),
            expected,
            w.n,
            w.k
        )));
    }
    if w.scales.len() != w.n {
        return Err(FocrError::Other(anyhow::anyhow!(
            "{context}: scales len {} != n {}",
            w.scales.len(),
            w.n
        )));
    }
    Ok(expected)
}

// ── §5.1 REUSE ────────────────────────────────────────────────────────────

/// `[m,k] x [k,n] -> [m,n]`, delegating to FrankenTorch's parallel sgemm.
///
/// # Errors
/// Returns [`FocrError::Other`] if the inner dimensions disagree
/// (`a.cols != b.rows`) or the kernel rejects the shapes.
pub fn matmul(a: &Mat, b: &Mat) -> FocrResult<Mat> {
    checked_mat_len("matmul lhs", a)?;
    checked_mat_len("matmul rhs", b)?;
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
    checked_mat_len("linear_int8_dynamic x", x)?;
    checked_qint8_len("linear_int8_dynamic weight", w)?;
    if x.cols != w.k {
        return Err(FocrError::Other(anyhow::anyhow!(
            "linear_int8_dynamic: x.cols {} != w.k {}",
            x.cols,
            w.k
        )));
    }
    if let Some(b) = bias
        && b.len() != w.n
    {
        return Err(FocrError::Other(anyhow::anyhow!(
            "linear_int8_dynamic: bias len {} != n {}",
            b.len(),
            w.n
        )));
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

/// TF-'SAME' padding amounts for ONE dimension (timm `padding.py`, E3/A8 —
/// tromr-spec §2a): `total = max((ceil(i/s)−1)·s + k − i, 0)`, split
/// `begin = total/2`, `end = total − begin` (right/bottom gets the extra —
/// the asymmetry that plain symmetric padding cannot express; stem 7×7 s2 at
/// H=128 pads 2 top / 3 bottom).
#[must_use]
pub fn tf_same_pad_amounts(i: usize, k: usize, s: usize) -> (usize, usize) {
    let total = (i.div_ceil(s) - 1) * s + k;
    let total = total.saturating_sub(i);
    (total / 2, total - total / 2)
}

/// Pad a flat NCHW tensor per TF-'SAME' for a `(kh, kw, sh, sw)` op, filling
/// with `fill` (`0.0` before [`conv2d`]; `f32::NEG_INFINITY` before
/// [`max_pool2d`] — timm `MaxPool2dSame` pads with −∞ so border maxima are
/// never fabricated from zeros). Returns `(padded, ph, pw)`; the SAME output
/// dims are `ceil(h/sh) × ceil(w/sw)`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn tf_same_pad(
    input: &[f32],
    batch: usize,
    ch: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    fill: f32,
) -> (Vec<f32>, usize, usize) {
    let (top, bottom) = tf_same_pad_amounts(h, kh, sh);
    let (left, right) = tf_same_pad_amounts(w, kw, sw);
    let (ph, pw) = (h + top + bottom, w + left + right);
    let mut out = vec![fill; batch * ch * ph * pw];
    for bc in 0..batch * ch {
        let src = &input[bc * h * w..(bc + 1) * h * w];
        let dst = &mut out[bc * ph * pw..(bc + 1) * ph * pw];
        for row in 0..h {
            let d = (row + top) * pw + left;
            dst[d..d + w].copy_from_slice(&src[row * w..(row + 1) * w]);
        }
    }
    (out, ph, pw)
}

/// Max-pool `k×k` stride `s` over a PRE-PADDED flat NCHW tensor (pair with
/// [`tf_same_pad`] using a `NEG_INFINITY` fill for the timm `MaxPool2dSame`
/// semantics — tromr-spec §2a stem `MaxPool2dSame(k3, s2)`). `(oh, ow)` are
/// the output spatial dims. Tight scalar loops (doctrine #3).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn max_pool2d(
    input: &[f32],
    batch: usize,
    ch: usize,
    ph: usize,
    pw: usize,
    k: usize,
    s: usize,
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    let mut out = vec![f32::NEG_INFINITY; batch * ch * oh * ow];
    for bc in 0..batch * ch {
        let src = &input[bc * ph * pw..(bc + 1) * ph * pw];
        let dst = &mut out[bc * oh * ow..(bc + 1) * oh * ow];
        for oy in 0..oh {
            for ox in 0..ow {
                let mut m = f32::NEG_INFINITY;
                for ky in 0..k {
                    let row = oy * s + ky;
                    for kx in 0..k {
                        m = m.max(src[row * pw + ox * s + kx]);
                    }
                }
                dst[oy * ow + ox] = m;
            }
        }
    }
    out
}

/// In-place GroupNorm (+ optional fused ReLU) over a flat NCHW tensor — the
/// TrOMR/pix2tex ResNetV2 backbone norm (E3/A8; tromr-spec §2a:
/// `GroupNormAct(num_groups=32, eps=1e-5)`, groups of 2 even at 64 channels;
/// the `norm3`/downsample instances skip the activation, hence `fuse_relu`).
///
/// Torch `nn.GroupNorm` semantics: per `(batch, group)`, mean and POPULATION
/// variance over the group's `(channels/groups) × spatial` elements, then the
/// per-CHANNEL affine `y = (x − μ)/√(σ² + eps) · γ_c + β_c`. First GroupNorm
/// in the repo (Baidu/GOT/SmolVLM2/OneChart are LayerNorm/RMSNorm only).
/// Tight scalar loops — LLVM autovectorizes (doctrine #3).
///
/// # Errors
/// Shape violations: `channels % groups != 0`, a length mismatch on `x`
/// (`batch·channels·spatial`) or on `weight`/`bias` (`channels`).
#[allow(clippy::too_many_arguments)]
pub fn group_norm(
    x: &mut [f32],
    batch: usize,
    channels: usize,
    spatial: usize,
    groups: usize,
    eps: f32,
    weight: &[f32],
    bias: &[f32],
    fuse_relu: bool,
) -> FocrResult<()> {
    if groups == 0 || !channels.is_multiple_of(groups) {
        return Err(FocrError::Other(anyhow::anyhow!(
            "group_norm: channels {channels} not divisible by groups {groups}"
        )));
    }
    if x.len() != batch * channels * spatial {
        return Err(FocrError::Other(anyhow::anyhow!(
            "group_norm: x len {} != batch {batch} * channels {channels} * spatial {spatial}",
            x.len()
        )));
    }
    if weight.len() != channels || bias.len() != channels {
        return Err(FocrError::Other(anyhow::anyhow!(
            "group_norm: weight/bias len {}/{} != channels {channels}",
            weight.len(),
            bias.len()
        )));
    }
    let cpg = channels / groups;
    let group_len = cpg * spatial;
    for b in 0..batch {
        for g in 0..groups {
            let start = (b * channels + g * cpg) * spatial;
            let slice = &mut x[start..start + group_len];
            // Population mean/variance in f64 accumulation (spatial × cpg can
            // reach 64·80·8 = 40960 elements — f32 running sums drift).
            let mut sum = 0.0f64;
            for &v in slice.iter() {
                sum += f64::from(v);
            }
            let mean = sum / group_len as f64;
            let mut var = 0.0f64;
            for &v in slice.iter() {
                let d = f64::from(v) - mean;
                var += d * d;
            }
            let var = var / group_len as f64;
            let inv = 1.0 / (var + f64::from(eps)).sqrt();
            let (mean, inv) = (mean as f32, inv as f32);
            for c in 0..cpg {
                let (gamma, beta) = (weight[g * cpg + c], bias[g * cpg + c]);
                for v in &mut slice[c * spatial..(c + 1) * spatial] {
                    let y = (*v - mean) * inv * gamma + beta;
                    *v = if fuse_relu { y.max(0.0) } else { y };
                }
            }
        }
    }
    Ok(())
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
    checked_mat_len("rms_norm x", x)?;
    if let Some(w) = weight
        && w.len() != x.cols
    {
        return Err(FocrError::Other(anyhow::anyhow!(
            "rms_norm: weight len {} != cols {}",
            w.len(),
            x.cols
        )));
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
    checked_mat_len("layer_norm x", x)?;
    if let Some(w) = weight
        && w.len() != x.cols
    {
        return Err(FocrError::Other(anyhow::anyhow!(
            "layer_norm: weight len {} != cols {}",
            w.len(),
            x.cols
        )));
    }
    if let Some(b) = bias
        && b.len() != x.cols
    {
        return Err(FocrError::Other(anyhow::anyhow!(
            "layer_norm: bias len {} != cols {}",
            b.len(),
            x.cols
        )));
    }
    let data = ft_kernel_cpu::layer_norm_forward_f32(&x.data, weight, bias, x.rows, x.cols, eps);
    Ok(Mat::from_vec(x.rows, x.cols, data))
}

/// Truthy diagnostic fallback that preserves the pre-optimization payload copy
/// for same-binary A/B measurements. The ownership-transfer path remains the
/// default.
const SOFTMAX_COPY_ENV: &str = "FOCR_SOFTMAX_COPY";

fn softmax_copy_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| switch_on(SOFTMAX_COPY_ENV))
}

fn install_softmax_output(x: &mut Mat, out: Vec<f32>, copy: bool) {
    if copy {
        x.data.copy_from_slice(&out);
    } else {
        x.data = out;
    }
}

/// In-place numerically-stable per-row softmax over the last dim
/// (`softmax_dim_tensor_contiguous_f32`, dim = 1).
///
/// Each row is softmaxed independently (max-subtract for overflow safety). The
/// kernel returns a fresh final buffer; transfer that allocation into `x` so
/// call sites keep their in-place API without copying the full payload. Truthy
/// [`SOFTMAX_COPY_ENV`] preserves the former payload copy for diagnostic A/Bs.
///
/// # Errors
/// Returns [`FocrError::Other`] if the kernel rejects the shape.
pub fn softmax_rows(x: &mut Mat) -> FocrResult<()> {
    checked_mat_len("softmax_rows x", x)?;
    if x.cols == 0 || x.rows == 0 {
        return Ok(());
    }
    let meta = meta_2d(x.rows, x.cols);
    let out =
        ft_kernel_cpu::softmax_dim_tensor_contiguous_f32(&x.data, &meta, 1).map_err(kernel_err)?;
    if out.len() != x.data.len() {
        return Err(FocrError::Other(anyhow::anyhow!(
            "softmax_rows: kernel output len {} != input len {}",
            out.len(),
            x.data.len()
        )));
    }
    install_softmax_output(x, out, softmax_copy_enabled());
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

/// In-place tanh-approximation GELU (`gelu_pytorch_tanh`) — the SigLIP MLP
/// activation (SmolVLM2 C3, OQ-1; `docs/zoo/smolvlm2-spec.md` §2):
///
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
///
/// Distinct from the SAM erf-[`gelu`] and the CLIP [`quick_gelu`] — a third
/// named variant so a wrong-GELU vision divergence stays loud. f32 math end to
/// end, matching torch's f32 CPU `gelu(approximate="tanh")` kernel; the SigLIP
/// seam parity gate (vision_siglip.rs) measures the residual ULP drift against
/// the oracle floor.
pub fn gelu_tanh(x: &mut Mat) {
    for v in &mut x.data {
        *v = gelu_tanh_scalar(*v);
    }
}

/// In-place ReLU — the OPT MLP activation (OneChart D4; census §4
/// `activation_function: relu`, the plain non-gated `fc1`/`fc2` pair).
/// Distinct from SiLU/GELU so a wrong-activation divergence stays loud.
pub fn relu(x: &mut Mat) {
    for v in &mut x.data {
        *v = v.max(0.0);
    }
}

/// `gelu_tanh` on a single scalar — the hand-verifiable kernel of
/// [`gelu_tanh`].
#[inline]
#[must_use]
pub fn gelu_tanh_scalar(x: f32) -> f32 {
    // sqrt(2/pi) as f32 — the constant torch's tanh-GELU uses (M_SQRT2/M_2_SQRTPI
    // composition lands on the same f32 value).
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    let inner = SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
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

// ── §5.2 BUILD: antialiased bicubic resize (position-table glue) ───────────

/// Antialiased bicubic resize of a channel-last `[ih, iw, dim]` grid to
/// `[oh, ow, dim]`: torch's CPU `F.interpolate(mode="bicubic", antialias=True,
/// align_corners=False)` (`_upsample_bicubic2d_aa`). Both DeepEncoder towers use
/// it to fit their learned absolute position tables to the runtime patch grid
/// (`get_abs_pos_sam`, `get_abs_pos`). A 640 view (v1's multi-page squash, a
/// gundam tile) needs SAM 64→40 and CLIP 16→10.
///
/// Separable, as in torch: the width pass runs first, then the height pass, and
/// a pass is skipped when its size is unchanged. Each output sums its
/// [`bicubic_aa_taps`] in tap order with fused multiply-adds, which is what
/// torch's compiled `output += t * wts` contracts to. Measured bit-identical to
/// torch 2.13 on the real 64→40 SAM and 16→10 CLIP tables (bd-p8hu).
pub fn resize_bicubic_aa(
    src: &[f32],
    ih: usize,
    iw: usize,
    dim: usize,
    oh: usize,
    ow: usize,
) -> FocrResult<Vec<f32>> {
    const CONTEXT: &str = "resize_bicubic_aa";
    if ih == 0 || iw == 0 || dim == 0 || oh == 0 || ow == 0 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "{CONTEXT}: source ({ih},{iw}), target ({oh},{ow}) and dim {dim} must be non-zero"
        )));
    }
    let src_len = checked_product(CONTEXT, &[ih, iw, dim])?;
    if src.len() != src_len {
        return Err(FocrError::Other(anyhow::anyhow!(
            "{CONTEXT}: source len {} != ih*iw*dim {ih}*{iw}*{dim} ({src_len})",
            src.len()
        )));
    }
    let row_len = checked_product(CONTEXT, &[ow, dim])?;
    checked_product(CONTEXT, &[ih, row_len])?;
    checked_product(CONTEXT, &[oh, row_len])?;

    let width_resized;
    let rows = if ow == iw {
        src
    } else {
        width_resized = resample_axis_bicubic_aa(src, ih, iw, dim, ow);
        width_resized.as_slice()
    };
    Ok(if oh == ih {
        rows.to_vec()
    } else {
        resample_axis_bicubic_aa(rows, 1, ih, row_len, oh)
    })
}

/// The product of `dims`, or an error naming `context` when it overflows.
fn checked_product(context: &str, dims: &[usize]) -> FocrResult<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            FocrError::Other(anyhow::anyhow!(
                "{context}: shape product {dims:?} overflows usize"
            ))
        })
}

/// One output sample's taps along an axis: the first source index and the
/// normalized weights of the consecutive sources from there.
#[derive(Debug)]
struct AaTaps {
    start: usize,
    weights: Vec<f32>,
}

/// Torch's antialiased-bicubic taps for resizing an axis of `n_in` samples to
/// `n_out` (`_compute_indices_min_size_weights_aa`, align_corners=False).
///
/// When shrinking, the Keys kernel ([`bicubic_aa_filter`]) is stretched by the
/// scale `n_in/n_out`, which widens its support from 2 source samples on each
/// side to `2·scale`; the weights are then normalized to sum to 1. The
/// precision follows torch's float kernel step by step: the centre and the
/// filter argument are formed in f64 and rounded to f32, the support bounds
/// truncate toward zero like its `int64_t` casts, and the sum is plain f32.
/// That reproduces its weights bitwise.
fn bicubic_aa_taps(n_in: usize, n_out: usize) -> Vec<AaTaps> {
    let scale = n_in as f32 / n_out as f32;
    let (support, invscale) = if scale >= 1.0 {
        (2.0 * scale, (1.0 / f64::from(scale)) as f32)
    } else {
        (2.0, 1.0)
    };
    let max_taps = support.ceil() as i64 * 2 + 1;
    (0..n_out)
        .map(|i| {
            let center = (f64::from(scale) * (i as f64 + 0.5)) as f32;
            let start = ((f64::from(center - support) + 0.5) as i64).max(0);
            let end = ((f64::from(center + support) + 0.5) as i64).min(n_in as i64);
            let len = (end - start).clamp(0, max_taps);
            let mut weights: Vec<f32> = (start..start + len)
                .map(|j| {
                    let arg = (f64::from(j as f32 - center) + 0.5) * f64::from(invscale);
                    bicubic_aa_filter(arg as f32)
                })
                .collect();
            let total = weights.iter().fold(0.0f32, |acc, &w| acc + w);
            if total != 0.0 {
                for w in &mut weights {
                    *w /= total;
                }
            }
            AaTaps {
                start: start as usize,
                weights,
            }
        })
        .collect()
}

/// Torch's antialias cubic filter (`HelperInterpCubic::aa_filter`, the Keys
/// kernel with `a = -0.5`), its polynomials evaluated with the FMAs torch's
/// compiled kernel contracts them to.
fn bicubic_aa_filter(x: f32) -> f32 {
    const A: f32 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0).mul_add(x, -(A + 3.0)) * x).mul_add(x, 1.0)
    } else if x < 2.0 {
        A.mul_add(x, -5.0 * A)
            .mul_add(x, 8.0 * A)
            .mul_add(x, -4.0 * A)
    } else {
        0.0
    }
}

/// Resample the middle axis of a row-major `[outer, n_in, inner]` array to
/// `n_out` samples with [`bicubic_aa_taps`], accumulating each output in tap
/// order with fused multiply-adds (the inner loops run over contiguous
/// `inner` rows, so LLVM vectorizes them).
fn resample_axis_bicubic_aa(
    src: &[f32],
    outer: usize,
    n_in: usize,
    inner: usize,
    n_out: usize,
) -> Vec<f32> {
    let taps = bicubic_aa_taps(n_in, n_out);
    let mut out = vec![0.0f32; outer * n_out * inner];
    for (src_block, out_block) in src
        .chunks_exact(n_in * inner)
        .zip(out.chunks_exact_mut(n_out * inner))
    {
        for (tap, dst) in taps.iter().zip(out_block.chunks_exact_mut(inner)) {
            for (j, &w) in tap.weights.iter().enumerate() {
                let row = &src_block[(tap.start + j) * inner..][..inner];
                if j == 0 {
                    for (d, &s) in dst.iter_mut().zip(row) {
                        *d = s * w;
                    }
                } else {
                    for (d, &s) in dst.iter_mut().zip(row) {
                        *d = s.mul_add(w, *d);
                    }
                }
            }
        }
    }
    out
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

    /// bd-4l71 residency precondition: [`matmul`] is INVARIANT to how its `m`
    /// rows are partitioned — row block `[r0, r1)` of `matmul(a, b)` equals
    /// `matmul(a_rows[r0..r1], b)` BIT-FOR-BIT. Each output element is an
    /// independent length-`k` dot product whose accumulation order depends on
    /// `k` alone, so the kernel's `m`-blocking cannot move a bit. The streamed
    /// vision MLP relies on exactly this to run in row chunks (bounded scratch)
    /// without changing a single output value.
    #[test]
    fn matmul_is_row_block_invariant() {
        let (m, k, n) = (301usize, 96usize, 160usize);
        let a = Mat::from_vec(
            m,
            k,
            (0..m * k)
                .map(|i| ((i % 37) as f32) * 0.031 - 0.5)
                .collect(),
        );
        let b = Mat::from_vec(
            k,
            n,
            (0..k * n)
                .map(|i| ((i % 53) as f32) * 0.017 - 0.4)
                .collect(),
        );
        let whole = matmul(&a, &b).expect("whole matmul");
        for chunk in [1usize, 7, 64, 128, 300, 301] {
            let mut stitched = Vec::with_capacity(m * n);
            for start in (0..m).step_by(chunk) {
                let rows = chunk.min(m - start);
                let part = Mat::from_vec(rows, k, a.data[start * k..(start + rows) * k].to_vec());
                stitched.extend_from_slice(&matmul(&part, &b).expect("chunk matmul").data);
            }
            assert_eq!(
                stitched, whole.data,
                "row-chunked matmul (chunk {chunk}) must be bit-identical to the whole GEMM"
            );
        }
    }

    #[test]
    fn matmul_rejects_inner_mismatch() {
        let a = Mat::zeros(2, 3);
        let b = Mat::zeros(4, 2); // 3 != 4
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn matmul_rejects_malformed_backing_data_without_panicking() {
        let a = Mat {
            rows: 1,
            cols: 2,
            data: vec![1.0],
        };
        let b = Mat::zeros(2, 1);
        assert_err_contains(matmul(&a, &b), "matmul lhs: data len 1 != rows*cols 2");
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

    #[test]
    fn rms_norm_rejects_malformed_backing_data_without_panicking() {
        let x = Mat {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0],
        };
        assert_err_contains(rms_norm(&x, None, 1e-6), "rms_norm x: data len 3");
    }

    #[test]
    fn layer_norm_rejects_malformed_backing_data_without_panicking() {
        let x = Mat {
            rows: 1,
            cols: 3,
            data: vec![1.0, 2.0],
        };
        assert_err_contains(layer_norm(&x, None, None, 1e-6), "layer_norm x: data len 2");
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

    #[test]
    fn softmax_output_copy_fallback_preserves_values_and_old_allocation() {
        let expected = vec![0.25, 0.75];

        let mut copied = Mat::from_vec(1, 2, vec![1.0, 2.0]);
        let copied_ptr = copied.data.as_ptr();
        install_softmax_output(&mut copied, expected.clone(), true);
        assert_eq!(copied.data, expected);
        assert_eq!(copied.data.as_ptr(), copied_ptr);

        let mut transferred = Mat::from_vec(1, 2, vec![1.0, 2.0]);
        let out = expected.clone();
        let out_ptr = out.as_ptr();
        install_softmax_output(&mut transferred, out, false);
        assert_eq!(transferred.data, expected);
        assert_eq!(transferred.data.as_ptr(), out_ptr);
    }

    #[test]
    fn softmax_rows_rejects_malformed_backing_data_without_panicking() {
        let mut x = Mat {
            rows: 1,
            cols: 4,
            data: vec![1.0, 2.0, 3.0],
        };
        assert_err_contains(softmax_rows(&mut x), "softmax_rows x: data len 3");
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

    #[test]
    fn linear_int8_dynamic_rejects_malformed_backing_data_without_panicking() {
        let x = Mat {
            rows: 1,
            cols: 2,
            data: vec![1.0],
        };
        let w = QInt8 {
            w: vec![1i8, 2].into(),
            scales: vec![1.0],
            n: 1,
            k: 2,
            layout: super::super::tensor::WeightLayout::RowMajor,
        };
        assert_err_contains(
            linear_int8_dynamic(&x, &w, None),
            "linear_int8_dynamic x: data len 1",
        );

        let x = Mat::from_vec(1, 2, vec![1.0, 2.0]);
        let short_weight = QInt8 {
            w: vec![1i8].into(),
            scales: vec![1.0],
            n: 1,
            k: 2,
            layout: super::super::tensor::WeightLayout::RowMajor,
        };
        assert_err_contains(
            linear_int8_dynamic(&x, &short_weight, None),
            "linear_int8_dynamic weight: weight len 1 != n*k 2",
        );

        let missing_scale = QInt8 {
            w: vec![1i8, 2].into(),
            scales: vec![],
            n: 1,
            k: 2,
            layout: super::super::tensor::WeightLayout::RowMajor,
        };
        assert_err_contains(
            linear_int8_dynamic(&x, &missing_scale, None),
            "linear_int8_dynamic weight: scales len 0 != n 1",
        );
    }

    #[test]
    fn linear_int8_dynamic_activation_rounds_ties_to_even() {
        // The ±127 endpoints force activation scale_a = 1.0. Output 0 reads
        // x=0.5, which must round to 0; output 1 reads x=2.5, which must round
        // to 2. A half-away dynamic quantizer would return [1, 3].
        let x = Mat::from_vec(
            2,
            8,
            vec![
                -127.0, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 127.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        );
        let w = QInt8::new(
            vec![
                0, 0, 0, 0, 1, 0, 0, 0, //
                0, 0, 0, 0, 0, 0, 1, 0,
            ],
            vec![1.0, 1.0],
            2,
            8,
        );
        let y = linear_int8_dynamic(&x, &w, None).unwrap();
        assert_eq!(y.shape(), (2, 2));
        assert_eq!(y.data, vec![0.0, 2.0, 0.0, 0.0]);

        let again = linear_int8_dynamic(&x, &w, None).unwrap();
        assert_eq!(y, again, "dynamic activation quant must be byte-identical");
    }

    /// group_norm vs a torch `nn.GroupNorm(3, 6, eps=1e-5)` oracle golden —
    /// B2×C6×H2×W3, groups of 2 (the same family as TrOMR's GN32-over-64ch).
    /// Generated 2026-07-05 in the pinned zoo venv (torch 2.12.1):
    /// `x = sin(0.13·arange)`, `γ = linspace(0.5,1.5,6)`, `β = linspace(-0.2,0.2,6)`.
    #[test]
    fn group_norm_matches_torch_golden() {
        const X: [f32; 72] = [
            0.0, 0.1296341, 0.2570806, 0.3801884, 0.4968801, 0.6051864, 0.7032794, 0.7895037,
            0.8624042, 0.9207506, 0.9635582, 0.9901046, 0.9999417, 0.9929036, 0.9691092, 0.9289597,
            0.873133, 0.8025711, 0.7184649, 0.6222337, 0.5155014, 0.4000695, 0.277886, 0.1510129,
            0.0215911, -0.1081951, -0.2361552, -0.3601299, -0.4780271, -0.5878571, -0.6877661,
            -0.7760682, -0.8512733, -0.9121122, -0.957558, -0.9868438, -0.9994755, -0.9952398,
            -0.9742084, -0.9367359, -0.8834547, -0.8152643, -0.7333152, -0.6389909, -0.5338824,
            -0.4197641, -0.2985622, -0.1723212, -0.0431721, 0.0867056, 0.21512, 0.3399035,
            0.4589513, 0.5702536, 0.6719319, 0.7622709, 0.8397455, 0.9030485, 0.9511114, 0.983123,
            0.9985433, 0.997112, 0.9788533, 0.9440752, 0.8933646, 0.8275774, 0.7478238, 0.6554497,
            0.5520141, 0.4392635, 0.3190989, 0.1935491,
        ];
        const Y: [f32; 72] = [
            -1.1128683, -0.9128186, -0.716145, -0.5261666, -0.3460895, -0.1789526, 0.1213925,
            0.3076768, 0.4651756, 0.5912306, 0.6837148, 0.7410672, 0.9592738, 0.9367534, 0.8606158,
            0.7321458, 0.5535116, 0.3277276, 0.1605169, -0.2158298, -0.6332453, -1.084684,
            -1.5625268, -2.0587103, 2.4798393, 1.9679233, 1.4632101, 0.9742165, 0.509194,
            0.0759915, -0.305477, -0.7073503, -1.0496179, -1.3265028, -1.5333323, -1.6666154,
            -0.7448562, -0.7371473, -0.6988704, -0.6306711, -0.5337003, -0.4095948, -0.2046283,
            0.0357078, 0.3035217, 0.5942926, 0.9031123, 1.2247713, -1.6645347, -1.3156482,
            -0.9706925, -0.6354902, -0.3156959, -0.0167077, 0.4023003, 0.6989031, 0.9532693,
            1.1611068, 1.3189077, 1.424009, 1.5083864, 1.5014455, 1.4129068, 1.2442629, 0.9983606,
            0.6793492, 0.3991693, -0.1176782, -0.6964164, -1.3272738, -1.9996133, -2.7020838,
        ];
        let weight: Vec<f32> = (0..6).map(|i| 0.5 + i as f32 * 0.2).collect();
        let bias: Vec<f32> = (0..6).map(|i| -0.2 + i as f32 * 0.08).collect();

        let mut x = X.to_vec();
        group_norm(&mut x, 2, 6, 6, 3, 1e-5, &weight, &bias, false).unwrap();
        let maxabs = x
            .iter()
            .zip(Y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(maxabs <= 2e-6, "vs torch golden: maxabs {maxabs}");

        // Fused ReLU == unfused + clamp (and matches the oracle's relu(y)).
        let mut fused = X.to_vec();
        group_norm(&mut fused, 2, 6, 6, 3, 1e-5, &weight, &bias, true).unwrap();
        let clamped: Vec<f32> = x.iter().map(|v| v.max(0.0)).collect();
        assert_eq!(fused, clamped, "fused ReLU must equal unfused + clamp");
    }

    /// TF-'SAME' pad arithmetic vs the timm `padding.py` formula — every case
    /// the TrOMR backbone hits (tromr-spec §2a).
    #[test]
    fn tf_same_pad_amounts_match_timm() {
        // Stem 7×7 s2 at H=128: total 5 → 2 top / 3 bottom (the spec's example).
        assert_eq!(tf_same_pad_amounts(128, 7, 2), (2, 3));
        assert_eq!(tf_same_pad_amounts(1280, 7, 2), (2, 3));
        // 3×3 s1 → symmetric 1/1 at any size; 1×1 anything → 0/0.
        assert_eq!(tf_same_pad_amounts(37, 3, 1), (1, 1));
        assert_eq!(tf_same_pad_amounts(64, 1, 1), (0, 0));
        assert_eq!(tf_same_pad_amounts(64, 1, 2), (0, 0));
        // k3 s2: even i → total 1 (asymmetric 0/1), odd i → total 2 (1/1).
        assert_eq!(tf_same_pad_amounts(6, 3, 2), (0, 1));
        assert_eq!(tf_same_pad_amounts(9, 3, 2), (1, 1));
    }

    /// tf_same_pad(−∞) + max_pool2d vs a timm-`pad_same` + `F.max_pool2d`
    /// oracle golden (torch 2.12.1, 2026-07-05): x = cos(0.37·arange) over
    /// (1,2,6,9), k3 s2 → padded (7,11), out (3,5). H pads 0/1 (asymmetric),
    /// W pads 1/1 — the dynamic-SAME case a fixed pad cannot express.
    #[test]
    fn max_pool2d_same_matches_timm_golden() {
        let x: Vec<f32> = (0..2 * 6 * 9).map(|i| (i as f32 * 0.37).cos()).collect();
        const Y: [f32; 30] = [
            1.0, 0.9323273, 0.4507553, 0.93477, 0.9999768, 0.9298415, 0.7338563, 0.7475899,
            0.9999071, 0.9999071, 0.7292103, 0.4628793, 0.9395249, 0.999791, 0.9247403, 0.4262586,
            0.7565722, 0.9996285, 0.9996285, 0.7198165, 0.4749172, 0.9441053, 0.9994195, 0.9194672,
            0.4138885, 0.7654141, 0.9991641, 0.9991641, 0.7102889, 0.1313064,
        ];
        let (padded, ph, pw) = tf_same_pad(&x, 1, 2, 6, 9, 3, 3, 2, 2, f32::NEG_INFINITY);
        assert_eq!((ph, pw), (7, 11), "padded dims match timm pad_same");
        let y = max_pool2d(&padded, 1, 2, ph, pw, 3, 2, 3, 5);
        assert_eq!(y.len(), Y.len());
        let maxabs = y
            .iter()
            .zip(Y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(maxabs <= 1e-6, "vs timm golden: maxabs {maxabs}");
        // No output may be −∞ (the pad fill must never win a real window).
        assert!(
            y.iter().all(|v| v.is_finite()),
            "pool output must be finite"
        );
    }

    /// The zero-fill pad path composes with `conv2d` at the stem geometry:
    /// 128×1280 k7 s2 pads to 133×1285 (timm golden shape) and convolves to
    /// the SAME output 64×640.
    #[test]
    fn tf_same_pad_stem_geometry_composes_with_conv2d() {
        let (h, w) = (128usize, 1280usize);
        let x = vec![1.0f32; h * w];
        let (padded, ph, pw) = tf_same_pad(&x, 1, 1, h, w, 7, 7, 2, 2, 0.0);
        assert_eq!((ph, pw), (133, 1285), "timm pad_same golden shape");
        // A 7×7 all-ones kernel over an all-ones interior: the CENTER output
        // (away from every border) must sum to exactly 49.
        let weight = vec![1.0f32; 7 * 7];
        let (oh, ow) = (h.div_ceil(2), w.div_ceil(2));
        let y = conv2d(&padded, &weight, None, 1, 1, ph, pw, 7, 7, oh, ow, 2, 2, 1);
        assert_eq!(y.len(), oh * ow);
        assert_eq!(y[(oh / 2) * ow + ow / 2], 49.0, "interior window sums 7×7");
        // The top-left output sees the 2-top/2-left pad: 5×5 real ones = 25.
        assert_eq!(y[0], 25.0, "corner window is 5×5 real after 2/3-2/3 pads");
    }

    #[test]
    fn group_norm_rejects_bad_shapes() {
        let w = vec![1.0f32; 6];
        let b = vec![0.0f32; 6];
        // channels not divisible by groups
        let mut x = vec![0.0f32; 2 * 6 * 6];
        assert!(group_norm(&mut x, 2, 6, 6, 4, 1e-5, &w, &b, false).is_err());
        // zero groups
        assert!(group_norm(&mut x, 2, 6, 6, 0, 1e-5, &w, &b, false).is_err());
        // x length mismatch
        let mut short = vec![0.0f32; 5];
        assert!(group_norm(&mut short, 2, 6, 6, 3, 1e-5, &w, &b, false).is_err());
        // weight/bias length mismatch
        let mut x2 = vec![0.0f32; 2 * 6 * 6];
        assert!(group_norm(&mut x2, 2, 6, 6, 3, 1e-5, &w[..4], &b, false).is_err());
    }

    // ── antialiased bicubic resize vs torch 2.13 (bd-p8hu) ─────────────────
    //
    // The expected values below were produced by torch's
    // `F.interpolate(mode="bicubic", antialias=True, align_corners=False)` on
    // CPU (the U2-C1 probe `aa_test_vectors.py`), so they pin the bitwise
    // reproduction rather than a tolerance.

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn bicubic_aa_taps_match_torch_on_the_sam_64_to_40_resize() {
        let taps = bicubic_aa_taps(64, 40);
        assert_eq!(taps.len(), 40);
        // Row 0 is clipped at the left border: four taps, renormalized.
        assert_eq!(taps[0].start, 0);
        assert_eq!(
            bits(&taps[0].weights),
            bits(&[0.611_380_93, 0.429_091_78, -0.018_212_724, -0.022_259_997])
        );
        // Row 17 is interior: the support 2·1.6 spans six source samples.
        assert_eq!(taps[17].start, 25);
        assert_eq!(
            bits(&taps[17].weights),
            bits(&[
                -0.034_223_188,
                0.024_600_342,
                0.509_622_9,
                0.509_622_9,
                0.024_600_342,
                -0.034_223_188,
            ])
        );
    }

    #[test]
    fn resize_bicubic_aa_shrink_matches_torch_bitwise() -> FocrResult<()> {
        // A 5×5 ramp with a spike at (2, 3), shrunk to 3×3.
        let src = [
            -2.0,
            -1.63,
            -1.26,
            -0.89,
            -0.52,
            -0.149_999_98,
            0.220_000_03,
            0.590_000_15,
            0.960_000_04,
            1.329_999_9,
            1.7,
            2.070_000_2,
            2.44,
            9.5,
            3.180_000_3,
            3.550_000_2,
            3.92,
            4.29,
            4.66,
            5.03,
            5.4,
            5.77,
            6.140_000_3,
            6.51,
            6.88,
        ];
        let want = [
            -1.214_461_2,
            -0.605_384_4,
            0.003_692_302_6,
            1.711_245_3,
            3.398_815_6,
            4.784_407_6,
            4.876_307,
            5.485_383_5,
            6.094_460_5,
        ];
        let got = resize_bicubic_aa(&src, 5, 5, 1, 3, 3)?;
        assert_eq!(bits(&got), bits(&want));
        Ok(())
    }

    #[test]
    fn resize_bicubic_aa_mixed_grow_and_shrink_matches_torch_bitwise() -> FocrResult<()> {
        // Channel-last 3×3×2 grown to 4 rows and shrunk to 2 columns: checks
        // the channel layout, the upsampling taps, and the width-then-height
        // order together.
        let src = [
            1.0, -2.0, 0.5, 4.0, 3.0, 0.25, -1.5, 2.0, 7.0, -3.0, 0.0, 1.0, 2.5, 0.5, -0.75, 6.0,
            4.0, -1.0,
        ];
        let want = [
            0.670_887,
            0.258_364_02,
            2.069_332_4,
            1.928_514_6,
            1.411_494_9,
            -0.033_856_93,
            2.575_616_6,
            0.137_403_92,
            1.609_888_3,
            1.054_905_3,
            2.620_989,
            0.078_508_59,
            1.141_691_9,
            2.842_091_8,
            2.177_005,
            1.788_751,
        ];
        let got = resize_bicubic_aa(&src, 3, 3, 2, 4, 2)?;
        assert_eq!(bits(&got), bits(&want));
        Ok(())
    }

    #[test]
    fn resize_bicubic_aa_keeps_unchanged_sizes_and_constants() -> FocrResult<()> {
        let src: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
        assert_eq!(resize_bicubic_aa(&src, 2, 3, 2, 2, 3)?, src);
        // Normalized taps preserve a constant field in both directions.
        let constant = vec![7.25f32; 16 * 16 * 3];
        for (oh, ow) in [(10, 10), (23, 9)] {
            let out = resize_bicubic_aa(&constant, 16, 16, 3, oh, ow)?;
            assert_eq!(out.len(), oh * ow * 3);
            assert!(out.iter().all(|v| (v - 7.25).abs() < 1e-5), "{oh}x{ow}");
        }
        Ok(())
    }

    #[test]
    fn resize_bicubic_aa_rejects_bad_geometry_without_panic() {
        assert_err_contains(resize_bicubic_aa(&[1.0; 3], 2, 2, 1, 1, 1), "source len");
        assert_err_contains(resize_bicubic_aa(&[], 0, 2, 1, 1, 1), "must be non-zero");
        assert_err_contains(resize_bicubic_aa(&[], 2, 2, 1, 0, 1), "must be non-zero");
        assert_err_contains(
            resize_bicubic_aa(&[], usize::MAX, 2, 1, 1, 1),
            "overflows usize",
        );
    }
}
