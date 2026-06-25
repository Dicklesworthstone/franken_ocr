//! CLIP-L/14 vision forward ([SPEC-047..050], PROPOSED_ARCHITECTURE.md §6.4).
//! Implemented by bd-1gv.9 (parity test bead bd-1gv.9.1).
//!
//! Fused tower ([SPEC-048]): CLIP embeddings take the SAM `x3` output as
//! `patch_embeds` (the CLIP patch-embed Conv2d is bypassed); the learned class
//! token is prepended; absolute position embeddings are added via `get_abs_pos`
//! (bicubic-interpolated to the runtime token length, CLS row passed through).
//!
//! Each `NoTPTransformerBlock` ([SPEC-049], `deepencoder.py:392-396`):
//! ```text
//!   h   = x + self_attn(layer_norm1(x))
//!   out = h + mlp(layer_norm2(h))      mlp = fc2(quick_gelu(fc1(·)))
//! ```
//! Attention is full (non-causal) SDPA with `qkv_proj`/`out_proj` biases; the
//! MLP uses CLIP `quick_gelu` `x·σ(1.702x)` ([SPEC-049], distinct from the SAM
//! erf-GELU and the decoder SiLU).
//!
//! Build params ([SPEC-047]): num_layers=24, hidden_size=1024,
//! num_attention_heads=16, ffn_hidden_size=4096, patch_size=14,
//! layernorm_epsilon=1e-5, pre_layernorm_epsilon=1e-5.
//!
//! All kernel calls funnel through [`crate::native_engine::nn`] (the frankentorch
//! facade). The bias-add after each `Linear`, the qkv head split / SDPA repack,
//! and the bicubic abs-pos interpolation are not facade ops, so they are
//! implemented inline here (see `facade_gaps`).

use super::nn;
use super::tensor::Mat;
use super::weights::Weights;
use crate::error::{FocrError, FocrResult};

/// CLIP-L/14 build parameters ([SPEC-047], `deepencoder.py:514-532`).
#[derive(Debug, Clone, Copy)]
pub struct ClipConfig {
    /// Transformer depth (number of `NoTPTransformerBlock`s).
    pub num_layers: usize,
    /// Model width / channel dim (`hidden_size`).
    pub hidden_size: usize,
    /// Attention heads (`num_attention_heads`).
    pub num_heads: usize,
    /// FFN inner width (`ffn_hidden_size`).
    pub ffn_hidden_size: usize,
    /// Patch size of the (bypassed) patch-embed conv — kept for provenance.
    pub patch_size: usize,
    /// LayerNorm epsilon for the per-block norms (`layernorm_epsilon`).
    pub layernorm_eps: f32,
    /// LayerNorm epsilon for the pre-transformer norm (`pre_layernorm_epsilon`).
    pub pre_layernorm_eps: f32,
}

impl Default for ClipConfig {
    /// The deployed CLIP-L/14-224 config ([SPEC-047]).
    fn default() -> Self {
        Self {
            num_layers: 24,
            hidden_size: 1024,
            num_heads: 16,
            ffn_hidden_size: 4096,
            patch_size: 14,
            layernorm_eps: 1e-5,
            pre_layernorm_eps: 1e-5,
        }
    }
}

impl ClipConfig {
    /// Per-head dimension (`hidden_size / num_heads`).
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

/// Affine LayerNorm parameters (`weight`, `bias`), each length `hidden_size`.
#[derive(Debug, Clone)]
pub struct LayerNormParams {
    /// Per-feature scale `γ`.
    pub weight: Vec<f32>,
    /// Per-feature shift `β`.
    pub bias: Vec<f32>,
}

/// A `nn.Linear(in, out)` weight in PyTorch `[out, in]` row-major layout plus an
/// optional bias of length `out`.
///
/// `y = x · Wᵀ + b`. Stored exactly as PyTorch ships it (no pre-transpose), so
/// the matmul transposes the weight at apply time.
#[derive(Debug, Clone)]
pub struct LinearParams {
    /// Row-major `[out, in]` weights.
    pub weight: Vec<f32>,
    /// Optional length-`out` bias.
    pub bias: Option<Vec<f32>>,
    /// Output features.
    pub out_features: usize,
    /// Input features.
    pub in_features: usize,
}

/// Weights of one `NoTPTransformerBlock` ([SPEC-049]).
#[derive(Debug, Clone)]
pub struct ClipBlockWeights {
    /// Pre-attention norm (`layer_norm1`).
    pub layer_norm1: LayerNormParams,
    /// Fused qkv projection `[3·hidden, hidden]` (`qkv_proj`, bias=True).
    pub qkv_proj: LinearParams,
    /// Output projection `[hidden, hidden]` (`out_proj`, bias=True).
    pub out_proj: LinearParams,
    /// Pre-MLP norm (`layer_norm2`).
    pub layer_norm2: LayerNormParams,
    /// MLP up-projection `[ffn, hidden]` (`fc1`, bias=True).
    pub fc1: LinearParams,
    /// MLP down-projection `[hidden, ffn]` (`fc2`, bias=True).
    pub fc2: LinearParams,
}

/// All CLIP tower weights needed for [`forward_with`].
///
/// This is the in-memory view the `.focrq` loader (bd-1es.3) will hydrate; the
/// public [`forward`] entrypoint will pull these out of [`Weights`] once that
/// reader exists. Keeping it explicit lets the tower be exercised end-to-end in
/// unit tests with no model present.
#[derive(Debug, Clone)]
pub struct ClipWeights {
    /// Learned class token, length `hidden_size` (`class_embedding`).
    pub class_embedding: Vec<f32>,
    /// Absolute position embedding table, `[num_positions, hidden_size]` row-major
    /// (`position_embedding.weight`; `num_positions = (image/patch)² + 1`).
    pub position_embedding: Vec<f32>,
    /// Number of position rows (`num_positions`).
    pub num_positions: usize,
    /// Pre-transformer LayerNorm (`pre_layrnorm`).
    pub pre_layernorm: LayerNormParams,
    /// The 24 transformer blocks, in order.
    pub blocks: Vec<ClipBlockWeights>,
}

/// Run the CLIP tower over the image with the SAM features as `patch_embeds`,
/// returning the per-patch CLIP hidden states (class token included at row 0).
///
/// This is the [`Weights`]-backed entrypoint the engine wires; until the
/// `.focrq` reader (bd-1es.3) exposes typed tensor accessors there is nothing to
/// hydrate a [`ClipWeights`] from, so it currently reports
/// [`FocrError::NotImplemented`]. The real math lives in [`forward_with`], which
/// is fully exercised by the unit tests below.
///
/// # Errors
/// [`FocrError::NotImplemented`] until the weight loader lands; thereafter
/// whatever [`forward_with`] returns.
pub fn forward(_weights: &Weights, _image: &Mat, _sam_features: &Mat) -> FocrResult<Mat> {
    Err(FocrError::NotImplemented(
        "native_engine::vision_clip::forward — awaiting .focrq weight accessors (bd-1es.3); \
         the tower math is in vision_clip::forward_with"
            .into(),
    ))
}

/// Run the CLIP tower with explicit weights ([SPEC-048..050]).
///
/// `sam_features` is the SAM `x3` feature map flattened to a `[num_patches,
/// hidden_size]` token matrix — i.e. `flatten(2).transpose(1,2)` already applied
/// so each row is one patch token of width `hidden_size`. (The deployed forward
/// bypasses CLIP's own patch-embed Conv2d and injects SAM's grid directly;
/// [SPEC-048], `deepencoder.py:274-283`.)
///
/// Returns the `[num_patches + 1, hidden_size]` hidden states with the class
/// token at row 0 (the caller drops it via `[:, 1:]` at concat time; [SPEC-051]).
///
/// # Errors
/// [`FocrError::Other`] on any shape mismatch (wrong `sam_features` width, a
/// weight whose dimensions disagree with [`ClipConfig`], or a kernel rejection).
pub fn forward_with(
    cfg: &ClipConfig,
    weights: &ClipWeights,
    sam_features: &Mat,
) -> FocrResult<Mat> {
    let dim = cfg.hidden_size;
    if sam_features.cols != dim {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip: sam_features width {} != hidden_size {}",
            sam_features.cols,
            dim
        )));
    }
    if weights.class_embedding.len() != dim {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip: class_embedding len {} != hidden_size {}",
            weights.class_embedding.len(),
            dim
        )));
    }
    if weights.blocks.len() != cfg.num_layers {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip: {} blocks != num_layers {}",
            weights.blocks.len(),
            cfg.num_layers
        )));
    }

    // ── Embeddings ([SPEC-048]) ────────────────────────────────────────────
    // Prepend the class token, then add the (interpolated) abs-pos embedding.
    let mut x = prepend_class_token(&weights.class_embedding, sam_features);
    let seq = x.rows; // num_patches + 1
    let pos = abs_pos_for_len(
        &weights.position_embedding,
        weights.num_positions,
        dim,
        seq,
    )?;
    add_in_place(&mut x, &pos)?;

    // ── pre_layrnorm ([SPEC-049], deepencoder.py:470-473) ──────────────────
    x = nn::layer_norm(
        &x,
        Some(&weights.pre_layernorm.weight),
        Some(&weights.pre_layernorm.bias),
        cfg.pre_layernorm_eps,
    )?;

    // ── 24 transformer blocks ──────────────────────────────────────────────
    for block in &weights.blocks {
        x = transformer_block(cfg, block, &x)?;
    }
    Ok(x)
}

/// One `NoTPTransformerBlock` ([SPEC-049], `deepencoder.py:392-396`):
/// `h = x + attn(LN1(x)); out = h + mlp(LN2(h))`.
fn transformer_block(
    cfg: &ClipConfig,
    w: &ClipBlockWeights,
    x: &Mat,
) -> FocrResult<Mat> {
    // h = x + self_attn(layer_norm1(x))
    let normed = nn::layer_norm(
        x,
        Some(&w.layer_norm1.weight),
        Some(&w.layer_norm1.bias),
        cfg.layernorm_eps,
    )?;
    let attn = self_attention(cfg, w, &normed)?;
    let h = add(x, &attn)?;

    // out = h + mlp(layer_norm2(h))
    let normed2 = nn::layer_norm(
        &h,
        Some(&w.layer_norm2.weight),
        Some(&w.layer_norm2.bias),
        cfg.layernorm_eps,
    )?;
    let mlp = feed_forward(w, &normed2)?;
    add(&h, &mlp)
}

/// `NoTPAttention` full SDPA ([SPEC-049], `deepencoder.py:314-371`).
///
/// `xqkv = qkv_proj(x)` `[seq, 3·dim]`; split into per-head q/k/v laid out
/// `[num_heads, seq, head_dim]` (head-major, the layout `nn::sdpa` consumes);
/// SDPA with `scale = 1/sqrt(head_dim)` and no causal mask; repack
/// `[seq, dim]`; `out_proj`.
fn self_attention(
    cfg: &ClipConfig,
    w: &ClipBlockWeights,
    x: &Mat,
) -> FocrResult<Mat> {
    let dim = cfg.hidden_size;
    let heads = cfg.num_heads;
    let hd = cfg.head_dim();
    let seq = x.rows;

    // Fused qkv projection -> [seq, 3*dim].
    let qkv = linear(&w.qkv_proj, x)?;
    debug_assert_eq!(qkv.cols, 3 * dim);

    // Repack into head-major flat buffers [heads, seq, hd] for q, k, v.
    // PyTorch view: xqkv.view(bsz, seq, 3, heads, hd) — so within a row the
    // layout is [t (3), head, hd]: column index = t*dim + head*hd + d.
    let mut q = vec![0.0f32; heads * seq * hd];
    let mut k = vec![0.0f32; heads * seq * hd];
    let mut v = vec![0.0f32; heads * seq * hd];
    for s in 0..seq {
        let row = qkv.row(s);
        for hh in 0..heads {
            for d in 0..hd {
                // Column index within the [t(3), head, hd] fused row.
                let q_src = hh * hd + d;
                let k_src = dim + hh * hd + d;
                let v_src = 2 * dim + hh * hd + d;
                let dst = hh * (seq * hd) + s * hd + d;
                q[dst] = row[q_src];
                k[dst] = row[k_src];
                v[dst] = row[v_src];
            }
        }
    }

    // Full (non-causal) SDPA. num_bh = batch(1) * heads.
    let scale = 1.0f32 / (hd as f32).sqrt();
    let ctx = nn::sdpa(&q, &k, &v, heads, seq, seq, hd, hd, scale, false);
    debug_assert_eq!(ctx.len(), heads * seq * hd);

    // Repack [heads, seq, hd] -> [seq, dim] (permute(0,2,1,3).reshape).
    let mut merged = Mat::zeros(seq, dim);
    for hh in 0..heads {
        for s in 0..seq {
            for d in 0..hd {
                let src = hh * (seq * hd) + s * hd + d;
                let dst = s * dim + hh * hd + d;
                merged.data[dst] = ctx[src];
            }
        }
    }

    // out_proj.
    linear(&w.out_proj, &merged)
}

/// `NoTPFeedForward`: `fc2(quick_gelu(fc1(x)))` ([SPEC-049],
/// `deepencoder.py:295-309`).
fn feed_forward(w: &ClipBlockWeights, x: &Mat) -> FocrResult<Mat> {
    let mut hidden = linear(&w.fc1, x)?;
    nn::quick_gelu(&mut hidden);
    linear(&w.fc2, &hidden)
}

/// Apply a PyTorch `nn.Linear`: `y = x · Wᵀ + b`.
///
/// `W` is `[out, in]` row-major; `nn::matmul` contracts the inner dims, so we
/// transpose `W` to `[in, out]` first, then add the optional bias per row. (The
/// facade has no fused linear-with-bias for f32 activations — only the int8
/// dynamic path — so the bias add is inline; see `facade_gaps`.)
fn linear(w: &LinearParams, x: &Mat) -> FocrResult<Mat> {
    if x.cols != w.in_features {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip linear: x.cols {} != in_features {}",
            x.cols,
            w.in_features
        )));
    }
    if w.weight.len() != w.out_features * w.in_features {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip linear: weight len {} != out*in {}",
            w.weight.len(),
            w.out_features * w.in_features
        )));
    }
    let wt = transpose(&w.weight, w.out_features, w.in_features);
    let wt_mat = Mat::from_vec(w.in_features, w.out_features, wt);
    let mut y = nn::matmul(x, &wt_mat)?;
    if let Some(b) = &w.bias {
        if b.len() != w.out_features {
            return Err(FocrError::Other(anyhow::anyhow!(
                "vision_clip linear: bias len {} != out_features {}",
                b.len(),
                w.out_features
            )));
        }
        for r in 0..y.rows {
            let row = y.row_mut(r);
            for (c, bv) in b.iter().enumerate() {
                row[c] += *bv;
            }
        }
    }
    Ok(y)
}

/// Transpose a row-major `[rows, cols]` flat matrix into `[cols, rows]`.
fn transpose(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    out
}

/// Prepend the class token as a new row 0, returning a `[patches+1, dim]` matrix.
fn prepend_class_token(class_embedding: &[f32], patches: &Mat) -> Mat {
    let dim = patches.cols;
    let seq = patches.rows + 1;
    let mut data = Vec::with_capacity(seq * dim);
    data.extend_from_slice(class_embedding);
    data.extend_from_slice(&patches.data);
    Mat::from_vec(seq, dim, data)
}

/// Build the position embedding for a runtime sequence length `seq`
/// (`get_abs_pos`, `deepencoder.py:199-235`).
///
/// `table` is `[num_positions, dim]`: row 0 is the CLS position, rows `1..` are
/// a `src×src` patch grid (`src = round(sqrt(num_positions - 1))`). The runtime
/// patch grid is `tgt×tgt` (`tgt = round(sqrt(seq - 1))`). When `src == tgt` the
/// table is returned as-is; otherwise the patch rows are bicubically
/// interpolated from `src×src` to `tgt×tgt`, the CLS row passing through
/// unchanged.
fn abs_pos_for_len(
    table: &[f32],
    num_positions: usize,
    dim: usize,
    seq: usize,
) -> FocrResult<Mat> {
    if table.len() != num_positions * dim {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip abs_pos: table len {} != num_positions*dim {}",
            table.len(),
            num_positions * dim
        )));
    }
    if seq == num_positions {
        return Ok(Mat::from_vec(seq, dim, table.to_vec()));
    }
    let src = isqrt(num_positions - 1);
    let tgt = isqrt(seq - 1);
    if src * src != num_positions - 1 || tgt * tgt != seq - 1 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip abs_pos: non-square grids (src²={}, num_patches={}, tgt²={}, runtime_patches={})",
            src * src,
            num_positions - 1,
            tgt * tgt,
            seq - 1
        )));
    }

    let mut out = vec![0.0f32; seq * dim];
    // CLS row passes through.
    out[..dim].copy_from_slice(&table[..dim]);

    if src == tgt {
        out[dim..].copy_from_slice(&table[dim..]);
        return Ok(Mat::from_vec(seq, dim, out));
    }

    // Bicubic interpolation per channel from src×src -> tgt×tgt. Source patch
    // grid begins at table row 1 (row 0 is CLS).
    let patch = &table[dim..]; // [src*src, dim]
    interpolate_bicubic_into(patch, src, src, &mut out[dim..], tgt, tgt, dim);
    Ok(Mat::from_vec(seq, dim, out))
}

/// Bicubic resize of a `[ih*iw, dim]` row-major grid into a `[oh*ow, dim]`
/// destination, `align_corners=False` (PyTorch `F.interpolate` convention).
///
/// Implemented inline because the facade has no spatial-resize op (see
/// `facade_gaps`). The cubic convolution uses the Keys kernel with `a = -0.75`
/// (PyTorch's default). NOTE: PyTorch additionally enables `antialias=True` for
/// the CLIP pos-embed path; the antialias prefilter is a measured-parity lever
/// deferred to the parity bead (bd-1gv.9.1) — this non-antialiased bicubic is
/// the identity path when `src == tgt`, which is the deployed CLIP-L/14-224 case
/// (16×16 patch grid both sides), so it never actually fires in the shipped
/// config and exists for completeness / non-224 grids.
fn interpolate_bicubic_into(
    src: &[f32],
    ih: usize,
    iw: usize,
    dst: &mut [f32],
    oh: usize,
    ow: usize,
    dim: usize,
) {
    let scale_h = ih as f32 / oh as f32;
    let scale_w = iw as f32 / ow as f32;
    for oy in 0..oh {
        // align_corners=False source coordinate.
        let sy = (oy as f32 + 0.5) * scale_h - 0.5;
        let iy = sy.floor();
        let fy = sy - iy;
        let iy = iy as isize;
        let wy = cubic_weights(fy);
        for ox in 0..ow {
            let sx = (ox as f32 + 0.5) * scale_w - 0.5;
            let ix = sx.floor();
            let fx = sx - ix;
            let ix = ix as isize;
            let wx = cubic_weights(fx);
            for c in 0..dim {
                let mut acc = 0.0f32;
                for (m, &wym) in wy.iter().enumerate() {
                    let yy = clamp_index(iy - 1 + m as isize, ih);
                    for (n, &wxn) in wx.iter().enumerate() {
                        let xx = clamp_index(ix - 1 + n as isize, iw);
                        acc += wym * wxn * src[(yy * iw + xx) * dim + c];
                    }
                }
                dst[(oy * ow + ox) * dim + c] = acc;
            }
        }
    }
}

/// Keys bicubic convolution weights for fractional offset `t ∈ [0,1)`, `a=-0.75`.
fn cubic_weights(t: f32) -> [f32; 4] {
    let a = -0.75f32;
    // Distances from the four neighbours at offsets -1,0,1,2 relative to floor.
    let w0 = cubic_kernel(a, 1.0 + t);
    let w1 = cubic_kernel(a, t);
    let w2 = cubic_kernel(a, 1.0 - t);
    let w3 = cubic_kernel(a, 2.0 - t);
    [w0, w1, w2, w3]
}

/// The cubic convolution kernel value at distance `x` (≥0) with parameter `a`.
fn cubic_kernel(a: f32, x: f32) -> f32 {
    let x = x.abs();
    if x <= 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

/// Clamp `i` into `[0, n)` (edge replication, matching `F.interpolate` borders).
fn clamp_index(i: isize, n: usize) -> usize {
    if i < 0 {
        0
    } else if i as usize >= n {
        n - 1
    } else {
        i as usize
    }
}

/// Integer square root (largest `r` with `r*r <= v`).
fn isqrt(v: usize) -> usize {
    if v == 0 {
        return 0;
    }
    let mut r = (v as f64).sqrt() as usize;
    while r * r > v {
        r -= 1;
    }
    while (r + 1) * (r + 1) <= v {
        r += 1;
    }
    r
}

/// `a + b` (fresh matrix), shapes must match.
fn add(a: &Mat, b: &Mat) -> FocrResult<Mat> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip add: shape mismatch [{},{}] vs [{},{}]",
            a.rows,
            a.cols,
            b.rows,
            b.cols
        )));
    }
    let data = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| x + y)
        .collect::<Vec<_>>();
    Ok(Mat::from_vec(a.rows, a.cols, data))
}

/// `a += b` in place, shapes must match.
fn add_in_place(a: &mut Mat, b: &Mat) -> FocrResult<()> {
    if a.rows != b.rows || a.cols != b.cols {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip add_in_place: shape mismatch [{},{}] vs [{},{}]",
            a.rows,
            a.cols,
            b.rows,
            b.cols
        )));
    }
    for (x, y) in a.data.iter_mut().zip(&b.data) {
        *x += *y;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny but structurally-faithful CLIP config: dim 4, 2 heads, ffn 8.
    fn tiny_cfg() -> ClipConfig {
        ClipConfig {
            num_layers: 2,
            hidden_size: 4,
            num_heads: 2,
            ffn_hidden_size: 8,
            patch_size: 14,
            layernorm_eps: 1e-5,
            pre_layernorm_eps: 1e-5,
        }
    }

    fn ln_identity(dim: usize) -> LayerNormParams {
        LayerNormParams {
            weight: vec![1.0; dim],
            bias: vec![0.0; dim],
        }
    }

    /// An identity-ish linear: weight = I (square) or zero-padded, bias 0.
    fn linear_identity(dim: usize) -> LinearParams {
        let mut w = vec![0.0f32; dim * dim];
        for i in 0..dim {
            w[i * dim + i] = 1.0;
        }
        LinearParams {
            weight: w,
            bias: Some(vec![0.0; dim]),
            out_features: dim,
            in_features: dim,
        }
    }

    /// qkv that yields q=k=v=x for every head (stacked three identities of dim).
    fn qkv_identity(dim: usize) -> LinearParams {
        // out = 3*dim, in = dim; rows [0..dim) -> q=I, [dim..2dim) -> k=I, etc.
        let mut w = vec![0.0f32; 3 * dim * dim];
        for t in 0..3 {
            for i in 0..dim {
                let out_row = t * dim + i;
                w[out_row * dim + i] = 1.0;
            }
        }
        LinearParams {
            weight: w,
            bias: Some(vec![0.0; 3 * dim]),
            out_features: 3 * dim,
            in_features: dim,
        }
    }

    fn fc1_identity_padded(dim: usize, ffn: usize) -> LinearParams {
        // [ffn, dim]; top dim rows = I, rest zero.
        let mut w = vec![0.0f32; ffn * dim];
        for i in 0..dim {
            w[i * dim + i] = 1.0;
        }
        LinearParams {
            weight: w,
            bias: Some(vec![0.0; ffn]),
            out_features: ffn,
            in_features: dim,
        }
    }

    fn fc2_identity_padded(dim: usize, ffn: usize) -> LinearParams {
        // [dim, ffn]; left dim cols = I, rest zero.
        let mut w = vec![0.0f32; dim * ffn];
        for i in 0..dim {
            w[i * ffn + i] = 1.0;
        }
        LinearParams {
            weight: w,
            bias: Some(vec![0.0; dim]),
            out_features: dim,
            in_features: ffn,
        }
    }

    fn block_identity(dim: usize, ffn: usize) -> ClipBlockWeights {
        ClipBlockWeights {
            layer_norm1: ln_identity(dim),
            qkv_proj: qkv_identity(dim),
            out_proj: linear_identity(dim),
            layer_norm2: ln_identity(dim),
            fc1: fc1_identity_padded(dim, ffn),
            fc2: fc2_identity_padded(dim, ffn),
        }
    }

    // ── transpose / linear ─────────────────────────────────────────────────

    #[test]
    fn transpose_roundtrips() {
        // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        let t = transpose(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn linear_applies_weight_and_bias() {
        // W = [[1,0,0],[0,1,0]] (out=2,in=3), b=[10,20]; x=[1,2,3] -> [11,22].
        let w = LinearParams {
            weight: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            bias: Some(vec![10.0, 20.0]),
            out_features: 2,
            in_features: 3,
        };
        let x = Mat::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let y = linear(&w, &x).unwrap();
        assert_eq!(y.shape(), (1, 2));
        assert!((y.data[0] - 11.0).abs() < 1e-6);
        assert!((y.data[1] - 22.0).abs() < 1e-6);
    }

    #[test]
    fn linear_rejects_dim_mismatch() {
        let w = linear_identity(4);
        let x = Mat::zeros(2, 3); // 3 != 4
        assert!(linear(&w, &x).is_err());
    }

    // ── class token + abs pos ──────────────────────────────────────────────

    #[test]
    fn class_token_prepended_as_row0() {
        let patches = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = prepend_class_token(&[7.0, 8.0, 9.0], &patches);
        assert_eq!(out.shape(), (3, 3));
        assert_eq!(out.row(0), &[7.0, 8.0, 9.0]);
        assert_eq!(out.row(1), &[1.0, 2.0, 3.0]);
        assert_eq!(out.row(2), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn abs_pos_passthrough_when_lengths_match() {
        // num_positions = 5 (1 CLS + 2x2 grid), dim 2, seq 5 -> identity.
        let table: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let pos = abs_pos_for_len(&table, 5, 2, 5).unwrap();
        assert_eq!(pos.shape(), (5, 2));
        assert_eq!(pos.data, table);
    }

    #[test]
    fn abs_pos_same_grid_size_is_identity_even_via_resize_branch() {
        // Force src==tgt path with seq != num_positions impossible; instead test
        // that a genuine 2x2->2x2 (num_pos 5, seq 5) is exact above; here check
        // the CLS row is always preserved on the interpolation branch (3x3->2x2).
        let np = 10; // 1 + 3*3
        let dim = 1;
        let mut table = vec![0.0f32; np * dim];
        table[0] = 99.0; // CLS marker
        for (i, slot) in table.iter_mut().enumerate().skip(1) {
            *slot = i as f32;
        }
        // seq = 1 + 2*2 = 5
        let pos = abs_pos_for_len(&table, np, dim, 5).unwrap();
        assert_eq!(pos.shape(), (5, 1));
        // CLS row preserved exactly.
        assert!((pos.data[0] - 99.0).abs() < 1e-6);
        // Interpolated values are convex-ish combinations within source range.
        for &v in &pos.data[1..] {
            assert!(v >= 0.5 && v <= 10.0, "interp out of expected range: {v}");
        }
    }

    #[test]
    fn abs_pos_rejects_non_square_grid() {
        // num_positions-1 = 3 is not a perfect square.
        let table = vec![0.0f32; 4 * 2];
        assert!(abs_pos_for_len(&table, 4, 2, 5).is_err());
    }

    // ── bicubic kernel sanity ──────────────────────────────────────────────

    #[test]
    fn cubic_weights_partition_of_unity() {
        for &t in &[0.0f32, 0.25, 0.5, 0.75, 0.999] {
            let w = cubic_weights(t);
            let s: f32 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "weights sum {s} != 1 at t={t}");
        }
    }

    #[test]
    fn cubic_weights_at_zero_is_unit_impulse() {
        // t=0 -> sample sits on a source pixel; weights = [0,1,0,0].
        let w = cubic_weights(0.0);
        assert!((w[0]).abs() < 1e-6);
        assert!((w[1] - 1.0).abs() < 1e-6);
        assert!((w[2]).abs() < 1e-6);
        assert!((w[3]).abs() < 1e-6);
    }

    // ── attention shape & residual structure ───────────────────────────────

    #[test]
    fn self_attention_preserves_shape() {
        let cfg = tiny_cfg();
        let block = block_identity(cfg.hidden_size, cfg.ffn_hidden_size);
        let x = Mat::from_vec(
            3,
            4,
            vec![
                0.1, 0.2, 0.3, 0.4, //
                0.5, 0.6, 0.7, 0.8, //
                0.9, 1.0, 1.1, 1.2,
            ],
        );
        let out = self_attention(&cfg, &block, &x).unwrap();
        assert_eq!(out.shape(), (3, 4));
    }

    /// With q=k=v=x and identity out_proj, attention is a softmax-weighted
    /// average of the value rows — every output row stays inside the convex hull
    /// of the inputs (each column bounded by per-column min/max).
    #[test]
    fn self_attention_is_convex_average_of_values() {
        let cfg = tiny_cfg();
        let block = block_identity(cfg.hidden_size, cfg.ffn_hidden_size);
        let x = Mat::from_vec(
            3,
            4,
            vec![
                0.0, 1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, 7.0, //
                8.0, 9.0, 10.0, 11.0,
            ],
        );
        let out = self_attention(&cfg, &block, &x).unwrap();
        for c in 0..4 {
            let col_min = (0..3).map(|r| x.get(r, c)).fold(f32::INFINITY, f32::min);
            let col_max = (0..3).map(|r| x.get(r, c)).fold(f32::NEG_INFINITY, f32::max);
            for r in 0..3 {
                let v = out.get(r, c);
                assert!(
                    v >= col_min - 1e-4 && v <= col_max + 1e-4,
                    "out[{r},{c}]={v} outside [{col_min},{col_max}]"
                );
            }
        }
    }

    // ── feed-forward = quick_gelu sandwiched in identities ─────────────────

    #[test]
    fn feed_forward_applies_quick_gelu() {
        let cfg = tiny_cfg();
        let block = block_identity(cfg.hidden_size, cfg.ffn_hidden_size);
        // fc1=I (padded), fc2=I (padded) => ff(x) == quick_gelu(x) elementwise.
        let x = Mat::from_vec(1, 4, vec![-1.0, 0.0, 1.0, 2.0]);
        let out = feed_forward(&block, &x).unwrap();
        for (i, &xi) in x.data.iter().enumerate() {
            let want = nn::quick_gelu_scalar(xi);
            assert!(
                (out.data[i] - want).abs() < 1e-5,
                "ff[{i}]={} != quick_gelu({xi})={want}",
                out.data[i]
            );
        }
    }

    // ── full tower forward_with ─────────────────────────────────────────────

    fn tiny_weights(cfg: &ClipConfig, num_patches: usize) -> ClipWeights {
        let dim = cfg.hidden_size;
        let num_positions = num_patches + 1;
        ClipWeights {
            class_embedding: vec![0.0; dim],
            position_embedding: vec![0.0; num_positions * dim],
            num_positions,
            pre_layernorm: ln_identity(dim),
            blocks: (0..cfg.num_layers)
                .map(|_| block_identity(dim, cfg.ffn_hidden_size))
                .collect(),
        }
    }

    #[test]
    fn forward_with_produces_class_plus_patches_rows() {
        let cfg = tiny_cfg();
        let num_patches = 4; // 2x2 grid -> square, no interpolation needed
        let w = tiny_weights(&cfg, num_patches);
        let sam = Mat::from_vec(
            num_patches,
            cfg.hidden_size,
            (0..num_patches * cfg.hidden_size)
                .map(|i| (i as f32) * 0.01)
                .collect(),
        );
        let out = forward_with(&cfg, &w, &sam).unwrap();
        // class token row + num_patches rows.
        assert_eq!(out.shape(), (num_patches + 1, cfg.hidden_size));
        assert!(out.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_with_rejects_wrong_width() {
        let cfg = tiny_cfg();
        let w = tiny_weights(&cfg, 4);
        let sam = Mat::zeros(4, cfg.hidden_size + 1); // wrong width
        assert!(forward_with(&cfg, &w, &sam).is_err());
    }

    #[test]
    fn forward_with_rejects_wrong_block_count() {
        let cfg = tiny_cfg();
        let mut w = tiny_weights(&cfg, 4);
        w.blocks.pop(); // now num_layers-1 blocks
        let sam = Mat::zeros(4, cfg.hidden_size);
        assert!(forward_with(&cfg, &w, &sam).is_err());
    }

    /// Residual identity check: with all-zero norms-gamma... not applicable;
    /// instead confirm the block residual structure by giving a block whose
    /// attention and mlp both produce ~0 (zero out_proj / zero fc2) so the block
    /// is the identity map x -> x.
    #[test]
    fn zeroed_sublayers_make_block_identity() {
        let cfg = tiny_cfg();
        let dim = cfg.hidden_size;
        let mut block = block_identity(dim, cfg.ffn_hidden_size);
        // Zero out_proj weight+bias -> attention contributes 0.
        block.out_proj.weight.iter_mut().for_each(|v| *v = 0.0);
        block.out_proj.bias = Some(vec![0.0; dim]);
        // Zero fc2 weight+bias -> mlp contributes 0.
        block.fc2.weight.iter_mut().for_each(|v| *v = 0.0);
        block.fc2.bias = Some(vec![0.0; dim]);
        let x = Mat::from_vec(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let out = transformer_block(&cfg, &block, &x).unwrap();
        for (o, i) in out.data.iter().zip(&x.data) {
            assert!((o - i).abs() < 1e-5, "block not identity: {o} vs {i}");
        }
    }

    #[test]
    fn isqrt_is_floor_sqrt() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(3), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(255), 15);
        assert_eq!(isqrt(256), 16);
        assert_eq!(isqrt(257), 16);
    }
}
