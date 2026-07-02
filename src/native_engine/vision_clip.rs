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

fn checked_shape_mul(context: &str, lhs: usize, rhs: usize, expression: &str) -> FocrResult<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "{context}: usize overflow computing {expression} ({lhs} * {rhs})"
        ))
    })
}

fn checked_shape_sub(context: &str, lhs: usize, rhs: usize, expression: &str) -> FocrResult<usize> {
    lhs.checked_sub(rhs).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "{context}: usize underflow computing {expression} ({lhs} - {rhs})"
        ))
    })
}

fn checked_shape_add(context: &str, lhs: usize, rhs: usize, expression: &str) -> FocrResult<usize> {
    lhs.checked_add(rhs).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "{context}: usize overflow computing {expression} ({lhs} + {rhs})"
        ))
    })
}

fn ensure_mat_data_len(mat: &Mat, context: &str) -> FocrResult<()> {
    let expected_len = checked_shape_mul(context, mat.rows, mat.cols, "rows*cols")?;
    if mat.data.len() == expected_len {
        return Ok(());
    }
    Err(FocrError::Other(anyhow::anyhow!(
        "{context}: data len {} != rows*cols {}",
        mat.data.len(),
        expected_len
    )))
}

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
/// This is the in-memory view the `.focrq` loader (bd-1es.3) hydrates; the
/// public [`forward`] entrypoint pulls these out of [`Weights`] through that
/// reader's named-tensor accessors. Keeping it explicit lets the tower be
/// exercised end-to-end in unit tests with no model present.
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
/// This is the [`Weights`]-backed entrypoint the engine wires: the `.focrq`
/// reader (bd-1es.3) exposes typed tensor accessors, so it hydrates a
/// [`ClipWeights`] from [`Weights`] and runs the real math in [`forward_with`],
/// which is fully exercised by the unit tests below.
///
/// # Errors
/// Propagates accessor errors from building the [`ClipWeights`] (e.g. a missing
/// or mis-shaped `model.vision_model.*` tensor) and whatever [`forward_with`]
/// returns.
pub fn forward(weights: &Weights, _image: &Mat, sam_features: &Mat) -> FocrResult<Mat> {
    let cfg = ClipConfig::default();
    ensure_mat_data_len(sam_features, "vision_clip forward sam_features")?;
    // SAM `x3` is [OUT_CH, num_patches] channel-major; CLIP consumes
    // [num_patches, hidden] (flatten(2).transpose(1,2)), so transpose it.
    let sam_t = transpose(&sam_features.data, sam_features.rows, sam_features.cols)?;
    let sam_mat = Mat::from_vec(sam_features.cols, sam_features.rows, sam_t);
    let cw = clip_weights_from(weights)?;
    forward_with(&cfg, &cw, &sam_mat)
}

/// Build a [`ClipWeights`] from the `model.vision_model.*` tensors (note the
/// preserved upstream `pre_layrnorm` typo). Dims read from each tensor shape.
/// `pub(crate)` so the batch spine (bd-1azu.10) hydrates ONCE per batch.
pub(crate) fn clip_weights_from(weights: &Weights) -> FocrResult<ClipWeights> {
    let p = "model.vision_model";
    let ln = |n: &str| -> FocrResult<LayerNormParams> {
        Ok(LayerNormParams {
            weight: weights.vec(&format!("{n}.weight"))?,
            bias: weights.vec(&format!("{n}.bias"))?,
        })
    };
    let lin = |n: &str| -> FocrResult<LinearParams> {
        let weight_name = format!("{n}.weight");
        let (out_features, in_features) = tensor_rank2_shape(weights, &weight_name)?;
        Ok(LinearParams {
            weight: weights.vec(&weight_name)?,
            bias: Some(weights.vec(&format!("{n}.bias"))?),
            out_features,
            in_features,
        })
    };
    let cfg = ClipConfig::default();
    let pos_name = format!("{p}.embeddings.position_embedding.weight");
    let (num_positions, _pos_dim) = tensor_rank2_shape(weights, &pos_name)?;
    let mut blocks = Vec::with_capacity(cfg.num_layers);
    for l in 0..cfg.num_layers {
        let b = format!("{p}.transformer.layers.{l}");
        blocks.push(ClipBlockWeights {
            layer_norm1: ln(&format!("{b}.layer_norm1"))?,
            qkv_proj: lin(&format!("{b}.self_attn.qkv_proj"))?,
            out_proj: lin(&format!("{b}.self_attn.out_proj"))?,
            layer_norm2: ln(&format!("{b}.layer_norm2"))?,
            fc1: lin(&format!("{b}.mlp.fc1"))?,
            fc2: lin(&format!("{b}.mlp.fc2"))?,
        });
    }
    Ok(ClipWeights {
        class_embedding: weights.vec(&format!("{p}.embeddings.class_embedding"))?,
        position_embedding: weights.vec(&pos_name)?,
        num_positions,
        pre_layernorm: ln(&format!("{p}.pre_layrnorm"))?,
        blocks,
    })
}

fn tensor_rank2_shape(weights: &Weights, name: &str) -> FocrResult<(usize, usize)> {
    let view = weights.tensor(name)?;
    let [rows, cols] = view.shape else {
        return Err(FocrError::FormatMismatch(format!(
            "tensor {name:?} has rank {}; expected 2 ([rows, cols])",
            view.shape.len()
        )));
    };
    Ok((*rows, *cols))
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
    ensure_mat_data_len(sam_features, "vision_clip forward_with sam_features")?;
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
    let mut x = prepend_class_token(&weights.class_embedding, sam_features)?;
    let seq = x.rows; // num_patches + 1
    let pos = abs_pos_for_len(&weights.position_embedding, weights.num_positions, dim, seq)?;
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

/// Batched CLIP tower over `V` views in ONE forward (bd-1azu.10).
///
/// Each view's `sam_features` is `[N, hidden]`; the views are stacked along a
/// leading batch dimension and every stage runs over the `[V·(N+1), dim]` buffer
/// at once. Per-view independence is preserved EXACTLY: the LayerNorm, linear,
/// FFN, pos-embed and class-token steps are row-wise (stacking is a no-op on the
/// math), and attention runs with `num_bh = V·heads` so each view's heads attend
/// ONLY within that view's `seq` tokens (block-diagonal — never across views).
///
/// This is byte-identical to calling [`forward_with`] per view: the f32 GEMMs are
/// M-independent (per output row the K-reduction is the same regardless of how
/// many rows are stacked) and `nn::sdpa` processes each `bh` block in isolation.
/// The single-forward shape is what the continuous-batch spine needs to defeat
/// the per-page Amdahl vision wall (Doctrine #5: one live forward).
///
/// Returns one `[N+1, hidden]` matrix per view (class token at row 0), matching
/// [`forward_with`]'s per-view contract.
///
/// # Errors
/// [`FocrError::Other`] on an empty batch, a ragged batch (views of differing
/// `N`), or any width/weight mismatch.
pub fn forward_with_batched(
    cfg: &ClipConfig,
    weights: &ClipWeights,
    sam_features_per_view: &[&Mat],
) -> FocrResult<Vec<Mat>> {
    let v = sam_features_per_view.len();
    if v == 0 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip forward_with_batched: empty view batch"
        )));
    }
    let dim = cfg.hidden_size;
    let n = sam_features_per_view[0].rows;
    for (i, sf) in sam_features_per_view.iter().enumerate() {
        if sf.cols != dim {
            return Err(FocrError::Other(anyhow::anyhow!(
                "vision_clip forward_with_batched: view {i} width {} != hidden_size {dim}",
                sf.cols
            )));
        }
        if sf.rows != n {
            return Err(FocrError::Other(anyhow::anyhow!(
                "vision_clip forward_with_batched: view {i} rows {} != {n} (ragged batch)",
                sf.rows
            )));
        }
        ensure_mat_data_len(sf, "vision_clip forward_with_batched sam_features")?;
    }
    if weights.class_embedding.len() != dim {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip forward_with_batched: class_embedding len {} != hidden_size {dim}",
            weights.class_embedding.len()
        )));
    }
    if weights.blocks.len() != cfg.num_layers {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip forward_with_batched: {} blocks != num_layers {}",
            weights.blocks.len(),
            cfg.num_layers
        )));
    }

    let seq = checked_shape_add("vision_clip forward_with_batched", n, 1, "N+1")?;
    // Position embedding for this seq is identical across views (depends on seq).
    let pos = abs_pos_for_len(&weights.position_embedding, weights.num_positions, dim, seq)?;
    ensure_mat_shape(&pos, seq, dim, "vision_clip forward_with_batched pos")?;

    // Build the stacked [V*seq, dim] buffer: per view, class token then patches,
    // then add the (shared) pos embedding to that view's block.
    let total_rows = checked_shape_mul("vision_clip forward_with_batched", v, seq, "V*seq")?;
    let stacked_len = checked_shape_mul(
        "vision_clip forward_with_batched",
        total_rows,
        dim,
        "V*seq*dim",
    )?;
    let mut data = vec![0.0f32; stacked_len];
    for (vv, sf) in sam_features_per_view.iter().enumerate() {
        let base = vv * seq * dim;
        data[base..base + dim].copy_from_slice(&weights.class_embedding);
        data[base + dim..base + seq * dim].copy_from_slice(&sf.data);
        for s in 0..seq {
            let off = base + s * dim;
            let prow = &pos.data[s * dim..(s + 1) * dim];
            for (d, pv) in prow.iter().enumerate() {
                data[off + d] += *pv;
            }
        }
    }
    let mut x = Mat::from_vec(total_rows, dim, data);

    // pre_layrnorm (row-wise -> identical to per-view).
    x = nn::layer_norm(
        &x,
        Some(&weights.pre_layernorm.weight),
        Some(&weights.pre_layernorm.bias),
        cfg.pre_layernorm_eps,
    )?;

    for block in &weights.blocks {
        x = transformer_block_batched(cfg, block, &x, v, seq)?;
    }

    // Split the [V*seq, dim] buffer back into V per-view [seq, dim] matrices.
    let mut out = Vec::with_capacity(v);
    for vv in 0..v {
        let base = vv * seq * dim;
        out.push(Mat::from_vec(
            seq,
            dim,
            x.data[base..base + seq * dim].to_vec(),
        ));
    }
    Ok(out)
}

/// [`forward_with_batched`] fed directly with raw SAM `x3` outputs
/// (`[OUT_CH, num_patches]` channel-major, exactly what [`super::vision_sam`]
/// produces): applies to each view the SAME `flatten(2).transpose(1,2)`
/// transpose [`forward`] performs, then runs the batched stack. Keeps the
/// SAM→CLIP layout knowledge inside this module so the batch spine
/// (bd-1azu.10) cannot drift from the per-view path.
///
/// # Errors
/// A transpose shape rejection, or anything [`forward_with_batched`] returns.
pub(crate) fn forward_batched_from_sam(
    cfg: &ClipConfig,
    weights: &ClipWeights,
    sam_features_per_view: &[&Mat],
) -> FocrResult<Vec<Mat>> {
    let mut transposed = Vec::with_capacity(sam_features_per_view.len());
    for sf in sam_features_per_view {
        ensure_mat_data_len(sf, "vision_clip forward_batched_from_sam sam_features")?;
        let t = transpose(&sf.data, sf.rows, sf.cols)?;
        transposed.push(Mat::from_vec(sf.cols, sf.rows, t));
    }
    let refs: Vec<&Mat> = transposed.iter().collect();
    forward_with_batched(cfg, weights, &refs)
}

/// Batched analogue of [`transformer_block`]: every stage is row-wise except the
/// attention, which uses [`self_attention_batched`] (`num_bh = V·heads`).
fn transformer_block_batched(
    cfg: &ClipConfig,
    w: &ClipBlockWeights,
    x: &Mat,
    v: usize,
    seq: usize,
) -> FocrResult<Mat> {
    let normed = nn::layer_norm(
        x,
        Some(&w.layer_norm1.weight),
        Some(&w.layer_norm1.bias),
        cfg.layernorm_eps,
    )?;
    let attn = self_attention_batched(cfg, w, &normed, v, seq)?;
    let h = add(x, &attn)?;

    let normed2 = nn::layer_norm(
        &h,
        Some(&w.layer_norm2.weight),
        Some(&w.layer_norm2.bias),
        cfg.layernorm_eps,
    )?;
    let mlp = feed_forward(w, &normed2)?;
    add(&h, &mlp)
}

/// Batched analogue of [`self_attention`] over a `[V*seq, dim]` buffer. The qkv
/// and out projections are M-batched (row-independent); the SDPA runs with
/// `num_bh = V·heads` where head block `view*heads + h` covers view `view`'s
/// `seq` tokens, so attention is strictly block-diagonal across views — byte
/// identical to running [`self_attention`] on each view separately.
fn self_attention_batched(
    cfg: &ClipConfig,
    w: &ClipBlockWeights,
    x: &Mat,
    v: usize,
    seq: usize,
) -> FocrResult<Mat> {
    let dim = cfg.hidden_size;
    let heads = cfg.num_heads;
    if heads == 0 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip self_attention_batched: num_heads must be non-zero"
        )));
    }
    if dim == 0 || !dim.is_multiple_of(heads) {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip self_attention_batched: hidden_size {dim} must be non-zero and divisible by heads {heads}"
        )));
    }
    let rows = checked_shape_mul("vision_clip self_attention_batched", v, seq, "V*seq")?;
    if x.rows != rows {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip self_attention_batched: x.rows {} != V*seq {rows}",
            x.rows
        )));
    }
    if x.cols != dim {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip self_attention_batched: x.cols {} != hidden_size {dim}",
            x.cols
        )));
    }
    ensure_mat_data_len(x, "vision_clip self_attention_batched input")?;
    let hd = dim / heads;
    let three_dim = checked_shape_mul("vision_clip self_attention_batched", 3, dim, "3*dim")?;
    let head_span = checked_shape_mul("vision_clip self_attention_batched", seq, hd, "seq*hd")?;
    let num_bh = checked_shape_mul("vision_clip self_attention_batched", v, heads, "V*heads")?;
    let buf_len = checked_shape_mul(
        "vision_clip self_attention_batched",
        num_bh,
        head_span,
        "V*heads*seq*hd",
    )?;

    // Fused qkv projection over the whole stacked buffer -> [V*seq, 3*dim].
    let qkv = linear(&w.qkv_proj, x)?;
    ensure_mat_shape(
        &qkv,
        rows,
        three_dim,
        "vision_clip self_attention_batched qkv",
    )?;

    // Repack into head-major [num_bh=V*heads, seq, hd] buffers, view-major.
    let mut q = vec![0.0f32; buf_len];
    let mut k = vec![0.0f32; buf_len];
    let mut vbuf = vec![0.0f32; buf_len];
    for view in 0..v {
        for s in 0..seq {
            let row = qkv.row(view * seq + s);
            for hh in 0..heads {
                let bh = view * heads + hh;
                for d in 0..hd {
                    let q_src = hh * hd + d;
                    let k_src = dim + hh * hd + d;
                    let v_src = 2 * dim + hh * hd + d;
                    let dst = bh * head_span + s * hd + d;
                    q[dst] = row[q_src];
                    k[dst] = row[k_src];
                    vbuf[dst] = row[v_src];
                }
            }
        }
    }

    let scale = 1.0f32 / (hd as f32).sqrt();
    let ctx = nn::sdpa(&q, &k, &vbuf, num_bh, seq, seq, hd, hd, scale, false);
    if ctx.len() != buf_len {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip self_attention_batched: sdpa context len {} != expected {buf_len}",
            ctx.len()
        )));
    }

    // Repack [num_bh, seq, hd] -> [V*seq, dim].
    let merged_len =
        checked_shape_mul("vision_clip self_attention_batched", rows, dim, "V*seq*dim")?;
    let mut merged = Mat::from_vec(rows, dim, vec![0.0f32; merged_len]);
    for view in 0..v {
        for hh in 0..heads {
            let bh = view * heads + hh;
            for s in 0..seq {
                for d in 0..hd {
                    let src = bh * head_span + s * hd + d;
                    let dst = (view * seq + s) * dim + hh * hd + d;
                    merged.data[dst] = ctx[src];
                }
            }
        }
    }

    linear(&w.out_proj, &merged)
}

/// One `NoTPTransformerBlock` ([SPEC-049], `deepencoder.py:392-396`):
/// `h = x + attn(LN1(x)); out = h + mlp(LN2(h))`.
fn transformer_block(cfg: &ClipConfig, w: &ClipBlockWeights, x: &Mat) -> FocrResult<Mat> {
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
fn self_attention(cfg: &ClipConfig, w: &ClipBlockWeights, x: &Mat) -> FocrResult<Mat> {
    let dim = cfg.hidden_size;
    let heads = cfg.num_heads;
    let seq = x.rows;
    if heads == 0 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip self_attention: num_heads must be non-zero"
        )));
    }
    if dim == 0 || !dim.is_multiple_of(heads) {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip self_attention: hidden_size {} must be non-zero and divisible by heads {}",
            dim,
            heads
        )));
    }
    if x.cols != dim {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip self_attention: x.cols {} != hidden_size {}",
            x.cols,
            dim
        )));
    }
    ensure_mat_data_len(x, "vision_clip self_attention input")?;
    let hd = dim / heads;
    let three_dim = checked_shape_mul("vision_clip self_attention", 3, dim, "3*hidden_size")?;
    let head_span = checked_shape_mul("vision_clip self_attention", seq, hd, "seq*head_dim")?;
    let qkv_buffer_len = checked_shape_mul(
        "vision_clip self_attention",
        heads,
        head_span,
        "heads*seq*head_dim",
    )?;

    // Fused qkv projection -> [seq, 3*dim].
    let qkv = linear(&w.qkv_proj, x)?;
    ensure_mat_shape(
        &qkv,
        seq,
        three_dim,
        "vision_clip self_attention qkv output",
    )?;

    // Repack into head-major flat buffers [heads, seq, hd] for q, k, v.
    // PyTorch view: xqkv.view(bsz, seq, 3, heads, hd) — so within a row the
    // layout is [t (3), head, hd]: column index = t*dim + head*hd + d.
    let mut q = vec![0.0f32; qkv_buffer_len];
    let mut k = vec![0.0f32; qkv_buffer_len];
    let mut v = vec![0.0f32; qkv_buffer_len];
    for s in 0..seq {
        let row = qkv.row(s);
        for hh in 0..heads {
            for d in 0..hd {
                // Column index within the [t(3), head, hd] fused row.
                let q_src = hh * hd + d;
                let k_src = dim + hh * hd + d;
                let v_src = 2 * dim + hh * hd + d;
                let dst = hh * head_span + s * hd + d;
                q[dst] = row[q_src];
                k[dst] = row[k_src];
                v[dst] = row[v_src];
            }
        }
    }

    // Full (non-causal) SDPA. num_bh = batch(1) * heads.
    let scale = 1.0f32 / (hd as f32).sqrt();
    let ctx = nn::sdpa(&q, &k, &v, heads, seq, seq, hd, hd, scale, false);
    let expected_ctx_len = qkv_buffer_len;
    if ctx.len() != expected_ctx_len {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip self_attention: sdpa context len {} != expected {}",
            ctx.len(),
            expected_ctx_len
        )));
    }

    // Repack [heads, seq, hd] -> [seq, dim] (permute(0,2,1,3).reshape).
    let merged_len = checked_shape_mul("vision_clip self_attention", seq, dim, "seq*hidden_size")?;
    let mut merged = Mat::from_vec(seq, dim, vec![0.0f32; merged_len]);
    for hh in 0..heads {
        for s in 0..seq {
            for d in 0..hd {
                let src = hh * head_span + s * hd + d;
                let dst = s * dim + hh * hd + d;
                merged.data[dst] = ctx[src];
            }
        }
    }

    // out_proj.
    let out = linear(&w.out_proj, &merged)?;
    ensure_mat_shape(
        &out,
        seq,
        dim,
        "vision_clip self_attention projection output",
    )?;
    Ok(out)
}

fn ensure_mat_shape(mat: &Mat, rows: usize, cols: usize, context: &str) -> FocrResult<()> {
    if mat.rows == rows && mat.cols == cols {
        return Ok(());
    }
    Err(FocrError::Other(anyhow::anyhow!(
        "{context}: shape {:?} != expected ({rows}, {cols})",
        mat.shape()
    )))
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
    ensure_mat_data_len(x, "vision_clip linear input")?;
    let expected_weight_len = checked_shape_mul(
        "vision_clip linear",
        w.out_features,
        w.in_features,
        "out*in",
    )?;
    if w.weight.len() != expected_weight_len {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip linear: weight len {} != out*in {}",
            w.weight.len(),
            expected_weight_len
        )));
    }
    let wt = transpose(&w.weight, w.out_features, w.in_features)?;
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
fn transpose(src: &[f32], rows: usize, cols: usize) -> FocrResult<Vec<f32>> {
    let expected_len = checked_shape_mul("vision_clip transpose", rows, cols, "rows*cols")?;
    if src.len() != expected_len {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip transpose: source len {} != rows*cols {}",
            src.len(),
            expected_len
        )));
    }
    let mut out = vec![0.0f32; expected_len];
    for c in 0..cols {
        let dst = &mut out[c * rows..(c + 1) * rows];
        for (r, slot) in dst.iter_mut().enumerate() {
            *slot = src[r * cols + c];
        }
    }
    Ok(out)
}

/// Prepend the class token as a new row 0, returning a `[patches+1, dim]` matrix.
fn prepend_class_token(class_embedding: &[f32], patches: &Mat) -> FocrResult<Mat> {
    let dim = patches.cols;
    if class_embedding.len() != dim {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip class token: class_embedding len {} != patch dim {}",
            class_embedding.len(),
            dim
        )));
    }
    ensure_mat_data_len(patches, "vision_clip class token patches")?;
    let seq = checked_shape_add("vision_clip class token", patches.rows, 1, "patch_rows+1")?;
    let out_len = checked_shape_mul("vision_clip class token", seq, dim, "seq*dim")?;
    let mut data = Vec::with_capacity(out_len);
    data.extend_from_slice(class_embedding);
    data.extend_from_slice(&patches.data);
    Ok(Mat::from_vec(seq, dim, data))
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
fn abs_pos_for_len(table: &[f32], num_positions: usize, dim: usize, seq: usize) -> FocrResult<Mat> {
    let expected_table_len = checked_shape_mul(
        "vision_clip abs_pos",
        num_positions,
        dim,
        "num_positions*dim",
    )?;
    if table.len() != expected_table_len {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip abs_pos: table len {} != num_positions*dim {}",
            table.len(),
            expected_table_len
        )));
    }
    if num_positions == 0 || seq == 0 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip abs_pos: num_positions ({num_positions}) and seq ({seq}) must be non-zero"
        )));
    }
    if seq == num_positions {
        return Ok(Mat::from_vec(seq, dim, table.to_vec()));
    }
    let num_patches =
        checked_shape_sub("vision_clip abs_pos", num_positions, 1, "num_positions-1")?;
    let runtime_patches = checked_shape_sub("vision_clip abs_pos", seq, 1, "seq-1")?;
    let src = isqrt(num_patches);
    let tgt = isqrt(runtime_patches);
    let src_square = checked_shape_mul("vision_clip abs_pos", src, src, "src*src")?;
    let tgt_square = checked_shape_mul("vision_clip abs_pos", tgt, tgt, "tgt*tgt")?;
    if src_square != num_patches || tgt_square != runtime_patches {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_clip abs_pos: non-square grids (src²={}, num_patches={}, tgt²={}, runtime_patches={})",
            src_square,
            num_patches,
            tgt_square,
            runtime_patches
        )));
    }

    let out_len = checked_shape_mul("vision_clip abs_pos", seq, dim, "seq*dim")?;
    let mut out = vec![0.0f32; out_len];
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
    ensure_mat_data_len(a, "vision_clip add lhs")?;
    ensure_mat_data_len(b, "vision_clip add rhs")?;
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
    ensure_mat_data_len(a, "vision_clip add_in_place lhs")?;
    ensure_mat_data_len(b, "vision_clip add_in_place rhs")?;
    for (x, y) in a.data.iter_mut().zip(&b.data) {
        *x += *y;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::focrq::{FocrqBuilder, WriteDType};
    use half::bf16;

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

    fn bf16_zeros(n: usize) -> Vec<u8> {
        (0..n)
            .flat_map(|_| bf16::from_f32(0.0).to_le_bytes())
            .collect()
    }

    // ── weight hydration error paths ───────────────────────────────────────

    #[test]
    fn clip_weights_from_rejects_rank1_linear_weight_without_panic() {
        let p = "model.vision_model";
        let layer0 = format!("{p}.transformer.layers.0");
        let mut b = FocrqBuilder::new();
        b.add_tensor(
            format!("{p}.embeddings.position_embedding.weight"),
            WriteDType::Bf16,
            vec![1, 1],
            bf16_zeros(1),
        )
        .unwrap();
        b.add_tensor(
            format!("{layer0}.layer_norm1.weight"),
            WriteDType::Bf16,
            vec![1],
            bf16_zeros(1),
        )
        .unwrap();
        b.add_tensor(
            format!("{layer0}.layer_norm1.bias"),
            WriteDType::Bf16,
            vec![1],
            bf16_zeros(1),
        )
        .unwrap();
        b.add_tensor(
            format!("{layer0}.self_attn.qkv_proj.weight"),
            WriteDType::Bf16,
            vec![4],
            bf16_zeros(4),
        )
        .unwrap();
        let weights = Weights::from_bytes(b.build()).unwrap();
        assert_err_contains(clip_weights_from(&weights), "rank 1");
    }

    #[test]
    fn clip_weights_from_rejects_rank1_position_embedding_without_panic() {
        let p = "model.vision_model";
        let mut b = FocrqBuilder::new();
        b.add_tensor(
            format!("{p}.embeddings.position_embedding.weight"),
            WriteDType::Bf16,
            vec![4],
            bf16_zeros(4),
        )
        .unwrap();
        let weights = Weights::from_bytes(b.build()).unwrap();
        assert_err_contains(clip_weights_from(&weights), "rank 1");
    }

    // ── transpose / linear ─────────────────────────────────────────────────

    #[test]
    fn transpose_roundtrips() -> FocrResult<()> {
        // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = transpose(&src, 2, 3)?;
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(transpose(&t, 3, 2)?, src);
        Ok(())
    }

    #[test]
    fn transpose_rejects_malformed_len_without_panic() {
        assert_err_contains(transpose(&[1.0], 2, 2), "source len");
    }

    #[test]
    fn transpose_rejects_shape_product_overflow_without_panic() {
        assert_err_contains(transpose(&[], usize::MAX, 2), "rows*cols");
    }

    #[test]
    fn linear_applies_weight_and_bias() -> FocrResult<()> {
        // W = [[1,0,0],[0,1,0]] (out=2,in=3), b=[10,20]; x=[1,2,3] -> [11,22].
        let w = LinearParams {
            weight: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            bias: Some(vec![10.0, 20.0]),
            out_features: 2,
            in_features: 3,
        };
        let x = Mat::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        let y = linear(&w, &x)?;
        assert_eq!(y.shape(), (1, 2));
        assert!((y.data[0] - 11.0).abs() < 1e-6);
        assert!((y.data[1] - 22.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn linear_rejects_dim_mismatch() {
        let w = linear_identity(4);
        let x = Mat::zeros(2, 3); // 3 != 4
        assert!(linear(&w, &x).is_err());
    }

    #[test]
    fn linear_rejects_weight_shape_product_overflow_without_panic() {
        let w = LinearParams {
            weight: Vec::new(),
            bias: None,
            out_features: usize::MAX,
            in_features: 2,
        };
        let x = Mat::zeros(1, 2);
        assert_err_contains(linear(&w, &x), "out*in");
    }

    #[test]
    fn linear_rejects_malformed_input_mat_without_panic() {
        let w = LinearParams {
            weight: vec![1.0, 1.0],
            bias: None,
            out_features: 1,
            in_features: 2,
        };
        let x = Mat {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0],
        };
        assert_err_contains(linear(&w, &x), "data len");
    }

    // ── class token + abs pos ──────────────────────────────────────────────

    #[test]
    fn class_token_prepended_as_row0() -> FocrResult<()> {
        let patches = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = prepend_class_token(&[7.0, 8.0, 9.0], &patches)?;
        assert_eq!(out.shape(), (3, 3));
        assert_eq!(out.row(0), &[7.0, 8.0, 9.0]);
        assert_eq!(out.row(1), &[1.0, 2.0, 3.0]);
        assert_eq!(out.row(2), &[4.0, 5.0, 6.0]);
        Ok(())
    }

    #[test]
    fn prepend_class_token_rejects_bad_class_len_without_panic() {
        let patches = Mat::zeros(2, 3);
        assert_err_contains(
            prepend_class_token(&[1.0, 2.0], &patches),
            "class_embedding len",
        );
    }

    #[test]
    fn prepend_class_token_rejects_malformed_patch_data_without_panic() {
        let patches = Mat {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0],
        };
        assert_err_contains(prepend_class_token(&[7.0, 8.0, 9.0], &patches), "data len");
    }

    #[test]
    fn prepend_class_token_rejects_seq_overflow_without_panic() {
        let patches = Mat {
            rows: usize::MAX,
            cols: 0,
            data: Vec::new(),
        };
        assert_err_contains(prepend_class_token(&[], &patches), "patch_rows+1");
    }

    #[test]
    fn abs_pos_passthrough_when_lengths_match() -> FocrResult<()> {
        // num_positions = 5 (1 CLS + 2x2 grid), dim 2, seq 5 -> identity.
        let table: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let pos = abs_pos_for_len(&table, 5, 2, 5)?;
        assert_eq!(pos.shape(), (5, 2));
        assert_eq!(pos.data, table);
        Ok(())
    }

    #[test]
    fn abs_pos_same_grid_size_is_identity_even_via_resize_branch() -> FocrResult<()> {
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
        let pos = abs_pos_for_len(&table, np, dim, 5)?;
        assert_eq!(pos.shape(), (5, 1));
        // CLS row preserved exactly.
        assert!((pos.data[0] - 99.0).abs() < 1e-6);
        // Interpolated values are convex-ish combinations within source range.
        for &v in &pos.data[1..] {
            assert!(
                (0.5..=10.0).contains(&v),
                "interp out of expected range: {v}"
            );
        }
        Ok(())
    }

    #[test]
    fn abs_pos_rejects_non_square_grid() {
        // num_positions-1 = 3 is not a perfect square.
        let table = vec![0.0f32; 4 * 2];
        assert!(abs_pos_for_len(&table, 4, 2, 5).is_err());
    }

    #[test]
    fn abs_pos_rejects_shape_product_overflow_without_panic() {
        assert_err_contains(abs_pos_for_len(&[], usize::MAX, 2, 1), "num_positions*dim");
    }

    #[test]
    fn abs_pos_rejects_zero_lengths_without_underflow() {
        assert_err_contains(abs_pos_for_len(&[], 0, 2, 1), "non-zero");

        let table = vec![0.0f32; 2];
        assert_err_contains(abs_pos_for_len(&table, 1, 2, 0), "non-zero");
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
    fn self_attention_preserves_shape() -> FocrResult<()> {
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
        let out = self_attention(&cfg, &block, &x)?;
        assert_eq!(out.shape(), (3, 4));
        Ok(())
    }

    #[test]
    fn self_attention_rejects_bad_head_config_without_panic() {
        let mut cfg = tiny_cfg();
        let block = block_identity(cfg.hidden_size, cfg.ffn_hidden_size);
        let x = Mat::zeros(2, cfg.hidden_size);

        cfg.num_heads = 0;
        assert_err_contains(self_attention(&cfg, &block, &x), "num_heads");

        cfg.num_heads = 3;
        assert_err_contains(self_attention(&cfg, &block, &x), "divisible");
    }

    #[test]
    fn self_attention_rejects_malformed_qkv_and_projection_shapes() {
        let cfg = tiny_cfg();
        let dim = cfg.hidden_size;
        let x = Mat::zeros(3, dim);

        let mut bad_qkv = block_identity(dim, cfg.ffn_hidden_size);
        bad_qkv.qkv_proj.out_features = 3 * dim - 1;
        bad_qkv.qkv_proj.weight =
            vec![0.0; bad_qkv.qkv_proj.out_features * bad_qkv.qkv_proj.in_features];
        bad_qkv.qkv_proj.bias = Some(vec![0.0; bad_qkv.qkv_proj.out_features]);
        assert_err_contains(
            self_attention(&cfg, &bad_qkv, &x),
            "self_attention qkv output",
        );

        let mut bad_proj = block_identity(dim, cfg.ffn_hidden_size);
        bad_proj.out_proj.out_features = dim - 1;
        bad_proj.out_proj.weight =
            vec![0.0; bad_proj.out_proj.out_features * bad_proj.out_proj.in_features];
        bad_proj.out_proj.bias = Some(vec![0.0; bad_proj.out_proj.out_features]);
        assert_err_contains(
            self_attention(&cfg, &bad_proj, &x),
            "self_attention projection output",
        );
    }

    #[test]
    fn self_attention_rejects_malformed_input_mat_without_panic() {
        let cfg = tiny_cfg();
        let block = block_identity(cfg.hidden_size, cfg.ffn_hidden_size);
        let x = Mat {
            rows: 2,
            cols: cfg.hidden_size,
            data: vec![0.0; cfg.hidden_size],
        };
        assert_err_contains(self_attention(&cfg, &block, &x), "data len");
    }

    /// With q=k=v=x and identity out_proj, attention is a softmax-weighted
    /// average of the value rows — every output row stays inside the convex hull
    /// of the inputs (each column bounded by per-column min/max).
    #[test]
    fn self_attention_is_convex_average_of_values() -> FocrResult<()> {
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
        let out = self_attention(&cfg, &block, &x)?;
        for c in 0..4 {
            let col_min = (0..3).map(|r| x.get(r, c)).fold(f32::INFINITY, f32::min);
            let col_max = (0..3)
                .map(|r| x.get(r, c))
                .fold(f32::NEG_INFINITY, f32::max);
            for r in 0..3 {
                let v = out.get(r, c);
                assert!(
                    v >= col_min - 1e-4 && v <= col_max + 1e-4,
                    "out[{r},{c}]={v} outside [{col_min},{col_max}]"
                );
            }
        }
        Ok(())
    }

    // ── feed-forward = quick_gelu sandwiched in identities ─────────────────

    #[test]
    fn feed_forward_applies_quick_gelu() -> FocrResult<()> {
        let cfg = tiny_cfg();
        let block = block_identity(cfg.hidden_size, cfg.ffn_hidden_size);
        // fc1=I (padded), fc2=I (padded) => ff(x) == quick_gelu(x) elementwise.
        let x = Mat::from_vec(1, 4, vec![-1.0, 0.0, 1.0, 2.0]);
        let out = feed_forward(&block, &x)?;
        for (i, &xi) in x.data.iter().enumerate() {
            let want = nn::quick_gelu_scalar(xi);
            assert!(
                (out.data[i] - want).abs() < 1e-5,
                "ff[{i}]={} != quick_gelu({xi})={want}",
                out.data[i]
            );
        }
        Ok(())
    }

    #[test]
    fn add_rejects_malformed_backing_data_without_panic() {
        let malformed_lhs = Mat {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0],
        };
        let good = Mat::zeros(2, 2);
        assert_err_contains(add(&malformed_lhs, &good), "add lhs");

        let malformed_rhs = Mat {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0],
        };
        assert_err_contains(add(&good, &malformed_rhs), "add rhs");
    }

    #[test]
    fn add_in_place_rejects_malformed_backing_data_before_mutating() {
        let mut malformed_lhs = Mat {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0],
        };
        let good = Mat::zeros(2, 2);
        let lhs_before = malformed_lhs.data.clone();
        assert_err_contains(add_in_place(&mut malformed_lhs, &good), "add_in_place lhs");
        assert_eq!(malformed_lhs.data, lhs_before);

        let mut good_lhs = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let malformed_rhs = Mat {
            rows: 2,
            cols: 2,
            data: vec![10.0, 20.0],
        };
        let good_before = good_lhs.data.clone();
        assert_err_contains(
            add_in_place(&mut good_lhs, &malformed_rhs),
            "add_in_place rhs",
        );
        assert_eq!(good_lhs.data, good_before);
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

    // ── bd-1azu.10: batched-CLIP parity (model-free, byte-exact) ────────────
    // Deterministic non-trivial weights so a cross-view attention leak OR an
    // M-dependent GEMM would change the output and fail the bit-exact assert.
    fn det(i: usize, salt: usize) -> f32 {
        let x = (i as u64)
            .wrapping_mul(2_654_435_761)
            .wrapping_add((salt as u64).wrapping_mul(40_503))
            % 1000;
        (x as f32) / 1000.0 - 0.5
    }
    fn rand_lin(out: usize, inn: usize, salt: usize) -> LinearParams {
        LinearParams {
            weight: (0..out * inn).map(|i| det(i, salt)).collect(),
            bias: Some((0..out).map(|i| det(i, salt + 1)).collect()),
            out_features: out,
            in_features: inn,
        }
    }
    fn rand_ln(dim: usize, salt: usize) -> LayerNormParams {
        LayerNormParams {
            weight: (0..dim).map(|i| 1.0 + det(i, salt)).collect(),
            bias: (0..dim).map(|i| det(i, salt + 2)).collect(),
        }
    }
    fn rand_weights(cfg: &ClipConfig, num_patches: usize) -> ClipWeights {
        let dim = cfg.hidden_size;
        let np = num_patches + 1;
        ClipWeights {
            class_embedding: (0..dim).map(|i| det(i, 11)).collect(),
            position_embedding: (0..np * dim).map(|i| det(i, 13)).collect(),
            num_positions: np,
            pre_layernorm: rand_ln(dim, 17),
            blocks: (0..cfg.num_layers)
                .map(|l| ClipBlockWeights {
                    layer_norm1: rand_ln(dim, 100 + l * 10),
                    qkv_proj: rand_lin(3 * dim, dim, 200 + l * 10),
                    out_proj: rand_lin(dim, dim, 300 + l * 10),
                    layer_norm2: rand_ln(dim, 400 + l * 10),
                    fc1: rand_lin(cfg.ffn_hidden_size, dim, 500 + l * 10),
                    fc2: rand_lin(dim, cfg.ffn_hidden_size, 600 + l * 10),
                })
                .collect(),
        }
    }
    fn parity_cfg() -> ClipConfig {
        ClipConfig {
            num_layers: 3,
            hidden_size: 8,
            num_heads: 2,
            ffn_hidden_size: 16,
            patch_size: 14,
            layernorm_eps: 1e-5,
            pre_layernorm_eps: 1e-5,
        }
    }
    fn parity_view(num_patches: usize, dim: usize, salt: usize) -> Mat {
        Mat::from_vec(
            num_patches,
            dim,
            (0..num_patches * dim)
                .map(|i| (((i + salt * 7) as f32) * 0.013).sin())
                .collect(),
        )
    }

    #[test]
    fn batched_clip_equals_per_view_byte_for_byte() -> FocrResult<()> {
        let cfg = parity_cfg();
        let num_patches = 5;
        let w = rand_weights(&cfg, num_patches);
        let views: Vec<Mat> = (0..4)
            .map(|vi| parity_view(num_patches, cfg.hidden_size, vi))
            .collect();
        let refs: Vec<&Mat> = views.iter().collect();
        let batched = forward_with_batched(&cfg, &w, &refs)?;
        assert_eq!(batched.len(), views.len());
        for (vi, view) in views.iter().enumerate() {
            let seq_out = forward_with(&cfg, &w, view)?;
            assert_eq!(batched[vi].shape(), seq_out.shape(), "view {vi} shape");
            assert_eq!(
                batched[vi].data, seq_out.data,
                "view {vi}: batched CLIP != per-view sequential (cross-view leak or M-dependence)"
            );
        }
        Ok(())
    }

    #[test]
    fn batched_clip_single_view_equals_forward_with() -> FocrResult<()> {
        let cfg = parity_cfg();
        let num_patches = 6;
        let w = rand_weights(&cfg, num_patches);
        let view = parity_view(num_patches, cfg.hidden_size, 3);
        let batched = forward_with_batched(&cfg, &w, &[&view])?;
        let seq_out = forward_with(&cfg, &w, &view)?;
        assert_eq!(batched.len(), 1);
        assert_eq!(batched[0].data, seq_out.data);
        Ok(())
    }

    #[test]
    fn batched_from_sam_equals_per_view_forward_path() -> FocrResult<()> {
        // Feed CHANNEL-MAJOR sam-style features ([dim, num_patches], what
        // vision_sam emits) through the from-sam wrapper and assert bit-equality
        // with the per-view path (the same transpose + forward_with) — the exact
        // seam the batch spine (bd-1azu.10) rides.
        let cfg = parity_cfg();
        let num_patches = 5;
        let w = rand_weights(&cfg, num_patches);
        let sams: Vec<Mat> = (0..3)
            .map(|vi| parity_view(cfg.hidden_size, num_patches, vi))
            .collect();
        let refs: Vec<&Mat> = sams.iter().collect();
        let batched = forward_batched_from_sam(&cfg, &w, &refs)?;
        assert_eq!(batched.len(), sams.len());
        for (vi, sam) in sams.iter().enumerate() {
            let t = transpose(&sam.data, sam.rows, sam.cols)?;
            let per_view = forward_with(&cfg, &w, &Mat::from_vec(sam.cols, sam.rows, t))?;
            assert_eq!(
                batched[vi].data, per_view.data,
                "view {vi}: from-sam batched CLIP != per-view transpose+forward_with"
            );
        }
        Ok(())
    }

    #[test]
    fn batched_clip_rejects_ragged_and_empty() {
        let cfg = parity_cfg();
        let w = rand_weights(&cfg, 4);
        assert!(forward_with_batched(&cfg, &w, &[]).is_err());
        let a = parity_view(4, cfg.hidden_size, 1);
        let b = parity_view(5, cfg.hidden_size, 2); // ragged: different N
        assert!(forward_with_batched(&cfg, &w, &[&a, &b]).is_err());
    }

    #[test]
    fn forward_with_produces_class_plus_patches_rows() -> FocrResult<()> {
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
        let out = forward_with(&cfg, &w, &sam)?;
        // class token row + num_patches rows.
        assert_eq!(out.shape(), (num_patches + 1, cfg.hidden_size));
        assert!(out.data.iter().all(|v| v.is_finite()));
        Ok(())
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

    #[test]
    fn forward_rejects_malformed_sam_features_before_weight_hydration() -> FocrResult<()> {
        let weights = Weights::from_bytes(FocrqBuilder::new().build())?;
        let image = Mat::zeros(1, 1);
        let sam = Mat {
            rows: 2,
            cols: 2,
            data: vec![0.0],
        };
        assert_err_contains(forward(&weights, &image, &sam), "sam_features");
        Ok(())
    }

    #[test]
    fn forward_with_rejects_malformed_sam_features_without_panic() {
        let cfg = tiny_cfg();
        let w = tiny_weights(&cfg, 2);
        let sam = Mat {
            rows: 2,
            cols: cfg.hidden_size,
            data: vec![0.0; cfg.hidden_size],
        };
        assert_err_contains(forward_with(&cfg, &w, &sam), "data len");
    }

    /// Residual identity check: with all-zero norms-gamma... not applicable;
    /// instead confirm the block residual structure by giving a block whose
    /// attention and mlp both produce ~0 (zero out_proj / zero fc2) so the block
    /// is the identity map x -> x.
    #[test]
    fn zeroed_sublayers_make_block_identity() -> FocrResult<()> {
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
        let out = transformer_block(&cfg, &block, &x)?;
        for (o, i) in out.data.iter().zip(&x.data) {
            assert!((o - i).abs() < 1e-5, "block not identity: {o} vs {i}");
        }
        Ok(())
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
