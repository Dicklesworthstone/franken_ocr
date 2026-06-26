//! SAM ViT-B encoder forward ([SPEC-040..046], PROPOSED_ARCHITECTURE.md §6.3).
//!
//! Real fp32 forward. Patch-embed Conv2d k16s16 -> 64x64 tokens (width 768);
//! learned `pos_embed` (1,64,64,768) bicubic-interpolated to the runtime grid;
//! 12 [`Block`]s with window attention (`window=14`, OQ-15) except global at
//! `[2,5,8,11]`; decomposed relative-position bias added to the SDPA logits; the
//! neck (Conv2d 768->256 k1 -> LayerNorm2d -> Conv2d 256->256 k3 p1 ->
//! LayerNorm2d) then two stride-2 downsamples (`net_2` 256->512, `net_3`
//! 512->1024) -> `[B, 1024, 16, 16]`, returned as a row-major `Mat` of shape
//! `[1024, 16*16]` (channels x flattened spatial), matching the `flatten(2)`
//! layout the bridge concatenates (OQ-6: `local_features_1.flatten(2)`).
//!
//! Weights are owned by the parallel weights wave; this module operates over a
//! [`SamWeights`] parameter bundle so the full math is unit-testable on tiny
//! synthetic inputs with no model present. The public [`forward`] entrypoint
//! adapts a loaded [`Weights`] into a [`SamWeights`] once the `.focrq` reader
//! exposes named tensors (Phase 2, bd-1es.3); until then it surfaces a clear
//! `NotImplemented` rather than fabricating output.

use super::nn;
use super::tensor::Mat;
use super::weights::Weights;
use crate::error::{FocrError, FocrResult};

// ── fixed SAM-ViT-B geometry ([SPEC-040..046]) ─────────────────────────────

/// Patch embedding / transformer width.
pub const EMBED_DIM: usize = 768;
/// Number of transformer blocks.
pub const DEPTH: usize = 12;
/// Attention heads per block.
pub const NUM_HEADS: usize = 12;
/// Per-head channel count (`768 / 12 = 64`).
pub const HEAD_DIM: usize = EMBED_DIM / NUM_HEADS;
/// Patch-embed kernel / stride (`16`), so `1024 -> 64`.
pub const PATCH: usize = 16;
/// Window size for non-global blocks (OQ-15: 14).
pub const WINDOW: usize = 14;
/// Block indices that run global (full-grid) attention.
pub const GLOBAL_BLOCKS: [usize; 4] = [2, 5, 8, 11];
/// Neck / `out_chans` channel count.
pub const NECK_CH: usize = 256;
/// `net_2` output channels.
pub const NET2_CH: usize = 512;
/// `net_3` output channels (the returned feature width).
pub const OUT_CH: usize = 1024;
/// LayerNorm eps for the transformer norms and `LayerNorm2d`.
pub const LN_EPS: f32 = 1e-6;
/// MLP hidden = `dim * mlp_ratio` (`mlp_ratio = 4`).
pub const MLP_HIDDEN: usize = EMBED_DIM * 4;

fn checked_shape_mul(context: &str, lhs: usize, rhs: usize, expression: &str) -> FocrResult<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "{context}: usize overflow computing {expression} ({lhs} * {rhs})"
        ))
    })
}

// ── parameter bundles ──────────────────────────────────────────────────────

/// A `nn.Linear` parameter pair (`[out, in]` row-major weight + length-`out`
/// bias). PyTorch stores `Linear.weight` as `[out_features, in_features]`, so
/// `y = x @ w^T + b`.
#[derive(Debug, Clone)]
pub struct Linear {
    /// `[out, in]` row-major weight.
    pub w: Vec<f32>,
    /// Length-`out` bias (may be empty for `bias=False`).
    pub b: Vec<f32>,
    /// Output features.
    pub out: usize,
    /// Input features.
    pub in_: usize,
}

impl Linear {
    /// `y[m,out] = x[m,in] @ w^T + b`. `x.cols` must equal `self.in_`.
    fn apply(&self, x: &Mat) -> FocrResult<Mat> {
        if x.cols != self.in_ {
            return Err(FocrError::Other(anyhow::anyhow!(
                "vision_sam linear: input cols {} != expected in_features {}",
                x.cols,
                self.in_
            )));
        }
        if self.w.len() != self.out * self.in_ {
            return Err(FocrError::Other(anyhow::anyhow!(
                "vision_sam linear: weight len {} != out*in {}*{}",
                self.w.len(),
                self.out,
                self.in_
            )));
        }
        if !self.b.is_empty() && self.b.len() != self.out {
            return Err(FocrError::Other(anyhow::anyhow!(
                "vision_sam linear: bias len {} != out_features {}",
                self.b.len(),
                self.out
            )));
        }
        // w is [out, in]; transpose to [in, out] so matmul([m,in],[in,out]).
        let wt = transpose(&self.w, self.out, self.in_);
        let wt_mat = Mat::from_vec(self.in_, self.out, wt);
        let mut y = nn::matmul(x, &wt_mat)?;
        if !self.b.is_empty() {
            for r in 0..y.rows {
                let row = y.row_mut(r);
                for (c, v) in row.iter_mut().enumerate() {
                    *v += self.b[c];
                }
            }
        }
        Ok(y)
    }
}

/// A LayerNorm affine pair over `cols` features.
#[derive(Debug, Clone)]
pub struct LayerNormP {
    /// Length-`cols` gain.
    pub w: Vec<f32>,
    /// Length-`cols` shift.
    pub b: Vec<f32>,
}

/// A 2-D convolution weight (`[out_ch, in_ch, kh, kw]` row-major) + optional
/// bias.
#[derive(Debug, Clone)]
pub struct Conv {
    /// `[out_ch, in_ch, kh, kw]` row-major.
    pub w: Vec<f32>,
    /// Optional length-`out_ch` bias.
    pub b: Option<Vec<f32>>,
    /// Output channels.
    pub out_ch: usize,
    /// Input channels.
    pub in_ch: usize,
    /// Kernel height.
    pub kh: usize,
    /// Kernel width.
    pub kw: usize,
}

/// Per-block attention parameters (qkv fused, output proj, rel-pos tables).
#[derive(Debug, Clone)]
pub struct AttnP {
    /// Fused qkv linear: `[3*dim, dim]` weight, `3*dim` bias.
    pub qkv: Linear,
    /// Output projection `[dim, dim]`.
    pub proj: Linear,
    /// `rel_pos_h`: `[2*size_h - 1, head_dim]` row-major.
    pub rel_pos_h: Vec<f32>,
    /// `rel_pos_w`: `[2*size_w - 1, head_dim]` row-major.
    pub rel_pos_w: Vec<f32>,
    /// Rel-pos table spatial size along H (window for windowed blocks, 64 for
    /// global). `rel_pos_h` has `2*size_h - 1` rows.
    pub size_h: usize,
    /// Rel-pos table spatial size along W.
    pub size_w: usize,
}

/// One transformer block's parameters.
#[derive(Debug, Clone)]
pub struct BlockP {
    /// `norm1` (pre-attention).
    pub norm1: LayerNormP,
    /// Attention.
    pub attn: AttnP,
    /// `norm2` (pre-MLP).
    pub norm2: LayerNormP,
    /// MLP `lin1` (`dim -> 4*dim`).
    pub lin1: Linear,
    /// MLP `lin2` (`4*dim -> dim`).
    pub lin2: Linear,
    /// Effective window size (0 => global block).
    pub window: usize,
}

/// The full SAM-ViT-B parameter set.
#[derive(Debug, Clone)]
pub struct SamWeights {
    /// Patch-embed conv (`3 -> 768`, k16 s16).
    pub patch_embed: Conv,
    /// Learned abs pos-embed, row-major `[grid_h, grid_w, dim]` (canonical
    /// `64x64x768`).
    pub pos_embed: Vec<f32>,
    /// Pos-embed source grid height (canonical 64).
    pub pos_grid_h: usize,
    /// Pos-embed source grid width (canonical 64).
    pub pos_grid_w: usize,
    /// The 12 transformer blocks.
    pub blocks: Vec<BlockP>,
    /// Neck conv1 (`768 -> 256`, k1, no bias).
    pub neck_conv1: Conv,
    /// Neck LayerNorm2d #1 (`256`).
    pub neck_ln1: LayerNormP,
    /// Neck conv2 (`256 -> 256`, k3 p1, no bias).
    pub neck_conv2: Conv,
    /// Neck LayerNorm2d #2 (`256`).
    pub neck_ln2: LayerNormP,
    /// `net_2` (`256 -> 512`, k3 s2 p1, no bias).
    pub net2: Conv,
    /// `net_3` (`512 -> 1024`, k3 s2 p1, no bias).
    pub net3: Conv,
}

// ── public entrypoints ─────────────────────────────────────────────────────

/// Run the SAM tower over a normalized `[3, H, W]` image, returning the `x3`
/// 1024-channel feature flattened to `[1024, (H/64spatial)^2]` — i.e. the
/// `net_3` output `[1, 1024, 16, 16]` reshaped channel-major
/// (`flatten(2)` layout, OQ-6).
///
/// # Errors
/// [`FocrError::NotImplemented`] until the `.focrq` reader (Phase 2) exposes
/// named SAM tensors to build a [`SamWeights`]. The real math lives in
/// [`forward_with`], which is exercised by the unit tests below.
pub fn forward(_weights: &Weights, _image: &Mat) -> FocrResult<Mat> {
    Err(FocrError::NotImplemented(
        "native_engine::vision_sam::forward — SAM weights wiring lands with the .focrq reader \
         (Phase 2, bd-1es.3); the fp32 forward math is implemented in `forward_with`"
            .into(),
    ))
}

/// Run the SAM tower with an explicit parameter bundle over a `[3, H, W]`
/// image (laid out `[in_ch=3, H, W]` row-major in `image.data`, `image.rows=3`,
/// `image.cols=H*W` with `H == W` a multiple of `PATCH`).
///
/// Returns the `net_3` feature as a row-major `[OUT_CH, gh3*gw3]` [`Mat`]
/// (channel-major / `flatten(2)` order).
///
/// # Errors
/// [`FocrError::Other`] on a shape contract violation (non-3-channel input,
/// non-square or non-`PATCH`-divisible spatial dims, or a kernel rejection).
pub fn forward_with(w: &SamWeights, image: &Mat, h: usize, win: usize) -> FocrResult<Mat> {
    if image.rows != 3 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_sam: expected 3 input channels, got {}",
            image.rows
        )));
    }
    if h == 0 || win == 0 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_sam: spatial dims ({h},{win}) must be non-zero"
        )));
    }
    let expected_cols = checked_shape_mul("vision_sam", h, win, "H*W")?;
    if image.cols != expected_cols {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_sam: image.cols {} != H*W {}*{} ({})",
            image.cols,
            h,
            win,
            expected_cols
        )));
    }
    if !h.is_multiple_of(PATCH) || !win.is_multiple_of(PATCH) {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_sam: spatial dims ({h},{win}) must be multiples of patch {PATCH}"
        )));
    }
    let gh = h / PATCH;
    let gw = win / PATCH;

    // ── patch embed: Conv2d(3->768, k16, s16), then permute B,C,H,W->B,H,W,C.
    // conv2d kernel wants pre-padded NCHW; patch embed has no padding.
    let dim = w.patch_embed.out_ch;
    let conv_out = nn::conv2d(
        &image.data,
        &w.patch_embed.w,
        w.patch_embed.b.as_deref(),
        1,
        3,
        h,
        win,
        PATCH,
        PATCH,
        gh,
        gw,
        PATCH,
        PATCH,
        dim,
    );
    // conv_out is [1, dim, gh, gw] (channel-major). Tokens we carry as
    // [gh*gw, dim] (spatial-major rows) for the transformer (NHWC flattened).
    let mut x = nchw_to_nhwc_rows(&conv_out, dim, gh, gw);

    // ── abs pos-embed (added once before the blocks; bicubic-interp if needed).
    let pos = abs_pos(&w.pos_embed, w.pos_grid_h, w.pos_grid_w, dim, gh, gw);
    debug_assert_eq!(pos.len(), x.data.len());
    for (xv, pv) in x.data.iter_mut().zip(pos.iter()) {
        *xv += *pv;
    }

    // ── 12 transformer blocks.
    for blk in &w.blocks {
        x = block_forward(blk, &x, gh, gw)?;
    }

    // ── neck: x is [gh*gw, dim] NHWC rows; neck operates NCHW.
    // permute(0,3,1,2): NHWC-rows -> NCHW flat.
    let x_nchw = nhwc_rows_to_nchw(&x, dim, gh, gw);

    // neck conv1: 768 -> 256, k1, no pad.
    let nc1 = conv_apply(&w.neck_conv1, &x_nchw, gh, gw, 0, 1)?;
    let nc1 = layer_norm_2d(&nc1, &w.neck_ln1, NECK_CH, gh, gw);
    // neck conv2: 256 -> 256, k3, pad1.
    let nc2 = conv_apply(&w.neck_conv2, &nc1, gh, gw, 1, 1)?;
    let neck = layer_norm_2d(&nc2, &w.neck_ln2, NECK_CH, gh, gw);

    // net_2: 256 -> 512, k3, s2, p1 -> grid /2.
    let (gh2, gw2) = (gh.div_ceil(2), gw.div_ceil(2));
    let x2 = conv_apply(&w.net2, &neck, gh, gw, 1, 2)?;
    // net_3: 512 -> 1024, k3, s2, p1 -> grid /2 again.
    let (gh3, gw3) = (gh2.div_ceil(2), gw2.div_ceil(2));
    let x3 = conv_apply(&w.net3, &x2, gh2, gw2, 1, 2)?;

    // x3 is [OUT_CH, gh3*gw3] NCHW flat — exactly flatten(2) layout.
    Ok(Mat::from_vec(OUT_CH, gh3 * gw3, x3))
}

// ── transformer block ──────────────────────────────────────────────────────

/// One [`BlockP`] over NHWC token rows `[gh*gw, dim]`.
///
/// `shortcut = x; x = norm1(x);` (window_partition if windowed) `x = attn(x);`
/// (window_unpartition); `x = shortcut + x; x = x + mlp(norm2(x))`.
fn block_forward(blk: &BlockP, x: &Mat, gh: usize, gw: usize) -> FocrResult<Mat> {
    let dim = x.cols;
    let normed = layer_norm_rows(x, &blk.norm1)?;

    let attn_out = if blk.window > 0 {
        // window_partition: pad to multiple of window, tile into win x win.
        let ws = blk.window;
        let pad_h = (ws - gh % ws) % ws;
        let pad_w = (ws - gw % ws) % ws;
        let hp = gh + pad_h;
        let wp = gw + pad_w;
        let nwin_h = hp / ws;
        let nwin_w = wp / ws;
        let nwin = nwin_h * nwin_w;

        // Build per-window token blocks [nwin][ws*ws, dim] (zero-padded tail).
        let mut windows = vec![0.0f32; nwin * ws * ws * dim];
        for wy in 0..nwin_h {
            for wx in 0..nwin_w {
                let widx = wy * nwin_w + wx;
                for ly in 0..ws {
                    for lx in 0..ws {
                        let gy = wy * ws + ly;
                        let gxx = wx * ws + lx;
                        let dst = ((widx * ws + ly) * ws + lx) * dim;
                        if gy < gh && gxx < gw {
                            let src = (gy * gw + gxx) * dim;
                            windows[dst..dst + dim].copy_from_slice(&normed.data[src..src + dim]);
                        }
                    }
                }
            }
        }

        // Attention per window (each window: ws x ws grid).
        let mut out_windows = vec![0.0f32; windows.len()];
        for widx in 0..nwin {
            let base = widx * ws * ws * dim;
            let win_in = Mat::from_vec(ws * ws, dim, windows[base..base + ws * ws * dim].to_vec());
            let win_out = attention(&blk.attn, &win_in, ws, ws)?;
            out_windows[base..base + ws * ws * dim].copy_from_slice(&win_out.data);
        }

        // window_unpartition: scatter back, strip padding.
        let mut merged = vec![0.0f32; gh * gw * dim];
        for wy in 0..nwin_h {
            for wx in 0..nwin_w {
                let widx = wy * nwin_w + wx;
                for ly in 0..ws {
                    for lx in 0..ws {
                        let gy = wy * ws + ly;
                        let gxx = wx * ws + lx;
                        if gy < gh && gxx < gw {
                            let src = ((widx * ws + ly) * ws + lx) * dim;
                            let dst = (gy * gw + gxx) * dim;
                            merged[dst..dst + dim].copy_from_slice(&out_windows[src..src + dim]);
                        }
                    }
                }
            }
        }
        Mat::from_vec(gh * gw, dim, merged)
    } else {
        attention(&blk.attn, &normed, gh, gw)?
    };
    ensure_same_shape("vision_sam block attention residual", &attn_out, x)?;

    // residual 1: x = shortcut + attn
    let mut h1 = x.clone();
    for (a, b) in h1.data.iter_mut().zip(attn_out.data.iter()) {
        *a += *b;
    }

    // residual 2: x = h1 + mlp(norm2(h1))
    let normed2 = layer_norm_rows(&h1, &blk.norm2)?;
    let mut mlp = blk.lin1.apply(&normed2)?;
    nn::gelu(&mut mlp);
    let mlp = blk.lin2.apply(&mlp)?;
    ensure_same_shape("vision_sam block mlp residual", &mlp, &h1)?;
    for (a, b) in h1.data.iter_mut().zip(mlp.data.iter()) {
        *a += *b;
    }
    Ok(h1)
}

// ── attention with decomposed relative position ([SPEC-044]) ───────────────

/// Multi-head attention over a `gh x gw` token grid `[gh*gw, dim]`, adding the
/// decomposed rel-pos bias to the logits before softmax. Returns `proj(out)`
/// shaped `[gh*gw, dim]`.
fn attention(p: &AttnP, x: &Mat, gh: usize, gw: usize) -> FocrResult<Mat> {
    let n = gh * gw;
    if x.rows != n {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_sam attention: input rows {} != gh*gw {}*{}",
            x.rows,
            gh,
            gw
        )));
    }
    let dim = x.cols;
    let nh = NUM_HEADS;
    if dim == 0 || !dim.is_multiple_of(nh) {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_sam attention: dim {} must be non-zero and divisible by heads {}",
            dim,
            nh
        )));
    }
    if p.qkv.in_ != dim || p.qkv.out != 3 * dim {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_sam attention: qkv shape out/in {}x{} incompatible with dim {}",
            p.qkv.out,
            p.qkv.in_,
            dim
        )));
    }
    if p.proj.in_ != dim || p.proj.out != dim {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_sam attention: proj shape out/in {}x{} incompatible with dim {}",
            p.proj.out,
            p.proj.in_,
            dim
        )));
    }
    let hd = dim / nh;
    ensure_rel_pos_len("rel_pos_h", p.size_h, hd, p.rel_pos_h.len())?;
    ensure_rel_pos_len("rel_pos_w", p.size_w, hd, p.rel_pos_w.len())?;
    let scale = (hd as f32).powf(-0.5);

    // qkv: [n, 3*dim].
    let qkv = p.qkv.apply(x)?;
    ensure_mat_shape(&qkv, n, 3 * dim, "vision_sam attention qkv output")?;
    // Split into per-head q,k,v: layout [n, 3, nh, hd] (PyTorch reshape order).
    // qkv row r: [3*dim] = for s in 0..3 { for head in 0..nh { hd } }.
    // We build flat per-head buffers [nh][n, hd] for q,k,v.
    let mut q = vec![0.0f32; nh * n * hd];
    let mut k = vec![0.0f32; nh * n * hd];
    let mut v = vec![0.0f32; nh * n * hd];
    for r in 0..n {
        let row = qkv.row(r);
        for head in 0..nh {
            for d in 0..hd {
                let qv = row[head * hd + d];
                let kv = row[(nh + head) * hd + d];
                let vv = row[(2 * nh + head) * hd + d];
                q[(head * n + r) * hd + d] = qv;
                k[(head * n + r) * hd + d] = kv;
                v[(head * n + r) * hd + d] = vv;
            }
        }
    }

    // Decomposed rel-pos bias [nh][n(q), n(k)] = rel_h[qy, ky] + rel_w[qx, kx].
    // Rh = get_rel_pos(gh, gh, rel_pos_h) -> [gh, gh, hd]
    // Rw = get_rel_pos(gw, gw, rel_pos_w) -> [gw, gw, hd]
    // rel_h[head, q, ky] = sum_c q[head,q,c] * Rh[qy, ky, c]
    // rel_w[head, q, kx] = sum_c q[head,q,c] * Rw[qx, kx, c]
    let rh = get_rel_pos(gh, gh, &p.rel_pos_h, p.size_h, hd);
    let rw = get_rel_pos(gw, gw, &p.rel_pos_w, p.size_w, hd);

    // Compute attention head-by-head with explicit logits so we can add bias.
    let mut out = vec![0.0f32; nh * n * hd]; // [nh][n, hd]
    for head in 0..nh {
        let qh = &q[head * n * hd..(head + 1) * n * hd];
        let kh = &k[head * n * hd..(head + 1) * n * hd];
        let vh = &v[head * n * hd..(head + 1) * n * hd];
        let (rel_h_bias, rel_w_bias) = decomposed_rel_pos_bias(qh, &rh, &rw, gh, gw, hd);

        // logits = scale * (Q @ K^T) + decomposed rel-pos bias.
        let qh_mat = Mat::from_vec(n, hd, qh.to_vec());
        let kt_mat = Mat::from_vec(hd, n, transpose_contiguous_stores(kh, n, hd));
        let mut lm = nn::matmul(&qh_mat, &kt_mat)?;
        for i in 0..n {
            for j in 0..n {
                let ky = j / gw;
                let kx = j % gw;
                let bh = rel_h_bias[i * gh + ky];
                let bw = rel_w_bias[i * gw + kx];
                lm.data[i * n + j] = scale * lm.data[i * n + j] + bh + bw;
            }
        }
        // softmax rows then weighted sum of v.
        nn::softmax_rows(&mut lm)?;
        let vh_mat = Mat::from_vec(n, hd, vh.to_vec());
        let head_out = nn::matmul(&lm, &vh_mat)?;
        out[head * n * hd..(head + 1) * n * hd].copy_from_slice(&head_out.data);
    }

    // Reassemble [n, dim] from [nh][n, hd] (head-major -> token rows).
    let mut ctx = vec![0.0f32; n * dim];
    for head in 0..nh {
        for r in 0..n {
            let src = (head * n + r) * hd;
            let dst = r * dim + head * hd;
            ctx[dst..dst + hd].copy_from_slice(&out[src..src + hd]);
        }
    }
    let ctx_mat = Mat::from_vec(n, dim, ctx);
    let y = p.proj.apply(&ctx_mat)?;
    ensure_mat_shape(&y, n, dim, "vision_sam attention projection output")?;
    Ok(y)
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

fn ensure_same_shape(context: &str, actual: &Mat, expected: &Mat) -> FocrResult<()> {
    if actual.shape() == expected.shape() {
        return Ok(());
    }
    Err(FocrError::Other(anyhow::anyhow!(
        "{context}: shape {:?} != expected {:?}",
        actual.shape(),
        expected.shape()
    )))
}

fn ensure_rel_pos_len(name: &str, size: usize, hd: usize, actual_len: usize) -> FocrResult<()> {
    let rows = size
        .checked_mul(2)
        .and_then(|n| n.checked_sub(1))
        .ok_or_else(|| {
            FocrError::Other(anyhow::anyhow!(
                "vision_sam attention: {name} size {size} is invalid"
            ))
        })?;
    let expected_len = rows.checked_mul(hd).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "vision_sam attention: {name} size {size} overflows for head_dim {hd}"
        ))
    })?;
    if actual_len == expected_len {
        return Ok(());
    }
    Err(FocrError::Other(anyhow::anyhow!(
        "vision_sam attention: {name} len {actual_len} != expected {expected_len} \
         for size {size} and head_dim {hd}"
    )))
}

fn decomposed_rel_pos_bias(
    qh: &[f32],
    rh: &[f32],
    rw: &[f32],
    gh: usize,
    gw: usize,
    hd: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = gh * gw;
    debug_assert_eq!(qh.len(), n * hd);
    debug_assert_eq!(rh.len(), gh * gh * hd);
    debug_assert_eq!(rw.len(), gw * gw * hd);

    let mut rel_h_bias = vec![0.0f32; n * gh];
    let mut rel_w_bias = vec![0.0f32; n * gw];
    for i in 0..n {
        let qy = i / gw;
        let qx = i % gw;
        let qi = &qh[i * hd..(i + 1) * hd];
        for ky in 0..gh {
            let rh_base = (qy * gh + ky) * hd;
            let mut bh = 0.0f32;
            for c in 0..hd {
                bh += qi[c] * rh[rh_base + c];
            }
            rel_h_bias[i * gh + ky] = bh;
        }
        for kx in 0..gw {
            let rw_base = (qx * gw + kx) * hd;
            let mut bw = 0.0f32;
            for c in 0..hd {
                bw += qi[c] * rw[rw_base + c];
            }
            rel_w_bias[i * gw + kx] = bw;
        }
    }
    (rel_h_bias, rel_w_bias)
}

/// `get_rel_pos(q_size, k_size, rel_pos)` -> `[q_size, k_size, head_dim]`.
///
/// The table `rel_pos` is `[2*size - 1, hd]`. When `size == q_size == k_size`
/// (our case — windows / global grid match the table), no interpolation is
/// needed and we index `rel_pos[(q - k) + (k_size - 1)]` directly (the
/// `q_coords - k_coords + (k_size-1)` formula with `q_size == k_size`).
fn get_rel_pos(q_size: usize, k_size: usize, rel_pos: &[f32], size: usize, hd: usize) -> Vec<f32> {
    let max_rel = 2 * q_size.max(k_size) - 1;
    // Resize the table to max_rel rows via linear interpolation if its row
    // count differs (matches F.interpolate(mode="linear") in get_rel_pos).
    let table_rows = rel_pos.len() / hd;
    debug_assert_eq!(table_rows, 2 * size - 1);
    let resized: Vec<f32> = if table_rows != max_rel {
        interp_linear_rows(rel_pos, table_rows, hd, max_rel)
    } else {
        rel_pos.to_vec()
    };

    let qf = q_size as f32;
    let kf = k_size as f32;
    let ratio_qk = (kf / qf).max(1.0);
    let ratio_kq = (qf / kf).max(1.0);
    let mut out = vec![0.0f32; q_size * k_size * hd];
    for qi in 0..q_size {
        for ki in 0..k_size {
            let qc = qi as f32 * ratio_qk;
            let kc = ki as f32 * ratio_kq;
            // PyTorch SAM uses max(q_size/k_size, 1.0) = ratio_kq for this offset
            // term (audit rank 8). Identical to ratio_qk only when q_size == k_size
            // (every current call site), so this hardens the helper for q != k.
            let rc = (qc - kc) + (k_size as f32 - 1.0) * ratio_kq;
            let idx = rc as usize; // .long() truncation
            let src = idx * hd;
            let dst = (qi * k_size + ki) * hd;
            out[dst..dst + hd].copy_from_slice(&resized[src..src + hd]);
        }
    }
    out
}

/// Linear (1-D) interpolation of a `[rows, hd]` table to `[new_rows, hd]`,
/// matching `F.interpolate(mode="linear", align_corners=False)` over the row
/// axis (per-feature). Only hit when a rel-pos table size mismatches the grid.
fn interp_linear_rows(src: &[f32], rows: usize, hd: usize, new_rows: usize) -> Vec<f32> {
    if new_rows == rows {
        return src.to_vec();
    }
    let mut out = vec![0.0f32; new_rows * hd];
    let scale = rows as f32 / new_rows as f32;
    for i in 0..new_rows {
        // align_corners=False source coordinate.
        let s = (i as f32 + 0.5) * scale - 0.5;
        let s_clamped = s.clamp(0.0, (rows - 1) as f32);
        let lo = s_clamped.floor() as usize;
        let hi = (lo + 1).min(rows - 1);
        let frac = s_clamped - lo as f32;
        for c in 0..hd {
            let a = src[lo * hd + c];
            let b = src[hi * hd + c];
            out[i * hd + c] = a + (b - a) * frac;
        }
    }
    out
}

// ── normalization helpers ──────────────────────────────────────────────────

/// LayerNorm over the last dim of token rows `[n, cols]` with affine params.
fn layer_norm_rows(x: &Mat, ln: &LayerNormP) -> FocrResult<Mat> {
    nn::layer_norm(x, Some(&ln.w), Some(&ln.b), LN_EPS)
}

/// `LayerNorm2d` over an NCHW-flat `[C, H*W]` buffer: normalize across the
/// CHANNEL axis at each spatial location, then per-channel affine
/// (`deepencoder.py:590-602`: `mean(1)`/`var(1)` over channels).
fn layer_norm_2d(x: &[f32], ln: &LayerNormP, ch: usize, gh: usize, gw: usize) -> Vec<f32> {
    let hw = gh * gw;
    debug_assert_eq!(x.len(), ch * hw);
    let mut out = vec![0.0f32; ch * hw];
    for s in 0..hw {
        // mean over channels at spatial location s.
        let mut mean = 0.0f32;
        for c in 0..ch {
            mean += x[c * hw + s];
        }
        mean /= ch as f32;
        let mut var = 0.0f32;
        for c in 0..ch {
            let d = x[c * hw + s] - mean;
            var += d * d;
        }
        var /= ch as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        for c in 0..ch {
            let norm = (x[c * hw + s] - mean) * inv;
            out[c * hw + s] = ln.w[c] * norm + ln.b[c];
        }
    }
    out
}

// ── conv + layout helpers ──────────────────────────────────────────────────

/// Apply a [`Conv`] over an NCHW-flat `[in_ch, gh*gw]` buffer with symmetric
/// zero padding `pad` and stride `stride`, returning the NCHW-flat output.
fn conv_apply(
    conv: &Conv,
    input: &[f32],
    gh: usize,
    gw: usize,
    pad: usize,
    stride: usize,
) -> FocrResult<Vec<f32>> {
    let ph = gh + 2 * pad;
    let pw = gw + 2 * pad;
    let padded = pad_nchw(input, conv.in_ch, gh, gw, pad);
    let oh = (ph - conv.kh) / stride + 1;
    let ow = (pw - conv.kw) / stride + 1;
    let out = nn::conv2d(
        &padded,
        &conv.w,
        conv.b.as_deref(),
        1,
        conv.in_ch,
        ph,
        pw,
        conv.kh,
        conv.kw,
        oh,
        ow,
        stride,
        stride,
        conv.out_ch,
    );
    debug_assert_eq!(out.len(), conv.out_ch * oh * ow);
    Ok(out)
}

/// Zero-pad an NCHW-flat `[ch, gh*gw]` buffer by `pad` on every spatial side ->
/// `[ch, (gh+2p)*(gw+2p)]`.
fn pad_nchw(input: &[f32], ch: usize, gh: usize, gw: usize, pad: usize) -> Vec<f32> {
    if pad == 0 {
        return input.to_vec();
    }
    let ph = gh + 2 * pad;
    let pw = gw + 2 * pad;
    let mut out = vec![0.0f32; ch * ph * pw];
    for c in 0..ch {
        for y in 0..gh {
            for x in 0..gw {
                let src = c * gh * gw + y * gw + x;
                let dst = c * ph * pw + (y + pad) * pw + (x + pad);
                out[dst] = input[src];
            }
        }
    }
    out
}

/// `[1, ch, gh, gw]` channel-major conv output -> NHWC token rows
/// `[gh*gw, ch]`.
fn nchw_to_nhwc_rows(nchw: &[f32], ch: usize, gh: usize, gw: usize) -> Mat {
    let n = gh * gw;
    let mut data = vec![0.0f32; n * ch];
    for c in 0..ch {
        for s in 0..n {
            data[s * ch + c] = nchw[c * n + s];
        }
    }
    Mat::from_vec(n, ch, data)
}

/// NHWC token rows `[gh*gw, ch]` -> NCHW-flat `[ch, gh*gw]`
/// (`permute(0,3,1,2)`).
fn nhwc_rows_to_nchw(x: &Mat, ch: usize, gh: usize, gw: usize) -> Vec<f32> {
    let n = gh * gw;
    debug_assert_eq!(x.rows, n);
    debug_assert_eq!(x.cols, ch);
    let mut out = vec![0.0f32; ch * n];
    for s in 0..n {
        let row = x.row(s);
        for c in 0..ch {
            out[c * n + s] = row[c];
        }
    }
    out
}

/// Transpose a `[rows, cols]` row-major matrix to `[cols, rows]`.
fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for c in 0..cols {
        let dst = &mut out[c * rows..(c + 1) * rows];
        for (r, slot) in dst.iter_mut().enumerate() {
            *slot = m[r * cols + c];
        }
    }
    out
}

fn transpose_contiguous_stores(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for c in 0..cols {
        let dst = &mut out[c * rows..(c + 1) * rows];
        for r in 0..rows {
            dst[r] = m[r * cols + c];
        }
    }
    out
}

// ── absolute position embedding (bicubic interp, [SPEC-042]) ───────────────

/// Build the abs pos-embed contribution for a `[gh, gw, dim]` grid from the
/// learned `[src_h, src_w, dim]` table, bicubic-interpolating if the runtime
/// grid differs (matches `get_abs_pos_sam`). Returns NHWC token-row order
/// `[gh*gw, dim]` flattened.
fn abs_pos(pos: &[f32], src_h: usize, src_w: usize, dim: usize, gh: usize, gw: usize) -> Vec<f32> {
    if src_h == gh && src_w == gw {
        return pos.to_vec(); // already [gh*gw, dim] NHWC rows
    }
    // Interpolate per-channel over the spatial grid (bicubic, align_corners=
    // False). pos is [src_h, src_w, dim] (channel-last); we resample each
    // channel independently into [gh, gw].
    let mut out = vec![0.0f32; gh * gw * dim];
    let scale_y = src_h as f32 / gh as f32;
    let scale_x = src_w as f32 / gw as f32;
    for oy in 0..gh {
        let sy = (oy as f32 + 0.5) * scale_y - 0.5;
        for ox in 0..gw {
            let sx = (ox as f32 + 0.5) * scale_x - 0.5;
            let dst = (oy * gw + ox) * dim;
            for c in 0..dim {
                out[dst + c] = bicubic_sample(pos, src_h, src_w, dim, c, sy, sx);
            }
        }
    }
    out
}

/// Bicubic sample of channel `c` from a `[src_h, src_w, dim]` channel-last
/// table at fractional `(sy, sx)` using the Catmull-Rom-ish cubic convolution
/// kernel (`a = -0.75`, PyTorch default), edge-clamped.
fn bicubic_sample(
    pos: &[f32],
    src_h: usize,
    src_w: usize,
    dim: usize,
    c: usize,
    sy: f32,
    sx: f32,
) -> f32 {
    let iy = sy.floor();
    let ix = sx.floor();
    let fy = sy - iy;
    let fx = sx - ix;
    let wy = cubic_weights(fy);
    let wx = cubic_weights(fx);
    let mut acc = 0.0f32;
    // indexed loop: spatial kernel offset
    #[allow(clippy::needless_range_loop)]
    for m in 0..4 {
        let yy = clamp_idx(iy as isize - 1 + m as isize, src_h);
        // indexed loop: spatial kernel offset
        #[allow(clippy::needless_range_loop)]
        for n in 0..4 {
            let xx = clamp_idx(ix as isize - 1 + n as isize, src_w);
            let val = pos[(yy * src_w + xx) * dim + c];
            acc += val * wy[m] * wx[n];
        }
    }
    acc
}

/// Cubic-convolution weights for the 4-tap stencil around fractional `t` in
/// `[0,1)` with `a = -0.75` (PyTorch `mode="bicubic"`).
fn cubic_weights(t: f32) -> [f32; 4] {
    let a = -0.75f32;
    // distances of the four samples (indices -1,0,1,2) from t.
    let d0 = 1.0 + t;
    let d1 = t;
    let d2 = 1.0 - t;
    let d3 = 2.0 - t;
    [
        cubic_k(d0, a),
        cubic_k(d1, a),
        cubic_k(d2, a),
        cubic_k(d3, a),
    ]
}

/// Keys cubic kernel `W(x)` with parameter `a`.
fn cubic_k(x: f32, a: f32) -> f32 {
    let x = x.abs();
    if x <= 1.0 {
        (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
    } else if x < 2.0 {
        a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a
    } else {
        0.0
    }
}

/// Clamp an index to `[0, n-1]`.
fn clamp_idx(i: isize, n: usize) -> usize {
    if i < 0 {
        0
    } else if i as usize >= n {
        n - 1
    } else {
        i as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Instant;

    /// Identity 1x1 conv (out_ch==in_ch, k=1) with an identity-ish weight is a
    /// channel mixing; here we use a diagonal weight so output == input.
    fn identity_conv1(ch: usize) -> Conv {
        let mut w = vec![0.0f32; ch * ch];
        for c in 0..ch {
            w[c * ch + c] = 1.0;
        }
        Conv {
            w,
            b: None,
            out_ch: ch,
            in_ch: ch,
            kh: 1,
            kw: 1,
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

    #[test]
    fn transpose_roundtrips() {
        let m = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
        let t = transpose(&m, 2, 3); // [3,2]
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(transpose(&t, 3, 2), m);
    }

    #[test]
    fn linear_applies_weight_and_bias() -> FocrResult<()> {
        // w = [[1,2,3],[4,5,6]] (out=2,in=3), b=[10,20]
        let lin = Linear {
            w: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            b: vec![10.0, 20.0],
            out: 2,
            in_: 3,
        };
        // x = [[1,1,1]] -> y = [1+2+3+10, 4+5+6+20] = [16, 35]
        let x = Mat::from_vec(1, 3, vec![1.0, 1.0, 1.0]);
        let y = lin.apply(&x)?;
        assert_eq!(y.shape(), (1, 2));
        assert!((y.data[0] - 16.0).abs() < 1e-5);
        assert!((y.data[1] - 35.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn linear_rejects_malformed_shapes_without_panic() {
        let x = Mat::from_vec(1, 3, vec![1.0, 2.0, 3.0]);
        assert_err_contains(
            Linear {
                w: vec![1.0; 5],
                b: vec![],
                out: 2,
                in_: 3,
            }
            .apply(&x),
            "weight len",
        );
        assert_err_contains(
            Linear {
                w: vec![1.0; 6],
                b: vec![0.0],
                out: 2,
                in_: 3,
            }
            .apply(&x),
            "bias len",
        );
        let bad_x = Mat::from_vec(1, 2, vec![1.0, 2.0]);
        assert_err_contains(
            Linear {
                w: vec![1.0; 6],
                b: vec![],
                out: 2,
                in_: 3,
            }
            .apply(&bad_x),
            "input cols",
        );
    }

    #[test]
    fn pad_nchw_zeros_border() {
        // single channel 2x2 = [[1,2],[3,4]], pad=1 -> 4x4 with zero border.
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = pad_nchw(&input, 1, 2, 2, 1);
        assert_eq!(out.len(), 16);
        // center 2x2 holds the original
        assert_eq!(out[4 + 1], 1.0);
        assert_eq!(out[4 + 2], 2.0);
        assert_eq!(out[8 + 1], 3.0);
        assert_eq!(out[8 + 2], 4.0);
        // corners are zero
        assert_eq!(out[0], 0.0);
        assert_eq!(out[15], 0.0);
    }

    #[test]
    fn nchw_nhwc_roundtrip() {
        // ch=2, grid 2x2 (n=4). NCHW: c0=[1,2,3,4], c1=[5,6,7,8].
        let nchw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let rows = nchw_to_nhwc_rows(&nchw, 2, 2, 2);
        // row 0 (spatial 0) = [c0=1, c1=5]
        assert_eq!(rows.row(0), &[1.0, 5.0]);
        assert_eq!(rows.row(3), &[4.0, 8.0]);
        let back = nhwc_rows_to_nchw(&rows, 2, 2, 2);
        assert_eq!(back, nchw);
    }

    #[test]
    fn layer_norm_2d_normalizes_channels() {
        // ch=2, grid 1x2 (hw=2). At spatial 0 channels=[1,3] -> mean 2, var 1.
        // normalized = [-1, 1]; affine w=[1,1], b=[0,0].
        let x = vec![1.0, 9.0, 3.0, 11.0]; // c0=[1,9], c1=[3,11] over hw=2
        let ln = LayerNormP {
            w: vec![1.0, 1.0],
            b: vec![0.0, 0.0],
        };
        let out = layer_norm_2d(&x, &ln, 2, 1, 2);
        // spatial 0: channels [1,3], mean 2, var 1 -> [-1, 1]
        assert!((out[0] - (-1.0)).abs() < 1e-3); // c0,s0
        assert!((out[2] - 1.0).abs() < 1e-3); // c1,s0
        // spatial 1: channels [9,11], mean 10, var 1 -> [-1, 1]
        assert!((out[1] - (-1.0)).abs() < 1e-3); // c0,s1
        assert!((out[3] - 1.0).abs() < 1e-3); // c1,s1
    }

    #[test]
    fn conv_apply_identity_1x1_preserves() -> FocrResult<()> {
        // 2 channels, 2x2 grid, identity 1x1 conv -> unchanged.
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let conv = identity_conv1(2);
        let out = conv_apply(&conv, &input, 2, 2, 0, 1)?;
        assert_eq!(out, input);
        Ok(())
    }

    #[test]
    fn cubic_weights_sum_to_one() {
        // The cubic-convolution kernel taps sum to 1 for any fractional t.
        for &t in &[0.0f32, 0.25, 0.5, 0.75, 0.99] {
            let w = cubic_weights(t);
            let s: f32 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "t={t} sum={s}");
        }
    }

    #[test]
    fn abs_pos_identity_when_grid_matches() {
        // src grid == target grid -> passthrough.
        let pos = vec![1.0, 2.0, 3.0, 4.0]; // 2x2x1
        let out = abs_pos(&pos, 2, 2, 1, 2, 2);
        assert_eq!(out, pos);
    }

    #[test]
    fn abs_pos_bicubic_constant_field_is_constant() {
        // A constant field must remain constant under bicubic resample
        // (partition-of-unity weights). src 4x4 of 7.0 -> target 6x6.
        let dim = 1;
        let pos = vec![7.0f32; 4 * 4 * dim];
        let out = abs_pos(&pos, 4, 4, dim, 6, 6);
        assert_eq!(out.len(), 6 * 6);
        for &v in &out {
            assert!((v - 7.0).abs() < 1e-3, "got {v}");
        }
    }

    #[test]
    fn get_rel_pos_indexes_table_directly() {
        // size==q==k==2 -> table has 2*2-1=3 rows; hd=1.
        // table rows [10, 20, 30] for offsets (q-k)+(k-1):
        //   q0,k0: 0-0+1 = 1 -> 20
        //   q0,k1: 0-1+1 = 0 -> 10
        //   q1,k0: 1-0+1 = 2 -> 30
        //   q1,k1: 1-1+1 = 1 -> 20
        let table = vec![10.0, 20.0, 30.0];
        let r = get_rel_pos(2, 2, &table, 2, 1);
        // layout [q, k, hd]
        assert_eq!(r[0], 20.0); // q0,k0
        assert_eq!(r[1], 10.0); // q0,k1
        assert_eq!(r[2], 30.0); // q1,k0
        assert_eq!(r[3], 20.0); // q1,k1
    }

    #[test]
    fn decomposed_rel_pos_bias_matches_direct_inner_loop_formula() {
        let (gh, gw, hd) = (3, 2, 5);
        let n = gh * gw;
        let qh: Vec<f32> = (0..n * hd)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.013)
            .collect();
        let rh: Vec<f32> = (0..gh * gh * hd)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.017)
            .collect();
        let rw: Vec<f32> = (0..gw * gw * hd)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.019)
            .collect();

        let (rel_h_bias, rel_w_bias) = decomposed_rel_pos_bias(&qh, &rh, &rw, gh, gw, hd);
        for i in 0..n {
            let qy = i / gw;
            let qx = i % gw;
            let qi = &qh[i * hd..(i + 1) * hd];
            for j in 0..n {
                let ky = j / gw;
                let kx = j % gw;
                let rh_base = (qy * gh + ky) * hd;
                let rw_base = (qx * gw + kx) * hd;
                let mut expected_h = 0.0f32;
                let mut expected_w = 0.0f32;
                for c in 0..hd {
                    expected_h += qi[c] * rh[rh_base + c];
                    expected_w += qi[c] * rw[rw_base + c];
                }
                assert_eq!(rel_h_bias[i * gh + ky], expected_h);
                assert_eq!(rel_w_bias[i * gw + kx], expected_w);
            }
        }
    }

    /// Build a tiny single-block SAM with `dim=NUM_HEADS*hd`. Here we use the
    /// real `EMBED_DIM`/`NUM_HEADS` but a tiny 2x2 grid (no window padding since
    /// global block) to exercise attention + rel-pos + residual + MLP shapes.
    fn tiny_block(window: usize) -> BlockP {
        let dim = EMBED_DIM;
        let hd = HEAD_DIM;
        // zero rel-pos so attention reduces to plain SDPA (tables sized to the
        // grid we test: 2x2 -> 2*2-1 = 3 rows).
        let size = 2;
        let rel_rows = 2 * size - 1;
        BlockP {
            norm1: LayerNormP {
                w: vec![1.0; dim],
                b: vec![0.0; dim],
            },
            attn: AttnP {
                qkv: Linear {
                    w: identity_block_3(dim),
                    b: vec![0.0; 3 * dim],
                    out: 3 * dim,
                    in_: dim,
                },
                proj: Linear {
                    w: identity_mat(dim),
                    b: vec![0.0; dim],
                    out: dim,
                    in_: dim,
                },
                rel_pos_h: vec![0.0; rel_rows * hd],
                rel_pos_w: vec![0.0; rel_rows * hd],
                size_h: size,
                size_w: size,
            },
            norm2: LayerNormP {
                w: vec![1.0; dim],
                b: vec![0.0; dim],
            },
            lin1: Linear {
                w: vec![0.0; MLP_HIDDEN * dim],
                b: vec![0.0; MLP_HIDDEN],
                out: MLP_HIDDEN,
                in_: dim,
            },
            lin2: Linear {
                w: vec![0.0; dim * MLP_HIDDEN],
                b: vec![0.0; dim],
                out: dim,
                in_: MLP_HIDDEN,
            },
            window,
        }
    }

    /// Identity `[dim, dim]` row-major.
    fn identity_mat(dim: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; dim * dim];
        for i in 0..dim {
            w[i * dim + i] = 1.0;
        }
        w
    }

    /// qkv weight `[3*dim, dim]` that copies x into each of q,k,v (identity per
    /// block).
    fn identity_block_3(dim: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; 3 * dim * dim];
        for s in 0..3 {
            for i in 0..dim {
                let row = s * dim + i;
                w[row * dim + i] = 1.0;
            }
        }
        w
    }

    fn attention_scalar_reference(p: &AttnP, x: &Mat, gh: usize, gw: usize) -> FocrResult<Mat> {
        let n = gh * gw;
        let dim = x.cols;
        let nh = NUM_HEADS;
        let hd = dim / nh;
        let scale = (hd as f32).powf(-0.5);
        let qkv = p.qkv.apply(x)?;
        let mut q = vec![0.0f32; nh * n * hd];
        let mut k = vec![0.0f32; nh * n * hd];
        let mut v = vec![0.0f32; nh * n * hd];
        for r in 0..n {
            let row = qkv.row(r);
            for head in 0..nh {
                for d in 0..hd {
                    q[(head * n + r) * hd + d] = row[head * hd + d];
                    k[(head * n + r) * hd + d] = row[(nh + head) * hd + d];
                    v[(head * n + r) * hd + d] = row[(2 * nh + head) * hd + d];
                }
            }
        }

        let rh = get_rel_pos(gh, gh, &p.rel_pos_h, p.size_h, hd);
        let rw = get_rel_pos(gw, gw, &p.rel_pos_w, p.size_w, hd);
        let mut out = vec![0.0f32; nh * n * hd];
        for head in 0..nh {
            let qh = &q[head * n * hd..(head + 1) * n * hd];
            let kh = &k[head * n * hd..(head + 1) * n * hd];
            let vh = &v[head * n * hd..(head + 1) * n * hd];
            let (rel_h_bias, rel_w_bias) = decomposed_rel_pos_bias(qh, &rh, &rw, gh, gw, hd);
            let mut logits = vec![0.0f32; n * n];
            for i in 0..n {
                let qi = &qh[i * hd..(i + 1) * hd];
                for j in 0..n {
                    let ky = j / gw;
                    let kx = j % gw;
                    let kj = &kh[j * hd..(j + 1) * hd];
                    let mut dot = 0.0f32;
                    for c in 0..hd {
                        dot += qi[c] * kj[c];
                    }
                    logits[i * n + j] =
                        scale * dot + rel_h_bias[i * gh + ky] + rel_w_bias[i * gw + kx];
                }
            }
            let mut lm = Mat::from_vec(n, n, logits);
            nn::softmax_rows(&mut lm)?;
            for i in 0..n {
                let probs = lm.row(i);
                let o = &mut out[(head * n + i) * hd..(head * n + i + 1) * hd];
                for (j, &pj) in probs.iter().enumerate() {
                    let vj = &vh[j * hd..(j + 1) * hd];
                    for c in 0..hd {
                        o[c] += pj * vj[c];
                    }
                }
            }
        }

        let mut ctx = vec![0.0f32; n * dim];
        for head in 0..nh {
            for r in 0..n {
                let src = (head * n + r) * hd;
                let dst = r * dim + head * hd;
                ctx[dst..dst + hd].copy_from_slice(&out[src..src + hd]);
            }
        }
        p.proj.apply(&Mat::from_vec(n, dim, ctx))
    }

    #[test]
    fn attention_gemm_matches_scalar_reference_with_relpos() -> FocrResult<()> {
        let dim = EMBED_DIM;
        let hd = HEAD_DIM;
        let grid = 3usize;
        let n = grid * grid;
        let rel_rows = 2 * grid - 1;
        let attn = AttnP {
            qkv: Linear {
                w: identity_block_3(dim),
                b: vec![0.0; 3 * dim],
                out: 3 * dim,
                in_: dim,
            },
            proj: Linear {
                w: identity_mat(dim),
                b: vec![0.0; dim],
                out: dim,
                in_: dim,
            },
            rel_pos_h: (0..rel_rows * hd)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.0011)
                .collect(),
            rel_pos_w: (0..rel_rows * hd)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.0009)
                .collect(),
            size_h: grid,
            size_w: grid,
        };
        let x = Mat::from_vec(
            n,
            dim,
            (0..n * dim)
                .map(|i| ((i % 29) as f32 - 14.0) * 0.003)
                .collect(),
        );
        let got = attention(&attn, &x, grid, grid)?;
        let expected = attention_scalar_reference(&attn, &x, grid, grid)?;
        let max_abs = got
            .data
            .iter()
            .zip(expected.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 2.0e-6, "max_abs={max_abs}");
        Ok(())
    }

    #[test]
    fn attention_rejects_malformed_qkv_and_projection_shapes() {
        let dim = EMBED_DIM;
        let x = Mat::from_vec(4, dim, vec![0.0; 4 * dim]);

        let mut bad_qkv = tiny_block(0).attn;
        bad_qkv.qkv.out = 3 * dim - 1;
        bad_qkv.qkv.w = vec![0.0; bad_qkv.qkv.out * bad_qkv.qkv.in_];
        bad_qkv.qkv.b = vec![0.0; bad_qkv.qkv.out];
        assert_err_contains(attention(&bad_qkv, &x, 2, 2), "qkv shape");

        let mut bad_proj = tiny_block(0).attn;
        bad_proj.proj.out = dim - 1;
        bad_proj.proj.w = vec![0.0; bad_proj.proj.out * bad_proj.proj.in_];
        bad_proj.proj.b = vec![0.0; bad_proj.proj.out];
        assert_err_contains(attention(&bad_proj, &x, 2, 2), "proj shape");
    }

    #[test]
    fn block_forward_preserves_shape_global() -> FocrResult<()> {
        // global block (window=0) on a 2x2 grid; zero MLP + zero rel-pos so
        // output = x + attn(LN(x)) with proj/qkv identity. We only assert shape
        // and that the residual path ran (output differs from input generally).
        let blk = tiny_block(0);
        let n = 4;
        let dim = EMBED_DIM;
        let mut data = vec![0.0f32; n * dim];
        for (i, v) in data.iter_mut().enumerate() {
            *v = ((i % 7) as f32) * 0.1 - 0.3;
        }
        let x = Mat::from_vec(n, dim, data);
        let out = block_forward(&blk, &x, 2, 2)?;
        assert_eq!(out.shape(), (n, dim));
        Ok(())
    }

    #[test]
    fn block_forward_windowed_pads_and_unpartitions() -> FocrResult<()> {
        // windowed block, window=3 over a 2x2 grid forces padding to 3x3 then
        // strips it back to 2x2. Shape must round-trip.
        let blk = tiny_block(3);
        let n = 4;
        let dim = EMBED_DIM;
        let data: Vec<f32> = (0..n * dim).map(|i| (i as f32 % 5.0) * 0.01).collect();
        let x = Mat::from_vec(n, dim, data);
        let out = block_forward(&blk, &x, 2, 2)?;
        assert_eq!(out.shape(), (n, dim));
        Ok(())
    }

    #[test]
    fn block_forward_rejects_mlp_residual_shape_mismatch() {
        let mut blk = tiny_block(0);
        blk.lin2.out = EMBED_DIM - 1;
        blk.lin2.w = vec![0.0; blk.lin2.out * blk.lin2.in_];
        blk.lin2.b = vec![0.0; blk.lin2.out];
        let n = 4;
        let dim = EMBED_DIM;
        let data: Vec<f32> = (0..n * dim).map(|i| (i as f32 % 5.0) * 0.01).collect();
        let x = Mat::from_vec(n, dim, data);

        assert_err_contains(
            block_forward(&blk, &x, 2, 2),
            "vision_sam block mlp residual",
        );
    }

    #[test]
    fn attention_zero_relpos_is_uniform_average_for_equal_q() -> FocrResult<()> {
        // With identical token vectors, equal logits => uniform softmax =>
        // attention output == the (shared) value vector (proj identity).
        let dim = EMBED_DIM;
        let hd = HEAD_DIM;
        let size = 2;
        let rel_rows = 2 * size - 1;
        let attn = AttnP {
            qkv: Linear {
                w: identity_block_3(dim),
                b: vec![0.0; 3 * dim],
                out: 3 * dim,
                in_: dim,
            },
            proj: Linear {
                w: identity_mat(dim),
                b: vec![0.0; dim],
                out: dim,
                in_: dim,
            },
            rel_pos_h: vec![0.0; rel_rows * hd],
            rel_pos_w: vec![0.0; rel_rows * hd],
            size_h: size,
            size_w: size,
        };
        // all 4 tokens identical = ones
        let x = Mat::from_vec(4, dim, vec![1.0; 4 * dim]);
        let out = attention(&attn, &x, 2, 2)?;
        assert_eq!(out.shape(), (4, dim));
        // uniform average of identical value vectors -> 1.0 everywhere
        for &v in &out.data {
            assert!((v - 1.0).abs() < 1e-4, "got {v}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "local perf probe; run explicitly with --ignored --nocapture"]
    fn sam_attention_relpos_bias_local_probe() -> FocrResult<()> {
        let dim = EMBED_DIM;
        let hd = HEAD_DIM;
        let grid = 14usize;
        let n = grid * grid;
        let rel_rows = 2 * grid - 1;
        let attn = AttnP {
            qkv: Linear {
                w: identity_block_3(dim),
                b: vec![0.0; 3 * dim],
                out: 3 * dim,
                in_: dim,
            },
            proj: Linear {
                w: identity_mat(dim),
                b: vec![0.0; dim],
                out: dim,
                in_: dim,
            },
            rel_pos_h: (0..rel_rows * hd)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.0007)
                .collect(),
            rel_pos_w: (0..rel_rows * hd)
                .map(|i| ((i % 19) as f32 - 9.0) * 0.0005)
                .collect(),
            size_h: grid,
            size_w: grid,
        };
        let x = Mat::from_vec(
            n,
            dim,
            (0..n * dim)
                .map(|i| ((i % 31) as f32 - 15.0) * 0.002)
                .collect(),
        );

        let runs = std::env::var("FOCR_SAM_ATTN_PROBE_RUNS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(3)
            .max(1);
        let warm = attention(&attn, &x, grid, grid)?;
        let warm_checksum: f32 = warm.data.iter().step_by(97).copied().sum();
        let start = Instant::now();
        let mut checksum = 0.0f32;
        for _ in 0..runs {
            let out = attention(&attn, &x, grid, grid)?;
            checksum += out.data.iter().step_by(97).copied().sum::<f32>();
        }
        let elapsed = start.elapsed();
        let total_ms = elapsed.as_secs_f64() * 1000.0;
        let avg_ms = total_ms / runs as f64;
        assert!(checksum.is_finite());
        println!(
            "{}",
            json!({
                "probe": "sam_attention_relpos_bias_local_probe",
                "grid": grid,
                "tokens": n,
                "dim": dim,
                "heads": NUM_HEADS,
                "head_dim": hd,
                "runs": runs,
                "total_ms": total_ms,
                "avg_ms": avg_ms,
                "warm_checksum": warm_checksum,
                "checksum": checksum
            })
        );
        Ok(())
    }

    #[test]
    fn forward_with_end_to_end_shapes() -> FocrResult<()> {
        // Tiny end-to-end: H=W=32 image -> patch grid 2x2 -> neck keeps 2x2 ->
        // net_2 /2 -> 1x1 -> net_3 /2 -> 1x1. Output [OUT_CH, 1].
        let h = 32;
        let gh = h / PATCH; // 2
        let gw = gh;

        // patch embed conv 3->768 k16 s16
        let patch_embed = Conv {
            w: vec![0.0; EMBED_DIM * 3 * PATCH * PATCH],
            b: Some(vec![0.0; EMBED_DIM]),
            out_ch: EMBED_DIM,
            in_ch: 3,
            kh: PATCH,
            kw: PATCH,
        };

        let blocks: Vec<BlockP> = (0..DEPTH)
            .map(|i| {
                let window = if GLOBAL_BLOCKS.contains(&i) {
                    0
                } else {
                    WINDOW
                };
                // rel-pos tables must be sized to either the window or the grid.
                let size = if window == 0 { gh } else { window };
                let rel_rows = 2 * size - 1;
                let dim = EMBED_DIM;
                let hd = HEAD_DIM;
                BlockP {
                    norm1: LayerNormP {
                        w: vec![1.0; dim],
                        b: vec![0.0; dim],
                    },
                    attn: AttnP {
                        qkv: Linear {
                            w: vec![0.0; 3 * dim * dim],
                            b: vec![0.0; 3 * dim],
                            out: 3 * dim,
                            in_: dim,
                        },
                        proj: Linear {
                            w: vec![0.0; dim * dim],
                            b: vec![0.0; dim],
                            out: dim,
                            in_: dim,
                        },
                        rel_pos_h: vec![0.0; rel_rows * hd],
                        rel_pos_w: vec![0.0; rel_rows * hd],
                        size_h: size,
                        size_w: size,
                    },
                    norm2: LayerNormP {
                        w: vec![1.0; dim],
                        b: vec![0.0; dim],
                    },
                    lin1: Linear {
                        w: vec![0.0; MLP_HIDDEN * dim],
                        b: vec![0.0; MLP_HIDDEN],
                        out: MLP_HIDDEN,
                        in_: dim,
                    },
                    lin2: Linear {
                        w: vec![0.0; dim * MLP_HIDDEN],
                        b: vec![0.0; dim],
                        out: dim,
                        in_: MLP_HIDDEN,
                    },
                    window,
                }
            })
            .collect();

        let w = SamWeights {
            patch_embed,
            pos_embed: vec![0.0; gh * gw * EMBED_DIM],
            pos_grid_h: gh,
            pos_grid_w: gw,
            blocks,
            neck_conv1: Conv {
                w: vec![0.0; NECK_CH * EMBED_DIM],
                b: None,
                out_ch: NECK_CH,
                in_ch: EMBED_DIM,
                kh: 1,
                kw: 1,
            },
            neck_ln1: LayerNormP {
                w: vec![1.0; NECK_CH],
                b: vec![0.0; NECK_CH],
            },
            neck_conv2: Conv {
                w: vec![0.0; NECK_CH * NECK_CH * 9],
                b: None,
                out_ch: NECK_CH,
                in_ch: NECK_CH,
                kh: 3,
                kw: 3,
            },
            neck_ln2: LayerNormP {
                w: vec![1.0; NECK_CH],
                b: vec![0.0; NECK_CH],
            },
            net2: Conv {
                w: vec![0.0; NET2_CH * NECK_CH * 9],
                b: None,
                out_ch: NET2_CH,
                in_ch: NECK_CH,
                kh: 3,
                kw: 3,
            },
            net3: Conv {
                w: vec![0.0; OUT_CH * NET2_CH * 9],
                b: None,
                out_ch: OUT_CH,
                in_ch: NET2_CH,
                kh: 3,
                kw: 3,
            },
        };

        let image = Mat::from_vec(3, h * h, vec![0.5; 3 * h * h]);
        let out = forward_with(&w, &image, h, h)?;
        // gh=2 -> net_2 -> 1 -> net_3 -> 1 ; 1024 channels x 1 spatial.
        assert_eq!(out.shape(), (OUT_CH, 1));
        // all-zero weights -> all-zero feature.
        assert!(out.data.iter().all(|&v| v.abs() < 1e-6));
        Ok(())
    }

    #[test]
    fn forward_with_rejects_bad_channels() {
        let w = tiny_weights_minimal();
        let bad = Mat::from_vec(2, 32 * 32, vec![0.0; 2 * 32 * 32]);
        assert!(forward_with(&w, &bad, 32, 32).is_err());
    }

    #[test]
    fn forward_with_rejects_non_patch_multiple() {
        let w = tiny_weights_minimal();
        // 20 is not a multiple of PATCH(16)
        let img = Mat::from_vec(3, 20 * 20, vec![0.0; 3 * 20 * 20]);
        assert!(forward_with(&w, &img, 20, 20).is_err());
    }

    #[test]
    fn forward_with_rejects_zero_spatial_dims_before_conv() {
        let w = tiny_weights_minimal();
        let img = Mat::from_vec(3, 0, Vec::new());
        assert!(matches!(
            forward_with(&w, &img, 0, 0),
            Err(err) if err.to_string().contains("non-zero")
        ));
    }

    #[test]
    fn forward_with_rejects_spatial_product_overflow_before_conv() {
        let w = tiny_weights_minimal();
        let img = Mat::from_vec(3, 0, Vec::new());
        assert!(matches!(
            forward_with(&w, &img, usize::MAX, 2),
            Err(err) if err.to_string().contains("H*W")
        ));
    }

    /// A structurally-valid all-zero SamWeights for negative-path tests
    /// (shape checks fire before any heavy compute).
    fn tiny_weights_minimal() -> SamWeights {
        let gh = 2;
        let blocks: Vec<BlockP> = (0..DEPTH)
            .map(|i| {
                let window = if GLOBAL_BLOCKS.contains(&i) {
                    0
                } else {
                    WINDOW
                };
                let size = if window == 0 { gh } else { window };
                let rel_rows = 2 * size - 1;
                let dim = EMBED_DIM;
                let hd = HEAD_DIM;
                BlockP {
                    norm1: LayerNormP {
                        w: vec![1.0; dim],
                        b: vec![0.0; dim],
                    },
                    attn: AttnP {
                        qkv: Linear {
                            w: vec![0.0; 3 * dim * dim],
                            b: vec![0.0; 3 * dim],
                            out: 3 * dim,
                            in_: dim,
                        },
                        proj: Linear {
                            w: vec![0.0; dim * dim],
                            b: vec![0.0; dim],
                            out: dim,
                            in_: dim,
                        },
                        rel_pos_h: vec![0.0; rel_rows * hd],
                        rel_pos_w: vec![0.0; rel_rows * hd],
                        size_h: size,
                        size_w: size,
                    },
                    norm2: LayerNormP {
                        w: vec![1.0; dim],
                        b: vec![0.0; dim],
                    },
                    lin1: Linear {
                        w: vec![0.0; MLP_HIDDEN * dim],
                        b: vec![0.0; MLP_HIDDEN],
                        out: MLP_HIDDEN,
                        in_: dim,
                    },
                    lin2: Linear {
                        w: vec![0.0; dim * MLP_HIDDEN],
                        b: vec![0.0; dim],
                        out: dim,
                        in_: MLP_HIDDEN,
                    },
                    window,
                }
            })
            .collect();
        SamWeights {
            patch_embed: Conv {
                w: vec![0.0; EMBED_DIM * 3 * PATCH * PATCH],
                b: Some(vec![0.0; EMBED_DIM]),
                out_ch: EMBED_DIM,
                in_ch: 3,
                kh: PATCH,
                kw: PATCH,
            },
            pos_embed: vec![0.0; gh * gh * EMBED_DIM],
            pos_grid_h: gh,
            pos_grid_w: gh,
            blocks,
            neck_conv1: Conv {
                w: vec![0.0; NECK_CH * EMBED_DIM],
                b: None,
                out_ch: NECK_CH,
                in_ch: EMBED_DIM,
                kh: 1,
                kw: 1,
            },
            neck_ln1: LayerNormP {
                w: vec![1.0; NECK_CH],
                b: vec![0.0; NECK_CH],
            },
            neck_conv2: Conv {
                w: vec![0.0; NECK_CH * NECK_CH * 9],
                b: None,
                out_ch: NECK_CH,
                in_ch: NECK_CH,
                kh: 3,
                kw: 3,
            },
            neck_ln2: LayerNormP {
                w: vec![1.0; NECK_CH],
                b: vec![0.0; NECK_CH],
            },
            net2: Conv {
                w: vec![0.0; NET2_CH * NECK_CH * 9],
                b: None,
                out_ch: NET2_CH,
                in_ch: NECK_CH,
                kh: 3,
                kw: 3,
            },
            net3: Conv {
                w: vec![0.0; OUT_CH * NET2_CH * 9],
                b: None,
                out_ch: OUT_CH,
                in_ch: NET2_CH,
                kh: 3,
                kw: 3,
            },
        }
    }
}
