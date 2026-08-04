//! Hybrid feature concat + projector ([SPEC-051/052],
//! PROPOSED_ARCHITECTURE.md §6.5).
//!
//! This is the 16x token-compression bridge between the two vision towers and
//! the decoder hidden rail. The SAM neck has already done the spatial 64->16
//! downsample (`net_2`/`net_3`, each stride 2: 64 -> 32 -> 16; [SPEC-046]), so a
//! 1024-view yields a 16x16 = 256-token SAM grid. The CLIP tower (fed SAM's
//! `x3` as `patch_embeds`) yields a matching 256-patch grid (+1 leading CLS).
//!
//! The hybrid feature is the **channel** concat of the two towers, in the EXACT
//! order the deployed forward uses (OQ-6, `modeling_unlimitedocr.py:503`):
//!
//! ```text
//! local_features = torch.cat(
//!     (local_features_2[:, 1:],                       # CLIP, drop CLS  -> [N, 1024]
//!      local_features_1.flatten(2).permute(0, 2, 1)), # SAM x3 flattened -> [N, 1024]
//!     dim=-1)                                         # -> [N, 2048]  (CLIP first, SAM second)
//! ```
//!
//! then the single linear projector `model.projector.layers` (`nn.Linear(2048,
//! 1280)`, [SPEC-016/052]) maps `[N, 2048] -> [N, 1280]`. NEITHER hybrid-split
//! MLP variant is used — `projector_type == "linear"` is a plain affine, no
//! GELU, no channel split (OQ-6).
//!
//! Output: the `N = 256` vision tokens per 1024-view at decoder hidden 1280.

use super::nn;
use super::tensor::Mat;
use super::weights::Weights;
use crate::error::{FocrError, FocrResult};

/// CLIP-L hidden width — the first concat operand's channel count ([SPEC-047]).
pub const CLIP_WIDTH: usize = 1024;
/// SAM `x3` channel count — the second concat operand's channel count
/// ([SPEC-046], `net_3` out).
pub const SAM_WIDTH: usize = 1024;
/// Projector input dim = `CLIP_WIDTH + SAM_WIDTH` ([SPEC-016/051]).
pub const PROJ_IN: usize = CLIP_WIDTH + SAM_WIDTH; // 2048
/// Projector output dim = decoder hidden size ([SPEC-016/052]).
pub const PROJ_OUT: usize = 1280;
/// Vision tokens emitted per 1024-view (16x16 SAM/CLIP grid).
pub const TOKENS_PER_VIEW: usize = 256;

/// Immutable projector parameters in the layout consumed by the GEMM facade.
/// Hydrating this once avoids widening and transposing the 2048x1280 weight for
/// every page while retaining the exact affine operation.
#[derive(Debug)]
pub(crate) struct ProjectorWeights {
    weight_t: Mat,
    bias: Vec<f32>,
}

fn checked_shape_mul(context: &str, lhs: usize, rhs: usize, expression: &str) -> FocrResult<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "{context}: usize overflow computing {expression} ({lhs} * {rhs})"
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

/// Concatenate CLIP (sans class token) with the flattened SAM feature and apply
/// the 2048 -> 1280 projector, returning the per-token vision embeddings.
///
/// `clip` is the raw CLIP tower output **with** its leading CLS token at row 0
/// (shape `[N+1, 1024]`); the CLS row is dropped here (`[:, 1:]`). `sam` is the
/// SAM `x3` feature already flattened to `[N, 1024]` (the
/// `flatten(2).permute(0, 2, 1)` of the `[B, 1024, h, w]` map). Both must agree
/// on `N` after the CLS drop.
///
/// The projector weight/bias are read from `weights`
/// (`model.projector.layers.{weight, bias}`, [SPEC-003]). Missing projector
/// tensors surface through the `Weights` accessor as [`FocrError::FormatMismatch`],
/// while the math path is exercised by [`concat_hybrid`] + [`project`] (and their
/// tests).
///
/// # Errors
/// * [`FocrError::Other`] on a shape mismatch (CLIP/SAM rows or channel widths).
/// * [`FocrError::FormatMismatch`] when the loaded artifact does not expose the
///   projector tensors.
pub fn forward(weights: &Weights, clip: &Mat, sam: &Mat) -> FocrResult<Mat> {
    let hybrid = hybrid_from_raw(clip, sam)?;
    let projector = projector_weights_from(weights)?;
    project_transposed(&hybrid, &projector.weight_t, Some(&projector.bias))
}

/// [`forward`] using an already-hydrated projector bundle.
///
/// # Errors
/// The same shape errors as [`forward`].
pub(crate) fn forward_with(projector: &ProjectorWeights, clip: &Mat, sam: &Mat) -> FocrResult<Mat> {
    let hybrid = hybrid_from_raw(clip, sam)?;
    project_transposed(&hybrid, &projector.weight_t, Some(&projector.bias))
}

fn hybrid_from_raw(clip: &Mat, sam: &Mat) -> FocrResult<Mat> {
    // `sam` is the raw vision_sam output [OUT_CH=1024, N=256] (channel-major);
    // the concat wants the flattened [N, 1024] (flatten(2).permute(0,2,1)).
    let sam_t = transpose(sam)?;
    concat_hybrid(clip, &sam_t)
}

/// Build the `[N, 2048]` hybrid feature from the CLIP and SAM tower outputs.
///
/// Exactly `torch.cat((clip[:, 1:], sam), dim=-1)` — CLIP channels first
/// (`0..1024`), SAM channels second (`1024..2048`), per row ([SPEC-051], OQ-6).
/// The CLIP CLS token (row 0) is dropped.
///
/// # Errors
/// Returns [`FocrError::Other`] if `clip` has no rows to drop the CLS from, if
/// the two operands disagree on token count after the drop, or if either width
/// is wrong.
pub fn concat_hybrid(clip: &Mat, sam: &Mat) -> FocrResult<Mat> {
    if clip.rows == 0 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_bridge: CLIP feature has 0 rows (need >=1 to drop the CLS token)"
        )));
    }
    if clip.cols != CLIP_WIDTH {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_bridge: CLIP width {} != {CLIP_WIDTH}",
            clip.cols
        )));
    }
    if sam.cols != SAM_WIDTH {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_bridge: SAM width {} != {SAM_WIDTH}",
            sam.cols
        )));
    }
    ensure_mat_data_len(clip, "vision_bridge concat CLIP")?;
    ensure_mat_data_len(sam, "vision_bridge concat SAM")?;
    // CLIP after dropping the leading CLS row.
    let n = clip.rows - 1;
    if sam.rows != n {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_bridge: token-count mismatch — CLIP[:,1:] has {n} rows, SAM has {} rows",
            sam.rows
        )));
    }

    let out_len = checked_shape_mul("vision_bridge concat", n, PROJ_IN, "tokens*PROJ_IN")?;
    let mut out = Mat::from_vec(n, PROJ_IN, vec![0.0f32; out_len]);
    for r in 0..n {
        // CLIP row r+1 (skip CLS at index 0) -> output cols [0, 1024).
        let clip_src = clip.row(r + 1);
        // SAM row r -> output cols [1024, 2048).
        let sam_src = sam.row(r);
        let dst = out.row_mut(r);
        dst[..CLIP_WIDTH].copy_from_slice(clip_src);
        dst[CLIP_WIDTH..].copy_from_slice(sam_src);
    }
    Ok(out)
}

/// Apply the single linear projector `nn.Linear(2048, 1280)` to the hybrid
/// feature: `y = x @ w^T + b` ([SPEC-052]).
///
/// `w` is the PyTorch-layout projector weight, row-major `[out=1280, in=2048]`
/// (`model.projector.layers.weight`). `bias` is the optional length-1280
/// `model.projector.layers.bias`. The contraction is done through the
/// FrankenTorch sgemm facade ([`nn::matmul`]) after a single transpose of `w`
/// into `[in, out]` GEMM-rhs layout.
///
/// # Errors
/// Returns [`FocrError::Other`] on any shape mismatch (`x.cols != 2048`, wrong
/// `w` length, or a `bias` whose length isn't 1280) or if the inner GEMM fails.
pub fn project(x: &Mat, w: &Mat, bias: Option<&[f32]>) -> FocrResult<Mat> {
    if x.cols != PROJ_IN {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_bridge::project: x.cols {} != projector input {PROJ_IN}",
            x.cols
        )));
    }
    if w.rows != PROJ_OUT || w.cols != PROJ_IN {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_bridge::project: projector weight is [{},{}], expected [{PROJ_OUT},{PROJ_IN}]",
            w.rows,
            w.cols
        )));
    }
    if let Some(b) = bias
        && b.len() != PROJ_OUT
    {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_bridge::project: bias len {} != {PROJ_OUT}",
            b.len()
        )));
    }
    ensure_mat_data_len(x, "vision_bridge::project input")?;
    ensure_mat_data_len(w, "vision_bridge::project weight")?;

    // PyTorch Linear stores weight as [out, in] and computes x @ w^T. The GEMM
    // facade wants [m,k] x [k,n], so transpose w -> [in, out] = [2048, 1280].
    let wt = transpose(w)?;
    project_transposed(x, &wt, bias)
}

fn project_transposed(x: &Mat, wt: &Mat, bias: Option<&[f32]>) -> FocrResult<Mat> {
    if x.cols != PROJ_IN || wt.shape() != (PROJ_IN, PROJ_OUT) {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_bridge::project_transposed: shapes [{}, {}] x [{}, {}] incompatible; \
             expected [N, {PROJ_IN}] x [{PROJ_IN}, {PROJ_OUT}]",
            x.rows,
            x.cols,
            wt.rows,
            wt.cols
        )));
    }
    if let Some(b) = bias
        && b.len() != PROJ_OUT
    {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_bridge::project_transposed: bias len {} != {PROJ_OUT}",
            b.len()
        )));
    }
    ensure_mat_data_len(x, "vision_bridge::project_transposed input")?;
    ensure_mat_data_len(wt, "vision_bridge::project_transposed weight")?;
    let mut y = nn::matmul(x, wt)?;

    if let Some(b) = bias {
        for r in 0..y.rows {
            let row = y.row_mut(r);
            for (c, slot) in row.iter_mut().enumerate() {
                *slot += b[c];
            }
        }
    }
    Ok(y)
}

/// Row-major transpose of `m` (`[r, c] -> [c, r]`).
///
/// Used to turn a PyTorch `[out, in]` Linear weight into the `[in, out]` GEMM
/// right-hand operand that [`nn::matmul`] expects.
fn transpose(m: &Mat) -> FocrResult<Mat> {
    ensure_mat_data_len(m, "vision_bridge transpose input")?;
    let (r, c) = (m.rows, m.cols);
    let out_len = checked_shape_mul("vision_bridge transpose", c, r, "cols*rows")?;
    let mut out = Mat::from_vec(c, r, vec![0.0f32; out_len]);
    for j in 0..c {
        let dst = &mut out.data[j * r..(j + 1) * r];
        for (i, slot) in dst.iter_mut().enumerate() {
            *slot = m.data[i * c + j];
        }
    }
    Ok(out)
}

/// Fetch the projector weight/bias from the loaded weight set.
///
/// `model.projector.layers.weight` is `[1280, 2048]`;
/// `model.projector.layers.bias` is `[1280]` ([SPEC-003/016]). The `Weights`
/// tensor-directory accessors surface a clean [`FocrError::FormatMismatch`] when
/// the tensors are absent. The numeric path ([`project`]) is fully implemented
/// and tested independently of the loader.
fn projector_params(weights: &Weights) -> FocrResult<(Mat, Vec<f32>)> {
    let w = weights.mat("model.projector.layers.weight")?;
    let b = weights.vec("model.projector.layers.bias")?;
    if b.len() != PROJ_OUT {
        return Err(FocrError::FormatMismatch(format!(
            "tensor \"model.projector.layers.bias\" has {} elements; expected {PROJ_OUT}",
            b.len()
        )));
    }
    if w.shape() != (PROJ_OUT, PROJ_IN) {
        return Err(FocrError::FormatMismatch(format!(
            "tensor \"model.projector.layers.weight\" has shape [{}, {}]; expected [{PROJ_OUT}, {PROJ_IN}]",
            w.rows, w.cols
        )));
    }
    Ok((w, b))
}

/// Hydrate and pretranspose the immutable Unlimited-OCR projector once.
pub(crate) fn projector_weights_from(weights: &Weights) -> FocrResult<ProjectorWeights> {
    let (weight, bias) = projector_params(weights)?;
    Ok(ProjectorWeights {
        weight_t: transpose(&weight)?,
        bias,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::focrq::{FocrqBuilder, WriteDType};

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

    fn weights_with_tiny_projector(bias_len: Option<usize>) -> Weights {
        let mut builder = FocrqBuilder::new();
        builder
            .add_tensor(
                "model.projector.layers.weight",
                WriteDType::F32,
                vec![1, 1],
                0.0f32.to_le_bytes().to_vec(),
            )
            .expect("add projector weight");
        if let Some(len) = bias_len {
            builder
                .add_tensor(
                    "model.projector.layers.bias",
                    WriteDType::F32,
                    vec![len],
                    vec![0; len * std::mem::size_of::<f32>()],
                )
                .expect("add projector bias");
        }
        Weights::from_bytes(builder.build()).expect("synthetic projector weights load")
    }

    /// Concat order is CLIP-first / SAM-second, with the CLIP CLS row dropped.
    /// CLIP = [[CLS], [c0..], [c1..]] (3 rows incl. CLS), SAM = [[s0..],[s1..]]
    /// (2 rows). Result rows = 2, cols = 2048; row r = clip[r+1] ++ sam[r].
    #[test]
    fn concat_drops_cls_and_orders_clip_then_sam() -> FocrResult<()> {
        // CLIP rows: 1 CLS + 2 real tokens; fill each row with a row-id marker
        // in col 0 and a distinct value in col 1 so we can trace placement.
        let mut clip = Mat::zeros(3, CLIP_WIDTH);
        clip.set(0, 0, 999.0); // CLS — must be dropped
        clip.set(1, 0, 11.0);
        clip.set(1, 5, 12.0);
        clip.set(2, 0, 21.0);
        clip.set(2, 5, 22.0);

        let mut sam = Mat::zeros(2, SAM_WIDTH);
        sam.set(0, 0, 110.0);
        sam.set(0, 7, 120.0);
        sam.set(1, 0, 210.0);
        sam.set(1, 7, 220.0);

        let h = concat_hybrid(&clip, &sam)?;
        assert_eq!(h.shape(), (2, PROJ_IN));

        // Row 0 takes CLIP token at index 1 (NOT the CLS) in the first 1024
        // cols, SAM token 0 in the second 1024 cols.
        assert_eq!(h.get(0, 0), 11.0);
        assert_eq!(h.get(0, 5), 12.0);
        assert_eq!(h.get(0, CLIP_WIDTH), 110.0); // SAM half starts at col 1024
        assert_eq!(h.get(0, CLIP_WIDTH + 7), 120.0);
        // The dropped CLS value must appear nowhere.
        assert!(h.data.iter().all(|&v| v != 999.0));

        // Row 1.
        assert_eq!(h.get(1, 0), 21.0);
        assert_eq!(h.get(1, 5), 22.0);
        assert_eq!(h.get(1, CLIP_WIDTH), 210.0);
        assert_eq!(h.get(1, CLIP_WIDTH + 7), 220.0);
        Ok(())
    }

    #[test]
    fn concat_rejects_token_count_mismatch() {
        // CLIP has 3 rows -> 2 after CLS drop, SAM has 3 -> mismatch.
        let clip = Mat::zeros(3, CLIP_WIDTH);
        let sam = Mat::zeros(3, SAM_WIDTH);
        assert!(concat_hybrid(&clip, &sam).is_err());
    }

    #[test]
    fn concat_rejects_empty_clip() {
        let clip = Mat::zeros(0, CLIP_WIDTH);
        let sam = Mat::zeros(0, SAM_WIDTH);
        assert!(concat_hybrid(&clip, &sam).is_err());
    }

    #[test]
    fn concat_rejects_bad_widths() {
        let clip = Mat::zeros(2, CLIP_WIDTH + 1);
        let sam = Mat::zeros(1, SAM_WIDTH);
        assert!(concat_hybrid(&clip, &sam).is_err());
        let clip = Mat::zeros(2, CLIP_WIDTH);
        let sam = Mat::zeros(1, SAM_WIDTH - 1);
        assert!(concat_hybrid(&clip, &sam).is_err());
    }

    #[test]
    fn concat_rejects_malformed_clip_data_without_panic() {
        let clip = Mat {
            rows: 2,
            cols: CLIP_WIDTH,
            data: vec![0.0; CLIP_WIDTH],
        };
        let sam = Mat::zeros(1, SAM_WIDTH);
        assert_err_contains(concat_hybrid(&clip, &sam), "concat CLIP");
    }

    #[test]
    fn concat_rejects_malformed_sam_data_without_panic() {
        let clip = Mat::zeros(2, CLIP_WIDTH);
        let sam = Mat {
            rows: 1,
            cols: SAM_WIDTH,
            data: Vec::new(),
        };
        assert_err_contains(concat_hybrid(&clip, &sam), "concat SAM");
    }

    /// Transpose turns `[out, in]` into `[in, out]` with element (i,j)->(j,i).
    #[test]
    fn transpose_swaps_indices() -> FocrResult<()> {
        let m = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = super::transpose(&m)?;
        assert_eq!(t.shape(), (3, 2));
        // original (0,1)=2 -> (1,0); (1,2)=6 -> (2,1)
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(1, 0), 2.0);
        assert_eq!(t.get(2, 0), 3.0);
        assert_eq!(t.get(0, 1), 4.0);
        assert_eq!(t.get(1, 1), 5.0);
        assert_eq!(t.get(2, 1), 6.0);
        Ok(())
    }

    #[test]
    fn transpose_rejects_malformed_data_without_panic() {
        let m = Mat {
            rows: 2,
            cols: 3,
            data: vec![0.0; 5],
        };
        assert_err_contains(super::transpose(&m), "data len");
    }

    #[test]
    fn transpose_rejects_shape_product_overflow_without_panic() {
        let m = Mat {
            rows: usize::MAX,
            cols: 2,
            data: Vec::new(),
        };
        assert_err_contains(super::transpose(&m), "rows*cols");
    }

    /// `project` must reproduce PyTorch `nn.Linear`: y = x @ w^T + b, with w in
    /// `[out, in]` layout. Use a tiny 2-in / 3-out problem stretched to the real
    /// 2048->1280 by padding with zeros, so the math is hand-checkable.
    ///
    /// Here we test the real shape with an identity-ish projector: a weight that
    /// selects the first PROJ_OUT input channels (w[o, o] = 1), so y[:, o] =
    /// x[:, o] for o < PROJ_OUT, plus a per-output bias.
    #[test]
    fn project_matches_linear_semantics() -> FocrResult<()> {
        let n = 2;
        // x: distinct value per (row, col-of-interest).
        let mut x = Mat::zeros(n, PROJ_IN);
        x.set(0, 0, 1.0);
        x.set(0, 1, 2.0);
        x.set(0, PROJ_OUT - 1, 7.0); // last selected output channel
        x.set(1, 0, 10.0);
        x.set(1, 1, 20.0);

        // w[o, in]: identity on the first PROJ_OUT input channels.
        let mut w = Mat::zeros(PROJ_OUT, PROJ_IN);
        for o in 0..PROJ_OUT {
            w.set(o, o, 1.0);
        }
        // bias: +0.5 on output channel 1 only.
        let mut bias = vec![0.0f32; PROJ_OUT];
        bias[1] = 0.5;

        let y = project(&x, &w, Some(&bias))?;
        assert_eq!(y.shape(), (n, PROJ_OUT));
        let wt = transpose(&w)?;
        let cached = project_transposed(&x, &wt, Some(&bias))?;
        assert_eq!(
            y.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            cached.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "pretransposing once must preserve every projector output bit"
        );

        // y[r, o] = x[r, o] (+ bias[o]).
        assert!((y.get(0, 0) - 1.0).abs() < 1e-5);
        assert!((y.get(0, 1) - (2.0 + 0.5)).abs() < 1e-5);
        assert!((y.get(0, PROJ_OUT - 1) - 7.0).abs() < 1e-5);
        assert!((y.get(1, 0) - 10.0).abs() < 1e-5);
        assert!((y.get(1, 1) - (20.0 + 0.5)).abs() < 1e-5);
        // An unset selected channel stays 0 (+ no bias).
        assert!(y.get(0, 5).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn project_no_bias_is_pure_gemm() -> FocrResult<()> {
        let n = 1;
        let mut x = Mat::zeros(n, PROJ_IN);
        x.set(0, 3, 4.0);
        let mut w = Mat::zeros(PROJ_OUT, PROJ_IN);
        // output channel 3 reads input channel 3 with gain 2.
        w.set(3, 3, 2.0);
        let y = project(&x, &w, None)?;
        assert!((y.get(0, 3) - 8.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn project_rejects_bad_input_width() {
        let x = Mat::zeros(1, PROJ_IN - 1);
        let w = Mat::zeros(PROJ_OUT, PROJ_IN);
        assert!(project(&x, &w, None).is_err());
    }

    #[test]
    fn project_rejects_bad_weight_shape() {
        let x = Mat::zeros(1, PROJ_IN);
        let w = Mat::zeros(PROJ_OUT, PROJ_IN + 1);
        assert!(project(&x, &w, None).is_err());
    }

    #[test]
    fn project_rejects_bad_bias_len() {
        let x = Mat::zeros(1, PROJ_IN);
        let w = Mat::zeros(PROJ_OUT, PROJ_IN);
        let bias = vec![0.0f32; PROJ_OUT - 1];
        assert!(project(&x, &w, Some(&bias)).is_err());
    }

    #[test]
    fn project_rejects_malformed_input_data_without_panic() {
        let x = Mat {
            rows: 2,
            cols: PROJ_IN,
            data: vec![0.0; PROJ_IN],
        };
        let w = Mat::zeros(PROJ_OUT, PROJ_IN);
        assert_err_contains(project(&x, &w, None), "project input");
    }

    #[test]
    fn project_rejects_malformed_weight_data_without_panic() {
        let x = Mat::zeros(1, PROJ_IN);
        let w = Mat {
            rows: PROJ_OUT,
            cols: PROJ_IN,
            data: Vec::new(),
        };
        assert_err_contains(project(&x, &w, None), "project weight");
    }

    #[test]
    fn forward_rejects_malformed_raw_sam_before_projector_lookup() {
        let w = Weights::default();
        let clip = Mat::zeros(TOKENS_PER_VIEW + 1, CLIP_WIDTH);
        let sam = Mat {
            rows: SAM_WIDTH,
            cols: TOKENS_PER_VIEW,
            data: Vec::new(),
        };
        assert_err_contains(forward(&w, &clip, &sam), "transpose input");
    }

    /// End-to-end shape contract for one 1024-view: 256+1 CLIP rows + 256 SAM
    /// rows -> 256 hybrid rows -> 256 projected tokens at hidden 1280.
    #[test]
    fn full_bridge_shapes_for_one_view() -> FocrResult<()> {
        let clip = Mat::zeros(TOKENS_PER_VIEW + 1, CLIP_WIDTH); // +1 CLS
        let sam = Mat::zeros(TOKENS_PER_VIEW, SAM_WIDTH);
        let hybrid = concat_hybrid(&clip, &sam)?;
        assert_eq!(hybrid.shape(), (TOKENS_PER_VIEW, PROJ_IN));

        let w = Mat::zeros(PROJ_OUT, PROJ_IN);
        let y = project(&hybrid, &w, None)?;
        assert_eq!(y.shape(), (TOKENS_PER_VIEW, PROJ_OUT));
        Ok(())
    }

    #[test]
    fn forward_missing_projector_tensor_errors_cleanly() {
        // The numeric path is done; an empty default weight set should surface
        // the missing projector tensor cleanly through the accessor layer.
        let w = Weights::default();
        let clip = Mat::zeros(TOKENS_PER_VIEW + 1, CLIP_WIDTH);
        let sam = Mat::zeros(SAM_WIDTH, TOKENS_PER_VIEW);
        let r = forward(&w, &clip, &sam);
        assert!(matches!(r, Err(FocrError::FormatMismatch(_))));
    }

    #[test]
    fn projector_params_requires_bias_tensor() {
        let weights = weights_with_tiny_projector(None);
        let err = projector_params(&weights).expect_err("missing affine bias must fail closed");
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(err.to_string().contains("model.projector.layers.bias"));
    }

    #[test]
    fn projector_params_rejects_wrong_bias_length() {
        let weights = weights_with_tiny_projector(Some(PROJ_OUT - 1));
        let err = projector_params(&weights).expect_err("malformed affine bias must fail closed");
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(err.to_string().contains("expected 1280"));
    }

    #[test]
    fn constants_are_consistent() {
        assert_eq!(PROJ_IN, 2048);
        assert_eq!(PROJ_OUT, 1280);
        assert_eq!(CLIP_WIDTH + SAM_WIDTH, PROJ_IN);
        assert_eq!(TOKENS_PER_VIEW, 256);
    }
}
