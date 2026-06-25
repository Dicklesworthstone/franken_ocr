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
/// (`model.projector.layers.{weight, bias}`, [SPEC-003]). The `Weights` accessor
/// surface is still being built by the loader wave; until it exposes the
/// projector tensors this returns [`FocrError::NotImplemented`], but the whole
/// math path is exercised by [`concat_hybrid`] + [`project`] (and their tests).
///
/// # Errors
/// * [`FocrError::Other`] on a shape mismatch (CLIP/SAM rows or channel widths).
/// * [`FocrError::NotImplemented`] until `Weights` exposes the projector tensors.
pub fn forward(weights: &Weights, clip: &Mat, sam: &Mat) -> FocrResult<Mat> {
    let hybrid = concat_hybrid(clip, sam)?;
    let (w, b) = projector_params(weights)?;
    project(&hybrid, &w, b.as_deref())
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
    // CLIP after dropping the leading CLS row.
    let n = clip.rows - 1;
    if sam.rows != n {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_bridge: token-count mismatch — CLIP[:,1:] has {n} rows, SAM has {} rows",
            sam.rows
        )));
    }

    let mut out = Mat::zeros(n, PROJ_IN);
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
    if let Some(b) = bias {
        if b.len() != PROJ_OUT {
            return Err(FocrError::Other(anyhow::anyhow!(
                "vision_bridge::project: bias len {} != {PROJ_OUT}",
                b.len()
            )));
        }
    }

    // PyTorch Linear stores weight as [out, in] and computes x @ w^T. The GEMM
    // facade wants [m,k] x [k,n], so transpose w -> [in, out] = [2048, 1280].
    let wt = transpose(w);
    let mut y = nn::matmul(x, &wt)?;

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
fn transpose(m: &Mat) -> Mat {
    let (r, c) = (m.rows, m.cols);
    let mut out = Mat::zeros(c, r);
    for i in 0..r {
        let src = m.row(i);
        for j in 0..c {
            out.data[j * r + i] = src[j];
        }
    }
    out
}

/// Fetch the projector weight/bias from the loaded weight set.
///
/// `model.projector.layers.weight` is `[1280, 2048]`;
/// `model.projector.layers.bias` is `[1280]` ([SPEC-003/016]). The `Weights`
/// tensor-directory accessors are still being landed by the loader wave; this
/// returns [`FocrError::NotImplemented`] until they exist. The numeric path
/// ([`project`]) is fully implemented and tested independently of the loader.
fn projector_params(_weights: &Weights) -> FocrResult<(Mat, Option<Vec<f32>>)> {
    // NOTE(loader-handoff): replace with the real lookups once `Weights` exposes
    // a tensor accessor, e.g.:
    //   let w = weights.mat("model.projector.layers.weight", PROJ_OUT, PROJ_IN)?;
    //   let b = weights.vec("model.projector.layers.bias").ok();
    //   Ok((w, b))
    Err(FocrError::NotImplemented(
        "native_engine::vision_bridge::projector_params — awaiting Weights tensor accessor \
         (model.projector.layers.{weight,bias}); concat_hybrid/project are implemented and tested"
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Concat order is CLIP-first / SAM-second, with the CLIP CLS row dropped.
    /// CLIP = [[CLS], [c0..], [c1..]] (3 rows incl. CLS), SAM = [[s0..],[s1..]]
    /// (2 rows). Result rows = 2, cols = 2048; row r = clip[r+1] ++ sam[r].
    #[test]
    fn concat_drops_cls_and_orders_clip_then_sam() {
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

        let h = concat_hybrid(&clip, &sam).unwrap();
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

    /// Transpose turns `[out, in]` into `[in, out]` with element (i,j)->(j,i).
    #[test]
    fn transpose_swaps_indices() {
        let m = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = super::transpose(&m);
        assert_eq!(t.shape(), (3, 2));
        // original (0,1)=2 -> (1,0); (1,2)=6 -> (2,1)
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(1, 0), 2.0);
        assert_eq!(t.get(2, 0), 3.0);
        assert_eq!(t.get(0, 1), 4.0);
        assert_eq!(t.get(1, 1), 5.0);
        assert_eq!(t.get(2, 1), 6.0);
    }

    /// `project` must reproduce PyTorch `nn.Linear`: y = x @ w^T + b, with w in
    /// `[out, in]` layout. Use a tiny 2-in / 3-out problem stretched to the real
    /// 2048->1280 by padding with zeros, so the math is hand-checkable.
    ///
    /// Here we test the real shape with an identity-ish projector: a weight that
    /// selects the first PROJ_OUT input channels (w[o, o] = 1), so y[:, o] =
    /// x[:, o] for o < PROJ_OUT, plus a per-output bias.
    #[test]
    fn project_matches_linear_semantics() {
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

        let y = project(&x, &w, Some(&bias)).unwrap();
        assert_eq!(y.shape(), (n, PROJ_OUT));

        // y[r, o] = x[r, o] (+ bias[o]).
        assert!((y.get(0, 0) - 1.0).abs() < 1e-5);
        assert!((y.get(0, 1) - (2.0 + 0.5)).abs() < 1e-5);
        assert!((y.get(0, PROJ_OUT - 1) - 7.0).abs() < 1e-5);
        assert!((y.get(1, 0) - 10.0).abs() < 1e-5);
        assert!((y.get(1, 1) - (20.0 + 0.5)).abs() < 1e-5);
        // An unset selected channel stays 0 (+ no bias).
        assert!(y.get(0, 5).abs() < 1e-5);
    }

    #[test]
    fn project_no_bias_is_pure_gemm() {
        let n = 1;
        let mut x = Mat::zeros(n, PROJ_IN);
        x.set(0, 3, 4.0);
        let mut w = Mat::zeros(PROJ_OUT, PROJ_IN);
        // output channel 3 reads input channel 3 with gain 2.
        w.set(3, 3, 2.0);
        let y = project(&x, &w, None).unwrap();
        assert!((y.get(0, 3) - 8.0).abs() < 1e-5);
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

    /// End-to-end shape contract for one 1024-view: 256+1 CLIP rows + 256 SAM
    /// rows -> 256 hybrid rows -> 256 projected tokens at hidden 1280.
    #[test]
    fn full_bridge_shapes_for_one_view() {
        let clip = Mat::zeros(TOKENS_PER_VIEW + 1, CLIP_WIDTH); // +1 CLS
        let sam = Mat::zeros(TOKENS_PER_VIEW, SAM_WIDTH);
        let hybrid = concat_hybrid(&clip, &sam).unwrap();
        assert_eq!(hybrid.shape(), (TOKENS_PER_VIEW, PROJ_IN));

        let w = Mat::zeros(PROJ_OUT, PROJ_IN);
        let y = project(&hybrid, &w, None).unwrap();
        assert_eq!(y.shape(), (TOKENS_PER_VIEW, PROJ_OUT));
    }

    #[test]
    fn forward_awaits_weights_accessor() {
        // The numeric path is done; only the loader handoff is pending.
        let w = Weights::default();
        let clip = Mat::zeros(TOKENS_PER_VIEW + 1, CLIP_WIDTH);
        let sam = Mat::zeros(TOKENS_PER_VIEW, SAM_WIDTH);
        let r = forward(&w, &clip, &sam);
        assert!(matches!(r, Err(FocrError::NotImplemented(_))));
    }

    #[test]
    fn constants_are_consistent() {
        assert_eq!(PROJ_IN, 2048);
        assert_eq!(PROJ_OUT, 1280);
        assert_eq!(CLIP_WIDTH + SAM_WIDTH, PROJ_IN);
        assert_eq!(TOKENS_PER_VIEW, 256);
    }
}
