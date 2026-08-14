//! TrOMR encoder (E3, bd-3jo6.5.3) — the fifth model lane's vision half
//! (census `docs/zoo/tromr-spec.md` §2a/§2b): a hybrid ResNetV2+ViT over one
//! grayscale staff crop `(1, 128, W)`, W ≤ 1280 a multiple of 16.
//!
//! Graph: TF-'SAME' stem `conv 1→64 k7 s2` → GN32+ReLU → −∞-pad max-pool k3
//! s2 → post-act Bottleneck stages `[2, 3, 7]` (widths 256/512/1024, strides
//! 1/2/2) → `(1024, 8, W/16)` → 1×1 proj to 256 + cls token + CROP-INDEXED
//! learned positions (row-major over an 80-wide table) → 4 pre-LN ViT blocks
//! (8 heads × 32, fused qkv, exact-erf GELU MLP 1024) → final LayerNorm →
//! `[1 + 8·W/16, 256]` — the cls token IS part of the decoder's
//! cross-attention context (§3: the connector is Identity).
//!
//! The stored backbone convs are PRE-WS-FOLDED (E2's export invokes timm's
//! own standardization arithmetic), so every conv here is a plain
//! [`nn::conv2d`] over a [`nn::tf_same_pad`]-prepared input — no runtime
//! weight standardization exists (spec §10.3). No conv biases anywhere in the
//! backbone; the backbone final norm is Identity (both census-confirmed
//! absent from the checkpoint).
//!
//! The 4-head AR decoder (E4, spec §4/§5) lives here too — a self-contained
//! x-transformers graph that does NOT ride `decoder_qwen2` (§10 non-fit):
//! 4 layers of ('a' causal self-attn, 'c' cross-attn over the encoder
//! context, 'f' GEGLU ff), all pre-LN (eps 1e-5) + residual, inner 512 ≠ dim
//! 256, GLU-gated bias-free `on_attn` out-projections, a summed 3-embedding
//! input (+ scaled learned positions), and FOUR parallel heads off one final
//! norm. [`generate`] is the port's deterministic per-head-argmax default;
//! explicit recognition options can select upstream top-k/T=0.2 sampling
//! with a caller-declared seed.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{FocrError, FocrResult};
use crate::music_diagnostics::TromrAttemptStage;
use crate::music_execution::{TromrExecutionContext, is_terminal_execution_error};

use super::nn;
use super::tensor::Mat;
use super::vision_sam::Linear;
use super::weights::Weights;

trait ExecutionControl {
    fn checkpoint(&self, stage: &'static str) -> FocrResult<()>;
    fn begin_forward_attempt(&mut self, stage: &'static str) -> FocrResult<()>;
    fn record_staff_detection(&mut self, _elapsed: std::time::Duration) {}
    fn set_attempt_location(&mut self, _detection_index: usize, _segment_index: Option<usize>) {}
    fn record_attempt_stage(&mut self, _stage: TromrAttemptStage, _elapsed: std::time::Duration) {}
    fn finish_attempt<T>(&mut self, _result: &Result<T, FocrError>) {}
    fn record_page_assembly(&mut self, _elapsed: std::time::Duration) {}
}

struct LegacyExecutionControl;

impl ExecutionControl for LegacyExecutionControl {
    fn checkpoint(&self, _stage: &'static str) -> FocrResult<()> {
        crate::cancel_checkpoint()
    }

    fn begin_forward_attempt(&mut self, _stage: &'static str) -> FocrResult<()> {
        crate::cancel_checkpoint()
    }
}

impl ExecutionControl for TromrExecutionContext {
    fn checkpoint(&self, stage: &'static str) -> FocrResult<()> {
        TromrExecutionContext::checkpoint(self, stage)
    }

    fn begin_forward_attempt(&mut self, stage: &'static str) -> FocrResult<()> {
        TromrExecutionContext::begin_forward_attempt(self, stage)
    }

    fn record_staff_detection(&mut self, elapsed: std::time::Duration) {
        TromrExecutionContext::record_staff_detection(self, elapsed);
    }

    fn set_attempt_location(&mut self, detection_index: usize, segment_index: Option<usize>) {
        TromrExecutionContext::set_attempt_location(self, detection_index, segment_index);
    }

    fn record_attempt_stage(&mut self, stage: TromrAttemptStage, elapsed: std::time::Duration) {
        TromrExecutionContext::record_attempt_stage(self, stage, elapsed);
    }

    fn finish_attempt<T>(&mut self, result: &Result<T, FocrError>) {
        TromrExecutionContext::finish_attempt(self, result);
    }

    fn record_page_assembly(&mut self, elapsed: std::time::Duration) {
        TromrExecutionContext::record_page_assembly(self, elapsed);
    }
}

/// Staff-crop input height (config `max_height`; spec §6 resizes to this).
pub const IMG_H: usize = 128;
/// ViT patch stride — the backbone's total 16× downsample (spec §2b).
pub const PATCH: usize = 16;
/// Encoder/decoder shared width (`emb_dim == dim == 256`, §3).
pub const DIM: usize = 256;
/// The learned position table is laid out for this many patch COLUMNS
/// (1280/16); a narrower crop crop-indexes its top-left block (§2b).
pub const POS_COLS: usize = 80;
/// Patch rows for the fixed 128-high input (128/16).
pub const POS_ROWS: usize = 8;
const VIT_HEADS: usize = 8;
const VIT_HEAD_DIM: usize = 32;
const GN_GROUPS: usize = 32;
const GN_EPS: f32 = 1e-5;
const LN_EPS: f32 = 1e-6;

/// One flat batch-1 NCHW feature map — the backbone currency.
struct Feature {
    data: Vec<f32>,
    ch: usize,
    h: usize,
    w: usize,
}

/// A backbone conv (no bias — census) + its following GroupNorm params.
/// `norm` is `None` only where the graph has a bare conv (never happens in
/// this backbone: every conv is followed by a GN, with or without ReLU).
struct ConvGn {
    w: Vec<f32>,
    out_ch: usize,
    in_ch: usize,
    k: usize,
    stride: usize,
    gn_w: Vec<f32>,
    gn_b: Vec<f32>,
}

impl ConvGn {
    /// TF-'SAME' pad (zero fill) → conv → GroupNorm(32, 1e-5) with optional
    /// fused ReLU.
    fn apply(&self, x: &Feature, relu: bool) -> FocrResult<Feature> {
        let (padded, ph, pw) = nn::tf_same_pad(
            &x.data,
            1,
            x.ch,
            x.h,
            x.w,
            self.k,
            self.k,
            self.stride,
            self.stride,
            0.0,
        );
        let (oh, ow) = (x.h.div_ceil(self.stride), x.w.div_ceil(self.stride));
        let mut data = nn::conv2d(
            &padded,
            &self.w,
            None,
            1,
            self.in_ch,
            ph,
            pw,
            self.k,
            self.k,
            oh,
            ow,
            self.stride,
            self.stride,
            self.out_ch,
        );
        nn::group_norm(
            &mut data,
            1,
            self.out_ch,
            oh * ow,
            GN_GROUPS,
            GN_EPS,
            &self.gn_w,
            &self.gn_b,
            relu,
        )?;
        Ok(Feature {
            data,
            ch: self.out_ch,
            h: oh,
            w: ow,
        })
    }
}

/// One post-act Bottleneck block (timm ResNetV2, preact=False — spec §2a):
/// `conv1 1×1 → GN+ReLU → conv2 3×3 (stride) → GN+ReLU → conv3 1×1 → GN(no
/// act) → + shortcut → ReLU`; block 0 of a stage downsamples the shortcut
/// with `1×1 (stride) → GN(no act)`.
struct Bottleneck {
    conv1: ConvGn,
    conv2: ConvGn,
    conv3: ConvGn,
    downsample: Option<ConvGn>,
}

impl Bottleneck {
    fn apply(&self, x: &Feature) -> FocrResult<Feature> {
        let shortcut = match &self.downsample {
            Some(d) => d.apply(x, false)?,
            None => Feature {
                data: x.data.clone(),
                ch: x.ch,
                h: x.h,
                w: x.w,
            },
        };
        let h = self.conv1.apply(x, true)?;
        let h = self.conv2.apply(&h, true)?;
        let mut h = self.conv3.apply(&h, false)?;
        if h.data.len() != shortcut.data.len() {
            return Err(FocrError::Other(anyhow::anyhow!(
                "tromr bottleneck: residual len {} != shortcut len {}",
                h.data.len(),
                shortcut.data.len()
            )));
        }
        for (a, b) in h.data.iter_mut().zip(shortcut.data.iter()) {
            *a = (*a + b).max(0.0);
        }
        Ok(h)
    }
}

/// One pre-LN ViT block (spec §2b): LN(1e-6) → fused-qkv MHA (8×32, scale
/// 32^-0.5) → +res; LN → fc1 1024 → exact-erf GELU → fc2 → +res.
struct VitBlock {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    qkv: Linear,
    proj: Linear,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    fc1: Linear,
    fc2: Linear,
}

/// The hydrated encoder weights.
pub struct TromrEncoderW {
    stem: ConvGn,
    stages: Vec<Vec<Bottleneck>>,
    patch_proj: Linear,
    cls_token: Vec<f32>,
    pos_embed: Vec<f32>,
    blocks: Vec<VitBlock>,
    final_ln_w: Vec<f32>,
    final_ln_b: Vec<f32>,
}

impl TromrEncoderW {
    /// Hydrate from the (WS-pre-folded) artifact — spec §12 names verbatim.
    ///
    /// # Errors
    /// A missing tensor or a shape violation.
    pub fn build(weights: &Weights) -> FocrResult<Self> {
        let b = "encoder.patch_embed.backbone.";
        let conv_gn = |conv: String,
                       norm: String,
                       out_ch: usize,
                       in_ch: usize,
                       k: usize,
                       stride: usize|
         -> FocrResult<ConvGn> {
            Ok(ConvGn {
                w: weights.vec(&conv)?,
                out_ch,
                in_ch,
                k,
                stride,
                gn_w: weights.vec(&format!("{norm}.weight"))?,
                gn_b: weights.vec(&format!("{norm}.bias"))?,
            })
        };

        let stem = conv_gn(
            format!("{b}stem.conv.weight"),
            format!("{b}stem.norm"),
            64,
            1,
            7,
            2,
        )?;

        // Stages [2, 3, 7]; (in, mid, out, stride) per census §2a/§12.
        let plan: [(usize, usize, usize, usize, usize); 3] = [
            (2, 64, 64, 256, 1),
            (3, 256, 128, 512, 2),
            (7, 512, 256, 1024, 2),
        ];
        let mut stages = Vec::with_capacity(3);
        for (s, &(blocks_n, stage_in, mid, out, stage_stride)) in plan.iter().enumerate() {
            let mut blocks = Vec::with_capacity(blocks_n);
            for blk in 0..blocks_n {
                let p = format!("{b}stages.{s}.blocks.{blk}.");
                let (in_ch, stride) = if blk == 0 {
                    (stage_in, stage_stride)
                } else {
                    (out, 1)
                };
                let downsample = if blk == 0 {
                    Some(conv_gn(
                        format!("{p}downsample.conv.weight"),
                        format!("{p}downsample.norm"),
                        out,
                        in_ch,
                        1,
                        stride,
                    )?)
                } else {
                    None
                };
                blocks.push(Bottleneck {
                    conv1: conv_gn(
                        format!("{p}conv1.weight"),
                        format!("{p}norm1"),
                        mid,
                        in_ch,
                        1,
                        1,
                    )?,
                    conv2: conv_gn(
                        format!("{p}conv2.weight"),
                        format!("{p}norm2"),
                        mid,
                        mid,
                        3,
                        stride,
                    )?,
                    conv3: conv_gn(
                        format!("{p}conv3.weight"),
                        format!("{p}norm3"),
                        out,
                        mid,
                        1,
                        1,
                    )?,
                    downsample,
                });
            }
            stages.push(blocks);
        }

        let lin = |wname: String, bname: String, out: usize, in_: usize| -> FocrResult<Linear> {
            Linear::from_row_major(&weights.vec(&wname)?, weights.vec(&bname)?, out, in_)
        };
        let mut blocks = Vec::with_capacity(4);
        for i in 0..4 {
            let p = format!("encoder.blocks.{i}.");
            blocks.push(VitBlock {
                ln1_w: weights.vec(&format!("{p}norm1.weight"))?,
                ln1_b: weights.vec(&format!("{p}norm1.bias"))?,
                qkv: lin(
                    format!("{p}attn.qkv.weight"),
                    format!("{p}attn.qkv.bias"),
                    3 * DIM,
                    DIM,
                )?,
                proj: lin(
                    format!("{p}attn.proj.weight"),
                    format!("{p}attn.proj.bias"),
                    DIM,
                    DIM,
                )?,
                ln2_w: weights.vec(&format!("{p}norm2.weight"))?,
                ln2_b: weights.vec(&format!("{p}norm2.bias"))?,
                fc1: lin(
                    format!("{p}mlp.fc1.weight"),
                    format!("{p}mlp.fc1.bias"),
                    4 * DIM,
                    DIM,
                )?,
                fc2: lin(
                    format!("{p}mlp.fc2.weight"),
                    format!("{p}mlp.fc2.bias"),
                    DIM,
                    4 * DIM,
                )?,
            });
        }

        Ok(Self {
            stem,
            stages,
            patch_proj: lin(
                "encoder.patch_embed.proj.weight".into(),
                "encoder.patch_embed.proj.bias".into(),
                DIM,
                1024,
            )?,
            cls_token: weights.vec("encoder.cls_token")?,
            pos_embed: weights.vec("encoder.pos_embed")?,
            blocks,
            final_ln_w: weights.vec("encoder.norm.weight")?,
            final_ln_b: weights.vec("encoder.norm.bias")?,
        })
    }
}

/// The ResNetV2 backbone: staff tensor `(1, 128, W)` → `(1024, 8, W/16)`.
fn backbone(w: &TromrEncoderW, pixels: &[f32], width: usize) -> FocrResult<Feature> {
    if width == 0 || !width.is_multiple_of(PATCH) || width > POS_COLS * PATCH {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr: width {width} must be a non-zero multiple of {PATCH} <= {} (spec §2b \
             crop-indexed positions go undefined past 1280)",
            POS_COLS * PATCH
        )));
    }
    if pixels.len() != IMG_H * width {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr: pixel buffer {} != 1*{IMG_H}*{width}",
            pixels.len()
        )));
    }
    let x = Feature {
        data: pixels.to_vec(),
        ch: 1,
        h: IMG_H,
        w: width,
    };
    // Stem: conv7 s2 (GN+ReLU) then the −∞-padded s2 max-pool.
    let x = w.stem.apply(&x, true)?;
    let (padded, ph, pw) =
        nn::tf_same_pad(&x.data, 1, x.ch, x.h, x.w, 3, 3, 2, 2, f32::NEG_INFINITY);
    let (oh, ow) = (x.h.div_ceil(2), x.w.div_ceil(2));
    let mut x = Feature {
        data: nn::max_pool2d(&padded, 1, x.ch, ph, pw, 3, 2, oh, ow),
        ch: x.ch,
        h: oh,
        w: ow,
    };
    for stage in &w.stages {
        for block in stage {
            x = block.apply(&x)?;
        }
    }
    Ok(x)
}

/// Channel-major `(C, H·W)` → token-major `[H·W, C]`.
fn tokens_from_feature(f: &Feature) -> Mat {
    let spatial = f.h * f.w;
    let mut out = vec![0.0f32; spatial * f.ch];
    for c in 0..f.ch {
        for s in 0..spatial {
            out[s * f.ch + c] = f.data[c * spatial + s];
        }
    }
    Mat::from_vec(spatial, f.ch, out)
}

/// Fused-qkv bidirectional MHA (8 heads × 32, scale 32^-0.5).
fn self_attention(blk: &VitBlock, x: &Mat) -> FocrResult<Mat> {
    let seq = x.rows;
    let qkv = blk.qkv.apply(x)?; // [seq, 768] = q|k|v
    let head_span = seq * VIT_HEAD_DIM;
    let mut qf = vec![0.0f32; VIT_HEADS * head_span];
    let mut kf = vec![0.0f32; VIT_HEADS * head_span];
    let mut vf = vec![0.0f32; VIT_HEADS * head_span];
    for s in 0..seq {
        let row = qkv.row(s);
        for h in 0..VIT_HEADS {
            let dst = h * head_span + s * VIT_HEAD_DIM;
            let src = h * VIT_HEAD_DIM;
            qf[dst..dst + VIT_HEAD_DIM].copy_from_slice(&row[src..src + VIT_HEAD_DIM]);
            kf[dst..dst + VIT_HEAD_DIM].copy_from_slice(&row[DIM + src..DIM + src + VIT_HEAD_DIM]);
            vf[dst..dst + VIT_HEAD_DIM]
                .copy_from_slice(&row[2 * DIM + src..2 * DIM + src + VIT_HEAD_DIM]);
        }
    }
    let scale = 1.0 / (VIT_HEAD_DIM as f32).sqrt();
    let ctx = nn::sdpa(
        &qf,
        &kf,
        &vf,
        VIT_HEADS,
        seq,
        seq,
        VIT_HEAD_DIM,
        VIT_HEAD_DIM,
        scale,
        false,
    );
    // Head-major back to [seq, 256].
    let mut merged = vec![0.0f32; seq * DIM];
    for h in 0..VIT_HEADS {
        for s in 0..seq {
            let src = h * head_span + s * VIT_HEAD_DIM;
            let dst = s * DIM + h * VIT_HEAD_DIM;
            merged[dst..dst + VIT_HEAD_DIM].copy_from_slice(&ctx[src..src + VIT_HEAD_DIM]);
        }
    }
    blk.proj.apply(&Mat::from_vec(seq, DIM, merged))
}

fn add_assign(x: &mut Mat, y: &Mat) -> FocrResult<()> {
    if x.rows != y.rows || x.cols != y.cols {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr add_assign: [{}, {}] += [{}, {}]",
            x.rows,
            x.cols,
            y.rows,
            y.cols
        )));
    }
    for (a, b) in x.data.iter_mut().zip(y.data.iter()) {
        *a += b;
    }
    Ok(())
}

/// The full E3 encoder: staff tensor `(1, 128, W)` flat → the decoder's
/// cross-attention context `[1 + 8·(W/16), 256]` (cls first — §2b).
///
/// # Errors
/// Shape violations, a missing tensor, or a kernel failure.
pub fn encode(w: &TromrEncoderW, pixels: &[f32], width: usize) -> FocrResult<Mat> {
    let feat = backbone(w, pixels, width)?;
    let x = tokens_from_feature(&feat); // [8·wp, 1024] row-major (r, c)
    let x = w.patch_proj.apply(&x)?; // [8·wp, 256]

    let (rows, wp) = (feat.h, feat.w);
    let seq = 1 + rows * wp;
    let mut tok = Mat::from_vec(seq, DIM, vec![0.0f32; seq * DIM]);
    // cls token + pos[0].
    for d in 0..DIM {
        tok.data[d] = w.cls_token[d] + w.pos_embed[d];
    }
    // Patch tokens + CROP-INDEXED positions: (r, c) → pos_embed[1 + r·80 + c].
    for r in 0..rows {
        for c in 0..wp {
            let t = 1 + r * wp + c;
            let pos = (1 + r * POS_COLS + c) * DIM;
            let src = (r * wp + c) * DIM;
            for d in 0..DIM {
                tok.data[t * DIM + d] = x.data[src + d] + w.pos_embed[pos + d];
            }
        }
    }

    for blk in &w.blocks {
        let h = nn::layer_norm(&tok, Some(&blk.ln1_w), Some(&blk.ln1_b), LN_EPS)?;
        let attn = self_attention(blk, &h)?;
        add_assign(&mut tok, &attn)?;
        let h2 = nn::layer_norm(&tok, Some(&blk.ln2_w), Some(&blk.ln2_b), LN_EPS)?;
        let mut m = blk.fc1.apply(&h2)?;
        nn::gelu(&mut m);
        let m = blk.fc2.apply(&m)?;
        add_assign(&mut tok, &m)?;
    }
    nn::layer_norm(&tok, Some(&w.final_ln_w), Some(&w.final_ln_b), LN_EPS)
}

// ───────────────────────── E4: the 4-head AR decoder ─────────────────────────

/// Decoder pre-branch LayerNorm eps (torch default — x-transformers passes
/// none; spec §4. NOTE: 1e-5, unlike the encoder's 1e-6).
const DEC_LN_EPS: f32 = 1e-5;
/// Attention inner width (8 heads × 64 — inner 512 ≠ dim 256, spec §4).
const DEC_INNER: usize = 512;
const DEC_HEADS: usize = 8;
const DEC_HEAD_DIM: usize = 64;
/// `max_seq_len` (config): the position table height AND the generate cap.
pub const MAX_SEQ: usize = 256;
/// Learned positions are scaled by `dim^-0.5 = 1/16` (x_transformers §4).
const POS_SCALE: f32 = 1.0 / 16.0;
/// Rhythm-stream generate seeds (config `bos_token`/`nonote_token`).
const SEED_RHYTHM: u32 = 1;
const SEED_NONOTE: u32 = 0;

/// One attention sublayer's weights: `to_{q,k,v} [512, 256]` and the
/// `on_attn` out projection `[512, 512]` — ALL bias-free (census §12/§16).
/// Stored as pre-transposed [`Linear`]s (bd-av64.10): the AR loop applies
/// these EVERY decode step, so building a projection per call re-transposed
/// the same weights once per token per sublayer.
struct AttnW {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
}

/// A pre-branch LayerNorm's affine params.
struct Ln {
    w: Vec<f32>,
    b: Vec<f32>,
}

/// One of the 4 decoder layers: ('a' self-attn, 'c' cross-attn, 'f' GEGLU
/// feed-forward), each pre-norm + residual (spec §4).
struct DecLayer {
    ln_a: Ln,
    self_attn: AttnW,
    ln_c: Ln,
    cross_attn: AttnW,
    ln_f: Ln,
    ff_proj: Linear,
    ff_out: Linear,
}

/// The hydrated TrOMR decoder (spec §12 names verbatim).
pub struct TromrDecoderW {
    rhythm_emb: Vec<f32>,
    pitch_emb: Vec<f32>,
    lift_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    layers: Vec<DecLayer>,
    final_ln: Ln,
    /// The four parallel per-stream heads (spec §4) — public: E7's assembly
    /// applies rhythm/pitch/lift per step, and the note head (inference-dead
    /// upstream, spec §5) stays exposed for the cert + future consistency
    /// diagnostics.
    pub head_rhythm: Linear,
    /// Pitch head `[71, 256]`.
    pub head_pitch: Linear,
    /// Lift head `[7, 256]`.
    pub head_lift: Linear,
    /// Note head `[2, 256]` (output-only; discarded at inference upstream).
    pub head_note: Linear,
}

impl TromrDecoderW {
    /// Hydrate from the artifact. The flat x-transformers layout indexes
    /// sublayers `layers.{i}` with `i%3` ⇒ 0='a', 1='c', 2='f' (spec §4);
    /// `layers.{i}.0.0` is the pre-branch norm, `layers.{i}.1` the branch.
    ///
    /// # Errors
    /// A missing tensor or a shape violation.
    pub fn build(weights: &Weights) -> FocrResult<Self> {
        let ln = |name: String| -> FocrResult<Ln> {
            Ok(Ln {
                w: weights.vec(&format!("{name}.weight"))?,
                b: weights.vec(&format!("{name}.bias"))?,
            })
        };
        let attn = |i: usize| -> FocrResult<AttnW> {
            let p = format!("decoder.net.attn_layers.layers.{i}.1.");
            let nb = |suffix: &str, out: usize, in_: usize| -> FocrResult<Linear> {
                Linear::from_row_major(
                    &weights.vec(&format!("{p}{suffix}.weight"))?,
                    Vec::new(),
                    out,
                    in_,
                )
            };
            Ok(AttnW {
                to_q: nb("to_q", DEC_INNER, DIM)?,
                to_k: nb("to_k", DEC_INNER, DIM)?,
                to_v: nb("to_v", DEC_INNER, DIM)?,
                to_out: nb("to_out.0", DEC_INNER, DEC_INNER)?,
            })
        };
        let head = |stream: &str, vocab: usize| -> FocrResult<Linear> {
            Linear::from_row_major(
                &weights.vec(&format!("decoder.net.to_logits_{stream}.weight"))?,
                weights.vec(&format!("decoder.net.to_logits_{stream}.bias"))?,
                vocab,
                DIM,
            )
        };
        let mut layers = Vec::with_capacity(4);
        for l in 0..4 {
            let base = 3 * l;
            layers.push(DecLayer {
                ln_a: ln(format!("decoder.net.attn_layers.layers.{base}.0.0"))?,
                self_attn: attn(base)?,
                ln_c: ln(format!("decoder.net.attn_layers.layers.{}.0.0", base + 1))?,
                cross_attn: attn(base + 1)?,
                ln_f: ln(format!("decoder.net.attn_layers.layers.{}.0.0", base + 2))?,
                ff_proj: Linear::from_row_major(
                    &weights.vec(&format!(
                        "decoder.net.attn_layers.layers.{}.1.net.0.proj.weight",
                        base + 2
                    ))?,
                    weights.vec(&format!(
                        "decoder.net.attn_layers.layers.{}.1.net.0.proj.bias",
                        base + 2
                    ))?,
                    2048,
                    DIM,
                )?,
                ff_out: Linear::from_row_major(
                    &weights.vec(&format!(
                        "decoder.net.attn_layers.layers.{}.1.net.3.weight",
                        base + 2
                    ))?,
                    weights.vec(&format!(
                        "decoder.net.attn_layers.layers.{}.1.net.3.bias",
                        base + 2
                    ))?,
                    DIM,
                    1024,
                )?,
            });
        }
        Ok(Self {
            rhythm_emb: weights.vec("decoder.net.rhythm_emb.emb.weight")?,
            pitch_emb: weights.vec("decoder.net.pitch_emb.emb.weight")?,
            lift_emb: weights.vec("decoder.net.lift_emb.emb.weight")?,
            pos_emb: weights.vec("decoder.net.pos_emb.emb.weight")?,
            layers,
            final_ln: ln("decoder.net.norm".into())?,
            head_rhythm: head("rhythm", 260)?,
            head_pitch: head("pitch", 71)?,
            head_lift: head("lift", 7)?,
            head_note: head("note", 2)?,
        })
    }
}

/// Bias-free `[out, in]` projection: `y = x @ w^T`.
fn proj_no_bias(x: &Mat, w: &Linear, out: usize) -> FocrResult<Mat> {
    if w.out != out {
        return Err(crate::FocrError::Other(anyhow::anyhow!(
            "tromr attention projection: out {} != expected {}",
            w.out,
            out
        )));
    }
    w.apply(x)
}

/// One `on_attn` attention branch (self or cross — spec §4): q from `x_q`,
/// k/v from `kv`, 8 heads × 64 at scale 1/8 (stable softmax inside the sdpa
/// kernel — OQ-T4), then `Linear(512→512, no bias)` + GLU (`a · σ(b)`).
fn glu_attention(a: &AttnW, x_q: &Mat, kv: &Mat, causal: bool) -> FocrResult<Mat> {
    let (seq_q, seq_k) = (x_q.rows, kv.rows);
    let q = proj_no_bias(x_q, &a.to_q, DEC_INNER)?;
    let k = proj_no_bias(kv, &a.to_k, DEC_INNER)?;
    let v = proj_no_bias(kv, &a.to_v, DEC_INNER)?;

    // Repack [seq, 512] → head-major [8, seq, 64].
    let pack = |m: &Mat, seq: usize| -> Vec<f32> {
        let span = seq * DEC_HEAD_DIM;
        let mut out = vec![0.0f32; DEC_HEADS * span];
        for s in 0..seq {
            let row = m.row(s);
            for h in 0..DEC_HEADS {
                let dst = h * span + s * DEC_HEAD_DIM;
                out[dst..dst + DEC_HEAD_DIM]
                    .copy_from_slice(&row[h * DEC_HEAD_DIM..(h + 1) * DEC_HEAD_DIM]);
            }
        }
        out
    };
    let (qf, kf, vf) = (pack(&q, seq_q), pack(&k, seq_k), pack(&v, seq_k));
    let scale = 1.0 / (DEC_HEAD_DIM as f32).sqrt();
    let ctx = nn::sdpa(
        &qf,
        &kf,
        &vf,
        DEC_HEADS,
        seq_q,
        seq_k,
        DEC_HEAD_DIM,
        DEC_HEAD_DIM,
        scale,
        causal,
    );
    // Merge back to [seq_q, 512].
    let span = seq_q * DEC_HEAD_DIM;
    let mut merged = vec![0.0f32; seq_q * DEC_INNER];
    for h in 0..DEC_HEADS {
        for s in 0..seq_q {
            let src = h * span + s * DEC_HEAD_DIM;
            let dst = s * DEC_INNER + h * DEC_HEAD_DIM;
            merged[dst..dst + DEC_HEAD_DIM].copy_from_slice(&ctx[src..src + DEC_HEAD_DIM]);
        }
    }
    // on_attn: Linear(512→512, no bias) then GLU split 2×256: `a · σ(b)`.
    let o = proj_no_bias(
        &Mat::from_vec(seq_q, DEC_INNER, merged),
        &a.to_out,
        DEC_INNER,
    )?;
    let mut out = vec![0.0f32; seq_q * DIM];
    for s in 0..seq_q {
        let row = o.row(s);
        for d in 0..DIM {
            out[s * DIM + d] = row[d] * (1.0 / (1.0 + (-row[DIM + d]).exp()));
        }
    }
    Ok(Mat::from_vec(seq_q, DIM, out))
}

/// The full-prefix decoder forward (upstream-faithful: NO KV cache — spec §4
/// notes upstream re-forwards the whole prefix; at 256×256 this is trivially
/// cheap, and a cache is a later bit-proven lever). Returns the final-normed
/// hidden `[t, 256]` for the (rhythm, pitch, lift) prefix over the encoder
/// `ctx` (`[1+8·wp, 256]`).
///
/// # Errors
/// Length mismatches between the three streams, an empty prefix, or a
/// prefix past [`MAX_SEQ`].
pub fn decoder_forward(
    w: &TromrDecoderW,
    ctx: &Mat,
    rhythm: &[u32],
    pitch: &[u32],
    lift: &[u32],
) -> FocrResult<Mat> {
    let t = rhythm.len();
    if t == 0 || t > MAX_SEQ || pitch.len() != t || lift.len() != t {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr decoder: stream lens (r {}, p {}, l {}) must be equal, 1..={MAX_SEQ}",
            rhythm.len(),
            pitch.len(),
            lift.len()
        )));
    }
    // x_t = rhythm_emb[r] + pitch_emb[p] + lift_emb[l] + pos[t]/16 (spec §4).
    let mut x = Mat::from_vec(t, DIM, vec![0.0f32; t * DIM]);
    for (i, ((&r, &p), &l)) in rhythm.iter().zip(pitch).zip(lift).enumerate() {
        let (r, p, l) = (r as usize, p as usize, l as usize);
        if r >= 260 || p >= 71 || l >= 7 {
            return Err(FocrError::Other(anyhow::anyhow!(
                "tromr decoder: id out of table at step {i} (r {r}, p {p}, l {l})"
            )));
        }
        for d in 0..DIM {
            x.data[i * DIM + d] = w.rhythm_emb[r * DIM + d]
                + w.pitch_emb[p * DIM + d]
                + w.lift_emb[l * DIM + d]
                + w.pos_emb[i * DIM + d] * POS_SCALE;
        }
    }
    for layer in &w.layers {
        let h = nn::layer_norm(&x, Some(&layer.ln_a.w), Some(&layer.ln_a.b), DEC_LN_EPS)?;
        let a = glu_attention(&layer.self_attn, &h, &h, true)?;
        add_assign(&mut x, &a)?;
        let h = nn::layer_norm(&x, Some(&layer.ln_c.w), Some(&layer.ln_c.b), DEC_LN_EPS)?;
        let c = glu_attention(&layer.cross_attn, &h, ctx, false)?;
        add_assign(&mut x, &c)?;
        let h = nn::layer_norm(&x, Some(&layer.ln_f.w), Some(&layer.ln_f.b), DEC_LN_EPS)?;
        // GEGLU: proj → chunk (x, gate) 2×1024 → x · GELU(gate) → out. The
        // gate halves are gathered into one Mat so the exact-erf GELU runs
        // vectorized, then multiplied back against the value halves.
        let pr = layer.ff_proj.apply(&h)?;
        let mut gate = Mat::from_vec(t, 1024, vec![0.0f32; t * 1024]);
        for s in 0..t {
            gate.data[s * 1024..(s + 1) * 1024].copy_from_slice(&pr.row(s)[1024..2048]);
        }
        nn::gelu(&mut gate);
        let mut gated = Mat::from_vec(t, 1024, vec![0.0f32; t * 1024]);
        for s in 0..t {
            let row = pr.row(s);
            for (g, (&x_val, &g_val)) in gated.data[s * 1024..(s + 1) * 1024].iter_mut().zip(
                row[..1024]
                    .iter()
                    .zip(gate.data[s * 1024..(s + 1) * 1024].iter()),
            ) {
                *g = x_val * g_val;
            }
        }
        let f = layer.ff_out.apply(&gated)?;
        add_assign(&mut x, &f)?;
    }
    nn::layer_norm(&x, Some(&w.final_ln.w), Some(&w.final_ln.b), DEC_LN_EPS)
}

/// The three generated id streams (seeds excluded), positionally rhythm /
/// pitch / lift end-to-end (the §4 naming-swap trap cancels; never "fix" it).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicStreams {
    /// Rhythm ids (the stream that carries `[EOS]`; includes it when emitted).
    pub rhythm: Vec<u32>,
    /// Pitch ids.
    pub pitch: Vec<u32>,
    /// Lift (accidental) ids.
    pub lift: Vec<u32>,
}

/// Stable schema for same-forward TrOMR candidate evidence.
pub const TROMR_CANDIDATE_LATTICE_SCHEMA_VERSION: u32 = 1;
/// Number of ranked alternatives retained per decoder head and position.
pub const TROMR_CANDIDATE_TOP_N: usize = 5;
/// Hard bound for the canonical binary form of one generated staff row.
pub const TROMR_MAX_CANDIDATE_LATTICE_CANONICAL_BYTES: usize = 256 * 1024;
/// Domain identifier included in every candidate lattice.
pub const TROMR_CANDIDATE_LATTICE_CONTRACT_ID: &str =
    "franken_ocr.tromr.same_forward_candidate_lattice.v1";
/// Clock-free, host-path-free canonical encoding carried by the lattice.
pub const TROMR_CANDIDATE_LATTICE_CANONICAL_ENCODING: &str =
    "franken_ocr.tromr.candidate_lattice.le.v1";
const TROMR_CANDIDATE_PREFIX_HASH_DOMAIN: &[u8] = b"franken_ocr.tromr.candidate_prefix.v1\0";

/// One of the three live TrOMR decoder heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TromrCandidateHeadV1 {
    /// Rhythm head, including the stream's EOS token.
    Rhythm,
    /// Pitch head.
    Pitch,
    /// Lift/accidental head.
    Lift,
}

impl TromrCandidateHeadV1 {
    const fn canonical_tag(self) -> u8 {
        match self {
            Self::Rhythm => 0,
            Self::Pitch => 1,
            Self::Lift => 2,
        }
    }

    const fn vocabulary_size(self) -> usize {
        match self {
            Self::Rhythm => 260,
            Self::Pitch => 71,
            Self::Lift => 7,
        }
    }
}

/// A retained candidate and its raw, uncalibrated decoder model score.
///
/// `model_score_f32_bits` is the exact IEEE-754 bit pattern emitted by the
/// corresponding head. It is not a probability or correctness confidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TromrCandidateModelScoreV1 {
    /// Token id in the head vocabulary.
    pub token_id: u32,
    /// Exact rank under descending model score with token-id tie breaking.
    pub rank_one_based: u32,
    /// Exact bits of the uncalibrated f32 model score.
    pub model_score_f32_bits: u32,
}

impl TromrCandidateModelScoreV1 {
    /// Decode the exact uncalibrated model score.
    #[must_use]
    pub const fn model_score(&self) -> f32 {
        f32::from_bits(self.model_score_f32_bits)
    }
}

/// Same-forward evidence for one decoder head at one emitted position.
///
/// The selected id and retained candidates come from the same head logits
/// used for generation. Scores are raw model scores, not confidence values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TromrCandidateHeadEvidenceV1 {
    /// Decoder head represented by this record.
    pub head: TromrCandidateHeadV1,
    /// Full head vocabulary size used to interpret ranks and truncation.
    pub vocabulary_size: u32,
    /// Id actually emitted by the configured deterministic selection rule.
    pub chosen_token_id: u32,
    /// Exact full-vocabulary rank of the chosen id.
    pub chosen_rank_one_based: u32,
    /// Exact bits of the chosen token's uncalibrated f32 model score.
    pub chosen_model_score_f32_bits: u32,
    /// Exact bits of `chosen score - best non-chosen score`.
    pub chosen_minus_best_alternative_model_score_f32_bits: u32,
    /// Best candidates in exact descending score/token-id order.
    pub retained_top_candidates: Vec<TromrCandidateModelScoreV1>,
    /// Full-vocabulary candidates omitted by the bounded top-N record.
    pub truncated_candidate_count: u32,
}

impl TromrCandidateHeadEvidenceV1 {
    /// Decode the selected token's exact uncalibrated model score.
    #[must_use]
    pub const fn chosen_model_score(&self) -> f32 {
        f32::from_bits(self.chosen_model_score_f32_bits)
    }

    /// Decode `chosen score - best non-chosen score`.
    #[must_use]
    pub const fn chosen_minus_best_alternative_model_score(&self) -> f32 {
        f32::from_bits(self.chosen_minus_best_alternative_model_score_f32_bits)
    }
}

/// Same-forward candidate evidence for one emitted token triple.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TromrCandidatePositionV1 {
    /// Zero-based generated position (seeds excluded).
    pub position_zero_based: u32,
    /// Total decoder prefix length, including the seed triple.
    pub prefix_length: u32,
    /// Start of the exact window supplied to the decoder.
    pub prefix_window_start: u32,
    /// Domain-separated digest of the exact decoder-input token triples.
    pub prefix_sha256: [u8; 32],
    /// Evidence in the fixed rhythm, pitch, lift order.
    pub heads: [TromrCandidateHeadEvidenceV1; 3],
    /// Whether the rhythm token emitted at this position is EOS.
    pub rhythm_emitted_eos: bool,
}

/// Bounded, canonical evidence from the exact forward passes used to decode a
/// staff row.
///
/// This first provider contract is deliberately tokenizer-agnostic: it binds
/// exact ids, ranks, model-score bits, prefixes, and chosen streams. Immutable
/// model/tokenizer/options and token-text binding belongs to the parent
/// recognition receipt. Nothing in this type is a calibrated confidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TromrCandidateLatticeV1 {
    /// Must equal [`TROMR_CANDIDATE_LATTICE_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Must equal [`TROMR_CANDIDATE_LATTICE_CONTRACT_ID`].
    pub contract_id: String,
    /// Must equal [`TROMR_CANDIDATE_LATTICE_CANONICAL_ENCODING`].
    pub canonical_encoding: String,
    /// Must equal [`TROMR_CANDIDATE_TOP_N`].
    pub top_n: u32,
    /// Exact generated streams duplicated here to make the evidence closed.
    pub chosen_streams: MusicStreams,
    /// One record for every emitted triple, including the EOS position.
    pub positions: Vec<TromrCandidatePositionV1>,
}

/// Generation result carrying both the existing streams and their exact
/// same-forward candidate lattice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TromrGenerationWithCandidateLatticeV1 {
    /// Existing generated streams, byte-for-byte selection compatible.
    pub streams: MusicStreams,
    /// Bounded evidence captured from those exact decoder forwards.
    pub candidate_lattice: TromrCandidateLatticeV1,
}

/// Stable schema for binding one candidate lattice to its exact row-local
/// TrOMR forward input.
pub const TROMR_FORWARD_CANDIDATE_LATTICE_SCHEMA_VERSION: u32 = 1;

/// One independently seeded decoder lattice, indexed against the row's ordered
/// [`TromrForwardInputV1`] vector.
///
/// Split recognition creates one of these per segment. Its candidate positions
/// must never be flattened across a segment boundary because every segment
/// restarts from TrOMR's seed triple and has its own prefix hash chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TromrForwardCandidateLatticeV1 {
    /// Must equal [`TROMR_FORWARD_CANDIDATE_LATTICE_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Zero-based index into the owning row's ordered forward-input vector.
    pub forward_input_index: u32,
    /// Exact same-forward lattice for only this forward input.
    pub candidate_lattice: TromrCandidateLatticeV1,
}

impl TromrForwardCandidateLatticeV1 {
    /// Construct and validate an indexed per-forward lattice.
    ///
    /// # Errors
    /// A non-representable index or invalid candidate lattice.
    pub fn new(
        forward_input_index: usize,
        candidate_lattice: TromrCandidateLatticeV1,
    ) -> FocrResult<Self> {
        let forward_input_index = u32::try_from(forward_input_index).map_err(|error| {
            candidate_lattice_mismatch(format!(
                "forward-input index is not representable as u32: {error}"
            ))
        })?;
        let indexed = Self {
            schema_version: TROMR_FORWARD_CANDIDATE_LATTICE_SCHEMA_VERSION,
            forward_input_index,
            candidate_lattice,
        };
        indexed.validate()?;
        Ok(indexed)
    }

    /// Validate this wrapper without imposing an enclosing vector position.
    ///
    /// # Errors
    /// An unsupported wrapper schema or invalid candidate lattice.
    pub fn validate(&self) -> FocrResult<()> {
        if self.schema_version != TROMR_FORWARD_CANDIDATE_LATTICE_SCHEMA_VERSION {
            return Err(candidate_lattice_mismatch(format!(
                "forward wrapper schema {} is unsupported; expected {}",
                self.schema_version, TROMR_FORWARD_CANDIDATE_LATTICE_SCHEMA_VERSION
            )));
        }
        self.candidate_lattice.validate()
    }

    fn validate_at_ordered_index(&self, expected_index: usize) -> FocrResult<()> {
        self.validate()?;
        let expected_index = u32::try_from(expected_index).map_err(|error| {
            candidate_lattice_mismatch(format!(
                "expected forward-input index is not representable as u32: {error}"
            ))
        })?;
        if self.forward_input_index != expected_index {
            return Err(candidate_lattice_mismatch(format!(
                "forward wrapper index {} is out of order; expected {expected_index}",
                self.forward_input_index
            )));
        }
        Ok(())
    }
}

impl TromrCandidateLatticeV1 {
    /// Validate all structural, ordering, prefix, EOS, and size invariants.
    ///
    /// A chosen rank beyond the retained top-N is faithfully recorded but
    /// cannot be independently recomputed from a bounded record. The later
    /// immutable parent receipt binds that fact to the exact model execution.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] when any contract invariant is violated.
    pub fn validate(&self) -> FocrResult<()> {
        validate_candidate_lattice(self)?;
        let encoded_len = encode_candidate_lattice_unchecked(self).len();
        if encoded_len > TROMR_MAX_CANDIDATE_LATTICE_CANONICAL_BYTES {
            return Err(candidate_lattice_mismatch(format!(
                "canonical evidence is {encoded_len} bytes; maximum is {TROMR_MAX_CANDIDATE_LATTICE_CANONICAL_BYTES}"
            )));
        }
        Ok(())
    }

    /// Encode the validated lattice without clocks, paths, maps, or float
    /// text formatting.
    ///
    /// # Errors
    /// The errors from [`Self::validate`].
    pub fn canonical_bytes(&self) -> FocrResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_candidate_lattice_unchecked(self))
    }

    /// SHA-256 of [`Self::canonical_bytes`].
    ///
    /// # Errors
    /// The errors from [`Self::canonical_bytes`].
    pub fn canonical_sha256(&self) -> FocrResult<[u8; 32]> {
        Ok(Sha256::digest(self.canonical_bytes()?).into())
    }
}

/// Stable schema carried by [`TromrRecognitionOptionsV1`].
pub const TROMR_RECOGNITION_OPTIONS_SCHEMA_VERSION: u32 = 1;

/// Mechanical token-selection mode for TrOMR's four-head decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TromrDecodeModeV1 {
    /// Deterministic per-head maximum-logit selection.
    Argmax,
    /// Upstream top-k(threshold 0.9), temperature 0.2 sampling with an
    /// explicit, platform-stable PCG32 seed.
    SeededTopKTemperature,
}

/// Mechanical policy for staff bands wider than the model position table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TromrSplitPolicyV1 {
    /// Refuse an over-budget row instead of approximating it in segments.
    Disabled,
    /// Experimental barline-aligned segment recognition and deterministic
    /// semantic splicing.
    ExperimentalBarlineSegments,
}

/// Fixed staff-image resampler implemented by the native TrOMR path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TromrStaffResamplerV1 {
    /// OpenCV-compatible half-pixel bilinear resize over u8 grayscale pixels.
    Cv2LinearU8V1,
}

/// Versioned, explicit recognition controls for every TrOMR path.
///
/// These are inference mechanics, not aesthetic choices. Core recognition
/// never reads process environment variables; the standalone CLI may parse
/// legacy configuration once and construct this value at its boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TromrRecognitionOptionsV1 {
    /// Must equal [`TROMR_RECOGNITION_OPTIONS_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Token-selection arithmetic.
    pub decode_mode: TromrDecodeModeV1,
    /// Required exactly for [`TromrDecodeModeV1::SeededTopKTemperature`].
    pub seed: Option<u64>,
    /// Over-budget staff handling.
    pub split_policy: TromrSplitPolicyV1,
    /// Fixed source-to-model resize implementation.
    pub staff_resampler: TromrStaffResamplerV1,
}

impl TromrRecognitionOptionsV1 {
    /// Explicit deterministic production default.
    #[must_use]
    pub const fn deterministic() -> Self {
        Self {
            schema_version: TROMR_RECOGNITION_OPTIONS_SCHEMA_VERSION,
            decode_mode: TromrDecodeModeV1::Argmax,
            seed: None,
            split_policy: TromrSplitPolicyV1::Disabled,
            staff_resampler: TromrStaffResamplerV1::Cv2LinearU8V1,
        }
    }

    /// Validate the version and cross-field seed contract.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] for an unsupported schema version, or
    /// [`FocrError::Usage`] when the seed is absent/present for the wrong mode.
    pub fn validate(self) -> FocrResult<Self> {
        if self.schema_version != TROMR_RECOGNITION_OPTIONS_SCHEMA_VERSION {
            return Err(FocrError::FormatMismatch(format!(
                "tromr recognition options schema {} is unsupported; expected {}",
                self.schema_version, TROMR_RECOGNITION_OPTIONS_SCHEMA_VERSION
            )));
        }
        match (self.decode_mode, self.seed) {
            (TromrDecodeModeV1::Argmax, Some(_)) => Err(FocrError::Usage(
                "tromr argmax mode must not carry a sampling seed".into(),
            )),
            (TromrDecodeModeV1::SeededTopKTemperature, None) => Err(FocrError::Usage(
                "tromr seeded_top_k_temperature mode requires an explicit seed".into(),
            )),
            _ => Ok(self),
        }
    }

    /// Parse the stable JSON value and validate all cross-field contracts.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] for malformed JSON, unknown fields, modes,
    /// or resamplers; otherwise the errors from [`Self::validate`].
    pub fn from_json(json: &str) -> FocrResult<Self> {
        serde_json::from_str::<Self>(json)
            .map_err(|error| {
                FocrError::FormatMismatch(format!("tromr recognition options JSON: {error}"))
            })?
            .validate()
    }

    /// Canonical clock-free JSON used by receipts and replay keys.
    ///
    /// # Errors
    /// The validation errors from [`Self::validate`], or a serialization error.
    pub fn canonical_json(self) -> FocrResult<String> {
        let normalized = self.validate()?;
        serde_json::to_string(&normalized).map_err(|error| {
            FocrError::Other(anyhow::anyhow!(
                "serialize tromr recognition options: {error}"
            ))
        })
    }

    /// SHA-256 identity of [`Self::canonical_json`], suitable for cache and
    /// replay keys without host paths or wall-clock data.
    ///
    /// # Errors
    /// The errors from [`Self::canonical_json`].
    pub fn replay_identity(self) -> FocrResult<String> {
        let digest: [u8; 32] = Sha256::digest(self.canonical_json()?.as_bytes()).into();
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut identity = String::with_capacity(64);
        for byte in digest {
            identity.push(HEX[usize::from(byte >> 4)] as char);
            identity.push(HEX[usize::from(byte & 0x0f)] as char);
        }
        Ok(identity)
    }

    fn decode_pick(self) -> FocrResult<DecodePick> {
        let normalized = self.validate()?;
        match normalized.decode_mode {
            TromrDecodeModeV1::Argmax => Ok(DecodePick::Argmax),
            TromrDecodeModeV1::SeededTopKTemperature => normalized
                .seed
                .map(|seed| DecodePick::SeededSample { seed })
                .ok_or_else(|| {
                    FocrError::Usage(
                        "tromr seeded_top_k_temperature mode requires an explicit seed".into(),
                    )
                }),
        }
    }
}

impl Default for TromrRecognitionOptionsV1 {
    fn default() -> Self {
        Self::deterministic()
    }
}

/// The per-step token pick. Argmax is the explicit deterministic default.
/// Seeded sampling preserves upstream top-k(thres 0.9), T=0.2 arithmetic when
/// a caller deliberately requests it; the caller-provided seed makes the
/// resulting stream replayable on every platform.
#[derive(Clone, Copy, Debug)]
pub enum DecodePick {
    /// Per-head argmax (the L4 oracle-parity mode; degenerate on real staves).
    Argmax,
    /// Upstream top-k(0.9)/T=0.2 multinomial from a pinned PCG32 seed.
    SeededSample {
        /// The caller-declared PCG32 stream seed.
        seed: u64,
    },
}

/// Minimal PCG32 (Melissa O'Neill's PCG-XSH-RR) — a tiny, dependency-free,
/// platform-stable PRNG for the seeded decode. NOT cryptographic.
struct Pcg32 {
    state: u64,
}

impl Pcg32 {
    fn new(seed: u64) -> Self {
        let mut s = Self {
            state: seed.wrapping_add(0x853c_49e6_748f_ea9b),
        };
        s.next_u32();
        s
    }
    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }
    /// U[0, 1) with 32-bit resolution.
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// Upstream `top_k(thres=0.9)` + `softmax(logits/T)` + multinomial, seeded:
/// keep the top `ceil(0.1·V)` logits (rhythm 26, pitch 8, lift 1 — lift is
/// de-facto argmax), temperature 0.2, CDF-walk the kept mass.
fn upstream_top_k_indices(logits: &[f32]) -> Vec<usize> {
    // With the fixed upstream threshold 0.9, ceil((1 - threshold) * V) is
    // exactly ceil(V / 10). Computing it through f32 makes the rhythm case
    // ceil(0.100000024 * 260) == 27 instead of the upstream result 26.
    let k = logits.len().div_ceil(10).max(1);
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    indices.truncate(k);
    indices
}

fn sample_top_k(logits: &[f32], rng: &mut Pcg32) -> u32 {
    const TEMPERATURE: f32 = 0.2;
    let idx = upstream_top_k_indices(logits);
    let k = idx.len();
    // softmax(logits/T) over the kept set (max-subtract stable).
    let m = logits[idx[0]] / TEMPERATURE;
    let weights: Vec<f32> = idx
        .iter()
        .map(|&i| (logits[i] / TEMPERATURE - m).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    let mut u = rng.next_f32() * total;
    for (w, &i) in weights.iter().zip(&idx) {
        if u < *w {
            return i as u32;
        }
        u -= w;
    }
    idx[k - 1] as u32
}

fn candidate_lattice_mismatch(message: impl Into<String>) -> FocrError {
    FocrError::FormatMismatch(format!("tromr candidate lattice: {}", message.into()))
}

/// Rank raw model scores exactly as the existing diagnostic oracle: descending
/// `f32::total_cmp`, then ascending token id. Rejecting non-finite head output
/// prevents an apparently precise evidence record from laundering invalid
/// model arithmetic.
fn ranked_candidate_ids(logits: &[f32]) -> FocrResult<Vec<usize>> {
    if logits.is_empty() {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr candidate capture: decoder head emitted no model scores"
        )));
    }
    if let Some((token_id, score)) = logits
        .iter()
        .copied()
        .enumerate()
        .find(|(_, score)| !score.is_finite())
    {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr candidate capture: decoder head emitted non-finite model score {score:?} for token {token_id}"
        )));
    }
    let mut ranked: Vec<usize> = (0..logits.len()).collect();
    ranked.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));
    Ok(ranked)
}

fn legacy_argmax_id(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold(
            (0usize, f32::NEG_INFINITY),
            |(best_id, best), (id, &score)| {
                if score > best {
                    (id, score)
                } else {
                    (best_id, best)
                }
            },
        )
        .0 as u32
}

fn capture_candidate_head_evidence(
    head: TromrCandidateHeadV1,
    logits: &[f32],
    chosen_token_id: u32,
    ranked: &[usize],
) -> FocrResult<TromrCandidateHeadEvidenceV1> {
    let vocabulary_size = head.vocabulary_size();
    if logits.len() != vocabulary_size || ranked.len() != vocabulary_size {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr candidate capture: {:?} head width/ranking was {}/{}, expected {vocabulary_size}",
            head,
            logits.len(),
            ranked.len()
        )));
    }
    let chosen = usize::try_from(chosen_token_id).map_err(|error| {
        FocrError::Other(anyhow::anyhow!(
            "tromr candidate capture: chosen token id conversion failed: {error}"
        ))
    })?;
    if chosen >= vocabulary_size {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr candidate capture: chosen {:?} token {chosen} is outside vocabulary {vocabulary_size}",
            head
        )));
    }
    let chosen_rank = ranked
        .iter()
        .position(|&token_id| token_id == chosen)
        .ok_or_else(|| {
            FocrError::Other(anyhow::anyhow!(
                "tromr candidate capture: chosen {:?} token {chosen} was absent from ranking",
                head
            ))
        })?;
    let best_alternative = ranked
        .iter()
        .copied()
        .find(|&token_id| token_id != chosen)
        .ok_or_else(|| {
            FocrError::Other(anyhow::anyhow!(
                "tromr candidate capture: {:?} vocabulary has no alternative token",
                head
            ))
        })?;
    let chosen_score = logits[chosen];
    let margin = chosen_score - logits[best_alternative];
    if !margin.is_finite() {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr candidate capture: {:?} chosen-minus-alternative model score is non-finite",
            head
        )));
    }
    let retained_count = TROMR_CANDIDATE_TOP_N.min(vocabulary_size);
    let retained_top_candidates = ranked
        .iter()
        .take(retained_count)
        .enumerate()
        .map(|(rank, &token_id)| TromrCandidateModelScoreV1 {
            token_id: token_id as u32,
            rank_one_based: (rank + 1) as u32,
            model_score_f32_bits: logits[token_id].to_bits(),
        })
        .collect();
    Ok(TromrCandidateHeadEvidenceV1 {
        head,
        vocabulary_size: vocabulary_size as u32,
        chosen_token_id,
        chosen_rank_one_based: (chosen_rank + 1) as u32,
        chosen_model_score_f32_bits: chosen_score.to_bits(),
        chosen_minus_best_alternative_model_score_f32_bits: margin.to_bits(),
        retained_top_candidates,
        truncated_candidate_count: (vocabulary_size - retained_count) as u32,
    })
}

fn select_and_capture_candidate_head(
    head: TromrCandidateHeadV1,
    logits: &[f32],
    rng: &mut Option<Pcg32>,
) -> FocrResult<(u32, TromrCandidateHeadEvidenceV1)> {
    let ranked = ranked_candidate_ids(logits)?;
    let chosen_token_id = match rng {
        Some(rng) => sample_top_k(logits, rng),
        None => legacy_argmax_id(logits),
    };
    let evidence = capture_candidate_head_evidence(head, logits, chosen_token_id, &ranked)?;
    Ok((chosen_token_id, evidence))
}

fn candidate_prefix_sha256(
    rhythm: &[u32],
    pitch: &[u32],
    lift: &[u32],
    window_start: usize,
) -> FocrResult<[u8; 32]> {
    if rhythm.is_empty()
        || rhythm.len() != pitch.len()
        || rhythm.len() != lift.len()
        || window_start > rhythm.len()
    {
        return Err(candidate_lattice_mismatch(format!(
            "invalid prefix streams r/p/l={}/{}/{} start={window_start}",
            rhythm.len(),
            pitch.len(),
            lift.len()
        )));
    }
    let mut digest = Sha256::new();
    digest.update(TROMR_CANDIDATE_PREFIX_HASH_DOMAIN);
    digest.update((rhythm.len() as u32).to_le_bytes());
    digest.update((window_start as u32).to_le_bytes());
    digest.update(((rhythm.len() - window_start) as u32).to_le_bytes());
    for ((rhythm_id, pitch_id), lift_id) in rhythm[window_start..]
        .iter()
        .zip(&pitch[window_start..])
        .zip(&lift[window_start..])
    {
        digest.update(rhythm_id.to_le_bytes());
        digest.update(pitch_id.to_le_bytes());
        digest.update(lift_id.to_le_bytes());
    }
    Ok(digest.finalize().into())
}

fn push_candidate_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn push_candidate_stream(out: &mut Vec<u8>, stream: &[u32]) {
    out.extend_from_slice(&(stream.len() as u32).to_le_bytes());
    for token_id in stream {
        out.extend_from_slice(&token_id.to_le_bytes());
    }
}

fn encode_candidate_lattice_unchecked(lattice: &TromrCandidateLatticeV1) -> Vec<u8> {
    let mut out = Vec::with_capacity(128 + lattice.positions.len() * 256);
    push_candidate_bytes(&mut out, lattice.contract_id.as_bytes());
    out.extend_from_slice(&lattice.schema_version.to_le_bytes());
    push_candidate_bytes(&mut out, lattice.canonical_encoding.as_bytes());
    out.extend_from_slice(&lattice.top_n.to_le_bytes());
    push_candidate_stream(&mut out, &lattice.chosen_streams.rhythm);
    push_candidate_stream(&mut out, &lattice.chosen_streams.pitch);
    push_candidate_stream(&mut out, &lattice.chosen_streams.lift);
    out.extend_from_slice(&(lattice.positions.len() as u32).to_le_bytes());
    for position in &lattice.positions {
        out.extend_from_slice(&position.position_zero_based.to_le_bytes());
        out.extend_from_slice(&position.prefix_length.to_le_bytes());
        out.extend_from_slice(&position.prefix_window_start.to_le_bytes());
        out.extend_from_slice(&position.prefix_sha256);
        out.push(u8::from(position.rhythm_emitted_eos));
        for head in &position.heads {
            out.push(head.head.canonical_tag());
            out.extend_from_slice(&head.vocabulary_size.to_le_bytes());
            out.extend_from_slice(&head.chosen_token_id.to_le_bytes());
            out.extend_from_slice(&head.chosen_rank_one_based.to_le_bytes());
            out.extend_from_slice(&head.chosen_model_score_f32_bits.to_le_bytes());
            out.extend_from_slice(
                &head
                    .chosen_minus_best_alternative_model_score_f32_bits
                    .to_le_bytes(),
            );
            out.extend_from_slice(&head.truncated_candidate_count.to_le_bytes());
            out.extend_from_slice(&(head.retained_top_candidates.len() as u32).to_le_bytes());
            for candidate in &head.retained_top_candidates {
                out.extend_from_slice(&candidate.token_id.to_le_bytes());
                out.extend_from_slice(&candidate.rank_one_based.to_le_bytes());
                out.extend_from_slice(&candidate.model_score_f32_bits.to_le_bytes());
            }
        }
    }
    out
}

fn candidate_precedes(score_a: f32, id_a: u32, score_b: f32, id_b: u32) -> bool {
    match score_a.total_cmp(&score_b) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Equal => id_a < id_b,
        std::cmp::Ordering::Less => false,
    }
}

fn validate_candidate_head(
    evidence: &TromrCandidateHeadEvidenceV1,
    expected_head: TromrCandidateHeadV1,
    expected_chosen_token_id: u32,
) -> FocrResult<()> {
    if evidence.head != expected_head {
        return Err(candidate_lattice_mismatch(format!(
            "head order contains {:?}; expected {:?}",
            evidence.head, expected_head
        )));
    }
    let vocabulary_size = expected_head.vocabulary_size();
    if evidence.vocabulary_size != vocabulary_size as u32 {
        return Err(candidate_lattice_mismatch(format!(
            "{:?} vocabulary is {}; expected {vocabulary_size}",
            expected_head, evidence.vocabulary_size
        )));
    }
    if evidence.chosen_token_id != expected_chosen_token_id
        || usize::try_from(evidence.chosen_token_id).map_or(true, |id| id >= vocabulary_size)
    {
        return Err(candidate_lattice_mismatch(format!(
            "{:?} chosen token {} does not match in-range stream token {expected_chosen_token_id}",
            expected_head, evidence.chosen_token_id
        )));
    }
    let chosen_rank = usize::try_from(evidence.chosen_rank_one_based).map_err(|error| {
        candidate_lattice_mismatch(format!("chosen rank conversion failed: {error}"))
    })?;
    if chosen_rank == 0 || chosen_rank > vocabulary_size {
        return Err(candidate_lattice_mismatch(format!(
            "{:?} chosen rank {chosen_rank} is outside 1..={vocabulary_size}",
            expected_head
        )));
    }
    let chosen_score = evidence.chosen_model_score();
    let margin = evidence.chosen_minus_best_alternative_model_score();
    if !chosen_score.is_finite() || !margin.is_finite() {
        return Err(candidate_lattice_mismatch(format!(
            "{:?} chosen score or margin is non-finite",
            expected_head
        )));
    }
    let retained_count = TROMR_CANDIDATE_TOP_N.min(vocabulary_size);
    if evidence.retained_top_candidates.len() != retained_count
        || evidence.truncated_candidate_count != (vocabulary_size - retained_count) as u32
    {
        return Err(candidate_lattice_mismatch(format!(
            "{:?} retained/truncated counts are {}/{}; expected {retained_count}/{}",
            expected_head,
            evidence.retained_top_candidates.len(),
            evidence.truncated_candidate_count,
            vocabulary_size - retained_count
        )));
    }
    for (index, candidate) in evidence.retained_top_candidates.iter().enumerate() {
        if candidate.rank_one_based != (index + 1) as u32
            || usize::try_from(candidate.token_id).map_or(true, |id| id >= vocabulary_size)
            || !candidate.model_score().is_finite()
        {
            return Err(candidate_lattice_mismatch(format!(
                "{:?} retained candidate at index {index} has invalid id/rank/score",
                expected_head
            )));
        }
        if evidence.retained_top_candidates[..index]
            .iter()
            .any(|earlier| earlier.token_id == candidate.token_id)
        {
            return Err(candidate_lattice_mismatch(format!(
                "{:?} retained token {} is duplicated",
                expected_head, candidate.token_id
            )));
        }
        if let Some(previous) = index
            .checked_sub(1)
            .and_then(|previous| evidence.retained_top_candidates.get(previous))
            && !candidate_precedes(
                previous.model_score(),
                previous.token_id,
                candidate.model_score(),
                candidate.token_id,
            )
        {
            return Err(candidate_lattice_mismatch(format!(
                "{:?} retained candidates are not in canonical score/id order",
                expected_head
            )));
        }
    }
    if chosen_rank <= retained_count {
        let retained = &evidence.retained_top_candidates[chosen_rank - 1];
        if retained.token_id != evidence.chosen_token_id
            || retained.model_score_f32_bits != evidence.chosen_model_score_f32_bits
        {
            return Err(candidate_lattice_mismatch(format!(
                "{:?} retained chosen token/score does not match rank {chosen_rank}",
                expected_head
            )));
        }
    } else {
        if evidence
            .retained_top_candidates
            .iter()
            .any(|candidate| candidate.token_id == evidence.chosen_token_id)
        {
            return Err(candidate_lattice_mismatch(format!(
                "{:?} chosen rank exceeds top-N but chosen id is retained",
                expected_head
            )));
        }
        let last = evidence
            .retained_top_candidates
            .last()
            .expect("TrOMR vocabularies exceed candidate top-N");
        if candidate_precedes(
            chosen_score,
            evidence.chosen_token_id,
            last.model_score(),
            last.token_id,
        ) {
            return Err(candidate_lattice_mismatch(format!(
                "{:?} non-retained chosen score outranks a retained candidate",
                expected_head
            )));
        }
    }
    let best_alternative = evidence
        .retained_top_candidates
        .iter()
        .find(|candidate| candidate.token_id != evidence.chosen_token_id)
        .expect("top-N contains an alternative token");
    let expected_margin = chosen_score - best_alternative.model_score();
    if !expected_margin.is_finite()
        || expected_margin.to_bits() != evidence.chosen_minus_best_alternative_model_score_f32_bits
    {
        return Err(candidate_lattice_mismatch(format!(
            "{:?} chosen-minus-best-alternative margin does not recompute",
            expected_head
        )));
    }
    Ok(())
}

fn validate_candidate_lattice(lattice: &TromrCandidateLatticeV1) -> FocrResult<()> {
    if lattice.schema_version != TROMR_CANDIDATE_LATTICE_SCHEMA_VERSION
        || lattice.contract_id != TROMR_CANDIDATE_LATTICE_CONTRACT_ID
        || lattice.canonical_encoding != TROMR_CANDIDATE_LATTICE_CANONICAL_ENCODING
        || lattice.top_n != TROMR_CANDIDATE_TOP_N as u32
    {
        return Err(candidate_lattice_mismatch(
            "unsupported schema, contract id, canonical encoding, or top-N",
        ));
    }
    let stream_len = lattice.chosen_streams.rhythm.len();
    if stream_len == 0
        || stream_len > MAX_SEQ
        || lattice.chosen_streams.pitch.len() != stream_len
        || lattice.chosen_streams.lift.len() != stream_len
        || lattice.positions.len() != stream_len
    {
        return Err(candidate_lattice_mismatch(format!(
            "chosen stream/position lengths r/p/l/evidence={}/{}/{}/{} must be equal and 1..={MAX_SEQ}",
            stream_len,
            lattice.chosen_streams.pitch.len(),
            lattice.chosen_streams.lift.len(),
            lattice.positions.len()
        )));
    }

    let mut prefix_rhythm = vec![SEED_RHYTHM];
    let mut prefix_pitch = vec![SEED_NONOTE];
    let mut prefix_lift = vec![SEED_NONOTE];
    let mut saw_eos = false;
    for (index, position) in lattice.positions.iter().enumerate() {
        let start = prefix_rhythm.len().saturating_sub(MAX_SEQ);
        if position.position_zero_based != index as u32
            || position.prefix_length != prefix_rhythm.len() as u32
            || position.prefix_window_start != start as u32
            || position.prefix_sha256
                != candidate_prefix_sha256(&prefix_rhythm, &prefix_pitch, &prefix_lift, start)?
        {
            return Err(candidate_lattice_mismatch(format!(
                "position {index} index/prefix identity does not recompute"
            )));
        }
        let rhythm_id = lattice.chosen_streams.rhythm[index];
        let pitch_id = lattice.chosen_streams.pitch[index];
        let lift_id = lattice.chosen_streams.lift[index];
        validate_candidate_head(&position.heads[0], TromrCandidateHeadV1::Rhythm, rhythm_id)?;
        validate_candidate_head(&position.heads[1], TromrCandidateHeadV1::Pitch, pitch_id)?;
        validate_candidate_head(&position.heads[2], TromrCandidateHeadV1::Lift, lift_id)?;

        let emitted_eos = rhythm_id == crate::tokenizer::music::EOS_ID;
        if position.rhythm_emitted_eos != emitted_eos || (emitted_eos && index + 1 != stream_len) {
            return Err(candidate_lattice_mismatch(format!(
                "position {index} has inconsistent or non-terminal rhythm EOS"
            )));
        }
        saw_eos |= emitted_eos;
        prefix_rhythm.push(rhythm_id);
        prefix_pitch.push(pitch_id);
        prefix_lift.push(lift_id);
    }
    if !saw_eos && stream_len != MAX_SEQ {
        return Err(candidate_lattice_mismatch(format!(
            "generation stopped after {stream_len} positions without rhythm EOS or the {MAX_SEQ}-position cap"
        )));
    }
    Ok(())
}

fn generate_with_control_and_forward<C, F>(
    w: &TromrDecoderW,
    ctx: &Mat,
    pick: DecodePick,
    control: &C,
    mut forward: F,
) -> FocrResult<TromrGenerationWithCandidateLatticeV1>
where
    C: ExecutionControl + ?Sized,
    F: FnMut(&TromrDecoderW, &Mat, &[u32], &[u32], &[u32]) -> FocrResult<Mat>,
{
    let mut rng = match pick {
        DecodePick::Argmax => None,
        DecodePick::SeededSample { seed } => Some(Pcg32::new(seed)),
    };
    let mut rhythm = vec![SEED_RHYTHM];
    let mut pitch = vec![SEED_NONOTE];
    let mut lift = vec![SEED_NONOTE];
    let mut positions = Vec::with_capacity(MAX_SEQ);
    for position_zero_based in 0..MAX_SEQ {
        control.checkpoint("decode-token")?;
        // Upstream windows the prefix to the LAST max_seq_len positions.
        let start = rhythm.len().saturating_sub(MAX_SEQ);
        let prefix_sha256 = candidate_prefix_sha256(&rhythm, &pitch, &lift, start)?;
        let hidden = forward(w, ctx, &rhythm[start..], &pitch[start..], &lift[start..])?;
        let last = Mat::from_vec(1, DIM, hidden.row(hidden.rows - 1).to_vec());
        let rhythm_logits = w.head_rhythm.apply(&last)?;
        let (rhythm_id, rhythm_evidence) = select_and_capture_candidate_head(
            TromrCandidateHeadV1::Rhythm,
            &rhythm_logits.data,
            &mut rng,
        )?;
        let pitch_logits = w.head_pitch.apply(&last)?;
        let (pitch_id, pitch_evidence) = select_and_capture_candidate_head(
            TromrCandidateHeadV1::Pitch,
            &pitch_logits.data,
            &mut rng,
        )?;
        let lift_logits = w.head_lift.apply(&last)?;
        let (lift_id, lift_evidence) = select_and_capture_candidate_head(
            TromrCandidateHeadV1::Lift,
            &lift_logits.data,
            &mut rng,
        )?;

        rhythm.push(rhythm_id);
        pitch.push(pitch_id);
        lift.push(lift_id);
        let rhythm_emitted_eos = rhythm_id == crate::tokenizer::music::EOS_ID;
        positions.push(TromrCandidatePositionV1 {
            position_zero_based: position_zero_based as u32,
            prefix_length: (rhythm.len() - 1) as u32,
            prefix_window_start: start as u32,
            prefix_sha256,
            heads: [rhythm_evidence, pitch_evidence, lift_evidence],
            rhythm_emitted_eos,
        });
        if rhythm_emitted_eos {
            break;
        }
    }
    let streams = MusicStreams {
        rhythm: rhythm[1..].to_vec(),
        pitch: pitch[1..].to_vec(),
        lift: lift[1..].to_vec(),
    };
    let candidate_lattice = TromrCandidateLatticeV1 {
        schema_version: TROMR_CANDIDATE_LATTICE_SCHEMA_VERSION,
        contract_id: TROMR_CANDIDATE_LATTICE_CONTRACT_ID.to_owned(),
        canonical_encoding: TROMR_CANDIDATE_LATTICE_CANONICAL_ENCODING.to_owned(),
        top_n: TROMR_CANDIDATE_TOP_N as u32,
        chosen_streams: streams.clone(),
        positions,
    };
    candidate_lattice.validate()?;
    Ok(TromrGenerationWithCandidateLatticeV1 {
        streams,
        candidate_lattice,
    })
}

/// Generation over the encoder context: seeds rhythm=[BOS]=1,
/// pitch=lift=nonote=0; stops on rhythm `[EOS]`=2 or after [`MAX_SEQ`]
/// steps. The note head is inference-dead (spec §5) and skipped. Candidate
/// evidence is captured internally and validated without changing this legacy
/// streams-only API.
///
/// # Errors
/// A decoder-forward, candidate-capture, or cooperative-cancellation failure.
fn generate_with_control<C: ExecutionControl + ?Sized>(
    w: &TromrDecoderW,
    ctx: &Mat,
    pick: DecodePick,
    control: &C,
) -> FocrResult<MusicStreams> {
    Ok(generate_with_control_and_forward(w, ctx, pick, control, decoder_forward)?.streams)
}

/// Generation using the legacy process-wide cancellation bridge.
///
/// # Errors
/// A decoder-forward failure or cooperative cancellation.
pub fn generate_with(w: &TromrDecoderW, ctx: &Mat, pick: DecodePick) -> FocrResult<MusicStreams> {
    generate_with_control(w, ctx, pick, &LegacyExecutionControl)
}

/// Generate streams plus a bounded lattice of same-forward alternatives.
///
/// This invokes the decoder exactly once per emitted position. The selected
/// streams and PCG32 sampling order are identical to [`generate_with`]; the
/// retained score bits are uncalibrated model scores, not confidences.
///
/// # Errors
/// A decoder-forward, candidate-capture, validation, or cancellation failure.
pub fn generate_with_candidate_lattice(
    w: &TromrDecoderW,
    ctx: &Mat,
    pick: DecodePick,
) -> FocrResult<TromrGenerationWithCandidateLatticeV1> {
    generate_with_control_and_forward(w, ctx, pick, &LegacyExecutionControl, decoder_forward)
}

/// The ARGMAX decode — the L4 oracle-parity mode (degenerate on real staves,
/// DISC-007; the product default is [`generate`]).
///
/// # Errors
/// A decoder-forward failure.
pub fn generate_argmax(w: &TromrDecoderW, ctx: &Mat) -> FocrResult<MusicStreams> {
    generate_with(w, ctx, DecodePick::Argmax)
}

/// Decode with explicit, validated recognition options.
///
/// # Errors
/// An invalid options contract or decoder-forward failure.
pub fn generate_with_recognition_options(
    w: &TromrDecoderW,
    ctx: &Mat,
    options: TromrRecognitionOptionsV1,
) -> FocrResult<MusicStreams> {
    generate_with(w, ctx, options.decode_pick()?)
}

/// The product default: per-head ARGMAX — deterministic, and MEASURED
/// equivalent to upstream's top-k/T=0.2 sampling on real staves (identical
/// SER 0.211 across the 4 committed examples, 2026-07-06 — the sharp T=0.2
/// almost always picks the argmax token; DISC-007's apparent "argmax
/// collapse" was a blank-input artifact of the upstream alpha bug).
///
/// # Errors
/// A decoder-forward failure.
pub fn generate(w: &TromrDecoderW, ctx: &Mat) -> FocrResult<MusicStreams> {
    generate_with_recognition_options(w, ctx, TromrRecognitionOptionsV1::default())
}

// ───────────────── E7: semantic merge + MusicXML assembly ─────────────────

/// Merge the three RAW id streams into the extended-PrIMuS semantic string
/// (upstream `inference.py`, ported verbatim over index-aligned tokens —
/// spec §8):
///
/// * rhythm `|` replaces the previous joiner with `|` (chord join,
///   bottom-to-top);
/// * a rhythm token CONTAINING `"note"` renders
///   `<pitch><lift?>_<duration>` — the pitch token verbatim (a `nonote`
///   pitch stays `nonote_<dur>`, exactly what upstream emits), the lift
///   letter appended only for the five real accidental classes;
/// * every other rhythm token passes through; all joined by `+`.
///
/// Port rules (spec §8, replacing upstream's delete-anywhere loop): the
/// streams stay INDEX-ALIGNED; the trailing rhythm `[EOS]` (and the aligned
/// pitch/lift tails) are stripped; any OTHER control id in any stream is a
/// decode error — fail loud, never skip-and-shift.
///
/// # Errors
/// Length mismatches, an id outside its table, or a mid-stream control id.
pub fn merge_semantic(
    tk: &crate::tokenizer::music::MusicTokenizer,
    streams: &MusicStreams,
) -> FocrResult<String> {
    use crate::tokenizer::music::{EOS_ID, Stream};
    let t = streams.rhythm.len();
    if t == 0 || streams.pitch.len() != t || streams.lift.len() != t {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr merge: stream lens (r {}, p {}, l {}) must be equal and non-zero",
            streams.rhythm.len(),
            streams.pitch.len(),
            streams.lift.len()
        )));
    }
    // Strip the trailing rhythm [EOS] and the aligned tails.
    let end = if streams.rhythm[t - 1] == EOS_ID {
        t - 1
    } else {
        t
    };
    let mut parts: Vec<String> = Vec::with_capacity(end);
    for j in 0..end {
        let r_tok = tk.token(Stream::Rhythm, streams.rhythm[j]).ok_or_else(|| {
            FocrError::Other(anyhow::anyhow!(
                "tromr merge: rhythm id {} out of table",
                streams.rhythm[j]
            ))
        })?;
        if matches!(r_tok, "[BOS]" | "[EOS]" | "[PAD]") {
            return Err(FocrError::Other(anyhow::anyhow!(
                "tromr merge: mid-stream rhythm control token {r_tok:?} at step {j} — decode error"
            )));
        }
        if r_tok == "|" {
            // Chord join: fuse with the PREVIOUS event.
            let Some(prev) = parts.last_mut() else {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "tromr merge: chord '|' with no preceding event"
                )));
            };
            prev.push('|');
            continue;
        }
        if r_tok.contains("note") {
            let p_tok = tk.token(Stream::Pitch, streams.pitch[j]).ok_or_else(|| {
                FocrError::Other(anyhow::anyhow!(
                    "tromr merge: pitch id {} out of table",
                    streams.pitch[j]
                ))
            })?;
            let l_tok = tk.token(Stream::Lift, streams.lift[j]).ok_or_else(|| {
                FocrError::Other(anyhow::anyhow!(
                    "tromr merge: lift id {} out of table",
                    streams.lift[j]
                ))
            })?;
            let lift = match l_tok {
                "lift_##" | "lift_#" | "lift_bb" | "lift_b" | "lift_N" => {
                    l_tok.rsplit('_').next().unwrap_or("")
                }
                _ => "",
            };
            let dur = r_tok.rsplit("note-").next().unwrap_or(r_tok);
            let rendered = format!("{p_tok}{lift}_{dur}");
            match parts.last_mut() {
                Some(prev) if prev.ends_with('|') => prev.push_str(&rendered),
                _ => parts.push(rendered),
            }
        } else {
            match parts.last_mut() {
                Some(prev) if prev.ends_with('|') => prev.push_str(r_tok),
                _ => parts.push(r_tok.to_owned()),
            }
        }
    }
    Ok(parts.join("+"))
}

/// The rhythm duration names → (MusicXML `<type>`, ticks at 64
/// divisions-per-quarter, dotted) — spec §8/§9 duration table.
fn duration_info(name: &str) -> Option<(&'static str, u32, bool)> {
    let (base, dotted) = match name.strip_suffix('.') {
        Some(b) => (b, true),
        None => (name, false),
    };
    let (xml, ticks) = match base {
        "long" => ("long", 1024),
        "breve" => ("breve", 512),
        "whole" => ("whole", 256),
        "half" => ("half", 128),
        "quarter" => ("quarter", 64),
        "eighth" => ("eighth", 32),
        "sixteenth" => ("16th", 16),
        "thirty_second" => ("32nd", 8),
        "sixty_fourth" => ("64th", 4),
        "hundred_twenty_eighth" => ("128th", 2),
        // The rhythm vocab's two finest rests use numeral names, not the
        // spelled-out forms. At 64 divisions/quarter a 256th is exactly 1
        // tick; a 512th would be 0.5, floored to 1 — `<type>` stays exact
        // and measure sums at that extreme are already model-approximate.
        "256th" => ("256th", 1),
        "512th" => ("512th", 1),
        _ => return None,
    };
    Some((xml, if dotted { ticks * 3 / 2 } else { ticks }, dotted))
}

/// Split a pitched atom `<head>_<duration>` on its separator underscore.
/// The duration name may itself contain underscores (`thirty_second`,
/// `sixty_fourth`, `hundred_twenty_eighth`), so a positional
/// `rsplit_once('_')` mis-splits those (bd-av64.1: `note-B4_thirty_second`
/// parsed as duration `"second"` and aborted the run). Scan candidate
/// separators left-to-right and take the first whose suffix is a known
/// duration — the longest duration candidate wins.
fn split_pitch_duration(atom: &str) -> Option<(&str, (&'static str, u32, bool))> {
    atom.match_indices('_')
        .find_map(|(i, _)| duration_info(&atom[i + 1..]).map(|info| (&atom[..i], info)))
}

/// `keySignature-XM` → MusicXML circle-of-fifths value (the 15 majors).
fn key_fifths(name: &str) -> Option<i32> {
    Some(match name {
        "CM" => 0,
        "GM" => 1,
        "DM" => 2,
        "AM" => 3,
        "EM" => 4,
        "BM" => 5,
        "F#M" => 6,
        "C#M" => 7,
        "FM" => -1,
        "BbM" => -2,
        "EbM" => -3,
        "AbM" => -4,
        "DbM" => -5,
        "GbM" => -6,
        "CbM" => -7,
        _ => return None,
    })
}

/// One parsed note within an event (chord group).
struct XmlNote {
    step: char,
    octave: u32,
    alter: Option<i32>,
    natural: bool,
    rest: bool,
    xml_type: &'static str,
    ticks: u32,
    dotted: bool,
}

/// Serialize the merged semantic string to partwise MusicXML (spec §8: the
/// primary interop export; the raw semantic string ships beside it in
/// `--json`). One part; measures split on `barline`; `multirest-N` expands
/// to N whole-measure rests; a `nonote_<dur>` event (the pitch head
/// abstained on a note step) renders as a rest of that duration — the
/// semantic string keeps the model-native `nonote` form for scoring.
///
/// # Errors
/// A token that parses as none of the §9 vocabulary classes.
pub fn semantic_to_musicxml(merged: &str) -> FocrResult<String> {
    staves_to_musicxml(std::slice::from_ref(&merged.to_owned()))
}

/// Multi-staff MusicXML: one `<part>` per staff (P1..PN, top-to-bottom —
/// the E5 full-page contract; cross-staff beat alignment is the deferred
/// `**kern` follow-up's concern).
///
/// # Errors
/// As [`semantic_to_musicxml`], per staff.
pub fn staves_to_musicxml(semantics: &[String]) -> FocrResult<String> {
    let mut part_list = String::new();
    let mut parts = String::new();
    for (i, merged) in semantics.iter().enumerate() {
        let id = i + 1;
        part_list.push_str(&format!(
            "<score-part id=\"P{id}\"><part-name>Staff {id}</part-name></score-part>"
        ));
        parts.push_str(&format!(
            "  <part id=\"P{id}\">\n{}\n  </part>\n",
            part_measures(merged)?
        ));
    }
    // Annotate-only musical-sanity observations (bd-av64.5): XML comments
    // never change the musical content, and importers ignore them.
    let mut annotations = String::new();
    for w in sanity_warnings(semantics) {
        annotations.push_str(&format!(
            "  <!--focr-sanity: {} part {} measure {}: {}-->\n",
            w.kind,
            w.part,
            w.measure,
            w.detail.replace("--", "-")
        ));
    }
    let xml = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <score-partwise version=\"4.0\">\n\
         \x20 <part-list>{part_list}</part-list>\n{parts}{annotations}</score-partwise>\n"
    );
    // Emit-time enforcement (bd-av64.3): a structural violation here is by
    // definition an emitter bug — never a model-quality artifact — so every
    // produced document is valid-by-construction or the run fails loud.
    let violations = validate_musicxml(&xml);
    if violations.is_empty() {
        Ok(xml)
    } else {
        Err(FocrError::Other(anyhow::anyhow!(
            "tromr xml: emitter produced invalid MusicXML (emitter bug): {}",
            violations.join("; ")
        )))
    }
}

/// One musical-sanity warning from [`sanity_warnings`] (bd-av64.5):
/// annotate-only observations about the RECOGNIZED content — never a
/// rejection (model output is legitimately imperfect; hard structural
/// legality is [`validate_musicxml`]'s job).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MusicWarning {
    /// Stable machine kind: `overfull_bar` | `underfull_bar` |
    /// `impossible_duration` | `key_mismatch`.
    pub kind: &'static str,
    /// 1-based part (staff) number.
    pub part: usize,
    /// 1-based measure number (0 = the warning is staff-level).
    pub measure: usize,
    /// Human detail with the numbers that triggered the flag.
    pub detail: String,
}

/// Musical-sanity analysis over per-staff semantic streams (bd-av64.5),
/// annotate-only per the alien-artifact contract (the deterministic
/// fallback IS annotation; auto-correction would need a measured-win
/// ledger):
///
/// * **bar sums** vs the active time signature — overfull always flags;
///   underfull flags EXCEPT for the classical exemptions: a pickup
///   (anacrusis) first measure, and a final measure (which may complement
///   the pickup or be cut by the line end). A mid-stream time-signature
///   change resets the expectation.
/// * **impossible durations** — a single note longer than the whole bar
///   (the real Cadwallader read contained a whole note in 3/4).
/// * **cross-staff key consistency** — staves of one system share a key
///   signature by engraving convention; disagreement flags every minority
///   staff with the majority as the suggestion (never rewritten).
#[must_use]
pub fn sanity_warnings(semantics: &[String]) -> Vec<MusicWarning> {
    let mut out = Vec::new();
    let mut keys: Vec<Option<String>> = Vec::new();
    for (pi, sem) in semantics.iter().enumerate() {
        let part = pi + 1;
        let mut bar_ticks_expected: Option<u32> = None;
        let mut measure = 1usize;
        let mut sum = 0u32;
        let mut key: Option<String> = None;
        let mut measure_flagged = false;
        let mut had_pickup_deficit = false;
        let mut pending: Vec<(usize, u32)> = Vec::new(); // underfull candidates
        for event in sem.split('+') {
            if event.is_empty() {
                continue;
            }
            if let Some(k) = event.strip_prefix("keySignature-") {
                key.get_or_insert_with(|| k.to_owned());
                continue;
            }
            if let Some(ts) = event.strip_prefix("timeSignature-") {
                bar_ticks_expected = match ts {
                    "C" => Some(256),
                    "C/" => Some(128),
                    other => other.split_once('/').and_then(|(b, t)| {
                        let b: u32 = b.parse().ok()?;
                        let t: u32 = t.parse().ok()?;
                        Some(b * 256 / t)
                    }),
                };
                continue;
            }
            if event == "barline" {
                if let Some(expected) = bar_ticks_expected {
                    if sum > expected && !measure_flagged {
                        out.push(MusicWarning {
                            kind: "overfull_bar",
                            part,
                            measure,
                            detail: format!("{sum} ticks in a {expected}-tick measure"),
                        });
                    } else if sum > 0 && sum < expected {
                        if measure == 1 {
                            // Anacrusis: classical and unflagged.
                            had_pickup_deficit = true;
                        } else {
                            // Defer: a FINAL underfull measure is exempt
                            // (complements the pickup / line-end cut).
                            pending.push((measure, sum));
                        }
                    }
                }
                measure += 1;
                sum = 0;
                measure_flagged = false;
                continue;
            }
            if event.starts_with("clef-") || event.starts_with("multirest-") {
                continue;
            }
            for atom in event.split('|') {
                let dur = if let Some(d) = atom.strip_prefix("rest-") {
                    duration_info(d)
                } else {
                    split_pitch_duration(atom).map(|(_, info)| info)
                };
                let Some((_, ticks, _)) = dur else { continue };
                if let Some(expected) = bar_ticks_expected
                    && ticks > expected
                    && !measure_flagged
                {
                    out.push(MusicWarning {
                        kind: "impossible_duration",
                        part,
                        measure,
                        detail: format!("{ticks}-tick note in a {expected}-tick measure"),
                    });
                    measure_flagged = true;
                }
                // Chord members sound together: count the group ONCE (the
                // longest member governs; approximating with the first).
                sum += ticks;
                break;
            }
        }
        // Trailing content without a final barline is a line-end cut: exempt.
        // Deferred underfull measures: the LAST one is exempt (may pair with
        // the pickup); earlier ones flag.
        let exempt_last = pending.len().saturating_sub(1);
        let _ = had_pickup_deficit;
        for &(m, got) in &pending[..exempt_last] {
            out.push(MusicWarning {
                kind: "underfull_bar",
                part,
                measure: m,
                detail: format!(
                    "{got} ticks in a {}-tick measure",
                    bar_ticks_expected.unwrap_or(0)
                ),
            });
        }
        keys.push(key);
    }
    // Cross-staff key consistency (only meaningful with >= 2 keyed staves).
    let known: Vec<(usize, &String)> = keys
        .iter()
        .enumerate()
        .filter_map(|(i, k)| k.as_ref().map(|k| (i, k)))
        .collect();
    if known.len() >= 2 {
        let mut counts: std::collections::BTreeMap<&String, usize> = Default::default();
        for (_, k) in &known {
            *counts.entry(k).or_default() += 1;
        }
        if counts.len() > 1 {
            let majority = counts
                .iter()
                .max_by_key(|entry| *entry.1)
                .map(|(k, _)| (*k).clone())
                .unwrap_or_default();
            for (i, k) in &known {
                if **k != majority {
                    out.push(MusicWarning {
                        kind: "key_mismatch",
                        part: i + 1,
                        measure: 0,
                        detail: format!(
                            "staff reads keySignature-{k} while the system majority is \
                             keySignature-{majority}"
                        ),
                    });
                }
            }
        }
    }
    out
}

/// Structural MusicXML lint over the emitter's output (bd-av64.3). Empty
/// result = pass. Rules: balanced tags under exactly one `score-partwise`
/// root; `<chord/>` never co-occurs with `<rest/>` in one note; a
/// `<chord/>` note directly follows another note in its measure; every
/// note carries a positive integer `<duration>`; `part-list` score-part
/// ids match the `<part>` ids in order. Musical bar-sum checks are
/// deliberately NOT here: model output is legitimately imperfect and
/// hard-failing on it would reject honest recognitions (the annotate-only
/// sanity pass, bd-av64.5, owns that concern).
pub fn validate_musicxml(xml: &str) -> Vec<String> {
    let mut violations = Vec::new();
    let mut stack: Vec<String> = Vec::new();
    let mut roots = 0usize;
    let mut score_part_ids: Vec<String> = Vec::new();
    let mut part_ids: Vec<String> = Vec::new();
    let mut in_note = false;
    let mut note_line = 0usize;
    let (mut note_chord, mut note_rest, mut note_chord_legal) = (false, false, false);
    let mut note_duration: Option<i64> = None;
    let mut prev_was_note = false;

    fn id_attr(raw: &str) -> Option<String> {
        let rest = &raw[raw.find("id=\"")? + 4..];
        Some(rest[..rest.find('"')?].to_owned())
    }
    let line_of = |pos: usize| xml[..pos].bytes().filter(|&b| b == b'\n').count() + 1;

    let mut pos = 0usize;
    while let Some(lt) = xml[pos..].find('<') {
        let start = pos + lt;
        let Some(gt) = xml[start..].find('>') else {
            violations.push(format!("unterminated tag at line {}", line_of(start)));
            return violations;
        };
        let raw = &xml[start + 1..start + gt];
        pos = start + gt + 1;
        if raw.starts_with('?') || raw.starts_with('!') {
            continue;
        }
        let closing = raw.starts_with('/');
        let self_closing = raw.ends_with('/');
        let name = raw
            .trim_start_matches('/')
            .trim_end_matches('/')
            .split_whitespace()
            .next()
            .unwrap_or("");
        if name.is_empty() {
            violations.push(format!("empty tag at line {}", line_of(start)));
            continue;
        }
        if closing {
            match stack.pop() {
                Some(open) if open == name => {}
                Some(open) => violations.push(format!(
                    "mismatched </{name}> closing <{open}> at line {}",
                    line_of(start)
                )),
                None => violations.push(format!(
                    "</{name}> with nothing open at line {}",
                    line_of(start)
                )),
            }
            if name == "note" && in_note {
                if note_chord && note_rest {
                    violations.push(format!(
                        "<chord/> co-occurs with <rest/> at line {note_line}"
                    ));
                }
                if note_chord && !note_chord_legal {
                    violations.push(format!(
                        "<chord/> note not directly preceded by a note at line {note_line}"
                    ));
                }
                match note_duration {
                    Some(v) if v > 0 => {}
                    Some(v) => {
                        violations.push(format!("non-positive <duration> {v} at line {note_line}"))
                    }
                    None => {
                        violations.push(format!("<note> missing <duration> at line {note_line}"));
                    }
                }
                in_note = false;
                prev_was_note = true;
            }
            continue;
        }
        match name {
            "score-partwise" if stack.is_empty() => roots += 1,
            "score-part" => match id_attr(raw) {
                Some(id) => score_part_ids.push(id),
                None => violations.push(format!(
                    "<score-part> missing id at line {}",
                    line_of(start)
                )),
            },
            "part" => match id_attr(raw) {
                Some(id) => part_ids.push(id),
                None => violations.push(format!("<part> missing id at line {}", line_of(start))),
            },
            "measure" | "attributes" => prev_was_note = false,
            "note" => {
                in_note = true;
                note_line = line_of(start);
                (note_chord, note_rest, note_chord_legal) = (false, false, prev_was_note);
                note_duration = None;
            }
            "chord" if in_note => note_chord = true,
            "rest" if in_note => note_rest = true,
            "duration" if in_note => {
                let text = &xml[pos..];
                let end = text.find('<').unwrap_or(0);
                match text[..end].trim().parse::<i64>() {
                    Ok(v) => note_duration = Some(v),
                    Err(_) => violations.push(format!(
                        "unparseable <duration> {:?} at line {}",
                        &text[..end],
                        line_of(start)
                    )),
                }
            }
            _ => {}
        }
        if !self_closing {
            stack.push(name.to_owned());
        }
    }
    if !stack.is_empty() {
        violations.push(format!("unclosed tags: {}", stack.join(", ")));
    }
    if roots != 1 {
        violations.push(format!(
            "expected exactly one <score-partwise> root, found {roots}"
        ));
    }
    if score_part_ids != part_ids {
        violations.push(format!(
            "part-list ids {score_part_ids:?} do not match part ids {part_ids:?}"
        ));
    }
    violations
}

/// The per-part measure builder (the body of one `<part>`).
fn part_measures(merged: &str) -> FocrResult<String> {
    let mut measures: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut attributes = String::new();
    let mut divisions_emitted = false;

    fn flush(current: &mut String, measures: &mut Vec<String>) {
        if !current.is_empty() {
            let n = measures.len() + 1;
            measures.push(format!("  <measure number=\"{n}\">\n{current}  </measure>"));
            current.clear();
        }
    }

    for event in merged.split('+') {
        if event.is_empty() {
            continue;
        }
        if let Some(clef) = event.strip_prefix("clef-") {
            let (sign, line) = clef.split_at(1);
            attributes.push_str(&format!(
                "      <clef><sign>{sign}</sign><line>{line}</line></clef>\n"
            ));
            continue;
        }
        if let Some(key) = event.strip_prefix("keySignature-") {
            let fifths = key_fifths(key).ok_or_else(|| {
                FocrError::Other(anyhow::anyhow!("tromr xml: unknown key {event:?}"))
            })?;
            attributes.push_str(&format!("      <key><fifths>{fifths}</fifths></key>\n"));
            continue;
        }
        if let Some(ts) = event.strip_prefix("timeSignature-") {
            let (beats, beat_type, symbol) = match ts {
                "C" => (4, 4, " symbol=\"common\""),
                "C/" => (2, 2, " symbol=\"cut\""),
                other => {
                    let (b, t) = other.split_once('/').ok_or_else(|| {
                        FocrError::Other(anyhow::anyhow!("tromr xml: bad time {event:?}"))
                    })?;
                    let b = b.parse::<u32>().map_err(|_| {
                        FocrError::Other(anyhow::anyhow!("tromr xml: bad beats {event:?}"))
                    })?;
                    let t = t.parse::<u32>().map_err(|_| {
                        FocrError::Other(anyhow::anyhow!("tromr xml: bad beat-type {event:?}"))
                    })?;
                    (b, t, "")
                }
            };
            attributes.push_str(&format!(
                "      <time{symbol}><beats>{beats}</beats><beat-type>{beat_type}</beat-type></time>\n"
            ));
            continue;
        }
        if event == "barline" {
            flush(&mut current, &mut measures);
            continue;
        }
        if let Some(n) = event.strip_prefix("multirest-") {
            let n: usize = n.parse().map_err(|_| {
                FocrError::Other(anyhow::anyhow!("tromr xml: bad multirest {event:?}"))
            })?;
            flush(&mut current, &mut measures);
            for _ in 0..n {
                current
                    .push_str("    <note><rest measure=\"yes\"/><duration>256</duration></note>\n");
                flush(&mut current, &mut measures);
            }
            continue;
        }

        // Note / rest event (possibly a `|`-joined chord group).
        let mut notes: Vec<XmlNote> = Vec::new();
        for atom in event.split('|') {
            if let Some(dur) = atom.strip_prefix("rest-") {
                let (xml_type, ticks, dotted) = duration_info(dur).ok_or_else(|| {
                    FocrError::Other(anyhow::anyhow!("tromr xml: unknown duration {atom:?}"))
                })?;
                notes.push(XmlNote {
                    step: 'C',
                    octave: 4,
                    alter: None,
                    natural: false,
                    rest: true,
                    xml_type,
                    ticks,
                    dotted,
                });
                continue;
            }
            let (head, (xml_type, ticks, dotted)) =
                split_pitch_duration(atom).ok_or_else(|| {
                    if atom.contains('_') {
                        FocrError::Other(anyhow::anyhow!("tromr xml: unknown duration {atom:?}"))
                    } else {
                        FocrError::Other(anyhow::anyhow!("tromr xml: unparseable event {atom:?}"))
                    }
                })?;
            if head == "nonote" {
                notes.push(XmlNote {
                    step: 'C',
                    octave: 4,
                    alter: None,
                    natural: false,
                    rest: true,
                    xml_type,
                    ticks,
                    dotted,
                });
                continue;
            }
            let body = head.strip_prefix("note-").ok_or_else(|| {
                FocrError::Other(anyhow::anyhow!("tromr xml: unparseable note {atom:?}"))
            })?;
            let mut it = body.chars();
            let step = it.next().ok_or_else(|| {
                FocrError::Other(anyhow::anyhow!("tromr xml: empty note {atom:?}"))
            })?;
            let octave: String = body[1..].chars().take_while(char::is_ascii_digit).collect();
            let acc = &body[1 + octave.len()..];
            let octave: u32 = octave
                .parse()
                .map_err(|_| FocrError::Other(anyhow::anyhow!("tromr xml: bad octave {atom:?}")))?;
            let (alter, natural) = match acc {
                "" => (None, false),
                "#" => (Some(1), false),
                "##" => (Some(2), false),
                "b" => (Some(-1), false),
                "bb" => (Some(-2), false),
                "N" => (Some(0), true),
                other => {
                    return Err(FocrError::Other(anyhow::anyhow!(
                        "tromr xml: unknown accidental {other:?} in {atom:?}"
                    )));
                }
            };
            notes.push(XmlNote {
                step,
                octave,
                alter,
                natural,
                rest: false,
                xml_type,
                ticks,
                dotted,
            });
        }

        // A '|'-joined group is a chord: only its pitched members may sound
        // together. MusicXML 4.0 forbids <chord/> on a rest (a rest cannot
        // sound simultaneously with a note in one voice), so a mixed group
        // drops its rests — the pitched notes carry the group's duration —
        // and an all-rest group collapses to its first rest (bd-av64.3; the
        // 2026-07-06 Cadwallader run emitted `<chord/><rest/>`, which
        // importers reject).
        if notes.iter().any(|n| !n.rest) {
            notes.retain(|n| !n.rest);
        } else {
            notes.truncate(1);
        }

        if !attributes.is_empty() {
            let divisions = if divisions_emitted {
                String::new()
            } else {
                divisions_emitted = true;
                "      <divisions>64</divisions>\n".to_owned()
            };
            current.push_str(&format!(
                "    <attributes>\n{divisions}{attributes}    </attributes>\n"
            ));
            attributes.clear();
        }
        for (i, n) in notes.iter().enumerate() {
            let mut body = String::new();
            if i > 0 {
                body.push_str("<chord/>");
            }
            if n.rest {
                body.push_str("<rest/>");
            } else {
                let alter = n
                    .alter
                    .map(|a| format!("<alter>{a}</alter>"))
                    .unwrap_or_default();
                body.push_str(&format!(
                    "<pitch><step>{}</step>{alter}<octave>{}</octave></pitch>",
                    n.step, n.octave
                ));
            }
            body.push_str(&format!(
                "<duration>{}</duration><type>{}</type>",
                n.ticks, n.xml_type
            ));
            if n.dotted {
                body.push_str("<dot/>");
            }
            if n.natural {
                body.push_str("<accidental>natural</accidental>");
            }
            current.push_str(&format!("    <note>{body}</note>\n"));
        }
    }
    flush(&mut current, &mut measures);
    Ok(measures.join("\n"))
}

// ───────────────── E9: the recognize assembly ─────────────────

/// The music-recognition result: the raw model-native semantic string (what
/// SER scoring consumes; ships in `--json`), partwise MusicXML (the primary
/// interop export — spec §8), and the exact inference controls needed to
/// replay it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MusicResult {
    /// The merged extended-PrIMuS semantic stream.
    pub semantic: String,
    /// Partwise MusicXML 4.0.
    pub musicxml: String,
    /// Validated, normalized inference controls used for this result.
    pub options: TromrRecognitionOptionsV1,
    /// SHA-256 of the canonical options JSON.
    pub options_identity: String,
    /// Ordered, independently seeded candidate evidence for every exact image
    /// forwarded while producing this row result. The index in each wrapper
    /// must equal its vector position and the matching forward-input index.
    pub forward_candidate_lattices: Vec<TromrForwardCandidateLatticeV1>,
}

impl MusicResult {
    /// Validate that candidate evidence accounts for every ordered forward
    /// input exactly once without flattening decoder prefixes between inputs.
    ///
    /// # Errors
    /// A count, ordering, wrapper-schema, or candidate-lattice mismatch.
    pub fn validate_forward_candidate_lattices(
        &self,
        expected_forward_input_count: usize,
    ) -> FocrResult<()> {
        if self.forward_candidate_lattices.len() != expected_forward_input_count {
            return Err(candidate_lattice_mismatch(format!(
                "music result has {} per-forward lattices for {expected_forward_input_count} forward inputs",
                self.forward_candidate_lattices.len()
            )));
        }
        for (expected_index, candidate) in self.forward_candidate_lattices.iter().enumerate() {
            candidate.validate_at_ordered_index(expected_index)?;
        }
        Ok(())
    }
}

/// Page-space staff bounding box `(x, y, w, h)`.
pub type StaffBBox = (usize, usize, usize, usize);

/// Mechanical outcome of one staff-row inference attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaffInferenceOutcome {
    Recognized,
    Skipped,
}

impl StaffInferenceOutcome {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Recognized => "recognized",
            Self::Skipped => "skipped",
        }
    }
}

/// Exact route by which a row-local semantic stream reached TrOMR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TromrRowInferenceRouteV1 {
    /// No staff was detected. The original selected-page raster was forwarded,
    /// but there is no detector-backed review crop or five-line evidence.
    NoDetectedStaffWholeRasterFallback,
    /// Exactly one staff was detected; the original selected-page raster was
    /// forwarded while the detected crop is retained separately for review.
    SingleDetectedStaffWholeRaster,
    /// A detected/refined/padded staff crop was forwarded once.
    DetectedStaffCrop,
    /// Experimental barline splitting forwarded multiple exact segment inputs.
    ExperimentalSplitSegments,
}

impl TromrRowInferenceRouteV1 {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoDetectedStaffWholeRasterFallback => "no_detected_staff_whole_raster_fallback",
            Self::SingleDetectedStaffWholeRaster => "single_detected_whole_raster",
            Self::DetectedStaffCrop => "detected_staff_crop",
            Self::ExperimentalSplitSegments => "experimental_split_segments",
        }
    }
}

/// Coordinate domain containing one exact TrOMR forward input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TromrModelInputSourceSpaceV1 {
    /// Original selected-page raster, before the detector's global deskew.
    SelectedPageRaster,
    /// Provider-owned detector/refinement inference canvas.
    ReviewCropCanvas,
}

impl TromrModelInputSourceSpaceV1 {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SelectedPageRaster => "selected_page_raster",
            Self::ReviewCropCanvas => "review_crop_canvas",
        }
    }
}

/// Exact pixels and role-specific geometry for one image forwarded through
/// TrOMR preprocessing. Split recognition has one entry per forwarded segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TromrForwardInputV1 {
    pub gray8: crate::preprocess::staff_detect::TromrGray8CropV1,
    pub source_space: TromrModelInputSourceSpaceV1,
    /// Bounding box in `source_space`; a split segment uses its exact rectangle
    /// within the review-crop canvas.
    pub source_bbox_xywh: StaffBBox,
    /// Synthetic white margins present inside this exact forwarded image,
    /// ordered top/right/bottom/left. This describes pixel content, not a
    /// transform from `source_bbox_xywh`.
    pub padding: crate::preprocess::staff_detect::StaffPadding,
    /// Present only when the exact forward canvas has proven line coordinates.
    pub staff_lines_y_in_canvas: Option<[usize; 5]>,
}

/// Detector and review-canvas five-line anchors, with spaces named explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TromrStaffLineEvidenceV1 {
    pub accepted_detector_lines_y_in_globally_deskewed_raster: [usize; 5],
    pub review_crop_staff_lines_y_in_canvas: [usize; 5],
}

/// Structured source/canvas/outcome evidence for one staff-row inference
/// attempt (bd-av64.16). On the per-crop route the index is the detector index;
/// zero-detection fallback instead synthesizes route-local attempt index 0.
/// This never implies system membership or persistent part identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaffInferenceEvidence {
    /// 0-based route-local attempt index, top-to-bottom. This is the detector
    /// index except for zero-detection whole-image fallback.
    pub index: usize,
    /// Real page-space bbox versus the padded model-inference canvas.
    pub geometry: crate::preprocess::staff_detect::StaffCropGeometry,
    /// Exact forward route. It never implies topology or recognition quality.
    pub route: TromrRowInferenceRouteV1,
    /// Every exact Gray8 image forwarded for this attempt, in forward order.
    pub forward_inputs: Vec<TromrForwardInputV1>,
    /// Detector-derived row-local crop used for musician review. Absent only
    /// for zero-detection whole-raster fallback.
    pub review_crop_gray8: Option<crate::preprocess::staff_detect::TromrGray8CropV1>,
    /// Geometry for `review_crop_gray8` in the globally deskewed detector
    /// raster plus its independent review canvas. Absent with the crop.
    pub review_crop_geometry: Option<crate::preprocess::staff_detect::StaffCropGeometry>,
    /// Required for detector-backed rows; absent for zero-detection fallback.
    pub staff_lines: Option<TromrStaffLineEvidenceV1>,
    /// Whether TrOMR returned a row-local semantic stream.
    pub outcome: StaffInferenceOutcome,
    /// Exact error when skipped; absent when recognized.
    pub reason: Option<String>,
}

/// The full E9 single-staff pipeline: §6 preprocess → the certified encoder →
/// deterministic argmax generate → §8 merge → MusicXML. The input must be a
/// single-staff crop (the width guard rejects > 1280 at h=128; full-page
/// staff detection is the E5 front end).
///
/// # Errors
/// A preprocess/width violation, a missing tensor, or a decode error.
fn recognize_with_control<C: ExecutionControl + ?Sized>(
    weights: &Weights,
    tk: &crate::tokenizer::music::MusicTokenizer,
    img: &image::DynamicImage,
    options: TromrRecognitionOptionsV1,
    control: &mut C,
) -> FocrResult<MusicResult> {
    let options = options.validate()?;
    let options_identity = options.replay_identity()?;
    control.begin_forward_attempt("staff-forward")?;
    let result = (|| {
        control.checkpoint("staff-preprocess")?;
        let preprocess_started = std::time::Instant::now();
        let preprocessed = match options.staff_resampler {
            TromrStaffResamplerV1::Cv2LinearU8V1 => crate::preprocess::tromr_staff_tensor(img),
        };
        control.record_attempt_stage(TromrAttemptStage::Preprocess, preprocess_started.elapsed());
        let (pixels, width) = preprocessed?;

        control.checkpoint("staff-encode")?;
        let encode_started = std::time::Instant::now();
        let encoded = (|| {
            let enc = TromrEncoderW::build(weights)?;
            encode(&enc, &pixels, width)
        })();
        control.record_attempt_stage(TromrAttemptStage::Encode, encode_started.elapsed());
        let ctx = encoded?;
        control.checkpoint("staff-encode")?;
        super::timing_log(&format!(
            "  tromr.encode {:.2}s (w {width}, {} ctx tokens)",
            encode_started.elapsed().as_secs_f64(),
            ctx.rows
        ));

        let decode_started = std::time::Instant::now();
        let decoded = (|| {
            let dec = TromrDecoderW::build(weights)?;
            generate_with_control_and_forward(
                &dec,
                &ctx,
                options.decode_pick()?,
                control,
                decoder_forward,
            )
        })();
        control.record_attempt_stage(TromrAttemptStage::Decode, decode_started.elapsed());
        let generation = decoded?;
        let forward_candidate =
            TromrForwardCandidateLatticeV1::new(0, generation.candidate_lattice)?;
        let streams = generation.streams;
        control.checkpoint("staff-semantic-assembly")?;
        super::timing_log(&format!(
            "  tromr.generate {} steps {:.2}s",
            streams.rhythm.len(),
            decode_started.elapsed().as_secs_f64()
        ));

        let semantic_started = std::time::Instant::now();
        let assembled: FocrResult<(String, String)> = (|| {
            let semantic = merge_semantic(tk, &streams)?;
            let musicxml = semantic_to_musicxml(&semantic)?;
            Ok((semantic, musicxml))
        })();
        control.record_attempt_stage(
            TromrAttemptStage::SemanticAssembly,
            semantic_started.elapsed(),
        );
        let (semantic, musicxml) = assembled?;
        control.checkpoint("staff-semantic-assembly")?;
        let result = MusicResult {
            semantic,
            musicxml,
            options,
            options_identity,
            forward_candidate_lattices: vec![forward_candidate],
        };
        result.validate_forward_candidate_lattices(1)?;
        Ok(result)
    })();
    control.finish_attempt(&result);
    result
}

/// The full E9 single-staff pipeline with explicit recognition mechanics and
/// the legacy process-wide cancellation bridge.
///
/// # Errors
/// A preprocess/width violation, a missing tensor, a decode error, or
/// cooperative cancellation.
pub fn recognize_with_options(
    weights: &Weights,
    tk: &crate::tokenizer::music::MusicTokenizer,
    img: &image::DynamicImage,
    options: TromrRecognitionOptionsV1,
) -> FocrResult<MusicResult> {
    recognize_with_control(weights, tk, img, options, &mut LegacyExecutionControl)
}

/// Recognize one staff with [`TromrRecognitionOptionsV1::deterministic`].
///
/// # Errors
/// A preprocess/width violation, a missing tensor, or a decode error.
pub fn recognize(
    weights: &Weights,
    tk: &crate::tokenizer::music::MusicTokenizer,
    img: &image::DynamicImage,
) -> FocrResult<MusicResult> {
    recognize_with_options(weights, tk, img, TromrRecognitionOptionsV1::deterministic())
}

/// Strip hallucinated leading attribute events (`clef-`/`keySignature-`/
/// `timeSignature-`) from a CONTINUATION segment's semantic stream: the
/// source engraving prints them only at the line start, so anything the
/// model emits at a mid-line segment boundary is an artifact of the cut
/// (bd-av64.4).
fn strip_leading_attrs(s: &str) -> &str {
    let mut rest = s;
    loop {
        let head = rest.split('+').next().unwrap_or("");
        if head.starts_with("clef-")
            || head.starts_with("keySignature-")
            || head.starts_with("timeSignature-")
        {
            rest = rest[head.len()..].trim_start_matches('+');
        } else {
            return rest;
        }
    }
}

fn append_split_music_result(
    semantic: &mut String,
    forward_candidate_lattices: &mut Vec<TromrForwardCandidateLatticeV1>,
    forward_input_index: usize,
    segment_result: FocrResult<MusicResult>,
) -> FocrResult<usize> {
    let mut segment = segment_result?;
    if forward_input_index != forward_candidate_lattices.len() {
        return Err(candidate_lattice_mismatch(format!(
            "split forward-input index {forward_input_index} is out of order after {} completed segments",
            forward_candidate_lattices.len()
        )));
    }
    segment.validate_forward_candidate_lattices(1)?;
    let segment_semantic_len = segment.semantic.len();
    let candidate = segment
        .forward_candidate_lattices
        .pop()
        .expect("validated single-forward result contains one lattice");
    let candidate =
        TromrForwardCandidateLatticeV1::new(forward_input_index, candidate.candidate_lattice)?;

    if semantic.is_empty() {
        *semantic = segment.semantic;
    } else {
        if !semantic.ends_with("barline") {
            semantic.push_str("+barline");
        }
        let continuation = strip_leading_attrs(&segment.semantic);
        if !continuation.is_empty() {
            semantic.push('+');
            semantic.push_str(continuation);
        }
    }
    forward_candidate_lattices.push(candidate);
    Ok(segment_semantic_len)
}

type SplitRecognitionResult =
    Result<Option<(MusicResult, Vec<TromrForwardInputV1>)>, (FocrError, Vec<TromrForwardInputV1>)>;

/// Recognize an over-budget staff band by splitting it at detected
/// barlines into segments that each fit the positional budget, running the
/// certified single-staff path per segment SEQUENTIALLY (doctrine #5), and
/// concatenating the semantic streams (bd-av64.4). Returns `Ok(None)` when
/// the band has no usable barlines to cut at — the caller falls through to
/// the normal path, whose clamp error becomes a per-staff skip.
///
/// Cuts land ON physical barlines, so a `barline` token is inserted at
/// each seam when the left segment did not already emit one; continuation
/// segments get their hallucinated leading clef/key/time stripped.
///
/// # Errors
/// A per-segment recognition failure (the whole staff then skips).
fn recognize_split_with_control<C: ExecutionControl + ?Sized>(
    weights: &Weights,
    tk: &crate::tokenizer::music::MusicTokenizer,
    crop: &crate::preprocess::staff_detect::StaffCrop,
    budget_px: usize,
    detection_index: usize,
    options: TromrRecognitionOptionsV1,
    control: &mut C,
) -> SplitRecognitionResult {
    let mut forward_inputs = Vec::new();
    let result = (|| -> FocrResult<Option<MusicResult>> {
        let options = options.validate()?;
        let options_identity = options.replay_identity()?;
        control.checkpoint("split-plan")?;
        let bars = crate::preprocess::staff_detect::barline_columns(crop);
        // Greedy plan: from each start, cut at the FARTHEST barline within
        // budget (a segment must also be meaningfully wide — at least one band
        // height — so degenerate cuts at the very start are ignored).
        let mut cuts = vec![0usize];
        let mut start = 0usize;
        while crop.w - start > budget_px {
            let limit = start + budget_px;
            let Some(&cut) = bars.iter().rfind(|&&b| b > start + crop.h && b <= limit) else {
                return Ok(None);
            };
            cuts.push(cut);
            start = cut;
        }
        cuts.push(crop.w);

        forward_inputs.reserve(cuts.len().saturating_sub(1));
        let mut semantic = String::new();
        let mut forward_candidate_lattices = Vec::with_capacity(cuts.len().saturating_sub(1));
        for (seg_idx, wnd) in cuts.windows(2).enumerate() {
            control.checkpoint("split-segment")?;
            control.set_attempt_location(detection_index, Some(seg_idx));
            let (a, b) = (wnd[0], wnd[1]);
            let mut seg = vec![0u8; crop.h * (b - a)];
            for row in 0..crop.h {
                seg[row * (b - a)..(row + 1) * (b - a)]
                    .copy_from_slice(&crop.gray[row * crop.w + a..row * crop.w + b]);
            }
            let gray8 = crate::preprocess::staff_detect::TromrGray8CropV1::from_tightly_packed(
                seg.clone(),
                b - a,
                crop.h,
            )?;
            forward_inputs.push(TromrForwardInputV1 {
                gray8,
                source_space: TromrModelInputSourceSpaceV1::ReviewCropCanvas,
                source_bbox_xywh: (a, 0, b - a, crop.h),
                padding: crop.padding,
                staff_lines_y_in_canvas: Some(crop.lines),
            });
            let buf = image::GrayImage::from_raw((b - a) as u32, crop.h as u32, seg).ok_or_else(
                || FocrError::Other(anyhow::anyhow!("tromr split: segment buffer mismatch")),
            )?;
            let t0 = std::time::Instant::now();
            let segment_semantic_len = append_split_music_result(
                &mut semantic,
                &mut forward_candidate_lattices,
                seg_idx,
                recognize_with_control(
                    weights,
                    tk,
                    &image::DynamicImage::ImageLuma8(buf),
                    options,
                    control,
                ),
            )?;
            super::timing_log(&format!(
                "    tromr.split seg {seg_idx} [{a}..{b}] {:.2}s ({segment_semantic_len} chars)",
                t0.elapsed().as_secs_f64(),
            ));
        }
        let musicxml = semantic_to_musicxml(&semantic)?;
        control.checkpoint("split-assembly")?;
        let result = MusicResult {
            semantic,
            musicxml,
            options,
            options_identity,
            forward_candidate_lattices,
        };
        result.validate_forward_candidate_lattices(forward_inputs.len())?;
        Ok(Some(result))
    })();
    match result {
        Ok(Some(music)) => Ok(Some((music, forward_inputs))),
        Ok(None) => Ok(None),
        Err(error) => Err((error, forward_inputs)),
    }
}

#[cfg(test)]
fn recognize_split(
    weights: &Weights,
    tk: &crate::tokenizer::music::MusicTokenizer,
    crop: &crate::preprocess::staff_detect::StaffCrop,
    budget_px: usize,
    options: TromrRecognitionOptionsV1,
) -> FocrResult<Option<MusicResult>> {
    recognize_split_with_control(
        weights,
        tk,
        crop,
        budget_px,
        0,
        options,
        &mut LegacyExecutionControl,
    )
    .map(|result| result.map(|(music, _)| music))
    .map_err(|(error, _)| error)
}

/// One staff the page path could NOT recognize: its detection index
/// (0-based, top-to-bottom over ALL detected staves), page-space bbox, and
/// the per-staff error text. The page as a whole still succeeds with the
/// staves that worked (bd-av64.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaffSkip {
    /// 0-based detection index over all detected staves, top-to-bottom.
    pub index: usize,
    /// Page-space bbox of the detected staff band.
    pub bbox: StaffBBox,
    /// The per-staff error, verbatim.
    pub reason: String,
}

/// Provider-owned explanation of how staff detection selected the TrOMR page
/// recognition route.
///
/// Zero and one detected staff both use whole-image recognition to preserve
/// the certified single-staff path. This enum keeps those cases distinct so
/// embedders never have to infer fallback status from a full-page bbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TromrStaffSegmentationDispositionV1 {
    /// Detection found no crop-eligible staff; recognition fell back to the
    /// whole input image.
    NoStaffDetectedWholeImageFallback,
    /// Detection found exactly one staff; the certified whole-image
    /// single-staff route recognized it without re-cropping.
    SingleStaffDetectedWholeImageRecognition,
    /// Detection found at least two staves; each detected crop was attempted
    /// independently in top-to-bottom order.
    MultipleStavesDetectedPerCropRecognition,
}

impl TromrStaffSegmentationDispositionV1 {
    #[must_use]
    pub const fn for_detected_staff_count(detected_staff_count: usize) -> Self {
        match detected_staff_count {
            0 => Self::NoStaffDetectedWholeImageFallback,
            1 => Self::SingleStaffDetectedWholeImageRecognition,
            _ => Self::MultipleStavesDetectedPerCropRecognition,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoStaffDetectedWholeImageFallback => "no_staff_detected_whole_image_fallback",
            Self::SingleStaffDetectedWholeImageRecognition => {
                "single_staff_detected_whole_image_recognition"
            }
            Self::MultipleStavesDetectedPerCropRecognition => {
                "multiple_staves_detected_per_crop_recognition"
            }
        }
    }

    #[must_use]
    pub const fn is_consistent_with(self, detected_staff_count: usize) -> bool {
        matches!(
            (self, detected_staff_count),
            (Self::NoStaffDetectedWholeImageFallback, 0)
                | (Self::SingleStaffDetectedWholeImageRecognition, 1)
                | (Self::MultipleStavesDetectedPerCropRecognition, 2..)
        )
    }
}

/// The full-page recognition draft: recognized staves (with their detection
/// index and bbox, top-to-bottom) plus skipped staves and detector residuals.
/// This is not publication authorization; call
/// [`Self::require_complete_for_publication`] before claiming page completeness.
pub struct PageRecognition {
    /// Exact crop-eligible staff count returned by detection before route
    /// selection, recognition, or skips.
    pub detected_staff_count: usize,
    /// Typed explanation of the route selected from `detected_staff_count`.
    pub staff_segmentation_disposition: TromrStaffSegmentationDispositionV1,
    /// `(route-local attempt index, result, bbox)` per recognized row. The
    /// index is detector-owned except for zero-detection fallback.
    pub staves: Vec<(usize, MusicResult, StaffBBox)>,
    /// Staves that failed per-staff recognition (empty on a clean page).
    pub skips: Vec<StaffSkip>,
    /// Complete ordered evidence for every row attempted.
    pub staff_evidence: Vec<StaffInferenceEvidence>,
    /// Pixel-free complete detector report, including candidate/residual
    /// accounting and every global/row-local Gray8 transform identity.
    pub staff_detection: crate::preprocess::staff_detect::StaffDetectionEvidenceV1,
    /// Provider-owned exact page/crop pixels and replayable detector geometry.
    ///
    /// This is intentionally separate from `staff_evidence`: selected-page
    /// and globally deskewed pixels are retained once per page, while
    /// [`Self::retained_geometry_for_staff_attempt`] maps an attempted row to
    /// its exact crop chain without cloning those page artifacts.
    pub retained_staff_detection:
        crate::preprocess::staff_detect::TromrRetainedStaffDetectionGeometryV1,
    /// Validated, normalized inference controls used for every row attempt.
    pub options: TromrRecognitionOptionsV1,
    /// SHA-256 of the canonical options JSON; suitable for replay/cache keys.
    pub options_identity: String,
}

impl PageRecognition {
    /// Whether retained staff-like residual evidence blocks a complete-page
    /// publication claim. Recognized rows remain available for explicit review.
    #[must_use]
    pub const fn publication_blocked(&self) -> bool {
        self.staff_detection.residual.unresolved
    }

    /// Replay the provider-owned detector transforms and prove that every
    /// recognition attempt addresses the same detector row and review pixels
    /// as the pixel-free ledger.
    pub fn validate_retained_geometry(&self) -> FocrResult<()> {
        self.staff_detection.validate()?;
        self.retained_staff_detection.validate()?;
        self.validate_retained_geometry_mapping()
    }

    fn validate_retained_geometry_mapping(&self) -> FocrResult<()> {
        crosscheck_retained_staff_detection_pair(
            self.detected_staff_count,
            &self.staff_detection,
            &self.retained_staff_detection,
        )?;
        if !self
            .staff_segmentation_disposition
            .is_consistent_with(self.detected_staff_count)
        {
            return Err(FocrError::FormatMismatch(
                "TrOMR page segmentation disposition differs from its detector census".into(),
            ));
        }

        let expected_attempt_count = self.detected_staff_count.max(1);
        if self.staff_evidence.len() != expected_attempt_count
            || self.staves.len() + self.skips.len() != expected_attempt_count
            || self.staves.windows(2).any(|pair| pair[0].0 >= pair[1].0)
            || self
                .skips
                .windows(2)
                .any(|pair| pair[0].index >= pair[1].index)
        {
            return Err(FocrError::FormatMismatch(
                "TrOMR page row-attempt or outcome census is not canonical".into(),
            ));
        }

        let selected = &self.retained_staff_detection.selected_page;
        let selected_gray8 = selected.gray8();
        let selected_geometry = crate::preprocess::staff_detect::StaffCropGeometry::unpadded((
            0,
            0,
            selected_gray8.width(),
            selected_gray8.height(),
        ));
        for (attempt_position, evidence) in self.staff_evidence.iter().enumerate() {
            let expected_index = if self.detected_staff_count == 0 {
                0
            } else {
                attempt_position
            };
            if evidence.index != expected_index {
                return Err(FocrError::FormatMismatch(
                    "TrOMR staff-attempt index differs from canonical detector order".into(),
                ));
            }

            let retained_crop = if self.detected_staff_count == 0 {
                None
            } else {
                self.retained_staff_detection.crops.get(evidence.index)
            };
            match retained_crop {
                None => {
                    if evidence.route
                        != TromrRowInferenceRouteV1::NoDetectedStaffWholeRasterFallback
                        || evidence.geometry != selected_geometry
                        || evidence.review_crop_gray8.is_some()
                        || evidence.review_crop_geometry.is_some()
                        || evidence.staff_lines.is_some()
                    {
                        return Err(FocrError::FormatMismatch(
                            "TrOMR zero-detection fallback carries detector-backed row evidence"
                                .into(),
                        ));
                    }
                }
                Some(retained) => {
                    let detector_crop =
                        self.staff_detection
                            .crops
                            .get(evidence.index)
                            .ok_or_else(|| {
                                FocrError::FormatMismatch(
                                    "TrOMR staff attempt has no pixel-free detector crop".into(),
                                )
                            })?;
                    let review_crop = evidence.review_crop_gray8.as_ref().ok_or_else(|| {
                        FocrError::FormatMismatch(
                            "TrOMR detector-backed attempt omitted its exact review crop".into(),
                        )
                    })?;
                    let expected_lines = TromrStaffLineEvidenceV1 {
                        accepted_detector_lines_y_in_globally_deskewed_raster: retained
                            .globally_deskewed_staff_lines
                            .y_rows,
                        review_crop_staff_lines_y_in_canvas: retained
                            .review_canvas_staff_lines
                            .y_rows,
                    };
                    if review_crop.artifact_identity() != retained.review_canvas.artifact_identity()
                        || evidence.review_crop_geometry != Some(detector_crop.geometry)
                        || evidence.staff_lines != Some(expected_lines)
                    {
                        return Err(FocrError::FormatMismatch(
                            "TrOMR row review pixels, geometry, or staff lines differ from retained detector geometry"
                                .into(),
                        ));
                    }
                }
            }

            match self.detected_staff_count {
                0 | 1 => {
                    let expected_route = if self.detected_staff_count == 0 {
                        TromrRowInferenceRouteV1::NoDetectedStaffWholeRasterFallback
                    } else {
                        TromrRowInferenceRouteV1::SingleDetectedStaffWholeRaster
                    };
                    let [forward_input] = evidence.forward_inputs.as_slice() else {
                        return Err(FocrError::FormatMismatch(
                            "TrOMR whole-raster route must retain exactly one forward input".into(),
                        ));
                    };
                    if evidence.route != expected_route
                        || evidence.geometry != selected_geometry
                        || forward_input.source_space
                            != TromrModelInputSourceSpaceV1::SelectedPageRaster
                        || forward_input.source_bbox_xywh
                            != (0, 0, selected_gray8.width(), selected_gray8.height())
                        || forward_input.padding
                            != crate::preprocess::staff_detect::StaffPadding::default()
                        || forward_input.staff_lines_y_in_canvas.is_some()
                        || &forward_input.gray8 != selected_gray8
                    {
                        return Err(FocrError::FormatMismatch(
                            "TrOMR whole-raster attempt differs from the exact retained selected page"
                                .into(),
                        ));
                    }
                }
                _ => {
                    let retained = retained_crop.ok_or_else(|| {
                        FocrError::FormatMismatch(
                            "TrOMR per-crop attempt has no retained detector crop".into(),
                        )
                    })?;
                    let detector_crop = &self.staff_detection.crops[evidence.index];
                    if evidence.geometry != detector_crop.geometry {
                        return Err(FocrError::FormatMismatch(
                            "TrOMR per-crop attempt geometry differs from detector geometry".into(),
                        ));
                    }
                    match evidence.route {
                        TromrRowInferenceRouteV1::DetectedStaffCrop => {
                            let [forward_input] = evidence.forward_inputs.as_slice() else {
                                return Err(FocrError::FormatMismatch(
                                    "TrOMR detected-crop route must retain exactly one forward input"
                                        .into(),
                                ));
                            };
                            if forward_input.source_space
                                != TromrModelInputSourceSpaceV1::ReviewCropCanvas
                                || forward_input.source_bbox_xywh
                                    != (
                                        0,
                                        0,
                                        retained.review_canvas.gray8().width(),
                                        retained.review_canvas.gray8().height(),
                                    )
                                || forward_input.padding != retained.padding_transform.padding
                                || forward_input.staff_lines_y_in_canvas
                                    != Some(retained.review_canvas_staff_lines.y_rows)
                                || forward_input.gray8.artifact_identity()
                                    != retained.review_canvas.artifact_identity()
                            {
                                return Err(FocrError::FormatMismatch(
                                    "TrOMR detected-crop forward input differs from retained review canvas"
                                        .into(),
                                ));
                            }
                        }
                        TromrRowInferenceRouteV1::ExperimentalSplitSegments => {
                            let review = retained.review_canvas.gray8();
                            if evidence.outcome == StaffInferenceOutcome::Recognized
                                && evidence.forward_inputs.is_empty()
                            {
                                return Err(FocrError::FormatMismatch(
                                    "TrOMR recognized split route retained no forward inputs"
                                        .into(),
                                ));
                            }
                            for forward_input in &evidence.forward_inputs {
                                let (x, y, width, height) = forward_input.source_bbox_xywh;
                                let in_bounds = x
                                    .checked_add(width)
                                    .is_some_and(|right| right <= review.width())
                                    && y.checked_add(height)
                                        .is_some_and(|bottom| bottom <= review.height());
                                if forward_input.source_space
                                    != TromrModelInputSourceSpaceV1::ReviewCropCanvas
                                    || y != 0
                                    || height != review.height()
                                    || width != forward_input.gray8.width()
                                    || height != forward_input.gray8.height()
                                    || !in_bounds
                                    || forward_input.staff_lines_y_in_canvas
                                        != Some(retained.review_canvas_staff_lines.y_rows)
                                {
                                    return Err(FocrError::FormatMismatch(
                                        "TrOMR split forward input lies outside its retained review canvas"
                                        .into(),
                                    ));
                                }
                                for row in 0..height {
                                    let source_start = (y + row) * review.width() + x;
                                    let forwarded_start = row * width;
                                    if review.pixels()[source_start..source_start + width]
                                        != forward_input.gray8.pixels()
                                            [forwarded_start..forwarded_start + width]
                                    {
                                        return Err(FocrError::FormatMismatch(
                                            "TrOMR split forward pixels differ from their retained review-canvas rectangle"
                                                .into(),
                                        ));
                                    }
                                }
                            }
                        }
                        TromrRowInferenceRouteV1::NoDetectedStaffWholeRasterFallback
                        | TromrRowInferenceRouteV1::SingleDetectedStaffWholeRaster => {
                            return Err(FocrError::FormatMismatch(
                                "TrOMR multi-staff page carries a whole-raster row route".into(),
                            ));
                        }
                    }
                }
            }

            let expected_bbox = retained_crop.map_or(
                (0, 0, selected_gray8.width(), selected_gray8.height()),
                |retained| retained.crop_transform.source_rect.as_bbox(),
            );
            let recognized = self
                .staves
                .iter()
                .filter(|(index, _, bbox)| *index == evidence.index && *bbox == expected_bbox)
                .count();
            let skipped = self
                .skips
                .iter()
                .filter(|skip| skip.index == evidence.index && skip.bbox == expected_bbox)
                .count();
            let outcome_is_consistent = match evidence.outcome {
                StaffInferenceOutcome::Recognized => {
                    recognized == 1 && skipped == 0 && evidence.reason.is_none()
                }
                StaffInferenceOutcome::Skipped => {
                    recognized == 0
                        && skipped == 1
                        && evidence.reason.as_ref()
                            == self
                                .skips
                                .iter()
                                .find(|skip| skip.index == evidence.index)
                                .map(|skip| &skip.reason)
                }
            };
            if !outcome_is_consistent {
                return Err(FocrError::FormatMismatch(
                    "TrOMR staff attempt differs from its recognized/skipped outcome census".into(),
                ));
            }
        }
        Ok(())
    }

    /// Return the exact retained detector crop for one entry in
    /// [`Self::staff_evidence`].
    ///
    /// `attempt_position` addresses the ordered attempt ledger, not an
    /// unvalidated caller-supplied detector index. The zero-detection
    /// whole-page fallback returns `Ok(None)` because no crop was detected.
    pub fn retained_geometry_for_staff_attempt(
        &self,
        attempt_position: usize,
    ) -> FocrResult<Option<&crate::preprocess::staff_detect::TromrRetainedStaffCropGeometryV1>>
    {
        self.validate_retained_geometry()?;
        let evidence = self.staff_evidence.get(attempt_position).ok_or_else(|| {
            FocrError::FormatMismatch(format!(
                "TrOMR staff attempt position {attempt_position} is outside the retained ledger"
            ))
        })?;
        if self.detected_staff_count == 0 {
            Ok(None)
        } else {
            self.retained_staff_detection
                .crops
                .get(evidence.index)
                .map(Some)
                .ok_or_else(|| {
                    FocrError::FormatMismatch(
                        "TrOMR staff attempt index has no retained detector crop".into(),
                    )
                })
        }
    }

    /// Revalidate the detector ledger and refuse a complete-page publication
    /// while any staff-like residual remains unresolved.
    pub fn require_complete_for_publication(&self) -> FocrResult<()> {
        self.validate_retained_geometry()?;
        self.staff_detection.require_complete()
    }
}

fn crosscheck_retained_staff_detection_pair(
    detected_staff_count: usize,
    staff_detection: &crate::preprocess::staff_detect::StaffDetectionEvidenceV1,
    retained: &crate::preprocess::staff_detect::TromrRetainedStaffDetectionGeometryV1,
) -> FocrResult<()> {
    if staff_detection.crops.len() != detected_staff_count
        || retained.crops.len() != detected_staff_count
        || staff_detection.global_deskew.input_gray8 != retained.selected_page.artifact_identity()
        || staff_detection.global_deskew.globally_deskewed_gray8
            != retained.globally_deskewed_page.artifact_identity()
        || staff_detection.global_deskew.transform_contract
            != retained.global_deskew_transform.transform_contract
        || staff_detection.global_deskew.angle_millidegrees
            != retained.global_deskew_transform.angle_millidegrees
    {
        return Err(FocrError::FormatMismatch(
            "TrOMR pixel-free and retained detector page census or global deskew differ".into(),
        ));
    }
    for (pixel_free, owned) in staff_detection.crops.iter().zip(&retained.crops) {
        let owned_geometry = crate::preprocess::staff_detect::StaffCropGeometry {
            source_bbox: owned.crop_transform.source_rect.as_bbox(),
            canvas_width: owned.review_canvas.gray8().width(),
            canvas_height: owned.review_canvas.gray8().height(),
            padding: owned.padding_transform.padding,
        };
        if pixel_free.geometry != owned_geometry
            || pixel_free.globally_deskewed_raster_lines
                != owned.globally_deskewed_staff_lines.y_rows
            || pixel_free.review_crop_staff_lines_y_in_canvas
                != owned.review_canvas_staff_lines.y_rows
            || pixel_free
                .row_refinement
                .source_crop_before_refinement_gray8
                != owned.pre_refinement_crop.artifact_identity()
            || pixel_free.row_refinement.refined_unpadded_crop_gray8
                != owned.refined_unpadded_crop.artifact_identity()
            || pixel_free.row_refinement.transform_contract
                != owned.row_refinement_transform.transform_contract
            || pixel_free.row_refinement.angle_millidegrees
                != owned.row_refinement_transform.angle_millidegrees
            || pixel_free.review_crop_gray8 != owned.review_canvas.artifact_identity()
        {
            return Err(FocrError::FormatMismatch(
                "TrOMR pixel-free crop evidence differs from retained detector geometry".into(),
            ));
        }
    }
    Ok(())
}

fn record_staff_outcome(
    bbox: StaffBBox,
    mut evidence: StaffInferenceEvidence,
    outcome: FocrResult<MusicResult>,
    staves: &mut Vec<(usize, MusicResult, StaffBBox)>,
    skips: &mut Vec<StaffSkip>,
    staff_evidence: &mut Vec<StaffInferenceEvidence>,
) -> FocrResult<()> {
    match outcome {
        Ok(result) => {
            result.validate_forward_candidate_lattices(evidence.forward_inputs.len())?;
            evidence.outcome = StaffInferenceOutcome::Recognized;
            evidence.reason = None;
            staves.push((evidence.index, result, bbox));
            staff_evidence.push(evidence);
        }
        Err(error) if is_terminal_execution_error(&error) => return Err(error),
        Err(error) => {
            let reason = error.to_string();
            evidence.outcome = StaffInferenceOutcome::Skipped;
            evidence.reason = Some(reason.clone());
            skips.push(StaffSkip {
                index: evidence.index,
                bbox,
                reason,
            });
            staff_evidence.push(evidence);
        }
    }
    Ok(())
}

struct PageStaffDetectionArtifacts {
    pixel_free: crate::preprocess::staff_detect::StaffDetectionEvidenceV1,
    retained: crate::preprocess::staff_detect::TromrRetainedStaffDetectionGeometryV1,
}

fn finish_page_recognition(
    detected_staff_count: usize,
    staves: Vec<(usize, MusicResult, StaffBBox)>,
    skips: Vec<StaffSkip>,
    staff_evidence: Vec<StaffInferenceEvidence>,
    staff_detection: PageStaffDetectionArtifacts,
    options: TromrRecognitionOptionsV1,
    options_identity: String,
) -> FocrResult<PageRecognition> {
    let PageStaffDetectionArtifacts {
        pixel_free: staff_detection,
        retained: retained_staff_detection,
    } = staff_detection;
    if detected_staff_count < 2 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr page: per-crop assembly requires at least 2 detected staves, got {detected_staff_count}"
        )));
    }
    if staves.len() + skips.len() != detected_staff_count {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr page: {} recognized plus {} skipped rows do not account for all {detected_staff_count} detected staves",
            staves.len(),
            skips.len()
        )));
    }
    if staff_evidence.len() != detected_staff_count {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr page: {} row-attempt evidence entries do not account for all {detected_staff_count} detected staves",
            staff_evidence.len()
        )));
    }
    staff_detection.validate()?;
    if staff_detection.crops.len() != detected_staff_count {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr page: detector evidence has {} crops, expected {detected_staff_count}",
            staff_detection.crops.len()
        )));
    }
    if staves.is_empty() {
        let reasons: Vec<String> = skips
            .iter()
            .map(|s| format!("staff {}: {}", s.index, s.reason))
            .collect();
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr page: all {} detected staves failed -- {}",
            skips.len(),
            reasons.join("; ")
        )));
    }
    let page = PageRecognition {
        detected_staff_count,
        staff_segmentation_disposition:
            TromrStaffSegmentationDispositionV1::MultipleStavesDetectedPerCropRecognition,
        staves,
        skips,
        staff_evidence,
        staff_detection,
        retained_staff_detection,
        options,
        options_identity,
    };
    page.validate_retained_geometry_mapping()?;
    Ok(page)
}

/// The E5 full-page pipeline: staff detection → per-staff [`recognize`]
/// (SEQUENTIAL, doctrine #5) → [`PageRecognition`].
///
/// Contract: 0 or 1 detected staves ⇒ the image uses whole-image recognition
/// (preserves the certified
/// single-staff path exactly — detection adds nothing there, and its error
/// IS the page error; bd-av64.13 MEASURED the alternative 2026-07-07 and
/// the gate said no — routing a 1-crop page through the refined band
/// regressed spohr_no17_top, dropping its time signature and flipping a
/// note, because band extraction re-trims pixels the knife-edge-sensitive
/// decode needed. The sub-degree-skew exposure on this route stays
/// documented in bd-av64.13/.15); ≥ 2 staves ⇒ the per-crop path,
/// top-to-bottom, where
/// ONE bad crop must never abort the page (bd-av64.2: a real book page with
/// one over-wide staff band previously died whole via `?`-propagation —
/// Cadwallader p169, 2026-07-06). A failed staff becomes a [`StaffSkip`];
/// the page errors only when EVERY detected staff fails, and that error
/// names each staff's reason.
///
/// # Errors
/// A detection failure, a whole-image route failure, or all-staves
/// failure on the per-crop path.
fn recognize_page_with_control<C: ExecutionControl + ?Sized>(
    weights: &Weights,
    tk: &crate::tokenizer::music::MusicTokenizer,
    img: &image::DynamicImage,
    options: TromrRecognitionOptionsV1,
    control: &mut C,
) -> FocrResult<PageRecognition> {
    let options = options.validate()?;
    let options_identity = options.replay_identity()?;
    super::timing_log(&format!(
        "  tromr.options identity={options_identity} value={}",
        options.canonical_json()?
    ));
    control.checkpoint("staff-detection")?;
    let detection_started = std::time::Instant::now();
    let detected = crate::preprocess::staff_detect::detect_staves_with_evidence(img);
    control.record_staff_detection(detection_started.elapsed());
    let report = detected?;
    let staff_detection = report.evidence()?;
    staff_detection.validate()?;
    let retained_staff_detection = report.retained_geometry;
    let crops = report.crops;
    control.checkpoint("staff-detection")?;
    let detected_staff_count = crops.len();
    let staff_segmentation_disposition =
        TromrStaffSegmentationDispositionV1::for_detected_staff_count(detected_staff_count);
    super::timing_log(&format!(
        "  tromr.staff_detect count={detected_staff_count} disposition={}",
        staff_segmentation_disposition.as_str()
    ));
    if detected_staff_count < 2 {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let model_input_gray8 = crate::preprocess::tromr_gray8_input(img)?;
        let detected_crop = crops.first();
        let review_crop_gray8 = detected_crop.map(|crop| crop.exact_gray8()).transpose()?;
        let review_crop_geometry = detected_crop.map(|crop| crop.geometry());
        let staff_lines = detected_crop.map(|crop| TromrStaffLineEvidenceV1 {
            accepted_detector_lines_y_in_globally_deskewed_raster: crop
                .globally_deskewed_raster_lines,
            review_crop_staff_lines_y_in_canvas: crop.lines,
        });
        let row_bbox = detected_crop.map_or((0, 0, w, h), |crop| crop.bbox);
        let route = if detected_crop.is_some() {
            TromrRowInferenceRouteV1::SingleDetectedStaffWholeRaster
        } else {
            TromrRowInferenceRouteV1::NoDetectedStaffWholeRasterFallback
        };
        control.set_attempt_location(0, None);
        let res = recognize_with_control(weights, tk, img, options, control)?;
        control.checkpoint("page-assembly")?;
        let assembly_started = std::time::Instant::now();
        let page = PageRecognition {
            detected_staff_count,
            staff_segmentation_disposition,
            staves: vec![(0, res, row_bbox)],
            skips: Vec::new(),
            staff_evidence: vec![StaffInferenceEvidence {
                index: 0,
                geometry: crate::preprocess::staff_detect::StaffCropGeometry::unpadded((
                    0, 0, w, h,
                )),
                route,
                forward_inputs: vec![TromrForwardInputV1 {
                    gray8: model_input_gray8,
                    source_space: TromrModelInputSourceSpaceV1::SelectedPageRaster,
                    source_bbox_xywh: (0, 0, w, h),
                    padding: crate::preprocess::staff_detect::StaffPadding::default(),
                    staff_lines_y_in_canvas: None,
                }],
                review_crop_gray8,
                review_crop_geometry,
                staff_lines,
                outcome: StaffInferenceOutcome::Recognized,
                reason: None,
            }],
            staff_detection,
            retained_staff_detection,
            options,
            options_identity,
        };
        page.validate_retained_geometry_mapping()?;
        control.record_page_assembly(assembly_started.elapsed());
        return Ok(page);
    }
    let mut staves = Vec::with_capacity(detected_staff_count);
    let mut skips = Vec::new();
    let mut staff_evidence = Vec::with_capacity(detected_staff_count);
    for (index, crop) in crops.into_iter().enumerate() {
        control.checkpoint("staff-boundary")?;
        control.set_attempt_location(index, None);
        let geometry = crop.geometry();
        let (cw, ch, bbox) = (crop.w, crop.h, crop.bbox);
        let review_crop_gray8 = crop.exact_gray8()?;
        let staff_lines = TromrStaffLineEvidenceV1 {
            accepted_detector_lines_y_in_globally_deskewed_raster: crop
                .globally_deskewed_raster_lines,
            review_crop_staff_lines_y_in_canvas: crop.lines,
        };
        // Over-budget bands (which the geometry pass could not fit,
        // bd-av64.14) can try barline splitting (bd-av64.4) — EXPERIMENTAL
        // and off by default: measured 2026-07-07, isolated segments are
        // out-of-distribution for the model (continuations lose absolute
        // pitch registration; rhythm agreement 0.2 vs the whole-staff read;
        // a pixel-space clef prepend measured WORSE). The explicit experimental
        // split policy arms it for recognition-count rescue where a skip is
        // worse than approximate content.
        let split_armed = matches!(
            options.split_policy,
            TromrSplitPolicyV1::ExperimentalBarlineSegments
        );
        let over_budget = split_armed && IMG_H * cw > POS_COLS * PATCH * ch;
        let (outcome, route, forward_inputs) = if over_budget {
            let budget_px = POS_COLS * PATCH * ch / IMG_H;
            match recognize_split_with_control(
                weights, tk, &crop, budget_px, index, options, control,
            ) {
                Ok(Some((res, inputs))) => {
                    super::timing_log(&format!(
                        "  tromr.staff {index} split-recognized ({cw}x{ch})"
                    ));
                    (
                        Ok(res),
                        TromrRowInferenceRouteV1::ExperimentalSplitSegments,
                        inputs,
                    )
                }
                Ok(None) => (
                    Err(FocrError::Other(anyhow::anyhow!(
                        "band resizes past the {} position budget and no usable \
                         barlines were found to split at ({cw}x{ch})",
                        POS_COLS * PATCH
                    ))),
                    TromrRowInferenceRouteV1::ExperimentalSplitSegments,
                    Vec::new(),
                ),
                Err((e, inputs)) => (
                    Err(e),
                    TromrRowInferenceRouteV1::ExperimentalSplitSegments,
                    inputs,
                ),
            }
        } else {
            let input = TromrForwardInputV1 {
                gray8: review_crop_gray8.clone(),
                source_space: TromrModelInputSourceSpaceV1::ReviewCropCanvas,
                source_bbox_xywh: (0, 0, cw, ch),
                padding: crop.padding,
                staff_lines_y_in_canvas: Some(crop.lines),
            };
            let result = image::GrayImage::from_raw(cw as u32, ch as u32, crop.gray)
                .ok_or_else(|| {
                    FocrError::Other(anyhow::anyhow!("tromr page: crop buffer shape mismatch"))
                })
                .and_then(|buf| {
                    recognize_with_control(
                        weights,
                        tk,
                        &image::DynamicImage::ImageLuma8(buf),
                        options,
                        control,
                    )
                });
            (
                result,
                TromrRowInferenceRouteV1::DetectedStaffCrop,
                vec![input],
            )
        };
        match &outcome {
            Ok(res) => {
                super::timing_log(&format!(
                    "  tromr.staff {index} ok ({cw}x{ch}, semantic {} chars)",
                    res.semantic.len()
                ));
            }
            Err(e) if !is_terminal_execution_error(e) => {
                super::timing_log(&format!("  tromr.staff {index} SKIP ({cw}x{ch}): {e}"));
            }
            Err(_) => {}
        }
        record_staff_outcome(
            bbox,
            StaffInferenceEvidence {
                index,
                geometry,
                route,
                forward_inputs,
                review_crop_gray8: Some(review_crop_gray8),
                review_crop_geometry: Some(geometry),
                staff_lines: Some(staff_lines),
                outcome: StaffInferenceOutcome::Skipped,
                reason: None,
            },
            outcome,
            &mut staves,
            &mut skips,
            &mut staff_evidence,
        )?;
    }
    control.checkpoint("page-assembly")?;
    let assembly_started = std::time::Instant::now();
    let page = finish_page_recognition(
        detected_staff_count,
        staves,
        skips,
        staff_evidence,
        PageStaffDetectionArtifacts {
            pixel_free: staff_detection,
            retained: retained_staff_detection,
        },
        options,
        options_identity,
    );
    control.record_page_assembly(assembly_started.elapsed());
    page
}

/// Recognize one page with explicit recognition mechanics and the legacy
/// process-wide cancellation bridge.
///
/// # Errors
/// A detection failure, cancellation, a whole-image route failure, or
/// all-staves failure on the per-crop path.
pub fn recognize_page_with_options(
    weights: &Weights,
    tk: &crate::tokenizer::music::MusicTokenizer,
    img: &image::DynamicImage,
    options: TromrRecognitionOptionsV1,
) -> FocrResult<PageRecognition> {
    recognize_page_with_control(weights, tk, img, options, &mut LegacyExecutionControl)
}

pub(crate) fn recognize_page_with_execution_context(
    weights: &Weights,
    tk: &crate::tokenizer::music::MusicTokenizer,
    img: &image::DynamicImage,
    options: TromrRecognitionOptionsV1,
    execution: &mut TromrExecutionContext,
) -> FocrResult<PageRecognition> {
    recognize_page_with_control(weights, tk, img, options, execution)
}

/// Recognize a page with [`TromrRecognitionOptionsV1::deterministic`].
///
/// # Errors
/// A detection failure, the single-staff fallback's failure, or all-staves
/// failure on the per-crop path.
pub fn recognize_page(
    weights: &Weights,
    tk: &crate::tokenizer::music::MusicTokenizer,
    img: &image::DynamicImage,
) -> FocrResult<PageRecognition> {
    recognize_page_with_options(weights, tk, img, TromrRecognitionOptionsV1::deterministic())
}

#[cfg(test)]
pub(crate) fn synthetic_staff_detection_pair_for_test(
    raster_width: usize,
    raster_height: usize,
    crops: &[(
        crate::preprocess::staff_detect::StaffCropGeometry,
        [usize; 5],
        [usize; 5],
    )],
) -> (
    crate::preprocess::staff_detect::StaffDetectionEvidenceV1,
    crate::preprocess::staff_detect::TromrRetainedStaffDetectionGeometryV1,
) {
    use crate::preprocess::staff_detect::{
        TROMR_RETAINED_STAFF_GEOMETRY_SCHEMA_V1, TromrCropTransformV1,
        TromrGeometryCoordinateSpaceV1, TromrGray8CropV1, TromrGray8StageV1,
        TromrPaddingTransformV1, TromrPixelRectV1, TromrRetainedStaffCropGeometryV1,
        TromrRetainedStaffDetectionGeometryV1, TromrStaffLineRowsV1, TromrVerticalShearTransformV1,
    };

    let staff_detection = crate::preprocess::staff_detect::synthetic_complete_evidence_for_test(
        raster_width,
        raster_height,
        crops,
    );
    let selected_page = TromrGray8StageV1::from_gray8(
        TromrGeometryCoordinateSpaceV1::SelectedPage,
        TromrGray8CropV1::from_tightly_packed(
            vec![255; raster_width * raster_height],
            raster_width,
            raster_height,
        )
        .expect("synthetic selected-page pixels"),
    )
    .expect("synthetic selected-page stage");
    let global_deskew_transform = TromrVerticalShearTransformV1::global_deskew(0);
    let globally_deskewed_page = global_deskew_transform
        .apply(&selected_page)
        .expect("synthetic global deskew");
    let retained_crops = crops
        .iter()
        .enumerate()
        .map(
            |(crop_index, (geometry, globally_deskewed_lines, review_lines))| {
                let crop_transform = TromrCropTransformV1::staff_from_globally_deskewed_page(
                    TromrPixelRectV1::from_bbox(geometry.source_bbox),
                );
                let pre_refinement_crop = crop_transform
                    .apply(&globally_deskewed_page)
                    .expect("synthetic exact crop");
                let pre_refinement_lines =
                    globally_deskewed_lines.map(|row| row - geometry.source_bbox.1);
                let row_refinement_transform = TromrVerticalShearTransformV1::row_refinement(0);
                let refined_unpadded_crop = row_refinement_transform
                    .apply(&pre_refinement_crop)
                    .expect("synthetic row refinement");
                let padding_transform = TromrPaddingTransformV1::review_canvas(geometry.padding);
                let review_canvas = padding_transform
                    .apply(&refined_unpadded_crop)
                    .expect("synthetic review canvas");
                assert_eq!(
                    (
                        review_canvas.gray8().width(),
                        review_canvas.gray8().height()
                    ),
                    (geometry.canvas_width, geometry.canvas_height)
                );
                assert_eq!(
                    *review_lines,
                    pre_refinement_lines.map(|row| row + geometry.padding.top)
                );
                TromrRetainedStaffCropGeometryV1 {
                    crop_index,
                    globally_deskewed_staff_lines: TromrStaffLineRowsV1 {
                        coordinate_space: TromrGeometryCoordinateSpaceV1::GloballyDeskewedPage,
                        y_rows: *globally_deskewed_lines,
                    },
                    crop_transform,
                    pre_refinement_crop,
                    pre_refinement_staff_lines: TromrStaffLineRowsV1 {
                        coordinate_space: TromrGeometryCoordinateSpaceV1::PreRefinementCrop,
                        y_rows: pre_refinement_lines,
                    },
                    row_refinement_transform,
                    refined_unpadded_crop,
                    refined_unpadded_staff_lines: TromrStaffLineRowsV1 {
                        coordinate_space: TromrGeometryCoordinateSpaceV1::RefinedUnpaddedCrop,
                        y_rows: pre_refinement_lines,
                    },
                    padding_transform,
                    review_canvas,
                    review_canvas_staff_lines: TromrStaffLineRowsV1 {
                        coordinate_space: TromrGeometryCoordinateSpaceV1::ReviewCanvas,
                        y_rows: *review_lines,
                    },
                }
            },
        )
        .collect();
    let retained = TromrRetainedStaffDetectionGeometryV1 {
        schema_version: TROMR_RETAINED_STAFF_GEOMETRY_SCHEMA_V1,
        selected_page,
        global_deskew_transform,
        globally_deskewed_page,
        crops: retained_crops,
    };
    staff_detection
        .validate()
        .expect("synthetic pixel-free detector evidence");
    retained
        .validate()
        .expect("synthetic retained detector geometry");
    crosscheck_retained_staff_detection_pair(crops.len(), &staff_detection, &retained)
        .expect("synthetic retained detector pair crosscheck");
    (staff_detection, retained)
}

#[cfg(test)]
fn synthetic_candidate_head_for_test(
    head: TromrCandidateHeadV1,
    chosen_token_id: u32,
) -> TromrCandidateHeadEvidenceV1 {
    let mut logits = (0..head.vocabulary_size())
        .map(|token_id| -(token_id as f32))
        .collect::<Vec<_>>();
    logits[chosen_token_id as usize] = 100.0;
    let ranked = ranked_candidate_ids(&logits).expect("synthetic scores rank");
    capture_candidate_head_evidence(head, &logits, chosen_token_id, &ranked)
        .expect("synthetic evidence captures")
}

#[cfg(test)]
pub(crate) fn synthetic_candidate_lattice_for_test(
    position_count: usize,
    terminal_eos: bool,
) -> TromrCandidateLatticeV1 {
    assert!((1..=MAX_SEQ).contains(&position_count));
    let mut chosen_streams = MusicStreams {
        rhythm: Vec::with_capacity(position_count),
        pitch: Vec::with_capacity(position_count),
        lift: Vec::with_capacity(position_count),
    };
    let mut prefix_rhythm = vec![SEED_RHYTHM];
    let mut prefix_pitch = vec![SEED_NONOTE];
    let mut prefix_lift = vec![SEED_NONOTE];
    let mut positions = Vec::with_capacity(position_count);
    for position_zero_based in 0..position_count {
        let start = prefix_rhythm.len().saturating_sub(MAX_SEQ);
        let rhythm_id = if terminal_eos && position_zero_based + 1 == position_count {
            crate::tokenizer::music::EOS_ID
        } else {
            3
        };
        let pitch_id = 4;
        let lift_id = 1;
        positions.push(TromrCandidatePositionV1 {
            position_zero_based: position_zero_based as u32,
            prefix_length: prefix_rhythm.len() as u32,
            prefix_window_start: start as u32,
            prefix_sha256: candidate_prefix_sha256(
                &prefix_rhythm,
                &prefix_pitch,
                &prefix_lift,
                start,
            )
            .expect("synthetic prefix hashes"),
            heads: [
                synthetic_candidate_head_for_test(TromrCandidateHeadV1::Rhythm, rhythm_id),
                synthetic_candidate_head_for_test(TromrCandidateHeadV1::Pitch, pitch_id),
                synthetic_candidate_head_for_test(TromrCandidateHeadV1::Lift, lift_id),
            ],
            rhythm_emitted_eos: rhythm_id == crate::tokenizer::music::EOS_ID,
        });
        chosen_streams.rhythm.push(rhythm_id);
        chosen_streams.pitch.push(pitch_id);
        chosen_streams.lift.push(lift_id);
        prefix_rhythm.push(rhythm_id);
        prefix_pitch.push(pitch_id);
        prefix_lift.push(lift_id);
    }
    TromrCandidateLatticeV1 {
        schema_version: TROMR_CANDIDATE_LATTICE_SCHEMA_VERSION,
        contract_id: TROMR_CANDIDATE_LATTICE_CONTRACT_ID.to_owned(),
        canonical_encoding: TROMR_CANDIDATE_LATTICE_CANONICAL_ENCODING.to_owned(),
        top_n: TROMR_CANDIDATE_TOP_N as u32,
        chosen_streams,
        positions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RETAINED_ROW_PNG_SHA256: &str =
        "c125146a8a3966ddcaf2979b5323e4bb8f98a4c453de8b2652e9ba93b059d86b";
    const RETAINED_ROW_GRAY8_SHA256: &str =
        "d9251c23de5f929d51547350f20f5e7c07dd466e641554c9617b0f642d0e491e";
    const RETAINED_ROW_PRE_REFINEMENT_GRAY8_SHA256: &str =
        "8aaf7ceb91b4184fe9d92df23bc9f9fa239544179a3ba72cb6612d1e7fbdac32";
    const RETAINED_ROW_REFINED_UNPADDED_GRAY8_SHA256: &str =
        "a2943bed7493d8dc7f6528efa0ee9a0aacda04813c0ec021245b953d51743490";
    const SPOHR_SOURCE_PDF_SHA256: &str =
        "9b6b4a84400932cf5ce93bbcdc87a7041809d35ed7fecdbea9a6ebe3c8e21dac";
    const TROMR_UPSTREAM_CHECKPOINT_SHA256: &str =
        "02925259ef59f5578a8c9e954ac363bb15538ea38ce73090b861c1519179f910";
    const TROMR_F32_SHA256: &str =
        "a9d41485a98534ad0a1f7c1ec624f0a92f3f092c7dc30ac5af636b50dc465edc";
    const TROMR_INT8_SHA256: &str =
        "cced11c0f05656dd54cc615a15939c472dc8f916f04ae154ea4a0364839f845a";
    const TROMR_TOKENIZER_SHA256: [&str; 4] = [
        "603bfef760e8424f7808acba423532b4beb2d88dbf085f81add6a8e543a34035",
        "2382e8b20c1473290e200789604656b3a06bdf4b55a0818a0f7d175e8cb64ade",
        "b61ba09cecd5bc343e6a038a2e26718b54cd3c08e8f9b72013ecf80c3cac86b2",
        "504d886d11e3c1fe92893abd46edfc68dfbe7a8eb83e6b51646532dad8a485e1",
    ];

    fn diagnostic_sha256(bytes: &[u8]) -> String {
        use sha2::Digest as _;
        format!("{:x}", sha2::Sha256::digest(bytes))
    }

    fn diagnostic_hex32(bytes: &[u8; 32]) -> String {
        use std::fmt::Write as _;
        let mut encoded = String::with_capacity(64);
        for byte in bytes {
            write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
        }
        encoded
    }

    fn diagnostic_f32_le_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn diagnostic_shear_gray_millidegrees(
        gray: &[u8],
        width: usize,
        height: usize,
        millidegrees: i32,
    ) -> Vec<u8> {
        if millidegrees == 0 {
            return gray.to_vec();
        }
        assert_eq!(gray.len(), width * height);
        let tangent = (f64::from(millidegrees) / 1_000.0).to_radians().tan();
        let mut sheared = vec![255u8; gray.len()];
        for y in 0..height {
            for x in 0..width {
                let shift = (tangent * x as f64).round() as isize;
                let new_y = y as isize - shift;
                if new_y >= 0 && (new_y as usize) < height {
                    sheared[new_y as usize * width + x] = gray[y * width + x];
                }
            }
        }
        sheared
    }

    struct DiagnosticRetainedRowInput {
        variant: &'static str,
        crop: crate::preprocess::staff_detect::TromrGray8CropV1,
        png: Vec<u8>,
        row_refinement_angle_millidegrees: i32,
        source_crop_before_refinement_sha256: String,
        refined_unpadded_crop_sha256: String,
    }

    fn diagnostic_png_bytes(evidence: &serde_json::Value) -> Vec<u8> {
        evidence
            .pointer("/selected_row/model_input_png/bytes")
            .and_then(serde_json::Value::as_array)
            .expect("retained evidence has model-input PNG bytes")
            .iter()
            .map(|value| {
                u8::try_from(value.as_u64().expect("PNG byte is unsigned"))
                    .expect("PNG byte is within u8")
            })
            .collect()
    }

    fn diagnostic_refined_input(evidence: &serde_json::Value) -> DiagnosticRetainedRowInput {
        let png = diagnostic_png_bytes(evidence);
        assert_eq!(png.len(), 117_629, "retained PNG byte length");
        assert_eq!(diagnostic_sha256(&png), RETAINED_ROW_PNG_SHA256);
        let crop = crate::preprocess::staff_detect::TromrGray8CropV1::from_lossless_png(&png)
            .expect("provider-produced L8 PNG decodes without conversion");
        assert_eq!((crop.width(), crop.height()), (2_398, 240));
        assert_eq!(diagnostic_sha256(crop.pixels()), RETAINED_ROW_GRAY8_SHA256);
        DiagnosticRetainedRowInput {
            variant: "row_refined_minus_200_millidegrees",
            crop,
            png,
            row_refinement_angle_millidegrees: -200,
            source_crop_before_refinement_sha256: RETAINED_ROW_PRE_REFINEMENT_GRAY8_SHA256
                .to_owned(),
            refined_unpadded_crop_sha256: RETAINED_ROW_REFINED_UNPADDED_GRAY8_SHA256.to_owned(),
        }
    }

    fn diagnostic_pre_refinement_input(
        evidence: &serde_json::Value,
        source_pdf: &std::path::Path,
    ) -> DiagnosticRetainedRowInput {
        let source_bytes = std::fs::read(source_pdf).expect("read exact Spohr source PDF");
        assert_eq!(diagnostic_sha256(&source_bytes), SPOHR_SOURCE_PDF_SHA256);
        let pages = crate::pdf::PdfPages::from_bytes(source_bytes).expect("parse exact Spohr PDF");
        assert_eq!(pages.len(), 266);
        let page = pages
            .render(54)
            .expect("render exact Spohr page 55 natively");
        assert_eq!((page.width(), page.height()), (2_696, 3_926));
        let report = crate::preprocess::staff_detect::detect_staves_with_evidence(&page)
            .expect("detect exact Spohr staves");
        assert_eq!(report.crops.len(), 12);
        assert_eq!(report.global_deskew.angle_millidegrees, 250);
        let refinement = report.crop_refinements[2];
        assert_eq!(refinement.angle_millidegrees, -200);
        assert_eq!(
            diagnostic_hex32(&refinement.source_crop_before_refinement_gray8.pixels_sha256),
            RETAINED_ROW_PRE_REFINEMENT_GRAY8_SHA256
        );
        assert_eq!(
            diagnostic_hex32(&refinement.refined_unpadded_crop_gray8.pixels_sha256),
            RETAINED_ROW_REFINED_UNPADDED_GRAY8_SHA256
        );
        let refined = &report.crops[2];
        assert_eq!(refined.bbox, (58, 1_008, 2_398, 226));
        assert_eq!(
            refined.padding,
            crate::preprocess::staff_detect::StaffPadding {
                top: 7,
                right: 0,
                bottom: 7,
                left: 0,
            }
        );
        assert_eq!(diagnostic_sha256(&refined.gray), RETAINED_ROW_GRAY8_SHA256);

        let (page_gray, page_width, page_height) = crate::preprocess::tromr_gray8_input(&page)
            .expect("represent exact Spohr page as Gray8")
            .into_tightly_packed();
        let globally_deskewed = diagnostic_shear_gray_millidegrees(
            &page_gray,
            page_width,
            page_height,
            report.global_deskew.angle_millidegrees,
        );
        let (x, y, width, height) = refined.bbox;
        let mut unpadded = vec![0u8; width * height];
        for row in 0..height {
            let source_start = (y + row) * page_width + x;
            unpadded[row * width..(row + 1) * width]
                .copy_from_slice(&globally_deskewed[source_start..source_start + width]);
        }
        assert_eq!(
            diagnostic_sha256(&unpadded),
            RETAINED_ROW_PRE_REFINEMENT_GRAY8_SHA256,
            "native PDF render + global deskew reconstructs the exact pre-refinement crop"
        );

        let canvas_height = height + refined.padding.top + refined.padding.bottom;
        let mut canvas = vec![255u8; width * canvas_height];
        for row in 0..height {
            let target_row = row + refined.padding.top;
            canvas[target_row * width..(target_row + 1) * width]
                .copy_from_slice(&unpadded[row * width..(row + 1) * width]);
        }
        let crop = crate::preprocess::staff_detect::TromrGray8CropV1::from_tightly_packed(
            canvas,
            width,
            canvas_height,
        )
        .expect("construct exact pre-refinement padded crop");
        let png = crop
            .to_lossless_png()
            .expect("encode pre-refinement crop with provider PNG encoder");
        assert_eq!(
            evidence["selected_row"]["receipt"]["row_refinement"]["angle_millidegrees"].as_i64(),
            Some(-200)
        );
        DiagnosticRetainedRowInput {
            variant: "source_crop_before_row_refinement_padded",
            crop,
            png,
            row_refinement_angle_millidegrees: 0,
            source_crop_before_refinement_sha256: RETAINED_ROW_PRE_REFINEMENT_GRAY8_SHA256
                .to_owned(),
            refined_unpadded_crop_sha256: RETAINED_ROW_REFINED_UNPADDED_GRAY8_SHA256.to_owned(),
        }
    }

    fn diagnostic_semantic_notes(semantic: &str) -> Vec<(String, String)> {
        const DURATIONS: [&str; 7] = [
            "sixty_fourth",
            "thirty_second",
            "sixteenth",
            "eighth",
            "quarter",
            "half",
            "whole",
        ];
        semantic
            .split('+')
            .filter_map(|token| {
                let body = token.strip_prefix("note-")?;
                let dotted = body.ends_with('.');
                let undotted = body.strip_suffix('.').unwrap_or(body);
                DURATIONS.iter().find_map(|duration| {
                    undotted.strip_suffix(&format!("_{duration}")).map(|pitch| {
                        let duration = if dotted {
                            format!("{duration}.")
                        } else {
                            (*duration).to_owned()
                        };
                        (pitch.to_owned(), duration)
                    })
                })
            })
            .collect()
    }

    fn diagnostic_truth_notes(truth: &serde_json::Value) -> Vec<(String, String)> {
        truth["measures"]
            .as_array()
            .expect("manual truth has measures")
            .iter()
            .flat_map(|measure| {
                measure["notes"]
                    .as_array()
                    .expect("manual truth measure has notes")
            })
            .map(|note| {
                let pitch = &note["pitch"];
                (
                    format!(
                        "{}{}",
                        pitch["step"].as_str().expect("truth pitch step"),
                        pitch["octave"].as_i64().expect("truth pitch octave")
                    ),
                    note["written_duration"]
                        .as_str()
                        .expect("truth written duration")
                        .to_owned(),
                )
            })
            .collect()
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct DiagnosticTruthNoteTarget {
        measure_one_based: usize,
        measure_note: usize,
        pitch: String,
        duration: String,
        rhythm_id: u32,
        pitch_id: u32,
        lift_id: u32,
    }

    struct DiagnosticOracleStreams {
        rhythm_only: MusicStreams,
        all_streams: MusicStreams,
        note_target_by_stream_position: Vec<Option<usize>>,
        barline_count: usize,
    }

    fn diagnostic_token_id(
        tokenizer: &crate::tokenizer::music::MusicTokenizer,
        stream: crate::tokenizer::music::Stream,
        vocab_size: u32,
        token: &str,
    ) -> u32 {
        (0..vocab_size)
            .find(|&id| tokenizer.token(stream, id) == Some(token))
            .unwrap_or_else(|| panic!("diagnostic tokenizer has no {stream:?} token {token:?}"))
    }

    fn diagnostic_truth_note_targets(
        truth: &serde_json::Value,
        tokenizer: &crate::tokenizer::music::MusicTokenizer,
    ) -> Vec<DiagnosticTruthNoteTarget> {
        use crate::tokenizer::music::Stream;

        truth["measures"]
            .as_array()
            .expect("manual truth has measures")
            .iter()
            .flat_map(|measure| {
                let measure_one_based = usize::try_from(
                    measure["one_based_measure"]
                        .as_u64()
                        .expect("truth measure number"),
                )
                .expect("truth measure number fits usize");
                measure["notes"]
                    .as_array()
                    .expect("manual truth measure has notes")
                    .iter()
                    .map(move |note| (measure_one_based, note))
            })
            .map(|(measure_one_based, note)| {
                let measure_note = usize::try_from(
                    note["measure_note"]
                        .as_u64()
                        .expect("truth measure-note number"),
                )
                .expect("truth measure-note number fits usize");
                let pitch = &note["pitch"];
                let pitch = format!(
                    "{}{}",
                    pitch["step"].as_str().expect("truth pitch step"),
                    pitch["octave"].as_i64().expect("truth pitch octave")
                );
                let alter = note["pitch"]["alter"]
                    .as_i64()
                    .expect("truth pitch alteration");
                let lift_token = match alter {
                    -2 => "lift_bb",
                    -1 => "lift_b",
                    0 => "lift_null",
                    1 => "lift_#",
                    2 => "lift_##",
                    other => panic!("unsupported diagnostic truth alteration {other}"),
                };
                let duration = note["written_duration"]
                    .as_str()
                    .expect("truth written duration")
                    .to_owned();
                DiagnosticTruthNoteTarget {
                    measure_one_based,
                    measure_note,
                    rhythm_id: diagnostic_token_id(
                        tokenizer,
                        Stream::Rhythm,
                        260,
                        &format!("note-{duration}"),
                    ),
                    pitch_id: diagnostic_token_id(
                        tokenizer,
                        Stream::Pitch,
                        71,
                        &format!("note-{pitch}"),
                    ),
                    lift_id: diagnostic_token_id(tokenizer, Stream::Lift, 7, lift_token),
                    pitch,
                    duration,
                }
            })
            .collect()
    }

    fn diagnostic_oracle_streams(
        tokenizer: &crate::tokenizer::music::MusicTokenizer,
        observed: &MusicStreams,
        targets: &[DiagnosticTruthNoteTarget],
    ) -> DiagnosticOracleStreams {
        use crate::tokenizer::music::{EOS_ID, Stream};

        assert_eq!(observed.rhythm.len(), observed.pitch.len());
        assert_eq!(observed.rhythm.len(), observed.lift.len());
        assert_eq!(observed.rhythm.last(), Some(&EOS_ID));

        let mut all_rhythm = observed.rhythm.clone();
        let mut all_pitch = observed.pitch.clone();
        let mut all_lift = observed.lift.clone();
        let mut note_target_by_stream_position = vec![None; observed.rhythm.len()];
        let mut target_index = 0usize;
        let mut current_measure = 1usize;
        let mut barline_count = 0usize;
        for (stream_position, &rhythm_id) in observed.rhythm.iter().enumerate() {
            let token = tokenizer
                .token(Stream::Rhythm, rhythm_id)
                .expect("observed rhythm id is in vocabulary");
            if token.starts_with("note-") {
                let target = targets
                    .get(target_index)
                    .expect("observed stream has no extra note events");
                assert_eq!(
                    target.measure_one_based, current_measure,
                    "truth/observed measure alignment at stream position {stream_position}"
                );
                all_rhythm[stream_position] = target.rhythm_id;
                all_pitch[stream_position] = target.pitch_id;
                all_lift[stream_position] = target.lift_id;
                note_target_by_stream_position[stream_position] = Some(target_index);
                target_index += 1;
            } else if token == "barline" {
                barline_count += 1;
                current_measure += 1;
            }
        }
        assert_eq!(target_index, targets.len(), "every truth note is aligned");

        DiagnosticOracleStreams {
            rhythm_only: MusicStreams {
                rhythm: all_rhythm.clone(),
                pitch: observed.pitch.clone(),
                lift: observed.lift.clone(),
            },
            all_streams: MusicStreams {
                rhythm: all_rhythm,
                pitch: all_pitch,
                lift: all_lift,
            },
            note_target_by_stream_position,
            barline_count,
        }
    }

    fn diagnostic_prefix_hidden(
        decoder: &TromrDecoderW,
        ctx: &Mat,
        streams: &MusicStreams,
    ) -> FocrResult<Mat> {
        assert!(!streams.rhythm.is_empty());
        assert_eq!(streams.rhythm.len(), streams.pitch.len());
        assert_eq!(streams.rhythm.len(), streams.lift.len());

        let mut rhythm = Vec::with_capacity(streams.rhythm.len());
        let mut pitch = Vec::with_capacity(streams.pitch.len());
        let mut lift = Vec::with_capacity(streams.lift.len());
        rhythm.push(SEED_RHYTHM);
        pitch.push(SEED_NONOTE);
        lift.push(SEED_NONOTE);
        rhythm.extend_from_slice(&streams.rhythm[..streams.rhythm.len() - 1]);
        pitch.extend_from_slice(&streams.pitch[..streams.pitch.len() - 1]);
        lift.extend_from_slice(&streams.lift[..streams.lift.len() - 1]);
        decoder_forward(decoder, ctx, &rhythm, &pitch, &lift)
    }

    fn diagnostic_rhythm_logits(
        decoder: &TromrDecoderW,
        ctx: &Mat,
        streams: &MusicStreams,
    ) -> FocrResult<Mat> {
        let hidden = diagnostic_prefix_hidden(decoder, ctx, streams)?;
        decoder.head_rhythm.apply(&hidden)
    }

    struct DiagnosticAllHeadLogits {
        rhythm: Mat,
        pitch: Mat,
        lift: Mat,
        note: Mat,
    }

    impl DiagnosticAllHeadLogits {
        fn get(&self, head: &str) -> &Mat {
            match head {
                "rhythm" => &self.rhythm,
                "pitch" => &self.pitch,
                "lift" => &self.lift,
                "note" => &self.note,
                other => panic!("unknown diagnostic TrOMR head {other}"),
            }
        }
    }

    fn diagnostic_all_head_logits(
        decoder: &TromrDecoderW,
        ctx: &Mat,
        streams: &MusicStreams,
    ) -> FocrResult<DiagnosticAllHeadLogits> {
        let hidden = diagnostic_prefix_hidden(decoder, ctx, streams)?;
        Ok(DiagnosticAllHeadLogits {
            rhythm: decoder.head_rhythm.apply(&hidden)?,
            pitch: decoder.head_pitch.apply(&hidden)?,
            lift: decoder.head_lift.apply(&hidden)?,
            note: decoder.head_note.apply(&hidden)?,
        })
    }

    fn diagnostic_write_f32_tensor(
        output_dir: &std::path::Path,
        filename: &str,
        shape: &[usize],
        values: &[f32],
    ) -> serde_json::Value {
        assert_eq!(values.len(), shape.iter().product::<usize>());
        assert!(values.iter().all(|value| value.is_finite()));
        assert_eq!(
            std::path::Path::new(filename).file_name(),
            Some(std::ffi::OsStr::new(filename))
        );
        let bytes = diagnostic_f32_le_bytes(values);
        std::fs::write(output_dir.join(filename), &bytes).expect("write diagnostic f32 tensor");
        serde_json::json!({
            "file": filename,
            "shape": shape,
            "dtype": "f32",
            "byte_order": "little",
            "layout": "row_major_c_contiguous",
            "byte_len": bytes.len(),
            "sha256": diagnostic_sha256(&bytes),
        })
    }

    fn diagnostic_read_f32_tensor(
        manifest_path: &std::path::Path,
        descriptor: &serde_json::Value,
    ) -> (Vec<usize>, Vec<f32>) {
        let filename = descriptor["file"].as_str().expect("tensor descriptor file");
        assert_eq!(
            std::path::Path::new(filename).file_name(),
            Some(std::ffi::OsStr::new(filename)),
            "diagnostic tensor file must be a basename"
        );
        assert_eq!(descriptor["dtype"].as_str(), Some("f32"));
        assert_eq!(descriptor["byte_order"].as_str(), Some("little"));
        assert!(
            matches!(
                descriptor["layout"].as_str(),
                Some("row_major_c_contiguous" | "chw_c_contiguous")
            ),
            "diagnostic tensor layout must be one reviewed contiguous layout"
        );
        let shape = descriptor["shape"].as_array().expect("tensor shape");
        assert!(!shape.is_empty());
        let shape = shape
            .iter()
            .map(|dimension| {
                usize::try_from(dimension.as_u64().expect("tensor dimension"))
                    .expect("tensor dimension fits")
            })
            .collect::<Vec<_>>();
        let directory = manifest_path
            .parent()
            .expect("fixture has parent directory");
        let bytes = std::fs::read(directory.join(filename)).expect("read upstream tensor");
        assert_eq!(
            u64::try_from(bytes.len()).expect("tensor byte length fits u64"),
            descriptor["byte_len"].as_u64().expect("tensor byte_len")
        );
        assert_eq!(
            diagnostic_sha256(&bytes),
            descriptor["sha256"].as_str().expect("tensor SHA-256")
        );
        let (words, remainder) = bytes.as_chunks::<4>();
        assert!(remainder.is_empty());
        let values = words
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect::<Vec<_>>();
        assert_eq!(values.len(), shape.iter().product::<usize>());
        assert!(values.iter().all(|value| value.is_finite()));
        (shape, values)
    }

    fn diagnostic_logits_comparison(native: &Mat, upstream: &[f32]) -> serde_json::Value {
        assert_eq!(native.data.len(), upstream.len());
        assert!(!native.data.is_empty());
        let mut mean_abs = 0.0f64;
        let mut argmax_mismatch_count = 0usize;
        let mut first_argmax_mismatch = None;
        for row in 0..native.rows {
            let native_row = native.row(row);
            let upstream_row = &upstream[row * native.cols..(row + 1) * native.cols];
            mean_abs += native_row
                .iter()
                .zip(upstream_row)
                .map(|(&left, &right)| f64::from((left - right).abs()))
                .sum::<f64>();
            if legacy_argmax_id(native_row) != legacy_argmax_id(upstream_row) {
                argmax_mismatch_count += 1;
                first_argmax_mismatch.get_or_insert(row);
            }
        }
        mean_abs /= native.data.len() as f64;
        serde_json::json!({
            "shape": [native.rows, native.cols],
            "cosine": cos(&native.data, upstream),
            "max_abs": maxabs(&native.data, upstream),
            "mean_abs": mean_abs,
            "argmax_mismatch_count": argmax_mismatch_count,
            "first_argmax_mismatch_position_zero_based": first_argmax_mismatch,
        })
    }

    fn diagnostic_streams_from_json(value: &serde_json::Value) -> MusicStreams {
        serde_json::from_value(value.clone()).expect("parse exact upstream TrOMR streams")
    }

    fn diagnostic_compare_upstream_fixture(
        fixture_path: &std::path::Path,
        input: &DiagnosticRetainedRowInput,
        preprocessed: &[f32],
        preprocessed_width: usize,
        streams: &MusicStreams,
        logits: &DiagnosticAllHeadLogits,
        first_truth_divergence: Option<(usize, u32)>,
    ) -> serde_json::Value {
        let fixture: serde_json::Value = serde_json::from_slice(
            &std::fs::read(fixture_path).expect("read exact upstream TrOMR fixture"),
        )
        .expect("parse exact upstream TrOMR fixture");
        assert_eq!(
            fixture["_meta"]["checkpoint_sha256"].as_str(),
            Some(TROMR_UPSTREAM_CHECKPOINT_SHA256)
        );
        let input_png_sha256 = diagnostic_sha256(&input.png);
        assert_eq!(
            fixture["_meta"]["page_sha256"].as_str(),
            Some(input_png_sha256.as_str())
        );
        let full = &fixture["free_argmax_full_logits"];
        assert_eq!(
            full["schema"].as_str(),
            Some("franken_ocr.tromr.upstream_free_argmax_full_logits.v1")
        );
        let upstream_streams = diagnostic_streams_from_json(&full["streams"]);
        let streams_exact = upstream_streams == *streams;
        let first_stream_difference = (0..streams.rhythm.len().min(upstream_streams.rhythm.len()))
            .find(|&index| {
                streams.rhythm[index] != upstream_streams.rhythm[index]
                    || streams.pitch[index] != upstream_streams.pitch[index]
                    || streams.lift[index] != upstream_streams.lift[index]
            })
            .or_else(|| {
                (streams.rhythm.len() != upstream_streams.rhythm.len())
                    .then_some(streams.rhythm.len().min(upstream_streams.rhythm.len()))
            });

        let (pre_shape, upstream_preprocessed) =
            diagnostic_read_f32_tensor(fixture_path, &fixture["preproc"]);
        assert_eq!(pre_shape, [1, IMG_H, preprocessed_width]);
        assert_eq!(upstream_preprocessed.len(), preprocessed.len());
        let one_lsb = 1.0f32 / (0.1738 * 255.0);
        let preprocess_past_one_and_half_lsb = preprocessed
            .iter()
            .zip(&upstream_preprocessed)
            .filter(|(native, upstream)| (**native - **upstream).abs() > one_lsb * 1.5)
            .count();

        let mut head_comparisons = serde_json::Map::new();
        let mut upstream_rhythm = None;
        for (head, vocabulary) in [
            ("rhythm", 260usize),
            ("pitch", 71),
            ("lift", 7),
            ("note", 2),
        ] {
            let native = logits.get(head);
            let (shape, upstream) = diagnostic_read_f32_tensor(fixture_path, &full["heads"][head]);
            assert_eq!(shape, [native.rows, vocabulary]);
            if head == "rhythm" {
                upstream_rhythm = Some(upstream.clone());
            }
            head_comparisons.insert(
                head.to_owned(),
                diagnostic_logits_comparison(native, &upstream),
            );
        }

        let first_manual_truth_divergence =
            first_truth_divergence.map(|(stream_position, expected_rhythm_id)| {
                assert!(stream_position < logits.rhythm.rows);
                let native_row = logits.rhythm.row(stream_position);
                let upstream_rhythm = upstream_rhythm
                    .as_ref()
                    .expect("upstream rhythm logits captured");
                let upstream_row =
                    &upstream_rhythm[stream_position * 260..(stream_position + 1) * 260];
                let native_argmax = legacy_argmax_id(native_row);
                let upstream_argmax = legacy_argmax_id(upstream_row);
                serde_json::json!({
                    "stream_position_zero_based": stream_position,
                    "expected_rhythm_id": expected_rhythm_id,
                    "native_argmax_id": native_argmax,
                    "upstream_argmax_id": upstream_argmax,
                    "native_expected_logit": native_row[expected_rhythm_id as usize],
                    "upstream_expected_logit": upstream_row[expected_rhythm_id as usize],
                    "native_argmax_logit": native_row[native_argmax as usize],
                    "upstream_argmax_logit": upstream_row[upstream_argmax as usize],
                    "expected_logit_abs_delta": (
                        native_row[expected_rhythm_id as usize]
                            - upstream_row[expected_rhythm_id as usize]
                    ).abs(),
                })
            });
        serde_json::json!({
            "schema": "franken_ocr.tromr.native_vs_upstream_retained_row.v1",
            "streams_exact": streams_exact,
            "first_stream_difference_position_zero_based": first_stream_difference,
            "native_streams": streams,
            "upstream_streams": upstream_streams,
            "preprocess": {
                "shape": [1, IMG_H, preprocessed_width],
                "native_sha256": diagnostic_sha256(&diagnostic_f32_le_bytes(preprocessed)),
                "upstream_sha256": fixture["preproc"]["sha256"],
                "cosine": cos(preprocessed, &upstream_preprocessed),
                "max_abs": maxabs(preprocessed, &upstream_preprocessed),
                "one_u8_lsb_normalized": one_lsb,
                "pixels_past_one_and_half_lsb": preprocess_past_one_and_half_lsb,
            },
            "heads": head_comparisons,
            "first_manual_truth_divergence": first_manual_truth_divergence,
        })
    }

    fn diagnostic_write_native_full_logits(
        output_dir: &std::path::Path,
        input: &DiagnosticRetainedRowInput,
        model_sha256: &str,
        preprocessed: &[f32],
        preprocessed_width: usize,
        streams: &MusicStreams,
        logits: &DiagnosticAllHeadLogits,
        first_truth_divergence: Option<(usize, u32)>,
        upstream_comparison: Option<&serde_json::Value>,
    ) -> (String, String) {
        std::fs::create_dir_all(output_dir).expect("create exact TrOMR diagnostic directory");
        let prefix = format!("native_{}", input.variant);
        let png_filename = format!("{prefix}_input.png");
        std::fs::write(output_dir.join(&png_filename), &input.png)
            .expect("write exact diagnostic input PNG");
        let preprocessed_descriptor = diagnostic_write_f32_tensor(
            output_dir,
            &format!("{prefix}_preprocessed.f32le.bin"),
            &[1, IMG_H, preprocessed_width],
            preprocessed,
        );
        let mut heads = serde_json::Map::new();
        for (head, vocabulary) in [
            ("rhythm", 260usize),
            ("pitch", 71),
            ("lift", 7),
            ("note", 2),
        ] {
            let values = logits.get(head);
            assert_eq!(
                (values.rows, values.cols),
                (streams.rhythm.len(), vocabulary)
            );
            heads.insert(
                head.to_owned(),
                diagnostic_write_f32_tensor(
                    output_dir,
                    &format!("{prefix}_free_argmax_{head}_logits.f32le.bin"),
                    &[values.rows, values.cols],
                    &values.data,
                ),
            );
        }
        let manifest = serde_json::json!({
            "schema": "franken_ocr.tromr.native_retained_row_full_logits.v1",
            "implementation": "franken_ocr_native_f32",
            "prefix_contract":
                "seed_rhythm_1_pitch_0_lift_0_then_free_argmax_previous_tokens_v1",
            "variant": input.variant,
            "model_sha256": model_sha256,
            "input": {
                "png_file": png_filename,
                "png_byte_len": input.png.len(),
                "png_sha256": diagnostic_sha256(&input.png),
                "gray8_dimensions": [input.crop.width(), input.crop.height()],
                "gray8_byte_len": input.crop.pixels().len(),
                "gray8_sha256": diagnostic_sha256(input.crop.pixels()),
                "row_refinement_angle_millidegrees":
                    input.row_refinement_angle_millidegrees,
                "source_crop_before_refinement_sha256":
                    input.source_crop_before_refinement_sha256,
                "refined_unpadded_crop_sha256": input.refined_unpadded_crop_sha256,
            },
            "preproc": preprocessed_descriptor,
            "free_argmax_full_logits": {
                "step_count": streams.rhythm.len(),
                "streams": streams,
                "heads": heads,
            },
            "first_manual_truth_divergence": first_truth_divergence.map(
                |(stream_position, expected_rhythm_id)| serde_json::json!({
                    "stream_position_zero_based": stream_position,
                    "expected_rhythm_id": expected_rhythm_id,
                    "free_rhythm_id": streams.rhythm[stream_position],
                })
            ),
            "upstream_comparison": upstream_comparison,
        });
        let filename = format!("{prefix}_full_logits.json");
        let bytes = serde_json::to_vec_pretty(&manifest).expect("serialize native logit manifest");
        std::fs::write(output_dir.join(&filename), &bytes).expect("write native logit manifest");
        (filename, diagnostic_sha256(&bytes))
    }

    fn diagnostic_ranked_ids(logits: &[f32]) -> Vec<usize> {
        ranked_candidate_ids(logits).expect("diagnostic logits are finite and non-empty")
    }

    fn diagnostic_top_k_temperature_probability(
        logits: &[f32],
        ranked: &[usize],
        target_id: u32,
    ) -> f64 {
        const TEMPERATURE: f32 = 0.2;
        let kept = logits.len().div_ceil(10).max(1);
        let kept_ids = &ranked[..kept];
        if !kept_ids.contains(&(target_id as usize)) {
            return 0.0;
        }
        let max = logits[kept_ids[0]] / TEMPERATURE;
        let total: f32 = kept_ids
            .iter()
            .map(|&id| (logits[id] / TEMPERATURE - max).exp())
            .sum();
        f64::from((logits[target_id as usize] / TEMPERATURE - max).exp() / total)
    }

    fn diagnostic_rhythm_readout(
        tokenizer: &crate::tokenizer::music::MusicTokenizer,
        logits: &[f32],
        expected_id: u32,
        observed_id: u32,
    ) -> serde_json::Value {
        use crate::tokenizer::music::Stream;

        let ranked = diagnostic_ranked_ids(logits);
        let rank = |id: u32| {
            ranked
                .iter()
                .position(|&candidate| candidate == id as usize)
                .expect("rhythm id is ranked")
                + 1
        };
        let candidate = |id: u32| {
            serde_json::json!({
                "id": id,
                "token": tokenizer.token(Stream::Rhythm, id),
                "logit": logits[id as usize],
                "rank": rank(id),
                "inside_upstream_top_k": rank(id) <= logits.len().div_ceil(10),
                "top_k_temperature_0_2_probability":
                    diagnostic_top_k_temperature_probability(logits, &ranked, id),
            })
        };
        let argmax_id = ranked[0] as u32;
        serde_json::json!({
            "upstream_kept_count": logits.len().div_ceil(10),
            "expected": candidate(expected_id),
            "observed": candidate(observed_id),
            "argmax": candidate(argmax_id),
            "argmax_margin_over_expected":
                logits[argmax_id as usize] - logits[expected_id as usize],
            "observed_margin_over_expected":
                logits[observed_id as usize] - logits[expected_id as usize],
            "top_five": ranked.iter().take(5).map(|&id| candidate(id as u32)).collect::<Vec<_>>(),
        })
    }

    fn diagnostic_rhythm_mode_summary(
        logits: &Mat,
        expected_rhythm: &[u32],
        note_target_by_stream_position: &[Option<usize>],
        first_free_divergence: Option<usize>,
    ) -> serde_json::Value {
        let mut argmax_exact = 0usize;
        let mut top_two = 0usize;
        let mut top_five = 0usize;
        let mut top_k = 0usize;
        let mut argmax_at_or_after_first_divergence = 0usize;
        let mut notes_at_or_after_first_divergence = 0usize;
        let mut margins = Vec::<f32>::new();
        for (stream_position, note_target) in note_target_by_stream_position.iter().enumerate() {
            if note_target.is_none() {
                continue;
            }
            let row = logits.row(stream_position);
            let ranked = diagnostic_ranked_ids(row);
            let expected_id = expected_rhythm[stream_position] as usize;
            let rank = ranked
                .iter()
                .position(|&id| id == expected_id)
                .expect("expected rhythm id is ranked")
                + 1;
            argmax_exact += usize::from(rank == 1);
            top_two += usize::from(rank <= 2);
            top_five += usize::from(rank <= 5);
            top_k += usize::from(rank <= row.len().div_ceil(10));
            margins.push(row[ranked[0]] - row[expected_id]);
            if first_free_divergence.is_some_and(|first| stream_position >= first) {
                notes_at_or_after_first_divergence += 1;
                argmax_at_or_after_first_divergence += usize::from(rank == 1);
            }
        }
        margins.sort_by(f32::total_cmp);
        let mean_margin =
            margins.iter().map(|&value| f64::from(value)).sum::<f64>() / margins.len() as f64;
        let median_margin = if margins.len().is_multiple_of(2) {
            f64::from(margins[margins.len() / 2 - 1] + margins[margins.len() / 2]) / 2.0
        } else {
            f64::from(margins[margins.len() / 2])
        };
        serde_json::json!({
            "note_count": margins.len(),
            "expected_argmax_count": argmax_exact,
            "expected_top_two_count": top_two,
            "expected_top_five_count": top_five,
            "expected_inside_upstream_top_k_count": top_k,
            "mean_argmax_margin_over_expected": mean_margin,
            "median_argmax_margin_over_expected": median_margin,
            "notes_at_or_after_first_free_divergence": notes_at_or_after_first_divergence,
            "expected_argmax_at_or_after_first_free_divergence":
                argmax_at_or_after_first_divergence,
        })
    }

    fn diagnostic_teacher_forced_rhythm_report(
        decoder: &TromrDecoderW,
        ctx: &Mat,
        tokenizer: &crate::tokenizer::music::MusicTokenizer,
        observed: &MusicStreams,
        targets: &[DiagnosticTruthNoteTarget],
        oracle: &DiagnosticOracleStreams,
        assert_observed_argmax_replay: bool,
    ) -> FocrResult<serde_json::Value> {
        use crate::tokenizer::music::Stream;

        let started = std::time::Instant::now();
        let observed_logits = diagnostic_rhythm_logits(decoder, ctx, observed)?;
        let oracle_rhythm_logits = diagnostic_rhythm_logits(decoder, ctx, &oracle.rhythm_only)?;
        let oracle_all_logits = diagnostic_rhythm_logits(decoder, ctx, &oracle.all_streams)?;
        let elapsed_ms = started.elapsed().as_millis();
        for logits in [&observed_logits, &oracle_rhythm_logits, &oracle_all_logits] {
            assert_eq!(logits.rows, observed.rhythm.len());
            assert_eq!(logits.cols, 260);
            assert!(logits.data.iter().all(|value| value.is_finite()));
        }

        if assert_observed_argmax_replay {
            for (stream_position, &observed_id) in observed.rhythm.iter().enumerate() {
                let ranked = diagnostic_ranked_ids(observed_logits.row(stream_position));
                assert_eq!(
                    ranked[0] as u32, observed_id,
                    "full causal replay differs from free argmax at stream position {stream_position}"
                );
            }
        }

        let first_free_divergence = oracle
            .note_target_by_stream_position
            .iter()
            .enumerate()
            .find_map(|(stream_position, note_target)| {
                note_target.filter(|_| {
                    observed.rhythm[stream_position] != oracle.all_streams.rhythm[stream_position]
                })?;
                Some(stream_position)
            });
        let events = (0..observed.rhythm.len())
            .map(|stream_position| {
                let expected_id = oracle.all_streams.rhythm[stream_position];
                let observed_id = observed.rhythm[stream_position];
                let note_target =
                    oracle.note_target_by_stream_position[stream_position].map(|target_index| {
                        let target = &targets[target_index];
                        serde_json::json!({
                            "truth_note_index_zero_based": target_index,
                            "measure_one_based": target.measure_one_based,
                            "measure_note": target.measure_note,
                            "pitch": target.pitch,
                            "duration": target.duration,
                            "oracle_pitch_id": target.pitch_id,
                            "oracle_lift_id": target.lift_id,
                        })
                    });
                serde_json::json!({
                    "stream_position_zero_based": stream_position,
                    "sequence_position_one_based": stream_position + 1,
                    "event_kind": if note_target.is_some() { "note" } else { "structural" },
                    "note_truth": note_target,
                    "expected_rhythm_id": expected_id,
                    "expected_rhythm_token": tokenizer.token(Stream::Rhythm, expected_id),
                    "free_rhythm_id": observed_id,
                    "free_rhythm_token": tokenizer.token(Stream::Rhythm, observed_id),
                    "observed_prefix": diagnostic_rhythm_readout(
                        tokenizer,
                        observed_logits.row(stream_position),
                        expected_id,
                        observed_id,
                    ),
                    "oracle_rhythm_prefix_observed_pitch_lift": diagnostic_rhythm_readout(
                        tokenizer,
                        oracle_rhythm_logits.row(stream_position),
                        expected_id,
                        observed_id,
                    ),
                    "oracle_all_streams_prefix": diagnostic_rhythm_readout(
                        tokenizer,
                        oracle_all_logits.row(stream_position),
                        expected_id,
                        observed_id,
                    ),
                })
            })
            .collect::<Vec<_>>();

        Ok(serde_json::json!({
            "schema": "franken_ocr.tromr.retained_row_teacher_forced_rhythm.v1",
            "method": "single_full_causal_forward_per_prefix_mode",
            "prefix_modes": [
                "observed_prefix",
                "oracle_rhythm_prefix_observed_pitch_lift",
                "oracle_all_streams_prefix",
            ],
            "upstream_top_k_threshold": 0.9,
            "upstream_temperature": 0.2,
            "upstream_rhythm_kept_count": 26,
            "elapsed_ms": elapsed_ms,
            "first_free_rhythm_divergence_stream_position_zero_based": first_free_divergence,
            "first_free_rhythm_divergence_note_index_zero_based": first_free_divergence.and_then(
                |position| oracle.note_target_by_stream_position[position]
            ),
            "summaries": {
                "observed_prefix": diagnostic_rhythm_mode_summary(
                    &observed_logits,
                    &oracle.all_streams.rhythm,
                    &oracle.note_target_by_stream_position,
                    first_free_divergence,
                ),
                "oracle_rhythm_prefix_observed_pitch_lift": diagnostic_rhythm_mode_summary(
                    &oracle_rhythm_logits,
                    &oracle.all_streams.rhythm,
                    &oracle.note_target_by_stream_position,
                    first_free_divergence,
                ),
                "oracle_all_streams_prefix": diagnostic_rhythm_mode_summary(
                    &oracle_all_logits,
                    &oracle.all_streams.rhythm,
                    &oracle.note_target_by_stream_position,
                    first_free_divergence,
                ),
            },
            "events": events,
        }))
    }

    fn diagnostic_recognize_with_streams(
        weights: &Weights,
        tokenizer: &crate::tokenizer::music::MusicTokenizer,
        image: &image::DynamicImage,
        options: TromrRecognitionOptionsV1,
        execution: &mut TromrExecutionContext,
    ) -> FocrResult<(MusicResult, MusicStreams, Mat, TromrDecoderW)> {
        let options = options.validate()?;
        let options_identity = options.replay_identity()?;
        execution.begin_forward_attempt("staff-forward")?;
        let result = (|| {
            execution.checkpoint("staff-preprocess")?;
            let preprocess_started = std::time::Instant::now();
            let preprocessed = match options.staff_resampler {
                TromrStaffResamplerV1::Cv2LinearU8V1 => {
                    crate::preprocess::tromr_staff_tensor(image)
                }
            };
            execution
                .record_attempt_stage(TromrAttemptStage::Preprocess, preprocess_started.elapsed());
            let (pixels, width) = preprocessed?;

            execution.checkpoint("staff-encode")?;
            let encode_started = std::time::Instant::now();
            let encoded = (|| {
                let encoder = TromrEncoderW::build(weights)?;
                encode(&encoder, &pixels, width)
            })();
            execution.record_attempt_stage(TromrAttemptStage::Encode, encode_started.elapsed());
            let ctx = encoded?;
            execution.checkpoint("staff-encode")?;

            let decode_started = std::time::Instant::now();
            let decoder = TromrDecoderW::build(weights)?;
            let decoded = generate_with_control_and_forward(
                &decoder,
                &ctx,
                options.decode_pick()?,
                execution,
                decoder_forward,
            );
            execution.record_attempt_stage(TromrAttemptStage::Decode, decode_started.elapsed());
            let generation = decoded?;
            let forward_candidate =
                TromrForwardCandidateLatticeV1::new(0, generation.candidate_lattice)?;
            let streams = generation.streams;
            execution.checkpoint("staff-semantic-assembly")?;

            let semantic_started = std::time::Instant::now();
            let assembled: FocrResult<(String, String)> = (|| {
                let semantic = merge_semantic(tokenizer, &streams)?;
                let musicxml = semantic_to_musicxml(&semantic)?;
                Ok((semantic, musicxml))
            })();
            execution.record_attempt_stage(
                TromrAttemptStage::SemanticAssembly,
                semantic_started.elapsed(),
            );
            let (semantic, musicxml) = assembled?;
            execution.checkpoint("staff-semantic-assembly")?;
            Ok((
                MusicResult {
                    semantic,
                    musicxml,
                    options,
                    options_identity,
                    forward_candidate_lattices: vec![forward_candidate],
                },
                streams,
                ctx,
                decoder,
            ))
        })();
        execution.finish_attempt(&result);
        result
    }

    /// Reconstruct the exact row-2 crop before the provider's -0.2 degree
    /// row-local refinement. The input PDF is rendered by `franken_ocr::pdf`,
    /// then the provider's own detector geometry and transformation identities
    /// are checked before any artifact is written. No external renderer or
    /// image converter participates.
    #[test]
    #[ignore = "requires exact retained bundle, Spohr source PDF, and output directory"]
    fn materialize_spohr_row_pre_refinement_diagnostic_crop() {
        let bundle_path = std::path::PathBuf::from(
            std::env::var_os("FOCR_TROMR_DIAGNOSTIC_BUNDLE")
                .expect("FOCR_TROMR_DIAGNOSTIC_BUNDLE must name the retained evidence bundle"),
        );
        let source_pdf = std::path::PathBuf::from(
            std::env::var_os("FOCR_TROMR_DIAGNOSTIC_SOURCE_PDF")
                .expect("FOCR_TROMR_DIAGNOSTIC_SOURCE_PDF must name the exact Spohr PDF"),
        );
        let output_dir = std::path::PathBuf::from(
            std::env::var_os("FOCR_TROMR_DIAGNOSTIC_OUT_DIR")
                .expect("FOCR_TROMR_DIAGNOSTIC_OUT_DIR must name an evidence directory"),
        );
        let evidence: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&bundle_path).expect("read retained evidence bundle"),
        )
        .expect("parse retained evidence bundle");
        let input = diagnostic_pre_refinement_input(&evidence, &source_pdf);
        std::fs::create_dir_all(&output_dir).expect("create diagnostic output directory");
        let png_filename = "spohr-page55-row02-pre-refinement-padded.png";
        std::fs::write(output_dir.join(png_filename), &input.png)
            .expect("write exact pre-refinement provider PNG");
        let manifest = serde_json::json!({
            "schema": "franken_ocr.tromr.pre_refinement_crop_materialization.v1",
            "provider_only": true,
            "source_pdf_sha256": SPOHR_SOURCE_PDF_SHA256,
            "selected_page_one_based": 55,
            "detection_index_zero_based": 2,
            "source_bbox_xywh_in_globally_deskewed_raster": [58, 1008, 2398, 226],
            "global_deskew_angle_millidegrees": 250,
            "omitted_row_refinement_angle_millidegrees": -200,
            "padding_trbl": [7, 0, 7, 0],
            "gray8_dimensions": [input.crop.width(), input.crop.height()],
            "gray8_sha256": diagnostic_sha256(input.crop.pixels()),
            "png_file": png_filename,
            "png_byte_len": input.png.len(),
            "png_sha256": diagnostic_sha256(&input.png),
            "source_crop_before_refinement_unpadded_sha256":
                RETAINED_ROW_PRE_REFINEMENT_GRAY8_SHA256,
            "refined_unpadded_crop_sha256": RETAINED_ROW_REFINED_UNPADDED_GRAY8_SHA256,
        });
        let manifest_bytes =
            serde_json::to_vec_pretty(&manifest).expect("serialize crop materialization manifest");
        let manifest_filename = "spohr-page55-row02-pre-refinement-padded.json";
        std::fs::write(output_dir.join(manifest_filename), &manifest_bytes)
            .expect("write crop materialization manifest");
        eprintln!(
            "{}",
            serde_json::json!({
                "event": "tromr_pre_refinement_crop_materialized",
                "result": "pass",
                "manifest_file": manifest_filename,
                "manifest_sha256": diagnostic_sha256(&manifest_bytes),
                "artifact": manifest,
            })
        );
    }

    /// Direct, retained-pixel TrOMR diagnostic. The default refined variant
    /// bypasses page detection and decodes the exact provider-produced L8 PNG.
    /// The explicit pre-refinement A/B variant reconstructs its crop through
    /// the provider's PDF renderer and staff detector, verifying every frozen
    /// geometry and pixel identity before the same single-staff forward.
    #[test]
    #[ignore = "requires exact retained row, model, tokenizer, and manual-truth paths"]
    fn retained_spohr_row_direct_model_diagnostic() {
        use std::collections::BTreeMap;
        use std::path::PathBuf;

        let bundle_path = PathBuf::from(
            std::env::var_os("FOCR_TROMR_DIAGNOSTIC_BUNDLE")
                .expect("FOCR_TROMR_DIAGNOSTIC_BUNDLE must name the retained evidence bundle"),
        );
        let model_path = PathBuf::from(
            std::env::var_os("FOCR_TROMR_DIAGNOSTIC_MODEL")
                .expect("FOCR_TROMR_DIAGNOSTIC_MODEL must name one exact TrOMR artifact"),
        );
        let tokenizer_dir = PathBuf::from(
            std::env::var_os("FOCR_TROMR_DIAGNOSTIC_TOKENIZER_DIR")
                .expect("FOCR_TROMR_DIAGNOSTIC_TOKENIZER_DIR must contain exact tables"),
        );
        let truth_path = PathBuf::from(
            std::env::var_os("FOCR_TROMR_DIAGNOSTIC_TRUTH")
                .expect("FOCR_TROMR_DIAGNOSTIC_TRUTH must name the manual row truth"),
        );

        let evidence: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&bundle_path).expect("read retained evidence bundle"),
        )
        .expect("parse retained evidence bundle");
        assert_eq!(
            evidence["selected_row"]["model_input_png"]["sha256"].as_str(),
            Some(RETAINED_ROW_PNG_SHA256)
        );
        assert_eq!(
            evidence["selected_row"]["model_input_gray8"]["pixels_sha256"].as_str(),
            Some(RETAINED_ROW_GRAY8_SHA256)
        );
        let variant = std::env::var("FOCR_TROMR_DIAGNOSTIC_INPUT_VARIANT")
            .unwrap_or_else(|_| "refined".to_owned());
        let input = match variant.as_str() {
            "refined" => diagnostic_refined_input(&evidence),
            "pre_refinement" => {
                let source_pdf = std::path::PathBuf::from(
                    std::env::var_os("FOCR_TROMR_DIAGNOSTIC_SOURCE_PDF")
                        .expect("pre_refinement requires FOCR_TROMR_DIAGNOSTIC_SOURCE_PDF"),
                );
                diagnostic_pre_refinement_input(&evidence, &source_pdf)
            }
            other => panic!(
                "FOCR_TROMR_DIAGNOSTIC_INPUT_VARIANT must be refined or pre_refinement, got {other}"
            ),
        };
        let image = image::GrayImage::from_raw(
            input.crop.width() as u32,
            input.crop.height() as u32,
            input.crop.pixels().to_vec(),
        )
        .expect("retained Gray8 dimensions match pixels");
        let image = image::DynamicImage::ImageLuma8(image);
        let (preprocessed, preprocessed_width) = crate::preprocess::tromr_staff_tensor(&image)
            .expect("exact retained row fits TrOMR preprocessing");
        assert_eq!(preprocessed_width, 1_264);

        let model_bytes = std::fs::read(&model_path).expect("read exact model artifact");
        let model_sha256 = diagnostic_sha256(&model_bytes);
        let quant = match model_sha256.as_str() {
            TROMR_F32_SHA256 => "f32",
            TROMR_INT8_SHA256 => "int8",
            other => panic!("unexpected TrOMR model SHA-256 {other}"),
        };
        let weights = Weights::from_bytes(model_bytes).expect("parse exact model artifact");
        assert_eq!(weights.model_id(), "tromr");

        let tokenizer_bytes = crate::tokenizer::music::TOKENIZER_FILENAMES.map(|filename| {
            std::fs::read(tokenizer_dir.join(filename)).expect("read exact tokenizer table")
        });
        for (index, bytes) in tokenizer_bytes.iter().enumerate() {
            assert_eq!(
                diagnostic_sha256(bytes),
                TROMR_TOKENIZER_SHA256[index],
                "tokenizer identity changed at index {index}"
            );
        }
        let tokenizer = crate::tokenizer::music::MusicTokenizer::from_owned_tables(tokenizer_bytes)
            .expect("parse exact tokenizer tables");

        let mode =
            std::env::var("FOCR_TROMR_DIAGNOSTIC_MODE").unwrap_or_else(|_| "argmax".to_owned());
        let options = match mode.as_str() {
            "argmax" => TromrRecognitionOptionsV1::deterministic(),
            "seeded" => TromrRecognitionOptionsV1 {
                decode_mode: TromrDecodeModeV1::SeededTopKTemperature,
                seed: Some(
                    std::env::var("FOCR_TROMR_DIAGNOSTIC_SEED")
                        .expect("seeded mode requires FOCR_TROMR_DIAGNOSTIC_SEED")
                        .parse()
                        .expect("diagnostic seed is a u64"),
                ),
                ..TromrRecognitionOptionsV1::deterministic()
            },
            other => panic!("FOCR_TROMR_DIAGNOSTIC_MODE must be argmax or seeded, got {other}"),
        };
        let execution_options = crate::music_execution::TromrExecutionOptionsV1 {
            schema_version: crate::music_execution::TROMR_EXECUTION_OPTIONS_SCHEMA_VERSION,
            setup_budget_ms: 60_000,
            per_forward_attempt_budget_ms: 600_000,
            max_forward_attempts: 1,
        };
        let mut execution = crate::music_execution::TromrExecutionContext::new(
            execution_options,
            crate::music_execution::MusicCancellationToken::new(),
        )
        .expect("finite diagnostic execution policy is valid");
        execution.mark_worker_started();
        let started = std::time::Instant::now();
        let outcome = diagnostic_recognize_with_streams(
            &weights,
            &tokenizer,
            &image,
            options,
            &mut execution,
        );
        let elapsed_ms = started.elapsed().as_millis();
        let diagnostics = execution.finish_diagnostics(&outcome);
        let (result, streams, ctx, decoder) =
            outcome.expect("direct retained-row recognition succeeds within 11 minutes");

        let truth: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&truth_path).expect("read manual row truth"))
                .expect("parse manual row truth");
        assert_eq!(
            truth["fixture_id"].as_str(),
            Some("spohr-page55-detection2-review-crop-v11")
        );
        assert_eq!(
            truth["source_scope"]["review_gray8_pixels_sha256"].as_str(),
            Some(RETAINED_ROW_GRAY8_SHA256)
        );
        let (teacher_forced_rhythm, first_truth_divergence) = if input.variant
            == "row_refined_minus_200_millidegrees"
        {
            let truth_targets = diagnostic_truth_note_targets(&truth, &tokenizer);
            let oracle = diagnostic_oracle_streams(&tokenizer, &streams, &truth_targets);
            assert_eq!(truth_targets.len(), 40);
            assert_eq!(oracle.barline_count, 6);
            let first = oracle
                .note_target_by_stream_position
                .iter()
                .enumerate()
                .find_map(|(position, target)| {
                    target.filter(|_| {
                        streams.rhythm[position] != oracle.all_streams.rhythm[position]
                    })?;
                    Some((position, oracle.all_streams.rhythm[position]))
                });
            assert_eq!(first, Some((3, 139)));
            (
                diagnostic_teacher_forced_rhythm_report(
                    &decoder,
                    &ctx,
                    &tokenizer,
                    &streams,
                    &truth_targets,
                    &oracle,
                    options.decode_mode == TromrDecodeModeV1::Argmax,
                )
                .expect("teacher-forced rhythm-head diagnostic succeeds"),
                first,
            )
        } else {
            (
                serde_json::json!({
                    "status": "not_applicable",
                    "reason": "manual truth is bound to the refined review crop; this run is the pre-refinement A/B input",
                }),
                None,
            )
        };

        let full_logits = (options.decode_mode == TromrDecodeModeV1::Argmax).then(|| {
            diagnostic_all_head_logits(&decoder, &ctx, &streams)
                .expect("full free-prefix head logits replay")
        });
        if let Some(logits) = &full_logits {
            for position in 0..streams.rhythm.len() {
                assert_eq!(
                    legacy_argmax_id(logits.rhythm.row(position)),
                    streams.rhythm[position],
                    "full rhythm replay at position {position}"
                );
                assert_eq!(
                    legacy_argmax_id(logits.pitch.row(position)),
                    streams.pitch[position],
                    "full pitch replay at position {position}"
                );
                assert_eq!(
                    legacy_argmax_id(logits.lift.row(position)),
                    streams.lift[position],
                    "full lift replay at position {position}"
                );
            }
        }

        let upstream_comparison =
            std::env::var_os("FOCR_TROMR_DIAGNOSTIC_UPSTREAM_FIXTURE").map(|path| {
                assert_eq!(quant, "f32", "upstream parity requires the f32 artifact");
                let logits = full_logits
                    .as_ref()
                    .expect("upstream parity requires deterministic argmax mode");
                diagnostic_compare_upstream_fixture(
                    &PathBuf::from(path),
                    &input,
                    &preprocessed,
                    preprocessed_width,
                    &streams,
                    logits,
                    first_truth_divergence,
                )
            });
        let native_full_logits_manifest =
            std::env::var_os("FOCR_TROMR_DIAGNOSTIC_OUT_DIR").map(|path| {
                let logits = full_logits
                    .as_ref()
                    .expect("full-logit artifacts require deterministic argmax mode");
                let output_dir = PathBuf::from(path);
                let (file, sha256) = diagnostic_write_native_full_logits(
                    &output_dir,
                    &input,
                    &model_sha256,
                    &preprocessed,
                    preprocessed_width,
                    &streams,
                    logits,
                    first_truth_divergence,
                    upstream_comparison.as_ref(),
                );
                serde_json::json!({
                    "file": file,
                    "sha256": sha256,
                    "output_directory_is_diagnostic_only": true,
                })
            });
        let observed = diagnostic_semantic_notes(&result.semantic);
        let expected = diagnostic_truth_notes(&truth);
        let mut observed_duration_histogram = BTreeMap::<String, usize>::new();
        let mut duration_confusion = BTreeMap::<String, usize>::new();
        for (_, duration) in &observed {
            *observed_duration_histogram
                .entry(duration.clone())
                .or_default() += 1;
        }
        for ((_, want), (_, got)) in expected.iter().zip(&observed) {
            *duration_confusion
                .entry(format!("{want}->{got}"))
                .or_default() += 1;
        }
        let aligned = expected.len().min(observed.len());
        let pitch_exact = expected
            .iter()
            .zip(&observed)
            .filter(|((want, _), (got, _))| want == got)
            .count();
        let duration_exact = expected
            .iter()
            .zip(&observed)
            .filter(|((_, want), (_, got))| want == got)
            .count();
        eprintln!(
            "{}",
            serde_json::json!({
                "schema": "franken_ocr.tromr.retained_row_diagnostic.v1",
                "quant": quant,
                "model_sha256": model_sha256,
                "input_variant": input.variant,
                "source_png_sha256": diagnostic_sha256(&input.png),
                "source_gray8_sha256": diagnostic_sha256(input.crop.pixels()),
                "source_dimensions": [input.crop.width(), input.crop.height()],
                "row_refinement_angle_millidegrees":
                    input.row_refinement_angle_millidegrees,
                "preprocessed_width": preprocessed_width,
                "recognition_options": options,
                "recognition_options_identity": options.replay_identity().expect("options identity"),
                "finite_execution_options": execution_options,
                "elapsed_ms": elapsed_ms,
                "execution_diagnostics": diagnostics,
                "expected_note_count": expected.len(),
                "observed_note_count": observed.len(),
                "aligned_note_count": aligned,
                "pitch_exact": pitch_exact,
                "duration_exact": duration_exact,
                "observed_duration_histogram": observed_duration_histogram,
                "duration_confusion": duration_confusion,
                "semantic": result.semantic,
                "teacher_forced_rhythm": teacher_forced_rhythm,
                "native_full_logits_manifest": native_full_logits_manifest,
                "upstream_comparison": upstream_comparison,
            })
        );
    }

    #[test]
    fn diagnostic_full_logits_binary_contract_is_stable() {
        let bytes = diagnostic_f32_le_bytes(&[0.0, 1.0, -2.5]);
        assert_eq!(
            bytes,
            vec![
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x20, 0xc0
            ]
        );
        assert_eq!(
            diagnostic_sha256(&bytes),
            "4356516ed57de986ba8080c557e8856871336d6a17b170fb946df125605466c9"
        );
    }

    fn fixture_tokenizer() -> crate::tokenizer::music::MusicTokenizer {
        crate::tokenizer::music::MusicTokenizer::from_dir(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tromr"),
        )
        .expect("committed tables load")
    }

    fn synthetic_music_result(label: &str) -> MusicResult {
        synthetic_music_result_with_positions(label, 1)
    }

    fn synthetic_music_result_with_positions(semantic: &str, position_count: usize) -> MusicResult {
        let options = TromrRecognitionOptionsV1::deterministic();
        MusicResult {
            semantic: semantic.to_owned(),
            musicxml: format!("<score-partwise>{semantic}</score-partwise>"),
            options,
            options_identity: options.replay_identity().expect("options identity"),
            forward_candidate_lattices: vec![
                TromrForwardCandidateLatticeV1::new(
                    0,
                    synthetic_candidate_lattice_for_test(position_count, true),
                )
                .expect("synthetic forward candidate"),
            ],
        }
    }

    fn synthetic_staff_evidence(index: usize, bbox: StaffBBox) -> StaffInferenceEvidence {
        let geometry = crate::preprocess::staff_detect::StaffCropGeometry::unpadded(bbox);
        let review_lines = [10, 20, 30, 40, 50];
        let gray8 = crate::preprocess::staff_detect::TromrGray8CropV1::from_tightly_packed(
            vec![255; bbox.2 * bbox.3],
            bbox.2,
            bbox.3,
        )
        .expect("synthetic row pixels");
        StaffInferenceEvidence {
            index,
            geometry,
            route: TromrRowInferenceRouteV1::DetectedStaffCrop,
            forward_inputs: vec![TromrForwardInputV1 {
                gray8: gray8.clone(),
                source_space: TromrModelInputSourceSpaceV1::ReviewCropCanvas,
                source_bbox_xywh: (0, 0, bbox.2, bbox.3),
                padding: crate::preprocess::staff_detect::StaffPadding::default(),
                staff_lines_y_in_canvas: Some(review_lines),
            }],
            review_crop_gray8: Some(gray8),
            review_crop_geometry: Some(geometry),
            staff_lines: Some(TromrStaffLineEvidenceV1 {
                accepted_detector_lines_y_in_globally_deskewed_raster: review_lines
                    .map(|line| bbox.1 + line),
                review_crop_staff_lines_y_in_canvas: review_lines,
            }),
            outcome: StaffInferenceOutcome::Skipped,
            reason: None,
        }
    }

    fn synthetic_page_recognition_with_retained_geometry(
        detected_staff_count: usize,
    ) -> PageRecognition {
        let raster_width = 160;
        let raster_height = detected_staff_count.max(1) * 80 + 20;
        let detector_crops = (0..detected_staff_count)
            .map(|index| {
                let bbox = (0, 10 + index * 80, raster_width, 60);
                let review_lines = [10, 20, 30, 40, 50];
                (
                    crate::preprocess::staff_detect::StaffCropGeometry::unpadded(bbox),
                    review_lines.map(|row| row + bbox.1),
                    review_lines,
                )
            })
            .collect::<Vec<_>>();
        let (staff_detection, retained_staff_detection) =
            synthetic_staff_detection_pair_for_test(raster_width, raster_height, &detector_crops);
        let attempt_count = detected_staff_count.max(1);
        let mut staves = Vec::with_capacity(attempt_count);
        let mut staff_evidence = Vec::with_capacity(attempt_count);
        for index in 0..attempt_count {
            let selected_geometry = crate::preprocess::staff_detect::StaffCropGeometry::unpadded((
                0,
                0,
                raster_width,
                raster_height,
            ));
            let selected_input = || TromrForwardInputV1 {
                gray8: retained_staff_detection.selected_page.gray8().clone(),
                source_space: TromrModelInputSourceSpaceV1::SelectedPageRaster,
                source_bbox_xywh: (0, 0, raster_width, raster_height),
                padding: crate::preprocess::staff_detect::StaffPadding::default(),
                staff_lines_y_in_canvas: None,
            };
            let (bbox, evidence) = if detected_staff_count == 0 {
                (
                    (0, 0, raster_width, raster_height),
                    StaffInferenceEvidence {
                        index: 0,
                        geometry: selected_geometry,
                        route: TromrRowInferenceRouteV1::NoDetectedStaffWholeRasterFallback,
                        forward_inputs: vec![selected_input()],
                        review_crop_gray8: None,
                        review_crop_geometry: None,
                        staff_lines: None,
                        outcome: StaffInferenceOutcome::Recognized,
                        reason: None,
                    },
                )
            } else {
                let retained = &retained_staff_detection.crops[index];
                let detector = &staff_detection.crops[index];
                let route = if detected_staff_count == 1 {
                    TromrRowInferenceRouteV1::SingleDetectedStaffWholeRaster
                } else {
                    TromrRowInferenceRouteV1::DetectedStaffCrop
                };
                let forward_inputs = if detected_staff_count == 1 {
                    vec![selected_input()]
                } else {
                    vec![TromrForwardInputV1 {
                        gray8: retained.review_canvas.gray8().clone(),
                        source_space: TromrModelInputSourceSpaceV1::ReviewCropCanvas,
                        source_bbox_xywh: (
                            0,
                            0,
                            retained.review_canvas.gray8().width(),
                            retained.review_canvas.gray8().height(),
                        ),
                        padding: retained.padding_transform.padding,
                        staff_lines_y_in_canvas: Some(retained.review_canvas_staff_lines.y_rows),
                    }]
                };
                (
                    detector.geometry.source_bbox,
                    StaffInferenceEvidence {
                        index,
                        geometry: if detected_staff_count == 1 {
                            selected_geometry
                        } else {
                            detector.geometry
                        },
                        route,
                        forward_inputs,
                        review_crop_gray8: Some(retained.review_canvas.gray8().clone()),
                        review_crop_geometry: Some(detector.geometry),
                        staff_lines: Some(TromrStaffLineEvidenceV1 {
                            accepted_detector_lines_y_in_globally_deskewed_raster: retained
                                .globally_deskewed_staff_lines
                                .y_rows,
                            review_crop_staff_lines_y_in_canvas: retained
                                .review_canvas_staff_lines
                                .y_rows,
                        }),
                        outcome: StaffInferenceOutcome::Recognized,
                        reason: None,
                    },
                )
            };
            staves.push((index, synthetic_music_result("clef-G2"), bbox));
            staff_evidence.push(evidence);
        }
        let options = TromrRecognitionOptionsV1::deterministic();
        PageRecognition {
            detected_staff_count,
            staff_segmentation_disposition:
                TromrStaffSegmentationDispositionV1::for_detected_staff_count(detected_staff_count),
            staves,
            skips: Vec::new(),
            staff_evidence,
            staff_detection,
            retained_staff_detection,
            options,
            options_identity: options.replay_identity().expect("test options identity"),
        }
    }

    #[test]
    fn page_recognition_maps_zero_single_and_multi_attempts_to_retained_geometry() {
        for detected_staff_count in [0, 1, 3] {
            let page = synthetic_page_recognition_with_retained_geometry(detected_staff_count);
            page.validate_retained_geometry()
                .expect("retained page geometry validates");
            for attempt_position in 0..detected_staff_count.max(1) {
                let retained = page
                    .retained_geometry_for_staff_attempt(attempt_position)
                    .expect("attempt position resolves");
                if detected_staff_count == 0 {
                    assert!(retained.is_none());
                } else {
                    assert_eq!(
                        retained.expect("detector-backed attempt").crop_index,
                        attempt_position
                    );
                }
            }
            assert!(
                page.retained_geometry_for_staff_attempt(detected_staff_count.max(1))
                    .is_err()
            );
        }
    }

    #[test]
    fn page_recognition_rejects_retained_geometry_and_attempt_census_drift() {
        let valid = synthetic_page_recognition_with_retained_geometry(2);

        let mut crop_rect = synthetic_page_recognition_with_retained_geometry(2);
        crop_rect.retained_staff_detection.crops[0]
            .crop_transform
            .source_rect
            .x += 1;
        assert!(crop_rect.validate_retained_geometry().is_err());

        let mut duplicate_index = synthetic_page_recognition_with_retained_geometry(2);
        duplicate_index.staff_evidence[1].index = 0;
        assert!(duplicate_index.validate_retained_geometry().is_err());
        assert!(
            duplicate_index
                .retained_geometry_for_staff_attempt(0)
                .is_err()
        );

        let mut review_identity = valid;
        review_identity.staff_detection.crops[0]
            .review_crop_gray8
            .pixels_sha256[0] ^= 1;
        assert!(review_identity.validate_retained_geometry().is_err());

        let mut forwarded_pixels = synthetic_page_recognition_with_retained_geometry(2);
        let input = &mut forwarded_pixels.staff_evidence[0].forward_inputs[0];
        let (width, height) = (input.gray8.width(), input.gray8.height());
        let mut pixels = input.gray8.pixels().to_vec();
        pixels[0] ^= 1;
        input.gray8 = crate::preprocess::staff_detect::TromrGray8CropV1::from_tightly_packed(
            pixels, width, height,
        )
        .expect("mutated but internally consistent Gray8 input");
        assert!(forwarded_pixels.validate_retained_geometry().is_err());
    }

    #[test]
    fn finish_page_recognition_preserves_the_validated_retained_chain() {
        let mut page = synthetic_page_recognition_with_retained_geometry(2);
        let (skipped_index, _, skipped_bbox) = page.staves.pop().expect("second synthetic staff");
        let skipped_reason = "synthetic decoder refusal".to_owned();
        page.skips.push(StaffSkip {
            index: skipped_index,
            bbox: skipped_bbox,
            reason: skipped_reason.clone(),
        });
        page.staff_evidence[skipped_index].outcome = StaffInferenceOutcome::Skipped;
        page.staff_evidence[skipped_index].reason = Some(skipped_reason);
        let finished = finish_page_recognition(
            page.detected_staff_count,
            page.staves,
            page.skips,
            page.staff_evidence,
            PageStaffDetectionArtifacts {
                pixel_free: page.staff_detection,
                retained: page.retained_staff_detection,
            },
            page.options,
            page.options_identity,
        )
        .expect("multi-staff page assembly");
        finished
            .validate_retained_geometry()
            .expect("finished page retains exact geometry");
        assert_eq!(
            finished
                .retained_geometry_for_staff_attempt(1)
                .expect("second attempt maps")
                .expect("second detector crop")
                .crop_index,
            1
        );
        assert_eq!(finished.skips[0].index, 1);
    }

    #[test]
    fn seeded_top_k_cutoff_matches_upstream_at_vocab_boundaries() {
        for (vocab_size, expected_kept) in [
            (260, 26),
            (71, 8),
            (11, 2),
            (10, 1),
            (9, 1),
            (7, 1),
            (2, 1),
            (1, 1),
        ] {
            let logits = (0..vocab_size).map(|id| id as f32).collect::<Vec<_>>();
            let kept = upstream_top_k_indices(&logits);
            assert_eq!(kept.len(), expected_kept, "vocabulary size {vocab_size}");
            assert_eq!(kept[0], vocab_size - 1);
            assert_eq!(kept[expected_kept - 1], vocab_size - expected_kept);
            if vocab_size > expected_kept {
                assert!(
                    !kept.contains(&(vocab_size - expected_kept - 1)),
                    "rank {} is outside the upstream kept set",
                    expected_kept + 1
                );
            }
        }
    }

    #[test]
    fn seeded_sampling_never_selects_the_first_excluded_rhythm_logit() {
        let logits = (0..260).map(|id| id as f32 / 100.0).collect::<Vec<_>>();
        let kept = upstream_top_k_indices(&logits);
        assert_eq!(kept.len(), 26);
        assert_eq!(kept.last(), Some(&234));
        assert!(!kept.contains(&233));
        let mut rng = Pcg32::new(0x5eed_cafe);
        for _ in 0..1_024 {
            let sampled = sample_top_k(&logits, &mut rng) as usize;
            assert!(kept.contains(&sampled), "sampled excluded id {sampled}");
        }
    }

    #[test]
    fn candidate_lattice_ranking_reuses_diagnostic_ties_and_rejects_nonfinite_scores() {
        assert_eq!(
            ranked_candidate_ids(&[1.0, 3.0, 3.0, -2.0]).expect("finite scores rank"),
            vec![1, 2, 0, 3]
        );
        assert_eq!(
            ranked_candidate_ids(&[f32::MAX, -f32::MAX, f32::MIN_POSITIVE, -0.0, 0.0,])
                .expect("extreme finite scores rank"),
            vec![0, 2, 4, 3, 1]
        );
        for invalid in [Vec::new(), vec![0.0, f32::NAN], vec![f32::INFINITY]] {
            assert!(ranked_candidate_ids(&invalid).is_err());
        }
    }

    #[test]
    fn candidate_lattice_retains_exact_top_five_and_full_chosen_rank() {
        let logits = (0..260)
            .map(|token_id| -(token_id as f32))
            .collect::<Vec<_>>();
        let ranked = ranked_candidate_ids(&logits).expect("scores rank");
        let evidence =
            capture_candidate_head_evidence(TromrCandidateHeadV1::Rhythm, &logits, 20, &ranked)
                .expect("evidence captures");
        assert_eq!(evidence.chosen_rank_one_based, 21);
        assert_eq!(evidence.chosen_model_score(), -20.0);
        assert_eq!(evidence.chosen_minus_best_alternative_model_score(), -20.0);
        assert_eq!(evidence.truncated_candidate_count, 255);
        assert_eq!(
            evidence
                .retained_top_candidates
                .iter()
                .map(|candidate| (
                    candidate.token_id,
                    candidate.rank_one_based,
                    candidate.model_score(),
                ))
                .collect::<Vec<_>>(),
            vec![
                (0, 1, 0.0),
                (1, 2, -1.0),
                (2, 3, -2.0),
                (3, 4, -3.0),
                (4, 5, -4.0)
            ]
        );
    }

    #[test]
    fn candidate_lattice_capture_preserves_sampling_stream_and_rng_position() {
        let cases = [
            (
                TromrCandidateHeadV1::Rhythm,
                (0..260).map(|id| id as f32 / 1_000.0).collect::<Vec<_>>(),
            ),
            (
                TromrCandidateHeadV1::Pitch,
                (0..71).map(|id| id as f32 / 1_000.0).collect::<Vec<_>>(),
            ),
            (
                TromrCandidateHeadV1::Lift,
                (0..7).map(|id| id as f32 / 1_000.0).collect::<Vec<_>>(),
            ),
        ];
        let seed = 0x0dd5_c0de_5eed_u64;
        let mut legacy_rng = Pcg32::new(seed);
        let legacy_ids = cases
            .iter()
            .map(|(_, logits)| sample_top_k(logits, &mut legacy_rng))
            .collect::<Vec<_>>();
        let mut evidence_rng = Some(Pcg32::new(seed));
        let evidence_ids = cases
            .iter()
            .map(|(head, logits)| {
                select_and_capture_candidate_head(*head, logits, &mut evidence_rng)
                    .expect("same-forward evidence captures")
                    .0
            })
            .collect::<Vec<_>>();
        assert_eq!(evidence_ids, legacy_ids);
        assert_eq!(
            evidence_rng.as_mut().expect("sampling rng").next_u32(),
            legacy_rng.next_u32(),
            "candidate sorting and capture must consume no RNG values"
        );
    }

    #[test]
    fn candidate_lattice_argmax_keeps_legacy_signed_zero_selection() {
        let mut logits = vec![-100.0; TromrCandidateHeadV1::Rhythm.vocabulary_size()];
        logits[3] = -0.0;
        logits[4] = 0.0;
        let (chosen, evidence) =
            select_and_capture_candidate_head(TromrCandidateHeadV1::Rhythm, &logits, &mut None)
                .expect("evidence captures");
        assert_eq!(chosen, 3, "legacy strict-greater fold keeps the first zero");
        assert_eq!(evidence.chosen_rank_one_based, 2);
        assert_eq!(evidence.retained_top_candidates[0].token_id, 4);
        assert_eq!(evidence.retained_top_candidates[1].token_id, 3);
    }

    #[test]
    fn candidate_lattice_canonical_contract_detects_mutation_and_is_bounded() {
        let lattice = synthetic_candidate_lattice_for_test(1, true);
        lattice
            .validate()
            .expect("one-position EOS lattice validates");
        let canonical = lattice.canonical_bytes().expect("canonical bytes");
        assert_eq!(
            diagnostic_sha256(&canonical),
            "8ea66b9ce3bb43f3536cf1504f976e4d5bc3fe1cfab27ee67caaa2f8589708a5"
        );
        assert_eq!(canonical, lattice.canonical_bytes().expect("bytes replay"));
        assert_eq!(
            lattice.canonical_sha256().expect("digest"),
            Sha256::digest(&canonical).as_slice()
        );
        let round_trip: TromrCandidateLatticeV1 =
            serde_json::from_str(&serde_json::to_string(&lattice).expect("lattice serializes"))
                .expect("lattice deserializes");
        assert_eq!(
            round_trip.canonical_bytes().expect("round-trip bytes"),
            canonical
        );

        let mut bad_prefix = lattice.clone();
        bad_prefix.positions[0].prefix_sha256[0] ^= 1;
        assert_eq!(
            bad_prefix
                .validate()
                .expect_err("prefix mutation fails")
                .kind(),
            "format_mismatch"
        );
        let mut bad_eos = lattice.clone();
        bad_eos.positions[0].rhythm_emitted_eos = false;
        assert!(bad_eos.validate().is_err());
        let mut bad_order = lattice;
        bad_order.positions[0].heads.swap(0, 1);
        assert!(bad_order.validate().is_err());

        let capped = synthetic_candidate_lattice_for_test(MAX_SEQ, false);
        capped
            .validate()
            .expect("MAX_SEQ lattice validates without EOS");
        assert!(
            capped
                .canonical_bytes()
                .expect("bounded maximum lattice")
                .len()
                <= TROMR_MAX_CANDIDATE_LATTICE_CANONICAL_BYTES
        );
    }

    #[test]
    fn split_candidate_lattices_preserve_independent_prefixes_semantics_and_order() {
        let first_semantic = "clef-G2+keySignature-CM+timeSignature-4/4+note-C4_quarter";
        let second_semantic = "clef-F4+keySignature-GM+timeSignature-3/4+note-D4_half+barline";
        let expected_semantic = concat!(
            "clef-G2+keySignature-CM+timeSignature-4/4+note-C4_quarter",
            "+barline+note-D4_half+barline"
        );

        let assemble = || {
            let mut semantic = String::new();
            let mut candidates = Vec::new();
            assert_eq!(
                append_split_music_result(
                    &mut semantic,
                    &mut candidates,
                    0,
                    Ok(synthetic_music_result_with_positions(first_semantic, 2)),
                )
                .expect("first split segment appends"),
                first_semantic.len()
            );
            assert_eq!(
                append_split_music_result(
                    &mut semantic,
                    &mut candidates,
                    1,
                    Ok(synthetic_music_result_with_positions(second_semantic, 3)),
                )
                .expect("second split segment appends"),
                second_semantic.len()
            );
            (semantic, candidates)
        };

        let (semantic, candidates) = assemble();
        assert_eq!(
            semantic, expected_semantic,
            "legacy semantic splice drifted"
        );
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.forward_input_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(candidates[0].candidate_lattice.positions.len(), 2);
        assert_eq!(candidates[1].candidate_lattice.positions.len(), 3);
        assert_eq!(
            candidates[0].candidate_lattice.positions[0].prefix_length,
            1
        );
        assert_eq!(
            candidates[1].candidate_lattice.positions[0].prefix_length, 1,
            "a split continuation must restart from the seed prefix"
        );
        assert_eq!(
            candidates[0].candidate_lattice.positions[0].prefix_sha256,
            candidates[1].candidate_lattice.positions[0].prefix_sha256,
            "each independent split forward starts from the exact same seed triple"
        );
        let options = TromrRecognitionOptionsV1::deterministic();
        let result = MusicResult {
            semantic: semantic.clone(),
            musicxml: format!("<score-partwise>{semantic}</score-partwise>"),
            options,
            options_identity: options.replay_identity().expect("options identity"),
            forward_candidate_lattices: candidates.clone(),
        };
        result
            .validate_forward_candidate_lattices(2)
            .expect("two split forwards validate in order");

        let replay = assemble();
        assert_eq!(replay.0, semantic);
        assert_eq!(
            replay.1, candidates,
            "split evidence order must replay exactly"
        );
    }

    #[test]
    fn split_candidate_append_is_atomic_under_cancellation_and_refuses_reordering() {
        let mut semantic = String::new();
        let mut candidates = Vec::new();
        append_split_music_result(
            &mut semantic,
            &mut candidates,
            0,
            Ok(synthetic_music_result_with_positions(
                "clef-G2+note-C4_quarter",
                2,
            )),
        )
        .expect("first split segment appends");
        let before_semantic = semantic.clone();
        let before_candidates = candidates.clone();

        let error =
            append_split_music_result(&mut semantic, &mut candidates, 1, Err(FocrError::Cancelled))
                .expect_err("cancelled segment must terminate split assembly");
        assert!(matches!(error, FocrError::Cancelled));
        assert_eq!(semantic, before_semantic);
        assert_eq!(candidates, before_candidates);

        let error = append_split_music_result(
            &mut semantic,
            &mut candidates,
            2,
            Ok(synthetic_music_result_with_positions(
                "clef-F4+note-C3_half",
                1,
            )),
        )
        .expect_err("a skipped forward-input index must be refused");
        assert_eq!(error.kind(), "format_mismatch");
        assert_eq!(semantic, before_semantic);
        assert_eq!(candidates, before_candidates);
    }

    #[test]
    fn diagnostic_oracle_streams_replace_only_note_targets() {
        use crate::tokenizer::music::{EOS_ID, Stream};

        let tokenizer = fixture_tokenizer();
        let truth = serde_json::json!({
            "measures": [{
                "one_based_measure": 1,
                "notes": [
                    {
                        "measure_note": 1,
                        "pitch": {"step": "E", "alter": 0, "octave": 5},
                        "written_duration": "eighth"
                    },
                    {
                        "measure_note": 2,
                        "pitch": {"step": "D", "alter": 0, "octave": 5},
                        "written_duration": "sixteenth"
                    }
                ]
            }]
        });
        let targets = diagnostic_truth_note_targets(&truth, &tokenizer);
        let observed = MusicStreams {
            rhythm: vec![15, 21, 143, 143, 5, EOS_ID],
            pitch: vec![0, 0, 40, 40, 0, 0],
            lift: vec![0, 0, 1, 1, 0, 0],
        };
        let oracle = diagnostic_oracle_streams(&tokenizer, &observed, &targets);
        assert_eq!(oracle.barline_count, 1);
        assert_eq!(
            oracle.note_target_by_stream_position,
            vec![None, None, Some(0), Some(1), None, None]
        );
        assert_eq!(oracle.rhythm_only.rhythm, vec![15, 21, 131, 139, 5, EOS_ID]);
        assert_eq!(oracle.rhythm_only.pitch, observed.pitch);
        assert_eq!(oracle.rhythm_only.lift, observed.lift);
        assert_eq!(
            tokenizer.token(Stream::Pitch, oracle.all_streams.pitch[2]),
            Some("note-E5")
        );
        assert_eq!(
            tokenizer.token(Stream::Pitch, oracle.all_streams.pitch[3]),
            Some("note-D5")
        );
        assert_eq!(
            tokenizer.token(Stream::Lift, oracle.all_streams.lift[2]),
            Some("lift_null")
        );
        assert_eq!(&oracle.all_streams.rhythm[..2], &[15, 21]);
        assert_eq!(&oracle.all_streams.rhythm[4..], &[5, EOS_ID]);
    }

    #[test]
    fn recognition_options_default_is_explicit_stable_argmax_contract() {
        let options = TromrRecognitionOptionsV1::default();
        assert_eq!(options.schema_version, 1);
        assert_eq!(options.decode_mode, TromrDecodeModeV1::Argmax);
        assert_eq!(options.seed, None);
        assert_eq!(options.split_policy, TromrSplitPolicyV1::Disabled);
        assert_eq!(
            options.staff_resampler,
            TromrStaffResamplerV1::Cv2LinearU8V1
        );
        assert_eq!(
            options.canonical_json().expect("default serializes"),
            r#"{"schema_version":1,"decode_mode":"argmax","seed":null,"split_policy":"disabled","staff_resampler":"cv2_linear_u8_v1"}"#
        );
        assert_eq!(
            options.replay_identity().expect("identity is stable"),
            options.replay_identity().expect("identity replays")
        );
        assert!(matches!(
            options.decode_pick().expect("default pick"),
            DecodePick::Argmax
        ));
    }

    #[test]
    fn recognition_options_sampling_and_split_are_explicit_and_replayable() {
        let sampled = TromrRecognitionOptionsV1 {
            decode_mode: TromrDecodeModeV1::SeededTopKTemperature,
            seed: Some(0xdec0_de01),
            split_policy: TromrSplitPolicyV1::ExperimentalBarlineSegments,
            ..TromrRecognitionOptionsV1::deterministic()
        };
        assert_eq!(
            TromrRecognitionOptionsV1::from_json(
                &sampled.canonical_json().expect("sampling serializes")
            )
            .expect("sampling round trips"),
            sampled
        );
        assert!(matches!(
            sampled.decode_pick().expect("sampling pick"),
            DecodePick::SeededSample { seed: 0xdec0_de01 }
        ));

        let different_seed = TromrRecognitionOptionsV1 {
            seed: Some(0xdec0_de02),
            ..sampled
        };
        let split_disabled = TromrRecognitionOptionsV1 {
            split_policy: TromrSplitPolicyV1::Disabled,
            ..sampled
        };
        let base_identity = sampled.replay_identity().expect("base identity");
        assert_ne!(
            base_identity,
            different_seed.replay_identity().expect("seed identity")
        );
        assert_ne!(
            base_identity,
            split_disabled.replay_identity().expect("split identity")
        );
    }

    #[test]
    fn recognition_options_refuse_invalid_seed_combinations_with_usage_kind() {
        for options in [
            TromrRecognitionOptionsV1 {
                seed: Some(7),
                ..TromrRecognitionOptionsV1::deterministic()
            },
            TromrRecognitionOptionsV1 {
                decode_mode: TromrDecodeModeV1::SeededTopKTemperature,
                seed: None,
                ..TromrRecognitionOptionsV1::deterministic()
            },
        ] {
            let error = options.validate().expect_err("invalid seed contract");
            assert_eq!(error.kind(), "usage");
            assert_eq!(error.exit_code(), crate::error::EXIT_USAGE);
        }
    }

    #[test]
    fn recognition_options_refuse_unknown_schema_fields_and_enum_values() {
        let cases = [
            r#"{"schema_version":2,"decode_mode":"argmax","seed":null,"split_policy":"disabled","staff_resampler":"cv2_linear_u8_v1"}"#,
            r#"{"schema_version":1,"decode_mode":"beam_search","seed":null,"split_policy":"disabled","staff_resampler":"cv2_linear_u8_v1"}"#,
            r#"{"schema_version":1,"decode_mode":"argmax","seed":null,"split_policy":"guess","staff_resampler":"cv2_linear_u8_v1"}"#,
            r#"{"schema_version":1,"decode_mode":"argmax","seed":null,"split_policy":"disabled","staff_resampler":"catmull_rom"}"#,
            r#"{"schema_version":1,"decode_mode":"argmax","seed":null,"split_policy":"disabled","staff_resampler":"cv2_linear_u8_v1","ambient":true}"#,
        ];
        for json in cases {
            let error = TromrRecognitionOptionsV1::from_json(json)
                .expect_err("unknown or unsupported option must refuse");
            assert_eq!(error.kind(), "format_mismatch", "input: {json}");
            assert_eq!(
                error.exit_code(),
                crate::error::EXIT_FORMAT_MISMATCH,
                "input: {json}"
            );
        }
    }

    #[test]
    fn tromr_production_core_has_no_ambient_environment_reads() {
        let source = include_str!("tromr.rs");
        let production = source
            .split_once("#[cfg(test)]")
            .expect("test module boundary")
            .0;
        assert!(!production.contains("std::env"));
        for legacy in [
            "FOCR_TROMR_SAMPLE",
            "FOCR_TROMR_SEED",
            "FOCR_TROMR_SPLIT",
            "FOCR_RESAMPLE",
        ] {
            assert!(
                !production.contains(legacy),
                "core must not inspect {legacy}; only the CLI boundary may parse it"
            );
        }
    }

    /// merge_semantic vs the UPSTREAM inference.py merge run over the SAME
    /// oracle argmax streams (golden generated 2026-07-05 in the pinned venv;
    /// the oracle streams live in tromr_oracle_fixtures.json — 42 ids/stream,
    /// rhythm trailing [EOS] stripped, upstream len 41/42/42 alignment holds
    /// because the only special is trailing).
    #[test]
    fn merge_semantic_matches_upstream_golden() {
        let tk = fixture_tokenizer();
        // The oracle argmax streams for examples/1.png (fixture copy — the
        // armed cert already proves our generate emits exactly these).
        let rhythm: Vec<u32> = vec![
            15, 21, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 5, 131, 131, 131, 131,
            131, 131, 131, 131, 131, 131, 131, 131, 5, 131, 131, 131, 131, 131, 131, 131, 131, 131,
            131, 131, 131, 5, 2,
        ];
        let pitch: Vec<u32> = vec![
            0, 0, 0, 0, 38, 39, 40, 41, 42, 43, 0, 38, 39, 40, 41, 40, 40, 41, 42, 43, 40, 38, 40,
            40, 40, 40, 0, 0, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 0, 0,
        ];
        let lift: Vec<u32> = vec![
            0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
        ];
        // NOTE: these literals are a FROZEN realistic stream (the 2026-07-05
        // oracle run, pre-DISC-007) paired with the upstream-merge golden
        // below — a self-consistent synthetic case pinning the MERGE math.
        // The ARMED cert covers the live fixture; this one never regenerates.
        let streams = MusicStreams {
            rhythm,
            pitch,
            lift,
        };
        let merged = merge_semantic(&tk, &streams).expect("merge runs");
        assert!(
            merged
                .starts_with("clef-G2+keySignature-CM+nonote_eighth+nonote_eighth+note-E5_eighth"),
            "{merged}"
        );
        assert!(
            merged.ends_with("barline"),
            "trailing EOS stripped: {merged}"
        );
        assert_eq!(merged.matches("barline").count(), 3, "{merged}");
        assert!(!merged.contains("[EOS]"), "{merged}");
    }

    #[test]
    fn merge_semantic_edges() {
        let tk = fixture_tokenizer();
        // Chord: rhythm [note-eighth(131), |(4), note-eighth] pitches C4/E4.
        let streams = MusicStreams {
            rhythm: vec![131, 4, 131],
            pitch: vec![29, 0, 31],
            lift: vec![1, 0, 3], // lift_null, nonote, lift_#
        };
        let merged = merge_semantic(&tk, &streams).expect("chord merges");
        // One event: first note, '|', second note with '#' attached.
        let p29 = tk
            .token(crate::tokenizer::music::Stream::Pitch, 29)
            .unwrap();
        let p31 = tk
            .token(crate::tokenizer::music::Stream::Pitch, 31)
            .unwrap();
        assert_eq!(merged, format!("{p29}_eighth|{p31}#_eighth"));

        // Mid-stream EOS is a decode error, not a skip.
        let bad = MusicStreams {
            rhythm: vec![131, 2, 131],
            pitch: vec![29, 0, 31],
            lift: vec![1, 0, 1],
        };
        assert!(
            merge_semantic(&tk, &bad).is_err(),
            "mid-stream EOS must fail loud"
        );

        // Length mismatch fails loud.
        let bad = MusicStreams {
            rhythm: vec![131],
            pitch: vec![29, 30],
            lift: vec![1],
        };
        assert!(merge_semantic(&tk, &bad).is_err());

        // Leading '|' (chord with no head) fails loud.
        let bad = MusicStreams {
            rhythm: vec![4, 131],
            pitch: vec![0, 29],
            lift: vec![0, 1],
        };
        assert!(merge_semantic(&tk, &bad).is_err());
    }

    #[test]
    fn musicxml_serializes_the_vocabulary() {
        let xml = semantic_to_musicxml(
            "clef-G2+keySignature-EbM+timeSignature-3/4+note-F4#_quarter.+note-C5_eighth|note-E5N_eighth+rest-half+barline+multirest-2+nonote_eighth",
        )
        .expect("serializes");
        for want in [
            "<divisions>64</divisions>",
            "<clef><sign>G</sign><line>2</line></clef>",
            "<key><fifths>-3</fifths></key>",
            "<time><beats>3</beats><beat-type>4</beat-type></time>",
            // dotted quarter with sharp: 64*1.5 = 96 ticks
            "<pitch><step>F</step><alter>1</alter><octave>4</octave></pitch><duration>96</duration><type>quarter</type><dot/>",
            // chord second note carries <chord/> + natural accidental
            "<chord/><pitch><step>E</step><alter>0</alter><octave>5</octave></pitch>",
            "<accidental>natural</accidental>",
            "<rest/><duration>128</duration><type>half</type>",
            "<rest measure=\"yes\"/>",
            "<measure number=\"4\">",
        ] {
            assert!(xml.contains(want), "missing {want:?} in:\n{xml}");
        }
        // multirest-2 = two of the whole-measure rests.
        assert_eq!(xml.matches("rest measure=\"yes\"").count(), 2);
        // The C/ cut-time + unknown-token error paths.
        assert!(semantic_to_musicxml("timeSignature-C/+note-C4_whole").is_ok());
        assert!(semantic_to_musicxml("garbage-token_xyz").is_err());
        assert!(semantic_to_musicxml("note-C4_gigasecond").is_err());
    }

    /// bd-av64.1 (vocab-exhaustive gate): EVERY rhythm-vocab token must
    /// flow through the XML emitter. The 2026-07-06 Cadwallader run
    /// crashed on a pitched `thirty_second` because `rsplit_once('_')`
    /// split inside the duration name, and this test's first run also
    /// caught `rest-256th`/`rest-512th` missing from the duration table —
    /// goldens never decoded either class.
    #[test]
    fn every_rhythm_vocab_token_renders_to_musicxml() {
        let raw = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/tromr/tokenizer_rhythm.json"),
        )
        .expect("vocab fixture reads");
        let json: serde_json::Value = serde_json::from_str(&raw).expect("vocab fixture parses");
        let vocab = json["model"]["vocab"].as_object().expect("vocab map");
        assert!(
            vocab.len() >= 200,
            "vocab unexpectedly small: {}",
            vocab.len()
        );
        // Every lift form the merge can append to a pitch head.
        let lifts = ["", "##", "#", "bb", "b", "N"];
        for token in vocab.keys() {
            let semantics: Vec<String> = match token.as_str() {
                // Stream controls never reach the emitter.
                "[PAD]" | "[BOS]" | "[EOS]" | "+" | "|" => continue,
                t if t.starts_with("note-") => {
                    let dur = &t["note-".len()..];
                    // The merge renders `{pitch}{lift}_{dur}` plus the
                    // pitch-head-abstained `nonote_{dur}` form.
                    lifts
                        .iter()
                        .map(|l| format!("clef-G2+note-C4{l}_{dur}+barline"))
                        .chain([format!("clef-G2+nonote_{dur}+barline")])
                        .collect()
                }
                t => vec![format!("clef-G2+{t}+note-C4_quarter+barline")],
            };
            for s in semantics {
                let out = semantic_to_musicxml(&s);
                assert!(
                    out.is_ok(),
                    "vocab token {token:?} failed via {s:?}: {}",
                    out.err().map(|e| e.to_string()).unwrap_or_default()
                );
            }
        }
    }

    /// bd-av64.1 regression: the exact Cadwallader failure atom, and the
    /// full multi-underscore family with tick values.
    #[test]
    fn multi_underscore_durations_parse_exactly() {
        let xml = semantic_to_musicxml(
            "clef-G2+note-B4_thirty_second+note-C5_sixty_fourth.+note-D5_hundred_twenty_eighth+barline",
        )
        .expect("multi-underscore durations render");
        assert!(
            xml.contains("<duration>8</duration><type>32nd</type>"),
            "32nd: {xml}"
        );
        // dotted 64th: 4 * 3/2 = 6 ticks
        assert!(
            xml.contains("<duration>6</duration><type>64th</type><dot/>"),
            "64th.: {xml}"
        );
        assert!(
            xml.contains("<duration>2</duration><type>128th</type>"),
            "128th: {xml}"
        );
        // The numeral-named finest rests (found missing by the vocab gate).
        let xml = semantic_to_musicxml("clef-G2+rest-256th+rest-512th+barline").expect("renders");
        assert!(
            xml.contains("<duration>1</duration><type>256th</type>"),
            "256th: {xml}"
        );
        assert!(
            xml.contains("<duration>1</duration><type>512th</type>"),
            "512th: {xml}"
        );
        // split_pitch_duration: head is untouched, longest duration wins.
        let (head, (xml_type, ticks, dotted)) =
            split_pitch_duration("note-B4_thirty_second").expect("splits");
        assert_eq!(
            (head, xml_type, ticks, dotted),
            ("note-B4", "32nd", 8, false)
        );
    }

    /// bd-av64.3: a '|'-joined group mixing pitched notes and rests must
    /// not emit `<chord/><rest/>` (importer-rejecting); rests drop from
    /// mixed groups, all-rest groups collapse to one rest.
    #[test]
    fn mixed_chord_groups_drop_rests_and_all_rest_groups_collapse() {
        let xml = semantic_to_musicxml("clef-G2+note-C4_eighth|rest-eighth|note-E4_eighth+barline")
            .expect("mixed group renders");
        assert!(
            !xml.contains("<chord/><rest/>"),
            "chord-on-rest leaked: {xml}"
        );
        assert_eq!(xml.matches("<note>").count(), 2, "rests must drop: {xml}");
        assert_eq!(
            xml.matches("<chord/>").count(),
            1,
            "one chord follower: {xml}"
        );

        let xml = semantic_to_musicxml("clef-G2+rest-quarter|rest-quarter+barline")
            .expect("all-rest group renders");
        assert_eq!(
            xml.matches("<note>").count(),
            1,
            "all-rest collapses: {xml}"
        );
        assert!(!xml.contains("<chord/>"), "no chord on the survivor: {xml}");

        // A rest FIRST in a mixed group: the pitched notes still form a
        // legal chord (first pitched note carries no <chord/>).
        let xml = semantic_to_musicxml("clef-G2+rest-eighth|note-C4_eighth|note-E4_eighth+barline")
            .expect("rest-first mixed group renders");
        assert!(validate_musicxml(&xml).is_empty(), "must validate: {xml}");
        assert_eq!(xml.matches("<chord/>").count(), 1);
    }

    /// bd-av64.5: each sanity rule on synthetic streams, including the
    /// classical exemptions (pickup first measure, final measure) and the
    /// mid-stream time-signature reset.
    #[test]
    fn sanity_rules_flag_and_exempt_correctly() {
        let w = |sems: &[&str]| {
            sanity_warnings(&sems.iter().map(|s| s.to_string()).collect::<Vec<_>>())
        };
        // Overfull: 4 quarters in 3/4.
        let ws = w(&[
            "clef-G2+timeSignature-3/4+note-C4_quarter+note-C4_quarter+note-C4_quarter+note-C4_quarter+barline+note-C4_quarter+note-C4_quarter+note-C4_quarter+barline",
        ]);
        assert!(
            ws.iter()
                .any(|x| x.kind == "overfull_bar" && x.measure == 1),
            "{ws:?}"
        );
        // Anacrasis exemption: underfull FIRST measure never flags; a full
        // middle; underfull FINAL exempt.
        let ws = w(&[
            "clef-G2+timeSignature-3/4+note-C4_quarter+barline+note-C4_quarter+note-C4_quarter+note-C4_quarter+barline+note-C4_half+barline",
        ]);
        assert!(ws.is_empty(), "pickup + final exemptions: {ws:?}");
        // Underfull MIDDLE measure flags.
        let ws = w(&[
            "clef-G2+timeSignature-3/4+note-C4_quarter+note-C4_quarter+note-C4_quarter+barline+note-C4_quarter+barline+note-C4_quarter+note-C4_quarter+note-C4_quarter+barline+note-C4_quarter+barline",
        ]);
        assert!(
            ws.iter()
                .any(|x| x.kind == "underfull_bar" && x.measure == 2),
            "{ws:?}"
        );
        // Impossible duration: whole note in 3/4 (the real Cadwallader read).
        let ws = w(&[
            "clef-G2+timeSignature-3/4+note-C4_whole+barline+note-C4_quarter+note-C4_quarter+note-C4_quarter+barline",
        ]);
        assert!(ws.iter().any(|x| x.kind == "impossible_duration"), "{ws:?}");
        // Time-signature change resets the expectation: 2/4 bar after a
        // mid-stream change must NOT flag.
        let ws = w(&[
            "clef-G2+timeSignature-3/4+note-C4_quarter+note-C4_quarter+note-C4_quarter+barline+timeSignature-2/4+note-C4_quarter+note-C4_quarter+barline+note-C4_quarter+note-C4_quarter+barline",
        ]);
        assert!(ws.is_empty(), "time change resets: {ws:?}");
        // Key mismatch across a system: minority staff flags with majority
        // suggestion (the real grand-staff read: -3 vs -1).
        let ws = w(&[
            "clef-G2+keySignature-FM+timeSignature-3/4+note-C4_quarter+note-C4_quarter+note-C4_quarter+barline",
            "clef-F4+keySignature-EbM+timeSignature-3/4+note-C3_quarter+note-C3_quarter+note-C3_quarter+barline",
            "clef-G2+keySignature-FM+timeSignature-3/4+note-C4_quarter+note-C4_quarter+note-C4_quarter+barline",
        ]);
        let km: Vec<_> = ws.iter().filter(|x| x.kind == "key_mismatch").collect();
        assert_eq!(km.len(), 1, "{ws:?}");
        assert_eq!(km[0].part, 2);
        assert!(
            km[0].detail.contains("FM"),
            "majority named: {}",
            km[0].detail
        );
        // Chord groups count once (three-note chord of quarters = ONE beat).
        let ws = w(&[
            "clef-G2+timeSignature-3/4+note-C4_quarter|note-E4_quarter|note-G4_quarter+note-C4_quarter+note-C4_quarter+barline+note-C4_quarter+note-C4_quarter+note-C4_quarter+barline",
        ]);
        assert!(ws.is_empty(), "chords count once: {ws:?}");
    }

    /// bd-av64.5: annotations are comments only — stripping them yields the
    /// exact document emitted before the sanity pass existed (annotate-only
    /// invariant), and annotated documents still validate.
    #[test]
    fn sanity_annotations_are_pure_comments() {
        let xml = semantic_to_musicxml(
            "clef-G2+timeSignature-3/4+note-C4_whole+barline+note-C4_quarter+note-C4_quarter+note-C4_quarter+barline",
        )
        .expect("emits");
        assert!(
            xml.contains("<!--focr-sanity: impossible_duration"),
            "{xml}"
        );
        assert!(
            validate_musicxml(&xml).is_empty(),
            "annotated doc validates"
        );
        let stripped: String = xml
            .lines()
            .filter(|l| !l.trim_start().starts_with("<!--focr-sanity:"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!stripped.contains("focr-sanity"), "comments strip cleanly");
        assert!(stripped.contains("<note>"), "content intact");
    }

    /// bd-av64.3: the structural validator flags each illegal shape (red
    /// fixtures, incl. the frozen 2026-07-06 chord-on-rest shape) and
    /// passes real emitted documents (green).
    #[test]
    fn musicxml_validator_red_and_green() {
        let wrap = |notes: &str| {
            format!(
                "<?xml version=\"1.0\"?><score-partwise version=\"4.0\">\
                 <part-list><score-part id=\"P1\"><part-name>S</part-name></score-part></part-list>\
                 <part id=\"P1\"><measure number=\"1\">{notes}</measure></part></score-partwise>"
            )
        };
        // RED 1: the exact grand.musicxml line-37 shape from the Cadwallader run.
        let bad = wrap(
            "<note><pitch><step>C</step><octave>4</octave></pitch><duration>16</duration></note>\
             <note><chord/><rest/><duration>16</duration></note>",
        );
        assert!(
            validate_musicxml(&bad)
                .iter()
                .any(|v| v.contains("co-occurs")),
            "chord-on-rest must flag: {:?}",
            validate_musicxml(&bad)
        );
        // RED 2: chord note with nothing before it.
        let bad = wrap(
            "<note><chord/><pitch><step>C</step><octave>4</octave></pitch><duration>16</duration></note>",
        );
        assert!(
            validate_musicxml(&bad)
                .iter()
                .any(|v| v.contains("preceded"))
        );
        // RED 3: missing duration.
        let bad = wrap("<note><pitch><step>C</step><octave>4</octave></pitch></note>");
        assert!(
            validate_musicxml(&bad)
                .iter()
                .any(|v| v.contains("missing <duration>"))
        );
        // RED 4: unbalanced tag.
        let bad =
            wrap("<note><pitch><step>C</step><octave>4</octave><duration>16</duration></note>");
        assert!(
            validate_musicxml(&bad)
                .iter()
                .any(|v| v.contains("mismatched"))
        );
        // RED 5: part-list/part id disagreement.
        let bad = "<?xml version=\"1.0\"?><score-partwise version=\"4.0\">\
                   <part-list><score-part id=\"P1\"><part-name>S</part-name></score-part></part-list>\
                   <part id=\"P2\"><measure number=\"1\"></measure></part></score-partwise>";
        assert!(
            validate_musicxml(bad)
                .iter()
                .any(|v| v.contains("do not match"))
        );
        // GREEN: a real emitted document, attributes + chords + multirest.
        let xml = semantic_to_musicxml(
            "clef-F4+keySignature-FM+timeSignature-3/4+note-C4_quarter|note-E4_quarter+rest-quarter+barline+multirest-2+barline+note-F3_half.+barline",
        )
        .expect("emits");
        assert_eq!(validate_musicxml(&xml), Vec::<String>::new());
    }

    fn zoo_dir() -> Option<std::path::PathBuf> {
        let dir = std::env::var_os("FOCR_TROMR_DIR").map(std::path::PathBuf::from)?;
        dir.join("tromr.focrq").is_file().then_some(dir)
    }

    fn read_f32(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path).expect("fixture bin reads");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    fn cos(a: &[f32], b: &[f32]) -> f64 {
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += f64::from(x) * f64::from(y);
            na += f64::from(x) * f64::from(x);
            nb += f64::from(y) * f64::from(y);
        }
        dot / (na.sqrt() * nb.sqrt()).max(1e-30)
    }

    fn maxabs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn width_and_buffer_guards_reject() {
        // Guards fire BEFORE weight access, so a dummy hydration works:
        // synthesize via the error path (no zoo needed).
        let Some(dir) = zoo_dir() else {
            eprintln!("[tromr-test] skip_no_model: FOCR_TROMR_DIR unset (guard leg included)");
            return;
        };
        let weights = Weights::load(&dir.join("tromr.focrq")).expect("artifact loads");
        let w = TromrEncoderW::build(&weights).expect("hydrates");
        // width not ×16, width 0, width > 1280, short buffer — all clean errors.
        assert!(encode(&w, &vec![0.0; IMG_H * 100], 100).is_err());
        assert!(encode(&w, &[], 0).is_err());
        assert!(encode(&w, &vec![0.0; IMG_H * 1296], 1296).is_err());
        assert!(encode(&w, &[0.0; 7], 800).is_err());
    }

    /// The E4 L3 cert (step-0 head logits) + L4 cert (argmax generate
    /// token-exact): the decoder runs over the ORACLE's encoder context
    /// (isolation — the encoder has its own cert), so any divergence is the
    /// decoder's. The oracle's argmax generate is proven deterministic in
    /// the fixture (`argmax_generate_deterministic: true`), so L4 expects
    /// EXACT streams. Model-gated skip-with-SUCCESS.
    #[test]
    fn tromr_decoder_matches_argmax_oracle() {
        let Some(dir) = zoo_dir() else {
            eprintln!("[tromr-test] skip_no_model: FOCR_TROMR_DIR unset");
            return;
        };
        let fx_path = dir.join("tromr_oracle_fixtures.json");
        if !fx_path.is_file() || !dir.join("tromr_seam_head0_rhythm.bin").is_file() {
            eprintln!("[tromr-test] skip_no_model: decoder fixtures absent");
            return;
        }
        let fx: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fx_path).unwrap()).unwrap();
        assert_eq!(
            fx["nondeterminism_floor"]["argmax_generate_deterministic"],
            serde_json::Value::Bool(true),
            "the oracle argmax run must be deterministic for an exact L4 gate"
        );

        let weights = Weights::load(&dir.join("tromr.focrq")).expect("artifact loads");
        let dec = TromrDecoderW::build(&weights).expect("decoder hydrates");
        let ctx_flat = read_f32(&dir.join("tromr_seam_encoder_out.bin"));
        let seq = ctx_flat.len() / DIM;
        let ctx = Mat::from_vec(seq, DIM, ctx_flat);

        // L3: step-0 hidden over the seeds → all four heads vs the oracle.
        let hidden = decoder_forward(&dec, &ctx, &[1], &[0], &[0]).expect("prefill runs");
        let last = Mat::from_vec(1, DIM, hidden.row(hidden.rows - 1).to_vec());
        for (stream, head) in [
            ("rhythm", &dec.head_rhythm),
            ("pitch", &dec.head_pitch),
            ("lift", &dec.head_lift),
            ("note", &dec.head_note),
        ] {
            let ours = head.apply(&last).expect("head applies");
            let oracle = read_f32(&dir.join(format!("tromr_seam_head0_{stream}.bin")));
            assert_eq!(ours.data.len(), oracle.len(), "{stream} head width");
            let (c, m) = (cos(&ours.data, &oracle), maxabs(&ours.data, &oracle));
            eprintln!("[tromr-cert] head0_{stream} cos {c:.8} maxabs {m:.3e}");
            assert!(c >= 0.9999, "head0_{stream} cos {c}");
        }

        // L4: full argmax generate over the oracle context — token-EXACT.
        // The injected counter proves candidate capture does not add a second
        // decoder forward at any emitted position.
        let forward_count = std::cell::Cell::new(0usize);
        let generation = generate_with_control_and_forward(
            &dec,
            &ctx,
            DecodePick::Argmax,
            &LegacyExecutionControl,
            |decoder, context, rhythm, pitch, lift| {
                forward_count.set(forward_count.get() + 1);
                decoder_forward(decoder, context, rhythm, pitch, lift)
            },
        )
        .expect("generate with same-forward evidence runs");
        let streams = &generation.streams;
        let want = |k: &str| -> Vec<u32> {
            fx["argmax_generate"][k]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
                .collect()
        };
        assert_eq!(streams.rhythm, want("rhythm"), "rhythm stream");
        assert_eq!(streams.pitch, want("pitch"), "pitch stream");
        assert_eq!(streams.lift, want("lift"), "lift stream");
        assert_eq!(forward_count.get(), streams.rhythm.len());
        assert_eq!(generation.candidate_lattice.chosen_streams, *streams);
        assert_eq!(
            generation.candidate_lattice.positions.len(),
            streams.rhythm.len()
        );
        assert_eq!(
            generation
                .candidate_lattice
                .positions
                .last()
                .map(|position| position.rhythm_emitted_eos),
            Some(true),
            "the EOS position and all three of its head records are retained"
        );
        generation
            .candidate_lattice
            .validate()
            .expect("oracle lattice validates");
        eprintln!(
            "[tromr-cert] L4 argmax generate EXACT: {} steps, rhythm ends [barline, EOS]",
            streams.rhythm.len()
        );

        // E7 tail: the certified streams flow through the merge + MusicXML
        // assembly (the merge math itself is golden-tested synthetically).
        let mtk = fixture_tokenizer();
        let merged = merge_semantic(&mtk, streams).expect("merge runs");
        assert!(
            merged.starts_with("clef-F4+keySignature-CM+"),
            "merged head (the GT's own opening): {merged}"
        );
        assert!(merged.ends_with("barline"), "trailing EOS stripped");
        let xml = semantic_to_musicxml(&merged).expect("xml serializes");
        assert!(
            xml.contains("<clef><sign>F</sign><line>4</line></clef>"),
            "clef in xml"
        );
        assert!(
            xml.contains("<measure number=\"3\">"),
            "3 measures (3 barlines)"
        );
        eprintln!("[tromr-cert] E7 merge+MusicXML over the certified streams OK");
    }

    /// The E9 L0b cert: OUR preprocess (image crate decode + float bilinear +
    /// cv2-luma/ink arithmetic) vs the cv2 reference tensor, envelope
    /// MEASURED; then the output-level gate — our preprocess through the
    /// certified encoder + decoder must reproduce the oracle's argmax
    /// streams EXACTLY (the honest test: does the ±1-LSB resample envelope
    /// move any token?). Model-gated skip-with-SUCCESS.
    #[test]
    fn tromr_preprocess_envelope_and_output_gate() {
        let Some(dir) = zoo_dir() else {
            eprintln!("[tromr-test] skip_no_model: FOCR_TROMR_DIR unset");
            return;
        };
        let fx_path = dir.join("tromr_oracle_fixtures.json");
        if !fx_path.is_file() {
            eprintln!("[tromr-test] skip_no_model: oracle fixtures absent");
            return;
        }
        let fx: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fx_path).unwrap()).unwrap();
        let page = fx["_meta"]["page"].as_str().unwrap();
        if !std::path::Path::new(page).is_file() {
            eprintln!("[tromr-test] skip_no_model: upstream example absent ({page})");
            return;
        }
        let img = image::open(page).expect("example decodes");
        let (pixels, width) = crate::preprocess::tromr_staff_tensor(&img).expect("preprocess runs");
        let oracle_w = fx["preproc"]["shape"][2].as_u64().unwrap() as usize;
        assert_eq!(
            width, oracle_w,
            "resize geometry must match readimg exactly"
        );

        // L0b envelope vs the cv2 reference (normalized units; 1 u8 LSB =
        // 0.02257). MEASURED, not asserted tight: the gate is the
        // output-level stream identity below (the DISC-001 pattern).
        let oracle = read_f32(&dir.join("tromr_preproc.bin"));
        let m = maxabs(&pixels, &oracle);
        let lsb = 1.0f32 / (0.1738 * 255.0);
        let n_off = pixels
            .iter()
            .zip(oracle.iter())
            .filter(|(a, b)| (**a - **b).abs() > lsb * 1.5)
            .count();
        eprintln!(
            "[tromr-cert] L0b preprocess maxabs {m:.4} ({:.2} LSB); {n_off}/{} pixels past 1.5 LSB",
            m / lsb,
            pixels.len()
        );

        // Output-level gate: the full OUR-pipeline must reproduce the
        // certified streams token-exactly.
        let weights = Weights::load(&dir.join("tromr.focrq")).expect("artifact loads");
        let enc = TromrEncoderW::build(&weights).expect("encoder hydrates");
        let dec = TromrDecoderW::build(&weights).expect("decoder hydrates");
        let ctx = encode(&enc, &pixels, width).expect("encode runs");
        let streams = generate_argmax(&dec, &ctx).expect("generate runs");
        let want = |k: &str| -> Vec<u32> {
            fx["argmax_generate"][k]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
                .collect()
        };
        assert_eq!(streams.rhythm, want("rhythm"), "rhythm via OUR preprocess");
        assert_eq!(streams.pitch, want("pitch"), "pitch via OUR preprocess");
        assert_eq!(streams.lift, want("lift"), "lift via OUR preprocess");
        eprintln!("[tromr-cert] E9 full-native pipeline streams EXACT via our preprocess");
    }

    /// The E8 L5 quality leg: token-level SER (edit distance over `+`-split
    /// events, chords as single events) of OUR deterministic-argmax pipeline
    /// against the four COMMITTED upstream ground truths (examples/{1..4}).
    /// Measurement-first: per-example SER printed; the aggregate gate is
    /// pinned from the first measured run. (Upstream itself SAMPLES at
    /// T=0.2 — the paper's 0.025 merged SER is a sampled-decode number on
    /// the in-distribution test set; argmax-on-4-examples is our honest,
    /// reproducible floor.) Model-gated skip-with-SUCCESS.
    #[test]
    fn tromr_ser_vs_committed_ground_truth() {
        let Some(dir) = zoo_dir() else {
            eprintln!("[tromr-test] skip_no_model: FOCR_TROMR_DIR unset");
            return;
        };
        let examples = dir.join("../tromr-upstream/examples");
        if !examples.join("1.png").is_file() {
            eprintln!("[tromr-test] skip_no_model: upstream examples absent");
            return;
        }
        let weights = Weights::load(&dir.join("tromr.focrq")).expect("artifact loads");
        let tk = fixture_tokenizer();

        fn ser(ours: &str, gt: &str) -> f64 {
            let a: Vec<&str> = ours.split('+').collect();
            let b: Vec<&str> = gt.split('+').collect();
            // Levenshtein over event tokens.
            let (n, m) = (a.len(), b.len());
            let mut prev: Vec<usize> = (0..=m).collect();
            let mut cur = vec![0usize; m + 1];
            for i in 1..=n {
                cur[0] = i;
                for j in 1..=m {
                    let cost = usize::from(a[i - 1] != b[j - 1]);
                    cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
                }
                std::mem::swap(&mut prev, &mut cur);
            }
            prev[m] as f64 / m.max(1) as f64
        }

        let mut sers = Vec::new();
        for i in 1..=4u32 {
            let img = image::open(examples.join(format!("{i}.png"))).expect("example decodes");
            let res = recognize(&weights, &tk, &img).expect("recognize runs");
            let gt = std::fs::read_to_string(examples.join(format!("{i}.txt")))
                .expect("ground truth reads");
            let gt = gt.trim().trim_matches('\'').trim();
            let s = ser(&res.semantic, gt);
            eprintln!(
                "[tromr-cert] L5 example {i}: SER {s:.3} (ours {} events, gt {} events)",
                res.semantic.split('+').count(),
                gt.split('+').count()
            );
            sers.push(s);
        }
        let mean = sers.iter().sum::<f64>() / sers.len() as f64;
        eprintln!("[tromr-cert] L5 SER mean {mean:.3} over 4 committed examples (argmax decode)");
        // MEASURED gates (2026-07-06, argmax == sampled on real inputs):
        // per-example 0.125 / 0.040 / 0.375 / 0.304, mean 0.211. Pinned with
        // ~15% headroom for cross-arch float wiggle; deterministic decode.
        assert!(
            mean <= 0.25,
            "L5 SER mean {mean} regressed past 0.25 (measured 0.211)"
        );
        assert!(
            sers.iter().all(|&s| s <= 0.45),
            "a per-example SER regressed past 0.45 (measured max 0.375): {sers:?}"
        );
    }

    /// The E5 page cert: examples 1 and 2 stacked into one tall page (white
    /// gaps) must detect as TWO staves, top-to-bottom, and each staff's
    /// recognition must score against ITS OWN ground truth (order proof) at
    /// a measured SER. Model-gated skip-with-SUCCESS.
    #[test]
    fn tromr_page_detects_and_reads_stacked_examples() {
        let Some(dir) = zoo_dir() else {
            eprintln!("[tromr-test] skip_no_model: FOCR_TROMR_DIR unset");
            return;
        };
        let examples = dir.join("../tromr-upstream/examples");
        if !examples.join("1.png").is_file() {
            eprintln!("[tromr-test] skip_no_model: upstream examples absent");
            return;
        }
        // Stack ex1 over ex2 on a white canvas with generous gaps.
        let a = image::open(examples.join("1.png")).expect("ex1").to_rgb8();
        let b = image::open(examples.join("2.png")).expect("ex2").to_rgb8();
        let w = a.width().max(b.width());
        let gap = 160u32;
        let h = a.height() + b.height() + 3 * gap;
        let mut page = image::RgbImage::from_pixel(w, h, image::Rgb([255, 255, 255]));
        image::imageops::overlay(&mut page, &a, 0, i64::from(gap));
        image::imageops::overlay(&mut page, &b, 0, i64::from(2 * gap + a.height()));
        let page = image::DynamicImage::ImageRgb8(page);

        let weights = Weights::load(&dir.join("tromr.focrq")).expect("artifact loads");
        let tk = fixture_tokenizer();
        let result = recognize_page(&weights, &tk, &page).expect("page runs");
        assert_eq!(result.detected_staff_count, 2);
        assert_eq!(
            result.staff_segmentation_disposition,
            TromrStaffSegmentationDispositionV1::MultipleStavesDetectedPerCropRecognition
        );
        assert!(
            result.skips.is_empty(),
            "clean page must skip nothing: {:?}",
            result.skips
        );
        let staves = result.staves;
        assert_eq!(staves.len(), 2, "two staves detected on the stacked page");
        assert_eq!(
            (staves[0].0, staves[1].0),
            (0, 1),
            "detection indices in order"
        );
        assert!(
            staves[0].2.1 < staves[1].2.1,
            "top-to-bottom order: {:?} vs {:?}",
            staves[0].2,
            staves[1].2
        );

        fn ser(ours: &str, gt: &str) -> f64 {
            let a: Vec<&str> = ours.split('+').collect();
            let b: Vec<&str> = gt.split('+').collect();
            let (n, m) = (a.len(), b.len());
            let mut prev: Vec<usize> = (0..=m).collect();
            let mut cur = vec![0usize; m + 1];
            for i in 1..=n {
                cur[0] = i;
                for j in 1..=m {
                    let cost = usize::from(a[i - 1] != b[j - 1]);
                    cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
                }
                std::mem::swap(&mut prev, &mut cur);
            }
            prev[m] as f64 / m.max(1) as f64
        }
        let gt1 = std::fs::read_to_string(examples.join("1.txt")).unwrap();
        let gt2 = std::fs::read_to_string(examples.join("2.txt")).unwrap();
        let (gt1, gt2) = (
            gt1.trim().trim_matches('\'').trim().to_owned(),
            gt2.trim().trim_matches('\'').trim().to_owned(),
        );
        let s00 = ser(&staves[0].1.semantic, &gt1);
        let s01 = ser(&staves[0].1.semantic, &gt2);
        let s11 = ser(&staves[1].1.semantic, &gt2);
        let s10 = ser(&staves[1].1.semantic, &gt1);
        eprintln!(
            "[tromr-cert] E5 page: staff0 SER-vs-gt1 {s00:.3} (vs-gt2 {s01:.3}); \
             staff1 SER-vs-gt2 {s11:.3} (vs-gt1 {s10:.3})"
        );
        // Order proof: each staff matches ITS OWN ground truth best.
        assert!(s00 < s01, "staff0 must read as example 1");
        assert!(s11 < s10, "staff1 must read as example 2");
        // MEASURED gates (2026-07-06): 0.125 / 0.040 — IDENTICAL to the
        // direct-crop SERs; the detector's crops cost nothing. Pinned with
        // headroom (deterministic pipeline).
        assert!(s00 <= 0.25, "staff0 SER {s00} regressed (measured 0.125)");
        assert!(s11 <= 0.15, "staff1 SER {s11} regressed (measured 0.040)");
    }

    /// bd-av64.16/bd-av64.18 pinned public reproduction: all sixteen visible
    /// rows on page 1 of Mozart K.387 must reach TrOMR inference through the
    /// embedded PDF renderer. This is a true ignored live gate: missing or
    /// changed artifacts fail when explicitly run instead of reporting a
    /// skip-with-success.
    #[test]
    #[ignore = "requires pinned Mozart K.387 PDF and TrOMR f32 artifacts"]
    fn mozart_k387_page_one_letterboxes_all_sixteen_rows_into_inference() {
        use sha2::{Digest, Sha256};

        const PDF_SHA256: &str = "64406ae67f690b32f689bb60169287d0a6d514d13437b6027ee999381a43cb01";
        const MODEL_SHA256: &str =
            "a9d41485a98534ad0a1f7c1ec624f0a92f3f092c7dc30ac5af636b50dc465edc";

        let pdf_path = std::path::PathBuf::from(
            std::env::var_os("FOCR_TEST_MOZART_K387_PDF")
                .expect("FOCR_TEST_MOZART_K387_PDF must name the pinned public PDF"),
        );
        let model_dir = std::path::PathBuf::from(
            std::env::var_os("FOCR_TROMR_DIR")
                .expect("FOCR_TROMR_DIR must contain tromr.focrq and tokenizer tables"),
        );
        let model_path = model_dir.join("tromr.focrq");
        let sha256 = |path: &std::path::Path| {
            let bytes = std::fs::read(path)
                .unwrap_or_else(|error| panic!("read pinned {}: {error}", path.display()));
            format!("{:x}", Sha256::digest(bytes))
        };
        assert_eq!(sha256(&pdf_path), PDF_SHA256, "PDF artifact changed");
        assert_eq!(sha256(&model_path), MODEL_SHA256, "model artifact changed");

        let raster_started = std::time::Instant::now();
        let pages = crate::pdf::PdfPages::open(&pdf_path).expect("native PDF open");
        let page = pages.render(0).expect("native page-1 render");
        let raster_elapsed = raster_started.elapsed();
        assert_eq!(
            (page.width(), page.height()),
            (5904, 7558),
            "pinned page dimensions"
        );

        let load_started = std::time::Instant::now();
        let weights = Weights::load(&model_path).expect("pinned TrOMR artifact loads");
        let tk = crate::tokenizer::music::MusicTokenizer::from_dir(&model_dir)
            .expect("pinned tokenizer tables load");
        let load_elapsed = load_started.elapsed();
        let inference_started = std::time::Instant::now();
        let result = recognize_page(&weights, &tk, &page).expect("all staff rows are accounted");
        let inference_elapsed = inference_started.elapsed();

        assert_eq!(result.detected_staff_count, 16);
        assert_eq!(
            result.staff_segmentation_disposition,
            TromrStaffSegmentationDispositionV1::MultipleStavesDetectedPerCropRecognition
        );
        assert_eq!(
            result.staff_evidence.len(),
            16,
            "four visible four-row systems must yield sixteen row attempts"
        );
        assert_eq!(result.staves.len(), 16, "all sixteen rows must recognize");
        assert!(
            result.skips.is_empty(),
            "no row may skip after lossless recovery: {:?}",
            result.skips
        );
        let mut padded_rows = 0usize;
        for evidence in &result.staff_evidence {
            let geometry = evidence.geometry;
            let source = geometry.source_bbox;
            let padding = geometry.padding;
            if padding.top + padding.right + padding.bottom + padding.left > 0 {
                padded_rows += 1;
            }
            assert_eq!(
                source.2 + padding.left + padding.right,
                geometry.canvas_width,
                "row {} horizontal source/canvas accounting",
                evidence.index
            );
            assert_eq!(
                source.3 + padding.top + padding.bottom,
                geometry.canvas_height,
                "row {} vertical source/canvas accounting",
                evidence.index
            );
            assert!(
                IMG_H * geometry.canvas_width <= POS_COLS * PATCH * geometry.canvas_height,
                "row {} canvas still exceeds the positional budget",
                evidence.index
            );
            assert_eq!(evidence.outcome, StaffInferenceOutcome::Recognized);
            assert!(evidence.reason.is_none());
            eprintln!(
                "[tromr-cert] mozart row={} source_bbox={:?} canvas={}x{} padding={:?} \
                 outcome={} reason={:?}",
                evidence.index + 1,
                source,
                geometry.canvas_width,
                geometry.canvas_height,
                padding,
                evidence.outcome.as_str(),
                evidence.reason
            );
        }
        assert!(padded_rows > 0, "fixture must exercise letterbox recovery");
        eprintln!(
            "[tromr-cert] mozart_k387 pdf_sha256={PDF_SHA256} model_sha256={MODEL_SHA256} \
             raster_ms={} load_ms={} inference_ms={} rows=16 recognized=16 skipped=0 padded_rows={padded_rows}",
            raster_elapsed.as_millis(),
            load_elapsed.as_millis(),
            inference_elapsed.as_millis(),
        );
    }

    /// bd-av64.2/bd-av64.16: ONE formerly unfittable staff band must not abort
    /// the page. Post bd-av64.16, lossless letterboxing carries the band into
    /// inference without relabeling an unrelated model error as a clamp. Two
    /// hand-drawn staves on a 12000px-wide canvas exercise both physical
    /// extension and synthetic padding. Content quality is NOT this test's
    /// concern (the SER certs own that); source/canvas accounting is.
    #[test]
    fn tromr_page_letterboxes_overwide_staff_and_keeps_the_rest() {
        let Some(dir) = zoo_dir() else {
            eprintln!("[tromr-test] skip_no_model: FOCR_TROMR_DIR unset");
            return;
        };
        let (page_w, h) = (12_000u32, 1_600u32);
        let mut page = image::RgbImage::from_pixel(page_w, h, image::Rgb([255, 255, 255]));
        for line in 0..5u32 {
            let y = 250 + line * 10; // fittable staff: ink 40..7040
            for dy in 0..2 {
                for x in 40..7_040u32 {
                    page.put_pixel(x, y + dy, image::Rgb([10, 10, 10]));
                }
            }
        }
        for line in 0..5u32 {
            let y = 1_400 + line * 10; // unfittable staff: full width
            for dy in 0..2 {
                for x in 0..page_w {
                    page.put_pixel(x, y + dy, image::Rgb([10, 10, 10]));
                }
            }
        }
        let page = image::DynamicImage::ImageRgb8(page);

        let weights = Weights::load(&dir.join("tromr.focrq")).expect("artifact loads");
        let tk = fixture_tokenizer();
        let result = recognize_page(&weights, &tk, &page)
            .expect("page with one formerly unfittable staff must succeed");
        eprintln!(
            "[tromr-cert] letterbox resilience: {} recognized, {} skipped ({:?})",
            result.staves.len(),
            result.skips.len(),
            result.skips.iter().map(|s| &s.reason).collect::<Vec<_>>()
        );
        assert_eq!(result.detected_staff_count, 2);
        assert_eq!(
            result.staff_segmentation_disposition,
            TromrStaffSegmentationDispositionV1::MultipleStavesDetectedPerCropRecognition
        );
        assert_eq!(result.staff_evidence.len(), 2);
        assert!(
            result.staff_evidence[1].geometry.padding.top
                + result.staff_evidence[1].geometry.padding.bottom
                > 0,
            "the second staff must exercise synthetic vertical padding"
        );
        for evidence in &result.staff_evidence {
            let geometry = evidence.geometry;
            assert!(IMG_H * geometry.canvas_width <= POS_COLS * PATCH * geometry.canvas_height);
            if let Some(reason) = &evidence.reason {
                assert!(
                    !reason.contains("1280") && !reason.contains("position clamp"),
                    "a downstream model failure must not be relabeled as a clamp: {reason}"
                );
            }
        }
    }

    /// bd-av64.4 split-quality pin: a staff narrow enough to run whole is
    /// ALSO run through the barline-split path with a forced budget.
    /// ATTRIBUTES (clef/key/time) must match exactly and rhythm-class
    /// content must broadly agree; ABSOLUTE PITCH REGISTRATION on
    /// continuation segments is a DOCUMENTED, MEASURED divergence (the
    /// model has no clef context mid-line and reads continuations octaves
    /// off; a pixel-space clef prepend measured WORSE — the model read the
    /// pasted clef as notes). Split therefore ships as a LAST-RESORT: it
    /// only fires where the alternative is a skip with zero content. This
    /// test pins today's measured rhythm agreement so improvements and
    /// regressions are both visible. Model-gated.
    #[test]
    fn tromr_split_matches_whole_staff_read() {
        let Some(dir) = zoo_dir() else {
            eprintln!("[tromr-test] skip_no_model: FOCR_TROMR_DIR unset");
            return;
        };
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/realscan_music/staves/spohr_no17_top.png");
        if !fixture.is_file() {
            eprintln!("[tromr-test] skip_no_model: realscan fixture absent");
            return;
        }
        let img = image::open(&fixture).expect("fixture opens");
        let crops = crate::preprocess::staff_detect::detect_staves(&img).expect("detect runs");
        assert_eq!(crops.len(), 1, "fixture is a single staff");
        let crop = &crops[0];

        let weights = Weights::load(&dir.join("tromr.focrq")).expect("artifact loads");
        let tk = fixture_tokenizer();

        let whole = {
            let buf = image::GrayImage::from_raw(crop.w as u32, crop.h as u32, crop.gray.clone())
                .expect("crop buffer");
            recognize(&weights, &tk, &image::DynamicImage::ImageLuma8(buf))
                .expect("whole-staff read")
        };
        let split = recognize_split(
            &weights,
            &tk,
            crop,
            crop.w * 2 / 3,
            TromrRecognitionOptionsV1 {
                split_policy: TromrSplitPolicyV1::ExperimentalBarlineSegments,
                ..TromrRecognitionOptionsV1::deterministic()
            },
        )
        .expect("split runs")
        .expect("fixture has usable barlines");

        // Attributes: identical.
        for attr in ["clef-", "keySignature-", "timeSignature-"] {
            let pick = |sem: &str| -> Vec<String> {
                sem.split('+')
                    .filter(|t| t.starts_with(attr))
                    .map(str::to_owned)
                    .collect()
            };
            assert_eq!(
                pick(&whole.semantic),
                pick(&split.semantic),
                "{attr} attributes must match"
            );
        }
        // Rhythm-class agreement: compare duration suffixes only (pitch
        // registration is the documented divergence). Require >= 60% of the
        // whole read's rhythm stream to appear in order in the split read
        // (measured 2026-07-07: whole 20 tokens / split ~45 — segments
        // re-read seam measures; the in-order rhythm core survives).
        let rhythms = |sem: &str| -> Vec<String> {
            sem.split('+')
                .filter(|t| t.starts_with("note-") || t.starts_with("rest-"))
                .filter_map(|t| {
                    split_pitch_duration(t)
                        .map(|(_, (x, _, d))| format!("{x}{}", if d { "." } else { "" }))
                })
                .collect()
        };
        let (wr, sr) = (rhythms(&whole.semantic), rhythms(&split.semantic));
        let mut it = sr.iter();
        let matched = wr.iter().filter(|w| it.by_ref().any(|s| s == *w)).count();
        let ratio = matched as f64 / wr.len().max(1) as f64;
        eprintln!(
            "[tromr-cert] split-vs-whole rhythm agreement {ratio:.3} ({matched}/{}; split ships \
             as LAST-RESORT: absolute octave on continuations is a documented divergence)",
            wr.len()
        );
        // Pinned at the MEASURED 2026-07-07 level (0.200) minus headroom:
        // this is a tripwire for regressions and a visible marker for any
        // future improvement — NOT an endorsement of split quality (the
        // path ships behind an explicit experimental split policy for exactly
        // this reason).
        assert!(
            ratio >= 0.15,
            "split rhythm stream regressed below the pinned floor: {ratio:.3}"
        );
        // Both outputs are structurally valid by construction (emit-time
        // validator), but assert anyway — the seam logic splices semantics.
        assert!(validate_musicxml(&split.musicxml).is_empty());
    }

    #[test]
    fn terminal_control_flow_is_never_downgraded_to_a_staff_skip() {
        for terminal in [
            FocrError::Cancelled,
            FocrError::Timeout("explicit page allowance exhausted".into()),
        ] {
            let mut staves = Vec::new();
            let mut skips = Vec::new();
            let mut evidence = Vec::new();
            let bbox = (0, 100, 800, 80);
            let error = record_staff_outcome(
                bbox,
                synthetic_staff_evidence(1, bbox),
                Err(terminal),
                &mut staves,
                &mut skips,
                &mut evidence,
            )
            .expect_err("control flow must terminate the page");
            assert!(is_terminal_execution_error(&error));
            assert!(staves.is_empty());
            assert!(skips.is_empty());
            assert!(evidence.is_empty());
        }
    }

    #[test]
    fn terminal_second_row_stops_before_later_rows_but_row_errors_remain_skips() {
        let outcomes = [
            Ok(synthetic_music_result("row-zero")),
            Err(FocrError::Cancelled),
            Ok(synthetic_music_result("must-not-run")),
        ];
        let mut calls = 0usize;
        let mut staves = Vec::new();
        let mut skips = Vec::new();
        let mut evidence = Vec::new();
        let mut terminal = None;
        for (index, outcome) in outcomes.into_iter().enumerate() {
            calls += 1;
            let bbox = (0, index * 100, 800, 80);
            if let Err(error) = record_staff_outcome(
                bbox,
                synthetic_staff_evidence(index, bbox),
                outcome,
                &mut staves,
                &mut skips,
                &mut evidence,
            ) {
                terminal = Some(error);
                break;
            }
        }
        assert_eq!(calls, 2, "row after cancellation must not be attempted");
        assert!(matches!(terminal, Some(FocrError::Cancelled)));
        assert_eq!(staves.len(), 1);
        assert!(skips.is_empty());
        assert_eq!(evidence.len(), 1);

        let bbox = (0, 200, 800, 80);
        record_staff_outcome(
            bbox,
            synthetic_staff_evidence(2, bbox),
            Err(FocrError::Other(anyhow::anyhow!(
                "row-local decoder failure"
            ))),
            &mut staves,
            &mut skips,
            &mut evidence,
        )
        .expect("a genuine row error remains recoverable");
        assert_eq!(skips.len(), 1);
        assert_eq!(skips[0].index, 2);
        assert_eq!(evidence.len(), 2);
        assert_eq!(evidence[1].outcome, StaffInferenceOutcome::Skipped);
    }

    #[test]
    fn staff_segmentation_disposition_is_count_derived_and_wire_stable() {
        for (detected_staff_count, expected, wire) in [
            (
                0,
                TromrStaffSegmentationDispositionV1::NoStaffDetectedWholeImageFallback,
                "no_staff_detected_whole_image_fallback",
            ),
            (
                1,
                TromrStaffSegmentationDispositionV1::SingleStaffDetectedWholeImageRecognition,
                "single_staff_detected_whole_image_recognition",
            ),
            (
                2,
                TromrStaffSegmentationDispositionV1::MultipleStavesDetectedPerCropRecognition,
                "multiple_staves_detected_per_crop_recognition",
            ),
            (
                16,
                TromrStaffSegmentationDispositionV1::MultipleStavesDetectedPerCropRecognition,
                "multiple_staves_detected_per_crop_recognition",
            ),
        ] {
            let disposition =
                TromrStaffSegmentationDispositionV1::for_detected_staff_count(detected_staff_count);
            assert_eq!(disposition, expected);
            assert!(disposition.is_consistent_with(detected_staff_count));
            assert_eq!(disposition.as_str(), wire);
            assert_eq!(
                serde_json::to_string(&disposition).expect("serialize disposition"),
                format!("\"{wire}\"")
            );
            assert_eq!(
                serde_json::from_str::<TromrStaffSegmentationDispositionV1>(&format!("\"{wire}\""))
                    .expect("deserialize disposition"),
                disposition
            );
        }
        assert!(
            serde_json::from_str::<TromrStaffSegmentationDispositionV1>(
                "\"future_unreviewed_route\""
            )
            .is_err()
        );
    }

    /// bd-av64.2: when every detected staff fails for reasons unrelated to
    /// canvas fit, the page error preserves and names each underlying reason.
    /// This unit-level assembly test does not need a model and therefore
    /// cannot become a skip-with-success.
    #[test]
    fn tromr_page_all_staves_failing_is_a_named_error() {
        let reasons = ["decoder exploded", "invalid semantic stream"];
        let skips: Vec<StaffSkip> = reasons
            .iter()
            .enumerate()
            .map(|(index, reason)| StaffSkip {
                index,
                bbox: (0, 100 * index, 800, 80),
                reason: (*reason).to_owned(),
            })
            .collect();
        let evidence: Vec<StaffInferenceEvidence> = skips
            .iter()
            .map(|skip| StaffInferenceEvidence {
                index: skip.index,
                geometry: crate::preprocess::staff_detect::StaffCropGeometry::unpadded(skip.bbox),
                route: TromrRowInferenceRouteV1::DetectedStaffCrop,
                forward_inputs: vec![TromrForwardInputV1 {
                    gray8: crate::preprocess::staff_detect::TromrGray8CropV1::from_tightly_packed(
                        vec![255; skip.bbox.2 * skip.bbox.3],
                        skip.bbox.2,
                        skip.bbox.3,
                    )
                    .expect("test crop"),
                    source_space: TromrModelInputSourceSpaceV1::ReviewCropCanvas,
                    source_bbox_xywh: (0, 0, skip.bbox.2, skip.bbox.3),
                    padding: crate::preprocess::staff_detect::StaffPadding::default(),
                    staff_lines_y_in_canvas: Some([1, 2, 3, 4, 5]),
                }],
                review_crop_gray8: None,
                review_crop_geometry: None,
                staff_lines: None,
                outcome: StaffInferenceOutcome::Skipped,
                reason: Some(skip.reason.clone()),
            })
            .collect();
        let options = TromrRecognitionOptionsV1::deterministic();
        let detector_crops = skips
            .iter()
            .map(|skip| {
                let review_lines = [1, 2, 3, 4, 5];
                (
                    crate::preprocess::staff_detect::StaffCropGeometry::unpadded(skip.bbox),
                    review_lines.map(|line| skip.bbox.1 + line),
                    review_lines,
                )
            })
            .collect::<Vec<_>>();
        let (staff_detection, retained_staff_detection) =
            synthetic_staff_detection_pair_for_test(800, 180, &detector_crops);
        let err = match finish_page_recognition(
            2,
            Vec::new(),
            skips,
            evidence,
            PageStaffDetectionArtifacts {
                pixel_free: staff_detection,
                retained: retained_staff_detection,
            },
            options,
            options.replay_identity().expect("default identity"),
        ) {
            Ok(_) => panic!("all failed rows must not yield an empty success"),
            Err(error) => error.to_string(),
        };
        assert!(
            err.contains("all 2 detected staves failed"),
            "error names the total: {err}"
        );
        assert!(err.contains("staff 0:"), "error names staff 0: {err}");
        assert!(err.contains("staff 1:"), "error names staff 1: {err}");
        assert!(err.contains(reasons[0]), "first cause preserved: {err}");
        assert!(err.contains(reasons[1]), "second cause preserved: {err}");
        assert!(
            !err.contains("position clamp"),
            "cause is not relabeled: {err}"
        );
    }

    /// The E3 L1/L2 cert: every oracle seam (stem, stages, patch proj, each
    /// ViT block, the final norm) at cosine ≥ 0.9999 with maxabs ledgered;
    /// the oracle's own floor on this stack is 0.0 (same- AND cross-thread),
    /// so every divergence below is OUR summation-order envelope, reported
    /// per seam. Model-gated skip-with-SUCCESS.
    #[test]
    fn tromr_encoder_matches_torch_oracle() {
        let Some(dir) = zoo_dir() else {
            eprintln!("[tromr-test] skip_no_model: FOCR_TROMR_DIR unset");
            return;
        };
        let fx_path = dir.join("tromr_oracle_fixtures.json");
        if !fx_path.is_file() {
            eprintln!(
                "[tromr-test] skip_no_model: oracle fixtures absent (gen_reference_fixtures_tromr.py)"
            );
            return;
        }
        let fx: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fx_path).unwrap()).unwrap();
        let width = fx["preproc"]["shape"][2].as_u64().unwrap() as usize;
        let pixels = read_f32(&dir.join("tromr_preproc.bin"));
        assert_eq!(pixels.len(), IMG_H * width, "preproc fixture shape");

        let weights = Weights::load(&dir.join("tromr.focrq")).expect("artifact loads");
        let w = TromrEncoderW::build(&weights).expect("hydrates");

        // Backbone seams (channel-major in the fixture, ours identical layout).
        let feat = backbone(&w, &pixels, width).expect("backbone runs");
        let stage2 = read_f32(&dir.join("tromr_seam_stage2.bin"));
        assert_eq!(feat.data.len(), stage2.len(), "stage2 shape");
        let (c, m) = (cos(&feat.data, &stage2), maxabs(&feat.data, &stage2));
        eprintln!("[tromr-cert] stage2 cos {c:.8} maxabs {m:.3e}");
        assert!(c >= 0.9999, "stage2 cos {c}");

        // Full encoder vs the final oracle output [1, seq, 256].
        let out = encode(&w, &pixels, width).expect("encode runs");
        let oracle = read_f32(&dir.join("tromr_seam_encoder_out.bin"));
        assert_eq!(out.data.len(), oracle.len(), "encoder_out shape");
        let (c, m) = (cos(&out.data, &oracle), maxabs(&out.data, &oracle));
        eprintln!(
            "[tromr-cert] encoder_out cos {c:.8} maxabs {m:.3e} (oracle floor 0.0 both legs)"
        );
        assert!(c >= 0.9999, "encoder_out cos {c}");
    }
}
