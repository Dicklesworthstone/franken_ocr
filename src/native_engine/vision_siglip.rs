//! SigLIP-B/16 vision tower for SmolVLM2 (C3, bd-3jo6.3.3) — the third vision
//! tower, a NEW machine per `docs/zoo/smolvlm2-spec.md` §2 (NOT a
//! `vision_sam.rs` variant): separate q/k/v/out projections **with bias**
//! (SAM fuses qkv), full **bidirectional** attention (no causal mask, no
//! windows, no decomposed rel-pos), a plain learned 1-D position table looked
//! up by the reference NaViT bucketize (NOT identity — the `(1-1e-6)` scale
//! makes the per-axis buckets `[0,0,1,…,30]`; see [`embed_frame`]),
//! **tanh-GELU** (OQ-1,
//! [`nn::gelu_tanh`] — SAM is erf, CLIP is quick), pre-LN blocks, and a final
//! `post_layernorm`. No neck / compressor — the SmolVLM2 connector
//! (pixel-shuffle, [`super::token_compress`]) follows this tower.
//!
//! Reuse (A8, bd-3jo6.1.8): the k16-s16 patch-embed drives the SAME
//! im2col+GEMM conv leaf SAM/GOT certify ([`vision_sam::conv_apply`]), and
//! attention runs on the fused [`nn::sdpa`] flash kernel exactly like CLIP
//! (`causal=false`) — share by import, never relocate certified code (the B3
//! precedent).
//!
//! Every dimension is compile-time known (doctrine: shape-specialized, no
//! runtime shape branching): 512² input → 32×32 = 1024 patch tokens, hidden
//! 768, 12 layers, 12 heads × head_dim 64 (scale 1/8), MLP 3072, LN ε=1e-6.
//! Weight-shape facts are byte-verified in spec §12 and re-asserted at
//! hydration — a mislabeled checkpoint fails loud, never silently mis-runs.

use crate::error::{FocrError, FocrResult};

use super::nn;
use super::tensor::Mat;
use super::vision_sam::{self, Conv, LayerNormP, Linear};
use super::weights::Weights;

/// Hidden width (spec §2).
pub const EMBED_DIM: usize = 768;
/// Encoder depth (spec §2/§12: layers 0..11 verified in the shard).
pub const DEPTH: usize = 12;
/// Attention heads (head_dim 64, scale 1/8).
pub const NUM_HEADS: usize = 12;
/// Per-head dim.
pub const HEAD_DIM: usize = 64;
/// Patch size (k16 s16 conv).
pub const PATCH: usize = 16;
/// Frame side — every SmolVLM2 frame is exactly 512×512 (spec §6).
pub const IMG_SIDE: usize = 512;
/// Token grid side (512/16).
pub const GRID: usize = IMG_SIDE / PATCH;
/// Tokens per frame (32² = 1024).
pub const TOKENS: usize = GRID * GRID;
/// MLP intermediate width (spec §12: fc1 [3072,768] verified).
pub const INTERMEDIATE: usize = 3072;
/// LayerNorm epsilon (`configuration_smolvlm.py` default; spec §2).
const LN_EPS: f32 = 1e-6;
/// Softmax scale 1/sqrt(64).
const ATTN_SCALE: f32 = 0.125;

/// One pre-LN SigLIP encoder block's parameters.
#[derive(Debug, Clone)]
pub struct SiglipBlockP {
    /// `layer_norm1` (pre-attention).
    pub ln1: LayerNormP,
    /// `self_attn.q_proj` (768→768, bias).
    pub q: Linear,
    /// `self_attn.k_proj`.
    pub k: Linear,
    /// `self_attn.v_proj`.
    pub v: Linear,
    /// `self_attn.out_proj`.
    pub out: Linear,
    /// `layer_norm2` (pre-MLP).
    pub ln2: LayerNormP,
    /// `mlp.fc1` (768→3072, bias).
    pub fc1: Linear,
    /// `mlp.fc2` (3072→768, bias).
    pub fc2: Linear,
}

/// The full SigLIP-B/16 parameter set for the SmolVLM2 vision tower.
#[derive(Debug, Clone)]
pub struct SiglipWeights {
    /// `embeddings.patch_embedding` — Conv(3→768, k16 s16, bias).
    pub patch_embed: Conv,
    /// `embeddings.position_embedding.weight` — `[1024, 768]` row-major,
    /// added by identity ids (no CLS token, no interpolation).
    pub pos_embed: Vec<f32>,
    /// The 12 encoder blocks.
    pub blocks: Vec<SiglipBlockP>,
    /// The final `post_layernorm`.
    pub post_ln: LayerNormP,
}

/// Hydrate a [`SiglipWeights`] from the named `{prefix}.*` tensors (canonical
/// prefix: the SMOLVLM2 descriptor's `vision_tower_prefix()`,
/// `"model.vision_model"`). Every shape is asserted against the spec-§12
/// byte-verified facts.
///
/// # Errors
/// [`FocrError::ModelNotFound`]/[`FocrError::FormatMismatch`] from the weight
/// accessors for missing tensors; [`FocrError::Other`] on any shape drift.
pub fn siglip_weights_from(weights: &Weights, prefix: &str) -> FocrResult<SiglipWeights> {
    let p = prefix;
    let linear = |name: &str, out: usize, in_: usize| -> FocrResult<Linear> {
        let w = weights.vec(&format!("{name}.weight"))?;
        let b = weights.vec(&format!("{name}.bias"))?;
        if w.len() != out * in_ || b.len() != out {
            return Err(FocrError::Other(anyhow::anyhow!(
                "vision_siglip {name}: weight/bias len ({}, {}) != ([{out},{in_}], {out})",
                w.len(),
                b.len()
            )));
        }
        Linear::from_row_major(&w, b, out, in_)
    };
    let ln = |name: &str| -> FocrResult<LayerNormP> {
        let w = weights.vec(&format!("{name}.weight"))?;
        let b = weights.vec(&format!("{name}.bias"))?;
        if w.len() != EMBED_DIM || b.len() != EMBED_DIM {
            return Err(FocrError::Other(anyhow::anyhow!(
                "vision_siglip {name}: affine len ({}, {}) != {EMBED_DIM}",
                w.len(),
                b.len()
            )));
        }
        Ok(LayerNormP { w, b })
    };

    let pe_w = weights.vec(&format!("{p}.embeddings.patch_embedding.weight"))?;
    let pe_b = weights.vec(&format!("{p}.embeddings.patch_embedding.bias"))?;
    if pe_w.len() != EMBED_DIM * 3 * PATCH * PATCH || pe_b.len() != EMBED_DIM {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_siglip patch_embedding: weight/bias len ({}, {}) != ([{EMBED_DIM},3,{PATCH},{PATCH}], {EMBED_DIM})",
            pe_w.len(),
            pe_b.len()
        )));
    }
    let patch_embed = Conv {
        w: pe_w,
        b: Some(pe_b),
        out_ch: EMBED_DIM,
        in_ch: 3,
        kh: PATCH,
        kw: PATCH,
    };

    let pos_embed = weights.vec(&format!("{p}.embeddings.position_embedding.weight"))?;
    if pos_embed.len() != TOKENS * EMBED_DIM {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_siglip position_embedding: len {} != [{TOKENS},{EMBED_DIM}]",
            pos_embed.len()
        )));
    }

    let mut blocks = Vec::with_capacity(DEPTH);
    for i in 0..DEPTH {
        let b = format!("{p}.encoder.layers.{i}");
        blocks.push(SiglipBlockP {
            ln1: ln(&format!("{b}.layer_norm1"))?,
            q: linear(&format!("{b}.self_attn.q_proj"), EMBED_DIM, EMBED_DIM)?,
            k: linear(&format!("{b}.self_attn.k_proj"), EMBED_DIM, EMBED_DIM)?,
            v: linear(&format!("{b}.self_attn.v_proj"), EMBED_DIM, EMBED_DIM)?,
            out: linear(&format!("{b}.self_attn.out_proj"), EMBED_DIM, EMBED_DIM)?,
            ln2: ln(&format!("{b}.layer_norm2"))?,
            fc1: linear(&format!("{b}.mlp.fc1"), INTERMEDIATE, EMBED_DIM)?,
            fc2: linear(&format!("{b}.mlp.fc2"), EMBED_DIM, INTERMEDIATE)?,
        });
    }

    Ok(SiglipWeights {
        patch_embed,
        pos_embed,
        blocks,
        post_ln: ln(&format!("{p}.post_layernorm"))?,
    })
}

/// Forward one 512² frame through the tower: normalized NCHW pixels
/// (`[3, 512, 512]` flat, the preprocess `x/127.5 - 1` rail) → `[1024, 768]`
/// post-LN token rows.
///
/// # Errors
/// [`FocrError::Other`] on a wrong pixel buffer length or any kernel-shape
/// violation.
pub fn forward_frame(w: &SiglipWeights, pixels: &[f32]) -> FocrResult<Mat> {
    if pixels.len() != 3 * IMG_SIDE * IMG_SIDE {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_siglip forward: pixel buffer len {} != 3*{IMG_SIDE}*{IMG_SIDE}",
            pixels.len()
        )));
    }
    let mut x = embed_frame(w, pixels)?;
    for blk in &w.blocks {
        encoder_block(blk, &mut x)?;
    }
    nn::layer_norm(&x, Some(&w.post_ln.w), Some(&w.post_ln.b), LN_EPS)
}

/// The embeddings stage: patch-embed conv (A8 leaf: im2col+GEMM, pad 0,
/// stride 16) → `[1024, 768]` token rows + the learned pos table looked up by
/// the reference NaViT bucketize ids. This is the oracle's
/// `hidden_states[0]` seam.
///
/// **The bucketize is NOT identity** (a census transcription error, caught by
/// this seam's parity gate 2026-07-02): `modeling_smolvlm.py` scales every
/// fractional coordinate by `(1 - 1e-6)` — `(i/32)*(1-1e-6)` — which pushes
/// each exact multiple JUST BELOW its own `i/32` boundary, so
/// `bucketize(·, right=True)` yields per-axis buckets `[0, 0, 1, 2, …, 30]`:
/// coordinate 0 and 1 share bucket 0 and bucket 31 is never used. For the
/// fixed full-mask 512² geometry this is exactly `i.saturating_sub(1)`
/// (proven: `(i/32)(1-1e-6)` is strictly below `i/32` and strictly above
/// `(i-1)/32` in f32 for `1 ≤ i ≤ 31`), verified bit-level against the live
/// module.
pub(crate) fn embed_frame(w: &SiglipWeights, pixels: &[f32]) -> FocrResult<Mat> {
    let nchw = vision_sam::conv_apply(&w.patch_embed, pixels, IMG_SIDE, IMG_SIDE, 0, PATCH)?;
    let mut x = vision_sam::nchw_to_nhwc_rows(&nchw, EMBED_DIM, GRID, GRID);
    let bucket = |i: usize| i.saturating_sub(1);
    for r in 0..GRID {
        for c in 0..GRID {
            let t = r * GRID + c;
            let pos_id = bucket(r) * GRID + bucket(c);
            let row = x.row_mut(t);
            let pos = &w.pos_embed[pos_id * EMBED_DIM..(pos_id + 1) * EMBED_DIM];
            for (v, p) in row.iter_mut().zip(pos) {
                *v += p;
            }
        }
    }
    Ok(x)
}

/// One pre-LN encoder block, in place:
/// `x += attn(LN1(x)); x += fc2(gelu_tanh(fc1(LN2(x))))`. The oracle's
/// `hidden_states[i+1]` seam.
pub(crate) fn encoder_block(blk: &SiglipBlockP, x: &mut Mat) -> FocrResult<()> {
    let h = nn::layer_norm(x, Some(&blk.ln1.w), Some(&blk.ln1.b), LN_EPS)?;
    let attn = self_attention(blk, &h)?;
    add_assign(x, &attn)?;

    let h2 = nn::layer_norm(x, Some(&blk.ln2.w), Some(&blk.ln2.b), LN_EPS)?;
    let mut m = blk.fc1.apply(&h2)?;
    nn::gelu_tanh(&mut m);
    let m = blk.fc2.apply(&m)?;
    add_assign(x, &m)
}

/// Forward `n_frames` stacked frames (`[F, 3, 512, 512]` flat) sequentially,
/// returning one `[1024, 768]` [`Mat`] per frame. Sequential per frame — all
/// parallelism stays inside the `ft-kernel-cpu` GEMMs (doctrine #5: no nested
/// rayon; the batched-GEMM stacking lever is a later, measured optimization).
///
/// # Errors
/// [`FocrError::Other`] on a length/frame-count mismatch or any per-frame
/// failure.
pub fn forward_frames(w: &SiglipWeights, pixels: &[f32], n_frames: usize) -> FocrResult<Vec<Mat>> {
    let frame_len = 3 * IMG_SIDE * IMG_SIDE;
    if n_frames == 0 || pixels.len() != n_frames * frame_len {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_siglip forward_frames: buffer len {} != n_frames {n_frames} * {frame_len}",
            pixels.len()
        )));
    }
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        out.push(forward_frame(
            w,
            &pixels[f * frame_len..(f + 1) * frame_len],
        )?);
    }
    Ok(out)
}

/// Bidirectional multi-head attention over one frame's tokens: separate
/// q/k/v projections (bias), head-major repack, fused [`nn::sdpa`] with
/// `causal=false` and scale 1/8, then `out_proj`. The CLIP `self_attention`
/// shape with SigLIP's separate projections.
fn self_attention(blk: &SiglipBlockP, x: &Mat) -> FocrResult<Mat> {
    let seq = x.rows;
    let q = blk.q.apply(x)?;
    let k = blk.k.apply(x)?;
    let v = blk.v.apply(x)?;

    // Repack [seq, 768] → head-major [heads, seq, 64].
    let head_span = seq * HEAD_DIM;
    let mut qf = vec![0.0f32; NUM_HEADS * head_span];
    let mut kf = vec![0.0f32; NUM_HEADS * head_span];
    let mut vf = vec![0.0f32; NUM_HEADS * head_span];
    for s in 0..seq {
        let (qr, kr, vr) = (q.row(s), k.row(s), v.row(s));
        for h in 0..NUM_HEADS {
            let src = h * HEAD_DIM;
            let dst = h * head_span + s * HEAD_DIM;
            qf[dst..dst + HEAD_DIM].copy_from_slice(&qr[src..src + HEAD_DIM]);
            kf[dst..dst + HEAD_DIM].copy_from_slice(&kr[src..src + HEAD_DIM]);
            vf[dst..dst + HEAD_DIM].copy_from_slice(&vr[src..src + HEAD_DIM]);
        }
    }

    let ctx = nn::sdpa(
        &qf, &kf, &vf, NUM_HEADS, seq, seq, HEAD_DIM, HEAD_DIM, ATTN_SCALE, false,
    );

    // Unpack head-major context back to [seq, 768] rows.
    let mut merged = Mat::zeros(seq, EMBED_DIM);
    for h in 0..NUM_HEADS {
        for s in 0..seq {
            let src = h * head_span + s * HEAD_DIM;
            let dst_row = merged.row_mut(s);
            dst_row[h * HEAD_DIM..(h + 1) * HEAD_DIM].copy_from_slice(&ctx[src..src + HEAD_DIM]);
        }
    }
    blk.out.apply(&merged)
}

/// Batched analogue of [`self_attention`] over an `[F·seq, dim]` stacked
/// buffer (bd-av64.10, the in-code "batched-GEMM stacking lever"): the
/// q/k/v/out projections are M-batched (row-independent, so each output row's
/// K-reduction is unchanged — the bd-1azu.10 byte-identity property), and the
/// SDPA runs with `num_bh = F·heads` where block `f·heads + h` covers frame
/// `f`'s tokens ONLY — strictly block-diagonal across frames, byte-identical
/// to attending each frame separately.
fn self_attention_batched(
    blk: &SiglipBlockP,
    x: &Mat,
    frames: usize,
    seq: usize,
) -> FocrResult<Mat> {
    let q = blk.q.apply(x)?;
    let k = blk.k.apply(x)?;
    let v = blk.v.apply(x)?;

    // Repack [F·seq, 768] → frame-then-head-major [F·heads, seq, 64].
    let head_span = seq * HEAD_DIM;
    let mut qf = vec![0.0f32; frames * NUM_HEADS * head_span];
    let mut kf = vec![0.0f32; frames * NUM_HEADS * head_span];
    let mut vf = vec![0.0f32; frames * NUM_HEADS * head_span];
    for f in 0..frames {
        for s in 0..seq {
            let row = f * seq + s;
            let (qr, kr, vr) = (q.row(row), k.row(row), v.row(row));
            for h in 0..NUM_HEADS {
                let src = h * HEAD_DIM;
                let dst = (f * NUM_HEADS + h) * head_span + s * HEAD_DIM;
                qf[dst..dst + HEAD_DIM].copy_from_slice(&qr[src..src + HEAD_DIM]);
                kf[dst..dst + HEAD_DIM].copy_from_slice(&kr[src..src + HEAD_DIM]);
                vf[dst..dst + HEAD_DIM].copy_from_slice(&vr[src..src + HEAD_DIM]);
            }
        }
    }

    let ctx = nn::sdpa(
        &qf,
        &kf,
        &vf,
        frames * NUM_HEADS,
        seq,
        seq,
        HEAD_DIM,
        HEAD_DIM,
        ATTN_SCALE,
        false,
    );

    // Unpack back to [F·seq, 768] rows.
    let mut merged = Mat::zeros(frames * seq, EMBED_DIM);
    for f in 0..frames {
        for h in 0..NUM_HEADS {
            for s in 0..seq {
                let src = (f * NUM_HEADS + h) * head_span + s * HEAD_DIM;
                let dst_row = merged.row_mut(f * seq + s);
                dst_row[h * HEAD_DIM..(h + 1) * HEAD_DIM]
                    .copy_from_slice(&ctx[src..src + HEAD_DIM]);
            }
        }
    }
    blk.out.apply(&merged)
}

/// One pre-LN encoder block over an `[F·seq, dim]` stacked buffer: identical
/// to [`encoder_block`] except the attention is frame-block-diagonal. The
/// norms, MLP, activation, and residuals are row-wise, so stacking is a no-op
/// on their math.
fn encoder_block_batched(
    blk: &SiglipBlockP,
    x: &mut Mat,
    frames: usize,
    seq: usize,
) -> FocrResult<()> {
    let h = nn::layer_norm(x, Some(&blk.ln1.w), Some(&blk.ln1.b), LN_EPS)?;
    let attn = self_attention_batched(blk, &h, frames, seq)?;
    add_assign(x, &attn)?;

    let h2 = nn::layer_norm(x, Some(&blk.ln2.w), Some(&blk.ln2.b), LN_EPS)?;
    let mut m = blk.fc1.apply(&h2)?;
    nn::gelu_tanh(&mut m);
    let m = blk.fc2.apply(&m)?;
    add_assign(x, &m)
}

/// [`forward_frames`] with the transformer stack run ONCE over all frames
/// stacked `[F·1024, 768]` (bd-av64.10): every GEMM sees M = F·1024 instead
/// of 13 small ramps, and the SDPA is frame-block-diagonal. Byte-identical to
/// the sequential path (proven by `batched_frames_match_sequential_byte_for_byte`
/// on the real block geometry, plus the armed oracle certs); the per-frame
/// patch embed stays per-frame (conv, small).
///
/// # Errors
/// [`FocrError::Other`] on a length/frame-count mismatch or any kernel-shape
/// violation.
pub fn forward_frames_batched(
    w: &SiglipWeights,
    pixels: &[f32],
    n_frames: usize,
) -> FocrResult<Vec<Mat>> {
    let frame_len = 3 * IMG_SIDE * IMG_SIDE;
    if n_frames == 0 || pixels.len() != n_frames * frame_len {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_siglip forward_frames_batched: buffer len {} != n_frames {n_frames} * {frame_len}",
            pixels.len()
        )));
    }
    // Per-frame embeddings, stacked row-major into [F·TOKENS, EMBED_DIM].
    let mut x = Mat::zeros(n_frames * TOKENS, EMBED_DIM);
    for f in 0..n_frames {
        let e = embed_frame(w, &pixels[f * frame_len..(f + 1) * frame_len])?;
        x.data[f * TOKENS * EMBED_DIM..(f + 1) * TOKENS * EMBED_DIM].copy_from_slice(&e.data);
    }
    for blk in &w.blocks {
        encoder_block_batched(blk, &mut x, n_frames, TOKENS)?;
    }
    let post = nn::layer_norm(&x, Some(&w.post_ln.w), Some(&w.post_ln.b), LN_EPS)?;
    // Split back into one [TOKENS, EMBED_DIM] Mat per frame.
    let mut out = Vec::with_capacity(n_frames);
    for f in 0..n_frames {
        out.push(Mat::from_vec(
            TOKENS,
            EMBED_DIM,
            post.data[f * TOKENS * EMBED_DIM..(f + 1) * TOKENS * EMBED_DIM].to_vec(),
        ));
    }
    Ok(out)
}

/// `a += b`, shape-checked.
fn add_assign(a: &mut Mat, b: &Mat) -> FocrResult<()> {
    if a.shape() != b.shape() {
        return Err(FocrError::Other(anyhow::anyhow!(
            "vision_siglip residual: shape {:?} != {:?}",
            a.shape(),
            b.shape()
        )));
    }
    for (x, y) in a.data.iter_mut().zip(&b.data) {
        *x += y;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic tiny-value synthetic weights with the REAL per-block
    /// geometry (768/12-head/3072 — the packing the tower is specialized to)
    /// but a caller-chosen depth: the plumbing tests run depth 1, because a
    /// full 12-block forward is ~213 GFLOP (~90 s in a dev build) and depth
    /// adds nothing to what they prove — the armed `siglip_matches_torch_oracle`
    /// cert exercises the real 12 layers on the real weights.
    fn synthetic_weights_depth(depth: usize) -> SiglipWeights {
        let wave = |n: usize, f: f32, a: f32| -> Vec<f32> {
            (0..n).map(|i| (i as f32 * f).sin() * a).collect()
        };
        let linear = |out: usize, in_: usize, seed: f32| {
            let w = wave(out * in_, 0.13 + seed, 0.02);
            Linear::from_row_major(&w, wave(out, 0.7 + seed, 0.01), out, in_)
                .expect("synthetic linear shape")
        };
        let ln = |seed: f32| LayerNormP {
            w: (0..EMBED_DIM)
                .map(|i| 1.0 + (i as f32 * seed).sin() * 0.05)
                .collect(),
            b: wave(EMBED_DIM, 0.3 + seed, 0.01),
        };
        let blocks = (0..depth)
            .map(|i| {
                let s = i as f32 * 0.01;
                SiglipBlockP {
                    ln1: ln(0.11 + s),
                    q: linear(EMBED_DIM, EMBED_DIM, s),
                    k: linear(EMBED_DIM, EMBED_DIM, s + 0.001),
                    v: linear(EMBED_DIM, EMBED_DIM, s + 0.002),
                    out: linear(EMBED_DIM, EMBED_DIM, s + 0.003),
                    ln2: ln(0.17 + s),
                    fc1: linear(INTERMEDIATE, EMBED_DIM, s + 0.004),
                    fc2: linear(EMBED_DIM, INTERMEDIATE, s + 0.005),
                }
            })
            .collect();
        SiglipWeights {
            patch_embed: Conv {
                w: wave(EMBED_DIM * 3 * PATCH * PATCH, 0.01, 0.05),
                b: Some(wave(EMBED_DIM, 0.5, 0.01)),
                out_ch: EMBED_DIM,
                in_ch: 3,
                kh: PATCH,
                kw: PATCH,
            },
            pos_embed: wave(TOKENS * EMBED_DIM, 0.023, 0.02),
            blocks,
            post_ln: ln(0.29),
        }
    }

    fn synthetic_weights() -> SiglipWeights {
        synthetic_weights_depth(1)
    }

    fn synthetic_pixels() -> Vec<f32> {
        (0..3 * IMG_SIDE * IMG_SIDE)
            .map(|i| ((i % 511) as f32 / 255.0) - 1.0)
            .collect()
    }

    #[test]
    fn forward_shapes_and_determinism() {
        let w = synthetic_weights();
        let px = synthetic_pixels();
        let a = forward_frame(&w, &px).expect("forward");
        assert_eq!((a.rows, a.cols), (TOKENS, EMBED_DIM));
        assert!(
            a.data.iter().all(|v| v.is_finite()),
            "non-finite activation"
        );
        // Bit-identical on a re-run (no hidden state, no RNG).
        let b = forward_frame(&w, &px).expect("forward twice");
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn pos_embed_moves_the_output() {
        // Zeroing the pos table must change the result (proves the add is live
        // and applied by identity ids).
        let w = synthetic_weights();
        let px = synthetic_pixels();
        let a = forward_frame(&w, &px).unwrap();
        let mut w2 = w.clone();
        w2.pos_embed = vec![0.0; TOKENS * EMBED_DIM];
        let b = forward_frame(&w2, &px).unwrap();
        assert_ne!(a.data, b.data);
    }

    #[test]
    fn attention_is_bidirectional() {
        // Perturbing the LAST patch must move the FIRST token's output — a
        // causal mask would forbid it. (Pixels of the last 16×16 patch live at
        // the tail of each channel plane.)
        let w = synthetic_weights();
        let mut px = synthetic_pixels();
        let a = forward_frame(&w, &px).unwrap();
        for c in 0..3 {
            let plane = (c + 1) * IMG_SIDE * IMG_SIDE;
            for v in &mut px[plane - PATCH..plane] {
                *v += 0.5;
            }
        }
        let b = forward_frame(&w, &px).unwrap();
        let first_a = &a.data[..EMBED_DIM];
        let first_b = &b.data[..EMBED_DIM];
        assert_ne!(first_a, first_b, "last-patch info did not reach token 0");
    }

    #[test]
    fn forward_frames_matches_per_frame() {
        let w = synthetic_weights();
        let px = synthetic_pixels();
        let mut two = px.clone();
        two.extend(px.iter().map(|v| -v));
        let outs = forward_frames(&w, &two, 2).unwrap();
        assert_eq!(outs.len(), 2);
        let single = forward_frame(&w, &px).unwrap();
        assert_eq!(outs[0].data, single.data, "frame 0 must equal solo forward");
        assert_ne!(outs[1].data, single.data);
    }

    #[test]
    fn error_handling() {
        let w = synthetic_weights();
        // Wrong pixel buffer length.
        assert!(forward_frame(&w, &[0.0; 100]).is_err());
        // Frame-count mismatch.
        assert!(forward_frames(&w, &[0.0; 100], 2).is_err());
        assert!(forward_frames(&w, &synthetic_pixels(), 0).is_err());
    }

    #[test]
    fn gelu_tanh_reference_values() {
        // Hand-checked against torch.nn.functional.gelu(x, approximate="tanh").
        assert_eq!(nn::gelu_tanh_scalar(0.0), 0.0);
        let close = |a: f32, b: f32| (a - b).abs() < 1e-6;
        assert!(close(nn::gelu_tanh_scalar(1.0), 0.841_192));
        assert!(close(nn::gelu_tanh_scalar(-1.0), -0.158_808));
        assert!(close(nn::gelu_tanh_scalar(3.0), 2.996_363));
        // Large |x| saturates to x / 0.
        assert!(close(nn::gelu_tanh_scalar(10.0), 10.0));
        assert!(close(nn::gelu_tanh_scalar(-10.0), 0.0));
    }

    /// **C3 L2 — per-seam bisect vs the torch oracle** (skip-with-SUCCESS
    /// without `FOCR_SMOLVLM2_DIR` or the dbg seam blobs): frame-0
    /// embeddings-out (`hidden_states[0]`) and block-0 out
    /// (`hidden_states[1]`), each held to cosine ≥ 0.9999. When the
    /// end-to-end cert fails, this localizes which stage diverged.
    #[test]
    fn siglip_seams_match_torch_oracle_frame0() {
        let Ok(dir) = std::env::var("FOCR_SMOLVLM2_DIR") else {
            return;
        };
        let pv_path = format!("{dir}/smolvlm2_pixel_values.bin");
        let h0_path = format!("{dir}/smolvlm2_dbg_vision_hidden_0_frame0.bin");
        let h1_path = format!("{dir}/smolvlm2_dbg_vision_hidden_1_frame0.bin");
        if !std::path::Path::new(&h0_path).is_file() {
            eprintln!("skip-with-SUCCESS: {h0_path} absent (npz→bin dbg extract)");
            return;
        }
        let read_f32 = |p: &str| -> Vec<f32> {
            std::fs::read(p)
                .expect("oracle blob reads")
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        let cos = |a: &[f32], b: &[f32]| -> f64 {
            let mut dot = 0.0f64;
            let (mut na, mut nb) = (0.0f64, 0.0f64);
            for (x, y) in a.iter().zip(b) {
                let (x, y) = (f64::from(*x), f64::from(*y));
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            dot / (na.sqrt() * nb.sqrt())
        };
        let pv = read_f32(&pv_path);
        let frame0 = &pv[..3 * IMG_SIDE * IMG_SIDE];
        let weights = Weights::load(std::path::Path::new(&format!("{dir}/model.safetensors")))
            .expect("weights load");
        let w = siglip_weights_from(&weights, "model.vision_model").expect("hydrate");

        let emb = embed_frame(&w, frame0).expect("embed");
        let h0 = read_f32(&h0_path);
        let c0 = cos(&emb.data, &h0);
        eprintln!("[C3 seam] embeddings-out cos={c0:.8}");

        let mut x = emb;
        encoder_block(&w.blocks[0], &mut x).expect("block 0");
        let h1 = read_f32(&h1_path);
        let c1 = cos(&x.data, &h1);
        eprintln!("[C3 seam] block-0-out    cos={c1:.8}");

        assert!(c0 >= 0.9999, "embeddings seam diverged: cos={c0:.8}");
        assert!(c1 >= 0.9999, "block-0 seam diverged: cos={c1:.8}");
    }

    // ── C3 parity cert (env-gated, real weights + oracle seams) ─────────────

    /// **C3 — SigLIP tower vs the torch oracle** (skip-with-SUCCESS without
    /// `FOCR_SMOLVLM2_DIR`): feed the oracle's own preprocessed
    /// `smolvlm2_pixel_values.bin` (seam-isolated — resize parity is its own
    /// rung, OQ-2/OQ-3), forward every frame, and hold the post-LN output to
    /// cosine ≥ 0.9999 per frame + a bounded max-abs against
    /// `smolvlm2_vision_post_ln.bin`.
    #[test]
    fn siglip_matches_torch_oracle() {
        let Ok(dir) = std::env::var("FOCR_SMOLVLM2_DIR") else {
            return;
        };
        let pv_path = format!("{dir}/smolvlm2_pixel_values.bin");
        let want_path = format!("{dir}/smolvlm2_vision_post_ln.bin");
        let model_path = format!("{dir}/model.safetensors");
        if !std::path::Path::new(&pv_path).is_file() {
            eprintln!("skip-with-SUCCESS: {pv_path} absent (run the vision oracle script)");
            return;
        }
        let read_f32 = |p: &str| -> Vec<f32> {
            std::fs::read(p)
                .expect("oracle blob reads")
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        let pv = read_f32(&pv_path);
        let frame_len = 3 * IMG_SIDE * IMG_SIDE;
        let n_frames = pv.len() / frame_len;
        assert_eq!(
            n_frames * frame_len,
            pv.len(),
            "pixel_values not [F,3,512,512]"
        );
        let want = read_f32(&want_path);
        assert_eq!(want.len(), n_frames * TOKENS * EMBED_DIM);

        let weights = Weights::load(std::path::Path::new(&model_path)).expect("weights load");
        let w = siglip_weights_from(&weights, "model.vision_model").expect("hydrate");
        let outs = forward_frames(&w, &pv, n_frames).expect("forward");

        let mut worst_cos = 1.0f64;
        let mut max_abs = 0.0f64;
        for (f, ours) in outs.iter().enumerate() {
            let oracle = &want[f * TOKENS * EMBED_DIM..(f + 1) * TOKENS * EMBED_DIM];
            let mut dot = 0.0f64;
            let (mut na, mut nb) = (0.0f64, 0.0f64);
            for (a, b) in ours.data.iter().zip(oracle) {
                let (a, b) = (f64::from(*a), f64::from(*b));
                dot += a * b;
                na += a * a;
                nb += b * b;
                max_abs = max_abs.max((a - b).abs());
            }
            let cos = dot / (na.sqrt() * nb.sqrt());
            worst_cos = worst_cos.min(cos);
        }
        eprintln!("[C3 parity] frames={n_frames} worst_cos={worst_cos:.8} maxabs={max_abs:.3e}");
        assert!(
            worst_cos >= 0.9999,
            "SigLIP per-frame cosine {worst_cos:.8} < 0.9999"
        );
        assert!(
            max_abs <= 1e-2,
            "SigLIP post-LN maxabs {max_abs:.3e} > 1e-2 — investigate before tightening"
        );
    }

    /// bd-av64.10: the frame-batched tower must equal the sequential path
    /// BIT-FOR-BIT — M-batched GEMMs keep each output row's K-reduction, and
    /// the SDPA blocks are frame-disjoint. Real per-block geometry (768/12
    /// heads/3072) at depth 1, three frames of distinct synthetic pixels.
    #[test]
    fn batched_frames_match_sequential_byte_for_byte() {
        let w = synthetic_weights_depth(1);
        let frames = 3usize;
        let frame_len = 3 * IMG_SIDE * IMG_SIDE;
        let pixels: Vec<f32> = (0..frames * frame_len)
            .map(|i| ((i % 251) as f32) / 251.0 - 0.5)
            .collect();
        let sequential = forward_frames(&w, &pixels, frames).expect("sequential");
        let batched = forward_frames_batched(&w, &pixels, frames).expect("batched");
        assert_eq!(sequential.len(), batched.len());
        for (f, (a, b)) in sequential.iter().zip(&batched).enumerate() {
            assert_eq!(a.shape(), b.shape(), "frame {f} shape");
            for (i, (x, y)) in a.data.iter().zip(&b.data).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "frame {f} element {i}: {x} vs {y}"
                );
            }
        }
    }

    #[test]
    fn batched_frames_reject_bad_buffer() {
        let w = synthetic_weights_depth(1);
        assert!(forward_frames_batched(&w, &[0.0; 7], 1).is_err());
        assert!(forward_frames_batched(&w, &[], 0).is_err());
    }
}
