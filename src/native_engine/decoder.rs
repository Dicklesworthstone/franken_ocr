//! The 12-layer DeepseekV2/Llama-MHA decoder driver ([SPEC-070..081],
//! PROPOSED_ARCHITECTURE.md §6.7).
//!
//! `embed_tokens` -> per layer `h = x + self_attn(input_layernorm(x))`;
//! `out = h + mlp(post_attention_layernorm(h))` -> final RMSNorm -> `lm_head`
//! GEMV 1280 -> 129280. RoPE = Llama variant (theta 10000, head_dim 128,
//! NEOX-style `rotate_half`, NOT the DeepseekV2 interleave — [SPEC-078]). Layer
//! 0 is dense MLP; layers 1..11 are MoE; all 12 layers use R-SWA attention.
//!
//! ## What lives here vs. the siblings
//!
//! This file is the *driver*: it owns the per-layer pre-norm residual loop
//! ([SPEC-072]), the token-embedding lookup ([SPEC-070]), the final norm, the
//! `lm_head` GEMV ([SPEC-081]), and the **RoPE** math ([SPEC-078]) applied to Q/K
//! before they are handed to attention. The attention kernel itself is
//! [`crate::native_engine::rswa::attention`]; the per-layer MLP/MoE block is
//! [`crate::native_engine::moe::forward`] (layers 1..11) or the dense SwiGLU MLP
//! ([SPEC-075], layer 0). Norms funnel through the [`crate::native_engine::nn`]
//! facade ([`nn::rms_norm`]); GEMMs through [`nn::matmul`].
//!
//! ## Weights plumbing (facade gap)
//!
//! [`super::weights::Weights`] is still a stub with no tensor accessors, so the
//! top-level [`forward`]/[`lm_head`] entrypoints that take `&Weights` cannot yet
//! pull the embedding table / per-layer norm weights / head matrix out of it —
//! they remain `NotImplemented` and will become a thin wiring shim once the
//! `.focrq` reader (bd-1es.3) exposes named tensors. ALL the load-bearing math —
//! token embed, RMSNorm composition, RoPE, the dense SwiGLU MLP, the lm_head
//! GEMV, and the full per-layer driver — is implemented and unit-tested here as
//! pure functions over explicit `&[f32]` / [`Mat`] weight slices, so wiring is a
//! mechanical hookup with zero remaining numerics risk.

use super::nn;
use super::tensor::Mat;
use super::weights::Weights;
use crate::error::{FocrError, FocrResult};

/// Decoder hyperparameters (compile-time shapes — plan §1.1 P2). Mirrors the
/// pinned `config.json` so the forward never reads runtime params.
pub mod config {
    /// Hidden size ([SPEC-010]).
    pub const HIDDEN_SIZE: usize = 1280;
    /// Dense MLP intermediate size ([SPEC-010]).
    pub const INTERMEDIATE_SIZE: usize = 6848;
    /// Decoder layer count ([SPEC-010]).
    pub const NUM_HIDDEN_LAYERS: usize = 12;
    /// Attention heads ([SPEC-010]); MHA (kv heads = heads, no GQA).
    pub const NUM_ATTENTION_HEADS: usize = 10;
    /// KV heads ([SPEC-010]).
    pub const NUM_KEY_VALUE_HEADS: usize = 10;
    /// Per-head dim (`hidden / heads`) ([SPEC-011]).
    pub const HEAD_DIM: usize = 128;
    /// Vocabulary size ([SPEC-010]).
    pub const VOCAB_SIZE: usize = 129280;
    /// RoPE theta ([SPEC-013/078]).
    pub const ROPE_THETA: f32 = 10000.0;
    /// RMSNorm epsilon ([SPEC-013/071]).
    pub const RMS_NORM_EPS: f32 = 1e-6;
    /// First dense layer count (layer 0 dense) ([SPEC-012]).
    pub const FIRST_K_DENSE_REPLACE: usize = 1;
    /// R-SWA window ([SPEC-015/090]).
    pub const RING_WINDOW: usize = 128;
    /// BOS token id ([SPEC-014]).
    pub const BOS_TOKEN_ID: u32 = 0;
    /// EOS token id ([SPEC-014]).
    pub const EOS_TOKEN_ID: u32 = 1;
}

// ── Token embedding ([SPEC-070]) ────────────────────────────────────────────

/// Gather rows of the `embed_tokens` table for a sequence of token ids
/// ([SPEC-070]).
///
/// `table` is the row-major `[vocab, hidden]` embedding matrix
/// (`model.embed_tokens.weight`); `ids` is the prompt/decode token sequence.
/// Returns the `[seq, hidden]` activation [`Mat`] that begins the residual
/// stream — exactly the `inputs_embeds` the connector then masked-scatters
/// vision features into.
///
/// # Errors
/// [`FocrError::Other`] if `table.len() != vocab * hidden`, or any id is
/// `>= vocab`.
pub fn embed_tokens(table: &[f32], vocab: usize, hidden: usize, ids: &[u32]) -> FocrResult<Mat> {
    if table.len() != vocab * hidden {
        return Err(FocrError::Other(anyhow::anyhow!(
            "embed_tokens: table len {} != vocab*hidden {}",
            table.len(),
            vocab * hidden
        )));
    }
    let seq = ids.len();
    let mut out = vec![0.0f32; seq * hidden];
    for (t, &id) in ids.iter().enumerate() {
        let row = id as usize;
        if row >= vocab {
            return Err(FocrError::Other(anyhow::anyhow!(
                "embed_tokens: id {row} out of range (vocab {vocab})"
            )));
        }
        let src = &table[row * hidden..(row + 1) * hidden];
        out[t * hidden..(t + 1) * hidden].copy_from_slice(src);
    }
    Ok(Mat::from_vec(seq, hidden, out))
}

// ── RoPE ([SPEC-078], Llama-style NEOX rotate_half) ─────────────────────────

/// Precomputed RoPE phases for the prompt positions: `cos[p][i]`, `sin[p][i]`
/// laid out `[seq, head_dim]` row-major, with the half-dim freqs duplicated into
/// both halves (Llama `cat(freqs, freqs)`), so they can be applied head-by-head
/// with a single index.
///
/// `inv_freq[i] = theta^(-2i/head_dim)` for `i in 0..head_dim/2`; for position
/// `p`, `angle = p * inv_freq[i]`. `cos`/`sin` each duplicate `i` into columns
/// `i` and `i + head_dim/2`.
#[derive(Debug, Clone)]
pub struct RopeTable {
    /// Per-(position, channel) cosine, row-major `[seq, head_dim]`.
    pub cos: Vec<f32>,
    /// Per-(position, channel) sine, row-major `[seq, head_dim]`.
    pub sin: Vec<f32>,
    /// Per-head dimension (here 128).
    pub head_dim: usize,
}

impl RopeTable {
    /// Build the cos/sin tables for the given absolute `position_ids`
    /// ([SPEC-095]: ALWAYS the true logical positions, never ring slots).
    ///
    /// Vanilla un-scaled base-`theta` RoPE (OQ-5: no YARN / NTK / linear
    /// scaling). `head_dim` must be even.
    ///
    /// # Panics
    /// Panics if `head_dim` is not even.
    #[must_use]
    pub fn build(position_ids: &[usize], head_dim: usize, theta: f32) -> Self {
        assert!(
            head_dim.is_multiple_of(2),
            "RopeTable: head_dim must be even"
        );
        let half = head_dim / 2;
        // inv_freq[i] = theta^(-2i/head_dim) = theta^(-(i/half)).  Computed in
        // f64 then narrowed — matches the HF float32 rotary embedding closely.
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                let exponent = (2 * i) as f64 / head_dim as f64;
                (1.0 / (theta as f64).powf(exponent)) as f32
            })
            .collect();
        let seq = position_ids.len();
        let mut cos = vec![0.0f32; seq * head_dim];
        let mut sin = vec![0.0f32; seq * head_dim];
        for (p_idx, &pos) in position_ids.iter().enumerate() {
            let base = p_idx * head_dim;
            for i in 0..half {
                let angle = pos as f32 * inv_freq[i];
                let (s, c) = angle.sin_cos();
                // Llama cat(freqs, freqs): freq i lands in column i and i+half.
                cos[base + i] = c;
                cos[base + half + i] = c;
                sin[base + i] = s;
                sin[base + half + i] = s;
            }
        }
        Self { cos, sin, head_dim }
    }
}

/// Apply Llama-style RoPE in place to a `[seq, num_heads * head_dim]` Q or K
/// activation ([SPEC-078]).
///
/// Each head's `head_dim` block is rotated by `rotate_half`:
/// `out = x * cos + rotate_half(x) * sin`, where
/// `rotate_half([a; b]) = [-b; a]` over the two contiguous halves (NEOX layout,
/// NOT the DeepseekV2 interleave). `rope` must have been built from the SAME
/// positions as the rows of `x` ([SPEC-095]).
///
/// # Errors
/// [`FocrError::Other`] on a shape mismatch (`x.cols % head_dim != 0`,
/// `x.rows != positions`, …).
pub fn apply_rope(x: &mut Mat, rope: &RopeTable) -> FocrResult<()> {
    let head_dim = rope.head_dim;
    if head_dim == 0 || !x.cols.is_multiple_of(head_dim) {
        return Err(FocrError::Other(anyhow::anyhow!(
            "apply_rope: cols {} not a multiple of head_dim {head_dim}",
            x.cols
        )));
    }
    let seq = x.rows;
    if rope.cos.len() != seq * head_dim {
        return Err(FocrError::Other(anyhow::anyhow!(
            "apply_rope: rope built for {} positions, x has {seq} rows",
            rope.cos.len() / head_dim
        )));
    }
    let num_heads = x.cols / head_dim;
    let half = head_dim / 2;
    for t in 0..seq {
        let rope_base = t * head_dim;
        let row = x.row_mut(t);
        for h in 0..num_heads {
            let hb = h * head_dim;
            // rotate_half over the two halves of THIS head's block.
            for i in 0..half {
                let a = row[hb + i]; // first half
                let b = row[hb + half + i]; // second half
                let cos_a = rope.cos[rope_base + i];
                let sin_a = rope.sin[rope_base + i];
                // rotate_half([a,b]) -> first half gets -b, second half gets a.
                row[hb + i] = a * cos_a - b * sin_a;
                row[hb + half + i] = b * cos_a + a * sin_a;
            }
        }
    }
    Ok(())
}

// ── Dense SwiGLU MLP ([SPEC-075], layer 0 & a shared-expert primitive) ───────

/// SwiGLU feed-forward `down(silu(gate(x)) * up(x))` over `[seq, hidden]`
/// activations ([SPEC-075]).
///
/// `gate_w` / `up_w` are row-major `[inter, hidden]` (`F.linear` weight layout,
/// `out_features x in_features`); `down_w` is `[hidden, inter]`. `act_fn = silu`.
/// This is the layer-0 dense MLP and the exact primitive shape the MoE shared /
/// routed experts reuse, so it lives here next to the driver and is shared by
/// the MoE module if needed.
///
/// # Errors
/// [`FocrError::Other`] on any weight-shape mismatch (propagated from the
/// underlying GEMMs).
pub fn dense_mlp(
    x: &Mat,
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    hidden: usize,
    inter: usize,
) -> FocrResult<Mat> {
    if x.cols != hidden {
        return Err(FocrError::Other(anyhow::anyhow!(
            "dense_mlp: x.cols {} != hidden {hidden}",
            x.cols
        )));
    }
    // gate = x @ gate_w^T  -> [seq, inter]
    let mut gate = linear_no_bias(x, gate_w, hidden, inter)?;
    // up   = x @ up_w^T    -> [seq, inter]
    let up = linear_no_bias(x, up_w, hidden, inter)?;
    // silu(gate) * up, elementwise.
    nn::silu(&mut gate);
    for (g, u) in gate.data.iter_mut().zip(up.data.iter()) {
        *g *= *u;
    }
    // down = (silu(gate)*up) @ down_w^T -> [seq, hidden]
    linear_no_bias(&gate, down_w, inter, hidden)
}

/// Bias-free linear `y = x @ w^T` where `w` is the PyTorch `[out, in]` row-major
/// weight (`F.linear` convention) — transposed on the fly into the `[in, out]`
/// layout [`nn::matmul`] contracts over.
///
/// Small/clear: builds the transpose once (the GEMM dominates), keeping the
/// driver readable. The int8 path ([`nn::linear_int8_dynamic`]) is the
/// kill-switch-gated perf swap and slots in here unchanged.
///
/// # Errors
/// [`FocrError::Other`] if `x.cols != in_` or `w.len() != out * in_`.
fn linear_no_bias(x: &Mat, w: &[f32], in_: usize, out: usize) -> FocrResult<Mat> {
    if x.cols != in_ {
        return Err(FocrError::Other(anyhow::anyhow!(
            "linear_no_bias: x.cols {} != in {in_}",
            x.cols
        )));
    }
    if w.len() != out * in_ {
        return Err(FocrError::Other(anyhow::anyhow!(
            "linear_no_bias: w len {} != out*in {}",
            w.len(),
            out * in_
        )));
    }
    // Transpose [out, in] -> [in, out] so matmul does [seq,in] x [in,out].
    let mut wt = vec![0.0f32; in_ * out];
    for o in 0..out {
        for i in 0..in_ {
            wt[i * out + o] = w[o * in_ + i];
        }
    }
    let w_mat = Mat::from_vec(in_, out, wt);
    nn::matmul(x, &w_mat)
}

// ── Final norm + lm_head ([SPEC-071], [SPEC-081]) ───────────────────────────

/// Final `model.norm` RMSNorm then `lm_head` projection `[seq, hidden] ->
/// [seq, vocab]` ([SPEC-071]/[SPEC-081]).
///
/// `norm_w` is the `model.norm.weight` (length `hidden`); `head_w` is the
/// row-major `[vocab, hidden]` `lm_head.weight` (non-tied — `tie_word_embeddings
/// = false`). Returns the f32 logits (HF casts to float for sampling; we are
/// already f32).
///
/// # Errors
/// [`FocrError::Other`] on any shape mismatch.
pub fn norm_and_lm_head(
    hidden: &Mat,
    norm_w: &[f32],
    head_w: &[f32],
    vocab: usize,
    eps: f32,
) -> FocrResult<Mat> {
    let normed = nn::rms_norm(hidden, Some(norm_w), eps)?;
    lm_head_proj(&normed, head_w, vocab)
}

/// Project (already-normed) hidden states to vocab logits — the bare
/// `lm_head` GEMM `[seq, hidden] x [vocab, hidden]^T -> [seq, vocab]`
/// ([SPEC-081]).
///
/// `head_w` is row-major `[vocab, hidden]`. For the single-token decode step
/// (`seq == 1`) this is a GEMV.
///
/// # Errors
/// [`FocrError::Other`] if `head_w.len() != vocab * hidden`.
pub fn lm_head_proj(hidden: &Mat, head_w: &[f32], vocab: usize) -> FocrResult<Mat> {
    let h = hidden.cols;
    linear_no_bias(hidden, head_w, h, vocab)
}

// ── Per-layer driver ([SPEC-072]) ──────────────────────────────────────────

/// The fp32 weights one decoder layer needs from the driver's point of view.
///
/// Norms + projections are explicit slices so the driver is fully testable
/// before the `.focrq` reader exists; the attention/MLP kernels consume the
/// rest through their own modules. `gate_w`/`up_w`/`down_w` are populated for the
/// **dense** layer-0 path; MoE layers (1..11) route through
/// [`crate::native_engine::moe::forward`] and ignore them.
#[derive(Debug, Clone)]
pub struct LayerWeights<'a> {
    /// `input_layernorm.weight`, length `hidden`.
    pub input_ln: &'a [f32],
    /// `post_attention_layernorm.weight`, length `hidden`.
    pub post_attn_ln: &'a [f32],
    /// `q_proj.weight`, row-major `[num_heads*head_dim, hidden]`.
    pub q_proj: &'a [f32],
    /// `k_proj.weight`, row-major `[num_kv_heads*head_dim, hidden]`.
    pub k_proj: &'a [f32],
    /// `v_proj.weight`, row-major `[num_kv_heads*head_dim, hidden]`.
    pub v_proj: &'a [f32],
    /// `o_proj.weight`, row-major `[hidden, num_heads*head_dim]`.
    pub o_proj: &'a [f32],
    /// Dense `gate_proj.weight` `[inter, hidden]` (layer 0 only).
    pub gate_w: &'a [f32],
    /// Dense `up_proj.weight` `[inter, hidden]` (layer 0 only).
    pub up_w: &'a [f32],
    /// Dense `down_proj.weight` `[hidden, inter]` (layer 0 only).
    pub down_w: &'a [f32],
}

/// Project Q/K/V from the (already input-normed) hidden states and apply RoPE to
/// Q and K ([SPEC-078]/[SPEC-090]).
///
/// Returns `(q, k, v)`, each `[seq, num_heads*head_dim]`, ready to hand to
/// [`crate::native_engine::rswa::attention`]. No QKV bias ([SPEC-090],
/// `attention_bias=false`).
///
/// # Errors
/// [`FocrError::Other`] on a projection / RoPE shape mismatch.
pub fn qkv_with_rope(
    normed: &Mat,
    lw: &LayerWeights<'_>,
    rope: &RopeTable,
    hidden: usize,
    qkv_dim: usize,
) -> FocrResult<(Mat, Mat, Mat)> {
    let mut q = linear_no_bias(normed, lw.q_proj, hidden, qkv_dim)?;
    let mut k = linear_no_bias(normed, lw.k_proj, hidden, qkv_dim)?;
    let v = linear_no_bias(normed, lw.v_proj, hidden, qkv_dim)?;
    apply_rope(&mut q, rope)?;
    apply_rope(&mut k, rope)?;
    Ok((q, k, v))
}

/// Self-attention sub-block for one layer, *given* the raw attention context
/// (the `[seq, num_heads*head_dim]` output of R-SWA, BEFORE `o_proj`).
///
/// Applies `o_proj`: `attn_out = context @ o_proj^T`, returning `[seq, hidden]`.
/// Split out so the driver can call the real [`crate::native_engine::rswa`]
/// kernel for the context and keep the projection here (and so it is unit
/// testable without the ring cache).
///
/// # Errors
/// [`FocrError::Other`] on a shape mismatch.
pub fn attn_output_proj(
    context: &Mat,
    o_proj: &[f32],
    hidden: usize,
    qkv_dim: usize,
) -> FocrResult<Mat> {
    linear_no_bias(context, o_proj, qkv_dim, hidden)
}

/// One full decoder layer, pre-norm residual ([SPEC-072]):
/// `h = x + o_proj(attn(rope(qkv(input_ln(x)))))`;
/// `out = h + mlp(post_attention_ln(h))`.
///
/// `attn_context` is a closure that runs the R-SWA attention kernel over the
/// rope'd `(q, k, v)` and returns the pre-`o_proj` context `[seq,
/// num_heads*head_dim]` — injected so this driver is testable against a pure
/// reference attention (and so the real ring-cache kernel
/// [`crate::native_engine::rswa::attention`] plugs in at the call site without
/// this function owning the per-layer cache). `mlp` is a closure that runs the
/// layer's MLP/MoE block over `[seq, hidden]` (dense for layer 0 via
/// [`dense_mlp`]; [`crate::native_engine::moe::forward`] for layers 1..11).
///
/// # Errors
/// Propagates any sub-step error.
// decoder forward: args are model state + tensors
#[allow(clippy::too_many_arguments)]
pub fn layer_forward<A, M>(
    x: &Mat,
    lw: &LayerWeights<'_>,
    rope: &RopeTable,
    hidden: usize,
    qkv_dim: usize,
    eps: f32,
    attn_context: A,
    mlp: M,
) -> FocrResult<Mat>
where
    A: FnOnce(&Mat, &Mat, &Mat) -> FocrResult<Mat>,
    M: FnOnce(&Mat) -> FocrResult<Mat>,
{
    // --- attention sub-block ---
    let normed = nn::rms_norm(x, Some(lw.input_ln), eps)?;
    let (q, k, v) = qkv_with_rope(&normed, lw, rope, hidden, qkv_dim)?;
    let context = attn_context(&q, &k, &v)?;
    let attn_out = attn_output_proj(&context, lw.o_proj, hidden, qkv_dim)?;
    let h = add_residual(x, &attn_out)?;

    // --- MLP / MoE sub-block ---
    let normed2 = nn::rms_norm(&h, Some(lw.post_attn_ln), eps)?;
    let mlp_out = mlp(&normed2)?;
    add_residual(&h, &mlp_out)
}

/// Elementwise residual add `a + b` for two equal-shaped activation [`Mat`]s.
///
/// # Errors
/// [`FocrError::Other`] if the shapes differ.
pub fn add_residual(a: &Mat, b: &Mat) -> FocrResult<Mat> {
    if a.shape() != b.shape() {
        return Err(FocrError::Other(anyhow::anyhow!(
            "add_residual: shape mismatch {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }
    let data = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x + y)
        .collect::<Vec<_>>();
    Ok(Mat::from_vec(a.rows, a.cols, data))
}

// ── Top-level entrypoints (Weights-backed; pending the .focrq reader) ───────

/// Project final hidden states to vocabulary logits (`lm_head`, [SPEC-081]).
///
/// # Errors
/// [`FocrError::NotImplemented`] until [`Weights`] exposes the `model.norm` /
/// `lm_head.weight` tensors (bd-1es.3). The math is implemented and tested in
/// [`norm_and_lm_head`] / [`lm_head_proj`]; this is a wiring shim awaiting the
/// reader.
pub fn lm_head(_weights: &Weights, _hidden: &Mat) -> FocrResult<Mat> {
    Err(FocrError::NotImplemented(
        "native_engine::decoder::lm_head — needs Weights tensor accessors (bd-1es.3); \
         math lives in decoder::norm_and_lm_head / lm_head_proj"
            .into(),
    ))
}

/// Run the full 12-layer decoder over `inputs_embeds`, returning the final
/// normalized hidden states (prefill); decode steps reuse the KV cache.
///
/// # Errors
/// [`FocrError::NotImplemented`] until [`Weights`] exposes the per-layer
/// norm/projection/expert tensors + the ring-cache plumbing (bd-1es.3 /
/// bd-1gv.17). The per-layer driver is implemented and tested in
/// [`layer_forward`]; this is the wiring shim awaiting those accessors.
pub fn forward(_weights: &Weights, _inputs_embeds: &Mat) -> FocrResult<Mat> {
    Err(FocrError::NotImplemented(
        "native_engine::decoder::forward — needs Weights tensor accessors + RingCache wiring \
         (bd-1es.3 / bd-1gv.17); per-layer math lives in decoder::layer_forward"
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Token embedding ([SPEC-070]) ────────────────────────────────────────

    #[test]
    fn embed_tokens_gathers_rows() {
        // vocab=3, hidden=2 table: row0=[0,1] row1=[2,3] row2=[4,5]
        let table = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let out = embed_tokens(&table, 3, 2, &[2, 0, 1, 2]).unwrap();
        assert_eq!(out.shape(), (4, 2));
        assert_eq!(out.row(0), &[4.0, 5.0]);
        assert_eq!(out.row(1), &[0.0, 1.0]);
        assert_eq!(out.row(2), &[2.0, 3.0]);
        assert_eq!(out.row(3), &[4.0, 5.0]);
    }

    #[test]
    fn embed_tokens_rejects_oob_id() {
        let table = vec![0.0, 1.0, 2.0, 3.0];
        assert!(embed_tokens(&table, 2, 2, &[5]).is_err());
    }

    #[test]
    fn embed_tokens_rejects_bad_table_len() {
        let table = vec![0.0, 1.0, 2.0];
        assert!(embed_tokens(&table, 2, 2, &[0]).is_err());
    }

    // ── RoPE ([SPEC-078]) ───────────────────────────────────────────────────

    #[test]
    fn rope_position_zero_is_identity() {
        // head_dim=4, single head; pos 0 => cos=1, sin=0 => x unchanged.
        let rope = RopeTable::build(&[0], 4, 10000.0);
        let mut x = Mat::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
        apply_rope(&mut x, &rope).unwrap();
        for (got, want) in x.data.iter().zip([1.0f32, 2.0, 3.0, 4.0].iter()) {
            assert!((got - want).abs() < 1e-6, "{got} != {want}");
        }
    }

    #[test]
    fn rope_matches_hand_computed_rotate_half() {
        // head_dim=2 => half=1, inv_freq[0] = 10000^0 = 1, so angle = pos.
        // rotate_half([a,b]) = [-b, a].
        // out[0] = a*cos - b*sin ; out[1] = b*cos + a*sin.
        let theta = 10000.0f32;
        let pos = 1usize;
        let rope = RopeTable::build(&[pos], 2, theta);
        let (a, b) = (3.0f32, 5.0f32);
        let mut x = Mat::from_vec(1, 2, vec![a, b]);
        apply_rope(&mut x, &rope).unwrap();
        let (s, c) = (pos as f32).sin_cos();
        let want0 = a * c - b * s;
        let want1 = b * c + a * s;
        assert!((x.data[0] - want0).abs() < 1e-5, "{} != {want0}", x.data[0]);
        assert!((x.data[1] - want1).abs() < 1e-5, "{} != {want1}", x.data[1]);
    }

    #[test]
    fn rope_preserves_norm_per_head() {
        // A rotation preserves the L2 norm of each head block.
        let rope = RopeTable::build(&[7, 13], 4, 10000.0);
        let mut x = Mat::from_vec(2, 8, (0..16).map(|i| (i as f32) * 0.25 - 2.0).collect());
        let head_norms = |data: &[f32]| -> Vec<f32> {
            let mut out = Vec::with_capacity(4);
            for t in 0..2 {
                for h in 0..2 {
                    let r = &data[t * 8 + h * 4..t * 8 + h * 4 + 4];
                    out.push(r.iter().map(|v| v * v).sum::<f32>());
                }
            }
            out
        };
        let before = head_norms(&x.data);
        apply_rope(&mut x, &rope).unwrap();
        let after = head_norms(&x.data);
        for (b, a) in before.iter().zip(after.iter()) {
            assert!((b - a).abs() < 1e-4, "norm changed: {b} -> {a}");
        }
    }

    #[test]
    fn rope_two_heads_share_phase() {
        // head_dim=2, two heads packed in one row -> each head rotated by the
        // SAME phase (RoPE is per-position, shared across heads).
        let rope = RopeTable::build(&[2], 2, 10000.0);
        let mut x = Mat::from_vec(1, 4, vec![1.0, 0.0, 0.0, 1.0]);
        apply_rope(&mut x, &rope).unwrap();
        let (s, c) = (2.0f32).sin_cos();
        // head0 = [1,0] -> [c, s]; head1 = [0,1] -> [-s, c]
        assert!((x.data[0] - c).abs() < 1e-5);
        assert!((x.data[1] - s).abs() < 1e-5);
        assert!((x.data[2] - (-s)).abs() < 1e-5);
        assert!((x.data[3] - c).abs() < 1e-5);
    }

    #[test]
    fn rope_rejects_bad_shape() {
        let rope = RopeTable::build(&[0, 1], 4, 10000.0);
        // cols (3) not a multiple of head_dim (4)
        let mut bad = Mat::zeros(2, 3);
        assert!(apply_rope(&mut bad, &rope).is_err());
        // wrong number of rows vs positions
        let mut wrong_rows = Mat::zeros(3, 4);
        assert!(apply_rope(&mut wrong_rows, &rope).is_err());
    }

    // ── Dense SwiGLU MLP ([SPEC-075]) ───────────────────────────────────────

    #[test]
    fn dense_mlp_matches_hand_computed() {
        // hidden=2, inter=2, single token x=[1,0].
        // gate_w=[[1,0],[0,1]] => gate = [1, 0]
        // up_w  =[[1,1],[1,1]] => up   = [1, 1]
        // silu(gate) = [silu(1), silu(0)] = [0.7310586, 0]
        // silu(gate)*up = [0.7310586, 0]
        // down_w=[[1,0],[0,1]] => out = [0.7310586, 0]
        let x = Mat::from_vec(1, 2, vec![1.0, 0.0]);
        let gate_w = vec![1.0, 0.0, 0.0, 1.0];
        let up_w = vec![1.0, 1.0, 1.0, 1.0];
        let down_w = vec![1.0, 0.0, 0.0, 1.0];
        let out = dense_mlp(&x, &gate_w, &up_w, &down_w, 2, 2).unwrap();
        assert_eq!(out.shape(), (1, 2));
        let silu1 = 1.0f32 / (1.0 + (-1.0f32).exp());
        assert!((out.data[0] - silu1).abs() < 1e-5, "{}", out.data[0]);
        assert!(out.data[1].abs() < 1e-6, "{}", out.data[1]);
    }

    #[test]
    fn dense_mlp_rejects_bad_hidden() {
        let x = Mat::from_vec(1, 3, vec![1.0, 0.0, 0.0]);
        assert!(dense_mlp(&x, &[1.0, 0.0], &[1.0, 0.0], &[1.0, 0.0], 2, 1).is_err());
    }

    // ── linear_no_bias (the F.linear [out,in] transpose) ────────────────────

    #[test]
    fn linear_no_bias_transposes_pytorch_layout() {
        // x=[1,2] (1x2 row); w=[out=3, in=2] row-major:
        //   w = [[1,0],[0,1],[1,1]]  => y = [x·w0, x·w1, x·w2] = [1, 2, 3]
        let x = Mat::from_vec(1, 2, vec![1.0, 2.0]);
        let w = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let y = linear_no_bias(&x, &w, 2, 3).unwrap();
        assert_eq!(y.shape(), (1, 3));
        assert_eq!(y.data, vec![1.0, 2.0, 3.0]);
    }

    // ── Residual add ([SPEC-072]) ───────────────────────────────────────────

    #[test]
    fn add_residual_sums_elementwise() {
        let a = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Mat::from_vec(2, 2, vec![10.0, 20.0, 30.0, 40.0]);
        let c = add_residual(&a, &b).unwrap();
        assert_eq!(c.data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn add_residual_rejects_shape_mismatch() {
        let a = Mat::zeros(2, 2);
        let b = Mat::zeros(2, 3);
        assert!(add_residual(&a, &b).is_err());
    }

    // ── Final norm + lm_head ([SPEC-071]/[SPEC-081]) ────────────────────────

    #[test]
    fn lm_head_proj_matches_matmul() {
        // hidden=2 vocab=3; head_w=[vocab,hidden] = [[1,0],[0,1],[1,1]]
        // hidden state [3,4] -> logits [3, 4, 7]
        let h = Mat::from_vec(1, 2, vec![3.0, 4.0]);
        let head_w = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let logits = lm_head_proj(&h, &head_w, 3).unwrap();
        assert_eq!(logits.shape(), (1, 3));
        assert_eq!(logits.data, vec![3.0, 4.0, 7.0]);
    }

    #[test]
    fn norm_and_lm_head_composes_rmsnorm_then_head() {
        // hidden=2, vocab=2. x=[3,4], norm_w=[1,1], eps=0.
        // rmsnorm: mean(x^2)=12.5, rstd=1/sqrt(12.5); normed=[3*rstd,4*rstd]
        // head_w = identity [[1,0],[0,1]] => logits = normed
        let h = Mat::from_vec(1, 2, vec![3.0, 4.0]);
        let norm_w = vec![1.0, 1.0];
        let head_w = vec![1.0, 0.0, 0.0, 1.0];
        let logits = norm_and_lm_head(&h, &norm_w, &head_w, 2, 0.0).unwrap();
        let rstd = 1.0f32 / 12.5f32.sqrt();
        assert!((logits.data[0] - 3.0 * rstd).abs() < 1e-5);
        assert!((logits.data[1] - 4.0 * rstd).abs() < 1e-5);
    }

    #[test]
    fn lm_head_last_row_is_full_last_row() {
        // The "compute only the rows you read" structural win: the decode driver
        // feeds lm_head ONLY the last hidden row (mod.rs). RMSNorm and the lm_head
        // linear are per-row independent, so the last-row slice MUST be exactly
        // (bit-for-bit) the last row of projecting all rows — proving the prefill
        // optimization (skip the [seq-1, vocab] head GEMM) changes nothing observable.
        let (seq, hidden, vocab) = (5usize, 8usize, 7usize);
        let h: Vec<f32> = (0..seq * hidden)
            .map(|i| ((i as f32) * 0.37).sin())
            .collect();
        let full = Mat::from_vec(seq, hidden, h);
        let norm_w: Vec<f32> = (0..hidden).map(|i| 0.5 + (i as f32) * 0.1).collect();
        let head_w: Vec<f32> = (0..vocab * hidden)
            .map(|i| ((i as f32) * 0.13).cos())
            .collect();
        let eps = 1e-6;

        let full_logits = norm_and_lm_head(&full, &norm_w, &head_w, vocab, eps).unwrap();
        let last = Mat::from_vec(1, hidden, full.row(seq - 1).to_vec());
        let last_logits = norm_and_lm_head(&last, &norm_w, &head_w, vocab, eps).unwrap();

        assert_eq!((last_logits.rows, last_logits.cols), (1, vocab));
        let full_last = &full_logits.data[(seq - 1) * vocab..seq * vocab];
        assert_eq!(
            last_logits.data, full_last,
            "last-row lm_head must be bit-identical to full[last]"
        );
    }

    // ── attn output proj / qkv ([SPEC-090]) ─────────────────────────────────

    #[test]
    fn attn_output_proj_projects_context() {
        // qkv_dim=2 hidden=2; context=[1,1]; o_proj=[[2,0],[0,3]] -> [2,3]
        let ctx = Mat::from_vec(1, 2, vec![1.0, 1.0]);
        let o = vec![2.0, 0.0, 0.0, 3.0];
        let out = attn_output_proj(&ctx, &o, 2, 2).unwrap();
        assert_eq!(out.data, vec![2.0, 3.0]);
    }

    #[test]
    fn qkv_with_rope_shapes_and_pos0_identity() {
        // hidden=2, single head head_dim=2 (qkv_dim=2), one token at pos 0.
        // q/k/v projections identity => q=k=v=normed; RoPE at pos0 = identity.
        let normed = Mat::from_vec(1, 2, vec![1.0, 2.0]);
        let ident = vec![1.0, 0.0, 0.0, 1.0];
        let lw = LayerWeights {
            input_ln: &[1.0, 1.0],
            post_attn_ln: &[1.0, 1.0],
            q_proj: &ident,
            k_proj: &ident,
            v_proj: &ident,
            o_proj: &ident,
            gate_w: &[],
            up_w: &[],
            down_w: &[],
        };
        let rope = RopeTable::build(&[0], 2, 10000.0);
        let (q, k, v) = qkv_with_rope(&normed, &lw, &rope, 2, 2).unwrap();
        assert_eq!(q.shape(), (1, 2));
        assert_eq!(v.data, vec![1.0, 2.0]); // v never rope'd
        assert_eq!(q.data, vec![1.0, 2.0]); // pos0 rope = identity
        assert_eq!(k.data, vec![1.0, 2.0]);
    }

    // ── Full layer driver ([SPEC-072]) ──────────────────────────────────────

    #[test]
    fn layer_forward_pre_norm_residual_identity_path() {
        // Build a layer where attention and MLP both return ZERO; then the
        // output must equal the input (pure residual passthrough), proving the
        // `x + sublayer(...)` wiring ([SPEC-072]).
        let hidden = 2usize;
        let qkv_dim = 2usize;
        let x = Mat::from_vec(2, hidden, vec![1.0, 2.0, 3.0, 4.0]);
        let ident = vec![1.0, 0.0, 0.0, 1.0];
        let lw = LayerWeights {
            input_ln: &[1.0, 1.0],
            post_attn_ln: &[1.0, 1.0],
            q_proj: &ident,
            k_proj: &ident,
            v_proj: &ident,
            o_proj: &ident,
            gate_w: &[],
            up_w: &[],
            down_w: &[],
        };
        let rope = RopeTable::build(&[0, 1], qkv_dim, 10000.0);
        let out = layer_forward(
            &x,
            &lw,
            &rope,
            hidden,
            qkv_dim,
            1e-6,
            // attention returns zeros
            |q, _k, _v| Ok(Mat::zeros(q.rows, q.cols)),
            // mlp returns zeros
            |n| Ok(Mat::zeros(n.rows, n.cols)),
        )
        .unwrap();
        assert_eq!(out.shape(), (2, hidden));
        for (got, want) in out.data.iter().zip(x.data.iter()) {
            assert!((got - want).abs() < 1e-6, "{got} != {want}");
        }
    }

    #[test]
    fn layer_forward_adds_both_sublayers() {
        // attn closure returns a constant context that o_proj maps to a known
        // vector; mlp adds another known vector. Output = x + attn_out + mlp_out.
        let hidden = 2usize;
        let qkv_dim = 2usize;
        let x = Mat::from_vec(1, hidden, vec![10.0, 20.0]);
        let ident = vec![1.0, 0.0, 0.0, 1.0];
        let lw = LayerWeights {
            input_ln: &[1.0, 1.0],
            post_attn_ln: &[1.0, 1.0],
            q_proj: &ident,
            k_proj: &ident,
            v_proj: &ident,
            o_proj: &ident, // o_proj identity => attn_out == context
            gate_w: &[],
            up_w: &[],
            down_w: &[],
        };
        let rope = RopeTable::build(&[0], qkv_dim, 10000.0);
        let out = layer_forward(
            &x,
            &lw,
            &rope,
            hidden,
            qkv_dim,
            1e-6,
            |_q, _k, _v| Ok(Mat::from_vec(1, qkv_dim, vec![1.0, 1.0])),
            |_n| Ok(Mat::from_vec(1, hidden, vec![100.0, 100.0])),
        )
        .unwrap();
        // x + [1,1] + [100,100] = [111, 121]
        assert!((out.data[0] - 111.0).abs() < 1e-4, "{}", out.data[0]);
        assert!((out.data[1] - 121.0).abs() < 1e-4, "{}", out.data[1]);
    }

    // ── Top-level stubs report the pending-wiring gap ───────────────────────

    #[test]
    fn top_level_entrypoints_are_pending_weights() {
        let w = Weights::default();
        let h = Mat::zeros(1, config::HIDDEN_SIZE);
        assert!(matches!(forward(&w, &h), Err(FocrError::NotImplemented(_))));
        assert!(matches!(lm_head(&w, &h), Err(FocrError::NotImplemented(_))));
    }
}
