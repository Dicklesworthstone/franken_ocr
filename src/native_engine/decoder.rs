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
//! ## Weights plumbing (wired)
//!
//! [`super::weights::Weights`] exposes named-tensor accessors (`mat`/`vec`,
//! widening BF16→f32 at the boundary), so the top-level [`forward`]/[`lm_head`]
//! entrypoints pull the per-layer norm/projection slices, the `model.norm`
//! weight, and the top-level `lm_head.weight` straight out of it. [`forward`] is
//! the **stateless, full-sequence prefill** path (full causal MHA via
//! [`prefill_attention`]); incremental decode threads a separate
//! `&mut [RingCache; 12]` (bd-1gv.17) and is out of scope here. ALL the
//! load-bearing math — token embed, RMSNorm, RoPE, the dense/MoE MLPs, the
//! lm_head GEMV, and the per-layer driver — stays unit-tested as pure functions
//! over explicit `&[f32]` / [`Mat`] slices; the entrypoints are the thin wiring
//! over them.

// Numerical kernel file: many hot loops index parallel stride-arrays by range
// (`a[r*stride..]`, `caches[layer]`/`weights[layer]`), where `clippy::needless_range_loop`
// is a false positive — the index is genuinely needed across several arrays.
#![allow(clippy::needless_range_loop)]

use super::moe;
use super::nn;
use super::rswa::{self, RingCache};
use super::tensor::{Mat, QInt8};
use super::weights::Weights;
use crate::error::{FocrError, FocrResult};
use crate::simd;
use rayon::prelude::*;

fn checked_shape_mul(context: &str, lhs: usize, rhs: usize, expression: &str) -> FocrResult<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "{context}: usize overflow computing {expression} ({lhs} * {rhs})"
        ))
    })
}

fn checked_mat_len(context: &str, x: &Mat) -> FocrResult<usize> {
    let expected = checked_shape_mul(context, x.rows, x.cols, "rows*cols")?;
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
    let expected_table_len = checked_shape_mul("embed_tokens", vocab, hidden, "vocab*hidden")?;
    if table.len() != expected_table_len {
        return Err(FocrError::Other(anyhow::anyhow!(
            "embed_tokens: table len {} != vocab*hidden {}",
            table.len(),
            expected_table_len
        )));
    }
    let seq = ids.len();
    let out_len = checked_shape_mul("embed_tokens", seq, hidden, "seq*hidden")?;
    let mut out = vec![0.0f32; out_len];
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
        let seq = position_ids.len();
        let table_len = seq.checked_mul(head_dim);
        assert!(
            table_len.is_some(),
            "RopeTable: seq*head_dim overflow ({seq} * {head_dim})"
        );
        let table_len = table_len.unwrap_or(0);
        if seq == 0 {
            return Self {
                cos: Vec::new(),
                sin: Vec::new(),
                head_dim,
            };
        }
        // inv_freq[i] = theta^(-2i/head_dim) = theta^(-(i/half)).  Computed in
        // f64 then narrowed — matches the HF float32 rotary embedding closely.
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                let exponent = (2 * i) as f64 / head_dim as f64;
                (1.0 / (theta as f64).powf(exponent)) as f32
            })
            .collect();
        let mut cos = vec![0.0f32; table_len];
        let mut sin = vec![0.0f32; table_len];
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
    checked_mat_len("apply_rope x", x)?;
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
    checked_mat_len("linear_no_bias x", x)?;
    if x.cols != in_ {
        return Err(FocrError::Other(anyhow::anyhow!(
            "linear_no_bias: x.cols {} != in {in_}",
            x.cols
        )));
    }
    let expected_weight_len = checked_shape_mul("linear_no_bias", out, in_, "out*in")?;
    if w.len() != expected_weight_len {
        return Err(FocrError::Other(anyhow::anyhow!(
            "linear_no_bias: w len {} != out*in {}",
            w.len(),
            expected_weight_len
        )));
    }
    // Transpose [out, in] -> [in, out] so matmul does [seq,in] x [in,out].
    let mut wt = vec![0.0f32; expected_weight_len];
    for i in 0..in_ {
        let dst = &mut wt[i * out..(i + 1) * out];
        for (o, slot) in dst.iter_mut().enumerate() {
            *slot = w[o * in_ + i];
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
    checked_mat_len("add_residual lhs", a)?;
    checked_mat_len("add_residual rhs", b)?;
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

// ── Prefill self-attention (full causal MHA over the whole sequence) ────────

/// Full causal multi-head self-attention over the prefill, *given* the already
/// RoPE'd `(q, k, v)` ([SPEC-090], baidu `SlidingWindowLlamaAttention` prefill
/// path `_attn_forward`).
///
/// In prefill the **entire** sequence is the R-SWA reference block (no token is
/// ever evicted), so a plain lower-triangular causal mask is exactly the
/// reference math — the ring buffer only ever restricts the *generated* tail
/// during incremental decode. `q`/`k`/`v` are token-major `[seq,
/// num_heads*head_dim]` (head `h` in columns `h*head_dim .. (h+1)*head_dim`).
///
/// Transposes to head-major `[num_heads, seq, head_dim]`, runs [`nn::sdpa`] with
/// `scale = 1/sqrt(head_dim)` and `causal = true`, then repacks the context back
/// to token-major `[seq, num_heads*head_dim]` for `o_proj`. MHA: `num_kv_heads ==
/// num_heads` (`repeat_kv` is a no-op), no QKV bias.
///
/// # Errors
/// [`FocrError::Other`] on a shape mismatch (`q/k/v` cols not `num_heads *
/// head_dim`, or `q/k/v` rows disagree).
pub fn prefill_attention(
    q: &Mat,
    k: &Mat,
    v: &Mat,
    num_heads: usize,
    head_dim: usize,
) -> FocrResult<Mat> {
    checked_mat_len("prefill_attention q", q)?;
    checked_mat_len("prefill_attention k", k)?;
    checked_mat_len("prefill_attention v", v)?;
    let dim = checked_shape_mul(
        "prefill_attention",
        num_heads,
        head_dim,
        "num_heads*head_dim",
    )?;
    let seq = q.rows;
    if q.cols != dim || k.cols != dim || v.cols != dim {
        return Err(FocrError::Other(anyhow::anyhow!(
            "prefill_attention: q/k/v cols ({},{},{}) != num_heads*head_dim {dim}",
            q.cols,
            k.cols,
            v.cols
        )));
    }
    if k.rows != seq || v.rows != seq {
        return Err(FocrError::Other(anyhow::anyhow!(
            "prefill_attention: q/k/v rows disagree ({}, {}, {})",
            seq,
            k.rows,
            v.rows
        )));
    }

    // Token-major [seq, num_heads*head_dim] -> head-major [num_heads, seq, head_dim].
    let span = seq * head_dim;
    let mut qh = vec![0.0f32; num_heads * span];
    let mut kh = vec![0.0f32; num_heads * span];
    let mut vh = vec![0.0f32; num_heads * span];
    for s in 0..seq {
        let (qr, kr, vr) = (q.row(s), k.row(s), v.row(s));
        for h in 0..num_heads {
            let src = h * head_dim;
            let dst = h * span + s * head_dim;
            qh[dst..dst + head_dim].copy_from_slice(&qr[src..src + head_dim]);
            kh[dst..dst + head_dim].copy_from_slice(&kr[src..src + head_dim]);
            vh[dst..dst + head_dim].copy_from_slice(&vr[src..src + head_dim]);
        }
    }

    // num_bh = batch(1) * num_heads; causal lower-triangular mask.
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let ctx = nn::sdpa(
        &qh, &kh, &vh, num_heads, seq, seq, head_dim, head_dim, scale, true,
    );
    let expected = num_heads * span;
    if ctx.len() != expected {
        return Err(FocrError::Other(anyhow::anyhow!(
            "prefill_attention: sdpa context len {} != expected {expected}",
            ctx.len()
        )));
    }

    // Head-major [num_heads, seq, head_dim] -> token-major [seq, num_heads*head_dim].
    let mut out = Mat::zeros(seq, dim);
    for h in 0..num_heads {
        for s in 0..seq {
            let src = h * span + s * head_dim;
            let dst = s * dim + h * head_dim;
            out.data[dst..dst + head_dim].copy_from_slice(&ctx[src..src + head_dim]);
        }
    }
    Ok(out)
}

// ── Top-level entrypoints (Weights-backed) ──────────────────────────────────

/// Final `model.norm` RMSNorm + `lm_head` projection over the decoder hidden
/// states ([SPEC-071]/[SPEC-081]).
///
/// Pulls `model.norm.weight` (length `hidden`) and the **top-level**
/// `lm_head.weight` (`[vocab, hidden] = [129280, 1280]`, NOT `model.lm_head`;
/// `tie_word_embeddings = false`) and composes the tested
/// [`norm_and_lm_head`]. Returns the f32 logits `[seq, vocab]`; the decode driver
/// feeds only the last hidden row, yielding `[1, vocab]`.
///
/// # Errors
/// [`FocrError::FormatMismatch`] if `model.norm.weight` / `lm_head.weight` are
/// absent, or [`FocrError::Other`] on a shape mismatch.
pub fn lm_head(weights: &Weights, hidden: &Mat) -> FocrResult<Mat> {
    let norm_w = weights.vec("model.norm.weight")?;
    let head = weights.mat("lm_head.weight")?;
    norm_and_lm_head(
        hidden,
        &norm_w,
        &head.data,
        config::VOCAB_SIZE,
        config::RMS_NORM_EPS,
    )
}

/// Run the full 12-layer DeepSeek-V2 MoE decoder over `inputs_embeds`,
/// returning the final `model.norm`-ready hidden states `[seq, hidden]`
/// ([SPEC-072], baidu `DeepseekV2Model.forward`).
///
/// Stateless, full-sequence (no KV cache): correct for **prefill / parity**. In
/// prefill every token is an R-SWA reference token, so attention is plain full
/// causal MHA (the ring cache only constrains incremental decode, which threads
/// a separate `&mut [RingCache; 12]` — out of scope here). Per layer 0..11:
///
/// 1. RMSNorm(`input_layernorm.weight`) -> Q/K/V proj (separate
///    `self_attn.{q,k,v}_proj.weight`) -> RoPE (theta 10000, NEOX `rotate_half`)
///    -> full causal SDPA ([`prefill_attention`]) -> `o_proj` -> residual.
/// 2. RMSNorm(`post_attention_layernorm.weight`) -> MLP: layer 0 dense
///    ([`moe::dense_forward`]), layers 1..11 MoE ([`moe::forward`]) -> residual.
///
/// Reuses the unit-tested [`layer_forward`] / [`qkv_with_rope`] / [`apply_rope`]
/// driver; the per-layer weight slices are pulled fresh from [`Weights`].
///
/// # Errors
/// [`FocrError::FormatMismatch`] on an absent tensor; [`FocrError::Other`] on a
/// shape mismatch or kernel error (propagated from the sub-steps).
pub fn forward(weights: &Weights, inputs_embeds: &Mat) -> FocrResult<Mat> {
    checked_mat_len("decoder::forward inputs_embeds", inputs_embeds)?;
    let hidden = config::HIDDEN_SIZE;
    let qkv_dim = checked_shape_mul(
        "decoder::forward",
        config::NUM_ATTENTION_HEADS,
        config::HEAD_DIM,
        "num_heads*head_dim",
    )?;
    let eps = config::RMS_NORM_EPS;
    if inputs_embeds.cols != hidden {
        return Err(FocrError::Other(anyhow::anyhow!(
            "decoder::forward: inputs_embeds cols {} != hidden {hidden}",
            inputs_embeds.cols
        )));
    }
    let seq = inputs_embeds.rows;

    // Absolute position_ids 0..seq ([SPEC-095]: true logical positions); one
    // RoPE table shared across all 12 layers.
    let positions: Vec<usize> = (0..seq).collect();
    let rope = RopeTable::build(&positions, config::HEAD_DIM, config::ROPE_THETA);

    let mut x = inputs_embeds.clone();
    for layer in 0..config::NUM_HIDDEN_LAYERS {
        let prefix = format!("model.layers.{layer}");
        // Owned per-layer weight Mats/Vecs; LayerWeights borrows their slices.
        let input_ln = weights.vec(&format!("{prefix}.input_layernorm.weight"))?;
        let post_attn_ln = weights.vec(&format!("{prefix}.post_attention_layernorm.weight"))?;
        let q_proj = weights.mat(&format!("{prefix}.self_attn.q_proj.weight"))?;
        let k_proj = weights.mat(&format!("{prefix}.self_attn.k_proj.weight"))?;
        let v_proj = weights.mat(&format!("{prefix}.self_attn.v_proj.weight"))?;
        let o_proj = weights.mat(&format!("{prefix}.self_attn.o_proj.weight"))?;

        let lw = LayerWeights {
            input_ln: &input_ln,
            post_attn_ln: &post_attn_ln,
            q_proj: &q_proj.data,
            k_proj: &k_proj.data,
            v_proj: &v_proj.data,
            o_proj: &o_proj.data,
            // Dense gate/up/down are pulled inside the MLP closure (moe::*), so
            // the driver's vestigial dense slots stay empty here.
            gate_w: &[],
            up_w: &[],
            down_w: &[],
        };

        x = layer_forward(
            &x,
            &lw,
            &rope,
            hidden,
            qkv_dim,
            eps,
            |q, k, v| prefill_attention(q, k, v, config::NUM_ATTENTION_HEADS, config::HEAD_DIM),
            |normed| {
                if layer < config::FIRST_K_DENSE_REPLACE {
                    moe::dense_forward(weights, normed)
                } else {
                    moe::forward(weights, normed, layer)
                }
            },
        )?;
    }
    Ok(x)
}

/// Repack token-major `[seq, num_heads*head_dim]` K/V (the layout
/// [`qkv_with_rope`] returns) into the head-major `[num_heads, seq, head_dim]`
/// flat layout that [`RingCache::record_prefill`] consumes (head `h`'s `seq`
/// rows are contiguous). Pure index gymnastics — no math.
///
/// # Errors
/// [`FocrError::Other`] on a `seq*head_dim`/`num_heads*…` overflow.
fn token_major_to_head_major(
    k: &Mat,
    v: &Mat,
    seq: usize,
    num_heads: usize,
    head_dim: usize,
) -> FocrResult<(Vec<f32>, Vec<f32>)> {
    let span = checked_shape_mul("token_major_to_head_major", seq, head_dim, "seq*head_dim")?;
    let total = checked_shape_mul(
        "token_major_to_head_major",
        num_heads,
        span,
        "num_heads*seq*head_dim",
    )?;
    let mut kh = vec![0.0f32; total];
    let mut vh = vec![0.0f32; total];
    for s in 0..seq {
        let (kr, vr) = (k.row(s), v.row(s));
        for h in 0..num_heads {
            let src = h * head_dim;
            let dst = h * span + s * head_dim;
            kh[dst..dst + head_dim].copy_from_slice(&kr[src..src + head_dim]);
            vh[dst..dst + head_dim].copy_from_slice(&vr[src..src + head_dim]);
        }
    }
    Ok((kh, vh))
}

// ── Dequant-once decoder weight cache (decode-throughput lever) ──────────────
//
// The `&Weights` decode path re-dequantized ~10 GB of bf16 expert weights from
// the payload EVERY token: `moe::forward` loads ALL 64 routed experts per MoE
// layer (x11 layers) even though only 6 of 64 are used per token, plus the
// attention projections and the [vocab, hidden] lm_head — the dominant decode
// cost (memory-bandwidth bound). This cache dequantizes every decoder tensor
// ONCE into an owned f32 image (~10.5 GB) that the (unchanged) kernels borrow by
// reference, so each subsequent step is pure GEMM. No math changes — only the
// weight source — so the cached output is identical to the `&Weights` path.

/// Owned, pre-dequantized decoder weights — built ONCE, borrowed by every prefill
/// and decode step. See module note above; this is the decode-throughput lever.
pub struct DecoderWeightCache {
    layers: Vec<CachedLayer>,
    /// Final `model.norm.weight` (RMSNorm before the head).
    final_norm: Vec<f32>,
    /// `lm_head.weight`, row-major `[vocab, hidden]`.
    lm_head: Vec<f32>,
}

/// One decoder layer's dequantized weights.
struct CachedLayer {
    input_ln: Vec<f32>,
    post_attn_ln: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
    mlp: CachedMlp,
}

/// A layer's dequantized MLP weights — dense (layer 0) or MoE (layers 1..11).
enum CachedMlp {
    Dense {
        gate: Vec<f32>,
        up: Vec<f32>,
        down: Vec<f32>,
    },
    Moe {
        /// Router gate `[N_ROUTED_EXPERTS, HIDDEN]`.
        gate: Vec<f32>,
        /// 64 routed experts, each `[gate_proj, up_proj, down_proj]`.
        experts: Vec<[Vec<f32>; 3]>,
        /// Fused shared expert `[gate_proj, up_proj, down_proj]`.
        shared: [Vec<f32>; 3],
    },
}

impl DecoderWeightCache {
    /// Dequantize every decoder tensor ONCE from [`Weights`]. Allocates ~10.5 GB
    /// of f32 for the 12-layer DeepSeek-V2 MoE decoder (64 experts/layer); the
    /// prefill + decode loop then run entirely off these owned buffers.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if any expected tensor is absent/mis-shaped.
    pub fn build(weights: &Weights) -> FocrResult<Self> {
        let mut layers = Vec::with_capacity(config::NUM_HIDDEN_LAYERS);
        for layer in 0..config::NUM_HIDDEN_LAYERS {
            let prefix = format!("model.layers.{layer}");
            let input_ln = weights.vec(&format!("{prefix}.input_layernorm.weight"))?;
            let post_attn_ln = weights.vec(&format!("{prefix}.post_attention_layernorm.weight"))?;
            let q_proj = weights.mat(&format!("{prefix}.self_attn.q_proj.weight"))?.data;
            let k_proj = weights.mat(&format!("{prefix}.self_attn.k_proj.weight"))?.data;
            let v_proj = weights.mat(&format!("{prefix}.self_attn.v_proj.weight"))?.data;
            let o_proj = weights.mat(&format!("{prefix}.self_attn.o_proj.weight"))?.data;
            let mlp = if layer < config::FIRST_K_DENSE_REPLACE {
                let p = format!("{prefix}.mlp");
                CachedMlp::Dense {
                    gate: weights.mat(&format!("{p}.gate_proj.weight"))?.data,
                    up: weights.mat(&format!("{p}.up_proj.weight"))?.data,
                    down: weights.mat(&format!("{p}.down_proj.weight"))?.data,
                }
            } else {
                let p = format!("{prefix}.mlp");
                let gate = weights.mat(&format!("{p}.gate.weight"))?.data;
                let mut experts = Vec::with_capacity(moe::config::N_ROUTED_EXPERTS);
                for e in 0..moe::config::N_ROUTED_EXPERTS {
                    experts.push([
                        weights.mat(&format!("{p}.experts.{e}.gate_proj.weight"))?.data,
                        weights.mat(&format!("{p}.experts.{e}.up_proj.weight"))?.data,
                        weights.mat(&format!("{p}.experts.{e}.down_proj.weight"))?.data,
                    ]);
                }
                let shared = [
                    weights.mat(&format!("{p}.shared_experts.gate_proj.weight"))?.data,
                    weights.mat(&format!("{p}.shared_experts.up_proj.weight"))?.data,
                    weights.mat(&format!("{p}.shared_experts.down_proj.weight"))?.data,
                ];
                CachedMlp::Moe {
                    gate,
                    experts,
                    shared,
                }
            };
            layers.push(CachedLayer {
                input_ln,
                post_attn_ln,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                mlp,
            });
        }
        let final_norm = weights.vec("model.norm.weight")?;
        let lm_head = weights.mat("lm_head.weight")?.data;
        Ok(Self {
            layers,
            final_norm,
            lm_head,
        })
    }
}

/// Borrow a cached layer's attention weights as a [`LayerWeights`] (MLP slots
/// stay empty — the MLP is run via [`cached_mlp`]).
fn cached_layer_weights(cl: &CachedLayer) -> LayerWeights<'_> {
    LayerWeights {
        input_ln: &cl.input_ln,
        post_attn_ln: &cl.post_attn_ln,
        q_proj: &cl.q_proj,
        k_proj: &cl.k_proj,
        v_proj: &cl.v_proj,
        o_proj: &cl.o_proj,
        gate_w: &[],
        up_w: &[],
        down_w: &[],
    }
}

/// Run one cached layer's MLP — dense (layer 0) or MoE — over the
/// `post_attention_layernorm`'d hidden, borrowing the dequantized weights (the
/// MoE router still selects top-k; only the GEMM runs, no dequant).
fn cached_mlp(mlp: &CachedMlp, normed: &Mat) -> FocrResult<Mat> {
    match mlp {
        CachedMlp::Dense { gate, up, down } => {
            let w = moe::MlpWeights {
                gate_proj: gate,
                up_proj: up,
                down_proj: down,
                hidden: moe::config::HIDDEN_SIZE,
                intermediate: moe::config::DENSE_INTERMEDIATE_SIZE,
            };
            moe::dense_mlp(normed, &w)
        }
        CachedMlp::Moe {
            gate,
            experts,
            shared,
        } => {
            let exp: Vec<moe::MlpWeights<'_>> = experts
                .iter()
                .map(|e| moe::MlpWeights {
                    gate_proj: &e[0],
                    up_proj: &e[1],
                    down_proj: &e[2],
                    hidden: moe::config::HIDDEN_SIZE,
                    intermediate: moe::config::MOE_INTERMEDIATE_SIZE,
                })
                .collect();
            let sh = moe::MlpWeights {
                gate_proj: &shared[0],
                up_proj: &shared[1],
                down_proj: &shared[2],
                hidden: moe::config::HIDDEN_SIZE,
                intermediate: moe::config::SHARED_INTERMEDIATE_SIZE,
            };
            moe::moe_block_default(normed, gate, &exp, &sh)
        }
    }
}

// ── Decode phase profiler (FOCR_PROFILE_DECODE) ──────────────────────────────
//
// Accumulates wall-nanoseconds per decode phase across all tokens so the
// throughput bottleneck is attributable (lm_head vs attention vs MoE experts vs
// routing). A handful of `Instant::now()` (~20 ns) + `fetch_add` per token —
// negligible vs the ~23 ms/token it measures, and entirely skipped when off.
pub mod prof {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Final RMSNorm + `lm_head` GEMV.
    pub static LMHEAD_NS: AtomicU64 = AtomicU64::new(0);
    /// Attention sub-block: q/k/v/o projections + RoPE + ring + R-SWA attention.
    pub static ATTN_NS: AtomicU64 = AtomicU64::new(0);
    /// MoE/MLP routed + shared experts (the SwiGLU GEMVs), excluding routing.
    pub static EXPERTS_NS: AtomicU64 = AtomicU64::new(0);
    /// MoE top-k router (softmax over the 64-expert gate).
    pub static ROUTE_NS: AtomicU64 = AtomicU64::new(0);

    pub fn enabled() -> bool {
        static FLAG: OnceLock<bool> = OnceLock::new();
        *FLAG.get_or_init(|| std::env::var_os("FOCR_PROFILE_DECODE").is_some())
    }
    #[inline]
    pub fn add(c: &AtomicU64, ns: u64) {
        c.fetch_add(ns, Ordering::Relaxed);
    }
    /// Reset all counters (call before a timed decode loop).
    pub fn reset() {
        for c in [&LMHEAD_NS, &ATTN_NS, &EXPERTS_NS, &ROUTE_NS] {
            c.store(0, Ordering::Relaxed);
        }
    }
    /// `(lmhead, attn, experts, route)` accumulated milliseconds.
    pub fn snapshot_ms() -> (f64, f64, f64, f64) {
        let ms = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64 / 1e6;
        (ms(&LMHEAD_NS), ms(&ATTN_NS), ms(&EXPERTS_NS), ms(&ROUTE_NS))
    }
}

// ── Bespoke single-token (m=1) decode GEMV — the decode-throughput kernel ────
//
// Decode is one token at a time: every projection is a matrix·vector product
// `y[N] = W[N,K] · x[K]`, with W stored `[out, in]` row-major (output-channel
// major) — so each output row is a CONTIGUOUS `K`-length dot product. The
// generic GEMM path transposed W every call and fell below ft-kernel-cpu's
// threading threshold (m=1), running single-threaded at ~0.4 GFLOP/s. This
// kernel instead: (1) no transpose — dots the native row layout; (2) fans the N
// independent rows across cores with rayon; (3) an 8-wide unrolled inner dot
// that LLVM lowers to NEON/AVX FMA. This is intentionally model-specialized, not
// a general BLAS — exactly the decode hot path and nothing else.

/// 8-wide unrolled f32 dot product (`sum a[i]*b[i]`). The fixed 8-lane reduction
/// order is deterministic and auto-vectorizes to NEON/AVX FMA under `-O3`.
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut acc = [0.0f32; 8];
    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        // Indexed (not iter) so the 8 independent FMAs vectorize cleanly.
        for (l, slot) in acc.iter_mut().enumerate() {
            *slot += a[base + l] * b[base + l];
        }
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for i in (chunks * 8)..n {
        s += a[i] * b[i];
    }
    s
}

/// Single-token GEMV: `y[o] = dot(x[0..k], w[o*k .. o*k+k])` for `o in 0..n`,
/// over `w` in `[n, k]` row-major layout. The `n` output rows are independent →
/// fanned across the rayon pool. Returns the `[n]` result.
fn gemv(x: &[f32], w: &[f32], n: usize, k: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(w.len(), n * k);
    let mut y = vec![0.0f32; n];
    // Chunked parallelism keeps per-task work well above rayon's dispatch cost
    // even for the small projections; the huge lm_head row-splits across cores.
    y.par_chunks_mut(64)
        .enumerate()
        .for_each(|(blk, ys)| {
            let base = blk * 64;
            for (j, slot) in ys.iter_mut().enumerate() {
                let o = base + j;
                *slot = dot_f32(x, &w[o * k..o * k + k]);
            }
        });
    y
}

/// SiLU / swish: `x * sigmoid(x) = x / (1 + e^-x)` ([SPEC-075], the SwiGLU gate).
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// One SwiGLU expert over a single decode row `x[hidden]`, off cached weights:
/// `down( silu(gate·x) * (up·x) )`. `inter` is the expert's intermediate width.
fn expert_gemv(x: &[f32], gate_w: &[f32], up_w: &[f32], down_w: &[f32], hidden: usize, inter: usize) -> Vec<f32> {
    let g = gemv(x, gate_w, inter, hidden);
    let u = gemv(x, up_w, inter, hidden);
    let mut act = vec![0.0f32; inter];
    for i in 0..inter {
        act[i] = silu(g[i]) * u[i];
    }
    gemv(&act, down_w, hidden, inter)
}

/// Decode MoE/MLP over a single `post_attention_layernorm`'d row, off the cached
/// weights — mirrors [`moe::moe_block_default`] (route top-k, weighted expert
/// sum, + shared expert) but specialized to `m == 1` with [`gemv`]. Bit-parity
/// with the GEMM path is gated by the cached-vs-stateless decode check.
fn decode_mlp(mlp: &CachedMlp, normed: &Mat) -> FocrResult<Vec<f32>> {
    let hidden = config::HIDDEN_SIZE;
    let row = normed.row(0);
    match mlp {
        CachedMlp::Dense { gate, up, down } => Ok(expert_gemv(
            row,
            gate,
            up,
            down,
            hidden,
            moe::config::DENSE_INTERMEDIATE_SIZE,
        )),
        CachedMlp::Moe {
            gate,
            experts,
            shared,
        } => {
            let routing = moe::route_default(normed, gate)?;
            let inter = moe::config::MOE_INTERMEDIATE_SIZE;
            let mut out = vec![0.0f32; hidden];
            for j in 0..moe::config::NUM_EXPERTS_PER_TOK {
                let e = routing.indices[0][j];
                let w = routing.weights[0][j];
                let y = expert_gemv(row, &experts[e][0], &experts[e][1], &experts[e][2], hidden, inter);
                for c in 0..hidden {
                    out[c] += w * y[c];
                }
            }
            // Shared expert (weight 1.0 over every token).
            let s = expert_gemv(
                row,
                &shared[0],
                &shared[1],
                &shared[2],
                hidden,
                moe::config::SHARED_INTERMEDIATE_SIZE,
            );
            for c in 0..hidden {
                out[c] += s[c];
            }
            Ok(out)
        }
    }
}

/// Final RMSNorm + `lm_head` over the decode hidden (`[1, hidden]`), off the
/// cached head weights via the bespoke [`gemv`] — the per-token `[vocab, hidden]`
/// projection was a major decode cost run single-threaded by the GEMM path.
///
/// # Errors
/// [`FocrError::Other`] on a shape mismatch.
pub fn lm_head_cached(wc: &DecoderWeightCache, hidden: &Mat) -> FocrResult<Mat> {
    if hidden.rows != 1 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "decoder::lm_head_cached: expected a single decode row, got {} rows",
            hidden.rows
        )));
    }
    let normed = nn::rms_norm(hidden, Some(&wc.final_norm), config::RMS_NORM_EPS)?;
    let logits = gemv(normed.row(0), &wc.lm_head, config::VOCAB_SIZE, config::HIDDEN_SIZE);
    Ok(Mat::from_vec(1, config::VOCAB_SIZE, logits))
}

/// Run the full 12-layer prefill over `inputs_embeds` exactly like [`forward`],
/// but ALSO capture each layer's RoPE'd K/V into a per-layer [`RingCache`] — the
/// R-SWA reference block ([SPEC-091], never evicted). Returns the final
/// `model.norm`-ready hidden `[seq, hidden]` (bit-identical to [`forward`]'s
/// output, same kernels) plus the 12 populated caches for
/// [`decode_step_with_cache`].
///
/// This is the prefill half of the O(n) cached decode: the stateless loop
/// re-runs [`forward`] over the whole growing sequence every step (O(n^2) — the
/// MLP/MoE re-processes all prior tokens); here we pay the prefill ONCE and then
/// extend by a single token per step.
///
/// # Errors
/// As [`forward`], plus a [`RingCache::record_prefill`] error if `seq` exceeds
/// the cache capacity (it is sized to `seq`, so only an internal inconsistency
/// trips it).
pub fn prefill_with_cache(
    wc: &DecoderWeightCache,
    inputs_embeds: &Mat,
) -> FocrResult<(Mat, Vec<RingCache>)> {
    checked_mat_len("decoder::prefill_with_cache inputs_embeds", inputs_embeds)?;
    let hidden = config::HIDDEN_SIZE;
    let num_heads = config::NUM_ATTENTION_HEADS;
    let head_dim = config::HEAD_DIM;
    let qkv_dim =
        checked_shape_mul("decoder::prefill_with_cache", num_heads, head_dim, "num_heads*head_dim")?;
    let eps = config::RMS_NORM_EPS;
    if inputs_embeds.cols != hidden {
        return Err(FocrError::Other(anyhow::anyhow!(
            "decoder::prefill_with_cache: inputs_embeds cols {} != hidden {hidden}",
            inputs_embeds.cols
        )));
    }
    let seq = inputs_embeds.rows;

    // Absolute positions 0..seq, one shared RoPE table ([SPEC-095]).
    let positions: Vec<usize> = (0..seq).collect();
    let rope = RopeTable::build(&positions, head_dim, config::ROPE_THETA);

    let mut caches: Vec<RingCache> = (0..config::NUM_HIDDEN_LAYERS)
        .map(|_| RingCache::new(seq.max(1)))
        .collect();

    let mut x = inputs_embeds.clone();
    for layer in 0..config::NUM_HIDDEN_LAYERS {
        let cl = &wc.layers[layer];
        let lw = cached_layer_weights(cl);

        // Attention sub-block — mirrors `layer_forward`, but intercepts (k, v) to
        // seed the ring cache's reference block before the prefill SDPA.
        let normed = nn::rms_norm(&x, Some(lw.input_ln), eps)?;
        let (q, k, v) = qkv_with_rope(&normed, &lw, &rope, hidden, qkv_dim)?;
        let (kh, vh) = token_major_to_head_major(&k, &v, seq, num_heads, head_dim)?;
        caches[layer].record_prefill(&kh, &vh, seq)?;
        let context = prefill_attention(&q, &k, &v, num_heads, head_dim)?;
        let attn_out = attn_output_proj(&context, lw.o_proj, hidden, qkv_dim)?;
        let h = add_residual(&x, &attn_out)?;

        // MLP / MoE sub-block (dense layer 0, MoE 1..11) off the cached weights.
        let normed2 = nn::rms_norm(&h, Some(lw.post_attn_ln), eps)?;
        let mlp_out = cached_mlp(&cl.mlp, &normed2)?;
        x = add_residual(&h, &mlp_out)?;
    }
    Ok((x, caches))
}

/// One incremental decode step over the per-layer [`RingCache`]s
/// ([SPEC-091..095]).
///
/// `token_embed` is the new token's embedding row `[1, hidden]` (from
/// [`embed_tokens`]); `position` is its TRUE absolute position (`prefill_len +
/// generated_so_far`), used for RoPE ([SPEC-095] — always the logical position,
/// never a ring slot). Each layer writes this token's RoPE'd K/V into its ring,
/// then runs R-SWA [`rswa::decode_attention`] over (reference ++ ring). Returns
/// the final `model.norm`-ready hidden `[1, hidden]` for `lm_head`.
///
/// For the first [`rswa::RING_WINDOW`] (128) decode steps this is bit-identical
/// to the stateless [`forward`] over the equivalent full sequence (the ring holds
/// every generated token; no eviction, so the union is the full causal context);
/// beyond that the generated tail is windowed to `W` — the reference model's
/// actual sliding-window behavior, which the stateless full-causal path does NOT
/// reproduce.
///
/// # Errors
/// As [`forward`], plus a [`RingCache::write_decode_step`] error if the caches
/// were not populated by [`prefill_with_cache`] first, or a shape mismatch.
pub fn decode_step_with_cache(
    wc: &DecoderWeightCache,
    caches: &mut [RingCache],
    token_embed: &Mat,
    position: usize,
) -> FocrResult<Mat> {
    let hidden = config::HIDDEN_SIZE;
    let qkv_dim = checked_shape_mul(
        "decoder::decode_step_with_cache",
        config::NUM_ATTENTION_HEADS,
        config::HEAD_DIM,
        "num_heads*head_dim",
    )?;
    let eps = config::RMS_NORM_EPS;
    if token_embed.rows != 1 || token_embed.cols != hidden {
        return Err(FocrError::Other(anyhow::anyhow!(
            "decoder::decode_step_with_cache: token_embed shape [{}, {}] != [1, {hidden}]",
            token_embed.rows,
            token_embed.cols
        )));
    }
    if caches.len() != config::NUM_HIDDEN_LAYERS {
        return Err(FocrError::Other(anyhow::anyhow!(
            "decoder::decode_step_with_cache: {} caches != {} layers",
            caches.len(),
            config::NUM_HIDDEN_LAYERS
        )));
    }

    // RoPE at the single TRUE absolute position (one row, one shared table).
    let rope = RopeTable::build(&[position], config::HEAD_DIM, config::ROPE_THETA);

    let mut x = token_embed.clone();
    for layer in 0..config::NUM_HIDDEN_LAYERS {
        let cl = &wc.layers[layer];

        // Attention via the bespoke m=1 GEMV: project q/k/v, RoPE q/k at the true
        // `position`, push K/V into the ring (the query attends to itself as the
        // newest ring token — OQ-3), R-SWA decode attention over reference ++
        // ring, then o_proj. seq == 1, so the token-major `[1, qkv_dim]` rows ARE
        // the head-major `[num_heads, head_dim]` flats the ring kernels expect.
        let normed = nn::rms_norm(&x, Some(&cl.input_ln), eps)?;
        let nrow = normed.row(0);
        let mut q = Mat::from_vec(1, qkv_dim, gemv(nrow, &cl.q_proj, qkv_dim, hidden));
        let mut k = Mat::from_vec(1, qkv_dim, gemv(nrow, &cl.k_proj, qkv_dim, hidden));
        let v = gemv(nrow, &cl.v_proj, qkv_dim, hidden);
        apply_rope(&mut q, &rope)?;
        apply_rope(&mut k, &rope)?;
        caches[layer].write_decode_step(&k.data, &v)?;
        let context = rswa::decode_attention(&caches[layer], &q.data)?;
        let attn_out = Mat::from_vec(1, hidden, gemv(&context.data, &cl.o_proj, hidden, qkv_dim));
        let h = add_residual(&x, &attn_out)?;

        // MLP / MoE via the bespoke GEMV (routed top-k experts + shared).
        let normed2 = nn::rms_norm(&h, Some(&cl.post_attn_ln), eps)?;
        let mlp_out = Mat::from_vec(1, hidden, decode_mlp(&cl.mlp, &normed2)?);
        x = add_residual(&h, &mlp_out)?;
    }
    Ok(x)
}

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  INT8 DECODE ENGINE — per-output-channel symmetric S8S8, NEON SDOT / VNNI  ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
// The f32 cache above dequantizes the whole 12-layer MoE decoder to ~10.5 GB of
// f32 and reads ~1.9 GB/token at decode — memory-bound on the ~120 GB/s M4 bus.
// This int8 engine stores the SAME weights as per-output-channel symmetric int8
// (~2.8 GB, 4x less traffic) and runs the hot GEMV/GEMM through the bespoke
// `simd::igemm_s8s8` backend (NEON `vdotq_s32` SDOT on Apple Silicon; AVX-512 /
// AVX-VNNI / AVX2 on x86 — the exact "fastest per-ISA" doctrine for THIS model).
//
// Precision: weights quantize as `w_scale[o] = max|W[o,:]|/127` (symmetric, the
// proven `nn::quantize_int8` / `ft_kernel_cpu::quantize_per_output_channel_i8`
// convention); activations quantize dynamically per row (per-tensor for the m=1
// decode GEMV), `a_scale = max|x|/127`, then dequant `acc · a_scale · w_scale[o]`
// in exact i32 — identical math to `nn::linear_int8_dynamic`. The router gate and
// every RMSNorm weight stay f32 (tiny, and top-k selection is precision-sensitive).
// Accuracy is VERIFIED end-to-end against the f32-stateless oracle + baidu CER.

/// Block of int8 GEMV output rows fanned to one rayon task (each a single
/// `simd::igemm_s8s8` SDOT call). Matches the f32 `gemv`'s 64-row blocking.
const I8_GEMV_BLOCK: usize = 64;

/// Dynamically quantize a single activation row `x[k]` to symmetric int8
/// (`a_scale = max|x|/127`, zero-point 0), returning `(xq, a_scale)`. The m=1
/// decode case of `ft_kernel_cpu`'s per-row `DynamicQuantizeLinear`.
#[inline]
fn quantize_row_i8(x: &[f32]) -> (Vec<i8>, f32) {
    let amax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let a_scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let inv = 1.0 / a_scale;
    let xq: Vec<i8> = x
        .iter()
        .map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (xq, a_scale)
}

/// Bespoke single-token (m=1) int8 GEMV — the decode-throughput kernel.
///
/// `y[o] = (Σ_i xq[i]·w[o,i]) · a_scale · w_scale[o]` for `o in 0..n`, with the
/// activation `x[k]` dynamically int8-quantized once, and the pre-quantized
/// weight `qw` in `[n,k]` output-channel-major int8 (the native `[out,in]`
/// checkpoint layout — no transpose). The `n` output rows fan across the rayon
/// pool in `I8_GEMV_BLOCK` chunks, each a single `simd::igemm_s8s8` SDOT/VNNI
/// call; i32 accumulation is exact (proven `K ≤ 6848 < i32::MAX`, `src/simd`).
fn gemv_i8(x: &[f32], qw: &QInt8) -> Vec<f32> {
    let k = qw.k;
    let n = qw.n;
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(qw.w.len(), n * k);
    debug_assert_eq!(qw.scales.len(), n);
    let (xq, a_scale) = quantize_row_i8(x);
    let mut y = vec![0.0f32; n];
    y.par_chunks_mut(I8_GEMV_BLOCK)
        .enumerate()
        .for_each(|(blk, ys)| {
            let base = blk * I8_GEMV_BLOCK;
            let cnt = ys.len();
            let mut acc = vec![0i32; cnt];
            simd::igemm_s8s8(&xq, &qw.w[base * k..(base + cnt) * k], 1, k, cnt, &mut acc);
            for (j, slot) in ys.iter_mut().enumerate() {
                *slot = acc[j] as f32 * a_scale * qw.scales[base + j];
            }
        });
    y
}

/// One SwiGLU expert over a single decode row `x[hidden]`, int8: `down(silu(gate·x) *
/// (up·x))`. The int8 twin of [`expert_gemv`]. Internally parallel (used for the
/// big dense layer-0 MLP + the shared expert).
fn expert_gemv_i8(x: &[f32], gate: &QInt8, up: &QInt8, down: &QInt8) -> Vec<f32> {
    let g = gemv_i8(x, gate);
    let u = gemv_i8(x, up);
    let inter = gate.n;
    let mut act = vec![0.0f32; inter];
    for i in 0..inter {
        act[i] = silu(g[i]) * u[i];
    }
    gemv_i8(&act, down)
}

/// SERIAL int8 GEMV from a pre-quantized activation — a single `simd::igemm_s8s8`
/// SDOT/VNNI call over all `n` rows, NO rayon. Used inside the per-expert rayon
/// tasks below, where parallelism is ACROSS experts (one task each) rather than
/// within: 6 routed experts is the right granularity to saturate the pool with
/// ONE dispatch, vs the 18 tiny internally-parallel GEMVs the per-row `gemv_i8`
/// would spawn (measured: MoE experts were 56% of decode, ~10% of SDOT peak —
/// dispatch-bound, not compute-bound).
fn gemv_i8_serial(xq: &[i8], a_scale: f32, qw: &QInt8) -> Vec<f32> {
    let k = qw.k;
    let n = qw.n;
    debug_assert_eq!(xq.len(), k);
    let mut acc = vec![0i32; n];
    simd::igemm_s8s8(xq, &qw.w, 1, k, n, &mut acc);
    let mut y = vec![0.0f32; n];
    for (o, slot) in y.iter_mut().enumerate() {
        *slot = acc[o] as f32 * a_scale * qw.scales[o];
    }
    y
}

/// One SwiGLU expert, fully SERIAL (the input is quantized ONCE and reused for
/// gate+up). The serial twin of [`expert_gemv_i8`] for cross-expert parallelism.
fn expert_gemv_i8_serial(x: &[f32], gate: &QInt8, up: &QInt8, down: &QInt8) -> Vec<f32> {
    let (xq, a_scale) = quantize_row_i8(x);
    let g = gemv_i8_serial(&xq, a_scale, gate);
    let u = gemv_i8_serial(&xq, a_scale, up);
    let inter = gate.n;
    let mut act = vec![0.0f32; inter];
    for i in 0..inter {
        act[i] = silu(g[i]) * u[i];
    }
    let (aq, a_scale2) = quantize_row_i8(&act);
    gemv_i8_serial(&aq, a_scale2, down)
}

/// Quantize a `[out, in]` row-major f32 weight to per-output-channel symmetric
/// int8 (`in` inferred from `w.len()/out`). Thin wrapper over [`nn::quantize_int8`].
fn quant_oc(w: &[f32], out: usize) -> QInt8 {
    nn::quantize_int8(w, out, w.len() / out)
}

/// `FOCR_QKV_FUSED`: at cache-build, STACK q/k/v into ONE `[3*qkv_dim, hidden]`
/// int8 weight so each decode token runs a SINGLE block-parallel [`gemv_i8`]
/// (one activation quantize, one rayon dispatch wave) instead of three. The
/// default path (flag unset) is byte-for-byte unchanged.
const QKV_FUSED_ENV: &str = "FOCR_QKV_FUSED";

/// Read [`QKV_FUSED_ENV`] ONCE into a process-wide bool (build-time only; never
/// touched per-token).
fn qkv_fused_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os(QKV_FUSED_ENV).is_some())
}

/// Stack three same-shaped `[n, k]` per-output-channel int8 weights into ONE
/// `[3n, k]` weight: int8 rows concatenated (q then k then v), per-channel
/// `scales` concatenated in the same order. Feeding the result to one
/// block-parallel [`gemv_i8`] yields output rows `[0,n)`, `[n,2n)`, `[2n,3n)`
/// that are BYTE-IDENTICAL to three separate `gemv_i8(x, {q,k,v})` calls: the
/// activation is quantized once to the same `(xq, a_scale)`, and each output row
/// `o` is an independent i32 SDOT of the SAME `xq` against the SAME `w[o,:]`
/// dequantized by the SAME `scales[o]` — the rayon row-chunking only changes
/// which rows share a task, never a per-row reduction.
fn fuse_qkv(q: &QInt8, k: &QInt8, v: &QInt8) -> QInt8 {
    debug_assert_eq!(q.k, k.k, "fuse_qkv: k contraction dim mismatch");
    debug_assert_eq!(q.k, v.k, "fuse_qkv: k contraction dim mismatch");
    debug_assert_eq!(q.n, k.n, "fuse_qkv: q/k output dim mismatch");
    debug_assert_eq!(q.n, v.n, "fuse_qkv: q/v output dim mismatch");
    let (n, kk) = (q.n, q.k);
    let mut w = Vec::with_capacity(n * kk * 3);
    w.extend_from_slice(&q.w);
    w.extend_from_slice(&k.w);
    w.extend_from_slice(&v.w);
    let mut scales = Vec::with_capacity(n * 3);
    scales.extend_from_slice(&q.scales);
    scales.extend_from_slice(&k.scales);
    scales.extend_from_slice(&v.scales);
    QInt8::new(w, scales, 3 * n, kk)
}

/// A layer's int8 MLP weights — dense (layer 0) or MoE (layers 1..11). Mirrors
/// [`CachedMlp`] but every projection is a [`QInt8`]; the MoE router `gate` stays
/// f32 (top-k routing is precision-sensitive and the gate is tiny: `[64,1280]`).
enum CachedMlpI8 {
    Dense {
        gate: QInt8,
        up: QInt8,
        down: QInt8,
    },
    Moe {
        gate: Vec<f32>,
        experts: Vec<[QInt8; 3]>,
        shared: [QInt8; 3],
    },
}

/// One decoder layer's int8 weights (attention projections + MLP). Norm weights
/// stay f32.
struct CachedLayerI8 {
    input_ln: Vec<f32>,
    post_attn_ln: Vec<f32>,
    q_proj: QInt8,
    k_proj: QInt8,
    v_proj: QInt8,
    /// Optional fused `[3*qkv_dim, hidden]` stack of `q_proj`/`k_proj`/`v_proj`,
    /// built ONLY when [`QKV_FUSED_ENV`] is set (the separate three are always
    /// retained for prefill). When present, decode runs one block-parallel GEMV
    /// over all `3*qkv_dim` output rows instead of three; byte-identical output.
    qkv: Option<QInt8>,
    o_proj: QInt8,
    mlp: CachedMlpI8,
}

/// The int8 twin of [`DecoderWeightCache`]: every load-bearing GEMM weight stored
/// per-output-channel symmetric int8 (~2.8 GB vs ~10.5 GB f32). Built ONCE; prefill
/// (via [`nn::linear_int8_dynamic`]) and decode (via [`gemv_i8`]) run off it.
pub struct DecoderWeightCacheI8 {
    layers: Vec<CachedLayerI8>,
    final_norm: Vec<f32>,
    lm_head: QInt8,
}

impl DecoderWeightCacheI8 {
    /// Quantize every decoder tensor ONCE from [`Weights`] to per-output-channel
    /// symmetric int8 (the dense layer-0 MLP, the 64 routed + 1 shared expert per
    /// MoE layer, all four attention projections, and the `lm_head`). The router
    /// gate and the RMSNorm weights stay f32.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if any expected tensor is absent/mis-shaped.
    pub fn build(weights: &Weights) -> FocrResult<Self> {
        let hidden = config::HIDDEN_SIZE;
        let qkv_dim = config::NUM_ATTENTION_HEADS * config::HEAD_DIM;
        let mut layers = Vec::with_capacity(config::NUM_HIDDEN_LAYERS);
        for layer in 0..config::NUM_HIDDEN_LAYERS {
            let prefix = format!("model.layers.{layer}");
            let input_ln = weights.vec(&format!("{prefix}.input_layernorm.weight"))?;
            let post_attn_ln = weights.vec(&format!("{prefix}.post_attention_layernorm.weight"))?;
            let q_proj = quant_oc(&weights.mat(&format!("{prefix}.self_attn.q_proj.weight"))?.data, qkv_dim);
            let k_proj = quant_oc(&weights.mat(&format!("{prefix}.self_attn.k_proj.weight"))?.data, qkv_dim);
            let v_proj = quant_oc(&weights.mat(&format!("{prefix}.self_attn.v_proj.weight"))?.data, qkv_dim);
            let o_proj = quant_oc(&weights.mat(&format!("{prefix}.self_attn.o_proj.weight"))?.data, hidden);
            // Fused q/k/v stack (FOCR_QKV_FUSED): built ONCE here so decode runs a
            // single block-parallel GEMV. Byte-identical to the 3 separate calls.
            let qkv = qkv_fused_enabled().then(|| fuse_qkv(&q_proj, &k_proj, &v_proj));
            let mlp = if layer < config::FIRST_K_DENSE_REPLACE {
                let p = format!("{prefix}.mlp");
                let inter = moe::config::DENSE_INTERMEDIATE_SIZE;
                CachedMlpI8::Dense {
                    gate: quant_oc(&weights.mat(&format!("{p}.gate_proj.weight"))?.data, inter),
                    up: quant_oc(&weights.mat(&format!("{p}.up_proj.weight"))?.data, inter),
                    down: quant_oc(&weights.mat(&format!("{p}.down_proj.weight"))?.data, hidden),
                }
            } else {
                let p = format!("{prefix}.mlp");
                let gate = weights.mat(&format!("{p}.gate.weight"))?.data;
                let inter = moe::config::MOE_INTERMEDIATE_SIZE;
                let mut experts = Vec::with_capacity(moe::config::N_ROUTED_EXPERTS);
                for e in 0..moe::config::N_ROUTED_EXPERTS {
                    experts.push([
                        quant_oc(&weights.mat(&format!("{p}.experts.{e}.gate_proj.weight"))?.data, inter),
                        quant_oc(&weights.mat(&format!("{p}.experts.{e}.up_proj.weight"))?.data, inter),
                        quant_oc(&weights.mat(&format!("{p}.experts.{e}.down_proj.weight"))?.data, hidden),
                    ]);
                }
                let si = moe::config::SHARED_INTERMEDIATE_SIZE;
                let shared = [
                    quant_oc(&weights.mat(&format!("{p}.shared_experts.gate_proj.weight"))?.data, si),
                    quant_oc(&weights.mat(&format!("{p}.shared_experts.up_proj.weight"))?.data, si),
                    quant_oc(&weights.mat(&format!("{p}.shared_experts.down_proj.weight"))?.data, hidden),
                ];
                CachedMlpI8::Moe { gate, experts, shared }
            };
            layers.push(CachedLayerI8 {
                input_ln,
                post_attn_ln,
                q_proj,
                k_proj,
                v_proj,
                qkv,
                o_proj,
                mlp,
            });
        }
        let final_norm = weights.vec("model.norm.weight")?;
        let lm_head = quant_oc(&weights.mat("lm_head.weight")?.data, config::VOCAB_SIZE);
        Ok(Self {
            layers,
            final_norm,
            lm_head,
        })
    }
}

/// One SwiGLU expert over a `[n_tok, hidden]` activation, int8 — `down(silu(gate·x)
/// * (up·x))` via [`nn::linear_int8_dynamic`]. The int8 twin of [`moe::expert_mlp`].
fn expert_mlp_i8(x: &Mat, gate: &QInt8, up: &QInt8, down: &QInt8) -> FocrResult<Mat> {
    let mut g = nn::linear_int8_dynamic(x, gate, None)?;
    nn::silu(&mut g);
    let u = nn::linear_int8_dynamic(x, up, None)?;
    if g.data.len() != u.data.len() {
        return Err(FocrError::Other(anyhow::anyhow!(
            "decoder::expert_mlp_i8: gate/up shape mismatch ({} vs {})",
            g.data.len(),
            u.data.len()
        )));
    }
    for (a, &b) in g.data.iter_mut().zip(u.data.iter()) {
        *a *= b;
    }
    nn::linear_int8_dynamic(&g, down, None)
}

/// Int8 MoE block over `[n_tok, hidden]` — the int8 twin of [`moe::moe_block`]:
/// route top-6 (f32 router gate), gather each selected token into a compact
/// activation, run [`expert_mlp_i8`], scatter back weighted, then add the shared
/// expert at weight 1.0. Mathematically identical structure; only the GEMMs go int8.
#[allow(clippy::needless_range_loop)]
fn moe_block_i8(
    hidden: &Mat,
    gate: &[f32],
    experts: &[[QInt8; 3]],
    shared: &[QInt8; 3],
) -> FocrResult<Mat> {
    let n_tok = hidden.rows;
    let h = hidden.cols;
    let routing = moe::route_default(hidden, gate)?;
    let mut out = Mat::zeros(n_tok, h);
    let mut per_expert: Vec<Vec<(usize, f32)>> = vec![Vec::new(); moe::config::N_ROUTED_EXPERTS];
    for t in 0..n_tok {
        for j in 0..moe::config::NUM_EXPERTS_PER_TOK {
            per_expert[routing.indices[t][j]].push((t, routing.weights[t][j]));
        }
    }
    for (e, members) in per_expert.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        let m = members.len();
        let mut sub = Mat::zeros(m, h);
        for (r, &(t, _w)) in members.iter().enumerate() {
            sub.row_mut(r).copy_from_slice(hidden.row(t));
        }
        let y = expert_mlp_i8(&sub, &experts[e][0], &experts[e][1], &experts[e][2])?;
        for (r, &(t, w)) in members.iter().enumerate() {
            let yr = y.row(r);
            let outr = out.row_mut(t);
            for c in 0..h {
                outr[c] += w * yr[c];
            }
        }
    }
    let shared_out = expert_mlp_i8(hidden, &shared[0], &shared[1], &shared[2])?;
    for (o, &s) in out.data.iter_mut().zip(shared_out.data.iter()) {
        *o += s;
    }
    Ok(out)
}

/// Run one int8 layer's MLP/MoE over the `post_attention_layernorm`'d hidden
/// (prefill, m>1). Int8 twin of [`cached_mlp`].
fn prefill_mlp_i8(mlp: &CachedMlpI8, normed: &Mat) -> FocrResult<Mat> {
    match mlp {
        CachedMlpI8::Dense { gate, up, down } => expert_mlp_i8(normed, gate, up, down),
        CachedMlpI8::Moe { gate, experts, shared } => moe_block_i8(normed, gate, experts, shared),
    }
}

/// Int8 decode MLP/MoE over a single `post_attention_layernorm`'d row. Int8 twin
/// of [`decode_mlp`] (route top-k via the f32 gate, weighted [`expert_gemv_i8`]
/// sum, + shared expert).
fn decode_mlp_i8(mlp: &CachedMlpI8, normed: &Mat) -> FocrResult<Vec<f32>> {
    let hidden = config::HIDDEN_SIZE;
    let row = normed.row(0);
    let profiling = prof::enabled();
    match mlp {
        CachedMlpI8::Dense { gate, up, down } => {
            let t = profiling.then(std::time::Instant::now);
            let y = expert_gemv_i8(row, gate, up, down);
            if let Some(t) = t {
                prof::add(&prof::EXPERTS_NS, t.elapsed().as_nanos() as u64);
            }
            Ok(y)
        }
        CachedMlpI8::Moe { gate, experts, shared } => {
            let tr = profiling.then(std::time::Instant::now);
            let routing = moe::route_default(normed, gate)?;
            if let Some(t) = tr {
                prof::add(&prof::ROUTE_NS, t.elapsed().as_nanos() as u64);
            }
            let te = profiling.then(std::time::Instant::now);
            // Fan the 6 routed experts across the pool as FULLY SERIAL tasks (one
            // dispatch, no nested parallelism), CONCURRENTLY with the larger shared
            // expert (inter 1792), which keeps its own internal parallelism via
            // `rayon::join`. Measured best of the schemes tried: the shared expert
            // gets the spare cores while the 6 routed saturate the rest, vs an
            // all-serial fan-out where the shared becomes a single-core long pole.
            let idx = routing.indices[0];
            let wts = routing.weights[0];
            let (partials, s) = rayon::join(
                || {
                    (0..moe::config::NUM_EXPERTS_PER_TOK)
                        .into_par_iter()
                        .map(|j| {
                            let e = idx[j];
                            let w = wts[j];
                            let mut y = expert_gemv_i8_serial(
                                row,
                                &experts[e][0],
                                &experts[e][1],
                                &experts[e][2],
                            );
                            for v in y.iter_mut() {
                                *v *= w;
                            }
                            y
                        })
                        .collect::<Vec<Vec<f32>>>()
                },
                || expert_gemv_i8(row, &shared[0], &shared[1], &shared[2]),
            );
            let mut out = vec![0.0f32; hidden];
            for p in &partials {
                for c in 0..hidden {
                    out[c] += p[c];
                }
            }
            for c in 0..hidden {
                out[c] += s[c];
            }
            if let Some(t) = te {
                prof::add(&prof::EXPERTS_NS, t.elapsed().as_nanos() as u64);
            }
            Ok(out)
        }
    }
}

/// Final RMSNorm + int8 `lm_head` over the decode hidden `[1, hidden]`. Int8 twin
/// of [`lm_head_cached`].
///
/// # Errors
/// [`FocrError::Other`] on a shape mismatch.
pub fn lm_head_cached_i8(wc: &DecoderWeightCacheI8, hidden: &Mat) -> FocrResult<Mat> {
    if hidden.rows != 1 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "decoder::lm_head_cached_i8: expected a single decode row, got {} rows",
            hidden.rows
        )));
    }
    let t = prof::enabled().then(std::time::Instant::now);
    let normed = nn::rms_norm(hidden, Some(&wc.final_norm), config::RMS_NORM_EPS)?;
    let logits = gemv_i8(normed.row(0), &wc.lm_head);
    if let Some(t) = t {
        prof::add(&prof::LMHEAD_NS, t.elapsed().as_nanos() as u64);
    }
    Ok(Mat::from_vec(1, config::VOCAB_SIZE, logits))
}

/// Int8 prefill: the [`prefill_with_cache`] structure with the GEMMs routed through
/// [`nn::linear_int8_dynamic`] (threaded SDOT/VNNI). Captures each layer's RoPE'd
/// K/V into the R-SWA ring just like the f32 path. Returns the final
/// `model.norm`-ready hidden `[seq, hidden]` + the 12 populated caches.
///
/// # Errors
/// As [`prefill_with_cache`].
pub fn prefill_with_cache_i8(
    wc: &DecoderWeightCacheI8,
    inputs_embeds: &Mat,
) -> FocrResult<(Mat, Vec<RingCache>)> {
    checked_mat_len("decoder::prefill_with_cache_i8 inputs_embeds", inputs_embeds)?;
    let hidden = config::HIDDEN_SIZE;
    let num_heads = config::NUM_ATTENTION_HEADS;
    let head_dim = config::HEAD_DIM;
    // Validate num_heads*head_dim doesn't overflow (the int8 linears infer the
    // qkv width from `q_proj.n`, so the product itself isn't threaded through).
    let _ = checked_shape_mul(
        "decoder::prefill_with_cache_i8",
        num_heads,
        head_dim,
        "num_heads*head_dim",
    )?;
    let eps = config::RMS_NORM_EPS;
    if inputs_embeds.cols != hidden {
        return Err(FocrError::Other(anyhow::anyhow!(
            "decoder::prefill_with_cache_i8: inputs_embeds cols {} != hidden {hidden}",
            inputs_embeds.cols
        )));
    }
    let seq = inputs_embeds.rows;
    let positions: Vec<usize> = (0..seq).collect();
    let rope = RopeTable::build(&positions, head_dim, config::ROPE_THETA);
    let mut caches: Vec<RingCache> = (0..config::NUM_HIDDEN_LAYERS)
        .map(|_| RingCache::new(seq.max(1)))
        .collect();

    let mut x = inputs_embeds.clone();
    for layer in 0..config::NUM_HIDDEN_LAYERS {
        let cl = &wc.layers[layer];
        let normed = nn::rms_norm(&x, Some(&cl.input_ln), eps)?;
        let mut q = nn::linear_int8_dynamic(&normed, &cl.q_proj, None)?;
        let mut k = nn::linear_int8_dynamic(&normed, &cl.k_proj, None)?;
        let v = nn::linear_int8_dynamic(&normed, &cl.v_proj, None)?;
        apply_rope(&mut q, &rope)?;
        apply_rope(&mut k, &rope)?;
        let (kh, vh) = token_major_to_head_major(&k, &v, seq, num_heads, head_dim)?;
        caches[layer].record_prefill(&kh, &vh, seq)?;
        let context = prefill_attention(&q, &k, &v, num_heads, head_dim)?;
        let attn_out = nn::linear_int8_dynamic(&context, &cl.o_proj, None)?;
        let h = add_residual(&x, &attn_out)?;
        let normed2 = nn::rms_norm(&h, Some(&cl.post_attn_ln), eps)?;
        let mlp_out = prefill_mlp_i8(&cl.mlp, &normed2)?;
        x = add_residual(&h, &mlp_out)?;
    }
    Ok((x, caches))
}

/// One int8 incremental decode step over the per-layer [`RingCache`]s. Int8 twin
/// of [`decode_step_with_cache`] — every projection via the bespoke m=1 [`gemv_i8`]
/// (NEON SDOT / x86 VNNI), R-SWA attention and ring contract identical.
///
/// # Errors
/// As [`decode_step_with_cache`].
pub fn decode_step_with_cache_i8(
    wc: &DecoderWeightCacheI8,
    caches: &mut [RingCache],
    token_embed: &Mat,
    position: usize,
) -> FocrResult<Mat> {
    let hidden = config::HIDDEN_SIZE;
    let qkv_dim = checked_shape_mul(
        "decoder::decode_step_with_cache_i8",
        config::NUM_ATTENTION_HEADS,
        config::HEAD_DIM,
        "num_heads*head_dim",
    )?;
    let eps = config::RMS_NORM_EPS;
    if token_embed.rows != 1 || token_embed.cols != hidden {
        return Err(FocrError::Other(anyhow::anyhow!(
            "decoder::decode_step_with_cache_i8: token_embed shape [{}, {}] != [1, {hidden}]",
            token_embed.rows,
            token_embed.cols
        )));
    }
    if caches.len() != config::NUM_HIDDEN_LAYERS {
        return Err(FocrError::Other(anyhow::anyhow!(
            "decoder::decode_step_with_cache_i8: {} caches != {} layers",
            caches.len(),
            config::NUM_HIDDEN_LAYERS
        )));
    }
    let rope = RopeTable::build(&[position], config::HEAD_DIM, config::ROPE_THETA);
    let profiling = prof::enabled();
    let mut x = token_embed.clone();
    for layer in 0..config::NUM_HIDDEN_LAYERS {
        let cl = &wc.layers[layer];
        let t_attn = profiling.then(std::time::Instant::now);
        let normed = nn::rms_norm(&x, Some(&cl.input_ln), eps)?;
        let nrow = normed.row(0);
        let (mut q, mut k, v) = if let Some(qkv) = &cl.qkv {
            // FOCR_QKV_FUSED: ONE quantize of `nrow`, ONE block-parallel GEMV over
            // all 3*qkv_dim output rows, then slice into q/k/v. Byte-identical to
            // the three-call `else` branch (each output row is an independent dot).
            let out = gemv_i8(nrow, qkv);
            let q = Mat::from_vec(1, qkv_dim, out[0..qkv_dim].to_vec());
            let k = Mat::from_vec(1, qkv_dim, out[qkv_dim..2 * qkv_dim].to_vec());
            let v = out[2 * qkv_dim..3 * qkv_dim].to_vec();
            (q, k, v)
        } else {
            let q = Mat::from_vec(1, qkv_dim, gemv_i8(nrow, &cl.q_proj));
            let k = Mat::from_vec(1, qkv_dim, gemv_i8(nrow, &cl.k_proj));
            let v = gemv_i8(nrow, &cl.v_proj);
            (q, k, v)
        };
        apply_rope(&mut q, &rope)?;
        apply_rope(&mut k, &rope)?;
        caches[layer].write_decode_step(&k.data, &v)?;
        let context = rswa::decode_attention(&caches[layer], &q.data)?;
        let attn_out = Mat::from_vec(1, hidden, gemv_i8(&context.data, &cl.o_proj));
        let h = add_residual(&x, &attn_out)?;
        if let Some(t) = t_attn {
            prof::add(&prof::ATTN_NS, t.elapsed().as_nanos() as u64);
        }
        let normed2 = nn::rms_norm(&h, Some(&cl.post_attn_ln), eps)?;
        let mlp_out = Mat::from_vec(1, hidden, decode_mlp_i8(&cl.mlp, &normed2)?);
        x = add_residual(&h, &mlp_out)?;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Fused q/k/v GEMV bit-identity (FOCR_QKV_FUSED, bd-241s) ──────────────────

    /// The fused `[3*qkv_dim, hidden]` GEMV must reproduce, BYTE-FOR-BYTE, the q/k/v
    /// that three separate [`gemv_i8`] calls produce. `qkv_dim` is deliberately NOT
    /// a multiple of [`I8_GEMV_BLOCK`] (64) so the fused output's rayon row-chunks
    /// straddle the q|k|v seams — proving each output row is an independent dot
    /// product whose value is invariant to how rows are grouped into tasks.
    #[test]
    fn fused_qkv_gemv_is_byte_identical_to_three_calls() {
        let qkv_dim = 70usize; // crosses a 64-row block boundary, not a multiple of 64
        let hidden = 96usize;
        // Deterministic int8 weights (full [-127,127] range) + per-channel scales.
        let mk = |salt: i32| -> QInt8 {
            let mut w = vec![0i8; qkv_dim * hidden];
            for o in 0..qkv_dim {
                for i in 0..hidden {
                    let raw = ((o as i32 * 31 + i as i32 * 7 + salt * 101) % 255) - 127;
                    w[o * hidden + i] = raw as i8;
                }
            }
            let scales: Vec<f32> = (0..qkv_dim)
                .map(|o| 1.0e-3 + (o as f32 + salt as f32 * 0.5) * 1.0e-4)
                .collect();
            QInt8::new(w, scales, qkv_dim, hidden)
        };
        let q_proj = mk(0);
        let k_proj = mk(1);
        let v_proj = mk(2);
        // Deterministic activation row spanning negatives/positives (exercises the
        // dynamic per-row quantize: amax, clamp, round).
        let nrow: Vec<f32> = (0..hidden)
            .map(|i| (i as f32 * 0.37).sin() * 2.5 - 0.4)
            .collect();

        // Default path: three separate GEMVs (each re-quantizes `nrow`).
        let q_sep = gemv_i8(&nrow, &q_proj);
        let k_sep = gemv_i8(&nrow, &k_proj);
        let v_sep = gemv_i8(&nrow, &v_proj);

        // Fused path: one stacked weight, one GEMV, sliced back.
        let fused = fuse_qkv(&q_proj, &k_proj, &v_proj);
        assert_eq!(fused.n, 3 * qkv_dim);
        assert_eq!(fused.k, hidden);
        let y = gemv_i8(&nrow, &fused);

        // Compare RAW BIT PATTERNS (not approximate) — this is a bit-exact lever.
        let bits = |s: &[f32]| s.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
        assert_eq!(bits(&q_sep), bits(&y[0..qkv_dim]), "q mismatch");
        assert_eq!(bits(&k_sep), bits(&y[qkv_dim..2 * qkv_dim]), "k mismatch");
        assert_eq!(
            bits(&v_sep),
            bits(&y[2 * qkv_dim..3 * qkv_dim]),
            "v mismatch"
        );
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

    // ── R-SWA cache repack ([SPEC-091]) ─────────────────────────────────────────

    #[test]
    fn token_major_to_head_major_transposes_kv() -> FocrResult<()> {
        // seq=2, num_heads=2, head_dim=2 -> token-major row r = [h0d0,h0d1,h1d0,h1d1].
        // row0 = [0,1, 2,3]  row1 = [4,5, 6,7]
        let k = Mat::from_vec(2, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let v = Mat::from_vec(2, 4, vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);
        let (kh, vh) = token_major_to_head_major(&k, &v, 2, 2, 2)?;
        // head-major [num_heads, seq, head_dim]: head0 = rows' first head across seq
        //   head0: s0=[0,1] s1=[4,5] ; head1: s0=[2,3] s1=[6,7]
        assert_eq!(kh, vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);
        assert_eq!(vh, vec![10.0, 11.0, 14.0, 15.0, 12.0, 13.0, 16.0, 17.0]);
        Ok(())
    }

    // ── Token embedding ([SPEC-070]) ────────────────────────────────────────

    #[test]
    fn embed_tokens_gathers_rows() -> FocrResult<()> {
        // vocab=3, hidden=2 table: row0=[0,1] row1=[2,3] row2=[4,5]
        let table = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let out = embed_tokens(&table, 3, 2, &[2, 0, 1, 2])?;
        assert_eq!(out.shape(), (4, 2));
        assert_eq!(out.row(0), &[4.0, 5.0]);
        assert_eq!(out.row(1), &[0.0, 1.0]);
        assert_eq!(out.row(2), &[2.0, 3.0]);
        assert_eq!(out.row(3), &[4.0, 5.0]);
        Ok(())
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

    #[test]
    fn embed_tokens_rejects_table_shape_product_overflow_without_panic() {
        assert_err_contains(embed_tokens(&[], usize::MAX, 2, &[]), "vocab*hidden");
    }

    #[test]
    fn embed_tokens_rejects_output_shape_product_overflow_without_panic() {
        assert_err_contains(embed_tokens(&[], 0, usize::MAX, &[0, 1]), "seq*hidden");
    }

    // ── RoPE ([SPEC-078]) ───────────────────────────────────────────────────

    #[test]
    fn rope_position_zero_is_identity() -> FocrResult<()> {
        // head_dim=4, single head; pos 0 => cos=1, sin=0 => x unchanged.
        let rope = RopeTable::build(&[0], 4, 10000.0);
        let mut x = Mat::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
        apply_rope(&mut x, &rope)?;
        for (got, want) in x.data.iter().zip([1.0f32, 2.0, 3.0, 4.0].iter()) {
            assert!((got - want).abs() < 1e-6, "{got} != {want}");
        }
        Ok(())
    }

    #[test]
    fn rope_matches_hand_computed_rotate_half() -> FocrResult<()> {
        // head_dim=2 => half=1, inv_freq[0] = 10000^0 = 1, so angle = pos.
        // rotate_half([a,b]) = [-b, a].
        // out[0] = a*cos - b*sin ; out[1] = b*cos + a*sin.
        let theta = 10000.0f32;
        let pos = 1usize;
        let rope = RopeTable::build(&[pos], 2, theta);
        let (a, b) = (3.0f32, 5.0f32);
        let mut x = Mat::from_vec(1, 2, vec![a, b]);
        apply_rope(&mut x, &rope)?;
        let (s, c) = (pos as f32).sin_cos();
        let want0 = a * c - b * s;
        let want1 = b * c + a * s;
        assert!((x.data[0] - want0).abs() < 1e-5, "{} != {want0}", x.data[0]);
        assert!((x.data[1] - want1).abs() < 1e-5, "{} != {want1}", x.data[1]);
        Ok(())
    }

    #[test]
    fn rope_preserves_norm_per_head() -> FocrResult<()> {
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
        apply_rope(&mut x, &rope)?;
        let after = head_norms(&x.data);
        for (b, a) in before.iter().zip(after.iter()) {
            assert!((b - a).abs() < 1e-4, "norm changed: {b} -> {a}");
        }
        Ok(())
    }

    #[test]
    fn rope_two_heads_share_phase() -> FocrResult<()> {
        // head_dim=2, two heads packed in one row -> each head rotated by the
        // SAME phase (RoPE is per-position, shared across heads).
        let rope = RopeTable::build(&[2], 2, 10000.0);
        let mut x = Mat::from_vec(1, 4, vec![1.0, 0.0, 0.0, 1.0]);
        apply_rope(&mut x, &rope)?;
        let (s, c) = (2.0f32).sin_cos();
        // head0 = [1,0] -> [c, s]; head1 = [0,1] -> [-s, c]
        assert!((x.data[0] - c).abs() < 1e-5);
        assert!((x.data[1] - s).abs() < 1e-5);
        assert!((x.data[2] - (-s)).abs() < 1e-5);
        assert!((x.data[3] - c).abs() < 1e-5);
        Ok(())
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

    #[test]
    fn rope_rejects_malformed_backing_data_without_panic() {
        let rope = RopeTable::build(&[0], 4, 10000.0);
        let mut bad = Mat {
            rows: 1,
            cols: 4,
            data: vec![1.0, 2.0, 3.0],
        };
        assert_err_contains(apply_rope(&mut bad, &rope), "apply_rope x: data len 3");
    }

    #[test]
    #[should_panic(expected = "RopeTable: seq*head_dim overflow")]
    fn rope_table_rejects_shape_product_overflow_before_allocating() {
        let _ = RopeTable::build(&[0, 1], usize::MAX - 1, 10000.0);
    }

    #[test]
    fn rope_table_empty_sequence_does_not_allocate_by_head_dim() {
        let rope = RopeTable::build(&[], usize::MAX - 1, 10000.0);
        assert_eq!(rope.head_dim, usize::MAX - 1);
        assert!(rope.cos.is_empty());
        assert!(rope.sin.is_empty());
    }

    // ── Dense SwiGLU MLP ([SPEC-075]) ───────────────────────────────────────

    #[test]
    fn dense_mlp_matches_hand_computed() -> FocrResult<()> {
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
        let out = dense_mlp(&x, &gate_w, &up_w, &down_w, 2, 2)?;
        assert_eq!(out.shape(), (1, 2));
        let silu1 = 1.0f32 / (1.0 + (-1.0f32).exp());
        assert!((out.data[0] - silu1).abs() < 1e-5, "{}", out.data[0]);
        assert!(out.data[1].abs() < 1e-6, "{}", out.data[1]);
        Ok(())
    }

    #[test]
    fn dense_mlp_rejects_bad_hidden() {
        let x = Mat::from_vec(1, 3, vec![1.0, 0.0, 0.0]);
        assert!(dense_mlp(&x, &[1.0, 0.0], &[1.0, 0.0], &[1.0, 0.0], 2, 1).is_err());
    }

    // ── linear_no_bias (the F.linear [out,in] transpose) ────────────────────

    #[test]
    fn linear_no_bias_transposes_pytorch_layout() -> FocrResult<()> {
        // x=[1,2] (1x2 row); w=[out=3, in=2] row-major:
        //   w = [[1,0],[0,1],[1,1]]  => y = [x·w0, x·w1, x·w2] = [1, 2, 3]
        let x = Mat::from_vec(1, 2, vec![1.0, 2.0]);
        let w = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let y = linear_no_bias(&x, &w, 2, 3)?;
        assert_eq!(y.shape(), (1, 3));
        assert_eq!(y.data, vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn linear_no_bias_transposes_multirow_nonsquare_layout() -> FocrResult<()> {
        let x = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let w = vec![
            1.0, 0.0, 0.0, // row 0 selects x0
            0.0, 1.0, 0.0, // row 1 selects x1
            0.0, 0.0, 1.0, // row 2 selects x2
            1.0, 1.0, 1.0, // row 3 sums the input row
        ];
        let y = linear_no_bias(&x, &w, 3, 4)?;
        assert_eq!(y.shape(), (2, 4));
        assert_eq!(y.data, vec![1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0]);
        Ok(())
    }

    #[test]
    fn linear_no_bias_rejects_weight_shape_product_overflow_without_panic() {
        let x = Mat::zeros(1, 2);
        assert_err_contains(linear_no_bias(&x, &[], 2, usize::MAX), "out*in");
    }

    #[test]
    fn linear_no_bias_rejects_malformed_input_backing_data_without_panic() {
        let x = Mat {
            rows: 1,
            cols: 2,
            data: vec![1.0],
        };
        assert_err_contains(
            linear_no_bias(&x, &[1.0, 0.0], 2, 1),
            "linear_no_bias x: data len 1",
        );
    }

    // ── Residual add ([SPEC-072]) ───────────────────────────────────────────

    #[test]
    fn add_residual_sums_elementwise() -> FocrResult<()> {
        let a = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Mat::from_vec(2, 2, vec![10.0, 20.0, 30.0, 40.0]);
        let c = add_residual(&a, &b)?;
        assert_eq!(c.data, vec![11.0, 22.0, 33.0, 44.0]);
        Ok(())
    }

    #[test]
    fn add_residual_rejects_shape_mismatch() {
        let a = Mat::zeros(2, 2);
        let b = Mat::zeros(2, 3);
        assert!(add_residual(&a, &b).is_err());
    }

    #[test]
    fn add_residual_rejects_malformed_backing_data_without_panic() {
        let a = Mat {
            rows: 1,
            cols: 2,
            data: vec![1.0],
        };
        let b = Mat::zeros(1, 2);
        assert_err_contains(add_residual(&a, &b), "add_residual lhs: data len 1");
    }

    // ── Final norm + lm_head ([SPEC-071]/[SPEC-081]) ────────────────────────

    #[test]
    fn lm_head_proj_matches_matmul() -> FocrResult<()> {
        // hidden=2 vocab=3; head_w=[vocab,hidden] = [[1,0],[0,1],[1,1]]
        // hidden state [3,4] -> logits [3, 4, 7]
        let h = Mat::from_vec(1, 2, vec![3.0, 4.0]);
        let head_w = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let logits = lm_head_proj(&h, &head_w, 3)?;
        assert_eq!(logits.shape(), (1, 3));
        assert_eq!(logits.data, vec![3.0, 4.0, 7.0]);
        Ok(())
    }

    #[test]
    fn norm_and_lm_head_composes_rmsnorm_then_head() -> FocrResult<()> {
        // hidden=2, vocab=2. x=[3,4], norm_w=[1,1], eps=0.
        // rmsnorm: mean(x^2)=12.5, rstd=1/sqrt(12.5); normed=[3*rstd,4*rstd]
        // head_w = identity [[1,0],[0,1]] => logits = normed
        let h = Mat::from_vec(1, 2, vec![3.0, 4.0]);
        let norm_w = vec![1.0, 1.0];
        let head_w = vec![1.0, 0.0, 0.0, 1.0];
        let logits = norm_and_lm_head(&h, &norm_w, &head_w, 2, 0.0)?;
        let rstd = 1.0f32 / 12.5f32.sqrt();
        assert!((logits.data[0] - 3.0 * rstd).abs() < 1e-5);
        assert!((logits.data[1] - 4.0 * rstd).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn lm_head_last_row_is_full_last_row() -> FocrResult<()> {
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

        let full_logits = norm_and_lm_head(&full, &norm_w, &head_w, vocab, eps)?;
        let last = Mat::from_vec(1, hidden, full.row(seq - 1).to_vec());
        let last_logits = norm_and_lm_head(&last, &norm_w, &head_w, vocab, eps)?;

        assert_eq!((last_logits.rows, last_logits.cols), (1, vocab));
        let full_last = &full_logits.data[(seq - 1) * vocab..seq * vocab];
        assert_eq!(
            last_logits.data, full_last,
            "last-row lm_head must be bit-identical to full[last]"
        );
        Ok(())
    }

    // ── attn output proj / qkv ([SPEC-090]) ─────────────────────────────────

    #[test]
    fn attn_output_proj_projects_context() -> FocrResult<()> {
        // qkv_dim=2 hidden=2; context=[1,1]; o_proj=[[2,0],[0,3]] -> [2,3]
        let ctx = Mat::from_vec(1, 2, vec![1.0, 1.0]);
        let o = vec![2.0, 0.0, 0.0, 3.0];
        let out = attn_output_proj(&ctx, &o, 2, 2)?;
        assert_eq!(out.data, vec![2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn qkv_with_rope_shapes_and_pos0_identity() -> FocrResult<()> {
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
        let (q, k, v) = qkv_with_rope(&normed, &lw, &rope, 2, 2)?;
        assert_eq!(q.shape(), (1, 2));
        assert_eq!(v.data, vec![1.0, 2.0]); // v never rope'd
        assert_eq!(q.data, vec![1.0, 2.0]); // pos0 rope = identity
        assert_eq!(k.data, vec![1.0, 2.0]);
        Ok(())
    }

    // ── Full layer driver ([SPEC-072]) ──────────────────────────────────────

    #[test]
    fn layer_forward_pre_norm_residual_identity_path() -> FocrResult<()> {
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
        )?;
        assert_eq!(out.shape(), (2, hidden));
        for (got, want) in out.data.iter().zip(x.data.iter()) {
            assert!((got - want).abs() < 1e-6, "{got} != {want}");
        }
        Ok(())
    }

    #[test]
    fn layer_forward_adds_both_sublayers() -> FocrResult<()> {
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
        )?;
        // x + [1,1] + [100,100] = [111, 121]
        assert!((out.data[0] - 111.0).abs() < 1e-4, "{}", out.data[0]);
        assert!((out.data[1] - 121.0).abs() < 1e-4, "{}", out.data[1]);
        Ok(())
    }

    // ── Top-level entrypoints are wired to the Weights accessors ────────────

    #[test]
    fn top_level_entrypoints_error_cleanly_on_empty_weights() {
        // `forward`/`lm_head` now look tensors up by name and delegate to the
        // tested driver; an empty `Weights::default()` has none, so they must
        // surface a clean `FormatMismatch` (tensor not found), never panic.
        let w = Weights::default();
        let h = Mat::zeros(1, config::HIDDEN_SIZE);
        assert!(matches!(forward(&w, &h), Err(FocrError::FormatMismatch(_))));
        assert!(matches!(lm_head(&w, &h), Err(FocrError::FormatMismatch(_))));
    }

    // ── Prefill attention shape + causal correctness ────────────────────────

    #[test]
    fn prefill_attention_shapes_and_first_token_self_only() -> FocrResult<()> {
        // Single head, head_dim=2, seq=2. With a causal mask, row 0 attends only
        // to itself, so its context == v[0] regardless of v[1] / the scores.
        let (num_heads, head_dim, seq) = (1usize, 2usize, 2usize);
        let q = Mat::from_vec(seq, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let k = Mat::from_vec(seq, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let v = Mat::from_vec(seq, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let out = prefill_attention(&q, &k, &v, num_heads, head_dim)?;
        assert_eq!(out.shape(), (seq, num_heads * head_dim));
        // Row 0 (causal): only key 0 visible -> context == v row 0.
        assert!((out.data[0] - 5.0).abs() < 1e-5, "{}", out.data[0]);
        assert!((out.data[1] - 6.0).abs() < 1e-5, "{}", out.data[1]);
        // Row 1 attends to both keys -> a convex blend of v[0] and v[1], so each
        // channel lies strictly between the two value rows.
        assert!(out.data[2] > 5.0 && out.data[2] < 7.0, "{}", out.data[2]);
        assert!(out.data[3] > 6.0 && out.data[3] < 8.0, "{}", out.data[3]);
        Ok(())
    }

    #[test]
    fn prefill_attention_rejects_bad_shape() {
        let q = Mat::zeros(2, 3); // 3 not a multiple of head_dim*heads=4
        let k = Mat::zeros(2, 4);
        let v = Mat::zeros(2, 4);
        assert!(prefill_attention(&q, &k, &v, 2, 2).is_err());
    }

    #[test]
    fn prefill_attention_rejects_malformed_backing_data_without_panic() {
        let q = Mat {
            rows: 2,
            cols: 4,
            data: vec![0.0; 7],
        };
        let k = Mat::zeros(2, 4);
        let v = Mat::zeros(2, 4);
        assert_err_contains(
            prefill_attention(&q, &k, &v, 2, 2),
            "prefill_attention q: data len 7",
        );
    }
}
