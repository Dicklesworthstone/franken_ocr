//! The Qwen2-arch **dense** decoder forward (GOT-OCR2.0, bead B5) — a thin driver
//! that composes the already-parity-tested leaf kernels in [`super::decoder`] with
//! a [`DecoderConfig`], rather than forking the DeepSeek-V2 MoE + R-SWA path (whose
//! `config` module and `RingCache`/`rswa` attention hardcode 1280/12/10/128 and a
//! 128-token sliding window — all wrong for GOT).
//!
//! GOT-OCR2's decoder is Qwen2-0.5B: 24 layers, hidden 1024, 16 heads (no GQA,
//! head_dim 64), dense SwiGLU (intermediate 2816, silu), RoPE θ=1e6, **full-causal**
//! attention (scale 1/8 = 1/√64), q/k/v_proj **with bias** (o_proj none), RMSNorm
//! ε=1e-6, and **tied** embeddings (`lm_head` == `embed_tokens`, stored once,
//! high-precision). Every one of those maps onto an existing kernel:
//! * q/k/v/o + gate/up/down GEMMs → [`nn::linear_int8_dynamic`] (int8, and it
//!   already carries the qkv bias) or the f32 [`super::decoder::linear_no_bias`];
//! * RoPE → [`super::decoder::RopeTable::build`] + [`super::decoder::apply_rope`]
//!   (NEOX rotate-half = HF Qwen2), built once at `head_dim=64, θ=1e6`;
//! * attention → [`super::decoder::prefill_attention`] (full-causal MHA, scale
//!   `1/√head_dim` = 1/8 at head_dim 64);
//! * norms → [`nn::rms_norm`]; tied head → [`super::decoder::norm_and_lm_head`].
//!
//! [`linear_auto`] picks int8 vs f32 **per GEMM** from the loaded tensor's dtype,
//! so the SAME forward certifies both the shipping `got-ocr2.int8.focrq` (int8
//! decoder GEMMs) and the raw bf16 `model.safetensors` (an f32 reference). Parity
//! is held against the bit-deterministic torch oracle (floor = 0) at the decoder
//! seam: feed the oracle's post-splice `hidden_0` and match the last-position
//! logits (see the `#[cfg(test)]` parity gate).
//!
//! Generation: [`generate_greedy`] is the correct O(n²) re-prefill path (the parity
//! oracle); [`generate_greedy_kvcache`] (bead B9) is the O(n)-per-token full-causal
//! **KV-cache** decode used in production — held bit-identical to `generate_greedy`
//! (hence the torch oracle L4) by routing every decode GEMM through the SAME
//! `nn::linear_int8_dynamic` (ties-to-even) the prefill uses.

use super::decoder;
use super::nn;
use super::tensor::{Mat, QInt8};
use super::weights::{DType, Weights};
use crate::error::{FocrError, FocrResult};

/// Config of a dense (Qwen2/Llama-style) decoder — the parameters the shared leaf
/// kernels need, so one driver serves GOT-OCR2 (and later SmolVLM2/OneChart).
#[derive(Debug, Clone, Copy)]
pub struct DecoderConfig {
    /// Residual-stream width (GOT 1024).
    pub hidden_size: usize,
    /// Dense SwiGLU inner width (GOT 2816).
    pub intermediate_size: usize,
    /// Number of transformer layers (GOT 24).
    pub num_hidden_layers: usize,
    /// Attention heads (GOT 16).
    pub num_attention_heads: usize,
    /// Per-head dim (GOT 64); `num_attention_heads * head_dim == hidden` for GOT.
    pub head_dim: usize,
    /// Vocabulary (GOT 151860).
    pub vocab_size: usize,
    /// RoPE base θ (GOT 1e6).
    pub rope_theta: f32,
    /// RMSNorm ε (GOT 1e-6).
    pub rms_norm_eps: f32,
    /// Whether q/k/v_proj carry a bias (GOT true; o_proj never does).
    pub attn_qkv_bias: bool,
}

impl DecoderConfig {
    /// The GOT-OCR2.0 (Qwen2-0.5B) decoder configuration (`config.json`, spec §4).
    #[must_use]
    pub fn got_ocr2() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 2816,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            head_dim: 64,
            vocab_size: 151_860,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            attn_qkv_bias: true,
        }
    }

    /// The q/k/v/o projection width (`num_attention_heads * head_dim`). Equals
    /// `hidden_size` for GOT (no GQA), but kept distinct so the driver survives a
    /// GQA model.
    #[must_use]
    pub fn qkv_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
}

/// One GEMM `y = x @ W^T (+ bias)` where `W` (name `weight_name` in `weights`) is
/// either a pre-quantized `QInt8PerChan` record (the `.focrq` shipping path →
/// [`nn::linear_int8_dynamic`], which dequantizes then adds `bias` in f32) or a
/// high-precision bf16/f32 record (the reference path → [`decoder::linear_no_bias`]
/// + a manual bias add). Dispatching per tensor lets ONE forward certify both.
fn linear_auto(
    weights: &Weights,
    x: &Mat,
    weight_name: &str,
    in_: usize,
    out: usize,
    bias: Option<&[f32]>,
) -> FocrResult<Mat> {
    let is_int8 = matches!(
        weights.record(weight_name).map(|r| r.dtype),
        Some(DType::QInt8PerChan)
    );
    if is_int8 {
        let qw = decoder::quant_oc_loaded(weights, weight_name, out)?;
        nn::linear_int8_dynamic(x, &qw, bias)
    } else {
        let w = weights.mat(weight_name)?;
        if w.data.len() != out * in_ {
            return Err(FocrError::FormatMismatch(format!(
                "decoder_qwen2: {weight_name} has {} elems, expected {out}*{in_}",
                w.data.len()
            )));
        }
        let mut y = decoder::linear_no_bias(x, &w.data, in_, out)?;
        if let Some(b) = bias {
            add_bias(&mut y, b);
        }
        Ok(y)
    }
}

/// Add a per-output-channel bias to each row (f32 reference path; the int8 path
/// adds bias inside `linear_int8_dynamic` at the same point — after dequant).
fn add_bias(y: &mut Mat, bias: &[f32]) {
    debug_assert_eq!(bias.len(), y.cols);
    for row in y.data.chunks_mut(y.cols) {
        for (v, &b) in row.iter_mut().zip(bias.iter()) {
            *v += b;
        }
    }
}

/// Reject Qwen3-style per-head q/k norms (Qwen2 has none) — their silent presence
/// would be a silent parity divergence (spec §13a OQ).
fn assert_no_qk_norm(weights: &Weights, layer_prefix: &str) -> FocrResult<()> {
    for suffix in [".self_attn.q_norm.weight", ".self_attn.k_norm.weight"] {
        let name = format!("{layer_prefix}{suffix}");
        if weights.record(&name).is_some() {
            return Err(FocrError::FormatMismatch(format!(
                "decoder_qwen2: unexpected {name} — Qwen2 has no q/k-norm; refusing to run \
                 a mismatched architecture"
            )));
        }
    }
    Ok(())
}

/// One Qwen2 dense layer over the prefill activations `x: [seq, hidden]`.
fn qwen2_layer(
    weights: &Weights,
    x: &Mat,
    layer: usize,
    rope: &decoder::RopeTable,
    cfg: &DecoderConfig,
) -> FocrResult<Mat> {
    let p = format!("model.layers.{layer}");
    assert_no_qk_norm(weights, &p)?;
    let eps = cfg.rms_norm_eps;
    let (hidden, qkv_dim, inter) = (cfg.hidden_size, cfg.qkv_dim(), cfg.intermediate_size);

    // ── attention ────────────────────────────────────────────────────────────
    let input_ln = weights.vec(&format!("{p}.input_layernorm.weight"))?;
    let normed = nn::rms_norm(x, Some(&input_ln), eps)?;

    let (q_b, k_b, v_b) = if cfg.attn_qkv_bias {
        (
            Some(weights.vec(&format!("{p}.self_attn.q_proj.bias"))?),
            Some(weights.vec(&format!("{p}.self_attn.k_proj.bias"))?),
            Some(weights.vec(&format!("{p}.self_attn.v_proj.bias"))?),
        )
    } else {
        (None, None, None)
    };
    let mut q = linear_auto(
        weights,
        &normed,
        &format!("{p}.self_attn.q_proj.weight"),
        hidden,
        qkv_dim,
        q_b.as_deref(),
    )?;
    let mut k = linear_auto(
        weights,
        &normed,
        &format!("{p}.self_attn.k_proj.weight"),
        hidden,
        qkv_dim,
        k_b.as_deref(),
    )?;
    let v = linear_auto(
        weights,
        &normed,
        &format!("{p}.self_attn.v_proj.weight"),
        hidden,
        qkv_dim,
        v_b.as_deref(),
    )?;

    decoder::apply_rope(&mut q, rope)?;
    decoder::apply_rope(&mut k, rope)?;
    let ctx = decoder::prefill_attention(&q, &k, &v, cfg.num_attention_heads, cfg.head_dim)?;
    let attn = linear_auto(
        weights,
        &ctx,
        &format!("{p}.self_attn.o_proj.weight"),
        qkv_dim,
        hidden,
        None,
    )?;
    let h = decoder::add_residual(x, &attn)?;

    // ── dense SwiGLU MLP ──────────────────────────────────────────────────────
    let post_ln = weights.vec(&format!("{p}.post_attention_layernorm.weight"))?;
    let normed2 = nn::rms_norm(&h, Some(&post_ln), eps)?;
    let mut g = linear_auto(
        weights,
        &normed2,
        &format!("{p}.mlp.gate_proj.weight"),
        hidden,
        inter,
        None,
    )?;
    nn::silu(&mut g);
    let u = linear_auto(
        weights,
        &normed2,
        &format!("{p}.mlp.up_proj.weight"),
        hidden,
        inter,
        None,
    )?;
    for (a, &b) in g.data.iter_mut().zip(u.data.iter()) {
        *a *= b;
    }
    let mlp = linear_auto(
        weights,
        &g,
        &format!("{p}.mlp.down_proj.weight"),
        inter,
        hidden,
        None,
    )?;
    decoder::add_residual(&h, &mlp)
}

/// Run the dense decoder prefill over `inputs_embeds: [seq, hidden]` (the
/// post-`<imgpad>`-splice decoder input) through all layers and the final norm +
/// **tied** lm_head, returning logits `[seq, vocab]`.
///
/// `weights` may be the int8 `got-ocr2.int8.focrq` or the raw bf16 safetensors;
/// [`linear_auto`] adapts per GEMM. The lm_head is always f32 (the tied
/// `model.embed_tokens.weight`, HP).
///
/// # Errors
/// [`FocrError`] on a shape mismatch, a missing tensor, or a rejected q/k-norm.
pub fn forward_prefill(
    weights: &Weights,
    cfg: &DecoderConfig,
    inputs_embeds: &Mat,
) -> FocrResult<Mat> {
    if inputs_embeds.cols != cfg.hidden_size {
        return Err(FocrError::FormatMismatch(format!(
            "decoder_qwen2: inputs_embeds cols {} != hidden {}",
            inputs_embeds.cols, cfg.hidden_size
        )));
    }
    let positions: Vec<usize> = (0..inputs_embeds.rows).collect();
    let rope = decoder::RopeTable::build(&positions, cfg.head_dim, cfg.rope_theta);

    let mut x = inputs_embeds.clone();
    for layer in 0..cfg.num_hidden_layers {
        x = qwen2_layer(weights, &x, layer, &rope, cfg)?;
    }

    // final RMSNorm + tied lm_head (embed_tokens^T, f32).
    let final_norm = weights.vec("model.norm.weight")?;
    let embed = weights.mat("model.embed_tokens.weight")?;
    decoder::norm_and_lm_head(
        &x,
        &final_norm,
        &embed.data,
        cfg.vocab_size,
        cfg.rms_norm_eps,
    )
}

/// Greedy (argmax, temperature-0) autoregressive decode from `inputs_embeds`,
/// generating up to `max_new` tokens and stopping at `eos`. Returns the generated
/// id-stream (excluding the prompt).
///
/// This is the **correct, unoptimized** generation path: each step re-runs the
/// full prefill over the grown sequence (O(n²) — a KV-cache decode-step is the
/// perf follow-on, bead B9), appending the argmax token's embedding. It reproduces
/// the torch oracle's greedy L4 output; correctness first, then speed (doctrine #1).
///
/// # Errors
/// Any [`forward_prefill`] error, or a missing `model.embed_tokens.weight`.
pub fn generate_greedy(
    weights: &Weights,
    cfg: &DecoderConfig,
    inputs_embeds: &Mat,
    max_new: usize,
    eos: u32,
) -> FocrResult<Vec<u32>> {
    let embed = weights.mat("model.embed_tokens.weight")?;
    let (vocab, hidden) = (embed.rows, embed.cols);
    let mut data = inputs_embeds.data.clone();
    let mut ids = Vec::new();
    for _ in 0..max_new {
        let rows = data.len() / hidden;
        let cur = Mat::from_vec(rows, hidden, std::mem::take(&mut data));
        let logits = forward_prefill(weights, cfg, &cur)?;
        let last = &logits.data[(logits.rows - 1) * vocab..];
        let next = last
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            })
            .0 as u32;
        ids.push(next);
        // reclaim the prefix embeds and append the chosen token's embedding.
        data = cur.data;
        if next == eos {
            break;
        }
        let te = decoder::embed_tokens(&embed.data, vocab, hidden, &[next])?;
        data.extend_from_slice(&te.data);
    }
    Ok(ids)
}

// ── B9: full-causal KV-cache decode (O(n)/token, replaces the O(n²) re-prefill) ──
//
// The decode path is held BIT-IDENTICAL to [`generate_greedy`] (hence to the torch
// oracle L4) by routing every decode GEMM through the SAME [`nn::linear_int8_dynamic`]
// (m=1) the prefill uses — its ties-to-even activation quant avoids the rounding gap
// the standalone `decoder::gemv_i8` (half-away) would introduce. The bespoke
// prequant-fused `gemv_i8_bias` is a ledgered perf follow-on; correctness first.

/// One decoder layer's full-causal KV cache: post-RoPE **K** and post-proj **V**,
/// token-major `[n_kv, qkv_dim]`, grown by one row per decode step. NOT R-SWA — GOT
/// Qwen2 attends the whole prefix (no window, no eviction, f32 KV).
struct Qwen2KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    n_kv: usize,
    qkv_dim: usize,
}

impl Qwen2KvCache {
    fn new(qkv_dim: usize, max_positions: usize) -> Self {
        Self {
            k: Vec::with_capacity(max_positions * qkv_dim),
            v: Vec::with_capacity(max_positions * qkv_dim),
            n_kv: 0,
            qkv_dim,
        }
    }
    /// Seed all `N` prefill rows at once (`k_all`/`v_all` are `[N, qkv_dim]`).
    fn seed(&mut self, k_all: &[f32], v_all: &[f32]) {
        self.k.extend_from_slice(k_all);
        self.v.extend_from_slice(v_all);
        self.n_kv += k_all.len() / self.qkv_dim;
    }
    /// Append one decode step's k/v row (each `[qkv_dim]`).
    fn append(&mut self, k_row: &[f32], v_row: &[f32]) {
        self.k.extend_from_slice(k_row);
        self.v.extend_from_slice(v_row);
        self.n_kv += 1;
    }
}

/// Full-causal m=1 attention: the single new query (`q_row`, `[qkv_dim]`) attends
/// ALL `n_kv` cached keys (every cached position ≤ the current one — no mask needed).
/// scale `1/√head_dim` = 1/8.
///
/// Reads the **token-major** cache directly — no per-step head-major repack (the old
/// `nn::sdpa` path allocated + copied `2·num_heads·n_kv·head_dim` floats EVERY token,
/// an O(n²) alloc-churn that dominated long-page decode). This bespoke scaled-dot-product
/// (per-head dot → softmax → weighted V, all f32) is the standard attention math; it is
/// argmax-exact vs the sdpa path (certified: `kvcache_greedy_matches_oracle_l4` still ==
/// the oracle L4). Per-step temp is just `[n_kv]` scores. (Perf lever, bead B9.)
fn qwen2_decode_attention(
    cache: &Qwen2KvCache,
    q_row: &[f32],
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let n_kv = cache.n_kv;
    let dim = num_heads * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; dim];
    let mut scores = vec![0.0f32; n_kv];
    for h in 0..num_heads {
        let qh = &q_row[h * head_dim..h * head_dim + head_dim];
        // scores[r] = scale · (q_h · k[r, head h]); track the max for a stable softmax.
        let mut smax = f32::NEG_INFINITY;
        for (r, s) in scores.iter_mut().enumerate() {
            let base = r * dim + h * head_dim;
            let kh = &cache.k[base..base + head_dim];
            let dot: f32 = qh.iter().zip(kh).map(|(&a, &b)| a * b).sum();
            *s = dot * scale;
            smax = smax.max(*s);
        }
        // softmax over the cached positions.
        let mut denom = 0.0f32;
        for s in &mut scores {
            *s = (*s - smax).exp();
            denom += *s;
        }
        let inv = 1.0 / denom;
        // out_h = Σ_r softmax[r] · v[r, head h].
        let oh = &mut out[h * head_dim..h * head_dim + head_dim];
        for (r, &s) in scores.iter().enumerate() {
            let w = s * inv;
            let base = r * dim + h * head_dim;
            let vh = &cache.v[base..base + head_dim];
            for (o, &vv) in oh.iter_mut().zip(vh) {
                *o += w * vv;
            }
        }
    }
    out
}

/// One layer's decode weights, loaded ONCE (int8 GEMMs read verbatim from the
/// `.focrq` `QInt8PerChan` records, f32 norms/biases) so no weight is re-read per token.
struct GotLayerW {
    input_ln: Vec<f32>,
    post_attn_ln: Vec<f32>,
    q: QInt8,
    k: QInt8,
    v: QInt8,
    o: QInt8,
    gate: QInt8,
    up: QInt8,
    down: QInt8,
    q_b: Vec<f32>,
    k_b: Vec<f32>,
    v_b: Vec<f32>,
}

/// The whole GOT decoder's decode-time weights (pre-loaded once for a generation).
struct GotDecodeWeights {
    layers: Vec<GotLayerW>,
    final_norm: Vec<f32>,
    embed: Vec<f32>,
    cfg: DecoderConfig,
}

impl GotDecodeWeights {
    fn build(weights: &Weights, cfg: &DecoderConfig) -> FocrResult<Self> {
        let (hidden, qkv_dim, inter) = (cfg.hidden_size, cfg.qkv_dim(), cfg.intermediate_size);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}");
            layers.push(GotLayerW {
                input_ln: weights.vec(&format!("{p}.input_layernorm.weight"))?,
                post_attn_ln: weights.vec(&format!("{p}.post_attention_layernorm.weight"))?,
                q: decoder::quant_oc_loaded(
                    weights,
                    &format!("{p}.self_attn.q_proj.weight"),
                    qkv_dim,
                )?,
                k: decoder::quant_oc_loaded(
                    weights,
                    &format!("{p}.self_attn.k_proj.weight"),
                    qkv_dim,
                )?,
                v: decoder::quant_oc_loaded(
                    weights,
                    &format!("{p}.self_attn.v_proj.weight"),
                    qkv_dim,
                )?,
                o: decoder::quant_oc_loaded(
                    weights,
                    &format!("{p}.self_attn.o_proj.weight"),
                    hidden,
                )?,
                gate: decoder::quant_oc_loaded(
                    weights,
                    &format!("{p}.mlp.gate_proj.weight"),
                    inter,
                )?,
                up: decoder::quant_oc_loaded(weights, &format!("{p}.mlp.up_proj.weight"), inter)?,
                down: decoder::quant_oc_loaded(
                    weights,
                    &format!("{p}.mlp.down_proj.weight"),
                    hidden,
                )?,
                q_b: weights.vec(&format!("{p}.self_attn.q_proj.bias"))?,
                k_b: weights.vec(&format!("{p}.self_attn.k_proj.bias"))?,
                v_b: weights.vec(&format!("{p}.self_attn.v_proj.bias"))?,
            });
        }
        Ok(Self {
            layers,
            final_norm: weights.vec("model.norm.weight")?,
            embed: weights.mat("model.embed_tokens.weight")?.data,
            cfg: *cfg,
        })
    }
}

/// Seeding prefill: run all `N` positions through the layers (BIT-IDENTICAL to
/// [`forward_prefill`] — same `linear_int8_dynamic` kernel), capturing each layer's
/// post-RoPE K + post-proj V into `caches`, and return the **last-position** logits.
fn forward_prefill_seed(
    w: &GotDecodeWeights,
    inputs_embeds: &Mat,
    caches: &mut [Qwen2KvCache],
) -> FocrResult<Vec<f32>> {
    let cfg = &w.cfg;
    let eps = cfg.rms_norm_eps;
    let mut x = inputs_embeds.clone();
    let positions: Vec<usize> = (0..inputs_embeds.rows).collect();
    let rope = decoder::RopeTable::build(&positions, cfg.head_dim, cfg.rope_theta);
    for (l, cl) in w.layers.iter().enumerate() {
        let normed = nn::rms_norm(&x, Some(&cl.input_ln), eps)?;
        let mut q = nn::linear_int8_dynamic(&normed, &cl.q, Some(&cl.q_b))?;
        let mut k = nn::linear_int8_dynamic(&normed, &cl.k, Some(&cl.k_b))?;
        let v = nn::linear_int8_dynamic(&normed, &cl.v, Some(&cl.v_b))?;
        decoder::apply_rope(&mut q, &rope)?;
        decoder::apply_rope(&mut k, &rope)?;
        caches[l].seed(&k.data, &v.data); // the very K/V prefill_attention consumes
        let ctx = decoder::prefill_attention(&q, &k, &v, cfg.num_attention_heads, cfg.head_dim)?;
        let attn = nn::linear_int8_dynamic(&ctx, &cl.o, None)?;
        let h = decoder::add_residual(&x, &attn)?;
        let normed2 = nn::rms_norm(&h, Some(&cl.post_attn_ln), eps)?;
        let mlp = decoder::expert_mlp_i8(&normed2, &cl.gate, &cl.up, &cl.down)?;
        x = decoder::add_residual(&h, &mlp)?;
    }
    let logits = decoder::norm_and_lm_head(&x, &w.final_norm, &w.embed, cfg.vocab_size, eps)?;
    Ok(logits.data[(logits.rows - 1) * cfg.vocab_size..].to_vec())
}

/// One decode step over a single token embedding `x: [1, hidden]` at absolute
/// `position`, appending to `caches`. Returns the `[vocab]` next-token logits.
fn qwen2_decode_step(
    w: &GotDecodeWeights,
    caches: &mut [Qwen2KvCache],
    x: &Mat,
    position: usize,
) -> FocrResult<Vec<f32>> {
    let cfg = &w.cfg;
    let (qkv_dim, eps) = (cfg.qkv_dim(), cfg.rms_norm_eps);
    let rope = decoder::RopeTable::build(&[position], cfg.head_dim, cfg.rope_theta);
    let mut x = x.clone();
    for (l, cl) in w.layers.iter().enumerate() {
        let normed = nn::rms_norm(&x, Some(&cl.input_ln), eps)?;
        let mut q = nn::linear_int8_dynamic(&normed, &cl.q, Some(&cl.q_b))?;
        let mut k = nn::linear_int8_dynamic(&normed, &cl.k, Some(&cl.k_b))?;
        let v = nn::linear_int8_dynamic(&normed, &cl.v, Some(&cl.v_b))?;
        decoder::apply_rope(&mut q, &rope)?;
        decoder::apply_rope(&mut k, &rope)?;
        caches[l].append(&k.data, &v.data);
        let ctx =
            qwen2_decode_attention(&caches[l], &q.data, cfg.num_attention_heads, cfg.head_dim);
        let ctx = Mat::from_vec(1, qkv_dim, ctx);
        let attn = nn::linear_int8_dynamic(&ctx, &cl.o, None)?;
        let h = decoder::add_residual(&x, &attn)?;
        let normed2 = nn::rms_norm(&h, Some(&cl.post_attn_ln), eps)?;
        let mlp = decoder::expert_mlp_i8(&normed2, &cl.gate, &cl.up, &cl.down)?;
        x = decoder::add_residual(&h, &mlp)?;
    }
    let logits = decoder::norm_and_lm_head(&x, &w.final_norm, &w.embed, cfg.vocab_size, eps)?;
    Ok(logits.data)
}

/// **O(n)-per-token** greedy decode: identical id-stream to [`generate_greedy`] but
/// with a full-causal KV cache instead of re-running prefill each step. The bit-for-bit
/// equality is enforced by reusing [`nn::linear_int8_dynamic`] for every GEMM (the
/// prefill kernel), so the decode never diverges from the certified path.
///
/// # Errors
/// Any prefill/decode-step error, or a missing `model.embed_tokens.weight`.
pub fn generate_greedy_kvcache(
    weights: &Weights,
    cfg: &DecoderConfig,
    inputs_embeds: &Mat,
    max_new: usize,
    eos: u32,
) -> FocrResult<Vec<u32>> {
    let w = GotDecodeWeights::build(weights, cfg)?;
    let n = inputs_embeds.rows;
    let mut caches: Vec<Qwen2KvCache> = (0..cfg.num_hidden_layers)
        .map(|_| Qwen2KvCache::new(cfg.qkv_dim(), n + max_new))
        .collect();
    let last_logits = forward_prefill_seed(&w, inputs_embeds, &mut caches)?;
    let mut ids = Vec::new();
    let mut next = argmax(&last_logits) as u32;
    for _ in 0..max_new {
        ids.push(next);
        if next == eos {
            break;
        }
        let te = decoder::embed_tokens(&w.embed, cfg.vocab_size, cfg.hidden_size, &[next])?;
        // the new token occupies the position after every currently-cached row.
        let position = caches[0].n_kv;
        let logits = qwen2_decode_step(&w, &mut caches, &te, position)?;
        next = argmax(&logits) as u32;
    }
    Ok(ids)
}

/// Argmax over a logit row (first max on ties) — the shared greedy pick.
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn got_config_is_the_censused_shape() {
        let c = DecoderConfig::got_ocr2();
        assert_eq!(c.hidden_size, 1024);
        assert_eq!(c.intermediate_size, 2816);
        assert_eq!(c.num_hidden_layers, 24);
        assert_eq!(c.num_attention_heads, 16);
        assert_eq!(c.head_dim, 64);
        assert_eq!(c.qkv_dim(), 1024);
        assert_eq!(c.vocab_size, 151_860);
        // full-causal scale must be exactly 1/8 (= 1/sqrt(64), what prefill_attention derives).
        assert!(((1.0 / (c.head_dim as f32).sqrt()) - 0.125).abs() < 1e-7);
    }

    /// Read a raw little-endian f32 blob.
    fn read_f32_le(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).expect("oracle blob");
        assert_eq!(bytes.len() % 4, 0, "not a whole f32 count");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            })
            .0
    }

    /// **B5 — the GOT decoder parity gate vs the bit-deterministic torch oracle.**
    /// Env-gated (skip-with-success when the artifacts are absent, per the
    /// model-gated pattern): `FOCR_GOT_MODEL` = the got-ocr2 `.focrq` (int8) or the
    /// raw bf16 `model.safetensors` (f32 reference); `FOCR_ORACLE_HIDDEN0` = the
    /// oracle's post-splice decoder input `[N,1024]`; `FOCR_ORACLE_LOGITS` = the
    /// oracle last-position logits `[vocab]`.
    ///
    /// CERTIFIED (2026-06-30, GOT-OCR2 weights): the **f32 reference** path
    /// (`model.safetensors`) matches the torch oracle to **cos = 1.000000** — every
    /// kernel is numerically exact — and the shipping **int8** path
    /// (`got-ocr2.int8.focrq`) to **cos = 0.9993**, with the greedy next-token
    /// **argmax = 9707 exact on both**.
    #[test]
    fn decoder_matches_torch_oracle() {
        let (Ok(model), Ok(h0), Ok(lg)) = (
            std::env::var("FOCR_GOT_MODEL"),
            std::env::var("FOCR_ORACLE_HIDDEN0"),
            std::env::var("FOCR_ORACLE_LOGITS"),
        ) else {
            return;
        };
        let cfg = DecoderConfig::got_ocr2();
        let weights = Weights::load(std::path::Path::new(&model)).expect("load GOT weights");

        let h0_flat = read_f32_le(&h0);
        let n = h0_flat.len() / cfg.hidden_size;
        assert_eq!(n * cfg.hidden_size, h0_flat.len(), "hidden0 not [N,1024]");
        let inputs = Mat::from_vec(n, cfg.hidden_size, h0_flat);

        let logits = forward_prefill(&weights, &cfg, &inputs).expect("prefill forward");
        assert_eq!(logits.cols, cfg.vocab_size);
        let ours = &logits.data[(logits.rows - 1) * logits.cols..];

        let oracle = read_f32_le(&lg);
        let oracle = &oracle[oracle.len() - cfg.vocab_size..]; // last position if [N,vocab]

        // greedy next-token identity (the load-bearing gate; must hold on int8 too).
        assert_eq!(
            argmax(ours),
            argmax(oracle),
            "next-token argmax diverged from the torch oracle"
        );
        // cosine + max-abs. Oracle floor = 0, so any residual is our numeric/quant
        // error: an int8 build stays high-cos; an f32 build is near bit-exact.
        let dot: f64 = ours
            .iter()
            .zip(oracle)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        let na: f64 = ours
            .iter()
            .map(|&a| f64::from(a) * f64::from(a))
            .sum::<f64>()
            .sqrt();
        let nb: f64 = oracle
            .iter()
            .map(|&b| f64::from(b) * f64::from(b))
            .sum::<f64>()
            .sqrt();
        let cos = dot / (na * nb);
        eprintln!(
            "[B5 parity] argmax={} cos={cos:.6} (oracle argmax={})",
            argmax(ours),
            argmax(oracle)
        );
        assert!(
            cos >= 0.99,
            "logit cosine {cos:.6} < 0.99 — decoder diverged"
        );
    }

    /// **B5/B7 — greedy generation matches the torch oracle's L4 output.** Feeds the
    /// oracle's post-splice `hidden_0` and greedily decodes; the ids must equal the
    /// oracle's greedy prefix `[9707, 38, 1793, 12, …]` (oracle_fixtures.json
    /// `l4_greedy_decode_ids`). Limited to 4 tokens (each step re-runs prefill —
    /// the unoptimized correct path). Env-gated like the parity gate.
    #[test]
    fn greedy_generation_matches_oracle_l4() {
        let (Ok(model), Ok(h0)) = (
            std::env::var("FOCR_GOT_MODEL"),
            std::env::var("FOCR_ORACLE_HIDDEN0"),
        ) else {
            return;
        };
        let cfg = DecoderConfig::got_ocr2();
        let weights = Weights::load(std::path::Path::new(&model)).expect("load GOT weights");
        let h0_flat = read_f32_le(&h0);
        let n = h0_flat.len() / cfg.hidden_size;
        let inputs = Mat::from_vec(n, cfg.hidden_size, h0_flat);

        // eos = <|im_end|> (151645); 4 new tokens is enough to prove the append loop.
        let ids = generate_greedy(&weights, &cfg, &inputs, 4, 151_645).expect("generate");
        eprintln!("[B5 gen] first ids = {ids:?}");
        // int8 build: argmax is exact on this confident prefix (matches the f32 oracle).
        assert_eq!(
            ids,
            vec![9707, 38, 1793, 12],
            "greedy ids diverged from the torch oracle L4"
        );
    }

    /// **B9 — the O(n) KV-cache decode reproduces the oracle L4** (transitively ==
    /// the certified `generate_greedy`, since it's held bit-identical by reusing the
    /// prefill's `linear_int8_dynamic` kernel). Runs 8 tokens — FAST here because the
    /// cache makes each step O(n_kv), unlike the re-prefill path. Env-gated.
    #[test]
    fn kvcache_greedy_matches_oracle_l4() {
        let (Ok(model), Ok(h0)) = (
            std::env::var("FOCR_GOT_MODEL"),
            std::env::var("FOCR_ORACLE_HIDDEN0"),
        ) else {
            return;
        };
        let cfg = DecoderConfig::got_ocr2();
        let weights = Weights::load(std::path::Path::new(&model)).expect("load GOT weights");
        let h0_flat = read_f32_le(&h0);
        let n = h0_flat.len() / cfg.hidden_size;
        let inputs = Mat::from_vec(n, cfg.hidden_size, h0_flat);

        let ids =
            generate_greedy_kvcache(&weights, &cfg, &inputs, 8, 151_645).expect("kvcache gen");
        eprintln!("[B9 kvcache] first ids = {ids:?}");
        // == the torch oracle greedy L4 prefix (oracle_fixtures.json l4_greedy_decode_ids).
        assert_eq!(
            ids,
            vec![9707, 38, 1793, 12, 93495, 17, 13, 15],
            "KV-cache decode diverged from the torch oracle L4"
        );
    }
}
