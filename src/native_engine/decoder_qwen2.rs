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
//! **KV-cache** decode used in production. The seeding prefill stays on the row-parallel
//! `nn::linear_int8_dynamic` (m = N); each single-token decode step (m = 1) instead
//! runs the **n-parallel** `gemv_i8_bias_prequant` — same int8 weights, activations
//! quantized ties-to-even (`quantize_row_i8_te`) to match the prefill — so the decode
//! stays **argmax-exact** to `generate_greedy` (hence the torch oracle L4) while giving
//! every core work at m = 1 (the row-parallel kernel is single-threaded there).

use super::batch_scheduler;
use super::decoder;
use super::nn;
use super::sampler;
use super::tensor::{Mat, QInt8, WeightLayout};
use super::weights::{DType, Weights};
use crate::error::{FocrError, FocrResult};
use rayon::prelude::*;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

// ── GOT decode perf levers (bd-3bom follow-on) — each gated so ONE build measures
//    every config, and the gate on a numerics-changing lever IS its doctrine-#2
//    kill-switch. Tri-state env: "1"/"int8"/"on" force ON, "0"/"f32"/"off" force OFF,
//    unset ⇒ the compiled default. Read ONCE into a process-wide bool. ─────────────

/// `FOCR_GOT_INT8_LMHEAD` — run the GOT `lm_head` as an int8 per-output-channel GEMV
/// (SDOT on Apple Silicon / AVX-512-VNNI·AVX-VNNI·AVX2 on Intel-AMD) instead of the f32
/// matmul over the 151860×1024 tied head, then recover the exact greedy pick with a small
/// f32 top-K refine ([`refine_topk_f32`]). The lm_head is the dominant memory-bound decode
/// stage (reads the ~0.6 GB f32 head per token); int8 cuts that to ~0.15 GB + moves it onto
/// the int8 matmul units (measured **8.2× on the head**, **1.85× on the whole GOT forward**).
///
/// int8-on-`lm_head` is BEYOND doctrine #2's validated set, so it stays a MEASURED
/// kill-switch — but **default ON**, because the top-K refine makes it **byte-identical** to
/// the f32 head (688/688 tokens on page_0107; == the torch oracle L4). `FOCR_GOT_INT8_LMHEAD=0`
/// forces the provably-bit-identical f32 head back. See artifacts/perf/bd-3bom/.
const GOT_INT8_LMHEAD_DEFAULT: bool = true;

/// `FOCR_GOT_SEQ_ATTN` — force the SERIAL per-head decode attention. The parallel path
/// (default) fans the independent heads across the rayon pool; it is bit-identical (each
/// head is a self-contained softmax over disjoint output lanes), so this gate exists only
/// as a measurement/safety switch, not a correctness one.
const GOT_PARALLEL_ATTN_DEFAULT: bool = true;

fn env_tristate(var: &str, default: bool) -> bool {
    match std::env::var(var)
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("1" | "int8" | "on" | "true" | "yes") => true,
        Some("0" | "f32" | "off" | "false" | "no") => false,
        _ => default,
    }
}

fn got_int8_lmhead_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| env_tristate("FOCR_GOT_INT8_LMHEAD", GOT_INT8_LMHEAD_DEFAULT))
}

fn got_parallel_attn_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    // FOCR_GOT_SEQ_ATTN=1 forces serial ⇒ parallel is the inverse.
    *FLAG.get_or_init(|| !env_tristate("FOCR_GOT_SEQ_ATTN", !GOT_PARALLEL_ATTN_DEFAULT))
}

/// The GOT `lm_head` weight, resolved ONCE per generation to either the f32 pre-transposed
/// `[hidden, vocab]` matrix (bit-identical to the certified path) or the int8 per-output-
/// channel quantization of the SAME tied head (the [`got_int8_lmhead_enabled`] lever).
enum LmHead {
    /// `[hidden, vocab]` row-major, matmul-ready (see [`decoder::norm_and_lm_head_pretransposed`]).
    F32(Mat),
    /// `[vocab, hidden]` per-output-channel int8 (the native checkpoint layout, fed to the
    /// n-parallel `gemv_i8` — argmax-exact to f32 on the oracle L4 / page_0107 CER).
    Int8(QInt8),
}

/// Decode-step wall-clock accumulators (ns), summed across the whole generate loop.
/// Read + printed only when `FOCR_TIMING` is set (perf bring-up); the `fetch_add`s
/// are `Relaxed` and off the hot numeric path, so they cost nothing measurable.
static DECODE_ATTN_NS: AtomicU64 = AtomicU64::new(0);
static DECODE_GEMV_NS: AtomicU64 = AtomicU64::new(0);
static DECODE_LMHEAD_NS: AtomicU64 = AtomicU64::new(0);

/// Config of a dense (Qwen2/Llama-style) decoder — the parameters the shared leaf
/// kernels need, so one driver serves GOT-OCR2 (and later SmolVLM2/OneChart).
/// The dense-decoder FAMILY — which per-layer op set / tensor naming the
/// engine runs. Additive: every certified config stays `QwenLlama` and takes
/// byte-identical branches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderFamily {
    /// RMSNorm + RoPE + gated SwiGLU (`gate/up/down`) — GOT-OCR2, SmolVLM2.
    QwenLlama,
    /// LayerNorm(+bias) + LEARNED absolute positions (offset table, no RoPE),
    /// plus a plain ReLU `fc1`/`fc2` MLP with all linears biased — OneChart's
    /// OPT-125M (census docs/zoo/onechart-spec.md §4; D4, bd-3jo6.4.4).
    Opt,
}

#[derive(Debug, Clone, Copy)]
pub struct DecoderConfig {
    /// The op-set / naming family ([`DecoderFamily`]).
    pub family: DecoderFamily,
    /// `Some((table_name, offset))` = learned absolute positions: row
    /// `i + offset` of the named `[rows, hidden]` table is ADDED to token `i`'s
    /// embedding before layer 0 (OPT: offset 2). `None` = RoPE.
    pub embed_positions: Option<(&'static str, usize)>,
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
    /// HF-builtin **global** no-repeat-n-gram size for greedy generation (GOT 20,
    /// hard-coded upstream in `chat()` — spec §12 OQ-8); `0` disables the guard
    /// (bd-ff4i: unguarded greedy repetition-runs on some real pages).
    pub no_repeat_ngram_size: usize,
    /// GQA key/value head count (A7, bd-3jo6.1.7): `== num_attention_heads` is
    /// plain MHA (GOT); `< num_attention_heads` shares each kv head across
    /// `kv_group()` query heads (SmolVLM2-500M: 15 q heads over 5 kv heads).
    /// Must divide `num_attention_heads` evenly.
    pub num_key_value_heads: usize,
    /// Decoder-layer tensor-name prefix INCLUDING the trailing dot (GOT
    /// `model.layers.`, SmolVLM2 `model.text_model.layers.`) — pinned against
    /// the `ModelArch` descriptor by a unit test so config and registry cannot
    /// drift (C5, bd-3jo6.3.5).
    pub layers_prefix: &'static str,
    /// The token-embedding table's tensor name.
    pub embed_tokens: &'static str,
    /// The final (pre-lm_head) RMSNorm weight's tensor name.
    pub final_norm: &'static str,
    /// `Some(name)` = the UNTIED lm_head tensor (SmolVLM2 `lm_head.weight`);
    /// `None` = tied to [`Self::embed_tokens`] (GOT). Drives BOTH lm_head
    /// resolution paths (f32 pretranspose AND the int8+refine lever) and —
    /// load-bearing — the top-K f32 refine's exact-recompute source.
    pub lm_head: Option<&'static str>,
}

impl DecoderConfig {
    /// The GOT-OCR2.0 (Qwen2-0.5B) decoder configuration (`config.json`, spec §4).
    #[must_use]
    pub fn got_ocr2() -> Self {
        Self {
            family: DecoderFamily::QwenLlama,
            embed_positions: None,
            hidden_size: 1024,
            intermediate_size: 2816,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            head_dim: 64,
            vocab_size: 151_860,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            attn_qkv_bias: true,
            no_repeat_ngram_size: 20,
            num_key_value_heads: 16,
            layers_prefix: "model.layers.",
            embed_tokens: "model.embed_tokens.weight",
            final_norm: "model.norm.weight",
            lm_head: None,
        }
    }

    /// The SmolVLM2-500M (SmolLM2-360M) decoder configuration
    /// (docs/zoo/smolvlm2-spec.md §4, census-pinned from the real config.json;
    /// C5, bd-3jo6.3.5): GQA 15 q heads over 5 kv heads, NO qkv bias, UNTIED
    /// lm_head, no upstream repetition guard.
    #[must_use]
    pub fn smolvlm2() -> Self {
        Self {
            family: DecoderFamily::QwenLlama,
            embed_positions: None,
            hidden_size: 960,
            intermediate_size: 2560,
            num_hidden_layers: 32,
            num_attention_heads: 15,
            head_dim: 64,
            vocab_size: 49_280,
            rope_theta: 100_000.0,
            rms_norm_eps: 1e-5,
            attn_qkv_bias: false,
            no_repeat_ngram_size: 0,
            num_key_value_heads: 5,
            layers_prefix: "model.text_model.layers.",
            embed_tokens: "model.text_model.embed_tokens.weight",
            final_norm: "model.text_model.norm.weight",
            lm_head: Some("lm_head.weight"),
        }
    }

    /// The OneChart (OPT-125M) decoder configuration (census
    /// docs/zoo/onechart-spec.md §4, pinned from the real config.json; D4):
    /// pre-LN LayerNorm+bias, learned offset-2 positions (NO RoPE), plain
    /// ReLU fc1/fc2 with ALL linears biased, MHA 12/12, TIED head, eos 2,
    /// no upstream repetition guard.
    #[must_use]
    pub fn onechart() -> Self {
        Self {
            family: DecoderFamily::Opt,
            embed_positions: Some(("model.decoder.embed_positions.weight", 2)),
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            head_dim: 64,
            vocab_size: 50_269,
            rope_theta: 10_000.0, // unused (learned positions)
            rms_norm_eps: 1e-5,   // nn.LayerNorm default (HF OPT)
            attn_qkv_bias: true,
            no_repeat_ngram_size: 0,
            num_key_value_heads: 12,
            layers_prefix: "model.decoder.layers.",
            embed_tokens: "model.decoder.embed_tokens.weight",
            final_norm: "model.decoder.final_layer_norm.weight",
            lm_head: None,
        }
    }

    /// The q / o projection width (`num_attention_heads * head_dim`). Equals
    /// `hidden_size` for GOT.
    #[must_use]
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// The k / v projection width (`num_key_value_heads * head_dim`); equals
    /// [`Self::q_dim`] for MHA, smaller under GQA.
    #[must_use]
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Query heads per kv head (`1` = MHA; SmolVLM2 = 3). Query head `h` reads
    /// kv head `h / kv_group()`.
    #[must_use]
    pub fn kv_group(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Enforce the documented GQA invariant (`num_key_value_heads` divides
    /// `num_attention_heads` evenly) — otherwise `broadcast_kv` leaves the
    /// trailing query lanes zeroed and `decode_attn_head` reads out of a
    /// too-short lane, both SILENT wrong-math paths (fresh-eyes fix). Called
    /// at both forward entries; the registered configs pass by construction.
    fn validate_gqa(&self) -> FocrResult<()> {
        if self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err(FocrError::Other(anyhow::anyhow!(
                "DecoderConfig: num_key_value_heads {} must divide num_attention_heads {} evenly",
                self.num_key_value_heads,
                self.num_attention_heads
            )));
        }
        Ok(())
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
    if cfg.family == DecoderFamily::Opt {
        return opt_layer(weights, x, layer, cfg);
    }
    let p = format!("{}{layer}", cfg.layers_prefix);
    assert_no_qk_norm(weights, &p)?;
    let eps = cfg.rms_norm_eps;
    let (hidden, inter) = (cfg.hidden_size, cfg.intermediate_size);
    let (q_dim, kv_dim) = (cfg.q_dim(), cfg.kv_dim());

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
        q_dim,
        q_b.as_deref(),
    )?;
    let mut k = linear_auto(
        weights,
        &normed,
        &format!("{p}.self_attn.k_proj.weight"),
        hidden,
        kv_dim,
        k_b.as_deref(),
    )?;
    let v = linear_auto(
        weights,
        &normed,
        &format!("{p}.self_attn.v_proj.weight"),
        hidden,
        kv_dim,
        v_b.as_deref(),
    )?;

    decoder::apply_rope(&mut q, rope)?;
    decoder::apply_rope(&mut k, rope)?;
    let ctx = prefill_attention_gqa(&q, &k, &v, cfg)?;
    let attn = linear_auto(
        weights,
        &ctx,
        &format!("{p}.self_attn.o_proj.weight"),
        q_dim,
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

/// One OPT (OneChart) decoder layer (census §4, HF `modeling_opt.py` with
/// `do_layer_norm_before=true`): `h += out_proj(attn(LN1(h)))` then
/// `h += fc2(relu(fc1(LN2(h))))` — LayerNorm WITH bias (the per-layer
/// pre-MLP norm is *named* `final_layer_norm`, the census naming hazard),
/// every linear biased, NO RoPE (learned positions were added before layer
/// 0), full-causal MHA. The q pre-scaling stays inside our attention kernel
/// (OQ-D5: one placement, mathematically equal to HF's q-side scale; the
/// armed oracle cert measures the residual drift).
fn opt_layer(weights: &Weights, x: &Mat, layer: usize, cfg: &DecoderConfig) -> FocrResult<Mat> {
    let p = format!("{}{layer}", cfg.layers_prefix);
    let eps = cfg.rms_norm_eps;
    let (hidden, inter) = (cfg.hidden_size, cfg.intermediate_size);
    let q_dim = cfg.q_dim();

    // ── attention (pre-LN, all linears biased) ──────────────────────────────
    let ln1_w = weights.vec(&format!("{p}.self_attn_layer_norm.weight"))?;
    let ln1_b = weights.vec(&format!("{p}.self_attn_layer_norm.bias"))?;
    let normed = nn::layer_norm(x, Some(&ln1_w), Some(&ln1_b), eps)?;

    let q_b = weights.vec(&format!("{p}.self_attn.q_proj.bias"))?;
    let k_b = weights.vec(&format!("{p}.self_attn.k_proj.bias"))?;
    let v_b = weights.vec(&format!("{p}.self_attn.v_proj.bias"))?;
    let q = linear_auto(
        weights,
        &normed,
        &format!("{p}.self_attn.q_proj.weight"),
        hidden,
        q_dim,
        Some(&q_b),
    )?;
    let k = linear_auto(
        weights,
        &normed,
        &format!("{p}.self_attn.k_proj.weight"),
        hidden,
        q_dim,
        Some(&k_b),
    )?;
    let v = linear_auto(
        weights,
        &normed,
        &format!("{p}.self_attn.v_proj.weight"),
        hidden,
        q_dim,
        Some(&v_b),
    )?;
    // NO RoPE — OPT positions are the learned table added before layer 0.
    let ctx = prefill_attention_gqa(&q, &k, &v, cfg)?;
    let out_b = weights.vec(&format!("{p}.self_attn.out_proj.bias"))?;
    let attn = linear_auto(
        weights,
        &ctx,
        &format!("{p}.self_attn.out_proj.weight"),
        q_dim,
        hidden,
        Some(&out_b),
    )?;
    let h = decoder::add_residual(x, &attn)?;

    // ── plain ReLU MLP (fc1/fc2, biased; pre-LN named `final_layer_norm`) ───
    let ln2_w = weights.vec(&format!("{p}.final_layer_norm.weight"))?;
    let ln2_b = weights.vec(&format!("{p}.final_layer_norm.bias"))?;
    let normed2 = nn::layer_norm(&h, Some(&ln2_w), Some(&ln2_b), eps)?;
    let fc1_b = weights.vec(&format!("{p}.fc1.bias"))?;
    let mut m = linear_auto(
        weights,
        &normed2,
        &format!("{p}.fc1.weight"),
        hidden,
        inter,
        Some(&fc1_b),
    )?;
    nn::relu(&mut m);
    let fc2_b = weights.vec(&format!("{p}.fc2.bias"))?;
    let mlp = linear_auto(
        weights,
        &m,
        &format!("{p}.fc2.weight"),
        inter,
        hidden,
        Some(&fc2_b),
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
    cfg.validate_gqa()?;
    if inputs_embeds.cols != cfg.hidden_size {
        return Err(FocrError::FormatMismatch(format!(
            "decoder_qwen2: inputs_embeds cols {} != hidden {}",
            inputs_embeds.cols, cfg.hidden_size
        )));
    }
    let normed = prefill_final_hidden(weights, cfg, inputs_embeds)?;
    let embed = match cfg.lm_head {
        Some(name) => weights.mat(name)?,
        None => weights.mat(cfg.embed_tokens)?,
    };
    decoder::linear_no_bias(&normed, &embed.data, cfg.hidden_size, cfg.vocab_size)
}

/// The prefill through all layers + the FINAL norm, WITHOUT the lm_head —
/// the post-final-norm hidden stream `[seq, hidden]` (the same values that
/// feed `lm_head`, and what OneChart's `num_decoder` taps at the `<Number>`
/// step — census §8 / bd-3jo6.4.5). [`forward_prefill`] is exactly this plus
/// the head projection.
///
/// # Errors
/// As [`forward_prefill`].
pub fn prefill_final_hidden(
    weights: &Weights,
    cfg: &DecoderConfig,
    inputs_embeds: &Mat,
) -> FocrResult<Mat> {
    cfg.validate_gqa()?;
    if inputs_embeds.cols != cfg.hidden_size {
        return Err(FocrError::FormatMismatch(format!(
            "decoder_qwen2: inputs_embeds cols {} != hidden {}",
            inputs_embeds.cols, cfg.hidden_size
        )));
    }
    let positions: Vec<usize> = (0..inputs_embeds.rows).collect();
    let rope = decoder::RopeTable::build(&positions, cfg.head_dim, cfg.rope_theta);

    let mut x = inputs_embeds.clone();
    // Learned absolute positions (OPT): row `i + offset` of the table is
    // added to token i's embedding before layer 0 (census §4: offset 2, ids
    // 0..L-1 for our unpadded single sequence — OQ-D6).
    if let Some((table, offset)) = cfg.embed_positions {
        let pos = weights.mat(table)?;
        if x.rows + offset > pos.rows {
            return Err(FocrError::FormatMismatch(format!(
                "decoder: seq {} + offset {offset} exceeds the {} learned positions (OQ-D7)",
                x.rows, pos.rows
            )));
        }
        for i in 0..x.rows {
            let row = x.row_mut(i);
            let p = &pos.data[(i + offset) * cfg.hidden_size..(i + offset + 1) * cfg.hidden_size];
            for (a, b) in row.iter_mut().zip(p) {
                *a += b;
            }
        }
    }
    for layer in 0..cfg.num_hidden_layers {
        x = qwen2_layer(weights, &x, layer, &rope, cfg)?;
    }

    let final_norm = weights.vec(cfg.final_norm)?;
    if cfg.family == DecoderFamily::Opt {
        // Model-level LayerNorm WITH bias.
        let fb = weights.vec(&cfg.final_norm.replace(".weight", ".bias"))?;
        return nn::layer_norm(&x, Some(&final_norm), Some(&fb), cfg.rms_norm_eps);
    }
    nn::rms_norm(&x, Some(&final_norm), cfg.rms_norm_eps)
}

/// Greedy (argmax, temperature-0) autoregressive decode from `inputs_embeds`,
/// generating up to `max_new` tokens and stopping at `eos`. Returns the generated
/// id-stream (excluding the prompt). Each pick runs through [`argmax_no_repeat`]
/// (`cfg.no_repeat_ngram_size`, the upstream global ban — bd-ff4i).
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
    let embed = weights.mat(cfg.embed_tokens)?;
    let (vocab, hidden) = (embed.rows, embed.cols);
    let mut data = inputs_embeds.data.clone();
    let mut ids = Vec::new();
    for _ in 0..max_new {
        crate::cancel_checkpoint()?;
        let rows = data.len() / hidden;
        let cur = Mat::from_vec(rows, hidden, std::mem::take(&mut data));
        let logits = forward_prefill(weights, cfg, &cur)?;
        // Slice by the LOGITS' own stride (== cfg.vocab_size, set by the
        // lm_head) — `vocab` above comes from embed.rows, which an untied
        // checkpoint could legitimately size differently (fresh-eyes fix).
        let last = &logits.data[(logits.rows - 1) * logits.cols..];
        let next = argmax_no_repeat(last, &ids, cfg.no_repeat_ngram_size) as u32;
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
/// token-major `[n_kv, kv_dim]`, grown by one row per decode step. NOT R-SWA — GOT
/// Qwen2 attends the whole prefix (no window, no eviction, f32 KV).
struct Qwen2KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    n_kv: usize,
    /// Row stride: `num_key_value_heads · head_dim` — the GQA-NATIVE width
    /// (`== q_dim` for MHA/GOT). The decode per-head reader maps query heads
    /// onto these lanes.
    kv_dim: usize,
}

impl Qwen2KvCache {
    fn new(kv_dim: usize, max_positions: usize) -> Self {
        Self {
            k: Vec::with_capacity(max_positions * kv_dim),
            v: Vec::with_capacity(max_positions * kv_dim),
            n_kv: 0,
            kv_dim,
        }
    }
    /// Seed all `N` prefill rows at once (`k_all`/`v_all` are `[N, kv_dim]`).
    fn seed(&mut self, k_all: &[f32], v_all: &[f32]) {
        self.k.extend_from_slice(k_all);
        self.v.extend_from_slice(v_all);
        self.n_kv += k_all.len() / self.kv_dim;
    }
    /// Append one decode step's k/v row (each `[kv_dim]`).
    fn append(&mut self, k_row: &[f32], v_row: &[f32]) {
        self.k.extend_from_slice(k_row);
        self.v.extend_from_slice(v_row);
        self.n_kv += 1;
    }
}

/// Full-causal m=1 attention: the single new query (`q_row`, `[q_dim]`) attends
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
    kv_group: usize,
) -> Vec<f32> {
    let dim = num_heads * head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; dim];
    if got_parallel_attn_enabled() {
        // Each head is a self-contained softmax writing a DISJOINT `[head_dim]` output
        // lane, so fanning the heads across the rayon pool is bit-identical (no cross-head
        // dependence, per-head accumulation order unchanged). `par_chunks_mut(head_dim)`
        // hands head `h` its own output slice + its own scratch.
        out.par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(h, oh)| decode_attn_head(cache, q_row, h, head_dim, kv_group, scale, oh));
    } else {
        for h in 0..num_heads {
            let oh = &mut out[h * head_dim..(h + 1) * head_dim];
            decode_attn_head(cache, q_row, h, head_dim, kv_group, scale, oh);
        }
    }
    out
}

/// One attention head's decode: `softmax(scale · q_h·Kᵀ) · V`, writing `[head_dim]` into
/// `oh`. Reads the token-major cache directly at its NATIVE `kv_dim` stride —
/// under GQA (`kv_group > 1`) query head `h` reads shared kv head
/// `h / kv_group` (A7); `kv_group == 1` reads head `h` verbatim (the certified
/// MHA path, index arithmetic unchanged: `kv_dim == num_heads·head_dim`). The
/// only per-head temp is `[n_kv]` scores. Identical math whether called
/// serially or from the rayon fan-out above.
#[inline]
fn decode_attn_head(
    cache: &Qwen2KvCache,
    q_row: &[f32],
    h: usize,
    head_dim: usize,
    kv_group: usize,
    scale: f32,
    oh: &mut [f32],
) {
    let n_kv = cache.n_kv;
    let kv_dim = cache.kv_dim;
    let kv_lane = (h / kv_group) * head_dim;
    let qh = &q_row[h * head_dim..h * head_dim + head_dim];
    let mut scores = vec![0.0f32; n_kv];
    // scores[r] = scale · (q_h · k[r, kv head h/kv_group]); track the max for a
    // stable softmax.
    let mut smax = f32::NEG_INFINITY;
    for (r, s) in scores.iter_mut().enumerate() {
        let base = r * kv_dim + kv_lane;
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
    // out_h = Σ_r softmax[r] · v[r, kv head h/kv_group].
    for (r, &s) in scores.iter().enumerate() {
        let w = s * inv;
        let base = r * kv_dim + kv_lane;
        let vh = &cache.v[base..base + head_dim];
        for (o, &vv) in oh.iter_mut().zip(vh) {
            *o += w * vv;
        }
    }
}

/// One layer's decode weights, loaded ONCE (int8 GEMMs read verbatim from the
/// `.focrq` `QInt8PerChan` records, f32 norms/biases) so no weight is re-read per token.
struct GotLayerW {
    input_ln: Vec<f32>,
    post_attn_ln: Vec<f32>,
    /// Fused q|k|v projection `[q_dim + 2·kv_dim, hidden]` (one int8 panel) so the decode
    /// quantizes the normed row ONCE and runs ONE n-parallel GEMV for all three.
    qkv: QInt8,
    /// Fused q|k|v bias, `None` for a bias-free arch (SmolVLM2; C5/D4) —
    /// [`decoder::gemv_i8_bias_prequant`] and [`nn::linear_int8_dynamic`] both
    /// skip the add on `None`.
    qkv_bias: Option<Vec<f32>>,
    o: QInt8,
    /// out_proj bias — `Some` only for the Opt family (every linear biased).
    o_bias: Option<Vec<f32>>,
    /// LayerNorm biases `(input_ln.bias, pre_mlp_ln.bias)` — `Some` selects
    /// LayerNorm (Opt); `None` selects RMSNorm (QwenLlama). Bias presence IS
    /// the family's norm choice, so the two can never drift apart.
    ln_bias: Option<(Vec<f32>, Vec<f32>)>,
    mlp: MlpW,
}

/// The per-layer MLP weights — gated SwiGLU (QwenLlama) or the plain biased
/// ReLU pair (Opt).
enum MlpW {
    SwiGlu {
        gate: QInt8,
        up: QInt8,
        down: QInt8,
    },
    ReluFc {
        fc1: QInt8,
        fc1_b: Vec<f32>,
        fc2: QInt8,
        fc2_b: Vec<f32>,
    },
}

/// Add the learned absolute-position rows (Opt) to `x` in place: token `i`
/// (absolute position `start + i`) gains table row `start + i + offset`
/// (census §4: offset 2; OQ-D7 enforces the table bound loudly).
fn add_learned_positions(x: &mut Mat, w: &GotDecodeWeights, start: usize) -> FocrResult<()> {
    let Some((table, offset)) = &w.embed_positions else {
        return Ok(());
    };
    let hidden = w.cfg.hidden_size;
    let rows = table.len() / hidden;
    if start + x.rows + offset > rows {
        return Err(FocrError::FormatMismatch(format!(
            "decoder: position {} + offset {offset} exceeds the {rows}-row learned table (OQ-D7)",
            start + x.rows
        )));
    }
    for i in 0..x.rows {
        let p = (start + i + offset) * hidden;
        let row = x.row_mut(i);
        for (a, b) in row.iter_mut().zip(&table[p..p + hidden]) {
            *a += b;
        }
    }
    Ok(())
}

/// RMSNorm (bias `None`) or LayerNorm (bias `Some`) — the decode-path family
/// norm switch (mirrors [`opt_layer`] vs the Qwen layer).
fn family_norm(x: &Mat, w: &[f32], b: Option<&[f32]>, eps: f32) -> FocrResult<Mat> {
    match b {
        Some(b) => nn::layer_norm(x, Some(w), Some(b), eps),
        None => nn::rms_norm(x, Some(w), eps),
    }
}

/// Concatenate the row-major q/k/v `[d, h]` int8 panels + scales into one
/// `[3·d, h]` [`QInt8`] — bit-identical per-output-channel to the three separate
/// GEMMs (each output channel keeps its own scale), but one dispatch.
fn concat_qkv(q: &QInt8, k: &QInt8, v: &QInt8) -> QInt8 {
    // Panels may be UNEQUAL under GQA (q: q_dim rows, k/v: kv_dim rows each).
    let n = q.n + k.n + v.n;
    let mut w = Vec::with_capacity(q.w.len() + k.w.len() + v.w.len());
    w.extend_from_slice(&q.w);
    w.extend_from_slice(&k.w);
    w.extend_from_slice(&v.w);
    let mut scales = Vec::with_capacity(n);
    scales.extend_from_slice(&q.scales);
    scales.extend_from_slice(&k.scales);
    scales.extend_from_slice(&v.scales);
    // Offline SMMLA panels (bd-2mo.3) concatenate exactly like rows when each
    // segment's row count is even (true for every registered q_dim/kv_dim, GQA
    // included: 960/320/320, 1024/1024/1024, 768/768/768), so the concatenated
    // stream IS the panel pack of the fused matrix.
    if q.layout == WeightLayout::SmmlaPanels
        && k.layout == WeightLayout::SmmlaPanels
        && v.layout == WeightLayout::SmmlaPanels
        && q.n.is_multiple_of(2)
        && k.n.is_multiple_of(2)
    {
        return QInt8::new_smmla_panels(w, scales, n, q.k);
    }
    debug_assert!(
        q.layout == WeightLayout::RowMajor
            && k.layout == WeightLayout::RowMajor
            && v.layout == WeightLayout::RowMajor,
        "concat_qkv: mixed or odd-row packed layouts are unreachable from the loader"
    );
    QInt8::new(w, scales, n, q.k)
}

/// The whole GOT decoder's decode-time weights (pre-loaded once for a generation).
struct GotDecodeWeights {
    layers: Vec<GotLayerW>,
    final_norm: Vec<f32>,
    /// Model-level LayerNorm bias — `Some` for the Opt family only.
    final_norm_bias: Option<Vec<f32>>,
    /// Learned absolute positions `([rows, hidden] table, offset)` — Opt.
    embed_positions: Option<(Vec<f32>, usize)>,
    /// `embed_tokens` `[vocab, hidden]`, row-major — the per-token embed lookup
    /// (also the lm_head SOURCE when the arch ties them; see `untied_head`).
    embed: Vec<f32>,
    /// The UNTIED lm_head `[vocab, hidden]` when `cfg.lm_head` names one
    /// (SmolVLM2); `None` = tied (the head reads `embed`). LOAD-BEARING for the
    /// int8 lever: the top-K f32 refine must recompute against THIS matrix —
    /// refining against `embed` on an untied arch silently corrupts the argmax.
    untied_head: Option<Vec<f32>>,
    /// The SAME weights transposed to `[hidden, vocab]` ONCE at build, matmul-ready for
    /// the decode-step `lm_head`. Either the f32 pre-transposed `[hidden, vocab]` head
    /// (bit-identical to the certified path — never re-transposes the ~0.6 GB matrix per
    /// token, which was 95% of decode wall-clock) or its int8 quantization
    /// ([`got_int8_lmhead_enabled`]). Built ONCE. See [`got_lm_head`].
    lm_head: LmHead,
    cfg: DecoderConfig,
}

impl GotDecodeWeights {
    /// The lm_head SOURCE matrix `[vocab, hidden]`: the arch's untied head when
    /// it stores one, else the tied embed table. The int8 lever's exact-f32
    /// refine MUST read this (never `embed` directly).
    fn head_matrix(&self) -> &[f32] {
        self.untied_head.as_deref().unwrap_or(&self.embed)
    }

    fn build(weights: &Weights, cfg: &DecoderConfig) -> FocrResult<Self> {
        cfg.validate_gqa()?;
        let (hidden, inter) = (cfg.hidden_size, cfg.intermediate_size);
        let (q_dim, kv_dim) = (cfg.q_dim(), cfg.kv_dim());
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("{}{l}", cfg.layers_prefix);
            let q =
                decoder::quant_oc_loaded(weights, &format!("{p}.self_attn.q_proj.weight"), q_dim)?;
            let k =
                decoder::quant_oc_loaded(weights, &format!("{p}.self_attn.k_proj.weight"), kv_dim)?;
            let v =
                decoder::quant_oc_loaded(weights, &format!("{p}.self_attn.v_proj.weight"), kv_dim)?;
            let qkv_bias = if cfg.attn_qkv_bias {
                let mut b = weights.vec(&format!("{p}.self_attn.q_proj.bias"))?;
                b.extend(weights.vec(&format!("{p}.self_attn.k_proj.bias"))?);
                b.extend(weights.vec(&format!("{p}.self_attn.v_proj.bias"))?);
                Some(b)
            } else {
                None
            };
            if cfg.family == DecoderFamily::Opt {
                layers.push(GotLayerW {
                    input_ln: weights.vec(&format!("{p}.self_attn_layer_norm.weight"))?,
                    post_attn_ln: weights.vec(&format!("{p}.final_layer_norm.weight"))?,
                    qkv: concat_qkv(&q, &k, &v),
                    qkv_bias,
                    o: decoder::quant_oc_loaded(
                        weights,
                        &format!("{p}.self_attn.out_proj.weight"),
                        hidden,
                    )?,
                    o_bias: Some(weights.vec(&format!("{p}.self_attn.out_proj.bias"))?),
                    ln_bias: Some((
                        weights.vec(&format!("{p}.self_attn_layer_norm.bias"))?,
                        weights.vec(&format!("{p}.final_layer_norm.bias"))?,
                    )),
                    mlp: MlpW::ReluFc {
                        fc1: decoder::quant_oc_loaded(weights, &format!("{p}.fc1.weight"), inter)?,
                        fc1_b: weights.vec(&format!("{p}.fc1.bias"))?,
                        fc2: decoder::quant_oc_loaded(weights, &format!("{p}.fc2.weight"), hidden)?,
                        fc2_b: weights.vec(&format!("{p}.fc2.bias"))?,
                    },
                });
                continue;
            }
            layers.push(GotLayerW {
                input_ln: weights.vec(&format!("{p}.input_layernorm.weight"))?,
                post_attn_ln: weights.vec(&format!("{p}.post_attention_layernorm.weight"))?,
                qkv: concat_qkv(&q, &k, &v),
                qkv_bias,
                o: decoder::quant_oc_loaded(
                    weights,
                    &format!("{p}.self_attn.o_proj.weight"),
                    hidden,
                )?,
                o_bias: None,
                ln_bias: None,
                mlp: MlpW::SwiGlu {
                    gate: decoder::quant_oc_loaded(
                        weights,
                        &format!("{p}.mlp.gate_proj.weight"),
                        inter,
                    )?,
                    up: decoder::quant_oc_loaded(
                        weights,
                        &format!("{p}.mlp.up_proj.weight"),
                        inter,
                    )?,
                    down: decoder::quant_oc_loaded(
                        weights,
                        &format!("{p}.mlp.down_proj.weight"),
                        hidden,
                    )?,
                },
            });
        }
        let embed = weights.mat(cfg.embed_tokens)?.data;
        // UNTIED head (SmolVLM2): its own [vocab, hidden] tensor; tied (GOT):
        // alias the embed table (never duplicate GOT's ~0.6 GB).
        let untied_head = match cfg.lm_head {
            Some(name) => Some(weights.mat(name)?.data),
            None => None,
        };
        let final_norm_bias = if cfg.family == DecoderFamily::Opt {
            Some(weights.vec(&cfg.final_norm.replace(".weight", ".bias"))?)
        } else {
            None
        };
        let embed_positions = match cfg.embed_positions {
            Some((name, off)) => Some((weights.mat(name)?.data, off)),
            None => None,
        };
        let (vocab, hidden) = (cfg.vocab_size, cfg.hidden_size);
        // Resolve the lm_head ONCE. int8 lever: quantize the tied `[vocab, hidden]` head to
        // per-output-channel int8 in its NATIVE layout (fed to the n-parallel `gemv_i8` —
        // SDOT on aarch64, VNNI on x86). f32 path: transpose `[vocab, hidden]` -> the
        // matmul-ready `[hidden, vocab]` ONCE (the SAME transpose `linear_no_bias` did per
        // call), so the decode-step head is a plain matmul with no per-token ~0.6 GB
        // re-transpose (which was 95% of decode wall-clock). Only ONE is built, so int8 also
        // halves the head's resident memory (no f32 `[hidden, vocab]` copy).
        let head_src: &[f32] = untied_head.as_deref().unwrap_or(&embed);
        // Lever policy (doctrine #2 / OQ-6): the int8+refine head is measured
        // byte-identical on GOT (tied) AND on the untied SmolVLM2 head
        // (2026-07-08 cert: paired describe outputs byte-identical f32-head vs
        // int8+refine, armed L5 VQA green under the lever, full describe/VQA
        // e2e suite pass; lm_head 6.99 -> 1.10 ms/tok, decode +40%). Both arms
        // now default ON; FOCR_GOT_INT8_LMHEAD=0 is the kill-switch back to
        // the f32 pretransposed head (not OnceLock-cached here, build runs
        // once per generation).
        let int8_head = if cfg.lm_head.is_some() {
            env_tristate("FOCR_GOT_INT8_LMHEAD", true)
        } else {
            got_int8_lmhead_enabled()
        };
        let lm_head = if int8_head {
            LmHead::Int8(nn::quantize_int8(head_src, vocab, hidden))
        } else {
            let mut wt = vec![0.0f32; head_src.len()];
            for o in 0..vocab {
                let src = &head_src[o * hidden..(o + 1) * hidden];
                for (i, &val) in src.iter().enumerate() {
                    wt[i * vocab + o] = val;
                }
            }
            LmHead::F32(Mat::from_vec(hidden, vocab, wt))
        };
        Ok(Self {
            layers,
            final_norm: weights.vec(cfg.final_norm)?,
            final_norm_bias,
            embed_positions,
            embed,
            untied_head,
            lm_head,
            cfg: *cfg,
        })
    }
}

/// How many of the int8-approx head logits get recomputed in exact f32 before the greedy
/// argmax (the [`refine_topk_f32`] near-lossless pass). 256 ≫ any realistic gap between the
/// true best token's int8 rank and rank 1, so the refined argmax matches the f32 head.
const GOT_LMHEAD_REFINE_K: usize = 256;

/// The GOT `lm_head` over a SINGLE hidden row `x_row` (`[1, hidden]`): the final
/// `rms_norm` then the head projection, dispatched to the f32 pre-transposed matmul
/// (bit-identical to the certified path) or the int8 per-channel `gemv_i8`
/// ([`got_int8_lmhead_enabled`]). Returns the `[vocab]` logits. Both the seeding prefill
/// (last position only — the rest are never argmaxed) and every decode step funnel through
/// here, so a whole generation uses ONE lm_head numeric path.
///
/// The int8 path is made **near-lossless** by [`refine_topk_f32`]: the fast int8 GEMV picks
/// the top candidates, then those are recomputed in exact f32 so the greedy pick matches the
/// f32 head (the argmax lives in the int8 top-K with overwhelming probability).
fn got_lm_head(w: &GotDecodeWeights, x_row: &Mat, eps: f32) -> FocrResult<Vec<f32>> {
    if let Some(fb) = &w.final_norm_bias {
        // Opt: model-level LayerNorm WITH bias, then the (pretransposed) head.
        let normed = nn::layer_norm(x_row, Some(&w.final_norm), Some(fb), eps)?;
        return match &w.lm_head {
            LmHead::F32(wt) => Ok(nn::matmul(&normed, wt)?.data),
            LmHead::Int8(q) => {
                let (xq, a) = decoder::quantize_row_i8_te(&normed.data);
                let mut logits = decoder::gemv_i8_bias_prequant(&xq, a, q, None);
                refine_topk_f32(
                    &mut logits,
                    &normed.data,
                    w.head_matrix(),
                    w.cfg.hidden_size,
                );
                Ok(logits)
            }
        };
    }
    match &w.lm_head {
        LmHead::F32(wt) => {
            Ok(decoder::norm_and_lm_head_pretransposed(x_row, &w.final_norm, wt, eps)?.data)
        }
        LmHead::Int8(q) => {
            let normed = nn::rms_norm(x_row, Some(&w.final_norm), eps)?;
            let (xq, a) = decoder::quantize_row_i8_te(&normed.data);
            let mut logits = decoder::gemv_i8_bias_prequant(&xq, a, q, None);
            refine_topk_f32(
                &mut logits,
                &normed.data,
                w.head_matrix(),
                w.cfg.hidden_size,
            );
            Ok(logits)
        }
    }
}

/// Recompute the `GOT_LMHEAD_REFINE_K` largest int8-approx `logits` in exact f32
/// (`normed · embed_row` — the SAME value the f32 head produces for that token), so the
/// greedy argmax over the refined vector matches the f32 lm_head. Makes the int8 lm_head
/// near-lossless: the true best token is in the int8 top-K with overwhelming probability, so
/// its refined logit is exact and wins. `embed` is the HEAD SOURCE `[vocab, hidden]`
/// row-major — the tied embed table OR the arch's untied lm_head
/// ([`GotDecodeWeights::head_matrix`]), never blindly the embed.
fn refine_topk_f32(logits: &mut [f32], normed: &[f32], embed: &[f32], hidden: usize) {
    let vocab = logits.len();
    let k = GOT_LMHEAD_REFINE_K.min(vocab);
    if k == 0 {
        return;
    }
    // Partition so idx[..k] index the k largest int8-approx logits (O(vocab), no full sort).
    let mut idx: Vec<u32> = (0..vocab as u32).collect();
    idx.select_nth_unstable_by(k - 1, |&a, &b| {
        logits[b as usize].total_cmp(&logits[a as usize])
    });
    for &t in &idx[..k] {
        let t = t as usize;
        let row = &embed[t * hidden..(t + 1) * hidden];
        logits[t] = normed.iter().zip(row).map(|(&x, &wv)| x * wv).sum();
    }
}

/// Seeding prefill: run all `N` positions through the layers (BIT-IDENTICAL to
/// [`forward_prefill`] — same `linear_int8_dynamic` kernel), capturing each layer's
/// post-RoPE K + post-proj V into `caches`, and return the **last-position** logits
/// (projected via [`got_lm_head`] on the single last row — the other N-1 rows are never
/// argmaxed, so we skip the full `[N, vocab]` head GEMM the naive path computed).
fn forward_prefill_seed(
    w: &GotDecodeWeights,
    inputs_embeds: &Mat,
    caches: &mut [Qwen2KvCache],
) -> FocrResult<Vec<f32>> {
    let cfg = &w.cfg;
    let eps = cfg.rms_norm_eps;
    let mut x = inputs_embeds.clone();
    add_learned_positions(&mut x, w, 0)?;
    let positions: Vec<usize> = (0..inputs_embeds.rows).collect();
    let rope = decoder::RopeTable::build(&positions, cfg.head_dim, cfg.rope_theta);
    let (q_dim, kv_dim) = (cfg.q_dim(), cfg.kv_dim());
    for (l, cl) in w.layers.iter().enumerate() {
        let ln1_b = cl.ln_bias.as_ref().map(|(a, _)| a.as_slice());
        let normed = family_norm(&x, &cl.input_ln, ln1_b, eps)?;
        // one fused q|k|v GEMM (m=N, row-parallel), then split the columns.
        let qkv = nn::linear_int8_dynamic(&normed, &cl.qkv, cl.qkv_bias.as_deref())?;
        let (mut q, mut k, v) = split_qkv_rows(&qkv, q_dim, kv_dim);
        if w.embed_positions.is_none() {
            decoder::apply_rope(&mut q, &rope)?;
            decoder::apply_rope(&mut k, &rope)?;
        }
        // The cache holds the GQA-native (unbroadcast) K/V; the decode step's
        // per-head kv mapping reads it at kv_dim stride.
        caches[l].seed(&k.data, &v.data);
        let ctx = prefill_attention_gqa(&q, &k, &v, cfg)?;
        let attn = nn::linear_int8_dynamic(&ctx, &cl.o, cl.o_bias.as_deref())?;
        let h = decoder::add_residual(&x, &attn)?;
        let ln2_b = cl.ln_bias.as_ref().map(|(_, b)| b.as_slice());
        let normed2 = family_norm(&h, &cl.post_attn_ln, ln2_b, eps)?;
        let mlp = match &cl.mlp {
            MlpW::SwiGlu { gate, up, down } => decoder::expert_mlp_i8(&normed2, gate, up, down)?,
            MlpW::ReluFc {
                fc1,
                fc1_b,
                fc2,
                fc2_b,
            } => {
                let mut m = nn::linear_int8_dynamic(&normed2, fc1, Some(fc1_b))?;
                nn::relu(&mut m);
                nn::linear_int8_dynamic(&m, fc2, Some(fc2_b))?
            }
        };
        x = decoder::add_residual(&h, &mlp)?;
    }
    let last = x.rows - 1;
    let x_last = Mat::from_vec(
        1,
        cfg.hidden_size,
        x.data[last * cfg.hidden_size..].to_vec(),
    );
    got_lm_head(w, &x_last, eps)
}

/// Split a fused `[N, q_dim + 2·kv_dim]` q|k|v activation into `[N, q_dim]` +
/// two `[N, kv_dim]` mats (column blocks; `kv_dim == q_dim` for MHA).
fn split_qkv_rows(fused: &Mat, q_dim: usize, kv_dim: usize) -> (Mat, Mat, Mat) {
    let n = fused.rows;
    let w = q_dim + 2 * kv_dim;
    let (mut q, mut k, mut v) = (
        vec![0.0f32; n * q_dim],
        vec![0.0f32; n * kv_dim],
        vec![0.0f32; n * kv_dim],
    );
    for r in 0..n {
        let row = &fused.data[r * w..(r + 1) * w];
        q[r * q_dim..(r + 1) * q_dim].copy_from_slice(&row[0..q_dim]);
        k[r * kv_dim..(r + 1) * kv_dim].copy_from_slice(&row[q_dim..q_dim + kv_dim]);
        v[r * kv_dim..(r + 1) * kv_dim].copy_from_slice(&row[q_dim + kv_dim..w]);
    }
    (
        Mat::from_vec(n, q_dim, q),
        Mat::from_vec(n, kv_dim, k),
        Mat::from_vec(n, kv_dim, v),
    )
}

/// Full-causal prefill attention with GQA head sharing (A7): MHA
/// (`kv_group() == 1`, the GOT path) calls [`decoder::prefill_attention`]
/// UNCHANGED — byte-identical to pre-GQA; a grouped config first broadcasts
/// K/V to the full query-head layout ([`broadcast_kv`]) so the certified MHA
/// kernel runs verbatim (lossless by construction: query head `h` sees exactly
/// kv head `h / kv_group()`'s rows).
fn prefill_attention_gqa(q: &Mat, k: &Mat, v: &Mat, cfg: &DecoderConfig) -> FocrResult<Mat> {
    if cfg.kv_group() == 1 {
        return decoder::prefill_attention(q, k, v, cfg.num_attention_heads, cfg.head_dim);
    }
    let k_full = broadcast_kv(k, cfg)?;
    let v_full = broadcast_kv(v, cfg)?;
    decoder::prefill_attention(q, &k_full, &v_full, cfg.num_attention_heads, cfg.head_dim)
}

/// Broadcast a GQA `[seq, kv_dim]` K or V to the full `[seq, q_dim]` head
/// layout: each kv head's `[head_dim]` lane is repeated `kv_group()` times so
/// query head `h` finds its shared kv head at lane `h`. A pure copy — no math.
fn broadcast_kv(m: &Mat, cfg: &DecoderConfig) -> FocrResult<Mat> {
    let (kv_dim, q_dim, hd, group) = (cfg.kv_dim(), cfg.q_dim(), cfg.head_dim, cfg.kv_group());
    if m.cols != kv_dim {
        return Err(FocrError::Other(anyhow::anyhow!(
            "broadcast_kv: cols {} != kv_dim {kv_dim}",
            m.cols
        )));
    }
    let mut out = vec![0.0f32; m.rows * q_dim];
    for r in 0..m.rows {
        let src = &m.data[r * kv_dim..(r + 1) * kv_dim];
        let dst = &mut out[r * q_dim..(r + 1) * q_dim];
        for g in 0..cfg.num_key_value_heads {
            let lane = &src[g * hd..(g + 1) * hd];
            for rep in 0..group {
                let h = g * group + rep;
                dst[h * hd..(h + 1) * hd].copy_from_slice(lane);
            }
        }
    }
    Ok(Mat::from_vec(m.rows, q_dim, out))
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
    let (hidden, eps) = (cfg.hidden_size, cfg.rms_norm_eps);
    let (num_heads, head_dim) = (cfg.num_attention_heads, cfg.head_dim);
    let (q_dim, kv_dim, kv_group) = (cfg.q_dim(), cfg.kv_dim(), cfg.kv_group());
    let rope = decoder::RopeTable::build(&[position], head_dim, cfg.rope_theta);
    let mut x = x.clone();
    add_learned_positions(&mut x, w, position)?;
    let tlayers = std::time::Instant::now();
    for (l, cl) in w.layers.iter().enumerate() {
        // ── attention: quantize the normed row ONCE, fused q|k|v GEMV (n-parallel) ──
        let ln1_b = cl.ln_bias.as_ref().map(|(a, _)| a.as_slice());
        let normed = family_norm(&x, &cl.input_ln, ln1_b, eps)?;
        let (xq, a) = decoder::quantize_row_i8_te(&normed.data);
        let qkv = decoder::gemv_i8_bias_prequant(&xq, a, &cl.qkv, cl.qkv_bias.as_deref());
        let mut q = Mat::from_vec(1, q_dim, qkv[0..q_dim].to_vec());
        let mut k = Mat::from_vec(1, kv_dim, qkv[q_dim..q_dim + kv_dim].to_vec());
        let v = &qkv[q_dim + kv_dim..q_dim + 2 * kv_dim];
        if w.embed_positions.is_none() {
            decoder::apply_rope(&mut q, &rope)?;
            decoder::apply_rope(&mut k, &rope)?;
        }
        caches[l].append(&k.data, v);
        let ta = std::time::Instant::now();
        let ctx = qwen2_decode_attention(&caches[l], &q.data, num_heads, head_dim, kv_group);
        DECODE_ATTN_NS.fetch_add(ta.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let (xqc, ac) = decoder::quantize_row_i8_te(&ctx);
        let attn = decoder::gemv_i8_bias_prequant(&xqc, ac, &cl.o, cl.o_bias.as_deref());
        let h = decoder::add_residual(&x, &Mat::from_vec(1, hidden, attn))?;
        // ── MLP: quantize normed2 ONCE (SwiGLU shares it for gate + up) ──────────
        let ln2_b = cl.ln_bias.as_ref().map(|(_, b)| b.as_slice());
        let normed2 = family_norm(&h, &cl.post_attn_ln, ln2_b, eps)?;
        let (xq2, a2) = decoder::quantize_row_i8_te(&normed2.data);
        let mlp_out = match &cl.mlp {
            MlpW::SwiGlu { gate, up, down } => {
                let mut g = Mat::from_vec(
                    1,
                    cfg.intermediate_size,
                    decoder::gemv_i8_bias_prequant(&xq2, a2, gate, None),
                );
                let u = decoder::gemv_i8_bias_prequant(&xq2, a2, up, None);
                nn::silu(&mut g);
                for (gv, &uv) in g.data.iter_mut().zip(u.iter()) {
                    *gv *= uv;
                }
                let (xq3, a3) = decoder::quantize_row_i8_te(&g.data);
                decoder::gemv_i8_bias_prequant(&xq3, a3, down, None)
            }
            MlpW::ReluFc {
                fc1,
                fc1_b,
                fc2,
                fc2_b,
            } => {
                let mut m = Mat::from_vec(
                    1,
                    cfg.intermediate_size,
                    decoder::gemv_i8_bias_prequant(&xq2, a2, fc1, Some(fc1_b)),
                );
                nn::relu(&mut m);
                let (xq3, a3) = decoder::quantize_row_i8_te(&m.data);
                decoder::gemv_i8_bias_prequant(&xq3, a3, fc2, Some(fc2_b))
            }
        };
        x = decoder::add_residual(&h, &Mat::from_vec(1, hidden, mlp_out))?;
    }
    DECODE_GEMV_NS.fetch_add(tlayers.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let thead = std::time::Instant::now();
    let logits = got_lm_head(w, &x, eps)?;
    DECODE_LMHEAD_NS.fetch_add(thead.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(logits)
}

// ─────────── A7.5: the batched dense decode step (bd-3jo6.1.7.5) ───────────
//
// The multi-stream twin of [`qwen2_decode_step`] for the dense zoo decoders
// (GOT/SmolVLM2 QwenLlama, OneChart Opt). The LOSSLESS contract: stream `s`'s
// logits are BYTE-FOR-BYTE the m=1 step run on stream `s` alone. That holds by
// construction:
//
//  * every per-stream op (family norm, ties-to-even quantize, RoPE at the
//    stream's own absolute position, learned-position add, full-causal
//    attention over the stream's own cache, residuals, SiLU/ReLU) is the SAME
//    code the m=1 step runs, on the same per-stream data;
//  * the six projections (qkv, o, gate, up / fc1, down / fc2) run as ONE
//    `M=B` [`decoder::gemm_i8_bias_prequant_batched`] each — M-independent
//    integer accumulation with per-row dequant identical to
//    [`decoder::gemv_i8_bias_prequant`] (the in-module gates prove both).

/// Per-stream full-causal KV for the dense batch spine: stream-major
/// `[stream][layer]` [`Qwen2KvCache`]s, each stream at its OWN length. The
/// dense analogue of `rswa::BatchedRingCache` (dense models attend the whole
/// prefix — no ring, no eviction).
struct BatchedQwen2KvCache {
    streams: Vec<Vec<Qwen2KvCache>>,
}

impl BatchedQwen2KvCache {
    /// Build from per-stream seeded caches (one inner `Vec` per stream,
    /// `n_layers` caches each — exactly what per-stream prefill seeding
    /// produces).
    fn from_streams(streams: Vec<Vec<Qwen2KvCache>>) -> Self {
        Self { streams }
    }

    fn num_streams(&self) -> usize {
        self.streams.len()
    }

    /// Stream `s`'s next absolute position (== its cached row count).
    #[cfg(test)]
    fn position(&self, s: usize) -> usize {
        self.streams[s][0].n_kv
    }
}

/// One batched decode step over `B` streams: `xs[s]` is stream `s`'s `[1, hidden]`
/// token embedding, `positions[s]` its absolute position (== `caches.position(s)`
/// in the steady state; passed explicitly so the caller stays in charge, exactly
/// like the m=1 signature). Returns `B` `[vocab]` logit rows.
///
/// # Errors
/// Any shape/norm error from the underlying kernels.
fn qwen2_batched_decode_step(
    w: &GotDecodeWeights,
    caches: &mut BatchedQwen2KvCache,
    active: &[usize],
    xs: &[Mat],
    positions: &[usize],
) -> FocrResult<Vec<Vec<f32>>> {
    let b = xs.len();
    debug_assert_eq!(b, positions.len());
    debug_assert_eq!(b, active.len());
    debug_assert!(active.iter().all(|&s| s < caches.num_streams()));
    let cfg = &w.cfg;
    let (hidden, eps) = (cfg.hidden_size, cfg.rms_norm_eps);
    let (num_heads, head_dim) = (cfg.num_attention_heads, cfg.head_dim);
    let (q_dim, kv_dim, kv_group) = (cfg.q_dim(), cfg.kv_dim(), cfg.kv_group());

    // Per-stream working rows + per-stream RoPE tables at each TRUE position.
    let mut x: Vec<Mat> = Vec::with_capacity(b);
    let mut ropes: Vec<decoder::RopeTable> = Vec::with_capacity(b);
    for (s, xin) in xs.iter().enumerate() {
        let mut row = xin.clone();
        add_learned_positions(&mut row, w, positions[s])?;
        x.push(row);
        ropes.push(decoder::RopeTable::build(
            &[positions[s]],
            head_dim,
            cfg.rope_theta,
        ));
    }

    for (l, cl) in w.layers.iter().enumerate() {
        // ── attention: per-stream norm + ties-to-even quantize, ONE M=B qkv GEMM ──
        let ln1_b = cl.ln_bias.as_ref().map(|(a, _)| a.as_slice());
        let mut prequant: Vec<(Vec<i8>, f32)> = Vec::with_capacity(b);
        for xs_row in &x {
            let normed = family_norm(xs_row, &cl.input_ln, ln1_b, eps)?;
            prequant.push(decoder::quantize_row_i8_te(&normed.data));
        }
        let rows: Vec<(&[i8], f32)> = prequant.iter().map(|(q, a)| (q.as_slice(), *a)).collect();
        let qkv_rows =
            decoder::gemm_i8_bias_prequant_batched(&rows, &cl.qkv, cl.qkv_bias.as_deref());

        // ── per-stream: split/RoPE/append/attend, then ONE M=B o_proj ──
        let mut ctx_prequant: Vec<(Vec<i8>, f32)> = Vec::with_capacity(b);
        for (s, qkv) in qkv_rows.iter().enumerate() {
            let mut q = Mat::from_vec(1, q_dim, qkv[0..q_dim].to_vec());
            let mut k = Mat::from_vec(1, kv_dim, qkv[q_dim..q_dim + kv_dim].to_vec());
            let v = &qkv[q_dim + kv_dim..q_dim + 2 * kv_dim];
            if w.embed_positions.is_none() {
                decoder::apply_rope(&mut q, &ropes[s])?;
                decoder::apply_rope(&mut k, &ropes[s])?;
            }
            caches.streams[active[s]][l].append(&k.data, v);
            let ctx = qwen2_decode_attention(
                &caches.streams[active[s]][l],
                &q.data,
                num_heads,
                head_dim,
                kv_group,
            );
            ctx_prequant.push(decoder::quantize_row_i8_te(&ctx));
        }
        let ctx_rows: Vec<(&[i8], f32)> = ctx_prequant
            .iter()
            .map(|(q, a)| (q.as_slice(), *a))
            .collect();
        let attn_rows =
            decoder::gemm_i8_bias_prequant_batched(&ctx_rows, &cl.o, cl.o_bias.as_deref());
        let mut h: Vec<Mat> = Vec::with_capacity(b);
        for (s, attn) in attn_rows.into_iter().enumerate() {
            h.push(decoder::add_residual(
                &x[s],
                &Mat::from_vec(1, hidden, attn),
            )?);
        }

        // ── MLP: per-stream norm2 + quantize ONCE, batched projections ──
        let ln2_b = cl.ln_bias.as_ref().map(|(_, bb)| bb.as_slice());
        let mut mlp_prequant: Vec<(Vec<i8>, f32)> = Vec::with_capacity(b);
        for hs in &h {
            let normed2 = family_norm(hs, &cl.post_attn_ln, ln2_b, eps)?;
            mlp_prequant.push(decoder::quantize_row_i8_te(&normed2.data));
        }
        let mlp_rows: Vec<(&[i8], f32)> = mlp_prequant
            .iter()
            .map(|(q, a)| (q.as_slice(), *a))
            .collect();
        let mlp_out: Vec<Vec<f32>> = match &cl.mlp {
            MlpW::SwiGlu { gate, up, down } => {
                let g_rows = decoder::gemm_i8_bias_prequant_batched(&mlp_rows, gate, None);
                let u_rows = decoder::gemm_i8_bias_prequant_batched(&mlp_rows, up, None);
                let mut act_prequant: Vec<(Vec<i8>, f32)> = Vec::with_capacity(b);
                for (g_row, u_row) in g_rows.into_iter().zip(&u_rows) {
                    let mut g = Mat::from_vec(1, cfg.intermediate_size, g_row);
                    nn::silu(&mut g);
                    for (gv, &uv) in g.data.iter_mut().zip(u_row.iter()) {
                        *gv *= uv;
                    }
                    act_prequant.push(decoder::quantize_row_i8_te(&g.data));
                }
                let act_rows: Vec<(&[i8], f32)> = act_prequant
                    .iter()
                    .map(|(q, a)| (q.as_slice(), *a))
                    .collect();
                decoder::gemm_i8_bias_prequant_batched(&act_rows, down, None)
            }
            MlpW::ReluFc {
                fc1,
                fc1_b,
                fc2,
                fc2_b,
            } => {
                let m_rows = decoder::gemm_i8_bias_prequant_batched(&mlp_rows, fc1, Some(fc1_b));
                let mut act_prequant: Vec<(Vec<i8>, f32)> = Vec::with_capacity(b);
                for m_row in m_rows {
                    let mut m = Mat::from_vec(1, cfg.intermediate_size, m_row);
                    nn::relu(&mut m);
                    act_prequant.push(decoder::quantize_row_i8_te(&m.data));
                }
                let act_rows: Vec<(&[i8], f32)> = act_prequant
                    .iter()
                    .map(|(q, a)| (q.as_slice(), *a))
                    .collect();
                decoder::gemm_i8_bias_prequant_batched(&act_rows, fc2, Some(fc2_b))
            }
        };
        for (s, out) in mlp_out.into_iter().enumerate() {
            x[s] = decoder::add_residual(&h[s], &Mat::from_vec(1, hidden, out))?;
        }
    }

    // lm_head stays per-stream through the SAME [`got_lm_head`] (its int8+top-K
    // refine lever composes per stream; batching the head is a ledgered perf
    // follow-on — correctness first).
    let mut logits = Vec::with_capacity(b);
    for xs_row in &x {
        logits.push(got_lm_head(w, xs_row, eps)?);
    }
    Ok(logits)
}

// ── A7.5 part 2: the dense continuous-batch generate (bd-3jo6.1.7.5) ──

/// The dense [`batch_scheduler::BatchStep`]: the scheduler's per-stream
/// "hidden" IS the `[1, vocab]` LOGITS row (seeded from prefill, replaced
/// each advance), which maps the dense m=1 loop onto the scheduler contract
/// with byte-equal semantics:
///
///  * the token pick is the SAME [`argmax_no_repeat`] over the SAME emitted
///    history (dense prompt ids are never in `generated` — prefill is
///    `inputs_embeds` — so `history == ids` exactly as the m=1 loop);
///  * the advance is [`qwen2_batched_decode_step`] over the non-EOS subset
///    (bit-identical per stream, gated in-module);
///  * positions mirror the m=1 `caches.n_kv` progression (`prefill_len`,
///    then +1 per step — the scheduler's own accounting).
struct DenseDecoderBatchStep<'w> {
    w: &'w GotDecodeWeights,
    caches: BatchedQwen2KvCache,
    eos: u32,
}

impl batch_scheduler::BatchStep for DenseDecoderBatchStep<'_> {
    fn step(
        &mut self,
        slots: &[batch_scheduler::StreamSlot<'_>],
    ) -> FocrResult<Vec<batch_scheduler::StreamOut>> {
        let cfg = &self.w.cfg;
        let n = cfg.no_repeat_ngram_size;
        // 1. Pick each stream's next token from its CURRENT logits row.
        let picks: Vec<(u32, bool)> = slots
            .iter()
            .map(|sl| {
                let t = argmax_no_repeat(&sl.hidden.data, sl.history, n) as u32;
                (t, t == self.eos)
            })
            .collect();
        // 2. Advance the non-EOS subset in ONE batched forward.
        let mut active: Vec<usize> = Vec::new();
        let mut embeds: Vec<Mat> = Vec::new();
        let mut positions: Vec<usize> = Vec::new();
        for (k, sl) in slots.iter().enumerate() {
            if !picks[k].1 {
                active.push(sl.slot_index);
                embeds.push(decoder::embed_tokens(
                    &self.w.embed,
                    cfg.vocab_size,
                    cfg.hidden_size,
                    &[picks[k].0],
                )?);
                positions.push(sl.position);
            }
        }
        let logits = if active.is_empty() {
            Vec::new()
        } else {
            qwen2_batched_decode_step(self.w, &mut self.caches, &active, &embeds, &positions)?
        };
        // 3. Assemble StreamOuts (EOS streams carry a dummy hidden the
        //    scheduler never reads).
        let mut li = 0usize;
        Ok(picks
            .into_iter()
            .map(|(token, is_eos)| {
                if is_eos {
                    batch_scheduler::StreamOut {
                        token,
                        is_eos,
                        new_hidden: Mat::from_vec(1, 1, vec![0.0]),
                    }
                } else {
                    let row = Mat::from_vec(1, cfg.vocab_size, logits[li].clone());
                    li += 1;
                    batch_scheduler::StreamOut {
                        token,
                        is_eos: false,
                        new_hidden: row,
                    }
                }
            })
            .collect())
    }
}

/// Continuous-batch greedy decode over MANY pages' `inputs_embeds` — the
/// batched twin of [`generate_greedy_kvcache`] (A7.5). The decode-time weights
/// build ONCE for the whole batch; each page's prefill seeds sequentially
/// (one live forward at a time, doctrine #5); the [`batch_scheduler`] then
/// drives batched decode steps with admit/retire/backfill. Per stream the
/// id-stream is BYTE-IDENTICAL to [`generate_greedy_kvcache`] run alone (the
/// in-module gate proves it on both dense families).
///
/// # Errors
/// Any prefill/decode/scheduler error; `model.embed_tokens.weight` missing.
pub fn generate_greedy_batched(
    weights: &Weights,
    cfg: &DecoderConfig,
    inputs_embeds: &[Mat],
    caps: &[usize],
    eos: u32,
) -> FocrResult<Vec<Vec<u32>>> {
    if inputs_embeds.is_empty() || caps.iter().all(|&c| c == 0) {
        return Ok(vec![Vec::new(); inputs_embeds.len()]);
    }
    let w = GotDecodeWeights::build(weights, cfg)?;
    generate_greedy_batched_with(&w, inputs_embeds, caps, eos)
}

/// [`generate_greedy_batched`] over PRE-BUILT decode weights — the testable
/// seam (the in-module gate drives synthetic weights through it).
fn generate_greedy_batched_with(
    w: &GotDecodeWeights,
    inputs_embeds: &[Mat],
    caps: &[usize],
    eos: u32,
) -> FocrResult<Vec<Vec<u32>>> {
    debug_assert_eq!(inputs_embeds.len(), caps.len());
    let cfg = &w.cfg;
    // The scheduler's global cap is the loosest page cap; tighter pages carry
    // their own per-stream `max_emit` (a page whose position budget binds
    // must emit EXACTLY what its solo run would).
    let max_new = caps.iter().copied().max().unwrap_or(0);
    let mut stream_caches: Vec<Vec<Qwen2KvCache>> = Vec::with_capacity(inputs_embeds.len());
    let mut pages: Vec<batch_scheduler::PageStream> = Vec::with_capacity(inputs_embeds.len());
    for (i, embeds) in inputs_embeds.iter().enumerate() {
        let mut caches: Vec<Qwen2KvCache> = (0..cfg.num_hidden_layers)
            .map(|_| Qwen2KvCache::new(cfg.kv_dim(), embeds.rows + caps[i]))
            .collect();
        let last_logits = forward_prefill_seed(w, embeds, &mut caches)?;
        stream_caches.push(caches);
        pages.push(
            batch_scheduler::PageStream::new(
                i,
                embeds.rows,
                &[],
                Mat::from_vec(1, cfg.vocab_size, last_logits),
            )
            .with_max_emit(caps[i]),
        );
    }
    let mut sched = batch_scheduler::BatchScheduler::from_env(max_new);
    let mut step = DenseDecoderBatchStep {
        w,
        caches: BatchedQwen2KvCache::from_streams(stream_caches),
        eos,
    };
    let out = sched.run(pages, &mut step)?;
    let stats = sched.stats();
    debug_assert!(
        stats.max_concurrent_forwards <= 1,
        "dense spine: >1 live forward"
    );
    Ok(out)
}

/// **O(n)-per-token** greedy decode: identical id-stream to [`generate_greedy`] but
/// with a full-causal KV cache instead of re-running prefill each step. The bit-for-bit
/// equality is enforced by reusing [`nn::linear_int8_dynamic`] for every GEMM (the
/// prefill kernel), so the decode never diverges from the certified path. Both paths
/// apply the same [`argmax_no_repeat`] global no-repeat-n-gram guard (bd-ff4i).
///
/// Precision caveat: with the int8+top-K-refine lm_head lever ON, the identity
/// to `generate_greedy` (whose head is always f32) holds as long as every
/// guard-masked argmax lands inside the refined top-K — logits BEYOND the top-K
/// are int8-approximate, so a ban deep enough to push the pick past K could in
/// principle flip a near-tie the f32 head would order differently. K = 256 ≫
/// any observed ban set (a 20-gram ban removes a handful of ids); the GOT
/// oracle certs + the 20-page sweep ran with the lever ON and stayed exact.
/// `FOCR_GOT_INT8_LMHEAD=0` removes the caveat entirely.
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
    generate_greedy_kvcache_with(&w, inputs_embeds, max_new, eos)
}

/// [`generate_greedy_kvcache`] over PRE-BUILT decode weights — the seam the
/// batched gate compares against (same body, weights built by the caller).
fn generate_greedy_kvcache_with(
    w: &GotDecodeWeights,
    inputs_embeds: &Mat,
    max_new: usize,
    eos: u32,
) -> FocrResult<Vec<u32>> {
    let cfg = &w.cfg;
    let n = inputs_embeds.rows;
    let mut caches: Vec<Qwen2KvCache> = (0..cfg.num_hidden_layers)
        .map(|_| Qwen2KvCache::new(cfg.kv_dim(), n + max_new))
        .collect();
    let timing = std::env::var_os("FOCR_TIMING").is_some();
    if timing {
        DECODE_ATTN_NS.store(0, Ordering::Relaxed);
        DECODE_GEMV_NS.store(0, Ordering::Relaxed);
        DECODE_LMHEAD_NS.store(0, Ordering::Relaxed);
    }
    let tseed = std::time::Instant::now();
    let last_logits = forward_prefill_seed(w, inputs_embeds, &mut caches)?;
    let seed_s = tseed.elapsed().as_secs_f64();
    let tdec = std::time::Instant::now();
    let mut ids = Vec::new();
    let mut next = argmax_no_repeat(&last_logits, &ids, cfg.no_repeat_ngram_size) as u32;
    for _ in 0..max_new {
        crate::cancel_checkpoint()?;
        ids.push(next);
        if next == eos {
            break;
        }
        let te = decoder::embed_tokens(&w.embed, cfg.vocab_size, cfg.hidden_size, &[next])?;
        // the new token occupies the position after every currently-cached row.
        let position = caches[0].n_kv;
        let logits = qwen2_decode_step(w, &mut caches, &te, position)?;
        next = argmax_no_repeat(&logits, &ids, cfg.no_repeat_ngram_size) as u32;
    }
    if timing {
        let dec_s = tdec.elapsed().as_secs_f64();
        let ns = 1e-9;
        let (attn, layers, head) = (
            DECODE_ATTN_NS.load(Ordering::Relaxed) as f64 * ns,
            DECODE_GEMV_NS.load(Ordering::Relaxed) as f64 * ns,
            DECODE_LMHEAD_NS.load(Ordering::Relaxed) as f64 * ns,
        );
        crate::progress::stderr_message(format_args!(
            "[focr-timing]   decode {} tok in {:.2}s ({:.1} tok/s) | seed(prefill {n} tok) {:.2}s | \
             layers {:.2}s (attn {:.2}s, gemv+misc {:.2}s) | lm_head {:.2}s",
            ids.len(),
            dec_s,
            ids.len() as f64 / dec_s.max(1e-9),
            seed_s,
            layers,
            attn,
            layers - attn,
            head,
        ));
    }
    Ok(ids)
}

/// Greedy pick with the HF-builtin **global** no-repeat-n-gram ban (bd-ff4i,
/// spec §12 OQ-8): mask to −∞ every token that would complete an `n`-gram already
/// present in the generated `ids` (reusing the sampler's `window == 0` global
/// scan), then argmax. `n == 0` disables (plain argmax). The guard cannot fire
/// before `ids` holds `n` tokens, so a repeat-free stream — e.g. the oracle-L4
/// cert sequence — stays BYTE-IDENTICAL to unguarded greedy. Scans the generated
/// stream only: upstream's HF processor also spans the prompt ids, which a text
/// completion can re-match only by re-emitting prompt text (no observed case).
/// Composes with the int8+top-K-refine head: a ban removes at most a handful of
/// ids per step, so the masked argmax still lands inside the refined top-K.
fn argmax_no_repeat(logits: &[f32], ids: &[u32], n: usize) -> usize {
    match sampler::masked_sliding_window_logits_if_needed(logits, ids, n, 0, &[]) {
        Some(masked) => argmax(&masked),
        None => argmax(logits),
    }
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
        // MHA identity: q_dim == kv_dim == hidden, group 1 (GQA is A7's delta).
        assert_eq!(c.num_key_value_heads, 16);
        assert_eq!(c.q_dim(), 1024);
        assert_eq!(c.kv_dim(), 1024);
        assert_eq!(c.kv_group(), 1);
        assert_eq!(c.vocab_size, 151_860);
        // full-causal scale must be exactly 1/8 (= 1/sqrt(64), what prefill_attention derives).
        assert!(((1.0 / (c.head_dim as f32).sqrt()) - 0.125).abs() < 1e-7);
        // upstream `chat()` hard-codes the HF global no_repeat_ngram_size=20 (OQ-8).
        assert_eq!(c.no_repeat_ngram_size, 20);
    }

    /// The SmolVLM2-500M census shape (docs/zoo/smolvlm2-spec.md §4), now the
    /// PUBLIC constructor (C5); this pin mirrors got_config_is_the_censused_shape.
    #[test]
    fn smolvlm2_config_is_the_censused_shape() {
        let c = DecoderConfig::smolvlm2();
        assert_eq!(c.hidden_size, 960);
        assert_eq!(c.intermediate_size, 2560);
        assert_eq!(c.num_hidden_layers, 32);
        assert_eq!(c.num_attention_heads, 15);
        assert_eq!(c.head_dim, 64);
        assert_eq!(c.vocab_size, 49_280);
        assert!(!c.attn_qkv_bias, "SmolLM2 carries no qkv bias");
        assert_eq!(c.no_repeat_ngram_size, 0, "no upstream repetition guard");
        assert_eq!(c.lm_head, Some("lm_head.weight"), "UNTIED head");
        // scale must be exactly 1/8 (= 1/sqrt(64), what prefill_attention derives).
        assert!(((1.0 / (c.head_dim as f32).sqrt()) - 0.125).abs() < 1e-7);
    }

    /// The DecoderConfig tensor names may never drift from the ModelArch
    /// registry descriptor (the convert side classifies by the SAME names).
    #[test]
    fn smolvlm2_config_names_match_the_descriptor() {
        let c = DecoderConfig::smolvlm2();
        let arch = super::super::model_arch::arch_by_id("smolvlm2").expect("registered");
        assert_eq!(c.layers_prefix, arch.decoder_layers_prefix());
        assert_eq!(c.embed_tokens, arch.embed_tokens_name());
        assert_eq!(
            c.lm_head.is_none(),
            arch.tie_word_embeddings(),
            "tied-ness must agree between config and descriptor"
        );
        let g = DecoderConfig::got_ocr2();
        let got = super::super::model_arch::arch_by_id("got-ocr2").expect("registered");
        assert_eq!(g.layers_prefix, got.decoder_layers_prefix());
        assert_eq!(g.embed_tokens, got.embed_tokens_name());
        assert!(got.tie_word_embeddings() && g.lm_head.is_none());
    }

    // ── A7.5: the batched dense decode bit-identity gates (bd-3jo6.1.7.5) ──

    /// Deterministic pseudo-random i8 weights / f32 rows (no rand dep; values
    /// span sign + magnitude so quant/dequant paths are exercised).
    fn det_i8(i: usize) -> i8 {
        (((i * 31 + 7) % 251) as i64 - 125) as i8
    }
    fn det_f32(i: usize) -> f32 {
        ((((i * 37 + 11) % 199) as f32) - 99.0) / 37.0
    }

    fn synth_q(n: usize, k: usize, seed: usize) -> QInt8 {
        let w: Vec<i8> = (0..n * k).map(|i| det_i8(i + seed)).collect();
        let scales: Vec<f32> = (0..n)
            .map(|i| 0.01 + det_f32(i + seed).abs() * 0.003)
            .collect();
        QInt8::new(w, scales, n, k)
    }

    /// A tiny synthetic dense model for a family: 2 layers, hidden 8, 2 heads,
    /// head_dim 4, inner 16, vocab 12 — every code path of the real shapes at
    /// unit-test cost.
    fn synth_weights(family: DecoderFamily) -> GotDecodeWeights {
        let (hidden, inter, vocab, layers_n) = (8usize, 16usize, 12usize, 2usize);
        let opt = matches!(family, DecoderFamily::Opt);
        let layers = (0..layers_n)
            .map(|l| {
                let seed = l * 10_000;
                GotLayerW {
                    input_ln: (0..hidden)
                        .map(|i| 1.0 + det_f32(i + seed) * 0.01)
                        .collect(),
                    post_attn_ln: (0..hidden)
                        .map(|i| 1.0 + det_f32(i + seed + 77) * 0.01)
                        .collect(),
                    qkv: synth_q(3 * hidden, hidden, seed + 1),
                    qkv_bias: Some(
                        (0..3 * hidden)
                            .map(|i| det_f32(i + seed + 2) * 0.05)
                            .collect(),
                    ),
                    o: synth_q(hidden, hidden, seed + 3),
                    o_bias: opt
                        .then(|| (0..hidden).map(|i| det_f32(i + seed + 4) * 0.05).collect()),
                    ln_bias: opt.then(|| {
                        (
                            (0..hidden).map(|i| det_f32(i + seed + 5) * 0.02).collect(),
                            (0..hidden).map(|i| det_f32(i + seed + 6) * 0.02).collect(),
                        )
                    }),
                    mlp: if opt {
                        MlpW::ReluFc {
                            fc1: synth_q(inter, hidden, seed + 7),
                            fc1_b: (0..inter).map(|i| det_f32(i + seed + 8) * 0.05).collect(),
                            fc2: synth_q(hidden, inter, seed + 9),
                            fc2_b: (0..hidden).map(|i| det_f32(i + seed + 10) * 0.05).collect(),
                        }
                    } else {
                        MlpW::SwiGlu {
                            gate: synth_q(inter, hidden, seed + 7),
                            up: synth_q(inter, hidden, seed + 8),
                            down: synth_q(hidden, inter, seed + 9),
                        }
                    },
                }
            })
            .collect();
        let embed: Vec<f32> = (0..vocab * hidden)
            .map(|i| det_f32(i + 555) * 0.2)
            .collect();
        // f32 pre-transposed [hidden, vocab] head from the tied embed table.
        let mut head_t = vec![0.0f32; hidden * vocab];
        for v in 0..vocab {
            for h in 0..hidden {
                head_t[h * vocab + v] = embed[v * hidden + h];
            }
        }
        GotDecodeWeights {
            layers,
            final_norm: (0..hidden).map(|i| 1.0 + det_f32(i + 999) * 0.01).collect(),
            final_norm_bias: opt.then(|| (0..hidden).map(|i| det_f32(i + 998) * 0.02).collect()),
            embed_positions: opt.then(|| {
                (
                    (0..64 * hidden)
                        .map(|i| det_f32(i + 424_242) * 0.03)
                        .collect(),
                    2,
                )
            }),
            embed,
            untied_head: None,
            lm_head: LmHead::F32(Mat::from_vec(hidden, vocab, head_t)),
            cfg: DecoderConfig {
                family,
                embed_positions: opt.then_some(("t", 2)),
                hidden_size: hidden,
                intermediate_size: inter,
                num_hidden_layers: layers_n,
                num_attention_heads: 2,
                head_dim: 4,
                vocab_size: vocab,
                rope_theta: 10_000.0,
                rms_norm_eps: 1e-6,
                attn_qkv_bias: true,
                no_repeat_ngram_size: 0,
                num_key_value_heads: 2,
                layers_prefix: "model.layers.",
                embed_tokens: "model.embed_tokens.weight",
                final_norm: "model.norm.weight",
                lm_head: None,
            },
        }
    }

    /// Seed one stream's caches with `n` deterministic prefill rows and return
    /// them (stream identity comes from `stream_seed`, so streams differ).
    fn seed_stream(w: &GotDecodeWeights, n: usize, stream_seed: usize) -> Vec<Qwen2KvCache> {
        let kv_dim = w.cfg.kv_dim();
        (0..w.cfg.num_hidden_layers)
            .map(|l| {
                let mut c = Qwen2KvCache::new(kv_dim, n + 32);
                let k: Vec<f32> = (0..n * kv_dim)
                    .map(|i| det_f32(i + stream_seed + l * 991) * 0.4)
                    .collect();
                let v: Vec<f32> = (0..n * kv_dim)
                    .map(|i| det_f32(i + stream_seed + l * 991 + 131) * 0.4)
                    .collect();
                c.seed(&k, &v);
                c
            })
            .collect()
    }

    /// THE lossless contract: for B streams at DIFFERENT lengths, every decode
    /// step's logits from [`qwen2_batched_decode_step`] are BIT-IDENTICAL
    /// (`f32::to_bits` equality) to [`qwen2_decode_step`] run per stream alone —
    /// for BOTH dense families (QwenLlama incl RoPE; Opt incl learned positions,
    /// LayerNorm biases, biased ReLU MLP).
    #[test]
    fn batched_dense_decode_step_is_bit_identical_per_stream() {
        for family in [DecoderFamily::QwenLlama, DecoderFamily::Opt] {
            let w = synth_weights(family);
            let hidden = w.cfg.hidden_size;
            let prefill_lens = [3usize, 5, 8];
            let b = prefill_lens.len();

            // Single-stream references: independent caches per stream.
            let mut solo: Vec<Vec<Qwen2KvCache>> = prefill_lens
                .iter()
                .enumerate()
                .map(|(s, &n)| seed_stream(&w, n, s * 100_003))
                .collect();
            // Batched twin: identically-seeded caches.
            let batched_streams: Vec<Vec<Qwen2KvCache>> = prefill_lens
                .iter()
                .enumerate()
                .map(|(s, &n)| seed_stream(&w, n, s * 100_003))
                .collect();
            let mut batched = BatchedQwen2KvCache::from_streams(batched_streams);

            for step in 0..4usize {
                // Per-stream token embeds differ across streams AND steps.
                let xs: Vec<Mat> = (0..b)
                    .map(|s| {
                        Mat::from_vec(
                            1,
                            hidden,
                            (0..hidden)
                                .map(|i| det_f32(i + s * 7_919 + step * 613) * 0.3)
                                .collect(),
                        )
                    })
                    .collect();
                let positions: Vec<usize> = (0..b).map(|s| batched.position(s)).collect();

                let active: Vec<usize> = (0..b).collect();
                let batch_logits =
                    qwen2_batched_decode_step(&w, &mut batched, &active, &xs, &positions)
                        .expect("batched");
                for s in 0..b {
                    let solo_logits =
                        qwen2_decode_step(&w, &mut solo[s], &xs[s], positions[s]).expect("solo");
                    let identical = solo_logits.len() == batch_logits[s].len()
                        && solo_logits
                            .iter()
                            .zip(&batch_logits[s])
                            .all(|(a, bb)| a.to_bits() == bb.to_bits());
                    assert!(
                        identical,
                        "family {family:?} stream {s} step {step}: batched logits != solo (LOSSLESS contract broken)"
                    );
                }
            }
            println!(
                "{{\"check\":\"batched_dense_decode_bit_identity\",\"family\":\"{family:?}\",\"streams\":{b},\"steps\":4,\"result\":\"pass\"}}"
            );
        }
    }

    /// THE scheduler-level lossless gate: [`generate_greedy_batched_with`]'s
    /// id-stream per page is IDENTICAL to [`generate_greedy_kvcache_with`] run
    /// per page alone — mixed prefill lengths, both dense families, and BOTH
    /// termination shapes (cap-bounded with an unreachable EOS; genuine EOS
    /// retirement mid-batch with backfill-order preserved).
    #[test]
    fn batched_generate_ids_match_solo_per_page() {
        for family in [DecoderFamily::QwenLlama, DecoderFamily::Opt] {
            let w = synth_weights(family);
            let hidden = w.cfg.hidden_size;
            // Three pages with DIFFERENT prompt lengths (deterministic embeds).
            let pages: Vec<Mat> = [3usize, 6, 4]
                .iter()
                .enumerate()
                .map(|(p, &rows)| {
                    Mat::from_vec(
                        rows,
                        hidden,
                        (0..rows * hidden)
                            .map(|i| det_f32(i + p * 50_021) * 0.25)
                            .collect(),
                    )
                })
                .collect();

            // Case 1: unreachable EOS — every stream runs to the cap.
            let max_new = 6usize;
            let solo: Vec<Vec<u32>> = pages
                .iter()
                .map(|e| generate_greedy_kvcache_with(&w, e, max_new, u32::MAX).expect("solo"))
                .collect();
            let caps = vec![max_new; pages.len()];
            let batched =
                generate_greedy_batched_with(&w, &pages, &caps, u32::MAX).expect("batched");
            assert_eq!(solo, batched, "family {family:?}: cap-bounded ids diverged");

            // Mixed PER-STREAM caps: each page must emit exactly its own
            // solo-run stream under its own cap (the position-budget case).
            let mixed = [6usize, 3, 5];
            let solo_mixed: Vec<Vec<u32>> = pages
                .iter()
                .zip(&mixed)
                .map(|(e, &m)| generate_greedy_kvcache_with(&w, e, m, u32::MAX).expect("solo"))
                .collect();
            let batched_mixed =
                generate_greedy_batched_with(&w, &pages, &mixed, u32::MAX).expect("batched");
            assert_eq!(
                solo_mixed, batched_mixed,
                "family {family:?}: mixed per-stream caps diverged"
            );

            // Case 2: a REAL mid-stream EOS — pick page 1's 3rd solo token as
            // EOS so that stream retires early while the others continue.
            let eos = solo[1][2];
            let solo_eos: Vec<Vec<u32>> = pages
                .iter()
                .map(|e| generate_greedy_kvcache_with(&w, e, max_new, eos).expect("solo"))
                .collect();
            let batched_eos =
                generate_greedy_batched_with(&w, &pages, &caps, eos).expect("batched");
            assert_eq!(
                solo_eos, batched_eos,
                "family {family:?}: EOS-retirement ids diverged"
            );
            assert!(
                solo_eos
                    .iter()
                    .any(|ids| ids.last() == Some(&eos) && ids.len() < max_new),
                "family {family:?}: the EOS case never actually retired early — test defanged"
            );
            println!(
                r#"{{"check":"batched_generate_ids_match_solo","family":"{family:?}","pages":3,"result":"pass"}}"#
            );
        }
    }

    /// The batched prequant GEMM helper equals the m=1 kernel per row (with and
    /// without bias), including mixed per-row scales.
    #[test]
    fn gemm_prequant_batched_matches_m1() {
        let qw = synth_q(24, 16, 42);
        let bias: Vec<f32> = (0..24).map(|i| det_f32(i + 4_242) * 0.1).collect();
        let raw: Vec<Vec<f32>> = (0..5)
            .map(|r| {
                (0..16)
                    .map(|i| det_f32(i + r * 331) * (r as f32 + 0.5))
                    .collect()
            })
            .collect();
        let prequant: Vec<(Vec<i8>, f32)> = raw
            .iter()
            .map(|row| decoder::quantize_row_i8_te(row))
            .collect();
        let rows: Vec<(&[i8], f32)> = prequant.iter().map(|(q, a)| (q.as_slice(), *a)).collect();
        for maybe_bias in [None, Some(bias.as_slice())] {
            let batched = decoder::gemm_i8_bias_prequant_batched(&rows, &qw, maybe_bias);
            for (r, (q, a)) in prequant.iter().enumerate() {
                let solo = decoder::gemv_i8_bias_prequant(q, *a, &qw, maybe_bias);
                assert!(
                    solo.iter()
                        .zip(&batched[r])
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                    "row {r} (bias={}) diverged",
                    maybe_bias.is_some()
                );
            }
        }
        println!("{{\"check\":\"gemm_prequant_batched_matches_m1\",\"result\":\"pass\"}}");
    }

    /// The int8 lever's refine source: untied arch ⇒ the untied head matrix,
    /// NEVER the embed table (a wrong source silently corrupts the argmax).
    #[test]
    fn head_matrix_prefers_the_untied_head() {
        let mk = |untied: Option<Vec<f32>>| GotDecodeWeights {
            layers: Vec::new(),
            final_norm: Vec::new(),
            final_norm_bias: None,
            embed_positions: None,
            embed: vec![1.0],
            untied_head: untied,
            lm_head: LmHead::F32(Mat::from_vec(1, 1, vec![0.0])),
            cfg: DecoderConfig::smolvlm2(),
        };
        assert_eq!(mk(Some(vec![2.0])).head_matrix(), &[2.0][..]);
        assert_eq!(mk(None).head_matrix(), &[1.0][..]);
    }

    /// **C5 — the SmolVLM2 decoder parity gate vs the torch oracle** (mirrors
    /// the B5 GOT gate above). Env-gated skip-with-success:
    /// `FOCR_SMOLVLM2_MODEL` = the smolvlm2 `.focrq` (int8) or the raw f32
    /// `model.safetensors`; `FOCR_SMOLVLM2_ORACLE_HIDDEN0` = the oracle's
    /// text-only-prompt `hidden_states[0]` `[N,960]` f32-LE;
    /// `FOCR_SMOLVLM2_ORACLE_LOGITS` = the oracle last-position logits
    /// `[49280]`. Fixtures come from `scripts/gen_reference_fixtures_smolvlm2.py`.
    #[test]
    fn smolvlm2_decoder_matches_torch_oracle() {
        let (Ok(model), Ok(h0), Ok(lg)) = (
            std::env::var("FOCR_SMOLVLM2_MODEL"),
            std::env::var("FOCR_SMOLVLM2_ORACLE_HIDDEN0"),
            std::env::var("FOCR_SMOLVLM2_ORACLE_LOGITS"),
        ) else {
            return;
        };
        let cfg = DecoderConfig::smolvlm2();
        let weights = Weights::load(std::path::Path::new(&model)).expect("load smolvlm2 weights");

        let h0_flat = read_f32_le(&h0);
        let n = h0_flat.len() / cfg.hidden_size;
        assert_eq!(n * cfg.hidden_size, h0_flat.len(), "hidden0 not [N,960]");
        let inputs = Mat::from_vec(n, cfg.hidden_size, h0_flat);

        let logits = forward_prefill(&weights, &cfg, &inputs).expect("prefill forward");
        assert_eq!(logits.cols, cfg.vocab_size);
        let ours = &logits.data[(logits.rows - 1) * logits.cols..];

        let oracle = read_f32_le(&lg);
        let oracle = &oracle[oracle.len() - cfg.vocab_size..];

        assert_eq!(
            argmax(ours),
            argmax(oracle),
            "next-token argmax diverged from the torch oracle"
        );
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
            "[C5 parity] argmax={} cos={cos:.6} (oracle argmax={})",
            argmax(ours),
            argmax(oracle)
        );
        assert!(
            cos >= 0.99,
            "logit cosine {cos:.6} < 0.99 — smolvlm2 decoder diverged"
        );
    }

    /// **C5 L4 — SmolVLM2 greedy decode certification**, precision-honest
    /// (mirrors what ACTUALLY held for GOT):
    ///
    /// * given the **f32 reference** (`model.safetensors`): the O(n²)
    ///   [`generate_greedy`] path (f32 GEMMs via [`linear_auto`]) must match
    ///   the f32 oracle's greedy id-stream EXACTLY — apples-to-apples;
    /// * given the **int8 artifact** (`.focrq`): [`generate_greedy_kvcache`]
    ///   must match [`generate_greedy`] on the SAME weights EXACTLY (both
    ///   decode int8 — the B9 bit-identity contract). The int8-vs-f32-oracle
    ///   stream may legitimately flip a near-tied token (int8 logit
    ///   cos ≈ 0.998, measured flip at step 7 on this prompt: "Paris"/"It",
    ///   both coherent) — that cross-precision delta is DISCREPANCIES
    ///   territory, not an exact gate.
    ///
    /// Expected ids come from the COMMITTED
    /// `tests/fixtures/smolvlm2/oracle_fixtures.json` (`l4_greedy.ids`), read
    /// at runtime so the cert self-arms once the oracle has been generated;
    /// skips-with-success while any artifact is absent.
    #[test]
    fn smolvlm2_kvcache_greedy_matches_oracle_l4() {
        let (Ok(model), Ok(h0)) = (
            std::env::var("FOCR_SMOLVLM2_MODEL"),
            std::env::var("FOCR_SMOLVLM2_ORACLE_HIDDEN0"),
        ) else {
            return;
        };
        let fixture_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/smolvlm2/oracle_fixtures.json"
        );
        let Ok(raw) = std::fs::read_to_string(fixture_path) else {
            eprintln!("skip-with-SUCCESS: {fixture_path} absent (oracle not yet generated)");
            return;
        };
        // Minimal, dependency-free extraction of l4_greedy.ids from the fixture.
        let ids_json = raw
            .split("\"l4_greedy\"")
            .nth(1)
            .and_then(|s| s.split("\"ids\"").nth(1))
            .and_then(|s| s.split('[').nth(1))
            .and_then(|s| s.split(']').next())
            .expect("l4_greedy.ids present in the fixture");
        let expected: Vec<u32> = ids_json
            .split(',')
            .map(|t| t.trim().parse().expect("id parses"))
            .collect();
        assert!(!expected.is_empty(), "fixture carries a greedy stream");

        let cfg = DecoderConfig::smolvlm2();
        let weights = Weights::load(std::path::Path::new(&model)).expect("load smolvlm2 weights");
        let h0_flat = read_f32_le(&h0);
        let n = h0_flat.len() / cfg.hidden_size;
        let inputs = Mat::from_vec(n, cfg.hidden_size, h0_flat);

        if model.ends_with(".focrq") {
            // int8 artifact: kvcache decode == the O(n²) re-prefill oracle on
            // the SAME int8 weights (the B9 bit-identity contract).
            let slow = generate_greedy(&weights, &cfg, &inputs, expected.len(), 49_279)
                .expect("re-prefill greedy decode");
            let fast = generate_greedy_kvcache(&weights, &cfg, &inputs, expected.len(), 49_279)
                .expect("kvcache greedy decode");
            assert_eq!(
                fast, slow,
                "smolvlm2 kvcache greedy != re-prefill greedy on the same int8 weights"
            );
            eprintln!("[C5 L4/int8] kvcache == re-prefill, {} ids", fast.len());
        } else {
            // f32 reference: the f32 decode must reproduce the f32 oracle's
            // greedy stream token-for-token.
            let ids = generate_greedy(&weights, &cfg, &inputs, expected.len(), 49_279)
                .expect("f32 greedy decode");
            assert_eq!(
                ids, expected,
                "smolvlm2 f32 greedy id-stream != torch oracle L4"
            );
            eprintln!("[C5 L4/f32] {} ids exact vs oracle", ids.len());
        }
    }

    #[test]
    fn gqa_dims_and_grouping() {
        let c = DecoderConfig::smolvlm2();
        assert_eq!(c.q_dim(), 960);
        assert_eq!(c.kv_dim(), 320);
        assert_eq!(c.kv_group(), 3);
    }

    #[test]
    fn broadcast_kv_repeats_each_kv_lane_group_times() {
        let mut c = DecoderConfig::smolvlm2();
        // tiny: 4 q heads over 2 kv heads, head_dim 2 → group 2.
        c.num_attention_heads = 4;
        c.num_key_value_heads = 2;
        c.head_dim = 2;
        let kv = Mat::from_vec(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let full = broadcast_kv(&kv, &c).expect("broadcast");
        // row 0: kv heads [1,2] and [3,4] → q lanes [1,2],[1,2],[3,4],[3,4].
        assert_eq!(full.data[0..8], [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
        assert_eq!(full.data[8..16], [5.0, 6.0, 5.0, 6.0, 7.0, 8.0, 7.0, 8.0]);
    }

    /// The decode path's NATIVE GQA head mapping (index arithmetic over the
    /// kv_dim-stride cache) must equal the independent broadcast-to-MHA
    /// computation bit-for-bit — two different implementations of the same
    /// math, so a mapping bug in either cannot self-confirm.
    #[test]
    fn gqa_decode_attention_matches_broadcast_mha_reference() {
        let (num_heads, kv_heads, head_dim, n_kv) = (6usize, 2usize, 4usize, 5usize);
        let group = num_heads / kv_heads;
        let (q_dim, kv_dim) = (num_heads * head_dim, kv_heads * head_dim);
        let f = |i: usize, salt: usize| (((i * 31 + salt * 17) % 97) as f32) * 0.03 - 1.2;

        // GQA-native cache [n_kv, kv_dim] + query row [q_dim].
        let mut cache = Qwen2KvCache::new(kv_dim, n_kv);
        let k: Vec<f32> = (0..n_kv * kv_dim).map(|i| f(i, 1)).collect();
        let v: Vec<f32> = (0..n_kv * kv_dim).map(|i| f(i, 2)).collect();
        cache.seed(&k, &v);
        let q_row: Vec<f32> = (0..q_dim).map(|i| f(i, 3)).collect();
        let native = qwen2_decode_attention(&cache, &q_row, num_heads, head_dim, group);

        // Independent reference: broadcast K/V to the full MHA layout, then run
        // the SAME attention with kv_group == 1.
        let mut bcast = Qwen2KvCache::new(q_dim, n_kv);
        let expand = |src: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; n_kv * q_dim];
            for r in 0..n_kv {
                for h in 0..num_heads {
                    let s = r * kv_dim + (h / group) * head_dim;
                    let d = r * q_dim + h * head_dim;
                    out[d..d + head_dim].copy_from_slice(&src[s..s + head_dim]);
                }
            }
            out
        };
        bcast.seed(&expand(&k), &expand(&v));
        let reference = qwen2_decode_attention(&bcast, &q_row, num_heads, head_dim, 1);

        assert_eq!(native.len(), reference.len());
        for (i, (a, b)) in native.iter().zip(&reference).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "GQA native decode != broadcast-MHA reference at {i}"
            );
        }
    }

    #[test]
    fn split_qkv_rows_handles_unequal_gqa_panels() {
        // 2 rows of [q_dim=4 | kv_dim=2 | kv_dim=2].
        let fused = Mat::from_vec(
            2,
            8,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            ],
        );
        let (q, k, v) = split_qkv_rows(&fused, 4, 2);
        assert_eq!((q.rows, q.cols), (2, 4));
        assert_eq!((k.rows, k.cols), (2, 2));
        assert_eq!((v.rows, v.cols), (2, 2));
        assert_eq!(q.data, vec![1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0]);
        assert_eq!(k.data, vec![5.0, 6.0, 13.0, 14.0]);
        assert_eq!(v.data, vec![7.0, 8.0, 15.0, 16.0]);
    }

    #[test]
    fn no_repeat_guard_bans_the_cycle_completion() {
        // ids end with the prefix [7, 8]; the 3-gram [7, 8, 9] already occurred, so
        // token 9 must be banned even though its logit is the max — the pick falls
        // to the runner-up (token 2).
        let ids = [7u32, 8, 9, 1, 7, 8];
        let mut logits = vec![0.0f32; 10];
        logits[9] = 5.0;
        logits[2] = 4.0;
        assert_eq!(argmax_no_repeat(&logits, &ids, 3), 2);
        // the plain argmax (guard disabled) still picks the banned token.
        assert_eq!(argmax_no_repeat(&logits, &ids, 0), 9);
    }

    #[test]
    fn no_repeat_guard_is_identity_on_a_clean_stream() {
        // no 3-gram repeats anywhere: the guard must not perturb the argmax.
        let ids = [1u32, 2, 3, 4, 5, 6];
        let mut logits = vec![0.0f32; 10];
        logits[7] = 3.0;
        assert_eq!(argmax_no_repeat(&logits, &ids, 3), argmax(&logits));
        // too short to hold a single n-gram: also identity.
        assert_eq!(argmax_no_repeat(&logits, &ids[..2], 3), argmax(&logits));
        // empty stream (the seed pick): identity.
        assert_eq!(argmax_no_repeat(&logits, &[], 3), argmax(&logits));
    }

    /// Read a raw little-endian f32 blob.
    fn read_f32_le(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).expect("oracle blob");
        assert_eq!(bytes.len() % 4, 0, "not a whole f32 count");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
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
