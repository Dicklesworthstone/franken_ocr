//! Model-architecture descriptors — the foundation of the multi-model "model zoo"
//! (epic `bd-3jo6`, task A1).
//!
//! This is the FIRST, deliberately **additive** step of the multi-model
//! generalization: it describes each model's identity + graph shape + decode
//! contract + tokenizer + tasks as plain data behind a [`ModelArch`] trait, and a
//! [`registry`] that maps a model id to its descriptor — WITHOUT yet rewiring the
//! live forward. So the Baidu Unlimited-OCR engine stays byte-identical (its
//! goldens, CER, and `robot selftest` are unchanged); nothing here is on the hot
//! path. Later foundation tasks (A2 `.focrq` v2 + registry, A3 `convert`, A4
//! `pull`, A5 the multi-task CLI, A6 tokenizers, A7 the shared decoder) dispatch
//! real behavior through this descriptor.
//!
//! The Unlimited-OCR descriptor's values are asserted against the live engine
//! constants (`FOCR_MODEL_LICENSE_NOTICE`, `DEFAULT_MODEL_PATH`,
//! `sampler::DecodeParams::default()`) in the unit tests, so the description can
//! never silently drift from the real model.

/// The vision-encoder family a model uses. Only [`VisionEncoder::SamClip`] (the
/// Baidu Unlimited-OCR tower) is implemented today; the rest are the zoo targets,
/// declared so the registry + `focr models` can describe them before they land.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VisionEncoder {
    /// SAM-ViT-B → 16× conv token-compressor → CLIP-L/14 (Baidu Unlimited-OCR).
    SamClip,
    /// SAM / VitDet-style ViT (GOT-OCR2.0, OneChart). [planned: B3/D3]
    SamVit,
    /// SigLIP ViT (SmolVLM2). [planned: C3]
    Siglip,
    /// ResNet stem + ViT (Polyphonic-TrOMR, pix2tex). [planned: E3/F2]
    ResNetVit,
    /// BEiT / DeiT ViT (TrOCR). [planned: F1]
    BeitVit,
}

/// The autoregressive-decoder family a model uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Decoder {
    /// DeepSeek-V2 MoE (64 routed + 2 shared experts, top-6) with R-SWA
    /// (Reference Sliding Window Attention, window 128) — the Baidu decoder.
    DeepSeekV2MoeRswa,
    /// Qwen2 dense (GOT-OCR2.0). [shared engine A7]
    Qwen2Dense,
    /// SmolLM2 / Llama-style dense (SmolVLM2). [A7]
    LlamaDense,
    /// OPT-125M dense (OneChart, D-census §4: pre-LN, ReLU fc1/fc2, learned
    /// absolute positions offset-2, NO RoPE, all-linears-biased, tied head).
    /// [planned: D4 on A7]
    OptDense,
    /// mBART / RoBERTa-style seq2seq decoder (TrOCR, pix2tex). [planned: A7]
    Seq2SeqDense,
}

/// The tokenizer family a model uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TokenizerKind {
    /// DeepSeek BPE over the committed `tokenizer.json` (Baidu Unlimited-OCR).
    DeepSeekBpe,
    /// Qwen2 BPE (GOT-OCR2.0). [A6/B6]
    Qwen2Bpe,
    /// SmolLM2 BPE (SmolVLM2). [A6/C6]
    SmolLm2Bpe,
    /// OPT / GPT-2 byte-level BPE over `vocab.json`+`merges.txt` (OneChart —
    /// D-census §7: NOT tiktoken, NOT Qwen). [planned: D9]
    Gpt2Bpe,
    /// A music-symbol vocabulary (Polyphonic-TrOMR). [planned: A6/E6]
    MusicVocab,
    /// SentencePiece / BART (TrOCR, pix2tex). [planned: A6/F]
    SentencePiece,
}

/// A task a model can serve — the routing key for the multi-task CLI (A5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Task {
    /// Document text → markdown (the default `focr ocr`).
    Ocr,
    /// Math / formula → LaTeX.
    Formula,
    /// Tables → structured markdown / TEDS.
    Tables,
    /// Chart / graph → structured data.
    Chart,
    /// Molecular structure → SMILES.
    Molecular,
    /// Geometry → tikz.
    Geometry,
    /// Sheet music → MusicXML / **kern.
    Music,
    /// Photo description / captioning.
    Describe,
    /// Visual question answering.
    Vqa,
    /// Handwriting recognition.
    Handwriting,
}

/// The greedy-decode contract a model ships with — a plain-data mirror of the
/// load-bearing knobs in [`sampler::DecodeParams`] (so a descriptor needs no
/// runtime). Asserted equal to the live `DecodeParams::default()` in tests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeContract {
    /// Sampling temperature; `0.0` ⇒ greedy argmax.
    pub temperature: f32,
    /// EOS token id.
    pub eos_token_id: u32,
    /// No-repeat n-gram size; `0` disables.
    pub no_repeat_ngram_size: usize,
    /// Sliding window for the no-repeat n-gram processor.
    pub ngram_window: usize,
}

/// The quantization policy (doctrine #2): which tensor groups are int8 vs kept
/// high-precision. Every franken_ocr arch shares the SAME validated policy —
/// decoder GEMMs int8, the entire vision tower + projector + router + embed + ALL
/// norms high-precision — but each declares it explicitly so `focr convert` (A3)
/// is data-driven rather than hardcoded per model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantPolicy {
    /// Decoder FFN / expert / projection GEMMs are int8.
    pub decoder_gemms_int8: bool,
    /// Vision tower, projector, router gate, `embed_tokens`, and ALL norms stay
    /// BF16/F32 (quantizing the vision path wrecks accuracy — CER 0.37).
    pub vision_high_precision: bool,
    /// `lm_head` int8 is BEYOND the validated set — it ships only behind a
    /// measured-CER kill-switch (doctrine #2 / OQ-14), never assumed lossless.
    pub lm_head_int8_killswitched: bool,
}

impl QuantPolicy {
    /// The fixed, validated franken_ocr quant policy (doctrine #2). Every arch
    /// uses this shape.
    pub const DOCTRINE: Self = Self {
        decoder_gemms_int8: true,
        vision_high_precision: true,
        lm_head_int8_killswitched: true,
    };
}

/// A model-architecture descriptor — the foundation `ModelArch` of the zoo. A
/// descriptor is pure metadata (no weights, no runtime); the forward dispatch is
/// wired through it in later A-tasks. `Send + Sync` so a `&'static dyn ModelArch`
/// can live in the process-global registry.
pub trait ModelArch: Send + Sync {
    /// Stable model id — the cache subdir, the `--model` value, and the manifest
    /// key, e.g. `"unlimited-ocr"`.
    fn id(&self) -> &'static str;
    /// Human display name, e.g. `"Baidu Unlimited-OCR"`.
    fn display_name(&self) -> &'static str;
    /// The redistribution / license notice that must travel with the model's
    /// `.focrq` and appear on agent-facing provenance surfaces.
    fn license_notice(&self) -> &'static str;
    /// The default `.focrq` basename in the cache, e.g. `"unlimited-ocr.focrq"`.
    fn default_artifact_basename(&self) -> &'static str;
    /// The vision-encoder family.
    fn vision_encoder(&self) -> VisionEncoder;
    /// The decoder family.
    fn decoder(&self) -> Decoder;
    /// The tokenizer family.
    fn tokenizer(&self) -> TokenizerKind;
    /// The quant policy (defaults to the doctrine policy; every arch uses it).
    fn quant_policy(&self) -> QuantPolicy {
        QuantPolicy::DOCTRINE
    }
    /// The greedy-decode contract this model ships with.
    fn decode_contract(&self) -> DecodeContract;
    /// The tasks this model serves (the CLI routes these to this arch).
    fn tasks(&self) -> &'static [Task];
    /// Whether this arch's forward is implemented TODAY (vs a planned zoo target
    /// the registry merely describes). Only Unlimited-OCR is `true` so far.
    fn implemented(&self) -> bool;
    /// Whether `lm_head.weight` is a tied byte-alias of `model.embed_tokens.weight`
    /// and should therefore be **omitted** from the `.focrq` (the loader reuses the
    /// stored embedding for the output projection). `false` (the default) means the
    /// converter stores `lm_head` as its own tensor, as Unlimited-OCR does. GOT-OCR2
    /// ties (verified byte-identical — `docs/zoo/got-ocr2-spec.md` §12).
    fn tie_word_embeddings(&self) -> bool {
        false
    }
    /// The `.focrq` tensor-name prefix for this arch's SAM-family vision tower.
    /// Baidu Unlimited-OCR (default) uses `model.sam_model`; GOT-OCR2 uses
    /// `model.vision_tower_high` (identical leaf names + geometry, different prefix).
    fn vision_tower_prefix(&self) -> &'static str {
        "model.sam_model"
    }
    /// The tensor-name prefix of this arch's autoregressive-decoder transformer
    /// layers — the subtree `focr convert` classifies for the doctrine-#2 int8
    /// GEMM set. Baidu Unlimited-OCR + GOT-OCR2 put the LM at the model top
    /// (`model.layers.`); SmolVLM2's Idefics3 splice nests it under
    /// `model.text_model.layers.` (census: `docs/zoo/smolvlm2-spec.md` §12).
    /// Making the prefix an arch fact keeps the converter's classification
    /// arch-aware instead of name-coincidence: a SigLIP vision block named
    /// `model.vision_model.encoder.layers.{i}.self_attn.q_proj.weight` must
    /// NEVER match the decoder int8 set.
    fn decoder_layers_prefix(&self) -> &'static str {
        "model.layers."
    }
    /// When `lm_head.weight` is stored (i.e. NOT tied/omitted): does the
    /// converter quantize it int8? Default `false`: Unlimited-OCR and SmolVLM2
    /// keep their untied heads high precision per doctrine #2. An architecture
    /// may opt in only after its independent quality gate is certified.
    fn lm_head_stored_int8(&self) -> bool {
        false
    }
    /// The token-embedding tensor name — the untied-`lm_head` convert-time
    /// verification compares `lm_head.weight` bytes against this tensor.
    /// Default `model.embed_tokens.weight`; SmolVLM2 nests it under
    /// `model.text_model.embed_tokens.weight`.
    fn embed_tokens_name(&self) -> &'static str {
        "model.embed_tokens.weight"
    }
}

/// The Baidu Unlimited-OCR architecture — the FIRST [`ModelArch`] implementation
/// and the project default (`focr ocr`). Its descriptor values are asserted
/// against the live engine constants in tests, so the description cannot drift.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnlimitedOcr;

impl ModelArch for UnlimitedOcr {
    fn id(&self) -> &'static str {
        "unlimited-ocr"
    }
    fn display_name(&self) -> &'static str {
        "Baidu Unlimited-OCR"
    }
    fn license_notice(&self) -> &'static str {
        crate::FOCR_MODEL_LICENSE_NOTICE
    }
    fn default_artifact_basename(&self) -> &'static str {
        "unlimited-ocr.focrq"
    }
    fn vision_encoder(&self) -> VisionEncoder {
        VisionEncoder::SamClip
    }
    fn decoder(&self) -> Decoder {
        Decoder::DeepSeekV2MoeRswa
    }
    fn tokenizer(&self) -> TokenizerKind {
        TokenizerKind::DeepSeekBpe
    }
    fn decode_contract(&self) -> DecodeContract {
        // The frozen single-image greedy contract (plan §6.10 / SPEC-100..103);
        // asserted equal to `sampler::DecodeParams::default()` in the tests.
        DecodeContract {
            temperature: 0.0,
            eos_token_id: 1,
            no_repeat_ngram_size: 35,
            ngram_window: 128,
        }
    }
    fn tasks(&self) -> &'static [Task] {
        &[Task::Ocr]
    }
    fn implemented(&self) -> bool {
        true
    }
}

/// The one process-global Unlimited-OCR descriptor instance.
static UNLIMITED_OCR: UnlimitedOcr = UnlimitedOcr;

/// A data-driven [`ModelArch`] for a PLANNED zoo model (epic bd-3jo6): described in
/// the registry — so `focr models` shows the roadmap — but with `implemented()` =
/// false until its forward lands. The graph-shape fields are confidently known
/// from the model survey; the exact decode contract + per-config values are filled
/// in (and the type upgraded to a real impl like [`UnlimitedOcr`]) when the model's
/// sub-epic ships. For a planned model the decode contract is informational and is
/// never used for inference.
pub struct PlannedArch {
    id: &'static str,
    display_name: &'static str,
    license_notice: &'static str,
    default_artifact_basename: &'static str,
    vision_encoder: VisionEncoder,
    decoder: Decoder,
    tokenizer: TokenizerKind,
    /// The greedy-decode contract, where known from the model's config/census
    /// (e.g. GOT-OCR2 from `docs/zoo/got-ocr2-spec.md`); `PLACEHOLDER_CONTRACT`
    /// until censused. Informational for a planned model (never used for inference).
    decode_contract: DecodeContract,
    tasks: &'static [Task],
    /// Whether the model ties `lm_head` to `embed_tokens` (so the converter omits
    /// `lm_head.weight`); see [`ModelArch::tie_word_embeddings`].
    tie_word_embeddings: bool,
    /// The SAM-family vision tower `.focrq` tensor-name prefix; see
    /// [`ModelArch::vision_tower_prefix`].
    vision_tower_prefix: &'static str,
    /// The AR-decoder transformer-layers tensor-name prefix (the int8 GEMM
    /// subtree); see [`ModelArch::decoder_layers_prefix`].
    decoder_layers_prefix: &'static str,
    /// Whether a STORED (untied) `lm_head.weight` joins this architecture's
    /// separately certified int8 set; see [`ModelArch::lm_head_stored_int8`].
    lm_head_stored_int8: bool,
    /// The token-embedding tensor name; see [`ModelArch::embed_tokens_name`].
    embed_tokens_name: &'static str,
    /// Whether this arch's forward RUNS today (`focr models` "ready"). GOT-OCR2 does
    /// (its full pipeline + KV-cache decode ship); the rest are still descriptors.
    implemented: bool,
}

/// A not-yet-determined greedy contract for a planned model whose config has not
/// been censused (temperature 0 = greedy; the rest zeroed).
const PLACEHOLDER_CONTRACT: DecodeContract = DecodeContract {
    temperature: 0.0,
    eos_token_id: 0,
    no_repeat_ngram_size: 0,
    ngram_window: 0,
};

impl ModelArch for PlannedArch {
    fn id(&self) -> &'static str {
        self.id
    }
    fn display_name(&self) -> &'static str {
        self.display_name
    }
    fn license_notice(&self) -> &'static str {
        self.license_notice
    }
    fn default_artifact_basename(&self) -> &'static str {
        self.default_artifact_basename
    }
    fn vision_encoder(&self) -> VisionEncoder {
        self.vision_encoder
    }
    fn decoder(&self) -> Decoder {
        self.decoder
    }
    fn tokenizer(&self) -> TokenizerKind {
        self.tokenizer
    }
    fn decode_contract(&self) -> DecodeContract {
        self.decode_contract
    }
    fn tasks(&self) -> &'static [Task] {
        self.tasks
    }
    fn implemented(&self) -> bool {
        self.implemented
    }
    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
    fn vision_tower_prefix(&self) -> &'static str {
        self.vision_tower_prefix
    }
    fn decoder_layers_prefix(&self) -> &'static str {
        self.decoder_layers_prefix
    }
    fn lm_head_stored_int8(&self) -> bool {
        self.lm_head_stored_int8
    }
    fn embed_tokens_name(&self) -> &'static str {
        self.embed_tokens_name
    }
}

// ── the planned zoo models (descriptors only; forwards land in sub-epics B-F) ──
// GOT-OCR2.0 — censused (docs/zoo/got-ocr2-spec.md, bd-3jo6.2.1): SAM-ViT-B encoder
// + Linear(1024→1024) connector + Qwen1.5/Qwen2-arch 0.5B DENSE decoder; the
// tokenizer is the original Qwen tiktoken BPE (qwen.tiktoken), not a HF tokenizer.json.
static GOT_OCR2: PlannedArch = PlannedArch {
    id: "got-ocr2",
    display_name: "GOT-OCR2.0",
    license_notice: "GOT-OCR2.0 (StepFun) - Apache-2.0",
    default_artifact_basename: "got-ocr2.focrq",
    vision_encoder: VisionEncoder::SamVit,
    decoder: Decoder::Qwen2Dense,
    tokenizer: TokenizerKind::Qwen2Bpe,
    // Real config (census): greedy, eos=151643 (<|endoftext|>), no_repeat_ngram=20
    // (HF builtin/global, window 0), max_new 4096, stop "<|im_end|>".
    decode_contract: DecodeContract {
        temperature: 0.0,
        eos_token_id: 151_643,
        no_repeat_ngram_size: 20,
        ngram_window: 0,
    },
    tasks: &[
        Task::Ocr,
        Task::Formula,
        Task::Tables,
        Task::Chart,
        Task::Molecular,
        Task::Geometry,
        Task::Music,
    ],
    // Verified byte-identical lm_head == embed_tokens (spec §12) → omit lm_head.
    tie_word_embeddings: true,
    vision_tower_prefix: "model.vision_tower_high",
    decoder_layers_prefix: "model.layers.", // Qwen2 LM at the model top (spec §13a)
    lm_head_stored_int8: true,              // moot — tied lm_head is omitted entirely
    embed_tokens_name: "model.embed_tokens.weight",
    implemented: true, // full pipeline + KV-cache decode ship (B1-B9,B11); focr ocr runs it
};
// SmolVLM2-500M — censused (docs/zoo/smolvlm2-spec.md, bd-3jo6.3.1): SigLIP-B/16@512²
// encoder + pixel-shuffle×4 + Linear(12288→960) connector + SmolLM2-360M Llama-dense
// decoder (32L, hidden 960, GQA 15q/5kv, UNTIED lm_head — byte-verified, spec §12).
// The Idefics3 splice nests the LM under `model.text_model.*` and the SigLIP tower
// under `model.vision_model.*` (exact tensor names: spec §12).
static SMOLVLM2: PlannedArch = PlannedArch {
    id: "smolvlm2",
    display_name: "SmolVLM2-500M",
    license_notice: "SmolVLM2-500M-Video-Instruct (HuggingFaceTB) - Apache-2.0",
    default_artifact_basename: "smolvlm2.focrq",
    vision_encoder: VisionEncoder::Siglip,
    decoder: Decoder::LlamaDense,
    tokenizer: TokenizerKind::SmolLm2Bpe,
    // Real config (census §8): greedy, eos=49279 (<end_of_utterance>), and NO
    // repetition guard upstream (`no_repeat_ngram_size` absent ⇒ 0, window 0).
    decode_contract: DecodeContract {
        temperature: 0.0,
        eos_token_id: 49_279,
        no_repeat_ngram_size: 0,
        ngram_window: 0,
    },
    tasks: &[Task::Describe, Task::Vqa],
    // Censused UNTIED (top-level `tie_word_embeddings: false`; lm_head vs
    // embed_tokens byte-verified DISTINCT — spec §12). The converter stores BOTH.
    tie_word_embeddings: false,
    vision_tower_prefix: "model.vision_model", // SigLIP tower (census §12)
    decoder_layers_prefix: "model.text_model.layers.", // Idefics3-nested LM (census §12)
    // Doctrine #2 / spec §11 (OQ-6): the untied lm_head [49280,960] stays
    // high-precision; int8-lm_head only behind a measured quality kill-switch.
    lm_head_stored_int8: false,
    embed_tokens_name: "model.text_model.embed_tokens.weight",
    // C1-C9 shipped: convert (C2), decoder (C5), tokenizer (C6), SigLIP +
    // pixel-shuffle connector (C3/C4, cert cos 1.0), prompt/preprocess (C7,
    // L0b maxabs 0.0 + L0c id-exact), `--task describe` routing (C9).
    implemented: true,
};
static ONECHART: PlannedArch = PlannedArch {
    id: "onechart",
    display_name: "OneChart",
    license_notice: "OneChart (kppkkp) - Apache-2.0",
    default_artifact_basename: "onechart.focrq",
    // CENSUSED (docs/zoo/onechart-spec.md, D1): SAM-ViT-B tower (GOT/Vary
    // lineage, prefix `model.vision_tower.` — NOT `vision_tower_high`), a
    // Linear(1024→768,bias) projector, an OPT-125M decoder (§4: pre-LN, ReLU
    // fc1/fc2, learned absolute positions offset-2, NO RoPE, MHA 12/12, all
    // linears biased), and the novel `num_decoder` number head (§8).
    vision_encoder: VisionEncoder::SamVit,
    decoder: Decoder::OptDense,
    tokenizer: TokenizerKind::Gpt2Bpe,
    // §10: greedy, eos 2 (`</s>`), NO upstream repetition guard; hard cap
    // total-seq ≤ 4096 (the learned position table, §4/OQ-D7).
    decode_contract: DecodeContract {
        temperature: 0.0,
        eos_token_id: 2,
        no_repeat_ngram_size: 0,
        ngram_window: 0,
    },
    tasks: &[Task::Chart],
    // §4: `lm_head.weight` and `model.decoder.embed_tokens.weight` are stored
    // byte-identical (SHA-proven) — TIED; the `.focrq` keeps ONE copy (the
    // GOT precedent; convert byte-verifies the tie before dropping).
    tie_word_embeddings: true,
    vision_tower_prefix: "model.vision_tower",
    decoder_layers_prefix: "model.decoder.layers.",
    // Tied head via embed_tokens stays high-precision until a measured
    // kill-switched int8 lever lands (doctrine #2).
    lm_head_stored_int8: false,
    embed_tokens_name: "model.decoder.embed_tokens.weight",
    // D1-D9 shipped: convert (D2), tokenizer (D9), vision (D3), OPT decoder
    // both paths (D4), number head + self-verify + recognize (D5),
    // `--task chart-data` routing (D7).
    implemented: true,
};
static TROMR: PlannedArch = PlannedArch {
    id: "tromr",
    display_name: "Polyphonic-TrOMR",
    license_notice: "Polyphonic-TrOMR (NetEase) - Apache-2.0",
    default_artifact_basename: "tromr.focrq",
    vision_encoder: VisionEncoder::ResNetVit,
    decoder: Decoder::Seq2SeqDense,
    tokenizer: TokenizerKind::MusicVocab,
    // CENSUSED (docs/zoo/tromr-spec.md §5/§9, E2): greedy argmax-forced (the
    // E8 oracle plan; upstream's top-k sampling is the OQ-T6 kill-switch),
    // stop on the RHYTHM stream's `[EOS]` = 2 (pad 0, bos 1, max_seq_len 256).
    decode_contract: DecodeContract {
        temperature: 0.0,
        eos_token_id: 2,
        no_repeat_ngram_size: 0,
        ngram_window: 0,
    },
    tasks: &[Task::Music],
    // §12: NO `lm_head.weight` at all — FOUR untied per-stream heads
    // (`decoder.net.to_logits_{lift,pitch,rhythm,note}`), each Linear+bias.
    tie_word_embeddings: false,
    // §12: the whole tower (ResNetV2 stem + hybrid ViT) lives under `encoder.`
    // (WS is FOLDED into the stored conv weights at export — no runtime WS).
    vision_tower_prefix: "encoder.",
    // §12: flat x-transformers layout, `i%3` ⇒ 0=self-attn 1=cross-attn 2=ff.
    decoder_layers_prefix: "decoder.net.attn_layers.layers.",
    // §11: heads are tiny ([260/71/7/2]×256 Linear+bias) — always HP; the 40
    // decoder GEMMs are int8 CANDIDATES behind a measured-lossless gate only
    // (default f32; see quant::convert::is_decoder_int8_tensor_for).
    lm_head_stored_int8: false,
    // §12: THREE input embeddings summed per step (lift/pitch/rhythm) — no
    // single embed_tokens. The rhythm stream is primary (drives EOS); the
    // untie-verify never fires (no `lm_head.weight` exists).
    embed_tokens_name: "decoder.net.rhythm_emb.emb.weight",
    // E2-E7+E9 shipped: convert (E2), tokenizer (E6), encoder cos 1.0 (E3),
    // decoder token-exact (E4), merge+MusicXML (E7), `--task music` routing.
    implemented: true,
};
static TROCR: PlannedArch = PlannedArch {
    id: "trocr",
    display_name: "TrOCR",
    license_notice: "TrOCR (Microsoft) - MIT",
    default_artifact_basename: "trocr.focrq",
    vision_encoder: VisionEncoder::BeitVit,
    decoder: Decoder::Seq2SeqDense,
    tokenizer: TokenizerKind::SentencePiece,
    decode_contract: PLACEHOLDER_CONTRACT,
    tasks: &[Task::Handwriting],
    tie_word_embeddings: false, // placeholder until censused (sub-epic F)
    vision_tower_prefix: "model.sam_model", // BeiT/ResNet; placeholder
    decoder_layers_prefix: "model.layers.", // placeholder until censused (sub-epic F)
    lm_head_stored_int8: true,  // placeholder until censused (sub-epic F)
    embed_tokens_name: "model.embed_tokens.weight",
    implemented: false,
};
static PIX2TEX: PlannedArch = PlannedArch {
    id: "pix2tex",
    display_name: "pix2tex (LaTeX-OCR)",
    license_notice: "pix2tex / LaTeX-OCR - MIT",
    default_artifact_basename: "pix2tex.focrq",
    vision_encoder: VisionEncoder::ResNetVit,
    decoder: Decoder::Seq2SeqDense,
    tokenizer: TokenizerKind::SentencePiece,
    decode_contract: PLACEHOLDER_CONTRACT,
    tasks: &[Task::Formula],
    tie_word_embeddings: false, // placeholder until censused (sub-epic F)
    vision_tower_prefix: "model.sam_model", // BeiT/ResNet; placeholder
    decoder_layers_prefix: "model.layers.", // placeholder until censused (sub-epic F)
    lm_head_stored_int8: true,  // placeholder until censused (sub-epic F)
    embed_tokens_name: "model.embed_tokens.weight",
    implemented: false,
};

/// The model registry, in priority order (the default + implemented first, then the
/// planned zoo models). This is the single place models register. Today exactly one
/// is IMPLEMENTED (Unlimited-OCR); the rest are planned descriptors (epic bd-3jo6)
/// shown by `focr models` and upgraded to real impls as their sub-epics land.
static REGISTRY: &[&dyn ModelArch] = &[
    &UNLIMITED_OCR,
    &GOT_OCR2,
    &SMOLVLM2,
    &ONECHART,
    &TROMR,
    &TROCR,
    &PIX2TEX,
];

/// The model registry slice (see [`static@REGISTRY`]).
#[must_use]
pub fn registry() -> &'static [&'static dyn ModelArch] {
    REGISTRY
}

/// Look up an architecture by its stable id; `None` if unknown.
#[must_use]
pub fn arch_by_id(id: &str) -> Option<&'static dyn ModelArch> {
    registry().iter().copied().find(|a| a.id() == id)
}

/// The default architecture (`focr ocr`): the Baidu Unlimited-OCR model.
#[must_use]
pub fn default_arch() -> &'static dyn ModelArch {
    &UNLIMITED_OCR
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_engine::sampler;
    use crate::{DEFAULT_MODEL_PATH, FOCR_MODEL_LICENSE_NOTICE};

    #[test]
    fn registry_lists_the_default_first_then_the_planned_zoo() {
        let archs = registry();
        // The default (Unlimited-OCR) is first + implemented.
        assert_eq!(archs[0].id(), "unlimited-ocr");
        assert!(archs[0].implemented());
        // The IMPLEMENTED set: Unlimited-OCR (fast plain OCR) + GOT-OCR2
        // (specialized structured OCR) + SmolVLM2 (photo describe/VQA, C1-C9)
        // — all run today via `focr ocr`.
        let mut implemented: Vec<&str> = archs
            .iter()
            .filter(|a| a.implemented())
            .map(|a| a.id())
            .collect();
        implemented.sort_unstable();
        assert_eq!(
            implemented,
            ["got-ocr2", "onechart", "smolvlm2", "tromr", "unlimited-ocr"]
        );
        // The remaining zoo models are present but NOT yet implemented (descriptors).
        for id in ["trocr", "pix2tex"] {
            let a = arch_by_id(id).unwrap_or_else(|| unreachable!("planned arch {id} registered"));
            assert!(!a.implemented(), "{id} is planned, not implemented");
        }
        // Every registered id is unique (a registry invariant the zoo relies on).
        let mut ids: Vec<&str> = archs.iter().map(|a| a.id()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), archs.len(), "model ids must be unique");
    }

    #[test]
    fn tromr_descriptor_is_the_censused_shape() {
        // E2 (bd-3jo6.5.2): the descriptor carries the tromr-spec §5/§9/§11/§12
        // census, not placeholders. Convert-critical fields first.
        let a = arch_by_id("tromr").expect("tromr registered");
        assert!(a.implemented(), "E2-E9 shipped (model #5)");
        assert_eq!(a.decoder(), Decoder::Seq2SeqDense);
        assert_eq!(a.tokenizer(), TokenizerKind::MusicVocab);
        // §12: no lm_head.weight exists — four untied per-stream heads; the
        // artifact stores everything high-precision by default (§11).
        assert!(
            !a.tie_word_embeddings(),
            "4 untied heads, no lm_head at all"
        );
        assert!(
            !a.lm_head_stored_int8(),
            "heads are tiny Linear+bias — always HP"
        );
        assert_eq!(a.vision_tower_prefix(), "encoder.");
        assert_eq!(a.decoder_layers_prefix(), "decoder.net.attn_layers.layers.");
        assert_eq!(a.embed_tokens_name(), "decoder.net.rhythm_emb.emb.weight");
        // §9: greedy argmax-forced, rhythm-stream EOS 2, no ngram guard.
        let c = a.decode_contract();
        assert_eq!(c.eos_token_id, 2, "rhythm [EOS]");
        assert_eq!(c.temperature, 0.0);
        assert_eq!(c.no_repeat_ngram_size, 0);
        assert_eq!(
            a.license_notice(),
            "Polyphonic-TrOMR (NetEase) - Apache-2.0"
        );
    }

    #[test]
    fn lookup_and_default_resolve() {
        assert_eq!(default_arch().id(), "unlimited-ocr");
        assert!(default_arch().implemented());
        assert!(
            !default_arch().lm_head_stored_int8(),
            "Unlimited-OCR keeps its untied lm_head high precision by default"
        );
        assert_eq!(
            arch_by_id("unlimited-ocr").map(ModelArch::id),
            Some("unlimited-ocr")
        );
        // GOT-OCR2 resolves and is now implemented (full pipeline + KV-cache decode ship).
        let got = arch_by_id("got-ocr2").expect("got-ocr2 is a registered arch");
        assert_eq!(got.id(), "got-ocr2");
        assert!(got.implemented());
        assert_eq!(got.decoder(), Decoder::Qwen2Dense);
        assert_eq!(got.tokenizer(), TokenizerKind::Qwen2Bpe);
        // An unknown id resolves to nothing.
        assert!(arch_by_id("does-not-exist").is_none());
    }

    /// The GOT-OCR2 descriptor matches the census (docs/zoo/got-ocr2-spec.md).
    #[test]
    fn got_ocr2_descriptor_matches_the_census() {
        let got = arch_by_id("got-ocr2").expect("got-ocr2 registered");
        assert!(got.implemented());
        assert_eq!(got.vision_encoder(), VisionEncoder::SamVit);
        assert_eq!(got.decoder(), Decoder::Qwen2Dense);
        assert_eq!(got.tokenizer(), TokenizerKind::Qwen2Bpe);
        // Real config from the census: greedy, eos=<|endoftext|>(151643), ngram 20.
        let c = got.decode_contract();
        assert_eq!(c.temperature, 0.0);
        assert_eq!(c.eos_token_id, 151_643);
        assert_eq!(c.no_repeat_ngram_size, 20);
        // Same doctrine-#2 quant policy (vision high-precision; decoder GEMMs int8).
        assert!(got.quant_policy().vision_high_precision);
        assert!(got.quant_policy().decoder_gemms_int8);
    }

    /// The SmolVLM2 descriptor matches the census (docs/zoo/smolvlm2-spec.md,
    /// bd-3jo6.3.1) — the C2 convert path keys its whole quant classification
    /// off these facts, so they are pinned here.
    #[test]
    fn smolvlm2_descriptor_matches_the_census() {
        let a = arch_by_id("smolvlm2").expect("smolvlm2 registered");
        assert!(a.implemented(), "sub-epic C shipped C1-C9 (describe/VQA)");
        assert_eq!(a.vision_encoder(), VisionEncoder::Siglip);
        assert_eq!(a.decoder(), Decoder::LlamaDense);
        assert_eq!(a.tokenizer(), TokenizerKind::SmolLm2Bpe);
        assert_eq!(a.tasks(), &[Task::Describe, Task::Vqa]);
        assert!(a.license_notice().contains("Apache-2.0"));
        assert!(a.license_notice().contains("HuggingFaceTB"));
        // Real config (census §8): greedy, eos=<end_of_utterance>(49279), and NO
        // upstream repetition guard (ngram 0 / window 0).
        let c = a.decode_contract();
        assert_eq!(c.temperature, 0.0);
        assert_eq!(c.eos_token_id, 49_279);
        assert_eq!(c.no_repeat_ngram_size, 0);
        assert_eq!(c.ngram_window, 0);
        // UNTIED lm_head (byte-verified, spec §12): stored AND high-precision,
        // matching Unlimited-OCR's conservative default; GOT ties and omits it.
        assert!(!a.tie_word_embeddings());
        assert!(!a.lm_head_stored_int8());
        // Idefics3-nested tensor namespaces (spec §12) — the arch-aware convert
        // classification hangs off these prefixes.
        assert_eq!(a.vision_tower_prefix(), "model.vision_model");
        assert_eq!(a.decoder_layers_prefix(), "model.text_model.layers.");
        assert_eq!(
            a.embed_tokens_name(),
            "model.text_model.embed_tokens.weight"
        );
        // Same doctrine-#2 quant policy shape as every arch.
        assert_eq!(a.quant_policy(), QuantPolicy::DOCTRINE);
    }

    /// The historical Qwen-shaped architectures share the original namespace
    /// layout and explicitly retain their certified int8 stored-head policy.
    /// Unlimited-OCR shares the namespace but its conservative high-precision
    /// head is pinned separately in `lookup_and_default_resolve`.
    #[test]
    fn arch_namespace_defaults_are_the_historical_layout() {
        for id in ["got-ocr2", "trocr", "pix2tex"] {
            let a = arch_by_id(id).expect("registered");
            assert_eq!(a.decoder_layers_prefix(), "model.layers.", "{id}");
            assert!(a.lm_head_stored_int8(), "{id}");
            assert_eq!(a.embed_tokens_name(), "model.embed_tokens.weight", "{id}");
        }
    }

    /// The OneChart descriptor matches the census (docs/zoo/onechart-spec.md,
    /// bd-3jo6.4.1) — the D2 convert classification hangs off these facts.
    #[test]
    fn onechart_descriptor_matches_the_census() {
        let a = arch_by_id("onechart").expect("onechart registered");
        assert!(a.implemented(), "sub-epic D shipped D1-D9 (chart-data)");
        assert_eq!(a.vision_encoder(), VisionEncoder::SamVit);
        assert_eq!(a.decoder(), Decoder::OptDense);
        assert_eq!(a.tokenizer(), TokenizerKind::Gpt2Bpe);
        assert_eq!(a.tasks(), &[Task::Chart]);
        assert!(a.license_notice().contains("Apache-2.0"));
        // §10: greedy, eos 2 (`</s>`), no upstream repetition guard.
        let c = a.decode_contract();
        assert_eq!(c.temperature, 0.0);
        assert_eq!(c.eos_token_id, 2);
        assert_eq!(c.no_repeat_ngram_size, 0);
        // §4: TIED head (both source tensors byte-identical, SHA-proven) —
        // the `.focrq` keeps ONE high-precision copy.
        assert!(a.tie_word_embeddings());
        assert!(!a.lm_head_stored_int8());
        // §13: OPT namespaces — the vision tower is `model.vision_tower.`
        // (NOT GOT's `vision_tower_high`), the LM nests under `model.decoder.`.
        assert_eq!(a.vision_tower_prefix(), "model.vision_tower");
        assert_eq!(a.decoder_layers_prefix(), "model.decoder.layers.");
        assert_eq!(a.embed_tokens_name(), "model.decoder.embed_tokens.weight");
    }

    /// The descriptor must match the LIVE engine constants, so it can never drift.
    #[test]
    fn unlimited_ocr_descriptor_matches_the_live_engine() {
        let a = UnlimitedOcr;
        // License notice is the single source of truth in lib.rs.
        assert_eq!(a.license_notice(), FOCR_MODEL_LICENSE_NOTICE);
        // The default artifact basename matches the resolved DEFAULT_MODEL_PATH.
        let want_basename = std::path::Path::new(DEFAULT_MODEL_PATH)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap();
        assert_eq!(a.default_artifact_basename(), want_basename);
        // Graph shape + tasks.
        assert_eq!(a.vision_encoder(), VisionEncoder::SamClip);
        assert_eq!(a.decoder(), Decoder::DeepSeekV2MoeRswa);
        assert_eq!(a.tokenizer(), TokenizerKind::DeepSeekBpe);
        assert_eq!(a.tasks(), &[Task::Ocr]);
        // Quant policy is the fixed doctrine policy.
        assert_eq!(a.quant_policy(), QuantPolicy::DOCTRINE);
        assert!(a.quant_policy().vision_high_precision);
        assert!(a.quant_policy().decoder_gemms_int8);
    }

    /// The descriptor's decode contract must equal the live frozen
    /// `DecodeParams::default()` — the contract the AR loop actually runs.
    #[test]
    fn unlimited_ocr_decode_contract_matches_sampler_default() {
        let c = UnlimitedOcr.decode_contract();
        let d = sampler::DecodeParams::default();
        assert_eq!(c.temperature, d.temperature);
        assert_eq!(c.eos_token_id, d.eos_token_id);
        assert_eq!(c.no_repeat_ngram_size, d.no_repeat_ngram_size);
        assert_eq!(c.ngram_window, d.ngram_window);
        // And the documented frozen values.
        assert_eq!(c.temperature, 0.0);
        assert_eq!(c.eos_token_id, 1);
        assert_eq!(c.no_repeat_ngram_size, 35);
        assert_eq!(c.ngram_window, 128);
    }
}
