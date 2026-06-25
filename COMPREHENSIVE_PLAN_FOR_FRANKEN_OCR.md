# COMPREHENSIVE PLAN FOR franken_ocr

**Master engineering plan — v2 (critique-applied + optimization-expanded)**
**Status:** architecture proposal / pre-Phase-0 (dramatically expanded; ready for `/beads-br` + `/beads-workflow`)
**Audience:** implementing agents (CPU-kernel, model-forward, CLI, conformance) and the lead architect
**Target model:** `baidu/Unlimited-OCR` (released ~2026-06-22; post-cutoff — all model facts below are from live-fetch research, re-verify before kernel work)

> **How to read this document.** Every model-specific claim is tagged **[VERIFIED]** (confirmed from a primary source — HF `config.json`, the index, the modeling source, or the LICENSE), **[REPORTED]** (from the paper or model card but not yet line-verified in code), or **[OPEN]** (an explicit open research question that MUST be resolved by reading the actual `modeling_*.py` before the relevant kernel is built). Do not promote an **[OPEN]** to a hard design assumption. The "Open Research Questions" register in §13 is the single index of everything still uncertain; a phase gate cannot pass while it depends on an unresolved **[OPEN]**.

---

## Table of contents

1. [Mission & non-negotiable goals](#1-mission--non-negotiable-goals)
2. [Target model dossier — Baidu Unlimited-OCR](#2-target-model-dossier--baidu-unlimited-ocr)
3. [Why pure-Rust + frankentorch + asupersync (the generality-tax wedge)](#3-why-pure-rust--frankentorch--asupersync)
4. [System architecture — crate layout & module breakdown](#4-system-architecture)
5. [Weight transformation pipeline](#5-weight-transformation-pipeline)
6. [Model-specific CPU kernel strategy](#6-model-specific-cpu-kernel-strategy)
7. [The `focr` CLI design](#7-the-focr-cli-design)
8. [Verification & conformance methodology](#8-verification--conformance-methodology)
9. [Performance methodology](#9-performance-methodology)
10. [Phased roadmap](#10-phased-roadmap)
11. [Risks & mitigations](#11-risks--mitigations)
12. [Success metrics](#12-success-metrics)
13. [Open research questions register](#13-open-research-questions-register)
14. [Skills, methodology & the path to beads](#14-skills-methodology--the-path-to-beads)

> **Optimization depth (§6.6–§6.13, §9.7).** The kernel catalog (per-arch SIMD dispatch, MoE token-grouping, FlashAttention-style R-SWA softmax, many-core/NUMA scaling, fusion, vectorized transcendentals, memory/allocator, PGO/BOLT) and the alien-artifact math families (rate-distortion bit-allocation, tail-risk CER bounds, conformal early-exit, submodular selection, USL pool-sizing) are where "make this model FLY on Apple Silicon and x86" actually lives. The verification machinery (§8.5–§8.6) is the three-pillar gauntlet that keeps every speed claim honest.

---

## 1. Mission & non-negotiable goals

**Mission.** `franken_ocr` is a pure-Rust (Rust 2024, nightly, `#![forbid(unsafe_code)]` everywhere except tightly-scoped SIMD-intrinsics modules), memory-safe, CPU-hyper-optimized **library + single-binary CLI (`focr`)** that runs the **Baidu Unlimited-OCR** vision-language document-parsing model **with no general ML framework**. We achieve this by transforming the model's bf16 weights into a custom quantized on-disk form (int8 first, int4 in refinement rounds) and writing **model-specific kernels** whose only job is to run *this one model* as fast as possible on:

- **Apple Silicon / ARM64** — NEON, FEAT_DotProd (SDOT), FEAT_MATMUL_INT8 (SMMLA / i8mm)
- **Intel / AMD x86-64** — AVX2, AVX-VNNI, AVX-512-VNNI (and AMX tiles where present)

CUDA is an explicit **later stretch goal** (Phase 6). **CPU is the priority** because most target machines (agent hosts, laptops, CI runners, edge boxes) lack a usable GPU.

The engine is built on **frankentorch** (custom tensors / CPU kernels — consumed at the *kernel* level, not the autograd level) and **asupersync** (structured-concurrency runtime — orchestration / cancellation / IO only). It is **agent-ergonomic** (robot / JSON / NDJSON mode, stable versioned schema, explicit exit codes) and **embeddable** as a plain Rust library with a blocking, sync public API.

### 1.1 Non-negotiable goals (the bar a release must clear)

| # | Goal | Operational definition | Verification owner |
|---|------|------------------------|--------------------|
| **G1** | **OCR accuracy matches the reference stack** | On a frozen golden image corpus, decoded text exact-match where the reference is deterministic; aggregate **CER / TEDS / Formula-CDM within a documented quantization tolerance** of the bf16 reference (target: int8 within noise, int4 within a small, *measured*, ledgered budget). | §8 conformance |
| **G2** | **CPU speed beats the proven CPU baseline — measured honestly, per stage** | **Decode-per-token** wall-clock on the same CPU **faster** than the Phase -1 CPU baseline that actually runs and is proven comparable: CPU-patched PyTorch/HF if it reproduces the CUDA oracle's tokens, otherwise llama.cpp GGUF / ONNX Runtime / MLAS labeled as such. **Vision-prefill** measured and reported honestly — *parity-or-slower is acceptable in the f32 v1* (frankentorch f32 matmul vs MKL/oneDNN; see §9.3, §11). **End-to-end-faster is a post-int4 / optional-int8-vision STRETCH, not a v1 hard gate.** All ratios are honest best-of-N with thread/allocator/precision fairness controls (§9.3). | §9 performance |
| **G3** | **Pure-Rust single static binary, cross-platform** | One `focr` binary per target (linux x86-64/arm64, darwin x86-64/arm64, windows-msvc x86-64), no Python, no FFI, no network at inference time, no GPU required. | §7, §10 Phase 5 |
| **G4** | **Memory-safe** | `#![forbid(unsafe_code)]` at every crate root; `unsafe` confined to named, audited SIMD modules behind an `#[allow(unsafe_code)]` island, each with a bit-identical scalar fallback. | §4, §6 |
| **G5** | **Agent-ergonomic** | `robot` subcommand emitting versioned NDJSON events with a self-describing `robot schema`; stable exit codes; `--json` everywhere; deterministic output under fixed sampling. | §7 |
| **G6** | **Embeddable** | Library API (`OcrEngine::recognize(...)`) is **synchronous and blocking**; async runtime is an owned implementation detail; no global state leaks to the host. | §3, §7 |
| **G7** | **Honest** | Every accepted numeric divergence from the reference has a `DISCREPANCIES.md` ledger entry (reference behavior, our impl, **measured** impact, kill-switch env var); every rejected optimization a `NEGATIVE_EVIDENCE.md` entry. No silent numerics changes. | §8, §9 |

> **G1 over G2, always.** Correctness outranks kernel speed (frankentorch's Non-Regression Rule). A faster kernel that drifts the OCR output is reverted. We ship speed *on top of* parity, never instead of it.

### 1.2 Explicit non-goals (v1)

- **Not** a general OCR framework, not a detect+recognize pipeline, not a multi-model zoo. One model, end-to-end.
- **Not** leaderboard SOTA. Unlimited-OCR sits ~93.9% on OmniDocBench v1.6, **behind** PaddleOCR-VL 1.6 (96.33) and MinerU 2.5-Pro (95.69) **[REPORTED]**. Our pitch is *fidelity to this model* + *bounded generated-token KV for long-document parsing on CPU* + *speed*, not beating the benchmark.
- **Not** training or fine-tuning. Inference only. We never need autograd, optimizers, or backward kernels.
- **No GPU in v1.** CUDA is Phase 6 and clearly stretch.

---

## 2. Target model dossier — Baidu Unlimited-OCR

> This section separates **[VERIFIED]** facts (config / index / license / modeling source) from **[REPORTED]** (paper / card / secondary) and **[OPEN]** (must be code-confirmed before kernel work). The mandate: *ground every model-specific claim in research and flag the uncertain.*

### 2.1 One-paragraph orientation

Unlimited-OCR is an **end-to-end vision-language model (VLM)**, *not* a detect+recognize pipeline. It is a direct derivative of **DeepSeek-OCR**: it reuses DeepSeek-OCR's **DeepEncoder** vision tower (SAM-ViT-B → 16× conv token-compressor → CLIP-L cascade) **verbatim** (frozen during Baidu's training), feeds a single linear projector, and decodes with a **DeepSeek-V2-style Mixture-of-Experts LLM**. Its one architectural novelty is replacing **every** decoder attention layer with **R-SWA (Reference Sliding Window Attention)**: each generated token attends to **all reference tokens** (visual + prompt prefix, kept as a frozen never-evicted global KV) **plus only the previous n = 128 generated tokens** via a ring-buffer KV cache. This makes decode-side KV cache **constant-bounded** (`L·(m + min(n,T)) ≤ L·(m+n)`) instead of growing with output length — that, not arbitrary input resolution, is what "Unlimited" means: constant-KV long-output decoding of dozens of pages in one 32K pass.

### 2.2 Identity, format, size, license — **[VERIFIED]**

| Field | Value | Source |
|-------|-------|--------|
| HF repo | `baidu/Unlimited-OCR` | HF |
| Architecture class | `UnlimitedOCRForCausalLM` (subclass of `DeepseekV2ForCausalLM`) | `config.json` |
| `model_type` | `unlimited-ocr` | `config.json` |
| Weight dtype | **bfloat16** (top-level + `language_config`) | `config.json` |
| Checkpoint | **single shard** `model-00001-of-000001.safetensors`, **6.67 GB** (`total_size` = 6,672,212,480 bytes), **2710 tensors** | `model.safetensors.index.json` |
| Params | **3B total**, ~500M activated/token **[REPORTED]** (paper abstract + §3.2 say 500M; Figure 2 labels "3B-MOE-A570M" → treat as **~500–570M**, **[OPEN]** exact figure) | paper |
| Tokenizer | `LlamaTokenizerFast`, **byte-level BPE**, `tokenizer.json` (9.98 MB), vocab **129280** (the DeepSeek vocab) | `tokenizer_config.json`, `tokenizer.json` |
| License | **MIT**, "Copyright (c) 2026 Baidu" — **identical** on HF `LICENSE` and GitHub `LICENSE` | `LICENSE` |
| Transformers (export tag vs runtime) | config records `4.46.3`; README pins runtime `transformers==4.57.1`, `torch==2.10.0` | `config.json` / README |

**License conclusion — [VERIFIED]: we MAY legally redistribute a converted/quantized (int8/int4) derivative of the weights AND ship our kernels, provided we preserve the MIT copyright + permission notice attributing Baidu.** No copyleft, no field-of-use, no non-commercial restriction. This de-risks the entire "custom quantized form" plan. (The README body itself states no license, but the standalone MIT `LICENSE` file in both repos is unambiguous and controls.)

### 2.3 Architecture components

#### 2.3.1 Vision tower = DeepEncoder (`model.sam_model.*`, `model.vision_model.*`) — reused from DeepSeek-OCR, frozen

- **SAM-ViT-B stage — [VERIFIED]** (`config.vision_config.sam_vit_b`): width 768, 12 layers, 12 heads, `global_attn_indexes [2,5,8,11]`, `downsample_channels [512,1024]`.
  - PatchEmbed `Conv2d` k=16 s=16: 1024×1024 → 64×64 = **4096 tokens**, width 768 **[REPORTED, from deepencoder.py excerpt]**.
  - **Window attention** (window=14 — **[OPEN, OQ-15]**: inherited from the SAM/DeepSeek-OCR lineage, NOT config-verified; only `global_attn_indexes [2,5,8,11]` is config-confirmed) for most blocks; **global attention only at blocks [2,5,8,11]**. (Window attention processes the *dense uncompressed* tokens; global attention is reserved for the compressed set — keeps activations low for high-res.) A wrong window size or windowed-block pos-embed scheme silently corrupts vision features → read `deepencoder.py` before building `vision_sam.rs`.
  - Learnable abs `pos_embed (1,64,64,768)`, **bicubic-interpolated** for non-1024 / tiled inputs.
  - **Neck**: `Conv2d(768→256,k1)+LayerNorm2d` → `Conv2d(256→256,k3,p1)+LN2d` → `Conv2d(→512,s2)` → `Conv2d(→1024,s2)`, output 16×16×1024.
- **16× token compression at the SAM→CLIP bridge — [VERIFIED config + REPORTED detail]**: the two stride-2 neck convs take 64×64 → 16×16 = **256 tokens**. "DeepEncoder can compress a 1024×1024 PDF-image to just 256 tokens."
- **CLIP-L stage — [VERIFIED]** (`config.vision_config.'clip-l-14-224'`): patch_size 14, 24 layers, width 1024, 16 heads, has `class_embedding` + abs pos embeddings. Blocks: LayerNorm → SDPA attention → residual → LayerNorm → FFN with **quick_gelu**.
- **Feature fusion → 2048-dim → projector.** SAM features + CLIP features are concatenated to **2048** dims and a **single linear projector** maps **2048 → 1280** (`projector_config`: `model_type=mlp_projector`, `projector_type=linear`, `input_dim=2048`, `n_embed=1280`) **[VERIFIED config]**. **[OPEN]** The *exact* concat order/path that forms the 2048-dim input (SAM-1024 ⊕ CLIP-1024 vs interleaved; deepencoder.py mentions `low_high_hybrid_split_mlp_gelu` / `hybrid_split_feature_mlp_gelu` variants) MUST be read line-by-line from `deepencoder.py` before implementing the projector.
- **Structural tokens — [VERIFIED]**: two learnable `nn.Parameter`s `model.image_newline` and `model.view_seperator` (sic) are inserted into the vision-token stream.
- **Fusion into the text stream — [VERIFIED]**: vision embeds are injected by `inputs_embeds.masked_scatter_(images_seq_mask, vision_embeds)` — true end-to-end fusion, **no cross-attention resampler / Q-former**.

#### 2.3.2 LLM decoder = DeepSeek-V2 MoE (`model.layers.*`, `model.embed_tokens`, `model.norm`, `lm_head`) — **[VERIFIED config]**

| Field | Value |
|-------|-------|
| `hidden_size` | 1280 |
| `num_hidden_layers` | 12 |
| `num_attention_heads` | 10 |
| `num_key_value_heads` | 10 (**no GQA**) |
| `intermediate_size` (dense MLP, layer 0) | 6848 |
| `moe_intermediate_size` (per expert) | 896 |
| `n_routed_experts` | 64 |
| `n_shared_experts` | 2 |
| `num_experts_per_tok` | 6 (top-6) |
| `topk_method` | `greedy` |
| `norm_topk_prob` | true |
| `n_group` / `topk_group` | 1 / 1 (no group routing) |
| `first_k_dense_replace` | 1 → **layer 0 is a DENSE `DeepseekV2MLP`; layers 1–11 are MoE** |
| `v_head_dim` | 128 (10·128 = 1280, internally consistent) |
| `use_mla` | **false** — **MLA / latent attention is DISABLED**; plain MHA |
| `kv_lora_rank` / `q_lora_rank` | null / null |
| `qk_nope_head_dim` / `qk_rope_head_dim` | 0 / 0 (**[OPEN]** Q/K head dim not directly given by config when `use_mla=false`; read from `modeling_deepseekv2.py` — almost certainly 128 from `q_proj` shape) |
| `max_position_embeddings` | 32768 |
| `vocab_size` | 129280 |
| Norm / activation | **RMSNorm** (LLM), **SiLU** (MLP/experts) |
| RoPE | `DeepseekV2RotaryEmbedding`; YARN/NTK classes present but likely inactive at 32K-native. **[OPEN]** `rope_theta` not surfaced (DeepseekV2 default 10000 assumed) — read directly before kernel work |
| `lm_head` tied | false |
| `bos`/`eos`/`pad` | 0 / 1 / 2 |

**MoE structure — [VERIFIED by tensor names]**: per MoE layer, router `mlp.gate.weight` (1280→64) → softmax → top-6 greedy → `norm_topk_prob`; each of 64 `mlp.experts.{0..63}.{gate,up,down}_proj` is a 1280↔896 SiLU-gated MLP; plus 2 always-on `mlp.shared_experts.{gate,up,down}_proj`. Layer 0 is a dense `mlp.{gate,up,down}_proj` with intermediate 6848.

#### 2.3.3 R-SWA — the one novel mechanism — **[REPORTED paper + VERIFIED config flag]**

- Replaces **all** decoder attention (every one of the 12 layers). `config.sliding_window = 128`, `config.sliding_window_size = 128`.
- `o_t = Σ_{j∈N(t)} α_tj v_j` where `N(t) = {all reference (visual+prompt prefix) tokens} ∪ {last n generated tokens}`, n=128.
- Reference tokens are a **frozen global KV** — never evicted, **excluded from state transitions** (preserving visual-token fidelity, "avoiding progressive blurring" vs vanilla SWA). Output tokens use a sliding window of n=128.
- **KV cache formula**: `C_RSWA(T) = L·(m + min(n,T)) ≤ L·(m+n)` — **constant**, vs MHA's `L·(m+T)`. L=12, n=128, m = reference length.
- **Implementation — [REPORTED, from modeling_deepseekv2.py]**: `class SlidingWindowLlamaAttention` uses a **ring-buffer KV cache** of width W=128: records `prefill_len` on first decode, warms up until cache reaches `prefill_len+W`, then in-place ring overwrites at `slot = prefill_len + ring_pos`, `ring_pos = (ring_pos+1) % W`. **This is the single most important kernel implication.**

**[OPEN] R-SWA boundary questions to confirm from `SlidingWindowLlamaAttention` source before building the attention mask:**
1. Does the reference set `m` include the running **prompt** tokens, or strictly the **visual** tokens? (Paper says "all reference tokens (visual tokens in OCR)" but Figure 2 caption says "visual+prompt".)
2. Is attention **uniformly** R-SWA across all 12 layers (the `first_k_dense_replace=1` split affects only the *MLP*, not attention — so attention is expected R-SWA everywhere, including layer 0), with **no** layer retaining full attention?
3. Exact warm-up vs ring-overwrite mask semantics during the first 128 decode steps.

### 2.4 Input / output contract

#### Input — **[VERIFIED config + REPORTED preprocessing]**

- RGB document image(s). `image_mean = image_std = [0.5,0.5,0.5]` → pixels scaled to [0,1] then `(x-0.5)/0.5` → **[-1,1]**.
- `patch_size=16`, `downsample_ratio=4`, `num_queries = ceil((image_size//patch_size)/downsample_ratio)`.
- Pad to square with **gray (127,127,127)** = `tuple(int(m*255))`, aspect preserved (`ImageOps.pad`).
- **Two retained resolution modes** (DeepEncoder natively had 5; Unlimited-OCR keeps 2):
  - **Base**: 1024×1024, `crop_mode=false` — multi-page/PDF. **256 visual tokens/page**.
  - **Gundam**: `base_size=1024`, `image_size=640`, `crop_mode=true` — dynamic-resolution tiling for a single page; global padded 1024 view (`global_view_pos='head'`) + 640×640 local crops via `dynamic_preprocess()` / `find_closest_aspect_ratio()` over `i*j ∈ [min_num, max_num]` (**[OPEN]** literal default bounds, reported 2..32). Each 640 crop ≈ **100 tokens**. `tile_tag='2D'`, `candidate_resolutions=[[1024,1024]]`.
- `add_special_token=false`, `image_token='<image>'`, `ignore_id=-100`, `processor_class='UnlimitedOCRHFProcessor'`.
- **Token budget is deterministic but it is NOT just 256 — count every token CLASS (OQ-18).** A 1024 view yields **256 image-feature tokens (16×16) PLUS structural tokens**: per the modeling source the placeholder structure is `(16 + 1)·16 + 1 = 273` (16 image tokens + 1 `image_newline` per row, × 16 rows, + 1 `view_seperator`). The decoder context = `image-feature + image-newline + view-separator + prompt + generated` tokens, and the **R-SWA reference block `m` and all KV/buffer sizing must account for every class**, not the 256 compression figure. A **machine-readable token/shape/buffer census** (§8.6, generated from the pinned config) is the source of truth and CI fails if the source/config changes. **[OPEN, OQ-18]** exact Gundam multi-tile token totals per aspect ratio.

#### Output — **[REPORTED + VERIFIED tokenizer]**

- Structured **Markdown**: HTML tables, **LaTeX** formulas, normalized **0–999 (×/999)** bounding-box coords, reading order, `<page>` separators for multi-page.
- Layout tags `<|ref|>…<|/ref|><|det|>…<|/det|>` parsed via regex; bbox coords `/999 → ×W/H`.
- Decode: greedy / low-temp argmax, strip trailing EOS `<｜end▁of▁sentence｜>`, and **`no_repeat_ngram_size=35` with a sliding `ngram_window` that DIFFERS by mode — 128 (single-image) vs 1024 (multi-image)** — these are **first-class generation semantics, not just CLI flags** (`SlidingWindowNoRepeatNgramProcessor` bans repeated 35-grams within the window). The sampler must replicate the exact window per mode (confirm from source in the truth pack).
- `infer(...)` signature **[VERIFIED]**: `model.infer(tokenizer, prompt='', image_file='', output_path='', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False, max_length=32768, tps_interval=0, no_repeat_ngram_size=0, ngram_window=0, temperature=0.0)`. Multi-page `infer_multi(...)` concatenates pages at `<image>` with a dummy crop tensor to force the no-crop branch.
- Prompts are free text, e.g. `'<image>document parsing.'`, `'<image>Multi page parsing.'`. **[OPEN]** Full prompt-mode taxonomy (free OCR vs layout vs grounding vs markdown) and whether bboxes are emitted in *all* modes is not documented — read the processor/`conversation.py`.

### 2.5 Capabilities & benchmarks — **[REPORTED]** (secondary/paper, not all line-verified)

- **Strong**: printed text (Text-Edit 0.038 on OmniDocBench v1.5), tables (TEDS 90.93 / TEDS-S 94.07), math/LaTeX (Formula-CDM 92.61), layout & reading order (Reading-Order-Edit 0.045). CJK + "40+ languages" **[OPEN, unverified count]**.
- **Weak**: handwriting ("limited but improving"), charts (not a headline strength).
- **OmniDocBench**: v1.5 overall **93.23** (vs DeepSeek-OCR 87.01, +6.22); v1.6 **93.92**. Peers (Jun-2026): PaddleOCR-VL 1.6 96.33, MinerU 2.5-Pro 95.69, PaddleOCR-VL 1.5 94.50, **Unlimited-OCR 93.92**, Qianfan-OCR 93.12, DeepSeek-OCR-2 91.09. → **We are not SOTA; we are a strong, long-horizon, constant-memory model.**
- **Long-horizon quality**: edit distance 2pg 0.0362 / 10pg 0.0526 / 40+pg 0.1069 (graceful).
- **Throughput**: ~5580 TPS on OmniDocBench (+12.7% vs DeepSeek-OCR), advantage growing with output length (+35% at 6144 out-tokens) because constant KV cache avoids long-context decode slowdown.

### 2.6 Prior-art quantizations — **[REPORTED, decisive for our recipe]**

**Two community quantizations already exist, and BOTH keep the vision encoder unquantized.**

1. **GGUF** (`sahilchachra/Unlimited-OCR-GGUF`): full K-quant ladder (BF16…IQ2_M, 1.15–5.47 GB), recommended **Q4_K_M (~1.82 GiB)**. **Vision/mmproj kept F16 (774 MiB)** — author states explicitly that quantizing the vision encoder hurts OCR. Requires llama.cpp **PR #17400 (not upstream)**. No published perplexity/CER.
2. **NVFP4** (`sahilchachra/Unlimited-OCR-NVFP4`): quantizes **only ~2196 linear modules of the MoE decoder** to FP4 (group size 16, NVFP4A16, **data-free PTQ**), keeping **vision tower, projector, embeddings, output layer, MoE router gate, and ALL norms in BF16**. ~2.93 GB. **Reports OCR output identical to BF16** — on the author's tested docs only; no published CER.
   - **⚠️ Module-count reconciliation — [OPEN, OQ-14].** "2196" does **not** reconcile with our census of quantizable decoder linears: experts `11·64·3 = 2112` + attention `12·4 = 48` + shared `11·2·3 = 66` + dense-layer-0 `3` (+ `lm_head` `1`) ≈ **2229–2230**, ~33 more than 2196. So NVFP4 is **not** quantizing the set we assume — it likely **skips attention `q/k/v/o` and/or `lm_head` and/or the shared experts**. We must dump NVFP4's scale keys to enumerate its exact set (OQ-14) before leaning on it.

→ **The core recipe is converged and validated for the EXPERT/MLP FFNs**: weight-only quant of the **decoder expert + dense FFN GEMMs**; **full precision** (or near) for vision encoder, projector, embeddings, **router gate**, and **all norms**. We adopt that core recipe (§5, §6). **But two of our choices go BEYOND the validated set and are explicitly OUR risk to retire by measurement (OQ-14): int8 on the attention `q/k/v/o` projections, and int8 on `lm_head`** — both gated behind a measured-CER kill-switch, neither assumed lossless "because NVFP4 did it." We may also use these artifacts as **golden oracles** during bring-up (dump their per-layer activations) without any framework runtime dependency.

### 2.7 Required neural-op set (the complete kernel surface)

Derived from the architecture; each tagged with frankentorch availability (see §4.3 for the full map).

1. **Conv2d** — SAM patch-embed (k16s16), SAM neck (k1, k3p1, two stride-2), CLIP patch-embed (k14s14). Small fixed set → hardcode shapes. *frankentorch: `conv2d_forward_f32` / `conv2d_im2col_f32` exist.*
2. **SAM window + global self-attention** (window=14; global at 2/5/8/11). *Needs windowed-mask variant on top of `sdpa_forward_f32`.*
3. **Bicubic pos-embed interpolation** (Gundam tiling). *frankentorch: only f64 autograd `interpolate` in ft-api — **GAP**, build f32.*
4. **CLIP-L self-attention + quick_gelu FFN.** *`sdpa_forward_f32` + new `quick_gelu`.*
5. **Feature concat → 2048 → linear projector → 1280.** *`linear_tensor_f32`.*
6. **`image_newline`/`view_seperator` insertion + masked-scatter fusion.** *new small kernel.*
7. **Token embedding lookup** (129280×1280). *index_select pattern.*
8. **RMSNorm.** *`rms_norm_forward_f32`.*
9. **RoPE** (theta default 10000; YARN present but likely unused). *new kernel.*
10. **R-SWA decode attention** (reference block + 128-ring-buffer). *new — the centerpiece.*
11. **DeepseekV2 MoE**: gate linear → softmax → top-6 greedy → `norm_topk_prob` → grouped SiLU-gated expert MLP + 2 shared experts. *new dispatch + int8 GEMM.*
12. **Dense MLP** for layer 0 (intermediate 6848). *`linear_int8_dynamic_f32` ×3 + SiLU.*
13. **Final RMSNorm + lm_head GEMM** (1280→129280) + autoregressive sampling with `no_repeat_ngram_size=35`. *int8 GEMM + sampler.*

**Activations**: SiLU (LLM), quick_gelu (CLIP), GELU (SAM MLPBlock). **Norms**: RMSNorm (LLM), LayerNorm + LayerNorm2d (vision). Keep vision-family and text-family kernels separate.

---

## 3. Why pure-Rust + frankentorch + asupersync

### 3.1 The generality-tax wedge — *one fixed model, compile-time-known shapes, no framework*

A general ML framework pays a **generality tax** on every op: dynamic dtype dispatch, arbitrary shapes, autograd tape bookkeeping, broadcast machinery, device abstraction, a scheduler that knows nothing about *this* graph. `franken_ocr` runs **exactly one model** whose every dimension is **known at compile time** — hidden 1280, 10 heads, head_dim 128, 64 experts, top-6, moe_intermediate 896, window 128, vocab 129280. That lets us:

- **Specialize kernels to fixed shapes** — `const`-generic tile sizes, no remainder handling for dims that are nice multiples (1280, 896, 128, 6848 all tile cleanly for SMMLA 8×8 and VNNI), no runtime shape branching in the hot loop.
- **Pre-pack weights offline** into arch-specific interleaved int8/int4 tiles — the converter knows the exact layout each kernel wants; the runtime never reshuffles.
- **Pre-allocate everything** — R-SWA bounds the generated-token KV to 128, so we allocate one fixed ring buffer per layer plus a reference block sized by the Phase -1 context/token census. We can allocate for the worst-case context once and never realloc, but the logical reference length `m` still grows with page/input length up to the 32K cap.
- **Skip autograd entirely** — inference only. No tape, no `requires_grad`, no backward. We reach **past** ft-api's stateful `FrankenTorchSession`/NodeId graph straight to `ft-kernel-cpu`'s ~465 free-standing pure kernel functions over `&[f32]` slices.
- **Fuse aggressively** — dequant+GEMM+bias+activation, RMSNorm+RoPE, router+gather — because no framework boundary forbids it.

This is the same wedge that made `franken_whisper` (the closest analog) and the `frankensearch` int8 reranker work: a hand-written model forward in plain Rust over a `Mat`/slice currency, with frankentorch supplying only the parallel SIMD GEMM and the dtype primitives.

### 3.2 Why frankentorch specifically (consumed at the kernel level)

frankentorch already ships **the single highest-value asset we need**: a complete, production-proven int8 dynamic-quant linear with multi-arch SIMD GEMM.

- `quantize_per_output_channel_i8(w, out, in_) -> (Vec<i8>, Vec<f32>)` — symmetric per-output-channel, zero-point 0, `scale = max|w_row|/127`.
- `linear_int8_dynamic_f32(x, m, k, w_i8, w_scales, n, bias) -> Vec<f32>` — per-row dynamic activation quant, **i32-accumulate** GEMM, dequant, +bias. Mirrors ONNX `DynamicQuantizeLinear` + `MatMulInteger`.
- Runtime-dispatched SIMD **int8 dot** (as of frankentorch `a84674c8`, 2026-06-24): **aarch64 SDOT** (`vdotq_s32`, gated on `dotprod`), **x86-64 AVX-512-VNNI** (`_mm512_dpbusd_epi32`, with the documented +128 unsigned-correction), **scalar fallback** — all **bit-identical** i32 accumulation (integer add is associative, so SIMD == scalar exactly; tolerance only needed for int8-vs-f32 drift, not for the SIMD path).
- This exact op is already in production in the frankensearch reranker — **a direct precedent for `franken_ocr`**.
- **⚠️ The honest performance state, the MEASURED lessons, and franken_ocr's actual edge.** The sibling reranker campaign (eidetic/frankensearch on frankentorch, M4) is the live, measured guide, and it has already disproved one tempting theory. **The bar is ONNX-int8 / MLAS: 7.6 / 14.5 / 41.4 ms** per doc at seq 128/256/512. The ratchet so far: scalar **23/53/146** → SDOT (bit-identical, 16 MAC/instr) **14.8/34/64** → **register-blocked** SDOT (4×4 tile, amortized loads) **12.4/28/56** — now **~1.4–1.9× behind ONNX**, closing but not yet dominating. The hard-won lessons that bind *this* project:
  1. **The gap is "kernels below peak," NOT framework overhead.** A hand-written *fused, tape-free* forward that replaced the SIMD/parallel kernels with **naive scalar-f32 attention/softmax/LayerNorm regressed the whole forward 3–10×** (38/194/580 ms). Deleting per-op tape/allocation overhead is necessary but **worthless unless every fused op stays at peak** (SIMD + parallel + int8/int4). Never trade a good kernel for a naive one.
  2. **Un-blocked SMMLA is a TRAP.** A tiled SMMLA with 2× the MAC density but no register blocking was *slower* than SDOT (19/41/77) because it is **load-bound** (≈2 loads : 1 SMMLA). The win requires **register/cache blocking (compute:load ≥ 2:1) + offline weight pre-packing** — that *is* the MLAS recipe.
  3. **int8 attention is a named lever ONNX uses and the reranker doesn't yet.** ONNX runs `Q·Kᵀ` and `scores·V` in int8; an f32 attention is ~20–33% of the matmuls left un-accelerated.
  4. **AMX-f32 does NOT beat ONNX-int8** (bandwidth-bound); AMX must be int8 and helps **compute-bound prefill, not memory-bound decode**; Apple's AMX is not directly programmable (Accelerate/BNNS is FFI, opt-in — the directly-usable Mac path is NEON SMMLA/SDOT).
  - **So franken_ocr's edge is the COMBINATION the reranker is assembling lever-by-lever, but which `franken_ocr` gets by construction:** **(a)** a *fused, tape-free, zero-per-op-allocation* forward for ONE model (it never builds a `FrankenTorchSession`/NodeId tape — §3.1), **(b)** with *every op at peak* (register-blocked SMMLA/VNNI linears + int8 attention where accuracy allows + vectorized norms/softmax — never naive), plus **(c)** the **int4 bandwidth win on the expert bulk (§6.3)** that the reranker doesn't have. We reuse the existing SDOT/VNNI dot as the bit-exact int8 baseline/oracle and **build** the register-blocked tiled GEMM as our contribution. **Honest bar:** strictly beating a years-tuned MLAS at *every* point is genuinely hard; the realistic and still-very-valuable target is **at-or-near ONNX on CPU, portable to the targets where `ort` cannot build, with bounded generated-token KV for long-document decode** (§1.1 G2, per-stage honest).

Plus the f32 building blocks: `conv2d_forward_f32`/`conv2d_im2col_f32`, `sdpa_forward_f32` (+ masked + GQA), `rms_norm_forward_f32`, `layer_norm_forward_f32`, `softmax_dim_*`, `silu`/`gelu`, `argmax_dim_*`, and `matmul_tensor_contiguous_f32`. And `ft-serialize::load_safetensors_from_bytes` (F32/F16/**BF16**→ in-process) for the converter.

**What frankentorch does NOT give us (gaps to build):** GPU (none — CPU only, by design — perfect for our priority); image decode/resize/normalize (none); f32 bicubic `interpolate` (only f64 autograd in ft-api); int8 as a safetensors load dtype (quantize in-process); the tiled SMMLA/AMX int8/int4 GEMM and int4 entirely (§3.2); the windowed-attention mask, RoPE, R-SWA, MoE dispatch, and quick_gelu (all model-specific, we build them). See §4.3 for the exact exists-vs-build table.

**Reusable sibling primitives for the gaps (research-confirmed — copy patterns, not whole crates).** `franken_numpy/fnp-ufunc`'s register-tiled, B-panel-packed, L2-cache-blocked GEMM (`matmul_accumulate_serial`, with its `MATMUL_MR`/`MATMUL_NR`/`nc≈256KiB` constants) is the **pattern to port to f32/int8** for our tiled kernels; `fnp-ndarray`'s `NdLayout` + `sliding_window_view` is a clean, stable-Rust, `forbid(unsafe)` basis for the **im2col** layout calculus; `frankenjax/fj-lax`'s activations (`gelu`/`silu`/`softmax`/`log_softmax`, pure `fn(&[f64])`) retype cleanly to f32; `fnp-io`'s NPY/NPZ is a convenient **golden-fixture interchange** for the reference oracle. None of the siblings ship image I/O or a strided NCHW conv — those we build fresh (the `image` crate + im2col → our GEMM). `franken_networkx` is string-keyed and irrelevant (a pixel-CCL would be a fresh integer disjoint-set, not needed for this VLM anyway).

### 3.3 Why asupersync (orchestration / cancellation / IO only)

asupersync is the structured-concurrency runtime (Tokio replacement; Cx-first; cancellation as an explicit request→drain→finalize protocol). The proven integration pattern (from `franken_whisper`) is **layered and the one to copy**:

- **`fn main()` stays SYNCHRONOUS** — no `#[asupersync::main]`. Parse clap, install a Ctrl+C `ShutdownController`, call a sync `run(cli)`, map errors → exit codes. The runtime lives **below** main, inside the engine, never spanning the process.
- **The engine OWNS one `Runtime`** (`RuntimeBuilder::new().worker_threads(2).blocking_threads(1,4).thread_name_prefix("focr").build()`). Public methods are **sync**: internally `runtime.handle().spawn(async {...})` then `runtime.block_on(handle)`. Callers see a blocking API (satisfies G6).
- **CPU-bound stages run via `spawn_blocking`** wrapped in `asupersync::time::timeout(wall_now(), budget, Box::pin(spawn_blocking(op)))` for per-page/per-stage budgets (env-overridable).
- **Intra-op math parallelism is the frankentorch kernel's OWN rayon pool**, NOT asupersync tasks. asupersync is orchestration + cancellation + IO; the heavy GEMMs saturate cores via the kernel's pool. **Pin the rayon global pool to physical cores, and constrain blocking concurrency to EXACTLY ONE live forward** (the sequential page loop, §6.5) so only one N-core fan-out ever runs — small `blocking_threads` (e.g. `(1,2)`) is a guard, not the mechanism. **"Streaming per-page" = streaming the *output* of a sequentially-processed page, never concurrent forwards** (which would oversubscribe rayon N×).
- **Streaming results** (per-page text as it completes) to the robot/NDJSON consumer use **`std::sync::mpsc::sync_channel` + `std::thread`** (bounded, proven, simple), main looping on `recv_timeout(~40ms)` until the worker finishes.
- **Cancellation**: a `Copy` `CancellationToken` (deadline + global `ShutdownController`), `checkpoint()` threaded into long inner loops (per-page, per-decode-step) so Ctrl+C / per-page timeout aborts at the next boundary. (Cancellation is cooperative — `spawn_blocking` keeps running on drop, so the token must go *into* the closure.)
- **Runtime capacity certificate (budgeted + explainable, per `asupersync-mega-skill`)**: streaming uses **bounded** channels (backpressure, never unbounded growth); errors/cancellation map to an `Outcome`-style result internally (not panics across the FFI-free boundary); and the runtime ships a **capacity certificate** as a gauntlet artifact (§8.5) — p95/p99 queueing evidence + a proof of **no nested-runtime / no rayon-pool oversubscription** (the USL pool caps of §6.9 / AF-5 hold under the many_pages soak).

> **Hard rule (from frankentorch + asupersync research): NEVER nest rayon under a held lock, and NEVER nest a second asupersync runtime inside a task.** The engine owns exactly one runtime; the model forward fans out across all cores internally; the outer page/document loop is **sequential** and calls the blocking facade. A `many_pages_without_deadlock` CI watchdog test (pages ≫ pool) hangs on regression. This is the durable fix the frankensearch deadlock saga converged on after 5 commits.

---

## 4. System architecture

### 4.1 Repository shape — single crate, two binaries (the `franken_whisper` template, verbatim)

`franken_ocr` is **one crate** (NOT a workspace), depending on sibling crates by path. This mirrors `franken_whisper` exactly and is the right shape for a single-model port.

```
franken_ocr/                          (crate: franken_ocr)
├── Cargo.toml                        # two [[bin]]: focr + franken_ocr, both -> src/main.rs
│                                     # [lints.rust] unsafe_code = "deny"; #![forbid(unsafe_code)] in lib.rs/main.rs
│                                     # path deps: ../frankentorch/crates/ft-kernel-cpu, ft-core, ft-serialize
│                                     #            asupersync = {version=">=0.3.5,<0.4", default-features=false}  (local /dp/asupersync = 0.3.5)
│                                     #            fsqlite (frankensqlite) — NEVER rusqlite
│                                     #            image / fast_image_resize (preproc), clap, serde_json
├── rust-toolchain.toml               # channel = "nightly" (REQUIRED for stdarch i8mm/dotprod)  [already present]
├── src/
│   ├── main.rs                       # sync fn main(): clap parse, ShutdownController, dispatch, robot mpsc loop
│   ├── lib.rs                        # thin re-exports: OcrEngine, OcrRequest/OcrResult, Error/Result, pipeline types
│   ├── cli.rs                        # clap-derive Cli/Command/Args, to_request() validation, robot summary
│   ├── orchestrator.rs               # OcrEngine (owns asupersync Runtime), Pipeline{Stage,Config,Builder},
│   │                                 #   PipelineCx + CancellationToken + FinalizerRegistry, run_stage_with_budget
│   ├── robot.rs                      # versioned NDJSON events, robot schema/health/backends, ROBOT_SCHEMA_VERSION
│   ├── conformance.rs                # tolerance structs, parity/invariant validators, rollout stages
│   ├── storage.rs                    # RunStore on fsqlite, _meta versioned schema, forward migrations
│   ├── sync.rs                       # JSONL export/import (locked, atomic, one-way audit)
│   ├── error.rs                      # FocrError/FocrResult, exit-code mapping
│   ├── preprocess/                   # IMAGE INGEST FRONT END (a known gap — built fresh)
│   │   ├── mod.rs                    # decode (image crate) -> RGB
│   │   ├── resize.rs                 # bilinear/bicubic resize keeping aspect (fast_image_resize or hand-rolled)
│   │   ├── normalize.rs             # ToTensor [0,1] -> (x-0.5)/0.5 -> [-1,1], CHW
│   │   ├── pad.rs                    # ImageOps.pad-equivalent, gray (127,127,127)
│   │   └── tile.rs                   # Base vs Gundam; dynamic_preprocess / find_closest_aspect_ratio
│   ├── tokenizer/                    # pure-Rust HF-tokenizers-JSON byte-level BPE (no SentencePiece)
│   │   ├── mod.rs                    # encode/decode over tokenizer.json (9.98MB), 129280 vocab
│   │   └── special.rs               # bos/eos/pad/<image>/<|ref|>/<|det|> handling
│   └── native_engine/                # THE MODEL PACKAGE (self-contained, plain Rust over Mat/slices)
│       ├── mod.rs                    # OcrModel: cached Arc + Weak global cache, resolve_model, header sniff
│       ├── weights.rs                # custom quantized format reader (.focrq) + safetensors fallback + manifest census
│       ├── tensor.rs                 # Mat { rows, cols, data: Vec<f32> } currency + quantized weight structs
│       ├── nn.rs                     # frankentorch facade: matmul/int8-linear/rmsnorm/softmax/silu/gelu/quick_gelu
│       ├── vision_sam.rs             # SAM-ViT-B: patch-embed conv, window/global attn, neck convs
│       ├── vision_clip.rs           # CLIP-L: patch-embed, 24-layer SDPA + quick_gelu FFN
│       ├── vision_bridge.rs          # 16x compression, feature concat->2048, linear projector->1280
│       ├── connector.rs              # image_newline/view_seperator insertion, masked-scatter fusion
│       ├── decoder.rs                # 12-layer loop: RMSNorm, RoPE, R-SWA attn, MoE/dense MLP, final norm
│       ├── rswa.rs                   # R-SWA ring-buffer KV cache + reference-block attention (centerpiece)
│       ├── moe.rs                    # router top-6 greedy, expert gather/dispatch, shared experts
│       ├── decode.rs                 # frozen DecodeParams/DecodeOutput contract + AR loop + no_repeat_ngram(35)
│       └── postprocess.rs            # EOS strip, <|ref|>/<|det|> regex, bbox /999 rescale, markdown assembly
├── tests/                            # conformance_harness.rs, native_engine_e2e.rs (model-gated), robot_contract
├── tests/fixtures/                   # golden corpus + frozen reference outputs (per-layer + end-to-end)
├── benches/                          # criterion: vision_encode, decode_token, end_to_end, gauntlet
├── scripts/                          # convert_weights, fetch_test_models, gen_reference_fixtures (Python oracle)
├── docs/                             # DISCREPANCIES.md, NEGATIVE_EVIDENCE.md, PERF_LEDGER.md, conformance-contract.md
└── .github/workflows/dist.yml        # 5-target cross-platform release matrix
```

**Why this shape:** `franken_whisper` proved that the single-crate + two-`[[bin]]` (full name + short alias, both → `src/main.rs`) layout with a self-contained `native_engine/` is the right scaffold. We take the **scaffold** (orchestrator / CLI / robot / conformance / storage / native_engine structure) and **not** whisper's domain code (audio/mel/dtw/streaming/backends).

### 4.2 Pipeline stages (composable, budgeted, cancellable)

A `PipelineStage` enum drives the forward, each with a per-stage env-overridable budget (`FOCR_STAGE_BUDGET_<STAGE>_MS`) and a `checkpoint()` boundary:

```
Decode/Load image → Preprocess (resize/pad/normalize/tile) → Tokenize prompt
  → Vision encode (SAM → bridge/compress → CLIP → concat → project)
  → Connector (insert structural tokens, masked-scatter into embed stream)
  → Prefill (build reference KV: visual + prompt) → Decode loop (R-SWA + MoE, AR)
  → Postprocess (EOS strip, ngram, tag parse, bbox rescale, markdown) → Emit
```

Vision encode is a **fixed per-page cost** (run once, ~256 image-feature tokens before connector structural tokens, dominated by SAM/CLIP attention+conv GEMMs over ~4096 patches). Decode is a **per-output-token cost** (memory-bound expert GEMV + 129K-vocab logits). These two cost centers are optimized **separately** (§6, §9).

### 4.3 Op → frankentorch map (exists vs must-build)

| Op | Where used | frankentorch status | Plan |
|----|-----------|---------------------|------|
| int8 dynamic-quant linear (SMMLA/SDOT/VNNI) | all decoder GEMMs, lm_head, projector(opt) | **EXISTS** `linear_int8_dynamic_f32` + `quantize_per_output_channel_i8` | reuse as-is; this is the crown asset |
| f32 linear | projector, vision (v1) | **EXISTS** `linear_tensor_f32` | reuse |
| Conv2d (im2col + GEMM) | SAM/CLIP patch-embed, SAM neck | **EXISTS** `conv2d_forward_f32` / `conv2d_im2col_f32` | reuse; hardcode the ~5 fixed shapes; patch-embed is non-overlapping → cheap exact im2col → 1 big GEMM |
| SDPA attention | SAM global, CLIP, (basis for R-SWA prefill) | **EXISTS** `sdpa_forward_f32` (+masked, +gqa) | reuse for CLIP/SAM-global |
| Windowed self-attention (window=14 **[OPEN, OQ-15]**) | SAM non-global blocks | **BUILD** | windowed mask/partition over sdpa; **confirm window size + pos-embed scheme from `deepencoder.py` first** |
| quick_gelu | CLIP FFN | **BUILD** (small) | `x * sigmoid(1.702x)` |
| GELU / SiLU | SAM MLP / LLM MLP | **EXISTS** gelu/silu tensor ops | reuse |
| RMSNorm | LLM | **EXISTS** `rms_norm_forward_f32` | reuse |
| LayerNorm / LayerNorm2d | vision | **EXISTS** `layer_norm_forward_f32` (2D variant **BUILD** thin wrapper) | reuse + wrap |
| Bicubic pos-embed interpolate (f32) | Gundam tiling | **GAP** (only f64 autograd in ft-api) | **BUILD** f32 bicubic |
| RoPE (theta 10000) | decoder attn | **BUILD** | new kernel (confirm theta first) |
| R-SWA decode attention (ring + ref block) | every decoder layer | **BUILD** (centerpiece) | new kernel + fixed ring buffer |
| MoE router top-6 greedy + norm_topk_prob | decoder layers 1–11 | **BUILD** | softmax + top-k + gather |
| Grouped expert SiLU-gated MLP | MoE | **BUILD** (uses int8 linear) | gather active experts → batched int8 GEMM |
| Token embedding lookup (f32-preserving) | embed | **BUILD** thin (index_select pattern) | preserve f32, don't materialize f64 |
| masked-scatter vision fusion | connector | **BUILD** (small) | new kernel |
| Image decode/resize/normalize/pad/tile | preprocess | **GAP** | **BUILD** (image / fast_image_resize) |
| BPE tokenizer (tokenizer.json) | tokenize/detok | **GAP** | **BUILD** pure-Rust HF-JSON BPE |
| Sampler + no_repeat_ngram(35) | decode | **BUILD** | argmax + n-gram blocklist |
| safetensors BF16 load | converter | **EXISTS** `load_safetensors_from_bytes` | reuse in converter |

---

## 5. Weight transformation pipeline

The mandate: **HF download → reference-parity load → custom on-disk quantized format (int8 first, int4 later)**, per-channel/per-group, with a deterministic round-trip story.

### 5.1 Stages

```
[1] DOWNLOAD (out-of-band, scripts/fetch_model.sh)
    baidu/Unlimited-OCR/model-00001-of-000001.safetensors (6.67 GB bf16, single shard) + tokenizer.json + config.json
    -> never fetched at inference time (G3: no network at runtime)

[2] REFERENCE-PARITY LOAD (scripts/convert_weights, a focr subcommand: `focr convert`)
    ft-serialize::load_safetensors_from_bytes(&blob) -> BTreeMap<String, DenseTensor>  (decodes BF16->f32)
    Validate against a WeightsManifest census: expected (name, shape) for all 2710 tensors.
       MISSING / SHAPE-MISMATCH / EXTRA -> LOUD named diff, refuse to proceed.
    Tensor remap: HF dotted state_dict paths -> our internal layout (see 5.3).

[3] QUANTIZE (the recipe from §2.6, validated by NVFP4/GGUF prior art)
    Decoder linears (q/k/v/o_proj, layer-0 dense gate/up/down, 64 experts x {gate,up,down}, 2 shared x {gate,up,down}, lm_head):
       -> int8 (Phase 2) per-output-channel symmetric (quantize_per_output_channel_i8 scheme), zero-point 0
       -> int4 (Phase 4) per-group symmetric, group size 16-32 (NVFP4 used 16), Q4_K_M-equivalent ~4.5-4.9 bpw,
          keep attention .v_proj and expert .down_proj one tier higher (llama.cpp *_M discipline)
    KEEP HIGH PRECISION (BF16 verbatim / F32) — DO NOT QUANTIZE BY DEFAULT:
       entire vision tower (sam_model.* + vision_model.*), projector, embed_tokens, MoE router gate, ALL norms,
       and lm_head unless the explicit measured `FOCR_INT8_LMHEAD` stage is enabled
       (`lm_head` is high-value but high-risk; quantize only behind its measured-accuracy kill-switch)

[4] PRE-PACK (arch-specific, offline — the "transform into a custom quantized form" mandate)
    Emit interleaved int8/int4 tile blobs:  one packing for SMMLA 2x2 (aarch64), one for VNNI/AMX (x86)
    so kernels load contiguous tiles with ZERO runtime shuffle (mirrors llama.cpp q4_0_4x8/q4_0_8x8 "aarch64" types).
    Pack expert weights so the per-token active experts are cache-contiguous.

[5] WRITE custom format (.focrq) — see 5.2
    Self-describing header + per-tensor records + the MIT attribution string (license compliance).

[6] ROUND-TRIP / DETERMINISM GATE (see 5.4)
```

### 5.2 Custom on-disk format `.focrq`

A safetensors-like, length-prefixed, self-describing container (clone the `franken_whisper` ggml/safetensors parser pattern: read whole file into one `Vec<u8>` blob, validate magic, read header, index tensors by byte range):

```
magic: b"FOCRQ\0"                              # loud rejection on mismatch
format_version: u32                            # bumped on any layout change
arch_target: enum { Generic, Aarch64Smmla, X86Vnni, X86Amx }   # which pre-packing
source_sha256: [u8;32]                         # sha256 of the source safetensors (provenance)
license_notice: utf8                           # "Copyright (c) 2026 Baidu — MIT License ..." (MUST be present)
model_config: json                             # frozen copy of the relevant config.json fields
tensor_directory: [
  { name, dtype: {F32,F16,BF16,QInt8PerChan,QInt4PerGroup}, shape, packing, byte_offset, byte_len,
    scales_offset, scales_len, group_size?, tier? }
]
payload: <raw bytes>
```

- **Quantized weights carry their scales inline.** int8: per-output-channel `Vec<f32>` scales (`scale = max|w_row|/127`, zero-point 0). int4: per-group scales (group 16–32) + tier metadata.
- **High-precision weights stored BF16 verbatim** (vision/projector/embeddings/norms/router, and `lm_head` when unquantized), dequantized BF16→f32 at load. **BF16, NOT F16:** the checkpoint is bf16 (1-8-7); narrowing to f16 (1-5-10) is *lossy* (different range + mantissa, clips at ±65504) and **not** a byte-identical round-trip — so f16 high-precision storage is only ever a **measured/accepted `DISCREPANCIES.md` divergence**, never the silent default. bf16 and f16 are both 2 bytes, so there is no disk reason to prefer the lossy one.
- The reader (`native_engine/weights.rs`) is a dependency-free byte-range index into one mmap/blob, exactly like whisper's loader, and performs the **manifest census** on load (catch wrong/stale weights at load, not as garbage output).

### 5.3 Tensor remapping (HF dotted paths → internal layout)

| HF path | Internal | Notes |
|---------|----------|-------|
| `model.sam_model.patch_embed.proj.{weight,bias}` | `vision.sam.patch_embed` | conv k16s16 |
| `model.sam_model.pos_embed` | `vision.sam.pos_embed` | (1,64,64,768), bicubic on tiling |
| `model.sam_model.blocks.{0..11}.*` | `vision.sam.block[i].*` | global at i∈{2,5,8,11} |
| `model.sam_model.neck.*` / net_2 / net_3 | `vision.sam.neck.*` | 1×1, 3×3p1, two stride-2 |
| `model.vision_model.embeddings.{patch_embedding,class_embedding,position}` | `vision.clip.embed.*` | patch14 |
| `model.vision_model.transformer.layers.{0..23}.*` | `vision.clip.layer[i].*` | SDPA + quick_gelu |
| `model.projector.layers.{weight,bias}` | `vision.projector` | single linear 2048→1280 |
| `model.image_newline`, `model.view_seperator` | `connector.image_newline`, `connector.view_sep` | nn.Parameter |
| `model.embed_tokens.weight` | `decoder.embed` | 129280×1280, **keep bf16** (verbatim) |
| `model.layers.{0..11}.self_attn.{q,k,v,o}_proj.weight` | `decoder.layer[i].attn.{q,k,v,o}` | int8, no fused MLA |
| `model.layers.0.mlp.{gate,up,down}_proj.weight` | `decoder.layer[0].dense.{gate,up,down}` | dense, intermediate 6848, int8 |
| `model.layers.{1..11}.mlp.gate.weight` | `decoder.layer[i].router` | **keep high precision** |
| `model.layers.{1..11}.mlp.experts.{0..63}.{gate,up,down}_proj.weight` | `decoder.layer[i].expert[e].{...}` | int8/int4, cache-contiguous pack |
| `model.layers.{1..11}.mlp.shared_experts.{gate,up,down}_proj.weight` | `decoder.layer[i].shared.{...}` | int8 |
| `model.layers.{0..11}.{input,post_attention}_layernorm.weight` | `decoder.layer[i].{norm1,norm2}` | RMSNorm, **keep f32** |
| `model.norm.weight` | `decoder.final_norm` | RMSNorm |
| `lm_head.weight` | `decoder.lm_head` | int8 (gated) |

### 5.4 Determinism & round-trip story

- **Bit-exact round-trip for high-precision tensors**: **BF16**/F32 stored verbatim → loaded → byte-identical. A `convert→load→re-serialize` test asserts byte equality. (Any bf16→f16 narrowing is a *lossy* transform that must be a ledgered `DISC-NNN`, never a silent storage choice.)
- **Deterministic quant**: `quantize_per_output_channel_i8` is a pure function of the input bytes; same source → same `.focrq` (assert `source_sha256` + a content hash of the output). No RNG, no calibration data in v1 (data-free PTG, validated by NVFP4 being data-free and OCR-identical).
- **Dequant determinism + i32-overflow safety (RECOMPUTED for THIS model — do NOT inherit frankensearch's `k≤1536` bound).** int8 GEMM is bit-identical across SIMD paths (integer add is exact/associative). The i32 accumulator must not overflow `i32::MAX = 2,147,483,647`. Worst-case `|acc|` per GEMM, signed×signed (`≤ K·127²`) and the **U8S8 / VNNI** path (unsigned activation `[0,255]` × signed weight, `≤ K·255·127`):
  - attention `q/k/v/o_proj`, expert/dense `gate`/`up`, `lm_head`: **K = 1280** → S8S8 ≤ 20.6M, U8S8 ≤ 41.4M.
  - expert `down_proj`: **K = 896** → S8S8 ≤ 14.5M, U8S8 ≤ 29.0M.
  - **dense layer-0 `down_proj`: K = 6848** (the worst case) → S8S8 ≤ **110.4M**, U8S8 ≤ **221.7M**.

  All paths fit i32 with ≥ 9× headroom — **but this is a proof obligation, not an assumption.** A unit test multiplies worst-case saturated operands at **K = 6848** on every kernel/arch and asserts the i32 result equals an i64 reference (and that no path silently saturates). If a future fused/grouped/accumulating path approaches `i32::MAX`, switch *that* path to i64/blocked accumulation. The SIMD path needs **no** tolerance vs scalar. Only the int8-vs-f32 *forward* drifts — and its `max_diff` must be **measured for this model**, not inherited (frankensearch's `0.055` is a BERT-reranker figure on a different shape/depth). The parity gate tolerances continuous logits within that *measured* budget while requiring exact argmax/token where the reference is deterministic (§8.2).
- **Provenance**: `.focrq` embeds `source_sha256` + `format_version` + frozen config; the loader refuses a `.focrq` whose `format_version` exceeds the binary's, and warns on `arch_target` mismatch (falls back to the generic packing).
- **One canonical artifact, many packings**: `focr convert --arch {aarch64-smmla,x86-vnni,x86-amx,generic}` emits per-arch blobs from the same source; CI verifies all packings dequantize to the same logical weights.

---

## 6. Model-specific CPU kernel strategy

### 6.1 The hot ops (profile-anchored, two cost centers)

| Op | Phase (prefill/decode) | Bound | Priority |
|----|------------------------|-------|----------|
| MoE expert GEMMs (6 routed + 2 shared, 1280↔896 ×3) | decode (per token) | memory | **HIGHEST** (bulk of params, per-token) |
| lm_head GEMV (1280→129280) | decode (per token) | memory | **HIGH** (huge, per token) |
| R-SWA decode attention (ref block + 128 ring) | decode (per token) | mixed | **HIGH** (centerpiece, every layer) |
| Vision attention (SAM window/global + CLIP) over ~4096/256 tokens | prefill (per page) | compute | **HIGH** (per page, SMMLA wins 2–2.5×) |
| Vision/decoder MLP GEMMs | prefill | compute | **MED** |
| Patch-embed conv (im2col→GEMM) | prefill | compute | **MED** |
| RMSNorm / LayerNorm / RoPE / softmax / SiLU / quick_gelu | both | mem/compute | **LOW** (autovectorize, don't hand-SIMD) |

### 6.2 The frankensearch lesson — the load-bearing constraint on HOW we optimize

> **Hand-wide-SIMD over scalar inner loops was 5× SLOWER than LLVM autovectorization.** The win came from **(a) full-core-parallel forward + (b) native int8 GEMM intrinsics (SMMLA/SDOT/VNNI)**, with LLVM autovectorizing the surrounding glue. This is memorialized in the shipped kernel comment and is non-negotiable doctrine.

Therefore:
1. **Do NOT hand-roll wide SIMD over the elementwise/norm/softmax/dequant glue.** Write those as tight scalar loops with `f64` accumulation where precision matters and let LLVM autovectorize. (frankentorch's norm/softmax/gelu already do this with `std::thread::scope` row-band parallelism gated by element-count thresholds.)
2. **DO write tight int8 GEMM/GEMV micro-kernels using NATIVE matmul intrinsics.** Reuse frankentorch's **existing bit-exact int8 dot** (`dot_i8_sdot` = `vdotq_s32`; `dot_i8_vnni512` = `_mm512_dpbusd_epi32`) as the **decode-GEMV baseline + correctness oracle**, and **BUILD the tiled register-blocked GEMM that frankentorch has only *named*** — it does not exist yet (verified at frankentorch `a84674c8`), and building it is our wedge:
   - **ARM — BUILD (the wedge)**: a tiled **SMMLA / i8mm** micro-kernel (`vmmlaq_s32`, 2×2 int8 outer-product, 32 MACs/instr) for **prefill GEMM**, amortizing loads + the int32 reduction across a register tile (this directly attacks the ~1.5–2.4×-vs-MLAS gap of §3.2). Reuse `dot_i8_sdot` (`vdotq_s32`) for **decode GEMV**. Gate on `is_aarch64_feature_detected!("i8mm")` / `("dotprod")`.
   - **x86**: `dot_i8_vnni512` (`_mm512_dpbusd_epi32`, AVX-512-VNNI, with the documented +128 sign-flip correction) as the workhorse; AVX-VNNI (`_mm256_dpbusd_epi32`) for non-512 parts; AMX tiles (Sapphire Rapids+) for prefill where present; **AVX2 fallback** (`_mm256_maddubs_epi16`+`_mm256_madd_epi16`). Dispatch tier: AMX > AVX512-VNNI > AVX-VNNI > AVX2 > scalar.
   - **Scalar fallback** that is bit-identical AND **cross-compiles to every target** (so the binary builds for, e.g., a target without the intrinsics; the right path is chosen at runtime).
3. **Parallelize the FORWARD across all physical cores** (per-layer / per-expert / per-row-block), via the kernel's own rayon pool — NEVER under a held lock, NEVER nested under an outer rayon `par_iter` (the deadlock rule).

### 6.3 Per-arch int8/int4 GEMM plan

**Mixed int4/int8 is the headline CPU-performance strategy — pushed as far as a measured CER bound allows (this is the whole point of the project).** Decode is **memory-bandwidth-bound**, so **int4 group-quantized weights (g = 16–32) halve the bytes streamed per token vs int8 → up to ~2× decode throughput**; applied to the dominant **expert FFN weights** (the bulk of the 3B params *and* the decode hot path) this is the single biggest decode lever. There is **no CPU int4 matmul instruction**, so int4 weights are **unpacked to int8 in-register and fed to the same SMMLA / VNNI / SDOT MAC** — the win is *bandwidth + footprint*, not extra compute (prefill stays compute-bound int8 on SMMLA/AMX, where int4 mainly helps the weight-stream). Accuracy-sensitive tensors stay **int8** (attention `v_proj`, expert `down_proj` kept one tier higher; `lm_head` int8 behind its kill-switch), and the **per-tensor int4-vs-int8-vs-bf16 split is CHOSEN by the rate-distortion / water-filling allocator (AF-1, §9.7)** — never uniform. Activations are **int8 dynamic per-row** (int4 activations are too lossy for exact-token OCR). KV-cache quant is low priority (R-SWA keeps it tiny, §6.4) but int8/int4 KV is available if Gundam/multi-page pressures memory. **Net target: int4 on the expert bulk + int8 on the sensitive minority + int8 activations — the maximal quantization the tail-risk CER gate (AF-2) permits.**

- **Activations**: per-row dynamic int8 quant at forward (frankentorch scheme). For VNNI/AMX, produce **U8S8** (uint8 activations with zero-point + int8 symmetric weights) since `vpdpbusd` wants unsigned activations × signed weights; for SMMLA/SDOT, signed×signed. The converter and kernel agree on the convention per arch.
- **Weights**: int8 per-output-channel symmetric (Phase 2); int4 per-group (16–32) symmetric (Phase 4), with `.v_proj` / expert `.down_proj` kept one tier higher.
- **i32 accumulate, scales applied post-accumulation**: `dequant = i32_acc · scale_w[chan/group] · scale_a[row]`, then +bias, then activation; requantize for the next layer where chained.
- **Target the int4 milestone at Q4_K_M-equivalent (~4.5–4.9 bpw), NOT sub-4-bit** — the accuracy cliff is *below* 4-bit, and OCR of dense numbers/tables/code is exact-token-sensitive (perplexity under-predicts this). Ship int8 (Q8_0-class, ~lossless) first as the correctness oracle.

### 6.4 Op-by-op kernel design

**Patch-embed conv** — non-overlapping (k=stride), so im2col is cheap and exact → a single big GEMM `[num_patches × (C·P·P)] @ [(C·P·P) × embed_dim]` → ideal SMMLA/VNNI target. **But keep the vision tower BF16/F32 in v1** (the validated recipe; ViT post-LayerNorm/post-GELU activations have severe outliers that wreck low-bit activation quant). Revisit int8 vision (per-channel weights + per-token dynamic act + outlier-aware / register-prefix) only after decoder parity is proven, as an isolated experiment.

**Vision attention** — SAM windowed (window=14, partition into 14×14 windows, local SDPA), SAM global (full SDPA at blocks 2/5/8/11), CLIP full SDPA (24 layers). All over a small token set (≤4096 dense / 256 compressed) → compute-bound → SMMLA prefill GEMM where quantized, else f32 sdpa.

**R-SWA decode attention (the centerpiece, `rswa.rs`)** — where the constant-KV gift pays off, **with one honest caveat**:
- **Constant MEMORY, NOT constant per-token COMPUTE.** R-SWA bounds the *generated-token* KV to a 128-ring, so decode-side **memory** is flat across output length. But the **reference block `m` grows ~256 tokens per page** (more in Gundam) and is attended **every** decode step, so per-token attention is `O(m + 128)` and its **compute scales linearly with page count** up to the 32K cap. Do not market decode as "constant-time per token across document sizes"; for many-page docs `Q · reference-K` over `m`≈thousands can rival the MoE cost (so R-SWA is re-ranked *per regime* in §9.1).
- **Pre-allocate, per layer, one fixed buffer sized for the WORST-CASE `m`** (≈ `32768 − 128`): reference KV block (length `m`, visual+prompt, **read-only for the whole generation**, never grows, never evicted) + a **128-wide ring buffer** for generated-token KV. No realloc; trivially memory-safe.
- Decode step `t`: compute Q for the new token; scores = (Q · reference-K block) ∪ (Q · ring-K), softmax over the union, weighted sum of (reference-V ∪ ring-V). Write the new K,V into the ring at `slot = prefill_len + (t % 128)` (after warm-up).
- Build the matmul tiling around the known max KV length (`m + 128`) — cache-resident for small `m`, streamed for large `m`. **[OPEN]** resolve the R-SWA boundary questions (§2.3.3 / OQ-1..3 and the multi-page span OQ-13) from `SlidingWindowLlamaAttention` *before* finalizing the mask.
- Plain MHA (no MLA decompression): `q/k/v/o_proj` separate dense GEMMs, 10 heads, head_dim 128. Simpler than full DeepSeek-V2.

**MoE dispatch (`moe.rs`)** — per token: router gate (1280→64, **f32, never quantized**) → softmax → top-6 greedy → `norm_topk_prob`. Gather the 6 active routed experts (pre-packed cache-contiguous) + 2 shared experts; each is a 1280→896 SiLU-gated MLP (`gate_proj`, `up_proj`, `down_proj`), int8/int4 GEMV. This + lm_head are the two hottest decode kernels → highest-value int8/int4 targets.

**lm_head + sampler (`decode.rs`)** — int8 GEMV 1280→129280 (gated kill-switch), then **argmax/top-k only** (no full softmax under greedy), with a **no_repeat_ngram_size=35** blocklist over the 129280 vocab (track last-35 n-grams, set repeats to -inf).

**Weight pre-packing / blocking / tiling** — offline arch-specific interleave (§5.4); blocking around fixed shapes (1280, 896, 6848, 128 tile cleanly); L2-cache-blocked B-panels (copy the frankentorch GEMM constants `MATMUL_MR`/`MATMUL_NR`/nc≈256KiB and the register-tile pattern, retyped to int8 accumulation).

**KV-cache quant — deprioritized.** R-SWA keeps the cache tiny (128 positions/layer for the sliding part) and this is plain MHA, so int8/int4 KV buys little. If Gundam/multi-page long-context pressures memory, add **int8 KV first** (safe, ~2×); int4 KV later only if needed.

### 6.5 Concurrency discipline (the deadlock-proof template)

- **Single `OcrModel` behind cache; SEQUENTIAL page/document loop**; each page's forward fans out across all cores internally via the kernel rayon pool. **No** page-level `par_iter` over a held lock.
- `linear_int8_dynamic_f32` and other rayon kernels are called from the sequential outer loop, never from inside an outer `par_iter` holding a `Mutex`.
- CI watchdog: `many_pages_without_deadlock` (pages ≫ pool) — hangs (CI timeout) on regression. This is the durable architectural fix, not a per-kernel patch.

### 6.6 Per-arch SIMD dispatch catalog (exact intrinsics + tile geometry + weight packing)

One `int8_gemm` / `int8_gemv` entrypoint, runtime-dispatched once per process (cached in a `OnceLock<IsaTier>`), bit-identical across every path (i32 accumulation is exact). `focr robot backends` reflects the selected tier. **The packing layout is chosen offline by `focr convert --arch` and recorded in the `.focrq` header** so the kernel loads contiguous register tiles with zero runtime shuffle (the llama.cpp `q4_0_4x8`/`q4_0_8x8` "aarch64 repack" lesson).

| Tier | ISA gate | GEMM (prefill, compute-bound) | GEMV (decode, BW-bound) | Weight pack | Tile |
|------|----------|-------------------------------|-------------------------|-------------|------|
| **A1** | aarch64 `i8mm` | **SMMLA** `vmmlaq_s32` — 2×2 int8 outer-product, 32 MACs/instr, 4× i32 lanes | SDOT | rows interleaved 2×8 (`A_2x8`), K-panel packed | MR×NR = 8×8 over K-blocks of 64–256 |
| **A2** | aarch64 `dotprod` | SDOT `vdotq_s32` (16 MAC/instr) blocked | SDOT | row-major + per-row scale | 4×16 |
| **X1** | x86 `avx512vnni+avx512bw` | **VPDPBUSD** `_mm512_dpbusd_epi32` (U8S8, 64 MAC/instr) | VPDPBUSD | u8 activation (+128 fold) × s8 weight, 4-row interleave | 8×16 (zmm) |
| **X2** | x86 `avxvnni` | `_mm256_dpbusd_epi32` (256-bit) | same | as X1, ymm | 8×8 |
| **X3** | x86 SPR+ `amx-int8` | **`_tile_dpbssd`** 16×16×64 int8 tile matmul → 16×16 i32 | (prefill only; decode uses X1) | AMX tile config (`STTILECFG`), 64-wide K | 16×16 |
| **X4** | x86 `avx2` | `_mm256_maddubs_epi16`→`_mm256_madd_epi16` (saturating, see ⚠) | same | u8×s8 | 8×8 |
| **S** | always (cross-compile floor) | scalar i32 MAC | scalar | row-major | — |

Dispatch order: **X3(AMX) > X1(VNNI-512) > X2(VNNI-256) > X4(AVX2)** on x86; **A1(SMMLA) > A2(SDOT)** on ARM; **S** everywhere as the build-always fallback. ⚠ **AVX2 `vpmaddubsw` saturates at i16** before the i32 widen, so the AVX2 path is *not* automatically bit-identical to the i32-exact paths for adversarial operands — it must carry its own overflow proof (split-accumulate every 2 K-steps, plan §5.4) or be marked a documented `DISCREPANCIES.md` divergence with a measured CER delta. The SMMLA/VNNI/SDOT/scalar paths share one i32 accumulator semantics and ARE bit-identical.

The two prefill matrix engines (SMMLA, AMX) are the *new* code that closes the §3.2 gap to MLAS; the decode GEMV paths reuse frankentorch's existing bit-exact `dot_i8_*`.

> **⚠ Measured lessons that bind this kernel (from the sibling reranker campaign — see §3.2).** (1) **Register/cache blocking + offline pre-packing are MANDATORY, not optional**: an un-blocked SMMLA (2× MAC density) was *slower* than SDOT because it is load-bound (≈2 loads : 1 SMMLA); the micro-kernel must reach **compute:load ≥ 2:1** (reuse each loaded activation/weight tile across many MACs). (2) **AMX must be int8** — AMX-f32 does not beat ONNX-int8 on these memory-bound sizes, and AMX helps **compute-bound prefill, not decode**. (3) **Apple's AMX is NOT directly programmable**: on Mac the directly-usable int8 matrix path is **NEON SMMLA/SDOT**; reaching the Apple AMX coprocessor means **Accelerate / BNNS**, which is C/FFI and therefore an **opt-in feature** (like mimalloc, §6.12), never the no-FFI default. (4) **Never replace a good SIMD/parallel kernel with a naive hand-written one** — naive scalar-f32 fused ops regressed the whole forward 3–10× (§3.2 lesson 1).

### 6.7 MoE expert dispatch — token-grouping turns scattered GEMVs into dense GEMMs (the biggest structural decode lever)

The MoE FFN is the bulk of the 3B params and the dominant decode cost. Naively, each token gathers its 6-of-64 routed experts and does six separate small GEMVs — memory-bound, poor reuse. Two regimes, two strategies:

- **Prefill (many tokens at once — the prompt + all vision tokens):** after the router assigns each of the `T` prefill tokens its top-6 experts, **sort/group tokens by expert id** (a counting-sort over 64 buckets, `O(T)`), then run **one dense int8 GEMM per active expert** over its grouped token-block (`tokens_e × 1280 → tokens_e × 896`, SwiGLU, `→ 1280`), scatter results back. This converts 64 scattered GEMVs into ≤64 dense SMMLA/AMX GEMMs with full register-tile reuse — the single largest prefill MoE win. (This is exactly the "DeepSeek MoE on Xeon AMX/VNNI" grouped-GEMM pattern.)
- **Decode (one token):** can't batch tokens, so minimize bytes streamed: experts are **pre-packed cache-contiguous per `(layer, expert)`** so the 8 active experts (6 routed + 2 shared) load as contiguous int4/int8 blocks; software-prefetch the next expert's weights while computing the current; the router (kept f32) selects which blocks to touch. int4 here halves the streamed bytes — the direct decode-throughput lever.
- **Router:** `1280 → 64` f32 GEMV → softmax → top-6 via partial selection (`select_nth`, not a full sort) → `norm_topk_prob`. Never quantized (NVFP4 keeps it; quantizing the gate drifts expert selection → cliff).
- **Fused SwiGLU:** compute `gate_proj` and `up_proj` in one fused pass (shared activation quant of the input row), then `silu(gate) * up`, then `down_proj` — one dequant of the input, not three.

### 6.8 R-SWA attention — online (FlashAttention-style) softmax over the reference block

The R-SWA score set per decode step is `Q · [reference-K (length m, up to ~32K) ∪ ring-K (128)]`. Materializing the full `1 × m` score row per head per layer is wasteful and, at `m ≈ 32K × 10 heads × 12 layers`, cache-hostile. Use **online softmax** (the FlashAttention recurrence): stream the reference-K/V block in cache-sized tiles, maintaining running `(m_max, l_sum, acc)` per head — never materialize the full score vector, never a second pass. Benefits: bounded working set (one K/V tile + the accumulator), single pass, and it composes with the int8 `Q·Kᵀ` GEMM (the reference block is large enough to be a real GEMM, not a GEMV). The 128-wide ring window is tiny and handled as a tail tile. This is the kernel that makes the "constant-memory, compute-grows-with-pages" R-SWA (§6.4) actually cache-efficient at 40-page scale. **[OPEN OQ-1..3, OQ-13]** still gate the exact mask; the online-softmax structure is independent of those answers.

**int8 attention as a named perf lever (behind an accuracy gate).** ONNX/MLAS runs the `Q·Kᵀ` and `scores·V` bmm in int8; the sibling reranker, with an f32 attention, leaves ~20–33% of its matmuls un-accelerated, and that is a measured part of its remaining gap. For the **vision attention** (prefill, compute-bound, many tokens), int8 `Q·Kᵀ`/`scores·V` on SMMLA is a real lever. For the **R-SWA decode attention** it is more delicate — the scores feed softmax/exp (error-amplifying), so it goes behind a `FOCR_INT8_ATTN` kill-switch with a measured-CER gate (AF-2). v1 keeps attention f32 for parity; int8 attention is a **Phase-3 perf lever, not a v1 default** — but it is on the list precisely because it is where ONNX earns part of its win.

### 6.9 Many-core & NUMA scaling — Apple Silicon P/E + Threadripper / EPYC (honest, measured)

"Make it fly on 64-core Threadrippers" requires an honest scaling model, not blind `par_iter` over 64 threads:

- **Two parallelism axes, never mixed blindly.** *Latency* (one page, faster): parallelize **within** exactly one live forward — prefill GEMMs over output-row blocks, MoE over experts, attention over heads. *Throughput* (many independent documents/pages): parallelize **across** workers only when each worker receives a disjoint thread/NUMA budget and does **not** spawn a full `num_cpus` rayon fan-out. `focr` may expose both (`--page-parallelism` vs single-doc), but the capacity certificate must prove `workers × per_worker_threads ≤ physical-core budget` and no nested global-pool oversubscription.
- **Prefill scales with cores; decode does NOT (it's memory-bandwidth-bound).** A single-token decode GEMV streams ~500M active params/token; beyond the cores needed to saturate DRAM bandwidth, more threads add nothing (and hurt via contention). Model decode with the **Universal Scalability Law** (USL: `C(N) = N / (1 + α(N−1) + βN(N−1))`; the queueing-theory family AF-5, §9.7) — fit α (contention) and β (cross-core coherency) from a 1/2/4/8/…/64-thread sweep, then **cap the decode thread pool at the USL peak**, not at `num_cpus`. Over-threading decode is a measured anti-win to ledger.
- **NUMA (Threadripper/EPYC multi-CCD, dual-socket).** The weight blob is read-only and hot. **First-touch or explicitly replicate the `.focrq` weights per NUMA node**, pin each page-worker (and its rayon sub-pool) to one node, and allocate per-page activation buffers node-local. Cross-node weight fetches are the silent killer on EPYC. Expose `FOCR_NUMA={replicate|interleave|local}`; default `local` with first-touch. Use `hwloc`/affinity (behind a feature; scalar/no-NUMA fallback elsewhere).
- **Apple Silicon P/E asymmetry.** M-series has performance + efficiency cores; rayon's work-stealing handles imbalance, but pin the heavy GEMM pool to **P-cores** (QoS `USER_INTERACTIVE`) and keep orchestration on E-cores. M4 has `i8mm` (SMMLA) + `dotprod` + `bf16` — verified (`sysctl hw.optional.arm.FEAT_I8MM`).
- **No oversubscription:** one global rayon pool sized to physical P-cores; the asupersync blocking pool stays tiny (§6.5); exactly one live forward fans out at a time.

### 6.10 Fusion catalog (no framework boundary forbids it)

Each fusion removes a memory round-trip of the activation; all are bit-exact (same arithmetic, fewer materializations) and gated by the isomorphism proof:

| Fused unit | Replaces | Win |
|-----------|----------|-----|
| dequant → int8-GEMM → bias → activation | 4 passes over the output | 1 pass; activation in registers |
| RMSNorm → RoPE | 2 passes over the hidden | 1 pass; RoPE reads normed Q/K live |
| gate_proj ‖ up_proj → SiLU·mul (SwiGLU) | 3 GEMMs + 2 passes | 1 input-dequant, fused elementwise |
| router → top-6 select → expert-gather | sort + gather | counting-sort + scatter (§6.7) |
| residual add → next RMSNorm | 2 passes | 1 pass |
| lm_head GEMV → argmax (greedy) | full 129280 softmax + argmax | argmax-only, no exp over vocab |

### 6.11 Vectorized transcendentals — the *measured* exception to "autovectorize the glue"

The frankensearch lesson (don't hand-wide-SIMD scalar loops) holds for **i32 MAC** glue. It does **not** automatically hold for **transcendentals** (`exp` in softmax; `sigmoid` in SiLU/quick_gelu), which LLVM will *not* autovectorize (it emits scalar `libm` calls). A vectorized **polynomial `exp`** (range-reduce + degree-(5–7) minimax poly, `std::simd` or the frankenjax `simd_poly_exp` pattern) is a known multi-× win for softmax/activations — but it is a *numerics-affecting lever*: gate it behind a `OnceLock` env switch, default it ON only after a measured A/B + a passing parity gate (online softmax logits within the int8 tolerance), and ledger it. This is the one place hand-SIMD is the right tool, and it must still clear the keep-gate.

### 6.12 Memory, allocator, and layout

- **Weight blob:** `mmap` the `.focrq` read-only (no 6→3 GB copy on load); `madvise(WILLNEED)` the active region; consider transparent/explicit **huge pages** for the multi-GB blob (fewer TLB misses on the GEMM weight streams).
- **Allocator:** the **default build uses the system allocator and links NO C/FFI**, preserving G3's pure-Rust / no-FFI single-binary story. `mimalloc` is an **opt-in, perf-only feature** (`--features mimalloc`) — it is a **C library (FFI)**, so it is *never* part of the default or the "no FFI" claim; it helps the per-page activation churn, and when a head-to-head perf claim uses it, both sides use the same allocator (§9.3 fairness).
- **Buffer reuse (the tape-truncation analog):** pre-allocate every per-forward buffer once (KV ring + reference block sized for worst-case `m`, attention scratch, MoE token-group buffers, the f32 activation rails) and **reuse across pages** — zero per-token / per-page allocation in the hot loop.
- **Alignment & SoA:** 64-byte-aligned activation and packed-weight buffers (clean SMMLA/AVX-512 loads); store packed weights and scales in struct-of-arrays so a tile load is one contiguous stream.
- **BF16/F32 bulk dequant:** the default high-precision (vision/embeddings/router/norm) tensors are BF16 on disk because the checkpoint is BF16. Bulk-convert BF16→f32 with SIMD/table-assisted paths where available; any f16 narrowing is a measured `DISCREPANCIES.md` entry, never the default.

### 6.13 Build-time optimization (free, model-agnostic, do them all)

- **LTO `fat` + `codegen-units=1` + `panic=abort`** on the shipping `release` profile (set in `Cargo.toml`); keep the separate `release-perf` profile (frame pointers, `lto=thin`, `strip=false`) for profiling only — never profile the size/abort build (profiling-software-performance build-flags rule).
- **PGO (profile-guided optimization):** instrument, run the golden OCR corpus (a representative dense page + a 10-page doc) to collect a profile, rebuild with it. The decode loop's branch layout (router top-6, ring-buffer wrap, ngram check) is exactly what PGO straightens. Bakes one `*.profdata` per release.
- **BOLT** (post-link binary optimization) on top of PGO for I-cache/branch layout of the hot kernels — a real "last 5–10%" lever for the tight GEMM/decode loops.
- **`target-feature` not `target-cpu=native`:** we ship one portable binary per arch and select SIMD at *runtime* (so `native` is wrong — it would break the cross-compile floor). Enable the baseline feature set per target and gate the rest with `is_*_feature_detected!`.

---

## 7. The `focr` CLI design

### 7.1 Binaries & entrypoint

Two `[[bin]]` (`focr` short + `franken_ocr` long), both → `src/main.rs`. **`fn main()` is SYNCHRONOUS** (clap parse → install `ShutdownController` (ctrlc) → sync `run(cli)` → exit-code mapping). The asupersync runtime is owned by `OcrEngine` below main (§3.3).

### 7.2 Subcommands

| Subcommand | Purpose |
|-----------|---------|
| `focr ocr <image>` | Primary: parse a document **image** → markdown (human default) or `--json`. **v1 is IMAGE-ONLY** (PDF: §7.7) |
| `focr ocr --robot` (or `focr robot run`) | Streaming **NDJSON** events for agents |
| `focr convert <safetensors> -o <.focrq> [--arch ...] [--quant int8\|int4]` | Offline weight transformation (§5) |
| `focr robot schema` | Self-describing event/contract schema (versioned) |
| `focr robot health` | Diagnostics: model present? arch features? threads? |
| `focr robot backends` | Detected SIMD tiers (SMMLA/SDOT/VNNI/AMX/scalar), core count |
| `focr runs [--id\|--limit\|--format plain\|json\|ndjson]` | Query run history (fsqlite) |
| `focr sync export-jsonl\|import-jsonl` | Audit export/import (locked, atomic, one-way) |
| `focr doctor` | Idempotent self-check/repair (model resolution, format version, perms) |

### 7.3 Robot / JSON / NDJSON contract

- **NDJSON event stream** (one JSON object per line), every line carrying `schema_version` (a `ROBOT_SCHEMA_VERSION` const). Event types: `run_start`, `stage` (with stage name, seq, elapsed, budget), `page` (per-page text/bbox as it completes — streaming), `run_complete`, `run_error`.
- `robot schema` **self-describes** all event types (machine-readable contract); a `robot_contract_tests.rs` validates emitted events against a frozen JSON schema fixture. Phase 0 may expose a schema seed, but it must still be compact single-line JSON for agent parsers.
- **Streaming**: worker `std::thread` runs the engine with a bounded `sync_channel`; main loops `recv_timeout(~40ms)` emitting events until `worker.is_finished()`, then drains and `join()`s.
- **Deterministic** under fixed sampling (`temperature=0` greedy) — same image+args → byte-identical output (a determinism gate asserts this).

### 7.4 Exit codes (stable, documented)

`0` success · `1` generic error · `2` usage/CLI error · `3` model not found / not resolvable · `4` input decode error (bad image/PDF) · `5` budget/timeout exceeded · `6` cancelled (Ctrl+C) · `7` format/version mismatch (`.focrq`). Mapped in `error.rs`; `robot run_error` carries the same code in its payload.

### 7.5 Env overrides & model resolution (no network at runtime)

- `resolve_model(spec)`: existing path OR short name searched across an ordered, env-driven list (`$FOCR_MODEL_DIR`, `~/.cache/franken_ocr/models`, …), failing with an **actionable** error listing every searched dir. `native_model_available()` does a cheap header sniff (magic bytes), no tensor load.
- Env: `FOCR_MODEL_DIR`, `FOCR_THREADS`, `FOCR_STAGE_BUDGET_*_MS`, `FOCR_QUANT` (int8/int4/f32 select if multiple `.focrq` present), `FOCR_FORCE_ARCH` (override SIMD dispatch for testing), `FOCR_NO_REPEAT_NGRAM` (default 35). Each numerics-affecting lever read once via `OnceLock`, documented in its own doc-comment, defaulted by a measured A/B, with an opt-out.

### 7.6 Cross-platform single binary

5-target release matrix (`.github/workflows/dist.yml`): linux x86-64/arm64, darwin x86-64/arm64, windows-msvc x86-64. Static where possible; one self-contained `focr` per target. Runtime SIMD dispatch means **one binary per arch** auto-selects the best int8 path (AMX/VNNI/SMMLA/SDOT) — no per-microarch builds. `install.sh` curl-pipe installer; tar.gz/zip archives + SHA-256 checksums.

### 7.7 PDF input is OUT OF SCOPE for v1 (image-only) — [OPEN decision]

The reference rasterizes PDFs with **pymupdf at 300 DPI**. A pure-Rust raster at **byte-parity** with MuPDF is a large, unscoped sub-project, and *any* pixel mismatch changes OCR output and would blow the **L0 preprocessing parity gate** (§8.2). So **v1 accepts images only**; PDFs are rasterized **out-of-band** (the user runs `pdftoppm`/`pymupdf` and passes pages as images). Re-introducing native PDF is an explicit decision: (a) bundle `pdfium`/MuPDF behind a feature flag (re-adds a C dependency — weigh against G3's pure-Rust single binary), or (b) a pure-Rust renderer **plus a rasterization-parity gate vs pymupdf@300DPI**. Until then, `pdf` does not appear in the CLI surface.

---

## 8. Verification & conformance methodology

Apply `/porting-to-rust`, `/running-the-gauntlet-on-your-rust-port`, `/testing-conformance-harnesses`, `/testing-golden-artifacts`. **Parity gate FIRST, perf second** (frankensearch doctrine).

### 8.1 The Python/HF reference oracle

A pinned environment (`torch==2.10.0`, `transformers==4.57.1`, `Pillow==12.1.1`, `pymupdf` — per README, **NOT** the 4.46.3 in config) runs `model.infer()` to capture golden outputs.

> ⚠️ **The official `infer()` path is CUDA-oriented** (`.cuda()` + CUDA autocast), so a CPU bf16 HF oracle is **not guaranteed to run as-is (OQ-17)**. We therefore SPLIT the oracle: **(correctness)** the golden fixtures come from the **unmodified official model on a CUDA host**, frozen once and committed — parity NEVER depends on CPU HF; **(performance)** the CPU baseline is separate — either a CPU-patched HF run (force tensors to CPU, disable autocast; Phase −1/0 must PROVE it reproduces the GPU oracle's tokens within the nondeterminism floor of §8.2) or, if CPU HF is infeasible, the perf claim is stated against the best CPU reference that actually runs (llama.cpp GGUF / ONNX) and labeled as such. G2's "beats the CPU reference" is honest only against whatever CPU reference is *proven* to run.

`scripts/gen_reference_fixtures.py`:
- **Per-layer activation dumps** (post-patch-embed, post-SAM, post-bridge/compress, post-CLIP, post-projector, post-connector, per-decoder-layer hidden states, pre-lm_head, logits) as `.npy` (load via `fnp-io` pattern or our own reader).
- **End-to-end outputs**: decoded text + bbox tags for a golden image corpus.
- Also dump the **community NVFP4/GGUF** per-layer activations as secondary oracles (no framework runtime dependency — just frozen fixtures).

### 8.2 Isomorphism / parity gates (the layered ladder)

| Gate | Granularity | Tolerance |
|------|-------------|-----------|
| **L0 preprocessing** | resized/normalized/padded tensor, tile geometry | exact (gray pad 127, ratio selection, [-1,1] normalize must match) |
| **L1 per-op** | each kernel vs oracle activation | cosine ≈ 1.0 (≥ 0.9999 f32; documented int8 tolerance) |
| **L2 per-layer** | per decoder-layer hidden state, per vision-stage output | cosine ≈ 1.0; max-abs-diff ledgered |
| **L3 logits** | pre-sampling logits | within int8/int4 quant tolerance (frankensearch precedent `max_diff ~0.055`); **argmax must match** where reference is deterministic |
| **L4 token** | decoded token sequence | **exact** under greedy where reference deterministic |
| **L5 end-to-end OCR** | decoded text + bbox on golden corpus | exact-match where deterministic; aggregate **CER / TEDS / Formula-CDM within documented budget** |

The parity test shape (copied from frankensearch `parity_logits_and_ranking_match_reference`): the **discrete output (decoded text) must be bit-exact** where the reference is deterministic; **continuous values (logits) only within a documented quantization tolerance**; a separate **determinism gate** asserts same-input-twice → identical output. Every perf commit re-states its parity receipt (e.g. "text exact, max logit diff 0.05, deterministic").

**⚠️ Establish the oracle's OWN nondeterminism floor FIRST.** The HF reference is frequently non-deterministic across torch thread counts / BLAS reduction order at the logit-tie level (bf16, 129280-vocab argmax). So before setting any tolerance: run the oracle **twice, and at two thread counts**, over the golden corpus; diff; record the **nondeterminism envelope** (per-token divergence rate, first-divergence position) as a committed fixture. Then **L4 "exact" is defined only over the prefix the oracle reproduces identically**, and the **L3 logit tolerance is derived from the measured oracle variance**, NOT from the imported frankensearch `0.055`. A franken_ocr int8 divergence *inside the oracle's own bf16 noise* is not a bug.

### 8.3 Differential + metamorphic + golden

- **Differential** (`/testing-conformance-harnesses`): our path vs the oracle, per-op and end-to-end, on the golden corpus + the community quant artifacts.
- **Metamorphic** (`/testing-metamorphic`): properties that must hold without an oracle — e.g. identity-resize invariance; 90°-rotation/transpose relationships on bbox coords; padding a page with whitespace must not change recognized text; Base-mode determinism across runs. **⚠️ Do NOT assert "multi-page concat = sum of single-page parses."** Under **R-SWA the multi-page decode is cross-page DEPENDENT**: in a single 32K pass, *all* pages' visual+prompt prefixes form one frozen reference block, so page *N* can attend to pages `1..N-1` — output is **not** a concatenation of independent single-page parses (OQ-13). The defensible metamorphic property is closer to the opposite: changing page order / earlier-page content *may* change later-page output. Lock down the precise property only after OQ-13 confirms the reference-block span from `SlidingWindowLlamaAttention` / `infer_multi`.
- **Golden artifacts** (`/testing-golden-artifacts`): freeze known-good `focr ocr --json` outputs for the corpus (insta-style snapshots) with canonicalization (strip timing, sort bbox); exact comparison catches regressions. Frozen reference fixture (`tests/fixtures/native/<doc>_reference.json`) from the Python oracle is the bar.
- **Tokenizer conformance (OQ-16)**: a dedicated exact-match corpus — diverse Unicode, CJK, math, code, the DeepSeek glyph specials, `<image>`/`<|ref|>`/`<|det|>` — round-tripped `encode`→`decode` against `LlamaTokenizerFast` over `tokenizer.json`. **Token-id-exact is an L0/L4 prerequisite** (a tokenizer mismatch corrupts every downstream gate). Freeze the reference id sequences as golden fixtures.
- **Model-gated e2e**: skip-with-SUCCESS when the model is absent (CI stays green without 6.67 GB weights); prove the native path actually ran (point any fallback at `/nonexistent`).

### 8.4 Negative-evidence ledger & release scorecard

- **`docs/NEGATIVE_EVIDENCE.md` + `docs/PERF_LEDGER.md` are ARTIFACT-GRAPH ledgers**, not prose. Copy the frankentorch format (`date | WIN/NEGATIVE(reverted) | lever | MEASURED before→after vs reference (ratio) | bit-exact correctness proof | Disposition KEEP/REVERT | "do not retry X unless Y" | per-lever W/L/N`) and additionally carry, per entry/row, the **FrankenSuite artifact-graph fields**: `claim_id`, `evidence_id`, **model source commit + fixture hash** (from the Phase −1 truth pack), the **CPU feature string** (the dispatched SIMD tier), the **exact command + env**, and the **fallback / kill-switch state** — so every claim is reproducible and traceable to the exact model version it was measured against. Per-lever artifact dirs `artifacts/perf/<bead>/` with paired baseline/after gauntlet logs + SHA-256.
- **`docs/DISCREPANCIES.md`** — every accepted numeric divergence: reference behavior, our impl, **measured** impact, kill-switch env var, resolution, tests affected, review date.
- **Release-readiness scorecard** (`/running-the-gauntlet-on-your-rust-port`): a convergent multi-round honest evaluation — parity (L0–L5 all green), surface-parity (CLI/contract), honest perf vs reference (gauntlet ratios), determinism, cross-platform build matrix, ledger completeness. A release cannot ship with a red parity cell or an unledgered divergence.

### 8.5 The gauntlet — three-pillar release certification (ML-System-class)

`franken_ocr` is an **ML-System-class** port (the `/running-the-gauntlet-on-your-rust-port` class shared with frankentorch / franken_whisper). The gauntlet is run as the release gate; its machinery, adopted verbatim:

- **Three pillars, no victory-on-one-while-another-regresses.** **(a) Performance** — honest per-stage ratios vs the reference (§9). **(b) Conformance** — same answer as the bf16 reference (the L0–L5 ladder + metamorphic + fault). **(c) Surface parity** — a **FeatureUniverse / SurfaceMatrix** accounting every CLI surface, robot event, and modeling feature as `present | partial | missing | n/a | excluded` (partial never rounds up; excluded still counts as coverage debt).
- **Oracle wiring (ML-System-class):** a test-only PyO3/subprocess bridge to the pinned reference with **`torch.use_deterministic_algorithms(True)`**, seeded RNG captured per call, and a **per-op ULP tolerance table** (default **4 ULP f32 matmul, 2 ULP elementwise**) as the L1/L2 comparator — *not* a hand-guessed epsilon. This bridge is never linked into the shipping `focr` inference binary, preserving G3's no-FFI runtime claim. `EngineIdentity::{Subject, Oracle}` asserted-distinct so the oracle is never compared against itself.
- **Conformal lower-bound release ratchet** (`/alien-artifact-coding` conformal family): the parity score is a **Beta-posterior per category × a distribution-free conformal band**; the **release decision uses the LOWER bound, not the point estimate** (`truncate_score` to 6 dp). A change may only land if it raises the lower bound without lowering any per-category bound — this is the formal version of "parity gate first."
- **E-processes (Ville's inequality) for invariants:** the load-bearing invariants — *KV cache never exceeds `L·(m+128)`*, *int8 i32 accumulator never overflows*, *same-input determinism*, *SIMD path == scalar path bit-identical* — are monitored as **anytime-valid e-processes** (`p₀, λ, α` per the hardware/software calibration), alarming on the first genuine violation across an unbounded test stream without Bonferroni penalty.
- **Convergence rule:** ≥10 full rounds, ≥2 consecutive clean rounds (<3 new genuine findings), every open hypothesis resolved (the per-pillar hypothesis ledgers). `convergence-tracker` gates it.
- **Keep-gate for every perf claim** (§9.2): profile-first evidence ≥0.1% self-time *before* the source touch; both focused + broad benches from the same git state / same `target/` / same minute; `release-perf` profile only; `cv_pct` reported (>5% = noise, ineligible); **MT8 attribution** (the win names a specific ≥0.1% frame); pass-over-pass `.bench-history` ratchet.

### 8.6 Conformance-harness & golden-artifact mechanics (the two testing skills, concretely)

- **`/testing-conformance-harnesses`:** a `ConformanceTest` trait (`name / category / requirement_level{Must,Should,May} / run`) with a **coverage-accounting matrix** (every MUST/SHOULD clause of *our* spec — the extracted `EXISTING_UNLIMITED_OCR_STRUCTURE.md`, §10 — enumerated, ≥0.95 MUST coverage to claim conformance). Differential testing (Pattern 1) against the PyO3 oracle is the gold standard here. **Fixture provenance is mandatory** (`PROVENANCE.md`: `transformers==4.57.1`, `torch==2.10.0`, git ref, exact command). Intentional divergences are **`XFAIL`, never `SKIP`**, each a `DISC-NNN` in `DISCREPANCIES.md`.
- **`/testing-golden-artifacts`:** freeze known-good outputs and diff forever, with the right pattern per artifact: **exact** (insta snapshots) for `focr ocr --json` structure and CLI help; **fuzzy** (epsilon / ULP) for logits and per-layer activation tensors; **scrubbed** (timing, run-id, durations) for robot NDJSON; **canonicalized** (line endings, paths) for cross-platform. `UPDATE_GOLDENS=1` workflow with mandatory `git diff` review; `*.actual` gitignored; CI never auto-updates.
- **Spec-first porting (`/porting-to-rust`):** *extract spec → implement from spec → never translate line-by-line.* The four documents: this file is the **`PLAN_TO_PORT`**; we additionally produce **`EXISTING_UNLIMITED_OCR_STRUCTURE.md`** (THE SPEC — every data structure, the exact preprocessing/tiling rules, `SlidingWindowLlamaAttention` semantics, the projector concat order, tokenizer rules, extracted verbatim from `modeling_*.py`/`deepencoder.py`/`conversation.py`), **`PROPOSED_ARCHITECTURE.md`** (the Rust design), and **`FEATURE_PARITY.md`** (the running conformance scoreboard). After the spec is extracted, kernels are implemented **from the spec, not from the Python** — this is what resolves the §13 `[OPEN]`s.

---

## 9. Performance methodology

Apply `/profiling-software-performance` (profile-first, rank hot paths) **then** `/extreme-software-optimization` (behavior-proof each optimization), with EV-ranked levers and honest revert discipline; plus `/alien-graveyard` + `/alien-artifact-coding` for radical ideas.

### 9.1 Profile-first, rank hot paths

Never optimize before profiling. Baseline p50/p95/p99 wall-clock + memory for: `focr ocr` end-to-end, vision-encode-per-page, decode-per-token. Flamegraph/samply (use the `[profile.release-perf]` profile: `debug=line-tables-only, lto=thin, codegen-units=1`). **Profile across the OUTPUT-LENGTH axis, because the cost center MOVES with it**: a **sparse page** (few output tokens) is *vision-prefill-dominated*; a **dense page / 10-page / 40-page** parse is *decode-dominated*. The profiling corpus MUST span {sparse page, dense page, 10-page, 40-page}, and hot paths are ranked **per regime**. ⚠️ The ranking below is an **a-priori HYPOTHESIS to be REPLACED by measured profiles before any kernel lands** (it inherits a generation-heavy prior; short-output OCR may invert it): MoE expert GEMMs (decode) ≈ lm_head GEMV (decode) > R-SWA attention (decode, growing with page count §6.4) > vision attention/MLP (prefill) > patch-embed conv (prefill) > norms/activations.

**Roofline / compute-floor target (the reranker's clearest lesson).** For each stage compute the **compute floor** (int8/int4 GEMM FLOPs ÷ the arch's peak int8 throughput) and the **memory floor** (bytes streamed ÷ DRAM bandwidth), and take the max: that is what a perfect kernel would hit. In the sibling campaign, **ONNX-int8 ran *near* the compute floor while the Rust path sat ~2× above it**, and the gap was **kernels below peak** (SDOT-not-SMMLA linears, f32-not-int8 attention), *not* framework overhead — a naive fused-forward that traded good kernels for naive ones regressed 3–10× and *disproved* the overhead theory. So the perf target is explicit and falsifiable: **drive each stage to its roofline**, and treat any stage sitting >~1.3× above its floor as a named, attackable lever (which kernel, which arch), never an excuse. The roofline numbers go in `PERF_LEDGER.md` next to every measured ratio.

### 9.2 The optimization loop (mandatory, per frankentorch Performance Doctrine)

For each lever, the 5-pass loop (`.skill-loop-progress.md` shape): **(1)** claim + local baseline with SHA; **(2)** apply ONE lever + bit-exact correctness proof; **(3)** rebench + Criterion p-value + Score vs keep-threshold (e.g. ≥ keep); **(4)** revert if below threshold (NO source landed, ledger the failure); **(5)** route to the next profile-backed hotspot. Correctness (L0–L5 parity) re-proven on every perf commit. **Honest revert discipline**: a lever that loses is reverted with no source landed and memorialized in `NEGATIVE_EVIDENCE.md` (the way "naive wide SIMD 5× slower" lives in the kernel comment).

### 9.3 Head-to-head gauntlet vs the real reference

`benches/gauntlet` shells out to the **Phase -1 proven CPU reference** (`FOCR_REFERENCE_CMD` / `FOCR_REFERENCE_PYTHON` env) per stage, parses elapsed, reports the **honest focr/reference ratio** (tag each row OK/warn/slower/"focr faster") — not a self-relative number. If CPU-patched HF cannot be proven equivalent to the CUDA oracle, the benchmark target is llama.cpp GGUF / ONNX Runtime / MLAS and is labeled that way. This is how G2 is proved, and it is only meaningful with **apples-to-apples fairness controls, all mandatory**:
- **Thread parity**: pin `OMP_NUM_THREADS` / torch `set_num_threads(N)` **equal to** focr's thread budget; **never benchmark torch at @64** (oversubscription inflates fake "wins" — a hardened frankentorch lesson; measure at @8/@32).
- **Allocator fairness**: build focr with the same allocator posture used for the claim (mimalloc behind a feature, §9.6), wired into the measured binary, not merely mentioned.
- **Best-of-N with warmup discard**; report the min and the precision of each side.
- **Precision annotation per row**: focr-int8 vs torch-bf16 (and torch-int8 if available) — a raw ratio across different numerics is meaningless without it.
- **PER-STAGE ratios** (preprocess / vision-encode / prefill / decode-per-token), so the honest story is visible: *decode faster, vision-prefill maybe slower in f32 v1* (§1.1 G2).

### 9.4 EV-ranked levers (candidates — pick biggest-EV first, record the decision)

Per the frankensearch precedent ("the largest perf lever is doc-level parallelism (~Ncores), not the dtype"), the likely biggest lever is **full-core-parallel forward**, *then* int8/int4 dtype on the hot GEMMs. Concrete EV-ranked candidates:

1. **Full-core-parallel forward** (kernel rayon pool, sequential outer loop) — likely highest EV, dtype-independent.
2. **int8 decoder GEMMs (expert + lm_head)** — biggest memory-traffic cut per token.
3. **int4-group expert weights** — biggest size win (experts are the bulk of params).
4. **R-SWA constant ring buffer** — preallocated, cache-resident, no realloc (free correctness+speed from the architecture).
5. **lm_head: argmax-only under greedy** (skip full softmax over 129280).
6. **Pre-packed arch-specific weight tiles** — zero runtime shuffle.

### 9.5 Radical ideas (`/alien-graveyard`, `/alien-artifact-coding`) — candidates to evaluate behind proofs

Each is a *hypothesis* to EV-rank, prototype, and behavior-prove (or revert + ledger):

- **Shape-specialized `const`-generic kernels** — bake 1280/896/128/6848 as const generics, eliminate runtime shape branching and remainder loops in the hot path.
- **Fused layers** — dequant+GEMM+bias+SiLU fused; RMSNorm+RoPE fused; router+top-6-gather fused; gate_proj/up_proj computed together then SiLU-multiplied.
- **Native-resolution token packing** — pack the deterministic per-page feature budget (256 base image-feature tokens / ~100 per 640 crop) plus connector structural tokens from the OQ-18 census into fixed contiguous buffers so prefill GEMMs are remainder-free.
- **Conformal / sequential-test early-exit decode (`/alien-artifact-coding`)** — a cheap draft (shared-experts-only or reduced top-k) proposes tokens, verified by the full forward; early-exit on easy (printed-text) regions **only under a calibrated guarantee**: a conformal / sequential-test gate (e.g. an SPRT / e-value test on the draft-vs-full logit margin) that **bounds the per-token disagreement probability at a chosen risk level α**, turning "high risk for exact-token OCR" into a *provable* token-flip bound. Behavior-prove no CER regression at α; revert + ledger if the bound is not met.
- **Decision-theoretic per-tensor quant (`/alien-artifact-coding`)** — frame int8-vs-int4-vs-tier per tensor as minimizing an **expected end-to-end loss with an explicit tail bound** (state = per-tensor bit choice; loss = measured CER impact, weighted toward dense numbers/tables where each token must be exact; constraint = total footprint), NOT a uniform bit-width. The accuracy cliff is below 4-bit, so the optimizer must respect a worst-case (not just mean) CER bound on the exact-token-sensitive corpus.
- **Reference-block KV sharing across pages** (multi-page) — **GATED ON OQ-13.** *Only sound if* the multi-page reference block does **not** make page *N* depend on `1..N-1`; if it does (the likely R-SWA reality, §8.3), pages are **not** independent and naive KV-sharing would silently corrupt later pages. If OQ-13 permits, exploit the constant-KV property for memory-flat parses — but behavior-prove against the **multi-page** reference output, never against single-page sums.
- **Outlier-aware int8 vision** (register-prefix / per-patch) — only if decoder parity is locked and vision quant is later shown to hold CER.

### 9.6 Honest perf hygiene

Allocator-fair comparison (mimalloc behind a feature for gauntlet fairness). Benchmarks **skip gracefully** (exit 0) if the model fixture is absent. A benchmark-guardrail gate catches regressions vs a frozen baseline. `PERF_LEDGER.md` records the honest ratio table.

### 9.7 Alien-artifact math families (compiled to runtime artifacts, not name-dropped)

Per `/alien-artifact-coding` (complexity budget for a >100K-LOC perf-critical project = **3–5 families**, natural-fit only, each compiled to a concrete artifact with proof obligations + a deterministic conservative fallback) and `/alien-graveyard` (EV = Impact·Confidence·Reuse / Effort·Friction ≥ 2.0, S/A/B/C tier, fallback trigger). These are the genuinely high-leverage, non-obvious levers — selected by *measured failure signature*, not novelty. Each is a `NEGATIVE_EVIDENCE`-gated hypothesis until proven.

**AF-1 — Rate-distortion + convex duality (water-filling) for optimal per-tensor quant allocation. [Tier A, EV high]**
The choice of {BF16/F32, int8, int4-g32, int4-g16} per quantizable tensor under a total-footprint budget *is* a rate-distortion problem: minimize end-to-end distortion `Σ_t D_t(b_t)` subject to `Σ_t bits_t(b_t) ≤ B`. The Lagrangian `Σ_t [D_t(b_t) + λ·bits_t(b_t)]` separates per-tensor; sweeping λ traces the **optimal bit-allocation frontier (water-filling)** — spend bits where the marginal distortion-per-bit `∂D_t/∂bits` is steepest (the wide `intermediate=6848` dense `down_proj`, the attention `v_proj`), starve the flat ones. *Artifact:* an offline `bit_allocation_table` in the `.focrq` header, computed by `focr convert --optimize-bits --budget <GB>` from per-tensor distortion curves `D_t(b)` measured as the layer-output cosine drop on a calibration batch. *Proof obligation:* the allocated config's end-to-end CER ≤ the uniform-bit config's at equal footprint, on the dense-numeric corpus. *Fallback:* uniform Q4_K_M-class allocation (§5). Replaces uniform bit-width (the naive baseline) and is the principled form of the §9.5 "expected-loss-guided per-layer quant."

**AF-2 — Distributionally-robust / tail-risk (CVaR + EVT) for worst-case CER. [Tier A, EV high]**
OCR fails in the **tail**: most pages are fine, but a quant choice that wrecks dense tables/sub-scripts/code is unacceptable even if mean-CER looks great (perplexity under-predicts exact-token failure). So the quant objective and the release gate optimize/bound **CVaR_α (the mean of the worst α-fraction of per-document CER)** and fit an **EVT (generalized-Pareto) tail** to the CER distribution to estimate the 99.9th-percentile document. *Artifact:* a `tail_risk_monitor` over the golden corpus emitting `(mean_CER, CVaR_0.1, EVT_p999)`; the release scorecard (§8) gates on the **CVaR/tail bound**, not the mean. *Proof obligation:* int4 CVaR_0.1 within the ledgered budget of f32. *Fallback:* keep the tail-offending tensor one precision tier higher (the llama.cpp `_M` discipline, derived rather than guessed).

**AF-3 — Conformal / sequential testing (SPRT, e-values) for provably-safe early-exit & speculative decode. [Tier B, EV medium-high]**
Speculative/early-exit decode is high-EV (skip the full forward on easy printed-text runs) but high-risk for exact-token OCR. Make it *provably* safe: gate each early-exit on an **anytime-valid sequential test (e-value / SPRT)** that bounds the probability the cheap draft disagrees with the full model at risk level α — accept the draft token only while the e-process stays below `1/α`; else fall back to the full forward. This converts "risky" into a **finite-sample token-flip guarantee** (Ville's inequality — the same machinery the gauntlet uses for invariants). *Artifact:* a `speculative_guard` e-process per decode step + a calibration fixture. *Proof obligation:* measured token-disagreement rate ≤ α on the golden corpus; no CER regression. *Fallback:* α→0 disables speculation (always full forward).

**AF-4 — Submodular maximization for the high-precision tensor set. [Tier B, EV medium]**
"Which tensors stay high-precision under a footprint budget" is a **monotone submodular maximization under a knapsack constraint** (accuracy-recovered is submodular in the kept set — diminishing returns). Greedy selection gives a `(1 − 1/e)` approximation guarantee — near-optimal, cheap, and *certifiable*, vs the hand-picked "keep vision/router/norms" heuristic (which greedy should recover and may refine). *Artifact:* `focr convert --select-high-precision-set --budget` emitting the chosen set + the marginal-gain ledger. *Proof obligation:* greedy set's CER ≤ heuristic set's at equal footprint. *Fallback:* the validated NVFP4/GGUF heuristic set (§2.6).

**AF-5 — Queueing theory (Universal Scalability Law) for many-core pool sizing. [Tier A, EV high, low effort]**
(Detailed in §6.9.) Fit USL `C(N) = N/(1+α(N−1)+βN(N−1))` to a thread-sweep of the decode GEMV and the prefill GEMM separately; **cap each pool at its USL peak** rather than `num_cpus`. On a 64-core Threadripper this is the difference between decode scaling to ~8–16 effective cores (bandwidth-bound) and pointlessly oversubscribing 64 (β-dominated slowdown). *Artifact:* a `pool_sizing` table per (arch, op-class) baked from the sweep; `focr robot backends` reports it. *Proof obligation:* measured throughput at the chosen N ≥ throughput at `num_cpus`. *Fallback:* physical-core count.

**Diminishing-returns discipline (per the skill):** stop escalating math when the dominant metric is within ~10–20% of optimal, the bottleneck becomes operational, or verification cost exceeds the gain. AF-1/AF-2/AF-5 are the load-bearing trio (quant quality + many-core); AF-3/AF-4 are upside levers behind their guarantees. Every family ships a **galaxy-brain transparency card** (equation · substituted values · plain-English intuition · validity assumptions · what would flip the decision) and an **assumptions ledger**; none ships without its deterministic fallback wired first.

---

## 10. Phased roadmap

Each phase: **goals · key tasks · exit gates**. Correctness before speed throughout. An exit gate cannot pass while it depends on an unresolved **[OPEN]** in §13.

### Phase −1 — Source/Oracle Truth Pack (the executable foundation — do this FIRST)

Per the `/idea-wizard` review, the single highest-EV move is **not a kernel** — it is an *executable truth pack* that stops the project from optimizing the wrong model semantics (the model changed on 2026-06-24).

- **Goals**: pin the exact sources, hash them, and answer every `[OPEN]`/`OQ` with a line-backed citation — *before* any kernel exists.
- **Tasks**: **pin the HF repo commit and GitHub commit** (Codex-reported `HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`, `GitHub 7e98affeacba24e95562fbaa234ddb89b856874a` — **verify against the live repos and record the verified hashes**); snapshot + SHA-256 every load-bearing source (`config.json`, `tokenizer.json`, `modeling_unlimitedocr.py`, `modeling_deepseekv2.py`, `deepencoder.py`, `conversation.py`, `model.safetensors.index.json`, `LICENSE`); generate the **machine-readable token/shape/buffer census** from the pinned config (§2.4, §8.6); **resolve every `[OPEN]`/OQ (§13) with a quoted, line-backed answer** from the pinned source; **decide the oracle strategy (OQ-17)** — prove CPU HF runs, else GPU-correctness + CPU-perf split (§8.1) — and capture one smoke fixture. All under `docs/truth-pack/` with a `claim_id → source-line` index.
- **Exit gates**: every OQ answered with a pinned-source citation; all source hashes recorded; census generated + CI-guarded against source drift; oracle strategy decided + smoke fixture captured. **No kernel work begins until this phase is green.**

### Phase 0 — Scaffold
- **Goals**: the `franken_whisper`-shaped skeleton compiles and runs end-to-end empty; toolchain/deps/CI wired.
- **Tasks**: single-crate layout, two `[[bin]]`, path deps (ft-kernel-cpu/ft-core/ft-serialize, asupersync `>=0.3.5,<0.4` default-features=false, fsqlite, image), `#![forbid(unsafe_code)]` + lint, `OcrEngine` owning the asupersync Runtime, sync `main()` + ShutdownController, robot NDJSON skeleton + `robot schema`, fsqlite `RunStore` + `_meta` schema, `convert`/`ocr`/`doctor` stubs, model-gated test harness, 5-target CI matrix (advisory), `docs/{DISCREPANCIES,NEGATIVE_EVIDENCE,PERF_LEDGER}.md` seeded, `scripts/fetch_model.sh` + `gen_reference_fixtures.py` skeleton.
- **Exit gates**: `cargo build` green on all 5 targets; `focr robot schema`/`health`/`backends` emit valid versioned JSON; empty-pipeline e2e runs and skips-green without weights; reference oracle env reproducibly produces fixtures.

### Phase 1 — fp32 reference-parity (correctness before speed)
- **Goals**: a **pure-f32**, framework-free forward that matches the bf16 reference within f32 tolerance, end-to-end, on the golden corpus. **No quantization, no hand-SIMD yet.**
- **Tasks**: **FIRST extract the spec** — `EXISTING_UNLIMITED_OCR_STRUCTURE.md` (`/porting-to-rust`): every data structure, the exact preprocessing/tiling rules, `SlidingWindowLlamaAttention` semantics, projector concat order, tokenizer rules — read from `modeling_*.py`/`deepencoder.py`/`conversation.py`, **resolving every kernel-blocking `[OPEN]`/`OQ` (§13)**. THEN implement *from the spec, not the Python*: pure-Rust tokenizer (tokenizer.json BPE, conformance-tested OQ-16); preprocess front end (decode/resize/pad/normalize/Base+Gundam tile, bicubic pos-embed interp); SAM-ViT-B (patch-embed conv, window/global attn, neck, 16× compression); CLIP-L (SDPA + quick_gelu); feature concat→projector; connector (structural tokens + masked-scatter); decoder (RMSNorm, RoPE, **R-SWA** with fixed ring buffer + online softmax §6.8, MoE top-6 + shared + dense layer-0, final norm); lm_head + greedy sampler + no_repeat_ngram(35); postprocess (EOS, tag regex, bbox /999, markdown).
- **Exit gates**: L0–L5 parity all green in f32 (per-layer cosine ≥ 0.9999, decoded text exact where reference deterministic, CER ≈ reference); determinism gate green; WeightsManifest census passes; no `[OPEN]` blocking any implemented kernel.

### Phase 2 — int8
- **Goals**: weight-only int8 decoder (the validated recipe), bit-identical-where-integer, OCR accuracy within documented int8 tolerance of f32.
- **Tasks**: `.focrq` format + `focr convert --quant int8` (per-output-channel symmetric, vision/projector/embeddings/router/norms kept **bf16**); per-row dynamic activation quant; round-trip/determinism gate. **Stage the quantization to de-risk — one lever at a time (`/extreme-software-optimization`), each its own parity gate + ledger entry**: **(2a)** int8 the **FFN / expert GEMMs ONLY** (the NVFP4/GGUF-validated set) → prove parity; **(2b)** THEN int8 **attention `q/k/v/o`** behind a separate `FOCR_INT8_ATTN` kill-switch → prove parity (OQ-14 risk); **(2c)** THEN int8 **`lm_head`** behind its own `FOCR_INT8_LMHEAD` kill-switch → prove parity. A regression in any stage reverts just that stage.
- **Exit gates**: int8 end-to-end CER within ledgered budget of f32 (target: within noise, matching NVFP4's "identical to BF16" bar); logits within tolerance, decoded text exact where deterministic; round-trip deterministic; size ≈ Q8_0-class.

### Phase 3 — multi-arch SIMD kernels
- **Goals**: the int8 GEMMs run on native intrinsics across arches; **the DECODE path is demonstrated faster than the Phase -1 proven CPU reference** (the defensible part of G2; end-to-end-faster is a later stretch, §1.1).
- **Tasks**: build the **per-arch SIMD dispatch catalog (§6.6)** — tiled **SMMLA / AMX prefill GEMM** + SDOT/VNNI decode GEMV, runtime feature dispatch (`robot backends` reflects it); offline arch-specific weight pre-packing (`--arch`); **MoE token-grouping (§6.7)**, **online-softmax R-SWA (§6.8)**, **fusion (§6.10)**, **vectorized transcendentals (§6.11)**, **NUMA + USL many-core pool sizing (§6.9 / AF-5)**, **memory/allocator (§6.12)**, **PGO + BOLT on the release binary (§6.13)**; profile-first → the §9.2 optimization loop on the **per-regime** ranked hot ops; gauntlet head-to-head vs reference (§8.5).
- **Exit gates**: bit-identical to scalar (integer accumulation); **gauntlet per-stage ratios recorded with the fairness controls of §9.3, and the decode-per-token path is faster than the Phase -1 proven CPU reference** on the primary arches (vision-prefill parity-or-slower in f32 is acceptable and recorded honestly); every landed lever has a `NEGATIVE_EVIDENCE.md`/`PERF_LEDGER.md` entry; parity still green.

### Phase 4 — int4 refinement
- **Goals**: int4-group decoder weights at Q4_K_M-equivalent, accuracy regression inside a small **measured** budget.
- **Tasks**: `focr convert --quant int4` (per-group 16–32, `.v_proj`/expert `.down_proj` one tier higher); int4 GEMM kernels; **expected-loss-guided per-layer quant** (choose tier by measured CER impact); dense-text accuracy curve per bit-width (tables/code/numbers/sub-superscripts — exact-token-sensitive corpus).
- **Exit gates**: int4 CER within ledgered budget (NOT below the 4-bit cliff); size ≈ 1.8–2.9 GB; no catastrophic regression on dense numeric/table content; every quant choice ledgered with its measured impact.

### Phase 5 — CLI hardening + cross-platform release
- **Goals**: production single-binary release, agent-ergonomics maxed.
- **Tasks**: finalize subcommands/exit codes/env overrides; robot contract tests + frozen schema; `doctor` (idempotent, reversible, capability reflection); streaming polish; **run the three-pillar gauntlet to convergence (§8.5) and produce the release-certification bundle**; 5-target release build + `install.sh` + checksums (`/release-preparations`); agent-ergonomics audit (`/agent-ergonomics-and-intuitiveness-maximization-for-cli-tools`); release-readiness scorecard.
- **Exit gates**: all 5 targets build + smoke-test; robot schema stable + self-describing; scorecard all-green; install.sh verified; ergonomics score above bar.

### Phase 6 — CUDA stretch
- **Goals** (explicitly stretch): optional CUDA acceleration of the hottest GEMMs, behind a feature flag, with **identical OCR output** to the CPU path.
- **Tasks**: GPU int8 GEMM for expert/lm_head; keep CPU as the reference and fallback; CPU↔GPU parity gate.
- **Exit gates**: GPU path bit-equivalent-where-integer to CPU; CPU remains default + fully functional; no regression to CPU path. (May be deferred indefinitely; CPU is the product.)

---

## 11. Risks & mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **License** — redistributing a quantized derivative | LOW (resolved) | **[VERIFIED] MIT** on both HF + GitHub; we may redistribute provided we ship the "Copyright (c) 2026 Baidu — MIT" notice (embedded in `.focrq` `license_notice`, in `--version`, in README). |
| **Accuracy under quant for small/dense text** | HIGH | Validated recipe (decoder-only weight quant, vision/projector/embeddings/router/norms high-precision) already shown OCR-lossless (NVFP4) / production-safe (Q4_K_M). int8 first as oracle; int4 ≥ Q4_K_M-equivalent (above the 4-bit cliff); gate on **CER over dense numeric/table corpus**, not perplexity; expected-loss-guided per-layer quant; keep `.v_proj`/expert `.down_proj` one tier higher. lm_head int8 behind a measured kill-switch. |
| **Scope creep** | MED | One model, inference only, no autograd, no multi-backend router, Base mode default (Gundam opt-in). Phase gates with hard exit criteria. Non-goals (§1.2) enforced. |
| **Cross-compile / nightly** | MED | nightly required for stdarch i8mm/dotprod (toolchain pinned, already present); scalar fallback **cross-compiles to every target**; runtime dispatch → one binary per arch; 5-target CI from Phase 0 (advisory, never blocks build). Confine nightly-`portable_simd` to kernel islands; rest stays stable-buildable. |
| **`[OPEN]` model details wrong** | MED-HIGH | Hard rule: no kernel ships against an unresolved `[OPEN]`; §13 register gates Phase 1; read `modeling_*.py`/`deepencoder.py`/`conversation.py` and the actual `SlidingWindowLlamaAttention` before building R-SWA/RoPE/projector; manifest census catches wrong/stale weights at load. |
| **Deadlock (nested rayon under lock)** | MED | Architectural fix: single model + sequential outer loop, forward fans out internally; never `par_iter` over a held lock; never nest a 2nd asupersync runtime; `many_pages_without_deadlock` CI watchdog. |
| **Vision int8 trap** (ViT activation outliers) | MED | Keep vision tower BF16/F32 in v1 (validated by both prior quants); int8 vision is a later isolated experiment only after decoder parity. |
| **frankentorch gaps** (no image I/O, no f32 bicubic, no GPU) | LOW-MED | Build the image front end fresh (`image`/`fast_image_resize`); build f32 bicubic; GPU is out of scope (Phase 6 stretch) and CPU-only frankentorch is *aligned* with our priority. |
| **Reference-version drift** (config 4.46.3 vs runtime 4.57.1) | LOW | Pin the oracle to README's runtime (`transformers==4.57.1`, `torch==2.10.0`); freeze fixtures; record the pin. |
| **End-to-end may not beat the CPU baseline in v1** (f32 vision tower vs MKL/oneDNN; frankentorch's own profiling tags it slower on some ops) | MED | **G2 reframed (§1.1) to per-stage honesty**: first prove which CPU baseline is legitimate (CPU-patched HF, llama.cpp, or ONNX); target the *decode* win (int8/int4), report vision-prefill parity-or-slower openly; "end-to-end faster" is a post-int4 / optional-int8-vision stretch, not a v1 gate. The unbuilt tiled SMMLA/VNNI GEMM (§3.2, §6.2) is the lever that narrows it. |
| **PDF rasterization parity** (CLI/README must not over-promise) | MED | **v1 is image-only (§7.7)**; PDFs rasterized out-of-band. Native PDF only behind an explicit decision (pdfium feature vs pure-Rust + a rasterization-parity gate vs pymupdf@300DPI); `pdf` stays off the CLI surface until then. |
| **Tokenizer mismatch** (pure-Rust BPE vs `LlamaTokenizerFast`) | MED | A wrong token id corrupts every downstream gate; dedicated tokenizer conformance corpus (OQ-16, §8.3), token-id-exact, frozen golden id sequences, before any decoder parity work. |

---

## 12. Success metrics

**Correctness (G1, gating)**
- L0–L5 parity all green; per-layer cosine ≥ 0.9999 (f32), decoded text exact where reference deterministic.
- End-to-end CER / TEDS / Formula-CDM within documented quantization budget: **int8 within noise** of bf16 (NVFP4 "identical to BF16" as the bar); **int4 within a small ledgered budget**, no catastrophic regression on dense numeric/table content.
- Determinism: same image+args → byte-identical output under greedy.

**Performance (G2, gating — per-stage honest, §1.1)**
- Gauntlet: **decode-per-token faster than the Phase -1 proven CPU reference** on the primary arches (the gating part); **vision-prefill ratio recorded honestly** (parity-or-slower acceptable in f32 v1); end-to-end-faster is a tracked *stretch*. Honest best-of-N with thread/allocator/precision fairness (§9.3) in `PERF_LEDGER.md`.
- Generated-token KV **memory** stays bounded across output length (R-SWA property preserved), with a preallocated reference-block cap from the token census. The logical reference block `m` and per-token *compute* still grow with page/input length up to the 32K context (§6.4) — not claimed page-constant.
- Decode throughput improves with int8/int4 (memory-bound win); the **built** tiled SMMLA/VNNI prefill ≥ ~2× the scalar prefill where measured, narrowing toward the ONNX/MLAS bar (§3.2).

**Footprint & portability (G3, G4)**
- One self-contained `focr` binary per target (5 targets), no Python/FFI/network/GPU at inference.
- `.focrq` size, with the **high-precision set kept bf16** (vision tower ≈ 774 MiB; `embed_tokens` 129280×1280 ≈ 330 MB bf16; **untied `lm_head`** another ≈ 330 MB bf16 unless int8'd; projector + router + norms small): **int8** decoder ≈ Q8_0-class; **int4-group** decoder ≈ **1.8–2.9 GB** *only if the arithmetic closes* — NVFP4 keeping embeddings + lm_head + vision in BF16 already lands at 2.93 GB, so hitting < 2.9 GB with int4 experts may require extra measured experiments such as int8 `lm_head` and possibly embeddings. Those are outside the validated recipe and must be gated by CER, kill-switches, and `DISCREPANCIES.md` entries. Switching bf16→f16 saves **nothing** — both are 2 bytes. The converter prints the realized per-section footprint; the target is a **demonstrated** number, not an assumed one.
- `#![forbid(unsafe_code)]` everywhere except named SIMD islands, each with a bit-identical scalar fallback.

**Agent ergonomics & honesty (G5, G7)**
- Stable versioned robot/NDJSON schema, self-describing, contract-tested; stable exit codes; `--json` everywhere.
- Every accepted divergence ledgered (`DISCREPANCIES.md`); every rejected optimization ledgered (`NEGATIVE_EVIDENCE.md`); release-readiness scorecard all-green before ship.
- Agent-ergonomics audit score above bar.

---

## 13. Open research questions register

**These MUST be resolved (by reading the actual model source / config / processor) before the dependent kernel ships. A phase exit gate cannot pass while it depends on an unresolved item here.**

| ID | Question | Blocks | Source to read |
|----|----------|--------|----------------|
| **OQ-1** | R-SWA reference set `m`: visual-only or visual+prompt? Exact mask boundary (prompt vs generated). | R-SWA attention (`rswa.rs`), prefill | `SlidingWindowLlamaAttention` in `modeling_deepseekv2.py` |
| **OQ-2** | Is attention uniformly R-SWA across all 12 layers (incl. layer 0)? Any layer with full attention? | decoder attention | `modeling_deepseekv2.py` |
| **OQ-3** | Exact warm-up vs ring-overwrite mask semantics during the first 128 decode steps. | R-SWA ring buffer | `SlidingWindowLlamaAttention` |
| **OQ-4** | Q/K head dim when `use_mla=false` and `qk_nope/rope_head_dim=0` (likely 128 from `q_proj` shape). | attention shapes | `modeling_deepseekv2.py` (q_proj weight shape) |
| **OQ-5** | `rope_theta` value (DeepseekV2 default 10000 assumed); is YARN/NTK active at 32K? | RoPE kernel | `configuration_deepseek_v2.py` / config `language_config` |
| **OQ-6** | Exact SAM⊕CLIP feature-fusion / concat order producing the 2048-dim projector input. | projector, vision bridge | `deepencoder.py` (line-level) |
| **OQ-7** | Gundam-mode exact vision-token count per aspect ratio; `dynamic_preprocess` `min_num`/`max_num` defaults (reported 2..32). | preprocess tiling, buffer sizing | `modeling_unlimitedocr.py` `dynamic_preprocess` |
| **OQ-8** | Full prompt-mode taxonomy (free OCR / layout / grounding / markdown) and whether bboxes are emitted in all modes. | postprocess, CLI prompt contract | `conversation.py` / processor |
| **OQ-9** | Exact activated-param count (500M vs 570M). | docs/claims only (not kernel-blocking) | paper Fig 2 / recompute from safetensors |
| **OQ-10** | Whether community GGUF (PR #17400) quantizes router/experts identically to NVFP4 (router-gate-in-f32). | quant recipe cross-check (not blocking) | GGUF repo / PR #17400 |
| **OQ-11** | Is int8 viable for the patch-embed conv specifically (cleaner stats than mid-network ViT)? | later vision-int8 experiment | isolated experiment (Phase 4+) |
| **OQ-12** | Benchmark cells (TPS, v1.6 peers, 40+pg edit distances) — confirm from PDF tables vs secondary sources. | docs/claims only | `Unlimited-OCR.pdf` tables |
| **OQ-13** | **Does R-SWA's reference block span ALL pages in a multi-page pass (making page *N* depend on pages `1..N-1`)?** Determines the §8.3 metamorphic gate AND whether the §9.5 cross-page reference-KV "sharing" idea is even sound. | §8.3 metamorphic gate, §9.5 KV-sharing, multi-page correctness | `SlidingWindowLlamaAttention` + `infer_multi` in `modeling_*.py` |
| **OQ-14** | **Enumerate the EXACT ~2196 linear modules NVFP4 quantizes** (dump its scale keys) and reconcile vs our ~2229 census — does it quantize attention `q/k/v/o`? `lm_head`? shared experts? Determines which quant choices are validated-by-prior-art vs **our** CER risk. | §2.6 recipe scope; int8-attention & int8-`lm_head` validation (§5, §6) | `sahilchachra/Unlimited-OCR-NVFP4` tensor/scale listing |
| **OQ-15** | **SAM windowed-attention window size (assumed 14) and the windowed-block position-embedding scheme** — inherited from the SAM/DeepSeek-OCR lineage, NOT config-verified. A wrong window/pos-embed silently corrupts vision features (surfaces only as a fuzzy L2 cosine). | SAM window attention (`vision_sam.rs`), L2 parity | `deepencoder.py` (line-level) |
| **OQ-16** | **Tokenizer parity**: exact byte-level-BPE encode/decode over `tokenizer.json` (9.98 MB, ~512 special tokens) vs `LlamaTokenizerFast` — pre-tokenizer regex, byte-fallback, special-token + DeepSeek-glyph handling. | pure-Rust tokenizer (`tokenizer/`), L0/L4 parity | `tokenizer.json` + a `LlamaTokenizerFast` round-trip corpus |
| **OQ-17** | **Does the official model run on CPU at all?** `infer()` is CUDA-oriented (`.cuda()` + autocast). Determine whether a CPU-patched HF run reproduces the GPU oracle's tokens — if not, perf is benchmarked against llama.cpp-GGUF / ONNX on CPU instead. Decides the oracle split (§8.1) and the meaning of G2. | §8.1 oracle, G2 perf claim, Phase −1 | `modeling_unlimitedocr.py` `infer()` (device/autocast) + a CPU smoke run |
| **OQ-18** | **Full token census per mode** — every token CLASS (image-feature 256, `image_newline` 16, `view_seperator` 1 ⇒ 273 at 1024; prompt; generated) and the exact **Gundam multi-tile** totals per aspect ratio; the `ngram_window` per mode (128 single / 1024 multi). Drives all KV/buffer sizing + the sampler. | preprocess, R-SWA buffer sizing, sampler, the census generator | `modeling_unlimitedocr.py` (`dynamic_preprocess`, placeholder assembly, ngram window) |

---

## 14. Skills, methodology & the path to beads

This plan is built to be executed through a specific set of skills. Each is named where it applies so that the upcoming **`/beads-br` + `/beads-workflow`** pass can turn this document into a dependency-aware work graph with the right methodology attached to each bead.

### 14.1 The named skills and where each one governs

| Skill | Role in franken_ocr | Primary sections / artifacts |
|-------|---------------------|------------------------------|
| **`asupersync-mega-skill`** | The runtime foundation: sync-shell `main`, engine-owns-one-`Runtime`, `spawn_blocking`+`timeout` budgets, Cx/checkpoint cancellation, **no nested runtime / no rayon-under-lock**. | §3.3, §4.2, §6.5 |
| **`/porting-to-rust`** | Spec-first discipline: extract `EXISTING_UNLIMITED_OCR_STRUCTURE.md` from the reference, then implement *from the spec*. Resolves the §13 `[OPEN]`s. | §8.6, §10 Phase 1 |
| **`/running-the-gauntlet-on-your-rust-port`** | The release gate: ML-System-class three-pillar (perf / conformance / surface) convergent certification, conformal lower-bound ratchet, e-processes, FeatureUniverse, keep-gate, ≥10-round convergence. | §8.5, §9.2, §10 Phase 5 |
| **`/testing-conformance-harnesses`** | `ConformanceTest` trait + MUST/SHOULD coverage matrix + fixture provenance + `XFAIL`/`DISCREPANCIES.md`. | §8.6 |
| **`/testing-golden-artifacts`** | Frozen golden outputs per artifact type: exact (insta), fuzzy (logits/ULP), scrubbed (NDJSON), canonicalized (cross-platform); `UPDATE_GOLDENS` review workflow. | §8.3, §8.6 |
| **`/profiling-software-performance`** | Profile-FIRST: baseline fingerprint + ranked hotspot table + hypothesis ledger under `tests/artifacts/perf/`, per output-length regime + per arch. No hotspot → no change. | §9.1 |
| **`/extreme-software-optimization`** | The Loop (baseline→profile→**prove isomorphism**→implement Score≥2.0 one-lever→verify golden→repeat); Opportunity Matrix; isomorphism-proof template (ideal for bit-exact int8). | §9.2, §6.10 |
| **`/alien-graveyard`** | EV-ranked lever harvesting (EV = Impact·Confidence·Reuse / Effort·Friction ≥ 2.0), recommendation contracts with fallback, and the **Step-8 beads operationalization** that feeds the beads pass. | §9.7 |
| **`/alien-artifact-coding`** | Compile advanced math to concrete artifacts with proof obligations + deterministic fallback: the AF-1..AF-5 families (rate-distortion bit-allocation, tail-risk CER, conformal early-exit, submodular selection, USL pool-sizing) + galaxy-brain transparency cards. | §9.7 |

### 14.2 Supporting ecosystem skills (used during execution, not the plan's spine)

`/deadlock-finder-and-fixer` (audit the rayon/lock topology — the deadlock saga must not recur) · `/rust-unsafe-code-exorcist` + `/rust-undefined-behavior-exorcist` (audit the SIMD `unsafe` islands; every load a `// SAFETY:`) · `/testing-fuzzing` + `/testing-metamorphic` (differential fuzz the tokenizer + image front end; metamorphic image transforms) · `/profiling-software-performance` LLM-AI reference (TTFT/TPOT/decode-per-token discipline) · `/agent-ergonomics-and-intuitiveness-maximization-for-cli-tools` (the `focr` robot surface, Phase 5) · `/release-preparations` + `/rust-crates-publishing` (Phase 5 cross-platform release) · `/beads-br` + `bv --robot-*` (the work graph) · `/cass` (mine prior reranker/kernel sessions before each perf bead — the negative-ledger mandate).

### 14.3 The path to beads (the next step, after this plan is approved)

Run **`/beads-br`** then **`/beads-workflow`** to convert this document into the work graph. The mapping:

- **Epics = the §10 phases** (Phase 0 scaffold → Phase 6 CUDA stretch), plus a **gauntlet-certification epic** (§8.5) and a **spec-extraction epic** (§8.6 / `EXISTING_UNLIMITED_OCR_STRUCTURE.md`).
- **Every `[OPEN]` / `OQ-N` (§13) becomes a P0 research bead** that *blocks* the kernel bead depending on it (the hard rule: no kernel ships against an unresolved OQ). `OQ-1..3` (R-SWA), `OQ-6` (projector concat), `OQ-13` (multi-page span), `OQ-15` (SAM window), `OQ-16` (tokenizer) are on the Phase-1 critical path.
- **Every kernel (§4.3, §6) becomes a task** with a unit-test bead + a parity-gate bead + (where perf-relevant) a bench bead as dependencies — the gauntlet's "every remediation bead has test+bench+doc dependencies" rule.
- **Each alien family AF-1..AF-5 (§9.7) becomes a spike bead** carrying its recommendation contract (EV, tier, proof obligation, fallback) in the body, gated behind its deterministic fallback.
- Polish in plan-space **4–5 passes** (`/alien-graveyard` Step 8 refinement prompt; do not oversimplify, never lose features), then validate with `bv --robot-insights | jq '.Cycles'` (must be empty) before implementation starts.

> Per the user's directive, the beads conversion happens **after** this plan is dramatically improved — which is the state of this document now. The beads pass is the immediate next action.

---

*End of plan. This is a living document: resolve `[OPEN]`/OQ items as the source is read, append every accepted divergence to `DISCREPANCIES.md` and every rejected lever to `NEGATIVE_EVIDENCE.md`, and re-state the parity receipt on every perf commit. The optimization catalog (§6.6–§6.13) and the alien-artifact families (§9.7) are the levers; the three-pillar gauntlet (§8.5) is the conscience that keeps every claim honest.*
