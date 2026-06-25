# PROPOSED_ARCHITECTURE.md — the franken_ocr Rust design

**Bead:** `bd-322.25` (the second of the four `/porting-to-rust` documents). This is the **design half** of spec-first porting: it maps THE SPEC ([`truth-pack/EXISTING_UNLIMITED_OCR_STRUCTURE.md`](truth-pack/EXISTING_UNLIMITED_OCR_STRUCTURE.md), the `[SPEC-NNN]` clauses) onto the concrete Rust crate layout of [`COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`](../COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md) (the `PLAN_TO_PORT`). **We implement FROM the spec via this design — never line-translate the Python.**

**Provenance.** Every structural claim about the model traces to the pinned source @ HF commit `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5` / GitHub `7e98affeacba24e95562fbaa234ddb89b856874a` (see [`truth-pack/PINNED_SOURCES.md`](truth-pack/PINNED_SOURCES.md), file SHA-256s in [`truth-pack/SOURCE_HASHES.md`](truth-pack/SOURCE_HASHES.md)). The token/shape/buffer numbers (273 tokens/1024-view, `m_max=32896`, KV 5120 B/token/layer, 2710 tensors, 2244 quantizable linears) are line-backed in [`truth-pack/CENSUS.md`](truth-pack/CENSUS.md). Every `[OPEN]`/`OQ` answer is in [`truth-pack/OQ_INDEX.md`](truth-pack/OQ_INDEX.md).

**Status.** LIVING DOCUMENT. Seeded at Phase −1 from the truth pack; updated as design decisions land. The running surface/feature scoreboard is its sibling, [`FEATURE_PARITY.md`](FEATURE_PARITY.md).

> **How this maps to the four `/porting-to-rust` docs (plan §8.6):**
> `PLAN_TO_PORT` = `COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md` · THE SPEC = `EXISTING_UNLIMITED_OCR_STRUCTURE.md` · **THE DESIGN = this file** · the scoreboard = `FEATURE_PARITY.md`.

---

## Table of contents

1. [Design principles (the load-bearing constraints)](#1-design-principles)
2. [Crate shape — single crate, two binaries (plan §4.1)](#2-crate-shape)
3. [The runtime ownership model — OcrEngine + asupersync (plan §3.3)](#3-runtime-ownership)
4. [The data currency — Mat / slices over ft-kernel-cpu (plan §3.1–§3.2)](#4-data-currency)
5. [The frankentorch facade — op → kernel boundary (plan §4.3)](#5-frankentorch-facade)
6. [Module-by-module: which SPEC clause each file implements](#6-module-by-module)
7. [The `.focrq` on-disk format (plan §5)](#7-focrq-format)
8. [Pipeline stages (plan §4.2)](#8-pipeline-stages)
9. [Kernel strategy summary (plan §6)](#9-kernel-strategy)
10. [Cross-cutting seams (robot / conformance / storage / error)](#10-cross-cutting-seams)
11. [SPEC-clause → module index (the traceability matrix)](#11-spec-to-module-index)

---

## 1. Design principles

These are the non-negotiable invariants the whole design is built to satisfy (from AGENTS.md "Doctrine" + plan §1.1 goals G1–G7). Every module below is shaped by them.

| # | Principle | Where it bites the design |
|---|-----------|---------------------------|
| **P1** | **Correctness outranks speed (G1 > G2).** | The f32 reference-parity forward (Phase 1) is the design's spine; quant/SIMD are *additive layers* behind kill-switches, never a different code path. |
| **P2** | **Generality-tax wedge: one fixed model, compile-time shapes.** | Shapes (hidden 1280, 10 heads, head_dim 128, 64 experts, top-6, moe-inter 896, window 128, vocab 129280) are `const`s, not runtime params. We reach **past** ft-api's stateful session/tape straight to `ft-kernel-cpu`'s free functions over `&[f32]`. |
| **P3** | **Pure-Rust single binary, no FFI at runtime (G3).** | No Python, no `ort`, no C at inference. `mimalloc`/NUMA/`pdfium` are opt-in features that *break* the "no FFI" claim and are never default. |
| **P4** | **Memory-safe: `#![forbid(unsafe_code)]` at every crate root (G4).** | `unsafe` lives only inside named SIMD islands (`native_engine/nn` dispatch), each with a bit-identical scalar fallback that cross-compiles to every target. |
| **P5** | **Sync, blocking, embeddable public API (G6).** | `OcrEngine::recognize(...)` is sync; the asupersync `Runtime` is an owned implementation detail below `main`. No global state leaks. |
| **P6** | **No nested rayon under a lock; no nested asupersync runtime.** | Single `OcrModel` behind a cache; **sequential** outer page loop; each forward fans out across cores via the kernel's own rayon pool, exactly one live forward at a time. |
| **P7** | **Honest, measured everything (G7).** | Every accepted numeric divergence → `docs/DISCREPANCIES.md`; every rejected lever → `docs/NEGATIVE_EVIDENCE.md`; the scoreboard ([`FEATURE_PARITY.md`](FEATURE_PARITY.md)) accounts every surface as present/partial/missing/excluded — partial never rounds up. |

---

## 2. Crate shape

`franken_ocr` is **one crate, not a workspace** (the `franken_whisper` template, verbatim — plan §4.1). Two `[[bin]]` targets (`focr` short + `franken_ocr` long) both point at `src/main.rs`; the explicit `[[bin]]` declarations disable the implicit package-named bin. Sibling crates are path deps:

- `ft-kernel-cpu`, `ft-core`, `ft-serialize` (frankentorch) — consumed at the **kernel** level (plan §3.2).
- `asupersync = {">=0.3.5,<0.4", default-features=false}` — orchestration/cancellation/IO only (plan §3.3).
- `fsqlite` (frankensqlite) — durable run state; **NEVER `rusqlite`**.
- `image` / `fast_image_resize`, `clap`, `serde_json` — front end + CLI.

```
franken_ocr/                          (crate: franken_ocr)
├── Cargo.toml                        # two [[bin]]; [lints.rust] unsafe_code="deny"; path deps; profiles (§6.13)
├── rust-toolchain.toml               # nightly (stdarch i8mm/dotprod, portable_simd)
├── src/
│   ├── main.rs                       # SYNC fn main(): clap → ShutdownController → run(cli) → exit code
│   ├── lib.rs                        # #![forbid(unsafe_code)]; re-exports OcrEngine, OcrRequest/Result, Error
│   ├── cli.rs                        # clap-derive Cli/Command/Args, to_request() validation, robot summary
│   ├── orchestrator.rs               # OcrEngine (owns the Runtime), Pipeline/Stage/Config/Builder, PipelineCx,
│   │                                 #   CancellationToken, FinalizerRegistry, run_stage_with_budget
│   ├── robot.rs                      # versioned NDJSON events + ROBOT_SCHEMA_VERSION + robot schema/health/backends
│   ├── conformance.rs                # tolerance structs, L0–L5 parity/invariant validator traits, rollout stages
│   ├── storage.rs                    # RunStore on fsqlite, _meta versioned schema, forward migrations
│   ├── sync.rs                       # JSONL export/import (locked, atomic, one-way audit)
│   ├── error.rs                      # FocrError/FocrResult + stable exit-code mapping
│   ├── preprocess/                   # IMAGE INGEST FRONT END (frankentorch gap — built fresh)
│   │   ├── mod.rs · resize.rs · normalize.rs · pad.rs · tile.rs
│   ├── tokenizer/                    # pure-Rust byte-level BPE over tokenizer.json (no SentencePiece)
│   │   ├── mod.rs · special.rs
│   └── native_engine/                # THE MODEL PACKAGE (plain Rust over Mat/slices)
│       ├── mod.rs · weights.rs · tensor.rs · nn.rs
│       ├── vision_sam.rs · vision_clip.rs · vision_bridge.rs · connector.rs
│       ├── decoder.rs · rswa.rs · moe.rs · decode.rs · postprocess.rs
├── tests/ · tests/fixtures/ · benches/ · scripts/ · docs/ · .github/workflows/dist.yml
```

We take whisper's **scaffold** (orchestrator/CLI/robot/conformance/storage/native_engine structure) and **not** its domain code (audio/mel/dtw). The scaffold is delivered by epic **bd-223** (Phase 0); module skeleton by **bd-223.1**.

---

## 3. Runtime ownership

The asupersync integration is the `franken_whisper` layered pattern, copied verbatim (plan §3.3, delivered by **bd-223.2**):

- **`fn main()` is SYNCHRONOUS** (no `#[asupersync::main]`). It parses clap, installs a Ctrl+C `ShutdownController`, calls a sync `run(cli)`, and maps errors → exit codes (§10). The runtime lives **below** main, inside the engine, never spanning the process — satisfies P5/G6.
- **`OcrEngine` owns exactly one `Runtime`** (`RuntimeBuilder::new().worker_threads(2).blocking_threads(1,4).thread_name_prefix("focr").build()`). Public methods are **sync**: internally `runtime.handle().spawn(async {...})` → `runtime.block_on(handle)`. Callers see a blocking API.
- **CPU-bound stages run via `spawn_blocking`** wrapped in `asupersync::time::timeout(wall_now(), budget, …)` for per-stage budgets (`FOCR_STAGE_BUDGET_<STAGE>_MS`).
- **Intra-op math parallelism is the frankentorch kernel's OWN rayon pool**, pinned to physical (P-)cores — NOT asupersync tasks. Blocking concurrency is constrained to **exactly one live forward** (the sequential page loop, P6/§6.5): "streaming per-page" = streaming the *output* of a sequentially-processed page, never concurrent forwards.
- **Streaming** per-page results to the robot/NDJSON consumer uses `std::sync::mpsc::sync_channel` + `std::thread` (bounded, backpressured); main loops on `recv_timeout(~40ms)` until the worker finishes (**bd-wp8.3**).
- **Cancellation** is a `Copy` `CancellationToken` (deadline + `ShutdownController`) threaded via `checkpoint()` into the per-page and per-decode-step loops, so Ctrl+C/timeout aborts at the next boundary. The token goes *into* the `spawn_blocking` closure (cooperative cancellation).
- **The hard rule:** the engine owns exactly one runtime; never nest a second asupersync runtime in a task; never nest rayon under a held lock. The `many_pages_without_deadlock` watchdog (pages ≫ pool — **bd-2ub2 / bd-wp8.3.1**) hangs on regression.

---

## 4. Data currency

The forward is a hand-written model over a **`Mat`/slice currency**, not a tensor-graph (plan §3.1):

- **`native_engine/tensor.rs`** defines `Mat { rows, cols, data: Vec<f32> }` (the f32 activation rail) plus the quantized-weight structs (`QInt8PerChan { qweight: Vec<i8>, scales: Vec<f32>, shape }`, `QInt4PerGroup { …, group_size, tier }`) and the BF16-verbatim high-precision tensor type. This is the only currency that crosses module boundaries.
- We **skip autograd entirely** (inference only) — no `FrankenTorchSession`, no `NodeId` tape, no `requires_grad`. We call `ft-kernel-cpu`'s ~465 free pure functions over `&[f32]` directly.
- **Buffer reuse (the tape-truncation analog, §6.12):** every per-forward buffer — the R-SWA reference block + 128-ring sized for worst-case `m`, attention scratch, MoE token-group buffers, the f32 activation rails — is preallocated once and reused across pages. Zero per-token / per-page allocation in the hot loop.
- **High-precision weights are BF16 on disk** (the checkpoint is bf16, 1-8-7) and widened BF16→f32 at load. **BF16, never F16** — narrowing to f16 is lossy and is only ever a measured `DISCREPANCIES.md` divergence, never the silent default (plan §5.2, §6.12).

---

## 5. frankentorch facade

`native_engine/nn.rs` is the **single boundary** between our model code and frankentorch. Every kernel call goes through it; nothing else in `native_engine/*` touches frankentorch directly. This is where the **op → kernel map (plan §4.3, the §4.3 table)** is realized. Three categories:

### 5.1 REUSE as-is (frankentorch ships it)
| Op | frankentorch function | Used by |
|----|----------------------|---------|
| int8 dynamic-quant linear (the crown asset) | `linear_int8_dynamic_f32` + `quantize_per_output_channel_i8` | all decoder GEMMs, lm_head, projector(opt) |
| f32 linear | `linear_tensor_f32` | projector, vision (v1) |
| Conv2d (im2col + GEMM) | `conv2d_forward_f32` / `conv2d_im2col_f32` | SAM/CLIP patch-embed, SAM neck (5 fixed shapes) |
| SDPA attention | `sdpa_forward_f32` (+masked, +gqa) | SAM-global, CLIP, R-SWA prefill basis |
| RMSNorm | `rms_norm_forward_f32` | decoder ([SPEC-071]) |
| LayerNorm | `layer_norm_forward_f32` | vision (LayerNorm2d = thin wrapper) |
| GELU / SiLU | gelu/silu tensor ops | SAM MLP / LLM MLP |
| safetensors BF16 load | `load_safetensors_from_bytes` (F32/F16/BF16) | the converter (§7) |

### 5.2 BUILD (model-specific, frankentorch gap)
| Op | Plan/SPEC ref | Built in | Delivering bead |
|----|---------------|----------|-----------------|
| Windowed self-attention (window 14, [SPEC-043/045], OQ-15) | §4.3 | `vision_sam.rs` | bd-1gv.6 |
| quick_gelu `x·σ(1.702x)` ([SPEC-049]) | §4.3 | `nn.rs` | bd-1gv.9 |
| f32 bicubic pos-embed interpolation ([SPEC-042/048]) | §4.3 | `vision_sam.rs`/`vision_clip.rs` | bd-1gv.4 |
| RoPE (Llama variant, theta 10000, [SPEC-078], OQ-5) | §4.3 | `decoder.rs` | bd-1gv.16 |
| **R-SWA decode attention (ring + ref block, the centerpiece)** ([SPEC-090..096]) | §4.3, §6.4, §6.8 | `rswa.rs` | bd-1gv.17 |
| MoE router top-6 greedy + norm_topk_prob ([SPEC-077]) | §4.3 | `moe.rs` | bd-1gv.18 |
| Grouped expert SiLU-gated MLP ([SPEC-076]) | §4.3, §6.7 | `moe.rs` | bd-1gv.19 |
| Token embedding lookup (bf16-preserving index_select) ([SPEC-070]) | §4.3 | `decoder.rs` | bd-1gv.14 |
| masked-scatter vision fusion ([SPEC-064/066]) | §4.3 | `connector.rs` | bd-1gv.11 |
| Image decode/resize/normalize/pad/tile ([SPEC-020..033]) | §4.3 | `preprocess/` | bd-1gv.2, bd-1gv.3 |
| BPE tokenizer over tokenizer.json ([SPEC-035], OQ-16) | §4.3 | `tokenizer/` | bd-1gv.1 |
| Sampler + no_repeat_ngram(35) ([SPEC-100..103]) | §4.3 | `decode.rs` | bd-1gv.22 |

### 5.3 BUILD (the perf wedge — Phase 3+, behind kill-switches)
The tiled register-blocked int8 GEMM that frankentorch has only *named* (verified absent at frankentorch `a84674c8`): **SMMLA/i8mm prefill GEMM** (bd-2mo.4), **AVX-512-VNNI** (bd-2mo.6), AVX-VNNI/AVX2/AMX tiers, SDOT decode GEMV (bd-2mo.5), all behind a runtime ISA dispatch (`OnceLock<IsaTier>`, bd-2mo.1) with a bit-identical scalar floor (bd-2mo.10). `nn.rs` exposes one `int8_gemm`/`int8_gemv` entrypoint; the dispatch and the +128 U8S8 correction live inside the SIMD island.

> **The facade is what keeps P1 honest:** the f32 path (5.1+5.2) is the parity spine; 5.3 is an additive, bit-identical-where-integer layer. A perf kernel that drifts OCR output is reverted at the facade, the rest of the model untouched.

---

## 6. Module-by-module

Each `native_engine/*.rs` module implements a contiguous block of THE SPEC. The full clause→module index is §11; this section gives the design intent per module.

### 6.1 `tokenizer/` — [SPEC-019, SPEC-035], OQ-16
Pure-Rust byte-level BPE over `tokenizer.json` (9.98 MB, base vocab 128000 + 830 added tokens). `mod.rs` = encode/decode (pre-tokenizer `Sequence`, byte-fallback, merges); `special.rs` = the special-token table (bos 0, eos 1, pad, `<image>` 128815, `<|ref|>`/`<|det|>`/`<|grounding|>`/`<|User|>`/`<|Assistant|>`). **Token-id-exact vs `LlamaTokenizerFast` is an L0/L4 prerequisite** — a mismatch corrupts every downstream gate (test bead bd-1gv.1.1). Spec extracted by bd-1gv.13 (decoder/output) and the existing THE SPEC.

### 6.2 `preprocess/` — [SPEC-018, SPEC-020..033]
The image front end (a frankentorch gap, built fresh). `mod.rs` decodes (EXIF-transpose, RGB [SPEC-020]); `normalize.rs` = ToTensor→Normalize(0.5,0.5) ⇒ [-1,1] [SPEC-021]; `pad.rs` = `ImageOps.pad` gray (127,127,127) [SPEC-022]; `resize.rs` = bilinear/bicubic aspect-preserving; `tile.rs` = **Base** (1024, crop_mode=false) vs **Gundam** (`dynamic_preprocess`/`find_closest_aspect_ratio`, min_num 2/max_num 32, OQ-7) [SPEC-023..029], the image-token id-stream layout [SPEC-028] (273 slots/1024-view per CENSUS (c)), BOS prepend + masks [SPEC-030], image-tensor packing `images=[(crop, ori)]` [SPEC-031]. **L0 parity = exact** (bd-1gv.3.1, bd-re8.4).

### 6.3 `vision_sam.rs` — [SPEC-040..046], OQ-15
SAM-ViT-B: patch-embed Conv2d k16s16 → 64×64=4096 tokens, width 768 [SPEC-041] (bd-1gv.5); learned pos_embed (1,64,64,768) bicubic-interpolated [SPEC-042] (bd-1gv.4); 12 blocks with **window attention (window=14) except global at [2,5,8,11]** [SPEC-043] (bd-1gv.6); decomposed relative-position bias [SPEC-044]; window partition/unpartition with padding [SPEC-045]; neck (1×1 → LN2d → 3×3p1 → LN2d → two stride-2) → B,1024,16,16 [SPEC-046], the **16× compression** to 256 tokens (bd-1gv.7). SAM MLPBlock uses **GELU** (distinct from CLIP quick_gelu / LLM SiLU — bd-1gv.28). L1/L2 parity bd-1gv.8.

### 6.4 `vision_clip.rs` — [SPEC-047..050]
CLIP-L/14: 24 layers, width 1024, 16 heads, patch 14 [SPEC-047]. **Fused tower:** CLIP embeddings take the SAM `x3` output as `patch_embeds` [SPEC-048/050]; class token prepended; abs pos via `get_abs_pos` (bicubic, no-op at 1024 — UNRESOLVED-3). Each block: `h = x + attn(LN1(x))`; `out = h + mlp(LN2(h))` with `mlp = fc2(quick_gelu(fc1(x)))`, full SDPA no causal mask, qkv/out bias=True [SPEC-049]. Delivered by bd-1gv.9, parity bd-1gv.9.1.

### 6.5 `vision_bridge.rs` — [SPEC-051/052], OQ-6
Hybrid feature = `concat(CLIP[:,1:], SAM_flat)` along channels → 2048 dims [SPEC-051], then the single linear projector 2048→1280 [SPEC-052/SPEC-016]. The exact concat order (OQ-6, RESOLVED in `oq/vision.md`) is implemented from the spec, not guessed. Delivered by bd-1gv.10, parity bd-1gv.10.1.

### 6.6 `connector.rs` — [SPEC-060..066]
Learned `image_newline`/`view_seperator` params (randn·1/√1280) [SPEC-060]; the vision-branch trigger [SPEC-061]; the **CROP branch** `[local, global, view_seperator]` spatial arrangement with per-row newline column [SPEC-062]; the **NO-CROP branch** [SPEC-063]; **masked_scatter** into the text embeds [SPEC-064] (`inputs_embeds = embed_tokens(input_ids)` then scatter, [SPEC-065]). **[SPEC-066 ORDERING INVARIANT]:** the token layout [SPEC-028] must produce exactly `(local)(global)(1 sep)` matching the feature concat order so masked_scatter aligns — the Rust port replicates this exact interleave. Delivered by bd-1gv.11, parity bd-1gv.11.1.

### 6.7 `decoder.rs` — [SPEC-070..081]
The 12-layer loop driver: `embed_tokens` ([SPEC-070], bd-1gv.14) → per layer `h = x + self_attn(input_layernorm(x))`; `out = h + mlp(post_attention_layernorm(h))` [SPEC-072] → final RMSNorm [SPEC-070] → `lm_head` GEMV 1280→129280 f32 [SPEC-081] (bd-1gv.21). RMSNorm (f32 variance, eps 1e-6) [SPEC-071] (bd-1gv.15). RoPE = **Llama** `_llama_apply_rotary_pos_emb` (NOT the DeepseekV2 interleaved MLA variant), theta 10000, head_dim 128 [SPEC-078], OQ-5 (bd-1gv.16). Position IDs [SPEC-079], 4D mask handling [SPEC-080]. Attention class selection: `use_mla=false` ⇒ `SlidingWindowLlamaAttention` for **all 12 layers** (OQ-2) [SPEC-073]; dense-vs-MoE per layer (layer 0 dense, 1..11 MoE) [SPEC-074]. Orchestration + full L2–L5 ladder by bd-1gv.24 / bd-1gv.24.1.

### 6.8 `rswa.rs` — [SPEC-090..096], OQ-1/2/3/13 (the centerpiece)
R-SWA = `SlidingWindowLlamaAttention`, the load-bearing custom kernel. 10/10 heads, head_dim 128, scale 1/√128, no QKV bias [SPEC-090]. **Three regimes** [SPEC-091]: true prefill (full causal, record `prefill_len`); warmup decode (cat-append until `cur_len ≥ prefill_len+W`); steady-state ring (overwrite `slot = prefill_len + ring_pos`, `ring_pos=(ring_pos+1)%128`, attention over the full cache with no causal mask). **[SPEC-095 PORT INVARIANT]:** RoPE uses the TRUE absolute `position_ids`, NOT the ring slot — physical slot decoupled from logical position. Design (CENSUS (d)): preallocate per layer one fixed buffer of `prefill_len + 128` (worst case `m_max = 32896`); reference block (BOS+visual+prompt, OQ-1 = the entire prefill) is **read-only, never evicted**; only the trailing 128 slides. **Compute scales with page count** (`O(m+128)`/step), memory is flat (§6.4). **Online (FlashAttention-style) softmax** over the reference block (§6.8, bd-1gv.17.1 / bd-2mo.14). Delivered by bd-1gv.17, parity bd-1gv.17.2.

### 6.9 `moe.rs` — [SPEC-074..077], §6.7
MoEGate: `logits = linear(hidden.f32, gate.f32)` → softmax(f32) → top-6 greedy → since `top_k>1 && norm_topk_prob=false`, `topk_weight *= routed_scaling_factor(1.0)` (**NO renormalization** [SPEC-013/077]) (bd-1gv.18). `moe_infer`: route each token to top-6 experts, each a 1280↔896 SiLU-gated MLP, weight + sum; plus 2 fused shared experts (one `DeepseekV2MLP`, intermediate 1792) [SPEC-076] (bd-1gv.19). Dense layer-0 SwiGLU intermediate 6848 [SPEC-075] (bd-1gv.20). Router **never quantized** (f32, gate-drift cliff). Prefill token-grouping (counting-sort → dense per-expert GEMMs, §6.7, bd-2mo.12); decode cache-contiguous per-(layer,expert) packing (bd-2mo.13).

### 6.10 `decode.rs` — [SPEC-100..103]
The frozen `DecodeParams`/`DecodeOutput` contract + AR loop. Greedy by default (`do_sample = temperature>0`, default 0.0 ⇒ argmax) [SPEC-100]; EOS=1, max_length 32768, use_cache [SPEC-101]; **`no_repeat_ngram_size=35`** with `ngram_window` **128 single / 1024 multi** [SPEC-102/103], OQ-18 — the custom `SlidingWindowNoRepeatNgramProcessor` (these are first-class generation semantics, not just CLI flags). lm_head→argmax-only fused under greedy (§6.10/bd-2mo.24). Delivered by bd-1gv.22, parity bd-1gv.22.1.

### 6.11 `postprocess.rs` — [SPEC-110..119]
Decode+strip EOS `<｜end▁of▁sentence｜>` [SPEC-110]; ref/det regex extraction [SPEC-111]; coordinate parse [SPEC-112]; **bbox /999 rescale** to pixels [SPEC-113]; image-label crops → `![](images/{idx}.jpg)` [SPEC-114]; other-label cleanup + `\coloneqq`→`:=` [SPEC-115]; multi-page `<PAGE>` split/rejoin [SPEC-118]. (Box-overlay drawing [SPEC-116] and the geometry/`line_type` special case [SPEC-117] are excluded from v1 — visualization-only; see [`FEATURE_PARITY.md`](FEATURE_PARITY.md).) Delivered by bd-1gv.23, parity bd-1gv.23.1.

### 6.12 `weights.rs` + `mod.rs` — the model package
`weights.rs` = the `.focrq` reader (dependency-free byte-range index into one mmap/blob, §7) + safetensors fallback + the **WeightsManifest census** on load (catch wrong/stale weights at load, not as garbage output — bd-1es.3). `mod.rs` = `OcrModel` cached behind `Arc` + `Weak` global cache, `resolve_model`, header sniff (`native_model_available`, bd-223.7).

---

## 7. `.focrq` format

The custom on-disk quantized container (plan §5.2; spec bead bd-1es.1; writer bd-1es.2; reader bd-1es.3). A safetensors-like, length-prefixed, self-describing blob — the `franken_whisper` ggml/safetensors parser pattern (read whole file into one `Vec<u8>`, validate magic, read header, index by byte range):

```
magic:            b"FOCRQ\0"                       # loud rejection on mismatch
format_version:   u32                              # bumped on any layout change; loader refuses version > binary's
arch_target:      enum { Generic, Aarch64Smmla, X86Vnni, X86Amx }   # which offline pre-packing (§5.4, bd-2mo.3)
source_sha256:    [u8;32]                          # provenance: sha256 of the source safetensors
license_notice:   utf8                             # "Copyright (c) 2026 Baidu — MIT License ..." (MUST be present)
model_config:     json                             # frozen copy of the relevant config.json fields
tensor_directory: [ { name, dtype, shape, packing, byte_offset, byte_len,
                      scales_offset, scales_len, group_size?, tier? } ]
payload:          <raw bytes>
```

- **dtype ∈ {F32, F16, BF16, QInt8PerChan, QInt4PerGroup}.** Quantized weights carry scales **inline**: int8 = per-output-channel `Vec<f32>` (`scale = max|w_row|/127`, zero-point 0); int4 = per-group (16–32) scales + tier.
- **High-precision tensors stored BF16 verbatim** (vision tower, projector, embed_tokens, MoE router gate, all norms, and lm_head when unquantized), dequantized BF16→f32 at load. **BF16, NOT F16** (P4/§5.2): f16 narrowing is lossy → a measured `DISCREPANCIES.md` entry only, never default.
- **The quant recipe is fixed (AGENTS.md doctrine #2):** quantize only the decoder FFN/expert GEMMs (the NVFP4/GGUF-validated set); int8 on attention `q/k/v/o` and `lm_head` go *beyond* it, behind `FOCR_INT8_ATTN`/`FOCR_INT8_LMHEAD` kill-switches gated on measured CER (OQ-14). Tensor remap HF dotted paths → internal layout per plan §5.3 (bd-1es.4).
- **Determinism + round-trip (§5.4, bd-1es.12):** BF16/F32 stored verbatim → byte-identical round-trip (bd-1es.2.1); quant is a pure function of input bytes (no RNG, data-free PTQ). **i32-overflow is a proof obligation** — worst case is dense layer-0 `down_proj` at **K=6848** (U8S8 ≤ 221.7M, fits i32 with 9× headroom), proven by a worst-case-K unit test on every arch (bd-1es.7 / bd-2mo.11). **Do NOT inherit frankensearch's `k≤1536` bound.**
- One canonical artifact, many packings: `focr convert --arch {…}` emits per-arch blobs from one source; CI verifies all dequant to the same logical weights (bd-2mo.3.1). The int4 loader reads `QInt4PerGroup` records (bd-3gaa.1).

---

## 8. Pipeline stages

The `PipelineStage` enum drives the forward (plan §4.2); each stage has an env-overridable budget (`FOCR_STAGE_BUDGET_<STAGE>_MS`) and a `checkpoint()` cancellation boundary. Assembled into `OcrEngine::recognize` / `focr ocr` by bd-1gv.27 (single-image) and bd-1gv.25 (multi-page):

```
Decode/Load image → Preprocess (resize/pad/normalize/tile) → Tokenize prompt
  → Vision encode (SAM → bridge/compress → CLIP → concat → project)
  → Connector (insert structural tokens, masked-scatter into the embed stream)
  → Prefill (build reference KV: visual + prompt) → Decode loop (R-SWA + MoE, AR)
  → Postprocess (EOS strip, ngram, tag parse, bbox /999, markdown) → Emit
```

**Two cost centers, optimized separately (§6, §9):** *Vision encode* is a fixed per-page cost (~256 image-feature tokens before structural tokens, dominated by SAM/CLIP attention+conv GEMMs over ~4096 patches → compute-bound). *Decode* is a per-output-token cost (memory-bound expert GEMV + 129K-vocab logits). The profiling corpus spans {sparse page, dense page, 10-page, 40-page} because the dominant cost center moves with output length (§9.1).

**Multi-page is cross-page DEPENDENT (OQ-13, [SPEC-118], bd-1gv.25/bd-1gv.26):** `infer_multi` concatenates all pages into ONE prefill/generate call, so page N attends to pages 1..N−1 (the reference block spans all pages). The design must **not** treat multi-page as a sum of independent single-page parses — the metamorphic gate (bd-re8.10) encodes the *opposite* property.

---

## 9. Kernel strategy

(Plan §6, the make-it-FLY epic **bd-2mo**.) The load-bearing constraints:

- **The frankensearch lesson (doctrine #3, §6.2):** hand-wide-SIMD over scalar inner loops measured **5× SLOWER** than LLVM autovectorization. So: (a) write norm/softmax/dequant/elementwise *glue* as tight scalar loops (f64 accumulation where precision matters) and let LLVM autovectorize; (b) write tight int8 GEMM/GEMV micro-kernels with **native matmul intrinsics** only. The one measured exception is vectorized transcendentals (poly `exp` for softmax/SiLU/quick_gelu — LLVM won't autovectorize `libm`), gated behind `FOCR_VEC_EXP` + a parity gate (§6.11, bd-2mo.20).
- **The edge (doctrine #4, §3.2):** the gap to ONNX/MLAS is *kernels below peak*, NOT framework overhead. The win is the combination franken_ocr has by construction — fused, tape-free, zero-per-op-alloc single-model forward, every op at peak (register-blocked SMMLA/VNNI + int8 attention where accuracy allows + vectorized norms), plus the int4 bandwidth win on the expert bulk. **Un-blocked SMMLA is a TRAP** (load-bound, slower than SDOT); the micro-kernel must reach compute:load ≥ 2:1 via register/cache blocking + offline pre-packing.
- **Per-arch dispatch catalog (§6.6, bd-2mo.1..10):** one `int8_gemm`/`int8_gemv` entrypoint, dispatched once (`OnceLock<IsaTier>`), bit-identical across paths (i32 accumulation is exact). Order: **x86** AMX > AVX512-VNNI > AVX-VNNI > AVX2 > scalar; **ARM** SMMLA > SDOT > scalar. `focr robot backends` reflects the tier (bd-2mo.2). ⚠ AVX2 `vpmaddubsw` saturates at i16 → carries its own overflow proof (split-accumulate, bd-2mo.9.1) or a ledgered divergence.
- **Mixed int4/int8 (§6.3, Phase 4 / bd-3gaa):** decode is bandwidth-bound → int4 group-quant (g=16–32) on the expert bulk halves bytes/token (~2× decode); accuracy-sensitive tensors (`v_proj`, expert `down_proj`, `lm_head`) stay one tier higher. No CPU int4 instruction → unpack int4→int8 in-register, feed the same MAC. The per-tensor int4/int8/bf16 split is chosen by the rate-distortion water-filling allocator (AF-1, §9.7, bd-ksps / bd-1xfa.1), bounded by the tail-risk CER gate (AF-2, bd-3upw / bd-1xfa.2).
- **MoE dispatch (§6.7):** prefill = counting-sort tokens by expert → one dense GEMM per active expert (bd-2mo.12); decode = cache-contiguous per-(layer,expert) packing + prefetch (bd-2mo.13). **R-SWA (§6.8):** online-softmax over the reference block, int8 attention behind `FOCR_INT8_ATTN` + CVaR tail gate (bd-2mo.15).
- **Many-core (§6.9 / AF-5, bd-2mo.21):** prefill scales with cores, decode does NOT (bandwidth-bound) — cap each pool at its **USL peak**, not `num_cpus`. NUMA: replicate the read-only weight blob per node (`FOCR_NUMA`). Apple Silicon: pin GEMM pool to P-cores. **Concurrency discipline (§6.5, P6):** single model, sequential outer loop, internal fan-out, one live forward.
- **Build-time (§6.13, bd-2mo.23):** release profile `lto=fat`+`codegen-units=1`+`panic=abort`; separate `release-perf` for profiling; PGO over the golden corpus; BOLT on the hot kernels; `target-feature` not `target-cpu=native` (runtime dispatch ⇒ one portable binary/arch).

---

## 10. Cross-cutting seams

- **`error.rs` — stable exit codes (plan §7.4):** `0` success · `1` generic · `2` usage/CLI · `3` model not found · `4` input decode · `5` budget/timeout · `6` cancelled · `7` format/version mismatch. `FocrError`/`FocrResult`; `robot run_error` carries the same code (bd-223.5).
- **`robot.rs` — agent ergonomics (plan §7.3, G5):** NDJSON event stream (one JSON/line, each carrying `schema_version`/`ROBOT_SCHEMA_VERSION`); events `run_start`, `stage`, `page`, `run_complete`, `run_error`; `robot schema` self-describes the contract; `robot health` / `robot backends` diagnostics. Deterministic under fixed sampling. Skeleton bd-223.3; frozen schema + contract test bd-wp8.2 / bd-zc1o / bd-wp8.2.1.
- **`storage.rs` — durable run state:** `RunStore` on **fsqlite** (never rusqlite), `_meta` versioned schema, forward migrations (bd-223.4). `sync.rs` = locked atomic one-way JSONL audit export/import (bd-wp8.11).
- **`conformance.rs` — the parity ladder (plan §8.2, epic bd-re8):** tolerance structs + L0–L5 validator traits + rollout stages (seed bd-223.9). L0 preprocessing exact (bd-re8.4); L1 per-op / L2 per-layer cosine ≈ 1.0 (bd-re8.5); L3 logits within *measured* tolerance + argmax match / L4 token exact (bd-re8.6); L5 end-to-end CER/TEDS/Formula-CDM within budget (bd-re8.7). **The oracle's own nondeterminism floor is established FIRST** (bd-re8.2) — tolerances derive from measured oracle variance, NOT the imported frankensearch `0.055`.
- **`cli.rs` — the surface (plan §7.2):** subcommands `ocr`, `convert`, `robot {run,schema,health,backends}`, `runs`, `sync`, `doctor`. **v1 is IMAGE-ONLY** — `pdf` does NOT appear (plan §7.7): pure-Rust MuPDF-parity raster is unscoped and any pixel mismatch blows the L0 gate. Finalized bd-wp8.1; `doctor` bd-wp8.4.

---

## 11. SPEC-to-module index

The traceability matrix the surface-parity pillar (bd-re8.13) and [`FEATURE_PARITY.md`](FEATURE_PARITY.md) consume. Every MUST/SHOULD clause of THE SPEC maps to a named module (≥0.95 MUST coverage to claim conformance, plan §8.6).

| SPEC clauses | Module(s) | Op-map (§4.3) | Phase | Bead |
|--------------|-----------|---------------|-------|------|
| SPEC-001..003 (architecture/weight prefixes) | `native_engine/mod.rs`, `weights.rs` | safetensors load | 1/2 | bd-1es.4, bd-1es.3 |
| SPEC-010..015 (decoder/MoE/sliding-window config) | `tensor.rs` consts, `decoder.rs` | — | 1 | bd-1gv.13 |
| SPEC-016..017 (projector/vision config) | `vision_bridge.rs`, `vision_*` | linear | 1 | bd-1gv.10 |
| SPEC-018 (normalize/processor) | `preprocess/normalize.rs` | preprocess | 1 | bd-1gv.2 |
| SPEC-019 (special token IDs) | `tokenizer/special.rs` | tokenizer | 1 | bd-1gv.1 |
| SPEC-020..033 (preprocess/tiling/packing) | `preprocess/{mod,pad,resize,tile}.rs` | image front end | 1 | bd-1gv.2, bd-1gv.3 |
| SPEC-034..036 (prompt formatting) | `tokenizer/`, `preprocess/tile.rs` | tokenizer | 1 | bd-1gv.1 |
| SPEC-040..046 (SAM tower) | `vision_sam.rs` | conv, windowed/global attn, bicubic | 1 | bd-1gv.5,6,7,28 |
| SPEC-047..050 (CLIP tower) | `vision_clip.rs` | SDPA, quick_gelu | 1 | bd-1gv.9 |
| SPEC-051..052 (concat + projector) | `vision_bridge.rs` | linear | 1 | bd-1gv.10 |
| SPEC-060..066 (connector + masked_scatter) | `connector.rs` | masked-scatter fusion | 1 | bd-1gv.11 |
| SPEC-070..072 (decoder stack/RMSNorm/layer) | `decoder.rs` | RMSNorm, embedding lookup | 1 | bd-1gv.14,15,24 |
| SPEC-073..074 (attn selection / dense-vs-MoE) | `decoder.rs`, `moe.rs` | dispatch | 1 | bd-1gv.18,19,20,24 |
| SPEC-075..077 (MLP / MoE / gate) | `moe.rs` | MoE router + grouped expert MLP | 1 | bd-1gv.18,19,20 |
| SPEC-078..080 (RoPE / position / mask) | `decoder.rs` | RoPE | 1 | bd-1gv.16 |
| SPEC-081 (lm_head + logits) | `decoder.rs` | int8/f32 GEMV | 1 | bd-1gv.21 |
| SPEC-090..096 (R-SWA ring buffer) | `rswa.rs` | **R-SWA decode attn (centerpiece)** | 1 | bd-1gv.17 |
| SPEC-100..103 (sampler / ngram) | `decode.rs` | sampler + no_repeat_ngram(35) | 1 | bd-1gv.22 |
| SPEC-110..115, 118..119 (postprocess) | `postprocess.rs` | — | 1 | bd-1gv.23 |
| SPEC-116..117 (box overlay / geometry) | — (excluded v1) | — | — | see FEATURE_PARITY |
| UNRESOLVED-1 (tokenizer.json / OQ-16) | `tokenizer/` | tokenizer | 1 | bd-1gv.1.1 |
| UNRESOLVED-2 (NVFP4 layout / OQ-14) | `weights.rs` (quant policy) | quant recipe | 2/4 | bd-1es.10/11 |
| UNRESOLVED-3 (CLIP get_abs_pos interp) | `vision_clip.rs` | f32 bicubic | 1 | bd-1gv.4 |

---

*End of design. This document is updated as design decisions land; the running surface/feature scoreboard it feeds is [`FEATURE_PARITY.md`](FEATURE_PARITY.md). Every claim here traces to THE SPEC's `[SPEC-NNN]` clauses (line-backed to the pinned source) and to a delivering bead. Per the doctrine: implement FROM the spec via this design, never line-translate the Python; resolve `[OPEN]`/OQ before the dependent kernel ships; ledger every accepted divergence and every rejected lever.*
