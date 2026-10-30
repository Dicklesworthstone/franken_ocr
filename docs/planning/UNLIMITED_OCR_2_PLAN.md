# Unlimited-OCR-2 in franken_ocr — Upgrade & Support Plan

> **Status:** planning (no code landed). **Created:** 2026-10-28. **Revision:** 3. Review rounds 1–4
> are integrated (spec fidelity, codebase grounding, strategy, consistency; see §15).
> **Companion to:** [`COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`](COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md). Its
> doctrine binds this plan.
> **Work graph:** this plan is mirrored 1:1 into beads under the epic "Unlimited-OCR-2: additive model
> + unlimited-horizon long-document parsing" (label `uocr2`). Once the beads exist they are the
> executable source of truth. This document holds the rationale and the spec.

---

## 0. How to read this plan

| Section | Contents |
|---|---|
| §1 | One-page summary |
| §2 | Verified ground truth about the new model. Every number was checked against the pinned source, config, weight headers or tech report on 2026-10-28. |
| §3 | Why the plan is shaped this way, from the user's side |
| §4 | Load-bearing decisions, each with its rationale |
| §5 | **Reference-semantics specification**, written so that nobody has to reopen the Python |
| §6 | franken_ocr architecture for the port, including every code touchpoint the review found |
| §7–§10 | Verification, performance, product surface, default policy |
| §11–§14 | Roadmap, task graph (mirrored into beads), risks, open questions |
| §15 | Review log |
| Appendices | Numbers, pins, code locations, related beads |

Terminology:

| Term | Meaning |
|---|---|
| **v1** | `baidu/Unlimited-OCR` @ HF `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`, the model franken_ocr runs today |
| **v2** | `baidu/Unlimited-OCR-2` @ HF `5632329a32e103d5f1add270681e0f5c1233c6e4` |
| **R-SWA** | v1's Reference Sliding Window Attention |
| **SR-SWA** | v2's Sparse Reference Sliding Window Attention ("DASH" in Baidu's code), with two stages, **Page Gate** and **Token Sieve**, named as in the report |
| **Document mode** | One `<PAGE>`-structured decode over many pages, windowed when long |
| **Window** | One chunk of the unlimited-horizon decoding scheme (§5.10) |
| **HF reference** | The transformers `infer()` path in the v2 repo |
| **SGLang reference** | The customized SGLang wheel shipped in the v2 repo |

---

## 1. Executive summary

**What v2 is.** v2 keeps v1's backbone exactly: the same vision tower, projector, MoE decoder
shapes, vocabulary and tokenizer model. The vision tower and projector weights are **byte-identical**
to v1 because they were frozen in training. Three things changed:

1. **Attention.** Every decoder layer's attention is now SR-SWA. Each query attends to a 128-token
   local window plus up to 12 router-selected 32-token blocks. Those blocks come from **its own
   page's image tokens**, or from all earlier blocks when its page has none yet. Per-step attention
   stays near 512 tokens, however long the document.
2. **Positions.** RoPE is two-axis: an intra-page token position plus a page-parity bit.
3. **Decode contract.** Prompts and outputs carry a `<PAGE>` token (id 128827) for every page, even
   a single page. A page-count guard makes the output contain exactly one `<PAGE>` per input page.

**What it buys users.**
- **Single pages:** essentially nothing in accuracy. OmniDocBench v1.6 goes from 93.92 to 94.02, and
  tables are slightly worse.
- **Multi-page documents:** a large gain. MPDocBench (≤ 30 pages) goes from 61.89 to 84.59, and
  cross-page text edit distance drops from 0.605 to 0.100.
- **Long documents:** a new capability. MarathonDocBench (100–300 pages) scores 77.06. v1 cannot run
  it: in franken_ocr, v1's prompt and output share 32,768 positions, so v1 stops at about 29 dense
  pages.
- **Books:** a path to arbitrarily long documents with bounded memory, via chunked "unlimited-horizon"
  decoding.
  - Baidu ships this only in GPU SGLang serving.
  - Making it work here also needs a streaming page source, because today's multi-page path holds
    every page raster in memory.

**The plan.**
1. Land v2 **additively**, as a new zoo model `unlimited-ocr-2`.
2. First, harden model identity and make dispatch **fail closed**: no v1 artifact can ever be read as
   v2, and no v2 artifact can ever run through v1 code.
3. Port SR-SWA, page2d RoPE and the v2 decode contract, gated rung by rung against an **f32 CPU HF
   oracle**. That is the precedent that gave v1 token-exact parity.
4. Certify a conservative int8 artifact.
5. Build the differentiator: **long-document parsing on a laptop CPU**, with:
   - bounded memory (streamed pages, windowed KV, block retirement);
   - page-by-page streaming output;
   - crash-safe, byte-identical resume.

When v2 is the resolved model, a PDF is parsed **as one document by default**. Every per-page cost is
the same inside a window as alone, so document mode adds cross-page structure and per-page output at
roughly per-page cost. Per-mode defaults move only on ground-truth evidence, with the incumbent
measured in the same invocation. The intended end state is one default model for every mode.

**What we are explicitly not doing:**
- replacing v1 in place;
- copying SGLang serving internals (radix cache, prefetch, Triton kernels);
- GPU paths;
- training;
- iOS/WASM v2 lanes in the first wave.

---

## 2. Verified ground truth about v2

### 2.1 Provenance and license

| Item | Value |
|---|---|
| HF repo / commit | `baidu/Unlimited-OCR-2` @ `5632329a32e103d5f1add270681e0f5c1233c6e4` (created 2026-10-26) |
| Weights | `model-00001-of-00002.safetensors` (4,998,430,312 B, sha256 `4c40b740f209fd7682040d4ebf19409208acde5c401ccd2dde426d0f41542f18`) and `model-00002-of-00002.safetensors` (1,689,875,264 B, sha256 `9490b82fa47f89e5110eb5577f8ab1b6694c12121905dbc086ddb10207a4ba0c`). `model.safetensors.index.json` gives `total_size` = 6,687,958,536. |
| Tensors | 2,819: 2,818 BF16 plus 1 I64 buffer |
| License | MIT. GitHub `baidu/Unlimited-OCR` `LICENSE` reads "MIT License, Copyright (c) 2026 Baidu", and the HF metadata says `license: mit`. **The HF v2 repo itself ships no LICENSE file.** Record that in provenance; the notice we redistribute is the GitHub text. |
| Reference stack (model card) | Python 3.10, `torch==2.13.0`, `transformers==4.52.1`, `tokenizers==0.21.4`, `huggingface-hub==0.36.0`, CUDA (tested on A800). **This differs from the v1 oracle pins** (torch 2.10.0, transformers 4.57.1). |
| Serving reference | `wheel/sglang-0.5.20.dev10+g8d62f5fb9c-py3-none-any.whl` (sha256 `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d`) |
| Tech report | `Unlimited-OCR-2.pdf` (sha256 `1eb5885f…9444`). The arXiv id in the card is v1's `2606.23050`. |
| Benchmark | `baidu/MarathonDocBench` (dataset, Apache-2.0): 100 documents of 100–300 pages (~18k pages, ~205k layout elements), zh/en, scored with the MPDocBench-Parse toolkit |

Source-file SHA-256 pins for the v2 commit are in Appendix B.

### 2.2 Architecture delta (v1 → v2)

| Component | v1 | v2 | Evidence |
|---|---|---|---|
| Decoder dims | 12 layers, hidden 1280, 10 MHA heads, head_dim 128, layer-0 dense (6848), layers 1–11 MoE 64 routed + 2 shared experts, top-6, moe_intermediate 896, vocab 129280, `use_mla=false` | **Identical** | config diff |
| Vision tower (SAM-B → conv compressor → CLIP-L/14), projector 2048→1280, `image_newline`, `view_seperator` | — | **Byte-identical weights.** Sampled tensors were compared byte-for-byte. Report §4.2 says DeepEncoder and aligner were frozen. `deepencoder.py` differs only in comments (the comment-stripped diff is empty). | range-read header and data compare |
| Decoder weights (attention, experts, router gate, embed_tokens, lm_head) | — | **Retrained.** Sampled 64 KiB slices differ in 22–98% of bf16 elements. | same |
| New tensors (per layer, ×12) | — | `self_attn.dash_router.{q_proj [128,1280], k_proj [128,1280], q_norm [128], k_norm [128], head_cls [128]}` and `self_attn.indexer.{q_proj [128,1280], k_proj [128,1280], q_norm [128], k_norm [128]}`, all BF16 (60 router + 48 indexer tensors) | header |
| New buffer | — | `model.vision_model.embeddings.position_ids` I64 [1,257], a constant arange | header |
| Attention | R-SWA: a frozen reference prefix plus a ring of 128 generated tokens; every reference token is attended at every step | **SR-SWA (DASH)**: block-sparse, ≤ 128 local + 12×32 remote tokens per query, in prefill **and** decode | §5.6–5.8 |
| RoPE | 1-D, theta 1e4, no scaling | Main and router: **page2d**, 63 intra-page token pairs (theta 1e4, "resample" spacing) + 1 page-parity pair (page mod 2, theta 256). Config `rope_theta=5e5` with YaRN ×4 is used only on the 1-D fallback path. | §5.5 |
| Context | `max_position_embeddings` 32768, shared by prompt and output | 131072. Positions only index blocks and causality; RoPE uses page-local coordinates. | config, `mod.rs:2750-2783` |
| Tokenizer | BPE vocab 128000 + 830 added (max id 128826) | Same BPE model, pre-tokenizer, normalizer and decoder. **+4 added non-special tokens**: `<PAGE>`=128827, `<BLOCK>`=128828, `<FORK>`=128829, `<CHILD>`=128830 (inference code reads only `<PAGE>`; the model emits `<BLOCK>`/`<FORK>` in its block grammar, §5.1; `<CHILD>` is in none of the nine oracle outputs). tokenizer.json sha256 `c55c165e…c589`. | tokenizer diff |
| Prompt | single: BOS + `<image>`×n + task text; multi: BOS + images + "Multi page parsing." | **Always** BOS + (`<PAGE>` + image tokens) per page + "Multi page parsing.", even for one page | §5.1 |
| Multi-page image size | each page squashed to 640² (111 tokens per page) | each page letterboxed to **1024** (273 tokens per page) in HF; SGLang defaults to gundam for every request (§5.11, DISC-U2-6) | §5.2 |
| Single-page default view | franken_ocr base 1024; HF v1 `infer` gundam ≤ 32 tiles | HF v2 `infer`: **gundam ≤ 16 tiles**; base available | §5.2 |
| Decode guards | n-gram 35 (128 single / 1024 multi) | n-gram *off* by default in HF `infer`; Baidu's evaluation and SGLang use 35/128/1024. A **page-count guard** is on by default: EOS banned until N `<PAGE>`, 20,000-token per-page cap. | §5.9 |

### 2.3 Benchmarks (from the tech report; we have not reproduced them)

| Benchmark | Metric | v1 | v2 | Context |
|---|---|---|---|---|
| OmniDocBench v1.6 (single page) | Overall ↑ | 93.92 | 94.02 | TextEdit 0.042→0.038, FormulaCDM 95.79→96.29, TableTEDS 90.16→89.60, ReadOrderEdit 0.129→0.128. Leaders: TeleOCR 96.91, OvisOCR2 96.47. |
| MPDocBench (≤30 pages) | Overall ↑ | 61.89 | 84.59 | TextEdit 0.167→0.052; T-TextEdit 0.605→0.100 |
| MarathonDocBench (100–300 pages) | Overall ↑ | n/a. The report says v1 is limited to about 40 pages; franken_ocr v1 shares 32,768 positions between prompt and output, so ≈29 pages at ~1,000 output tokens per page. | 77.06 | General VLMs score 7.5–33.7 |
| Per-step decode latency at 100 pages | — | — | 5.6× faster than R-SWA, 7.0× faster than full attention | GPU, **gundam**, batch 3 |
| Peak KV at hundred-page scale | — | — | 47.00 GB unwindowed → 1.77 GB windowed | GPU; windowing = §5.10 |

**Unknown crop mode.** The report does not state the crop mode used for the MPDocBench and
MarathonDocBench scores. Its latency table uses gundam, and the SGLang processor defaults to gundam
for every request, multi-page included. Baidu's long-document numbers are therefore plausibly gundam
multi-page, while HF `infer_multi` uses base (OQ-V2-14).

**Takeaway.** v2's value is **cross-page structure and length**, not single-page accuracy. Every
default-policy decision in §10 follows from that.

---

## 3. User-value analysis

### 3.1 Who uses franken_ocr, and what v2 changes for each

| User | Today (v1) | With this plan |
|---|---|---|
| **Single-page CLI and agent users** (`focr ocr page.png`) | v1 base mode, certified | **Nothing breaks.** v2 is opt-in via `--model unlimited-ocr-2`, at roughly equal accuracy (report, gundam) and roughly equal cost (analytic estimate; U2-F1 measures it). The default changes only if v2 is shown non-inferior on ground truth (§10). |
| **PDF users** (`focr ocr doc.pdf`, per page today) | Each page is parsed independently by v1 | After v2 passes the multi-page gate, **a PDF is one document by default with v2** (D17). Cross-page structure, the page-count guarantee and per-page JSON/figures cost roughly what per-page parsing costs. `--independent-pages` opts out. |
| **Multi-page users** (`--multi-page`, 5–50 pages) | One cross-page pass whose prompt (111 tokens per page) and output share 32,768 positions (≈29 dense pages). Weaker cross-page structure. Every raster is held in RAM. `--json` returns `pages: []`. | Markedly better cross-page text and tables (MPDocBench +22.7). A real per-page `pages[]`, layout boxes and figures. No practical length limit and bounded memory. When v1 hits its 32K budget, the error names `focr pull unlimited-ocr-2`. |
| **Book and archive digitizers** (100s–1000s of pages) | Not possible | **New capability:** windowed parsing on CPU with bounded memory, streamed page by page, resumable after a crash or Ctrl-C. Inputs: scanned PDFs (JPEG/CCITT/Flate pages) or a directory of page images. JBIG2/JPX/vector pages are **skipped with a `page` `skipped` event**, dropped from the prompt and kept in the source-page map, so numbering never shifts. Expect hours per 1,000 pages on CPU (estimate, §8.4). |
| **Rust embedders** | Five `recognize_multi_page*` variants (whole document, blocking) | One `recognize_document(source, &opts, sink)` with streaming, windowing and resume. The old variants are removed with a minor-version bump and a migration table (AGENTS.md: no compat shims). |
| **Agents (robot mode)** | `run_start`, `run_complete`/`run_error`, and a `page` event whose `page` mixes source page and emission index. The schema lists event kinds only, so no field can be validated (`stage` is advertised but has no emitter). | `robot schema` gains field-level `event_fields` (U2-X2, M0). On v2 runs, `page` gains `source_page`, `window`, `tokens`, `finish` and `model_id` (additive); v1 `page` events are unchanged. A new `window` event. `run_complete` gains `model_id`, `source_sha256`, `pages_out`, `windows`, `extra_markers`, `finish_reason` and `warnings[]`; a partial-output `run_error` carries the same partial fields. `robot triage` recommends v2 for long documents. The schema stays v1 (additive). |
| **Holders of v1 artifacts** | `unlimited-ocr.v0.7.0.int8.focrq` | Keeps working, byte-identically, throughout this plan (T3 gate). No forced downloads; v1 stays pullable. |

### 3.2 Principles that follow

1. **Additive, never destructive.** v2 is a new zoo entry. v1 keeps byte-identical behavior, enforced
   by a token-id golden gate (§7.6). Any v1 output change is a regression.
2. **Fail closed.** Every decode path, lever, daemon and converter either supports v2 explicitly or
   refuses it before weights load. No code path may silently run v2 weights through v1 math, or the
   reverse (D18).
3. **Evidence-gated defaults, per mode.** Single page, PDF/document and batch are separate decisions.
   Each flip needs ground-truth evidence with v1 live in the same invocation. Self-speedups and
   self-parity are not user-value evidence.
4. **Reference fidelity first.** The HF `infer()` path, run in **f32 on CPU**, is the parity oracle
   for unwindowed decoding. Windowed decoding follows the SGLang semantics and is gated against an
   independently written Python windowed oracle (§7.5).
5. **Ship value incrementally.** Every milestone (§11) ends in something a user can run. v1 users get
   early wins: the CJK streaming fix (U2-X1), the streaming page source (U2-D0) and the v1→v2 pull
   hint.
6. **CPU-honest performance.** v2 is not a single-page speedup on CPU. The mandate is no single-page
   regression and bounded long-document cost, measured per the perf ledger rules. Whole-process RSS
   is a gated number, not just KV.
7. **Agent ergonomics are part of the product.** New capabilities get robot events, schema entries,
   surface-matrix rows and `doctor`/`triage` awareness in the same milestone.

### 3.3 Non-goals (first wave)

- GPU, CUDA, SGLang or vLLM serving.
- Training or fine-tuning.
- Reproducing SGLang's radix cache, prefetch or Triton kernels.
- v2 on iOS, WASM or the website. That is milestone M5, with its own int4 calibration.
- Removing v1 in this plan.

**Intended end state.** If both gates in §10 pass, v2 becomes the single default for every mode and
for a bare `focr pull`, so a user needs one artifact and one resident daemon. Existing v1 installs
are never re-downloaded automatically, and v1 stays pullable.

---

## 4. Key decisions (with rationale)

**D1. Add `unlimited-ocr-2` as a separate zoo model rather than replacing v1 in place.**
*Why:*
- Single-page accuracy is a wash, so an in-place swap risks regressions (table TEDS is slightly lower)
  for no gain in the most common mode.
- The identity machinery maps "absent model_id" to "the default arch", so an in-place swap would
  silently load every existing v1 artifact as v2 (§6.1).
- Per-mode defaults need both models available.
- The shared code is most of the engine, so keeping v1 is cheap.

**D2. Harden identity before anything else.**
- An absent `model_id` permanently means `"unlimited-ocr"` (v1), never `default_arch()`.
- All **48 `default_arch()` call sites in `src/`** (19 in `convert.rs`) are audited. Each one either
  keeps a genuine "default model" meaning or becomes an explicit capability or identity method on
  `ModelArch`.
- Tokenizers are validated per arch: the required token ids must be present (`<PAGE>`=128827 for v2),
  and the tokenizer.json SHA-256 must match the arch pin.

*Why:* this removes the most dangerous failure, a silent misload or a mismatched tokenizer sidecar,
independently of v2. It is also a prerequisite for ever changing a default.

**D3. Keep the decoder dimension constants compile-time. Add a variant axis for attention, RoPE and
decode contract, not a runtime shape spec.**
- v1 and v2 share every dimension (§2.2), and the doctrine prefers fixed shapes in hot loops.
- The variance is one enum, `UnlimitedVariant::{V1Rswa, V2SrSwa}`. It is chosen from the arch at load
  time and dispatched once per layer call, with no trait objects in inner loops.
- The existing `ModelArch::decode_contract()` gains the v2 fields; no new method is added for that.
- `Decoder::DeepSeekV2MoeRswa` gains a sibling, `Decoder::DeepSeekV2MoeSrSwa`.

**D4. The primary parity oracle for unwindowed decoding is the HF `infer()` path, in f32 on CPU.
bf16 runs form the cross-precision ledger.**
*Why:*
- v1's correctness oracle has never run on the dev host under CUDA. v1 reached L4 token-exact 1.0 and
  L5 CER 0.0 against `gen_reference_fixtures.py --device cpu --cpu-patch --f32` (fixture set
  `fixtures/native_f32`, `tests/fixtures/PROVENANCE.md`).
- With f32 on both sides, Token Sieve routing should match exactly even where selection is genuinely
  selective. Most routing fragility is a bf16-vs-f32 artifact.
- On CPU the HF reference takes the dense SDPA path (§5.7). That path is semantically identical to the
  GPU Triton path; only the numerics differ.

Where HF and SGLang differ (§5.11), unwindowed mode matches HF and the difference is recorded in
`docs/DISCREPANCIES.md`.

**D5. Windowed (unlimited-horizon) decoding follows SGLang semantics, gated by a Python windowed
oracle written from the wheel source.**
*Why:*
- HF `infer()` never splits documents. SGLang is the only implementation, and Baidu's long-document
  numbers came from it.
- The Python windowed oracle (U2-D2a; U2-D2b is the parity gate) runs §5.10 on the HF model's DASH cache. It is written **from the
  wheel source by a different author than the Rust implementation**, so a misreading of §5.10 is not
  implemented twice. Where the oracle and §5.10 disagree, §5.10 is fixed.

We do **not** replicate the non-streaming stitching quirk that drops leading blank bodies while bodies
are surplus (§5.10.8, DISC-U2-4). The wheel's incremental `stream_owned_text` doesn't have it (its
streaming handler's end-of-window `select_batch_output` does), and it would corrupt page alignment.

**D6. Port in this order: single-page base, then multi-page base, then windowed, then gundam.**
- **Single-page base is routing-degenerate.** A page's 273 image tokens span ≤ 10 blocks and K=12, so
  every query selects all remote candidate blocks; only the continuous entmax bias varies. That makes
  it the cheapest place to prove the attention math.
- **Multi-page base is NOT degenerate.**
  - For pages 2..N, the page's `<PAGE>` marker and its first `128 − (s mod 32)` image tokens
    (s = the page's first image row) have no remote own-page block. That is 97–128 image tokens.
    They fall back to *all* remote blocks (§5.6.4) and route selectively over every earlier page; from
    page 3 on there are more than 12 candidates and top-K binds.
  - That is about 35–47% of each later page's image tokens, plus the instruction tokens. It is the
    cross-page channel, and the reason exact routing equality needs the f32 oracle.
- **Decode in base mode is degenerate:** own-page image blocks number ≤ 10 per page.
- **Gundam pages carry 273 (small image, both sides ≤ 640) or 483–2,033 image tokens** (Appendix A).
  Routing is selective once more than 12 own-page blocks are remote. 2×1 and 1×2 pages have about 16
  image blocks: on a single-page document, or a document's last page, the first ~20 decode rows see
  ≤ 12 remote own-page blocks and are not selective; every earlier page's decode rows are selective.
  Gundam is also uncertified for v1 today.

**D7. Keep the conservative int8 recipe policy unchanged, and enforce it by name.**
- Quantize exactly the 2,148 FFN/expert matrices.
- Keep everything else at source precision: attention q/k/v/o, `lm_head`, `embed_tokens`, the MoE
  router gate, all norms, the vision tower, the projector, **and every router/indexer tensor**.
- Recipe id: `unlimited-ocr-2-ffn-int8-attn-bf16-lmhead-bf16-v1`.
- `recipe.rs::is_decoder_attn_proj` currently matches `.self_attn.` + `q_proj.weight`, which also
  catches `self_attn.dash_router.q_proj` and `self_attn.indexer.q_proj`. It is narrowed to the exact
  `self_attn.{q,k,v,o}_proj.weight` names.
- The v1 WASM int4 recipe is refused for v2.

*Why:* doctrine #2. The router projections are small (0.33M MACs per layer per token) and steer a
discrete selection, so quantizing them is all risk and no gain.

**D8. Keep the unused `indexer.*` tensors in the artifact. Verify the I64 `position_ids` buffer and
skip it.**
- At inference, scoring uses `dash_router.*`. The `indexer` module only supplies page and RoPE helper
  methods (§5.6.7).
- Keeping it makes the census exact and costs ≈7.9 MB at bf16. Router plus indexer together are
  ≈15.7 MB, the v2−v1 `total_size` delta of 15,746,056 B.
- `position_ids` must equal `arange(257)`. Both the converter **and the raw safetensors loader** verify
  that and skip it.
  - Today both parsers reject I64 outright (`weights.rs:197-205`, `mod.rs:1442-1450`), so M1's raw-BF16
    load needs this.
  - `.focrq` has no I64 dtype, and the CLIP path already derives positions.

**D9. Start the KV cache as f32. Retirement is block-granular and part of the core v2 cache. bf16 KV
is a separate lossy lever.**
- A 32-row block is retired once both conditions hold:
  1. no non-open query can read it again: it holds no image token, or all its image pages are behind
     the current query page;
  2. it lies wholly behind the local window.
- Retirement is enabled only after the first generated `<PAGE>`.
- An open-row detector guards it, with a deterministic fallback to no retirement (§5.12).
- `M = min(16, nblk)` always uses the **logical** (pre-retirement) block count.

*Why:*
- Without retirement, an unwindowed 20-page run holds about 3 GiB of KV against v1's ~0.3 GiB. That is
  a memory regression on the very path whose default should flip.
- Row-level retirement is wrong. Page boundaries are never block-aligned (markers sit at 1+274p, and
  274 ≡ 18 mod 32), and a selected block is read whole.

**D10. v2 n-gram defaults are 35, with window 128 for a single page and 1024 for documents.**
- That is Baidu's evaluation setting and v1's contract. The oracle passes these explicitly, because HF
  `infer()` defaults to off.
- The n-gram domain is **generated ids only**, plus the hidden forced tokens in windowed mode.
  - This is a new domain parameter: the live v1 sampler's history includes prompt ids (`mod.rs` ~3829,
    ~3932), so it is not a reuse.
  - `--no-repeat-ngram 0` turns it off.

**D11. The per-page output cap is 20,000 tokens by default** (HF and README), set with
`--max-page-tokens`.
- **v2 document mode has no global generated cap by default.** `--max-length` and
  `FOCR_MAX_NEW_TOKENS` (default 32,768, a v1 global cap) apply **per page** when set. Otherwise a
  v1-tuned environment would truncate a book after about 30 pages.
- `FOCR_STAGE_BUDGET_FORWARD_MS` applies per window, scaled by the pages in the window. It is checked
  **cooperatively inside the document loop**, so completed pages survive.
  - The outer `run_blocking_stage_with_budget` timeout (`lib.rs:829-858`) discards partial output. It
    stays only as a hang backstop: the cooperative budget plus a grace margin.
  - An exhausted window checkpoints (D14), writes the completed pages, and exits 5 (`EXIT_TIMEOUT`)
    through a `run_error` carrying `finish_reason: "budget"`.
  - Cancellation exits 6 (`EXIT_CANCELLED`) the same way, with `finish_reason: "cancelled"`.
  - Producers: U2-B6 builds the partial-output path and produces `length` and `budget`; U2-D1a
    produces per-window `budget`; U2-E2 produces `cancelled` (cooperative cancellation in the driver);
    U2-D4 checkpoints the last completed window on each of those stops.
- **Truncation is never silent.**
  - A page stopped by the **per-page cap** reports `finish: "page_cap"`, and the run continues.
  - A page interrupted by the **position limit** reports `finish: "length"`. The run reports
    `finish_reason: "length"` on its `run_error` and exits 5, with the completed pages written (§9.2).
  - **Pre-windowing page limit.** Until U2-D1b windows longer documents, the default unwindowed path
    refuses a v2 document above the page limit up front, with exit 2.
    - U2-B6 starts the limit at 60 pages: `floor((131,072 − 64) / (274 + 1,900))`.
    - U2-C4 re-derives it from the measured distribution, as the largest N with
      `64 + 274·N + (N − 1)·p90 + p99 ≤ 131,072`, where p90 and p99 are generated tokens per page. Using
      per-page p99 for every page would be far too strict: page_0590 alone emits ~4,033 tokens.
    - An explicit unwindowed request (`Some(PageWindow::Off)`, or `--page-window off` once U2-D5
      lands) bypasses the refusal and relies on the exit-5 position stop. The library default
      (`DocumentOptions.page_window: None`, U2-E2 owns `PageWindow`) keeps the refusal, so default
      and explicit `Off` stay distinguishable. U2-B6 implements the bypass as an `enforce_page_limit`
      parameter of its document loop, which U2-D2b's S5 comparison sets directly.
    - Tests use "limit + 1 pages", never a literal.
- U2-C4 records the per-page generated-token distribution, and the cap default is revisited on that
  evidence. A runaway page costs minutes of CPU.

**D12. v2's single-page crop mode is decided by evidence (U2-G2).**
- Until then, v2 single pages run in **base** mode, which is faster on CPU and matches v1's default
  vision cost.
- Gundam is available via `--crop-mode gundam`, uncertified until U2-G1.
- Document-mode crop mode is decided the same way (OQ-V2-14). Document mode ships in base; gundam
  document mode is an evaluated option, not a blocker.

*Why:* Baidu's default is gundam, but gundam's CPU vision cost is ≈1.7–6.4× base (estimate from
§8.1's per-view figures), and v2's single-page edge is negligible either way.

**D13. Windowed decoding defaults to W=10 own pages per window** (Baidu's README; the wheel's env
default is 1).
- Documents with N ≤ W pages run unwindowed.
- `--page-window N|auto|off`, where `auto` derives W from the memory budget.

*Why:* unwindowed is the HF parity mode, and for short documents windowing only adds risk.

**D14. Long runs are resumable, and resume is byte-identical.**
- After each window:
  - each completed page is appended to an **NDJSON journal** `.partial`, as one serialized `PageResult`
    per line (U2-E3: text, layout, figure boxes, `source_page`, `half`, `finish`, tokens), then fsynced.
    A plain text append could not represent `pages[]` or figures;
  - the retained tail (§5.10.3) is written **atomically** to a checkpoint sidecar file. That is
    ≈400–560 rows, ≈47–66 MiB of f32 K/V and ~3 MiB of index keys; page state is replayed, not stored.
  - Only the latest checkpoint is kept, replaced by atomic rename.
- **Where they live:**
  - with `-o <out>`: `<out>.partial` and the checkpoint sidecar sit beside the output;
  - with stdout output: both live in `<cache_root>/document-runs/<resume_key>/`. They are never put
    under `models/`, where doctor's orphaned-partial detector quarantines any `*.partial`
    (`src/doctor.rs:246-285`).
- **No run-store index.** Checkpoints are located by resume key alone, and the run store's schema is
  unchanged. Run-store writes happen only on the CLI thread, before and after the engine call.
- **Resume key:** binary version, artifact SHA-256, input SHA-256, page selection, every decode
  parameter, and target arch.
  - On mismatch, byte-identical resume is refused and `--resume=recompute` (window recompute, §5.10.5,
    not byte-identical) is offered. A recompute resume adds `warnings[{kind:"resume_recompute"}]`.
  - Re-running the same command with a matching interrupted run resumes it. The terminal event
    carries `resumed_from`.

*Why:*
- `RunStore` (`src/storage.rs:68`, `FOCR_RUN_STORE` at `:34`) is best-effort telemetry: a failed write
  only prints a stderr line. Its schema is frozen (`SCHEMA_VERSION = 1`, `:31`;
  `tests/fixtures/runs_schema.json`), and it is being refactored under bd-pfui. Sidecar files avoid
  both problems.
- A 1,000-page CPU run takes hours, so users must be able to stop and resume without changing the
  result.

**D15. Robot changes are additive.**
- The existing `page` event (bd-2z0y decoded, bd-fck1 skipped) keeps every field. **On v2 runs** it
  gains `source_page`, `half`, `window` and `model_id` (decoded and skipped), plus `tokens`, `finish`
  and `elapsed_ms` (decoded only). That fixes today's mix, where skipped events carry the source page
  and decoded events the emission index, for v2. **v1 `page` events are unchanged**: v1 merges pages,
  so its bodies have no reliable source page.
- `window` is the only new event kind.
- U2-X2 (M0) adds field-level `event_fields` to `robot schema`. Every bead that adds a field or a
  warning kind extends it in the same change (U2-A7, D0, B6, E1, E3, D4, D5), so agents can validate
  every event.

*Why:* the robot contract is stable, versioned NDJSON (AGENTS.md "Agent Ergonomics").

**D16. Sharded-safetensors loading is generic across the zoo.**
- **Raw loads** check the index, the per-shard headers and the census on every load, and present one
  logical tensor directory. That matches v1: raw loads hash nothing today, and `source_sha256` is empty
  for safetensors.
- **Full shard SHA-256s** are verified by `focr convert`, which stamps them, and by `doctor`.
- `.focrq` has a single 32-byte `source_sha256` (preamble[11..43]).
  - For multi-shard sources it holds **SHA-256 over the concatenation of the shard SHA-256s in index
    order**. Index order is the distinct shard file names of the index's `weight_map`, sorted
    ascending (`docs/focrq-format.md`, "Sharded sources").
  - `provenance_json` lists each shard's name, size and SHA-256.

*Why:* v2 is the first default-family checkpoint with two shards, and future zoo models will shard
too.

**D17. With v2, a PDF is one document by default** (once the multi-page gate passes, §10).
- When the resolved model is v2, `focr ocr doc.pdf` runs document mode, windowed when N > W.
  `--independent-pages` restores per-page parsing.
- Image lists in `ocr-batch` stay independent unless `--multi-page` is given; a list of images is
  often unrelated.
- Document mode composes with `--pages`, `--split-spreads` (each logical half is one `<PAGE>`),
  `--json` and `--extract-figures`.
- It keeps a source-page map, so skipped or split pages never shift numbering.

*Why:*
- Under SR-SWA every per-page cost is the same inside a window as alone: 1024 vision, ~274 prefill
  tokens, ≤ ~512 attended tokens per step. So document mode costs about N single-page runs (estimate,
  measured by U2-F1).
- It adds cross-page structure, the page-count guarantee and per-page output.
- v1's trade-off was different (640 squash, attention growing with N, 32K shared positions), which is
  why v1 kept document mode opt-in.

**D18. v2 dispatch is fail-closed.** Each of the following either supports v2, is **ignored** with an
`ignored_lever` warning (it cannot reach a v2 path), or is refused with exit 2 **before weights load**
(a new pre-load check returning `FocrError::Usage`):
- every decode path: the conservative cached path (`generate_cached` over `DecoderWeightCache`: int8
  FFN, f32 attention/router/`lm_head`; where the certified artifact runs), the `_i8` cached path and
  its triggers (`FOCR_DECODE_INT8`, `ocr-batch --experimental-full-int8`, `force_int8_decode()`, WASM
  int4), chunked prefill, the stateless f32 path, batch spine, spec-verify;
- every entry point, including `OcrModel::recognize_multi_page*`, which never takes v1's 640-squash
  loop for v2;
- every lever: `FOCR_SPEC_DECODE`, `FOCR_SPEC_VERIFY`, `FOCR_BATCH_SPINE`, `FOCR_INT8_KV`,
  `FOCR_DECODE_INT8`, `FOCR_INT8_ATTN`, `FOCR_INT8_LMHEAD`, `FOCR_QKV_FUSED`, `FOCR_DECODE_STATELESS`,
  `FOCR_FUSE_NGRAM_LMHEAD`, `FOCR_FUSE_NORM_QUANT`, `FOCR_LMHEAD_SHARD`, `FOCR_LMHEAD_SHARD_TILES`,
  `FOCR_ATTN_GEMM`, `FOCR_RSWA_PARALLEL_ATTN`, `FOCR_PREFILL_CHUNK`, `FOCR_RUNAWAY_GUARD`;
- the resident daemon;
- the converter recipes.

Routing is an exhaustive `match` on `unlimited_variant()`. No `_ =>` arm may reach the v1 forward.

Levers are classified by **effective state** (U2-A7's `UOCR2_LEVER_POLICY`):
- value-parsed levers set to their kill-switch value (e.g. `FOCR_QKV_FUSED=0`) are no-ops for v2.
  **Presence-based** levers (`FOCR_LMHEAD_SHARD`, `FOCR_PREFILL_CHUNK`, `FOCR_FUSE_NORM_QUANT`,
  `FOCR_SPEC_DECODE`, `FOCR_SPEC_VERIFY`, `FOCR_FUSE_NGRAM_LMHEAD`, `FOCR_PROFILE_DECODE`) are armed by
  any value, including `0` (`lmhead_shard_enabled` is `var_os(..).is_some()`, `decoder.rs:2115`), so
  they are classified by presence;
- levers that reach only v1 R-SWA or `_i8` paths are **ignored** for v2, with
  `warnings[{kind:"ignored_lever", env}]`. Refusing them would break v1 users' exported tuning
  environments and guard nothing:
  - `FOCR_QKV_FUSED` (default ON, read only in `DecoderWeightCacheI8::build`, `decoder.rs:2222-2231`);
  - `FOCR_ATTN_GEMM`, `FOCR_RSWA_PARALLEL_ATTN`, `FOCR_INT8_KV` (read only in `rswa::decode_attention`
    and `RingCache::new`; ignorable only because U2-B5's SR-SWA arm never calls those);
  - `FOCR_FUSE_NORM_QUANT` (only `decode_step_with_cache_i8`, `decoder.rs:2990`),
    `FOCR_FUSE_NGRAM_LMHEAD` (only `generate_cached_i8`, `mod.rs:3939`), `FOCR_SPEC_DECODE` (only
    `generate_cached_i8`, `mod.rs:3949`), `FOCR_SPEC_VERIFY` (no production reader);
  - `FOCR_INT8_ATTN`, `FOCR_INT8_LMHEAD` (read only by `Recipe::from_env()` in
    `require_experimental_full_int8_recipe`, `mod.rs:480`; convert, census and loader use
    `validated_default`);
  - `FOCR_RESAMPLE` (v2 hard-wires PIL bicubic);
  - `FOCR_RUNAWAY_GUARD`, but only once the v2 arm stops building it: today `RunawayGuard` is built
    for every path (`mod.rs:3727`) and checked in `generate_cached` (`:3830`) and `generate_stateless`
    (`:4284`), so U2-B5/B6 must not build or check it on the v2 arm;
- only levers that would change v2 numerics on a reachable v2 path are refused. `FOCR_LMHEAD_SHARD` and
  `FOCR_LMHEAD_SHARD_TILES` reach `lm_head_cached` (`decoder.rs:1088-1090`) on the conservative path,
  so they were **refused until U2-C2**. Levers that select a refused path are refused with it:
  `FOCR_DECODE_INT8` (the `_i8` path), `FOCR_PREFILL_CHUNK` (chunked prefill, until U2-F2) and
  `FOCR_BATCH_SPINE` (until U2-F5); `FOCR_DECODE_STATELESS` selects a supported path;
- U2-C2 re-runs L4 with each "refuse until C2" lever on and flips each pass to supported-exact.
  *Outcome (2026-10-30):* `FOCR_FUSE_SWIGLU`, `FOCR_INT8_AUTOVEC=0`, `FOCR_LMHEAD_SHARD` (with
  `FOCR_LMHEAD_SHARD_TILES=7`) and `FOCR_SOFTMAX_COPY` give the f32 paths the oracle's ids on every L4
  case and the int8-FFN path its unarmed ids, so they are supported. `FOCR_MOE_SCORE_ORDER=1` routes
  experts in score order, not the reference's `torch.topk` slot order: the f32 paths still match, but
  the int8-FFN decode's ids move (S2's first departure from step 44 to 103, page_0014's from 47 to 44,
  S3 p9_p14 from 314 to 446 generated ids), so it stays refused.

*Why:*
- Today `forward` and `forward_dynamic` (`mod.rs:2105`, `:2137`) dispatch on id strings and fall
  through to the v1 `forward_pre` (`:2394`) for unknown ids. Only `ensure_arch_implemented`
  (`mod.rs:2376`) guards them, and it returns `NotImplemented` (exit 1) inside `forward_pre`, after
  load and preprocessing.
- `spec_decode_i8` (`mod.rs:4064`) is reachable only through `generate_cached_i8` (`mod.rs:3887`), i.e.
  the full-int8 or WASM-int4 paths, behind `require_full_int8_authorization()` (`:3895`). The certified
  path cannot reach it, so `FOCR_SPEC_DECODE` is ignored for v2; the `_i8` path itself is refused.
- v2 weights run through R-SWA code produce confident garbage with exit 0, the worst possible outcome
  for a user.

**D19. Document mode uses a streaming page source and a whole-process memory budget.**
- Pages are rasterized and preprocessed per window: W own pages plus 1 lookahead.
- Vision features are computed and dropped per page.
- Only page metadata and the source-page map scale with N.

*Why:*
- `recognize_pdf_multi_page` collects every selected page as a `DynamicImage` and clones the vector
  (`cli.rs:1821-1862`). At ~25 MB per 300-dpi letter page, a 300-page book holds ~7.5 GB per copy
  before any KV exists.
- Streaming also bounds time-to-first-page to one window.

**D20. Pages are split on token id 128827, and each page is decoded by a UTF-8-safe incremental
decoder.**
*Why:*
- The guard guarantees `<PAGE>` is a single token, so splitting on the id is exact and cheaper than
  re-scanning decoded text.
- Today the streaming observer in `recognize_multi_page_pres` (`mod.rs:2723`, observer `:2786-2824`)
  sets `stream_broken = true` permanently after a split multi-byte UTF-8 character (`:2802-2811`). That
  also suppresses the final flush (`:2821`), so `page` events stop for the rest of a Chinese document.
  `PageStream` itself (`postprocess.rs:531`) has no error state.
  - This already affects v1; U2-X1 fixes it.
  - MarathonDocBench is half Chinese.

**D21. Both entmax domains are implemented**, behind `EntmaxDomain::{TopM, AllCandidates}`.
- HF top-M is the parity default.
- The long-document default is chosen on ground truth (U2-C5b).

*Why:*
- Fallback queries are about 35–47% of every later page's image tokens in multi-page base, so the
  HF/SGLang difference is not a corner case.
- HF entmax combined with SGLang windowing is a combination no Baidu stack ran, and Baidu's
  long-document numbers are SGLang's.
- The second domain is one small function.

**D22. Each model gets its own artifact version, and the v2 census is a delta.**
- `versioned_quant_path` (`mod.rs:1022`) uses the global `UNLIMITED_OCR_ARTIFACT_VERSION` ("0.7.0",
  `lib.rs:74`) for every model id. It runs before load, from a spec path, with no arch in hand
  (`short_model_candidates(search_dir, spec, quant)`, `:1030`). So it will derive the arch with
  `arch_by_id(stem)`, falling back to v1, and use that arch's `artifact_version()`.
- The embedded v2 census lists:
  - only the 109 tensors v2 adds over v1;
  - the SHA-256 of the resulting 2,819-entry list;
  - full-tensor SHA-256s of every vision, projector, newline and separator tensor asserted
    byte-identical to v1. This is the mechanical vision-reuse proof.

*Why:* a second full census would duplicate 2,710 identical entries, and one global version cannot
name two artifacts.

**D23. Library first: one document API.**
- `recognize_document(source, &opts, sink)` replaces the five `recognize_multi_page*` variants.
- The CLI's document mode, streaming (D5) and resume (D4) are built on it.
- The crate is published, so this lands with a minor-version bump and a CHANGELOG migration table,
  both owned by U2-E2.
- U2-E2 also migrates the iOS crate's single call site (`focr-ios/src/lib.rs:~1152`,
  `engine.model.recognize_multi_page_dynamic`) onto an `OcrModel`-level document driver; focr-ios holds
  `Arc<OcrModel>`, not an `OcrEngine` (`focr-ios/src/lib.rs:280-284`). U2-H1 does the iOS product work.

*Why:* AGENTS.md forbids compatibility wrappers, and the product is library-first.

**D24. Document mode never aborts on one bad page.**
- A page that cannot be rasterized is skipped. That covers JBIG2 and JPX (`pdf.rs:317-322`) and
  vector/text content (`pdf.rs:160-167`, bd-0a7). The skipped page gets a `page` `skipped` event, is
  excluded from the prompt, and keeps its slot in the source-page map.
- v1 multi-page already skips such pages (`cli.rs:1836-1846`, aborting only if none renders, `:1853`).
  The defect is numbering: decoded `page` events are numbered by body ordinal (`postprocess.rs:535-536`,
  `cli.rs:1868-1874`), so after a skip they are misnumbered and can collide with the skipped page's
  number. The source-page map fixes that for v2 (`source_page`). v1 keeps body-ordinal numbering,
  because v1 merges pages and its bodies have no reliable source page.
- A directory of page images (natural sort) is a first-class document input. It is the realistic
  route for books rasterized out of band.

**D25. Model resolution is deterministic and visible.**
- Precedence: `--model` > `FOCR_MODEL_PATH` > the mode default (§10) > v1.
- If the mode default is not installed, v1 serves the run and prints a one-line pull hint. The hint
  goes to stderr, plus a `run_complete.warnings[]` entry of kind `pull_hint` (§9.2).
- JSON output and `run_complete` carry `model_id` and `source_sha256`. The run store records both
  in its existing `model_version_tag` column as `<model_id>@<source_sha256>`, with no schema change.
  Before U2-E1 the CLI filled that column with the `--model` path or "default".
- The digest is the `source_sha256` a `.focrq` records in its fixed prefix, `null` for a raw
  checkpoint. (This plan first called it `artifact_sha256`; U2-E1 renamed it, because hashing the
  artifact file on every run would cost seconds for a multi-GB file and raw loads hash nothing by
  design. A field named `artifact_sha256` would invite comparison with the artifact file's hash,
  which it is not.)

*Why:* "v2 when installed" would otherwise make pipeline output depend on cache state invisibly.

---

## 5. Reference-semantics specification (v2)

Citations point into the pinned v2 sources at commit `5632329a`:

| Tag | File |
|---|---|
| `MU` | `modeling_unlimitedocr.py` |
| `MD` | `modeling_deepseekv2.py` (DASH overlay) |
| `SB` | `_sparse_base_impl.py` |
| `DA` | `dash_attention.py` |
| `TR` | `sparse_gather_e2e_triton.py` |
| `W/…` | SGLang wheel, under `sglang/srt/` |

Claims marked **[W]** were verified against the wheel by the spec-fidelity review.

### 5.1 Prompt and tokenization

**Prompt string** (`MU:1073-1082`). The default is `"<PAGE><image>" * n_pages + "Multi page
parsing."`, even for `n_pages == 1`. HF handles three cases:

| Caller prompt | HF behavior |
|---|---|
| No `<image>` | Prepends `"<PAGE><image>"*n` |
| One `<image>` and n > 1 images | Replaces it with `"<PAGE><image>"*n` |
| Already n `<image>` placeholders | Uses it **as is**, which can lack `<PAGE>` (§5.5 fallback) |

franken_ocr always builds the prompt itself and never accepts caller `<image>` placeholders for v2.

**Token sequence** (`MU:531-580`):
`[BOS=0] ( [<PAGE>=128827] [<image>=128815] × n_img(p) )_{p=1..N} [instruction tokens]`
- BOS is prepended manually.
- Text segments are encoded with `add_special_tokens=False` (`MU:550,570-571`).
- There is no chat template and there are no role tokens. The SGLang template `unlimited-ocr` has
  empty roles and separators.
- The trailing EOS that the SGLang/vLLM tokenizer appends is stripped in inference mode
  (`W/configs/unlimited_ocr.py:502-507`) [W].

**`<PAGE>`.** It must tokenize to the single id 128827. It is an *added, non-special* token.
- franken_ocr's tokenizer already splits on all added tokens longest-first, regardless of the
  `special` flag.
- It decodes non-special added tokens literally, even with `skip_special`.
- The remaining v2 work is the required-token check and conformance fixtures (U2-A4).
- The tokenizer.json normalizer is empty.

**Image-token counts.** With nq = ceil((640/16)/4) = 10 and nqb = ceil((1024/16)/4) = 16:
- base (no crop): `nqb*(nqb+1) + 1 = 273` (`MU:564`);
- gundam with tiles (w_t × h_t, w_t>1 or h_t>1): `273 + (nq*w_t + 1) * (nq*h_t)` (`MU:565-567`).

**Generation output.**
- The output is `<PAGE>body_1<PAGE>body_2…<PAGE>body_N`, then EOS (id 1), which is not appended.
  Each `<PAGE>` *opens* a page.
- Body text does **not** use v1's grounding grammar (U2-E3b, found in U2-B6). Every block opens with
  `<BLOCK>{label}[<LEVELn>][x1, y1, x2, y2]<FORK>`, and its text runs to the next block:
  ```
  <PAGE><BLOCK>header[391, 31, 513, 53]<FORK>( 535 )
  <BLOCK>title<LEVEL1>[223, 178, 656, 197]<FORK>APPENDIX TO CHAPTER XXIV.
  ```
  - `<BLOCK>` (128828) and `<FORK>` (128829) are added tokens. `<LEVEL1>`/`<LEVEL2>` are plain BPE
    text and appear only on `title`.
  - Coordinates are 0–999 per page, as in v1. v1's `x/999·W` rescale lands on the right regions of
    page_0590, checked by cropping.
  - The nine oracle outputs carry 431 blocks with twelve labels: `text`, `title`, `header`, `footer`,
    `page_number`, `page_footnote`, `cross_text`, `append_text`, `table`, `image`, `image_caption`
    and `other`. `image` and `other` blocks have no text.
  - `cross_text` is a page's last text block when its paragraph runs onto the next page. It ends
    mid-sentence. `append_text` opens the next page with that paragraph's continuation. The model
    does not always pair them.
  - Neither reference parses the grammar. HF's `re_match` (`MU:44-59`) knows only v1's spans, so its
    `result.md` keeps the prefixes raw. franken_ocr's rendering is DISC-U2-13.
- franken_ocr segments pages on the **token id** (D20), not on decoded text.

### 5.2 Preprocessing (`MU:531-580`)

**Normalization.** mean = std = 0.5, so `x/127.5 − 1`. Pad color is (127,127,127). Each global view is
`ImageOps.pad(image, (1024,1024))`: an aspect-preserving letterbox with PIL BICUBIC and color 127
(`MU:545,561`).

**Crop mode selection.**
- In HF, `crop_mode=None` means gundam when there is exactly one image and base otherwise (`MU:541`).
- `infer_multi` forces base (`MU:1117`).
- SGLang defaults to gundam for every request (DISC-U2-6).
- franken_ocr's v2 default is base for single pages and documents alike (D12).

**Base mode.** One global 1024 view per page, **with no 640 pre-resize**. This is a deliberate change
from v1's multi-page path ("vLLM's training-parity fix", `MU:536-538`).

**Resampling kernel.** For v1, multi-page forces PIL-bicubic squash and single-page uses the
CatmullRom default (DISC-001, `FOCR_RESAMPLE`). For v2, the 1024 letterbox uses the **PIL-bicubic
exact kernel for both single pages and documents**: that is what the reference computes, and L0 is
exact. Any faster kernel is a lossy lever with its own CER-gated DISC entry.

**Gundam mode.**
- If both image sides are ≤ 640, there are no tiles: ratio (1,1) (`MU:554`).
- Otherwise the image goes through `dynamic_preprocess(min_num=2, max_num=16, image_size=640)`
  (`MU:229,1043`). v1 used max_num=32.
- Any grid with i·j ∈ [2,16] is allowed (`MU:236`), so 1×16 and 16×1 are possible.
- Tiles are 640² crops of the resized image.

**Vision-feature scatter order** (`MU:731-790`, identical to v1):
- tiled: `[local features with image_newline per row] ++ [global features with image_newline per row]
  ++ [view_seperator]`;
- untiled: `[global with newline] ++ [view_seperator]`.

Multi-page concatenates per page, in order. franken_ocr's `connector::fuse_no_crop`
(`connector.rs:403`) already handles interleaved `<PAGE>` tokens.

### 5.3 Vision tower

Unchanged from v1: same code and byte-identical weights. franken_ocr's SAM, CLIP, bridge and connector
code, and its vision parity evidence, apply as-is. The delta census (D22) proves the identity
mechanically with full-tensor SHA-256s.

### 5.4 Page bookkeeping

These definitions drive RoPE (§5.5) and the Page Gate (§5.6.4).
- `ids` is the token-id sequence, `img = ids==128815`, `mk = ids==128827`.
- `cum[i]` is the inclusive count of `mk` up to position i.
- `last_img` is the index of the last image token.
- `N = cum[last_img]` is the number of prompt pages. v2 prompts are always in marker mode (N > 0).

**Image-token page id** (0-based; `SB:2433-2435`): `img_page[i] = cum[i] − 1` for image tokens, and −1
otherwise.

**Query page id** (`SB:2439-2456`; inference sets `_PAGE_SCOPE_PREFILL=True`, `MD:333`):

| Token | Query page |
|---|---|
| Prefill tokens ≤ `last_img` (markers, image tokens) | `cum − 1` |
| BOS (cum=0) | −1 |
| Instruction tokens after `last_img` | −1 |
| Generated tokens | `cum − 1 − N` (−1 if negative) |

So the k-th generated `<PAGE>` and its body map to prompt page k−1.

**Per-block image-page range** (block size 32; `MU:481-528`).
- For each KV block, `bmax` and `bmin` are the max and min image page ids among its tokens, or −1 if
  it has none.
- Both are computed once at prefill, from the prompt.
- HF decode ignores generated ids for these ranges, even if id 128815 were sampled (`MU:514-528`).

**Decode query page.**
- Start with `mcum = cum[-1]` (all prompt markers) and `pp = N` (`MU:506,512`).
- Each **fed** `<PAGE>` increments `mcum`.
- `qp = mcum − 1 − pp`, or −1 if that is negative.

For standard prompts `cum[-1] == N`, so the first fed generated `<PAGE>` gives qp = 0.

This HF seed applies to window 0, recomputed windows and unwindowed runs. A **reused window** (§5.10.4:
no BOS, no instruction) is seeded from the wheel's carry instead (`W/session/uocr_doc_reuse.py:433-437,
1103-1109`). With M = the `<PAGE>` count of the rewritten prompt and pp = M, the gate carry is
mcum = pp + 1, so qp = 0 before the forced `<PAGE>` and qp = 1 after it. Seeding a reused window with
the HF rule would give qp = 0 for the whole first own page, which is wrong. §5.5 gives the matching RoPE
carry.

### 5.5 page2d RoPE (main attention **and** router)

Coordinates come from `SB:2511-2568` (prefill) and `SB:1451-1489` (decode state machine).

**Bucket per token.**
- Prefill tokens ≤ `last_img`: bucket = `cum`, so 1..N.
- Generated tokens and instruction tokens: bucket = `cum − N`, clamped ≥ 0 (`SB:2554-2555`).
- BOS and the instruction therefore land in bucket 0, the reserved "no page" bucket.

**`tok_pos`.** The 0-based running count of earlier tokens in the same bucket.
- `pf_cnt[b]` counts only bucket b's prefill rows.
- A response page's tokens continue after that page's prompt tokens, so the response `<PAGE>` of page
  b gets `tok_pos = pf_cnt[b] = 1 + n_img(b)`.

**Page position.** `page_pos = bucket`, folded to `bucket mod 2` at angle computation
(`sparse_rope_page_mod=2`).

**Decode state machine.** Per sequence, the state is `(page, tok, pf_cnt[], n_pg)`, with
`n_pg = N+1`.
- Seed: `page` = the bucket of the last prompt token (0, the instruction bucket), and `tok` = **that
  token's `tok_pos`** (`SB:1473-1474`).
- On each fed token (`SB:1484-1489`):
  - if it is `<PAGE>`: `page = min(page+1, n_pg−1)` and `tok = pf_cnt[page]`;
  - otherwise: `tok += 1`.
- Extra markers clamp to the last page. HF then resets `tok = pf_cnt[N]` (DISC-U2-7). That duplicates
  the coordinate of page N's response `<PAGE>`, so "(bucket, tok_pos) is unique" and "state machine ≡
  prefill builder" hold only for sequences with ≤ N generated markers. The prefill builder has no upper
  clamp (`SB:2555,2561`).
- **Reused-window seed** (§5.10.4; `W/session/uocr_doc_reuse.py:433-437`). A reused window has no BOS
  and no instruction, so its last prompt token sits in bucket M (its last local page), and the HF seed
  would clamp the forced `<PAGE>` to bucket M. Instead, with M = the `<PAGE>` count of the rewritten
  prompt and pp = M:
  - **RoPE carry:** the marker count is M + 1, so the response bucket is min(max(mc − pp, 0), pp) = 1.
    `cnt[]` = `replay_coords` over the rewritten prompt only: rows (a), (b), (c) and the newly
    prefilled pages; the text tail (d) is not counted (wheel `:292`, `:1100,1122-1124`). Only cnt[≥2]
    is ever used.
  - The forced `<PAGE>` then gets bucket 2 and `tok = cnt[2] = 1 + n_img(lo)` (the §5.10.4 table).
  - The HF seed applies only to window 0, recomputed windows and unwindowed runs.

**Frequencies** (`SB:2027-2061`). dim = 128, half = 64, page_pairs = 1, so n_tok = 63.
- `inv_tok[i] = 10000^(−i/63)` for i in 0..63. This is "resample" mode; note the /63, not 2i/128.
  Compute it in the source's operation order, `1.0 / (10000 ** (f32(i) / 63))` in f32 (`SB:2032`),
  and assert the 63 values against the oracle's dumped `inv_tok`; pin them as constants if Rust's
  `powf` differs from torch's `pow`.
- `inv_page[0] = 256^0 = 1`.
- `ang = [tok_pos·inv_tok[0..63], (page_pos mod 2)·1]` gives 64 angles, and `emb = cat(ang, ang)`.
- Rotation is NeoX `rotate_half` via transformers' llama `apply_rotary_pos_emb` (`SB:39-42`):
  `x·cos + rotate_half(x)·sin`, with `rotate_half([x1,x2]) = [−x2, x1]`. It is not the interleaved
  local variant at `SB:463`.
- cos and sin are computed in fp32, then cast to the activation dtype (`SB:2061`).

**Router RoPE.** The router uses the **same** 2-axis RoPE, with identical parameters and coordinates,
on its 128-d q/k (`MD:126-143,177-189,510-511`; `SB:1417-1425` vs `SB:2331-2347`).

**Scale.** The main-attention softmax scale is 1/√128 (`DA:937`). YaRN's `attention_factor` does not
apply on this path.

**Fallback.** If a row has no `<PAGE>` or no image token, HF falls back to:
- 1-D YaRN RoPE (theta 5e5, ×4) on main attention (LlamaRotaryEmbedding) and the router
  (`SB:2337-2340,2542-2552`);
- page ids `run//273` (`SB:2434`);
- no guard (`expected_pages=0`, `MU:585,1141`).

This never triggers with the default or `infer_multi` prompts. It does trigger for a caller prompt that
already has n `<image>` placeholders and no `<PAGE>`. SGLang has no 1-D fallback at all
[W: `models/unlimited_ocr_v2.py:151-152`]. franken_ocr always builds `<PAGE><image>` prompts (§5.1)
and **does not implement the fallback**.

### 5.6 Token Sieve router (per layer, per query)

**Constants.** Inference reads the checkpoint snapshot of `sparse_attention_config` (`MD:56-64,
296-335`):

| Constant | Value |
|---|---|
| block_size | 32 |
| local_blocks | 4 (local_window = 128 tokens) |
| K (read budget) | `dash_topk_blocks` = 12 |
| M (entmax window) | 16 (`dash_topm`) |
| idx_scale | `dash_scale_per_layer[l]/√128` = 128/√128 = 11.3137085 for all 12 layers |
| floor | `dash_floor_per_layer[l]` = 1e-4 |
| σ | 1.0 |
| stage0 | `head_cls` |

Pitfalls:
- `_persist_router_cfg` overwrites the live config dict at construction, and inference reads the
  snapshot taken before that.
- The env var `DASH_STAGE0` would override stage0 (`MD:312`). The oracle must leave it unset.

**5.6.1 Router projections** (`MD:365-377`). The router input is the same normalized tensor that feeds
q/k/v. `SB:3692-3697` applies `input_layernorm` before `self_attn`, and `MD:369` maps NaN and ±Inf to
0:
```
h  = nan_to_num(input_layernorm(x))     # same tensor that feeds q_proj/k_proj/v_proj
iq = q_norm(q_proj(h))                  # [128]
ik = k_norm(k_proj(h))                  # [128]
```
`q_norm` and `k_norm` are **Gemma-style RMSNorm** (`DA:844-855`; eps = `rms_norm_eps` = 1e-6):
```
x32 = nan_to_num(x.float())
y   = x32 * rsqrt(mean(x32²) + 1e-6)
y   = clamp(y, ±1e4) * (1 + w)
y   = clamp(y, ±1e4).to(activation_dtype)        # rounded back to bf16 in the reference
```
- Next, 2-axis RoPE (§5.5) is applied to iq and ik **in the activation dtype**, with cos and sin also
  cast.
- Only then are they cast to f32, with NaN/Inf → 0 (`MD:375-376`).
- Index keys are cached **post-norm, post-RoPE, f32** (`MD:440-444,515-517`).
- In the f32 oracle every cast is a no-op. Against bf16 references these roundings are part of the
  cross-precision ledger, because routing is discrete.

**5.6.2 Block summaries (stage0 = head_cls)** (`MD:391-408`). For block c and its valid tokens j:
```
a_j   = clamp((ik_j · head_cls) / √128, ±1e4)
w_j   = exp(a_j − max_valid a) · valid_j ;  w_j /= max(Σ w, 1e-6)
k̄_c  = nan_to_num(Σ_j w_j · ik_j)
```
A partial tail block is never remote: remote requires `c ≤ q//32 − 4`, so every remote block is full
and lies before q. Validity only matters for the tail block.

The source scales by multiplying with `inv_sqrt_d = 1.0/math.sqrt(d)` (`MD:387,401,411`), not by
dividing by √128. Implementations use `× f32(1/√128)` so the f32 rounding matches.

**5.6.3 Remote block scores** (`MD:409-417`; `SCORE_NEG` at `DA:741`):
```
s_c = clamp((iq · k̄_c) / √128, ±1e4)   if c ≤ floor(q_pos/32) − 4 and c has a valid token
s_c = SCORE_NEG (−1e9)                   otherwise   ("not remote")
```
`q_pos` is the query's **1-D position within the current KV sequence**. In windowed mode that
position is window-local.

**5.6.4 Page Gate (candidate set).**
- **Decode** (`MU:481-528`): `allow[c] = (bmin[c] ≥ 0) ∧ (bmin[c] ≤ qp ≤ bmax[c]) ∧ (qp ≥ 0)`.
  - Generated-text blocks (bmin = −1) are never allowed.
  - If `qp < 0`, or no block is allowed, every block is allowed. This is an **open row**.
  - The gate applies only because `image_only=True` (`MD:523-526`, config).
- **Prefill** (`MD:108-123`, `SB:2629-2649,2706-2737`):
  - `cand[q,c]` is true when block c contains any image token with `img_page == query page of q`
    (≥ 0).
  - If q has no such block, every block is a candidate for q (per-query fallback).
- **Then** (`MD:339-348`):
  ```
  remote = s > SCORE_NEG/2
  cand   = remote ∧ allow_or_cand
  if not any(cand): cand = remote        # also an open row
  ```
  - If nothing is remote (q is in the first 4 blocks), the query attends to its local window only.
  - In multi-page base prefill, the `<PAGE>` marker and the first `128 − (s mod 32)` image tokens of
    pages 2..N (s = the page's first image row) hit `not any(cand)` and route over all earlier blocks
    (D6). That is 97–128 image tokens, about 35–47% of 273. A page-p query at row s+t has a remote
    own-page block iff t ≥ 128 − (s mod 32) (`MD:414-415`, `keep = blk_ids <= qblk − local_blocks`).
    Simulated with the HF rules on a 6-page base prompt: 108/122/104/118/100 fallback image rows per
    page, plus the marker.

**5.6.5 Entmax-1.5 over the top-M window (HF semantics; `EntmaxDomain::TopM`)** (`MD:346-362`,
`DA:772-813`):
```
x_c  = s_c · idx_scale  if cand[c] else −1e4 ; nan/±inf → −1e4
(xv, xi) = topk(x, M = min(16, nblk))          # nblk = LOGICAL block count, per FORWARD; descending
prior = entmax15(xv):
    z = (xv − max(xv)) / 2
    sort z descending; for k = 1..M: mean_k, meansq_k, ss_k = k(meansq_k − mean_k²),
    δ_k = (1 − ss_k)/k, τ_k = mean_k − sqrt(max(δ_k, 0))
    support = #{k : τ_k ≤ z_sorted[k]} clamped ≥ 1 ; τ* = τ_support
    p = max(z − τ*, 0)²
is_cand = cand[xi]
K' = min(12, M) ; take the first K' slots: idx = xi[:K'], p = prior[:K'], valid = is_cand[:K']
logp = ln(max(p, floor)) ; μ = mean of logp over valid slots (count clamped ≥ 1)
bias = (logp − μ)/σ  if valid else 0          # no clamp at inference
```
- The selected blocks are `idx` where `valid`, each carrying scalar `bias`.
- Zero-prior candidate blocks inside the first K' slots **are still read**, with bias `ln(1e-4) − μ`.
- **`nblk` is per forward, not per query** (`MD:437`, `MD:351-352`, `MD:519`).
  - HF prefill is defined only on an empty cache: nblk = ceil(S/32), where S is the forward's query
    count (= its KV length; `MD:436-437`), and the router scores only this forward's index keys
    (`MD:440-445`). nblk is identical for every query row, so early prefill rows use
    M = min(16, ceil(S/32)), not their causal block count.
  - For an extend over a non-empty cache (a reused window, §5.10.4), franken_ocr defines
    nblk = ceil(total KV rows/32) and scores against all cached index keys. U2-D2a's oracle implements
    the same rule explicitly.
  - In decode, nblk = ceil((pos+1)/32).
  - Chunked prefill (F2) must therefore use the unchunked M = min(16, ceil(S_prompt/32)) for every
    chunk, with pads counted from the logical nblk. With both, chunked ≡ unchunked selection holds
    unconditionally.
  - Re-prefill ≡ incremental decode (§5.8) is bitwise only if, on every row, the max candidate
    x ≥ −1e4+2 and no candidate has x ≤ −1e4. Tests assert that precondition, or log the affected rows.
- **Padding.** z = (x − max)/2 and τ* ≥ z_max − 1 (`DA:762-798`).
  - A pad (x = −1e4) has p = 0 iff z_pad = (−1e4 − m)/2 ≤ τ*, where m is the max candidate x.
    **m ≥ −1e4+2 is sufficient**, since then z_pad ≤ −1 ≤ τ*. Below that, pads *may* enter the support
    and move τ*; tests must not assert p_pad > 0 merely because m < −1e4+2.
  - Computing top-M over candidates only and **merging** min(M, nblk − #cand) pads into the full top-M
    order is exact in both cases. Appending pads after the candidates is not.
  - Both the F2 lever and D9 retirement rely on the merged form.
- `topk` tie order among equal scores follows torch's descending `topk`. Ties among *candidates* at
  the K' boundary are a tie-flip class (§7.2).

**5.6.6 `EntmaxDomain::AllCandidates` (SGLang semantics).** SGLang runs entmax over **all** candidate
blocks (`W/layers/attention/route_triton.py:12,175-202`) [W]:
- τ comes from a 24-step bisection over [−1, 0] with fp64 sums;
- the read set is `cand ∩ {z > thr}`, found with a 40-step count bisection, so a tie at 12th place can
  read fewer than 12 blocks.

HF and SGLang agree whenever the candidate count is ≤ 16, up to bisection precision and tie handling.
That covers every own-page query in base mode. They do **not** agree on:
- fallback queries (BOS, instruction, early image blocks of pages 2..N, generated text before the
  first `<PAGE>`);
- gundam pages.

franken_ocr implements `AllCandidates` with an exact τ (sort-based, as in §5.6.5 but over all
candidates) and records the bisection difference as part of DISC-U2-1. The read rule follows the wheel
(`route_triton.py:183,191-202,789`):
- non-candidates are excluded from the τ sum;
- the read set is the top K = min(budget 12, M 16) = 12 candidates **by score z**, which includes
  zero-prior candidates (bias ln(1e-4) − μ). Ranking by prior is ambiguous at p = 0;
- every candidate is read when there are ≤ 12.

AllCandidates ≡ TopM tests must exclude 12th-place ties, rows whose max candidate x < −1e4+2, and
rows with any candidate x ≤ −1e4.

**5.6.7 Unused indexer.** `self_attn.indexer.{q,k}_proj` are not used at inference; the dash path
scores with `dash_router.*`. The indexer object only hosts page and RoPE helper methods (`MD:96-143`).

### 5.7 Block-sparse biased attention

For a query at position q with selected blocks B, each with bias β_b (`MD:454-481` decode;
`DA:1230-1261` dense prefill; `TR:37-118` Triton prefill):
```
local(j)  = (q − 128 < j ≤ q)                                # 128 tokens including self
remote(j) = (j ≤ q) ∧ (block(j) ∈ B) ∧ ¬local(j)             # local wins on overlap
logit_j   = (q·k_j)/√128 + (remote(j) ? β_block(j) : 0)
out       = softmax over {j : local(j) ∨ remote(j)} of logit_j, applied to v_j
```
- Attention is computed per head. The 10 heads share the selection, because the router has one head
  per layer (`MD:480`).
- The output goes through `o_proj` as before. There is no attention bias anywhere
  (`attention_bias=false`).

**Reference numerics.** All paths are semantically identical; only the rounding differs.
- **GPU prefill: the Triton e2e kernel.** The preconditions are `router.use_triton and q.is_cuda and
  attn_mask is None` (`MD:447-450`); the model forward drops the mask (`SB:~3960`).
  - The bias is fp32.
  - q·scale is rounded to bf16 before QK.
  - P is rounded to bf16 before PV.
  - The online softmax runs in fp32.
- **CPU prefill**, or when the Triton import fails, or when positions ≠ arange (`TR:370-381`):
  `_forward_dense` runs SDPA with an additive mask, bias included, in the activation dtype (bf16, or
  f32 in the f32 oracle).
- **Decode on both devices:** `_decode_attend` does fp32 math over activation-dtype q/K/V. The output
  is cast to the activation dtype before `o_proj` (`MD:530`).
- **Logits:** `lm_head(h)` in the activation dtype, then `.float()` (`MU:1165-1167`).

franken_ocr computes in f32 throughout. The parity target is the f32 oracle (D4). Against bf16, the
tolerance is measured (OQ-V2-3).

### 5.8 Prefill vs decode

- **Prefill** (`q_len > 1`): every prompt token is a query with its own routing (§5.6) and sparse
  attention (§5.7). The router index keys of all prompt tokens are cached.
- **Decode** (`q_len == 1`):
  1. append the token's K, V and index key;
  2. recompute its block's summary;
  3. route against all blocks, with candidates per §5.6.4;
  4. attend per §5.7.
- **Positions:** `arange` in prefill and `S + len − 1` in decode (`MU:1163,1188`).
- **Re-prefill ≡ incremental.** Incremental decode and a fresh prefill over prompt + generated[:t]
  should select identically: both use causal per-query routing and the same page ids. This is a
  metamorphic test (§7.4), not an assumption, and it runs on one code path (§6.2, the conservative
  cache). It has two preconditions:
  - no generated token equals 128815. A re-prefill would treat such a token as an image token, while
    decode ignores it;
  - on every row, the max candidate x ≥ −1e4+2 and no candidate has x ≤ −1e4. Prefill computes M from
    the whole forward's nblk and decode from the cache length (§5.6.5), so the pad counts differ, and
    the two agree only under that condition.
- **Only the last position** goes through `lm_head` after prefill (`MU:1165-1167`).
- **KV:** HF keeps the **full** DynamicCache. Ring and sliding windows are disabled, and nothing is
  evicted (`MU:1151-1153,1160`; `MD:509,516`).

### 5.9 Decode loop (`MU:1124-1200`)

1. **Greedy** when temperature is 0. Argmax over the f32-converted logits takes the **lowest id on
   ties** (`MU:1167,1176`). With bf16 logits, ties are much likelier than in f32. Temperature sampling
   is out of scope for parity.
2. **Logit processors**, per step in HF order (`MU:1168-1172`): n-gram first, then the guard.
   - **n-gram ban** (when `no_repeat_ngram_size > 0`; D10), over *generated ids only*
     (`MU:608-623`):
     - with `prefix = gen[-(n−1):]`, ban `gen[i+n−1]` for each i in `[max(0, len−window), len−n+1)`
       where `gen[i:i+n−1] == prefix`;
     - `ngram_window ≤ 0` means **100** (`MU:1092`);
     - a whitelist set is subtracted from the bans (`MU:620`).
   - **Page-count guard** (`MU:585-604`). Definitions:
     - `expected` = the count of `<PAGE>` in the prompt;
     - `mc` = the count of `<PAGE>` in gen;
     - `page_len` = tokens since the last generated `<PAGE>`, excluding the marker itself. Before the
       first generated `<PAGE>`, it counts from the start of generation (`MU:593-595,1142`).
     ```
     if mc > expected: only EOS allowed
     elif 0 < max_page_tokens ≤ page_len: only (<PAGE> if mc < expected else EOS) allowed
     elif mc < expected: EOS banned
     ```
     The guard is disabled when the prompt has no `<PAGE>` (`MU:585,1170`). `max_page_tokens ≤ 0`
     disables only the cap (`MU:600`).
3. **Stop conditions:**
   - EOS (not appended);
   - `max_new_tokens`, default `max(1, max_length − S)` with `max_length` 131072;
   - `pos+1 ≥ max_length` (`MU:1089,1179-1190`).
4. **Order hazard** (OQ-V2-7, DISC-U2-2). If the n-gram ban removes the token the guard then forces,
   every logit is −inf and HF's argmax emits id 0. SGLang applies the guard first, and a forced token
   skips the n-gram ban (`W/…/custom_logit_processor.py:194-262,325-338`) [W].
   - **franken_ocr makes forced tokens authoritative** (SGLang order).
   - The oracle comparison asserts that this case never occurs on any fixture.
5. **`<PAGE>` on the last page** (DISC-U2-3). On the last page of an unsplit request, SGLang also bans
   `<PAGE>` when not `end_on_marker`. HF does not.
   - With HF semantics an extra `<PAGE>` is possible, followed by a forced EOS. Our splitter counts it
     (`extra_markers`) and drops the empty trailing body.
   - Unwindowed mode follows HF; windowed mode follows SGLang (§5.10.7). In SGLang the ban reaches
     only a last window that was recomputed, because `end_on_marker = (lookahead > 0) || reused`
     (`CLP:227`): every non-last window has a lookahead, and every reused window sets
     `_uocr_end_on_marker` (`:1109`).

### 5.10 Windowed ("unlimited-horizon") decoding (SGLang semantics) [W]

The source notes are `docs/truth-pack/oq/uocr2-windowed-sglang.md` (U2-A1; `W/` is the pinned
dev10 wheel's `sglang/srt/`, and its section 5 reviews the dev13 delta). Citations are under
`W/session/uocr_doc_reuse.py` and `W/entrypoints/openai/doc_split.py` unless noted.

**5.10.1 Planning.**
- **Window size.** W own pages per window. franken_ocr defaults to 10 (D13); the wheel's env default
  is 1 (`W/environ.py:1354-1358`).
  - The wheel keeps W per mode: a base limit for base/large/small/tiny, and a gundam limit otherwise
    (`doc_split.py:337-343`).
  - Own pages per window = `min((ctx − 65) // per_page_est, W)`, with `per_page_est` = 274 for base
    and 1514 for gundam (`doc_split.py:28-35,141-143`).
  - In the wheel, `ctx` is the server's `max_req_input_len` (`W/entrypoints/openai/serving_chat.py:378`)
    = min(context_len − 1, KV pool − 1) − 5 (`W/managers/tp_worker.py:544-557`). context_len defaults
    to min((W+la)·2034 + W·max_page_tokens + 1024, 131072), with W the larger of the gundam and base
    windows, or to 131,072 outright when the doc limit is off or the page cap is ≤ 0
    (`W/configs/model_config.py:305-327`). So under the README configuration ctx ≤ 131,066. It does
    not matter at W = 10, where the `W` term binds.
  - franken_ocr's bound adds the output side, because outputs share the window's 131,072 positions:
    own pages per window = min(prompt-derived bound, W, the U2-B6/C4 pre-windowing page limit, D11).
- **Ownership.** Windows are contiguous and own their pages. Every window except the last also encodes
  **one lookahead page**, the next window's first page, which is never transcribed.
- **Short documents.** If `N ≤ W`, decoding is unwindowed. The wheel never splits a document that
  fits: it returns one window `[(0, N, 0)]`, with no lookahead and `max_tokens` clamped to its
  capacity, or no plan at all (`doc_split.py:150-156`). franken_ocr has no global cap in document mode
  (D11), so the clamp does not arise.

**5.10.2 Window 0 and any recomputed window.**
- Prompt: `[BOS] (<PAGE> img)×(own) [<PAGE> img(lookahead)] "Multi page parsing."`, with **page_base =
  lo**.
- Guard: `expected = #markers − context_pages(0) − lookahead`, which equals the number of own pages.
- A window with a lookahead ends when the model emits the `<PAGE>` that would open the lookahead page,
  because `marks > expected` forces EOS. A natural EOS on the last own page is also accepted.

**5.10.3 Retention at the end of window k.** This applies only if window k ended by EOS (or a stop
string) and is not the last window. Keep these KV rows with their router index keys, and free the
rest. The wheel keeps the whole 32-row KV pages that hold them, clipped to the allocation
(`tail_ranges`, `:323-366`; `_trim_to_tails` frees only the gaps, `:903-932`). Only these rows are
read, so that affects memory only. Page state is **not** copied; it is replayed in the new frame (§5.10.4; `:500-516`, where
`move_kv_cache` is followed by `_write_page_state` from `replay_page_isolation`, and `:1120-1138`):
- **(a)** the `<PAGE>` marker of window k's last own page. This page is the **tail page**, global page
  lo−1.
- **(b)** the **image tail** `[E − 128 − (E mod 32), E)`, where E is one past the tail page's last
  image token. That is 128–159 tokens, block-aligned (`:256-268`).
- **(c)** the whole lookahead page, marker and image tokens. It is window k+1's first own page, global
  page lo.
- **(d)** the **text tail**: up to 128 trailing generated tokens of the tail page's body, never
  including a marker.

**5.10.4 Window k+1 with reuse.**

*KV layout, in order:*
1. copied (a) at position 0;
2. copied (b);
3. copied (c);
4. **prefill** of own pages lo+1..hi−1 and of window k+1's own lookahead page;
5. copied (d).

There is **no BOS and no instruction**. **page_base = lo − 1**, because the tail page is local page 0.
1-D positions are window-local; row 0 is position 0.

*No re-rotation.* **Copied K and index keys are not re-rotated.** Intra-page `tok_pos` is unchanged by
construction, and page parity is preserved by this rule
(`W/layers/attention/unlimited_ocr_v2_backend.py:1001-1031`):
```
rope_page_coord = local_bucket + (page_base mod 2)   if local_bucket > 0
                = 0                                   if local_bucket == 0   (BOS/instruction)
```
- A recomputed window numbers its own pages 1..n, with page_base = lo.
- A reused window numbers the tail page 1, the first own page 2, and so on, with page_base = lo − 1.
- Either way, `local_bucket + page_base` ≡ global bucket (= global page + 1) mod 2 for every row.

*Recomputed state.* **Router block summaries are recomputed** for the new block grouping. **Page state
is replayed** in the new frame. A slot's img_page is the page its key belongs to in the Page Gate, −1
on every non-image row (`replay_page_isolation`, `:368-387`; the text splice writes
`(-1, *plan.iso)`, `:1274`). The query page applies only to rows that query in the new frame; copied
and spliced rows never do.

| Row class | Slot img_page | Query page | RoPE bucket (before base) | tok_pos |
|---|---|---|---|---|
| (a) tail marker | −1 | – | 1 | kept (copied) |
| (b) tail image rows | 0 | – | 1 | kept (copied) |
| (c) lookahead marker | −1 | – | 2 | kept (copied) |
| (c) lookahead images | 1 | – | 2 | kept (copied) |
| first newly prefilled marker (page lo+1) | −1 | 2 | 3 | 0 |
| (d) text tail | −1 | – | 1 (carry `mc + CONTEXT_PAGES`, `:409-437,1274`) | kept (copied) |
| forced `<PAGE>` (opens page lo) | −1 | 1 | 2 | `pf_cnt[2] = 1 + n_img(lo)` |

U2-D1a's first deliverable re-derives this table from the wheel and asserts it in unit tests. U2-D2b
checks it against the U2-D2a Python windowed oracle.

*Starting the decode.*
- **The sampled prefill token is discarded.** The first decode input is a **forced `<PAGE>`**. The
  wheel notes that copied KV otherwise re-predicts EOS (18 of 94 MPDocBench follow-ups).
- If the text-splice allocation fails, only the forced `<PAGE>` is spliced (`:1257-1285`).
- The forced tokens (text tail + `<PAGE>`) count as generated for the n-gram domain. They are hidden
  from output and do not consume the token budget.

*Decode seed.* A reused window carries its decode seed from the replay (`:433-437`), not from the HF
rule (§5.4, §5.5):
- RoPE carry: marker count M + 1, so the response bucket is 1;
- gate carry: mcum = pp + 1, so qp = 0;
- the forced `<PAGE>` then gets bucket 2, `tok = cnt[2] = 1 + n_img(lo)`, and qp = 1, matching the
  table above.

*Guard.* `context_pages = 1` and `end_on_marker = true`. `expected = #markers(prompt) − 1 − lookahead`,
i.e. the own pages. The forced `<PAGE>` counts as the first generated mark.

*One-page variant.* When window k+1 is the **last** window and has exactly one own page
(`p_new == 1`, `:260`), segment (c) is copied **except its final image token, which is re-prefilled**
(`copy_end = la_end − 1`, "straddle", `:284-287`). The source confirms why: the extend step needs at
least one token to run.
- `p_new` counts every page of the sent prompt, lookahead included, so a non-last window, even at
  W == 1, always copies (c) whole.
- The variant therefore happens when N mod W == 1, or for the last window when W == 1.

**5.10.5 Fallback.** Window k+1 is recomputed exactly like window 0, with page_base = lo, when:
- (i) there is no retained entry: window k was not stopped by EOS or a stop string, was aborted, was
  the last window, or was retracted or evicted (`:836-847,711-795`);
- (ii) the follow-up was retracted or already has output (`:956`);
- (iii) `next_page ≠ lo` (`:1049`);
- (iv) the layout mismatches, or `E − 128 − (E mod 32) ≤ marker index`, i.e. the tail page has fewer
  than 128 + (E mod 32) image tokens (`:256-268`);
- (v) there was no output at all, or `len(tail) + 2 ≥ max_new_tokens` (`:1069-1077`);
- (vi) KV allocation fails (`:1093`);
- (vii) a multimodal item straddles the copy boundary (`:1095`);
- (viii) the reuse planning raises; `try_match_prefix` catches it and recomputes (`:963-971`);
- (ix) a matched follow-up cannot be admitted while nothing runs, so its reuse is dropped after the
  prompt rewrite and the prompt is restored (`:792-795` → `_drop_reuse` `:684-691` →
  `_restore_prompt` `:660-681`). franken_ocr has no shared KV pool; this maps onto (vi).

A finished window whose prompt does not parse is never retained (`next_page_after` returns None,
`:884-885`), a case of (i). The `cache.page_size != bs` check (`:1047`) cannot fire, because the
model override forces page_size to the 32-row block
(`W/arg_groups/model_overrides/unlimited_ocr_v2.py:24-32`).

An empty last body is **not** a fallback (`:319-320`). A recomputed window has BOS and the instruction,
and no cross-window context.

**Two of these causes cannot happen with real v2 pages:**
- (iv)'s **short-tail branch** needs n_img < 128 + (E mod 32) ≤ 159, but every v2 page has ≥ 273
  image tokens (`MU:564`).
- (v)'s "no output" cannot happen with the guard on, because EOS is banned while marks < expected
  (`custom_logit_processor.py:256-257`, `MU:603`). So every window emits at least one `<PAGE>`.

Tests therefore inject fallbacks through a test hook that forces a reason: drop the retained entry,
mutate the layout, or simulate an allocation failure.

**A recomputed window as a reference.** Up to and including the `<PAGE>` that opens the lookahead page,
a recomputed window equals, token for token, an unwindowed run over the same pages + lookahead executed
with:
- page_base = lo (or lo even);
- expected = the own pages;
- end-on-marker;
- guard-first order.

For a recomputed **last** window (no lookahead), the reference runs with end_on_marker = false, so the
last-page `<PAGE>` ban applies (§5.10.7), and it is compared through EOS. U2-D2a derives this
configuration (page_base, expected marker count, end_on_marker, logit-processor order) from the wheel
source and records it in its notes.

HF `infer_multi` is **not** that reference for any window, whatever lo's parity:
- HF's expected count is all markers (`MU:1141`), so it keeps transcribing the lookahead page, while
  the wheel's excludes it and ends on the marker (`custom_logit_processor.py:227-237`);
- HF applies n-gram before the guard (`MU:1168-1172`);
- when lo is odd, the page parity differs as well (`serving_chat.py:512-515`, `KEY_PAGE_BASE: lo`;
  `unlimited_ocr_v2_backend.py:1008-1031`).

**5.10.6 Positions.** 1-D positions are window-local; they only index blocks and causality. There are
no absolute document positions, and the 131072 limit applies per window.

**5.10.7 Guard and caps.**
- The per-page cap counts tokens since the last `<PAGE>`, including the forced one.
- The wheel's per-window capacity is `min(ctx − 64 − per_page·(own+lookahead), own·max_page_tokens +
  256)`. The second term uses the effective cap and is omitted when the cap is ≤ 0
  (`doc_split.py:89-114`). franken_ocr has no global budget by default (D11).
- **The last-page `<PAGE>` ban (DISC-U2-3) is narrower than it looks.**
  `end_on_marker = (lookahead > 0) || reused` (`custom_logit_processor.py:227`), and `end_on_marker`
  short-circuits the ban (`:258-262`). Every non-last window has a lookahead, and every reused window
  sets `_uocr_end_on_marker = True` (`:1109`).
  - Neither is therefore **ever banned**. On a reused last window's last own page EOS is allowed, and
    an extra `<PAGE>` forces EOS and is cut or dropped as surplus.
  - The ban applies only to a last window that was recomputed (fallback), and to unsplit requests.
  - `_restore_prompt` resets the flag (`:669`).
- SGLang's guard can lag one step under overlap scheduling, so an extra token can precede the forced
  EOS (`:811-812`, `_cut_at_extra_marker`). franken_ocr has no overlap scheduling, and SGLang
  comparisons tolerate the lag (DISC-U2-8).
- **Wheel configuration differences.** franken_ocr follows the README configuration: cap 20,000 and the
  full guard. Two wheel behaviors outside it are recorded here and not replicated:
  - With server `SGLANG_UOCR_MAX_PAGE_TOKENS ≤ 0` (the env default is 0, `W/environ.py:1374`),
    lookahead and reuse windows get `page_stop_only` (`doc_split.py:315-322`,
    `custom_logit_processor.py:250-253`). Only "marks > expected → EOS" then applies: no EOS ban and no
    cap.
  - In SGLang, an n-gram `window_size ≤ 0` disables the ban (`custom_logit_processor.py:349`). HF
    maps it to 100 (`MU:1092`), and franken_ocr follows HF.

**5.10.8 Stitching** (`doc_split.py:251-275`).
- Split each window's visible text on `<PAGE>`. We split on token ids (D20).
- In the wheel, a recomputed window keeps non-blank text before its first `<PAGE>` as a separate body:
  `[head, *tail] if leads_with_body or head.strip()`. With front-anchored ownership that shifts every
  page by one and drops the last own page's body.
- **Deviation (DISC-U2-9):** franken_ocr applies U2-B6's rule wherever pages are stitched, recomputed
  windows included. A non-blank head is prepended to the first own page's body, with
  `warnings[{kind:"leading_text"}]`. So the same PDF aligns identically at N = 10 and N = 11.
- A reused window's visible text starts with the body of its first own page.
- Ownership is `bodies[ctx:input_pages]`. **All** surplus bodies beyond the own pages are dropped, and
  pages are concatenated in order.
- **Deviation (D5, DISC-U2-4):** we do not replicate `select_batch_output`'s drop of leading blank
  bodies while bodies are surplus,
  `while len(bodies) > input_pages and not bodies[0].strip(): bodies = bodies[1:]`
  (`doc_split.py:253-254`). It shifts alignment. The incremental `stream_owned_text` (`:260-275`) has
  no such drop; the streaming handler's end-of-window `select_batch_output` call
  (`serving_chat.py:1114-1124`) does, and stops sending text if ownership moves.

**5.10.9 Within-window retirement.** SGLang retires nothing within a window. franken_ocr's
block-granular retirement (§5.12) applies inside every window, under the same proof.

It must never free a row that §5.10.3 retains, and unconstrained retirement would:
- **(b)** The `<PAGE>` that opens the lookahead page moves qp past the tail page, so the image-tail
  blocks die at the final step.
- **(a)** When the tail marker at local row 1 + 274·j is ≡ 31 (mod 32), i.e. j ≡ 7 (mod 16), it ends a
  block holding only page j−1 images. That block dies once qp = j. Example: W = 8, window 0, row 1919.
- **(d)** Row q − 128 dies whenever it ends a block.

So every window with a successor sets a **pinned set**:
- the last own page's marker block and image-tail blocks;
- the lookahead rows;
- a one-row margin on generated-text blocks.

Retirement is also suspended on every guard-forced step (U2-B4 mechanism, U2-D1a sets it).

### 5.11 Known divergences and our choices

| ID | Divergence | HF | SGLang | franken_ocr |
|---|---|---|---|---|
| DISC-U2-1 | entmax domain | top-16 window | all candidates; bisection τ; ties can read fewer than 12 | Both (`EntmaxDomain`, D21); HF is the parity default; the long-document default is chosen on ground truth (U2-C5b) |
| DISC-U2-2 | n-gram vs guard order | n-gram first (can yield id 0) | guard first; a forced token skips n-gram | Forced token authoritative; assert never triggered on fixtures |
| DISC-U2-3 | `<PAGE>` ban on the last page | no ban | ban when not end_on_marker; end_on_marker holds whenever the window has a lookahead or is reused, so only a recomputed last window (or an unsplit request) is banned | HF when unwindowed, SGLang when windowed |
| DISC-U2-4 | leading blank bodies dropped while surplus, in stitching | n/a | `select_batch_output` drops them (`doc_split.py:253-254`; the non-streaming path and the streaming handler's end-of-window call), which shifts alignment | Never drop |
| DISC-U2-5 | numerics | GPU: Triton (bf16 q·scale and P; fp32 bias and softmax). CPU: SDPA in the activation dtype. Decode: fp32 over bf16. | bf16 Triton | f32 throughout; parity target is the f32 oracle |
| DISC-U2-6 | crop mode for multi-image requests | base (`infer_multi`) | gundam by default (`W/multimodal/processors/unlimited_ocr.py:23`) | Base by default; gundam document mode evaluated (OQ-V2-14) |
| DISC-U2-7 | page2d tok after an extra `<PAGE>` | reset to `pf_cnt[N]` (`SB:1486-1488`) | keeps the running count (`W/…/page2d_rope.py:579-591`) | HF; only reachable on a guard-forced EOS step |
| DISC-U2-8 | guard lag | none | can lag one step under overlap scheduling | None; SGLang comparisons tolerate it |
| DISC-U2-9 | non-blank text before the first `<PAGE>` (leading head) | `_uocr_save_results` drops it (`MU:1210`, `split('<PAGE>')[1:]`) | a recomputed window keeps it as body 0 with front-anchored ownership, which shifts pages and drops the last one (`doc_split.py:251-255`) | Prepend it to the first own page's body with `warnings[{kind:"leading_text"}]`, unwindowed and windowed alike (U2-B6, U2-D1a) |
| DISC-U2-10 | EXIF orientation | ignored: `Image.open(f).convert('RGB')` (`MU:1070`) | ignored: `smart_rgb_conversion = False` (`base_processor.py:215`), so a plain `convert("RGB")` | Applied, as for v1 (the shared decoder, `src/preprocess/mod.rs` `decode_path`). Only an image with a non-identity Orientation tag differs, and the oracle corpus has none (U2-B7) |
| DISC-U2-11 | FFN/expert weight storage in the default decoder | f32 parity oracle over the bf16 checkpoint | bf16 | The shipping cache keeps the 2,148 FFN/expert matrices per-channel int8 (the fixed recipe); attention, router, norms and `lm_head` stay f32. Its greedy ids leave the f32 oracle's at a near-tie, where the all-f32 cache (`FOCR_DECODE_F32`, otherwise the same ops) keeps the oracle's token: S1 page_0014 at step 47 (the oracle's token 0.214 below the int8-FFN pick), page_0590 at step 32 (0.156), S2 at step 44 (0.156), S3 p9_p14 at step 73 (0.065); page_0009 is exact. On the long lane the f32 cache is exact over S3-p10 (6,272 ids) and S3-p20 (16,880 ids), with block retirement on and off, while the int8-FFN path leaves at S3-p10 step 203 (0.017) and S3-p20 step 118 (0.056). It still ends at `stop` with one body per page, 6,277 and 17,165 ids long. No layer routes differently at any of them (U2-C2). U2-C4 measures the impact at L5 and files the entry |
| DISC-U2-12 | page state of a reused window's last prompt row when no text is spliced (the text KV allocation failed) | n/a (no windows) | the first decode step's carry is written into that row's slot (`_seed_continuation`, `W/session/uocr_doc_reuse.py:490-497`), so its image page reads −1 | The reused-window seed (`DecodePageState::reused_window`) carries the step and the row keeps its page. Both stacks read a row's page only through its 32-row block's (min, max), which skips −1, so they differ only when that row opens its block (rewritten prompt length ≡ 1 mod 32). franken_ocr's KV is never pool-allocated, so only the `marker_only_splice` test hook reaches the path; `w2-n5-markeronly-t12` (prompt 956 rows) is token-exact (U2-D1a) |
| DISC-U2-13 | rendering of the block grammar (`<BLOCK>label[<LEVELn>][box]<FORK>text`, §5.1) | `result.md` keeps every prefix raw: `re_match` (`MU:44-59`) knows only v1's spans | no postprocess (returns the generated text) | Each block prefix is a span under v1's rule (`postprocess.rs` `scan_block_spans`): an `image` block becomes `![](images/…)` and is a figure, every other prefix is deleted. The text so has v1's shape (titles unmarked, one newline between blocks, tables as HTML). `layout` keeps each block's label, its `level` and its box rescaled as v1's are. A prefix off the grammar, such as one cut off by a page's token cap, stays in the text. `cross_text`/`append_text` continuations stay on their own pages. Measured: all 431 blocks of the nine oracle outputs parse, and none of the grammar reaches the markdown (U2-E3b) |

**Filing.** `scripts/check_ledgers.py:310` matches only `^## DISC-\d+:`, so a `## DISC-U2-k:` heading
would escape the required-field, kill-switch and uniqueness checks.
- Each DISC-U2-k is therefore filed in `docs/DISCREPANCIES.md` under the next free numeric heading,
  titled `## DISC-NNN: U2-k — <title>`. The checker stays as it is and enforces the fields.
- The plan and the beads keep "DISC-U2-k" as the cross-reference name.
- U2-C4 writes or confirms the entries, with measured impact. U2-C1, C2, B2 and B6 supply numbers
  before C4 closes. U2-C5b, F4 and G1 run later: they append measured impact to the entry filed by C4,
  or file a new numeric entry, with `check_ledgers.py` green.

### 5.12 Block-granular retirement (franken_ocr optimization; exact under the guard)

**Rule.** In decode, after the first generated `<PAGE>` (qp ≥ 0), block c is **dead** once both of
these hold:
1. **No future non-open query can select it.** Either `bmin[c] = −1` (no image token: BOS,
   instruction-only and generated-text blocks), or `bmax[c] < qp` (all its image pages are behind the
   current query page, and qp never decreases).
2. **It lies wholly behind the local window**: the block's last row ≤ q − 128 (exclusive end
   ≤ q − 127), because the local window is (q−128, q] (`MD:472-473`). q only increases.

A dead block's K/V rows, index keys and k̄ may be freed. The **logical** block count, and with it
`M = min(16, nblk)`, is unchanged.

**Why this is exact.**
- **Non-open rows.** A dead block is never a candidate, so its x is −1e4. That affects neither τ* nor
  the read set (§5.6.5). A dead block is also never in the local window.
- **Open rows** read every remote block, but they only occur in three cases:
  - *qp < 0* (before the first generated `<PAGE>`). Retirement is disabled until then.
  - *No allowed block with qp ≥ 0.* That only happens when qp ≥ N, i.e. right after an extra `<PAGE>`.
    The guard (`mc > expected`) then restricts that step's logits to EOS, so its output does not
    depend on attention, and no later step exists.
  - *Allowed blocks exist but none is remote,* which falls back to all remote blocks. This is
    impossible in base and gundam: the current page's image blocks are prompt blocks, and at least 5 of
    its ≥ 8 full blocks lie ≥ 4 blocks behind any decode query.

**Why row-level retirement is wrong.** Page boundaries are not block-aligned: base markers sit at
1+274p, and 274 ≡ 18 mod 32. A selected block is read whole, and k̄ uses all 32 index keys. So a block
straddling pages p−1 and p (bmin = p−1, bmax = p) stays live while qp = p.

**Proof obligation and runtime check (U2-B4).**
- A detector asserts that no open row occurs while dead blocks exist, except on a step whose output is
  fully guard-forced.
- If it fires, the run records a ledger event and re-runs the current window without retirement. That
  is the deterministic fallback.
- Retirement is disabled whenever the guard is disabled.

**Effect.** Unwindowed decode holds only:
- the prompt blocks of the current and future pages;
- boundary blocks;
- the image-bearing last prompt block;
- the local window.

Generated text older than the local window costs nothing. This is v1's ring, generalized.

---

## 6. franken_ocr architecture for v2

### 6.1 Identity, loading and tokenizer (U2-A2, U2-A3, U2-A5, U2-A7)

**Current hazards** (code review, 2026-10-28). These places read "default arch" as "v1":
- **weights.rs:**
  - `:73-76` maps an absent `model_id` to `default_arch()`;
  - `:100` (license notice), `:898` (`Default for Weights`), `:2086`;
  - `:1238` tags raw safetensors as the default.
- **focrq.rs `:593-603`:** the writer omits `model_id` for the default.
- **mod.rs:**
  - `:1221-1224`, plus `:1231,1573,1699,2626,2663,2701,2851`;
  - `:1472` `validate_safetensors_header` always runs the v1 census;
  - `:1022` `versioned_quant_path` uses one global version;
  - `:2105/2137` `forward`/`forward_dynamic` fall through to v1 for unknown ids;
  - `:3268-3285` accepts the tokenizer.json sidecar without a hash check.
- **convert.rs:**
  - `:124,181`;
  - `:254` stamps the manifest `quant_recipe` only for the default;
  - `:292` (F16 guard);
  - `:392` builds the wasm int4 artifact only for the default;
  - 5 production `default_arch()` sites in all (`:124/181/254/292/392`); the other 14 grep hits are
    below `mod tests` (`:1011`).
- **Totals:** `default_arch()` has 48 grep hits: 23 production sites (convert.rs 5), the definition
  (`model_arch.rs:569`) and 24 test hits (including the `weights.rs:2086` test helper).
- **unlimited_ocr_census.rs:** `:110,366`.
- **cli.rs:**
  - `:130` (`--version` license);
  - `:686` (`convert --model-id` defaults to `unlimited-ocr`);
  - `:1686` (auto-download is v1 only);
  - `:2350` (convert's pinned v1 source-SHA check) and `:2420` (the receipt's `quant_recipe`);
  - `:2675-2711` (models `guidance` map `:2675`; the ID column is `{:<14}` at `:2684,2711`, but
    `unlimited-ocr-2` is 15 chars);
  - `:2949` (`health` reports one model path);
  - `:3036` (v1-only pull hint);
  - `:4108`.
- **Elsewhere:**
  - `robot.rs:109` schema (`FOCR_MODEL_LICENSE_NOTICE`); health is `cli.rs:2949`;
  - `doctor.rs:121` (default model only), with flat directory scans in `detect_stale_focrq_format`
    (`:142`) and `detect_unreadable_cache_entries` (`:186`); orphan scanning already recurses (`:273`);
  - `storage.rs:54` (the `model_version_tag` field; `RunStore` `:68`, `FOCR_RUN_STORE` `:34`);
  - `simd/dispatch.rs:636` (per-model rollup ids);
  - tests `cli_robot_golden.rs:1143`, the exact `robot selftest` per-model rollup list
    `["unlimited-ocr","got-ocr2","smolvlm2","onechart"]`, fed by `simd/dispatch.rs:636`, which changes
    if v2 joins it; and `:2141,2162`, the `focr models` `models[0]` checks.

**Changes.**

*Identity.*
- Add `model_arch::UNLIMITED_OCR_V1_ID = "unlimited-ocr"`. An absent `model_id` maps to it,
  permanently.
- The writer always emits `model_id` for non-v1 archs and keeps omitting it for v1, so v1 artifacts
  stay byte-identical.
- Raw safetensors identify their arch from the census: v1 census → v1, v2 census → v2, neither →
  refuse. `validate_safetensors_header` branches per arch.

*Capability methods on `ModelArch`* replace identity checks:
- `tensor_census()`. It is not called `census()`, which `Weights::census(expected_names)`
  (`weights.rs:1253`) already names;
- `required_quant_recipe(quant)`. This **moves** the free function `dist::required_quant_recipe(model_id,
  quant)` (`dist.rs:485-495`) into the arch, and updates its callers (`dist.rs:500/504`,
  `cli.rs:2666`). No wrapper, no second table;
- the existing `decode_contract()`, extended;
- `unlimited_variant() -> Option<UnlimitedVariant>`, with `UnlimitedVariant::{V1Rswa, V2SrSwa}` defined
  in `model_arch.rs` by U2-A2;
- `artifact_version()`;
- `required_token_ids()`;
- `tokenizer_sha256()`. No runtime tokenizer hash exists today: v1's `a02f8fd5…55a8b4` appears only in
  tests and scripts, and is pinned here as v1's value;
- the existing `license_notice()` (`model_arch.rs:151`) and `quant_policy()` (`:161`). Note that two
  `QuantPolicy` types already exist (`model_arch.rs:118` struct, `quant/recipe.rs:54` enum).

*Tokenizer sidecar.* Per-model cache subdirectories hold their own `tokenizer.json`. The loader
verifies `tokenizer_sha256()` and `required_token_ids()`. Today `validate_pinned_ids`
(`tokenizer/mod.rs:489-532`) only validates tokens that are present, so a v1 tokenizer would be
silently accepted for v2.

*Fail-closed dispatch (U2-A7, D18).*
- `forward` and `forward_dynamic` match exhaustively on decoder family plus variant.
- Every lever in D18 is classified in `UOCR2_LEVER_POLICY` and checked before weights load. A refusal is
  a new pre-load check returning `FocrError::Usage` (exit 2). `ensure_arch_implemented` (`mod.rs:2376`)
  returns exit 1 after load, so it is not reused; other zoo archs keep their exit codes.
- A census test fails on any `FOCR_*` variable missing from the table. It also enumerates the dynamic
  `FOCR_STAGE_BUDGET_{stage}_MS` (`lib.rs:811`), and covers `FOCR_MAX_NEW_TOKENS` and
  `FOCR_NO_REPEAT_NGRAM` (redefined per page for v2). Each bead that adds a `FOCR_*` variable adds its
  row; debug and test-only variables go on an allow-list.
- bd-3ryl (task-vs-arch guard) is the general version; A7 implements the v2 cases and links it.

*Tests.*
- A v1 artifact without `model_id` loads as v1 under a **test-only registry whose default is v2**.
- A v2 artifact without `model_id` is impossible to write.
- Raw safetensors are arch-detected by census.
- A v1 tokenizer beside a v2 artifact is refused. The recovery names `focr pull <model>` for archs with a manifest entry. Before U2-E4 publishes v2, it names the pinned HF `tokenizer.json` (commit and SHA-256) instead.
- A v2 artifact under every lever reports "supported" or "refused", never "ran v1".

**Sharded sources (U2-A3, D16).**
- `--model` or `FOCR_MODEL_PATH` pointing at a directory with `model.safetensors.index.json` resolves
  every shard.
  - Each shard's header and the census are checked on every load.
  - Shards are owned or mapped per `mmap_requested()` (`weights.rs:873-881`: on by default on iOS;
    `FOCR_MMAP=0` forces owned).
- `RAW_SAFETENSORS_SHARD_NAME` (`mod.rs:333,857`) becomes a per-arch shard list.
- `focr convert <dir>` streams shard by shard instead of reading one file fully into memory
  (`std::fs::read` at `cli.rs:2332`). It computes full shard hashes and stamps the multi-shard
  `source_sha256` convention.
- Convert checks the census against `--model-id` before the pinned v1 SHA check (`cli.rs:2350`). A
  mismatch is `Usage` (exit 2) naming both ids. `ConvertArgs.model_id` (`cli.rs:686`,
  `default_value = "unlimited-ocr"`) becomes `Option<String>`.
- The I64 `position_ids` is verified and skipped at parse time (D8).

**v2 census (U2-A5, D22).** It is an embedded **delta**, `unlimited_ocr_2_manifest.json` (a new data
file), containing:
- the 109 added tensors with names, shapes and dtypes;
- the SHA-256 of the resulting 2,819-entry list;
- full-tensor SHA-256s for the byte-identical vision tensors;
- pins: HF commit `5632329a…`, both shard SHA-256s and sizes, the tokenizer.json SHA-256, and the
  verified-artifact SHA (filled at C3).

`unlimited_ocr_census.rs` is generalized to hold the v1 census plus the v2 delta. There is no second
census module.

### 6.2 Decoder structure (U2-B1…B6)

| Piece | Location | Notes |
|---|---|---|
| Dimension constants | `decoder::config`, `moe::config`, `rswa.rs` | Unchanged (D3) |
| Variant enum | `UnlimitedVariant` | Chosen once from the arch |
| Layer forward | Attention-kind enum: `LayerAttention::Rswa(&mut RingCache)` / `LayerAttention::SrSwa(&mut SrSwaLayerCache, &SrSwaLayerWeights, &PageCoords)` | The attention call gains access to the normalized hidden state `h` (the router input) and the page coordinates, in **three places**: `layer_forward`/`forward` (stateless, `&Weights`; `decoder.rs:507/747`); the monolithic branch of `prefill_with_cache_chunked` (`decoder.rs:1374-1400`, which mirrors `layer_forward` inline); and `decode_step_with_cache` (`decoder.rs:1428-1502`). `prefill_with_cache` always delegates to `prefill_with_cache_chunked` (`:1276`). MoE, FFN, norms and `lm_head` code is shared verbatim. |
| New module `src/native_engine/srswa.rs` | Genuinely new functionality, like `rswa.rs` | Contents listed below |
| `RopeTable` | Gains `RopeTable::from_angles` | page2d tables reuse the apply kernel (`decoder.rs:238`) |
| Weight caches | `DecoderWeightCache` (`decoder.rs:812`), `DecoderWeightCacheI8` (`:2326`) | Gain the per-layer router weights, widened once from bf16 to f32 |
| Shared `decoder::prefill_attention` (`:577`) | Also used by `decoder_qwen2.rs:1266,1270` (GOT, SmolVLM2, OneChart) | **Not modified.** v2 prefill attention lives in `srswa.rs`. |
| v2 generate loop | `mod.rs`, next to the v1 loops | Contents listed below |

`srswa.rs` contains:
- `PageCoords` and the decode state machine, including `page_base` (§5.4–5.5, §5.10.4);
- `Page2dRope` tables;
- `SrSwaLayerWeights`;
- `SrSwaLayerCache`: K/V, index keys, `kbar`, `bmin`/`bmax`, page state, and a dead-block set with
  the logical `nblk`;
- the router (§5.6): `project`, `summaries`, `scores`, `page_gate`, `entmax15` (`TopM` and
  `AllCandidates`), `select_bias`;
- attention (§5.7), with a dense scalar oracle and a gathered fast path;
- prefill and decode entry points;
- `retire_dead_blocks` with the open-row detector (§5.12);
- `splice_from(prev_tail)` (§5.10.4);
- `serialize_tail` and `deserialize_tail` (D14).

The v2 generate loop contains:
- the prompt builder (§5.1);
- the guard (§5.9);
- the generated-only n-gram domain (D10);
- per-page budgets (D11);
- token-id page segmentation with a UTF-8-safe incremental decoder (D20);
- EOS handling and length limits.

It reuses the domain-agnostic per-token sampling primitives from `sampler.rs`.

**Path matrix (D18).**

| Path / lever | v2 first wave | Later |
|---|---|---|
| Conservative cached prefill/decode (`generate_cached` → `prefill_with_cache`, `decode_step_with_cache`; int8 FFN, high-precision attention/router/`lm_head`) | **Support**. The certified artifact runs here, and so does a raw BF16 checkpoint: `DecoderWeightCache::build` (`decoder.rs:837`) int8-quantizes the 2,148 FFN/expert matrices even from BF16 (`quant_ffn_loaded` → `quant_oc_loaded`, `decoder.rs:1928` → `:1838`). There is **no all-f32 cached path**; the comment at `mod.rs:3765-3770` saying otherwise is stale, and U2-B5 fixes it. v1's own L4 rung builds this cache (`tests/parity_ladder.rs:788`). (`mod.rs:3736-3800`) | — |
| int8 cached (`generate_cached_i8` → `prefill_with_cache_i8`, `decode_step_with_cache_i8`, `decoder.rs:2826,2956`), reached only by the experimental full-int8 switch (`FOCR_DECODE_INT8`, `ocr-batch --experimental-full-int8`, `force_int8_decode()`) and the WASM int4 artifact | Refuse | Backlog |
| `OcrModel::recognize_multi_page*` (`mod.rs:2620/2654/2691`; arch guards at `:2626/2663/2701` and `ensure_arch_implemented` at `:2634/2671/2709`) | Refuse until U2-B6, then route to the v2 document loop. Never v1's 640-squash loop. | U2-E2 replaces them |
| Stateless f32 path (`generate_stateless` → `decoder::forward(&Weights)`, `decoder.rs:699`; `mod.rs:4246/4278`; `FOCR_DECODE_STATELESS`) | **Support**. The only all-f32 native path: the f32-oracle rungs (L2 routing, L3, L4) run here (§7.2) | — |
| Chunked prefill (`decoder.rs:1198` `chunk_prefill_attention`, `:1287` `prefill_with_cache_chunked`, `:2842` `prefill_with_cache_i8_chunked`; `FOCR_PREFILL_CHUNK`, presence-based: any value arms it) | Refuse when `prefill_chunk_size()` is `Some` | F2 |
| Batched / spine (`batched_decode_step_i8*` `decoder.rs:3211,3243`; `recognize_batch_spine` `mod.rs:2984`; `batch_scheduler.rs:43,440,487`) | Refuse | F5 |
| Spec decode / verify (`spec_decode_i8` `mod.rs:4064`, reachable only via `generate_cached_i8` `:3887`; `verify_forward*` `decoder.rs:3495,3575`, no production caller) | Unreachable from the v2 paths: the `_i8` path is refused, and `FOCR_SPEC_DECODE`/`FOCR_SPEC_VERIFY` are ignored (row below) | Backlog |
| `FOCR_LMHEAD_SHARD`, `FOCR_LMHEAD_SHARD_TILES` (presence-based, `decoder.rs:2098,2105,2115`; reach `lm_head_cached`, `decoder.rs:1088-1090`, on the conservative path) | Refuse | Supported-exact since U2-C2 (L4 with the lever on, §6.2) |
| `FOCR_INT8_KV`, `FOCR_ATTN_GEMM`, `FOCR_RSWA_PARALLEL_ATTN` (read only in `rswa::decode_attention`/`RingCache::new`, which the SR-SWA arm never calls); `FOCR_QKV_FUSED` (default ON, `_i8` only); `FOCR_FUSE_NORM_QUANT` (`decoder.rs:2990`, `_i8` only); `FOCR_FUSE_NGRAM_LMHEAD`, `FOCR_SPEC_DECODE` (`generate_cached_i8` only, `mod.rs:3939,3949`); `FOCR_SPEC_VERIFY` (no production reader); `FOCR_INT8_ATTN`, `FOCR_INT8_LMHEAD` (full-int8 recipe only, `mod.rs:480`); `FOCR_RESAMPLE` | Ignored, with `warnings[{kind:"ignored_lever"}]` | — |
| `RunawayGuard`, `FOCR_RUNAWAY_GUARD` (`sampler.rs:53`) | Ignored, **once U2-B5/B6 stop building it on the v2 arm**: today it is built for every path (`mod.rs:3727`) and checked in `generate_cached` (`:3830`) and `generate_stateless` (`:4284`). The page guard plus the per-page cap replace it | — |
| Resident daemon (`cli.rs:1650`, `resident.rs`) | Refuse for v2 until its protocol carries the v2 flags | Backlog (bd-fuh2) |

Parity tests that must gain v2 lanes or explicit v2 skips:
- `batched_attention_parity`;
- `batched_forward_parity`;
- `batched_decode_bit_identity`;
- `batched_moe_parity`;
- `batched_sampler_parity`;
- `batched_igemm_parity`;
- `chunked_prefill_parity`;
- `int4_packed_parity`;
- `lmhead_shard_parity`;
- `parity_ladder` (`:130`);
- `spec_decode_gate`;
- `spec_ring_rollback`;
- `spec_verify_forward_parity`.

**v1 non-regression.** Every v1 token-id golden stays byte-identical (U2-T3 gate), and no v1 code path
observes v2 code.

### 6.3 Memory model (whole process)

**Peak RSS** = artifact bytes (~4.17 GB owned; less with `FOCR_MMAP=1`) + hydrated vision caches +
widened decoder weights + rasters and preprocessed tensors of the pages in flight + KV.

**KV per token** (12 layers):
- K+V f32: 12 × 2 × 1280 × 4 B = **120 KiB**;
- index keys: 12 × 128 × 4 B = **6 KiB**;
- block summaries: negligible.

Worked examples assume about 1,000 output tokens per page. They are analytic estimates; U2-F1
measures the real numbers.

| Scenario | KV rows resident | f32 KV without retirement | f32 KV with block retirement (D9) |
|---|---|---|---|
| Single page, base | ~280 prompt + ~1,000 out | ~0.15 GiB | ~0.05 GiB (prompt + local window) |
| 20 pages unwindowed | 20×274 + 20,000 ≈ 25.5k | ~3.0 GiB | ≤ ~0.65 GiB at start (all future prompt pages), falling as pages pass |
| W=10 window | 11×274 ≈ 3k prompt + 10k out | ~1.5 GiB | ≤ ~0.35 GiB |
| v1 multi-page, 20 pages (comparison) | 20×111 + ring 128 | ~0.27 GiB | — |

- **Rasters.** With the streaming page source (D19, U2-D0), only W+1 page rasters are in flight, about
  25 MB each at 300 dpi letter. Today `--multi-page` holds all of them, twice.
- **Estimator.** U2-D3 adds a whole-process memory-budget estimator. `--page-window auto` uses it, and
  robot `window` events report it.

### 6.4 Concurrency (doctrine #5)

- One live forward at a time.
- Windows run strictly sequentially, and per-page vision encoding is sequential within a window.
- The page source may rasterize the next window's pages on the same thread between windows, never
  concurrently with a forward.
- All parallelism is intra-op via the kernel rayon pool: heads × query chunks in prefill attention,
  the GEMMs, and vision.
- No rayon under a held lock, and no nested asupersync runtime.
- Checkpoint I/O happens between windows, never inside a forward.
- `tests/many_pages_without_deadlock.rs` has no multi-page scenario today. U2-T2 adds windowed v2
  document scenarios.

### 6.5 Quantization and artifact (U2-C3)

**Recipe.** `unlimited-ocr-2-ffn-int8-attn-bf16-lmhead-bf16-v1`, with its id constant next to the
existing recipe ids (`src/quant/convert.rs:46/60`) and its tensor predicates in `src/quant/recipe.rs`.
- **int8:** exactly the 2,148 FFN/expert matrices:
  - `model.layers.0.mlp.{gate,up,down}_proj`;
  - `model.layers.{1..11}.mlp.experts.{0..63}.{gate,up,down}_proj`;
  - `model.layers.{1..11}.mlp.shared_experts.{gate,up,down}_proj`.
- **High precision** (BF16 source): everything else, including `self_attn.dash_router.*`,
  `self_attn.indexer.*` and `mlp.gate.weight`.
- **Matcher.** `is_decoder_attn_proj` (`recipe.rs:195`) is narrowed to the exact
  `self_attn.{q,k,v,o}_proj.weight` names. A test asserts that under every attention or `lm_head`
  lever, the v2 int8 set contains 48 attention projections when the lever is on, and **0**
  router/indexer tensors.
- **Converter path.** The non-default path (`convert.rs` ~124-178) must use this explicit recipe, not
  its "every `self_attn` projection" default.
- The I64 `position_ids` is verified to equal `arange(257)` and skipped (D8).

**Counts.** int8 = 2,148. High precision = 2,819 − 2,148 − 1 = 670. The i32-accumulation proof at
K=6848 is unchanged.

**WASM.** The v1 int4 recipe (`src/quant/convert.rs:60`,
`unlimited-ocr-wasm-experts-int4-attn-int8-lmhead-int8-v1`) and its loader branches refuse v2.
`validate_unlimited_ocr_quant_records` returns Ok for non-default ids at `mod.rs:1699`, so v2 never
reaches `:1710`. The refusal must sit before that early return, in the resolver probe
(`mod.rs:1105-1153`), and in `artifact_stores_gated_int8` (`:3128`).

**Loader gate.** It rejects any v2 `.focrq` whose dtypes violate the recipe (a quantized router, a wrong
recipe id), or whose `source_sha256` does not match the multi-shard convention (D16). A refusal is
`FocrError::FormatMismatch` → **exit 7**, naming the tensor, as v1 does (`mod.rs:1757`).

**Artifact.**
- `unlimited-ocr-2.v<ver>.int8.focrq`, about 4.17 GB (estimate: v1's 4,157,448,783 B plus ~16 MB of
  router/indexer).
- The per-model version (D22) is resolved by `versioned_quant_path`.
- It installs under `~/.cache/franken_ocr/models/unlimited-ocr-2/` with its own `tokenizer.json`.
- It is published as release parts under 2 GiB.
- The HF mirror is per bd-3p8y/bd-aio5. GitHub release parts are primary.

### 6.6 Document pipeline (U2-D0, U2-E2, U2-E3)

**Page sources.** The PDF page source wraps the lazy primitive `PdfPages::open`/`render(idx)`
(`pdf.rs:89/144`). `pdf::walk_document` (`pdf.rs:1032`) is push-style (render, then recognize), only
pdf.rs tests call it, and bd-3gyt re-expresses it over the page source. U2-D0 implements the
multi-page part and coordinates with bd-3gyt and bd-paom (`related` links, not blockers).
- The page source is **lazy**: it yields a page's raster on demand, plus per-page `(width, height)`,
  and `PageSource::fetch` re-renders a page for figure crops.
- It must be `Send + 'static`: the whole document runs inside one `run_blocking_stage_with_budget`
  (`lib.rs:765-770`), not on the orchestrating thread.
- It records a source-page map entry for every page, including skipped pages, spread halves and
  `--pages` selections. Skips reuse or extend the existing `pdf::SkippedPage { page, reason }`
  (`pdf.rs:960`).
- The memory fix replaces the eager `Vec` plus `images.clone()` (`cli.rs:1862`). Unrasterizable pages
  are already skipped today (D24); the source-page map fixes the numbering of decoded pages after a
  skip.

```
PageSource (lazy) ─▶ window planner (§5.10.1) ─▶ per window:
    rasterize + preprocess own+lookahead pages ─▶ vision per page ─▶ drop rasters
    ─▶ prefill (or splice + prefill) ─▶ decode with guard ─▶ per-page bodies (token-id split)
    ─▶ sink(PageResult{source_page, text, layout, figures}) ─▶ checkpoint (D14)
```

**Driver and sinks.** `recognize_document` (D23) drives the pipeline. The CLI is a sink: markdown,
JSON `pages[]`, figures, or robot events.

**JSON pages.** `--multi-page --json` returned `pages: []` (`cli.rs:1911`) until U2-E3 (§9.3).
- With v2 it returns real pages.
- With v1 it returns real pages from `<PAGE>` splitting only when the body count is exactly N
  (U2-X2's shared body-count function). Otherwise it keeps `pages: []` (type-stable), and the
  `v1_page_merge` warning explains why. All three recorded v1 fixtures merge pages (p9_p14: 1 body for
  2 pages; p10: 9 of 10; p20: 7 of 20), so the exactly-N case is exercised with a synthetic string.
- The existing JSON keys stay: `pages[]` entries `{page, layout, half}` (`half` absent when unsplit)
  and the top-level `figures[]` of `{label, page, bbox, path}` (`cli.rs:961-1000`). U2-E3 adds
  `run_index`, `text` and the other `PageResult` fields.

---

## 7. Verification plan

### 7.1 Oracle (U2-A6)

**Primary oracle: HF v2 on CPU in f32.**
- **Environment.** A venv with python 3.10, torch 2.13.0, transformers 4.52.1, tokenizers 0.21.4,
  addict, einops and easydict, **never shared with the v1 oracle**. Model at commit `5632329a`.
- **Shims.** Mirror v1's `--device cpu --cpu-patch --f32` shims (`scripts/gen_reference_fixtures.py`:
  flags `:225-245`, `_install_cpu_patches` `:292-321`, `_install_f32_patches` `:324-347`). `.cuda()` and
  autocast become no-ops, and the whole pipeline computes in float32. That includes the image casts:
  `MU:561,566` call `transform(...).to(torch.bfloat16)`, which v1's existing `Tensor.to` wrapper
  rewrites to float32 (`:338-347`). Without it the L0 dumps would be bf16-rounded. The bf16 zero crop
  placeholder at `MU:579` is not rewritten and is unused in base mode.
- **DASH paths.** Prefill takes `_forward_dense` (SDPA) on CPU, and decode takes `_decode_attend`
  (§5.7). `DASH_STAGE0` stays unset; `TIME_DIAG` and `SPARSE_DIAG_LOG` stay off.
- **Step 1 of A6 is feasibility.** If any DASH path is CUDA-only, the oracle script ports its dense
  fallback (a few dozen lines against §5.6–5.7). The port is verified once against GPU bf16 on S1 if a
  GPU is available; otherwise, against the unpatched dense path in bf16 on CPU.

**Secondary oracle (cross-precision ledger):** the same runs in bf16, on CPU and on GPU if available.

**Script.** `scripts/gen_reference_fixtures_uocr2.py`; per-model oracle scripts are existing
precedent.
- It drives `model.infer()` / `infer_multi()` with explicit `no_repeat_ngram_size=35`,
  `ngram_window=128|1024`, `max_page_tokens=20000`, `page_count_guard=True` and an explicit
  `crop_mode`.
- Dumps are taken through forward hooks:
  - prompt ids and preprocessed tensors;
  - per-layer input hidden states;
  - per layer, for selected query rows **including fallback queries**: router `iq`/`ik` (post-RoPE),
    `kbar`, block scores, candidate mask, entmax prior, and topi/bias/valid;
  - per-layer attention output;
  - final hidden state, last-position logits and generated ids;
  - per-step dumps for the first 64 decode steps and for steps around every `<PAGE>`.

**Fixture sets:**

| Set | Contents |
|---|---|
| S1 | 1 page, base (`crop_mode=False`): 3 pages from the 20-page corpus, including the hard-EOS page |
| S2 | 3-page document, base multi |
| S3 | The existing multi-page fixtures (p9_p14, p10, p20) under v2 |
| S4 | 1 page, gundam (`crop_mode=True`) |
| S5 | One 40–60 page document with ground truth (a MarathonDocBench subset, Apache-2.0), unwindowed. CPU f32 if the time is tolerable; otherwise bf16 on GPU, as ledger only. |
| S1-corpus | v2 f32 CPU single-page base references for all 20 corpus pages (U2-C4's gate input), plus **v1 f32** CPU references for the same 20 pages from the existing `native_f32`/`--cpu-patch` generator, so C4's baseline is symmetric. Not an M0 exit item: U2-C4 produces it with U2-A6's script if absent and records the commands in `tests/fixtures/PROVENANCE.md`. U2-G1 generates the 20-page gundam set with the same script (`crop_mode=True`). |

**Nondeterminism floor.** Run each fixture twice, at two thread counts. Record bitwise token equality
and the max hidden-state deltas. These set the tolerances (AGENTS testing policy).

### 7.2 Parity ladder rungs for v2 (U2-C1, U2-C2)

| Rung | What | Gate |
|---|---|---|
| L0 | Preprocessing tensors; prompt ids (incl. `<PAGE>`=128827); image-token counts | exact (pil-bicubic) |
| L1 | page2d coordinates and cos/sin; Gemma norm; `kbar`; scores; Page Gate; entmax; selection; bias | coordinates exact; floats within the f32 floor |
| L2 | Per-layer hidden states; **routing equality** (topi/valid exact vs the f32 oracle on S1–S3, **including fallback queries**); attention outputs | cosine ≥ the floor-derived threshold. Against bf16, mismatches fall in the tie-flip class, which has a ceiling. |
| L3 | Last-position logits | within the measured f32 tolerance |
| L4 | Generated tokens | exact vs the f32 oracle on S1–S3 |
| L5 | Corpus CER (§7.3) | budgets |

**Which native path each rung runs on** (§6.2):
- The f32-oracle rungs (L2 routing equality, L3, L4) run on the **stateless f32 path**: `generate_stateless`
  → `decoder::forward(&Weights)`, over raw BF16 widened to f32. It is the only all-f32 native path.
- The **shipping cached path** (`generate_cached` over `DecoderWeightCache`: int8 FFN/experts, f32
  attention/router/`lm_head`) also runs L4 on S1–S3, and token-exactness is expected there, as on v1's
  L4 rung, which builds the same cache.
  - Any mismatch blocks M1 unless it is root-caused, from the first-mismatch step and the trace diff,
    to the int8 FFN quant effect.
  - A root-caused mismatch is recorded as a DISC entry, and the path is certified at L5 (U2-C4).

**Routing-margin diagnostics.** For every selection mismatch, log the score margin between the K'-th
and (K'+1)-th candidates and the entmax support size. A mismatch below the measured float noise is a
"tie-flip": a discrepancy class with a ceiling, not a free pass.

### 7.3 Corpus gates (U2-C4)

**Single page.** The 20-page corpus, v2 base mode, comparing the v2 int8 artifact against the v2 f32
HF reference.
- **Gate shape per `docs/conformance/RATCHET.md` §4.2.** Graded metrics enter as thresholded
  pass/fail, never as raw means. Two facts shape the gate:
  - 0.25 is v1's **corpus-mean** budget. v1 passes at 0.193, yet page_0590 alone scores 0.6165
    (`docs/DISCREPANCIES.md:158-161`), so a per-page 0.25 would fail v1 itself.
  - With n = 20, every category below `MIN_CALIBRATION_N` falls back to a point estimate
    (`src/conformance.rs:834-843`), and a new category is admissible by definition (RATCHET.md §2).
    A v2 lane alone can never fail.
- **The gate (U2-C4; all required):**
  1. a per-page CER budget pre-registered before the run (proposed: 0.45 vs the v2 f32 reference);
  2. v1 int8 vs the **v1 f32** reference (U2-A6's S1-corpus row), run through the same harness, in the
     same invocation, as the baseline. Both lanes use the same reference precision; against a BF16
     reference, v1's drift would be inflated and v2's admission made easier;
  3. `ratchet_decide(v1 bounds, v2 bounds)` admits category `all`;
  4. every per-category fallback card is ≥ v1's point estimate;
  5. aggregate normalized CER ≤ 0.25 is a hard gate, reported next to v1's 0.193;
  6. EOS on every hard page (below).
- C4 also captures v2 goldens, which become a v2 lane in U2-T3's gate for every later v2 change.
- Hard pages must emit EOS. Under the guard, EOS is reachable only after `<PAGE>`.

**Multi-page.** S2 and S3 against the v2 reference: per-document CER, page-count equality, and the
per-page generated-token distribution (which feeds D11 and the re-derived pre-windowing page limit).

Every accepted divergence is recorded in `docs/DISCREPANCIES.md` with the artifact SHA, under a numeric
`## DISC-NNN: U2-k — …` heading (§5.11). C4 also declares `tests/fixtures/uocr2_v2_goldens/` in
`tests/fixtures/MANIFEST.toml` (which `scripts/check_fixture_manifest.py` requires) and in
`tests/fixtures/PROVENANCE.md`. The upstream-drift check (R20) is not part of C4; U2-C2 runs it at M1 and
U2-E4 before publishing.

### 7.4 Property and metamorphic tests (acceptance criteria of their owning beads)

| Property | Bead |
|---|---|
| entmax-1.5 (both domains): sums to 1; support ≥ 1; monotone; shift-invariant; matches brute-force τ; top-M with merged padding ≡ full top-M, including rows whose max candidate x < −1e4+2 (§5.6.5) | B2 |
| Page coordinates: bijective (bucket, tok) per sequence and decode state machine ≡ prefill builder on teacher-forced sequences with ≤ N generated markers (DISC-U2-7 tested separately); reused-window seed (§5.5); page_base parity invariant for every row with local_bucket > 0 (§5.10.4) | B1 |
| Retirement: retired vs unretired runs give exact equality on S1–S3 fixtures; the detector fires on a synthetic extra-marker sequence; the pinned set survives (W=8 window 0, marker row 1919 ≡ 31 mod 32); no retirement on guard-forced steps | B4 (mechanism, synthetic), C2 (S1–S3 fixtures), D1a (pinned set) |
| Guard: never more than N markers; no EOS before N; the cap forces correctly; forced tokens are authoritative | B6 |
| **Re-prefill ≡ incremental** on the conservative cached path (both §5.8 preconditions), retirement on and off (retired ≡ unretired); stateless-vs-cached agreement is L4's gate (§7.2), with its DISC escape | C2 |
| Planner partition/lookahead/budget (own pages ≤ the page limit); splice copies equal their sources; reused-window seed and replayed page state equal the §5.10.4 table; one-page variant; stitching ownership, leading-text prepend (DISC-U2-9) and no leading-blank drop; fallbacks injected through a test hook | D1a |
| Windowed pipeline: one vision encode per page (fallback recomputes counted separately); rasters dropped after their window; documents of page limit + 1 pages run windowed instead of being refused | D1b |
| An unwindowed v2 document of page limit + 1 pages is refused with exit 2 by default (an explicit `Some(PageWindow::Off)` bypasses it); a position-limit stop writes the completed pages and ends in `run_error` with `finish_reason: "length"` and exit 5, never silent truncation | B6 |
| v1 multi-page reports what happened: `v1_truncated` at the clamp (via `FOCR_TEST_MAX_POSITIONS`), `v1_page_merge` on every recorded fixture (p9_p14 gives 1 body for 2 pages), a synthetic exactly-N string gives no warning; exit 0 and v1 text unchanged; every emitted robot event validates against `event_fields` | X2 |
| Added-token splitting is idempotent; `<PAGE>` round-trips; fuzz over interleavings | A4 |
| UTF-8-safe incremental decode across split CJK characters and page boundaries | X1 |

### 7.5 Windowed-mode validation (U2-D2a oracle, U2-D2b gate)

1. **Splice parity.** `gen_reference_fixtures_uocr2.py --windowed W` (U2-D2a) implements the wheel's
   windowing on the HF model's DASH cache. It slices rows (a)–(d), replays window k+1's page state in
   the new frame, and forces `<PAGE>`.
   - It is **ported function by function** from `W/session/uocr_doc_reuse.py`,
     `W/entrypoints/openai/doc_split.py` and `W/layers/attention/unlimited_ocr_v2_backend.py:1001-1031`,
     by an author other than U2-D1a's implementer, from the pinned wheel's source files only (U2-A1
     pins the wheel by SHA-256). That author never reads the U2-A1 truth-pack notes or the SGLang spec
     comment (they are the first reading), and does not read §5.10 or U2-D1a until the oracle's
     self-tests pass, so it is an independent reading, not a restatement of ours.
   - U2-D2a depends only on A1 and A6. U2-D1a depends on U2-D2a, so the oracle exists before the Rust
     windowed core is written.
   - U2-D2b uses it to gate U2-D1a token-exactly (f32) at W=2 and W=3 on 5–9-page documents, which
     gives 1–4 splices each. One case must have N mod W == 1, to exercise the one-page variant.
   - Generated ids are compared exactly. Bodies are compared after applying the leading-text rule
     (DISC-U2-9) to the oracle's ids.
   - Fallback-recompute windows are forced through a test hook (natural fallbacks (iv) and (v) cannot
     happen with real v2 pages, §5.10.5). Up to and including the `<PAGE>` that opens the lookahead page,
     each must equal an unwindowed run over the same pages + lookahead, configured the way the wheel
     configures a recomputed window: page_base, the expected marker count, end_on_marker and the
     logit-processor order are derived from the wheel source and recorded in the oracle's notes. A
     recomputed **last** window (no lookahead) runs with end_on_marker = false and is compared through
     EOS. HF `infer_multi` cannot serve as that reference, because its page parity, expected count and
     processor order follow an unwindowed request.
   - If a GPU host is available, it is compared once against the SGLang wheel. That comparison is
     informational, because of DISC-U2-1/5/8.
2. **Invariants.**
   - exactly N page bodies (minus skipped pages);
   - order preserved;
   - no lookahead page transcribed;
   - resume byte-identical (D14);
   - peak RSS within the window plan.

   W ≥ N is unwindowed by definition, so that case is a smoke test only. In the oracle's degenerate
   case (W ≥ N), the windowed oracle, run through its own loop (one window, no lookahead; never by
   delegating to `infer_multi`), equals the unwindowed HF `infer_multi` output token for token.
3. **Accuracy.** On S5 and a MarathonDocBench subset, compare windowed (W=10) and unwindowed CER, both
   against ground truth. The unwindowed side runs U2-B6's document loop with
   `enforce_page_limit = false` (what `Some(PageWindow::Off)` maps onto), which bypasses the
   pre-windowing refusal.
   Windowed must not be worse than unwindowed beyond a pre-registered tolerance; Baidu calls the scheme
   approximate. U2-D2b records a pass/fail verdict against that tolerance, and U2-E5a reads the verdict.

### 7.6 v1 non-regression gate (U2-T3)

**Capture.** Before any shared-code refactor, capture a model-gated v1 golden set:
- generated **token ids** for S1- and S3-equivalent v1 inputs;
- the existing parity fixtures;
- multi-page streaming `page` events.

**Gate.**
- Every U2 bead that touches the decoder, loader, tokenizer, sampler, PDF walk or CLI output must keep
  these byte-identical.
- The gate runs in `scripts/check.sh` when weights are present, and skips with SUCCESS otherwise. The
  "native path ran" proof applies.

`robot selftest` is a GEMM-vs-scalar check (`cli.rs:3159`). It stays as a kernel guard but is **not**
decoder evidence.

### 7.7 E2E scripts and logging (U2-T1a, U2-T1b)

`scripts/uocr2_e2e.sh` emits NDJSON logs per `docs/TEST_LOGGING.md`. Each step writes one line with
timings, model id, artifact SHA, peak RSS, outcome, and the native-path proof fields `native_path_ran`
and `fallback_target`. Model gating is per step: a step that needs the v2 artifact skips with SUCCESS
(exit 0, logged "skipped: weights absent") when it is absent.

**T1a (M2).** (a), (b), (c), (h) and (r) run only under `FOCR_E2E_LONG=1`, because they decode long or
repeated inputs and the U2-T3 v2 lane already decodes the same pages in check.sh. (j)'s refusal half
and (p) use synthetic artifacts and always run.
- (a) single page, base, and (b) 3-page document;
- (c) 20-page document;
- (g) v1 regression check;
- (h) `--json --extract-figures` per-page outputs in document mode;
- (i) `focr pull unlimited-ocr-2` against the **published** manifest, only when `FOCR_E2E_NETWORK=1`
  (skip with SUCCESS otherwise). Part URLs must be HTTPS (`dist.rs:347-398`), so a local-parts run is not
  possible; U2-E4 unit-tests the offline pieces (a local manifest via `FOCR_MANIFEST_URL=<path>`, parse,
  recipe check, install layout);
- (j) lever matrix: refused levers exit 2 before weights load, ignored levers run with an
  `ignored_lever` warning;
- (k) CJK streaming across a split character;
- (l) v1 honest reporting (U2-X2, reaching the clamp through `FOCR_TEST_MAX_POSITIONS`) plus the
  v1→v2 pull hint (U2-E4);
- (m′) the U2-B6 pre-windowing limit: a document of page limit + 1 pages is refused with exit 2. Added
  only while U2-D1b is open; U2-D1b deletes it when it lifts the limit, and T1b's (m″) owns the
  replacement;
- (p) health, doctor and triage (added by U2-E7);
- (r) `ocr-batch --model unlimited-ocr-2` over 3 images, and with `--multi-page` over one 3-page
  document. Outputs equal the solo v2 runs.

**T1b (M3).** Every T1b step decodes long documents, so each runs only under `FOCR_E2E_LONG=1`;
`check.sh` runs none by default.
- (d) 35-page windowed document, W=10, built by concatenating corpus pages;
- (e) resume after an injected cancellation at window 2, byte-identical;
- (f) every line from (d), (e) and (m) validates against `robot schema`'s `event_fields` (added by
  U2-X2 and extended since);
- (m) page-failure skip with source-page map continuity;
- (m″) a document of N pages, where N is (page limit + 1) rounded up to the next N with N mod 10 == 1,
  runs windowed at W=10 in ceil(N/10) windows, exit 0, with N bodies in order. The last window has one
  own page;
- (n) directory input;
- (o) a ≥ 300-page synthetic run, recording the RSS bound and time-to-first-page;
- (q) a 35-page windowed `--json --extract-figures` run: `pages[]` has length 35, and figures are named
  by source page.

---

## 8. Performance plan (CPU)

### 8.1 Cost model (analytic; MACs per generated token; verified by U2-F1)

| Component | MACs per token |
|---|---|
| Attention q/k/v/o projections | 12 × 4 × 1280² ≈ 79M |
| SR-SWA router | 12 × 2 × 128 × 1280 ≈ 3.9M |
| MoE layers 1–11 (6 routed + 2-expert-equivalent shared) | ≈ 303M |
| Dense layer 0 | ≈ 26M |
| lm_head | 165M |
| Attention scores + AV (v2) | ≤ 12 × 10 × 128 × 2 × 512 ≈ 16M |
| **Total (v2)** | **≈ 0.59G** |

**Comparison with v1.**
- **Single page.** v1 attends to ~408 tokens, about 12.5M MACs of attention, so **decode cost is
  roughly equal** and no speedup is expected.
- **Multi-page decode.** v1 attends to `m = 111·N + 128` tokens.
  - At N=100 that adds ~344M MACs per token (+58%), while v2 stays flat.
  - On decode MACs alone, v2 pulls ahead near 50 pages. But **v1 cannot reach 50 dense pages**: its
    prompt and output share 32,768 positions (`mod.rs:2750-2783`), so it stops near 29 pages at ~1,000
    output tokens per page. Beyond that, only v2 runs.
- **Prefill.**
  - Attention: v1 full causal costs ≈ 15,360·S² MACs, v2 sparse ≈ S·15.7M.
  - Linear prefill work is S × ~430M for both, but v2's S per page is 2.5× larger (274 vs 111).
- **Vision per page.**
  - SAM-B at 1024 is ≈0.45 TMAC, against ≈0.15 TMAC at 640. v2 document-mode vision therefore costs
    ≈2.8× v1 multi-page vision.
  - It costs the **same** as v1 single-page base, since both run at 1024. PERF_LEDGER G2 rows
    (2026-07-08, M-class host, 8 threads) put v1 vision-encode at ≈4.2 s per page.
  - Gundam vision is ≈1.7–6.4× base, depending on the tile count (estimate).

**Honest summary.**
- Single page: parity.
- Documents vs per-page parsing: about the same per page, plus cross-page value (D17).
- Documents vs v1 multi-page: more vision and prefill, less attention, and no length ceiling.

The default-flip gates (§10) treat performance as a **budget, not a selling point**.

### 8.2 Exact levers (each with a bit-identity proof; U2-F2, U2-F3)

U2-F2 owns levers 1 and 6. Levers 2–4 are built into the core (U2-B4 implements incremental `kbar`;
U2-B3 builds the gathered path and the prefill tiling), and lever 5 is U2-D1b's behavior with its
own counter test. F1's breakdown measures them; there is no separate pass for them.
1. **Candidate-only scoring**, with min(M, nblk − #cand) −1e4 pads **merged** into the full top-M order.
   This is exactly equivalent per §5.6.5 in every case; appending the pads is not.
2. **Incremental `kbar`.** Complete blocks are immutable, so only the tail block is recomputed, plus
   copied segments on splice.
3. **Gathered attention layout.** Contiguous per-block K/V slices; the head-major inner loop is left
   to LLVM autovectorization (doctrine #3).
4. **Prefill attention parallelism.** Rayon over (head, query-chunk) tiles inside the forward.
5. **Vision once per page in windowed mode.** Lookahead KV is copied, never re-encoded.
6. **Chunked prefill for v2.** This lifts the D18 refusal. Every chunk uses the unchunked
   M = min(16, ceil(S_prompt/32)) and counts pads from the logical nblk (§5.6.5), so routing
   selections match unchunked prefill unconditionally. Any logit difference is accumulation order in
   the chunked attention, which U2-F2 must show is zero.

### 8.3 Lossy levers (kill-switch gated, CER-measured; U2-F4)

- bf16 KV storage, which halves KV memory.
- int8 attention/`lm_head` and int4 experts are **out of scope** for v2 until the equivalent v1 gates
  exist (doctrine #2).

### 8.4 Benchmarks and wall time

**Benchmark rules.** Every row runs v1 and v2 **in the same invocation**, at the same thread count, on
the same inputs. That is the suite rule: a win needs the incumbent live.
- Rows reuse the existing `FOCR_TIMING=1` rows unchanged: `vision_tower` (`mod.rs:2406`),
  `multi_page.vision` (`:2743`), `prefill … (N tokens)` (`:3812`) and `decode … (N tokens, s/tok)`
  (`:3856`). `scripts/gauntlet_timing.py:124-151` parses them with strict regexes. v2 adds only
  `srswa.router`, `srswa.attend`, `srswa.kbar`, a per-page vision row and `pagesource`.
- **Peak RSS** comes from the gauntlet's `/usr/bin/time` capture (`gauntlet_timing.py`), because there is
  no portable in-process helper (`music_diagnostics.rs:472-485` is Linux-only).
- Rows go to `docs/PERF_LEDGER.md`, losing rows included. Each row fits `check_ledgers.py:149-218`: 21
  columns, with an `evidence_id` under `artifacts/perf/` that has a manifest.

**Book wall time.** Estimate, to be replaced by U2-F1 numbers: 15–60 s per page on M-class hosts,
which is roughly 4–17 hours per 1,000 pages. The docs say so, and progress output shows a live ETA.

---

## 9. Product surface

### 9.1 CLI (U2-E1, U2-D5)

**Model selection.** `--model unlimited-ocr-2`.
- `--model` already resolves bare ids. Candidates are `<id>`, `<id>.focrq`,
  `<id>.v<ver>.{int8,int4}.focrq` and `<id>.{int8,int4}.focrq`.
- Search order: `FOCR_MODEL_DIR` entries, then `<cache>/models`, then sorted `<cache>/models/*/`
  (`mod.rs:917-943`). This order already exists (bd-av64.7). U2-E1 adds no tier: it adds the
  header-arch check, closes the two short-circuits below, and pins the order with tests.
- **Two short-circuits must not fire for a bare registry id**:
  - each search dir short-circuits if it is itself a raw package, for any spec (`mod.rs:1162-1166`, via
    `:855-864`). So `FOCR_MODEL_DIR` pointing at a v1 raw dir would load v1 for
    `--model unlimited-ocr-2`;
  - `resolve_model` errors on a CWD directory named like the spec (`mod.rs:1955-1961`).
  The header arch must equal the requested id, or the result is `ModelNotFound`. Tests cover both.
- With D22, the version comes from the arch.
- **Overrides.** Today the CLI forwards `--max-length`, `--no-repeat-ngram`, `--ngram-window` and
  `--crop-mode` only when they differ from v1's defaults (`cli.rs:409-437`: 32768 / 35 / 128, and crop
  only when `gundam`). U2-E1 makes them `Option<…>` (or checks clap's `ValueSource::CommandLine`), so an
  explicit `--crop-mode base` or `--no-repeat-ngram 35` reaches a v2 model whose defaults differ. v1's
  `--ngram-window 0` means the HF builtin (`sampler.rs:101-102`); for v2 it maps to 100 (§5.9).

**Single page.** `focr ocr page.png --model unlimited-ocr-2 [--crop-mode base|gundam]`.

**Documents.** With v2 resolved, `focr ocr doc.pdf` runs document mode once E5a flips. Until then, add
`--multi-page`.
```
focr ocr doc.pdf --model unlimited-ocr-2 [--pages …] [--split-spreads]
    [--page-window N|auto|off] [--max-page-tokens N] [--resume[=recompute]]
    [--independent-pages] [--json] [-o out.md|out.json] [--extract-figures]
focr ocr ./scanned_book_pages/ --model unlimited-ocr-2 [--multi-page]   # directory = one document
                                                    # (D24); --multi-page needed until E5a flips
```
- **Flag owners.** `--page-window` is U2-D5's. `--resume[=recompute]` and `--keep-checkpoint` are
  U2-D4's (help text and FEATURE_PARITY rows); U2-D5 only cross-references them in help.
- **Refusals today** (`cli.rs`): non-PDF input with `--multi-page` (`:1386-1392`); `--multi-page` with
  split or figures (`:1393-1399`); and a mode-independent `--split-spreads` + `--extract-figures`
  refusal (`:1401-1408`).
  - U2-D0 relaxes `:1386` for directories.
  - U2-E3 lifts `:1393-1399` and `:1401-1408` for v2 document mode only, with half-qualified figure
    names (`page{N}{l|r}_figure_{M}`).
  - v1 keeps all three refusals, and a test asserts that v1 still exits 2.

**Batch.** `focr ocr-batch … [--multi-page] --model unlimited-ocr-2`. `focr ocr --multi-page` is
PDF-only today, so image lists go through `ocr-batch --multi-page` (T1a step (r) covers it).

**Exit codes.** Existing codes only (`src/error.rs`).
- **Unchanged invariant:** `run_complete` is the success terminal (`robot.rs:140`), and every nonzero
  exit ends with `run_error` (`cli.rs:3250-3259`).
- **New partial-output path.** The document driver returns its completed pages plus a `finish_reason`.
  The CLI writes the completed pages (`-o` or stdout) first, then exits:
  - **5** (`EXIT_TIMEOUT`) for `length` (v2's per-window position limit) or `budget`;
  - **6** (`EXIT_CANCELLED`) for `cancelled`.
  Its `run_error` carries the additive partial-output fields (§9.2), and U2-D4 writes the checkpoint.
- **Refusals:**
  - v2 path/lever refusals exit 2 before load (D18);
  - loader-gate recipe refusals exit 7 (`FormatMismatch`, §6.5);
  - an over-limit unwindowed document exits 2 (D11).

**`focr models`.** It lists `unlimited-ocr-2` with status, tasks, recipe, the note "best for:
documents and long documents", and pull state. The ID column (`{:<14}`, `cli.rs:2684`) is widened. Status is
rendered in two places, JSON (`cli.rs:2593`) and the table (`:2691`), both from
`ModelArch::maturity()`. The guidance text lives in the `guidance` map (`:2675`).

**Help text.** It explains when to use v1 vs v2 and corrects the v1 limit text (U2-X2):
- `cli.rs:230` ("must fit the 32K context") and `cli.rs:349` ("32K (~290 pages)") become "32,768
  positions shared by prompt and output (≈29 dense pages)". `cli.rs:34` is the `DEFAULT_MAX_LENGTH`
  constant, not help text.
- `lib.rs:705-714` documents only the prompt-overflow error and omits the output clamp. It gains the
  clamp and the warnings.
- `README.md:411-412` ("roughly 290 pages; over-budget requests fail … instead of truncating") is false
  given the clamp (`mod.rs:2778-2785`; the clamp is at `:2783`), and is corrected too.

**v1 budget errors.** Once the artifact is published (E4), the `v1_truncated` and `v1_page_merge`
warnings and the prompt-too-long error name `focr pull unlimited-ocr-2` as their `recovery`.
- The warnings get it through U2-X2's shared remediation function.
- The prompt-too-long error is `FocrError::Other` at `mod.rs:2750-2758` (exit 1), and gets it there.
- `with_pull_hint` (`cli.rs:3036-3042`) is applied only to `ModelNotFound` today (`cli.rs:1704/1707`).
  It becomes model-aware.

### 9.2 Robot mode (additive; D15)

**`page`** keeps its existing fields and, on v2 runs, adds new ones. Today the two statuses differ
(`robot.rs:205-232`, `cli.rs:1841`):
- decoded: `{status, page (model emission index), chars, text}`;
- skipped: `{status, page (1-based source PDF page), error_kind, message}`.

Both keep those fields and `page` semantics unchanged. **On v2 runs**, both gain:
```
source_page:<1-based PDF page / file index>, half:"left"|"right"|null, window:<k>|null, model_id
```
Decoded v2 events also gain `tokens`, `finish:"marker"|"eos"|"page_cap"|"length"` and `elapsed_ms`.
- **v1 `page` events are unchanged**: v1 merges pages, so its bodies have no reliable source page.
- It is emitted as each page body completes, using token-id segmentation (D20).
- **Ownership:** U2-E3 (M2) adds every field except `window`, with their volatile-list entries, so
  v2's document mode and streaming ship with them. U2-D5 adds `window`.

**`window`** (new):
```
{event:"window", index, pages:[lo,hi], lookahead:0|1, reused:bool, fallback_reason|null,
 prefill_tokens, generated_tokens, kv_bytes_peak, rss_bytes_peak, elapsed_ms, eta_ms}
```

**`run_complete` additions:** `model_id`, `source_sha256`, `pages_out`, `pages_skipped`, `windows`,
`extra_markers` (U2-D5 adds `pages_out`, `pages_skipped`, `windows` and `extra_markers`),
`finish_reason` and `warnings[]` (U2-X2), and `resumed_from` (owned by U2-D4). The library mirror is
U2-E2's `DocumentResult`, which gains `warnings: Vec<Warning>` with the same `{kind, message, recovery,
…}` entries, so library-side kinds reach the CLI and library users.

**Terminal events.** `run_complete` stays the success terminal (`robot.rs:140`), and every nonzero exit
ends with `run_error`. A nonzero exit never follows `run_complete`.
- `finish_reason` on `run_complete` (every one) is `stop`, or `length` for v1 truncation only (exit 0).
  A multi-page v1 truncation carries `v1_truncated`; a v1 page merge keeps `stop` with `v1_page_merge`;
  a single-page v1 run stopped by `--max-length` reports `length` with no warning.
- A **partial-output `run_error`** (exit 5 or 6, after the completed pages are written) carries the
  additive fields `finish_reason`, `pages_out` and `warnings[]`, and from U2-D4, `checkpoint` and
  `resumed_from`. U2-B6 builds that path and produces `length` (v2's position limit) and `budget`;
  U2-D1a produces per-window `budget`; U2-E2 produces `cancelled`; U2-D4 checkpoints on each.
- `ocr_robot_*_stream_matches_exit_code` (`tests/cli_robot_golden.rs:1365-1613`) is updated for the new
  fields, and the schema's `status` string with it.

**`warnings[]`.**
- **Shape:** `[{kind, message, recovery, …kind-specific fields}]`.
  - The key is `recovery`, matching `run_error`'s existing key (`robot.rs:187`). Never `remediation`.
  - Robot: present only when non-empty, on `run_complete` and on a partial-output `run_error`. That
    follows the `low_yield` precedent (`robot.rs:155-174`).
  - `--json`: entries are merged into the **existing** `warnings` array (`cli.rs:1045,1578`). Its music
    elements `{kind, part, measure, detail}` are unchanged.
  - `ocr-batch --multi-page`: entries also go in its `batch.multi_page` object (`cli.rs:2212-2238`),
    plus the human stderr line.
- **Canonical kinds:**
  - `v1_truncated`, `v1_page_merge` (U2-X2);
  - `v2_length`, `leading_text` (U2-B6);
  - `pull_hint` (U2-E1, U2-E4);
  - `resume_recompute` (U2-D4);
  - `ignored_files` (U2-D0);
  - `ignored_lever` (U2-A7);
  - `uncertified_crop_mode` (U2-E1).
- Every older "note" is a `warnings[]` entry, or a plain stderr line where it is purely human output.
  No new event kind is added for them.
- U2-X2 ships `finish_reason` and `warnings[]` for v1 in M0.

**Contract upkeep.**
- **Field-level contract (U2-X2, M0).** `robot_schema()` gains an additive `event_fields` key,
  `{kind: {required, optional}}`, for every kind in `EVENT_KINDS`.
  - `stage` has no emitter today, so it is listed as `{reserved: true}`.
  - A stream contract test validates every emitted line in `tests/cli_robot_golden.rs` against it.
    Today the schema lists only event kinds, so no field can be validated.
  - Every later bead that adds fields or warning kinds extends `event_fields` in the same change
    (U2-A7, D0, B6, E1, E3, D4, D5; D5 owns the `window` event).
- `robot schema` and the frozen fixture `tests/fixtures/robot_schema_v1.json` are regenerated with
  `UPDATE_GOLDENS=1` as a reviewed golden change. The schema stays v1 (additive).
- **Test-only position limit.** `DecodeParams.max_positions` (`#[doc(hidden)]`) is also settable
  through the hidden env `FOCR_TEST_MAX_POSITIONS`, read once at startup.
  - A value above the arch's context is refused with exit 2.
  - It is listed as test-only in `ENVIRONMENT_VARIABLES` and on U2-A7's census allow-list.
  - CLI and e2e tests use it to reach the clamp and position-limit branches.
- **Surface matrix.** `docs/FEATURE_PARITY.md` §12 (CLI) and §13 (robot) get the new rows (a `window`
  row, new flags) and a rollup bump. `tests/surface_matrix.rs` parses those sections; its rows live
  only in the doc.
- Coordinate with bd-wp8.2, bd-wp8.3 and bd-znwm.
- **`robot triage`.** `RobotCmd::Triage` is a unit variant with no input today (`cli.rs:818`,
  `robot_triage_payload()` at `:2976`).
  - U2-E7 adds an optional `[INPUT]`. Output is unchanged without one, and the change is recorded in
    FEATURE_PARITY §12.
  - With a PDF over about 10 pages as input, it recommends v2 document mode when v2 is installed, and
    `focr pull unlimited-ocr-2` when it is not.

### 9.3 Structured document output (U2-E3)

v2's token-level `<PAGE>` segmentation gives document mode:
- a per-page `pages[]` array, with layout boxes (0–999 per page) mapped through each source page's
  original size. The layout and figures come from v2's block grammar, which U2-E3b parses
  (§5.1, DISC-U2-13);
- per-page figure extraction (`page{N}_figure_{M}`);
- the source-page map.

This lifts, for v2 document mode only, the `--multi-page` refusal of `--extract-figures` and
`--split-spreads` (`cli.rs:1393-1399`) and the mode-independent split + figures refusal
(`cli.rs:1401-1408`), with half-qualified figure names (`page{N}{l|r}_figure_{M}`). v1 keeps both.
- Figure pages are re-fetched through `PageSource::fetch` in both unwindowed and windowed mode, because
  document mode prefills every page before decoding.
- Coordinate with bd-hfdt-admission-prereqs-dfr1, bd-json-per-span-text-wbbu and bd-v73v.

*As built (2026-10-30).*
- `PageResult` and `Half` live in `native_engine`, so the library type builds without the `pdf`
  feature. `postprocess::finalize_page(body, index, size)` turns one body into its text, layout and
  figure boxes; the document markdown is the page texts joined by `<PAGE>`.
- The v2 document loop always runs the token-id page stream, which records each page's generated-id
  count and wall time. A page's `finish` is the guard's (`stats.page_finish`); the streaming sink gets
  the same `PageResult` the final `pages[]` holds.
- The CLI decides "this model has exact pages" by the bounded header probe after the mode default
  (`paged_model_id`), so the refusals stay for v1 and for an absent model.
- v1 takes decision (a): pages only when the text has exactly one body per page.

### 9.4 Library (U2-E2, D23)

```rust
let engine = OcrEngine::new()?;
let opts = DocumentOptions::builder()
    .model(ModelChoice::Id("unlimited-ocr-2"))
    .page_window(PageWindow::Auto)
    .max_page_tokens(20_000)
    .resume(ResumePolicy::IfMatching)
    .build();
let doc = engine.recognize_document(
    PageSource::pdf("book.pdf", PageSelection::All),   // or PageSource::dir("pages/")
    &opts,
    |page: &PageResult| { /* stream */ },
)?;
```
- The call is synchronous and blocking; the runtime is owned internally.
- **Page window.** `DocumentOptions.page_window` is `Option<PageWindow>`, and U2-E2 owns the
  `PageWindow::{N, Auto, Off}` enum. `None` is the default policy: until U2-D1b lands, unwindowed with
  U2-B6's page limit enforced; after it, W=10 for N > W. `Some(PageWindow::Off)` is the explicit
  unwindowed request that bypasses the limit (U2-B6's `enforce_page_limit = false`).
- **Result.** `DocumentResult { pages_out, pages_skipped, windows, extra_markers, finish_reason,
  warnings, resumed_from, model_id, source_sha256 }`. `warnings: Vec<Warning>` carries U2-X2's
  `{kind, message, recovery, …}` entries, so library-side kinds (`leading_text`, `v2_length`,
  `v1_page_merge`, `resume_recompute`, `ignored_files`) reach the CLI and library users.
- It replaces the five `recognize_multi_page*` wrappers (`lib.rs:703-802`), with a minor-version bump
  and a CHANGELOG migration table (owned by U2-E2).
  - Other callers migrate in the same change: `cli.rs:2221-2222`, `tests/e2e_recognize.rs:684-817` and
    `tests/parity_ladder.rs:1996`.
  - Docs that name the old API are updated: `DISCREPANCIES.md:410` and `FEATURE_PARITY.md:76`.
- **iOS.** focr-ios holds `FocrEngine { model: Arc<OcrModel> }` (`focr-ios/src/lib.rs:280-284`) and
  never builds an `OcrEngine`. Its single call (`:~1152`) is
  `engine.model.recognize_multi_page_dynamic(images)`.
  - So a public `OcrModel`-level document driver sits behind `OcrEngine::recognize_document`, and
    focr-ios moves onto it.
  - focr-ios's private `fn recognize_document` (`:401`) is renamed `recognize_single_image`, so it does
    not shadow the new name.
  - Verify with `cargo check --locked --workspace --all-targets`. check.sh's root-package check does not
    compile workspace members (`Cargo.toml:206-208`, with no `default-members`).
- `stream_pages` (`lib.rs:282`) stays public API. It is exercised by `lib.rs:961`,
  `tests/many_pages_without_deadlock.rs:1189` and `tests/cancel_and_panic_faults.rs:199,219`; no
  production code calls it.
- `recognize_pdf` (bd-paom) is its per-page sibling over the same walk.
- The engine caches one model (`lib.rs:404-436` `model_at`), so per-mode defaults resolve the model
  before engine construction (D25).

*As built (U2-E2, 2026-10-30).* The call is
`engine.recognize_document(model_path, source, &options, on_item, on_page)`, wrapping
`OcrModel::recognize_document(&mut source, &options, on_item, on_page)`. Version 0.10.0; the CHANGELOG
holds the migration table. Where it departs from the sketch above:
- **Model.** The model is the explicit `model_path` (D25 resolves it first), not a `ModelChoice` option.
- **Options.** `DocumentOptions { page_window, resume }` is `#[non_exhaustive]` with chaining setters,
  not a builder. `max_page_tokens` and `entmax_domain` are left out because they already have one
  route each (`DecodeOverrides`, `--max-page-tokens`; `FOCR_SRSWA_ENTMAX`), and a second would
  duplicate config. `crop_mode` is left out because a document reads base views only (DISC-U2-6).
- **Sink.** `on_page` streams `StreamedPage` (a body, and its `PageResult` when the model splits pages
  exactly), because v1's bodies can merge pages and have none. `on_item` sees every page before it is
  read, a skipped one included.
- **Result.** `DocumentResult { markdown, pages, pages_out, skipped, finish_reason, extra_markers,
  warnings, model }`. `skipped` lists the pages (with reasons), not a count; `model` is
  `ModelIdentity { model_id, source_sha256 }`. `windows` and `resumed_from` arrive with U2-D1b and
  U2-D4.
- **Cancellation.** Ctrl+C or `request_shutdown()` stops Unlimited-OCR-2's document loop at its next
  step with `finish_reason: cancelled`; the CLI writes the pages, then exits 6. v1 still fails with
  `Cancelled`.
- **Features.** Documents need the `pdf` feature, which every shipped build enables, because the page
  source and `SkippedPage` live behind it. A pure core without `pdf` has no document API (bd-x7jt).

### 9.5 Distribution (U2-E4)

- **Manifest.** A manifest-v2 `models["unlimited-ocr-2"]` entry holds:
  - `license_notice`;
  - `version` (D22). `ModelEntry` (`dist.rs:134-148`) has no such field today, so serde would drop it.
    Add `#[serde(default)] version: Option<String>` with a parse test. `MANIFEST_SCHEMA_VERSION` stays
    2;
  - `quants.int8 {recipe, focrq{filename,size,sha256,parts[]}}`;
  - `tokenizer {filename, size, sha256, parts[]}`, the same `RemoteFile` shape as every other download
    (`dist.rs:165-186`).
- **Recipe check.** The table moves from `dist.rs::required_quant_recipe` (`:485`) into
  `ModelArch::required_quant_recipe` (§6.1), with a v2 arm. The manifest recipe is checked before
  download (`dist.rs:1150-1155`).
- **Pull.** `focr pull unlimited-ocr-2` installs into the model subdirectory, with artifact and sidecar
  in one transaction (bd-jprb).
- **Docs and hints.** The install docs' size text is updated, and `with_pull_hint` (`cli.rs:3036`)
  becomes model-aware.
- **Before publishing,** U2-E4 runs U2-A1's upstream-drift check (R20).
- **HF mirror.** Per bd-3p8y/bd-aio5; non-blocking.

### 9.6 Health and doctor (U2-E7)

- **`robot health`** reports every installed model (`cli.rs:2949` reports one), with artifact SHA,
  recipe and tokenizer SHA. `robot health` is frozen byte-exact against
  `tests/fixtures/golden/robot_health.golden` (`cli_robot_golden.rs:894-931`), so multi-model health is a
  reviewed golden change.
- **`doctor`** scans per-model subdirectories: `detect_model_not_resolvable` (`doctor.rs:121`),
  `detect_stale_focrq_format` (`:142`) and `detect_unreadable_cache_entries` (`:186`). Orphan scanning
  already recurses (`:273`). `detect_model_not_resolvable` is not a no-op: it checks only the default
  model spec and ignores the doctor root, so it too gains the per-model tier (each installed model's
  resolvability is reported).
  - It detects a v1 tokenizer beside a v2 artifact and reports `tokenizer_mismatch` as
    `RefusedUnsafe { recommended_command: "focr pull unlimited-ocr-2" }`.
  - There is no `.bak` next to the file and no network access inside `doctor --fix`.
  - Doctor's laws (`doctor.rs:8-19`) put every `--fix` write behind the `mutation` chokepoint: a
    backup into `.doctor/runs/<id>/backups/`, before and after SHA-256 in `actions.jsonl`, and
    revertible with `focr doctor undo`. Any future automatic fix must be a new op there.
- **`robot triage`** is v2-aware (§9.2).

### 9.7 Documentation (U2-E6a at M2, U2-E6b at M4)

- **README:**
  - the model table;
  - a "Long documents and books" section;
  - inputs and limitations: JBIG2/JPX/vector pages are skipped, and the directory route is
    recommended;
  - book wall-time estimates;
  - v2 limitations: single-page parity only; windowed mode is approximate (per Baidu); gundam is
    uncertified until G1; there is no global heading hierarchy across windows (per the report).
- **AGENTS.md** "What this model is": a v2 paragraph.
- **CHANGELOG**: U2-E2 writes the API migration table, and the docs beads link to it.
- **The master plan's §13**: a cross-reference to OQ-V2.
- **DISCREPANCIES** and **PERF_LEDGER** rows. U2-C4 files the DISC-U2-k entries under numeric
  `## DISC-NNN: U2-k — …` headings (§5.11), and the docs beads link to them.

---

## 10. Default-policy decision framework (U2-E5a, U2-E5b)

| Mode | Today | After M2 (artifact certified) | After M4, if gates pass |
|---|---|---|---|
| `focr ocr page.png` | v1 base | v1 (v2 opt-in) | v2 if the single-page gate passes (E5b), else v1 |
| `focr ocr doc.pdf` | v1, per page | v1, per page (v2 document mode opt-in) | **v2 document mode** (D17) if the multi-page gate passes (E5a). `--independent-pages` keeps per-page parsing, with the single-page default model. |
| `--multi-page` | v1 (32,768 positions shared; ~29 dense pages) | v1 (v2 opt-in). v1 budget errors name `focr pull unlimited-ocr-2`. | v2 if the multi-page gate passes |
| `ocr-batch` | v1 | v1 | Follows the single-page decision; `--multi-page` follows the multi-page decision |
| `focr pull` (no argument) | v1 artifact | v1 | v1 until **both** gates pass, then v2 (D25 end state). Whichever of E5a and E5b flips second sets this entry. v1 installs are never auto-upgraded into a 4 GB download. |

**Model resolution (D25):** `--model` > `FOCR_MODEL_PATH` > the mode default above > v1. A pull hint
is printed when the mode default is missing. JSON output and `run_complete` carry `model_id` and
`source_sha256`.

**Multi-page gate (E5a; all required).**
1. C4 green on the int8 artifact.
2. A **ground-truth head-to-head** (U2-C5a/C5b) on ≥ 30 documents of 5–60 pages and ≥ 5 documents of
   100+ pages. v2 document mode must beat v1 multi-page **and** v1 per-page.
   - **Statistic:** the lower bound of a paired per-document bootstrap of (ED_v1 − ED_v2) is above 0.
   - **No category regression** beyond the pre-registered tolerance.
   - **v1 failures count.** A v1 run that errors on the 32K budget or truncates scores edit distance
     1.0 for every page it did not produce. No document is dropped for failing on one side.
   - **At least a third of the documents are non-Baidu** (OQ-V2-15), because MarathonDocBench and
     MPDocBench are Baidu-built.
3. Windowed-mode validation (§7.5) green.
4. **Cost bounds.** CPU wall time per completed page is within 1.5× of v1 multi-page at ≤ 20 pages, and
   within 1.2× of v1 per-page parsing. Peak RSS at 20 pages is ≤ v1 multi-page + 1 GiB. Both numbers
   are published either way. Evidence: U2-F1 rows for ≤ 10 pages (unwindowed), and U2-C5b cells for
   11–20 pages at W=10, because the shipped default windows those documents.
5. The surface ledger is complete, and the robot schema golden is updated.

**Single-page gate (E5b; all required).**
1. v2, in its chosen crop mode (G2), is **non-inferior** to v1 on ≥ 300 ground-truth pages.
   - **Statistic:** the upper bound of a paired bootstrap of (CER_v2 − CER_v1) is below δ.
   - Table pages pass the same test on TEDS.
   - δ (proposed: 0.005) is committed before the run.
   - A superiority bound cannot pass at the reported 93.92 vs 94.02, and it would freeze a two-model
     default forever.
2. CPU wall time ≤ 1.1× v1.
3. Hard-page EOS behavior is no worse.
4. Peak RSS ≤ v1 + 0.5 GiB.

**Evidence sets.**
- **MarathonDocBench** documents and pages (Apache-2.0, per-page ground truth).
- **Non-Baidu documents** with ground truth (OQ-V2-15). Candidates:
  - public-domain book scans with existing human-proofread transcriptions;
  - born-digital PDFs rasterized out-of-band for evaluation only, with text-layer ground truth
    normalized for reading order.
- **OmniDocBench** (OpenDataLab, not Baidu) for single pages, if its license allows local evaluation
  (OQ-V2-12).
- **Metrics:** our minimal normalized edit distance, plus the official MPDocBench-Parse toolkit, run
  offline, for reported numbers.
- **Compute budget** (estimate): both models over 35 documents is ≈ 15–40 CPU-hours. C5 runs it once
  per candidate artifact.

---

## 11. Roadmap and milestones

Each milestone ends with something a user can run, and each has an exit gate.

**M0 — Foundation and early v1 wins.**
- **Work:**
  - T3: v1 golden gate;
  - A1: truth pack;
  - A2: identity;
  - A3: sharded sources;
  - A4: tokenizer;
  - A5: arch and census;
  - A6: oracle;
  - A7: fail-closed dispatch;
  - **X1:** CJK streaming fix (v1);
  - **X2:** v1 multi-page honest reporting: truncation and page-merge warnings, `finish_reason`,
    corrected help text, and the field-level `event_fields` robot contract;
  - **D0:** streaming page source (v1 multi-page memory).
- **User value:**
  - v1 Chinese multi-page streaming no longer dies after a split character;
  - v1 `--multi-page` no longer exits 0 silently when it truncates (about 30–295 pages) or merges
    pages. Every recorded fixture merges: p20 gives 7 bodies for 20 pages, p10 9 of 10, p9_p14 1 of 2;
  - agents can validate every robot event's fields (`event_fields`);
  - v1 `--multi-page` no longer holds every raster twice;
  - a v1 tokenizer sidecar can no longer be silently accepted for the wrong arch.
- **Exit:**
  - v1 is byte-identical;
  - the v2 checkpoint loads (census OK) behind `implemented() == false`, and every path refuses it
    explicitly (exit 2, before weights load);
  - oracle fixtures S1–S4 are generated (f32 CPU), with the floor recorded. The S1-corpus references
    (§7.1) are not an M0 exit item.

**M1 — Correct native v2 (experimental).**
- **Work:** B1–B7, C1, C2.
- **User value:** `focr ocr --model /path/to/Unlimited-OCR-2/` (raw BF16 shards, experimental) works
  for single pages (base) and for unwindowed documents.
- **Exit:**
  - L0–L4 green on S1–S3 against the f32 oracle on the stateless f32 path;
  - `generate_cached` L4 token-exact on S1–S3, or any mismatch root-caused to the int8 FFN quant
    effect and recorded as a DISC (§7.2);
  - retirement exact on the fixtures, or `FOCR_SRSWA_RETIRE` defaulted to 0 with a filed bug. E5a's RSS
    bound then decides whether that matters;
  - v1 gate green.

**M2 — Certified opt-in model.**
- **Work:** C3, C4, C5a, E1, E3, E4, E6a, E7, F1, T1a.
- **User value:**
  - `focr pull unlimited-ocr-2`;
  - `--model unlimited-ocr-2` for single pages and for documents up to the U2-B6 page limit
    (unwindowed), with per-page JSON, figures and the extended `page` events (U2-E3);
  - v1 budget errors point to v2.
- **Exit:** corpus gates green; artifact published; perf and RSS ledger rows published; C5a evidence
  recorded.

**M3 — Unlimited-horizon books.**
- **Work:** E2, D1a, D1b, D2a, D2b, D3, D4, D5, C5b, T1b, T2, F2, F3. D2a, the independent oracle,
  can start as soon as A1 and A6 close.
- **User value:**
  - 1,000-page PDFs or page directories on CPU, with bounded memory, streaming, and resume;
  - robot `page` and `window` events;
  - `recognize_document`.
- **Exit:**
  - §7.5 green;
  - watchdog green;
  - T1b green;
  - a measured ≥ 300-page run on a 16 GB laptop-class host, with peak RSS ≤ 8 GiB at W=10 and the first
    page emitted after one window, published in PERF_LEDGER.

**M4 — Evidence-gated document default.**
- **Work:** E5a, E6b.
- **Exit:** the decision table is published, and document/multi-page defaults flip only if the gate
  passes.

**M4b — Single-page decision (never blocks M4).** G1, G2, E5b.

**M5 — Ecosystem and further perf (backlog).** F4 bf16 KV; F5 cross-document batching; H1
iOS/WASM/site v2 lanes.

**Critical path.** The longest blocking chain in the bead graph has 14 beads (recomputed after bead
polish round 3. The round-3 edges B4 → B3, D1a → D2a, F4 → D2b and G2 → C5b do not lengthen it:
B3 sits at B2's depth, so A1 → B1 → B3 → B4 → … is an equally long chain; D2a sits at depth 3; and
F4 and G2 sit on short side branches. The round-4b edges A7 → X2 and D0 → X2 do not lengthen it
either: X2 sits at depth 2, below A7's A5 input, and D0 only feeds D1b, which waits on E2):
```
A1 → B1 → B2 → B4 → B5 → B6 → E3 → E2 → D1b → D4 → D5 → T1b → E5a → E6b
```
E3 owns the library `PageResult` that E2 reuses. The certification track runs alongside and joins at
T1b and E5a:
```
T3 → A2 → A5 → A7 → B5 → C1 → C2 → C4 → E4 → T1a → T1b
T3 → X2 → B6, A7, D0
B1 → B3 → B4
A1, A6 → D2a
B4, B6, D2a → D1a ─┬→ D2b (← D2a) → C5b → E5a
D0, E2 ────────────┴→ D1b → D3, D4 → D5 → T1b
```
A3, A4, A6, B7, X1 and D2a run in parallel from the start, or as soon as their deps close. D2a lands
before D1a, so the independent oracle exists before the Rust windowed core is written.
G1, C5b → G2 → E5b runs off the critical path (G2 reads C5b's long-document evidence). F4 waits on
D2b, whose subset it reuses. Bead ids are listed in §12.

---

## 12. Task graph (mirrored into beads; labels in brackets)

Each entry is written **ID — title** [labels] (blocking deps) — `bead id`. This list was regenerated
from the live `br` graph after bead polish round 3; when they disagree, the beads win. Each bead carries
milestone labels `uocr2-m0`…`uocr2-m5` (omitted here) and:
- the full spec excerpt it needs;
- acceptance criteria;
- the tests it owns (§7.4);
- logging requirements;
- rationale.

**Epic U2.** Unlimited-OCR-2: additive model + unlimited-horizon long-document parsing +
evidence-gated defaults [uocr2,epic].

**A — Foundation, identity, safety**
- **U2-T3** — v1 token-id golden capture and non-regression gate (before any shared refactor) [uocr2,conformance,test] () — `bd-uocr2-epic-cgux.1`
- **U2-A1** — Truth pack v2: pin sources, wheel and SGLang notes; register OQ-V2 [uocr2,conformance] () — `bd-uocr2-epic-cgux.2`
- **U2-A2** — Model-identity hardening: absent model_id ≡ v1 forever; capability methods; per-arch tokenizer validation [uocr2,identity] (T3) — `bd-uocr2-epic-cgux.3`
- **U2-A3** — Sharded safetensors sources (index + N shards): raw load, streaming `focr convert`, I64 verify-and-skip, multi-shard provenance [uocr2,convert,weights] (T3) — `bd-uocr2-epic-cgux.4`
- **U2-A4** — Tokenizer v2: added tokens 128827–128830, v2 tokenizer pin, conformance vs HF [uocr2,conformance,tokenizer] (T3,A1) — `bd-uocr2-epic-cgux.5`
- **U2-A5** — Register `unlimited-ocr-2` ModelArch (implemented=false) + delta census + vision byte-identity hashes + per-model artifact version + recipe id [uocr2,identity] (A1,A2,A3,A4) — `bd-uocr2-epic-cgux.6`
- **U2-A6** — v2 reference oracle: f32 CPU primary (feasibility first), bf16 ledger, dumps, fixtures S1–S5, nondeterminism floor [uocr2,conformance,oracle] (A1) — `bd-uocr2-epic-cgux.7`
- **U2-A7** — Fail-closed v2 dispatch: every decode path, lever, daemon and recipe supports v2, is ignored with a warning, or refuses it before weights load [uocr2,identity,safety] (A5,X2) — `bd-uocr2-epic-cgux.8`
- **U2-X1** — Fix multi-page streaming: page events stop permanently after a split multi-byte UTF-8 char (v1 bug); UTF-8-safe incremental decoder [uocr2,bug,streaming] (T3) — `bd-uocr2-epic-cgux.9`
- **U2-X2** — v1 multi-page honest reporting: truncation and page-merge warnings, run_complete finish_reason/warnings, robot event_fields, help-text fix [uocr2,bug,cli,robot] (T3) — `bd-uocr2-epic-cgux.51`

**B — Native SR-SWA decoder**
- **U2-B1** — page2d coordinates, decode state machine, page_base, and RoPE tables [uocr2,decoder,rope] (A1) — `bd-uocr2-epic-cgux.10`
- **U2-B2** — Token Sieve router math (projections, Gemma norm, RoPE, kbar, scores, Page Gate, both entmax domains, top-K, bias) with a scalar oracle [uocr2,attention,decoder] (B1) — `bd-uocr2-epic-cgux.11`
- **U2-B3** — Block-sparse biased attention (prefill and decode): dense scalar oracle plus gathered path [uocr2,attention,decoder,kernel] (B1) — `bd-uocr2-epic-cgux.12`
- **U2-B4** — SR-SWA caches: K/V, index keys, incremental kbar, bmin/bmax, page state, block-granular retirement with open-row detector, memory accounting [uocr2,decoder,memory] (B1,B2,B3) — `bd-uocr2-epic-cgux.13`
- **U2-B5** — Decoder integration: LayerAttention dispatch; v2 conservative cached (certified-artifact path) + stateless f32 paths; weight caches; v1 byte-identity [uocr2,decoder] (T3,A5,A7,B3,B4) — `bd-uocr2-epic-cgux.14`
- **U2-B6** — v2 decode contract: prompt, guard, generated-only n-gram, per-page budgets, token-id page split, postprocess; flip implemented=true (experimental) [uocr2,decoder,sampler] (A4,A6,X1,B5,B7,X2) — `bd-uocr2-epic-cgux.15`
- **U2-B7** — v2 preprocessing: base-1024 letterbox (PIL-bicubic exact), gundam max 16 tiles, small-image rule [uocr2,preprocess] (T3,A1,A6) — `bd-uocr2-epic-cgux.16`

**C — Parity and certification**
- **U2-C1** — Parity ladder L0–L2 for v2 vs the f32 oracle, incl. routing equality on fallback queries + margin diagnostics [uocr2,conformance,test] (A6,B5,B7) — `bd-uocr2-epic-cgux.17`
- **U2-C2** — Parity ladder L3–L4 + re-prefill≡incremental metamorphic + cross-page invariants [uocr2,conformance,test] (B6,C1) — `bd-uocr2-epic-cgux.18`
- **U2-C3** — Conservative int8 artifact for v2 (recipe, matcher narrowing, convert from shards, loader gate, WASM refusal) [uocr2,convert,quant] (A3,A5) — `bd-uocr2-epic-cgux.19`
- **U2-C4** — L5 corpus gates (ratcheted) for the v2 int8 artifact + DISC entries [uocr2,conformance,quant] (C2,C3) — `bd-uocr2-epic-cgux.20`
- **U2-C5a** — Ground-truth head-to-head: single page + documents ≤ W pages (v1 vs v2) + ground-truth sourcing incl. non-Baidu [uocr2,eval] (C4) — `bd-uocr2-epic-cgux.21`
- **U2-C5b** — Ground-truth head-to-head: long documents (windowed v2 vs failure-scored v1) + entmax-domain and doc crop-mode evidence [uocr2,eval,longdoc] (C5a,D3,D2b) — `bd-uocr2-epic-cgux.22`

**D — Documents and unlimited horizon**
- **U2-D0** — Streaming page source: lazy per-window rasterization, source-page map, directory input, page-failure skip; multi-page CLI moved onto it [uocr2,longdoc,pdf] (T3,X2) — `bd-uocr2-epic-cgux.23`
- **U2-D1a** — Windowed decoding core: planner, lookahead, retention, splice (incl. one-page variant), forced `<PAGE>`, fallback, stitching, page_base, worked coordinate table [uocr2,decoder,longdoc] (B4,B6,D2a) — `bd-uocr2-epic-cgux.24`
- **U2-D1b** — Windowed document pipeline: streaming page source per window, raster dropping, once-per-page vision, recognize_document integration, W=10 default [uocr2,decoder,longdoc,pdf] (D0,E2,D1a) — `bd-uocr2-epic-cgux.49`
- **U2-D2a** — Independent Python windowed oracle, written from the SGLang wheel source (not from the Rust implementation) [uocr2,conformance,longdoc,oracle] (A1,A6) — `bd-uocr2-epic-cgux.50`
- **U2-D2b** — Windowed validation: splice parity W=2/3 vs the independent Python oracle, invariants, accuracy vs unwindowed [uocr2,conformance,longdoc] (D2a,D1a) — `bd-uocr2-epic-cgux.25`
- **U2-D3** — Whole-process memory estimator, `--page-window auto`, RSS measurement [uocr2,longdoc,memory] (D1b) — `bd-uocr2-epic-cgux.26`
- **U2-D4** — Durable resume: sidecar checkpoints, `.partial` output, resume key, byte-identical resume [uocr2,longdoc,storage] (D1b) — `bd-uocr2-epic-cgux.27`
- **U2-D5** — Streaming UX: page.window field, new window event, progress/ETA, --page-window flag [uocr2,cli,longdoc,robot] (D3,D4,E1,D1b) — `bd-uocr2-epic-cgux.28`

**E — Product surface**
- **U2-E1** — CLI/robot basics for v2: id resolution with per-model version, crop-mode semantics, `focr models` row, model_id in outputs, schema/surface rows [uocr2,cli,robot] (B6) — `bd-uocr2-epic-cgux.29`
- **U2-E2** — Library `recognize_document` replacing `recognize_multi_page*`; streaming sink; window/resume options; CLI built on it [uocr2,library] (B6,D0,E3) — `bd-uocr2-epic-cgux.30`
- **U2-E3** — Structured document output: per-page `pages[]`, layout, figures, split-spreads from token-level `<PAGE>` segmentation [uocr2,cli,postprocess] (B6,D0) — `bd-uocr2-epic-cgux.31`
- **U2-E4** — Distribution: manifest entry, artifact parts, `focr pull unlimited-ocr-2`, tokenizer sidecar transaction, v1→v2 pull hints [uocr2,dist,release] (C4,E1,X2) — `bd-uocr2-epic-cgux.32`
- **U2-E5a** — Document/multi-page default policy (D17, D25) and flip if the gate passes [uocr2,policy] (A2,C5b,D4,D5,E3,E4,F1,T1b,T2) — `bd-uocr2-epic-cgux.33`
- **U2-E5b** — Single-page default policy and flip if the gate passes [uocr2,policy] (C5a,E4,G2) — `bd-uocr2-epic-cgux.34`
- **U2-E6a** — Docs for opt-in v2: README, AGENTS, CHANGELOG, linked to the DISC and PERF_LEDGER evidence [uocr2,docs] (E1,E3,E4,F1) — `bd-uocr2-epic-cgux.35`
- **U2-E6b** — Docs for books and defaults: long-document section, wall-time, limitations, migration table [uocr2,docs] (D4,D5,E5a,T1b) — `bd-uocr2-epic-cgux.36`
- **U2-E7** — Health/doctor/triage awareness of v2 (multi-model health, sidecar checks, recommendations) [uocr2,cli,doctor] (E4,T1a) — `bd-uocr2-epic-cgux.37`

**F — Performance**
- **U2-F1** — Same-invocation v1 vs v2 baseline + FOCR_TIMING rows + RSS + PERF_LEDGER [uocr2,perf] (A6,C2,C3) — `bd-uocr2-epic-cgux.38`
- **U2-F2** — Exact SR-SWA levers: candidate-only scoring and v2 chunked prefill; measure B3/B4's built-in levers [uocr2,kernel,perf] (F1) — `bd-uocr2-epic-cgux.39`
- **U2-F3** — Document-mode vision cost at 1024/page: measure + exact levers; once-per-page vision in windows [uocr2,perf,vision] (F1,D1b) — `bd-uocr2-epic-cgux.40`
- **U2-F4** — bf16 KV storage option (lossy, CER kill-switch) [uocr2,memory,perf] (C4,D3,D4,D2b) — `bd-uocr2-epic-cgux.41`
- **U2-F5** — Cross-document batching of windows, byte-identical per document (backlog) [uocr2,batch,perf] (C4,D1b) — `bd-uocr2-epic-cgux.42`

**G — Gundam**
- **U2-G1** — Gundam v2 parity + certification (selective routing; int8 flip sensitivity) [uocr2,conformance,preprocess] (B7,C2,C3,C4) — `bd-uocr2-epic-cgux.43`
- **U2-G2** — v2 crop-mode decisions: single page (base vs gundam) and whether gundam document mode earns its CPU cost [uocr2,eval] (C5a,C5b,G1) — `bd-uocr2-epic-cgux.44`

**T — Tests**
- **U2-T1a** — e2e script `scripts/uocr2_e2e.sh` with NDJSON logging, M2 scenarios [uocr2,e2e,logging,test] (A7,E1,E3,E4) — `bd-uocr2-epic-cgux.45`
- **U2-T1b** — e2e book scenarios: windowed, resume, robot, skip, directory, ≥300 pages with RSS [uocr2,e2e,longdoc,test] (D3,D4,D5,T1a) — `bd-uocr2-epic-cgux.46`
- **U2-T2** — Concurrency watchdog scenarios for windowed v2 documents [uocr2,test] (D4,D1b) — `bd-uocr2-epic-cgux.47`

**H — Ecosystem**
- **U2-H1** — iOS/WASM/site v2 lanes (int4 calibration, artifacts, catalog, iOS multi-page API migration) (backlog) [uocr2,ios,wasm] (E2,E4) — `bd-uocr2-epic-cgux.48`

**Related existing beads.** These are relation edges, not blockers; see Appendix D.

---

## 13. Risks and mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Discrete routing makes parity fragile (fallback queries, gundam) | M | H | f32 CPU oracle (D4) compared against the stateless f32 native path; routing-equality rung that includes fallback queries; margin diagnostics; a tie-flip class with a ceiling; `nblk`-per-forward and padding preconditions asserted (§5.6.5) |
| R2 | Reference stack drift (transformers 4.52.1 vs v1's 4.57.1) | H | M | Separate pinned venv; `check_oracle_provenance.py` gains v2 pin constants beside its hard-coded `PIN_TORCH`/`PIN_TRANSFORMERS` (`:28-29`) |
| R3 | HF config quirks: the `_CKPT_SP` snapshot, the `DASH_STAGE0` env var, and `_PAGE_SCOPE_PREFILL` being set only inside `infer` | M | H | Run the oracle only via `infer()`; snapshot values are documented in §5.6 |
| R4 | CPU vision cost at 1024 per page makes documents slower than v1 multi-page | H | M | Honest ledger; treat cost as a budget; F3 levers; v2 matches v1 per-page parsing, which is also 1024 |
| R5 | Memory for long documents (120 KiB per token in f32) | H | H | Block retirement (D9), windowing, bf16 KV (F4), the estimator, `auto` W |
| R6 | Silent misload of v1 artifacts after a default change | M | H | A2 lands first; a test-only registry with v2 as default |
| R7 | The shared-decoder refactor perturbs v1 | M | H | T3 token-id goldens before B5; enum dispatch; the shared `prefill_attention` stays untouched |
| R8 | Windowed mode is approximate (Baidu's word) | M | M | Splice parity against a Python oracle; accuracy vs unwindowed (`Some(PageWindow::Off)`) and vs ground truth, with a pass/fail verdict against a pre-registered tolerance; W tunable |
| R9 | The windowed reference exists only as a GPU SGLang wheel | H | L | Independent Python windowed oracle on the HF model, ported function by function from the wheel before U2-D1a is written; an optional SGLang run |
| R10 | Distribution blocked by the missing HF org (bd-3p8y) | H | L | GitHub release parts are primary |
| R11 | License ambiguity (no LICENSE file in the HF v2 repo) | L | M | Use the GitHub LICENSE text and the HF metadata; record both in the truth pack and the artifact notice |
| R12 | Process ceremony crowding out shipping | M | M | Every bead closes on cited evidence; tests live in the beads that own them; no spec-only beads beyond A1 |
| R13 | No CUDA host (the v1 bf16 oracle never ran here) | H | H | f32 CPU oracle as primary (D4); GPU only for the optional ledger |
| R14 | Eager rasterization (plus a clone) exhausts RAM on books before KV matters | H | H | Streaming page source (U2-D0); RSS exit gate at M3 |
| R15 | A v2 artifact silently runs v1 math through an unaudited lever or the unknown-id fallthrough | M | H | D18 fail-closed dispatch: a pre-load refusal (exit 2), `UOCR2_LEVER_POLICY` classified by effective state (presence-based levers by presence; `FOCR_RUNAWAY_GUARD` ignored only once the v2 arm stops building it), a census test over every `FOCR_*` variable, and a per-lever test (U2-A7) |
| R16 | Baidu built both ground-truth benchmarks, so the head-to-head may favor v2's training distribution | M | M | ≥ 1/3 non-Baidu documents (OQ-V2-15) |
| R17 | Real book PDFs use JBIG2, JPX or vector content the PDF path refuses (bd-0a7) | H | M | Skip pages and keep the source map; directory input (D24); a follow-up decoder evaluation |
| R18 | Book runs take hours on CPU (estimate: 4–17 h per 1,000 pages), exposed to sleep and throttling | H | M | Byte-identical resume; partial output written before an exit 5/6 (budget checked cooperatively inside the document loop, D11); an ETA; honest docs |
| R19 | A split default means two artifacts and possibly two resident daemons | M | M | Non-inferiority single-page gate; one-default end state (D25) |
| R20 | Upstream fixes after the pin (the repo was 2 days old when pinned) | M | M | Pin; the upstream-drift check runs at M1 (U2-C2) and before publishing (U2-E4); changes re-enter via A1 |
| R21 | Streaming dies on CJK (the PageStream bug), and MarathonDocBench is half Chinese | H | M | U2-X1 plus token-id splitting (D20) |
| R22 | The quant matcher int8s the router/indexer projections | M | H | Exact-name matcher; a test asserting 0 router tensors under every lever (D7) |
| R23 | The global artifact version breaks v2 artifact resolution | H | M | Per-model version (D22) |
| R24 | bd-pfui changes storage and runtime while D4 lands | M | M | Checkpoints are sidecar files located by resume key, not RunStore blobs; the run store's schema is unchanged and gets no index row (D14) |

---

## 14. Open questions register (OQ-V2)

| ID | Question | Status | Resolution / owner |
|---|---|---|---|
| OQ-V2-1 | SR-SWA candidate semantics | RESOLVED | §5.6.4 |
| OQ-V2-2 | Router constants at inference | RESOLVED | §5.6 |
| OQ-V2-3 | Parity tolerance: f32 native vs the f32 oracle; bf16 ledger tolerance | PARTIAL: U2-C1 set the f32 gate (every stage of every layer at cosine ≥ 0.9999; coordinates, Page Gate and teacher-forced routing exact; chained flips only within 2× the layer's score noise; measured: 5 single-row tie-flips over S1–S4). The bf16 ledger tolerance remains | U2-A6 floor + U2-C1; bf16 at U2-C4 (`docs/truth-pack/OQ_INDEX.md`) |
| OQ-V2-4 | Does HF v2 run in f32 on CPU (dense SDPA path), and is a ported dense fallback needed? | RESOLVED (yes; no port: prefill `_forward_dense`, decode `_decode_attend`, deterministic algorithms on) | U2-A6 (`OQ_INDEX.md`) |
| OQ-V2-5 | Are the `indexer.*` projections used at inference? | RESOLVED (no) | §5.6.7, D8 |
| OQ-V2-6 | 1-D RoPE fallback | RESOLVED (franken builds `<PAGE>` prompts; no fallback) | §5.5 |
| OQ-V2-7 | n-gram vs guard order hazard | RESOLVED (the forced token wins) | DISC-U2-2 |
| OQ-V2-8 | Tokenizer handling of non-special added tokens | RESOLVED: the splitter handles them (U2-A4's 227-case golden and fuzz), and the `unlimited-ocr-2` arch pins the sidecar and its required ids (U2-A5) | Closed by U2-A5 |
| OQ-V2-9 | Windowed accuracy vs unwindowed | OPEN | U2-D2b |
| OQ-V2-10 | v2 single-page base vs gundam | OPEN | U2-G2 |
| OQ-V2-11 | Block-granular retirement is output-exact under the guard | RESOLVED (mechanism, 2026-10-29): retired and unretired synthetic S1–S3-shaped runs agree bit for bit; the detector passes only guard-forced open rows; pinned rows survive; fixture equality is U2-C2's | U2-B4 (`srswa.rs` cache tests) |
| OQ-V2-12 | Licenses of evaluation sets beyond MarathonDocBench (MPDocBench, OmniDocBench) | OPEN | U2-C5a |
| OQ-V2-13 | Does the conservative int8 recipe hold v2 CER within budget? | OPEN | U2-C4 |
| OQ-V2-14 | Document-mode crop mode: Baidu's long-document numbers are plausibly gundam (the SGLang default) | OPEN | U2-C5b resolves it, or defers its gundam cells to U2-G2, which runs them on C5b's long-document subset (same harness and invocation) and resolves it |
| OQ-V2-15 | Non-Baidu ground-truth documents for the head-to-head (sources, licenses, normalization) | OPEN | U2-C5a |
| OQ-V2-16 | Entmax domain for long documents (TopM vs AllCandidates) | OPEN | U2-C5b |

---

## 15. Review log

**Round 1 (2026-10-28).** Three independent, fresh-context reviews ran as Claude Code subagents:
- **Spec fidelity**, against the pinned Python and the SGLang wheel.
- **Codebase grounding**, against the repo and the existing beads.
- **Strategic**, using the planning-workflow review prompt.

**Spec fidelity: integrated.**
- **2 WRONG items:**
  - row-level retire-on-advance, now block-granular (§5.12);
  - the reference prefill numerics: Triton on GPU, dense on CPU (§5.7).
- **Router and prompt details:**
  - router input and dtype path (§5.6.1);
  - prompt variants and fallback triggers (§5.1, §5.5);
  - SGLang's gundam default (DISC-U2-6);
  - SGLang τ bisection (§5.6.6).
- **Decode-loop details:**
  - decode-loop details (§5.9);
  - extra-marker tok (DISC-U2-7);
  - the decode query-page initial value and the seeding rule;
  - the stateless-test precondition.
- **Windowing:**
  - the one-page splice variant;
  - text-tail coordinates (§5.10.4);
  - the exact fallback list (§5.10.5);
  - stitching ownership (§5.10.8);
  - window planning and capacity (§5.10.1, §5.10.7);
  - the guard lag (DISC-U2-8).
- **Numbers and citations:**
  - the gundam token range (Appendix A);
  - sizes in D8;
  - citation fixes.

I read the page_base parity rule directly from `unlimited_ocr_v2_backend.py:1001-1031` and
`uocr_doc_reuse.py:109-121`.

**Codebase grounding: integrated.**
- **Identity and dispatch:**
  - the full identity-hazard list (§6.1);
  - the decoder path matrix and lever refusals (§6.2, D18).
- **Weights, quant and tokenizer:**
  - the recipe matcher (D7);
  - I64 parsing (D8);
  - tokenizer sidecar validation (D2);
  - the global artifact version (D22);
  - the single `.focrq` `source_sha256` (D16).
- **Existing surfaces the plan had wrong:**
  - the existing `page` event and `decode_contract()`;
  - the generated-only n-gram domain (D10);
  - `selftest` is not decoder evidence (§7.6);
  - the watchdog has no multi-page scenario (§6.4);
  - `--multi-page --json` returns `pages: []` (§6.6);
  - RunStore is best-effort (D14).
- **Related beads:** Appendix D.

**Also noted.** bd-wp8.2.2 (the P0 robot-schema golden bug) appears stale.
`tests/fixtures/robot_schema_v1.json` already lists `staff` and `music_warning`, and
`tests/cli_robot_golden.rs:723-724` asserts them. Its owner should re-verify and close it; this plan
does not depend on it.

**Strategic: accepted P1–P16 and all honesty corrections, with these modifications.**
- **P6** (windowed oracle): the Python oracle must be written from the wheel source by an author
  other than the Rust implementer, so a misread §5.10 is not implemented twice.
- **P7** (non-Baidu evidence): concrete ground-truth sources are named, and sourcing is an explicit
  open question (OQ-V2-15) rather than an assumed input.
- **P12** (book inputs): the pure-Rust JBIG2/JPEG 2000 decoder evaluation is left to bd-0a7's owners
  as a linked follow-up. Candidate crates must be verified for maturity, license and no-FFI before
  adoption.
- **P13** (raw loads): confirmed against the code. v1 raw loads already hash nothing
  (`source_sha256` is empty for safetensors), so v2 follows the same policy.

**Round 2 (2026-10-28): bead polish.** After the plan was converted into 51 beads (epic
`bd-uocr2-epic-cgux` plus 50 children), an independent read-only reviewer checked every bead against
the code. It reported 33 findings, all integrated. Load-bearing claims were spot-verified in the
tree first. The material corrections:
- **Certified path (§6.2, D18).** The conservative artifact runs `generate_cached` →
  `prefill_with_cache`/`decode_step_with_cache` with int8 FFN inside `DecoderWeightCache`
  (`mod.rs:3736-3800`).
  - The `_i8` cached path serves only the experimental full-int8 switch and WASM int4, so v2 refuses
    it. U2-B5 is f32 cached plus stateless.
  - `recognize_multi_page*` gained a refuse row (A7) and is routed to the v2 document loop (B6).
- **Retirement inside windows (§5.10.9).** Unconstrained block retirement would free retention rows
  (a), (b) and (d). U2-D1a now sets a pinned set, and retirement is suspended on guard-forced steps.
  The regression case is W=8 window 0, marker row 1919.
- **Never truncate silently (D11).**
  - U2-B6 refuses unwindowed v2 documents over 60 pages with exit 2 until U2-D1b.
  - A position-limit stop gives `finish_reason: "length"` and exit 5.
  - Cancellation exits 6.
  - U2-E4 also surfaces v1's existing silent clamp (`mod.rs:2778-2785`).
- **Splits.**
  - U2-D1 → D1a (windowed core: B4, B6) and D1b (pipeline: D1a, D0, E2; W=10 default; lifts the
    60-page limit).
  - U2-D2 → D2a (independent Python oracle; deps A1, A6 only, so it runs in parallel and cannot copy
    the Rust code) and D2b (the parity gate).
- **Gate shape (§7.3).** Per-page pass/fail with per-category Jeffreys/Hoeffding bounds per
  RATCHET.md §4.2. Aggregate CER vs 0.193 is informational only. (Superseded in round 3: aggregate
  ≤ 0.25 is a hard gate.)
- **Ownership fixes.**
  - X1 names `postprocess::PageStream`.
  - E1 owns D25 and the mode-default table.
  - E2 migrates the `focr-ios` call site.
  - D4 owns stale-checkpoint reporting.
  - A4 publishes pins only (A5 depends on it).
  - C1 covers prefill rows only (decode steps are in C2).
- **Dependency edges.**
  - Added: E5a ← T1b, T2, E3, F1; E6a ← F1, C5a; E4 ← E1; F1 ← C3; F4 ← D4; G1 ← C4; H1 ← E2;
    B6 ← A6; B7 ← A6, T3.
  - D3, D4, D5, F3, F5 and T2 moved from D1 to D1b.
  - §12 was regenerated from the live graph: 116 blocking edges; `br dep cycles` shows only the two
    cycles that predate this plan.

**Round 3 (2026-10-28): bead polish round 2.** A second independent read-only reviewer checked
self-containment, test and logging ownership, plan features lost in conversion, value to users, and
dependency sanity. It reported 39 findings. The load-bearing ones were verified in code before
integration; one was corrected rather than taken as written:
- **C4 could not fail (§7.3).** A per-page 0.25 budget would fail v1 itself. A new ratchet category is
  admissible by definition, and n = 20 forces fallback bounds. So the gate now:
  - pre-registers a per-page budget;
  - runs a same-invocation v1 baseline through `ratchet_decide`;
  - keeps aggregate ≤ 0.25 as a hard gate.

  C4 also captures v2 goldens, which give U2-T3 a v2 lane that every later v2 change must keep green.
- **v1 page merging.** Verified in `tests/fixtures/multi_page/p20_meta.json`: 20 pages, 7 markers,
  ended by EOS with no clamp.
  - The reviewer proposed counting "N−1 markers". I corrected that to `finalize_multi`'s body count:
    one body per marker plus a non-blank leading chunk (GH #17). The p10 fixture's 9 markers can be
    complete.
  - New M0 bead **U2-X2** reports truncation and merging for v1 now, via `run_complete.finish_reason`
    and `warnings[]`. That replaces every undefined "robot note" (§9.2), and it fixes v1's help text.
  - U2-E3 emits v1 `pages[]` only when the body count is N. U2-C5a/C5b score document-level edit
    distance first.
- **Robot contract.** `robot_schema()` has no field-level description, so U2-D5 adds `event_fields`,
  and the e2e steps validate against it.
- **Windowed figures.** Rasters are dropped before decode, so U2-D0 adds `PageSource::fetch` and
  per-page sizes. U2-D1b re-fetches figure pages, and U2-T1b gains a figures step.
- **Resume journal.** U2-D4's `.partial` is an NDJSON `PageResult` journal. For stdout runs it lives in
  `document-runs/`, because doctor quarantines `*.partial` under `models/` (`doctor.rs:246-285`).
- **Fail-closed completeness.** Twelve-plus `FOCR_*` levers were missing from U2-A7's matrix (verified
  by grep). U2-A7 now classifies every lever in a `V2_LEVER_POLICY` table, with a census test that
  scans `src/`.
- **Smaller changes:**
  - sharded-directory tokenizer resolution (A3);
  - eager tokenizer validation before weights load (A2);
  - the test-log schema enums (C1) and the TEST_LOGGING contract in T1a/T1b;
  - the `EntmaxDomain` selector (B6);
  - `ModelArch::maturity()` (B6, E1, C4);
  - directory input in the mode table (E1, E5a);
  - a first-run download that follows the table (E1);
  - the 60-page limit arithmetic, re-derived by C4;
  - R20 drift checks (C2, E4);
  - named S2/S4 fixtures (A6);
  - wheel-file citation keys (D1a).
- **Edges.**
  - B3 depends on B1, not B2.
  - E2 depends on E3, which owns `PageResult`.
  - E6a no longer waits on the 15–40 CPU-hour C5a evaluation.
  - B6 and E4 depend on X2.
  - E7 depends on T1a and adds its own e2e step. The reviewer proposed the reverse edge, but that put
    E7 on the critical path.
- **Persistence.** The 307-line SGLang windowing spec existed only in the session scratchpad. It is
  now attached to U2-A1 as a `br` comment.
- **Result.** 52 beads, 120 blocking edges, longest chain 14 (§11). `br dep cycles` still shows only
  the two cycles that predate this plan.

**Round 4 (2026-10-28): fresh-eyes review.** Three independent, read-only reviewers ran in parallel:
- **Spec fidelity** of the decoder, windowing, oracle and gundam beads and §5, against the pinned
  Python and the SGLang wheel (19 findings);
- **Codebase claims** in all 52 beads and §6/§9/Appendix C (50 findings);
- **Consistency, completeness and ceremony** across the beads and the plan (42 findings).

About 110 findings in all. The high-severity ones were verified in the code before they were applied:
the int8 FFN inside `DecoderWeightCache`, all three multi-page fixtures opening with `<PAGE>`, the
exit-code map, HTTPS-only part URLs, the DISC heading regex, doctor's mutation laws, the
success-terminal `run_complete`, the outer timeout discarding partial output, and the wheel's
reused-window carry and `end_on_marker`.

What changed:
- **Spec (§5).**
  - The reused-window decode seed (§5.4, §5.5, §5.10.4).
  - `nblk` per forward, and padding exact only when merged (§5.6.5).
  - The AllCandidates read set (§5.6.6).
  - The last-window `<PAGE>` ban reaches only recomputed windows (§5.10.7), plus two wheel
    configuration differences.
  - The recompute-equivalence reference (§5.10.5); fallbacks (iv) and (v) are unreachable, so tests
    use a hook.
  - Page state is replayed, not copied (§5.10.3). `ctx` is the server's input limit (§5.10.1). The
    one-page variant is confirmed.
  - Fallback rows are 97–128 image tokens per page, not "half" (D6, §5.6.4).
  - The retirement window end is stated exactly (§5.12).
  - DISC-U2-4 is a drop, not a reorder, and **DISC-U2-9** (leading text) is new.
- **Native paths (§6.2, §7.2).** There is no all-f32 cached path: `DecoderWeightCache` keeps the FFN
  int8 even from BF16. The f32-oracle rungs run on the stateless f32 path. The cached path is
  L4-checked and L5-certified. "Stateless ≡ cached" became "re-prefill ≡ incremental" on one path.
- **Product surface (§9).**
  - A nonzero exit always ends in `run_error`, now carrying partial-output fields.
  - Budgets are checked cooperatively, so completed pages survive.
  - `warnings[]` uses the `recovery` key, merges into the existing `--json` array, and has a canonical
    kinds list.
  - `event_fields` moves to U2-X2 (M0), and the `page` field additions to U2-E3 (M2).
  - `FOCR_TEST_MAX_POSITIONS` lets CLI tests reach the limits.
  - Doctor reports `RefusedUnsafe` instead of writing a `.bak`.
  - Loader recipe refusals are exit 7, and lever refusals are a new pre-load `Usage` check (exit 2).
  - Unreachable levers are ignored with a warning instead of refused.
- **Gates.**
  - C4's v1 baseline uses v1 **f32** references (U2-A6's S1-corpus row), so both lanes are symmetric.
  - The page limit re-derivation uses p90 plus one p99 tail, not per-page p99, and
    `Some(PageWindow::Off)` bypasses the refusal.
  - R20 runs in C2 and E4, not C4.
  - DISC-U2-k entries are filed under numeric `DISC-NNN` headings, so `check_ledgers.py` enforces them.
- **Ownership and process.**
  - D14's journal and checkpoint need no run-store index or schema change.
  - E2 owns the CHANGELOG migration table and the iOS call-site move (focr-ios holds `OcrModel`).
  - U2-D2a is ported from the wheel source before U2-D1a and is not read against our spec.
  - T1a (a)/(b) run only under `FOCR_E2E_LONG`, (i) only under `FOCR_E2E_NETWORK`, and new step (r)
    covers `ocr-batch`.
  - `V2_LEVER_POLICY` was renamed `UOCR2_LEVER_POLICY`, because "v2" already names the manifest
    schema, the focrq header and the wasm artifact.
  - Citations were corrected throughout §6/§9 and Appendix C.
- **Edges.**
  - B4 depends on B3, whose `BlockKv` view it implements.
  - D1a depends on D2a, so the oracle exists first.
  - F4 depends on D2b, whose subset it reuses; G2 depends on C5b, whose long-document evidence it
    reads.
  - None lengthens the 14-bead critical path. The review of these edits (round 4b) added A7 → X2 and
    D0 → X2.
- **Rejected.**
  - The claim that D0, D1b and C5b are blocked by bd-paom, bd-3gyt, bd-0a7, bd-0a7.4 and bd-322.18
    (`br show` lists them as `related`).
  - Moving E2 after D1b: D1b integrates through E2's `recognize_document`.
- **Deferred to the user.** An earlier `--multi-page` default flip for documents within the page
  limit, on C5a evidence at M2. It would split default behavior by page count.

**Round 4b (2026-10-29): review of round 4's edits.** Three independent, read-only reviewers read the
round-4 diffs (`round3_beads.diff`, `round3_plan.diff`) for code claims (16 findings), spec fidelity
(14) and consistency (25), about 55 in all. The load-bearing code facts were re-verified before they
were applied: `RunawayGuard` built at `mod.rs:3727` and checked at `:3830`/`:4284`;
`lmhead_shard_enabled` presence-based at `decoder.rs:2115`; `p_new` counting lookahead (wheel `:260`);
HF prefill nblk from the forward's query count (`MD:436-445`).
- **Levers (D18, §6.2).** `FOCR_RUNAWAY_GUARD` reaches both v2 paths today, so it is ignored only once
  U2-B5/B6 stop building it on the v2 arm. Presence-based levers are classified by presence.
  `FOCR_INT8_ATTN`/`_LMHEAD`, `FOCR_FUSE_NGRAM_LMHEAD`, `FOCR_SPEC_DECODE`/`_VERIFY` and
  `FOCR_FUSE_NORM_QUANT` are ignored (they reach only `_i8` or full-int8 code), and
  `FOCR_LMHEAD_SHARD`/`_TILES` are refused until U2-C2.
- **Spec (§5).** The one-page splice applies to the last window only. HF prefill nblk is defined on
  an empty cache, with franken_ocr's rule for extends over a reused window. The pad condition is
  sufficient, not iff, and the re-prefill precondition adds "no candidate x ≤ −1e4"; chunked ≡
  unchunked selection is unconditional. `cnt[]` replays the rewritten prompt only. `inv_tok` is
  asserted against the oracle's dump. The last-page ban rationale is `end_on_marker = (lookahead > 0)
  || reused`. `ctx` subtracts the wheel's −1/−5 margins. The streaming handler's end-of-window
  `select_batch_output` drops leading blanks too. A recomputed last window's reference runs with
  end_on_marker = false.
- **Product surface (§3, §4, §9).** `PageWindow`'s default (`None`, limit enforced) is distinct from an
  explicit `Some(PageWindow::Off)`. v1 `page` events are unchanged; the new fields are v2-only. The
  stop producers are named (B6 `length`/`budget`, D1a per-window `budget`, E2 `cancelled`, D4
  checkpoints). D4 owns `--resume`/`--keep-checkpoint`. `DocumentResult` gains `warnings`. The
  `event_fields` extender list is corrected. The model-resolution tiers already exist (bd-av64.7), and
  `detect_model_not_resolvable` checks only the default spec.
- **Gates (§7, §11).** The M1 exit separates the stateless f32 L0–L4 gate from the cached path's L4
  (token-exact, or a root-caused int8 FFN DISC). Retirement is owned by B4 (synthetic), C2 (fixtures)
  and D1a (pinned set). D2a reads only the pinned wheel source and runs its own loop for W ≥ N. e2e
  gating is per step, every T1b step runs under `FOCR_E2E_LONG`, and (m″) is sized from the page limit.
- **Edges.** A7 → X2 and D0 → X2 (both register warning kinds in X2's `event_fields`). Neither
  lengthens the critical path.
- **Citations.** `cli.rs:2684`, `cli.rs:1704/1707`, `mod.rs:2778-2785`, final flush `:2821`,
  `_install_f32_patches` `:324-347`, `MU:561,566`, `cli_robot_golden.rs:1365-1613`,
  `quant_ffn_loaded` → `quant_oc_loaded`.

**Not yet done.** No GPT Pro or other non-Claude review round has run. The planning workflow
recommends 4–5 rounds with GPT Pro Extended Reasoning. The exact prompt is in the planning-workflow
skill ("Carefully review this entire plan…"); paste this file after it.

---

## Appendix A — Token and image counts

| Mode | Image tokens per page | Prompt tokens per page incl. `<PAGE>` |
|---|---|---|
| Base 1024 | 273 | 274 |
| Gundam 2×1 (minimum tiled) | 273 + (21)·10 = 483 | 484 |
| Gundam 1×2 | 273 + (11)·20 = 493 | 494 |
| Gundam 2×2 | 273 + (21)·20 = 693 | 694 |
| Gundam 4×4 | 273 + (41)·40 = 1,913 | 1,914 |
| Gundam 1×16 (maximum) | 273 + (11)·160 = 2,033 | 2,034 |
| v1 multi-page (640 squash) | 111 | 111 (+ separator handling per v1) |

Base markers sit at positions 1 + 274·p (p 0-based), and 274 ≡ 18 (mod 32), so page boundaries are
never block-aligned (§5.12).

## Appendix B — v2 source-file SHA-256 (commit 5632329a)

```
config.json                   607d130a70d5af5a35329aa237acd68651f45504c62cc7ced981066f4e781126
generation_config.json        dcd0479f3a12e5e4413d956197f8a4074408e4901834b38c7e036519134e05af
configuration_deepseek_v2.py  b8470dd616ba8745fce6e27b093aef73a098863cc891b2477dcf9326a36000f7
conversation.py               ec7b6ce89bcda643de1f43269ffa66a7b2e65dc3ed30e427958f776546b4ba03
dash_attention.py             960b460acd816df8e561e73dccd5683acb95445eddc723c4f0d6ed1f462b2f77
deepencoder.py                419b63388c1645b9b72c716070cdca36cd0be6047049424dc75941d3a00aaf1f
modeling_deepseekv2.py        24a15dfd47578ab24a7a4d4e977dc5d7a23dfef0a2a4c4bf73fcf3e2ce7acaf2
modeling_unlimitedocr.py      f6df813cef23cc28b1daca40fd8cb20740d05a238e7b9dfa2334531fdcf39f1f
_sparse_base_impl.py          86da9c5638704bb9338575040568a97b3e10820bbeb426f51c07e866f3423ccd
packing_patch.py              e60d842c210c9f1d6cd2ae328cf9127a75f7941a0cea2a8e9abc14662f770c6c
sparse_gather_triton.py       c3de717ffaca40240c665b8d645f15a9d1999feeabaa94beb35caad877c2f0f4
sparse_gather_e2e_triton.py   58f1bc46093d47b00744970677e66bfbd7485a2d13810f46d13d867ceda13dd4
special_tokens_map.json       9e89493d7938e7710c524dbaff613c3648f0a018262417af78505128004e5cb4
tokenizer.json                c55c165e32c5a67830362a53e1f548c8930dcc8469efa30ff3252a982799c589
tokenizer_config.json         ceba7d221f265cca4c98b5e0a1d48c10107270b7b29c12396610d6d0fd64b126
model.safetensors.index.json  dc60a31ecb77eb03e166345d18cbe25bc71d2a19cc6fc089d8a9ac1c7f209671
Unlimited-OCR-2.pdf           1eb5885f4ddfa4ec6fbff375092377f30b5d6a5e034ec4936213f125bf2f9444
```

## Appendix C — Code locations touched (current tree, 2026-10-28)

| Area | Location |
|---|---|
| Decoder constants and RoPE | `src/native_engine/decoder.rs:82-109` (`ROPE_THETA` :98; RoPE table :157-238) |
| Decoder paths | `decoder.rs`: `prefill_attention` :577 (shared with `decoder_qwen2.rs:1266,1270`); stateless `forward` :699; `DecoderWeightCache` :812 (`build` :837 keeps FFN int8; `quant_ffn_loaded` :1928 → `quant_oc_loaded` :1838); `chunk_prefill_attention` :1198; `prefill_with_cache` :1272 (delegates to `prefill_with_cache_chunked` :1287; monolithic branch :1374-1400); `decode_step_with_cache` :1428-1502; `DecoderWeightCacheI8` :2326; `prefill_with_cache_i8` :2826/:2842; `decode_step_with_cache_i8` :2956; `batched_decode_step_i8*` :3211/:3243; `verify_forward*` :3495/:3575 |
| MoE constants and router | `src/native_engine/moe.rs:53-76,118-120,482-500` |
| R-SWA cache and attention | `src/native_engine/rswa.rs:46-50,180-219,484-624,997-1007` |
| Generation loops and levers | `mod.rs`: `forward` :2105; `forward_dynamic` :2137; `ensure_arch_implemented` :2376; `forward_pre` :2394; `generate_cached` :3773; `generate_cached_i8` :3887; `spec_decode_i8` :4064 (reached only via `generate_cached_i8`); `generate_stateless` :4246; `PagePrefill` :1926; `recognize_batch_spine` :2984; `batch_scheduler.rs:43,440,487` |
| Multi-page path | `mod.rs:2602-2830` (32K checks :2750-2783; streaming observer in `recognize_multi_page_pres` :2723, `stream_broken` :2802-2811, final flush :2821; `PageStream` is `postprocess.rs:531`, doc `:517-529`); `build_prompt_multi` :4399-4419; `MULTI_PAGE_BASE_SIZE` 640 :358; `cli.rs:1821-1862` (`recognize_pdf_multi_page`, eager raster + clone); `cli.rs:1386-1408` (three refusals); `cli.rs:1911` (`pages: []`); `lib.rs:282` (`stream_pages`); `lib.rs:703-802` (wrappers) |
| Sampler | `src/native_engine/sampler.rs:27-45,514-640`; n-gram history includes prompt ids (`mod.rs` ~3829, ~3932) |
| Postprocess | `src/native_engine/postprocess.rs:38,497-597` (`finalize_multi` keeps a non-blank leading chunk, GH #17) |
| Tokenizer | `src/tokenizer/mod.rs:86-119,489-532`; sidecar `mod.rs:3268-3285` |
| Identity | `weights.rs:60-110,174,197-205,898,1238,2086`; `focrq.rs:588-606`; `model_arch.rs:143-257,545-571`; `mod.rs:333,857,1022,1221-1231,1416-1473,1573,1699,2626,2663,2701,2851` |
| Census | `src/native_engine/unlimited_ocr_census.rs:24-55,110,328-394`; `unlimited_ocr_manifest.json` |
| Convert and recipe | `src/quant/convert.rs:46,60,124-178,181,254,292,392`; `src/quant/recipe.rs:122-129,195,236,332`; `cli.rs:686,2332,2350,2420` |
| CLI surfaces | `cli.rs:34` (constant), `:130,230,349`, `:409-437` (override forwarding), `:818` (`RobotCmd::Triage`), `:1045,1578,1594` (JSON `warnings`), `cli.rs:1650,1686,2675-2711,2949,3036,3159,4108`; `robot.rs` (`EVENT_KINDS`, `page_decoded_event`, `page_skipped_event`); `doctor.rs:121,142,186,246-285`; `storage.rs:31,34,54,68` (`SCHEMA_VERSION`, `FOCR_RUN_STORE`, `model_version_tag`, `RunStore`) |
| Distribution | `src/dist.rs:35,93-185,485-520,~1103-1160`; `models/manifest-v2.json`; `lib.rs:74,89-96` |
| Preprocessing | `src/preprocess/mod.rs:47-75` (`MAX_NUM` 32 :66), `141-144,251-283,350-416`; `connector.rs:403` (`fuse_no_crop`) |
| Tests to extend | `tests/many_pages_without_deadlock.rs`; `tests/surface_matrix.rs`; `tests/cli_robot_golden.rs:723-724,768,894-931,1143,1365-1613,2141,2162` (health golden at 894-931, exit-code streams at 1365-1613); `tests/fixtures/golden/robot_health.golden`; `tests/fixtures/MANIFEST.toml`; `tests/fixtures/robot_schema_v1.json`; `parity_ladder.rs:130`; the spec/batched/chunked parity tests |
| Oracle scripts | `scripts/gen_reference_fixtures.py:92-125` (pins and modes), `:225-245` (flags), `:292-321` (`_install_cpu_patches`), `:324-347` (`_install_f32_patches`; `Tensor.to` wrapper `:338-347`); `oracle_bridge.py`; `oracle_nondeterminism_floor.py`; `check_oracle_provenance.py` (`PIN_TORCH`/`PIN_TRANSFORMERS` :28-29); `check_ledgers.py:310` (DISC heading regex); `tests/fixtures/PROVENANCE.md` |
| Other targets | `focr-ios/src/lib.rs:280-284,394,401,494,1152`; `ios/Sources/{LabView.swift:453,768, ModelStore.swift:106}`; `site/model-manifest.js:44`; `site/functions/model/[[path]].js:35`; `focr-wasm/src/lib.rs` |

## Appendix D — Related existing beads

| Bead | Status | Relation |
|---|---|---|
| bd-3jo6 Model Zoo epic | open | The U2 epic is a zoo addition (related) |
| bd-3gyt consolidate PDF page walks | open | U2-D0 implements the multi-page part over `PdfPages::render`; bd-3gyt re-expresses `pdf::walk_document` on the page source (related, not blocking) |
| bd-paom `recognize_pdf` | open | U2-E2's per-page sibling over the same walk |
| bd-pfui asupersync 0.5 storage/RunStore | in progress | U2-D4 avoids conflict by using sidecar checkpoints |
| bd-2mo.30 hotspot campaign (vision statics cache: child bd-2mo.30.11) | in progress | U2-B5/F3 reuse the vision path; coordinate |
| bd-2mo.28 int8/int4 KV | open | U2-F4's bf16 KV is the v2 analogue |
| bd-wp8.2 (freeze robot schema), bd-wp8.3 (streaming transport), bd-znwm (ocr-batch streaming), bd-mgd3 (per-page progress) | open | U2-D5/E1 robot changes go through these |
| bd-hfdt-admission-prereqs-dfr1, bd-json-per-span-text-wbbu, bd-v73v | open | U2-E3 structured output |
| bd-0a7 / bd-0a7.4 native PDF and vector rasterization | open | Book inputs (D24); the decoder evaluation lives there |
| bd-3p8y, bd-aio5, bd-we84, bd-b8cg (HF mirror); bd-jprb (atomic pull) | open | U2-E4 |
| bd-3ryl task-vs-arch guard | open | U2-A7 is the v2 case |
| bd-fuh2 resident daemon follow-ups | open | The daemon refuses v2 until its protocol carries the v2 flags |
| bd-1azu.12 (shared-prefix KV), bd-1azu.34/38/40/41/44 (spec decode/spine), bd-1azu.55/56 (megakernel) | open | Levers that refuse v2 (D18) |
| bd-5k7a iOS multi-page | in progress | U2-H1 |
| bd-322.18 40+ page benchmark cells | open | U2-C5b evidence |
| bd-10sb testing infrastructure epic | open | U2-T1a/T1b/T2/T3 |
| bd-wp8.2.2 robot schema golden (P0) | open, appears stale | Not a dependency |
