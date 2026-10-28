# Unlimited-OCR-2 in franken_ocr — Upgrade & Support Plan

> **Status:** planning (no code landed). **Created:** 2026-10-28. **Scope owner:** franken_ocr maintainers.
> **Companion to:** [`COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`](COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md) (the master plan; its doctrine binds this plan).
> **Work graph:** this plan is mirrored 1:1 into beads under the epic titled
> "Unlimited-OCR-2: additive model + unlimited-horizon long-document parsing" (label `uocr2`). Once the
> beads exist they are the executable source of truth; this document is the rationale and the spec.

---

## 0. How to read this plan

- §1 is the one-page summary.
- §2 is the verified ground truth about the new model. Every number in it was checked against the
  pinned source, config, weight headers or tech report on 2026-10-28.
- §3 explains why the plan is shaped the way it is, from the user's point of view.
- §4 lists the load-bearing decisions, each with its rationale.
- §5 is the **reference-semantics specification**. It is the part an implementer needs, and it is
  written so nobody has to reopen the Python.
- §6 is the franken_ocr architecture for the port.
- §7–§10 cover verification, performance, product surface and default policy.
- §11 is the phased roadmap with exit gates, §12 is the task graph (mirrored into beads), §13 lists
  risks, §14 is the open-questions register, and the appendices hold numbers and pins.

Terminology:
- **v1:** `baidu/Unlimited-OCR` at HF commit `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`, the model
  franken_ocr runs today.
- **v2:** `baidu/Unlimited-OCR-2` at HF commit `5632329a32e103d5f1add270681e0f5c1233c6e4`.
- **R-SWA:** v1's Reference Sliding Window Attention.
- **SR-SWA:** v2's Sparse Reference Sliding Window Attention. Baidu's code calls it "DASH".
- **Page Gate / Token Sieve:** the two stages of SR-SWA, named as in the tech report.
- **Window:** one chunk of the unlimited-horizon (chunked) decoding scheme in §5.10.
- **HF reference:** the transformers `infer()` path in the v2 repo.
- **SGLang reference:** the customized SGLang wheel shipped in the v2 repo.

---

## 1. Executive summary

**What v2 is.** v2 keeps v1's exact backbone: the same vision tower, projector, MoE decoder shapes,
vocabulary and tokenizer model. The vision tower and projector weights are **byte-identical** to
v1, because they were frozen during training. Three things changed:
1. **Attention.** Every decoder layer's attention is now SR-SWA. Each query attends to a 128-token
   local window plus up to 12 router-selected 32-token blocks, drawn only from **its own page's
   image tokens**. That keeps per-step attention bounded at about 512 tokens, regardless of
   document length.
2. **Positions.** RoPE is now two-axis: an intra-page token position plus a page-parity bit.
3. **Decode contract.** Prompts and outputs use a `<PAGE>` token (id 128827) for every page,
   including single pages. A page-count guard makes the output contain exactly one `<PAGE>` per
   input page.

**What it buys users.**
- **Single pages:** essentially nothing in accuracy. On OmniDocBench v1.6, v1 scores 93.92 and v2
  94.02; tables are slightly worse.
- **Multi-page documents:** a large gain. On MPDocBench (up to 30 pages), v1 scores 61.89 and v2
  84.59. Cross-page text edit distance drops from 0.605 to 0.100.
- **Hundred-page documents:** a new capability. v2 scores 77.06 on MarathonDocBench, which v1
  cannot run at all.
- **Books:** a path to arbitrarily long documents with bounded memory, via chunked "unlimited-horizon"
  decoding. Baidu ships that only for GPU SGLang serving.

**The plan in one paragraph.**
1. Land v2 **additively**, as a new zoo model `unlimited-ocr-2` next to v1.
2. Harden model identity first, so that no existing v1 artifact can ever be misread as v2.
3. Port SR-SWA, page2d RoPE and the v2 decode contract by parity-gating against the HF reference.
4. Certify a conservative int8 artifact using the same recipe policy as v1.
5. Then build the user-facing differentiator: **long-document parsing on a laptop CPU with bounded
   memory, page-by-page streaming and crash-safe resume**, implementing Baidu's windowed decoding
   semantics natively.

Default model choices change **per mode, only on ground-truth evidence, with the incumbent measured
in the same invocation**. The expected outcome is that `--multi-page` moves to v2. The single-page
default moves only if v2 is at least as accurate and no slower on CPU.

**What we are explicitly not doing:**
- replacing v1 in place;
- copying SGLang's serving internals (radix cache, prefetch, Triton kernels);
- GPU paths;
- training;
- iOS/WASM v2 lanes in the first wave.

---

## 2. Verified ground truth about v2

### 2.1 Provenance and license

| Item | Value |
|---|---|
| HF repo / commit | `baidu/Unlimited-OCR-2` @ `5632329a32e103d5f1add270681e0f5c1233c6e4` (created 2026-10-26) |
| Weights | `model-00001-of-00002.safetensors` (4,998,430,312 B, sha256 `4c40b740f209fd7682040d4ebf19409208acde5c401ccd2dde426d0f41542f18`) and `model-00002-of-00002.safetensors` (1,689,875,264 B, sha256 `9490b82fa47f89e5110eb5577f8ab1b6694c12121905dbc086ddb10207a4ba0c`). `model.safetensors.index.json` `total_size` = 6,687,958,536 |
| Tensors | 2,819 (2,818 BF16 + 1 I64 buffer) |
| License | MIT. GitHub `baidu/Unlimited-OCR` `LICENSE` reads "MIT License, Copyright (c) 2026 Baidu", and the HF metadata says `license: mit`. **The HF v2 repo itself ships no LICENSE file.** Record that in the provenance doc; the notice we redistribute is the GitHub text. |
| Reference stack (model card) | Python 3.10, `torch==2.13.0`, `transformers==4.52.1`, `tokenizers==0.21.4`, `huggingface-hub==0.36.0`, CUDA (tested on A800). **This differs from our v1 oracle pins** (torch 2.10.0, transformers 4.57.1). |
| Serving reference | `wheel/sglang-0.5.20.dev10+g8d62f5fb9c-py3-none-any.whl` (sha256 `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d`) |
| Tech report | `Unlimited-OCR-2.pdf` (sha256 `1eb5885f…9444`). The arXiv id in the card is v1's `2606.23050`. |
| Benchmark | `baidu/MarathonDocBench` (dataset, Apache-2.0): 100 documents of 100–300 pages (~18k pages, ~205k layout elements), zh/en, scored with the MPDocBench-Parse toolkit |

Source-file SHA-256 pins (v2 commit) are in Appendix B.

### 2.2 Architecture delta (v1 → v2)

| Component | v1 | v2 | Evidence |
|---|---|---|---|
| Decoder dims | 12 layers, hidden 1280, 10 MHA heads, head_dim 128, layer-0 dense (6848), layers 1–11 MoE 64 routed + 2 shared, top-6, moe_intermediate 896, vocab 129280, `use_mla=false` | **Identical** | config diff |
| Vision tower (SAM-B → conv compressor → CLIP-L/14), projector 2048→1280, `image_newline`, `view_seperator` | — | **Byte-identical weights**: sampled SAM/CLIP/projector/newline/separator tensors compared byte-for-byte. The report §4.2 says the DeepEncoder and aligner were frozen. `deepencoder.py` differs only in comments. | range-read header and data compare |
| Decoder weights (attention, experts, router gate, embed_tokens, lm_head) | — | **Retrained.** Sampled 64 KiB slices differ in 22–98% of bf16 elements. | same |
| New tensors (per layer, ×12) | — | `self_attn.dash_router.{q_proj [128,1280], k_proj [128,1280], q_norm [128], k_norm [128], head_cls [128]}` and `self_attn.indexer.{q_proj [128,1280], k_proj [128,1280], q_norm [128], k_norm [128]}`, all BF16 | header |
| New buffer | — | `model.vision_model.embeddings.position_ids` I64 [1,257] (a constant arange) | header |
| Attention | R-SWA: frozen reference prefix plus a ring of 128 generated tokens; every reference token is attended every step | **SR-SWA (DASH)**: block-sparse, ≤ 128 local + 12×32 remote tokens per query, in both prefill and decode | §5.6–5.8 |
| RoPE | 1-D, theta 1e4, no scaling | Main and router attention: **page2d** = 63 intra-page token pairs (theta 1e4, "resample" spacing) + 1 page-parity pair (page mod 2, theta 256). `rope_theta=5e5` with YaRN ×4 appears in config, but only the 1-D fallback path uses it. | §5.5 |
| Context | `max_position_embeddings` 32768 | 131072 (positions only index blocks and causality; RoPE uses page-local coordinates) | config |
| Tokenizer | BPE vocab 128000 + 830 added (max id 128826) | Same BPE model, pre-tokenizer, normalizer and decoder. **+4 added non-special tokens**: `<PAGE>`=128827, `<BLOCK>`=128828, `<FORK>`=128829, `<CHILD>`=128830 (the last three are unused by the inference code). tokenizer.json sha256 `c55c165e…c589`. | tokenizer diff |
| Prompt | single page: BOS + `<image>`×n + "document parsing."-style task text; multi: BOS + images + "Multi page parsing." | **Always** BOS + (`<PAGE>` + image tokens) per page + "Multi page parsing.", even for one page | §5.1 |
| Multi-page image size | each page squashed to 640² (111 tokens per page) | each page letterboxed to **1024** (273 tokens per page) | §5.2 |
| Single-page default view | franken_ocr uses base 1024; HF v1 `infer` defaults to gundam with up to 32 tiles | HF v2 `infer` defaults to **gundam with up to 16 tiles**; base is available | §5.2 |
| Decode guards | n-gram 35 (128 single / 1024 multi) | n-gram is *off* by default in HF `infer`; Baidu's evaluation uses 35/128/1024. A **page-count guard** is on by default (EOS banned until N `<PAGE>`, 20,000 tokens per page cap). | §5.9 |

### 2.3 Benchmarks (from the tech report; not reproduced by us)

| Benchmark | Metric | v1 | v2 | Context |
|---|---|---|---|---|
| OmniDocBench v1.6 (single page) | Overall ↑ | 93.92 | 94.02 | TextEdit 0.042→0.038, FormulaCDM 95.79→96.29, TableTEDS 90.16→89.60, ReadOrderEdit 0.129→0.128. Leaders: TeleOCR 96.91, OvisOCR2 96.47. |
| MPDocBench (≤30 pages) | Overall ↑ | 61.89 | 84.59 | TextEdit 0.167→0.052; T-TextEdit 0.605→0.100 |
| MarathonDocBench (100–300 pages) | Overall ↑ | n/a (v1 is limited to about 40 pages) | 77.06 | General VLMs score 7.5–33.7 |
| Per-step decode latency at 100 pages | — | — | 5.6× faster than R-SWA, 7.0× faster than full attention | GPU, gundam, batch 3 |
| Peak KV at hundred-page scale | — | — | 47.00 GB unwindowed → 1.77 GB windowed | GPU; windowing = §5.10 |

**Takeaway for this plan:** v2's value is **cross-page structure and length**, not single-page
accuracy. Every default-policy decision in §10 follows from that.

---

## 3. User-value analysis

### 3.1 Who uses franken_ocr, and what v2 changes for each

| User | Today (v1) | With this plan |
|---|---|---|
| **Single-page CLI and agent users** (`focr ocr page.png`) | v1 base mode, certified | **Nothing breaks.** v2 is optional via `--model unlimited-ocr-2`, at roughly equal accuracy and cost. The default changes only if the ground-truth head-to-head (§10) says v2 is at least as good. |
| **Multi-page PDF users** (5–50 pages, `--multi-page`) | Cross-page pass capped at 32K context (~290 pages at 111 tokens per page); weaker cross-page structure | v2 multi-page brings markedly better cross-page text and tables (MPDocBench +22.7). It also adds **per-page structured output** (JSON layout and figure extraction per page, which `--multi-page` cannot do today), made possible by explicit `<PAGE>` segmentation. |
| **Book and archive digitizers** (100s–1000s of pages) | Not possible | **New capability:** windowed unlimited-horizon parsing on CPU with bounded memory, streamed page by page, resumable after a crash or Ctrl-C |
| **Rust embedders** | `recognize_multi_page*` (whole document, blocking) | A document-level API with streaming callbacks, window and resume options; existing calls keep working |
| **Agents (robot mode)** | `run_complete` and stage events | Additive `page` and `window` events, a model id in `run_complete`, and `robot triage` recommending v2 for long documents. The schema stays versioned and additive. |
| **Holders of v1 artifacts** | `unlimited-ocr.v0.7.0.int8.focrq` | Keep working forever, bit-identically. No forced downloads; v1 stays pullable. |

### 3.2 Principles that follow

1. **Additive, never destructive.** v2 is a new zoo entry. v1 keeps byte-identical behavior,
   enforced by a gate (§7.6). The work graph treats any v1 output change as a regression.
2. **Evidence-gated defaults, decided per mode.** Single-page, per-page PDF, cross-page multi-page
   and batch are separate decisions. Each flip needs ground-truth evidence with v1 live in the same
   invocation. A self-speedup or self-parity result is not evidence of user value.
3. **Reference fidelity first.** The HF `infer()` path is the parity oracle for unwindowed
   decoding. Windowed decoding follows the SGLang reference semantics and is validated by
   degenerate equivalence, invariants and accuracy (§7.5).
4. **Ship value incrementally.** Every milestone (§11) ends in something a user can run.
5. **CPU-honest performance.** v2 is not a single-page speedup on CPU, and the plan says so. The
   performance mandate is "no single-page regression; bounded long-document cost," measured the way
   the perf ledger requires.
6. **Agent ergonomics are part of the product.** New capabilities get robot events, schema
   entries, surface-matrix rows and `doctor` and `triage` awareness in the same milestone.

### 3.3 Non-goals (first wave)

- GPU, CUDA, SGLang or vLLM serving.
- Training or fine-tuning.
- Reproducing SGLang's radix cache, prefetch or Triton kernels.
- v2 on iOS, WASM or the website. That is milestone M5; it needs its own int4 calibration.
- Removing v1. Retirement is a separate, later decision, taken only if v2 dominates in every mode.

---

## 4. Key decisions (with rationale)

**D1. Add `unlimited-ocr-2` as a separate zoo model rather than replacing v1 in place.**
*Why:*
- Single-page accuracy is a wash, so an in-place swap risks regressions (table TEDS is slightly
  lower) for no gain in the most common mode.
- The artifact-identity machinery currently treats "absent model_id" as "the default arch." An
  in-place swap would silently load every existing v1 artifact as v2 (§6.1).
- Per-mode defaults need both models available at once.
- The shared code is most of the engine, so keeping v1 is cheap.

**D2. Harden identity before anything else.** An absent `model_id` must permanently mean
`"unlimited-ocr"` (v1), never `default_arch()`. Every `== default_arch()` check that really means
"is v1" becomes an explicit capability or identity method on `ModelArch`.
*Why:* it removes the single most dangerous failure mode (silent misload) independently of v2, and
it is a prerequisite for ever changing the default.

**D3. Keep the decoder dimension constants compile-time. Add a variant axis for attention, RoPE
and decode contract, not a runtime shape spec.**
*Why:* v1 and v2 share every dimension (§2.2). The doctrine prefers fixed shapes in hot loops, and
the existing type-level shapes (`[_; NUM_EXPERTS_PER_TOK]`, `[f32; HEAD_DIM]`) stay valid. The
variance is one enum (`UnlimitedVariant::{V1Rswa, V2SrSwa}`) chosen from the arch at load time and
dispatched once per layer call. There are no trait objects in the inner loops.

**D4. Use the HF `infer()` path as the parity oracle for unwindowed decoding.**
*Why:*
- It is deterministic and runnable with pinned packages.
- It is the "transformers" path Baidu documents.
- It exposes every intermediate (router scores, selections, biases) we need for rung-by-rung parity.

SGLang differs in a few places (§5.11). Where they differ, we match HF in unwindowed mode and
record the difference in `docs/DISCREPANCIES.md`.

**D5. Use the SGLang reference semantics for windowed (unlimited-horizon) decoding.**
*Why:* HF `infer()` never splits documents. SGLang is the only implementation, it is what Baidu's
long-document benchmark numbers came from, and the chunked scheme is what makes books feasible. We
implement the semantics natively (§5.10). We do **not** replicate a post-processing quirk in its
non-streaming stitching path that moves a blank first page to the end of its window
(§5.10.8, DISC). The streaming path in the same wheel does not have it, and replicating it would
corrupt page alignment for users.

**D6. Port in order: base mode first, then multi-page base, then windowed, then gundam.**
*Why:*
- In base mode a page's 273 image tokens span at most 10 blocks, and the read budget is K=12. For
  every query with an own-page candidate set, Token Sieve therefore selects **all** own-page remote
  blocks. Only the continuous entmax bias varies, so routing cannot flip between implementations.
  That makes base mode the cheapest place to prove the attention math.
- Multi-page base is v2's headline accuracy win.
- Gundam pages carry 900–1,900 image tokens, so routing becomes truly selective. Gundam is also
  uncertified for v1 today.

**D7. Use the conservative int8 recipe policy unchanged.** Quantize exactly the 2,148
FFN/expert matrices. Keep everything else at source precision: attention q/k/v/o, `lm_head`,
`embed_tokens`, the MoE router gate, all norms, the vision tower, the projector, **and all new
router/indexer tensors**. The recipe id is `unlimited-ocr-2-ffn-int8-attn-bf16-lmhead-bf16-v1`.
*Why:* doctrine #2. It is the validated set for this backbone. The router projections are small
(0.33M MACs per layer per token) and steer a discrete selection, so quantizing them is all risk and
no gain.

**D8. Keep the unused `indexer.*` tensors in the artifact; drop only the I64 `position_ids` buffer.**
*Why:*
- At inference, scoring uses `dash_router.*`. The `indexer` module supplies only helper logic in
  the dash stage, so its projections are unused (§5.6.7).
- Keeping them makes the census exact and costs about 15.7 MB (0.4%).
- `position_ids` is a constant `arange(257)`. The converter verifies that and drops it, recording
  the rule in the census. `.focrq` has no I64 tensor dtype, and the CLIP path already derives
  positions.

**D9. Start the KV cache as f32 and full-length within a window. Earn memory reductions as
separate, proven levers.**
*Why:* exactness first. The levers are:
- retire-on-advance within a window, which is output-exact under the page guard (§5.12, proof
  obligation);
- windowing;
- bf16 KV, which is lossy and kill-switch gated.

**D10. n-gram defaults for v2 are 35 with window 128 (single page) or 1024 (multi-page).** That is
Baidu's evaluation setting and v1's existing contract. The oracle passes these explicitly, because
HF `infer()` defaults to off.
*Why:* it matches the published numbers and gives continuity for users. A flag exposes `0` (off).

**D11. Per-page output cap: 20,000 tokens per page by default** (HF and README), configurable with
`--max-page-tokens`.

**D12. Single-page crop mode for v2 is decided by evidence (milestone M4).** Until then, v2 single
pages run in **base** mode, which is faster on CPU and has the same vision cost as v1's default.
Gundam is available via `--crop-mode gundam` (uncertified until G1).
*Why:* Baidu's default is gundam, but gundam's CPU vision cost is 3–7× base, and v2's single-page
accuracy edge is negligible either way. The choice deserves measurement, not assumption.

**D13. Windowed decoding defaults to W=10 pages per window (Baidu-validated). Documents with at
most W pages run unwindowed.** `--page-window N|auto|off`; `auto` derives W from a memory budget.
*Why:* unwindowed is the HF parity mode, and for short documents windowing only adds risk.

**D14. Long runs are resumable and resume is byte-identical.** Completed windows are checkpointed
to the fsqlite run store, together with the retained tail state that the next window splices
(§5.10.4).
*Why:* a 1,000-page run on CPU takes hours. Users must be able to stop and resume without changing
the result.

**D15. The robot schema changes are additive only:** `page` and `window` events, plus fields on
`run_complete`.
*Why:* stable versioned NDJSON contract (AGENTS.md "Agent Ergonomics").

**D16. Generalize sharded-safetensors loading for all zoo models**, not as a v2 special case.
*Why:* v2 is the first default-family checkpoint with two shards, and future zoo models will be
sharded too.

---

## 5. Reference-semantics specification (v2)

The citations point into the pinned v2 sources: `MU` = `modeling_unlimitedocr.py`,
`MD` = `modeling_deepseekv2.py` (the DASH overlay), `SB` = `_sparse_base_impl.py`,
`DA` = `dash_attention.py`, `W/…` = the SGLang wheel. Line numbers refer to commit `5632329a`.

### 5.1 Prompt and tokenization

**Layout.**
- Prompt string: `"<PAGE><image>" * n_pages + instruction`, with the default instruction
  `"Multi page parsing."`. This holds even when `n_pages == 1` (`MU:1073-1080`).
- The token sequence is `[BOS=0] ( [<PAGE>=128827] [<image>=128815] × n_img(p) )_{p=1..N}
  [instruction tokens]` (`MU:531-580`).
- There is no chat template and there are no role tokens. The SGLang template `unlimited-ocr` has
  empty roles and separators.
- The trailing EOS appended by the vLLM/SGLang tokenizer is stripped in inference mode
  (`W/configs/unlimited_ocr.py:503-507`).

**Tokenization.**
- Text segments are encoded with `add_special_tokens=False`.
- `<PAGE>` must tokenize to the single id 128827. It is an *added, non-special* token, so the
  added-token split must recognize it before BPE.
- Decoding with `skip_special_tokens=False` renders it as the literal text `<PAGE>`.

**Image-token counts.** With nq = ceil((640/16)/4) = 10 and nqb = ceil((1024/16)/4) = 16:
- **Base (no crop):** `nqb*(nqb+1) + 1 = 273` (`MU:562`).
- **Gundam with tiles (w_t × h_t, where w_t>1 or h_t>1):** `273 + (nq*w_t + 1) * (nq*h_t)`
  (`MU:563-566`).

**Generation output.**
- Format: `<PAGE>body_1<PAGE>body_2…<PAGE>body_N` followed by EOS (id 1), which is dropped.
- Each `<PAGE>` *opens* a page.
- Body text keeps v1's grounding grammar (`<|ref|>…<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>`,
  0–999 normalized per page). The post-processor is unchanged apart from page splitting on the
  `<PAGE>` text.

### 5.2 Preprocessing (`MU:531-580`)

**Normalization.** mean = std = 0.5, i.e. `x/127.5 − 1`. Pad color (127,127,127). Each global view
is `ImageOps.pad(image, (1024,1024))`, an aspect-preserving letterbox using PIL BICUBIC resampling.

**Crop policy.** `crop_mode=None` means gundam when exactly one image and base otherwise. `infer_multi`
forces base (`MU:1117`). franken_ocr's v2 default for single pages is base (D12).

**Base mode.** One global 1024 view per page, **with no 640 pre-resize**. This is a deliberate
change from v1's multi-page path ("vLLM's training-parity fix", `MU:536-538`).

**Gundam mode.**
- If both image sides are ≤ 640, there are no tiles (ratio (1,1)).
- Otherwise `dynamic_preprocess(min_num=2, max_num=16, image_size=640)` applies. **v1 used
  max_num=32.**
- The tiles are 640² crops of the resized image.

**Vision-feature order in the embedding scatter** (`MU:731-790`):
- Tiled pages: `[local features with image_newline per row] ++ [global features with image_newline
  per row] ++ [view_seperator]`.
- Untiled pages: `[global with newline] ++ [view_seperator]`.
- This is identical to v1. Multi-page concatenates per page, in order.

### 5.3 Vision tower

Unchanged from v1: same code and byte-identical weights. franken_ocr's existing SAM, CLIP, bridge
and connector code and its vision parity evidence apply as-is. U2-B8 proves this mechanically.

### 5.4 Page bookkeeping

These definitions drive both RoPE (§5.5) and the Page Gate (§5.6.4). Let `ids` be the token-id
sequence, `img = ids==128815` and `mk = ids==128827`. Let `cum[i]` be the inclusive count of `mk`
up to position i, and `last_img` the index of the last image token. `N = cum[last_img]` is the
number of prompt pages, and **marker mode** means `N > 0`. v2 prompts are always in marker mode.
`in_prefill[i] = (i ≤ last_img)`.

**Image-token page id** (0-based): `img_page[i] = cum[i] − 1` for image tokens and −1 otherwise
(`SB:2404-2440`).

**Query page id** (`SB:2440-2458`, inference sets `_PAGE_SCOPE_PREFILL=True`, `MD:333`):

| Token | Query page |
|---|---|
| Prefill tokens up to `last_img` (markers and image tokens) | `cum − 1` |
| BOS (cum=0) | −1 |
| Instruction tokens after `last_img` | −1 |
| Generated tokens | `cum − 1 − N` (−1 if negative) |

So the k-th generated `<PAGE>` and its body map to page k−1, which is the prompt's page k−1.

**Per-block image-page range** (block size 32, `MU:481-528`): for each KV block,
- `bmax` is the max image page id among its tokens (−1 if none);
- `bmin` is the min image page id (−1 if none).

These are computed once at prefill, from the prompt. Generated tokens are never image tokens.

**Decode query page.** `marker_cum` is incremented when the *fed* token is `<PAGE>`. Then
`qp = marker_cum − 1 − N`, with −1 if negative.

### 5.5 page2d RoPE (main attention **and** router)

**Coordinates.** `SB:2511-2557` for prefill, `SB:1451-1489` for the decode state machine.

**Bucket per token.**
- Prefill tokens up to `last_img`: bucket = `cum` (so 1..N).
- Generated tokens and the instruction: bucket = `cum − N`, clamped to ≥ 0.
- BOS and instruction therefore land in bucket 0, the reserved "no page" bucket.

**Token position.** `tok_pos` is the 0-based running count of earlier tokens in the same bucket. A
response page's tokens continue after that page's prompt tokens, so the response `<PAGE>` of page b
gets `tok_pos = 1 + n_img(b)` (that is, `pf_cnt[b]`).

**Page position.** `page_pos = bucket`, folded to `bucket mod 2` at the angle computation
(`sparse_rope_page_mod=2`).

**Decode state machine.** Per layer, the state is `(page, tok, pf_cnt[], n_pg)`.
- It is seeded from the last prefill coordinate: page 0 (instruction bucket), with `tok` = the last
  bucket-0 count.
- On each fed token:
  - if it is `<PAGE>`: `page = min(page+1, n_pg−1)` and `tok = pf_cnt[page]`;
  - otherwise: `tok += 1`.
- `n_pg = N+1`. Extra markers clamp to the last page.

**Frequencies** (`SB:1984-2068`). dim = 128, half = 64, page_pairs = 1, so n_tok = 63.
- `inv_tok[i] = 10000^(−i/63)` for i in 0..63. This is "resample" mode; note the /63, not 2i/128.
- `inv_page[0] = 256^0 = 1`.
- `ang = [tok_pos·inv_tok[0..63], (page_pos mod 2)·1]` (64 angles); `emb = cat(ang, ang)`.
- Apply with the NeoX `rotate_half` convention: `x·cos + rotate_half(x)·sin`, where
  `rotate_half([x1,x2]) = [−x2, x1]`.
- cos and sin are computed in fp32, then cast to the activation dtype (bf16 in the reference).
- The router uses the **same** 2-axis RoPE and coordinates on its 128-d q/k (`index_dim ==
  head_dim`, so the 2-D branch is used, `MD:126-143`).

**Scaling.** The main-attention softmax scale is 1/√128. YaRN `attention_factor=1.0` does not
apply on this path.

**Fallback.** If a row has no `<PAGE>` or no image token, both paths fall back to 1-D RoPE (theta
5e5, YaRN ×4). v2 prompts always carry markers, so franken_ocr **rejects** v2 inputs without image
or marker structure rather than implementing the fallback (OQ-V2-6).

### 5.6 Token Sieve router (per layer, per query)

Constants come from the checkpoint snapshot of `sparse_attention_config`, which inference reads
(`MD:56-64,296-335`):

| Constant | Value |
|---|---|
| block_size | 32 |
| local_blocks | 4, so local_window = 128 tokens |
| K (read budget) | `dash_topk_blocks` = 12 |
| M (entmax window) | 16 (`dash_topm`) |
| idx_scale | `dash_scale_per_layer[l]/√128` = 128/√128 = 11.3137085 for all 12 layers |
| floor | `dash_floor_per_layer[l]` = 1e-4 |
| σ | 1.0 |
| stage0 | `head_cls` |

**Pitfall.** The live config dict is overwritten at construction by `_persist_router_cfg`. Inference
reads a **snapshot** taken before that. The env var `DASH_STAGE0` would override stage0, and the
oracle must leave it unset.

**5.6.1 Router projections** (`MD:365-377`). With h = the layer's input hidden state (the
post-attention-norm input to self_attn):
```
iq = q_norm(q_proj(h))          # [128]
ik = k_norm(k_proj(h))          # [128]
```
`q_norm`/`k_norm` are **Gemma-style RMSNorm**:
```
x32 = nan_to_num(x.float())
y   = x32 * rsqrt(mean(x32²) + 1e-6)
y   = clamp(y, ±1e4) * (1 + w)
y   = clamp(y, ±1e4)
```
- Apply 2-axis RoPE (§5.5) to iq and ik.
- Cast to f32, replacing NaN/Inf with 0.
- Index keys are cached **post-norm, post-RoPE**.

**5.6.2 Block summaries (stage0 = head_cls)** (`MD:391-408`). For block c, with valid tokens j in c:
```
a_j   = clamp((ik_j · head_cls) / √128, ±1e4)
w_j   = softmax over valid j of a_j      # max-subtracted; denominator clamped ≥ 1e-6
k̄_c  = Σ_j w_j · ik_j                    # nan_to_num → 0
```

**5.6.3 Remote block scores** (`MD:409-417`):
```
s_c = clamp((iq · k̄_c) / √128, ±1e4)   if c ≤ floor(q_pos/32) − 4 and c has a valid token
s_c = SCORE_NEG (−1e9)                   otherwise   ("not remote")
```
`q_pos` is the **1-D absolute position** of the query within the KV sequence of the current window.

**5.6.4 Page Gate (candidate set).**
- **Decode** (`MU:481-528`): `allow[c] = (bmin[c] ≥ 0) ∧ (bmin[c] ≤ qp ≤ bmax[c]) ∧ (qp ≥ 0)`. If
  `qp < 0` or no block is allowed, every block is allowed (open row).
- **Prefill** (`MD:108-123`, `SB:2629-2649,2706-2737`): `cand[q,c]` is true when block c contains
  any image token whose `img_page` equals the query page of q (≥ 0). If q has no such block, every
  block is a candidate for q (per-query fallback).
- **Then** (`MD:339-348`):
  ```
  remote = s > SCORE_NEG/2
  cand   = remote ∧ allow_or_cand
  if not any(cand): cand = remote
  ```
  If nothing is remote (the query is in the first 4 blocks), the query attends to its local window
  only.

**5.6.5 Entmax-1.5 over the top-M window (HF semantics)** (`MD:349-362`, `DA:772-813`):
```
x_c  = s_c · idx_scale  if cand[c] else −1e4 ; nan/±inf → −1e4
(xv, xi) = topk(x, M = min(16, nblk))          # descending
prior = entmax15(xv):
    z = (xv − max(xv)) / 2
    sort z descending; for k = 1..M: mean_k, meansq_k, ss_k = k(meansq_k − mean_k²),
    δ_k = (1 − ss_k)/k, τ_k = mean_k − sqrt(max(δ_k, 0))
    support = #{k : τ_k ≤ z_sorted[k]} clamped ≥ 1 ; τ* = τ_support
    p = max(z − τ*, 0)²
is_cand = cand[xi]
K' = min(12, M) ; take the first K' slots: idx = xi[:K'], p = prior[:K'], valid = is_cand[:K']
logp = ln(max(p, floor)) ; μ = mean of logp over valid slots (count clamped ≥ 1)
bias = (logp − μ)/σ  if valid else 0
```
Selected blocks are `idx` where `valid`, each carrying a scalar `bias`. Zero-prior blocks inside
the first K' slots are still read, with bias `ln(1e-4) − μ`.

**5.6.6 SGLang difference.** SGLang runs entmax over **all** candidate blocks, using an exact τ
bisection, rather than the top-16 window. The two agree whenever the candidate count is ≤ 16. That
covers every own-page query in base mode (≤ 10 blocks). It does **not** cover fallback queries
(BOS, instruction, the first image blocks of a page, generated text before the first `<PAGE>`) or
gundam pages. We implement HF semantics (D4); the difference is recorded as DISC-U2-1.

**5.6.7 Unused indexer.** `self_attn.indexer.{q,k}_proj` are not used at inference (the dash path
scores with `dash_router.*`). The indexer object only hosts page and RoPE helper methods.

### 5.7 Block-sparse biased attention

For a query at position q with selected blocks B (each with bias β_b) (`MD:454-481` for decode,
`DA:1230-1261` for prefill):
```
local(j)  = (q − 128 < j ≤ q)
remote(j) = (j ≤ q) ∧ (block(j) ∈ B) ∧ ¬local(j)          # local wins on overlap
logit_j   = (q·k_j)/√128 + (remote(j) ? β_block(j) : 0)
out       = softmax over {j : local(j) ∨ remote(j)} of logit_j, applied to v_j
```
- Per head; 10 heads share the selection, since the router is one head per layer.
- In the reference, prefill runs SDPA in **bf16**, with the additive mask (including the bias) cast
  to bf16. Decode runs in fp32.
- franken_ocr computes in f32 throughout. The parity tolerance is measured, not assumed (OQ-V2-3).
- The output goes through `o_proj` as before. There is no attention bias anywhere
  (`attention_bias=false`).

### 5.8 Prefill vs decode

- **Prefill** (`q_len > 1`): every prompt token is a query with its own routing (§5.6) and sparse
  attention (§5.7). The router index keys of all prompt tokens are cached.
- **Decode** (`q_len == 1`):
  - append the token's K, V and index key;
  - recompute its block's summary;
  - route against all blocks (candidates per §5.6.4);
  - attend per §5.7.
- **Stateless equivalence.** The reference decode path and a fresh prefill over the full sequence
  should select identically, because both use causal per-query routing and the same page ids. This
  is a metamorphic test (§7.4), not an assumption.
- **Only the last position** goes through `lm_head` after prefill (`MU:1143-1145`).

### 5.9 Decode loop (`MU:1124-1200`)

1. **Greedy** when temperature is 0. Temperature sampling is out of scope for parity.
2. **Logit processors, per step:**
   - **n-gram ban** (when enabled, D10), applied to *generated ids only*. With
     `prefix = gen[-(n−1):]`, ban `gen[i+n−1]` for every i in `[max(0, len−window), len−n+1)` where
     `gen[i:i+n−1] == prefix`.
   - **Page-count guard** (HF semantics), with `expected = #<PAGE> in prompt`, `mc = #<PAGE> in
     gen`, and `page_len` = tokens since the last generated `<PAGE>`:
     ```
     if mc > expected: only EOS allowed
     elif 0 < max_page_tokens ≤ page_len: only (<PAGE> if mc < expected else EOS) allowed
     elif mc < expected: EOS banned
     ```
3. **Stop** at EOS, at `max_new_tokens` (default `max_length − prompt_len`, with `max_length`
   131072), or when `pos+1 ≥ max_length`.
4. **Order hazard (OQ-V2-7).** HF applies the n-gram ban **before** the guard. If the ban removed
   the token the guard then forces, every logit is −inf and HF's argmax emits id 0. SGLang applies
   the guard first and a forced token short-circuits. **franken_ocr makes forced tokens
   authoritative** (SGLang order). The oracle comparison asserts this case never occurs on any
   fixture, and the difference is recorded as DISC-U2-2.
5. **HF vs SGLang guard difference.** On the last page of an unsplit request, SGLang also bans
   `<PAGE>` (when not `end_on_marker`); HF does not. With HF semantics, an extra `<PAGE>` is
   possible and is followed by a forced EOS. Our page splitter drops a trailing empty page body and
   counts it (`extra_markers`). Unwindowed mode follows HF; windowed mode follows SGLang (§5.10.7).

### 5.10 Windowed ("unlimited-horizon") decoding (SGLang reference semantics)

The wheel citations come from the spec extracted on 2026-10-28; the full notes travel with the
truth pack (U2-A1).

**5.10.1 Planning.**
- The document has N pages and a window size of W own pages; the default is 10 (D13).
- Windows are contiguous and own their pages: `[0,W), [W,2W), …`.
- Every window except the last also encodes **one lookahead page**: the next window's first page.
  It is encoded but never transcribed.
- If `N ≤ W`, decoding is unwindowed (the plain §5.9 loop).

**5.10.2 Window 0, and any recomputed window.**
- Prompt: `[BOS] (<PAGE> img)×(own pages) [<PAGE> img(lookahead)] "Multi page parsing."`, with
  `page_base = lo`.
- Guard: `expected = #markers − context_pages(0) − lookahead`, which equals the number of own
  pages.
- A window with a lookahead ends when the model emits the `<PAGE>` that would open the lookahead
  page: `marks > expected` forces EOS. A natural EOS on the last own page is also accepted.

**5.10.3 Retention at the end of window k** (only if it finished by EOS and is not the last
window). Keep exactly these KV rows, together with the router index keys and the per-slot page
state, and free the rest:
- (a) the `<PAGE>` marker of the last own page;
- (b) the **image tail** `[E − 128 − (E mod 32), E)`, where E is one past the last own page's last
  image token (128–159 tokens, block-aligned);
- (c) the whole lookahead page (marker and image tokens);
- (d) the **text tail**: up to 128 trailing generated tokens of the last own page's body, never
  including a marker.

**5.10.4 Window k+1 with reuse.**
- **New KV sequence, in order:**
  1. copied (a), at position 0;
  2. copied (b);
  3. copied (c), which is window k+1's own first page;
  4. **prefill** of the remaining own pages `lo+1..hi−1` and of window k+1's own lookahead page;
  5. copied (d), the text tail.
- **No BOS and no instruction.**
- `page_base = lo − 1`, because the tail page is local page 0.
- **Copied K and index keys are not re-rotated.** page2d coordinates are page-local, and
  `page_pos ≡ global_page+1 (mod 2)` is preserved by the `page_base` parity rule.
- **Router block summaries are recomputed** for the new block grouping.
- The per-slot page state is replayed in the new window's frame. The text tail is assigned to
  page 0's bucket.
- **The sampled prefill token is discarded.** The first decode input is a **forced `<PAGE>`**. The
  wheel comments that copied KV otherwise re-predicts EOS (18 of 94 MPDocBench follow-ups).
- The forced tokens (text tail + `<PAGE>`) count as generated for the n-gram domain, but are hidden
  from output.
- Guard: `context_pages = 1`, `end_on_marker = true`, and `expected = #markers(prompt) − 1 −
  lookahead` (own pages).

**5.10.5 Fallback.** If retention is not possible (window ended by length or cancel, a page has
< ~129 image tokens, page discontinuity, a missing text tail, or memory pressure), window k+1 is
recomputed exactly like window 0. Its prompt includes BOS and the instruction, and it has no
cross-window context.

**5.10.6 Positions.** 1-D positions (which only index blocks and causality) are window-local: KV
row 0 is position 0. There are no absolute document positions, so the 131072 limit applies per
window.

**5.10.7 Guard and caps.** The per-page cap counts tokens since the last `<PAGE>`, including the
forced one. The global token budget is per document; window i gets
`min(remaining, capacity_i)`, where
`capacity_i = min(ctx − 64 − per_page·(own+lookahead), own·max_page_tokens + 256)`. Spliced and
forced tokens do not consume the budget.

**5.10.8 Stitching.** Split each window's visible text on `<PAGE>`:
- A reused window's visible text starts with the body of its first own page.
- Keep the bodies of own pages only, and concatenate in order.
- A trailing empty body (the window ended on the lookahead marker) is dropped.
- **Deviation (D5):** never reorder a blank first page.

**5.10.9 Within-window retirement.** SGLang retires nothing within a window. Our optional
retirement is §5.12.

### 5.11 Known HF-vs-SGLang divergences and our choices

| ID | Divergence | HF | SGLang | franken_ocr |
|---|---|---|---|---|
| DISC-U2-1 | entmax domain | top-16 window | all candidates | HF (D4); measure on gundam and fallback queries |
| DISC-U2-2 | n-gram vs guard order | n-gram first (can yield id 0) | guard first, forced token authoritative | forced token authoritative; assert never triggered on fixtures |
| DISC-U2-3 | `<PAGE>` ban on the last page of an unsplit request | no ban | ban when not end_on_marker | HF unwindowed; SGLang inside windowed mode |
| DISC-U2-4 | windowed stitching blank-page reorder | n/a | non-streaming path reorders | never reorder |
| DISC-U2-5 | numeric precision | bf16 prefill SDPA (bf16 mask/bias), fp32 decode | bf16 Triton | f32 throughout (measured tolerance) |

### 5.12 Retire-on-advance (franken_ocr optimization, output-exact under the guard)

**Claim.** In decode (unwindowed, or within a window), once the fed token is the `<PAGE>` that
opens page k, these become unreachable:
- every KV row belonging to image pages < k;
- every generated-text row older than the local window.

They are unreachable **unless** the step is an open row (qp < 0 or no allowed block).

**Why open rows are harmless with the guard on.**
- qp < 0 happens only before the first generated `<PAGE>`, when nothing has been retired yet.
- "No allowed block" with qp ≥ 0 happens only when qp ≥ N, i.e. after an extra `<PAGE>`. The guard
  then forces EOS on the very next step (`mc > expected`). That step's logits are fully overridden,
  and no later step reads its KV.

**Proof obligation and runtime check.** The proof obligation is tested in U2-D3. A runtime detector
asserts that no open row ever needs retired rows in a step whose output is not guard-forced. If it
fires, the run records a ledger event and the deterministic fallback is to re-run that window
without retirement. With the guard disabled, retirement is disabled too.

---

## 6. franken_ocr architecture for v2

### 6.1 Identity and loading (U2-A2, U2-A3, U2-A5)

**Current hazards.**
- `weights.rs:73-76` and `mod.rs:1221-1224` map an absent `model_id` to `default_arch()`.
- `focrq.rs:593-603` omits `model_id` when it equals the default.
- Raw safetensors are tagged as the default (`weights.rs:1238`).
- `mod.rs:1231,1573,1699,2626,2663,2701,2851`, `cli.rs:2350,2420` and `convert.rs:124` all mean
  "is v1" when they test `== default_arch()`.

**Change.**
- Introduce `model_arch::UNLIMITED_OCR_V1_ID = "unlimited-ocr"` and `fn legacy_absent_model_id() ->
  &'static str` returning it permanently.
- The writer always emits `model_id` for non-v1 archs and continues to omit it for v1, which keeps
  v1 artifacts byte-identical.
- Raw safetensors identify their arch from the census match: v1 census → v1, v2 census → v2,
  neither → refuse. This replaces the current "assume default" behavior.
- Replace identity checks with `ModelArch` capability methods:
  - `supports_cross_page()` (true for v1 and v2);
  - `census()` (an `Option<&'static Census>`);
  - `required_quant_recipe(quant)`;
  - `decode_contract()`;
  - `unlimited_variant()` (`Some(V1Rswa | V2SrSwa)`).
- **Tests:**
  - a v1 artifact without `model_id` loads as v1 under a test-only registry where the default is
    v2;
  - a v2 artifact without `model_id` is impossible to write;
  - raw safetensors are arch-detected by census.

**Sharded sources (U2-A3).**
- `FOCR_MODEL_PATH` or `--model` pointing at a directory with `model.safetensors.index.json`
  resolves every shard, verifies each shard's SHA-256 against the arch's pins, and presents one
  logical tensor directory.
- `focr convert <dir>` accepts the same input.
- Byte-owned by default; `FOCR_MMAP=1` maps each shard. Generic for the whole zoo.

**v2 census (U2-A5).**
- An embedded manifest `unlimited_ocr_2_manifest.json` (new data file) lists all 2,819 tensor
  names, shapes and dtypes, with pins:
  - HF commit `5632329a…`;
  - both shard SHA-256s and sizes;
  - the tokenizer.json SHA;
  - the verified-artifact SHA, filled in at C3.
- `unlimited_ocr_census.rs` is generalized to hold two `Census` values, v1 and v2. There is no
  second census module (no file proliferation).

### 6.2 Decoder structure (U2-B1…B6)

| Piece | Location | Notes |
|---|---|---|
| Dimension constants | `decoder::config`, `moe::config`, `rswa.rs` constants | Unchanged (D3) |
| Variant enum | `UnlimitedVariant` | Chosen once from the arch |
| Layer forward | Generic over an attention-kind enum: `LayerAttention::Rswa(&mut RingCache)` / `LayerAttention::SrSwa(&mut SrSwaLayerCache, &SrSwaLayerWeights)` | MoE, FFN, norms and `lm_head` code is shared verbatim |
| New module `src/native_engine/srswa.rs` | Genuinely new functionality (like `rswa.rs`) | Contents below |
| `RopeTable` | Gains `RopeTable::from_angles` | page2d tables reuse the same apply kernel (`apply_rope`, `decoder.rs:238`) |
| v2 generate loop | `mod.rs`, next to the v1 path | Prompt builder (§5.1), guard (§5.9), n-gram (reuse `sampler.rs` sliding-window n-gram over generated ids), `<PAGE>`-aware `PageStream`, EOS handling, length limits (131072) |

Contents of `srswa.rs`:
- `PageCoords` and the decode state machine (§5.4–5.5);
- `Page2dRope` tables;
- `SrSwaLayerWeights` (router `q/k_proj` as pre-transposed f32 or bf16→f32, norm weights,
  `head_cls`);
- `SrSwaLayerCache`, described below;
- router functions (§5.6): `project`, `summaries`, `scores`, `page_gate`, `entmax15_topm`,
  `select_bias`;
- the attention kernel (§5.7), with a dense scalar oracle and a gathered fast path;
- prefill and decode entry points;
- `retire_through(page)` (§5.12);
- `splice_from(prev_tail)` (§5.10.4);
- `serialize_tail` and `deserialize_tail` (D14).

`SrSwaLayerCache` holds, per layer:
- K and V, contiguous f32 `[cap_tokens × 1280]` with a live length;
- index keys, f32 `[cap_tokens × 128]`;
- `kbar` `[cap_blocks × 128]` and validity;
- `bmin` and `bmax` per block;
- page state: `marker_cum`, N, `pf_cnt`, page and tok;
- a retired-row bitmap or range set.

**v1 non-regression.** All existing v1 fixtures, goldens and selftests must remain byte-identical
across the refactor (U2-T4 gate). No v1 code path may observe the v2 code.

### 6.3 Memory model

Per token, with 12 layers:
- K+V in f32 = 12 × 2 × 1280 × 4 B = **120 KiB**;
- index keys = 12 × 128 × 4 B = **6 KiB**;
- block summaries are negligible.

Worked examples, assuming roughly 1,000 output tokens per page. These are analytic estimates;
U2-F1 measures the real numbers.

| Scenario | Tokens resident | f32 KV |
|---|---|---|
| Single page, base | ~275 prompt + ~1,000 out | ~0.16 GiB |
| 20 pages unwindowed | 20×274 + 20,000 ≈ 25.5k | ~3.0 GiB; ~0.65 GiB with retirement (prefill image KV dominates) |
| 100 pages unwindowed | ~127k | ~15 GiB; also hits 131k max_length. **Windowing is required on CPU.** |
| W=10 window | (11×274 ≈ 3k prompt) + 10k out | ~1.5 GiB; ~0.4 GiB with retirement |

The plan adds a memory-budget estimator (U2-D3), used by `--page-window auto` and reported in
robot `window` events.

### 6.4 Concurrency (doctrine #5)

- One live forward at a time.
- Windows run strictly sequentially.
- Per-page vision encoding is sequential within a window.
- All parallelism is intra-op via the kernel rayon pool: heads × query chunks in prefill attention,
  GEMMs and vision.
- No rayon under a held lock, and no nested asupersync runtime.
- Resume and checkpoint I/O happens between windows, never inside a forward.
- The `many_pages_without_deadlock` watchdog is extended to windowed v2 (U2-T2).

### 6.5 Quantization and artifact (U2-C3)

**Recipe.** `unlimited-ocr-2-ffn-int8-attn-bf16-lmhead-bf16-v1`. Classification is by name prefix
in `recipe.rs`:
- int8: exactly the 2,148 FFN/expert matrices (`model.layers.0.mlp.{gate,up,down}_proj`,
  `model.layers.{1..11}.mlp.experts.{0..63}.{gate,up,down}_proj`,
  `model.layers.{1..11}.mlp.shared_experts.{gate,up,down}_proj`);
- high precision (BF16 source): everything else, including `self_attn.dash_router.*`,
  `self_attn.indexer.*` and `mlp.gate.weight`;
- the I64 `position_ids` is verified to equal `arange(257)` and is dropped (D8).

**Counts and gates.**
- int8 = 2,148; high-precision tensors = 2,819 − 2,148 − 1 = 670.
- The loader gate rejects any v2 `.focrq` whose dtypes violate the recipe or whose `source_sha256`
  values don't match the pinned shard pair.
- The i32-accumulation proof at K=6848 is unchanged (same dense layer-0 `down_proj`).

**Artifact.**
- `unlimited-ocr-2.v<ver>.int8.focrq`, about 4.2 GB.
- Installed under `~/.cache/franken_ocr/models/unlimited-ocr-2/` with **its own** `tokenizer.json`
  (the v2 hash). The per-model subdirectory prevents a sidecar collision with v1's flat
  `tokenizer.json`.
- Split into parts under 2 GiB for GitHub release assets; the HF mirror is per bd-3p8y and
  bd-aio5.

---

## 7. Verification plan

### 7.1 Oracle (U2-A6, U2-A7)

**Environment.**
- A separate pinned venv: python 3.10, torch 2.13.0, transformers 4.52.1, tokenizers 0.21.4,
  addict, einops, easydict.
- Model at commit `5632329a`, bf16, on CUDA.
- `DASH_STAGE0` unset; `TIME_DIAG` and `SPARSE_DIAG_LOG` off.
- **Never shares a venv with the v1 oracle.**

**Script.** `scripts/gen_reference_fixtures_uocr2.py`. Per-model oracle scripts are existing
precedent (`gen_reference_fixtures_got.py`).
- It drives `model.infer()` / `infer_multi()` with explicit `no_repeat_ngram_size=35`,
  `ngram_window=128|1024`, `max_page_tokens=20000`, `page_count_guard=True`, and an explicit
  `crop_mode`.

**Dumps, through forward hooks.**
- Prompt token ids.
- Preprocessed tensors.
- Per-layer input hidden states.
- Per layer, for selected query rows:
  - router iq and ik (post-RoPE);
  - `kbar`;
  - block scores;
  - candidate mask;
  - entmax prior;
  - topi, bias and valid.
- Per-layer attention output.
- Final hidden state, last-position logits and generated ids.
- **Generation runs on GPU.** Per-step dumps are captured for the first 64 decode steps and for
  steps around every `<PAGE>`.

**Fixture sets.**

| Set | Contents |
|---|---|
| S1 | One page, base (`crop_mode=False`): 3 pages from the existing 20-page corpus, including the hard page used for page0590-style EOS checks |
| S2 | 3-page document, base multi |
| S3 | The existing multi-page fixtures (p9_p14, p10, p20) rerun under v2 |
| S4 | One page, gundam (`crop_mode=True`) |
| S5 | One 40–60 page document with ground truth (a MarathonDocBench subset, Apache-2.0), unwindowed on GPU, for D2 equivalence and accuracy work |

**Nondeterminism floor.** Run each fixture twice. Record bitwise token equality and the max
hidden-state deltas, which set the tolerances (AGENTS testing policy).

**CPU feasibility (OQ-V2-4).** Patch `.cuda()` and run the dense fallback on CPU to see whether CPU
tokens match GPU. If they do, a CI-friendly CPU oracle and CPU perf baseline exist. If not, the
oracle stays GPU-only and the CPU baseline uses v1 franken_ocr (incumbent comparison).

### 7.2 Parity ladder rungs for v2 (U2-C1, U2-C2)

| Rung | What | Gate |
|---|---|---|
| L0 | Preprocessing tensors; prompt ids (incl. `<PAGE>`=128827); image-token counts | exact (pil-bicubic mode) |
| L1 | page2d coordinates and cos/sin; Gemma norm; `kbar`; scores; entmax; selection; bias | coordinates exact; floats within measured tol |
| L2 | Per-layer hidden states; **routing equality** (topi and valid sets exact on S1/S2; bias within tol); attention outputs | cosine ≥ the floor-derived threshold |
| L3 | Last-position logits | within measured tolerance |
| L4 | Generated tokens | exact where the reference is deterministic (S1, S2); S3 within CER budget |
| L5 | Corpus CER (§7.3) | budgets |

**Routing-margin diagnostics.** For every selection mismatch, log the score margin between the
12th and 13th candidates and the entmax support size. A mismatch with a margin below the measured
float noise is classified "tie-flip". This is a discrepancy class with a ceiling, not a free pass.

### 7.3 Corpus gates (U2-C4)

**Single page.**
- The existing 20-page corpus, v2 base mode.
- The v2 int8 artifact vs the v2 BF16 HF reference: aggregate normalized CER with the same budget
  as v1 (0.25).
- Hard pages must emit EOS; the guard makes EOS reachable only after `<PAGE>`.

**Multi-page.**
- S2 and S3 vs the v2 reference: per-document CER and page-count equality.

**Ledger.** All numbers go to `docs/DISCREPANCIES.md`, with the artifact SHA.

### 7.4 Property and metamorphic tests (U2-T3)

- entmax-1.5: sums to 1; support ≥ 1; monotone in input; shift-invariant; matches the brute-force τ
  solve.
- Page coordinates: bijective (bucket, tok) per sequence; the decode state machine equals the
  prefill builder on teacher-forced sequences.
- Guard state machine: never emits more than N markers; EOS is impossible before N; the cap forces
  correctly.
- **Stateless ≡ cached:** prefill-recompute of a prefix produces the same selections and logits as
  incremental decode (tolerance-gated).
- Window planner: partition, lookahead and budget invariants.
- Splice: copied rows equal their sources; replayed page state equals a fresh computation; `kbar`
  is recomputed for the new grouping.
- Stitching: bodies by own page; order preserved; a trailing empty body is dropped; a blank first
  page is never reordered.
- Tokenizer: added-token splitting is idempotent; `<PAGE>` round-trips; fuzz over random
  interleavings of added tokens and text.

### 7.5 Windowed-mode validation (U2-D2)

1. **Degenerate equivalence.** With W ≥ N, output is byte-identical to unwindowed.
2. **Invariants.**
   - exactly N page bodies;
   - order preserved;
   - no lookahead page is ever transcribed;
   - resume (byte-identical, D14);
   - memory bounded by the window plan.
3. **Accuracy.** On S5 and a MarathonDocBench subset:
   - windowed (W=10) vs unwindowed CER, both against ground truth;
   - windowed must not be worse than unwindowed beyond a stated tolerance.
   - Baidu describes its scheme as "approximate; validated by OCR accuracy."
4. **Optional GPU SGLang runs.** Run the pinned wheel on one or two long documents to compare
   per-page text with ours. This is informational, because of DISC-U2-1/5.

### 7.6 v1 non-regression gate (U2-T4)

Before any shared-code refactor, capture a v1 golden set (model-gated):
- token ids for the S1/S3-equivalent v1 inputs;
- `robot selftest` output;
- existing parity fixtures.

Every U2 bead touching the decoder, loader, tokenizer or sampler must keep these byte-identical.
The gate runs in `scripts/check.sh` when weights are present (skip-with-success otherwise, per the
model-gated convention), and the "native path ran" proof applies.

### 7.7 E2E scripts and logging (U2-T1)

**Script.** `scripts/uocr2_e2e.sh`. It is model-gated and emits structured NDJSON logs per
`docs/TEST_LOGGING.md` (one line per step, with timings, model id, artifact SHA and outcome).

**Scenarios.**
- (a) single page base;
- (b) 3-page multi;
- (c) 20-page multi;
- (d) windowed 35-page document with W=10, built by concatenating corpus pages;
- (e) resume after injected cancellation at window 2, which must be byte-identical;
- (f) robot schema validation of `page`, `window` and `run_complete`;
- (g) the v1 regression check;
- (h) `--multi-page --json --extract-figures` per-page outputs;
- (i) `focr pull unlimited-ocr-2` against a local manifest.

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
| Attention scores and AV (v2) | ≤ 12 × 10 × 128 × 2 × 512 ≈ 16M |
| **Total (v2)** | **≈ 0.59G** |

**Comparisons with v1.**
- **Single page:** v1 attends to ~408 tokens, so its attention is about 12.5M. **Decode cost is
  roughly equal**; no speedup is expected.
- **Multi-page decode:** v1 attends to `m = 111·N + 128`. At N=100 that adds about 344M per token
  (+58%), and at N=290 about 1G. v2 stays flat. v2 wins beyond roughly 50 pages, and only v2 can
  go past about 290 pages.
- **Prefill:**
  - v1 full causal attention ≈ 15,360·S² MACs;
  - v2 sparse ≈ S·15.7M;
  - linear prefill work is S × ~430M for both, and v2's S per page is 2.5× larger (274 vs 111
    tokens).
- **Vision per page:** SAM-B at 1024 is about 0.45 TMAC, against about 0.15 TMAC at 640, so v2
  multi-page vision costs about 2.8× v1 multi-page vision.

**Honest summary.**
- Single page: parity.
- Short multi-page documents: comparable; v2 spends more on vision and prefill and less on
  attention.
- Long documents: v2 wins, and it is the only option at book scale.

The default-flip gates (§10) treat performance as a **budget, not a selling point**, except at long
document lengths.

### 8.2 Exact levers (each with bit-identity proof; U2-F2, U2-F3)

1. **Candidate-only scoring.** Score only allowed blocks plus, for open rows, all remote blocks.
   This is identical to scoring everything and masking, because non-candidates are set to −1e4
   before top-M.
2. **Incremental `kbar`.** Complete blocks are immutable; recompute only the tail block, plus
   copied segments on splice.
3. **Gathered attention layout.** Contiguous per-block K/V slices with the head-major inner loop
   left to LLVM autovec (doctrine #3: no hand-rolled wide SIMD over scalar loops).
4. **Prefill attention parallelism.** Rayon over (head, query-chunk) tiles inside the forward.
5. **Vision once per page in windowed mode.** Lookahead KV is copied, never re-encoded.
6. **Retire-on-advance** (§5.12), for memory and cache locality.

### 8.3 Lossy levers (kill-switch gated, CER-measured; U2-F4)

- bf16 KV storage, which halves KV memory.
- The int8 attention, `lm_head` and int4 expert levers are **out of scope** for v2 until the
  equivalent v1 gates exist (doctrine #2).

### 8.4 Benchmarks

Every row runs v1 and v2 in the **same invocation**, at the same thread count, on the same inputs
(suite rule: a win needs the incumbent live), with `FOCR_TIMING=1` rows for:
- `srswa.router`, `srswa.attend`, `srswa.kbar`;
- `vision.page`, `prefill`, `decode`.

Results go to `docs/PERF_LEDGER.md`, including losing rows.

---

## 9. Product surface

### 9.1 CLI (U2-E1, U2-D4, U2-D5)

- **Model selection.** `--model unlimited-ocr-2`: `--model` accepts zoo **ids** in addition to
  paths and filenames, resolving through the per-model cache subdirectory. Ids are accretive across
  the whole zoo.
- **Single page:** `focr ocr page.png --model unlimited-ocr-2 [--crop-mode base|gundam]`.
- **Multi-page:** `focr ocr doc.pdf --multi-page --model unlimited-ocr-2 [--pages …]
  [--page-window N|auto|off] [--max-page-tokens N] [--resume RUN_ID] [--json] [-o out.md|out.json]
  [--extract-figures]`.
- **Batch:** `focr ocr-batch … --multi-page --model unlimited-ocr-2`.
- **Exit codes.** Existing codes only. Running out of budget is 5, with a partial-output note.
- **`focr models`.** Lists `unlimited-ocr-2` with status, tasks, recipe, the "best for: multi-page
  and long documents" guidance, and its pull state.
- **Help text.** Explains when to choose v1 vs v2.

### 9.2 Robot mode (additive)

**`page` event.**
```
{event:"page", page:<1-based source page>, index:<1-based in run>, window:<k>, chars,
 tokens, finish:"marker|eos|page_cap", elapsed_ms}
```
Emitted as each page body completes (streaming).

**`window` event.**
```
{event:"window", index, pages:[lo,hi], lookahead:0|1, reused:bool, prefill_tokens,
 generated_tokens, kv_bytes_peak, elapsed_ms}
```

**`run_complete` additions.** `model_id`, `pages_out`, `windows`, `extra_markers`, and
`finish_reason` (`stop|length|cancelled`).

**Contract upkeep.**
- `robot schema` and the frozen schema fixture are updated, coordinating with the open schema-golden
  bug bd-wp8.2.2.
- `tests/surface_matrix.rs` and `docs/FEATURE_PARITY.md` get rows for every new flag and event.
- `robot triage` recommends `--multi-page --model unlimited-ocr-2` for PDFs with more than about 10
  pages when v2 is installed.

### 9.3 Structured multi-page output (U2-E3)

v2's explicit `<PAGE>` segmentation lets `--multi-page` produce:
- a per-page `pages[]` array with layout boxes mapped to each source page;
- figure extraction per page (`page{N}_figure_{M}`).

Today `--multi-page` refuses `--extract-figures` because of per-page semantics, and this lifts that
restriction for v2. Boxes are 0–999 normalized per page in the model output and are mapped with
that page's original size.

### 9.4 Library (U2-E2)

```rust
let engine = OcrEngine::new()?;
let opts = DocumentOptions::builder()
    .model(ModelChoice::Id("unlimited-ocr-2"))
    .page_window(PageWindow::Auto)
    .max_page_tokens(20_000)
    .resume(None)
    .build();
let doc = engine.recognize_document(PageSource::pdf("book.pdf", PageSelection::All), &opts,
    |page: &PageResult| { /* stream */ })?;
```
- Synchronous and blocking, with the runtime owned internally.
- The existing `recognize_multi_page*` calls keep their signatures and work with v2 artifacts.
- Coordinate with bd-paom (`recognize_pdf`) and bd-3gyt (consolidated PDF walks) so there is a
  single document-walk path.

### 9.5 Distribution (U2-E4)

- A manifest-v2 `models["unlimited-ocr-2"]` entry with `license_notice`, `quants.int8 {recipe,
  focrq{filename,size,sha256,parts[]}}` and `tokenizer` (the v2 `tokenizer.json`).
- `dist.rs::required_quant_recipe` gains a v2 arm.
- `focr pull unlimited-ocr-2` installs into the model subdirectory.
- The install docs size text is updated.
- The HF mirror follows bd-3p8y / bd-aio5 (non-blocking; GitHub release parts are primary).

### 9.6 Health and doctor (U2-E7)

- `robot health` reports installed models, including v2, with artifact SHA and recipe.
- `doctor` detects a v2 artifact with a v1 tokenizer sidecar (hash mismatch) and offers the
  reversible fix of re-pulling the sidecar.
- `robot triage` is v2-aware.

### 9.7 Documentation (U2-E6)

- README: the model table; a "Long documents and books" section; v2 limitations (single-page
  parity, CPU vision cost at 1024 per page, windowed mode is approximate per Baidu, gundam
  uncertified until G1).
- AGENTS.md "What this model is": add the v2 paragraph.
- CHANGELOG.
- The master plan §13: a cross-reference to the OQ-V2 register.
- DISCREPANCIES (DISC-U2-*) and PERF_LEDGER rows.

---

## 10. Default-policy decision framework (U2-E5)

| Mode | Today | After M2 (artifact certified) | After M4, if gates pass |
|---|---|---|---|
| `focr ocr page.png` | v1 base | v1 (v2 opt-in) | v2 only if the single-page gate passes, else v1 |
| `focr ocr doc.pdf` (per page) | v1 | v1 | follows the single-page decision |
| `--multi-page` | v1 (≤32K context) | v1 (v2 opt-in) | **v2 when installed** if the multi-page gate passes; v1 otherwise. A one-line stderr/robot note names the model used. |
| `ocr-batch` | v1 | v1 | follows the single-page decision |
| `focr pull` (no argument) | v1 artifact | v1 | unchanged; `focr pull unlimited-ocr-2` is explicit (v1 users are never auto-upgraded into a 4 GB download) |

**Multi-page gate (all required).**
1. C4 green on the int8 artifact.
2. The ground-truth head-to-head (U2-C5) on at least 30 multi-page documents (5–60 pages) shows v2
   ≥ v1 on aggregate normalized text edit distance, with a lower confidence bound above v1 (the
   conformal-ratchet style of `docs/conformance/RATCHET.md`) and no category regression beyond the
   stated tolerance.
3. Windowed-mode validation (§7.5) is green.
4. CPU wall time per page is within 1.5× of v1 multi-page at ≤ 20 pages. The cost is published
   either way.
5. The surface ledger is complete.

**Single-page gate (all required).**
1. v2 (in its chosen crop mode, from G2) ≥ v1 on ground-truth CER over at least 100 pages, with a
   lower bound above v1.
2. CPU wall time ≤ 1.1× v1.
3. Hard-page EOS behavior is no worse.

**Evidence sets.**
- MarathonDocBench pages and documents (Apache-2.0; it has per-page ground truth).
- Optionally MPDocBench and OmniDocBench subsets, if their licenses allow local evaluation
  (OQ-V2-12).
- Metric: our minimal normalized edit distance, plus the official MPDocBench-Parse toolkit offline
  for reported numbers.

---

## 11. Roadmap and milestones

Each milestone ends with something a user can run, and has an exit gate.

**M0 — Foundation (no user-visible change).**
- **Work:** A1 truth pack and OQ register; A2 identity hardening; A3 sharded sources; A4 tokenizer;
  A5 arch and census; A6/A7 oracle; T4 v1 golden capture.
- **Exit:** v1 byte-identical; the v2 checkpoint loads (census ok) behind
  `implemented() == false`; oracle fixtures S1–S4 generated, with the nondeterminism floor
  recorded.

**M1 — Correct native v2 (experimental).**
- **Work:** B1–B8, C1, C2.
- **User value:** `focr ocr --model /path/to/Unlimited-OCR-2` (raw BF16, experimental flag) works
  for single pages (base) and unwindowed multi-page.
- **Exit:** L0–L4 green on S1–S3; v1 gate green.

**M2 — Certified opt-in model.**
- **Work:** C3, C4, E1, E3, E4, E7, F1, F2, T1 (partial).
- **User value:** `focr pull unlimited-ocr-2`; `--model unlimited-ocr-2` for single pages and
  multi-page, with per-page JSON and figures.
- **Exit:** corpus gates green; artifact published; perf ledger rows published.

**M3 — Unlimited-horizon books.**
- **Work:** D1–D5, E2, T2, T1 (full), F3.
- **User value:** 1,000-page PDFs on CPU with bounded memory, streaming, resumable; robot `page` and
  `window` events; the library document API.
- **Exit:** §7.5 green; watchdog green; e2e (d)/(e) green.

**M4 — Evidence-gated defaults.**
- **Work:** C5, G1, G2, E5, E6.
- **Exit:** the decision table published, with defaults flipped only where the gates pass.

**M5 — Ecosystem and further perf (backlog).**
- **Work:** F4 bf16 KV; F5 v2 batch spine; H1 iOS/WASM/site v2 lanes.

**Critical path.** A1 → (B1, B2) → B3 → B4 → B5 → B6 → C2 → C3/C4 → E4 → (D1 → D2) → C5 → E5.
A2, A3, A4, A6, B7 and T4 run in parallel off the start.

---

## 12. Task graph (mirrored into beads; labels in brackets)

Format: **ID — title** [labels] (deps). Each bead carries the full spec excerpt from §5 that it
needs, acceptance criteria, tests, logging requirements and rationale.

**Epic U2.** Unlimited-OCR-2: additive model + unlimited-horizon long-document parsing + evidence-gated defaults [uocr2,epic].

**A — Foundation and identity**
- **U2-A1** — Truth pack v2: pin sources, write OQ-V2 answers (SR-SWA, page2d RoPE, decode contract, preprocessing, tokenizer, windowed semantics), register in plan §13 and OQ_INDEX [uocr2,research,conformance] ()
- **U2-A2** — Model-identity hardening: absent model_id ≡ v1 forever; capability methods replace `== default_arch()` identity checks; census-based raw-safetensors arch detection [uocr2,identity] ()
- **U2-A3** — Sharded safetensors sources (index.json + N shards) for raw load and `focr convert`, with per-shard SHA pins [uocr2,weights,convert] ()
- **U2-A4** — Tokenizer v2: added non-special tokens 128827–128830; v2 tokenizer pin; conformance fixtures vs HF [uocr2,tokenizer,conformance] (A1)
- **U2-A5** — Register `unlimited-ocr-2` ModelArch + embedded 2,819-tensor census + pins + recipe id (implemented=false until B6) [uocr2,identity] (A1,A2,A3)
- **U2-A6** — v2 reference oracle generator + fixtures S1–S5 + nondeterminism floor [uocr2,conformance,oracle] (A1)
- **U2-A7** — HF v2 on CPU feasibility (OQ-V2-4) + CPU baseline decision [uocr2,conformance,research] (A6)

**B — Native SR-SWA decoder**
- **U2-B1** — page2d coordinates + decode state machine + RoPE tables [uocr2,decoder,rope] (A1)
- **U2-B2** — Token Sieve router math (proj/Gemma-norm/RoPE/kbar/scores/Page Gate/entmax top-M/top-K/bias) with scalar oracle [uocr2,decoder,attention] (A1,B1)
- **U2-B3** — Block-sparse biased attention (local ∪ selected) prefill + decode, dense scalar oracle + gathered path [uocr2,decoder,attention,kernel] (B2)
- **U2-B4** — SR-SWA KV/router caches, block page ranges, incremental kbar, memory accounting [uocr2,decoder,memory] (B1,B2)
- **U2-B5** — Decoder integration: attention-kind dispatch; v2 prefill/decode; v1 byte-identity preserved [uocr2,decoder] (B3,B4,A5,T4)
- **U2-B6** — v2 decode contract: prompt, guard, n-gram, lengths, `<PAGE>` stream, postprocess; flip implemented=true (experimental) [uocr2,decoder,sampler] (A4,B5,B7)
- **U2-B7** — v2 preprocessing: base-1024 letterbox multi-page (273/page), gundam max 16 tiles, small-image rule [uocr2,preprocess] (A1)
- **U2-B8** — Vision reuse proof: byte-identical vision tensors v1↔v2 + vision rungs under v2 [uocr2,vision,conformance] (A5,A6)

**C — Parity and certification**
- **U2-C1** — Ladder L0–L2 for v2 incl. routing-equality rung + margin diagnostics [uocr2,conformance,test] (B5,B7,A6)
- **U2-C2** — Ladder L3–L4 + stateless≡cached metamorphic + cross-page invariants [uocr2,conformance,test] (C1,B6)
- **U2-C3** — Conservative int8 artifact for v2 (recipe, convert from shards, loader gate) [uocr2,quant,convert] (A5,A3)
- **U2-C4** — L5 corpus gates for the v2 int8 artifact (single-page corpus, multi-page fixtures, hard pages) + DISC entries [uocr2,conformance,quant] (C2,C3)
- **U2-C5** — Ground-truth head-to-head v1 vs v2 (single page + multi-page + long docs) → decision table [uocr2,eval] (C4,D2)

**D — Unlimited-horizon (books)**
- **U2-D1** — Windowed decoding: planner, lookahead, retention, splice, forced `<PAGE>`, fallback, stitching [uocr2,decoder,longdoc] (B6,B4)
- **U2-D2** — Windowed validation: degenerate equivalence, invariants, accuracy vs unwindowed, optional SGLang runs [uocr2,conformance,longdoc] (D1,A6)
- **U2-D3** — Memory bounding: retire-on-advance (exact under guard + detector), memory estimator, `--page-window auto` [uocr2,memory,longdoc] (D1)
- **U2-D4** — Durable resume: window checkpoints in fsqlite, `--resume`, byte-identical resume [uocr2,longdoc,storage] (D1)
- **U2-D5** — Streaming output: per-page emission, robot `page`/`window` events, progress/ETA, `--page-window` and `--max-page-tokens` flags [uocr2,cli,robot,longdoc] (D1,E1)

**E — Product surface**
- **U2-E1** — CLI and robot basics for v2: `--model` id aliases, crop-mode semantics, `focr models` row, `run_complete` fields, schema and surface-matrix rows [uocr2,cli,robot] (B6)
- **U2-E2** — Library document API with streaming, window and resume options [uocr2,library] (D4,D5)
- **U2-E3** — Multi-page structured output: per-page layout/figures/JSON from `<PAGE>` segmentation [uocr2,cli,postprocess] (B6)
- **U2-E4** — Distribution: manifest entry, artifact parts, `focr pull unlimited-ocr-2`, install text, HF mirror coordination [uocr2,dist,release] (C4)
- **U2-E5** — Evidence-gated per-mode default policy and flip [uocr2,policy] (C5,E4,A2,G2)
- **U2-E6** — Documentation: README/AGENTS/CHANGELOG/plan/DISCREPANCIES/PERF_LEDGER [uocr2,docs] (E4,E5)
- **U2-E7** — Health, doctor and triage awareness of v2 (sidecar hash checks, recommendations) [uocr2,cli,doctor] (E4)

**F — Performance**
- **U2-F1** — Same-invocation v1 vs v2 baseline + FOCR_TIMING rows + PERF_LEDGER [uocr2,perf] (C2)
- **U2-F2** — Exact SR-SWA levers: candidate-only scoring, incremental kbar, gathered layout, prefill parallelism [uocr2,perf,kernel] (F1)
- **U2-F3** — Multi-page vision cost at 1024/page: measure + exact levers; once-per-page vision in windowed mode [uocr2,perf,vision] (F1,D1)
- **U2-F4** — bf16 KV storage option (lossy, CER kill-switch) [uocr2,perf,memory] (D3,C4)
- **U2-F5** — v2 continuous-batch spine for ocr-batch (backlog) [uocr2,perf,batch] (C4)

**G — Gundam single page**
- **U2-G1** — Gundam v2 parity + certification (non-degenerate routing; int8 flip sensitivity) [uocr2,conformance,preprocess] (C2,C3,B7)
- **U2-G2** — v2 single-page crop-mode decision (base vs gundam) by ground truth + CPU time [uocr2,eval] (G1,C5)

**T — Testing infrastructure**
- **U2-T1** — Model-gated e2e script `scripts/uocr2_e2e.sh` with NDJSON logging, scenarios (a)–(i) [uocr2,test,e2e,logging] (E1,E3,D4,D5)
- **U2-T2** — Concurrency watchdog for windowed v2 (many_pages_without_deadlock) [uocr2,test] (D1)
- **U2-T3** — Property/metamorphic suite additions [uocr2,test] (B2,B6,D1)
- **U2-T4** — v1 golden capture + non-regression gate in check.sh (before any shared refactor) [uocr2,test,conformance] ()

**H — Ecosystem**
- **U2-H1** — iOS/WASM/site v2 lanes (int4 calibration, artifacts, catalog) (backlog) [uocr2,ios,wasm] (E4)

**Related existing beads** (relation edges, not blockers):
- bd-3jo6 (Model Zoo epic);
- bd-3p8y and bd-aio5 (HF mirror);
- bd-wp8.2.2 (robot schema golden);
- bd-mgd3 (per-page progress telemetry);
- bd-paom and bd-3gyt (PDF document walk).

---

## 13. Risks and mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Discrete routing makes parity fragile (gundam, fallback queries) | M | H | Base mode first, where selection is degenerate; routing-equality rung with margin diagnostics; tie-flip discrepancy class with a ceiling |
| R2 | Reference stack drift (transformers 4.52.1 vs our 4.57.1) | H | M | Separate pinned venv; provenance checks (`check_oracle_provenance.py` extended) |
| R3 | HF config quirks: the snapshot `_CKPT_SP`, the `DASH_STAGE0` env, `_PAGE_SCOPE_PREFILL` set only inside `infer` | M | H | Oracle only via `infer()`; the truth pack documents the snapshot values (§5.6) |
| R4 | CPU vision cost at 1024 per page makes short multi-page slower than v1 | H | M | Honest perf ledger; per-mode gates treat cost as a budget; vision levers (F3) |
| R5 | Memory for long documents (120 KiB per token in f32) | H | H | Windowing (D1), retirement (D3), bf16 KV (F4), memory estimator, `auto` window size |
| R6 | Silent misload of v1 artifacts after a default change | M | H | A2 first, with tests that simulate a changed default |
| R7 | The shared-decoder refactor perturbs v1 | M | H | T4 golden gate before B5; enum dispatch; no shared mutable state |
| R8 | Windowed mode is approximate (Baidu's words) | M | M | Accuracy validation vs unwindowed and ground truth; W ≥ N degenerate equivalence; W user-tunable |
| R9 | The windowed-mode reference only exists as a GPU SGLang wheel | H | L | Native semantics from the spec; invariants plus ground truth instead of byte parity; optional SGLang runs |
| R10 | Distribution blocked by the missing HF org (bd-3p8y) | H | L | GitHub release parts are primary |
| R11 | License ambiguity (no LICENSE in the HF v2 repo) | L | M | Use the GitHub LICENSE text and HF metadata; record both in the truth pack and the artifact notice |
| R12 | Process ceremony crowding out shipping (suite reward-hacking patterns) | M | M | Every bead closes on cited evidence (tests, fixture diffs, ledger rows); no spec-only "progress" beads beyond A1 |

---

## 14. Open questions register (OQ-V2)

| ID | Question | Status | Resolution / owner |
|---|---|---|---|
| OQ-V2-1 | SR-SWA candidate semantics (prefill per-query own-page image blocks; decode `[bmin,bmax]` allow; fallback to all remote) | RESOLVED | §5.6.4 (`MD:108-123,339-348`, `MU:481-528`, `SB:2629-2737`) |
| OQ-V2-2 | Exact router constants at inference (K=12, M=16, scale 128/√128, floor 1e-4, σ=1, head_cls) | RESOLVED | §5.6 (`MD:296-335`, config snapshot) |
| OQ-V2-3 | Parity tolerance given reference bf16 prefill SDPA vs our f32 | OPEN | U2-A6 floor + U2-C1 |
| OQ-V2-4 | Does HF v2 run on CPU (dense fallback) and reproduce GPU tokens? | OPEN | U2-A7 |
| OQ-V2-5 | Are the `indexer.*` projections used at inference? | RESOLVED (no) | §5.6.7; keep in artifact (D8) |
| OQ-V2-6 | 1-D RoPE fallback for marker-less inputs | RESOLVED (reject such inputs) | §5.5; v2 prompts always carry markers |
| OQ-V2-7 | n-gram vs guard order hazard (HF can emit id 0) | RESOLVED (forced wins) | DISC-U2-2; assert on fixtures |
| OQ-V2-8 | Tokenizer handling of non-special added tokens 128827–128830 | OPEN | U2-A4 conformance |
| OQ-V2-9 | Windowed-mode accuracy vs unwindowed on long documents | OPEN | U2-D2 |
| OQ-V2-10 | v2 single-page base vs gundam quality and cost | OPEN | U2-G2 |
| OQ-V2-11 | Retire-on-advance is output-exact under the guard | OPEN (proof obligation) | U2-D3, §5.12 |
| OQ-V2-12 | Licenses of ground-truth evaluation sets beyond MarathonDocBench (MPDocBench, OmniDocBench) for local evaluation | OPEN | U2-C5 |
| OQ-V2-13 | Does the conservative int8 FFN recipe hold v2 CER within budget? | OPEN | U2-C4 |
| OQ-V2-14 | Activated parameters (report: "A570M" in a figure vs 500M in text) | DEFERRED (docs only) | — |

---

## Appendix A — Token and image counts

| Mode | Image tokens per page | Prompt tokens per page incl. `<PAGE>` |
|---|---|---|
| Base 1024 | 273 | 274 |
| Gundam 2×2 tiles | 273 + (21)·20 = 693 | 694 |
| Gundam 4×4 tiles (max 16) | 273 + (41)·40 = 1,913 | 1,914 |
| v1 multi-page (640 squash) | 111 | 111 (+ separator handling per v1) |

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

## Appendix C — Code locations touched (v1 baseline, for implementers)

| Area | Current location |
|---|---|
| Decoder constants | `src/native_engine/decoder.rs:82-109` (incl. `ROPE_THETA` at :98; RoPE table at :157-238) |
| MoE constants and router | `src/native_engine/moe.rs:53-76,118-120,482-500` |
| R-SWA cache and attention | `src/native_engine/rswa.rs:46-50,180-219,484-624,997-1007` |
| Decode step | `src/native_engine/decoder.rs:1428-1502` |
| Multi-page path | `src/native_engine/mod.rs:2602-2830`, `build_prompt_multi` at :4399-4419; 32K checks at :2750, :2783; `MULTI_PAGE_BASE_SIZE` 640 at :358 |
| Sampler (n-gram, EOS, max_length) | `src/native_engine/sampler.rs:27-45,514-640` |
| Postprocess (`<PAGE>`, PageStream) | `src/native_engine/postprocess.rs:38,497-560` |
| Tokenizer specials | `src/tokenizer/mod.rs:86-119,489-532` |
| Identity | `src/native_engine/weights.rs:60-110,1238`; `src/quant/focrq.rs:588-606`; `src/native_engine/model_arch.rs:143-257,545-571` |
| Census | `src/native_engine/unlimited_ocr_census.rs:24-55,328-394`; `src/native_engine/unlimited_ocr_manifest.json` |
| Convert and recipe | `src/quant/convert.rs:46,60,124`; `src/quant/recipe.rs:122-129,236`; `src/cli.rs:2350,2420` |
| Distribution | `src/dist.rs:35,93-185,485-520`; `models/manifest-v2.json` |
| Preprocessing | `src/preprocess/mod.rs:47-75,141-144,251-283,350-416` |
| Oracle scripts | `scripts/gen_reference_fixtures.py:92-125`; `scripts/oracle_bridge.py`; `scripts/oracle_nondeterminism_floor.py`; `scripts/check_oracle_provenance.py` |
| Truth pack | `docs/truth-pack/{OQ_INDEX.md,PINNED_SOURCES.md,SOURCE_HASHES.md,oq/*.md}` |
