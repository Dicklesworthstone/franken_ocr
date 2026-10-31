# Truth Pack — Secondary (non-blocking) OQ Answers

**Bead context:** `PM1-census-generator` (secondary / non-blocking OQs).
**Pinned model source:** HF commit `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`,
snapshotted at `/Users/jemanuel/projects/franken_ocr/docs/truth-pack/snapshots/`.

All definitive numbers are LINE-BACKED. Where the pinned source cannot answer an OQ
(e.g. needs the external NVFP4 repo), it is explicitly marked a blocker with the exact
artifact still required.

---

## OQ-9 — Recompute the "~500M activated params" from the index/config

### Question (verbatim)

> OQ-9 (recompute ~500M activated params from the index tensor sizes).

### ANSWER (definitive — analytic, with a caveat)

**The index does NOT carry per-tensor shapes** — `model.safetensors.index.json` has only
`{"metadata": {...}, "weight_map": {...}}` (verified: its top-level keys are exactly
`['metadata', 'weight_map']`, and each value is just the shard filename). So activated
params are recomputed **analytically from `config.json` dims**, with the index used only to
confirm *which* modules exist and their multiplicities (section (a) of `CENSUS.md`).

**Activated parameters (language model), top-6-of-64 routing:**

| Component | Params | Note |
|---|---:|---|
| Attention × 12 layers | 78.64 M | q+k+v+o, each 1280×1280, ×12 |
| Dense MLP (layer 0) | 26.30 M | 2·1280·6848 + 6848·1280 |
| Shared expert × 11 MoE layers | 75.69 M | width = 896·2 = 1792, ×11 |
| Active routed experts (top-6) × 11 layers | 227.08 M | 6 × (2·1280·896 + 896·1280) × 11 |
| Router gate (1280×64) × 11 | 0.90 M | negligible |
| `lm_head` (129280×1280) | 165.48 M | |
| **Activated subtotal (compute, excl. embed lookup)** | **≈ 574.1 M** | |
| `embed_tokens` (129280×1280) | 165.48 M | lookup table, not a matmul |
| **Activated incl. embed table** | **≈ 739.6 M** | |

**Interpretation of "~500M":** the figure is the **language-model activated compute**
budget. Our exact recompute is **≈574 M** counting `lm_head`, or **≈408 M** if you also
exclude `lm_head` (decode token-by-token, head is one row) — both straddle the round
"~500M" estimate, which is therefore corroborated as the right order of magnitude for the
*activated* (not total) param count. Total language-model params (all 64 experts dense) =
**≈2.93 B**; whole-model `total_size` ⇒ **≈3.336 B** params bf16 (`6 672 212 480 / 2`),
the extra ~0.40 B being the SAM + CLIP vision towers + projector.

### Driving dims (all `config.json`)

- `hidden_size: 1280` — `config.json:29,97`
- `intermediate_size: 6848` — `config.json:30,98`
- `moe_intermediate_size: 896` — `config.json:34,102`
- `n_routed_experts: 64`, `num_experts_per_tok: 6` — `config.json:36,39 / 104,107`
- `n_shared_experts: 2` — `config.json:37,105`
- `num_hidden_layers: 12`, `first_k_dense_replace: 1` — `config.json:40,28`
- `vocab_size: 129280` — `config.json:51,118`
- `lm_head: true` (separate head, not tied) — `config.json:32,100`

> **Status: ANSWERED (analytic).** Definitive *given config dims*. A byte-exact
> per-tensor recompute would require dumping shapes from the safetensors header (the
> `.safetensors` file itself, not the index) — see "what to dump" in OQ-14. UNBLOCKS the
> perf/memory-budget bead and the MoE-routing kernel sizing.

---

## OQ-14 — Enumerate OUR quantizable-Linear set; note the exact NVFP4 module set needs the external repo

### Question (verbatim)

> OQ-14 (enumerate OUR quantizable-linear set = the ~2229; note the exact NVFP4 module set
> requires the external sahilchachra/Unlimited-OCR-NVFP4 scale keys — mark what to dump).

### ANSWER (definitive for OUR set; BLOCKED for the exact NVFP4 set)

**OUR quantizable-Linear candidate set (derivable from the pinned bf16 index) = 2244
`*_proj.weight` tensors (+ `lm_head` = 2245).** Full breakdown in `CENSUS.md` §(a):

| Bucket | Count |
|---|---:|
| Routed experts (`*.mlp.experts.*.{gate,up,down}_proj.weight`) | 2112 |
| Language attention (`*.self_attn.{q,k,v,o}_proj.weight`) | 48 |
| Shared experts (`*.mlp.shared_experts.{gate,up,down}_proj.weight`) | 33 |
| Dense MLP layer 0 (`layers.0.mlp.{gate,up,down}_proj.weight`) | 3 |
| Vision tower (`*.vision_model.*.self_attn.{qkv,out}_proj.weight`) | 48 |
| `lm_head.weight` | 1 |
| **TOTAL** | **2245** |

> The bead's "~2229" is an estimate; the real `*_proj.weight` count is **2244** (the
> estimate dropped the 48 vision projections and over-counted shared as 66 instead of 33 —
> these two corrections net out near the estimate but the precise set is 2244 / 2245).

**The pinned snapshot is the bf16 model and contains NO quantization metadata:**
- `config.json` has **no** `quantization_config` key (verified absent).
- No `nvfp4`/`fp4`/`weight_scale`/`scale_inv`/`amax`/`block_size`/`qweight` keys appear in
  any `.py` or `.json` in `snapshots/`. The only `quant` hits are an unrelated generic
  comment (`modeling_deepseekv2.py:1062-1064`: "Handle the case where the model is
  quantized" / `_pre_quantization_dtype`) and tokenizer vocab entries — neither defines an
  NVFP4 scheme.

**Therefore the EXACT NVFP4 module set is a BLOCKER:** which subset of the 2244 is actually
quantized to NVFP4 (and the per-block scales) is defined only by the external
`sahilchachra/Unlimited-OCR-NVFP4` checkpoint, which is **not present locally**.

### What to dump from the external NVFP4 repo (to resolve)

From `sahilchachra/Unlimited-OCR-NVFP4` obtain and dump:
1. Its **`model.safetensors.index.json`** `weight_map` — list every key, then diff the
   suffixes against our 2244. NVFP4 typically replaces `*.weight` with paired
   `*.weight` (packed fp4 / uint8) **plus** scale tensors. Dump the scale-key naming, e.g.
   `*.weight_scale`, `*.weight_scale_2`, `*.input_scale`, or `*.weight_global_scale`
   (NVFP4 = E2M1 4-bit + per-block FP8 E4M3 scale + per-tensor global scale).
2. Its **`config.json` `quantization_config`** block — `quant_method` (expect
   `"nvfp4"`/`"modelopt"`/`"compressed-tensors"`), `group_size` (NVFP4 block = 16),
   `ignore`/`exclude` list (which Linears stay bf16 — commonly `lm_head`, the router
   `mlp.gate`, embeddings, and frequently the **whole vision tower**).
3. Per-tensor **shapes + dtypes** from the safetensors headers, to confirm fp4-packing
   layout (packed last-dim/2, scale shapes `[out, in/group_size]`).

> **Status: ANSWERED for OUR candidate set (2244/2245); BLOCKED for the exact NVFP4
> set + scales** — needs the external `sahilchachra/Unlimited-OCR-NVFP4` `index.json` +
> `quantization_config` + safetensors headers dumped per above. UNBLOCKS the NVFP4
> dequant kernel bead and the quant-loader bead.

### The NVFP4 set, dumped (2026-10-31): RESOLVED

Source: `sahilchachra/Unlimited-OCR-NVFP4@2ac0aa28a3509c7056648ffc0d4f9fdb79a454d0`, its
`config.json` and the header of `model-00001-of-000001.safetensors` (7,102 tensors).

`quantization_config`: `quant_method: compressed-tensors` 0.16.0, `format:
nvfp4-pack-quantized`, targets `Linear`; weights only (`input_activations: null`), float
4-bit, symmetric, `strategy: tensor_group`, `group_size: 16`, scales `float8_e4m3fn`, no
KV-cache scheme. `ignore`: `lm_head`, `embed_tokens`, `sam_model`, `vision_model`,
`projector`, `image_newline`, `view_seperator`, the router `mlp.gate`, and every `norm`.

**Quantized: exactly 2,196 modules**, each stored as a `U8` packed weight, an `F8_E4M3`
`weight_scale` and an `F32` global scale:

| Bucket | Count |
|---|---:|
| Routed experts `{gate,up,down}_proj` | 2,112 |
| Shared experts `{gate,up,down}_proj` | 33 |
| Dense layer-0 MLP `{gate,up,down}_proj` | 3 |
| Language attention `{q,k,v,o}_proj` | 48 |

**Kept BF16:** `lm_head`, `embed_tokens`, the 11 router gates `mlp.gate`, and the whole
vision tower and projector (SAM, the CLIP `qkv_proj`/`out_proj`, embeddings).

Against our 2,245 candidates, NVFP4 leaves out the 48 vision projections and `lm_head`, and
quantizes the 48 attention projections our conservative recipe keeps high. So NVFP4
validates our router, `lm_head`, embedding and vision choices, and does **not** validate
keeping attention high: that choice rests on our own measurements (`NEGATIVE_EVIDENCE.md`).

---

## OQ-10 / OQ-11 / OQ-12 — DEFERRED (non-blocking); OQ-10 and OQ-12 since resolved below

### Status

The OQ-10/11/12 *question text* is not present in the local truth-pack (`SOURCE_HASHES.md`,
`oq/vision.md`, `oq/rope-and-config.md`, and the snapshots contain no verbatim OQ-10/11/12
statements). They are **not blocking** the `PM1-census-generator` bead — the census
(token/shape/buffer) is fully resolved from `config.json` + `index.json` above and in
`CENSUS.md`. They are marked **DEFERRED** here rather than answered, to avoid fabricating
content against questions whose authoritative wording I cannot quote.

### OQ-11 — int8 patch-embed conv: UNMEASURED, deferred (vision stays high-precision)

> **Question (now pinned from plan §13, `COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md:888`,
> and bead `bd-322.17`):** *"Is int8 viable for the patch-embed conv specifically
> (cleaner stats than mid-network ViT)?"*

**Disposition: UNMEASURED.** No in-repo experiment isolates the patch-embed conv. The
only quant measurements on record (`docs/NEGATIVE_EVIDENCE.md`) are *decoder* levers —
int8/int4 attention `q/k/v/o`, KV cache, and `lm_head` (`FOCR_ATTN_GEMM`, `FOCR_INT8_KV`,
`FOCR_LMHEAD_INT4`, `FOCR_QKV_FUSED`, `FOCR_ATTN_SERIAL`); none quantize any vision
tensor. The vision tower is validated **f32-only** (`docs/VISION_PARITY_RESULTS.md`:
full-tower cosine 0.9996, *"quantized kernels are a separate, later comparison"*). There
is therefore **no measurement** — neither for the whole vision tower nor for the
patch-embed conv in isolation — that bears on this question.

**Recorded hypothesis (untested):** the patch-embed conv is the *most promising*
vision-int8 candidate because its activation stats are the cleanest in the tower — it is
non-overlapping (`k=stride`, so im2col is exact, plan §6.4 `:494`) and sits *before* any
LayerNorm/GELU, so it never sees the post-LayerNorm/post-GELU outliers that *"wreck
low-bit activation quant"* mid-network (plan §6.4 `:494`, §10 risk table `:839`). This is
a hypothesis, **not** a result.

**What would resolve it:** an isolated A/B that quantizes **only** the patch-embed conv
(per-channel int8 weights + per-token dynamic activation quant) vs the f32 baseline,
gated on the 20-page **CER** budget, behind a kill-switch — run **after** decoder parity
is locked (Phase 4+; depends on the PM1-oracle per-layer activation dumps). Until that
A/B is run, vision stays BF16/F32 by **doctrine #2** (quantize the decoder FFN/experts
ONLY); OQ-11 remains a **deferred, non-blocking** experiment marker.

### Why deferred (not blocking)

- The census bead's four deliverables (a–d) are complete and line-backed without
  10/11/12.
- Answering them blind risks hallucinating the question — explicitly disallowed by the
  assignment. The index/config can almost certainly *support* answers once the questions
  are pinned; the gap is the **question text**, not the source.

### What would resolve them

Provide the verbatim OQ-10/OQ-11/OQ-12 statements (from the Phase -1 OQ ledger /
bead tracker). Given likely topics, the resolving source is already local:
- If about **tokenizer / special tokens / `<image>` id**: `tokenizer.json`,
  `tokenizer_config.json`, `special_tokens_map.json` (all present in `snapshots/`).
- If about **vision/SAM neck or projector dims**: already answered in `oq/vision.md`;
  cross-reference `deepencoder.py` + `config.json` `vision_config`/`projector_config`.
- If about **chat template / conversation roles**: `conversation.py`,
  `tokenizer_config.json` chat_template.
- If about **generation/sampling config**: `modeling_unlimitedocr.py:787` `infer(...)`
  defaults (`max_length=32768`, `temperature=0.0`, `no_repeat_ngram_size`, etc.).

> **Status: DEFERRED, non-blocking.** Needs only the verbatim question text; the backing
> source files are already in `snapshots/`. Does NOT block the census bead.

## OQ-10 — RESOLVED (2026-10-31, bead bd-322.16)

> **Question (plan §13):** does the community GGUF (llama.cpp PR #17400) quantize the
> router and experts identically to NVFP4, and in particular keep the router gate in f32?

Source: `sahilchachra/Unlimited-OCR-GGUF@0dc781d8a23f52963918ebd5b2d1b9fe61504661`, the GGUF
headers of `Q4_K_M` (the recommended file), `IQ2_M` (the smallest) and `Q8_0`, and
`mmproj-Unlimited-OCR-F16.gguf`. Architecture `deepseek2-ocr`, 155 language tensors,
imatrix-calibrated.

**The router gate stays F32 in all three:** `blk.N.ffn_gate_inp.weight` is F32 in all 11 MoE
layers. NVFP4 keeps it BF16, the stored precision. So both leave the router unquantized,
in different float formats. llama.cpp's quantizer skips `ffn_gate_inp` by default, so this
is the tool's rule, not a choice made for this model.

The rest is **not** identical to NVFP4:

| Tensor | Q4_K_M | IQ2_M | Q8_0 | NVFP4 |
|---|---|---|---|---|
| routed experts gate/up | Q4_K | IQ2_S | Q8_0 | FP4 |
| routed experts down | Q5_0 ×6, Q8_0 ×5 | IQ4_NL | Q8_0 | FP4 |
| shared experts | Q4_K, down Q4_K/Q6_K | IQ2_S | Q8_0 | FP4 |
| attention q/k | Q4_K | IQ2_S | Q8_0 | FP4 |
| attention o | Q4_K | IQ3_S | Q8_0 | FP4 |
| attention v | Q4_K ×6, Q6_K ×6 | Q4_K | Q8_0 | FP4 |
| `lm_head` (`output`) | Q6_K | Q5_K | Q8_0 | BF16 |
| `embed_tokens` | Q4_K | IQ3_S | Q8_0 | BF16 |
| router gate | F32 | F32 | F32 | BF16 |
| vision + projector | F16/F32 in `mmproj` | same | same | BF16 |

GGUF quantizes `lm_head` and the embeddings as well, which NVFP4 does not. Both keep vision
and the router high, which matches our recipe. Neither keeps attention high.

## OQ-12 — RESOLVED (2026-10-31, bead bd-322.18)

> **Question (plan §13):** confirm the benchmark cells (TPS, v1.6 peers, 40+pg edit
> distances) from the PDF tables, against the secondary sources §2.5 cites.

Source: `Unlimited-OCR.pdf`, sha256 `d4cc0b2e…74a8b` (`SOURCE_HASHES.md`), 14 pages. The
pinned HF revision `3a7f4db` and the pinned GitHub commit `7e98aff` serve byte-identical
copies.

| Plan §2.5 cell | Paper | Verdict |
|---|---|---|
| v1.5 overall 93.23, DeepSeek-OCR 87.01, +6.22 | Table 1 | VERIFIED |
| v1.5 Text-Edit 0.038, Formula-CDM 92.61, TEDS 90.93, TEDS-S 94.07, Read-order 0.045 | Table 1 | VERIFIED |
| v1.6 overall 93.92 | Table 1 | VERIFIED |
| v1.6 peers: PaddleOCR-VL 1.6 96.33, MinerU 2.5-Pro 95.69, PaddleOCR-VL 1.5 94.50 | absent | NOT IN THE PAPER. Stays REPORTED. |
| v1.6 peers: Qianfan-OCR 93.12, DeepSeek-OCR-2 91.09 | Table 1: 93.90 and 90.25 | CONTRADICTED. The plan's figures come from another source. |
| Long-horizon edit 2pg 0.0362, 10pg 0.0526, 40+pg 0.1069 | Table 3 | VERIFIED |
| ~5580 TPS, +12.7% vs DeepSeek-OCR | §5.2: 5580 vs 4951 | VERIFIED, with scope: GPU tokens/s at 512 concurrency, "Base" mode |
| +35% at 6144 out-tokens | Table 4: 7847.71 vs 5822.87 | VERIFIED, with scope: a theoretical ceiling with a 10-token prefill, not a document run |

What the plan left out:
- Table 1's v1.6 block lists end-to-end VLMs only (HunyuanOCR 89.95, DeepSeek-OCR 2 90.25,
  dots.ocr 90.77, FireRed-OCR 93.26, Logics-Parsing-v2 93.33, Qianfan-OCR 93.90), and on
  it Unlimited-OCR is first at 93.92. That ranking is the paper's "end-to-end SOTA" claim.
  The plan's "behind PaddleOCR-VL 1.6 and MinerU 2.5-Pro" compares against models the
  paper did not include. Both statements can be true. Neither is wrong to quote, as long as
  its source is named.
- Table 3 is not monotonic: 15 pages 0.0787, 20 pages 0.0572. At 40+ pages, Distinct-35
  falls to 96.90%. The paper attributes the 40+ errors to small text under "Base" mode.
- No TPS figure in the paper is a CPU figure, so none of them bounds this port.
