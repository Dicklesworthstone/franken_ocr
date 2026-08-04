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

---

## OQ-10 / OQ-11 / OQ-12 — DEFERRED (non-blocking)

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
