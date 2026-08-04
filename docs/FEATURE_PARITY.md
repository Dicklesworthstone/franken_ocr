# FEATURE_PARITY.md — the running conformance scoreboard (FeatureUniverse / SurfaceMatrix)

**Bead:** `bd-322.25` (the fourth of the four `/porting-to-rust` documents). This is the **gauntlet's surface-parity input** (plan §8.5/§8.6): the single living table that accounts EVERY modeling feature, every op (§4.3), every CLI surface (§7), every robot event (§7.3), and every parity gate (§8.2) as `present | partial | missing | n/a | excluded`. The three-pillar release certification (bd-re8.13) and the release-readiness scorecard (bd-wp8.10) READ this file. **What is not enumerated here is silent coverage debt the gauntlet cannot catch** — so this file is the source of truth for surface parity, cross-checked against the §4.3 op map, §7.2 subcommand table, §7.3 robot events, and §8.2 L0–L5 gates.

**Provenance.** Pinned source @ HF `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5` / GitHub `7e98affeacba24e95562fbaa234ddb89b856874a` ([`truth-pack/PINNED_SOURCES.md`](truth-pack/PINNED_SOURCES.md); SHA-256s in [`truth-pack/SOURCE_HASHES.md`](truth-pack/SOURCE_HASHES.md)). Feature rows trace to THE SPEC's `[SPEC-NNN]` clauses ([`truth-pack/EXISTING_UNLIMITED_OCR_STRUCTURE.md`](truth-pack/EXISTING_UNLIMITED_OCR_STRUCTURE.md)) and to the Rust design ([`PROPOSED_ARCHITECTURE.md`](PROPOSED_ARCHITECTURE.md)). Counts are line-backed in [`truth-pack/CENSUS.md`](truth-pack/CENSUS.md).

**Status (measured, 2026-07-06).** The Phase −1 all-`missing` seed is superseded: every row now carries a MEASURED status with its evidence cited in the Notes cell (FeatureUniverse swept under bd-1630; SurfaceMatrix under bd-re8.13/bd-wp8.11). The `Bead` column names the bead that *delivered* (or still owes) the feature; the `Parity` column is the gate level (L0–L5, §8.2) that proves it; `Status` flips to `present` only when that bead landed AND its parity gate is green. **`partial` never rounds up to `present`. `excluded` still counts as coverage debt** (it is enumerated, with a reason).

---

## 0. Status legend, parity-level legend, and the scoring contract

### Status (the SurfaceMatrix cell value)
| Status | Meaning |
|--------|---------|
| `present` | Implemented AND its parity gate (the `Parity` column) is green. |
| `partial` | Implemented but its parity gate is not fully green, OR only some sub-cases pass. **Never rounds up to `present`.** |
| `missing` | Not implemented. (The seed value of every row.) |
| `n/a` | Not applicable to this port (no equivalent surface). |
| `excluded` | Deliberately out of v1 scope, with a reason. **Counts as coverage debt** — it is enumerated, not omitted. |

### Parity level (the L0–L5 gate that proves a row, plan §8.2)
`L0` preprocessing (exact) · `L1` per-op (cosine ≈ 1.0) · `L2` per-layer (cosine ≈ 1.0, max-abs-diff ledgered) · `L3` logits (within *measured* quant tolerance + argmax match) · `L4` token (exact where reference deterministic) · `L5` end-to-end OCR (CER/TEDS/Formula-CDM within documented budget). `SURF` = surface/contract parity (CLI/robot/exit-code), proven by a contract test, not the numeric ladder.

### Requirement level (conformance accounting, plan §8.6)
`MUST` / `SHOULD` / `MAY` per the `ConformanceTest` trait. **≥0.95 of MUST clauses must be enumerated-and-covered to claim conformance.** Every MUST row below is line-backed to a `[SPEC-NNN]` clause.

### Scoreboard rollup (seed)
Two enumerated populations: the **FeatureUniverse** (numbered modeling-feature / op / quant rows, §1–§11) and the **SurfaceMatrix** (un-numbered CLI / robot / gauntlet / alien rows, §12–§15). Both are accounted; the gauntlet reads both.

| Metric | FeatureUniverse (§1–§11, numbered #1–#128) | SurfaceMatrix (§12–§15) | Total |
|--------|-------------------------------------------:|------------------------:|------:|
| Total enumerated rows | **128** | **59** | **187** |
| `present` | 115 † | 46 | 161 |
| `partial` | 7 † | 10 | 17 |
| `missing` | 2 † | 3 | 5 |
| `excluded` (coverage debt, reasoned) | 4 | 0 (`pdf` re-scored `partial` — scanned path shipped v0.2.0) | 4 |
| `n/a` | 0 | 0 | 0 |

> † §1–§11 statuses were flipped from the Phase −1 seed to MEASURED values on
> 2026-07-06 (bd-1630): three read-only evidence sweeps scored every numbered
> row against the shipped tree and the armed-green L0–L5 ladder; each row's
> Notes cell cites its implementation site and proving gate. The SurfaceMatrix
> (§12–§15) was brought current the same day (bd-re8.13).
> `tests/surface_matrix.rs` locks enumeration and recomputes this rollup.

| Conformance metric | Value (measured 2026-07-06) |
|--------------------|----------------------|
| MUST clauses enumerated (across both populations) | 93 |
| MUST coverage (enumerated / SPEC MUST) | **1.00** (every `[SPEC-NNN]` MUST clause has a row) |
| MUST rows `present` / total MUST rows | **139 / 147** (the 8 non-present MUST rows are enumerated debt: see the `partial`/`missing` rows) |

> The 4 `excluded` rows are reasoned coverage debt (§16), not omissions; the gauntlet can only account what is listed.

> **CI doc-lint contract (bd-322.25 TESTS REQUIRED).** This file is parseable into the FeatureUniverse table: every feature row has a valid `Status` ∈ {present, partial, missing, n/a, excluded}, a valid `Parity` ∈ {L0..L5, SURF, n/a}, a `Bead` id (or `—`), and a `Req` ∈ {MUST, SHOULD, MAY, n/a}. The lint emits one NDJSON line `{doc, n_features, n_present, n_partial, n_missing, n_excluded, must_coverage}` and fails if any row is malformed or if a `[SPEC-NNN]` MUST clause has no row. The enumerated MUST set must cover the §4.3 op map + the §7.2 surface.

---

## 1. Modeling features — preprocessing & prompt (SPEC §1–§3)

| # | Feature | SPEC | Req | Status | Parity | Module | Bead | Notes |
|---|---------|------|-----|--------|--------|--------|------|-------|
| 1 | Image load + EXIF-transpose + RGB | SPEC-020 | MUST | present | L0 | preprocess/mod.rs | bd-1gv.2 | decode_image + EXIF-transpose (preprocess/mod.rs:379); L0 armed green |
| 2 | Normalize ToTensor→(0.5,0.5) ⇒ [-1,1] | SPEC-021 | MUST | present | L0 | preprocess/normalize.rs | bd-1gv.2 | [-1,1] normalize (mod.rs:667); L0 anchor NORM_LO/HI |
| 3 | Pad-to-square gray (127,127,127) | SPEC-022 | MUST | present | L0 | preprocess/pad.rs | bd-1gv.2 | gray-127 pad (mod.rs:73,479); L0 anchor GRAY_PAD==127 |
| 4 | Crop decision (crop_mode, ≤640 both ⇒ [1,1]) | SPEC-023 | MUST | present | L0 | preprocess/tile.rs | bd-1gv.3 | crop decision ≤640⇒[1,1] (mod.rs:548); L0 + unit tests |
| 5 | `dynamic_preprocess` tiling (min 2/max 32, row-major) | SPEC-024 | MUST | present | L0 | preprocess/tile.rs | bd-1gv.3 | dynamic_preprocess tiling (mod.rs:546-610); gundam census test |
| 6 | `find_closest_aspect_ratio` (tie-break larger area) | SPEC-025 | MUST | present | L0 | preprocess/tile.rs | bd-1gv.3 | find_closest_aspect_ratio tie-break (mod.rs:637,1184); L0 |
| 7 | crop_ratio (width_crop_num, height_crop_num) | SPEC-026 | MUST | present | L0 | preprocess/tile.rs | bd-1gv.3 | CropGrid (mod.rs:173,203,230); L0 |
| 8 | Token query counts (base 16, tile 10) | SPEC-027 | MUST | present | L0 | preprocess/tile.rs | bd-1gv.3 | num_queries base16/tile10 (mod.rs:138); base_global_placeholder_is_273 |
| 9 | Image-token id-stream layout (2D, 273/1024-view) | SPEC-028 | MUST | present | L0 | preprocess/tile.rs | bd-1gv.3 | 273/1024-view layout (mod.rs:1106); L0 anchor SLOTS_PER_1024_VIEW |
| 10 | Non-crop branch single global block | SPEC-029 | MUST | present | L0 | preprocess/tile.rs | bd-1gv.3 | non-crop single global block (mod.rs:83,1251); L0 |
| 11 | BOS prepend + images_seq_mask | SPEC-030 | MUST | present | L0 | preprocess/tile.rs, tokenizer/ | bd-1gv.3 | BOS prepend + images_seq_mask (build_prompt native_engine/mod.rs:2483); L4 + L2 inputs_embeds |
| 12 | Image tensor packing `images=[(crop,ori)]` | SPEC-031 | MUST | present | L0 | preprocess/tile.rs | bd-1gv.3 | views() tiles-then-global packing (native_engine/mod.rs:2466); L0 |
| 13 | `valid_img_tokens` accounting (metric only) | SPEC-032 | MAY | excluded | n/a | — | — | (unchanged; §16 coverage debt — metric-only, not in the forward) |
| 14 | Multi-image path (infer_multi, no-crop) | SPEC-033 | MUST | present | L5 | mod.rs, connector.rs | bd-1gv.25 | recognize_multi_page ships the full infer_multi contract (Base-640 x N at ONE <image> position, single cross-page prefill+decode, ngram_window 1024, 32K guard, finalize_multi <PAGE> assembly) + ocr-batch --multi-page; real-model 2-page e2e green (227-token prefill == exact census); per-page streaming + PDF wiring split to follow-up; L5 CER vs infer_multi oracle = bd-1gv.26 |
| 15 | Prompt `plain` template (empty sep/roles) | SPEC-034 | MUST | present | L4 | tokenizer/, preprocess/tile.rs | bd-1gv.1 | plain template, empty sep/roles (native_engine/mod.rs:125); L4 |
| 16 | Prompt split on `<image>` (add_special_tokens=False) | SPEC-035 | MUST | present | L0 | tokenizer/ | bd-1gv.1 | split on <image> add_special=False (tokenizer/mod.rs:532); L0/L4 + tokenizer conformance |
| 17 | Roles `<\|User\|>`/`<\|Assistant\|>` (absent in plain output) | SPEC-036 | SHOULD | present | L4 | tokenizer/ | bd-1gv.1 | roles absent in plain output (native_engine/mod.rs:126); L4 |
| 18 | Bicubic image resize (aspect-preserving) | SPEC-021/024 | MUST | present | L0 | preprocess/resize.rs | bd-1gv.2 | Pillow-exact bicubic (preprocess/pil_resample.rs); L0 exact under FOCR_RESAMPLE=pil-bicubic, default within bd-30me envelope |

## 2. Modeling features — tokenizer (SPEC-019, OQ-16)

| # | Feature | SPEC | Req | Status | Parity | Module | Bead | Notes |
|---|---------|------|-----|--------|--------|--------|------|-------|
| 19 | Byte-level BPE encode/decode (tokenizer.json) | UNRESOLVED-1/OQ-16 | MUST | present | L0 | tokenizer/mod.rs | bd-1gv.1 | byte-level BPE encode/decode (tokenizer/mod.rs); tokenizer_conformance round-trip |
| 20 | Pre-tokenizer `Sequence` + byte-fallback + merges | OQ-16 | MUST | present | L0 | tokenizer/mod.rs | bd-1gv.1 | pretok Sequence + byte-fallback (tokenizer/pretok.rs); conformance gate |
| 21 | Special tokens (bos 0/eos 1/pad/`<image>`128815/ref/det/grounding) | SPEC-014/019 | MUST | present | L0 | tokenizer/special.rs | bd-1gv.1 | special ids pinned (tokenizer/mod.rs:80-115); special_id_constants tests |
| 22 | Tokenizer conformance corpus (CJK/math/code/glyphs) | OQ-16 | MUST | present | L4 | tests/ | bd-1gv.1.1 | 14-category corpus + 49KB golden ids; conformance gate (L4) |

## 3. Modeling features — vision tower SAM (SPEC-040..046, OQ-15)

| # | Feature | SPEC | Req | Status | Parity | Module | Bead | Notes |
|---|---------|------|-----|--------|--------|--------|------|-------|
| 23 | SAM build params (768/12/12, global [2,5,8,11]) | SPEC-040 | MUST | present | L2 | vision_sam.rs | bd-1gv.5 | SAM geometry + GLOBAL_BLOCKS (vision_sam.rs:26); L1 sam_output |
| 24 | SAM patch-embed Conv2d k16s16 → 64×64×768 | SPEC-041 | MUST | present | L1 | vision_sam.rs | bd-1gv.5 | patch_embed conv k16s16 (vision_sam.rs + nn.rs:179); L1 |
| 25 | SAM abs pos_embed (1,64,64,768) + bicubic interp | SPEC-042 | MUST | present | L1 | vision_sam.rs | bd-1gv.4 | abs pos_embed bicubic interp (vision_sam.rs:1404); L1 |
| 26 | SAM window attention (window=14) | SPEC-043 | MUST | present | L2 | vision_sam.rs | bd-1gv.6 | window attention w=14 (vision_sam.rs:553,703); L1/L2 |
| 27 | SAM global attention (blocks 2/5/8/11) | SPEC-043 | MUST | present | L2 | vision_sam.rs | bd-1gv.6 | global attention blocks 2/5/8/11 (vision_sam.rs:342); L1 |
| 28 | SAM decomposed relative-position bias | SPEC-044 | MUST | present | L1 | vision_sam.rs | bd-1gv.6 | decomposed rel-pos bias (vision_sam.rs:867); L1 |
| 29 | SAM window partition/unpartition (pad to mult) | SPEC-045 | MUST | present | L1 | vision_sam.rs | bd-1gv.6 | window partition/unpartition (vision_sam.rs:703); L1 (SPEC-045 via range cite) |
| 30 | SAM neck + downsample (1×1/3×3/2×stride-2) → 16×16×1024 | SPEC-046 | MUST | present | L1 | vision_sam.rs | bd-1gv.7 | neck + net2/net3 downsample (vision_sam.rs:516-531); L1 |
| 31 | SAM MLPBlock GELU activation | SPEC-043 | MUST | present | L1 | nn.rs | bd-1gv.28 | MLPBlock GELU (vision_sam.rs:724, nn.rs:503 + hand-computed test); L1 |
| 32 | LayerNorm2d (vision) | SPEC-046 | MUST | present | L1 | nn.rs | bd-1gv.7 | LayerNorm2d (nn.rs:416, used in neck); L1 |

## 4. Modeling features — vision tower CLIP + bridge (SPEC-047..052, OQ-6)

| # | Feature | SPEC | Req | Status | Parity | Module | Bead | Notes |
|---|---------|------|-----|--------|--------|--------|------|-------|
| 33 | CLIP build params (24/1024/16, patch 14) | SPEC-047 | MUST | present | L2 | vision_clip.rs | bd-1gv.9 | CLIP 24/1024 patch14 (vision_clip.rs:68,88); L1 clip_output |
| 34 | CLIP embeddings take SAM features as patch_embeds (fused) | SPEC-048 | MUST | present | L1 | vision_clip.rs | bd-1gv.9 | CLIP embeddings take SAM patch_embeds (vision_clip.rs:296); L1 |
| 35 | CLIP get_abs_pos bicubic interp branch | UNRESOLVED-3 | SHOULD | partial | L1 | vision_clip.rs | bd-1gv.4 | abs_pos_for_len implemented (vision_clip.rs:834) but the bicubic branch is a no-op at 1024 — L1 exercises only pass-through |
| 36 | CLIP 24-layer transformer (SDPA, no causal) | SPEC-049 | MUST | present | L2 | vision_clip.rs | bd-1gv.9 | 24-layer no-causal transformer (vision_clip.rs:596); L1/L2 |
| 37 | quick_gelu `x·σ(1.702x)` | SPEC-049 | MUST | present | L1 | nn.rs | bd-1gv.9 | quick_gelu (nn.rs:517 + hand-computed test); L1 |
| 38 | CLIP call sig `vision_model(image, sam_features)` | SPEC-050 | MUST | present | L2 | vision_clip.rs | bd-1gv.9 | vision_model(img, sam_feat) signature (vision_clip.rs:253); L1 (SPEC-050 via range) |
| 39 | Hybrid concat(CLIP[:,1:], SAM_flat) → 2048 | SPEC-051 | MUST | present | L2 | vision_bridge.rs | bd-1gv.10 | hybrid concat CLIP[:,1:]+SAM→2048 (vision_bridge.rs:95); L1/L2 projector_output |
| 40 | Linear projector 2048→1280 | SPEC-052/016 | MUST | present | L1 | vision_bridge.rs | bd-1gv.10 | linear projector 2048→1280 (vision_bridge.rs:146); L1/L2 |
| 41 | Vision+ingest L0–L2 parity-ladder harness | §8.2 | MUST | present | L2 | tests/ | bd-1gv.12 | L0–L2 ladder harness armed green (parity_ladder.rs rungs) |

## 5. Modeling features — connector (SPEC-060..066)

| # | Feature | SPEC | Req | Status | Parity | Module | Bead | Notes |
|---|---------|------|-----|--------|--------|--------|------|-------|
| 42 | Learned image_newline/view_seperator params | SPEC-060 | MUST | present | L1 | connector.rs | bd-1gv.11 | learned image_newline/view_seperator (connector.rs:30); L2 inputs_embeds |
| 43 | Vision-branch trigger condition | SPEC-061 | MUST | partial | L2 | connector.rs | bd-1gv.11 | prefill-only fusion is structural (build_inputs_embeds one-shot, native_engine/mod.rs:1817) but SPEC-061 uncited and no dedicated trigger-guard test |
| 44 | CROP branch `[local,global,view_sep]` arrangement | SPEC-062 | MUST | partial | L2 | connector.rs | bd-1gv.11 | assemble_crop_block implemented + unit-tested (connector.rs:208) but the live L0/L2 oracle compare runs NO-CROP only (Gundam numeric parity not wired; parity_ladder.rs:1054) |
| 45 | NO-CROP branch (global + sep per image) | SPEC-063 | MUST | present | L2 | connector.rs | bd-1gv.11 | assemble_global_block no-crop branch (connector.rs:167); L2 armed green |
| 46 | masked_scatter into text embeds | SPEC-064 | MUST | present | L2 | connector.rs | bd-1gv.11 | masked_scatter (connector.rs:346 + order test); L2 |
| 47 | inputs_embeds source (embed_tokens) | SPEC-065 | MUST | present | L2 | decoder.rs, connector.rs | bd-1gv.14 | inputs_embeds from embed_tokens (connector.rs:339); L2 |
| 48 | Ordering invariant (token layout = feature concat) | SPEC-066 | MUST | present | L2 | connector.rs | bd-1gv.11.1 | ordering invariant checked in code (connector.rs:23,336,370); L2 |

## 6. Modeling features — decoder & MoE (SPEC-070..081)

| # | Feature | SPEC | Req | Status | Parity | Module | Bead | Notes |
|---|---------|------|-----|--------|--------|--------|------|-------|
| 49 | Decoder stack (embed/12 layers/final norm) | SPEC-070 | MUST | present | L2 | decoder.rs | bd-1gv.24 | 12-layer decode driver (decoder.rs:669); L4 + layer-count assert (parity_ladder.rs:1368) |
| 50 | RMSNorm (f32 variance, eps 1e-6) | SPEC-071 | MUST | present | L1 | nn.rs | bd-1gv.15 | RMSNorm f32 var eps 1e-6 (nn.rs:400 + hand-computed test); L1 |
| 51 | Decoder layer pre-norm residual | SPEC-072 | MUST | present | L2 | decoder.rs | bd-1gv.24 | pre-norm residual layer (decoder.rs:481); L2 |
| 52 | Attention class = SlidingWindowLlamaAttention (all 12) | SPEC-073 | MUST | present | L2 | rswa.rs | bd-1gv.17 | R-SWA attention all 12 layers (rswa.rs; decoder.rs:8); batched_attention_parity + L2/L4 |
| 53 | Dense-vs-MoE per layer (0 dense, 1..11 MoE) | SPEC-074 | MUST | present | L2 | decoder.rs, moe.rs | bd-1gv.24 | first_k_dense_replace=1 (moe.rs:637); L2/L4 |
| 54 | Dense MLP SwiGLU (layer 0, intermediate 6848) | SPEC-075 | MUST | present | L2 | moe.rs | bd-1gv.20 | dense SwiGLU 6848 (moe.rs:68); chunked_prefill_parity |
| 55 | MoE forward (top-6 route + 2 fused shared) | SPEC-076 | MUST | present | L2 | moe.rs | bd-1gv.19 | MoE top-6 + 2 fused shared experts (moe.rs); batched_moe_parity |
| 56 | MoEGate (softmax top-6 greedy, NO renorm) | SPEC-077 | MUST | present | L2 | moe.rs | bd-1gv.18 | MoEGate softmax top-6 NO renorm (moe.rs:184 + raw-softmax tests) |
| 57 | SiLU activation (LLM/expert) | SPEC-075 | MUST | present | L1 | nn.rs | bd-1gv.19.1 | SiLU (nn.rs:485); L1 |
| 58 | RoPE Llama variant (theta 10000, head_dim 128) | SPEC-078 | MUST | present | L1 | decoder.rs | bd-1gv.16 | RoPE theta 10000 hd128 NEOX (decoder.rs:89); L1 + chunked_prefill qkv_with_rope |
| 59 | Position IDs (arange / cumsum) | SPEC-079 | MUST | partial | L2 | decoder.rs | bd-1gv.24 | arange position ids implemented (decoder.rs:710), proven transitively by L4 token-exact; the cumsum/masked-position branch is absent (unneeded for unpadded v1, sub-case uncovered) |
| 60 | 4D causal mask handling (decode=None, prefill=causal) | SPEC-080 | MUST | present | L2 | decoder.rs, rswa.rs | bd-1gv.17 | causal prefill / no-mask decode (decoder.rs + rswa.rs:22); chunked_prefill_parity causal tiling |
| 61 | lm_head GEMV 1280→129280 (f32) + logits.float() | SPEC-081 | MUST | present | L3 | decoder.rs | bd-1gv.21 | lm_head GEMV 1280→129280 (decoder.rs); L3 rung + lmhead_shard_parity bit-identity |
| 62 | Token embedding lookup (bf16-preserving index_select) | SPEC-070 | MUST | present | L1 | decoder.rs | bd-1gv.14 | embed_tokens bf16 index_select (decoder.rs:3684); L5 CER 0 end-to-end |

## 7. Modeling features — R-SWA ring buffer (SPEC-090..096, the centerpiece)

| # | Feature | SPEC | Req | Status | Parity | Module | Bead | Notes |
|---|---------|------|-----|--------|--------|--------|------|-------|
| 63 | R-SWA heads/dims (10/10, head_dim 128, scale 1/√128) | SPEC-090 | MUST | present | L2 | rswa.rs | bd-1gv.17 | 10 heads hd128 scale 1/√128 no-bias (rswa.rs:42 + constants_match_spec test) |
| 64 | Regime 1: true prefill (full causal, record prefill_len) | SPEC-091 | MUST | present | L2 | rswa.rs | bd-1gv.17 | regime 1 true prefill (rswa.rs record_prefill + boundary test) |
| 65 | Regime 2: warmup decode (cat-append until prefill+W) | SPEC-091 | MUST | present | L2 | rswa.rs | bd-1gv.17 | regime 2 warmup append (rswa.rs + warmup_appends_without_eviction) |
| 66 | Regime 3: steady-state ring (in-place overwrite, no grow) | SPEC-091 | MUST | present | L2 | rswa.rs | bd-1gv.17 | regime 3 steady-state ring overwrite (rswa.rs + modulo-W tests) |
| 67 | Effective attention window (prefill_len + 128) | SPEC-094 | MUST | present | L2 | rswa.rs | bd-1gv.17 | effective window prefill+128, ref never evicted (rswa.rs:2197 large_batch_invariants) |
| 68 | PORT INVARIANT: RoPE uses true position, not ring slot | SPEC-095 | MUST | present | L2 | rswa.rs | bd-1gv.17.2 | PORT INVARIANT RoPE true position not ring slot (rswa.rs:28; spec_ring_rollback slot-decoupling) |
| 69 | Preallocated fixed ring + reference buffer (m_max 32896) | CENSUS (d) | MUST | present | L2 | rswa.rs | bd-1gv.17 | preallocated worst-case ring+reference (rswa.rs:184 + budget-arithmetic test) |
| 70 | Online (FlashAttention-style) softmax over ref block | §6.8 | SHOULD | present | L2 | rswa.rs | bd-1gv.17.1 | online FlashAttention-style softmax (rswa.rs:891 + online_matches_naive test) |
| 71 | KV-cap invariant (never exceeds L·(m+128)) | §8.5 | MUST | present | L2 | rswa.rs, conformance.rs | bd-1gv.24.1 | KV-cap invariant (rswa.rs:2226 + NEW kv_cap_ring_bound_holds_under_overfill 5×W overfill verdict in spec_ring_rollback.rs, wired to the e-process monitor bd-re8.15) |

## 8. Modeling features — sampler & postprocess (SPEC-100..119)

| # | Feature | SPEC | Req | Status | Parity | Module | Bead | Notes |
|---|---------|------|-----|--------|--------|--------|------|-------|
| 72 | Greedy default (temp 0 ⇒ argmax; temp>0 sample) | SPEC-100 | MUST | present | L4 | decode.rs | bd-1gv.22 | greedy default / temp>0 sample (sampler.rs:325); L4 token_exact 1.0 + batched_sampler_parity |
| 73 | EOS=1 / max_length 32768 / use_cache | SPEC-101 | MUST | present | L4 | decode.rs | bd-1gv.22 | EOS=1 max_length 32768 (sampler.rs); spec_decode_gate + L4 EOS stop |
| 74 | no_repeat_ngram options dispatch | SPEC-102 | MUST | present | L4 | decode.rs | bd-1gv.22 | no_repeat_ngram options dispatch (sampler.rs); spec_decode_gate non-frozen params test |
| 75 | SlidingWindowNoRepeatNgramProcessor (35, win 128/1024) | SPEC-103 | MUST | present | L4 | decode.rs | bd-1gv.22 | SlidingWindowNoRepeatNgram size 35 win 128/1024 (sampler.rs:33); pinned-constants test |
| 76 | Decode + strip EOS string | SPEC-110 | MUST | present | L5 | postprocess.rs | bd-1gv.23 | strip EOS (postprocess.rs:113 + test) |
| 77 | re_match ref/det regex extraction | SPEC-111 | MUST | present | L5 | postprocess.rs | bd-1gv.23 | ref/det regex extraction (postprocess.rs:130 + span tests) |
| 78 | Coordinate parse (extract_coordinates_and_label) | SPEC-112 | MUST | present | L5 | postprocess.rs | bd-1gv.23 | coordinate parse (postprocess.rs:265 + tests) |
| 79 | bbox /999 → pixel rescale | SPEC-113 | MUST | present | L5 | postprocess.rs | bd-1gv.23 | bbox /999 rescale w/ python int truncation parity (postprocess.rs:100 + test); metamorphic pad_coord |
| 80 | image-label crops → markdown `![](images/..)` | SPEC-114 | SHOULD | present | L5 | postprocess.rs | bd-1gv.23 | image-label crops → markdown refs (postprocess.rs:401 + tests) |
| 81 | other-label cleanup + `\coloneqq`/`\eqqcolon` | SPEC-115 | MUST | present | L5 | postprocess.rs | bd-1gv.23 | other-label cleanup + coloneqq normalize (postprocess.rs:373 + tests) |
| 82 | bbox overlay drawing (result_with_boxes.jpg) | SPEC-116 | MAY | excluded | n/a | — | — | (unchanged; §16) |
| 83 | geometry/line_type special case (geo.jpg) | SPEC-117 | MAY | excluded | n/a | — | — | (unchanged; §16) |
| 84 | Multi-page `<PAGE>` split/rejoin | SPEC-118 | MUST | present | L5 | postprocess.rs | bd-1gv.25 | <PAGE> split/rejoin multi-page (postprocess.rs:38,489 + tests) |
| 85 | test_compress metric (output/valid_img tokens) | SPEC-119 | MAY | excluded | n/a | — | — | (unchanged; §16) |

## 9. Op map — frankentorch facade (plan §4.3)

| # | Op | §4.3 status | Req | Status | Parity | Module | Bead | Notes |
|---|----|-------------|-----|--------|--------|--------|------|-------|
| 86 | int8 dynamic-quant linear (SMMLA/SDOT/VNNI) | EXISTS reuse | MUST | present | L3 | nn.rs | bd-1es.9 | linear_int8_dynamic → simd dispatch (nn.rs:151); robot selftest + batched_igemm_parity all tiers |
| 87 | f32 linear | EXISTS reuse | MUST | present | L1 | nn.rs | bd-1gv.10 | f32 linear (nn.rs:104); L1/L3 |
| 88 | Conv2d (im2col+GEMM) | EXISTS reuse | MUST | present | L1 | nn.rs | bd-1gv.5 | conv2d im2col+GEMM (nn.rs:185); L1 sam parity |
| 89 | SDPA attention (+masked/+gqa) | EXISTS reuse | MUST | present | L1 | nn.rs | bd-1gv.9 | SDPA (nn.rs:375); batched_attention_parity + L1 |
| 90 | Windowed self-attention (window 14) | BUILD | MUST | present | L2 | vision_sam.rs | bd-1gv.6 | windowed self-attention w14 (vision_sam.rs:39,809); L1 |
| 91 | quick_gelu | BUILD | MUST | present | L1 | nn.rs | bd-1gv.9 | quick_gelu (nn.rs:517); L1 |
| 92 | GELU / SiLU | EXISTS reuse | MUST | present | L1 | nn.rs | bd-1gv.28/19.1 | gelu/silu (nn.rs:503/488); unit tests + L1/L4 |
| 93 | RMSNorm | EXISTS reuse | MUST | present | L1 | nn.rs | bd-1gv.15 | rms_norm (nn.rs:400 + hand-computed test) |
| 94 | LayerNorm / LayerNorm2d | EXISTS + wrap | MUST | present | L1 | nn.rs | bd-1gv.7 | layer_norm/2d (nn.rs:424 + tests) |
| 95 | f32 bicubic pos-embed interpolate | GAP BUILD | MUST | present | L1 | vision_sam.rs | bd-1gv.4 | f32 bicubic pos-embed interp (vision_sam.rs:506); L1 + PIL goldens script |
| 96 | RoPE (theta 10000) | BUILD | MUST | present | L1 | decoder.rs | bd-1gv.16 | RoPE table (decoder.rs:89,152); spec_verify_forward_parity + L4 |
| 97 | R-SWA decode attention (ring + ref block) | BUILD centerpiece | MUST | present | L2 | rswa.rs | bd-1gv.17 | R-SWA decode attention (rswa.rs); batched_attention_parity + spec_ring_rollback |
| 98 | MoE router top-6 greedy + norm_topk_prob | BUILD | MUST | present | L2 | moe.rs | bd-1gv.18 | MoE router top-6 greedy (moe.rs:414); batched_moe_parity |
| 99 | Grouped expert SiLU-gated MLP | BUILD | MUST | present | L2 | moe.rs | bd-1gv.19 | grouped expert SwiGLU MLP (moe.rs:396); batched_moe_parity |
| 100 | Token embedding lookup (f32-preserving) | BUILD thin | MUST | present | L1 | decoder.rs | bd-1gv.14 | embed lookup f32-preserving (decoder.rs:116); L2 fusion rung |
| 101 | masked-scatter vision fusion | BUILD | MUST | present | L2 | connector.rs | bd-1gv.11 | masked-scatter fusion (connector.rs:346,403); L2 inputs_embeds |
| 102 | Image decode/resize/normalize/pad/tile | GAP BUILD | MUST | present | L0 | preprocess/ | bd-1gv.2/3 | preprocess pipeline (src/preprocess); L0 rung (default resample within DISC-001 envelope; bit-exact under FOCR_RESAMPLE=pil-bicubic); L5 CER 0.0 |
| 103 | BPE tokenizer (tokenizer.json) | GAP BUILD | MUST | present | L0 | tokenizer/ | bd-1gv.1 | BPE tokenizer (src/tokenizer); tokenizer_conformance |
| 104 | Sampler + no_repeat_ngram(35) | BUILD | MUST | present | L4 | decode.rs | bd-1gv.22 | sampler + ngram-35 (sampler.rs:33); batched_sampler_parity + L4 EXACT |
| 105 | safetensors BF16 load | EXISTS reuse | MUST | present | SURF | weights.rs | bd-1es.4 | safetensors BF16 load (weights.rs:162); convert tests + e2e |

## 10. Op map — perf kernels (plan §6.6, Phase 3+, behind kill-switches)

| # | Op | §6.6 tier | Req | Status | Parity | Module | Bead | Notes |
|---|----|-----------|-----|--------|--------|--------|------|-------|
| 106 | Runtime ISA capability + effective dense-route dispatch | all | SHOULD | present | L3 | nn.rs | bd-2mo.1 | OnceLock IsaTier capability; effective route + branch-derived selftest trace; all-tier FOCR_FORCE_ARCH subprocess sweep |
| 107 | aarch64 SMMLA/i8mm prefill GEMM (the wedge) | A1 | SHOULD | present | L3 | nn.rs (island) | bd-2mo.4 | aarch64 SMMLA 8×8 prefill GEMM (arm.rs:199); batched_igemm_parity smmla + selftest |
| 108 | aarch64 SDOT decode GEMV | A2 | SHOULD | present | L3 | nn.rs (island) | bd-2mo.5 | forced SDOT route remains exact; measured-faster LLVM autovec is default for ordinary Apple dense GEMV; packed routes separate |
| 109 | x86 AVX-512-VNNI GEMM/GEMV (U8S8 +128) | X1 | SHOULD | present | L3 | nn.rs (island) | bd-2mo.6 | x86 AVX-512-VNNI (x86.rs:122); vs-scalar gate, arch-gated on x86 hosts |
| 110 | x86 AVX-VNNI (256-bit) | X2 | SHOULD | present | L3 | nn.rs (island) | bd-2mo.7 | x86 AVX-VNNI 256-bit (x86.rs:129); same gate |
| 111 | x86 AMX-int8 prefill (_tile_dpbssd, feature) | X3 | MAY | missing | L3 | nn.rs (island) | bd-2mo.8 | no AMX kernel (dispatch.rs:19-21 states x86.rs implements no AMX; MAY tier) |
| 112 | x86 AVX2 fallback (maddubs→madd, i16-sat proof) | X4 | SHOULD | present | L3 | nn.rs (island) | bd-2mo.9 | AVX2 fallback, non-saturating vpmaddwd (x86.rs:268,285; bd-2mo.9.1); vs-scalar gates |
| 113 | Scalar int8 GEMM/GEMV floor (cross-compile) | S | MUST | present | L3 | nn.rs | bd-2mo.10 | scalar int8 floor (src/simd/scalar.rs) — IS the bit-exact oracle in every gate |
| 114 | i32-overflow proof at worst-case K=6848 | §5.4 | MUST | present | L3 | tests/ | bd-2mo.11 | i32 overflow proof K=6848 (tests/int32_overflow_proof.rs:198; rejects the 1536 bound) |
| 115 | Offline arch-specific weight pre-packing (--arch) | §5.4 | SHOULD | present | L3 | weights.rs | bd-2mo.3 | aarch64-smmla emits REAL offline panels (pack.rs single source of truth; convert emits, loader keeps-on-SMMLA-tier / un-permutes otherwise); packed-B kernel bit-identical (arm.rs i8mm gate) + decoder layout-parity + lossless/determinism receipts; VNNI/AMX stay tag-only (no packed-consuming x86 kernel exists; row-major IS AVX2's zero-shuffle layout) |
| 116 | MoE prefill token-grouping (counting-sort → GEMMs) | §6.7 | SHOULD | present | L2 | moe.rs | bd-2mo.12 | MoE prefill gather-by-expert grouped GEMMs (moe.rs:396); batched_moe_parity bit-exact |
| 117 | int8 attention (Q·Kᵀ, scores·V) + CVaR gate | §6.8 | MAY | partial | L3 | rswa.rs | bd-2mo.15 | int8 QK/PV attention shipped + parity-gated behind FOCR_INT8_KV (rswa.rs:1060,1836 ≤1e-6 + overflow proof) — env name differs from the row note (FOCR_INT8_ATTN) and the CVaR gate is not runtime-wired |
| 118 | int4 group-quant GEMM (unpack→int8 MAC) | §6.3 | SHOULD | present | L3 | nn.rs (island) | bd-2ela | int4 group-quant GEMM (int4.rs:170); int4_packed_parity unpack-equivalence |
| 119 | Vectorized poly-exp (softmax/SiLU/quick_gelu) | §6.11 | MAY | missing | L3 | nn.rs | bd-2mo.20 | no FOCR_VEC_EXP / vectorized poly-exp anywhere (MAY tier, not built) |

## 11. `.focrq` weight transformation & quant recipe (plan §5)

| # | Feature | Plan | Req | Status | Parity | Module | Bead | Notes |
|---|---------|------|-----|--------|--------|--------|------|-------|
| 120 | `.focrq` format spec + version/provenance | §5.2 | MUST | present | SURF | docs/ | bd-1es.1 | FOCRQ magic + FORMAT_VERSION=1 + license/sha256 slots (focrq.rs:60,65); check_focrq_format.py lints the doc |
| 121 | `.focrq` writer + reader (byte-range, manifest census) | §5.2 | MUST | present | SURF | weights.rs | bd-1es.2/3 | writer + reader byte-range manifest (focrq.rs + weights.rs:463); round-trip tests bf16/f32/qint8/qint4 |
| 122 | Tensor remap (HF dotted → internal) | §5.3 | MUST | present | SURF | weights.rs | bd-1es.4 | HF-dotted key census/classification (convert.rs:655 tests; canonical-key policy) |
| 123 | Per-output-channel int8 quantizer (zp 0) | §5.1 | MUST | present | L3 | weights.rs | bd-1es.5 | per-out-channel symmetric int8 zp0 clamp±127 (int8.rs:145,47 + tests) |
| 124 | Per-row dynamic int8 activation quant (S8S8/U8S8) | §6.3 | MUST | present | L3 | nn.rs | bd-1es.8 | per-row dynamic activation quant S8S8/U8S8 (nn.rs:151 + igemm parity + selftest u8s8) |
| 125 | int8 attention q/k/v/o (FOCR_INT8_ATTN kill-switch) | §5/§6 | MAY | partial | L3 | weights.rs | bd-1es.10 / bd-2mo.30.5 | Static recipe gate and experimental runtime exist, but the default artifact now keeps q/k/v/o BF16 and the all-int8 cache requires all three explicit switches; measured CER acceptance is still outstanding. |
| 126 | int8 lm_head (FOCR_INT8_LMHEAD kill-switch) | §5/§6 | MAY | partial | L3 | weights.rs | bd-1es.11 / bd-2mo.30.5 | Kernel parity exists, but quality certification is not complete. The default artifact keeps `lm_head` BF16 and the legacy all-int8 cache fails closed unless both independent gates are armed. |
| 127 | int4 per-group quantizer (16–32, tier discipline) | §6.3 | SHOULD | present | L3 | weights.rs | bd-lsu3 | int4 group 16/32 tier discipline (int4.rs:49); int4_packed_parity + focrq qint4 roundtrip |
| 128 | High-precision set kept BF16 (vision/proj/embed/router/norms) | §5.1 | MUST | present | SURF | weights.rs | bd-1es.6 | KeepBf16 high-precision set (recipe.rs:15-20,58 + policy tests) |

---

## 12. CLI surface (plan §7.2) — the SurfaceMatrix

> Statuses brought current 2026-07-06 (bd-re8.13): the Phase −1 all-`missing`
> seed is superseded by shipped, contract-tested surfaces (v0.1.0–v0.3.0 +
> the model-zoo waves). Notes cite the proving test/release. The enumeration
> test (`tests/surface_matrix.rs`) asserts every live subcommand and every
> frozen-schema robot event/exit-code has a row here.

| Subcommand / surface | §7 | Req | Status | Parity | Bead | Notes |
|----------------------|-----|-----|--------|--------|------|-------|
| `focr ocr <image>` → markdown / `--json` | §7.2 | MUST | present | SURF | bd-1gv.27 | v0.1.0; goldens (`cli_robot_golden`) + armed e2e (L5 CER 0.0) |
| `focr ocr -o/--output FILE` (.md / .json-with-boxes) | §7.2 | MUST | present | SURF | bd-sreb | v0.3.0; goldens |
| `focr ocr --extract-figures [--figures-dir]` | §7.2 | SHOULD | present | SURF | bd-sreb | v0.3.0; figure PNG/JPG chosen by content |
| `focr ocr --task format\|music\|describe\|vqa\|chart-data` (zoo lanes) | §7.2 | MUST | partial | SURF | bd-av64 | GOT/SmolVLM2/OneChart/TrOMR lanes shipped w/ armed certs; `music` partial on real-input hardening (bd-av64: duration crash, wide-staff abort) |
| `focr ocr --robot` / `focr robot run` (NDJSON stream) | §7.2 | MUST | present | SURF | bd-223.3 | contract tests green (bd-zc1o); internals polish tracked bd-wp8.3 |
| `focr ocr-batch` (load-once multi-image throughput) | §7.2 | SHOULD | partial | SURF | bd-1azu | spine + batched-parity tests green; NO CLI golden yet (an untested surface never rounds up) |
| `focr convert <st> -o <.focrq> [--arch][--quant]` | §7.2 | MUST | present | SURF | bd-1es.6 | byte-parity vs published artifact re-proven 2026-07-06 (sha d8c5fcf2…) |
| `focr pull [model] [--quant]` (manifest + verify) | §7.2 | MUST | present | SURF | bd-3u6x | verified vs HF; native Windows proven (bd-15ow); zoo artifact publication user-gated (bd-av64.8) |
| `focr models` (zoo discovery: id, tasks, status) | §7.2 | MUST | present | SURF | bd-3jo6.1.13 | CLI shipped + goldens; A13 docs/runbook half still open |
| `focr robot schema` (self-describing contract) | §7.2 | MUST | present | SURF | bd-wp8.2 | versioned; frozen fixture |
| `focr robot health` (model/arch/threads diagnostics) | §7.2 | MUST | present | SURF | bd-223.3 | incl `threads` budget field (bd-223.2) |
| `focr robot backends` (effective dense route + hardware tiers + USL pool sizing) | §7.2 | MUST | present | SURF | bd-2mo.2 | separates effective ordinary-dense route from hardware IsaTier; goldens |
| `focr robot selftest` (runtime int8 kernel parity on host silicon) | §7.2 | MUST | present | SURF | bd-223.13 | 24/24 on native Win10; AVX2 ceiling proven on 5995WX |
| `focr robot triage` (one-round-trip agent mega-command) | §7.2 | SHOULD | present | SURF | bd-wp8.7 | quick_ref + health + state-aware recommendations + exit codes in ONE JSON; pinned by agent_ergonomics_regression |
| `focr runs [--id\|--limit\|--format]` | §7.2 | SHOULD | present | SURF | bd-wp8.11 | frozen contract `runs_schema.json` + populated-store matrix through the real binary (json/ndjson/--id/--limit/plain); empty history = exit 0 |
| `focr sync export-jsonl\|import-jsonl` | §7.2 | SHOULD | present | SURF | bd-wp8.11 | locked atomic temp+fsync+rename, byte-identical re-export, one-way contract documented; migration + exit-7 refusal tested (bd-223.4) |
| `focr doctor` (idempotent self-check/repair) | §7.2 | SHOULD | present | SURF | bd-wp8.4 | LIVE: detect-then-fix, single-chokepoint mutation (backup-first, hash-logged), undo byte-identical, dry-run zero-blast, capabilities/robot-docs/robot-triage; 8/8 fixture roundtrips (`tests/doctor_fixtures.rs`) |
| Exit codes 0..7 (stable, documented) | §7.4 | MUST | present | SURF | bd-223.5 | error.rs mapping + schema `exit_codes` + contract tests |
| Env overrides (FOCR_MODEL_DIR/THREADS/STAGE_BUDGET/QUANT/NUMA…) | §7.5 | MUST | present | SURF | bd-223.7 | OnceLock; FOCR_THREADS physical-core budget (bd-223.2) |
| Model resolution (no network at runtime) + header sniff | §7.5 | MUST | present | SURF | bd-223.7 | default auto-resolves pulled int8 (bd-3u6x); dotfile-safe shard globs |
| `--version` carries Baidu MIT attribution | §11 | MUST | present | SURF | bd-223.14 | license compliance |
| `pdf` input (native scanned fast path) | §7.7 | SHOULD | partial | SURF | bd-0a7.4 | **moved from `excluded`**: scanned-PDF native path shipped v0.2.0 (lopdf, decompress-bomb bounded bd-2zpu); vector-page rasterization deferred |
| 5-target single-binary cross-platform build | §7.6 | MUST | present | SURF | bd-wp8.5 | v0.1.0–v0.3.0 released: darwin×2, linux×2, win-msvc; local cross-build runbook |
| aarch64-windows target | §7.6 | MAY | missing | SURF | bd-3u97 | open; win-msvc x86_64 ships today |

## 13. Robot / NDJSON event contract (plan §7.3)

| Event / contract | §7.3 | Req | Status | Parity | Bead | Notes |
|------------------|------|-----|--------|--------|------|-------|
| `run_start` event | §7.3 | MUST | present | SURF | bd-223.3 | carries schema_version; contract test |
| `stage` event (name, seq, elapsed, budget) | §7.3 | MUST | present | SURF | bd-223.3 | contract test |
| `page` event (per-page text/bbox, streaming) | §7.3 | MUST | present | SURF | bd-wp8.3 | incl per-page skip signal (bd-fck1, v0.3.0); bounded-stream scaffold bd-223.2 |
| `staff` event (TrOMR staff result) | §7.3 | MUST | present | SURF | bd-av64.2 | additive v1 music event; carries 1-based staff index, total, bbox, status, and optional skip reason |
| `music_warning` event (TrOMR sanity annotation) | §7.3 | MUST | present | SURF | bd-av64.5 | additive v1 music event; reports staff-level key/time/bar-sum warnings without failing OCR |
| `run_complete` event | §7.3 | MUST | present | SURF | bd-223.3 | contract test |
| `run_error` event (carries exit code) | §7.3 | MUST | present | SURF | bd-223.5 | contract test |
| `ROBOT_SCHEMA_VERSION` on every line | §7.3 | MUST | present | SURF | bd-223.3 | stable versioned |
| Frozen JSON-schema fixture + contract test | §7.3 | MUST | present | SURF | bd-zc1o | `tests/fixtures/robot_schema_v1.json` + scrubbed goldens |
| Deterministic under fixed sampling (byte-identical) | §7.3 | MUST | present | SURF | bd-3kge | shared determinism gate + metamorphic FOCR_THREADS axis |

## 14. Parity gates & gauntlet machinery (plan §8.2, §8.5)

| Gate / machinery | §8 | Req | Status | Parity | Bead | Notes |
|------------------|-----|-----|--------|--------|------|-------|
| Oracle nondeterminism-floor characterization | §8.2 | MUST | present | n/a | bd-re8.2 | floors measured + recorded in fixture `_meta`; sets all tolerances |
| L0 preprocessing parity gate (exact) | §8.2 | MUST | present | L0 | bd-re8.4 | armed green (scorecard 2026-07-06: max_abs 0.0078 ≤ envelope) |
| L1 per-op + L2 per-layer parity gates | §8.2 | MUST | present | L1/L2 | bd-re8.5 | armed green (cosine 1−4e-13; L2 max-abs 8.8e-5 ledgered) |
| L3 logits + L4 token parity gates | §8.2 | MUST | present | L3/L4 | bd-re8.6 | armed green (L4 token_exact 1.0 on reproducible prefix) |
| L5 end-to-end OCR parity (CER/TEDS/Formula-CDM) | §8.2 | MUST | present | L5 | bd-re8.7 | armed green (CER 0.0 both fixture pages) |
| PyO3/subprocess oracle bridge (ULP tolerance, deterministic) | §8.5 | MUST | present | n/a | bd-re8.3 | test-only; `check_release_linkage.py` guards no-FFI ship |
| Differential test suite (per-op + e2e) | §8.3 | MUST | present | L5 | bd-re8.9 | in `parity_ladder.rs` (oracle differential guard) |
| Metamorphic suite (resize/rotation/whitespace; OQ-13 cross-page) | §8.3 | SHOULD | present | L5 | bd-re8.10 | `tests/metamorphic.rs`; armed relations byte-identical; MR-2-live/MR-5 honestly gated |
| Golden-artifact suite (insta/fuzzy/scrubbed/canonicalized) | §8.3 | MUST | present | SURF | bd-re8.11 | `cli_robot_golden` + UPDATE_GOLDENS review loop |
| ConformanceTest trait + coverage matrix (≥0.95 MUST) | §8.6 | MUST | present | n/a | bd-re8.12 | registry + SPEC-side matrix ≥0.95 green; XFAIL discipline over emission sites |
| Model-gated e2e (skip-green w/o weights, prove native ran) | §8.3 | MUST | present | L5 | bd-29wv | `/nonexistent` fallback proof pattern suite-wide |
| `many_pages_without_deadlock` watchdog | §6.5 | MUST | present | n/a | bd-2ub2 | green + injected-hang detector + live over-budget trip demo; cancel/panic variants bd-1ryu |
| asupersync capacity certificate (p95/p99, bounded stream, pool stability) | §6.9 | MUST | present | n/a | bd-re8.18 | armed heavy 2026-07-06: p95 7.41 s/page, 48/48 Ok, width stable; `focr-capacity-certificate/v1` |
| L0–L5 ladder scorecard runner (per-commit parity receipt) | §8.4 | MUST | present | n/a | bd-re8.19 | `scripts/ladder_scorecard.sh`; armed all-green fixture committed |
| Input-fault robustness suite (typed errors, frozen exit codes) | §8.5 | MUST | present | SURF | bd-15kd | `tests/fault_suite.rs`: corrupt/truncated/zero-byte→4, corrupt artifact→7 (caught+fixed a real exit-1 misclassification), task-mismatch→2, missing→3; armed 4/5 legs gated; cancellation owned by bd-1ryu |
| Three-pillar release certification (perf/conformance/surface) | §8.5 | MUST | partial | n/a | bd-re8.13 | math+methodology+self-test shipped (`gauntlet_cert.py`); matrix brought current; converged RUN is bd-wp8.8 |
| Conformal lower-bound release ratchet | §8.5 | SHOULD | present | n/a | bd-re8.14 | Rust impl (`src/conformance.rs`) + Python reference, cross-checked; RATCHET.md ledger |
| E-processes for load-bearing invariants (Ville) | §8.5 | SHOULD | partial | n/a | bd-re8.15 | math + self-test in `gauntlet_cert.py`; wiring to live invariant streams open |
| Head-to-head gauntlet bench vs CPU reference (per-stage, fair) | §9.3 | MUST | partial | n/a | bd-re8.17, bd-2mo.26 | zoo ratios measured (A11, fairness-pinned); unlimited-ocr Phase −1 baseline half open |
| Release-readiness scorecard (all-green ship gate) | §8.4 | MUST | partial | n/a | bd-wp8.10 | components exist (ladder receipt, capacity cert, ratchet); the tying gate open |

## 15. Alien-artifact families (plan §9.7) — upside levers behind guarantees

| Family | §9.7 | Req | Status | Parity | Bead | Notes |
|--------|------|-----|--------|--------|------|-------|
| AF-1 rate-distortion water-filling bit allocation | §9.7 | SHOULD | partial | L5 | bd-ksps, bd-1xfa.1 | `scripts/af1_bit_allocator.py` + `src/quant/bit_allocator.rs` shipped; full table integration open |
| AF-2 tail-risk CVaR + EVT worst-case CER gate | §9.7 | SHOULD | partial | L5 | bd-3upw, bd-1xfa.2 | `scripts/af2_tail_risk.py` math shipped; not yet wired as a release gate |
| AF-3 conformal/SPRT early-exit + speculative decode | §9.7 | MAY | missing | L4 | bd-1xfa.3 | SPIKE open (spec-decode ring exists; safe-exit bound not) |
| AF-4 submodular high-precision tensor selection | §9.7 | MAY | missing | L5 | bd-1xfa.4 | SPIKE open |
| AF-5 USL many-core pool sizing | §9.7 | SHOULD | partial | n/a | bd-2mo.21, bd-1xfa.5 | `scripts/af5_usl_fit.py` + `src/adaptive/usl.rs` shipped; NUMA pool caps open |

---

## 16. Coverage-debt register (the `excluded` rows, reasoned)

`excluded` rows are enumerated coverage debt, NOT silent omissions (plan §8.5: "excluded still counts as coverage debt"). Each carries a reason and a re-open condition:

| Feature | SPEC | Reason excluded from v1 | Re-open condition |
|---------|------|-------------------------|-------------------|
| `valid_img_tokens` accounting (#13) | SPEC-032 | Compression-ratio metric, not part of the forward; no OCR-output impact. | If `test_compress` is exposed as a CLI diagnostic. |
| bbox overlay drawing (#82) | SPEC-116 | Visualization-only (`result_with_boxes.jpg`); not core OCR output; needs image-draw deps. | If a `--draw-boxes` surface is requested. |
| geometry/`line_type` special case (#83) | SPEC-117 | Rare visualization path (`geo.jpg`); `eval()`-based; not core text/table/formula output. | If geometry parsing is a v2 target. |
| `test_compress` metric (#85) | SPEC-119 | Diagnostic compression ratio; not OCR output. | If exposed as a diagnostic subcommand. |
| `pdf` CLI input | §7.7 | Pure-Rust MuPDF-parity raster is unscoped; any pixel mismatch blows the L0 gate (§7.7). | pdfium feature flag (re-adds C dep) OR pure-Rust renderer + rasterization-parity gate vs pymupdf@300DPI. |

---

*End of scoreboard. LIVING DOCUMENT — seeded at Phase −1 (all rows `missing`), updated every phase as beads land and parity gates go green. Read by the three-pillar release certification (bd-re8.13) and the release-readiness scorecard (bd-wp8.10). A row flips `missing → partial → present` only as its delivering bead lands and its `Parity` gate (L0–L5/SURF) turns green; `partial` never rounds up; a feature accidentally omitted here is silent coverage debt the gauntlet cannot catch — cross-check against §4.3 / §7.2 / §7.3 / §8.2 on every update.*
