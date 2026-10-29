# Open-Question Resolution Index (claim → source) — Truth Pack, Phase −1

> Resolves bead `PM1-claim-index`. Every [OPEN]/OQ from plan §13 is answered from
> the pinned source (HF `3a7f4db…`, hashes in `SOURCE_HASHES.md`), each answer
> line-backed in the listed file. **No kernel ships against an unresolved [OPEN]**
> (AGENTS.md) — this index is the proof that gate is satisfiable.

| OQ | Status | Answer file | One-line resolution (full quote + `file:line` in the answer file) |
|----|--------|-------------|-------------------------------------------------------------------|
| **OQ-1** | RESOLVED | `oq/rswa-attention.md` | R-SWA reference set `m` = the **entire prefill** (BOS + visual + prompt), boundary = `prefill_len`; ring overwrites only slots ≥ `prefill_len`, so the prefix is **permanent/never evicted** (`modeling_deepseekv2.py:1322,1363-1364`). |
| **OQ-2** | RESOLVED | `oq/rswa-attention.md` | **All 12 layers** (incl. layer 0) are uniformly `SlidingWindowLlamaAttention`; no full-attention layer (selection keyed only on `use_mla=false`) (`:1398-1405,1618-1622`). |
| **OQ-3** | RESOLVED | `oq/rswa-attention.md` | Warm-up appends (no eviction) for the first 128 steps, then in-place ring overwrite `slot = prefill_len + ring_pos`, `ring_pos=(ring_pos+1)%128`; decode applies **no causal mask** (window enforced physically) (`:1334-1342,1361-1367`). |
| **OQ-4** | RESOLVED | `oq/rswa-attention.md` | Q/K/V `head_dim = 1280/10 = 128`; plain MHA 10/10 heads; `qk_nope/rope_head_dim=0` are MLA-only, unused (`config.json:97,106`; `modeling_deepseekv2.py:37-40,1286-1288`). |
| **OQ-5** | RESOLVED | `oq/rope-and-config.md` | `rope_theta=10000.0`, `rope_scaling=None` → **vanilla un-scaled RoPE**, no YARN/NTK/mscale, across the full 32768 context (`configuration_deepseek_v2.py:155-156,199-200`). |
| **OQ-6** | RESOLVED | `oq/vision.md` | Exact SAM⊕CLIP concat order forming the 2048-dim projector input (quoted from the DeepEncoder forward in `deepencoder.py`). |
| **OQ-7** | RESOLVED | `oq/preprocess-infer.md` | Gundam `dynamic_preprocess`/`find_closest_aspect_ratio` `min_num`/`max_num` + per-aspect-ratio tile math (`modeling_unlimitedocr.py`). |
| **OQ-8** | RESOLVED | `oq/preprocess-infer.md` | Prompt-mode taxonomy + whether `<\|ref\|>`/`<\|det\|>` bboxes emit per mode (`conversation.py`/processor). |
| **OQ-9** | RESOLVED (analytic) | `oq/secondary.md` | ~500M activated params recomputed from config dims (header-exact recompute deferred — needs the safetensors header; non-blocking). |
| **OQ-10** | DEFERRED (non-blocking) | `oq/secondary.md` | GGUF router/expert quant vs NVFP4 — needs the external GGUF repo; plan marks non-blocking. |
| **OQ-11** | DEFERRED (non-blocking) | `oq/secondary.md` | int8 patch-embed conv viability — **UNMEASURED**: no in-repo experiment isolates the patch-embed conv (`NEGATIVE_EVIDENCE.md` has decoder-only quant levers; vision is f32-validated at cosine 0.9996). Would need an A/B quantizing ONLY the patch-embed conv vs f32, gated on CER, post-decoder-parity (Phase 4+); vision stays high-precision by doctrine #2. |
| **OQ-12** | DEFERRED (non-blocking) | `oq/secondary.md` | Benchmark cells — secondary/docs-only; confirm from the PDF tables later. |
| **OQ-13** | RESOLVED | `oq/rswa-attention.md` | **Multi-page is cross-page DEPENDENT**: `infer_multi` concatenates all pages into ONE prefill/`generate()` call, so page N attends to 1..N−1; `config.sliding_window` nulled to stop `DynamicCache` truncation; cross-page prefix KV permanent (`modeling_unlimitedocr.py:1198-1212,1233-1237,1240-1256`). |
| **OQ-14** | PARTIAL | `oq/secondary.md` | Our quantizable-linear set enumerated (~2229 from the index); the EXACT NVFP4 module set needs the external `sahilchachra/Unlimited-OCR-NVFP4` scale keys (dump `*.weight_scale`/`quantization_config`). |
| **OQ-15** | RESOLVED | `oq/vision.md` | SAM windowed-attention window size + windowed-block pos-embed scheme + `global_attn_indexes [2,5,8,11]` (quoted from `deepencoder.py`). |
| **OQ-16** | RESOLVED | `oq/tokenizer.md` | Tokenizer = `LlamaTokenizerFast`, byte-level **BPE** (`tokenizer.json` hashed: vocab 128000 + 830 added tokens, pre-tokenizer `Sequence`); the exact merges/specials a pure-Rust tokenizer must replicate are specified. |
| **OQ-17** | RESOLVED | `oq/preprocess-infer.md` | `infer()` is **CUDA-oriented** (`.cuda()` + autocast) → correctness oracle runs GPU, CPU-perf baseline is separate/proven (decides G2; see `gen_reference_fixtures.py`). |
| **OQ-18** | RESOLVED | `oq/preprocess-infer.md` | Full per-mode token census: base-1024 placeholder `(16+1)·16+1 = 273` slots; Gundam multi-tile totals; `no_repeat_ngram_size=35` with `ngram_window` **128 single / 1024 multi** (`modeling_unlimitedocr.py` `infer`/`infer_multi`). |

**Disposition:** 14 RESOLVED, 1 PARTIAL (OQ-14, external NVFP4 set), 3 DEFERRED-non-blocking (OQ-10/11/12). **Zero blocking OQs remain** — the Phase-1 kernels are unblocked. The PARTIAL/DEFERRED items carry explicit "needs X" retry conditions and gate only later, optional work (NVFP4 cross-check, int8-vision experiment, benchmark-cell confirmation), not the f32 reference-parity forward.

---

## OQ-V2 — Unlimited-OCR-2 (bead U2-A1)

> The v2 spec is [`UNLIMITED_OCR_2_PLAN.md` §5](../planning/UNLIMITED_OCR_2_PLAN.md#5-reference-semantics-specification-v2)
> and its OQ register is [§14](../planning/UNLIMITED_OCR_2_PLAN.md#14-open-questions-register-oq-v2);
> this section indexes that register and does not restate it. Every answer cites the v2
> pin (HF `5632329…`, hashes in `SOURCE_HASHES.md` "Unlimited-OCR-2"). The windowed-mode
> semantics behind §5.10 `[W]` are in `oq/uocr2-windowed-sglang.md`. The v1 rows above
> describe the v1 checkpoint only: v2 replaces R-SWA with SR-SWA, 1-D RoPE with page2d
> RoPE and adds the `<PAGE>` tokens, so the v1 answers to OQ-1–5, 8, 9, 13, 14, 16 and 18
> do not carry over and their v2 equivalents live here.

| OQ | Status | Answered in / owning bead | Question |
|----|--------|---------------------------|----------|
| **OQ-V2-1** | RESOLVED | plan §5.6.4 | SR-SWA candidate semantics |
| **OQ-V2-2** | RESOLVED | plan §5.6 | Router constants at inference |
| **OQ-V2-3** | OPEN | U2-A6 floor (`bd-uocr2-epic-cgux.7`) + U2-C1 (`.17`) | Parity tolerance: f32 native vs the f32 oracle; bf16 ledger tolerance |
| **OQ-V2-4** | OPEN | U2-A6 step 1 (`.7`) | Does HF v2 run in f32 on CPU (dense SDPA path); is a ported dense fallback needed? |
| **OQ-V2-5** | RESOLVED (no) | plan §5.6.7, D8 | Are the `indexer.*` projections used at inference? |
| **OQ-V2-6** | RESOLVED | plan §5.5 | 1-D RoPE fallback: franken builds `<PAGE>` prompts, so none |
| **OQ-V2-7** | RESOLVED | plan §5.9, DISC-U2-2 | n-gram vs guard order hazard: the forced token wins |
| **OQ-V2-8** | PARTIAL (conformance proven; registry wiring pending) | U2-A4 (`.5`): `tokenizer::tests::uocr2_token_id_conformance_gate` (227 cases, 0 mismatches against `tokenizers==0.21.4`), `uocr2_added_tokens_stay_whole_under_fuzz`, `uocr2_real_tokenizer_keeps_added_tokens_whole_under_fuzz`; `docs/truth-pack/oq/tokenizer.md` § Unlimited-OCR-2. U2-A5 (`.6`) closes it | Non-special added tokens: split whole and kept by skip-special decode. The required-token check (`UNLIMITED_OCR_V2_TOKENS`) is wired by U2-A5 |
| **OQ-V2-9** | OPEN | U2-D2b (`.25`) | Windowed accuracy vs unwindowed |
| **OQ-V2-10** | OPEN | U2-G2 (`.44`) | v2 single-page base vs gundam |
| **OQ-V2-11** | RESOLVED (mechanism; fixtures at U2-C2) | U2-B4 (`.13`): `src/native_engine/srswa.rs` tests `retired_runs_equal_unretired_runs_bit_for_bit`, `the_detector_passes_guard_forced_steps_and_fires_otherwise`, `pinned_rows_survive_to_the_end_of_a_window` | Block-granular retirement is output-exact under the guard |
| **OQ-V2-12** | OPEN | U2-C5a (`.21`) | Licenses of evaluation sets beyond MarathonDocBench (MPDocBench, OmniDocBench) |
| **OQ-V2-13** | OPEN | U2-C4 (`.20`) | Does the conservative int8 recipe hold v2 CER within budget? |
| **OQ-V2-14** | OPEN | U2-C5b (`.22`), else U2-G2 (`.44`) on C5b's subset | Document-mode crop mode (Baidu's long-document numbers are plausibly gundam) |
| **OQ-V2-15** | OPEN | U2-C5a (`.21`) | Non-Baidu ground-truth documents for the head-to-head |
| **OQ-V2-16** | OPEN | U2-C5b (`.22`) | Entmax domain for long documents (TopM vs AllCandidates) |

**Disposition:** 7 RESOLVED, 1 PARTIAL, 8 OPEN, each OPEN item owned by the bead named.
Kernel gates: OQ-V2-3/4 block parity sign-off (U2-A6, U2-C1). OQ-V2-11 no longer blocks
block retirement; `FOCR_SRSWA_RETIRE` stays default-on only if U2-C2's fixtures also agree. The rest gate evaluation or defaults (C4, C5a/b, D2b, G2),
not the f32 reference forward. Bead ids are children of epic `bd-uocr2-epic-cgux`.
