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
