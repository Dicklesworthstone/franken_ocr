# GOT-OCR2 / Model-Zoo — next steps (design plans)

Implementation-ready plans from the parallel scout workflow `got-zoo-next-scout` (2026-07-01),
one section per workstream. B10 (`--format`) shipped this turn; the rest are captured here so
the work starts in plan-space (per the beads doctrine) with the load-bearing details preserved.

---

## 1. B10 — `--format` CLI (bd-3jo6.2.10) — DONE (2026-07-01)

GOT's specialized outputs (math LaTeX, Markdown tables, chart data, SMILES molecules, TikZ
geometry, `**kern` sheet music) are **all the single `"OCR with format: "` instruction** — the
model auto-selects the formalism (spec §5/§8; the only prompt delta is plain=287 ids vs
format=289 ids, proved by `got::tests::format_prompt_swaps_the_instruction`). So B10 is a
boolean `--format`, not a `--task` enum. Shipped: `--format` on `OcrRequestArgs` → `OcrRequest`
→ a process-global `native_engine::force_got_format` (mirrors `force_int8_decode`, keeps the
frozen Baidu `OcrEngine`/`OcrModel` signatures) → read in `forward_got`. Default plain OCR is
byte-identical. Env alias `FOCR_GOT_FORMAT`.

**Not done — blocked on assets:** the fine-grained region selectors (prompts 3–5: `<bbox> OCR:`,
`[color] OCR:`, region-reference) are INPUT-side and need `--box`/`--color` args — a later bead.

## 2. TEST-CORPUS GAP (new — blocks specialized-mode *accuracy* validation)

**No math/formula/table/chart/sheet-music/molecular test image exists anywhere** on this machine
(scout inventory: 20 navy-book plain-text page PNGs + `sample_text.png` only; `~/Downloads` has
just the navy PDF). So `--format` is wired + plumbing-tested, but its LaTeX/table/chart *accuracy*
is unproven. Need to obtain or synthesize a small specialized corpus (rendered equations via
matplotlib/LaTeX, a table screenshot, a matplotlib chart, a Verovio-rendered staff, an RDKit
molecule) + vision/hand ground truth, then add a format-mode e2e/CER gate mirroring B8.

## 3. bd-34zu — decode alloc reuse (DEFERRED — low ROI)

The decode path is entirely overwrite-only scratch, so a `DecodeScratch` + `_into` kernel
variants (`gemv_i8_bias_prequant_into`, `rms_norm_into`) would be bit-identical. BUT the scout
inventory shows the ~16 per-layer allocs are small (4–12 KB; ×24×688 ≈ 264K allocs) — the
allocator handles these in ~µs, so the realistic ceiling is ~0.05–0.1 s against a 6.35 s
gemv+misc that is **GEMV-compute (bandwidth) bound**, not alloc-bound. The one larger alloc is
`refine_topk_f32`'s `Vec<u32>[vocab≈151860]` (607 KB/token) — cheap to hoist to a reused buffer
if we touch this at all. **Verdict: not worth the `_into`-kernel surgery now; revisit only if a
profile shows alloc pressure.** Full per-alloc table in the scout output.

## 4. bd-e4yr — route GOT/Qwen2 int8 prefill onto `src/simd` (x86 win) — FEASIBLE, LOW RISK

**Value is x86-only** (on Apple Silicon both the ft-kernel prefill and the simd path use SDOT;
the win is on AVX2/AVX-VNNI x86 — trj/hetzner — where ft-kernel's `linear_int8_dynamic` falls to
**scalar** int8 and has no SMMLA). Key finding: `decoder::gemv_i8_batched` (decoder.rs) is
**already an `m=N` blocked driver** over `simd::igemm_s8s8` (the full ISA ladder — SDOT/SMMLA/
AVX2/AVX-VNNI/AVX-512-VNNI, 2×2 register-blocked, cross-tier parity-tested). Plan: add a
bias-aware, row-major-out `linear_i8_prefill(x, qw, bias)` shaped like `gemv_i8_batched`, then
swap the `nn::linear_int8_dynamic` calls in `forward_prefill_seed` (qkv + o + mlp).

**The one footgun (bit-exactness — must keep the prefill certs green):** ft-kernel quantizes
prefill activations as `round_ties_even(v / scale)` — a **division**, NOT the `v * inv` used by
`quantize_row_i8_te` nor the half-away `.round()` of `quantize_row_i8`. The new path MUST match
ft-kernel's per-row quantizer (division + ties-to-even), the same int8 weights, and the same
left-associative dequant, or `decoder_matches_torch_oracle` / `kvcache_greedy_matches_oracle_l4`
drift. Validate bit-identity on the M4 (both cert + a page diff), then measure the speedup on an
x86 host.

## 5. A7 — shared DENSE decoder engine (bd-3jo6.1.7) — re-parameterization, GQA is the hard delta

Generalize the GOT/Qwen2 driver (`decoder_qwen2.rs`) into a config-driven engine for
Qwen2Dense (GOT/OneChart), Llama, and SmolLM2 (SmolVLM2), zero GOT regression. Most of the driver
is **already `DecoderConfig`-generic** (hidden/inter/layers/heads/head_dim/vocab, RoPE θ+head_dim,
rms_eps, `attn_qkv_bias`, per-GEMM int8/f32 dispatch via `linear_auto`). The deltas to add to
`DecoderConfig` + the functions each touches:

- **GQA (the hard one — `num_kv_heads < num_heads`):** today `qkv_dim()` is used for BOTH q and
  kv; `Qwen2KvCache`, `concat_qkv`, `split_qkv_rows` assume 3 equal panels; `decode_attn_head`
  indexes `r*dim + h*head_dim` with `dim = num_heads*head_dim`; `prefill_attention` is MHA-only
  (`repeat_kv` is a no-op). Needs a `num_key_value_heads` field + kv-head broadcast in both the
  prefill and the decode attention + the kv-cache stride.
- **RoPE variants** (YARN/NTK scaling — currently vanilla NEOX rotate-half, no scaling),
  **tied-vs-untied lm_head**, **activation** (silu vs gelu), **norm** variant.

Maps to A7.1 (int8 micro-kernels — **already arch-optimal**, per the kernel map), A7.2 (attention
GQA/RoPE + the int8+refine lm_head shipped in bd-2dlz), A7.3 (MLP/norms/embed/lm_head), A7.4
(fused decode driver — largely done for the dense case), A7.5 (batch spine). Keep GOT byte-identical
(the four env-gated GOT oracle certs stay green throughout).
