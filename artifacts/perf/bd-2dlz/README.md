# bd-2dlz — GOT-OCR2 decode perf campaign (int8+refine head · parallel attn · last-row seed)

Follow-on to [bd-3bom](../bd-3bom/) (lm_head pre-transpose, 487→41 s). Three doctrine-compliant
levers on the remaining `got forward` time, each env-gated (one build measured every config;
the gate on the numerics lever is its doctrine-#2 kill-switch). **Verdict: 3 WINS, 0 revert.**

## Result (page_0107.png, Apple M4 `aarch64+neon+dotprod`, got-ocr2.int8.focrq, 688 tokens)

| stage | BEFORE (bd-3bom) | AFTER (new default) | lever |
|---|--:|--:|---|
| vision + splice | 5.84 s | 5.69 s | — |
| seed prefill | 1.16 s | **0.23 s** | last-row-only head |
| decode attn | 4.90 s | **2.29 s** | parallel per-head |
| decode gemv+misc | 7.59 s | 6.35 s | — |
| **lm_head** | 16.18 s | **1.94 s** | **int8 + top-K f32 refine** |
| decode total | 30.07 s (23.9 tok/s) | **10.63 s (64.7 tok/s)** | |
| **got forward** | **35.93 s** | **16.6 s** | **2.16×** |

Cumulative from the 8-min starting point: **got forward 482.85 s → 16.6 s (29×)**, decode
**1.4 → 64.7 tok/s (46×)**.

## The three levers

1. **int8 `lm_head` + top-K f32 refine** — `FOCR_GOT_INT8_LMHEAD`, **default ON**. The lm_head
   reads the ~0.6 GB f32 tied head per token (memory-bound). Quantize it to per-output-channel
   int8 (the `gemv_i8` path — **SDOT on Apple Silicon, AVX-512-VNNI/AVX-VNNI/AVX2 on Intel-AMD**,
   register-blocked, N-parallel, ~0.15 GB), then recompute the **top-256** candidates in exact
   f32 (`normed · embed_row`) so the greedy argmax matches the f32 head. **8.3× on the head.**
   Kill-switch `FOCR_GOT_INT8_LMHEAD=0` → the provably-bit-identical f32 head.
2. **Parallel per-head decode attention** — `FOCR_GOT_SEQ_ATTN` disables. Each of the 16 heads is
   a self-contained softmax over disjoint output lanes → the rayon fan-out is **bit-identical**.
   **2.1× on attention** (5.02→2.43 s, isolated in config C).
3. **Last-row-only seed head** — the seeding prefill projected the full `[N, vocab]` head then
   sliced the last row; now it projects only the last row (the others are never argmaxed).
   **5× on seed** (1.16→0.23 s). Bit-identical.

## Correctness (doctrine #1 — parity first)

The int8 head is BEYOND doctrine #2's validated set, so it is a MEASURED kill-switch — and it is
**validated, not assumed**:

- **Byte-identical output** to the f32 head on page_0107 (688/688 tokens) — the refine recovers
  the exact greedy pick.
- The env-gated **torch-oracle certs RAN on this default and PASSED**: `kvcache_greedy_matches_oracle_l4`
  (decode == oracle L4 `[9707,38,1793,12,93495,17,13,15]`), `decoder_matches_torch_oracle`,
  `recognize_reads_the_sample_image_e2e` (committed golden), `prompt_id_oracle_cross_check`.
- Full gate: `cargo fmt`/`clippy -D warnings` clean, **745 lib tests / 0 failed**, `ubs --diff` 0 critical.

Parallel attn + last-row seed are bit-identical by construction (no numeric change).

## Kernel-map note (why no new kernels / no bf16)

A subagent mapped `ft-kernel-cpu` + franken_ocr `src/simd`: the decode int8 GEMV already dispatches
the **full per-arch ISA ladder** (SDOT / SMMLA-not-on-M-series / AVX2 / AVX-VNNI / AVX-512-VNNI),
register-blocked, N-parallel, and **bandwidth-bound at m=1** — so these wins come from moving work
ONTO it, not writing kernels. **No bf16/BFMMLA kernel exists** in the stack, so a bf16 vision lever
was ruled out (and vision int8 wrecks OCR per doctrine #2). Open x86 lever: GOT prefill + qkv still
route through `nn::linear_int8_dynamic` (ft-kernel-cpu), which is scalar int8 on x86 below
AVX-512-VNNI — a separate follow-up.

Raw before/after timing lines: `perf_before_page0107.txt`, `perf_after_page0107.txt`; `SHA256SUMS` anchors them.
