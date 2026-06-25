# franken_ocr — Negative-Evidence Ledger

This ledger records optimization attempts and design levers that **failed,
regressed, were neutral, or could not be measured head-to-head**. It exists to
prevent stale optimism from being reused as proof, and to stop the swarm from
re-attempting a lever that has already been shown not to pay.

**A "win" only counts with a head-to-head MEASURED ratio against a real
reference and a correctness proof.** Anything else lands here, not in
`docs/PERF_LEDGER.md`. Do not retry a rejected lever unless its explicit retry
condition is satisfied.

## Per-entry schema

Every entry records:

```
date | WIN / NEGATIVE(reverted) | lever (what was tried, where)
  claim_id / evidence_id                         # artifact graph IDs
  model source commit + fixture hash             # exact reference provenance
  CPU feature string + exact command/env         # reproduces the run
  fallback / kill-switch state                   # proves what path was active
  measured before -> after vs reference (ratio)   # real numbers or "blocked: <why>"
  bit-exact correctness proof                      # test name + result, or the precision contract
  disposition: KEEP / REVERT
  do-not-retry: "do not retry X unless Y"          # the explicit retry condition
  per-lever tally: W / L / N                        # wins / losses / neutral across attempts
  agent                                             # who ran it
```

A lever that does not clear its measurement bar is **REVERTED**, not kept. The
`per-lever tally` accumulates across attempts so a thrice-failed idea is visibly
dead.

---

## Known negative results inherited from sibling projects

These are **not** franken_ocr measurements. They are carried over from
`frankensearch` / `frankentorch` because franken_ocr will hit the identical
kernel-design decisions, and re-litigating them would waste swarm time. Treat
them as priors, then re-confirm on *this* model's exact shapes before relying on
them.

### NE-INH-1 — naive hand-written wide-SIMD int8 dot was ~5× SLOWER than LLVM autovectorization

- **lever:** replace a scalar / autovectorized int8 dot-product inner loop with a
  hand-written wide-SIMD (manually unrolled vector-width) implementation.
- **measured (frankensearch / frankentorch):** the hand-rolled wide-SIMD int8 dot
  ran **~5× SLOWER** than simply letting LLVM autovectorize the straightforward
  scalar loop. The compiler's autovectorizer already produced better code than
  the naive intrinsics path.
- **disposition:** REVERT (never landed as the default).
- **do-not-retry:** do **not** retry naive, manually-vectorized wide-SIMD over a
  clean autovectorizable scalar int8 dot **unless** the kernel is a *tiled*
  GEMM using the dedicated dot-product instructions (NEON `SDOT`, i8mm `SMMLA`,
  AVX-512-VNNI `VPDPBUSD`, AMX) with register-blocking and accumulator tiling —
  i.e. a fundamentally different kernel shape, not a wider scalar loop. A flat
  wide-SIMD dot is a known dead end.
- **per-lever tally:** W 0 / L 1 / N 0 (inherited)
- **agent:** inherited (frankensearch / frankentorch)

### NE-INH-2 — frankentorch's SDOT/VNNI int8 dot is still ~1.5–2.4× behind ONNX/MLAS

- **lever:** frankentorch's current int8 dot-product path using `SDOT` (aarch64)
  and VNNI (x86) for matmul.
- **measured (frankentorch):** even with the dedicated dot-product instructions,
  the int8 matmul path remains **~1.5–2.4× behind ONNX Runtime / MLAS** on CPU.
  The gap is real and persistent.
- **diagnosis:** the missing piece is a **model-specific tiled `SMMLA`/`VNNI`
  GEMM** with proper register blocking, packed/pre-transposed weights, and
  accumulator tiling — i.e. the kernel franken_ocr's whole thesis is built on.
  This is the **unbuilt fix**, not a refutation of the approach. Closing this gap
  on Unlimited-OCR's fixed GEMM shapes is the central technical bet.
- **disposition:** N/A — this is the gap franken_ocr exists to close, recorded so
  nobody declares victory on the un-tiled `SDOT`/`VNNI` path or mistakes the
  current frankentorch number for the ceiling.
- **do-not-retry:** do **not** claim a CPU int8 GEMM win **unless** it is measured
  against ONNX/MLAS or the Phase -1 proven CPU baseline on this model's actual shapes
  with the tiled GEMM in place — the un-tiled dot path is already known to lose.
- **per-lever tally:** W 0 / L 1 / N 0 (inherited; the tiled-GEMM fix is unbuilt)
- **agent:** inherited (frankentorch)

### NE-INH-3 — un-blocked tiled SMMLA was SLOWER than SDOT (load-bound)

- **lever:** a tiled `SMMLA` (i8mm) int8 GEMM with 2× the MAC density of `SDOT`,
  but WITHOUT register/cache blocking.
- **measured (frankensearch/frankentorch, M4):** **19 / 41 / 77 ms** vs SDOT's
  14.8 / 34 / 64 — a **regression**, despite double the MAC throughput, because the
  kernel re-loads the activation for every weight pair (≈**2 loads : 1 SMMLA**) and
  is therefore **load-bound, not compute-bound**. Extra MAC throughput is wasted
  when you are memory-bound.
- **disposition:** REVERT.
- **do-not-retry:** do **not** add a wider/denser matmul instruction (SMMLA, AMX)
  **unless** the micro-kernel already has **register/cache blocking with
  compute:load ≥ 2:1 and offline-pre-packed weights**. The instruction is not the
  lever; the blocking is.
- **per-lever tally:** W 0 / L 1 / N 0 (inherited)
- **agent:** inherited (frankensearch/frankentorch)

### NE-INH-4 — AMX-f32 (Accelerate) does NOT beat ONNX-int8

- **lever:** route the matmuls through Apple's AMX coprocessor in **f32** (via
  Accelerate/numpy) as a "Mac finisher".
- **measured (M4):** ~**11 / 28 / 77 ms** f32 — does not beat ONNX-int8
  (7.6/14.5/41.4), because f32 streams **4× the bytes** of int8 on these
  **memory-bound** sizes, and the element-wise ops (softmax/GELU/transpose) are not
  on AMX anyway.
- **disposition:** REVERT (not the easy finisher).
- **do-not-retry:** do **not** chase AMX **unless** it is **int8** (low bandwidth),
  applied to **compute-bound prefill** (not memory-bound decode), AND the FFI cost
  of Accelerate/BNNS is accepted as an **opt-in feature** (the directly-programmable
  Mac int8 path is NEON SMMLA/SDOT, no FFI).
- **per-lever tally:** W 0 / L 1 / N 0 (inherited)
- **agent:** inherited (frankensearch/frankentorch)

### NE-INH-5 — naive hand-written "fused tape-free forward" regressed 3–10× (the most clarifying failure)

- **lever:** delete the per-op framework tape/dispatch overhead by hand-writing a
  single fused forward — BUT with **naive scalar-f32 attention / softmax /
  LayerNorm** replacing the library's SIMD/parallel kernels.
- **measured (frankensearch, M4):** **38 / 194 / 580 ms** — a **3–10× regression**
  (seq512 was 10× the kernel version). This **disproved the "the gap is all
  framework overhead" theory**: the real gap to ONNX is **kernels below peak**
  (SDOT-not-SMMLA linears, f32-not-int8 attention), not per-op tape cost.
- **disposition:** REVERT.
- **do-not-retry:** the fused, tape-free, zero-per-op-allocation forward is the
  RIGHT architecture (franken_ocr is built that way), but **every fused op must
  stay at peak** (SIMD + parallel + int8/int4). Do **not** trade a good library
  kernel for a naive hand-written one — ever. Measure framework-tax savings only
  with at-peak ops on both sides.
- **lesson for franken_ocr:** out-SPECIALIZE ONNX (fused single-model forward) AND
  keep every op at peak; both are required, neither alone wins.
- **per-lever tally:** W 0 / L 1 / N 0 (inherited)
- **agent:** inherited (frankensearch)

---

## franken_ocr measurements

_None yet. No franken_ocr lever has been measured. This section stays empty until
a real head-to-head ratio with a correctness proof exists — no fabricated
results._
