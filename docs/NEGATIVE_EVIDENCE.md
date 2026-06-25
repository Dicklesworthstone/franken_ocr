# franken_ocr — Negative-Evidence Ledger

This ledger records optimization attempts and design levers that **failed,
regressed, were neutral, or could not be measured head-to-head**. It exists to
prevent stale optimism from being reused as proof, and to stop the swarm from
re-attempting a lever that has already been shown not to pay.

**A "win" only counts with a head-to-head MEASURED ratio against a real
reference and a correctness proof.** Anything else lands here, not in
`docs/PERF_LEDGER.md`. Do not retry a rejected lever unless its explicit retry
condition is satisfied.

This is an **artifact-graph ledger** (plan §8.4), not prose: every entry carries
the FrankenSuite artifact-graph fields so each claim is reproducible and traceable
to the exact model version it was measured against.

## Canonical provenance source (the truth pack)

Every entry's provenance fields resolve against the **Phase −1 truth pack**, the
single immutable anchor for "which model, which sources, which numbers":

- **Model source commit:** Hugging Face `baidu/Unlimited-OCR`
  **`3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`** (GitHub
  `7e98affeacba24e95562fbaa234ddb89b856874a`), verified 2026-06-25 via
  `git ls-remote` — see `docs/truth-pack/PINNED_SOURCES.md`.
- **Source / fixture hashes:** the SHA-256 of every load-bearing source
  (`config.json`, `modeling_unlimitedocr.py`, `modeling_deepseekv2.py`,
  `deepencoder.py`, `tokenizer.json`, …) is recorded in
  **`docs/truth-pack/SOURCE_HASHES.md`**. The `model source commit + fixture hash`
  field of every entry below cites `(file_sha256, line range)` against that table;
  the **weights fixture hash** is the SHA-256 of `model-00001-of-000001.safetensors`
  (recorded in `SOURCE_HASHES.md` once fetched out-of-band) plus the `.focrq`
  conversion hash for the precision actually measured.
- **Runtime pin:** the reference oracle stack is `torch==2.10.0`,
  `transformers==4.57.1`, `Pillow==12.1.1` (`PINNED_SOURCES.md`); a number measured
  against any other stack is **not comparable** and does not belong here.

If `SOURCE_HASHES.md` ever fails to verify, the upstream model moved: STOP, re-pin
(`PINNED_SOURCES.md`), and re-confirm every entry whose provenance points at the
old commit. A franken_ocr entry without a resolvable truth-pack provenance is
**incomplete and may not be cited as evidence**.

## Per-entry schema

Every entry records (the frankentorch format **plus** the artifact-graph fields):

```
date | WIN / NEGATIVE(reverted) | lever (what was tried, where)
  claim_id / evidence_id                         # artifact-graph IDs (claim under test → evidence dir)
  model source commit + fixture hash             # truth-pack provenance: HF 3a7f4db… + (file_sha256, lines)
                                                  #   from SOURCE_HASHES.md, plus .focrq/weights hash
  CPU feature string                             # the DISPATCHED SIMD tier (e.g. aarch64+neon+dotprod+i8mm,
                                                  #   x86_64+avx2+avxvnni, x86_64+avx512vnni) — not the host's max
  exact command + env                            # the literal gauntlet invocation + FOCR_*/OMP_NUM_THREADS/RAYON_* set
  fallback / kill-switch state                   # which path was active: FOCR_INT8_ATTN / FOCR_INT8_LMHEAD /
                                                  #   mimalloc feature / int4-group on|off — proves what ran
  measured before -> after vs reference (ratio)   # real numbers or "blocked: <why>" (ratio = ref_time / focr_time)
  bit-exact correctness proof                      # test name + result, or the precision contract (ULP/CER bound)
  disposition: KEEP / REVERT
  do-not-retry: "do not retry X unless Y"          # the explicit retry condition
  per-lever tally: W / L / N                        # wins / losses / neutral across attempts
  agent                                             # who ran it
  evidence dir: artifacts/perf/<bead>/             # paired baseline/after gauntlet logs + SHA-256 manifest
```

A lever that does not clear its measurement bar is **REVERTED**, not kept. The
`per-lever tally` accumulates across attempts so a thrice-failed idea is visibly
dead. The **evidence dir** `artifacts/perf/<bead>/` holds the paired baseline/after
gauntlet logs and their SHA-256 manifest — the `evidence_id` points at it, so the
ledger row and the raw artifacts are graph-linked.

**Provenance scope of the inherited priors below.** The `NE-INH-*` entries are
carried over from `frankensearch` / `frankentorch` and were measured on **those**
models, *not* on Unlimited-OCR at `3a7f4db…`. Their provenance field is therefore
`inherited (pre-truth-pack)` by construction: they are **priors to re-confirm on
this model's exact shapes**, never franken_ocr evidence. The first real
franken_ocr entry — and every one after — MUST carry full truth-pack provenance.

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
- **provenance:** `inherited (pre-truth-pack)` — measured on frankensearch/
  frankentorch, NOT on Unlimited-OCR `3a7f4db…`; a prior to re-confirm on this
  model's exact GEMM shapes, not franken_ocr evidence.
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
- **provenance:** `inherited (pre-truth-pack)` — frankentorch measurement; the gap
  it names is the one franken_ocr exists to close on Unlimited-OCR `3a7f4db…`'s
  fixed GEMM shapes (`SOURCE_HASHES.md`: `config.json`, `model.safetensors.index.json`).
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
- **provenance:** `inherited (pre-truth-pack)` — frankensearch/frankentorch on M4;
  re-confirm against Unlimited-OCR `3a7f4db…`'s `down_proj` (K=6848) before relying
  on it (the load-bound regime depends on this model's exact tile shapes).
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
- **provenance:** `inherited (pre-truth-pack)` — M4 Accelerate/AMX-f32 vs ONNX-int8;
  a memory-bandwidth prior, re-confirm on this model's prefill shapes before any
  AMX experiment lands as a franken_ocr lever.
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
- **provenance:** `inherited (pre-truth-pack)` — frankensearch M4 seq{128/256/512};
  the architectural lesson (fused forward with at-peak ops) is adopted by
  franken_ocr, but the regression numbers are NOT this model's — the first
  franken_ocr fused-forward measurement carries truth-pack provenance.
- **per-lever tally:** W 0 / L 1 / N 0 (inherited)
- **agent:** inherited (frankensearch)

---

## franken_ocr measurements

_None yet. No franken_ocr lever has been measured. This section stays empty until
a real head-to-head ratio with a correctness proof exists — no fabricated
results._

The first real entry MUST carry **full truth-pack provenance** (model commit
`3a7f4db…` + `(file_sha256, lines)` from `SOURCE_HASHES.md` + weights/`.focrq`
hash) and a paired `artifacts/perf/<bead>/` evidence dir. Shape to follow (a
**template**, not a measurement — note the empty number fields):

```
2026-MM-DD | <WIN|NEGATIVE(reverted)> | <lever, file:fn>
  claim_id: <e.g. CLAIM-int8-expert-ffn-decode>   evidence_id: artifacts/perf/<bead>/
  model source commit + fixture hash:
    HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5
    config.json sha256 27246d03…  (SOURCE_HASHES.md)
    model-00001-of-000001.safetensors sha256 <recorded-when-fetched>
    <model>.focrq sha256 <conversion hash for the precision measured>
  CPU feature string: <dispatched tier, e.g. aarch64+neon+dotprod+i8mm>
  exact command + env:
    cargo bench -p focr --bench gauntlet -- decode-per-token
    FOCR_REFERENCE_PYTHON=<onnx|hf>  OMP_NUM_THREADS=8  RAYON_NUM_THREADS=8
    (reference torch set_num_threads(8) — NEVER @64, §9.3)
  fallback / kill-switch state: FOCR_INT8_ATTN=<0|1>  FOCR_INT8_LMHEAD=<0|1>
    int4-group=<off|g32|g16>  allocator=<system|mimalloc-feature>
  measured before -> after vs reference: <ref_ms> / <focr_ms> -> ratio <x.xx>  (or "blocked: <why>")
  correctness proof: <test name> -> <pass|CER Δ within AF-2 budget|4-ULP table>
  disposition: <KEEP|REVERT>
  do-not-retry: "do not retry <X> unless <Y>"
  per-lever tally: W <n> / L <n> / N <n>
  agent: <pane/agent id>
```
