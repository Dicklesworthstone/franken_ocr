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
  CPU feature string                             # the DISPATCHED SIMD tier (e.g. aarch64+neon+dotprod,
                                                  #   aarch64+neon+i8mm, x86_64+avx2+avxvnni,
                                                  #   x86_64+avx512vnni) — not the host's max
  exact command + env                            # the literal gauntlet invocation + FOCR_*/OMP_NUM_THREADS/RAYON_* set
  fallback / kill-switch state                   # which path was active: FOCR_INT8_ATTN / FOCR_INT8_LMHEAD /
                                                  #   mimalloc feature / int4-group on|off — proves what ran
  measured before -> after vs reference (ratio)   # real numbers or "blocked: <why>" (ratio = ref_time / focr_time)
  bit-exact correctness proof:                     # test name + result, or the precision contract (ULP/CER bound)
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

No `PERF_LEDGER.md` head-to-head row exists yet: there is still no certified
Phase -1 CPU-reference ratio for this path. The entries below are local,
synthetic before/after microbenches that retire or keep one narrow loop lever
and preserve the raw artifact bundles for gauntlet follow-up; they are not G2
claims.

2026-07-07 | NEGATIVE(reverted) | row-tiled SAM global-attention score matrix (bd-av64.10, commits c5e535a+8bd4037 reverted)
  claim_id: CLAIM-bd-av64.10-attn-row-tile   evidence_id: artifacts/perf/bd-av64.10-rowtile/
  model source commit + fixture hash: unlimited-ocr.int8.focrq sha256 d8c5fcf2… + got-ocr2.int8.focrq sha256 4da43d79…; page_0009.png + got sample_text.png
  CPU feature string: aarch64+neon (f32 attention path)
  exact command + env: interleaved same-regime pairs, FOCR_SAM_TILE=0 (untiled) vs =128, FOCR_TIMING=1 focr ocr page_0009.png; 4 pairs
  fallback / kill-switch state: FOCR_SAM_TILE measurement knob (removed with the revert)
  measured before -> after vs reference:
    HYPOTHESIS: the [4096,4096] per-head score matrix (64 MB) makes four DRAM round-trips (GEMM-write, bias, softmax, GEMM-read);
    a [128,4096] row tile stays cache-resident through all four stages -> save the traffic.
    BYTE-IDENTITY: PROVEN — tiled outputs byte-identical on BOTH real models (unlimited page + GOT sample), i.e. the ft sgemm
    per-element K-reduction order is M-invariant. The restructuring was numerically pure; it just wasn't faster:
    UNTILED WINS ALL 4 PAIRS: sam.forward 3.62/3.62/3.74/3.71 s untiled vs 3.76/3.84/3.86/3.76 s tiled (~+3% regression tiled).
    ROOT CAUSE: tiling turns 2 large GEMMs/head into 64 small dispatches/head (1024/block) — dispatch + lost intra-GEMM
    parallelism outweigh the cache benefit; on M4's unified memory the score-matrix round-trips were never the wall
    (heads already overlap traffic with compute across the 10-way head parallelism).
  bit-exact correctness proof: byte-identical outputs both arms (above); revert restores the exact pre-lever source
  disposition: REVERT (8bd4037 reverts c5e535a, which a concurrent session had committed mid-measurement)
  do-not-retry: "do not re-tile SAM attention on Apple-silicon unified memory unless a profile shows the score matrix
    actually DRAM-bound (e.g. a low-bandwidth x86 target) — and then tile WITHOUT multiplying GEMM dispatches (fused
    bias+softmax inside one blocked kernel, not a loop of nn::matmul calls)"
  per-lever tally: W 0 / L 1 / N 0
  agent: RubyCove
  evidence dir: artifacts/perf/bd-av64.10-rowtile/

2026-07-09 | NEGATIVE(reverted) | SDOT micro-tile dispatch for the DENSE int8 GEMMs on Apple Silicon (`arm::igemm_s8s8`/`igemm_u8s8` preferring `aarch64_impl::igemm_s8s8_sdot`)
  claim_id: CLAIM-bd-2mo51-sdot-gemv   evidence_id: artifacts/perf/bd-2mo.5.1/ (quiet-window tier microbench logs + README tables)
  model source commit + fixture hash: baidu/Unlimited-OCR 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5 (shard sha256 2bc48a7a…); page_0009.png (the A11 corpus page)
  CPU feature string: aarch64+neon+dotprod (Apple M4; i8mm present but half-rate)
  exact command + env: microbench `FOCR_FORCE_ARCH={sdot,smmla,scalar} cargo bench --bench igemm_tier` (quiet window, error bars 1-10%); e2e `FOCR_DECODE_INT8=1 FOCR_TIMING=1 focr ocr page_0009.png` interleaved A/B pre-lever (FOCR_FORCE_ARCH sdot vs scalar) and post-lever (default vs FOCR_INT8_AUTOVEC=0)
  fallback / kill-switch state: the REPLACEMENT (LLVM-autovec scalar dispatch via `autovec_preferred()`) ships default-ON on macOS aarch64 with `FOCR_INT8_AUTOVEC=0` restoring SDOT; `FOCR_FORCE_ARCH` overrides still honored; detect_tier()/robot backends still report the hardware truth; int4-packed + offline-SMMLA paths unaffected
  measured before -> after vs reference:
    microbench (ns/iter, model shapes): sdot 93,775 / 511,095 / 505,927 (m=1 attn/down/gate_up) vs autovec 21,145 / 114,818 / 114,942 — autovec 4.4-4.5x FASTER at m=1, 2.0x at m=16; SMMLA 2.3-6.2x behind SDOT everywhere (bd-2mo.4.1 verdict)
    e2e int8 decode (98 tokens, identical output bytes): 1.61s/1.58s (sdot) vs 1.27s/1.26s (autovec) pre-lever; 1.64s vs 1.27s post-lever kill-switch A/B = 22% decode win, decode/tok 16.8ms -> 13ms
  bit-exact correctness proof: all tiers are proven bit-identical (exact i32 integer accumulation): the simd tier-parity suite (53 tests incl the randomized + constant-extreme oracles at K=6848) holds every route equal, and both e2e A/B arms emitted BYTE-IDENTICAL 98-token output on page_0009 — the lever changes WHICH exact kernel runs, never a bit of the result.
  WHY (attribution): NE-INH-1's inherited prior finally caught the m=1 GEMV — the hand-written SDOT micro-tile pays per-call setup/blocking the fused autovectorized loop never does, and current LLVM (nightly, 2026) vectorizes the plain i8-dot loop better than the hand kernel at every model shape. The SDOT preference predated this toolchain and was never re-measured at m=1 against autovec.
  disposition: REVERT (the SDOT preference for the dense igemm entrypoints); the replacement autovec dispatch ships default-ON on macOS aarch64 behind FOCR_INT8_AUTOVEC (=0 restores SDOT).
  per-lever tally: W 0 / L 1 / N 0 (the SDOT-at-dense-GEMM preference, first honest m=1-vs-autovec measurement)
  agent: claude (bd-2mo.5.1 / bd-2mo.4.1 quiet-window session, 2026-07-09)
  evidence dir: artifacts/perf/bd-2mo.5.1/ (tier microbench logs, README tables, SHA256SUMS)
  do-not-retry: do NOT re-prefer the SDOT micro-tile for the dense igemm entrypoints on Apple Silicon without a fresh quiet-window tier microbench + e2e pair on the CURRENT toolchain showing SDOT >= autovec; re-measure on new silicon (M5+), new LLVM major, or if the GEMV call pattern changes shape (e.g. fused multi-row batching lands).

2026-07-07 | NEGATIVE(reverted) | polynomial-exp softmax in SAM attention (bd-av64.10 remaining-lever list, `FOCR_SAM_FAST_EXP` in `nn::softmax_rows_fast`)
  claim_id: CLAIM-bd-av64.10-simd-exp   evidence_id: artifacts/perf/bd-av64.10-simd-exp/ (pointers; measured tables inline below)
  model source commit + fixture hash: unlimited-ocr.int8.focrq sha256 d8c5fcf2… (published); real page fixture = franken_ocr_work/pages/page_0009.png (the A11 corpus page)
  CPU feature string: aarch64+neon (softmax is the f32 path; int8 tier irrelevant)
  exact command + env: paired A/B, 2 runs per arm, adjacent same-regime: FOCR_TIMING=1 [FOCR_SAM_FAST_EXP=1] focr ocr --model unlimited-ocr.int8.focrq page_0009.png
  fallback / kill-switch state: lever env-gated OFF by default; unarmed path structurally unchanged (branch at the single softmax call site)
  measured before -> after vs reference:
    DESIGN: Cephes-split branch-free poly exp (LLVM-autovectorizable, honoring doctrine #3) replacing the per-element libm expf
    inside a transcription of the reference kernel's exact max/pairwise-sum/divide structure. UNIT ACCURACY PROVEN: worst rel err
    2.53e-7 (~2.1 ulp) over [-87, 10]; softmax probability drift <= 1.1e-8 absolute vs the reference kernel.
    WALL CLOCK: sam.forward ref 3.49/3.80 s vs fast 3.75/3.48 s — ZERO. ROOT CAUSE: the "softmax exp = 0.15 s/block" target was
    profiled BEFORE pass 2 made global attention head-parallel; the exp now runs inside the parallel section, amortized across all
    cores (~0.06 s wall ceiling). The lever's premise had already been optimized away by a different lever.
    OUTPUT: the <=1.1e-8 probability drift STILL forked the greedy decode on the FIRST measured page — "CHARING CROSS" (the
    reference's CORRECT reading of the real place name) degraded to "CHAIRING CROSS", plus a heading trajectory fork
    ("THE WORLD'S FIRST" -> "THE WORLD'S LITTLE TOUR"). The fast arm was self-consistent across runs — a deterministic wrong answer.
  bit-exact correctness proof: revert is a net-zero source diff; unarmed byte-identity was structural (branch)
  disposition: REVERT (code removed same-day; the exp implementation + accuracy tests live in this entry's session history)
  do-not-retry: "do not retry approximate exp (or ANY sub-ulp numerics substitution) in the vision softmax unless (1) a fresh
    profile shows it on the sequential critical path again (e.g. after an online-softmax restructuring changes the parallel
    shape), AND (2) the full 20-page corpus token-exactness gate arbitrates. LESSON: 1e-8 probability drift flips greedy tokens
    on real pages — the vision path has NO numerics slack; 'accuracy proven at the unit level' does not transfer to token-exactness."
  per-lever tally: W 0 / L 1 / N 0
  agent: RubyCove
  evidence dir: none (reproduces from the two commands above at this entry's commit)

2026-07-07 | NEGATIVE(reverted) | TrOMR 1-crop-page routing through the refined staff band (bd-av64.13 item b, `recognize_page` single-staff branch)
  claim_id: CLAIM-bd-av64.13-onecrop-route   evidence_id: artifacts/perf/bd-av64.13/ (pointers; measured tables inline below + the bead close note)
  model source commit + fixture hash: same TrOMR export/artifact as the TTA entry below; fixtures = committed realscan_music corpus v1
  CPU feature string: arm64 Apple M4 (f32 forward)
  exact command + env: FOCR_BIN=<release focr> bash scripts/realscan_music_gate.sh  (routing active unconditionally in the build under test; whole-image fallback retained on crop failure)
  fallback / kill-switch state: none (the lever was the default path in the test build; reverted same-day)
  measured before -> after vs reference:
    HYPOTHESIS: a page whose detector finds EXACTLY 1 staff should route through the refined crop (FIT-FIRST geometry +
    per-band residual-skew refinement) so the measured -0.7 deg key knife-edge gets lever-1 protection on 1-crop inputs.
    GATE VERDICT: spohr_no17_top (the committed golden, a clean single-staff crop) REGRESSED — time signature dropped
    ("time != 3/4") AND bar-1 note flip (E5:quarter -> G5:quarter); golden no longer byte-stable. Every other fixture held.
    MECHANISM: band extraction re-trims an already-tight crop; the decode is knife-edge sensitive to exactly those margins
    (the same sensitivity FIT-FIRST exists to protect on the >=2-staff path, where historic geometry is preserved bit-identically).
    On a pre-cropped staff the whole image IS the historic geometry.
  bit-exact correctness proof: revert restores gate ALL PASS incl. golden byte-stable (re-run same command)
  disposition: REVERT (contract comment at recognize_page's <2 branch records the measurement in-code)
  do-not-retry: "do not route 1-crop pages through band extraction unless the crop is proven pixel-identical to the input
    (or the corpus gains skewed single-staff fixtures that measure a net win, bd-av64.15) — the certified whole-image read
    is the historic geometry for pre-cropped staves"
  per-lever tally: W 0 / L 1 / N 0
  agent: RubyCove
  evidence dir: none (reproduces from the committed corpus at the pre-revert commit)

2026-07-07 | NEGATIVE(reverted) | TrOMR micro-rotation self-consistency vote (bd-av64.13 lever 2, `FOCR_TROMR_TTA=3` in `src/native_engine/tromr.rs::recognize_voted`)
  claim_id: CLAIM-bd-av64.13-tta-vote   evidence_id: artifacts/perf/bd-av64.13/ (pointers; per-candidate logs inline below + the bead close note)
  model source commit + fixture hash:
    Polyphonic-TrOMR export model.safetensors sha256 41c88802… (TROMR_EXPORT_MANIFEST.json); tromr.focrq sha256 a9d41485… (all-f32)
    fixtures: tests/fixtures/realscan_music (corpus v1, committed) — the measuring instrument, not a synthetic bench
  CPU feature string: arm64 Apple M4 (TrOMR forward is f32; no SIMD tier lever involved)
  exact command + env:
    FOCR_TROMR_TTA=3 FOCR_BIN=<release focr> bash scripts/realscan_music_gate.sh   (vs the identical unarmed control)
    per-candidate diagnostics: FOCR_TROMR_TTA=3 FOCR_TIMING=1 focr ocr --model tromr.focrq --task music <staff.png>
  fallback / kill-switch state: lever was env-gated OFF by default; the unarmed path was byte-identical to the certified pipeline (early-return)
  measured before -> after vs reference:
    DESIGN: decode each staff at 0.0/-0.4/+0.4 deg (deskew shear kernel), majority-vote key+time, tie-break by fewest
    bar-sum sanity warnings, final tie -> the 0 deg certified decode.
    corpus gate: unarmed ALL PASS (22.5 s) -> armed 1 FAIL (spohr_no17_sys attributes REGRESSED, time != 3/4) at 62.9 s (2.8x);
    the no21 time XFAIL (the lever's target) did NOT flip.
    MECHANISM (per-candidate logs): micro-rotation makes the model DROP the time signature on degraded 1843 scans
    (3 of 4 measured staves omit timeSignature at ±0.4 deg). An omitted attribute (a) cannot form a majority and
    (b) INVERTS the bar-sum scorer — no declared time means no checkable bar constraint, so the degraded decode scores
    0 warnings and beats the honest 0 deg decode that declared its time and got flagged (no17_sys staff 1: correct
    "3/4 + 3 warnings" lost to "no time + 0 warnings"; no21_sys staff 2: the TRUTH "3/4" at +0.4 deg lost to the silent 0 deg).
    SCORER-V2 DRY RUN (attribute-presence ranked above warning count, evaluated against the same logged candidates):
    fixes the no17_sys regression but still does not flip no21 (staff 2's presence-tie resolves to "C" = 4/4, staff 1's
    majority is 4/4 either way) -> best-known variant nets ZERO corpus improvement at 2.8x cost.
  bit-exact correctness proof: unarmed path structurally unchanged (env early-return); control gate run ALL PASS incl. golden byte-stable
  disposition: REVERT (code removed same-day; the pure vote fn + property tests live in this entry's commit history)
  do-not-retry: "do not retry rotation-TTA voting on TrOMR unless (1) the bd-av64.15 corpus expansion provides a held-out
    calibration set (scorer tuning on the 6 measuring fixtures is overfit by construction) AND (2) the scorer ranks
    attribute PRESENCE above warning count — omission-inversion is the failure mode that killed v1"
  per-lever tally: W 0 / L 1 / N 0
  agent: RubyCove
  evidence dir: none (fixtures are committed; per-candidate logs reproduce with the two commands above at the pre-revert commit)

2026-06-25 | NEGATIVE(reverted) | strided-destination projector transpose in `src/native_engine/vision_bridge.rs::transpose`
  claim_id: CLAIM-bd-1gv.10.1-projector-transpose-store-order   evidence_id: artifacts/perf/bd-1gv.10.1/
  model source commit + fixture hash:
    HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5
    config.json sha256 27246d03fd670904ec9601b1cb0861fbb79ec076830771daa8d943d6229946f9 (SOURCE_HASHES.md)
    synthetic projector fixture: artifacts/perf/bd-1gv.10.1/projector_bench_main.rs sha256 999973e4948e232ec955ae0691ce2dfcc2b362e2ddfc759b4122cc7aa58144ee (SHA256SUMS)
  CPU feature string: arm64 Apple M4, dotprod=1, i8mm=1; projector path is f32, no SIMD tier override
  exact command + env:
    hyperfine --warmup 2 --runs 9 --export-json artifacts/perf/bd-1gv.10.1/{baseline,after}_projector_hyperfine.json
    RAYON_NUM_THREADS=8 OMP_NUM_THREADS=8 /Volumes/USBNVME16TB/temp_agent_space/focr_projector_bench_target/release-perf/focr_projector_bench --iters 16
  fallback / kill-switch state: no FOCR_* performance kill-switches set; allocator=system; harness calls the real `vision_bridge::project`
  measured before -> after vs reference:
    local focr-only projector microbench (no reference ratio): 187.9 ms +/- 4.2 ms -> 132.5 ms +/- 11.0 ms for 16 calls, mean speedup 1.419x; PERF_LEDGER ineligible until the gauntlet has a pinned CPU reference row
  bit-exact correctness proof:
    smoke checksum unchanged for 4 calls (-0.039779253 before and after); `CARGO_TARGET_DIR=/Volumes/USBNVME16TB/temp_agent_space/focr_verify_target_whitecave TMPDIR=/Volumes/USBNVME16TB/temp_agent_space/tmp cargo test --lib native_engine::vision_bridge -- --nocapture` -> 13 passed, 0 failed
  disposition: REVERT
  do-not-retry: "do not return to output-strided transpose stores for projector weights unless a new head-to-head gauntlet row proves a different packed/projector path wins on the pinned fixture"
  per-lever tally: W 0 / L 1 / N 0
  agent: WhiteCave
  evidence dir: artifacts/perf/bd-1gv.10.1/

2026-06-25 | WIN | per-query decomposed rel-pos bias precompute in `src/native_engine/vision_sam.rs::attention`
  claim_id: CLAIM-bd-3n16-sam-relpos-bias-precompute   evidence_id: artifacts/perf/bd-3n16/
  model source commit + fixture hash:
    HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5
    config.json sha256 27246d03fd670904ec9601b1cb0861fbb79ec076830771daa8d943d6229946f9 (SOURCE_HASHES.md)
    synthetic SAM attention fixture: ignored unit probe `sam_attention_relpos_bias_local_probe`; evidence README sha256 6a0f8c7bc22b5ad5012ed546ab1443a4897ea43e5be31b6314d42d25f3ae721c (SHA256SUMS)
  CPU feature string: arm64 Apple M4, dotprod=1, i8mm=1; SAM attention probe is f32 through the current frankentorch facade, no SIMD tier override
  exact command + env:
    FOCR_SAM_ATTN_PROBE_RUNS=5 CARGO_TARGET_DIR=target-codex-verify cargo +nightly --config patch.crates-io.asupersync.path='"/dp/asupersync"' test --profile release-perf sam_attention_relpos_bias_local_probe --lib -- --ignored --nocapture
  fallback / kill-switch state: no FOCR_* performance kill-switches set; allocator=system; harness calls the real `vision_sam::attention`
  measured before -> after vs reference:
    local focr-only synthetic SAM attention probe (no reference ratio): 26.8490916 ms -> 17.787375 ms average for 5 calls, local speedup 1.509x and 33.75% less wall time; PERF_LEDGER ineligible until the gauntlet has a pinned CPU reference row
  bit-exact correctness proof:
    output checksum unchanged (`-0.009587256237864494` before and after); `decomposed_rel_pos_bias_matches_direct_inner_loop_formula` proves the precomputed H/W bias tables match the old direct inner-loop formula exactly; `CARGO_TARGET_DIR=target-codex-verify cargo +nightly --config patch.crates-io.asupersync.path='"/dp/asupersync"' test vision_sam::tests --lib -- --nocapture` -> 17 passed, 0 failed, 1 ignored
  disposition: KEEP
  do-not-retry: "do not recompute decomposed rel-pos dot products inside the SAM key loop unless a future batched-QK/probs@V rewrite proves a faster and parity-preserving full attention path on the pinned fixture"
  per-lever tally: W 1 / L 0 / N 0
  agent: WhiteCave
  evidence dir: artifacts/perf/bd-3n16/

2026-06-25 | NEGATIVE(reverted) | QKV split slice-copy + key-grid coordinate hoist in `src/native_engine/vision_sam.rs::attention`
  claim_id: CLAIM-bd-3n16-sam-qkv-grid-hoist   evidence_id: artifacts/perf/bd-3n16/
  model source commit + fixture hash:
    HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5
    config.json sha256 27246d03fd670904ec9601b1cb0861fbb79ec076830771daa8d943d6229946f9 (SOURCE_HASHES.md)
    synthetic SAM attention fixture: ignored unit probe `sam_attention_relpos_bias_local_probe`; baseline artifact sha256 ec449a86484cf0d8b709b8a71a011ff379a7a3efe6751c2568e911f2a72dc9b7 and attempt artifact sha256 950af2eb4723834c9fdb777848b1b3a0777054bcab3d7dbdd52fac4bfa064b5e (SHA256SUMS)
  CPU feature string: arm64 Apple M4, dotprod=1, i8mm=1; SAM attention probe is f32 through the current frankentorch facade, no SIMD tier override
  exact command + env:
    FOCR_SAM_ATTN_PROBE_RUNS=7 CARGO_TARGET_DIR=target-codex-verify cargo +nightly --config patch.crates-io.asupersync.path='"/dp/asupersync"' test --profile release-perf sam_attention_relpos_bias_local_probe --lib -- --ignored --nocapture
  fallback / kill-switch state: no FOCR_* performance kill-switches set; allocator=system; harness calls the real `vision_sam::attention`
  measured before -> after vs reference:
    local focr-only synthetic SAM attention probe (no reference ratio): 17.53473214285714 ms -> 18.066375 ms average for 7 calls, a 3.03% slowdown; PERF_LEDGER ineligible until the gauntlet has a pinned CPU reference row
  bit-exact correctness proof:
    output checksum unchanged (`-0.013422159478068352` before and after); the local experiment added helper tests proving the slice-copy QKV split and precomputed grid coordinates matched the old indexing formulas, but the code was reverted before commit because the timing regressed
  disposition: REVERT
  do-not-retry: "do not retry QKV slice-copy splitting or key-coordinate hoisting as standalone SAM attention levers; revisit only inside a larger batched-QK/probs@V rewrite if profiling shows QKV split or coordinate math is a named hotspot"
  per-lever tally: W 0 / L 1 / N 0
  agent: WhiteCave
  evidence dir: artifacts/perf/bd-3n16/

2026-06-25 | WIN | per-head GEMM QK^T + probs@V in `src/native_engine/vision_sam.rs::attention`
  claim_id: CLAIM-bd-3n16-sam-gemm-attention   evidence_id: artifacts/perf/bd-3n16/
  model source commit + fixture hash:
    HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5
    config.json sha256 27246d03fd670904ec9601b1cb0861fbb79ec076830771daa8d943d6229946f9 (SOURCE_HASHES.md)
    synthetic SAM attention fixture: ignored unit probe `sam_attention_relpos_bias_local_probe`; baseline artifact `gemm_baseline_7run.txt`, attempt artifact `gemm_attempt_7run.txt`, and README hashes recorded in SHA256SUMS
  CPU feature string: arm64 Apple M4, dotprod=1, i8mm=1; SAM attention probe is f32 through the current frankentorch facade, no SIMD tier override
  exact command + env:
    FOCR_SAM_ATTN_PROBE_RUNS=7 CARGO_TARGET_DIR=target-codex-verify timeout 240s cargo +nightly --config patch.crates-io.asupersync.path='"/dp/asupersync"' test --profile release-perf sam_attention_relpos_bias_local_probe --lib -- --ignored --nocapture
  fallback / kill-switch state: no FOCR_* performance kill-switches set; allocator=system; harness calls the real `vision_sam::attention`
  measured before -> after vs reference:
    local focr-only synthetic SAM attention probe (no reference ratio): 18.075154714285716 ms -> 12.591648857142857 ms average for 7 calls, local speedup 1.435488x and 30.3373% less wall time; PERF_LEDGER ineligible until the gauntlet has a pinned CPU reference row
  bit-exact correctness proof:
    not bit-exact because the frankentorch GEMM changes f32 accumulation order; checksum drift stayed tiny over 7 calls (`-0.013422159478068352` -> `-0.013422181829810143`), and `attention_gemm_matches_scalar_reference_with_relpos` compares the GEMM path against the old scalar loop with non-zero rel-pos tables at `max_abs <= 2e-6`; `CARGO_TARGET_DIR=target-codex-verify timeout 240s cargo +nightly --config patch.crates-io.asupersync.path='"/dp/asupersync"' test vision_sam::tests --lib -- --nocapture` -> 18 passed, 0 failed, 1 ignored
  disposition: KEEP
  do-not-retry: "do not return this SAM attention stage to scalar per-query QK/probs@V loops unless a full L1/L2 parity gate or pinned gauntlet row proves the GEMM accumulation drift is unacceptable"
  per-lever tally: W 2 / L 1 / N 0
  agent: WhiteCave
  evidence dir: artifacts/perf/bd-3n16/

2026-06-26 | NEGATIVE(reverted) | int4 `lm_head` decode GEMV (`decoder::gemv_i4` + `FOCR_LMHEAD_INT4` packed-int4 cache via `simd::int4::igemm_s4s8`)
  claim_id: CLAIM-int4-lmhead-decode-gemv   evidence_id: artifacts/perf/bd-int4-lmhead/
  model source commit + fixture hash:
    HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5
    config.json sha256 27246d03fd670904ec9601b1cb0861fbb79ec076830771daa8d943d6229946f9 (SOURCE_HASHES.md)
    lm_head.weight from the real model-00001-of-000001.safetensors shard (work-dir `model/`), packed per-group int4 (g32 and g16) IN-PROCESS at cache build — no `.focrq` written for this throwaway experiment
  CPU feature string: arm64 Apple M4, aarch64+neon+dotprod (SDOT tier). int4 path = `simd::int4::igemm_s4s8`, which nibble-unpacks to an int8 buffer (`unpack_nibbles_neon`) then runs the int8 SDOT on it
  exact command + env:
    scratchpad/perfonly.sh on page_0023 (821 decode tokens), profiling on:
    env FOCR_DECODE_INT8=1 FOCR_LMHEAD_INT4=1 FOCR_PROFILE_DECODE=1 FOCR_TIMING=1 FOCR_STAGE_BUDGET_FORWARD_MS=7200000 focr ocr page_0023.png --model <work>/model
    (g16 row: FOCR_LMHEAD_INT4=16; baseline row: FOCR_LMHEAD_INT4 unset. RAYON pool = default physical cores; allocator=system)
  fallback / kill-switch state: FOCR_LMHEAD_INT4 unset → `lm_head_i4 = None` → int8 `gemv_i8` (the default, unaffected by this experiment). FOCR_DECODE_INT8=1 active throughout; no FOCR_FORCE_ARCH override (native SDOT tier)
  measured before -> after vs reference:
    local focr-only decode profile (no torch reference ratio yet): lm_head phase 2130 ms (int8) -> 12278 ms (int4 g32) / 12720 ms (int4 g16) = ~5.8x SLOWER; whole-decode 15.75 s / 0.019 s-tok -> 25.85 s / 0.031 (g32), 26.12 s / 0.032 (g16). PERF_LEDGER-ineligible (no pinned CPU reference row exists).
  bit-exact correctness proof:
    not run — REJECTED ON PERF ALONE (a 5.8x lm_head regression; no 20-page CER spent on a reverted path). int4 round-trip + GEMM correctness is independently covered by the bit-exact oracle tests in src/quant/int4.rs and src/simd/int4.rs; this entry is purely a throughput rejection.
  disposition: REVERT
  do-not-retry: "do not retry int4 for ANY decode GEMV (lm_head, experts, q/k/v/o) while `simd::int4::igemm_s4s8` unpacks nibbles to an int8 buffer in memory before the dot — that costs ~2.5x int8's memory traffic (read 0.5 B/wt packed + write 1 B/wt int8 + read 1 B/wt int8) and is strictly slower than int8 on this CPU decode. Retry ONLY after a NATIVE packed-int4 dot exists (nibbles consumed in-register straight into the SDOT/SMMLA MAC, NO int8 materialization). Even then the ceiling is tiny: lm_head is ~14% of decode (perfect int4 ≤ ~4% of a full page); experts 45% + attn 37% are overhead/dispatch-bound (~1-3 GB/s, far below bandwidth), where a weight-BYTES lever cannot help. The decode lever is dispatch/alloc-overhead reduction, not byte reduction. This single most-favorable (bandwidth-bound) tensor regressing closes the int4-EXPERTS arm of the blend too (same unpack kernel, less-favorable regime)."
  per-lever tally: W 0 / L 1 / N 0
  agent: opus-4.8 (int4/int8 blend sweep)
  evidence dir: artifacts/perf/bd-int4-lmhead/

2026-06-26 | NEGATIVE(reverted) | serial/quantize-once decode attn projections (`FOCR_ATTN_SERIAL` in `decoder::decode_step_with_cache_i8`)
  claim_id: CLAIM-attn-serial-qkv-projections   evidence_id: artifacts/perf/bd-attn-serial/
  model source commit + fixture hash:
    HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5
    config.json sha256 27246d03fd670904ec9601b1cb0861fbb79ec076830771daa8d943d6229946f9 (SOURCE_HASHES.md)
    q/k/v/o_proj.weight from the real model-00001-of-000001.safetensors shard (work-dir `model/`), per-output-channel symmetric int8 (`quant_oc`) — same weights as the default path
  CPU feature string: arm64 Apple M4, aarch64+neon+dotprod (SDOT tier). serial path = `gemv_i8_serial` (single `simd::igemm_s8s8` over all n, no rayon); default = `gemv_i8` (par_chunks_mut(64) → ~20 SDOT tasks)
  exact command + env:
    scratchpad/perfonly.sh on page_0023 (821 decode tokens), profiling on:
    env FOCR_DECODE_INT8=1 FOCR_ATTN_SERIAL=1 FOCR_PROFILE_DECODE=1 FOCR_TIMING=1 FOCR_STAGE_BUDGET_FORWARD_MS=7200000 focr ocr page_0023.png --model <work>/model
    (baseline row: FOCR_ATTN_SERIAL unset. RAYON pool = default physical cores; allocator=system)
  fallback / kill-switch state: FOCR_ATTN_SERIAL unset → default parallel `gemv_i8` q/k/v/o (unaffected). FOCR_DECODE_INT8=1 active; no FOCR_FORCE_ARCH (native SDOT)
  measured before -> after vs reference:
    local focr-only decode profile (no torch reference ratio yet): attn phase 5436 ms (default) -> 6456 ms (serial) = +19% REGRESSION; whole-decode 15.15 s / 0.018 s-tok -> 15.73 s / 0.019. The untouched lm_head/experts/route phases drifted +-370 ms run-to-run; the +1020 ms attn move is ~2.7x that noise band. PERF_LEDGER-ineligible (no pinned CPU reference row).
  bit-exact correctness proof:
    PROVABLY bit-identical to the default path — same `quantize_row_i8(nrow)` (shared across q/k/v) and the same EXACT i32 SDOT accumulation per output row regardless of the per-64-block split, so q/k/v/o outputs are byte-for-byte equal. No CER run needed; rejected on PERF alone.
  disposition: REVERT
  do-not-retry: "do not re-serialize the decode attn projections (q/k/v/o) — they are NOT dispatch-bound; the rayon block-parallel `gemv_i8` (1280-wide m=1 GEMV across ~20 cores) genuinely beats single-core serial SDOT, so serializing loses more to single-threading than it saves on dispatch + the one extra quantize. The attn-phase time is dominated by the f32 `rswa::decode_attention` (R-SWA over ~277 ref + 128 ring keys x 10 heads x 12 layers), NOT the int8 projections — that f32 attention is the real attn lever. Retry a projection change ONLY if it FUSES q/k/v into one wide [3*1280,1280] GEMV (fewer dispatches while KEEPING block-parallelism), never serial-single-core."
  per-lever tally: W 0 / L 1 / N 0
  agent: opus-4.8 (int4/int8 blend sweep)
  evidence dir: artifacts/perf/bd-attn-serial/

2026-07-07 | NEGATIVE(reverted) | ngram-ban folded into the int8 lm_head dequant epilogue (`decoder::fuse_ngram_lmhead` + `FOCR_FUSE_NGRAM_LMHEAD`, bd-2mo.24 / bd-1azu.54 Lever 3)
  claim_id: CLAIM-bd-2mo24-fuse-ngram-lmhead   evidence_id: artifacts/perf/bd-2mo.24/
  model source commit + fixture hash:
    HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5
    page_0023.png sha256 a74adb4f437d7955f5f75d3e4f053562c9b7e20cd54840d4488ac2f61ef3f761 (the 821-token ngram-heavy page); weights unlimited-ocr.int8.focrq (the pulled artifact, convert byte-parity vs published d8c5fcf2)
  CPU feature string: arm64 Apple M4, aarch64+neon+dotprod (SDOT tier)
  exact command + env:
    3+3 A/B, same load regime, back-to-back:
    env [FOCR_FUSE_NGRAM_LMHEAD=1] FOCR_TIMING=1 FOCR_THREADS=8 FOCR_DECODE_INT8=1 focr ocr page_0023.png --model ~/.cache/franken_ocr/models/unlimited-ocr.int8.focrq
    (baseline rows: flag unset -> sampler copy-then-mask after lm_head_cached_i8)
  fallback / kill-switch state: FOCR_FUSE_NGRAM_LMHEAD unset (the default, unchanged by this verdict) -> the separate mask pass; FOCR_QKV_FUSED default-ON (its independent WIN); FOCR_DECODE_INT8=1; no FOCR_FORCE_ARCH (native SDOT)
  measured before -> after vs reference:
    PERF (M4, within-regime 3x3): decode_i8 best 16.43 s / 0.020 s-tok (off) -> 16.40 s / 0.020 s-tok (on) = 0.2%, inside run-to-run noise (off spread 16.43-17.07 s). The eliminated copy-then-mask pass over the 129,280-logit row is sub-millisecond against a ~20 ms decode step; there is nothing material to save on this axis.
    ACCURACY: outputs byte-identical across all runs (on_1/2/3 == off_1) - the fusion is exact by construction, as its unit gate proves.
  bit-exact correctness proof:
    `fused_ngram_lmhead_is_byte_identical_to_separate_mask` (decoder.rs unit gate) + all six A/B outputs byte-identical on the ngram-heavy page.
  disposition: REVERT
  do-not-retry: "the lever is CORRECT but does not pay: the masked-id set is tiny and the mask pass is sub-ms vs a ~20 ms step. Leave the code in place behind the flag (harmless, tested); do not flip the default unless a workload materially changes the arithmetic - e.g. a much larger ban set (multi-image ngram_window=1024 with long repetitive histories) or a decode step an order of magnitude faster, in which case re-run THIS A/B on that workload first."
  per-lever tally: W 0 / L 0 / N 1
  agent: fable-5 (bd-2mo.24 verdict run)
  evidence dir: artifacts/perf/bd-2mo.24/

2026-06-27 | WIN | fused [3*1280,1280] q/k/v decode GEMV (`decoder::fuse_qkv` + `FOCR_QKV_FUSED`; one `simd::igemm_s8s8` over a stacked [3840,1280] weight instead of three [1280,1280] calls)
  claim_id: CLAIM-bd-1waa-qkv-fused-decode-gemv   evidence_id: artifacts/perf/bd-1waa/
  model source commit + fixture hash:
    HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5
    config.json sha256 27246d03fd670904ec9601b1cb0861fbb79ec076830771daa8d943d6229946f9 (SOURCE_HASHES.md)
    q/k/v_proj.weight from the real model-00001-of-000001.safetensors shard (work-dir `model/`), per-output-channel symmetric int8 (`quant_oc`); the fused cache stacks the three [1280,1280] weights into one [3840,1280] QInt8 with concatenated per-row scales — same bytes, same scales, one tensor
  CPU feature string: arm64 Apple M4, aarch64+neon+dotprod (SDOT tier); fused path = one block-parallel `simd::igemm_s8s8` over [3840,1280]
  exact command + env:
    scratchpad/bench_config.sh on page_0023 (821 decode tokens) for perf, 20-page ocr-batch for CER:
    env FOCR_DECODE_INT8=1 FOCR_QKV_FUSED=1 FOCR_PROFILE_DECODE=1 FOCR_TIMING=1 FOCR_STAGE_BUDGET_FORWARD_MS=7200000 focr ocr page_0023.png --model <work>/model
    (baseline row: FOCR_QKV_FUSED unset. RAYON pool = default physical cores; allocator=system)
  fallback / kill-switch state: FOCR_QKV_FUSED unset → `cl.qkv = None` → the original three separate `gemv_i8` q/k/v calls (the default). FOCR_DECODE_INT8=1 active; no FOCR_FORCE_ARCH (native SDOT)
  measured before -> after vs reference:
    M4 (aarch64+neon+dotprod): local focr-only decode profile (no torch reference ratio yet): whole-decode 16.55 s / 0.020 s-tok -> 15.08 s / 0.018 s-tok = ~8.9% faster decode; attn phase 5671 ms -> 4929 ms (the rest drifts run-to-run on the untouched phases).
    CROSS-ARCH (trj = AMD Threadripper PRO 5995WX, Zen3, x86_64+avx2, NO VNNI, 128 threads): whole-decode 233.68 s / 0.285 s-tok -> 184.66 s / 0.225 s-tok = ~21% faster decode — a BIGGER relative win than M4, because AVX2 has no native int8 dot so per-GEMV dispatch + activation-reload overhead is higher and collapsing 3 GEMVs→1 saves more (decode attn phase 83581 -> 59767 ms). PERF_LEDGER-ineligible (no pinned CPU reference row).
  bit-exact correctness proof:
    PROVABLY byte-identical to the three-call path — unit test `fused_qkv_gemv_is_byte_identical_to_three_calls` (per-output-row i32 SDOT accumulation is independent of whether q/k/v are one stacked tensor or three), AND end-to-end on the worst page: `focr ocr page_0590` (the longest, runaway-prone table page) with FOCR_QKV_FUSED=1 is byte-identical to scalar base — both 8756 chars, sha256 63c55f7da9fd9918bbb90acbb10384243d468f73edb675451aef2c6e344a20a1; 20-page content CER 0.2116 == base 0.2116
  disposition: KEEP
  do-not-retry: "this is the KEPT lossless attention-projection lever — exactly the fuse-q/k/v-into-one-wide-GEMV move the bd-attn-serial retry predicate called for (fewer dispatches while KEEPING block-parallelism). Do not replace it with serial-single-core qkv (bd-attn-serial: +19%) or int4 qkv (bd-int4-lmhead: unpack kernel, 5.8x). The next attention lever is the f32 `rswa::decode_attention` itself — but ONLY bit-exactly: the non-bit-exact GEMM/int8-KV variants in this same bead both degenerate on page_0590 (the two REVERT entries below)."
  per-lever tally: W 1 / L 0 / N 0
  agent: opus-4.8 (decode-attention lever sweep)
  evidence dir: artifacts/perf/bd-1waa/

2026-06-27 | NEGATIVE(reverted) | batched per-head GEMM decode attention (`rswa::decode_attention_gemm` + `FOCR_ATTN_GEMM`: QK^T / softmax / probs@V as blocked GEMMs)
  claim_id: CLAIM-bd-1waa-gemm-decode-attention   evidence_id: artifacts/perf/bd-1waa/
  model source commit + fixture hash:
    HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5
    config.json sha256 27246d03fd670904ec9601b1cb0861fbb79ec076830771daa8d943d6229946f9 (SOURCE_HASHES.md)
    R-SWA decode attention over the live KV ring (≤277 ref + 128 window keys × 10 heads × 12 layers); f32 throughout, no weight tensor (operates on cached K/V activations)
  CPU feature string: arm64 Apple M4, aarch64+neon+dotprod (SDOT tier); GEMM path = frankentorch f32 GEMM for per-head QK^T and probs@V (reorders the f32 accumulation vs the scalar per-key loop)
  exact command + env:
    scratchpad/bench_config.sh / CER gate on page_0023 (perf) + 20-page ocr-batch (CER):
    env FOCR_DECODE_INT8=1 FOCR_ATTN_GEMM=1 [FOCR_QKV_FUSED=1] FOCR_PROFILE_DECODE=1 FOCR_TIMING=1 FOCR_STAGE_BUDGET_FORWARD_MS=7200000 focr ocr page_0023.png --model <work>/model
    (baseline row: FOCR_ATTN_GEMM unset → scalar. RAYON pool = default physical cores; allocator=system)
  fallback / kill-switch state: FOCR_ATTN_GEMM unset → `decode_attention_scalar` (the default bit-exact per-key loop). FOCR_INT8_KV unset. FOCR_DECODE_INT8=1 active; no FOCR_FORCE_ARCH (native SDOT)
  measured before -> after vs reference:
    PERF (M4): whole-decode 16.55 s / 0.020 s-tok -> 14.60 s (gemm) / 14.08 s (gemm+qkv) / ~0.018 s-tok; attn phase 5671 ms -> 4970 (gemm) / 4341 (gemm+qkv) ms — a real ~12–15% decode win.
    PERF (x86 trj, Zen3/avx2): the GEMM attention does NOT pay on AVX2 — gemm 211.45 s (−9.5% vs base) and gemm+qkv 230.68 s (only −1.3%, i.e. WORSE than the lossless qkv-alone's 184.66 s / −21%). The frankentorch f32 batched GEMM is not well-served on AVX2; on x86 the GEMM attention actively regresses the qkv win.
    ACCURACY: 20-page content CER 0.2116 -> 1.3030 — CATASTROPHIC, and ENTIRELY one page: 19/20 stay bit-near-exact (4 byte-exact, rest CER < 0.04), but page_0590 (the longest, a repetitive ship-loss TABLE) RUNS AWAY 8755 -> 91243 chars (CER 4.23 that page; gemmqkv_perpage_cer.txt). Rejected on ACCURACY regardless of the perf win; PERF_LEDGER-ineligible (no pinned CPU reference row).
  bit-exact correctness proof:
    NOT bit-exact by construction (the batched GEMM reorders f32 accumulation in QK^T / softmax / probs@V). The drift is tiny per-token but on page_0590 it tips the autoregressive sampler past the EOS-emission tipping point into a degenerate `<tr><td>..Hornet..David Comin..</td></tr>` row-repeat that never terminates (page_0590_runaway_tail.txt). This is the SAME f32-reorder as the SAM vision GEMM (bd-3n16, KEPT) — harmless there because vision attention feeds a projector; disqualifying here because decode attention feeds a token sampler whose EOS timing is fragile on long repetitive pages.
  disposition: REVERT
  do-not-retry: "do not enable FOCR_ATTN_GEMM (non-bit-exact decode attention) unless decode is first made robust to f32-accumulation drift on long repetitive pages — e.g. a bit-exact blocked GEMM that matches the scalar per-key accumulation order, OR a repetition/no-EOS stop guard in the sampler (a semantics change that must clear its own 20-page CER gate). The ~12–15% decode win is real but is already captured LOSSLESSLY by FOCR_QKV_FUSED (the WIN above); the marginal extra (~3–6%) is not worth a page-killing runaway. Bit-exactness is mandatory for decode attention, optional for vision attention."
  per-lever tally: W 0 / L 1 / N 0
  agent: opus-4.8 (decode-attention lever sweep)
  evidence dir: artifacts/perf/bd-1waa/

2026-06-27 | NEGATIVE(reverted) | int8-quantized KV cache + int8 QK decode attention (`rswa::decode_attention_int8` + `FOCR_INT8_KV`)
  claim_id: CLAIM-bd-1waa-int8-kv-decode-attention   evidence_id: artifacts/perf/bd-1waa/
  model source commit + fixture hash:
    HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5
    config.json sha256 27246d03fd670904ec9601b1cb0861fbb79ec076830771daa8d943d6229946f9 (SOURCE_HASHES.md)
    R-SWA decode KV ring quantized per-row to int8 (K and V); int8 QK via `simd::igemm_s8s8` (i32 accum, 127²·128 = 2.06M per dot < i32::MAX), f32 softmax, int8 V dequant per row; ≤277 ref + 128 window keys × 10 heads × 12 layers
  CPU feature string: arm64 Apple M4, aarch64+neon+dotprod (SDOT tier); int8 QK = `simd::igemm_s8s8`
  exact command + env:
    scratchpad/bench_config.sh / CER gate on page_0023 (perf) + 20-page ocr-batch (CER):
    env FOCR_DECODE_INT8=1 FOCR_INT8_KV=1 FOCR_PROFILE_DECODE=1 FOCR_TIMING=1 FOCR_STAGE_BUDGET_FORWARD_MS=7200000 focr ocr page_0023.png --model <work>/model
    (baseline row: FOCR_INT8_KV unset → scalar f32 KV. RAYON pool = default physical cores; allocator=system)
  fallback / kill-switch state: FOCR_INT8_KV unset → `decode_attention_scalar` (default f32 KV). FOCR_ATTN_GEMM unset. FOCR_DECODE_INT8=1 active; no FOCR_FORCE_ARCH (native SDOT)
  measured before -> after vs reference:
    PERF (M4): whole-decode 16.55 s / 0.020 s-tok -> 13.97 s / 0.017 s-tok — the FASTEST variant on M4 (~15% decode win; int8 KV halves KV bandwidth + uses SDOT QK).
    PERF (x86 trj, Zen3/avx2): does NOT carry to x86 — int8kv decode 230.51 s / 0.281 s-tok = only −1.3% vs base, far behind the lossless qkv-alone (184.66 s / −21%). The int8-KV dequant + AVX2-emulated int8 QK overhead eats the bandwidth saving; the int8-KV speedup is an M4-only effect.
    ACCURACY: 20-page content CER 0.2116 -> 0.3277 (+55% relative). Not a 1.30 blow-up like FOCR_ATTN_GEMM, but the SAME root cause on the SAME page: int8 KV is lossier than the f32-GEMM reorder, so page_0590 degenerates into the same no-EOS runaway; under the CER-gate forward budget that page's int8-KV decode does not finish -> empty output -> CER 1.0 on that page, which drives the entire +0.116 aggregate (an unbudgeted single-page int8-KV run was still grinding the page_0590 runaway at 7+ min when killed). The other 19 pages are unaffected. PERF_LEDGER-ineligible.
  bit-exact correctness proof:
    NOT bit-exact by construction (int8 quantization of the KV cache is lossy). Same EOS-fragility failure as FOCR_ATTN_GEMM on page_0590, manifesting as a budget-timeout empty page rather than a 91k runaway because the int8-KV decode is slower per runaway token. int8 QK i32-accumulation exactness (no overflow at head_dim=128) is independently covered by the `simd::igemm_s8s8` oracle tests; this entry is an end-to-end ACCURACY rejection.
  disposition: REVERT
  do-not-retry: "do not enable FOCR_INT8_KV unless decode is first made robust to long-page degeneration (same predicate as FOCR_ATTN_GEMM) AND the int8-KV 20-page CER is shown within budget vs base 0.2116. Even then the ceiling is small: int8-KV's only marginal gain over the LOSSLESS FOCR_QKV_FUSED is ~6% decode, not worth a lossy KV cache that fails the hardest page. The attention here (≤405 keys × 10 heads) is not the decode bottleneck — experts (~44%) + the f32 element-wise attention overhead are; a KV-bytes lever cannot move them."
  per-lever tally: W 0 / L 1 / N 0
  agent: opus-4.8 (decode-attention lever sweep)
  evidence dir: artifacts/perf/bd-1waa/

2026-07-09 | NEGATIVE(reverted) | on-the-fly BF16 widening for conservative decode q/k/v/o projections and `lm_head` (`src/native_engine/decoder.rs`)
  claim_id: CLAIM-bd-2mo30-bf16-streaming-decode   evidence_id: artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-bf16-stream/
  model source commit + fixture hash:
    HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5
    config.json sha256 27246d03fd670904ec9601b1cb0861fbb79ec076830771daa8d943d6229946f9 (SOURCE_HASHES.md)
    model-00001-of-000001.safetensors sha256 2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6
    conservative recipe `.focrq` sha256 573340710167697891bf52dfa4cbb5d0a02a68f3011c01f8ef83fd34622fb592 (4,157,448,783 bytes; 2,148 FFN/expert QInt8 tensors; attention and lm_head BF16)
    page_0014.png sha256 f1e35a58f673036b16302c86676f1ec6a89218fb4e666c477dbb4ae21b904df2 (495 generated tokens)
    source baseline HEAD 58cf4e196e787fff8a2e83b2d5478541c64a3ee4 plus the bd-2mo.30 conservative-recipe worktree
  CPU feature string: Apple M4 arm64, 10 physical cores; high-precision GEMV uses the scalar eight-lane reduction that LLVM autovectorizes (no FOCR_FORCE_ARCH override)
  exact command + env:
    six adjacent runs in order `FOCR_BF16_STREAM=0,1,1,0,0,1`, each:
    `FOCR_BF16_STREAM=<mode> FOCR_THREADS=8 FOCR_TIMING=1 FOCR_PROFILE_DECODE=1 FOCR_STAGE_BUDGET_FORWARD_MS=7200000 /usr/bin/time -p /Volumes/USBNVME16TB/temp_agent_space/cargo-target/release/focr ocr /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/pages/page_0014.png --model /Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model/unlimited-ocr.recipe-v1.20260709T2304.focrq`
  fallback / kill-switch state: `FOCR_BF16_STREAM=0` eagerly widens high-precision decoder weights once into the mixed cache; `=1` retains source BF16 words and widens exactly during each dot product. FOCR_DECODE_INT8/FOCR_INT8_ATTN/FOCR_INT8_LMHEAD unset; allocator=system; conservative FFN-only int8 recipe active.
  measured before -> after vs reference:
    local focr-only interleaved A/B (no pinned torch CPU reference ratio): eager-f32 decode 14.05/13.94/13.89 s, median 13.94 s -> streamed-BF16 14.47/14.61/14.58 s, median 14.58 s = 4.59% slower (0.956x throughput). Median lm_head phase 4,783 -> 5,139 ms (+7.44%); attention phase 5,592 -> 5,876 ms (+5.08%). Untouched expert/route phases remained stable. Whole-page real time median 19.69 -> 20.34 s (+3.30%). Score is negative despite reducing the high-precision cache by about 156 MB.
  bit-exact correctness proof:
    the experimental unit gate compared every f32 output bit against eager widening under the same eight-lane reduction; all six end-to-end Markdown outputs are byte-identical, sha256 7190e15dcbe5a85caf1fc61d2ac27aa2fc997e841965aceb0566cc39c8e13d0a. Correctness passed, performance failed.
  disposition: REVERT (implementation and kill switch removed; evidence retained)
  do-not-retry: "do not stream BF16 through scalar per-element widening in decode GEMVs. Retry only with a native BF16 matrix/dot kernel that consumes packed BF16 and amortizes conversion while preserving the accepted decode accuracy contract; a memory-only argument is insufficient."
  per-lever tally: W 0 / L 1 / N 0
  agent: BrownFox
  evidence dir: artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-bf16-stream/

The first real entry MUST carry **full truth-pack provenance** (model commit
`3a7f4db…` + `(file_sha256, lines)` from `SOURCE_HASHES.md` + weights/`.focrq`
hash) and a paired `artifacts/perf/<bead>/` evidence dir. Shape to follow (a
**template**, not a measurement — note the empty number fields):

The evidence dir must include a hash manifest named `SHA256SUMS`,
`SHA256SUMS.txt`, `sha256sums.txt`, `sha256.txt`, `manifest.sha256`, or
`manifest.json`; `scripts/check_ledgers.py` rejects real entries whose evidence
dirs exist but are not hash-anchored.

```
2026-MM-DD | <WIN|NEGATIVE(reverted)> | <lever, file:fn>
  claim_id: <e.g. CLAIM-int8-expert-ffn-decode>   evidence_id: artifacts/perf/<bead>/
  model source commit + fixture hash:
    HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5
    config.json sha256 27246d03…  (SOURCE_HASHES.md)
    model-00001-of-000001.safetensors sha256 <recorded-when-fetched>
    <model>.focrq sha256 <conversion hash for the precision measured>
  CPU feature string: <dispatched tier, e.g. aarch64+neon+dotprod or aarch64+neon+i8mm>
  exact command + env:
    cargo bench -p focr --bench gauntlet -- decode-per-token
    FOCR_REFERENCE_BACKEND=<onnx|hf|gguf>  OMP_NUM_THREADS=8  RAYON_NUM_THREADS=8
    (reference torch set_num_threads(8) — NEVER @64, §9.3)
  fallback / kill-switch state: FOCR_INT8_ATTN=<0|1>  FOCR_INT8_LMHEAD=<0|1>
    int4-group=<off|g32|g16>  allocator=<system|mimalloc-feature>
  measured before -> after vs reference: <ref_ms> / <focr_ms> -> ratio <x.xx>  (or "blocked: <why>")
  bit-exact correctness proof: <test name> -> <pass|CER Δ within AF-2 budget|4-ULP table>
  disposition: <KEEP|REVERT>
  do-not-retry: "do not retry <X> unless <Y>"
  per-lever tally: W <n> / L <n> / N <n>
  agent: <pane/agent id>
```
