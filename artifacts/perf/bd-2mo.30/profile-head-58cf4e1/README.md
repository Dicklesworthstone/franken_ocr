# bd-2mo.30 Current-HEAD Profile

Status: **profiling in progress; timing publication is not yet eligible**.

This bundle applies `profiling-software-performance` to the post-autovec
`v0.6.0` tree at `58cf4e196e787fff8a2e83b2d5478541c64a3ee4`. It is the prerequisite
for the serial `extreme-software-optimization` loop in
`.skill-loop-progress.md`.

## Scenarios

| Regime | Fixture | Expected output | Primary metric | Budget |
|---|---|---|---|---|
| sparse page | `page_0009.png` | `golden_sparse.md` SHA-256 `7125f593...934` | end-to-end and vision p95 | no regression; rank the dominant fixed page cost |
| dense page | `page_0014.png` | `golden_dense.md` SHA-256 `7190e15d...d0a` | decode ms/token and end-to-end p95 | no regression; rank per-token costs |
| 10-page document | pending | byte-stable multi-page output | pages/s, ms/token, peak RSS | bounded memory and no oversubscription |
| 20-page corpus | pending (20 real pages are present; `._*` files are metadata) | byte-stable multi-page output | pages/s, ms/token, peak RSS | bounded memory and no oversubscription |

All current runs use the system allocator, eight threads, the published int8
artifact, greedy decode, and the same current-source `release-perf` binary.
Post-profile audit found two attribution constraints: profiled HEAD bypassed the
mmap loader and retained an owned model buffer, and the artifact quantizes
attention plus `lm_head` beyond the frozen default recipe. These numbers describe
that experimental full-int8 path, not the release-eligible conservative baseline.
The nominal SIMD tier also reported `sdot` while the effective dense IGEMM route
was the sampled scalar-autovec loop. `fingerprint.json` records the exact hashes
and these corrected loader/dispatch observations.

## Preliminary Stage Attribution

These single-run values are useful for attribution but **not** publishable
latency baselines: the initial load average was 7.02 and Spotlight was consuming
about one core. The required 20-run quiet-window distributions remain open.

| Stage | Sparse | Dense |
|---|---:|---:|
| end to end | 7.79 s | 13.43 s |
| preprocess | 0.03 s | 0.02 s |
| vision tower | 4.97 s | 4.79 s |
| decoder cache build | 0.18 s | 0.17 s |
| prefill, 277 tokens | 0.55 s | 0.53 s |
| decode | 1.45 s / 98 tokens | 7.34 s / 495 tokens |
| decode per token | 14.8 ms | 14.8 ms |
| peak footprint | 9.29 GB | 9.31 GB |

Within dense decode, the instrumented phase sum was 7.073 s: experts 2.957 s
(41.8%), attention 2.580 s (36.5%), lm-head 1.058 s (15.0%), and routing
0.478 s (6.8%). Sparse decode produced the same ordering.

SAM dominates the fixed page cost: `vision.sam` was 3.76-3.93 s. The twelve
SAM MLP calls sum to roughly 1.6 s, four global-attention calls to roughly
1.3 s, and eight window-attention calls to roughly 0.7 s.

## Profiler Evidence

- `dense-cpu.json.gz`: valid `samply` profile, 1 ms interval, current binary.
- `dense-sample.txt`: independent 15-second macOS `sample` capture with line
  symbols and an aggregated top-of-stack table.
- `raw/smoke_{sparse,dense}.stderr`: nested stage timings, decode buckets, and
  `/usr/bin/time -lp` resource counters.
- `dense-alloc.trace`: incomplete. Instruments suspended the target behind the
  macOS privacy/security handoff and never started recording; the profiling
  processes were terminated after four minutes. No allocation conclusion is
  drawn from this file.

The independent sample names the autovectorized scalar `igemm_s8s8` loop as the
largest active leaf (17,847 samples aggregated across workers), followed by
R-SWA `decode_attention` (1,236) and frankentorch dynamic-int8 linear work
(1,213). It also shows substantial condition-variable/yield samples, which
supports measuring Rayon scheduling and pool size but does not by itself prove
oversubscription.

Warm runs reported zero block input/output operations. That rejects disk I/O as
the steady-state inference bottleneck for these two scenarios, but they used an
owned model buffer because production `OcrModel::load` bypassed the mmap-capable
loader at this HEAD. Cold startup, owned-vs-mapped RSS, and artifact installation
remain separate regimes.

## Exact Commands

```text
RUSTFLAGS='-C force-frame-pointers=yes' CARGO_TARGET_DIR=/private/tmp/cc_tgt_dev rch exec -- cargo build --profile release-perf --bin focr

FOCR_DECODE_INT8=1 FOCR_TIMING=1 FOCR_PROFILE_DECODE=1 FOCR_THREADS=8 RAYON_NUM_THREADS=8 OMP_NUM_THREADS=8 /usr/bin/time -lp /private/tmp/cc_tgt_dev/release-perf/focr ocr PAGE --model /Users/jemanuel/.cache/franken_ocr/models/unlimited-ocr.int8.focrq -o GOLDEN

FOCR_DECODE_INT8=1 FOCR_TIMING=1 FOCR_PROFILE_DECODE=1 FOCR_THREADS=8 RAYON_NUM_THREADS=8 OMP_NUM_THREADS=8 samply record --save-only -o dense-cpu.json.gz -- /private/tmp/cc_tgt_dev/release-perf/focr ocr page_0014.png --model unlimited-ocr.int8.focrq
```

RCH reported zero healthy workers and ran the build locally. CASS searches for
prior sessions did not return within 20 seconds and were interrupted; the
committed ledgers and artifact bundles were used instead.

## Remaining Profiling Gates

1. Capture 20-run quiet-window sparse and dense baselines with p50/p95/p99,
   throughput, RSS, and variance.
2. Capture explicit 10-page and 20-page multi-page/batch regimes.
3. Obtain allocation evidence after macOS Instruments permission is available,
   or use an equivalent allocator-native profiler.
4. Reconcile CPU sampling, stage instrumentation, and roofline distance into
   the final ranked handoff before touching optimization source.
5. Re-run after the mmap production-path fix and after the conservative quant
   recipe is actually enforced; do not promote this full-int8 run as baseline.
