# U2-D3 memory calibration

Evidence for bead `bd-uocr2-epic-cgux.26` and the `U2-D3-*` rows of
`docs/PERF_LEDGER.md`. The question: does the whole-process estimate behind
`--page-window auto` bound what a real run uses?

`prereg.md` and `page_sets.md` were posted to the bead before the first
calibration run. Nothing in them changed afterwards.

**Rule.** Measured peak RSS is at most the estimate × 1.10 on every calibration
run. A miss means fixing the model and re-running every row, not widening the
margin.

**Result: all three pass, with no change to the model.**

| Run | Pages | Effective window | Estimate (bytes) | Peak RSS (bytes) | RSS ÷ estimate | Peak footprint (bytes) | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cal3` | 3 | 3 | 10,548,059,341 | 10,050,764,800 | 0.953 | 10,045,120,384 | 176.02 |
| `cal20` | 20 | 10 | 11,059,241,645 | 11,133,009,920 | 1.007 | 11,127,874,416 | 2,434.39 |
| `cal36` | 36 | 10 | 11,059,241,645 | 10,876,928,000 | 0.984 | 11,133,019,016 | 5,442.83 |
| `mapped20` (extra, not calibration) | 20 | 10 | 11,059,241,645 | 10,755,047,424 | 0.972 | 6,643,540,776 | 4,850.30 |

Peak RSS is `maximum resident set size` from `/usr/bin/time -l`. On every run it
equals the `peak RSS` the run's own `memory:` trace row reports, to the byte.
`cal36`'s footprint is above its RSS: RSS can undercount once the system
compresses pages. That footprint is also inside the 1.10 bound.

Each run read every page, skipped none and recomputed no vision fallback. No page
came back empty.

## How the runs were made

- Each run is a fresh process, because the counter is a high-water mark.
- Binary: `release-perf` with `CARGO_PROFILE_RELEASE_PERF_LTO=false` and
  `CARGO_PROFILE_RELEASE_PERF_CODEGEN_UNITS=16`, so not the ledger's protocol
  profile. It was built on 2026-10-31 at 02:37 from the working tree over commit
  `70c58299`.
- Artifact: `unlimited-ocr-2.v0.1.0.int8.focrq`, sha256
  `54b8a1ecaf0c1311eab5bb16cf224bca67772056e1d713d5538b5486035b4398`.
- Default thread budget (36), 1,000 output tokens per page assumed,
  `FOCR_NO_RESIDENT=1 FOCR_TIMING=1`, run under `taskpolicy -b`.
- The host was busy with the U2-C5a evaluation throughout, load average 120 or
  more. Peak memory does not need a quiet host. **The wall times here are not speed
  measurements**, and no speed claim rests on them.
- The estimator's three fitted constants (90 MB process, 2.6 GB vision scratch,
  0.6 GB decode scratch) came from exploratory runs on a different document, so
  none of these pages informed it.

## Mapped weights: `mapped20`

The same 20 pages as `cal20` with `FOCR_MMAP=1`. It was run after calibration, for
the M3 exit, and is labelled as an extra.

- **Output:** the JSON file is byte-identical to `cal20`'s (sha256 `30ec1b0d…` for
  both; `cmp` exits 0).
- **Peak footprint:** 6,643,540,776 bytes against 11,127,874,416, which is 40.3%
  smaller.
- **Peak RSS:** 3.4% smaller. Mapped file pages still count towards RSS, and on a
  256 GiB host nothing pushes them out.
- **Time, one unpaired run each on a loaded host:** wall 1.99×, CPU time 1.16×,
  page faults 3 → 3,348,348, decode 0.232–0.276 s per token against 0.126–0.135.
  This shows a cost worth measuring. It does not establish its size.

## The M3 exit

M3 asks for peak RSS ≤ 8 GiB at W=10 on a run of at least 300 pages on a 16 GB
laptop-class host. U2-T1b judges it, and this bead does not. The evidence so far:

| Configuration | Peak RSS | Peak footprint | Against 8 GiB |
|---|---:|---:|---|
| Owned weights, W=10, 20 pages | 10.37 GiB | 10.36 GiB | **Miss** on both |
| `FOCR_MMAP=1`, W=10, 20 pages | 10.02 GiB | 6.19 GiB | **Miss** on RSS, within on footprint |

The footprint is the memory only this process can hold. The rest of the mapped RSS
is clean file pages the kernel can drop, but this host never needed to, so a run
here cannot show what RSS does on a 16 GB host.

## Files

- `prereg.md`, `page_sets.md`: posted before the first run.
- `cal3.err`, `cal20.err`, `cal36.err`, `mapped20.err`: each run's stderr, with the
  timing trace and `/usr/bin/time -l` output.
- `outputs.sha256`: SHA-256 of each run's output JSON. The JSON itself is left out
  because it holds the text of the evaluation pages.
- `SHA256SUMS`: the manifest of this directory.
