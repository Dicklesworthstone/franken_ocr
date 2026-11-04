# bd-zex4: thread-width sweep (2026-11-01)

The question was whether `default_thread_budget()` (36 on this host, every
physical core across both Apple tiers) is the right default width for the kernel
pool.

## Setup

- Host: Apple M5 Ultra, `hw.perflevel0.physicalcpu` = 12 cores of the fastest
  tier, `hw.perflevel1.physicalcpu` = 24 of the slower one (`host.txt`). macOS
  names them Super and Performance (`hw.perflevel0.name`, `hw.perflevel1.name`,
  read 2026-11-02), so "performance cores" here would mean the slower 24.
- Binary: the `release-perf` focr that ran U2-C5a, sha256 `b6c5b571…`.
- Weights: the int8 artifacts that ran U2-C5a:
  - `unlimited-ocr.int8.focrq` sha256 `573340710167…fb592`;
  - `unlimited-ocr-2.v0.1.0.int8.focrq` sha256 `54b8a1ecaf0c…4b4398`.
- Pages: corpus page_0014, page_0023 and page_0590, single-page `focr ocr`, cold
  process each run, `FOCR_NO_RESIDENT=1`.
- Arms: the incumbent default (`FOCR_THREADS` unset, which resolves to 36) and
  `FOCR_THREADS` 8, 12, 18, 24 and 36.
  - Three passes, with the arm order rotated each pass.
  - A fourth pass ran 1, 2 and 4 threads once each, as the N = 1 anchor for the
    AF-5 USL fit (`scripts/af5_usl_fit.py`).
- Each run recorded `/usr/bin/time -l` real, user, sys, max RSS and peak
  footprint, plus the output's sha256 (`runs.tsv`). `zex4_sweep.sh` and
  `zex4_summarize.py` are the scripts as run. `summary.md` is the summarizer's
  output, and `usl_*.json` are its fit inputs.

## The host was not quiet

The bead asked for a quiet host, and it did not get one. Other tenants'
processes kept running throughout: about ten single-threaded fuzzers, browsers
and other agents.
- Load average was 28.9 / 38.5 / 39.9 at the start (`host.txt`).
- Over the sampled last half hour, load1 was 26.9–71.7, and the other tenants
  used about 15 cores' worth of CPU on average (`load_samples.log`, one line a
  minute from 15:40).
- Runs drifted slower as that load rose. Of the 36 best-of-3 picks (model × arm
  × page), 23 come from pass 1, 13 from pass 2 and none from pass 3. The rotated
  order spreads the drift over all arms.

So every comparison here is paired, against the incumbent in the same
invocation, under this host's real shared load. **How large the gap is on an idle
host was not measured.** `quiet_ab.sh` (session scratchpad) was to measure it:
page_0023, default vs 12, both models, once load1 and load5 were both below 6. It
waited from 2026-11-01 19:57, found no such moment in 24 h, and gave up at
2026-11-02 19:59 without running (its log line, in the session scratchpad's
`c5a_concurrent.log`).

## Result

Best of 3 per page, summed over the three pages:

| model | default (36) | 12 threads | wall ratio | CPU s at 36 | CPU s at 12 | sys share at 36 / 12 |
|---|---|---|---|---|---|---|
| v1 (unlimited-ocr) | 245.3 s | 93.8 s | 2.62× | 5,397 | 860 | 81% / 35% |
| v2 (unlimited-ocr-2) | 157.7 s | 64.7 s | 2.44× | 3,347 | 539 | 80% / 35% |

- **Fastest width:** 12 is the fastest arm for both models. 8 and 18 are within
  8% of it, and 24 is 17–19% slower.
  - Explicit 36 matches the default to within 1–4%, as it must, since they are
    the same width.
  - The default is slower than 4 threads, which took 174 s for v1 and 105 s
    for v2.
- **Output bytes:** identical for every page across all 126 runs, every width
  included. Width is a pure performance lever here, not a numerics change.
- **AF-5 USL fit:** for both models, alpha 0.021 / 0.053 and beta 0.0085 / 0.0086
  give a peak near 11 threads (R² 0.91 / 0.92, flagged noisy). Decision:
  `cap-at-usl-peak`, 11.
- **Kernel time:** sys time grows with width, from 4% at 1 thread to 81% at 36.
  That fits the fork-join reading in the bead: equal `par_chunks_mut` shares mean
  the slow cores set the pace while the rest spin and yield in the kernel.

## Decision

The default follows the fastest tier, `hw.perflevel0.physicalcpu` (12 here),
on any Mac that reports tiers. The bead's criterion was "not slower on wall
time". On this host, at this load, 12 threads took about 0.4× the wall and
one-sixth the CPU time, for the same bytes. iOS keeps its own rule, because no
on-device measurement exists yet (bd-v7ry). The change is
`default_thread_budget()` in `src/lib.rs`, which landed in commit 5b4bbcdc.

Five v2 runs from pass 2 (page_0014 at 36 and at the default; page_0023 at 8,
12 and 18) overlapped a one-thread scoring script of this session for about two
minutes. None of them is a best-of-3 pick.
