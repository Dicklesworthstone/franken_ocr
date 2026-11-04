Calibration pre-registration, posted before any calibration run (2026-10-31).

**Rule.** The estimator is conservative when, on every calibration run, measured peak RSS is at most the estimate × 1.10, so the margin is 10%. A run over that bound is a failure of the estimator. The response is to fix the model and re-run every row, not to widen the margin. The margin does not change after the first run.

**Over-estimation is reported too.** It has no pass threshold, but `auto` picks a smaller window than it needs when the estimate runs high, so each row records measured ÷ estimate.

**Runs.** Unlimited-OCR-2 (`unlimited-ocr-2.v0.1.0.int8.focrq`, sha256 54b8a1ec…), one run each for 3, 20 and at least 35 pages, at W=10 with the default thread budget. Each run is a fresh process, because `ru_maxrss` is a high-water mark. Page sets are fixed before running, in name order, with no page swapped out after a result.

**What counts as measured.** The process high-water mark, from `/usr/bin/time -l` (`maximum resident set size`) for the whole run, cross-checked against the last window's `rss_bytes_peak`. They are one kernel counter read at different moments, so a material gap means a bug.

**Output tokens.** The estimate takes p90 tokens per page as an explicit assumption, 1,000 until measured. A calibration run must state the value it used, and that value is fixed before the run.

**Concurrent load.** Peak RSS is not a timing measurement, so these runs do not need a quiet host. Any overlap with the C5a evaluation goes in the row's notes. Thread width does change the peak, so if bd-zex4 changes the default thread budget, the rows are re-measured at the new default.

**The M3 exit.** ≤ 8 GiB at W=10 on a run of at least 300 pages is judged by U2-T1b. Calibration here does not decide it, and a miss is reported as a miss. Evidence so far, from before this bead: one page already peaks at 7.93 GiB, so that exit is at risk.

**Losing rows** go in `docs/PERF_LEDGER.md` with the rest.
