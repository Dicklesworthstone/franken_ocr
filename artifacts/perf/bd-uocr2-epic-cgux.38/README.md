# U2-F1: v2 against v1 in the same invocation (bd-uocr2-epic-cgux.38)

Measured 2026-11-01 from 16:12 to 16:55 (−0500), on the pinned `release-perf` binary (sha256
`b6c5b571…`, in `host.txt`).

**Settings.** Every run used `FOCR_THREADS=12` (the fastest width in the bd-zex4 sweep),
`FOCR_NO_RESIDENT=1` and `FOCR_TIMING=1`, under `/usr/bin/time -l`.

**Artifacts.** These are the two conservative int8 artifacts that U2-C5a ran:
- v2: `unlimited-ocr-2.v0.1.0.int8.focrq` (U2-C3/C4, sha256 `54b8a1ec…`, 4,173,206,653 bytes)
- v1: `unlimited-ocr.int8.focrq` (sha256 `57334071…`, 4,157,448,783 bytes)

Full hashes are in the PERF_LEDGER rows.

**Cells** (from `u2f1_bench.sh`):
- **Single pages.** S1 pages page_0009, page_0014 and page_0590 in base mode, 3 reps per model. The model order alternates each rep.
- **Documents.** Corpus pages strung together: `s2` (pages 9, 14 and 19), `p10` and `p20`. Page hashes are in `pages.sha256`.
  - Each document ran once per arm: v2 `--multi-page` (document mode), v1 `--multi-page`, and v1 per page.
  - v2 `p20` ran windowed, with 10 own pages per window (the auto default). `s2` and `p10` ran unwindowed.

**Host.** Shared. Load average was 26.9–59.2, and other tenants used 13–21 of the 36 cores.
`load_samples.log` has one sample a minute: time, load 1/5/15, and the top 15 non-focr processes' CPU %.
Only the pairing means anything, and each document cell is a single run.

## Stage table

Single pages show the best of 3 reps; documents show their one run.

`u2f1_summarize.py <this dir>` rebuilds `summary.json` and this table from `runs.tsv` and the `*.time` logs:
- vision = `vision.sam` + `vision.clip` + `vision.bridge`
- prefill = `prefill` rows, plus v2's `window.recompute` / `window.splice` rows
- decode = `decode` rows

| cell | input | model | mode | reps | wall s | s/page | RSS GiB | vision s | prefill s (tokens) | decode s (tokens) | s/token | chars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single | page_0009 | unlimited-ocr | base | 3 | 5.54 | 5.54 | 9.09 | 2.25 | 0.61 (277) | 1.64 (98) | 0.0167 | 106 |
| single | page_0009 | unlimited-ocr-2 | base | 3 | 4.90 | 4.90 | 9.09 | 2.21 | 0.75 (279) | 1.05 (50) | 0.0210 | 89 |
| single | page_0014 | unlimited-ocr | base | 3 | 12.12 | 12.12 | 9.07 | 2.19 | 0.69 (277) | 8.30 (495) | 0.0168 | 530 |
| single | page_0014 | unlimited-ocr-2 | base | 3 | 12.65 | 12.65 | 9.08 | 2.22 | 0.75 (279) | 8.76 (441) | 0.0199 | 525 |
| single | page_0590 | unlimited-ocr | base | 3 | 79.15 | 79.15 | 9.07 | 2.62 | 0.93 (277) | 73.17 (4033) | 0.0181 | 11763 |
| single | page_0590 | unlimited-ocr-2 | base | 3 | 37.47 | 37.47 | 9.09 | 2.27 | 0.73 (279) | 33.41 (1644) | 0.0203 | 4497 |
| doc | s2 | unlimited-ocr-2 | multi | 1 | 33.42 | 11.14 | 9.25 | 6.25 | 1.45 (827) | 24.71 (1232) | 0.0201 | 2306 |
| doc | s2 | unlimited-ocr | multi | 1 | 65.14 | 21.71 | 7.60 | 2.52 | 0.86 (338) | 60.58 (3223) | 0.0188 | 4617 |
| doc | s2 | unlimited-ocr | per-page | 1 | 51.01 | 17.00 | 9.14 | 6.62 | 2.08 (831) | 41.06 (2283) | 0.0180 | 3091 |
| doc | p10 | unlimited-ocr-2 | multi | 1 | 151.66 | 15.17 | 9.74 | 19.87 | 3.02 (2745) | 127.35 (6277) | 0.0203 | 19595 |
| doc | p10 | unlimited-ocr | multi | 1 | 139.12 | 13.91 | 7.91 | 7.95 | 1.96 (1115) | 127.69 (6435) | 0.0198 | 15445 |
| doc | p10 | unlimited-ocr | per-page | 1 | 169.71 | 16.97 | 9.14 | 21.21 | 6.82 (2770) | 139.89 (7329) | 0.0191 | 20386 |
| doc | p20 | unlimited-ocr-2 | multi | 1 | 427.12 | 21.36 | 10.37 | 40.70 | 6.35 (5485) | 378.02 (17614) | 0.0215 | 48667 |
| doc | p20 | unlimited-ocr | multi | 1 | 621.71 | 31.09 | 8.16 | 15.51 | 2.76 (2225) | 601.61 (30543) | 0.0197 | 73461 |
| doc | p20 | unlimited-ocr | per-page | 1 | 410.77 | 20.54 | 9.15 | 41.47 | 13.31 (5540) | 353.77 (20089) | 0.0176 | 54323 |

**Two v1 runs ran away, so their wall overstates v2's lead.**
- `single page_0590`: a table-row loop, 4,033 tokens against v2's 1,644.
- `doc p20 --multi-page`: a counting loop, "\( 10^{-1643} \) to \( 10^{-1644} \) …", for 30,543 tokens.

`u2f1_guard_replay.py` replays the runaway detectors on these outputs. Its results are on bd-gftg.

## Against the plan's expectations (§8.1)

1. **Decode per token, expected ≈ v1: refuted.** v2's decode costs more per token in every pairing:
   - 12–26% more on single pages;
   - 6–22% more than v1 per page on documents;
   - 2–9% more than v1 `--multi-page`, whose attention grows with the document.

   Neither the analytic MACs nor the weight files explain this:
   - MACs per token are about equal (≈ 0.59 G, of which attention is ≤ 16 M on either model).
   - The two weight files differ by 16 MB.

   The likely cause is v2's per-token SR-SWA work: router scoring, block selection and gather. That work is small in MACs but bound by latency. The `srswa.router` / `srswa.attend` / `srswa.kbar` rows that would show it need code, and the build gate holds that code.
2. **v2 multi-page decode, expected flat: not shown either way.**
   - v2 moved from 0.0201 to 0.0203 to 0.0215 s/token from 3 to 10 to 20 pages.
   - v1 `--multi-page` moved from 0.0188 to 0.0198 to 0.0197.
   - These are single runs on a host whose load swung from 27 to 59.
3. **Prefill: as planned.** v2 prefills 274 tokens per page and v1 multi-page 111 (5,485 against 2,225 at 20 pages). That costs 6.35 s against 2.76 s.
4. **Vision: as planned, slightly under.**
   - v2 document mode spends 1.99–2.08 s per page.
   - That is 2.5–2.6× v1 `--multi-page` (0.78–0.84 s); the plan expected 2.8×.
   - It matches v1 per page (2.07–2.21 s).
   - On single pages both models spend 2.2–2.6 s.
   - The G2 figure of ≈ 4.2 s per page was measured on a different host at 8 threads.
5. **Where v2's document wall goes:**
   - decode: 74%, 84% and 89% of wall at 3, 10 and 20 pages;
   - vision: 19%, 13% and 10%.

## The §10 cost bounds (U2-E5a item 4)

| | 3 pages | 10 pages | 20 pages (W=10) | bound |
|---|---|---|---|---|
| v2 wall per page ÷ v1 `--multi-page` | 0.51 | 1.09 | 0.69 (v1 ran away) | ≤ 1.5 |
| v2 wall per page ÷ v1 per page | 0.66 | 0.89 | 1.04 | ≤ 1.2 |
| v2 peak RSS − v1 `--multi-page` | +1.65 GiB | +1.83 GiB | +2.21 GiB | ≤ +1 GiB |
| v2 peak RSS − v1 per page | +0.11 GiB | +0.60 GiB | +1.22 GiB | (no bound) |

**The wall bounds pass at every size** (one run each). **The RSS bound fails at every size.**

The RSS gap is mostly not v2's KV:
- v1 `--multi-page` sits about 1.5 GiB below v1's own single page (7.60–8.16 against 9.07–9.15 GiB), because it reads pages at 640 px.
- v2's single page (9.09 GiB) is level with v1's single page.
- The U2-D3 memory note (`artifacts/perf/bd-uocr2-epic-cgux.26/README.md`) puts the peak at the 1024-px SAM forward and the CLIP step-up before decoding.
- KV is a small part: `p20`'s first window logged a KV peak of 397 MB.

To pass, v2 would have to cut its vision-stage peak by about 1.2 GiB. Windowing and KV work cannot do that.

## Wall time per page (replaces the plan's 15–60 s book estimate)

v2 document mode took 11.1, 15.2 and 21.4 s per page on the 3-, 10- and 20-page documents. Those pages averaged 411, 628 and 881 generated tokens.

Per page, the cost is about:
- 2.0 s of vision;
- 0.3–0.5 s of prefill;
- 0.020–0.0215 s per generated token.

How much text a page holds sets its cost:
- About 3–6 hours per 1,000 pages like these.
- About 44 s per dense 2,000-token page, or about 12 hours per 1,000 such pages.

U2-C5a's MarathonDocBench documents averaged 55.7 s per page (`docs/UOCR2_EVIDENCE.md`, Cost). Those ran at the 36-thread default with other work on the host, and bd-zex4 measured that default at 2.4–2.6× the wall of 12 threads.

Runs longer than 20 pages are not measured here.

## Files

- `runs.tsv`: one row per run: wall, user, sys, max RSS, peak footprint, output characters, exit code.
- `*.time`: each run's stderr, with the `FOCR_TIMING` rows and the `/usr/bin/time -l` report.
- `*.r1.md`: the first rep's output for every cell. Single-page outputs are byte-identical across reps (see the PERF_LEDGER rows).
- `host.txt`, `load_samples.log`, `pages.sha256`.
- `summary.json`: written by `u2f1_summarize.py`.
- `u2f1_bench.sh`: the harness, kept as run, with its scratchpad paths.
- `u2f1_ledger.py`: writes the PERF_LEDGER rows (`U2F1-*-20261101`).
- `u2f1_guard_replay.py`: the bd-gftg detector replay on these outputs.
- `SHA256SUMS`.
