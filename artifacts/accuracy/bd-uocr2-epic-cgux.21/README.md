# U2-C5a head-to-head: run records, scores and post-hoc diagnostics

U2-C5a compares Unlimited-OCR (v1) with Unlimited-OCR-2 (v2), on single pages and on whole documents. It ran 2026-10-31 to 2026-11-01 on an Apple M5 Ultra under shared load. The results and their interpretation are in `docs/UOCR2_EVIDENCE.md`. This directory keeps the records those numbers come from, so they survive the session scratchpad.

- Driver: `scripts/uocr2_head_to_head.py`, with subcommands `select`, `fetch`, `run`, `rerun`, `precision`, `score`, `report`.
- Binary: `focr 0.10.0` release-perf, built 2026-10-30 22:27 at the pinned build root, sha256 `b6c5b571…552de2a3`.
- Models (`artifacts.json`): v1 `unlimited-ocr.int8.focrq` sha256 `57334071…22fb592`; v2 `unlimited-ocr-2.v0.1.0.int8.focrq` sha256 `54b8a1ec…35b4398`.

## The readings are not in this directory

The model outputs contain the pages' text. OmniDocBench is licensed for research use only, with no redistribution (see `items.json` → `sources`), so the raw readings live outside the repository:

```
~/.local/share/franken_ocr/evidence/bd-uocr2-epic-cgux.21/
  runs/        every cell's focr JSON (single, docs, rerun, precision)
  text_layer/  ground truth taken from the born-digital PDFs' text layers
  diag_f32/    post-hoc f32 and page-3 re-read outputs; docs/ is the f32 document sweep
  guard_live/  post-hoc live runaway-guard outputs
```

- `readings.sha256` hashes every file there. Do not commit them.
- The `output` paths in `runs.ndjson` and `rerun.ndjson` point at the session scratchpad. Replace `…/scratchpad/c5a_data/` with the directory above.

The inputs are rebuilt from the datasets with `select` and `fetch`; the sources, revisions, seeds and truth hashes are in `items.json`. `inputs.sha256` hashes all 686 staged input files.

## What is here

| path | contents |
|---|---|
| `items.json` | The 320 single pages and 32 documents, with sources, revisions and seeds. |
| `artifacts.json` | Binary and model paths and hashes. |
| `runs.ndjson`, `scores.ndjson` | One record per run (exit, wall, peak RSS, load at start) and one per scored item and cell. |
| `rerun.ndjson` | The 42 budget-lost pages rerun alone with no budget, the registered sensitivity. |
| `precision.ndjson`, `precision_summary.json` | The f32-vs-int8 check on 40 single pages. |
| `summary.json`, `report.md` | The report the driver wrote, including the reruns. |
| `c5a_*.log` | The score, report, rerun and precision logs as run. |
| `scripts/` | The post-hoc scripts below and the diagnostic run scripts. |

## Post hoc (gates nothing)

| file | what it shows |
|---|---|
| `posthoc/score_f32_pages.out` | v2's four category-regression pages read at f32 behave as at int8. `ch` is the scored text view, not the raw markdown. |
| `posthoc/score_0789_f32.out` | Doc 0789 read at f32: ED_all 0.0467 against int8's 0.8554. |
| `posthoc/counterfactual_0789.out` | Doc 0789 with page 3 cut where a runaway guard would close it. |
| `posthoc/score_0789_p3_r2.out` | Doc 0789 with page 3 read alone at int8: ED_all 0.1254. The document mean v2 − v1 per page moves from +0.0132 to −0.0096. |
| `posthoc/score_f32_docs.out` | All 32 documents read as documents at f32 (`scripts/diag_f32_docs.sh`, scored by `scripts/score_f32_docs.py`, analysis fixed before the results). No f32 page reaches the page cap. f32 − int8 is −0.0249 [−0.0757, +0.0007], all of it 0789; f32 − v1 per page is −0.0117 [−0.0229, −0.0026]. |
| `posthoc/guard_calib.ndjson`, `posthoc/guard_policy.out` | The runaway guard replayed on every C5a output, and reject vs keep-the-prefix. |
| `posthoc/truth_fp.ndjson`, `posthoc/truth_fp.out` | The guard on the ground truth itself, to find false positives. |
| `posthoc/guard_live_summary.md` | The guard on real decodes, against the replay. Wall times ran under heavier load than the reruns. |

Beads: bd-uocr2-epic-cgux.21 (U2-C5a), .33 (E5a), bd-gftg (runaway guard), bd-jem5 and bd-yrr1 (page-cap warning and re-read).

The f32 document sweep finished on 2026-11-02 at 02:52 ET: all 31 documents exited 0, with no disk-guard pause. Its 126 readings are under `diag_f32/docs/` and in `readings.sha256`.
