Calibration page sets, fixed before the first calibration run (2026-10-31).

A directory of 36 distinct PNGs in name order: the 20 corpus pages (`~/.cache/franken_ocr/corpus/pages`), then pages 1–10 of the Marathon document `mp3lan_0226-p213-222`, then pages 1–6 of `mp3lan_0146-p62-70`. Content sha256 of the 36 files in order: 8bade0fd6ad8a430…

- 3-page run: the first 3 files.
- 20-page run: the first 20.
- 36-page run: all of them.

All three use the default thread budget and W=10, with 1,000 tokens per page assumed. The estimator was built from exploratory runs on a different document (`mp3lan_0165-p165-173`, pages 1–3), so none of these pages informed it.

The constants those exploratory runs set: 90 MB for the process, 2.6 GB of vision scratch, 0.6 GB of decode scratch. Everything else in the estimate comes from the artifact's tensor directory or the plan's row arithmetic.
