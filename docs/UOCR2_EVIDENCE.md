# Unlimited-OCR-2 vs Unlimited-OCR: ground-truth head-to-head (U2-C5a)

Bead: `bd-uocr2-epic-cgux.21`. Driver: `scripts/uocr2_head_to_head.py`
(`select`, `fetch`, `run`, `rerun`, `score`, `report`, `selftest`). The
evaluation data (OmniDocBench pages, MarathonDocBench spans, the non-Baidu
PDFs) is downloaded and scored locally and never committed; the driver's
`NON_BAIDU` table is the sources manifest (URL, SHA-256, language, licence),
and `OMNI_REVISION` / `MARATHON_REVISION` pin the two datasets.

The protocol below was registered on the bead on 2026-10-30, before any
evaluation run, and is repeated here unchanged. Everything that happened
differently is listed under Deviations. A loss is reported as a loss.

## Pre-registered protocol

**Artifacts.** v1 `unlimited-ocr.int8.focrq` (sha256 `573340…fb592`), v2
`unlimited-ocr-2.v0.1.0.int8.focrq` (sha256 `54b8a1ec…`, certified by U2-C4).
Binary: a release-perf `focr` from the working tree. Base crop mode for both.
`FOCR_NO_RESIDENT=1`. Same host, default thread count.

**Sets.**
- Single pages: OmniDocBench v1.5 (research-only licence; 1,651 pages). The
  frame is every page with no `*_mask` / `need_mask` element (1,525 pages). The
  sample is 320 pages stratified by `data_source`, proportional allocation
  (largest remainder), `random.Random(20261030)`, pages sorted by `image_path`
  within each stratum.
- Documents of 5–10 pages, read whole (one v2 window):
  - 20 MarathonDocBench spans (Apache-2.0): 20 documents from the 100 sorted
    by doc_id, `random.Random(20261031)`. Per document, L = randint(5, 10) and
    start = randint(1, N − L + 1) from the same generator.
  - 12 non-Baidu born-digital PDFs (37.5% of documents): six US Supreme Court
    opinions, NIST FIPS 197-upd1 and IR 7298r2, arXiv 1706.03762v7, and the
    Wikipedia PDF renders of en "Optical character recognition", zh "印刷术"
    and zh "造纸术". Spans as above with `random.Random(20261101)` and
    L = randint(5, min(10, N)). Pages come from `pdftoppm -r 150 -png`. Ground
    truth is the text layer, from `pdftotext -enc UTF-8` per page.

**Runs.**
- Single page: `focr ocr-batch --model <id> --json`, 8 chunks of 40 pages.
  Chunk k runs v1 first when k is even and v2 first when k is odd.
- Documents: three cells per document, each `focr ocr <page-dir> --json`:
  - v2 `--model unlimited-ocr-2 --multi-page`;
  - v1 `--model unlimited-ocr --multi-page`;
  - v1 per page (no `--multi-page`).

  The cell order rotates with the document index.
- Every process runs under `/usr/bin/time -l` (wall, peak RSS), and the load
  average is logged at its start.

**Normalization N(s)**, applied to both sides, in this order:
1. NFKC.
2. Drop `<PAGE>` markers and markdown image references.
3. HTML-unescape, then drop HTML tags.
4. Drop the math delimiters `$`, `\(`, `\)`, `\[`, `\]`.
5. Drop leading `#` heading markers, `**`, `__`, backticks and leading list
   bullets.
6. Remove every whitespace character. There is no case folding.

NED(a, b) = lev(N(a), N(b)) / max(|N(a)|, |N(b)|), and 0 when both are empty.

**Ground truth.** OmniDocBench and MarathonDocBench elements are taken in
`order`; elements with no order follow by box top, then left.
- Furniture is excluded from content: header, footer, page number, header and
  footer images, abandon, formula number.
- The text view is the text-like categories.
- The all view adds table cells (tags dropped) and display formulas (`latex`)
  in place.

Text-layer ground truth (non-Baidu): see Amendment 1.

**Predictions.**
- Furniture removal: each ground-truth furniture string removes at most one
  prediction line. That is the closest remaining line, provided its NED is at
  most 0.25.
- Text view: the markdown without tables and display math.
- All view: the markdown with tables and math kept (tags dropped by N).

**Metrics.**
- CER_text (single page) is the NED of the text views. Pages with an empty
  ground-truth text view are left out of CER_text only.
- ED_all is the NED of the all views. It is secondary for single pages and
  primary for documents, over the page-ordered concatenation.
- TEDS, on pages with exactly one ground-truth table: the best exact
  (Zhang–Shasha) tree-edit similarity against any predicted table, and 0 if
  there is none.
- FormulaED, on pages with display formulas: the NED of the concatenated
  formulas.
- **A run that errors or produces nothing scores NED 1.0 on every metric of
  every page it should have produced.** A truncated document keeps its missing
  text in ED.

**Statistics.** Bootstrap: 10,000 resamples, numpy `default_rng(20261030)`,
percentile 95% interval.
- Single page, primary: Δ = CER_text(v2) − CER_text(v1). v2 is non-inferior
  when the upper end is below δ = 0.005.
- Tables: Δ = TEDS(v1) − TEDS(v2), with the same δ.
- Documents, primary: Δ = ED_all(v1) − ED_all(v2), separately against v1
  multi-page and v1 per page. v2 wins when the lower end is above 0 in both
  comparisons.
- Categories, with tolerance τ = 0.02: a category with n ≥ 15 regresses when
  its mean Δ (v2 worse) is above τ. Smaller categories are reported without a
  verdict.
- Cost per cell: wall seconds per completed page and peak RSS, against the
  plan §10 bounds.

**Amendment 1** (2026-10-30, still before any run). The registered text-layer
furniture rule classed table cells, section letters and repeated mid-page
lines as running headers. The amended rule: a text-layer line is furniture
when all of the following hold:
- it is among the first or last three non-empty lines of its page;
- its digit-free normalized form has at least 5 characters;
- that form recurs, at a page edge, on at least 3 pages of the PDF.

A bare page number near a page edge is still furniture. The drawn sample has
320 pages (66 single-table, 72 with display formulas, 5 with an empty text
view) and 32 documents with 224 pages, 12 of them non-Baidu.

## Deviations

1. **Driver timeout.** The driver's 3 h per-process timeout was never
   registered, and it would have scored the live `single/v1/chunk6` as 40
   failures.
   - The driver was held before the timeout fired.
   - The chunk's row was written by hand once `focr` exited. Its `wall_s` is
     focr's `seconds_total`, and its peak RSS is the highest 5 s sample, a
     lower bound. The row says so in its `note`.
   - The run then restarted with a 7 h timeout that kills the whole process
     group.
2. **The v1 arm is confounded by bd-3ttp.** In the pinned binary, a page that
   overruns the 600 s forward budget keeps decoding, and the next pages time
   out behind it.
   - Under the registered failure rule these pages score 1.0 for v1. v2 stops
     itself at the budget and keeps its text.
   - This was found before any score was computed. The registered sensitivity
     analysis therefore reruns every single page an arm lost to the budget,
     alone and unbudgeted (`FOCR_STAGE_BUDGET_FORWARD_MS=0`, same binary): 36
     v1 pages and 6 v2 pages.
   - Both results are reported. Where they disagree, the rerun describes the
     models and the as-run result describes the binary.
3. **Rows added after the run started, reported only, gating nothing:**
   - CER_text over the pages both arms completed;
   - per-cell failure counts by `data_source`;
   - the AF-2 tail table (mean, VaR, CVaR_0.1, EVT p99.9, `scripts/af2_tail_risk.py`).
4. **Wall-clock contamination.** Other work ran on the host during the
   evaluation, and the default 36-thread budget spends most CPU time in the
   kernel (bd-zex4). Cost figures are reported with that flag.

## Results

The run is still in progress (2026-11-01). The as-run scoring and the
sensitivity reruns are recorded here when they finish.
