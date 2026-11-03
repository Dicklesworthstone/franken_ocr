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
- MarathonDocBench leaves 2,185 of its 10,319 table annotations (21%) with no
  html and no text, and none is marked `ignore`. On such a page the all view has
  no table, so a table the model reads counts as inserted text. Here 21 of the
  142 MarathonDocBench pages are affected, in 5 of the 20 documents. This was
  found after scoring, in U2-D2b's interim (bd-uocr2-epic-cgux.25), and the
  documents are not re-scored.

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
4. **Precision check, added after the run (`precision`, gates nothing).** The
   single-page tie raised the question whether int8 costs v2 more than v1.
   - Sample: 40 pages, `random.Random(20261102)`, from those both int8 arms
     completed.
   - Both models read them again on the all-f32 cache from the raw checkpoints
     (`FOCR_DECODE_F32=1`, no stage budget, same binary).
   - Each page is scored against ground truth and paired with its as-run
     int8 score.
5. **Wall-clock contamination.** Other work ran on the host during the
   evaluation, and the default 36-thread budget spends most CPU time in the
   kernel (bd-zex4). Cost figures are reported with that flag.

## Results

The run ended 2026-11-01 06:25 EST with every process exit 0. These are the
as-run results, the registered sensitivity reruns (Deviation 2, finished
14:21 EST) and the precision check (Deviation 4). The run records, scores and
post-hoc outputs behind them are in `artifacts/accuracy/bd-uocr2-epic-cgux.21/`.
Its README says where the raw readings are kept, outside the repository, because
OmniDocBench may not be redistributed.

### Pre-registered tests, as run

| test | n | mean | 95% interval | verdict |
|---|---|---|---|---|
| single page, CER_text(v2) − CER_text(v1) | 315 | −0.0694 | [−0.0985, −0.0414] | non-inferior |
| tables, TEDS(v1) − TEDS(v2) | 66 | −0.0344 | [−0.0946, +0.0178] | not shown non-inferior |
| documents, ED_all(v1 multi-page) − ED_all(v2) | 32 | +0.4146 | [+0.3075, +0.5177] | v2 better |
| documents, ED_all(v1 per page) − ED_all(v2) | 32 | −0.0132 | [−0.0616, +0.0155] | v2 not better |

- **Documents:** v2 had to beat both v1 cells, and it beat only v1 multi-page.
  The document test fails.
- **Category regressions (n ≥ 15, τ = 0.02):**
  - single page `data_source=note` (n = 25): +0.0707 [+0.0029, +0.1654];
  - single page `language=en_ch_mixed` (n = 23): +0.0936 [+0.0124, +0.2007];
  - documents against v1 per page, MarathonDocBench (n = 20): v2 worse by
    0.0222 [−0.1000, +0.0229].
- **Plan §10:** neither default changes on this evidence.
  - The single-page gate needs tables to pass too.
  - The multi-page gate needs v2 to beat both v1 cells, and also needs the
    100+-page documents of U2-C5b.

### Sensitivity reruns (Deviation 2, registered)

All 42 budget pages (36 v1, 6 v2) were read again alone with no budget, and
all finished. With the rerun scores in place of the failures:

| test | n | mean | 95% interval | verdict |
|---|---|---|---|---|
| single page, CER_text(v2) − CER_text(v1) | 315 | −0.0166 | [−0.0325, +0.0003] | non-inferior |
| tables, TEDS(v1) − TEDS(v2) | 66 | +0.0109 | [−0.0149, +0.0400] | not shown non-inferior |

- **The two analyses agree on both verdicts.** Per Deviation 2, the rerun
  describes the models.
  - On text, v2's margin is about a quarter of the as-run one, and the
    interval touches zero.
  - On tables, v2 is slightly worse on average. 66 tables cannot show
    non-inferiority either way.
  - So the single-page gate fails on tables in both analyses.
- **v1's 36 pages: at least 10 are v1's own runaways, and the rest were
  cascade victims.**
  - The 10 have output whose second half compresses to 12% of its size or
    less, or ran for 1,000 s or more. Seven ran for 1,167–1,662 s. Five of
    those wrote about 31,000–32,700 tokens (re-encoded from their text), at
    v1's 32,768-token length limit.
  - On those 10, v1's CER_text averages 0.900, against v2's 0.596.
  - The other 26 finished in 26–761 s, with mean CER_text 0.396.
- **One rerun row is corrected by hand.** The `b2cc5729` rerun was paused
  for about three minutes during a host disk emergency (09:25 EST).
  - The signal also reached its `/usr/bin/time` wrapper, which logged 376 s.
  - focr itself ran for 1,597 s, and the next rerun started 1,598 s later.
  - The row's `wall_s` is focr's own time, and it carries a note. Its score
    comes from focr's output and is unaffected.
- **v2's 6 pages still run away without a budget.**
  - Five have repetitive output, `notes_…_50` among them: 46,063 characters
    in 956 s.
  - Their CER_text averages 0.912, so the budget was not what cost v2 these
    pages.
- **Tail (AF-2):** with the reruns, CVaR_0.1 is 0.809 for v1 and 0.697 for
  v2, so the tail gate passes.

### What drives the results (read after scoring; not part of the protocol)

- **Single page: the 36 budget pages decide the text result.**
  - As run, v1 lost 36 pages to the bd-3ttp cascade (Deviation 2), 23 of them
    newspaper pages, and each scores 1.0. v2 lost none.
  - Over the 279 pages both completed, CER_text(v2) − CER_text(v1) is +0.0021
    [−0.0128, +0.0175]: a tie, and not non-inferior at δ = 0.005.
  - The reruns (above) show that at least 10 of the 36 are v1's own
    runaways, and the rest were cascade victims. Scored as the models wrote
    them, v2 stays non-inferior (−0.0166).
  - That margin comes from the 36 pages, split about evenly:
    - v1's 10 runaways contribute −0.0097 to the mean over 315 pages;
    - the 26 victims, mostly dense newspapers, contribute −0.0081 (v1
      0.396, v2 0.297);
    - the other 279 pages contribute +0.0019.
- **Tables: one page decides the TEDS mean, and the test cannot pass on this
  corpus.**
  - On `yanbaor2_89c3718f…pdf_14`, v2 reads the 8 × 10 table cell for cell
    correctly. It writes its check marks as ✓ (U+2713) and ✗ (U+2717). The
    ground truth and v1 write them as √ (U+221A) and × (U+00D7). TEDS
    compares characters, so v2 scores 0.315 against v1's 0.996.
  - That page is 95% of the summed TEDS difference.
  - Mapping v2's marks to the ground truth's glyphs and rescoring, with the
    driver's scorer and bootstrap, is post hoc. It changes two pages and gives
    TEDS(v1) − TEDS(v2) = −0.0005 [−0.0210, +0.0174]. That is still not
    non-inferior, because the interval is wide.
  - At δ = 0.005 a zero difference needs about 950 single-table pages.
    OmniDocBench has 341, so the registered table test cannot pass here even
    when the models tie.
- **The two single-page category regressions rest on four pages:**
  - `notes_…_50`: v2 counted ordinals (1st … 267th) until the 600 s budget
    stopped it, 9,210 characters against 1,140. Each cell differs, so the
    n-gram guard cannot stop the run. Rerun with no budget, it counted on
    for 956 s.
  - `notes_…_69`: a two-column word list. v2 reads it row by row, with the
    words right (it reads "cage" where v1 reads "case"). Whole-page NED
    punishes the order: 0.732 against 0.024.
  - `jiaocai_needrop_en_1118`: v2 writes the page number and footer on one
    line ("110 高途课堂 · 秋季班"), so the furniture rule removes neither. The
    page's text view is 19 characters ("Words and Expressions"), and the 11
    left over score 0.367; v1 scores 0.
  - `jiaocai_needrop_en_443`: `\underline` markup and options on one line,
    and two words dropped.

  Without the first two, the `note` mean Δ is about +0.009.
- **Documents against v1 per page: one document.**
  - On `marathon-mp3lan_0789-p223-227`, v2 ran page 3 (a numeric table) to the
    20,000-token page cap, 33,404 characters.
  - It then opened page 4 with 20 malformed `<BLOCK>title<LEVEL1>` prefixes.
  - ED_all is 0.855 against 0.158 for v1 per page.
  - The run exited 0 with no warning (bd-jem5).
  - On the other 31 documents, v1 per page minus v2 averages +0.0089. v2 is
    lower on 17, higher on 12, and tied on 2.
  - Without it, the MarathonDocBench mean is +0.0134 (n = 19), so that
    category regression is the same document.
- **v1 multi-page** wrote one body per page on 6 of 32 documents.
  - 19 had fewer bodies and were warned (`v1_page_merge`, `v1_truncated`).
  - 7 had more bodies and were not warned (bd-17zw).
  - Its mean ED_all is 0.519, against 0.105 for v2 and 0.092 for v1 per page.
- **Precision:** f32 does not explain `notes_…_50`, which runs away at f32
  too. It does explain document 0789's page 3, which f32 reads as a table
  (post hoc, under "Runaways after the run").

### Precision check (Deviation 4; not pre-registered, gates nothing)

Both models read the 40 sampled pages again on the all-f32 path, from the raw
checkpoints. No page failed. CER_text, mean over the 40 pages:

| | v1 | v2 |
|---|---|---|
| int8, as run | 0.1403 | 0.1216 |
| f32 | 0.1399 | 0.1198 |
| f32 − int8 | −0.0004 [−0.0034, +0.0032] | −0.0018 [−0.0050, +0.0001] |

- v2 − v1 is −0.0186 [−0.0472, +0.0072] at int8 and −0.0201 [−0.0468, +0.0030]
  at f32. The int8 artifacts change neither model's error by more than about
  0.002, and they do not change the comparison.
- The sample comes from pages both int8 arms completed, so it covers typical
  pages, not the runaways. The runaways were read at f32 separately (below).
- f32 wall time was 3,157 s for v1 and 3,058 s for v2 over the 40 pages. No
  other evaluation step ran beside them; the host flag of Deviation 5 still
  applies.

### Runaways after the run (post hoc, gates nothing)

The records and scripts are in `artifacts/accuracy/bd-uocr2-epic-cgux.21/posthoc/`.

- **f32 behaves like int8 on the four category-regression pages.**
  - `needrop_en_443` and `notes_…_69` are byte-identical.
  - `needrop_en_1118` differs in its last few characters, at the same CER (0.367).
  - On `notes_…_50`, f32 writes int8's whole output and keeps counting to
    "1293rd" (46,064 characters, 1,704 s). CER is 0.909 at both precisions.
  - These regressions are v2's own behaviour, not quantization.
- **Document 0789 does not run away at f32.**
  - Read as a document from the raw checkpoint, page 3 closes as a
    5,789-character table. ED_all is 0.0467, against 0.8554 at int8 and
    0.1585 for v1 per page.
  - The two precisions diverge within the first 50–85 characters of pages
    1–3, which is ordinary greedy divergence.
  - int8 can therefore tip a document page into a counting loop that f32
    avoids.
- **The other 31 documents at f32 show it only on 0789.** All 32 documents
  were read as documents at f32 (`posthoc/score_f32_docs.out`, analysis fixed
  before the results; readings hashed in `readings.sha256`).
  - No f32 page reached the per-page cap; int8 reached it once, on 0789's
    page 3.
  - Per document, f32 − int8 is −0.0249 [−0.0757, +0.0007]. Without 0789 it
    is +0.0004, and no document moves more than 0.0061: f32 is better on 8,
    worse on 9 and equal on 14 (post hoc).
  - f32 − v1 per page is −0.0117 [−0.0229, −0.0026], where int8 − v1 per page
    is +0.0132 [−0.0155, +0.0615]. Against v1 multi-page it is −0.4395
    [−0.5342, −0.3427] at f32 and −0.4146 [−0.5174, −0.3080] at int8.
  - So the int8 artifact's document-level cost is this one runaway, which
    decides the v2-vs-v1-per-page sign. The registered comparison stands
    as run.
- **Read alone, the int8 page closes too.**
  - v2 int8 reading 0789's page 3 by itself writes a 5,735-character table,
    within NED 0.031 of the f32 page.
  - With that page in place of the capped one, document ED_all is 0.1254.
    The document mean v2 − v1 per page moves from +0.0132 [−0.0155, +0.0615]
    to −0.0096 [−0.0189, −0.0022].
  - Page 3 is the only `page_cap` page among v2's 224 document pages, so
    "re-read a capped page alone" changes only this page.
  - The policy was chosen after seeing 0789, so this is a sensitivity, not a
    result (bd-yrr1).
  - Live, with `FOCR_REREAD_CAPPED_PAGES=1` and the forward budget off, the
    re-read lands on the same 1,759-token table, and ED_all is 0.1254
    (bd-yrr1, 2026-11-02).
  - Adding v2's per-page runaway guard (`FOCR_RUNAWAY_GUARD=1`, bd-gftg)
    closes page 3 as `runaway` at about 8,700 tokens instead of 20,000. Decode
    tokens fall from 24,094 to 12,697, and wall time from 3,494 s to 1,628 s
    under different host load.
    - Page 4 then loses the malformed `<BLOCK>title<LEVEL…` prefix that
      followed the 20,000-token loop, and ED_all is 0.0877.
    - The document mean v2 − v1 per page is −0.0108 [−0.0207, −0.0026].
    - All of this is post hoc and n = 1.
- **The runaway guard** (v1, opt-in when measured, `FOCR_RUNAWAY_GUARD=1`; bd-gftg):
  - Replayed over every C5a output of 2,560 tokens or more, it fires on none
    of 99 legitimate outputs. It catches 13 of 19 v1 runaways and, in replay
    only, 6 of 8 v2 runaways. At the time the guard was v1-only; v2 now runs
    a per-page variant that closes the page as `runaway` (bd-gftg).
  - On the ground truth itself it never fires on OmniDocBench. MarathonDocBench
    gives 9 per-page fires at the 8,192-token floor: 7 on broken ground truth
    and 2 on real chart data.
  - Live, on the 7 v1 single pages the replay aborts, it fires on all 7. Each
    fire comes at the replay's token or up to 2,816 tokens later, and together
    they skip about 62% of those pages' decode tokens.
  - An aborted page returns no text. With v1's reruns substituted, rejecting
    the fired pages raises v1's mean CER_text from 0.1663 to 0.1681. Keeping
    their prefixes lowers it to 0.1654.
  - Since bd-gftg (owner decision, 2026-11-02; DISC-023), the guard is on by
    default for v1 single pages. A stopped page keeps its prefix and exits 5
    with `finish_reason: runaway`; `FOCR_RUNAWAY_GUARD=0` turns it off.

### Tail risk (AF-2; not pre-registered, gates nothing)

| set | baseline cell | CVaR_0.1 base / v2 | EVT p99.9 base / v2 | gate |
|---|---|---|---|---|
| single page, CER_text | v1 | 1.000 / 0.700 | 1.000 / 1.000 | pass |
| documents, ED_all | v1 multi-page | 0.933 / 0.634 | 0.976 / 0.854 | pass |
| documents, ED_all | v1 per page | 0.452 / 0.634 | 0.804 / 0.854 | fail |

The failing row is document 0789 again. The document bounds are the
empirical quantile, because 32 documents give fewer than 8 exceedances.

### Cost (contaminated host, Deviation 5)

| cell | pages | wall s/page | peak RSS GiB |
|---|---|---|---|
| v1 single | 320 | 131.9 | 9.68 |
| v2 single | 320 | 92.0 | 9.91 |
| v2 document | 224 | 55.7 | 9.85 |
| v1 multi-page | 224 | 100.9 | 8.08 |
| v1 per page | 224 | 47.8 | 9.46 |

- **Single page:** v1's figure includes its budget cascades.
  - Over the 284 pages both completed, v2 takes 0.96× v1's mean time.
  - Its median per-page ratio is 1.085, and its ratio of medians is 1.126.
  - The bound is 1.1×. Peak RSS is +0.23 GiB, under the 0.5 GiB bound.
- **Documents (≤ 10 pages):**
  - v2 takes 1.17× v1 per page (bound 1.2×) and 0.55× v1 multi-page
    (bound 1.5×).
  - Peak RSS is 1.77 GiB above v1 multi-page. The bound is +1 GiB, and it
    applies at 20 pages, which U2-F1 measures.
  - U2-F1 (post hoc, 12 threads, `artifacts/perf/bd-uocr2-epic-cgux.38`)
    measured +2.21 GiB at 20 pages, so the bound fails.
    - Most of the gap is v1 multi-page's 640-px vision, not v2's KV.
    - The two wall bounds pass there, but those are single runs.
- **Thread width (post hoc, bd-zex4):** every cell ran at the default 36
  threads. The width sweep after the evaluation used the same binary and ran
  under similar shared load (`artifacts/perf/bd-zex4`).
  - It measured 36 threads at 2.6× (v1) and 2.4× (v2) the wall of 12, with
    identical output.
  - So the costs above, and the 600 s budget losses, came at a width that slows
    both models alike. The pairing holds; the absolute times are not the
    engine's best.
