# Library Output Contract

What a program that links franken_ocr in process can rely on. Written for hfdt's
citation anchors, and true for any embedder.

Current version: **`franken_ocr::OUTPUT_CONTRACT_VERSION` = 1**. A new field keeps
the version. A changed meaning, unit or order bumps it.

`tests/library_contract.rs` holds this contract in place: the dependency line, and
golden spans for a clean page, a table page, a two-column page, a page of a long
document and a rotated page, all from real model output.

## Linking

```toml
[dependencies]
franken_ocr = { path = "../franken_ocr", default-features = false, features = ["library"] }
```

`library` is not in a published release yet: crates.io stops at 0.9.0, which has no
such feature. Until the next release, link a checkout by path, with `../frankentorch`
beside it, because `ft-core` and `ft-kernel-cpu` are path dependencies.

`library` gives you the models, kernels, page sources, PDF pages and mapped weights.
It links no CLI, async runtime, run store, signal handler, network stack or
allocator override, and it needs no `[patch]` section: `asupersync` and `fsqlite`
are not in the tree. `the_library_feature_links_no_cli_runtime_store_or_network`
reads the real resolved tree and fails on any of them. `OcrEngine` owns an async
runtime, so it stays in the `native` feature. Use `OcrModel` directly, as below.

Commit your own lockfile. Your resolve, not ours, picks the numerics crates under
`frankentorch-kernel-cpu`. Our lock has `wide 1.6.0`, which is yanked, so a fresh
resolve takes `wide 1.7.1`. Three pages (2,482 generated tokens) read the same under
both: identical text, layout, token counts and finish reasons. That is one
comparison, so run it again when `wide` or `safe_arch` moves
(`scripts/e2e_hfdt_admission_surface.py`).

The weights are not bundled. `OcrModel::load` takes the path of a `.focrq` artifact,
and Unlimited-OCR-2 needs its `tokenizer.json` beside that file. The library has no
path that downloads anything.

## Reading a document

```rust
use franken_ocr::DocumentOptions;
use franken_ocr::native_engine::OcrModel;
use franken_ocr::page_source::PageSource;

let model = OcrModel::load(&model_path)?;
let mut source = PageSource::open(&input)?; // a PDF, an image or a directory of images
let document =
    model.recognize_document(&mut source, &DocumentOptions::default(), &mut |_| {}, None)?;
for page in &document.pages {
    for span in &page.layout {
        // span.label, span.boxes, span.level, span.text
    }
}
```

`PageSource::images` takes a list of paths and `PageSource::rasters` takes decoded
pages. `recognize_document` blocks until the document ends. The two callbacks report
pages as they are read, and a windowed run delivers finished pages at the end of each
window, not one by one. Models outside the Unlimited-OCR family return
`FocrError::NotImplemented`.

## A page

`DocumentResult::pages` holds one `PageResult` per page read, in run order.
`PageResult::to_json` gives the same fields as `focr ocr --json` `pages[]`.

| Field | JSON | Meaning |
|---|---|---|
| `source_page` | `page` | 1-based page of the source. This is the number to cite. |
| `half` | `half` | `left` or `right` for a split book spread, otherwise absent. |
| `run_index` | `run_index` | 1-based position among the pages the model read. A skipped page does not get one, so this can differ from `page`. |
| `window` | `window` | The decode window that read the page, absent for one pass. |
| `size` | `width`, `height` | The raster the boxes are in, in pixels. |
| `text` | `text` | The page's markdown. |
| `layout` | `layout` | The page's spans, below. |
| `tokens` | `tokens` | Tokens generated for the page, when the model counts them. |
| `finish` | `finish` | How the page ended, below. |
| `raw` | none | The body as the model wrote it, boxes on its 0–999 grid. |
| `figures` | none | Figure boxes, numbered as `text` numbers them. |
| `elapsed_ms` | none | Decode time. It differs run to run. |

`pages` is empty for a v1 document whose text did not split into one body per page.
Its `v1_page_merge` warning (fewer bodies than pages) or `v1_extra_pages` warning
(more) says so. Unlimited-OCR-2 always gives one entry per
page read.

## A span

| Field | JSON | Meaning |
|---|---|---|
| `label` | `label` | The model's label, such as `text`, `title`, `table`, `header`, `footer`, `page_number`, `image`, `image_caption`. The set is the model's and is not closed. |
| `boxes` | `boxes` | One or more `[x1, y1, x2, y2]`. |
| `level` | `level` | The heading level of a v2 `title`, otherwise absent. |
| `text` | `text` | What the model wrote for the span, normalized as the markdown is. Absent for an image span, and for a marker with nothing after it. |

There is no block level between page and span. A span is the model's own unit: a
paragraph, a heading, a table, a line of an index.

**Boxes are pixels of that page's raster,** origin top left, `x` right and `y` down.
The raster is the page after any EXIF orientation is applied. For a split spread it
is that half. For a PDF it is the page's embedded scan, not PDF points.

Each coordinate is `int(coord / 999 * extent)`, truncated toward zero, where `coord`
is the model's 0–999 value and `extent` is `width` or `height`. Going back to the
model's grid loses up to a pixel.

**Boxes are as the model wrote them.** They are not clamped to the page and not put
in order, so a coordinate past 999 lands past the page edge and `x2` can be below
`x1`. Every box in the golden pages is in bounds and ordered, but that is the model's
habit and not a promise. Clamp before you crop. `a_box_is_mapped_as_written` holds
this.

**Spans are in the order the model read the page.** Nothing sorts them, so
`text` follows the same order. A two-column page goes down the left column and then
down the right (`library_two_column_page.golden`). The order is the model's judgment
and it can be wrong on a hard layout. Use the boxes if you need geometry.

**A table is a `table` span** whose `text` is the HTML the model wrote:
`<table><tr><td>…</td></tr></table>`, with `rowspan`, `colspan` and HTML entities such
as `&#x27;` as written. The markup is not validated and table cells are not separate
spans. Cells on a ruled scan can contain long dot leaders, and a bad decode can leave
empty cells or stray tokens at the end of a table.

## Confidence

There is no per-span confidence, and version 1 does not promise one. The number the
decoder has is a token's softmax probability, which is not calibrated: it does not
mean "right this often", and treating it that way would put a false number in a
citation. Calibrated confidence needs per-token capture through the windows and the
journal, then calibration against labeled pages. It would arrive as a new field, so
it would not bump this version.

## Rotated pages

Nothing detects rotation or deskews. A page turned 90° in its pixels is read sideways.
`width` and `height` swap and the boxes are in the turned frame, so a line of text gets
a tall, narrow box. The boxes stay usable and the text gets worse: on our test page the
model wrote `MINIMAX HISTORY OF THE ROYAL NAVY` for `MILITARY HISTORY OF THE ROYAL NAVY`
and `1602` for `1603`, both of which it reads correctly upright. Turn a page upright
before you read it. EXIF orientation is applied for you, because that is metadata and
not pixels.

## How a page ends

`finish` is one of:

| Value | Meaning |
|---|---|
| `marker` | The model opened the next page. |
| `eos` | The model ended the document. |
| `page_cap` | The per-page cap forced the page closed. The page is probably cut short, and its text probably repeats; the run's `v2_page_cap` warning lists it. With `DocumentOptions::reread_capped`, the page is read again alone after the document; a re-read the model ends replaces the page, and the run's `v2_page_reread` warning lists it. |
| `runaway` | With `FOCR_RUNAWAY_GUARD=1` (off by default for Unlimited-OCR-2), the runaway guard found the page repeating and closed it as the cap does. The page keeps its text so far, which is cut short; the run's `v2_page_cap` warning lists it under `pages` and `runaway`. `DocumentOptions::reread_capped` reads it again as it does a `page_cap` page. |
| `length` | The position limit stopped decoding inside this page. |
| `budget` | The stage budget stopped decoding inside this page. |
| `cancelled` | `request_shutdown` stopped decoding inside this page. |

`DocumentResult::finish_reason` says how the whole decode ended: `stop`, `length`,
`budget`, `cancelled`, or `runaway` (a v1 document decode the runaway guard stopped,
under `FOCR_RUNAWAY_GUARD=1`). Anything but `stop` means the document is partial, and
`pages` holds what was completed.

A v1 single page (`OcrModel::recognize` and its siblings) runs the runaway guard by
default (bd-gftg, DISC-023). Once the page passes 8,192 tokens and keeps repeating, the
decode stops and the call returns the text before that point, and
`OcrModel::take_last_decode` reports `finish_reason: runaway`. `FOCR_RUNAWAY_GUARD=0`
decodes to the token limit, as the reference does.

`skipped` lists source pages that did not rasterize, each with the engine's reason.
The usual causes are JPEG 2000, JBIG2 and born-digital pages with no full-page image.
They leave no `PageResult`. `recognize_document` returns an error only when no page
rasterized at all. `warnings` carries everything else worth knowing about the run.

## Side effects

**Files.** With `DocumentOptions::default()` the engine writes nothing. A real read
with empty `HOME`, `TMPDIR` and working directory left all three empty. Two things
change that: `DocumentOptions::run` keeps a journal and checkpoint where you point it,
and `FOCR_CALIB_OUT` writes calibration statistics.

**Stderr.** An experimental model prints one note per load:
`[focr] note: <id> is experimental: not yet certified against its reference`. No
shipped model is experimental today; unlimited-ocr-2 has been ready since its
certified artifact was published.

**Environment.** The engine reads `FOCR_*` tuning variables from the process
environment, as the CLI does. `src/native_engine/lever_policy.rs` lists each one and
what it does. Most change speed or memory only. `FOCR_THREADS`, `FOCR_MMAP`,
`FOCR_STAGE_BUDGET_FORWARD_MS` and `FOCR_CALIB_OUT` are the ones to know.

**Threads.** The kernels run on rayon's global pool. franken_ocr builds that pool once
at the thread budget: `set_thread_budget`, else `FOCR_THREADS`, else the physical core
count, which on an Apple-silicon Mac counts its fastest core tier only. If your program
built the global pool first, yours stands and the engine runs at its width. The budget
latches for the process, so set it before the first read.

**Shutdown.** `request_shutdown` is process-wide and installs no signal handler. Wire
it to your own. A cancelled Unlimited-OCR-2 document returns the pages it completed
with `finish_reason: cancelled`, and v1 returns `FocrError::Cancelled`. Call
`reset_shutdown` before the next document.

**Memory.** Unlimited-OCR-2 peaks at about 10.4 GiB of physical footprint at the
default window of 10 pages. The window bounds it: 20 and 36 pages measured the same.
`FOCR_MMAP=1` maps the weights, which brings the footprint to about 6.2 GiB at about
twice the wall time, while RSS stays near 10 GiB because mapped pages still count
toward it. `PageWindow::Auto` picks the largest window whose estimate fits the memory
budget. See `docs/planning/UNLIMITED_OCR_2_PLAN.md` §6.3.

## Not promised

- Identical output across model versions. The artifact is part of the input.
- Identical output across versions of the numerics crates. See Linking.
- Anything about `raw`'s grammar, which belongs to the model.
- The order of keys in `to_json`.
