# franken_ocr Library Integration

## Table of Contents

- [Public Shape](#public-shape)
- [Cargo Dependency](#cargo-dependency)
- [Minimal Example](#minimal-example)
- [Structured Layout Output](#structured-layout-output)
- [Engine Lifetime](#engine-lifetime)
- [Async Hosts](#async-hosts)
- [Dynamic Images and PDFs](#dynamic-images-and-pdfs)
- [Batching](#batching)
- [Model Architectures](#model-architectures)
- [Model Paths](#model-paths)
- [Error Handling](#error-handling)
- [Testing Integrations](#testing-integrations)
- [Production Checklist](#production-checklist)
- [Anti-Patterns](#anti-patterns)

## Public Shape

`franken_ocr` is both a CLI project and a reusable Rust library. The intended
embedding surface is synchronous and blocking:

- `OcrEngine::new() -> FocrResult<OcrEngine>`
- `OcrEngine::model_path()`
- `OcrEngine::recognize(&Path) -> FocrResult<String>`
- `OcrEngine::recognize_with_model(model_path: &Path, image_path: &Path) -> FocrResult<String>`
- `OcrEngine::recognize_with_layout(&Path) -> FocrResult<RecognizedDocument>`
- `OcrEngine::recognize_with_layout_model(model_path: &Path, image_path: &Path) -> FocrResult<RecognizedDocument>`
- `OcrEngine::recognize_dynamic(image::DynamicImage) -> FocrResult<String>`
- `OcrEngine::recognize_dynamic_with_model(model_path: &Path, image: image::DynamicImage) -> FocrResult<String>`
- `OcrEngine::recognize_dynamic_with_layout(image::DynamicImage) -> FocrResult<RecognizedDocument>`
- `OcrEngine::recognize_dynamic_with_layout_model(model_path: &Path, image: image::DynamicImage) -> FocrResult<RecognizedDocument>`
- `OcrEngine::recognize_with_figures(&Path) -> FocrResult<(RecognizedDocument, Vec<ExtractedFigure>)>`
- `OcrEngine::recognize_with_figures_model(model_path: &Path, image_path: &Path) -> FocrResult<(RecognizedDocument, Vec<ExtractedFigure>)>`
- `OcrEngine::recognize_dynamic_with_figures(image::DynamicImage) -> FocrResult<(RecognizedDocument, Vec<ExtractedFigure>)>`
- `OcrEngine::recognize_dynamic_with_figures_model(model_path: &Path, image:
  image::DynamicImage) -> FocrResult<(RecognizedDocument, Vec<ExtractedFigure>)>`
- `OcrEngine::recognize_batch(&[&Path]) -> FocrResult<Vec<FocrResult<String>>>`
- `OcrEngine::recognize_batch_with_model(model_path: &Path, images: &[&Path])`
- `OcrEngine::recognize_multi_page(&[&Path]) -> FocrResult<String>`
- `OcrEngine::recognize_multi_page_with_model(model_path: &Path, images: &[&Path]) -> FocrResult<String>`
- `OcrEngine::recognize_multi_page_dynamic(Vec<image::DynamicImage>) -> FocrResult<String>`
- `OcrEngine::recognize_multi_page_dynamic_with_model(model_path: &Path, images:
  Vec<image::DynamicImage>) -> FocrResult<String>`
- `RecognizedDocument { markdown: String, layout: Vec<LayoutSpan> }`
- `LayoutSpan { label: String, boxes: Vec<[i64; 4]> }`
- `ExtractedFigure { index, label, bbox, markdown_ref, image }`
- `franken_ocr::model_arch::{registry, arch_by_id, default_arch}`
- `franken_ocr::native_engine::set_preprocess_overrides(PreprocessOverrides)`
  for crate-level preprocess policy, not per-request routing.

Current `bd-223.2` source also exposes runtime-control helpers:
`request_shutdown()`, `shutdown_requested()`, `reset_shutdown()`,
`cancel_checkpoint()`, `thread_budget()`, and `stream_pages()`. The shutdown
flag is process-global, matching the one-live-forward doctrine; do not present
it as a per-request cancellation token for concurrent tenants.

Current source after closed `bd-223.4` exposes
`franken_ocr::storage::{RunStore, RunRecord}` and `FOCR_RUN_STORE` for local run
telemetry and JSONL audit sync. This is useful for CLI telemetry and audit
tools, but it is not required for basic `OcrEngine` embedding.

Check `src/lib.rs` before relying on a signature. The project is still moving.

## Cargo Dependency

Path dependency from a sibling project:

```toml
[dependencies]
franken_ocr = { path = "../franken_ocr" }
```

Git dependency, only when the repo is published and pinned:

```toml
[dependencies]
franken_ocr = { git = "https://github.com/Dicklesworthstone/franken_ocr", rev = "<commit>" }
```

Use Rust nightly when building the project from source. The repo depends on
nightly-capable CPU-kernel work and Rust 2024 settings.

## Minimal Example

```rust
use std::path::Path;
use franken_ocr::OcrEngine;

fn main() -> franken_ocr::FocrResult<()> {
    let engine = OcrEngine::new()?;
    let text = engine.recognize(Path::new("invoice.png"))?;
    println!("{text}");
    Ok(())
}
```

Explicit model:

```rust
use std::path::Path;
use franken_ocr::OcrEngine;

fn run() -> franken_ocr::FocrResult<String> {
    let engine = OcrEngine::new()?;
    engine.recognize_with_model(
        Path::new("/opt/models/unlimited-ocr-int8.focrq"),
        Path::new("invoice.png"),
    )
}
```

Specialized GOT-OCR2 model:

```rust
use std::path::Path;
use franken_ocr::OcrEngine;

fn run_formula() -> franken_ocr::FocrResult<String> {
    let engine = OcrEngine::new()?;
    engine.recognize_with_model(
        Path::new("/opt/models/got-ocr2.int8.focrq"),
        Path::new("formula.png"),
    )
}
```

That example selects the GOT model, but it does not by itself select GOT's
structured `OCR with format:` prompt. In the current public library surface,
CLI-style `--task` is not a typed Rust API. Library integrations choose:

- explicit model path with `recognize_with_model` / `recognize_batch_with_model`,
- process environment `FOCR_GOT_FORMAT=1`, or
- source-internal `native_engine::force_got_format(true)` when working inside
  the crate.

Avoid toggling a process-global mode concurrently around a shared engine. If a
service needs both plain GOT OCR and formatted GOT output, isolate that policy at
the worker/process boundary until a richer public request type lands.

The same process-policy rule applies to preprocessing. The CLI maps
`--base-size`, `--image-size`, and `--crop-mode` to
`native_engine::PreprocessOverrides`; library users can set the same global with
`set_preprocess_overrides` before constructing/loading the engine. Use this for
dedicated workers or tests, not as a per-request toggle around a shared engine.
Default `base` is the certified path. Explicit Gundam tiling has first e2e
evidence (`bd-1e9n`) but still needs target-corpus proof before parity claims.

## Structured Layout Output

Use layout-aware methods when downstream code needs bounding boxes. Markdown and
layout are parsed from the same decoded model output, so they should not drift.

```rust
use std::path::Path;
use franken_ocr::{LayoutSpan, OcrEngine, RecognizedDocument};

fn recognize_boxes() -> franken_ocr::FocrResult<RecognizedDocument> {
    let engine = OcrEngine::new()?;
    let doc = engine.recognize_with_layout(Path::new("invoice.png"))?;

    for LayoutSpan { label, boxes } in &doc.layout {
        for [x1, y1, x2, y2] in boxes {
            eprintln!("{label}: ({x1},{y1})-({x2},{y2})");
        }
    }

    Ok(doc)
}
```

This is the library equivalent of `focr ocr --json` and `focr ocr -o out.json`.

For `--extract-figures` style integrations, use `recognize_with_figures` or
`recognize_dynamic_with_figures`. The engine returns crops as `ExtractedFigure`s;
the caller owns encoding them as PNG/JPEG and replacing each `markdown_ref` with
the final file path. The dynamic variant is the right primitive for PDF pages
because the page raster is already the crop source.

## Engine Lifetime

Use one long-lived engine per process or worker. `OcrEngine` owns the internal
runtime details and model cache. Creating it for every image creates avoidable
setup cost and can stress runtime/model lifecycle assumptions.

In current `bd-223.2` source, `OcrEngine` work can observe a process-global
shutdown flag through `cancel_checkpoint()` and return `FocrError::Cancelled`
at page or decode-step boundaries. Embedders may call `request_shutdown()` for a
whole-process abort and `reset_shutdown()` only when they intentionally keep the
process alive after a cancelled batch. Independent per-request cancellation is a
follow-up, not the current helper shape.

Recommended service shape:

```rust
use std::path::Path;
use std::sync::Arc;
use franken_ocr::{FocrResult, OcrEngine};

pub struct OcrService {
    engine: Arc<OcrEngine>,
}

impl OcrService {
    pub fn new() -> FocrResult<Self> {
        Ok(Self { engine: Arc::new(OcrEngine::new()?) })
    }

    pub fn recognize(&self, path: &Path) -> FocrResult<String> {
        self.engine.recognize(path)
    }
}
```

Do not add a second asupersync runtime inside calls. The library owns that.

## Async Hosts

The public API blocks. In async applications, isolate the blocking call at the
boundary your runtime provides. For example, a Tokio host can use
`spawn_blocking`, but should still reuse the same engine:

```rust
let engine = engine.clone();
let path = path.to_owned();
let result = tokio::task::spawn_blocking(move || engine.recognize(&path)).await??;
```

Do not fan out many concurrent OCR calls simply because the host is async. The
project doctrine is one live forward at a time, with math parallelism inside the
kernel/runtime path.
Use the current `thread_budget()` / `FOCR_THREADS` policy for host-wide capacity
reasoning when that source is present; do not size outer async concurrency from
`available_parallelism()`.

## Dynamic Images and PDFs

`OcrEngine::recognize` takes document image paths. CLI PDF routing is not hidden
inside that method. Library users who already decoded an image, or who want
native PDF support, should use the dynamic-image API:

```rust
use std::path::Path;
use franken_ocr::{FocrResult, OcrEngine, pdf::PdfPages};

fn recognize_pdf(path: &Path) -> FocrResult<String> {
    let engine = OcrEngine::new()?;
    let pages = PdfPages::open(path)?;
    let mut document = String::new();

    for idx in 0..pages.len() {
        let image = pages.render(idx)?;
        let page = engine.recognize_dynamic(image)?;
        if idx > 0 {
            document.push_str("\n\n");
        }
        document.push_str(page.trim_end());
    }

    Ok(document)
}
```

`PdfPages` is a scanned-PDF fast path. It renders one page at a time and returns
`FocrError::InputDecode` for unsupported `JPXDecode`, `JBIG2Decode`,
unsupported color spaces, or born-digital/vector pages with no image XObject.
Rasterize those PDFs out of band and pass page images if the pure-Rust fast path
does not cover them.

If a PDF integration needs per-page boxes, call
`recognize_dynamic_with_layout` for each rendered page and keep each page's
`layout` next to its page number. Do not try to recover boxes by parsing the
rendered markdown.

If it also needs figures, call `recognize_dynamic_with_figures` per page and
namespace filenames by page, mirroring the CLI's `page{N}_figure_{M}` behavior.

For one Unlimited-OCR cross-page document pass, collect page images and call
`recognize_multi_page_dynamic` / `recognize_multi_page_dynamic_with_model`
instead of looping `recognize_dynamic`. The result is one markdown document with
`<PAGE>` separators, not per-page `RecognizedDocument` layout data. This is the
library equivalent of CLI PDF `--multi-page` after rasterization.

## Batching

Batch API shape:

```rust
let inputs = [Path::new("p1.png"), Path::new("p2.png")];
let refs: Vec<&Path> = inputs.iter().copied().collect();
let results = engine.recognize_batch(&refs)?;

for item in results {
    match item {
        Ok(markdown) => println!("{markdown}"),
        Err(err) => eprintln!("page failed: {err}"),
    }
}
```

Interpretation:

- Outer `Err` means setup/model/global failure.
- Inner `Err` means that specific image failed.
- Batch APIs accept image paths; they do not perform the CLI PDF routing.
- Do not flatten this into all-or-nothing unless product requirements demand it.

Dense batch-source boundary:

- Committed source proves the default Unlimited-OCR spine separately from the
  GOT dense `recognize_batch_dense_got` path; treat those as distinct evidence
  families.
- Closed `bd-3jo6.1.7.5` now proves the dense zoo path separately as well:
  `OcrModel::recognize_batch_dense` routes `got-ocr2|smolvlm2|onechart`, with
  `smolvlm2::recognize_batch`, `onechart::recognize_batch`,
  `generate_greedy_batched(..., caps: &[usize], ...)`, and
  `PageStream::with_max_emit`. Library callers can treat this as current
  lossless batch behavior when `FOCR_BATCH_SPINE` is armed, while still probing
  the exact installed binary/source revision before promising it to users.
- Source at or after `d25dbd7` additionally makes GOT-OCR2
  `got::recognize_batch` hydrate SAM/projector/embed state once per batch.
  That is a setup/splice amortization inside the batch path; it does not change
  the API shape, PDF routing boundary, or the need to check the exact binary.
- `FOCR_BATCH_PACK` is an admission-order lever, not a semantic API: packed
  batches must restore result order and leave each page's token stream
  byte-identical.
- Throughput integration notes must stay scoped: SmolVLM2 and OneChart have
  decode-share self-relative wins, GOT is vision-dominated on the cited fixtures,
  and broad batched `lm_head` plus fairness-controlled rows remain follow-ups.

## Model Architectures

`franken_ocr::model_arch` is the library mirror of `focr models`: a registry of
model descriptors with id, display name, tasks, tokenizer/decoder/vision family,
default artifact basename, license notice, and `implemented()` / `status`
semantics. Use it for discovery and validation, not as proof that weights are
installed.

Important current distinction:

- `unlimited-ocr` is the ready/default model and is what `focr pull` normally
installs.
- `got-ocr2` is also ready/implemented in current source. `focr pull got-ocr2`
  installs `got-ocr2.int8.focrq` plus `qwen.tiktoken`. A self-converted GOT
  artifact must carry `model_id = "got-ocr2"` and have `qwen.tiktoken` beside
  the `.focrq`.
- Use `got-ocr2` for heavier specialized structured-output work: formulas,
  tables, charts, molecular SMILES, geometry, or sheet music. Do not swap it in
  as the default fast plain-text OCR model unless that tradeoff is intended.
- Default `OcrEngine` dispatch runs GOT through `format=false`. The CLI exposes
  `--format`, and the runtime also honors `FOCR_GOT_FORMAT` /
  `native_engine::force_got_format(true)` for GOT's `OCR with format: `
  Mathpix-Markdown mode. That switch is process-global; avoid changing it
  concurrently around shared engines.
- CLI `focr ocr --task formula|tables|chart|molecular|geometry|music` is a
  convenience layer over GOT model selection guidance plus `format=true`. It is
  not currently a library enum or per-call option.
- SmolVLM2 describe/VQA is gated by artifact and revision: current source
  routes an engine whose `.focrq` has `model_id=smolvlm2` through
  `src/native_engine/smolvlm2.rs`. `--question` maps to process-global
  `native_engine::set_smolvlm2_question`, and `FOCR_SMOLVLM2_QUESTION` is the
  env fallback. The route is implemented, but avoid concurrent per-request
  question changes around a shared engine because the question is
  process-global; attach DISC-003 plus exact C8/C10/A11 evidence to
  quality/perf claims.
- OneChart chart-data is implemented when the loaded `.focrq` has
  `model_id=onechart`: current `OcrEngine` dispatch reaches
  `forward_onechart`, which returns the repaired chart dict text and logs
  `reliable_distance` / `reliable` timing metadata. Use an explicit
  `onechart.int8.focrq` path with the OPT tokenizer files beside it; the file
  may come from `focr pull onechart` after `bd-av64.7` or from local conversion.
  D3 means
  `onechart_view_tensor` plus `vision_features` have seam proof; D4-prefill
  means `DecoderConfig::onechart` plus `build_inputs_embeds` can match
  last-position oracle logits; D4 cached decode means `generate_greedy_kvcache`
  and `opt_kvcache_matches_greedy_and_oracle` prove a bounded OPT decode seam;
  D5 means the native module assembles `json_text`, optional `pred_locs`,
  `reliable_distance`, and `reliable`; D6-D8 mean public route/e2e wiring is
  closed. Keep quality claims scoped to `bd-2lje`/A11 evidence.
- Polyphonic-TrOMR is a current runtime descriptor for single-staff and v1
  printed/scanned full-page OMR, with a packaged `f32` pull artifact after
  `bd-av64.7`. Current source can convert a WS-folded checkpoint to
  `tromr.focrq`, load the four WordLevel music tokenizer tables beside the
  artifact, run E3/E4 encoder+decoder code, merge streams with
  `merge_semantic`, serialize partwise MusicXML with `semantic_to_musicxml`, and
  dispatch through `forward_tromr` / `tromr::recognize_page` when the `.focrq`
  `model_id` is `tromr`. E5/E10 are closed for the v1 detector/runtime path.
  Closed `bd-av64.2` adds survivable full-page staff handling plus an API/CLI
  observability side channel: `MusicPageMeta` is stored on `OcrModel`,
  consumed through `OcrEngine::take_music_page_meta()`, and used by the CLI for
  robot `staff` events plus detection-ordered music JSON `staves`. It is a
  single-consumer "take" API for the most recent music forward, so multi-page
  PDF music currently exposes only the last page's staves through this channel.
  Do not translate that into TrOMR int8, camera dewarp, default/lossless
  barline quality, unconstrained quality/perf proof, a perf win/int8 perf row,
  or `**kern` export. `FOCR_TROMR_SPLIT=1` is separate experimental
  over-budget-staff rescue, not a library-wide quality guarantee. `bd-2sez` is
  a f32 baseline loss row, not proof that the native lane is faster than
  upstream torch.
  TrOCR and pix2tex remain roadmap descriptors unless source/tests prove
  otherwise.

Do not dispatch product behavior solely from a task name. Check the model row's
`implemented()` value, the artifact's `model_id`, and the exact binary/source
revision.

For GOT integrations, record the exact commit, `.focrq` path,
`qwen.tiktoken` hash, image fixture, task/format intent, and whether the run
used plain mode or `--format`/`FOCR_GOT_FORMAT`. Do not invent task-specific
downstream APIs until upstream exposes a richer per-call selector.

## Model Paths

Model resolution can come from:

1. Explicit path in `recognize_with_model`.
2. `FOCR_MODEL_PATH`.
3. `FOCR_MODEL_DIR` search paths and cache defaults populated by `focr pull`.

Fresh `focr pull` installs `unlimited-ocr.int8.focrq`. Current resolution also
checks quant-suffixed names (`unlimited-ocr.int8.focrq`,
`unlimited-ocr.int4.focrq`) when the default or bare `unlimited-ocr.focrq` name
is requested, and `FOCR_QUANT` can prefer a quant variant.
`focr pull got-ocr2` installs `got-ocr2.int8.focrq`; pass it explicitly through
`recognize_with_model` or set `FOCR_MODEL_PATH`.

Keep inference hosts offline by pre-populating model artifacts. Avoid network
acquisition from request handlers, jobs, or tests that are supposed to be
hermetic.

## Error Handling

Use the typed `FocrError` categories when exposed. Known user-facing classes map
to CLI exit codes:

- `ModelNotFound`
- `InputDecode`
- `Timeout`
- `Cancelled`
- `FormatMismatch`
- `NotImplemented`

Do not scrape human error messages if the typed value is available.

## Testing Integrations

Practical tests for downstream projects:

- Unit-test code paths with a wrapper trait around your own OCR boundary.
- Integration-test "model missing" behavior with `FOCR_MODEL_PATH=/nonexistent`.
- If real model artifacts are available, run one golden image test and assert
  stable structural output, not an overbroad byte-for-byte claim unless the
  upstream gate promises determinism.
- For robot consumers, test NDJSON parsing separately from OCR success.

## Production Checklist

- Pin the `franken_ocr` revision.
- Verify `focr robot schema` against your parser expectations.
- Install `.focrq` and tokenizer artifacts during deployment, not first request;
  use `focr pull` for the default model and `focr pull got-ocr2` for GOT.
- Set `FOCR_MODEL_PATH` or pass explicit model paths.
- Bound request time at the service layer and map timeout errors cleanly.
- Log model path, artifact hash when available, exit/error class, and elapsed
  time.
- Keep experimental lossy env vars off unless the project has recorded parity
  evidence for your exact workload.

## Anti-Patterns

| Anti-pattern | Risk |
|--------------|------|
| One engine per HTTP request | setup churn and runtime stress |
| Nested runtime per OCR call | deadlock/oversubscription risk |
| Concurrent outer page loop | conflicts with one-forward doctrine |
| Network pull in production inference | latency, outage, and hermeticity risk |
| Treating inner batch errors as impossible | hides partial failures |
| Expecting `OcrEngine::recognize(Path)` to route PDFs | use `PdfPages` + `recognize_dynamic`, or the CLI PDF router |
| Scraping boxes out of markdown | use `RecognizedDocument.layout` from layout-aware APIs |
| Expecting the engine to write figure files | `ExtractedFigure.image` is returned; the caller writes files and rewrites refs |
| Quantizing extra surfaces downstream | breaks parity contract |
