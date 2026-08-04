# franken_ocr Library Integration

## Table of Contents

- [Public Shape](#public-shape)
- [Cargo Dependency](#cargo-dependency)
- [Minimal Example](#minimal-example)
- [Engine Lifetime](#engine-lifetime)
- [Async Hosts](#async-hosts)
- [Batching](#batching)
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
- `OcrEngine::recognize_batch(&[&Path]) -> FocrResult<Vec<FocrResult<String>>>`
- `OcrEngine::recognize_batch_with_model(&Path, &[&Path])`

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

## Engine Lifetime

Use one long-lived engine per process or worker. `OcrEngine` owns the internal
runtime details and model cache. Creating it for every image creates avoidable
setup cost and can stress runtime/model lifecycle assumptions.

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
- Do not flatten this into all-or-nothing unless product requirements demand it.

## Model Paths

Model resolution can come from:

1. Explicit path in `recognize_with_model`.
2. `FOCR_MODEL_PATH`.
3. `FOCR_MODEL_DIR` search paths and cache defaults populated by `focr pull`.

Keep inference hosts offline by pre-populating model artifacts. Avoid network
pulls from request handlers, jobs, or tests that are supposed to be hermetic.

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
- Install `.focrq` and tokenizer artifacts during deployment, not first request.
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
| Quantizing extra surfaces downstream | breaks parity contract |
