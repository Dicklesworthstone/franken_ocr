---
name: focr
description: >-
  Use focr/franken_ocr OCR CLI, robot NDJSON, model artifacts, and Rust
  OcrEngine. Use when running focr, embedding OcrEngine, or editing franken_ocr.
dependencies:
  - "franken_ocr repo or installed focr binary"
  - "Rust nightly when building from source"
  - "jq for robot/JSON inspection"
  - "Baidu Unlimited-OCR source weights or a .focrq model for real inference"
---

# focr / franken_ocr

## One Rule

Correctness gates outrank throughput. For real OCR behavior, prefer source,
parity fixtures, and Beads evidence over optimistic README wording or stale
binaries.

## Truth Stack

Use this order when facts conflict:

1. Current `franken_ocr` source and tests.
2. Live help from the exact binary that will run, after confirming it is not
   stale.
3. `README.md`, `COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`, and docs.
4. `br`/`bv` issue evidence.
5. `cass` session history.

If a command exists in source but not in `focr --help`, treat the binary as
stale and run from source or rebuild before drawing conclusions.

## Fast Probe

```bash
cd /Users/jemanuel/projects/franken_ocr
git status --short --branch
rg -n "enum Commands|enum RobotCommands|pub struct OcrEngine|pub fn recognize" src
cargo run --bin focr -- --help   # if live help is needed and build cost is acceptable
```

For an installed binary:

```bash
command -v focr && focr --version && focr --help
```

## Common Workflows

### Acquire a model for real OCR

```bash
focr pull
focr robot health | jq .
```

`pull` downloads the int8 `.focrq` model plus tokenizer into the local cache,
verifies hashes, and is the only normal networked inference-prep step.

### OCR one image

```bash
focr ocr invoice.png --json
focr ocr invoice.png --model ~/.cache/franken_ocr/models/model.focrq
```

Plain human mode writes markdown/text. `--json` writes machine-readable JSON.
Use `--robot` when the caller consumes NDJSON lifecycle events.

### OCR a batch

```bash
focr ocr-batch --json page-1.png page-2.png page-3.png
```

The library result shape is `Result<Vec<Result<String>>>`: outer errors mean
setup/model failure; inner errors are per-image failures.

### Robot integration

```bash
focr robot schema | jq .
focr robot health | jq .
focr robot selftest | jq .
focr robot run invoice.png | jq -c .
```

Robot output must stay line-oriented, versioned, and free of human decoration.

### Convert source weights

```bash
focr convert \
  /models/Unlimited-OCR/model.safetensors \
  -o ~/.cache/franken_ocr/models/unlimited-ocr-int8.focrq \
  --quant int8
```

Only `int8` conversion is implemented. `int4` is intentionally phase-gated.

## Library Integration

```rust
use std::path::Path;
use franken_ocr::OcrEngine;

fn main() -> franken_ocr::FocrResult<()> {
    let engine = OcrEngine::new()?;
    let markdown = engine.recognize(Path::new("invoice.png"))?;
    println!("{markdown}");
    Ok(())
}
```

Create one long-lived `OcrEngine` per process. It is synchronous and blocking;
wrap it at the service boundary if your application is async. Do not create one
engine per request, do not nest runtimes around it, and do not run multiple live
forwards against one model for throughput experiments until the project exposes
an explicit safe policy.

## Stable Exit Codes

| Code | Meaning |
|------|---------|
| 0 | success |
| 1 | generic failure or not implemented |
| 2 | usage |
| 3 | model not found |
| 4 | input decode |
| 5 | timeout |
| 6 | cancelled |
| 7 | format mismatch |

## Development Rules in `franken_ocr`

1. Read `AGENTS.md`, `README.md`, and for kernel/model work
   `COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`.
2. Use `br`/`bv` in robot or JSON modes only: `br ready --json`,
   `br show <id> --json`, `bv --robot-triage`.
3. Never run bare `bv`; never assume `br` commits anything.
4. For substantive source changes, run `scripts/check.sh` or the equivalent
   `cargo fmt --check`, `cargo check --all-targets`,
   `cargo clippy --all-targets -- -D warnings`, `cargo test`, then `ubs`.
5. Respect unresolved `[OPEN]`/OQ gates before kernels or lossy optimizations.

## Do Not

| Anti-pattern | Correct move |
|--------------|--------------|
| Trust an old installed binary | Probe source and rebuild/run from source |
| Parse robot output as one JSON document | Consume NDJSON line by line |
| Enable lossy experimental env vars casually | Require parity/CER evidence and a kill switch |
| Use `focr pull` during inference | Pull once, then run offline |
| Create one `OcrEngine` per OCR request | Reuse one engine and batch where appropriate |
| Treat int4 as available | Keep it phase-gated until source says otherwise |

## Reference Index

Open only what the task needs:

| Need | Reference |
|------|-----------|
| CLI commands and examples | [CLI.md](references/CLI.md) |
| Embedding `franken_ocr` in Rust | [LIBRARY.md](references/LIBRARY.md) |
| Robot NDJSON and automation | [ROBOT.md](references/ROBOT.md) |
| Models, `.focrq`, env vars | [ARTIFACTS-AND-ENV.md](references/ARTIFACTS-AND-ENV.md) |
| Source-development workflow | [DEVELOPMENT.md](references/DEVELOPMENT.md) |
| Current Beads/BV reality | [BEADS-REALITY.md](references/BEADS-REALITY.md) |
| Failure diagnosis | [TROUBLESHOOTING.md](references/TROUBLESHOOTING.md) |
| Reusable expert operators | [OPERATORS.md](references/OPERATORS.md) |
| Research notes behind this skill | [RESEARCH.md](references/RESEARCH.md) |

## Validate This Skill

```bash
.claude/skills/focr/scripts/validate.py .claude/skills/focr
```
