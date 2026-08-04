# focr CLI Reference

## Table of Contents

- [Contract](#contract)
- [Binary Selection](#binary-selection)
- [Commands](#commands)
- [OCR Command](#ocr-command)
- [Batch OCR](#batch-ocr)
- [Model Pull](#model-pull)
- [Conversion](#conversion)
- [Robot Commands](#robot-commands)
- [Scaffolded Commands](#scaffolded-commands)
- [Examples](#examples)
- [Validation](#validation)

## Contract

`focr` is the short binary for `franken_ocr`. The long binary
`franken_ocr` exists too. Both are thin shims over `franken_ocr::cli_main()`;
when behavior differs, the binary is stale or the source is dirty.

Inference is local after model acquisition. The normal runtime should not need
network access.

## Binary Selection

Prefer the exact binary path used by the system under test.

```bash
command -v focr
focr --version
focr --help
```

When working from source:

```bash
cd /Users/jemanuel/projects/franken_ocr
cargo run --bin focr -- --help
```

Stale-binary warning signs:

- Source contains `ocr-batch`, `pull`, or `robot selftest`, but help does not.
- Installed `focr` is missing while repo builds.
- A target-dir binary prints an older command set after source changes.

Do not update this skill from stale help. Inspect `src/cli.rs` first.

## Commands

| Command | Status | Purpose |
|---------|--------|---------|
| `ocr <image>` | implemented path | OCR one document image |
| `ocr-batch <images...>` | implemented path | OCR multiple document images |
| `convert` | int8 implemented | Convert Baidu safetensors/tokenizer into `.focrq` |
| `pull` | implemented path | Download and verify packaged model artifacts |
| `robot schema` | implemented path | Emit robot schema metadata |
| `robot health` | implemented path | Machine-readable health probe |
| `robot backends` | implemented path | Emit backend availability/capability data |
| `robot selftest` | implemented path | Run fast machine-readable self-check |
| `robot run <image>` | implemented path | OCR with NDJSON lifecycle events |
| `doctor` | scaffolded | Human diagnostics; may be incomplete |
| `runs` | scaffolded | Run history; may be incomplete |
| `sync` | scaffolded | Persistence/sync surface; may be incomplete |

Treat scaffolded commands as unstable until source and tests prove behavior.

## OCR Command

Typical use:

```bash
focr ocr page.png
focr ocr page.png --json
focr ocr page.png --model /path/to/model.focrq
```

Important flags:

| Flag | Use |
|------|-----|
| `--json` | Return structured JSON instead of human markdown/text |
| `--robot` | Emit NDJSON robot events |
| `--model <path>` | Use explicit `.focrq` or model artifact path |
| `--base-size`, `--image-size`, `--crop-mode` | Match reference image preprocessing knobs |
| `--max-length`, `--temperature`, `--no-repeat-ngram`, `--ngram-window` | Decode controls |

Timeouts are controlled by stage-budget env vars such as
`FOCR_STAGE_BUDGET_FORWARD_MS`; there is no `ocr --timeout` flag in the current
source.

Input scope for v1 is document images: PNG, JPG, and similar raster image
formats. PDFs are rasterized out of band before `focr` sees them.

## Batch OCR

```bash
focr ocr-batch --json page-1.png page-2.png page-3.png
```

Use batch mode when the caller has multiple document images and wants one setup
cost. Batch behavior may preserve per-page failure information rather than
aborting the entire batch for every input failure. Confirm exact JSON shape from
the current source or a golden test before binding a production parser.

Useful batch env vars seen in source:

| Env var | Meaning |
|---------|---------|
| `FOCR_BATCH_SPINE` | Select or force the batch spine path when supported |
| `FOCR_BATCH_SIZE` | Tune batch sizing when supported |

Do not invent concurrency around batch mode. Let the engine own its runtime and
kernel fanout.

## Model Pull

```bash
focr pull
focr pull --quant int8 --json
focr pull --manifest ./manifest.json
```

`pull` downloads the packaged int8 `.focrq` model and tokenizer, validates
hashes, and writes into the local cache. It is the sanctioned networked setup
step. Inference should use cached artifacts.

Useful env vars:

| Env var | Meaning |
|---------|---------|
| `FOCR_MODEL_DIR` | Extra model search path for inference resolution |
| `FOCR_MODEL_PATH` | Exact model path for inference |
| `FOCR_MANIFEST_URL` | Override manifest URL for pull |

On Windows x86_64, OCR support may be proven while `pull` still has a network
gap. Work around by copying cache artifacts or passing `--model`.

## Conversion

```bash
focr convert \
  /models/Unlimited-OCR/model.safetensors \
  -o /models/franken_ocr/unlimited-ocr-int8.focrq \
  --quant int8
```

Conversion facts:

- `.focrq` format version 1 uses magic `FOCRQ\0`.
- The file carries source sha256 metadata and Baidu MIT notice text.
- `int8` is implemented for the validated conversion lane.
- `int4` is not a casual option; it is phase-gated behind evidence.
- Do not convert or redistribute artifacts without preserving required license
  metadata.

## Robot Commands

```bash
focr robot schema | jq .
focr robot health | jq .
focr robot backends | jq .
focr robot selftest | jq .
focr robot run page.png | jq -c .
```

Robot mode is for agents and automation. Consume stdout as line-oriented NDJSON
when using `run`. Never rely on human prose in robot mode.

See ROBOT.md for event and parser rules.

## Scaffolded Commands

`doctor`, `runs`, and `sync` may exist before they are fully useful. If a task
depends on one of them:

1. Inspect `src/cli.rs` and tests.
2. Run the command against the exact binary.
3. Treat `NotImplemented` exit code 1 as a truthful phase signal.
4. File or update a Beads issue if the missing behavior blocks the task.

## Examples

Human OCR:

```bash
focr pull
focr ocr receipt.jpg > receipt.md
```

Automation OCR:

```bash
set -o pipefail
focr robot run receipt.jpg | tee receipt.ndjson | jq -c .
```

Explicit offline model:

```bash
FOCR_MODEL_PATH=/opt/models/unlimited-ocr-int8.focrq focr ocr receipt.jpg --json
```

Source truth probe:

```bash
rg -n "enum Commands|enum RobotCommands|struct OcrArgs|struct ConvertArgs" src/cli.rs
```

## Validation

Before documenting a new CLI behavior, run:

```bash
focr --help
focr robot schema | jq .
focr robot selftest | jq .
```

If the binary cannot be trusted, cite the source lines and say the live binary
was stale instead of pretending help output is current.
