# Artifacts and Environment

## Table of Contents

- [Model Artifact Model](#model-artifact-model)
- [Cache and Paths](#cache-and-paths)
- [`.focrq` Format](#focrq-format)
- [Conversion Lane](#conversion-lane)
- [Runtime Env Vars](#runtime-env-vars)
- [Experimental Env Vars](#experimental-env-vars)
- [Offline Deployments](#offline-deployments)
- [Platform Notes](#platform-notes)
- [License Notes](#license-notes)

## Model Artifact Model

`franken_ocr` targets Baidu Unlimited-OCR. Real inference needs either:

- the source Baidu safetensors weights for conversion, or
- a packaged `.focrq` artifact and tokenizer/cache state for runtime use.

The project intends no general ML framework at inference time. Model artifacts
are local files after setup.

## Cache and Paths

Common path controls:

| Env var | Use |
|---------|-----|
| `FOCR_MODEL_PATH` | Exact model file to load |
| `FOCR_MODEL_DIR` | Extra model search path for inference resolution |
| `FOCR_MANIFEST_URL` | Pull manifest override |

Resolution guidance:

1. Use explicit CLI `--model` or library `recognize_with_model` for tests.
2. Use `FOCR_MODEL_PATH` in deployments.
3. Use cache defaults for local development after `focr pull`.

Do not hide model resolution failures. Exit code 3 is actionable.

## `.focrq` Format

Current format facts from source/docs:

- format version: 1
- magic: `FOCRQ\0`
- stores quantized model payload and metadata
- records source sha256 information
- carries required Baidu MIT notice metadata

Format mismatch maps to exit code 7. A format mismatch is not a retryable OCR
failure; update the artifact or binary.

## Conversion Lane

Implemented:

```bash
focr convert model.safetensors -o model.focrq --quant int8
```

Current policy:

- Decoder FFN/expert GEMMs are the validated quantization surface.
- Vision tower, projector, embeddings, router gate, and norms stay high
  precision unless a future gate proves otherwise.
- `int4` is not an available default; it requires separate parity evidence.

When documenting conversion results, include source sha256, output path, quant
mode, binary revision, and validation command.

## Runtime Env Vars

User-facing or operationally relevant env vars seen in source:

| Env var | Use |
|---------|-----|
| `FOCR_MODEL_PATH` | exact model artifact |
| `FOCR_MODEL_DIR` | extra model search path |
| `FOCR_MANIFEST_URL` | pull manifest URL |
| `FOCR_NO_REPEAT_NGRAM` | decoding repetition control |
| `FOCR_FORCE_ARCH` | force architecture/backend probe path |
| `FOCR_STAGE_BUDGET_FORWARD_MS` | stage budget/timing control |
| `FOCR_BATCH_SPINE` | batch path selection |
| `FOCR_BATCH_SIZE` | batch sizing |
| `FOCR_TIMING` | timing output/control |

Always verify current names in `src/cli.rs`, `src/dist.rs`, and related modules.

## Experimental Env Vars

Treat these as gated development levers, not production knobs:

| Env var family | Caution |
|----------------|---------|
| `FOCR_INT8_KV` | attention/KV quantization can affect OCR output |
| `FOCR_ATTN_GEMM` | attention kernel substitution needs parity proof |
| `FOCR_SPEC_DECODE` | speculative decode must be bit-identical or gated |
| fusion/tile/internal kernel toggles | benchmark only with parity evidence |

If enabling an experimental path, record:

- exact env vars,
- model artifact hash,
- corpus/images,
- expected loss or CER/TEDS budget,
- fallback trigger,
- Beads issue or evidence ledger.

## Offline Deployments

Recommended:

1. Build/pin `focr` and the `franken_ocr` library revision.
2. Run `focr pull` or `focr convert` during image/build preparation.
3. Copy `.focrq` and tokenizer/cache artifacts into the target.
4. Set `FOCR_MODEL_PATH`.
5. Run `focr robot selftest`.
6. Disable network during inference tests.

Never make the first production OCR request responsible for downloading the
model.

## Platform Notes

The project prioritizes CPU:

- Apple Silicon / ARM64: NEON, dotprod, i8mm lanes.
- x86-64: AVX2, AVX-VNNI, AVX-512-VNNI, and AMX where applicable.

Windows x86_64 support may be proven for OCR while `focr pull` still has a
network transport gap. ARM64 Windows should not be claimed unless current
Beads/source evidence says so.

## License Notes

Baidu Unlimited-OCR is MIT licensed. Packaged derivative artifacts must preserve
the required notice. Do not strip license metadata from `.focrq` conversion or
release workflows.
