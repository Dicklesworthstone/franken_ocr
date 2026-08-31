# Robot Mode Reference

## Table of Contents

- [Purpose](#purpose)
- [Commands](#commands)
- [NDJSON Rules](#ndjson-rules)
- [Event Types](#event-types)
- [Parser Pattern](#parser-pattern)
- [Exit Codes](#exit-codes)
- [Recovery Matrix](#recovery-matrix)
- [Contract Tests](#contract-tests)
- [Do Not](#do-not)

## Purpose

Robot mode is the agent-first interface for `focr`. It is for scripts, agents,
CI, and services that need stable machine-readable output. Human decoration
belongs in human mode only.

The contract is versioned by `ROBOT_SCHEMA_VERSION`. Always check schema before
pinning a parser.

## Commands

```bash
focr robot schema
focr robot health
focr robot backends
focr robot triage
focr robot run page.png
focr robot run --model got-ocr2.int8.focrq --format formula.png
focr robot run --task tables --model got-ocr2.int8.focrq table.png
focr robot run --task describe --model smolvlm2.int8.focrq --question "What is on the shelf?" photo.jpg
focr robot run --pages 3,5-9 --split-spreads scan.pdf
```

`robot run` emits lifecycle events while OCR work happens. Other robot commands
normally emit a single JSON object. `robot run` shares `ocr` request flags,
including GOT-OCR2 `--format`, `--task`, crop/decode tuning flags, `--model`,
and SmolVLM2 `--question`; it must still emit only NDJSON events. GOT
structured tasks (`formula`, `tables`, `chart`, `molecular`, `geometry`,
`music`) imply GOT format mode and should fail with a usage error before weights
load if the caller would otherwise use the default plain-text model. SmolVLM2
`describe` is different: it must not imply GOT `--format`, it requires a
SmolVLM2 artifact/model spec, and it may carry `--question`.
PDF `--pages` and `--split-spreads` are accepted by `robot run` through the
shared OCR request path. `--pages` is 1-based PDF page selection; split-spread
events should preserve source page identity and expose left/right logical half
metadata rather than silently pretending they were original pages.

`robot triage` emits one JSON object, not NDJSON. It is the agent-first
orientation command from closed `bd-wp8.7`: `schema_version`, `command:
"robot.triage"`, `quick_ref`, `health`, `recommendations`, `commands`, and
`exit_codes`. Use it when an agent needs a one-round-trip answer to "what should
I do next?" or "which setup command is missing?" For example, a missing model
should produce a pull recommendation before OCR commands; a ready model should
include runnable OCR commands.

Use `robot backends` before performance or host-capability claims. It reflects
detected/selected SIMD tiers, logical CPU count, and related backend facts
without loading OCR weights. In current `bd-223.2` source it also reports
`threads`, the resolved `FOCR_THREADS` / physical-core runtime budget; `robot
health` reports the same budget. Use `robot selftest` when the claim is
stronger: "the selected int8 kernel on this host matches the scalar oracle." Do
not substitute `health` for either one; `health` is a cache/config readiness
probe. At/after `ad3ad20`, selftest also includes a machine-readable `models`
rollup for `unlimited-ocr`, `got-ocr2`, `smolvlm2`, and `onechart`; every
verdict is derived from the case rows for that model. TrOMR is absent because
its published int8 artifact is storage-only and runtime dequants through f32
accessors rather than int8 decoder kernels.

`FOCR_FORCE_ARCH=scalar|sdot|smmla|avx2|avxvnni|avx512vnni` can force an
available tier for backend/selftest/perf diagnosis. Unsupported forced tiers are
not support evidence. Apple Silicon currently prefers SDOT over SMMLA; non-Apple
aarch64 may prefer SMMLA; x86 tiers are AVX2, AVX-VNNI, and AVX-512-VNNI. AMX
is not current unless `robot backends` starts advertising it. A `selftest` pass
is parity evidence, not a timing or throughput claim.

## NDJSON Rules

For `robot run`:

- stdout is newline-delimited JSON.
- Each line must parse independently.
- Consumers should process events incrementally.
- stderr is for diagnostics only and must not be mixed into the event stream.
- Do not expect one giant JSON document.

Parser assumption:

```text
for each stdout line:
  parse JSON object
  inspect schema_version and event/type
  update state machine
```

## Event Types

Current source wires schema version 1 and the following event names:

| Event | Purpose |
|-------|---------|
| `run_start` | Run metadata and start signal |
| `stage` | Progress/stage transition |
| `page` | Per-page/page-level result signal |
| `staff` | TrOMR full-page music staff result, status `ok` or `skipped` |
| `music_warning` | TrOMR annotate-only musical-sanity observation |
| `run_complete` | Successful completion |
| `run_error` | Structured failure |

The exact payload shape can evolve. Bind parsers to `robot schema`, not to a
memory of this file.
For split-spread PDFs, expect `page` events to carry enough page/half metadata
to distinguish original page `N` left/right logical pages. Do not collapse those
events into a flat page counter if the caller needs traceability.
For TrOMR music runs, expect additive schema-v1 `staff` events shaped around
`{staff, total, bbox, status, reason?}`: `staff` is 1-based, `bbox` is
page-space `[x, y, w, h]`, `status` is `ok` or `skipped`, and `reason` is present
only for skips. Do not wait for `staff_detection` / `staff_result`; those are
stale bead acceptance names, not emitted event kinds.
Also expect additive schema-v1 `music_warning` events after `bd-av64.5` when the
recognized music has sanity observations; source at/after `0b74af0` advertises
that event in `tests/fixtures/robot_schema_v1.json` and
`tests/cli_robot_golden.rs`. Payload shape is
`{kind, part, measure, detail}` where `kind` is one of
`overfull_bar`, `underfull_bar`, `impossible_duration`, or `key_mismatch`;
`measure: 0` means staff-level. Treat these as telemetry/annotations, not as
automatic correction or proof that the output is musically correct.

## Parser Pattern

Shell validation:

```bash
set -o pipefail
focr robot run page.png \
  | tee run.ndjson \
  | while IFS= read -r line; do
      printf '%s\n' "$line" | jq -e . >/dev/null || exit 1
    done
```

Rust parser sketch:

```rust
for line in stdout.lines() {
    let value: serde_json::Value = serde_json::from_str(line?)?;
    let version = value.get("schema_version").and_then(|v| v.as_u64());
    if version != Some(1) {
        anyhow::bail!("unsupported focr robot schema: {version:?}");
    }
    match value.get("event").and_then(|v| v.as_str()) {
        Some("run_start") => {}
        Some("stage") => {}
        Some("page") => {}
        Some("run_complete") => {}
        Some("run_error") => {}
        other => anyhow::bail!("unknown focr event: {other:?}"),
    }
}
```

Prefer typed structs in production once schema is pinned by tests.

## Exit Codes

| Code | Meaning | Automation response |
|------|---------|---------------------|
| 0 | success | consume complete output |
| 1 | generic or not implemented | inspect structured error; update Bead if phase gap |
| 2 | usage | fix caller arguments |
| 3 | model not found | set `FOCR_MODEL_PATH`, `FOCR_MODEL_DIR`, or pass `--model` |
| 4 | input decode | reject or rerasterize the input image |
| 5 | timeout | retry only if caller budget permits |
| 6 | cancelled | propagate cancellation |
| 7 | format mismatch | update artifact or converter |

Do not convert every nonzero code into a generic "OCR failed" bucket.
In current `bd-223.2` source, Ctrl+C sets cooperative cancellation and returns
`Cancelled` / exit 6 at the next `cancel_checkpoint()` boundary; a second
Ctrl+C hard-exits 130 for a wedged stage. If a binary lacks this behavior,
classify it as pre-`bd-223.2`, stale, or release-lagged before changing docs.

## Recovery Matrix

| Symptom | Likely cause | First move |
|---------|--------------|------------|
| `jq` fails on robot output | stale binary or human text leaked | rerun `robot schema`; inspect `src/robot.rs` |
| Missing `run_complete` | process failed or stream truncated | check exit code and last event |
| Exit 3 | no model artifact | run `focr pull`, set explicit model path, or set `FOCR_MODEL_DIR` |
| Exit 4 | unsupported/bad image input | inspect image decode path and rerasterize upstream |
| Exit 6 | cooperative cancellation | propagate abort; do not retry as a model failure |
| Exit 7 | `.focrq` version/hash issue | verify artifact and converter source |
| `robot selftest` missing | stale binary | inspect `src/cli.rs` and run from source |
| `threads` missing from `robot backends` | pre-bd-223.2 binary or stale source | inspect `thread_budget` / `robot_backends_payload` |
| `robot triage` missing | pre-bd-wp8.7 binary or stale source | inspect `robot_triage_payload`; run from source if needed |
| `robot triage` recommends OCR before pull on an empty cache | regression in agent ergonomics | inspect `tests/agent_ergonomics_regression.rs` |
| Split-spread robot output loses original page/half | PDF event regression | inspect `split_spread` and robot page payload tests |
| TrOMR music run emits no `staff` events | stale binary, non-TrOMR route, missing artifacts, or no full-page music path | rerun `focr robot schema`; inspect `src/robot.rs::EVENT_KINDS` and `src/cli.rs` `take_music_page_meta` |
| Expected TrOMR warnings do not appear | stale binary, non-TrOMR route, warning-free output, or pre-bd-av64.5 source | inspect `src/robot.rs::music_warning_event`, `tromr::sanity_warnings`, and JSON `warnings` |
| Robot schema golden fails after `staff` appears | stale binary/source, stale `bd-wp8.2.2` tracker state, or a real fixture diff | at/after `adb4ee6`, `staff` should already be in `tests/fixtures/robot_schema_v1.json` and advertised-events assertion; rerun focused golden tests before changing schema version |
| Backend/SIMD claim needed | use `robot backends` first | then run `robot selftest` for parity |
| Need model-specific SIMD parity | inspect `robot selftest.models` | TrOMR absent means storage-int8 dequants through f32 accessors, not a missing schema row |
| Speedup claim cites only `robot selftest` | missing perf proof | use OP-GB and PERF_LEDGER/gauntlet rows |
| Unknown event | schema advanced | read `robot schema`, update parser and tests |

## Contract Tests

When changing robot behavior in `franken_ocr`, look for golden tests such as
`tests/cli_robot_golden.rs`. A good contract test proves:

- command exits with expected code,
- every stdout line is JSON,
- schema version is present,
- expected event names appear,
- human decoration is absent.
- host-dependent fields such as `logical_cpus` and current `bd-223.2` `threads`
  are scrubbed in golden tests.
- `robot triage` has self-describing schema, quick_ref, recommendations,
  commands, exit code docs, and stdout purity.

For downstream consumers, keep a fixture generated from the pinned `focr`
revision and test parser rejection on unknown schema versions.

## Do Not

| Anti-pattern | Why |
|--------------|-----|
| Parse robot stdout as one JSON document | `run` is NDJSON |
| Accept unversioned events | breaks forward compatibility |
| Read human stderr for business logic | diagnostic text is unstable |
| Ignore exit code after parsing stdout | stream may end with failure |
| Add progress prose to robot stdout | corrupts automation |
| Treat `robot triage` as human help text | it is a JSON contract for agents |
| Flatten split-spread page events without source/half metadata | loses auditability for scanned book spreads |
