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

The contract is versioned. Always check schema before pinning a parser.

## Commands

```bash
focr robot schema
focr robot health
focr robot backends
focr robot selftest
focr robot run page.png
```

`robot run` emits lifecycle events while OCR work happens. Other robot commands
normally emit a single JSON object.

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
| `run_complete` | Successful completion |
| `run_error` | Structured failure |

The exact payload shape can evolve. Bind parsers to `robot schema`, not to a
memory of this file.

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
| 3 | model not found | run `focr pull`, set `FOCR_MODEL_PATH`, or pass `--model` |
| 4 | input decode | reject or rerasterize the input image |
| 5 | timeout | retry only if caller budget permits |
| 6 | cancelled | propagate cancellation |
| 7 | format mismatch | update artifact or converter |

Do not convert every nonzero code into a generic "OCR failed" bucket.

## Recovery Matrix

| Symptom | Likely cause | First move |
|---------|--------------|------------|
| `jq` fails on robot output | stale binary or human text leaked | rerun `robot schema`; inspect `src/robot.rs` |
| Missing `run_complete` | process failed or stream truncated | check exit code and last event |
| Exit 3 | no model artifact | set explicit model path or run `focr pull` |
| Exit 7 | `.focrq` version/hash issue | verify artifact and converter source |
| `robot selftest` missing | stale binary | run from current source |
| Unknown event | schema advanced | read `robot schema`, update parser and tests |

## Contract Tests

When changing robot behavior in `franken_ocr`, look for golden tests such as
`tests/cli_robot_golden.rs`. A good contract test proves:

- command exits with expected code,
- every stdout line is JSON,
- schema version is present,
- expected event names appear,
- human decoration is absent.

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
