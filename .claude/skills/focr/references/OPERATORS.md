# focr Expert Operators

Use these as reusable moves when ordinary command lookup is not enough.

## Table of Contents

- [LC: Live Contract Probe](#lc-live-contract-probe)
- [OA: Offline Acquisition](#oa-offline-acquisition)
- [OE: One Engine](#oe-one-engine)
- [PQ: Parity First](#pq-parity-first)
- [RP: Robot Purity](#rp-robot-purity)
- [SQ: Stale Binary Quarantine](#sq-stale-binary-quarantine)
- [LQ: Lossy Lever Quarantine](#lq-lossy-lever-quarantine)
- [TI: Tracker-Informed Claim](#ti-tracker-informed-claim)

## LC: Live Contract Probe

Purpose: resolve source/help/schema disagreement.

Steps:

1. `git status --short --branch`
2. `rg -n "enum Commands|enum RobotCommands" src/cli.rs`
3. `rg -n "schema_version|run_start|run_error" src/robot.rs tests`
4. Run exact binary help if feasible.
5. Classify: current, stale binary, scaffolded, or unimplemented.

Output format:

```text
contract: current|stale-binary|scaffolded|unimplemented
evidence: <source/test/help commands>
next: <rebuild/run-source/file-bead/update-doc>
```

## OA: Offline Acquisition

Purpose: prepare inference without runtime network.

Steps:

1. Acquire source weights or run `focr pull` during setup.
2. Verify hashes/artifact metadata.
3. Set `FOCR_MODEL_PATH` in runtime environment.
4. Run `focr robot selftest`.
5. Disable network in an inference smoke test when possible.

Reject a deployment plan that downloads the model on first request.

## OE: One Engine

Purpose: keep Rust integration aligned with runtime doctrine.

Steps:

1. Create one `OcrEngine`.
2. Store it in application state.
3. Route requests through a blocking boundary if host is async.
4. Use batch APIs for multi-image work.
5. Avoid outer parallel page loops unless upstream exposes a safe policy.

Review smell: `OcrEngine::new()` appears inside a hot request handler.

## PQ: Parity First

Purpose: evaluate any optimization or quantization claim.

Steps:

1. Identify exact behavior surface.
2. Establish baseline output and metrics.
3. Change one lever.
4. Run parity/golden/CER evidence.
5. Keep only if the gate passes; otherwise revert or keep behind a kill switch.
6. Record accepted divergence or negative evidence in the project docs.

Never justify an OCR regression with throughput alone.

## RP: Robot Purity

Purpose: protect automation consumers.

Checks:

```bash
focr robot schema | jq .
focr robot run page.png | while IFS= read -r line; do
  printf '%s\n' "$line" | jq -e . >/dev/null || exit 1
done
```

Rules:

- stdout is JSON/NDJSON only,
- schema is versioned,
- exit code is checked,
- human messages go to stderr or human commands.

## SQ: Stale Binary Quarantine

Purpose: avoid false docs from old builds.

Trigger:

- help output conflicts with source,
- command missing from installed binary,
- behavior contradicts tests.

Action:

1. Mark binary stale in notes/final answer.
2. Use source and tests for truth.
3. Rebuild or run from source only if needed and feasible.
4. Do not edit docs to match stale output.

## LQ: Lossy Lever Quarantine

Purpose: contain experimental env vars and quantization changes.

Before enabling a lossy path, require:

- model artifact hash,
- corpus/image list,
- metric and allowed budget,
- deterministic fallback,
- env var list,
- Beads issue or evidence ledger.

If any item is missing, leave the lever off.

## TI: Tracker-Informed Claim

Purpose: answer capability questions honestly.

Steps:

1. Search source for the surface.
2. Search tests for proof.
3. Search `br` for the feature.
4. Use `bv --robot-triage` only for graph-level context.
5. If CASS has prior sessions, treat them as history, not live truth.

Final phrasing should distinguish:

- implemented and tested,
- implemented but not fully proven,
- scaffolded,
- planned,
- blocked.
