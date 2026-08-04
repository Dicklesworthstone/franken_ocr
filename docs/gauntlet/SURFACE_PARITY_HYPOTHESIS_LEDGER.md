# Surface Parity Hypothesis Ledger

## Experiment `EXP-1404` - Certification evidence is independently recomputable

| field | value |
|---|---|
| `experiment_id` | `EXP-1404` |
| `pillar` | `surface` |
| `created_at_utc` | `2026-07-10T08:55:00Z` |
| `created_by_agent` | `BrownFox` |
| `bead_id` | `bd-2mo.30.1` |
| `parent_hypothesis_id` | `PASS-14-GAUNTLET-INTEGRITY` |
| `status` | `CLOSED` |

### Hypothesis

Every release-performance and correctness claim can be recomputed from bounded,
hash-bound raw evidence tied to the exact source, binary, fixture, and reference
model, with contradictory aggregates rejected fail-closed.

### Motivation

Fresh-eyes tracing found that internally consistent aggregate JSON could disagree
with raw timings or unbound CER texts while still satisfying the old verifier.

### Minimal Reproducer

Mutate one raw timing, one aggregate sample, one CER input, one binary receipt,
and one reference-model source hash independently in otherwise valid fixtures.

### Expected Signal

Every mutation produces a named red reason; the unmodified producer output
remains usable and recomputes to the exact certified values under bounded reads.

### Falsifiability Criteria

Any contradictory or unbound fixture that remains eligible, any pathname escape
or unbounded evidence read, or a verifier-only schema with no usable producer
path falsifies the integrity claim.

### One-Line Invocation

`python3 scripts/gauntlet_cert.py --self-test`

### Results Inline

```yaml
result_status: NO_EVIDENCE
result_summary: the ten-input producer-to-certificate harness now independently replays bounded timing, CER, binary, build, workspace, local-dependency, Cargo-config, and reference bytes while rejecting every named mutation, but existing release bundles lack the new receipts and remain deliberately ineligible
result_evidence_paths:
  - artifacts/perf/bd-2mo.30/gauntlet-integrity-audit
closed_at_utc: 2026-07-10T11:23:40Z
retry_condition_predicate: blocked until the canonical producer runs from a clean current HEAD and creates current build, source-pack, binary, reference, inference, timing, and CER receipts for row generation and release readiness
```

### Closure Predicate

Keep the experiment open until adversarial producer-to-certificate self-tests
cover every named mutation and current release readiness remains red unless all
new receipts are present and current.
