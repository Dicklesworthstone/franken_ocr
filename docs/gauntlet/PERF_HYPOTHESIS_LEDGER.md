# Performance Hypothesis Ledger

## Experiment `EXP-1402` - Whole-program PGO over the representative corpus

| field | value |
|---|---|
| `experiment_id` | `EXP-1402` |
| `pillar` | `perf` |
| `created_at_utc` | `2026-07-10T07:37:36Z` |
| `created_by_agent` | `BrownFox` |
| `bead_id` | `bd-2mo.23` |
| `parent_hypothesis_id` | `PASS-12-PGO` |
| `status` | `CLOSED` |

### Hypothesis

Whole-program profile guidance would improve both dense decode and the capped
ten-page wall time without moving any numerical or dispatch contract.

### Motivation

The decode loop contains branch-heavy routing and ring-buffer control whose
layout can improve without runtime complexity or model-specific numerics.

### Minimal Reproducer

Compare ordinary `release-perf` and PGO-use binaries in a reversed/interleaved
page workload and a separately measured capped ten-page workload.

### Expected Signal

At least a two-percent win in both decode and capped ten-page wall time, stable
CV, byte-identical page output, and semantic identity after stripping timings.

### Falsifiability Criteria

Failure of either precommitted workload gate, any output drift, or a greater than
two-percent regression in a major phase rejects the whole-program profile.

### One-Line Invocation

`RUSTFLAGS='-Cprofile-use=focr.profdata -Cllvm-args=-pgo-warn-missing-function' cargo build --profile release-perf --bin focr`

### Results Inline

```yaml
result_status: NO_EVIDENCE
result_summary: decode improved 14.52 percent but capped ten-page wall improved only 1.14 percent while prefill regressed about 56 percent, so the PGO build was rejected
result_evidence_paths:
  - artifacts/perf/bd-2mo.30/profile-recipe-5733407/pass12-pgo-20260710T073736Z
closed_at_utc: 2026-07-10T08:10:00Z
retry_condition_predicate: reconsider only inside the broader phase-selective or prefill-weighted PGO experiment with the same two-workload gate
```

### Closure Predicate

Close only when both workloads, output identity, variance, provenance, and the
explicit retry condition are present in a hash-verified evidence directory.

## Proposed Experiment `PROPOSAL-ALIEN-01` - Proof-carrying fixed-shape expert microkernel

| field | value |
|---|---|
| `proposal_id` | `PROPOSAL-ALIEN-01` |
| `proposal_pillar` | `perf` |
| `proposed_at_utc` | `2026-07-11T19:25:00Z` |
| `proposed_by_agent` | `NavyTiger` |
| `target_bead_id` | `bd-2mo.13` |
| `hypothesis_family` | `ALIEN-PROOF-CARRYING-EXPERT-KERNEL` |
| `proposal_status` | `BLOCKED_ON_STRICT_PROFILE` |

### Contract

- Evidence: the attribution profile assigns 41.8 percent of dense decode to
  experts, and scalar/autovec int8 GEMV is its largest active leaf. Strict
  current-tree profiling must confirm this rank before source work begins.
- EV/relevance: `(impact 5 * confidence 4 * reuse 5) / (effort 4 * friction 2)
  = 12.5`; project relevance `4.55/5` because it targets the largest measured
  exact-compute leaf and reuses the existing dispatch/proof surface.
- State space: fixed `(shape, ISA, MR, NR, K-unroll, packing, prefetch-distance,
  row-chunk)` tuples for one hottest m=1 expert projection.
- Actions: offline mutation of exactly one schedule dimension; runtime selects
  only a hash-bound accepted schedule and performs no JIT or online search.
- Loss: measured expert cycles plus code-size/audit cost, with infinite loss for
  any i32 oracle, dequant-bit, output-byte, or receipt mismatch.
- Confidence/calibration: balanced A/B confidence interval and both-arm CV;
  validate exact saturated/random operands through `K=6848` before timing.
- Conservative fallback: the current autovec/platform dispatch. The planned
  experiment switch must restore it without rebuilding.
- One lever: replace only the hottest `m=1` expert contraction; routing,
  quantization, activation, combine order, and every other shape remain fixed.
- Keep gate: both-arm CV <= 5 percent, expert median >= 5 percent faster, dense
  decode >= 2 percent faster, broad wall/RSS regression <= 1 percent, and exact
  sparse/dense/20-page output identity.
- Revert gate: any proof mismatch, schedule-receipt drift, or failure of either
  focused or broad threshold. Record the rejected schedule in
  `docs/NEGATIVE_EVIDENCE.md` and its hash-verified evidence directory.

## Proposed Experiment `PROPOSAL-ALIEN-02` - Single-call exact SAM global attention

| field | value |
|---|---|
| `proposal_id` | `PROPOSAL-ALIEN-02` |
| `proposal_pillar` | `perf` |
| `proposed_at_utc` | `2026-07-11T19:25:00Z` |
| `proposed_by_agent` | `NavyTiger` |
| `target_bead_id` | `bd-2mo.30` |
| `hypothesis_family` | `ALIEN-EXACT-SAM-GLOBAL-ATTENTION` |
| `proposal_status` | `BLOCKED_ON_STRICT_PROFILE` |

### Contract

- Evidence: the attribution profile assigns about 1.3 seconds to four SAM
  global-attention blocks. The rejected row-tiling attempt lost because it
  multiplied external GEMM dispatches, not because exact fusion was disproven.
- EV/relevance: `(impact 5 * confidence 3 * reuse 3) / (effort 4 * friction 2)
  = 5.625`; project relevance `4.15/5` because the target is measured and the
  prior negative result supplies a precise countermeasure.
- State space: one fixed global-attention block/head shape with internally
  blocked QK, decomposed relative-position bias, exact row softmax, and PV.
- Actions: baseline two-GEMM path or one fused kernel invocation; no sequence of
  small public GEMM calls is an admissible action.
- Loss: global-block and vision time plus scratch traffic, with infinite loss
  for L1 bit drift, token drift, or changed QK/softmax/PV reduction order.
- Confidence/calibration: byte comparison first, then balanced stage-level A/B
  with CV and score-buffer traffic measured under the same quiet-host gate.
- Conservative fallback: the current large-GEMM attention path; the planned
  experiment switch must select it in the same binary.
- One lever: remove score-buffer materialization for one block/head shape while
  preserving every arithmetic order and all surrounding scheduling.
- Keep gate: score-buffer traffic reduced >= 90 percent, global-block median
  >= 10 percent faster, vision >= 3 percent faster, CV <= 5 percent, and broad
  end-to-end regression <= 1 percent.
- Revert gate: any correctness drift or failure to beat the tuned large-GEMM
  path. Retry only after a single-call design changes the measured bottleneck.

## Proposed Experiment `PROPOSAL-ALIEN-03` - Guarded next-token expert prefetch

| field | value |
|---|---|
| `proposal_id` | `PROPOSAL-ALIEN-03` |
| `proposal_pillar` | `perf` |
| `proposed_at_utc` | `2026-07-11T19:25:00Z` |
| `proposed_by_agent` | `NavyTiger` |
| `target_bead_id` | `bd-2mo.13` |
| `hypothesis_family` | `ALIEN-LEARNING-AUGMENTED-EXPERT-PREFETCH` |
| `proposal_status` | `BLOCKED_ON_PMU_EVIDENCE` |

### Adaptive Controller Contract

- Evidence prerequisite: PMU evidence must first attribute a material expert
  share to weight-latency stalls. Expert wall share alone is insufficient.
- EV/relevance: `(impact 4 * confidence 2 * reuse 4) / (effort 3 * friction 2)
  = 5.333`; project relevance `3.20/5`, deliberately discounted for the absent
  PMU attribution and calibration corpus.
- State space: `(layer, prior-token same-layer top6, prior-layer top6,
  token-class, panel-residency, model-hash, CPU-hash)`.
- Actions: `no_prefetch` or a bounded ordered expert-panel list with a fixed
  prefetch distance. Authoritative routing and arithmetic never change.
- Loss matrix: useful stall cycles saved minus weighted wasted bytes and LLC
  eviction; any output change has infinite loss. Tail loss includes p95 and
  CVaR stall regression, not mean-only speed.
- Posterior/confidence: a Beta posterior for useful-prefetch probability plus a
  lower credible bound; report Brier score and ECE over a held-out corpus.
- Calibration gate: shadow mode must show lower-bound useful probability
  >= 0.70, wasted bytes <= 25 percent, and no p95 pollution before actions arm.
- Deterministic fallback: unknown state, model/CPU hash drift, low confidence,
  bandwidth pollution, regret breach, or budget exhaustion selects
  `no_prefetch` immediately.
- Evidence ledger: emit per-token predicted panels, authoritative top6, useful
  bytes, wasted bytes, confidence, chosen action, fallback reason, and regret
  statistic to a hash-bound NDJSON artifact.
- One lever: hints for the next token in one decoder layer only. Routing,
  execution order, and all tensor bytes remain fixed.
- Keep gate: expert median >= 5 percent and decode >= 2 percent faster, CV <= 5
  percent, p95 no worse than 2 percent, exact output bytes, and unchanged RSS.
- Revert gate: any broad regression, calibration breach, drift-trigger failure,
  or evidence-ledger gap. Retry only after new PMU evidence identifies the same
  latency mechanism under a changed model, cache hierarchy, or batch regime.
