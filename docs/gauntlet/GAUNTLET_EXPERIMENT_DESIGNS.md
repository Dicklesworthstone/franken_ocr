# Gauntlet Experiment Designs

## Experiment `EXP-1401` - Stream source BF16 through decode GEMVs

| field | value |
|---|---|
| `experiment_id` | `EXP-1401` |
| `pillar` | `perf` |
| `created_at_utc` | `2026-07-10T02:08:00Z` |
| `created_by_agent` | `BrownFox` |
| `bead_id` | `bd-2mo.30` |
| `parent_hypothesis_id` | `PASS-01-BF16-STREAM` |
| `status` | `CLOSED` |

### Hypothesis

Reading BF16 decoder weights directly inside GEMV would reduce widened-cache
traffic without changing accumulation order or decoded output.

### Motivation

Decoder expert and head weight traffic dominated the first measured profile, so
removing an expanded f32 cache appeared to offer an accretive bandwidth win.

### Minimal Reproducer

Run the interleaved six-sample page workload with the streaming mode disabled and
enabled, holding model, page, worker count, precision gates, and decode cap fixed.

### Expected Signal

Byte-identical Markdown and at least a two-percent median decode improvement with
no material regression in the measured `lm_head` phase.

### Falsifiability Criteria

Any output mismatch, or a stable decode or `lm_head` regression beyond noise,
falsifies the proposed source-BF16 consumption path.

### One-Line Invocation

`FOCR_BF16_STREAM=1 FOCR_THREADS=8 FOCR_TIMING=1 FOCR_PROFILE_DECODE=1 focr ocr page_0014.png --model unlimited-ocr.recipe-v1.focrq`

### Results Inline

```yaml
result_status: NO_EVIDENCE
result_summary: outputs remained byte-identical but median decode regressed 4.59 percent and lm_head regressed 7.44 percent, so the lever was reverted
result_evidence_paths:
  - artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-bf16-stream
closed_at_utc: 2026-07-10T03:20:00Z
retry_condition_predicate: retry only if a profiler attributes a clearly-above-noise share to BF16 widening after a native packed-BF16 kernel exists
```

### Closure Predicate

Close after output identity, interleaved measurements, a hash manifest, and the
negative-evidence retry predicate are all recorded.

## Retrospective Campaign Cards

These cards reconstruct the precommitted lever, fallback, falsifier, result,
and evidence for campaign Passes 2-11. They do not upgrade local A/B results to
strict current-tree or reference-backed wins.

## Experiment `EXP-1405` - Parallel independent R-SWA heads

| field | value |
|---|---|
| `experiment_id` | `EXP-1405` |
| `pillar` | `perf` |
| `created_at_utc` | `2026-07-11T19:25:00Z` |
| `created_by_agent` | `NavyTiger` |
| `bead_id` | `bd-2mo.30` |
| `parent_hypothesis_id` | `PASS-02-RSWA-PARALLEL-HEADS` |
| `status` | `CLOSED` |

- Hypothesis: schedule ten independent scalar R-SWA heads across Rayon without
  changing any within-head arithmetic order.
- One lever: `FOCR_RSWA_PARALLEL_ATTN=1` versus the serial `=0` path.
- Keep/falsify: require byte identity, a focused attention gain above noise, and
  no end-to-end regression; any within-head reorder or output drift rejects it.
- Fallback: `FOCR_RSWA_PARALLEL_ATTN=0` restores serial execution.
- Result: `PROVISIONAL_LOCAL_WIN`; attention median improved 13.59 percent and
  dense decode improved 6.09 percent in the exploratory A/B.
- Evidence: `artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-rswa-par-heads/`.

### Hypothesis

Parallel independent heads reduce scalar R-SWA attention time without changing
the exact within-head online-softmax or accumulation order.

### Motivation

The attribution profile placed R-SWA among the dominant dense-decode phases,
and ten disjoint heads expose scheduling parallelism without numeric fusion.

### Minimal Reproducer

Run the same dense page and long sentinel with the serial and parallel switches
in balanced order, fixed at eight workers and the conservative model recipe.

### Expected Signal

Byte-identical outputs and at least a two-percent median attention improvement
with no greater than one-percent broad regression.

### Falsifiability Criteria

Any output mismatch, within-head reorder, CV above five percent, focused gain
below two percent, or broad regression above one percent rejects promotion.

### One-Line Invocation

`FOCR_RSWA_PARALLEL_ATTN=1 FOCR_THREADS=8 FOCR_PROFILE_DECODE=1 focr ocr page_0014.png --model unlimited-ocr.recipe-v1.focrq`

### Results Inline

```yaml
result_status: NO_EVIDENCE
result_summary: exploratory local attention and decode gains passed byte identity, but strict current-tree sparse dense and 20-page evidence is still absent
result_evidence_paths:
  - artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-rswa-par-heads
closed_at_utc: 2026-07-11T19:35:00Z
retry_condition_predicate: blocked until bd-2mo.30.15 captures the strict current-tree balanced A/B under the quiet-host gate
```

### Closure Predicate

The retrospective card closes when the local evidence is hash-verified and its
strict-current qualification gap is explicit; it does not promote the lever.

## Experiment `EXP-1406` - Interleave exact lm-head rows

| field | value |
|---|---|
| `experiment_id` | `EXP-1406` |
| `pillar` | `perf` |
| `created_at_utc` | `2026-07-11T19:25:00Z` |
| `created_by_agent` | `NavyTiger` |
| `bead_id` | `bd-2mo.30` |
| `parent_hypothesis_id` | `PASS-03-LMHEAD-ROW-INTERLEAVE` |
| `status` | `CLOSED` |

- Hypothesis: reuse each hidden activation across two or four vocabulary rows
  while preserving each row's eight-lane f32 reduction order.
- One lever: change only the number of adjacent rows processed by the scalar
  high-precision lm-head loop.
- Keep/falsify: require bit identity and at least a two-percent lm-head gain;
  stable phase regression rejects the lever.
- Fallback: one row at a time.
- Result: `NO_EVIDENCE`; two/four rows regressed lm-head by 29.02/39.21 percent,
  so the experiment was source-reverted.
- Evidence: `artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-lmhead-row-interleave/`.

### Hypothesis

Walking one activation across two or four exact lm-head rows reduces repeated
activation loads enough to offset the extra accumulator pressure.

### Motivation

The high-precision lm-head was a measured dense-decode phase, and adjacent rows
can share activation reads without changing any row's reduction order.

### Minimal Reproducer

Compare row widths one, two, and four in reverse order on page 0014 with fixed
model, workers, precision, and stage instrumentation.

### Expected Signal

Bit-identical logits/output and at least a two-percent median lm-head gain.

### Falsifiability Criteria

Any bit drift or a stable lm-head/decode regression beyond noise rejects the
scalar interleave.

### One-Line Invocation

`FOCR_LMHEAD_ROW_INTERLEAVE=2 FOCR_THREADS=8 FOCR_PROFILE_DECODE=1 focr ocr page_0014.png --model unlimited-ocr.recipe-v1.focrq`

### Results Inline

```yaml
result_status: NO_EVIDENCE
result_summary: exact scalar row interleave regressed lm-head by 29.02 percent at width two and 39.21 percent at width four
result_evidence_paths:
  - artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-lmhead-row-interleave
closed_at_utc: 2026-07-11T19:35:00Z
retry_condition_predicate: worth reconsidering when a native multi-row microkernel proves activation reuse without the measured scalar accumulator-pressure regression
```

### Closure Predicate

Close after exact output proof, reverse-order timing, source revert, hash
manifest, and the native-microkernel retry predicate are recorded.

## Experiment `EXP-1407` - Sequential lm-head vocabulary tiles

| field | value |
|---|---|
| `experiment_id` | `EXP-1407` |
| `pillar` | `perf` |
| `created_at_utc` | `2026-07-11T19:25:00Z` |
| `created_by_agent` | `NavyTiger` |
| `bead_id` | `bd-2mo.30` |
| `parent_hypothesis_id` | `PASS-04-LMHEAD-VOCAB-TILES` |
| `status` | `CLOSED` |

- Hypothesis: contiguous vocabulary panels improve locality without changing a
  logit's reduction order.
- One lever: existing `FOCR_LMHEAD_SHARD_TILES=2|8|32` schedule only.
- Keep/falsify: require byte identity and a stable lm-head improvement over the
  bracketed monolithic baseline.
- Fallback: `FOCR_LMHEAD_SHARD` unset selects the monolithic head.
- Result: `NO_EVIDENCE`; cost worsened monotonically as tile count increased.
- Evidence: `artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-lmhead-vocab-tiles/`.

### Hypothesis

Sequential contiguous vocabulary panels improve cache residency enough to
offset repeated scheduling over the exact high-precision lm-head.

### Motivation

Lm-head weight traffic is large and the existing switch allowed a source-free
schedule test while retaining each logit's exact reduction.

### Minimal Reproducer

Bracket tiles 2, 8, and 32 with monolithic runs on the same dense page and fixed
eight-worker conservative configuration.

### Expected Signal

Byte-identical output and a stable lm-head improvement over both monolithic
brackets.

### Falsifiability Criteria

A neutral result, monotonic regression with tile count, or output drift rejects
sequential tiling on this topology.

### One-Line Invocation

`FOCR_LMHEAD_SHARD=1 FOCR_LMHEAD_SHARD_TILES=8 FOCR_THREADS=8 FOCR_PROFILE_DECODE=1 focr ocr page_0014.png --model unlimited-ocr.recipe-v1.focrq`

### Results Inline

```yaml
result_status: NO_EVIDENCE
result_summary: sequential vocabulary tiling was byte-identical but regressed lm-head monotonically as tile count increased
result_evidence_paths:
  - artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-lmhead-vocab-tiles
closed_at_utc: 2026-07-11T19:35:00Z
retry_condition_predicate: worth reconsidering when concurrent NUMA-local tiles or fused exact argmax remove the repeated sequential scheduling cost
```

### Closure Predicate

Close after bracketed measurements, byte identity, default-off disposition,
hash verification, and the topology-specific retry condition are present.

## Experiment `EXP-1408` - Whole-forward worker count

| field | value |
|---|---|
| `experiment_id` | `EXP-1408` |
| `pillar` | `perf` |
| `created_at_utc` | `2026-07-11T19:25:00Z` |
| `created_by_agent` | `NavyTiger` |
| `bead_id` | `bd-2mo.30` |
| `parent_hypothesis_id` | `PASS-05-WORKER-COUNT` |
| `status` | `CLOSED` |

- Hypothesis: six or ten workers may fit the mixed attention/expert/lm-head
  workload better than the inherited eight-worker setting.
- One lever: move `FOCR_THREADS`, `RAYON_NUM_THREADS`, and `OMP_NUM_THREADS`
  together across 6/8/10.
- Keep/falsify: require identical output and a broad decode gain without moving
  another major phase backward.
- Fallback: deterministic eight-worker setting.
- Result: `NO_EVIDENCE`; six and ten workers regressed median decode by 8.08 and
  2.46 percent respectively.
- Evidence: `artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-thread-count/`.

### Hypothesis

Six or ten whole-forward workers fit the mixed M4 attention, expert, and
lm-head workload better than the inherited eight-worker setting.

### Motivation

Different phases showed heterogeneous scaling, so core count alone could not
justify the benchmark worker setting.

### Minimal Reproducer

Sweep 8, 6, and 10 workers in reverse order while moving FOCR, Rayon, and OMP
counts together on the same dense page.

### Expected Signal

Byte-identical output and a broad decode improvement without a compensating
major-phase regression.

### Falsifiability Criteria

If an alternate count loses broad decode or merely shifts time between major
phases, retain eight workers.

### One-Line Invocation

`FOCR_THREADS=10 RAYON_NUM_THREADS=10 OMP_NUM_THREADS=10 FOCR_PROFILE_DECODE=1 focr ocr page_0014.png --model unlimited-ocr.recipe-v1.focrq`

### Results Inline

```yaml
result_status: NO_EVIDENCE
result_summary: six and ten workers preserved output but regressed median decode by 8.08 and 2.46 percent versus eight
result_evidence_paths:
  - artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-thread-count
closed_at_utc: 2026-07-11T19:35:00Z
retry_condition_predicate: retry only if this workload class exhibits measurable phase-mix or CPU-topology changes relative to the recorded Apple M4 run
```

### Closure Predicate

Close after the reverse-order sweep, phase attribution, exact-output proof,
eight-worker fallback, and topology-change retry rule are recorded.

## Experiment `EXP-1409` - Mmap-capable artifact loading

| field | value |
|---|---|
| `experiment_id` | `EXP-1409` |
| `pillar` | `perf` |
| `created_at_utc` | `2026-07-11T19:25:00Z` |
| `created_by_agent` | `NavyTiger` |
| `bead_id` | `bd-2mo.30` |
| `parent_hypothesis_id` | `PASS-06-MMAP-LOAD` |
| `status` | `CLOSED` |

- Hypothesis: mmap reduces warm-cache artifact ingestion cost without changing
  validated record bytes.
- One lever: mmap-backed versus owned buffered loading; model math stays fixed.
- Keep/falsify: require byte identity and lower startup wall/system CPU, then
  separately reject default-on use if inode immutability is not enforceable.
- Fallback: environment unset uses owned bytes; `FOCR_MMAP=1` is explicit opt-in.
- Result: `PROVISIONAL_LOCAL_WIN` for the capability, but rejected as the
  shipping default because same-inode truncation can fault the process.
- Evidence: `artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-mmap-load/`.

### Hypothesis

Mmap-backed `.focrq` loading reduces warm-cache startup wall and system CPU
without changing any validated record bytes or inference math.

### Motivation

The multi-gigabyte model artifact makes ingestion ownership observable even in
a short capped-decode workload.

### Minimal Reproducer

Run mmap and buffered-owned loading in reverse order with a 32-token cap and
fixed model, page, workers, allocator, and compute switches.

### Expected Signal

Byte-identical output, lower startup wall/system CPU, and stable vision,
prefill, and decode stages.

### Falsifiability Criteria

Any byte drift, compute-stage movement, absent startup gain, or inability to
enforce inode immutability rejects mmap as a general default.

### One-Line Invocation

`FOCR_MMAP=1 FOCR_MAX_NEW_TOKENS=32 FOCR_THREADS=8 focr ocr page_0014.png --model unlimited-ocr.recipe-v1.focrq`

### Results Inline

```yaml
result_status: NO_EVIDENCE
result_summary: local startup gains were real, but safety review rejected mmap as default and strict current-tree reference qualification is absent
result_evidence_paths:
  - artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-mmap-load
closed_at_utc: 2026-07-11T19:35:00Z
retry_condition_predicate: blocked until an enforceable immutable-inode mechanism exists and bd-2mo.30.15 repeats the startup A/B
```

### Closure Predicate

Close when local timing/identity evidence and the safety rejection are both
recorded; the capability remains opt-in and is not a certified default win.

## Experiment `EXP-1410` - Chunk the reference prefill

| field | value |
|---|---|
| `experiment_id` | `EXP-1410` |
| `pillar` | `perf` |
| `created_at_utc` | `2026-07-11T19:25:00Z` |
| `created_by_agent` | `NavyTiger` |
| `bead_id` | `bd-2mo.30` |
| `parent_hypothesis_id` | `PASS-07-PREFILL-CHUNK` |
| `status` | `CLOSED` |

- Hypothesis: smaller reference chunks improve cache locality enough to repay
  repeated scheduling and layer overhead.
- One lever: `FOCR_PREFILL_CHUNK=256|128|64` versus monolithic prefill.
- Keep/falsify: require byte identity and a prefill gain in the capped workload.
- Fallback: environment unset selects monolithic prefill.
- Result: `NO_EVIDENCE`; prefill regressed 32.8, 56.8, and 98.4 percent.
- Evidence: `artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-prefill-chunk/`.

### Hypothesis

Splitting a 277-token reference prefill into smaller chunks improves locality
enough to repay repeated layer and scheduling overhead.

### Motivation

The existing bit-exact chunk switch allowed a focused scheduling test without
changing attention arithmetic or decode length.

### Minimal Reproducer

Bracket chunk sizes 256, 128, and 64 with monolithic capped-decode runs under
the same conservative model and eight-worker configuration.

### Expected Signal

Byte-identical output and a lower prefill median than the monolithic brackets.

### Falsifiability Criteria

Stable prefill regression, especially monotonic regression as chunks shrink,
rejects short-reference chunking.

### One-Line Invocation

`FOCR_PREFILL_CHUNK=128 FOCR_MAX_NEW_TOKENS=32 FOCR_THREADS=8 focr ocr page_0014.png --model unlimited-ocr.recipe-v1.focrq`

### Results Inline

```yaml
result_status: NO_EVIDENCE
result_summary: all chunk sizes preserved bytes but regressed prefill from 32.8 through 98.4 percent as chunks shrank
result_evidence_paths:
  - artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-prefill-chunk
closed_at_utc: 2026-07-11T19:35:00Z
retry_condition_predicate: worth reconsidering when substantially longer references or real overlap can amortize repeated chunk scheduling
```

### Closure Predicate

Close after bracketed timing, byte identity, monolithic fallback, hash
verification, and a length/overlap-specific retry predicate are present.

## Experiment `EXP-1411` - Transfer softmax result ownership

| field | value |
|---|---|
| `experiment_id` | `EXP-1411` |
| `pillar` | `perf` |
| `created_at_utc` | `2026-07-11T19:25:00Z` |
| `created_by_agent` | `NavyTiger` |
| `bead_id` | `bd-2mo.30.9` |
| `parent_hypothesis_id` | `PASS-08-SOFTMAX-OWNERSHIP` |
| `status` | `CLOSED` |

- Hypothesis: assign the already-owned kernel result instead of copying the
  complete SAM softmax payload.
- One lever: ownership transfer only; arithmetic and layout remain unchanged.
- Keep/falsify: require identical elements and a focused SAM/vision gain; any
  output drift or neutral strict-current result rejects it.
- Fallback: truthy `FOCR_SOFTMAX_COPY=1` restores the copy in the same binary.
- Result: `PROVISIONAL_LOCAL_WIN`; exploratory SAM blocks improved 3.27 percent.
- Evidence: `artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-softmax-ownership/`.

### Hypothesis

Assigning the kernel-owned softmax vector avoids a redundant SAM payload copy
without changing any element or output layout.

### Motivation

SAM softmax touches enough data that ownership transfer can remove gigabytes of
read/write traffic while keeping numerical semantics identical.

### Minimal Reproducer

Compare same-workload copy and ownership paths on sparse/dense pages, then on
capped 10/20-page batches under the strict current-tree protocol.

### Expected Signal

Byte-identical output and at least a two-percent median SAM/vision improvement
with no broad regression.

### Falsifiability Criteria

Any element drift, CV above five percent, focused gain below two percent, or
broad regression above one percent rejects promotion.

### One-Line Invocation

`FOCR_SOFTMAX_COPY=0 FOCR_THREADS=8 FOCR_PROFILE_DECODE=1 focr ocr page_0009.png --model unlimited-ocr.recipe-v1.focrq`

### Results Inline

```yaml
result_status: NO_EVIDENCE
result_summary: the local before-after binary result favored ownership transfer, but strict same-binary current-tree and multipage qualification is pending
result_evidence_paths:
  - artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-softmax-ownership
closed_at_utc: 2026-07-11T19:35:00Z
retry_condition_predicate: blocked until bd-2mo.30.15 completes sparse dense 10-page and 20-page balanced same-binary measurements
```

### Closure Predicate

The retrospective card closes with honest provisional status; the implementation
cannot move to the reference-backed performance ledger until the strict gate.

## Experiment `EXP-1412` - Cache Unlimited vision statics

| field | value |
|---|---|
| `experiment_id` | `EXP-1412` |
| `pillar` | `perf` |
| `created_at_utc` | `2026-07-11T19:25:00Z` |
| `created_by_agent` | `NavyTiger` |
| `bead_id` | `bd-2mo.30.11` |
| `parent_hypothesis_id` | `PASS-09-VISION-STATICS` |
| `status` | `CLOSED` |

- Hypothesis: hydrate immutable SAM weights and transpose the bridge projector
  once per model rather than once per page.
- One lever: model-owned Unlimited vision static cache only.
- Keep/falsify: require timing-stripped batch identity, later-page vision gain,
  no deadlock, and no retained-RSS increase.
- Fallback: `FOCR_UNLIMITED_VISION_CACHE=0` rebuilds the statics per page.
- Result: `PROVISIONAL_LOCAL_WIN`; five-page wall improved 3.76 percent and max
  RSS fell by 106.2 MB in the exploratory A/B.
- Evidence: `artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-unlimited-vision-cache/`.

### Hypothesis

Model-owned immutable SAM/projector statics remove repeated page hydration and
transpose costs without increasing retained memory or changing output.

### Motivation

Load-once batch execution previously rebuilt immutable vision state per page,
while the CLIP lane already demonstrated the intended cache ownership pattern.

### Minimal Reproducer

Run cache off/on in the same binary over repeated-page and sorted 10/20-page
batches, stripping timing fields before semantic comparison.

### Expected Signal

Identical per-page results, later-page vision and batch-wall gains above two
percent, no deadlock, and no max-RSS increase.

### Falsifiability Criteria

Any result drift, retained-memory growth, watchdog failure, CV above five
percent, or broad regression above one percent rejects promotion.

### One-Line Invocation

`FOCR_UNLIMITED_VISION_CACHE=1 FOCR_MAX_NEW_TOKENS=32 FOCR_THREADS=8 focr ocr-batch page_*.png --model unlimited-ocr.recipe-v1.focrq --json`

### Results Inline

```yaml
result_status: NO_EVIDENCE
result_summary: the local five-page result favored model-owned statics with lower RSS, but strict current-tree 10-page and 20-page distributions are pending
result_evidence_paths:
  - artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-unlimited-vision-cache
closed_at_utc: 2026-07-11T19:35:00Z
retry_condition_predicate: blocked until bd-2mo.30.15 captures balanced current-tree 10-page and 20-page cache-off and cache-on distributions
```

### Closure Predicate

Close the retrospective record after local identity/RSS evidence and the exact
strict-current blocker are both machine-readable.

## Experiment `EXP-1413` - Apple dense-int8 effective route

| field | value |
|---|---|
| `experiment_id` | `EXP-1413` |
| `pillar` | `perf` |
| `created_at_utc` | `2026-07-11T19:25:00Z` |
| `created_by_agent` | `NavyTiger` |
| `bead_id` | `bd-2mo.30` |
| `parent_hypothesis_id` | `PASS-10-APPLE-AUTOVEC` |
| `status` | `CLOSED` |

- Hypothesis: LLVM's scalar-loop autovectorization still beats the hand-written
  SDOT micro-tile for the conservative decoder's m=1 expert contractions.
- One lever: default autovec versus `FOCR_INT8_AUTOVEC=0` forced SDOT.
- Keep/falsify: require exact integer/output parity and a focused expert/decode
  gain; an operator-error sample is excluded but retained in evidence.
- Fallback: `FOCR_INT8_AUTOVEC=0` restores SDOT.
- Result: `PROVISIONAL_LOCAL_WIN`; expert median improved 33.21 percent.
- Evidence: `artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-apple-int8-autovec/`.

### Hypothesis

LLVM-autovectorized exact int8 contraction beats the hand-written SDOT route
for the conservative decoder's ordinary m=1 expert work.

### Motivation

The profile's hottest active leaf was scalar/autovec int8 GEMV, and prior M4
microbenchmarks warned that hand-written wide SIMD could lose at this shape.

### Minimal Reproducer

Run default autovec and forced SDOT in interleaved order on dense and capped
10/20-page conservative workloads, retaining any operator-error log separately.

### Expected Signal

Exact output parity and at least a two-percent expert/decode improvement without
moving broad wall or RSS backward.

### Falsifiability Criteria

Any output drift, CV above five percent, focused gain below two percent, or
broad regression above one percent rejects autovec promotion.

### One-Line Invocation

`FOCR_INT8_AUTOVEC=1 FOCR_MAX_NEW_TOKENS=32 FOCR_THREADS=8 FOCR_PROFILE_DECODE=1 focr ocr page_0014.png --model unlimited-ocr.recipe-v1.focrq`

### Results Inline

```yaml
result_status: NO_EVIDENCE
result_summary: local capped decode favored autovec over SDOT, but only two valid samples per arm and no strict multipage current-tree distribution exist
result_evidence_paths:
  - artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-apple-int8-autovec
closed_at_utc: 2026-07-11T19:35:00Z
retry_condition_predicate: blocked until bd-2mo.30.15 repeats dense 10-page and 20-page expert-stage measurements under the strict CV gate
```

### Closure Predicate

Close the retrospective card with local exact-route evidence and an explicit
strict-current blocker; keep forced SDOT as the deterministic proof fallback.

## Experiment `EXP-1414` - Forced Apple dense-int8 tiers

| field | value |
|---|---|
| `experiment_id` | `EXP-1414` |
| `pillar` | `perf` |
| `created_at_utc` | `2026-07-11T19:25:00Z` |
| `created_by_agent` | `NavyTiger` |
| `bead_id` | `bd-2mo.30` |
| `parent_hypothesis_id` | `PASS-11-APPLE-TIER-SWEEP` |
| `status` | `CLOSED` |

- Hypothesis: a forced SDOT or SMMLA intrinsic route may beat autovec after the
  conservative recipe narrows int8 work to FFN/expert matrices.
- One lever: fresh processes force `scalar`, `sdot`, or `smmla` only.
- Keep/falsify: require byte identity and a stable focused win over autovec.
- Fallback: environment unset restores autovec; forced routes remain available
  for portability and parity diagnostics.
- Result: `NO_EVIDENCE`; SDOT and SMMLA regressed expert time by 48.75 and
  327.42 percent, so neither is promoted on Apple M4.
- Evidence: `artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-arm-tier-force/`.

### Hypothesis

A forced SDOT or SMMLA tier can beat scalar/autovec after the conservative
recipe narrows int8 work to FFN and expert matrices.

### Motivation

Every hardware route must remain measured and parity-proven rather than being
promoted from nominal instruction capability.

### Minimal Reproducer

Run fresh processes forced to scalar, SDOT, and SMMLA twice per tier in reverse
order with fixed capped decode and all non-int8 stages unchanged.

### Expected Signal

Byte-identical output and a stable expert/decode win for an intrinsic tier over
the scalar/autovec baseline.

### Falsifiability Criteria

If either intrinsic tier regresses the expert phase or broad decode, retain it
only for portability and proof.

### One-Line Invocation

`FOCR_FORCE_ARCH=sdot FOCR_MAX_NEW_TOKENS=32 FOCR_THREADS=8 FOCR_PROFILE_DECODE=1 focr ocr page_0014.png --model unlimited-ocr.recipe-v1.focrq`

### Results Inline

```yaml
result_status: NO_EVIDENCE
result_summary: SDOT and SMMLA preserved output but regressed expert time by 48.75 and 327.42 percent versus scalar autovec
result_evidence_paths:
  - artifacts/perf/bd-2mo.30/profile-recipe-5733407/ab-arm-tier-force
closed_at_utc: 2026-07-11T19:35:00Z
retry_condition_predicate: retry only if a profiler attributes a clearly-above-noise share to changed intrinsic kernel loading packing or scheduling costs
```

### Closure Predicate

Close after forced-route proof, reverse-order timing, retained diagnostic
fallbacks, hash verification, and a kernel-change retry predicate are recorded.
