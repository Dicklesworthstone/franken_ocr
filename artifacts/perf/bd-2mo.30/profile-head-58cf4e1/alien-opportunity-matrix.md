# Alien Opportunity Matrix

These candidates combine the canonical `alien-graveyard` corpus,
`alien-artifact-coding`, the current profile, and the project's negative
evidence. The skill's `/data/projects/...` aliases are not mounted on this Mac,
so the same canonical files were read from their live local checkout:

- `/Users/jemanuel/projects/alien_cs_graveyard/alien_cs_graveyard.md`
  (`sha256:c33d78f712325cda80578b5e4fbdb3c3affb5ab6ae64a84cc1aa46ac1a6797a0`)
- `/Users/jemanuel/projects/alien_cs_graveyard/high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md`
  (`sha256:4065b4eb12ded01f413f3ee0a6623a2c3c2b586ed73d84d057d61a98c20e70c4`)

The compressed `references/GRAVEYARD-CATALOG.md` was used only as an execution
aid. Novel OCR synthesis is labeled explicitly.

Score uses the graveyard contract:

```text
EV = impact * confidence * reuse / (effort * friction)
```

All factors are 1-5. Only EV >= 2 proceeds to a spike. A spike is still
negative evidence until its correctness and performance gates pass.

| Rank | Candidate | Origin | Target | EV | Tier | Existing lane |
|---:|---|---|---|---:|---|---|
| 1 | proof-carrying offline microkernel synthesis | catalog | expert/prefill fixed-shape GEMM | 12.5 | A | refine SIMD beads |
| 2 | single-call exact SAM global-attention kernel | catalog + retry-predicate synthesis | 1.3 s global attention | 6.0 | A | new precise retry |
| 3 | learning-augmented expert prefetch | catalog | 2.96 s expert stage | 6.0 | A | refine `bd-2mo.13` |
| 4 | certified exact lm-head search | catalog seed + novel synthesis | 1.06 s lm-head | 4.5 | A | new/refine lm-head lane |
| 5 | progressive bitplane lm-head with exact refinement | novel synthesis | lm-head bandwidth | 4.5 | A | refine int4 lanes |
| 6 | exact quantized-zero expert-block skipping | catalog seed + novel synthesis | expert down projection | 4.0 | B | new |
| 7 | Hessian/error-feedback expert quantization | novel synthesis | int4 quality frontier | 4.0 | B | refine AF-1/int4 |
| 8 | content-addressed vision memoization | catalog seed + novel synthesis | repeated/template pages | 3.0 | B | new workload-specific |

## A1: Proof-Carrying Offline Microkernel Synthesis

Search fixed-shape schedules offline rather than trusting either handwritten
intrinsics or LLVM globally. State is `(shape, ISA, MR, NR, K-unroll, pack,
prefetch, row chunk, epilogue fusion)`. Actions mutate one schedule parameter.
Loss combines measured cycles, code size, audit surface, and any proof failure;
a proof failure has infinite loss. CEGIS/adversarial operands reject candidates
that differ from canonical i32 accumulation.

- Proof: bit-identical scalar comparison, saturated K=6848, every forced tier,
  golden token identity.
- Measurement: quiet micro distributions plus full expert/prefill/decode stage.
- Fallback: current static dispatch; no runtime JIT.
- Retry guard: bake only per-shape/per-microarchitecture winners; LLVM-major and
  silicon changes invalidate the schedule receipt.

## A2: Single-Call Exact SAM Global Attention

Fuse QK, decomposed relative-position bias, exact row-softmax, and PV inside one
head/block kernel without materializing the 4096x4096 score matrix or dispatching
many small GEMMs. This satisfies the existing row-tiling retry predicate only if
it remains one internally blocked kernel.

- Proof: preserve score reduction and softmax order; L1 activation comparator;
  L2 vision cosine/max-abs; L4/L5 tokens including page 0009.
- Measurement: dispatch count, score-buffer bytes, cache/DRAM traffic, global
  block time, vision time.
- Fallback: current large-GEMM attention.
- Falsifier: memory traffic is immaterial or the fused kernel loses tuned GEMM
  throughput.

## A3: Learning-Augmented Expert Prefetch

Routing remains authoritative; prediction can only change prefetch order.

- State: `(layer, prior-layer expert set, prior-token same-layer set, token
  class, residency tier)`.
- Actions: `no_prefetch` or an ordered expert/distance list.
- Loss: stall cycles + weighted wasted bytes + LLC eviction penalty.
- Posterior/confidence: calibrated transition-table coverage with a lower bound
  on useful-prefetch probability and drift monitoring per architecture.
- Calibration metric: useful-prefetch ratio and excess bytes on a held-out page
  corpus.
- Fallback trigger: unknown state, low lower-bound confidence, metadata drift,
  or bandwidth pollution returns to current deterministic/no-prefetch behavior.
- Evidence: decision ledger per token/layer plus expert-stage counters.

## A4: Certified Exact Lm-Head Search

Build an offline hierarchy over 129,280 weight rows with outward-rounded
blockwise inner-product upper bounds. Runtime ordering may use temporal logits
or grammar history, but pruning is legal only when a certified upper bound
cannot beat the exact incumbent. Candidate leaves use the canonical int8 dot.

- Proof: exact token and lowest-id tie behavior under n-gram bans; randomized
  hidden vectors, page 0590, 20-page corpus, and batched decode.
- Measurement: exact rows and bytes evaluated, certification rate, index bytes,
  load cost, lm-head and total decode time.
- Fallback: any malformed index, loose certificate, non-greedy request, or
  missing full logits runs the complete scan.
- Falsifier: high-dimensional bounds remain too loose or irregular traversal
  costs more than the saved row scans.

## A5: Progressive Bitplane Lm-Head

Scan a signed high-nibble base and retain residual-norm bounds. Refine only rows
whose residual interval can still beat the incumbent; fall back to the canonical
int8 row when the margin cannot certify the exact winner.

- Hard prerequisite: a true in-register packed-int4 dot. Materializing int8 is
  forbidden by the ledgered 5.8x regression.
- Proof: exact int8-baseline token and tie behavior at every step.
- Measurement: base/refinement bytes, refinement rate, certification rate.
- Fallback: full canonical int8 scan.

## B1: Exact Quantized-Zero Expert Skipping

Measure contiguous all-zero blocks after the existing activation quantization.
Only products whose quantized activation bytes are exactly zero may be omitted.
A column-blocked down-projection layout plus a mask can then avoid weight reads
without changing any i32 sum.

- Proof: bit-identical accumulators across tiers and page 0590 output.
- Measurement: zero density by layer/expert, avoided bytes, mask overhead.
- Fallback: fixed offline density threshold selects the dense kernel;
  `FOCR_ZERO_BLOCK_SKIP=0` disables the path.
- Falsifier: insufficient block density or layout overhead exceeds saved work.

## B2: Hessian/Error-Feedback Expert Quantization

Use held-out activation covariance to deterministically quantize expert blocks
with sequential error compensation. Runtime stays the same; only conversion
changes.

- Proof: layer cosine/logits/tokens, dense 20-page CER, CVaR/EVT tail, combined
  lossy-stack gate.
- Measurement: equal-footprint quality and throughput against symmetric int4.
- Fallback: current symmetric int8/per-group-int4 recipe.
- Falsifier: calibration overfit, unstable solve, or tail regression.

## B3: Content-Addressed Vision Memoization

Cache whole-page vision output only for byte-identical normalized tensors,
model hash, and preprocessing config. A hash hit is verified by exact bytes.
S3-FIFO bounds the cache. Reuse of partial SAM windows is allowed only before
the first global-attention block and only with position identity.

- Proof: byte-identical vision activation and OCR output.
- Measurement: duplicate/template hit rate, hash cost, bytes, vision wall.
- Fallback: any mismatch or miss executes the normal tower.
- Falsifier: realistic hit rate is too low or cache pressure harms the model.

## Explicit Exclusions

The queue does not duplicate already-operationalized resident experts,
expert-as-service, conformal layer skipping/drafting, EAGLE/tree verification,
USL/NUMA, native packed int4, AF-1/AF-2, megakernels, batch spines, or PGO/BOLT.
It also does not retry approximate decode attention, int8 KV, polynomial
softmax, materialized int4, serial QKV, or naive SAM row tiling without first
satisfying their ledgered retry predicates.
