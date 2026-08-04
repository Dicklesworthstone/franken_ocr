# Conformance Hypothesis Ledger

## Experiment `EXP-1403` - Deterministic page-0590 runaway rejection

| field | value |
|---|---|
| `experiment_id` | `EXP-1403` |
| `pillar` | `conformance` |
| `created_at_utc` | `2026-07-10T08:20:44Z` |
| `created_by_agent` | `BrownFox` |
| `bead_id` | `bd-2mo.30.12` |
| `parent_hypothesis_id` | `PASS-13-PAGE0590` |
| `status` | `CLOSED` |

### Hypothesis

A token-level periodicity and novelty controller can reject sustained runaway
tails without synthesizing EOS, changing ordinary output, or promoting the
unvalidated full-int8 attention and head recipe.

### Motivation

The Pass 13 conservative artifact reached the token cap on page 0590, while
full-int8 was faster but measurably worsened normalized CER by 31.82 percent.
The subsequent exact Torch routing correction supplied a distinct root-cause
fix that had to be measured before promoting a new output controller.

### Minimal Reproducer

Replay the pinned page and BF16 reference through conservative decode with the
controller off and on, then run the complete clean-page corpus for false
positives and tail/CER impact.

### Expected Signal

The controller remains inert on accepted outputs, returns typed timeout evidence
for the known runaway, and clears the predeclared false-positive posterior and
loss-matrix gates before any default changes.

### Falsifiability Criteria

Any accepted-output byte drift, synthetic success, false positive outside the
calibrated bound, or failure to improve the pinned runaway CER/tail disposition
rejects the controller policy.

### One-Line Invocation

`FOCR_RUNAWAY_GUARD=1 focr ocr page_0590.png --model unlimited-ocr.recipe-v1.focrq`

### Results Inline

```yaml
result_status: NO_EVIDENCE
result_summary: the opt-in runaway controller was neither promoted nor needed; exact Torch 2.10 MoE routing removed the no-EOS failure, emitted EOS at 4033 tokens on page_0590, and passed the complete 20-page corpus at normalized CER 0.1930792459 within the 0.25 budget, while page_0590 CER 0.6164885983 remains an explicit high-tail caveat
result_evidence_paths:
  - artifacts/perf/bd-2mo.30/profile-recipe-5733407/pass15-exact-torch-corpus-20260710T182636Z
result_evidence_hashes:
  - summary.json=1b43e4d0bc35054e1b0068a65bcb9a7724ed548e3afb70c0b42da56f263e9bcc
  - binary=6a401be3bfd6bfb0a9766e2ff333489a5e02b2a5f0641376b6ab5fc6885e6953
  - model=573340710167697891bf52dfa4cbb5d0a02a68f3011c01f8ef83fd34622fb592
  - page_0590_input=6d71d9c94f2370f51824fb91e3291ce4c64052979adc8f3b14dfe618683512d3
  - page_0590_reference=6542b1d31b64103e9a56104738bf9038487877e8408b8ac34d8d10d1f5d2c8cd
  - page_0590_output=57d9b5a6686e4c8f877b42390c74b85f77ef789036351cfbd0f373194d10997a
  - page_0590_compare=49d0b7e9074348f6e38e44d843c7f3f2e7091de643f0ba331dd874ae61f930bb
  - ocr-batch.json=7d6e1f5d7823c4a297b5fcc07f7ade44a0ce36f0cd89a14f4d6a424345db572c
  - corpus.compare.json=38680e64677ab30f4842aa53fa98eb9c1ef0caefca3e293163f1e34b501d9f43
closed_at_utc: 2026-07-10T19:18:27Z
retry_condition_predicate: retry only if this workload class exhibits measurable no-EOS regression under exact Torch routing or a future numerical lever recreates a hash-bound runaway; the controller remains opt-in and unpromoted
```

### Closure Predicate

Close with `NO_EVIDENCE` when a distinct correctness fix removes the no-EOS
failure and clears the pinned page/corpus gate without arming the controller.
Reopen only under the recorded retry predicate; this closure does not convert
the opt-in controller into accepted runtime behavior.
