# RATCHET.md — the conformal lower-bound release ratchet (bd-re8.14)

The release decision is never taken on a point estimate. "99.2 % pass" on a
finite calibration corpus is overconfident: it ignores sampling luck, so a
noisy improvement can masquerade as progress. The ratchet takes every release
decision on a statistically defensible **lower bound**, per category, and the
committed floor only ever moves up.

Implementation: `src/conformance.rs` (`category_bound`, `ratchet_decide`,
`transparency_card`, `RATCHET_ALPHA`, `MIN_CALIBRATION_N`). Consumed by the
three-pillar certification (bd-re8.13) as its release-decision input.

## 1. The bound

Per category, with `s` passes and `f` failures over `n = s + f` calibration
items:

```
beta_lower = BetaQuantile(s + 1/2, f + 1/2; alpha)        # Jeffreys posterior
dkw_lower  = max(0, s/n − sqrt(ln(1/alpha) / (2n)))       # distribution-free (Hoeffding, one-sided)
lower      = truncate_6dp( min(beta_lower, dkw_lower) )   # the decision value
```

- `alpha = 0.05` (`RATCHET_ALPHA`): each bound is a one-sided 95 % lower
  confidence limit.
- **The stingier instrument wins.** The Beta posterior prices binomial
  sampling uncertainty under exchangeability; the Hoeffding band prices
  distribution-freeness. Taking `min` means the bound survives whichever
  assumption is weaker on the day.
- **Truncation, not rounding**, to 6 dp (`truncate_score`): the decision
  boundary is deterministic and reproducible from the counts alone.

## 2. The ratchet rule

A change may land **only if no per-category lower bound drops below its
committed floor**:

- A category that **holds or raises** its bound: admissible.
- A category that **lowers** its bound: **rejected**, even if the aggregate
  improves — the no-cross-regression principle, encoded at the statistics
  layer (`per_category_regression_blocks_even_when_aggregate_improves`).
- A baseline category **missing from the candidate**: dropped coverage,
  **rejected**.
- A **new** category: admissible; its bound becomes the initial floor.

The committed floor set is monotone: landing a change updates each floor to
`max(old, new)`. (The floor ledger is wired with the three-pillar cert,
bd-re8.13, where the per-category corpus counts live; until then the
machinery + tests stand ready and the transparency card documents every
computed bound.)

## 3. The checked small-corpus precondition (review-r1 addendum)

A too-small calibration corpus makes the conformal bound meaninglessly low —
the prior dominates and a flawless category reads as near-failing, blocking
every landing. This is a **checked precondition, not advice**:

- `MIN_CALIBRATION_N = 20`. Derivation (asserted by
  `min_calibration_n_is_the_computed_threshold`): the smallest `n` at which a
  perfect record's Jeffreys 95 % lower bound clears 0.9 is **n = 18**
  (bound 0.900124; n = 17 gives 0.894652). We take 20 (bound 0.909524) for a
  whole-number margin above the boundary.
- Below it the ratchet **refuses to decide conformally** and falls back to
  the deterministic raw point estimate (`BoundMethod::DeterministicFallback`),
  ledgered as such — the Alien-Artifact conservative-fallback contract. A
  perfect 10-item category holds a 0.95 floor instead of tripping a spurious
  red (`small_corpus_takes_the_deterministic_fallback`).

## 4. Validity assumptions (the ledger)

Every transparency card restates these; they are the conditions under which
the bound is honest:

1. **Exchangeability**: calibration items are i.i.d.-like draws from the
   deployment distribution. A corpus that over-samples easy pages inflates
   every bound — corpus composition is a review obligation, not a statistic.
2. **Bernoulli outcomes**: pass/fail per item, no partial credit. Graded
   metrics (CER) enter as thresholded pass/fail against their documented
   budget, never as raw means.
3. **Adequate n**: `n ≥ MIN_CALIBRATION_N` for the conformal path (§3).
4. **Fixed alpha**: 0.05, chosen once; shopping alpha after seeing the data
   invalidates the bound.

## 5. The transparency card

`transparency_card(&bound)` emits one JSON object per bound: the equation,
the substituted values (`s`, `f`, `n`, `alpha`, both instrument values, the
decision value, the method), the plain-English intuition, the assumptions
above, and **what would flip the decision**. The release log carries the card
inline so a reviewer can re-derive the decision by hand from the card alone.

## 6. Reference values (independent cross-check, 2026-07-06)

Verified against an independent stdlib-Python port of the same continued
fraction (`category_bound_matches_known_posteriors`):

| counts | Jeffreys lower | Hoeffding lower | decision |
|--------|----------------|-----------------|----------|
| 20/20 | 0.909523 | 0.726333 | 0.726333 (band binds) |
| 95/100 | 0.904229 | 0.827612 | 0.827612 (band binds) |

The Hoeffding band binds until `n` is large (it shrinks as `1/sqrt(n)` from a
distribution-free budget); the Beta posterior takes over on large corpora.
This is by design — small-n decisions should be the most conservative.
