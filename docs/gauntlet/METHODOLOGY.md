# METHODOLOGY.md — the three-pillar release gauntlet (ML-System-class)

> **Beads:** `VERIFY-three-pillar-cert` (= `bd-re8.13`), `VERIFY-conformal-ratchet` (= `bd-re8.14`),
> `VERIFY-eprocess-invariants` (= `bd-re8.15`). DESIGN/methodology artifact.
>
> This document is the design-of-record for how `franken_ocr` is **certified for release**. It
> operationalizes plan §8.5 (*the gauntlet — three-pillar release certification*) for this specific
> port, naming the exact oracle wiring, the exact conformal ratchet math, the exact e-process
> calibration, and the exact keep-gate. It is the methodology; the running scoreboard it reads is
> [`../FEATURE_PARITY.md`](../FEATURE_PARITY.md) (the FeatureUniverse / SurfaceMatrix), and its math is
> implemented and self-tested in [`../../scripts/gauntlet_cert.py`](../../scripts/gauntlet_cert.py).

`franken_ocr` is an **ML-System-class** port in the `/running-the-gauntlet-on-your-rust-port` taxonomy
— the same class as frankentorch, frankenjax, and franken_whisper. That class assignment is not
cosmetic: it fixes the oracle bridge (PyO3 in-process with `torch.use_deterministic_algorithms(True)`),
the tensor comparator (`TensorSpec` + a **per-op ULP tolerance table**, *not* a hand-guessed epsilon),
the five checkpoint-save + two distributed-collective boundaries that do not apply here (we are
single-process, single-shard), and the e-process invariant calibration split. Everything below is the
ML-System-class machinery, instantiated against the pinned Unlimited-OCR reference.

**Provenance.** Pinned source @ HF `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5` / GitHub
`7e98affeacba24e95562fbaa234ddb89b856874a`
([`../truth-pack/PINNED_SOURCES.md`](../truth-pack/PINNED_SOURCES.md);
SHA-256s in [`../truth-pack/SOURCE_HASHES.md`](../truth-pack/SOURCE_HASHES.md)). Reference environment:
`torch==2.10.0`, `transformers==4.57.1`, `Pillow==12.1.1`. Numeric invariant constants
(`m_max = 32896`, `W = 128`, `L = 12`, `K_max = 6848`, `vocab = 129280`) are line-backed in
[`../truth-pack/CENSUS.md`](../truth-pack/CENSUS.md) §(d) and SPEC-090..096.

---

## 0. The One Rule and the three pillars

> **The agent is forbidden from declaring victory on one pillar while another regresses.** A faster
> kernel that drifts the OCR output is reverted with no source landed (`docs/NEGATIVE_EVIDENCE.md`); a
> conformance fix that regresses decode-per-token is ledgered (`docs/PERF_LEDGER.md`); a surface feature
> that ships without its parity gate green stays `partial` and never rounds up. (AGENTS.md Doctrine #1:
> G1 > G2, parity first.)

The gauntlet decomposes the release question into three independently-measured pillars, each with its
own evidence bundle and its own gate:

| Pillar | Question | Evidence | Gate |
|--------|----------|----------|------|
| **(a) Performance** | Is each stage's honest ratio vs the proven CPU reference at or below its roofline, without a pass-over-pass regression? | `benches/gauntlet` per-stage ratios (§9.3 fairness controls) + `PERF_LEDGER.md` roofline columns + `.bench-history` | **keep-gate** (§5): `cv_pct ≤ 5`, MT8 frame ≥ 0.1%, both gates same run window, never torch@64 |
| **(b) Conformance** | Does the port produce the **same answer as the bf16 reference** for the same input, including under fault? | L0–L5 parity ladder (§8.2) + differential + metamorphic + the e-process invariant stream | **conformal lower-bound ratchet** (§3): release uses the LOWER bound; raise it without lowering any per-category bound |
| **(c) Surface parity** | What fraction of the reference's declared surface does the port implement, and what is explicitly excluded? | `FEATURE_PARITY.md` FeatureUniverse + SurfaceMatrix (`present \| partial \| missing \| n/a \| excluded`) | **feature-coverage gate** (§4): partial never rounds up; excluded still counts as coverage debt |

The three pillars share one kernel — **Subject / Oracle / Comparator** — described next.

---

## 1. The oracle wiring (Subject / Oracle / Comparator)

### 1.1 EngineIdentity — never compare the oracle against itself

Every comparator output carries a discriminator:

```
EngineIdentity::Subject  = "franken_ocr"          // the Rust port (focr engine)
EngineIdentity::Oracle   = "unlimited-ocr-oracle" // pinned torch reference via PyO3/subprocess
```

The two labels are **asserted-distinct at the comparator entry point**. This prevents the highest-value
false green in the whole gauntlet: a refactor that accidentally points both `subject` and `reference`
at the same engine (e.g. both at the torch reference, or both at the Rust path) — every test passes,
the suite reports 100%, and nothing was tested. The preflight doctor checks
`subject_identity == "franken_ocr"` and `reference_identity == "unlimited-ocr-oracle"` at harness entry
(operator `🪞 Engine-Identity-Guard`).

### 1.2 The PyO3 / subprocess bridge — test-only, never linked into `focr`

The reference is reached through a **test-only** bridge. The decisive constraint, from plan §8.5 and G3
(the no-FFI single-binary runtime claim): **this bridge is never linked into the shipping `focr`
inference binary.** It lives only under `tests/` / `benches/gauntlet`, behind a cfg/feature that the
release build does not enable. The shipping binary has no Python, no FFI, no network — the bridge is a
*verification* dependency, not a *runtime* one.

Determinism pinning is a **harness invariant**, set once in `setup()`, not per-test (a single test that
forgets it is silently nondeterministic):

```python
# pinned at harness start, via PyO3 / the subprocess reference:
torch.use_deterministic_algorithms(True)
torch.manual_seed(DEFAULT_SEED)            # seeded RNG captured per call
# config drift guard: assert the live versions match the pinned contract
assert transformers.__version__ == "4.57.1"
assert torch.__version__ == "2.10.0"
```

> **The split oracle (OQ-17).** The official `infer()` path is CUDA-oriented (`.cuda()` + CUDA
> autocast), so a CPU bf16 HF oracle is **not guaranteed to run as-is**. Per plan §8.1 we therefore
> SPLIT the oracle: **(correctness)** golden fixtures come from the unmodified official model on a CUDA
> host, frozen once and committed — parity NEVER depends on CPU HF — and **(performance)** the CPU
> baseline is separate (CPU-patched HF if proven equivalent within the §8.2 nondeterminism floor, else
> the best CPU reference that actually runs: llama.cpp GGUF / ONNX Runtime / MLAS, labeled as such).
> The PyO3 determinism-pinning above governs the *correctness* oracle capture; the *perf* reference is
> driven by `benches/gauntlet` per §9.3.

### 1.3 The per-op ULP tolerance table — the L1/L2 comparator

The continuous-tensor comparator is **not** a single hand-guessed epsilon. It is a per-op ULP table
(units in the last place of IEEE-754 f32), stored verbatim in the contract file
`docs/contracts/ulp_tolerance_v1.toml` and applied by `TensorSpec`-normalized comparison. The
ML-System-class defaults, anchored to PyTorch's own `torch.testing.assert_close` tolerances:

| Op family | Tolerance | Where it bites in `franken_ocr` |
|-----------|-----------|---------------------------------|
| `matmul` f32 | **4 ULP** | SAM/CLIP attention & MLP GEMMs, projector 2048→1280, lm_head 1280→129280 |
| Elementwise (add, mul, …) | **2 ULP** | residual adds, RoPE rotation, masked-scatter fusion |
| Reductions (sum, mean) | **8 ULP** f32 | RMSNorm variance, MoE router softmax normalizer |
| Transcendentals (exp, log, sin, cos) | **8 ULP** | softmax `exp`, SiLU/quick_gelu `sigmoid`, RoPE sin/cos tables |
| `softmax` outputs | sum-to-1.0 within **1e-7** (f32) | online-softmax R-SWA, MoE gate, lm_head distribution |

`TensorSpec { shape, dtype, device, requires_grad, data_hash(BLAKE3) }` normalizes both sides before
comparison so a shape or dtype mismatch is caught before the numeric compare runs.

**Two non-defaults are load-bearing for this port:**

1. **The int8-vs-f32 *forward* drift is NOT a ULP question.** The ULP table governs `f32-Subject vs
   f32-Oracle` agreement (where the two should agree to a few ULP). The int8-quantized forward drifts
   by a *measured* amount that is a property of this model's shapes/depth — it is **not** the imported
   frankensearch BERT figure `0.055`, and it is **not** a ULP. L3 logits are compared within that
   *measured* int8 budget while requiring **exact argmax / exact token** where the reference is
   deterministic (§8.2 / SPEC). The ULP table and the int8 budget are two different comparators applied
   at two different ladder rungs.

2. **The SIMD path needs no tolerance vs scalar — it is bit-identical.** Integer add is associative, so
   the int8 i32-accumulate GEMM is *exactly* the same across SDOT / SMMLA / VNNI-512 / VNNI-256 /
   scalar (the one exception, AVX2 `vpmaddubsw` i16-saturation, carries its own overflow proof or a
   ledgered `DISCREPANCIES.md` divergence — plan §5.4). This is monitored as an **e-process invariant**
   (INV-SIMD-SCALAR, §6), not by a tolerance.

> **`🎚 Raise-ULP-Tolerance` is gated.** A ULP-tolerance change must be justified, scoped to the
> specific operator, and accompanied by the before/after max-rel-error snapshot. Loosening the table to
> make a test pass is the ML-class anti-pattern; the table is a contract, bumped by a bead, not a knob.

### 1.4 The L0–L5 parity ladder (the conformance evidence)

The ladder (plan §8.2) is the conformance pillar's per-rung evidence. Each rung's comparator is named:

| Gate | Granularity | Comparator |
|------|-------------|------------|
| **L0** preprocessing | resized/normalized/padded tensor, tile geometry | **exact** (gray pad 127, ratio selection, `[-1,1]` normalize) |
| **L1** per-op | each kernel vs oracle activation | per-op ULP table (§1.3); cosine ≥ 0.9999 f32 |
| **L2** per-layer | per decoder-layer hidden state, per vision-stage output | per-op ULP table; max-abs-diff ledgered |
| **L3** logits | pre-sampling logits | **measured** int8 budget + **argmax must match** where deterministic |
| **L4** token | decoded token sequence | **exact** under greedy where reference deterministic |
| **L5** end-to-end OCR | decoded text + bbox on golden corpus | exact-match where deterministic; CER / TEDS / Formula-CDM within documented budget |

> **The oracle's own nondeterminism floor is established FIRST** (bd-re8.2). The bf16 / 129280-vocab
> argmax reference is frequently nondeterministic across torch thread counts / BLAS reduction order at
> the logit-tie level. Run the oracle twice, at two thread counts, over the golden corpus; record the
> nondeterminism envelope (per-token divergence rate, first-divergence position) as a committed
> fixture. **L4 "exact" is defined only over the prefix the oracle reproduces identically**, and the
> L3 budget is *derived from* the measured oracle variance. A `franken_ocr` int8 divergence *inside the
> oracle's own bf16 noise* is not a bug.

---

## 2. Pillar (c) — the FeatureUniverse / SurfaceMatrix

Surface parity is measured by [`../FEATURE_PARITY.md`](../FEATURE_PARITY.md): the single living table
that enumerates **every** modeling feature, op (§4.3), CLI surface (§7.2), robot event (§7.3), parity
gate (§8.2), and alien-artifact family (§9.7), each as a cell value:

```
present | partial | missing | n/a | excluded
```

It is split into two enumerated populations the gauntlet reads together:

- **FeatureUniverse** — the numbered modeling-feature / op / quant rows (`#1..#128`), the unit of the
  Beta-posterior parity score (§3).
- **SurfaceMatrix** — the un-numbered CLI / robot / gauntlet / alien rows, proven by contract tests
  (`SURF`), not the numeric ladder.

### 2.1 The four loader-enforced FeatureUniverse invariants

The score is meaningless unless these hold; the loader **rejects** the universe on violation:

1. **`partial` never rounds up to `present`.** A `partial` is half-credit in the Beta evidence (0.5),
   but its *reported* status is `partial`, never "0.5 present". (Conflating them is the K-4 anti-pattern
   in scoring form.)
2. **`excluded` still counts as coverage debt.** It is enumerated with a reason and a re-open
   condition in `FEATURE_PARITY.md` §16, not omitted. For a strict-100% claim, excluded debt must be
   retired or the claim is false. (The 5 v1 exclusions: `valid_img_tokens`, bbox-overlay drawing,
   geometry/`line_type`, `test_compress`, and `pdf` input.)
3. **Per-category weights sum to exactly 1.0.** `|Σ weights − 1.0| > 1e-9` is a load error.
   Approximately-1.0 is undefined behavior — it makes the per-category Beta evidence un-normalizable.
4. **Deterministic iteration order by FeatureId.** `present` rows are summed in a fixed order so the
   same source tree produces the same per-category score, the same global score, and the same SHA-256
   of the emitted scorecard on x86, ARM, and WASM. `HashMap` iteration is forbidden (non-deterministic
   order ⇒ non-deterministic f64 sum ⇒ flickering ratchet).

### 2.2 Categories and weights for `franken_ocr`

The FeatureUniverse categories follow the §1–§11 grouping of `FEATURE_PARITY.md`. The release-time
category weights (each category's rows are weight-normalized to sum to 1.0 *within* the category, then
the categories themselves carry a release weight reflecting blast-radius if broken):

| Category | Release weight | Rationale (blast radius for OCR exactness) |
|----------|---------------:|--------------------------------------------|
| Preprocess & prompt (§1) | 0.12 | L0 exactness gates every downstream rung; a pad/ratio bug corrupts everything |
| Tokenizer (§2) | 0.10 | token-id-exact is an L0/L4 prerequisite; a mismatch corrupts every gate |
| Vision SAM (§3) | 0.10 | quant here wrecks OCR (both prior quants keep it BF16) |
| Vision CLIP + bridge (§4) | 0.08 | concat order / projector is load-bearing for feature alignment |
| Connector (§5) | 0.08 | masked-scatter ordering invariant; misalignment silently corrupts |
| Decoder & MoE (§6) | 0.16 | the bulk of compute and the routed-expert correctness surface |
| **R-SWA ring buffer (§7)** | 0.14 | the centerpiece; the KV-cap + RoPE-true-position invariants live here |
| Sampler & postprocess (§8) | 0.10 | greedy/no_repeat_ngram + bbox rescale = the user-visible output |
| Op map — facade (§9) | 0.06 | the kernel surface the ladder rungs exercise |
| Quant recipe (§11) | 0.06 | the `.focrq` recipe invariant (high-precision set kept BF16) |

> Perf kernels (§10) are behind kill-switches and scored under Performance, not Conformance; the
> alien-artifact families (§15) are upside levers behind their own proof obligations and do not gate
> the strict release. Weights are declared once in `docs/contracts/focr_score_contract.toml` and held
> constant across the ratchet's lifetime (changing them mid-ratchet invalidates the high-water mark).

---

## 3. Pillar (b) gate — the conformal lower-bound release ratchet

This is the formal version of "parity gate first." The conformance pillar's release decision is **not**
the point-estimate pass rate; it is a **distribution-free conformal lower bound** on a per-category
Beta posterior, and a change may land **only if it raises the lower bound without lowering any
per-category bound**. Math implemented + self-tested in
[`../../scripts/gauntlet_cert.py`](../../scripts/gauntlet_cert.py).

### 3.1 Layer 1 — the Beta posterior per category

Each (category, feature) outcome is scored:

| Outcome | Success contribution | Failure contribution |
|---------|---------------------|----------------------|
| `present` (passing) | `1.0 × feature_weight` | 0 |
| `partial` | `0.5 × feature_weight` | `0.5 × feature_weight` (counts in **both** α and β) |
| `missing` | 0 | `1.0 × feature_weight` |
| `excluded` | (skipped; full weight stays in the denominator for strict-100% claims — see §2.1.2) | — |

The per-category pass rate is a Beta posterior with a uniform (Jeffreys-like) prior:

```
theta_c ~ Beta(alpha_prior + Σ weighted_successes,
                beta_prior  + Σ weighted_failures)
prior:  alpha_prior = beta_prior = 1.0   (uniform on [0,1]; declared in the score contract)
```

The point-estimate global score is the category-weighted sum of posterior means:

```
S_mean = Σ_c category_weight_c · E[theta_c]          where  E[theta_c] = alpha_c / (alpha_c + beta_c)
```

> **A partial contributes to BOTH α and β** (half-success + half-failure). Crediting it only to α biases
> the posterior up — the classic scoring pitfall.

### 3.2 Layer 2 — the distribution-free conformal band

The Beta posterior alone assumes outcomes are exchangeable Bernoulli draws. Real OCR-conformance
streams are not: they are **heavy-tailed** (a few dense-table / sub-script / code failures dominate),
**bimodal** (printed text passes ~100%, a specific failure class clusters), and **regime-shifting** (a
new fixture, a quant change, a kernel swap moves the distribution mid-run). Conformal prediction
(Vovk-Gammerman-Shafer 2005) gives finite-sample coverage `P(R_{n+1} ≤ q) ≥ 1 − α` **regardless of the
underlying distribution**. Cost: wider intervals. Benefit: honest under exactly the pathologies OCR
conformance exhibits.

Calibration recipe (held-out — never the cycle being scored, or coverage is anti-conservative):

1. Hold out a calibration set of conformance outcomes (deterministic split by `corpus_entry_id`;
   `n_cal ≥ 100`, `≥ 200` recommended).
2. Compute per-category nonconformity residuals `R_i = |observed_pass_rate_i − E[theta_c]|` from
   **prior** cycles.
3. Sort ascending; the `(1 − α)` empirical quantile `q = R_{⌈(1−α)(n_cal+1)⌉}` is the conformal
   half-width.
4. The conformal lower bound on the global score is `S_lower = max(0, S_mean − q)` (and per-category
   `theta_c^lower = max(0, E[theta_c] − q)`), at confidence `0.95` (the operating point that balances
   coverage against ratchet progress; `0.99` stalls — the band never moves).

### 3.3 Layer 3 — the release decision uses the LOWER bound

```
release-eligible  ⟺  truncate_score(S_lower) ≥ ratchet.current_lower_bound
                       AND  truncate_score(theta_c^lower) ≥ ratchet.per_category_bounds[c]  ∀ c
```

The release certificate's `parity_score` field is `S_lower`, **not** `S_mean`. Justification:

- **Asymmetric cost.** Shipping worse-than-advertised is far more expensive than better-than-advertised
  (for exact-token OCR, a regression on dense numerics is a silent data-corruption bug). The lower bound
  is the conservative side.
- **Adversarial reading.** A reviewer hostile to the claim asks "is the *lower* bound above the
  ratchet?" — if yes, the claim survives a hostile read.
- **Ratchet monotonicity.** Releasing on the point estimate lets noise occasionally bump the floor past
  where it deserves. The lower bound ensures every advance is supported by evidence at the coverage
  level. Intermediate dashboards MAY show `S_mean` ("are we trending up?"); the release decision uses
  `S_lower`.

### 3.4 `truncate_score` to 6 decimal places — cross-platform determinism

```rust
pub fn truncate_score(x: f64) -> f64 { (x * 1_000_000.0).floor() / 1_000_000.0 }
```

x86, ARM, and WASM differ at the LSB of IEEE-754 double; the difference propagates through the Beta
arithmetic and the quantile bisection, so two builds that should produce the same score differ at the
15th decimal place and the byte-wise ratchet diff *flickers*. The fix: **truncate (not round)** every
score to 6 decimal places at the boundary where it enters a comparison, a ratchet update, or a written
artifact. 6 places sits comfortably above the workload noise floor (`cv_pct` 3–5% ≫ 1e-6) and above the
cross-arch LSB drift. Truncation is associative across the ULP; rounding mode (banker's vs nearest-up)
is itself a cross-platform variable. **Where it is called:** the final `parity_score`, every
per-category bound, every field two architectures will diff. **Where it is NOT:** intermediate
computations (truncation compounds; only the leaf-of-output truncates) and the α/β sufficient
statistics (truncation loses information).

### 3.5 The ratchet state machine (`Allow | Block | Quarantine | Waiver`)

`reports/ratchet_state.json` is the committed, monotone high-water mark. Per run:

| Decision | Condition | Effect |
|----------|-----------|--------|
| **Allow** | `S_lower ≥` ratchet AND every per-category bound `≥` its persisted bound | Update `ratchet_state.json` (new bounds + commit_sha + timestamp + advance_reason). |
| **Block** | Any bound below ratchet and no active waiver | Exit non-zero; CI fails; the change cannot land. |
| **Quarantine** | Global holds but exactly one per-category bound dipped by ≤ 0.005 | Exit non-zero; block until the dip is resolved or a waiver recorded (7-day deadline). |
| **Waiver** | An active `waivers/<id>.toml` covers the specific (category, magnitude, expiry) | Allow with an auditable `[WAIVED-<id>]` trace. |

**Waiver discipline** (legitimate downgrades — a correctness fix that costs a perf-adjacent feature):
≥2 approvers in distinct roles, prose justification naming an incident or bead-id, evidence as artifact
paths (not narrative), ≤90-day expiry, and a specific scope (category + bound kind + new bound). On
expiry the ratchet reverts to demanding the old bound — forcing a real fix or an explicit renewal, never
quiet drift. A waiver never permanently lowers `ratchet_state.json`; it carves a temporary, visible
exception.

### 3.6 Worked numbers for `franken_ocr` (illustrative, not yet measured)

At a hypothetical mature round, suppose the R-SWA category (§7, weight 0.14) is `Beta(95, 7)` after the
ring-buffer beads land (point mean 0.931, Beta 95% lower 0.866) and the conformal half-width from prior
cycles is `q = 0.048`:

```
theta_RSWA^lower = max(0, 0.931 − 0.048) = 0.883            # conformal lower, this category
```

If a speculative-decode lever (AF-3) raises `S_mean` from 0.951 → 0.958 but the conformal half-width
*widens* to `q = 0.071` (because the draft-vs-full margin introduces a heavy tail on dense tables), then
`S_lower = 0.958 − 0.071 = 0.887 < 0.903` (the persisted floor) → **Block**. The point estimate went up;
the *honest* lower bound went down; the ratchet correctly refuses. This is the conformal band doing its
job: it refuses to release when the residual distribution grows a tail the Beta model didn't capture.

---

## 4. Pillar (c) gate — feature-coverage release gate

The surface gate reads `FEATURE_PARITY.md` and enforces:

- **MUST coverage ≥ 0.95** — every MUST `[SPEC-NNN]` clause has a row, and ≥ 0.95 of the MUST rows are
  `present` (their `Parity` gate green) to claim conformance. (Phase −1 seed: every row `missing`, so
  the value now is the *complete enumeration* — the gauntlet can only account what is listed.)
- **No `present` cell whose `Parity` gate (L0–L5 / SURF) is not green.** A row flips
  `missing → partial → present` only as its delivering bead lands AND its gate turns green.
- **`excluded` rows are reasoned coverage debt**, cross-checked: a `Missing` measured against a
  `supported` declaration is a regression; a `Passing` measured against an `excluded` declaration is an
  inconsistency (drop the exclusion).
- **The doc-lint contract** (`FEATURE_PARITY.md` §0): the file parses into the FeatureUniverse table;
  the lint emits one NDJSON line `{doc, n_features, n_present, n_partial, n_missing, n_excluded,
  must_coverage}` and fails on any malformed row or any MUST clause without a row.

---

## 5. Pillar (a) gate — the keep-gate for every perf claim

No performance claim counts unless it clears the keep-gate (plan §9.2; AGENTS.md Doctrine #8). Each row
is **non-negotiable**:

| Rule | Requirement |
|------|-------------|
| **Profile-first** | Evidence the touched code is ≥ **0.1% self-time** *before* the source touch (a profile frame, quoted). Below 0.1% is the **micro-lever trap**. |
| **MT8 attribution** (`⤴`) | The kept win names a specific ≥ 0.1% profile frame — "closed the 0.31% `rswa::online_softmax` reference-block residual", with the citation. |
| **Both gates, same run window** (`🔁`) | The focused microbench *and* the broad end-to-end bench moved in the **same git state, same `target/`, same machine, same minute**. A focused win that doesn't move the broad gate is suspect. |
| **`release-perf` profile only** | `[profile.release-perf]` (`debug=line-tables-only, lto=thin, codegen-units=1`). Never benchmark a debug or default-release build. |
| **`cv_pct` reported** | Every microbench reports its coefficient of variation. **`cv_pct > 5` is noise** and ineligible for keep — it does not enter the ratchet. |
| **Pass-over-pass ratchet** | `.bench-history` thresholds: primary regression ≥ −3%, geomean ≥ −5%, per-category geomean ≥ −10%, p90 ≥ −15%, throughput ≥ −5%. A regression past these blocks the bench gate. |

### 5.1 Head-to-head fairness controls (§9.3) — all mandatory

The honest ratio is `focr / reference` per stage (preprocess / vision-encode / prefill /
decode-per-token), tagged OK / warn / slower / "focr faster" — never a self-relative number. It is only
meaningful with **all** of:

- **Thread parity** — pin `OMP_NUM_THREADS` / torch `set_num_threads(N)` **equal to** focr's thread
  budget. **NEVER benchmark torch at @64** — oversubscription inflates fake "wins" (a hardened
  frankentorch lesson). Measure at @8 / @32, and let the §9.7 USL fit cap the decode pool at its peak,
  not at `num_cpus`.
- **Allocator fairness** — build focr with the same allocator posture used for the claim (mimalloc
  behind a feature), wired into the measured binary, not merely mentioned.
- **Best-of-N with warmup discard** — report the min and the precision of each side.
- **Precision annotation per row** — `focr-int8` vs `torch-bf16` (and `torch-int8` if available). A raw
  ratio across different numerics is meaningless without it.

> **The honest target is per-stage, not end-to-end.** Decode-per-token must be **faster than the proven
> CPU reference** on the primary arches (the gating part). Vision-prefill **parity-or-slower in f32 v1
> is acceptable and recorded honestly** (§1.1 G2). End-to-end-faster is a tracked *stretch*, not a gate.

### 5.2 Evidence is reconstructed, not trusted

A summary number is never the evidence. The producer records enough bounded, hash-bound material for
`gauntlet_row.py` and `gauntlet_cert.py` to replay the claim independently:

| Contract | Producer obligation | Row / certificate replay |
|----------|---------------------|--------------------------|
| **Raw timing** (`focr-gauntlet-raw-timing/v1`) | Emit one record per measured run. The focr side binds the exact meta, stdout, and stderr bytes; the reference side binds each stage sample and the deterministic text SHA-256. Warmups are identified and excluded. | Reconstruct the per-stage sample vectors and recompute `n`, best, median, mean, `cv_pct`, and token-normalized decode time. Any aggregate/raw disagreement, missing run, identity drift, or unhashed raw file refuses the row. |
| **Physical OCR comparison** (`focr-ocr-comparison/v1`) | Bundle the exact reference text and every measured focr stdout. Record the named transforms, normalization, integer edit distances, denominators, and input bindings. | Re-read the physical bytes, require identical focr output across timed runs and the reference hash on every reference invocation, reapply `identity-v1` / `strip-unlimited-det-spans-v1`, and recompute normalized Levenshtein CER from the authoritative integers. A prose CER claim or JSON receipt without its source bytes is ineligible. |
| **Trusted subject build** (`focr-source-input-manifest/v1`, `focr-build-receipt/v1`) | Before compilation, capture the clean source closure for the workspace and reachable local path dependencies, including effective Cargo configuration. Bind the exact RCH `--locked --profile release-perf` command, target, toolchain, Rust flags, manifest root, and binary hash/size. | Recompute the source root and require the current closure to match; reject dirty tracked or untracked build inputs, source drift during compilation, a reused output directory, a mismatched binary, or a non-`release-perf` receipt. `gauntlet_focr.sh` executes an evidence-local copy of the receipted binary and checks the source binary, copy, receipt, and model before and after capture. |
| **Pinned reference inference** (`focr-reference-model-manifest/v1`, `focr-reference-inference-binding/v1`) | Hash the exact 12-file Unlimited-OCR runtime model against the truth pack before and after inference; validate the 2,710-entry index and its single registered shard. Bind the model-root hash, page, entry/setup Python sources, argv, thread pins, backend, precision, package versions, and a newly created evidence-local `HF_MODULES_CACHE`. | Require the canonical manifest and inference binding in the bundled reference timing document, rederive their domain-separated roots, and reject a missing/extra file, changed page/source/model, stale modules cache, unregistered entry, or stack drift. |
| **Source-to-evidence lineage** (`focr-gauntlet-row/v3`) | Record `source_git_head`, the build manifest's `source_root`, a domain-separated `producer_root` over the exact gauntlet validators/producers/config, and one canonical `allowed_evidence_path` below `artifacts/perf/`. | Recompute both roots from the source commit. The final evidence HEAD must equal the source commit or descend from it; every path touched by every intervening commit must remain inside the one declared evidence subtree. Source, validator, runbook/config, ledger, rename, merge, and source-pack drift fail closed. There is no generic ledger exception. |

The exact reference-model set is: `config.json`, `configuration_deepseek_v2.py`, `conversation.py`,
`deepencoder.py`, `model-00001-of-000001.safetensors`, `model.safetensors.index.json`,
`modeling_deepseekv2.py`, `modeling_unlimitedocr.py`, `processor_config.json`,
`special_tokens_map.json`, `tokenizer.json`, and `tokenizer_config.json`. Their exact sizes and SHA-256s
are the entries in [`../truth-pack/SOURCE_HASHES.md`](../truth-pack/SOURCE_HASHES.md); a directory that
merely contains files with those names is not sufficient.

The resulting performance evidence bundle contains the raw timing observations, physical correctness
inputs, build/source receipts, captured subject identity, reference manifest, and inference binding.
Certification replays those bundled bytes and does **not** need the external 6.67 GB model directory.
The model and pinned Python environment remain mandatory to *produce or regenerate* reference evidence;
the bundle is a verification artifact, not a substitute for rerunning inference.

The v3 producer root is
`sha256("focr-gauntlet-producer-root/v1\\0" || sorted(path, size, sha256))` over the
committed timing, reference, roofline, row, certificate, correctness, ledger-check, and runbook inputs
listed by `PRODUCER_PATHS` in both independent validators. A dirty live producer is refused at row
generation. At certification, Git ancestry is checked commit by commit so a validator/config edit that
was later reverted cannot hide in the endpoint tree. An operator who changes `docs/PERF_LEDGER.md` must
make that change part of a new source commit and recapture; `APPLY=1` is therefore refused by the strict
runbook rather than treated as an evidence-only change.

`--smoke`, synthetic stub captures, and plumbing self-tests are deliberately stamped
synthetic/non-citable. They can prove that the harness is wired, but `gauntlet_row.py` refuses them and
they cannot satisfy a performance, parity, readiness, or release cell.

### 5.3 Residual provenance limits

The ordinary drift model is fail-closed, but two stronger adversarial claims remain out of scope. The
pre/post snapshots cannot detect a hostile actor that mutates bytes, lets the compiler or interpreter
consume them, and restores the exact original identity before the post-check. The RCH receipt binds the
requested command, local source closure, toolchain/configuration, and returned binary, but it is not a
cryptographic attestation from the remote worker that those exact bytes were the only compilation
inputs. These limitations must not be described as solved supply-chain attestation.

---

## 6. The e-processes (Ville) — anytime-valid monitoring of the load-bearing invariants

The four load-bearing invariants are not asserted once and forgotten; they are monitored as
**anytime-valid e-processes** (Howard-Ramdas-McAuliffe-Sekhon 2021) over an **unbounded** test stream.
Each invariant emits one observation per operation (`0` = held, `1` = violated); an e-process
accumulates an e-value `E_t`; **Ville's inequality** guarantees

```
P_{H_0}(∃t : E_t ≥ 1/α) ≤ α
```

so the harness can **check after every operation and reject the null the moment `E_t ≥ 1/α` — with no
Bonferroni correction** (the whole point: classical fixed-N tests inflate Type-I to `1 − (1−α)^N` under
repeated peeking; e-processes are designed for "watch forever, stop on first genuine violation"). Math
implemented + self-tested in [`../../scripts/gauntlet_cert.py`](../../scripts/gauntlet_cert.py).

### 6.1 The update rule and global e-value

```
E_t = E_{t-1} · ( (1 − λ) + λ · x_t / p0 )        E_0 = 1,   x_t ∈ {0, 1}
rejected  ⟺  E_t ≥ 1/α
```

This is a non-negative supermartingale under `H_0` (`E[(1−λ) + λ·x_t/p0 | F_{t-1}] ≤ (1−λ) + λ = 1`), so
Ville applies. The global e-value across the invariants is the **arithmetic mean**:

```
E_global(t) = (1/N) · Σ_i E_i(t)
```

— which is itself an e-process under the global null **regardless of dependence** between invariants
(sum of supermartingales is a supermartingale; division by a constant preserves it). Geometric mean or
product would require independence; max loses information. The orchestrator watches **both** `E_global`
(family-wise rejection) and each `E_i` (single-invariant triage at lower confidence).

### 6.2 The four `franken_ocr` invariants and their calibration

The calibration splits **hardware-enforced** invariants (guaranteed by the CPU's integer semantics /
the deterministic-flag — a violation is almost certainly a CPU/logic bug, so the prior is tight) from
**software-enforced** (guaranteed by a code path — violations are rare but plausible under benign
causes, so the prior is looser).

| Invariant | Statement (line-backed) | Class | `p0` | `λ` | `α` | `1/α` |
|-----------|-------------------------|-------|-----:|----:|----:|------:|
| **INV-KV-CAP** | KV cache never exceeds `L·(m + 128)`: with `L = 12`, `W = 128`, worst-case `m_max = 32768 + 128 = 32896` keys (CENSUS §(d); SPEC-094). The ring overwrites in place at `slot = prefill_len + ring_pos` and never grows. | Software (code-path enforced; ring buffer is preallocated, `m_max = 32896`) | `1e-6` | `0.9` | `0.001` | `1,000` |
| **INV-I32-NOOVERFLOW** | The int8 GEMM i32 accumulator never overflows `i32::MAX = 2,147,483,647`. Worst-case `|acc|` at `K_max = 6848`: signed×signed `≤ K·127²`, U8S8/VNNI `≤ K·255·127` — both fit i32 with **≥ 9× headroom** (plan §5.4). A unit test asserts the i32 result equals an i64 reference at `K = 6848` on every kernel/arch. | Software (overflow is a code/shape property, not CPU-guaranteed; the AVX2 i16-saturation path is excluded — it carries its own proof) | `1e-6` | `0.9` | `0.001` | `1,000` |
| **INV-DETERMINISM** | Same input twice → byte-identical output (decoded text + logits + robot NDJSON, scrubbed of timing/run-id). The determinism gate (§8.2). | Hardware (deterministic kernels + fixed greedy sampling; a divergence is a genuine nondeterminism bug, extremely rare) | `1e-9` | `0.999` | `1e-6` | `1,000,000` |
| **INV-SIMD-SCALAR** | The dispatched SIMD int8 path (SDOT / SMMLA / VNNI-512 / VNNI-256) is **bit-identical** to the scalar floor — integer add is associative, so SIMD == scalar *exactly* (plan §5.4 / §4.3). | Hardware (CPU integer-add semantics; a mismatch is a CPU or codegen bug — `p0 = 1e-9`) | `1e-9` | `0.999` | `1e-6` | `1,000,000` |

> **Why the split matters numerically** (figures verified in `gauntlet_cert.py --self-test`). Under
> **hardware** calibration (`p0 = 1e-9`, `λ = 0.999`, threshold `1e6`) one healthy observation multiplies
> `E_t` by `1 − λ = 0.001` (decay) and one violation multiplies by `(1 − λ) + λ/p0 ≈ 9.99e8`. A *single*
> violation against a `1e-9` null is fully consistent with the null and does **not** reject (`E_t` may
> have decayed far below 1 first); a *burst* of a few consecutive violations drives `E_t` past `1e6`
> within ~4 observations. Under **software** calibration (`p0 = 1e-6`, `λ = 0.9`, threshold `1e3`) one
> violation multiplies by `(1 − λ) + λ/p0 ≈ 9e5`, which **already exceeds `1e3`** — so a software
> invariant alarms on the *first* genuine violation, while the e-VALUE decays back toward 0 under
> sustained health (`λ = 0.9` forgives benign noise in the trajectory; the rejection latch fires on the
> first crossing). Calibrating a CAS-grade invariant like INV-SIMD-SCALAR as if it were software
> (`p0 = 1e-6`) would blunt its alarm by 1000× and let a genuine bit-divergence accumulate before
> crossing; that is the central e-process pitfall.

### 6.3 Operational discipline (the e-process pitfalls, made rules)

- **Emit on every operation, both `0` and `1`.** If only violations (`x = 1`) are ever fed, the e-value
  never decays under health and eventually crosses `1/α` from random-walk noise. Every GEMM, every
  decode step, every forward feeds an observation.
- **Never reset `E_t` to "avoid runaway".** The supermartingale property requires uninterrupted
  accumulation; a reset breaks Ville's inequality. Persist `E_t` to `eprocess_state.json` per soak
  round and resume on restart (or accept fresh-process semantics and document it).
- **Wire the drift monitor to all paths, not just the slow path.** Sampling only the fallback biases
  the stream; the fast int8 kernel's behavior must also update the e-value.
- **Per-invariant `α` looser than global is fine** (`α = 0.001` per-invariant under a `1e-6` global
  threshold satisfies the union bound); the reverse makes the global level meaningless.

The e-process stream runs under the soak campaign (fault / crash-boundary / fuzz / the `many_pages`
watchdog, plan §6.5) — the place where an invariant violation is most likely to surface across an
unbounded observation count.

---

## 7. Convergence — ≥10 rounds, ≥2 consecutive clean

The gauntlet is run to **convergence**, not to a deadline:

1. **Minimum 10 full rounds** of the perf / conformance / surface loop (the Phase 5–11 reapply loop).
2. **Two consecutive clean rounds** — each producing **< 3 new genuine findings** (computed across the
   three negative-evidence ledgers + every per-bucket findings file).
3. **Every open hypothesis resolved** — the per-pillar hypothesis ledgers are empty (each closed with a
   theory-kill or a remediation bead).

A `convergence-tracker` computes the round-over-round new-finding counts and **exits non-zero until all
three conditions hold**; it is wired as the CI gate that lets Phase 16 (final artifacts +
certification bundle) run. Compaction-survival: these markdown files are the source of truth, so the
agent can drop back in mid-run.

---

## 8. The strict release certificate

A `strict-conformant-release.v1` certificate ships only when **all four constants** hold exactly:

```
CERTIFICATION_MIN_VERIFICATION_PCT            = 100.0   # every required ProofObligation satisfied
CERTIFICATION_REQUIRED_SUITE_PASS_RATE_PCT    = 100.0   # certifying suite 100% (not "all but 3 flaky")
CERTIFICATION_MAX_HIGH_SEVERITY_COUNTEREXAMPLES = 0     # zero TrueDivergence; zero open critical bead
CERTIFICATION_MAX_EVIDENCE_AGE_HOURS          = 24      # every cited artifact fresh within 24h
```

and the bundle carries: the confidence-gate JSON (`release_decision == "Allow"`), the verification
contract (every row `pass`), the release certificate (≥3 distinct signers, detached-PGP-signed over the
Merkle root of the bundle), the CI manifest, the benchmark summary (all 5 pass-over-pass thresholds),
`scorecards.json` (per-category Beta + conformal lower bound), the critical-path report
(`open == 0 AND waived == 0` for High/Critical), and `ratchet_state.json` (internally consistent,
monotone). The certificate's `parity_score` field is the **`truncate_score`'d conformal LOWER bound** of
§3, not the point estimate. A failure of any single gate **blocks**; there is no partial-strict variant
(a release either meets the bar or is shipped as a clearly-labeled `provisional-release.v1` with the
specific deviations enumerated).

The certificate is verified from the selected bundle rather than from live external weights or an
operator's filesystem paths. All cited raw timings, OCR source texts, subject build receipts, and
reference provenance records must be inside and hash-covered by that bundle. A dirty source/build
closure cannot produce eligible evidence, and a dirty live worktree cannot certify an otherwise valid
bundle. The signed certificate binds `evidence_git_head` to the final evidence commit. Generated
certificate material is allowed only under the git-ignored `.gauntlet-output/` tree; it is never written
into the tracked `docs/gauntlet/bundle/` snapshot.

> **`franken_ocr`-specific ship gates layered on top** (plan §10 Phase 5, AGENTS.md): no red parity
> cell and no unledgered divergence; decode-per-token faster than the proven CPU reference on the
> primary arches; vision-prefill ratio recorded honestly; `--version` carries the Baidu MIT
> attribution; all six portable runtime-dispatch distribution jobs green (four
> Unix/macOS plus native Windows x86-64 and ARM64); both Linux assets linked and
> `readelf`-audited for a glibc 2.17 floor; both Windows assets installed offline
> through `install.ps1` with exact-byte and failed-replacement preservation
> proofs; the robot NDJSON contract test passing against its frozen schema; the
> `pdf` row remains `excluded` with its re-open condition intact. A tag build is
> ineligible unless its exact `vX.Y.Z` name matches `Cargo.toml` and its commit is
> reachable from `origin/main`.

> **Current status is distribution-pending, not certified.** These evidence-integrity contracts do not
> turn the present tree green. The exact Torch routing candidate now emits EOS on `page0590` and passes
> the complete local 20-page CER budget; its hard-page CER tail remains explicitly documented. The
> compatible artifact is still unpublished, and the clean-tree bundle, platform distribution, signing,
> and remaining readiness cells must pass before any strict release certificate may be issued.

---

## 9. How the pieces connect (the gauntlet dataflow)

```
truth-pack/  (pinned source, SHA-256, census constants, OQ answers)
    │
    ├──► scripts/gen_reference_fixtures.py ──► golden fixtures (GPU-correctness oracle, frozen)
    │                                          + oracle nondeterminism floor (run twice / 2 threads)
    │
    ▼
PyO3/subprocess oracle bridge  (test-only, EngineIdentity::{Subject,Oracle}, use_deterministic_algorithms)
    │   per-op ULP table (docs/contracts/ulp_tolerance_v1.toml)  +  measured int8 budget
    ▼
L0─L5 parity ladder ──┐
metamorphic + diff ───┤──► CONFORMANCE pillar ──► scripts/gauntlet_cert.py (Beta + conformal band)
e-process invariants ─┘                                  │
                                                         ▼
FEATURE_PARITY.md (FeatureUniverse/SurfaceMatrix) ──► SURFACE pillar ──► feature-coverage gate
                                                         │
benches/gauntlet (per-stage ratios, §9.3 fairness) ──► PERFORMANCE pillar ──► keep-gate + .bench-history
                                                         │
                                                         ▼
            convergence-tracker (≥10 rounds, ≥2 clean) ──► ratchet (Allow|Block|Quarantine|Waiver)
                                                         │
                                                         ▼
                 strict-conformant-release.v1 certificate bundle  (parity_score = conformal LOWER bound)
```

The Rust homes of this machinery (per `PROPOSED_ARCHITECTURE.md`): `conformance.rs` (tolerance structs
+ L0–L5 validator traits + rollout stages, with the KV-cap and SIMD==scalar invariants registered),
`rswa.rs` (the INV-KV-CAP / INV-DETERMINISM observation source, the centerpiece), `nn.rs` (the
INV-I32-NOOVERFLOW / INV-SIMD-SCALAR observation source), `tests/` + `benches/gauntlet` (the test-only
oracle bridge, never linked into `focr`).

---

## 10. Anti-patterns this methodology forbids

| Anti-pattern | Symptom | Guard |
|--------------|---------|-------|
| Oracle compared against itself | Apparent 100% pass | `EngineIdentity::{Subject,Oracle}` asserted-distinct; preflight doctor checks the labels |
| Release on the point estimate | A regression ships because `S_mean` "looks great" | Release uses the conformal LOWER bound (§3.3) |
| `cv_pct` dropped from the report | Noise looks like signal | Every microbench reports `cv_pct`; `> 5` is ineligible (§5) |
| torch benched at @64 | Inflated fake "wins" | Thread parity; measure at @8 / @32 (§5.1) |
| Loosening the ULP table to pass | Silent numeric drift | `🎚 Raise-ULP-Tolerance` requires per-op justification + a max-rel-error snapshot (§1.3) |
| Resetting the e-value | Lost evidence; broken Ville bound | Never reset; persist and resume (§6.3) |
| Calibrating SIMD==scalar as software | 1000× blunted alarm | Hardware calibration `p0 = 1e-9` for INV-SIMD-SCALAR / INV-DETERMINISM (§6.2) |
| `partial` rounded up to `present` | Overstated surface parity | `partial` is 0.5 in the Beta evidence but never reported as `present` (§2.1.1) |
| `excluded` silently omitted | Hidden coverage debt | Every exclusion enumerated with a reason + re-open condition (§2.1.2) |
| Inheriting frankensearch's `0.055` int8 budget | Wrong tolerance for this model | The int8 budget is **measured for this model**, derived from the oracle's own bf16 floor (§1.3, §1.4) |
| Trusting aggregate timing JSON | Edited medians/CV can masquerade as measurements | Reconstruct every aggregate from hash-bound `focr-gauntlet-raw-timing/v1` observations (§5.2) |
| Trusting a CER string or receipt alone | Self-attested correctness without the compared text | Bundle both physical texts and replay the full `focr-ocr-comparison/v1` calculation (§5.2) |
| Timing an arbitrary target-tree binary | A rebuild can silently change the measured subject | Clean producer receipt plus an evidence-local binary copy with pre/post identity checks (§5.2) |
| Treating a model directory name as provenance | Different weights or remote code can impersonate the pin | Exact 12-file model manifest + inference-source binding + fresh `HF_MODULES_CACHE` (§5.2) |
| Certifying a convenient descendant HEAD | Validator/config/source changes can inherit old measurements | Row v3 producer/source roots plus commit-by-commit, one-subtree evidence lineage (§5.2) |
| Generating certificates into tracked docs | Generated control files dirty or overwrite their own inputs | Generation is confined to ignored `.gauntlet-output/`; the certificate binds the final evidence HEAD (§8) |
| Citing smoke output | Plumbing success is mistaken for release evidence | Smoke/synthetic records are stamped non-citable and refused by the normal row path (§5.2) |

---

*End of methodology. The gauntlet is the conscience that keeps every speed and parity claim honest: it
reads [`../FEATURE_PARITY.md`](../FEATURE_PARITY.md), runs the math in
[`../../scripts/gauntlet_cert.py`](../../scripts/gauntlet_cert.py), and certifies a release only when
all three pillars are green at the conformal lower bound across ≥10 converged rounds. Beads
`VERIFY-three-pillar-cert` / `VERIFY-conformal-ratchet` / `VERIFY-eprocess-invariants`
(= `bd-re8.13/14/15`).*
