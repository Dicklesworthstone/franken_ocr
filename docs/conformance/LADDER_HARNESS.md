# LADDER_HARNESS.md — how the L0–L5 ladder + oracle differential RUNS

> **Scope.** [`PARITY_LADDER.md`](PARITY_LADDER.md) is the *design* (what each gate
> compares, at what granularity, against which oracle, within what tolerance).
> **This document is the *harness*** — how the Rust test code in
> [`../../tests/parity_ladder.rs`](../../tests/parity_ladder.rs) (the rungs) +
> [`../../tests/support/parity_harness.rs`](../../tests/support/parity_harness.rs)
> (the shared comparator) actually executes that design: what runs always-on,
> what is gated, how a skip stays honest, and the COVERAGE / DISCREPANCIES /
> PROVENANCE discipline the suite holds itself to.
>
> Beads: the rungs implement `VERIFY-ladder-l0` (`bd-re8.4`) …
> `VERIFY-ladder-l5` (`bd-re8.7`) and `VERIFY-differential-suite` (`bd-re8.9`);
> the comparator is the shared infra those beads (and the golden suite
> `bd-re8.11`) reuse — it does not re-invent the cosine/ULP comparator.

---

## 0. The two files and the firewall between them

| File | Role | Compiles & self-tests without weights/fixtures? |
|------|------|:--:|
| `tests/support/parity_harness.rs` | **pure comparator infra** — `NormalizedValue`/`TensorSpec`, `cosine`, the per-op ULP table, the non-determinism scrubbers, the fixture loader, the oracle-nondeterminism-floor helper, the structured logger | **yes** — it depends on nothing in `src/` that is mid-flux; it works on `&[f32]` / `serde_json::Value`. Its inline `#[cfg(test)]` unit tests exercise the comparator MATH on **synthetic vectors only**. |
| `tests/parity_ladder.rs` | the **L0–L5 rung skeletons** + the oracle-differential test, plus the always-on stable-surface anchors | **yes** — the rungs run their gating logic and emit a structured line every time; the oracle RUNS are gated. |

`parity_ladder.rs` declares the comparator with
`#[path="support/parity_harness.rs"] mod parity_harness;` so there is exactly one
copy of the comparator, shared by every rung (and available to the golden suite to
reuse, per [`GOLDEN.md`](GOLDEN.md) §2C: "reuses the comparator from
`VERIFY-ladder-l1-l2`, it does not re-invent it").

**The firewall:** the comparator never reaches into `src/native_engine` internals
(those are owned by other agents and mid-flux). It targets the **genuinely-stable
surface** — the error exit codes (`src/error.rs`), the robot schema (`src/robot.rs`),
and the frozen oracle fixture format — so the harness compiles and its math is
proven *today*, while the engine seam-capture API it WILL call is still landing.

---

## 1. The two execution modes of every rung

Each rung (`l0_preprocess_exact` … `l5_end_to_end_cer_budget`, plus
`differential_per_op_vs_bf16_oracle`) has the **same shape**:

```
1. Logger::setup(seed)                         # structured-log scope opens
2. ALWAYS-ON contract anchors                  # the rung's pure/deterministic checks
                                               #   (run on every box, no oracle)
3. gate:  if !fixtures_present() || !model_present()
              Logger::skip_no_model(reason)    # SUCCESS line: WHY it skipped,
              Logger::result("skip_no_model")  #   + native_path proof (/nonexistent)
              return                            # green, not a fake pass
4. FIXTURE+MODEL PRESENT: the live oracle compare, logging parity rows
5. Logger::result(outcome, elapsed_us)         # terminal line
```

- **Always-on (step 2)** is the part that needs no oracle because *the reference is
  deterministic*: L0's gray-pad-127 / `[-1,1]` / 273-token-census constants, L3's
  "the tolerance is DERIVED from the §2 floor, never the imported 0.055" derivation
  on a synthetic two-run pair, L4's exact-prefix logic on synthetic streams, L5's
  CER metric on synthetic strings, and the differential row-shape + EngineIdentity
  guard. These give the suite **real, always-green coverage on a CI box with no
  model** (the GOLDEN.md §3 principle: the surface/fuzzy legs run without weights).
- **Gated (step 3)** is the part that needs the CUDA-host oracle fixtures and/or
  the 6.67 GB weights. Absent ⇒ **skip-with-SUCCESS**, never a silent skip.

### The skip is a SUCCESS line, not a hole

A model/fixture-gated skip emits `event=skip, result=skip_no_model` carrying:

```jsonc
{"schema_version":1,"ts":"[ts]","test":"L5_e2e","case":"corpus","run_seq":3,
 "event":"skip","result":"skip_no_model",
 "reason":"L5 end-to-end OCR compare needs the golden <doc>_reference.json …",
 "native_path_ran":false,"fallback_target":"/nonexistent"}
```

`native_path_ran` + `fallback_target:"/nonexistent"` are the **native-path proof**
from the frozen `tests/fixtures/test_log_schema.json` (`native_path_proof`): a skip
that *claims* to have run the native path must prove it by pointing the fallback at
`/nonexistent`, so a silently-skipped suite is never mistaken for a pass (AGENTS.md
Testing Policy; PARITY_LADDER §6.2/§7).

---

## 2. The comparator (what `parity_harness.rs` gives every rung)

| Piece | What it is | Backed by |
|-------|------------|-----------|
| `TensorSpec` / `NormalizedValue` | shape+dtype normalization; `check_against` rejects a shape/dtype mismatch **before** any numeric compare, naming both sides | METHODOLOGY §1.3 |
| `cosine` / `max_abs_diff` | the L1/L2 ≥ 0.9999 gate (f64-accumulated so the metric is more stable than what it measures) + the per-layer max-abs ledger | PARITY_LADDER §3.2 |
| `ulp_table` / `OpFamily` / `ulp_compare` / `ulp_distance` | the **per-op ULP table — 4 ULP f32 matmul, 2 ULP elementwise, 8 ULP reductions/transcendentals** — with a monotone-ordinal ULP metric and a self-diagnosing report (worst index + max-abs-diff) | METHODOLOGY §1.3 |
| `scrub_volatile` / `SCRUB_KEYS` | non-determinism scrubbers: mask `ts`/`*_ms`/`run_id`/abs-paths to stable placeholders, **keep the field present** (a dropped field still fails) | GOLDEN.md §2D/§5 |
| `FixtureLoader` / `read_npy_f32` / `ReferenceGolden` | reads `gen_reference_fixtures.py` output: the `<doc>_reference.json` golden + the `<stage>.npy` activations (a hand-rolled minimal `<f4` C-order npy reader), and **reports presence** so a rung can gate | PARITY_LADDER §1 |
| `OracleFloor` / `establish_floor` | **the keystone** — compare two oracle runs (per-logit spread + reproducible token prefix) and DERIVE the L3 tolerance + L4 exact-prefix from the measurement, never a guess | PARITY_LADDER §2 |
| `golden_diff` / `update_goldens` / `actual_sidecar` | hand-rolled `UPDATE_GOLDENS` golden loop: on mismatch write `<golden>.actual` + fail with the line diff; `UPDATE_GOLDENS=1` blesses | GOLDEN.md §4/§5 |
| `Logger` / `validate_event` / `load_log_schema` | structured NDJSON emission + a contract test that every emitted event validates against the frozen `test_log_schema.json` | the detailed-logging requirement |

### The two-comparator rule (do not conflate)

Two *different* comparators apply at two *different* rungs (METHODOLOGY §1.3
non-default #1):

- **The ULP table** governs `f32-Subject vs f32-Oracle` agreement (L1/L2) — where
  the two *should* agree to a few ULP.
- **The MEASURED int8 budget** governs the quantized forward (L3) — a *measured*
  property of this model's shapes/depth, **derived from `establish_floor`**, never
  a ULP and **never the imported frankensearch `0.055`**. The harness proves this
  derivation in L3's always-on path (it asserts the derived tolerance is not
  silently `0.055`).

---

## 3. COVERAGE matrix (rung × status)

The discipline (`/testing-conformance-harnesses`): the matrix is the honest
account of what the harness *can* check today vs. what is enumerated coverage debt.
"Always-on" = runs green on a bare CI box; "Gated" = skip-with-SUCCESS until the
fixtures + engine land.

| Rung | Always-on coverage (runs today) | Gated coverage (needs fixtures + engine) | Status |
|------|----------------------------------|-------------------------------------------|--------|
| **L0** preprocess (EXACT) | gray-pad-127, `[-1,1]` normalize bounds, 273-token-census constants | full value-exact preprocessed-tensor + image-token-id-stream compare | **partial** — anchors live; tensor compare gated on the Phase-1 preprocess front end (`bd-1gv.2/3`) |
| **L1** per-op (cosine) | cosine/ULP comparator math (synthetic) | per-stage `.npy` vs engine seam, cosine ≥ 0.9999 + ULP | **gated** — fixture+engine pending |
| **L2** per-layer (cosine + ledger) | 12-decoder-layer + 3-vision-seam census; max-abs ledger shape | all 12 `decoder_layer_NN_hidden` + vision seams, per-layer max-abs ledger | **gated** |
| **L3** logits (MEASURED + argmax) | the §2-floor → L3-tolerance derivation (synthetic two-run pair); argmax-stability | `lm_head_logits` within measured budget + argmax-match-where-deterministic | **gated** |
| **L4** token (EXACT prefix) | exact-prefix logic over the reproducible prefix (synthetic) | greedy token stream EXACT over the §2 prefix per doc | **gated** |
| **L5** e2e (exact-where-det + CER budget) | CER metric (Levenshtein) on synthetic strings | `focr ocr --json` vs golden text+bbox; CER/TEDS/Formula-CDM budget | **gated** (model-gated e2e) |
| **differential** per-op (vs bf16) | row-shape contract + EngineIdentity-distinct guard | per-op + e2e diff vs the bf16 oracle through the ULP / L3–L5 tolerances | **gated** |
| **surface** (always-on anchors) | stable exit codes, robot schema self-describe, scrubber, comparator-normalize-first | — | **present** |

The harness emits a coverage rollup implicitly via every rung's terminal
`result` line; a CI consumer counts `result=pass` vs `result=skip_no_model` vs
`result=xfail` to read the matrix. **No rung silently does nothing** — every rung
logs, even when it skips.

---

## 4. DISCREPANCIES discipline — XFAIL, not SKIP

The accepted-divergence rule (`/testing-conformance-harnesses`, PARITY_LADDER §6.2):

- A **SKIP** silently drops a clause from coverage — forbidden for a *known*
  divergence. The only legitimate skip here is **model/fixture absence**
  (`skip_no_model`), which is a *capability* gate, not a *correctness* statement.
- A **known, accepted numeric divergence is `XFAIL`** — the clause stays in the
  matrix as a ledgered `DISC-NNN` in [`../DISCREPANCIES.md`](../DISCREPANCIES.md)
  (reference behavior, our impl, **measured** impact, kill-switch env var,
  resolution, review date). The harness logs it as `result=xfail` with the
  `disc:"DISC-NNN"` reference, so it is visible, not hidden.

The distinction in the log `result` enum (`test_log_schema.json`):
`pass | fail | xfail | skip | skip_no_model`. `xfail` = "we *expect* this to
diverge, here is the ledger entry"; `skip_no_model` = "we *could not run* because
the model/fixtures are absent" — a SUCCESS, but never an XFAIL and never a silent
hole. **L0's gated branch, when fixtures+model are present but the engine seam is
a stub, logs `xfail`** (the divergence is "not yet implemented", a tracked debt),
not `skip` — the engine being incomplete is a known state, not silent coverage
loss.

---

## 5. PROVENANCE — every measured result traces to the pinned model

Fixture provenance is **mandatory** (PARITY_LADDER §8; GOLDEN.md §Provenance). A
golden whose provenance does not resolve to the pinned source is **incomplete** and
may NOT be recorded as a pass. The harness enforces this:

- `FixtureLoader::check_provenance` asserts each loaded golden's `provenance`
  carries `hf_commit == 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5` and
  `pinned_torch == 2.10.0` (the constants are line-backed in `parity_harness.rs`
  from `gen_reference_fixtures.py` / `docs/truth-pack/PINNED_SOURCES.md`). A wrong
  commit/stack is an `error` line, not a pass.
- Every `parity` log line carries `oracle_fixture` (the seam file) and
  `oracle_sha256` (the array hash from the fixture manifest), so a measured result
  is traceable to the exact bytes it ran against.
- The runtime pin is `torch==2.10.0`, `transformers==4.57.1`, `Pillow==12.1.1`;
  `gen_reference_fixtures.py` refuses to generate against any other stack, so a
  committed fixture is *by construction* on the pin.

If `docs/truth-pack/SOURCE_HASHES.md` fails to verify, the model moved: **STOP**,
re-pin, re-confirm (PARITY_LADDER §8). A harness result against an unpinned stack
is not comparable.

---

## 6. The oracle-nondeterminism floor — established FIRST (the keystone)

> *A franken_ocr int8 divergence inside the oracle's own bf16 noise is not a bug.*
> This is the entire reason `establish_floor` runs before any L3/L4 tolerance is set.

The procedure (PARITY_LADDER §2; `bd-re8.2`), realized by
`parity_harness::establish_floor`:

1. The oracle is run **twice** (two thread counts) via
   `gen_reference_fixtures.py … --run-tag a` / `--run-tag b`.
2. `establish_floor(logits_a, logits_b, tokens_a, tokens_b)` measures the
   **per-logit max-abs spread** and the **longest identically-reproduced decoded
   token prefix** between the two runs.
3. The two downstream tolerances are **derived** from that measurement:
   - `OracleFloor::l3_logit_tolerance()` → the L3 budget (the measured spread);
   - `OracleFloor::l4_exact_prefix()` → the prefix L4 asserts bit-exact over.

L3's always-on path proves this derivation pipeline on a synthetic two-run pair, so
the keystone discipline ("derive, never guess; never `0.055`") is *exercised* even
with no real oracle, and the live path consumes the same helper the moment the
two-run fixtures are committed.

---

## 7. Running the harness

```bash
# Always-on coverage (no weights, no fixtures): comparator math + surface anchors
# all run green; every gated rung logs a skip_no_model SUCCESS line.
cargo test --test parity_ladder

# With the CUDA-host oracle fixtures committed under tests/fixtures/native
# (or pointed at by FOCR_FIXTURES_DIR) and the weights resolvable, the gated
# rungs run the live compare:
FOCR_FIXTURES_DIR=/path/to/native FOCR_MODEL_PATH=/path/to/model.focrq \
    cargo test --test parity_ladder -- --nocapture   # --nocapture to see the NDJSON

# Bless a golden after a reviewed change (GOLDEN.md §4 — CI NEVER sets this):
UPDATE_GOLDENS=1 cargo test --test parity_ladder
```

Every line is NDJSON on **stderr** (so it never pollutes captured stdout) and
conforms to `tests/fixtures/test_log_schema.json` — the suite's own
`validate_event` contract test asserts that conformance, so the log contract
cannot drift unnoticed.

---

## 8. Relationship to the rest of the conformance pillar

| Suite | Question | Shares with this harness |
|-------|----------|--------------------------|
| **Differential** (`bd-re8.9`) | "same as the bf16 reference (any input)?" | the ULP comparator + the L3–L5 tolerances; one differential test lives **in** this file |
| **Metamorphic** ([`METAMORPHIC.md`](METAMORPHIC.md), `bd-re8.10`) | "self-consistent under transforms (no oracle)?" | the scrubbers + the determinism discipline MR-4 underwrites |
| **Golden** ([`GOLDEN.md`](GOLDEN.md), `bd-re8.11`) | "did the frozen surface/numeric output change?" | **reuses** the cosine/ULP comparator (does not re-invent it) + the `golden_diff` loop |
| **Gauntlet** ([`../gauntlet/METHODOLOGY.md`](../gauntlet/METHODOLOGY.md)) | "release-eligible at the conformal lower bound?" | the L0–L5 scorecard rows are the conformance pillar's per-rung evidence |

Differential = "same as reference (any input)"; metamorphic = "self-consistent
under transforms (no oracle)"; golden = "no regression vs frozen good output". This
harness is the L0–L5 spine all three hang off, and its comparator is the single
shared kernel.
