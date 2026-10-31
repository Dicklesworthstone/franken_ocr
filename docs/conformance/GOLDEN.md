# GOLDEN.md — the golden-artifact strategy

**Bead:** `bd-re8.11` (VERIFY-golden-suite) — the golden-artifact leg of the
conformance pillar (plan §8.3, §8.6, [`/testing-golden-artifacts`]). Golden
artifacts **freeze known-good outputs and diff them forever**: the cheapest,
highest-signal regression catcher in the project. They cover what differential
and metamorphic testing do not — the **exact CLI / robot surface contract** and
the **per-layer numeric fingerprints** — so a regression is caught the moment it
lands, with the exact before/after visible in the PR diff.

> **The one load-bearing discipline: the RIGHT pattern per artifact type.** A
> wrong pattern is a useless (or actively harmful) test. An *exact* snapshot of a
> timing-laden NDJSON stream **flaps** on every run and gets ignored; a *fuzzy*
> snapshot of `--help` text **hides** a real regression behind a tolerance.
> Pattern-per-artifact (§2) is mandatory, not stylistic.

## Provenance (the reference bar every golden is measured against)

Pinned source @ HF
**`3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`** / GitHub
`7e98affeacba24e95562fbaa234ddb89b856874a`
([`../truth-pack/PINNED_SOURCES.md`](../truth-pack/PINNED_SOURCES.md); SHA-256s in
[`../truth-pack/SOURCE_HASHES.md`](../truth-pack/SOURCE_HASHES.md)). The
end-to-end golden bar — `tests/fixtures/native/<doc>_reference.json` — is produced
by the **PyTorch oracle** (`scripts/gen_reference_fixtures.py`) under the runtime
pin `torch==2.10.0`, `transformers==4.57.1`, `Pillow==12.1.1` on a CUDA host
(OQ-17: the official `infer()` path is GPU-only). Every fixture carries a
**`PROVENANCE.md`** recording model sha256, the HF commit, the stack versions, the
generation config, and the exact command — a golden whose provenance cannot be
resolved to a pinned source is **incomplete** (the same artifact-graph discipline
as [`../DISCREPANCIES.md`](../DISCREPANCIES.md)).

> Fixture-provenance is **mandatory** (plan §8.6). The reference oracle is GPU-only
> and frozen *once*; parity NEVER depends on a live CPU HF run. The per-layer
> activation goldens (`.npy`) are the secondary oracles for the fuzzy/ULP pattern;
> they too are committed with provenance.

---

## 1. The five artifact types and where they live

| # | Artifact | Pattern (§2) | Stored at | Produced by |
|---|---|---|---|---|
| A | `focr ocr --json` structure | **exact** (insta) | `tests/snapshots/ocr_json__*.snap` | the engine, on the golden corpus |
| B | CLI `--help` / subcommand help / `robot schema` | **exact** (insta) | `tests/snapshots/cli_help__*.snap` | clap render + `robot::robot_schema()` |
| C | logits + per-layer activation tensors | **fuzzy** (ULP / epsilon) | `tests/fixtures/native/activations/<doc>/<stage>.npy` | the oracle (`gen_reference_fixtures.py`) |
| D | robot NDJSON event stream | **scrubbed** (timing/run-id) | `tests/snapshots/robot_ndjson__*.snap` | the engine in `--robot` mode |
| E | end-to-end reference output (the BAR) | **canonicalized** exact | `tests/fixtures/native/<doc>_reference.json` | the oracle, frozen once |

`tests/snapshots/` holds the insta `.snap` files (committed). `tests/fixtures/`
holds the oracle-produced reference JSON + activation `.npy` (committed, with
`PROVENANCE.md`). All `*.actual` / `*.snap.new` are **gitignored** (§5).

---

## 2. Pattern-per-artifact (the mandatory discipline)

### 2A. Exact — `focr ocr --json` structure (insta)

The `--json` payload is a **stable contract** an agent parses. Freeze its exact
structure with an [`insta`](https://insta.rs) snapshot. To stay deterministic
across runs/platforms, **canonicalize before snapshotting** (this is *part of the
exact pattern*, not the separate "canonicalized" pattern — the canonicalization
makes the exact comparison meaningful):

- **strip volatile fields**: any `timing_ms`, `elapsed_ms`, `run_id`, `started_at`
  → use insta **redactions** (`insta::assert_json_snapshot!(value, { ".timing_ms"
  => "[ms]", ".run_id" => "[run-id]" })`) so the *shape* is frozen exact while the
  volatile leaf is masked. Redaction (not deletion) keeps the field's *presence*
  under test — losing a field is a contract break the snapshot must still catch.
- **sort bbox arrays** into a canonical order (e.g. lexicographic by
  `(y1, x1, x2, y2)` then label) so a non-deterministic emission order does not
  flap the snapshot. The *content* is frozen; the *order* is canonical.
- **canonical number formatting** for any float in the JSON (fixed precision) so a
  platform's float-to-string does not differ.

The snapshot is `assert_json_snapshot!` (insta's JSON mode, which pretty-prints
canonically). Exact means **byte-for-byte after canonicalization** — a single
changed key, value, or nesting level fails the snapshot and shows the diff in the
PR.

### 2B. Exact — CLI `--help` and `robot schema` (insta)

`--help`, each subcommand's `--help`, and `robot schema` are the **human/agent
surface contract**. Snapshot them **exact** (`insta::assert_snapshot!` on the
rendered string). These must be byte-stable: a wording change, a reordered flag, a
dropped subcommand, or a `ROBOT_SCHEMA_VERSION` bump all surface here and *should*
require a deliberate golden update. `robot schema` in particular is the
machine-readable contract (`src/robot.rs`, `ROBOT_SCHEMA_VERSION`); its exact
snapshot is the frozen schema fixture the robot-contract test (`bd-zc1o`)
validates emitted events against.

> Capture help by invoking the binary (`assert_cmd`) **or** by rendering the clap
> `Command` in-process; pin `term_width` (e.g. `Command::term_width(80)`) so the
> wrap is deterministic across the test host's terminal width — an un-pinned width
> is the classic `--help` snapshot flake.

### 2C. Fuzzy — logits + per-layer activation tensors (ULP / epsilon)

Floating-point tensors **cannot** be frozen exact across architectures (different
SIMD reduction orders produce different-but-correct last-bit results). Freeze them
**fuzzy**, comparing against the oracle `.npy` with the **per-op ULP tolerance
table** (plan §8.5): default **4 ULP for f32 matmul outputs, 2 ULP for elementwise
ops**, and the **measured** int8/int4 quant tolerance for the quantized stages
(derived from the oracle's own bf16 non-determinism floor, §8.2 — *not* a
hand-guessed epsilon, *not* the inherited frankensearch `0.055`). The comparator:

- loads the oracle activation `<stage>.npy` and the engine's same-stage tensor;
- asserts shape-exact, then **per-element** within the ULP/epsilon budget for that
  stage's op class (matmul vs. elementwise vs. quantized);
- reports `max_abs_diff`, `max_ulp`, and the offending index on failure, and logs
  the per-stage cosine (the L1/L2 ladder also wants `cosine ≥ 0.9999` for f32);
- consumes the same ULP table as the L1/L2 differential gates (this golden suite
  **reuses** the comparator from `VERIFY-ladder-l1-l2`, it does not re-invent it).

This is the **fuzzy** pattern: a *tolerance*, not a tautology. The tolerance is
named, justified, and ledgered (a widened tolerance is a `DISC-NNN` entry, never a
silent epsilon bump).

### 2D. Scrubbed — robot NDJSON event stream

The robot NDJSON stream (`run_start`, `stage`, `page`, `run_complete`,
`run_error` — `src/robot.rs`, plan §7.3) carries **timing and identity fields that
change every run** (`elapsed`, `started_at`, `run_id`, per-stage durations,
budgets). An *exact* snapshot of the raw stream would flap on every run. The
**scrubbed** pattern: pipe each event line through a **scrubber** that replaces
volatile fields with stable placeholders **before** snapshotting:

```jsonc
// raw          {"schema_version":1,"event":"stage","name":"vision","seq":2,"elapsed_ms":143}
// scrubbed     {"schema_version":1,"event":"stage","name":"vision","seq":2,"elapsed_ms":"[ms]"}
```

Scrub list (the fields whose *value* is volatile but whose *presence* is contract):
`elapsed_ms`, `*_ms`, `duration*`, `run_id`, `started_at`, `finished_at`,
`timestamp`, any absolute path (→ `[path]`). **Scrub, do not delete** — the field
must remain present so a *dropped* field is still caught. The scrubbed stream is
then snapshotted **exact** with insta (the event *sequence*, kinds, and stable
payload fields are frozen byte-for-byte). This is also the input to the
robot-contract test (`bd-zc1o`): the scrubbed goldens are the frozen-schema
fixtures emitted events are validated against.

### 2E. Canonicalized — cross-platform stability

The 5-target release matrix (linux/darwin × x86-64/arm64, windows-msvc x86-64,
plan §7.6) must produce **one** golden, not five. Canonicalize every textual
artifact (A, B, D, E) for cross-platform determinism:

- **line endings** → `\n` (strip `\r`), so Windows CRLF does not fork the golden;
- **path separators** → `/`, and **absolute paths** → a stable token (`[cwd]`,
  `[model-dir]`), so a host-specific path does not leak into the snapshot;
- **bbox / list ordering** → canonical sort (same rule as 2A), so a
  platform/thread-order difference does not flap;
- **float formatting** → fixed precision;
- **the int8 SIMD-tier string** in `robot backends` is **scrubbed** (it differs
  per host — AMX vs VNNI vs SMMLA vs SDOT vs scalar) unless the test *pins*
  `FOCR_FORCE_ARCH` to make it deterministic.

Canonicalization runs *inside* the comparator for A/B/D/E so the **same golden**
passes on every target. insta `Settings` (filters / redactions) carry the
canonicalization rules so they are applied uniformly.

---

## 3. The reference bar (artifact E) — the oracle is the source of truth

The end-to-end golden `tests/fixtures/native/<doc>_reference.json` is produced by
the **unmodified PyTorch oracle on a CUDA host** and **frozen once** (plan §8.1,
OQ-17). `focr ocr --json` must match it **after canonicalization** (strip timing,
sort bbox). This is the L5 bar (plan §8.2): decoded text exact where the reference
is deterministic, bbox tuples equal after the canonical sort, aggregate CER/TEDS
within the documented budget for the slices where exact-match does not apply.

- The oracle's **own non-determinism floor** is established first (run twice / two
  thread counts, plan §8.2): the reference JSON records the prefix the oracle
  reproduces identically, and "exact match" is defined only over that prefix. A
  franken_ocr divergence *inside* the oracle's bf16 noise is **not** a golden
  failure.
- **Model-gated**: the e2e golden tests **skip-with-SUCCESS** when the 6.67 GB
  weights are absent (CI stays green without them, plan §8.3); when present, they
  prove the native path actually ran (point any fallback at `/nonexistent`). The
  *surface* goldens (B: help/schema) and the *fuzzy* activation goldens (C,
  committed `.npy`) run **without** the weights, so the suite has real
  always-on coverage even on a CI box with no model.

---

## 4. The `UPDATE_GOLDENS` review workflow (a human lands every golden)

A golden is a **promise**: "this output is correct; alert me if it ever changes."
Auto-updating goldens **destroys** that promise — it silently re-blesses whatever
the code now emits, including a regression. So the workflow is deliberately
human-in-the-loop:

```bash
# 1. A change alters an output. The golden test FAILS, writing a *.actual / *.snap.new
#    next to the committed golden and printing the diff.
cargo test --test golden            # -> FAIL, shows the before/after diff

# 2. A HUMAN inspects the diff. Is the change intended and correct?
cargo insta review                  # interactive accept/reject per snapshot
#    or, for the non-insta goldens (activation .npy, reference JSON):
git diff --no-index tests/fixtures/.../golden.json tests/fixtures/.../golden.json.actual

# 3. ONLY a deliberate, reviewed update regenerates the golden:
UPDATE_GOLDENS=1 cargo test --test golden     # rewrites the committed goldens
INSTA_UPDATE=always cargo test                #   (insta's equivalent env)

# 4. The regenerated golden is committed WITH a mandatory `git diff` review in the PR.
git add tests/snapshots tests/fixtures && git diff --cached   # reviewer sees the change
```

**Rules (non-negotiable):**

1. **`UPDATE_GOLDENS=1` is the only way to rewrite a committed golden.** The
   default `cargo test` run **never** writes over a golden — it only writes
   `*.actual` and fails.
2. **A `git diff` review is mandatory** before a regenerated golden lands — the
   reviewer sees exactly what changed and must judge it correct. The diff *is* the
   review artifact.
3. **CI NEVER sets `UPDATE_GOLDENS` / `INSTA_UPDATE`.** CI runs the goldens in
   *compare* mode only; a mismatch is a build failure, not an auto-fix. A test in
   the suite **asserts** the CI environment does not carry the update flag (§6).
4. A golden regenerated by the oracle (E, C) additionally requires a **provenance
   refresh** (re-stamp `PROVENANCE.md` with the new model/stack/command) — a
   golden whose provenance is stale is rejected.

This is the same conscience as the negative-evidence ledger: nothing changes the
"known-good" baseline without a human looking at the measured before/after.

### 4a. The token-id golden gate (U2-T3)

`tests/uocr2_goldens.rs` pins the engine's real outputs at the **token-id** level,
per model lane: the generated ids, the final markdown, and (for one streamed
case) the robot `page` events. One drifted token can survive detokenize and
postprocess unchanged, so markdown goldens alone cannot catch it. The v1 lane
(`tests/fixtures/uocr2_v1_goldens/`) is the safety net under every
Unlimited-OCR-2 change to shared code. The v2 lane
(`tests/fixtures/uocr2_v2_goldens/`, U2-C4) pins the int8 artifact U2-C4
certified, so every later change on the v2 path keeps its outputs or comes
with a reviewed recapture.

```bash
scripts/token_golden_gate.sh                    # fast subset; scripts/check.sh runs it
FOCR_E2E_LONG=1 scripts/token_golden_gate.sh    # full set: required before closing any bead
                                                # that touches the decoder, loader, tokenizer,
                                                # sampler, PDF walk or CLI output, and at
                                                # every milestone exit
UPDATE_GOLDENS=1 FOCR_E2E_LONG=1 scripts/token_golden_gate.sh   # reviewed recapture only
```

* **Model-gated.** The script builds the optimized `release-perf` profile only
  when an `unlimited-ocr*.focrq` is on the machine; the test then compares only
  on the artifact SHA-256 that `CAPTURE.json` pins (a different artifact logs
  `skip_no_model`). Inputs come from `FOCR_CORPUS_DIR` (default
  `~/.cache/franken_ocr/corpus/pages`), checked against the captured SHA-256s.
* **Fast subset:** v1: `page_0009`, `page_0014`, the 2-page `p9_p14` document, and
  `p9_p14` streamed. v2: `page_0009`, `page_0014`, the 3-page S2 document, and S2
  streamed. **Long** (v1 only): `page_0590`, `p10`, `p20`.
* **One lane at a time on recapture.** A recapture takes the first artifact that
  exists, `FOCR_MODEL_PATH` first, so it names the lane's artifact: for v2, the
  `.focrq` in a directory that also holds v2's `tokenizer.json`.
* **Robot events compare on a canonical projection:** each event on the golden's
  own fields, minus the shared volatile list (`parity_harness::is_scrub_key` —
  `SCRUB_KEYS` plus any `_ms`/`_us`/`_seconds`/`_utc` key). A field added later is
  ignored (robot changes are additive); a changed, dropped or extra event fails.
* **A mismatch names the first differing token index and both ids,** and writes
  `*.actual` beside the golden. Determinism is part of the contract: the gate must
  pass at any `FOCR_THREADS`.
* **Regeneration** follows §4 exactly: `UPDATE_GOLDENS=1`, a reviewed `git diff`,
  and a commit message that says why the tokens moved.

---

## 5. `*.actual` and the gitignore contract

When a golden test fails it writes the *observed* output next to the committed
golden so a human can diff it. These observed artifacts are **transient and must
never be committed**:

- insta writes `*.snap.new` (pending snapshots);
- the custom comparators (C/E) write `*.actual` (e.g.
  `<doc>_reference.json.actual`, `<stage>.npy.actual`).

Both patterns are **gitignored**. The repo `.gitignore` must carry:

```gitignore
# Golden-artifact transient/observed outputs — NEVER commit (docs/conformance/GOLDEN.md §5)
*.actual
*.snap.new
```

> **Action item (outside this doc's write scope).** The repo `.gitignore` does
> **not** yet contain `*.actual` / `*.snap.new`. Adding these two lines is a
> prerequisite for the golden suite landing (`bd-re8.11`) and belongs to the bead
> that wires the suite into `tests/`. This document is the spec that mandates it;
> the `.gitignore` edit is a one-line follow-up the implementing bead applies.
> A suite test **asserts** these patterns are ignored (§6) so a missing rule is
> caught, not assumed.

(The committed `.snap` files are **not** ignored — they are the golden. Only the
`.new` / `.actual` *observed* outputs are.)

---

## 6. Test inventory (what the golden suite must contain)

- **Exact (insta)**: `ocr_json__<doc>` per corpus doc (2A); `cli_help__root`,
  `cli_help__ocr`, `cli_help__convert`, `cli_help__robot`, `robot_schema` (2B).
- **Fuzzy (ULP)**: `activation__<doc>__<stage>` per dumped seam (post-SAM,
  post-CLIP, post-projector, per-decoder-layer hidden state, pre-lm_head logits),
  consuming the shared ULP comparator (2C).
- **Scrubbed**: `robot_ndjson__<doc>` over the streamed event sequence (2D).
- **Canonicalized**: the canonicalization is *inside* the A/B/D/E comparators
  (2E); a unit test exercises the canonicalizer (line-ending, path, bbox-sort,
  float-format) on hand-built inputs.
- **The guard tests (the suite tests its own discipline):**
  - a test asserting **`*.actual` and `*.snap.new` are gitignored** (read
    `.gitignore`, assert the patterns are present — §5);
  - a test asserting **CI does not set `UPDATE_GOLDENS` / `INSTA_UPDATE`** (read
    the env in the CI marker; fail if an update flag is live — §4 rule 3);
  - a test asserting every committed reference fixture has a resolvable
    `PROVENANCE.md` (§Provenance).
- **Structured logging on mismatch**: each comparator emits an NDJSON line
  `{"suite":"golden","artifact":"<id>","pattern":"exact|fuzzy|scrubbed",
  "result":"pass|fail","max_ulp":N,"detail":"…"}` so a CI failure is
  self-describing.

## 7. Relationship to determinism and the other suites

The golden suite is **downstream of the engine's determinism guarantee**: the
exact (A/B) and scrubbed (D) snapshots are only non-flaky because
[`METAMORPHIC.md`](METAMORPHIC.md) MR-4 proves `focr(x)` is byte-identical across
runs and thread counts (greedy decode; int8 GEMM bit-identical across SIMD paths;
deterministic reduction order). MR-4 is the upstream guarantee; the golden
snapshots are the downstream regression catch. The fuzzy (C) goldens share the
ULP comparator with the L1/L2 differential gates; the reference bar (E) is the L5
gate's input.

| Suite | Question | This suite's tie-in |
|---|---|---|
| **Metamorphic** ([`METAMORPHIC.md`](METAMORPHIC.md), `bd-re8.10`) | "self-consistent under transforms?" | MR-4 makes A/B/D snapshots non-flaky |
| **Differential** (`bd-re8.9`) | "same as the oracle, per-op?" | shares the C ULP comparator |
| **Golden** (this, `bd-re8.11`) | "did the frozen surface/numeric output change?" | **catalogued here** |
| **Robot contract** (`bd-zc1o`) | "do emitted events match the frozen schema?" | consumes the D scrubbed goldens + the B `robot schema` snapshot |

The golden leg feeds the gauntlet's **conformance pillar** (plan §8.5): a golden
mismatch is a hard parity-cell failure; the release scorecard cannot ship with a
red golden cell or an unreviewed golden update (plan §8.4). Every golden is a
`ConformanceTest` (requirement level `MUST` for A/B/D/E, `MUST` for the
`*.actual`-gitignored and no-CI-auto-update guards), contributing to the ≥0.95
MUST-coverage accounting (plan §8.6).
