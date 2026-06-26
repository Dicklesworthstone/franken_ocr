# LOGGING_AND_E2E.md — the structured test-logging contract, the model-gated e2e runner, the deadlock watchdog, and the benchmark guardrail

> **Beads:** `bd-n68o` (TEST-log-conventions), `bd-29wv` (TEST-model-gated-runner),
> `bd-223.8` / `bd-10sb` (the `many_pages_without_deadlock` watchdog scaffold +
> E-TEST epic), and the benchmark-guardrail gate (plan §9.6, epic `bd-10sb` →
> `TEST-bench-guardrail`). This is the **DESIGN** half: the frozen schemas and the
> harness strategy every test bead (`P1-*`/`P2-*`/`P3-*`/`P4-*`, E-VERIFY) plugs
> into. The Rust emitter/validator/runner modules (`tests/common/`) are delivered
> by the implementing beads against the contracts pinned here.
>
> **Status.** LIVING DOCUMENT. The schemas below are **frozen contracts**:
> `TEST_LOG_SCHEMA_VERSION` is bumped (never silently changed) on any breaking
> change, exactly like `ROBOT_SCHEMA_VERSION` in [`src/robot.rs`](../../src/robot.rs).
>
> **Why this exists.** AGENTS.md and the master plan are explicit: the user wants
> *comprehensive unit tests AND end-to-end test scripts with great detailed
> logging*. The whole project lives or dies on doctrine **G1 > G2 — correctness
> outranks speed, always** (AGENTS.md doctrine #1): a faster kernel that drifts the
> OCR output is reverted and memorialized in
> [`docs/NEGATIVE_EVIDENCE.md`](../NEGATIVE_EVIDENCE.md). That doctrine is only
> *enforceable* if the test machinery is first-class — a failing test must be
> diagnosable **from its logs alone**, and the gauntlet's artifact graph (plan
> §8.4/§8.5) must be able to ingest test evidence as machine-readable records.

---

## Table of contents

1. [Scope and the four contracts](#1-scope)
2. [Principles (the load-bearing constraints)](#2-principles)
3. [The structured test-logging contract — NDJSON `TestLog`](#3-the-testlog-contract)
   - 3.1 [Stream rules](#31-stream-rules)
   - 3.2 [The frozen field schema (per `event`)](#32-the-frozen-field-schema)
   - 3.3 [Closed enums](#33-closed-enums)
   - 3.4 [The per-case record (worked example)](#34-the-per-case-record)
   - 3.5 [Buffering, flush-on-panic, and hot-loop cost](#35-buffering-flush-on-panic-hot-loop-cost)
   - 3.6 [The schema validator](#36-the-schema-validator)
4. [The model-gated e2e runner strategy](#4-the-model-gated-e2e-runner)
   - 4.1 [Skip-with-SUCCESS when the weights are absent](#41-skip-with-success)
   - 4.2 [Prove the native path ran (`/nonexistent` fallbacks + side-effect assertions)](#42-prove-the-native-path-ran)
   - 4.3 [Absent vs. present-but-corrupt — skip vs. fail](#43-absent-vs-corrupt)
   - 4.4 [The `E2eRun` helper and the e2e CLI script](#44-the-e2erun-helper)
   - 4.5 [The error-path e2e (the §7.4 exit-code contract)](#45-the-error-path-e2e)
5. [The `many_pages_without_deadlock` watchdog](#5-the-many-pages-watchdog)
6. [The benchmark-guardrail gate](#6-the-benchmark-guardrail-gate)
7. [How it all wires into CI](#7-ci-wiring)
8. [Cross-references](#8-cross-references)

---

## 1. Scope

This document pins **four** test-infrastructure contracts that the rest of the
test suite consumes. They share one substrate — line-oriented NDJSON — so that a
single validator and a single artifact-capture step (`TEST_LOG_DIR`) cover all of
them.

| # | Contract | Owning bead | Rust home (delivered later) |
|---|----------|-------------|------------------------------|
| **C1** | The structured **`TestLog` NDJSON** schema every stage emits | `bd-n68o` | `tests/common/test_log.rs` + `tests/common/log_schema.rs` |
| **C2** | The **model-gated e2e runner** (skip-with-SUCCESS; prove native ran) | `bd-29wv` | `tests/common/model_gate.rs` |
| **C3** | The **`many_pages_without_deadlock`** watchdog | `bd-223.8` | `tests/concurrency_watchdog.rs` |
| **C4** | The **benchmark-guardrail** regression gate | `bd-10sb`→`TEST-bench-guardrail` | `benches/gauntlet.rs` + `scripts/bench_guardrail.py` |

**Out of scope (owned elsewhere, referenced here):**

- The **parity-ladder semantics** (L0–L5 tolerances, ULP tables, the conformal
  lower-bound ratchet, e-processes) are owned by **E-VERIFY** (`bd-re8`). C1 only
  defines *how* a parity result is *logged*; the *thresholds* come from E-VERIFY's
  `tolerances.toml`.
- The **oracle nondeterminism floor** itself is owned by `bd-re8.2` and
  [`scripts/oracle_nondeterminism_floor.py`](../../scripts/oracle_nondeterminism_floor.py).
  C1 carries the resulting `nondeterminism_envelope` on every parity line; it does
  not compute it.
- **Golden-fixture loading / canonicalization / `PROVENANCE.md`** is owned by
  `TEST-fixture-harness`; the e2e runner (C2) consumes it for the frozen-reference
  compare.

---

## 2. Principles

These bind every contract below.

| # | Principle | Where it bites |
|---|-----------|----------------|
| **TL1** | **Logging is a feature, not decoration.** | Every stage emits a machine-parseable line; a failing test is diagnosable from the captured stream alone. No bare `println!`/`eprintln!` for test telemetry. |
| **TL2** | **Data-only on stdout, diagnostics on stderr, exit 0 = success.** | Matches the project's robot/CLI conventions (AGENTS.md "Agent Ergonomics", the `cass`/`bv` style). NDJSON is never interleaved with human decoration on the same stream. |
| **TL3** | **Versioned, closed schemas.** | `TEST_LOG_SCHEMA_VERSION` (a Rust `const`) mirrors `ROBOT_SCHEMA_VERSION`. `event`/`stage`/`result`/`dtype`/`simd_tier`/`gate`/`metric` are **closed enums** — a typo'd `stage` is a *bug the validator rejects*, not a silently-accepted free string. |
| **TL4** | **Skip-with-SUCCESS, never skip-silently-fail.** | A model-gated test with weights absent returns `Ok` and logs `result:"skip_no_model"` (CI green, but the skip is *visible*). A model **present but broken** is a hard FAIL — never a skip. |
| **TL5** | **Prove the native path ran.** | When weights are present, every non-native fallback is pointed at `/nonexistent` so a silent degradation to a stub *explodes*; the test additionally asserts real native side-effects (parsed `.focrq` header values, the OQ-18 token census, per-decoder-layer stage lines). |
| **TL6** | **No mocks where a real path exists.** | The e2e drives the real `focr` binary (or `OcrEngine::recognize`) against real golden images and the real frozen reference JSON (`/testing-perfect-e2e-integration-tests-with-logging-and-no-mocks`). |
| **TL7** | **A divergence inside the oracle's own noise is not a bug.** | Parity lines carry the `nondeterminism_envelope`; L4 "exact" is asserted only over the prefix the bf16 oracle reproduces identically (§8.2). The log makes the distinction explicit so a reviewer never mistakes oracle noise for drift. |
| **TL8** | **The `avx2` tier is the one non-bit-identical path.** | AVX2 `vpmaddubsw` saturates at i16 (plan §6.6 ⚠). A parity line with `simd_tier:"avx2"` compared against an i32-exact reference must carry `avx2_exception` pointing at its `DISC-NNN` — so the validator/reader never treats the documented divergence as a bug. |
| **TL9** | **Flush-on-panic / flush-on-drop.** | Logs may buffer (the zero-hot-loop-cost rule), so a *crashing* test must still leave its NDJSON up to the failure point on disk — otherwise the most important failure produces no log. |

---

## 3. The `TestLog` contract

### 3.1 Stream rules

- **One JSON object per line.** No pretty-printing, no trailing commas, UTF-8,
  `\n`-terminated. A consumer reads the stream with a line splitter and
  `serde_json::from_str` per line.
- **Two sinks.** During `cargo test`, lines are written to:
  1. `stderr` of the test process (so a developer watching the run sees them), and
  2. a file under `$TEST_LOG_DIR/<test_binary>.<pid>.ndjson` when `TEST_LOG_DIR`
     is set (CI sets it so the validator + artifact capture can run over the whole
     captured stream — plan §8.4).
  The *data surface* compared by golden tests (e2e `--json` output, robot NDJSON)
  goes to **stdout**; `TestLog` telemetry never pollutes stdout (TL2).
- **Scrubbable time.** `ts` is **monotonic-relative microseconds since the test
  process started**, never wall-clock — so it is reproducible-modulo-timing and
  the fixture-harness canonicalizer strips it (`ts`, `elapsed_us`, any duration)
  before a golden compare.
- **Correlation.** Every line of one pipeline invocation shares a `run_seq` (a
  monotonic per-test invocation counter) so the lines of a single forward are
  joinable even when a test runs the pipeline more than once (determinism = run
  twice; the watchdog = N pages). An optional `trace_id` carries the same role
  across processes.

### 3.2 The frozen field schema (per `event`)

`TEST_LOG_SCHEMA_VERSION = 1`.

**Common fields — present on EVERY line:**

| Field | Type | Meaning |
|-------|------|---------|
| `schema_version` | `u32` | `TEST_LOG_SCHEMA_VERSION`; a mismatch is a validator error. |
| `ts` | `u64` | Monotonic-relative microseconds since process start (scrubbable). |
| `run_seq` | `u64` | Per-test invocation counter; correlates all lines of one forward. |
| `test` | `string` | The test function name (e.g. `native_engine_e2e::gundam_golden`). |
| `case` | `string` | Sub-case id (e.g. the document stem, the tier name, the page index). |
| `event` | enum | One of `setup \| stage \| parity \| assert \| skip \| result \| error`. |
| `result` | enum | `pass \| fail \| xfail \| skip \| skip_no_model` (required on `result`/`skip`; on other events it is the running disposition). |
| `trace_id` | `string?` | Optional cross-process correlation id. |

**`event:"stage"` — the load-bearing telemetry line (the "great logging" mandate):**

| Field | Type | Meaning |
|-------|------|---------|
| `stage` | enum | The pipeline stage (closed enum, §3.3). |
| `layer_idx` | `u32?` | Decoder-layer index on per-layer stages (`rmsnorm`/`rope`/`rswa_attn`/`moe_*`/`dense_mlp`), so a per-layer L2 divergence names its layer. |
| `inputs` | `object` | Named input descriptors (e.g. `{"image":"doc01.png","prompt":"<image>document parsing."}`). |
| `shapes` | `array` | Tensor shapes touched, e.g. `[[256,768]]`, `[[273,1280]]`, `[[1,129280]]`. |
| `dtype` | enum | `f32 \| bf16 \| i8 \| i4 \| u8` (the activation/weight dtype at this stage). |
| `elapsed_us` | `u64` | Wall time for this stage (scrubbable). |
| `simd_tier` | enum | The dispatched ISA tier (closed enum, §3.3). |
| `seed` | `u64?` | RNG seed where any randomness is involved (kept 0 under greedy). |
| `avx2_exception` | `string?` | A `DISC-NNN` ref; **required** when `simd_tier:"avx2"` and the line participates in a bit-identity comparison (TL8). |

**`event:"parity"` — a single ladder comparison vs the oracle:**

| Field | Type | Meaning |
|-------|------|---------|
| `gate` | enum | `L0 \| L1 \| L2 \| L3 \| L4 \| L5` (or `SURF` for the contract test). |
| `metric` | enum | `cosine \| max_abs_diff \| cer \| teds \| formula_cdm \| argmax_match \| token_exact`. |
| `value` | `number` | The measured metric value. |
| `tolerance` | `number` | The threshold from E-VERIFY's `tolerances.toml` (NOT a hand-guessed epsilon). |
| `oracle_fixture` | `string` | The frozen reference path (`tests/fixtures/native/<doc>_reference.json` or `<stage>.npy`). |
| `oracle_sha256` | `string` | SHA-256 of the oracle fixture (provenance). |
| `nondeterminism_envelope` | `object` | The measured oracle variance for this comparison surface (`{divergence_rate, first_divergence_pos, logit_spread}`) from `oracle_nondeterminism_envelope.json` (§8.2). |
| `pass` | `bool` | Whether `value` is within `tolerance` (for `L4`, *over the reproducible prefix only*). |

**`event:"result"` / `event:"skip"` — the per-test (or per-case) verdict:**

| Field | Type | Meaning |
|-------|------|---------|
| `result` | enum | The final disposition (see common fields). |
| `searched_dirs` | `array?` | On `skip_no_model`: every dir the model resolver searched (so a developer who *expected* the model to be found sees why it was not — §7.5). |
| `native_path_ran` | `bool?` | On a model-present e2e: `true` proves the real engine executed (TL5). |
| `fallback_target` | `string?` | Must be `"/nonexistent"` on a native-path proof (so a silent fallback would have errored). |

**`event:"error"` (or `result:"fail"`) — a first-class diagnostic:**

When `result:"fail"`, a `diag` object is **required**:

| `diag` field | Type | Meaning |
|--------------|------|---------|
| `error_kind` | `string` | A short machine token (e.g. `shape_mismatch`, `format_version`, `oracle_diff`). |
| `focr_exit_code` | `i32` | The §7.4 exit code the equivalent `FocrError` maps to (so the stream carries the public contract). |
| `message` | `string` | A human-readable one-liner (stderr-grade detail). |

### 3.3 Closed enums

The validator rejects any value not in these sets (a typo is a real bug — TL3).

```text
event       = setup | stage | parity | assert | skip | result | error
result      = pass | fail | xfail | skip | skip_no_model
dtype       = f32 | bf16 | i8 | i4 | u8
gate        = L0 | L1 | L2 | L3 | L4 | L5 | SURF
metric      = cosine | max_abs_diff | cer | teds | formula_cdm | argmax_match | token_exact
simd_tier   = smmla | sdot | vnni512 | vnni256 | amx | avx2 | scalar
stage       = decode_image | preprocess | tokenize
            | vision_sam | vision_clip | vision_bridge | connector
            | prefill | embed | rmsnorm | rope | rswa_attn
            | moe_router | moe_expert | dense_mlp | lm_head | kv_cache
            | decode | postprocess | emit
```

The `stage` enum deliberately covers the **decoder internals** (`embed`,
`rmsnorm`, `rope`, `rswa_attn`, `moe_router`, `moe_expert`, `dense_mlp`,
`lm_head`, `kv_cache`) so "great detailed logging at EVERY stage" is real and not
just coarse phases. These names are the §4.2 / §6 module boundaries in
[`PROPOSED_ARCHITECTURE.md`](../PROPOSED_ARCHITECTURE.md) §6.

### 3.4 The per-case record (worked example)

A single Gundam golden run produces one correlated record like the following
(`run_seq:7`, abridged to the load-bearing stages; `ts`/`elapsed_us` scrubbed to
`…` here for illustration — they are concrete `u64`s on the wire):

```json
{"schema_version":1,"ts":0,"run_seq":7,"test":"native_engine_e2e::gundam_golden","case":"doc01","event":"setup","result":"pass","inputs":{"mode":"gundam","model":"/home/u/.cache/franken_ocr/models/unlimited-ocr"}}
{"schema_version":1,"ts":120,"run_seq":7,"test":"native_engine_e2e::gundam_golden","case":"doc01","event":"stage","stage":"preprocess","result":"pass","inputs":{"image":"doc01.png"},"shapes":[[1024,1024,3]],"dtype":"f32","elapsed_us":3100,"simd_tier":"scalar","seed":0}
{"schema_version":1,"ts":260,"run_seq":7,"test":"native_engine_e2e::gundam_golden","case":"doc01","event":"stage","stage":"vision_sam","result":"pass","shapes":[[4096,768],[1024,256]],"dtype":"bf16","elapsed_us":41200,"simd_tier":"smmla","seed":0}
{"schema_version":1,"ts":520,"run_seq":7,"test":"native_engine_e2e::gundam_golden","case":"doc01","event":"stage","stage":"connector","result":"pass","shapes":[[273,1280]],"dtype":"f32","elapsed_us":900,"simd_tier":"scalar","seed":0,"inputs":{"image_feature_tokens":256,"image_newline":16,"view_seperator":1,"placeholder_total":273}}
{"schema_version":1,"ts":700,"run_seq":7,"test":"native_engine_e2e::gundam_golden","case":"doc01","event":"stage","stage":"prefill","result":"pass","shapes":[[2,10,128]],"dtype":"bf16","elapsed_us":58000,"simd_tier":"smmla","seed":0,"inputs":{"reference_block_m":511,"ring_window":128}}
{"schema_version":1,"ts":900,"run_seq":7,"test":"native_engine_e2e::gundam_golden","case":"doc01","event":"stage","stage":"rswa_attn","layer_idx":0,"result":"pass","shapes":[[1,10,128]],"dtype":"bf16","elapsed_us":420,"simd_tier":"smmla","seed":0,"inputs":{"m":639,"slot":"ring"}}
{"schema_version":1,"ts":1300,"run_seq":7,"test":"native_engine_e2e::gundam_golden","case":"doc01","event":"stage","stage":"lm_head","result":"pass","shapes":[[1,129280]],"dtype":"i8","elapsed_us":2600,"simd_tier":"vnni512","seed":0,"inputs":{"greedy_argmax_only":true,"no_repeat_ngram_size":35,"ngram_window":128}}
{"schema_version":1,"ts":2200,"run_seq":7,"test":"native_engine_e2e::gundam_golden","case":"doc01","event":"parity","gate":"L4","metric":"token_exact","value":1.0,"tolerance":1.0,"oracle_fixture":"tests/fixtures/native/doc01_reference.json","oracle_sha256":"<sha>","nondeterminism_envelope":{"divergence_rate":0.0,"first_divergence_pos":214,"logit_spread":0.031},"pass":true}
{"schema_version":1,"ts":2400,"run_seq":7,"test":"native_engine_e2e::gundam_golden","case":"doc01","event":"result","result":"pass","native_path_ran":true,"fallback_target":"/nonexistent"}
```

Read top-to-bottom this is the entire forward, stage by stage, with the
load-bearing facts visible: the **273-token** OQ-18 census at the connector, the
R-SWA **reference-block length `m`** and **ring window 128** at prefill, the
per-layer R-SWA line keyed by `layer_idx`, the `lm_head` argmax-only-under-greedy
flag, and an **L4 token-exact** parity line that *carries the nondeterminism
envelope* so the `first_divergence_pos:214` boundary is auditable. A failing run
localizes instantly to the diverging stage.

### 3.5 Buffering, flush-on-panic, hot-loop cost

- **Emit only at stage boundaries**, never per inner-loop iteration. The decode
  hot path must not allocate per token to log — buffer the line, flush at the
  `decode`/`lm_head` stage boundary (TL9, plan note "logs must be cheap to emit").
- The emitter holds a `BufWriter` and registers a **flush-on-drop** (`impl Drop`)
  **and** a panic hook so a `panic!`/failed `assert!` still leaves the NDJSON up to
  the failure point on disk. This is non-negotiable: the most important failure
  must not be the one that produces no log.

### 3.6 The schema validator

`tests/common/log_schema.rs` (delivered by `bd-n68o`) parses each captured line
and enforces:

1. `schema_version == TEST_LOG_SCHEMA_VERSION` (else error).
2. The common fields are present and well-typed.
3. The per-`event` required fields are present (e.g. `stage`/`shapes`/`dtype`/
   `simd_tier` on `stage`; `gate`/`metric`/`value`/`tolerance` on `parity`; the
   `diag` object on `result:"fail"`; `searched_dirs` on `skip_no_model`).
4. Every enum-valued field is in its closed set (§3.3) — an unknown
   `event`/`stage`/`result`/`dtype`/`simd_tier`/`gate`/`metric` is **rejected**.
5. The TL8 rule: a `parity` line whose nearest `stage` line has `simd_tier:"avx2"`
   and that asserts a bit-identity must carry `avx2_exception`.

**CI ordering (plan, `bd-4yks`):** the `TEST_LOG_DIR` NDJSON is captured *during*
`cargo test`, then the validator runs over the **whole captured stream** as its
own step and fails CI on any malformed line — the ordering matters so the
validator sees a complete stream. A self-test feeds the validator a deliberately
malformed line (missing required field / unknown enum / mismatched
`schema_version`) and asserts rejection.

---

## 4. The model-gated e2e runner

> **Bead `bd-29wv`** — `tests/common/model_gate.rs`. The Unlimited-OCR checkpoint
> is a **single 6.67 GB bf16 shard** (`model-00001-of-000001.safetensors`,
> `total_size = 6 672 212 480` bytes, 2710 tensors — CENSUS quick-reference). It
> **cannot live in CI**. So e2e tests are **model-gated**: green without weights,
> *meaningful* and *loud* with them.

### 4.1 Skip-with-SUCCESS

```text
fn model_gate() -> Option<ResolvedModel>
```

1. **Resolve** the model via `resolve_model(spec)` over the documented search path
   (`$FOCR_MODEL_DIR`, `~/.cache/franken_ocr/models`, …) and **header-sniff** it —
   the `.focrq` `FOCRQ\0` magic, or the safetensors header — a *cheap* check, **no
   tensor load** (we do not read 6.67 GB just to decide whether to skip).
2. **Absent → skip-with-SUCCESS.** The test returns `Ok(())` and emits one line
   `{"event":"skip","result":"skip_no_model","searched_dirs":[…]}`. CI is green,
   but the skip is **visible** in the log and never silently passes a missing
   assertion (TL4). The harness convention logs this as `SKIPPED-NO-MODEL`.

### 4.2 Prove the native path ran

This is the **critical guard** (TL5, AGENTS.md): a green test that secretly took a
stub/fallback path is a *false* green. Two independent mechanisms, both required:

1. **Point every fallback at `/nonexistent`.** Before running, the runner sets a
   clean, explicit env where every non-native/alternate code path resolves to
   `/nonexistent` (any reference-cmd fallback, any quant-selection fallback). If
   the test accidentally exercised a fallback instead of the real native engine,
   it **errors loudly** rather than passing. The runner does *not* inherit the
   developer's shell `FOCR_FORCE_ARCH` / `FOCR_QUANT` overrides into the proof
   (it sets a deterministic env).
2. **Assert real native side-effects**, not a self-reported flag (a stub could
   fake a boolean). Prefer effects only the real engine produces:
   - the parsed `.focrq` header values match the **WeightsManifest census** (the
     2710-tensor / `*_proj.weight = 2244` accounting from CENSUS);
   - the connector emitted the **OQ-18 token census** (`273` placeholder tokens at
     base 1024: `256` image-feature + `16` image_newline + `1` view_seperator);
   - the decode loop emitted real **per-decoder-layer** `stage` lines (`layer_idx`
     0..11) and an `lm_head` line with the 129280-vocab last dim;
   - the dispatched `simd_tier` was actually set on the `stage` lines.

   The `result` line then carries `native_path_ran:true` and
   `fallback_target:"/nonexistent"`. **A test that "passes" without ever entering
   `native_engine/` is a FAILURE**, proven by an injected fault (force the native
   engine to error → the test must go RED, not skip).

### 4.3 Absent vs. corrupt — skip vs. fail

The gate must distinguish two states that look superficially similar:

| Model state | Disposition | Why |
|-------------|-------------|-----|
| **Absent** (no resolvable model) | `skip_no_model`, **green** | CI without the 6.67 GB shard must be green. |
| **Present but corrupt / wrong `format_version`** | **FAIL** with the §7.4 mapping (`exit 7` format/version, `exit 3` not-resolvable) | A corrupt model is a real failure, *never* a skip — silently skipping a corrupt model hides a real defect. |

### 4.4 The `E2eRun` helper

```text
fn run_e2e(image: &Path, args: &E2eArgs) -> E2eRun
```

Invokes the **real** `focr` binary (or the library `OcrEngine::recognize`),
captures the stdout `--json` / robot NDJSON **and** the structured test-log
stream, and returns the parsed output plus the proof flags
(`native_path_ran`, `fallback_target`). It delegates to the **determinism gate**
(`TEST-determinism-gate`): under `temperature=0` greedy it runs twice and asserts
**byte-identical** output (determinism only holds under greedy — §7.3).

The standalone CLI script [`scripts/run_e2e.sh`](../../scripts/run_e2e.sh)
(delivered by `bd-34ic`) is the no-mocks integration test: for each golden doc it
runs `focr ocr tests/fixtures/golden_corpus/<doc>.png --json`, pipes the output to
a comparator that diffs against `tests/fixtures/native/<doc>_reference.json` **after
canonicalization** (scrub `timing`/`run_id`/`duration_ms`, sort bboxes, normalize
line endings), emits an NDJSON per-doc summary, and exits `0` on all-pass /
non-zero on any diff — and **skips-green (exit 0, logged)** when no model is
resolvable. It covers **both modes** (Base 1024×1024 `crop_mode=false`, and Gundam
`base_size=1024`/`image_size=640`/`crop_mode=true`) and a **10-page multi-page**
golden.

> ⚠️ **Multi-page is cross-page DEPENDENT (OQ-13).** The multi-page reference is a
> single `infer_multi` oracle run, **not** a concatenation of single-page parses:
> in one 32K pass, all pages' visual+prompt prefixes form one frozen reference
> block, so page *N* attends to pages `1..N-1`. The e2e compares against that
> single oracle output and **must not** assert `multi-page == Σ single-page`. The
> defensible metamorphic property is the opposite (page order may change later-page
> output) and is owned by E-VERIFY (`bd-re8.10`).

> ⚠️ **L4 "exact" is bounded by the nondeterminism floor.** Text is compared
> byte-exactly only up to the `first_divergence_pos` boundary carried in
> `oracle_nondeterminism_envelope.json`; beyond it the bf16 oracle is
> non-deterministic and the comparison falls back to **CER-within-budget** (TL7).

### 4.5 The error-path e2e

Several of the §7.4 exit codes fire **before** the forward, so they need **no
weights** — a rare chance for a *meaningful* e2e in the always-on no-weights lane
(plan `bd-34ic` REVIEW-2). The e2e drives the real binary and asserts the process
exit code **and** the `robot run_error` payload code **and** `FocrError::exit_code()`
all agree (one source of truth — [`src/error.rs`](../../src/error.rs)):

| Error case | Exit code | Needs weights? |
|------------|-----------|----------------|
| Missing / unresolvable model (actionable searched-dirs message) | **3** | no |
| Corrupt / truncated input image | **4** | no |
| Corrupt or wrong-`format_version` `.focrq` | **7** | no (header sniff) |
| Stage budget exceeded (`FOCR_STAGE_BUDGET_*_MS` set tiny) | **5** | yes |
| SIGINT / Ctrl-C mid-run (finalizer cleanup runs) | **6** | yes |

A piping + help smoke also asserts `focr ocr <golden> --json | <parse>` yields one
JSON object (data-only on stdout, diagnostics on stderr, exit 0) and that
`focr --help` / `focr --version` / `focr robot schema` emit stable, parseable
output.

---

## 5. The `many_pages_without_deadlock` watchdog

> **Beads `bd-223.8` / `bd-10sb`** — `tests/concurrency_watchdog.rs`. This is the
> durable architectural check that the frankensearch deadlock saga converged on
> after 5 commits (AGENTS.md doctrine #5, plan §3.3/§6.5). **A deadlock manifests
> as a hang, not a panic — so the hang *is* the signal**, caught by a hard
> wall-clock timeout.

**The invariant under test (P6, the concurrency discipline):**

- **Exactly one `OcrModel`** behind an `Arc`/`Weak` cache.
- A **sequential** outer page/document loop — "streaming per-page" streams the
  *output* of a sequentially-processed page, never concurrent forwards. **One live
  forward at a time.**
- Each forward fans out across all cores via the **kernel's own rayon pool**
  (pinned to physical cores), **NOT** asupersync tasks.
- **NEVER** nest rayon under a held lock; **NEVER** nest a second asupersync
  runtime inside a task.

**The test shape:**

1. Drive the engine over **pages ≫ pool** (dozens of placeholder/empty pages
   through the sequential outer loop). The page count is chosen to far exceed the
   rayon pool width and the asupersync worker count so any oversubscription /
   nested-pool / lock-ordering regression has room to deadlock.
2. Run under a **hard wall-clock timeout**. The test spawns the work on a thread
   and `recv_timeout`s on a completion channel; if the budget elapses without
   completion, the watchdog **fails** (a deadlock has no other signal). In CI the
   job *also* carries a hard per-job timeout so a true hang cannot run forever.
3. **Phase-0 reality:** the watchdog runs against the *current* (possibly
   empty-bodied) sequential pipeline, so it **passes now** and stays a tripwire as
   the real forward lands — it does not wait for kernels to exist.

**Proven once, by construction:** an injected regression (a `par_iter` *under a
held lock*, or a nested asupersync runtime inside the per-page task) makes the
watchdog **time out**. That fault-injection self-test is `#[ignore]` in the green
PR lane (it deliberately hangs) and lives in the **scheduled meta-test lane**
(`TEST-r2-meta-test-lane`) with inversion-assertion semantics, so the safety net
is continuously proven to still catch the fault. A red there means a *meta*-
regression (the safety net broke), distinct from a normal failure.

Every page emits a `stage`/`result` line sharing a `run_seq`, so the captured
NDJSON shows the sequential page loop advancing — and, on a hang, exactly which
page index stalled.

---

## 6. The benchmark-guardrail gate

> **Bead `bd-10sb` → `TEST-bench-guardrail`** — `benches/gauntlet.rs` +
> `scripts/bench_guardrail.py`. Plan §9.6: *"Benchmarks skip gracefully (exit 0)
> if the model fixture is absent. A benchmark-guardrail gate catches regressions
> vs a frozen baseline."*

**What it does:**

1. **Skips-green without fixtures.** Every criterion bench
   (`vision_encode`, `decode_token`, `end_to_end`, `gauntlet`) checks
   `native_model_available(resolve_model(...))`; when false it emits one line
   `{"event":"skip","result":"skip_no_model","bench":"<name>"}` and returns
   **without registering a criterion group (exit 0)**. A missing 6.67 GB file
   **never** red-flags CI.
2. **Compares to a frozen baseline.** With fixtures present, the guardrail reads
   the per-stage timings (criterion's machine output / `.bench-history`) and
   compares each stage's p50 to a **committed frozen baseline**, flagging any
   stage that regressed beyond a documented threshold.
3. **Flag-only first, hard-fail later** (§9.6 honest-hygiene caution). Wall-clock
   benches are noisy until baselines stabilize, so the guardrail is an
   **initially non-blocking, flag-only** CI job that *reports* regressions; it
   tightens to a hard gate once baselines settle. This mirrors the
   coverage-report discipline (`bd-10sb.2`).
4. **Fairness annotations are mandatory** (plan §9.3). Every guardrail row carries
   the fairness controls so a ratio is meaningful: **thread parity** (focr's
   thread budget `== OMP_NUM_THREADS`/`torch.set_num_threads(N)`; **never bench
   torch @64** — oversubscription inflates fake wins), **allocator posture**
   (mimalloc behind a feature wired into the *measured* binary, not merely
   mentioned), **precision annotation** (focr-int8 vs torch-bf16 — a raw ratio
   across different numerics is meaningless without it), **best-of-N with warmup
   discard**, and **`cv_pct`** (a coefficient of variation >5% is noise and the
   row is ineligible). The `gauntlet` bench shells out to the **Phase −1 proven
   CPU reference** via `FOCR_REFERENCE_CMD` / `FOCR_REFERENCE_BACKEND` (plan §9.3)
   and reports the **honest reference/focr ratio** (`>1.0` means `focr` is
   faster), not a self-relative number.
5. **Profile.** Benches build with the `[profile.release-perf]` profile
   (`debug=line-tables-only`, `lto=thin`, `codegen-units=1`) so timings reflect a
   profiling-grade build, not the dev profile.

**The guardrail's per-row record** is the same artifact-graph shape as
[`docs/PERF_LEDGER.md`](../PERF_LEDGER.md): `claim_id`, `evidence_id`
(`artifacts/perf/<bead>/`), `model_commit`, `fixture_hash`, `arch/cpu_features`,
`stage`, `focr_ms`, `ref_ms`, `ratio`, `floor_kind`/`floor_ms`/`dist_above_floor`
(the roofline), `precision`, `threads`, `allocator`, `command/env`,
`fallback/kill-switch state`. A guardrail run that lands a *win* writes that row;
a regression is flagged, not silently absorbed.

**Self-test:** an injected slowdown (a stage made artificially slower than the
frozen baseline) makes the guardrail flag the regression — proving the gate
actually compares. Like the watchdog's injected-deadlock, this fault-injection
self-test lives in the scheduled meta-test lane.

---

## 7. CI wiring

> **Bead `bd-4yks`** — `.github/workflows/ci.yml` (distinct from `dist.yml`, the
> 5-target *release* matrix). The capstone that turns these harnesses into an
> always-on safety net. The full design is in that bead; the load-bearing points
> for *this* document:

- **Always-on no-weights lane** (the default): `cargo fmt --check`,
  `cargo check --all-targets`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test`, `ubs $(git diff --name-only)`. Model-gated tests **skip-with-
  SUCCESS** (logged `skip_no_model`); CI is **green without weights**.
- **Log-schema validation** runs **after** capture: `cargo test` writes
  `$TEST_LOG_DIR/*.ndjson`, then the §3.6 validator runs over the whole captured
  stream and **fails CI on any malformed line**.
- **`many_pages_without_deadlock`** runs with a **hard job timeout** (the deadlock
  signal) against the real-topology stub variant (no weights).
- **Bench-guardrail** is a separate **flag-only** job (skips-green without
  fixtures; `release-perf` profile; fairness annotations).
- **Scheduled meta-test lane** (`TEST-r2-meta-test-lane`) runs the deliberately-
  failing/hanging `#[ignore]` self-tests (injected deadlock, injected slowdown,
  broken native path, malformed log line, injected nondeterminism) with
  inversion-assertion semantics, so the safety nets are continuously proven to
  still catch faults.
- **Scheduled model-FULL lane** (optional, self-hosted runner with the weights +
  a CUDA host for the oracle) runs the real model-gated e2e + the gauntlet
  head-to-head — clearly separated from the always-on no-weights CI (the official
  `infer()` is CUDA-oriented, OQ-17).
- **Artifact capture:** archive `TEST_LOG_DIR` NDJSON + `artifacts/perf/<bead>/`
  logs as CI artifacts for the §8.4 evidence graph.

---

## 8. Cross-references

- AGENTS.md — "Testing Policy", "Agent Ergonomics Requirements", doctrine #1
  (G1>G2), #5 (no nested rayon/runtime; the watchdog), #6 (the K=6848 overflow
  proof obligation surfaced in `stage` lines).
- [`COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`](../../COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md)
  §3.3 (runtime ownership / the deadlock rule), §8.2 (the L0–L5 ladder + the
  nondeterminism floor), §8.3 (model-gated e2e, metamorphic, golden), §8.4
  (the artifact-graph ledgers), §9.3 (the head-to-head fairness controls), §9.6
  (honest perf hygiene; the guardrail).
- [`docs/PROPOSED_ARCHITECTURE.md`](../PROPOSED_ARCHITECTURE.md) §3 (runtime
  ownership), §6 (the module / `stage`-enum boundaries), §10 (the §7.4 exit-code
  seam, the robot NDJSON seam).
- [`src/robot.rs`](../../src/robot.rs) — `ROBOT_SCHEMA_VERSION` (the versioning
  precedent `TEST_LOG_SCHEMA_VERSION` mirrors).
- [`src/error.rs`](../../src/error.rs) — `FocrError::exit_code()` (the single
  source of truth the error-path e2e asserts).
- [`scripts/oracle_nondeterminism_floor.py`](../../scripts/oracle_nondeterminism_floor.py)
  — produces the `nondeterminism_envelope` every parity line carries.
- [`scripts/gen_reference_fixtures.py`](../../scripts/gen_reference_fixtures.py)
  — produces the frozen `<doc>_reference.json` + per-stage `.npy` the e2e compares
  against.

*End. The schemas in §3.2/§3.3 are frozen contracts; bump `TEST_LOG_SCHEMA_VERSION`
on any breaking change and never alter an existing field's meaning silently.*
