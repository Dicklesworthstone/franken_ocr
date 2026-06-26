# BENCH_HARNESS.md — the head-to-head PERFORMANCE GAUNTLET (pillar (a))

> **Pillar (a)** of the three-pillar release gauntlet (`docs/gauntlet/METHODOLOGY.md`
> §5; `COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md` §9). This document describes the
> bench harness that MEASURES franken_ocr's forward path, compares it head-to-head
> against a proven CPU reference under fairness controls, and ratchets the result
> pass-over-pass. It is the running counterpart to the conformance/surface math in
> [`../../scripts/gauntlet_cert.py`](../../scripts/gauntlet_cert.py).
>
> **The harness MEASURES. There are no fabricated numbers anywhere in it.** Every
> row is a real wall-clock sample of a real kernel; the honest bar (doctrine #4)
> is encoded as *comments + ratchet assertions*, never as a hard-coded "win".

---

## 0. Where it lives

| File | Role |
|------|------|
| `benches/gauntlet_harness.rs` | The nightly `#[bench]` runner: drives the microbenches, the gated head-to-head stages, the scaffolded future-kernel slots, and the ratchet. |
| `benches/support/perf_harness.rs` | Pure infra (no `franken_ocr` dep, no serde): the `BenchRecord` result row, hand-rolled JSON, the `SampleStats` (p50/p90/CoV), the five-gate `Ratchet`, the `Fairness` knobs. Unit-tested standalone. |
| `reports/bench/latest.json` | The persisted `.bench-history` high-water mark the ratchet reads/writes when `FOCR_BENCH_HISTORY=reports/bench/latest.json` is set (git-tracked once baselines settle). |
| `docs/gauntlet/BENCH_HARNESS.md` | This document. |

The bench target is **auto-discovered on nightly** via `#![feature(test)]` +
`extern crate test;` + `#[bench] fn gauntlet(b: &mut test::Bencher)`. There is
**no `[[bench]]` manifest entry and no criterion** — adding either would edit
`Cargo.toml`, which the harness deliberately does not do. The `perf_harness`
module is pulled in with `#[path = "support/perf_harness.rs"] mod perf_harness;`.

Nothing here is linked into the shipping `focr` binary — like the PyO3 oracle
bridge, the harness is a verification-only artifact under `benches/`, so G3's
no-FFI single-binary runtime claim is preserved.

---

## 1. The honest bar (doctrine #4) — what "winning" actually means

The gap to ONNX/MLAS is **kernels below peak, NOT framework overhead.** This was
corrected by measurement (plan §3.2, `docs/NEGATIVE_EVIDENCE.md` NE-INH-3/4/5):

- A naive "fused tape-free forward" that swapped SIMD kernels for scalar-f32 ops
  regressed **3–10×**.
- An un-blocked SMMLA was **slower** than SDOT (load-bound).
- AMX-f32 **did not beat** ONNX-int8.

So the edge franken_ocr is *built* to have is the **combination it has by
construction**: a fused, tape-free, zero-per-op-allocation single-model forward,
**with every op at peak** (register-blocked SMMLA/VNNI linears + int8 attention
where accuracy allows + vectorized norms/softmax, NEVER naive), plus the **int4
bandwidth win** on the expert bulk.

| Stage | Honest target | Gate? |
|-------|---------------|-------|
| preprocess | parity-or-better (cheap) | no |
| vision-encode (prefill) | **parity-or-slower in f32 v1 is acceptable, recorded honestly** | no |
| decoder prefill GEMM | narrowing toward ONNX/MLAS via the built SMMLA/VNNI tiers | no |
| **decode-per-token** | **faster than the proven CPU reference on the primary arches** | **YES — the gating part** |
| end-to-end | faster is a tracked *stretch*, not a gate | no |

The bar lives in the harness as the `PRIMARY_BENCH`
(`decode_per_token_ref_gemv`) carrying the strictest (−3 %) ratchet gate, and as
the `head_to_head:decode_per_token` stage note — **as assertions and comments,
never as a fabricated baseline number**.

---

## 2. What runs without weights / without a baseline (always-green CI)

CI has no 6.67 GB weights and no torch/ONNX. The harness is built so a no-weights
run is **green, informative, and never silently empty**:

1. **Self-relative microbenches — ALWAYS run.** The reference/scalar kernels the
   harness writes itself (`ref_gemv_f32`, `ref_gemv_int8`, `ref_rms_norm_row`)
   need no model and no baseline. These are the rows that feed the pass-over-pass
   ratchet *today*, before any int8/int4/SMMLA kernel lands. Per doctrine #3 they
   are tight **scalar** loops that LLVM autovectorizes — the harness does NOT
   hand-roll wide SIMD.

2. **Head-to-head e2e stages — model-gated AND baseline-gated.** Each of
   `{preprocess, vision_encode, prefill, decode_per_token}` checks for a model
   artifact (`$FOCR_MODEL_DIR`, header-sniffed — no 6.67 GB read; either a
   directory containing `.focrq`/safetensors or a direct `.focrq`/safetensors
   file), a reference command (`$FOCR_REFERENCE_CMD`), and a concrete reference
   backend label
   (`$FOCR_REFERENCE_BACKEND`). Absent any of these, the stage
   **skips-with-SUCCESS**, emitting one NDJSON line that says exactly what it
   *would* have measured and why it skipped:

   ```json
   {"event":"skip","result":"skip_no_model_and_no_baseline",
    "bench":"head_to_head:decode_per_token",
    "would_measure":"R-SWA decode step vs reference (MUST be faster than CPU reference — the gate)"}
   ```

   The skip reasons are `skip_no_model_and_no_baseline`, `skip_no_model`,
   `skip_no_baseline`, `skip_no_reference_backend`,
   `skip_no_model_and_no_reference_backend`, and `ready`. A missing weights file
   **never** red-flags CI (LOGGING_AND_E2E §6), but an unlabeled reference backend
   also never becomes a measured timing event with `reference_backend:"unknown"`.

3. **Future-kernel slots — scaffolded, clearly logged, never silently empty.**
   The int8/int4/SMMLA GEMM tiers land in Phase 2–4 (plan §5.3). Until each
   kernel exists the harness logs its slot so the gauntlet *declares its intent*:

   ```json
   {"event":"scaffold","result":"future_kernel_slot","slot":"smmla_i8mm_prefill_gemm",
    "lands_in":"Phase 4 (plan §5.3 / §6.6 per-arch SIMD dispatch catalog)",
    "would_compare_to":"ref_gemv_int8 / matmul facade (bit-exact i32-accum; ≥~2x scalar prefill)"}
   ```

   The day the kernel lands, the `log_scaffold(...)` call is swapped for a real
   `records.push(...)` measured against the named reference row above — that
   reference *is* the isomorphism anchor (the fast path must match it bit-exactly
   in integer accumulation and within the measured int8 budget in f32).

---

## 3. The five `.bench-history` ratchet gates

Only **keep-eligible** rows (`cv_pct ≤ 5`) participate — a coefficient of
variation above 5 % is noise and the row is excluded from the ratchet entirely
(METHODOLOGY §5; the `keep_eligible` field on every emitted row records this).

The current round is compared against the persisted `latest.json` high-water
mark. A change **blocks** if any gate regresses past its floor:

| # | Gate | Threshold | Scope |
|---|------|----------:|-------|
| 1 | **primary** p50 | **−3 %** | the named primary bench (decode-per-token) |
| 2 | **geomean** p50 | **−5 %** | geomean of `current/base` p50 ratios over all paired benches |
| 3 | **per-category** geomean p50 | **−10 %** | geomean within each coarse bucket (`decode`, `prefill`, `vision`, `norm`, `kernel`) |
| 4 | **p90** | **−15 %** | worst single-bench tail regression (one bad tail blocks) |
| 5 | **throughput** | **−5 %** | worst single-bench throughput drop (higher-is-better) |

Signs: a *regression* is a worsening — for latency an increase, for throughput a
decrease — and the threshold is the worst tolerated relative move. A bench
present now but absent from the baseline is a **new** bench: it cannot regress,
so it never blocks; it seeds its own floor on the next write.

**The advance is monotone.** On `Allow`, the ratchet merges the round into the
baseline keeping the *best* value per bench (min latency, max throughput) — the
floor only ever tightens, mirroring `gauntlet_cert.py`'s `max(current, floor)`
move. A `Block` is logged (`ratchet_verdict` line) but does **not** panic the
bench; the CI guardrail job parses that line and fails the **flag-only** job
(LOGGING_AND_E2E §6.3 — flag-only first, hard gate once baselines settle).

The ratchet math is unit-tested with synthetic histories in
`benches/support/perf_harness.rs` (`mod tests`): each gate has a test that
*allows* an improvement / within-threshold wobble and *blocks* a past-threshold
regression, plus the new-bench, noisy-row-exclusion, monotone-advance, and
first-round-against-empty cases.

---

## 4. Fairness controls (plan §9.3) — ALL mandatory

A head-to-head ratio is meaningless without all of these, recorded on **every
row** (`perf_harness::Fairness`):

- **Thread parity.** focr's thread budget (`$FOCR_THREADS`, default **8**) is the
  number the reference baseline MUST be pinned to (`OMP_NUM_THREADS` /
  `torch.set_num_threads(N)`). The reference timing envelope must report the
  actual reference thread count, and the harness rejects missing or mismatched
  proof before it can become a measured row. **NEVER benchmark torch @64** —
  oversubscription inflates a fake "win" (a hardened frankentorch lesson). Measure
  at @8 / @32; let the §9.7 USL fit cap the decode pool at its peak, not at
  `num_cpus`.
- **Allocator fairness.** The allocator posture is read from how the *measured
  binary was built* (`Allocator::from_build`, a real `cfg!`, not an env var), so a
  row can never claim `mimalloc` on a binary that did not link it (the §9.3 trap:
  "wired into the measured binary, not merely mentioned"). The dep-free bench
  target currently has no allocator feature, so it honestly reports `system`; a
  future `--features mimalloc` claim flips the tag via the `cfg!`.
- **Reference identity and precision annotation.** Each row records the focr
  precision (`f32` / `bf16` / `int8` / `int4`) and, on a head-to-head, the
  `reference_backend` plus the reference precision. A raw ratio across different
  backends or numerics (`focr-int8` vs `torch-bf16`, ONNX vs HF vs GGUF) is a
  different claim, and the row says which. The reference timing envelope must
  report a non-empty `precision` or `reference_precision`; otherwise the harness
  logs `head_to_head_reference_error` instead of a timing row.
- **Best-of-N with warmup discard.** `SampleStats::from_durations` discards the
  warmup samples and reports p50/p90/**min** (the best-of-N) plus `cv_pct`.

The honest `reference/focr` ratio is computed per stage (`>1.0` means `focr`
is faster) and tagged
`focr_faster` / `ok` / `warn` / `slower` (`BenchRecord::ratio_tag`) — never a
self-relative number.

---

## 5. The environment contract

Mirrors `scripts/oracle_bridge.py` and plan §9.3 verbatim:

| Env var | Meaning | Default |
|---------|---------|---------|
| `FOCR_MODEL_DIR` | Where the 6.67 GB weights / `.focrq` live (header-sniffed; no tensor load to decide whether to skip). May be a directory containing `.focrq`/safetensors or a direct `.focrq`/safetensors artifact path. | unset ⇒ skip-with-SUCCESS |
| `FOCR_REFERENCE_CMD` | The Phase −1 proven CPU reference command shelled out per stage. | unset ⇒ skip-with-SUCCESS |
| `FOCR_REFERENCE_BACKEND` | Which reference backend the command speaks (`onnx` \| `hf` \| `gguf` \| `mlas` or another ledger-approved label). This is a backend label, not a precision annotation. | unset ⇒ skip-with-SUCCESS (`skip_no_reference_backend` once a command/model are present) |
| `FOCR_THREADS` | focr's positive thread budget; the reference is pinned to the SAME N. | `8`; unset, invalid, or `0` ⇒ `8` (NEVER 64) |
| `FOCR_BENCH_HISTORY` | Opt-in path for the monotone `.bench-history` ratchet, usually `reports/bench/latest.json`. | unset ⇒ log `ratchet_history_disabled`; do not write history |

### `FOCR_REFERENCE_CMD` timing envelope

When `FOCR_MODEL_DIR`, `FOCR_REFERENCE_CMD`, and `FOCR_REFERENCE_BACKEND` are all
present, the harness executes the reference command once per head-to-head stage
through the platform shell. For each invocation it sets:

| Env var | Value |
|---------|-------|
| `FOCR_GAUNTLET_STAGE` | One of `preprocess`, `vision_encode`, `prefill`, `decode_per_token`. |
| `FOCR_STAGE` | Alias with the same value, for smaller shell/python helpers. |
| `FOCR_THREADS` | The pinned thread budget. |
| `OMP_NUM_THREADS` / `TORCH_NUM_THREADS` / `RAYON_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `VECLIB_MAXIMUM_THREADS` / `BLIS_NUM_THREADS` / `NUMEXPR_NUM_THREADS` | The same pinned thread budget, so torch/ONNX/BLAS/Rayon helpers cannot drift to `@64`. |

The command's last non-empty JSON object line on stdout is the timing envelope.
The minimal successful shape is:

```json
{"stage":"decode_per_token","result":"pass","p50_ms":14.5,"precision":"bf16","threads":8}
```

Accepted duration keys are seconds (`p50_s`, `elapsed_s`, `seconds`, `ref_s`,
`reference_s`) or milliseconds (`p50_ms`, `elapsed_ms`, `milliseconds`,
`ref_ms`, `reference_ms`). The `result` field is required and must be `pass`,
`ok`, or `measured` for a timing row; values starting with `skip` are logged as
`head_to_head_reference_skip`; `fail`/`error`, missing `result`, non-zero exit,
malformed JSON, a stage mismatch, or a non-positive/non-finite duration are logged
as `head_to_head_reference_error`.

Measured envelopes must also carry a positive integer thread-proof field matching
the pinned `$FOCR_THREADS`: `threads`, `reference_threads`, `torch_threads`, or
`determinism.torch_threads`. Missing proof or `64` when the harness pinned `8`
is logged as `head_to_head_reference_error`, not as a timing row.

Measured envelopes must also carry a non-empty reference precision annotation:
`precision` or `reference_precision`. Missing or blank precision is rejected for
the same reason as missing thread proof: without it the head-to-head row cannot
support an honest `focr-int8` vs `torch-bf16` (or `int8` vs `int8`) claim.

This deliberately produces **reference timing events**, not PERF_LEDGER rows by
itself. A ledger row requires the matching native stage timer too, so the ratio
is a real `reference/focr` measurement rather than a reference-only number.

If CPU-patched HF cannot be proven equivalent to the CUDA correctness oracle
(OQ-17), the perf baseline is llama.cpp GGUF / ONNX Runtime / MLAS and is
**labeled as such** in `reference_backend`. The numeric posture still belongs in
`reference_precision`. The correctness oracle and the perf reference are split
(METHODOLOGY §1.2); this harness drives the **perf** side.

---

## 6. The proof obligations the perf rows carry (doctrine #6)

The int8 reference GEMV is measured at the **worst-case K = 6848** (the dense
layer-0 `down_proj`). That same shape is the home of the int8 **i32-accumulation
overflow proof obligation**: a unit test
(`int8_i32_accumulation_no_overflow_at_k6848`) asserts the i32 accumulator equals
an i64 oracle at K = 6848 under saturated worst-case operands. Worst case
signed×signed is `K·127² = 6848·16129 = 110,451,392 ≪ i32::MAX = 2,147,483,647`
(≈19× headroom); U8S8/VNNI `≤ K·255·127 ≈ 221.7M` still fits. We do **not** inherit
frankensearch's `k ≤ 1536` bound. The real SIMD tiers carry the same proof on
every arch (the `INV-I32-NOOVERFLOW` e-process, METHODOLOGY §6).

The harness also keeps the high-precision set honest: `ref_rms_norm_row` is a
**high-precision** norm (doctrine #2 — norms stay BF16/F32), and the
vectorized-transcendental lever must beat it *within the 8-ULP reduction
tolerance*, not by dropping precision.

---

## 7. Running it

```bash
# Always-green default (no weights, no baseline): self-relative microbenches +
# scaffold/skip logging. Does not write a bench-history file.
cargo +nightly bench --bench gauntlet_harness

# Opt into the monotone bench-history ratchet when the perf lane intends to
# update or compare the high-water mark.
FOCR_BENCH_HISTORY=reports/bench/latest.json \
cargo +nightly bench --bench gauntlet_harness

# Head-to-head (self-hosted model-FULL lane, weights + reference present):
FOCR_MODEL_DIR=~/.cache/franken_ocr/model \
FOCR_REFERENCE_CMD="python3 scripts/oracle_bridge.py --perf" \
FOCR_REFERENCE_BACKEND=onnx \
FOCR_THREADS=8 \
FOCR_BENCH_HISTORY=reports/bench/latest.json \
OMP_NUM_THREADS=8 RAYON_NUM_THREADS=8 \
cargo +nightly bench --bench gauntlet_harness

# The ratchet math + the i32-overflow proof also run under cargo test (the bench
# microbench loops use a tiny budget there so `cargo test` stays fast):
cargo +nightly test --bench gauntlet_harness
```

> **Profile.** Real perf claims build with `[profile.release-perf]`
> (`debug=line-tables-only, lto=thin, codegen-units=1`) so timings reflect a
> profiling-grade build, never a debug or default-release build (METHODOLOGY §5).

Every measured row, every skip, every scaffold, every ratchet gate, and the final
verdict is emitted as **one NDJSON object per line** (data-only on stdout, the
robot/agent style — TL2). With no `FOCR_BENCH_HISTORY`, the harness emits
`ratchet_history_disabled` and leaves the checkout untouched. When
`FOCR_BENCH_HISTORY` is set, `Allow` writes the new high-water mark to that path.

---

## 8. The `.bench-history` discipline

- `reports/bench/latest.json` is the committed, **monotone** high-water mark
  (`artifact: "franken_ocr.bench-history.v1"`). It is read at the start of every
  opted-in round (`FOCR_BENCH_HISTORY=reports/bench/latest.json`) and rewritten
  only on `Allow`, keeping the best value per bench. The default no-env run does
  not create or mutate generated benchmark artifacts in the working tree.
- The JSON is **hand-rolled and byte-stable** (sorted keys via `BTreeMap`, a fixed
  key order, shortest-round-trippable f64 formatting) so the git diff of the
  history file is meaningful and never flickers — the same cross-arch-determinism
  discipline as `gauntlet_cert.py`'s `truncate_score`. A missing file is the
  first-ever round (every gate vacuously passes, the round seeds the baseline); a
  *present-but-malformed* file is a **loud error**, never a silent empty baseline.
- A regression past any of the five gates **blocks** the bench gate. The
  fault-injection self-test (an injected slowdown made the guardrail flag the
  regression) lives in the scheduled meta-test lane (LOGGING_AND_E2E §6).
- Every kept win additionally writes a `PERF_LEDGER.md` row and a
  `NEGATIVE_EVIDENCE.md` entry if reverted (the 5-pass loop, doctrine #8); the
  ratchet is the automated floor, the ledgers are the human-auditable narrative.

---

## 9. Cross-references

- `AGENTS.md` — doctrine #3 (never hand-roll wide SIMD), #4 (the honest bar /
  kernels-below-peak), #6 (the K=6848 i32-overflow proof obligation), #8 (honest
  measured everything, the fairness controls, never torch @64).
- `docs/gauntlet/METHODOLOGY.md` §5 (the keep-gate + the five ratchet thresholds),
  §5.1 (the head-to-head fairness controls), §6 (the e-process invariants the perf
  rows surface).
- `COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md` §9 (performance methodology), §9.3
  (head-to-head + fairness), §9.6 (honest perf hygiene + the guardrail), §9.7 (the
  alien-artifact USL pool-sizing the thread budget reflects).
- `docs/testing/LOGGING_AND_E2E.md` §6 (the benchmark-guardrail gate this harness
  feeds), §4 (the model-gated skip-with-SUCCESS pattern it mirrors).
- `scripts/gauntlet_cert.py` — the conformance/surface ratchet math whose
  `Allow | Block` shape and monotone-high-water-mark discipline this perf ratchet
  mirrors.

## 10. Open dependency notes (for the owning agent / `deps_wanted`)

- The harness is auto-discovered as `benches/gauntlet_harness.rs`; **no
  `Cargo.toml` change is required** for nightly `#[bench]` discovery. *If* a
  `[[bench]] name = "gauntlet_harness" harness = false` manifest entry is later
  wanted (e.g. to run a custom main instead of the libtest harness), that is a
  Cargo.toml edit the bench-harness agent must NOT make silently — file it as a
  dependency request for whoever owns the manifest.
- `reports/bench/latest.json` should be added to the committed tree (and to CI
  artifact capture) once baselines stabilize; until then it is regenerated each
  round and the guardrail is flag-only.
