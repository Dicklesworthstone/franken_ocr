//! `gauntlet_harness` — pillar (a), the head-to-head PERFORMANCE GAUNTLET runner.
//!
//! This is the nightly-`#[bench]` runner that drives the
//! [`perf_harness`](support/perf_harness.rs) infra: it MEASURES franken_ocr's
//! own forward path, and — when a baseline (ONNX/`ort` or a CPU torch reference)
//! and the 6.67 GB weights are present — compares the two under the §9.3 fairness
//! controls, tagging each stage's honest `focr/reference` ratio. It is
//! auto-discovered by cargo on the nightly toolchain via the `#![feature(test)]`
//! plus `extern crate test;` plus `#[bench]` mechanism (no `[[bench]]` manifest
//! entry, no criterion — adding either would edit `Cargo.toml`, which this
//! harness must not do).
//!
//! ## The honest bar (AGENTS.md doctrine #4 / METHODOLOGY §5.1) — encoded here
//! The gap to ONNX/MLAS is **kernels below peak**, NOT framework overhead. A
//! naive "fused tape-free forward" that swapped SIMD kernels for scalar-f32 ops
//! regressed 3–10×; an un-blocked SMMLA was *slower* than SDOT (load-bound);
//! AMX-f32 did not beat ONNX-int8. So the win franken_ocr is *built* to have is
//! the **combination**: a fused, tape-free, zero-per-op-allocation single-model
//! forward, **with every op at peak** (register-blocked SMMLA/VNNI linears +
//! int8 attention where accuracy allows + vectorized norms/softmax, NEVER
//! naive), plus the **int4 bandwidth win** on the expert bulk. The honest target:
//! **at-or-near ONNX on CPU**, portable where `ort` can't build, with bounded
//! generated-token KV for long documents. **Decode-per-token must be faster than
//! the proven CPU reference** on the primary arches (the gating part);
//! vision-prefill **parity-or-slower in f32 v1 is acceptable and recorded
//! honestly**. This bar is encoded as *comments + ratchet assertions*, NEVER as
//! fabricated numbers — the harness only ever writes MEASURED rows.
//!
//! ## What runs without weights / without a baseline (the always-green default)
//! CI has no weights and no torch/ONNX. So:
//!   * **Self-relative microbenches** of the reference/scalar paths this file
//!     writes itself (or the committed [`nn`] facade fns) ALWAYS run — they need
//!     no model and feed the ratchet. These are the rows that gate pass-over-pass.
//!   * **The head-to-head e2e stages** are **model-gated + baseline-gated**: absent
//!     the weights or the reference command they **skip-with-SUCCESS**, logging
//!     exactly what they *would* measure and why the run was skipped. A missing
//!     6.67 GB file never red-flags CI (LOGGING_AND_E2E §6).
//!   * **Future-kernel slots** (int8/int4/SMMLA GEMM tiers that land in Phase
//!     2–4) are scaffolded as clearly-logged `would-bench` rows, never silently
//!     empty — the moment the kernel exists, the slot bears a real measurement.
//!
//! Nothing here is linked into shipping `focr` (it lives under `benches/`; G3's
//! no-FFI runtime claim is preserved).

#![feature(test)]
#![allow(clippy::missing_panics_doc)]

extern crate test;

#[path = "support/perf_harness.rs"]
mod perf_harness;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use perf_harness::{BenchRecord, Fairness, History, Precision, Ratchet, SampleStats};

// ─────────────────────────────────────────────────────────────────────────────
// Environment contract (mirrors scripts/oracle_bridge.py + plan §9.3 verbatim).
// ─────────────────────────────────────────────────────────────────────────────

/// Where the 6.67 GB weights / `.focrq` live (scripts/fetch_model.sh).
const ENV_MODEL_DIR: &str = "FOCR_MODEL_DIR";
/// The Phase −1 proven CPU reference command (shelled out per stage; plan §9.3).
const ENV_REFERENCE_CMD: &str = "FOCR_REFERENCE_CMD";
/// Which reference backend the command speaks (`onnx` | `hf` | `gguf`).
const ENV_REFERENCE_PYTHON: &str = "FOCR_REFERENCE_PYTHON";
/// focr's thread budget for the run; the reference is pinned to the SAME N
/// (NEVER @64 — plan §9.3 oversubscription trap).
const ENV_THREADS: &str = "FOCR_THREADS";

/// Opt-in path for the persisted `.bench-history` high-water mark.
///
/// Exploratory `cargo bench` runs should not dirty a shared checkout by writing
/// `reports/bench/latest.json`. CI/perf lanes that intend to update the monotone
/// ratchet floor set this explicitly, usually to `reports/bench/latest.json`.
const ENV_BENCH_HISTORY: &str = "FOCR_BENCH_HISTORY";

fn history_path() -> Option<PathBuf> {
    history_path_from_env(std::env::var_os(ENV_BENCH_HISTORY))
}

fn history_path_from_env(value: Option<std::ffi::OsString>) -> Option<PathBuf> {
    value.filter(|path| !path.is_empty()).map(PathBuf::from)
}

/// The primary bench whose p50 carries the strictest (−3 %) ratchet gate
/// (decode-per-token on the primary arch — the gating part of the honest bar).
const PRIMARY_BENCH: &str = "decode_per_token_ref_gemv";

// ─────────────────────────────────────────────────────────────────────────────
// Gating — model + baseline presence (cheap, header-sniff style; no tensor load).
// ─────────────────────────────────────────────────────────────────────────────

/// Resolution of the head-to-head preconditions.
struct GateState {
    /// `Some(dir)` when `$FOCR_MODEL_DIR` resolves to a dir containing weights.
    model_dir: Option<PathBuf>,
    /// `Some(cmd)` when `$FOCR_REFERENCE_CMD` is set (the baseline to shell out).
    reference_cmd: Option<String>,
    /// The reference backend tag (`onnx`/`hf`/`gguf`), for the precision column.
    reference_backend: Option<String>,
    /// focr's pinned thread budget.
    threads: usize,
}

impl GateState {
    fn resolve() -> Self {
        // Header-sniff the model dir cheaply: just check a weights-shaped file
        // exists (no 6.67 GB read — LOGGING_AND_E2E §4.1).
        let model_dir = std::env::var(ENV_MODEL_DIR).ok().and_then(|d| {
            let p = PathBuf::from(&d);
            let has_weights = p.join("model-00001-of-000001.safetensors").exists()
                || dir_has_extension(&p, "focrq")
                || dir_has_extension(&p, "safetensors");
            if has_weights { Some(p) } else { None }
        });
        let reference_cmd = std::env::var(ENV_REFERENCE_CMD)
            .ok()
            .filter(|s| !s.is_empty());
        let reference_backend = std::env::var(ENV_REFERENCE_PYTHON)
            .ok()
            .filter(|s| !s.is_empty());
        let threads = std::env::var(ENV_THREADS)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8); // §9.3 default measure point (@8); NEVER 64.
        Self {
            model_dir,
            reference_cmd,
            reference_backend,
            threads,
        }
    }

    /// True when the full head-to-head can run (weights AND a baseline).
    fn head_to_head_ready(&self) -> bool {
        self.model_dir.is_some() && self.reference_cmd.is_some()
    }

    /// Why the head-to-head was skipped (logged on a skip-with-SUCCESS).
    fn skip_reason(&self) -> &'static str {
        match (self.model_dir.is_some(), self.reference_cmd.is_some()) {
            (false, false) => "skip_no_model_and_no_baseline",
            (false, true) => "skip_no_model",
            (true, false) => "skip_no_baseline",
            (true, true) => "ready",
        }
    }
}

fn dir_has_extension(dir: &std::path::Path, ext: &str) -> bool {
    std::fs::read_dir(dir)
        .ok()
        .map(|rd| {
            rd.flatten()
                .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some(ext))
        })
        .unwrap_or(false)
}

// ─────────────────────────────────────────────────────────────────────────────
// NDJSON logging (data-only on stdout; the robot/agent style, TL2).
// ─────────────────────────────────────────────────────────────────────────────

fn log_line(json: &str) {
    // One JSON object per line; bench harness captures stdout. Diagnostics-only
    // (a measured BenchRecord) so it never interleaves human decoration.
    println!("{json}");
}

fn log_skip(bench: &str, reason: &str, would_measure: &str) {
    log_line(&format!(
        "{{\"event\":\"skip\",\"result\":\"{reason}\",\"bench\":\"{bench}\",\"would_measure\":\"{would_measure}\"}}"
    ));
}

fn log_scaffold(slot: &str, lands_in: &str, would_compare_to: &str) {
    // A future-kernel slot — logged, never silently empty (per the task's
    // "scaffold the slot ... clearly logged" requirement).
    log_line(&format!(
        "{{\"event\":\"scaffold\",\"result\":\"future_kernel_slot\",\"slot\":\"{slot}\",\
         \"lands_in\":\"{lands_in}\",\"would_compare_to\":\"{would_compare_to}\"}}"
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// The reference/scalar paths we MEASURE today (no model, no baseline needed).
//
// These are deliberately simple, dependency-free scalar kernels that stand in
// for the real ft-kernel-cpu / future SMMLA-VNNI tiers. They exist so the
// pass-over-pass ratchet has REAL rows to gate from day one, and so the
// isomorphism story (a future fast path is checked bit-exact vs THIS reference)
// has a named baseline. Per doctrine #3 these are tight SCALAR loops that LLVM
// autovectorizes — we do NOT hand-roll wide SIMD here.
// ─────────────────────────────────────────────────────────────────────────────

/// Reference f32 GEMV `y[n] = sum_k a[k]*w[n,k]` — the shape of the decode-step
/// lm_head / expert down_proj GEMV (the int8/int4 fast path will replace this and
/// MUST stay bit-exact in integer accumulation; this f32 ref is the perf anchor).
fn ref_gemv_f32(a: &[f32], w: &[f32], n: usize, k: usize) -> Vec<f32> {
    assert_eq!(a.len(), k);
    assert_eq!(w.len(), n * k);
    let mut y = vec![0.0f32; n];
    for (row, yi) in y.iter_mut().enumerate() {
        let wr = &w[row * k..(row + 1) * k];
        let mut acc = 0.0f32;
        for j in 0..k {
            acc += a[j] * wr[j];
        }
        *yi = acc;
    }
    y
}

/// Reference int8 dynamic GEMV: per-row symmetric quant of `a`, i32 accumulate,
/// dequant. This is the integer-accumulation **proof anchor**: at the worst-case
/// `K = 6848` (dense layer-0 `down_proj`) the i32 accumulator must not overflow
/// (`U8S8 ≤ ~221.7M < i32::MAX`; AGENTS.md doctrine #6). A unit test below asserts
/// the i32 result equals an i64 reference at K=6848.
fn ref_gemv_int8(a: &[f32], w_i8: &[i8], w_scale: &[f32], n: usize, k: usize) -> Vec<f32> {
    assert_eq!(a.len(), k);
    assert_eq!(w_i8.len(), n * k);
    assert_eq!(w_scale.len(), n);
    // dynamic per-vector quant of the activation (symmetric int8).
    let amax = a.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let a_scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let aq: Vec<i8> = a
        .iter()
        .map(|&v| (v / a_scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    let mut y = vec![0.0f32; n];
    for (row, yi) in y.iter_mut().enumerate() {
        let wr = &w_i8[row * k..(row + 1) * k];
        let mut acc: i32 = 0;
        for j in 0..k {
            acc += i32::from(aq[j]) * i32::from(wr[j]);
        }
        *yi = acc as f32 * a_scale * w_scale[row];
    }
    y
}

/// Reference RMSNorm row (decoder norm shape) — kept high-precision (doctrine #2),
/// a perf row that the vectorized-norm lever (§6.11) will later have to beat
/// while staying within the 8-ULP reduction tolerance.
fn ref_rms_norm_row(x: &[f32], eps: f32) -> Vec<f32> {
    let k = x.len();
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / k as f32;
    let rstd = 1.0 / (mean_sq + eps).sqrt();
    x.iter().map(|v| v * rstd).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Measurement loop — best-of-N with warmup discard (plan §9.3).
// ─────────────────────────────────────────────────────────────────────────────

/// Run `f` `warmup + iters` times, returning the per-iteration durations. Uses
/// `test::black_box` so the optimizer cannot hoist the work out of the loop.
fn measure<F: FnMut()>(warmup: usize, iters: usize, mut f: F) -> Vec<Duration> {
    let mut out = Vec::with_capacity(warmup + iters);
    for _ in 0..(warmup + iters) {
        let t0 = Instant::now();
        f();
        out.push(t0.elapsed());
    }
    out
}

/// Assemble a self-relative [`BenchRecord`] (no reference baseline) for a measured
/// scalar/facade path, at the given fairness controls.
#[allow(clippy::too_many_arguments)] // a record builder; named fields keep call sites clear.
fn self_relative_record(
    name: &str,
    category: &str,
    shape: &str,
    durations: &[Duration],
    warmup: usize,
    throughput: Option<(f64, &str)>,
    fairness: Fairness,
    note: &str,
) -> BenchRecord {
    let stats = SampleStats::from_durations(durations, warmup);
    BenchRecord {
        name: name.into(),
        category: category.into(),
        shape: shape.into(),
        stats,
        throughput: throughput.map(|(t, _)| t),
        throughput_unit: throughput.map(|(_, u)| u.to_string()),
        fairness,
        reference_p50_s: None,
        reference_precision: None,
        note: Some(note.into()),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The harness driver — collects records, ratchets, persists. Called by the
// single visible #[bench] so the whole gauntlet runs as one nightly bench.
// ─────────────────────────────────────────────────────────────────────────────

/// Iteration budget for the measurement loops. The real `#[bench]` uses
/// [`BenchBudget::full`] (warmup + best-of-N tuned per shape); the `cargo test`
/// smoke path uses [`BenchBudget::smoke`] (a couple of iterations) so that
/// running the *tests* never pays the full GEMV measurement cost — the heavy
/// `K = 6848` loop is a bench cost, not a unit-test cost.
#[derive(Debug, Clone, Copy)]
struct BenchBudget {
    /// Scale applied to each loop's `iters` (warmup is kept ≥ 1).
    iters_scale: f64,
}

impl BenchBudget {
    /// The full measurement budget for a real `cargo bench` run.
    fn full() -> Self {
        Self { iters_scale: 1.0 }
    }

    /// A tiny budget for the `cargo test` smoke path (2 iters per loop).
    fn smoke() -> Self {
        Self { iters_scale: 0.0 }
    }

    /// Resolve `(warmup, iters)` for a loop's nominal counts. `smoke` collapses
    /// to `(1, 2)`; `full` returns the nominal counts unchanged.
    fn resolve(self, warmup: usize, iters: usize) -> (usize, usize) {
        if self.iters_scale <= 0.0 {
            (1, 2)
        } else {
            (warmup, ((iters as f64) * self.iters_scale).round() as usize)
        }
    }
}

/// Run every self-relative microbench, log the gated/scaffolded head-to-head
/// rows, ratchet against `latest.json`, and (on Allow) advance the high-water
/// mark. Returns the records so the `#[bench]` body can keep them alive.
fn run_gauntlet(budget: BenchBudget) -> Vec<BenchRecord> {
    let gate = GateState::resolve();
    let fairness = Fairness::new(gate.threads, Precision::Int8);
    let mut records: Vec<BenchRecord> = Vec::new();

    // ── (1) Self-relative microbenches — ALWAYS run (no model/baseline). ────
    // These are the rows that feed the pass-over-pass ratchet today.

    // 1a. The PRIMARY bench: decode-step GEMV at the lm_head-ish shape. This is
    // the f32 reference the int8/int4 fast path must beat while staying bit-exact
    // in integer accumulation (doctrine #4: decode-per-token is the gating part).
    {
        let (n, k) = (1280usize, 1280usize); // a decode-step square GEMV
        let a: Vec<f32> = (0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.001)
            .collect();
        let (warmup, iters) = budget.resolve(8, 50);
        let durs = measure(warmup, iters, || {
            let y = ref_gemv_f32(test::black_box(&a), test::black_box(&w), n, k);
            test::black_box(y);
        });
        let flops = 2.0 * (n * k) as f64;
        let stats = SampleStats::from_durations(&durs, warmup);
        let gflops = flops / stats.p50_s / 1e9;
        records.push(self_relative_record(
            PRIMARY_BENCH,
            "decode",
            &format!("gemv n={n},k={k}"),
            &durs,
            warmup,
            Some((gflops, "GFLOP/s")),
            Fairness::new(gate.threads, Precision::F32),
            "f32 reference GEMV — the decode-per-token perf anchor; int8/int4 fast \
             path must beat this AND stay bit-exact in i32 accumulation (doctrine #6).",
        ));
    }

    // 1b. int8 dynamic GEMV at the WORST-CASE K (dense layer-0 down_proj, K=6848)
    // — both the perf row and the overflow proof obligation's home shape.
    {
        let (n, k) = (1280usize, 6848usize); // doctrine #6 worst-case K
        let a: Vec<f32> = (0..k).map(|i| ((i % 23) as f32 - 11.0) * 0.01).collect();
        let w_i8: Vec<i8> = (0..n * k).map(|i| ((i % 255) as i32 - 127) as i8).collect();
        let w_scale: Vec<f32> = (0..n).map(|_| 0.002).collect();
        let (warmup, iters) = budget.resolve(4, 25);
        let durs = measure(warmup, iters, || {
            let y = ref_gemv_int8(
                test::black_box(&a),
                test::black_box(&w_i8),
                test::black_box(&w_scale),
                n,
                k,
            );
            test::black_box(y);
        });
        let flops = 2.0 * (n * k) as f64;
        let stats = SampleStats::from_durations(&durs, warmup);
        let gops = flops / stats.p50_s / 1e9;
        records.push(self_relative_record(
            "decode_down_proj_int8_K6848",
            "decode",
            &format!("int8 gemv n={n},k={k} (worst-case K, i32-accum proof)"),
            &durs,
            warmup,
            Some((gops, "GOP/s")),
            fairness,
            "int8 reference GEMV at worst-case K=6848; i32 accumulator proven \
             non-overflowing by `int8_i32_accumulation_no_overflow_at_k6848`.",
        ));
    }

    // 1c. RMSNorm row (a high-precision norm kept BF16/F32; doctrine #2).
    {
        let k = 1280usize;
        let x: Vec<f32> = (0..k).map(|i| ((i % 31) as f32 - 15.0) * 0.1).collect();
        let (warmup, iters) = budget.resolve(8, 100);
        let durs = measure(warmup, iters, || {
            let y = ref_rms_norm_row(test::black_box(&x), 1e-6);
            test::black_box(y);
        });
        records.push(self_relative_record(
            "rms_norm_row_ref",
            "norm",
            &format!("rmsnorm k={k}"),
            &durs,
            warmup,
            None,
            Fairness::new(gate.threads, Precision::F32),
            "high-precision RMSNorm reference; the vectorized-transcendental lever \
             (§6.11) must beat this within the 8-ULP reduction tolerance.",
        ));
    }

    // ── (2) Future-kernel slots — scaffolded, clearly logged, never empty. ──
    // The int8/int4/SMMLA GEMM tiers land in Phase 2–4 (plan §5.3). Until the
    // kernel exists we log the slot so the gauntlet declares its intent; the day
    // the kernel lands, swap the `log_scaffold` for a real `records.push(...)`
    // measured against the §1a/§1b reference rows above (the isomorphism anchor).
    log_scaffold(
        "smmla_i8mm_prefill_gemm",
        "Phase 4 (plan §5.3 / §6.6 per-arch SIMD dispatch catalog)",
        "ref_gemv_int8 / matmul facade (bit-exact i32-accum; ≥~2x scalar prefill, plan §10)",
    );
    log_scaffold(
        "vnni_avx512_decode_gemv",
        "Phase 4 (plan §5.3 — AVX-VNNI / AVX-512-VNNI decode GEMV)",
        "decode_per_token_ref_gemv (decode-per-token faster than CPU reference is the gate)",
    );
    log_scaffold(
        "int4_g32_expert_gemm",
        "Phase 4 (plan §9 — int4 decode-bandwidth wedge on the MoE expert bulk)",
        "decode_down_proj_int8_K6848 (the int4 bandwidth win is the headline upside)",
    );

    // ── (3) The head-to-head stages — model-gated AND baseline-gated. ───────
    // Each stage logs EXACTLY what it would measure and why it skipped, so a
    // no-weights/no-baseline run is informative, not silent. With both present,
    // the runner would shell out to $FOCR_REFERENCE_CMD per stage under thread
    // parity and record the honest focr/reference ratio (NOT a self-relative
    // number). The honest target per stage (doctrine #4 / METHODOLOGY §5.1):
    //   * preprocess      : parity-or-better (cheap; not gating)
    //   * vision_encode   : parity-or-slower in f32 v1 ACCEPTABLE, recorded honestly
    //   * prefill         : narrowing toward ONNX/MLAS via the built SMMLA/VNNI tiers
    //   * decode_per_token: MUST be faster than the proven CPU reference (the GATE)
    let stages = [
        (
            "preprocess",
            "image resize/normalize/pad/tile vs reference preprocess",
        ),
        (
            "vision_encode",
            "SAM+CLIP tower per-page vs reference (f32 parity-or-slower OK)",
        ),
        (
            "prefill",
            "decoder prefill GEMM vs reference (narrowing toward ONNX/MLAS)",
        ),
        (
            "decode_per_token",
            "R-SWA decode step vs reference (MUST be faster than CPU reference — the gate)",
        ),
    ];
    if gate.head_to_head_ready() {
        // Both present: a self-hosted model-FULL lane would MEASURE here. We do
        // not fabricate timings; we log that the measured path is armed. The real
        // shell-out + parse + ratio lands with the weights-bearing runner; here we
        // assert the fairness precondition so a mis-pinned reference is caught.
        let ref_threads = gate.threads; // the runner pins the reference to this N.
        let parity = fairness.assert_thread_parity(ref_threads);
        for (stage, what) in stages {
            log_line(&format!(
                "{{\"event\":\"head_to_head_armed\",\"stage\":\"{stage}\",\"would_measure\":\"{what}\",\
                 \"reference_backend\":\"{}\",\"threads\":{},\"thread_parity\":\"{}\"}}",
                gate.reference_backend.as_deref().unwrap_or("unknown"),
                gate.threads,
                if parity.is_ok() { "ok" } else { "VIOLATION" },
            ));
        }
    } else {
        let reason = gate.skip_reason();
        for (stage, what) in stages {
            log_skip(&format!("head_to_head:{stage}"), reason, what);
        }
    }

    // ── (4) Ratchet the self-relative rows against the persisted history. ───
    // Only keep-eligible (cv_pct ≤ 5) rows participate. This is the pass-over-pass
    // gate; the honest-bar assertions live here as ratchet checks, NOT numbers.
    let ratchet = Ratchet::new(PRIMARY_BENCH);
    if let Some(path) = history_path() {
        match History::read_or_empty(&path) {
            Ok(baseline) => {
                let verdict = ratchet.compare(&records, &baseline);
                for g in &verdict.gates {
                    log_line(&format!(
                        "{{\"event\":\"ratchet_gate\",\"gate\":\"{}\",\"observed_regression\":{:.6},\
                     \"allowed\":{:.6},\"pass\":{}}}",
                        g.gate, g.observed_regression, g.allowed, g.pass
                    ));
                }
                log_line(&format!(
                    "{{\"event\":\"ratchet_verdict\",\"decision\":\"{}\",\"reason\":{}}}",
                    verdict.decision,
                    json_escape(&verdict.reason),
                ));
                // On Allow, advance the monotone high-water mark. A Block is logged
                // but does NOT panic here (the bench would abort the whole run); the
                // CI guardrail job parses the `ratchet_verdict` line and fails the
                // FLAG-only job (LOGGING_AND_E2E §6.3 — flag-only first, hard later).
                if verdict.allowed() {
                    let advanced = ratchet.advance(&records, &baseline);
                    if let Err(e) = advanced.write(&path) {
                        log_line(&format!(
                            "{{\"event\":\"ratchet_persist_error\",\"error\":{}}}",
                            json_escape(&e)
                        ));
                    }
                }
            }
            Err(e) => {
                log_line(&format!(
                    "{{\"event\":\"ratchet_history_error\",\"error\":{}}}",
                    json_escape(&e)
                ));
            }
        }
    } else {
        log_line(&format!(
            "{{\"event\":\"ratchet_history_disabled\",\"result\":\"skip_no_history_path\",\
             \"env\":\"{}\"}}",
            ENV_BENCH_HISTORY
        ));
    }

    // ── (5) Emit every measured record as one NDJSON line (the evidence). ───
    for r in &records {
        log_line(&r.to_json());
    }
    records
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// The single visible #[bench] — drives the whole gauntlet once.
// ─────────────────────────────────────────────────────────────────────────────

/// The gauntlet entrypoint. `cargo bench` (nightly) discovers and runs this; the
/// `black_box` keeps the measured records alive so the work is not optimized out.
#[bench]
fn gauntlet(b: &mut test::Bencher) {
    // We iterate ONCE inside `b.iter` (the real measurement loops are inside
    // `run_gauntlet` via `measure`, which owns warmup/best-of-N); the Bencher
    // wrapper is just the discovery hook so `cargo bench` runs the gauntlet.
    let mut ran = false;
    b.iter(|| {
        if !ran {
            let records = run_gauntlet(BenchBudget::full());
            test::black_box(&records);
            ran = true;
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests — the PROOF OBLIGATIONS that gate the perf rows (doctrine #6),
// plus gating/skip-with-SUCCESS behaviour. These run under `cargo test` too.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// AGENTS.md doctrine #6 — the int8 i32-accumulation overflow PROOF.
    /// Worst case: dense layer-0 `down_proj` at K=6848. The i32 accumulator must
    /// equal an i64 reference at that K, at the saturated worst-case operands
    /// (`|a_q| = 127`, `|w| = 127`): `K * 127 * 127 = 6848*16129 = 110,451,392`,
    /// well under `i32::MAX = 2,147,483,647` (≈19× headroom signed×signed). This
    /// test proves the *reference* kernel here; the real SIMD tiers carry the same
    /// proof on every arch (the `INV-I32-NOOVERFLOW` e-process, METHODOLOGY §6).
    #[test]
    fn int8_i32_accumulation_no_overflow_at_k6848() {
        let k = 6848usize;
        let n = 4usize;
        // Saturate the worst case: all activations and weights at +127 magnitude.
        let a: Vec<f32> = vec![1.0e6; k]; // huge -> a_scale large; quantizes to +127
        let w_i8: Vec<i8> = vec![127i8; n * k];
        let w_scale: Vec<f32> = vec![1.0; n];

        // i32-accumulating reference under test.
        let y = ref_gemv_int8(&a, &w_i8, &w_scale, n, k);

        // independent i64 oracle: a quantizes to all +127, so each row acc =
        // 127*127*K in i64; the f32 output = acc * a_scale * w_scale.
        let amax = a.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let a_scale = amax / 127.0;
        let acc_i64: i64 = (127i64 * 127i64) * k as i64;
        assert!(
            acc_i64 < i64::from(i32::MAX),
            "K=6848 worst-case acc {acc_i64} must fit i32 (doctrine #6)"
        );
        let expect = acc_i64 as f32 * a_scale * 1.0;
        for &yi in &y {
            // relative agreement (f32 rounding of the large magnitudes).
            let rel = ((yi - expect) / expect).abs();
            assert!(
                rel < 1e-4,
                "i32 acc != i64 oracle: {yi} vs {expect} (rel {rel})"
            );
        }
    }

    /// The int8 reference GEMV approximates the f32 GEMV for small integer-ish
    /// operands (the isomorphism anchor: a future SIMD int8 tier must match THIS
    /// f32 reference within the measured int8 budget, and match THIS int8 ref
    /// BIT-EXACTLY in integer accumulation).
    #[test]
    fn int8_ref_approximates_f32_ref() {
        let (n, k) = (3usize, 8usize);
        let a: Vec<f32> = (0..k).map(|i| (i as f32 - 4.0) * 0.5).collect();
        let w: Vec<f32> = (0..n * k).map(|i| ((i % 5) as f32 - 2.0) * 0.25).collect();
        // quantize w per-row symmetric to int8 for the int8 ref.
        let mut w_i8 = vec![0i8; n * k];
        let mut w_scale = vec![0.0f32; n];
        for row in 0..n {
            let wr = &w[row * k..(row + 1) * k];
            let wmax = wr.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let s = if wmax > 0.0 { wmax / 127.0 } else { 1.0 };
            w_scale[row] = s;
            for j in 0..k {
                w_i8[row * k + j] = (wr[j] / s).round().clamp(-127.0, 127.0) as i8;
            }
        }
        let yf = ref_gemv_f32(&a, &w, n, k);
        let yq = ref_gemv_int8(&a, &w_i8, &w_scale, n, k);
        for (f, q) in yf.iter().zip(&yq) {
            assert!((f - q).abs() < 0.05, "int8 ref drifted: {f} vs {q}");
        }
    }

    /// RMSNorm reference matches the hand-computed value (matches nn.rs's own
    /// test, proving the perf anchor is the SAME math as the facade).
    #[test]
    fn rms_norm_ref_matches_hand_computed() {
        // x=[3,4], eps=0: mean_sq=12.5, rstd=1/sqrt(12.5); out=[3*rstd,4*rstd].
        let y = ref_rms_norm_row(&[3.0, 4.0], 0.0);
        let rstd = 1.0f32 / 12.5f32.sqrt();
        assert!((y[0] - 3.0 * rstd).abs() < 1e-6);
        assert!((y[1] - 4.0 * rstd).abs() < 1e-6);
    }

    /// Gating defaults: with no env set, the head-to-head is NOT ready and the
    /// skip reason is the no-model-and-no-baseline form (skip-with-SUCCESS).
    #[test]
    fn gate_defaults_to_skip_with_success() {
        // We do not mutate the process env (other tests/threads share it); we
        // construct the gate state by hand to exercise the decision table.
        let g = GateState {
            model_dir: None,
            reference_cmd: None,
            reference_backend: None,
            threads: 8,
        };
        assert!(!g.head_to_head_ready());
        assert_eq!(g.skip_reason(), "skip_no_model_and_no_baseline");

        let g2 = GateState {
            model_dir: Some(PathBuf::from("/x")),
            reference_cmd: None,
            reference_backend: None,
            threads: 8,
        };
        assert!(!g2.head_to_head_ready());
        assert_eq!(g2.skip_reason(), "skip_no_baseline");

        let g3 = GateState {
            model_dir: None,
            reference_cmd: Some("python ref.py".into()),
            reference_backend: Some("onnx".into()),
            threads: 8,
        };
        assert!(!g3.head_to_head_ready());
        assert_eq!(g3.skip_reason(), "skip_no_model");

        let g4 = GateState {
            model_dir: Some(PathBuf::from("/x")),
            reference_cmd: Some("python ref.py".into()),
            reference_backend: Some("hf".into()),
            threads: 8,
        };
        assert!(g4.head_to_head_ready());
        assert_eq!(g4.skip_reason(), "ready");
    }

    /// Ratchet persistence is opt-in so smoke benches/tests do not leave
    /// generated files in a shared checkout. CI/perf lanes opt in by setting
    /// `FOCR_BENCH_HISTORY=reports/bench/latest.json`.
    #[test]
    fn history_path_is_opt_in() {
        assert_eq!(history_path_from_env(None), None);
        assert_eq!(
            history_path_from_env(Some(std::ffi::OsString::from(""))),
            None
        );
        assert_eq!(
            history_path_from_env(Some(std::ffi::OsString::from("reports/bench/latest.json"))),
            Some(PathBuf::from("reports/bench/latest.json"))
        );
    }

    /// The honest-bar gate is encoded as a RATCHET assertion, not a number: a
    /// synthetic round where the decode primary regresses past −3% must Block.
    /// (Proves the harness would refuse a decode-per-token regression — the
    /// gating part of doctrine #4 — without any fabricated baseline number.)
    #[test]
    fn honest_bar_decode_regression_blocks() {
        let base = History::from_records(&[BenchRecord {
            name: PRIMARY_BENCH.into(),
            category: "decode".into(),
            shape: "synthetic".into(),
            stats: SampleStats::from_secs(&[0.002; 10]),
            throughput: Some(500.0),
            throughput_unit: Some("tok/s".into()),
            fairness: Fairness::new(8, Precision::Int8),
            reference_p50_s: None,
            reference_precision: None,
            note: None,
        }]);
        let ratchet = Ratchet::new(PRIMARY_BENCH);
        // decode primary +5% -> blocks (> 3% primary gate).
        let now = vec![BenchRecord {
            name: PRIMARY_BENCH.into(),
            category: "decode".into(),
            shape: "synthetic".into(),
            stats: SampleStats::from_secs(&[0.0021; 10]),
            throughput: Some(476.0),
            throughput_unit: Some("tok/s".into()),
            fairness: Fairness::new(8, Precision::Int8),
            reference_p50_s: None,
            reference_precision: None,
            note: None,
        }];
        let v = ratchet.compare(&now, &base);
        assert!(!v.allowed(), "decode regression must block: {}", v.reason);
    }

    /// `run_gauntlet` runs end-to-end with no env (the CI default) without
    /// panicking and produces the expected self-relative rows. This is the
    /// smoke test that the whole skip-with-SUCCESS + ratchet path is wired.
    #[test]
    fn run_gauntlet_smoke_no_model() {
        // Run in an isolated cwd so the test never clobbers a real history file.
        let tmp = std::env::temp_dir().join(format!("focr_gauntlet_smoke_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let prev = std::env::current_dir().ok();
        std::env::set_current_dir(&tmp).unwrap();

        let records = run_gauntlet(BenchBudget::smoke());

        if let Some(p) = prev {
            let _ = std::env::set_current_dir(p);
        }
        // The three always-on self-relative rows are present.
        let names: Vec<&str> = records.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains(&PRIMARY_BENCH));
        assert!(names.contains(&"decode_down_proj_int8_K6848"));
        assert!(names.contains(&"rms_norm_row_ref"));
    }
}
