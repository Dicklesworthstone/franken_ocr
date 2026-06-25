//! `many_pages_without_deadlock` — the concurrency WATCHDOG from doctrine #5.
//!
//! This is the durable fix the frankensearch deadlock saga converged on, made
//! executable. Doctrine #5 (AGENTS.md):
//!
//! > NEVER nest rayon under a held lock; NEVER nest a second asupersync runtime
//! > inside a task. Single `OcrModel` behind a cache; **sequential** outer
//! > page/document loop; each forward fans out across all cores internally via
//! > the kernel's own rayon pool (pinned to physical cores, one live forward at
//! > a time). A `many_pages_without_deadlock` CI watchdog (pages ≫ pool) hangs
//! > on regression.
//!
//! ## What this protects
//!
//! A regression that (a) nests rayon under a held lock, (b) spins a *second*
//! asupersync runtime inside a task, or (c) drives the outer page loop with a
//! concurrent fan-out instead of the mandated **sequential** drive will DEADLOCK
//! or livelock when pages ≫ the kernel rayon pool size. A plain
//! `for page in pages { engine.recognize(...) }` would then simply hang forever
//! and a naive test would hang with it (CI timeout, no diagnosis).
//!
//! ## The watchdog shape
//!
//! We do NOT rely on the harness/CI process timeout (which produces an
//! undiagnosed "test timed out" with no breadcrumbs). Instead each scenario runs
//! the whole batch on a dedicated worker thread and the test thread blocks on
//! `mpsc::Receiver::recv_timeout(BUDGET)`. If the batch finishes in time we get
//! its result; if the budget elapses first we FAIL LOUDLY with
//! `"DEADLOCK SUSPECTED"` plus the exact pool size, pages issued, completion
//! count, and budget — a self-diagnosing failure (testing skill: make failures
//! self-diagnosing).
//!
//! ## Two layers, both bounded by the same watchdog
//!
//! 1. **Sequential outer loop** (`many_sequential_pages_complete_within_budget`):
//!    one `OcrEngine`, many sequential `recognize_with_model` calls, pages ≫
//!    pool. Today each call returns a clean fast `ModelNotFound` (no weights),
//!    so this exercises the ORCHESTRATION / outer-loop + per-call
//!    runtime-`block_on` path that the deadlock rule guards — exactly the path a
//!    nested-runtime or rayon-under-lock regression would hang on. When weights
//!    land it exercises the full forward (gated behind `FOCR_MODEL_PATH`, same
//!    skip-with-SUCCESS contract as the model-gated e2e tests).
//!
//! 2. **Engine-construction-under-concurrency**
//!    (`concurrent_engine_construction_does_not_deadlock`): many threads each
//!    build their own `OcrEngine` (each owns one asupersync runtime) and drive a
//!    sequential sub-batch — racing runtime construction + the global weak model
//!    cache in `native_engine`, again bounded by the wall-clock watchdog.
//!
//! ## Logging
//!
//! Detailed logging is a first-class requirement here. Every scenario emits
//! structured NDJSON lines (conforming to `tests/fixtures/test_log_schema.json`:
//! `schema_version`/`ts`/`test`/`case`/`run_seq`/`event`/`result` + scenario
//! fields) on what it exercised, the inputs, the pool size detected, the pages
//! issued, the completion count, the elapsed time, and the explicit timeout
//! budget. Model-gated skips emit a `result:"skip_no_model"` SUCCESS line
//! explaining why, with the `native_path_ran` / `fallback_target` proof fields.
//!
//! ## API assumed (for batch-verify reconciliation)
//!
//! ```ignore
//! franken_ocr::OcrEngine::new() -> franken_ocr::FocrResult<OcrEngine>
//! franken_ocr::OcrEngine::recognize(&self, image_path: &Path) -> FocrResult<String>
//! franken_ocr::OcrEngine::recognize_with_model(&self, model_path: &Path, image_path: &Path) -> FocrResult<String>
//! franken_ocr::OcrEngine::model_path() -> std::path::PathBuf
//! franken_ocr::FocrError::ModelNotFound(String)  // exit code 3
//! franken_ocr::MODEL_PATH_ENV: &str  // "FOCR_MODEL_PATH"
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use franken_ocr::{FocrError, MODEL_PATH_ENV, OcrEngine};

// ───────────────────────────── tunables ─────────────────────────────

/// Test name carried in every NDJSON line (schema common field `test`).
const TEST: &str = "many_pages_without_deadlock";

/// Robot-log schema version. Mirrors `tests/fixtures/test_log_schema.json`.
const LOG_SCHEMA_VERSION: u32 = 1;

/// A guaranteed-absent model artifact. The fallback target MUST be `/nonexistent`
/// per the harness contract (`native_path_proof.required_fallback` in the schema
/// fixture) so the model-gated skip provably went down the native resolution
/// path and returned a clean `ModelNotFound` rather than silently no-op'ing.
const ABSENT_MODEL: &str = "/nonexistent/franken_ocr/model.focrq";

/// A synthetic per-page document path. Never read in the no-weights path (the
/// model resolves to `ModelNotFound` before any image decode), so it need not
/// exist on disk.
const SYNTHETIC_PAGE: &str = "/nonexistent/franken_ocr/page.png";

/// How many pages over the detected pool size to drive. Chosen so `pages ≫
/// pool`: even on a 32-physical-core host this is comfortably more pages than
/// cores (the regression the watchdog catches starves once pages exceed the
/// pool, so the multiplier is what matters, not the absolute count).
const PAGES_MULTIPLIER: usize = 24;

/// A floor on pages so the watchdog still has teeth on a reported 1-core host.
const MIN_PAGES: usize = 256;

/// Threads that concurrently build engines in scenario 2. Deliberately exceeds
/// a typical physical core count to stress runtime-construction races.
const CONSTRUCTION_THREADS: usize = 16;

/// Pages each construction thread drives sequentially through its own engine.
const PAGES_PER_CONSTRUCTION_THREAD: usize = 24;

/// Hard wall-clock budget for a whole batch. This is the watchdog deadline:
/// the no-weights batch is thousands of fast `ModelNotFound` returns (each
/// microseconds), so a healthy run finishes far inside this; only a genuine
/// hang (nested runtime / rayon-under-lock / non-sequential fan-out) exhausts
/// it. Generous enough to never flake on a loaded CI box, tight enough that a
/// true deadlock is caught in well under a CI job timeout.
const WATCHDOG_BUDGET: Duration = Duration::from_secs(120);

/// When the heavy (weights-present) branch is enabled, give the real forward a
/// far larger budget — a full 6.67 GB-model multi-page decode is slow. Still a
/// hard ceiling: a deadlock in the real forward must not hang CI indefinitely.
const WATCHDOG_BUDGET_HEAVY: Duration = Duration::from_secs(1800);

// ───────────────────────────── logging ─────────────────────────────

/// Monotonic run-sequence counter so every emitted line has a unique `run_seq`
/// (schema common field), making interleaved scenario output reorderable.
static RUN_SEQ: AtomicUsize = AtomicUsize::new(0);

fn next_seq() -> usize {
    RUN_SEQ.fetch_add(1, Ordering::Relaxed)
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Emit one structured NDJSON log line to stderr (so it never contaminates any
/// stdout data channel and always shows under `cargo test -- --nocapture`).
///
/// `case` is the scenario name; `event`/`result` are the schema enums; `fields`
/// is the already-serialized body of scenario-specific key/values (no leading or
/// trailing comma, e.g. `r#""pool":8,"pages_issued":256"#`).
fn log_line(case: &str, event: &str, result: &str, fields: &str) {
    let head = format!(
        r#"{{"schema_version":{LOG_SCHEMA_VERSION},"ts":{ts},"test":"{TEST}","case":"{case}","run_seq":{seq},"event":"{event}","result":"{result}""#,
        ts = now_millis(),
        seq = next_seq(),
    );
    if fields.is_empty() {
        eprintln!("{head}}}");
    } else {
        eprintln!("{head},{fields}}}");
    }
}

// ───────────────────────────── watchdog core ─────────────────────────────

/// Outcome of a watchdogged batch: either the worker returned its
/// completion-count result inside the budget, or the budget elapsed first.
enum Watch<T> {
    /// Batch finished in time, carrying the worker's payload and its own
    /// wall-clock elapsed (measured inside the worker, before the channel send).
    Finished { payload: T, elapsed: Duration },
    /// The budget elapsed without the worker reporting — DEADLOCK SUSPECTED.
    TimedOut,
}

/// Run `work` on a dedicated worker thread and block the test thread on a
/// `recv_timeout(budget)`. The worker sends `(payload, elapsed)` exactly once on
/// completion. If the deadline fires first we return `TimedOut` WITHOUT joining
/// the worker (a deadlocked worker would never join — joining it would hang the
/// very test that is meant to report the hang). The leaked worker is acceptable:
/// the test is failing the run anyway and the process is about to abort.
fn run_with_watchdog<T, F>(budget: Duration, work: F) -> Watch<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<(T, Duration)>();
    // Detached worker: we intentionally do not retain/join the handle so a hung
    // worker cannot block the watchdog's own timeout path.
    let _worker = thread::Builder::new()
        .name(format!("{TEST}-worker"))
        .spawn(move || {
            let started = Instant::now();
            let payload = work();
            let elapsed = started.elapsed();
            // If the receiver already gave up (timed out) this send fails; that
            // is fine — we are on the failure path and about to abort.
            let _ = tx.send((payload, elapsed));
        })
        .expect("spawn watchdog worker thread");

    match rx.recv_timeout(budget) {
        Ok((payload, elapsed)) => Watch::Finished { payload, elapsed },
        Err(mpsc::RecvTimeoutError::Timeout) => Watch::TimedOut,
        // The worker panicked and dropped the sender without sending. Surface it
        // as a timeout-style failure so the caller reports a diagnosable line
        // (a panic in the worker is itself a watchdog-worthy regression).
        Err(mpsc::RecvTimeoutError::Disconnected) => Watch::TimedOut,
    }
}

// ───────────────────────────── helpers ─────────────────────────────

/// Detected concurrency width. Doctrine #5 pins the kernel rayon pool to
/// *physical* cores; `std::thread::available_parallelism` reports the logical
/// width the platform will give us, which is the closest stable stdlib proxy and
/// — being ≥ physical — only makes "pages ≫ pool" a *stronger* condition. This
/// is the same source `robot backends` uses for `logical_cpus` (src/cli.rs), so
/// the watchdog and the diagnostics agree on the number.
fn detected_pool() -> usize {
    thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Pages to issue: `pages ≫ pool`, floored at `MIN_PAGES`.
fn pages_for(pool: usize) -> usize {
    (pool * PAGES_MULTIPLIER).max(MIN_PAGES)
}

/// Resolve whether the heavy (weights-present) branch is enabled, and to what
/// artifact. We only take the heavy branch when `FOCR_MODEL_PATH` is set AND the
/// artifact exists on disk — otherwise we run the no-weights orchestration
/// branch and skip-with-SUCCESS for the heavy path (same contract as the
/// model-gated e2e tests).
fn heavy_model_path() -> Option<PathBuf> {
    let p = std::env::var_os(MODEL_PATH_ENV).map(PathBuf::from)?;
    if p.exists() { Some(p) } else { None }
}

// ───────────────────────────── scenario 1 ─────────────────────────────

/// THE watchdog: one engine, a **sequential** outer loop over pages ≫ pool,
/// asserting the whole batch completes within the hard wall-clock budget.
#[test]
fn many_sequential_pages_complete_within_budget() {
    let case = "sequential_outer_loop";
    let pool = detected_pool();
    let pages = pages_for(pool);

    // Heavy vs no-weights branch selection, logged either way.
    let heavy = heavy_model_path();
    let (model_path, budget, native_path_ran, fallback_target) = match &heavy {
        Some(p) => (p.clone(), WATCHDOG_BUDGET_HEAVY, true, p.display().to_string()),
        None => (
            PathBuf::from(ABSENT_MODEL),
            WATCHDOG_BUDGET,
            true,
            ABSENT_MODEL.to_string(),
        ),
    };

    log_line(
        case,
        "setup",
        "pass",
        &format!(
            r#""seed":0,"pool":{pool},"pages_issued":{pages},"timeout_budget_secs":{budget},"mode":"{mode}","model_path":{mp},"pages_multiplier":{PAGES_MULTIPLIER},"native_path_ran":{native_path_ran},"fallback_target":"{fallback_target}""#,
            budget = budget.as_secs(),
            mode = if heavy.is_some() { "heavy_forward" } else { "no_weights_orchestration" },
            mp = json_str(&model_path.display().to_string()),
        ),
    );

    // The batch body, moved onto the watchdog worker. Returns a structured
    // completion report. The outer loop is strictly SEQUENTIAL (doctrine #5):
    // one `recognize` returns fully before the next begins, on a single engine
    // that owns exactly one asupersync runtime.
    let model_path_for_worker = model_path.clone();
    let heavy_branch = heavy.is_some();
    let outcome = run_with_watchdog(budget, move || {
        let engine = OcrEngine::new().expect("OcrEngine::new builds its single owned runtime");
        let page = Path::new(SYNTHETIC_PAGE);

        let mut completed = 0usize;
        let mut model_not_found = 0usize;
        let mut other_terminal = 0usize;
        let mut ok_results = 0usize;

        for _ in 0..pages {
            match engine.recognize_with_model(&model_path_for_worker, page) {
                Ok(_text) => {
                    // Only reachable on the heavy branch once weights + the full
                    // forward land. The point of the watchdog is that the loop
                    // PROGRESSES, whatever each call returns.
                    ok_results += 1;
                }
                Err(FocrError::ModelNotFound(_)) => {
                    model_not_found += 1;
                }
                Err(_other) => {
                    // The pipeline is wired but some stages still surface
                    // NotImplemented once a path resolves; that is still
                    // forward progress for the deadlock watchdog's purposes.
                    other_terminal += 1;
                }
            }
            completed += 1;
        }
        BatchReport {
            completed,
            ok_results,
            model_not_found,
            other_terminal,
        }
    });

    match outcome {
        Watch::TimedOut => {
            // Self-diagnosing failure: dump everything an investigator needs.
            log_line(
                case,
                "error",
                "fail",
                &format!(
                    r#""diag":{{"error_kind":"DEADLOCK_SUSPECTED","focr_exit_code":5,"message":"batch did not complete within watchdog budget"}},"pool":{pool},"pages_issued":{pages},"timeout_budget_secs":{}"#,
                    budget.as_secs(),
                ),
            );
            panic!(
                "DEADLOCK SUSPECTED: the sequential {pages}-page batch (pool={pool}, \
                 pages≫pool by {PAGES_MULTIPLIER}×) did NOT complete within the \
                 {budget}s wall-clock watchdog budget. Doctrine #5 regression: a \
                 nested asupersync runtime, rayon nested under a held lock, or a \
                 non-sequential outer page loop will starve/hang exactly here. \
                 model_path={mp}",
                budget = budget.as_secs(),
                mp = model_path.display(),
            );
        }
        Watch::Finished { payload, elapsed } => {
            let BatchReport {
                completed,
                ok_results,
                model_not_found,
                other_terminal,
            } = payload;

            // Hard assertion #1: every issued page was driven to a terminal
            // result. A silent early break would itself be a defect.
            assert_eq!(
                completed, pages,
                "expected all {pages} pages driven to completion, got {completed}"
            );

            // Hard assertion #2: it finished inside the budget (the watchdog
            // already guarantees this, but we re-assert on the measured elapsed
            // so the message carries the number).
            assert!(
                elapsed < budget,
                "batch elapsed {elapsed:?} must be < watchdog budget {budget:?}"
            );

            log_line(
                case,
                "result",
                "pass",
                &format!(
                    r#""elapsed_us":{elapsed_us},"pool":{pool},"pages_issued":{pages},"completed":{completed},"ok_results":{ok_results},"model_not_found":{model_not_found},"other_terminal":{other_terminal},"timeout_budget_secs":{budget_s},"slack_secs":{slack},"per_page_avg_us":{per_page}"#,
                    elapsed_us = elapsed.as_micros(),
                    budget_s = budget.as_secs(),
                    slack = budget.as_secs().saturating_sub(elapsed.as_secs()),
                    per_page = elapsed.as_micros() / (completed.max(1) as u128),
                ),
            );

            if heavy_branch {
                // Heavy branch: at least one full forward must have produced
                // output (or the run is not actually exercising weights).
                assert!(
                    ok_results > 0 || other_terminal > 0,
                    "heavy branch (FOCR_MODEL_PATH set) drove {pages} pages but produced \
                     no Ok and no post-resolution error — model path resolved to \
                     ModelNotFound {model_not_found} times, which means the artifact \
                     was not actually loaded"
                );
                log_line(
                    case,
                    "assert",
                    "pass",
                    &format!(
                        r#""assertion":"heavy_forward_progressed","pass":true,"ok_results":{ok_results},"other_terminal":{other_terminal}"#
                    ),
                );
            } else {
                // No-weights branch: every page should have returned the clean
                // fast `ModelNotFound`. This proves we exercised the native
                // resolution + per-call `block_on` orchestration path (which the
                // deadlock rule guards) `pages` times without a hang — AND emits
                // the skip-with-SUCCESS line for the heavy forward we couldn't run.
                assert_eq!(
                    model_not_found, pages,
                    "no-weights branch: expected all {pages} calls to return \
                     ModelNotFound (clean fast error), got {model_not_found}"
                );
                log_line(
                    case,
                    "skip",
                    "skip_no_model",
                    &format!(
                        r#""reason":"FOCR_MODEL_PATH unset or artifact absent — exercised the sequential outer-loop + per-call block_on orchestration path that doctrine #5 protects ({pages} clean ModelNotFound returns under the watchdog); the full forward branch is gated and will run once the 6.67 GB weights land","native_path_ran":true,"fallback_target":"{ABSENT_MODEL}","pages_proved":{pages}"#
                    ),
                );
            }
        }
    }
}

// ───────────────────────────── scenario 2 ─────────────────────────────

/// Engine-construction-under-concurrency: many threads each build their OWN
/// `OcrEngine` (each owning one asupersync runtime) and drive a sequential
/// sub-batch, racing runtime construction and the global weak model cache in
/// `native_engine`. The whole thing is bounded by the same wall-clock watchdog —
/// a regression that serializes engine construction behind a contended lock, or
/// deadlocks the weak cache, hangs here.
#[test]
fn concurrent_engine_construction_does_not_deadlock() {
    let case = "engine_construction_under_concurrency";
    let pool = detected_pool();
    let total_pages = CONSTRUCTION_THREADS * PAGES_PER_CONSTRUCTION_THREAD;

    log_line(
        case,
        "setup",
        "pass",
        &format!(
            r#""seed":0,"pool":{pool},"construction_threads":{CONSTRUCTION_THREADS},"pages_per_thread":{PAGES_PER_CONSTRUCTION_THREAD},"total_pages":{total_pages},"timeout_budget_secs":{budget},"model_path":"{ABSENT_MODEL}","native_path_ran":true,"fallback_target":"{ABSENT_MODEL}""#,
            budget = WATCHDOG_BUDGET.as_secs(),
        ),
    );

    let outcome = run_with_watchdog(WATCHDOG_BUDGET, move || {
        // Each thread constructs its own engine concurrently and runs a
        // sequential sub-batch. We join all of them inside the worker so the
        // watchdog sees a single completion event.
        let handles: Vec<_> = (0..CONSTRUCTION_THREADS)
            .map(|tid| {
                thread::Builder::new()
                    .name(format!("{TEST}-ctor-{tid}"))
                    .spawn(move || {
                        let engine =
                            OcrEngine::new().expect("each thread builds its own engine runtime");
                        let model = Path::new(ABSENT_MODEL);
                        let page = Path::new(SYNTHETIC_PAGE);
                        let mut not_found = 0usize;
                        for _ in 0..PAGES_PER_CONSTRUCTION_THREAD {
                            // Any non-`ModelNotFound` terminal result is still
                            // forward progress; the watchdog only cares that the
                            // loop returns, but counting `ModelNotFound` lets the
                            // no-weights assertion prove the native path resolved.
                            if let Err(FocrError::ModelNotFound(_)) =
                                engine.recognize_with_model(model, page)
                            {
                                not_found += 1;
                            }
                        }
                        not_found
                    })
                    .expect("spawn ctor thread")
            })
            .collect();

        let mut total_not_found = 0usize;
        let mut joined = 0usize;
        for h in handles {
            total_not_found += h.join().expect("ctor thread panicked");
            joined += 1;
        }
        (joined, total_not_found)
    });

    match outcome {
        Watch::TimedOut => {
            log_line(
                case,
                "error",
                "fail",
                &format!(
                    r#""diag":{{"error_kind":"DEADLOCK_SUSPECTED","focr_exit_code":5,"message":"concurrent engine construction + sequential sub-batches did not complete within watchdog budget"}},"pool":{pool},"construction_threads":{CONSTRUCTION_THREADS},"total_pages":{total_pages},"timeout_budget_secs":{}"#,
                    WATCHDOG_BUDGET.as_secs(),
                ),
            );
            panic!(
                "DEADLOCK SUSPECTED: {CONSTRUCTION_THREADS} threads each constructing \
                 an OcrEngine and driving {PAGES_PER_CONSTRUCTION_THREAD} sequential \
                 pages ({total_pages} total, pool={pool}) did NOT complete within the \
                 {budget}s watchdog budget. A contended lock around runtime/weak-cache \
                 construction, or a second nested runtime, hangs exactly here.",
                budget = WATCHDOG_BUDGET.as_secs(),
            );
        }
        Watch::Finished {
            payload: (joined, total_not_found),
            elapsed,
        } => {
            assert_eq!(
                joined, CONSTRUCTION_THREADS,
                "expected all {CONSTRUCTION_THREADS} construction threads to join"
            );
            assert_eq!(
                total_not_found, total_pages,
                "every page across every thread should have cleanly resolved to \
                 ModelNotFound (got {total_not_found}/{total_pages})"
            );
            log_line(
                case,
                "result",
                "pass",
                &format!(
                    r#""elapsed_us":{elapsed_us},"pool":{pool},"construction_threads":{CONSTRUCTION_THREADS},"total_pages":{total_pages},"model_not_found":{total_not_found},"timeout_budget_secs":{budget_s},"slack_secs":{slack}"#,
                    elapsed_us = elapsed.as_micros(),
                    budget_s = WATCHDOG_BUDGET.as_secs(),
                    slack = WATCHDOG_BUDGET.as_secs().saturating_sub(elapsed.as_secs()),
                ),
            );
            log_line(
                case,
                "skip",
                "skip_no_model",
                &format!(
                    r#""reason":"no weights — proved concurrent engine construction + the global weak model cache de-dup path do not deadlock under {total_pages} sequential page drives across {CONSTRUCTION_THREADS} threads; full forward gated on the 6.67 GB weights","native_path_ran":true,"fallback_target":"{ABSENT_MODEL}""#
                ),
            );
        }
    }
}

// ───────────────────────────── scenario 3 ─────────────────────────────

/// Sanity: the watchdog itself FIRES on a genuine hang. A watchdog that can't
/// detect a deadlock is worthless, so we feed `run_with_watchdog` a worker that
/// blocks forever and assert it returns `TimedOut` inside a tiny budget. This
/// keeps the watchdog honest without ever risking the real test hanging (the
/// blocked worker is detached and the process tears it down at exit).
#[test]
fn watchdog_detects_a_real_hang() {
    let case = "watchdog_self_check";
    let tiny = Duration::from_millis(250);
    log_line(
        case,
        "setup",
        "pass",
        &format!(r#""seed":0,"timeout_budget_secs":0,"budget_ms":{}"#, tiny.as_millis()),
    );

    let started = Instant::now();
    // Annotate the worker's return type explicitly (`usize`): the body diverges
    // (`loop { park }` has type `!`), which coerces to `usize` so the watchdog's
    // generic `T` is pinned without a fragile unreachable-expression attribute
    // and without an explicit `-> ()` (which clippy flags as a unit return).
    let outcome = run_with_watchdog(tiny, || -> usize {
        // Park forever — this models a deadlocked forward.
        loop {
            thread::park();
        }
    });
    let observed = started.elapsed();

    match outcome {
        Watch::TimedOut => {
            // The watchdog must not return appreciably before its budget (it
            // should block for ~`tiny`, then fire).
            assert!(
                observed >= tiny,
                "watchdog fired too early: observed {observed:?} < budget {tiny:?}"
            );
            log_line(
                case,
                "assert",
                "pass",
                &format!(
                    r#""assertion":"watchdog_fires_on_hang","pass":true,"observed_ms":{}"#,
                    observed.as_millis(),
                ),
            );
            log_line(
                case,
                "result",
                "pass",
                &format!(r#""elapsed_us":{}"#, observed.as_micros()),
            );
        }
        Watch::Finished { .. } => {
            log_line(
                case,
                "error",
                "fail",
                r#""diag":{"error_kind":"WATCHDOG_DID_NOT_FIRE","focr_exit_code":1,"message":"a forever-blocked worker did not trip the watchdog"}"#,
            );
            panic!(
                "watchdog self-check FAILED: a worker that blocks forever should have \
                 produced Watch::TimedOut, but the watchdog reported Finished. The \
                 deadlock watchdog is broken and would NOT catch a real doctrine #5 \
                 regression."
            );
        }
    }
}

// ───────────────────────────── small utilities ─────────────────────────────

/// The structured completion report from a watchdogged batch.
struct BatchReport {
    completed: usize,
    ok_results: usize,
    model_not_found: usize,
    other_terminal: usize,
}

/// Minimal JSON string escaper for the handful of values (paths) we embed. We
/// hand-roll it rather than pull `serde_json` into the embedded-fields path so a
/// path with a quote or backslash can't produce a malformed log line. (The rest
/// of the harness already depends on `serde_json`, but the deps note records
/// that we deliberately avoided a heavier helper here.)
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
