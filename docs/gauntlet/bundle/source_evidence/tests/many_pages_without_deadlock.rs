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
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use franken_ocr::{
    FocrError, MODEL_PATH_ENV, OcrEngine, kernel_pool_width, stream_pages, thread_budget,
};

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
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
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
        Some(p) => (
            p.clone(),
            WATCHDOG_BUDGET_HEAVY,
            true,
            p.display().to_string(),
        ),
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
            mode = if heavy.is_some() {
                "heavy_forward"
            } else {
                "no_weights_orchestration"
            },
            mp = json_str(&model_path.display().to_string()),
        ),
    );

    // The batch body, moved onto the watchdog worker. Returns a structured
    // completion report. The outer loop is strictly SEQUENTIAL (doctrine #5):
    // one `recognize` returns fully before the next begins, on a single engine
    // that owns exactly one asupersync runtime.
    let model_path_for_worker = model_path.clone();
    let heavy_branch = heavy.is_some();
    // Heavy branch page selection (fresh-eyes fix): the synthetic /nonexistent
    // page decode-errors before any forward, so a heavy run against it proved
    // only orchestration while claiming full-forward coverage. With
    // `FOCR_WATCHDOG_IMAGE` set (the spine scenario's contract) the heavy loop
    // drives REAL forwards; without it the heavy branch stays an orchestration
    // proof and says so in its log line.
    let heavy_page = std::env::var_os("FOCR_WATCHDOG_IMAGE")
        .map(PathBuf::from)
        .filter(|p| heavy_branch && p.exists());
    let heavy_forwards_real = heavy_page.is_some();
    let outcome = run_with_watchdog(budget, move || {
        let engine = OcrEngine::new().expect("OcrEngine::new builds its single owned runtime");
        let page_buf = heavy_page.unwrap_or_else(|| PathBuf::from(SYNTHETIC_PAGE));
        let page = page_buf.as_path();

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
                if heavy_forwards_real {
                    // A real decodable page was supplied: full forwards must
                    // SUCCEED — an input-decode error would previously count
                    // as "progress" here and fake full-forward coverage.
                    assert!(
                        ok_results > 0,
                        "heavy branch drove {pages} real pages (FOCR_WATCHDOG_IMAGE) but \
                         produced no Ok result (other_terminal={other_terminal}, \
                         model_not_found={model_not_found})"
                    );
                } else {
                    // Weights present but no real page: the loop progressed
                    // through model-load + input-decode orchestration only.
                    // Say so honestly instead of claiming forward coverage.
                    assert!(
                        ok_results > 0 || other_terminal > 0,
                        "heavy branch (FOCR_MODEL_PATH set) drove {pages} pages but produced \
                         no Ok and no post-resolution error — model path resolved to \
                         ModelNotFound {model_not_found} times, which means the artifact \
                         was not actually loaded"
                    );
                }
                log_line(
                    case,
                    "assert",
                    "pass",
                    &format!(
                        r#""assertion":"heavy_forward_progressed","pass":true,"ok_results":{ok_results},"other_terminal":{other_terminal},"real_forwards":{heavy_forwards_real}"#
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

// ───────────────────────────── scenario 1b: the batched spine ─────────────────────────────

/// bd-1azu.14: the SPINE watchdog. Drives pages ≫ pool through the
/// continuous-batch scheduler (`recognize_batch` with `FOCR_BATCH_SPINE` armed
/// by the caller — env is process-immutable here, so the B sweep + spine=0
/// control run live in `scripts/spine_watchdog_sweep.sh`) and asserts, beyond
/// completion-within-budget: (a) the PROCESS-WIDE one-live-forward gauge never
/// exceeded 1 — and it counts vision-encode (per-page AND batched) + page
/// prefill + every scheduler decode step, so a stray concurrent vision fan-out
/// fails the gate; (b) no forward began while the model-cache mutex guard was
/// held (the rayon-under-lock deadlock class).
///
/// Model-gated skip-with-SUCCESS: needs `FOCR_MODEL_PATH` (real weights),
/// `FOCR_BATCH_SPINE` (armed externally), and `FOCR_WATCHDOG_IMAGE` (a real
/// decodable page); absent any of them it logs `skip_no_model` and passes.
#[test]
fn spine_many_pages_one_live_forward_within_budget() {
    let case = "batched_spine_one_live_forward";
    let pool = detected_pool();

    let heavy = heavy_model_path();
    // Value-parsed to MATCH the engine's kill-switch semantics (`=0` disables):
    // the scenario arms only when the spine would actually engage.
    let spine_armed = std::env::var("FOCR_BATCH_SPINE").is_ok_and(|v| {
        !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "off" | "false" | "no"
        )
    });
    let image = std::env::var_os("FOCR_WATCHDOG_IMAGE")
        .map(PathBuf::from)
        .filter(|p| p.exists());
    let batch_size = std::env::var("FOCR_BATCH_SIZE").unwrap_or_else(|_| "default".into());

    let (Some(model_path), true, Some(image)) = (heavy, spine_armed, image) else {
        log_line(
            case,
            "skip",
            "skip_no_model",
            &format!(
                r#""reason":"needs FOCR_MODEL_PATH + FOCR_BATCH_SPINE + FOCR_WATCHDOG_IMAGE (env is process-immutable in-test; scripts/spine_watchdog_sweep.sh arms all three and sweeps FOCR_BATCH_SIZE)","native_path_ran":true,"fallback_target":"{ABSENT_MODEL}""#
            ),
        );
        return;
    };

    // Full real forwards are expensive (~10s+/page): pages ≫ pool is expressed
    // against the SCHEDULER's admission width (B, from FOCR_BATCH_SIZE), not the
    // rayon pool — the sweep script runs B ∈ {1, 4, big} so streams retire and
    // backfill many times at B ≪ pages. 12 pages keeps the heavy run inside the
    // budget while still churning admission at every swept B.
    let pages = 12usize;

    log_line(
        case,
        "setup",
        "pass",
        &format!(
            r#""seed":0,"pool":{pool},"pages_issued":{pages},"batch_size_env":{bs},"timeout_budget_secs":{budget},"mode":"heavy_spine","model_path":{mp},"image":{img},"native_path_ran":true,"fallback_target":{mp}"#,
            budget = WATCHDOG_BUDGET_HEAVY.as_secs(),
            bs = json_str(&batch_size),
            mp = json_str(&model_path.display().to_string()),
            img = json_str(&image.display().to_string()),
        ),
    );

    // Reset the gauge so this scenario observes only its own forwards.
    let _ = franken_ocr::native_engine::forward_gauge_take();

    let outcome = run_with_watchdog(WATCHDOG_BUDGET_HEAVY, move || {
        let engine = OcrEngine::new().expect("OcrEngine::new builds its single owned runtime");
        let page_paths: Vec<PathBuf> = (0..pages).map(|_| image.clone()).collect();
        let page_refs: Vec<&Path> = page_paths.iter().map(PathBuf::as_path).collect();
        let results = engine
            .recognize_batch_with_model(&model_path, &page_refs)
            .expect("recognize_batch runs");
        let ok = results.iter().filter(|r| r.is_ok()).count();
        let err = results.len() - ok;
        (results.len(), ok, err)
    });

    match outcome {
        Watch::TimedOut => {
            log_line(
                case,
                "watchdog",
                "fail",
                &format!(
                    r#""assertion":"completes_within_budget","pass":false,"pool":{pool},"pages_issued":{pages},"timeout_budget_secs":{}"#,
                    WATCHDOG_BUDGET_HEAVY.as_secs()
                ),
            );
            panic!(
                "DEADLOCK SUSPECTED: spine batch (pages={pages}, pool={pool}, \
                 FOCR_BATCH_SIZE={batch_size}) did not complete within {}s",
                WATCHDOG_BUDGET_HEAVY.as_secs()
            );
        }
        Watch::Finished {
            payload: (completed, ok, err),
            elapsed,
        } => {
            let (max_forwards, under_guard) = franken_ocr::native_engine::forward_gauge_take();
            log_line(
                case,
                "complete",
                "pass",
                &format!(
                    r#""completed":{completed},"ok":{ok},"err":{err},"elapsed_ms":{},"max_concurrent_forwards":{max_forwards},"guard_held_during_fanout":{under_guard},"batch_size_env":{bs}"#,
                    elapsed.as_millis(),
                    bs = json_str(&batch_size),
                ),
            );
            assert_eq!(completed, pages, "every page must produce a result slot");
            assert_eq!(
                err, 0,
                "heavy spine run: every page should decode (got {err} errors)"
            );
            // EXACTLY 1, not <= 1: pages completed, so forwards MUST have run — a max
            // of 0 means the instrumentation got disconnected (a refactor that
            // silently stops calling enter_forward would otherwise fake-pass).
            assert_eq!(
                max_forwards, 1,
                "ONE-LIVE-FORWARD VIOLATION: the process-wide gauge saw \
                 {max_forwards} concurrent forwards (vision/prefill/decode-step); \
                 1 is the only healthy value once pages completed (0 = gauge \
                 disconnected, >1 = concurrency violation)"
            );
            assert!(
                !under_guard,
                "LOCK-DISCIPLINE VIOLATION: a forward began while the \
                 model-cache mutex guard was held (rayon-under-lock class)"
            );
        }
    }
}

// ───────────────────────────── scenario 1c: the zoo lanes ─────────────────────────────

/// One zoo model lane the A11 watchdog drives.
struct ZooLane {
    /// Scenario-case suffix + log label.
    name: &'static str,
    /// The lane's arming env var, pointing at its zoo model dir.
    dir_env: &'static str,
    /// The int8 artifact filename inside that dir.
    artifact: &'static str,
    /// A COMMITTED, really-decodable fixture image (repo-relative) — the same
    /// per-model sample the lane's own certs use, so a heavy run drives REAL
    /// full forwards, never an input-decode error masquerading as coverage.
    image: &'static str,
}

const ZOO_LANES: [ZooLane; 3] = [
    ZooLane {
        name: "got_ocr2",
        dir_env: "FOCR_GOT_DIR",
        artifact: "got-ocr2.int8.focrq",
        image: "tests/fixtures/got/sample_text.png",
    },
    ZooLane {
        name: "smolvlm2",
        dir_env: "FOCR_SMOLVLM2_DIR",
        artifact: "smolvlm2.int8.focrq",
        image: "tests/fixtures/smolvlm2/sample_photo.png",
    },
    ZooLane {
        name: "onechart",
        dir_env: "FOCR_ONECHART_DIR",
        artifact: "onechart.int8.focrq",
        image: "tests/fixtures/onechart/sample_chart.png",
    },
];

/// Sequential real forwards per armed zoo lane. Unlike the no-weights batch
/// (microsecond `ModelNotFound` returns, so pages ≫ pool is nearly free), a
/// zoo forward costs seconds-to-tens-of-seconds, so the base multiplier would
/// blow any honest budget. The doctrine-#5 regressions this scenario guards —
/// a nested runtime, rayon under a held lock, a gauge violation — manifest on
/// the FIRST forward or on the first weak-cache REUSE call; several sequential
/// forwards through one engine cover both plus steady-state repetition.
const ZOO_PAGES_PER_LANE: usize = 6;

/// Interleave rounds for the cross-lane cache-swap pass (each round visits
/// every armed lane once through ONE shared engine).
const ZOO_INTERLEAVE_ROUNDS: usize = 2;

/// Resolve a lane's artifact + image, if armed. `None` = skip-with-SUCCESS.
fn zoo_lane_paths(lane: &ZooLane) -> Option<(PathBuf, PathBuf)> {
    let dir = std::env::var_os(lane.dir_env).map(PathBuf::from)?;
    let model = dir.join(lane.artifact);
    let image = Path::new(env!("CARGO_MANIFEST_DIR")).join(lane.image);
    (model.is_file() && image.is_file()).then_some((model, image))
}

/// Drive `pages` items through `engine.recognize_with_model` sequentially,
/// counting terminal outcomes (the shared batch body of both zoo passes).
fn drive_zoo_batch(engine: &OcrEngine, items: &[(PathBuf, PathBuf)]) -> BatchReport {
    let mut report = BatchReport {
        completed: 0,
        ok_results: 0,
        model_not_found: 0,
        other_terminal: 0,
    };
    for (model, image) in items {
        match engine.recognize_with_model(model, image) {
            Ok(_) => report.ok_results += 1,
            Err(FocrError::ModelNotFound(_)) => report.model_not_found += 1,
            Err(_) => report.other_terminal += 1,
        }
        report.completed += 1;
    }
    report
}

/// bd-3jo6.1.11 (A11): the ZOO watchdog. Doctrine #5 applies to every model
/// lane, not just the base model — each zoo forward (GOT / SmolVLM2 /
/// OneChart) fans out through the SAME kernel rayon pool under the same
/// one-live-forward discipline, and sequential calls share the weak model
/// cache and per-model tokenizer state. Two passes, both watchdog-bounded:
///
/// 1. **Per-lane**: one engine, `ZOO_PAGES_PER_LANE` sequential REAL forwards
///    over the lane's committed cert image. Every call must return `Ok` — the
///    artifact and image are both known-good, so any error is a real defect,
///    not "progress".
/// 2. **Cross-lane interleave**: ONE engine, the armed lanes visited
///    round-robin. Each adjacent pair forces a model swap through the weak
///    cache (drop + reload + tokenizer switch) — the multi-model risk surface
///    the per-lane pass cannot reach.
///
/// Heavy-only, model-gated skip-with-SUCCESS per lane: the no-weights fast
/// path CANNOT reach zoo dispatch (`arch()` is read from the loaded artifact,
/// so an absent model is `ModelNotFound` before any zoo code runs) and is
/// already proved by `many_sequential_pages_complete_within_budget`; the skip
/// line says so instead of claiming coverage. Arming contract (env is
/// process-immutable in-test, so the caller sets these): `FOCR_GOT_DIR` /
/// `FOCR_SMOLVLM2_DIR` / `FOCR_ONECHART_DIR`, plus optionally
/// `FOCR_MAX_NEW_TOKENS` to bound per-forward decode cost (a capped run's
/// tokens are a true prefix — it never alters per-step math).
#[test]
fn zoo_models_sequential_pages_complete_within_budget() {
    let pool = detected_pool();
    let mut armed: Vec<(&ZooLane, PathBuf, PathBuf)> = Vec::new();

    for lane in &ZOO_LANES {
        let case = format!("zoo_sequential_{}", lane.name);
        match zoo_lane_paths(lane) {
            None => {
                log_line(
                    &case,
                    "skip",
                    "skip_no_model",
                    &format!(
                        r#""reason":"{env} unset or {artifact}/fixture image absent — the zoo dispatch is unreachable without the loaded artifact (arch() comes from the weights), and the no-weights orchestration path is already proved by many_sequential_pages_complete_within_budget","native_path_ran":true,"fallback_target":"{ABSENT_MODEL}""#,
                        env = lane.dir_env,
                        artifact = lane.artifact,
                    ),
                );
                continue;
            }
            Some((model, image)) => {
                log_line(
                    &case,
                    "setup",
                    "pass",
                    &format!(
                        r#""seed":0,"pool":{pool},"pages_issued":{ZOO_PAGES_PER_LANE},"timeout_budget_secs":{budget},"mode":"heavy_forward","model_path":{mp},"image_path":{ip}"#,
                        budget = WATCHDOG_BUDGET_HEAVY.as_secs(),
                        mp = json_str(&model.display().to_string()),
                        ip = json_str(&image.display().to_string()),
                    ),
                );

                let items: Vec<(PathBuf, PathBuf)> = (0..ZOO_PAGES_PER_LANE)
                    .map(|_| (model.clone(), image.clone()))
                    .collect();
                let outcome = run_with_watchdog(WATCHDOG_BUDGET_HEAVY, move || {
                    let engine =
                        OcrEngine::new().expect("OcrEngine::new builds its single owned runtime");
                    drive_zoo_batch(&engine, &items)
                });
                report_zoo_outcome(&case, lane.name, pool, ZOO_PAGES_PER_LANE, outcome);
                armed.push((lane, model, image));
            }
        }
    }

    // Cross-lane interleave: only meaningful with ≥ 2 armed lanes (one lane
    // has no swap seam; zero lanes was fully skip-logged above).
    let case = "zoo_interleaved_cache_swap";
    if armed.len() < 2 {
        log_line(
            case,
            "skip",
            "skip_no_model",
            &format!(
                r#""reason":"needs >= 2 armed zoo lanes for a model-swap seam (armed: {})","native_path_ran":true,"fallback_target":"{ABSENT_MODEL}""#,
                armed.len(),
            ),
        );
        return;
    }

    let items: Vec<(PathBuf, PathBuf)> = (0..ZOO_INTERLEAVE_ROUNDS)
        .flat_map(|_| {
            armed
                .iter()
                .map(|(_, model, image)| (model.clone(), image.clone()))
                .collect::<Vec<_>>()
        })
        .collect();
    let pages = items.len();
    log_line(
        case,
        "setup",
        "pass",
        &format!(
            r#""seed":0,"pool":{pool},"pages_issued":{pages},"timeout_budget_secs":{budget},"mode":"heavy_forward","lanes":{lanes},"rounds":{ZOO_INTERLEAVE_ROUNDS}"#,
            budget = WATCHDOG_BUDGET_HEAVY.as_secs(),
            lanes = armed.len(),
        ),
    );
    let outcome = run_with_watchdog(WATCHDOG_BUDGET_HEAVY, move || {
        let engine = OcrEngine::new().expect("OcrEngine::new builds its single owned runtime");
        drive_zoo_batch(&engine, &items)
    });
    report_zoo_outcome(case, "interleave", pool, pages, outcome);
}

/// Shared assertion + logging tail for both zoo passes: completion within
/// budget, every page driven terminal, and every call `Ok` (known-good
/// artifact + committed image ⇒ an error is a defect, never "progress").
fn report_zoo_outcome(
    case: &str,
    label: &str,
    pool: usize,
    pages: usize,
    outcome: Watch<BatchReport>,
) {
    match outcome {
        Watch::TimedOut => {
            log_line(
                case,
                "error",
                "fail",
                &format!(
                    r#""diag":{{"error_kind":"DEADLOCK_SUSPECTED","focr_exit_code":5,"message":"zoo batch did not complete within watchdog budget"}},"pool":{pool},"pages_issued":{pages},"timeout_budget_secs":{}"#,
                    WATCHDOG_BUDGET_HEAVY.as_secs(),
                ),
            );
            panic!(
                "DEADLOCK SUSPECTED ({label}): the sequential {pages}-page zoo batch \
                 (pool={pool}) did NOT complete within the {}s wall-clock watchdog \
                 budget. Doctrine #5 regression in a zoo lane: nested runtime, \
                 rayon under a held lock, or a cache-swap hang.",
                WATCHDOG_BUDGET_HEAVY.as_secs(),
            );
        }
        Watch::Finished { payload, elapsed } => {
            let BatchReport {
                completed,
                ok_results,
                model_not_found,
                other_terminal,
            } = payload;
            assert_eq!(
                completed, pages,
                "{label}: expected all {pages} pages driven to completion, got {completed}"
            );
            assert_eq!(
                ok_results, pages,
                "{label}: every zoo forward must succeed on the committed cert image \
                 (ok={ok_results}, model_not_found={model_not_found}, \
                 other_terminal={other_terminal})"
            );
            log_line(
                case,
                "result",
                "pass",
                &format!(
                    r#""elapsed_us":{elapsed_us},"pool":{pool},"pages_issued":{pages},"completed":{completed},"ok_results":{ok_results},"per_page_avg_us":{per_page}"#,
                    elapsed_us = elapsed.as_micros(),
                    per_page = elapsed.as_micros() / (completed.max(1) as u128),
                ),
            );
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
        &format!(
            r#""seed":0,"timeout_budget_secs":0,"budget_ms":{}"#,
            tiny.as_millis()
        ),
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

// ─────────────── scenario 5: the capacity certificate (bd-re8.18) ───────────────

/// Nearest-rank percentile: the `ceil(k/100 · n)`-th smallest sample
/// (1-indexed). The standard definition, so every number in the certificate is
/// reproducible by hand from the raw samples. Sorts a copy.
fn percentile_nearest_rank(samples: &[u128], k: usize) -> u128 {
    assert!(
        !samples.is_empty() && (1..=100).contains(&k),
        "percentile needs samples and k in 1..=100"
    );
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = (k * sorted.len()).div_ceil(100).max(1);
    sorted[rank - 1]
}

/// Hand-worked nearest-rank check (the classic 5-sample set): p30 of
/// [15,20,35,40,50] is the ceil(1.5)=2nd smallest = 20; p50 → 3rd = 35;
/// p100 → 5th = 50; p1 → 1st = 15.
#[test]
fn percentile_nearest_rank_hand_worked() {
    let s = [35u128, 20, 15, 50, 40];
    assert_eq!(percentile_nearest_rank(&s, 30), 20);
    assert_eq!(percentile_nearest_rank(&s, 50), 35);
    assert_eq!(percentile_nearest_rank(&s, 95), 50);
    assert_eq!(percentile_nearest_rank(&s, 100), 50);
    assert_eq!(percentile_nearest_rank(&s, 1), 15);
    log_line(
        "percentile_hand_worked",
        "result",
        "pass",
        r#""checks":5,"definition":"nearest_rank""#,
    );
}

/// How many pages the certificate's consumer deliberately lags on, and how
/// often, to force the bounded buffer full (the backpressure proof). Only the
/// first `LAG_PAGES` are lagged so the soak's tail runs at full speed.
const LAG_PAGES: usize = 64;
const LAG_EVERY: usize = 16;
const LAG_SLEEP: Duration = Duration::from_millis(2);

/// The bounded stream capacity under certification. Small on purpose: the
/// in-flight bound (`capacity + 1`) must be provable with a laggy consumer.
const STREAM_CAPACITY: usize = 4;

/// bd-re8.18: the asupersync capacity certificate. Drives the many-pages soak
/// through the BOUNDED `stream_pages` channel with a deliberately laggy
/// consumer, measures per-page latency at the producer, and emits ONE
/// structured `capacity_certificate` NDJSON artifact carrying:
///
///  - **p50/p95/p99/max queueing evidence** (nearest-rank over every page);
///  - **the bounded-channel proof**: observed max in-flight
///    (`produced − consumed`) never exceeds `capacity + 1` — backpressure,
///    never unbounded growth — AND the buffer provably FILLED at least once
///    under the injected consumer lag (fast mode), so the bound was exercised,
///    not just never approached;
///  - **the no-oversubscription proof**: the kernel rayon pool width measured
///    before and after the soak is identical (no second pool, no mid-run
///    growth — the N× multiplication doctrine #5 forbids) and within the
///    platform's logical width;
///  - **the no-nested-runtime evidence**: the soak runs under the same
///    wall-clock watchdog as scenario 1 (a nested runtime or rayon-under-lock
///    regression hangs exactly there), on a single engine, with a strictly
///    sequential producer.
///
/// Heavy branch (`FOCR_MODEL_PATH` + `FOCR_WATCHDOG_IMAGE`): real forwards
/// give real queueing numbers. Unarmed: the certificate still proves the
/// bounded stream + pool stability over the orchestration path, and `mode`
/// says which one it certified.
///
/// Armed sizing (`FOCR_CAPACITY_PAGES`): a real forward is seconds per page
/// (SAM-B vision encode dominates), so the default 256-page soak cannot fit
/// ANY real model inside the heavy watchdog budget (measured 2026-07-06:
/// 256 GOT int8 pages > 1800 s — the watchdog fired, correctly). Armed runs
/// size the soak with this env var instead; the override is floored at
/// `2 × pool` so pages > pool always holds, applies ONLY to the armed-heavy
/// branch, and the artifact records the page count — never a silent cap.
#[test]
fn capacity_certificate_bounded_stream_soak() {
    let case = "capacity_certificate";
    let pool = detected_pool();
    let heavy = heavy_model_path();
    let pages = match std::env::var("FOCR_CAPACITY_PAGES") {
        Ok(v) if heavy.is_some() => {
            let requested: usize = v
                .parse()
                .expect("FOCR_CAPACITY_PAGES must be a positive integer");
            requested.max(pool * 2)
        }
        _ => pages_for(pool),
    };
    let (model_path, budget) = match &heavy {
        Some(p) => (p.clone(), WATCHDOG_BUDGET_HEAVY),
        None => (PathBuf::from(ABSENT_MODEL), WATCHDOG_BUDGET),
    };
    let heavy_page = std::env::var_os("FOCR_WATCHDOG_IMAGE")
        .map(PathBuf::from)
        .filter(|p| heavy.is_some() && p.exists());
    let real_forwards = heavy_page.is_some();
    let mode = if real_forwards {
        "heavy_forward"
    } else if heavy.is_some() {
        // Weights present but no decodable page: the soak certifies model-load
        // + input-decode orchestration only, and says so.
        "heavy_orchestration_no_image"
    } else {
        "no_weights_orchestration"
    };

    let width_before = kernel_pool_width();
    let budget_threads = thread_budget();
    log_line(
        case,
        "setup",
        "pass",
        &format!(
            r#""seed":0,"pool":{pool},"pages_issued":{pages},"timeout_budget_secs":{bs},"mode":"{mode}","model_path":{mp},"stream_capacity":{STREAM_CAPACITY},"kernel_pool_width_before":{width_before},"thread_budget":{budget_threads}"#,
            bs = budget.as_secs(),
            mp = json_str(&model_path.display().to_string()),
        ),
    );

    let model_path_for_worker = model_path.clone();
    let outcome = run_with_watchdog(budget, move || {
        let engine = OcrEngine::new().expect("OcrEngine::new builds its single owned runtime");
        let page_buf = heavy_page.unwrap_or_else(|| PathBuf::from(SYNTHETIC_PAGE));

        // `produced` is bumped by the producer thread right before each send;
        // the consumer reads it to compute live in-flight. That makes the
        // observed maximum a true upper-bound witness on channel occupancy.
        let produced = Arc::new(AtomicUsize::new(0));
        let produced_probe = Arc::clone(&produced);

        let mut issued = 0usize;
        let mut samples: Vec<u128> = Vec::with_capacity(pages);
        let mut consumed = 0usize;
        let mut max_in_flight = 0usize;
        // Terminal-outcome tally: [Ok, ModelNotFound, other].
        let mut tally = [0usize; 3];

        let streamed = stream_pages(
            STREAM_CAPACITY,
            move || {
                if issued == pages {
                    return Ok(None);
                }
                issued += 1;
                let started = Instant::now();
                let kind: u8 = match engine.recognize_with_model(&model_path_for_worker, &page_buf)
                {
                    Ok(_) => 0,
                    Err(FocrError::ModelNotFound(_)) => 1,
                    Err(_) => 2,
                };
                let latency_us = started.elapsed().as_micros();
                produced_probe.fetch_add(1, Ordering::SeqCst);
                Ok(Some((latency_us, kind)))
            },
            |(latency_us, kind): (u128, u8)| {
                consumed += 1;
                let in_flight = produced.load(Ordering::SeqCst).saturating_sub(consumed);
                max_in_flight = max_in_flight.max(in_flight);
                samples.push(latency_us);
                tally[kind as usize] += 1;
                if consumed <= LAG_PAGES && consumed.is_multiple_of(LAG_EVERY) {
                    thread::sleep(LAG_SLEEP);
                }
            },
        )
        .expect("bounded page stream completes");
        (streamed, samples, max_in_flight, tally)
    });

    let (streamed, samples, max_in_flight, tally, elapsed) = match outcome {
        Watch::TimedOut => {
            log_line(
                case,
                "error",
                "fail",
                &format!(
                    r#""diag":{{"error_kind":"DEADLOCK_SUSPECTED","focr_exit_code":5,"message":"capacity soak did not complete within watchdog budget"}},"pool":{pool},"pages_issued":{pages},"timeout_budget_secs":{}"#,
                    budget.as_secs(),
                ),
            );
            panic!(
                "DEADLOCK SUSPECTED: the capacity-certificate soak ({pages} pages through \
                 the bounded stream) did NOT complete within the {}s watchdog budget — \
                 a nested runtime, rayon under a held lock, or an unbounded channel \
                 stall manifests exactly here.",
                budget.as_secs(),
            );
        }
        Watch::Finished { payload, elapsed } => {
            let (streamed, samples, max_in_flight, tally) = payload;
            (streamed, samples, max_in_flight, tally, elapsed)
        }
    };

    // Every page streamed, exactly once, in order (the stream returns the count).
    assert_eq!(
        streamed, pages,
        "expected {pages} pages streamed, got {streamed}"
    );
    assert_eq!(samples.len(), pages, "one latency sample per page");

    // The bounded-channel proof: occupancy never exceeded capacity + 1 (the
    // buffered items plus the one in the producer's hand mid-send)…
    assert!(
        max_in_flight <= STREAM_CAPACITY + 1,
        "bounded channel violated: observed {max_in_flight} in flight > capacity {STREAM_CAPACITY} + 1"
    );
    // …and, in fast mode, the bound was actually REACHED under the injected
    // lag — a bound never approached is a bound never tested. (The heavy
    // branch's real forwards are slower than the consumer, so the buffer
    // legitimately may never fill there; the artifact records the observed
    // value either way.)
    let backpressure_engaged = max_in_flight >= STREAM_CAPACITY;
    if !real_forwards {
        assert!(
            backpressure_engaged,
            "consumer lag ({LAG_PAGES} pages, {LAG_SLEEP:?} every {LAG_EVERY}) never filled the \
             {STREAM_CAPACITY}-slot buffer (max in-flight {max_in_flight}) — the backpressure \
             path went unexercised"
        );
    }

    // The no-oversubscription proof: pool width identical before/after, and
    // never wider than the platform's logical width.
    let width_after = kernel_pool_width();
    assert_eq!(
        width_before, width_after,
        "kernel rayon pool width CHANGED across the soak ({width_before} → {width_after}): \
         a second pool or mid-run growth is the doctrine-#5 oversubscription class"
    );
    assert!(
        width_after <= pool,
        "kernel pool width {width_after} exceeds the platform logical width {pool}"
    );

    // Outcome-shape honesty per branch (same contract as scenario 1).
    let [ok_results, model_not_found, other_terminal] = tally;
    if real_forwards {
        assert!(
            ok_results > 0,
            "heavy branch drove {pages} real pages but produced no Ok result \
             (model_not_found={model_not_found}, other_terminal={other_terminal})"
        );
    } else if heavy.is_none() {
        assert_eq!(
            model_not_found, pages,
            "no-weights branch: every call must return the clean fast ModelNotFound"
        );
    } else {
        // Weights without a real page: progress means post-resolution results,
        // never a silent ModelNotFound (that would mean the artifact never loaded).
        assert!(
            ok_results > 0 || other_terminal > 0,
            "heavy weights present but every call returned ModelNotFound ({model_not_found}) — \
             the artifact was not actually loaded"
        );
    }

    let p50 = percentile_nearest_rank(&samples, 50);
    let p95 = percentile_nearest_rank(&samples, 95);
    let p99 = percentile_nearest_rank(&samples, 99);
    let max = *samples.iter().max().expect("non-empty samples");
    assert!(
        p50 <= p95 && p95 <= p99 && p99 <= max,
        "percentiles must be monotone"
    );

    // THE artifact: one line, self-contained, consumed by the three-pillar cert.
    log_line(
        case,
        "artifact",
        "pass",
        &format!(
            r#""artifact":"capacity_certificate","schema":"focr-capacity-certificate/v1","mode":"{mode}","pages":{pages},"pool_logical":{pool},"thread_budget":{budget_threads},"kernel_pool_width":{width_after},"pool_width_stable":true,"latency_us":{{"p50":{p50},"p95":{p95},"p99":{p99},"max":{max},"samples":{n}}},"stream_capacity":{STREAM_CAPACITY},"max_in_flight":{max_in_flight},"bounded_channel_verified":true,"backpressure_engaged":{backpressure_engaged},"ok_results":{ok_results},"model_not_found":{model_not_found},"other_terminal":{other_terminal},"elapsed_us":{elapsed_us},"no_nested_runtime_evidence":"watchdogged completion + single engine + sequential producer (scenario 1b gauge covers one-live-forward)""#,
            n = samples.len(),
            elapsed_us = elapsed.as_micros(),
        ),
    );
    log_line(
        case,
        "result",
        if heavy.is_none() {
            "skip_no_model"
        } else {
            "pass"
        },
        &format!(
            r#""elapsed_us":{},"pages_issued":{pages},"certified_mode":"{mode}","native_path_ran":true,"fallback_target":{ft}"#,
            elapsed.as_micros(),
            ft = json_str(&model_path.display().to_string()),
        ),
    );
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
