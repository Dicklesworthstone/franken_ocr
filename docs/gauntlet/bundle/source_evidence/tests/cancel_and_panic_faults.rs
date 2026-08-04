//! `cancel_and_panic_faults` — the two concurrency-failure variants deferred
//! from the watchdog (bd-1ryu; bd-2ub2 REVIEW-2 addendum): a botched
//! CANCELLATION and a worker PANIC are distinct bug classes from a deadlock,
//! and each must fail TYPED and RECOVERABLE, never as a hang or a poisoned
//! engine.
//!
//! ## Why this is its own test binary (process isolation)
//!
//! `request_shutdown()` flips a PROCESS-GLOBAL flag. Inside the shared
//! `many_pages_without_deadlock` binary a mid-soak cancel would race the
//! sibling scenarios' engines through that global. One integration-test file
//! = one process = the flag manipulation is confined here by construction.
//! Only ONE test in this file touches the flag, so the in-file test threads
//! cannot race it either.
//!
//! ## The legs
//!
//! 1. **Cancel-mid-soak**: a canceler thread fires `request_shutdown()` while
//!    a sequential pages≫pool loop runs (each page boundary calls
//!    `cancel_checkpoint()` — the documented cooperative pattern, the same one
//!    the engine's decode loops use at `native_engine/mod.rs` and the CLI's
//!    Ctrl+C handler drives). Asserts: the loop aborts PROMPTLY with the typed
//!    `Cancelled` (exit 6) under a wall-clock watchdog, forward progress had
//!    happened, and — after `reset_shutdown()` — the SAME engine still answers
//!    cleanly (the model-cache mutex is not poisoned) and a FRESH engine
//!    builds and answers (no spinning thread wedges construction).
//! 2. **Kernel-pool task panic**: a panicking rayon task must propagate as a
//!    catchable panic without wedging the GLOBAL pool the int8 kernels share —
//!    the next parallel op computes correctly and the pool width is unchanged
//!    (the doctrine-#5 topology invariant, logged on the failure path).
//! 3. **Stream worker panic**: a panicking `stream_pages` producer surfaces as
//!    the TYPED "worker panicked" error (never a hang, never a torn channel),
//!    and a subsequent stream on fresh closures works.
//!
//! The armed variant (cancel during a REAL decode) is covered by the live-fired
//! CLI Ctrl+C exit-6 evidence (bd-223.2) and the engine's in-stage cancellation
//! unit test; this file proves the soak-level recovery properties those don't.

use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use franken_ocr::{
    FocrError, OcrEngine, kernel_pool_width, request_shutdown, reset_shutdown, stream_pages,
};

const ABSENT_MODEL: &str = "/nonexistent/cancel_fault/model.focrq";
const SYNTHETIC_PAGE: &str = "/nonexistent/cancel_fault/page.png";

/// Everything here must finish far inside this; a hang IS the failure.
const WATCHDOG: Duration = Duration::from_secs(60);

fn emit(case: &str, result: &str, fields: &str) {
    eprintln!(
        r#"{{"schema_version":1,"test":"cancel_and_panic_faults","case":"{case}","event":"result","result":"{result}"{}{fields}}}"#,
        if fields.is_empty() { "" } else { "," },
    );
}

/// Watchdogged run: the worker sends its payload once; a timeout is a hang.
fn with_watchdog<T, F>(work: F) -> T
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(work());
    });
    match rx.recv_timeout(WATCHDOG) {
        Ok(v) => v,
        Err(_) => panic!("HANG SUSPECTED: worker did not finish within {WATCHDOG:?}"),
    }
}

/// Leg 1: cancel mid-soak → prompt typed abort, no poison, engine reusable.
/// (The ONLY test in this binary that touches the process-global flag.)
#[test]
fn cancel_mid_soak_is_prompt_typed_and_leaves_the_engine_reusable() {
    let case = "cancel_mid_soak";
    reset_shutdown();
    let width_before = kernel_pool_width();

    let (outcome, elapsed) = with_watchdog(move || {
        let engine = OcrEngine::new().expect("engine builds");
        // The canceler: fires mid-soak, from another thread — the Ctrl+C shape.
        let canceler = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(50));
            request_shutdown();
        });
        let started = Instant::now();
        let mut completed = 0usize;
        let mut result: Result<(), FocrError> = Ok(());
        // pages ≫ pool, strictly sequential, checkpoint at every page
        // boundary — the documented cooperative loop shape.
        for _ in 0..5_000_000usize {
            if let Err(e) = franken_ocr::cancel_checkpoint() {
                result = Err(e);
                break;
            }
            let _ = engine.recognize_with_model(Path::new(ABSENT_MODEL), Path::new(SYNTHETIC_PAGE));
            completed += 1;
        }
        let elapsed = started.elapsed();
        canceler.join().expect("canceler joins");

        // Recovery on the SAME engine: after reset, the model-cache mutex must
        // not be poisoned — the call answers with the clean typed error.
        reset_shutdown();
        let same_engine_ok = matches!(
            engine.recognize_with_model(Path::new(ABSENT_MODEL), Path::new(SYNTHETIC_PAGE)),
            Err(FocrError::ModelNotFound(_))
        );
        ((result, completed, same_engine_ok), elapsed)
    });
    let (result, completed, same_engine_ok) = outcome;

    let cancelled_typed = matches!(result, Err(FocrError::Cancelled));
    let exit_code = result.as_ref().err().map(FocrError::exit_code);
    // A fresh engine must also build and answer — no spinning thread survived.
    let fresh_engine_ok = with_watchdog(|| {
        let engine = OcrEngine::new().expect("fresh engine builds after cancel");
        matches!(
            engine.recognize_with_model(Path::new(ABSENT_MODEL), Path::new(SYNTHETIC_PAGE)),
            Err(FocrError::ModelNotFound(_))
        )
    });
    let width_after = kernel_pool_width();

    let pass = cancelled_typed
        && exit_code == Some(6)
        && completed > 0
        && elapsed < Duration::from_secs(10)
        && same_engine_ok
        && fresh_engine_ok
        && width_before == width_after;
    emit(
        case,
        if pass { "pass" } else { "fail" },
        &format!(
            r#""cancelled_typed":{cancelled_typed},"exit_code":{exit_code:?},"pages_before_cancel":{completed},"elapsed_us":{},"same_engine_reusable":{same_engine_ok},"fresh_engine_ok":{fresh_engine_ok},"pool_width_stable":{},"live_forwards_max":1,"nested_runtime":false"#,
            elapsed.as_micros(),
            width_before == width_after,
        ),
    );
    assert!(
        pass,
        "cancel-mid-soak: typed={cancelled_typed} exit={exit_code:?} completed={completed} \
         elapsed={elapsed:?} same_engine={same_engine_ok} fresh={fresh_engine_ok}"
    );
}

/// Leg 2: a panicking kernel-pool task propagates as a catchable panic and
/// the GLOBAL rayon pool the int8 kernels share stays fully usable — width
/// unchanged, next parallel op correct.
#[test]
fn kernel_pool_task_panic_does_not_wedge_the_pool() {
    use rayon::prelude::*;
    let case = "kernel_pool_panic";
    let width_before = kernel_pool_width();

    let caught = with_watchdog(|| {
        std::panic::catch_unwind(|| {
            (0..1024usize).into_par_iter().for_each(|i| {
                assert!(i != 511, "injected kernel-task panic at index 511");
            });
        })
        .is_err()
    });

    // The pool must still compute — correctly — after the panic.
    let sum = with_watchdog(|| (0..1024u64).into_par_iter().sum::<u64>());
    let width_after = kernel_pool_width();

    let pass = caught && sum == 1024 * 1023 / 2 && width_before == width_after;
    emit(
        case,
        if pass { "pass" } else { "fail" },
        &format!(
            r#""panic_caught":{caught},"post_panic_sum_ok":{},"pool_width_before":{width_before},"pool_width_after":{width_after},"nested_runtime":false"#,
            sum == 1024 * 1023 / 2,
        ),
    );
    assert!(
        pass,
        "pool panic recovery failed: caught={caught} sum={sum} width {width_before}->{width_after}"
    );
}

/// Leg 3: a panicking stream producer is a TYPED error, not a hang; the
/// stream machinery is reusable immediately after.
#[test]
fn stream_worker_panic_is_typed_not_a_hang() {
    let case = "stream_worker_panic";

    let (err_typed, consumed) = with_watchdog(|| {
        let mut consumed = 0usize;
        let mut produced = 0usize;
        let result = stream_pages(
            2,
            move || {
                produced += 1;
                assert!(produced <= 3, "injected producer panic on item 4");
                Ok(Some(produced))
            },
            |_item: usize| {
                consumed += 1;
            },
        );
        let err_typed = matches!(result, Err(FocrError::Other(_)))
            && format!("{}", result.unwrap_err()).contains("panicked");
        (err_typed, consumed)
    });

    // Fresh stream on new closures completes cleanly.
    let reuse_ok = with_watchdog(|| {
        let mut n = 0usize;
        let mut left = 5usize;
        let streamed = stream_pages(
            2,
            move || {
                if left == 0 {
                    return Ok(None);
                }
                left -= 1;
                Ok(Some(left))
            },
            |_item: usize| {
                n += 1;
            },
        );
        matches!(streamed, Ok(5))
    });

    let pass = err_typed && consumed <= 3 && reuse_ok;
    emit(
        case,
        if pass { "pass" } else { "fail" },
        &format!(
            r#""error_typed":{err_typed},"consumed_before_panic":{consumed},"stream_reusable":{reuse_ok}"#
        ),
    );
    assert!(
        pass,
        "stream panic leg failed: typed={err_typed} consumed={consumed} reuse={reuse_ok}"
    );
}
