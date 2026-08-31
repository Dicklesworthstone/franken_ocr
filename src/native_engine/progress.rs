//! The forward-progress seam: an optional, process-global sink that the
//! browser playground installs so a multi-minute `recognize` can report where
//! it is instead of freezing behind one synchronous call.
//!
//! ## Contract
//!
//! * **Off by default, and free when off.** [`set_progress_sink`] is the only
//!   way to arm it; until it is called every entry point is a relaxed
//!   [`AtomicBool`] load and a return — no lock, no allocation, no formatting.
//!   The native CLI never arms it, so the native path pays exactly one
//!   predictable, never-taken branch per hook site (hooks live on outer loops:
//!   per vision block, per decoded token — never inside a kernel).
//! * **Events are advisory.** A sink that panics, blocks, or is re-entered
//!   must not be able to break a run: the emit path takes the sink lock with
//!   `try_lock` and silently skips the event if it cannot get it, and a
//!   poisoned lock disarms nothing.
//! * **Numerics are untouched.** Nothing here feeds a kernel; a run with a
//!   sink installed produces byte-identical output to one without.
//!
//! ## Threading
//!
//! Events are emitted ONLY from the thread that entered the forward — the
//! outer loops in [`super`], [`super::tromr`], [`super::vision_sam`],
//! [`super::vision_clip`] and [`super::decoder_qwen2`] all run on the calling
//! thread (doctrine #5: the page/staff/token loops are sequential; parallelism
//! lives inside the kernels). This is load-bearing for the wasm consumer,
//! whose sink closes over a `js_sys::Function` that is not really `Send`:
//! calling it from a rayon worker would be unsound. Do not add a hook inside a
//! `par_iter` body.
//!
//! ## Event vocabulary
//!
//! `total == 0` means indeterminate.
//!
//! | `stage`        | `current` / `total`                                  |
//! |----------------|------------------------------------------------------|
//! | `preprocess`   | `0/0` at entry, `1/1` when the tensor is ready        |
//! | `staff`        | staff index / staff count (TrOMR pages only)          |
//! | `vision`       | encoder blocks finished / blocks planned              |
//! | `prefill`      | `0/n` at entry, `n/n` when the KV cache is warm       |
//! | `decode`       | tokens emitted / the decode cap (`max_length`)        |
//! | `postprocess`  | `0/0` — decode finished, assembling the document      |

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

thread_local! {
    /// External progress code may itself touch an instrumented path. Suppress
    /// nested observations on that same thread without holding the sink mutex.
    static IN_CALLBACK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct CallbackGuard;

impl CallbackGuard {
    fn enter() -> Option<Self> {
        IN_CALLBACK.with(|active| {
            if active.replace(true) {
                None
            } else {
                Some(Self)
            }
        })
    }
}

impl Drop for CallbackGuard {
    fn drop(&mut self) {
        IN_CALLBACK.with(|active| active.set(false));
    }
}

/// One progress observation. `total == 0` marks an indeterminate stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgressEvent {
    /// Stage name from the vocabulary in the module docs (a `&'static str`, so
    /// emitting one never allocates).
    pub stage: &'static str,
    /// Work finished within the stage.
    pub current: u64,
    /// Work planned for the stage, or `0` when it is not known ahead of time.
    pub total: u64,
}

/// The installed observer. `Send + Sync` is required so the sink can live in a
/// `static`; see the threading note above for why it is nevertheless only ever
/// invoked on the thread that entered the forward.
pub type ProgressSink = Arc<dyn Fn(ProgressEvent) + Send + Sync>;

/// The fast-path flag. Read (relaxed) at the top of every hook; when it is
/// `false` nothing else in this module is touched.
static ARMED: AtomicBool = AtomicBool::new(false);
static SINK: Mutex<Option<ProgressSink>> = Mutex::new(None);
static VISION_TOTAL: AtomicU64 = AtomicU64::new(0);
static VISION_DONE: AtomicU64 = AtomicU64::new(0);

/// Install (or, with `None`, remove) the process-global progress sink.
///
/// Installing replaces any previous sink. The caller owns the lifetime: a wasm
/// consumer installs once after module init and never clears it.
pub fn set_progress_sink(sink: Option<ProgressSink>) {
    let armed = sink.is_some();
    // A sink that panicked mid-callback poisons the lock; recover rather than
    // leave the seam permanently unsettable.
    let mut slot = SINK.lock().unwrap_or_else(|e| e.into_inner());
    *slot = sink;
    // Publish AFTER the slot is written so no hook can observe `armed` with a
    // stale sink (the lock's release edge orders the write).
    ARMED.store(armed, Ordering::Relaxed);
}

/// Whether a sink is installed. Hook sites use this to skip building anything.
#[must_use]
#[inline]
pub fn enabled() -> bool {
    ARMED.load(Ordering::Relaxed)
}

/// Report one observation. A no-op when no sink is installed.
#[inline]
pub fn emit(stage: &'static str, current: u64, total: u64) {
    if !enabled() {
        return;
    }
    emit_cold(ProgressEvent {
        stage,
        current,
        total,
    });
}

#[cold]
#[inline(never)]
fn emit_cold(event: ProgressEvent) {
    let Some(_callback_guard) = CallbackGuard::enter() else {
        return;
    };
    // `try_lock`, never `lock`: an event is worth less than any risk of
    // stalling a forward. Clone the Arc and release the registry mutex before
    // invoking external code: a callback is allowed to replace or clear itself,
    // which would otherwise recurse into `set_progress_sink` and self-deadlock.
    let sink = {
        let Ok(slot) = SINK.try_lock() else { return };
        slot.as_ref().map(Arc::clone)
    };
    if let Some(sink) = sink {
        sink(event);
    }
}

/// Declare the vision encoder's block budget for the run about to start and
/// reset the block counter. Called by the OUTERMOST owner of the vision pass
/// (the per-view loop for Unlimited-OCR, `tromr::encode` for TrOMR) so the
/// per-block hooks inside the towers can report one continuous count.
///
/// A no-op when no sink is installed; [`vision_step`] emits nothing until a
/// budget has been declared, so a tower reached from an unhooked path stays
/// silent rather than reporting against a stale total.
pub fn vision_begin(total_blocks: u64) {
    if !enabled() {
        return;
    }
    VISION_TOTAL.store(total_blocks, Ordering::Relaxed);
    VISION_DONE.store(0, Ordering::Relaxed);
    emit("vision", 0, total_blocks);
}

/// Record one finished vision-encoder block against the budget from
/// [`vision_begin`].
#[inline]
pub fn vision_step() {
    if !enabled() {
        return;
    }
    let total = VISION_TOTAL.load(Ordering::Relaxed);
    if total == 0 {
        return;
    }
    let done = VISION_DONE.fetch_add(1, Ordering::Relaxed) + 1;
    emit("vision", done.min(total), total);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// Serializes the tests in this module: the sink is process-global.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn disabled_by_default_and_free() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_progress_sink(None);
        assert!(!enabled());
        // Every entry point must be a silent no-op with no sink installed.
        emit("decode", 7, 100);
        vision_begin(12);
        vision_step();
    }

    #[test]
    fn events_reach_an_installed_sink() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let seen: Arc<StdMutex<Vec<ProgressEvent>>> = Arc::new(StdMutex::new(Vec::new()));
        let sink_seen = Arc::clone(&seen);
        set_progress_sink(Some(Arc::new(move |ev| {
            sink_seen.lock().unwrap_or_else(|e| e.into_inner()).push(ev);
        })));
        assert!(enabled());
        emit("preprocess", 0, 0);
        vision_begin(3);
        vision_step();
        vision_step();
        vision_step();
        // Past the declared budget the count saturates rather than lying.
        vision_step();
        emit("decode", 2, 256);
        set_progress_sink(None);
        assert!(!enabled());
        emit("decode", 3, 256); // dropped: the sink is gone

        let events = seen.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let expected = [
            ProgressEvent {
                stage: "preprocess",
                current: 0,
                total: 0,
            },
            ProgressEvent {
                stage: "vision",
                current: 0,
                total: 3,
            },
            ProgressEvent {
                stage: "vision",
                current: 1,
                total: 3,
            },
            ProgressEvent {
                stage: "vision",
                current: 2,
                total: 3,
            },
            ProgressEvent {
                stage: "vision",
                current: 3,
                total: 3,
            },
            ProgressEvent {
                stage: "vision",
                current: 3,
                total: 3,
            },
            ProgressEvent {
                stage: "decode",
                current: 2,
                total: 256,
            },
        ];
        assert_eq!(events, expected);
    }

    #[test]
    fn a_reentrant_sink_cannot_deadlock_the_forward() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let count = Arc::new(StdMutex::new(0usize));
        for _ in 0..10 {
            let sink_count = Arc::clone(&count);
            set_progress_sink(Some(Arc::new(move |_| {
                *sink_count.lock().unwrap_or_else(|e| e.into_inner()) += 1;
                // A sink that emits again re-enters the seam; the thread-local
                // callback guard drops that nested observation without retaining
                // the registry mutex across caller code.
                emit("decode", 0, 0);
            })));
            emit("decode", 1, 8);
        }
        set_progress_sink(None);
        assert_eq!(*count.lock().unwrap_or_else(|e| e.into_inner()), 10);
    }

    #[test]
    fn a_sink_can_clear_itself_without_deadlocking() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let count = Arc::new(StdMutex::new(0usize));
        for _ in 0..10 {
            let sink_count = Arc::clone(&count);
            set_progress_sink(Some(Arc::new(move |_| {
                *sink_count.lock().unwrap_or_else(|e| e.into_inner()) += 1;
                set_progress_sink(None);
            })));
            emit("decode", 1, 8);
            emit("decode", 2, 8);
        }
        assert_eq!(*count.lock().unwrap_or_else(|e| e.into_inner()), 10);
        assert!(!enabled());
    }
}
