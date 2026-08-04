//! Human progress reporting for long OCR runs (GitHub issue #2).
//!
//! One self-contained, dependency-free progress line on STDERR for the
//! long-running drivers: the per-page PDF loops, the `--multi-page` raster +
//! cross-page decode, and the sequential `ocr-batch` loop. A big scanned book
//! is minutes of silent compute otherwise; the bar answers "is it alive, how
//! far along, and roughly how long to go" (done/total, percent, elapsed, and a
//! simple linear ETA).
//!
//! Design constraints, in priority order:
//!
//! 1. **Machine output is sacred.** focr's robot NDJSON stream and `--json`
//!    outputs are consumed by agents; a progress bar must never be able to
//!    corrupt them. Three independent guards enforce that:
//!    * the bar writes ONLY to stderr — stdout is never touched;
//!    * callers construct it explicitly disabled in `--robot` / `--json`
//!      modes (the `human_mode` argument to [`Progress::new`]);
//!    * it auto-disables when stderr is not an interactive terminal, so a
//!      piped or redirected run sees the exact byte stream it always saw
//!      (`\r` redraw sequences never land in a log file).
//! 2. **No new dependencies.** The repo's dependency block is deliberately
//!    minimal and every entry is justified; a `\r`-redraw line over
//!    `std::io::stderr` needs none of `indicatif`'s machinery. ASCII-only
//!    output, so it renders on any terminal.
//! 3. **Explicit kill switch.** `FOCR_NO_PROGRESS=1` (any value other than
//!    empty/`0`) disables the bar even on a TTY, and `TERM=dumb` terminals
//!    are respected — same conventions the wider CLI ecosystem uses.
//!
//! When the bar is disabled every method is a no-op and [`Progress::note`]
//! degrades to a plain `eprintln!`, so non-TTY stderr output is byte-identical
//! to what the CLI printed before this module existed.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, PoisonError, TryLockError};
use std::time::Instant;

/// Environment kill switch: set (to anything but empty/`0`) to suppress the
/// bar even when stderr is an interactive terminal.
pub const NO_PROGRESS_ENV: &str = "FOCR_NO_PROGRESS";

/// Width of the `[####----]` fill region, in cells.
const BAR_CELLS: usize = 24;

/// Width of the currently rendered line. Interrupt and error paths use this
/// atomic fallback so they never need the state mutex owned by a runtime
/// callback.
static ACTIVE_LINE_WIDTH: AtomicUsize = AtomicUsize::new(0);

/// Ownership token for [`ACTIVE_LINE_WIDTH`]. Detached renderers can overlap
/// briefly across raster/decode phases and retry attempts; an old renderer may
/// clear only the line it originally published.
static ACTIVE_GENERATION: AtomicUsize = AtomicUsize::new(0);
static NEXT_GENERATION: AtomicUsize = AtomicUsize::new(1);

/// Ctrl+C suppression is process-wide: once cancellation is requested, no
/// progress renderer may redraw over the interrupt notice.
static PROGRESS_SUPPRESSED: AtomicBool = AtomicBool::new(false);

/// Serialize terminal writes and the final clear. Streaming callbacks never
/// take this lock; only the detached UI renderer does, so a stalled terminal
/// cannot pin an `OcrEngine` blocking-pool worker.
static OUTPUT_LOCK: Mutex<()> = Mutex::new(());

/// Mutable bar state behind the [`Progress`] handle. `Arc` so a boxed
/// `'static` page sink (see [`Progress::page_sink`]) can advance the same bar
/// the owning driver renders.
struct Inner {
    /// Short driver tag rendered after `[focr]` (e.g. `ocr`, `raster`, `batch`).
    label: &'static str,
    /// Total number of work items (pages / images).
    total: usize,
    /// Completed work items.
    done: usize,
    /// Human description of the item in flight (e.g. `page 3/12`).
    current: String,
    /// Run start, for elapsed/ETA.
    started: Instant,
}

/// Immutable renderer input. Streaming code clones this under the state mutex
/// and releases that mutex before any terminal I/O.
struct RenderFrame {
    label: &'static str,
    total: usize,
    done: usize,
    current: String,
    elapsed_secs: f64,
}

impl From<&Inner> for RenderFrame {
    fn from(inner: &Inner) -> Self {
        Self {
            label: inner.label,
            total: inner.total,
            done: inner.done,
            current: inner.current.clone(),
            elapsed_secs: inner.started.elapsed().as_secs_f64(),
        }
    }
}

/// State shared with the blocking-pool page callback and detached renderer.
/// The callback only mutates this small state and sends a channel command;
/// terminal I/O is confined to the renderer thread.
struct Shared {
    state: Mutex<Inner>,
    finished: AtomicBool,
    suspended: AtomicBool,
    generation: usize,
}

enum RenderCommand {
    Draw,
    Finish(Option<Sender<()>>),
}

/// A single-line stderr progress bar; disabled instances are free no-ops.
///
/// Dropping an unfinished bar retires it and requests a clear without joining
/// the detached renderer. Timeout and cancellation therefore never wait for a
/// terminal write or for a model-runtime callback.
pub struct Progress {
    inner: Option<Arc<Shared>>,
    renderer: Option<Sender<RenderCommand>>,
}

impl Progress {
    /// A bar over `total` items, enabled only when ALL of these hold:
    /// `human_mode` (the caller is NOT in `--robot`/`--json` mode), `total`
    /// is non-zero, and stderr is an interactive terminal that has not opted
    /// out (see [`progress_allowed`]).
    pub fn new(label: &'static str, total: usize, human_mode: bool) -> Self {
        if !human_mode
            || total == 0
            || PROGRESS_SUPPRESSED.load(Ordering::Acquire)
            || !stderr_supports_progress()
        {
            return Self {
                inner: None,
                renderer: None,
            };
        }
        let inner = Arc::new(Shared {
            state: Mutex::new(Inner {
                label,
                total,
                done: 0,
                current: String::new(),
                started: Instant::now(),
            }),
            finished: AtomicBool::new(false),
            suspended: AtomicBool::new(false),
            generation: NEXT_GENERATION.fetch_add(1, Ordering::Relaxed),
        });
        let (renderer, commands) = channel();
        let renderer_state = Arc::clone(&inner);
        if std::thread::Builder::new()
            .name("focr-progress".into())
            .spawn(move || render_commands(renderer_state, commands))
            .is_err()
        {
            return Self {
                inner: None,
                renderer: None,
            };
        }
        Self {
            inner: Some(inner),
            renderer: Some(renderer),
        }
    }

    /// True when the bar will actually draw. Drivers with a cheaper
    /// non-streaming path may branch on this to keep the disabled path
    /// byte-for-byte what it always was.
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some() && self.renderer.is_some()
    }

    /// Mark `current` as the item now in flight and redraw. The fill/percent
    /// reflect COMPLETED items, so a 12-page run draws `0%` while page 1
    /// decodes and reaches `100%` only as the line is retired.
    pub fn start_item(&self, current: impl Into<String>) {
        let Some(inner) = &self.inner else { return };
        if inner.finished.load(Ordering::Acquire) {
            return;
        }
        inner.suspended.store(false, Ordering::Release);
        let mut g = lock(&inner.state);
        if inner.finished.load(Ordering::Acquire) {
            return;
        }
        g.current = current.into();
        drop(g);
        self.send(RenderCommand::Draw);
    }

    /// Record one completed item. No redraw: the next `start_item` (or sink
    /// tick) renders the advanced state, and the terminal `finish` clears the
    /// line anyway — drawing here would just flicker.
    pub fn complete_item(&self) {
        let Some(inner) = &self.inner else { return };
        if inner.finished.load(Ordering::Acquire) {
            return;
        }
        let mut g = lock(&inner.state);
        if inner.finished.load(Ordering::Acquire) {
            return;
        }
        g.done = (g.done + 1).min(g.total);
    }

    /// Temporarily clear the line (state kept) so the caller can print other
    /// output — e.g. `ocr-batch` writing a result block — without the bar
    /// visually fused to it. The next `start_item` restores the bar.
    pub fn suspend(&self) {
        let Some(inner) = &self.inner else { return };
        if inner.finished.load(Ordering::Acquire) {
            return;
        }
        inner.suspended.store(true, Ordering::Release);
        clear_bar_line(inner.generation);
    }

    /// Print a one-off stderr message (a skipped-page warning) WITHOUT
    /// corrupting the bar: clear the line, print, redraw. Disabled bars
    /// degrade to a plain `eprintln!`, keeping non-TTY stderr byte-identical
    /// to the pre-bar CLI.
    pub fn note(&self, msg: &str) {
        let Some(inner) = &self.inner else {
            stderr_message(format_args!("{msg}"));
            return;
        };
        if inner.finished.load(Ordering::Acquire) || PROGRESS_SUPPRESSED.load(Ordering::Acquire) {
            return;
        }
        inner.suspended.store(true, Ordering::Release);
        {
            let _output = output_lock();
            clear_generation_locked(inner.generation);
            let mut err = std::io::stderr().lock();
            let _ = writeln!(err, "{msg}");
            let _ = err.flush();
        }
        inner.suspended.store(false, Ordering::Release);
        self.send(RenderCommand::Draw);
    }

    /// Retire the bar and make every later call (including stray sink ticks) a
    /// no-op. This normal-completion boundary waits for the detached renderer
    /// to clear its line before subsequent stdout/stderr output is allowed.
    pub fn finish(&self) {
        let Some(inner) = &self.inner else { return };
        if inner.finished.swap(true, Ordering::AcqRel) {
            return;
        }
        inner.suspended.store(true, Ordering::Release);
        let (ack_tx, ack_rx) = channel();
        self.send(RenderCommand::Finish(Some(ack_tx)));
        if ack_rx.recv().is_err() {
            clear_bar_line(inner.generation);
        }
    }

    /// Retire without waiting for terminal I/O. Error/timeout paths and `Drop`
    /// use this form so a stalled TTY can never delay runtime teardown.
    pub fn retire(&self) {
        let Some(inner) = &self.inner else { return };
        if inner.finished.swap(true, Ordering::AcqRel) {
            return;
        }
        inner.suspended.store(true, Ordering::Release);
        self.send(RenderCommand::Finish(None));
        try_clear_bar_line(inner.generation);
    }

    /// A `'static` boxed per-page sink ([`crate::PageSink`]) that advances
    /// this bar as the multi-page decode crosses each `<PAGE>` boundary — the
    /// human-mode twin of the robot NDJSON `page` events. The callback only
    /// updates state and performs a nonblocking channel wake. A detached UI
    /// thread owns all stderr I/O, so a stalled terminal can never pin the
    /// model runtime's blocking worker or its timeout cleanup.
    pub fn page_sink(&self) -> crate::PageSink {
        let Some(inner) = &self.inner else {
            return Box::new(|_page, _body| {});
        };
        let Some(renderer) = self.renderer.as_ref().cloned() else {
            return Box::new(|_page, _body| {});
        };
        let inner = Arc::clone(inner);
        Box::new(move |page, _body| {
            if inner.finished.load(Ordering::Acquire) || PROGRESS_SUPPRESSED.load(Ordering::Acquire)
            {
                return;
            }
            let mut g = lock(&inner.state);
            if inner.finished.load(Ordering::Acquire) {
                return;
            }
            g.done = page.min(g.total);
            g.current = format!("page {page}/{}", g.total);
            drop(g);
            let _ = renderer.send(RenderCommand::Draw);
        })
    }

    fn send(&self, command: RenderCommand) {
        if let Some(renderer) = &self.renderer {
            let _ = renderer.send(command);
        }
    }
}

impl Drop for Progress {
    fn drop(&mut self) {
        self.retire();
    }
}

/// Lock helper: a panicked holder can't poison anything worse than a garbled
/// progress line, so recover instead of unwrapping.
fn lock(inner: &Mutex<Inner>) -> std::sync::MutexGuard<'_, Inner> {
    inner.lock().unwrap_or_else(PoisonError::into_inner)
}

fn output_lock() -> std::sync::MutexGuard<'static, ()> {
    OUTPUT_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

fn try_output_lock() -> Option<std::sync::MutexGuard<'static, ()>> {
    match OUTPUT_LOCK.try_lock() {
        Ok(output) => Some(output),
        Err(TryLockError::Poisoned(e)) => Some(e.into_inner()),
        Err(TryLockError::WouldBlock) => None,
    }
}

/// Detached renderer for all progress output. Callers, including the model
/// runtime's page sink, only update state and send an unbounded-channel command;
/// they never touch stderr or wait for terminal I/O.
fn render_commands(inner: Arc<Shared>, commands: Receiver<RenderCommand>) {
    while let Ok(command) = commands.recv() {
        match command {
            RenderCommand::Draw => draw_latest(&inner),
            RenderCommand::Finish(ack) => {
                clear_bar_line(inner.generation);
                if let Some(ack) = ack {
                    let _ = ack.send(());
                }
                return;
            }
        }
    }
    clear_bar_line(inner.generation);
}

fn draw_latest(inner: &Shared) {
    let frame = {
        let g = lock(&inner.state);
        RenderFrame::from(&*g)
    };
    let _ = draw_frame(inner, &frame);
}

/// Redraw the line in place (`\r` + content + blank the stale tail).
/// Write errors are ignored — a broken stderr must never fail an OCR run.
/// Render one immutable frame. The caller must not hold `Shared::state` when
/// invoked from the detached streaming renderer.
fn draw_frame(inner: &Shared, frame: &RenderFrame) -> usize {
    let _output = output_lock();
    if inner.finished.load(Ordering::Acquire)
        || inner.suspended.load(Ordering::Acquire)
        || PROGRESS_SUPPRESSED.load(Ordering::Acquire)
    {
        clear_generation_locked(inner.generation);
        return 0;
    }
    let line = render_line(
        frame.label,
        frame.done,
        frame.total,
        &frame.current,
        frame.elapsed_secs,
    );
    let width = line.chars().count();
    let previous_width = ACTIVE_LINE_WIDTH.load(Ordering::Acquire);
    let pad = previous_width.saturating_sub(width);
    ACTIVE_GENERATION.store(inner.generation, Ordering::Release);
    ACTIVE_LINE_WIDTH.store(width.max(previous_width), Ordering::Release);
    let mut err = std::io::stderr().lock();
    let _ = write!(err, "\r{line}{:pad$}", "");
    let _ = err.flush();
    drop(err);
    if inner.finished.load(Ordering::Acquire)
        || inner.suspended.load(Ordering::Acquire)
        || PROGRESS_SUPPRESSED.load(Ordering::Acquire)
    {
        clear_generation_locked(inner.generation);
        0
    } else {
        width
    }
}

/// Clear this bar's line while [`OUTPUT_LOCK`] is already held. A stale
/// renderer cannot clear a line published by a newer bar generation.
fn clear_generation_locked(generation: usize) {
    if ACTIVE_GENERATION.load(Ordering::Acquire) != generation {
        return;
    }
    let width = ACTIVE_LINE_WIDTH.swap(0, Ordering::AcqRel);
    ACTIVE_GENERATION.store(0, Ordering::Release);
    if width == 0 {
        return;
    }
    let mut err = std::io::stderr().lock();
    let _ = write!(err, "\r{:width$}\r", "");
    let _ = err.flush();
    ACTIVE_LINE_WIDTH.store(0, Ordering::Release);
}

/// Render one bar line (pure, for tests). ASCII only; never contains `\r` or
/// `\n` — the caller owns cursor movement.
fn render_line(label: &str, done: usize, total: usize, current: &str, elapsed_secs: f64) -> String {
    let total = total.max(1);
    let done = done.min(total);
    let filled = done * BAR_CELLS / total;
    let pct = done * 100 / total;
    let mut line = format!(
        "[focr] {label} [{}{}] {pct:>3}%",
        "#".repeat(filled),
        "-".repeat(BAR_CELLS - filled)
    );
    if !current.is_empty() {
        line.push(' ');
        // Filenames can contain CR/LF/ESC on Unix and arbitrary Unicode. Keep
        // the terminal control line single-line, escape-free, and one byte per
        // display cell by replacing everything outside printable ASCII.
        line.extend(current.chars().map(|ch| {
            if ch.is_ascii() && !ch.is_ascii_control() {
                ch
            } else {
                '?'
            }
        }));
    }
    line.push_str(&format!(", {} elapsed", fmt_duration(elapsed_secs)));
    // Linear ETA once at least one item has completed (before that there is
    // no rate to extrapolate; after the last item the bar is about to clear).
    if done > 0 && done < total {
        let eta = elapsed_secs / done as f64 * (total - done) as f64;
        line.push_str(&format!(", ~{} left", fmt_duration(eta)));
    }
    line
}

/// Clear a currently rendered line before an interrupt notice or terminal
/// error is printed.
///
/// This does not take the progress-state mutex. The atomic width is
/// conservative and a write failure is intentionally ignored, exactly like
/// ordinary redraws.
pub fn clear_active_line() {
    let _output = output_lock();
    clear_active_line_locked();
}

/// Print one complete stderr line without allowing it to fuse with an active
/// progress bar. Runtime diagnostics use this instead of writing to stderr
/// directly because model callbacks can overlap a detached renderer.
pub fn stderr_message(args: std::fmt::Arguments<'_>) {
    let _output = output_lock();
    clear_active_line_locked();
    let mut err = std::io::stderr().lock();
    let _ = writeln!(err, "{args}");
    let _ = err.flush();
}

/// Best-effort variant of [`stderr_message`] for teardown paths. Returns
/// `false` rather than waiting behind a renderer that may be stalled in a
/// terminal write.
pub fn try_stderr_message(args: std::fmt::Arguments<'_>) -> bool {
    let Some(_output) = try_output_lock() else {
        return false;
    };
    clear_active_line_locked();
    let mut err = std::io::stderr().lock();
    let _ = writeln!(err, "{args}");
    let _ = err.flush();
    true
}

fn clear_bar_line(generation: usize) {
    let _output = output_lock();
    clear_generation_locked(generation);
}

/// Permanently suppress progress for this process and best-effort clear the
/// active line. This must return promptly from the Ctrl+C callback even when a
/// renderer is stalled in terminal I/O.
pub fn suppress_for_interrupt() {
    PROGRESS_SUPPRESSED.store(true, Ordering::Release);
    try_clear_active_line();
}

/// Best-effort nonblocking clear used by [`Progress::retire`]. If a renderer
/// currently owns the output lock, it observes `finished` and clears before its
/// next loop iteration; timeout cleanup never waits for terminal I/O.
fn try_clear_bar_line(generation: usize) {
    if let Some(_output) = try_output_lock() {
        clear_generation_locked(generation);
    }
}

fn try_clear_active_line() {
    if let Some(_output) = try_output_lock() {
        clear_active_line_locked();
    }
}

/// Clear while [`OUTPUT_LOCK`] is already held.
fn clear_active_line_locked() {
    let width = ACTIVE_LINE_WIDTH.swap(0, Ordering::AcqRel);
    ACTIVE_GENERATION.store(0, Ordering::Release);
    if width == 0 {
        return;
    }
    let mut err = std::io::stderr().lock();
    let _ = write!(err, "\r{:width$}\r", "");
    let _ = err.flush();
}

/// `62.4s -> "1m02s"`, `3.2s -> "3s"`, `3725s -> "1h02m"`. Coarse on purpose:
/// an OCR ETA is an estimate, not a stopwatch.
fn fmt_duration(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{}h{:02}m", s / 3600, (s % 3600) / 60)
    }
}

/// The full runtime gate for the current process's stderr.
fn stderr_supports_progress() -> bool {
    // Timing mode emits a high-volume stderr trace from inside the model
    // runtime. Keeping that diagnostic stream line-oriented is more useful
    // than repeatedly clearing and redrawing a progress bar around it.
    if std::env::var_os("FOCR_TIMING").is_some() {
        return false;
    }
    progress_allowed(
        std::io::stderr().is_terminal(),
        std::env::var("TERM").ok().as_deref(),
        std::env::var(NO_PROGRESS_ENV).ok().as_deref(),
    )
}

/// Pure decision function (unit-testable without touching the real process
/// environment): a bar is allowed only on an interactive, non-`dumb` terminal
/// that has not set the [`NO_PROGRESS_ENV`] kill switch.
pub fn progress_allowed(
    stderr_is_tty: bool,
    term: Option<&str>,
    no_progress: Option<&str>,
) -> bool {
    if !stderr_is_tty {
        return false;
    }
    if term == Some("dumb") {
        return false;
    }
    match no_progress {
        None => true,
        Some(v) => {
            let v = v.trim();
            v.is_empty() || v == "0"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    static TEST_GLOBAL_LOCK: Mutex<()> = Mutex::new(());

    fn test_bar(total: usize) -> (Progress, Arc<Shared>, std::thread::JoinHandle<()>) {
        let shared = Arc::new(Shared {
            state: Mutex::new(Inner {
                label: "ocr",
                total,
                done: 0,
                current: String::new(),
                started: Instant::now(),
            }),
            finished: AtomicBool::new(false),
            suspended: AtomicBool::new(false),
            generation: NEXT_GENERATION.fetch_add(1, Ordering::Relaxed),
        });
        let (renderer, commands) = channel();
        let renderer_state = Arc::clone(&shared);
        let renderer_thread = std::thread::spawn(move || render_commands(renderer_state, commands));
        (
            Progress {
                inner: Some(Arc::clone(&shared)),
                renderer: Some(renderer),
            },
            shared,
            renderer_thread,
        )
    }

    #[test]
    fn not_a_tty_never_draws() {
        // The non-negotiable robot/pipe guarantee: no TTY, no bar — whatever
        // TERM or the kill switch say.
        assert!(!progress_allowed(false, None, None));
        assert!(!progress_allowed(false, Some("xterm-256color"), None));
        assert!(!progress_allowed(false, Some("xterm"), Some("0")));
    }

    #[test]
    fn tty_gate_respects_term_and_kill_switch() {
        assert!(progress_allowed(true, None, None));
        assert!(progress_allowed(true, Some("xterm-256color"), None));
        assert!(!progress_allowed(true, Some("dumb"), None));
        assert!(!progress_allowed(true, Some("xterm"), Some("1")));
        assert!(!progress_allowed(true, Some("xterm"), Some("true")));
        // Explicit opt-back-in values keep the bar.
        assert!(progress_allowed(true, Some("xterm"), Some("0")));
        assert!(progress_allowed(true, Some("xterm"), Some("")));
    }

    #[test]
    fn robot_and_json_modes_construct_disabled() {
        // `human_mode = false` is how the CLI spells --robot/--json; the bar
        // must be inert regardless of the terminal.
        let bar = Progress::new("ocr", 10, false);
        assert!(!bar.is_enabled());
        // And every method on a disabled bar is a safe no-op.
        bar.start_item("page 1/10");
        bar.complete_item();
        bar.suspend();
        bar.retire();
        let mut sink = bar.page_sink();
        sink(3, "body");
    }

    #[test]
    fn zero_total_constructs_disabled() {
        assert!(!Progress::new("ocr", 0, true).is_enabled());
    }

    #[test]
    fn render_line_shape() {
        let line = render_line("ocr", 4, 12, "page 5/12", 42.0);
        assert!(line.starts_with("[focr] ocr ["));
        assert!(line.contains(" 33%"), "{line}");
        assert!(line.contains("page 5/12"), "{line}");
        assert!(line.contains("42s elapsed"), "{line}");
        // 4 done in 42s -> 10.5 s/page -> 8 remaining ~ 84s = 1m24s.
        assert!(line.contains("~1m24s left"), "{line}");
        // Single line, ASCII only — the caller owns all cursor movement.
        assert!(!line.contains('\r') && !line.contains('\n'));
        assert!(line.is_ascii());
    }

    #[test]
    fn render_line_sanitizes_terminal_control_and_wide_characters() {
        let line = render_line(
            "batch",
            0,
            1,
            "image 1/1: bad\r\n\u{1b}[31m-wide-\u{754c}.png",
            0.0,
        );
        assert!(line.is_ascii(), "{line:?}");
        assert!(!line.contains('\r') && !line.contains('\n'));
        assert!(!line.contains('\u{1b}'));
        assert!(line.contains("bad???[31m-wide-?.png"), "{line:?}");
    }

    #[test]
    fn finish_does_not_wait_for_page_sink_state_lock() {
        let _serial = TEST_GLOBAL_LOCK.lock().unwrap();
        let (bar, shared, renderer) = test_bar(2);
        let guard = lock(&shared.state);
        let (tx, rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            bar.finish();
            let _ = tx.send(());
        });

        assert!(
            rx.recv_timeout(Duration::from_secs(1)).is_ok(),
            "finish blocked behind a busy page-sink mutex"
        );
        drop(guard);
        worker.join().expect("finish worker");
        renderer.join().expect("progress renderer");
        assert!(shared.finished.load(Ordering::Acquire));
    }

    #[test]
    fn renderer_cannot_publish_after_retirement() {
        let _serial = TEST_GLOBAL_LOCK.lock().unwrap();
        ACTIVE_LINE_WIDTH.store(0, Ordering::Release);
        ACTIVE_GENERATION.store(0, Ordering::Release);
        let (bar, shared, progress_renderer) = test_bar(2);
        let frame = {
            let mut g = lock(&shared.state);
            g.current = "page 1/2".into();
            RenderFrame::from(&*g)
        };
        let output_guard = output_lock();
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let renderer_state = Arc::clone(&shared);
        let renderer = std::thread::spawn(move || {
            let _ = started_tx.send(());
            let rendered_width = draw_frame(&renderer_state, &frame);
            let _ = done_tx.send(rendered_width);
        });
        started_rx.recv().expect("renderer starts");

        bar.retire();
        assert!(
            done_rx.try_recv().is_err(),
            "renderer must be at output gate"
        );
        drop(output_guard);
        assert_eq!(
            done_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("renderer retires"),
            0
        );
        renderer.join().expect("renderer worker");
        progress_renderer.join().expect("progress renderer");
        assert_eq!(ACTIVE_LINE_WIDTH.load(Ordering::Acquire), 0);
    }

    #[test]
    fn stale_renderer_cannot_clear_a_newer_generation() {
        let _serial = TEST_GLOBAL_LOCK.lock().unwrap();
        let old_generation = NEXT_GENERATION.fetch_add(1, Ordering::Relaxed);
        let newer_generation = NEXT_GENERATION.fetch_add(1, Ordering::Relaxed);
        ACTIVE_GENERATION.store(newer_generation, Ordering::Release);
        ACTIVE_LINE_WIDTH.store(37, Ordering::Release);

        let shared = Arc::new(Shared {
            state: Mutex::new(Inner {
                label: "old",
                total: 1,
                done: 0,
                current: String::new(),
                started: Instant::now(),
            }),
            finished: AtomicBool::new(true),
            suspended: AtomicBool::new(true),
            generation: old_generation,
        });
        let (commands, receiver) = channel();
        let renderer = std::thread::spawn(move || render_commands(shared, receiver));
        let (ack_tx, ack_rx) = channel();
        commands
            .send(RenderCommand::Finish(Some(ack_tx)))
            .expect("finish stale renderer");
        ack_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("stale renderer acknowledges retirement");
        renderer.join().expect("stale renderer");

        assert_eq!(ACTIVE_GENERATION.load(Ordering::Acquire), newer_generation);
        assert_eq!(ACTIVE_LINE_WIDTH.load(Ordering::Acquire), 37);
        ACTIVE_GENERATION.store(0, Ordering::Release);
        ACTIVE_LINE_WIDTH.store(0, Ordering::Release);
    }

    #[test]
    fn teardown_stderr_never_waits_for_terminal_output() {
        let _serial = TEST_GLOBAL_LOCK.lock().unwrap();
        let output_guard = output_lock();
        let (tx, rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let wrote = try_stderr_message(format_args!("teardown"));
            let _ = tx.send(wrote);
        });

        assert!(
            !rx.recv_timeout(Duration::from_secs(1))
                .expect("teardown stderr returns"),
            "teardown stderr unexpectedly acquired the busy output lock"
        );
        worker.join().expect("teardown stderr worker");
        drop(output_guard);
    }

    #[test]
    fn page_sink_never_waits_for_terminal_output() {
        let _serial = TEST_GLOBAL_LOCK.lock().unwrap();
        let (bar, _shared, renderer) = test_bar(2);
        let mut sink = bar.page_sink();
        let output_guard = output_lock();
        let (tx, rx) = mpsc::channel();
        let callback = std::thread::spawn(move || {
            sink(1, "body");
            let _ = tx.send(());
        });

        assert!(
            rx.recv_timeout(Duration::from_secs(1)).is_ok(),
            "model-runtime callback waited for terminal I/O"
        );
        callback.join().expect("page callback");
        drop(output_guard);
        bar.finish();
        renderer.join().expect("progress renderer");
    }

    #[test]
    fn render_line_eta_bounds() {
        // No ETA before the first completion (no rate to extrapolate)...
        assert!(!render_line("ocr", 0, 5, "page 1/5", 3.0).contains("left"));
        // ...and none once everything is done.
        let done = render_line("ocr", 5, 5, "", 30.0);
        assert!(!done.contains("left"));
        assert!(done.contains("100%"));
    }

    #[test]
    fn duration_formatting() {
        assert_eq!(fmt_duration(3.2), "3s");
        assert_eq!(fmt_duration(62.4), "1m02s");
        assert_eq!(fmt_duration(3725.0), "1h02m");
        assert_eq!(fmt_duration(-1.0), "0s");
    }
}
