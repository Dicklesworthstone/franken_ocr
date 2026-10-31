//! The `focr ocr-batch --progress-events` stream (bd-mgd3): one JSON object per
//! line, to a file or stderr, so a long batch can be watched while `--json`
//! stdout stays the one final object.
//!
//! Every line reports something that already happened; nothing is estimated
//! during a decode. Lines are written from the run's single driver, so the
//! stream adds no concurrent work.
//!
//! - `batch_start` names the resolved thread budget, model and backend.
//! - `image_complete` and `image_error` report one image of an independent
//!   batch, as it ends.
//! - `page_complete` and `page_skipped` report one page of a `--multi-page`
//!   document. Those pages are one cross-page pass, not independent images,
//!   and `batch_start` says so.
//! - `batch_cancelled` names what a Ctrl+C interrupted.
//! - `batch_end` is the last line of a started run, with its outcome.
//!
//! `pages_per_hour` is the pages done so far over the wall time since
//! `batch_start`, which includes the model load. `eta_ms` is `null` until one
//! item has ended, then the mean wall time of the items ended so far times the
//! items left ([`PageEta`]); the first item carries the load, so it starts high
//! and falls. A failed image counts toward the estimate, not toward the rate.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use serde_json::{Map, Value, json};

use crate::DocumentEvent;
use crate::error::{FocrError, FocrResult};
use crate::page_source::PageItem;
use crate::progress::{self, PageEta};

/// The stream's own version, independent of the robot schema's.
pub const BATCH_EVENTS_SCHEMA_VERSION: u32 = 1;

/// The destination that means stderr.
pub const STDERR_DESTINATION: &str = "-";

/// Every event, in the order a run can emit them.
pub const BATCH_EVENT_KINDS: &[&str] = &[
    "batch_start",
    "image_complete",
    "image_error",
    "page_complete",
    "page_skipped",
    "batch_cancelled",
    "batch_end",
];

const PROGRESS_FIELDS: [&str; 7] = [
    "total",
    "cumulative_ms",
    "ok_images",
    "failed_images",
    "pages_per_hour",
    "eta_ms",
    "elapsed_ms",
];

/// How a batch reads its inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchMode {
    /// Each image parsed on its own.
    Images,
    /// One `--multi-page` document.
    Document,
}

/// The resolved run `batch_start` reports.
#[derive(Clone, Copy, Debug)]
pub struct BatchRun<'a> {
    pub mode: BatchMode,
    pub total: usize,
    pub threads: usize,
    pub model_path: &'a Path,
    /// The id the model's header names, when it resolves.
    pub model_id: Option<&'a str>,
    pub backend: &'a str,
    pub decode_mode: &'a str,
}

/// The `ocr-batch --progress-events` contract for `focr robot schema`.
#[must_use]
pub fn schema() -> Value {
    let fields = |extra: &[&str]| {
        let mut required = vec!["schema_version", "event"];
        required.extend_from_slice(extra);
        json!({"required": required, "optional": []})
    };
    let item = |extra: &[&str]| {
        let mut required: Vec<&str> = extra.to_vec();
        required.extend_from_slice(&PROGRESS_FIELDS);
        fields(&required)
    };
    json!({
        "schema_version": BATCH_EVENTS_SCHEMA_VERSION,
        "flag": "focr ocr-batch --progress-events <PATH>",
        "stderr_destination": STDERR_DESTINATION,
        "events": BATCH_EVENT_KINDS,
        "event_fields": {
            "batch_start": fields(&[
                "mode", "independent_images", "total", "threads", "model_path", "model_id",
                "backend", "decode_mode",
            ]),
            "image_complete": item(&["ordinal", "image"]),
            "image_error": item(&["ordinal", "image", "error", "exit_code"]),
            "page_complete": item(&["ordinal", "source_page", "image", "window"]),
            "page_skipped": item(&["source_page", "image", "error"]),
            "batch_cancelled": fields(&[
                "image", "total", "cumulative_ms", "ok_images", "failed_images",
            ]),
            "batch_end": fields(&[
                "outcome", "exit_code", "error", "total", "cumulative_ms", "ok_images",
                "failed_images", "pages_per_hour",
            ]),
        },
        "outcomes": ["completed", "cancelled", "failed"],
        "estimates": "pages_per_hour is pages done over the wall time since batch_start, \
                      including the model load. eta_ms is null until one item has ended, \
                      then the mean wall time of the items ended so far times the items left.",
    })
}

enum Destination {
    Stderr,
    Writer(Box<dyn Write + Send>),
}

/// The running counts each line reports.
#[derive(Debug, Default)]
struct Tally {
    total: usize,
    ok: usize,
    failed: usize,
    last_ms: u64,
    eta: PageEta,
}

impl Tally {
    /// Count one item that ended at `now_ms`, returning its wall time.
    fn ended(&mut self, ok: bool, now_ms: u64) -> u64 {
        if ok {
            self.ok += 1;
        } else {
            self.failed += 1;
        }
        let elapsed = now_ms.saturating_sub(self.last_ms);
        self.last_ms = now_ms;
        self.eta.add_window(1, elapsed);
        elapsed
    }

    fn pages_per_hour(&self, now_ms: u64) -> Option<f64> {
        (now_ms > 0).then(|| self.ok as f64 * 3_600_000.0 / now_ms as f64)
    }

    fn counts(&self, event: &mut Map<String, Value>, now_ms: u64) {
        event.insert("total".into(), self.total.into());
        event.insert("cumulative_ms".into(), now_ms.into());
        event.insert("ok_images".into(), self.ok.into());
        event.insert("failed_images".into(), self.failed.into());
    }

    /// The fields of an item that took `elapsed_ms`.
    fn progress(&self, event: &mut Map<String, Value>, now_ms: u64, elapsed_ms: u64) {
        self.counts(event, now_ms);
        let left = self.total.saturating_sub(self.ok + self.failed);
        event.insert("elapsed_ms".into(), elapsed_ms.into());
        event.insert("pages_per_hour".into(), self.pages_per_hour(now_ms).into());
        event.insert("eta_ms".into(), self.eta.remaining_ms(left).into());
    }
}

struct Stream {
    out: Destination,
    now_ms: Box<dyn Fn() -> u64 + Send>,
    tally: Tally,
    started: bool,
    cancel_reported: bool,
    write_failed: bool,
}

impl Stream {
    /// Write one line. The stream is observability: a destination that fails
    /// is reported once and dropped, and the batch goes on.
    fn write(&mut self, event: &str, fields: Map<String, Value>) {
        if self.write_failed {
            return;
        }
        let mut line = Map::new();
        line.insert("schema_version".into(), BATCH_EVENTS_SCHEMA_VERSION.into());
        line.insert("event".into(), event.into());
        line.extend(fields);
        let line = Value::Object(line).to_string();
        let written = match &mut self.out {
            Destination::Stderr => {
                progress::stderr_message(format_args!("{line}"));
                Ok(())
            }
            Destination::Writer(out) => writeln!(out, "{line}").and_then(|()| out.flush()),
        };
        if let Err(err) = written {
            self.write_failed = true;
            progress::stderr_message(format_args!(
                "[focr] warning: --progress-events stopped: {err}"
            ));
        }
    }

    fn cancelled(&mut self, image: Option<&Path>) {
        let now = (self.now_ms)();
        let mut fields = Map::new();
        fields.insert(
            "image".into(),
            image.map(|path| path.display().to_string()).into(),
        );
        self.tally.counts(&mut fields, now);
        self.cancel_reported = true;
        self.write("batch_cancelled", fields);
    }
}

/// The batch's stream; clones share it, so a document's sinks can write to it.
#[derive(Clone)]
pub struct BatchEvents {
    stream: Arc<Mutex<Stream>>,
}

impl std::fmt::Debug for BatchEvents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchEvents").finish_non_exhaustive()
    }
}

impl BatchEvents {
    /// Open `destination`, truncating a file there, or stderr for `-`.
    ///
    /// # Errors
    ///
    /// [`FocrError::Usage`] when the file cannot be created, before any work.
    pub fn open(destination: &Path) -> FocrResult<Self> {
        let out = if destination == Path::new(STDERR_DESTINATION) {
            Destination::Stderr
        } else {
            let file = File::create(destination).map_err(|err| {
                FocrError::Usage(format!(
                    "--progress-events cannot write {}: {err}",
                    destination.display()
                ))
            })?;
            Destination::Writer(Box::new(file))
        };
        Ok(Self::new(out, None))
    }

    fn new(out: Destination, clock: Option<Box<dyn Fn() -> u64 + Send>>) -> Self {
        let now_ms = clock.unwrap_or_else(|| {
            let started = Instant::now();
            Box::new(move || u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX))
        });
        Self {
            stream: Arc::new(Mutex::new(Stream {
                out,
                now_ms,
                tally: Tally::default(),
                started: false,
                cancel_reported: false,
                write_failed: false,
            })),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Stream> {
        self.stream.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Report the resolved run, before the first item.
    pub fn start(&self, run: &BatchRun<'_>) {
        let mut stream = self.lock();
        stream.tally.total = run.total;
        stream.started = true;
        let Value::Object(fields) = json!({
            "mode": match run.mode {
                BatchMode::Images => "images",
                BatchMode::Document => "document",
            },
            "independent_images": run.mode == BatchMode::Images,
            "total": run.total,
            "threads": run.threads,
            "model_path": run.model_path.display().to_string(),
            "model_id": run.model_id,
            "backend": run.backend,
            "decode_mode": run.decode_mode,
        }) else {
            unreachable!("an object literal")
        };
        stream.write("batch_start", fields);
    }

    /// Report image `ordinal` (1-based) of an independent batch. A cancelled
    /// image did not end, so it is `batch_cancelled`, not an error.
    pub fn image_ended(&self, ordinal: usize, image: &Path, outcome: &FocrResult<String>) {
        let mut stream = self.lock();
        if matches!(outcome, Err(FocrError::Cancelled)) {
            stream.cancelled(Some(image));
            return;
        }
        let now = (stream.now_ms)();
        let elapsed = stream.tally.ended(outcome.is_ok(), now);
        let mut fields = Map::new();
        fields.insert("ordinal".into(), ordinal.into());
        fields.insert("image".into(), image.display().to_string().into());
        stream.tally.progress(&mut fields, now, elapsed);
        let event = match outcome {
            Ok(_) => "image_complete",
            Err(err) => {
                fields.insert("error".into(), err.to_string().into());
                fields.insert("exit_code".into(), err.exit_code().into());
                "image_error"
            }
        };
        stream.write(event, fields);
    }

    /// The sink that reports a `--multi-page` document's pages as they close.
    /// `images` are the batch's inputs, in the document's order.
    pub fn document_sink(&self, images: Vec<PathBuf>) -> crate::DocumentSink {
        let events = self.clone();
        Box::new(move |event: &DocumentEvent<'_>| {
            let DocumentEvent::Page(streamed) = event else {
                return;
            };
            let page = streamed.page;
            let source_page = page.map(|page| page.source_page);
            let image = source_page.and_then(|n| images.get(n - 1));
            let mut stream = events.lock();
            let now = (stream.now_ms)();
            let elapsed = stream.tally.ended(true, now);
            let mut fields = Map::new();
            fields.insert("ordinal".into(), streamed.index.into());
            fields.insert("source_page".into(), source_page.into());
            fields.insert(
                "image".into(),
                image.map(|path| path.display().to_string()).into(),
            );
            fields.insert("window".into(), page.and_then(|page| page.window).into());
            stream.tally.progress(&mut fields, now, elapsed);
            stream.write("page_complete", fields);
        })
    }

    /// The sink that reports a `--multi-page` input that did not decode.
    pub fn page_read_sink(&self, images: Vec<PathBuf>) -> Box<dyn FnMut(&PageItem) + Send> {
        let events = self.clone();
        Box::new(move |item: &PageItem| {
            let Err(err) = &item.raster else {
                return;
            };
            let image = images.get(item.source_page - 1);
            let mut stream = events.lock();
            let now = (stream.now_ms)();
            let elapsed = stream.tally.ended(false, now);
            let mut fields = Map::new();
            fields.insert("source_page".into(), item.source_page.into());
            fields.insert(
                "image".into(),
                image.map(|path| path.display().to_string()).into(),
            );
            fields.insert("error".into(), err.to_string().into());
            stream.tally.progress(&mut fields, now, elapsed);
            stream.write("page_skipped", fields);
        })
    }

    /// Report the run's end. A run stopped before `batch_start` writes nothing:
    /// its refusal is on stderr and in its exit code.
    pub fn end(&self, outcome: &FocrResult<()>) {
        let mut stream = self.lock();
        if !stream.started {
            return;
        }
        if matches!(outcome, Err(FocrError::Cancelled)) && !stream.cancel_reported {
            stream.cancelled(None);
        }
        let now = (stream.now_ms)();
        let (name, exit_code, error) = match outcome {
            Ok(()) => ("completed", 0, None),
            Err(FocrError::Cancelled) => ("cancelled", FocrError::Cancelled.exit_code(), None),
            Err(err) => ("failed", err.exit_code(), Some(err.to_string())),
        };
        let mut fields = Map::new();
        fields.insert("outcome".into(), name.into());
        fields.insert("exit_code".into(), exit_code.into());
        fields.insert("error".into(), error.into());
        stream.tally.counts(&mut fields, now);
        fields.insert(
            "pages_per_hour".into(),
            stream.tally.pages_per_hour(now).into(),
        );
        stream.write("batch_end", fields);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    #[derive(Clone, Default)]
    struct Lines(Arc<Mutex<Vec<u8>>>);

    impl Write for Lines {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    struct Failing;

    impl Write for Failing {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("disk full"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    struct Harness {
        events: BatchEvents,
        lines: Lines,
        clock: Arc<AtomicU64>,
    }

    impl Harness {
        fn new() -> Self {
            let lines = Lines::default();
            let clock = Arc::new(AtomicU64::new(0));
            let reader = Arc::clone(&clock);
            let events = BatchEvents::new(
                Destination::Writer(Box::new(lines.clone())),
                Some(Box::new(move || reader.load(Ordering::SeqCst))),
            );
            Self {
                events,
                lines,
                clock,
            }
        }

        fn at(&self, ms: u64) {
            self.clock.store(ms, Ordering::SeqCst);
        }

        fn start(&self, mode: BatchMode, total: usize) {
            self.events.start(&BatchRun {
                mode,
                total,
                threads: 12,
                model_path: Path::new("/models/unlimited-ocr.int8.focrq"),
                model_id: Some("unlimited-ocr"),
                backend: "neon",
                decode_mode: "conservative_recipe",
            });
        }

        fn lines(&self) -> Vec<Value> {
            let bytes = self.lines.0.lock().unwrap();
            String::from_utf8_lossy(&bytes)
                .lines()
                .map(|line| serde_json::from_str(line).expect("one object per line"))
                .collect()
        }
    }

    /// Every line carries exactly the fields [`schema`] declares for it.
    fn assert_matches_schema(lines: &[Value]) {
        let schema = schema();
        for line in lines {
            let event = line["event"].as_str().expect("event");
            let declared = &schema["event_fields"][event]["required"];
            let mut declared: Vec<&str> = declared
                .as_array()
                .unwrap_or_else(|| panic!("{event} undeclared"))
                .iter()
                .map(|field| field.as_str().expect("field"))
                .collect();
            declared.sort_unstable();
            let mut keys: Vec<&str> = line
                .as_object()
                .expect("object")
                .keys()
                .map(String::as_str)
                .collect();
            keys.sort_unstable();
            assert_eq!(keys, declared, "{line}");
            assert_eq!(line["schema_version"], BATCH_EVENTS_SCHEMA_VERSION);
        }
    }

    fn kinds(lines: &[Value]) -> Vec<&str> {
        lines.iter().map(|l| l["event"].as_str().unwrap()).collect()
    }

    #[test]
    fn a_batch_reports_each_image_with_a_measured_rate_and_estimate() {
        let run = Harness::new();
        run.start(BatchMode::Images, 3);
        run.at(40_000);
        let page = Ok("body".to_string());
        run.events.image_ended(1, Path::new("a.png"), &page);
        run.at(60_000);
        run.events.image_ended(2, Path::new("b.png"), &page);
        run.at(80_000);
        run.events.image_ended(3, Path::new("c.png"), &page);
        run.events.end(&Ok(()));

        let lines = run.lines();
        assert_matches_schema(&lines);
        assert_eq!(
            kinds(&lines),
            [
                "batch_start",
                "image_complete",
                "image_complete",
                "image_complete",
                "batch_end"
            ]
        );
        assert_eq!(lines[0]["mode"], "images");
        assert_eq!(lines[0]["independent_images"], true);
        assert_eq!(lines[0]["threads"], 12);
        assert_eq!(lines[0]["model_id"], "unlimited-ocr");
        assert_eq!(lines[0]["backend"], "neon");

        assert_eq!(lines[1]["ordinal"], 1);
        assert_eq!(lines[1]["image"], "a.png");
        assert_eq!(lines[1]["elapsed_ms"], 40_000);
        // 40 s a page so far, with 2 left.
        assert_eq!(lines[1]["eta_ms"], 80_000);
        assert_eq!(lines[1]["pages_per_hour"], 90.0);
        // 30 s a page so far, with 1 left.
        assert_eq!(lines[2]["elapsed_ms"], 20_000);
        assert_eq!(lines[2]["eta_ms"], 30_000);
        assert_eq!(lines[3]["eta_ms"], 0);
        assert_eq!(lines[3]["cumulative_ms"], 80_000);

        assert_eq!(lines[4]["outcome"], "completed");
        assert_eq!(lines[4]["exit_code"], 0);
        assert_eq!(lines[4]["ok_images"], 3);
        assert_eq!(lines[4]["pages_per_hour"], 135.0);
    }

    #[test]
    fn a_failed_image_counts_toward_the_estimate_but_not_the_rate() {
        let run = Harness::new();
        run.start(BatchMode::Images, 2);
        run.at(10_000);
        let failed = Err(FocrError::InputDecode("not an image".into()));
        run.events.image_ended(1, Path::new("bad.png"), &failed);
        run.at(40_000);
        run.events
            .image_ended(2, Path::new("good.png"), &Ok(String::new()));
        run.events.end(&Ok(()));

        let lines = run.lines();
        assert_matches_schema(&lines);
        assert_eq!(lines[1]["event"], "image_error");
        assert_eq!(lines[1]["error"], "input decode error: not an image");
        assert_eq!(
            lines[1]["exit_code"],
            FocrError::InputDecode(String::new()).exit_code()
        );
        assert_eq!(lines[1]["failed_images"], 1);
        assert_eq!(lines[1]["ok_images"], 0);
        assert_eq!(lines[1]["pages_per_hour"], 0.0);
        assert_eq!(lines[1]["eta_ms"], 10_000);
        // A failed image does not fail the run.
        assert_eq!(lines[3]["outcome"], "completed");
        assert_eq!(lines[3]["failed_images"], 1);
        assert_eq!(lines[3]["pages_per_hour"], 90.0);
    }

    #[test]
    fn a_cancelled_image_did_not_end() {
        let run = Harness::new();
        run.start(BatchMode::Images, 3);
        run.at(30_000);
        run.events
            .image_ended(1, Path::new("a.png"), &Ok(String::new()));
        run.at(35_000);
        run.events
            .image_ended(2, Path::new("b.png"), &Err(FocrError::Cancelled));
        run.events.end(&Err(FocrError::Cancelled));

        let lines = run.lines();
        assert_matches_schema(&lines);
        assert_eq!(
            kinds(&lines),
            [
                "batch_start",
                "image_complete",
                "batch_cancelled",
                "batch_end"
            ]
        );
        assert_eq!(lines[2]["image"], "b.png");
        assert_eq!(lines[2]["ok_images"], 1);
        assert_eq!(lines[2]["failed_images"], 0);
        assert_eq!(lines[3]["outcome"], "cancelled");
        assert_eq!(lines[3]["exit_code"], FocrError::Cancelled.exit_code());
        assert_eq!(lines[3]["error"], Value::Null);
    }

    #[test]
    fn a_document_cancelled_in_its_pass_reports_no_image() {
        let run = Harness::new();
        run.start(BatchMode::Document, 3);
        run.at(1_000);
        run.events.end(&Err(FocrError::Cancelled));

        let lines = run.lines();
        assert_matches_schema(&lines);
        assert_eq!(lines[0]["independent_images"], false);
        assert_eq!(
            kinds(&lines),
            ["batch_start", "batch_cancelled", "batch_end"]
        );
        assert_eq!(lines[1]["image"], Value::Null);
    }

    #[test]
    fn a_failed_run_ends_with_its_error_and_exit_code() {
        let run = Harness::new();
        run.start(BatchMode::Document, 3);
        run.at(1_000);
        let failed = FocrError::Timeout("stage budget".into());
        let code = failed.exit_code();
        run.events.end(&Err(failed));

        let lines = run.lines();
        assert_matches_schema(&lines);
        assert_eq!(lines[1]["outcome"], "failed");
        assert_eq!(lines[1]["exit_code"], code);
        assert!(lines[1]["error"].as_str().unwrap().contains("stage budget"));
    }

    #[test]
    fn a_run_refused_before_it_started_writes_nothing() {
        let run = Harness::new();
        run.events.end(&Err(FocrError::Usage("refused".into())));
        assert!(run.lines().is_empty());
    }

    #[test]
    fn a_failing_destination_does_not_fail_the_batch() {
        let events = BatchEvents::new(Destination::Writer(Box::new(Failing)), Some(Box::new(|| 0)));
        events.start(&BatchRun {
            mode: BatchMode::Images,
            total: 1,
            threads: 1,
            model_path: Path::new("m"),
            model_id: None,
            backend: "scalar",
            decode_mode: "conservative_recipe",
        });
        assert!(events.lock().write_failed);
        // Later lines are dropped, not retried.
        events.image_ended(1, Path::new("a.png"), &Ok(String::new()));
        events.end(&Ok(()));
    }

    #[test]
    fn a_destination_that_cannot_be_created_is_a_usage_error() {
        let err = BatchEvents::open(Path::new("/nonexistent/dir/events.ndjson")).unwrap_err();
        assert!(matches!(err, FocrError::Usage(_)), "{err}");
        assert!(err.to_string().contains("--progress-events"), "{err}");
    }

    #[test]
    fn the_schema_declares_every_event_kind() {
        let schema = schema();
        let declared = schema["event_fields"].as_object().expect("event_fields");
        for kind in BATCH_EVENT_KINDS {
            assert!(declared.contains_key(*kind), "{kind}");
        }
        assert_eq!(declared.len(), BATCH_EVENT_KINDS.len());
    }
}
