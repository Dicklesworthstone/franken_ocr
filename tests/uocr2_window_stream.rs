//! `uocr2_window_stream` — U2-D5's robot stream on real Unlimited-OCR-2 runs.
//!
//! Each test runs the `focr` binary and validates every event field by field
//! against the frozen `event_fields`. A windowed run streams `page` events
//! naming their window, a `window` event after each window's pages, and the
//! run's counts on `run_complete`. `--page-window off` streams no window, and
//! past the page limit it runs in one pass instead of being refused.
//!
//! **Gating** (LOGGING_AND_E2E.md §4): `FOCR_WINDOW_STREAM_MODEL` names the v2
//! artifact; without it each test logs `skip_no_model` and passes. Pages come
//! from `FOCR_CORPUS_DIR` (default `~/.cache/franken_ocr/corpus/pages`), PNGs
//! in name order. Pages are capped at 32 tokens, because the stream is under
//! test, not the text. Decoding wants an optimized profile:
//!
//! ```text
//! FOCR_WINDOW_STREAM_MODEL=… cargo test --locked --profile release-perf --test uocr2_window_stream
//! ```

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;

#[path = "support/parity_harness.rs"]
mod parity_harness;

use franken_ocr::native_engine::UOCR2_UNWINDOWED_PAGE_LIMIT;
use parity_harness::Logger;
use serde_json::{Value, json};

const TEST: &str = "uocr2_window_stream";

/// One decode at a time: each takes most of the machine.
static SERIAL: Mutex<()> = Mutex::new(());

fn corpus_pages(take: usize) -> Vec<PathBuf> {
    let dir = match std::env::var_os("FOCR_CORPUS_DIR") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => PathBuf::from(std::env::var_os("HOME").expect("HOME"))
            .join(".cache/franken_ocr/corpus/pages"),
    };
    let mut pages: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "png"))
        .collect();
    pages.sort();
    assert!(!pages.is_empty(), "{} holds no pages", dir.display());
    // A document longer than the corpus repeats it.
    pages.iter().cycle().take(take).cloned().collect()
}

/// A fresh root for one test, holding a `pages` directory of `count` pages.
fn document(name: &str, count: usize) -> PathBuf {
    let root = std::env::temp_dir().join(format!("{TEST}-{name}-{}", std::process::id()));
    let (home, pages) = (root.join("home"), root.join("pages"));
    std::fs::create_dir_all(&home).expect("a home");
    std::fs::create_dir_all(&pages).expect("a pages directory");
    for (k, page) in corpus_pages(count).iter().enumerate() {
        std::fs::copy(page, pages.join(format!("{k:04}.png"))).expect("copy a page");
    }
    root
}

struct Run {
    exit: Option<i32>,
    events: Vec<Value>,
}

impl Run {
    fn of(&self, kind: &str) -> Vec<&Value> {
        self.events.iter().filter(|e| e["event"] == kind).collect()
    }

    fn last(&self) -> &Value {
        self.events.last().expect("a terminal event")
    }
}

/// `event_fields`, from the frozen schema fixture.
fn event_fields() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/robot_schema_v1.json");
    let schema: Value =
        serde_json::from_slice(&std::fs::read(path).expect("read the schema")).expect("JSON");
    schema["event_fields"].clone()
}

/// `event` carries every field its kind requires and nothing unlisted.
fn validate(event: &Value, fields: &Value) {
    let kind = event["event"].as_str().expect("an event kind");
    let spec = &fields[kind];
    let names = |key: &str| -> Vec<String> {
        spec[key]
            .as_array()
            .unwrap_or_else(|| panic!("{kind}: no {key}"))
            .iter()
            .map(|name| name.as_str().expect("a field name").to_owned())
            .collect()
    };
    let (required, optional) = (names("required"), names("optional"));
    let object = event.as_object().expect("an object");
    for field in &required {
        assert!(object.contains_key(field), "{kind} lacks {field}: {event}");
    }
    for field in object.keys() {
        assert!(
            required.contains(field) || optional.contains(field),
            "{kind} field {field} is undeclared: {event}"
        );
    }
}

/// `focr ocr <root>/pages --multi-page --robot <args>` on `artifact`.
fn focr(root: &Path, artifact: &Path, args: &[&str]) -> Run {
    focr_with(root, artifact, args, &[])
}

/// [`focr`] with `envs` set.
fn focr_with(root: &Path, artifact: &Path, args: &[&str], envs: &[(&str, &str)]) -> Run {
    let output = Command::new(env!("CARGO_BIN_EXE_focr"))
        .arg("ocr")
        .arg(root.join("pages"))
        .args(["--multi-page", "--robot", "--max-page-tokens", "32"])
        .arg("--model")
        .arg(artifact)
        .args(args)
        .env("HOME", root.join("home"))
        .env("FOCR_NO_RESIDENT", "1")
        .envs(envs.iter().copied())
        .stderr(Stdio::inherit())
        .output()
        .expect("run focr");
    let events: Vec<Value> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| serde_json::from_str(line).expect("an NDJSON event"))
        .collect();
    let fields = event_fields();
    for event in &events {
        validate(event, &fields);
    }
    Run {
        exit: output.status.code(),
        events,
    }
}

fn model() -> Result<PathBuf, String> {
    std::env::var_os("FOCR_WINDOW_STREAM_MODEL")
        .map(PathBuf::from)
        .ok_or_else(|| "FOCR_WINDOW_STREAM_MODEL unset: point it at the v2 artifact".to_owned())
}

#[test]
fn a_windowed_run_streams_its_windows_after_their_pages() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut log = Logger::new(TEST, "windowed");
    let artifact = match model() {
        Ok(artifact) => artifact,
        Err(why) => return log.skip_no_model(&why),
    };
    let root = document("windowed", 5);
    let run = focr(&root, &artifact, &["--page-window", "2"]);
    assert_eq!(run.exit, Some(0), "{:?}", run.last());

    let stream: Vec<(&str, u64)> = run
        .events
        .iter()
        .filter_map(|event| match event["event"].as_str() {
            Some("page") => Some(("page", event["window"].as_u64().expect("a window"))),
            Some("window") => Some(("window", event["index"].as_u64().expect("an index"))),
            _ => None,
        })
        .collect();
    assert_eq!(
        stream,
        [
            ("page", 0),
            ("page", 0),
            ("window", 0),
            ("page", 1),
            ("page", 1),
            ("window", 1),
            ("page", 2),
            ("window", 2),
        ]
    );

    let windows = run.of("window");
    let spans: Vec<(&Value, &Value)> = windows
        .iter()
        .map(|window| (&window["pages"], &window["lookahead"]))
        .collect();
    assert_eq!(
        spans,
        [
            (&json!([0, 2]), &json!(1)),
            (&json!([2, 4]), &json!(1)),
            (&json!([4, 5]), &json!(0)),
        ]
    );
    assert!(windows.iter().all(|window| window["eta_ms"].is_u64()));
    assert_eq!(windows[2]["eta_ms"], 0, "nothing is left after the last");

    let last = run.last();
    assert_eq!(last["event"], "run_complete");
    assert_eq!(last["pages_out"], 5);
    assert_eq!(last["pages_skipped"], 0);
    assert_eq!(last["windows"], 3);
    assert!(last["extra_markers"].is_u64());
    log.assertion("windowed_stream", true);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn an_unwindowed_run_streams_no_window() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut log = Logger::new(TEST, "unwindowed");
    let artifact = match model() {
        Ok(artifact) => artifact,
        Err(why) => return log.skip_no_model(&why),
    };
    let root = document("unwindowed", 3);
    let run = focr(&root, &artifact, &["--page-window", "off"]);
    assert_eq!(run.exit, Some(0), "{:?}", run.last());
    let pages = run.of("page");
    assert_eq!(pages.len(), 3);
    assert!(pages.iter().all(|page| page["window"].is_null()));
    assert!(run.of("window").is_empty());
    let last = run.last();
    assert_eq!(
        (&last["pages_out"], &last["windows"]),
        (&json!(3), &json!(0))
    );
    log.assertion("unwindowed_stream", true);
    let _ = std::fs::remove_dir_all(&root);
}

/// bd-yrr1: under `FOCR_REREAD_CAPPED_PAGES`, each page the 32-token cap
/// closed is read again alone, reaches the cap again and keeps its text. The
/// stream and the markdown are the run's without it, the decode's warnings
/// stay the document's, and `run_complete` adds one `v2_page_reread` warning
/// that lists every capped page as kept.
#[test]
fn a_capped_page_read_again_that_caps_again_keeps_its_text() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut log = Logger::new(TEST, "reread_capped");
    let artifact = match model() {
        Ok(artifact) => artifact,
        Err(why) => return log.skip_no_model(&why),
    };
    let root = document("reread", 3);
    let args = ["--page-window", "2"];
    let plain = focr(&root, &artifact, &args);
    let reread = focr_with(
        &root,
        &artifact,
        &args,
        &[("FOCR_REREAD_CAPPED_PAGES", "1")],
    );
    assert_eq!(plain.exit, Some(0), "{:?}", plain.last());
    assert_eq!(reread.exit, Some(0), "{:?}", reread.last());

    let capped: Vec<u64> = plain
        .of("page")
        .iter()
        .filter(|page| page["finish"] == "page_cap")
        .map(|page| page["page"].as_u64().expect("a page index"))
        .collect();
    assert!(!capped.is_empty(), "a 32-token cap closes a corpus page");
    assert_eq!(plain.of("page"), reread.of("page"), "nothing streams again");

    let (plain_end, reread_end) = (plain.last(), reread.last());
    assert_eq!(reread_end["event"], "run_complete");
    assert_eq!(plain_end["markdown"], reread_end["markdown"]);
    assert_eq!(plain_end["finish_reason"], reread_end["finish_reason"]);
    let warnings =
        |end: &Value| -> Vec<Value> { end["warnings"].as_array().cloned().unwrap_or_default() };
    let (decode, rereads): (Vec<Value>, Vec<Value>) = warnings(reread_end)
        .into_iter()
        .partition(|warning| warning["kind"] != "v2_page_reread");
    assert_eq!(decode, warnings(plain_end), "the decode's warnings stay");
    assert_eq!(rereads.len(), 1, "{rereads:?}");
    assert_eq!(rereads[0]["pages"], json!([]));
    assert_eq!(rereads[0]["kept"], json!(capped));
    log.assertion("reread_capped", true);
    let _ = std::fs::remove_dir_all(&root);
}

/// The bead's `--page-window off` case: a document one page over the limit
/// runs in one pass, so it finishes, or stops at the position limit with its
/// pages out and exit 5.
#[test]
fn page_window_off_runs_past_the_page_limit() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut log = Logger::new(TEST, "off_past_limit");
    let artifact = match model() {
        Ok(artifact) => artifact,
        Err(why) => return log.skip_no_model(&why),
    };
    let pages = UOCR2_UNWINDOWED_PAGE_LIMIT + 1;
    let root = document("off", pages);
    let run = focr(&root, &artifact, &["--page-window", "off"]);
    assert!(run.of("window").is_empty(), "it ran in one pass");
    let last = run.last();
    match run.exit {
        Some(0) => {
            assert_eq!(last["event"], "run_complete");
            assert_eq!(last["pages_out"], pages);
            assert_eq!(last["windows"], 0);
        }
        Some(5) => {
            assert_eq!(last["event"], "run_error");
            assert_eq!(last["finish_reason"], "length");
            assert!(last["pages_out"].as_u64().is_some_and(|out| out > 0));
        }
        exit => panic!("exit {exit:?}: {last}"),
    }
    log.assertion("off_past_limit", true);
    let _ = std::fs::remove_dir_all(&root);
}
