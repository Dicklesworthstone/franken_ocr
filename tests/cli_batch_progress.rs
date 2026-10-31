//! `focr ocr-batch --progress-events` on the real binary (bd-mgd3).
//!
//! The no-weight tests run everywhere: images that fail (no model) and a forced
//! cancellation, compared with a scrubbed golden. With a v1 artifact in
//! `FOCR_BATCH_PROGRESS_MODEL`, the armed test reads real pages, a real broken
//! image and a real Ctrl+C.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

const MODEL_ENV: &str = "FOCR_BATCH_PROGRESS_MODEL";
const CORPUS_ENV: &str = "FOCR_CORPUS_DIR";
const GOLDEN: &str = "tests/fixtures/batch_progress_cancelled.ndjson";
const NO_MODEL: &str = "/nonexistent/model.focrq";
const EXIT_MODEL_NOT_FOUND: i32 = 3;
const EXIT_CANCELLED: i32 = 6;
const FORCE_ERROR_ENV: &str = "FOCR_TEST_FORCE_ERROR";

fn focr() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_focr"));
    command
        .env_remove(MODEL_ENV)
        .env_remove(FORCE_ERROR_ENV)
        .env_remove("FOCR_BATCH_SPINE")
        .env("FOCR_NO_RESIDENT", "1");
    command
}

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("focr-batch-events-{test}-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn events_at(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .expect("events file")
        .lines()
        .map(|line| serde_json::from_str(line).expect("one object per line"))
        .collect()
}

fn kinds(events: &[Value]) -> Vec<&str> {
    events
        .iter()
        .map(|event| event["event"].as_str().expect("event"))
        .collect()
}

/// Every line carries exactly what `focr robot schema` declares for its event.
fn assert_declared(events: &[Value]) {
    let out = focr()
        .args(["robot", "schema"])
        .output()
        .expect("robot schema");
    let schema: Value = serde_json::from_slice(&out.stdout).expect("schema JSON");
    let contract = &schema["batch_progress"];
    for event in events {
        let kind = event["event"].as_str().expect("event");
        let mut declared: Vec<&str> = contract["event_fields"][kind]["required"]
            .as_array()
            .unwrap_or_else(|| panic!("{kind} undeclared"))
            .iter()
            .map(|field| field.as_str().expect("field"))
            .collect();
        declared.sort_unstable();
        let mut keys: Vec<&str> = event
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(keys, declared, "{event}");
        assert_eq!(event["schema_version"], contract["schema_version"]);
    }
}

/// `--json` stdout is the one final object, nothing else.
fn final_object(out: &Output) -> Value {
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(stdout.lines().count(), 1, "stdout: {stdout}");
    serde_json::from_str(&stdout).expect("stdout is one JSON object")
}

/// `value` with what varies between runs replaced by what it means.
fn scrub(value: &Value, key: &str) -> Value {
    match (key, value) {
        ("cumulative_ms" | "elapsed_ms", _) => "<ms>".into(),
        ("threads", _) => "<threads>".into(),
        ("backend", _) => "<backend>".into(),
        ("pages_per_hour", _) => "<rate>".into(),
        ("eta_ms", Value::Number(_)) => "<number>".into(),
        ("image", Value::String(path)) => Path::new(path)
            .file_name()
            .map_or_else(String::new, |name| name.to_string_lossy().into_owned())
            .into(),
        (_, Value::Object(fields)) => Value::Object(
            fields
                .iter()
                .map(|(field, inner)| (field.clone(), scrub(inner, field)))
                .collect(),
        ),
        _ => value.clone(),
    }
}

/// A failing image is an `image_error`, a forced Ctrl+C ends the run at the
/// image it hit, and `--json` stdout stays one object holding what ended.
#[test]
fn a_cancelled_batch_reports_what_ended_and_stops() {
    let dir = scratch("cancelled");
    let path = dir.join("events.ndjson");
    let out = focr()
        .args(["ocr-batch", "a.png", "b.png", "c.png", "--json"])
        .args(["--model", NO_MODEL, "--progress-events"])
        .arg(&path)
        .env(FORCE_ERROR_ENV, "cancelled@2")
        .output()
        .expect("focr");
    assert_eq!(
        out.status.code(),
        Some(EXIT_CANCELLED),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let events = events_at(&path);
    assert_declared(&events);
    let scrubbed: String = events
        .iter()
        .map(|event| format!("{}\n", scrub(event, "")))
        .collect();
    if std::env::var_os("UPDATE_GOLDENS").is_some() {
        fs::write(GOLDEN, &scrubbed).expect("golden");
    }
    let golden = fs::read_to_string(GOLDEN).expect("golden");
    assert_eq!(scrubbed, golden);

    let value = final_object(&out);
    assert_eq!(value["stopped"], "cancelled");
    assert_eq!(value["count"], 3);
    let results = value["results"].as_array().expect("results");
    assert_eq!(results.len(), 1, "only the image that ended: {value}");
    assert_eq!(results[0]["image"], "a.png");
    assert_eq!(results[0]["ok"], false);
}

/// Without the flag, a Ctrl+C ends the run the same way.
#[test]
fn a_cancelled_batch_without_the_stream_still_stops() {
    let out = focr()
        .args(["ocr-batch", "a.png", "b.png", "c.png", "--json"])
        .args(["--model", NO_MODEL])
        .env(FORCE_ERROR_ENV, "cancelled@1")
        .output()
        .expect("focr");
    assert_eq!(out.status.code(), Some(EXIT_CANCELLED));
    let value = final_object(&out);
    assert_eq!(value["stopped"], "cancelled");
    assert_eq!(value["results"].as_array().map(Vec::len), Some(0));
}

/// A batch that read no image fails with the first failure's code, once every
/// image has been reported, and `-` leaves stdout pure.
#[test]
fn image_errors_go_to_stderr_with_stdout_pure() {
    let out = focr()
        .args(["ocr-batch", "a.png", "b.png", "--json"])
        .args(["--model", NO_MODEL, "--progress-events", "-"])
        .output()
        .expect("focr");
    assert_eq!(out.status.code(), Some(EXIT_MODEL_NOT_FOUND));
    let value = final_object(&out);
    assert_eq!(value["results"].as_array().map(Vec::len), Some(2));

    let stderr = String::from_utf8_lossy(&out.stderr);
    let events: Vec<Value> = stderr
        .lines()
        .filter(|line| line.starts_with('{'))
        .map(|line| serde_json::from_str(line).expect("one object per line"))
        .collect();
    assert_declared(&events);
    assert_eq!(
        kinds(&events),
        ["batch_start", "image_error", "image_error", "batch_end"]
    );
    assert_eq!(events[0]["model_id"], Value::Null);
    assert_eq!(events[0]["model_path"], NO_MODEL);
    assert!(events[0]["threads"].as_u64().is_some_and(|n| n > 0));
    assert_eq!(events[3]["outcome"], "failed");
    assert_eq!(events[3]["exit_code"], EXIT_MODEL_NOT_FOUND);
    assert_eq!(events[3]["failed_images"], 2);
    assert_eq!(events[3]["ok_images"], 0);
}

/// `--fail-fast` ends the run at the first image that fails, with its code.
#[test]
fn fail_fast_stops_at_the_first_failed_image() {
    let dir = scratch("fail-fast");
    let path = dir.join("events.ndjson");
    let out = focr()
        .args([
            "ocr-batch",
            "a.png",
            "b.png",
            "c.png",
            "--json",
            "--fail-fast",
        ])
        .args(["--model", NO_MODEL, "--progress-events"])
        .arg(&path)
        .output()
        .expect("focr");
    assert_eq!(out.status.code(), Some(EXIT_MODEL_NOT_FOUND));
    let value = final_object(&out);
    assert_eq!(value["stopped"], "image_failed");
    assert_eq!(value["results"].as_array().map(Vec::len), Some(1));

    let events = events_at(&path);
    assert_declared(&events);
    assert_eq!(kinds(&events), ["batch_start", "image_error", "batch_end"]);
    assert_eq!(events[2]["outcome"], "failed");
    assert!(events[2]["error"].as_str().is_some());
}

/// `--inputs-from` runs after the named images, in order, from a file or stdin.
#[test]
fn a_batch_reads_its_inputs_from_a_list() {
    let dir = scratch("inputs-from");
    let list = dir.join("pages.txt");
    fs::write(&list, "\u{feff}b.png\r\n\n  spaced name.png\nd.png").expect("list");
    let out = focr()
        .args(["ocr-batch", "a.png", "--json", "--model", NO_MODEL])
        .arg("--inputs-from")
        .arg(&list)
        .output()
        .expect("focr");
    let value = final_object(&out);
    let read: Vec<&str> = value["results"]
        .as_array()
        .expect("results")
        .iter()
        .map(|result| result["image"].as_str().expect("image"))
        .collect();
    assert_eq!(read, ["a.png", "b.png", "  spaced name.png", "d.png"]);

    let mut child = focr()
        .args([
            "ocr-batch",
            "--json",
            "--model",
            NO_MODEL,
            "--inputs-from",
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("focr");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(b"e.png\n")
        .expect("write");
    let out = child.wait_with_output().expect("focr");
    let value = final_object(&out);
    assert_eq!(value["results"][0]["image"], "e.png");
}

#[test]
fn a_batch_with_no_inputs_is_refused() {
    let out = focr().arg("ocr-batch").output().expect("focr");
    assert_eq!(out.status.code(), Some(2), "named or listed, one is needed");

    let dir = scratch("empty-list");
    let list = dir.join("pages.txt");
    fs::write(&list, "\n\n").expect("list");
    let out = focr()
        .args(["ocr-batch", "--inputs-from"])
        .arg(&list)
        .output()
        .expect("focr");
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--inputs-from"));

    let out = focr()
        .args(["ocr-batch", "--inputs-from", "/nonexistent/pages.txt"])
        .output()
        .expect("focr");
    assert_eq!(out.status.code(), Some(2));
}

/// Two inputs for one `--output-dir` file are refused before any work, and the
/// directory is not created.
#[test]
fn output_dir_refuses_two_inputs_for_one_file() {
    let dir = scratch("output-collision").join("out");
    let out = focr()
        .args([
            "ocr-batch",
            "one/page.png",
            "two/PAGE.jpg",
            "--model",
            NO_MODEL,
        ])
        .arg("--output-dir")
        .arg(&dir)
        .output()
        .expect("focr");
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("one/page.png") && stderr.contains("two/PAGE.jpg"),
        "{stderr}"
    );
    assert!(!dir.exists());
}

#[test]
fn image_by_image_flags_refuse_one_document() {
    for flag in [&["--output-dir", "out"][..], &["--fail-fast"][..]] {
        let out = focr()
            .args(["ocr-batch", "a.png", "b.png", "--multi-page"])
            .args(flag)
            .output()
            .expect("focr");
        assert_eq!(out.status.code(), Some(2), "{flag:?}");
    }
}

#[test]
fn the_stream_refuses_a_destination_it_cannot_write_before_any_work() {
    let out = focr()
        .args(["ocr-batch", "a.png", "--progress-events"])
        .arg("/nonexistent/dir/events.ndjson")
        .output()
        .expect("focr");
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--progress-events"));
}

#[test]
fn the_spine_has_no_per_image_progress_to_report() {
    let dir = scratch("spine");
    let path = dir.join("events.ndjson");
    let out = focr()
        .args([
            "ocr-batch",
            "a.png",
            "--model",
            NO_MODEL,
            "--progress-events",
        ])
        .arg(&path)
        .env("FOCR_BATCH_SPINE", "1")
        .output()
        .expect("focr");
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("FOCR_BATCH_SPINE"));
    assert!(!path.exists(), "refused before the stream opened");
}

/// Real pages, a real broken image, and a real Ctrl+C after the first image.
#[test]
fn a_real_batch_reports_every_image_and_a_real_interrupt() {
    let Some(model) = std::env::var_os(MODEL_ENV) else {
        eprintln!("skip_no_model: {MODEL_ENV} unset");
        return;
    };
    let corpus = std::env::var_os(CORPUS_ENV).map_or_else(
        || {
            PathBuf::from(std::env::var_os("HOME").expect("HOME"))
                .join(".cache/franken_ocr/corpus/pages")
        },
        PathBuf::from,
    );
    let page = |name: &str| corpus.join(name);
    let dir = scratch("real");
    let broken = dir.join("broken.png");
    fs::write(&broken, b"not an image").expect("broken image");

    let path = dir.join("complete.ndjson");
    let saved = dir.join("markdown");
    let out = focr()
        .args(["ocr-batch", "--json", "--model"])
        .arg(&model)
        .arg(page("page_0009.png"))
        .arg(&broken)
        .arg(page("page_0014.png"))
        .arg("--progress-events")
        .arg(&path)
        .arg("--output-dir")
        .arg(&saved)
        .output()
        .expect("focr");
    assert_eq!(out.status.code(), Some(0));
    let value = final_object(&out);
    let results = value["results"].as_array().expect("results");
    assert_eq!(results.len(), 3);
    let written = saved.join("page_0014.md");
    assert_eq!(results[2]["output"].as_str(), written.to_str());
    assert_eq!(
        Some(fs::read_to_string(&written).expect("saved").as_str()),
        results[2]["markdown"].as_str()
    );
    assert!(
        !saved.join("broken.md").exists(),
        "no file for a failed image"
    );

    let events = events_at(&path);
    assert_declared(&events);
    assert_eq!(
        kinds(&events),
        [
            "batch_start",
            "image_complete",
            "image_error",
            "image_complete",
            "batch_end"
        ]
    );
    assert_eq!(events[0]["model_id"], "unlimited-ocr");
    assert_eq!(events[1]["ordinal"], 1);
    assert!(events[1]["eta_ms"].as_u64().is_some_and(|ms| ms > 0));
    assert!(
        events[1]["pages_per_hour"]
            .as_f64()
            .is_some_and(|r| r > 0.0)
    );
    assert_eq!(events[2]["image"].as_str(), broken.to_str());
    assert_eq!(events[3]["eta_ms"], 0);
    assert_eq!(events[4]["ok_images"], 2);
    assert_eq!(events[4]["failed_images"], 1);
    assert_eq!(results[0]["ok"], true);
    assert_eq!(results[1]["ok"], false);

    // One document: its pages are not independent images, and an input that
    // does not decode is left out of the pass.
    let path = dir.join("document.ndjson");
    let out = focr()
        .args(["ocr-batch", "--multi-page", "--json", "--model"])
        .arg(&model)
        .arg(page("page_0009.png"))
        .arg(&broken)
        .arg(page("page_0014.png"))
        .arg("--progress-events")
        .arg(&path)
        .output()
        .expect("focr");
    assert_eq!(out.status.code(), Some(0));
    final_object(&out);
    let events = events_at(&path);
    assert_declared(&events);
    let names = kinds(&events);
    assert_eq!(events[0]["mode"], "document");
    assert_eq!(events[0]["independent_images"], false);
    assert_eq!(
        names.iter().filter(|name| **name == "page_skipped").count(),
        1
    );
    assert!(names.contains(&"page_complete"), "{names:?}");
    assert_eq!(names.last(), Some(&"batch_end"));
    assert_eq!(events[events.len() - 1]["outcome"], "completed");

    let path = dir.join("interrupted.ndjson");
    let child = focr()
        .args(["ocr-batch", "--json", "--model"])
        .arg(&model)
        .arg(page("page_0009.png"))
        .arg(page("page_0014.png"))
        .arg(page("page_0019.png"))
        .arg("--progress-events")
        .arg(&path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("focr");
    let deadline = Instant::now() + Duration::from_secs(600);
    while !fs::read_to_string(&path).is_ok_and(|text| text.contains("image_complete")) {
        assert!(Instant::now() < deadline, "no image ended");
        std::thread::sleep(Duration::from_millis(50));
    }
    let status = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .expect("kill");
    assert!(status.success());
    let out = child.wait_with_output().expect("focr");
    assert_eq!(out.status.code(), Some(EXIT_CANCELLED));
    let value = final_object(&out);
    assert_eq!(value["stopped"], "cancelled");

    let events = events_at(&path);
    assert_declared(&events);
    let names = kinds(&events);
    assert_eq!(names[names.len() - 2..], ["batch_cancelled", "batch_end"]);
    assert_eq!(events[events.len() - 1]["outcome"], "cancelled");
    assert_eq!(
        events[events.len() - 1]["ok_images"],
        names
            .iter()
            .filter(|name| **name == "image_complete")
            .count()
    );
}
