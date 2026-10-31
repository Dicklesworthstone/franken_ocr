//! `uocr2_resume` — U2-D4's proof that a resumed document is byte-identical
//! to an uninterrupted one.
//!
//! A windowed Unlimited-OCR-2 document is parsed whole, then parsed again and
//! cancelled after window 1 and after window 2, each time resumed from its
//! journal and checkpoint. The markdown, every page result, every generated
//! id, every window's record and the document's counters must equal the
//! uninterrupted run's. A differing window is refused by name, and a recompute
//! finishes labelled as what it is.
//!
//! The CLI's side runs the `focr` binary: a run interrupted by SIGINT exits 6
//! with its pages written and its files in place, `focr doctor` reports a
//! stdout run with its command, and the same command finishes the document
//! with the bytes and figure files of an uninterrupted run.
//!
//! `request_shutdown()` flips a process-global flag, so this test owns its
//! process.
//!
//! **Gating** (LOGGING_AND_E2E.md §4): `FOCR_RESUME_MODEL` names the v2
//! artifact; without it each test logs `skip_no_model` and passes. Pages come
//! from `FOCR_CORPUS_DIR` (default `~/.cache/franken_ocr/corpus/pages`), PNGs
//! in name order: `FOCR_RESUME_PAGES` of them (default 5) in windows of
//! `FOCR_RESUME_WINDOW` (default 2) for the library test, and 12 in the
//! CLI's windows of 10 for the binary's. Decoding wants an optimized profile:
//!
//! ```text
//! FOCR_RESUME_MODEL=… cargo test --locked --profile release-perf --test uocr2_resume
//! ```
//! The bead's full library run is 25 pages in windows of 10.

use std::io::BufRead as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;

#[path = "support/parity_harness.rs"]
mod parity_harness;

use franken_ocr::native_engine::checkpoint::{INPUT_FIELD, SELECTION_FIELD};
use franken_ocr::native_engine::window::WindowStats;
use franken_ocr::native_engine::{DocumentResult, DocumentStats, FinishReason, OcrModel};
use franken_ocr::page_source::PageSource;
use franken_ocr::{
    DocumentOptions, FocrError, PageFinish, PageWindow, ResumeKey, ResumePolicy, RunFiles,
    StreamedPage, request_shutdown, reset_shutdown,
};
use parity_harness::Logger;

const TEST: &str = "uocr2_resume";

/// One decode at a time: each takes most of the machine.
static SERIAL: Mutex<()> = Mutex::new(());

fn env_count(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

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
    assert!(
        pages.len() >= take,
        "{} holds {} pages, the test reads {take}",
        dir.display(),
        pages.len()
    );
    pages.truncate(take);
    pages
}

struct Observed {
    document: DocumentResult,
    ids: Vec<u32>,
    stats: DocumentStats,
}

/// Parse `pages`, requesting shutdown as page `stop_after` streams.
fn parse(
    model: &OcrModel,
    pages: &[PathBuf],
    options: &DocumentOptions,
    stop_after: Option<usize>,
) -> Result<Observed, FocrError> {
    let _ = model.take_last_decode();
    let mut source = PageSource::images(pages.to_vec()).expect("open the pages");
    let mut seen = 0;
    let mut sink = |_: &StreamedPage<'_>| {
        seen += 1;
        if Some(seen) == stop_after {
            request_shutdown();
        }
    };
    let document = model.recognize_document(&mut source, options, &mut |_| {}, Some(&mut sink));
    reset_shutdown();
    let document = document?;
    let report = model.take_last_decode().expect("a decode report");
    Ok(Observed {
        document,
        ids: report.ids,
        stats: report.document.expect("document stats"),
    })
}

/// A window's record without its wall time.
fn timeless(windows: &[WindowStats]) -> Vec<WindowStats> {
    let mut windows = windows.to_vec();
    for window in &mut windows {
        window.elapsed_ms = 0;
    }
    windows
}

fn fresh_files(root: &Path, name: &str) -> RunFiles {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).expect("a run directory");
    let key = ResumeKey::new()
        .with(INPUT_FIELD, "the corpus pages")
        .with(SELECTION_FIELD, "all");
    RunFiles::beside(&dir.join("doc.md"), key)
}

#[test]
fn a_cancelled_document_resumes_byte_identical() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut log = Logger::new(TEST, "byte_identical");
    let Some(artifact) = std::env::var_os("FOCR_RESUME_MODEL").map(PathBuf::from) else {
        log.skip_no_model("FOCR_RESUME_MODEL unset: point it at the v2 artifact");
        return;
    };
    let pages = corpus_pages(env_count("FOCR_RESUME_PAGES", 5));
    let window = env_count("FOCR_RESUME_WINDOW", 2);
    let model = OcrModel::load(&artifact).expect("load the v2 artifact");
    let windowed = DocumentOptions::default().page_window(PageWindow::Pages(window));
    let root = std::env::temp_dir().join(format!("focr-resume-{}", std::process::id()));

    let whole = parse(&model, &pages, &windowed, None).expect("the uninterrupted run");
    assert_eq!(whole.document.finish_reason, FinishReason::Stop);
    assert_eq!(whole.document.pages.len(), pages.len());
    assert!(
        whole.document.windows.len() > 2,
        "{} pages in windows of {window} is too short to stop twice",
        pages.len()
    );

    for stop_windows in [1, 2] {
        let files = fresh_files(&root, &format!("stop{stop_windows}"));
        let done = stop_windows * window;
        let options = windowed.clone().run(files.clone());
        let stopped = parse(&model, &pages, &options, Some(done)).expect("the stopped run");
        assert_eq!(stopped.document.finish_reason, FinishReason::Cancelled);
        // The cancel lands in the next window, whose first page it interrupts.
        assert_eq!(stopped.document.pages.len(), done + 1);
        assert_eq!(
            stopped.document.pages[done].finish,
            Some(PageFinish::Cancelled)
        );
        assert_eq!(stopped.document.resumed_from, None);
        assert!(files.journal.exists() && files.checkpoint.exists());

        // Another window is another run: refused by name, nothing touched.
        let other = DocumentOptions::default()
            .page_window(PageWindow::Pages(window + 1))
            .resume(ResumePolicy::IfMatching)
            .run(files.clone());
        match parse(&model, &pages, &other, None) {
            Err(FocrError::Usage(why)) => {
                assert!(why.contains("window"), "{why}");
                assert!(why.contains("--resume=recompute"), "{why}");
            }
            other => panic!("a differing window must be refused: {:?}", other.is_ok()),
        }

        let resumed = parse(
            &model,
            &pages,
            &options.clone().resume(ResumePolicy::IfMatching),
            None,
        )
        .expect("the resumed run");
        assert_eq!(resumed.document.finish_reason, FinishReason::Stop);
        assert_eq!(resumed.document.resumed_from, Some(done + 1));
        assert_eq!(resumed.document.markdown, whole.document.markdown);
        assert_eq!(resumed.document.pages, whole.document.pages);
        assert_eq!(resumed.ids, whole.ids, "generated ids");
        assert_eq!(resumed.stats, whole.stats, "document counters");
        assert_eq!(
            timeless(&resumed.document.windows),
            timeless(&whole.document.windows)
        );
        assert_eq!(resumed.document.warnings, whole.document.warnings);
        files.remove(false).expect("remove the run's files");
        assert!(!files.journal.exists() && !files.checkpoint.exists());
    }

    // A recompute finishes the document and says what it did.
    let files = fresh_files(&root, "recompute");
    let options = windowed.clone().run(files.clone());
    parse(&model, &pages, &options, Some(window)).expect("the stopped run");
    let recomputed = parse(
        &model,
        &pages,
        &options.clone().resume(ResumePolicy::Recompute),
        None,
    )
    .expect("the recomputed run");
    assert_eq!(recomputed.document.finish_reason, FinishReason::Stop);
    assert_eq!(recomputed.document.pages.len(), pages.len());
    assert_eq!(recomputed.document.resumed_from, Some(window + 1));
    let warning = recomputed
        .document
        .warnings
        .iter()
        .find(|w| w.kind == "resume_recompute")
        .expect("a resume_recompute warning");
    assert_eq!(warning.fields["from_page"], window + 1);
    let next = &recomputed.document.windows[1];
    assert!(!next.reused && next.fallback_reason.is_some());
    files.remove(false).expect("remove the run's files");
    log.emit("resume", "pass", serde_json::Map::new());
}

/// A `focr` run's events and exit.
struct Run {
    exit: Option<i32>,
    events: Vec<serde_json::Value>,
}

impl Run {
    fn pages(&self) -> usize {
        let decoded = |e: &&serde_json::Value| e["event"] == "page" && e["status"] == "decoded";
        self.events.iter().filter(decoded).count()
    }

    fn last(&self) -> &serde_json::Value {
        self.events.last().expect("a terminal event")
    }
}

/// `focr ocr <args> --robot`, with SIGINT as page `interrupt_at` streams.
fn focr_ocr(home: &Path, args: &[&std::ffi::OsStr], interrupt_at: Option<usize>) -> Run {
    let mut child = Command::new(env!("CARGO_BIN_EXE_focr"))
        .arg("ocr")
        .args(args)
        .arg("--robot")
        .env("HOME", home)
        .env("FOCR_NO_RESIDENT", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn focr");
    let stdout = child.stdout.take().expect("a piped stdout");
    let mut run = Run {
        exit: None,
        events: Vec::new(),
    };
    for line in std::io::BufReader::new(stdout).lines() {
        let event: serde_json::Value =
            serde_json::from_str(&line.expect("a line")).expect("an NDJSON event");
        run.events.push(event);
        if Some(run.pages()) == interrupt_at && run.last()["event"] == "page" {
            let pid = child.id().to_string();
            let sent = Command::new("kill").args(["-INT", &pid]).status();
            assert!(sent.expect("run kill").success());
        }
    }
    run.exit = child.wait().expect("wait for focr").code();
    run
}

/// Every file under `dir`, by name, with its bytes.
fn files_in(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .map(|entry| {
            let path = entry.expect("a directory entry").path();
            let name = path
                .file_name()
                .expect("a name")
                .to_string_lossy()
                .into_owned();
            let bytes = std::fs::read(&path).expect("read a figure");
            (name, bytes)
        })
        .collect();
    files.sort();
    files
}

#[cfg(unix)]
#[test]
fn a_stopped_cli_run_exits_6_and_the_same_command_finishes_it_identically() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut log = Logger::new(TEST, "cli");
    let Some(artifact) = std::env::var_os("FOCR_RESUME_MODEL").map(PathBuf::from) else {
        log.skip_no_model("FOCR_RESUME_MODEL unset: point it at the v2 artifact");
        return;
    };
    const PAGES: usize = 12;
    const WINDOW: usize = 10;
    let root = std::env::temp_dir().join(format!("focr-resume-cli-{}", std::process::id()));
    let (home, input) = (root.join("home"), root.join("pages"));
    std::fs::create_dir_all(&home).expect("a home");
    std::fs::create_dir_all(&input).expect("an input directory");
    for page in corpus_pages(PAGES) {
        let name = page.file_name().expect("a name");
        std::fs::copy(&page, input.join(name)).expect("copy a page");
    }
    let document = |out: &Path, figures: &Path| -> Vec<std::ffi::OsString> {
        let mut args: Vec<std::ffi::OsString> = vec![
            input.clone().into(),
            "--multi-page".into(),
            "--model".into(),
            artifact.clone().into(),
            "--extract-figures".into(),
            "--figures-dir".into(),
            figures.into(),
        ];
        if out.as_os_str() != "-" {
            args.extend(["-o".into(), out.into()]);
        }
        args
    };
    let run = |args: &[std::ffi::OsString], interrupt_at| {
        let refs: Vec<&std::ffi::OsStr> = args.iter().map(std::ffi::OsString::as_os_str).collect();
        focr_ocr(&home, &refs, interrupt_at)
    };

    let (whole_out, whole_figures) = (root.join("whole.json"), root.join("whole-figures"));
    let whole = run(&document(&whole_out, &whole_figures), None);
    assert_eq!(whole.exit, Some(0), "{:?}", whole.last());
    assert!(whole.last().get("resumed_from").is_none());
    assert!(!PathBuf::from(format!("{}.partial", whole_out.display())).exists());
    assert!(!PathBuf::from(format!("{}.focr-ckpt", whole_out.display())).exists());
    let whole_bytes = std::fs::read(&whole_out).expect("the output");
    let whole_text = whole.last()["markdown"].clone();

    // With -o: JSON, figures, files beside the output.
    let (out, figures) = (root.join("doc.json"), root.join("figures"));
    let args = document(&out, &figures);
    let stopped = run(&args, Some(WINDOW));
    assert_eq!(stopped.exit, Some(6), "{:?}", stopped.last());
    let last = stopped.last();
    assert_eq!(last["event"], "run_error");
    assert_eq!(last["finish_reason"], "cancelled");
    assert_eq!(last["pages_out"], WINDOW + 1, "the interrupted page counts");
    let checkpoint = format!("{}.focr-ckpt", out.display());
    assert_eq!(last["checkpoint"], checkpoint.as_str());
    assert!(last.get("resumed_from").is_none());
    assert!(stopped.events.iter().all(|e| e["event"] != "run_complete"));
    assert!(out.exists(), "the completed pages are written");
    assert!(Path::new(&checkpoint).exists());
    let journal = format!("{}.partial", out.display());
    assert!(Path::new(&journal).exists());

    let resumed = run(&args, None);
    assert_eq!(resumed.exit, Some(0), "{:?}", resumed.last());
    assert_eq!(resumed.last()["resumed_from"], WINDOW + 1);
    assert_eq!(resumed.pages(), PAGES, "the journal's pages stream again");
    assert_eq!(std::fs::read(&out).expect("the output"), whole_bytes);
    assert_eq!(files_in(&figures), files_in(&whole_figures));
    assert!(!Path::new(&journal).exists());
    assert!(!Path::new(&checkpoint).exists());

    // To stdout: the files live under the cache, by key, and the doctor
    // reports them.
    let (no_output, figures) = (PathBuf::from("-"), root.join("stdout-figures"));
    let args = document(&no_output, &figures);
    let stopped = run(&args, Some(WINDOW));
    assert_eq!(stopped.exit, Some(6), "{:?}", stopped.last());
    let checkpoint = stopped.last()["checkpoint"].as_str().expect("a checkpoint");
    assert!(checkpoint.starts_with(home.to_str().expect("a UTF-8 home")));
    let doctor = Command::new(env!("CARGO_BIN_EXE_focr"))
        .args(["doctor", "--json"])
        .env("HOME", &home)
        .output()
        .expect("run the doctor");
    let report: serde_json::Value = serde_json::from_slice(&doctor.stdout).expect("doctor JSON");
    let findings = report["findings"].as_array().expect("findings");
    let run_finding = findings
        .iter()
        .find(|f| f["detector"] == "interrupted_document_run")
        .expect("the stopped run is reported");
    let command = run_finding["fixability"]["recommended_command"]
        .as_str()
        .expect("its command");
    assert!(
        command.starts_with("focr ocr ") && command.contains("--multi-page"),
        "{command}"
    );
    assert!(Path::new(checkpoint).exists(), "the doctor touched nothing");

    let resumed = run(&args, None);
    assert_eq!(resumed.exit, Some(0), "{:?}", resumed.last());
    assert_eq!(resumed.last()["resumed_from"], WINDOW + 1);
    assert_eq!(
        resumed.pages(),
        PAGES,
        "a resumed stdout run re-emits every page"
    );
    assert_eq!(resumed.last()["markdown"], whole_text);
    assert!(!Path::new(checkpoint).exists());
    log.emit("resume", "pass", serde_json::Map::new());
}
