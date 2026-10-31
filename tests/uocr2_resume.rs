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
//! `request_shutdown()` flips a process-global flag, so this test owns its
//! process, and everything runs in one `#[test]`.
//!
//! **Gating** (LOGGING_AND_E2E.md §4): `FOCR_RESUME_MODEL` names the v2
//! artifact; without it the test logs `skip_no_model` and passes. Pages come
//! from `FOCR_CORPUS_DIR` (default `~/.cache/franken_ocr/corpus/pages`), PNGs
//! in name order, `FOCR_RESUME_PAGES` of them (default 5) in windows of
//! `FOCR_RESUME_WINDOW` (default 2). Decoding wants an optimized profile:
//!
//! ```text
//! FOCR_RESUME_MODEL=… cargo test --locked --profile release-perf --test uocr2_resume
//! ```
//! The bead's full run is 25 pages in windows of 10.

use std::path::{Path, PathBuf};

#[path = "support/parity_harness.rs"]
mod parity_harness;

use franken_ocr::native_engine::checkpoint::{INPUT_FIELD, SELECTION_FIELD};
use franken_ocr::native_engine::window::WindowStats;
use franken_ocr::native_engine::{DocumentResult, DocumentStats, FinishReason, OcrModel};
use franken_ocr::page_source::PageSource;
use franken_ocr::{
    DocumentOptions, FocrError, PageWindow, ResumeKey, ResumePolicy, RunFiles, StreamedPage,
    request_shutdown, reset_shutdown,
};
use parity_harness::Logger;

const TEST: &str = "uocr2_resume";

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
    RunFiles {
        journal: dir.join("doc.md.partial"),
        checkpoint: dir.join("doc.md.focr-ckpt"),
        key: ResumeKey::new()
            .with(INPUT_FIELD, "the corpus pages")
            .with(SELECTION_FIELD, "all"),
    }
}

#[test]
fn a_cancelled_document_resumes_byte_identical() {
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
        assert_eq!(stopped.document.pages.len(), done);
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
