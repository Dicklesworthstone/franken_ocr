//! `uocr2_memory` — U2-D3's proof that the whole-process estimate holds on a
//! real run.
//!
//! A 20-page Unlimited-OCR-2 document decodes in the default windows. The
//! process's high-water RSS at the end must not exceed the estimate by more
//! than the margin registered on the bead before the first calibration run,
//! and every window must have reported the high-water mark it reached.
//!
//! RSS is a whole-process high-water mark, so this test owns its process.
//!
//! **Gating** (LOGGING_AND_E2E.md §4): `FOCR_MEMORY_MODEL` names the v2
//! artifact; without it the test logs `skip_no_model` and passes. Pages come
//! from `FOCR_CORPUS_DIR` (default `~/.cache/franken_ocr/corpus/pages`), the
//! first `FOCR_MEMORY_PAGES` PNGs in name order (default 20). Decoding wants
//! an optimized profile:
//!
//! ```text
//! FOCR_MEMORY_MODEL=… cargo test --locked --profile release-perf --test uocr2_memory
//! ```

use std::path::PathBuf;

#[path = "support/parity_harness.rs"]
mod parity_harness;

use franken_ocr::native_engine::memory::peak_rss_bytes;
use franken_ocr::native_engine::{FinishReason, OcrModel};
use franken_ocr::page_source::PageSource;
use franken_ocr::{DocumentOptions, PageWindow};
use parity_harness::Logger;

/// The margin registered on the bead before the first calibration run.
const MARGIN: f64 = 1.10;
const PAGE_WINDOW: usize = 10;
const TOKENS_PER_PAGE: usize = 1_000;

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

#[test]
fn a_real_document_stays_within_its_estimate() {
    let mut log = Logger::new("uocr2_memory", "twenty_pages");
    let Some(artifact) = std::env::var_os("FOCR_MEMORY_MODEL").map(PathBuf::from) else {
        log.skip_no_model("FOCR_MEMORY_MODEL unset: point it at the v2 artifact");
        return;
    };
    let pages = corpus_pages(
        std::env::var("FOCR_MEMORY_PAGES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(20),
    );
    let model = OcrModel::load(&artifact).expect("load the v2 artifact");
    let estimate = model
        .estimate_peak_bytes(PAGE_WINDOW, TOKENS_PER_PAGE)
        .expect("estimate the run");

    let mut source = PageSource::images(pages.clone()).expect("open the pages");
    let options = DocumentOptions::default().page_window(PageWindow::Pages(PAGE_WINDOW));
    let document = model
        .recognize_document(&mut source, &options, &mut |_| {}, None)
        .expect("decode the document");
    assert_eq!(document.finish_reason, FinishReason::Stop);
    assert_eq!(document.pages.len(), pages.len());

    let measured = peak_rss_bytes().expect("a peak RSS reading on this platform");
    eprintln!(
        "estimate {} bytes, measured {measured} bytes, {:.3} of the estimate",
        estimate.total,
        measured as f64 / estimate.total as f64
    );
    assert!(
        measured as f64 <= estimate.total as f64 * MARGIN,
        "measured {measured} bytes against an estimate of {} bytes: {:?}",
        estimate.total,
        estimate.breakdown
    );

    let readings: Vec<u64> = document
        .windows
        .iter()
        .map(|window| window.rss_bytes_peak.expect("each window reports its peak"))
        .collect();
    assert!(
        readings.windows(2).all(|pair| pair[0] <= pair[1]),
        "a high-water mark never falls: {readings:?}"
    );
    assert!(readings.iter().all(|&reading| reading <= measured));
    log.assertion("measured_within_estimate", true);
}
