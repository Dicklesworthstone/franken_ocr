//! `library_contract` — what a program linking franken_ocr in process relies on
//! (`docs/library-output.md`).
//!
//!  1. **The dependency line.** The `library` feature links no CLI, async
//!     runtime, run store, signal handler, network stack or allocator override.
//!     This reads the real resolved tree, so a dependency that slips into the
//!     feature fails here the day it lands.
//!  2. **The output.** A page's spans are frozen on real model output: a clean
//!     page, a table page, a two-column page, a page of a long document and a
//!     rotated page, as the model wrote them. They run through the same
//!     `finalize_page` a document does, so no weights are needed, and a change to
//!     a label, box, order or text is a reviewed golden diff. `UPDATE_GOLDENS=1`
//!     is the only writer (GOLDEN.md §4).

use std::path::{Path, PathBuf};
use std::process::Command;

use franken_ocr::OUTPUT_CONTRACT_VERSION;
use franken_ocr::native_engine::LayoutSpan;
use franken_ocr::native_engine::postprocess::finalize_page;

/// Crates a library consumer must not inherit.
const FORBIDDEN: [&str; 16] = [
    "asupersync",
    "clap",
    "ctrlc",
    "fsqlite",
    "fsqlite-types",
    "hyper",
    "mimalloc",
    "nix",
    "reqwest",
    "rustls",
    "tokio",
    "tracing",
    "tracing-subscriber",
    "ureq",
    "uuid",
    "webpki-roots",
];

/// The crates the feature exists to provide.
const REQUIRED: [&str; 4] = ["frankentorch-core", "image", "lopdf", "memmap2"];

/// A frame every box maps through, wider than tall as no fixture page is.
const SIZE: (u32, u32) = (1234, 987);

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The crate names in the resolved normal-edge tree of the `library` feature.
fn library_tree() -> Vec<String> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let out = Command::new(cargo)
        .arg("tree")
        .arg("--manifest-path")
        .arg(manifest_dir().join("Cargo.toml"))
        .args([
            "--locked",
            "--offline",
            "--no-default-features",
            "--features",
            "library",
            "--edges",
            "normal",
            "--prefix",
            "none",
        ])
        .output()
        .expect("spawn cargo tree");
    assert!(
        out.status.success(),
        "cargo tree: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut names: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| line.split_whitespace().next().map(str::to_owned))
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}

#[test]
fn the_library_feature_links_no_cli_runtime_store_or_network() {
    let tree = library_tree();
    let inherited: Vec<&&str> = FORBIDDEN
        .iter()
        .filter(|name| tree.iter().any(|crate_name| crate_name == *name))
        .collect();
    assert!(
        inherited.is_empty(),
        "inherited {inherited:?} from {tree:?}"
    );
    for name in REQUIRED {
        assert!(
            tree.iter().any(|crate_name| crate_name == name),
            "{name} missing from {tree:?}"
        );
    }
}

/// `tests/fixtures/uocr2_oracle/<file>`'s page `page` (0-based) as the model wrote it.
fn oracle_body(file: &str, page: usize) -> String {
    let path = manifest_dir()
        .join("tests/fixtures/uocr2_oracle")
        .join(file);
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    raw.split("<PAGE>")
        .nth(page + 1)
        .unwrap_or_else(|| panic!("{file}: no page {page}"))
        .to_owned()
}

/// A page turned 90°, as the model wrote it (`tests/fixtures/PROVENANCE.md`).
fn rotated_body() -> String {
    let path = manifest_dir().join("tests/fixtures/library/rotated_page.md");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A page's spans, one JSON line each, in the order the contract promises.
fn snapshot(layout: &[LayoutSpan]) -> String {
    let header = serde_json::json!({
        "contract": OUTPUT_CONTRACT_VERSION,
        "width": SIZE.0,
        "height": SIZE.1,
        "spans": layout.len(),
    });
    std::iter::once(header)
        .chain(layout.iter().map(LayoutSpan::to_json))
        .map(|line| line.to_string() + "\n")
        .collect()
}

fn assert_golden(name: &str, actual: &str) {
    let dir = manifest_dir().join("tests/fixtures/golden");
    let golden = dir.join(format!("{name}.golden"));
    if matches!(std::env::var("UPDATE_GOLDENS").ok().as_deref(), Some("1")) {
        std::fs::write(&golden, actual).expect("write the golden");
        return;
    }
    let expected = std::fs::read_to_string(&golden).unwrap_or_else(|e| {
        panic!(
            "{}: {e}; review, then bless with UPDATE_GOLDENS=1",
            golden.display()
        )
    });
    if expected != actual {
        let candidate: &Path = &dir.join(format!("{name}.actual"));
        std::fs::write(candidate, actual).expect("write the candidate");
        panic!(
            "{name}: differs from its golden; candidate at {}",
            candidate.display()
        );
    }
}

/// Every span's boxes stay on the page, and the same body reads the same way twice.
fn assert_well_formed(name: &str, body: &str, layout: &[LayoutSpan]) {
    assert!(!layout.is_empty(), "{name}: no spans");
    for span in layout {
        assert!(!span.boxes.is_empty(), "{name}: {span:?} has no box");
        for &[x1, y1, x2, y2] in &span.boxes {
            assert!((0..=i64::from(SIZE.0)).contains(&x1), "{name}: {span:?}");
            assert!((0..=i64::from(SIZE.0)).contains(&x2), "{name}: {span:?}");
            assert!((0..=i64::from(SIZE.1)).contains(&y1), "{name}: {span:?}");
            assert!((0..=i64::from(SIZE.1)).contains(&y2), "{name}: {span:?}");
        }
    }
    assert_eq!(
        snapshot(layout),
        snapshot(&finalize_page(body, 0, SIZE).layout),
        "{name}"
    );
}

#[test]
fn a_page_of_spans_is_frozen_on_real_model_output() {
    for (name, body, label) in [
        (
            "library_clean_page",
            oracle_body("S1-page_0014.md", 0),
            "title",
        ),
        (
            "library_table_page",
            oracle_body("S1-page_0590.md", 0),
            "table",
        ),
        (
            "library_long_document_page",
            oracle_body("S3-p20.md", 7),
            "text",
        ),
        (
            "library_two_column_page",
            oracle_body("S3-p20.md", 19),
            "header",
        ),
        ("library_rotated_page", rotated_body(), "page_number"),
    ] {
        let layout = finalize_page(&body, 0, SIZE).layout;
        assert_well_formed(name, &body, &layout);
        assert!(
            layout.iter().any(|span| span.label == label),
            "{name}: no {label} span"
        );
        assert_golden(name, &snapshot(&layout));
    }
}

#[test]
fn a_table_span_holds_the_table_the_model_wrote() {
    let body = oracle_body("S1-page_0590.md", 0);
    let layout = finalize_page(&body, 0, SIZE).layout;
    let table = layout
        .iter()
        .find(|span| span.label == "table")
        .expect("a table span");
    let text = table.text.as_deref().expect("a table span has text");
    assert!(
        text.contains("<table>") && text.contains("</table>"),
        "{text}"
    );
}

/// Spans stay in the model's reading order: down the left column, then down
/// the right, however the boxes would sort.
#[test]
fn a_two_column_page_keeps_the_order_the_model_read_it() {
    let body = oracle_body("S3-p20.md", 19);
    let layout = finalize_page(&body, 0, SIZE).layout;
    let middle = i64::from(SIZE.0) / 2;
    let in_right_column: Vec<bool> = layout
        .iter()
        .filter(|span| span.label == "text")
        .map(|span| span.boxes[0][0] > middle)
        .collect();
    let first_right = in_right_column
        .iter()
        .position(|&right| right)
        .expect("a right column");
    assert!(first_right > 0, "no left column");
    assert!(
        in_right_column[first_right..].iter().all(|&right| right),
        "the columns interleave: {in_right_column:?}"
    );
}

/// A box is the model's, mapped to pixels: not clamped and not put in order.
#[test]
fn a_box_is_mapped_as_written() {
    let body = "<BLOCK>text[1200, 40, 20, 999]<FORK>past the edge, x2 below x1";
    let layout = finalize_page(body, 0, SIZE).layout;
    let [x1, _, x2, y2] = layout[0].boxes[0];
    assert!(x1 > i64::from(SIZE.0), "{x1} was clamped");
    assert!(x2 < x1, "{x2} was put in order");
    assert_eq!(y2, i64::from(SIZE.1));
}
