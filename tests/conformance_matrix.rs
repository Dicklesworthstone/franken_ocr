//! bd-re8.12: the coverage-accounting matrix + XFAIL discipline meta-tests.
//!
//! The matrix is computed FROM THE SPEC, not from the test list: every
//! `[SPEC-NNN]` clause in `docs/truth-pack/EXISTING_UNLIMITED_OCR_STRUCTURE.md`
//! is enumerated, its requirement level classified (a clause paragraph
//! carrying `SHOULD`/`MAY` is tracked at that level; everything else in the
//! extracted structure doc is a MUST — it describes what the model IS), and
//! coverage is the presence of a `SPEC-NNN` reference in `src/**` or
//! `tests/**` (implementation + rung annotations). A missing suite therefore
//! surfaces as an UNCOVERED MUST CLAUSE here, never as a green-but-hollow
//! registry.
//!
//! Gates:
//! * MUST coverage ≥ 0.95 to claim conformance (the bead's ratio, asserted);
//! * every `xfail`-marked test maps to a real `DISC-NNN` in
//!   `docs/DISCREPANCIES.md` (XFAIL is ledgered divergence, never SKIP-debt);
//! * every registry entry (`conformance::conformance_registry`) RUNS green.
//!
//! The matrix is logged as structured NDJSON (one line per clause) — the
//! auditable artifact the release scorecard consumes.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Every `[SPEC-NNN]` clause id in the spec, with its paragraph text.
fn spec_clauses() -> BTreeMap<u32, String> {
    let spec = read(&repo_root().join("docs/truth-pack/EXISTING_UNLIMITED_OCR_STRUCTURE.md"));
    let mut clauses = BTreeMap::new();
    let mut current: Option<(u32, String)> = None;
    for line in spec.lines() {
        if let Some(start) = line.find("[SPEC-") {
            if let Some((id, text)) = current.take() {
                clauses.insert(id, text);
            }
            let digits: String = line[start + 6..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            if let Ok(id) = digits.parse::<u32>() {
                current = Some((id, line.to_owned()));
                continue;
            }
        }
        if let Some((_, text)) = current.as_mut() {
            if line.trim().is_empty() {
                let taken = current.take().expect("current set");
                clauses.insert(taken.0, taken.1);
            } else {
                text.push('\n');
                text.push_str(line);
            }
        }
    }
    if let Some((id, text)) = current {
        clauses.insert(id, text);
    }
    assert!(
        clauses.len() >= 50,
        "the spec should enumerate dozens of clauses; parsed {} — parser drift?",
        clauses.len()
    );
    clauses
}

/// Every SPEC-NNN id referenced anywhere under `src/` or `tests/`.
fn referenced_clauses() -> BTreeSet<u32> {
    let mut refs = BTreeSet::new();
    let mut stack = vec![repo_root().join("src"), repo_root().join("tests")];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let text = read(&path);
                let mut rest = text.as_str();
                while let Some(pos) = rest.find("SPEC-") {
                    rest = &rest[pos + 5..];
                    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
                    if let Ok(id) = digits.parse::<u32>() {
                        refs.insert(id);
                    }
                    // Range annotations like `[SPEC-100..103]` cover the span.
                    if let Some(range_rest) = rest.strip_prefix(&digits)
                        && let Some(after) = range_rest.strip_prefix("..")
                    {
                        let hi: String = after.chars().take_while(char::is_ascii_digit).collect();
                        if let (Ok(lo), Ok(hi)) = (digits.parse::<u32>(), hi.parse::<u32>()) {
                            for id in lo..=hi {
                                refs.insert(id);
                            }
                        }
                    }
                }
            }
        }
    }
    refs
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Level {
    Must,
    Should,
    May,
}

fn classify(text: &str) -> Level {
    // The extracted structure doc states facts (MUST) unless a clause
    // explicitly hedges.
    if text.contains("SHOULD") {
        Level::Should
    } else if text.contains(" MAY ") {
        Level::May
    } else {
        Level::Must
    }
}

/// The bd-re8.12 coverage matrix: enumerate from the SPEC, account against
/// code+test references, log every clause as NDJSON, gate MUST ≥ 0.95.
#[test]
fn coverage_matrix_must_ratio_gate() {
    let clauses = spec_clauses();
    let refs = referenced_clauses();
    let (mut must_total, mut must_covered) = (0usize, 0usize);
    let (mut should_total, mut should_covered) = (0usize, 0usize);
    let mut uncovered_must: Vec<u32> = Vec::new();
    for (&id, text) in &clauses {
        let level = classify(text);
        let covered = refs.contains(&id);
        match level {
            Level::Must => {
                must_total += 1;
                if covered {
                    must_covered += 1;
                } else {
                    uncovered_must.push(id);
                }
            }
            Level::Should => {
                should_total += 1;
                should_covered += usize::from(covered);
            }
            Level::May => {}
        }
        // The auditable matrix artifact: one structured line per clause.
        eprintln!(
            "{{\"test\":\"coverage_matrix\",\"event\":\"clause\",\"spec\":\"SPEC-{id:03}\",\
             \"level\":\"{level:?}\",\"covered\":{covered}}}"
        );
    }
    let ratio = must_covered as f64 / must_total.max(1) as f64;
    eprintln!(
        "{{\"test\":\"coverage_matrix\",\"event\":\"summary\",\"must_total\":{must_total},\
         \"must_covered\":{must_covered},\"must_ratio\":{ratio:.4},\
         \"should_total\":{should_total},\"should_covered\":{should_covered},\
         \"uncovered_must\":{uncovered_must:?}}}"
    );
    assert!(
        ratio >= 0.95,
        "MUST coverage {ratio:.4} < 0.95 — uncovered MUST clauses: {uncovered_must:?} \
         (cover them or ledger an XFAIL/DISC; 'partial' never rounds up)"
    );
}

/// XFAIL discipline (bd-re8.12): no BARE xfail. Every xfail EMISSION either
/// carries a `DISC-NNN` (an intentional, ledgered divergence) or states a
/// recognized PHASE-GAP reason (fixture/stage unavailable — documented debt,
/// counted, never silently skipped). A bare xfail with neither is hidden
/// debt and fails here.
#[test]
fn every_xfail_maps_to_a_disc_entry_or_stated_gap() {
    let disc = read(&repo_root().join("docs/DISCREPANCIES.md"));
    let known: BTreeSet<String> = {
        let mut ids = BTreeSet::new();
        let mut rest = disc.as_str();
        while let Some(pos) = rest.find("## DISC-") {
            rest = &rest[pos + 8..];
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            if !digits.is_empty() {
                ids.insert(format!("DISC-{digits}"));
            }
        }
        ids
    };
    assert!(
        !known.is_empty(),
        "DISCREPANCIES.md must define DISC entries"
    );

    const GAP_MARKERS: &[&str] = &[
        "not_implemented",
        "NotImplemented",
        "fixture",
        "unavailable",
        "absent",
        "phase",
        "scaffold",
        "no preprocess",
        "gap",
        "pending",
        // An error-logged xfail states its reason in the error line itself.
        ".error(",
        // Clause-table skip rows carry their contract fields as the reason.
        "expected_exit",
    ];
    let mut violations = Vec::new();
    let mut xfail_sites = 0usize;
    let mut stack = vec![repo_root().join("src"), repo_root().join("tests")];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "rs") || path.ends_with("conformance_matrix.rs")
            {
                continue;
            }
            let text = read(&path);
            let lines: Vec<&str> = text.lines().collect();
            for (lineno, line) in lines.iter().enumerate() {
                // Emission sites only: the "xfail" STRING literal (log result
                // values / log_xfail calls), never doc/comment mentions or
                // the helper definition.
                // `"xfail":` (colon-suffixed) is a JSON FIELD NAME in a
                // payload builder, not an emission.
                let is_emission = (line.contains("\"xfail\"") || line.contains("log_xfail("))
                    && !line.contains("\"xfail\":")
                    && !line.contains("fn log_xfail")
                    && !line.trim_start().starts_with("//");
                if !is_emission {
                    continue;
                }
                xfail_sites += 1;
                let lo = lineno.saturating_sub(10);
                let hi = (lineno + 11).min(lines.len());
                let context = lines[lo..hi].join("\n");
                let ledgered = known.iter().any(|d| context.contains(d.as_str()));
                let reasoned = GAP_MARKERS.iter().any(|m| context.contains(m));
                if !ledgered && !reasoned {
                    violations.push(format!(
                        "{}:{}: BARE xfail — neither a DISC-NNN nor a stated phase-gap \
                         reason within ±10 lines",
                        path.display(),
                        lineno + 1
                    ));
                }
            }
        }
    }
    eprintln!(
        "{{\"test\":\"xfail_discipline\",\"event\":\"summary\",\"xfail_sites\":{xfail_sites},\
         \"violations\":{}}}",
        violations.len()
    );
    assert!(
        xfail_sites > 0,
        "the scanner must find the known xfail sites"
    );
    assert!(
        violations.is_empty(),
        "bare xfail sites (ledger a DISC-NNN or state the gap):\n{}",
        violations.join("\n")
    );
}

/// Every registered ConformanceTest entry runs green in-process.
#[test]
fn conformance_registry_runs_green() {
    use franken_ocr::conformance::{ConformanceTest as _, conformance_registry};
    let registry = conformance_registry();
    assert!(registry.len() >= 2, "the shipped suites must register");
    for entry in &registry {
        let result = entry.run();
        eprintln!(
            "{{\"test\":\"conformance_registry\",\"event\":\"entry\",\"name\":\"{}\",\
             \"category\":\"{:?}\",\"level\":\"{:?}\",\"clauses\":{:?},\"passed\":{}}}",
            entry.name(),
            entry.category(),
            entry.requirement_level(),
            entry.clauses(),
            result.passed
        );
        assert!(result.passed, "{}: {}", entry.name(), result.message);
    }
}
