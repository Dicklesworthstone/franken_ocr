//! `surface_matrix` — the SurfaceMatrix enumeration lock (bd-re8.13).
//!
//! The three-pillar release certification's surface pillar scores against
//! `docs/FEATURE_PARITY.md` §12–§15. That file is only trustworthy if it
//! ENUMERATES everything: a CLI subcommand or robot event with no row is
//! silent coverage debt the gauntlet cannot see. This suite locks the matrix
//! against exactly that drift:
//!
//!  1. **CLI completeness** — every live `focr` subcommand (from clap
//!     introspection, `CommandFactory`, so a new subcommand fails here the
//!     day it lands) and every `focr robot` subcommand has a §12 row.
//!  2. **Robot-contract completeness** — every event and every exit code in
//!     the FROZEN schema (`tests/fixtures/robot_schema_v1.json`) has a §13
//!     row / §12 exit-code row.
//!  3. **Vocabulary** — every scoreboard row carries a valid
//!     `Status ∈ {present, partial, missing, n/a, excluded}` and
//!     `Req ∈ {MUST, SHOULD, MAY, n/a}` (the doc-lint contract, bd-322.25).
//!  4. **Rollup honesty** — the §0 SurfaceMatrix rollup column equals a
//!     recount of the actual §12–§15 cells (`partial` NEVER silently counted
//!     as `present`; a hand-edited rollup that drifts from its cells fails).
//!
//! The debt totals are emitted as one NDJSON line — the release scorecard
//! (bd-wp8.10) and the gauntlet read those, never a rounded-up summary.

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::CommandFactory;
use franken_ocr::cli::Cli;

const STATUS_VOCAB: [&str; 5] = ["present", "partial", "missing", "n/a", "excluded"];
const REQ_VOCAB: [&str; 4] = ["MUST", "SHOULD", "MAY", "n/a"];

fn parity_md() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs/FEATURE_PARITY.md");
    std::fs::read_to_string(&path).expect("docs/FEATURE_PARITY.md is committed")
}

/// One parsed scoreboard row.
#[derive(Debug)]
struct Row {
    section: u32,
    surface: String,
    req: String,
    status: String,
}

/// Parse every status-bearing table row, tagged with its `## N.` section.
/// A row is status-bearing when one of its cells is EXACTLY a status token —
/// the same rule the §0 rollup recount uses, so the two cannot diverge.
fn parse_rows(md: &str) -> Vec<Row> {
    let mut section = 0u32;
    let mut rows = Vec::new();
    for line in md.lines() {
        if let Some(rest) = line.strip_prefix("## ")
            && let Some((n, _)) = rest.split_once('.')
            && let Ok(n) = n.parse::<u32>()
        {
            section = n;
            continue;
        }
        if !line.starts_with('|') || line.starts_with("|-") {
            continue;
        }
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        // cells[0] and cells[last] are the empty outside-pipe fragments.
        if cells.len() < 6 {
            continue;
        }
        if let Some(status_idx) = cells
            .iter()
            .position(|c| STATUS_VOCAB.contains(&c.to_lowercase().as_str()))
        {
            // Req sits immediately before Status in every scoreboard table.
            let req = cells.get(status_idx - 1).copied().unwrap_or("");
            rows.push(Row {
                section,
                surface: cells[1].to_string(),
                req: req.to_string(),
                status: cells[status_idx].to_lowercase(),
            });
        }
    }
    rows
}

fn surface_rows(rows: &[Row]) -> Vec<&Row> {
    rows.iter()
        .filter(|r| (12..=15).contains(&r.section))
        .collect()
}

fn emit(check: &str, ok: bool, fields: &str) {
    eprintln!(
        r#"{{"schema_version":1,"test":"surface_matrix","case":"{check}","event":"result","result":"{}"{}{fields}}}"#,
        if ok { "pass" } else { "fail" },
        if fields.is_empty() { "" } else { "," },
    );
}

/// Every live CLI subcommand (top-level + `robot` nested), from clap
/// introspection, appears in some §12 row. A subcommand added without a
/// scoreboard row fails HERE, the day it lands.
#[test]
fn every_cli_subcommand_has_a_surface_row() {
    let md = parity_md();
    let rows = parse_rows(&md);
    let sm: Vec<&Row> = surface_rows(&rows);
    let cmd = Cli::command();

    let mut missing: Vec<String> = Vec::new();
    let mut covered = 0usize;
    for sub in cmd.get_subcommands() {
        let name = sub.get_name().to_string();
        let hit = sm
            .iter()
            .any(|r| r.surface.contains(&format!("focr {name}")) || r.surface.contains(&name));
        if hit {
            covered += 1;
        } else {
            missing.push(name.clone());
        }
        if name == "robot" {
            for nested in sub.get_subcommands() {
                let nname = nested.get_name().to_string();
                let nhit = sm
                    .iter()
                    .any(|r| r.surface.contains(&format!("robot {nname}")));
                if nhit {
                    covered += 1;
                } else {
                    missing.push(format!("robot {nname}"));
                }
            }
        }
    }
    let ok = missing.is_empty();
    emit(
        "cli_subcommand_enumeration",
        ok,
        &format!(r#""covered":{covered},"missing":{missing:?}"#).replace('"', "\""),
    );
    assert!(
        ok,
        "live CLI subcommands with NO SurfaceMatrix row (silent coverage debt): {missing:?}"
    );
}

/// Every event and exit code in the FROZEN robot schema has a row.
#[test]
fn every_frozen_robot_event_and_exit_code_has_a_row() {
    let md = parity_md();
    let rows = parse_rows(&md);
    let sm = surface_rows(&rows);
    let schema_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/robot_schema_v1.json");
    let schema: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(schema_path).expect("frozen schema"))
            .expect("schema parses");

    let mut missing: Vec<String> = Vec::new();
    let events = schema["events"]
        .as_array()
        .expect("schema.events is an array");
    let event_names: Vec<String> = events
        .iter()
        .map(|e| {
            e.as_str()
                .map(String::from)
                .or_else(|| e["name"].as_str().map(String::from))
                .or_else(|| e["event"].as_str().map(String::from))
                .expect("each schema event carries its name")
        })
        .collect();
    for name in &event_names {
        if !sm.iter().any(|r| r.surface.contains(&format!("`{name}`"))) {
            missing.push(format!("event {name}"));
        }
    }
    let codes = schema["exit_codes"].as_array().expect("schema.exit_codes");
    let max_code = codes
        .iter()
        .filter_map(|c| c["code"].as_u64())
        .max()
        .expect("codes non-empty");
    let exit_row = format!("Exit codes 0..{max_code}");
    if !sm.iter().any(|r| r.surface.contains(&exit_row)) {
        missing.push(exit_row);
    }
    let ok = missing.is_empty();
    emit(
        "robot_contract_enumeration",
        ok,
        &format!(
            r#""events":{},"exit_codes":{},"missing":{missing:?}"#,
            event_names.len(),
            codes.len()
        ),
    );
    assert!(
        ok,
        "frozen robot-contract entries with NO SurfaceMatrix row: {missing:?}"
    );
}

/// The doc-lint vocabulary contract over EVERY scoreboard row (both
/// populations): valid Status, valid Req.
#[test]
fn every_row_carries_valid_status_and_req() {
    let md = parity_md();
    let rows = parse_rows(&md);
    assert!(
        rows.len() > 150,
        "scoreboard parse collapsed: {} rows",
        rows.len()
    );
    let bad: Vec<String> = rows
        .iter()
        .filter(|r| {
            !STATUS_VOCAB.contains(&r.status.as_str()) || !REQ_VOCAB.contains(&r.req.as_str())
        })
        .map(|r| {
            format!(
                "§{} {} (req={:?}, status={:?})",
                r.section, r.surface, r.req, r.status
            )
        })
        .collect();
    let ok = bad.is_empty();
    emit(
        "row_vocabulary",
        ok,
        &format!(r#""rows":{},"bad":{}"#, rows.len(), bad.len()),
    );
    assert!(ok, "malformed scoreboard rows: {bad:?}");
}

/// Rollup honesty: the §0 SurfaceMatrix column equals the recount of the
/// §12–§15 cells. `partial` never counts as `present` — structurally, because
/// the recount buckets by exact token; and a hand-edited rollup that drifts
/// from its cells fails here.
#[test]
fn rollup_matches_recounted_cells() {
    let md = parity_md();
    let rows = parse_rows(&md);
    let sm = surface_rows(&rows);
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &sm {
        *counts.entry(r.status.as_str()).or_default() += 1;
    }

    // Parse the §0 rollup's SurfaceMatrix column: rows like
    // `| `present` | 0 † | 39 | 39 |` — the third pipe-cell.
    let mut rollup: BTreeMap<String, usize> = BTreeMap::new();
    let mut total_row: Option<usize> = None;
    for line in md.lines() {
        if !line.starts_with('|') {
            continue;
        }
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        if cells.len() < 5 {
            continue;
        }
        let label = cells[1].trim_matches('`').trim();
        let key = match label {
            "present" | "partial" | "missing" | "n/a" => label.to_string(),
            l if l.starts_with("excluded") => "excluded".to_string(),
            l if l.starts_with("Total enumerated rows") => {
                total_row = cells[3].trim_matches(['*', ' ']).parse::<usize>().ok();
                continue;
            }
            _ => continue,
        };
        // SurfaceMatrix column = cells[3]; take the leading integer (the
        // excluded cell carries a parenthetical note).
        let n: String = cells[3].chars().take_while(char::is_ascii_digit).collect();
        if let Ok(n) = n.parse::<usize>() {
            rollup.insert(key, n);
        }
    }

    let mut mismatches: Vec<String> = Vec::new();
    for status in STATUS_VOCAB {
        let counted = counts.get(status).copied().unwrap_or(0);
        let claimed = rollup.get(status).copied().unwrap_or(0);
        if counted != claimed {
            mismatches.push(format!(
                "{status}: rollup says {claimed}, cells say {counted}"
            ));
        }
    }
    if let Some(total) = total_row {
        if total != sm.len() {
            mismatches.push(format!(
                "total: rollup says {total}, cells say {}",
                sm.len()
            ));
        }
    } else {
        mismatches.push("rollup total row not found".to_string());
    }

    let ok = mismatches.is_empty();
    let debt = counts.get("partial").copied().unwrap_or(0)
        + counts.get("missing").copied().unwrap_or(0)
        + counts.get("excluded").copied().unwrap_or(0);
    emit(
        "rollup_honesty",
        ok,
        &format!(
            r#""surface_rows":{},"present":{},"partial":{},"missing":{},"excluded":{},"coverage_debt":{debt},"mismatches":{mismatches:?}"#,
            sm.len(),
            counts.get("present").copied().unwrap_or(0),
            counts.get("partial").copied().unwrap_or(0),
            counts.get("missing").copied().unwrap_or(0),
            counts.get("excluded").copied().unwrap_or(0),
        ),
    );
    assert!(
        ok,
        "§0 rollup drifted from the §12–§15 cells: {mismatches:?}"
    );
}
