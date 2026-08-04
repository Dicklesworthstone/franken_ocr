//! `agent_ergonomics_regression` — the pinning tests for the bd-wp8.7 audit
//! (G5: the agent is the primary user; every applied change ships a
//! regression test that fails if reverted).
//!
//! The audited surfaces, scores, and citations live in
//! `docs/ergonomics/AUDIT.md`. Each test here pins ONE applied change from
//! the audit's required set:
//!
//!  - the `robot triage` MEGA-COMMAND (quick_ref + health + recommendations
//!    + commands + exit codes in one round-trip, Axiom 0);
//!  - the self-describing contract surface (`robot schema` — an agent reads
//!    the contract from the tool, never an out-of-band doc);
//!  - `--json`/structured output on read-side commands with a PURE stdout
//!    (Axiom 4/8: `focr X --json | jq` works without `grep -v`);
//!  - the actionable ERROR REWRITE (what failed + where it looked + the
//!    exact copy-pasteable command to type next — never bare "see --help");
//!  - the TYPO / intent-inference handler (`--jsno` → "did you mean
//!    '--json'", Levenshtein-1 on the most common wrong invocation);
//!  - no-results-is-success (`runs` on empty history = exit 0, Axiom 5).

use std::path::PathBuf;
use std::process::{Command, Output};

fn hermetic_home() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("focr_ergo_home_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn run_focr(args: &[&str]) -> Output {
    let home = hermetic_home();
    // Unique store per invocation: the in-file tests run in parallel and a
    // shared store would collide on the open/lock path.
    let store = home.join(format!(
        "runs_{}.db",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    Command::new(env!("CARGO_BIN_EXE_focr"))
        .args(args)
        .env_remove("FOCR_MODEL_PATH")
        .env_remove("FOCR_MODEL_DIR")
        .env("FOCR_RUN_STORE", store)
        .env("HOME", &home)
        .env("LOCALAPPDATA", &home)
        .env("USERPROFILE", &home)
        .output()
        .expect("spawn focr")
}

fn emit(case: &str, ok: bool, fields: &str) {
    eprintln!(
        r#"{{"schema_version":1,"test":"agent_ergonomics_regression","case":"{case}","event":"result","result":"{}"{}{fields}}}"#,
        if ok { "pass" } else { "fail" },
        if fields.is_empty() { "" } else { "," },
    );
}

/// The mega-command: ONE round-trip carries quick_ref + health +
/// state-aware recommendations + command templates + the exit-code
/// dictionary, on a pure-JSON stdout.
#[test]
fn robot_triage_is_a_one_round_trip_mega_command() {
    let out = run_focr(&["robot", "triage"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("triage stdout is ONE pure JSON object");
    let has_all = !v["quick_ref"].is_null()
        && !v["health"].is_null()
        && v["recommendations"]
            .as_array()
            .is_some_and(|r| !r.is_empty())
        && !v["commands"].is_null()
        && v["exit_codes"].as_array().is_some_and(|c| c.len() == 8);
    // State-aware: hermetic env has NO model, so the first recommendation
    // must be the acquisition command, copy-pasteable.
    let recommends_pull = v["recommendations"][0]
        .as_str()
        .is_some_and(|r| r.starts_with("focr pull"));
    let ok = out.status.code() == Some(0) && has_all && recommends_pull;
    emit(
        "mega_command",
        ok,
        &format!(r#""has_all_sections":{has_all},"recommends_pull_first":{recommends_pull}"#),
    );
    assert!(
        ok,
        "triage payload incomplete or not state-aware:\n{stdout}"
    );
}

/// The self-describing contract: `robot schema` emits the versioned event +
/// exit-code contract from the TOOL itself.
#[test]
fn robot_schema_is_self_describing() {
    let out = run_focr(&["robot", "schema"]);
    let v: serde_json::Value = serde_json::from_str(String::from_utf8_lossy(&out.stdout).trim())
        .expect("schema stdout is pure JSON");
    let ok = out.status.code() == Some(0)
        && v["schema_version"].as_i64() == Some(1)
        && v["events"].as_array().is_some_and(|e| !e.is_empty())
        && v["exit_codes"].as_array().is_some_and(|c| c.len() == 8);
    emit("self_describing_contract", ok, "");
    assert!(ok, "robot schema not self-describing");
}

/// Axiom 4/8: read-side `--json` stdout is PURE data — parseable as-is, no
/// diagnostic lines to strip. (`runs --format json` on a fresh store.)
#[test]
fn read_side_json_stdout_is_pure_data() {
    for args in [&["runs", "--format", "json"][..], &["robot", "health"][..]] {
        let out = run_focr(args);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let parsed = serde_json::from_str::<serde_json::Value>(stdout.trim()).is_ok();
        let ok = out.status.code() == Some(0) && parsed;
        emit(
            "stdout_purity",
            ok,
            &format!(r#""argv":{args:?},"parsed_without_stripping":{parsed}"#),
        );
        assert!(ok, "{args:?}: stdout not pure JSON:\n{stdout}");
    }
}

/// The error rewrite: a missing model tells the agent WHAT failed, WHERE it
/// looked (the searched directories), and the EXACT command to type next
/// (`focr pull`) — never a bare "see --help".
#[test]
fn model_not_found_error_is_actionable() {
    let img = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/got/sample_text.png");
    let out = run_focr(&["ocr", img.to_str().unwrap()]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let ok = out.status.code() == Some(3)
        && stderr.contains("focr pull")
        && (stderr.contains("searched") || stderr.contains("FOCR_MODEL_PATH"));
    emit(
        "actionable_error",
        ok,
        &format!(
            r#""exit_code":{:?},"has_pull_hint":{},"names_search_context":{}"#,
            out.status.code(),
            stderr.contains("focr pull"),
            stderr.contains("searched") || stderr.contains("FOCR_MODEL_PATH"),
        ),
    );
    assert!(ok, "model-not-found error not actionable:\n{stderr}");
}

/// The typo handler: the most common wrong flag (`--jsno`) gets a
/// did-you-mean pointing at `--json` (Levenshtein-1 intent inference).
#[test]
fn common_flag_typo_gets_did_you_mean() {
    let out = run_focr(&["runs", "--jsno"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let ok = out.status.code() == Some(2) && stderr.contains("--json");
    emit(
        "typo_intent_inference",
        ok,
        &format!(
            r#""exit_code":{:?},"suggests_json":{}"#,
            out.status.code(),
            stderr.contains("--json")
        ),
    );
    assert!(
        ok,
        "--jsno did not suggest --json (usage exit 2 expected):\n{stderr}"
    );
}

/// Axiom 5: no results is SUCCESS — `runs` on an empty store exits 0 with an
/// empty array, never an error an agent has to special-case.
#[test]
fn empty_history_is_success_not_error() {
    let out = run_focr(&["runs", "--format", "json"]);
    let v: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&out.stdout).trim()).expect("runs JSON");
    let ok = out.status.code() == Some(0)
        && v["count"].as_u64() == Some(0)
        && v["runs"].as_array().is_some_and(Vec::is_empty);
    emit("no_results_is_success", ok, "");
    assert!(ok, "empty history must be exit 0 + empty array");
}
