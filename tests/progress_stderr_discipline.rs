//! `progress_stderr_discipline` — end-to-end pins for the GH #2 progress bar.
//!
//! The bar's contract (src/progress.rs) is that machine-readable output can
//! NEVER be corrupted by progress rendering:
//!
//!  * the bar writes only to stderr;
//!  * `--robot` / `--json` runs construct it disabled;
//!  * a non-TTY stderr (every piped/CI invocation, including these tests)
//!    auto-disables it.
//!
//! These tests run the real `focr` binary through the drivers that carry the
//! bar (`ocr-batch` sequential loop, `ocr --robot`) in a hermetic no-model
//! HOME and assert the streams stay exactly as machine-clean as before the
//! bar existed: no `\r` redraw bytes, no ANSI escapes, parseable JSON/NDJSON
//! where promised. The batch cases construct and tick the disabled non-TTY
//! bar; the robot case independently pins its machine-stream boundary.

use std::path::PathBuf;
use std::process::{Command, Output};

/// A minimal valid 8x8 grayscale PNG (zlib-compressed, filter 0 rows) so the
/// batch loop gets past input decoding and into the recognize call without
/// needing model weights on disk.
const TINY_PNG: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 8, 0, 0, 0, 8, 8, 0, 0,
    0, 0, 225, 100, 225, 87, 0, 0, 0, 14, 73, 68, 65, 84, 120, 156, 99, 248, 15, 5, 12, 148, 49, 0,
    247, 192, 63, 193, 178, 52, 65, 187, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];

fn hermetic_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "focr_progress_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir).expect("create hermetic dir");
    dir
}

/// Run `focr` with a hermetic HOME (no model, no shared run store) and stdio
/// PIPED — the exact non-TTY condition the bar must auto-disable under.
fn run_focr(home: &PathBuf, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_focr"))
        .args(args)
        .env_remove("FOCR_MODEL_PATH")
        .env_remove("FOCR_MODEL_DIR")
        .env_remove("FOCR_NO_PROGRESS")
        .env("FOCR_RUN_STORE", home.join("runs.db"))
        .env("HOME", home)
        .env("LOCALAPPDATA", home)
        .env("USERPROFILE", home)
        .output()
        .expect("spawn focr")
}

/// The bytes a progress bar could leak: carriage returns (in-place redraws)
/// and ANSI escape sequences. Machine consumers tolerate neither.
fn assert_no_progress_bytes(stream: &str, what: &str) {
    assert!(
        !stream.contains('\r'),
        "{what} must carry no \\r redraw bytes, got: {stream:?}"
    );
    assert!(
        !stream.contains('\u{1b}'),
        "{what} must carry no ANSI escapes, got: {stream:?}"
    );
}

/// `ocr-batch` (human mode, piped): the sequential loop constructs and ticks
/// the bar around every image, and with stderr a pipe it must emit NOTHING of
/// it — stdout/stderr stay the pre-bar streams.
#[test]
fn batch_human_mode_piped_emits_no_bar_bytes() {
    let home = hermetic_dir("batch_human");
    let a = home.join("a.png");
    let b = home.join("b.png");
    std::fs::write(&a, TINY_PNG).unwrap();
    std::fs::write(&b, TINY_PNG).unwrap();
    let out = run_focr(
        &home,
        &["ocr-batch", a.to_str().unwrap(), b.to_str().unwrap()],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!stderr.contains('\r'), "piped stderr got \\r: {stderr:?}");
    assert!(
        !stderr.contains('\u{1b}'),
        "piped stderr got ANSI escapes: {stderr:?}"
    );
    assert!(!stdout.contains('\r'), "stdout got \\r: {stdout:?}");
    // The hermetic HOME has no model, so each image fails with the actionable
    // model message — on the SAME clean streams as before the bar existed.
    assert!(
        stderr.contains("batch complete"),
        "batch summary line survives: {stderr:?}"
    );
}

/// `ocr-batch --json`: one pure-JSON stdout object, no bar bytes anywhere
/// (the bar is constructed disabled in `--json` mode before the TTY check
/// even runs).
#[test]
fn batch_json_mode_stdout_stays_pure_json() -> Result<(), String> {
    let home = hermetic_dir("batch_json");
    let a = home.join("a.png");
    std::fs::write(&a, TINY_PNG).unwrap();
    let out = run_focr(&home, &["ocr-batch", "--json", a.to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let v: serde_json::Value = serde_json::from_str(stdout.trim())
        .map_err(|e| format!("--json stdout must be ONE JSON object ({e}): {stdout:?}"))?;
    assert_eq!(v["command"], "ocr-batch");
    assert_no_progress_bytes(&stderr, "--json stderr");
    Ok(())
}

/// `ocr --robot`: the NDJSON contract — every stdout line parses as JSON and
/// no progress bytes appear on either stream.
#[test]
fn robot_ndjson_stream_carries_no_bar_bytes() -> Result<(), String> {
    let home = hermetic_dir("robot");
    let a = home.join("a.png");
    std::fs::write(&a, TINY_PNG).unwrap();
    let out = run_focr(&home, &["ocr", "--robot", a.to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stdout.is_empty(), "robot mode streams NDJSON events");
    for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
        let _: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| format!("robot stdout line must be JSON ({e}): {line:?}"))?;
    }
    assert!(!stdout.contains('\r'), "robot stdout got \\r: {stdout:?}");
    assert_no_progress_bytes(&stderr, "robot stderr");
    Ok(())
}
