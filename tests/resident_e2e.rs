//! The resident daemon on real weights (beads bd-fuh2, bd-mapo, bd-u77k).
//!
//! Set `FOCR_RESIDENT_E2E_MODEL` to a v1 artifact and
//! `FOCR_RESIDENT_E2E_V2_MODEL` to an Unlimited-OCR-2 one. Without them the
//! tests log `skip_no_model`. Corpus pages come from `FOCR_CORPUS_DIR`, or
//! `~/.cache/franken_ocr/corpus/pages`.
//!
//! A run the daemon served never loads the model in the client, so the client's
//! peak memory is what says which path ran: an inline run holds gigabytes.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};

use serde_json::Value;

const MODEL_ENV: &str = "FOCR_RESIDENT_E2E_MODEL";
const CORPUS_ENV: &str = "FOCR_CORPUS_DIR";
const PAGE: &str = "page_0009.png";
/// A served client stays far under this; an inline run is several times over.
const SERVED_PEAK_BYTES: u64 = 1 << 30;

/// `focr ocr` under `/usr/bin/time`, which reports the client's peak memory.
fn ocr(page: &Path, model: &Path, resident: Option<&Path>) -> Command {
    let mut command = Command::new("/usr/bin/time");
    command
        .arg(if cfg!(target_os = "macos") {
            "-l"
        } else {
            "-v"
        })
        .arg(env!("CARGO_BIN_EXE_focr"))
        .arg("ocr")
        .arg(page)
        .arg("--model")
        .arg(model)
        .env_remove("FOCR_MODEL_PATH")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    match resident {
        Some(dir) => command
            .env_remove("FOCR_NO_RESIDENT")
            .env("FOCR_RESIDENT_DIR", dir)
            .env("FOCR_RESIDENT_IDLE_SECS", "120"),
        None => command.env("FOCR_NO_RESIDENT", "1"),
    };
    command
}

fn robot_ocr(page: &Path, model: &Path, resident: Option<&Path>) -> Command {
    let mut command = ocr(page, model, resident);
    command.arg("--robot");
    command
}

/// The stages a robot run entered, in order, and the markdown it finished with.
fn robot_run(stdout: &str) -> (Vec<String>, String) {
    let mut stages = Vec::new();
    let mut markdown = None;
    for line in stdout.lines() {
        let event: Value = serde_json::from_str(line).expect("one JSON object per line");
        match event["event"].as_str() {
            Some("stage") => stages.push(event["name"].as_str().expect("name").to_owned()),
            Some("run_complete") => markdown = event["markdown"].as_str().map(str::to_owned),
            _ => {}
        }
    }
    (stages, markdown.expect("run_complete"))
}

fn peak_bytes(out: &Output) -> u64 {
    let stderr = String::from_utf8_lossy(&out.stderr);
    stderr
        .lines()
        .find_map(|line| {
            let line = line.trim();
            if let Some(kib) = line.strip_prefix("Maximum resident set size (kbytes): ") {
                return kib.parse::<u64>().ok().map(|kib| kib * 1024);
            }
            line.strip_suffix("maximum resident set size")?
                .trim()
                .parse()
                .ok()
        })
        .unwrap_or_else(|| panic!("no peak memory in: {stderr}"))
}

fn finish(child: Child) -> Output {
    child.wait_with_output().expect("focr")
}

/// A client's answer, having checked that the daemon served it.
fn served(out: &Output) -> String {
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let peak = peak_bytes(out);
    assert!(peak < SERVED_PEAK_BYTES, "the client held {peak} bytes");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Stops the daemon the state file names, whatever the test did.
struct Daemon(PathBuf);

impl Drop for Daemon {
    fn drop(&mut self) {
        let Ok(entries) = fs::read_dir(&self.0) else {
            return;
        };
        for entry in entries.flatten() {
            let state = fs::read_to_string(entry.path()).unwrap_or_default();
            if let Ok(value) = serde_json::from_str::<Value>(&state)
                && let Some(pid) = value.get("pid").and_then(Value::as_u64)
            {
                let _ = Command::new("kill").arg(pid.to_string()).status();
            }
        }
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn a_resident_run_matches_an_inline_run() {
    let Some(model) = std::env::var_os(MODEL_ENV).map(PathBuf::from) else {
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
    let page = corpus.join(PAGE);
    let dir = std::env::temp_dir().join(format!("focr-resident-e2e-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("state dir");
    let _daemon = Daemon(dir.clone());

    let inline = ocr(&page, &model, None).output().expect("focr");
    assert_eq!(inline.status.code(), Some(0));
    assert!(
        peak_bytes(&inline) > SERVED_PEAK_BYTES,
        "an inline run holds the model"
    );
    let expected = String::from_utf8_lossy(&inline.stdout);

    let inline_robot = robot_ocr(&page, &model, None).output().expect("focr");
    assert_eq!(inline_robot.status.code(), Some(0));
    let (stages, markdown) = robot_run(&String::from_utf8_lossy(&inline_robot.stdout));
    assert_eq!(stages, ["load", "forward"], "inline");

    // Cold, then warm: the first spawns the daemon and is served by it, and
    // the stages it entered reach the client. A warm run has nothing to load.
    for (run, want) in [
        ("cold", &["load", "forward"][..]),
        ("warm", &["forward"][..]),
    ] {
        let out = robot_ocr(&page, &model, Some(&dir)).output().expect("focr");
        let (stages, text) = robot_run(&served(&out));
        assert_eq!(stages, want, "{run}");
        assert_eq!(text, markdown, "{run}");
    }
    let warm = ocr(&page, &model, Some(&dir)).output().expect("focr");
    assert_eq!(served(&warm), expected, "warm");

    // Two clients at once: the second is acknowledged while the first's page
    // runs, so it queues rather than loading a second copy of the model.
    let first = ocr(&page, &model, Some(&dir)).spawn().expect("focr");
    let second = ocr(&page, &model, Some(&dir)).spawn().expect("focr");
    assert_eq!(served(&finish(first)), expected, "first of two");
    assert_eq!(served(&finish(second)), expected, "second of two");
}

const V2_MODEL_ENV: &str = "FOCR_RESIDENT_E2E_V2_MODEL";

/// What one `focr ocr` run showed its caller.
struct Seen {
    code: Option<i32>,
    stdout: String,
    /// The run's own stderr lines: warnings, notes and the error.
    said: Vec<String>,
    peak: u64,
    wall_s: f64,
}

fn seen(mut command: Command) -> Seen {
    let started = std::time::Instant::now();
    let out = command.output().expect("focr");
    let wall_s = started.elapsed().as_secs_f64();
    let stderr = String::from_utf8_lossy(&out.stderr);
    Seen {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        said: stderr
            .lines()
            .filter(|line| line.starts_with("[focr] ") || line.starts_with("focr: "))
            .map(str::to_owned)
            .collect(),
        peak: peak_bytes(&out),
        wall_s,
    }
}

/// bd-u77k: Unlimited-OCR-2 served cold and warm reports what it reports
/// inline, byte for byte: `--json` with its `finish_reason` and `warnings[]`
/// (an ignored lever's among them), the stderr warnings and experimental
/// note, and the exit code, 0 for a page its `--max-page-tokens` cap closed
/// and 5 for one stopped at its position limit. Set
/// `FOCR_RESIDENT_E2E_V2_MODEL` to a v2 artifact beside its tokenizer.
#[test]
fn a_resident_v2_run_matches_an_inline_run() {
    let Some(model) = std::env::var_os(V2_MODEL_ENV).map(PathBuf::from) else {
        eprintln!("skip_no_model: {V2_MODEL_ENV} unset");
        return;
    };
    let corpus = std::env::var_os(CORPUS_ENV).map_or_else(
        || {
            PathBuf::from(std::env::var_os("HOME").expect("HOME"))
                .join(".cache/franken_ocr/corpus/pages")
        },
        PathBuf::from,
    );
    let page = corpus.join(PAGE);
    let root = std::env::temp_dir().join(format!("focr-resident-v2-e2e-{}", std::process::id()));

    // A one-position limit refuses with the prompt's length.
    let mut probe = ocr(&page, &model, None);
    probe.env("FOCR_TEST_MAX_POSITIONS", "1");
    let probe = seen(probe);
    let prompt: usize = probe
        .said
        .iter()
        .find_map(|line| line.split("prompt is ").nth(1))
        .and_then(|rest| rest.split(' ').next())
        .and_then(|n| n.replace(',', "").parse().ok())
        .unwrap_or_else(|| panic!("no prompt length in {:?}", probe.said));
    let limit = (prompt + 40).to_string();

    let cases: [(&str, &[&str], &[(&str, &str)], i32); 2] = [
        ("page_cap", &["--json", "--max-page-tokens", "48"], &[], 0),
        (
            "positions",
            &["--json"],
            &[("FOCR_TEST_MAX_POSITIONS", limit.as_str())],
            5,
        ),
    ];
    for (case, args, envs, code) in cases {
        // Each case's environment is its daemon's, so each has its own.
        let dir = root.join(case);
        fs::create_dir_all(&dir).expect("state dir");
        let _daemon = Daemon(dir.clone());
        let run = |resident: Option<&Path>| {
            let mut command = ocr(&page, &model, resident);
            command.args(args).env("FOCR_QKV_FUSED", "0");
            for (key, value) in envs {
                command.env(key, value);
            }
            seen(command)
        };
        let inline = run(None);
        assert_eq!(inline.code, Some(code), "{case}: {:?}", inline.said);
        assert!(inline.peak > SERVED_PEAK_BYTES, "{case}: inline holds the model");
        let json: Value = serde_json::from_str(&inline.stdout).expect("--json");
        let kinds: Vec<&str> = json["warnings"]
            .as_array()
            .expect("warnings[]")
            .iter()
            .filter_map(|w| w["kind"].as_str())
            .collect();
        assert!(kinds.contains(&"ignored_lever"), "{case}: {kinds:?}");
        assert_eq!(kinds.contains(&"v2_length"), code == 5, "{case}: {kinds:?}");
        let cold = run(Some(&dir));
        let warm = run(Some(&dir));
        for (run, served) in [("cold", &cold), ("warm", &warm)] {
            assert_eq!(served.code, inline.code, "{case} {run}: {:?}", served.said);
            assert_eq!(served.stdout, inline.stdout, "{case} {run}");
            assert_eq!(served.said, inline.said, "{case} {run}");
            assert!(
                served.peak < SERVED_PEAK_BYTES,
                "{case} {run}: the client held {} bytes",
                served.peak
            );
        }
        println!(
            "{}",
            serde_json::json!({
                "test": "a_resident_v2_run_matches_an_inline_run", "case": case,
                "exit_code": code, "finish_reason": json["finish_reason"],
                "warning_kinds": kinds, "inline_s": inline.wall_s, "cold_s": cold.wall_s,
                "warm_s": warm.wall_s, "inline_peak": inline.peak, "warm_peak": warm.peak,
            })
        );
    }
    let _ = fs::remove_dir_all(&root);
}
