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
    seen_of(&out, started.elapsed().as_secs_f64())
}

fn seen_of(out: &Output, wall_s: f64) -> Seen {
    let stderr = String::from_utf8_lossy(&out.stderr);
    Seen {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        said: stderr
            .lines()
            .filter(|line| line.starts_with("[focr] ") || line.starts_with("focr: "))
            .map(str::to_owned)
            .collect(),
        peak: peak_bytes(out),
        wall_s,
    }
}

/// An [`ocr`] command sent one SIGINT `after` into its run, as Ctrl+C sends
/// a bare `focr`. The signal goes to `focr` itself, not to its group: macOS
/// `time` passes on a SIGINT it receives, so focr would get two, and a second
/// is its hard exit 130.
#[cfg(unix)]
fn interrupted(mut command: Command, after: std::time::Duration) -> Seen {
    let started = std::time::Instant::now();
    let child = command.spawn().expect("focr");
    std::thread::sleep(after);
    let found = Command::new("pgrep")
        .args(["-P", &child.id().to_string()])
        .output()
        .expect("pgrep");
    let focr = String::from_utf8_lossy(&found.stdout).trim().to_owned();
    let signalled = !focr.is_empty()
        && Command::new("kill")
            .args(["-INT", &focr])
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
    let out = child.wait_with_output().expect("focr");
    let seen = seen_of(&out, started.elapsed().as_secs_f64());
    assert!(
        signalled,
        "kill -INT {focr:?} after {after:?}: the run had ended, exit {:?}: {:?}",
        seen.code, seen.said
    );
    seen
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

    // (case, extra args, extra env, exit code)
    type Case<'a> = (&'a str, &'a [&'a str], &'a [(&'a str, &'a str)], i32);
    let cases: [Case; 2] = [
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
        assert!(
            inline.peak > SERVED_PEAK_BYTES,
            "{case}: inline holds the model"
        );
        let json: Value = serde_json::from_str(&inline.stdout).expect("--json");
        let kinds: Vec<&str> = json["warnings"]
            .as_array()
            .expect("warnings[]")
            .iter()
            .filter_map(|w| w["kind"].as_str())
            .collect();
        assert!(kinds.contains(&"ignored_lever"), "{case}: {kinds:?}");
        assert_eq!(kinds.contains(&"v2_length"), code == 5, "{case}: {kinds:?}");
        // bd-jem5: a page the cap closed is named by a run warning.
        let capped = json["pages"]
            .as_array()
            .is_some_and(|pages| pages.iter().any(|p| p["finish"] == "page_cap"));
        assert_eq!(kinds.contains(&"v2_page_cap"), capped, "{case}: {kinds:?}");
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

/// bd-toeb: Ctrl+C on a served Unlimited-OCR-2 page ends it as it ends
/// inline: the text decoded so far is written and the run exits 6. The
/// daemon is spawned by an earlier run, as in a shell, so the signal reaches
/// the client alone and only the client's half-close can stop the daemon's
/// decode. A cancel lands at a time, not a token, so the two runs stop at
/// different points; each must write a prefix of the page the uncancelled
/// run writes. The page is a dense one under a 300-token cap, so the decode
/// outlasts the timing noise: a short page's served decode is over before a
/// cancel aimed at its middle arrives.
#[cfg(unix)]
#[test]
fn a_cancelled_resident_v2_page_writes_what_it_has() {
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
    let page = corpus.join("page_0590.png");
    let dir = std::env::temp_dir().join(format!("focr-resident-cancel-e2e-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("state dir");
    let _daemon = Daemon(dir.clone());
    let run = |resident: Option<&Path>, tokens: &str| {
        let mut command = ocr(&page, &model, resident);
        command
            .args(["--raw", "--max-page-tokens", tokens])
            .env("FOCR_QKV_FUSED", "0");
        command
    };

    assert_eq!(seen(run(Some(&dir), "1")).code, Some(0), "cold");

    let mut full_pages = Vec::new();
    for (arm, resident) in [("inline", None), ("served", Some(dir.as_path()))] {
        // A one-token run times all but the decode, a capped run the page.
        let setup = seen(run(resident, "1"));
        let full = seen(run(resident, "300"));
        assert_eq!(full.code, Some(0), "{arm} full: {:?}", full.said);
        let after = (setup.wall_s + full.wall_s) / 2.0;
        let cancelled = interrupted(
            run(resident, "300"),
            std::time::Duration::from_secs_f64(after),
        );
        assert_eq!(cancelled.code, Some(6), "{arm}: {:?}", cancelled.said);
        let partial = cancelled.stdout.trim_end();
        let whole = full.stdout.trim_end();
        assert!(!partial.is_empty(), "{arm}: the cancel wrote nothing");
        assert!(
            partial.len() < whole.len() && whole.starts_with(partial),
            "{arm}: {partial:?} is not a cut of {whole:?}"
        );
        if resident.is_some() {
            assert!(
                cancelled.peak < SERVED_PEAK_BYTES,
                "the cancelled client held {} bytes: it ran inline",
                cancelled.peak
            );
            // The daemon serves on, its cancelled decode stopped: one still
            // decoding the abandoned page would queue this run behind the rest.
            let next = seen(run(resident, "1"));
            assert_eq!(next.code, Some(0), "after the cancel: {:?}", next.said);
            assert!(next.peak < SERVED_PEAK_BYTES, "after the cancel: inline");
            let rest = full.wall_s - after;
            assert!(
                next.wall_s < setup.wall_s + rest / 2.0,
                "the daemon decoded on after the cancel: {:.1}s for a {:.1}s run",
                next.wall_s,
                setup.wall_s
            );
            println!(
                "{}",
                serde_json::json!({
                    "test": "a_cancelled_resident_v2_page_writes_what_it_has",
                    "after_cancel_s": next.wall_s, "setup_s": setup.wall_s,
                })
            );
        }
        println!(
            "{}",
            serde_json::json!({
                "test": "a_cancelled_resident_v2_page_writes_what_it_has", "arm": arm,
                "setup_s": setup.wall_s, "full_s": full.wall_s, "cancel_after_s": after,
                "cancelled_s": cancelled.wall_s, "partial_chars": partial.chars().count(),
                "page_chars": whole.chars().count(),
            })
        );
        full_pages.push(full.stdout);
    }
    assert_eq!(
        full_pages[0], full_pages[1],
        "inline and served pages differ"
    );
}
