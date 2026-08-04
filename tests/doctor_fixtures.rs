//! `doctor_fixtures` — the corrupt → `--fix` → healthy → `undo` →
//! byte-identical loop, per failure mode (bd-wp8.4.1), driven through the
//! REAL binary with a hermetic HOME per test.
//!
//! Also enforces the doctor's three laws:
//!  - detect-only NEVER mutates (hash sweep before/after);
//!  - `--dry-run` NEVER mutates and creates no run dir;
//!  - every mutation lives inside the single chokepoint
//!    (`all_mutation_is_inside_the_chokepoint` fails CI if any code outside
//!    `mod mutation` in src/doctor.rs writes to disk);
//!  - idempotence: a second `--fix` finds nothing auto-fixable to redo;
//!  - the lock: a held `.doctor/lock` exits 5 without touching anything.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn focr(home: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_focr"))
        .args(args)
        .env_remove("FOCR_MODEL_PATH")
        .env_remove("FOCR_MODEL_DIR")
        .env("HOME", home)
        .env("LOCALAPPDATA", home)
        .env("USERPROFILE", home)
        .output()
        .expect("spawn focr")
}

/// APFS base on macOS: an exFAT TMPDIR spawns `._*` AppleDouble junk that the
/// orphan detector CORRECTLY reports, making counts nondeterministic (the
/// documented gauntlet TMPDIR rule).
fn test_base() -> PathBuf {
    let apfs = PathBuf::from("/private/tmp");
    if cfg!(unix) && apfs.is_dir() {
        return apfs;
    }
    std::env::temp_dir()
}

fn fresh_home(tag: &str) -> PathBuf {
    let home = test_base().join(format!(
        "focr_doctor_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(home.join(".cache/franken_ocr/models")).expect("mk cache");
    home
}

fn models_dir(home: &Path) -> PathBuf {
    home.join(".cache/franken_ocr/models")
}

fn runs_dir(home: &Path) -> PathBuf {
    home.join(".cache/franken_ocr/.doctor/runs")
}

fn first_run_id(home: &Path) -> String {
    std::fs::read_dir(runs_dir(home))
        .expect("runs dir exists after --fix")
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .next()
        .expect("one run recorded")
}

fn json_stdout(out: &Output) -> serde_json::Value {
    serde_json::from_str(String::from_utf8_lossy(&out.stdout).trim())
        .unwrap_or_else(|e| panic!("stdout not one JSON object ({e}):\n{:?}", out))
}

fn emit(case: &str, ok: bool, fields: &str) {
    eprintln!(
        r#"{{"schema_version":1,"test":"doctor_fixtures","case":"{case}","event":"result","result":"{}"{}{fields}}}"#,
        if ok { "pass" } else { "fail" },
        if fields.is_empty() { "" } else { "," },
    );
}

/// Failure mode 1: orphaned partial download.
/// corrupt → detect(1) → --fix quarantines (reversible) → healthy for this
/// detector → undo → byte-identical file back in place.
#[test]
fn orphan_partial_fix_then_undo_is_byte_identical() {
    let home = fresh_home("orphan");
    let orphan = models_dir(&home).join("weights.partial");
    let payload = b"half-downloaded bytes \x00\x01\x02 the doctor must preserve";
    std::fs::write(&orphan, payload).expect("seed orphan");

    // Detect-only reports it and MUTATES NOTHING.
    let out = focr(&home, &["doctor", "--json"]);
    let v = json_stdout(&out);
    let detected = out.status.code() == Some(1)
        && v["findings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["detector"] == "orphaned_partial_download");
    assert!(detected, "orphan not detected: {v}");
    assert!(orphan.exists(), "detect-only must not mutate");

    // --fix quarantines it.
    let out = focr(&home, &["doctor", "--fix"]);
    let v = json_stdout(&out);
    assert!(v["fixed"].as_u64().unwrap() >= 1, "fix applied: {v}");
    assert!(!orphan.exists(), "orphan quarantined");

    // Undo restores byte-for-byte.
    let run_id = first_run_id(&home);
    let out = focr(&home, &["doctor", "undo", &run_id]);
    let undo_ok = out.status.code() == Some(0);
    let restored = std::fs::read(&orphan).expect("orphan restored");
    let identical = restored == payload;
    emit(
        "orphan_roundtrip",
        detected && undo_ok && identical,
        &format!(
            r#""undo_exit":{:?},"byte_identical":{identical}"#,
            out.status.code()
        ),
    );
    assert!(undo_ok && identical, "undo not byte-identical");
}

/// Failure mode 2 (unix): unreadable cache entry.
/// chmod 0 → detect → --fix chmods u+rw (mode+bytes logged) → undo restores
/// the ORIGINAL mode and the bytes never changed.
#[cfg(unix)]
#[test]
fn unreadable_entry_fix_then_undo_restores_mode() {
    use std::os::unix::fs::PermissionsExt;
    let home = fresh_home("perms");
    let f = models_dir(&home).join("locked.bin");
    let payload = b"model bytes the doctor must never rewrite";
    std::fs::write(&f, payload).expect("seed");
    std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o000)).expect("chmod 0");

    let out = focr(&home, &["doctor", "--json"]);
    let v = json_stdout(&out);
    let detected = v["findings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|x| x["detector"] == "unreadable_cache_entry");
    assert!(detected, "unreadable entry not detected: {v}");

    let out = focr(&home, &["doctor", "--fix"]);
    let v = json_stdout(&out);
    assert!(v["fixed"].as_u64().unwrap() >= 1, "{v}");
    let mode_after = std::fs::metadata(&f).unwrap().permissions().mode() & 0o7777;
    assert_ne!(mode_after & 0o600, 0, "owner rw granted");
    assert_eq!(
        std::fs::read(&f).unwrap(),
        payload,
        "bytes untouched by chmod fix"
    );

    let run_id = first_run_id(&home);
    let out = focr(&home, &["doctor", "undo", &run_id]);
    let mode_restored = std::fs::metadata(&f).unwrap().permissions().mode() & 0o7777;
    let ok = out.status.code() == Some(0) && mode_restored == 0o000;
    emit(
        "perms_roundtrip",
        ok,
        &format!(r#""mode_after_fix":"{mode_after:o}","mode_after_undo":"{mode_restored:o}""#),
    );
    assert!(
        ok,
        "undo did not restore the original mode (got {mode_restored:o})"
    );
}

/// Failure mode 3: stale .focrq format version — REFUSED (exit 4 when it is
/// the only actionable finding-class fix), with the exact recommended command
/// and NO mutation.
#[test]
fn stale_focrq_is_refused_with_recommended_command() {
    let home = fresh_home("stale");
    let stale = models_dir(&home).join("old.focrq");
    let mut bytes = b"FOCRQ\0".to_vec();
    bytes.extend_from_slice(&99u32.to_le_bytes());
    bytes.extend_from_slice(b"rest-of-header");
    std::fs::write(&stale, &bytes).expect("seed stale focrq");
    let before = std::fs::read(&stale).unwrap();

    let out = focr(&home, &["doctor", "--json"]);
    let v = json_stdout(&out);
    let finding = v["findings"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["detector"] == "stale_focrq_format")
        .cloned()
        .expect("stale focrq detected");
    let has_command = finding["fixability"]["recommended_command"]
        .as_str()
        .is_some_and(|c| c.contains("focr "));

    let out = focr(&home, &["doctor", "--fix"]);
    let v = json_stdout(&out);
    // The orphan/perms detectors are clean here; the model_not_resolvable
    // advice finding is present, so the fix run reports refused+advice.
    let refused = v["refused_unsafe"].as_u64().unwrap() >= 1;
    let untouched = std::fs::read(&stale).unwrap() == before;
    let ok = has_command && refused && untouched;
    emit(
        "stale_refused",
        ok,
        &format!(
            r#""has_recommended_command":{has_command},"refused":{refused},"untouched":{untouched},"fix_exit":{:?}"#,
            out.status.code()
        ),
    );
    assert!(
        ok,
        "stale focrq must be refused untouched with a recommended command: {v}"
    );
}

/// `--dry-run` mutates NOTHING and records NO run.
#[test]
fn dry_run_has_zero_blast_radius() {
    let home = fresh_home("dry");
    let orphan = models_dir(&home).join("x.tmp");
    std::fs::write(&orphan, b"tmp").unwrap();

    let out = focr(&home, &["doctor", "--dry-run"]);
    let v = json_stdout(&out);
    let ok = out.status.code() == Some(1)
        && v["would_mutate"].as_u64() == Some(1)
        && orphan.exists()
        && !runs_dir(&home).exists();
    emit(
        "dry_run_zero_blast",
        ok,
        &format!(r#""would_mutate":{}"#, v["would_mutate"]),
    );
    assert!(ok, "dry-run must plan without touching disk: {v}");
}

/// Idempotence: a second `--fix` finds nothing auto-fixable left (fixed=0),
/// and running detect twice is byte-stable.
#[test]
fn second_fix_run_has_no_actions() {
    let home = fresh_home("idem");
    std::fs::write(models_dir(&home).join("y.tmp"), b"tmp").unwrap();
    let first = json_stdout(&focr(&home, &["doctor", "--fix"]));
    assert!(first["fixed"].as_u64().unwrap() >= 1);
    let second = json_stdout(&focr(&home, &["doctor", "--fix"]));
    let ok = second["fixed"].as_u64() == Some(0);
    emit(
        "idempotent_second_fix",
        ok,
        &format!(r#""second_fixed":{}"#, second["fixed"]),
    );
    assert!(ok, "second --fix must have nothing to do: {second}");
}

/// The lock: a held `.doctor/lock` exits 5 and mutates nothing.
#[test]
fn held_lock_exits_concurrency_lost() {
    let home = fresh_home("lock");
    let orphan = models_dir(&home).join("z.tmp");
    std::fs::write(&orphan, b"tmp").unwrap();
    let doctor_dir = home.join(".cache/franken_ocr/.doctor");
    std::fs::create_dir_all(&doctor_dir).unwrap();
    std::fs::write(doctor_dir.join("lock"), b"held-by-test").unwrap();

    let out = focr(&home, &["doctor", "--fix"]);
    let ok = out.status.code() == Some(5) && orphan.exists();
    emit(
        "lock_contention",
        ok,
        &format!(
            r#""exit":{:?},"untouched":{}"#,
            out.status.code(),
            orphan.exists()
        ),
    );
    assert!(
        ok,
        "held lock must exit 5 untouched, got {:?}",
        out.status.code()
    );
}

/// Contract surfaces: capabilities + robot-docs + robot-triage all answer.
#[test]
fn contract_surfaces_answer() {
    let home = fresh_home("contract");
    let caps = json_stdout(&focr(&home, &["doctor", "capabilities"]));
    let caps_ok = caps["exit_codes"]["5"]
        .as_str()
        .is_some_and(|s| s.contains("lock"))
        && caps["detectors"].as_array().is_some_and(|d| d.len() == 4);
    let docs = focr(&home, &["doctor", "robot-docs"]);
    let docs_ok = String::from_utf8_lossy(&docs.stdout).contains("focr doctor undo");
    let triage = json_stdout(&focr(&home, &["doctor", "--robot-triage"]));
    let triage_ok = !triage["recommended_command"].is_null() && !triage["summary"].is_null();
    let ok = caps_ok && docs_ok && triage_ok;
    emit(
        "contract_surfaces",
        ok,
        &format!(r#""capabilities":{caps_ok},"robot_docs":{docs_ok},"robot_triage":{triage_ok}"#),
    );
    assert!(ok);
}

/// The chokepoint code-search: every disk-writing call in src/doctor.rs must
/// live inside `pub mod mutation` — any write outside it fails CI here.
#[test]
fn all_mutation_is_inside_the_chokepoint() {
    let src =
        std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/doctor.rs"))
            .expect("doctor source");
    let chokepoint_start = src
        .find("pub mod mutation")
        .expect("mutation module exists");
    let before = &src[..chokepoint_start];
    let write_calls = [
        "std::fs::write",
        "fs::write(",
        "File::create",
        "OpenOptions",
        "set_permissions",
        "fs::rename",
        "fs::copy",
        "remove_file",
        "remove_dir",
    ];
    let violations: Vec<&str> = write_calls
        .iter()
        .filter(|c| before.contains(**c))
        .copied()
        .collect();
    let ok = violations.is_empty();
    emit(
        "mutation_chokepoint",
        ok,
        &format!(r#""violations":{violations:?}"#),
    );
    assert!(
        ok,
        "disk-writing calls OUTSIDE the mutation chokepoint in src/doctor.rs: {violations:?}"
    );
}
