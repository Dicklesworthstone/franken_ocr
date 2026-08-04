//! `fault_suite` — the input-fault robustness leg of the conformance pillar
//! (bd-15kd; plan §8.5's third leg: ladder + metamorphic + FAULT).
//!
//! Every fault class must surface as a TYPED [`FocrError`] with its frozen
//! §7.4 exit code — never a panic, never a hang, never a generic exit 1 where
//! a specific code exists. The legs:
//!
//!  - **corrupt / truncated / zero-byte images** → `InputDecode` (exit 4),
//!    always-on through the public `preprocess_image` entry;
//!  - **corrupt model artifact** → `FormatMismatch` (exit 7), through the
//!    REAL binary (bogus `.focrq` magic AND a lying `.safetensors` header);
//!  - **malformed task/model combo** → usage (exit 2), real binary;
//!  - **missing model** → `ModelNotFound` (exit 3) — already gated suite-wide
//!    (`/nonexistent` proof pattern, `exit_code_conformance`), asserted here
//!    once more through the hermetic binary for the leg's completeness;
//!  - **model-gated legs** (corrupt image AFTER a real model loads → exit 4;
//!    stage-budget exceeded → exit 5): skip-with-SUCCESS without weights,
//!    same contract as every armed rung. Mid-decode CANCELLATION (exit 6) is
//!    deliberately owned by bd-1ryu (needs process isolation for the global
//!    shutdown flag) — enumerated here so the leg's coverage is honest.
//!
//! PDF decompression-bomb bounding (the other input-fault class) is proven by
//! the `src/pdf.rs` unit tests (bd-2zpu); this suite covers the decode paths
//! those don't.

use std::path::PathBuf;
use std::process::{Command, Output};

use franken_ocr::preprocess::{self, PreprocessMode};
use franken_ocr::{FocrError, MODEL_PATH_ENV};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("focr_fault_suite_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    dir.join(name)
}

fn emit(case: &str, result: &str, fields: &str) {
    eprintln!(
        r#"{{"schema_version":1,"test":"fault_suite","case":"{case}","event":"result","result":"{result}"{}{fields}}}"#,
        if fields.is_empty() { "" } else { "," },
    );
}

/// Run the real binary hermetically: no model resolvable (HOME pinned to an
/// empty temp dir so a developer's pulled cache artifact cannot leak in).
fn run_focr(args: &[&str], extra_env: &[(&str, &str)]) -> Output {
    let home = scratch("hermetic_home");
    let _ = std::fs::create_dir_all(&home);
    let mut c = Command::new(env!("CARGO_BIN_EXE_focr"));
    c.args(args)
        .env_remove(MODEL_PATH_ENV)
        .env_remove("FOCR_MODEL_DIR")
        .env("HOME", &home)
        .env("LOCALAPPDATA", &home)
        .env("USERPROFILE", &home);
    for (k, v) in extra_env {
        c.env(k, v);
    }
    c.output().expect("spawn focr")
}

/// A real, decodable page image committed in-tree (content is irrelevant to
/// the fault legs; it just must decode).
fn real_image() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/got/sample_text.png")
}

// ───────────────────── always-on: image decode faults ─────────────────────

/// Corrupt, truncated, and zero-byte inputs all return the TYPED
/// `InputDecode` error (exit 4) through the public preprocess entry — never
/// a panic, never a misclassified error kind.
#[test]
fn corrupt_truncated_and_zero_byte_images_are_typed_input_decode() {
    let real = std::fs::read(real_image()).expect("fixture image readable");
    let cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "garbage_bytes",
            b"this is not an image at all \x00\x01\x02".to_vec(),
        ),
        ("truncated_png", real[..real.len() / 3].to_vec()),
        ("zero_byte", Vec::new()),
        // A PNG signature with nothing behind it: the header-sniff succeeds,
        // the decode itself must still fail TYPED.
        (
            "signature_only",
            vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A],
        ),
    ];
    for (name, bytes) in cases {
        let p = scratch(&format!("{name}.png"));
        std::fs::write(&p, &bytes).expect("write fault input");
        let err = preprocess::preprocess_image(&p, PreprocessMode::default())
            .expect_err("fault input must not preprocess");
        let (kind, exit) = (err.kind(), err.exit_code());
        let pass = matches!(&err, FocrError::InputDecode(_)) && exit == 4;
        emit(
            &format!("image_{name}"),
            if pass { "pass" } else { "fail" },
            &format!(
                r#""error_kind":"{kind}","exit_code":{exit},"bytes":{}"#,
                bytes.len()
            ),
        );
        assert!(
            pass,
            "{name}: expected typed InputDecode/exit 4, got kind={kind} exit={exit}: {err}"
        );
        let _ = std::fs::remove_file(&p);
    }
}

// ───────────────── always-on: model-artifact faults (real binary) ─────────────────

/// A model artifact with a bogus magic AND one with a lying safetensors
/// header both exit 7 (`format_mismatch`) through the real binary — the
/// loader must classify a corrupt artifact as a FORMAT problem, not a
/// missing model or a generic failure.
#[test]
fn corrupt_model_artifact_exits_format_mismatch() {
    let img = real_image();
    let img = img.to_str().unwrap();
    let cases: Vec<(&str, &str, Vec<u8>)> = vec![
        (
            "bogus_magic_focrq",
            "bogus.focrq",
            b"NOTFOCRQ garbage body".to_vec(),
        ),
        // 8-byte safetensors header length that overruns the 4 KB file — the
        // exFAT-AppleDouble class the ladder resolver hit (bd-re8.19).
        ("lying_safetensors_header", "model.safetensors", {
            let mut v = u64::MAX.to_le_bytes().to_vec();
            v.extend_from_slice(&[0u8; 64]);
            v
        }),
    ];
    for (name, filename, bytes) in cases {
        let p = scratch(filename);
        std::fs::write(&p, &bytes).expect("write bogus artifact");
        let out = run_focr(&["ocr", "--model", p.to_str().unwrap(), img], &[]);
        let code = out.status.code();
        let pass = code == Some(7);
        emit(
            &format!("artifact_{name}"),
            if pass { "pass" } else { "fail" },
            &format!(r#""exit_code":{code:?}"#),
        );
        assert!(
            pass,
            "{name}: expected exit 7 (format_mismatch), got {code:?}\nstderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let _ = std::fs::remove_file(&p);
    }
}

/// Missing model stays the clean typed exit 3 through the hermetic binary —
/// the leg's baseline (proves the artifact faults above are NOT just
/// resolution failures wearing a different hat).
#[test]
fn missing_model_exits_model_not_found() {
    let img = real_image();
    let out = run_focr(
        &[
            "ocr",
            "--model",
            "/nonexistent/fault_suite/model.focrq",
            img.to_str().unwrap(),
        ],
        &[],
    );
    let code = out.status.code();
    let pass = code == Some(3);
    emit(
        "missing_model",
        if pass { "pass" } else { "fail" },
        &format!(
            r#""exit_code":{code:?},"fallback_target":"/nonexistent/fault_suite/model.focrq","native_path_ran":true"#
        ),
    );
    assert!(pass, "expected exit 3, got {code:?}");
}

/// A task the named model knowably cannot serve is a USAGE error (exit 2)
/// before any load is attempted — malformed request, not a runtime fault.
#[test]
fn malformed_task_model_combo_is_usage() {
    let img = real_image();
    let out = run_focr(
        &[
            "ocr",
            "--task",
            "music",
            "--model",
            "/nonexistent/unlimited-ocr.int8.focrq",
            img.to_str().unwrap(),
        ],
        &[],
    );
    let code = out.status.code();
    let pass = code == Some(2);
    emit(
        "task_model_mismatch",
        if pass { "pass" } else { "fail" },
        &format!(
            r#""exit_code":{code:?},"task":"music","model":"unlimited-ocr (knowably not tromr)""#
        ),
    );
    assert!(
        pass,
        "expected usage exit 2 for --task music on a knowably-non-tromr model, got {code:?}\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ───────────────── model-gated legs (skip-with-SUCCESS unarmed) ─────────────────

/// With REAL weights resolvable, a corrupt page image must fail exit 4
/// AFTER the model loads (the decode fault, not a resolution artifact), and
/// a starved stage budget must fail exit 5. Without weights both legs log
/// `skip_no_model` and pass — same contract as every armed rung. The
/// cancellation leg (exit 6) is owned by bd-1ryu (process isolation).
#[test]
fn armed_corrupt_image_and_budget_faults_are_typed() {
    let Some(model) = std::env::var_os(MODEL_PATH_ENV)
        .map(PathBuf::from)
        .filter(|p| p.exists())
    else {
        emit(
            "armed_faults",
            "skip_no_model",
            r#""reason":"FOCR_MODEL_PATH unset/absent — armed corrupt-image (exit 4) and stage-budget (exit 5) legs need real weights; the always-on legs above cover the typed decode/artifact paths","native_path_ran":true,"fallback_target":"/nonexistent""#,
        );
        return;
    };
    let model = model.to_str().unwrap().to_string();

    // Corrupt page after a real model load: typed exit 4.
    let bad = scratch("armed_corrupt_page.png");
    std::fs::write(&bad, b"not an image").expect("write corrupt page");
    let out = run_focr(&["ocr", "--model", &model, bad.to_str().unwrap()], &[]);
    let code = out.status.code();
    let decode_pass = code == Some(4);
    emit(
        "armed_corrupt_image",
        if decode_pass { "pass" } else { "fail" },
        &format!(r#""exit_code":{code:?},"model_loaded":true"#),
    );
    assert!(
        decode_pass,
        "armed corrupt page: expected exit 4, got {code:?}"
    );
    let _ = std::fs::remove_file(&bad);

    // Starved forward-stage budget: typed exit 5 (timeout/budget).
    let img = real_image();
    let out = run_focr(
        &["ocr", "--model", &model, img.to_str().unwrap()],
        &[("FOCR_STAGE_BUDGET_FORWARD_MS", "1")],
    );
    let code = out.status.code();
    let budget_pass = code == Some(5);
    emit(
        "armed_stage_budget",
        if budget_pass { "pass" } else { "fail" },
        &format!(r#""exit_code":{code:?},"budget_ms":1"#),
    );
    assert!(
        budget_pass,
        "starved stage budget: expected exit 5, got {code:?}"
    );
}
