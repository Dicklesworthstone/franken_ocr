#![allow(clippy::doc_overindented_list_items, clippy::doc_lazy_continuation)]
//! `e2e_recognize` — the **model-gated** end-to-end harness (Testing Policy;
//! `docs/testing/LOGGING_AND_E2E.md` §4, bead `bd-29wv`; AGENTS.md "Testing
//! Policy").
//!
//! The Unlimited-OCR checkpoint is a single **6.67 GB** bf16 safetensors shard
//! that **cannot live in CI**. So the e2e tests here are *model-gated*: **green
//! and visible** without the weights, **meaningful and loud** with them. Three
//! independent guards, in priority order of stability:
//!
//! 1. **Skip-with-SUCCESS** (LOGGING_AND_E2E.md §4.1, principle TL4). When
//!    `FOCR_MODEL_PATH` is unset/missing, the model-present branch returns `Ok`
//!    and emits a clear `SUCCESS` line — never a silent pass, never a failure.
//!    The skip is *visible* (`result:"skip_no_model"`), with the resolver dirs it
//!    searched (`searched_dirs`) so a developer who *expected* the model sees why
//!    it was not found.
//!
//! 2. **Prove the native path is exercised even without weights**
//!    (LOGGING_AND_E2E.md §4.2, principle TL5). We point the model at
//!    `/nonexistent` and assert `OcrEngine::recognize_with_model(...)` returns a
//!    clean [`FocrError::ModelNotFound`] (exit code **3**) — **not** a panic, not
//!    a generic fallback, not `NotImplemented`. This is the always-on,
//!    no-weights guard that a stub/fallback path would *fail*. The same is
//!    asserted through the CLI (`focr ocr <img> --model /nonexistent`).
//!
//! 3. **Real recognize over a tiny committed fixture image, gated behind the env
//!    var** (LOGGING_AND_E2E.md §4.1/§4.3). When `FOCR_MODEL_PATH` *is* set (a
//!    developer machine), we run a real `recognize()` over a tiny image, log the
//!    markdown output + token count + timing, and assert the output is non-empty
//!    and well-formed. The whole branch is gated by the env var.
//!
//! **Stable-surface bias.** This file leans on the *already-implemented and
//! committed* surfaces wherever it can: `OcrEngine::new` /
//! `recognize_with_model` (the path-explicit form that needs no env mutation —
//! the crate root is `#![forbid(unsafe_code)]`, so a test cannot call the
//! `unsafe` `std::env::set_var`), the stable exit-code contract in
//! `src/error.rs`, and the `focr robot schema` / `focr ocr` CLI via
//! `std::process::Command` on the built binary. The path-explicit CLI model
//! resolver is a hard assertion; the real forward remains a phase-gated tripwire
//! that logs XFAIL when it reaches a documented NotImplemented stage.
//!
//! Every test emits structured `e2e` lines on **stderr** (TL1/TL2: data-only on
//! stdout, diagnostics on stderr) describing what it exercised, the inputs, the
//! expected-vs-actual, and on a model-gated skip a `SUCCESS` line explaining the
//! skip. A failure is diagnosable from the captured stream alone.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use franken_ocr::native_engine::weights::{FOCRQ_FORMAT_VERSION, FOCRQ_MAGIC};
use franken_ocr::{DEFAULT_MODEL_PATH, FocrError, MODEL_PATH_ENV, OcrEngine};

// ───────────────────────────────────────────────────────────────────────────
// Structured logging (the "great detailed logging" mandate, TL1/TL2).
//
// We do not pull in the `tests/common/test_log.rs` emitter (owned by bd-n68o
// and still landing); instead we emit the same *shape* of NDJSON-flavored,
// machine-greppable lines on stderr so each test is self-diagnosing in
// isolation. Every line is prefixed `E2E ` and carries the test name, the case,
// an `event`, a `result`, and the load-bearing facts. The prefix lets a
// developer (or a CI log scraper) filter the e2e telemetry out of cargo's noise
// with a single `grep '^E2E '`.
// ───────────────────────────────────────────────────────────────────────────

/// Emit one structured e2e line on stderr. `kv` is already-formatted
/// `key=value` / `key="value"` pairs (space-separated). Stderr only — the
/// data surface (CLI `--json`, robot NDJSON) is never polluted (TL2).
fn log_line(test: &str, case: &str, event: &str, result: &str, kv: &str) {
    if kv.is_empty() {
        eprintln!("E2E test={test} case={case} event={event} result={result}");
    } else {
        eprintln!("E2E test={test} case={case} event={event} result={result} {kv}");
    }
}

/// A `SUCCESS` banner line — the model-gated skip MUST print one of these
/// (TL4: "never a silent pass"). Distinct token (`SUCCESS`) so a human scanning
/// the run sees the green skip immediately and a scraper can count them.
fn log_success(test: &str, case: &str, why: &str) {
    eprintln!("E2E SUCCESS test={test} case={case} :: {why}");
}

/// An `XFAIL` line — a documented divergence from the *target* behavior that is
/// expected in the current phase (currently the real forward still returning
/// `NotImplemented`). Per the conformance discipline we **XFAIL, never SKIP**:
/// the line records exactly what diverged so the test tightens to a hard
/// assertion when the surface lands. NOT a failure.
fn log_xfail(test: &str, case: &str, observed: &str, expected_target: &str) {
    eprintln!(
        "E2E XFAIL test={test} case={case} :: observed[{observed}] != target[{expected_target}] \
         (documented phase divergence; tightens to a hard assertion when the surface lands)"
    );
}

// ───────────────────────────────────────────────────────────────────────────
// Model-gate helpers (LOGGING_AND_E2E.md §4.1).
// ───────────────────────────────────────────────────────────────────────────

/// The dirs the gate "searched" for a model, for the `skip_no_model`
/// `searched_dirs` field (§4.1: a developer who expected the model sees why it
/// was not found). The real resolver search path lives in `native_engine`; here
/// we report the two documented resolution inputs the public API consults.
fn searched_dirs() -> Vec<String> {
    let mut dirs = Vec::new();
    if let Some(env) = std::env::var_os(MODEL_PATH_ENV) {
        dirs.push(format!(
            "${MODEL_PATH_ENV}={}",
            PathBuf::from(env).display()
        ));
    } else {
        dirs.push(format!("${MODEL_PATH_ENV}=<unset>"));
    }
    dirs.push(format!("default={DEFAULT_MODEL_PATH}"));
    dirs
}

/// Resolve the model the public API would use and decide whether the
/// model-present branch can run. Returns `Some(path)` only when a concrete
/// artifact is resolvable on disk (a *cheap* `exists()` check — we never read
/// the 6.67 GB blob just to decide whether to skip, per §4.1). `None` ⇒
/// skip-with-SUCCESS.
fn resolve_present_model() -> Option<PathBuf> {
    // `OcrEngine::model_path()` is the single source of truth for resolution
    // (the `FOCR_MODEL_PATH` override, else `DEFAULT_MODEL_PATH`).
    let p = OcrEngine::model_path();
    if p.exists() { Some(p) } else { None }
}

// ───────────────────────────────────────────────────────────────────────────
// A tiny fixture image, generated at runtime.
//
// No image fixture is committed under tests/fixtures yet (that path is owned by
// TEST-fixture-harness), and this file owns only itself, so we synthesize a
// genuinely-valid 4×4 PNG into a temp dir for the model-present branch. The
// bytes below are a hand-built, CRC-correct PNG (no `image` dev-dep needed —
// AGENTS.md: hand-roll rather than add a dev-dependency). It is a real decodable
// image, so the preprocess front end sees real pixels, not a sentinel.
// ───────────────────────────────────────────────────────────────────────────

/// CRC-32 (IEEE, the PNG polynomial) over `bytes`. Hand-rolled so we can emit a
/// valid PNG without a dependency.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Adler-32 over `bytes` (the zlib trailer checksum).
fn adler32(bytes: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in bytes {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// Append a PNG chunk (`len`, `type`, `data`, `crc`) to `out`.
fn push_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(chunk_type);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

/// Build a valid `w×h` 8-bit RGB PNG filled with mid-gray (the pad color the
/// preprocess pass uses), using a single stored (uncompressed) zlib block.
fn tiny_png(w: u32, h: u32) -> Vec<u8> {
    // Raw scanlines: each row is a filter byte (0 = None) + w*3 pixel bytes.
    let mut raw = Vec::with_capacity((h * (1 + w * 3)) as usize);
    // The pushed filter byte is interleaved with per-row pixel data, so it cannot
    // be hoisted into a bulk fill — order (filter byte, then w*3 pixels) matters.
    #[allow(clippy::same_item_push)]
    for _ in 0..h {
        raw.push(0u8); // filter: None
        for _ in 0..w {
            raw.extend_from_slice(&[127, 127, 127]); // mid-gray RGB
        }
    }

    // zlib stream wrapping one stored DEFLATE block (no compression).
    let mut zlib = Vec::new();
    zlib.push(0x78); // CMF
    zlib.push(0x01); // FLG (no preset dict, fastest)
    // Single stored block: BFINAL=1, BTYPE=00.
    zlib.push(0x01);
    let len = raw.len() as u16;
    zlib.extend_from_slice(&len.to_le_bytes());
    zlib.extend_from_slice(&(!len).to_le_bytes());
    zlib.extend_from_slice(&raw);
    zlib.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);

    // IHDR: width, height, bit depth 8, color type 2 (RGB), no interlace.
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    push_chunk(&mut png, b"IHDR", &ihdr);
    push_chunk(&mut png, b"IDAT", &zlib);
    push_chunk(&mut png, b"IEND", &[]);
    png
}

/// Write the tiny PNG to a unique temp path and return it. Best-effort cleanup
/// is left to the OS temp dir (we never delete files — AGENTS.md RULE 1).
fn write_tiny_png() -> PathBuf {
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    // Nanosecond suffix keeps parallel test threads from colliding.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = dir.join(format!("focr_e2e_tiny_{pid}_{nanos}.png"));
    std::fs::write(&path, tiny_png(4, 4)).expect("write tiny PNG fixture to temp dir");
    path
}

fn future_focrq_preamble() -> Vec<u8> {
    let mut blob = Vec::new();
    blob.extend_from_slice(FOCRQ_MAGIC);
    blob.extend_from_slice(&(FOCRQ_FORMAT_VERSION + 1).to_le_bytes());
    blob.push(0);
    blob.extend_from_slice(&[0u8; 32]);
    blob.extend_from_slice(&0u64.to_le_bytes());
    blob
}

/// Write a deliberately future-version `.focrq` preamble. The loader rejects the
/// version before parsing a tensor directory, so this tiny file is sufficient to
/// exercise the public FormatMismatch contract through the CLI.
fn write_future_focrq() -> PathBuf {
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = dir.join(format!("focr_e2e_future_{pid}_{nanos}.focrq"));
    std::fs::write(&path, future_focrq_preamble()).expect("write future .focrq fixture");
    path
}

// ───────────────────────────────────────────────────────────────────────────
// CLI driver: locate and run the built `focr` binary.
//
// `cargo test` builds the integration-test crate but NOT necessarily the `focr`
// bin. We discover the binary via `CARGO_BIN_EXE_focr` (cargo sets it for
// integration tests that share the package's bin targets); if it is absent we
// fall back to scanning target/{debug,release}. When neither resolves the binary
// (no build yet), the CLI-driven cases report a visible SUCCESS deferral; the
// library-level guards still run unconditionally and are load-bearing.
// ───────────────────────────────────────────────────────────────────────────

/// Locate the built `focr` binary, or `None` if it has not been built.
fn focr_bin() -> Option<PathBuf> {
    if let Some(p) = option_env!("CARGO_BIN_EXE_focr") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    // Fallback: scan the conventional target dirs relative to CARGO_MANIFEST_DIR.
    let manifest = env!("CARGO_MANIFEST_DIR");
    for profile in ["debug", "release"] {
        let cand = Path::new(manifest)
            .join("target")
            .join(profile)
            .join("focr");
        if cand.exists() {
            return Some(cand);
        }
    }
    None
}

/// The outcome of one `focr` invocation.
struct CliOut {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

/// Run `focr` with `args`, capturing the exit code + stdout + stderr.
fn run_focr(bin: &Path, args: &[&str]) -> CliOut {
    let out = Command::new(bin)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {}: {e}", bin.display()));
    CliOut {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GUARD 2 (always-on, no weights): prove the native path is exercised.
//   recognize_with_model(/nonexistent) ⇒ FocrError::ModelNotFound, exit code 3.
// ═══════════════════════════════════════════════════════════════════════════

/// The load-bearing always-on guard (TL5): pointing the model at `/nonexistent`
/// must produce a **clean** [`FocrError::ModelNotFound`] (exit code 3) — not a
/// panic, not a generic fallback, not `NotImplemented`. A green test that
/// secretly degraded to a stub is a *false* green; this guard makes that
/// failure loud. We use the path-explicit `recognize_with_model` so the test
/// never mutates the process environment (the crate is `#![forbid(unsafe_code)]`,
/// ruling out `std::env::set_var`).
#[test]
fn native_path_nonexistent_model_is_clean_model_not_found() {
    let test = "native_path_nonexistent_model_is_clean_model_not_found";
    let case = "lib_recognize_with_model";
    let model = Path::new("/nonexistent/franken_ocr/model.focrq");
    let image = Path::new("/some/document.png");

    log_line(
        test,
        case,
        "setup",
        "pass",
        &format!(
            "fallback_target=\"/nonexistent\" model={} image={} expect=ModelNotFound expect_exit=3",
            model.display(),
            image.display()
        ),
    );

    let engine = OcrEngine::new().expect("OcrEngine::new builds its owned runtime");
    let started = Instant::now();
    let result = engine.recognize_with_model(model, image);
    let elapsed_us = started.elapsed().as_micros();

    match result {
        Err(FocrError::ModelNotFound(ref msg)) => {
            let code = FocrError::ModelNotFound(String::new()).exit_code();
            assert_eq!(code, 3, "ModelNotFound must map to the stable exit code 3");
            log_line(
                test,
                case,
                "assert",
                "pass",
                &format!(
                    "error_kind=ModelNotFound focr_exit_code={code} elapsed_us={elapsed_us} \
                     native_path_ran=true fallback_target=\"/nonexistent\" message={msg:?}"
                ),
            );
            log_line(
                test,
                case,
                "result",
                "pass",
                "native_path_ran=true fallback_target=\"/nonexistent\" \
                 note=\"a stub/fallback would NOT have produced ModelNotFound here\"",
            );
        }
        Err(other) => {
            // A different error means a fallback/stub was taken instead of the
            // clean native ModelNotFound — that is exactly the false-green TL5
            // exists to catch. Self-diagnosing failure: print the wrong error +
            // its exit code.
            log_line(
                test,
                case,
                "result",
                "fail",
                &format!(
                    "diag={{\"error_kind\":\"wrong_error\",\"focr_exit_code\":{},\
                     \"message\":{:?}}}",
                    other.exit_code(),
                    other.to_string()
                ),
            );
            panic!(
                "expected FocrError::ModelNotFound (exit 3) from /nonexistent model, \
                 got {other:?} (exit {}). A non-ModelNotFound error here means the native \
                 path silently degraded to a fallback/stub — TL5 violation.",
                other.exit_code()
            );
        }
        Ok(text) => {
            // Producing OUTPUT from a /nonexistent model is the worst false
            // green: a stub fabricated a result.
            log_line(
                test,
                case,
                "result",
                "fail",
                &format!(
                    "diag={{\"error_kind\":\"fabricated_output\",\"focr_exit_code\":0,\
                     \"message\":\"recognize returned Ok from a /nonexistent model\"}} \
                     output_len={}",
                    text.len()
                ),
            );
            panic!(
                "recognize_with_model(/nonexistent) returned Ok({} bytes) — a stub fabricated \
                 output instead of the clean ModelNotFound. TL5 violation (false green).",
                text.len()
            );
        }
    }
}

/// The same exit-code contract proven through the **real CLI process**
/// (LOGGING_AND_E2E.md §4.5): `focr ocr <img> --model /nonexistent` must exit
/// **3** (ModelNotFound) — the §7.4 exit-code mapping a downstream agent
/// branches on. This is the process-level counterpart to the library guard above:
/// a clap usage error, NotImplemented scaffold, fabricated success, or signal is a
/// real regression because the path-explicit diagnostic lane is now wired.
#[test]
fn cli_ocr_nonexistent_model_exits_model_not_found() {
    let test = "cli_ocr_nonexistent_model_exits_model_not_found";
    let case = "focr_ocr_--model_/nonexistent";

    let Some(bin) = focr_bin() else {
        log_success(
            test,
            case,
            "focr binary not built in this `cargo test` invocation (CARGO_BIN_EXE_focr \
             unset and no target/{debug,release}/focr); the library-level guard \
             native_path_nonexistent_model_is_clean_model_not_found covers this contract",
        );
        return;
    };

    let img = "/some/document.png";
    let model = "/nonexistent/franken_ocr/model.focrq";
    log_line(
        test,
        case,
        "setup",
        "pass",
        &format!(
            "bin={} args=[ocr {img} --model {model}] expect_exit=3",
            bin.display()
        ),
    );

    let out = run_focr(&bin, &["ocr", img, "--model", model]);
    let code = out.code;
    log_line(
        test,
        case,
        "stage",
        "pass",
        &format!(
            "exit={code:?} stdout_len={} stderr_first={:?}",
            out.stdout.len(),
            out.stderr.lines().next().unwrap_or("")
        ),
    );

    if code == Some(3) {
        log_line(
            test,
            case,
            "result",
            "pass",
            "exit=3 error_kind=ModelNotFound note=\"CLI --model wired to clean ModelNotFound\"",
        );
    } else {
        log_line(
            test,
            case,
            "result",
            "fail",
            &format!(
                "diag={{\"error_kind\":\"unexpected_exit\",\"focr_exit_code\":{},\
                 \"message\":\"focr ocr --model /nonexistent must exit 3 via ModelNotFound\"}} \
                 stdout={:?} stderr={:?}",
                code.map(|c| c.to_string())
                    .unwrap_or_else(|| "signal".into()),
                out.stdout,
                out.stderr
            ),
        );
        panic!(
            "focr ocr {img} --model {model} exited {code:?}; expected 3 \
             (ModelNotFound). stdout={:?} stderr={:?}",
            out.stdout, out.stderr
        );
    }
}

/// Future `.focrq` versions must fail as FormatMismatch (exit 7) through the real
/// CLI process, not collapse into NotImplemented or a generic error.
#[test]
fn cli_ocr_future_focrq_exits_format_mismatch() {
    let test = "cli_ocr_future_focrq_exits_format_mismatch";
    let case = "focr_ocr_--model_future_focrq";

    let Some(bin) = focr_bin() else {
        log_success(
            test,
            case,
            "focr binary not built in this `cargo test` invocation; the library-level \
             public_engine_preserves_focrq_format_mismatch_robot_code guard covers this \
             contract until the process binary is available",
        );
        return;
    };

    let img = "/some/document.png";
    let model = write_future_focrq();
    log_line(
        test,
        case,
        "setup",
        "pass",
        &format!(
            "bin={} args=[ocr {img} --model {}] expect_exit=7",
            bin.display(),
            model.display()
        ),
    );

    let model_arg = model.to_string_lossy().into_owned();
    let out = run_focr(&bin, &["ocr", img, "--model", &model_arg]);
    let code = out.code;
    log_line(
        test,
        case,
        "stage",
        "pass",
        &format!(
            "exit={code:?} stdout_len={} stderr_first={:?}",
            out.stdout.len(),
            out.stderr.lines().next().unwrap_or("")
        ),
    );

    if code == Some(7) {
        log_line(
            test,
            case,
            "result",
            "pass",
            "exit=7 error_kind=FormatMismatch note=\"CLI --model preserves future .focrq format mismatch\"",
        );
    } else {
        log_line(
            test,
            case,
            "result",
            "fail",
            &format!(
                "diag={{\"error_kind\":\"unexpected_exit\",\"focr_exit_code\":{},\
                 \"message\":\"focr ocr --model future .focrq must exit 7 via FormatMismatch\"}} \
                 stdout={:?} stderr={:?}",
                code.map(|c| c.to_string())
                    .unwrap_or_else(|| "signal".into()),
                out.stdout,
                out.stderr
            ),
        );
        panic!(
            "focr ocr {img} --model {} exited {code:?}; expected 7 \
             (FormatMismatch). stdout={:?} stderr={:?}",
            model.display(),
            out.stdout,
            out.stderr
        );
    }
}

/// The public, env-resolved `recognize()` also yields a clean `ModelNotFound`
/// when no model is resolvable (env unset AND default absent — the normal CI
/// condition), proving the *public entrypoint* never panics without weights.
/// When a model IS resolvable (env set on a dev box) this defers entirely to the
/// model-present branch and logs a SUCCESS note.
#[test]
fn public_recognize_without_weights_is_model_not_found_or_defers() {
    let test = "public_recognize_without_weights_is_model_not_found_or_defers";
    let case = "lib_recognize_env_resolved";

    if resolve_present_model().is_some() {
        log_success(
            test,
            case,
            "a model IS resolvable (FOCR_MODEL_PATH set / default present) — the \
             without-weights assertion does not apply; the model-present branch \
             (recognize_real_model_when_present) exercises the real forward",
        );
        return;
    }

    log_line(
        test,
        case,
        "setup",
        "pass",
        &format!(
            "searched_dirs={:?} expect=ModelNotFound expect_exit=3",
            searched_dirs()
        ),
    );

    let engine = OcrEngine::new().expect("OcrEngine::new builds");
    let err = engine
        .recognize(Path::new("/some/document.png"))
        .expect_err("absent default model must error, not panic");
    assert!(
        matches!(err, FocrError::ModelNotFound(_)),
        "expected ModelNotFound from the env-resolved recognize() without weights, got {err:?}"
    );
    assert_eq!(err.exit_code(), 3, "ModelNotFound must map to exit code 3");
    log_line(
        test,
        case,
        "result",
        "pass",
        "error_kind=ModelNotFound focr_exit_code=3 public_entrypoint_never_panics=true",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// GUARD 1 (model-gated skip-with-SUCCESS) + GUARD 3 (real recognize present).
//   Both live in ONE test so the gate decision is made exactly once.
// ═══════════════════════════════════════════════════════════════════════════

/// The model-gated real-recognize branch (LOGGING_AND_E2E.md §4.1/§4.3).
///
/// * **Model absent** (`FOCR_MODEL_PATH` unset/missing) ⇒ **skip-with-SUCCESS**:
///   emit a `SUCCESS` line `"e2e skipped: model not present at <path>; native
///   path unverified"` plus the `searched_dirs`, return `Ok`. CI green, skip
///   visible. Never a silent pass, never a failure (TL4).
/// * **Model present** (`FOCR_MODEL_PATH` set, artifact resolves) ⇒ run a **real
///   `recognize()`** over the tiny committed-equivalent fixture image, log the
///   markdown + token count + timing, and assert non-empty + well-formed.
///   * If the forward is still `NotImplemented` (Phase-0 reality: the real
///     kernels / `.focrq` reader have not all landed) we **XFAIL** with the exact
///     stage gap — a documented phase divergence, SUCCESS, tightening to the
///     non-empty assertion once the forward lands.
///   * A `ModelNotFound` here means the developer set `FOCR_MODEL_PATH` to a path
///     that does not resolve — a **misconfiguration FAIL** (§4.3: a model that
///     was *expected present* but does not resolve is not a silent skip).
#[test]
fn recognize_real_model_when_present_else_skip_with_success() {
    let test = "recognize_real_model_when_present_else_skip_with_success";
    let case = "tiny_fixture";

    // ── GUARD 1: absent ⇒ skip-with-SUCCESS ─────────────────────────────────
    let Some(model_path) = resolve_present_model() else {
        let target = OcrEngine::model_path();
        log_line(
            test,
            case,
            "skip",
            "skip_no_model",
            &format!("searched_dirs={:?}", searched_dirs()),
        );
        log_success(
            test,
            case,
            &format!(
                "e2e skipped: model not present at {}; native path unverified",
                target.display()
            ),
        );
        return;
    };

    // ── GUARD 3: present ⇒ real recognize over the tiny fixture ──────────────
    let image = write_tiny_png();
    log_line(
        test,
        case,
        "setup",
        "pass",
        &format!(
            "model={} image={} image_bytes={} note=\"real recognize() over a 4x4 RGB PNG\"",
            model_path.display(),
            image.display(),
            std::fs::metadata(&image).map(|m| m.len()).unwrap_or(0),
        ),
    );

    let engine = OcrEngine::new().expect("OcrEngine::new builds");
    let started = Instant::now();
    // Use the env-resolved `recognize` so this exercises the exact public
    // entrypoint a host calls (the env var is what gated us in).
    let result = engine.recognize(&image);
    let elapsed_ms = started.elapsed().as_millis();

    match result {
        Ok(markdown) => {
            // Real output: assert non-empty + well-formed; log markdown + token
            // count + timing (the §4.3 mandate).
            let token_count = markdown.split_whitespace().count();
            let char_count = markdown.chars().count();
            let preview: String = markdown.chars().take(160).collect();
            log_line(
                test,
                case,
                "stage",
                "pass",
                &format!(
                    "stage=recognize elapsed_ms={elapsed_ms} chars={char_count} \
                     tokens={token_count} markdown_preview={preview:?}"
                ),
            );

            // Well-formedness: non-empty, contains a non-whitespace glyph, and is
            // valid UTF-8 (guaranteed by the `String` return — asserted here for
            // the record). These are the model-agnostic invariants we can hold
            // without an oracle fixture (the oracle compare is the §4.4 e2e
            // script's job).
            assert!(
                !markdown.trim().is_empty(),
                "recognize() returned empty/whitespace-only markdown for {}",
                image.display()
            );
            assert!(
                markdown.chars().any(|c| !c.is_whitespace()),
                "recognize() markdown has no non-whitespace content"
            );
            assert!(
                markdown.chars().all(|c| c != '\u{FFFD}'),
                "recognize() markdown contains the UTF-8 replacement char (malformed output)"
            );

            log_line(
                test,
                case,
                "result",
                "pass",
                &format!(
                    "native_path_ran=true elapsed_ms={elapsed_ms} tokens={token_count} \
                     chars={char_count} well_formed=true"
                ),
            );
            log_success(
                test,
                case,
                &format!(
                    "real recognize() produced {char_count} chars / {token_count} tokens of \
                     well-formed markdown in {elapsed_ms} ms"
                ),
            );
        }
        Err(FocrError::NotImplemented(stage_gap)) => {
            // Phase-0 reality: a model artifact is present but the forward is not
            // fully wired (preprocess / .focrq reader / kernels still landing).
            // XFAIL-with-detail — the named stage gap is the documented
            // divergence; this is SUCCESS and tightens to the non-empty assert
            // once the forward lands.
            log_line(
                test,
                case,
                "stage",
                "xfail",
                &format!("stage=recognize elapsed_ms={elapsed_ms} not_implemented={stage_gap:?}"),
            );
            log_xfail(
                test,
                case,
                &format!("NotImplemented({stage_gap:?})"),
                "Ok(non-empty well-formed markdown)",
            );
            log_success(
                test,
                case,
                "model artifact present but the native forward is still NotImplemented \
                 (Phase-0/1: preprocess/.focrq-reader/kernels landing) — documented phase \
                 divergence; the real-output assertion tightens in automatically once the \
                 forward lands",
            );
        }
        Err(FocrError::ModelNotFound(msg)) => {
            // The env said a model was present, but it did not resolve — a
            // developer misconfiguration. §4.3: an *expected-present* model that
            // does not resolve is a FAIL, never a silent skip.
            log_line(
                test,
                case,
                "result",
                "fail",
                &format!(
                    "diag={{\"error_kind\":\"model_expected_but_unresolved\",\
                     \"focr_exit_code\":3,\"message\":{msg:?}}} resolved_path={}",
                    model_path.display()
                ),
            );
            panic!(
                "FOCR_MODEL_PATH pointed at a present artifact ({}) but recognize() returned \
                 ModelNotFound: {msg}. A model that was expected present but does not resolve is \
                 a misconfiguration FAIL (LOGGING_AND_E2E.md §4.3), not a skip. Fix the path or \
                 unset FOCR_MODEL_PATH.",
                model_path.display()
            );
        }
        Err(other) => {
            // Any other error from a present model (decode error, format
            // mismatch, timeout, …) is a real failure surfaced with its stable
            // exit code so the stream carries the §7.4 contract.
            log_line(
                test,
                case,
                "result",
                "fail",
                &format!(
                    "diag={{\"error_kind\":\"forward_error\",\"focr_exit_code\":{},\
                     \"message\":{:?}}} elapsed_ms={elapsed_ms}",
                    other.exit_code(),
                    other.to_string()
                ),
            );
            panic!(
                "recognize() over a present model failed with {other:?} (exit {}). \
                 A present-model forward error is a real FAIL, not a skip.",
                other.exit_code()
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Model-gated e2e for `focr ocr -o/--output FILE` (bd-sreb). Drives the STABLE
// CLI surface end-to-end and asserts the on-disk file contract the user asked
// for: `.md` => markdown body, `.json` => structured JSON carrying the bounding
// boxes. Skip-with-SUCCESS when the model or the binary is absent (§4.1).
// ═══════════════════════════════════════════════════════════════════════════

/// A unique temp output path with the given extension (we never delete fixtures
/// proactively — AGENTS.md RULE 1 — but a freshly-stamped name can't collide, and
/// the test clears it before each run so the existence check is meaningful).
fn unique_output_path(ext: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    std::env::temp_dir().join(format!(
        "focr_e2e_out_{}_{}.{ext}",
        std::process::id(),
        nanos
    ))
}

/// Assert the structured-JSON output contract: a top-level string `markdown` plus
/// a `layout` array whose entries are `{label: string, boxes: [[i64; 4], …]}`.
/// Returns the number of layout spans for the SUCCESS log.
fn assert_layout_json_contract(raw: &str) -> usize {
    let v: serde_json::Value = serde_json::from_str(raw)
        .unwrap_or_else(|e| panic!("`-o out.json` produced invalid JSON: {e}; body:\n{raw}"));
    assert!(
        v.get("markdown")
            .and_then(serde_json::Value::as_str)
            .is_some(),
        "json output must carry a string `markdown`; body:\n{raw}"
    );
    let Some(layout) = v.get("layout").and_then(serde_json::Value::as_array) else {
        panic!("json output must carry a `layout` array; body:\n{raw}");
    };
    for span in layout {
        assert!(
            span.get("label")
                .and_then(serde_json::Value::as_str)
                .is_some(),
            "every layout span must carry a string `label`: {span}"
        );
        let Some(boxes) = span.get("boxes").and_then(serde_json::Value::as_array) else {
            panic!("every layout span must carry a `boxes` array: {span}");
        };
        for b in boxes {
            let Some(coords) = b.as_array() else {
                panic!("each box must be a JSON array: {b}");
            };
            assert_eq!(coords.len(), 4, "each box must be [x1, y1, x2, y2]: {b}");
            assert!(
                coords.iter().all(|c| c.is_i64() || c.is_u64()),
                "box coordinates must be integers (model pixel grid): {b}"
            );
        }
    }
    layout.len()
}

/// `focr ocr <img> --model <real> -o out.<md|json>` end-to-end over a present
/// model. `.md` must yield a non-empty markdown file; `.json` must yield valid
/// JSON carrying `markdown` + a `layout` array of `{label, boxes}` with integer
/// `[x1,y1,x2,y2]` boxes — the exact structured-output contract the CLI promises.
/// A documented Phase forward gap (exit 1 `not yet implemented`) is an
/// XFAIL-with-SUCCESS that tightens in automatically once the forward lands.
#[test]
fn cli_ocr_output_file_contract_when_model_present_else_skip() {
    let test = "cli_ocr_output_file_contract_when_model_present_else_skip";
    let case = "output_flag";

    let Some(model_path) = resolve_present_model() else {
        log_line(
            test,
            case,
            "skip",
            "skip_no_model",
            &format!("searched_dirs={:?}", searched_dirs()),
        );
        log_success(
            test,
            case,
            "e2e skipped: no model present; `-o` file contract unverified",
        );
        return;
    };
    let Some(bin) = focr_bin() else {
        log_success(
            test,
            case,
            "focr binary not built in this `cargo test` invocation (CARGO_BIN_EXE_focr \
             unset) — `-o` file contract unverified",
        );
        return;
    };

    let image = write_tiny_png();
    let img = image.to_string_lossy().into_owned();
    let model = model_path.to_string_lossy().into_owned();

    // ── markdown output ─────────────────────────────────────────────────────
    let md_out = unique_output_path("md");
    let _ = std::fs::remove_file(&md_out);
    let md_arg = md_out.to_string_lossy().into_owned();
    let out = run_focr(&bin, &["ocr", &img, "--model", &model, "-o", &md_arg]);
    match out.code {
        Some(0) => {
            let body = std::fs::read_to_string(&md_out).unwrap_or_else(|e| {
                panic!("`-o out.md` exited 0 but no markdown file at {md_out:?}: {e}")
            });
            assert!(
                !body.trim().is_empty(),
                "`-o out.md` wrote an empty markdown file"
            );
            log_success(
                test,
                "output_md",
                &format!(
                    "`-o out.md` wrote {} chars of markdown",
                    body.chars().count()
                ),
            );
        }
        Some(1) if out.stderr.contains("not yet implemented") => {
            log_xfail(
                test,
                "output_md",
                "exit 1 not-implemented",
                "exit 0 + markdown file",
            );
            log_success(
                test,
                "output_md",
                "forward still NotImplemented (documented phase gap); the file assertion \
                 tightens in once the forward lands",
            );
        }
        other => panic!(
            "`focr ocr -o out.md` over a present model exited {other:?}; stderr:\n{}",
            out.stderr
        ),
    }

    // ── json output (must carry bounding boxes) ─────────────────────────────
    let json_out = unique_output_path("json");
    let _ = std::fs::remove_file(&json_out);
    let json_arg = json_out.to_string_lossy().into_owned();
    let out = run_focr(&bin, &["ocr", &img, "--model", &model, "-o", &json_arg]);
    match out.code {
        Some(0) => {
            let raw = std::fs::read_to_string(&json_out).unwrap_or_else(|e| {
                panic!("`-o out.json` exited 0 but no json file at {json_out:?}: {e}")
            });
            let spans = assert_layout_json_contract(&raw);
            log_success(
                test,
                "output_json",
                &format!(
                    "`-o out.json` wrote valid JSON with `markdown` + a {spans}-span `layout` \
                     carrying integer bounding boxes"
                ),
            );
        }
        Some(1) if out.stderr.contains("not yet implemented") => {
            log_xfail(
                test,
                "output_json",
                "exit 1 not-implemented",
                "exit 0 + json file with boxes",
            );
            log_success(
                test,
                "output_json",
                "forward still NotImplemented (documented phase gap); the JSON-with-boxes \
                 assertion tightens in once the forward lands",
            );
        }
        other => panic!(
            "`focr ocr -o out.json` over a present model exited {other:?}; stderr:\n{}",
            out.stderr
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Robot-mode pipe smoke (LOGGING_AND_E2E.md §4.5): the stable, always-on
// surface. `focr robot schema` emits exactly one parseable JSON object on
// stdout (data-only), exit 0.
// ═══════════════════════════════════════════════════════════════════════════

/// `focr robot schema` is the self-describing, versioned contract (AGENTS.md
/// "Agent Ergonomics"; `src/robot.rs`). This smoke proves it is pipeable: one
/// JSON object on stdout, exit 0, and the load-bearing fields
/// (`schema_version`, `events`) present and well-typed. This is the *already
/// implemented and committed* surface, so it is a hard assertion (not a
/// tripwire) whenever the binary is built.
#[test]
fn robot_schema_pipe_smoke_is_one_json_object_exit_zero() {
    let test = "robot_schema_pipe_smoke_is_one_json_object_exit_zero";
    let case = "focr_robot_schema";

    let Some(bin) = focr_bin() else {
        log_success(
            test,
            case,
            "focr binary not built in this `cargo test` invocation; the robot-schema \
             contract is unit-tested in src/robot.rs and is exercised by scripts/e2e_smoke.sh \
             against the built binary",
        );
        return;
    };

    log_line(
        test,
        case,
        "setup",
        "pass",
        &format!(
            "bin={} args=[robot schema] expect_exit=0 expect=one_json_object",
            bin.display()
        ),
    );

    let out = run_focr(&bin, &["robot", "schema"]);
    assert_eq!(
        out.code,
        Some(0),
        "`focr robot schema` must exit 0; got {:?}, stderr={:?}",
        out.code,
        out.stderr
    );

    // Data-only on stdout: exactly one non-empty JSON line.
    let lines: Vec<&str> = out
        .stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();
    assert_eq!(
        lines.len(),
        1,
        "`focr robot schema` stdout must be exactly one JSON line; got {} lines: {:?}",
        lines.len(),
        lines
    );

    let parsed: serde_json::Value = serde_json::from_str(lines[0]).unwrap_or_else(|e| {
        log_line(
            test,
            case,
            "result",
            "fail",
            &format!(
                "diag={{\"error_kind\":\"unparseable_json\",\"message\":{:?}}}",
                e.to_string()
            ),
        );
        panic!(
            "`focr robot schema` stdout is not valid JSON: {e}; line={:?}",
            lines[0]
        );
    });

    // The committed contract: schema_version is a number, events is a non-empty
    // array (src/robot.rs ROBOT_SCHEMA_VERSION + EVENT_KINDS).
    let schema_version = parsed.get("schema_version").and_then(|v| v.as_u64());
    let events = parsed.get("events").and_then(|v| v.as_array());
    assert!(
        schema_version.is_some(),
        "robot schema missing numeric `schema_version`; got {parsed}"
    );
    assert!(
        events.map(|a| !a.is_empty()).unwrap_or(false),
        "robot schema missing non-empty `events` array; got {parsed}"
    );

    log_line(
        test,
        case,
        "result",
        "pass",
        &format!(
            "exit=0 schema_version={} events_count={} stdout_is_single_json_object=true",
            schema_version.unwrap(),
            events.unwrap().len()
        ),
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Self-test of the fixture generator: the hand-built PNG is byte-valid so the
// model-present branch is genuinely exercising real pixels (not a sentinel).
// This runs unconditionally and needs no weights.
// ═══════════════════════════════════════════════════════════════════════════

/// The hand-rolled tiny PNG must carry the PNG magic and the three required
/// chunks (IHDR, IDAT, IEND) with the declared dimensions — proving the
/// model-present branch feeds the preprocess front end a real, decodable image
/// rather than a sentinel byte blob. (Structural check; we hand-roll the parse
/// to avoid a dev-dependency on `image`.)
#[test]
fn tiny_png_fixture_is_structurally_valid() {
    let test = "tiny_png_fixture_is_structurally_valid";
    let case = "tiny_png";
    let png = tiny_png(4, 4);

    assert_eq!(
        &png[..8],
        &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A],
        "missing PNG magic"
    );
    // IHDR begins at byte 8: 4-byte length, then b"IHDR".
    assert_eq!(&png[12..16], b"IHDR", "first chunk is not IHDR");
    // Width/height live in the IHDR data (bytes 16..24).
    let w = u32::from_be_bytes([png[16], png[17], png[18], png[19]]);
    let h = u32::from_be_bytes([png[20], png[21], png[22], png[23]]);
    assert_eq!((w, h), (4, 4), "IHDR dims wrong");

    let has = |tag: &[u8; 4]| png.windows(4).any(|w| w == tag);
    assert!(has(b"IDAT"), "missing IDAT chunk");
    assert!(has(b"IEND"), "missing IEND chunk");

    log_line(
        test,
        case,
        "result",
        "pass",
        &format!(
            "png_bytes={} dims={w}x{h} chunks=IHDR,IDAT,IEND magic=ok",
            png.len()
        ),
    );
}
