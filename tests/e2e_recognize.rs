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

#[path = "support/parity_harness.rs"]
mod parity_harness;

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
    // `recognize()` resolves through the FULL model-resolution policy — the
    // `FOCR_MODEL_PATH` override, then the search dirs including the
    // quant-suffixed names a `focr pull` installs (`unlimited-ocr.int8.focrq`,
    // bd-3u6x) — so this guard must use the same surface. A bare
    // `model_path().exists()` check misses a pulled artifact and lets the
    // without-weights branch run against a resolvable model (surfaced
    // 2026-07-06 when a dev cache was repopulated: recognize() found the int8
    // artifact and failed with InputDecode instead of ModelNotFound).
    franken_ocr::native_engine::OcrModel::resolve_model(&OcrEngine::model_path()).ok()
}

/// Env that arms the REAL-FORWARD e2e branches. A resolvable model alone
/// must not arm them: a debug-profile full vision+decode forward is a
/// multi-minute affair per test (measured 2026-07-06: a repopulated dev
/// cache sent three present-model branches into the 600s forward budget on
/// an M4 — exit 5, `cargo test` red for ~30 minutes). Set
/// `FOCR_E2E_REAL_MODEL=1` — typically with a release-built harness — to
/// run them; unset ⇒ skip-with-SUCCESS even when a model is cached.
const REAL_MODEL_ARM_ENV: &str = "FOCR_E2E_REAL_MODEL";

/// The model path for the real-forward branches: resolvable AND armed.
fn armed_present_model() -> Option<PathBuf> {
    match std::env::var_os(REAL_MODEL_ARM_ENV) {
        Some(v) if !v.is_empty() && v != "0" => resolve_present_model(),
        _ => None,
    }
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
/// bd-1gv.25 S3: the MULTI-PAGE cross-page pass end-to-end over the real
/// model (gated exactly like the single-image e2e): two tiny pages become ONE
/// document — Base-640 preprocess ×2, one fused cross-page prefill, one AR
/// decode, `finalize_multi` assembly. Asserts the public facade returns a
/// well-formed `<PAGE>`-assembled markdown and that BOTH non-Unlimited
/// refusal modes stay typed (the /nonexistent-model leg is covered by the CLI
/// golden). Absent/unarmed ⇒ skip-with-SUCCESS (§4.3).
#[test]
fn recognize_multi_page_real_model_when_present_else_skip_with_success() {
    let test = "recognize_multi_page_real_model_when_present_else_skip_with_success";
    let case = "two_tiny_pages_one_document";

    let Some(model_path) = armed_present_model() else {
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
                "multi-page e2e skipped: model not present or {REAL_MODEL_ARM_ENV} unarmed; \
                 cross-page path unverified"
            ),
        );
        return;
    };

    let page_a = write_tiny_png();
    let page_b = write_tiny_png();
    log_line(
        test,
        case,
        "setup",
        "pass",
        &format!(
            "model={} pages=2 note=\"one cross-page pass over two 4x4 RGB PNGs (Base-640)\"",
            model_path.display(),
        ),
    );

    let engine = OcrEngine::new().expect("OcrEngine::new builds");
    let started = Instant::now();
    let result = engine.recognize_multi_page(&[page_a.as_path(), page_b.as_path()]);
    let elapsed_ms = started.elapsed().as_millis();

    match result {
        Ok(markdown) => {
            assert!(
                markdown.starts_with("<PAGE>"),
                "finalize_multi assembly must lead with the page marker; got: {:?}",
                &markdown[..markdown.len().min(120)]
            );
            let seps = markdown.matches("<PAGE>").count();
            log_line(
                test,
                case,
                "result",
                "pass",
                &format!(
                    "elapsed_ms={elapsed_ms} markdown_bytes={} page_markers={seps} \
                     note=\"cross-page pass completed; ngram_window=1024 in force\"",
                    markdown.len(),
                ),
            );
        }
        Err(FocrError::NotImplemented(what)) => {
            log_line(
                test,
                case,
                "result",
                "xfail",
                &format!("elapsed_ms={elapsed_ms} stage_gap={what:?}"),
            );
            log_success(
                test,
                case,
                "multi-page forward hit a documented NotImplemented stage gap (XFAIL)",
            );
        }
        Err(other) => {
            log_line(
                test,
                case,
                "result",
                "fail",
                &format!("elapsed_ms={elapsed_ms} error={other:?}"),
            );
            panic!("multi-page e2e failed on a present model: {other:?}");
        }
    }
}

/// bd-2z0y: the STREAMING multi-page pass end-to-end over the real model —
/// the per-page sink fires from the decode driver at `<PAGE>` boundaries,
/// pages arrive in order starting at 1, the stream's page count equals the
/// terminal assembly's marker count, and the terminal markdown is byte-equal
/// to the non-streaming pass (streaming adds visibility, never changes the
/// result). Absent/unarmed ⇒ skip-with-SUCCESS.
#[test]
fn multi_page_streaming_matches_terminal_assembly_when_armed() {
    let test = "multi_page_streaming_matches_terminal_assembly_when_armed";
    let case = "two_tiny_pages_streamed";

    let Some(model_path) = armed_present_model() else {
        log_success(
            test,
            case,
            &format!("streaming e2e skipped: model not present or {REAL_MODEL_ARM_ENV} unarmed"),
        );
        return;
    };

    let page_a = write_tiny_png();
    let page_b = write_tiny_png();
    let (streamed, events, markers) = stream_against_plain(&model_path, &[page_a, page_b]);
    log_line(
        test,
        case,
        "result",
        "pass",
        &format!(
            "streamed_pages={} markers={} markdown_bytes={} note=\"stream == terminal assembly\"",
            events.len(),
            markers,
            streamed.len(),
        ),
    );
}

/// Run `pages` as one streamed and one plain multi-page pass and check the
/// streaming invariants: the terminal assembly is byte-identical, there is one
/// `page` event per terminal `<PAGE>` marker, pages arrive in order from 1, and
/// the streamed bodies are the document's trimmed `<PAGE>` chunks (so they
/// finalize to the terminal text). Returns the terminal text, the events and
/// the marker count.
fn stream_against_plain(
    model_path: &Path,
    pages: &[PathBuf],
) -> (String, Vec<(usize, String)>, usize) {
    let load = |p: &PathBuf| image::open(p).expect("page fixture decodes");
    let engine = OcrEngine::new().expect("OcrEngine::new builds");
    let events: std::sync::Arc<std::sync::Mutex<Vec<(usize, String)>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink_events = events.clone();
    let sink = Box::new(move |page: usize, body: &str| {
        sink_events
            .lock()
            .expect("sink mutex")
            .push((page, body.to_string()));
    });
    let streamed = engine
        .recognize_multi_page_dynamic_streaming_with_model(
            model_path,
            pages.iter().map(load).collect(),
            sink,
        )
        .expect("streaming multi-page pass");
    let plain = engine
        .recognize_multi_page_dynamic_with_model(model_path, pages.iter().map(load).collect())
        .expect("plain multi-page pass");
    assert_eq!(
        streamed, plain,
        "streaming must not change the terminal assembly"
    );
    let events = events.lock().expect("events mutex").clone();
    let markers = streamed.matches("<PAGE>").count();
    assert_eq!(
        events.len(),
        markers,
        "one streamed page per terminal <PAGE> marker; events: {events:?}"
    );
    for (i, (page, _)) in events.iter().enumerate() {
        assert_eq!(*page, i + 1, "pages must stream in order from 1");
    }
    let bodies: Vec<&str> = events.iter().map(|(_, body)| body.as_str()).collect();
    let rebuilt = franken_ocr::native_engine::postprocess::finalize_multi(
        &format!("<PAGE>{}", bodies.join("<PAGE>")),
        pages.len(),
    )
    .expect("finalize the streamed bodies");
    assert_eq!(
        rebuilt, streamed,
        "the streamed bodies must be the terminal text's <PAGE> chunks"
    );
    (streamed, events, markers)
}

/// U2-X1: Chinese text whose rarer characters the v1 tokenizer splits across
/// byte tokens. Before the fix the first split character stopped every later
/// `page` event; now the stream keeps up to the terminal marker count.
#[test]
fn multi_page_streaming_survives_split_cjk_characters_when_armed() {
    let test = "multi_page_streaming_survives_split_cjk_characters_when_armed";
    let case = "cjk_stream_two_pages";
    let Some(model_path) = armed_present_model() else {
        log_success(
            test,
            case,
            &format!(
                "CJK streaming e2e skipped: model not present or {REAL_MODEL_ARM_ENV} unarmed"
            ),
        );
        return;
    };
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cjk_stream");
    let pages = vec![fixtures.join("page_1.png"), fixtures.join("page_2.png")];
    let (streamed, events, markers) = stream_against_plain(&model_path, &pages);
    // The run must really stream a split character. Before the fix one in any
    // body lost that page's event (or the final flush), so the event count
    // checked above would fall short of the markers.
    let tok_path = model_path.with_file_name("tokenizer.json");
    let tok = franken_ocr::tokenizer::Tokenizer::from_file(&tok_path)
        .unwrap_or_else(|e| panic!("tokenizer beside the model ({}): {e}", tok_path.display()));
    let is_split = |c: char| tok.encode(&c.to_string()).is_ok_and(|ids| ids.len() > 1);
    let first_split_event = events
        .iter()
        .position(|(_, body)| body.chars().any(|c| !c.is_ascii() && is_split(c)))
        .unwrap_or_else(|| panic!("no split CJK character was streamed: {events:?}"));
    assert!(markers >= 2, "both pages must stream: {events:?}");
    let split_chars: String = streamed
        .chars()
        .filter(|&c| !c.is_ascii() && is_split(c))
        .collect();
    log_line(
        test,
        case,
        "result",
        "pass",
        &format!(
            "streamed_pages={} markers={markers} first_split_event={first_split_event} \
             split_chars={split_chars:?}",
            events.len()
        ),
    );
}

#[test]
fn recognize_real_model_when_present_else_skip_with_success() {
    let test = "recognize_real_model_when_present_else_skip_with_success";
    let case = "tiny_fixture";

    // ── GUARD 1: absent ⇒ skip-with-SUCCESS ─────────────────────────────────
    let Some(model_path) = armed_present_model() else {
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
                "e2e skipped: model not present or {REAL_MODEL_ARM_ENV} unarmed (target {}); native path unverified",
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
            // bd-3kge: the SHARED determinism gate over the public entrypoint
            // — a second recognize() of the same image must be BYTE-IDENTICAL
            // under greedy (our-engine determinism; a divergence is a real
            // bug, never noise).
            let second = engine
                .recognize(&image)
                .expect("second recognize for the determinism gate");
            parity_harness::assert_outputs_deterministic(
                test,
                case,
                1,
                markdown.as_bytes(),
                second.as_bytes(),
            );
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

    let Some(model_path) = armed_present_model() else {
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

/// Assert `path` is a non-empty PNG or JPEG by magic bytes (the two formats the
/// figure extractor writes).
fn assert_is_png_or_jpeg(path: &Path) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read figure {path:?}: {e}"));
    assert!(!bytes.is_empty(), "figure {path:?} is empty");
    let is_png = bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    let is_jpeg = bytes.starts_with(&[0xFF, 0xD8, 0xFF]);
    assert!(
        is_png || is_jpeg,
        "figure {path:?} is neither PNG nor JPEG (magic {:02X?})",
        &bytes[..bytes.len().min(8)]
    );
}

/// Model-gated e2e for `focr ocr --extract-figures` (bd-23s8). With a real model,
/// `-o out.md --extract-figures` must exit clean and write a non-empty markdown
/// file; if the model grounds any figure regions, every file it wrote into
/// `out_figures/` is a valid PNG/JPEG AND is referenced by the markdown. (A 4×4
/// fixture rarely yields figures, so figure PRESENCE is not asserted — the path
/// running clean and any written figure being valid + referenced is.)
/// Skip-with-SUCCESS when the model or binary is absent.
#[test]
fn cli_ocr_extract_figures_when_model_present_else_skip() {
    let test = "cli_ocr_extract_figures_when_model_present_else_skip";
    let case = "extract_figures";

    let Some(model_path) = armed_present_model() else {
        log_success(
            test,
            case,
            "e2e skipped: no model present; --extract-figures path unverified",
        );
        return;
    };
    let Some(bin) = focr_bin() else {
        log_success(
            test,
            case,
            "focr binary not built (CARGO_BIN_EXE_focr unset); --extract-figures unverified",
        );
        return;
    };

    let image = write_tiny_png();
    let img = image.to_string_lossy().into_owned();
    let model = model_path.to_string_lossy().into_owned();
    let md_out = unique_output_path("md");
    let stem = md_out
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("doc")
        .to_string();
    let figdir = md_out.with_file_name(format!("{stem}_figures"));
    let _ = std::fs::remove_file(&md_out);
    let _ = std::fs::remove_dir_all(&figdir);
    let md_arg = md_out.to_string_lossy().into_owned();

    let out = run_focr(
        &bin,
        &[
            "ocr",
            &img,
            "--model",
            &model,
            "-o",
            &md_arg,
            "--extract-figures",
        ],
    );
    match out.code {
        Some(0) => {
            let body = std::fs::read_to_string(&md_out).unwrap_or_else(|e| {
                panic!("--extract-figures exited 0 but no md at {md_out:?}: {e}")
            });
            assert!(
                !body.trim().is_empty(),
                "--extract-figures wrote an empty markdown file"
            );
            let mut n = 0usize;
            if figdir.is_dir() {
                for entry in std::fs::read_dir(&figdir).expect("read figures dir") {
                    let p = entry.expect("figures dir entry").path();
                    if p.is_file() {
                        assert_is_png_or_jpeg(&p);
                        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or_default();
                        assert!(
                            body.contains(name),
                            "figure file {name} is not referenced by the markdown:\n{body}"
                        );
                        n += 1;
                    }
                }
            }
            log_success(
                test,
                case,
                &format!("--extract-figures ran clean; {n} valid figure(s) written + referenced"),
            );
        }
        Some(1) if out.stderr.contains("not yet implemented") => {
            log_xfail(
                test,
                case,
                "exit 1 not-implemented",
                "exit 0 + figures path",
            );
            log_success(
                test,
                case,
                "forward still NotImplemented (documented phase gap); tightens in once it lands",
            );
        }
        other => panic!(
            "`focr ocr --extract-figures` over a present model exited {other:?}; stderr:\n{}",
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

// ═══════════════════════════════════════════════════════════════════════════
// U2-X2: v1 honest reporting (`finish_reason`, `v1_truncated`), model-gated
// and armed like the other real-forward branches. Inputs are corpus pages
// (`FOCR_CORPUS_DIR`, default `~/.cache/franken_ocr/corpus/pages`).
// ═══════════════════════════════════════════════════════════════════════════

/// The corpus pages, or `None` (skip-with-SUCCESS) when absent.
fn corpus_pages(names: &[&str]) -> Option<Vec<PathBuf>> {
    let dir = std::env::var_os("FOCR_CORPUS_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join(".cache/franken_ocr/corpus/pages"))
        })?;
    let pages: Vec<PathBuf> = names.iter().map(|n| dir.join(n)).collect();
    pages.iter().all(|p| p.is_file()).then_some(pages)
}

/// Run `focr` with extra environment.
fn run_focr_env(bin: &Path, args: &[&str], envs: &[(&str, String)]) -> CliOut {
    let mut command = Command::new(bin);
    command.args(args);
    for (key, value) in envs {
        command.env(key, value);
    }
    let out = command
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {}: {e}", bin.display()));
    CliOut {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// The armed model, binary and three corpus pages for the X2 cases.
fn x2_inputs(test: &str) -> Option<(PathBuf, PathBuf, Vec<PathBuf>)> {
    let Some(model) = armed_present_model() else {
        log_line(
            test,
            "gate",
            "skip",
            "skip_no_model",
            "reason=\"FOCR_E2E_REAL_MODEL unset or no model\"",
        );
        log_success(
            test,
            "gate",
            "model-gated: arm with FOCR_E2E_REAL_MODEL=1 and a resolvable model",
        );
        return None;
    };
    let Some(bin) = focr_bin() else {
        log_success(test, "gate", "no built focr binary");
        return None;
    };
    let Some(pages) = corpus_pages(&["page_0009.png", "page_0014.png", "page_0019.png"]) else {
        log_line(
            test,
            "gate",
            "skip",
            "skip_no_model",
            "reason=\"corpus pages absent\"",
        );
        log_success(
            test,
            "gate",
            "set FOCR_CORPUS_DIR to the 20-page corpus pages dir",
        );
        return None;
    };
    Some((model, bin, pages))
}

/// One `ocr-batch --multi-page --json` run over the three pages.
fn batch_multi_page(
    bin: &Path,
    model: &Path,
    pages: &[PathBuf],
    envs: &[(&str, String)],
) -> (CliOut, Option<serde_json::Value>) {
    let mut args = vec!["ocr-batch", "--multi-page", "--json"];
    let page_strs: Vec<String> = pages.iter().map(|p| p.display().to_string()).collect();
    args.extend(page_strs.iter().map(String::as_str));
    let mut envs = envs.to_vec();
    envs.push(("FOCR_MODEL_PATH", model.display().to_string()));
    envs.push(("FOCR_NO_RESIDENT", "1".to_string()));
    let out = run_focr_env(bin, &args, &envs);
    let value = out
        .stdout
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .find(|v| v["command"] == "batch.multi_page");
    (out, value)
}

fn warning_kinds(value: &serde_json::Value) -> Vec<String> {
    value["warnings"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|w| w["kind"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

/// The position clamp: a budget just past the 3-page prompt stops the decode
/// early. Exit 0, `finish_reason: length`, a `v1_truncated` warning with
/// cause `positions`, and its stderr line.
#[test]
fn v1_multi_page_clamp_reports_truncation_when_armed() {
    const TEST: &str = "v1_multi_page_clamp_reports_truncation_when_armed";
    let Some((model, bin, pages)) = x2_inputs(TEST) else {
        return;
    };
    // Probe the prompt length: a 1-position budget fails with it in the message.
    let (probe, _) = batch_multi_page(
        &bin,
        &model,
        &pages,
        &[("FOCR_TEST_MAX_POSITIONS", "1".into())],
    );
    let text = format!("{}{}", probe.stdout, probe.stderr);
    let prompt_len: usize = text
        .split("prefix is ")
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("no prompt length in the over-budget error: {text}"));
    let budget = prompt_len + 48;
    let (out, value) = batch_multi_page(
        &bin,
        &model,
        &pages,
        &[("FOCR_TEST_MAX_POSITIONS", budget.to_string())],
    );
    let value = value.unwrap_or_else(|| panic!("no batch.multi_page object: {}", out.stdout));
    assert_eq!(
        out.code,
        Some(0),
        "v1 truncation keeps exit 0: {}",
        out.stderr
    );
    assert_eq!(value["finish_reason"], "length");
    let warning = &value["warnings"][0];
    assert_eq!(warning["kind"], "v1_truncated", "{value}");
    assert_eq!(warning["cause"], "positions");
    assert_eq!(warning["expected"], 3);
    let got = warning["got"].as_u64().expect("got");
    assert!(got < 3, "{value}");
    assert!(
        out.stderr
            .contains(&format!("[focr] warning: stopped after {got} of 3 pages")),
        "{}",
        out.stderr
    );
    log_line(
        TEST,
        "clamp",
        "result",
        "pass",
        &format!(
            "expected_pages=3 bodies={got} finish_reason=length warning_kinds={:?} exit_code=0 \
             budget={budget} prompt_len={prompt_len}",
            warning_kinds(&value)
        ),
    );
}

/// A user token cap below the clamp: the same kind with cause `user_cap`.
#[test]
fn v1_multi_page_user_cap_reports_truncation_when_armed() {
    const TEST: &str = "v1_multi_page_user_cap_reports_truncation_when_armed";
    let Some((model, bin, pages)) = x2_inputs(TEST) else {
        return;
    };
    let (out, value) = batch_multi_page(
        &bin,
        &model,
        &pages,
        &[("FOCR_MAX_NEW_TOKENS", "24".into())],
    );
    let value = value.unwrap_or_else(|| panic!("no batch.multi_page object: {}", out.stdout));
    assert_eq!(out.code, Some(0), "{}", out.stderr);
    assert_eq!(value["finish_reason"], "length");
    let warning = &value["warnings"][0];
    assert_eq!(
        (warning["kind"].as_str(), warning["cause"].as_str()),
        (Some("v1_truncated"), Some("user_cap"))
    );
    assert_eq!(warning["recovery"], "raise `--max-length`");
    log_line(
        TEST,
        "user_cap",
        "result",
        "pass",
        &format!(
            "expected_pages=3 bodies={} finish_reason=length warning_kinds={:?} exit_code=0",
            warning["got"],
            warning_kinds(&value)
        ),
    );
}

/// Single pages report `finish_reason` and never warn: `stop` at EOS,
/// `length` under `--max-length`.
#[test]
fn v1_single_page_reports_finish_reason_when_armed() {
    const TEST: &str = "v1_single_page_reports_finish_reason_when_armed";
    let Some((model, bin, pages)) = x2_inputs(TEST) else {
        return;
    };
    let page = pages[0].display().to_string();
    for (case, extra, want) in [("healthy", None, "stop"), ("capped", Some("8"), "length")] {
        let mut args = vec!["ocr", "--robot", page.as_str()];
        if let Some(cap) = extra {
            args.extend(["--max-length", cap]);
        }
        let store = unique_output_path("db");
        let envs = [
            ("FOCR_MODEL_PATH", model.display().to_string()),
            ("FOCR_NO_RESIDENT", "1".into()),
            ("FOCR_RUN_STORE", store.display().to_string()),
        ];
        let out = run_focr_env(&bin, &args, &envs);
        let complete = out
            .stdout
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .find(|v| v["event"] == "run_complete")
            .unwrap_or_else(|| panic!("{case}: no run_complete: {}{}", out.stdout, out.stderr));
        assert_eq!(out.code, Some(0), "{case}");
        assert_eq!(complete["finish_reason"], want, "{case}");
        assert!(complete.get("warnings").is_none(), "{case}: {complete}");
        // U2-E1: the model that ran, and the run store's tag for it. A
        // `.focrq` records its source digest; a raw checkpoint has none.
        assert_eq!(complete["model_id"], "unlimited-ocr", "{case}: {complete}");
        let tag = match complete["source_sha256"].as_str() {
            Some(sha) => {
                assert!(
                    sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit()),
                    "{case}: {sha}"
                );
                format!("unlimited-ocr@{sha}")
            }
            None => {
                assert!(complete["source_sha256"].is_null(), "{case}: {complete}");
                "unlimited-ocr".to_owned()
            }
        };
        let runs = run_focr_env(&bin, &["runs", "--json", "--limit", "1"], &envs);
        let runs: serde_json::Value =
            serde_json::from_str(runs.stdout.trim()).expect("focr runs --json");
        assert_eq!(runs["runs"][0]["model_version_tag"], tag.as_str(), "{case}");
        let _ = std::fs::remove_file(&store);
        log_line(
            TEST,
            case,
            "result",
            "pass",
            &format!("expected_pages=1 finish_reason={want} warning_kinds=[] exit_code=0"),
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// U2-A7: Unlimited-OCR-2 is fail-closed through the CLI (no weights needed).
// ═══════════════════════════════════════════════════════════════════════════

/// A census-exact Unlimited-OCR-2 raw checkpoint, built from the committed
/// census files: one safetensors file whose payload is sparse (zeros) except
/// the I64 `position_ids` (`arange`). Its header identifies it as v2, and no
/// weight byte is ever read, because the run is refused first.
fn write_sparse_v2_stand_in(dir: &Path) -> PathBuf {
    use std::io::{Seek as _, SeekFrom, Write as _};
    let census = |name: &str| -> serde_json::Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/native_engine")
            .join(name);
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read census manifest"))
            .expect("census manifest parses")
    };
    let v1 = census("unlimited_ocr_manifest.json");
    let delta = census("unlimited_ocr_2_manifest.json");
    let mut tensors = std::collections::BTreeMap::new();
    for table in [&v1["tensors"], &delta["added_tensors"]] {
        for (name, tensor) in table.as_object().expect("tensor table") {
            let dtype = tensor["source_dtype"]
                .as_str()
                .expect("source_dtype")
                .to_owned();
            let shape: Vec<u64> = serde_json::from_value(tensor["shape"].clone()).expect("shape");
            tensors.insert(name.clone(), (dtype, shape));
        }
    }
    let mut entries = Vec::new();
    let mut offset = 0u64;
    for (name, (dtype, shape)) in &tensors {
        let elem = match dtype.as_str() {
            "F32" => 4,
            "F16" | "BF16" => 2,
            other => panic!("a source census holds no {other} tensor"),
        };
        let len = shape.iter().product::<u64>() * elem;
        entries.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":{shape:?},"data_offsets":[{offset},{}]}}"#,
            offset + len
        ));
        offset += len;
    }
    let mut aranges = Vec::new();
    for (name, rule) in delta["int64_tensors"].as_object().expect("int64 table") {
        assert_eq!(rule["content"], "arange", "{name}");
        let shape: Vec<u64> = serde_json::from_value(rule["shape"].clone()).expect("shape");
        let count = shape.iter().product::<u64>();
        entries.push(format!(
            r#""{name}":{{"dtype":"I64","shape":{shape:?},"data_offsets":[{offset},{}]}}"#,
            offset + count * 8
        ));
        aranges.push((offset, count));
        offset += count * 8;
    }
    let header = format!("{{{}}}", entries.join(","));
    let path = dir.join("unlimited-ocr-2-stand-in.safetensors");
    let mut file = std::fs::File::create(&path).expect("create stand-in");
    file.write_all(&(header.len() as u64).to_le_bytes())
        .expect("write prefix");
    file.write_all(header.as_bytes()).expect("write header");
    file.set_len(8 + header.len() as u64 + offset)
        .expect("sparse payload");
    for (start, count) in aranges {
        let bytes: Vec<u8> = (0..count as i64).flat_map(i64::to_le_bytes).collect();
        file.seek(SeekFrom::Start(8 + header.len() as u64 + start))
            .expect("seek to the I64 tensor");
        file.write_all(&bytes).expect("write arange");
    }
    path
}

/// Run `focr` with exactly `envs` among the `FOCR_*` variables: the caller's
/// own `FOCR_*` settings would otherwise steer the lever policy.
fn run_focr_only(bin: &Path, args: &[&str], envs: &[(&str, String)]) -> CliOut {
    let mut command = Command::new(bin);
    command.args(args);
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("FOCR_") {
            command.env_remove(key);
        }
    }
    for (key, value) in envs {
        command.env(key, value);
    }
    let out = command
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {}: {e}", bin.display()));
    CliOut {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// U2-A7, U2-B6: through the CLI, an Unlimited-OCR-2 checkpoint's levers are
/// judged before anything else is read. An armed unsupported lever is named
/// and refused with exit 2; an ignored lever is a one-line stderr warning.
/// An admitted run reaches the tokenizer sidecar check next, which refuses
/// the stand-in's foreign sidecar with exit 7 — still before a weight byte
/// is read. The resident daemon is never spawned.
#[test]
fn cli_judges_unlimited_ocr_2_levers_then_its_sidecar_before_weights() {
    const TEST: &str = "cli_judges_unlimited_ocr_2_levers_then_its_sidecar_before_weights";
    const FOREIGN_SIDECAR: &str = "tokenizer.json has sha256";
    let Some(bin) = focr_bin() else {
        log_success(
            TEST,
            "no_bin",
            "focr binary not built; CLI lever matrix deferred",
        );
        return;
    };
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("focr_e2e_uocr2_{}_{nanos}", std::process::id()));
    let resident_dir = dir.join("resident");
    std::fs::create_dir_all(&resident_dir).expect("create temp dirs");
    let model = write_sparse_v2_stand_in(&dir);
    // A foreign sidecar: an admitted run stops at its check (exit 7), so the
    // sparse payload is never decoded.
    std::fs::write(dir.join("tokenizer.json"), "{}").expect("foreign sidecar");
    let model = model.display().to_string();
    let image = write_tiny_png();
    let image = image.display().to_string();

    // (case, extra args, env, the lever refused, the levers warned about)
    type Case<'a> = (
        &'a str,
        &'a [&'a str],
        &'a [(&'a str, &'a str)],
        Option<&'a str>,
        &'a [&'a str],
    );
    let cases: [Case; 8] = [
        ("bare", &["ocr"], &[], None, &[]),
        (
            "prefill_chunk",
            &["ocr"],
            &[("FOCR_PREFILL_CHUNK", "64")],
            Some("FOCR_PREFILL_CHUNK"),
            &[],
        ),
        (
            "decode_int8",
            &["ocr"],
            &[("FOCR_DECODE_INT8", "1")],
            Some("FOCR_DECODE_INT8"),
            &[],
        ),
        (
            "lmhead_shard_zero",
            &["ocr"],
            &[("FOCR_LMHEAD_SHARD", "0")],
            Some("FOCR_LMHEAD_SHARD"),
            &[],
        ),
        (
            "batch_spine_off",
            &["ocr"],
            &[("FOCR_BATCH_SPINE", "0")],
            None,
            &["FOCR_BATCH_SPINE"],
        ),
        (
            "ignored_levers",
            &["ocr"],
            &[
                ("FOCR_QKV_FUSED", "0"),
                ("FOCR_RUNAWAY_GUARD", "maybe"),
                ("FOCR_THREADS", "2"),
            ],
            None,
            &["FOCR_QKV_FUSED", "FOCR_RUNAWAY_GUARD"],
        ),
        (
            "robot_batch_spine",
            &["ocr", "--robot"],
            &[("FOCR_BATCH_SPINE", "1")],
            Some("FOCR_BATCH_SPINE"),
            &[],
        ),
        (
            "ocr_batch_full_int8",
            &["ocr-batch", "--experimental-full-int8"],
            &[("FOCR_INT8_ATTN", "1"), ("FOCR_INT8_LMHEAD", "1")],
            Some("--experimental-full-int8"),
            &["FOCR_INT8_ATTN", "FOCR_INT8_LMHEAD"],
        ),
    ];
    for (case, command, levers, refused, warned) in cases {
        let mut args: Vec<&str> = command.to_vec();
        args.extend([image.as_str(), "--model", model.as_str()]);
        let mut envs: Vec<(&str, String)> = vec![
            ("FOCR_RESIDENT_DIR", resident_dir.display().to_string()),
            // Were a daemon spawned anyway, it would not linger.
            ("FOCR_RESIDENT_IDLE_SECS", "2".into()),
            // A safety net only: were the gate bypassed, the sparse payload
            // would be mapped, not copied into memory.
            ("FOCR_MMAP", "1".into()),
        ];
        envs.extend(
            levers
                .iter()
                .map(|(key, value)| (*key, (*value).to_owned())),
        );
        let started = Instant::now();
        let out = run_focr_only(&bin, &args, &envs);
        let elapsed_us = started.elapsed().as_micros();
        let robot = command.contains(&"--robot");
        let message = if robot {
            let events: Vec<serde_json::Value> = out
                .stdout
                .lines()
                .map(|l| {
                    serde_json::from_str(l).unwrap_or_else(|e| panic!("{case}: stdout {l:?}: {e}"))
                })
                .collect();
            let error = events
                .iter()
                .find(|v| v["event"] == "run_error")
                .unwrap_or_else(|| panic!("{case}: no run_error: {}", out.stdout));
            assert_eq!(error["code"], out.code.unwrap_or(-1), "{case}: {error}");
            error["message"].as_str().unwrap_or_default().to_owned()
        } else {
            out.stderr.clone()
        };
        let warning_lines: Vec<&str> = out
            .stderr
            .lines()
            .filter(|l| l.starts_with("[focr] warning: FOCR_"))
            .collect();
        let mut pass = warning_lines.len() == warned.len();
        match refused {
            Some(lever) => {
                pass &= out.code == Some(2)
                    && message.contains(&format!("{lever} is not supported for unlimited-ocr-2"))
                    && !message.contains(FOREIGN_SIDECAR);
            }
            None => pass &= out.code == Some(7) && message.contains(FOREIGN_SIDECAR),
        }
        for lever in warned {
            pass &= warning_lines
                .iter()
                .any(|l| l.contains(&format!("{lever} has no effect on unlimited-ocr-2")));
        }
        println!(
            "{}",
            serde_json::json!({
                "schema_version": 1, "ts": 0.0, "test": TEST, "case": case, "run_seq": 0,
                "event": "assert", "result": if pass { "pass" } else { "fail" },
                "assertion": "levers named (exit 2) or warned, then the sidecar (exit 7), before weights",
                "pass": pass, "arch": "unlimited-ocr-2", "args": args, "levers": levers,
                "expected_refused": refused, "expected_warned": warned,
                "exit_code": out.code, "message": message.trim(),
                "warning_lines": warning_lines, "elapsed_us": elapsed_us,
                "weights_touched": false,
            })
        );
        assert!(
            pass,
            "{case}: exit {:?}\nstdout:\n{}\nstderr:\n{}",
            out.code, out.stdout, out.stderr
        );
    }
    // D21: an unknown `FOCR_SRSWA_ENTMAX` is refused at admission (exit 2),
    // before the sidecar; a known one is admitted, labelled experimental, and
    // stops at the sidecar (exit 7).
    for (value, code) in [("bogus", 2), ("all", 7)] {
        let args = ["ocr", image.as_str(), "--model", model.as_str()];
        let envs: Vec<(&str, String)> = vec![
            ("FOCR_RESIDENT_DIR", resident_dir.display().to_string()),
            ("FOCR_RESIDENT_IDLE_SECS", "2".into()),
            ("FOCR_MMAP", "1".into()),
            ("FOCR_SRSWA_ENTMAX", value.to_owned()),
        ];
        let out = run_focr_only(&bin, &args, &envs);
        let refused = out.stderr.contains(&format!(
            "FOCR_SRSWA_ENTMAX={value:?} is not an entmax domain"
        ));
        let labelled = out
            .stderr
            .contains("[focr] note: unlimited-ocr-2 is experimental");
        let sidecar = out.stderr.contains(FOREIGN_SIDECAR);
        let pass = out.code == Some(code)
            && if code == 2 {
                refused && !labelled && !sidecar
            } else {
                !refused && labelled && sidecar
            };
        println!(
            "{}",
            serde_json::json!({
                "schema_version": 1, "ts": 0.0, "test": TEST, "case": format!("entmax_{value}"),
                "run_seq": 0, "event": "assert", "result": if pass { "pass" } else { "fail" },
                "assertion": "an unknown entmax domain exits 2 at admission; a known one is admitted",
                "pass": pass, "arch": "unlimited-ocr-2", "exit_code": out.code,
                "expected_exit_code": code, "message": out.stderr.trim(),
            })
        );
        assert!(
            pass,
            "entmax {value}: exit {:?}\nstderr:\n{}",
            out.code, out.stderr
        );
    }
    let spawned: Vec<_> = std::fs::read_dir(&resident_dir)
        .expect("read resident dir")
        .collect();
    assert!(
        spawned.is_empty(),
        "a resident daemon was spawned for v2: {spawned:?}"
    );

    // The v1 wasm int4 recipe is refused for v2 by arch, before any shard is
    // read (exit 1, as for every model without a certified int4 recipe).
    let output = dir.join("v2.int4.focrq").display().to_string();
    let args = [
        "convert",
        model.as_str(),
        "--output",
        output.as_str(),
        "--quant",
        "int4",
        "--model-id",
        "unlimited-ocr-2",
    ];
    let started = Instant::now();
    let out = run_focr_only(&bin, &args, &[]);
    let pass = out.code == Some(1)
        && out
            .stderr
            .contains("--model-id \"unlimited-ocr-2\" has no certified int4 recipe")
        && !Path::new(&output).exists();
    println!(
        "{}",
        serde_json::json!({
            "schema_version": 1, "ts": 0.0, "test": TEST, "case": "convert_int4", "run_seq": 0,
            "event": "assert", "result": if pass { "pass" } else { "fail" },
            "assertion": "v2 int4 convert refused by arch before any shard is read",
            "pass": pass, "arch": "unlimited-ocr-2", "args": args, "exit_code": out.code,
            "message": out.stderr.trim(), "elapsed_us": started.elapsed().as_micros(),
            "weights_touched": false,
        })
    );
    assert!(pass, "convert_int4: exit {:?}\n{}", out.code, out.stderr);
}

/// U2-C3's loader gate through the CLI (model-gated: the converted v2
/// `.focrq` supplies a census-exact header). Each case is a sparse copy of the
/// real header, untouched or with one field tampered, behind a payload that is
/// never read. The untouched copy passes the gate and reaches the lever
/// admission, where an armed refused lever stops it with exit 2; a wrong
/// recipe id (v1's, or the wasm int4 one), a quantized router
/// tensor, or a non-convention `source_sha256` is refused with exit 7 naming
/// the field or tensor. Both `--model <file>` and short-name resolution
/// (`--model unlimited-ocr-2` under `FOCR_MODEL_DIR`) take the same gate.
#[test]
fn cli_holds_v2_focrq_to_its_recipe_by_path_and_by_short_name() {
    const TEST: &str = "cli_holds_v2_focrq_to_its_recipe_by_path_and_by_short_name";
    let Some(real) = std::env::var_os("FOCR_TEST_UNLIMITED_2_RECIPE_MODEL").map(PathBuf::from)
    else {
        log_success(
            TEST,
            "no_model",
            "FOCR_TEST_UNLIMITED_2_RECIPE_MODEL unset; v2 .focrq gate deferred",
        );
        return;
    };
    let Some(bin) = focr_bin() else {
        log_success(
            TEST,
            "no_bin",
            "focr binary not built; v2 .focrq gate deferred",
        );
        return;
    };
    use std::io::{Read as _, Write as _};
    let mut file = std::fs::File::open(&real).expect("open the v2 artifact");
    let file_len = file.metadata().expect("stat the v2 artifact").len();
    let mut preamble = [0u8; 51];
    file.read_exact(&mut preamble).expect("read the preamble");
    let header_len = u64::from_le_bytes(preamble[43..51].try_into().expect("8 bytes"));
    let mut header = vec![0u8; usize::try_from(header_len).expect("header fits")];
    file.read_exact(&mut header).expect("read the header");
    let header: serde_json::Value = serde_json::from_slice(&header).expect("header JSON");
    let payload_len = file_len - 51 - header_len;

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let root =
        std::env::temp_dir().join(format!("focr_e2e_v2_gate_{}_{nanos}", std::process::id()));
    let image = write_tiny_png();
    let image = image.display().to_string();

    let router = "model.layers.4.self_attn.dash_router.q_proj.weight";
    let flipped_sha = {
        let sha = header["source_sha256"].as_str().expect("source_sha256");
        format!(
            "{}{}",
            if sha.starts_with('0') { '1' } else { '0' },
            &sha[1..]
        )
    };
    // (case, tamper, expected exit, every substring the message must hold)
    type Tamper = fn(&mut serde_json::Value, &str);
    let cases: [(&str, Tamper, i32, Vec<String>); 5] = [
        (
            "untouched",
            |_, _| {},
            2,
            vec!["FOCR_PREFILL_CHUNK is not supported for unlimited-ocr-2".into()],
        ),
        (
            "v1_recipe_id",
            |h, _| {
                h["packing_manifest"]["quant_recipe"] =
                    "unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1".into();
            },
            7,
            vec!["quant_recipe must be exactly".into()],
        ),
        (
            "wasm_int4_recipe_id",
            |h, _| {
                h["packing_manifest"]["quant_recipe"] =
                    "unlimited-ocr-wasm-experts-int4-attn-int8-lmhead-int8-v1".into();
            },
            7,
            vec![
                "quant_recipe must be exactly".into(),
                "no int4 or wasm recipe".into(),
            ],
        ),
        (
            "quantized_router",
            |h, _| {
                // Re-declare the router BF16 matrix as int8 inside its own
                // byte range: half for the weights, then its scales.
                let record =
                    &mut h["tensors"]["model.layers.4.self_attn.dash_router.q_proj.weight"];
                let offset = record["byte_offset"].as_u64().expect("offset");
                let rows = record["shape"][0].as_u64().expect("rows");
                let half = record["byte_len"].as_u64().expect("byte_len") / 2;
                record["dtype"] = "QInt8PerChan".into();
                record["byte_len"] = half.into();
                record["scales_offset"] = (offset + half).into();
                record["scales_len"] = (rows * 4).into();
            },
            7,
            vec![router.into(), "expected BF16".into()],
        ),
        (
            "flipped_source_sha256",
            |h, flipped| h["source_sha256"] = flipped.into(),
            7,
            vec!["source_sha256".into()],
        ),
    ];
    let mut failures = Vec::new();
    for (case, tamper, expected, needles) in &cases {
        let mut tampered = header.clone();
        tamper(&mut tampered, &flipped_sha);
        let bytes = serde_json::to_vec(&tampered).expect("serialize header");
        let mut head = preamble;
        head[43..51].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
        if *case == "flipped_source_sha256" {
            for (at, pair) in flipped_sha.as_bytes().chunks(2).enumerate() {
                let hex = std::str::from_utf8(pair).expect("ascii");
                head[11 + at] = u8::from_str_radix(hex, 16).expect("hex digit pair");
            }
        }
        let dir = root.join(case);
        std::fs::create_dir_all(&dir).expect("case dir");
        let path = dir.join("unlimited-ocr-2.v0.1.0.int8.focrq");
        let mut out = std::fs::File::create(&path).expect("create stand-in");
        out.write_all(&head).expect("write preamble");
        out.write_all(&bytes).expect("write header");
        out.set_len(51 + bytes.len() as u64 + payload_len)
            .expect("sparse payload");
        drop(out);

        let file_spec = path.display().to_string();
        // An armed refused lever stops the untouched copy at admission (exit
        // 2), right after the gate, so its sparse payload is never decoded;
        // a tampered copy is refused by the gate first (exit 7).
        let lever = ("FOCR_PREFILL_CHUNK", "64".to_owned());
        for (route, spec, envs) in [
            ("path", file_spec.as_str(), vec![lever.clone()]),
            (
                "short_name",
                "unlimited-ocr-2",
                vec![lever.clone(), ("FOCR_MODEL_DIR", dir.display().to_string())],
            ),
        ] {
            let args = ["ocr", image.as_str(), "--model", spec];
            let started = Instant::now();
            let got = run_focr_only(&bin, &args, &envs);
            let pass = got.code == Some(*expected)
                && needles
                    .iter()
                    .all(|needle| got.stderr.contains(needle.as_str()));
            println!(
                "{}",
                serde_json::json!({
                    "schema_version": 1, "ts": 0.0, "test": TEST, "case": format!("{case}:{route}"),
                    "run_seq": 0, "event": "assert", "result": if pass { "pass" } else { "fail" },
                    "assertion": "v2 .focrq held to its recipe before any weight is read",
                    "pass": pass, "args": args, "expected_exit": expected, "exit_code": got.code,
                    "needles": needles, "message": got.stderr.trim(),
                    "elapsed_us": started.elapsed().as_micros(), "weights_touched": false,
                })
            );
            if !pass {
                failures.push(format!(
                    "{case}:{route}: exit {:?}: {}",
                    got.code,
                    got.stderr.trim()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

// ═══════════════════════════════════════════════════════════════════════════
// U2-B6: Unlimited-OCR-2 through the CLI, model-gated on its raw checkpoint
// (`FOCR_UOCR2_SOURCE_DIR`) and the corpus pages of S1/S2. Exact token parity
// is U2-C2's; these runs check routing, page counts, partial output and exits.
// ═══════════════════════════════════════════════════════════════════════════

/// The raw v2 checkpoint, the binary and S2's pages (p9, p14, p19), or `None`
/// (skip-with-SUCCESS).
fn v2_inputs(test: &str) -> Option<(PathBuf, PathBuf, Vec<PathBuf>)> {
    let Some(model) = std::env::var_os("FOCR_UOCR2_SOURCE_DIR")
        .map(PathBuf::from)
        .filter(|dir| dir.is_dir())
    else {
        log_line(
            test,
            "gate",
            "skip",
            "skip_no_model",
            "reason=\"FOCR_UOCR2_SOURCE_DIR unset\"",
        );
        log_success(
            test,
            "gate",
            "model-gated: set FOCR_UOCR2_SOURCE_DIR to the raw Unlimited-OCR-2 checkpoint",
        );
        return None;
    };
    let Some(bin) = focr_bin() else {
        log_success(test, "gate", "no built focr binary");
        return None;
    };
    let Some(pages) = corpus_pages(&["page_0009.png", "page_0014.png", "page_0019.png"]) else {
        log_line(
            test,
            "gate",
            "skip",
            "skip_no_model",
            "reason=\"corpus pages absent\"",
        );
        log_success(
            test,
            "gate",
            "set FOCR_CORPUS_DIR to the 20-page corpus pages dir",
        );
        return None;
    };
    Some((model, bin, pages))
}

/// `focr args --model <v2 dir>`, without the resident daemon or any inherited
/// `FOCR_*` variable but `envs`.
fn run_v2(bin: &Path, model: &Path, args: &[&str], envs: &[(&str, String)]) -> CliOut {
    let model = model.display().to_string();
    let mut args = args.to_vec();
    args.extend(["--model", model.as_str()]);
    let mut envs = envs.to_vec();
    envs.push(("FOCR_NO_RESIDENT", "1".into()));
    run_focr_only(bin, &args, &envs)
}

/// The JSON lines of `stdout`.
fn json_lines(stdout: &str) -> Vec<serde_json::Value> {
    stdout
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .collect()
}

/// A PDF whose pages each hold one PNG's RGB pixels as a lossless
/// (FlateDecode) image, so its raster equals the PNG's.
fn pdf_of_pages(pages: &[PathBuf]) -> PathBuf {
    use lopdf::{Object, Stream, dictionary};
    let mut doc = lopdf::Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let mut kids: Vec<Object> = Vec::new();
    for page in pages {
        let rgb = image::open(page).expect("corpus page decodes").to_rgb8();
        let (w, h) = rgb.dimensions();
        let mut stream = Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Image",
                "Width" => i64::from(w),
                "Height" => i64::from(h),
                "ColorSpace" => "DeviceRGB",
                "BitsPerComponent" => 8,
            },
            rgb.into_raw(),
        );
        stream.compress().expect("compress the page image");
        let image_id = doc.add_object(stream);
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Resources" => dictionary! { "XObject" => dictionary! { "Im0" => image_id } },
            "MediaBox" => vec![0.into(), 0.into(), i64::from(w).into(), i64::from(h).into()],
        });
        kids.push(page_id.into());
    }
    let count = i64::try_from(pages.len()).expect("page count fits");
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! { "Type" => "Pages", "Kids" => kids, "Count" => count }),
    );
    let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog_id);
    let path = unique_output_path("pdf");
    doc.save(&path).expect("save the page PDF");
    path
}

/// U2-B6 acceptance: one page through `focr ocr`, and a three-page document
/// through both multi-page entry points (`ocr-batch --multi-page` over the
/// images, `ocr <pdf> --multi-page` over a PDF of the same pixels), decode
/// to completion with one body per page. The two documents are the same
/// decode, so their markdown is identical.
#[test]
fn v2_cli_decodes_a_page_and_a_document_through_every_entry_point_when_armed() {
    const TEST: &str = "v2_cli_decodes_a_page_and_a_document_through_every_entry_point_when_armed";
    let Some((model, bin, pages)) = v2_inputs(TEST) else {
        return;
    };
    let page_args: Vec<String> = pages.iter().map(|p| p.display().to_string()).collect();

    let started = Instant::now();
    let out = run_v2(&bin, &model, &["ocr", "--robot", &page_args[0]], &[]);
    let complete = json_lines(&out.stdout)
        .into_iter()
        .find(|v| v["event"] == "run_complete")
        .unwrap_or_else(|| panic!("S1: no run_complete:\n{}\n{}", out.stdout, out.stderr));
    let markdown = complete["markdown"].as_str().unwrap_or_default();
    assert_eq!(out.code, Some(0), "S1: {}", out.stderr);
    assert_eq!(complete["finish_reason"], "stop", "S1: {complete}");
    assert!(complete.get("warnings").is_none(), "S1: {complete}");
    // U2-E1: a raw checkpoint names its model and records no digest.
    assert_eq!(complete["model_id"], "unlimited-ocr-2", "S1: {complete}");
    assert!(complete["source_sha256"].is_null(), "S1: {complete}");
    assert!(!markdown.trim().is_empty(), "S1: empty markdown");
    // U2-E3b: the block grammar is rendered away, as v1's spans are.
    for grammar in ["<BLOCK>", "<FORK>", "<LEVEL"] {
        assert!(!markdown.contains(grammar), "S1: {grammar} in:\n{markdown}");
    }
    assert!(
        out.stderr
            .contains("[focr] note: unlimited-ocr-2 is experimental"),
        "S1: not labelled experimental:\n{}",
        out.stderr
    );
    log_line(
        TEST,
        "S1",
        "result",
        "pass",
        &format!(
            "finish_reason=stop chars={} exit_code=0 wall_s={:.1}",
            markdown.chars().count(),
            started.elapsed().as_secs_f64()
        ),
    );

    let started = Instant::now();
    let mut args = vec!["ocr-batch", "--multi-page", "--json"];
    args.extend(page_args.iter().map(String::as_str));
    let out = run_v2(&bin, &model, &args, &[]);
    let batch = json_lines(&out.stdout)
        .into_iter()
        .find(|v| v["command"] == "batch.multi_page")
        .unwrap_or_else(|| panic!("S2 batch: no result:\n{}\n{}", out.stdout, out.stderr));
    let batch_markdown = batch["markdown"].as_str().unwrap_or_default().to_owned();
    assert_eq!(out.code, Some(0), "S2 batch: {}", out.stderr);
    assert_eq!(batch["finish_reason"], "stop", "S2 batch: {batch}");
    assert!(batch.get("warnings").is_none(), "S2 batch: {batch}");
    assert_eq!(batch["model_id"], "unlimited-ocr-2", "S2 batch: {batch}");
    assert_eq!(
        batch_markdown.matches("<PAGE>").count(),
        pages.len(),
        "S2 batch: one body per page:\n{batch_markdown}"
    );
    for grammar in ["<BLOCK>", "<FORK>", "<LEVEL"] {
        assert!(
            !batch_markdown.contains(grammar),
            "S2 batch: {grammar} in:\n{batch_markdown}"
        );
    }
    log_line(
        TEST,
        "S2_batch",
        "result",
        "pass",
        &format!(
            "pages=3 bodies=3 finish_reason=stop exit_code=0 wall_s={:.1}",
            started.elapsed().as_secs_f64()
        ),
    );

    let started = Instant::now();
    let pdf = pdf_of_pages(&pages);
    let pdf_arg = pdf.display().to_string();
    let out = run_v2(
        &bin,
        &model,
        &["ocr", &pdf_arg, "--multi-page", "--json"],
        &[],
    );
    let _ = std::fs::remove_file(&pdf);
    let value = json_lines(&out.stdout)
        .into_iter()
        .find(|v| v.get("markdown").is_some())
        .unwrap_or_else(|| panic!("S2 pdf: no result:\n{}\n{}", out.stdout, out.stderr));
    assert_eq!(out.code, Some(0), "S2 pdf: {}", out.stderr);
    assert_eq!(
        value["markdown"].as_str().unwrap_or_default(),
        batch_markdown,
        "S2: the PDF and the image list are the same document"
    );
    log_line(
        TEST,
        "S2_pdf",
        "result",
        "pass",
        &format!(
            "pages=3 bodies=3 equals_batch=true exit_code=0 wall_s={:.1}",
            started.elapsed().as_secs_f64()
        ),
    );
}

/// U2-B6: a document that stops early writes the pages it produced, then
/// exits 5. At the position limit (`FOCR_TEST_MAX_POSITIONS`, 60 positions
/// past S2's prompt) the PDF run writes two bodies to `-o`, and its robot
/// stream ends with a `run_error` carrying `finish_reason: length`,
/// `pages_out` and a `v2_length` warning, with no `run_complete`. With the
/// forward budget spent (`FOCR_STAGE_BUDGET_FORWARD_MS=1`) the image-list
/// run prints its (empty) document with `finish_reason: budget`, then exits 5.
#[test]
fn v2_cli_writes_a_stopped_document_then_exits_5_when_armed() {
    const TEST: &str = "v2_cli_writes_a_stopped_document_then_exits_5_when_armed";
    let Some((model, bin, pages)) = v2_inputs(TEST) else {
        return;
    };
    let page_args: Vec<String> = pages.iter().map(|p| p.display().to_string()).collect();
    let mut batch_args = vec!["ocr-batch", "--multi-page", "--json"];
    batch_args.extend(page_args.iter().map(String::as_str));

    // Probe the prompt length: a 1-position limit refuses with it.
    let probe = run_v2(
        &bin,
        &model,
        &batch_args,
        &[("FOCR_TEST_MAX_POSITIONS", "1".into())],
    );
    let prompt_len: usize = probe
        .stderr
        .split("prompt is ")
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .and_then(|n| n.replace(',', "").parse().ok())
        .unwrap_or_else(|| panic!("no prompt length in the refusal: {}", probe.stderr));

    let started = Instant::now();
    let pdf = pdf_of_pages(&pages);
    let pdf_arg = pdf.display().to_string();
    let output = unique_output_path("md");
    let output_arg = output.display().to_string();
    let limit = prompt_len + 60;
    let out = run_v2(
        &bin,
        &model,
        &[
            "ocr",
            &pdf_arg,
            "--multi-page",
            "--robot",
            "-o",
            &output_arg,
        ],
        &[("FOCR_TEST_MAX_POSITIONS", limit.to_string())],
    );
    let _ = std::fs::remove_file(&pdf);
    let written = std::fs::read_to_string(&output).unwrap_or_default();
    let _ = std::fs::remove_file(&output);
    let events = json_lines(&out.stdout);
    assert_eq!(out.code, Some(5), "length: {}", out.stderr);
    assert!(
        events.iter().all(|v| v["event"] != "run_complete"),
        "length: a partial run never completes: {}",
        out.stdout
    );
    let last = events.last().expect("robot events");
    assert_eq!(last["event"], "run_error", "length: {last}");
    assert_eq!(last["code"], 5, "length: {last}");
    assert_eq!(last["finish_reason"], "length", "length: {last}");
    assert_eq!(last["pages_out"], 2, "length: {last}");
    assert!(
        warning_kinds(last).contains(&"v2_length".to_owned()),
        "length: {last}"
    );
    assert_eq!(
        written.matches("<PAGE>").count(),
        2,
        "length: the produced bodies are written:\n{written}"
    );
    log_line(
        TEST,
        "position_limit",
        "result",
        "pass",
        &format!(
            "prompt_len={prompt_len} limit={limit} pages_out=2 finish_reason=length \
             warning_kinds={:?} exit_code=5 wall_s={:.1}",
            warning_kinds(last),
            started.elapsed().as_secs_f64()
        ),
    );

    let started = Instant::now();
    let out = run_v2(
        &bin,
        &model,
        &batch_args,
        &[("FOCR_STAGE_BUDGET_FORWARD_MS", "1".into())],
    );
    let batch = json_lines(&out.stdout)
        .into_iter()
        .find(|v| v["command"] == "batch.multi_page")
        .unwrap_or_else(|| panic!("budget: no result:\n{}\n{}", out.stdout, out.stderr));
    assert_eq!(out.code, Some(5), "budget: {}", out.stderr);
    assert_eq!(batch["finish_reason"], "budget", "budget: {batch}");
    assert!(
        warning_kinds(&batch).contains(&"v2_budget".to_owned()),
        "budget: {batch}"
    );
    assert!(
        out.stderr
            .contains("stopped early (budget) after 0 of 3 page bodies"),
        "budget: {}",
        out.stderr
    );
    log_line(
        TEST,
        "budget",
        "result",
        "pass",
        &format!(
            "pages_out=0 finish_reason=budget warning_kinds={:?} exit_code=5 wall_s={:.1}",
            warning_kinds(&batch),
            started.elapsed().as_secs_f64()
        ),
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// U2-D0: a directory of page images is a document. Its pages are read lazily
// through the page source, in natural order; files that are not page images
// are left out with one `ignored_files` warning.
// ═══════════════════════════════════════════════════════════════════════════

/// A fresh directory holding `pages` (tiny PNGs) and `strays` (text files).
fn page_dir(tag: &str, pages: &[&str], strays: &[&str]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("focr_e2e_pages_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create page dir");
    for name in pages {
        std::fs::write(dir.join(name), tiny_png(4, 4)).expect("write page");
    }
    for name in strays {
        std::fs::write(dir.join(name), "not a page").expect("write stray file");
    }
    dir
}

/// Always on (no weights): `focr ocr <dir>` reads the directory as a
/// document, reports its stray file once, and keeps every refusal a PDF has.
#[test]
fn directory_input_is_a_document_and_keeps_its_refusals_without_a_model() {
    const TEST: &str = "directory_input_is_a_document_and_keeps_its_refusals_without_a_model";
    let Some(bin) = focr_bin() else {
        log_success(TEST, "gate", "no built focr binary");
        return;
    };
    let dir = page_dir("refusals", &["p1.png", "p2.png"], &["notes.txt"]);
    let dir_arg = dir.display().to_string();
    let absent = "/nonexistent-model.focrq";
    let envs = [("FOCR_NO_RESIDENT", "1".to_string())];

    // Accepted as a document: the stray file is reported when the source
    // opens, then the absent model ends the run (exit 3).
    for (case, extra) in [("per_page", &[][..]), ("multi_page", &["--multi-page"][..])] {
        let mut args = vec!["ocr", dir_arg.as_str(), "--model", absent];
        args.extend_from_slice(extra);
        let out = run_focr_only(&bin, &args, &envs);
        assert_eq!(out.code, Some(3), "{case}: {}", out.stderr);
        assert_eq!(
            out.stderr.matches("are not page images").count(),
            1,
            "{case}: one ignored_files warning:\n{}",
            out.stderr
        );
        assert!(out.stderr.contains("notes.txt"), "{case}: {}", out.stderr);
        log_line(
            TEST,
            case,
            "result",
            "pass",
            "exit_code=3 ignored_files_warnings=1",
        );
    }

    // `--pages` counts the directory's images.
    let out = run_focr_only(
        &bin,
        &["ocr", &dir_arg, "--pages", "3", "--model", absent],
        &envs,
    );
    assert_eq!(out.code, Some(2), "pages: {}", out.stderr);
    assert!(
        out.stderr.contains("this document has 2 page(s)"),
        "pages: {}",
        out.stderr
    );
    log_line(TEST, "pages_out_of_range", "result", "pass", "exit_code=2");

    // `--multi-page` still refuses `--extract-figures` on a directory.
    let output = dir.join("out.md").display().to_string();
    let out = run_focr_only(
        &bin,
        &[
            "ocr",
            &dir_arg,
            "--multi-page",
            "--extract-figures",
            "-o",
            &output,
            "--model",
            absent,
        ],
        &envs,
    );
    assert_eq!(out.code, Some(2), "figures: {}", out.stderr);
    assert!(
        out.stderr.contains("does not compose"),
        "figures: {}",
        out.stderr
    );
    log_line(TEST, "multi_page_figures", "result", "pass", "exit_code=2");

    // A single image is still not a document.
    let png = write_tiny_png();
    let png_arg = png.display().to_string();
    let out = run_focr_only(
        &bin,
        &["ocr", &png_arg, "--multi-page", "--model", absent],
        &envs,
    );
    let _ = std::fs::remove_file(&png);
    assert_eq!(out.code, Some(2), "image: {}", out.stderr);
    assert!(
        out.stderr.contains("ocr-batch --multi-page"),
        "image: {}",
        out.stderr
    );
    log_line(TEST, "image_multi_page", "result", "pass", "exit_code=2");

    // A directory without a page image is an input error (exit 4).
    let empty = page_dir("empty", &[], &["notes.txt"]);
    let empty_arg = empty.display().to_string();
    let out = run_focr_only(&bin, &["ocr", &empty_arg, "--model", absent], &envs);
    assert_eq!(out.code, Some(4), "empty: {}", out.stderr);
    assert!(
        out.stderr.contains("has no page images"),
        "empty: {}",
        out.stderr
    );
    log_line(TEST, "no_page_images", "result", "pass", "exit_code=4");

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&empty);
}

/// v1, armed: a directory read through the page source is the same document
/// as the image list. Its pages are named so natural order and byte order
/// disagree (`page_9` before `page_10`). The `--multi-page` pass equals
/// `ocr-batch --multi-page` over the same files, and both runs report the
/// stray file once on `run_complete` / `--json`; the per-page pass numbers
/// its pages by source page.
#[test]
fn v1_directory_input_matches_the_image_list_when_armed() {
    const TEST: &str = "v1_directory_input_matches_the_image_list_when_armed";
    let Some((model, bin, pages)) = x2_inputs(TEST) else {
        return;
    };
    let dir = page_dir("v1_document", &[], &["notes.txt"]);
    std::fs::copy(&pages[0], dir.join("page_9.png")).expect("copy page 9");
    std::fs::copy(&pages[1], dir.join("page_10.png")).expect("copy page 14");
    let dir_arg = dir.display().to_string();
    let envs = [
        ("FOCR_MODEL_PATH", model.display().to_string()),
        ("FOCR_NO_RESIDENT", "1".to_string()),
    ];

    let started = Instant::now();
    let (batch_out, batch) = batch_multi_page(&bin, &model, &pages[..2], &[]);
    let batch = batch.unwrap_or_else(|| panic!("batch: no result:\n{}", batch_out.stderr));
    assert_eq!(batch_out.code, Some(0), "batch: {}", batch_out.stderr);
    let out = run_focr_env(&bin, &["ocr", &dir_arg, "--multi-page", "--robot"], &envs);
    let complete = json_lines(&out.stdout)
        .into_iter()
        .find(|v| v["event"] == "run_complete")
        .unwrap_or_else(|| {
            panic!(
                "multi-page: no run_complete:\n{}\n{}",
                out.stdout, out.stderr
            )
        });
    assert_eq!(out.code, Some(0), "multi-page: {}", out.stderr);
    assert_eq!(
        complete["markdown"], batch["markdown"],
        "the directory and the image list are the same document"
    );
    let ignored: Vec<&serde_json::Value> = complete["warnings"]
        .as_array()
        .map(|a| a.iter().filter(|w| w["kind"] == "ignored_files").collect())
        .unwrap_or_default();
    assert_eq!(ignored.len(), 1, "multi-page: {complete}");
    assert_eq!(ignored[0]["count"], 1, "multi-page: {complete}");
    assert_eq!(
        ignored[0]["examples"],
        serde_json::json!(["notes.txt"]),
        "multi-page: {complete}"
    );
    log_line(
        TEST,
        "multi_page",
        "result",
        "pass",
        &format!(
            "equals_batch=true ignored_files=1 exit_code=0 wall_s={:.1}",
            started.elapsed().as_secs_f64()
        ),
    );

    let started = Instant::now();
    let out = run_focr_env(&bin, &["ocr", &dir_arg, "--json"], &envs);
    let value = json_lines(&out.stdout)
        .into_iter()
        .find(|v| v.get("pages").is_some())
        .unwrap_or_else(|| panic!("per-page: no result:\n{}\n{}", out.stdout, out.stderr));
    assert_eq!(out.code, Some(0), "per-page: {}", out.stderr);
    let numbers: Vec<u64> = value["pages"]
        .as_array()
        .map(|a| a.iter().filter_map(|p| p["page"].as_u64()).collect())
        .unwrap_or_default();
    assert_eq!(numbers, vec![1, 2], "per-page: {value}");
    assert!(
        warning_kinds(&value).contains(&"ignored_files".to_owned()),
        "per-page: {value}"
    );
    log_line(
        TEST,
        "per_page",
        "result",
        "pass",
        &format!(
            "pages=[1,2] ignored_files=1 exit_code=0 wall_s={:.1}",
            started.elapsed().as_secs_f64()
        ),
    );
    let _ = std::fs::remove_dir_all(&dir);
}
