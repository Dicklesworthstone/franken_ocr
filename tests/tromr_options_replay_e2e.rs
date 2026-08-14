//! Model-backed exact-replay proof for explicit TrOMR recognition options.
//!
//! Run through `scripts/tromr_options_replay_e2e.sh`. The wrapper builds this
//! test before replacing `PATH`, so native PDF decode and TrOMR inference cannot
//! hide a successful `focr`, `pdftoppm`, ImageMagick, or other helper process.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use franken_ocr::music_input::{
    ConsumedBytesIdentity, ImmutableMusicInputBundle, MusicInputExpectations, MusicInputOptions,
    MusicSourceKind,
};
use franken_ocr::tokenizer::music::TOKENIZER_FILENAMES;
use franken_ocr::{
    FocrError, MusicCancellationToken, OcrEngine, TromrExecutionOptionsV1,
    TromrRecognitionOptionsV1, TromrSplitPolicyV1,
};
use sha2::{Digest, Sha256};

const SPOHR_PDF_BYTES: u64 = 23_927_736;
const SPOHR_PDF_SHA256: &str = "9b6b4a84400932cf5ce93bbcdc87a7041809d35ed7fecdbea9a6ebe3c8e21dac";
const TROMR_F32_BYTES: u64 = 86_168_002;
const TROMR_F32_SHA256: &str = "a9d41485a98534ad0a1f7c1ec624f0a92f3f092c7dc30ac5af636b50dc465edc";
const TROMR_INT8_BYTES: u64 = 61_107_485;
const TROMR_INT8_SHA256: &str = "cced11c0f05656dd54cc615a15939c472dc8f916f04ae154ea4a0364839f845a";
const PROVIDER_BASE_REVISION: &str = "c5a0e368b1be33187a17cd9aa716653ce6230590";
const TOKENIZER_SHA256: [&str; 4] = [
    "603bfef760e8424f7808acba423532b4beb2d88dbf085f81add6a8e543a34035",
    "2382e8b20c1473290e200789604656b3a06bdf4b55a0818a0f7d175e8cb64ade",
    "b61ba09cecd5bc343e6a038a2e26718b54cd3c08e8f9b72013ecf80c3cac86b2",
    "504d886d11e3c1fe92893abd46edfc68dfbe7a8eb83e6b51646532dad8a485e1",
];

#[derive(Clone, Copy, Debug)]
struct ProcessMetrics {
    cpu_ticks: Option<u64>,
    current_rss_kib: Option<u64>,
    peak_rss_kib: Option<u64>,
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn identity_json(identity: ConsumedBytesIdentity) -> serde_json::Value {
    serde_json::json!({
        "byte_len": identity.byte_len,
        "sha256": identity.sha256_hex(),
        "blake3": identity.blake3_prefixed(),
    })
}

fn collect_rust_sources(dir: &Path, root: &Path, sources: &mut Vec<(String, PathBuf)>) {
    for entry in std::fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("walk provider source {}: {error}", dir.display()))
    {
        let entry = entry.expect("provider source entry is readable");
        let path = entry.path();
        let file_type = entry
            .file_type()
            .unwrap_or_else(|error| panic!("inspect provider source {}: {error}", path.display()));
        if file_type.is_dir() {
            collect_rust_sources(&path, root, sources);
        } else if file_type.is_file() && path.extension().is_some_and(|ext| ext == "rs") {
            let relative = path
                .strip_prefix(root)
                .expect("provider source lies beneath its root")
                .components()
                .map(|component| component.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            sources.push((relative, path));
        }
    }
}

fn provider_source_tree_sha256() -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut sources = Vec::new();
    collect_rust_sources(&root.join("src"), root, &mut sources);
    sources.sort_by(|left, right| left.0.cmp(&right.0));
    let mut inventory = Vec::new();
    for (relative, path) in sources {
        let digest = sha256_hex(&read(&path));
        inventory.extend_from_slice(digest.as_bytes());
        inventory.extend_from_slice(b"  ");
        inventory.extend_from_slice(relative.as_bytes());
        inventory.push(b'\n');
    }
    sha256_hex(&inventory)
}

fn linux_status_kib(name: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let value = line.strip_prefix(name)?.trim();
        value.strip_suffix(" kB")?.trim().parse().ok()
    })
}

fn linux_cpu_ticks() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let after_command = stat.get(stat.rfind(')')? + 2..)?;
    let fields: Vec<&str> = after_command.split_whitespace().collect();
    let user: u64 = fields.get(11)?.parse().ok()?;
    let system: u64 = fields.get(12)?.parse().ok()?;
    user.checked_add(system)
}

fn process_metrics() -> ProcessMetrics {
    ProcessMetrics {
        cpu_ticks: linux_cpu_ticks(),
        current_rss_kib: linux_status_kib("VmRSS:"),
        peak_rss_kib: linux_status_kib("VmHWM:"),
    }
}

fn delta(after: Option<u64>, before: Option<u64>) -> Option<u64> {
    after.zip(before).and_then(|(a, b)| a.checked_sub(b))
}

fn shell_quote(value: &Path) -> String {
    format!("'{}'", value.display().to_string().replace('\'', "'\"'\"'"))
}

#[derive(Clone)]
struct PinnedRuntimeCase {
    model_dir: PathBuf,
    source_path: PathBuf,
    model_path: PathBuf,
    quant: String,
    selected_page: usize,
    recognition: TromrRecognitionOptionsV1,
    expectations: MusicInputExpectations,
}

impl PinnedRuntimeCase {
    fn options(&self, execution: TromrExecutionOptionsV1) -> MusicInputOptions {
        MusicInputOptions {
            page: Some(self.selected_page),
            recognition: self.recognition,
            execution,
            expectations: self.expectations.clone(),
        }
    }
}

fn pinned_runtime_case() -> PinnedRuntimeCase {
    let model_dir = PathBuf::from(
        std::env::var_os("FOCR_TROMR_DIR")
            .expect("FOCR_TROMR_DIR must contain a pinned TrOMR model and tokenizer tables"),
    );
    let source_path = PathBuf::from(
        std::env::var_os("FOCR_TROMR_PDF")
            .expect("FOCR_TROMR_PDF must name the public 1843 Spohr scan"),
    );
    let quant = std::env::var("FOCR_TROMR_QUANT").unwrap_or_else(|_| "int8".to_owned());
    let (model_filename, expected_model_bytes, expected_model_sha256) = match quant.as_str() {
        "f32" => ("tromr.focrq", TROMR_F32_BYTES, TROMR_F32_SHA256),
        "int8" => ("tromr.int8.focrq", TROMR_INT8_BYTES, TROMR_INT8_SHA256),
        other => panic!("FOCR_TROMR_QUANT must be f32 or int8, got {other:?}"),
    };
    let selected_page = std::env::var("FOCR_TROMR_PAGE")
        .unwrap_or_else(|_| "100".to_owned())
        .parse::<usize>()
        .expect("FOCR_TROMR_PAGE must be a positive one-based page");
    assert!(selected_page > 0, "FOCR_TROMR_PAGE must be positive");
    let model_path = model_dir.join(model_filename);

    let source_bytes = read(&source_path);
    let model_bytes = read(&model_path);
    let tokenizer_bytes = TOKENIZER_FILENAMES.map(|filename| read(&model_dir.join(filename)));
    assert_eq!(source_bytes.len() as u64, SPOHR_PDF_BYTES);
    assert_eq!(sha256_hex(&source_bytes), SPOHR_PDF_SHA256);
    assert_eq!(model_bytes.len() as u64, expected_model_bytes);
    assert_eq!(sha256_hex(&model_bytes), expected_model_sha256);
    for (index, bytes) in tokenizer_bytes.iter().enumerate() {
        assert_eq!(
            sha256_hex(bytes),
            TOKENIZER_SHA256[index],
            "pinned tokenizer identity changed: {}",
            TOKENIZER_FILENAMES[index]
        );
    }
    let expectations = MusicInputExpectations {
        source_sha256: Some(sha256(&source_bytes)),
        model_sha256: Some(sha256(&model_bytes)),
        tokenizer_sha256: tokenizer_bytes.each_ref().map(|bytes| Some(sha256(bytes))),
    };

    PinnedRuntimeCase {
        model_dir,
        source_path,
        model_path,
        quant,
        selected_page,
        recognition: TromrRecognitionOptionsV1::deterministic(),
        expectations,
    }
}

fn positive_millis_from_env(name: &str, default: u64) -> u64 {
    let value = std::env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .parse::<u64>()
        .unwrap_or_else(|error| panic!("{name} must be a positive integer: {error}"));
    assert!(value > 0, "{name} must be positive");
    value
}

fn open_runtime_bundle(
    case: &PinnedRuntimeCase,
    execution: TromrExecutionOptionsV1,
) -> (ImmutableMusicInputBundle, serde_json::Value) {
    let metrics_before = process_metrics();
    let started = Instant::now();
    let prepared = ImmutableMusicInputBundle::open_observed(
        &case.source_path,
        &case.model_path,
        case.options(execution),
    )
    .expect("native provider pins PDF/model/tokenizers and renders the selected page");
    let bundle = prepared.bundle;
    let wall_ms = started.elapsed().as_millis();
    let metrics_after = process_metrics();
    assert_eq!(bundle.provenance().source_kind, MusicSourceKind::Pdf);
    assert_eq!(bundle.provenance().selected_page, case.selected_page);
    assert_eq!(bundle.provenance().raster_width, 2_696);
    assert_eq!(bundle.provenance().raster_height, 3_926);
    let diagnostics = serde_json::json!({
        "phase": "immutable_bundle_preparation",
        "provider_diagnostics": prepared.diagnostics,
        "wall_ms": wall_ms,
        "cpu_ticks": delta(metrics_after.cpu_ticks, metrics_before.cpu_ticks),
        "current_rss_kib": metrics_after.current_rss_kib,
        "peak_rss_kib": metrics_after.peak_rss_kib,
        "bundle_sha256": bundle.provenance().bundle_sha256_hex(),
        "raster_sha256": bundle.provenance().raster_sha256_hex(),
        "recognition_options_identity": bundle.provenance().recognition_options_identity,
        "execution_options_identity": bundle.provenance().execution_options_identity,
        "options_sha256": bundle.provenance().options_sha256_hex(),
        "replay_sha256": bundle.provenance().replay_sha256_hex(),
    });
    (bundle, diagnostics)
}

fn recovery_probe(
    engine: &OcrEngine,
    case: &PinnedRuntimeCase,
    terminal_bound_ms: u64,
    after: &str,
) -> serde_json::Value {
    let execution = TromrExecutionOptionsV1 {
        setup_budget_ms: 30_000,
        per_forward_attempt_budget_ms: 1,
        max_forward_attempts: 1,
        ..TromrExecutionOptionsV1::default()
    };
    let (bundle, preparation) = open_runtime_bundle(case, execution);
    let metrics_before = process_metrics();
    let started = Instant::now();
    let failure = engine
        .recognize_immutable_music_observed(bundle, MusicCancellationToken::new())
        .expect_err("one-attempt recovery probe must exhaust its explicit policy");
    let wall_ms = started.elapsed().as_millis();
    let metrics_after = process_metrics();
    assert!(
        matches!(&failure.error, FocrError::Timeout(_)),
        "recovery probe after {after} returned {:?}",
        failure.error
    );
    let detail = failure.error.to_string();
    assert!(
        detail.contains("after 1 forward attempts")
            || detail.contains("maximum of 1 forward attempts"),
        "recovery probe after {after} did not acquire admission and start a real forward: {detail}"
    );
    assert!(
        wall_ms < u128::from(terminal_bound_ms),
        "recovery probe after {after} exceeded {terminal_bound_ms}ms: {wall_ms}ms"
    );
    serde_json::json!({
        "after": after,
        "preparation": preparation,
        "execution_options": execution,
        "execution_options_identity": execution.replay_identity().expect("valid recovery policy"),
        "outcome_kind": failure.error.kind(),
        "detail": detail,
        "provider_diagnostics": failure.diagnostics,
        "wall_ms": wall_ms,
        "cpu_ticks": delta(metrics_after.cpu_ticks, metrics_before.cpu_ticks),
        "current_rss_kib": metrics_after.current_rss_kib,
        "peak_rss_kib": metrics_after.peak_rss_kib,
        "forward_attempt_started": true,
    })
}

#[test]
#[ignore = "requires FOCR_TROMR_DIR and FOCR_TROMR_PDF with the pinned public artifacts"]
fn native_spohr_pdf_has_exact_two_run_options_replay() {
    assert_eq!(
        std::env::var_os("PATH").as_deref(),
        Some(std::ffi::OsStr::new(
            "/nonexistent/franken_ocr_process_trap"
        )),
        "run through scripts/tromr_options_replay_e2e.sh so helper processes cannot succeed"
    );
    let model_dir = PathBuf::from(
        std::env::var_os("FOCR_TROMR_DIR")
            .expect("FOCR_TROMR_DIR must contain tromr.focrq and four tokenizer tables"),
    );
    let source_path = PathBuf::from(
        std::env::var_os("FOCR_TROMR_PDF")
            .expect("FOCR_TROMR_PDF must name the public 1843 Spohr scan"),
    );
    let quant = std::env::var("FOCR_TROMR_QUANT").unwrap_or_else(|_| "f32".to_owned());
    let (model_filename, expected_model_bytes, expected_model_sha256) = match quant.as_str() {
        "f32" => ("tromr.focrq", TROMR_F32_BYTES, TROMR_F32_SHA256),
        "int8" => ("tromr.int8.focrq", TROMR_INT8_BYTES, TROMR_INT8_SHA256),
        other => panic!("FOCR_TROMR_QUANT must be f32 or int8, got {other:?}"),
    };
    let selected_page = std::env::var("FOCR_TROMR_PAGE")
        .unwrap_or_else(|_| "100".to_owned())
        .parse::<usize>()
        .expect("FOCR_TROMR_PAGE must be a positive one-based page");
    assert!(selected_page > 0, "FOCR_TROMR_PAGE must be positive");
    let model_path = model_dir.join(model_filename);
    let provider_source_tree_sha256 = provider_source_tree_sha256();

    let source_bytes = read(&source_path);
    let model_bytes = read(&model_path);
    let tokenizer_bytes = TOKENIZER_FILENAMES.map(|filename| read(&model_dir.join(filename)));
    assert_eq!(source_bytes.len() as u64, SPOHR_PDF_BYTES);
    assert_eq!(sha256_hex(&source_bytes), SPOHR_PDF_SHA256);
    assert_eq!(model_bytes.len() as u64, expected_model_bytes);
    assert_eq!(sha256_hex(&model_bytes), expected_model_sha256);
    for (index, bytes) in tokenizer_bytes.iter().enumerate() {
        assert_eq!(
            sha256_hex(bytes),
            TOKENIZER_SHA256[index],
            "pinned tokenizer identity changed: {}",
            TOKENIZER_FILENAMES[index]
        );
    }

    let expectations = MusicInputExpectations {
        source_sha256: Some(sha256(&source_bytes)),
        model_sha256: Some(sha256(&model_bytes)),
        tokenizer_sha256: tokenizer_bytes.each_ref().map(|bytes| Some(sha256(bytes))),
    };
    let recognition = TromrRecognitionOptionsV1::deterministic();
    let options = MusicInputOptions {
        page: Some(selected_page),
        recognition,
        execution: franken_ocr::TromrExecutionOptionsV1::default(),
        expectations,
    };

    let metrics_before = process_metrics();
    let first_started = Instant::now();
    eprintln!(
        "{}",
        serde_json::json!({
            "schema": "franken_ocr.tromr_options_replay_e2e.progress.v1",
            "phase": "run_start",
            "run": 1,
            "model_quant": quant,
            "selected_page": selected_page,
            "recognition_options_identity": recognition.replay_identity().expect("valid options identity"),
        })
    );
    let first_bundle = ImmutableMusicInputBundle::open(&source_path, &model_path, options.clone())
        .expect("native franken_ocr PDF bundle opens from exact owned bytes");
    assert_eq!(first_bundle.provenance().source_kind, MusicSourceKind::Pdf);
    assert_eq!(first_bundle.provenance().selected_page, selected_page);
    assert_eq!(first_bundle.provenance().raster_width, 2_696);
    assert_eq!(first_bundle.provenance().raster_height, 3_926);
    let first = OcrEngine::new()
        .expect("embedded engine constructs")
        .recognize_immutable_music(first_bundle)
        .expect("first embedded native-PDF TrOMR recognition succeeds");
    let first_provenance = first.provenance();
    let first_page_meta = first.page_meta();
    let first_musicxml = first.musicxml();
    let first_wall_ms = first_started.elapsed().as_millis();
    let metrics_after_first = process_metrics();
    eprintln!(
        "{}",
        serde_json::json!({
            "schema": "franken_ocr.tromr_options_replay_e2e.progress.v1",
            "phase": "run_complete",
            "run": 1,
            "wall_ms": first_wall_ms,
            "cpu_ticks": delta(metrics_after_first.cpu_ticks, metrics_before.cpu_ticks),
            "current_rss_kib": metrics_after_first.current_rss_kib,
            "peak_rss_kib": metrics_after_first.peak_rss_kib,
            "detected_staff_count": first_page_meta.detected_staff_count,
            "staff_segmentation_disposition": first_page_meta.staff_segmentation_disposition,
            "attempts": first_page_meta.staff_evidence.len(),
            "fragments": first_page_meta.fragments.len(),
            "skips": first_page_meta.skips.len(),
        })
    );

    let second_started = Instant::now();
    eprintln!(
        "{}",
        serde_json::json!({
            "schema": "franken_ocr.tromr_options_replay_e2e.progress.v1",
            "phase": "run_start",
            "run": 2,
            "model_quant": quant,
            "selected_page": selected_page,
            "recognition_options_identity": first_provenance.recognition_options_identity,
        })
    );
    let second_bundle = ImmutableMusicInputBundle::open(&source_path, &model_path, options.clone())
        .expect("same exact provider inputs open for replay");
    let second = OcrEngine::new()
        .expect("second embedded engine constructs")
        .recognize_immutable_music(second_bundle)
        .expect("second embedded native-PDF TrOMR recognition succeeds");
    let second_provenance = second.provenance();
    let second_page_meta = second.page_meta();
    let second_musicxml = second.musicxml();
    let second_wall_ms = second_started.elapsed().as_millis();
    let metrics_after_second = process_metrics();
    eprintln!(
        "{}",
        serde_json::json!({
            "schema": "franken_ocr.tromr_options_replay_e2e.progress.v1",
            "phase": "run_complete",
            "run": 2,
            "wall_ms": second_wall_ms,
            "cpu_ticks": delta(metrics_after_second.cpu_ticks, metrics_after_first.cpu_ticks),
            "current_rss_kib": metrics_after_second.current_rss_kib,
            "peak_rss_kib": metrics_after_second.peak_rss_kib,
            "detected_staff_count": second_page_meta.detected_staff_count,
            "staff_segmentation_disposition": second_page_meta.staff_segmentation_disposition,
            "attempts": second_page_meta.staff_evidence.len(),
            "fragments": second_page_meta.fragments.len(),
            "skips": second_page_meta.skips.len(),
        })
    );

    assert_eq!(first_provenance, second_provenance, "replay receipt drift");
    assert_eq!(
        first_page_meta.detected_staff_count, second_page_meta.detected_staff_count,
        "detected-staff count drift"
    );
    assert_eq!(
        first_page_meta.staff_segmentation_disposition,
        second_page_meta.staff_segmentation_disposition,
        "segmentation disposition drift"
    );
    assert_eq!(
        first_page_meta.staff_evidence, second_page_meta.staff_evidence,
        "row-attempt evidence drift"
    );
    assert_eq!(
        first_page_meta.fragments, second_page_meta.fragments,
        "recognized semantic fragments drift"
    );
    assert_eq!(
        first_page_meta.staves, second_page_meta.staves,
        "staff ledger drift"
    );
    assert_eq!(
        first_page_meta.skips, second_page_meta.skips,
        "skip ledger drift"
    );
    assert_eq!(
        first_page_meta.warnings, second_page_meta.warnings,
        "warning drift"
    );
    assert_eq!(first_musicxml, second_musicxml, "MusicXML bytes drift");
    assert!(
        !first_page_meta.staff_evidence.is_empty(),
        "no staff attempts recorded"
    );
    assert!(
        !first_page_meta.fragments.is_empty(),
        "no score fragments recognized"
    );
    let xml_violations = franken_ocr::native_engine::tromr::validate_musicxml(first_musicxml);
    assert!(
        xml_violations.is_empty(),
        "provider MusicXML violations: {xml_violations:?}"
    );

    let changed_recognition = TromrRecognitionOptionsV1 {
        split_policy: TromrSplitPolicyV1::ExperimentalBarlineSegments,
        ..recognition
    };
    let changed_bundle = ImmutableMusicInputBundle::open(
        &source_path,
        &model_path,
        MusicInputOptions {
            recognition: changed_recognition,
            ..options.clone()
        },
    )
    .expect("one declared option change remains a valid provider bundle");
    assert_ne!(
        first_provenance.recognition_options_identity,
        changed_bundle.provenance().recognition_options_identity,
        "recognition option identity ignored split policy"
    );
    assert_ne!(
        first_provenance.replay_sha256,
        changed_bundle.provenance().replay_sha256,
        "provider replay identity ignored split policy"
    );
    let changed_execution_bundle = ImmutableMusicInputBundle::open(
        &source_path,
        &model_path,
        MusicInputOptions {
            execution: franken_ocr::TromrExecutionOptionsV1 {
                setup_budget_ms: options.execution.setup_budget_ms + 1,
                ..options.execution
            },
            ..options
        },
    )
    .expect("one execution-policy change remains a valid provider bundle");
    assert_eq!(
        first_provenance.recognition_options_identity,
        changed_execution_bundle
            .provenance()
            .recognition_options_identity,
        "execution-only change altered recognition identity"
    );
    assert_ne!(
        first_provenance.execution_options_identity,
        changed_execution_bundle
            .provenance()
            .execution_options_identity,
        "execution identity ignored setup budget"
    );
    assert_ne!(
        first_provenance.options_sha256,
        changed_execution_bundle.provenance().options_sha256,
        "combined options identity ignored execution policy"
    );
    assert_ne!(
        first_provenance.replay_sha256,
        changed_execution_bundle.provenance().replay_sha256,
        "provider replay identity ignored execution policy"
    );

    let attempts: Vec<_> = first_page_meta
        .staff_evidence
        .iter()
        .map(|attempt| {
            serde_json::json!({
                "detection_index": attempt.index,
                "source_bbox": attempt.geometry.source_bbox,
                "canvas_width": attempt.geometry.canvas_width,
                "canvas_height": attempt.geometry.canvas_height,
                "padding": {
                    "top": attempt.geometry.padding.top,
                    "right": attempt.geometry.padding.right,
                    "bottom": attempt.geometry.padding.bottom,
                    "left": attempt.geometry.padding.left,
                },
                "outcome": attempt.outcome.as_str(),
                "reason": attempt.reason,
            })
        })
        .collect();
    let fragments: Vec<_> = first_page_meta
        .fragments
        .iter()
        .map(|fragment| {
            serde_json::json!({
                "detection_index": fragment.detection_index,
                "source_bbox": fragment.bbox,
                "semantic_sha256": sha256_hex(fragment.semantic.as_bytes()),
                "semantic_bytes": fragment.semantic.len(),
            })
        })
        .collect();
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target"));
    let repro = format!(
        "FOCR_TROMR_DIR={} FOCR_TROMR_QUANT={} FOCR_TROMR_PDF={} FOCR_TROMR_PAGE={} CARGO_TARGET_DIR={} ./scripts/tromr_options_replay_e2e.sh",
        shell_quote(&model_dir),
        quant,
        shell_quote(&source_path),
        selected_page,
        shell_quote(&target_dir),
    );
    let tokenizer_receipts: Vec<_> = first_provenance
        .tokenizers
        .iter()
        .copied()
        .map(identity_json)
        .collect();
    let runs = serde_json::json!([
        {
            "run": 1,
            "wall_ms": first_wall_ms,
            "cpu_ticks": delta(metrics_after_first.cpu_ticks, metrics_before.cpu_ticks),
            "current_rss_kib": metrics_after_first.current_rss_kib,
            "peak_rss_kib": metrics_after_first.peak_rss_kib,
        },
        {
            "run": 2,
            "wall_ms": second_wall_ms,
            "cpu_ticks": delta(metrics_after_second.cpu_ticks, metrics_after_first.cpu_ticks),
            "current_rss_kib": metrics_after_second.current_rss_kib,
            "peak_rss_kib": metrics_after_second.peak_rss_kib,
        }
    ]);
    let execution_policy_change = serde_json::json!({
        "execution_options_identity": changed_execution_bundle
            .provenance()
            .execution_options_identity
            .clone(),
        "options_sha256": changed_execution_bundle.provenance().options_sha256_hex(),
        "replay_sha256": changed_execution_bundle.provenance().replay_sha256_hex(),
    });
    eprintln!(
        "{}",
        serde_json::json!({
            "schema": "franken_ocr.tromr_options_replay_e2e.v1",
            "provider_path": "ImmutableMusicInputBundle -> PdfPages -> OcrEngine -> TrOMR",
            "provider_package_version": env!("CARGO_PKG_VERSION"),
            "provider_base_revision": PROVIDER_BASE_REVISION,
            "provider_source_tree_sha256": provider_source_tree_sha256,
            "external_process_path": false,
            "model_quant": quant,
            "source_kind": first_provenance.source_kind.as_str(),
            "source": identity_json(first_provenance.source),
            "model": identity_json(first_provenance.model),
            "tokenizers": tokenizer_receipts,
            "bundle_sha256": first_provenance.bundle_sha256_hex(),
            "raster_sha256": first_provenance.raster_sha256_hex(),
            "options_sha256": first_provenance.options_sha256_hex(),
            "recognition_options": first_provenance.recognition_options,
            "recognition_options_identity": first_provenance.recognition_options_identity,
            "recognition_options_sha256": first_provenance.recognition_options_sha256_hex(),
            "execution_options": first_provenance.execution_options,
            "execution_options_identity": first_provenance.execution_options_identity,
            "execution_options_sha256": first_provenance.execution_options_sha256_hex(),
            "replay_sha256": first_provenance.replay_sha256_hex(),
            "changed_options_identity": changed_bundle.provenance().recognition_options_identity,
            "changed_replay_sha256": changed_bundle.provenance().replay_sha256_hex(),
            "execution_policy_change": execution_policy_change,
            "selected_page": first_provenance.selected_page,
            "page_count": first_provenance.page_count,
            "raster_width": first_provenance.raster_width,
            "raster_height": first_provenance.raster_height,
            "detected_staff_count": first_page_meta.detected_staff_count,
            "staff_segmentation_disposition": first_page_meta.staff_segmentation_disposition,
            "attempts": attempts,
            "fragments": fragments,
            "skips": first_page_meta.skips.len(),
            "warnings": first_page_meta.warnings.len(),
            "warnings_sha256": sha256_hex(format!("{:?}", first_page_meta.warnings).as_bytes()),
            "musicxml_sha256": sha256_hex(first_musicxml.as_bytes()),
            "runs": runs,
            "exact_repro": repro,
        })
    );
}

#[test]
#[ignore = "requires FOCR_TROMR_DIR and FOCR_TROMR_PDF with the pinned public artifacts"]
fn native_spohr_pdf_explicit_timeout_cancel_and_same_engine_reuse() {
    assert_eq!(
        std::env::var_os("PATH").as_deref(),
        Some(std::ffi::OsStr::new(
            "/nonexistent/franken_ocr_process_trap"
        )),
        "run through scripts/tromr_options_replay_e2e.sh so helper processes cannot succeed"
    );
    let case = pinned_runtime_case();
    let terminal_bound_ms = positive_millis_from_env("FOCR_TROMR_TERMINAL_BOUND_MS", 120_000);
    let cancel_after_ms = positive_millis_from_env("FOCR_TROMR_CANCEL_AFTER_MS", 10_000);
    assert!(
        cancel_after_ms < terminal_bound_ms,
        "cancellation delay must leave room for the terminal-latency bound"
    );
    let engine = OcrEngine::new().expect("embedded engine constructs");

    let timeout_execution = TromrExecutionOptionsV1 {
        setup_budget_ms: 100,
        per_forward_attempt_budget_ms: 100,
        max_forward_attempts: 1,
        ..TromrExecutionOptionsV1::default()
    };
    let (timeout_bundle, timeout_preparation) = open_runtime_bundle(&case, timeout_execution);
    let timeout_provenance = timeout_bundle.provenance().clone();
    let metrics_before_timeout = process_metrics();
    let timeout_started = Instant::now();
    let timeout_failure = engine
        .recognize_immutable_music_observed(timeout_bundle, MusicCancellationToken::new())
        .expect_err("real pinned request must exhaust the deliberately tiny explicit policy");
    let timeout_wall_ms = timeout_started.elapsed().as_millis();
    let metrics_after_timeout = process_metrics();
    assert!(
        matches!(&timeout_failure.error, FocrError::Timeout(_)),
        "tight real-input policy returned {:?}",
        timeout_failure.error
    );
    assert!(
        timeout_failure
            .error
            .to_string()
            .contains("explicit cumulative page allowance"),
        "timeout did not name the explicit policy: {}",
        timeout_failure.error
    );
    assert!(
        timeout_wall_ms < u128::from(terminal_bound_ms),
        "tight real-input timeout exceeded {terminal_bound_ms}ms: {timeout_wall_ms}ms"
    );
    let timeout = serde_json::json!({
        "preparation": timeout_preparation,
        "execution_options": timeout_execution,
        "execution_options_identity": timeout_execution
            .replay_identity()
            .expect("valid timeout policy"),
        "outcome_kind": timeout_failure.error.kind(),
        "detail": timeout_failure.error.to_string(),
        "provider_diagnostics": timeout_failure.diagnostics,
        "wall_ms": timeout_wall_ms,
        "cpu_ticks": delta(metrics_after_timeout.cpu_ticks, metrics_before_timeout.cpu_ticks),
        "current_rss_kib": metrics_after_timeout.current_rss_kib,
        "peak_rss_kib": metrics_after_timeout.peak_rss_kib,
        "published_score": false,
    });

    // This must acquire the process-wide forward permit and start one real
    // model attempt. If the timed-out worker detached or leaked admission, the
    // probe would expire at `forward-admission` with zero attempts instead.
    let reuse_after_timeout = recovery_probe(&engine, &case, terminal_bound_ms, "timeout");

    let cancel_execution = TromrExecutionOptionsV1::default();
    let (cancel_bundle, cancel_preparation) = open_runtime_bundle(&case, cancel_execution);
    let cancel_provenance = cancel_bundle.provenance().clone();
    let cancellation = MusicCancellationToken::new();
    let cancel_handle = cancellation.clone();
    let (cancelled_at_tx, cancelled_at_rx) = std::sync::mpsc::sync_channel(1);
    let canceler = std::thread::Builder::new()
        .name("tromr-real-e2e-canceler".to_owned())
        .spawn(move || {
            std::thread::sleep(Duration::from_millis(cancel_after_ms));
            cancel_handle.cancel();
            cancelled_at_tx
                .send(Instant::now())
                .expect("test receiver remains live");
        })
        .expect("spawn cancellation requester");
    let metrics_before_cancel = process_metrics();
    let cancel_started = Instant::now();
    let cancel_failure = engine
        .recognize_immutable_music_observed(cancel_bundle, cancellation)
        .expect_err("real pinned request must observe per-request cancellation");
    let cancel_returned = Instant::now();
    let cancel_wall_ms = cancel_started.elapsed().as_millis();
    let metrics_after_cancel = process_metrics();
    let cancellation_requested_at = cancelled_at_rx
        .recv()
        .expect("cancellation requester reports its checkpoint");
    canceler.join().expect("cancellation requester exits");
    let cancel_terminal_latency_ms = cancel_returned
        .saturating_duration_since(cancellation_requested_at)
        .as_millis();
    assert!(
        matches!(&cancel_failure.error, FocrError::Cancelled),
        "real mid-forward cancellation returned {:?}",
        cancel_failure.error
    );
    assert!(
        cancel_terminal_latency_ms < u128::from(terminal_bound_ms),
        "cancelled worker took {cancel_terminal_latency_ms}ms to quiesce after cancellation; bound is {terminal_bound_ms}ms"
    );
    assert!(
        cancel_wall_ms < u128::from(cancel_after_ms.saturating_add(terminal_bound_ms)),
        "cancelled call exceeded delay plus terminal bound: {cancel_wall_ms}ms"
    );
    let cancellation_result = serde_json::json!({
        "preparation": cancel_preparation,
        "execution_options": cancel_execution,
        "execution_options_identity": cancel_execution
            .replay_identity()
            .expect("valid cancellation policy"),
        "cancel_after_ms": cancel_after_ms,
        "terminal_latency_ms": cancel_terminal_latency_ms,
        "outcome_kind": cancel_failure.error.kind(),
        "detail": cancel_failure.error.to_string(),
        "provider_diagnostics": cancel_failure.diagnostics,
        "wall_ms": cancel_wall_ms,
        "cpu_ticks": delta(metrics_after_cancel.cpu_ticks, metrics_before_cancel.cpu_ticks),
        "current_rss_kib": metrics_after_cancel.current_rss_kib,
        "peak_rss_kib": metrics_after_cancel.peak_rss_kib,
        "published_score": false,
    });

    let reuse_after_cancellation =
        recovery_probe(&engine, &case, terminal_bound_ms, "cancellation");
    assert_eq!(
        timeout_provenance.bundle_sha256,
        cancel_provenance.bundle_sha256
    );
    assert_eq!(
        timeout_provenance.raster_sha256,
        cancel_provenance.raster_sha256
    );
    assert_eq!(
        timeout_provenance.recognition_options_sha256,
        cancel_provenance.recognition_options_sha256
    );
    assert_ne!(
        timeout_provenance.execution_options_sha256,
        cancel_provenance.execution_options_sha256
    );
    assert_ne!(
        timeout_provenance.replay_sha256,
        cancel_provenance.replay_sha256
    );

    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target"));
    let repro = format!(
        "FOCR_TROMR_CASE=budget-cancel FOCR_TROMR_DIR={} FOCR_TROMR_QUANT={} FOCR_TROMR_PDF={} FOCR_TROMR_PAGE={} FOCR_TROMR_CANCEL_AFTER_MS={} FOCR_TROMR_TERMINAL_BOUND_MS={} CARGO_TARGET_DIR={} ./scripts/tromr_options_replay_e2e.sh",
        shell_quote(&case.model_dir),
        case.quant,
        shell_quote(&case.source_path),
        case.selected_page,
        cancel_after_ms,
        terminal_bound_ms,
        shell_quote(&target_dir),
    );
    let tokenizer_receipts: Vec<_> = timeout_provenance
        .tokenizers
        .iter()
        .copied()
        .map(identity_json)
        .collect();
    eprintln!(
        "{}",
        serde_json::json!({
            "schema": "franken_ocr.tromr_execution_terminal_e2e.v1",
            "provider_path": "ImmutableMusicInputBundle -> PdfPages -> OcrEngine -> TrOMR",
            "provider_package_version": env!("CARGO_PKG_VERSION"),
            "provider_base_revision": PROVIDER_BASE_REVISION,
            "provider_source_tree_sha256": provider_source_tree_sha256(),
            "external_process_path": false,
            "model_quant": case.quant,
            "source": identity_json(timeout_provenance.source),
            "model": identity_json(timeout_provenance.model),
            "tokenizers": tokenizer_receipts,
            "selected_page": case.selected_page,
            "page_count": timeout_provenance.page_count,
            "raster_width": timeout_provenance.raster_width,
            "raster_height": timeout_provenance.raster_height,
            "recognition_options": timeout_provenance.recognition_options,
            "recognition_options_identity": timeout_provenance.recognition_options_identity,
            "recognition_options_sha256": timeout_provenance.recognition_options_sha256_hex(),
            "timeout": timeout,
            "reuse_after_timeout": reuse_after_timeout,
            "cancellation": cancellation_result,
            "reuse_after_cancellation": reuse_after_cancellation,
            "same_engine": true,
            "failure_payload_contract": "typed terminal error; no MusicXML or page metadata returned",
            "exact_repro": repro,
        })
    );
}
