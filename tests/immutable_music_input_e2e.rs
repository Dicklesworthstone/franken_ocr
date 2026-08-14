//! Public, model-backed proof for the immutable embedded TrOMR input contract.
//!
//! Run through `scripts/immutable_music_input_e2e.sh`. The wrapper builds this
//! test first, then executes the test binary with a forbidden-process `PATH`.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use franken_ocr::music_input::{
    ConsumedBytesIdentity, ImmutableMusicInputBundle, MusicInputExpectations, MusicInputOptions,
};
use franken_ocr::tokenizer::music::TOKENIZER_FILENAMES;
use franken_ocr::{OcrEngine, TromrRecognitionOptionsV1};
use sha2::{Digest, Sha256};

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn identity_json(identity: ConsumedBytesIdentity) -> serde_json::Value {
    serde_json::json!({
        "byte_len": identity.byte_len,
        "sha256": identity.sha256_hex(),
        "blake3": identity.blake3_prefixed(),
    })
}

fn unique_stage_dir(model_dir: &Path) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after Unix epoch")
        .as_nanos();
    model_dir.parent().unwrap_or(model_dir).join(format!(
        "franken-ocr-immutable-swap-{}-{nonce}",
        std::process::id()
    ))
}

fn stage_link_or_copy(source: &Path, destination: &Path) {
    if std::fs::hard_link(source, destination).is_err() {
        std::fs::copy(source, destination).unwrap_or_else(|error| {
            panic!(
                "stage {} at {}: {error}",
                source.display(),
                destination.display()
            )
        });
    }
}

fn replace_staged_path(path: &Path, role: &str) -> Vec<u8> {
    let filename = path
        .file_name()
        .expect("staged artifact has a filename")
        .to_string_lossy();
    let pinned_path = path.with_file_name(format!("{filename}.pinned-a"));
    std::fs::rename(path, &pinned_path).unwrap_or_else(|error| {
        panic!(
            "preserve staged A {} at {}: {error}",
            path.display(),
            pinned_path.display()
        )
    });
    let replacement = format!("replacement-B:{role}").into_bytes();
    std::fs::write(path, &replacement)
        .unwrap_or_else(|error| panic!("replace staged {} with B: {error}", path.display()));
    replacement
}

#[test]
#[ignore = "requires FOCR_TROMR_DIR with the public TrOMR model artifact"]
fn model_backed_immutable_music_receipt_is_complete_and_path_free() {
    let model_dir = PathBuf::from(
        std::env::var_os("FOCR_TROMR_DIR")
            .expect("FOCR_TROMR_DIR must contain tromr.focrq and four tokenizer tables"),
    );
    let model_path = model_dir.join("tromr.focrq");
    let source_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/realscan_music/staves/spohr_no17_top.png");
    assert!(source_path.is_file(), "missing public real-scan fixture");
    assert!(model_path.is_file(), "missing TrOMR model artifact");

    let stage_dir = unique_stage_dir(&model_dir);
    std::fs::create_dir_all(&stage_dir).expect("create immutable path-swap stage");
    let staged_source_path = stage_dir.join("score.png");
    let staged_model_path = stage_dir.join("tromr.focrq");
    stage_link_or_copy(&source_path, &staged_source_path);
    stage_link_or_copy(&model_path, &staged_model_path);
    for filename in TOKENIZER_FILENAMES {
        stage_link_or_copy(&model_dir.join(filename), &stage_dir.join(filename));
    }

    let source_bytes = read(&source_path);
    let model_bytes = read(&model_path);
    let tokenizer_bytes = TOKENIZER_FILENAMES.map(|filename| read(&model_dir.join(filename)));
    let expectations = MusicInputExpectations {
        source_sha256: Some(sha256(&source_bytes)),
        model_sha256: Some(sha256(&model_bytes)),
        tokenizer_sha256: tokenizer_bytes.each_ref().map(|bytes| Some(sha256(bytes))),
    };
    let options = MusicInputOptions {
        recognition: TromrRecognitionOptionsV1::deterministic(),
        expectations,
        ..MusicInputOptions::default()
    };
    let bundle =
        ImmutableMusicInputBundle::open(&staged_source_path, &staged_model_path, options.clone())
            .expect("provider pins and validates every TrOMR input");
    let receipt_before_forward = bundle.provenance().clone();

    assert_eq!(
        receipt_before_forward.source.byte_len,
        source_bytes.len() as u64
    );
    assert_eq!(
        receipt_before_forward.model.byte_len,
        model_bytes.len() as u64
    );
    for (actual, expected) in receipt_before_forward
        .tokenizers
        .iter()
        .zip(&tokenizer_bytes)
    {
        assert_eq!(actual.byte_len, expected.len() as u64);
        assert_eq!(actual.sha256, sha256(expected));
    }

    let replacement_source = replace_staged_path(&staged_source_path, "source");
    let replacement_model = replace_staged_path(&staged_model_path, "model");
    let replacement_tokenizers = TOKENIZER_FILENAMES
        .map(|filename| replace_staged_path(&stage_dir.join(filename), filename));
    assert_ne!(
        receipt_before_forward.source.sha256,
        sha256(&replacement_source)
    );
    assert_ne!(
        receipt_before_forward.model.sha256,
        sha256(&replacement_model)
    );
    for (retained, replacement) in receipt_before_forward
        .tokenizers
        .iter()
        .zip(&replacement_tokenizers)
    {
        assert_ne!(retained.sha256, sha256(replacement));
    }

    let result = OcrEngine::new()
        .expect("embedded engine constructs")
        .recognize_immutable_music(bundle)
        .expect("embedded immutable TrOMR recognition succeeds");
    let result_provenance = result.provenance();
    let result_page_meta = result.page_meta();
    let result_musicxml = result.musicxml();
    assert_eq!(result_provenance, &receipt_before_forward);
    assert_eq!(
        result_page_meta.recognition_options,
        result_provenance.recognition_options
    );
    assert_eq!(
        result_page_meta.recognition_options_identity,
        result_provenance.recognition_options_identity
    );
    assert_eq!(result_page_meta.detected_staff_count, 1);
    assert_eq!(
        result_page_meta.staff_segmentation_disposition,
        franken_ocr::TromrStaffSegmentationDispositionV1::SingleStaffDetectedWholeImageRecognition
    );
    assert_eq!(result_page_meta.staff_evidence.len(), 1);
    assert_eq!(result_page_meta.fragments.len(), 1);
    assert!(result_page_meta.skips.is_empty());
    assert!(result_musicxml.contains("<score-partwise"));
    let xml_violations = franken_ocr::native_engine::tromr::validate_musicxml(result_musicxml);
    assert!(
        xml_violations.is_empty(),
        "provider MusicXML violations: {xml_violations:?}"
    );

    let replay = ImmutableMusicInputBundle::open(&source_path, &model_path, options)
        .expect("same exact inputs pin again");
    assert_eq!(replay.provenance(), result_provenance);

    let tokenizer_receipts: Vec<_> = result_provenance
        .tokenizers
        .iter()
        .copied()
        .map(identity_json)
        .collect();
    let attempts: Vec<_> = result_page_meta
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
    eprintln!(
        "{}",
        serde_json::json!({
            "schema": "franken_ocr.immutable_music_e2e.v1",
            "source_kind": result_provenance.source_kind.as_str(),
            "source": identity_json(result_provenance.source),
            "model": identity_json(result_provenance.model),
            "tokenizers": tokenizer_receipts,
            "bundle_sha256": result_provenance.bundle_sha256_hex(),
            "raster_sha256": result_provenance.raster_sha256_hex(),
            "options_sha256": result_provenance.options_sha256_hex(),
            "recognition_options": result_provenance.recognition_options,
            "recognition_options_identity": result_provenance.recognition_options_identity,
            "recognition_options_sha256": result_provenance.recognition_options_sha256_hex(),
            "execution_options": result_provenance.execution_options,
            "execution_options_identity": result_provenance.execution_options_identity,
            "execution_options_sha256": result_provenance.execution_options_sha256_hex(),
            "replay_sha256": result_provenance.replay_sha256_hex(),
            "selected_page": result_provenance.selected_page,
            "page_count": result_provenance.page_count,
            "raster_width": result_provenance.raster_width,
            "raster_height": result_provenance.raster_height,
            "detected_staff_count": result_page_meta.detected_staff_count,
            "staff_segmentation_disposition": result_page_meta.staff_segmentation_disposition,
            "attempts": attempts,
            "recognized_fragments": result_page_meta.fragments.len(),
            "skips": result_page_meta.skips.len(),
            "warnings": result_page_meta.warnings.len(),
            "musicxml_sha256": format!("{:x}", Sha256::digest(result_musicxml.as_bytes())),
            "path_replacement_proof": {
                "source_model_and_four_tokenizer_paths_replaced_before_forward": true,
                "forward_completed_from_provider_owned_a_bytes": true,
                "replacement_b_source_sha256": format!("{:x}", Sha256::digest(&replacement_source)),
                "replacement_b_model_sha256": format!("{:x}", Sha256::digest(&replacement_model)),
            },
        })
    );
}
