//! The versioned NDJSON robot contract (plan §7.3).
//!
//! Robot mode is agent-first: one JSON object per line, every line carrying
//! `schema_version`. `robot schema` self-describes the event types so a consumer
//! can validate against a stable contract. This is the Phase-0 seed; event
//! payloads are finalized + contract-tested in Phase 5.

use crate::{
    FOCR_MODEL_LICENSE_NOTICE,
    error::{EXIT_CODE_TABLE, FocrError},
};
use serde_json::{Value, json};

/// The robot event-stream schema version. Bumped on any breaking contract change.
pub const ROBOT_SCHEMA_VERSION: u32 = 1;

/// The event kinds emitted on the NDJSON stream (plan §7.3).
pub const EVENT_KINDS: &[&str] = &[
    "run_start",
    "stage",
    "page",
    "staff",
    "music_warning",
    "run_complete",
    "run_error",
];

/// A machine-readable, self-describing schema for the robot event stream.
pub fn robot_schema() -> Value {
    json!({
        "schema_version": ROBOT_SCHEMA_VERSION,
        "events": EVENT_KINDS,
        "exit_codes": EXIT_CODE_TABLE,
        "model_license_notice": FOCR_MODEL_LICENSE_NOTICE,
        "status": "skeleton — run_start/run_complete (carries `markdown`)/run_error are wired; the streaming stage/page event payloads are finalized + contract-tested in Phase 5 (plan §7.3)"
    })
}

/// Build the robot-mode `run_start` event for the current command.
///
/// The full Phase-1+ pipeline will attach run IDs, model identity, and resolved
/// options. Phase 0 still emits the event so the stream shape is stable before a
/// terminal `run_error`.
pub fn run_start_event(command: &str) -> Value {
    json!({
        "schema_version": ROBOT_SCHEMA_VERSION,
        "event": "run_start",
        "command": command,
    })
}

/// Build the robot-mode terminal `run_complete` event carrying the recognized
/// document markdown.
///
/// This is the SUCCESS terminal of the OCR event stream: a machine consumer reads
/// the recognized text from `markdown` here (the human and `--json` modes print it
/// directly instead). The `run_complete` kind is already advertised by
/// [`robot_schema`], so finalizing its payload does not change the advertised
/// event set and [`ROBOT_SCHEMA_VERSION`] is unchanged.
pub fn run_complete_event(markdown: &str) -> Value {
    json!({
        "schema_version": ROBOT_SCHEMA_VERSION,
        "event": "run_complete",
        "markdown": markdown,
    })
}

/// Build the robot-mode `run_error` event from the canonical [`FocrError`].
///
/// This is the only place that shapes a `run_error` payload. The numeric `code`
/// is read directly from [`FocrError::exit_code`] so robot consumers and process
/// supervisors observe the same stable contract.
pub fn run_error_event(err: &FocrError) -> Value {
    json!({
        "schema_version": ROBOT_SCHEMA_VERSION,
        "event": "run_error",
        "error_kind": err.kind(),
        "code": err.exit_code(),
        "message": err.to_string(),
    })
}

/// Build a robot-mode `page` event recording that one document page was SKIPPED
/// mid-run (e.g. an undecodable PDF page the resilient document loop dropped to
/// keep the rest of the document).
///
/// This is the machine-stream counterpart of the human stderr warning: without
/// it a robot consumer would receive a short document with no signal that pages
/// were dropped. `page` is 1-based; `error_kind` mirrors [`FocrError::kind`] so
/// the skip reason is machine-classifiable, and `message` carries the human
/// detail. `page` is an advertised [`EVENT_KINDS`] kind, so this finalizes a slice
/// of its payload without changing the advertised event set (schema stays v1).
/// Build a robot-mode `page` event for one DECODED page of a `--multi-page`
/// cross-page pass (bd-2z0y): emitted as the page's `<PAGE>` boundary is
/// crossed in the token stream, so a machine consumer sees per-page progress
/// (and the raw body) during a long single-pass decode instead of one silent
/// wait for `run_complete`. `page` is 1-based in the MODEL's emission order;
/// `text` is the trimmed raw body (the polished markdown arrives in the
/// terminal `run_complete`). Additive payload on the advertised `page` kind —
/// schema stays v1.
#[must_use]
pub fn page_decoded_event(page: usize, text: &str) -> Value {
    json!({
        "schema_version": ROBOT_SCHEMA_VERSION,
        "event": "page",
        "status": "decoded",
        "page": page,
        "chars": text.chars().count(),
        "text": text,
    })
}

pub fn page_skipped_event(page: usize, err: &FocrError) -> Value {
    json!({
        "schema_version": ROBOT_SCHEMA_VERSION,
        "event": "page",
        "status": "skipped",
        "page": page,
        "error_kind": err.kind(),
        "message": err.to_string(),
    })
}

/// One `staff` event from a TrOMR full-page music run (bd-av64.2).
///
/// `total` is the number of inference attempts emitted for this page. The
/// detector count and route disposition are separate because zero detections
/// still produce one whole-image fallback attempt. This row either recognized into a local
/// semantic stream (`status: "ok"`) or was skipped with a reason. The real
/// page bbox stays distinct from bd-av64.16's padded inference canvas. This
/// evidence is row-local and makes no system/part claim. Additive to the v1
/// event set, matching the `page` precedent from bd-fck1.
pub fn staff_event(
    index: usize,
    total: usize,
    detected_staff_count: usize,
    staff_segmentation_disposition:
        crate::native_engine::tromr::TromrStaffSegmentationDispositionV1,
    geometry: crate::preprocess::staff_detect::StaffCropGeometry,
    outcome: crate::native_engine::tromr::StaffInferenceOutcome,
    reason: Option<&str>,
) -> Value {
    debug_assert!(staff_segmentation_disposition.is_consistent_with(detected_staff_count));
    let bbox = geometry.source_bbox;
    let status = match outcome {
        crate::native_engine::tromr::StaffInferenceOutcome::Recognized => "ok",
        crate::native_engine::tromr::StaffInferenceOutcome::Skipped => "skipped",
    };
    let mut event = serde_json::json!({
        "schema_version": ROBOT_SCHEMA_VERSION,
        "event": "staff",
        "staff": index + 1,
        "total": total,
        "detected_staff_count": detected_staff_count,
        "staff_segmentation_disposition": staff_segmentation_disposition,
        "bbox": [bbox.0, bbox.1, bbox.2, bbox.3],
        "source_bbox": [bbox.0, bbox.1, bbox.2, bbox.3],
        "inference_canvas": {
            "width": geometry.canvas_width,
            "height": geometry.canvas_height,
        },
        "padding": {
            "top": geometry.padding.top,
            "right": geometry.padding.right,
            "bottom": geometry.padding.bottom,
            "left": geometry.padding.left,
        },
        "outcome": outcome.as_str(),
        "status": status,
    });
    if let Some(reason) = reason {
        event["reason"] = serde_json::json!(reason);
    }
    event
}

/// One annotate-only musical-sanity observation from a TrOMR music run
/// (bd-av64.5): `kind` is machine-stable (`overfull_bar` | `underfull_bar`
/// | `impossible_duration` | `key_mismatch`), `measure` 0 means the
/// warning is staff-level. Additive to the v1 event set like `staff`.
pub fn music_warning_event(kind: &str, part: usize, measure: usize, detail: &str) -> Value {
    serde_json::json!({
        "schema_version": ROBOT_SCHEMA_VERSION,
        "event": "music_warning",
        "kind": kind,
        "part": part,
        "measure": measure,
        "detail": detail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// bd-av64.2: the `staff` event shape — 1-based staff number, total,
    /// bbox array, status, and a reason only when skipped.
    #[test]
    fn staff_event_shapes_ok_and_skipped() {
        let unpadded =
            crate::preprocess::staff_detect::StaffCropGeometry::unpadded((0, 292, 1168, 115));
        let ok = staff_event(
            0,
            5,
            5,
            crate::native_engine::tromr::TromrStaffSegmentationDispositionV1::MultipleStavesDetectedPerCropRecognition,
            unpadded,
            crate::native_engine::tromr::StaffInferenceOutcome::Recognized,
            None,
        );
        assert_eq!(ok["event"], "staff");
        assert_eq!(ok["staff"], 1);
        assert_eq!(ok["total"], 5);
        assert_eq!(ok["detected_staff_count"], 5);
        assert_eq!(
            ok["staff_segmentation_disposition"],
            "multiple_staves_detected_per_crop_recognition"
        );
        assert_eq!(ok["bbox"], serde_json::json!([0, 292, 1168, 115]));
        assert_eq!(ok["source_bbox"], ok["bbox"]);
        assert_eq!(ok["inference_canvas"]["width"], 1168);
        assert_eq!(ok["inference_canvas"]["height"], 115);
        assert_eq!(ok["padding"]["top"], 0);
        assert_eq!(ok["outcome"], "recognized");
        assert_eq!(ok["status"], "ok");
        assert!(ok.get("reason").is_none());
        let fallback = staff_event(
            0,
            1,
            0,
            crate::native_engine::tromr::TromrStaffSegmentationDispositionV1::NoStaffDetectedWholeImageFallback,
            unpadded,
            crate::native_engine::tromr::StaffInferenceOutcome::Recognized,
            None,
        );
        assert_eq!(fallback["staff"], 1);
        assert_eq!(fallback["total"], 1);
        assert_eq!(fallback["detected_staff_count"], 0);
        assert_eq!(
            fallback["staff_segmentation_disposition"],
            "no_staff_detected_whole_image_fallback"
        );
        let skip = staff_event(
            3,
            5,
            5,
            crate::native_engine::tromr::TromrStaffSegmentationDispositionV1::MultipleStavesDetectedPerCropRecognition,
            crate::preprocess::staff_detect::StaffCropGeometry {
                source_bbox: (0, 663, 1168, 100),
                canvas_width: 1168,
                canvas_height: 117,
                padding: crate::preprocess::staff_detect::StaffPadding {
                    top: 8,
                    right: 0,
                    bottom: 9,
                    left: 0,
                },
            },
            crate::native_engine::tromr::StaffInferenceOutcome::Skipped,
            Some("decoder error"),
        );
        assert_eq!(skip["staff"], 4);
        assert_eq!(skip["bbox"], serde_json::json!([0, 663, 1168, 100]));
        assert_eq!(skip["inference_canvas"]["height"], 117);
        assert_eq!(skip["padding"]["top"], 8);
        assert_eq!(skip["padding"]["bottom"], 9);
        assert_eq!(skip["outcome"], "skipped");
        assert_eq!(skip["reason"], "decoder error");
        assert!(EVENT_KINDS.contains(&"staff"), "schema advertises staff");
    }

    #[test]
    fn schema_advertises_all_events() {
        let s = robot_schema();
        assert_eq!(s["schema_version"], ROBOT_SCHEMA_VERSION);
        assert_eq!(
            s["events"].as_array().map(Vec::len),
            Some(EVENT_KINDS.len())
        );
        assert_eq!(
            s["exit_codes"].as_array().map(Vec::len),
            Some(EXIT_CODE_TABLE.len())
        );
        assert_eq!(s["model_license_notice"], FOCR_MODEL_LICENSE_NOTICE);
    }

    #[test]
    fn run_error_event_uses_error_exit_code_for_every_variant() {
        let cases = [
            FocrError::Usage("bad flag".into()),
            FocrError::ModelNotFound("missing".into()),
            FocrError::InputDecode("bad image".into()),
            FocrError::Timeout("stage".into()),
            FocrError::Cancelled,
            FocrError::FormatMismatch("bad header".into()),
            FocrError::NotImplemented("phase gap".into()),
            FocrError::Other(anyhow::anyhow!("misc")),
        ];

        for err in &cases {
            let event = run_error_event(err);
            eprintln!(
                "{}",
                serde_json::json!({
                    "suite": "robot",
                    "test": "run_error_event_uses_error_exit_code_for_every_variant",
                    "variant": err.kind(),
                    "exit_code": err.exit_code(),
                    "robot_code": event["code"],
                })
            );
            assert_eq!(event["schema_version"], ROBOT_SCHEMA_VERSION);
            assert_eq!(event["event"], "run_error");
            assert_eq!(event["error_kind"], err.kind());
            assert_eq!(event["code"], err.exit_code());
            assert_eq!(event["message"], err.to_string());
        }
    }

    #[test]
    fn run_start_event_carries_schema_and_command() {
        let event = run_start_event("ocr");
        assert_eq!(event["schema_version"], ROBOT_SCHEMA_VERSION);
        assert_eq!(event["event"], "run_start");
        assert_eq!(event["command"], "ocr");
    }

    #[test]
    fn run_complete_event_carries_recognized_markdown() {
        let event = run_complete_event("# Title\n\nbody text");
        assert_eq!(event["schema_version"], ROBOT_SCHEMA_VERSION);
        assert_eq!(event["event"], "run_complete");
        assert_eq!(event["markdown"], "# Title\n\nbody text");
        // The terminal success event is an advertised kind (no schema bump).
        assert!(EVENT_KINDS.contains(&"run_complete"));
    }

    #[test]
    fn page_decoded_event_carries_the_streamed_body() {
        let event = page_decoded_event(2, "# Chapter\nBody.");
        assert_eq!(event["schema_version"], ROBOT_SCHEMA_VERSION);
        assert_eq!(event["event"], "page");
        assert_eq!(event["status"], "decoded");
        assert_eq!(event["page"], 2);
        assert_eq!(event["chars"], 15);
        assert_eq!(event["text"], "# Chapter\nBody.");
        // `page` is an advertised kind, so no schema bump is implied.
        assert!(EVENT_KINDS.contains(&"page"));
    }

    #[test]
    fn page_skipped_event_classifies_the_skip() {
        let event = page_skipped_event(7, &FocrError::InputDecode("bad xobject".into()));
        assert_eq!(event["schema_version"], ROBOT_SCHEMA_VERSION);
        assert_eq!(event["event"], "page");
        assert_eq!(event["status"], "skipped");
        assert_eq!(event["page"], 7);
        assert_eq!(event["error_kind"], "input_decode");
        assert_eq!(event["message"], "input decode error: bad xobject");
        // `page` is an advertised kind, so no schema bump is implied.
        assert!(EVENT_KINDS.contains(&"page"));
    }
}
