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
pub const EVENT_KINDS: &[&str] = &["run_start", "stage", "page", "run_complete", "run_error"];

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
