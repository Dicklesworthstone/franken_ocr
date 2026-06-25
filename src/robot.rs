//! The versioned NDJSON robot contract (plan §7.3).
//!
//! Robot mode is agent-first: one JSON object per line, every line carrying
//! `schema_version`. `robot schema` self-describes the event types so a consumer
//! can validate against a stable contract. This is the Phase-0 seed; event
//! payloads are finalized + contract-tested in Phase 5.

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
        "status": "skeleton — event payloads are finalized + contract-tested in Phase 5 (plan §7.3)"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_advertises_all_events() {
        let s = robot_schema();
        assert_eq!(s["schema_version"], ROBOT_SCHEMA_VERSION);
        assert_eq!(s["events"].as_array().unwrap().len(), EVENT_KINDS.len());
    }
}
