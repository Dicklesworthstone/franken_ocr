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

/// The environment variables an agent can steer runs with, self-described so a
/// consumer never has to scrape the README. `(name, summary)` — additive rows
/// keep the schema at v1.
pub const ENVIRONMENT_VARIABLES: &[(&str, &str)] = &[
    (
        "FOCR_MODEL_PATH",
        "override the model artifact path (a .focrq blob or safetensors directory)",
    ),
    (
        "FOCR_MODEL_DIR",
        "extra model search roots before the default cache",
    ),
    (
        "FOCR_QUANT",
        "pick the quant-suffixed artifact during cache resolution",
    ),
    (
        "FOCR_THREADS",
        "kernel thread budget (default: physical cores)",
    ),
    (
        "FOCR_MAX_NEW_TOKENS",
        "cap generated tokens (an explicit --max-length outranks it)",
    ),
    (
        "FOCR_STAGE_BUDGET_FORWARD_MS",
        "forward stage budget in ms (default 600000; 0 or `unlimited` disables the timeout)",
    ),
    (
        "FOCR_TIMING",
        "emit nested stage-timing rows on stderr (disables the resident daemon and progress bar)",
    ),
    (
        "FOCR_NO_PROGRESS",
        "disable the interactive stderr progress bar",
    ),
    (
        "FOCR_MMAP",
        "opt trusted immutable artifacts into read-only mmap",
    ),
    (
        "FOCR_FORCE_ARCH",
        "force a SIMD tier (sdot/smmla/scalar/avx2/avxvnni/avx512vnni)",
    ),
    ("FOCR_RUN_STORE", "override the run-history database path"),
    (
        "FOCR_NO_RESIDENT",
        "1/true disables the resident warm-model daemon (flag analog: --no-resident)",
    ),
    (
        "FOCR_RESIDENT_IDLE_SECS",
        "resident daemon idle unload period (default 600)",
    ),
    (
        "FOCR_RESIDENT_DIR",
        "override the resident daemon state-file directory",
    ),
    (
        "FOCR_RESIDENT_LOG",
        "append resident daemon diagnostics to this file",
    ),
    (
        "FOCR_RESIDENT_SPAWN_WAIT_SECS",
        "how long a client waits for a freshly spawned daemon to accept (default 30)",
    ),
    (
        "FOCR_RESIDENT_CLIENT_TIMEOUT_SECS",
        "client-side reply timeout for a resident request (default 600)",
    ),
    (
        "FOCR_TEST_MAX_POSITIONS",
        "TEST-ONLY: lower the position budget prompt and output share (1..=32768; a larger \
         value exits 2) so tests reach the multi-page clamp on small documents",
    ),
    (
        "FOCR_SRSWA_ENTMAX",
        "unlimited-ocr-2 router entmax domain: `topm` (default, HF parity) or `all` (SGLang); \
         another value exits 2",
    ),
    (
        "FOCR_SRSWA_RETIRE",
        "0/off disables unlimited-ocr-2 KV block retirement (a kill switch; output is identical)",
    ),
];

/// Required and optional fields per event kind, so a consumer can validate an
/// event and not only its kind (U2-X2). A change that adds a field extends this
/// table in the same change; `tests/cli_robot_golden.rs` validates every event
/// it sees against it. `stage` has no emitter yet.
pub fn event_fields() -> Value {
    json!({
        "run_start": {
            "required": ["schema_version", "event", "command"],
            "optional": [],
        },
        "stage": { "reserved": true },
        "page": {
            "required": ["schema_version", "event", "status", "page"],
            // The source-page fields are on Unlimited-OCR-2 runs (U2-E3);
            // `tokens`, `finish` and `elapsed_ms` on its decoded pages.
            "optional": [
                "chars", "text", "error_kind", "message", "source_page", "half", "model_id",
                "tokens", "finish", "elapsed_ms",
            ],
        },
        "staff": {
            "required": ["schema_version", "event", "staff", "total", "bbox", "status"],
            "optional": ["reason"],
        },
        "music_warning": {
            "required": ["schema_version", "event", "kind", "part", "measure", "detail"],
            "optional": [],
        },
        "run_complete": {
            "required": ["schema_version", "event", "markdown"],
            // `finish_reason` is on every run whose model records a decode
            // report (every Unlimited-OCR run); `warnings` only when non-empty;
            // `model_id` and `source_sha256` (null for a raw checkpoint)
            // whenever the model that ran is known.
            "optional": [
                "finish_reason", "warnings", "low_yield", "yield_chars", "input_megapixels",
                "model_id", "source_sha256",
            ],
        },
        "run_error": {
            "required": ["schema_version", "event", "error_kind", "code", "message", "recovery"],
            // A partial-output run_error (completed pages written, then a
            // nonzero exit) adds these; their producers are the v2 beads.
            "optional": ["finish_reason", "pages_out", "warnings", "model_id", "source_sha256"],
        },
    })
}

/// A machine-readable, self-describing schema for the robot event stream.
pub fn robot_schema() -> Value {
    let environment: Vec<Value> = ENVIRONMENT_VARIABLES
        .iter()
        .map(|(name, summary)| json!({ "name": name, "summary": summary }))
        .collect();
    json!({
        "schema_version": ROBOT_SCHEMA_VERSION,
        "events": EVENT_KINDS,
        "event_fields": event_fields(),
        "exit_codes": EXIT_CODE_TABLE,
        "model_license_notice": FOCR_MODEL_LICENSE_NOTICE,
        "environment_variables": environment,
        // Every machine-stable `warnings[].kind` (U2-X2, U2-A7): additive.
        "warning_kinds": crate::native_engine::RUN_WARNING_KINDS,
        // How an agent orients in one round-trip each (the franken_whisper
        // `agent_discovery` pattern): additive keys, schema stays v1.
        "agent_discovery": {
            "triage": "focr robot triage",
            "health": "focr robot health",
            "backends": "focr robot backends",
            "selftest": "focr robot selftest",
            "models": "focr models --json",
            "doctor": "focr doctor --json",
        },
        "stdout_contract": "robot mode emits one JSON object per line on stdout and NOTHING else there; diagnostics go to stderr",
        "status": "live — run_start/run_complete (carries `markdown`, `finish_reason`, `model_id` and `source_sha256`, plus `warnings` when a decode reported any)/run_error plus the streaming page, staff, and music_warning events are wired and contract-tested against `event_fields`; `run_error` carries `recovery`, plus `finish_reason`, `pages_out`, `warnings`, `model_id` and `source_sha256` when a document stopped early (exit 5, or 6 when cancelled; no `run_complete`)"
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

/// [`run_complete_event`] with the low-yield assessment attached when one was
/// flagged (GH #15): `low_yield: true` plus `yield_chars` and
/// `input_megapixels`, so a machine consumer can detect a page-sized input
/// that "succeeded" with almost no text. The fields are ADDITIVE on the
/// already-advertised `run_complete` kind and absent entirely on healthy
/// runs, so existing consumers and goldens are unaffected and
/// [`ROBOT_SCHEMA_VERSION`] is unchanged.
///
/// U2-X2 adds `finish_reason` (`stop` | `length` | `budget`) whenever the decode
/// recorded one, and `warnings` (`{kind, message, recovery, ...}`) only when
/// non-empty, so a healthy run differs from before only by `finish_reason`.
/// U2-E1 adds `model_id` and `source_sha256` whenever the model that ran is
/// known.
pub fn run_complete_event_assessed(
    markdown: &str,
    low_yield: Option<&crate::tall::LowYield>,
    finish_reason: Option<crate::native_engine::FinishReason>,
    warnings: &[crate::native_engine::RunWarning],
    model: Option<&crate::native_engine::ModelIdentity>,
) -> Value {
    let mut event = run_complete_event(markdown);
    if let Some(model) = model {
        model.merge_into(&mut event);
    }
    if let Some(assessment) = low_yield {
        event["low_yield"] = json!(true);
        event["yield_chars"] = json!(assessment.yield_chars);
        event["input_megapixels"] = json!(assessment.input_megapixels);
    }
    if let Some(reason) = finish_reason {
        event["finish_reason"] = json!(reason.as_str());
    }
    if !warnings.is_empty() {
        event["warnings"] = warnings
            .iter()
            .map(crate::native_engine::RunWarning::to_json)
            .collect();
    }
    event
}

/// Build the robot-mode `run_error` event from the canonical [`FocrError`].
///
/// This is the only place that shapes a `run_error` payload. The numeric `code`
/// is read directly from [`FocrError::exit_code`] so robot consumers and process
/// supervisors observe the same stable contract.
pub fn run_error_event(err: &FocrError) -> Value {
    let mut event = json!({
        "schema_version": ROBOT_SCHEMA_VERSION,
        "event": "run_error",
        "error_kind": err.kind(),
        "code": err.exit_code(),
        "message": err.to_string(),
        // One actionable next step, machine-consumable without parsing prose
        // (additive field; schema stays v1).
        "recovery": err.remediation(),
    });
    // A partial-output run (U2-B6): the pages it wrote and why it stopped.
    if let FocrError::Partial(partial) = err {
        event["finish_reason"] = partial.finish_reason.into();
        event["pages_out"] = partial.pages_out.into();
        if !partial.warnings.is_empty() {
            event["warnings"] = partial.warnings.clone().into();
        }
        event["model_id"] = partial.model_id.into();
        event["source_sha256"] = partial.source_sha256.clone().into();
    }
    event
}

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

/// [`page_decoded_event`] for a page of a model that splits pages exactly
/// (Unlimited-OCR-2, U2-E3): `page` stays the emission index, and the event
/// adds the page's `source_page`, `half` when split, `model_id`, and its
/// `tokens`, `finish` and `elapsed_ms`. v1 bodies may merge pages, so v1
/// events name no source page.
#[must_use]
pub fn document_page_decoded_event(page: &crate::PageResult, body: &str) -> Value {
    let mut event = page_decoded_event(page.run_index, body);
    event["source_page"] = page.source_page.into();
    if let Some(half) = page.half {
        event["half"] = half.as_str().into();
    }
    event["model_id"] = page.model_id.into();
    if let Some(tokens) = page.tokens {
        event["tokens"] = tokens.into();
    }
    if let Some(finish) = page.finish {
        event["finish"] = finish.as_str().into();
    }
    if let Some(elapsed_ms) = page.elapsed_ms {
        event["elapsed_ms"] = elapsed_ms.into();
    }
    event
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

/// [`page_skipped_event`] on a run of a model that splits pages exactly
/// (Unlimited-OCR-2, U2-E3): adds `source_page`, the same 1-based source page
/// as `page`, so decoded and skipped events share one key, and `model_id`.
#[must_use]
pub fn document_page_skipped_event(page: usize, err: &FocrError, model_id: &str) -> Value {
    let mut event = page_skipped_event(page, err);
    event["source_page"] = page.into();
    event["model_id"] = model_id.into();
    event
}

/// One `staff` event from a TrOMR full-page music run (bd-av64.2): the
/// detector found `total` staves; this staff either recognized into the
/// MusicXML (`status: "ok"`) or was skipped with a reason. Additive to the
/// v1 event set (a consumer ignoring unknown kinds is unaffected), matching
/// the `page` precedent from bd-fck1.
pub fn staff_event(
    index: usize,
    total: usize,
    bbox: (usize, usize, usize, usize),
    status: &str,
    reason: Option<&str>,
) -> Value {
    let mut event = serde_json::json!({
        "schema_version": ROBOT_SCHEMA_VERSION,
        "event": "staff",
        "staff": index + 1,
        "total": total,
        "bbox": [bbox.0, bbox.1, bbox.2, bbox.3],
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
        let ok = staff_event(0, 5, (0, 292, 1168, 115), "ok", None);
        assert_eq!(ok["event"], "staff");
        assert_eq!(ok["staff"], 1);
        assert_eq!(ok["total"], 5);
        assert_eq!(ok["bbox"], serde_json::json!([0, 292, 1168, 115]));
        assert_eq!(ok["status"], "ok");
        assert!(ok.get("reason").is_none());
        let skip = staff_event(3, 5, (0, 663, 1168, 114), "skipped", Some("1280 clamp"));
        assert_eq!(skip["staff"], 4);
        assert_eq!(skip["reason"], "1280 clamp");
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
            for field in ["finish_reason", "pages_out", "warnings"] {
                assert!(event.get(field).is_none(), "{field} only on partial output");
            }
        }
    }

    /// U2-B6: a partial-output `run_error` carries exit 5 and its additive
    /// fields, each declared optional in `event_fields`.
    #[test]
    fn partial_output_run_error_carries_its_pages_and_warnings() {
        let warning = json!({"kind": "v2_length", "message": "m", "recovery": "r",
                             "expected": 3, "got": 2});
        let err = FocrError::Partial(Box::new(crate::error::PartialOutput {
            finish_reason: "length",
            pages_out: 2,
            warnings: vec![warning.clone()],
            model_id: "unlimited-ocr-2",
            source_sha256: None,
            message: "stopped in page 2 of 3".into(),
        }));
        let event = run_error_event(&err);
        assert_eq!(event["code"], 5);
        assert_eq!(event["error_kind"], "timeout");
        assert_eq!(event["finish_reason"], "length");
        assert_eq!(event["pages_out"], 2);
        assert_eq!(event["warnings"], json!([warning]));
        assert_eq!(event["model_id"], "unlimited-ocr-2");
        assert_eq!(event["source_sha256"], serde_json::Value::Null);
        assert!(
            event["message"]
                .as_str()
                .is_some_and(|m| m.contains("stopped in page 2 of 3")),
            "{event}"
        );
        let fields = event_fields();
        let spec = &fields["run_error"];
        for key in event.as_object().expect("an object").keys() {
            let declared = [&spec["required"], &spec["optional"]]
                .iter()
                .filter_map(|list| list.as_array())
                .flatten()
                .any(|name| name == key);
            assert!(declared, "{key} is declared for run_error");
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

    /// Every key of `event` is declared for `page`, and every required key
    /// is present.
    fn assert_page_fields(event: &Value) {
        let fields = event_fields();
        let spec = &fields["page"];
        let declared = |list: &Value| -> Vec<String> {
            list.as_array()
                .expect("a field list")
                .iter()
                .map(|k| k.as_str().expect("a field name").to_string())
                .collect()
        };
        let (required, optional) = (declared(&spec["required"]), declared(&spec["optional"]));
        let keys = event.as_object().expect("an object");
        for key in keys.keys() {
            assert!(
                required.contains(key) || optional.contains(key),
                "{key} is declared for page"
            );
        }
        for key in &required {
            assert!(keys.contains_key(key), "{key} is present: {event}");
        }
    }

    /// U2-E3: an Unlimited-OCR-2 run's `page` events name the source page.
    /// `page` keeps its meaning per status (the emission index when decoded,
    /// the source page when skipped), and `source_page` is on both.
    #[test]
    fn document_page_events_name_the_source_page() {
        let page = crate::PageResult {
            run_index: 2,
            source_page: 3,
            half: Some(crate::Half::Left),
            window: None,
            text: "markdown".into(),
            layout: Vec::new(),
            figures: Vec::new(),
            tokens: Some(41),
            finish: Some(crate::PageFinish::PageCap),
            elapsed_ms: Some(950),
            model_id: "unlimited-ocr-2",
        };
        let decoded = document_page_decoded_event(&page, "raw body");
        assert_eq!(
            decoded,
            json!({
                "schema_version": ROBOT_SCHEMA_VERSION, "event": "page", "status": "decoded",
                "page": 2, "chars": 8, "text": "raw body", "source_page": 3, "half": "left",
                "model_id": "unlimited-ocr-2", "tokens": 41, "finish": "page_cap",
                "elapsed_ms": 950,
            })
        );
        assert_page_fields(&decoded);
        let whole =
            document_page_decoded_event(&crate::PageResult { half: None, ..page }, "raw body");
        assert!(whole.get("half").is_none(), "absent on a whole page");

        let skipped = document_page_skipped_event(
            4,
            &FocrError::InputDecode("bad xobject".into()),
            "unlimited-ocr-2",
        );
        assert_eq!(
            (&skipped["page"], &skipped["source_page"]),
            (&json!(4), &json!(4))
        );
        assert_eq!(skipped["model_id"], "unlimited-ocr-2");
        assert_page_fields(&skipped);
        // v1 events keep their keys.
        assert_page_fields(&page_decoded_event(1, "body"));
        assert!(page_decoded_event(1, "body").get("source_page").is_none());
    }
}
