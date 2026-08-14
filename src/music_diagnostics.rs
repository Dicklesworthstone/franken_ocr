//! Runtime-only diagnostics for immutable music input preparation and TrOMR.
//!
//! These measurements are intentionally noncanonical. They never participate
//! in provenance, replay identities, cache keys, or semantic equality.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::FocrError;

pub const MUSIC_DIAGNOSTICS_SCHEMA_VERSION: u32 = 1;
pub const MUSIC_DIAGNOSTICS_TIMING_CONTRACT: &str = "monotonic_process_local_noncanonical_v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicRunOutcomeV1 {
    Success,
    Timeout,
    Cancelled,
    Error,
}

impl MusicRunOutcomeV1 {
    pub(crate) fn from_result<T>(result: &Result<T, FocrError>) -> Self {
        match result {
            Ok(_) => Self::Success,
            Err(FocrError::Timeout(_)) => Self::Timeout,
            Err(FocrError::Cancelled) => Self::Cancelled,
            Err(_) => Self::Error,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ProcessResourceDiagnosticsV1 {
    /// `linux_procfs_process_v1` when sampled, otherwise `unavailable_v1`.
    pub capability: String,
    /// Linux process user+system clock ticks consumed during the measured span.
    pub cpu_ticks_delta: Option<u64>,
    /// Process high-water resident set at the end of the measured span.
    pub peak_rss_kib: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MusicInputPreparationDiagnosticsV1 {
    pub schema_version: u32,
    pub timing_contract: String,
    pub outcome: MusicRunOutcomeV1,
    pub error_kind: Option<String>,
    /// Path-free runtime context. The returned typed error remains authoritative.
    pub detail: Option<String>,
    pub source_read_wall_micros: u64,
    pub model_read_wall_micros: u64,
    /// Rhythm, pitch, lift, and note tokenizer tables, in provider order.
    pub tokenizer_read_wall_micros: [u64; 4],
    pub pdf_parse_wall_micros: u64,
    pub page_raster_wall_micros: u64,
    pub image_decode_wall_micros: u64,
    pub tokenizer_parse_wall_micros: u64,
    pub model_parse_wall_micros: u64,
    pub total_wall_micros: u64,
    pub resources: ProcessResourceDiagnosticsV1,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TromrForwardAttemptDiagnosticsV1 {
    pub attempt_index: u32,
    pub detection_index: Option<usize>,
    pub segment_index: Option<usize>,
    pub preprocess_wall_micros: u64,
    pub encode_wall_micros: u64,
    pub decode_wall_micros: u64,
    pub semantic_assembly_wall_micros: u64,
    pub total_wall_micros: u64,
    pub outcome: MusicRunOutcomeV1,
    pub error_kind: Option<String>,
    pub detail: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TromrExecutionDiagnosticsV1 {
    pub schema_version: u32,
    pub timing_contract: String,
    pub execution_options_identity: String,
    pub outcome: MusicRunOutcomeV1,
    pub error_kind: Option<String>,
    /// Path-free stage context; never a copy of an input-path-bearing error.
    pub detail: Option<String>,
    pub blocking_queue_wait_micros: u64,
    pub forward_admission_wait_micros: u64,
    pub staff_detection_wall_micros: u64,
    pub attempts: Vec<TromrForwardAttemptDiagnosticsV1>,
    pub page_assembly_wall_micros: u64,
    pub musicxml_assembly_wall_micros: u64,
    pub publication_wall_micros: u64,
    pub total_wall_micros: u64,
    pub attempts_started: u32,
    pub earned_allowance_ms: u64,
    pub resources: ProcessResourceDiagnosticsV1,
}

impl TromrExecutionDiagnosticsV1 {
    pub(crate) fn unavailable(execution_options_identity: String, error: &FocrError) -> Self {
        let outcome = match error {
            FocrError::Timeout(_) => MusicRunOutcomeV1::Timeout,
            FocrError::Cancelled => MusicRunOutcomeV1::Cancelled,
            _ => MusicRunOutcomeV1::Error,
        };
        Self {
            schema_version: MUSIC_DIAGNOSTICS_SCHEMA_VERSION,
            timing_contract: MUSIC_DIAGNOSTICS_TIMING_CONTRACT.to_owned(),
            execution_options_identity,
            outcome,
            error_kind: Some(error.kind().to_owned()),
            detail: Some(path_free_error_detail(error)),
            blocking_queue_wait_micros: 0,
            forward_admission_wait_micros: 0,
            staff_detection_wall_micros: 0,
            attempts: Vec::new(),
            page_assembly_wall_micros: 0,
            musicxml_assembly_wall_micros: 0,
            publication_wall_micros: 0,
            total_wall_micros: 0,
            attempts_started: 0,
            earned_allowance_ms: 0,
            resources: ProcessResourceDiagnosticsV1 {
                capability: "unavailable_v1".to_owned(),
                cpu_ticks_delta: None,
                peak_rss_kib: None,
            },
        }
    }
}

#[derive(Clone, Copy)]
struct ResourceSnapshot {
    cpu_ticks: Option<u64>,
}

impl ResourceSnapshot {
    fn capture() -> Self {
        Self {
            cpu_ticks: process_cpu_ticks(),
        }
    }

    fn finish(self) -> ProcessResourceDiagnosticsV1 {
        let after = process_cpu_ticks();
        let cpu_ticks_delta = after
            .zip(self.cpu_ticks)
            .and_then(|(after, before)| after.checked_sub(before));
        let peak_rss_kib = process_peak_rss_kib();
        ProcessResourceDiagnosticsV1 {
            capability: if cpu_ticks_delta.is_some() || peak_rss_kib.is_some() {
                "linux_procfs_process_v1".to_owned()
            } else {
                "unavailable_v1".to_owned()
            },
            cpu_ticks_delta,
            peak_rss_kib,
        }
    }
}

pub(crate) struct MusicInputPreparationDiagnosticsBuilder {
    started: Instant,
    resources: ResourceSnapshot,
    source_read_wall_micros: u64,
    model_read_wall_micros: u64,
    tokenizer_read_wall_micros: [u64; 4],
    pdf_parse_wall_micros: u64,
    page_raster_wall_micros: u64,
    image_decode_wall_micros: u64,
    tokenizer_parse_wall_micros: u64,
    model_parse_wall_micros: u64,
}

impl MusicInputPreparationDiagnosticsBuilder {
    pub(crate) fn new() -> Self {
        Self {
            started: Instant::now(),
            resources: ResourceSnapshot::capture(),
            source_read_wall_micros: 0,
            model_read_wall_micros: 0,
            tokenizer_read_wall_micros: [0; 4],
            pdf_parse_wall_micros: 0,
            page_raster_wall_micros: 0,
            image_decode_wall_micros: 0,
            tokenizer_parse_wall_micros: 0,
            model_parse_wall_micros: 0,
        }
    }

    pub(crate) fn record_source_read(&mut self, elapsed: Duration) {
        self.source_read_wall_micros = duration_micros(elapsed);
    }

    pub(crate) fn record_model_read(&mut self, elapsed: Duration) {
        self.model_read_wall_micros = duration_micros(elapsed);
    }

    pub(crate) fn record_tokenizer_read(&mut self, index: usize, elapsed: Duration) {
        if let Some(slot) = self.tokenizer_read_wall_micros.get_mut(index) {
            *slot = duration_micros(elapsed);
        }
    }

    pub(crate) fn record_pdf_parse(&mut self, elapsed: Duration) {
        self.pdf_parse_wall_micros = duration_micros(elapsed);
    }

    pub(crate) fn record_page_raster(&mut self, elapsed: Duration) {
        self.page_raster_wall_micros = duration_micros(elapsed);
    }

    pub(crate) fn record_image_decode(&mut self, elapsed: Duration) {
        self.image_decode_wall_micros = duration_micros(elapsed);
    }

    pub(crate) fn record_tokenizer_parse(&mut self, elapsed: Duration) {
        self.tokenizer_parse_wall_micros = duration_micros(elapsed);
    }

    pub(crate) fn record_model_parse(&mut self, elapsed: Duration) {
        self.model_parse_wall_micros = duration_micros(elapsed);
    }

    pub(crate) fn finish<T>(
        self,
        result: &Result<T, FocrError>,
    ) -> MusicInputPreparationDiagnosticsV1 {
        MusicInputPreparationDiagnosticsV1 {
            schema_version: MUSIC_DIAGNOSTICS_SCHEMA_VERSION,
            timing_contract: MUSIC_DIAGNOSTICS_TIMING_CONTRACT.to_owned(),
            outcome: MusicRunOutcomeV1::from_result(result),
            error_kind: result.as_ref().err().map(|error| error.kind().to_owned()),
            detail: result.as_ref().err().map(path_free_error_detail),
            source_read_wall_micros: self.source_read_wall_micros,
            model_read_wall_micros: self.model_read_wall_micros,
            tokenizer_read_wall_micros: self.tokenizer_read_wall_micros,
            pdf_parse_wall_micros: self.pdf_parse_wall_micros,
            page_raster_wall_micros: self.page_raster_wall_micros,
            image_decode_wall_micros: self.image_decode_wall_micros,
            tokenizer_parse_wall_micros: self.tokenizer_parse_wall_micros,
            model_parse_wall_micros: self.model_parse_wall_micros,
            total_wall_micros: duration_micros(self.started.elapsed()),
            resources: self.resources.finish(),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum TromrAttemptStage {
    Preprocess,
    Encode,
    Decode,
    SemanticAssembly,
}

struct AttemptBuilder {
    started: Instant,
    attempt_index: u32,
    detection_index: Option<usize>,
    segment_index: Option<usize>,
    preprocess_wall_micros: u64,
    encode_wall_micros: u64,
    decode_wall_micros: u64,
    semantic_assembly_wall_micros: u64,
}

impl AttemptBuilder {
    fn finish(
        self,
        outcome: MusicRunOutcomeV1,
        error_kind: Option<String>,
        detail: Option<String>,
    ) -> TromrForwardAttemptDiagnosticsV1 {
        TromrForwardAttemptDiagnosticsV1 {
            attempt_index: self.attempt_index,
            detection_index: self.detection_index,
            segment_index: self.segment_index,
            preprocess_wall_micros: self.preprocess_wall_micros,
            encode_wall_micros: self.encode_wall_micros,
            decode_wall_micros: self.decode_wall_micros,
            semantic_assembly_wall_micros: self.semantic_assembly_wall_micros,
            total_wall_micros: duration_micros(self.started.elapsed()),
            outcome,
            error_kind,
            detail,
        }
    }
}

pub(crate) struct TromrExecutionDiagnosticsBuilder {
    started: Instant,
    resources: ResourceSnapshot,
    execution_options_identity: String,
    blocking_queue_wait_micros: u64,
    forward_admission_wait_micros: u64,
    staff_detection_wall_micros: u64,
    attempts: Vec<TromrForwardAttemptDiagnosticsV1>,
    active_attempt: Option<AttemptBuilder>,
    next_location: (Option<usize>, Option<usize>),
    page_assembly_wall_micros: u64,
    musicxml_assembly_wall_micros: u64,
    publication_wall_micros: u64,
}

impl TromrExecutionDiagnosticsBuilder {
    pub(crate) fn new(execution_options_identity: String) -> Self {
        Self {
            started: Instant::now(),
            resources: ResourceSnapshot::capture(),
            execution_options_identity,
            blocking_queue_wait_micros: 0,
            forward_admission_wait_micros: 0,
            staff_detection_wall_micros: 0,
            attempts: Vec::new(),
            active_attempt: None,
            next_location: (None, None),
            page_assembly_wall_micros: 0,
            musicxml_assembly_wall_micros: 0,
            publication_wall_micros: 0,
        }
    }

    pub(crate) fn mark_worker_started(&mut self) {
        self.blocking_queue_wait_micros = duration_micros(self.started.elapsed());
    }

    pub(crate) fn record_forward_admission(&mut self, elapsed: Duration) {
        self.forward_admission_wait_micros = duration_micros(elapsed);
    }

    pub(crate) fn record_staff_detection(&mut self, elapsed: Duration) {
        self.staff_detection_wall_micros = self
            .staff_detection_wall_micros
            .saturating_add(duration_micros(elapsed));
    }

    pub(crate) fn set_attempt_location(
        &mut self,
        detection_index: usize,
        segment_index: Option<usize>,
    ) {
        self.next_location = (Some(detection_index), segment_index);
    }

    pub(crate) fn begin_attempt(&mut self, attempt_index: u32) {
        debug_assert!(self.active_attempt.is_none());
        self.active_attempt = Some(AttemptBuilder {
            started: Instant::now(),
            attempt_index,
            detection_index: self.next_location.0,
            segment_index: self.next_location.1,
            preprocess_wall_micros: 0,
            encode_wall_micros: 0,
            decode_wall_micros: 0,
            semantic_assembly_wall_micros: 0,
        });
    }

    pub(crate) fn record_attempt_stage(&mut self, stage: TromrAttemptStage, elapsed: Duration) {
        let Some(attempt) = self.active_attempt.as_mut() else {
            return;
        };
        let value = duration_micros(elapsed);
        match stage {
            TromrAttemptStage::Preprocess => attempt.preprocess_wall_micros = value,
            TromrAttemptStage::Encode => attempt.encode_wall_micros = value,
            TromrAttemptStage::Decode => attempt.decode_wall_micros = value,
            TromrAttemptStage::SemanticAssembly => {
                attempt.semantic_assembly_wall_micros = value;
            }
        }
    }

    pub(crate) fn finish_attempt<T>(&mut self, result: &Result<T, FocrError>) {
        let Some(attempt) = self.active_attempt.take() else {
            return;
        };
        self.attempts.push(attempt.finish(
            MusicRunOutcomeV1::from_result(result),
            result.as_ref().err().map(|error| error.kind().to_owned()),
            result.as_ref().err().map(path_free_error_detail),
        ));
    }

    pub(crate) fn record_page_assembly(&mut self, elapsed: Duration) {
        self.page_assembly_wall_micros = self
            .page_assembly_wall_micros
            .saturating_add(duration_micros(elapsed));
    }

    pub(crate) fn record_musicxml_assembly(&mut self, elapsed: Duration) {
        self.musicxml_assembly_wall_micros = duration_micros(elapsed);
    }

    pub(crate) fn record_publication(&mut self, elapsed: Duration) {
        self.publication_wall_micros = duration_micros(elapsed);
    }

    pub(crate) fn finish<T>(
        mut self,
        result: &Result<T, FocrError>,
        attempts_started: u32,
        earned_allowance_ms: u64,
    ) -> TromrExecutionDiagnosticsV1 {
        if let Some(attempt) = self.active_attempt.take() {
            self.attempts.push(attempt.finish(
                MusicRunOutcomeV1::from_result(result),
                result.as_ref().err().map(|error| error.kind().to_owned()),
                result.as_ref().err().map(path_free_error_detail),
            ));
        }
        TromrExecutionDiagnosticsV1 {
            schema_version: MUSIC_DIAGNOSTICS_SCHEMA_VERSION,
            timing_contract: MUSIC_DIAGNOSTICS_TIMING_CONTRACT.to_owned(),
            execution_options_identity: self.execution_options_identity,
            outcome: MusicRunOutcomeV1::from_result(result),
            error_kind: result.as_ref().err().map(|error| error.kind().to_owned()),
            detail: result.as_ref().err().map(path_free_error_detail),
            blocking_queue_wait_micros: self.blocking_queue_wait_micros,
            forward_admission_wait_micros: self.forward_admission_wait_micros,
            staff_detection_wall_micros: self.staff_detection_wall_micros,
            attempts: self.attempts,
            page_assembly_wall_micros: self.page_assembly_wall_micros,
            musicxml_assembly_wall_micros: self.musicxml_assembly_wall_micros,
            publication_wall_micros: self.publication_wall_micros,
            total_wall_micros: duration_micros(self.started.elapsed()),
            attempts_started,
            earned_allowance_ms,
            resources: self.resources.finish(),
        }
    }
}

fn duration_micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

fn path_free_error_detail(error: &FocrError) -> String {
    match error {
        FocrError::Timeout(detail) => detail.clone(),
        FocrError::Cancelled => "cooperative cancellation observed".to_owned(),
        FocrError::Usage(_) => "invalid explicit music request".to_owned(),
        FocrError::ModelNotFound(_) => "required TrOMR artifact unavailable".to_owned(),
        FocrError::InputDecode(_) => "music input decode failed".to_owned(),
        FocrError::FormatMismatch(_) => "provider format or schema mismatch".to_owned(),
        FocrError::NotImplemented(_) => "provider capability not implemented".to_owned(),
        FocrError::Other(_) => "provider execution failed".to_owned(),
    }
}

#[cfg(target_os = "linux")]
fn process_cpu_ticks() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let after_command = stat.get(stat.rfind(')')? + 2..)?;
    let fields: Vec<&str> = after_command.split_whitespace().collect();
    let user: u64 = fields.get(11)?.parse().ok()?;
    let system: u64 = fields.get(12)?.parse().ok()?;
    user.checked_add(system)
}

#[cfg(not(target_os = "linux"))]
fn process_cpu_ticks() -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
fn process_peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let value = line.strip_prefix("VmHWM:")?.trim();
        value.strip_suffix(" kB")?.trim().parse().ok()
    })
}

#[cfg(not(target_os = "linux"))]
fn process_peak_rss_kib() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_are_explicitly_noncanonical_and_json_stable() {
        let mut builder = TromrExecutionDiagnosticsBuilder::new("policy-id".to_owned());
        builder.mark_worker_started();
        builder.set_attempt_location(3, None);
        builder.begin_attempt(1);
        builder.record_attempt_stage(TromrAttemptStage::Preprocess, Duration::from_micros(7));
        builder.finish_attempt(&Ok::<_, FocrError>(()));
        let diagnostics = builder.finish(&Ok::<_, FocrError>(()), 1, 20);
        assert_eq!(diagnostics.outcome, MusicRunOutcomeV1::Success);
        assert_eq!(diagnostics.attempts[0].detection_index, Some(3));
        assert_eq!(diagnostics.attempts[0].preprocess_wall_micros, 7);
        let json = serde_json::to_value(&diagnostics).expect("serialize diagnostics");
        assert_eq!(json["timing_contract"], MUSIC_DIAGNOSTICS_TIMING_CONTRACT);
        assert!(json.get("replay_sha256").is_none());
    }

    #[test]
    fn observed_diagnostics_never_copy_host_paths_from_typed_errors() {
        let error = FocrError::ModelNotFound(
            "/home/private-user/models/tromr.focrq was not found".to_owned(),
        );
        let diagnostics = TromrExecutionDiagnosticsV1::unavailable("policy-id".to_owned(), &error);
        assert_eq!(diagnostics.error_kind.as_deref(), Some("model_not_found"));
        let detail = diagnostics.detail.expect("path-free detail");
        assert!(!detail.contains("/home/private-user"));
        assert_eq!(detail, "required TrOMR artifact unavailable");

        let builder = MusicInputPreparationDiagnosticsBuilder::new();
        let result: Result<(), FocrError> = Err(FocrError::InputDecode(
            "/secret/scans/score.pdf could not be decoded".to_owned(),
        ));
        let preparation = builder.finish(&result);
        assert_eq!(preparation.error_kind.as_deref(), Some("input_decode"));
        assert!(
            !preparation
                .detail
                .expect("path-free detail")
                .contains("/secret")
        );
    }
}
