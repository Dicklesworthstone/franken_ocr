//! Explicit execution policy and cooperative cancellation for embedded TrOMR.
//!
//! The policy is canonical and replay-affecting. The cancellation token and
//! monotonic clock are runtime controls and deliberately never serialize into a
//! receipt or cache key.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{FocrError, FocrResult};
use crate::music_diagnostics::{
    TromrAttemptStage, TromrExecutionDiagnosticsBuilder, TromrExecutionDiagnosticsV1,
};

/// Stable schema carried by [`TromrExecutionOptionsV1`].
pub const TROMR_EXECUTION_OPTIONS_SCHEMA_VERSION: u32 = 1;

/// Hard provider ceiling for one declared TrOMR page execution.
///
/// Callers can choose a smaller bound. The ceiling prevents overflow-shaped or
/// accidentally unbounded requests while leaving room for unusually dense
/// historical score pages on CPU-only hosts.
pub const MAX_TROMR_PAGE_BUDGET_MS: u64 = 4 * 60 * 60 * 1_000;

/// Hard provider ceiling for actual model forwards on one page.
pub const MAX_TROMR_FORWARD_ATTEMPTS: u32 = 64;

/// Versioned execution limits for one embedded TrOMR page.
///
/// The currently earned page allowance is:
///
/// `setup_budget_ms + attempts_started * per_forward_attempt_budget_ms`
///
/// An attempt is earned immediately before one actual model forward. A normal
/// staff consumes one attempt; every experimental split segment consumes one
/// attempt. Unused earlier allowance carries forward, but a slow early attempt
/// cannot borrow allowance from model forwards that have not started. The
/// maximum remains bounded by [`max_forward_attempts`](Self::max_forward_attempts)
/// and [`MAX_TROMR_PAGE_BUDGET_MS`].
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TromrExecutionOptionsV1 {
    /// Must equal [`TROMR_EXECUTION_OPTIONS_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Allowance for admission, tokenizer access, staff detection, and assembly.
    pub setup_budget_ms: u64,
    /// Additional cumulative allowance earned by each actual model forward.
    pub per_forward_attempt_budget_ms: u64,
    /// Maximum actual staff/segment model forwards permitted for the page.
    pub max_forward_attempts: u32,
}

impl TromrExecutionOptionsV1 {
    /// Explicit bounded production default.
    #[must_use]
    pub const fn bounded_default() -> Self {
        Self {
            schema_version: TROMR_EXECUTION_OPTIONS_SCHEMA_VERSION,
            setup_budget_ms: 60_000,
            // A real int8 selected-row forward has measured 223.3s on a
            // CPU-only host. Five minutes preserves bounded headroom for that
            // model work while the 32-attempt maximum remains below the hard
            // provider page ceiling.
            per_forward_attempt_budget_ms: 300_000,
            max_forward_attempts: 32,
        }
    }

    /// Validate schema, nonzero fields, checked arithmetic, and hard ceilings.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] for an unsupported schema, or
    /// [`FocrError::Usage`] for invalid or unbounded limits.
    pub fn validate(self) -> FocrResult<Self> {
        if self.schema_version != TROMR_EXECUTION_OPTIONS_SCHEMA_VERSION {
            return Err(FocrError::FormatMismatch(format!(
                "tromr execution options schema {} is unsupported; expected {}",
                self.schema_version, TROMR_EXECUTION_OPTIONS_SCHEMA_VERSION
            )));
        }
        if self.setup_budget_ms == 0 {
            return Err(FocrError::Usage(
                "tromr setup_budget_ms must be greater than zero".into(),
            ));
        }
        if self.per_forward_attempt_budget_ms == 0 {
            return Err(FocrError::Usage(
                "tromr per_forward_attempt_budget_ms must be greater than zero".into(),
            ));
        }
        if self.max_forward_attempts == 0 {
            return Err(FocrError::Usage(
                "tromr max_forward_attempts must be greater than zero".into(),
            ));
        }
        if self.max_forward_attempts > MAX_TROMR_FORWARD_ATTEMPTS {
            return Err(FocrError::Usage(format!(
                "tromr max_forward_attempts {} exceeds provider ceiling {}",
                self.max_forward_attempts, MAX_TROMR_FORWARD_ATTEMPTS
            )));
        }
        let maximum = self.maximum_page_budget_ms()?;
        if maximum > MAX_TROMR_PAGE_BUDGET_MS {
            return Err(FocrError::Usage(format!(
                "tromr maximum page budget {maximum}ms exceeds provider ceiling \
                 {MAX_TROMR_PAGE_BUDGET_MS}ms"
            )));
        }
        Ok(self)
    }

    /// Parse and validate the stable JSON representation.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] for malformed JSON or unknown fields;
    /// otherwise the errors from [`Self::validate`].
    pub fn from_json(json: &str) -> FocrResult<Self> {
        serde_json::from_str::<Self>(json)
            .map_err(|error| {
                FocrError::FormatMismatch(format!("tromr execution options JSON: {error}"))
            })?
            .validate()
    }

    /// Canonical clock-free JSON for provenance and replay keys.
    ///
    /// # Errors
    /// The errors from [`Self::validate`], or a serialization failure.
    pub fn canonical_json(self) -> FocrResult<String> {
        let normalized = self.validate()?;
        serde_json::to_string(&normalized).map_err(|error| {
            FocrError::Other(anyhow::anyhow!(
                "serialize tromr execution options: {error}"
            ))
        })
    }

    /// SHA-256 of [`Self::canonical_json`].
    ///
    /// # Errors
    /// The errors from [`Self::canonical_json`].
    pub fn replay_identity(self) -> FocrResult<String> {
        let digest: [u8; 32] = Sha256::digest(self.canonical_json()?.as_bytes()).into();
        Ok(hex_sha256(&digest))
    }

    /// Maximum page allowance if every declared forward attempt starts.
    ///
    /// # Errors
    /// [`FocrError::Usage`] if the declared arithmetic overflows.
    pub fn maximum_page_budget_ms(self) -> FocrResult<u64> {
        self.allowance_after_attempts(self.max_forward_attempts)
    }

    /// Cumulative page allowance after exactly `attempts_started` forwards.
    ///
    /// # Errors
    /// [`FocrError::Usage`] if the attempt count exceeds the declared maximum or
    /// if checked arithmetic overflows.
    pub fn allowance_after_attempts(self, attempts_started: u32) -> FocrResult<u64> {
        if attempts_started > self.max_forward_attempts {
            return Err(FocrError::Usage(format!(
                "tromr attempts_started {attempts_started} exceeds declared maximum {}",
                self.max_forward_attempts
            )));
        }
        self.per_forward_attempt_budget_ms
            .checked_mul(u64::from(attempts_started))
            .and_then(|attempt_allowance| self.setup_budget_ms.checked_add(attempt_allowance))
            .ok_or_else(|| {
                FocrError::Usage(
                    "tromr execution budget arithmetic overflowed u64 milliseconds".into(),
                )
            })
    }
}

impl Default for TromrExecutionOptionsV1 {
    fn default() -> Self {
        Self::bounded_default()
    }
}

/// Cloneable, per-request cooperative cancellation for embedded music OCR.
///
/// This value intentionally implements neither `Serialize` nor receipt identity.
/// Clone it for the thread or signal bridge that may request cancellation while
/// the synchronous recognition call owns another clone.
#[derive(Clone, Default)]
pub struct MusicCancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl MusicCancellationToken {
    /// Create an uncancelled request token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cooperative cancellation. Idempotent and nonblocking.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

impl fmt::Debug for MusicCancellationToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MusicCancellationToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

type ElapsedSource = Arc<dyn Fn() -> Duration + Send + Sync>;

/// Runtime state for one TrOMR page execution.
///
/// Kept crate-private because embedders declare policy and cancellation, not
/// mutable accounting state. The provider creates exactly one context inside
/// the blocking worker and threads it through every staff/segment forward.
pub(crate) struct TromrExecutionContext {
    options: TromrExecutionOptionsV1,
    cancellation: MusicCancellationToken,
    elapsed: ElapsedSource,
    attempts_started: u32,
    diagnostics: TromrExecutionDiagnosticsBuilder,
}

impl TromrExecutionContext {
    /// Start a production context against a monotonic [`Instant`].
    pub(crate) fn new(
        options: TromrExecutionOptionsV1,
        cancellation: MusicCancellationToken,
    ) -> FocrResult<Self> {
        let started = Instant::now();
        Self::with_elapsed_source(options, cancellation, Arc::new(move || started.elapsed()))
    }

    fn with_elapsed_source(
        options: TromrExecutionOptionsV1,
        cancellation: MusicCancellationToken,
        elapsed: ElapsedSource,
    ) -> FocrResult<Self> {
        let options = options.validate()?;
        let diagnostics = TromrExecutionDiagnosticsBuilder::new(options.replay_identity()?);
        Ok(Self {
            options,
            cancellation,
            elapsed,
            attempts_started: 0,
            diagnostics,
        })
    }

    /// Validate cancellation and the currently earned cumulative allowance.
    pub(crate) fn checkpoint(&self, stage: &'static str) -> FocrResult<()> {
        if self.cancellation.is_cancelled() {
            return Err(FocrError::Cancelled);
        }
        let allowance_ms = self
            .options
            .allowance_after_attempts(self.attempts_started)?;
        if (self.elapsed)() >= Duration::from_millis(allowance_ms) {
            return Err(FocrError::Timeout(format!(
                "tromr {stage} exceeded explicit cumulative page allowance of \
                 {allowance_ms}ms after {} forward attempts",
                self.attempts_started
            )));
        }
        Ok(())
    }

    /// Earn one attempt's allowance immediately before an actual model forward.
    ///
    /// The pre-grant checkpoint is load-bearing: work already over budget cannot
    /// borrow allowance from a future staff or split segment.
    pub(crate) fn begin_forward_attempt(&mut self, stage: &'static str) -> FocrResult<()> {
        self.checkpoint(stage)?;
        if self.attempts_started == self.options.max_forward_attempts {
            return Err(FocrError::Timeout(format!(
                "tromr {stage} would exceed declared maximum of {} forward attempts",
                self.options.max_forward_attempts
            )));
        }
        self.attempts_started += 1;
        self.diagnostics.begin_attempt(self.attempts_started);
        let checkpoint = self.checkpoint(stage);
        if checkpoint.is_err() {
            self.diagnostics.finish_attempt(&checkpoint);
        }
        checkpoint
    }

    pub(crate) fn mark_worker_started(&mut self) {
        self.diagnostics.mark_worker_started();
    }

    pub(crate) fn record_forward_admission(&mut self, elapsed: Duration) {
        self.diagnostics.record_forward_admission(elapsed);
    }

    pub(crate) fn record_staff_detection(&mut self, elapsed: Duration) {
        self.diagnostics.record_staff_detection(elapsed);
    }

    pub(crate) fn set_attempt_location(
        &mut self,
        detection_index: usize,
        segment_index: Option<usize>,
    ) {
        self.diagnostics
            .set_attempt_location(detection_index, segment_index);
    }

    pub(crate) fn record_attempt_stage(&mut self, stage: TromrAttemptStage, elapsed: Duration) {
        self.diagnostics.record_attempt_stage(stage, elapsed);
    }

    pub(crate) fn finish_attempt<T>(&mut self, result: &Result<T, FocrError>) {
        self.diagnostics.finish_attempt(result);
    }

    pub(crate) fn record_page_assembly(&mut self, elapsed: Duration) {
        self.diagnostics.record_page_assembly(elapsed);
    }

    pub(crate) fn record_musicxml_assembly(&mut self, elapsed: Duration) {
        self.diagnostics.record_musicxml_assembly(elapsed);
    }

    pub(crate) fn record_publication(&mut self, elapsed: Duration) {
        self.diagnostics.record_publication(elapsed);
    }

    pub(crate) fn finish_diagnostics<T>(
        self,
        result: &Result<T, FocrError>,
    ) -> TromrExecutionDiagnosticsV1 {
        let earned_allowance_ms = self
            .options
            .allowance_after_attempts(self.attempts_started)
            .unwrap_or(u64::MAX);
        self.diagnostics
            .finish(result, self.attempts_started, earned_allowance_ms)
    }

    pub(crate) fn attempts_started(&self) -> u32 {
        self.attempts_started
    }

    pub(crate) fn earned_allowance_ms(&self) -> FocrResult<u64> {
        self.options.allowance_after_attempts(self.attempts_started)
    }

    pub(crate) fn elapsed_ms(&self) -> u64 {
        u64::try_from((self.elapsed)().as_millis()).unwrap_or(u64::MAX)
    }
}

/// Whether an error controls the whole request rather than one recoverable row.
#[must_use]
pub(crate) const fn is_terminal_execution_error(error: &FocrError) -> bool {
    matches!(error, FocrError::Cancelled | FocrError::Timeout(_))
}

fn hex_sha256(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::music_diagnostics::MusicRunOutcomeV1;
    use std::sync::atomic::AtomicU64;

    fn manual_context(
        options: TromrExecutionOptionsV1,
    ) -> (
        TromrExecutionContext,
        Arc<AtomicU64>,
        MusicCancellationToken,
    ) {
        let now_ms = Arc::new(AtomicU64::new(0));
        let clock = Arc::clone(&now_ms);
        let cancellation = MusicCancellationToken::new();
        let context = TromrExecutionContext::with_elapsed_source(
            options,
            cancellation.clone(),
            Arc::new(move || Duration::from_millis(clock.load(Ordering::Acquire))),
        )
        .expect("manual execution context");
        (context, now_ms, cancellation)
    }

    fn compact_options() -> TromrExecutionOptionsV1 {
        TromrExecutionOptionsV1 {
            schema_version: TROMR_EXECUTION_OPTIONS_SCHEMA_VERSION,
            setup_budget_ms: 10,
            per_forward_attempt_budget_ms: 20,
            max_forward_attempts: 3,
        }
    }

    #[test]
    fn default_policy_is_explicit_canonical_and_replay_stable() {
        let options = TromrExecutionOptionsV1::default();
        assert_eq!(options.validate().expect("default validates"), options);
        assert_eq!(
            options.canonical_json().expect("canonical JSON"),
            r#"{"schema_version":1,"setup_budget_ms":60000,"per_forward_attempt_budget_ms":300000,"max_forward_attempts":32}"#
        );
        let identity = options.replay_identity().expect("identity");
        assert_eq!(
            identity,
            "e03fdc857d7204c3593476daffa9f381c7a12bbd69427d5d88e60dea5dbb1fcf"
        );
        assert_eq!(
            identity,
            options.replay_identity().expect("identity replays")
        );
        assert_eq!(
            options.maximum_page_budget_ms().expect("maximum"),
            9_660_000
        );
    }

    #[test]
    fn default_policy_admits_observed_int8_selected_row_and_reports_exact_bound() {
        const OBSERVED_STAFF_DETECTION_MS: u64 = 17_700;
        const OBSERVED_TOTAL_EXECUTION_MS: u64 = 242_400;
        const ONE_FORWARD_ALLOWANCE_MS: u64 = 360_000;

        let options = TromrExecutionOptionsV1::default();
        assert_eq!(
            options
                .allowance_after_attempts(1)
                .expect("one-forward allowance"),
            ONE_FORWARD_ALLOWANCE_MS
        );

        let (mut successful, successful_now, _) = manual_context(options);
        successful_now.store(OBSERVED_STAFF_DETECTION_MS, Ordering::Release);
        successful
            .begin_forward_attempt("staff-forward")
            .expect("observed detection remains inside setup allowance");
        successful_now.store(OBSERVED_TOTAL_EXECUTION_MS, Ordering::Release);
        successful
            .checkpoint("staff-semantic-assembly")
            .expect("observed selected-row execution remains inside earned allowance");
        successful.finish_attempt(&Ok::<(), FocrError>(()));
        let success_diagnostics = successful.finish_diagnostics(&Ok::<(), FocrError>(()));
        assert_eq!(success_diagnostics.outcome, MusicRunOutcomeV1::Success);
        assert_eq!(success_diagnostics.attempts_started, 1);
        assert_eq!(
            success_diagnostics.earned_allowance_ms,
            ONE_FORWARD_ALLOWANCE_MS
        );
        assert_eq!(
            success_diagnostics.execution_options_identity,
            options.replay_identity().expect("default identity")
        );
        assert_eq!(success_diagnostics.error_kind, None);
        assert_eq!(success_diagnostics.detail, None);

        let (mut expired, expired_now, _) = manual_context(options);
        expired_now.store(OBSERVED_STAFF_DETECTION_MS, Ordering::Release);
        expired
            .begin_forward_attempt("staff-forward")
            .expect("forward starts inside setup allowance");
        expired_now.store(ONE_FORWARD_ALLOWANCE_MS, Ordering::Release);
        let terminal: Result<(), FocrError> = Err(expired
            .checkpoint("staff-semantic-assembly")
            .expect_err("the explicit one-forward bound remains terminal"));
        assert!(matches!(&terminal, Err(FocrError::Timeout(_))));
        let timeout_diagnostics = expired.finish_diagnostics(&terminal);
        assert_eq!(timeout_diagnostics.outcome, MusicRunOutcomeV1::Timeout);
        assert_eq!(timeout_diagnostics.error_kind.as_deref(), Some("timeout"));
        assert_eq!(
            timeout_diagnostics.detail.as_deref(),
            Some(
                "tromr staff-semantic-assembly exceeded explicit cumulative page allowance of \
                 360000ms after 1 forward attempts"
            )
        );
        assert_eq!(timeout_diagnostics.attempts_started, 1);
        assert_eq!(
            timeout_diagnostics.earned_allowance_ms,
            ONE_FORWARD_ALLOWANCE_MS
        );
        assert_eq!(timeout_diagnostics.attempts.len(), 1);
        assert_eq!(
            timeout_diagnostics.attempts[0].outcome,
            MusicRunOutcomeV1::Timeout
        );
    }

    #[test]
    fn policy_refuses_unknown_schema_fields_zero_overflow_and_provider_ceiling() {
        for invalid in [
            TromrExecutionOptionsV1 {
                schema_version: 2,
                ..compact_options()
            },
            TromrExecutionOptionsV1 {
                setup_budget_ms: 0,
                ..compact_options()
            },
            TromrExecutionOptionsV1 {
                per_forward_attempt_budget_ms: 0,
                ..compact_options()
            },
            TromrExecutionOptionsV1 {
                max_forward_attempts: 0,
                ..compact_options()
            },
            TromrExecutionOptionsV1 {
                max_forward_attempts: MAX_TROMR_FORWARD_ATTEMPTS + 1,
                ..compact_options()
            },
            TromrExecutionOptionsV1 {
                setup_budget_ms: MAX_TROMR_PAGE_BUDGET_MS + 1,
                ..compact_options()
            },
            TromrExecutionOptionsV1 {
                setup_budget_ms: u64::MAX,
                per_forward_attempt_budget_ms: u64::MAX,
                max_forward_attempts: 2,
                ..compact_options()
            },
        ] {
            assert!(invalid.validate().is_err(), "must refuse {invalid:?}");
        }
        let unknown = r#"{"schema_version":1,"setup_budget_ms":10,"per_forward_attempt_budget_ms":20,"max_forward_attempts":3,"ambient":true}"#;
        assert!(matches!(
            TromrExecutionOptionsV1::from_json(unknown),
            Err(FocrError::FormatMismatch(_))
        ));
        let canonical = compact_options().canonical_json().expect("canonical JSON");
        assert_eq!(
            TromrExecutionOptionsV1::from_json(&canonical).expect("round trip"),
            compact_options()
        );
    }

    #[test]
    fn cumulative_allowance_scales_exactly_with_started_forwards() {
        let options = compact_options();
        assert_eq!(options.allowance_after_attempts(0).expect("setup"), 10);
        assert_eq!(options.allowance_after_attempts(1).expect("one"), 30);
        assert_eq!(options.allowance_after_attempts(2).expect("two"), 50);
        assert_eq!(options.allowance_after_attempts(3).expect("maximum"), 70);
        assert!(options.allowance_after_attempts(4).is_err());
    }

    #[test]
    fn future_forward_allowance_cannot_be_borrowed() {
        let (mut context, now, _) = manual_context(compact_options());
        now.store(10, Ordering::Release);
        let error = context
            .begin_forward_attempt("staff-forward")
            .expect_err("expired setup cannot borrow first attempt allowance");
        assert!(matches!(error, FocrError::Timeout(_)));
        assert_eq!(context.attempts_started(), 0);
    }

    #[test]
    fn each_actual_forward_earns_one_bounded_allowance_slice() {
        let (mut context, now, _) = manual_context(compact_options());
        for (attempt, elapsed_ms) in [(1, 9), (2, 29), (3, 49)] {
            now.store(elapsed_ms, Ordering::Release);
            context
                .begin_forward_attempt("staff-forward")
                .expect("attempt starts within earned allowance");
            assert_eq!(context.attempts_started(), attempt);
            context.finish_attempt(&Ok::<(), FocrError>(()));
        }
        let error = context
            .begin_forward_attempt("split-segment-forward")
            .expect_err("fourth actual model forward exceeds declared maximum");
        assert!(matches!(error, FocrError::Timeout(_)));
        assert_eq!(context.attempts_started(), 3);
    }

    #[test]
    fn cancellation_is_per_request_and_terminal_at_next_checkpoint() {
        let (context_a, _, token_a) = manual_context(compact_options());
        let (context_b, _, token_b) = manual_context(compact_options());
        token_a.cancel();
        assert!(matches!(
            context_a.checkpoint("decode-token"),
            Err(FocrError::Cancelled)
        ));
        assert!(context_b.checkpoint("decode-token").is_ok());
        assert!(!token_b.is_cancelled());
        assert!(is_terminal_execution_error(&FocrError::Cancelled));
        assert!(is_terminal_execution_error(&FocrError::Timeout(
            "page".into()
        )));
        assert!(!is_terminal_execution_error(&FocrError::Other(
            anyhow::anyhow!("row-local")
        )));
    }

    #[test]
    fn policy_identity_changes_when_any_declared_limit_changes() {
        let base = compact_options();
        for changed in [
            TromrExecutionOptionsV1 {
                setup_budget_ms: 11,
                ..base
            },
            TromrExecutionOptionsV1 {
                per_forward_attempt_budget_ms: 21,
                ..base
            },
            TromrExecutionOptionsV1 {
                max_forward_attempts: 2,
                ..base
            },
        ] {
            assert_ne!(
                base.replay_identity().expect("base identity"),
                changed.replay_identity().expect("changed identity")
            );
        }
    }

    #[test]
    fn production_execution_policy_has_no_ambient_environment_reads() {
        let source = include_str!("music_execution.rs");
        let production = source.split_once("#[cfg(test)]").expect("test boundary").0;
        assert!(!production.contains("std::env"));
        assert!(!production.contains("FOCR_STAGE_BUDGET"));
        assert!(!production.contains("FOCR_TROMR"));
    }
}
