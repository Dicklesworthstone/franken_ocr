//! Error type and the **stable** process exit-code mapping (plan §7.4).
//!
//! Exit codes are a public contract: `robot run_error` carries the same code in
//! its payload, and agents branch on them. Do not renumber existing codes.

use serde::Serialize;
use thiserror::Error;

/// Result alias used throughout the crate.
pub type FocrResult<T> = Result<T, FocrError>;

/// One row in the frozen public exit-code contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ExitCodeSpec {
    /// Stable process/robot exit code.
    pub code: i32,
    /// Stable machine-readable category name.
    pub name: &'static str,
    /// Human-readable meaning of the category.
    pub meaning: &'static str,
}

pub const EXIT_GENERIC: i32 = 1;
pub const EXIT_USAGE: i32 = 2;
pub const EXIT_MODEL_NOT_FOUND: i32 = 3;
pub const EXIT_INPUT_DECODE: i32 = 4;
pub const EXIT_TIMEOUT: i32 = 5;
pub const EXIT_CANCELLED: i32 = 6;
pub const EXIT_FORMAT_MISMATCH: i32 = 7;
pub const EXIT_LOW_YIELD: i32 = 8;

/// The frozen exit-code table (plan §7.4).
///
/// `0` is intentionally absent from [`FocrError::exit_code`] because it is not
/// an error variant; robot schema includes it so agents have the full table in
/// one machine-readable place.
pub const EXIT_CODE_TABLE: &[ExitCodeSpec] = &[
    ExitCodeSpec {
        code: 0,
        name: "success",
        meaning: "successful completion",
    },
    ExitCodeSpec {
        code: EXIT_GENERIC,
        name: "generic",
        meaning: "generic error or not-yet-implemented surface",
    },
    ExitCodeSpec {
        code: EXIT_USAGE,
        name: "usage",
        meaning: "usage or CLI argument error",
    },
    ExitCodeSpec {
        code: EXIT_MODEL_NOT_FOUND,
        name: "model_not_found",
        meaning: "model artifact was not found or could not be resolved",
    },
    ExitCodeSpec {
        code: EXIT_INPUT_DECODE,
        name: "input_decode",
        meaning: "input image or page could not be decoded",
    },
    ExitCodeSpec {
        code: EXIT_TIMEOUT,
        name: "timeout",
        meaning: "budget or timeout was exceeded",
    },
    ExitCodeSpec {
        code: EXIT_CANCELLED,
        name: "cancelled",
        meaning: "operation was cancelled cooperatively",
    },
    ExitCodeSpec {
        code: EXIT_FORMAT_MISMATCH,
        name: "format_mismatch",
        meaning: "format or version mismatch",
    },
    ExitCodeSpec {
        code: EXIT_LOW_YIELD,
        name: "low_yield",
        meaning: "run completed but a large input yielded almost no text and \
                  --fail-on-low-yield was set",
    },
];

/// Top-level error. Each variant maps to a stable exit code via
/// [`FocrError::exit_code`].
#[derive(Debug, Error)]
pub enum FocrError {
    /// Usage / CLI argument error.
    #[error("usage error: {0}")]
    Usage(String),

    /// The model could not be found or resolved (plan §7.5).
    #[error("model not found / not resolvable: {0}")]
    ModelNotFound(String),

    /// An input image (or PDF page) could not be decoded.
    #[error("input decode error: {0}")]
    InputDecode(String),

    /// A per-stage / per-page budget or deadline was exceeded.
    #[error("budget/timeout exceeded: {0}")]
    Timeout(String),

    /// Cancelled (Ctrl+C / cooperative cancellation).
    #[error("cancelled")]
    Cancelled,

    /// A `.focrq` (or other) format/version mismatch.
    #[error("format/version mismatch: {0}")]
    FormatMismatch(String),

    /// A large input produced almost no recognized text and the run was
    /// asked to treat that as fatal (`--fail-on-low-yield`, GH #15).
    #[error("low yield: {0}")]
    LowYield(String),

    /// A surface that is planned but not yet implemented in this phase.
    #[error("not yet implemented: {0}")]
    NotImplemented(String),

    /// Any other error.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl FocrError {
    /// Stable process exit code (plan §7.4). Keep in sync with the `robot
    /// run_error` payload. `0` is reserved for success.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self {
            FocrError::Usage(_) => EXIT_USAGE,
            FocrError::ModelNotFound(_) => EXIT_MODEL_NOT_FOUND,
            FocrError::InputDecode(_) => EXIT_INPUT_DECODE,
            FocrError::Timeout(_) => EXIT_TIMEOUT,
            FocrError::Cancelled => EXIT_CANCELLED,
            FocrError::FormatMismatch(_) => EXIT_FORMAT_MISMATCH,
            FocrError::LowYield(_) => EXIT_LOW_YIELD,
            FocrError::NotImplemented(_) | FocrError::Other(_) => EXIT_GENERIC,
        }
    }

    /// One actionable next step per error category, deliberately separate from
    /// the message so agents can branch on it without parsing prose (the
    /// frankentts/franken_whisper `remediation`/`recovery` pattern). Surfaced
    /// in the robot `run_error` payload as `recovery`.
    #[must_use]
    pub fn remediation(&self) -> &'static str {
        match self {
            FocrError::Usage(_) => {
                "re-run with corrected arguments; `focr <command> --help` documents the surface, \
                 and `focr robot triage` returns command templates"
            }
            FocrError::ModelNotFound(_) => {
                "run `focr pull` to install the default model (or `focr pull <model>` for a zoo \
                 model); pin an explicit artifact with --model or FOCR_MODEL_PATH"
            }
            FocrError::InputDecode(_) => {
                "verify the input is a supported image (PNG/JPG/…) or a SCANNED image-based PDF; \
                 born-digital/vector PDFs must be rasterized out of band first \
                 (e.g. `pdftoppm -png -r 200`)"
            }
            FocrError::Timeout(_) => {
                "raise the stage budget with FOCR_STAGE_BUDGET_FORWARD_MS (0 = unlimited), cap \
                 generation with --max-length, or check for memory pressure/swapping"
            }
            FocrError::Cancelled => {
                "the run was interrupted (Ctrl+C / cooperative shutdown); re-run to continue"
            }
            FocrError::FormatMismatch(_) => {
                "the artifact/manifest does not match this binary's contract; re-run `focr pull` \
                 for a compatible artifact and do not rename artifacts to bypass verification"
            }
            FocrError::LowYield(_) => {
                "the input is likely a low-DPI or extreme-aspect capture: re-capture at higher \
                 resolution (glyphs below ~12px are unrecoverable by any OCR engine), or drop \
                 --fail-on-low-yield to accept the sparse result"
            }
            FocrError::NotImplemented(_) => {
                "this surface is planned but not implemented; `focr models` shows ready vs \
                 planned models and `focr robot triage` lists working commands"
            }
            FocrError::Other(_) => {
                "inspect the message; `focr doctor` diagnoses local install/cache problems and \
                 `focr robot health` reports model/threads/arch state"
            }
        }
    }

    /// Stable machine-readable error category used by robot events.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            FocrError::Usage(_) => "usage",
            FocrError::ModelNotFound(_) => "model_not_found",
            FocrError::InputDecode(_) => "input_decode",
            FocrError::Timeout(_) => "timeout",
            FocrError::Cancelled => "cancelled",
            FocrError::FormatMismatch(_) => "format_mismatch",
            FocrError::LowYield(_) => "low_yield",
            FocrError::NotImplemented(_) => "not_implemented",
            FocrError::Other(_) => "generic",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn exit_codes_are_stable() {
        assert_eq!(FocrError::Usage("x".into()).exit_code(), EXIT_USAGE);
        assert_eq!(
            FocrError::ModelNotFound("x".into()).exit_code(),
            EXIT_MODEL_NOT_FOUND
        );
        assert_eq!(
            FocrError::InputDecode("x".into()).exit_code(),
            EXIT_INPUT_DECODE
        );
        assert_eq!(FocrError::Timeout("x".into()).exit_code(), EXIT_TIMEOUT);
        assert_eq!(FocrError::Cancelled.exit_code(), EXIT_CANCELLED);
        assert_eq!(
            FocrError::FormatMismatch("x".into()).exit_code(),
            EXIT_FORMAT_MISMATCH
        );
        assert_eq!(FocrError::LowYield("x".into()).exit_code(), EXIT_LOW_YIELD);
        assert_eq!(
            FocrError::NotImplemented("x".into()).exit_code(),
            EXIT_GENERIC
        );
        assert_eq!(
            FocrError::Other(anyhow::anyhow!("x")).exit_code(),
            EXIT_GENERIC
        );
    }

    #[test]
    fn exit_code_table_is_complete_and_unique() {
        let codes: BTreeSet<i32> = EXIT_CODE_TABLE.iter().map(|row| row.code).collect();
        assert_eq!(
            codes,
            BTreeSet::from([
                0,
                EXIT_GENERIC,
                EXIT_USAGE,
                EXIT_MODEL_NOT_FOUND,
                EXIT_INPUT_DECODE,
                EXIT_TIMEOUT,
                EXIT_CANCELLED,
                EXIT_FORMAT_MISMATCH,
                EXIT_LOW_YIELD,
            ])
        );
        assert_eq!(codes.len(), EXIT_CODE_TABLE.len());
    }

    #[test]
    fn error_kinds_are_stable() {
        let cases: &[(FocrError, &str)] = &[
            (FocrError::Usage("x".into()), "usage"),
            (FocrError::ModelNotFound("x".into()), "model_not_found"),
            (FocrError::InputDecode("x".into()), "input_decode"),
            (FocrError::Timeout("x".into()), "timeout"),
            (FocrError::Cancelled, "cancelled"),
            (FocrError::FormatMismatch("x".into()), "format_mismatch"),
            (FocrError::LowYield("x".into()), "low_yield"),
            (FocrError::NotImplemented("x".into()), "not_implemented"),
            (FocrError::Other(anyhow::anyhow!("x")), "generic"),
        ];
        for (err, kind) in cases {
            assert_eq!(err.kind(), *kind);
        }
    }
}
