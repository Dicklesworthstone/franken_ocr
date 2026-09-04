//! Immutable, exact-consumption input bundle for embedded TrOMR recognition.
//!
//! The bundle pins source, model, and tokenizer bytes once, validates and hashes
//! those same owned buffers, and retains them through inference. Filesystem paths
//! are diagnostic labels only; none enter canonical identities.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use image::DynamicImage;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{FocrError, FocrResult};
use crate::music_diagnostics::{
    MusicInputPreparationDiagnosticsBuilder, MusicInputPreparationDiagnosticsV1,
    TromrExecutionDiagnosticsV1,
};
use crate::music_execution::TromrExecutionOptionsV1;
use crate::native_engine::OcrModel;
use crate::native_engine::tromr::TromrRecognitionOptionsV1;
use crate::native_engine::weights::Weights;
use crate::tokenizer::music::{MusicTokenizer, Stream, TOKENIZER_FILENAMES};

/// Maximum encoded source size accepted by the immutable music-input API.
pub const MAX_MUSIC_SOURCE_BYTES: u64 = 1 << 30;
/// TrOMR artifacts are currently under 100 MiB; one GiB leaves ample headroom
/// while preventing an unbounded owned read.
pub const MAX_MUSIC_MODEL_BYTES: u64 = 1 << 30;
/// Each TrOMR WordLevel table is tiny (the largest shipped table is ~11 KiB).
pub const MAX_MUSIC_TOKENIZER_BYTES: u64 = 1 << 20;

const BUNDLE_DOMAIN: &[u8] = b"franken_ocr.music_input.bundle.v1\0";
const RASTER_DOMAIN: &[u8] = b"franken_ocr.music_input.raster.rgba8.v1\0";
const RECOGNITION_OPTIONS_DOMAIN: &[u8] = b"franken_ocr.music_input.recognition_options.v1\0";
const EXECUTION_OPTIONS_DOMAIN: &[u8] = b"franken_ocr.music_input.execution_options.v1\0";
const OPTIONS_DOMAIN: &[u8] = b"franken_ocr.music_input.combined_options.v1\0";
const REPLAY_DOMAIN: &[u8] = b"franken_ocr.music_input.replay.v2\0";
const IMMUTABLE_MUSIC_RECOGNITION_SEAL_DOMAIN: &[u8] =
    b"franken_ocr.music_input.immutable_music_recognition.seal.v2\0";
const IMMUTABLE_MUSIC_PARENT_LEDGER_DOMAIN: &[u8] =
    b"franken_ocr.music_input.parent_recognition_ledger.v2\0";
const COMBINED_MUSICXML_DOMAIN: &[u8] = b"franken_ocr.music_input.combined_musicxml.v1\0";
const SELECTED_ROW_DOMAIN: &[u8] = b"franken_ocr.music_input.selected_row.v4\0";
const SELECTED_WARNINGS_DOMAIN: &[u8] = b"franken_ocr.music_input.selected_warnings.v1\0";
const MODEL_INPUT_PNG_DOMAIN: &[u8] = b"franken_ocr.music_input.model_input_png.v1\0";
const REVIEW_CROP_PNG_DOMAIN: &[u8] = b"franken_ocr.music_input.review_crop_png.v1\0";
const SEMANTIC_DOMAIN: &[u8] = b"franken_ocr.music_input.row_semantic.v1\0";
const MUSICXML_DOMAIN: &[u8] = b"franken_ocr.music_input.row_musicxml.v1\0";
const STAFF_LINES_DOMAIN: &[u8] = b"franken_ocr.music_input.staff_lines.v1\0";
const CANDIDATE_EVIDENCE_BUNDLE_DOMAIN: &[u8] =
    b"franken_ocr.music_input.candidate_evidence_bundle.v1\0";

pub const SELECTED_MUSIC_ROW_RECEIPT_SCHEMA_VERSION: u32 = 4;
pub const SELECTED_MUSIC_ROW_RECEIPT_CONTRACT_ID: &str =
    "franken_ocr.immutable_selected_music_row.v4";
pub const SELECTED_MUSIC_ROW_RECEIPT_CANONICAL_ENCODING: &str =
    "fixed_width_le_length_prefixed_bytes_v4";
pub const SELECTED_MUSIC_WARNINGS_CANONICAL_ENCODING: &str =
    "ordered_parent_ledger_with_selected_indices_v1";
pub const TROMR_STAFF_LINE_COORDINATE_CONTRACT: &str =
    "globally_deskewed_raster_y_and_review_canvas_y_v1";
pub const TROMR_SELECTED_ROW_SPLIT_POLICY: &str = "disabled";
/// Exact candidate bundle bytes are stored under `canonical_identity.blake3`;
/// inline receipts carry only this contract, count, and full byte identity.
pub const TROMR_CANDIDATE_EVIDENCE_STORAGE_CONTRACT: &str =
    "external_exact_canonical_bytes_by_blake3_identity_v1";
/// Refusal ceiling for one row's losslessly recoverable candidate evidence.
pub const MAX_MUSIC_CANDIDATE_EVIDENCE_BUNDLE_BYTES: usize = 4 * 1024 * 1024;
pub const MUSIC_CANDIDATE_EVIDENCE_BUNDLE_SCHEMA_VERSION: u32 = 1;
pub const MUSIC_CANDIDATE_EVIDENCE_BUNDLE_CONTRACT_ID: &str =
    "franken_ocr.music_candidate_evidence_bundle.v1";
pub const MUSIC_CANDIDATE_EVIDENCE_BUNDLE_CANONICAL_ENCODING: &str =
    "serde_json_struct_order_raw_f32_bits_v1";
pub const IMMUTABLE_MUSIC_PARENT_LEDGER_SCHEMA_VERSION: u32 = 2;
pub const IMMUTABLE_MUSIC_PARENT_LEDGER_CONTRACT_ID: &str =
    "franken_ocr.immutable_music_parent_ledger.v2";
pub const IMMUTABLE_MUSIC_PARENT_LEDGER_CANONICAL_ENCODING: &str =
    "fixed_width_le_length_prefixed_bytes_v2";

/// Identity of one exact owned buffer consumed by the provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumedBytesIdentity {
    /// Exact number of bytes hashed and retained.
    pub byte_len: u64,
    /// SHA-256 over exactly those bytes.
    pub sha256: [u8; 32],
    /// BLAKE3 over exactly those bytes, for content-addressed embedders.
    pub blake3: [u8; 32],
}

impl ConsumedBytesIdentity {
    fn of(bytes: &[u8]) -> Self {
        Self {
            byte_len: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            sha256: Sha256::digest(bytes).into(),
            blake3: *blake3::hash(bytes).as_bytes(),
        }
    }

    /// Lowercase 64-character SHA-256 for JSON/receipt surfaces.
    #[must_use]
    pub fn sha256_hex(&self) -> String {
        hex_sha256(&self.sha256)
    }

    /// `blake3:<lowercase-hex>` identity for MTDT-style artifact graphs.
    #[must_use]
    pub fn blake3_prefixed(&self) -> String {
        format!("blake3:{}", blake3::Hash::from_bytes(self.blake3).to_hex())
    }
}

/// The encoded source modality selected from its owned bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MusicSourceKind {
    /// One encoded image decoded by franken_ocr's native image path.
    Image,
    /// A provider-renderable PDF parsed and rasterized by [`crate::pdf::PdfPages`],
    /// including scanned, layered-MRC, and supported vector/text pages.
    Pdf,
}

impl MusicSourceKind {
    /// Stable receipt spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Pdf => "pdf",
        }
    }
}

/// Optional caller assertions over the exact buffers the provider will use.
/// A mismatch refuses before inference; there is no fallback or substitution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MusicInputExpectations {
    pub source_sha256: Option<[u8; 32]>,
    pub model_sha256: Option<[u8; 32]>,
    /// Rhythm, pitch, lift, and note, matching [`TOKENIZER_FILENAMES`].
    pub tokenizer_sha256: [Option<[u8; 32]>; 4],
}

/// Inference-affecting source selection plus exact-content assertions.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MusicInputOptions {
    /// One-based PDF page. `None` selects page 1; images reject any value.
    pub page: Option<usize>,
    /// Explicit, validated mechanics for the TrOMR forward. Core recognition
    /// never consults process environment variables for these controls.
    pub recognition: TromrRecognitionOptionsV1,
    /// Explicit bounded execution policy. This affects admission, timeout, and
    /// maximum forward-attempt behavior, so it is part of canonical replay
    /// identity. Runtime cancellation state is deliberately not included.
    pub execution: TromrExecutionOptionsV1,
    pub expectations: MusicInputExpectations,
}

/// Provider-returned exact-consumption and replay identities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MusicInputProvenance {
    pub source_kind: MusicSourceKind,
    pub source: ConsumedBytesIdentity,
    pub model: ConsumedBytesIdentity,
    /// Rhythm, pitch, lift, and note, matching [`TOKENIZER_FILENAMES`].
    pub tokenizers: [ConsumedBytesIdentity; 4],
    pub page_count: usize,
    pub selected_page: usize,
    pub raster_width: u32,
    pub raster_height: u32,
    /// Validated recognition mechanics represented to TrOMR.
    pub recognition_options: TromrRecognitionOptionsV1,
    /// TrOMR's SHA-256 identity of its canonical recognition-options JSON.
    pub recognition_options_identity: String,
    /// Validated bounded execution policy represented to the TrOMR worker.
    pub execution_options: TromrExecutionOptionsV1,
    /// SHA-256 identity of canonical execution-policy JSON.
    pub execution_options_identity: String,
    /// Domain-separated identity over source/model/tokenizer component records.
    pub bundle_sha256: [u8; 32],
    /// Domain-separated identity over the exact RGBA8 raster represented to the
    /// TrOMR page pipeline, including dimensions.
    pub raster_sha256: [u8; 32],
    /// Domain-separated identity over canonical recognition-options JSON.
    pub recognition_options_sha256: [u8; 32],
    /// Domain-separated identity over canonical execution-policy JSON.
    pub execution_options_sha256: [u8; 32],
    /// Domain-separated identity over both inference-affecting option records.
    pub options_sha256: [u8; 32],
    /// Domain-separated identity over bundle + raster + options identities.
    pub replay_sha256: [u8; 32],
}

impl MusicInputProvenance {
    #[must_use]
    pub fn bundle_sha256_hex(&self) -> String {
        hex_sha256(&self.bundle_sha256)
    }

    #[must_use]
    pub fn raster_sha256_hex(&self) -> String {
        hex_sha256(&self.raster_sha256)
    }

    #[must_use]
    pub fn options_sha256_hex(&self) -> String {
        hex_sha256(&self.options_sha256)
    }

    #[must_use]
    pub fn recognition_options_sha256_hex(&self) -> String {
        hex_sha256(&self.recognition_options_sha256)
    }

    #[must_use]
    pub fn execution_options_sha256_hex(&self) -> String {
        hex_sha256(&self.execution_options_sha256)
    }

    #[must_use]
    pub fn replay_sha256_hex(&self) -> String {
        hex_sha256(&self.replay_sha256)
    }
}

/// Successful immutable TrOMR recognition. The score fragments and attempt
/// evidence come from the same forward whose exact inputs are recorded here.
///
/// The aggregate is opaque because row selection is an evidence-authority
/// boundary. Callers may inspect shared references or consume a validated
/// projection, but cannot replace model output and ask the provider to receipt
/// the replacement.
///
/// ```compile_fail
/// use franken_ocr::music_input::ImmutableMusicRecognition;
///
/// fn substitute_never_inferred_evidence(mut recognition: ImmutableMusicRecognition) {
///     recognition.page_meta.fragments.clear();
/// }
/// ```
#[derive(Clone, Debug)]
pub struct ImmutableMusicRecognition {
    musicxml: String,
    page_meta: crate::native_engine::MusicPageMeta,
    provenance: MusicInputProvenance,
    ledger_seal_sha256: [u8; 32],
    parent_ledger_receipt: ImmutableMusicParentLedgerReceiptV1,
}

/// Owned fields released only after the opaque recognition and its complete
/// provider ledger have been revalidated. The tuple carries no `Validated`
/// name because callers may mutate the fields after taking ownership and cannot
/// install them back into an [`ImmutableMusicRecognition`].
pub type ImmutableMusicRecognitionOwnedTuple = (
    String,
    crate::native_engine::MusicPageMeta,
    MusicInputProvenance,
    ImmutableMusicParentLedgerReceiptV1,
);

/// Raw byte identity used by selected-row receipt artifacts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectedMusicArtifactIdentity {
    pub byte_len: u64,
    pub sha256: [u8; 32],
    pub blake3: [u8; 32],
    pub domain_identity_sha256: [u8; 32],
}

/// Closed set of selected-row byte roles with provider-owned domain
/// separation. Callers choose a semantic role, never an arbitrary domain tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectedMusicArtifactRoleV1 {
    ModelInputPng,
    ReviewCropPng,
    Semantic,
    MusicXml,
    SelectedWarningsCanonical,
    CandidateEvidenceBundle,
}

impl SelectedMusicArtifactRoleV1 {
    const fn domain(self) -> &'static [u8] {
        match self {
            Self::ModelInputPng => MODEL_INPUT_PNG_DOMAIN,
            Self::ReviewCropPng => REVIEW_CROP_PNG_DOMAIN,
            Self::Semantic => SEMANTIC_DOMAIN,
            Self::MusicXml => MUSICXML_DOMAIN,
            Self::SelectedWarningsCanonical => SELECTED_WARNINGS_DOMAIN,
            Self::CandidateEvidenceBundle => CANDIDATE_EVIDENCE_BUNDLE_DOMAIN,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::ModelInputPng => "model-input PNG",
            Self::ReviewCropPng => "review-crop PNG",
            Self::Semantic => "row semantic",
            Self::MusicXml => "row MusicXML",
            Self::SelectedWarningsCanonical => "selected-warning canonical bytes",
            Self::CandidateEvidenceBundle => "candidate-evidence bundle",
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MusicCandidateEvidenceCanonicalPayloadV1 {
    schema_version: u32,
    contract_id: String,
    canonical_encoding: String,
    storage_contract: String,
    forward_candidate_lattices: Vec<crate::native_engine::tromr::TromrForwardCandidateLatticeV1>,
}

/// Exact, bounded, losslessly recoverable same-forward candidate evidence for
/// one recognized row.
///
/// The canonical bytes are suitable for an embedder-owned content-addressed
/// store under `canonical_identity.blake3`. Receipts intentionally carry only
/// the identity, count, and [`TROMR_CANDIDATE_EVIDENCE_STORAGE_CONTRACT`]; this
/// value carries the exact omitted bytes and can recover the typed lattice
/// without running OCR again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImmutableMusicCandidateEvidenceBundleV1 {
    pub schema_version: u32,
    pub contract_id: &'static str,
    pub canonical_encoding: &'static str,
    pub storage_contract: &'static str,
    pub forward_candidate_lattices:
        Vec<crate::native_engine::tromr::TromrForwardCandidateLatticeV1>,
    pub canonical_bytes: Vec<u8>,
    pub canonical_identity: SelectedMusicArtifactIdentity,
}

impl ImmutableMusicCandidateEvidenceBundleV1 {
    fn payload(
        forward_candidate_lattices: Vec<
            crate::native_engine::tromr::TromrForwardCandidateLatticeV1,
        >,
    ) -> MusicCandidateEvidenceCanonicalPayloadV1 {
        MusicCandidateEvidenceCanonicalPayloadV1 {
            schema_version: MUSIC_CANDIDATE_EVIDENCE_BUNDLE_SCHEMA_VERSION,
            contract_id: MUSIC_CANDIDATE_EVIDENCE_BUNDLE_CONTRACT_ID.to_owned(),
            canonical_encoding: MUSIC_CANDIDATE_EVIDENCE_BUNDLE_CANONICAL_ENCODING.to_owned(),
            storage_contract: TROMR_CANDIDATE_EVIDENCE_STORAGE_CONTRACT.to_owned(),
            forward_candidate_lattices,
        }
    }

    fn validate_lattices(
        lattices: &[crate::native_engine::tromr::TromrForwardCandidateLatticeV1],
    ) -> FocrResult<()> {
        if lattices.is_empty() {
            return Err(selected_row_error(
                "candidate-evidence bundle contains no forward lattice",
            ));
        }
        for (expected_index, lattice) in lattices.iter().enumerate() {
            lattice.validate()?;
            if lattice.forward_input_index
                != u32::try_from(expected_index).map_err(|_| {
                    selected_row_error("candidate-evidence forward-input index exceeds u32")
                })?
            {
                return Err(selected_row_error(format!(
                    "candidate-evidence forward-input index {} is out of order; expected {expected_index}",
                    lattice.forward_input_index
                )));
            }
        }
        Ok(())
    }

    fn encode_payload(payload: &MusicCandidateEvidenceCanonicalPayloadV1) -> FocrResult<Vec<u8>> {
        let canonical_bytes = serde_json::to_vec(payload).map_err(|error| {
            FocrError::Other(anyhow::anyhow!(
                "serialize music candidate-evidence bundle: {error}"
            ))
        })?;
        if canonical_bytes.len() > MAX_MUSIC_CANDIDATE_EVIDENCE_BUNDLE_BYTES {
            return Err(selected_row_error(format!(
                "candidate-evidence bundle is {} bytes; maximum is {MAX_MUSIC_CANDIDATE_EVIDENCE_BUNDLE_BYTES}",
                canonical_bytes.len()
            )));
        }
        Ok(canonical_bytes)
    }

    /// Build canonical evidence from the exact ordered provider lattices.
    pub fn reconstruct(
        forward_candidate_lattices: Vec<
            crate::native_engine::tromr::TromrForwardCandidateLatticeV1,
        >,
    ) -> FocrResult<Self> {
        Self::validate_lattices(&forward_candidate_lattices)?;
        let payload = Self::payload(forward_candidate_lattices.clone());
        let canonical_bytes = Self::encode_payload(&payload)?;
        let canonical_identity = selected_music_artifact_identity_v1(
            SelectedMusicArtifactRoleV1::CandidateEvidenceBundle,
            &canonical_bytes,
        );
        Ok(Self {
            schema_version: MUSIC_CANDIDATE_EVIDENCE_BUNDLE_SCHEMA_VERSION,
            contract_id: MUSIC_CANDIDATE_EVIDENCE_BUNDLE_CONTRACT_ID,
            canonical_encoding: MUSIC_CANDIDATE_EVIDENCE_BUNDLE_CANONICAL_ENCODING,
            storage_contract: TROMR_CANDIDATE_EVIDENCE_STORAGE_CONTRACT,
            forward_candidate_lattices,
            canonical_bytes,
            canonical_identity,
        })
    }

    /// Recover and independently validate exact typed evidence from CAS bytes.
    pub fn recover(canonical_bytes: Vec<u8>) -> FocrResult<Self> {
        if canonical_bytes.is_empty()
            || canonical_bytes.len() > MAX_MUSIC_CANDIDATE_EVIDENCE_BUNDLE_BYTES
        {
            return Err(selected_row_error(format!(
                "candidate-evidence recovery bytes must be 1..={MAX_MUSIC_CANDIDATE_EVIDENCE_BUNDLE_BYTES}"
            )));
        }
        let payload: MusicCandidateEvidenceCanonicalPayloadV1 = serde_json::from_slice(
            &canonical_bytes,
        )
        .map_err(|error| {
            selected_row_error(format!(
                "candidate-evidence canonical bytes are missing, truncated, or malformed: {error}"
            ))
        })?;
        if payload.schema_version != MUSIC_CANDIDATE_EVIDENCE_BUNDLE_SCHEMA_VERSION
            || payload.contract_id != MUSIC_CANDIDATE_EVIDENCE_BUNDLE_CONTRACT_ID
            || payload.canonical_encoding != MUSIC_CANDIDATE_EVIDENCE_BUNDLE_CANONICAL_ENCODING
            || payload.storage_contract != TROMR_CANDIDATE_EVIDENCE_STORAGE_CONTRACT
        {
            return Err(selected_row_error(
                "candidate-evidence recovery literal mismatch",
            ));
        }
        Self::validate_lattices(&payload.forward_candidate_lattices)?;
        if Self::encode_payload(&payload)? != canonical_bytes {
            return Err(selected_row_error(
                "candidate-evidence bytes are not the canonical encoding",
            ));
        }
        let canonical_identity = selected_music_artifact_identity_v1(
            SelectedMusicArtifactRoleV1::CandidateEvidenceBundle,
            &canonical_bytes,
        );
        Ok(Self {
            schema_version: MUSIC_CANDIDATE_EVIDENCE_BUNDLE_SCHEMA_VERSION,
            contract_id: MUSIC_CANDIDATE_EVIDENCE_BUNDLE_CONTRACT_ID,
            canonical_encoding: MUSIC_CANDIDATE_EVIDENCE_BUNDLE_CANONICAL_ENCODING,
            storage_contract: TROMR_CANDIDATE_EVIDENCE_STORAGE_CONTRACT,
            forward_candidate_lattices: payload.forward_candidate_lattices,
            canonical_bytes,
            canonical_identity,
        })
    }

    /// Validate the typed lattice, exact canonical bytes, and CAS identity.
    pub fn validate(&self) -> FocrResult<()> {
        if self.schema_version != MUSIC_CANDIDATE_EVIDENCE_BUNDLE_SCHEMA_VERSION
            || self.contract_id != MUSIC_CANDIDATE_EVIDENCE_BUNDLE_CONTRACT_ID
            || self.canonical_encoding != MUSIC_CANDIDATE_EVIDENCE_BUNDLE_CANONICAL_ENCODING
            || self.storage_contract != TROMR_CANDIDATE_EVIDENCE_STORAGE_CONTRACT
        {
            return Err(selected_row_error(
                "candidate-evidence bundle frozen literal mismatch",
            ));
        }
        let expected = Self::reconstruct(self.forward_candidate_lattices.clone())?;
        if expected.canonical_bytes != self.canonical_bytes
            || expected.canonical_identity != self.canonical_identity
        {
            return Err(selected_row_error(
                "candidate-evidence typed fields, bytes, or CAS identity differ",
            ));
        }
        Ok(())
    }

    /// Validate recovered bytes against an identity carried by a receipt.
    pub fn validate_against_identity(
        &self,
        expected: SelectedMusicArtifactIdentity,
    ) -> FocrResult<()> {
        self.validate()?;
        if self.canonical_identity != expected {
            return Err(selected_row_error(
                "candidate-evidence CAS bytes do not match the receipt identity",
            ));
        }
        Ok(())
    }
}

/// Fixed-width source/canvas geometry used by the parent-ledger receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImmutableMusicParentGeometryV1 {
    pub source_bbox_xywh: [u32; 4],
    pub canvas_wh: [u32; 2],
    pub padding_trbl: [u32; 4],
}

/// One recognized fragment in the complete parent census.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImmutableMusicParentFragmentV1 {
    pub detection_index: u32,
    pub bbox_xywh: [u32; 4],
    pub semantic: SelectedMusicArtifactIdentity,
    pub musicxml: SelectedMusicArtifactIdentity,
    /// Declares that exact bytes are external to this compact inline receipt.
    pub candidate_evidence_storage_contract: &'static str,
    /// Ordered candidate lattices, one per exact TrOMR forward input.
    pub candidate_forward_input_count: u32,
    /// Identity of bytes returned by
    /// [`ImmutableMusicRecognition::candidate_evidence_for_fragment_ordinal`].
    pub candidate_evidence_bundle: SelectedMusicArtifactIdentity,
}

/// One recognized-staff census entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImmutableMusicParentRecognizedStaffV1 {
    pub detection_index: u32,
    pub bbox_xywh: [u32; 4],
}

/// One skipped-staff census entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImmutableMusicParentSkipV1 {
    pub detection_index: u32,
    pub bbox_xywh: [u32; 4],
    pub reason: String,
}

/// Exact pixel-free identity and geometry for one TrOMR forward input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImmutableMusicParentForwardInputV1 {
    pub gray8: crate::preprocess::staff_detect::TromrGray8ArtifactIdentityV1,
    pub source_space: crate::native_engine::tromr::TromrModelInputSourceSpaceV1,
    pub source_bbox_xywh: [u32; 4],
    pub padding_trbl: [u32; 4],
    pub staff_lines_y_in_canvas: Option<[u32; 5]>,
}

/// Detector/review line anchors in fixed-width coordinate fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImmutableMusicParentStaffLinesV1 {
    pub accepted_detector_lines_y_in_globally_deskewed_raster: [u32; 5],
    pub review_crop_staff_lines_y_in_canvas: [u32; 5],
}

/// One complete recognition-attempt entry with identity-only nonselected
/// raster evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImmutableMusicParentAttemptV1 {
    pub detection_index: u32,
    pub geometry: ImmutableMusicParentGeometryV1,
    pub route: crate::native_engine::tromr::TromrRowInferenceRouteV1,
    pub forward_inputs: Vec<ImmutableMusicParentForwardInputV1>,
    pub review_crop_gray8: Option<crate::preprocess::staff_detect::TromrGray8ArtifactIdentityV1>,
    pub review_crop_geometry: Option<ImmutableMusicParentGeometryV1>,
    pub staff_lines: Option<ImmutableMusicParentStaffLinesV1>,
    pub outcome: crate::native_engine::tromr::StaffInferenceOutcome,
    pub reason: Option<String>,
}

/// One warning in the complete parent warning census.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImmutableMusicParentWarningV1 {
    pub kind: String,
    pub part: u32,
    pub measure: u32,
    pub detail: String,
}

/// Public typed projection covered by the parent recognition-ledger receipt.
///
/// Large nonselected raster buffers are represented by exact dimensioned
/// identities. All structural, textual-error, warning, option, and provenance
/// fields remain explicit so an embedder can serialize and reconstruct this
/// value without invoking OCR.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImmutableMusicParentLedgerFieldsV1 {
    pub combined_musicxml: SelectedMusicArtifactIdentity,
    pub detected_staff_count: u32,
    pub staff_segmentation_disposition:
        crate::native_engine::tromr::TromrStaffSegmentationDispositionV1,
    pub staff_detection: crate::preprocess::staff_detect::StaffDetectionEvidenceV1,
    pub fragments: Vec<ImmutableMusicParentFragmentV1>,
    pub recognized_staves: Vec<ImmutableMusicParentRecognizedStaffV1>,
    pub skips: Vec<ImmutableMusicParentSkipV1>,
    pub attempts: Vec<ImmutableMusicParentAttemptV1>,
    pub warnings: Vec<ImmutableMusicParentWarningV1>,
    pub page_recognition_options_canonical_json: String,
    pub page_recognition_options_identity: String,
    pub source_kind: MusicSourceKind,
    pub source: ConsumedBytesIdentity,
    pub model: ConsumedBytesIdentity,
    pub tokenizer_filenames: [&'static str; 4],
    pub tokenizers: [ConsumedBytesIdentity; 4],
    pub page_count: u32,
    pub selected_page_one_based: u32,
    pub raster_width: u32,
    pub raster_height: u32,
    pub recognition_options_canonical_json: String,
    pub recognition_options_identity: String,
    pub execution_options_canonical_json: String,
    pub execution_options_identity: String,
    pub bundle_sha256: [u8; 32],
    pub raster_sha256: [u8; 32],
    pub recognition_options_sha256: [u8; 32],
    pub execution_options_sha256: [u8; 32],
    pub options_sha256: [u8; 32],
    pub replay_sha256: [u8; 32],
}

/// Provider-owned compact receipt over the complete recognition ledger.
///
/// Reconstructing and validating this unsigned receipt proves canonical and
/// structural self-consistency. It does not authenticate a hostile caller who
/// changes every field and recomputes every unsigned digest. Provider origin
/// is established in-process by the opaque [`ImmutableMusicRecognition`]
/// capability; durable hostile-storage authenticity requires an embedder-owned
/// signature or authenticated store around these canonical bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImmutableMusicParentLedgerReceiptV1 {
    pub schema_version: u32,
    pub contract_id: &'static str,
    pub canonical_encoding: &'static str,
    pub fields: ImmutableMusicParentLedgerFieldsV1,
    pub canonical_bytes: Vec<u8>,
    pub canonical_identity: SelectedMusicArtifactIdentity,
}

/// Canonical parent-warning ledger plus the entries applying to this row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedMusicWarnings {
    pub parent_warnings: Vec<crate::native_engine::tromr::MusicWarning>,
    pub selected_parent_indices: Vec<u32>,
    pub selected_warnings: Vec<crate::native_engine::tromr::MusicWarning>,
    pub canonical_encoding: &'static str,
    pub canonical_bytes: Vec<u8>,
    pub canonical_identity: SelectedMusicArtifactIdentity,
}

/// Provider-owned canonical receipt for one detector-backed, non-split row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedMusicRowReceipt {
    pub schema_version: u32,
    pub contract_id: &'static str,
    pub canonical_encoding: &'static str,
    pub selected_page_one_based: u32,
    pub page_count: u32,
    pub detection_index: u32,
    pub successful_fragment_ordinal_zero_based: u32,
    pub legacy_row_ordinal_one_based: u32,
    pub detected_staff_count: u32,
    pub staff_segmentation_disposition:
        crate::native_engine::tromr::TromrStaffSegmentationDispositionV1,
    pub inference_route: crate::native_engine::tromr::TromrRowInferenceRouteV1,
    pub global_deskew: crate::preprocess::staff_detect::TromrGlobalDeskewEvidenceV1,
    pub row_refinement: crate::preprocess::staff_detect::TromrRowRefinementEvidenceV1,
    pub split_policy: &'static str,
    pub forward_input_count: u32,
    pub model_input_source_space: crate::native_engine::tromr::TromrModelInputSourceSpaceV1,
    pub model_input_source_bbox_xywh: [u32; 4],
    pub review_crop_source_bbox_xywh_in_globally_deskewed_raster: [u32; 4],
    pub model_input_canvas_wh: [u32; 2],
    pub model_input_padding_trbl: [u32; 4],
    pub review_crop_canvas_wh: [u32; 2],
    pub review_crop_padding_trbl: [u32; 4],
    pub accepted_detector_lines_y_in_globally_deskewed_raster: [u32; 5],
    pub review_crop_staff_lines_y_in_canvas: [u32; 5],
    pub model_input_staff_lines_y_in_canvas: Option<[u32; 5]>,
    pub staff_line_coordinate_contract: &'static str,
    pub staff_lines_identity: SelectedMusicArtifactIdentity,
    pub source_kind: MusicSourceKind,
    pub source: ConsumedBytesIdentity,
    pub model: ConsumedBytesIdentity,
    pub tokenizer_filenames: [&'static str; 4],
    pub tokenizers: [ConsumedBytesIdentity; 4],
    pub raster_width: u32,
    pub raster_height: u32,
    pub raster_sha256: [u8; 32],
    pub bundle_sha256: [u8; 32],
    pub recognition_options_identity: String,
    pub execution_options_identity: String,
    pub recognition_options_sha256: [u8; 32],
    pub execution_options_sha256: [u8; 32],
    pub options_sha256: [u8; 32],
    pub parent_replay_sha256: [u8; 32],
    /// Full canonical identity of the reconstructible parent-ledger receipt.
    pub parent_ledger_identity: SelectedMusicArtifactIdentity,
    pub model_input_gray8: SelectedMusicArtifactIdentity,
    pub model_input_png: SelectedMusicArtifactIdentity,
    pub review_crop_gray8: SelectedMusicArtifactIdentity,
    pub review_crop_png: SelectedMusicArtifactIdentity,
    pub semantic: SelectedMusicArtifactIdentity,
    pub musicxml: SelectedMusicArtifactIdentity,
    /// Declares where the exact omitted candidate bytes are stored.
    pub candidate_evidence_storage_contract: &'static str,
    /// Must equal the selected row's exact forward-input count.
    pub candidate_forward_input_count: u32,
    /// Identity of [`ImmutableSelectedMusicRow::candidate_evidence`].
    pub candidate_evidence_bundle: SelectedMusicArtifactIdentity,
    pub parent_warning_count: u32,
    pub selected_warning_count: u32,
    pub selected_warning_parent_indices: Vec<u32>,
    pub warnings: SelectedMusicArtifactIdentity,
    pub canonical_bytes: Vec<u8>,
    pub canonical_identity: SelectedMusicArtifactIdentity,
}

/// Exact selected row and provider-materialized review artifacts.
#[derive(Clone, Debug)]
pub struct ImmutableSelectedMusicRow {
    semantic: String,
    musicxml: String,
    model_input_gray8: crate::preprocess::staff_detect::TromrGray8CropV1,
    model_input_png: Vec<u8>,
    review_crop_gray8: crate::preprocess::staff_detect::TromrGray8CropV1,
    review_crop_png: Vec<u8>,
    staff_lines: crate::native_engine::tromr::TromrStaffLineEvidenceV1,
    candidate_evidence: ImmutableMusicCandidateEvidenceBundleV1,
    warnings: SelectedMusicWarnings,
    receipt: SelectedMusicRowReceipt,
    parent_ledger_receipt: ImmutableMusicParentLedgerReceiptV1,
}

/// Owned extraction returned only after the provider revalidates the sealed
/// selected-row aggregate. These parts are intended for immediate transfer to
/// an embedder's own immutable artifact store.
#[derive(Clone, Debug)]
pub struct ValidatedSelectedMusicRowParts {
    semantic: String,
    musicxml: String,
    model_input_gray8: crate::preprocess::staff_detect::TromrGray8CropV1,
    model_input_png: Vec<u8>,
    review_crop_gray8: crate::preprocess::staff_detect::TromrGray8CropV1,
    review_crop_png: Vec<u8>,
    staff_lines: crate::native_engine::tromr::TromrStaffLineEvidenceV1,
    candidate_evidence: ImmutableMusicCandidateEvidenceBundleV1,
    warnings: SelectedMusicWarnings,
    receipt: SelectedMusicRowReceipt,
}

pub type SelectedMusicRowOwnedTuple = (
    String,
    String,
    crate::preprocess::staff_detect::TromrGray8CropV1,
    Vec<u8>,
    crate::preprocess::staff_detect::TromrGray8CropV1,
    Vec<u8>,
    crate::native_engine::tromr::TromrStaffLineEvidenceV1,
    ImmutableMusicCandidateEvidenceBundleV1,
    SelectedMusicWarnings,
    SelectedMusicRowReceipt,
);

impl ValidatedSelectedMusicRowParts {
    #[must_use]
    pub const fn receipt(&self) -> &SelectedMusicRowReceipt {
        &self.receipt
    }

    #[must_use]
    pub const fn candidate_evidence(&self) -> &ImmutableMusicCandidateEvidenceBundleV1 {
        &self.candidate_evidence
    }

    /// Consume the sealed transfer value. The tuple deliberately carries no
    /// `Validated` name after callers take ownership and may mutate fields.
    #[must_use]
    pub fn into_owned_tuple(self) -> SelectedMusicRowOwnedTuple {
        (
            self.semantic,
            self.musicxml,
            self.model_input_gray8,
            self.model_input_png,
            self.review_crop_gray8,
            self.review_crop_png,
            self.staff_lines,
            self.candidate_evidence,
            self.warnings,
            self.receipt,
        )
    }
}

fn selected_row_error(message: impl Into<String>) -> FocrError {
    FocrError::FormatMismatch(format!(
        "immutable selected music row contract: {}",
        message.into()
    ))
}

fn checked_u32(label: &str, value: usize) -> FocrResult<u32> {
    u32::try_from(value).map_err(|_| selected_row_error(format!("{label} exceeds u32")))
}

fn checked_bbox(label: &str, bbox: (usize, usize, usize, usize)) -> FocrResult<[u32; 4]> {
    Ok([
        checked_u32(&format!("{label}.x"), bbox.0)?,
        checked_u32(&format!("{label}.y"), bbox.1)?,
        checked_u32(&format!("{label}.width"), bbox.2)?,
        checked_u32(&format!("{label}.height"), bbox.3)?,
    ])
}

fn checked_lines(label: &str, lines: [usize; 5]) -> FocrResult<[u32; 5]> {
    let mut out = [0u32; 5];
    for (index, value) in lines.into_iter().enumerate() {
        out[index] = checked_u32(&format!("{label}[{index}]"), value)?;
    }
    Ok(out)
}

fn checked_padding(
    label: &str,
    padding: crate::preprocess::staff_detect::StaffPadding,
) -> FocrResult<[u32; 4]> {
    Ok([
        checked_u32(&format!("{label}.top"), padding.top)?,
        checked_u32(&format!("{label}.right"), padding.right)?,
        checked_u32(&format!("{label}.bottom"), padding.bottom)?,
        checked_u32(&format!("{label}.left"), padding.left)?,
    ])
}

fn selected_artifact_identity(domain: &[u8], bytes: &[u8]) -> SelectedMusicArtifactIdentity {
    let raw = ConsumedBytesIdentity::of(bytes);
    let mut hasher = Sha256::new();
    hasher.update(domain);
    update_field(&mut hasher, bytes);
    SelectedMusicArtifactIdentity {
        byte_len: raw.byte_len,
        sha256: raw.sha256,
        blake3: raw.blake3,
        domain_identity_sha256: hasher.finalize().into(),
    }
}

/// Calculate the provider-frozen raw and role-separated identity for exact
/// selected-row bytes.
///
/// This is the transport-side counterpart of provider validation: embedders
/// can verify retained PNG, semantic, MusicXML, and warning bytes without OCR,
/// image conversion, or caller-controlled domain strings.
#[must_use]
pub fn selected_music_artifact_identity_v1(
    role: SelectedMusicArtifactRoleV1,
    bytes: &[u8],
) -> SelectedMusicArtifactIdentity {
    selected_artifact_identity(role.domain(), bytes)
}

/// Verify exact bytes against a provider-frozen selected-artifact role and
/// identity.
pub fn verify_selected_music_artifact_identity_v1(
    role: SelectedMusicArtifactRoleV1,
    bytes: &[u8],
    expected: SelectedMusicArtifactIdentity,
) -> FocrResult<()> {
    if selected_music_artifact_identity_v1(role, bytes) != expected {
        return Err(selected_row_error(format!(
            "{} identity does not match exact retained bytes",
            role.label()
        )));
    }
    Ok(())
}

fn selected_gray8_artifact_identity(
    value: crate::preprocess::staff_detect::TromrGray8ArtifactIdentityV1,
) -> SelectedMusicArtifactIdentity {
    SelectedMusicArtifactIdentity {
        byte_len: value.byte_len,
        sha256: value.pixels_sha256,
        blake3: value.pixels_blake3,
        domain_identity_sha256: value.identity_sha256,
    }
}

fn tromr_gray8_identity_from_selected(
    value: SelectedMusicArtifactIdentity,
    width: usize,
    height: usize,
) -> FocrResult<crate::preprocess::staff_detect::TromrGray8ArtifactIdentityV1> {
    let width =
        u64::try_from(width).map_err(|_| selected_row_error("selected Gray8 width exceeds u64"))?;
    let height = u64::try_from(height)
        .map_err(|_| selected_row_error("selected Gray8 height exceeds u64"))?;
    if width
        .checked_mul(height)
        .is_none_or(|expected| expected != value.byte_len)
    {
        return Err(selected_row_error(
            "selected Gray8 byte length differs from its dimensions",
        ));
    }
    Ok(
        crate::preprocess::staff_detect::TromrGray8ArtifactIdentityV1 {
            width,
            height,
            row_stride_bytes: width,
            byte_len: value.byte_len,
            pixels_sha256: value.sha256,
            pixels_blake3: value.blake3,
            identity_sha256: value.domain_identity_sha256,
        },
    )
}

fn selected_staff_lines_identity(
    accepted_detector_lines_y_in_globally_deskewed_raster: [u32; 5],
    review_crop_staff_lines_y_in_canvas: [u32; 5],
    model_input_staff_lines_y_in_canvas: Option<[u32; 5]>,
) -> SelectedMusicArtifactIdentity {
    let mut canonical = Vec::new();
    append_canonical_field(
        &mut canonical,
        TROMR_STAFF_LINE_COORDINATE_CONTRACT.as_bytes(),
    );
    for values in [
        accepted_detector_lines_y_in_globally_deskewed_raster,
        review_crop_staff_lines_y_in_canvas,
    ] {
        append_u32_array(&mut canonical, values);
    }
    match model_input_staff_lines_y_in_canvas {
        Some(lines) => {
            canonical.push(1);
            append_u32_array(&mut canonical, lines);
        }
        None => canonical.push(0),
    }
    selected_artifact_identity(STAFF_LINES_DOMAIN, &canonical)
}

/// Derive the fixed-contract identity for typed selected-row staff-line
/// coordinates without copying provider canonicalization logic.
///
/// Accepted detector lines use globally deskewed page-raster Y coordinates;
/// review and optional model lines use their respective canvas Y coordinates.
/// Every present five-line array must be strictly increasing. Bounds remain a
/// property of the corresponding parent/selected geometry validation.
pub fn selected_music_staff_lines_identity_v1(
    accepted_detector_lines_y_in_globally_deskewed_raster: [u32; 5],
    review_crop_staff_lines_y_in_canvas: [u32; 5],
    model_input_staff_lines_y_in_canvas: Option<[u32; 5]>,
) -> FocrResult<SelectedMusicArtifactIdentity> {
    for (label, lines) in [
        (
            "accepted detector lines",
            accepted_detector_lines_y_in_globally_deskewed_raster,
        ),
        ("review-crop lines", review_crop_staff_lines_y_in_canvas),
    ] {
        if !lines.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err(selected_row_error(format!(
                "{label} are not strictly increasing"
            )));
        }
    }
    if model_input_staff_lines_y_in_canvas
        .is_some_and(|lines| !lines.windows(2).all(|pair| pair[0] < pair[1]))
    {
        return Err(selected_row_error(
            "model-input lines are not strictly increasing",
        ));
    }
    Ok(selected_staff_lines_identity(
        accepted_detector_lines_y_in_globally_deskewed_raster,
        review_crop_staff_lines_y_in_canvas,
        model_input_staff_lines_y_in_canvas,
    ))
}

fn append_canonical_field(out: &mut Vec<u8>, value: &[u8]) {
    out.extend_from_slice(&u64::try_from(value.len()).unwrap_or(u64::MAX).to_le_bytes());
    out.extend_from_slice(value);
}

fn append_consumed_identity(out: &mut Vec<u8>, value: ConsumedBytesIdentity) {
    out.extend_from_slice(&value.byte_len.to_le_bytes());
    out.extend_from_slice(&value.sha256);
    out.extend_from_slice(&value.blake3);
}

fn append_selected_identity(out: &mut Vec<u8>, value: SelectedMusicArtifactIdentity) {
    out.extend_from_slice(&value.byte_len.to_le_bytes());
    out.extend_from_slice(&value.sha256);
    out.extend_from_slice(&value.blake3);
    out.extend_from_slice(&value.domain_identity_sha256);
}

fn append_u32_array<const N: usize>(out: &mut Vec<u8>, values: [u32; N]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn append_usize(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes());
}

fn append_usize_array<const N: usize>(out: &mut Vec<u8>, values: [usize; N]) {
    for value in values {
        append_usize(out, value);
    }
}

fn append_optional_string(out: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            out.push(1);
            append_canonical_field(out, value.as_bytes());
        }
        None => out.push(0),
    }
}

fn append_gray8_artifact_identity(
    out: &mut Vec<u8>,
    value: crate::preprocess::staff_detect::TromrGray8ArtifactIdentityV1,
) {
    out.extend_from_slice(&value.width.to_le_bytes());
    out.extend_from_slice(&value.height.to_le_bytes());
    out.extend_from_slice(&value.row_stride_bytes.to_le_bytes());
    out.extend_from_slice(&value.byte_len.to_le_bytes());
    out.extend_from_slice(&value.pixels_sha256);
    out.extend_from_slice(&value.pixels_blake3);
    out.extend_from_slice(&value.identity_sha256);
}

fn append_parent_geometry(out: &mut Vec<u8>, value: ImmutableMusicParentGeometryV1) {
    append_u32_array(out, value.source_bbox_xywh);
    append_u32_array(out, value.canvas_wh);
    append_u32_array(out, value.padding_trbl);
}

fn append_detector_geometry(
    out: &mut Vec<u8>,
    value: crate::preprocess::staff_detect::StaffCropGeometry,
) {
    append_usize_array(
        out,
        [
            value.source_bbox.0,
            value.source_bbox.1,
            value.source_bbox.2,
            value.source_bbox.3,
        ],
    );
    append_usize(out, value.canvas_width);
    append_usize(out, value.canvas_height);
    for padding in [
        value.padding.top,
        value.padding.right,
        value.padding.bottom,
        value.padding.left,
    ] {
        append_usize(out, padding);
    }
}

fn append_staff_detection_evidence(
    out: &mut Vec<u8>,
    value: &crate::preprocess::staff_detect::StaffDetectionEvidenceV1,
) {
    append_canonical_field(out, value.global_deskew.transform_contract.as_bytes());
    out.extend_from_slice(&value.global_deskew.angle_millidegrees.to_le_bytes());
    append_gray8_artifact_identity(out, value.global_deskew.input_gray8);
    append_gray8_artifact_identity(out, value.global_deskew.globally_deskewed_gray8);

    append_usize(out, value.crops.len());
    for crop in &value.crops {
        append_detector_geometry(out, crop.geometry);
        append_usize_array(out, crop.globally_deskewed_raster_lines);
        append_usize_array(out, crop.review_crop_staff_lines_y_in_canvas);
        append_canonical_field(out, crop.row_refinement.transform_contract.as_bytes());
        out.extend_from_slice(&crop.row_refinement.angle_millidegrees.to_le_bytes());
        append_gray8_artifact_identity(
            out,
            crop.row_refinement.source_crop_before_refinement_gray8,
        );
        append_gray8_artifact_identity(out, crop.row_refinement.refined_unpadded_crop_gray8);
        append_gray8_artifact_identity(out, crop.review_crop_gray8);
    }

    append_usize(out, value.candidates.len());
    for candidate in &value.candidates {
        append_canonical_field(out, candidate.origin.as_str().as_bytes());
        match candidate.origin {
            crate::preprocess::staff_detect::StaffCandidateOrigin::Global => out.push(0),
            crate::preprocess::staff_detect::StaffCandidateOrigin::Local {
                window_index,
                y_start,
                y_end,
            } => {
                out.push(1);
                append_usize(out, window_index);
                append_usize(out, y_start);
                append_usize(out, y_end);
            }
        }
        out.extend_from_slice(&candidate.profile_peak.to_le_bytes());
        out.extend_from_slice(&candidate.line_threshold.to_le_bytes());
        out.extend_from_slice(&candidate.minimum_horizontal_span_basis_points.to_le_bytes());
        for strength in candidate.line_profile_strengths {
            out.extend_from_slice(&strength.to_le_bytes());
        }
        out.extend_from_slice(&candidate.profile_floor.to_le_bytes());
        out.extend_from_slice(&candidate.profile_sum.to_le_bytes());
        append_usize_array(out, candidate.lines);
        append_usize(out, candidate.y_extent.0);
        append_usize(out, candidate.y_extent.1);
        out.extend_from_slice(&candidate.mean_spacing_milli.to_le_bytes());
        out.extend_from_slice(&candidate.spacing_reference_milli.to_le_bytes());
        out.extend_from_slice(&candidate.spacing_consistency_basis_points.to_le_bytes());
        out.extend_from_slice(&candidate.uniformity_basis_points.to_le_bytes());
        append_canonical_field(out, candidate.disposition.as_str().as_bytes());
        append_optional_string(out, candidate.reason);
    }

    out.extend_from_slice(&value.residual.staff_like_ink.to_le_bytes());
    out.extend_from_slice(&value.residual.covered_staff_like_ink.to_le_bytes());
    out.extend_from_slice(&value.residual.coverage_basis_points.to_le_bytes());
    append_usize(out, value.residual.unresolved_candidates.len());
    for lines in &value.residual.unresolved_candidates {
        append_usize_array(out, *lines);
    }
    out.push(u8::from(value.residual.unresolved));
}

fn append_selected_transform_evidence(
    out: &mut Vec<u8>,
    global: crate::preprocess::staff_detect::TromrGlobalDeskewEvidenceV1,
    row: crate::preprocess::staff_detect::TromrRowRefinementEvidenceV1,
) {
    append_canonical_field(out, global.transform_contract.as_bytes());
    out.extend_from_slice(&global.angle_millidegrees.to_le_bytes());
    append_gray8_artifact_identity(out, global.input_gray8);
    append_gray8_artifact_identity(out, global.globally_deskewed_gray8);
    append_canonical_field(out, row.transform_contract.as_bytes());
    out.extend_from_slice(&row.angle_millidegrees.to_le_bytes());
    append_gray8_artifact_identity(out, row.source_crop_before_refinement_gray8);
    append_gray8_artifact_identity(out, row.refined_unpadded_crop_gray8);
}

fn immutable_music_parent_ledger_canonical_bytes(
    fields: &ImmutableMusicParentLedgerFieldsV1,
) -> Vec<u8> {
    let mut canonical = Vec::new();
    append_canonical_field(
        &mut canonical,
        IMMUTABLE_MUSIC_PARENT_LEDGER_CONTRACT_ID.as_bytes(),
    );
    canonical.extend_from_slice(&IMMUTABLE_MUSIC_PARENT_LEDGER_SCHEMA_VERSION.to_le_bytes());
    append_canonical_field(
        &mut canonical,
        IMMUTABLE_MUSIC_PARENT_LEDGER_CANONICAL_ENCODING.as_bytes(),
    );
    append_selected_identity(&mut canonical, fields.combined_musicxml);
    canonical.extend_from_slice(&fields.detected_staff_count.to_le_bytes());
    append_canonical_field(
        &mut canonical,
        fields.staff_segmentation_disposition.as_str().as_bytes(),
    );
    append_staff_detection_evidence(&mut canonical, &fields.staff_detection);

    append_usize(&mut canonical, fields.fragments.len());
    for fragment in &fields.fragments {
        canonical.extend_from_slice(&fragment.detection_index.to_le_bytes());
        append_u32_array(&mut canonical, fragment.bbox_xywh);
        append_selected_identity(&mut canonical, fragment.semantic);
        append_selected_identity(&mut canonical, fragment.musicxml);
        append_canonical_field(
            &mut canonical,
            fragment.candidate_evidence_storage_contract.as_bytes(),
        );
        canonical.extend_from_slice(&fragment.candidate_forward_input_count.to_le_bytes());
        append_selected_identity(&mut canonical, fragment.candidate_evidence_bundle);
    }
    append_usize(&mut canonical, fields.recognized_staves.len());
    for staff in &fields.recognized_staves {
        canonical.extend_from_slice(&staff.detection_index.to_le_bytes());
        append_u32_array(&mut canonical, staff.bbox_xywh);
    }
    append_usize(&mut canonical, fields.skips.len());
    for skip in &fields.skips {
        canonical.extend_from_slice(&skip.detection_index.to_le_bytes());
        append_u32_array(&mut canonical, skip.bbox_xywh);
        append_canonical_field(&mut canonical, skip.reason.as_bytes());
    }
    append_usize(&mut canonical, fields.attempts.len());
    for attempt in &fields.attempts {
        canonical.extend_from_slice(&attempt.detection_index.to_le_bytes());
        append_parent_geometry(&mut canonical, attempt.geometry);
        append_canonical_field(&mut canonical, attempt.route.as_str().as_bytes());
        append_usize(&mut canonical, attempt.forward_inputs.len());
        for input in &attempt.forward_inputs {
            append_gray8_artifact_identity(&mut canonical, input.gray8);
            append_canonical_field(&mut canonical, input.source_space.as_str().as_bytes());
            append_u32_array(&mut canonical, input.source_bbox_xywh);
            append_u32_array(&mut canonical, input.padding_trbl);
            match input.staff_lines_y_in_canvas {
                Some(lines) => {
                    canonical.push(1);
                    append_u32_array(&mut canonical, lines);
                }
                None => canonical.push(0),
            }
        }
        match attempt.review_crop_gray8 {
            Some(identity) => {
                canonical.push(1);
                append_gray8_artifact_identity(&mut canonical, identity);
            }
            None => canonical.push(0),
        }
        match attempt.review_crop_geometry {
            Some(geometry) => {
                canonical.push(1);
                append_parent_geometry(&mut canonical, geometry);
            }
            None => canonical.push(0),
        }
        match attempt.staff_lines {
            Some(lines) => {
                canonical.push(1);
                append_u32_array(
                    &mut canonical,
                    lines.accepted_detector_lines_y_in_globally_deskewed_raster,
                );
                append_u32_array(&mut canonical, lines.review_crop_staff_lines_y_in_canvas);
            }
            None => canonical.push(0),
        }
        append_canonical_field(&mut canonical, attempt.outcome.as_str().as_bytes());
        append_optional_string(&mut canonical, attempt.reason.as_deref());
    }
    append_usize(&mut canonical, fields.warnings.len());
    for warning in &fields.warnings {
        append_canonical_field(&mut canonical, warning.kind.as_bytes());
        canonical.extend_from_slice(&warning.part.to_le_bytes());
        canonical.extend_from_slice(&warning.measure.to_le_bytes());
        append_canonical_field(&mut canonical, warning.detail.as_bytes());
    }
    for value in [
        fields.page_recognition_options_canonical_json.as_bytes(),
        fields.page_recognition_options_identity.as_bytes(),
        fields.source_kind.as_str().as_bytes(),
    ] {
        append_canonical_field(&mut canonical, value);
    }
    append_consumed_identity(&mut canonical, fields.source);
    append_consumed_identity(&mut canonical, fields.model);
    for (filename, tokenizer) in fields.tokenizer_filenames.iter().zip(fields.tokenizers) {
        append_canonical_field(&mut canonical, filename.as_bytes());
        append_consumed_identity(&mut canonical, tokenizer);
    }
    canonical.extend_from_slice(&fields.page_count.to_le_bytes());
    canonical.extend_from_slice(&fields.selected_page_one_based.to_le_bytes());
    canonical.extend_from_slice(&fields.raster_width.to_le_bytes());
    canonical.extend_from_slice(&fields.raster_height.to_le_bytes());
    for value in [
        fields.recognition_options_canonical_json.as_bytes(),
        fields.recognition_options_identity.as_bytes(),
        fields.execution_options_canonical_json.as_bytes(),
        fields.execution_options_identity.as_bytes(),
    ] {
        append_canonical_field(&mut canonical, value);
    }
    for digest in [
        fields.bundle_sha256,
        fields.raster_sha256,
        fields.recognition_options_sha256,
        fields.execution_options_sha256,
        fields.options_sha256,
        fields.replay_sha256,
    ] {
        canonical.extend_from_slice(&digest);
    }
    canonical
}

fn parent_ledger_error(message: impl Into<String>) -> FocrError {
    FocrError::FormatMismatch(format!(
        "immutable music parent ledger contract: {}",
        message.into()
    ))
}

fn detector_geometry_as_parent(
    value: crate::preprocess::staff_detect::StaffCropGeometry,
) -> FocrResult<ImmutableMusicParentGeometryV1> {
    Ok(ImmutableMusicParentGeometryV1 {
        source_bbox_xywh: checked_bbox("parent detector source bbox", value.source_bbox)?,
        canvas_wh: [
            checked_u32("parent detector canvas width", value.canvas_width)?,
            checked_u32("parent detector canvas height", value.canvas_height)?,
        ],
        padding_trbl: checked_padding("parent detector padding", value.padding)?,
    })
}

fn validate_parent_geometry(value: ImmutableMusicParentGeometryV1) -> FocrResult<()> {
    let [_, _, width, height] = value.source_bbox_xywh;
    let [canvas_width, canvas_height] = value.canvas_wh;
    let [top, right, bottom, left] = value.padding_trbl;
    if width == 0
        || height == 0
        || width
            .checked_add(left)
            .and_then(|sum| sum.checked_add(right))
            != Some(canvas_width)
        || height
            .checked_add(top)
            .and_then(|sum| sum.checked_add(bottom))
            != Some(canvas_height)
    {
        return Err(parent_ledger_error(
            "source bbox plus padding does not equal the attempt canvas",
        ));
    }
    Ok(())
}

fn validate_parent_ledger_fields(fields: &ImmutableMusicParentLedgerFieldsV1) -> FocrResult<()> {
    use crate::native_engine::tromr::{
        StaffInferenceOutcome, TromrModelInputSourceSpaceV1, TromrRowInferenceRouteV1,
        TromrStaffSegmentationDispositionV1,
    };

    fields.staff_detection.validate()?;
    if fields.combined_musicxml.byte_len == 0
        || fields.tokenizer_filenames != TOKENIZER_FILENAMES
        || fields.selected_page_one_based == 0
        || fields.selected_page_one_based > fields.page_count
        || !fields
            .staff_segmentation_disposition
            .is_consistent_with(fields.detected_staff_count as usize)
        || fields.staff_detection.crops.len() != fields.detected_staff_count as usize
    {
        return Err(parent_ledger_error(
            "frozen provenance or detector census invariant failed",
        ));
    }

    let recognition_options = crate::native_engine::tromr::TromrRecognitionOptionsV1::from_json(
        &fields.recognition_options_canonical_json,
    )?;
    if recognition_options.canonical_json()? != fields.recognition_options_canonical_json
        || recognition_options.replay_identity()? != fields.recognition_options_identity
        || fields.page_recognition_options_canonical_json
            != fields.recognition_options_canonical_json
        || fields.page_recognition_options_identity != fields.recognition_options_identity
        || component_options_digest(
            RECOGNITION_OPTIONS_DOMAIN,
            &fields.recognition_options_canonical_json,
        ) != fields.recognition_options_sha256
    {
        return Err(parent_ledger_error(
            "recognition options are not canonical or identity-complete",
        ));
    }
    let execution_options = crate::music_execution::TromrExecutionOptionsV1::from_json(
        &fields.execution_options_canonical_json,
    )?;
    if execution_options.canonical_json()? != fields.execution_options_canonical_json
        || execution_options.replay_identity()? != fields.execution_options_identity
        || component_options_digest(
            EXECUTION_OPTIONS_DOMAIN,
            &fields.execution_options_canonical_json,
        ) != fields.execution_options_sha256
        || options_digest(
            fields.recognition_options_sha256,
            fields.execution_options_sha256,
        ) != fields.options_sha256
        || bundle_digest(fields.source, fields.model, &fields.tokenizers) != fields.bundle_sha256
        || replay_digest(
            fields.bundle_sha256,
            fields.raster_sha256,
            fields.options_sha256,
        ) != fields.replay_sha256
    {
        return Err(parent_ledger_error(
            "execution, bundle, option, or replay identity is inconsistent",
        ));
    }

    let expected_attempts = if fields.detected_staff_count < 2 {
        1usize
    } else {
        fields.detected_staff_count as usize
    };
    if fields.attempts.len() != expected_attempts
        || fields.fragments.len() != fields.recognized_staves.len()
    {
        return Err(parent_ledger_error(
            "attempt or recognized-fragment census is not closed",
        ));
    }
    match fields.detected_staff_count {
        0 | 1
            if fields.fragments.len() != 1
                || fields.recognized_staves.len() != 1
                || !fields.skips.is_empty() =>
        {
            return Err(parent_ledger_error(
                "whole-raster route must contain exactly one recognized result and no skips",
            ));
        }
        2.. if fields.fragments.len() + fields.skips.len() != expected_attempts => {
            return Err(parent_ledger_error(
                "multi-crop recognized plus skipped census is not closed",
            ));
        }
        _ => {}
    }

    let fragment_indices = fields
        .fragments
        .iter()
        .map(|fragment| fragment.detection_index)
        .collect::<Vec<_>>();
    let staff_indices = fields
        .recognized_staves
        .iter()
        .map(|staff| staff.detection_index)
        .collect::<Vec<_>>();
    let skip_indices = fields
        .skips
        .iter()
        .map(|skip| skip.detection_index)
        .collect::<Vec<_>>();
    let attempt_indices = fields
        .attempts
        .iter()
        .map(|attempt| attempt.detection_index)
        .collect::<Vec<_>>();
    let strict_unique = |indices: &[u32]| indices.windows(2).all(|pair| pair[0] < pair[1]);
    if fragment_indices != staff_indices
        || !strict_unique(&fragment_indices)
        || !strict_unique(&skip_indices)
        || attempt_indices
            != (0..u32::try_from(expected_attempts).unwrap_or(u32::MAX)).collect::<Vec<_>>()
        || fields
            .fragments
            .iter()
            .zip(&fields.recognized_staves)
            .any(|(fragment, staff)| fragment.bbox_xywh != staff.bbox_xywh)
        || fields.fragments.iter().any(|fragment| {
            fragment.semantic.byte_len == 0
                || fragment.musicxml.byte_len == 0
                || fragment.candidate_evidence_storage_contract
                    != TROMR_CANDIDATE_EVIDENCE_STORAGE_CONTRACT
                || fragment.candidate_forward_input_count == 0
                || fragment.candidate_evidence_bundle.byte_len == 0
        })
        || fields.skips.iter().any(|skip| skip.reason.is_empty())
        || fields.warnings.iter().any(|warning| {
            canonical_music_warning_kind(&warning.kind).is_err()
                || warning.part == 0
                || warning.part > u32::try_from(fields.fragments.len()).unwrap_or(u32::MAX)
                || warning.detail.is_empty()
        })
    {
        return Err(parent_ledger_error(
            "row indices, bboxes, warnings, text, or candidate identities are inconsistent",
        ));
    }

    let full_raster_geometry = ImmutableMusicParentGeometryV1 {
        source_bbox_xywh: [0, 0, fields.raster_width, fields.raster_height],
        canvas_wh: [fields.raster_width, fields.raster_height],
        padding_trbl: [0; 4],
    };
    for attempt in &fields.attempts {
        validate_parent_geometry(attempt.geometry)?;
        for input in &attempt.forward_inputs {
            input.gray8.validate_shape()?;
            let [_, _, width, height] = input.source_bbox_xywh;
            if width == 0
                || height == 0
                || input.gray8.width != u64::from(width)
                || input.gray8.height != u64::from(height)
            {
                return Err(parent_ledger_error(
                    "forward input identity disagrees with its source bbox",
                ));
            }
        }
        if let Some(review) = attempt.review_crop_gray8 {
            review.validate_shape()?;
        }
        if let Some(geometry) = attempt.review_crop_geometry {
            validate_parent_geometry(geometry)?;
            if attempt.review_crop_gray8.is_none_or(|review| {
                review.width != u64::from(geometry.canvas_wh[0])
                    || review.height != u64::from(geometry.canvas_wh[1])
            }) {
                return Err(parent_ledger_error(
                    "review crop identity disagrees with its canvas geometry",
                ));
            }
        } else if attempt.review_crop_gray8.is_some() || attempt.staff_lines.is_some() {
            return Err(parent_ledger_error(
                "review pixels or lines exist without review geometry",
            ));
        }

        let recognized = fragment_indices
            .binary_search(&attempt.detection_index)
            .is_ok();
        let skipped = skip_indices.binary_search(&attempt.detection_index).is_ok();
        if recognized == skipped
            || (recognized
                && (attempt.outcome != StaffInferenceOutcome::Recognized
                    || attempt.reason.is_some()))
            || (skipped
                && (attempt.outcome != StaffInferenceOutcome::Skipped || attempt.reason.is_none()))
        {
            return Err(parent_ledger_error(
                "attempt outcome/reason disagrees with recognized and skipped censuses",
            ));
        }
        if recognized {
            let fragment = fields
                .fragments
                .iter()
                .find(|fragment| fragment.detection_index == attempt.detection_index)
                .expect("recognized index was found in fragment census");
            if fragment.candidate_forward_input_count
                != u32::try_from(attempt.forward_inputs.len()).unwrap_or(u32::MAX)
            {
                return Err(parent_ledger_error(
                    "candidate-evidence count differs from exact forward-input count",
                ));
            }
        }

        let detector_crop = fields
            .staff_detection
            .crops
            .get(attempt.detection_index as usize);
        if let Some(crop) = detector_crop {
            let expected_lines = ImmutableMusicParentStaffLinesV1 {
                accepted_detector_lines_y_in_globally_deskewed_raster: checked_lines(
                    "parent receipt detector lines",
                    crop.globally_deskewed_raster_lines,
                )?,
                review_crop_staff_lines_y_in_canvas: checked_lines(
                    "parent receipt review lines",
                    crop.review_crop_staff_lines_y_in_canvas,
                )?,
            };
            if attempt.staff_lines != Some(expected_lines) {
                return Err(parent_ledger_error(
                    "attempt line coordinates diverge from detector evidence",
                ));
            }
        }
        match attempt.route {
            TromrRowInferenceRouteV1::NoDetectedStaffWholeRasterFallback => {
                if fields.detected_staff_count != 0
                    || attempt.detection_index != 0
                    || attempt.geometry != full_raster_geometry
                    || attempt.forward_inputs.len() != 1
                    || attempt.forward_inputs[0].source_space
                        != TromrModelInputSourceSpaceV1::SelectedPageRaster
                    || attempt.forward_inputs[0].gray8
                        != fields.staff_detection.global_deskew.input_gray8
                    || attempt.review_crop_gray8.is_some()
                    || attempt.review_crop_geometry.is_some()
                    || attempt.staff_lines.is_some()
                {
                    return Err(parent_ledger_error(
                        "zero-detection attempt contains fabricated detector evidence",
                    ));
                }
            }
            TromrRowInferenceRouteV1::SingleDetectedStaffWholeRaster => {
                let crop = detector_crop.ok_or_else(|| {
                    parent_ledger_error("single-detection attempt has no detector crop")
                })?;
                if fields.detected_staff_count != 1
                    || attempt.detection_index != 0
                    || attempt.geometry != full_raster_geometry
                    || attempt.forward_inputs.len() != 1
                    || attempt.forward_inputs[0].source_space
                        != TromrModelInputSourceSpaceV1::SelectedPageRaster
                    || attempt.forward_inputs[0].gray8
                        != fields.staff_detection.global_deskew.input_gray8
                    || attempt.review_crop_gray8 != Some(crop.review_crop_gray8)
                    || attempt.review_crop_geometry
                        != Some(detector_geometry_as_parent(crop.geometry)?)
                {
                    return Err(parent_ledger_error(
                        "single-detection whole-raster attempt diverges from detector evidence",
                    ));
                }
            }
            TromrRowInferenceRouteV1::DetectedStaffCrop
            | TromrRowInferenceRouteV1::ExperimentalSplitSegments => {
                let crop = detector_crop.ok_or_else(|| {
                    parent_ledger_error("detector-backed attempt has no detector crop")
                })?;
                if fields.detected_staff_count < 2
                    || attempt.geometry != detector_geometry_as_parent(crop.geometry)?
                    || attempt.review_crop_gray8 != Some(crop.review_crop_gray8)
                    || attempt.review_crop_geometry != Some(attempt.geometry)
                    || attempt.forward_inputs.iter().any(|input| {
                        input.source_space != TromrModelInputSourceSpaceV1::ReviewCropCanvas
                    })
                    || (attempt.route == TromrRowInferenceRouteV1::DetectedStaffCrop
                        && (attempt.forward_inputs.len() != 1
                            || attempt.forward_inputs[0].gray8 != crop.review_crop_gray8))
                {
                    return Err(parent_ledger_error(
                        "detector-backed attempt diverges from retained crop evidence",
                    ));
                }
            }
        }
    }
    if fields.staff_segmentation_disposition
        != TromrStaffSegmentationDispositionV1::for_detected_staff_count(
            fields.detected_staff_count as usize,
        )
    {
        return Err(parent_ledger_error(
            "staff segmentation disposition differs from the exact detector count",
        ));
    }
    Ok(())
}

impl ImmutableMusicParentLedgerReceiptV1 {
    /// Reconstruct a canonical parent receipt from serialized typed fields.
    /// No OCR, filesystem access, or raw nonselected crop pixels are needed.
    /// This checks self-consistency, not adversarial authenticity.
    pub fn reconstruct(fields: ImmutableMusicParentLedgerFieldsV1) -> FocrResult<Self> {
        validate_parent_ledger_fields(&fields)?;
        let canonical_bytes = immutable_music_parent_ledger_canonical_bytes(&fields);
        let canonical_identity =
            selected_artifact_identity(IMMUTABLE_MUSIC_PARENT_LEDGER_DOMAIN, &canonical_bytes);
        Ok(Self {
            schema_version: IMMUTABLE_MUSIC_PARENT_LEDGER_SCHEMA_VERSION,
            contract_id: IMMUTABLE_MUSIC_PARENT_LEDGER_CONTRACT_ID,
            canonical_encoding: IMMUTABLE_MUSIC_PARENT_LEDGER_CANONICAL_ENCODING,
            fields,
            canonical_bytes,
            canonical_identity,
        })
    }

    /// Rebuild the canonical bytes from the public typed fields.
    #[must_use]
    pub fn expected_canonical_bytes(&self) -> Vec<u8> {
        immutable_music_parent_ledger_canonical_bytes(&self.fields)
    }

    /// Rebuild the domain-separated canonical identity from typed fields.
    #[must_use]
    pub fn expected_canonical_identity(&self) -> SelectedMusicArtifactIdentity {
        selected_artifact_identity(
            IMMUTABLE_MUSIC_PARENT_LEDGER_DOMAIN,
            &self.expected_canonical_bytes(),
        )
    }

    /// Validate frozen literals, closed-world fields, canonical bytes, and
    /// domain-separated identity after a serialization round trip.
    pub fn validate(&self) -> FocrResult<()> {
        if self.schema_version != IMMUTABLE_MUSIC_PARENT_LEDGER_SCHEMA_VERSION
            || self.contract_id != IMMUTABLE_MUSIC_PARENT_LEDGER_CONTRACT_ID
            || self.canonical_encoding != IMMUTABLE_MUSIC_PARENT_LEDGER_CANONICAL_ENCODING
        {
            return Err(parent_ledger_error("frozen receipt literal mismatch"));
        }
        validate_parent_ledger_fields(&self.fields)?;
        if self.canonical_bytes != self.expected_canonical_bytes()
            || self.canonical_identity != self.expected_canonical_identity()
        {
            return Err(parent_ledger_error(
                "canonical bytes or identity do not match the typed fields",
            ));
        }
        Ok(())
    }

    /// Derive the exact selected-warning canonical bytes, mapping, and role
    /// identity for one successful parent fragment ordinal.
    ///
    /// The ordinal is zero-based in the ordered `fields.fragments` census.
    /// This validates the complete parent receipt first and requires no OCR,
    /// raw crop pixels, or CLI process.
    pub fn selected_warning_evidence_for_fragment_ordinal(
        &self,
        successful_fragment_ordinal_zero_based: u32,
    ) -> FocrResult<SelectedMusicWarnings> {
        self.validate()?;
        let ordinal = usize::try_from(successful_fragment_ordinal_zero_based)
            .map_err(|_| parent_ledger_error("successful fragment ordinal exceeds usize"))?;
        if ordinal >= self.fields.fragments.len() {
            return Err(parent_ledger_error(format!(
                "successful fragment ordinal {ordinal} is outside the {}-fragment parent census",
                self.fields.fragments.len()
            )));
        }
        let parent_warnings = parent_music_warnings(&self.fields)?;
        selected_warning_evidence(
            &parent_warnings,
            ordinal
                .checked_add(1)
                .ok_or_else(|| parent_ledger_error("selected warning ordinal overflow"))?,
        )
    }

    /// Verify exact candidate bytes recovered from an embedder CAS against one
    /// fragment's parent-ledger identity and forward-input count.
    pub fn validate_candidate_evidence_for_fragment_ordinal(
        &self,
        successful_fragment_ordinal_zero_based: u32,
        candidate_evidence: &ImmutableMusicCandidateEvidenceBundleV1,
    ) -> FocrResult<()> {
        self.validate()?;
        let ordinal = usize::try_from(successful_fragment_ordinal_zero_based)
            .map_err(|_| parent_ledger_error("candidate fragment ordinal exceeds usize"))?;
        let fragment = self.fields.fragments.get(ordinal).ok_or_else(|| {
            parent_ledger_error(format!(
                "candidate fragment ordinal {ordinal} is outside the {}-fragment parent census",
                self.fields.fragments.len()
            ))
        })?;
        candidate_evidence.validate_against_identity(fragment.candidate_evidence_bundle)?;
        if fragment.candidate_evidence_storage_contract != candidate_evidence.storage_contract
            || fragment.candidate_forward_input_count
                != u32::try_from(candidate_evidence.forward_candidate_lattices.len())
                    .unwrap_or(u32::MAX)
        {
            return Err(parent_ledger_error(
                "candidate-evidence storage contract or forward count differs from parent",
            ));
        }
        Ok(())
    }

    /// Refuse a complete-page publication claim while preserving this receipt
    /// as a structurally valid row-review context.
    pub fn require_complete_for_publication(&self) -> FocrResult<()> {
        self.validate()?;
        self.fields.staff_detection.require_complete()
    }
}

fn seal_usize(hasher: &mut Sha256, label: &str, value: usize) -> FocrResult<()> {
    let value = u64::try_from(value)
        .map_err(|_| selected_row_error(format!("{label} exceeds u64 while sealing")))?;
    hasher.update(value.to_le_bytes());
    Ok(())
}

fn seal_bytes(hasher: &mut Sha256, label: &str, value: &[u8]) -> FocrResult<()> {
    seal_usize(hasher, label, value.len())?;
    hasher.update(value);
    Ok(())
}

fn seal_bbox(
    hasher: &mut Sha256,
    label: &str,
    bbox: crate::native_engine::tromr::StaffBBox,
) -> FocrResult<()> {
    for (field, value) in [
        ("x", bbox.0),
        ("y", bbox.1),
        ("width", bbox.2),
        ("height", bbox.3),
    ] {
        seal_usize(hasher, &format!("{label}.{field}"), value)?;
    }
    Ok(())
}

fn seal_padding(
    hasher: &mut Sha256,
    label: &str,
    padding: crate::preprocess::staff_detect::StaffPadding,
) -> FocrResult<()> {
    for (field, value) in [
        ("top", padding.top),
        ("right", padding.right),
        ("bottom", padding.bottom),
        ("left", padding.left),
    ] {
        seal_usize(hasher, &format!("{label}.{field}"), value)?;
    }
    Ok(())
}

fn seal_geometry(
    hasher: &mut Sha256,
    label: &str,
    geometry: crate::preprocess::staff_detect::StaffCropGeometry,
) -> FocrResult<()> {
    seal_bbox(
        hasher,
        &format!("{label}.source_bbox"),
        geometry.source_bbox,
    )?;
    seal_usize(
        hasher,
        &format!("{label}.canvas_width"),
        geometry.canvas_width,
    )?;
    seal_usize(
        hasher,
        &format!("{label}.canvas_height"),
        geometry.canvas_height,
    )?;
    seal_padding(hasher, &format!("{label}.padding"), geometry.padding)
}

fn seal_lines(hasher: &mut Sha256, label: &str, lines: [usize; 5]) -> FocrResult<()> {
    for (index, value) in lines.into_iter().enumerate() {
        seal_usize(hasher, &format!("{label}[{index}]"), value)?;
    }
    Ok(())
}

fn seal_gray8(
    hasher: &mut Sha256,
    label: &str,
    gray8: &crate::preprocess::staff_detect::TromrGray8CropV1,
) -> FocrResult<()> {
    seal_usize(hasher, &format!("{label}.width"), gray8.width())?;
    seal_usize(hasher, &format!("{label}.height"), gray8.height())?;
    seal_usize(
        hasher,
        &format!("{label}.row_stride_bytes"),
        gray8.row_stride_bytes(),
    )?;
    seal_bytes(hasher, &format!("{label}.pixels"), gray8.pixels())?;
    hasher.update(gray8.pixels_sha256());
    hasher.update(gray8.pixels_blake3());
    hasher.update(gray8.identity_sha256());
    Ok(())
}

fn seal_consumed_identity(hasher: &mut Sha256, value: ConsumedBytesIdentity) {
    hasher.update(value.byte_len.to_le_bytes());
    hasher.update(value.sha256);
    hasher.update(value.blake3);
}

fn immutable_music_parent_ledger_fields(
    musicxml: &str,
    meta: &crate::native_engine::MusicPageMeta,
    provenance: &MusicInputProvenance,
) -> FocrResult<ImmutableMusicParentLedgerFieldsV1> {
    let fragments = meta
        .fragments
        .iter()
        .map(|fragment| {
            let row_musicxml =
                crate::native_engine::tromr::semantic_to_musicxml(&fragment.semantic)?;
            let candidate_evidence = ImmutableMusicCandidateEvidenceBundleV1::reconstruct(
                fragment.forward_candidate_lattices.clone(),
            )?;
            Ok(ImmutableMusicParentFragmentV1 {
                detection_index: checked_u32(
                    "parent fragment detection index",
                    fragment.detection_index,
                )?,
                bbox_xywh: checked_bbox("parent fragment bbox", fragment.bbox)?,
                semantic: selected_music_artifact_identity_v1(
                    SelectedMusicArtifactRoleV1::Semantic,
                    fragment.semantic.as_bytes(),
                ),
                musicxml: selected_music_artifact_identity_v1(
                    SelectedMusicArtifactRoleV1::MusicXml,
                    row_musicxml.as_bytes(),
                ),
                candidate_evidence_storage_contract: TROMR_CANDIDATE_EVIDENCE_STORAGE_CONTRACT,
                candidate_forward_input_count: checked_u32(
                    "parent candidate forward-input count",
                    candidate_evidence.forward_candidate_lattices.len(),
                )?,
                candidate_evidence_bundle: candidate_evidence.canonical_identity,
            })
        })
        .collect::<FocrResult<Vec<_>>>()?;
    let recognized_staves = meta
        .staves
        .iter()
        .map(|(detection_index, bbox)| {
            Ok(ImmutableMusicParentRecognizedStaffV1 {
                detection_index: checked_u32(
                    "parent recognized staff detection index",
                    *detection_index,
                )?,
                bbox_xywh: checked_bbox("parent recognized staff bbox", *bbox)?,
            })
        })
        .collect::<FocrResult<Vec<_>>>()?;
    let skips = meta
        .skips
        .iter()
        .map(|skip| {
            Ok(ImmutableMusicParentSkipV1 {
                detection_index: checked_u32("parent skip detection index", skip.index)?,
                bbox_xywh: checked_bbox("parent skip bbox", skip.bbox)?,
                reason: skip.reason.clone(),
            })
        })
        .collect::<FocrResult<Vec<_>>>()?;
    let attempts = meta
        .staff_evidence
        .iter()
        .map(|attempt| {
            let forward_inputs = attempt
                .forward_inputs
                .iter()
                .map(|input| {
                    Ok(ImmutableMusicParentForwardInputV1 {
                        gray8: input.gray8.artifact_identity(),
                        source_space: input.source_space,
                        source_bbox_xywh: checked_bbox(
                            "parent forward input bbox",
                            input.source_bbox_xywh,
                        )?,
                        padding_trbl: checked_padding(
                            "parent forward input padding",
                            input.padding,
                        )?,
                        staff_lines_y_in_canvas: input
                            .staff_lines_y_in_canvas
                            .map(|lines| checked_lines("parent forward input lines", lines))
                            .transpose()?,
                    })
                })
                .collect::<FocrResult<Vec<_>>>()?;
            Ok(ImmutableMusicParentAttemptV1 {
                detection_index: checked_u32("parent attempt detection index", attempt.index)?,
                geometry: detector_geometry_as_parent(attempt.geometry)?,
                route: attempt.route,
                forward_inputs,
                review_crop_gray8: attempt
                    .review_crop_gray8
                    .as_ref()
                    .map(crate::preprocess::staff_detect::TromrGray8CropV1::artifact_identity),
                review_crop_geometry: attempt
                    .review_crop_geometry
                    .map(detector_geometry_as_parent)
                    .transpose()?,
                staff_lines: attempt
                    .staff_lines
                    .map(|lines| -> FocrResult<_> {
                        Ok(ImmutableMusicParentStaffLinesV1 {
                            accepted_detector_lines_y_in_globally_deskewed_raster: checked_lines(
                                "parent accepted detector lines",
                                lines.accepted_detector_lines_y_in_globally_deskewed_raster,
                            )?,
                            review_crop_staff_lines_y_in_canvas: checked_lines(
                                "parent review crop lines",
                                lines.review_crop_staff_lines_y_in_canvas,
                            )?,
                        })
                    })
                    .transpose()?,
                outcome: attempt.outcome,
                reason: attempt.reason.clone(),
            })
        })
        .collect::<FocrResult<Vec<_>>>()?;
    let warnings = meta
        .warnings
        .iter()
        .map(|warning| {
            Ok(ImmutableMusicParentWarningV1 {
                kind: warning.kind.to_owned(),
                part: checked_u32("parent warning part", warning.part)?,
                measure: checked_u32("parent warning measure", warning.measure)?,
                detail: warning.detail.clone(),
            })
        })
        .collect::<FocrResult<Vec<_>>>()?;
    Ok(ImmutableMusicParentLedgerFieldsV1 {
        combined_musicxml: selected_artifact_identity(
            COMBINED_MUSICXML_DOMAIN,
            musicxml.as_bytes(),
        ),
        detected_staff_count: checked_u32(
            "parent detected staff count",
            meta.detected_staff_count,
        )?,
        staff_segmentation_disposition: meta.staff_segmentation_disposition,
        staff_detection: meta.staff_detection.clone(),
        fragments,
        recognized_staves,
        skips,
        attempts,
        warnings,
        page_recognition_options_canonical_json: meta.recognition_options.canonical_json()?,
        page_recognition_options_identity: meta.recognition_options_identity.clone(),
        source_kind: provenance.source_kind,
        source: provenance.source,
        model: provenance.model,
        tokenizer_filenames: TOKENIZER_FILENAMES,
        tokenizers: provenance.tokenizers,
        page_count: checked_u32("parent page count", provenance.page_count)?,
        selected_page_one_based: checked_u32("parent selected page", provenance.selected_page)?,
        raster_width: provenance.raster_width,
        raster_height: provenance.raster_height,
        recognition_options_canonical_json: provenance.recognition_options.canonical_json()?,
        recognition_options_identity: provenance.recognition_options_identity.clone(),
        execution_options_canonical_json: provenance.execution_options.canonical_json()?,
        execution_options_identity: provenance.execution_options_identity.clone(),
        bundle_sha256: provenance.bundle_sha256,
        raster_sha256: provenance.raster_sha256,
        recognition_options_sha256: provenance.recognition_options_sha256,
        execution_options_sha256: provenance.execution_options_sha256,
        options_sha256: provenance.options_sha256,
        replay_sha256: provenance.replay_sha256,
    })
}

fn immutable_music_parent_ledger_receipt(
    musicxml: &str,
    meta: &crate::native_engine::MusicPageMeta,
    provenance: &MusicInputProvenance,
) -> FocrResult<ImmutableMusicParentLedgerReceiptV1> {
    ImmutableMusicParentLedgerReceiptV1::reconstruct(immutable_music_parent_ledger_fields(
        musicxml, meta, provenance,
    )?)
}

/// Bind every field that can authorize a selected-row receipt. This is a
/// private capability seal, not a public artifact identity: external safe Rust
/// can recompute the digest but cannot construct or modify the opaque aggregate
/// that stores it.
fn immutable_music_recognition_ledger_seal(
    musicxml: &str,
    meta: &crate::native_engine::MusicPageMeta,
    provenance: &MusicInputProvenance,
) -> FocrResult<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(IMMUTABLE_MUSIC_RECOGNITION_SEAL_DOMAIN);
    seal_bytes(&mut hasher, "combined musicxml", musicxml.as_bytes())?;

    seal_usize(
        &mut hasher,
        "detected staff count",
        meta.detected_staff_count,
    )?;
    seal_bytes(
        &mut hasher,
        "staff segmentation disposition",
        meta.staff_segmentation_disposition.as_str().as_bytes(),
    )?;
    let mut detector_evidence = Vec::new();
    append_staff_detection_evidence(&mut detector_evidence, &meta.staff_detection);
    seal_bytes(&mut hasher, "staff detection evidence", &detector_evidence)?;
    seal_usize(&mut hasher, "fragment count", meta.fragments.len())?;
    for (index, fragment) in meta.fragments.iter().enumerate() {
        seal_usize(
            &mut hasher,
            &format!("fragments[{index}].detection_index"),
            fragment.detection_index,
        )?;
        seal_bbox(
            &mut hasher,
            &format!("fragments[{index}].bbox"),
            fragment.bbox,
        )?;
        seal_bytes(
            &mut hasher,
            &format!("fragments[{index}].semantic"),
            fragment.semantic.as_bytes(),
        )?;
        let candidate_evidence = ImmutableMusicCandidateEvidenceBundleV1::reconstruct(
            fragment.forward_candidate_lattices.clone(),
        )?;
        seal_bytes(
            &mut hasher,
            &format!("fragments[{index}].candidate_evidence"),
            &candidate_evidence.canonical_bytes,
        )?;
    }
    seal_usize(&mut hasher, "recognized staff count", meta.staves.len())?;
    for (index, (detection_index, bbox)) in meta.staves.iter().enumerate() {
        seal_usize(
            &mut hasher,
            &format!("staves[{index}].detection_index"),
            *detection_index,
        )?;
        seal_bbox(&mut hasher, &format!("staves[{index}].bbox"), *bbox)?;
    }
    seal_usize(&mut hasher, "skip count", meta.skips.len())?;
    for (index, skip) in meta.skips.iter().enumerate() {
        seal_usize(
            &mut hasher,
            &format!("skips[{index}].detection_index"),
            skip.index,
        )?;
        seal_bbox(&mut hasher, &format!("skips[{index}].bbox"), skip.bbox)?;
        seal_bytes(
            &mut hasher,
            &format!("skips[{index}].reason"),
            skip.reason.as_bytes(),
        )?;
    }

    seal_usize(
        &mut hasher,
        "staff attempt count",
        meta.staff_evidence.len(),
    )?;
    for (index, attempt) in meta.staff_evidence.iter().enumerate() {
        let label = format!("staff_evidence[{index}]");
        seal_usize(
            &mut hasher,
            &format!("{label}.detection_index"),
            attempt.index,
        )?;
        seal_geometry(&mut hasher, &format!("{label}.geometry"), attempt.geometry)?;
        seal_bytes(
            &mut hasher,
            &format!("{label}.route"),
            attempt.route.as_str().as_bytes(),
        )?;
        seal_usize(
            &mut hasher,
            &format!("{label}.forward_input_count"),
            attempt.forward_inputs.len(),
        )?;
        for (input_index, input) in attempt.forward_inputs.iter().enumerate() {
            let input_label = format!("{label}.forward_inputs[{input_index}]");
            seal_gray8(&mut hasher, &format!("{input_label}.gray8"), &input.gray8)?;
            seal_bytes(
                &mut hasher,
                &format!("{input_label}.source_space"),
                input.source_space.as_str().as_bytes(),
            )?;
            seal_bbox(
                &mut hasher,
                &format!("{input_label}.source_bbox_xywh"),
                input.source_bbox_xywh,
            )?;
            seal_padding(
                &mut hasher,
                &format!("{input_label}.padding"),
                input.padding,
            )?;
            match input.staff_lines_y_in_canvas {
                Some(lines) => {
                    hasher.update([1]);
                    seal_lines(
                        &mut hasher,
                        &format!("{input_label}.staff_lines_y_in_canvas"),
                        lines,
                    )?;
                }
                None => hasher.update([0]),
            }
        }
        match &attempt.review_crop_gray8 {
            Some(review) => {
                hasher.update([1]);
                seal_gray8(&mut hasher, &format!("{label}.review_crop_gray8"), review)?;
            }
            None => hasher.update([0]),
        }
        match attempt.review_crop_geometry {
            Some(geometry) => {
                hasher.update([1]);
                seal_geometry(
                    &mut hasher,
                    &format!("{label}.review_crop_geometry"),
                    geometry,
                )?;
            }
            None => hasher.update([0]),
        }
        match attempt.staff_lines {
            Some(lines) => {
                hasher.update([1]);
                seal_lines(
                    &mut hasher,
                    &format!("{label}.accepted_detector_lines_y_in_globally_deskewed_raster"),
                    lines.accepted_detector_lines_y_in_globally_deskewed_raster,
                )?;
                seal_lines(
                    &mut hasher,
                    &format!("{label}.review_crop_staff_lines_y_in_canvas"),
                    lines.review_crop_staff_lines_y_in_canvas,
                )?;
            }
            None => hasher.update([0]),
        }
        seal_bytes(
            &mut hasher,
            &format!("{label}.outcome"),
            attempt.outcome.as_str().as_bytes(),
        )?;
        match &attempt.reason {
            Some(reason) => {
                hasher.update([1]);
                seal_bytes(&mut hasher, &format!("{label}.reason"), reason.as_bytes())?;
            }
            None => hasher.update([0]),
        }
    }

    seal_usize(&mut hasher, "warning count", meta.warnings.len())?;
    for (index, warning) in meta.warnings.iter().enumerate() {
        seal_bytes(
            &mut hasher,
            &format!("warnings[{index}].kind"),
            warning.kind.as_bytes(),
        )?;
        seal_usize(
            &mut hasher,
            &format!("warnings[{index}].part"),
            warning.part,
        )?;
        seal_usize(
            &mut hasher,
            &format!("warnings[{index}].measure"),
            warning.measure,
        )?;
        seal_bytes(
            &mut hasher,
            &format!("warnings[{index}].detail"),
            warning.detail.as_bytes(),
        )?;
    }
    seal_bytes(
        &mut hasher,
        "page recognition options",
        meta.recognition_options.canonical_json()?.as_bytes(),
    )?;
    seal_bytes(
        &mut hasher,
        "page recognition options identity",
        meta.recognition_options_identity.as_bytes(),
    )?;

    seal_bytes(
        &mut hasher,
        "source kind",
        provenance.source_kind.as_str().as_bytes(),
    )?;
    seal_consumed_identity(&mut hasher, provenance.source);
    seal_consumed_identity(&mut hasher, provenance.model);
    seal_usize(&mut hasher, "tokenizer count", provenance.tokenizers.len())?;
    for tokenizer in provenance.tokenizers {
        seal_consumed_identity(&mut hasher, tokenizer);
    }
    seal_usize(&mut hasher, "page count", provenance.page_count)?;
    seal_usize(&mut hasher, "selected page", provenance.selected_page)?;
    hasher.update(provenance.raster_width.to_le_bytes());
    hasher.update(provenance.raster_height.to_le_bytes());
    seal_bytes(
        &mut hasher,
        "provenance recognition options",
        provenance.recognition_options.canonical_json()?.as_bytes(),
    )?;
    seal_bytes(
        &mut hasher,
        "provenance recognition options identity",
        provenance.recognition_options_identity.as_bytes(),
    )?;
    seal_bytes(
        &mut hasher,
        "provenance execution options",
        provenance.execution_options.canonical_json()?.as_bytes(),
    )?;
    seal_bytes(
        &mut hasher,
        "provenance execution options identity",
        provenance.execution_options_identity.as_bytes(),
    )?;
    for digest in [
        provenance.bundle_sha256,
        provenance.raster_sha256,
        provenance.recognition_options_sha256,
        provenance.execution_options_sha256,
        provenance.options_sha256,
        provenance.replay_sha256,
    ] {
        hasher.update(digest);
    }
    Ok(hasher.finalize().into())
}

impl SelectedMusicRowReceipt {
    fn rebuild_canonical_bytes(&self) -> Vec<u8> {
        let mut canonical = Vec::new();
        append_canonical_field(&mut canonical, self.contract_id.as_bytes());
        canonical.extend_from_slice(&self.schema_version.to_le_bytes());
        append_canonical_field(&mut canonical, self.canonical_encoding.as_bytes());
        canonical.extend_from_slice(&self.selected_page_one_based.to_le_bytes());
        canonical.extend_from_slice(&self.page_count.to_le_bytes());
        canonical.extend_from_slice(&self.detection_index.to_le_bytes());
        canonical.extend_from_slice(&self.successful_fragment_ordinal_zero_based.to_le_bytes());
        canonical.extend_from_slice(&self.legacy_row_ordinal_one_based.to_le_bytes());
        canonical.extend_from_slice(&self.detected_staff_count.to_le_bytes());
        append_canonical_field(
            &mut canonical,
            self.staff_segmentation_disposition.as_str().as_bytes(),
        );
        append_canonical_field(&mut canonical, self.inference_route.as_str().as_bytes());
        append_selected_transform_evidence(&mut canonical, self.global_deskew, self.row_refinement);
        append_canonical_field(&mut canonical, self.split_policy.as_bytes());
        canonical.extend_from_slice(&self.forward_input_count.to_le_bytes());
        append_canonical_field(
            &mut canonical,
            self.model_input_source_space.as_str().as_bytes(),
        );
        for values in [
            self.model_input_source_bbox_xywh,
            self.review_crop_source_bbox_xywh_in_globally_deskewed_raster,
        ] {
            for value in values {
                canonical.extend_from_slice(&value.to_le_bytes());
            }
        }
        for values in [
            &self.model_input_canvas_wh[..],
            &self.model_input_padding_trbl[..],
            &self.review_crop_canvas_wh[..],
            &self.review_crop_padding_trbl[..],
        ] {
            for value in values {
                canonical.extend_from_slice(&value.to_le_bytes());
            }
        }
        for values in [
            self.accepted_detector_lines_y_in_globally_deskewed_raster,
            self.review_crop_staff_lines_y_in_canvas,
        ] {
            for value in values {
                canonical.extend_from_slice(&value.to_le_bytes());
            }
        }
        match self.model_input_staff_lines_y_in_canvas {
            Some(lines) => {
                canonical.push(1);
                for value in lines {
                    canonical.extend_from_slice(&value.to_le_bytes());
                }
            }
            None => canonical.push(0),
        }
        append_canonical_field(
            &mut canonical,
            self.staff_line_coordinate_contract.as_bytes(),
        );
        append_selected_identity(&mut canonical, self.staff_lines_identity);
        append_canonical_field(&mut canonical, self.source_kind.as_str().as_bytes());
        append_consumed_identity(&mut canonical, self.source);
        append_consumed_identity(&mut canonical, self.model);
        for (filename, tokenizer) in self.tokenizer_filenames.iter().zip(self.tokenizers) {
            append_canonical_field(&mut canonical, filename.as_bytes());
            append_consumed_identity(&mut canonical, tokenizer);
        }
        canonical.extend_from_slice(&self.raster_width.to_le_bytes());
        canonical.extend_from_slice(&self.raster_height.to_le_bytes());
        canonical.extend_from_slice(&self.raster_sha256);
        canonical.extend_from_slice(&self.bundle_sha256);
        append_canonical_field(&mut canonical, self.recognition_options_identity.as_bytes());
        append_canonical_field(&mut canonical, self.execution_options_identity.as_bytes());
        canonical.extend_from_slice(&self.recognition_options_sha256);
        canonical.extend_from_slice(&self.execution_options_sha256);
        canonical.extend_from_slice(&self.options_sha256);
        canonical.extend_from_slice(&self.parent_replay_sha256);
        append_selected_identity(&mut canonical, self.parent_ledger_identity);
        for identity in [
            self.model_input_gray8,
            self.model_input_png,
            self.review_crop_gray8,
            self.review_crop_png,
            self.semantic,
            self.musicxml,
        ] {
            append_selected_identity(&mut canonical, identity);
        }
        append_canonical_field(
            &mut canonical,
            self.candidate_evidence_storage_contract.as_bytes(),
        );
        canonical.extend_from_slice(&self.candidate_forward_input_count.to_le_bytes());
        append_selected_identity(&mut canonical, self.candidate_evidence_bundle);
        canonical.extend_from_slice(&self.parent_warning_count.to_le_bytes());
        canonical.extend_from_slice(&self.selected_warning_count.to_le_bytes());
        canonical.extend_from_slice(
            &u32::try_from(self.selected_warning_parent_indices.len())
                .unwrap_or(u32::MAX)
                .to_le_bytes(),
        );
        for index in &self.selected_warning_parent_indices {
            canonical.extend_from_slice(&index.to_le_bytes());
        }
        append_selected_identity(&mut canonical, self.warnings);
        canonical
    }

    /// Rebuild the deterministic canonical payload from the typed fields.
    ///
    /// This is a transport verifier, not a recognition constructor: only an
    /// [`ImmutableSelectedMusicRow`] returned by [`ImmutableMusicRecognition::select_row`]
    /// carries provider-authorized semantic and pixel evidence.
    #[must_use]
    pub fn expected_canonical_bytes(&self) -> Vec<u8> {
        self.rebuild_canonical_bytes()
    }

    /// Rebuild the canonical selected-row identity from the typed fields.
    #[must_use]
    pub fn expected_canonical_identity(&self) -> SelectedMusicArtifactIdentity {
        let canonical = self.rebuild_canonical_bytes();
        selected_artifact_identity(SELECTED_ROW_DOMAIN, &canonical)
    }

    /// Recompute the canonical payload and its domain identity.
    pub fn validate(&self) -> FocrResult<()> {
        if self.schema_version != SELECTED_MUSIC_ROW_RECEIPT_SCHEMA_VERSION
            || self.contract_id != SELECTED_MUSIC_ROW_RECEIPT_CONTRACT_ID
            || self.canonical_encoding != SELECTED_MUSIC_ROW_RECEIPT_CANONICAL_ENCODING
            || self.staff_line_coordinate_contract != TROMR_STAFF_LINE_COORDINATE_CONTRACT
            || self.split_policy != TROMR_SELECTED_ROW_SPLIT_POLICY
            || self.candidate_evidence_storage_contract != TROMR_CANDIDATE_EVIDENCE_STORAGE_CONTRACT
            || self.tokenizer_filenames != TOKENIZER_FILENAMES
        {
            return Err(selected_row_error("receipt frozen literal mismatch"));
        }
        self.global_deskew.validate()?;
        self.row_refinement.validate()?;
        if self.selected_page_one_based == 0
            || self.selected_page_one_based > self.page_count
            || self.forward_input_count != 1
            || self.candidate_forward_input_count != self.forward_input_count
            || self.candidate_evidence_bundle.byte_len == 0
            || self.legacy_row_ordinal_one_based
                != self
                    .successful_fragment_ordinal_zero_based
                    .saturating_add(1)
            || self.selected_warning_count
                != u32::try_from(self.selected_warning_parent_indices.len()).unwrap_or(u32::MAX)
            || self.selected_warning_count > self.parent_warning_count
            || self
                .selected_warning_parent_indices
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .selected_warning_parent_indices
                .iter()
                .any(|index| *index >= self.parent_warning_count)
            || self.parent_ledger_identity.byte_len == 0
        {
            return Err(selected_row_error(
                "receipt closed-world scalar invariant failed",
            ));
        }
        let rebuilt = self.expected_canonical_bytes();
        if rebuilt != self.canonical_bytes
            || self.expected_canonical_identity() != self.canonical_identity
        {
            return Err(selected_row_error(
                "receipt canonical bytes or identity do not match fields",
            ));
        }
        Ok(())
    }

    /// Validate this selected-row receipt as an exact projection of one
    /// reconstructible parent recognition ledger.
    ///
    /// [`Self::validate`] proves only that the selected receipt is internally
    /// canonical. This gate additionally binds every selected provenance,
    /// geometry, transform, line, Gray8, semantic, MusicXML, and warning field
    /// to the parent receipt. It therefore rejects a caller who changes and
    /// recanonicalizes only the selected layer while leaving the parent fixed.
    /// Like the unsigned parent receipt itself, this proves cross-layer
    /// consistency rather than hostile-storage authenticity.
    pub fn validate_against_parent_receipt(
        &self,
        parent: &ImmutableMusicParentLedgerReceiptV1,
    ) -> FocrResult<()> {
        use crate::native_engine::tromr::{
            StaffInferenceOutcome, TromrRecognitionOptionsV1, TromrSplitPolicyV1,
        };

        self.validate()?;
        parent.validate()?;
        let fields = &parent.fields;
        if self.parent_ledger_identity != parent.canonical_identity {
            return Err(selected_row_error(
                "selected receipt does not bind the supplied parent-ledger identity",
            ));
        }

        let recognition_options =
            TromrRecognitionOptionsV1::from_json(&fields.recognition_options_canonical_json)?;
        if recognition_options.split_policy != TromrSplitPolicyV1::Disabled
            || self.split_policy != TROMR_SELECTED_ROW_SPLIT_POLICY
        {
            return Err(selected_row_error(
                "selected receipt is not backed by a disabled-split parent recognition",
            ));
        }
        if self.selected_page_one_based != fields.selected_page_one_based
            || self.page_count != fields.page_count
            || self.detected_staff_count != fields.detected_staff_count
            || self.staff_segmentation_disposition != fields.staff_segmentation_disposition
            || self.source_kind != fields.source_kind
            || self.source != fields.source
            || self.model != fields.model
            || self.tokenizer_filenames != fields.tokenizer_filenames
            || self.tokenizers != fields.tokenizers
            || self.raster_width != fields.raster_width
            || self.raster_height != fields.raster_height
            || self.raster_sha256 != fields.raster_sha256
            || self.bundle_sha256 != fields.bundle_sha256
            || self.recognition_options_identity != fields.recognition_options_identity
            || self.execution_options_identity != fields.execution_options_identity
            || self.recognition_options_sha256 != fields.recognition_options_sha256
            || self.execution_options_sha256 != fields.execution_options_sha256
            || self.options_sha256 != fields.options_sha256
            || self.parent_replay_sha256 != fields.replay_sha256
        {
            return Err(selected_row_error(
                "selected source, page, model, tokenizer, option, or replay projection differs from parent",
            ));
        }

        let detection_index = usize::try_from(self.detection_index)
            .map_err(|_| selected_row_error("selected detection index exceeds usize"))?;
        let attempt = fields
            .attempts
            .get(detection_index)
            .ok_or_else(|| selected_row_error("selected detection has no parent attempt"))?;
        let detector_crop = fields
            .staff_detection
            .crops
            .get(detection_index)
            .ok_or_else(|| selected_row_error("selected detection has no parent detector crop"))?;
        let (fragment_ordinal, fragment) = fields
            .fragments
            .iter()
            .enumerate()
            .find(|(_, fragment)| fragment.detection_index == self.detection_index)
            .ok_or_else(|| selected_row_error("selected detection has no parent fragment"))?;
        let fragment_ordinal_u32 = u32::try_from(fragment_ordinal)
            .map_err(|_| selected_row_error("parent fragment ordinal exceeds u32"))?;
        if self.successful_fragment_ordinal_zero_based != fragment_ordinal_u32
            || self.legacy_row_ordinal_one_based != fragment_ordinal_u32.saturating_add(1)
            || attempt.detection_index != self.detection_index
            || attempt.route != self.inference_route
            || attempt.outcome != StaffInferenceOutcome::Recognized
            || attempt.reason.is_some()
            || fragment.semantic != self.semantic
            || fragment.musicxml != self.musicxml
            || fragment.candidate_evidence_storage_contract
                != self.candidate_evidence_storage_contract
            || fragment.candidate_forward_input_count != self.candidate_forward_input_count
            || fragment.candidate_evidence_bundle != self.candidate_evidence_bundle
            || fragment.bbox_xywh != self.review_crop_source_bbox_xywh_in_globally_deskewed_raster
        {
            return Err(selected_row_error(
                "selected index, route, fragment, semantic, MusicXML, or ordinal differs from parent",
            ));
        }

        let [forward] = attempt.forward_inputs.as_slice() else {
            return Err(selected_row_error(
                "selected parent attempt does not contain exactly one forward input",
            ));
        };
        let review_geometry = attempt.review_crop_geometry.ok_or_else(|| {
            selected_row_error("selected parent attempt has no review-crop geometry")
        })?;
        let review_gray8 = attempt.review_crop_gray8.ok_or_else(|| {
            selected_row_error("selected parent attempt has no review-crop Gray8 identity")
        })?;
        let staff_lines = attempt.staff_lines.ok_or_else(|| {
            selected_row_error("selected parent attempt has no staff-line evidence")
        })?;
        let model_canvas_wh = [
            u32::try_from(forward.gray8.width)
                .map_err(|_| selected_row_error("parent model width exceeds u32"))?,
            u32::try_from(forward.gray8.height)
                .map_err(|_| selected_row_error("parent model height exceeds u32"))?,
        ];
        let review_canvas_wh = [
            u32::try_from(review_gray8.width)
                .map_err(|_| selected_row_error("parent review width exceeds u32"))?,
            u32::try_from(review_gray8.height)
                .map_err(|_| selected_row_error("parent review height exceeds u32"))?,
        ];
        let detector_geometry = detector_geometry_as_parent(detector_crop.geometry)?;
        let detector_accepted_lines = checked_lines(
            "selected parent detector lines",
            detector_crop.globally_deskewed_raster_lines,
        )?;
        let detector_review_lines = checked_lines(
            "selected parent review lines",
            detector_crop.review_crop_staff_lines_y_in_canvas,
        )?;
        if self.forward_input_count != 1
            || self.model_input_source_space != forward.source_space
            || self.model_input_source_bbox_xywh != forward.source_bbox_xywh
            || self.model_input_canvas_wh != model_canvas_wh
            || self.model_input_padding_trbl != forward.padding_trbl
            || self.model_input_staff_lines_y_in_canvas != forward.staff_lines_y_in_canvas
            || self.model_input_gray8 != selected_gray8_artifact_identity(forward.gray8)
            || self.review_crop_source_bbox_xywh_in_globally_deskewed_raster
                != review_geometry.source_bbox_xywh
            || self.review_crop_canvas_wh != review_geometry.canvas_wh
            || self.review_crop_canvas_wh != review_canvas_wh
            || self.review_crop_padding_trbl != review_geometry.padding_trbl
            || self.review_crop_gray8 != selected_gray8_artifact_identity(review_gray8)
        {
            return Err(selected_row_error(
                "selected model/review bbox, canvas, padding, or exact Gray8 identity differs from parent",
            ));
        }
        if self.global_deskew != fields.staff_detection.global_deskew
            || self.row_refinement != detector_crop.row_refinement
            || detector_geometry.source_bbox_xywh
                != self.review_crop_source_bbox_xywh_in_globally_deskewed_raster
            || detector_geometry.canvas_wh != self.review_crop_canvas_wh
            || detector_geometry.padding_trbl != self.review_crop_padding_trbl
            || selected_gray8_artifact_identity(detector_crop.review_crop_gray8)
                != self.review_crop_gray8
            || self.accepted_detector_lines_y_in_globally_deskewed_raster
                != staff_lines.accepted_detector_lines_y_in_globally_deskewed_raster
            || self.review_crop_staff_lines_y_in_canvas
                != staff_lines.review_crop_staff_lines_y_in_canvas
            || self.accepted_detector_lines_y_in_globally_deskewed_raster != detector_accepted_lines
            || self.review_crop_staff_lines_y_in_canvas != detector_review_lines
            || self.staff_lines_identity
                != selected_music_staff_lines_identity_v1(
                    staff_lines.accepted_detector_lines_y_in_globally_deskewed_raster,
                    staff_lines.review_crop_staff_lines_y_in_canvas,
                    forward.staff_lines_y_in_canvas,
                )?
        {
            return Err(selected_row_error(
                "selected detector transforms, crop geometry, or staff-line projection differs from parent",
            ));
        }

        let parent_warnings = parent_music_warnings(fields)?;
        let expected_warnings = selected_warning_evidence(
            &parent_warnings,
            fragment_ordinal
                .checked_add(1)
                .ok_or_else(|| selected_row_error("selected warning ordinal overflow"))?,
        )?;
        if self.parent_warning_count != u32::try_from(parent_warnings.len()).unwrap_or(u32::MAX)
            || self.selected_warning_count
                != u32::try_from(expected_warnings.selected_warnings.len()).unwrap_or(u32::MAX)
            || self.selected_warning_parent_indices != expected_warnings.selected_parent_indices
            || self.warnings != expected_warnings.canonical_identity
        {
            return Err(selected_row_error(
                "selected warning census or identity differs from parent",
            ));
        }
        Ok(())
    }
}

impl ImmutableSelectedMusicRow {
    #[must_use]
    pub fn semantic(&self) -> &str {
        &self.semantic
    }

    #[must_use]
    pub fn musicxml(&self) -> &str {
        &self.musicxml
    }

    #[must_use]
    pub fn model_input_gray8(&self) -> &crate::preprocess::staff_detect::TromrGray8CropV1 {
        &self.model_input_gray8
    }

    #[must_use]
    pub fn model_input_png(&self) -> &[u8] {
        &self.model_input_png
    }

    #[must_use]
    pub fn review_crop_gray8(&self) -> &crate::preprocess::staff_detect::TromrGray8CropV1 {
        &self.review_crop_gray8
    }

    #[must_use]
    pub fn review_crop_png(&self) -> &[u8] {
        &self.review_crop_png
    }

    #[must_use]
    pub const fn staff_lines(&self) -> &crate::native_engine::tromr::TromrStaffLineEvidenceV1 {
        &self.staff_lines
    }

    /// Exact same-forward candidate evidence whose bytes and CAS identity are
    /// bound by both the selected and parent receipts.
    #[must_use]
    pub const fn candidate_evidence(&self) -> &ImmutableMusicCandidateEvidenceBundleV1 {
        &self.candidate_evidence
    }

    #[must_use]
    pub const fn warnings(&self) -> &SelectedMusicWarnings {
        &self.warnings
    }

    #[must_use]
    pub const fn receipt(&self) -> &SelectedMusicRowReceipt {
        &self.receipt
    }

    /// Exact reconstructible parent ledger used by this selected-row
    /// aggregate's cross-layer validation. The originating
    /// [`ImmutableMusicRecognition`] exposes the same receipt for independent
    /// transport alongside [`ValidatedSelectedMusicRowParts`].
    #[must_use]
    pub const fn parent_ledger_receipt(&self) -> &ImmutableMusicParentLedgerReceiptV1 {
        &self.parent_ledger_receipt
    }

    /// Recompute every selected artifact and canonical receipt identity.
    /// Callers may use this immediately before copying public fields across an
    /// FFI/serialization boundary; `select_row` also calls it before return.
    pub fn validate(&self) -> FocrResult<()> {
        self.receipt
            .validate_against_parent_receipt(&self.parent_ledger_receipt)?;
        self.candidate_evidence
            .validate_against_identity(self.receipt.candidate_evidence_bundle)?;
        if self.candidate_evidence.storage_contract
            != self.receipt.candidate_evidence_storage_contract
            || u32::try_from(self.candidate_evidence.forward_candidate_lattices.len())
                .unwrap_or(u32::MAX)
                != self.receipt.candidate_forward_input_count
        {
            return Err(selected_row_error(
                "selected candidate-evidence contract or count is inconsistent",
            ));
        }
        self.model_input_gray8.validate()?;
        self.review_crop_gray8.validate()?;
        if self.semantic.is_empty() {
            return Err(selected_row_error("selected semantic stream is empty"));
        }
        let decoded_model = crate::preprocess::staff_detect::TromrGray8CropV1::from_lossless_png(
            &self.model_input_png,
        )?;
        let decoded_review = crate::preprocess::staff_detect::TromrGray8CropV1::from_lossless_png(
            &self.review_crop_png,
        )?;
        if decoded_model != self.model_input_gray8 || decoded_review != self.review_crop_gray8 {
            return Err(selected_row_error(
                "selected PNG does not decode to its retained Gray8 artifact",
            ));
        }
        crate::preprocess::staff_detect::verify_tromr_gray8_artifact_identity_v1(
            self.model_input_gray8.pixels(),
            self.model_input_gray8.width(),
            self.model_input_gray8.height(),
            tromr_gray8_identity_from_selected(
                self.receipt.model_input_gray8,
                self.model_input_gray8.width(),
                self.model_input_gray8.height(),
            )?,
        )?;
        crate::preprocess::staff_detect::verify_tromr_gray8_artifact_identity_v1(
            self.review_crop_gray8.pixels(),
            self.review_crop_gray8.width(),
            self.review_crop_gray8.height(),
            tromr_gray8_identity_from_selected(
                self.receipt.review_crop_gray8,
                self.review_crop_gray8.width(),
                self.review_crop_gray8.height(),
            )?,
        )?;
        if verify_selected_music_artifact_identity_v1(
            SelectedMusicArtifactRoleV1::ModelInputPng,
            &self.model_input_png,
            self.receipt.model_input_png,
        )
        .is_err()
            || verify_selected_music_artifact_identity_v1(
                SelectedMusicArtifactRoleV1::ReviewCropPng,
                &self.review_crop_png,
                self.receipt.review_crop_png,
            )
            .is_err()
            || verify_selected_music_artifact_identity_v1(
                SelectedMusicArtifactRoleV1::Semantic,
                self.semantic.as_bytes(),
                self.receipt.semantic,
            )
            .is_err()
            || verify_selected_music_artifact_identity_v1(
                SelectedMusicArtifactRoleV1::MusicXml,
                self.musicxml.as_bytes(),
                self.receipt.musicxml,
            )
            .is_err()
        {
            return Err(selected_row_error(
                "selected artifact identity does not match retained bytes",
            ));
        }
        let expected_musicxml = crate::native_engine::tromr::semantic_to_musicxml(&self.semantic)?;
        if expected_musicxml != self.musicxml
            || !crate::native_engine::tromr::validate_musicxml(&self.musicxml).is_empty()
        {
            return Err(selected_row_error(
                "selected MusicXML is not the validated provider projection of semantic",
            ));
        }
        let selected_part = usize::try_from(self.receipt.legacy_row_ordinal_one_based)
            .map_err(|_| selected_row_error("legacy row ordinal exceeds usize"))?;
        let expected_warnings =
            selected_warning_evidence(&self.warnings.parent_warnings, selected_part)?;
        if expected_warnings != self.warnings
            || self.receipt.parent_warning_count
                != u32::try_from(self.warnings.parent_warnings.len()).unwrap_or(u32::MAX)
            || self.receipt.selected_warning_count
                != u32::try_from(self.warnings.selected_warnings.len()).unwrap_or(u32::MAX)
            || self.receipt.selected_warning_parent_indices != self.warnings.selected_parent_indices
            || self.receipt.warnings != self.warnings.canonical_identity
        {
            return Err(selected_row_error(
                "selected warning mapping or identity is inconsistent",
            ));
        }
        let accepted = checked_lines(
            "accepted detector lines",
            self.staff_lines
                .accepted_detector_lines_y_in_globally_deskewed_raster,
        )?;
        let review = checked_lines(
            "review crop staff lines",
            self.staff_lines.review_crop_staff_lines_y_in_canvas,
        )?;
        if accepted
            != self
                .receipt
                .accepted_detector_lines_y_in_globally_deskewed_raster
            || review != self.receipt.review_crop_staff_lines_y_in_canvas
            || selected_music_staff_lines_identity_v1(
                accepted,
                review,
                self.receipt.model_input_staff_lines_y_in_canvas,
            )? != self.receipt.staff_lines_identity
        {
            return Err(selected_row_error(
                "selected staff-line evidence or identity is inconsistent",
            ));
        }
        Ok(())
    }

    /// Validate the complete aggregate and transfer its owned parts without
    /// cloning. The returned value has passed the same gate used by
    /// [`ImmutableMusicRecognition::select_row`].
    pub fn into_validated_parts(self) -> FocrResult<ValidatedSelectedMusicRowParts> {
        self.validate()?;
        Ok(ValidatedSelectedMusicRowParts {
            semantic: self.semantic,
            musicxml: self.musicxml,
            model_input_gray8: self.model_input_gray8,
            model_input_png: self.model_input_png,
            review_crop_gray8: self.review_crop_gray8,
            review_crop_png: self.review_crop_png,
            staff_lines: self.staff_lines,
            candidate_evidence: self.candidate_evidence,
            warnings: self.warnings,
            receipt: self.receipt,
        })
    }
}

fn selected_warning_evidence(
    parent: &[crate::native_engine::tromr::MusicWarning],
    selected_part: usize,
) -> FocrResult<SelectedMusicWarnings> {
    let mut canonical = Vec::new();
    append_canonical_field(
        &mut canonical,
        SELECTED_MUSIC_WARNINGS_CANONICAL_ENCODING.as_bytes(),
    );
    canonical.extend_from_slice(&checked_u32("warning count", parent.len())?.to_le_bytes());
    let mut selected_parent_indices = Vec::new();
    let mut selected_warnings = Vec::new();
    for (index, warning) in parent.iter().enumerate() {
        canonical.extend_from_slice(&checked_u32("warning index", index)?.to_le_bytes());
        append_canonical_field(&mut canonical, warning.kind.as_bytes());
        canonical.extend_from_slice(&checked_u32("warning part", warning.part)?.to_le_bytes());
        canonical
            .extend_from_slice(&checked_u32("warning measure", warning.measure)?.to_le_bytes());
        append_canonical_field(&mut canonical, warning.detail.as_bytes());
        if warning.part == selected_part {
            selected_parent_indices.push(checked_u32("selected warning index", index)?);
            selected_warnings.push(warning.clone());
        }
    }
    canonical.extend_from_slice(
        &checked_u32("selected warning count", selected_parent_indices.len())?.to_le_bytes(),
    );
    for index in &selected_parent_indices {
        canonical.extend_from_slice(&index.to_le_bytes());
    }
    Ok(SelectedMusicWarnings {
        parent_warnings: parent.to_vec(),
        selected_parent_indices,
        selected_warnings,
        canonical_encoding: SELECTED_MUSIC_WARNINGS_CANONICAL_ENCODING,
        canonical_identity: selected_music_artifact_identity_v1(
            SelectedMusicArtifactRoleV1::SelectedWarningsCanonical,
            &canonical,
        ),
        canonical_bytes: canonical,
    })
}

fn parent_music_warnings(
    fields: &ImmutableMusicParentLedgerFieldsV1,
) -> FocrResult<Vec<crate::native_engine::tromr::MusicWarning>> {
    fields
        .warnings
        .iter()
        .map(|warning| {
            Ok(crate::native_engine::tromr::MusicWarning {
                kind: canonical_music_warning_kind(&warning.kind)?,
                part: usize::try_from(warning.part)
                    .map_err(|_| selected_row_error("parent warning part exceeds usize"))?,
                measure: usize::try_from(warning.measure)
                    .map_err(|_| selected_row_error("parent warning measure exceeds usize"))?,
                detail: warning.detail.to_owned(),
            })
        })
        .collect()
}

fn canonical_music_warning_kind(value: &str) -> FocrResult<&'static str> {
    match value {
        "overfull_bar" => Ok("overfull_bar"),
        "underfull_bar" => Ok("underfull_bar"),
        "impossible_duration" => Ok("impossible_duration"),
        "key_mismatch" => Ok("key_mismatch"),
        _ => Err(parent_ledger_error(format!(
            "unknown music warning kind {value:?}"
        ))),
    }
}

fn validate_parent_review_evidence(
    entry: &crate::native_engine::tromr::StaffInferenceEvidence,
    expected_bbox: Option<(usize, usize, usize, usize)>,
    provenance: &MusicInputProvenance,
) -> FocrResult<()> {
    let review = entry
        .review_crop_gray8
        .as_ref()
        .ok_or_else(|| selected_row_error("detector-backed parent attempt lacks review pixels"))?;
    review.validate()?;
    let geometry = entry.review_crop_geometry.ok_or_else(|| {
        selected_row_error("detector-backed parent attempt lacks review geometry")
    })?;
    let lines = entry
        .staff_lines
        .ok_or_else(|| selected_row_error("detector-backed parent attempt lacks line evidence"))?;
    if expected_bbox != Some(geometry.source_bbox)
        || geometry.canvas_width != review.width()
        || geometry.canvas_height != review.height()
        || geometry.source_bbox.2 == 0
        || geometry.source_bbox.3 == 0
    {
        return Err(selected_row_error(
            "parent review bbox/canvas disagrees with row ledger or pixels",
        ));
    }
    let (x, y, width, height) = geometry.source_bbox;
    if x.checked_add(width)
        .is_none_or(|right| right > provenance.raster_width as usize)
        || y.checked_add(height)
            .is_none_or(|bottom| bottom > provenance.raster_height as usize)
        || width
            .checked_add(geometry.padding.left)
            .and_then(|value| value.checked_add(geometry.padding.right))
            != Some(geometry.canvas_width)
        || height
            .checked_add(geometry.padding.top)
            .and_then(|value| value.checked_add(geometry.padding.bottom))
            != Some(geometry.canvas_height)
    {
        return Err(selected_row_error(
            "parent review bbox/padding lies outside raster or canvas",
        ));
    }
    let accepted = lines.accepted_detector_lines_y_in_globally_deskewed_raster;
    let review_lines = lines.review_crop_staff_lines_y_in_canvas;
    if !accepted.windows(2).all(|pair| pair[0] < pair[1])
        || accepted.iter().any(|line| *line < y || *line >= y + height)
        || !review_lines.windows(2).all(|pair| pair[0] < pair[1])
        || review_lines
            .iter()
            .any(|line| *line < geometry.padding.top || *line >= geometry.padding.top + height)
    {
        return Err(selected_row_error(
            "parent staff lines are unordered or outside source ink bounds",
        ));
    }
    Ok(())
}

fn review_unpadded_identity(
    review: &crate::preprocess::staff_detect::TromrGray8CropV1,
    geometry: crate::preprocess::staff_detect::StaffCropGeometry,
) -> FocrResult<crate::preprocess::staff_detect::TromrGray8ArtifactIdentityV1> {
    let source_width = geometry.source_bbox.2;
    let source_height = geometry.source_bbox.3;
    let mut pixels = Vec::with_capacity(
        source_width
            .checked_mul(source_height)
            .ok_or_else(|| selected_row_error("review inner-crop dimensions overflow"))?,
    );
    for row in 0..source_height {
        let canvas_row = geometry
            .padding
            .top
            .checked_add(row)
            .ok_or_else(|| selected_row_error("review inner-crop row overflow"))?;
        let start = canvas_row
            .checked_mul(review.width())
            .and_then(|value| value.checked_add(geometry.padding.left))
            .ok_or_else(|| selected_row_error("review inner-crop offset overflow"))?;
        let end = start
            .checked_add(source_width)
            .ok_or_else(|| selected_row_error("review inner-crop end overflow"))?;
        pixels.extend_from_slice(
            review
                .pixels()
                .get(start..end)
                .ok_or_else(|| selected_row_error("review inner-crop lies outside pixels"))?,
        );
    }
    Ok(
        crate::preprocess::staff_detect::TromrGray8CropV1::from_tightly_packed(
            pixels,
            source_width,
            source_height,
        )?
        .artifact_identity(),
    )
}

fn validate_parent_music_ledgers(
    meta: &crate::native_engine::MusicPageMeta,
    provenance: &MusicInputProvenance,
) -> FocrResult<()> {
    use crate::native_engine::tromr::StaffInferenceOutcome;

    meta.staff_detection.validate()?;
    if meta.staff_detection.crops.len() != meta.detected_staff_count
        || meta.staff_detection.global_deskew.input_gray8.width
            != u64::from(provenance.raster_width)
        || meta.staff_detection.global_deskew.input_gray8.height
            != u64::from(provenance.raster_height)
    {
        return Err(selected_row_error(
            "parent detector report disagrees with detected count or selected raster shape",
        ));
    }

    let expected_attempts = if meta.detected_staff_count < 2 {
        1
    } else {
        meta.detected_staff_count
    };
    if meta.staff_evidence.len() != expected_attempts {
        return Err(selected_row_error(format!(
            "parent attempt ledger has {} entries, expected {expected_attempts}",
            meta.staff_evidence.len()
        )));
    }
    if meta.fragments.len() != meta.staves.len() {
        return Err(selected_row_error(
            "parent fragment and recognized-staff ledgers differ in length",
        ));
    }
    match meta.detected_staff_count {
        0 | 1 if meta.fragments.len() != 1 || meta.staves.len() != 1 || !meta.skips.is_empty() => {
            return Err(selected_row_error(
                "whole-raster parent route must contain exactly one recognized result and no skips",
            ));
        }
        2.. if meta.fragments.len() + meta.skips.len() != meta.detected_staff_count => {
            return Err(selected_row_error(
                "parent recognized plus skipped census is not closed",
            ));
        }
        _ => {}
    }
    let strictly_increasing_indices = |indices: &[usize]| {
        indices.windows(2).all(|pair| pair[0] < pair[1])
            && indices
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                == indices.len()
    };
    let fragment_indices: Vec<_> = meta
        .fragments
        .iter()
        .map(|fragment| fragment.detection_index)
        .collect();
    let staff_indices: Vec<_> = meta.staves.iter().map(|(index, _)| *index).collect();
    let skip_indices: Vec<_> = meta.skips.iter().map(|skip| skip.index).collect();
    let evidence_indices: Vec<_> = meta
        .staff_evidence
        .iter()
        .map(|entry| entry.index)
        .collect();
    if !strictly_increasing_indices(&fragment_indices)
        || !strictly_increasing_indices(&staff_indices)
        || !strictly_increasing_indices(&skip_indices)
        || !strictly_increasing_indices(&evidence_indices)
        || fragment_indices != staff_indices
    {
        return Err(selected_row_error(
            "parent row ledgers are not ordered, unique, and mutually aligned",
        ));
    }
    if evidence_indices != (0..expected_attempts).collect::<Vec<_>>() {
        return Err(selected_row_error(
            "parent attempt indices are not the exact contiguous detector census",
        ));
    }
    for (fragment, (_, bbox)) in meta.fragments.iter().zip(&meta.staves) {
        if fragment.bbox != *bbox {
            return Err(selected_row_error(
                "parent fragment bbox disagrees with recognized-staff bbox",
            ));
        }
        let attempt = meta
            .staff_evidence
            .get(fragment.detection_index)
            .ok_or_else(|| selected_row_error("parent fragment has no matching attempt"))?;
        let candidate_evidence = ImmutableMusicCandidateEvidenceBundleV1::reconstruct(
            fragment.forward_candidate_lattices.clone(),
        )?;
        if candidate_evidence.forward_candidate_lattices.len() != attempt.forward_inputs.len() {
            return Err(selected_row_error(
                "parent fragment candidate evidence does not account for every forward input",
            ));
        }
    }
    for entry in &meta.staff_evidence {
        let recognized = fragment_indices.binary_search(&entry.index).is_ok();
        let skipped = skip_indices.binary_search(&entry.index).is_ok();
        if recognized == skipped
            || (recognized && entry.outcome != StaffInferenceOutcome::Recognized)
            || (recognized && entry.reason.is_some())
            || (skipped && entry.outcome != StaffInferenceOutcome::Skipped)
            || (skipped && entry.reason.is_none())
        {
            return Err(selected_row_error(
                "parent attempt outcome/reason disagrees with recognized/skip ledgers",
            ));
        }
        for input in &entry.forward_inputs {
            input.gray8.validate()?;
            if input.source_bbox_xywh.2 == 0
                || input.source_bbox_xywh.3 == 0
                || input
                    .source_bbox_xywh
                    .0
                    .checked_add(input.source_bbox_xywh.2)
                    .is_none()
                || input
                    .source_bbox_xywh
                    .1
                    .checked_add(input.source_bbox_xywh.3)
                    .is_none()
            {
                return Err(selected_row_error(
                    "parent forward-input geometry is invalid",
                ));
            }
            match input.source_space {
                crate::native_engine::tromr::TromrModelInputSourceSpaceV1::SelectedPageRaster => {
                    if input.source_bbox_xywh
                        != (
                            0,
                            0,
                            provenance.raster_width as usize,
                            provenance.raster_height as usize,
                        )
                        || input.gray8.width() != provenance.raster_width as usize
                        || input.gray8.height() != provenance.raster_height as usize
                        || input.gray8.artifact_identity()
                            != meta.staff_detection.global_deskew.input_gray8
                        || input.padding != crate::preprocess::staff_detect::StaffPadding::default()
                        || input.staff_lines_y_in_canvas.is_some()
                    {
                        return Err(selected_row_error(
                            "selected-page forward input does not cover the exact unpadded raster",
                        ));
                    }
                }
                crate::native_engine::tromr::TromrModelInputSourceSpaceV1::ReviewCropCanvas => {
                    let review_geometry = entry.review_crop_geometry.ok_or_else(|| {
                        selected_row_error("review-canvas input has no review geometry")
                    })?;
                    if input.source_bbox_xywh.1 != 0
                        || input.source_bbox_xywh.2 != input.gray8.width()
                        || input.source_bbox_xywh.3 != input.gray8.height()
                        || input
                            .source_bbox_xywh
                            .0
                            .checked_add(input.gray8.width())
                            .is_none_or(|right| right > review_geometry.canvas_width)
                        || input.gray8.height() != review_geometry.canvas_height
                        || input.padding != review_geometry.padding
                    {
                        return Err(selected_row_error(
                            "review-canvas forward input geometry/padding is inconsistent",
                        ));
                    }
                }
            }
            if input.staff_lines_y_in_canvas.is_some_and(|lines| {
                !lines.windows(2).all(|pair| pair[0] < pair[1])
                    || lines.iter().any(|line| {
                        *line < input.padding.top
                            || *line >= input.gray8.height().saturating_sub(input.padding.bottom)
                    })
            }) {
                return Err(selected_row_error(
                    "parent forward-input staff lines are invalid",
                ));
            }
        }
        if recognized && entry.forward_inputs.is_empty() {
            return Err(selected_row_error(
                "recognized parent attempt has no exact forward input",
            ));
        }
        let expected_bbox = meta
            .fragments
            .iter()
            .find(|fragment| fragment.detection_index == entry.index)
            .map(|fragment| fragment.bbox)
            .or_else(|| {
                meta.skips
                    .iter()
                    .find(|skip| skip.index == entry.index)
                    .map(|skip| skip.bbox)
            });
        let full_raster_geometry = crate::preprocess::staff_detect::StaffCropGeometry::unpadded((
            0,
            0,
            provenance.raster_width as usize,
            provenance.raster_height as usize,
        ));
        match entry.route {
            crate::native_engine::tromr::TromrRowInferenceRouteV1::NoDetectedStaffWholeRasterFallback => {
                if meta.detected_staff_count != 0
                    || entry.index != 0
                    || entry.geometry != full_raster_geometry
                    || entry.forward_inputs.len() != 1
                    || entry.forward_inputs[0].source_space
                        != crate::native_engine::tromr::TromrModelInputSourceSpaceV1::SelectedPageRaster
                    || entry.forward_inputs[0].gray8.width() != provenance.raster_width as usize
                    || entry.forward_inputs[0].gray8.height() != provenance.raster_height as usize
                    || entry.review_crop_gray8.is_some()
                    || entry.review_crop_geometry.is_some()
                    || entry.staff_lines.is_some()
                {
                    return Err(selected_row_error(
                        "zero-detection parent route contains fabricated detector evidence",
                    ));
                }
            }
            crate::native_engine::tromr::TromrRowInferenceRouteV1::SingleDetectedStaffWholeRaster => {
                if meta.detected_staff_count != 1
                    || entry.index != 0
                    || entry.geometry != full_raster_geometry
                    || entry.forward_inputs.len() != 1
                    || entry.forward_inputs[0].source_space
                        != crate::native_engine::tromr::TromrModelInputSourceSpaceV1::SelectedPageRaster
                    || entry.forward_inputs[0].gray8.width() != provenance.raster_width as usize
                    || entry.forward_inputs[0].gray8.height() != provenance.raster_height as usize
                {
                    return Err(selected_row_error(
                        "single-detected parent route invariants failed",
                    ));
                }
                validate_parent_review_evidence(entry, expected_bbox, provenance)?;
            }
            crate::native_engine::tromr::TromrRowInferenceRouteV1::DetectedStaffCrop => {
                if meta.detected_staff_count < 2
                    || entry.index >= meta.detected_staff_count
                    || entry.forward_inputs.len() != 1
                    || entry.forward_inputs[0].source_space
                        != crate::native_engine::tromr::TromrModelInputSourceSpaceV1::ReviewCropCanvas
                {
                    return Err(selected_row_error(
                        "detected-crop parent route invariants failed",
                    ));
                }
                validate_parent_review_evidence(entry, expected_bbox, provenance)?;
                if entry.review_crop_geometry != Some(entry.geometry) {
                    return Err(selected_row_error(
                        "detected-crop parent attempt geometry differs from review geometry",
                    ));
                }
                if entry.review_crop_gray8.as_ref() != Some(&entry.forward_inputs[0].gray8) {
                    return Err(selected_row_error(
                        "detected-crop parent model/review pixels differ",
                    ));
                }
            }
            crate::native_engine::tromr::TromrRowInferenceRouteV1::ExperimentalSplitSegments => {
                if meta.detected_staff_count < 2
                    || entry.index >= meta.detected_staff_count
                    || meta.recognition_options.split_policy
                        != crate::native_engine::tromr::TromrSplitPolicyV1::ExperimentalBarlineSegments
                    || entry.forward_inputs.iter().any(|input| {
                        input.source_space
                            != crate::native_engine::tromr::TromrModelInputSourceSpaceV1::ReviewCropCanvas
                    })
                {
                    return Err(selected_row_error(
                        "experimental-split parent route invariants failed",
                    ));
                }
                validate_parent_review_evidence(entry, expected_bbox, provenance)?;
                if entry.review_crop_geometry != Some(entry.geometry) {
                    return Err(selected_row_error(
                        "experimental-split parent attempt geometry differs from review geometry",
                    ));
                }
            }
        }
        if !matches!(
            entry.route,
            crate::native_engine::tromr::TromrRowInferenceRouteV1::NoDetectedStaffWholeRasterFallback
        ) {
            let detector_crop = meta
                .staff_detection
                .crops
                .get(entry.index)
                .ok_or_else(|| selected_row_error("parent attempt has no detector crop"))?;
            let review = entry.review_crop_gray8.as_ref().ok_or_else(|| {
                selected_row_error("detector-backed parent attempt lacks review pixels")
            })?;
            let geometry = entry.review_crop_geometry.ok_or_else(|| {
                selected_row_error("detector-backed parent attempt lacks review geometry")
            })?;
            let lines = entry.staff_lines.ok_or_else(|| {
                selected_row_error("detector-backed parent attempt lacks line evidence")
            })?;
            if geometry != detector_crop.geometry
                || review.artifact_identity() != detector_crop.review_crop_gray8
                || lines.accepted_detector_lines_y_in_globally_deskewed_raster
                    != detector_crop.globally_deskewed_raster_lines
                || lines.review_crop_staff_lines_y_in_canvas
                    != detector_crop.review_crop_staff_lines_y_in_canvas
                || review_unpadded_identity(review, geometry)?
                    != detector_crop.row_refinement.refined_unpadded_crop_gray8
            {
                return Err(selected_row_error(
                    "attempt review evidence diverges from detector transform ledger",
                ));
            }
        }
    }
    for skip in &meta.skips {
        let entry = meta
            .staff_evidence
            .iter()
            .find(|entry| entry.index == skip.index)
            .ok_or_else(|| selected_row_error("skip has no matching attempt evidence"))?;
        if entry.reason.as_deref() != Some(skip.reason.as_str())
            || entry.geometry.source_bbox != skip.bbox
        {
            return Err(selected_row_error(
                "skip reason/bbox disagrees with matching attempt evidence",
            ));
        }
    }
    if provenance.selected_page == 0 || provenance.selected_page > provenance.page_count {
        return Err(selected_row_error(
            "parent selected page is outside page count",
        ));
    }
    let semantics: Vec<String> = meta
        .fragments
        .iter()
        .map(|fragment| fragment.semantic.clone())
        .collect();
    if crate::native_engine::tromr::sanity_warnings(&semantics) != meta.warnings {
        return Err(selected_row_error(
            "parent warning ledger is not the provider recomputation of fragment semantics",
        ));
    }
    Ok(())
}

fn validate_immutable_music_recognition_parent(
    meta: &crate::native_engine::MusicPageMeta,
    provenance: &MusicInputProvenance,
) -> FocrResult<()> {
    validate_parent_music_ledgers(meta, provenance)?;
    if meta.recognition_options != provenance.recognition_options
        || meta.recognition_options_identity != provenance.recognition_options_identity
    {
        return Err(selected_row_error(
            "page recognition options diverge from immutable provenance",
        ));
    }
    if !meta
        .staff_segmentation_disposition
        .is_consistent_with(meta.detected_staff_count)
    {
        return Err(selected_row_error(
            "staff segmentation disposition disagrees with detected count",
        ));
    }
    Ok(())
}

impl ImmutableMusicRecognition {
    /// Construct the opaque recognition only at the provider boundary after a
    /// complete forward. No public from-parts path exists.
    pub(crate) fn from_provider_output(
        musicxml: String,
        page_meta: crate::native_engine::MusicPageMeta,
        provenance: MusicInputProvenance,
    ) -> FocrResult<Self> {
        validate_immutable_music_recognition_parent(&page_meta, &provenance)?;
        let ledger_seal_sha256 =
            immutable_music_recognition_ledger_seal(&musicxml, &page_meta, &provenance)?;
        let parent_ledger_receipt =
            immutable_music_parent_ledger_receipt(&musicxml, &page_meta, &provenance)?;
        Ok(Self {
            musicxml,
            page_meta,
            provenance,
            ledger_seal_sha256,
            parent_ledger_receipt,
        })
    }

    /// Legacy combined MusicXML emitted by the same provider forward.
    /// Treat it as a review draft until [`Self::require_complete_for_publication`]
    /// succeeds; selected-row receipts remain valid when that gate refuses.
    #[must_use]
    pub fn musicxml(&self) -> &str {
        &self.musicxml
    }

    /// Read-only complete page ledger used by selected-row admission.
    #[must_use]
    pub const fn page_meta(&self) -> &crate::native_engine::MusicPageMeta {
        &self.page_meta
    }

    /// Read-only exact-consumption and replay provenance.
    #[must_use]
    pub const fn provenance(&self) -> &MusicInputProvenance {
        &self.provenance
    }

    /// Provider-private complete-ledger seal carried into every selected-row
    /// receipt. Callers may compare the digest across transfer boundaries but
    /// cannot mint a sealed recognition aggregate from it.
    #[must_use]
    pub const fn ledger_seal_sha256(&self) -> [u8; 32] {
        self.ledger_seal_sha256
    }

    /// Reconstructible complete parent-ledger receipt. This can be cloned or
    /// serialized independently of the opaque recognition aggregate.
    #[must_use]
    pub const fn parent_ledger_receipt(&self) -> &ImmutableMusicParentLedgerReceiptV1 {
        &self.parent_ledger_receipt
    }

    /// Materialize the exact candidate bundle omitted from the inline parent
    /// receipt so an embedder can persist it in CAS without rerunning OCR.
    pub fn candidate_evidence_for_fragment_ordinal(
        &self,
        successful_fragment_ordinal_zero_based: u32,
    ) -> FocrResult<ImmutableMusicCandidateEvidenceBundleV1> {
        self.validate_selected_row_parent()?;
        let ordinal = usize::try_from(successful_fragment_ordinal_zero_based)
            .map_err(|_| selected_row_error("candidate fragment ordinal exceeds usize"))?;
        let fragment = self.page_meta.fragments.get(ordinal).ok_or_else(|| {
            selected_row_error(format!(
                "candidate fragment ordinal {ordinal} is outside the {}-fragment parent census",
                self.page_meta.fragments.len()
            ))
        })?;
        let candidate_evidence = ImmutableMusicCandidateEvidenceBundleV1::reconstruct(
            fragment.forward_candidate_lattices.clone(),
        )?;
        self.parent_ledger_receipt
            .validate_candidate_evidence_for_fragment_ordinal(
                successful_fragment_ordinal_zero_based,
                &candidate_evidence,
            )?;
        Ok(candidate_evidence)
    }

    /// Revalidate the recognition and refuse complete-page publication while
    /// unresolved detector residuals remain. Row selection deliberately uses
    /// structural validation instead, retaining those residuals as page context.
    pub fn require_complete_for_publication(&self) -> FocrResult<()> {
        self.validate_selected_row_parent()?;
        self.page_meta.require_complete_for_publication()
    }

    /// Revalidate the opaque aggregate, then consume it into ordinary owned
    /// fields. There is deliberately no inverse constructor.
    pub fn into_owned_parts(self) -> FocrResult<ImmutableMusicRecognitionOwnedTuple> {
        self.validate_selected_row_parent()?;
        Ok((
            self.musicxml,
            self.page_meta,
            self.provenance,
            self.parent_ledger_receipt,
        ))
    }

    #[cfg(test)]
    fn reseal_after_test_fixture_mutation(&mut self) -> FocrResult<()> {
        self.ledger_seal_sha256 = immutable_music_recognition_ledger_seal(
            &self.musicxml,
            &self.page_meta,
            &self.provenance,
        )?;
        // Negative fixtures deliberately violate the public ledger. In that
        // case retain the prior receipt so the subsequent production gate,
        // rather than fixture setup, observes the mismatch.
        if let Ok(receipt) =
            immutable_music_parent_ledger_receipt(&self.musicxml, &self.page_meta, &self.provenance)
        {
            self.parent_ledger_receipt = receipt;
        }
        Ok(())
    }

    /// Validate the closed provider-owned parent ledger used by selected-row
    /// admission, without selecting or materializing any row artifact. This is
    /// structural validation, not a complete-page publication claim.
    ///
    /// Embedders must run this gate before classifying a requested detection
    /// index as merely unavailable. Otherwise a corrupt count or ledger could
    /// mask a provider-evidence failure as a normal out-of-range request.
    pub fn validate_selected_row_parent(&self) -> FocrResult<()> {
        let recomputed = immutable_music_recognition_ledger_seal(
            &self.musicxml,
            &self.page_meta,
            &self.provenance,
        )?;
        if recomputed != self.ledger_seal_sha256 {
            return Err(selected_row_error(
                "private provider ledger seal does not match the recognition aggregate",
            ));
        }
        self.parent_ledger_receipt.validate()?;
        let expected_receipt = immutable_music_parent_ledger_receipt(
            &self.musicxml,
            &self.page_meta,
            &self.provenance,
        )?;
        if expected_receipt != self.parent_ledger_receipt {
            return Err(parent_ledger_error(
                "receipt does not match the opaque recognition aggregate",
            ));
        }
        validate_immutable_music_recognition_parent(&self.page_meta, &self.provenance)
    }

    /// Select one detector-backed, non-split TrOMR row by detection index.
    ///
    /// This method revalidates every redundant provider ledger before it
    /// materializes exact Gray8/PNG review artifacts and row-local MusicXML.
    /// It makes no topology, persistent-part, publication, or quality claim.
    pub fn select_row(&self, detection_index: usize) -> FocrResult<ImmutableSelectedMusicRow> {
        use crate::native_engine::tromr::{
            StaffInferenceOutcome, TromrModelInputSourceSpaceV1, TromrRowInferenceRouteV1,
            TromrSplitPolicyV1, TromrStaffSegmentationDispositionV1,
        };

        let meta = &self.page_meta;
        self.validate_selected_row_parent()?;
        if meta.recognition_options.split_policy != TromrSplitPolicyV1::Disabled {
            return Err(selected_row_error(
                "selected-row receipts require split_policy=disabled",
            ));
        }
        if meta.fragments.len() != meta.staves.len() {
            return Err(selected_row_error(
                "fragment and recognized-staff ledgers have different lengths",
            ));
        }
        if meta.detected_staff_count >= 2
            && meta.fragments.len() + meta.skips.len() != meta.detected_staff_count
        {
            return Err(selected_row_error(
                "recognized plus skipped rows do not equal detected staff count",
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        for evidence in &meta.staff_evidence {
            if !seen.insert(evidence.index) {
                return Err(selected_row_error(
                    "duplicate staff-evidence detection index",
                ));
            }
        }

        let fragments: Vec<_> = meta
            .fragments
            .iter()
            .enumerate()
            .filter(|(_, fragment)| fragment.detection_index == detection_index)
            .collect();
        if fragments.len() != 1 {
            return Err(selected_row_error(format!(
                "detection index {detection_index} matched {} fragments, expected exactly one",
                fragments.len()
            )));
        }
        let (fragment_ordinal, fragment) = fragments[0];
        let staves: Vec<_> = meta
            .staves
            .iter()
            .filter(|(index, _)| *index == detection_index)
            .collect();
        let evidence: Vec<_> = meta
            .staff_evidence
            .iter()
            .filter(|entry| entry.index == detection_index)
            .collect();
        if staves.len() != 1 || evidence.len() != 1 {
            return Err(selected_row_error(format!(
                "detection index {detection_index} must match one staff and one attempt"
            )));
        }
        if meta.skips.iter().any(|skip| skip.index == detection_index) {
            return Err(selected_row_error(format!(
                "detection index {detection_index} appears in both recognized and skipped ledgers"
            )));
        }
        let evidence = evidence[0];
        if staves[0].1 != fragment.bbox {
            return Err(selected_row_error(
                "fragment bbox disagrees with recognized-staff bbox",
            ));
        }
        if evidence.outcome != StaffInferenceOutcome::Recognized || evidence.reason.is_some() {
            return Err(selected_row_error(
                "selected attempt is not an unqualified recognized outcome",
            ));
        }
        if evidence.forward_inputs.len() != 1 {
            return Err(selected_row_error(format!(
                "selected attempt forwarded {} inputs; exactly one is required",
                evidence.forward_inputs.len()
            )));
        }
        let forward = &evidence.forward_inputs[0];
        forward.gray8.validate()?;
        let review = evidence
            .review_crop_gray8
            .as_ref()
            .ok_or_else(|| selected_row_error("detector-backed review crop is absent"))?;
        review.validate()?;
        let review_geometry = evidence
            .review_crop_geometry
            .ok_or_else(|| selected_row_error("review-crop geometry is absent"))?;
        let staff_lines = evidence
            .staff_lines
            .ok_or_else(|| selected_row_error("detector-backed five-line evidence is absent"))?;
        if review_geometry.source_bbox != fragment.bbox
            || review_geometry.canvas_width != review.width()
            || review_geometry.canvas_height != review.height()
        {
            return Err(selected_row_error(
                "review crop geometry disagrees with fragment bbox or retained pixels",
            ));
        }
        if staff_lines
            .review_crop_staff_lines_y_in_canvas
            .iter()
            .any(|line| *line >= review.height())
        {
            return Err(selected_row_error(
                "review-crop staff line lies outside the retained canvas",
            ));
        }
        let (bbox_x, bbox_y, bbox_width, bbox_height) = fragment.bbox;
        if bbox_width == 0
            || bbox_height == 0
            || bbox_x
                .checked_add(bbox_width)
                .is_none_or(|right| right > self.provenance.raster_width as usize)
            || bbox_y
                .checked_add(bbox_height)
                .is_none_or(|bottom| bottom > self.provenance.raster_height as usize)
        {
            return Err(selected_row_error(
                "review source bbox lies outside the selected-page raster",
            ));
        }
        if bbox_width
            .checked_add(review_geometry.padding.left)
            .and_then(|value| value.checked_add(review_geometry.padding.right))
            != Some(review_geometry.canvas_width)
            || bbox_height
                .checked_add(review_geometry.padding.top)
                .and_then(|value| value.checked_add(review_geometry.padding.bottom))
                != Some(review_geometry.canvas_height)
        {
            return Err(selected_row_error(
                "review source bbox plus padding does not equal review canvas",
            ));
        }
        let strictly_ordered = |lines: &[usize; 5]| lines.windows(2).all(|pair| pair[0] < pair[1]);
        if !strictly_ordered(&staff_lines.accepted_detector_lines_y_in_globally_deskewed_raster)
            || staff_lines
                .accepted_detector_lines_y_in_globally_deskewed_raster
                .iter()
                .any(|line| *line >= self.provenance.raster_height as usize)
        {
            return Err(selected_row_error(
                "accepted detector lines are not strictly ordered in raster bounds",
            ));
        }
        if !strictly_ordered(&staff_lines.review_crop_staff_lines_y_in_canvas) {
            return Err(selected_row_error(
                "review-crop lines are not strictly ordered",
            ));
        }

        match evidence.route {
            TromrRowInferenceRouteV1::NoDetectedStaffWholeRasterFallback => {
                return Err(selected_row_error(
                    "zero-detection fallback has no selectable detector-backed row",
                ));
            }
            TromrRowInferenceRouteV1::ExperimentalSplitSegments => {
                return Err(selected_row_error(
                    "experimental split rows are not selectable by this contract",
                ));
            }
            TromrRowInferenceRouteV1::SingleDetectedStaffWholeRaster => {
                if meta.detected_staff_count != 1
                    || detection_index != 0
                    || meta.staff_segmentation_disposition
                        != TromrStaffSegmentationDispositionV1::SingleStaffDetectedWholeImageRecognition
                    || forward.source_space != TromrModelInputSourceSpaceV1::SelectedPageRaster
                    || forward.source_bbox_xywh
                        != (
                            0,
                            0,
                            self.provenance.raster_width as usize,
                            self.provenance.raster_height as usize,
                        )
                    || forward.padding != crate::preprocess::staff_detect::StaffPadding::default()
                    || forward.staff_lines_y_in_canvas.is_some()
                {
                    return Err(selected_row_error(
                        "single-detected whole-raster route invariants failed",
                    ));
                }
            }
            TromrRowInferenceRouteV1::DetectedStaffCrop => {
                if meta.detected_staff_count < 2
                    || detection_index >= meta.detected_staff_count
                    || meta.staff_segmentation_disposition
                        != TromrStaffSegmentationDispositionV1::MultipleStavesDetectedPerCropRecognition
                    || forward.source_space != TromrModelInputSourceSpaceV1::ReviewCropCanvas
                    || forward.source_bbox_xywh
                        != (0, 0, review.width(), review.height())
                    || forward.padding != review_geometry.padding
                    || forward.staff_lines_y_in_canvas
                        != Some(staff_lines.review_crop_staff_lines_y_in_canvas)
                    || forward.gray8 != *review
                    || evidence.geometry != review_geometry
                {
                    return Err(selected_row_error(
                        "detected-staff-crop route invariants failed",
                    ));
                }
            }
        }

        if fragment.semantic.is_empty() {
            return Err(selected_row_error("selected semantic stream is empty"));
        }
        let musicxml = crate::native_engine::tromr::semantic_to_musicxml(&fragment.semantic)?;
        let violations = crate::native_engine::tromr::validate_musicxml(&musicxml);
        if !violations.is_empty() {
            return Err(selected_row_error(format!(
                "provider-generated row MusicXML failed validation: {}",
                violations.join("; ")
            )));
        }
        let model_input_png = forward.gray8.to_lossless_png()?;
        let review_crop_png = review.to_lossless_png()?;
        if crate::preprocess::staff_detect::TromrGray8CropV1::from_lossless_png(&model_input_png)?
            != forward.gray8
            || crate::preprocess::staff_detect::TromrGray8CropV1::from_lossless_png(
                &review_crop_png,
            )? != *review
        {
            return Err(selected_row_error(
                "provider PNG round-trip changed selected Gray8 pixels",
            ));
        }
        let warnings = selected_warning_evidence(&meta.warnings, fragment_ordinal + 1)?;
        for warning in &warnings.parent_warnings {
            if !matches!(
                warning.kind,
                "overfull_bar" | "underfull_bar" | "impossible_duration" | "key_mismatch"
            ) || warning.part == 0
                || warning.part > meta.fragments.len()
                || warning.detail.is_empty()
            {
                return Err(selected_row_error(
                    "parent warning ledger contains an invalid warning",
                ));
            }
        }

        let model_gray_identity = SelectedMusicArtifactIdentity {
            byte_len: u64::try_from(forward.gray8.pixels().len()).unwrap_or(u64::MAX),
            sha256: forward.gray8.pixels_sha256(),
            blake3: forward.gray8.pixels_blake3(),
            domain_identity_sha256: forward.gray8.identity_sha256(),
        };
        let review_gray_identity = SelectedMusicArtifactIdentity {
            byte_len: u64::try_from(review.pixels().len()).unwrap_or(u64::MAX),
            sha256: review.pixels_sha256(),
            blake3: review.pixels_blake3(),
            domain_identity_sha256: review.identity_sha256(),
        };
        let model_png_identity = selected_music_artifact_identity_v1(
            SelectedMusicArtifactRoleV1::ModelInputPng,
            &model_input_png,
        );
        let review_png_identity = selected_music_artifact_identity_v1(
            SelectedMusicArtifactRoleV1::ReviewCropPng,
            &review_crop_png,
        );
        let semantic_identity = selected_music_artifact_identity_v1(
            SelectedMusicArtifactRoleV1::Semantic,
            fragment.semantic.as_bytes(),
        );
        let musicxml_identity = selected_music_artifact_identity_v1(
            SelectedMusicArtifactRoleV1::MusicXml,
            musicxml.as_bytes(),
        );
        let candidate_evidence = ImmutableMusicCandidateEvidenceBundleV1::reconstruct(
            fragment.forward_candidate_lattices.clone(),
        )?;
        if candidate_evidence.forward_candidate_lattices.len() != evidence.forward_inputs.len() {
            return Err(selected_row_error(
                "selected candidate evidence does not account for every exact forward input",
            ));
        }

        let selected_page_one_based = checked_u32("selected page", self.provenance.selected_page)?;
        let page_count = checked_u32("page count", self.provenance.page_count)?;
        let detection_index_u32 = checked_u32("detection index", detection_index)?;
        let successful_fragment_ordinal_zero_based =
            checked_u32("successful fragment ordinal", fragment_ordinal)?;
        let legacy_row_ordinal_one_based = checked_u32(
            "legacy row ordinal",
            fragment_ordinal
                .checked_add(1)
                .ok_or_else(|| selected_row_error("legacy row ordinal arithmetic overflow"))?,
        )?;
        let detected_staff_count = checked_u32("detected staff count", meta.detected_staff_count)?;
        let model_bbox = checked_bbox("model input source bbox", forward.source_bbox_xywh)?;
        let review_bbox = checked_bbox("review crop source bbox", fragment.bbox)?;
        let model_wh = [
            checked_u32("model input width", forward.gray8.width())?,
            checked_u32("model input height", forward.gray8.height())?,
        ];
        let review_wh = [
            checked_u32("review crop width", review.width())?,
            checked_u32("review crop height", review.height())?,
        ];
        let model_padding = checked_padding("model input padding", forward.padding)?;
        let review_padding = checked_padding("review crop padding", review_geometry.padding)?;
        let accepted_lines = checked_lines(
            "accepted detector lines",
            staff_lines.accepted_detector_lines_y_in_globally_deskewed_raster,
        )?;
        let review_lines = checked_lines(
            "review crop staff lines",
            staff_lines.review_crop_staff_lines_y_in_canvas,
        )?;
        let model_lines = forward
            .staff_lines_y_in_canvas
            .map(|lines| checked_lines("model input staff lines", lines))
            .transpose()?;
        let staff_lines_identity =
            selected_music_staff_lines_identity_v1(accepted_lines, review_lines, model_lines)?;

        let mut canonical = Vec::new();
        append_canonical_field(
            &mut canonical,
            SELECTED_MUSIC_ROW_RECEIPT_CONTRACT_ID.as_bytes(),
        );
        canonical.extend_from_slice(&SELECTED_MUSIC_ROW_RECEIPT_SCHEMA_VERSION.to_le_bytes());
        append_canonical_field(
            &mut canonical,
            SELECTED_MUSIC_ROW_RECEIPT_CANONICAL_ENCODING.as_bytes(),
        );
        canonical.extend_from_slice(&selected_page_one_based.to_le_bytes());
        canonical.extend_from_slice(&page_count.to_le_bytes());
        canonical.extend_from_slice(&detection_index_u32.to_le_bytes());
        canonical.extend_from_slice(&successful_fragment_ordinal_zero_based.to_le_bytes());
        canonical.extend_from_slice(&legacy_row_ordinal_one_based.to_le_bytes());
        canonical.extend_from_slice(&detected_staff_count.to_le_bytes());
        append_canonical_field(
            &mut canonical,
            meta.staff_segmentation_disposition.as_str().as_bytes(),
        );
        append_canonical_field(&mut canonical, evidence.route.as_str().as_bytes());
        let detector_crop = meta
            .staff_detection
            .crops
            .get(detection_index)
            .ok_or_else(|| selected_row_error("selected row has no detector crop evidence"))?;
        append_selected_transform_evidence(
            &mut canonical,
            meta.staff_detection.global_deskew,
            detector_crop.row_refinement,
        );
        append_canonical_field(&mut canonical, TROMR_SELECTED_ROW_SPLIT_POLICY.as_bytes());
        canonical.extend_from_slice(&1u32.to_le_bytes());
        append_canonical_field(&mut canonical, forward.source_space.as_str().as_bytes());
        for values in [model_bbox, review_bbox] {
            for value in values {
                canonical.extend_from_slice(&value.to_le_bytes());
            }
        }
        for values in [
            &model_wh[..],
            &model_padding[..],
            &review_wh[..],
            &review_padding[..],
        ] {
            for value in values {
                canonical.extend_from_slice(&value.to_le_bytes());
            }
        }
        for values in [accepted_lines, review_lines] {
            for value in values {
                canonical.extend_from_slice(&value.to_le_bytes());
            }
        }
        match model_lines {
            Some(lines) => {
                canonical.push(1);
                for value in lines {
                    canonical.extend_from_slice(&value.to_le_bytes());
                }
            }
            None => canonical.push(0),
        }
        append_canonical_field(
            &mut canonical,
            TROMR_STAFF_LINE_COORDINATE_CONTRACT.as_bytes(),
        );
        append_selected_identity(&mut canonical, staff_lines_identity);
        append_canonical_field(
            &mut canonical,
            self.provenance.source_kind.as_str().as_bytes(),
        );
        append_consumed_identity(&mut canonical, self.provenance.source);
        append_consumed_identity(&mut canonical, self.provenance.model);
        for (filename, tokenizer) in TOKENIZER_FILENAMES.iter().zip(self.provenance.tokenizers) {
            append_canonical_field(&mut canonical, filename.as_bytes());
            append_consumed_identity(&mut canonical, tokenizer);
        }
        canonical.extend_from_slice(&self.provenance.raster_width.to_le_bytes());
        canonical.extend_from_slice(&self.provenance.raster_height.to_le_bytes());
        canonical.extend_from_slice(&self.provenance.raster_sha256);
        canonical.extend_from_slice(&self.provenance.bundle_sha256);
        append_canonical_field(
            &mut canonical,
            self.provenance.recognition_options_identity.as_bytes(),
        );
        append_canonical_field(
            &mut canonical,
            self.provenance.execution_options_identity.as_bytes(),
        );
        canonical.extend_from_slice(&self.provenance.recognition_options_sha256);
        canonical.extend_from_slice(&self.provenance.execution_options_sha256);
        canonical.extend_from_slice(&self.provenance.options_sha256);
        canonical.extend_from_slice(&self.provenance.replay_sha256);
        append_selected_identity(
            &mut canonical,
            self.parent_ledger_receipt.canonical_identity,
        );
        for identity in [
            model_gray_identity,
            model_png_identity,
            review_gray_identity,
            review_png_identity,
            semantic_identity,
            musicxml_identity,
        ] {
            append_selected_identity(&mut canonical, identity);
        }
        append_canonical_field(
            &mut canonical,
            TROMR_CANDIDATE_EVIDENCE_STORAGE_CONTRACT.as_bytes(),
        );
        let candidate_forward_input_count = checked_u32(
            "selected candidate forward-input count",
            candidate_evidence.forward_candidate_lattices.len(),
        )?;
        canonical.extend_from_slice(&candidate_forward_input_count.to_le_bytes());
        append_selected_identity(&mut canonical, candidate_evidence.canonical_identity);
        let parent_warning_count =
            checked_u32("parent warning count", warnings.parent_warnings.len())?;
        let selected_warning_count =
            checked_u32("selected warning count", warnings.selected_warnings.len())?;
        canonical.extend_from_slice(&parent_warning_count.to_le_bytes());
        canonical.extend_from_slice(&selected_warning_count.to_le_bytes());
        canonical.extend_from_slice(
            &checked_u32(
                "selected warning parent-index count",
                warnings.selected_parent_indices.len(),
            )?
            .to_le_bytes(),
        );
        for index in &warnings.selected_parent_indices {
            canonical.extend_from_slice(&index.to_le_bytes());
        }
        append_selected_identity(&mut canonical, warnings.canonical_identity);
        let canonical_identity = selected_artifact_identity(SELECTED_ROW_DOMAIN, &canonical);
        let receipt = SelectedMusicRowReceipt {
            schema_version: SELECTED_MUSIC_ROW_RECEIPT_SCHEMA_VERSION,
            contract_id: SELECTED_MUSIC_ROW_RECEIPT_CONTRACT_ID,
            canonical_encoding: SELECTED_MUSIC_ROW_RECEIPT_CANONICAL_ENCODING,
            selected_page_one_based,
            page_count,
            detection_index: detection_index_u32,
            successful_fragment_ordinal_zero_based,
            legacy_row_ordinal_one_based,
            detected_staff_count,
            staff_segmentation_disposition: meta.staff_segmentation_disposition,
            inference_route: evidence.route,
            global_deskew: meta.staff_detection.global_deskew,
            row_refinement: detector_crop.row_refinement,
            split_policy: TROMR_SELECTED_ROW_SPLIT_POLICY,
            forward_input_count: 1,
            model_input_source_space: forward.source_space,
            model_input_source_bbox_xywh: model_bbox,
            review_crop_source_bbox_xywh_in_globally_deskewed_raster: review_bbox,
            model_input_canvas_wh: model_wh,
            model_input_padding_trbl: model_padding,
            review_crop_canvas_wh: review_wh,
            review_crop_padding_trbl: review_padding,
            accepted_detector_lines_y_in_globally_deskewed_raster: accepted_lines,
            review_crop_staff_lines_y_in_canvas: review_lines,
            model_input_staff_lines_y_in_canvas: model_lines,
            staff_line_coordinate_contract: TROMR_STAFF_LINE_COORDINATE_CONTRACT,
            staff_lines_identity,
            source_kind: self.provenance.source_kind,
            source: self.provenance.source,
            model: self.provenance.model,
            tokenizer_filenames: TOKENIZER_FILENAMES,
            tokenizers: self.provenance.tokenizers,
            raster_width: self.provenance.raster_width,
            raster_height: self.provenance.raster_height,
            raster_sha256: self.provenance.raster_sha256,
            bundle_sha256: self.provenance.bundle_sha256,
            recognition_options_identity: self.provenance.recognition_options_identity.clone(),
            execution_options_identity: self.provenance.execution_options_identity.clone(),
            recognition_options_sha256: self.provenance.recognition_options_sha256,
            execution_options_sha256: self.provenance.execution_options_sha256,
            options_sha256: self.provenance.options_sha256,
            parent_replay_sha256: self.provenance.replay_sha256,
            parent_ledger_identity: self.parent_ledger_receipt.canonical_identity,
            model_input_gray8: model_gray_identity,
            model_input_png: model_png_identity,
            review_crop_gray8: review_gray_identity,
            review_crop_png: review_png_identity,
            semantic: semantic_identity,
            musicxml: musicxml_identity,
            candidate_evidence_storage_contract: TROMR_CANDIDATE_EVIDENCE_STORAGE_CONTRACT,
            candidate_forward_input_count,
            candidate_evidence_bundle: candidate_evidence.canonical_identity,
            parent_warning_count,
            selected_warning_count,
            selected_warning_parent_indices: warnings.selected_parent_indices.clone(),
            warnings: warnings.canonical_identity,
            canonical_bytes: canonical,
            canonical_identity,
        };
        let selected = ImmutableSelectedMusicRow {
            semantic: fragment.semantic.clone(),
            musicxml,
            model_input_gray8: forward.gray8.clone(),
            model_input_png,
            review_crop_gray8: review.clone(),
            review_crop_png,
            staff_lines,
            candidate_evidence,
            warnings,
            receipt,
            parent_ledger_receipt: self.parent_ledger_receipt.clone(),
        };
        selected.validate()?;
        Ok(selected)
    }
}

/// Successful immutable input preparation with runtime-only measurements.
pub struct PreparedImmutableMusicInput {
    pub bundle: ImmutableMusicInputBundle,
    pub diagnostics: MusicInputPreparationDiagnosticsV1,
}

/// Failed immutable input preparation. No partially prepared bundle escapes.
#[derive(Debug)]
pub struct MusicInputPreparationFailure {
    pub error: FocrError,
    /// Boxed so a diagnostics-rich failure remains cheap to return by value.
    pub diagnostics: Box<MusicInputPreparationDiagnosticsV1>,
}

/// Successful TrOMR recognition plus noncanonical runtime diagnostics.
#[derive(Debug)]
pub struct ImmutableMusicRecognitionWithDiagnostics {
    pub recognition: ImmutableMusicRecognition,
    pub diagnostics: TromrExecutionDiagnosticsV1,
}

/// Failed TrOMR recognition plus noncanonical runtime diagnostics.
///
/// Deliberately carries no MusicXML or page metadata.
#[derive(Debug)]
pub struct ImmutableMusicRecognitionFailure {
    pub error: FocrError,
    /// Boxed so a diagnostics-rich failure remains cheap to return by value.
    pub diagnostics: Box<TromrExecutionDiagnosticsV1>,
}

/// Fully pinned TrOMR inputs. Fields are private so callers cannot replace a
/// component after its identity is computed.
pub struct ImmutableMusicInputBundle {
    source_bytes: Arc<[u8]>,
    raster: DynamicImage,
    model: Arc<OcrModel>,
    recognition_options: TromrRecognitionOptionsV1,
    execution_options: TromrExecutionOptionsV1,
    provenance: MusicInputProvenance,
}

impl ImmutableMusicInputBundle {
    /// Open every required artifact once, then construct the immutable bundle
    /// from the owned bytes read through those descriptors.
    ///
    /// The model resolver may select a concrete artifact path, but hashing,
    /// parsing, and inference all use the single owned model buffer read after
    /// resolution. The four tokenizer paths are each opened once beside that
    /// resolved model. Later path replacement cannot affect this bundle.
    pub fn open(
        source_path: &Path,
        model_path: &Path,
        options: MusicInputOptions,
    ) -> FocrResult<Self> {
        Self::open_observed(source_path, model_path, options)
            .map_or_else(|failure| Err(failure.error), |prepared| Ok(prepared.bundle))
    }

    /// Open an immutable bundle and return noncanonical preparation timings.
    ///
    /// A failure contains diagnostics but never exposes partially parsed input.
    pub fn open_observed(
        source_path: &Path,
        model_path: &Path,
        options: MusicInputOptions,
    ) -> Result<PreparedImmutableMusicInput, MusicInputPreparationFailure> {
        let mut diagnostics = MusicInputPreparationDiagnosticsBuilder::new();
        let result =
            Self::open_with_diagnostics(source_path, model_path, options, &mut diagnostics);
        let finished = diagnostics.finish(&result);
        match result {
            Ok(bundle) => Ok(PreparedImmutableMusicInput {
                bundle,
                diagnostics: finished,
            }),
            Err(error) => Err(MusicInputPreparationFailure {
                error,
                diagnostics: Box::new(finished),
            }),
        }
    }

    fn open_with_diagnostics(
        source_path: &Path,
        model_path: &Path,
        options: MusicInputOptions,
        diagnostics: &mut MusicInputPreparationDiagnosticsBuilder,
    ) -> FocrResult<Self> {
        let source_started = std::time::Instant::now();
        let source_result = read_owned_file(
            source_path,
            "music source",
            MAX_MUSIC_SOURCE_BYTES,
            ReadRole::Source,
        );
        diagnostics.record_source_read(source_started.elapsed());
        let source_bytes = source_result?;
        if let Some(page) = options.page
            && !source_bytes.starts_with(b"%PDF-")
        {
            return Err(FocrError::Usage(format!(
                "TrOMR page {page} was requested for an image source; page selection is PDF-only"
            )));
        }
        let resolved_model = OcrModel::resolve_model(model_path)?;
        let model_started = std::time::Instant::now();
        let model_result = read_owned_file(
            &resolved_model,
            "TrOMR model",
            MAX_MUSIC_MODEL_BYTES,
            ReadRole::Model,
        );
        diagnostics.record_model_read(model_started.elapsed());
        let model_bytes = model_result?;
        let model_dir = resolved_model.parent().unwrap_or_else(|| Path::new("."));
        let mut tokenizer_bytes: [Vec<u8>; 4] = std::array::from_fn(|_| Vec::new());
        for (index, filename) in TOKENIZER_FILENAMES.iter().enumerate() {
            let started = std::time::Instant::now();
            let result = read_owned_file(
                &model_dir.join(filename),
                filename,
                MAX_MUSIC_TOKENIZER_BYTES,
                ReadRole::Model,
            );
            diagnostics.record_tokenizer_read(index, started.elapsed());
            tokenizer_bytes[index] = result?;
        }
        Self::from_owned_parts_with_label(
            source_bytes,
            model_bytes,
            tokenizer_bytes,
            resolved_model,
            options,
            diagnostics,
        )
    }

    /// Construct from already-owned source/model/tokenizer buffers. No path is
    /// opened, and all five buffers are hashed before being moved into the exact
    /// parser/inference objects that retain them.
    pub fn from_owned_parts(
        source_bytes: Vec<u8>,
        model_bytes: Vec<u8>,
        tokenizer_bytes: [Vec<u8>; 4],
        options: MusicInputOptions,
    ) -> FocrResult<Self> {
        let mut diagnostics = MusicInputPreparationDiagnosticsBuilder::new();
        Self::from_owned_parts_with_label(
            source_bytes,
            model_bytes,
            tokenizer_bytes,
            PathBuf::from("<owned-tromr-model>"),
            options,
            &mut diagnostics,
        )
    }

    fn from_owned_parts_with_label(
        source_bytes: Vec<u8>,
        model_bytes: Vec<u8>,
        tokenizer_bytes: [Vec<u8>; 4],
        model_path_label: PathBuf,
        options: MusicInputOptions,
        diagnostics: &mut MusicInputPreparationDiagnosticsBuilder,
    ) -> FocrResult<Self> {
        validate_nonempty_and_bounded("music source", &source_bytes, MAX_MUSIC_SOURCE_BYTES)?;
        validate_nonempty_and_bounded("TrOMR model", &model_bytes, MAX_MUSIC_MODEL_BYTES)?;
        for (filename, bytes) in TOKENIZER_FILENAMES.iter().zip(&tokenizer_bytes) {
            validate_nonempty_and_bounded(filename, bytes, MAX_MUSIC_TOKENIZER_BYTES)?;
        }
        let recognition_options = options.recognition.validate()?;
        let recognition_options_json = recognition_options.canonical_json()?;
        let recognition_options_identity = recognition_options.replay_identity()?;
        let execution_options = options.execution.validate()?;
        let execution_options_json = execution_options.canonical_json()?;
        let execution_options_identity = execution_options.replay_identity()?;

        let source = ConsumedBytesIdentity::of(&source_bytes);
        let model_identity = ConsumedBytesIdentity::of(&model_bytes);
        let tokenizers = tokenizer_bytes
            .each_ref()
            .map(|bytes| ConsumedBytesIdentity::of(bytes));
        verify_expected("music source", source, options.expectations.source_sha256)?;
        verify_expected(
            "TrOMR model",
            model_identity,
            options.expectations.model_sha256,
        )?;
        for index in 0..TOKENIZER_FILENAMES.len() {
            verify_expected(
                TOKENIZER_FILENAMES[index],
                tokenizers[index],
                options.expectations.tokenizer_sha256[index],
            )?;
        }

        let source_bytes: Arc<[u8]> = source_bytes.into();
        verify_retained_identity("music source", source, &source_bytes)?;
        let source_kind = if source_bytes.starts_with(b"%PDF-") {
            MusicSourceKind::Pdf
        } else {
            MusicSourceKind::Image
        };
        let (raster, page_count, selected_page) = match source_kind {
            MusicSourceKind::Pdf => {
                let parse_started = std::time::Instant::now();
                let pages_result =
                    crate::pdf::PdfPages::from_shared_bytes(Arc::clone(&source_bytes));
                diagnostics.record_pdf_parse(parse_started.elapsed());
                let pages = pages_result?;
                verify_retained_identity("music PDF source", source, pages.source_bytes())?;
                let selected_page = options.page.unwrap_or(1);
                if selected_page == 0 || selected_page > pages.len() {
                    return Err(FocrError::Usage(format!(
                        "TrOMR PDF page {selected_page} is outside 1..={}",
                        pages.len()
                    )));
                }
                let page_count = pages.len();
                let raster_started = std::time::Instant::now();
                let raster_result = pages.render(selected_page - 1);
                diagnostics.record_page_raster(raster_started.elapsed());
                let raster = raster_result?;
                (raster, page_count, selected_page)
            }
            MusicSourceKind::Image => {
                if let Some(page) = options.page {
                    return Err(FocrError::Usage(format!(
                        "TrOMR page {page} was requested for an image source; page selection is PDF-only"
                    )));
                }
                let decode_started = std::time::Instant::now();
                let decoded = crate::preprocess::decode_bytes(&source_bytes);
                diagnostics.record_image_decode(decode_started.elapsed());
                (decoded?, 1, 1)
            }
        };

        let tokenizer_started = std::time::Instant::now();
        let tokenizer_result = MusicTokenizer::from_owned_tables(tokenizer_bytes);
        diagnostics.record_tokenizer_parse(tokenizer_started.elapsed());
        let tokenizer = tokenizer_result?;
        for ((filename, stream), expected) in TOKENIZER_FILENAMES
            .iter()
            .zip([Stream::Rhythm, Stream::Pitch, Stream::Lift, Stream::Note])
            .zip(tokenizers)
        {
            verify_retained_identity(filename, expected, tokenizer.source_bytes(stream))?;
        }
        let model_started = std::time::Instant::now();
        let weights_result = Weights::from_bytes(model_bytes);
        diagnostics.record_model_parse(model_started.elapsed());
        let weights = weights_result?;
        verify_retained_identity("TrOMR model", model_identity, weights.source_bytes())?;
        let model = OcrModel::from_owned_tromr_parts(model_path_label, weights, tokenizer)?;

        let bundle_sha256 = bundle_digest(source, model_identity, &tokenizers);
        let raster_sha256 = raster_digest(&raster);
        let recognition_options_sha256 =
            component_options_digest(RECOGNITION_OPTIONS_DOMAIN, &recognition_options_json);
        let execution_options_sha256 =
            component_options_digest(EXECUTION_OPTIONS_DOMAIN, &execution_options_json);
        let options_sha256 = options_digest(recognition_options_sha256, execution_options_sha256);
        let replay_sha256 = replay_digest(bundle_sha256, raster_sha256, options_sha256);
        let provenance = MusicInputProvenance {
            source_kind,
            source,
            model: model_identity,
            tokenizers,
            page_count,
            selected_page,
            raster_width: raster.width(),
            raster_height: raster.height(),
            recognition_options,
            recognition_options_identity,
            execution_options,
            execution_options_identity,
            bundle_sha256,
            raster_sha256,
            recognition_options_sha256,
            execution_options_sha256,
            options_sha256,
            replay_sha256,
        };
        Ok(Self {
            source_bytes,
            raster,
            model,
            recognition_options,
            execution_options,
            provenance,
        })
    }

    #[must_use]
    pub fn provenance(&self) -> &MusicInputProvenance {
        &self.provenance
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        Arc<OcrModel>,
        DynamicImage,
        Arc<[u8]>,
        TromrRecognitionOptionsV1,
        TromrExecutionOptionsV1,
        MusicInputProvenance,
    ) {
        (
            self.model,
            self.raster,
            self.source_bytes,
            self.recognition_options,
            self.execution_options,
            self.provenance,
        )
    }
}

#[derive(Clone, Copy)]
enum ReadRole {
    Source,
    Model,
}

fn read_owned_file(
    path: &Path,
    label: &str,
    max_bytes: u64,
    role: ReadRole,
) -> FocrResult<Vec<u8>> {
    let file = std::fs::File::open(path).map_err(|error| match role {
        ReadRole::Source => {
            FocrError::InputDecode(format!("open {label} at {}: {error}", path.display()))
        }
        ReadRole::Model => {
            FocrError::ModelNotFound(format!("open {label} at {}: {error}", path.display()))
        }
    })?;
    let metadata = file.metadata().map_err(|error| {
        FocrError::FormatMismatch(format!(
            "inspect {label} at {} through its open descriptor: {error}",
            path.display()
        ))
    })?;
    if !metadata.is_file() {
        return Err(FocrError::FormatMismatch(format!(
            "{label} at {} is not a regular file",
            path.display()
        )));
    }
    read_bounded(file, label, max_bytes, Some(metadata.len()))
}

fn read_bounded<R: Read>(
    source: R,
    label: &str,
    max_bytes: u64,
    expected_len: Option<u64>,
) -> FocrResult<Vec<u8>> {
    if expected_len.is_some_and(|len| len == 0 || len > max_bytes) {
        return Err(FocrError::FormatMismatch(format!(
            "{label} descriptor length must be within 1..={max_bytes} bytes"
        )));
    }
    let mut bounded = source.take(max_bytes.saturating_add(1));
    let mut bytes = Vec::new();
    bounded
        .read_to_end(&mut bytes)
        .map_err(|error| FocrError::FormatMismatch(format!("read exact {label} bytes: {error}")))?;
    validate_nonempty_and_bounded(label, &bytes, max_bytes)?;
    let actual_len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if let Some(expected_len) = expected_len
        && actual_len != expected_len
    {
        return Err(FocrError::FormatMismatch(format!(
            "{label} changed while being pinned: descriptor reported {expected_len} bytes, read {actual_len}"
        )));
    }
    Ok(bytes)
}

fn validate_nonempty_and_bounded(label: &str, bytes: &[u8], max_bytes: u64) -> FocrResult<()> {
    let len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if len == 0 || len > max_bytes {
        return Err(FocrError::FormatMismatch(format!(
            "{label} byte length {len} is outside 1..={max_bytes}"
        )));
    }
    Ok(())
}

fn verify_expected(
    label: &str,
    actual: ConsumedBytesIdentity,
    expected: Option<[u8; 32]>,
) -> FocrResult<()> {
    if let Some(expected) = expected
        && actual.sha256 != expected
    {
        return Err(FocrError::FormatMismatch(format!(
            "{label} exact-consumption SHA-256 mismatch: expected {}, consumed {}",
            hex_sha256(&expected),
            actual.sha256_hex()
        )));
    }
    Ok(())
}

fn verify_retained_identity(
    label: &str,
    expected: ConsumedBytesIdentity,
    retained: &[u8],
) -> FocrResult<()> {
    let actual = ConsumedBytesIdentity::of(retained);
    if actual != expected {
        return Err(FocrError::Other(anyhow::anyhow!(
            "internal immutable-input contract violation: {label} retained {} bytes with SHA-256 {}, expected {} bytes with SHA-256 {}",
            actual.byte_len,
            actual.sha256_hex(),
            expected.byte_len,
            expected.sha256_hex()
        )));
    }
    Ok(())
}

fn update_field(hasher: &mut Sha256, value: &[u8]) {
    hasher.update(u64::try_from(value.len()).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(value);
}

fn update_identity(hasher: &mut Sha256, role: &[u8], identity: ConsumedBytesIdentity) {
    update_field(hasher, role);
    hasher.update(identity.byte_len.to_le_bytes());
    hasher.update(identity.sha256);
}

fn bundle_digest(
    source: ConsumedBytesIdentity,
    model: ConsumedBytesIdentity,
    tokenizers: &[ConsumedBytesIdentity; 4],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(BUNDLE_DOMAIN);
    update_identity(&mut hasher, b"source", source);
    update_identity(&mut hasher, b"model", model);
    for (filename, identity) in TOKENIZER_FILENAMES.iter().zip(tokenizers) {
        update_identity(&mut hasher, filename.as_bytes(), *identity);
    }
    hasher.finalize().into()
}

fn raster_digest(raster: &DynamicImage) -> [u8; 32] {
    let rgba = raster.to_rgba8();
    let mut hasher = Sha256::new();
    hasher.update(RASTER_DOMAIN);
    hasher.update(rgba.width().to_le_bytes());
    hasher.update(rgba.height().to_le_bytes());
    update_field(&mut hasher, rgba.as_raw());
    hasher.finalize().into()
}

fn component_options_digest(domain: &[u8], canonical_json: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    update_field(&mut hasher, canonical_json.as_bytes());
    hasher.finalize().into()
}

fn options_digest(recognition: [u8; 32], execution: [u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(OPTIONS_DOMAIN);
    hasher.update(recognition);
    hasher.update(execution);
    hasher.finalize().into()
}

fn replay_digest(bundle: [u8; 32], raster: [u8; 32], options: [u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(REPLAY_DOMAIN);
    hasher.update(bundle);
    hasher.update(raster);
    hasher.update(options);
    hasher.finalize().into()
}

fn hex_sha256(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_engine::tromr::TromrSplitPolicyV1;
    use crate::quant::focrq::FocrqBuilder;

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "franken_ocr_music_input_{label}_{}_{}",
            std::process::id(),
            nonce
        ))
    }

    fn tokenizer_bytes() -> [Vec<u8>; 4] {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tromr");
        TOKENIZER_FILENAMES
            .map(|filename| std::fs::read(root.join(filename)).expect("tokenizer fixture"))
    }

    fn tiny_png() -> Vec<u8> {
        let image = DynamicImage::ImageRgb8(image::RgbImage::from_fn(6, 4, |x, y| {
            image::Rgb([(x * 31) as u8, (y * 47) as u8, 113])
        }));
        let mut bytes = Vec::new();
        image
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .expect("encode tiny PNG");
        bytes
    }

    fn empty_tromr_model() -> Vec<u8> {
        let notice = crate::native_engine::model_arch::arch_by_id("tromr")
            .expect("TrOMR registry entry")
            .license_notice();
        FocrqBuilder::new()
            .with_model_id("tromr")
            .with_license_notice(notice)
            .build()
    }

    fn selected_test_provenance(width: u32, height: u32) -> MusicInputProvenance {
        let source = ConsumedBytesIdentity::of(b"selected-test-pdf");
        let model = ConsumedBytesIdentity::of(b"selected-test-model");
        let tokenizers = [
            ConsumedBytesIdentity::of(b"rhythm"),
            ConsumedBytesIdentity::of(b"pitch"),
            ConsumedBytesIdentity::of(b"lift"),
            ConsumedBytesIdentity::of(b"note"),
        ];
        let recognition_options = TromrRecognitionOptionsV1::deterministic();
        let execution_options = TromrExecutionOptionsV1::default();
        let recognition_json = recognition_options
            .canonical_json()
            .expect("recognition JSON");
        let execution_json = execution_options.canonical_json().expect("execution JSON");
        let bundle_sha256 = bundle_digest(source, model, &tokenizers);
        let raster = DynamicImage::ImageLuma8(image::GrayImage::from_pixel(
            width,
            height,
            image::Luma([255]),
        ));
        let raster_sha256 = raster_digest(&raster);
        let recognition_options_sha256 =
            component_options_digest(RECOGNITION_OPTIONS_DOMAIN, &recognition_json);
        let execution_options_sha256 =
            component_options_digest(EXECUTION_OPTIONS_DOMAIN, &execution_json);
        let options_sha256 = options_digest(recognition_options_sha256, execution_options_sha256);
        MusicInputProvenance {
            source_kind: MusicSourceKind::Pdf,
            source,
            model,
            tokenizers,
            page_count: 3,
            selected_page: 2,
            raster_width: width,
            raster_height: height,
            recognition_options,
            recognition_options_identity: recognition_options
                .replay_identity()
                .expect("recognition identity"),
            execution_options,
            execution_options_identity: execution_options
                .replay_identity()
                .expect("execution identity"),
            bundle_sha256,
            raster_sha256,
            recognition_options_sha256,
            execution_options_sha256,
            options_sha256,
            replay_sha256: replay_digest(bundle_sha256, raster_sha256, options_sha256),
        }
    }

    fn selected_test_row(
        index: usize,
        y: usize,
        semantic: &str,
    ) -> (
        crate::native_engine::MusicRowFragment,
        (usize, crate::native_engine::tromr::StaffBBox),
        crate::native_engine::tromr::StaffInferenceEvidence,
    ) {
        use crate::native_engine::tromr::{
            StaffInferenceEvidence, StaffInferenceOutcome, TromrForwardInputV1,
            TromrModelInputSourceSpaceV1, TromrRowInferenceRouteV1, TromrStaffLineEvidenceV1,
        };
        let bbox = (2, y, 8, 12);
        let pixels: Vec<u8> = (0..96).map(|value| (value % 251) as u8).collect();
        let gray8 =
            crate::preprocess::staff_detect::TromrGray8CropV1::from_tightly_packed(pixels, 8, 12)
                .expect("test crop");
        let geometry = crate::preprocess::staff_detect::StaffCropGeometry::unpadded(bbox);
        let canvas_lines = [1, 3, 5, 7, 9];
        (
            crate::native_engine::MusicRowFragment {
                detection_index: index,
                bbox,
                semantic: semantic.to_owned(),
                forward_candidate_lattices: vec![
                    crate::native_engine::tromr::TromrForwardCandidateLatticeV1::new(
                        0,
                        crate::native_engine::tromr::synthetic_candidate_lattice_for_test(1, true),
                    )
                    .expect("test candidate lattice"),
                ],
            },
            (index, bbox),
            StaffInferenceEvidence {
                index,
                geometry,
                route: TromrRowInferenceRouteV1::DetectedStaffCrop,
                forward_inputs: vec![TromrForwardInputV1 {
                    gray8: gray8.clone(),
                    source_space: TromrModelInputSourceSpaceV1::ReviewCropCanvas,
                    source_bbox_xywh: (0, 0, 8, 12),
                    padding: crate::preprocess::staff_detect::StaffPadding::default(),
                    staff_lines_y_in_canvas: Some(canvas_lines),
                }],
                review_crop_gray8: Some(gray8),
                review_crop_geometry: Some(geometry),
                staff_lines: Some(TromrStaffLineEvidenceV1 {
                    accepted_detector_lines_y_in_globally_deskewed_raster: [
                        y + 1,
                        y + 3,
                        y + 5,
                        y + 7,
                        y + 9,
                    ],
                    review_crop_staff_lines_y_in_canvas: canvas_lines,
                }),
                outcome: StaffInferenceOutcome::Recognized,
                reason: None,
            },
        )
    }

    fn selected_test_recognition() -> ImmutableMusicRecognition {
        let semantics = [
            "clef-G2+rest-quarter|rest-quarter+barline",
            "clef-F4+rest-quarter|rest-quarter+barline",
        ];
        let rows = [
            selected_test_row(0, 8, semantics[0]),
            selected_test_row(1, 30, semantics[1]),
        ];
        let fragments = rows.iter().map(|row| row.0.clone()).collect();
        let staves = rows.iter().map(|row| row.1).collect();
        let staff_evidence = rows.iter().map(|row| row.2.clone()).collect();
        let detector_crops = rows
            .iter()
            .map(|row| {
                let evidence = &row.2;
                let lines = evidence.staff_lines.expect("test row lines");
                (
                    evidence.review_crop_geometry.expect("test review geometry"),
                    lines.accepted_detector_lines_y_in_globally_deskewed_raster,
                    lines.review_crop_staff_lines_y_in_canvas,
                )
            })
            .collect::<Vec<_>>();
        let mut staff_detection =
            crate::preprocess::staff_detect::synthetic_complete_evidence_for_test(
                20,
                60,
                &detector_crops,
            );
        for (crop, row) in staff_detection.crops.iter_mut().zip(&rows) {
            let review = row.2.review_crop_gray8.as_ref().expect("test review crop");
            let identity = review.artifact_identity();
            crop.row_refinement.source_crop_before_refinement_gray8 = identity;
            crop.row_refinement.refined_unpadded_crop_gray8 = identity;
            crop.review_crop_gray8 = identity;
        }
        let recognition_options = TromrRecognitionOptionsV1::deterministic();
        ImmutableMusicRecognition::from_provider_output(
            "<score-partwise version=\"4.0\"/>".to_owned(),
            crate::native_engine::MusicPageMeta {
                detected_staff_count: 2,
                staff_segmentation_disposition: crate::native_engine::tromr::TromrStaffSegmentationDispositionV1::MultipleStavesDetectedPerCropRecognition,
                fragments,
                staves,
                skips: Vec::new(),
                staff_evidence,
                staff_detection,
                warnings: crate::native_engine::tromr::sanity_warnings(
                    &semantics.map(str::to_owned),
                ),
                recognition_options,
                recognition_options_identity: recognition_options
                    .replay_identity()
                    .expect("options identity"),
            },
            selected_test_provenance(20, 60),
        )
        .expect("valid provider test recognition")
    }

    fn refresh_test_staff_detection_from_attempts(recognition: &mut ImmutableMusicRecognition) {
        let detector_crops = recognition
            .page_meta
            .staff_evidence
            .iter()
            .take(recognition.page_meta.detected_staff_count)
            .map(|attempt| {
                let geometry = attempt
                    .review_crop_geometry
                    .expect("detector-backed test attempt geometry");
                let lines = attempt.staff_lines.expect("detector-backed test lines");
                (
                    geometry,
                    lines.accepted_detector_lines_y_in_globally_deskewed_raster,
                    lines.review_crop_staff_lines_y_in_canvas,
                )
            })
            .collect::<Vec<_>>();
        let mut staff_detection =
            crate::preprocess::staff_detect::synthetic_complete_evidence_for_test(
                recognition.provenance.raster_width as usize,
                recognition.provenance.raster_height as usize,
                &detector_crops,
            );
        for (crop, attempt) in staff_detection
            .crops
            .iter_mut()
            .zip(&recognition.page_meta.staff_evidence)
        {
            let review = attempt
                .review_crop_gray8
                .as_ref()
                .expect("detector-backed test review crop");
            let geometry = attempt
                .review_crop_geometry
                .expect("detector-backed test review geometry");
            let unpadded = review_unpadded_identity(review, geometry)
                .expect("derive detector test unpadded identity");
            crop.row_refinement.source_crop_before_refinement_gray8 = unpadded;
            crop.row_refinement.refined_unpadded_crop_gray8 = unpadded;
            crop.review_crop_gray8 = review.artifact_identity();
        }
        recognition.page_meta.staff_detection = staff_detection;
    }

    fn selected_test_recognition_with_unresolved_context() -> ImmutableMusicRecognition {
        let mut recognition = selected_test_recognition();
        let original = recognition.page_meta.staff_detection.clone();
        let detector_crops = original
            .crops
            .iter()
            .map(|crop| {
                (
                    crop.geometry,
                    crop.globally_deskewed_raster_lines,
                    crop.review_crop_staff_lines_y_in_canvas,
                )
            })
            .collect::<Vec<_>>();
        let mut unresolved =
            crate::preprocess::staff_detect::synthetic_unresolved_evidence_for_test(
                20,
                60,
                &detector_crops,
                [20, 22, 24, 26, 29],
            );
        unresolved.global_deskew = original.global_deskew;
        unresolved.crops = original.crops;
        unresolved
            .validate()
            .expect("unresolved selected-row context validates");
        recognition.page_meta.staff_detection = unresolved;
        recognition
            .reseal_after_test_fixture_mutation()
            .expect("reseal unresolved parent context");
        recognition
    }

    #[test]
    fn selected_row_materializes_exact_provider_artifacts_and_receipt() {
        let selected = selected_test_recognition()
            .select_row(1)
            .expect("select row");
        selected.validate().expect("aggregate validates");
        assert_eq!(
            selected.semantic(),
            "clef-F4+rest-quarter|rest-quarter+barline"
        );
        assert_eq!(
            crate::preprocess::staff_detect::TromrGray8CropV1::from_lossless_png(
                selected.review_crop_png()
            )
            .expect("strict provider PNG decode"),
            *selected.review_crop_gray8()
        );
        assert_eq!(
            selected.receipt().model_input_png.sha256,
            selected.receipt().review_crop_png.sha256,
            "detected route uses byte-identical PNGs"
        );
        assert_ne!(
            selected.receipt().model_input_png.domain_identity_sha256,
            selected.receipt().review_crop_png.domain_identity_sha256,
            "role-specific domains remain distinct even for equal bytes"
        );
        assert_eq!(
            selected_music_staff_lines_identity_v1(
                selected
                    .receipt()
                    .accepted_detector_lines_y_in_globally_deskewed_raster,
                selected.receipt().review_crop_staff_lines_y_in_canvas,
                selected.receipt().model_input_staff_lines_y_in_canvas,
            )
            .expect("derive typed staff-line identity"),
            selected.receipt().staff_lines_identity
        );
        assert!(
            selected_music_staff_lines_identity_v1([1, 2, 2, 4, 5], [1, 2, 3, 4, 5], None,)
                .is_err()
        );
        let parts = selected.into_validated_parts().expect("sealed extraction");
        assert_eq!(parts.receipt().detection_index, 1);
        assert_eq!(parts.receipt().selected_page_one_based, 2);
        assert_eq!(parts.receipt().page_count, 3);
        assert_eq!(parts.receipt().source_kind, MusicSourceKind::Pdf);
    }

    #[test]
    fn candidate_evidence_round_trips_through_cas_and_binds_both_receipts() {
        let recognition = selected_test_recognition();
        let bundle = recognition
            .candidate_evidence_for_fragment_ordinal(1)
            .expect("materialize exact candidate bundle");
        bundle.validate().expect("candidate bundle validates");
        assert_eq!(bundle.forward_candidate_lattices.len(), 1);
        assert!(!bundle.canonical_bytes.is_empty());
        assert!(bundle.canonical_bytes.len() <= MAX_MUSIC_CANDIDATE_EVIDENCE_BUNDLE_BYTES);
        eprintln!(
            "candidate_evidence_cas case=baseline bytes={} sha256={} blake3={} forwards={}",
            bundle.canonical_bytes.len(),
            hex_sha256(&bundle.canonical_identity.sha256),
            blake3::Hash::from_bytes(bundle.canonical_identity.blake3).to_hex(),
            bundle.forward_candidate_lattices.len(),
        );

        let mut cas = std::collections::BTreeMap::new();
        cas.insert(
            bundle.canonical_identity.blake3,
            bundle.canonical_bytes.clone(),
        );
        let recovered = ImmutableMusicCandidateEvidenceBundleV1::recover(
            cas.get(&bundle.canonical_identity.blake3)
                .expect("CAS object exists")
                .clone(),
        )
        .expect("recover exact candidate bundle from CAS bytes");
        assert_eq!(recovered, bundle);
        assert_eq!(
            ImmutableMusicCandidateEvidenceBundleV1::recover(recovered.canonical_bytes.clone())
                .expect("deterministic second recovery"),
            recovered,
        );
        recognition
            .parent_ledger_receipt()
            .validate_candidate_evidence_for_fragment_ordinal(1, &recovered)
            .expect("parent receipt accepts recovered CAS bytes");

        let selected = recognition.select_row(1).expect("select candidate row");
        assert_eq!(selected.candidate_evidence(), &recovered);
        assert_eq!(
            selected.receipt().candidate_evidence_storage_contract,
            TROMR_CANDIDATE_EVIDENCE_STORAGE_CONTRACT
        );
        assert_eq!(selected.receipt().candidate_forward_input_count, 1);
        assert_eq!(
            selected.receipt().candidate_evidence_bundle,
            recovered.canonical_identity
        );
        assert_eq!(
            selected.parent_ledger_receipt().fields.fragments[1].candidate_evidence_bundle,
            recovered.canonical_identity
        );

        let maximum_row = ImmutableMusicCandidateEvidenceBundleV1::reconstruct(vec![
            crate::native_engine::tromr::TromrForwardCandidateLatticeV1::new(
                0,
                crate::native_engine::tromr::synthetic_candidate_lattice_for_test(
                    crate::native_engine::tromr::MAX_SEQ,
                    false,
                ),
            )
            .expect("maximum-row lattice"),
        ])
        .expect("maximum legal row remains under bundle ceiling");
        eprintln!(
            "candidate_evidence_cas case=max_row bytes={} ceiling={}",
            maximum_row.canonical_bytes.len(),
            MAX_MUSIC_CANDIDATE_EVIDENCE_BUNDLE_BYTES,
        );
        assert!(maximum_row.canonical_bytes.len() <= MAX_MUSIC_CANDIDATE_EVIDENCE_BUNDLE_BYTES);
    }

    #[test]
    fn candidate_evidence_rejects_field_missing_truncated_and_cas_mutations() {
        type Mutation = fn(&mut ImmutableMusicCandidateEvidenceBundleV1);
        let selected = selected_test_recognition()
            .select_row(0)
            .expect("select candidate row");
        let baseline = selected.candidate_evidence().clone();
        let mutations: [(&str, Mutation); 5] = [
            ("score bits", |bundle| {
                bundle.forward_candidate_lattices[0]
                    .candidate_lattice
                    .positions[0]
                    .heads[0]
                    .chosen_model_score_f32_bits ^= 1;
            }),
            ("chosen rank", |bundle| {
                bundle.forward_candidate_lattices[0]
                    .candidate_lattice
                    .positions[0]
                    .heads[0]
                    .chosen_rank_one_based ^= 1;
            }),
            ("prefix", |bundle| {
                bundle.forward_candidate_lattices[0]
                    .candidate_lattice
                    .positions[0]
                    .prefix_sha256[0] ^= 1;
            }),
            ("truncation", |bundle| {
                bundle.forward_candidate_lattices[0]
                    .candidate_lattice
                    .positions[0]
                    .heads[0]
                    .truncated_candidate_count += 1;
            }),
            ("chosen stream", |bundle| {
                bundle.forward_candidate_lattices[0]
                    .candidate_lattice
                    .chosen_streams
                    .rhythm[0] ^= 1;
            }),
        ];
        for (label, mutate) in mutations {
            let mut changed = baseline.clone();
            mutate(&mut changed);
            eprintln!("candidate_evidence_mutation case={label}");
            assert!(
                changed.validate().is_err(),
                "{label} mutation must invalidate exact candidate evidence"
            );
        }

        assert!(ImmutableMusicCandidateEvidenceBundleV1::recover(Vec::new()).is_err());
        assert!(
            ImmutableMusicCandidateEvidenceBundleV1::recover(
                baseline.canonical_bytes[..baseline.canonical_bytes.len() / 2].to_vec(),
            )
            .is_err()
        );
        assert!(
            ImmutableMusicCandidateEvidenceBundleV1::recover(vec![
                b'x';
                MAX_MUSIC_CANDIDATE_EVIDENCE_BUNDLE_BYTES
                    + 1
            ])
            .is_err()
        );

        let alternate = ImmutableMusicCandidateEvidenceBundleV1::reconstruct(vec![
            crate::native_engine::tromr::TromrForwardCandidateLatticeV1::new(
                0,
                crate::native_engine::tromr::synthetic_candidate_lattice_for_test(2, true),
            )
            .expect("alternate valid lattice"),
        ])
        .expect("alternate valid bundle");
        assert_ne!(alternate.canonical_identity, baseline.canonical_identity);
        assert!(
            selected
                .parent_ledger_receipt()
                .validate_candidate_evidence_for_fragment_ordinal(0, &alternate)
                .is_err(),
            "valid bytes under the wrong CAS identity must refuse"
        );

        let mut mutated_selected = selected.clone();
        mutated_selected.candidate_evidence.canonical_bytes[0] ^= 1;
        assert!(mutated_selected.validate().is_err());

        let mut recanonicalized_selected = selected.clone();
        recanonicalized_selected.receipt.candidate_evidence_bundle = alternate.canonical_identity;
        recanonicalized_selected.receipt.canonical_bytes =
            recanonicalized_selected.receipt.expected_canonical_bytes();
        recanonicalized_selected.receipt.canonical_identity = recanonicalized_selected
            .receipt
            .expected_canonical_identity();
        assert!(
            recanonicalized_selected
                .receipt
                .validate_against_parent_receipt(&recanonicalized_selected.parent_ledger_receipt,)
                .is_err()
        );

        let mut changed_parent = selected_test_recognition();
        changed_parent.page_meta.fragments[0].forward_candidate_lattices =
            alternate.forward_candidate_lattices;
        assert!(
            changed_parent.validate_selected_row_parent().is_err(),
            "candidate substitution must invalidate the private provider seal"
        );

        let mut missing_parent = selected_test_recognition();
        missing_parent.page_meta.fragments[0]
            .forward_candidate_lattices
            .clear();
        assert!(
            missing_parent.validate_selected_row_parent().is_err(),
            "missing candidate evidence must never degrade to a receipted review row"
        );
    }

    #[test]
    fn parent_candidate_bundle_preserves_split_forward_order_and_prefix_resets() {
        use crate::native_engine::tromr::{
            TromrForwardCandidateLatticeV1, TromrRowInferenceRouteV1,
        };

        let mut recognition = selected_test_recognition();
        let split_options = TromrRecognitionOptionsV1 {
            split_policy: TromrSplitPolicyV1::ExperimentalBarlineSegments,
            ..TromrRecognitionOptionsV1::deterministic()
        };
        let split_identity = split_options.replay_identity().expect("split identity");
        recognition.page_meta.recognition_options = split_options;
        recognition.page_meta.recognition_options_identity = split_identity.clone();
        recognition.provenance.recognition_options = split_options;
        recognition.provenance.recognition_options_identity = split_identity;
        let recognition_json = split_options.canonical_json().expect("split JSON");
        recognition.provenance.recognition_options_sha256 =
            component_options_digest(RECOGNITION_OPTIONS_DOMAIN, &recognition_json);
        recognition.provenance.options_sha256 = options_digest(
            recognition.provenance.recognition_options_sha256,
            recognition.provenance.execution_options_sha256,
        );
        recognition.provenance.replay_sha256 = replay_digest(
            recognition.provenance.bundle_sha256,
            recognition.provenance.raster_sha256,
            recognition.provenance.options_sha256,
        );

        recognition.page_meta.staff_evidence[0].route =
            TromrRowInferenceRouteV1::ExperimentalSplitSegments;
        let second_forward = recognition.page_meta.staff_evidence[0].forward_inputs[0].clone();
        recognition.page_meta.staff_evidence[0]
            .forward_inputs
            .push(second_forward);
        recognition.page_meta.fragments[0]
            .forward_candidate_lattices
            .push(
                TromrForwardCandidateLatticeV1::new(
                    1,
                    crate::native_engine::tromr::synthetic_candidate_lattice_for_test(2, true),
                )
                .expect("second split lattice"),
            );
        recognition
            .reseal_after_test_fixture_mutation()
            .expect("reseal split candidate fixture");
        recognition
            .validate_selected_row_parent()
            .expect("split parent candidate ledger validates");

        let bundle = recognition
            .candidate_evidence_for_fragment_ordinal(0)
            .expect("materialize split candidate bundle");
        eprintln!(
            "candidate_evidence_split bytes={} sha256={} forwards={} positions={:?}",
            bundle.canonical_bytes.len(),
            hex_sha256(&bundle.canonical_identity.sha256),
            bundle.forward_candidate_lattices.len(),
            bundle
                .forward_candidate_lattices
                .iter()
                .map(|candidate| candidate.candidate_lattice.positions.len())
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            bundle
                .forward_candidate_lattices
                .iter()
                .map(|candidate| candidate.forward_input_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            bundle.forward_candidate_lattices[0]
                .candidate_lattice
                .positions[0]
                .prefix_length,
            1
        );
        assert_eq!(
            bundle.forward_candidate_lattices[1]
                .candidate_lattice
                .positions[0]
                .prefix_length,
            1,
            "the second forward must restart from its own seed prefix"
        );
        assert_ne!(
            bundle.forward_candidate_lattices[0]
                .candidate_lattice
                .canonical_sha256()
                .expect("first digest"),
            bundle.forward_candidate_lattices[1]
                .candidate_lattice
                .canonical_sha256()
                .expect("second digest"),
        );
        assert!(recognition.select_row(0).is_err());
    }

    #[test]
    fn nonselected_semantic_change_preserves_row_artifacts_but_moves_parent_lineage() {
        let mut baseline_recognition = selected_test_recognition();
        let baseline_semantics = baseline_recognition
            .page_meta
            .fragments
            .iter()
            .map(|fragment| fragment.semantic.clone())
            .collect::<Vec<_>>();
        baseline_recognition.page_meta.warnings =
            crate::native_engine::tromr::sanity_warnings(&baseline_semantics);
        baseline_recognition.musicxml =
            crate::native_engine::tromr::staves_to_musicxml(&baseline_semantics)
                .expect("baseline provider MusicXML");
        baseline_recognition
            .reseal_after_test_fixture_mutation()
            .expect("seal baseline provider recognition");

        let mut changed_recognition = baseline_recognition.clone();
        changed_recognition.page_meta.fragments[1].semantic =
            "clef-F4+note-C3_quarter|rest-quarter+barline".to_owned();
        let changed_semantics = changed_recognition
            .page_meta
            .fragments
            .iter()
            .map(|fragment| fragment.semantic.clone())
            .collect::<Vec<_>>();
        changed_recognition.page_meta.warnings =
            crate::native_engine::tromr::sanity_warnings(&changed_semantics);
        changed_recognition.musicxml =
            crate::native_engine::tromr::staves_to_musicxml(&changed_semantics)
                .expect("changed provider MusicXML");
        changed_recognition
            .reseal_after_test_fixture_mutation()
            .expect("seal changed provider recognition");

        assert_eq!(
            baseline_recognition.provenance(),
            changed_recognition.provenance(),
            "changing a provider output row does not rewrite immutable input replay provenance"
        );
        assert_ne!(
            baseline_recognition
                .parent_ledger_receipt()
                .canonical_identity,
            changed_recognition
                .parent_ledger_receipt()
                .canonical_identity,
            "the complete parent ledger must record a nonselected semantic/output change"
        );

        let baseline = baseline_recognition
            .select_row(0)
            .expect("select unchanged baseline row");
        let changed = changed_recognition
            .select_row(0)
            .expect("select unchanged row after nonselected mutation");
        baseline
            .validate()
            .expect("baseline selected row validates");
        changed
            .validate()
            .expect("changed-parent selected row validates");

        assert_eq!(baseline.semantic(), changed.semantic());
        assert_eq!(baseline.musicxml(), changed.musicxml());
        assert_eq!(baseline.model_input_gray8(), changed.model_input_gray8());
        assert_eq!(baseline.model_input_png(), changed.model_input_png());
        assert_eq!(baseline.review_crop_gray8(), changed.review_crop_gray8());
        assert_eq!(baseline.review_crop_png(), changed.review_crop_png());
        assert_eq!(baseline.staff_lines(), changed.staff_lines());
        assert_eq!(baseline.warnings(), changed.warnings());
        for (baseline_identity, changed_identity) in [
            (
                baseline.receipt().model_input_gray8,
                changed.receipt().model_input_gray8,
            ),
            (
                baseline.receipt().model_input_png,
                changed.receipt().model_input_png,
            ),
            (
                baseline.receipt().review_crop_gray8,
                changed.receipt().review_crop_gray8,
            ),
            (
                baseline.receipt().review_crop_png,
                changed.receipt().review_crop_png,
            ),
            (baseline.receipt().semantic, changed.receipt().semantic),
            (baseline.receipt().musicxml, changed.receipt().musicxml),
            (
                baseline.receipt().staff_lines_identity,
                changed.receipt().staff_lines_identity,
            ),
            (baseline.receipt().warnings, changed.receipt().warnings),
        ] {
            assert_eq!(baseline_identity, changed_identity);
        }
        assert_eq!(
            baseline.receipt().parent_replay_sha256,
            changed.receipt().parent_replay_sha256,
            "same immutable input retains the same replay identity"
        );
        assert_ne!(
            baseline.receipt().parent_ledger_identity,
            changed.receipt().parent_ledger_identity,
            "selected receipt must bind the changed complete-parent identity"
        );
        assert_ne!(
            baseline.receipt().canonical_identity,
            changed.receipt().canonical_identity,
            "selected receipt identity must move with its parent-ledger binding"
        );
    }

    #[test]
    fn selected_row_parent_receipt_reconstructs_without_ocr_or_crop_pixels() {
        let recognition = selected_test_recognition();
        let expected = recognition.parent_ledger_receipt().clone();
        expected.validate().expect("provider parent receipt");

        let rebuilt = ImmutableMusicParentLedgerReceiptV1::reconstruct(expected.fields.clone())
            .expect("reconstruct parent receipt from public fields");
        assert_eq!(rebuilt, expected);
        assert_eq!(rebuilt.expected_canonical_bytes(), rebuilt.canonical_bytes);
        assert_eq!(
            rebuilt.expected_canonical_identity(),
            rebuilt.canonical_identity
        );

        let (_, _, _, released_receipt) = recognition
            .into_owned_parts()
            .expect("validated owned extraction includes receipt");
        assert_eq!(released_receipt, expected);
    }

    #[test]
    fn parent_receipt_derives_exact_selected_warning_transport_without_ocr() {
        let recognition = selected_test_recognition();
        let selected = recognition.select_row(1).expect("select second fragment");
        let warnings = recognition
            .parent_ledger_receipt()
            .selected_warning_evidence_for_fragment_ordinal(
                selected.receipt().successful_fragment_ordinal_zero_based,
            )
            .expect("derive warnings from parent receipt");
        assert_eq!(&warnings, selected.warnings());
        verify_selected_music_artifact_identity_v1(
            SelectedMusicArtifactRoleV1::SelectedWarningsCanonical,
            &warnings.canonical_bytes,
            warnings.canonical_identity,
        )
        .expect("warning role identity verifies exact provider bytes");

        let mut changed_bytes = warnings.canonical_bytes.clone();
        changed_bytes.push(0);
        assert!(
            verify_selected_music_artifact_identity_v1(
                SelectedMusicArtifactRoleV1::SelectedWarningsCanonical,
                &changed_bytes,
                warnings.canonical_identity,
            )
            .is_err()
        );
        assert!(
            recognition
                .parent_ledger_receipt()
                .selected_warning_evidence_for_fragment_ordinal(u32::MAX)
                .is_err()
        );
    }

    #[test]
    fn selected_artifact_role_api_separates_equal_png_bytes_and_rejects_role_drift() {
        let selected = selected_test_recognition()
            .select_row(1)
            .expect("select detector-backed crop");
        assert_eq!(selected.model_input_png(), selected.review_crop_png());
        let model = selected_music_artifact_identity_v1(
            SelectedMusicArtifactRoleV1::ModelInputPng,
            selected.model_input_png(),
        );
        let review = selected_music_artifact_identity_v1(
            SelectedMusicArtifactRoleV1::ReviewCropPng,
            selected.review_crop_png(),
        );
        assert_eq!(model, selected.receipt().model_input_png);
        assert_eq!(review, selected.receipt().review_crop_png);
        assert_eq!(model.sha256, review.sha256);
        assert_eq!(model.blake3, review.blake3);
        assert_ne!(model.domain_identity_sha256, review.domain_identity_sha256);
        verify_selected_music_artifact_identity_v1(
            SelectedMusicArtifactRoleV1::ModelInputPng,
            selected.model_input_png(),
            model,
        )
        .expect("correct PNG role verifies");
        assert!(
            verify_selected_music_artifact_identity_v1(
                SelectedMusicArtifactRoleV1::ReviewCropPng,
                selected.model_input_png(),
                model,
            )
            .is_err()
        );

        let mut refreshed_model = selected.clone();
        refreshed_model
            .receipt
            .model_input_png
            .domain_identity_sha256[0] ^= 1;
        refreshed_model.receipt.canonical_bytes =
            refreshed_model.receipt.expected_canonical_bytes();
        refreshed_model.receipt.canonical_identity =
            refreshed_model.receipt.expected_canonical_identity();
        refreshed_model
            .receipt
            .validate()
            .expect("model-PNG selected receipt was internally refreshed");
        assert!(refreshed_model.validate().is_err());

        let mut refreshed_review = selected;
        refreshed_review
            .receipt
            .review_crop_png
            .domain_identity_sha256[0] ^= 1;
        refreshed_review.receipt.canonical_bytes =
            refreshed_review.receipt.expected_canonical_bytes();
        refreshed_review.receipt.canonical_identity =
            refreshed_review.receipt.expected_canonical_identity();
        refreshed_review
            .receipt
            .validate()
            .expect("review-PNG selected receipt was internally refreshed");
        assert!(refreshed_review.validate().is_err());
    }

    #[test]
    fn exact_gray8_verifier_rejects_model_and_review_identity_co_mutation() {
        let mut selected = selected_test_recognition()
            .select_row(1)
            .expect("select detector-backed crop");
        let mut forged = selected.parent_ledger_receipt.fields.attempts[1].forward_inputs[0].gray8;
        forged.identity_sha256[0] ^= 1;

        assert!(
            crate::preprocess::staff_detect::verify_tromr_gray8_artifact_identity_v1(
                selected.model_input_gray8().pixels(),
                selected.model_input_gray8().width(),
                selected.model_input_gray8().height(),
                forged,
            )
            .is_err(),
            "model bytes must reject a co-mutated Gray8 domain identity"
        );
        assert!(
            crate::preprocess::staff_detect::verify_tromr_gray8_artifact_identity_v1(
                selected.review_crop_gray8().pixels(),
                selected.review_crop_gray8().width(),
                selected.review_crop_gray8().height(),
                forged,
            )
            .is_err(),
            "review bytes must reject a co-mutated Gray8 domain identity"
        );

        selected.parent_ledger_receipt.fields.attempts[1].forward_inputs[0].gray8 = forged;
        selected.parent_ledger_receipt.fields.attempts[1].review_crop_gray8 = Some(forged);
        selected.parent_ledger_receipt.fields.staff_detection.crops[1].review_crop_gray8 = forged;
        selected.parent_ledger_receipt.canonical_bytes =
            selected.parent_ledger_receipt.expected_canonical_bytes();
        selected.parent_ledger_receipt.canonical_identity =
            selected.parent_ledger_receipt.expected_canonical_identity();
        selected
            .parent_ledger_receipt
            .validate()
            .expect("co-mutated unsigned parent receipt is internally consistent");

        let forged_selected = selected_gray8_artifact_identity(forged);
        selected.receipt.parent_ledger_identity = selected.parent_ledger_receipt.canonical_identity;
        selected.receipt.model_input_gray8 = forged_selected;
        selected.receipt.review_crop_gray8 = forged_selected;
        selected.receipt.canonical_bytes = selected.receipt.expected_canonical_bytes();
        selected.receipt.canonical_identity = selected.receipt.expected_canonical_identity();
        selected
            .receipt
            .validate_against_parent_receipt(&selected.parent_ledger_receipt)
            .expect("both co-mutated unsigned receipt layers agree structurally");
        assert!(
            selected.validate().is_err(),
            "provider aggregate validation must close the unsigned-receipt gap with exact bytes"
        );
    }

    #[test]
    fn selected_row_parent_receipt_rejects_open_censuses_and_stale_canonical_layers() {
        let receipt = selected_test_recognition().parent_ledger_receipt().clone();

        let mut missing = receipt.fields.clone();
        missing.attempts.pop();
        assert!(ImmutableMusicParentLedgerReceiptV1::reconstruct(missing).is_err());

        let mut extra = receipt.fields.clone();
        extra.attempts.push(extra.attempts[1].clone());
        assert!(ImmutableMusicParentLedgerReceiptV1::reconstruct(extra).is_err());

        let mut detector_drift = receipt.clone();
        detector_drift.fields.staff_detection.candidates[0].profile_sum += 1;
        assert!(detector_drift.validate().is_err());

        let mut co_mutated_but_not_reidentified = receipt.clone();
        co_mutated_but_not_reidentified
            .fields
            .combined_musicxml
            .sha256[0] ^= 1;
        co_mutated_but_not_reidentified.canonical_bytes =
            co_mutated_but_not_reidentified.expected_canonical_bytes();
        assert!(co_mutated_but_not_reidentified.validate().is_err());
    }

    #[test]
    fn selected_row_parent_receipt_binds_detector_transform_identities() {
        let receipt = selected_test_recognition().parent_ledger_receipt().clone();

        let mut global = receipt.clone();
        global
            .fields
            .staff_detection
            .global_deskew
            .globally_deskewed_gray8
            .identity_sha256[0] ^= 1;
        assert!(global.validate().is_err());

        let mut row = receipt;
        row.fields.staff_detection.crops[1]
            .row_refinement
            .refined_unpadded_crop_gray8
            .identity_sha256[0] ^= 1;
        assert!(row.validate().is_err());
    }

    #[test]
    fn selected_row_parent_projection_rejects_recanonicalized_selected_layer_drift() {
        fn recanonicalize(selected: &mut ImmutableSelectedMusicRow) {
            selected.receipt.canonical_bytes = selected.receipt.expected_canonical_bytes();
            selected.receipt.canonical_identity = selected.receipt.expected_canonical_identity();
            selected
                .receipt
                .validate()
                .expect("mutated selected layer is internally recanonicalized");
        }

        let selected = selected_test_recognition()
            .select_row(1)
            .expect("select row with parent receipt");
        selected
            .receipt()
            .validate_against_parent_receipt(selected.parent_ledger_receipt())
            .expect("unmodified selected receipt projects exactly from parent");

        let mut cases = Vec::new();

        let mut bbox = selected.clone();
        bbox.receipt
            .review_crop_source_bbox_xywh_in_globally_deskewed_raster[0] += 1;
        cases.push(("review bbox", bbox));

        let mut canvas = selected.clone();
        canvas.receipt.model_input_canvas_wh[0] += 1;
        cases.push(("model canvas", canvas));

        let mut padding = selected.clone();
        padding.receipt.review_crop_padding_trbl[0] += 1;
        cases.push(("review padding", padding));

        let mut lines = selected.clone();
        lines
            .receipt
            .accepted_detector_lines_y_in_globally_deskewed_raster[2] += 1;
        lines.receipt.staff_lines_identity = selected_music_staff_lines_identity_v1(
            lines
                .receipt
                .accepted_detector_lines_y_in_globally_deskewed_raster,
            lines.receipt.review_crop_staff_lines_y_in_canvas,
            lines.receipt.model_input_staff_lines_y_in_canvas,
        )
        .expect("mutated line coordinates remain ordered");
        cases.push(("detector lines", lines));

        let mut model_gray8 = selected.clone();
        model_gray8.receipt.model_input_gray8.sha256[0] ^= 1;
        cases.push(("model Gray8", model_gray8));

        let mut review_gray8 = selected.clone();
        review_gray8.receipt.review_crop_gray8.blake3[0] ^= 1;
        cases.push(("review Gray8", review_gray8));

        let mut musicxml = selected;
        musicxml.receipt.musicxml.domain_identity_sha256[0] ^= 1;
        cases.push(("MusicXML", musicxml));

        for (label, mut candidate) in cases {
            recanonicalize(&mut candidate);
            assert!(
                candidate
                    .receipt()
                    .validate_against_parent_receipt(candidate.parent_ledger_receipt())
                    .is_err(),
                "recanonicalized {label} drift must differ from the fixed parent"
            );
            assert!(
                candidate.validate().is_err(),
                "ordinary aggregate validation must enforce parent projection for {label}"
            );
        }
    }

    #[test]
    fn selected_row_survives_structurally_valid_unresolved_parent_context() {
        let recognition = selected_test_recognition_with_unresolved_context();
        recognition
            .validate_selected_row_parent()
            .expect("row admission uses structural parent validation");
        assert!(recognition.page_meta().publication_blocked());
        assert!(recognition.require_complete_for_publication().is_err());

        let parent = recognition.parent_ledger_receipt().clone();
        parent
            .validate()
            .expect("unresolved parent receipt is structurally valid");
        assert!(parent.fields.staff_detection.residual.unresolved);
        assert!(parent.require_complete_for_publication().is_err());
        assert_eq!(
            ImmutableMusicParentLedgerReceiptV1::reconstruct(parent.fields.clone())
                .expect("unresolved parent fields reconstruct"),
            parent
        );

        let selected = recognition
            .select_row(0)
            .expect("valid detector-backed row remains selectable");
        selected
            .validate()
            .expect("selected review draft validates");
        assert_eq!(
            selected.receipt().parent_ledger_identity,
            parent.canonical_identity,
            "selected receipt binds the unresolved parent context exactly"
        );
    }

    #[test]
    fn selected_parent_validation_rejects_a_corrupt_count_before_index_classification() {
        let valid = selected_test_recognition();
        valid
            .validate_selected_row_parent()
            .expect("closed parent evidence validates independently of an index");

        let mut corrupt = valid;
        corrupt.page_meta.detected_staff_count = 1;
        corrupt
            .reseal_after_test_fixture_mutation()
            .expect("reseal corrupt-count fixture");
        let error = corrupt
            .validate_selected_row_parent()
            .expect_err("a corrupt parent count must fail before out-of-range classification");
        assert!(
            error.to_string().contains("parent attempt ledger")
                || error.to_string().contains("staff segmentation disposition")
                || error
                    .to_string()
                    .contains("frozen provenance or detector census"),
            "unexpected parent-validation error: {error}"
        );
    }

    #[test]
    fn private_ledger_seal_rejects_same_shaped_never_inferred_substitutions() {
        let mut pixels = selected_test_recognition();
        let (width, height) = {
            let original = &pixels.page_meta.staff_evidence[0].forward_inputs[0].gray8;
            (original.width(), original.height())
        };
        let replacement = crate::preprocess::staff_detect::TromrGray8CropV1::from_tightly_packed(
            vec![17; width * height],
            width,
            height,
        )
        .expect("same-shaped replacement pixels");
        let replacement_identity = replacement.artifact_identity();
        let attempt = &mut pixels.page_meta.staff_evidence[0];
        attempt.forward_inputs[0].gray8 = replacement.clone();
        attempt.review_crop_gray8 = Some(replacement);
        let detector_crop = &mut pixels.page_meta.staff_detection.crops[0];
        detector_crop.review_crop_gray8 = replacement_identity;
        detector_crop
            .row_refinement
            .source_crop_before_refinement_gray8 = replacement_identity;
        detector_crop.row_refinement.refined_unpadded_crop_gray8 = replacement_identity;
        validate_parent_music_ledgers(&pixels.page_meta, &pixels.provenance)
            .expect("same-shaped pixels satisfy the public structural ledger");
        let error = pixels
            .select_row(0)
            .expect_err("unsealed replacement pixels must never be receipted");
        assert!(
            error.to_string().contains("private provider ledger seal"),
            "unexpected pixel-substitution error: {error}"
        );

        let mut semantic = selected_test_recognition();
        semantic.page_meta.fragments[0].semantic = "clef-G2+rest-half+barline".to_owned();
        let semantics = semantic
            .page_meta
            .fragments
            .iter()
            .map(|fragment| fragment.semantic.clone())
            .collect::<Vec<_>>();
        semantic.page_meta.warnings = crate::native_engine::tromr::sanity_warnings(&semantics);
        semantic.musicxml = crate::native_engine::tromr::staves_to_musicxml(&semantics)
            .expect("replacement semantic remains structurally legal");
        validate_parent_music_ledgers(&semantic.page_meta, &semantic.provenance)
            .expect("replacement semantic has a self-consistent public ledger");
        let error = semantic
            .select_row(0)
            .expect_err("unsealed replacement semantic must never be receipted");
        assert!(
            error.to_string().contains("private provider ledger seal"),
            "unexpected semantic-substitution error: {error}"
        );
    }

    #[test]
    fn parent_validation_rejects_selected_and_unrelated_attempt_geometry_drift() {
        let mut selected = selected_test_recognition();
        selected.page_meta.staff_evidence[0].geometry.source_bbox.0 += 1;
        selected
            .reseal_after_test_fixture_mutation()
            .expect("reseal selected-geometry fixture");
        let error = selected
            .select_row(0)
            .expect_err("selected attempt geometry drift must fail");
        assert!(
            error
                .to_string()
                .contains("attempt geometry differs from review geometry")
                || error
                    .to_string()
                    .contains("detector-backed attempt diverges from retained crop evidence"),
            "unexpected selected-geometry error: {error}"
        );

        let mut unrelated = selected_test_recognition();
        unrelated.page_meta.staff_evidence[1].geometry.source_bbox.0 += 1;
        unrelated
            .reseal_after_test_fixture_mutation()
            .expect("reseal unrelated-geometry fixture");
        let error = unrelated
            .select_row(0)
            .expect_err("unrelated attempt geometry drift must fail the complete parent gate");
        assert!(
            error
                .to_string()
                .contains("attempt geometry differs from review geometry")
                || error
                    .to_string()
                    .contains("detector-backed attempt diverges from retained crop evidence"),
            "unexpected unrelated-geometry error: {error}"
        );
    }

    #[test]
    fn selected_row_rejects_closed_world_ledger_mutations() {
        let mut duplicate = selected_test_recognition();
        duplicate
            .page_meta
            .staff_evidence
            .push(duplicate.page_meta.staff_evidence[1].clone());
        duplicate
            .reseal_after_test_fixture_mutation()
            .expect("reseal duplicate fixture");
        assert!(duplicate.select_row(0).is_err());

        let mut out_of_range = selected_test_recognition();
        out_of_range.page_meta.staff_evidence[1].index = 99;
        out_of_range
            .reseal_after_test_fixture_mutation()
            .expect("reseal out-of-range fixture");
        assert!(out_of_range.select_row(0).is_err());

        let mut unrelated_route = selected_test_recognition();
        unrelated_route.page_meta.staff_evidence[1].route =
            crate::native_engine::tromr::TromrRowInferenceRouteV1::ExperimentalSplitSegments;
        unrelated_route
            .reseal_after_test_fixture_mutation()
            .expect("reseal unrelated-route fixture");
        assert!(unrelated_route.select_row(0).is_err());

        let mut unrelated_bbox = selected_test_recognition();
        unrelated_bbox.page_meta.fragments[1].bbox.1 = 9_999;
        unrelated_bbox.page_meta.staves[1].1 = unrelated_bbox.page_meta.fragments[1].bbox;
        unrelated_bbox.page_meta.staff_evidence[1]
            .review_crop_geometry
            .as_mut()
            .expect("geometry")
            .source_bbox = unrelated_bbox.page_meta.fragments[1].bbox;
        unrelated_bbox
            .reseal_after_test_fixture_mutation()
            .expect("reseal unrelated-bbox fixture");
        assert!(unrelated_bbox.select_row(0).is_err());

        let mut warning_drift = selected_test_recognition();
        warning_drift
            .page_meta
            .warnings
            .push(crate::native_engine::tromr::MusicWarning {
                kind: "underfull_bar",
                part: 1,
                measure: 1,
                detail: "fabricated warning".to_owned(),
            });
        warning_drift
            .reseal_after_test_fixture_mutation()
            .expect("reseal warning-drift fixture");
        assert!(warning_drift.select_row(0).is_err());
    }

    #[test]
    fn selected_row_and_receipt_detect_artifact_mutations() {
        let selected = selected_test_recognition()
            .select_row(0)
            .expect("select row");

        let mut png = selected.clone();
        png.model_input_png[8] ^= 1;
        assert!(png.validate().is_err());

        let mut semantic = selected.clone();
        semantic.semantic.push_str("+rest-quarter");
        assert!(semantic.validate().is_err());

        let mut lines = selected.clone();
        lines.staff_lines.review_crop_staff_lines_y_in_canvas[2] += 1;
        assert!(lines.validate().is_err());

        let mut receipt = selected.clone();
        receipt.receipt.parent_replay_sha256[0] ^= 1;
        assert!(receipt.validate().is_err());

        let mut parent_identity = selected.clone();
        parent_identity
            .receipt
            .parent_ledger_identity
            .domain_identity_sha256[0] ^= 1;
        assert!(parent_identity.validate().is_err());

        let mut warnings = selected;
        warnings
            .warnings
            .parent_warnings
            .push(crate::native_engine::tromr::MusicWarning {
                kind: "underfull_bar",
                part: 1,
                measure: 2,
                detail: "mutation".to_owned(),
            });
        assert!(warnings.validate().is_err());
    }

    #[test]
    fn selected_row_accepts_single_whole_raster_and_padded_crop_routes() {
        use crate::native_engine::tromr::{
            TromrForwardInputV1, TromrModelInputSourceSpaceV1, TromrRowInferenceRouteV1,
            TromrStaffSegmentationDispositionV1,
        };

        let mut single = selected_test_recognition();
        single.page_meta.detected_staff_count = 1;
        single.page_meta.staff_segmentation_disposition =
            TromrStaffSegmentationDispositionV1::SingleStaffDetectedWholeImageRecognition;
        single.page_meta.fragments.truncate(1);
        single.page_meta.staves.truncate(1);
        single.page_meta.staff_evidence.truncate(1);
        single.page_meta.warnings =
            crate::native_engine::tromr::sanity_warnings(&[single.page_meta.fragments[0]
                .semantic
                .clone()]);
        let full = crate::preprocess::staff_detect::TromrGray8CropV1::from_tightly_packed(
            vec![255; 20 * 60],
            20,
            60,
        )
        .expect("full raster");
        let attempt = &mut single.page_meta.staff_evidence[0];
        attempt.route = TromrRowInferenceRouteV1::SingleDetectedStaffWholeRaster;
        attempt.geometry =
            crate::preprocess::staff_detect::StaffCropGeometry::unpadded((0, 0, 20, 60));
        attempt.forward_inputs = vec![TromrForwardInputV1 {
            gray8: full,
            source_space: TromrModelInputSourceSpaceV1::SelectedPageRaster,
            source_bbox_xywh: (0, 0, 20, 60),
            padding: crate::preprocess::staff_detect::StaffPadding::default(),
            staff_lines_y_in_canvas: None,
        }];
        refresh_test_staff_detection_from_attempts(&mut single);
        single
            .reseal_after_test_fixture_mutation()
            .expect("reseal single-raster fixture");
        let selected = single.select_row(0).expect("single route selects");
        assert_eq!(
            selected.receipt().inference_route,
            TromrRowInferenceRouteV1::SingleDetectedStaffWholeRaster
        );
        assert_ne!(
            selected.model_input_gray8().identity_sha256(),
            selected.review_crop_gray8().identity_sha256()
        );

        let mut padded = selected_test_recognition();
        let bbox = (2, 8, 8, 10);
        padded.page_meta.fragments[0].bbox = bbox;
        padded.page_meta.staves[0].1 = bbox;
        let evidence = &mut padded.page_meta.staff_evidence[0];
        let padding = crate::preprocess::staff_detect::StaffPadding {
            top: 1,
            right: 0,
            bottom: 1,
            left: 0,
        };
        let geometry = crate::preprocess::staff_detect::StaffCropGeometry {
            source_bbox: bbox,
            canvas_width: 8,
            canvas_height: 12,
            padding,
        };
        evidence.geometry = geometry;
        evidence.review_crop_geometry = Some(geometry);
        evidence.forward_inputs[0].padding = padding;
        refresh_test_staff_detection_from_attempts(&mut padded);
        padded
            .reseal_after_test_fixture_mutation()
            .expect("reseal padded fixture");
        padded
            .select_row(0)
            .expect("padded detected crop selects")
            .validate()
            .expect("padded selected aggregate validates");
    }

    #[test]
    fn selected_row_rejects_extra_results_on_zero_and_one_detection_routes() {
        use crate::native_engine::tromr::{
            TromrRowInferenceRouteV1, TromrStaffSegmentationDispositionV1,
        };

        let mut single = selected_test_recognition();
        single.page_meta.detected_staff_count = 1;
        single.page_meta.staff_segmentation_disposition =
            TromrStaffSegmentationDispositionV1::SingleStaffDetectedWholeImageRecognition;
        single.page_meta.staff_evidence.truncate(1);
        single.page_meta.staff_evidence[0].route =
            TromrRowInferenceRouteV1::SingleDetectedStaffWholeRaster;
        single
            .reseal_after_test_fixture_mutation()
            .expect("reseal malformed single-detection fixture");
        assert!(single.select_row(0).is_err());

        let mut zero = selected_test_recognition();
        zero.page_meta.detected_staff_count = 0;
        zero.page_meta.staff_segmentation_disposition =
            TromrStaffSegmentationDispositionV1::NoStaffDetectedWholeImageFallback;
        zero.page_meta.staff_evidence.truncate(1);
        zero.page_meta.staff_evidence[0].route =
            TromrRowInferenceRouteV1::NoDetectedStaffWholeRasterFallback;
        zero.page_meta.staff_evidence[0].review_crop_gray8 = None;
        zero.page_meta.staff_evidence[0].review_crop_geometry = None;
        zero.page_meta.staff_evidence[0].staff_lines = None;
        zero.reseal_after_test_fixture_mutation()
            .expect("reseal malformed zero-detection fixture");
        assert!(zero.select_row(0).is_err());
    }

    #[test]
    fn selected_row_refuses_absent_skipped_zero_detection_and_split_rows() {
        assert!(selected_test_recognition().select_row(99).is_err());

        let mut skipped = selected_test_recognition();
        let fragment = skipped.page_meta.fragments.pop().expect("second fragment");
        skipped.page_meta.staves.pop();
        skipped.page_meta.staff_evidence[1].outcome =
            crate::native_engine::tromr::StaffInferenceOutcome::Skipped;
        skipped.page_meta.staff_evidence[1].reason = Some("test skip".to_owned());
        skipped
            .page_meta
            .skips
            .push(crate::native_engine::tromr::StaffSkip {
                index: 1,
                bbox: fragment.bbox,
                reason: "test skip".to_owned(),
            });
        skipped.page_meta.warnings =
            crate::native_engine::tromr::sanity_warnings(&[skipped.page_meta.fragments[0]
                .semantic
                .clone()]);
        skipped
            .reseal_after_test_fixture_mutation()
            .expect("reseal skipped fixture");
        assert!(skipped.select_row(1).is_err());

        let mut split = selected_test_recognition();
        split.page_meta.staff_evidence[0].route =
            crate::native_engine::tromr::TromrRowInferenceRouteV1::ExperimentalSplitSegments;
        split
            .reseal_after_test_fixture_mutation()
            .expect("reseal split fixture");
        assert!(split.select_row(0).is_err());

        let mut zero = selected_test_recognition();
        zero.page_meta.detected_staff_count = 0;
        zero.page_meta.staff_segmentation_disposition = crate::native_engine::tromr::TromrStaffSegmentationDispositionV1::NoStaffDetectedWholeImageFallback;
        zero.page_meta.fragments.truncate(1);
        zero.page_meta.staves.truncate(1);
        zero.page_meta.staff_evidence.truncate(1);
        zero.page_meta.fragments[0].bbox = (0, 0, 20, 60);
        zero.page_meta.staves[0].1 = (0, 0, 20, 60);
        zero.page_meta.warnings =
            crate::native_engine::tromr::sanity_warnings(&[zero.page_meta.fragments[0]
                .semantic
                .clone()]);
        let evidence = &mut zero.page_meta.staff_evidence[0];
        evidence.route =
            crate::native_engine::tromr::TromrRowInferenceRouteV1::NoDetectedStaffWholeRasterFallback;
        evidence.geometry =
            crate::preprocess::staff_detect::StaffCropGeometry::unpadded((0, 0, 20, 60));
        evidence.forward_inputs = vec![crate::native_engine::tromr::TromrForwardInputV1 {
            gray8: crate::preprocess::staff_detect::TromrGray8CropV1::from_tightly_packed(
                vec![255; 20 * 60],
                20,
                60,
            )
            .expect("full raster"),
            source_space:
                crate::native_engine::tromr::TromrModelInputSourceSpaceV1::SelectedPageRaster,
            source_bbox_xywh: (0, 0, 20, 60),
            padding: crate::preprocess::staff_detect::StaffPadding::default(),
            staff_lines_y_in_canvas: None,
        }];
        evidence.review_crop_gray8 = None;
        evidence.review_crop_geometry = None;
        evidence.staff_lines = None;
        refresh_test_staff_detection_from_attempts(&mut zero);
        zero.reseal_after_test_fixture_mutation()
            .expect("reseal zero-detection fixture");
        assert!(zero.select_row(0).is_err());
    }

    #[test]
    fn bounded_reader_rejects_zero_short_extra_and_oversized_inputs() {
        assert!(read_bounded(std::io::Cursor::new(Vec::<u8>::new()), "x", 4, None).is_err());
        assert!(read_bounded(std::io::Cursor::new(b"abc"), "x", 4, Some(4)).is_err());
        assert!(read_bounded(std::io::Cursor::new(b"abcde"), "x", 4, None).is_err());
        assert!(read_bounded(std::io::Cursor::new(b"abcd"), "x", 4, Some(3)).is_err());
        assert_eq!(
            read_bounded(std::io::Cursor::new(b"abcd"), "x", 4, Some(4)).unwrap(),
            b"abcd"
        );
    }

    #[test]
    fn owned_bundle_identities_are_stable_path_free_and_component_sensitive() {
        let source = tiny_png();
        let model = empty_tromr_model();
        let tokenizers = tokenizer_bytes();
        let first = ImmutableMusicInputBundle::from_owned_parts(
            source.clone(),
            model.clone(),
            tokenizers.clone(),
            MusicInputOptions::default(),
        )
        .expect("first owned bundle");
        let second = ImmutableMusicInputBundle::from_owned_parts(
            source.clone(),
            model.clone(),
            tokenizers.clone(),
            MusicInputOptions::default(),
        )
        .expect("second owned bundle");
        assert_eq!(first.provenance(), second.provenance());
        assert_eq!(first.provenance().source_kind, MusicSourceKind::Image);
        assert_eq!(first.provenance().source.byte_len, source.len() as u64);
        assert_eq!(
            first.provenance().source.blake3_prefixed(),
            format!("blake3:{}", blake3::hash(&source).to_hex())
        );
        assert_eq!(first.provenance().model.byte_len, model.len() as u64);
        for (identity, bytes) in first.provenance().tokenizers.iter().zip(&tokenizers) {
            assert_eq!(identity.byte_len, bytes.len() as u64);
        }
        assert_eq!(first.provenance().raster_width, 6);
        assert_eq!(first.provenance().raster_height, 4);
        assert_eq!(
            first.provenance().recognition_options,
            TromrRecognitionOptionsV1::deterministic()
        );
        assert_eq!(
            first.provenance().recognition_options_identity,
            TromrRecognitionOptionsV1::deterministic()
                .replay_identity()
                .expect("deterministic options identity")
        );
        assert_eq!(
            first.provenance().execution_options,
            TromrExecutionOptionsV1::default()
        );
        assert_eq!(
            first.provenance().execution_options_identity,
            TromrExecutionOptionsV1::default()
                .replay_identity()
                .expect("default execution identity")
        );
        assert_eq!(first.provenance().bundle_sha256_hex().len(), 64);
        assert_eq!(
            first.provenance().recognition_options_sha256_hex().len(),
            64
        );
        assert_eq!(first.provenance().execution_options_sha256_hex().len(), 64);
        assert_eq!(first.provenance().replay_sha256_hex().len(), 64);

        let mut changed_tokenizers = tokenizers;
        changed_tokenizers[0].push(b'\n');
        let changed = ImmutableMusicInputBundle::from_owned_parts(
            source,
            model,
            changed_tokenizers,
            MusicInputOptions::default(),
        )
        .expect("whitespace-modified tokenizer remains structurally valid");
        assert_ne!(
            first.provenance().tokenizers[0],
            changed.provenance().tokenizers[0]
        );
        assert_ne!(
            first.provenance().bundle_sha256,
            changed.provenance().bundle_sha256
        );
        assert_ne!(
            first.provenance().replay_sha256,
            changed.provenance().replay_sha256
        );

        let experimental = ImmutableMusicInputBundle::from_owned_parts(
            tiny_png(),
            empty_tromr_model(),
            tokenizer_bytes(),
            MusicInputOptions {
                recognition: TromrRecognitionOptionsV1 {
                    split_policy: TromrSplitPolicyV1::ExperimentalBarlineSegments,
                    ..TromrRecognitionOptionsV1::deterministic()
                },
                ..MusicInputOptions::default()
            },
        )
        .expect("explicit experimental split bundle");
        assert_eq!(
            first.provenance().bundle_sha256,
            experimental.provenance().bundle_sha256
        );
        assert_eq!(
            first.provenance().raster_sha256,
            experimental.provenance().raster_sha256
        );
        assert_ne!(
            first.provenance().options_sha256,
            experimental.provenance().options_sha256
        );
        assert_ne!(
            first.provenance().replay_sha256,
            experimental.provenance().replay_sha256
        );

        let changed_execution = ImmutableMusicInputBundle::from_owned_parts(
            tiny_png(),
            empty_tromr_model(),
            tokenizer_bytes(),
            MusicInputOptions {
                execution: TromrExecutionOptionsV1 {
                    setup_budget_ms: 60_001,
                    ..TromrExecutionOptionsV1::default()
                },
                ..MusicInputOptions::default()
            },
        )
        .expect("changed bounded execution-policy bundle");
        assert_eq!(
            first.provenance().bundle_sha256,
            changed_execution.provenance().bundle_sha256
        );
        assert_eq!(
            first.provenance().raster_sha256,
            changed_execution.provenance().raster_sha256
        );
        assert_eq!(
            first.provenance().recognition_options_sha256,
            changed_execution.provenance().recognition_options_sha256
        );
        assert_ne!(
            first.provenance().execution_options_identity,
            changed_execution.provenance().execution_options_identity
        );
        assert_ne!(
            first.provenance().execution_options_sha256,
            changed_execution.provenance().execution_options_sha256
        );
        assert_ne!(
            first.provenance().options_sha256,
            changed_execution.provenance().options_sha256
        );
        assert_ne!(
            first.provenance().replay_sha256,
            changed_execution.provenance().replay_sha256
        );
    }

    #[test]
    fn opened_bundle_is_independent_of_later_same_path_replacement() {
        let dir = unique_temp_dir("path_swap");
        std::fs::create_dir_all(&dir).expect("create isolated test directory");
        let source_path = dir.join("score.png");
        let model_path = dir.join("tromr.focrq");
        let source_a = tiny_png();
        let model_a = empty_tromr_model();
        let tokenizer_a = tokenizer_bytes();
        std::fs::write(&source_path, &source_a).expect("write source A");
        std::fs::write(&model_path, &model_a).expect("write model A");
        for (filename, bytes) in TOKENIZER_FILENAMES.iter().zip(&tokenizer_a) {
            std::fs::write(dir.join(filename), bytes).expect("write tokenizer A");
        }

        let bundle = ImmutableMusicInputBundle::open(
            &source_path,
            &model_path,
            MusicInputOptions::default(),
        )
        .expect("pin bundle A");

        let source_b = vec![0x41; source_a.len()];
        let model_b = vec![0x42; model_a.len()];
        std::fs::write(&source_path, &source_b).expect("replace source path with B");
        std::fs::write(&model_path, &model_b).expect("replace model path with B");
        for filename in TOKENIZER_FILENAMES {
            std::fs::write(dir.join(filename), b"replacement B")
                .expect("replace tokenizer path with B");
        }

        assert_eq!(
            bundle.provenance().source,
            ConsumedBytesIdentity::of(&source_a)
        );
        assert_eq!(
            bundle.provenance().model,
            ConsumedBytesIdentity::of(&model_a)
        );
        assert_eq!(
            bundle.provenance().tokenizers,
            tokenizer_a
                .each_ref()
                .map(|bytes| ConsumedBytesIdentity::of(bytes))
        );
        assert_eq!(&*bundle.source_bytes, source_a);
        assert_ne!(
            bundle.provenance().source,
            ConsumedBytesIdentity::of(&source_b)
        );
        assert_ne!(
            bundle.provenance().model,
            ConsumedBytesIdentity::of(&model_b)
        );
    }

    #[test]
    fn observed_open_separates_preparation_timings_from_replay_provenance() {
        let dir = unique_temp_dir("observed_preparation");
        std::fs::create_dir_all(&dir).expect("create isolated test directory");
        let source_path = dir.join("score.png");
        let model_path = dir.join("tromr.focrq");
        std::fs::write(&source_path, tiny_png()).expect("write source");
        std::fs::write(&model_path, empty_tromr_model()).expect("write model");
        for (filename, bytes) in TOKENIZER_FILENAMES.iter().zip(tokenizer_bytes()) {
            std::fs::write(dir.join(filename), bytes).expect("write tokenizer");
        }

        let prepared = ImmutableMusicInputBundle::open_observed(
            &source_path,
            &model_path,
            MusicInputOptions::default(),
        )
        .expect("observed immutable input preparation");
        assert_eq!(
            prepared.diagnostics.outcome,
            crate::music_diagnostics::MusicRunOutcomeV1::Success
        );
        assert!(prepared.diagnostics.error_kind.is_none());
        assert!(prepared.diagnostics.detail.is_none());
        assert_eq!(prepared.diagnostics.pdf_parse_wall_micros, 0);
        assert_eq!(prepared.diagnostics.page_raster_wall_micros, 0);
        assert_eq!(
            prepared.bundle.provenance().source_kind,
            MusicSourceKind::Image
        );
        let json = serde_json::to_value(&prepared.diagnostics).expect("serialize diagnostics");
        assert!(json.get("replay_sha256").is_none());
        assert_eq!(
            json["timing_contract"],
            crate::music_diagnostics::MUSIC_DIAGNOSTICS_TIMING_CONTRACT
        );
    }

    #[test]
    fn image_page_selection_refuses_before_model_resolution() {
        let dir = unique_temp_dir("image_page_before_model");
        std::fs::create_dir_all(&dir).expect("create isolated test directory");
        let source_path = dir.join("score.png");
        std::fs::write(&source_path, tiny_png()).expect("write image source");

        let error = ImmutableMusicInputBundle::open(
            &source_path,
            &dir.join("missing-tromr.focrq"),
            MusicInputOptions {
                page: Some(2),
                ..MusicInputOptions::default()
            },
        )
        .err()
        .expect("image page selection must refuse before model resolution");

        assert_eq!(error.kind(), "usage");
        assert_eq!(
            error.to_string(),
            "usage error: TrOMR page 2 was requested for an image source; page selection is PDF-only"
        );
    }

    #[test]
    fn invalid_recognition_options_refuse_before_decode_or_inference() {
        let options = MusicInputOptions {
            recognition: TromrRecognitionOptionsV1 {
                schema_version: u32::MAX,
                ..TromrRecognitionOptionsV1::deterministic()
            },
            ..MusicInputOptions::default()
        };
        let error = ImmutableMusicInputBundle::from_owned_parts(
            b"not decoded because options validation runs first".to_vec(),
            empty_tromr_model(),
            tokenizer_bytes(),
            options,
        )
        .err()
        .expect("unsupported recognition schema must refuse");
        assert!(error.to_string().contains("options schema"));
    }

    #[test]
    fn invalid_execution_options_refuse_before_decode_or_inference() {
        let options = MusicInputOptions {
            execution: TromrExecutionOptionsV1 {
                setup_budget_ms: 0,
                ..TromrExecutionOptionsV1::default()
            },
            ..MusicInputOptions::default()
        };
        let error = ImmutableMusicInputBundle::from_owned_parts(
            b"not decoded because execution validation runs first".to_vec(),
            empty_tromr_model(),
            tokenizer_bytes(),
            options,
        )
        .err()
        .expect("zero execution setup budget must refuse");
        assert!(error.to_string().contains("setup_budget_ms"));
    }

    #[test]
    fn malformed_pdf_model_and_tokenizer_refuse_without_fallback() {
        let invalid_pdf = ImmutableMusicInputBundle::from_owned_parts(
            b"%PDF-not-a-document".to_vec(),
            empty_tromr_model(),
            tokenizer_bytes(),
            MusicInputOptions::default(),
        )
        .err()
        .expect("malformed PDF must refuse");
        assert!(invalid_pdf.to_string().to_ascii_lowercase().contains("pdf"));

        let invalid_model = ImmutableMusicInputBundle::from_owned_parts(
            tiny_png(),
            b"not a model".to_vec(),
            tokenizer_bytes(),
            MusicInputOptions::default(),
        )
        .err()
        .expect("malformed model must refuse");
        assert!(!invalid_model.to_string().is_empty());

        let mut invalid_tokenizers = tokenizer_bytes();
        invalid_tokenizers[2] = b"{}".to_vec();
        let invalid_tokenizer = ImmutableMusicInputBundle::from_owned_parts(
            tiny_png(),
            empty_tromr_model(),
            invalid_tokenizers,
            MusicInputOptions::default(),
        )
        .err()
        .expect("malformed tokenizer must refuse");
        assert!(
            invalid_tokenizer
                .to_string()
                .contains("tokenizer_lift.json")
        );
    }

    #[test]
    fn expected_hash_mismatch_refuses_before_model_or_inference() {
        let options = MusicInputOptions {
            expectations: MusicInputExpectations {
                source_sha256: Some([0x55; 32]),
                ..MusicInputExpectations::default()
            },
            ..MusicInputOptions::default()
        };
        let error = ImmutableMusicInputBundle::from_owned_parts(
            tiny_png(),
            b"not even parsed because source expectation fails".to_vec(),
            tokenizer_bytes(),
            options,
        )
        .err()
        .expect("wrong expected source hash must refuse");
        assert!(
            error
                .to_string()
                .contains("music source exact-consumption SHA-256 mismatch")
        );
    }

    #[test]
    fn caller_mutation_after_owned_construction_cannot_change_retained_source() {
        let source = tiny_png();
        let mut caller_copy = source.clone();
        let bundle = ImmutableMusicInputBundle::from_owned_parts(
            source,
            empty_tromr_model(),
            tokenizer_bytes(),
            MusicInputOptions::default(),
        )
        .expect("owned bundle");
        let before = bundle.provenance().source;
        caller_copy.fill(0);
        assert_eq!(bundle.provenance().source, before);
        assert_eq!(ConsumedBytesIdentity::of(&bundle.source_bytes), before);
    }

    #[test]
    fn public_engine_refuses_an_execution_policy_not_bound_into_the_bundle() {
        let bundle = ImmutableMusicInputBundle::from_owned_parts(
            tiny_png(),
            empty_tromr_model(),
            tokenizer_bytes(),
            MusicInputOptions::default(),
        )
        .expect("default-policy bundle");
        let mismatched = TromrExecutionOptionsV1 {
            setup_budget_ms: TromrExecutionOptionsV1::default().setup_budget_ms + 1,
            ..TromrExecutionOptionsV1::default()
        };
        let error = crate::OcrEngine::new()
            .expect("engine constructs")
            .recognize_immutable_music_with_execution(
                bundle,
                mismatched,
                crate::MusicCancellationToken::new(),
            )
            .expect_err("unreceipted policy override must refuse before forward");
        assert_eq!(error.kind(), "usage");
        assert!(
            error
                .to_string()
                .contains("does not match immutable bundle")
        );
    }

    #[test]
    fn observed_execution_failure_carries_diagnostics_but_no_score_payload() {
        let bundle = ImmutableMusicInputBundle::from_owned_parts(
            tiny_png(),
            empty_tromr_model(),
            tokenizer_bytes(),
            MusicInputOptions::default(),
        )
        .expect("immutable bundle");
        let cancellation = crate::MusicCancellationToken::new();
        cancellation.cancel();

        let failure = crate::OcrEngine::new()
            .expect("engine constructs")
            .recognize_immutable_music_observed(bundle, cancellation)
            .expect_err("pre-cancelled recognition must refuse before forward");

        assert!(matches!(failure.error, FocrError::Cancelled));
        assert_eq!(
            failure.diagnostics.outcome,
            crate::music_diagnostics::MusicRunOutcomeV1::Cancelled
        );
        assert_eq!(failure.diagnostics.error_kind.as_deref(), Some("cancelled"));
        assert_eq!(failure.diagnostics.attempts_started, 0);
        assert_eq!(
            failure.diagnostics.detail.as_deref(),
            Some("cooperative cancellation observed")
        );
        let diagnostics_json =
            serde_json::to_value(&failure.diagnostics).expect("serialize diagnostics");
        assert!(diagnostics_json.get("musicxml").is_none());
        assert!(diagnostics_json.get("page_meta").is_none());
    }
}
