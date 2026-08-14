//! Authoritative published lineage for the provider-owned TrOMR model.
//!
//! NetEase published the inference graph, four tokenizer tables, and an
//! epoch-47 state dictionary, but not the training implementation. This module
//! keeps those two facts together: every published and provider-derived edge is
//! pinned, while every unavailable original-training field is classified
//! explicitly. It is a provenance contract, not a reconstructed training
//! recipe and not a model-quality claim.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::dist::{self, TROMR_F32_RECIPE, TROMR_INT8_RECIPE};
use crate::error::{FocrError, FocrResult};

const LINEAGE_JSON: &str = include_str!("tromr_lineage_manifest.json");
const HISTORY_AUDIT_JSON: &str = include_str!("tromr_upstream_history_audit.json");
const EXPORT_RECEIPT_JSON: &str = include_str!("tromr_export_receipt.json");
const LINEAGE_JSON_SHA256: &str =
    "6eea7a84b9de1f9f957b2d0159a84c5d64deeb8199d95956b527dae7e24a92d3";
const HISTORY_AUDIT_JSON_SHA256: &str =
    "02a211cb7d29a884ee7b369624df29f2e1fef9823ea3987920cd936a34c44efd";
const EXPORT_RECEIPT_JSON_SHA256: &str =
    "ce1c2e199107b2b8532cf4a12e9e8b195462a53860b7a09e0254efdc7638389c";

/// Version of the public TrOMR lineage receipt.
pub const TROMR_LINEAGE_SCHEMA_VERSION: &str = "franken_ocr.tromr_lineage.v1";
/// Stable semantic identity for the first provider-owned lineage contract.
pub const TROMR_LINEAGE_CONTRACT_ID: &str =
    "netease-polyphonic-tromr-d1aa83a3-franken-ocr-lineage-v1";
/// Canonical JSON identity of [`tromr_lineage_receipt`].
pub const TROMR_LINEAGE_CANONICAL_SHA256: &str =
    "1ad70680205890e70396388a320ee828cd120052bd771840ea68e8a4cdc52d72";

const UPSTREAM_REPOSITORY: &str = "https://github.com/NetEase/Polyphonic-TrOMR";
const UPSTREAM_COMMIT: &str = "d1aa83a34fb4a05f33ceb4f917917b88600a9bc6";
const UPSTREAM_TREE: &str = "9cbeec06f6337ec51b9f40010b35d731b27c1231";
const RAW_CHECKPOINT_SHA256: &str =
    "02925259ef59f5578a8c9e954ac363bb15538ea38ce73090b861c1519179f910";
const FOLDED_SAFETENSORS_SHA256: &str =
    "41c88802fbf24c43d7515d94b5552a850b4cfd85f1a3d605e9eb4f841fe141eb";
const F32_FOCRQ_SHA256: &str = "a9d41485a98534ad0a1f7c1ec624f0a92f3f092c7dc30ac5af636b50dc465edc";
const INT8_FOCRQ_SHA256: &str = "cced11c0f05656dd54cc615a15939c472dc8f916f04ae154ea4a0364839f845a";

/// Authority class for one field in the lineage receipt.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TromrEvidenceClassV1 {
    /// Exact source or artifact in the immutable official repository.
    OfficialSource,
    /// Fact stated by the authors in the versioned paper, without executable source.
    OfficialPaper,
    /// Deterministic conversion or packaging operation owned by franken_ocr.
    ProviderDerivation,
    /// A new versioned choice owned by this project, never attributed upstream.
    ProjectDecision,
    /// Sought in authoritative material but not published.
    UnavailableUpstream,
    /// Informative secondary evidence that cannot establish an authority edge.
    UnverifiedSecondary,
}

/// Named artifact roles in the raw-checkpoint-to-runtime chain.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TromrArtifactRoleV1 {
    RawCheckpoint,
    RhythmTokenizer,
    PitchTokenizer,
    LiftTokenizer,
    NoteTokenizer,
    FoldedSafetensors,
    F32Focrq,
    Int8Focrq,
}

/// Immutable identity of the provider contract carrying the lineage receipt.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TromrProviderIdentityV1 {
    pub package_name: String,
    pub package_version: String,
    pub upstream_release_tag: String,
    pub upstream_base_commit: String,
    pub local_contract_revision: String,
    pub source_inventory_canonicalization: String,
    pub source_inventory_sha256: String,
    pub full_tree_binding: String,
}

/// Immutable official repository identity.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TromrUpstreamSourceV1 {
    pub repository_url: String,
    pub default_branch: String,
    pub commit: String,
    pub tree: String,
    pub reachable_commit_count: u32,
    pub history_audit_path: String,
    pub history_audit_bytes: u64,
    pub history_audit_sha256: String,
    pub tag_count: u32,
    pub release_count: u32,
    pub license_spdx: String,
    pub license_sha256: String,
}

/// Versioned paper authority used only for facts the paper actually states.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TromrPaperAuthorityV1 {
    pub evidence_class: TromrEvidenceClassV1,
    pub title: String,
    pub arxiv_id: String,
    pub version: String,
    pub url: String,
    pub pdf_url: String,
    pub pdf_bytes: u64,
    pub pdf_sha256: String,
    pub source_url: String,
    pub source_bytes: u64,
    pub source_sha256: String,
    pub tex_path: String,
    pub tex_sha256: String,
}

/// One hash-pinned source file from the official repository.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TromrOfficialFileV1 {
    pub role: String,
    pub path: String,
    pub sha256: String,
    pub git_blob: String,
}

/// One hash-pinned source file implementing a provider derivation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TromrProviderSourceV1 {
    pub role: String,
    pub path: String,
    pub sha256: String,
}

/// One immutable model, tokenizer, or conversion artifact.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TromrLineageArtifactV1 {
    pub role: TromrArtifactRoleV1,
    pub filename: String,
    pub bytes: Option<u64>,
    pub sha256: String,
    pub git_blob: Option<String>,
    pub source_sha256: Option<String>,
    pub recipe: Option<String>,
    pub tensor_count: Option<u32>,
    pub parameter_count: Option<u64>,
    pub value_bytes: Option<u64>,
    pub tensor_value_inventory_sha256: Option<String>,
    pub evidence_class: TromrEvidenceClassV1,
    pub retained_in_provider_tree: bool,
}

/// One deterministic edge in the provider conversion graph.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TromrConversionStepV1 {
    pub step_id: String,
    pub from_role: TromrArtifactRoleV1,
    pub to_role: TromrArtifactRoleV1,
    pub evidence_class: TromrEvidenceClassV1,
    pub implementation_path: String,
    pub input_tensor_count: u32,
    pub output_tensor_count: u32,
    pub operation: String,
}

/// Exhaustive negative evidence supporting unavailable original-training fields.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TromrNegativeTrainingEvidenceV1 {
    pub readme_path: String,
    pub readme_sha256: String,
    pub line: u32,
    pub statement: String,
    pub reachable_commit_count: u32,
    pub history_audit_sha256: String,
    pub audit_method: String,
}

/// Repeated provider conversion proof and its declared accepted replay layer.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TromrConversionReplayV1 {
    pub receipt_path: String,
    pub receipt_bytes: u64,
    pub receipt_sha256: String,
    pub identical_regeneration_count: u32,
    pub regenerated_safetensors_bytes: u64,
    pub regenerated_safetensors_sha256: String,
    pub accepted_safetensors_sha256: String,
    pub byte_exact: bool,
    pub names_shapes_dtypes_equal: bool,
    pub exact_value_tensor_count: u32,
    pub tolerance_value_tensor_count: u32,
    pub max_abs: String,
    pub max_abs_contract: String,
    pub outcome: String,
    pub honest_gap: String,
}

/// Evidence disposition for one original-training fact or missing fact.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TromrTrainingFieldV1 {
    pub name: String,
    pub evidence_class: TromrEvidenceClassV1,
    pub value: Option<String>,
    pub evidence_refs: Vec<String>,
}

/// Published availability boundary for the original NetEase training run.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TromrTrainingAvailabilityV1 {
    pub published_training_code: bool,
    pub negative_evidence: TromrNegativeTrainingEvidenceV1,
    pub fields: Vec<TromrTrainingFieldV1>,
}

/// Complete provider-owned lineage receipt for the accepted TrOMR baseline.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TromrLineageReceiptV1 {
    pub schema: String,
    pub contract_id: String,
    pub provider: TromrProviderIdentityV1,
    pub upstream: TromrUpstreamSourceV1,
    pub paper: TromrPaperAuthorityV1,
    pub source_files: Vec<TromrOfficialFileV1>,
    pub checkpoint: TromrLineageArtifactV1,
    pub tokenizers: Vec<TromrLineageArtifactV1>,
    pub provider_sources: Vec<TromrProviderSourceV1>,
    pub conversion_artifacts: Vec<TromrLineageArtifactV1>,
    pub runtime_artifacts: Vec<TromrLineageArtifactV1>,
    pub conversion_steps: Vec<TromrConversionStepV1>,
    pub conversion_replay: TromrConversionReplayV1,
    pub training_availability: TromrTrainingAvailabilityV1,
}

/// Runtime observation supplied by an embedder after reading one exact artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TromrObservedArtifactV1 {
    pub role: TromrArtifactRoleV1,
    pub filename: String,
    pub bytes: u64,
    pub sha256: String,
    pub source_sha256: Option<String>,
    /// Recipe declared by the accepted lineage/distribution mapping.
    ///
    /// Current TrOMR FOCRQ headers do not self-describe this field. The exact
    /// artifact hash closes the link to this external, provider-owned claim.
    pub declared_recipe: Option<String>,
    pub tensor_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TromrHistoryCommitV1 {
    commit: String,
    tree: String,
    path_count: u32,
    path_inventory_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TromrUpstreamHistoryAuditV1 {
    schema: String,
    repository_url: String,
    head_commit: String,
    head_tree: String,
    commit_order: String,
    path_inventory_canonicalization: String,
    commits: Vec<TromrHistoryCommitV1>,
    union_path_count: u32,
    union_path_inventory_sha256: String,
    union_paths: Vec<String>,
    sought_owner_surfaces: Vec<String>,
    owner_surface_path_matches: Vec<String>,
    reviewed_authoritative_text_paths: Vec<String>,
    conclusion: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TromrTensorInventoryRowV1 {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TromrExportEnvironmentV1 {
    python_implementation: String,
    python_version: String,
    torch: String,
    safetensors: String,
    numpy: String,
    system: String,
    machine: String,
    byteorder: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TromrExportSourceV1 {
    bytes: u64,
    sha256: String,
    tensor_count: u32,
    parameter_count: u64,
    value_bytes: u64,
    tensor_value_inventory_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TromrWsFoldReceiptV1 {
    eps: f64,
    variance: String,
    reference: String,
    proof: String,
    folded_convs: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TromrAcceptedFocrqV1 {
    bytes: u64,
    sha256: String,
    format_version: u32,
    model_id: String,
    source_sha256: String,
    tensor_count: u32,
    value_bytes: u64,
    tensor_value_inventory_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TromrTensorDifferenceV1 {
    name: String,
    generated_sha256: String,
    accepted_sha256: String,
    max_abs: f64,
    mean_abs: f64,
    different_elements: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TromrAcceptedValueComparisonV1 {
    names_shapes_dtypes_equal: bool,
    exact_value_tensor_count: u32,
    tolerance_value_tensor_count: u32,
    max_abs: f64,
    mean_of_tensor_mean_abs: f64,
    different_element_count: u64,
    differences: Vec<TromrTensorDifferenceV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TromrExportReceiptV1 {
    purpose: String,
    script: String,
    environment: TromrExportEnvironmentV1,
    source_pth: TromrExportSourceV1,
    ws_fold: TromrWsFoldReceiptV1,
    dropped: Vec<String>,
    tensors_out: u32,
    model_safetensors_bytes: u64,
    model_safetensors_sha256: String,
    expected_model_safetensors_sha256: String,
    expected_model_safetensors_match: bool,
    accepted_replay_outcome: String,
    accepted_max_abs_contract: f64,
    tensor_value_inventory_sha256: String,
    accepted_focrq: TromrAcceptedFocrqV1,
    accepted_value_comparison: TromrAcceptedValueComparisonV1,
    source_tensor_inventory: Vec<TromrTensorInventoryRowV1>,
    output_tensor_inventory: Vec<TromrTensorInventoryRowV1>,
    accepted_tensor_inventory: Vec<TromrTensorInventoryRowV1>,
    license: String,
}

static LINEAGE: OnceLock<Result<TromrLineageReceiptV1, String>> = OnceLock::new();

/// Return the parsed and fully validated embedded lineage receipt.
///
/// # Errors
/// Fails closed when the embedded bytes, canonical identity, official pins,
/// retained audit receipts, conversion graph, training-field census, provider
/// sources, or distribution manifest drift.
pub fn tromr_lineage_receipt() -> FocrResult<&'static TromrLineageReceiptV1> {
    LINEAGE
        .get_or_init(parse_and_verify_embedded)
        .as_ref()
        .map_err(|message| {
            FocrError::FormatMismatch(format!(
                "embedded TrOMR lineage receipt is invalid: {message}"
            ))
        })
}

impl TromrLineageReceiptV1 {
    /// Parse and structurally validate a portable receipt supplied by an embedder.
    ///
    /// Unknown, missing, duplicate, or type-confused fields are rejected by the
    /// serde schema before semantic validation runs.
    pub fn from_json(json: &str) -> FocrResult<Self> {
        let receipt: Self = serde_json::from_str(json).map_err(|error| {
            FocrError::FormatMismatch(format!("TrOMR lineage JSON is invalid: {error}"))
        })?;
        receipt.validate()?;
        Ok(receipt)
    }

    /// Parse a receipt and require exact equality with the accepted provider baseline.
    pub fn from_accepted_json(json: &str) -> FocrResult<Self> {
        let receipt = Self::from_json(json)?;
        receipt.validate_accepted_baseline()?;
        Ok(receipt)
    }

    /// Validate portable structure and internal cross-links without blessing it.
    ///
    /// This deliberately does not compare against the accepted repository,
    /// artifact, or provider pins. Call validate_accepted_baseline for that
    /// stronger claim.
    pub fn validate(&self) -> FocrResult<()> {
        exact("schema", &self.schema, TROMR_LINEAGE_SCHEMA_VERSION)?;
        nonempty("contract_id", &self.contract_id)?;
        nonempty("provider.package_name", &self.provider.package_name)?;
        nonempty("provider.package_version", &self.provider.package_version)?;
        require_git_oid(
            "provider.upstream_base_commit",
            &self.provider.upstream_base_commit,
        )?;
        require_sha256(
            "provider.source_inventory_sha256",
            &self.provider.source_inventory_sha256,
        )?;
        require_git_oid("upstream.commit", &self.upstream.commit)?;
        require_git_oid("upstream.tree", &self.upstream.tree)?;
        require_sha256(
            "upstream.history_audit_sha256",
            &self.upstream.history_audit_sha256,
        )?;
        require_sha256("paper.pdf_sha256", &self.paper.pdf_sha256)?;
        require_sha256("paper.source_sha256", &self.paper.source_sha256)?;
        require_sha256("paper.tex_sha256", &self.paper.tex_sha256)?;
        if self.paper.pdf_bytes == 0 || self.paper.source_bytes == 0 {
            return invalid("paper authority declares a zero-byte artifact".into());
        }
        self.validate_source_shapes()?;
        self.validate_artifact_shapes()?;
        self.validate_conversion_graph()?;
        self.validate_training_shapes()?;
        self.validate_conversion_replay_shape()?;
        Ok(())
    }

    /// Validate this receipt as the one exact accepted provider baseline.
    pub fn validate_accepted_baseline(&self) -> FocrResult<()> {
        self.validate()?;
        self.validate_upstream()?;
        self.validate_sources()?;
        self.validate_artifacts()?;
        self.validate_training_availability()?;
        self.validate_conversion_replay()?;
        self.validate_embedded_receipts()?;
        self.validate_provider_source_bytes()?;
        self.validate_distribution_manifest()?;

        let canonical = self.canonical_sha256_unchecked()?;
        if canonical != TROMR_LINEAGE_CANONICAL_SHA256 {
            return invalid(format!(
                "canonical identity is {canonical}, expected pinned {TROMR_LINEAGE_CANONICAL_SHA256}"
            ));
        }
        Ok(())
    }

    /// Stable compact JSON; struct field order and vector order are canonical.
    pub fn canonical_json(&self) -> FocrResult<String> {
        self.validate()?;
        serde_json::to_string(self).map_err(|error| {
            FocrError::FormatMismatch(format!(
                "TrOMR lineage canonical serialization failed: {error}"
            ))
        })
    }

    /// SHA-256 of [`Self::canonical_json`].
    pub fn canonical_sha256(&self) -> FocrResult<String> {
        self.validate()?;
        self.canonical_sha256_unchecked()
    }

    /// Find one expected artifact by its globally unique role.
    #[must_use]
    pub fn artifact(&self, role: TromrArtifactRoleV1) -> Option<&TromrLineageArtifactV1> {
        std::iter::once(&self.checkpoint)
            .chain(self.tokenizers.iter())
            .chain(self.conversion_artifacts.iter())
            .chain(self.runtime_artifacts.iter())
            .find(|artifact| artifact.role == role)
    }

    /// Find one original-training field by its stable name.
    #[must_use]
    pub fn training_field(&self, name: &str) -> Option<&TromrTrainingFieldV1> {
        self.training_availability
            .fields
            .iter()
            .find(|field| field.name == name)
    }

    /// Verify a caller-observed artifact against its pinned role.
    ///
    /// The caller computes bytes/SHA from the exact descriptor it consumes and
    /// supplies source/tensor metadata from that same open. `declared_recipe`
    /// comes from the accepted distribution mapping because the current TrOMR
    /// FOCRQ header does not carry a recipe field.
    pub fn verify_observed_artifact(&self, observed: &TromrObservedArtifactV1) -> FocrResult<()> {
        self.validate()?;
        let expected = self.artifact(observed.role).ok_or_else(|| {
            FocrError::FormatMismatch(format!(
                "TrOMR observed artifact has unknown role {:?}",
                observed.role
            ))
        })?;
        if observed.filename != expected.filename {
            return invalid(format!(
                "observed {:?} filename is {:?}, expected {:?}",
                observed.role, observed.filename, expected.filename
            ));
        }
        if let Some(expected_bytes) = expected.bytes {
            if observed.bytes != expected_bytes {
                return invalid(format!(
                    "observed {:?} byte length is {}, expected {expected_bytes}",
                    observed.role, observed.bytes
                ));
            }
        } else if observed.bytes == 0 {
            return invalid(format!(
                "observed {:?} has zero bytes but its unpinned size must still be nonzero",
                observed.role
            ));
        }
        if observed.sha256 != expected.sha256 {
            return invalid(format!(
                "observed {:?} SHA-256 is {}, expected {}",
                observed.role, observed.sha256, expected.sha256
            ));
        }
        if observed.source_sha256 != expected.source_sha256 {
            return invalid(format!(
                "observed {:?} source SHA-256 is {:?}, expected {:?}",
                observed.role, observed.source_sha256, expected.source_sha256
            ));
        }
        if observed.declared_recipe != expected.recipe {
            return invalid(format!(
                "observed {:?} declared recipe is {:?}, expected {:?}",
                observed.role, observed.declared_recipe, expected.recipe
            ));
        }
        if observed.tensor_count != expected.tensor_count {
            return invalid(format!(
                "observed {:?} tensor count is {:?}, expected {:?}",
                observed.role, observed.tensor_count, expected.tensor_count
            ));
        }
        Ok(())
    }

    fn validate_source_shapes(&self) -> FocrResult<()> {
        let mut official_roles = BTreeSet::new();
        let mut official_paths = BTreeSet::new();
        for source in &self.source_files {
            nonempty("official source role", &source.role)?;
            nonempty("official source path", &source.path)?;
            if !official_roles.insert(source.role.as_str()) {
                return invalid(format!("duplicate official source role {:?}", source.role));
            }
            if !official_paths.insert(source.path.as_str()) {
                return invalid(format!("duplicate official source path {:?}", source.path));
            }
            require_sha256(&format!("official source {}", source.path), &source.sha256)?;
            require_git_oid(
                &format!("official source {} blob", source.path),
                &source.git_blob,
            )?;
        }

        let mut provider_roles = BTreeSet::new();
        let mut provider_paths = BTreeSet::new();
        for source in &self.provider_sources {
            nonempty("provider source role", &source.role)?;
            nonempty("provider source path", &source.path)?;
            if !provider_roles.insert(source.role.as_str()) {
                return invalid(format!("duplicate provider source role {:?}", source.role));
            }
            if !provider_paths.insert(source.path.as_str()) {
                return invalid(format!("duplicate provider source path {:?}", source.path));
            }
            require_sha256(&format!("provider source {}", source.path), &source.sha256)?;
        }

        let mut ordered: Vec<_> = self.provider_sources.iter().collect();
        ordered.sort_by(|left, right| left.path.cmp(&right.path));
        let mut digest = Sha256::new();
        for source in ordered {
            digest.update(source.path.as_bytes());
            digest.update([0]);
            digest.update(source.sha256.as_bytes());
            digest.update([b'\n']);
        }
        let actual = format!("{:x}", digest.finalize());
        if actual != self.provider.source_inventory_sha256 {
            return invalid(format!(
                "provider source inventory is {actual}, declared {}",
                self.provider.source_inventory_sha256
            ));
        }
        Ok(())
    }

    fn validate_artifact_shapes(&self) -> FocrResult<()> {
        let mut roles = BTreeSet::new();
        for artifact in std::iter::once(&self.checkpoint)
            .chain(self.tokenizers.iter())
            .chain(self.conversion_artifacts.iter())
            .chain(self.runtime_artifacts.iter())
        {
            if !roles.insert(artifact.role) {
                return invalid(format!("duplicate artifact role {:?}", artifact.role));
            }
            nonempty("artifact filename", &artifact.filename)?;
            require_sha256(
                &format!("artifact {:?} SHA-256", artifact.role),
                &artifact.sha256,
            )?;
            if let Some(source) = artifact.source_sha256.as_deref() {
                require_sha256(&format!("artifact {:?} source", artifact.role), source)?;
            }
            if let Some(blob) = artifact.git_blob.as_deref() {
                require_git_oid(&format!("artifact {:?} Git blob", artifact.role), blob)?;
            }
            if let Some(inventory) = artifact.tensor_value_inventory_sha256.as_deref() {
                require_sha256(
                    &format!("artifact {:?} tensor inventory", artifact.role),
                    inventory,
                )?;
            }
            if artifact.bytes == Some(0)
                || artifact.tensor_count == Some(0)
                || artifact.parameter_count == Some(0)
                || artifact.value_bytes == Some(0)
            {
                return invalid(format!(
                    "artifact {:?} declares a zero size/count",
                    artifact.role
                ));
            }
            let inventory_fields = [
                artifact.parameter_count.is_some(),
                artifact.value_bytes.is_some(),
                artifact.tensor_value_inventory_sha256.is_some(),
            ];
            if inventory_fields.iter().any(|present| *present)
                && !inventory_fields.iter().all(|present| *present)
            {
                return invalid(format!(
                    "artifact {:?} has a partial tensor-value inventory",
                    artifact.role
                ));
            }
        }
        Ok(())
    }

    fn validate_training_shapes(&self) -> FocrResult<()> {
        let mut names = BTreeSet::new();
        for field in &self.training_availability.fields {
            nonempty("training field name", &field.name)?;
            if !names.insert(field.name.as_str()) {
                return invalid(format!("duplicate training field {:?}", field.name));
            }
            if field.evidence_refs.is_empty()
                || field
                    .evidence_refs
                    .iter()
                    .any(|reference| reference.is_empty())
            {
                return invalid(format!(
                    "training field {:?} has no evidence refs",
                    field.name
                ));
            }
            match field.evidence_class {
                TromrEvidenceClassV1::UnavailableUpstream => {
                    if field.value.is_some()
                        || !field
                            .evidence_refs
                            .iter()
                            .all(|reference| reference.starts_with("official:"))
                    {
                        return invalid(format!(
                            "unavailable training field {:?} must have no value and official negative refs",
                            field.name
                        ));
                    }
                }
                TromrEvidenceClassV1::OfficialSource => {
                    if field.value.as_deref().is_none_or(str::is_empty)
                        || !field
                            .evidence_refs
                            .iter()
                            .all(|reference| reference.starts_with("official:"))
                    {
                        return invalid(format!(
                            "official-source training field {:?} lacks an official value/ref",
                            field.name
                        ));
                    }
                }
                TromrEvidenceClassV1::OfficialPaper => {
                    if field.value.as_deref().is_none_or(str::is_empty)
                        || !field
                            .evidence_refs
                            .iter()
                            .all(|reference| reference.starts_with("official-paper:"))
                    {
                        return invalid(format!(
                            "official-paper training field {:?} lacks a paper value/ref",
                            field.name
                        ));
                    }
                }
                TromrEvidenceClassV1::ProviderDerivation
                | TromrEvidenceClassV1::ProjectDecision
                | TromrEvidenceClassV1::UnverifiedSecondary => {
                    return invalid(format!(
                        "original-training field {:?} uses disallowed evidence class {:?}",
                        field.name, field.evidence_class
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_conversion_replay_shape(&self) -> FocrResult<()> {
        let replay = &self.conversion_replay;
        require_sha256("conversion replay receipt", &replay.receipt_sha256)?;
        require_sha256(
            "conversion replay regenerated safetensors",
            &replay.regenerated_safetensors_sha256,
        )?;
        require_sha256(
            "conversion replay accepted safetensors",
            &replay.accepted_safetensors_sha256,
        )?;
        let max_abs = replay
            .max_abs
            .parse::<f64>()
            .map_err(|_| FocrError::FormatMismatch("TrOMR replay max_abs is not numeric".into()))?;
        let contract = replay.max_abs_contract.parse::<f64>().map_err(|_| {
            FocrError::FormatMismatch("TrOMR replay max_abs_contract is not numeric".into())
        })?;
        if !max_abs.is_finite()
            || !contract.is_finite()
            || max_abs < 0.0
            || contract < 0.0
            || replay.exact_value_tensor_count + replay.tolerance_value_tensor_count != 260
        {
            return invalid("conversion replay has an invalid value/tolerance census".into());
        }
        if replay.outcome == "value_tolerance"
            && (replay.byte_exact || !replay.names_shapes_dtypes_equal || max_abs > contract)
        {
            return invalid("conversion replay falsely claims value_tolerance".into());
        }
        Ok(())
    }

    fn canonical_sha256_unchecked(&self) -> FocrResult<String> {
        let json = serde_json::to_vec(self).map_err(|error| {
            FocrError::FormatMismatch(format!(
                "TrOMR lineage canonical serialization failed: {error}"
            ))
        })?;
        Ok(hex_sha256(&json))
    }

    fn validate_upstream(&self) -> FocrResult<()> {
        exact("schema", &self.schema, TROMR_LINEAGE_SCHEMA_VERSION)?;
        exact("contract_id", &self.contract_id, TROMR_LINEAGE_CONTRACT_ID)?;
        exact(
            "provider.package_name",
            &self.provider.package_name,
            "franken_ocr",
        )?;
        exact(
            "provider.package_version",
            &self.provider.package_version,
            "0.7.2",
        )?;
        exact(
            "provider.upstream_release_tag",
            &self.provider.upstream_release_tag,
            "v0.7.2",
        )?;
        exact(
            "provider.upstream_base_commit",
            &self.provider.upstream_base_commit,
            "c5a0e368b1be33187a17cd9aa716653ce6230590",
        )?;
        exact(
            "provider.local_contract_revision",
            &self.provider.local_contract_revision,
            "franken-ocr-mtdt-lineage-v24",
        )?;
        exact(
            "provider.full_tree_binding",
            &self.provider.full_tree_binding,
            "consumer_recursive_source_inventory",
        )?;
        exact(
            "upstream.repository_url",
            &self.upstream.repository_url,
            UPSTREAM_REPOSITORY,
        )?;
        exact(
            "upstream.default_branch",
            &self.upstream.default_branch,
            "master",
        )?;
        exact("upstream.commit", &self.upstream.commit, UPSTREAM_COMMIT)?;
        exact("upstream.tree", &self.upstream.tree, UPSTREAM_TREE)?;
        exact(
            "upstream.history_audit_path",
            &self.upstream.history_audit_path,
            "src/native_engine/tromr_upstream_history_audit.json",
        )?;
        if self.upstream.history_audit_bytes != HISTORY_AUDIT_JSON.len() as u64 {
            return invalid(format!(
                "upstream.history_audit_bytes is {}, embedded bytes are {}",
                self.upstream.history_audit_bytes,
                HISTORY_AUDIT_JSON.len()
            ));
        }
        exact(
            "upstream.history_audit_sha256",
            &self.upstream.history_audit_sha256,
            HISTORY_AUDIT_JSON_SHA256,
        )?;
        if self.upstream.reachable_commit_count != 17 {
            return invalid(format!(
                "upstream.reachable_commit_count is {}, expected 17",
                self.upstream.reachable_commit_count
            ));
        }
        if self.upstream.tag_count != 0 || self.upstream.release_count != 0 {
            return invalid("upstream unexpectedly claims a tag or GitHub release".into());
        }
        exact(
            "upstream.license_spdx",
            &self.upstream.license_spdx,
            "Apache-2.0",
        )?;
        exact(
            "upstream.license_sha256",
            &self.upstream.license_sha256,
            "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        )?;
        if self.paper.evidence_class != TromrEvidenceClassV1::OfficialPaper {
            return invalid("paper authority is not classified official_paper".into());
        }
        exact("paper.arxiv_id", &self.paper.arxiv_id, "2308.09370")?;
        exact("paper.version", &self.paper.version, "v1")?;
        exact(
            "paper.url",
            &self.paper.url,
            "https://arxiv.org/abs/2308.09370v1",
        )?;
        exact(
            "paper.pdf_url",
            &self.paper.pdf_url,
            "https://arxiv.org/pdf/2308.09370v1",
        )?;
        if self.paper.pdf_bytes != 1_627_131 || self.paper.source_bytes != 8_040_787 {
            return invalid("paper PDF/source byte lengths drifted".into());
        }
        exact(
            "paper.pdf_sha256",
            &self.paper.pdf_sha256,
            "ee79554cda98fab1ea7928d8208f25c043450bff01e9147b449629a58aa94aca",
        )?;
        exact(
            "paper.source_url",
            &self.paper.source_url,
            "https://export.arxiv.org/e-print/2308.09370v1",
        )?;
        exact(
            "paper.source_sha256",
            &self.paper.source_sha256,
            "52565a986c0c854d24b7c3a71c8b034c1a80f7c0ccf7aca725e58bc533344924",
        )?;
        exact("paper.tex_path", &self.paper.tex_path, "Template.tex")?;
        exact(
            "paper.tex_sha256",
            &self.paper.tex_sha256,
            "acc301212e717e3b6422462b6def2cab11fa51c7ee51494386802347085bbe31",
        )?;
        Ok(())
    }

    fn validate_sources(&self) -> FocrResult<()> {
        if self.source_files.len() != 10 {
            return invalid(format!(
                "official source-file census has {}, expected 10",
                self.source_files.len()
            ));
        }
        if self.provider_sources.len() != 8 {
            return invalid(format!(
                "provider source-file census has {}, expected 8",
                self.provider_sources.len()
            ));
        }
        let mut official_roles = BTreeSet::new();
        let mut official_paths = BTreeSet::new();
        for source in &self.source_files {
            if !official_roles.insert(source.role.as_str()) {
                return invalid(format!("duplicate official source role {:?}", source.role));
            }
            if !official_paths.insert(source.path.as_str()) {
                return invalid(format!("duplicate official source path {:?}", source.path));
            }
            require_sha256(&format!("official source {}", source.path), &source.sha256)?;
            require_git_oid(
                &format!("official source {} blob", source.path),
                &source.git_blob,
            )?;
        }
        let mut provider_roles = BTreeSet::new();
        let mut provider_paths = BTreeSet::new();
        for source in &self.provider_sources {
            if !provider_roles.insert(source.role.as_str()) {
                return invalid(format!("duplicate provider source role {:?}", source.role));
            }
            if !provider_paths.insert(source.path.as_str()) {
                return invalid(format!("duplicate provider source path {:?}", source.path));
            }
            require_sha256(&format!("provider source {}", source.path), &source.sha256)?;
        }
        if !provider_paths.contains("scripts/gen_tromr_safetensors.py")
            || !provider_paths.contains("src/quant/convert.rs")
            || !provider_paths.contains("models/manifest-v2.json")
            || !provider_paths.contains("src/native_engine/tromr_upstream_history_audit.json")
            || !provider_paths.contains("src/native_engine/tromr_export_receipt.json")
        {
            return invalid("provider source census omits a conversion owner".into());
        }
        Ok(())
    }

    fn validate_artifacts(&self) -> FocrResult<()> {
        if self.checkpoint.role != TromrArtifactRoleV1::RawCheckpoint {
            return invalid("checkpoint role is not raw_checkpoint".into());
        }
        if self.tokenizers.len() != 4
            || self.conversion_artifacts.len() != 1
            || self.runtime_artifacts.len() != 2
        {
            return invalid(format!(
                "artifact census is checkpoint=1 tokenizers={} conversion={} runtime={}; expected 1/4/1/2",
                self.tokenizers.len(),
                self.conversion_artifacts.len(),
                self.runtime_artifacts.len()
            ));
        }
        let expected_tokenizer_order = [
            TromrArtifactRoleV1::RhythmTokenizer,
            TromrArtifactRoleV1::PitchTokenizer,
            TromrArtifactRoleV1::LiftTokenizer,
            TromrArtifactRoleV1::NoteTokenizer,
        ];
        if self
            .tokenizers
            .iter()
            .map(|artifact| artifact.role)
            .ne(expected_tokenizer_order)
        {
            return invalid("tokenizer role order is not rhythm,pitch,lift,note".into());
        }

        let mut roles = BTreeSet::new();
        for artifact in std::iter::once(&self.checkpoint)
            .chain(self.tokenizers.iter())
            .chain(self.conversion_artifacts.iter())
            .chain(self.runtime_artifacts.iter())
        {
            if !roles.insert(artifact.role) {
                return invalid(format!("duplicate artifact role {:?}", artifact.role));
            }
            require_sha256(
                &format!("artifact {:?} SHA-256", artifact.role),
                &artifact.sha256,
            )?;
            if let Some(source) = artifact.source_sha256.as_deref() {
                require_sha256(&format!("artifact {:?} source", artifact.role), source)?;
            }
            if let Some(blob) = artifact.git_blob.as_deref() {
                require_git_oid(&format!("artifact {:?} Git blob", artifact.role), blob)?;
            }
            if artifact.bytes == Some(0) || artifact.tensor_count == Some(0) {
                return invalid(format!(
                    "artifact {:?} declares a zero size/count",
                    artifact.role
                ));
            }
        }

        exact(
            "raw checkpoint SHA-256",
            &self.checkpoint.sha256,
            RAW_CHECKPOINT_SHA256,
        )?;
        if self.checkpoint.bytes != Some(86_254_711)
            || self.checkpoint.tensor_count != Some(261)
            || self.checkpoint.parameter_count != Some(21_534_232)
            || self.checkpoint.value_bytes != Some(86_136_928)
            || self.checkpoint.tensor_value_inventory_sha256.as_deref()
                != Some("0e080bdf0309b1cb4c3322abd789dc0d879d3f6ca58dd8a17aac7f35b246c1c1")
            || self.checkpoint.evidence_class != TromrEvidenceClassV1::OfficialSource
        {
            return invalid(
                "raw checkpoint size/tensor/value-inventory/authority contract drifted".into(),
            );
        }
        let folded = self
            .artifact(TromrArtifactRoleV1::FoldedSafetensors)
            .expect("census checked folded artifact");
        exact(
            "folded safetensors SHA-256",
            &folded.sha256,
            FOLDED_SAFETENSORS_SHA256,
        )?;
        if folded.tensor_value_inventory_sha256.as_deref()
            != Some("59af5214d0f73eceaba23dc5e7c60941673a4f642b4577c5e1b4183e49cda200")
            || folded.parameter_count != Some(21_533_972)
            || folded.value_bytes != Some(86_135_888)
        {
            return invalid("accepted folded-safetensors value inventory drifted".into());
        }
        let f32 = self
            .artifact(TromrArtifactRoleV1::F32Focrq)
            .expect("census checked f32 artifact");
        exact("f32 FOCRQ SHA-256", &f32.sha256, F32_FOCRQ_SHA256)?;
        if f32.tensor_value_inventory_sha256 != folded.tensor_value_inventory_sha256
            || f32.parameter_count != folded.parameter_count
            || f32.value_bytes != folded.value_bytes
        {
            return invalid("f32 FOCRQ values do not bind the folded inventory".into());
        }
        let int8 = self
            .artifact(TromrArtifactRoleV1::Int8Focrq)
            .expect("census checked int8 artifact");
        exact("int8 FOCRQ SHA-256", &int8.sha256, INT8_FOCRQ_SHA256)?;
        Ok(())
    }

    fn validate_conversion_graph(&self) -> FocrResult<()> {
        if self.conversion_steps.len() != 3 {
            return invalid(format!(
                "conversion graph has {} steps, expected 3",
                self.conversion_steps.len()
            ));
        }
        let provider_paths: BTreeSet<&str> = self
            .provider_sources
            .iter()
            .map(|source| source.path.as_str())
            .collect();
        for step in &self.conversion_steps {
            if step.evidence_class != TromrEvidenceClassV1::ProviderDerivation {
                return invalid(format!(
                    "conversion step {:?} is not provider_derivation",
                    step.step_id
                ));
            }
            if !provider_paths.contains(step.implementation_path.as_str()) {
                return invalid(format!(
                    "conversion step {:?} implementation {:?} is not hash-pinned",
                    step.step_id, step.implementation_path
                ));
            }
            let from = self.artifact(step.from_role).ok_or_else(|| {
                FocrError::FormatMismatch(format!(
                    "TrOMR conversion step {:?} has unknown input role {:?}",
                    step.step_id, step.from_role
                ))
            })?;
            let to = self.artifact(step.to_role).ok_or_else(|| {
                FocrError::FormatMismatch(format!(
                    "TrOMR conversion step {:?} has unknown output role {:?}",
                    step.step_id, step.to_role
                ))
            })?;
            if from.tensor_count != Some(step.input_tensor_count)
                || to.tensor_count != Some(step.output_tensor_count)
            {
                return invalid(format!(
                    "conversion step {:?} tensor counts do not match its artifacts",
                    step.step_id
                ));
            }
            if to.source_sha256.as_deref() != Some(from.sha256.as_str()) {
                return invalid(format!(
                    "conversion step {:?} output source_sha256 does not bind its input",
                    step.step_id
                ));
            }
        }
        Ok(())
    }

    fn validate_training_availability(&self) -> FocrResult<()> {
        if self.training_availability.published_training_code {
            return invalid("receipt falsely claims published original training code".into());
        }
        let negative = &self.training_availability.negative_evidence;
        exact("negative readme path", &negative.readme_path, "README.md")?;
        let readme = self
            .source_files
            .iter()
            .find(|source| source.path == negative.readme_path)
            .ok_or_else(|| {
                FocrError::FormatMismatch(
                    "TrOMR lineage validation failed: negative evidence README is not in the official source census"
                        .into(),
                )
            })?;
        exact(
            "negative readme SHA-256",
            &negative.readme_sha256,
            &readme.sha256,
        )?;
        exact(
            "official README SHA-256",
            &negative.readme_sha256,
            "0944e5e45ee007ec0654c657a075a2bbe313d9ae8cdc96947e68ae81a03d5317",
        )?;
        exact(
            "negative readme statement",
            &negative.statement,
            "The training code will be open source later.",
        )?;
        if negative.line != 5 || negative.reachable_commit_count != 17 {
            return invalid("negative-evidence README line or history count drifted".into());
        }
        exact(
            "negative history identity",
            &negative.history_audit_sha256,
            HISTORY_AUDIT_JSON_SHA256,
        )?;
        if negative.audit_method.trim().is_empty() {
            return invalid("negative-evidence audit method is empty".into());
        }

        const EXPECTED: [(&str, TromrEvidenceClassV1); 29] = [
            (
                "published_training_entrypoint",
                TromrEvidenceClassV1::UnavailableUpstream,
            ),
            (
                "dataset_generation_code",
                TromrEvidenceClassV1::UnavailableUpstream,
            ),
            (
                "exact_corpus_sample_identities",
                TromrEvidenceClassV1::UnavailableUpstream,
            ),
            (
                "split_construction",
                TromrEvidenceClassV1::UnavailableUpstream,
            ),
            (
                "tokenizer_construction_inputs",
                TromrEvidenceClassV1::UnavailableUpstream,
            ),
            (
                "augmentation_sequence",
                TromrEvidenceClassV1::UnavailableUpstream,
            ),
            (
                "preprocessing_training_sequence",
                TromrEvidenceClassV1::UnavailableUpstream,
            ),
            (
                "seed_and_order_policy",
                TromrEvidenceClassV1::UnavailableUpstream,
            ),
            (
                "initialization_and_pretraining",
                TromrEvidenceClassV1::UnavailableUpstream,
            ),
            (
                "per_head_cross_entropy",
                TromrEvidenceClassV1::OfficialSource,
            ),
            (
                "consistency_loss_gamma",
                TromrEvidenceClassV1::OfficialSource,
            ),
            (
                "complete_loss_aggregation",
                TromrEvidenceClassV1::UnavailableUpstream,
            ),
            ("paper_loss_lambda", TromrEvidenceClassV1::OfficialPaper),
            ("paper_loss_beta", TromrEvidenceClassV1::OfficialPaper),
            ("optimizer_family", TromrEvidenceClassV1::OfficialPaper),
            ("initial_learning_rate", TromrEvidenceClassV1::OfficialPaper),
            (
                "optimizer_betas_epsilon_weight_decay",
                TromrEvidenceClassV1::UnavailableUpstream,
            ),
            (
                "scheduler_and_warmup",
                TromrEvidenceClassV1::UnavailableUpstream,
            ),
            (
                "gradient_clipping_accumulation_amp",
                TromrEvidenceClassV1::UnavailableUpstream,
            ),
            ("batch_size", TromrEvidenceClassV1::OfficialPaper),
            ("input_geometry", TromrEvidenceClassV1::OfficialPaper),
            ("generated_staff_count", TromrEvidenceClassV1::OfficialPaper),
            ("cmsd_staff_crop_count", TromrEvidenceClassV1::OfficialPaper),
            ("training_hardware", TromrEvidenceClassV1::OfficialPaper),
            (
                "stopping_and_checkpoint_selection",
                TromrEvidenceClassV1::UnavailableUpstream,
            ),
            (
                "musescore_and_generation_tool_versions",
                TromrEvidenceClassV1::UnavailableUpstream,
            ),
            (
                "original_evaluation_entrypoint",
                TromrEvidenceClassV1::UnavailableUpstream,
            ),
            (
                "resumable_optimizer_scheduler_state",
                TromrEvidenceClassV1::OfficialSource,
            ),
            (
                "checkpoint_epoch_label",
                TromrEvidenceClassV1::OfficialSource,
            ),
        ];
        if self.training_availability.fields.len() != EXPECTED.len() {
            return invalid(format!(
                "training-field census has {}, expected {}",
                self.training_availability.fields.len(),
                EXPECTED.len()
            ));
        }
        for (field, (expected_name, expected_class)) in
            self.training_availability.fields.iter().zip(EXPECTED)
        {
            if field.name != expected_name || field.evidence_class != expected_class {
                return invalid(format!(
                    "training field {:?} has class {:?}; expected {:?}/{expected_class:?}",
                    field.name, field.evidence_class, expected_name
                ));
            }
            if field.evidence_refs.is_empty() || field.evidence_refs.iter().any(String::is_empty) {
                return invalid(format!(
                    "training field {:?} has no evidence refs",
                    field.name
                ));
            }
            match field.evidence_class {
                TromrEvidenceClassV1::UnavailableUpstream => {
                    if field.value.is_some()
                        || !field
                            .evidence_refs
                            .iter()
                            .all(|reference| reference.starts_with("official:"))
                    {
                        return invalid(format!(
                            "unavailable training field {:?} must have no value and official negative refs",
                            field.name
                        ));
                    }
                }
                TromrEvidenceClassV1::OfficialSource => {
                    if field.value.as_deref().is_none_or(str::is_empty)
                        || !field
                            .evidence_refs
                            .iter()
                            .all(|reference| reference.starts_with("official:"))
                    {
                        return invalid(format!(
                            "official-source training field {:?} lacks an official value/ref",
                            field.name
                        ));
                    }
                }
                TromrEvidenceClassV1::OfficialPaper => {
                    if field.value.as_deref().is_none_or(str::is_empty)
                        || !field
                            .evidence_refs
                            .iter()
                            .all(|reference| reference.starts_with("official-paper:"))
                    {
                        return invalid(format!(
                            "official-paper training field {:?} lacks a paper value/ref",
                            field.name
                        ));
                    }
                }
                TromrEvidenceClassV1::ProviderDerivation
                | TromrEvidenceClassV1::ProjectDecision
                | TromrEvidenceClassV1::UnverifiedSecondary => {
                    return invalid(format!(
                        "original-training field {:?} uses disallowed evidence class {:?}",
                        field.name, field.evidence_class
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_conversion_replay(&self) -> FocrResult<()> {
        let replay = &self.conversion_replay;
        exact(
            "conversion replay receipt path",
            &replay.receipt_path,
            "src/native_engine/tromr_export_receipt.json",
        )?;
        if replay.receipt_bytes != EXPORT_RECEIPT_JSON.len() as u64 {
            return invalid(format!(
                "conversion replay receipt bytes are {}, embedded bytes are {}",
                replay.receipt_bytes,
                EXPORT_RECEIPT_JSON.len()
            ));
        }
        exact(
            "conversion replay receipt SHA-256",
            &replay.receipt_sha256,
            EXPORT_RECEIPT_JSON_SHA256,
        )?;
        if replay.identical_regeneration_count != 2
            || replay.regenerated_safetensors_bytes != 86_166_472
            || replay.exact_value_tensor_count != 220
            || replay.tolerance_value_tensor_count != 40
            || replay.byte_exact
            || !replay.names_shapes_dtypes_equal
        {
            return invalid("conversion replay census or replay layer drifted".into());
        }
        exact(
            "conversion replay regenerated SHA-256",
            &replay.regenerated_safetensors_sha256,
            "6101efe24503a4246443ced5e777f3fc0c8a2156078587a3880057773f9b46b9",
        )?;
        exact(
            "conversion replay accepted SHA-256",
            &replay.accepted_safetensors_sha256,
            FOLDED_SAFETENSORS_SHA256,
        )?;
        exact(
            "conversion replay outcome",
            &replay.outcome,
            "value_tolerance",
        )?;
        if replay.honest_gap.trim().is_empty() {
            return invalid("conversion replay omits its honest gap".into());
        }
        Ok(())
    }

    fn validate_embedded_receipts(&self) -> FocrResult<()> {
        if hex_sha256(HISTORY_AUDIT_JSON.as_bytes()) != HISTORY_AUDIT_JSON_SHA256 {
            return invalid("embedded upstream-history audit bytes drifted".into());
        }
        if hex_sha256(EXPORT_RECEIPT_JSON.as_bytes()) != EXPORT_RECEIPT_JSON_SHA256 {
            return invalid("embedded checkpoint-export receipt bytes drifted".into());
        }

        let history: TromrUpstreamHistoryAuditV1 = serde_json::from_str(HISTORY_AUDIT_JSON)
            .map_err(|error| {
                FocrError::FormatMismatch(format!(
                    "TrOMR upstream-history audit JSON is invalid: {error}"
                ))
            })?;
        self.validate_history_audit(&history)?;

        let export: TromrExportReceiptV1 =
            serde_json::from_str(EXPORT_RECEIPT_JSON).map_err(|error| {
                FocrError::FormatMismatch(format!(
                    "TrOMR checkpoint-export receipt JSON is invalid: {error}"
                ))
            })?;
        self.validate_export_receipt(&export)?;
        Ok(())
    }

    fn validate_history_audit(&self, audit: &TromrUpstreamHistoryAuditV1) -> FocrResult<()> {
        exact(
            "history audit schema",
            &audit.schema,
            "franken_ocr.tromr_upstream_history_audit.v1",
        )?;
        exact(
            "history audit repository",
            &audit.repository_url,
            UPSTREAM_REPOSITORY,
        )?;
        exact(
            "history audit head commit",
            &audit.head_commit,
            UPSTREAM_COMMIT,
        )?;
        exact("history audit head tree", &audit.head_tree, UPSTREAM_TREE)?;
        exact(
            "history audit commit order",
            &audit.commit_order,
            "git rev-list --reverse HEAD",
        )?;
        nonempty(
            "history audit path canonicalization",
            &audit.path_inventory_canonicalization,
        )?;
        if audit.commits.len() != 17 {
            return invalid(format!(
                "history audit has {} commits, expected 17",
                audit.commits.len()
            ));
        }
        let mut commits = BTreeSet::new();
        for row in &audit.commits {
            require_git_oid("history audit commit", &row.commit)?;
            require_git_oid("history audit tree", &row.tree)?;
            require_sha256("history audit path inventory", &row.path_inventory_sha256)?;
            if row.path_count == 0 || !commits.insert(row.commit.as_str()) {
                return invalid("history audit has an empty or duplicate commit row".into());
            }
        }
        if audit
            .commits
            .last()
            .is_none_or(|row| row.commit != UPSTREAM_COMMIT || row.tree != UPSTREAM_TREE)
        {
            return invalid("history audit does not terminate at the accepted head".into());
        }

        if audit.union_path_count as usize != audit.union_paths.len()
            || audit.union_paths.len() != 53
        {
            return invalid("history audit union path census drifted".into());
        }
        let mut previous = None;
        let mut union_digest = Sha256::new();
        for path in &audit.union_paths {
            nonempty("history audit union path", path)?;
            if previous.is_some_and(|prior: &String| prior >= path) {
                return invalid("history audit union paths are not unique lexical order".into());
            }
            union_digest.update(path.as_bytes());
            union_digest.update([b'\n']);
            previous = Some(path);
        }
        let union_sha256 = format!("{:x}", union_digest.finalize());
        exact(
            "history audit union path inventory",
            &audit.union_path_inventory_sha256,
            &union_sha256,
        )?;
        if audit.sought_owner_surfaces.len() < 8
            || !audit.owner_surface_path_matches.is_empty()
            || audit.reviewed_authoritative_text_paths.len() < 9
            || !audit
                .reviewed_authoritative_text_paths
                .iter()
                .any(|path| path == "README.md")
            || audit.conclusion.trim().is_empty()
        {
            return invalid("history audit search evidence is incomplete".into());
        }
        Ok(())
    }

    fn validate_export_receipt(&self, export: &TromrExportReceiptV1) -> FocrResult<()> {
        nonempty("export purpose", &export.purpose)?;
        exact(
            "export implementation",
            &export.script,
            "scripts/gen_tromr_safetensors.py",
        )?;
        exact(
            "export Python implementation",
            &export.environment.python_implementation,
            "CPython",
        )?;
        exact(
            "export Python version",
            &export.environment.python_version,
            "3.9.25",
        )?;
        exact("export torch", &export.environment.torch, "1.11.0+cpu")?;
        exact(
            "export safetensors",
            &export.environment.safetensors,
            "0.4.5",
        )?;
        exact("export numpy", &export.environment.numpy, "1.26.4")?;
        exact("export system", &export.environment.system, "Linux")?;
        exact("export machine", &export.environment.machine, "x86_64")?;
        exact("export byteorder", &export.environment.byteorder, "little")?;

        if export.source_pth.bytes != 86_254_711
            || export.source_pth.sha256 != RAW_CHECKPOINT_SHA256
            || export.source_pth.tensor_count != 261
            || export.source_pth.parameter_count != 21_534_232
            || export.source_pth.value_bytes != 86_136_928
            || export.source_pth.tensor_value_inventory_sha256
                != "0e080bdf0309b1cb4c3322abd789dc0d879d3f6ca58dd8a17aac7f35b246c1c1"
        {
            return invalid("export raw-checkpoint census drifted".into());
        }
        if export.ws_fold.eps != 0.000_001
            || export.ws_fold.variance != "population (unbiased=False)"
            || export.ws_fold.folded_convs.len() != 40
            || export.ws_fold.reference.trim().is_empty()
            || export.ws_fold.proof.trim().is_empty()
            || export.dropped.len() != 1
            || export.dropped[0] != "decoder.note_mask"
            || export.tensors_out != 260
        {
            return invalid("export WS/drop contract drifted".into());
        }
        if export.model_safetensors_bytes != 86_166_472
            || export.model_safetensors_sha256
                != "6101efe24503a4246443ced5e777f3fc0c8a2156078587a3880057773f9b46b9"
            || export.expected_model_safetensors_sha256 != FOLDED_SAFETENSORS_SHA256
            || export.expected_model_safetensors_match
            || export.accepted_replay_outcome != "value_tolerance"
            || export.accepted_max_abs_contract != 0.000_001
            || export.tensor_value_inventory_sha256
                != "bcd45e78b7ad1e704a631f4862118f3af2eaba3198bc44e004c53e5e4e2de98a"
        {
            return invalid("export regenerated artifact contract drifted".into());
        }

        let accepted = &export.accepted_focrq;
        if accepted.bytes != 86_168_002
            || accepted.sha256 != F32_FOCRQ_SHA256
            || accepted.format_version != 1
            || accepted.model_id != "tromr"
            || accepted.source_sha256 != FOLDED_SAFETENSORS_SHA256
            || accepted.tensor_count != 260
            || accepted.value_bytes != 86_135_888
            || accepted.tensor_value_inventory_sha256
                != "59af5214d0f73eceaba23dc5e7c60941673a4f642b4577c5e1b4183e49cda200"
        {
            return invalid("export accepted FOCRQ contract drifted".into());
        }

        validate_tensor_inventory(
            "raw checkpoint",
            &export.source_tensor_inventory,
            261,
            21_534_232,
            86_136_928,
            &export.source_pth.tensor_value_inventory_sha256,
        )?;
        validate_tensor_inventory(
            "regenerated safetensors",
            &export.output_tensor_inventory,
            260,
            21_533_972,
            86_135_888,
            &export.tensor_value_inventory_sha256,
        )?;
        validate_tensor_inventory(
            "accepted f32 FOCRQ",
            &export.accepted_tensor_inventory,
            260,
            21_533_972,
            86_135_888,
            &accepted.tensor_value_inventory_sha256,
        )?;

        if export
            .output_tensor_inventory
            .iter()
            .map(|row| (&row.name, &row.dtype, &row.shape, row.bytes))
            .ne(export
                .accepted_tensor_inventory
                .iter()
                .map(|row| (&row.name, &row.dtype, &row.shape, row.bytes)))
        {
            return invalid("export generated/accepted tensor schemas disagree".into());
        }
        let output_by_name: BTreeMap<&str, &TromrTensorInventoryRowV1> = export
            .output_tensor_inventory
            .iter()
            .map(|row| (row.name.as_str(), row))
            .collect();
        let accepted_by_name: BTreeMap<&str, &TromrTensorInventoryRowV1> = export
            .accepted_tensor_inventory
            .iter()
            .map(|row| (row.name.as_str(), row))
            .collect();
        let inventory_difference_names: BTreeSet<&str> = export
            .output_tensor_inventory
            .iter()
            .zip(&export.accepted_tensor_inventory)
            .filter(|(generated, accepted)| generated.sha256 != accepted.sha256)
            .map(|(generated, _)| generated.name.as_str())
            .collect();
        let comparison = &export.accepted_value_comparison;
        if !comparison.names_shapes_dtypes_equal
            || comparison.exact_value_tensor_count != 220
            || comparison.tolerance_value_tensor_count != 40
            || comparison.differences.len() != 40
            || comparison.max_abs > export.accepted_max_abs_contract
            || !comparison.max_abs.is_finite()
            || !comparison.mean_of_tensor_mean_abs.is_finite()
        {
            return invalid("export value-tolerance comparison drifted".into());
        }
        let folded: BTreeSet<&str> = export
            .ws_fold
            .folded_convs
            .iter()
            .map(String::as_str)
            .collect();
        let mut difference_names = BTreeSet::new();
        let mut different_elements = 0u64;
        let mut recomputed_max_abs = 0.0f64;
        let mut recomputed_mean_abs_sum = 0.0f64;
        for difference in &comparison.differences {
            require_sha256(
                "export generated tensor difference",
                &difference.generated_sha256,
            )?;
            require_sha256(
                "export accepted tensor difference",
                &difference.accepted_sha256,
            )?;
            if !difference_names.insert(difference.name.as_str())
                || !difference.max_abs.is_finite()
                || !difference.mean_abs.is_finite()
                || difference.max_abs <= 0.0
                || difference.mean_abs <= 0.0
                || difference.mean_abs > difference.max_abs
                || difference.max_abs > export.accepted_max_abs_contract
                || difference.different_elements == 0
            {
                return invalid("export tensor-difference row is invalid".into());
            }
            let generated = output_by_name.get(difference.name.as_str()).ok_or_else(|| {
                FocrError::FormatMismatch(format!(
                    "TrOMR lineage validation failed: export difference {:?} is absent from generated inventory",
                    difference.name
                ))
            })?;
            let accepted = accepted_by_name.get(difference.name.as_str()).ok_or_else(|| {
                FocrError::FormatMismatch(format!(
                    "TrOMR lineage validation failed: export difference {:?} is absent from accepted inventory",
                    difference.name
                ))
            })?;
            if difference.generated_sha256 != generated.sha256
                || difference.accepted_sha256 != accepted.sha256
                || generated.sha256 == accepted.sha256
                || difference.different_elements > generated.bytes / 4
            {
                return invalid(format!(
                    "export difference {:?} does not bind its generated/accepted inventory rows",
                    difference.name
                ));
            }
            recomputed_max_abs = recomputed_max_abs.max(difference.max_abs);
            recomputed_mean_abs_sum += difference.mean_abs;
            different_elements = different_elements
                .checked_add(difference.different_elements)
                .ok_or_else(|| {
                    FocrError::FormatMismatch(
                        "TrOMR export different-element count overflows".into(),
                    )
                })?;
        }
        let recomputed_mean_abs = recomputed_mean_abs_sum / comparison.differences.len() as f64;
        if difference_names != folded
            || difference_names != inventory_difference_names
            || different_elements != comparison.different_element_count
            || recomputed_max_abs.to_bits() != comparison.max_abs.to_bits()
            || recomputed_mean_abs.to_bits() != comparison.mean_of_tensor_mean_abs.to_bits()
            || export.license
                != "Apache-2.0 (NetEase Polyphonic-TrOMR — NOTICE carried to distribution)"
        {
            return invalid("export difference/license closure drifted".into());
        }
        Ok(())
    }

    fn validate_provider_source_bytes(&self) -> FocrResult<()> {
        for source in &self.provider_sources {
            let bytes: &[u8] = match source.path.as_str() {
                "Cargo.toml" => include_bytes!("../../Cargo.toml"),
                "scripts/gen_tromr_safetensors.py" => {
                    include_bytes!("../../scripts/gen_tromr_safetensors.py")
                }
                "scripts/tromr_convert_e2e.sh" => {
                    include_bytes!("../../scripts/tromr_convert_e2e.sh")
                }
                "src/native_engine/tromr_upstream_history_audit.json" => {
                    HISTORY_AUDIT_JSON.as_bytes()
                }
                "src/native_engine/tromr_export_receipt.json" => EXPORT_RECEIPT_JSON.as_bytes(),
                "src/quant/convert.rs" => include_bytes!("../quant/convert.rs"),
                "docs/zoo/tromr-spec.md" => include_bytes!("../../docs/zoo/tromr-spec.md"),
                "models/manifest-v2.json" => include_bytes!("../../models/manifest-v2.json"),
                other => {
                    return invalid(format!(
                        "provider source {other:?} has no embedded-byte verifier"
                    ));
                }
            };
            let actual = hex_sha256(bytes);
            if actual != source.sha256 {
                return invalid(format!(
                    "provider source {:?} SHA-256 is {actual}, receipt declares {}",
                    source.path, source.sha256
                ));
            }
        }
        Ok(())
    }

    fn validate_distribution_manifest(&self) -> FocrResult<()> {
        let manifest = dist::builtin_manifest()?;
        let tromr = manifest.models.get("tromr").ok_or_else(|| {
            FocrError::FormatMismatch(
                "TrOMR lineage cannot resolve models.tromr in manifest-v2".into(),
            )
        })?;
        exact(
            "distribution TrOMR license",
            &tromr.license_notice,
            "Polyphonic-TrOMR (NetEase) - Apache-2.0",
        )?;

        for (tag, role, recipe) in [
            ("f32", TromrArtifactRoleV1::F32Focrq, TROMR_F32_RECIPE),
            ("int8", TromrArtifactRoleV1::Int8Focrq, TROMR_INT8_RECIPE),
        ] {
            let quant = tromr.quants.get(tag).ok_or_else(|| {
                FocrError::FormatMismatch(format!(
                    "TrOMR distribution manifest omits quant {tag:?}"
                ))
            })?;
            let expected = self
                .artifact(role)
                .expect("artifact census already checked");
            let expected_recipe = expected.recipe.as_deref().ok_or_else(|| {
                FocrError::FormatMismatch(format!(
                    "TrOMR lineage validation failed: lineage {tag} recipe is absent"
                ))
            })?;
            exact(&format!("lineage {tag} recipe"), expected_recipe, recipe)?;
            exact(
                &format!("distribution {tag} recipe"),
                &quant.recipe,
                expected_recipe,
            )?;
            exact(
                &format!("distribution {tag} filename"),
                &quant.focrq.filename,
                &expected.filename,
            )?;
            exact(
                &format!("distribution {tag} SHA-256"),
                &quant.focrq.sha256,
                &expected.sha256,
            )?;
            if Some(quant.focrq.size) != expected.bytes {
                return invalid(format!(
                    "distribution {tag} bytes are {}, expected {:?}",
                    quant.focrq.size, expected.bytes
                ));
            }
        }

        let mut distributed = vec![&tromr.tokenizer];
        distributed.extend(tromr.sidecars.iter());
        for expected in &self.tokenizers {
            let actual = distributed
                .iter()
                .find(|file| file.filename == expected.filename)
                .ok_or_else(|| {
                    FocrError::FormatMismatch(format!(
                        "TrOMR distribution manifest omits tokenizer {:?}",
                        expected.filename
                    ))
                })?;
            exact(
                &format!("distribution tokenizer {} SHA-256", expected.filename),
                &actual.sha256,
                &expected.sha256,
            )?;
            if Some(actual.size) != expected.bytes {
                return invalid(format!(
                    "distribution tokenizer {} bytes are {}, expected {:?}",
                    expected.filename, actual.size, expected.bytes
                ));
            }
        }
        Ok(())
    }
}

fn parse_and_verify_embedded() -> Result<TromrLineageReceiptV1, String> {
    let raw_sha256 = hex_sha256(LINEAGE_JSON.as_bytes());
    if raw_sha256 != LINEAGE_JSON_SHA256 {
        return Err(format!(
            "embedded JSON SHA-256 is {raw_sha256}, expected {LINEAGE_JSON_SHA256}"
        ));
    }
    let receipt: TromrLineageReceiptV1 =
        serde_json::from_str(LINEAGE_JSON).map_err(|error| error.to_string())?;
    receipt
        .validate_accepted_baseline()
        .map_err(|error| error.to_string())?;
    Ok(receipt)
}

fn exact(field: &str, actual: &str, expected: &str) -> FocrResult<()> {
    if actual == expected {
        Ok(())
    } else {
        invalid(format!(
            "{field} is {actual:?}, expected pinned value {expected:?}"
        ))
    }
}

fn nonempty(field: &str, value: &str) -> FocrResult<()> {
    if value.trim().is_empty() {
        invalid(format!("{field} is empty"))
    } else {
        Ok(())
    }
}

fn validate_tensor_inventory(
    label: &str,
    rows: &[TromrTensorInventoryRowV1],
    expected_count: u32,
    expected_parameters: u64,
    expected_bytes: u64,
    expected_inventory_sha256: &str,
) -> FocrResult<()> {
    if rows.len() != expected_count as usize {
        return invalid(format!(
            "{label} tensor inventory has {} rows, expected {expected_count}",
            rows.len()
        ));
    }
    require_sha256(
        &format!("{label} tensor inventory identity"),
        expected_inventory_sha256,
    )?;

    let mut previous_name: Option<&str> = None;
    let mut parameter_count = 0u64;
    let mut value_bytes = 0u64;
    let mut canonical_rows = Vec::with_capacity(rows.len());
    for row in rows {
        nonempty(&format!("{label} tensor name"), &row.name)?;
        if previous_name.is_some_and(|previous| previous >= row.name.as_str()) {
            return invalid(format!(
                "{label} tensor inventory names are not unique lexical order"
            ));
        }
        previous_name = Some(row.name.as_str());
        exact(
            &format!("{label} tensor {} dtype", row.name),
            &row.dtype,
            "float32",
        )?;
        if row.shape.is_empty() || row.shape.contains(&0) {
            return invalid(format!(
                "{label} tensor {:?} has an empty or zero-dimensional shape",
                row.name
            ));
        }
        let row_parameters = row
            .shape
            .iter()
            .try_fold(1u64, |product, dimension| product.checked_mul(*dimension))
            .ok_or_else(|| {
                FocrError::FormatMismatch(format!(
                    "TrOMR lineage validation failed: {label} tensor {:?} shape overflows",
                    row.name
                ))
            })?;
        let row_bytes = row_parameters.checked_mul(4).ok_or_else(|| {
            FocrError::FormatMismatch(format!(
                "TrOMR lineage validation failed: {label} tensor {:?} byte count overflows",
                row.name
            ))
        })?;
        if row.bytes != row_bytes {
            return invalid(format!(
                "{label} tensor {:?} declares {} bytes, shape requires {row_bytes}",
                row.name, row.bytes
            ));
        }
        require_sha256(&format!("{label} tensor {}", row.name), &row.sha256)?;
        parameter_count = parameter_count.checked_add(row_parameters).ok_or_else(|| {
            FocrError::FormatMismatch(format!(
                "TrOMR lineage validation failed: {label} parameter count overflows"
            ))
        })?;
        value_bytes = value_bytes.checked_add(row.bytes).ok_or_else(|| {
            FocrError::FormatMismatch(format!(
                "TrOMR lineage validation failed: {label} byte count overflows"
            ))
        })?;

        let mut canonical = BTreeMap::new();
        canonical.insert("bytes", serde_json::json!(row.bytes));
        canonical.insert("dtype", serde_json::json!(row.dtype));
        canonical.insert("name", serde_json::json!(row.name));
        canonical.insert("sha256", serde_json::json!(row.sha256));
        canonical.insert("shape", serde_json::json!(row.shape));
        canonical_rows.push(canonical);
    }
    if parameter_count != expected_parameters || value_bytes != expected_bytes {
        return invalid(format!(
            "{label} tensor totals are {parameter_count} parameters/{value_bytes} bytes, expected {expected_parameters}/{expected_bytes}"
        ));
    }
    let canonical = serde_json::to_vec(&canonical_rows).map_err(|error| {
        FocrError::FormatMismatch(format!(
            "TrOMR {label} tensor inventory serialization failed: {error}"
        ))
    })?;
    let actual_inventory_sha256 = hex_sha256(&canonical);
    if actual_inventory_sha256 != expected_inventory_sha256 {
        return invalid(format!(
            "{label} tensor inventory is {actual_inventory_sha256}, expected {expected_inventory_sha256}"
        ));
    }
    Ok(())
}

fn require_sha256(field: &str, value: &str) -> FocrResult<()> {
    if is_lower_hex(value, 64) {
        Ok(())
    } else {
        invalid(format!(
            "{field} is not 64 lowercase hexadecimal characters"
        ))
    }
}

fn require_git_oid(field: &str, value: &str) -> FocrResult<()> {
    if is_lower_hex(value, 40) {
        Ok(())
    } else {
        invalid(format!("{field} is not a 40-character Git object id"))
    }
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hex_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn invalid<T>(message: String) -> FocrResult<T> {
    Err(FocrError::FormatMismatch(format!(
        "TrOMR lineage validation failed: {message}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedded_value() -> serde_json::Value {
        serde_json::from_str(LINEAGE_JSON).expect("embedded lineage JSON parses as a value")
    }

    fn mutate(mut value: serde_json::Value, edit: impl FnOnce(&mut serde_json::Value)) -> String {
        edit(&mut value);
        serde_json::to_string(&value).expect("mutated lineage serializes")
    }

    #[test]
    fn embedded_lineage_is_canonical_complete_and_honestly_negative() {
        let receipt = tromr_lineage_receipt().expect("embedded lineage validates");
        assert_eq!(receipt.schema, TROMR_LINEAGE_SCHEMA_VERSION);
        assert_eq!(
            receipt.canonical_sha256().expect("canonical hash"),
            TROMR_LINEAGE_CANONICAL_SHA256
        );
        assert!(!receipt.training_availability.published_training_code);
        assert_eq!(receipt.training_availability.fields.len(), 29);
        assert_eq!(
            receipt
                .training_field("published_training_entrypoint")
                .expect("training entrypoint disposition")
                .evidence_class,
            TromrEvidenceClassV1::UnavailableUpstream
        );
        assert_eq!(
            receipt
                .training_field("optimizer_family")
                .expect("paper optimizer fact")
                .evidence_class,
            TromrEvidenceClassV1::OfficialPaper
        );
    }

    #[test]
    fn canonical_round_trip_is_whitespace_and_host_independent() {
        let receipt = tromr_lineage_receipt().expect("embedded lineage validates");
        let canonical = receipt.canonical_json().expect("canonical JSON");
        let parsed = TromrLineageReceiptV1::from_json(&format!("\n  {canonical}\n"))
            .expect("formatted canonical receipt validates");
        assert_eq!(parsed, *receipt);
        assert_eq!(
            parsed.canonical_sha256().expect("round-trip hash"),
            TROMR_LINEAGE_CANONICAL_SHA256
        );
        assert!(!canonical.contains(env!("CARGO_MANIFEST_DIR")));
    }

    #[test]
    fn portable_validation_does_not_silently_bless_a_different_baseline() {
        let json = mutate(embedded_value(), |value| {
            value["contract_id"] = "independent-portable-tromr-lineage".into();
        });
        let portable = TromrLineageReceiptV1::from_json(&json)
            .expect("well-formed independent receipt remains portable");
        assert_eq!(portable.contract_id, "independent-portable-tromr-lineage");

        let error = TromrLineageReceiptV1::from_accepted_json(&json)
            .expect_err("portable receipt must not inherit accepted-baseline authority");
        assert!(error.to_string().contains("contract_id"), "{error}");
    }

    #[test]
    fn observed_artifact_verification_binds_role_bytes_hash_source_recipe_and_count() {
        let receipt = tromr_lineage_receipt().expect("embedded lineage validates");
        let expected = receipt
            .artifact(TromrArtifactRoleV1::Int8Focrq)
            .expect("int8 artifact");
        let baseline = TromrObservedArtifactV1 {
            role: expected.role,
            filename: expected.filename.clone(),
            bytes: expected.bytes.expect("runtime artifact has pinned bytes"),
            sha256: expected.sha256.clone(),
            source_sha256: expected.source_sha256.clone(),
            declared_recipe: expected.recipe.clone(),
            tensor_count: expected.tensor_count,
        };
        receipt
            .verify_observed_artifact(&baseline)
            .expect("exact observation accepted");

        let cases: [(&str, Box<dyn Fn(&mut TromrObservedArtifactV1)>); 6] = [
            ("filename", Box::new(|item| item.filename.push('x'))),
            ("bytes", Box::new(|item| item.bytes += 1)),
            ("sha", Box::new(|item| item.sha256.replace_range(0..1, "0"))),
            (
                "source",
                Box::new(|item| item.source_sha256 = Some("0".repeat(64))),
            ),
            (
                "recipe",
                Box::new(|item| item.declared_recipe = Some("other".into())),
            ),
            ("count", Box::new(|item| item.tensor_count = Some(259))),
        ];
        for (name, edit) in cases {
            let mut changed = baseline.clone();
            edit(&mut changed);
            let error = receipt
                .verify_observed_artifact(&changed)
                .expect_err("mutated observation must fail");
            assert!(
                error.to_string().contains("observed"),
                "{name}: unexpected error: {error}"
            );
        }
    }

    #[test]
    fn finite_authority_and_schema_mutations_fail_closed_with_named_reasons() {
        let cases = [
            (
                "community fork",
                mutate(embedded_value(), |value| {
                    value["upstream"]["repository_url"] =
                        "https://github.com/community/Polyphonic-TrOMR".into();
                }),
                "upstream.repository_url",
            ),
            (
                "mutable ref",
                mutate(embedded_value(), |value| {
                    value["upstream"]["commit"] = "master".into();
                }),
                "upstream.commit",
            ),
            (
                "checkpoint substitution",
                mutate(embedded_value(), |value| {
                    value["checkpoint"]["sha256"] = "0".repeat(64).into();
                    value["conversion_artifacts"][0]["source_sha256"] = "0".repeat(64).into();
                }),
                "raw checkpoint SHA-256",
            ),
            (
                "tokenizer reorder",
                mutate(embedded_value(), |value| {
                    value["tokenizers"].as_array_mut().unwrap().swap(0, 1);
                }),
                "tokenizer role order",
            ),
            (
                "fabricated training availability",
                mutate(embedded_value(), |value| {
                    value["training_availability"]["published_training_code"] = true.into();
                }),
                "falsely claims",
            ),
            (
                "laundered missing field",
                mutate(embedded_value(), |value| {
                    value["training_availability"]["fields"][0]["evidence_class"] =
                        "official_source".into();
                    value["training_availability"]["fields"][0]["value"] = "guessed.py".into();
                }),
                "training field",
            ),
            (
                "conversion discontinuity",
                mutate(embedded_value(), |value| {
                    value["runtime_artifacts"][0]["source_sha256"] = "0".repeat(64).into();
                }),
                "output source_sha256",
            ),
            (
                "license substitution",
                mutate(embedded_value(), |value| {
                    value["upstream"]["license_sha256"] = "0".repeat(64).into();
                }),
                "upstream.license_sha256",
            ),
            (
                "paper substitution",
                mutate(embedded_value(), |value| {
                    value["paper"]["pdf_sha256"] = "0".repeat(64).into();
                }),
                "paper.pdf_sha256",
            ),
            (
                "provider source substitution",
                mutate(embedded_value(), |value| {
                    value["provider_sources"][0]["sha256"] = "0".repeat(64).into();
                }),
                "provider source inventory",
            ),
            (
                "history receipt substitution",
                mutate(embedded_value(), |value| {
                    value["upstream"]["history_audit_sha256"] = "0".repeat(64).into();
                }),
                "upstream.history_audit_sha256",
            ),
            (
                "missing export receipt owner",
                mutate(embedded_value(), |value| {
                    value["provider_sources"]
                        .as_array_mut()
                        .expect("provider source array")
                        .remove(4);
                }),
                "provider source inventory",
            ),
            (
                "runtime recipe substitution",
                mutate(embedded_value(), |value| {
                    value["runtime_artifacts"][0]["recipe"] = "untracked-recipe".into();
                }),
                "lineage f32 recipe",
            ),
            (
                "unavailable field without evidence",
                mutate(embedded_value(), |value| {
                    value["training_availability"]["fields"][0]["evidence_refs"] =
                        serde_json::json!([]);
                }),
                "training field",
            ),
            (
                "regenerated export substitution",
                mutate(embedded_value(), |value| {
                    value["conversion_replay"]["regenerated_safetensors_sha256"] =
                        "0".repeat(64).into();
                }),
                "conversion replay regenerated SHA-256",
            ),
        ];
        for (name, json, needle) in cases {
            let error = TromrLineageReceiptV1::from_accepted_json(&json)
                .expect_err("finite lineage mutation must fail");
            assert!(
                error.to_string().contains(needle),
                "{name}: expected {needle:?}, got {error}"
            );
        }

        let duplicate = LINEAGE_JSON.replacen(
            "\"schema\": \"franken_ocr.tromr_lineage.v1\",",
            "\"schema\": \"franken_ocr.tromr_lineage.v1\",\n  \"schema\": \"duplicate\",",
            1,
        );
        let error = TromrLineageReceiptV1::from_json(&duplicate)
            .expect_err("duplicate known field must fail");
        assert!(error.to_string().contains("duplicate field"));

        let unknown = LINEAGE_JSON.replacen(
            "\"schema\": \"franken_ocr.tromr_lineage.v1\",",
            "\"schema\": \"franken_ocr.tromr_lineage.v1\",\n  \"mystery\": true,",
            1,
        );
        let error =
            TromrLineageReceiptV1::from_json(&unknown).expect_err("unknown field must fail");
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn export_receipt_reconstructs_inventory_and_difference_aggregates() {
        let lineage = tromr_lineage_receipt().expect("embedded lineage validates");
        let baseline: TromrExportReceiptV1 =
            serde_json::from_str(EXPORT_RECEIPT_JSON).expect("export receipt parses");
        lineage
            .validate_export_receipt(&baseline)
            .expect("full export receipt validates");

        let cases = [
            (
                "difference hash substitution",
                mutate(
                    serde_json::from_str(EXPORT_RECEIPT_JSON).expect("export JSON value"),
                    |value| {
                        value["accepted_value_comparison"]["differences"][0]["generated_sha256"] =
                            "0".repeat(64).into();
                    },
                ),
                "does not bind",
            ),
            (
                "aggregate substitution",
                mutate(
                    serde_json::from_str(EXPORT_RECEIPT_JSON).expect("export JSON value"),
                    |value| {
                        value["accepted_value_comparison"]["mean_of_tensor_mean_abs"] = 0.into();
                    },
                ),
                "difference/license closure",
            ),
            (
                "per-tensor tolerance breach",
                mutate(
                    serde_json::from_str(EXPORT_RECEIPT_JSON).expect("export JSON value"),
                    |value| {
                        value["accepted_value_comparison"]["differences"][0]["max_abs"] =
                            serde_json::json!(0.01);
                    },
                ),
                "tensor-difference row",
            ),
            (
                "inventory value hash substitution",
                mutate(
                    serde_json::from_str(EXPORT_RECEIPT_JSON).expect("export JSON value"),
                    |value| {
                        value["output_tensor_inventory"][0]["sha256"] = "0".repeat(64).into();
                    },
                ),
                "tensor inventory is",
            ),
        ];
        for (name, json, needle) in cases {
            let changed: TromrExportReceiptV1 =
                serde_json::from_str(&json).expect("mutated export receipt remains typed");
            let error = lineage
                .validate_export_receipt(&changed)
                .expect_err("mutated export receipt must fail");
            assert!(
                error.to_string().contains(needle),
                "{name}: expected {needle:?}, got {error}"
            );
        }
    }

    #[test]
    fn distribution_manifest_and_lineage_rows_are_one_contract() {
        let receipt = tromr_lineage_receipt().expect("embedded lineage validates");
        receipt
            .validate_distribution_manifest()
            .expect("distribution rows match lineage");
        assert_eq!(
            receipt
                .artifact(TromrArtifactRoleV1::F32Focrq)
                .and_then(|artifact| artifact.recipe.as_deref()),
            Some(TROMR_F32_RECIPE)
        );
        assert_eq!(
            receipt
                .artifact(TromrArtifactRoleV1::Int8Focrq)
                .and_then(|artifact| artifact.recipe.as_deref()),
            Some(TROMR_INT8_RECIPE)
        );
    }
}
