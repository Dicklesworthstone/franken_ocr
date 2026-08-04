//! Exact production tensor census for the pinned Unlimited-OCR checkpoint.
//!
//! The generic [`super::weights::Weights::census`] API checks names only.  The
//! production contract is stronger: the verified source shard has exactly
//! 2,710 tensors, and the conservative `.focrq` recipe fixes each tensor's
//! logical shape and storage dtype.  This module keeps that complete manifest
//! embedded in the binary and applies the same bounded, named comparison to
//! both the header probe and the post-load directory. This is an exact schema
//! census; payload identity is established separately by artifact SHA-256. The
//! manifest distinguishes the safetensors index's tensor-payload `total_size`
//! from the full shard size (8-byte length prefix + JSON header + payload).

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::Deserialize;

use super::model_arch;
use super::weights::{DType, TensorRecord, Weights};
use crate::error::{FocrError, FocrResult};
use crate::quant::convert::UNLIMITED_OCR_INT8_RECIPE_ID;
use crate::quant::recipe::Recipe;

const MANIFEST_JSON: &str = include_str!("unlimited_ocr_manifest.json");
const MANIFEST_SHA256: &str = "24ff1cfffe71eec6f07bfae8e8eb12b342877bccc43ad6ae4a4a4ceffb76edd3";
const MANIFEST_SCHEMA: &str = "franken_ocr.unlimited_ocr_tensor_manifest.v1";
const HF_COMMIT: &str = "3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5";
const SOURCE_FILE: &str = "model-00001-of-000001.safetensors";
const SOURCE_SHA256: &str = "2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6";
const SOURCE_INDEX_SHA256: &str =
    "354be1f2dcfb72ebb385e25465522ce5413a77c36f3b35fec088a3162a11af99";
const VERIFIED_ARTIFACT_SHA256: &str =
    "573340710167697891bf52dfa4cbb5d0a02a68f3011c01f8ef83fd34622fb592";
const SOURCE_INDEX_TOTAL_SIZE: u64 = 6_672_212_480;
const SOURCE_FILE_SIZE: u64 = 6_672_547_120;
const TENSOR_COUNT: usize = 2_710;
const DIFF_PREVIEW_LIMIT: usize = 6;

/// Refuse conversion inputs whose output the production artifact loader would
/// later reject. The conservative Unlimited-OCR format is bound to the exact
/// pinned source shard, not merely a same-shaped safetensors serialization.
pub(crate) fn validate_conversion_source_sha256(actual: &[u8; 32]) -> FocrResult<()> {
    let actual = actual
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if actual == SOURCE_SHA256 {
        Ok(())
    } else {
        Err(FocrError::FormatMismatch(format!(
            "Unlimited-OCR conversion requires pinned source shard SHA-256 {SOURCE_SHA256}; got {actual}. Supply the canonical checkpoint or use raw compatible weights directly"
        )))
    }
}

#[derive(Debug, Deserialize)]
struct UnlimitedOcrManifest {
    schema: String,
    model_id: String,
    hf_commit: String,
    source_safetensors: String,
    source_sha256: String,
    source_index_total_size: u64,
    source_file_size: u64,
    source_index_sha256: String,
    tensor_count: usize,
    quant_recipe: String,
    verified_conservative_artifact_sha256: String,
    tensors: BTreeMap<String, ManifestTensor>,
}

#[derive(Debug, Deserialize)]
struct ManifestTensor {
    shape: Vec<usize>,
    source_dtype: DType,
    storage_dtype: DType,
}

static MANIFEST: OnceLock<Result<UnlimitedOcrManifest, String>> = OnceLock::new();

fn manifest() -> FocrResult<&'static UnlimitedOcrManifest> {
    MANIFEST
        .get_or_init(parse_and_verify_manifest)
        .as_ref()
        .map_err(|message| {
            FocrError::FormatMismatch(format!(
                "embedded Unlimited-OCR tensor manifest is invalid: {message}"
            ))
        })
}

fn parse_and_verify_manifest() -> Result<UnlimitedOcrManifest, String> {
    use sha2::Digest as _;

    let manifest_sha256 = format!("{:x}", sha2::Sha256::digest(MANIFEST_JSON.as_bytes()));
    if manifest_sha256 != MANIFEST_SHA256 {
        return Err(format!(
            "embedded JSON sha256 is {manifest_sha256}, expected {MANIFEST_SHA256}"
        ));
    }
    let manifest: UnlimitedOcrManifest =
        serde_json::from_str(MANIFEST_JSON).map_err(|error| error.to_string())?;

    let metadata_checks = [
        ("schema", manifest.schema.as_str(), MANIFEST_SCHEMA),
        (
            "model_id",
            manifest.model_id.as_str(),
            model_arch::default_arch().id(),
        ),
        ("hf_commit", manifest.hf_commit.as_str(), HF_COMMIT),
        (
            "source_safetensors",
            manifest.source_safetensors.as_str(),
            SOURCE_FILE,
        ),
        (
            "source_sha256",
            manifest.source_sha256.as_str(),
            SOURCE_SHA256,
        ),
        (
            "source_index_sha256",
            manifest.source_index_sha256.as_str(),
            SOURCE_INDEX_SHA256,
        ),
        (
            "quant_recipe",
            manifest.quant_recipe.as_str(),
            UNLIMITED_OCR_INT8_RECIPE_ID,
        ),
        (
            "verified_conservative_artifact_sha256",
            manifest.verified_conservative_artifact_sha256.as_str(),
            VERIFIED_ARTIFACT_SHA256,
        ),
    ];
    for (field, actual, expected) in metadata_checks {
        if actual != expected {
            return Err(format!(
                "{field} is {actual:?}, expected pinned value {expected:?}"
            ));
        }
    }
    if manifest.source_index_total_size != SOURCE_INDEX_TOTAL_SIZE {
        return Err(format!(
            "source_index_total_size is {}, expected {SOURCE_INDEX_TOTAL_SIZE}",
            manifest.source_index_total_size
        ));
    }
    if manifest.source_file_size != SOURCE_FILE_SIZE {
        return Err(format!(
            "source_file_size is {}, expected {SOURCE_FILE_SIZE}",
            manifest.source_file_size
        ));
    }
    if manifest.tensor_count != TENSOR_COUNT || manifest.tensors.len() != TENSOR_COUNT {
        return Err(format!(
            "tensor_count declares {} and directory contains {}, expected {TENSOR_COUNT}",
            manifest.tensor_count,
            manifest.tensors.len()
        ));
    }

    let recipe = Recipe::validated_default();
    for (name, tensor) in &manifest.tensors {
        if tensor.shape.is_empty() || tensor.shape.contains(&0) {
            return Err(format!(
                "tensor {name:?} has invalid shape {:?}",
                tensor.shape
            ));
        }
        if tensor.source_dtype != DType::BF16 {
            return Err(format!(
                "tensor {name:?} source dtype is {:?}, expected BF16",
                tensor.source_dtype
            ));
        }
        let expected_storage = if recipe.is_quantized(name) {
            DType::QInt8PerChan
        } else {
            DType::BF16
        };
        if tensor.storage_dtype != expected_storage {
            return Err(format!(
                "tensor {name:?} storage dtype is {:?}, recipe requires {expected_storage:?}",
                tensor.storage_dtype
            ));
        }
    }
    Ok(manifest)
}

#[derive(Clone, Copy)]
enum CensusSurface {
    SourceSafetensors,
    ConservativeFocrq,
}

impl CensusSurface {
    fn label(self) -> &'static str {
        match self {
            Self::SourceSafetensors => "source safetensors",
            Self::ConservativeFocrq => "conservative .focrq",
        }
    }

    fn expected_dtype(self, tensor: &ManifestTensor) -> DType {
        match self {
            Self::SourceSafetensors => tensor.source_dtype,
            Self::ConservativeFocrq => tensor.storage_dtype,
        }
    }
}

#[derive(Default)]
struct NamedDiff {
    total: usize,
    examples: Vec<String>,
}

impl NamedDiff {
    fn push(&mut self, example: String) {
        self.total += 1;
        if self.examples.len() < DIFF_PREVIEW_LIMIT {
            self.examples.push(example);
        }
    }

    fn append_to(&self, output: &mut String, label: &str) {
        if self.total == 0 {
            return;
        }
        use std::fmt::Write;
        let _ = write!(
            output,
            "; {label} {}: [{}",
            self.total,
            self.examples.join(", ")
        );
        if self.total > self.examples.len() {
            let _ = write!(output, ", ... {} more", self.total - self.examples.len());
        }
        output.push(']');
    }
}

fn validate_records<'a>(
    records: impl IntoIterator<Item = (&'a str, &'a TensorRecord)>,
    surface: CensusSurface,
) -> FocrResult<()> {
    let manifest = manifest()?;
    let actual: BTreeMap<&str, &TensorRecord> = records.into_iter().collect();
    let mut missing = NamedDiff::default();
    let mut unexpected = NamedDiff::default();
    let mut wrong_shape = NamedDiff::default();
    let mut wrong_dtype = NamedDiff::default();

    for (name, expected) in &manifest.tensors {
        let Some(record) = actual.get(name.as_str()) else {
            missing.push(name.clone());
            continue;
        };
        if record.shape != expected.shape {
            wrong_shape.push(format!(
                "{name} expected {:?} got {:?}",
                expected.shape, record.shape
            ));
        }
        let expected_dtype = surface.expected_dtype(expected);
        if record.dtype != expected_dtype {
            wrong_dtype.push(format!(
                "{name} expected {expected_dtype:?} got {:?}",
                record.dtype
            ));
        }
    }
    for name in actual.keys() {
        if !manifest.tensors.contains_key(*name) {
            unexpected.push((*name).to_owned());
        }
    }

    if missing.total == 0
        && unexpected.total == 0
        && wrong_shape.total == 0
        && wrong_dtype.total == 0
    {
        return Ok(());
    }

    let mut message = format!(
        "Unlimited-OCR {} tensor census failed: expected {TENSOR_COUNT} tensors, found {}",
        surface.label(),
        actual.len()
    );
    missing.append_to(&mut message, "missing");
    unexpected.append_to(&mut message, "unexpected");
    wrong_shape.append_to(&mut message, "wrong shape");
    wrong_dtype.append_to(&mut message, "wrong dtype");
    Err(FocrError::FormatMismatch(message))
}

/// Validate the bounded header of the pinned raw source shard.
pub(super) fn validate_source_header(records: &BTreeMap<String, TensorRecord>) -> FocrResult<()> {
    validate_records(
        records.iter().map(|(name, record)| (name.as_str(), record)),
        CensusSurface::SourceSafetensors,
    )
}

/// Validate identity, recipe, and complete tensor schema of a conservative
/// Unlimited-OCR `.focrq` bounded header.
pub(super) fn validate_focrq_header(
    records: &BTreeMap<String, TensorRecord>,
    source_sha256: &str,
    quant_recipe: Option<&str>,
) -> FocrResult<()> {
    let manifest = manifest()?;
    if source_sha256 != manifest.source_sha256 {
        return Err(FocrError::FormatMismatch(format!(
            "Unlimited-OCR .focrq source_sha256 is {source_sha256:?}, expected pinned source {:?}",
            manifest.source_sha256
        )));
    }
    if quant_recipe != Some(manifest.quant_recipe.as_str()) {
        return Err(FocrError::FormatMismatch(format!(
            "Unlimited-OCR .focrq packing_manifest.quant_recipe must be exactly {:?}, got {quant_recipe:?}",
            manifest.quant_recipe
        )));
    }
    validate_records(
        records.iter().map(|(name, record)| (name.as_str(), record)),
        CensusSurface::ConservativeFocrq,
    )
}

/// Re-run the complete schema check against the directory produced by the real
/// mmap/owned loader, immediately before the production forward receives it.
/// The bounded header path separately verifies the recipe string on the same
/// open descriptor; this post-load check proves name/shape/dtype parity survived
/// the parser handoff.
pub(super) fn validate_loaded_weights(weights: &Weights) -> FocrResult<()> {
    if weights.model_id() != model_arch::default_arch().id() {
        return Ok(());
    }
    let surface = if weights.is_focrq() {
        if weights.source_sha256() != SOURCE_SHA256 {
            return Err(FocrError::FormatMismatch(format!(
                "Unlimited-OCR .focrq source_sha256 is {:?}, expected pinned source {SOURCE_SHA256:?}",
                weights.source_sha256()
            )));
        }
        CensusSurface::ConservativeFocrq
    } else {
        CensusSurface::SourceSafetensors
    };
    let mut records = Vec::with_capacity(weights.len());
    for name in weights.names() {
        let Some(record) = weights.record(name) else {
            return Err(FocrError::FormatMismatch(format!(
                "weights directory lost tensor {name:?} while running the production census"
            )));
        };
        records.push((name, record));
    }
    validate_records(records, surface)
}

#[cfg(test)]
mod tests {
    use std::io::Read;
    use std::path::PathBuf;

    use sha2::{Digest, Sha256};

    use super::*;

    fn sha256_file(path: &std::path::Path) -> String {
        let mut file = std::fs::File::open(path).expect("open real model artifact");
        let mut digest = Sha256::new();
        let mut buffer = vec![0u8; 16 * 1024 * 1024];
        loop {
            let read = file.read(&mut buffer).expect("hash real model artifact");
            if read == 0 {
                break;
            }
            digest.update(&buffer[..read]);
        }
        format!("{:x}", digest.finalize())
    }

    fn synthetic_directory(surface: CensusSurface) -> BTreeMap<String, TensorRecord> {
        manifest()
            .expect("embedded production manifest parses")
            .tensors
            .iter()
            .map(|(name, tensor)| {
                (
                    name.clone(),
                    TensorRecord {
                        dtype: surface.expected_dtype(tensor),
                        shape: tensor.shape.clone(),
                        byte_offset: 0,
                        byte_len: 0,
                        scales_offset: 0,
                        scales_len: 0,
                        group_size: 0,
                        tier: 0,
                    },
                )
            })
            .collect()
    }

    #[test]
    fn manifest_matches_pinned_census_and_recipe_exactly() {
        let manifest = manifest().expect("embedded production manifest parses");
        assert_eq!(manifest.source_index_sha256, SOURCE_INDEX_SHA256);
        assert_eq!(manifest.source_index_total_size, SOURCE_INDEX_TOTAL_SIZE);
        assert_eq!(manifest.source_file_size, SOURCE_FILE_SIZE);
        assert_eq!(manifest.tensors.len(), TENSOR_COUNT);
        for landmark in [
            "lm_head.weight",
            "model.embed_tokens.weight",
            "model.layers.11.mlp.experts.63.down_proj.weight",
            "model.vision_model.transformer.layers.23.self_attn.out_proj.weight",
        ] {
            assert!(
                manifest.tensors.contains_key(landmark),
                "embedded production census omits {landmark}"
            );
        }

        let int8_count = manifest
            .tensors
            .values()
            .filter(|tensor| tensor.storage_dtype == DType::QInt8PerChan)
            .count();
        let bf16_count = manifest
            .tensors
            .values()
            .filter(|tensor| tensor.storage_dtype == DType::BF16)
            .count();
        assert_eq!((int8_count, bf16_count), (2_148, 562));
    }

    #[test]
    fn census_rejects_missing_extra_rename_shape_and_dtype_with_names() {
        let valid = synthetic_directory(CensusSurface::ConservativeFocrq);
        validate_records(
            valid.iter().map(|(name, record)| (name.as_str(), record)),
            CensusSurface::ConservativeFocrq,
        )
        .expect("exact synthetic manifest must pass");

        let mut missing = synthetic_directory(CensusSurface::ConservativeFocrq);
        missing.remove("lm_head.weight");
        let error = validate_records(
            missing.iter().map(|(name, record)| (name.as_str(), record)),
            CensusSurface::ConservativeFocrq,
        )
        .expect_err("missing tensor must fail");
        assert!(error.to_string().contains("missing 1: [lm_head.weight]"));

        let mut extra = synthetic_directory(CensusSurface::ConservativeFocrq);
        let record = extra["lm_head.weight"].clone();
        extra.insert("unexpected.weight".into(), record);
        let error = validate_records(
            extra.iter().map(|(name, record)| (name.as_str(), record)),
            CensusSurface::ConservativeFocrq,
        )
        .expect_err("extra tensor must fail");
        assert!(error.to_string().contains("unexpected.weight"));

        let mut renamed = synthetic_directory(CensusSurface::ConservativeFocrq);
        let record = renamed.remove("lm_head.weight").expect("manifest lm_head");
        renamed.insert("lm_heads.weight".into(), record);
        let error = validate_records(
            renamed.iter().map(|(name, record)| (name.as_str(), record)),
            CensusSurface::ConservativeFocrq,
        )
        .expect_err("rename must fail as a bounded missing/extra diff");
        let text = error.to_string();
        assert!(text.contains("lm_head.weight"));
        assert!(text.contains("lm_heads.weight"));

        let mut wrong_shape = synthetic_directory(CensusSurface::ConservativeFocrq);
        wrong_shape
            .get_mut("lm_head.weight")
            .expect("manifest lm_head")
            .shape[0] -= 1;
        let error = validate_records(
            wrong_shape
                .iter()
                .map(|(name, record)| (name.as_str(), record)),
            CensusSurface::ConservativeFocrq,
        )
        .expect_err("wrong shape must fail");
        assert!(error.to_string().contains("wrong shape 1: [lm_head.weight"));

        let mut wrong_dtype = synthetic_directory(CensusSurface::ConservativeFocrq);
        wrong_dtype
            .get_mut("lm_head.weight")
            .expect("manifest lm_head")
            .dtype = DType::QInt8PerChan;
        let error = validate_records(
            wrong_dtype
                .iter()
                .map(|(name, record)| (name.as_str(), record)),
            CensusSurface::ConservativeFocrq,
        )
        .expect_err("wrong dtype must fail");
        assert!(error.to_string().contains("wrong dtype 1: [lm_head.weight"));
    }

    #[test]
    fn focrq_identity_rejects_wrong_source_and_recipe() {
        let directory = synthetic_directory(CensusSurface::ConservativeFocrq);
        validate_focrq_header(
            &directory,
            SOURCE_SHA256,
            Some(UNLIMITED_OCR_INT8_RECIPE_ID),
        )
        .expect("pinned identity must pass");

        let source_error = validate_focrq_header(
            &directory,
            &"00".repeat(32),
            Some(UNLIMITED_OCR_INT8_RECIPE_ID),
        )
        .expect_err("wrong source must fail");
        assert!(source_error.to_string().contains("source_sha256"));

        let recipe_error =
            validate_focrq_header(&directory, SOURCE_SHA256, Some("legacy-full-int8"))
                .expect_err("wrong recipe must fail");
        assert!(recipe_error.to_string().contains("quant_recipe"));
    }

    #[test]
    fn real_conservative_artifact_matches_manifest_when_configured() {
        let Some(path) = std::env::var_os("FOCR_TEST_UNLIMITED_RECIPE_MODEL").map(PathBuf::from)
        else {
            eprintln!(
                "skip-with-SUCCESS: set FOCR_TEST_UNLIMITED_RECIPE_MODEL to the verified .focrq"
            );
            return;
        };

        assert_eq!(sha256_file(&path), VERIFIED_ARTIFACT_SHA256);

        let (file, file_len) = super::super::open_model_file(&path).expect("open model descriptor");
        super::super::validate_model_header_from_reader(&file, file_len)
            .expect("bounded real header must match the production census");
        let weights = Weights::load_opened(file, &path).expect("mmap real artifact");
        validate_loaded_weights(&weights).expect("real loaded directory must match the census");
        assert_eq!(weights.len(), TENSOR_COUNT);
    }

    #[test]
    fn real_source_shard_matches_manifest_when_configured() {
        let Some(path) = std::env::var_os("FOCR_TEST_UNLIMITED_SOURCE_MODEL").map(PathBuf::from)
        else {
            eprintln!(
                "skip-with-SUCCESS: set FOCR_TEST_UNLIMITED_SOURCE_MODEL to the verified shard"
            );
            return;
        };

        assert_eq!(sha256_file(&path), SOURCE_SHA256);
        assert_eq!(
            std::fs::metadata(&path)
                .expect("stat real source shard")
                .len(),
            SOURCE_FILE_SIZE,
            "full source shard size includes its safetensors prefix and JSON header"
        );
        let (file, file_len) =
            super::super::open_model_file(&path).expect("open source descriptor");
        super::super::validate_model_header_from_reader(&file, file_len)
            .expect("bounded source header must match the production census");
        let weights = Weights::load_opened(file, &path).expect("mmap real source shard");
        validate_loaded_weights(&weights).expect("real source directory must match the census");
        assert_eq!(weights.len(), TENSOR_COUNT);
    }

    #[test]
    fn conversion_source_identity_fails_before_emitting_an_unloadable_artifact() {
        let mut expected = [0u8; 32];
        for (index, byte) in expected.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&SOURCE_SHA256[index * 2..index * 2 + 2], 16)
                .expect("source SHA constant is hex");
        }
        validate_conversion_source_sha256(&expected).expect("pinned source accepted");
        let error = validate_conversion_source_sha256(&[0u8; 32])
            .expect_err("reserialized source must fail before conversion output");
        let message = error.to_string();
        assert!(message.contains(SOURCE_SHA256));
        assert!(message.contains(&"0".repeat(64)));
    }
}
