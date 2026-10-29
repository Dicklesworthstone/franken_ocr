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

use super::model_arch::{self, ModelArch, TensorCensus};
use super::shards::Int64Rule;
use super::weights::{DType, TensorRecord, Weights};
use crate::error::{FocrError, FocrResult};
use crate::quant::convert::{UNLIMITED_OCR_INT8_RECIPE_ID, UNLIMITED_OCR_WASM_INT4_RECIPE_ID};
use crate::quant::recipe::{Recipe, WasmInt4Policy, classify_wasm_experts_int4};

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

/// The pinned source shards of `census`'s checkpoint, in index order: each
/// shard's file name and SHA-256. A single-entry list is also the shard name
/// the model resolver looks for in a directory without an index.
pub(crate) fn source_shards(census: TensorCensus) -> &'static [(&'static str, &'static str)] {
    match census {
        TensorCensus::UnlimitedOcrV1 => &[(SOURCE_FILE, SOURCE_SHA256)],
    }
}

/// The I64 tensors `census` accepts only to verify and skip (D8).
pub(crate) fn int64_rules(census: TensorCensus) -> &'static [Int64Rule] {
    match census {
        TensorCensus::UnlimitedOcrV1 => &[],
    }
}

/// Refuse a conversion source whose shard count differs from the pinned
/// source's, before any shard is read.
#[cfg_attr(not(feature = "native"), allow(dead_code))] // sole non-test caller is the CLI convert path
pub(crate) fn validate_conversion_shard_count(
    census: TensorCensus,
    count: usize,
) -> FocrResult<()> {
    let pinned = source_shards(census);
    if count == pinned.len() {
        return Ok(());
    }
    Err(FocrError::FormatMismatch(format!(
        "{} conversion requires its pinned {}-shard source checkpoint; this source has {count} \
         shard(s)",
        census_model_id(census),
        pinned.len()
    )))
}

/// Refuse conversion inputs whose output the production artifact loader would
/// later reject: shard `position`'s SHA-256 must be the pinned one. The
/// conservative format is bound to the exact pinned source shards, not merely
/// a same-shaped safetensors serialization.
#[cfg_attr(not(feature = "native"), allow(dead_code))] // sole non-test caller is the CLI convert path
pub(crate) fn validate_conversion_source_shard(
    census: TensorCensus,
    position: usize,
    actual: &[u8; 32],
) -> FocrResult<()> {
    let actual = actual
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let pinned = source_shards(census);
    let Some(&(name, expected)) = pinned.get(position) else {
        return validate_conversion_shard_count(census, position + 1);
    };
    if actual == expected {
        return Ok(());
    }
    Err(FocrError::FormatMismatch(format!(
        "{} conversion requires pinned source shard {name} SHA-256 {expected}; got {actual}. \
         Supply the canonical checkpoint or use raw compatible weights directly",
        census_model_id(census)
    )))
}

/// The model id a census describes.
fn census_model_id(census: TensorCensus) -> &'static str {
    match census {
        TensorCensus::UnlimitedOcrV1 => model_arch::UNLIMITED_OCR_V1_ID,
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
            model_arch::UNLIMITED_OCR_V1_ID,
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
    /// The explicitly-tagged NON-DEFAULT wasm experts-int4 recipe (bd-4l71):
    /// same 2,710-tensor name/shape census, but expert FFN tensors stored
    /// QInt4PerGroup and attention/lm_head/embed stored QInt8PerChan.
    WasmInt4Focrq,
}

impl CensusSurface {
    fn label(self) -> &'static str {
        match self {
            Self::SourceSafetensors => "source safetensors",
            Self::ConservativeFocrq => "conservative .focrq",
            Self::WasmInt4Focrq => "wasm experts-int4 .focrq",
        }
    }

    fn expected_dtype(self, name: &str, tensor: &ManifestTensor) -> DType {
        match self {
            Self::SourceSafetensors => tensor.source_dtype,
            Self::ConservativeFocrq => tensor.storage_dtype,
            // The wasm quantized set is a superset of the conservative one, so
            // every KeepHighPrecision name stores exactly what the conservative
            // manifest stores (BF16) — the storage dtype cross-checks that.
            Self::WasmInt4Focrq => match classify_wasm_experts_int4(name) {
                WasmInt4Policy::ExpertInt4 => DType::QInt4PerGroup,
                WasmInt4Policy::Int8 => DType::QInt8PerChan,
                WasmInt4Policy::KeepHighPrecision => tensor.storage_dtype,
            },
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
        let expected_dtype = surface.expected_dtype(name, expected);
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

/// Identify a raw safetensors checkpoint, which declares no model id, by the
/// exact embedded census its directory matches. A shard matching none is
/// refused with every census's diff, never assumed to be the default model.
pub(super) fn identify_raw<'a, I>(records: I) -> FocrResult<&'static dyn ModelArch>
where
    I: IntoIterator<Item = (&'a str, &'a TensorRecord)> + Clone,
{
    let mut failures = Vec::new();
    for &arch in model_arch::registry() {
        let Some(census) = arch.tensor_census() else {
            continue;
        };
        let checked = match census {
            TensorCensus::UnlimitedOcrV1 => {
                validate_records(records.clone(), CensusSurface::SourceSafetensors)
            }
        };
        match checked {
            Ok(()) => return Ok(arch),
            Err(FocrError::FormatMismatch(diff)) => failures.push(format!("{}: {diff}", arch.id())),
            Err(other) => return Err(other),
        }
    }
    Err(FocrError::FormatMismatch(format!(
        "raw safetensors checkpoint matches no registered model's tensor census [{}]",
        failures.join("; ")
    )))
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
    let surface = if quant_recipe == Some(manifest.quant_recipe.as_str()) {
        CensusSurface::ConservativeFocrq
    } else if quant_recipe == Some(UNLIMITED_OCR_WASM_INT4_RECIPE_ID) {
        // The explicitly-tagged non-default wasm recipe (bd-4l71): same pinned
        // source + name/shape census, wasm storage dtypes. Never selected by
        // the default resolver — explicit path / from_weights only.
        CensusSurface::WasmInt4Focrq
    } else {
        return Err(FocrError::FormatMismatch(format!(
            "Unlimited-OCR .focrq packing_manifest.quant_recipe must be exactly {:?} (or the \
             explicitly non-default {UNLIMITED_OCR_WASM_INT4_RECIPE_ID:?}), got {quant_recipe:?}",
            manifest.quant_recipe
        )));
    };
    validate_records(
        records.iter().map(|(name, record)| (name.as_str(), record)),
        surface,
    )
}

/// Re-run the complete schema check against the directory produced by the real
/// mmap/owned loader, immediately before the production forward receives it.
/// The bounded header path separately verifies the recipe string on the same
/// open descriptor; this post-load check proves name/shape/dtype parity survived
/// the parser handoff.
pub(super) fn validate_loaded_weights(weights: &Weights) -> FocrResult<()> {
    match model_arch::arch_by_id(weights.model_id()).and_then(|arch| arch.tensor_census()) {
        Some(TensorCensus::UnlimitedOcrV1) => {}
        None => return Ok(()),
    }
    let surface = if weights.is_focrq() {
        if weights.source_sha256() != SOURCE_SHA256 {
            return Err(FocrError::FormatMismatch(format!(
                "Unlimited-OCR .focrq source_sha256 is {:?}, expected pinned source {SOURCE_SHA256:?}",
                weights.source_sha256()
            )));
        }
        if weights.quant_recipe() == Some(UNLIMITED_OCR_WASM_INT4_RECIPE_ID) {
            CensusSurface::WasmInt4Focrq
        } else {
            CensusSurface::ConservativeFocrq
        }
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

/// `(name, source dtype, shape)` of every tensor in the pinned v1 source
/// checkpoint, for tests that need a census-exact header without the weights.
#[cfg(test)]
pub(super) fn v1_source_tensors() -> Vec<(String, DType, Vec<usize>)> {
    manifest()
        .expect("embedded production manifest parses")
        .tensors
        .iter()
        .map(|(name, tensor)| (name.clone(), tensor.source_dtype, tensor.shape.clone()))
        .collect()
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
                        dtype: surface.expected_dtype(name, tensor),
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

    fn identify(directory: &BTreeMap<String, TensorRecord>) -> FocrResult<&'static dyn ModelArch> {
        identify_raw(
            directory
                .iter()
                .map(|(name, record)| (name.as_str(), record)),
        )
    }

    #[test]
    fn raw_checkpoints_are_identified_by_census_not_by_the_default() {
        let source = synthetic_directory(CensusSurface::SourceSafetensors);
        assert_eq!(
            identify(&source).expect("the v1 census matches").id(),
            model_arch::UNLIMITED_OCR_V1_ID
        );
        let got = model_arch::arch_by_id("got-ocr2").expect("got-ocr2 registered");
        let flip = model_arch::test_default::flip(got);
        assert_eq!(
            identify(&source).expect("still v1").id(),
            model_arch::UNLIMITED_OCR_V1_ID,
            "the default model plays no part"
        );
        drop(flip);

        // A shard no census describes (v1 less one tensor, plus a foreign one)
        // is refused with every census's diff, never assumed to be a model.
        let mut foreign = source.clone();
        foreign.remove("lm_head.weight");
        let mut router = source["model.norm.weight"].clone();
        router.shape = vec![32, 1280];
        foreign.insert("model.layers.0.self_attn.router.weight".into(), router);
        let message = identify(&foreign)
            .expect_err("matches no census")
            .to_string();
        assert!(
            message.contains("matches no registered model's tensor census"),
            "{message}"
        );
        for arch in model_arch::registry() {
            if arch.tensor_census().is_some() {
                assert!(message.contains(&format!("{}: ", arch.id())), "{message}");
            }
        }
        assert!(message.contains("missing 1: [lm_head.weight]"), "{message}");
        assert!(
            message.contains("unexpected 1: [model.layers.0.self_attn.router.weight]"),
            "{message}"
        );
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

    /// bd-4l71: the explicitly-tagged wasm experts-int4 recipe is accepted with
    /// its OWN storage dtypes (experts QInt4PerGroup, attn/lm_head/embed
    /// QInt8PerChan, rest BF16) — and each surface rejects the other's dtypes.
    #[test]
    fn wasm_int4_focrq_surface_accepts_wasm_dtypes_and_rejects_cross_recipe() {
        let wasm = synthetic_directory(CensusSurface::WasmInt4Focrq);
        // Sanity: the wasm directory really stores the three storage classes.
        assert_eq!(
            wasm["model.layers.11.mlp.experts.63.down_proj.weight"].dtype,
            DType::QInt4PerGroup
        );
        assert_eq!(wasm["lm_head.weight"].dtype, DType::QInt8PerChan);
        assert_eq!(wasm["model.embed_tokens.weight"].dtype, DType::QInt8PerChan);
        assert_eq!(
            wasm["model.layers.0.self_attn.q_proj.weight"].dtype,
            DType::QInt8PerChan
        );
        assert_eq!(wasm["model.norm.weight"].dtype, DType::BF16);
        assert_eq!(
            wasm["model.layers.5.mlp.gate.weight"].dtype,
            DType::BF16,
            "router gate must stay high precision even under the wasm recipe"
        );

        validate_focrq_header(
            &wasm,
            SOURCE_SHA256,
            Some(crate::quant::convert::UNLIMITED_OCR_WASM_INT4_RECIPE_ID),
        )
        .expect("wasm directory + wasm recipe id must pass");

        // Wasm dtypes under the CONSERVATIVE recipe id must fail (the default
        // recipe is untouched)...
        let error = validate_focrq_header(&wasm, SOURCE_SHA256, Some(UNLIMITED_OCR_INT8_RECIPE_ID))
            .expect_err("wasm dtypes must fail the conservative surface");
        assert!(error.to_string().contains("wrong dtype"));

        // ...and conservative dtypes under the wasm recipe id must fail too.
        let conservative = synthetic_directory(CensusSurface::ConservativeFocrq);
        let error = validate_focrq_header(
            &conservative,
            SOURCE_SHA256,
            Some(crate::quant::convert::UNLIMITED_OCR_WASM_INT4_RECIPE_ID),
        )
        .expect_err("conservative dtypes must fail the wasm surface");
        assert!(error.to_string().contains("wrong dtype"));
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
        let census = TensorCensus::UnlimitedOcrV1;
        validate_conversion_shard_count(census, 1).expect("the pinned source is one shard");
        validate_conversion_source_shard(census, 0, &expected).expect("pinned source accepted");
        let error = validate_conversion_source_shard(census, 0, &[0u8; 32])
            .expect_err("reserialized source must fail before conversion output");
        let message = error.to_string();
        assert!(message.contains(SOURCE_SHA256));
        assert!(message.contains(SOURCE_FILE));
        assert!(message.contains(&"0".repeat(64)));
        let error = validate_conversion_shard_count(census, 2)
            .expect_err("a resharded source is not the pinned checkpoint");
        assert!(error.to_string().contains("1-shard"), "{error}");
        let error = validate_conversion_source_shard(census, 1, &expected)
            .expect_err("a second shard has no pin");
        assert!(error.to_string().contains("2 shard(s)"), "{error}");
    }
}
