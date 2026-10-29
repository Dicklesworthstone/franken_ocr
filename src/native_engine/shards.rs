//! Sharded safetensors sources (U2-A3, D16): `model.safetensors.index.json`
//! plus N shard files, presented to the loader and the converter as one
//! logical checkpoint. Generic across the model zoo; a single-shard checkpoint
//! is the N = 1 case.
//!
//! - [`ShardIndex`] parses and validates the index. Its shard order — the
//!   distinct shard file names, ascending — is the "index order" every other
//!   piece uses.
//! - [`parse_safetensors_header`] is the one safetensors-header parser (plain
//!   files, every shard, and the bounded header probe). Float tensors become
//!   directory records; I64 tensors are set aside for verify-and-skip
//!   ([`Int64Rule`]) and are never exposed as weights.
//! - [`merge_shard_headers`] checks every shard's header against the index and
//!   rebases its records into one logical directory.
//! - [`stream_sha256`] is the streaming shard hash that `focr convert` and
//!   `doctor` share; [`source_sha256`] is the `.focrq` multi-shard convention.
//! - [`validate_source_shards`] is the `.focrq` loader gate on a multi-shard
//!   provenance.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::model_arch::ModelArch;
use super::weights::{DType, TensorRecord};
use crate::error::{FocrError, FocrResult};

/// File name of a sharded checkpoint's index.
pub const SHARD_INDEX_NAME: &str = "model.safetensors.index.json";

/// Upper bound on the size of an index file (the v2 index is 248 KiB).
const MAX_INDEX_BYTES: u64 = 8 * 1024 * 1024;

/// Read size of the streaming shard hash.
const HASH_CHUNK_BYTES: usize = 8 * 1024 * 1024;

/// A parsed, validated `model.safetensors.index.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardIndex {
    /// Distinct shard file names, ascending: the index order.
    shards: Vec<String>,
    /// Tensor name → position of its shard in `shards`.
    weight_map: BTreeMap<String, usize>,
    /// Tensors per shard, by position.
    shard_tensor_counts: Vec<usize>,
    /// `metadata.total_size`: the summed byte length of every tensor.
    total_size: u64,
}

impl ShardIndex {
    /// Parse an index file's bytes.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if the JSON is malformed, lacks
    /// `metadata.total_size`, maps no tensors, or names a shard that is not a
    /// plain `*.safetensors` file name beside the index.
    pub fn parse(bytes: &[u8]) -> FocrResult<Self> {
        #[derive(Deserialize)]
        struct RawIndex {
            metadata: RawMetadata,
            weight_map: BTreeMap<String, String>,
        }
        #[derive(Deserialize)]
        struct RawMetadata {
            total_size: u64,
        }

        let raw: RawIndex = serde_json::from_slice(bytes).map_err(|error| {
            FocrError::FormatMismatch(format!("{SHARD_INDEX_NAME} is invalid: {error}"))
        })?;
        if raw.weight_map.is_empty() {
            return Err(FocrError::FormatMismatch(format!(
                "{SHARD_INDEX_NAME} maps no tensors"
            )));
        }
        let mut shards: Vec<String> = raw.weight_map.values().cloned().collect();
        shards.sort_unstable();
        shards.dedup();
        for shard in &shards {
            validate_shard_file_name(shard)?;
        }
        let mut shard_tensor_counts = vec![0usize; shards.len()];
        let mut weight_map = BTreeMap::new();
        for (tensor, shard) in raw.weight_map {
            let position = shards
                .binary_search(&shard)
                .map_err(|_| FocrError::Other(anyhow::anyhow!("shard list lost {shard:?}")))?;
            shard_tensor_counts[position] += 1;
            weight_map.insert(tensor, position);
        }
        Ok(Self {
            shards,
            weight_map,
            shard_tensor_counts,
            total_size: raw.metadata.total_size,
        })
    }

    /// Shard file names in index order.
    #[must_use]
    pub fn shards(&self) -> &[String] {
        &self.shards
    }

    /// Number of tensors the index maps.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    /// `metadata.total_size`.
    #[must_use]
    pub fn total_size(&self) -> u64 {
        self.total_size
    }

    /// Check that every tensor the shard at `position` holds is one the index
    /// maps to it.
    fn check_shard_membership(
        &self,
        position: usize,
        header: &SafetensorsHeader,
    ) -> FocrResult<()> {
        let shard = &self.shards[position];
        for name in header.names() {
            match self.weight_map.get(name) {
                Some(&mapped) if mapped == position => {}
                Some(&mapped) => {
                    return Err(FocrError::FormatMismatch(format!(
                        "shard {shard} holds tensor {name:?}, which {SHARD_INDEX_NAME} maps to {}",
                        self.shards[mapped]
                    )));
                }
                None => {
                    return Err(FocrError::FormatMismatch(format!(
                        "shard {shard} holds tensor {name:?}, which {SHARD_INDEX_NAME} does not list"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Check that the shard at `position` holds every tensor the index maps to
    /// it (run after [`Self::check_shard_membership`], so a count shortfall is
    /// a missing tensor).
    fn check_shard_complete(&self, position: usize, header: &SafetensorsHeader) -> FocrResult<()> {
        let shard = &self.shards[position];
        if header.len() != self.shard_tensor_counts[position] {
            let missing = self
                .weight_map
                .iter()
                .find(|&(name, &mapped)| mapped == position && !header.contains(name))
                .map_or("", |(name, _)| name.as_str());
            return Err(FocrError::FormatMismatch(format!(
                "{SHARD_INDEX_NAME} maps tensor {missing:?} to {shard}, which does not hold it"
            )));
        }
        Ok(())
    }
}

/// A shard named by an index must be a plain `*.safetensors` file in the
/// index's own directory.
fn validate_shard_file_name(shard: &str) -> FocrResult<()> {
    let plain = !shard.starts_with('.')
        && !shard.contains(['/', '\\'])
        && Path::new(shard).file_name() == Some(OsStr::new(shard));
    if plain && shard.len() > ".safetensors".len() && shard.ends_with(".safetensors") {
        return Ok(());
    }
    Err(FocrError::FormatMismatch(format!(
        "{SHARD_INDEX_NAME} names shard {shard:?}, which is not a plain *.safetensors file name \
         beside the index"
    )))
}

/// Whether `path` names a sharded checkpoint's index.
#[must_use]
pub fn is_index_path(path: &Path) -> bool {
    path.file_name() == Some(OsStr::new(SHARD_INDEX_NAME))
}

/// Read and parse the index at `path`, bounded at [`MAX_INDEX_BYTES`].
///
/// # Errors
/// [`FocrError::ModelNotFound`] if the file cannot be read;
/// [`FocrError::FormatMismatch`] if it is too large or invalid.
pub fn read_index(path: &Path) -> FocrResult<ShardIndex> {
    let file = std::fs::File::open(path).map_err(|error| {
        FocrError::ModelNotFound(format!("cannot read {}: {error}", path.display()))
    })?;
    let mut bytes = Vec::new();
    file.take(MAX_INDEX_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| {
            FocrError::ModelNotFound(format!("cannot read {}: {error}", path.display()))
        })?;
    if bytes.len() as u64 > MAX_INDEX_BYTES {
        return Err(FocrError::FormatMismatch(format!(
            "{} exceeds the {MAX_INDEX_BYTES}-byte index limit",
            path.display()
        )));
    }
    ShardIndex::parse(&bytes)
        .map_err(|error| FocrError::FormatMismatch(format!("{}: {error}", path.display())))
}

/// The shard files of a safetensors source, in index order: the shards an
/// index names, or one plain safetensors file.
#[derive(Debug)]
pub struct SourceShards {
    index: Option<ShardIndex>,
    files: Vec<(String, PathBuf)>,
}

impl SourceShards {
    /// Plan the shards of `resolved`, the path the model resolver returned: a
    /// [`SHARD_INDEX_NAME`] file, or one safetensors file.
    ///
    /// # Errors
    /// As [`read_index`], or [`FocrError::ModelNotFound`] when an indexed
    /// shard is missing.
    pub fn open(resolved: &Path) -> FocrResult<Self> {
        if !is_index_path(resolved) {
            let name = resolved
                .file_name()
                .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
            return Ok(Self {
                index: None,
                files: vec![(name, resolved.to_path_buf())],
            });
        }
        let index = read_index(resolved)?;
        let dir = resolved.parent().unwrap_or_else(|| Path::new("."));
        let mut files = Vec::with_capacity(index.shards().len());
        for shard in index.shards() {
            let path = dir.join(shard);
            if !path.is_file() {
                return Err(FocrError::ModelNotFound(format!(
                    "{} lists shard {shard}, but {} is missing",
                    resolved.display(),
                    path.display()
                )));
            }
            files.push((shard.clone(), path));
        }
        Ok(Self {
            index: Some(index),
            files,
        })
    }

    /// The index, or `None` for one plain file.
    #[must_use]
    pub fn index(&self) -> Option<&ShardIndex> {
        self.index.as_ref()
    }

    /// `(file name, path)` per shard, in index order.
    #[must_use]
    pub fn files(&self) -> &[(String, PathBuf)] {
        &self.files
    }
}

/// An open shard descriptor. Header validation and the weight load read the
/// same descriptor, so a path swap between them cannot substitute bytes.
#[derive(Debug)]
pub(crate) struct OpenedShard {
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) file: std::fs::File,
    pub(crate) len: u64,
}

/// A sharded checkpoint with every shard open.
#[derive(Debug)]
pub(crate) struct OpenedShards {
    pub(crate) index: ShardIndex,
    pub(crate) shards: Vec<OpenedShard>,
}

/// Read the index at `index_path` and open each shard it names.
pub(crate) fn open_shards(index_path: &Path) -> FocrResult<OpenedShards> {
    let source = SourceShards::open(index_path)?;
    let index = source.index.ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "{} is not a {SHARD_INDEX_NAME}",
            index_path.display()
        ))
    })?;
    let mut shards = Vec::with_capacity(source.files.len());
    for (name, path) in source.files {
        let file = std::fs::File::open(&path).map_err(|error| {
            FocrError::ModelNotFound(format!("cannot open shard {}: {error}", path.display()))
        })?;
        let metadata = file.metadata().map_err(|error| {
            FocrError::ModelNotFound(format!("cannot stat shard {}: {error}", path.display()))
        })?;
        if !metadata.is_file() {
            return Err(FocrError::FormatMismatch(format!(
                "shard {} is not a regular file",
                path.display()
            )));
        }
        shards.push(OpenedShard {
            name,
            path,
            file,
            len: metadata.len(),
        });
    }
    Ok(OpenedShards { index, shards })
}

/// Name the shard a structural error came from. Other error kinds pass
/// through unchanged.
pub(crate) fn in_shard(shard: &str, error: FocrError) -> FocrError {
    match error {
        FocrError::FormatMismatch(message) => {
            FocrError::FormatMismatch(format!("shard {shard}: {message}"))
        }
        other => other,
    }
}

/// One I64 tensor of a safetensors header, set aside for verify-and-skip.
/// Offsets are payload-relative, like [`TensorRecord`]'s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Int64Record {
    pub(crate) shape: Vec<usize>,
    pub(crate) byte_offset: usize,
    pub(crate) byte_len: usize,
}

/// A parsed safetensors header: float tensors as directory records, I64
/// tensors set aside.
#[derive(Debug, Default)]
pub(crate) struct SafetensorsHeader {
    pub(crate) records: BTreeMap<String, TensorRecord>,
    pub(crate) int64: BTreeMap<String, Int64Record>,
}

impl SafetensorsHeader {
    /// Every tensor name, float and I64.
    fn names(&self) -> impl Iterator<Item = &str> {
        self.records
            .keys()
            .chain(self.int64.keys())
            .map(String::as_str)
    }

    /// Number of tensors, float and I64.
    pub(crate) fn len(&self) -> usize {
        self.records.len() + self.int64.len()
    }

    fn contains(&self, name: &str) -> bool {
        self.records.contains_key(name) || self.int64.contains_key(name)
    }

    /// Summed byte length of every tensor, as `metadata.total_size` counts it.
    fn tensor_bytes(&self) -> u64 {
        self.records
            .values()
            .map(|record| record.byte_len as u64)
            .chain(self.int64.values().map(|record| record.byte_len as u64))
            .sum()
    }
}

/// Parse a safetensors header JSON object: `name -> {dtype, shape,
/// data_offsets: [begin, end]}` with offsets relative to the payload, plus an
/// optional `__metadata__` key, which is skipped.
///
/// # Errors
/// [`FocrError::FormatMismatch`] on malformed JSON, a malformed entry, reversed
/// offsets, or a dtype other than F32/F16/BF16/I64.
pub(crate) fn parse_safetensors_header(header_json: &[u8]) -> FocrResult<SafetensorsHeader> {
    #[derive(Deserialize)]
    struct Entry {
        dtype: String,
        shape: Vec<usize>,
        data_offsets: [usize; 2],
    }

    let raw: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(header_json)
        .map_err(|error| {
            FocrError::FormatMismatch(format!("safetensors header JSON invalid: {error}"))
        })?;
    let mut header = SafetensorsHeader::default();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let entry: Entry = serde_json::from_value(value).map_err(|error| {
            FocrError::FormatMismatch(format!("safetensors entry {name:?} invalid: {error}"))
        })?;
        let [begin, end] = entry.data_offsets;
        if end < begin {
            return Err(FocrError::FormatMismatch(format!(
                "safetensors entry {name:?} has end {end} < begin {begin}"
            )));
        }
        let dtype = match entry.dtype.as_str() {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "I64" => {
                header.int64.insert(
                    name,
                    Int64Record {
                        shape: entry.shape,
                        byte_offset: begin,
                        byte_len: end - begin,
                    },
                );
                continue;
            }
            other => {
                return Err(FocrError::FormatMismatch(format!(
                    "unsupported safetensors dtype {other:?} for {name:?} (expected \
                     F32/F16/BF16, or I64 for a census verify-and-skip tensor)"
                )));
            }
        };
        header.records.insert(
            name,
            TensorRecord {
                dtype,
                shape: entry.shape,
                byte_offset: begin,
                byte_len: end - begin,
                scales_offset: 0,
                scales_len: 0,
                group_size: 0,
                tier: 0,
            },
        );
    }
    Ok(header)
}

/// Check each I64 record against a payload of `payload_len` bytes: in bounds,
/// exactly `numel × 8` bytes, and disjoint from every other tensor. The float
/// records are validated by the directory validators.
pub(crate) fn validate_int64_records(
    header: &SafetensorsHeader,
    payload_len: usize,
) -> FocrResult<()> {
    if header.int64.is_empty() {
        return Ok(());
    }
    for (name, record) in &header.int64 {
        let expected = record
            .shape
            .iter()
            .try_fold(8usize, |bytes, &dim| bytes.checked_mul(dim))
            .ok_or_else(|| {
                FocrError::FormatMismatch(format!(
                    "I64 tensor {name:?}: shape {:?} overflows usize",
                    record.shape
                ))
            })?;
        if record.byte_len != expected {
            return Err(FocrError::FormatMismatch(format!(
                "I64 tensor {name:?}: byte_len {} != shape×8 {expected} (shape {:?})",
                record.byte_len, record.shape
            )));
        }
        let end = record
            .byte_offset
            .checked_add(record.byte_len)
            .ok_or_else(|| {
                FocrError::FormatMismatch(format!("I64 tensor {name:?} byte range overflows"))
            })?;
        if end > payload_len {
            return Err(FocrError::FormatMismatch(format!(
                "I64 tensor {name:?} ends at {end} but payload is {payload_len} bytes"
            )));
        }
    }
    let mut ranges: Vec<(usize, usize, &str)> = header
        .records
        .iter()
        .map(|(name, record)| (record.byte_offset, record.byte_len, name.as_str()))
        .chain(
            header
                .int64
                .iter()
                .map(|(name, record)| (record.byte_offset, record.byte_len, name.as_str())),
        )
        .filter(|&(_, len, _)| len != 0)
        .map(|(start, len, name)| (start, start.saturating_add(len), name))
        .collect();
    ranges.sort_unstable();
    for pair in ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(FocrError::FormatMismatch(format!(
                "safetensors tensors {:?} and {:?} overlap",
                pair[0].2, pair[1].2
            )));
        }
    }
    Ok(())
}

/// Merge the headers of every shard, in index order, into one logical
/// directory. `bases[i]` is the logical offset at which shard `i`'s payload
/// starts; each record's offset is rebased onto it.
///
/// Checks, in order: no tensor in two shards; every tensor a shard holds is
/// one the index maps to it; every tensor the index maps to a shard is there;
/// the tensors' summed size equals `metadata.total_size`.
pub(crate) fn merge_shard_headers(
    index: &ShardIndex,
    shards: Vec<(SafetensorsHeader, u64)>,
) -> FocrResult<SafetensorsHeader> {
    if shards.len() != index.shards.len() {
        return Err(FocrError::Other(anyhow::anyhow!(
            "merging {} shard headers for an index of {} shards",
            shards.len(),
            index.shards.len()
        )));
    }
    let mut origin: BTreeMap<&str, usize> = BTreeMap::new();
    for (position, (header, _)) in shards.iter().enumerate() {
        for name in header.names() {
            if let Some(first) = origin.insert(name, position) {
                return Err(FocrError::FormatMismatch(format!(
                    "tensor {name:?} appears in both {} and {}",
                    index.shards[first], index.shards[position]
                )));
            }
        }
    }
    for (position, (header, _)) in shards.iter().enumerate() {
        index.check_shard_membership(position, header)?;
    }
    for (position, (header, _)) in shards.iter().enumerate() {
        index.check_shard_complete(position, header)?;
    }
    let tensor_bytes: u64 = shards.iter().map(|(header, _)| header.tensor_bytes()).sum();
    if tensor_bytes != index.total_size {
        return Err(FocrError::FormatMismatch(format!(
            "{SHARD_INDEX_NAME} metadata.total_size is {} but its shards hold {tensor_bytes} \
             tensor bytes",
            index.total_size
        )));
    }

    let mut merged = SafetensorsHeader::default();
    for (header, base) in shards {
        let rebase = |name: &str, offset: usize| {
            u64::try_from(offset)
                .ok()
                .and_then(|offset| offset.checked_add(base))
                .and_then(|offset| usize::try_from(offset).ok())
                .ok_or_else(|| {
                    FocrError::FormatMismatch(format!(
                        "tensor {name:?} offset exceeds this platform's addressable size"
                    ))
                })
        };
        for (name, mut record) in header.records {
            record.byte_offset = rebase(&name, record.byte_offset)?;
            merged.records.insert(name, record);
        }
        for (name, mut record) in header.int64 {
            record.byte_offset = rebase(&name, record.byte_offset)?;
            merged.int64.insert(name, record);
        }
    }
    Ok(merged)
}

/// What a verify-and-skip I64 tensor must contain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int64Content {
    /// Element `i` of the flattened tensor equals `i` (`torch.arange`).
    Arange,
}

/// A tensor that a census accepts as I64 only to verify its content and drop
/// it (D8). Any other I64 tensor is an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Int64Rule {
    /// Tensor name.
    pub name: &'static str,
    /// Exact shape.
    pub shape: &'static [usize],
    /// Required content.
    pub content: Int64Content,
}

/// The verify-and-skip rules of `arch`'s tensor census: none for an arch
/// without one, or for an unidentified checkpoint.
pub(crate) fn int64_rules(arch: Option<&dyn ModelArch>) -> &'static [Int64Rule] {
    #[cfg(test)]
    if let Some(rules) = test_rules::current() {
        return rules;
    }
    arch.and_then(|arch| arch.tensor_census())
        .map_or(&[], super::unlimited_ocr_census::int64_rules)
}

/// Whose rules judged an I64 tensor, for error messages.
pub(crate) fn rules_owner(arch: Option<&dyn ModelArch>) -> &'static str {
    arch.map_or("an unidentified checkpoint", |arch| arch.id())
}

/// Check every I64 tensor against `rules` by name and shape. Header-only.
pub(crate) fn check_int64_entries(
    int64: &BTreeMap<String, Int64Record>,
    rules: &[Int64Rule],
    owner: &str,
) -> FocrResult<()> {
    for (name, record) in int64 {
        let Some(rule) = rules.iter().find(|rule| rule.name == name) else {
            return Err(FocrError::FormatMismatch(format!(
                "safetensors tensor {name:?} is I64 {:?}; {owner} accepts I64 only for its \
                 census's verify-and-skip tensors ({})",
                record.shape,
                if rules.is_empty() {
                    "it has none".to_owned()
                } else {
                    rules
                        .iter()
                        .map(|rule| rule.name)
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            )));
        };
        if record.shape != rule.shape {
            return Err(FocrError::FormatMismatch(format!(
                "I64 tensor {name:?} has shape {:?}, but {owner}'s census requires {:?}",
                record.shape, rule.shape
            )));
        }
    }
    Ok(())
}

/// Check that every rule's tensor is present. Only a whole checkpoint can
/// answer this; one shard of it cannot.
pub(crate) fn check_int64_rules_present(
    int64: &BTreeMap<String, Int64Record>,
    rules: &[Int64Rule],
    owner: &str,
) -> FocrResult<()> {
    for rule in rules {
        if !int64.contains_key(rule.name) {
            return Err(FocrError::FormatMismatch(format!(
                "{owner}'s census requires the I64 tensor {:?} {:?} (verified, then skipped), \
                 which this checkpoint lacks",
                rule.name, rule.shape
            )));
        }
    }
    Ok(())
}

/// Verify the payload of every I64 tensor against its rule. `bytes_of`
/// resolves a record to its payload bytes. Run [`check_int64_entries`] first.
pub(crate) fn verify_int64_payloads<'a>(
    int64: &BTreeMap<String, Int64Record>,
    rules: &[Int64Rule],
    mut bytes_of: impl FnMut(&str, &Int64Record) -> FocrResult<&'a [u8]>,
) -> FocrResult<()> {
    for (name, record) in int64 {
        let rule = rules.iter().find(|rule| rule.name == name).ok_or_else(|| {
            FocrError::FormatMismatch(format!("I64 tensor {name:?} has no verify-and-skip rule"))
        })?;
        verify_int64_content(name, rule, bytes_of(name, record)?)?;
    }
    Ok(())
}

/// Verify one I64 tensor's little-endian bytes against `rule`.
pub(crate) fn verify_int64_content(name: &str, rule: &Int64Rule, bytes: &[u8]) -> FocrResult<()> {
    let numel = rule.shape.iter().product::<usize>();
    if bytes.len() != numel.saturating_mul(8) {
        return Err(FocrError::FormatMismatch(format!(
            "I64 tensor {name:?} has {} bytes; shape {:?} needs {}",
            bytes.len(),
            rule.shape,
            numel.saturating_mul(8)
        )));
    }
    match rule.content {
        Int64Content::Arange => {
            for (position, word) in bytes.as_chunks::<8>().0.iter().enumerate() {
                let value = i64::from_le_bytes(*word);
                if i64::try_from(position).ok() != Some(value) {
                    return Err(FocrError::FormatMismatch(format!(
                        "I64 tensor {name:?} element {position} is {value}; the census requires \
                         arange (element i == i)"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// One source shard's identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardDigest {
    /// Shard file name.
    pub name: String,
    /// Byte size of the whole file.
    pub size: u64,
    /// SHA-256 of the whole file.
    pub sha256: [u8; 32],
}

impl ShardDigest {
    /// Lowercase-hex SHA-256.
    #[must_use]
    pub fn sha256_hex(&self) -> String {
        hex(&self.sha256)
    }
}

/// Stream `reader` to its end through SHA-256 in fixed-size chunks, handing
/// each chunk to `sink` as it is hashed. Returns the byte count and the
/// digest. `origin` names the source in errors.
///
/// This is the one streaming shard hash: `focr convert` keeps the bytes it
/// converts, and `doctor`'s full-hash check drops them.
///
/// # Errors
/// [`FocrError::ModelNotFound`] on a read error, or whatever `sink` returns.
pub fn stream_sha256(
    mut reader: impl Read,
    origin: &Path,
    mut sink: impl FnMut(&[u8]) -> FocrResult<()>,
) -> FocrResult<(u64, [u8; 32])> {
    use sha2::Digest as _;

    let mut hasher = sha2::Sha256::new();
    let mut chunk = vec![0u8; HASH_CHUNK_BYTES];
    let mut total = 0u64;
    loop {
        let read = match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => {
                return Err(FocrError::ModelNotFound(format!(
                    "cannot read {}: {error}",
                    origin.display()
                )));
            }
        };
        hasher.update(&chunk[..read]);
        sink(&chunk[..read])?;
        total += read as u64;
    }
    Ok((total, hasher.finalize().into()))
}

/// Read one whole shard into memory, hashing it in the same pass.
///
/// # Errors
/// [`FocrError::ModelNotFound`] if the file cannot be read;
/// [`FocrError::FormatMismatch`] if it changes size while it is read.
pub fn read_shard(name: &str, path: &Path) -> FocrResult<(Vec<u8>, ShardDigest)> {
    let file = std::fs::File::open(path).map_err(|error| {
        FocrError::ModelNotFound(format!("cannot read {}: {error}", path.display()))
    })?;
    let len = file
        .metadata()
        .map_err(|error| {
            FocrError::ModelNotFound(format!("cannot stat {}: {error}", path.display()))
        })?
        .len();
    let capacity = usize::try_from(len).map_err(|_| {
        FocrError::FormatMismatch(format!(
            "{} ({len} bytes) exceeds this platform's addressable size",
            path.display()
        ))
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    let (size, sha256) = stream_sha256(file, path, |chunk| {
        if bytes.len() + chunk.len() > capacity {
            return Err(FocrError::FormatMismatch(format!(
                "{} grew while it was read",
                path.display()
            )));
        }
        bytes.extend_from_slice(chunk);
        Ok(())
    })?;
    if size != len {
        return Err(FocrError::FormatMismatch(format!(
            "{} changed size while it was read ({len} -> {size} bytes)",
            path.display()
        )));
    }
    Ok((
        bytes,
        ShardDigest {
            name: name.to_owned(),
            size,
            sha256,
        },
    ))
}

/// Hash every shard of the source at `resolved` (an index or one plain file)
/// without keeping the bytes: the full-hash verification `doctor` runs on a
/// raw shard directory.
///
/// # Errors
/// As [`SourceShards::open`] and [`stream_sha256`].
pub fn hash_shards(resolved: &Path) -> FocrResult<Vec<ShardDigest>> {
    let source = SourceShards::open(resolved)?;
    let mut digests = Vec::with_capacity(source.files.len());
    for (name, path) in &source.files {
        let file = std::fs::File::open(path).map_err(|error| {
            FocrError::ModelNotFound(format!("cannot read {}: {error}", path.display()))
        })?;
        let (size, sha256) = stream_sha256(file, path, |_| Ok(()))?;
        digests.push(ShardDigest {
            name: name.clone(),
            size,
            sha256,
        });
    }
    Ok(digests)
}

/// The `.focrq` `source_sha256` of a source whose shards hash to `digests`,
/// in index order: a single shard's own SHA-256, or, for N ≥ 2 shards,
/// SHA-256 over the concatenation of the shard SHA-256s (D16).
#[must_use]
pub fn source_sha256(digests: &[ShardDigest]) -> [u8; 32] {
    use sha2::Digest as _;

    debug_assert!(!digests.is_empty(), "a source has at least one shard");
    if let [single] = digests {
        return single.sha256;
    }
    let mut hasher = sha2::Sha256::new();
    for digest in digests {
        hasher.update(digest.sha256);
    }
    hasher.finalize().into()
}

/// The `provenance` JSON object a `.focrq` built from `digests` carries:
/// `{"source_shards":[{"name","size","sha256"}, …]}` for N ≥ 2 shards, and
/// nothing for one shard, whose artifact bytes stay as they always were.
#[must_use]
pub fn provenance_json(digests: &[ShardDigest]) -> Option<String> {
    if digests.len() < 2 {
        return None;
    }
    let shards: Vec<serde_json::Value> = digests
        .iter()
        .map(|digest| {
            serde_json::json!({
                "name": digest.name,
                "size": digest.size,
                "sha256": digest.sha256_hex(),
            })
        })
        .collect();
    Some(serde_json::json!({ "source_shards": shards }).to_string())
}

/// The `provenance` object of a `.focrq` header, as far as the loader reads
/// it. Other keys are forward-compatible metadata and are ignored.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct FocrqProvenance {
    #[serde(default)]
    source_shards: Option<Vec<ProvenanceShard>>,
}

/// One shard of a multi-shard `.focrq` provenance.
#[derive(Debug, Clone, Deserialize)]
struct ProvenanceShard {
    name: String,
    size: u64,
    sha256: String,
}

/// The loader gate on a `.focrq` provenance: a `source_shards` list has at
/// least two well-formed entries, and `source_sha256_hex` is the convention
/// over them ([`source_sha256`]). A mismatch is
/// [`FocrError::FormatMismatch`] (exit 7).
pub(crate) fn validate_source_shards(
    provenance: Option<&FocrqProvenance>,
    source_sha256_hex: &str,
) -> FocrResult<()> {
    let Some(shards) = provenance.and_then(|provenance| provenance.source_shards.as_ref()) else {
        return Ok(());
    };
    if shards.len() < 2 {
        return Err(FocrError::FormatMismatch(format!(
            ".focrq provenance.source_shards lists {} shard(s); a single-shard source records \
             its own SHA-256 and no shard list",
            shards.len()
        )));
    }
    let mut digests = Vec::with_capacity(shards.len());
    for shard in shards {
        validate_shard_file_name(&shard.name).map_err(|_| {
            FocrError::FormatMismatch(format!(
                ".focrq provenance shard name {:?} is not a plain *.safetensors file name",
                shard.name
            ))
        })?;
        let sha256 = parse_sha256_hex(&shard.sha256).ok_or_else(|| {
            FocrError::FormatMismatch(format!(
                ".focrq provenance shard {} has sha256 {:?}, not 64 lowercase hex digits",
                shard.name, shard.sha256
            ))
        })?;
        if shard.size == 0 {
            return Err(FocrError::FormatMismatch(format!(
                ".focrq provenance shard {} has size 0",
                shard.name
            )));
        }
        digests.push(ShardDigest {
            name: shard.name.clone(),
            size: shard.size,
            sha256,
        });
    }
    if !digests.windows(2).all(|pair| pair[0].name < pair[1].name) {
        return Err(FocrError::FormatMismatch(
            ".focrq provenance.source_shards is not in index order (ascending, distinct names)"
                .into(),
        ));
    }
    let expected = hex(&source_sha256(&digests));
    if source_sha256_hex != expected {
        return Err(FocrError::FormatMismatch(format!(
            ".focrq source_sha256 is {source_sha256_hex:?}, but its provenance lists {} source \
             shards whose convention digest is {expected:?}",
            digests.len()
        )));
    }
    Ok(())
}

fn parse_sha256_hex(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64
        || !text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return None;
    }
    let mut out = [0u8; 32];
    for (position, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[position * 2..position * 2 + 2], 16).ok()?;
    }
    Some(out)
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Test seam: verify-and-skip rules for checkpoints no registered census
/// describes yet (U2-A5 supplies the real ones from the census).
#[cfg(test)]
pub(crate) mod test_rules {
    use super::Int64Rule;
    use std::cell::Cell;

    thread_local! {
        static RULES: Cell<Option<&'static [Int64Rule]>> = const { Cell::new(None) };
    }

    /// Restores the previous rules on drop.
    pub(crate) struct Guard(Option<&'static [Int64Rule]>);

    impl Drop for Guard {
        fn drop(&mut self) {
            RULES.with(|cell| cell.set(self.0));
        }
    }

    /// Make [`super::int64_rules`] return `rules` on this thread.
    pub(crate) fn supply(rules: &'static [Int64Rule]) -> Guard {
        Guard(RULES.with(|cell| cell.replace(Some(rules))))
    }

    pub(super) fn current() -> Option<&'static [Int64Rule]> {
        RULES.with(Cell::get)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// One synthetic tensor: name, safetensors dtype, shape, raw bytes.
    pub(crate) type SyntheticTensor = (&'static str, &'static str, Vec<usize>, Vec<u8>);

    /// A safetensors file holding `tensors` contiguously, in the given order,
    /// with a `__metadata__` key.
    pub(crate) fn safetensors_file(tensors: &[SyntheticTensor]) -> Vec<u8> {
        let mut entries = vec![r#""__metadata__":{"format":"pt"}"#.to_owned()];
        let mut payload = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let begin = payload.len();
            payload.extend_from_slice(bytes);
            entries.push(format!(
                r#""{name}":{{"dtype":"{dtype}","shape":{shape:?},"data_offsets":[{begin},{}]}}"#,
                payload.len()
            ));
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(&payload);
        file
    }

    /// An index mapping each `(tensor, shard)` pair, with `total_size`.
    pub(crate) fn index_json(map: &[(&str, &str)], total_size: u64) -> Vec<u8> {
        let weight_map: serde_json::Map<String, serde_json::Value> = map
            .iter()
            .map(|(tensor, shard)| ((*tensor).to_owned(), serde_json::json!(shard)))
            .collect();
        serde_json::to_vec(&serde_json::json!({
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }))
        .expect("index serializes")
    }

    /// `arange(n)` as little-endian I64 bytes.
    pub(crate) fn arange_bytes(n: usize) -> Vec<u8> {
        (0..n as i64).flat_map(i64::to_le_bytes).collect()
    }

    /// The rule the synthetic checkpoints' I64 tensor satisfies.
    pub(crate) static POSITION_IDS_RULE: [Int64Rule; 1] = [Int64Rule {
        name: "vision.position_ids",
        shape: &[1, 4],
        content: Int64Content::Arange,
    }];

    fn header_of(file: &[u8]) -> SafetensorsHeader {
        let len = u64::from_le_bytes(file[..8].try_into().expect("8-byte prefix")) as usize;
        parse_safetensors_header(&file[8..8 + len]).expect("synthetic header parses")
    }

    fn message(result: FocrResult<impl std::fmt::Debug>) -> String {
        result.expect_err("must be refused").to_string()
    }

    fn temp_dir(label: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("franken_ocr_shards_{label}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn index_parses_its_shards_in_index_order() {
        let index = ShardIndex::parse(&index_json(
            &[
                ("b", "model-00002-of-00002.safetensors"),
                ("a", "model-00001-of-00002.safetensors"),
                ("c", "model-00002-of-00002.safetensors"),
            ],
            24,
        ))
        .expect("index parses");
        assert_eq!(
            index.shards(),
            [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors"
            ]
        );
        assert_eq!(index.tensor_count(), 3);
        assert_eq!(index.total_size(), 24);
        assert!(is_index_path(Path::new("/x/model.safetensors.index.json")));
        assert!(!is_index_path(Path::new("/x/model.safetensors")));
    }

    #[test]
    fn index_refuses_malformed_inputs_with_clear_errors() {
        let no_total = br#"{"metadata":{},"weight_map":{"a":"m.safetensors"}}"#;
        assert!(message(ShardIndex::parse(no_total)).contains("total_size"));
        let empty = br#"{"metadata":{"total_size":0},"weight_map":{}}"#;
        assert!(message(ShardIndex::parse(empty)).contains("maps no tensors"));
        for bad in [
            "../model.safetensors",
            "sub/model.safetensors",
            "sub\\model.safetensors",
            "model.bin",
            ".safetensors",
            ".hidden.safetensors",
            "tokenizer.json",
        ] {
            let error = message(ShardIndex::parse(&index_json(&[("a", bad)], 2)));
            assert!(
                error.contains("not a plain *.safetensors"),
                "{bad}: {error}"
            );
        }
    }

    #[test]
    fn merge_checks_every_shard_against_the_index() {
        let one = safetensors_file(&[
            ("a", "BF16", vec![2], vec![1, 0, 2, 0]),
            ("b", "F32", vec![1], vec![0; 4]),
        ]);
        let two = safetensors_file(&[("c", "BF16", vec![1], vec![3, 0])]);
        let map = [
            ("a", "s1.safetensors"),
            ("b", "s1.safetensors"),
            ("c", "s2.safetensors"),
        ];
        let index = ShardIndex::parse(&index_json(&map, 10)).expect("index");
        let merged = merge_shard_headers(
            &index,
            vec![(header_of(&one), 100), (header_of(&two), 1_000)],
        )
        .expect("consistent shards merge");
        assert_eq!(merged.records["a"].byte_offset, 100);
        assert_eq!(merged.records["b"].byte_offset, 104);
        assert_eq!(merged.records["c"].byte_offset, 1_000);

        // The same tensor in two shards.
        let dup = safetensors_file(&[
            ("c", "BF16", vec![1], vec![3, 0]),
            ("a", "BF16", vec![2], vec![0; 4]),
        ]);
        let error = message(merge_shard_headers(
            &index,
            vec![(header_of(&one), 0), (header_of(&dup), 0)],
        ));
        assert!(
            error.contains("\"a\" appears in both s1.safetensors and s2.safetensors"),
            "{error}"
        );

        // A tensor the index maps to a shard that lacks it.
        let lacking = safetensors_file(&[("a", "BF16", vec![2], vec![0; 4])]);
        let error = message(merge_shard_headers(
            &index,
            vec![(header_of(&lacking), 0), (header_of(&two), 0)],
        ));
        assert!(
            error.contains("maps tensor \"b\" to s1.safetensors, which does not hold it"),
            "{error}"
        );

        // A tensor the index does not list.
        let extra = safetensors_file(&[
            ("c", "BF16", vec![1], vec![3, 0]),
            ("z", "BF16", vec![1], vec![0; 2]),
        ]);
        let error = message(merge_shard_headers(
            &index,
            vec![(header_of(&one), 0), (header_of(&extra), 0)],
        ));
        assert!(
            error.contains("holds tensor \"z\", which model.safetensors.index.json does not list"),
            "{error}"
        );

        // A tensor in the shard the index does not map it to.
        let moved_one = safetensors_file(&[("a", "BF16", vec![2], vec![0; 4])]);
        let moved_two = safetensors_file(&[
            ("b", "F32", vec![1], vec![0; 4]),
            ("c", "BF16", vec![1], vec![3, 0]),
        ]);
        let error = message(merge_shard_headers(
            &index,
            vec![(header_of(&moved_one), 0), (header_of(&moved_two), 0)],
        ));
        assert!(error.contains("s2.safetensors holds tensor \"b\", which model.safetensors.index.json maps to s1.safetensors"), "{error}");

        // metadata.total_size disagreeing with the shards.
        let index = ShardIndex::parse(&index_json(&map, 11)).expect("index");
        let error = message(merge_shard_headers(
            &index,
            vec![(header_of(&one), 0), (header_of(&two), 0)],
        ));
        assert!(
            error.contains("total_size is 11 but its shards hold 10 tensor bytes"),
            "{error}"
        );
    }

    #[test]
    fn header_parser_sets_i64_aside_and_refuses_other_dtypes() {
        let file = safetensors_file(&[
            ("w", "BF16", vec![2], vec![0; 4]),
            ("vision.position_ids", "I64", vec![1, 4], arange_bytes(4)),
        ]);
        let header = header_of(&file);
        assert_eq!(header.records.keys().collect::<Vec<_>>(), ["w"]);
        assert_eq!(header.int64["vision.position_ids"].byte_len, 32);
        validate_int64_records(&header, 36).expect("in bounds and disjoint");
        assert!(message(validate_int64_records(&header, 35)).contains("ends at 36"));

        let wrong_len = safetensors_file(&[("p", "I64", vec![1, 4], arange_bytes(3))]);
        assert!(message(validate_int64_records(&header_of(&wrong_len), 24)).contains("shape×8 32"));

        let mut overlapping = header_of(&file);
        overlapping
            .int64
            .get_mut("vision.position_ids")
            .expect("i64")
            .byte_offset = 2;
        assert!(message(validate_int64_records(&overlapping, 64)).contains("overlap"));

        for dtype in ["I32", "U8", "F64", "BOOL"] {
            let file = safetensors_file(&[("x", dtype, vec![1], vec![0; 8])]);
            let len = u64::from_le_bytes(file[..8].try_into().expect("prefix")) as usize;
            let error = message(parse_safetensors_header(&file[8..8 + len]));
            assert!(
                error.contains("unsupported safetensors dtype"),
                "{dtype}: {error}"
            );
        }
        let reversed = br#"{"x":{"dtype":"BF16","shape":[1],"data_offsets":[4,2]}}"#;
        assert!(message(parse_safetensors_header(reversed)).contains("end 2 < begin 4"));
    }

    #[test]
    fn i64_tensors_are_verified_against_the_census_rule_then_skipped() {
        let rules: &[Int64Rule] = &POSITION_IDS_RULE;
        let rule = &rules[0];
        let file = safetensors_file(&[("vision.position_ids", "I64", vec![1, 4], arange_bytes(4))]);
        let header = header_of(&file);

        check_int64_entries(&header.int64, rules, "test-arch").expect("named by the rule");
        check_int64_rules_present(&header.int64, rules, "test-arch").expect("present");
        verify_int64_content(rule.name, rule, &arange_bytes(4)).expect("exact arange passes");

        let mut perturbed = arange_bytes(4);
        perturbed[16] = 7;
        let error = message(verify_int64_content(rule.name, rule, &perturbed));
        assert!(error.contains("element 2 is 7"), "{error}");
        assert!(
            message(verify_int64_content(rule.name, rule, &arange_bytes(3))).contains("needs 32")
        );

        // An I64 tensor no rule names is still an error.
        let error = message(check_int64_entries(&header.int64, &[], "unlimited-ocr"));
        assert!(
            error.contains("is I64 [1, 4]; unlimited-ocr accepts I64 only"),
            "{error}"
        );
        assert!(error.contains("it has none"), "{error}");
        let other =
            safetensors_file(&[("text.token_type_ids", "I64", vec![1, 4], arange_bytes(4))]);
        let error = message(check_int64_entries(
            &header_of(&other).int64,
            rules,
            "test-arch",
        ));
        assert!(
            error.contains("vision.position_ids"),
            "names the accepted tensors: {error}"
        );

        let wrong_shape =
            safetensors_file(&[("vision.position_ids", "I64", vec![4], arange_bytes(4))]);
        let error = message(check_int64_entries(
            &header_of(&wrong_shape).int64,
            rules,
            "test-arch",
        ));
        assert!(error.contains("requires [1, 4]"), "{error}");

        let error = message(check_int64_rules_present(
            &BTreeMap::new(),
            rules,
            "test-arch",
        ));
        assert!(error.contains("which this checkpoint lacks"), "{error}");

        // The real censuses: v1 has no I64 tensor; nothing applies without a census.
        assert!(int64_rules(None).is_empty());
        let v1 = crate::native_engine::model_arch::arch_by_id("unlimited-ocr").expect("v1");
        assert!(int64_rules(Some(v1)).is_empty());
        let guard = test_rules::supply(&POSITION_IDS_RULE);
        assert_eq!(int64_rules(None), &POSITION_IDS_RULE);
        drop(guard);
        assert!(
            int64_rules(None).is_empty(),
            "the guard restores the census rules"
        );
    }

    #[test]
    fn multi_shard_source_sha256_matches_a_hand_computed_vector() {
        use sha2::Digest as _;
        let digest = |name: &str, content: &[u8]| ShardDigest {
            name: name.to_owned(),
            size: content.len() as u64,
            sha256: sha2::Sha256::digest(content).into(),
        };
        let one = digest("model-00001-of-00002.safetensors", b"shard one");
        let two = digest("model-00002-of-00002.safetensors", b"shard two");
        assert_eq!(
            one.sha256_hex(),
            "96af8b4933b944bc61107844c9cd0e6986f4add076f2db54982e868cf0630c96"
        );
        // python3 -c "import hashlib as h; a=h.sha256(b'shard one').digest();
        //             b=h.sha256(b'shard two').digest(); print(h.sha256(a+b).hexdigest())"
        assert_eq!(
            hex(&source_sha256(&[one.clone(), two.clone()])),
            "8e761e854cce76f2e12441944ac81d8072868c50781fb68093b04ab91c37bcd3"
        );
        assert_eq!(
            hex(&source_sha256(&[two, one.clone()])),
            "f32e1a6a34c40a8ef3e6eeebe682786c748f9dfead2d903233161c1249820191",
            "order matters"
        );
        assert_eq!(
            source_sha256(std::slice::from_ref(&one)),
            one.sha256,
            "one shard keeps its own SHA"
        );
        assert_eq!(
            provenance_json(&[one]),
            None,
            "one shard records no shard list"
        );
    }

    #[test]
    fn streaming_hash_matches_a_one_shot_digest_on_every_path() {
        use sha2::Digest as _;
        let content: Vec<u8> = (0..(2 * HASH_CHUNK_BYTES + 12_345))
            .map(|i| (i % 251) as u8)
            .collect();
        let expected: [u8; 32] = sha2::Sha256::digest(&content).into();
        let mut seen = 0usize;
        let (size, sha) = stream_sha256(content.as_slice(), Path::new("mem"), |chunk| {
            seen += chunk.len();
            Ok(())
        })
        .expect("stream hashes");
        assert_eq!(
            (size, sha, seen),
            (content.len() as u64, expected, content.len())
        );

        let dir = temp_dir("stream_hash");
        let one = safetensors_file(&[("a", "BF16", vec![1], vec![1, 0])]);
        let two = safetensors_file(&[("b", "BF16", vec![1], vec![2, 0])]);
        std::fs::write(dir.join("m-1.safetensors"), &one).expect("write shard 1");
        std::fs::write(dir.join("m-2.safetensors"), &two).expect("write shard 2");
        std::fs::write(
            dir.join(SHARD_INDEX_NAME),
            index_json(&[("a", "m-1.safetensors"), ("b", "m-2.safetensors")], 4),
        )
        .expect("write index");

        let (bytes, digest) =
            read_shard("m-1.safetensors", &dir.join("m-1.safetensors")).expect("read");
        assert_eq!(bytes, one);
        assert_eq!(digest.sha256, <[u8; 32]>::from(sha2::Sha256::digest(&one)));
        let digests = hash_shards(&dir.join(SHARD_INDEX_NAME)).expect("hash the indexed source");
        assert_eq!(digests.len(), 2);
        assert_eq!(digests[0], digest, "doctor and convert share one hash");
        assert_eq!(digests[1].size, two.len() as u64);
        let single = hash_shards(&dir.join("m-2.safetensors")).expect("hash a plain file");
        assert_eq!(single, digests[1..]);

        std::fs::write(
            dir.join(SHARD_INDEX_NAME),
            index_json(&[("a", "m-1.safetensors"), ("b", "m-3.safetensors")], 4),
        )
        .expect("rewrite index");
        let error = hash_shards(&dir.join(SHARD_INDEX_NAME)).expect_err("missing shard");
        assert!(matches!(error, FocrError::ModelNotFound(_)), "{error}");
        assert!(
            error.to_string().contains("lists shard m-3.safetensors"),
            "{error}"
        );
    }

    #[test]
    fn provenance_gate_accepts_the_convention_and_refuses_the_rest() {
        use sha2::Digest as _;
        let digests: Vec<ShardDigest> = ["a", "b"]
            .iter()
            .map(|tag| ShardDigest {
                name: format!("model-{tag}.safetensors"),
                size: 9,
                sha256: sha2::Sha256::digest(tag.as_bytes()).into(),
            })
            .collect();
        let convention = hex(&source_sha256(&digests));
        let provenance: FocrqProvenance =
            serde_json::from_str(&provenance_json(&digests).expect("two shards")).expect("parses");
        validate_source_shards(Some(&provenance), &convention).expect("the convention passes");
        validate_source_shards(None, "anything").expect("no provenance, no gate");
        let error = message(validate_source_shards(
            Some(&provenance),
            &digests[0].sha256_hex(),
        ));
        assert!(error.contains("convention digest"), "{error}");

        let refuse = |json: &str, needle: &str| {
            let provenance: FocrqProvenance = serde_json::from_str(json).expect("parses");
            let error = message(validate_source_shards(Some(&provenance), &convention));
            assert!(error.contains(needle), "{json}: {error}");
        };
        let a = digests[0].sha256_hex();
        let b = digests[1].sha256_hex();
        refuse(
            &format!(
                r#"{{"source_shards":[{{"name":"model-a.safetensors","size":9,"sha256":"{a}"}}]}}"#
            ),
            "lists 1 shard(s)",
        );
        refuse(
            &format!(
                r#"{{"source_shards":[{{"name":"model-b.safetensors","size":9,"sha256":"{b}"}},{{"name":"model-a.safetensors","size":9,"sha256":"{a}"}}]}}"#
            ),
            "not in index order",
        );
        refuse(
            &format!(
                r#"{{"source_shards":[{{"name":"model-a.safetensors","size":9,"sha256":"{}"}},{{"name":"model-b.safetensors","size":9,"sha256":"{b}"}}]}}"#,
                a.to_uppercase()
            ),
            "not 64 lowercase hex",
        );
        refuse(
            &format!(
                r#"{{"source_shards":[{{"name":"../a.safetensors","size":9,"sha256":"{a}"}},{{"name":"model-b.safetensors","size":9,"sha256":"{b}"}}]}}"#
            ),
            "not a plain",
        );
        refuse(
            &format!(
                r#"{{"source_shards":[{{"name":"model-a.safetensors","size":0,"sha256":"{a}"}},{{"name":"model-b.safetensors","size":9,"sha256":"{b}"}}]}}"#
            ),
            "size 0",
        );
    }
}
