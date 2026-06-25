//! The `.focrq` container **writer** — the byte-exact inverse of the committed
//! reader [`crate::native_engine::weights::Weights::from_focrq_bytes`].
//!
//! This is the heart of `focr convert`'s serialization step. The reader is the
//! source of truth for the on-disk byte layout; this writer emits exactly what
//! that reader parses, and the round-trip tests below prove it by writing tiny
//! containers and reading them back through `Weights::from_bytes`.
//!
//! ## On-disk layout (matches `from_focrq_bytes`)
//!
//! ```text
//! preamble (51 bytes):
//!   magic            b"FOCRQ\0"        (6 bytes)
//!   format_version   u32 LE            (4 bytes)
//!   arch_target      u8                (1 byte)
//!   source_sha256    [u8; 32]          (32 bytes)
//!   header_len       u64 LE            (8 bytes)
//! header_json[header_len]              (canonical UTF-8 JSON)
//! payload            <raw tensor + scale bytes, payload-relative offsets>
//! ```
//!
//! The header JSON the reader deserializes is `FocrqHeader { tensors:
//! BTreeMap<String, TensorRecord>, arch_target: u8, source_sha256: String,
//! license_notice: String }`. Each [`TensorRecord`] carries `dtype`, `shape`,
//! `byte_offset`, `byte_len`, and (for quantized dtypes) `scales_offset`,
//! `scales_len`, `group_size`, `tier`. The reader ignores any *extra* header
//! keys (serde does not deny unknown fields), so this writer additionally emits
//! the richer `docs/focrq-format.md` provenance/config/manifest fields for
//! forward-compatible artifacts without breaking the committed reader.
//!
//! ## Determinism (`docs/focrq-format.md` §"Writer Determinism")
//!
//! For a fixed input the output is byte-identical across runs: tensors are
//! emitted in **sorted name order**, the header JSON is hand-built in a fixed
//! canonical form, payload ranges are deterministic, and there is no RNG.
//! `byte_len` of every record equals `shape × dtype` exactly (what the reader's
//! `validate_directory` requires).

use std::collections::BTreeMap;
use std::path::Path;

use crate::error::{FocrError, FocrResult};

/// The `.focrq` magic — must match
/// [`crate::native_engine::weights::FOCRQ_MAGIC`] byte-for-byte.
pub const FOCRQ_MAGIC: &[u8; 6] = b"FOCRQ\0";

/// The format version this writer emits — must equal the reader's
/// [`crate::native_engine::weights::FOCRQ_FORMAT_VERSION`] (the reader refuses a
/// version greater than its own).
pub const FOCRQ_FORMAT_VERSION: u32 = 1;

/// Optional 64-byte payload alignment (`docs/focrq-format.md` §Payload). Off by
/// default so the round-trip-against-the-reader form is maximally simple; the
/// reader does not *require* alignment (it bounds-checks explicit offsets), so
/// turning it on changes only the inter-tensor padding, never correctness.
const ALIGN: usize = 64;

/// On-disk element dtype tag — the exact strings the reader's `DType`
/// `Deserialize` accepts (`"F32"`, `"F16"`, `"BF16"`, `"QInt8PerChan"`,
/// `"QInt4PerGroup"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteDType {
    /// IEEE-754 f32, little-endian (4 bytes/elem).
    F32,
    /// IEEE-754 f16, little-endian (2 bytes/elem). Reserved; v1 writer should not
    /// emit unless explicitly ledgered (BF16 is the high-precision store).
    F16,
    /// bfloat16, little-endian (2 bytes/elem) — the verbatim high-precision store.
    Bf16,
    /// Symmetric per-output-channel int8; inline f32 scale per output channel.
    QInt8PerChan,
    /// Group-quantized int4 (two nibbles/byte); inline per-group f32 scales.
    QInt4PerGroup,
}

impl WriteDType {
    /// The exact JSON string the reader's `DType` deserializes.
    #[must_use]
    fn as_json_str(self) -> &'static str {
        match self {
            WriteDType::F32 => "F32",
            WriteDType::F16 => "F16",
            WriteDType::Bf16 => "BF16",
            WriteDType::QInt8PerChan => "QInt8PerChan",
            WriteDType::QInt4PerGroup => "QInt4PerGroup",
        }
    }

    /// Whether this dtype carries inline scales (a quantized dtype).
    #[must_use]
    fn is_quantized(self) -> bool {
        matches!(self, WriteDType::QInt8PerChan | WriteDType::QInt4PerGroup)
    }

    /// Expected payload byte length for `numel` elements of this dtype — the same
    /// rule the reader's `expected_byte_len` enforces (int4 packs 2/byte).
    #[must_use]
    fn expected_byte_len(self, numel: usize) -> usize {
        match self {
            WriteDType::F32 => numel * 4,
            WriteDType::F16 | WriteDType::Bf16 => numel * 2,
            WriteDType::QInt8PerChan => numel,
            WriteDType::QInt4PerGroup => numel / 2,
        }
    }
}

/// One tensor staged for writing: dtype + shape + raw payload bytes + (for
/// quantized dtypes) inline scale bytes and group/tier metadata.
#[derive(Debug, Clone)]
struct PendingTensor {
    dtype: WriteDType,
    shape: Vec<usize>,
    data: Vec<u8>,
    scales: Vec<u8>,
    group_size: usize,
    tier: u8,
}

impl PendingTensor {
    #[allow(dead_code)]
    fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A deterministic `.focrq` container builder.
///
/// Stage tensors with [`FocrqBuilder::add_tensor`] /
/// [`FocrqBuilder::add_quantized`], then [`FocrqBuilder::build`] to the in-memory
/// blob or [`FocrqBuilder::write`] to a path. Tensors are emitted in sorted name
/// order so the output is byte-identical across runs.
#[derive(Debug, Clone)]
pub struct FocrqBuilder {
    arch_target: u8,
    source_sha256: [u8; 32],
    license_notice: String,
    /// Optional canonical JSON snippets for the richer forward-compat header
    /// fields (`provenance`, `model_config`, `packing_manifest`). Empty ⇒ the
    /// field is omitted. These are read back transparently (the reader ignores
    /// unknown keys).
    provenance_json: Option<String>,
    model_config_json: Option<String>,
    packing_manifest_json: Option<String>,
    align: bool,
    /// `name -> staged tensor`. `BTreeMap` ⇒ deterministic sorted emission.
    tensors: BTreeMap<String, PendingTensor>,
}

impl Default for FocrqBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FocrqBuilder {
    /// A fresh builder with the default Baidu MIT license notice, `Generic` arch
    /// target, and a zeroed source hash (set them with the `with_*` methods).
    #[must_use]
    pub fn new() -> Self {
        Self {
            arch_target: 0,
            source_sha256: [0u8; 32],
            license_notice: "Copyright (c) 2026 Baidu. MIT License.".to_string(),
            provenance_json: None,
            model_config_json: None,
            packing_manifest_json: None,
            align: false,
            tensors: BTreeMap::new(),
        }
    }

    /// Set the arch-target packing byte (`0` Generic, `1` Aarch64Smmla, `2`
    /// X86Vnni, `3` X86Amx).
    #[must_use]
    pub fn with_arch_target(mut self, arch: u8) -> Self {
        self.arch_target = arch;
        self
    }

    /// Set the source-safetensors sha256 (provenance, 32 bytes).
    #[must_use]
    pub fn with_source_sha256(mut self, sha: [u8; 32]) -> Self {
        self.source_sha256 = sha;
        self
    }

    /// Set the license notice (must be the non-empty Baidu MIT attribution in a
    /// real artifact).
    #[must_use]
    pub fn with_license_notice(mut self, notice: impl Into<String>) -> Self {
        self.license_notice = notice.into();
        self
    }

    /// Attach a canonical-JSON `provenance` object (forward-compat header field;
    /// the reader ignores it). Caller supplies already-canonical JSON.
    #[must_use]
    pub fn with_provenance_json(mut self, json: impl Into<String>) -> Self {
        self.provenance_json = Some(json.into());
        self
    }

    /// Attach a canonical-JSON `model_config` object (forward-compat header
    /// field).
    #[must_use]
    pub fn with_model_config_json(mut self, json: impl Into<String>) -> Self {
        self.model_config_json = Some(json.into());
        self
    }

    /// Attach a canonical-JSON `packing_manifest` object (forward-compat header
    /// field).
    #[must_use]
    pub fn with_packing_manifest_json(mut self, json: impl Into<String>) -> Self {
        self.packing_manifest_json = Some(json.into());
        self
    }

    /// Enable 64-byte payload alignment for tensor data and scale ranges
    /// (`docs/focrq-format.md` §Payload). Padding bytes are zeroed.
    #[must_use]
    pub fn with_alignment(mut self, on: bool) -> Self {
        self.align = on;
        self
    }

    /// Stage a high-precision (F32/F16/BF16) tensor.
    ///
    /// `bytes` must be the raw little-endian payload of `shape × dtype` length;
    /// this is validated at [`build`](Self::build) time.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if `name` is a duplicate, the dtype is a
    /// quantized one (use [`add_quantized`](Self::add_quantized)), or
    /// `bytes.len()` disagrees with `shape × dtype`.
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        dtype: WriteDType,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    ) -> FocrResult<()> {
        let name = name.into();
        if dtype.is_quantized() {
            return Err(FocrError::FormatMismatch(format!(
                "add_tensor: {name:?} is a quantized dtype {:?}; use add_quantized",
                dtype
            )));
        }
        self.insert_checked(name, dtype, shape, bytes, Vec::new(), 0, 0)
    }

    /// Stage a quantized (QInt8PerChan/QInt4PerGroup) tensor with inline scales.
    ///
    /// `data` is the packed weight payload (int8 bytes, or int4 nibbles 2/byte);
    /// `scales` is the little-endian f32 inline scale array. `group_size`/`tier`
    /// apply to int4 (`group_size = 0`, `tier = 0` for int8).
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] on a duplicate name, a non-quantized dtype,
    /// or a `data` length that disagrees with `shape × dtype`.
    // kernel signature: args are tensor dims/scales
    #[allow(clippy::too_many_arguments)]
    pub fn add_quantized(
        &mut self,
        name: impl Into<String>,
        dtype: WriteDType,
        shape: Vec<usize>,
        data: Vec<u8>,
        scales: Vec<u8>,
        group_size: usize,
        tier: u8,
    ) -> FocrResult<()> {
        let name = name.into();
        if !dtype.is_quantized() {
            return Err(FocrError::FormatMismatch(format!(
                "add_quantized: {name:?} dtype {:?} is not quantized; use add_tensor",
                dtype
            )));
        }
        self.insert_checked(name, dtype, shape, data, scales, group_size, tier)
    }

    /// Insert with a byte-length sanity check matching the reader's
    /// `validate_directory` rule (`byte_len == shape × dtype`).
    // kernel signature: args are tensor dims/scales
    #[allow(clippy::too_many_arguments)]
    fn insert_checked(
        &mut self,
        name: String,
        dtype: WriteDType,
        shape: Vec<usize>,
        data: Vec<u8>,
        scales: Vec<u8>,
        group_size: usize,
        tier: u8,
    ) -> FocrResult<()> {
        if self.tensors.contains_key(&name) {
            return Err(FocrError::FormatMismatch(format!(
                "add tensor: duplicate name {name:?}"
            )));
        }
        let numel: usize = shape.iter().product();
        let expected = dtype.expected_byte_len(numel);
        if data.len() != expected {
            return Err(FocrError::FormatMismatch(format!(
                "tensor {name:?}: data len {} != shape×dtype {} ({:?}, shape {:?})",
                data.len(),
                expected,
                dtype,
                shape
            )));
        }
        self.tensors.insert(
            name,
            PendingTensor {
                dtype,
                shape,
                data,
                scales,
                group_size,
                tier,
            },
        );
        Ok(())
    }

    /// Number of staged tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether no tensors are staged.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Serialize the staged tensors to the full `.focrq` blob (preamble + header
    /// + payload), byte-exactly readable by the committed reader.
    #[must_use]
    pub fn build(&self) -> Vec<u8> {
        // ── Pass 1: lay out the payload, recording each tensor's payload-relative
        // data/scale offsets. ──
        let mut payload: Vec<u8> = Vec::new();
        let mut records: Vec<(String, TensorLayout)> = Vec::with_capacity(self.tensors.len());

        for (name, t) in &self.tensors {
            self.maybe_align(&mut payload);
            let byte_offset = payload.len();
            payload.extend_from_slice(&t.data);
            let byte_len = t.data.len();

            let (scales_offset, scales_len) = if t.dtype.is_quantized() {
                self.maybe_align(&mut payload);
                let so = payload.len();
                payload.extend_from_slice(&t.scales);
                (so, t.scales.len())
            } else {
                (0usize, 0usize)
            };

            records.push((
                name.clone(),
                TensorLayout {
                    dtype: t.dtype,
                    shape: t.shape.clone(),
                    byte_offset,
                    byte_len,
                    scales_offset,
                    scales_len,
                    group_size: t.group_size,
                    tier: t.tier,
                },
            ));
        }

        // ── Pass 2: build the canonical header JSON. ──
        let header = self.build_header_json(&records);
        let header_bytes = header.into_bytes();

        // ── Pass 3: assemble preamble + header + payload. ──
        let mut blob = Vec::with_capacity(51 + header_bytes.len() + payload.len());
        blob.extend_from_slice(FOCRQ_MAGIC);
        blob.extend_from_slice(&FOCRQ_FORMAT_VERSION.to_le_bytes());
        blob.push(self.arch_target);
        blob.extend_from_slice(&self.source_sha256);
        blob.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        blob.extend_from_slice(&header_bytes);
        blob.extend_from_slice(&payload);
        blob
    }

    /// Serialize and write the `.focrq` blob to `path`.
    ///
    /// # Errors
    /// [`FocrError::Other`] if the file write fails.
    pub fn write(&self, path: &Path) -> FocrResult<()> {
        let blob = self.build();
        std::fs::write(path, &blob).map_err(|e| {
            FocrError::Other(anyhow::anyhow!("writing .focrq to {}: {e}", path.display()))
        })
    }

    /// Pad the payload to the next 64-byte boundary with zeros (no-op unless
    /// alignment is enabled).
    fn maybe_align(&self, payload: &mut Vec<u8>) {
        if !self.align {
            return;
        }
        let rem = payload.len() % ALIGN;
        if rem != 0 {
            payload.resize(payload.len() + (ALIGN - rem), 0);
        }
    }

    /// Build the canonical header JSON exactly as the reader's `FocrqHeader`
    /// deserializes it: the `tensors` map + `arch_target` + `source_sha256` +
    /// `license_notice`, plus the forward-compat `format_version` /
    /// `provenance` / `model_config` / `packing_manifest` fields (ignored by the
    /// reader). Keys are emitted in a fixed order for byte-stable output.
    fn build_header_json(&self, records: &[(String, TensorLayout)]) -> String {
        let mut s = String::new();
        s.push('{');

        // arch_target (u8, the reader's field).
        s.push_str("\"arch_target\":");
        s.push_str(&self.arch_target.to_string());
        s.push(',');

        // format_version (forward-compat / spec field).
        s.push_str("\"format_version\":");
        s.push_str(&FOCRQ_FORMAT_VERSION.to_string());
        s.push(',');

        // license_notice (the reader's field).
        s.push_str("\"license_notice\":");
        push_json_string(&mut s, &self.license_notice);
        s.push(',');

        // model_config (forward-compat; caller-supplied canonical JSON).
        if let Some(mc) = &self.model_config_json {
            s.push_str("\"model_config\":");
            s.push_str(mc);
            s.push(',');
        }

        // packing_manifest (forward-compat).
        if let Some(pm) = &self.packing_manifest_json {
            s.push_str("\"packing_manifest\":");
            s.push_str(pm);
            s.push(',');
        }

        // provenance (forward-compat).
        if let Some(pv) = &self.provenance_json {
            s.push_str("\"provenance\":");
            s.push_str(pv);
            s.push(',');
        }

        // source_sha256 as a hex string (the reader's String field; it prefers
        // this over the preamble bytes when non-empty).
        s.push_str("\"source_sha256\":");
        push_json_string(&mut s, &hex_encode(&self.source_sha256));
        s.push(',');

        // tensors map (the reader's BTreeMap<String, TensorRecord>).
        s.push_str("\"tensors\":{");
        for (i, (name, layout)) in records.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            push_json_string(&mut s, name);
            s.push(':');
            layout.push_record_json(&mut s);
        }
        s.push('}');

        s.push('}');
        s
    }
}

/// The computed on-disk layout of one tensor (after payload placement) — the
/// data the reader's `TensorRecord` needs.
struct TensorLayout {
    dtype: WriteDType,
    shape: Vec<usize>,
    byte_offset: usize,
    byte_len: usize,
    scales_offset: usize,
    scales_len: usize,
    group_size: usize,
    tier: u8,
}

impl TensorLayout {
    /// Emit this record as the canonical JSON object the reader's `TensorRecord`
    /// deserializes (`dtype`, `shape`, `byte_offset`, `byte_len`, and — for
    /// quantized dtypes — `scales_offset`, `scales_len`, `group_size`, `tier`).
    fn push_record_json(&self, s: &mut String) {
        s.push('{');
        s.push_str("\"byte_len\":");
        s.push_str(&self.byte_len.to_string());
        s.push_str(",\"byte_offset\":");
        s.push_str(&self.byte_offset.to_string());
        s.push_str(",\"dtype\":");
        push_json_string(s, self.dtype.as_json_str());

        if self.dtype.is_quantized() {
            s.push_str(",\"group_size\":");
            s.push_str(&self.group_size.to_string());
            s.push_str(",\"scales_len\":");
            s.push_str(&self.scales_len.to_string());
            s.push_str(",\"scales_offset\":");
            s.push_str(&self.scales_offset.to_string());
        }

        s.push_str(",\"shape\":[");
        for (i, d) in self.shape.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&d.to_string());
        }
        s.push(']');

        if self.dtype.is_quantized() {
            s.push_str(",\"tier\":");
            s.push_str(&self.tier.to_string());
        }

        s.push('}');
    }
}

/// Append a JSON-escaped string literal (with surrounding quotes) to `s`.
///
/// Escapes the JSON-mandatory control characters and `"` / `\`. The tensor names
/// and the Baidu/MIT notice are plain ASCII/UTF-8 with at most these; this keeps
/// the writer dependency-free and the output canonical.
fn push_json_string(s: &mut String, value: &str) {
    s.push('"');
    for ch in value.chars() {
        match ch {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                s.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => s.push(c),
        }
    }
    s.push('"');
}

/// Lowercase-hex-encode a byte slice (matches the reader's `hex_encode`).
fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_engine::weights::{DType, Weights};
    use half::bf16;

    fn bf16_le(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect()
    }

    fn f32_le(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|&v| v.to_le_bytes()).collect()
    }

    // ── round-trip through the committed reader ─────────────────────────────

    #[test]
    fn roundtrips_bf16_tensor_through_reader() {
        let vals = [1.0f32, -2.0, 0.5, 3.0, 0.0, -0.25];
        let mut b = FocrqBuilder::new()
            .with_arch_target(2)
            .with_source_sha256([7u8; 32]);
        b.add_tensor("w", WriteDType::Bf16, vec![2, 3], bf16_le(&vals))
            .unwrap();
        let blob = b.build();

        let w = Weights::from_bytes(blob).unwrap();
        assert!(w.is_focrq());
        assert_eq!(w.len(), 1);
        assert_eq!(w.arch_target(), 2);
        assert_eq!(w.source_sha256(), &"07".repeat(32));
        let view = w.tensor("w").unwrap();
        assert_eq!(view.dtype, DType::BF16);
        assert_eq!(view.shape, &[2, 3]);
        let m = w.mat("w").unwrap();
        assert_eq!(m.shape(), (2, 3));
        assert_eq!(m.data, vals);
    }

    #[test]
    fn roundtrips_f32_tensor_through_reader() {
        let vals = [1.5f32, -0.125, 1024.0, -3.0];
        let mut b = FocrqBuilder::new();
        b.add_tensor("bias", WriteDType::F32, vec![4], f32_le(&vals))
            .unwrap();
        let w = Weights::from_bytes(b.build()).unwrap();
        let m = w.mat("bias").unwrap();
        assert_eq!(m.shape(), (1, 4));
        assert_eq!(m.data, vals);
    }

    #[test]
    fn roundtrips_two_tensors_by_byte_range() {
        let a = [1.0f32, 2.0];
        let bb = [9.0f32, 8.0, 7.0];
        let mut b = FocrqBuilder::new();
        b.add_tensor("a", WriteDType::Bf16, vec![2], bf16_le(&a))
            .unwrap();
        b.add_tensor("b", WriteDType::F32, vec![3], f32_le(&bb))
            .unwrap();
        let w = Weights::from_bytes(b.build()).unwrap();
        assert_eq!(w.mat("a").unwrap().data, vec![1.0, 2.0]);
        assert_eq!(w.mat("b").unwrap().data, vec![9.0, 8.0, 7.0]);
    }

    #[test]
    fn roundtrips_qint8_through_reader() {
        // n=2, k=3: 6 int8 weights + 2 f32 scales.
        let w_bytes: Vec<u8> = [1i8, -2, 3, 4, -5, 6].iter().map(|&v| v as u8).collect();
        let scale_bytes = f32_le(&[0.1, 0.2]);
        let mut b = FocrqBuilder::new();
        b.add_quantized(
            "q",
            WriteDType::QInt8PerChan,
            vec![2, 3],
            w_bytes,
            scale_bytes,
            0,
            0,
        )
        .unwrap();
        let w = Weights::from_bytes(b.build()).unwrap();
        let q = w.qint8("q").unwrap();
        assert_eq!(q.n, 2);
        assert_eq!(q.k, 3);
        assert_eq!(q.w, vec![1i8, -2, 3, 4, -5, 6]);
        assert_eq!(q.scales, vec![0.1, 0.2]);
    }

    #[test]
    fn roundtrips_qint4_through_reader() {
        // n=2, k=4, group_size=2 => 2 packed bytes/row (4 total), 4 scales.
        let packed = vec![0x21u8, 0x43, 0x65, 0x87];
        let scale_bytes = f32_le(&[0.1, 0.2, 0.3, 0.4]);
        let mut b = FocrqBuilder::new();
        b.add_quantized(
            "e",
            WriteDType::QInt4PerGroup,
            vec![2, 4],
            packed.clone(),
            scale_bytes,
            2,
            3,
        )
        .unwrap();
        let w = Weights::from_bytes(b.build()).unwrap();
        let q = w.qint4("e").unwrap();
        assert_eq!(q.n, 2);
        assert_eq!(q.k, 4);
        assert_eq!(q.group_size, 2);
        assert_eq!(q.tier, 3);
        assert_eq!(q.packed, packed);
        assert_eq!(q.scales, vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn license_notice_survives_roundtrip() {
        let mut b =
            FocrqBuilder::new().with_license_notice("Copyright (c) 2026 Baidu. MIT License.");
        b.add_tensor("x", WriteDType::Bf16, vec![1], bf16_le(&[1.0]))
            .unwrap();
        let w = Weights::from_bytes(b.build()).unwrap();
        assert_eq!(w.license_notice(), "Copyright (c) 2026 Baidu. MIT License.");
    }

    #[test]
    fn forward_compat_header_fields_are_ignored_by_reader() {
        // Extra provenance/model_config/packing_manifest keys must not break the
        // reader (serde ignores unknown fields).
        let mut b = FocrqBuilder::new()
            .with_provenance_json(r#"{"hf_commit":"abc","source_sha256_hex":"00"}"#)
            .with_model_config_json(r#"{"hidden_size":1280,"use_mla":false}"#)
            .with_packing_manifest_json(r#"{"quant_recipe":"decoder-ffn-int8-v1"}"#);
        b.add_tensor("x", WriteDType::Bf16, vec![2], bf16_le(&[1.0, 2.0]))
            .unwrap();
        let w = Weights::from_bytes(b.build()).unwrap();
        assert_eq!(w.mat("x").unwrap().data, vec![1.0, 2.0]);
    }

    // ── census interop ──────────────────────────────────────────────────────

    #[test]
    fn written_blob_passes_reader_census() {
        let mut b = FocrqBuilder::new();
        b.add_tensor("alpha", WriteDType::F32, vec![1], f32_le(&[1.0]))
            .unwrap();
        b.add_tensor("beta", WriteDType::F32, vec![1], f32_le(&[2.0]))
            .unwrap();
        let w = Weights::from_bytes(b.build()).unwrap();
        assert!(w.census(["alpha", "beta"]).is_ok());
        assert!(w.census(["alpha"]).is_err());
    }

    // ── determinism ─────────────────────────────────────────────────────────

    #[test]
    fn build_is_byte_deterministic() {
        let make = || {
            let mut b = FocrqBuilder::new()
                .with_arch_target(1)
                .with_source_sha256([5u8; 32]);
            // Insert in non-sorted order; sorted emission must make output equal.
            b.add_tensor("zeta", WriteDType::Bf16, vec![2], bf16_le(&[3.0, 4.0]))
                .unwrap();
            b.add_tensor("alpha", WriteDType::F32, vec![2], f32_le(&[1.0, 2.0]))
                .unwrap();
            b.build()
        };
        assert_eq!(make(), make());
    }

    #[test]
    fn tensor_emission_order_is_sorted_by_name() {
        let mut b = FocrqBuilder::new();
        b.add_tensor("zeta", WriteDType::F32, vec![1], f32_le(&[1.0]))
            .unwrap();
        b.add_tensor("alpha", WriteDType::F32, vec![1], f32_le(&[2.0]))
            .unwrap();
        b.add_tensor("mid", WriteDType::F32, vec![1], f32_le(&[3.0]))
            .unwrap();
        let w = Weights::from_bytes(b.build()).unwrap();
        let names: Vec<&str> = w.names().collect();
        assert_eq!(names, vec!["alpha", "mid", "zeta"]);
    }

    // ── file write/load round-trip ──────────────────────────────────────────

    #[test]
    fn write_to_file_and_load_through_reader() {
        let vals = [1.0f32, -2.0, 4.0, 8.0];
        let mut b = FocrqBuilder::new()
            .with_arch_target(1)
            .with_source_sha256([3u8; 32]);
        b.add_tensor("t", WriteDType::Bf16, vec![2, 2], bf16_le(&vals))
            .unwrap();

        let path = std::env::temp_dir().join(format!(
            "focrq_writer_{}_{}.focrq",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        b.write(&path).unwrap();
        let w = Weights::load(&path).unwrap();
        let m = w.mat("t").unwrap();
        assert_eq!(m.shape(), (2, 2));
        assert_eq!(m.data, vals);
        let _ = std::fs::remove_file(&path);
    }

    // ── alignment ───────────────────────────────────────────────────────────

    #[test]
    fn aligned_payload_still_roundtrips_through_reader() {
        let a = [1.0f32, 2.0, 3.0];
        let bb = [9.0f32];
        let mut b = FocrqBuilder::new().with_alignment(true);
        b.add_tensor("a", WriteDType::F32, vec![3], f32_le(&a))
            .unwrap();
        b.add_tensor("b", WriteDType::Bf16, vec![1], bf16_le(&bb))
            .unwrap();
        let blob = b.build();
        let w = Weights::from_bytes(blob).unwrap();
        assert_eq!(w.mat("a").unwrap().data, vec![1.0, 2.0, 3.0]);
        assert_eq!(w.mat("b").unwrap().data, vec![9.0]);
    }

    // ── end-to-end with the quantizers ──────────────────────────────────────

    #[test]
    fn int8_quantizer_output_roundtrips_through_writer_and_reader() {
        use crate::quant::int8::quantize_int8_f32;
        // A 2x3 weight; quantize then write then read back as QInt8.
        let w = [127.0f32, 0.0, -64.0, 254.0, -254.0, 0.0];
        let q = quantize_int8_f32(&w, 2, 3);
        let mut b = FocrqBuilder::new();
        b.add_quantized(
            "expert.down_proj",
            WriteDType::QInt8PerChan,
            vec![2, 3],
            q.weight_bytes(),
            q.scale_bytes(),
            0,
            0,
        )
        .unwrap();
        let weights = Weights::from_bytes(b.build()).unwrap();
        let rq = weights.qint8("expert.down_proj").unwrap();
        assert_eq!(rq.n, 2);
        assert_eq!(rq.k, 3);
        assert_eq!(rq.w, q.q);
        assert_eq!(rq.scales, q.scales);
    }

    #[test]
    fn int4_packer_output_roundtrips_through_writer_and_reader() {
        use crate::quant::int4::pack_int4_f32;
        // n=1, k=32, group 16 -> 16 packed bytes, 2 scales.
        let vals: Vec<f32> = (0..32).map(|i| (i as f32) - 16.0).collect();
        let q = pack_int4_f32(&vals, 1, 32, 16);
        let mut b = FocrqBuilder::new();
        b.add_quantized(
            "expert.up_proj",
            WriteDType::QInt4PerGroup,
            vec![1, 32],
            q.packed_bytes(),
            q.scale_bytes(),
            16,
            4,
        )
        .unwrap();
        let weights = Weights::from_bytes(b.build()).unwrap();
        let rq = weights.qint4("expert.up_proj").unwrap();
        assert_eq!(rq.n, 1);
        assert_eq!(rq.k, 32);
        assert_eq!(rq.group_size, 16);
        assert_eq!(rq.packed, q.packed);
        assert_eq!(rq.scales, q.scales);
    }

    // ── error paths ─────────────────────────────────────────────────────────

    #[test]
    fn rejects_wrong_byte_len() {
        let mut b = FocrqBuilder::new();
        // shape [2,3] bf16 = 12 bytes, supply 4.
        let err = b
            .add_tensor("x", WriteDType::Bf16, vec![2, 3], vec![0u8; 4])
            .unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
    }

    #[test]
    fn rejects_duplicate_name() {
        let mut b = FocrqBuilder::new();
        b.add_tensor("x", WriteDType::F32, vec![1], f32_le(&[1.0]))
            .unwrap();
        let err = b
            .add_tensor("x", WriteDType::F32, vec![1], f32_le(&[2.0]))
            .unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
    }

    #[test]
    fn add_tensor_rejects_quantized_dtype() {
        let mut b = FocrqBuilder::new();
        let err = b
            .add_tensor("x", WriteDType::QInt8PerChan, vec![1], vec![0u8; 1])
            .unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
    }

    #[test]
    fn add_quantized_rejects_high_precision_dtype() {
        let mut b = FocrqBuilder::new();
        let err = b
            .add_quantized("x", WriteDType::F32, vec![1], vec![0u8; 4], vec![], 0, 0)
            .unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
    }
}
