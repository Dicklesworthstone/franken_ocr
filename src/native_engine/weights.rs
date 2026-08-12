//! `.focrq` reader + safetensors fallback + the WeightsManifest census
//! (PROPOSED_ARCHITECTURE.md §6.12, §7).
//!
//! Two on-disk forms feed one accessor surface:
//!
//! * The **`.focrq` container** ([SPEC §7]): a self-describing, length-prefixed
//!   blob — magic `b"FOCRQ\0"`, a `u32` format version, an arch-target byte, a
//!   32-byte source sha256, a license notice, a frozen `model_config` JSON, then
//!   a JSON **tensor directory** (`name -> {dtype, shape, byte_offset,
//!   byte_len, scales_offset, scales_len, group_size?, tier?}`) indexing one
//!   contiguous payload by byte range. Quantized tensors carry their scales
//!   inline (int8 per-output-channel, int4 per-group). The container is read
//!   with the `franken_whisper` ggml/safetensors parser pattern: keep one
//!   backing blob, validate the magic, parse the header, then index the payload
//!   by byte range — no per-tensor copies until an accessor asks. The backing is
//!   an owned `Vec<u8>` by default. `FOCR_MMAP=1` explicitly opts into a read-only
//!   mmap for immutable, trusted artifacts.
//!
//! * The **safetensors fallback**: the upstream 6.67 GB bf16 shard, so weights
//!   can load before the quantizer/converter exists. Standard safetensors
//!   layout (`u64` LE header length + JSON tensor directory + payload). bf16 is
//!   widened to f32 via the `half` crate at the accessor boundary
//!   (PROPOSED_ARCHITECTURE.md §6.12: **BF16, never F16** — f16 narrowing is a
//!   measured `DISCREPANCIES.md` divergence, never the silent default).
//!
//! Both forms expose the same accessor API the vision/decoder modules are
//! blocked on:
//! * [`Weights::tensor`] — a zero-copy [`TensorView`] (dtype + shape + raw
//!   payload slice) for a named tensor.
//! * [`Weights::mat`] — a fully widened f32 [`Mat`] (2-D tensors only).
//! * [`Weights::qint8`] / [`Weights::qint4`] — typed quantized weights.
//!
//! On load a **WeightsManifest census** (PROPOSED_ARCHITECTURE.md §6.12, §7)
//! checks the loaded tensor set against an expected name set so a wrong/stale
//! checkpoint is rejected at load time with [`FocrError::FormatMismatch`], not
//! surfaced later as garbage OCR output.
//!
//! Memory mapping note: `Weights::load` defaults to owned bytes because a mapped
//! file that another process truncates can fault the process. `FOCR_MMAP=1` opts
//! into `memmap2` only for deployment envelopes where the artifact inode is
//! immutable for the lifetime of the process. Mapping failures still fall back
//! to owned bytes.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use half::bf16;
use serde::Deserialize;

use super::model_arch;
use super::tensor::{Mat, QInt4, QInt8};
use crate::FOCR_MODEL_LICENSE_NOTICE;
use crate::error::{FocrError, FocrResult};
use crate::quant::int4::VALID_GROUP_SIZES;

fn checked_shape_len(name: &str, lhs: usize, rhs: usize, expr: &str) -> FocrResult<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        FocrError::FormatMismatch(format!(
            "tensor {name:?}: {expr} overflows usize ({lhs} * {rhs})"
        ))
    })
}

/// Resolve the architecture id a `.focrq` declares against the model registry.
///
/// An **absent** `model_id` (`""`, the v1 form, since `model_id` is a v2 header
/// addition) resolves to the default architecture — Unlimited-OCR — so every
/// pre-existing v1 artifact keeps loading unchanged. A **non-empty** id that is
/// not in the registry is a forward-incompatible artifact and is **refused**
/// loudly (mirroring the `format_version > max` rejection), rather than silently
/// loaded as the default model.
fn resolve_model_id(declared: &str) -> FocrResult<&'static str> {
    if declared.is_empty() {
        return Ok(model_arch::default_arch().id());
    }
    model_arch::arch_by_id(declared)
        .map(|arch| arch.id())
        .ok_or_else(|| {
            FocrError::FormatMismatch(format!(
                ".focrq declares unknown model_id {declared:?} \
                 (not in the model registry; this binary cannot load it)"
            ))
        })
}

/// Validate the `.focrq` license notice against the notice the *declared
/// architecture* requires.
///
/// * **Unlimited-OCR** (and every v1 artifact, which resolves to it): the Baidu
///   MIT contract, accepting any semantically-equivalent notice that carries the
///   required copyright + MIT-license tokens (back-compat with the historical
///   check — the `model_config`/notice text was hand-authored in early artifacts).
/// * **Any other registered arch** (e.g. GOT-OCR2 → Apache-2.0/StepFun): the
///   notice must equal that arch's declared [`model_arch::ModelArch::license_notice`] exactly.
///
/// `model_id` is the already-resolved (registry-known) id from
/// [`resolve_model_id`], so the lookup here cannot miss.
fn validate_license_notice(notice: &str, model_id: &str) -> FocrResult<()> {
    if model_id == model_arch::default_arch().id() {
        return if notice == FOCR_MODEL_LICENSE_NOTICE
            || (notice.contains("Copyright (c) 2026 Baidu") && notice.contains("MIT License"))
        {
            Ok(())
        } else {
            Err(FocrError::FormatMismatch(
                ".focrq license_notice must include Copyright (c) 2026 Baidu and MIT License"
                    .into(),
            ))
        };
    }
    match model_arch::arch_by_id(model_id) {
        Some(arch) if notice == arch.license_notice() => Ok(()),
        Some(arch) => Err(FocrError::FormatMismatch(format!(
            ".focrq license_notice does not match the registered {model_id} notice \
             (expected {:?})",
            arch.license_notice()
        ))),
        None => Err(FocrError::FormatMismatch(format!(
            ".focrq license_notice: unknown model_id {model_id:?}"
        ))),
    }
}

fn validate_source_sha256_hex(source_sha256: &str) -> FocrResult<()> {
    if source_sha256.len() == 64
        && source_sha256
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    {
        Ok(())
    } else {
        Err(FocrError::FormatMismatch(
            ".focrq source_sha256 must be 64 lowercase hex chars when present".into(),
        ))
    }
}

/// The `.focrq` container magic ([SPEC §7]). Six bytes including the trailing
/// `\0` — a loud, byte-exact rejection on mismatch.
pub const FOCRQ_MAGIC: &[u8; 6] = b"FOCRQ\0";

/// The `.focrq` format version this build writes/reads ([SPEC §7]). The loader
/// **refuses** any blob whose `format_version > FOCRQ_FORMAT_VERSION` (a newer
/// layout than this binary understands), per the plan's
/// "loader refuses version > binary's".
pub const FOCRQ_FORMAT_VERSION: u32 = 1;

/// Fixed `.focrq` preamble size: `magic(6) + format_version(4) + arch(1) +
/// source_sha256(32) + header_len(8)`. Everything after it is the header JSON.
pub const FOCRQ_PREAMBLE: usize = 6 + 4 + 1 + 32 + 8;

/// Highest defined `.focrq` architecture-target byte in format v1.
const MAX_ARCH_TARGET: u8 = 3;

/// Validate that an on-disk architecture target is one of the closed v1 enum.
pub(super) fn validate_arch_target(arch_target: u8) -> FocrResult<()> {
    if arch_target <= MAX_ARCH_TARGET {
        return Ok(());
    }
    Err(FocrError::FormatMismatch(format!(
        ".focrq arch_target {arch_target} is unsupported; format v1 defines only 0..={MAX_ARCH_TARGET}"
    )))
}

/// On-disk element dtype of a stored tensor ([SPEC §7] dtype set).
///
/// High-precision tensors are stored BF16 verbatim (the checkpoint is bf16) and
/// widened BF16→f32 at the accessor boundary. Quantized tensors carry scales
/// inline. `F16` exists only so the loader can *read* an f16 blob if one is ever
/// produced as a ledgered divergence — it is never the default high-precision
/// store (PROPOSED_ARCHITECTURE.md §6.12: BF16, NOT F16).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum DType {
    /// IEEE-754 single precision, stored little-endian (4 bytes/elem).
    F32,
    /// IEEE-754 half precision, stored little-endian (2 bytes/elem).
    F16,
    /// bfloat16 (1-8-7), stored little-endian (2 bytes/elem) — the verbatim
    /// checkpoint store for high-precision tensors.
    BF16,
    /// Symmetric per-output-channel int8 (`scale = max|w_row|/127`, zero-point
    /// 0); scales stored inline as one f32 per output channel.
    QInt8PerChan,
    /// Group-quantized int4 (two nibbles/byte) with one f32 scale per
    /// `group_size`-element group along the contraction dim, plus a per-tensor
    /// precision tier.
    QInt4PerGroup,
}

impl DType {
    /// Map a safetensors dtype string (e.g. `"BF16"`, `"F32"`) to a [`DType`].
    ///
    /// The upstream shard is bf16; F32/F16 are accepted for robustness against
    /// mixed-precision checkpoints. Quantized dtypes never appear in a raw
    /// safetensors header (they are a `.focrq`-only concept).
    fn from_safetensors_str(s: &str) -> FocrResult<Self> {
        match s {
            "F32" => Ok(DType::F32),
            "F16" => Ok(DType::F16),
            "BF16" => Ok(DType::BF16),
            other => Err(FocrError::FormatMismatch(format!(
                "unsupported safetensors dtype {other:?} \
                 (expected F32/F16/BF16; the upstream Unlimited-OCR shard is BF16)"
            ))),
        }
    }
}

/// One entry of the `.focrq` / safetensors tensor directory: dtype + shape +
/// byte range into the single payload (plus inline scale range for quantized
/// tensors). This is the dependency-free byte-range index (§7); resolving a
/// tensor is a slice into the backing buffer, no copy.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorRecord {
    /// Element dtype of the stored bytes.
    pub dtype: DType,
    /// Logical shape (row-major). A 2-D `[rows, cols]` tensor maps to a [`Mat`].
    pub shape: Vec<usize>,
    /// Byte offset of the tensor's data within the payload.
    pub byte_offset: usize,
    /// Byte length of the tensor's data.
    pub byte_len: usize,
    /// Byte offset of the inline scales (quantized dtypes only; `0` otherwise).
    #[serde(default)]
    pub scales_offset: usize,
    /// Byte length of the inline scales (quantized dtypes only; `0` otherwise).
    #[serde(default)]
    pub scales_len: usize,
    /// Elements per quantization group along the contraction dim
    /// (`QInt4PerGroup` only).
    #[serde(default)]
    pub group_size: usize,
    /// Per-tensor precision tier from the water-filling allocator
    /// (`QInt4PerGroup` only; plan §9.7).
    #[serde(default)]
    pub tier: u8,
}

impl TensorRecord {
    /// Total element count, saturating at [`usize::MAX`] instead of panicking on
    /// a malformed in-memory shape. Loader validation uses [`Self::checked_numel`]
    /// so corrupt on-disk directories still fail loudly.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape
            .iter()
            .copied()
            .fold(1usize, usize::saturating_mul)
    }

    fn checked_numel(&self, name: &str) -> FocrResult<usize> {
        self.shape.iter().copied().try_fold(1usize, |acc, dim| {
            acc.checked_mul(dim).ok_or_else(|| {
                FocrError::FormatMismatch(format!(
                    "tensor {name:?}: shape {:?} element count overflows usize",
                    self.shape
                ))
            })
        })
    }

    /// Bytes-per-element for the **payload** of this dtype (scales are separate).
    #[must_use]
    fn elem_bytes(&self) -> usize {
        match self.dtype {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::QInt8PerChan => 1,
            // int4 packs two elements per byte; bytes are `numel / 2`.
            DType::QInt4PerGroup => 0,
        }
    }

    /// Expected payload byte length implied by `shape` + `dtype`, used by the
    /// census to catch a directory that disagrees with its own shapes.
    fn expected_byte_len(&self, name: &str) -> FocrResult<usize> {
        let numel = self.checked_numel(name)?;
        match self.dtype {
            DType::QInt4PerGroup => {
                if !numel.is_multiple_of(2) {
                    return Err(FocrError::FormatMismatch(format!(
                        "tensor {name:?}: QInt4 shape {:?} has odd element count {numel}",
                        self.shape
                    )));
                }
                Ok(numel / 2)
            }
            _ => numel.checked_mul(self.elem_bytes()).ok_or_else(|| {
                FocrError::FormatMismatch(format!(
                    "tensor {name:?}: byte length for {:?}, shape {:?} overflows usize",
                    self.dtype, self.shape
                ))
            }),
        }
    }
}

/// The parsed `.focrq` / safetensors header: the tensor directory plus the
/// provenance/config metadata (`.focrq` only). Deserialized straight from the
/// header JSON.
#[derive(Debug, Clone, Deserialize)]
struct FocrqHeader {
    /// `name -> record`. `BTreeMap` so the census is order-independent and the
    /// name set is deterministic.
    tensors: BTreeMap<String, TensorRecord>,
    /// The arch-target packing byte (Generic/Aarch64Smmla/X86Vnni/X86Amx).
    #[serde(default)]
    arch_target: u8,
    /// 32-byte source-safetensors sha256, hex-encoded (provenance).
    #[serde(default)]
    source_sha256: String,
    /// MIT/Baidu model-weights notice — MUST be present in a real artifact.
    /// Writers use [`crate::FOCR_MODEL_LICENSE_NOTICE`] as the single source of
    /// truth (plan §2.2 / §11); the reader accepts semantically equivalent
    /// notices that include the required copyright and MIT-license tokens.
    #[serde(default)]
    license_notice: String,
    /// The model-architecture id (`ModelArch::id`, e.g. `"got-ocr2"`) this
    /// artifact declares, so the loader selects the right arch from the registry.
    /// **Absent in v1 artifacts** (the field is a v2 addition) ⇒ deserializes to
    /// `""` ⇒ [`resolve_model_id`] defaults it to `unlimited-ocr` (the only model
    /// v1 ever produced), so every existing `.focrq` keeps loading unchanged.
    #[serde(default)]
    model_id: String,
    /// Optional packing manifest carrying the declared quant-recipe id (e.g.
    /// the conservative `unlimited-ocr-ffn-int8-…` or the explicitly non-default
    /// wasm `unlimited-ocr-wasm-experts-int4-…`). Retained so the post-load
    /// recipe validation and the decoder cache builders can branch on the
    /// artifact's own declaration (bd-4l71). Absent ⇒ `None`.
    #[serde(default)]
    packing_manifest: Option<FocrqPackingManifest>,
}

/// The subset of the `.focrq` `packing_manifest` header object the loader
/// retains (the rest is forward-compat and ignored).
#[derive(Debug, Clone, Deserialize)]
struct FocrqPackingManifest {
    #[serde(default)]
    quant_recipe: Option<String>,
}

/// A zero-copy view of one stored tensor: its dtype, shape, and the raw payload
/// bytes (plus inline scale bytes for quantized dtypes). The vision/decoder
/// modules widen this to f32 / int8 / int4 as the kernel demands.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    /// On-disk dtype of `data`.
    pub dtype: DType,
    /// Logical row-major shape.
    pub shape: &'a [usize],
    /// Raw payload bytes for this tensor (length == `record.byte_len`).
    pub data: &'a [u8],
    /// Raw inline scale bytes (empty for non-quantized dtypes).
    pub scales: &'a [u8],
    /// Group size (int4 only; `0` otherwise).
    pub group_size: usize,
    /// Precision tier (int4 only; `0` otherwise).
    pub tier: u8,
}

impl TensorView<'_> {
    /// Total element count of the tensor, saturating at [`usize::MAX`] rather
    /// than panicking if a caller constructs a malformed view.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape
            .iter()
            .copied()
            .fold(1usize, usize::saturating_mul)
    }

    /// Widen the whole tensor to an owned f32 `Vec` (any of F32/F16/BF16).
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if the dtype is a quantized one (use
    /// [`Weights::qint8`] / [`Weights::qint4`] instead) or the byte length is
    /// not a whole number of elements.
    pub fn to_f32_vec(&self) -> FocrResult<Vec<f32>> {
        decode_f32(self.dtype, self.data)
    }
}

/// The loaded weight set for one model: the whole backing blob plus the tensor
/// directory indexing it by byte range (PROPOSED_ARCHITECTURE.md §6.12).
///
/// `bytes` holds one backing blob: an owned buffer by default, or an explicitly
/// opted-in read-only mmap for trusted immutable artifacts.
/// For `.focrq` the directory's byte offsets are payload-relative, so
/// `payload_base` records where the payload begins inside `bytes`; for
/// safetensors the same field points just past the header. Every accessor
/// resolves `bytes[payload_base + off .. payload_base + off + len]`.
#[derive(Debug)]
pub struct Weights {
    /// The entire on-disk blob — owned bytes or a read-only mmap ([`Backing`]),
    /// behind an [`Arc`] so zero-copy weight views ([`SharedBytes`]) can hold
    /// the blob alive without duplicating payload bytes (bd-4l71 residency:
    /// the wasm artifact's 1.8 GB expert payload must not exist twice).
    bytes: std::sync::Arc<Backing>,
    /// Offset of the payload region within `bytes`.
    payload_base: usize,
    /// `name -> record` directory.
    directory: BTreeMap<String, TensorRecord>,
    /// `.focrq` arch-target byte (`0` for safetensors).
    arch_target: u8,
    /// hex source-sha256 (`""` for safetensors).
    source_sha256: String,
    /// The `.focrq` license notice (MIT/Baidu; `""` for safetensors).
    license_notice: String,
    /// The resolved model-architecture id (`ModelArch::id`). A v2 `.focrq`
    /// declares it; a v1 `.focrq` or raw safetensors resolves to the default
    /// `unlimited-ocr`. Always a registry-known id (validated at load).
    model_id: &'static str,
    /// Whether this came from a `.focrq` container (vs. a raw safetensors shard).
    is_focrq: bool,
    /// The quant-recipe id the artifact's `packing_manifest` declares (`None`
    /// for safetensors, recipe-less `.focrq`, or a manifest without the field).
    quant_recipe: Option<String>,
}

/// One piece of a [`Backing::Segmented`] blob: the absolute offset the piece
/// starts at, and its bytes.
#[derive(Debug)]
pub struct BlobSegment {
    /// Absolute offset of `bytes[0]` within the logical blob.
    start: u64,
    /// This piece's bytes.
    bytes: Vec<u8>,
}

/// The weight blob's storage: an owned buffer (the safe default), a read-only
/// memory map explicitly requested with `FOCR_MMAP=1`, or a SEGMENTED blob —
/// several buffers that together cover `[0, len)`.
///
/// The segmented arm exists for wasm32 (bd-syf2): Rust caps ANY single
/// allocation (and any slice) at `isize::MAX` = 2 GiB there, so the 3.0 GB
/// artifact simply cannot live in one `Vec<u8>` even though the 4 GiB heap has
/// room. Segment boundaries are planned to fall on TENSOR boundaries, so every
/// tensor's payload (and its inline scales) stays contiguous inside exactly one
/// segment and every accessor keeps returning a plain `&[u8]`.
///
/// There is deliberately NO `Deref<Target = [u8]>`: a segmented blob has no
/// single slice. [`Backing::range`] is the one chokepoint every reader goes
/// through, and it REFUSES a range that straddles two segments rather than
/// silently stitching a copy.
enum Backing {
    /// Fully-owned bytes (`std::fs::read` / in-memory blobs).
    Owned(Vec<u8>),
    /// Read-only file mapping.
    #[cfg(feature = "mmap")]
    Mapped(memmap2::Mmap),
    /// Contiguous, ordered, non-overlapping pieces covering `[0, len)`.
    Segmented {
        /// Pieces in ascending `start` order, each starting where the previous
        /// ended (validated by [`Weights::from_segments`]).
        segments: Vec<BlobSegment>,
        /// Total logical length (the sum of the segment lengths).
        len: u64,
    },
}

impl Backing {
    /// Total logical blob length.
    fn len(&self) -> u64 {
        match self {
            Backing::Owned(v) => v.len() as u64,
            #[cfg(feature = "mmap")]
            Backing::Mapped(m) => m.len() as u64,
            Backing::Segmented { len, .. } => *len,
        }
    }

    /// Resolve `[start, start + len)` to `(segment index, offset within that
    /// segment)`, or `None` if the range runs past the blob **or straddles a
    /// segment boundary**. Owned/Mapped blobs are one segment (index 0), so the
    /// native path is a pure bounds check.
    fn locate(&self, start: u64, len: usize) -> Option<(usize, usize)> {
        let end = start.checked_add(len as u64)?;
        if end > self.len() {
            return None;
        }
        match self {
            Backing::Owned(_) => Some((0, usize::try_from(start).ok()?)),
            #[cfg(feature = "mmap")]
            Backing::Mapped(_) => Some((0, usize::try_from(start).ok()?)),
            Backing::Segmented { segments, .. } => {
                // Rightmost segment whose start is <= `start`.
                let idx = match segments.binary_search_by_key(&start, |s| s.start) {
                    Ok(i) => i,
                    Err(0) => return None,
                    Err(i) => i - 1,
                };
                let seg = &segments[idx];
                if end > seg.start + seg.bytes.len() as u64 {
                    // Straddles into the next segment — a planning bug, never
                    // papered over with a copy.
                    return None;
                }
                Some((idx, usize::try_from(start - seg.start).ok()?))
            }
        }
    }

    /// The bytes of segment `idx` (index 0 is the whole blob for Owned/Mapped).
    fn segment_bytes(&self, idx: usize) -> &[u8] {
        match self {
            Backing::Owned(v) => {
                debug_assert_eq!(idx, 0);
                v
            }
            #[cfg(feature = "mmap")]
            Backing::Mapped(m) => {
                debug_assert_eq!(idx, 0);
                m
            }
            Backing::Segmented { segments, .. } => &segments[idx].bytes,
        }
    }

    /// THE chokepoint: borrow `[start, start + len)` as one contiguous slice,
    /// or `None` when the range is out of bounds or crosses a segment edge.
    fn range(&self, start: u64, len: usize) -> Option<&[u8]> {
        let (idx, off) = self.locate(start, len)?;
        self.segment_bytes(idx).get(off..off + len)
    }
}

impl std::fmt::Debug for Backing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backing::Owned(v) => write!(f, "Backing::Owned({} bytes)", v.len()),
            #[cfg(feature = "mmap")]
            Backing::Mapped(m) => write!(f, "Backing::Mapped({} bytes)", m.len()),
            Backing::Segmented { segments, len } => write!(
                f,
                "Backing::Segmented({len} bytes in {} segments)",
                segments.len()
            ),
        }
    }
}

/// The fixed-size `.focrq` preamble, decoded.
struct FocrqPreamble {
    /// Architecture-target byte from the preamble (the header may override it).
    arch_target: u8,
    /// Hex source sha256 from the preamble (the header may override it).
    source_sha256: String,
    /// Length of the header JSON that follows the preamble.
    header_len: usize,
}

/// Decode the fixed `.focrq` preamble from its first [`FOCRQ_PREAMBLE`] bytes
/// (magic included). Shared by the whole-blob loader and the wasm prefix
/// planner so the two can never disagree about where the header ends.
fn parse_focrq_preamble(bytes: &[u8]) -> FocrResult<FocrqPreamble> {
    if bytes.len() < FOCRQ_PREAMBLE {
        return Err(FocrError::FormatMismatch(format!(
            ".focrq truncated: {} bytes < {FOCRQ_PREAMBLE}-byte preamble",
            bytes.len()
        )));
    }
    if &bytes[..FOCRQ_MAGIC.len()] != FOCRQ_MAGIC {
        return Err(FocrError::FormatMismatch(
            ".focrq magic mismatch".to_owned(),
        ));
    }
    let mut cur = FOCRQ_MAGIC.len();
    let version = read_u32_le(&bytes[cur..cur + 4], ".focrq format_version")?;
    cur += 4;
    if version > FOCRQ_FORMAT_VERSION {
        return Err(FocrError::FormatMismatch(format!(
            ".focrq format_version {version} is newer than this binary supports \
             (max {FOCRQ_FORMAT_VERSION})"
        )));
    }
    let arch_target = bytes[cur];
    cur += 1;
    let source_sha256 = hex_encode(&bytes[cur..cur + 32]);
    cur += 32;
    let header_len = read_u64_len_le(&bytes[cur..cur + 8], ".focrq header_len")?;
    Ok(FocrqPreamble {
        arch_target,
        source_sha256,
        header_len,
    })
}

/// Default maximum planned segment size: 1 GiB. Rust caps a single allocation
/// (and any slice) at `isize::MAX` = 2 GiB on wasm32; 1 GiB leaves the
/// allocator room to place several segments inside the 4 GiB heap without a
/// near-limit request ever being attempted.
pub const MAX_BLOB_SEGMENT: u64 = 1 << 30;

/// A single tensor larger than this cannot be planned into a segment — it
/// would need one allocation above the wasm32 `isize::MAX` slice ceiling.
const MAX_SINGLE_SEGMENT: u64 = 3 << 29; // 1.5 GiB

/// What [`focrq_segment_plan`] concluded from a `.focrq` prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentPlan {
    /// The prefix contained the whole header; these are the planned segment
    /// byte LENGTHS in order (they sum to the artifact size, and every segment
    /// edge falls on a tensor boundary).
    Planned {
        /// Per-segment byte lengths, in order from offset 0.
        segment_lens: Vec<u64>,
        /// Absolute offset where the payload begins (end of the header JSON);
        /// segment 0 always contains the whole preamble + header.
        payload_base: u64,
    },
    /// The prefix was too short to contain preamble + header: refetch at least
    /// `need_bytes` and call again.
    NeedPrefix {
        /// Minimum prefix length that will contain the complete header.
        need_bytes: u64,
    },
}

/// Plan how to cut a `.focrq` artifact of `total_bytes` into segments of at
/// most `max_segment` bytes, cutting ONLY at tensor boundaries (bd-syf2).
///
/// `prefix` must be the artifact's leading bytes; only the preamble + header
/// JSON are read from it. Segment 0 starts at absolute 0 and always contains
/// the preamble + header (plus as many whole tensors as fit); every later
/// segment starts exactly where the previous ended, so the segments cover
/// `[0, total_bytes)` with no gap and no overlap — the contract
/// [`Weights::from_segments`] validates independently.
///
/// A NON-`.focrq` blob (a raw safetensors shard) has no such directory here; it
/// is planned as ONE segment when it fits under `max_segment` and refused
/// otherwise, because there is no way to cut it safely without parsing its own
/// directory.
///
/// # Errors
/// [`FocrError::FormatMismatch`] if the header is malformed, if a single tensor
/// is too large to be one segment, or if the tensor layout has no clean cut
/// (overlapping/interleaved extents) that keeps every tensor contiguous.
pub fn focrq_segment_plan(
    prefix: &[u8],
    total_bytes: u64,
    max_segment: u64,
) -> FocrResult<SegmentPlan> {
    if max_segment == 0 {
        return Err(FocrError::FormatMismatch(
            "segment plan: max_segment must be non-zero".to_owned(),
        ));
    }
    if total_bytes == 0 {
        return Err(FocrError::FormatMismatch(
            "segment plan: total_bytes must be non-zero".to_owned(),
        ));
    }
    // Not a `.focrq`: no tensor directory to cut on. One segment or refuse.
    if prefix.len() < FOCRQ_MAGIC.len() || &prefix[..FOCRQ_MAGIC.len()] != FOCRQ_MAGIC {
        if total_bytes <= max_segment.max(MAX_SINGLE_SEGMENT) {
            return Ok(SegmentPlan::Planned {
                segment_lens: vec![total_bytes],
                payload_base: 0,
            });
        }
        return Err(FocrError::FormatMismatch(format!(
            "segment plan: a non-.focrq blob of {total_bytes} bytes cannot be segmented \
             (no tensor directory to cut on)"
        )));
    }
    if prefix.len() < FOCRQ_PREAMBLE {
        return Ok(SegmentPlan::NeedPrefix {
            need_bytes: FOCRQ_PREAMBLE as u64,
        });
    }
    let preamble = parse_focrq_preamble(prefix)?;
    let header_end = FOCRQ_PREAMBLE
        .checked_add(preamble.header_len)
        .ok_or_else(|| FocrError::FormatMismatch(".focrq header_len overflows".into()))?;
    if prefix.len() < header_end {
        return Ok(SegmentPlan::NeedPrefix {
            need_bytes: header_end as u64,
        });
    }
    let header: FocrqHeader = serde_json::from_slice(&prefix[FOCRQ_PREAMBLE..header_end])
        .map_err(|e| FocrError::FormatMismatch(format!(".focrq header JSON invalid: {e}")))?;
    let payload_base = header_end as u64;
    if payload_base > total_bytes {
        return Err(FocrError::FormatMismatch(format!(
            "segment plan: header ends at {payload_base} past the artifact end ({total_bytes})"
        )));
    }

    // Absolute [start, end) extent of every tensor (payload AND inline scales),
    // sorted by start. A cut is legal only at a position no extent spans.
    let mut extents: Vec<(u64, u64)> = Vec::with_capacity(header.tensors.len());
    for (name, rec) in &header.tensors {
        let mut lo = u64::MAX;
        let mut hi = 0u64;
        for (off, len) in [
            (rec.byte_offset, rec.byte_len),
            (rec.scales_offset, rec.scales_len),
        ] {
            if len == 0 {
                continue;
            }
            let start = payload_base + off as u64;
            let end = start.checked_add(len as u64).ok_or_else(|| {
                FocrError::FormatMismatch(format!("tensor {name:?} byte range overflows"))
            })?;
            lo = lo.min(start);
            hi = hi.max(end);
        }
        if hi == 0 {
            continue; // zero-length record: nothing to place
        }
        if hi - lo > MAX_SINGLE_SEGMENT {
            return Err(FocrError::FormatMismatch(format!(
                "segment plan: tensor {name:?} spans {} bytes, above the {MAX_SINGLE_SEGMENT}-byte \
                 single-segment ceiling",
                hi - lo
            )));
        }
        if hi > total_bytes {
            return Err(FocrError::FormatMismatch(format!(
                "segment plan: tensor {name:?} ends at {hi} past the artifact end ({total_bytes})"
            )));
        }
        extents.push((lo, hi));
    }
    extents.sort_unstable();

    let mut segment_lens: Vec<u64> = Vec::new();
    let mut seg_start = 0u64; // segment 0 owns preamble + header
    let mut cut_at = payload_base; // rightmost legal cut seen so far
    for (lo, hi) in extents {
        // `cut_at` is the running max end of everything placed so far. It is a
        // LEGAL cut only when this tensor starts at or after it (otherwise the
        // extents interleave and cutting there would split this tensor).
        // Close the segment when adding this tensor would overflow the budget
        // and a legal cut exists — never mid-tensor, never a copy.
        if hi - seg_start > max_segment && cut_at > seg_start && lo >= cut_at {
            segment_lens.push(cut_at - seg_start);
            seg_start = cut_at;
        }
        cut_at = cut_at.max(hi);
    }
    // Trailing bytes (padding after the last tensor) join the final segment.
    if total_bytes - seg_start > MAX_SINGLE_SEGMENT {
        return Err(FocrError::FormatMismatch(format!(
            "segment plan: final segment would be {} bytes, above the {MAX_SINGLE_SEGMENT}-byte \
             ceiling (tensor layout offers no earlier clean cut)",
            total_bytes - seg_start
        )));
    }
    segment_lens.push(total_bytes - seg_start);
    Ok(SegmentPlan::Planned {
        segment_lens,
        payload_base,
    })
}

/// A zero-copy, owning view of a byte range inside a model blob: an
/// [`Arc`](std::sync::Arc) on the whole [`Backing`] plus a validated range.
/// This is how quantized weight structs reference their payload bytes WITHOUT
/// duplicating them next to the resident artifact (bd-4l71 residency work):
/// cloning is an `Arc` bump, and the view keeps the blob alive on its own.
/// The range is RESOLVED at construction (segment index + offset inside that
/// segment), so the `Deref` below is an infallible index — a segmented blob
/// cannot make a live view dangle or straddle.
#[derive(Clone)]
pub struct SharedBytes {
    backing: std::sync::Arc<Backing>,
    segment: usize,
    off: usize,
    len: usize,
}

impl SharedBytes {
    /// Resolve `[start, start + len)` inside `backing`, or `None` if it is out
    /// of bounds or crosses a segment boundary.
    fn new(backing: std::sync::Arc<Backing>, start: u64, len: usize) -> Option<Self> {
        let (segment, off) = backing.locate(start, len)?;
        Some(Self {
            backing,
            segment,
            off,
            len,
        })
    }
}

impl std::ops::Deref for SharedBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // Resolved and bounds-checked by `new`; the backing is immutable and
        // held alive by this handle's `Arc`.
        &self.backing.segment_bytes(self.segment)[self.off..self.off + self.len]
    }
}

impl std::fmt::Debug for SharedBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SharedBytes({} bytes at segment {} offset {} of {} blob bytes)",
            self.len,
            self.segment,
            self.off,
            self.backing.len()
        )
    }
}

impl PartialEq for SharedBytes {
    fn eq(&self, other: &Self) -> bool {
        self[..] == other[..]
    }
}

/// The ONE unsafe call in the loader — the read-only mmap island, mirroring
/// the `simd::arm`/`simd::x86` island pattern (crate policy is
/// `deny(unsafe_code)` with documented, minimal islands).
#[cfg(feature = "mmap")]
#[allow(unsafe_code)]
mod mmap_island {
    /// Map `file` read-only.
    ///
    /// # Safety argument
    /// `Mmap::map` is `unsafe` because another process truncating the file
    /// mid-use can fault later accesses. Production does not enter this island
    /// by default. `FOCR_MMAP=1` is an explicit operator assertion that the
    /// already-open artifact inode is trusted and immutable for the process
    /// lifetime (for example, a read-only image layer). Mapping is read-only and
    /// callers retain the descriptor-backed mapping for its full use.
    pub(super) fn map_readonly(file: &std::fs::File) -> std::io::Result<memmap2::Mmap> {
        // SAFETY: the explicit opt-in asserts an immutable artifact inode; see
        // the function doc for the complete deployment envelope.
        let map = unsafe { memmap2::Mmap::map(file) }?;
        // Tell the kernel not to read ahead. The access pattern over a MoE
        // artifact is emphatically NOT sequential: a token routes to 6 of 64
        // experts per layer, so touching one expert's rows under the default
        // sequential heuristic drags in neighboring experts this token will
        // never read — inflating resident pages and stealing IO bandwidth. On a
        // memory-constrained device that is the difference between a working set
        // that fits and one that thrashes. Advice is a hint: a failure here
        // costs paging efficiency, never correctness, so it is not fatal.
        //
        // `madvise` is POSIX; `memmap2` gates both `Advice` and `advise` behind
        // `cfg(unix)`, so this must be gated too or the Windows targets in the
        // release matrix fail to compile.
        #[cfg(unix)]
        let _ = map.advise(memmap2::Advice::Random);
        Ok(map)
    }
}

/// Whether to enter the mmap island for this load.
///
/// `FOCR_MMAP` is the explicit operator switch everywhere, and its absence means
/// "owned bytes" on every desktop and server target — the safe default, because
/// a concurrent writer truncating the artifact's inode faults the process, and a
/// path/rename convention is not an enforceable immutability mechanism
/// (`docs/NEGATIVE_EVIDENCE.md`, mmap-as-default REJECTED).
///
/// **iOS inverts that default, because iOS supplies the mechanism that was
/// missing.** The artifact lives inside the app's own container, which the
/// sandbox makes unreachable to every other process on the device; nothing but
/// this app can open it for writing, and this app never rewrites an installed
/// artifact in place (the downloader stages to a temp file and renames, which
/// leaves the mapped inode untouched). That is the enforceable immutability the
/// rejection asked for, so the trade flips: mapping turns a 3 GB blob from dirty
/// anonymous heap — which is what jetsam counts and kills for — into clean
/// file-backed pages the kernel may evict and re-read at will. On a phone that
/// is not a throughput tweak, it is the difference between running and being
/// terminated. `FOCR_MMAP=0` still forces the owned path back for diagnostics.
pub(super) fn mmap_requested() -> bool {
    match std::env::var("FOCR_MMAP").ok() {
        Some(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "on" | "true" | "yes"
        ),
        None => cfg!(target_os = "ios"),
    }
}

impl Default for Weights {
    /// An empty weight set (no tensors). Used by forward-module tests to
    /// construct a `Weights` without a real blob: every accessor on it returns
    /// [`FocrError::FormatMismatch`] (tensor not found). Modules that are still
    /// unwired return `NotImplemented` before touching weights; modules whose
    /// loader handoff has landed look tensors up by name and the empty default
    /// cleanly errors rather than panicking.
    fn default() -> Self {
        Self {
            bytes: std::sync::Arc::new(Backing::Owned(Vec::new())),
            payload_base: 0,
            directory: BTreeMap::new(),
            arch_target: 0,
            source_sha256: String::new(),
            license_notice: String::new(),
            model_id: model_arch::default_arch().id(),
            is_focrq: false,
            quant_recipe: None,
        }
    }
}

impl Weights {
    /// Load a `.focrq` blob, or fall back to a raw safetensors shard, from
    /// `path`.
    ///
    /// The safe default reads an owned `Vec<u8>`. `FOCR_MMAP=1` attempts a
    /// read-only mmap for a trusted immutable artifact; any mmap failure still
    /// falls back to owned bytes. The magic decides the
    /// format (`b"FOCRQ\0"` → `.focrq`, the safetensors `u64`
    /// header-length prefix otherwise). The census is NOT run here (the caller
    /// supplies the expected name set via [`Weights::load_with_census`]); a
    /// bare `load` just parses and indexes.
    ///
    /// # Errors
    /// * [`FocrError::ModelNotFound`] if the file can't be read.
    /// * [`FocrError::FormatMismatch`] on a bad magic, an unreadable header, a
    ///   version newer than this binary, or a directory that overruns the
    ///   payload.
    pub fn load(path: &Path) -> FocrResult<Self> {
        let file = std::fs::File::open(path).map_err(|e| {
            FocrError::ModelNotFound(format!("cannot read weights at {}: {e}", path.display()))
        })?;
        Self::load_opened(file, path)
    }

    /// Load from an already-open model file without consulting `path` again.
    ///
    /// Production model loading uses this after validating the same descriptor's
    /// bounded header. Keeping the descriptor pinned across validation and mmap /
    /// owned fallback prevents a path or symlink swap from substituting different
    /// bytes between those two steps.
    pub(super) fn load_opened(file: std::fs::File, path: &Path) -> FocrResult<Self> {
        Self::load_opened_with_mmap_policy(file, path, mmap_requested())
    }

    fn load_opened_with_mmap_policy(
        mut file: std::fs::File,
        path: &Path,
        mmap_requested: bool,
    ) -> FocrResult<Self> {
        #[cfg(feature = "mmap")]
        if mmap_requested && let Ok(map) = mmap_island::map_readonly(&file) {
            return Self::from_backing(Backing::Mapped(map));
        }
        #[cfg(not(feature = "mmap"))]
        let _ = mmap_requested; // no mmap island in the core build

        // Header validation advances the descriptor offset. Rewind this SAME
        // open file before the owned fallback; reopening `path` here would
        // reintroduce the validation/load TOCTOU that this API exists to close.
        file.seek(SeekFrom::Start(0)).map_err(|e| {
            FocrError::ModelNotFound(format!("cannot read weights at {}: {e}", path.display()))
        })?;
        let capacity = file
            .metadata()
            .ok()
            .and_then(|metadata| usize::try_from(metadata.len()).ok())
            .unwrap_or(0);
        // EXACT-SIZE read (bd-4l71 residency). `read_to_end` into a
        // `with_capacity(len)` buffer still probes for EOF once the buffer is
        // full, and that probe `reserve()`s past capacity — amortized growth
        // then asks the allocator for a SECOND, twice-as-large buffer for a
        // multi-GB artifact (fatal on wasm32, where the linear memory would
        // have to grow to ~2x the model). Sizing the buffer to the file and
        // `read_exact`-ing into it asks for the blob's bytes exactly once. A
        // zero/unknown size (an unseekable or racing file) keeps the historical
        // `read_to_end` path, and a short/long file surfaces the same IO error
        // shape it always did.
        let bytes = if capacity > 0 {
            let mut bytes = vec![0u8; capacity];
            file.read_exact(&mut bytes).map_err(|e| {
                FocrError::ModelNotFound(format!("cannot read weights at {}: {e}", path.display()))
            })?;
            bytes
        } else {
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes).map_err(|e| {
                FocrError::ModelNotFound(format!("cannot read weights at {}: {e}", path.display()))
            })?;
            bytes
        };
        Self::from_backing(Backing::Owned(bytes))
    }

    /// Load and immediately run the WeightsManifest census against
    /// `expected_names` (PROPOSED_ARCHITECTURE.md §6.12, §7).
    ///
    /// This is the load-time guard the model package wires in: a wrong/stale
    /// checkpoint (missing or extra tensors) is rejected here, not later as
    /// garbage output.
    ///
    /// # Errors
    /// As [`Weights::load`], plus [`FocrError::FormatMismatch`] if the loaded
    /// tensor set does not equal `expected_names`.
    pub fn load_with_census<I, S>(path: &Path, expected_names: I) -> FocrResult<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let w = Self::load(path)?;
        w.census(expected_names)?;
        Ok(w)
    }

    /// Parse + index an already-read blob (the testable core of [`load`]).
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] on any structural problem.
    pub fn from_bytes(bytes: Vec<u8>) -> FocrResult<Self> {
        Self::from_backing(Backing::Owned(bytes))
    }

    /// Parse + index a blob that arrived as SEGMENTS — the wasm32 path
    /// (bd-syf2), where no single `Vec<u8>` may exceed `isize::MAX` = 2 GiB so
    /// a 3 GB artifact must live in several buffers.
    ///
    /// `segments` is `(absolute start, bytes)` and must be ordered,
    /// non-overlapping, and exactly cover `[0, total)` starting at 0. Segment
    /// edges must additionally fall on TENSOR boundaries: after the directory
    /// parses, every tensor's payload and inline scales must resolve inside ONE
    /// segment or the load is refused (a mid-tensor split is a planning bug —
    /// see [`focrq_segment_plan`]). Everything downstream is identical to
    /// [`Weights::from_bytes`]: same header validation, same accessors, same
    /// bytes.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] on a gap/overlap/ordering violation, on any
    /// structural problem the owned path would reject, or if a tensor straddles
    /// a segment boundary.
    pub fn from_segments(segments: Vec<(u64, Vec<u8>)>) -> FocrResult<Self> {
        let mut expected_start = 0u64;
        let mut built: Vec<BlobSegment> = Vec::with_capacity(segments.len());
        for (index, (start, bytes)) in segments.into_iter().enumerate() {
            if start != expected_start {
                return Err(FocrError::FormatMismatch(format!(
                    "segmented weights: segment {index} starts at {start}, expected \
                     {expected_start} (segments must be ordered and contiguous from 0)"
                )));
            }
            if bytes.is_empty() {
                return Err(FocrError::FormatMismatch(format!(
                    "segmented weights: segment {index} is empty"
                )));
            }
            expected_start = start.checked_add(bytes.len() as u64).ok_or_else(|| {
                FocrError::FormatMismatch(
                    "segmented weights: total length overflows u64".to_owned(),
                )
            })?;
            built.push(BlobSegment { start, bytes });
        }
        if built.is_empty() {
            return Err(FocrError::FormatMismatch(
                "segmented weights: no segments".to_owned(),
            ));
        }
        Self::from_backing(Backing::Segmented {
            segments: built,
            len: expected_start,
        })
    }

    /// Test-only probe: did this load take the mmap backing?
    #[cfg(test)]
    pub(super) fn is_mapped(&self) -> bool {
        matches!(*self.bytes, Backing::Mapped(_))
    }

    /// Parse + index a blob in either backing (the shared core of
    /// [`Weights::load`] / [`Weights::from_bytes`]).
    fn from_backing(bytes: Backing) -> FocrResult<Self> {
        let bytes = std::sync::Arc::new(bytes);
        let magic = bytes.range(0, FOCRQ_MAGIC.len());
        if magic == Some(FOCRQ_MAGIC) {
            Self::from_focrq_bytes(bytes)
        } else {
            Self::from_safetensors_bytes(bytes)
        }
    }

    /// Parse a `.focrq` blob (magic already confirmed by the caller).
    ///
    /// Layout: `magic[6] | format_version u32 LE | arch_target u8 |
    /// source_sha256[32] | header_len u64 LE | header_json[header_len] |
    /// payload`. The provenance/config and the tensor directory all live in the
    /// header JSON (the byte fields before it are the fixed-size preamble that a
    /// reader can validate before parsing JSON).
    fn from_focrq_bytes(bytes: std::sync::Arc<Backing>) -> FocrResult<Self> {
        let preamble_bytes = bytes.range(0, FOCRQ_PREAMBLE).ok_or_else(|| {
            FocrError::FormatMismatch(format!(
                ".focrq truncated: {} bytes < {FOCRQ_PREAMBLE}-byte preamble",
                bytes.len()
            ))
        })?;
        let preamble = parse_focrq_preamble(preamble_bytes)?;
        let header_end = FOCRQ_PREAMBLE
            .checked_add(preamble.header_len)
            .ok_or_else(|| FocrError::FormatMismatch(".focrq header_len overflows".into()))?;
        let header_json = bytes
            .range(FOCRQ_PREAMBLE as u64, preamble.header_len)
            .ok_or_else(|| {
                FocrError::FormatMismatch(format!(
                    ".focrq header ({} bytes) overruns file ({} bytes) or crosses a segment edge",
                    preamble.header_len,
                    bytes.len()
                ))
            })?;
        let header: FocrqHeader = serde_json::from_slice(header_json)
            .map_err(|e| FocrError::FormatMismatch(format!(".focrq header JSON invalid: {e}")))?;
        let (preamble_arch, preamble_sha) = (preamble.arch_target, preamble.source_sha256);
        // Resolve the declared architecture FIRST (absent ⇒ v1 default unlimited-ocr;
        // unknown non-empty id ⇒ loud refusal) so the license check below can demand
        // the right notice for that arch (Baidu/MIT vs e.g. GOT-OCR2 Apache-2.0).
        let model_id = resolve_model_id(&header.model_id)?;
        validate_license_notice(&header.license_notice, model_id)?;
        if !header.source_sha256.is_empty() {
            validate_source_sha256_hex(&header.source_sha256)?;
        }

        let payload_base = header_end;
        let payload_len = payload_len_of(&bytes, payload_base)?;

        // Prefer the header's own fields; the preamble bytes are a cheap
        // pre-JSON sanity surface (and the source of truth if the header omits
        // them).
        let arch_target = if header.arch_target != 0 {
            header.arch_target
        } else {
            preamble_arch
        };
        validate_arch_target(arch_target)?;
        validate_directory(&header.tensors, payload_len, arch_target)?;
        validate_segment_containment(&bytes, payload_base, &header.tensors)?;
        let source_sha256 = if header.source_sha256.is_empty() {
            preamble_sha
        } else {
            header.source_sha256
        };

        Ok(Self {
            bytes,
            payload_base,
            directory: header.tensors,
            arch_target,
            source_sha256,
            license_notice: header.license_notice,
            model_id,
            is_focrq: true,
            quant_recipe: header
                .packing_manifest
                .and_then(|manifest| manifest.quant_recipe),
        })
    }

    /// Parse a raw safetensors blob: `header_len u64 LE | header_json |
    /// payload`. The JSON maps `name -> {dtype, shape, data_offsets:[beg,end]}`
    /// (offsets are payload-relative); a `__metadata__` key, if present, is
    /// skipped.
    fn from_safetensors_bytes(bytes: std::sync::Arc<Backing>) -> FocrResult<Self> {
        let len_prefix = bytes.range(0, 8).ok_or_else(|| {
            FocrError::FormatMismatch(format!(
                "safetensors truncated: {} bytes < 8-byte header length prefix",
                bytes.len()
            ))
        })?;
        let header_len = read_u64_len_le(len_prefix, "safetensors header_len")?;
        let header_end = 8usize
            .checked_add(header_len)
            .ok_or_else(|| FocrError::FormatMismatch("safetensors header_len overflows".into()))?;
        let header_json = bytes.range(8, header_len).ok_or_else(|| {
            FocrError::FormatMismatch(format!(
                "safetensors header ({header_len} bytes) overruns file ({} bytes) or crosses a \
                 segment edge",
                bytes.len()
            ))
        })?;

        #[derive(Deserialize)]
        struct StEntry {
            dtype: String,
            shape: Vec<usize>,
            data_offsets: [usize; 2],
        }

        // The header is a flat object of name -> entry, with one reserved
        // `__metadata__` string-map key. Parse to `serde_json::Value` so we can
        // skip the reserved key without a custom visitor.
        let raw: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(header_json)
            .map_err(|e| {
            FocrError::FormatMismatch(format!("safetensors header JSON invalid: {e}"))
        })?;

        let mut directory = BTreeMap::new();
        for (name, value) in raw {
            if name == "__metadata__" {
                continue;
            }
            let entry: StEntry = serde_json::from_value(value).map_err(|e| {
                FocrError::FormatMismatch(format!("safetensors entry {name:?} invalid: {e}"))
            })?;
            let [beg, end] = entry.data_offsets;
            if end < beg {
                return Err(FocrError::FormatMismatch(format!(
                    "safetensors entry {name:?} has end {end} < beg {beg}"
                )));
            }
            directory.insert(
                name,
                TensorRecord {
                    dtype: DType::from_safetensors_str(&entry.dtype)?,
                    shape: entry.shape,
                    byte_offset: beg,
                    byte_len: end - beg,
                    scales_offset: 0,
                    scales_len: 0,
                    group_size: 0,
                    tier: 0,
                },
            );
        }

        let payload_base = header_end;
        let payload_len = payload_len_of(&bytes, payload_base)?;
        validate_directory(&directory, payload_len, 0)?;
        validate_segment_containment(&bytes, payload_base, &directory)?;

        Ok(Self {
            bytes,
            payload_base,
            directory,
            arch_target: 0,
            source_sha256: String::new(),
            license_notice: String::new(),
            // A raw safetensors shard is the upstream Unlimited-OCR checkpoint.
            model_id: model_arch::default_arch().id(),
            is_focrq: false,
            quant_recipe: None,
        })
    }

    /// Run the WeightsManifest census: the loaded tensor name set MUST equal
    /// `expected_names` exactly (PROPOSED_ARCHITECTURE.md §6.12, §7).
    ///
    /// Reports up to a handful of missing and unexpected names so a stale
    /// checkpoint is diagnosable, not a silent count mismatch.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if any expected tensor is missing or any
    /// loaded tensor is unexpected.
    pub fn census<I, S>(&self, expected_names: I) -> FocrResult<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let expected: std::collections::BTreeSet<String> = expected_names
            .into_iter()
            .map(|s| s.as_ref().to_owned())
            .collect();

        let missing: Vec<&str> = expected
            .iter()
            .filter(|n| !self.directory.contains_key(n.as_str()))
            .map(String::as_str)
            .collect();
        let unexpected: Vec<&str> = self
            .directory
            .keys()
            .filter(|n| !expected.contains(n.as_str()))
            .map(String::as_str)
            .collect();

        if missing.is_empty() && unexpected.is_empty() {
            return Ok(());
        }

        Err(FocrError::FormatMismatch(format!(
            "weights census failed: expected {} tensors, found {} \
             (missing {}: {}; unexpected {}: {})",
            expected.len(),
            self.directory.len(),
            missing.len(),
            preview(&missing),
            unexpected.len(),
            preview(&unexpected),
        )))
    }

    /// Number of tensors in the directory.
    #[must_use]
    pub fn len(&self) -> usize {
        self.directory.len()
    }

    /// Whether the directory is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.directory.is_empty()
    }

    /// Whether this was loaded from a `.focrq` container (vs. a safetensors
    /// shard).
    #[must_use]
    pub fn is_focrq(&self) -> bool {
        self.is_focrq
    }

    /// The `.focrq` arch-target packing byte (`0` for safetensors).
    #[must_use]
    pub fn arch_target(&self) -> u8 {
        self.arch_target
    }

    /// The hex source-safetensors sha256 (`""` for safetensors).
    #[must_use]
    pub fn source_sha256(&self) -> &str {
        &self.source_sha256
    }

    /// The quant-recipe id this artifact's `packing_manifest` declares, if any
    /// (`None` for safetensors and recipe-less `.focrq`). The loader and the
    /// decoder cache builders branch on this to honor the explicitly-tagged
    /// non-default wasm experts-int4 recipe (bd-4l71).
    #[must_use]
    pub fn quant_recipe(&self) -> Option<&str> {
        self.quant_recipe.as_deref()
    }

    /// The `.focrq` license notice (MIT/Baidu; `""` for safetensors). The
    /// container spec requires this be present in a real artifact ([SPEC §7]).
    #[must_use]
    pub fn license_notice(&self) -> &str {
        &self.license_notice
    }

    /// The resolved model-architecture id ([`model_arch::ModelArch::id`]) this
    /// artifact declares — e.g. `"unlimited-ocr"` or `"got-ocr2"`. A v1 `.focrq` (no
    /// `model_id` header) and a raw safetensors shard both report the default
    /// `"unlimited-ocr"`. Always a registry-known id (validated at load), so the
    /// engine can look the arch up with [`model_arch::arch_by_id`] infallibly.
    #[must_use]
    pub fn model_id(&self) -> &'static str {
        self.model_id
    }

    /// Iterate the tensor names (sorted, since the directory is a `BTreeMap`).
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.directory.keys().map(String::as_str)
    }

    /// Whether a tensor with this name exists.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.directory.contains_key(name)
    }

    /// Look up the directory record for `name` (dtype/shape/offsets), without
    /// resolving its bytes.
    #[must_use]
    pub fn record(&self, name: &str) -> Option<&TensorRecord> {
        self.directory.get(name)
    }

    /// A zero-copy [`TensorView`] (dtype + shape + payload slice) for `name` —
    /// the accessor the vision/decoder forward modules consume.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if `name` is absent or its byte range
    /// falls outside the payload.
    pub fn tensor(&self, name: &str) -> FocrResult<TensorView<'_>> {
        let rec = self.directory.get(name).ok_or_else(|| {
            FocrError::FormatMismatch(format!("tensor {name:?} not found in weights directory"))
        })?;
        let data = self.payload_slice(name, rec.byte_offset, rec.byte_len)?;
        let scales: &[u8] = if rec.scales_len == 0 {
            &[]
        } else {
            self.payload_slice(name, rec.scales_offset, rec.scales_len)?
        };
        Ok(TensorView {
            dtype: rec.dtype,
            shape: &rec.shape,
            data,
            scales,
            group_size: rec.group_size,
            tier: rec.tier,
        })
    }

    /// A fully widened f32 [`Mat`] for a 2-D tensor `name` (any of F32/F16/BF16).
    ///
    /// The crown high-precision accessor: every BF16 weight is widened to the
    /// f32 activation rail here. A 1-D tensor is treated as `[1, len]` so the
    /// connector params (`image_newline`, `view_seperator`) and norms load
    /// through the same path.
    ///
    /// # Errors
    /// * [`FocrError::FormatMismatch`] if `name` is absent, the tensor is
    ///   quantized (use [`Weights::qint8`] / [`Weights::qint4`]), or its rank is
    ///   not 1 or 2.
    pub fn mat(&self, name: &str) -> FocrResult<Mat> {
        let view = self.tensor(name)?;
        let (rows, cols) = match view.shape.len() {
            1 => (1usize, view.shape[0]),
            2 => (view.shape[0], view.shape[1]),
            n => {
                return Err(FocrError::FormatMismatch(format!(
                    "tensor {name:?} has rank {n}; mat() needs a 1-D or 2-D tensor"
                )));
            }
        };
        if view.dtype == DType::QInt8PerChan {
            // Transparent dequant-on-access (bd-av64.12): an int8-stored
            // GEMM read through the f32 accessor reconstructs
            // `w[o][j] = qw[o][j] * scale[o]` — the DEFINED meaning of the
            // record. `qint8()` already un-permutes any offline packing, so
            // row-major holds here. Consumers that want int8 COMPUTE keep
            // calling `qint8()` directly; this arm only makes f32 engines
            // (TrOMR's Seq2SeqDense forward) able to run quantized-storage
            // artifacts.
            let q = self.qint8(name)?;
            return Ok(Mat::from_vec(q.n, q.k, dequant_qint8(&q)));
        }
        let data = view
            .to_f32_vec()
            .map_err(|e| FocrError::FormatMismatch(format!("tensor {name:?}: {e}")))?;
        let expected_len = checked_shape_len(name, rows, cols, "rows*cols")?;
        if data.len() != expected_len {
            return Err(FocrError::FormatMismatch(format!(
                "tensor {name:?} element count {} != rows*cols {}",
                data.len(),
                expected_len
            )));
        }
        Ok(Mat::from_vec(rows, cols, data))
    }

    /// Widen a tensor `name` to a flat f32 `Vec` (any rank, any of F32/F16/BF16)
    /// — the accessor for 1-D params (norms, biases, the connector
    /// `image_newline` / `view_seperator`) where a [`Mat`] shape is not wanted.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if `name` is absent or the tensor is
    /// quantized.
    pub fn vec(&self, name: &str) -> FocrResult<Vec<f32>> {
        let view = self.tensor(name)?;
        if view.dtype == DType::QInt8PerChan {
            // See `mat()`: transparent per-channel dequant of int8 records.
            let q = self.qint8(name)?;
            return Ok(dequant_qint8(&q));
        }
        view.to_f32_vec()
            .map_err(|e| FocrError::FormatMismatch(format!("tensor {name:?}: {e}")))
    }

    /// Reconstruct a symmetric per-output-channel [`QInt8`] for `name`.
    ///
    /// The directory record must be `QInt8PerChan` with a 2-D `[n, k]` shape;
    /// `n * k` int8 payload bytes + `n` f32 inline scales.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if `name` is absent, not `QInt8PerChan`,
    /// not rank-2, or its byte/scale lengths disagree with the shape.
    pub fn qint8(&self, name: &str) -> FocrResult<QInt8> {
        let view = self.tensor(name)?;
        if view.dtype != DType::QInt8PerChan {
            return Err(FocrError::FormatMismatch(format!(
                "tensor {name:?} is {:?}, not QInt8PerChan",
                view.dtype
            )));
        }
        if view.shape.len() != 2 {
            return Err(FocrError::FormatMismatch(format!(
                "QInt8 tensor {name:?} has rank {}; expected 2 ([n, k])",
                view.shape.len()
            )));
        }
        let (n, k) = (view.shape[0], view.shape[1]);
        let expected_len = if self.arch_target == 1 {
            crate::simd::pack::smmla_packed_len(n, k)
        } else {
            checked_shape_len(name, n, k, "n*k")?
        };
        if view.data.len() != expected_len {
            return Err(FocrError::FormatMismatch(format!(
                "QInt8 tensor {name:?}: {} payload bytes != expected {} (arch_target {})",
                view.data.len(),
                expected_len,
                self.arch_target
            )));
        }
        // int8 payload is a direct byte→i8 reinterpret (little-endian-agnostic;
        // one byte per element).
        //
        // An `--arch aarch64-smmla` artifact (arch_target 1, bd-2mo.3) stores
        // SMMLA panels. When the host actually dispatches the SMMLA tier the
        // panels are kept AS-IS — the decode GEMV hands them to `vmmlaq_s32`
        // with zero runtime shuffle (the whole point of the offline packing).
        // Any other tier un-permutes back to canonical row-major here — a
        // one-time load cost, lossless by construction, so every consumer
        // keeps its row-major contract (degrade to generic, never UB).
        if self.arch_target == 1 && crate::simd::detected_tier() == crate::simd::IsaTier::Smmla {
            let packed: Vec<i8> = view.data.iter().map(|&b| b as i8).collect();
            let scales = decode_f32_le(view.scales)?;
            if scales.len() != n {
                return Err(FocrError::FormatMismatch(format!(
                    "QInt8 tensor {name:?}: {} scales != n {}",
                    scales.len(),
                    n
                )));
            }
            return Ok(QInt8::new_smmla_panels(packed, scales, n, k));
        }
        if self.arch_target == 1 {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                crate::progress::stderr_message(format_args!(
                    "[focr] arch mismatch: .focrq is packed for aarch64-smmla but this \
                     host dispatches {}; un-permuting to the generic layout at load \
                     (correct, but the offline packing buys nothing here)",
                    crate::simd::tier_string()
                ));
            });
            let packed: Vec<i8> = view.data.iter().map(|&b| b as i8).collect();
            let w = crate::simd::pack::smmla_unpack_panels(&packed, n, k)
                .map_err(|e| FocrError::FormatMismatch(format!("QInt8 tensor {name:?}: {e}")))?;
            let scales = decode_f32_le(view.scales)?;
            if scales.len() != n {
                return Err(FocrError::FormatMismatch(format!(
                    "QInt8 tensor {name:?}: {} scales != n {}",
                    scales.len(),
                    n
                )));
            }
            return Ok(QInt8::new(w, scales, n, k));
        }
        // Canonical row-major int8: the payload bytes ARE the weight image, so
        // reference them ZERO-COPY out of the blob (bd-4l71 residency — the
        // wasm recipe's attention + `lm_head` int8 set is ~244 MB that must not
        // exist twice). Values are identical to the historical
        // `view.data.iter().map(|&b| b as i8)` copy.
        let scales = decode_f32_le(view.scales)?;
        if scales.len() != n {
            return Err(FocrError::FormatMismatch(format!(
                "QInt8 tensor {name:?}: {} scales != n {}",
                scales.len(),
                n
            )));
        }
        let record = self
            .directory
            .get(name)
            .expect("tensor() resolved this record above");
        let shared = self.payload_shared(name, record.byte_offset, record.byte_len)?;
        Ok(QInt8::new_shared(shared, scales, n, k))
    }

    /// Reconstruct a group-quantized [`QInt4`] for `name`.
    ///
    /// The directory record must be `QInt4PerGroup` with a 2-D `[n, k]` shape
    /// (`k` even), `n * k / 2` packed bytes, `group_size`, and
    /// `n * (k / group_size)` f32 inline scales.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] on a dtype/shape/length mismatch.
    pub fn qint4(&self, name: &str) -> FocrResult<QInt4> {
        let view = self.tensor(name)?;
        if view.dtype != DType::QInt4PerGroup {
            return Err(FocrError::FormatMismatch(format!(
                "tensor {name:?} is {:?}, not QInt4PerGroup",
                view.dtype
            )));
        }
        if view.shape.len() != 2 {
            return Err(FocrError::FormatMismatch(format!(
                "QInt4 tensor {name:?} has rank {}; expected 2 ([n, k])",
                view.shape.len()
            )));
        }
        let (n, k) = (view.shape[0], view.shape[1]);
        if k % 2 != 0 {
            return Err(FocrError::FormatMismatch(format!(
                "QInt4 tensor {name:?}: k {k} is not even (two nibbles per byte)"
            )));
        }
        if !VALID_GROUP_SIZES.contains(&view.group_size) {
            return Err(FocrError::FormatMismatch(format!(
                "QInt4 tensor {name:?}: group_size {} must be 16 or 32",
                view.group_size
            )));
        }
        if k % view.group_size != 0 {
            return Err(FocrError::FormatMismatch(format!(
                "QInt4 tensor {name:?}: group_size {} does not divide k {k}",
                view.group_size
            )));
        }
        let expected_packed = checked_shape_len(name, n, k / 2, "n*k/2")?;
        if view.data.len() != expected_packed {
            return Err(FocrError::FormatMismatch(format!(
                "QInt4 tensor {name:?}: {} packed bytes != n*k/2 {}",
                view.data.len(),
                expected_packed
            )));
        }
        let expected_scales = checked_shape_len(name, n, k / view.group_size, "n*(k/group_size)")?;
        if view.scales.len() != expected_scales * 4 {
            return Err(FocrError::FormatMismatch(format!(
                "QInt4 tensor {name:?}: {} scale bytes != n*(k/group_size)*f32 {}",
                view.scales.len(),
                expected_scales * 4
            )));
        }
        // ZERO-COPY residency (bd-4l71): the packed nibbles and the raw LE
        // scale bytes are referenced straight out of the loaded blob (Arc'd
        // range views) instead of being duplicated next to it — the wasm
        // artifact's ~1.8 GB expert payload must exist ONCE. Consumers decode
        // scale subranges per use (`GroupScales::slice_f32`, identical values).
        let record = self
            .directory
            .get(name)
            .expect("tensor() resolved this record above");
        let packed = self.payload_shared(name, record.byte_offset, record.byte_len)?;
        let scales = self.payload_shared(name, record.scales_offset, record.scales_len)?;
        Ok(QInt4 {
            packed: crate::native_engine::tensor::PackedBytes::Shared(packed),
            scales: crate::native_engine::tensor::GroupScales::RawLe(scales),
            n,
            k,
            group_size: view.group_size,
            tier: view.tier,
        })
    }

    /// Resolve a payload byte range as an OWNING zero-copy [`SharedBytes`]
    /// view (an `Arc` on the whole blob + the validated range) — the residency
    /// primitive quantized weight structs use to reference their payload
    /// without duplicating it (bd-4l71).
    fn payload_shared(&self, name: &str, off: usize, len: usize) -> FocrResult<SharedBytes> {
        let start = absolute_payload_start(name, self.payload_base, off)?;
        SharedBytes::new(std::sync::Arc::clone(&self.bytes), start, len)
            .ok_or_else(|| payload_range_error(name, off, len))
    }

    /// Resolve a payload byte range, bounds-checked against the backing blob
    /// (and, for a segmented blob, required to live inside ONE segment — the
    /// planner guarantees that, so a failure here is a corrupt plan, never a
    /// reason to stitch a copy).
    fn payload_slice(&self, name: &str, off: usize, len: usize) -> FocrResult<&[u8]> {
        let start = absolute_payload_start(name, self.payload_base, off)?;
        self.bytes
            .range(start, len)
            .ok_or_else(|| payload_range_error(name, off, len))
    }
}

/// Absolute blob offset of a payload-relative tensor offset.
fn absolute_payload_start(name: &str, payload_base: usize, off: usize) -> FocrResult<u64> {
    (payload_base as u64)
        .checked_add(off as u64)
        .ok_or_else(|| FocrError::FormatMismatch(format!("tensor {name:?} byte range overflows")))
}

/// The one error shape for a payload range that is out of bounds OR crosses a
/// segment edge (both mean "this blob cannot serve this tensor contiguously").
fn payload_range_error(name: &str, off: usize, len: usize) -> FocrError {
    let range_end = off
        .checked_add(len)
        .map_or_else(|| "<overflow>".to_owned(), |end| end.to_string());
    FocrError::FormatMismatch(format!(
        "tensor {name:?} range [{off}, {range_end}) overruns the payload or crosses a segment \
         boundary"
    ))
}

/// Payload length (blob length minus the header) as a `usize`.
fn payload_len_of(bytes: &Backing, payload_base: usize) -> FocrResult<usize> {
    let len = bytes
        .len()
        .checked_sub(payload_base as u64)
        .ok_or_else(|| FocrError::FormatMismatch("payload base past end of blob".to_owned()))?;
    usize::try_from(len).map_err(|_| {
        FocrError::FormatMismatch(format!(
            "payload of {len} bytes does not fit this target's usize"
        ))
    })
}

/// For a SEGMENTED blob (bd-syf2), prove up front that every tensor's payload
/// and inline scales resolve inside ONE segment. A segment plan that cut
/// mid-tensor is refused at load — naming the tensor — instead of failing much
/// later inside a forward. Owned/Mapped blobs are one segment, so this is a
/// no-op there (and the loop is skipped entirely).
fn validate_segment_containment(
    bytes: &Backing,
    payload_base: usize,
    directory: &BTreeMap<String, TensorRecord>,
) -> FocrResult<()> {
    if !matches!(bytes, Backing::Segmented { .. }) {
        return Ok(());
    }
    for (name, rec) in directory {
        for (off, len, what) in [
            (rec.byte_offset, rec.byte_len, "payload"),
            (rec.scales_offset, rec.scales_len, "scales"),
        ] {
            if len == 0 {
                continue;
            }
            let start = absolute_payload_start(name, payload_base, off)?;
            if bytes.range(start, len).is_none() {
                return Err(FocrError::FormatMismatch(format!(
                    "segmented weights: tensor {name:?} {what} [{off}, {}) crosses a segment \
                     boundary — segment edges must fall on tensor boundaries",
                    off.saturating_add(len)
                )));
            }
        }
    }
    Ok(())
}

/// Validate every directory record against the payload length + its own shape.
///
/// Catches an offset/len that overruns the payload, or a byte length that
/// disagrees with `shape × dtype` — both signals of a corrupt or mismatched
/// blob, surfaced as a load-time error.
fn validate_directory(
    directory: &BTreeMap<String, TensorRecord>,
    payload_len: usize,
    arch_target: u8,
) -> FocrResult<()> {
    for (name, rec) in directory {
        let end = rec.byte_offset.checked_add(rec.byte_len).ok_or_else(|| {
            FocrError::FormatMismatch(format!("tensor {name:?} byte range overflows"))
        })?;
        if end > payload_len {
            return Err(FocrError::FormatMismatch(format!(
                "tensor {name:?} ends at {end} but payload is {payload_len} bytes"
            )));
        }
        // `--arch aarch64-smmla` (arch_target 1, bd-2mo.3) stores int8 payloads
        // as SMMLA panels: `ceil(n/2)*ceil(k/8)*16` bytes (== n*k whenever the
        // shape tiles cleanly). Every other dtype is arch-independent.
        let expected =
            if arch_target == 1 && rec.dtype == DType::QInt8PerChan && rec.shape.len() == 2 {
                crate::simd::pack::smmla_packed_len(rec.shape[0], rec.shape[1])
            } else {
                rec.expected_byte_len(name)?
            };
        if rec.byte_len != expected {
            return Err(FocrError::FormatMismatch(format!(
                "tensor {name:?}: byte_len {} != shape×dtype {} ({:?}, shape {:?})",
                rec.byte_len, expected, rec.dtype, rec.shape
            )));
        }
        let scales_end = rec
            .scales_offset
            .checked_add(rec.scales_len)
            .ok_or_else(|| {
                FocrError::FormatMismatch(format!("tensor {name:?} scales range overflows"))
            })?;
        if scales_end > payload_len {
            return Err(FocrError::FormatMismatch(format!(
                "tensor {name:?} scales end at {scales_end} but payload is {payload_len} bytes"
            )));
        }
    }
    validate_non_overlapping_ranges(directory)?;
    Ok(())
}

/// Reject aliased tensor-data and quantization-scale intervals.
///
/// The v1 format has no `alias_of` field, so every non-empty payload range must
/// be disjoint. Keeping this check shared ensures the bounded production header
/// probe and the full mmap/owned parser enforce the same structural contract.
pub(super) fn validate_non_overlapping_ranges(
    directory: &BTreeMap<String, TensorRecord>,
) -> FocrResult<()> {
    let mut ranges = Vec::with_capacity(directory.len().saturating_mul(2));
    for (name, record) in directory {
        let data_end = record
            .byte_offset
            .checked_add(record.byte_len)
            .ok_or_else(|| {
                FocrError::FormatMismatch(format!("tensor {name:?} byte range overflows"))
            })?;
        if record.byte_len != 0 {
            ranges.push((record.byte_offset, data_end, name.as_str(), "data"));
        }

        let scales_end = record
            .scales_offset
            .checked_add(record.scales_len)
            .ok_or_else(|| {
                FocrError::FormatMismatch(format!("tensor {name:?} scales range overflows"))
            })?;
        if record.scales_len != 0 {
            ranges.push((record.scales_offset, scales_end, name.as_str(), "scales"));
        }
    }

    ranges.sort_unstable_by(|left, right| {
        (left.0, left.1, left.2, left.3).cmp(&(right.0, right.1, right.2, right.3))
    });
    for pair in ranges.windows(2) {
        let previous = pair[0];
        let current = pair[1];
        if current.0 < previous.1 {
            return Err(FocrError::FormatMismatch(format!(
                "payload ranges overlap: tensor {:?} {} [{}, {}) overlaps tensor {:?} {} [{}, {})",
                previous.2,
                previous.3,
                previous.0,
                previous.1,
                current.2,
                current.3,
                current.0,
                current.1,
            )));
        }
    }
    Ok(())
}

/// Widen a typed byte buffer to f32 for an F32/F16/BF16 dtype.
///
/// # Errors
/// [`FocrError::FormatMismatch`] for a quantized dtype or a length that is not a
/// whole number of elements.
/// Reconstruct the f32 weights a symmetric per-output-channel int8 record
/// denotes: `w[o][j] = qw[o][j] * scale[o]` (bd-av64.12 dequant-on-access).
fn dequant_qint8(q: &QInt8) -> Vec<f32> {
    let mut out = Vec::with_capacity(q.w.len());
    for (o, &scale) in q.scales.iter().enumerate() {
        out.extend(
            q.w[o * q.k..(o + 1) * q.k]
                .iter()
                .map(|&v| f32::from(v) * scale),
        );
    }
    out
}

fn decode_f32(dtype: DType, data: &[u8]) -> FocrResult<Vec<f32>> {
    match dtype {
        DType::F32 => decode_f32_le(data),
        DType::F16 => {
            if !data.len().is_multiple_of(2) {
                return Err(FocrError::FormatMismatch(format!(
                    "F16 byte len {} is not a multiple of 2",
                    data.len()
                )));
            }
            Ok(data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes(*c).to_f32())
                .collect())
        }
        DType::BF16 => {
            if !data.len().is_multiple_of(2) {
                return Err(FocrError::FormatMismatch(format!(
                    "BF16 byte len {} is not a multiple of 2",
                    data.len()
                )));
            }
            // half::bf16 -> f32 is exact (bf16 is the high 16 bits of f32);
            // PROPOSED_ARCHITECTURE.md §6.12: widen BF16→f32, never narrow.
            Ok(data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| bf16::from_le_bytes(*c).to_f32())
                .collect())
        }
        DType::QInt8PerChan | DType::QInt4PerGroup => Err(FocrError::FormatMismatch(format!(
            "decode_f32: {dtype:?} is quantized; use qint8()/qint4()"
        ))),
    }
}

/// Decode a little-endian f32 byte buffer.
fn decode_f32_le(data: &[u8]) -> FocrResult<Vec<f32>> {
    if !data.len().is_multiple_of(4) {
        return Err(FocrError::FormatMismatch(format!(
            "F32 byte len {} is not a multiple of 4",
            data.len()
        )));
    }
    Ok(data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn read_u32_le(bytes: &[u8], field: &str) -> FocrResult<u32> {
    let arr: [u8; 4] = bytes.try_into().map_err(|_| {
        FocrError::FormatMismatch(format!("{field} truncated: {} bytes < 4", bytes.len()))
    })?;
    Ok(u32::from_le_bytes(arr))
}

fn read_u64_len_le(bytes: &[u8], field: &str) -> FocrResult<usize> {
    let arr: [u8; 8] = bytes.try_into().map_err(|_| {
        FocrError::FormatMismatch(format!("{field} truncated: {} bytes < 8", bytes.len()))
    })?;
    let raw = u64::from_le_bytes(arr);
    usize::try_from(raw).map_err(|_| {
        FocrError::FormatMismatch(format!(
            "{field} {raw} exceeds this platform's addressable size"
        ))
    })
}

/// Lowercase-hex-encode a byte slice (for the source-sha256 provenance field).
fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Render the first few names of a census discrepancy list, with an ellipsis if
/// truncated.
fn preview(names: &[&str]) -> String {
    const MAX: usize = 6;
    if names.len() <= MAX {
        format!("[{}]", names.join(", "))
    } else {
        format!("[{}, …]", names[..MAX].join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-assemble a minimal `.focrq` blob: preamble + header JSON + payload.
    /// `payload` is the concatenated tensor bytes; the directory offsets are
    /// payload-relative.
    fn build_focrq(
        version: u32,
        arch: u8,
        sha: [u8; 32],
        directory_json: &str,
        payload: &[u8],
    ) -> Vec<u8> {
        build_focrq_with_license(
            version,
            arch,
            sha,
            directory_json,
            payload,
            FOCR_MODEL_LICENSE_NOTICE,
        )
    }

    fn build_focrq_with_license(
        version: u32,
        arch: u8,
        sha: [u8; 32],
        directory_json: &str,
        payload: &[u8],
        license_notice: &str,
    ) -> Vec<u8> {
        build_focrq_with_license_and_header_source_sha(
            version,
            arch,
            sha,
            directory_json,
            payload,
            license_notice,
            "",
        )
    }

    fn build_focrq_with_license_and_header_source_sha(
        version: u32,
        arch: u8,
        sha: [u8; 32],
        directory_json: &str,
        payload: &[u8],
        license_notice: &str,
        header_source_sha: &str,
    ) -> Vec<u8> {
        let license_json = format!(
            "\"{}\"",
            license_notice.replace('\\', "\\\\").replace('"', "\\\"")
        );
        let source_sha_json = format!(
            "\"{}\"",
            header_source_sha.replace('\\', "\\\\").replace('"', "\\\"")
        );
        let header = format!(
            "{{\"tensors\":{directory_json},\"arch_target\":{arch},\
             \"source_sha256\":{source_sha_json},\"license_notice\":{license_json}}}"
        );
        let mut blob = Vec::new();
        blob.extend_from_slice(FOCRQ_MAGIC);
        blob.extend_from_slice(&version.to_le_bytes());
        blob.push(arch);
        blob.extend_from_slice(&sha);
        blob.extend_from_slice(&(header.len() as u64).to_le_bytes());
        blob.extend_from_slice(header.as_bytes());
        blob.extend_from_slice(payload);
        blob
    }

    /// Hand-assemble a v2 `.focrq` blob that declares a `model_id` (the A2 arch
    /// tag), with an arbitrary license notice so the arch-aware license check can
    /// be exercised independently of the tensor payload.
    fn build_focrq_with_model_id(
        directory_json: &str,
        payload: &[u8],
        license_notice: &str,
        model_id: &str,
    ) -> Vec<u8> {
        let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        let header = format!(
            "{{\"tensors\":{directory_json},\"arch_target\":0,\"source_sha256\":\"\",\
             \"license_notice\":\"{}\",\"model_id\":\"{}\"}}",
            esc(license_notice),
            esc(model_id),
        );
        let mut blob = Vec::new();
        blob.extend_from_slice(FOCRQ_MAGIC);
        blob.extend_from_slice(&FOCRQ_FORMAT_VERSION.to_le_bytes());
        blob.push(0);
        blob.extend_from_slice(&[0u8; 32]);
        blob.extend_from_slice(&(header.len() as u64).to_le_bytes());
        blob.extend_from_slice(header.as_bytes());
        blob.extend_from_slice(payload);
        blob
    }

    /// Hand-assemble a minimal safetensors blob from `(name, dtype, shape,
    /// bytes)` tensors laid out contiguously in directory order.
    fn build_safetensors(tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
        let mut entries = Vec::new();
        let mut payload = Vec::new();
        for (name, dtype, shape, data) in tensors {
            let beg = payload.len();
            payload.extend_from_slice(data);
            let end = payload.len();
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\
                 \"data_offsets\":[{beg},{end}]}}"
            ));
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut blob = Vec::new();
        blob.extend_from_slice(&(header.len() as u64).to_le_bytes());
        blob.extend_from_slice(header.as_bytes());
        blob.extend_from_slice(&payload);
        blob
    }

    fn bf16_le_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect()
    }

    fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|&v| v.to_le_bytes()).collect()
    }

    fn synthetic_weights(record: TensorRecord, bytes: Vec<u8>) -> Weights {
        Weights {
            bytes: std::sync::Arc::new(Backing::Owned(bytes)),
            payload_base: 0,
            directory: BTreeMap::from([("x".to_owned(), record)]),
            arch_target: 0,
            source_sha256: String::new(),
            license_notice: String::new(),
            model_id: model_arch::default_arch().id(),
            is_focrq: true,
            quant_recipe: None,
        }
    }

    // ── .focrq round-trip ──────────────────────────────────────────────────

    #[test]
    fn focrq_roundtrips_bf16_tensor_bit_exactly() {
        // bf16-representable values (every value below is exact in bf16).
        let vals = [1.0f32, -2.0, 0.5, 3.0, 0.0, -0.25];
        let payload = bf16_le_bytes(&vals);
        let dir = format!(
            "{{\"w\":{{\"dtype\":\"BF16\",\"shape\":[2,3],\
             \"byte_offset\":0,\"byte_len\":{}}}}}",
            payload.len()
        );
        let blob = build_focrq(1, 2, [7u8; 32], &dir, &payload);

        let w = Weights::from_bytes(blob).unwrap();
        assert!(w.is_focrq());
        assert_eq!(w.len(), 1);
        assert_eq!(w.arch_target(), 2);
        assert_eq!(w.source_sha256(), &"07".repeat(32));
        assert_eq!(w.license_notice(), FOCR_MODEL_LICENSE_NOTICE);

        let view = w.tensor("w").unwrap();
        assert_eq!(view.dtype, DType::BF16);
        assert_eq!(view.shape, &[2, 3]);

        let m = w.mat("w").unwrap();
        assert_eq!(m.shape(), (2, 3));
        // bf16 widening of these exact values is bit-exact.
        assert_eq!(m.data, vals);
    }

    #[test]
    fn rejects_focrq_without_baidu_mit_license_notice() {
        let blob = build_focrq_with_license(1, 0, [0u8; 32], "{}", &[], "Copyright (c) 2026 Baidu");
        let err = Weights::from_bytes(blob).unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(format!("{err}").contains("license_notice"));
        assert!(format!("{err}").contains("MIT License"));
    }

    // ── model_id arch tag (A2) ──────────────────────────────────────────────

    /// The GOT-OCR2 notice the registry declares (used so the test never drifts
    /// from `model_arch`'s source of truth).
    fn got_ocr2_notice() -> &'static str {
        crate::native_engine::model_arch::arch_by_id("got-ocr2")
            .expect("got-ocr2 is a registered arch")
            .license_notice()
    }

    #[test]
    fn focrq_absent_model_id_resolves_to_unlimited_ocr() {
        // A v1 blob (no model_id key at all) loads and reports the default arch.
        let blob = build_focrq(1, 0, [0u8; 32], "{}", &[]);
        let w = Weights::from_bytes(blob).unwrap();
        assert_eq!(w.model_id(), "unlimited-ocr");
    }

    #[test]
    fn focrq_empty_model_id_string_resolves_to_unlimited_ocr() {
        // The key physically present but empty (serde-default equivalent) also
        // resolves to the default, and still demands the Baidu/MIT notice.
        let blob = build_focrq_with_model_id("{}", &[], FOCR_MODEL_LICENSE_NOTICE, "");
        let w = Weights::from_bytes(blob).unwrap();
        assert_eq!(w.model_id(), "unlimited-ocr");
    }

    #[test]
    fn focrq_declares_got_ocr2_with_apache_notice_loads() {
        // A planned (not-yet-implemented) arch's weights still LOAD — only the
        // forward is gated — and the arch-specific Apache-2.0 notice is accepted.
        let blob = build_focrq_with_model_id("{}", &[], got_ocr2_notice(), "got-ocr2");
        let w = Weights::from_bytes(blob).unwrap();
        assert_eq!(w.model_id(), "got-ocr2");
    }

    #[test]
    fn focrq_unknown_model_id_is_refused() {
        // A forward-incompatible artifact (id this binary's registry lacks) is a
        // loud rejection, not a silent fallback to the default model.
        let blob =
            build_focrq_with_model_id("{}", &[], FOCR_MODEL_LICENSE_NOTICE, "totally-bogus-model");
        let err = Weights::from_bytes(blob).unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(format!("{err}").contains("unknown model_id"));
    }

    #[test]
    fn focrq_got_ocr2_with_wrong_notice_is_refused() {
        // got-ocr2 declared but carrying the Baidu/MIT notice (not its Apache-2.0
        // one) ⇒ the arch-aware license check refuses it.
        let blob = build_focrq_with_model_id("{}", &[], FOCR_MODEL_LICENSE_NOTICE, "got-ocr2");
        let err = Weights::from_bytes(blob).unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(format!("{err}").contains("does not match the registered got-ocr2 notice"));
    }

    #[test]
    fn safetensors_reports_default_model_id() {
        // A raw upstream shard is the Unlimited-OCR checkpoint.
        let blob = build_safetensors(&[("w", "BF16", vec![1], bf16_le_bytes(&[1.0]))]);
        let w = Weights::from_bytes(blob).unwrap();
        assert_eq!(w.model_id(), "unlimited-ocr");
        assert!(!w.is_focrq());
    }

    #[test]
    fn focrq_header_source_sha256_overrides_prefix_when_valid() {
        let vals = [1.0f32];
        let payload = bf16_le_bytes(&vals);
        let dir = format!(
            "{{\"w\":{{\"dtype\":\"BF16\",\"shape\":[1],\
             \"byte_offset\":0,\"byte_len\":{}}}}}",
            payload.len()
        );
        let header_sha = "ab".repeat(32);
        let blob = build_focrq_with_license_and_header_source_sha(
            1,
            0,
            [7u8; 32],
            &dir,
            &payload,
            FOCR_MODEL_LICENSE_NOTICE,
            &header_sha,
        );

        let w = Weights::from_bytes(blob).unwrap();
        assert_eq!(w.source_sha256(), header_sha);
    }

    #[test]
    fn rejects_malformed_focrq_header_source_sha256_override() {
        let vals = [1.0f32];
        let payload = bf16_le_bytes(&vals);
        let dir = format!(
            "{{\"w\":{{\"dtype\":\"BF16\",\"shape\":[1],\
             \"byte_offset\":0,\"byte_len\":{}}}}}",
            payload.len()
        );
        let blob = build_focrq_with_license_and_header_source_sha(
            1,
            0,
            [7u8; 32],
            &dir,
            &payload,
            FOCR_MODEL_LICENSE_NOTICE,
            "AB-not-lowercase-or-64-hex",
        );

        let err = Weights::from_bytes(blob).unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(format!("{err}").contains("source_sha256"));
        assert!(format!("{err}").contains("64 lowercase hex"));
    }

    #[test]
    fn focrq_roundtrips_f32_tensor_bit_exactly() {
        let vals = [1.5f32, -0.125, 1024.0, -3.0];
        let payload = f32_le_bytes(&vals);
        let dir = format!(
            "{{\"b\":{{\"dtype\":\"F32\",\"shape\":[4],\
             \"byte_offset\":0,\"byte_len\":{}}}}}",
            payload.len()
        );
        let blob = build_focrq(1, 0, [0u8; 32], &dir, &payload);
        let w = Weights::from_bytes(blob).unwrap();
        // 1-D tensor loads as [1, 4].
        let m = w.mat("b").unwrap();
        assert_eq!(m.shape(), (1, 4));
        assert_eq!(m.data, vals);
    }

    #[test]
    fn focrq_two_tensors_index_by_byte_range() {
        let a = bf16_le_bytes(&[1.0, 2.0]); // 4 bytes
        let b = f32_le_bytes(&[9.0, 8.0, 7.0]); // 12 bytes
        let mut payload = a.clone();
        payload.extend_from_slice(&b);
        let dir = format!(
            "{{\"a\":{{\"dtype\":\"BF16\",\"shape\":[2],\"byte_offset\":0,\"byte_len\":{}}},\
              \"b\":{{\"dtype\":\"F32\",\"shape\":[3],\"byte_offset\":{},\"byte_len\":{}}}}}",
            a.len(),
            a.len(),
            b.len()
        );
        let blob = build_focrq(1, 0, [0u8; 32], &dir, &payload);
        let w = Weights::from_bytes(blob).unwrap();
        assert_eq!(w.mat("a").unwrap().data, vec![1.0, 2.0]);
        assert_eq!(w.mat("b").unwrap().data, vec![9.0, 8.0, 7.0]);
    }

    #[test]
    fn focrq_qint8_roundtrips() {
        // n=2, k=3: 6 int8 weights + 2 f32 scales, scales after the weights.
        let w_bytes: Vec<u8> = [1i8, -2, 3, 4, -5, 6].iter().map(|&v| v as u8).collect();
        let scale_bytes = f32_le_bytes(&[0.1, 0.2]);
        let mut payload = w_bytes.clone();
        payload.extend_from_slice(&scale_bytes);
        let dir = "{\"q\":{\"dtype\":\"QInt8PerChan\",\"shape\":[2,3],\
             \"byte_offset\":0,\"byte_len\":6,\"scales_offset\":6,\"scales_len\":8}}";
        let blob = build_focrq(1, 0, [0u8; 32], dir, &payload);
        let w = Weights::from_bytes(blob).unwrap();
        let q = w.qint8("q").unwrap();
        assert_eq!(q.n, 2);
        assert_eq!(q.k, 3);
        assert_eq!(&q.w[..], &[1i8, -2, 3, 4, -5, 6]);
        assert_eq!(q.scales, vec![0.1, 0.2]);
    }

    // ── segmented backing (bd-syf2 wasm32 >2 GiB blobs) ─────────────────────

    /// A small multi-tensor `.focrq` exercising every accessor family: BF16
    /// (mat/vec), `QInt8PerChan` (qint8), `QInt4PerGroup` (qint4).
    fn segmented_fixture_blob() -> Vec<u8> {
        let mut builder = crate::quant::focrq::FocrqBuilder::new()
            .with_license_notice(FOCR_MODEL_LICENSE_NOTICE)
            .with_alignment(false);
        builder
            .add_tensor(
                "a.bf16",
                crate::quant::focrq::WriteDType::Bf16,
                vec![2, 3],
                bf16_le_bytes(&[1.0, -2.0, 4.0, 8.0, -0.5, 0.25]),
            )
            .expect("bf16 tensor");
        builder
            .add_quantized(
                "b.int8",
                crate::quant::focrq::WriteDType::QInt8PerChan,
                vec![2, 3],
                [1i8, -2, 3, 4, -5, 6].iter().map(|&v| v as u8).collect(),
                f32_le_bytes(&[0.1, 0.2]),
                0,
                0,
            )
            .expect("int8 tensor");
        builder
            .add_quantized(
                "c.int4",
                crate::quant::focrq::WriteDType::QInt4PerGroup,
                vec![2, 16],
                (0u8..16).collect(),
                f32_le_bytes(&[0.3, 0.4]),
                16,
                3,
            )
            .expect("int4 tensor");
        builder
            .add_tensor(
                "d.bf16",
                crate::quant::focrq::WriteDType::Bf16,
                vec![4],
                bf16_le_bytes(&[3.0, 5.0, -7.0, 9.0]),
            )
            .expect("trailing bf16 tensor");
        builder.build()
    }

    /// Split `blob` into `(start, bytes)` segments at the given absolute cut
    /// offsets (each cut becomes a segment boundary).
    fn split_at(blob: &[u8], cuts: &[u64]) -> Vec<(u64, Vec<u8>)> {
        let mut out = Vec::new();
        let mut prev = 0u64;
        for &cut in cuts {
            out.push((prev, blob[prev as usize..cut as usize].to_vec()));
            prev = cut;
        }
        out.push((prev, blob[prev as usize..].to_vec()));
        out
    }

    /// bd-syf2: a blob loaded from SEGMENTS is indistinguishable from the same
    /// blob loaded whole — every accessor returns identical bytes/values. This
    /// is what lets wasm32 (2 GiB max single allocation) hold a 3 GB artifact.
    #[test]
    fn from_segments_matches_from_bytes_across_every_accessor() {
        let blob = segmented_fixture_blob();
        let whole = Weights::from_bytes(blob.clone()).expect("whole blob loads");

        // Cut on tensor boundaries taken from the whole load's own directory.
        let payload_base = whole.payload_base as u64;
        let mut cuts: Vec<u64> = whole
            .directory
            .values()
            .map(|rec| {
                payload_base
                    + (rec.byte_offset + rec.byte_len).max(rec.scales_offset + rec.scales_len)
                        as u64
            })
            .collect();
        cuts.sort_unstable();
        cuts.retain(|&c| c < blob.len() as u64);
        assert!(cuts.len() >= 3, "fixture must offer several tensor edges");
        let segmented =
            Weights::from_segments(split_at(&blob, &cuts)).expect("segmented blob loads");

        let names: Vec<String> = whole.names().map(str::to_owned).collect();
        assert_eq!(
            segmented.names().collect::<Vec<_>>(),
            whole.names().collect::<Vec<_>>()
        );
        assert_eq!(segmented.arch_target(), whole.arch_target());
        assert_eq!(segmented.license_notice(), whole.license_notice());
        for name in &names {
            let (ws, ss) = (
                whole.tensor(name).expect("whole view"),
                segmented.tensor(name).expect("segmented view"),
            );
            assert_eq!(ss.dtype, ws.dtype, "{name} dtype");
            assert_eq!(ss.shape, ws.shape, "{name} shape");
            assert_eq!(ss.data, ws.data, "{name} payload bytes");
            assert_eq!(ss.scales, ws.scales, "{name} scale bytes");
        }
        assert_eq!(
            segmented.mat("a.bf16").unwrap().data,
            whole.mat("a.bf16").unwrap().data
        );
        assert_eq!(
            segmented.vec("d.bf16").unwrap(),
            whole.vec("d.bf16").unwrap()
        );
        let (qs, qw) = (
            segmented.qint8("b.int8").unwrap(),
            whole.qint8("b.int8").unwrap(),
        );
        assert_eq!(&qs.w[..], &qw.w[..]);
        assert_eq!(qs.scales, qw.scales);
        let (q4s, q4w) = (
            segmented.qint4("c.int4").unwrap(),
            whole.qint4("c.int4").unwrap(),
        );
        assert_eq!(&q4s.packed[..], &q4w.packed[..]);
        assert_eq!(q4s.scales.to_vec(), q4w.scales.to_vec());
        // The zero-copy views survive the loader handle, segmented or not.
        drop(segmented);
        assert_eq!(&q4s.packed[..], &q4w.packed[..]);
    }

    /// bd-syf2 planted negative: a plan that cuts INSIDE a tensor is refused at
    /// load, naming the tensor — never silently stitched into a copy.
    #[test]
    fn from_segments_rejects_a_split_inside_a_tensor() {
        let blob = segmented_fixture_blob();
        let whole = Weights::from_bytes(blob.clone()).expect("whole blob loads");
        let rec = whole.record("c.int4").expect("fixture tensor");
        // One byte into the int4 payload: a boundary no tensor may straddle.
        let bad_cut = whole.payload_base as u64 + rec.byte_offset as u64 + 1;
        let err = Weights::from_segments(split_at(&blob, &[bad_cut]))
            .expect_err("a mid-tensor split must be refused");
        let message = err.to_string();
        assert!(
            message.contains("c.int4") && message.contains("segment boundary"),
            "error must name the straddling tensor: {message}"
        );

        // Ordering/coverage violations are refused too.
        assert!(
            Weights::from_segments(vec![(1, blob.clone())]).is_err(),
            "a blob that does not start at 0 must be refused"
        );
        assert!(
            Weights::from_segments(vec![(0, blob[..64].to_vec()), (65, blob[65..].to_vec()),])
                .is_err(),
            "a gap between segments must be refused"
        );
        assert!(
            Weights::from_segments(Vec::new()).is_err(),
            "an empty segment list must be refused"
        );
    }

    /// bd-syf2: [`focrq_segment_plan`] cuts only at tensor edges, covers the
    /// artifact exactly, and the resulting segments load — the exact contract
    /// the wasm two-phase staging depends on. A short prefix asks for more
    /// bytes instead of guessing.
    #[test]
    fn focrq_segment_plan_cuts_only_on_tensor_boundaries() {
        let blob = segmented_fixture_blob();
        let total = blob.len() as u64;

        // A prefix shorter than the header reports how much is needed.
        // Probing converges in at most two steps: an 8-byte prefix can only ask
        // for the fixed preamble; the preamble then reveals the header length.
        let mut have = 8usize;
        let mut probes = 0;
        while let SegmentPlan::NeedPrefix { need_bytes } =
            focrq_segment_plan(&blob[..have], total, 64).expect("prefix probes cleanly")
        {
            probes += 1;
            assert!(
                need_bytes as usize > have,
                "a probe must ask for MORE bytes"
            );
            assert!(
                (need_bytes as usize) <= blob.len(),
                "the header is far smaller than the artifact"
            );
            have = need_bytes as usize;
            assert!(probes <= 2, "probing must converge in two steps");
        }
        assert_eq!(probes, 2, "8 bytes -> preamble -> header");

        // A tiny budget forces several cuts; each must land on a tensor edge.
        let SegmentPlan::Planned {
            segment_lens,
            payload_base,
        } = focrq_segment_plan(&blob, total, 24).expect("plan over the full prefix")
        else {
            panic!("the full prefix contains the header");
        };
        assert!(
            segment_lens.len() > 1,
            "a 24-byte budget must produce multiple segments, got {segment_lens:?}"
        );
        assert_eq!(segment_lens.iter().sum::<u64>(), total, "exact coverage");
        assert!(
            segment_lens[0] >= payload_base,
            "segment 0 must contain the whole preamble + header"
        );

        // The planned segmentation loads and agrees with the whole blob.
        let mut segments = Vec::new();
        let mut off = 0u64;
        for len in &segment_lens {
            segments.push((off, blob[off as usize..(off + len) as usize].to_vec()));
            off += len;
        }
        let planned = Weights::from_segments(segments).expect("planned segments load");
        let whole = Weights::from_bytes(blob).expect("whole blob loads");
        let names: Vec<String> = whole.names().map(str::to_owned).collect();
        for name in &names {
            assert_eq!(
                planned.tensor(name).unwrap().data,
                whole.tensor(name).unwrap().data,
                "{name} payload after planned segmentation"
            );
        }
    }

    /// bd-4l71 residency: a canonical row-major `QInt8PerChan` record is
    /// BORROWED from the loaded blob instead of being copied next to it (the
    /// wasm recipe's attention + `lm_head` set is ~244 MB). The borrowed view
    /// must be byte-for-byte the historical `view.data.iter().map(|&b| b as i8)`
    /// copy, and it must stay valid after the `Weights` is dropped (the view
    /// owns an `Arc` on the blob).
    #[test]
    fn qint8_row_major_payload_is_borrowed_zero_copy_and_byte_identical() {
        let raw: Vec<i8> = vec![1, -2, 3, 4, -5, 6];
        let w_bytes: Vec<u8> = raw.iter().map(|&v| v as u8).collect();
        let scale_bytes = f32_le_bytes(&[0.1, 0.2]);
        let mut payload = w_bytes.clone();
        payload.extend_from_slice(&scale_bytes);
        let dir = "{\"q\":{\"dtype\":\"QInt8PerChan\",\"shape\":[2,3],\
             \"byte_offset\":0,\"byte_len\":6,\"scales_offset\":6,\"scales_len\":8}}";
        let blob = build_focrq(1, 0, [0u8; 32], dir, &payload);
        let weights = Weights::from_bytes(blob).unwrap();
        let q = weights.qint8("q").unwrap();
        assert!(
            matches!(q.w, crate::native_engine::tensor::Int8Weights::Shared(_)),
            "a row-major int8 record must be borrowed, not copied"
        );
        assert_eq!(&q.w[..], &raw[..], "borrowed bytes == the historical copy");
        assert_eq!(q.scales, vec![0.1, 0.2]);
        assert_eq!((q.n, q.k), (2, 3));

        // The view keeps the blob alive on its own.
        drop(weights);
        assert_eq!(&q.w[..], &raw[..], "the view outlives the Weights handle");
    }

    #[test]
    fn qint8_records_dequant_on_access_via_mat_and_vec() {
        // bd-av64.12: an int8-stored GEMM read through the f32 accessors must
        // reconstruct `w[o][j] = qw[o][j] * scale[o]` EXACTLY (same arithmetic
        // as the expectation below), so f32 engines run quantized-storage
        // artifacts transparently.
        let w_bytes: Vec<u8> = [1i8, -2, 3, 4, -5, 6].iter().map(|&v| v as u8).collect();
        let scale_bytes = f32_le_bytes(&[0.1, 0.2]);
        let mut payload = w_bytes;
        payload.extend_from_slice(&scale_bytes);
        let dir = "{\"q\":{\"dtype\":\"QInt8PerChan\",\"shape\":[2,3],\
             \"byte_offset\":0,\"byte_len\":6,\"scales_offset\":6,\"scales_len\":8}}";
        let blob = build_focrq(1, 0, [0u8; 32], dir, &payload);
        let w = Weights::from_bytes(blob).unwrap();
        let expect = vec![
            1.0f32 * 0.1,
            -2.0f32 * 0.1,
            3.0f32 * 0.1,
            4.0f32 * 0.2,
            -5.0f32 * 0.2,
            6.0f32 * 0.2,
        ];
        let m = w.mat("q").unwrap();
        assert_eq!((m.rows, m.cols), (2, 3), "mat keeps the [n, k] shape");
        assert_eq!(m.data, expect, "mat dequantizes per output channel");
        assert_eq!(w.vec("q").unwrap(), expect, "vec dequantizes identically");
    }

    #[test]
    fn focrq_qint4_roundtrips() {
        // n=2, k=16, group_size=16 => 8 packed bytes/row (16 total), 2 scales.
        let packed: Vec<u8> = (0u8..16).collect();
        let scale_bytes = f32_le_bytes(&[0.1, 0.2]);
        let mut payload = packed.clone();
        payload.extend_from_slice(&scale_bytes);
        let dir = "{\"e\":{\"dtype\":\"QInt4PerGroup\",\"shape\":[2,16],\
             \"byte_offset\":0,\"byte_len\":16,\"scales_offset\":16,\"scales_len\":8,\
             \"group_size\":16,\"tier\":3}}";
        let blob = build_focrq(1, 0, [0u8; 32], dir, &payload);
        let w = Weights::from_bytes(blob).unwrap();
        let q = w.qint4("e").unwrap();
        assert_eq!(q.n, 2);
        assert_eq!(q.k, 16);
        assert_eq!(q.group_size, 16);
        assert_eq!(q.tier, 3);
        assert_eq!(&q.packed[..], &packed[..]);
        assert_eq!(q.scales.to_vec(), vec![0.1, 0.2]);
    }

    #[test]
    fn focrq_load_from_temp_file_roundtrips() {
        let vals = [1.0f32, -2.0, 4.0, 8.0];
        let payload = bf16_le_bytes(&vals);
        let dir = format!(
            "{{\"t\":{{\"dtype\":\"BF16\",\"shape\":[2,2],\"byte_offset\":0,\"byte_len\":{}}}}}",
            payload.len()
        );
        let blob = build_focrq(1, 1, [3u8; 32], &dir, &payload);

        let dir_path = std::env::temp_dir().join(format!(
            "focrq_test_{}_{}.focrq",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&dir_path, &blob).unwrap();
        let w = Weights::load(&dir_path).unwrap();
        let m = w.mat("t").unwrap();
        assert_eq!(m.shape(), (2, 2));
        assert_eq!(m.data, vals);
        // Clean up the temp file (test-created; allowed by RULE 1 since we made
        // it this run and it is a scratch artifact, but be defensive).
        let _ = std::fs::remove_file(&dir_path);
    }

    // ── census ──────────────────────────────────────────────────────────────

    #[test]
    fn census_accepts_exact_set() {
        let payload = bf16_le_bytes(&[1.0, 2.0]);
        let dir = format!(
            "{{\"x\":{{\"dtype\":\"BF16\",\"shape\":[2],\"byte_offset\":0,\"byte_len\":{}}}}}",
            payload.len()
        );
        let blob = build_focrq(1, 0, [0u8; 32], &dir, &payload);
        let w = Weights::from_bytes(blob).unwrap();
        assert!(w.census(["x"]).is_ok());
    }

    #[test]
    fn census_rejects_missing_tensor() {
        let payload = bf16_le_bytes(&[1.0, 2.0]);
        let dir = format!(
            "{{\"x\":{{\"dtype\":\"BF16\",\"shape\":[2],\"byte_offset\":0,\"byte_len\":{}}}}}",
            payload.len()
        );
        let blob = build_focrq(1, 0, [0u8; 32], &dir, &payload);
        let w = Weights::from_bytes(blob).unwrap();
        // Expect two tensors; only one is present.
        let err = w.census(["x", "y"]).unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(format!("{err}").contains("missing"));
        assert_eq!(err.exit_code(), 7);
    }

    #[test]
    fn census_rejects_unexpected_tensor() {
        let a = bf16_le_bytes(&[1.0]);
        let b = bf16_le_bytes(&[2.0]);
        let mut payload = a.clone();
        payload.extend_from_slice(&b);
        let dir = format!(
            "{{\"x\":{{\"dtype\":\"BF16\",\"shape\":[1],\"byte_offset\":0,\"byte_len\":{}}},\
              \"stale\":{{\"dtype\":\"BF16\",\"shape\":[1],\"byte_offset\":{},\"byte_len\":{}}}}}",
            a.len(),
            a.len(),
            b.len()
        );
        let blob = build_focrq(1, 0, [0u8; 32], &dir, &payload);
        let w = Weights::from_bytes(blob).unwrap();
        let err = w.census(["x"]).unwrap_err();
        assert!(format!("{err}").contains("unexpected"));
    }

    #[test]
    fn load_with_census_threads_through() {
        let payload = bf16_le_bytes(&[1.0, 2.0]);
        let dir = format!(
            "{{\"only\":{{\"dtype\":\"BF16\",\"shape\":[2],\"byte_offset\":0,\"byte_len\":{}}}}}",
            payload.len()
        );
        let blob = build_focrq(1, 0, [0u8; 32], &dir, &payload);
        // Reuse from_bytes path indirectly via a temp file.
        let path = std::env::temp_dir().join(format!("focrq_census_{}.focrq", std::process::id()));
        std::fs::write(&path, &blob).unwrap();
        assert!(Weights::load_with_census(&path, ["only"]).is_ok());
        assert!(Weights::load_with_census(&path, ["only", "missing"]).is_err());
        let _ = std::fs::remove_file(&path);
    }

    // ── safetensors fallback ─────────────────────────────────────────────────

    #[test]
    fn safetensors_header_parses_and_widens_bf16() {
        let vals = [1.0f32, -2.0, 0.5, 3.0];
        let blob =
            build_safetensors(&[("model.norm.weight", "BF16", vec![4], bf16_le_bytes(&vals))]);
        let w = Weights::from_bytes(blob).unwrap();
        assert!(!w.is_focrq());
        assert_eq!(w.len(), 1);
        assert!(w.contains("model.norm.weight"));
        let m = w.mat("model.norm.weight").unwrap();
        assert_eq!(m.shape(), (1, 4));
        assert_eq!(m.data, vals);
    }

    #[test]
    fn vec_accessor_widens_1d_param() {
        // The connector params (image_newline / view_seperator) are 1-D [1280]
        // BF16 in the real model; here a tiny [3].
        let vals = [0.25f32, -0.5, 1.0];
        let blob =
            build_safetensors(&[("model.image_newline", "BF16", vec![3], bf16_le_bytes(&vals))]);
        let w = Weights::from_bytes(blob).unwrap();
        let v = w.vec("model.image_newline").unwrap();
        assert_eq!(v, vals);
        assert!(w.vec("missing").is_err());
    }

    #[test]
    fn default_is_empty_and_accessors_error_not_panic() {
        // The sibling forward modules construct Weights::default() for their
        // NotImplemented-path tests; it must be a valid empty set whose
        // accessors error rather than panic.
        let w = Weights::default();
        assert!(w.is_empty());
        assert_eq!(w.len(), 0);
        assert!(w.tensor("anything").is_err());
        assert!(w.mat("anything").is_err());
        assert!(w.vec("anything").is_err());
        assert!(w.census(["x"]).is_err());
        assert!(w.census(std::iter::empty::<&str>()).is_ok());
    }

    #[test]
    fn safetensors_skips_metadata_key() {
        // Manually splice a __metadata__ entry into the header.
        let vals = [1.0f32, 2.0];
        let payload = f32_le_bytes(&vals);
        let header = format!(
            "{{\"__metadata__\":{{\"format\":\"pt\"}},\
              \"w\":{{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,{}]}}}}",
            payload.len()
        );
        let mut blob = Vec::new();
        blob.extend_from_slice(&(header.len() as u64).to_le_bytes());
        blob.extend_from_slice(header.as_bytes());
        blob.extend_from_slice(&payload);
        let w = Weights::from_bytes(blob).unwrap();
        assert_eq!(w.len(), 1);
        assert_eq!(w.mat("w").unwrap().data, vals);
    }

    #[test]
    fn safetensors_two_tensors_index_correctly() {
        let blob = build_safetensors(&[
            ("a", "F32", vec![2], f32_le_bytes(&[1.0, 2.0])),
            ("b", "BF16", vec![3], bf16_le_bytes(&[4.0, 5.0, 6.0])),
        ]);
        let w = Weights::from_bytes(blob).unwrap();
        assert_eq!(w.mat("a").unwrap().data, vec![1.0, 2.0]);
        assert_eq!(w.mat("b").unwrap().data, vec![4.0, 5.0, 6.0]);
    }

    // ── bf16/f16 widening ────────────────────────────────────────────────────

    #[test]
    fn bf16_widening_of_known_values() {
        // bf16 keeps the high 16 bits of an f32; these are all exact.
        let vals = [0.0f32, 1.0, -1.0, 2.0, 0.5, -0.5, 256.0, -3.0];
        let bytes = bf16_le_bytes(&vals);
        let out = decode_f32(DType::BF16, &bytes).unwrap();
        assert_eq!(out, vals);
    }

    #[test]
    fn bf16_widening_truncates_mantissa_not_silently_wrong() {
        // 1.1 is NOT bf16-exact: bf16(1.1) widened back is 1.1015625.
        let bytes = bf16::from_f32(1.1).to_le_bytes();
        let out = decode_f32(DType::BF16, &bytes).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.101_562_5).abs() < 1e-7);
    }

    #[test]
    fn f16_widening_of_known_values() {
        let vals = [0.0f32, 1.0, -2.0, 0.25];
        let bytes: Vec<u8> = vals
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        let out = decode_f32(DType::F16, &bytes).unwrap();
        assert_eq!(out, vals);
    }

    // ── error paths ──────────────────────────────────────────────────────────

    #[test]
    fn rejects_unknown_magic() {
        // Not FOCRQ, and not a valid safetensors (header_len overruns).
        let mut blob = Vec::new();
        blob.extend_from_slice(&(9999u64).to_le_bytes());
        blob.extend_from_slice(b"junk");
        let err = Weights::from_bytes(blob).unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
    }

    #[test]
    fn rejects_future_focrq_version() {
        let payload = bf16_le_bytes(&[1.0]);
        let dir = format!(
            "{{\"x\":{{\"dtype\":\"BF16\",\"shape\":[1],\"byte_offset\":0,\"byte_len\":{}}}}}",
            payload.len()
        );
        let blob = build_focrq(FOCRQ_FORMAT_VERSION + 1, 0, [0u8; 32], &dir, &payload);
        let err = Weights::from_bytes(blob).unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(format!("{err}").contains("newer than this binary"));
    }

    #[test]
    fn rejects_unknown_arch_target() {
        let payload = bf16_le_bytes(&[1.0]);
        let dir = format!(
            "{{\"x\":{{\"dtype\":\"BF16\",\"shape\":[1],\"byte_offset\":0,\"byte_len\":{}}}}}",
            payload.len()
        );
        let blob = build_focrq(1, MAX_ARCH_TARGET + 1, [0u8; 32], &dir, &payload);
        let err = Weights::from_bytes(blob).expect_err("unknown arch target must fail");
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(err.to_string().contains("arch_target 4 is unsupported"));
    }

    #[test]
    fn rejects_directory_overrunning_payload() {
        // byte_len claims 100 bytes but payload only has 4.
        let payload = bf16_le_bytes(&[1.0, 2.0]); // 4 bytes
        let dir = "{\"x\":{\"dtype\":\"BF16\",\"shape\":[2],\"byte_offset\":0,\"byte_len\":100}}";
        let blob = build_focrq(1, 0, [0u8; 32], dir, &payload);
        let err = Weights::from_bytes(blob).unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
    }

    #[test]
    fn rejects_byte_len_disagreeing_with_shape() {
        // shape [2,3]=6 bf16 elems => 12 bytes, but we claim 4.
        let payload = bf16_le_bytes(&[1.0, 2.0]); // 4 bytes
        let dir = "{\"x\":{\"dtype\":\"BF16\",\"shape\":[2,3],\"byte_offset\":0,\"byte_len\":4}}";
        let blob = build_focrq(1, 0, [0u8; 32], dir, &payload);
        let err = Weights::from_bytes(blob).unwrap_err();
        assert!(format!("{err}").contains("shape×dtype") || format!("{err}").contains("overruns"));
    }

    #[test]
    fn rejects_overlapping_tensor_data_ranges_at_load_time() {
        let dir = "{\"a\":{\"dtype\":\"BF16\",\"shape\":[2],\"byte_offset\":0,\"byte_len\":4},\
             \"b\":{\"dtype\":\"BF16\",\"shape\":[2],\"byte_offset\":0,\"byte_len\":4}}";
        let blob = build_focrq(1, 0, [0u8; 32], dir, &[0u8; 4]);
        let err = Weights::from_bytes(blob).expect_err("aliased tensor data must fail");
        let text = err.to_string();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(text.contains("payload ranges overlap"), "{text}");
        assert!(text.contains("\"a\" data"), "{text}");
        assert!(text.contains("\"b\" data"), "{text}");
    }

    #[test]
    fn rejects_tensor_data_and_scale_alias_at_load_time() {
        let dir = "{\"q\":{\"dtype\":\"QInt8PerChan\",\"shape\":[1,4],\
             \"byte_offset\":0,\"byte_len\":4,\"scales_offset\":0,\"scales_len\":4}}";
        let blob = build_focrq(1, 0, [0u8; 32], dir, &[0u8; 4]);
        let err = Weights::from_bytes(blob).expect_err("aliased data/scales must fail");
        let text = err.to_string();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(text.contains("payload ranges overlap"), "{text}");
        assert!(text.contains("\"q\" data"), "{text}");
        assert!(text.contains("\"q\" scales"), "{text}");
    }

    #[test]
    fn rejects_shape_numel_overflow_without_panicking() {
        let dir = format!(
            "{{\"x\":{{\"dtype\":\"BF16\",\"shape\":[{},2],\"byte_offset\":0,\"byte_len\":0}}}}",
            usize::MAX
        );
        let blob = build_focrq(1, 0, [0u8; 32], &dir, &[]);
        let err = Weights::from_bytes(blob).unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(format!("{err}").contains("element count overflows"));
    }

    #[test]
    fn mat_accessor_rejects_shape_product_overflow() {
        let w = synthetic_weights(
            TensorRecord {
                dtype: DType::BF16,
                shape: vec![usize::MAX, 2],
                byte_offset: 0,
                byte_len: 0,
                scales_offset: 0,
                scales_len: 0,
                group_size: 0,
                tier: 0,
            },
            Vec::new(),
        );
        let err = w.mat("x").unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(format!("{err}").contains("rows*cols overflows"));
    }

    #[test]
    fn qint8_accessor_rejects_shape_product_overflow() {
        let w = synthetic_weights(
            TensorRecord {
                dtype: DType::QInt8PerChan,
                shape: vec![usize::MAX, 2],
                byte_offset: 0,
                byte_len: 0,
                scales_offset: 0,
                scales_len: 0,
                group_size: 0,
                tier: 0,
            },
            Vec::new(),
        );
        let err = w.qint8("x").unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(format!("{err}").contains("n*k overflows"));
    }

    #[test]
    fn qint4_accessor_rejects_shape_product_overflow() {
        let w = synthetic_weights(
            TensorRecord {
                dtype: DType::QInt4PerGroup,
                shape: vec![usize::MAX, 32],
                byte_offset: 0,
                byte_len: 0,
                scales_offset: 0,
                scales_len: 0,
                group_size: 16,
                tier: 0,
            },
            Vec::new(),
        );
        let err = w.qint4("x").unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(format!("{err}").contains("n*k/2 overflows"));
    }

    #[test]
    fn qint4_accessor_rejects_noncanonical_group_size_even_when_it_divides_k() {
        let packed = vec![0u8; 16];
        let scale_bytes = f32_le_bytes(&[1.0, 1.0, 1.0, 1.0]);
        let mut payload = packed;
        payload.extend_from_slice(&scale_bytes);
        let dir = "{\"q\":{\"dtype\":\"QInt4PerGroup\",\"shape\":[1,32],\
             \"byte_offset\":0,\"byte_len\":16,\"scales_offset\":16,\"scales_len\":16,\
             \"group_size\":8,\"tier\":1}}";
        let blob = build_focrq(1, 0, [0u8; 32], dir, &payload);
        let w = Weights::from_bytes(blob).unwrap();
        let err = w.qint4("q").unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(format!("{err}").contains("must be 16 or 32"));
    }

    #[test]
    fn rejects_qint4_odd_numel_at_load_time() {
        let dir = "{\"odd\":{\"dtype\":\"QInt4PerGroup\",\"shape\":[1,3],\
             \"byte_offset\":0,\"byte_len\":1,\"scales_offset\":1,\"scales_len\":0,\
             \"group_size\":1}}";
        let blob = build_focrq(1, 0, [0u8; 32], dir, &[0u8]);
        let err = Weights::from_bytes(blob).unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(format!("{err}").contains("odd element count"));
    }

    #[test]
    fn tensor_not_found_is_error() {
        let payload = bf16_le_bytes(&[1.0]);
        let dir = format!(
            "{{\"x\":{{\"dtype\":\"BF16\",\"shape\":[1],\"byte_offset\":0,\"byte_len\":{}}}}}",
            payload.len()
        );
        let blob = build_focrq(1, 0, [0u8; 32], &dir, &payload);
        let w = Weights::from_bytes(blob).unwrap();
        let err = w.tensor("nope").unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(w.mat("nope").is_err());
        assert!(w.qint8("nope").is_err());
    }

    #[test]
    fn mat_rejects_qint4_tensor() {
        let packed: Vec<u8> = (0u8..8).collect();
        let scale_bytes = f32_le_bytes(&[0.1]);
        let mut payload = packed.clone();
        payload.extend_from_slice(&scale_bytes);
        let dir = "{\"q\":{\"dtype\":\"QInt4PerGroup\",\"shape\":[1,16],\
             \"byte_offset\":0,\"byte_len\":8,\"scales_offset\":8,\"scales_len\":4,\
             \"group_size\":16,\"tier\":3}}";
        let blob = build_focrq(1, 0, [0u8; 32], dir, &payload);
        let w = Weights::from_bytes(blob).unwrap();
        // qint8 now transparently dequantizes through mat()/vec(); qint4 stays
        // a typed quantized accessor only until that path has a proven f32
        // meaning.
        assert!(w.mat("q").is_err());
        assert!(w.vec("q").is_err());
        assert!(w.qint4("q").is_ok());
    }

    /// bd-2mo.3: an `--arch aarch64-smmla` artifact (arch_target 1, panel
    /// payload) loads per the DISPATCHED tier — panels kept verbatim
    /// (zero-shuffle) when SMMLA is selected, un-permuted to canonical
    /// row-major otherwise. Tier-portable: this pins whichever branch this
    /// host takes, and the OTHER branch's correctness is gated by the
    /// packed-B kernel parity + decoder layout-parity tests.
    #[test]
    fn packed_focrq_loads_per_dispatched_tier() {
        let (n, k) = (3usize, 5usize); // odd n + k off the 8-boundary (padding)
        let w_rm: Vec<i8> = (0..n * k).map(|i| (i as i8) - 7).collect();
        let (panels, _, _) = crate::simd::pack::smmla_pack_panels(&w_rm, 0, n, k, k);
        let panel_bytes: Vec<u8> = panels.iter().map(|&v| v as u8).collect();
        let scale_bytes = f32_le_bytes(&[0.1, 0.2, 0.3]);
        let mut payload = panel_bytes.clone();
        payload.extend_from_slice(&scale_bytes);
        let dir = format!(
            "{{\"q\":{{\"dtype\":\"QInt8PerChan\",\"shape\":[{n},{k}],             \"byte_offset\":0,\"byte_len\":{},\"scales_offset\":{},\"scales_len\":12}}}}",
            panel_bytes.len(),
            panel_bytes.len()
        );
        let blob = build_focrq(1, 1, [0u8; 32], &dir, &payload);
        let w =
            Weights::from_bytes(blob).expect("packed artifact loads (census accepts panel len)");
        assert_eq!(w.arch_target(), 1);
        let q = w.qint8("q").expect("qint8 readback");
        assert_eq!((q.n, q.k), (n, k));
        assert_eq!(q.scales, vec![0.1, 0.2, 0.3]);
        if crate::simd::detected_tier() == crate::simd::IsaTier::Smmla {
            assert_eq!(
                q.layout,
                crate::native_engine::tensor::WeightLayout::SmmlaPanels,
                "SMMLA host must keep the offline panels (zero-shuffle)"
            );
            assert_eq!(&q.w[..], &panels[..], "panel bytes verbatim");
        } else {
            assert_eq!(
                q.layout,
                crate::native_engine::tensor::WeightLayout::RowMajor,
                "non-SMMLA host must un-permute to canonical row-major"
            );
            assert_eq!(&q.w[..], &w_rm[..], "un-permute is lossless");
        }
        println!(
            r#"{{"check":"packed_focrq_load","tier":"{}","layout":"{:?}","result":"pass"}}"#,
            crate::simd::tier_string(),
            q.layout
        );
    }

    /// A corrupt panel payload (length disagreeing with the packed rule)
    /// fails the census loudly instead of mis-slicing.
    #[test]
    fn packed_focrq_rejects_wrong_panel_length() {
        let (n, k) = (3usize, 5usize);
        // Deliberately store the ROW-MAJOR length (15) under arch_target 1;
        // the packed rule wants ceil(3/2)*ceil(5/8)*16 = 32.
        let w_bytes: Vec<u8> = (0..n * k).map(|i| i as u8).collect();
        let scale_bytes = f32_le_bytes(&[0.1, 0.2, 0.3]);
        let mut payload = w_bytes.clone();
        payload.extend_from_slice(&scale_bytes);
        let dir = format!(
            "{{\"q\":{{\"dtype\":\"QInt8PerChan\",\"shape\":[{n},{k}],             \"byte_offset\":0,\"byte_len\":{},\"scales_offset\":{},\"scales_len\":12}}}}",
            w_bytes.len(),
            w_bytes.len()
        );
        let blob = build_focrq(1, 1, [0u8; 32], &dir, &payload);
        let err = Weights::from_bytes(blob).unwrap_err();
        assert!(
            matches!(err, FocrError::FormatMismatch(_)),
            "wrong panel length must FormatMismatch, got {err:?}"
        );
    }

    /// bd-2mo.22: the explicitly opted-in mapped view is BYTE-IDENTICAL to the
    /// safe default owned read — every tensor, every scale, the whole directory.
    #[test]
    fn mmap_load_is_byte_identical_to_owned_read() {
        let (n, k) = (3usize, 5usize);
        let w_rm: Vec<i8> = (0..n * k).map(|i| (i as i8) - 7).collect();
        let w_bytes: Vec<u8> = w_rm.iter().map(|&v| v as u8).collect();
        let scale_bytes = f32_le_bytes(&[0.1, 0.2, 0.3]);
        let mut payload = w_bytes.clone();
        payload.extend_from_slice(&scale_bytes);
        let dir = format!(
            "{{\"q\":{{\"dtype\":\"QInt8PerChan\",\"shape\":[{n},{k}],             \"byte_offset\":0,\"byte_len\":{},\"scales_offset\":{},\"scales_len\":12}}}}",
            w_bytes.len(),
            w_bytes.len()
        );
        let blob = build_focrq(1, 0, [0u8; 32], &dir, &payload);

        let mut tmp = std::env::temp_dir();
        tmp.push(format!("focr_mmap_eq_{}.focrq", std::process::id()));
        std::fs::write(&tmp, &blob).expect("write temp artifact");

        let owned_default = Weights::load(&tmp).expect("default owned load");
        assert!(
            !owned_default.is_mapped() || mmap_requested(),
            "load may map only after explicit opt-in"
        );
        let file = std::fs::File::open(&tmp).expect("reopen temp artifact");
        let mapped =
            Weights::load_opened_with_mmap_policy(file, &tmp, true).expect("explicit mmap load");
        assert!(mapped.is_mapped(), "explicit mmap policy must map");
        let owned = Weights::from_bytes(blob).expect("owned parse");

        let a: Vec<&str> = mapped.names().collect();
        let b: Vec<&str> = owned.names().collect();
        assert_eq!(a, b, "directory identical");
        let qa = mapped.qint8("q").expect("mapped qint8");
        let qb = owned.qint8("q").expect("owned qint8");
        assert_eq!(qa.w, qb.w);
        assert_eq!(qa.scales, qb.scales);
        println!(r#"{{"check":"mmap_owned_equivalence","result":"pass"}}"#);
    }

    #[test]
    fn load_missing_file_is_model_not_found() {
        let err = Weights::load(Path::new("/definitely/not/a/real/weights.focrq")).unwrap_err();
        assert!(matches!(err, FocrError::ModelNotFound(_)));
        assert_eq!(err.exit_code(), 3);
    }

    #[test]
    fn names_are_sorted_and_complete() {
        let blob = build_safetensors(&[
            ("zeta", "F32", vec![1], f32_le_bytes(&[1.0])),
            ("alpha", "F32", vec![1], f32_le_bytes(&[2.0])),
            ("mid", "F32", vec![1], f32_le_bytes(&[3.0])),
        ]);
        let w = Weights::from_bytes(blob).unwrap();
        let names: Vec<&str> = w.names().collect();
        assert_eq!(names, vec!["alpha", "mid", "zeta"]);
    }
}
