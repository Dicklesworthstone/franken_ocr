//! The data currency for the native forward (PROPOSED_ARCHITECTURE.md §4).
//!
//! The whole hand-written model speaks one f32 activation rail — [`Mat`], a
//! row-major `[rows, cols]` matrix over a flat `Vec<f32>` — plus the quantized
//! weight structs that the int8/int4 GEMM paths consume. This is the *only*
//! currency that crosses `native_engine/*` module boundaries (no tensor graph,
//! no autograd, no `ft-api` session/tape — we reach straight to the
//! `ft-kernel-cpu` free functions over `&[f32]`).
//!
//! Numerics note (P1): `Mat` is f32 because the parity spine is f32. Quantized
//! weights ([`QInt8`], [`QInt4`]) are an *additive* layer behind kill-switches
//! (plan §5.3), never a separate code path — they dequantize back into the same
//! f32 rail at the GEMM boundary.

/// A row-major `[rows, cols]` f32 matrix — the activation currency.
///
/// `data.len() == rows * cols`; element `(r, c)` lives at `data[r * cols + c]`.
/// This is the contiguous layout every `ft-kernel-cpu` f32 entrypoint expects
/// (it matches a `TensorMeta::from_shape(vec![rows, cols], F32, Cpu)` exactly).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mat {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row-major elements, length `rows * cols`.
    pub data: Vec<f32>,
}

fn shape_len(context: &str, rows: usize, cols: usize) -> usize {
    let len = rows.checked_mul(cols);
    assert!(
        len.is_some(),
        "{context}: rows*cols overflow ({rows} * {cols})"
    );
    len.unwrap_or(0)
}

impl Mat {
    /// Construct from an explicit shape + backing vector.
    ///
    /// # Panics
    /// Panics if `data.len() != rows * cols` (a shape/length contract
    /// violation is a programming error, caught early).
    #[must_use]
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        let len = shape_len("Mat::from_vec", rows, cols);
        assert_eq!(
            data.len(),
            len,
            "Mat::from_vec: data len {} != rows*cols {}",
            data.len(),
            len
        );
        Self { rows, cols, data }
    }

    /// An uninitialized-shaped matrix filled with zeros.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let len = shape_len("Mat::zeros", rows, cols);
        Self {
            rows,
            cols,
            data: vec![0.0f32; len],
        }
    }

    /// Alias for [`Mat::zeros`] sized like a fresh activation buffer.
    ///
    /// Distinct name kept for call-site readability where a buffer is being
    /// *allocated* (vs. a genuine all-zero constant); both produce the same
    /// value.
    #[must_use]
    pub fn new(rows: usize, cols: usize) -> Self {
        Self::zeros(rows, cols)
    }

    /// Total element count (`rows * cols`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the matrix holds no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// `(rows, cols)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Element `(r, c)`.
    ///
    /// # Panics
    /// Panics if `r >= rows` or `c >= cols`.
    #[must_use]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        assert!(r < self.rows && c < self.cols, "Mat::get out of bounds");
        self.data[r * self.cols + c]
    }

    /// Set element `(r, c)`.
    ///
    /// # Panics
    /// Panics if `r >= rows` or `c >= cols`.
    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        assert!(r < self.rows && c < self.cols, "Mat::set out of bounds");
        self.data[r * self.cols + c] = v;
    }

    /// Borrow row `r` as a contiguous `cols`-length slice.
    ///
    /// # Panics
    /// Panics if `r >= rows`.
    #[must_use]
    pub fn row(&self, r: usize) -> &[f32] {
        assert!(r < self.rows, "Mat::row out of bounds");
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Mutably borrow row `r` as a contiguous `cols`-length slice.
    ///
    /// # Panics
    /// Panics if `r >= rows`.
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        assert!(r < self.rows, "Mat::row_mut out of bounds");
        let c = self.cols;
        &mut self.data[r * c..(r + 1) * c]
    }
}

/// Storage for a [`QInt8`]'s int8 weight buffer: an owned `Vec<i8>` (weights
/// quantized at load, fused/repacked weights, tests) or a zero-copy
/// [`SharedBytes`](super::weights::SharedBytes) view into the loaded model blob
/// (bd-4l71 residency: a `.focrq` `QInt8PerChan` payload is ALREADY the exact
/// row-major `[n, k]` int8 byte image, so copying it next to the resident
/// artifact wastes ~244 MB on the wasm recipe's attention + `lm_head` set).
///
/// The `u8 -> i8` reinterpretation is a same-size, alignment-1 plain-old-data
/// cast performed by `bytemuck::cast_slice` — no `unsafe` in this crate, and
/// the VALUES are exactly what the historical `view.data.iter().map(|&b| b as
/// i8)` copy produced, so every consumer is bit-exactly storage-agnostic.
#[derive(Debug, Clone)]
pub enum Int8Weights {
    /// Fully-owned int8 weights.
    Owned(Vec<i8>),
    /// Zero-copy view into the loaded blob (holds the blob alive via `Arc`).
    Shared(super::weights::SharedBytes),
}

impl std::ops::Deref for Int8Weights {
    type Target = [i8];
    fn deref(&self) -> &[i8] {
        match self {
            Int8Weights::Owned(v) => v,
            Int8Weights::Shared(s) => bytemuck::cast_slice(s),
        }
    }
}

impl From<Vec<i8>> for Int8Weights {
    fn from(v: Vec<i8>) -> Self {
        Int8Weights::Owned(v)
    }
}

impl PartialEq for Int8Weights {
    fn eq(&self, other: &Self) -> bool {
        self[..] == other[..]
    }
}

/// A symmetric per-output-channel int8-quantized linear weight.
///
/// Stored in PyTorch `[out, in]` row-major layout (`n = out`, `k = in`): `w` is
/// `n * k` int8 weights, `scales` is one f32 per output channel
/// (`scale[o] = max(|w_row|) / 127`, zero-point 0). This is exactly what
/// [`ft_kernel_cpu::linear_int8_dynamic_f32`] consumes; build it with
/// [`super::nn::quantize_int8`] (which wraps
/// `ft_kernel_cpu::quantize_per_output_channel_i8`).
///
/// The quant recipe is fixed (plan §5.3): only the decoder FFN/expert GEMMs are
/// quantized by default; attention/lm_head int8 is opt-in behind kill-switches.
#[derive(Debug, Clone, PartialEq)]
pub struct QInt8 {
    /// Int8 weights in the byte order [`Self::layout`] declares (row-major
    /// `[n, k]` unless an offline-packed artifact kept its panels). May be an
    /// owned buffer or a zero-copy view into the loaded blob
    /// ([`Int8Weights::Shared`], bd-4l71 residency).
    pub w: Int8Weights,
    /// Per-output-channel scales, length `n`.
    pub scales: Vec<f32>,
    /// Output dimension (number of rows / output channels).
    pub n: usize,
    /// Input dimension (contraction length / number of columns).
    pub k: usize,
    /// Byte order of `w`. [`WeightLayout::RowMajor`] everywhere except when
    /// the loader keeps an `--arch aarch64-smmla` artifact's offline panels
    /// because the SMMLA tier is dispatched (bd-2mo.3 zero-shuffle path).
    pub layout: WeightLayout,
}

/// Byte order of a [`QInt8`]'s weight buffer.
///
/// The quantized VALUES are identical in either layout (the packing is a pure
/// zero-padded permutation — [`crate::simd::pack`]); only the byte order
/// differs. Every GEMV entry point in `decoder.rs` accepts both and produces
/// bit-identical i32 accumulations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightLayout {
    /// Canonical PyTorch `[out, in]` row-major (`n * k` bytes).
    RowMajor,
    /// Offline SMMLA `[2 rows × 8 cols]` panels
    /// (`ceil(n/2) * ceil(k/8) * 16` bytes; see
    /// [`crate::simd::pack::smmla_pack_panels`]).
    SmmlaPanels,
}

impl QInt8 {
    /// Construct from raw quantized parts.
    ///
    /// # Panics
    /// Panics on a length/shape mismatch (`w.len() != n*k` or
    /// `scales.len() != n`).
    #[must_use]
    pub fn new(w: Vec<i8>, scales: Vec<f32>, n: usize, k: usize) -> Self {
        let len = shape_len("QInt8::new", n, k);
        assert_eq!(w.len(), len, "QInt8: w len {} != n*k {}", w.len(), len);
        assert_eq!(
            scales.len(),
            n,
            "QInt8: scales len {} != n {}",
            scales.len(),
            n
        );
        Self {
            w: w.into(),
            scales,
            n,
            k,
            layout: WeightLayout::RowMajor,
        }
    }

    /// Construct from a ZERO-COPY row-major int8 payload view into the loaded
    /// blob (bd-4l71 residency) — the borrowing twin of [`Self::new`], with the
    /// same shape contract and the same observable weights.
    ///
    /// # Panics
    /// Panics on a length/shape mismatch (`w.len() != n*k` or
    /// `scales.len() != n`).
    #[must_use]
    pub fn new_shared(
        w: super::weights::SharedBytes,
        scales: Vec<f32>,
        n: usize,
        k: usize,
    ) -> Self {
        let len = shape_len("QInt8::new_shared", n, k);
        assert_eq!(w.len(), len, "QInt8: w len {} != n*k {}", w.len(), len);
        assert_eq!(
            scales.len(),
            n,
            "QInt8: scales len {} != n {}",
            scales.len(),
            n
        );
        Self {
            w: Int8Weights::Shared(w),
            scales,
            n,
            k,
            layout: WeightLayout::RowMajor,
        }
    }

    /// Construct from OFFLINE-packed SMMLA panels (`focr convert --arch
    /// aarch64-smmla`, kept packed by the loader when the SMMLA tier is
    /// dispatched — bd-2mo.3).
    ///
    /// # Panics
    /// Panics on a length mismatch (`w.len() !=
    /// [`crate::simd::pack::smmla_packed_len`]` or `scales.len() != n`).
    #[must_use]
    pub fn new_smmla_panels(w: Vec<i8>, scales: Vec<f32>, n: usize, k: usize) -> Self {
        let len = crate::simd::pack::smmla_packed_len(n, k);
        assert_eq!(
            w.len(),
            len,
            "QInt8: panel len {} != ceil(n/2)*ceil(k/8)*16 {}",
            w.len(),
            len
        );
        assert_eq!(
            scales.len(),
            n,
            "QInt8: scales len {} != n {}",
            scales.len(),
            n
        );
        Self {
            w: w.into(),
            scales,
            n,
            k,
            layout: WeightLayout::SmmlaPanels,
        }
    }

    /// The weight-buffer byte length [`Self::layout`] implies (`n*k` row-major;
    /// `ceil(n/2)*ceil(k/8)*16` for offline SMMLA panels).
    #[must_use]
    pub fn expected_w_len(&self) -> usize {
        match self.layout {
            WeightLayout::RowMajor => self.n * self.k,
            WeightLayout::SmmlaPanels => crate::simd::pack::smmla_packed_len(self.n, self.k),
        }
    }
}

/// Storage for a [`QInt4`]'s packed nibble payload: an owned buffer (synthetic
/// weights / tests) or a zero-copy [`SharedBytes`] view into the loaded model
/// blob (the bd-4l71 residency mode — the wasm artifact's ~1.26 GB expert
/// payload must never exist twice). Both deref to the identical `[n, k/2]`
/// byte layout, so every consumer is storage-agnostic.
#[derive(Debug, Clone)]
pub enum PackedBytes {
    /// Fully-owned packed nibbles.
    Owned(Vec<u8>),
    /// Zero-copy view into the loaded blob (holds the blob alive via `Arc`).
    Shared(super::weights::SharedBytes),
}

impl std::ops::Deref for PackedBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            PackedBytes::Owned(v) => v,
            PackedBytes::Shared(s) => s,
        }
    }
}

impl From<Vec<u8>> for PackedBytes {
    fn from(v: Vec<u8>) -> Self {
        PackedBytes::Owned(v)
    }
}

impl PartialEq for PackedBytes {
    fn eq(&self, other: &Self) -> bool {
        self[..] == other[..]
    }
}

/// Storage for a [`QInt4`]'s per-group scales: owned f32 (synthetic weights /
/// tests) or the artifact's little-endian f32 bytes referenced zero-copy
/// ([`SharedBytes`], bd-4l71 residency: ~0.53 GB of expert scales stay in the
/// blob). The raw form is decoded per-use into a caller scratch — real `.focrq`
/// payload offsets carry no alignment guarantee, so the bytes cannot be
/// reinterpreted as `&[f32]` in place; the decoded VALUES are identical
/// (`f32::from_le_bytes`, the same decode the eager loader performed).
#[derive(Debug, Clone)]
pub enum GroupScales {
    /// Fully-owned decoded scales.
    Owned(Vec<f32>),
    /// Little-endian f32 bytes referenced zero-copy from the loaded blob
    /// (`4 * len` bytes).
    RawLe(super::weights::SharedBytes),
}

impl GroupScales {
    /// Number of f32 scales.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            GroupScales::Owned(v) => v.len(),
            GroupScales::RawLe(bytes) => bytes.len() / 4,
        }
    }

    /// Whether there are no scales.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow scales `[start, end)` as `&[f32]`, decoding through `scratch`
    /// when the storage is raw little-endian bytes. The returned values are
    /// identical for both storages (same `f32::from_le_bytes` decode the eager
    /// loader used), so callers are bit-exactly storage-agnostic.
    ///
    /// # Panics
    /// Panics if `start > end` or `end > self.len()` (caller shape bug).
    #[must_use]
    pub fn slice_f32<'a>(
        &'a self,
        start: usize,
        end: usize,
        scratch: &'a mut Vec<f32>,
    ) -> &'a [f32] {
        assert!(
            start <= end && end <= self.len(),
            "GroupScales::slice_f32: [{start}, {end}) out of bounds (len {})",
            self.len()
        );
        match self {
            GroupScales::Owned(v) => &v[start..end],
            GroupScales::RawLe(bytes) => {
                scratch.clear();
                scratch.extend(
                    bytes[start * 4..end * 4]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c)),
                );
                scratch
            }
        }
    }

    /// Decode every scale to an owned `Vec<f32>`.
    #[must_use]
    pub fn to_vec(&self) -> Vec<f32> {
        let mut scratch = Vec::new();
        self.slice_f32(0, self.len(), &mut scratch).to_vec()
    }
}

impl From<Vec<f32>> for GroupScales {
    fn from(v: Vec<f32>) -> Self {
        GroupScales::Owned(v)
    }
}

impl PartialEq for GroupScales {
    fn eq(&self, other: &Self) -> bool {
        // Semantic equality: the decoded f32 values (the only thing consumers
        // ever observe).
        self.len() == other.len() && self.to_vec() == other.to_vec()
    }
}

/// Group-quantized int4 weight (the Phase-4 decode-bandwidth wedge, plan §9).
///
/// int4 nibbles are packed two-per-byte in `packed` (`n * k / 2` bytes, `k`
/// even), with one f32 `scale` per `group_size`-element group along the
/// contraction dim. `tier` records the per-tensor precision choice from the
/// rate-distortion allocator. No CPU has an int4 MAC, so the GEMM paths unpack
/// int4 -> int8 in-register and feed the same int8 MAC; this struct only
/// *carries* the packing. The payload/scale storages may be zero-copy views
/// into the loaded blob ([`PackedBytes::Shared`] / [`GroupScales::RawLe`]) so
/// the wasm artifact's expert bulk is never duplicated in residency (bd-4l71).
#[derive(Debug, Clone, PartialEq)]
pub struct QInt4 {
    /// Two int4 nibbles per byte, row-major `[n, k/2]`.
    pub packed: PackedBytes,
    /// Per-group scales, length `n * (k / group_size)`.
    pub scales: GroupScales,
    /// Output dimension.
    pub n: usize,
    /// Input dimension (contraction length; must be even and a multiple of
    /// `group_size`).
    pub k: usize,
    /// Elements per quantization group along the contraction dim (typ. 16–32).
    pub group_size: usize,
    /// Per-tensor precision tier from the water-filling allocator (plan §9.7).
    pub tier: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat_from_vec_roundtrips() {
        let m = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.shape(), (2, 3));
        assert_eq!(m.len(), 6);
        assert!(!m.is_empty());
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 2), 3.0);
        assert_eq!(m.get(1, 0), 4.0);
        assert_eq!(m.get(1, 2), 6.0);
    }

    #[test]
    fn mat_zeros_and_new_agree() {
        let z = Mat::zeros(3, 4);
        let n = Mat::new(3, 4);
        assert_eq!(z, n);
        assert!(z.data.iter().all(|&v| v == 0.0));
        assert_eq!(z.len(), 12);
    }

    #[test]
    fn mat_row_is_contiguous() {
        let m = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(m.row(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn mat_set_and_row_mut() {
        let mut m = Mat::zeros(2, 2);
        m.set(0, 1, 7.0);
        assert_eq!(m.get(0, 1), 7.0);
        m.row_mut(1).copy_from_slice(&[8.0, 9.0]);
        assert_eq!(m.row(1), &[8.0, 9.0]);
    }

    #[test]
    #[should_panic(expected = "data len")]
    fn mat_from_vec_rejects_bad_len() {
        let _ = Mat::from_vec(2, 3, vec![1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "Mat::from_vec: rows*cols overflow")]
    fn mat_from_vec_rejects_shape_overflow() {
        let _ = Mat::from_vec(usize::MAX, 2, Vec::new());
    }

    #[test]
    #[should_panic(expected = "Mat::zeros: rows*cols overflow")]
    fn mat_zeros_rejects_shape_overflow_before_allocating() {
        let _ = Mat::zeros(usize::MAX, 2);
    }

    #[test]
    fn qint8_new_validates_shape() {
        let q = QInt8::new(vec![1i8, 2, 3, 4, 5, 6], vec![0.1, 0.2], 2, 3);
        assert_eq!(q.n, 2);
        assert_eq!(q.k, 3);
        assert_eq!(q.w.len(), 6);
        assert_eq!(q.scales.len(), 2);
    }

    #[test]
    #[should_panic(expected = "w len")]
    fn qint8_rejects_bad_weight_len() {
        let _ = QInt8::new(vec![1i8, 2, 3], vec![0.1, 0.2], 2, 3);
    }

    #[test]
    #[should_panic(expected = "QInt8::new: rows*cols overflow")]
    fn qint8_rejects_shape_overflow() {
        let _ = QInt8::new(Vec::new(), Vec::new(), usize::MAX, 2);
    }

    #[test]
    fn qint4_placeholder_constructs() {
        // group_size 16 over k=16 => 1 group/row * n=2 = 2 scales; k/2=8 bytes/row.
        let q = QInt4 {
            packed: (0u8..16).collect::<Vec<u8>>().into(),
            scales: vec![0.1, 0.2].into(),
            n: 2,
            k: 16,
            group_size: 16,
            tier: 1,
        };
        assert_eq!(q.packed.len(), q.n * (q.k / 2));
        assert_eq!(q.scales.len(), q.n * (q.k / q.group_size));
    }

    #[test]
    fn group_scales_raw_le_decodes_identically_to_owned() {
        let values = vec![0.125f32, -3.5, 1.0e-3, 0.0, 7.25];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // An Owned storage and a decoded RawLe must agree on every observation.
        let owned = GroupScales::Owned(values.clone());
        assert_eq!(owned.len(), 5);
        assert_eq!(owned.to_vec(), values);
        let mut scratch = Vec::new();
        assert_eq!(owned.slice_f32(1, 4, &mut scratch), &values[1..4]);
        // RawLe is exercised through the loader in weights.rs tests (SharedBytes
        // needs a blob); here pin the byte layout the decode assumes.
        assert_eq!(bytes.len(), values.len() * 4);
        let decoded: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(decoded, values);
    }
}
