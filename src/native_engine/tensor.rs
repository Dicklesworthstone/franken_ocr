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
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row-major elements, length `rows * cols`.
    pub data: Vec<f32>,
}

impl Mat {
    /// Construct from an explicit shape + backing vector.
    ///
    /// # Panics
    /// Panics if `data.len() != rows * cols` (a shape/length contract
    /// violation is a programming error, caught early).
    #[must_use]
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Mat::from_vec: data len {} != rows*cols {}",
            data.len(),
            rows * cols
        );
        Self { rows, cols, data }
    }

    /// An uninitialized-shaped matrix filled with zeros.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0f32; rows * cols],
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
    /// Row-major `[n, k]` int8 weights.
    pub w: Vec<i8>,
    /// Per-output-channel scales, length `n`.
    pub scales: Vec<f32>,
    /// Output dimension (number of rows / output channels).
    pub n: usize,
    /// Input dimension (contraction length / number of columns).
    pub k: usize,
}

impl QInt8 {
    /// Construct from raw quantized parts.
    ///
    /// # Panics
    /// Panics on a length/shape mismatch (`w.len() != n*k` or
    /// `scales.len() != n`).
    #[must_use]
    pub fn new(w: Vec<i8>, scales: Vec<f32>, n: usize, k: usize) -> Self {
        assert_eq!(w.len(), n * k, "QInt8: w len {} != n*k {}", w.len(), n * k);
        assert_eq!(scales.len(), n, "QInt8: scales len {} != n {}", scales.len(), n);
        Self { w, scales, n, k }
    }
}

/// Group-quantized int4 weight (the Phase-4 decode-bandwidth wedge, plan §9).
///
/// Placeholder layout: int4 nibbles are packed two-per-byte in `packed`
/// (`n * k / 2` bytes, `k` even), with one f32 `scale` per `group_size`-element
/// group along the contraction dim. `tier` records the per-tensor precision
/// choice from the rate-distortion allocator. No CPU has an int4 MAC, so the
/// loader unpacks int4 -> int8 in-register and feeds the same int8 kernel; this
/// struct only *carries* the packing. Construction/dequant land with bd-3gaa.1.
#[derive(Debug, Clone, PartialEq)]
pub struct QInt4 {
    /// Two int4 nibbles per byte, row-major `[n, k/2]`.
    pub packed: Vec<u8>,
    /// Per-group scales, length `n * (k / group_size)`.
    pub scales: Vec<f32>,
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
    fn qint4_placeholder_constructs() {
        // group_size 2 over k=4 => 2 groups/row * n=2 = 4 scales; k/2=2 bytes/row.
        let q = QInt4 {
            packed: vec![0x21, 0x43, 0x65, 0x87],
            scales: vec![0.1, 0.2, 0.3, 0.4],
            n: 2,
            k: 4,
            group_size: 2,
            tier: 1,
        };
        assert_eq!(q.packed.len(), q.n * (q.k / 2));
        assert_eq!(q.scales.len(), q.n * (q.k / q.group_size));
    }
}
