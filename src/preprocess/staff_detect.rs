//! E5 (bd-3jo6.5.5): the TrOMR staff-detection front end — full page →
//! ordered single-staff crops (tromr-spec §7, v1 scope: printed/scanned
//! pages, bounded global and row-local shear deskew; camera dewarp and
//! barline-split chunking are filed follow-ups). Every applied transform is
//! retained as a typed operation plus exact pre/post Gray8 pixels and identities.
//!
//! Classical CV, pure Rust, no new dependencies:
//!
//! 1. ink gray plane (the DISC-007 rule: inverted alpha only when alpha
//!    varies; else cv2 fixed-point luma) → Otsu binarization;
//! 2. global deskew by shear: the angle in ±5° that MAXIMIZES the row-
//!    projection variance (staff lines align → sharp peaks), coarse 1° then
//!    fine 0.25°;
//! 3. horizontal-run row projection (runs span at least 15% of page width) →
//!    global candidates plus at most thirty-one bounded, overlapping staff-scale
//!    local-window passes over that same profile;
//! 4. groups of 5 consecutive bands with near-uniform spacing (≤ 25%
//!    deviation) become candidates, then deterministic global-first merging
//!    records every accepted, duplicate, and rejected candidate;
//! 5. an explicit residual-coverage check refuses unresolved staff-like ink;
//! 6. each staff is cropped full-width with a vertical margin of twice the
//!    line spacing × 2 (ledger lines, dynamics), clamped to the page.
//!
//! The detector report returns crops top-to-bottom with page bboxes, an
//! ordered candidate ledger, and residual evidence. The strict convenience
//! API returns only crops after proving no unresolved residual candidate.

use image::{DynamicImage, ImageDecoder, ImageEncoder};
use sha2::{Digest, Sha256};

use crate::error::{FocrError, FocrResult};

const TROMR_GRAY8_CROP_DOMAIN: &[u8] = b"franken_ocr.tromr.gray8_crop.v1\0";

/// Frozen description of the bounded whole-page vertical shear.
pub const TROMR_GLOBAL_DESKEW_TRANSFORM_CONTRACT: &str =
    "vertical_shear_nearest_white_fill_global_millidegrees_v1";
/// Frozen description of the bounded row-local vertical shear.
pub const TROMR_ROW_REFINEMENT_TRANSFORM_CONTRACT: &str =
    "vertical_shear_nearest_white_fill_row_local_millidegrees_v1";
/// Frozen description of the half-open extraction from the globally deskewed
/// page into one pre-refinement staff crop.
pub const TROMR_STAFF_CROP_TRANSFORM_CONTRACT: &str =
    "axis_aligned_half_open_crop_global_page_to_staff_v1";
/// Frozen description of the lossless white padding from a refined crop onto
/// the review/model-input canvas.
pub const TROMR_REVIEW_PADDING_TRANSFORM_CONTRACT: &str =
    "gray8_white_fill_tblr_padding_refined_crop_to_review_canvas_v1";
/// Schema carried by the provider-owned, replayable staff-geometry chain.
pub const TROMR_RETAINED_STAFF_GEOMETRY_SCHEMA_V1: &str =
    "franken_ocr.tromr.retained_staff_geometry.v1";

/// Closed coordinate spaces used by the TrOMR page-to-row geometry chain.
///
/// Crop-relative spaces deliberately do not carry a row index in the enum.
/// The owning [`TromrRetainedStaffCropGeometryV1`] supplies that index, which
/// prevents coordinates from one crop being silently compared to another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TromrGeometryCoordinateSpaceV1 {
    /// DISC-007 Gray8 pixels for the selected input page before deskew.
    SelectedPage,
    /// Same-size page raster after the selected global vertical shear.
    GloballyDeskewedPage,
    /// Half-open staff bbox extracted from the globally deskewed page.
    PreRefinementCrop,
    /// Same-size staff crop after optional row-local vertical shear.
    RefinedUnpaddedCrop,
    /// Refined crop positioned on the exact white review/inference canvas.
    ReviewCanvas,
}

impl TromrGeometryCoordinateSpaceV1 {
    /// Stable machine-readable spelling for receipts and robot surfaces.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SelectedPage => "selected_page",
            Self::GloballyDeskewedPage => "globally_deskewed_page",
            Self::PreRefinementCrop => "pre_refinement_crop",
            Self::RefinedUnpaddedCrop => "refined_unpadded_crop",
            Self::ReviewCanvas => "review_canvas",
        }
    }
}

/// Axis-aligned half-open pixel rectangle in one declared coordinate space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TromrPixelRectV1 {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl TromrPixelRectV1 {
    #[must_use]
    pub const fn from_bbox(bbox: (usize, usize, usize, usize)) -> Self {
        Self {
            x: bbox.0,
            y: bbox.1,
            width: bbox.2,
            height: bbox.3,
        }
    }

    #[must_use]
    pub const fn as_bbox(self) -> (usize, usize, usize, usize) {
        (self.x, self.y, self.width, self.height)
    }
}

/// Pixel-free identity of one tightly packed provider-owned Gray8 raster.
///
/// This is deliberately sufficient for an embedder to serialize and rebuild
/// a receipt without retaining nonselected crop pixels. It does not authorize
/// new pixels: provider code that still owns pixels must compare this value to
/// [`TromrGray8CropV1::artifact_identity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TromrGray8ArtifactIdentityV1 {
    pub width: u64,
    pub height: u64,
    pub row_stride_bytes: u64,
    pub byte_len: u64,
    pub pixels_sha256: [u8; 32],
    pub pixels_blake3: [u8; 32],
    pub identity_sha256: [u8; 32],
}

impl TromrGray8ArtifactIdentityV1 {
    fn from_tightly_packed(pixels: &[u8], width: usize, height: usize) -> FocrResult<Self> {
        if width == 0 || height == 0 {
            return Err(FocrError::Other(anyhow::anyhow!(
                "TrOMR Gray8 identity dimensions must be non-zero, got {width}x{height}"
            )));
        }
        let expected_len = width.checked_mul(height).ok_or_else(|| {
            FocrError::Other(anyhow::anyhow!("TrOMR Gray8 identity dimensions overflow"))
        })?;
        if pixels.len() != expected_len {
            return Err(FocrError::Other(anyhow::anyhow!(
                "TrOMR Gray8 identity has {} bytes, expected {expected_len} for {width}x{height}",
                pixels.len()
            )));
        }
        let width = u64::try_from(width)
            .map_err(|_| FocrError::Other(anyhow::anyhow!("TrOMR Gray8 width exceeds u64")))?;
        let height = u64::try_from(height)
            .map_err(|_| FocrError::Other(anyhow::anyhow!("TrOMR Gray8 height exceeds u64")))?;
        let byte_len = u64::try_from(pixels.len()).map_err(|_| {
            FocrError::Other(anyhow::anyhow!("TrOMR Gray8 byte length exceeds u64"))
        })?;
        let pixels_sha256 = Sha256::digest(pixels).into();
        let pixels_blake3 = *blake3::hash(pixels).as_bytes();
        let mut identity = Sha256::new();
        identity.update(TROMR_GRAY8_CROP_DOMAIN);
        identity.update(b"gray8\0top_to_bottom\0left_to_right\0");
        identity.update(width.to_le_bytes());
        identity.update(height.to_le_bytes());
        identity.update(width.to_le_bytes());
        identity.update(byte_len.to_le_bytes());
        identity.update(pixels);
        Ok(Self {
            width,
            height,
            row_stride_bytes: width,
            byte_len,
            pixels_sha256,
            pixels_blake3,
            identity_sha256: identity.finalize().into(),
        })
    }

    /// Validate the closed-world shape fields available without raw pixels.
    pub fn validate_shape(&self) -> FocrResult<()> {
        if self.width == 0
            || self.height == 0
            || self.row_stride_bytes != self.width
            || self
                .width
                .checked_mul(self.height)
                .is_none_or(|expected| expected != self.byte_len)
        {
            return Err(FocrError::FormatMismatch(
                "TrOMR Gray8 artifact identity has inconsistent dimensions or byte length".into(),
            ));
        }
        Ok(())
    }
}

/// Calculate the provider-frozen identity for one tightly packed Gray8 raster.
///
/// Encoding (`gray8`), row/column order, stride (`width`), and the identity
/// domain are fixed by this API. Callers supply only exact pixels and non-zero
/// dimensions; no arbitrary encoding or domain tag is accepted.
pub fn tromr_gray8_artifact_identity_v1(
    pixels: &[u8],
    width: usize,
    height: usize,
) -> FocrResult<TromrGray8ArtifactIdentityV1> {
    TromrGray8ArtifactIdentityV1::from_tightly_packed(pixels, width, height)
}

/// Verify exact tightly packed Gray8 bytes and dimensions against a
/// provider-frozen artifact identity.
pub fn verify_tromr_gray8_artifact_identity_v1(
    pixels: &[u8],
    width: usize,
    height: usize,
    expected: TromrGray8ArtifactIdentityV1,
) -> FocrResult<()> {
    let actual = tromr_gray8_artifact_identity_v1(pixels, width, height)?;
    if actual != expected {
        return Err(FocrError::FormatMismatch(
            "TrOMR Gray8 artifact identity does not match exact tightly packed pixels and dimensions"
                .into(),
        ));
    }
    Ok(())
}

/// Exact provider-owned Gray8 inference canvas for one TrOMR row attempt.
///
/// Pixels are retained, not merely summarized, so an embedder can materialize
/// the exact review crop without re-detecting a staff or invoking another image
/// implementation. The canonical byte order is tightly packed rows from top
/// to bottom, with columns ordered left to right.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TromrGray8CropV1 {
    /// Exact unsigned 8-bit grayscale pixels represented to TrOMR.
    pixels: Vec<u8>,
    /// Pixel columns per row.
    width: usize,
    /// Number of rows.
    height: usize,
    /// Bytes between consecutive row starts; exactly `width` in v1.
    row_stride_bytes: usize,
    /// SHA-256 over the exact `pixels` buffer only.
    pixels_sha256: [u8; 32],
    /// BLAKE3 over the exact `pixels` buffer only.
    pixels_blake3: [u8; 32],
    /// Domain-separated SHA-256 over encoding, dimensions, stride, row order,
    /// byte length, and the exact pixel bytes.
    identity_sha256: [u8; 32],
}

impl TromrGray8CropV1 {
    pub(crate) fn from_tightly_packed(
        pixels: Vec<u8>,
        width: usize,
        height: usize,
    ) -> FocrResult<Self> {
        let identity = tromr_gray8_artifact_identity_v1(&pixels, width, height)?;
        Ok(Self {
            pixels,
            width,
            height,
            row_stride_bytes: width,
            pixels_sha256: identity.pixels_sha256,
            pixels_blake3: identity.pixels_blake3,
            identity_sha256: identity.identity_sha256,
        })
    }

    /// Stable encoding label for receipt and robot surfaces.
    #[must_use]
    pub const fn encoding(&self) -> &'static str {
        "gray8"
    }

    /// Stable row traversal label for receipt and robot surfaces.
    #[must_use]
    pub const fn row_order(&self) -> &'static str {
        "top_to_bottom"
    }

    /// Stable column traversal label for receipt and robot surfaces.
    #[must_use]
    pub const fn column_order(&self) -> &'static str {
        "left_to_right"
    }

    /// Exact retained pixels in canonical row-major order.
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    #[must_use]
    pub const fn row_stride_bytes(&self) -> usize {
        self.row_stride_bytes
    }

    #[must_use]
    pub const fn pixels_sha256(&self) -> [u8; 32] {
        self.pixels_sha256
    }

    #[must_use]
    pub const fn pixels_blake3(&self) -> [u8; 32] {
        self.pixels_blake3
    }

    #[must_use]
    pub const fn identity_sha256(&self) -> [u8; 32] {
        self.identity_sha256
    }

    /// Pixel-free, dimensioned identity suitable for durable receipts.
    #[must_use]
    pub fn artifact_identity(&self) -> TromrGray8ArtifactIdentityV1 {
        TromrGray8ArtifactIdentityV1 {
            width: u64::try_from(self.width).unwrap_or(u64::MAX),
            height: u64::try_from(self.height).unwrap_or(u64::MAX),
            row_stride_bytes: u64::try_from(self.row_stride_bytes).unwrap_or(u64::MAX),
            byte_len: u64::try_from(self.pixels.len()).unwrap_or(u64::MAX),
            pixels_sha256: self.pixels_sha256,
            pixels_blake3: self.pixels_blake3,
            identity_sha256: self.identity_sha256,
        }
    }

    pub(crate) fn validate(&self) -> FocrResult<()> {
        verify_tromr_gray8_artifact_identity_v1(
            &self.pixels,
            self.width,
            self.height,
            self.artifact_identity(),
        )
    }

    pub(crate) fn into_tightly_packed(self) -> (Vec<u8>, usize, usize) {
        (self.pixels, self.width, self.height)
    }

    /// Deterministically encode the retained crop as a lossless grayscale PNG.
    ///
    /// # Errors
    /// Returns an error when dimensions exceed the PNG encoder limits or the
    /// provider encoder rejects the canonical Gray8 buffer.
    pub fn to_lossless_png(&self) -> FocrResult<Vec<u8>> {
        self.validate()?;
        let width = u32::try_from(self.width)
            .map_err(|_| FocrError::Other(anyhow::anyhow!("TrOMR crop width exceeds u32")))?;
        let height = u32::try_from(self.height)
            .map_err(|_| FocrError::Other(anyhow::anyhow!("TrOMR crop height exceeds u32")))?;
        let mut png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut png)
            .write_image(&self.pixels, width, height, image::ExtendedColorType::L8)
            .map_err(|error| {
                FocrError::Other(anyhow::anyhow!("encode TrOMR Gray8 crop as PNG: {error}"))
            })?;
        Ok(png)
    }

    /// Decode a provider-produced lossless PNG without color conversion.
    /// Only native grayscale 8-bit PNG is accepted, so verification cannot
    /// silently coerce RGB, alpha, or higher bit-depth review artifacts.
    pub fn from_lossless_png(png: &[u8]) -> FocrResult<Self> {
        let decoder = image::codecs::png::PngDecoder::new(std::io::Cursor::new(png))
            .map_err(|error| FocrError::InputDecode(format!("decode TrOMR crop PNG: {error}")))?;
        if decoder.color_type() != image::ColorType::L8 {
            return Err(FocrError::FormatMismatch(format!(
                "TrOMR crop PNG must be grayscale 8-bit (L8), got {:?}",
                decoder.color_type()
            )));
        }
        let (width, height) = decoder.dimensions();
        let byte_len = usize::try_from(decoder.total_bytes()).map_err(|_| {
            FocrError::FormatMismatch("TrOMR crop PNG decoded byte length exceeds usize".into())
        })?;
        let mut pixels = vec![0u8; byte_len];
        decoder
            .read_image(&mut pixels)
            .map_err(|error| FocrError::InputDecode(format!("read TrOMR crop PNG: {error}")))?;
        Self::from_tightly_packed(pixels, width as usize, height as usize)
    }
}

/// Provider-owned exact Gray8 pixels bound to one geometry coordinate space.
///
/// Unlike [`TromrGray8ArtifactIdentityV1`], this value retains the bytes needed
/// to replay every transform without re-reading or re-rendering the source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TromrGray8StageV1 {
    coordinate_space: TromrGeometryCoordinateSpaceV1,
    gray8: TromrGray8CropV1,
}

impl TromrGray8StageV1 {
    fn new(
        coordinate_space: TromrGeometryCoordinateSpaceV1,
        gray8: TromrGray8CropV1,
    ) -> FocrResult<Self> {
        gray8.validate()?;
        Ok(Self {
            coordinate_space,
            gray8,
        })
    }

    /// Bind an exact provider Gray8 raster to one declared geometry space.
    ///
    /// This is the reconstruction path for persisted provider PNGs. The space
    /// declaration is subsequently proven by a typed transform replay; this
    /// constructor alone does not claim that a transform occurred.
    pub fn from_gray8(
        coordinate_space: TromrGeometryCoordinateSpaceV1,
        gray8: TromrGray8CropV1,
    ) -> FocrResult<Self> {
        Self::new(coordinate_space, gray8)
    }

    /// Coordinate space in which these pixels are addressed.
    #[must_use]
    pub const fn coordinate_space(&self) -> TromrGeometryCoordinateSpaceV1 {
        self.coordinate_space
    }

    /// Exact retained provider-owned Gray8 raster.
    #[must_use]
    pub const fn gray8(&self) -> &TromrGray8CropV1 {
        &self.gray8
    }

    /// Pixel-free identity of the retained raster.
    #[must_use]
    pub fn artifact_identity(&self) -> TromrGray8ArtifactIdentityV1 {
        self.gray8.artifact_identity()
    }

    /// Verify the retained bytes against the embedded dimensions and hashes.
    pub fn validate(&self) -> FocrResult<()> {
        self.gray8.validate()
    }
}

/// Synthetic white margin added around a detected source crop before model
/// inference. The tuple ordering is deliberately named here rather than left
/// implicit so embedders cannot confuse canvas padding with page-space ink.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StaffPadding {
    /// White rows above the source crop.
    pub top: usize,
    /// White columns to the right of the source crop.
    pub right: usize,
    /// White rows below the source crop.
    pub bottom: usize,
    /// White columns to the left of the source crop.
    pub left: usize,
}

/// Exact nearest-neighbor vertical shear between two declared Gray8 spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TromrVerticalShearTransformV1 {
    pub transform_contract: &'static str,
    pub source_space: TromrGeometryCoordinateSpaceV1,
    pub target_space: TromrGeometryCoordinateSpaceV1,
    pub angle_millidegrees: i32,
    pub fill_gray8: u8,
}

impl TromrVerticalShearTransformV1 {
    #[must_use]
    pub const fn global_deskew(angle_millidegrees: i32) -> Self {
        Self {
            transform_contract: TROMR_GLOBAL_DESKEW_TRANSFORM_CONTRACT,
            source_space: TromrGeometryCoordinateSpaceV1::SelectedPage,
            target_space: TromrGeometryCoordinateSpaceV1::GloballyDeskewedPage,
            angle_millidegrees,
            fill_gray8: 255,
        }
    }

    #[must_use]
    pub const fn row_refinement(angle_millidegrees: i32) -> Self {
        Self {
            transform_contract: TROMR_ROW_REFINEMENT_TRANSFORM_CONTRACT,
            source_space: TromrGeometryCoordinateSpaceV1::PreRefinementCrop,
            target_space: TromrGeometryCoordinateSpaceV1::RefinedUnpaddedCrop,
            angle_millidegrees,
            fill_gray8: 255,
        }
    }

    /// Validate the closed transform definition independently of any pixels.
    pub fn validate(&self) -> FocrResult<()> {
        let global = self.transform_contract == TROMR_GLOBAL_DESKEW_TRANSFORM_CONTRACT
            && self.source_space == TromrGeometryCoordinateSpaceV1::SelectedPage
            && self.target_space == TromrGeometryCoordinateSpaceV1::GloballyDeskewedPage
            && (-5_000..=5_000).contains(&self.angle_millidegrees)
            && self.angle_millidegrees % 250 == 0;
        let row = self.transform_contract == TROMR_ROW_REFINEMENT_TRANSFORM_CONTRACT
            && self.source_space == TromrGeometryCoordinateSpaceV1::PreRefinementCrop
            && self.target_space == TromrGeometryCoordinateSpaceV1::RefinedUnpaddedCrop
            && (-1_500..=1_500).contains(&self.angle_millidegrees)
            && self.angle_millidegrees % 100 == 0
            && (self.angle_millidegrees == 0 || self.angle_millidegrees.unsigned_abs() >= 200);
        if self.fill_gray8 != 255 || (!global && !row) {
            return Err(FocrError::FormatMismatch(
                "TrOMR vertical-shear transform violates its coordinate, angle-grid, or fill contract"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Apply this exact provider transform to retained source pixels.
    pub fn apply(&self, source: &TromrGray8StageV1) -> FocrResult<TromrGray8StageV1> {
        self.validate()?;
        source.validate()?;
        if source.coordinate_space != self.source_space {
            return Err(FocrError::FormatMismatch(format!(
                "TrOMR vertical shear expected {} pixels, got {}",
                self.source_space.as_str(),
                source.coordinate_space.as_str()
            )));
        }
        let gray8 = source.gray8();
        let pixels = shear_gray_millidegrees(
            gray8.pixels(),
            gray8.width(),
            gray8.height(),
            self.angle_millidegrees,
        );
        TromrGray8StageV1::new(
            self.target_space,
            TromrGray8CropV1::from_tightly_packed(pixels, gray8.width(), gray8.height())?,
        )
    }

    /// Replay and require exact pixel, shape, and coordinate-space equality.
    pub fn validate_replay(
        &self,
        source: &TromrGray8StageV1,
        target: &TromrGray8StageV1,
    ) -> FocrResult<()> {
        target.validate()?;
        let replayed = self.apply(source)?;
        if &replayed != target {
            return Err(FocrError::FormatMismatch(
                "TrOMR vertical-shear replay differs from retained target pixels".into(),
            ));
        }
        Ok(())
    }
}

/// Exact half-open crop from the globally deskewed page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TromrCropTransformV1 {
    pub transform_contract: &'static str,
    pub source_space: TromrGeometryCoordinateSpaceV1,
    pub target_space: TromrGeometryCoordinateSpaceV1,
    pub source_rect: TromrPixelRectV1,
}

impl TromrCropTransformV1 {
    #[must_use]
    pub const fn staff_from_globally_deskewed_page(source_rect: TromrPixelRectV1) -> Self {
        Self {
            transform_contract: TROMR_STAFF_CROP_TRANSFORM_CONTRACT,
            source_space: TromrGeometryCoordinateSpaceV1::GloballyDeskewedPage,
            target_space: TromrGeometryCoordinateSpaceV1::PreRefinementCrop,
            source_rect,
        }
    }

    pub fn validate(&self) -> FocrResult<()> {
        if self.transform_contract != TROMR_STAFF_CROP_TRANSFORM_CONTRACT
            || self.source_space != TromrGeometryCoordinateSpaceV1::GloballyDeskewedPage
            || self.target_space != TromrGeometryCoordinateSpaceV1::PreRefinementCrop
            || self.source_rect.width == 0
            || self.source_rect.height == 0
        {
            return Err(FocrError::FormatMismatch(
                "TrOMR staff-crop transform violates its coordinate or non-empty-rectangle contract"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Extract the declared rectangle without resampling or color conversion.
    pub fn apply(&self, source: &TromrGray8StageV1) -> FocrResult<TromrGray8StageV1> {
        self.validate()?;
        source.validate()?;
        if source.coordinate_space != self.source_space {
            return Err(FocrError::FormatMismatch(format!(
                "TrOMR crop expected {} pixels, got {}",
                self.source_space.as_str(),
                source.coordinate_space.as_str()
            )));
        }
        let rect = self.source_rect;
        let x_end = rect.x.checked_add(rect.width).ok_or_else(|| {
            FocrError::FormatMismatch("TrOMR crop horizontal extent overflows".into())
        })?;
        let y_end = rect.y.checked_add(rect.height).ok_or_else(|| {
            FocrError::FormatMismatch("TrOMR crop vertical extent overflows".into())
        })?;
        let source_gray8 = source.gray8();
        if x_end > source_gray8.width() || y_end > source_gray8.height() {
            return Err(FocrError::FormatMismatch(
                "TrOMR crop rectangle exceeds the retained source raster".into(),
            ));
        }
        let output_len = rect.width.checked_mul(rect.height).ok_or_else(|| {
            FocrError::FormatMismatch("TrOMR crop output dimensions overflow".into())
        })?;
        let mut pixels = vec![0u8; output_len];
        for output_row in 0..rect.height {
            let source_row = rect.y + output_row;
            let source_start = source_row * source_gray8.width() + rect.x;
            let output_start = output_row * rect.width;
            pixels[output_start..output_start + rect.width]
                .copy_from_slice(&source_gray8.pixels()[source_start..source_start + rect.width]);
        }
        TromrGray8StageV1::new(
            self.target_space,
            TromrGray8CropV1::from_tightly_packed(pixels, rect.width, rect.height)?,
        )
    }

    pub fn validate_replay(
        &self,
        source: &TromrGray8StageV1,
        target: &TromrGray8StageV1,
    ) -> FocrResult<()> {
        target.validate()?;
        let replayed = self.apply(source)?;
        if &replayed != target {
            return Err(FocrError::FormatMismatch(
                "TrOMR crop replay differs from retained pre-refinement pixels".into(),
            ));
        }
        Ok(())
    }
}

/// Exact white-fill placement of a refined crop on its review canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TromrPaddingTransformV1 {
    pub transform_contract: &'static str,
    pub source_space: TromrGeometryCoordinateSpaceV1,
    pub target_space: TromrGeometryCoordinateSpaceV1,
    pub padding: StaffPadding,
    pub fill_gray8: u8,
}

impl TromrPaddingTransformV1 {
    #[must_use]
    pub const fn review_canvas(padding: StaffPadding) -> Self {
        Self {
            transform_contract: TROMR_REVIEW_PADDING_TRANSFORM_CONTRACT,
            source_space: TromrGeometryCoordinateSpaceV1::RefinedUnpaddedCrop,
            target_space: TromrGeometryCoordinateSpaceV1::ReviewCanvas,
            padding,
            fill_gray8: 255,
        }
    }

    pub fn validate(&self) -> FocrResult<()> {
        if self.transform_contract != TROMR_REVIEW_PADDING_TRANSFORM_CONTRACT
            || self.source_space != TromrGeometryCoordinateSpaceV1::RefinedUnpaddedCrop
            || self.target_space != TromrGeometryCoordinateSpaceV1::ReviewCanvas
            || self.fill_gray8 != 255
        {
            return Err(FocrError::FormatMismatch(
                "TrOMR review-padding transform violates its coordinate or fill contract".into(),
            ));
        }
        Ok(())
    }

    /// Place the retained source exactly at `(left, top)` on a white canvas.
    pub fn apply(&self, source: &TromrGray8StageV1) -> FocrResult<TromrGray8StageV1> {
        self.validate()?;
        source.validate()?;
        if source.coordinate_space != self.source_space {
            return Err(FocrError::FormatMismatch(format!(
                "TrOMR padding expected {} pixels, got {}",
                self.source_space.as_str(),
                source.coordinate_space.as_str()
            )));
        }
        let source_gray8 = source.gray8();
        let width = source_gray8
            .width()
            .checked_add(self.padding.left)
            .and_then(|value| value.checked_add(self.padding.right))
            .ok_or_else(|| FocrError::FormatMismatch("TrOMR padded width overflows".into()))?;
        let height = source_gray8
            .height()
            .checked_add(self.padding.top)
            .and_then(|value| value.checked_add(self.padding.bottom))
            .ok_or_else(|| FocrError::FormatMismatch("TrOMR padded height overflows".into()))?;
        let output_len = width.checked_mul(height).ok_or_else(|| {
            FocrError::FormatMismatch("TrOMR padded canvas dimensions overflow".into())
        })?;
        let mut pixels = vec![self.fill_gray8; output_len];
        for source_row in 0..source_gray8.height() {
            let target_row = self.padding.top + source_row;
            let target_start = target_row * width + self.padding.left;
            let source_start = source_row * source_gray8.width();
            pixels[target_start..target_start + source_gray8.width()].copy_from_slice(
                &source_gray8.pixels()[source_start..source_start + source_gray8.width()],
            );
        }
        TromrGray8StageV1::new(
            self.target_space,
            TromrGray8CropV1::from_tightly_packed(pixels, width, height)?,
        )
    }

    pub fn validate_replay(
        &self,
        source: &TromrGray8StageV1,
        target: &TromrGray8StageV1,
    ) -> FocrResult<()> {
        target.validate()?;
        let replayed = self.apply(source)?;
        if &replayed != target {
            return Err(FocrError::FormatMismatch(
                "TrOMR padding replay differs from retained review-canvas pixels".into(),
            ));
        }
        Ok(())
    }
}

/// Five ordered staff-line center rows in one explicit coordinate space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TromrStaffLineRowsV1 {
    pub coordinate_space: TromrGeometryCoordinateSpaceV1,
    pub y_rows: [usize; 5],
}

impl TromrStaffLineRowsV1 {
    /// Require the five rows to be strictly ordered, in-bounds, and expressed
    /// in the same coordinate space as `stage`.
    pub fn validate_for(&self, stage: &TromrGray8StageV1) -> FocrResult<()> {
        if self.coordinate_space != stage.coordinate_space
            || !self.y_rows.windows(2).all(|pair| pair[0] < pair[1])
            || self.y_rows.iter().any(|row| *row >= stage.gray8().height())
        {
            return Err(FocrError::FormatMismatch(
                "TrOMR staff-line rows violate their coordinate-space, ordering, or raster bounds"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Auditable geometry for one staff inference (bd-av64.16). `source_bbox`
/// always addresses real pixels on the deskewed page. `canvas_*` and
/// `padding` describe the separate in-memory image presented to TrOMR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaffCropGeometry {
    /// `(x, y, w, h)` addressing real pixels on the deskewed page.
    pub source_bbox: (usize, usize, usize, usize),
    /// Width of the in-memory image passed to preprocessing/model inference.
    pub canvas_width: usize,
    /// Height of the in-memory image passed to preprocessing/model inference.
    pub canvas_height: usize,
    /// Synthetic white margin positioning the source crop on that canvas.
    pub padding: StaffPadding,
}

impl StaffCropGeometry {
    /// Geometry for an inference image that has no synthetic padding.
    #[must_use]
    pub const fn unpadded(source_bbox: (usize, usize, usize, usize)) -> Self {
        Self {
            source_bbox,
            canvas_width: source_bbox.2,
            canvas_height: source_bbox.3,
            padding: StaffPadding {
                top: 0,
                right: 0,
                bottom: 0,
                left: 0,
            },
        }
    }
}

/// One detected staff: the inference-canvas ink-gray plane plus its distinct
/// page-space source geometry (post-deskew coordinates).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaffCrop {
    /// Row-major gray pixels (ink dark), `h x w`. Synthetic pad pixels are
    /// exactly white (`255`).
    pub gray: Vec<u8>,
    /// Inference-canvas width.
    pub w: usize,
    /// Inference-canvas height.
    pub h: usize,
    /// `(x, y, w, h)` of only the real source pixels on the deskewed page.
    /// Synthetic padding never changes this box.
    pub bbox: (usize, usize, usize, usize),
    /// The five staff-line center rows, inference-canvas-relative (top to
    /// bottom) -- the anchor for barline detection (bd-av64.4).
    pub lines: [usize; 5],
    /// Accepted detector line centers in the globally deskewed raster, before
    /// any row-local refinement or inference-canvas padding.
    pub globally_deskewed_raster_lines: [usize; 5],
    /// Synthetic white canvas padding, separate from [`Self::bbox`].
    pub padding: StaffPadding,
}

impl StaffCrop {
    /// Return the source-vs-canvas geometry carried into model inference.
    #[must_use]
    pub const fn geometry(&self) -> StaffCropGeometry {
        StaffCropGeometry {
            source_bbox: self.bbox,
            canvas_width: self.w,
            canvas_height: self.h,
            padding: self.padding,
        }
    }

    /// Snapshot the exact refined/padded inference canvas before its pixels
    /// are moved into an image buffer for TrOMR preprocessing.
    pub(crate) fn exact_gray8(&self) -> FocrResult<TromrGray8CropV1> {
        TromrGray8CropV1::from_tightly_packed(self.gray.clone(), self.w, self.h)
    }
}

/// Provider-owned exact pixel and coordinate chain for one accepted staff row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TromrRetainedStaffCropGeometryV1 {
    /// Stable top-to-bottom accepted-crop index.
    pub crop_index: usize,
    /// Staff-line rows in the globally deskewed page.
    pub globally_deskewed_staff_lines: TromrStaffLineRowsV1,
    /// Exact page-to-crop extraction.
    pub crop_transform: TromrCropTransformV1,
    /// Exact extracted pixels before row-local refinement.
    pub pre_refinement_crop: TromrGray8StageV1,
    /// Staff-line rows after translation into the extracted crop.
    pub pre_refinement_staff_lines: TromrStaffLineRowsV1,
    /// Exact optional row-local shear.
    pub row_refinement_transform: TromrVerticalShearTransformV1,
    /// Exact pixels after row-local refinement and before padding.
    pub refined_unpadded_crop: TromrGray8StageV1,
    /// Re-detected staff-line rows in the refined crop.
    pub refined_unpadded_staff_lines: TromrStaffLineRowsV1,
    /// Exact white-fill placement on the review canvas.
    pub padding_transform: TromrPaddingTransformV1,
    /// Exact review/model-input canvas before TrOMR resize.
    pub review_canvas: TromrGray8StageV1,
    /// Staff-line rows translated onto the review canvas.
    pub review_canvas_staff_lines: TromrStaffLineRowsV1,
}

impl TromrRetainedStaffCropGeometryV1 {
    fn validate_against_page(
        &self,
        expected_crop_index: usize,
        globally_deskewed_page: &TromrGray8StageV1,
    ) -> FocrResult<()> {
        if self.crop_index != expected_crop_index {
            return Err(FocrError::FormatMismatch(
                "TrOMR retained crop index is not canonical top-to-bottom order".into(),
            ));
        }
        self.globally_deskewed_staff_lines
            .validate_for(globally_deskewed_page)?;
        self.crop_transform
            .validate_replay(globally_deskewed_page, &self.pre_refinement_crop)?;
        self.pre_refinement_staff_lines
            .validate_for(&self.pre_refinement_crop)?;
        self.row_refinement_transform
            .validate_replay(&self.pre_refinement_crop, &self.refined_unpadded_crop)?;
        self.refined_unpadded_staff_lines
            .validate_for(&self.refined_unpadded_crop)?;
        self.padding_transform
            .validate_replay(&self.refined_unpadded_crop, &self.review_canvas)?;
        self.review_canvas_staff_lines
            .validate_for(&self.review_canvas)?;

        let rect = self.crop_transform.source_rect;
        let crop_bottom = rect.y.checked_add(rect.height).ok_or_else(|| {
            FocrError::FormatMismatch("TrOMR retained crop vertical extent overflows".into())
        })?;
        let mut expected_pre_refinement = [0usize; 5];
        for (index, page_row) in self
            .globally_deskewed_staff_lines
            .y_rows
            .iter()
            .copied()
            .enumerate()
        {
            if page_row < rect.y || page_row >= crop_bottom {
                return Err(FocrError::FormatMismatch(
                    "TrOMR globally deskewed staff line lies outside its retained crop".into(),
                ));
            }
            expected_pre_refinement[index] = page_row - rect.y;
        }
        let mut expected_review = [0usize; 5];
        for (index, row) in self
            .refined_unpadded_staff_lines
            .y_rows
            .iter()
            .copied()
            .enumerate()
        {
            expected_review[index] = row
                .checked_add(self.padding_transform.padding.top)
                .ok_or_else(|| {
                    FocrError::FormatMismatch(
                        "TrOMR retained review staff-line translation overflows".into(),
                    )
                })?;
        }
        if self.globally_deskewed_staff_lines.coordinate_space
            != TromrGeometryCoordinateSpaceV1::GloballyDeskewedPage
            || self.pre_refinement_staff_lines.coordinate_space
                != TromrGeometryCoordinateSpaceV1::PreRefinementCrop
            || self.refined_unpadded_staff_lines.coordinate_space
                != TromrGeometryCoordinateSpaceV1::RefinedUnpaddedCrop
            || self.review_canvas_staff_lines.coordinate_space
                != TromrGeometryCoordinateSpaceV1::ReviewCanvas
            || self.pre_refinement_staff_lines.y_rows != expected_pre_refinement
            || (self.row_refinement_transform.angle_millidegrees == 0
                && self.refined_unpadded_staff_lines.y_rows
                    != self.pre_refinement_staff_lines.y_rows)
            || self.review_canvas_staff_lines.y_rows != expected_review
        {
            return Err(FocrError::FormatMismatch(
                "TrOMR retained staff-line coordinate chain is inconsistent".into(),
            ));
        }
        Ok(())
    }
}

/// Complete provider-owned page-to-row geometry and pixel chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TromrRetainedStaffDetectionGeometryV1 {
    pub schema_version: &'static str,
    /// Exact selected-page DISC-007 Gray8 pixels.
    pub selected_page: TromrGray8StageV1,
    /// Exact selected whole-page shear.
    pub global_deskew_transform: TromrVerticalShearTransformV1,
    /// Exact globally deskewed page pixels from which every staff is cropped.
    pub globally_deskewed_page: TromrGray8StageV1,
    /// Accepted staff rows in canonical top-to-bottom order.
    pub crops: Vec<TromrRetainedStaffCropGeometryV1>,
}

impl TromrRetainedStaffDetectionGeometryV1 {
    /// Replay and validate the entire retained provider-owned geometry chain.
    pub fn validate(&self) -> FocrResult<()> {
        if self.schema_version != TROMR_RETAINED_STAFF_GEOMETRY_SCHEMA_V1
            || self.selected_page.coordinate_space != TromrGeometryCoordinateSpaceV1::SelectedPage
            || self.globally_deskewed_page.coordinate_space
                != TromrGeometryCoordinateSpaceV1::GloballyDeskewedPage
        {
            return Err(FocrError::FormatMismatch(
                "TrOMR retained geometry has an unknown schema or page coordinate space".into(),
            ));
        }
        self.global_deskew_transform
            .validate_replay(&self.selected_page, &self.globally_deskewed_page)?;
        if self.crops.windows(2).any(|pair| {
            pair[0].crop_transform.source_rect.y >= pair[1].crop_transform.source_rect.y
        }) {
            return Err(FocrError::FormatMismatch(
                "TrOMR retained crops are not in strict top-to-bottom page order".into(),
            ));
        }
        for (index, crop) in self.crops.iter().enumerate() {
            crop.validate_against_page(index, &self.globally_deskewed_page)?;
        }
        Ok(())
    }
}

/// Whole-page Gray8 transformation selected by the detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TromrGlobalDeskewEvidenceV1 {
    pub transform_contract: &'static str,
    pub angle_millidegrees: i32,
    pub input_gray8: TromrGray8ArtifactIdentityV1,
    pub globally_deskewed_gray8: TromrGray8ArtifactIdentityV1,
}

impl TromrGlobalDeskewEvidenceV1 {
    /// Validate the bounded integer search grid and raster shape chain.
    pub fn validate(&self) -> FocrResult<()> {
        self.input_gray8.validate_shape()?;
        self.globally_deskewed_gray8.validate_shape()?;
        if self.transform_contract != TROMR_GLOBAL_DESKEW_TRANSFORM_CONTRACT
            || !(-5_000..=5_000).contains(&self.angle_millidegrees)
            || self.angle_millidegrees % 250 != 0
            || self.input_gray8.width != self.globally_deskewed_gray8.width
            || self.input_gray8.height != self.globally_deskewed_gray8.height
            || (self.angle_millidegrees == 0 && self.input_gray8 != self.globally_deskewed_gray8)
        {
            return Err(FocrError::FormatMismatch(
                "TrOMR global deskew evidence violates its bounded transform contract".into(),
            ));
        }
        Ok(())
    }
}

/// One crop's post-extraction, pre-letterbox row-local refinement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TromrRowRefinementEvidenceV1 {
    pub transform_contract: &'static str,
    pub angle_millidegrees: i32,
    pub source_crop_before_refinement_gray8: TromrGray8ArtifactIdentityV1,
    pub refined_unpadded_crop_gray8: TromrGray8ArtifactIdentityV1,
}

impl TromrRowRefinementEvidenceV1 {
    /// Validate the bounded integer search grid and unpadded shape chain.
    pub fn validate(&self) -> FocrResult<()> {
        self.source_crop_before_refinement_gray8.validate_shape()?;
        self.refined_unpadded_crop_gray8.validate_shape()?;
        let angle = self.angle_millidegrees;
        if self.transform_contract != TROMR_ROW_REFINEMENT_TRANSFORM_CONTRACT
            || !(-1_500..=1_500).contains(&angle)
            || angle % 100 != 0
            || (angle != 0 && angle.unsigned_abs() < 200)
            || self.source_crop_before_refinement_gray8.width
                != self.refined_unpadded_crop_gray8.width
            || self.source_crop_before_refinement_gray8.height
                != self.refined_unpadded_crop_gray8.height
            || (angle == 0
                && self.source_crop_before_refinement_gray8 != self.refined_unpadded_crop_gray8)
        {
            return Err(FocrError::FormatMismatch(
                "TrOMR row refinement evidence violates its bounded transform contract".into(),
            ));
        }
        Ok(())
    }
}

/// Pixel-free retained evidence for one accepted detector crop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TromrDetectedCropEvidenceV1 {
    pub geometry: StaffCropGeometry,
    pub globally_deskewed_raster_lines: [usize; 5],
    pub review_crop_staff_lines_y_in_canvas: [usize; 5],
    pub row_refinement: TromrRowRefinementEvidenceV1,
    pub review_crop_gray8: TromrGray8ArtifactIdentityV1,
}

impl TromrDetectedCropEvidenceV1 {
    fn validate(&self, global: TromrGray8ArtifactIdentityV1) -> FocrResult<()> {
        self.row_refinement.validate()?;
        self.review_crop_gray8.validate_shape()?;
        let geometry = self.geometry;
        let source_width = u64::try_from(geometry.source_bbox.2).unwrap_or(u64::MAX);
        let source_height = u64::try_from(geometry.source_bbox.3).unwrap_or(u64::MAX);
        if geometry.source_bbox.2 == 0
            || geometry.source_bbox.3 == 0
            || geometry
                .source_bbox
                .0
                .checked_add(geometry.source_bbox.2)
                .is_none_or(|right| right > usize::try_from(global.width).unwrap_or(usize::MAX))
            || geometry
                .source_bbox
                .1
                .checked_add(geometry.source_bbox.3)
                .is_none_or(|bottom| bottom > usize::try_from(global.height).unwrap_or(usize::MAX))
            || self
                .row_refinement
                .source_crop_before_refinement_gray8
                .width
                != source_width
            || self
                .row_refinement
                .source_crop_before_refinement_gray8
                .height
                != source_height
            || self.review_crop_gray8.width
                != u64::try_from(geometry.canvas_width).unwrap_or(u64::MAX)
            || self.review_crop_gray8.height
                != u64::try_from(geometry.canvas_height).unwrap_or(u64::MAX)
            || geometry
                .source_bbox
                .2
                .checked_add(geometry.padding.left)
                .and_then(|value| value.checked_add(geometry.padding.right))
                != Some(geometry.canvas_width)
            || geometry
                .source_bbox
                .3
                .checked_add(geometry.padding.top)
                .and_then(|value| value.checked_add(geometry.padding.bottom))
                != Some(geometry.canvas_height)
            || !self
                .globally_deskewed_raster_lines
                .windows(2)
                .all(|pair| pair[0] < pair[1])
            || self.globally_deskewed_raster_lines.iter().any(|line| {
                *line < geometry.source_bbox.1
                    || *line >= geometry.source_bbox.1 + geometry.source_bbox.3
            })
            || !self
                .review_crop_staff_lines_y_in_canvas
                .windows(2)
                .all(|pair| pair[0] < pair[1])
            || self.review_crop_staff_lines_y_in_canvas.iter().any(|line| {
                *line < geometry.padding.top
                    || *line
                        >= geometry
                            .canvas_height
                            .saturating_sub(geometry.padding.bottom)
            })
        {
            return Err(FocrError::FormatMismatch(
                "TrOMR detected-crop evidence violates its coordinate or identity chain".into(),
            ));
        }
        Ok(())
    }
}

/// Projection scope that produced one five-line staff candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaffCandidateOrigin {
    /// The original whole-page projection.
    Global,
    /// One bounded local projection over the already deskewed page.
    Local {
        /// Stable zero-based window index, top to bottom.
        window_index: usize,
        /// Inclusive page-space start row.
        y_start: usize,
        /// Exclusive page-space end row.
        y_end: usize,
    },
}

impl StaffCandidateOrigin {
    /// Stable machine-readable origin kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Local { .. } => "local",
        }
    }
}

/// Final disposition of one geometrically assessed staff candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaffCandidateDisposition {
    /// Admitted to crop construction.
    Accepted,
    /// Geometrically invalid or explicitly classified as non-score ruling.
    Rejected,
    /// Valid, but already represented by a preferred earlier candidate.
    Duplicate,
}

impl StaffCandidateDisposition {
    /// Stable machine-readable disposition.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
            Self::Duplicate => "duplicate",
        }
    }
}

/// Auditable evidence for one consecutive five-line candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaffCandidateEvidence {
    /// Projection scope that proposed the candidate.
    pub origin: StaffCandidateOrigin,
    /// Row-profile peak within that scope.
    pub profile_peak: u32,
    /// Minimum row ink count used to form line bands.
    pub line_threshold: u32,
    /// Minimum horizontal run extent used to build this profile, in basis
    /// points of page width (1,500 means 15%).
    pub minimum_horizontal_span_basis_points: u16,
    /// Horizontal-run profile strengths at the five selected line centers.
    pub line_profile_strengths: [u32; 5],
    /// Weakest of the five selected line-profile strengths.
    pub profile_floor: u32,
    /// Sum of the five selected line-profile strengths.
    pub profile_sum: u64,
    /// Five page-space line centers, top to bottom.
    pub lines: [usize; 5],
    /// Inclusive/exclusive page-space vertical extent of the five centers.
    pub y_extent: (usize, usize),
    /// Mean inter-line spacing multiplied by 1,000.
    pub mean_spacing_milli: u32,
    /// Page-level reference spacing used only to choose among overlapping
    /// alternatives, multiplied by 1,000.
    pub spacing_reference_milli: u32,
    /// 0..=10,000 agreement with `spacing_reference_milli`.
    pub spacing_consistency_basis_points: u16,
    /// 0..=10,000 score; 10,000 means all four gaps are equal.
    pub uniformity_basis_points: u16,
    /// Merge/validation result.
    pub disposition: StaffCandidateDisposition,
    /// Stable machine-readable explanation for rejection or deduplication.
    pub reason: Option<&'static str>,
}

/// Coverage and unresolved-residual summary for one page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaffResidualEvidence {
    /// Profile ink on unique accepted plus unresolved staff-like line rows.
    pub staff_like_ink: u64,
    /// Portion of `staff_like_ink` accounted for by accepted rows.
    pub covered_staff_like_ink: u64,
    /// Covered portion on a 0..=10,000 scale.
    pub coverage_basis_points: u16,
    /// Near-uniform five-line groups not explained by an accepted row.
    pub unresolved_candidates: Vec<[usize; 5]>,
    /// True when publication must stop for explicit review.
    pub unresolved: bool,
}

/// Pixel-free, complete staff-detection ledger suitable for durable receipts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaffDetectionEvidenceV1 {
    pub global_deskew: TromrGlobalDeskewEvidenceV1,
    pub crops: Vec<TromrDetectedCropEvidenceV1>,
    pub candidates: Vec<StaffCandidateEvidence>,
    pub residual: StaffResidualEvidence,
}

impl StaffDetectionEvidenceV1 {
    /// Validate the detector census, coordinate chain, and residual arithmetic.
    pub fn validate(&self) -> FocrResult<()> {
        self.global_deskew.validate()?;
        for crop in &self.crops {
            crop.validate(self.global_deskew.globally_deskewed_gray8)?;
        }
        if !self
            .crops
            .windows(2)
            .all(|pair| pair[0].geometry.source_bbox.1 < pair[1].geometry.source_bbox.1)
        {
            return Err(FocrError::FormatMismatch(
                "TrOMR detected-crop evidence is not in strict page order".into(),
            ));
        }
        let accepted_lines = self
            .candidates
            .iter()
            .filter(|candidate| candidate.disposition == StaffCandidateDisposition::Accepted)
            .map(|candidate| candidate.lines)
            .collect::<Vec<_>>();
        let crop_lines = self
            .crops
            .iter()
            .map(|crop| crop.globally_deskewed_raster_lines)
            .collect::<Vec<_>>();
        if accepted_lines != crop_lines {
            return Err(FocrError::FormatMismatch(
                "TrOMR accepted-candidate census differs from retained crop evidence".into(),
            ));
        }
        let global_height = usize::try_from(self.global_deskew.globally_deskewed_gray8.height)
            .map_err(|_| {
                FocrError::FormatMismatch(
                    "TrOMR detector raster height exceeds this platform".into(),
                )
            })?;
        let candidate_order_key = |candidate: &StaffCandidateEvidence| {
            let origin = match candidate.origin {
                StaffCandidateOrigin::Global => (0usize, 0usize),
                StaffCandidateOrigin::Local { window_index, .. } => (1, window_index),
            };
            (
                candidate.lines[0],
                candidate.lines[4],
                origin,
                candidate.disposition.as_str(),
            )
        };
        if self
            .candidates
            .windows(2)
            .any(|pair| candidate_order_key(&pair[0]) > candidate_order_key(&pair[1]))
        {
            return Err(FocrError::FormatMismatch(
                "TrOMR candidate ledger is not in canonical page order".into(),
            ));
        }
        let mut spacing_references = std::collections::BTreeMap::<u16, u32>::new();
        for candidate in &self.candidates {
            let gaps = candidate
                .lines
                .windows(2)
                .map(|pair| pair[1] - pair[0])
                .collect::<Vec<_>>();
            let gap_sum = gaps.iter().sum::<usize>();
            let expected_mean_milli =
                u32::try_from(gap_sum.saturating_mul(1_000) / 4).unwrap_or(u32::MAX);
            let max_scaled_deviation = gaps
                .iter()
                .map(|gap| gap.saturating_mul(4).abs_diff(gap_sum))
                .max()
                .unwrap_or(usize::MAX);
            let expected_uniformity = if gap_sum == 0 {
                0
            } else {
                let penalty = max_scaled_deviation
                    .saturating_mul(10_000)
                    .checked_div(gap_sum)
                    .unwrap_or(10_000)
                    .min(10_000);
                u16::try_from(10_000 - penalty).unwrap_or(0)
            };
            let expected_threshold = match candidate.origin {
                StaffCandidateOrigin::Global => candidate.profile_peak / 2,
                StaffCandidateOrigin::Local { .. } => candidate.profile_peak.saturating_mul(9) / 20,
            }
            .max(1);
            let expected_spacing_consistency = if candidate.spacing_reference_milli == 0 {
                10_000
            } else {
                let penalty = u64::from(
                    candidate
                        .mean_spacing_milli
                        .abs_diff(candidate.spacing_reference_milli),
                )
                .saturating_mul(10_000)
                .checked_div(u64::from(candidate.spacing_reference_milli))
                .unwrap_or(10_000)
                .min(10_000);
                u16::try_from(10_000 - penalty).unwrap_or(0)
            };
            let allowed_reason = matches!(
                candidate.reason,
                None | Some("mean_spacing_below_two_rows")
                    | Some("spacing_deviation_exceeds_25_percent")
                    | Some("six_or_more_comparable_uniform_lines")
                    | Some("same_five_line_page_geometry")
                    | Some("overlapping_five_line_page_geometry")
                    | Some("horizontal_extent_below_15_percent")
            );
            if !candidate.lines.windows(2).all(|pair| pair[0] < pair[1])
                || candidate.lines[4] >= global_height
                || candidate.y_extent != (candidate.lines[0], candidate.lines[4] + 1)
                || candidate.profile_peak == 0
                || candidate.line_threshold != expected_threshold
                || candidate.line_profile_strengths.iter().any(|strength| {
                    *strength < candidate.line_threshold || *strength > candidate.profile_peak
                })
                || candidate.profile_floor
                    != candidate
                        .line_profile_strengths
                        .iter()
                        .copied()
                        .min()
                        .unwrap_or(0)
                || candidate.profile_sum
                    != candidate
                        .line_profile_strengths
                        .iter()
                        .map(|value| u64::from(*value))
                        .sum::<u64>()
                || candidate.mean_spacing_milli != expected_mean_milli
                || candidate.uniformity_basis_points != expected_uniformity
                || candidate.spacing_consistency_basis_points != expected_spacing_consistency
                || !matches!(
                    candidate.minimum_horizontal_span_basis_points,
                    STAFF_MIN_HORIZONTAL_SPAN_BPS | AUDIT_MIN_HORIZONTAL_SPAN_BPS
                )
                || !allowed_reason
                || (candidate.disposition == StaffCandidateDisposition::Accepted
                    && candidate.reason.is_some())
                || (candidate.disposition != StaffCandidateDisposition::Accepted
                    && candidate.reason.is_none())
            {
                return Err(FocrError::FormatMismatch(
                    "TrOMR candidate ledger contains inconsistent derived evidence".into(),
                ));
            }
            if let Some(reference) = spacing_references.insert(
                candidate.minimum_horizontal_span_basis_points,
                candidate.spacing_reference_milli,
            ) && reference != candidate.spacing_reference_milli
            {
                return Err(FocrError::FormatMismatch(
                    "TrOMR candidate ledger has inconsistent spacing references".into(),
                ));
            }
            if let StaffCandidateOrigin::Local { y_start, y_end, .. } = candidate.origin
                && (y_start >= y_end || candidate.lines[0] < y_start || candidate.lines[4] >= y_end)
            {
                return Err(FocrError::FormatMismatch(
                    "TrOMR local candidate lies outside its projection window".into(),
                ));
            }
        }
        let expected_coverage = if self.residual.staff_like_ink == 0 {
            10_000
        } else {
            u16::try_from(
                self.residual
                    .covered_staff_like_ink
                    .saturating_mul(10_000)
                    .checked_div(self.residual.staff_like_ink)
                    .unwrap_or(0)
                    .min(10_000),
            )
            .unwrap_or(0)
        };
        if self.residual.covered_staff_like_ink > self.residual.staff_like_ink
            || self.residual.coverage_basis_points != expected_coverage
            || self.residual.unresolved != !self.residual.unresolved_candidates.is_empty()
            || !self.residual.unresolved_candidates.iter().all(|lines| {
                lines.windows(2).all(|pair| pair[0] < pair[1])
                    && lines[4] < global_height
                    && self.candidates.iter().any(|candidate| {
                        candidate.lines == *lines
                            && candidate.disposition == StaffCandidateDisposition::Rejected
                            && candidate.reason == Some("spacing_deviation_exceeds_25_percent")
                            && candidate.uniformity_basis_points >= 6_000
                    })
            })
        {
            return Err(FocrError::FormatMismatch(
                "TrOMR residual evidence contains inconsistent completeness accounting".into(),
            ));
        }
        Ok(())
    }

    /// Refuse publication when the retained residual ledger is unresolved.
    pub fn require_complete(&self) -> FocrResult<()> {
        self.validate()?;
        if self.residual.unresolved {
            return Err(FocrError::Other(anyhow::anyhow!(
                "staff_detect: unresolved staff-like residual ink: {} candidate(s), coverage {}/10000; inspect detect_staves_with_evidence",
                self.residual.unresolved_candidates.len(),
                self.residual.coverage_basis_points
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn synthetic_complete_evidence_for_test(
    raster_width: usize,
    raster_height: usize,
    crops: &[(StaffCropGeometry, [usize; 5], [usize; 5])],
) -> StaffDetectionEvidenceV1 {
    let global = TromrGray8CropV1::from_tightly_packed(
        vec![255; raster_width * raster_height],
        raster_width,
        raster_height,
    )
    .expect("synthetic detector raster")
    .artifact_identity();
    let crop_evidence = crops
        .iter()
        .map(|(geometry, detector_lines, review_lines)| {
            let source = TromrGray8CropV1::from_tightly_packed(
                vec![255; geometry.source_bbox.2 * geometry.source_bbox.3],
                geometry.source_bbox.2,
                geometry.source_bbox.3,
            )
            .expect("synthetic source crop")
            .artifact_identity();
            let review = TromrGray8CropV1::from_tightly_packed(
                vec![255; geometry.canvas_width * geometry.canvas_height],
                geometry.canvas_width,
                geometry.canvas_height,
            )
            .expect("synthetic review crop")
            .artifact_identity();
            TromrDetectedCropEvidenceV1 {
                geometry: *geometry,
                globally_deskewed_raster_lines: *detector_lines,
                review_crop_staff_lines_y_in_canvas: *review_lines,
                row_refinement: TromrRowRefinementEvidenceV1 {
                    transform_contract: TROMR_ROW_REFINEMENT_TRANSFORM_CONTRACT,
                    angle_millidegrees: 0,
                    source_crop_before_refinement_gray8: source,
                    refined_unpadded_crop_gray8: source,
                },
                review_crop_gray8: review,
            }
        })
        .collect::<Vec<_>>();
    let candidates = crop_evidence
        .iter()
        .map(|crop| {
            let lines = crop.globally_deskewed_raster_lines;
            let mean_spacing_milli =
                u32::try_from((lines[4] - lines[0]).saturating_mul(250)).unwrap_or(u32::MAX);
            StaffCandidateEvidence {
                origin: StaffCandidateOrigin::Global,
                profile_peak: 1,
                line_threshold: 1,
                minimum_horizontal_span_basis_points: STAFF_MIN_HORIZONTAL_SPAN_BPS,
                line_profile_strengths: [1; 5],
                profile_floor: 1,
                profile_sum: 5,
                lines,
                y_extent: (lines[0], lines[4] + 1),
                mean_spacing_milli,
                spacing_reference_milli: mean_spacing_milli,
                spacing_consistency_basis_points: 10_000,
                uniformity_basis_points: 10_000,
                disposition: StaffCandidateDisposition::Accepted,
                reason: None,
            }
        })
        .collect::<Vec<_>>();
    let covered = u64::try_from(candidates.len().saturating_mul(5)).unwrap_or(u64::MAX);
    let evidence = StaffDetectionEvidenceV1 {
        global_deskew: TromrGlobalDeskewEvidenceV1 {
            transform_contract: TROMR_GLOBAL_DESKEW_TRANSFORM_CONTRACT,
            angle_millidegrees: 0,
            input_gray8: global,
            globally_deskewed_gray8: global,
        },
        crops: crop_evidence,
        candidates,
        residual: StaffResidualEvidence {
            staff_like_ink: covered,
            covered_staff_like_ink: covered,
            coverage_basis_points: 10_000,
            unresolved_candidates: Vec::new(),
            unresolved: false,
        },
    };
    evidence.validate().expect("synthetic detector evidence");
    evidence
}

#[cfg(test)]
pub(crate) fn synthetic_unresolved_evidence_for_test(
    raster_width: usize,
    raster_height: usize,
    crops: &[(StaffCropGeometry, [usize; 5], [usize; 5])],
    unresolved_lines: [usize; 5],
) -> StaffDetectionEvidenceV1 {
    let mut evidence = synthetic_complete_evidence_for_test(raster_width, raster_height, crops);
    let gaps = unresolved_lines
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    let gap_sum = gaps.iter().sum::<usize>();
    let mean_spacing_milli = u32::try_from(gap_sum.saturating_mul(1_000) / 4).unwrap_or(u32::MAX);
    let max_scaled_deviation = gaps
        .iter()
        .map(|gap| gap.saturating_mul(4).abs_diff(gap_sum))
        .max()
        .unwrap_or(usize::MAX);
    let uniformity_basis_points = if gap_sum == 0 {
        0
    } else {
        let penalty = max_scaled_deviation
            .saturating_mul(10_000)
            .checked_div(gap_sum)
            .unwrap_or(10_000)
            .min(10_000);
        u16::try_from(10_000 - penalty).unwrap_or(0)
    };
    let spacing_reference_milli = evidence
        .candidates
        .first()
        .map_or(mean_spacing_milli, |candidate| {
            candidate.spacing_reference_milli
        });
    let spacing_penalty = u64::from(mean_spacing_milli.abs_diff(spacing_reference_milli))
        .saturating_mul(10_000)
        .checked_div(u64::from(spacing_reference_milli.max(1)))
        .unwrap_or(10_000)
        .min(10_000);
    evidence.candidates.push(StaffCandidateEvidence {
        origin: StaffCandidateOrigin::Global,
        profile_peak: 1,
        line_threshold: 1,
        minimum_horizontal_span_basis_points: STAFF_MIN_HORIZONTAL_SPAN_BPS,
        line_profile_strengths: [1; 5],
        profile_floor: 1,
        profile_sum: 5,
        lines: unresolved_lines,
        y_extent: (unresolved_lines[0], unresolved_lines[4] + 1),
        mean_spacing_milli,
        spacing_reference_milli,
        spacing_consistency_basis_points: u16::try_from(10_000 - spacing_penalty).unwrap_or(0),
        uniformity_basis_points,
        disposition: StaffCandidateDisposition::Rejected,
        reason: Some("spacing_deviation_exceeds_25_percent"),
    });
    evidence.candidates.sort_by_key(|candidate| {
        let origin = match candidate.origin {
            StaffCandidateOrigin::Global => (0usize, 0usize),
            StaffCandidateOrigin::Local { window_index, .. } => (1, window_index),
        };
        (
            candidate.lines[0],
            candidate.lines[4],
            origin,
            candidate.disposition.as_str(),
        )
    });
    evidence.residual.staff_like_ink = evidence.residual.covered_staff_like_ink.saturating_add(5);
    evidence.residual.coverage_basis_points = u16::try_from(
        evidence
            .residual
            .covered_staff_like_ink
            .saturating_mul(10_000)
            .checked_div(evidence.residual.staff_like_ink)
            .unwrap_or(0),
    )
    .unwrap_or(0);
    evidence.residual.unresolved_candidates = vec![unresolved_lines];
    evidence.residual.unresolved = true;
    evidence
        .validate()
        .expect("synthetic unresolved detector evidence");
    evidence
}

/// Staff crops plus the evidence needed to audit page completeness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaffDetectionReport {
    /// Whole-page pre/post Gray8 identity and selected shear.
    pub global_deskew: TromrGlobalDeskewEvidenceV1,
    /// Exact provider-owned page/crop pixels and replayable geometry chain.
    pub retained_geometry: TromrRetainedStaffDetectionGeometryV1,
    /// Accepted top-to-bottom staff crops.
    pub crops: Vec<StaffCrop>,
    /// Pre/post unpadded identity and row-local shear for each crop.
    pub crop_refinements: Vec<TromrRowRefinementEvidenceV1>,
    /// Accepted, rejected, and duplicate candidate ledger.
    pub candidates: Vec<StaffCandidateEvidence>,
    /// Residual staff-like ink accounting.
    pub residual: StaffResidualEvidence,
}

impl StaffDetectionReport {
    /// Project the complete report into a pixel-free durable ledger.
    pub fn evidence(&self) -> FocrResult<StaffDetectionEvidenceV1> {
        self.retained_geometry.validate()?;
        if self.crops.len() != self.crop_refinements.len()
            || self.crops.len() != self.retained_geometry.crops.len()
            || self.global_deskew.input_gray8
                != self.retained_geometry.selected_page.artifact_identity()
            || self.global_deskew.globally_deskewed_gray8
                != self
                    .retained_geometry
                    .globally_deskewed_page
                    .artifact_identity()
            || self.global_deskew.transform_contract
                != self
                    .retained_geometry
                    .global_deskew_transform
                    .transform_contract
            || self.global_deskew.angle_millidegrees
                != self
                    .retained_geometry
                    .global_deskew_transform
                    .angle_millidegrees
        {
            return Err(FocrError::FormatMismatch(
                "staff detector retained geometry differs from its evidence census or global deskew"
                    .into(),
            ));
        }
        for ((crop, row_refinement), retained) in self
            .crops
            .iter()
            .zip(&self.crop_refinements)
            .zip(&self.retained_geometry.crops)
        {
            if retained.crop_transform.source_rect.as_bbox() != crop.bbox
                || retained.globally_deskewed_staff_lines.y_rows
                    != crop.globally_deskewed_raster_lines
                || retained.review_canvas_staff_lines.y_rows != crop.lines
                || retained.pre_refinement_crop.artifact_identity()
                    != row_refinement.source_crop_before_refinement_gray8
                || retained.refined_unpadded_crop.artifact_identity()
                    != row_refinement.refined_unpadded_crop_gray8
                || retained.row_refinement_transform.transform_contract
                    != row_refinement.transform_contract
                || retained.row_refinement_transform.angle_millidegrees
                    != row_refinement.angle_millidegrees
                || retained.padding_transform.padding != crop.padding
                || retained.review_canvas.artifact_identity()
                    != crop.exact_gray8()?.artifact_identity()
            {
                return Err(FocrError::FormatMismatch(
                    "staff detector retained crop pixels or coordinates differ from the public crop/evidence chain"
                        .into(),
                ));
            }
        }
        let crops = self
            .crops
            .iter()
            .zip(&self.crop_refinements)
            .map(|(crop, row_refinement)| {
                Ok(TromrDetectedCropEvidenceV1 {
                    geometry: crop.geometry(),
                    globally_deskewed_raster_lines: crop.globally_deskewed_raster_lines,
                    review_crop_staff_lines_y_in_canvas: crop.lines,
                    row_refinement: *row_refinement,
                    review_crop_gray8: crop.exact_gray8()?.artifact_identity(),
                })
            })
            .collect::<FocrResult<Vec<_>>>()?;
        let evidence = StaffDetectionEvidenceV1 {
            global_deskew: self.global_deskew,
            crops,
            candidates: self.candidates.clone(),
            residual: self.residual.clone(),
        };
        evidence.validate()?;
        Ok(evidence)
    }

    /// Require this report to account for all significant staff-like ink.
    /// The report remains inspectable when this returns an error.
    ///
    /// # Errors
    /// Unresolved staff-like residual ink survived the bounded local passes.
    pub fn require_complete(&self) -> FocrResult<()> {
        self.evidence()?.require_complete()
    }
}

/// Otsu's threshold over a 256-bin histogram. The returned `t` is the LAST
/// value of the dark class (ink = `v <= t` — dark pixels on a light page).
fn otsu_threshold(gray: &[u8]) -> u8 {
    let mut hist = [0u64; 256];
    for &v in gray {
        hist[v as usize] += 1;
    }
    let total: u64 = gray.len() as u64;
    let sum_all: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum();
    let (mut w_b, mut sum_b, mut best_t, mut best_var) = (0u64, 0.0f64, 0u8, -1.0f64);
    for (t, &count) in hist.iter().enumerate() {
        w_b += count;
        if w_b == 0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0 {
            break;
        }
        sum_b += t as f64 * count as f64;
        let m_b = sum_b / w_b as f64;
        let m_f = (sum_all - sum_b) / w_f as f64;
        let var = w_b as f64 * w_f as f64 * (m_b - m_f) * (m_b - m_f);
        if var > best_var {
            best_var = var;
            best_t = t as u8;
        }
    }
    best_t
}

/// Row-projection ink counts under a shear of `tan_a` (column x shifts down
/// by `tan_a · x` rows) — evaluated WITHOUT materializing the sheared image.
fn sheared_row_profile(ink: &[bool], w: usize, h: usize, tan_a: f64) -> Vec<u32> {
    let mut profile = vec![0u32; h];
    for y in 0..h {
        let row = &ink[y * w..(y + 1) * w];
        for (x, &is_ink) in row.iter().enumerate() {
            if is_ink {
                let shift = (tan_a * x as f64).round() as isize;
                let ny = y as isize - shift;
                if ny >= 0 && (ny as usize) < h {
                    profile[ny as usize] += 1;
                }
            }
        }
    }
    profile
}

/// Row projection of only genuinely horizontal ruling. A run must span at
/// least 15% of the page width, matching tromr-spec section 7; bounded gaps
/// tolerate scan dropout up to 0.5% of page width without letting separate
/// words join into a line. Both bounds are scale-derived rather than capped in
/// pixels so a 2x raster presents the same geometry to the detector.
/// This prevents dense text, beams, noteheads, and stems from becoming local-
/// threshold peaks. Multiple qualifying runs on one row are summed, so side-
/// by-side score snippets remain one row candidate while genuinely small
/// inline examples stay below completeness significance.
const STAFF_MIN_HORIZONTAL_SPAN_BPS: u16 = 1_500;
const AUDIT_MIN_HORIZONTAL_SPAN_BPS: u16 = 500;

fn horizontal_run_profile_with_minimum_span(
    ink: &[bool],
    w: usize,
    h: usize,
    minimum_span_basis_points: u16,
) -> Vec<u32> {
    let min_run = w
        .saturating_mul(usize::from(minimum_span_basis_points))
        .div_ceil(10_000)
        .max(2);
    let max_gap = w.div_ceil(200).max(2);
    let mut profile = vec![0u32; h];
    for y in 0..h {
        let row = &ink[y * w..(y + 1) * w];
        let mut start = None::<usize>;
        let mut last_ink = 0usize;
        for (x, &is_ink) in row.iter().enumerate() {
            if is_ink {
                if let Some(run_start) = start {
                    if x - last_ink - 1 > max_gap {
                        let span = last_ink - run_start + 1;
                        if span >= min_run {
                            profile[y] =
                                profile[y].saturating_add(u32::try_from(span).unwrap_or(u32::MAX));
                        }
                        start = Some(x);
                    }
                } else {
                    start = Some(x);
                }
                last_ink = x;
            }
        }
        if let Some(run_start) = start {
            let span = last_ink - run_start + 1;
            if span >= min_run {
                profile[y] = profile[y].saturating_add(u32::try_from(span).unwrap_or(u32::MAX));
            }
        }
    }
    profile
}

fn horizontal_run_profile(ink: &[bool], w: usize, h: usize) -> Vec<u32> {
    horizontal_run_profile_with_minimum_span(ink, w, h, STAFF_MIN_HORIZONTAL_SPAN_BPS)
}

fn profile_variance(profile: &[u32]) -> f64 {
    let n = profile.len() as f64;
    let mean = profile.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    profile
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n
}

/// The global deskew angle in integer millidegrees, bounded to ±5°:
/// coarse 1° sweep then a clamped 0.25° fine sweep.
fn deskew_angle_millidegrees(ink: &[bool], w: usize, h: usize) -> i32 {
    let score = |millidegrees: i32| -> f64 {
        let degrees = f64::from(millidegrees) / 1_000.0;
        profile_variance(&sheared_row_profile(ink, w, h, degrees.to_radians().tan()))
    };
    let mut best = (0i32, score(0));
    for d in -5..=5 {
        let millidegrees = d * 1_000;
        let s = score(millidegrees);
        if s > best.1 {
            best = (millidegrees, s);
        }
    }
    let coarse = best.0;
    let mut fine = best;
    for i in -3..=3 {
        let millidegrees = (coarse + i * 250).clamp(-5_000, 5_000);
        let s = score(millidegrees);
        if s > fine.1 {
            fine = (millidegrees, s);
        }
    }
    fine.0
}

/// Shear the gray plane vertically by `-tan(angle)·x` (fills with 255 =
/// paper). Adequate for the ≤5° global-deskew scope.
fn shear_gray(gray: &[u8], w: usize, h: usize, deg: f64) -> Vec<u8> {
    if deg == 0.0 {
        return gray.to_vec();
    }
    let tan_a = deg.to_radians().tan();
    let mut out = vec![255u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let shift = (tan_a * x as f64).round() as isize;
            let ny = y as isize - shift;
            if ny >= 0 && (ny as usize) < h {
                out[ny as usize * w + x] = gray[y * w + x];
            }
        }
    }
    out
}

fn shear_gray_millidegrees(gray: &[u8], w: usize, h: usize, millidegrees: i32) -> Vec<u8> {
    shear_gray(gray, w, h, f64::from(millidegrees) / 1_000.0)
}

/// Merge threshold-passing rows into bands, returning each band's center row.
fn line_band_centers(profile: &[u32], min_count: u32) -> Vec<usize> {
    let mut centers = Vec::new();
    let mut start: Option<usize> = None;
    for (y, &c) in profile.iter().enumerate() {
        if c >= min_count {
            start.get_or_insert(y);
        } else if let Some(s) = start.take() {
            centers.push((s + y - 1) / 2);
        }
    }
    if let Some(s) = start {
        centers.push((s + profile.len() - 1) / 2);
    }
    centers
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GeometryCandidate {
    lines: [usize; 5],
    mean_spacing_milli: u32,
    uniformity_basis_points: u16,
    accepted: bool,
    reason: Option<&'static str>,
}

fn assess_geometry(five: [usize; 5]) -> GeometryCandidate {
    let gaps = [
        five[1] - five[0],
        five[2] - five[1],
        five[3] - five[2],
        five[4] - five[3],
    ];
    let total_span = five[4] - five[0];
    let max_scaled_deviation = gaps
        .iter()
        .map(|gap| gap.saturating_mul(4).abs_diff(total_span))
        .max()
        .unwrap_or(usize::MAX);
    let mean_spacing_milli = u32::try_from(total_span.saturating_mul(250)).unwrap_or(u32::MAX);
    let uniformity_basis_points = if total_span == 0 {
        0
    } else {
        let penalty = max_scaled_deviation
            .saturating_mul(10_000)
            .checked_div(total_span)
            .unwrap_or(10_000)
            .min(10_000);
        u16::try_from(10_000 - penalty).unwrap_or(0)
    };
    let (accepted, reason) = if total_span < 8 {
        (false, Some("mean_spacing_below_two_rows"))
    } else if max_scaled_deviation > total_span / 4 {
        (false, Some("spacing_deviation_exceeds_25_percent"))
    } else {
        (true, None)
    };
    GeometryCandidate {
        lines: five,
        mean_spacing_milli,
        uniformity_basis_points,
        accepted,
        reason,
    }
}

fn candidate_profile_strength(profile: &[u32], lines: &[usize; 5]) -> (u32, u64) {
    let mut floor = u32::MAX;
    let mut total = 0u64;
    for &line in lines {
        let strength = profile.get(line).copied().unwrap_or(0);
        floor = floor.min(strength);
        total = total.saturating_add(u64::from(strength));
    }
    (floor, total)
}

/// Assess every consecutive five-center window, then recover candidates whose
/// endpoints and three interior centers form the same arithmetic progression
/// while skipping at most five nuisance peaks. The bounded endpoint search is
/// what keeps beams, ledger fragments, and notehead rows from displacing a
/// genuine staff line merely because they happen to clear the local threshold.
fn geometry_candidates(
    profile: &[u32],
    centers: &[usize],
    recover_interleaved_peaks: bool,
) -> Vec<GeometryCandidate> {
    let mut candidates = centers
        .windows(5)
        .map(|window| assess_geometry([window[0], window[1], window[2], window[3], window[4]]))
        .collect::<Vec<_>>();

    if !recover_interleaved_peaks {
        return candidates;
    }

    for start_index in 0..centers.len() {
        let first_end = start_index.saturating_add(4);
        let last_end = start_index
            .saturating_add(9)
            .min(centers.len().saturating_sub(1));
        for end_index in first_end..=last_end {
            let first = centers[start_index];
            let last = centers[end_index];
            let span = last.saturating_sub(first);
            if span < 8 {
                continue;
            }
            let tolerance = span.div_ceil(16).max(1);
            let interior = (1usize..4)
                .map(|slot| {
                    let expected =
                        first.saturating_add(span.saturating_mul(slot).saturating_add(2) / 4);
                    centers[start_index + 1..end_index]
                        .iter()
                        .copied()
                        .filter(|center| center.abs_diff(expected) <= tolerance)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            if interior.iter().any(Vec::is_empty) {
                continue;
            }

            let mut best = None::<GeometryCandidate>;
            for &second in &interior[0] {
                for &third in &interior[1] {
                    for &fourth in &interior[2] {
                        if !(first < second && second < third && third < fourth && fourth < last) {
                            continue;
                        }
                        let candidate = assess_geometry([first, second, third, fourth, last]);
                        if !candidate.accepted {
                            continue;
                        }
                        let strength = candidate_profile_strength(profile, &candidate.lines);
                        let replace = best.as_ref().is_none_or(|current| {
                            let current_strength =
                                candidate_profile_strength(profile, &current.lines);
                            (strength, candidate.uniformity_basis_points)
                                > (current_strength, current.uniformity_basis_points)
                                || ((strength, candidate.uniformity_basis_points)
                                    == (current_strength, current.uniformity_basis_points)
                                    && candidate.lines < current.lines)
                        });
                        if replace {
                            best = Some(candidate);
                        }
                    }
                }
            }
            if let Some(candidate) = best
                && !candidates
                    .iter()
                    .any(|existing| existing.lines == candidate.lines)
            {
                candidates.push(candidate);
            }
        }
    }
    candidates.sort_by_key(|candidate| candidate.lines);
    candidates
}

/// A sixth equally spaced peak is classified as non-score ruling only when
/// all six rows carry comparable ink. This keeps tablature/rules out without
/// discarding a legitimate five-line staff merely because a short ledger or
/// prose rule happens to sit one spacing away.
fn is_comparable_six_line_ruling(profile: &[u32], candidate: &GeometryCandidate) -> bool {
    if !candidate.accepted {
        return false;
    }
    let total_span = candidate.lines[4] - candidate.lines[0];
    let spacing = total_span.div_ceil(4).max(1);
    let tolerance = spacing.div_ceil(4).max(1);
    let comparable = |sixth: usize| {
        let mut min_ink = u32::MAX;
        let mut max_ink = 0u32;
        for row in candidate.lines.into_iter().chain(std::iter::once(sixth)) {
            let ink = profile.get(row).copied().unwrap_or(0);
            min_ink = min_ink.min(ink);
            max_ink = max_ink.max(ink);
        }
        max_ink > 0 && u64::from(min_ink) * 100 >= u64::from(max_ink) * 85
    };

    let strongest_in = |start: usize, end: usize| {
        profile
            .get(start..end)
            .and_then(|band| {
                band.iter()
                    .enumerate()
                    .max_by_key(|(_, ink)| *ink)
                    .map(|(offset, _)| start + offset)
            })
            .filter(|row| profile[*row] > 0)
    };
    let before = candidate.lines[0]
        .checked_sub(spacing)
        .and_then(|expected| {
            strongest_in(
                expected.saturating_sub(tolerance),
                expected
                    .saturating_add(tolerance)
                    .saturating_add(1)
                    .min(profile.len()),
            )
        });
    let expected_after = candidate.lines[4].saturating_add(spacing);
    let after = strongest_in(
        expected_after.saturating_sub(tolerance).min(profile.len()),
        expected_after
            .saturating_add(tolerance)
            .saturating_add(1)
            .min(profile.len()),
    );
    before.is_some_and(comparable) || after.is_some_and(comparable)
}

/// Group line centers into 5-line staves with near-uniform spacing (≤ 25%
/// max deviation from the mean gap). Greedy left-to-right — staves do not
/// overlap on a page.
fn group_staves(centers: &[usize]) -> Vec<[usize; 5]> {
    geometry_candidates(&[], centers, false)
        .into_iter()
        .filter(|candidate| candidate.accepted)
        .map(|candidate| candidate.lines)
        .collect()
}

/// Bounded staff-scale windows with 50% overlap. The global projection is
/// still evaluated first; these windows only change the local peak used for
/// line-band thresholding. The one-sixteenth-page target is clamped to at
/// least 128 rows so a complete five-line group fits across every boundary.
/// At most thirty-one windows are returned for any input.
fn local_detection_windows(height: usize) -> Vec<(usize, usize)> {
    if height < 2 {
        return Vec::new();
    }
    let window_height = height.div_ceil(16).max(128).min(height);
    if window_height >= height {
        return Vec::new();
    }
    let stride = window_height.div_ceil(2).max(1);
    let mut windows = Vec::with_capacity(31);
    let mut start = 0usize;
    loop {
        let end = start.saturating_add(window_height).min(height);
        windows.push((start, end));
        if end == height {
            break;
        }
        let next = start
            .saturating_add(stride)
            .min(height.saturating_sub(window_height));
        if next <= start {
            break;
        }
        start = next;
    }
    debug_assert!(windows.len() <= 31);
    windows
}

fn same_staff_geometry(left: &[usize; 5], right: &[usize; 5]) -> bool {
    let left_spacing = (left[4] - left[0]).div_ceil(4);
    let right_spacing = (right[4] - right[0]).div_ceil(4);
    let tolerance = left_spacing.min(right_spacing).div_ceil(4).max(2);
    left.iter()
        .zip(right)
        .all(|(a, b)| a.abs_diff(*b) <= tolerance)
}

/// Smallest canvas height whose exact aspect ratio fits a `target_h`-high,
/// `max_w`-wide model canvas. This intentionally does not rely on TrOMR's
/// later width rounding: no admitted source column is bought by a lossy
/// resize-rounding accident.
fn minimum_canvas_height(width: usize, target_h: usize, max_w: usize) -> FocrResult<usize> {
    if width == 0 || target_h == 0 || max_w == 0 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "staff_detect: invalid letterbox budget width={width} target_h={target_h} max_w={max_w}"
        )));
    }
    let scaled = width.checked_mul(target_h).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "staff_detect: letterbox height arithmetic overflow for width={width} target_h={target_h}"
        ))
    })?;
    Ok(scaled.div_ceil(max_w))
}

fn fits_canvas_budget(width: usize, height: usize, target_h: usize, max_w: usize) -> bool {
    minimum_canvas_height(width, target_h, max_w).is_ok_and(|needed| height >= needed)
}

fn translate_staff_lines(
    lines: &[usize; 5],
    crop_y: usize,
    crop_height: usize,
) -> FocrResult<[usize; 5]> {
    let mut translated = [0usize; 5];
    for (index, &line) in lines.iter().enumerate() {
        let local = line.checked_sub(crop_y).ok_or_else(|| {
            FocrError::Other(anyhow::anyhow!(
                "staff_detect: accepted line {line} precedes crop start {crop_y}"
            ))
        })?;
        if local >= crop_height {
            return Err(FocrError::Other(anyhow::anyhow!(
                "staff_detect: accepted line {line} lies outside crop {crop_y}..{}",
                crop_y.saturating_add(crop_height)
            )));
        }
        translated[index] = local;
    }
    Ok(translated)
}

/// Losslessly center one source crop on the minimum-height white canvas that
/// satisfies TrOMR's positional budget (bd-av64.16). An odd padding row goes
/// on the bottom, making the placement deterministic. The page-space bbox is
/// deliberately untouched.
fn letterbox_to_budget(crop: &mut StaffCrop, target_h: usize, max_w: usize) -> FocrResult<()> {
    let needed_h = minimum_canvas_height(crop.w, target_h, max_w)?;
    if crop.h >= needed_h {
        return Ok(());
    }

    let source_h = crop.h;
    let source_len = crop.w.checked_mul(source_h).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "staff_detect: source canvas length overflow for {}x{source_h}",
            crop.w
        ))
    })?;
    if crop.gray.len() != source_len {
        return Err(FocrError::Other(anyhow::anyhow!(
            "staff_detect: source canvas shape mismatch ({} bytes for {}x{source_h})",
            crop.gray.len(),
            crop.w
        )));
    }
    let canvas_len = crop.w.checked_mul(needed_h).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "staff_detect: inference canvas length overflow for {}x{needed_h}",
            crop.w
        ))
    })?;
    let total_pad = needed_h - source_h;
    let pad_top = total_pad / 2;
    let pad_bottom = total_pad - pad_top;
    if crop
        .lines
        .iter()
        .any(|&line| line.checked_add(pad_top).is_none())
    {
        return Err(FocrError::Other(anyhow::anyhow!(
            "staff_detect: staff-line translation overflow for top padding {pad_top}"
        )));
    }
    let mut canvas = vec![255u8; canvas_len];
    for row in 0..source_h {
        let source = &crop.gray[row * crop.w..(row + 1) * crop.w];
        let canvas_row = row + pad_top;
        canvas[canvas_row * crop.w..(canvas_row + 1) * crop.w].copy_from_slice(source);
    }
    crop.gray = canvas;
    crop.h = needed_h;
    crop.lines = crop.lines.map(|line| line + pad_top);
    crop.padding = StaffPadding {
        top: pad_top,
        right: 0,
        bottom: pad_bottom,
        left: 0,
    };
    Ok(())
}

fn append_projection_candidates(
    profile: &[u32],
    y_start: usize,
    y_end: usize,
    origin: StaffCandidateOrigin,
    evidence: &mut Vec<StaffCandidateEvidence>,
    accepted: &mut Vec<([usize; 5], usize)>,
) {
    let Some(window) = profile.get(y_start..y_end) else {
        return;
    };
    let peak = window.iter().copied().max().unwrap_or(0);
    if peak == 0 {
        return;
    }
    let threshold = match origin {
        StaffCandidateOrigin::Global => peak / 2,
        // A local band exists specifically to keep one darker neighboring
        // staff from suppressing a complete fainter row. Retaining the global
        // 50% threshold here left the fifth Mozart viola line just below the
        // cutoff (four lines were present, so no residual five-line candidate
        // could be formed). The bounded 45% floor still requires five actual
        // horizontal-run peaks plus the unchanged spacing and six-line gates.
        StaffCandidateOrigin::Local { .. } => peak.saturating_mul(9) / 20,
    }
    .max(1);
    let centers = line_band_centers(window, threshold)
        .into_iter()
        .map(|center| center + y_start)
        .collect::<Vec<_>>();
    let recover_interleaved_peaks = matches!(origin, StaffCandidateOrigin::Local { .. });
    for candidate in geometry_candidates(profile, &centers, recover_interleaved_peaks) {
        let six_line_ruling = is_comparable_six_line_ruling(profile, &candidate);
        let candidate_accepted = candidate.accepted && !six_line_ruling;
        let evidence_index = evidence.len();
        let line_profile_strengths = candidate
            .lines
            .map(|line| profile.get(line).copied().unwrap_or(0));
        let (profile_floor, profile_sum) = candidate_profile_strength(profile, &candidate.lines);
        evidence.push(StaffCandidateEvidence {
            origin,
            profile_peak: peak,
            line_threshold: threshold,
            minimum_horizontal_span_basis_points: STAFF_MIN_HORIZONTAL_SPAN_BPS,
            line_profile_strengths,
            profile_floor,
            profile_sum,
            lines: candidate.lines,
            y_extent: (candidate.lines[0], candidate.lines[4].saturating_add(1)),
            mean_spacing_milli: candidate.mean_spacing_milli,
            spacing_reference_milli: 0,
            spacing_consistency_basis_points: 0,
            uniformity_basis_points: candidate.uniformity_basis_points,
            disposition: if candidate_accepted {
                StaffCandidateDisposition::Accepted
            } else {
                StaffCandidateDisposition::Rejected
            },
            reason: if six_line_ruling {
                Some("six_or_more_comparable_uniform_lines")
            } else {
                candidate.reason
            },
        });
        if candidate_accepted {
            accepted.push((candidate.lines, evidence_index));
        }
    }
}

fn candidate_preferred(
    challenger: &StaffCandidateEvidence,
    incumbent: &StaffCandidateEvidence,
) -> bool {
    let origin_rank = |origin| match origin {
        StaffCandidateOrigin::Global => 1u8,
        StaffCandidateOrigin::Local { .. } => 0,
    };
    let challenger_score = (
        origin_rank(challenger.origin),
        challenger.spacing_consistency_basis_points,
        challenger.profile_floor,
        challenger.profile_sum,
        challenger.uniformity_basis_points,
    );
    let incumbent_score = (
        origin_rank(incumbent.origin),
        incumbent.spacing_consistency_basis_points,
        incumbent.profile_floor,
        incumbent.profile_sum,
        incumbent.uniformity_basis_points,
    );
    challenger_score > incumbent_score
        || (challenger_score == incumbent_score && challenger.lines < incumbent.lines)
}

#[derive(Clone, Default)]
struct CandidateSelection {
    evidence_indices: Vec<usize>,
    spacing_consistency_sum: u64,
    profile_floor_sum: u64,
    profile_strength_sum: u64,
    uniformity_sum: u64,
    global_count: usize,
}

impl CandidateSelection {
    fn with_candidate(
        mut self,
        evidence_index: usize,
        evidence: &[StaffCandidateEvidence],
    ) -> Self {
        let candidate = &evidence[evidence_index];
        self.evidence_indices.push(evidence_index);
        self.spacing_consistency_sum = self
            .spacing_consistency_sum
            .saturating_add(u64::from(candidate.spacing_consistency_basis_points));
        self.profile_floor_sum = self
            .profile_floor_sum
            .saturating_add(u64::from(candidate.profile_floor));
        self.profile_strength_sum = self
            .profile_strength_sum
            .saturating_add(candidate.profile_sum);
        self.uniformity_sum = self
            .uniformity_sum
            .saturating_add(u64::from(candidate.uniformity_basis_points));
        self.global_count = self.global_count.saturating_add(usize::from(matches!(
            candidate.origin,
            StaffCandidateOrigin::Global
        )));
        self
    }
}

fn selection_preferred(
    challenger: &CandidateSelection,
    incumbent: &CandidateSelection,
    evidence: &[StaffCandidateEvidence],
) -> bool {
    let challenger_score = (
        challenger.evidence_indices.len(),
        challenger.spacing_consistency_sum,
        challenger.profile_floor_sum,
        challenger.profile_strength_sum,
        challenger.uniformity_sum,
        challenger.global_count,
    );
    let incumbent_score = (
        incumbent.evidence_indices.len(),
        incumbent.spacing_consistency_sum,
        incumbent.profile_floor_sum,
        incumbent.profile_strength_sum,
        incumbent.uniformity_sum,
        incumbent.global_count,
    );
    if challenger_score != incumbent_score {
        return challenger_score > incumbent_score;
    }
    let mut challenger_lines = challenger
        .evidence_indices
        .iter()
        .map(|index| evidence[*index].lines)
        .collect::<Vec<_>>();
    let mut incumbent_lines = incumbent
        .evidence_indices
        .iter()
        .map(|index| evidence[*index].lines)
        .collect::<Vec<_>>();
    challenger_lines.sort_unstable();
    incumbent_lines.sort_unstable();
    challenger_lines < incumbent_lines
}

fn merge_staff_candidates(profile: &[u32]) -> (Vec<[usize; 5]>, Vec<StaffCandidateEvidence>) {
    let mut evidence = Vec::new();
    let mut proposed = Vec::new();
    append_projection_candidates(
        profile,
        0,
        profile.len(),
        StaffCandidateOrigin::Global,
        &mut evidence,
        &mut proposed,
    );
    for (window_index, (y_start, y_end)) in local_detection_windows(profile.len())
        .into_iter()
        .enumerate()
    {
        append_projection_candidates(
            profile,
            y_start,
            y_end,
            StaffCandidateOrigin::Local {
                window_index,
                y_start,
                y_end,
            },
            &mut evidence,
            &mut proposed,
        );
    }

    let mut reference_spacings = evidence
        .iter()
        .filter(|candidate| {
            candidate.disposition == StaffCandidateDisposition::Accepted
                && matches!(candidate.origin, StaffCandidateOrigin::Global)
        })
        .map(|candidate| candidate.mean_spacing_milli)
        .collect::<Vec<_>>();
    if reference_spacings.is_empty() {
        reference_spacings.extend(
            evidence
                .iter()
                .filter(|candidate| candidate.disposition == StaffCandidateDisposition::Accepted)
                .map(|candidate| candidate.mean_spacing_milli),
        );
    }
    reference_spacings.sort_unstable();
    let spacing_reference_milli = reference_spacings
        .get(reference_spacings.len() / 2)
        .copied()
        .unwrap_or(0);
    for candidate in &mut evidence {
        candidate.spacing_reference_milli = spacing_reference_milli;
        candidate.spacing_consistency_basis_points = if spacing_reference_milli == 0 {
            10_000
        } else {
            let penalty = u64::from(
                candidate
                    .mean_spacing_milli
                    .abs_diff(spacing_reference_milli),
            )
            .saturating_mul(10_000)
            .checked_div(u64::from(spacing_reference_milli))
            .unwrap_or(10_000)
            .min(10_000);
            u16::try_from(10_000 - penalty).unwrap_or(0)
        };
    }

    // Collapse the same page geometry first. Global evidence wins this narrow
    // equivalence class; local-only alternatives use the auditable strength
    // and uniformity scores above rather than window iteration order.
    let mut representatives = Vec::<([usize; 5], usize)>::new();
    for (lines, evidence_index) in proposed {
        if let Some(existing_index) = representatives
            .iter()
            .position(|(existing, _)| same_staff_geometry(existing, &lines))
        {
            let incumbent_evidence_index = representatives[existing_index].1;
            if candidate_preferred(
                &evidence[evidence_index],
                &evidence[incumbent_evidence_index],
            ) {
                evidence[incumbent_evidence_index].disposition =
                    StaffCandidateDisposition::Duplicate;
                evidence[incumbent_evidence_index].reason = Some("same_five_line_page_geometry");
                representatives[existing_index] = (lines, evidence_index);
            } else {
                evidence[evidence_index].disposition = StaffCandidateDisposition::Duplicate;
                evidence[evidence_index].reason = Some("same_five_line_page_geometry");
            }
        } else {
            representatives.push((lines, evidence_index));
        }
    }

    // Weighted interval scheduling admits the maximum number of vertically
    // noninterlacing staff rows. Confidence scores resolve alternative line
    // paths within a row, while count-first scoring prevents one wide bridge
    // candidate from erasing two legitimate adjacent rows.
    representatives.sort_by_key(|(lines, _)| (lines[4], lines[0], *lines));
    let mut best = vec![CandidateSelection::default(); representatives.len() + 1];
    for position in 1..=representatives.len() {
        let (_, evidence_index) = representatives[position - 1];
        let start = evidence[evidence_index].lines[0];
        let predecessor_count =
            representatives[..position - 1].partition_point(|(lines, _)| lines[4] < start);
        let include = best[predecessor_count]
            .clone()
            .with_candidate(evidence_index, &evidence);
        let exclude = best[position - 1].clone();
        best[position] = if selection_preferred(&include, &exclude, &evidence) {
            include
        } else {
            exclude
        };
    }

    let selected = &best[representatives.len()].evidence_indices;
    for (_, evidence_index) in &representatives {
        if !selected.contains(evidence_index) {
            evidence[*evidence_index].disposition = StaffCandidateDisposition::Rejected;
            evidence[*evidence_index].reason = Some("overlapping_five_line_page_geometry");
        }
    }
    let mut accepted = selected
        .iter()
        .map(|index| evidence[*index].lines)
        .collect::<Vec<_>>();
    accepted.sort_by_key(|lines| lines[0]);
    debug_assert!(accepted.windows(2).all(|pair| pair[0][4] < pair[1][0]));
    evidence.sort_by_key(|candidate| {
        let origin = match candidate.origin {
            StaffCandidateOrigin::Global => (0usize, 0usize),
            StaffCandidateOrigin::Local { window_index, .. } => (1, window_index),
        };
        (
            candidate.lines[0],
            candidate.lines[4],
            origin,
            candidate.disposition.as_str(),
        )
    });
    (accepted, evidence)
}

fn append_below_minimum_extent_evidence(
    main_profile: &[u32],
    audit_profile: &[u32],
    accepted: &[[usize; 5]],
    evidence: &mut Vec<StaffCandidateEvidence>,
) {
    let (_, audit_evidence) = merge_staff_candidates(audit_profile);
    for mut candidate in audit_evidence.into_iter().filter(|candidate| {
        candidate.disposition == StaffCandidateDisposition::Accepted
            && candidate
                .lines
                .iter()
                .all(|line| main_profile.get(*line).copied().unwrap_or(0) == 0)
            && !accepted
                .iter()
                .any(|staff| same_staff_geometry(staff, &candidate.lines))
    }) {
        if evidence
            .iter()
            .any(|existing| same_staff_geometry(&existing.lines, &candidate.lines))
        {
            continue;
        }
        candidate.disposition = StaffCandidateDisposition::Rejected;
        candidate.reason = Some("horizontal_extent_below_15_percent");
        candidate.minimum_horizontal_span_basis_points = AUDIT_MIN_HORIZONTAL_SPAN_BPS;
        evidence.push(candidate);
    }
    evidence.sort_by_key(|candidate| {
        let origin = match candidate.origin {
            StaffCandidateOrigin::Global => (0usize, 0usize),
            StaffCandidateOrigin::Local { window_index, .. } => (1, window_index),
        };
        (
            candidate.lines[0],
            candidate.lines[4],
            origin,
            candidate.disposition.as_str(),
        )
    });
}

fn residual_evidence(
    profile: &[u32],
    accepted: &[[usize; 5]],
    candidates: &[StaffCandidateEvidence],
) -> StaffResidualEvidence {
    let mut unresolved_candidates = Vec::<[usize; 5]>::new();
    for candidate in candidates {
        let significant = candidate.disposition == StaffCandidateDisposition::Rejected
            && candidate.reason == Some("spacing_deviation_exceeds_25_percent")
            && candidate.uniformity_basis_points >= 6_000;
        if !significant {
            continue;
        }
        let spacing = (candidate.lines[4] - candidate.lines[0]).div_ceil(4).max(1);
        let tolerance = spacing.div_ceil(4).max(2);
        let explained_lines = candidate
            .lines
            .iter()
            .filter(|line| {
                accepted.iter().any(|staff| {
                    staff
                        .iter()
                        .any(|accepted_line| accepted_line.abs_diff(**line) <= tolerance)
                })
            })
            .count();
        if explained_lines >= 3
            || unresolved_candidates
                .iter()
                .any(|existing| same_staff_geometry(existing, &candidate.lines))
        {
            continue;
        }
        unresolved_candidates.push(candidate.lines);
    }

    let mut accepted_rows = accepted
        .iter()
        .flat_map(|staff| staff.iter().copied())
        .collect::<Vec<_>>();
    accepted_rows.sort_unstable();
    accepted_rows.dedup_by(|left, right| left.abs_diff(*right) <= 2);
    let mut staff_like_rows = accepted_rows.clone();
    staff_like_rows.extend(
        unresolved_candidates
            .iter()
            .flat_map(|staff| staff.iter().copied()),
    );
    staff_like_rows.sort_unstable();
    staff_like_rows.dedup_by(|left, right| left.abs_diff(*right) <= 2);

    let staff_like_ink = staff_like_rows
        .iter()
        .filter_map(|row| profile.get(*row))
        .map(|ink| u64::from(*ink))
        .sum::<u64>();
    let covered_staff_like_ink = staff_like_rows
        .iter()
        .filter(|row| {
            accepted_rows
                .iter()
                .any(|accepted_row| accepted_row.abs_diff(**row) <= 2)
        })
        .filter_map(|row| profile.get(*row))
        .map(|ink| u64::from(*ink))
        .sum::<u64>();
    let coverage_basis_points = if staff_like_ink == 0 {
        10_000
    } else {
        u16::try_from(
            covered_staff_like_ink
                .saturating_mul(10_000)
                .checked_div(staff_like_ink)
                .unwrap_or(0)
                .min(10_000),
        )
        .unwrap_or(0)
    };
    StaffResidualEvidence {
        staff_like_ink,
        covered_staff_like_ink,
        coverage_basis_points,
        unresolved: !unresolved_candidates.is_empty(),
        unresolved_candidates,
    }
}

/// Detect staves on a full page (tromr-spec §7 v1). Returns crops
/// top-to-bottom; an empty result means "no 5-line staff found" (the caller
/// decides whether to fall back to whole-image recognition). A near-staff
/// residual that survives the bounded local passes refuses instead of being
/// silently published as a complete page.
///
/// # Errors
/// A degenerate image or unresolved staff-like residual ink.
pub fn detect_staves(img: &DynamicImage) -> FocrResult<Vec<StaffCrop>> {
    let report = detect_staves_with_evidence(img)?;
    report.require_complete()?;
    Ok(report.crops)
}

/// Detect staves and retain the complete candidate/residual ledger even when
/// the strict [`detect_staves`] wrapper would refuse publication.
///
/// # Errors
/// A degenerate (zero-sized) image or invalid crop geometry.
pub fn detect_staves_with_evidence(img: &DynamicImage) -> FocrResult<StaffDetectionReport> {
    let selected_page = TromrGray8StageV1::new(
        TromrGeometryCoordinateSpaceV1::SelectedPage,
        super::tromr_gray8_input(img)?,
    )?;
    let (w, h) = (
        selected_page.gray8().width(),
        selected_page.gray8().height(),
    );
    let input_gray8 = selected_page.artifact_identity();
    let thr = otsu_threshold(selected_page.gray8().pixels());
    let ink: Vec<bool> = selected_page
        .gray8()
        .pixels()
        .iter()
        .map(|&value| value <= thr)
        .collect();

    let angle_millidegrees = deskew_angle_millidegrees(&ink, w, h);
    let global_deskew_transform = TromrVerticalShearTransformV1::global_deskew(angle_millidegrees);
    let globally_deskewed_page = global_deskew_transform.apply(&selected_page)?;
    let globally_deskewed_gray8 = globally_deskewed_page.artifact_identity();
    let global_deskew = TromrGlobalDeskewEvidenceV1 {
        transform_contract: TROMR_GLOBAL_DESKEW_TRANSFORM_CONTRACT,
        angle_millidegrees,
        input_gray8,
        globally_deskewed_gray8,
    };
    global_deskew.validate()?;
    let gray = globally_deskewed_page.gray8().pixels();
    let ink: Vec<bool> = gray.iter().map(|&v| v <= thr).collect();

    let profile = horizontal_run_profile(&ink, w, h);
    let audit_profile =
        horizontal_run_profile_with_minimum_span(&ink, w, h, AUDIT_MIN_HORIZONTAL_SPAN_BPS);
    let peak = profile.iter().copied().max().unwrap_or(0);
    if peak == 0 {
        let mut candidates = Vec::new();
        append_below_minimum_extent_evidence(&profile, &audit_profile, &[], &mut candidates);
        let retained_geometry = TromrRetainedStaffDetectionGeometryV1 {
            schema_version: TROMR_RETAINED_STAFF_GEOMETRY_SCHEMA_V1,
            selected_page,
            global_deskew_transform,
            globally_deskewed_page,
            crops: Vec::new(),
        };
        return Ok(StaffDetectionReport {
            global_deskew,
            retained_geometry,
            crops: Vec::new(),
            crop_refinements: Vec::new(),
            candidates,
            residual: StaffResidualEvidence {
                staff_like_ink: 0,
                covered_staff_like_ink: 0,
                coverage_basis_points: 10_000,
                unresolved_candidates: Vec::new(),
                unresolved: false,
            },
        });
    }
    let (staves, mut candidates) = merge_staff_candidates(&profile);
    if staves.windows(2).any(|pair| pair[0][4] >= pair[1][0]) {
        return Err(FocrError::Other(anyhow::anyhow!(
            "staff_detect: internal noninterlacing invariant failed after candidate selection"
        )));
    }
    append_below_minimum_extent_evidence(&profile, &audit_profile, &staves, &mut candidates);
    let residual = residual_evidence(&profile, &staves, &candidates);

    // The model's positional budget: a crop resized to h=128 may span at
    // most 1280 columns (tromr POS_COLS * PATCH). Band geometry below is
    // shaped so real full-width systems FIT instead of hard-failing the
    // clamp (bd-av64.14; measured 2026-07-06: full-page-width bands with
    // page margins included blew the budget on every dense real scan, while
    // recognition quality is insensitive to GENEROUS bands and catastrophic
    // only for over-TIGHT ones).
    let budget = crate::native_engine::tromr::POS_COLS * crate::native_engine::tromr::PATCH;
    let img_h = crate::native_engine::tromr::IMG_H;

    let mut crops = Vec::with_capacity(staves.len());
    for (i, five) in staves.iter().enumerate() {
        let spacing = (five[4] - five[0]) as f64 / 4.0;
        let margin = (2.0 * spacing).round() as usize * 2;
        let y0_classic = five[0].saturating_sub(margin);
        let y1_classic = five[4].saturating_add(margin).saturating_add(1).min(h);

        // FIT-FIRST (bd-av64.14): when the classic full-width band already
        // fits the positional budget, keep it BIT-IDENTICAL to the historic
        // geometry — recognition is knife-edge sensitive to crop margins,
        // so geometry only changes where the old geometry hard-failed.
        if fits_canvas_budget(w, y1_classic - y0_classic, img_h, budget) {
            let ch = y1_classic - y0_classic;
            let lines = translate_staff_lines(five, y0_classic, ch)?;
            let mut crop = vec![0u8; ch * w];
            crop.copy_from_slice(&gray[y0_classic * w..y1_classic * w]);
            crops.push(StaffCrop {
                gray: crop,
                w,
                h: ch,
                bbox: (0, y0_classic, w, ch),
                lines,
                globally_deskewed_raster_lines: *five,
                padding: StaffPadding::default(),
            });
            continue;
        }

        // Over budget: (1) trim the band to its ink extent (staff lines span
        // the system, so any column inside holds >= 5 ink pixels; a >= 2
        // floor keeps specks from stretching the band to the page margins),
        // padded by ~2 line-spacings; (2) if still over, extend vertically
        // toward the neighbor midlines (measured: a staff at ~30% of the
        // frame still reads correctly, while width overflow is a hard
        // failure). If real page space still cannot satisfy the budget,
        // lossless white letterboxing below supplies only synthetic vertical
        // margin without borrowing neighboring ink (bd-av64.16).
        let lo_bound = if i > 0 {
            staves[i - 1][4].saturating_add(five[0]) / 2
        } else {
            0
        };
        let hi_bound = if i + 1 < staves.len() {
            five[4].saturating_add(staves[i + 1][0]).div_ceil(2)
        } else {
            h
        };
        let mut y0 = y0_classic.max(lo_bound);
        let mut y1 = y1_classic.min(hi_bound);
        let col_ink = |x: usize, a: usize, b: usize| -> usize {
            (a..b).filter(|&y| gray[y * w + x] <= thr).count()
        };
        let mut x0 = 0;
        while x0 < w && col_ink(x0, y0, y1) < 2 {
            x0 += 1;
        }
        let mut x1 = w;
        while x1 > x0 && col_ink(x1 - 1, y0, y1) < 2 {
            x1 -= 1;
        }
        if x1 <= x0 {
            x0 = 0;
            x1 = w;
        }
        let pad = (2.0 * spacing).round() as usize;
        x0 = x0.saturating_sub(pad);
        x1 = x1.saturating_add(pad).min(w);
        let step = spacing.max(1.0).round() as usize;
        while !fits_canvas_budget(x1 - x0, y1 - y0, img_h, budget)
            && (y0 > lo_bound || y1 < hi_bound)
        {
            y0 = y0.saturating_sub(step).max(lo_bound);
            y1 = (y1 + step).min(hi_bound).min(h);
        }

        let (ch, cw) = (y1 - y0, x1 - x0);
        let lines = translate_staff_lines(five, y0, ch)?;
        let mut crop = vec![0u8; ch * cw];
        for (row, y) in (y0..y1).enumerate() {
            crop[row * cw..(row + 1) * cw].copy_from_slice(&gray[y * w + x0..y * w + x1]);
        }
        crops.push(StaffCrop {
            gray: crop,
            w: cw,
            h: ch,
            bbox: (x0, y0, cw, ch),
            lines,
            globally_deskewed_raster_lines: *five,
            padding: StaffPadding::default(),
        });
    }
    let mut crop_refinements = Vec::with_capacity(crops.len());
    let mut retained_crops = Vec::with_capacity(crops.len());
    for (crop_index, crop) in crops.iter_mut().enumerate() {
        let globally_deskewed_staff_lines = TromrStaffLineRowsV1 {
            coordinate_space: TromrGeometryCoordinateSpaceV1::GloballyDeskewedPage,
            y_rows: crop.globally_deskewed_raster_lines,
        };
        let crop_transform = TromrCropTransformV1::staff_from_globally_deskewed_page(
            TromrPixelRectV1::from_bbox(crop.bbox),
        );
        let pre_refinement_crop = TromrGray8StageV1::new(
            TromrGeometryCoordinateSpaceV1::PreRefinementCrop,
            crop.exact_gray8()?,
        )?;
        let pre_refinement_staff_lines = TromrStaffLineRowsV1 {
            coordinate_space: TromrGeometryCoordinateSpaceV1::PreRefinementCrop,
            y_rows: crop.lines,
        };
        let row_refinement = refine_band_skew(crop)?;
        let row_refinement_transform =
            TromrVerticalShearTransformV1::row_refinement(row_refinement.angle_millidegrees);
        let refined_unpadded_crop = TromrGray8StageV1::new(
            TromrGeometryCoordinateSpaceV1::RefinedUnpaddedCrop,
            crop.exact_gray8()?,
        )?;
        let refined_unpadded_staff_lines = TromrStaffLineRowsV1 {
            coordinate_space: TromrGeometryCoordinateSpaceV1::RefinedUnpaddedCrop,
            y_rows: crop.lines,
        };
        letterbox_to_budget(crop, img_h, budget)?;
        let padding_transform = TromrPaddingTransformV1::review_canvas(crop.padding);
        let review_canvas = TromrGray8StageV1::new(
            TromrGeometryCoordinateSpaceV1::ReviewCanvas,
            crop.exact_gray8()?,
        )?;
        let review_canvas_staff_lines = TromrStaffLineRowsV1 {
            coordinate_space: TromrGeometryCoordinateSpaceV1::ReviewCanvas,
            y_rows: crop.lines,
        };
        crop_refinements.push(row_refinement);
        retained_crops.push(TromrRetainedStaffCropGeometryV1 {
            crop_index,
            globally_deskewed_staff_lines,
            crop_transform,
            pre_refinement_crop,
            pre_refinement_staff_lines,
            row_refinement_transform,
            refined_unpadded_crop,
            refined_unpadded_staff_lines,
            padding_transform,
            review_canvas,
            review_canvas_staff_lines,
        });
    }
    let retained_geometry = TromrRetainedStaffDetectionGeometryV1 {
        schema_version: TROMR_RETAINED_STAFF_GEOMETRY_SCHEMA_V1,
        selected_page,
        global_deskew_transform,
        globally_deskewed_page,
        crops: retained_crops,
    };
    Ok(StaffDetectionReport {
        global_deskew,
        retained_geometry,
        crops,
        crop_refinements,
        candidates,
        residual,
    })
}

/// Refine one band's residual skew (bd-av64.13 lever 1). The GLOBAL deskew
/// leaves per-staff residuals on real book pages (paper bow, per-plate
/// tilt), and recognition sits on a measured knife-edge: a -0.7 degree
/// rotation flipped a key signature read. Fine grid: +-1.5 degrees, step
/// 0.1, maximizing row-profile variance over THIS band only. Applied only
/// when the winner is >= 0.2 degrees away from flat — straight bands stay
/// BIT-IDENTICAL (the fit-first lesson: geometry changes only where they
/// pay). Lines are re-derived from the sheared profile; if the 5-line
/// group cannot be re-found the refinement is abandoned.
fn refine_band_skew(crop: &mut StaffCrop) -> FocrResult<TromrRowRefinementEvidenceV1> {
    let source_crop_before_refinement_gray8 =
        tromr_gray8_artifact_identity_v1(&crop.gray, crop.w, crop.h)?;
    let unchanged = || TromrRowRefinementEvidenceV1 {
        transform_contract: TROMR_ROW_REFINEMENT_TRANSFORM_CONTRACT,
        angle_millidegrees: 0,
        source_crop_before_refinement_gray8,
        refined_unpadded_crop_gray8: source_crop_before_refinement_gray8,
    };
    let thr = otsu_threshold(&crop.gray);
    let ink: Vec<bool> = crop.gray.iter().map(|&v| v <= thr).collect();
    let score = |millidegrees: i32| -> f64 {
        let degrees = f64::from(millidegrees) / 1_000.0;
        profile_variance(&sheared_row_profile(
            &ink,
            crop.w,
            crop.h,
            degrees.to_radians().tan(),
        ))
    };
    let flat = score(0);
    let mut best = (0i32, flat);
    for i in -15..=15i32 {
        let millidegrees = i * 100;
        if millidegrees == 0 {
            continue;
        }
        let sc = score(millidegrees);
        if sc > best.1 {
            best = (millidegrees, sc);
        }
    }
    if best.0.unsigned_abs() < 200 {
        return Ok(unchanged());
    }
    let sheared = shear_gray_millidegrees(&crop.gray, crop.w, crop.h, best.0);
    let sheared_ink: Vec<bool> = sheared.iter().map(|&v| v <= thr).collect();
    let profile = sheared_row_profile(&sheared_ink, crop.w, crop.h, 0.0);
    let peak = profile.iter().copied().max().unwrap_or(0);
    if peak == 0 {
        return Ok(unchanged());
    }
    let centers = line_band_centers(&profile, peak / 2);
    let staves = group_staves(&centers);
    let Some(five) = staves.first() else {
        return Ok(unchanged());
    };
    let refined_unpadded_crop_gray8 = tromr_gray8_artifact_identity_v1(&sheared, crop.w, crop.h)?;
    crop.gray = sheared;
    crop.lines = *five;
    let evidence = TromrRowRefinementEvidenceV1 {
        transform_contract: TROMR_ROW_REFINEMENT_TRANSFORM_CONTRACT,
        angle_millidegrees: best.0,
        source_crop_before_refinement_gray8,
        refined_unpadded_crop_gray8,
    };
    evidence.validate()?;
    Ok(evidence)
}

/// Candidate barline columns within a staff band (bd-av64.4): the centers
/// of thin vertical ink runs spanning the full five-line staff. A column
/// qualifies when >= 95% of the rows between the outer staff lines are ink
/// AND both outer lines are inked within one row; note stems rarely bridge
/// both outer lines, and beams/noteheads fail the thin-run filter (a
/// qualifying run wider than ~half a line-spacing is engraving, not a
/// barline). Isolation is enforced by requiring the columns flanking a run
/// to fall below half coverage. Classical CV only — no ML.
#[must_use]
pub fn barline_columns(crop: &StaffCrop) -> Vec<usize> {
    let thr = otsu_threshold(&crop.gray);
    let (l0, l4) = (crop.lines[0], crop.lines[4]);
    if l4 <= l0 || l4 >= crop.h {
        return Vec::new();
    }
    // Line CENTERS carry +-2 rows of detection error on real scans, so the
    // coverage window is the INTERIOR span [l0+1, l4-1] at a 92% floor, and
    // the outer-line presence checks tolerate +-2 rows (measured on the
    // 1843 Spohr fixture: true barlines are 100% covered, the strict
    // full-span/95%/+-1 form missed every one of them).
    let (a, b) = (l0 + 1, l4 - 1);
    let span = b - a + 1;
    let need = span * 92 / 100;
    let spacing = (l4 - l0) / 4;
    let max_run = (spacing / 2).max(2);
    let ink_at = |x: usize, y: usize| crop.gray[y * crop.w + x] <= thr;
    let coverage = |x: usize| (a..=b).filter(|&y| ink_at(x, y)).count();
    let near =
        |x: usize, l: usize| (l.saturating_sub(2)..=(l + 2).min(crop.h - 1)).any(|y| ink_at(x, y));
    // The stem/clef discriminator: a BARLINE's ink is confined to the staff
    // (nothing significant beyond either outer line), while stems run past
    // one outer line toward their beam and clef glyphs overshoot both. A
    // column with >30% ink in the spacing-tall zone outside either outer
    // line is engraving, not a barline.
    let outside_clear = |x: usize| {
        let zone_above = l0.saturating_sub(spacing)..l0.saturating_sub(2);
        let zone_below = (l4 + 3).min(crop.h)..(l4 + 1 + spacing).min(crop.h);
        let frac = |zone: std::ops::Range<usize>| {
            let len = zone.len();
            if len == 0 {
                return false;
            }
            zone.filter(|&y| ink_at(x, y)).count() * 10 > len * 3
        };
        !frac(zone_above) && !frac(zone_below)
    };
    let qualifies =
        |x: usize| coverage(x) >= need && near(x, l0) && near(x, l4) && outside_clear(x);
    let mut out = Vec::new();
    let mut run_start: Option<usize> = None;
    for x in 0..=crop.w {
        let q = x < crop.w && qualifies(x);
        match (q, run_start) {
            (true, None) => run_start = Some(x),
            (false, Some(s)) => {
                run_start = None;
                let e = x;
                if e - s <= max_run {
                    // GAP QUALITY: a true barline sits in an inter-measure
                    // gap — a spacing/2-wide zone on EACH side (skipping 2
                    // anti-aliased edge columns, which measured 80-90%
                    // coverage on the 1843 fixture) where every column is
                    // near-empty apart from the staff lines themselves
                    // (5 lines x 2px ~= 12%; floor at 30%). Stems inside
                    // beamed runs always have neighbor ink and fail this.
                    let gap = (spacing / 2).max(3);
                    let all_clear = |range: std::ops::Range<usize>| {
                        range
                            .filter(|&x| x < crop.w)
                            .all(|x| coverage(x) * 10 < span * 3)
                    };
                    let left_clear =
                        s < 3 || all_clear(s.saturating_sub(2 + gap)..s.saturating_sub(2));
                    let right_clear = e + 3 >= crop.w
                        || all_clear((e + 2).min(crop.w)..(e + 2 + gap).min(crop.w));
                    if left_clear && right_clear {
                        out.push((s + e - 1) / 2);
                    }
                }
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tromr_gray8_crop_rejects_invalid_shapes() {
        assert!(TromrGray8CropV1::from_tightly_packed(Vec::new(), 0, 1).is_err());
        assert!(TromrGray8CropV1::from_tightly_packed(Vec::new(), 1, 0).is_err());
        assert!(TromrGray8CropV1::from_tightly_packed(vec![0; 3], 2, 2).is_err());
        assert!(TromrGray8CropV1::from_tightly_packed(Vec::new(), usize::MAX, 2).is_err());
    }

    #[test]
    fn tromr_gray8_crop_identity_is_stable_and_pixel_sensitive() {
        let first =
            TromrGray8CropV1::from_tightly_packed(vec![0, 1, 2, 3], 2, 2).expect("valid crop");
        let again =
            TromrGray8CropV1::from_tightly_packed(vec![0, 1, 2, 3], 2, 2).expect("same crop");
        let changed =
            TromrGray8CropV1::from_tightly_packed(vec![0, 1, 2, 4], 2, 2).expect("mutated crop");
        assert_eq!(first, again);
        assert_ne!(first.pixels_sha256(), changed.pixels_sha256());
        assert_ne!(first.identity_sha256(), changed.identity_sha256());

        let transport_identity =
            tromr_gray8_artifact_identity_v1(first.pixels(), first.width(), first.height())
                .expect("calculate public transport identity");
        assert_eq!(transport_identity, first.artifact_identity());
        verify_tromr_gray8_artifact_identity_v1(
            first.pixels(),
            first.width(),
            first.height(),
            transport_identity,
        )
        .expect("verify exact transport bytes");
        assert!(
            verify_tromr_gray8_artifact_identity_v1(
                changed.pixels(),
                first.width(),
                first.height(),
                transport_identity,
            )
            .is_err()
        );
        let mut changed_domain = transport_identity;
        changed_domain.identity_sha256[0] ^= 1;
        assert!(
            verify_tromr_gray8_artifact_identity_v1(
                first.pixels(),
                first.width(),
                first.height(),
                changed_domain,
            )
            .is_err()
        );

        let mut internally_tampered = first;
        internally_tampered.pixels[0] ^= 1;
        assert!(internally_tampered.validate().is_err());
    }

    #[test]
    fn typed_geometry_transforms_replay_exact_owned_pixels() {
        let selected = TromrGray8StageV1::new(
            TromrGeometryCoordinateSpaceV1::SelectedPage,
            TromrGray8CropV1::from_tightly_packed((0u8..48).collect(), 6, 8)
                .expect("selected page"),
        )
        .expect("selected stage");
        let persisted_png = selected
            .gray8()
            .to_lossless_png()
            .expect("persist selected page");
        let reconstructed = TromrGray8StageV1::from_gray8(
            TromrGeometryCoordinateSpaceV1::SelectedPage,
            TromrGray8CropV1::from_lossless_png(&persisted_png).expect("reopen selected page"),
        )
        .expect("reconstruct typed stage");
        assert_eq!(reconstructed, selected);
        let global_transform = TromrVerticalShearTransformV1::global_deskew(0);
        let global = global_transform.apply(&selected).expect("global deskew");
        global_transform
            .validate_replay(&selected, &global)
            .expect("global replay");
        assert_eq!(
            global.coordinate_space(),
            TromrGeometryCoordinateSpaceV1::GloballyDeskewedPage
        );
        assert_eq!(global.gray8().pixels(), selected.gray8().pixels());

        let crop_transform =
            TromrCropTransformV1::staff_from_globally_deskewed_page(TromrPixelRectV1 {
                x: 1,
                y: 1,
                width: 3,
                height: 6,
            });
        let pre_refinement = crop_transform.apply(&global).expect("exact crop");
        assert_eq!(
            pre_refinement.gray8().pixels(),
            &[
                7, 8, 9, 13, 14, 15, 19, 20, 21, 25, 26, 27, 31, 32, 33, 37, 38, 39
            ]
        );
        crop_transform
            .validate_replay(&global, &pre_refinement)
            .expect("crop replay");

        let row_transform = TromrVerticalShearTransformV1::row_refinement(0);
        let refined = row_transform
            .apply(&pre_refinement)
            .expect("row refinement");
        row_transform
            .validate_replay(&pre_refinement, &refined)
            .expect("row replay");
        let padding_transform = TromrPaddingTransformV1::review_canvas(StaffPadding {
            top: 1,
            right: 2,
            bottom: 2,
            left: 1,
        });
        let review = padding_transform.apply(&refined).expect("review canvas");
        padding_transform
            .validate_replay(&refined, &review)
            .expect("padding replay");
        assert_eq!((review.gray8().width(), review.gray8().height()), (6, 9));
        assert!(
            review.gray8().pixels()[..6]
                .iter()
                .all(|pixel| *pixel == 255)
        );
        assert_eq!(&review.gray8().pixels()[7..10], &[7, 8, 9]);

        let mut tampered = review.clone();
        tampered.gray8.pixels[0] ^= 1;
        assert!(
            padding_transform
                .validate_replay(&refined, &tampered)
                .is_err()
        );
        assert!(crop_transform.apply(&selected).is_err());
        assert!(
            TromrVerticalShearTransformV1::global_deskew(125)
                .validate()
                .is_err()
        );
        assert!(
            TromrCropTransformV1::staff_from_globally_deskewed_page(TromrPixelRectV1 {
                x: 5,
                y: 0,
                width: 2,
                height: 1,
            })
            .apply(&global)
            .is_err()
        );
    }

    fn synthetic_detector_evidence() -> StaffDetectionEvidenceV1 {
        synthetic_complete_evidence_for_test(
            20,
            24,
            &[(
                StaffCropGeometry::unpadded((0, 5, 20, 12)),
                [6, 8, 10, 12, 14],
                [1, 3, 5, 7, 9],
            )],
        )
    }

    #[test]
    fn transform_evidence_rejects_off_grid_angles_and_zero_angle_identity_drift() {
        let valid = synthetic_detector_evidence();
        valid.validate().expect("synthetic detector evidence");

        let mut global_angle = valid.clone();
        global_angle.global_deskew.angle_millidegrees = 125;
        assert!(global_angle.validate().is_err());

        let mut global_identity = valid.clone();
        global_identity
            .global_deskew
            .globally_deskewed_gray8
            .identity_sha256[0] ^= 1;
        assert!(global_identity.validate().is_err());

        let mut row_angle = valid.clone();
        row_angle.crops[0].row_refinement.angle_millidegrees = 100;
        assert!(row_angle.validate().is_err());

        let mut row_identity = valid;
        row_identity.crops[0]
            .row_refinement
            .refined_unpadded_crop_gray8
            .pixels_sha256[0] ^= 1;
        assert!(row_identity.validate().is_err());
    }

    #[test]
    fn detector_evidence_rejects_candidate_and_residual_accounting_drift() {
        let valid = synthetic_detector_evidence();

        let mut threshold = valid.clone();
        threshold.candidates[0].line_threshold += 1;
        assert!(threshold.validate().is_err());

        let mut derived_spacing = valid.clone();
        derived_spacing.candidates[0].mean_spacing_milli += 1;
        assert!(derived_spacing.validate().is_err());

        let mut omitted_candidate = valid.clone();
        omitted_candidate.candidates.clear();
        assert!(omitted_candidate.validate().is_err());

        let mut fabricated_residual = valid;
        fabricated_residual
            .residual
            .unresolved_candidates
            .push([6, 8, 10, 12, 14]);
        fabricated_residual.residual.unresolved = true;
        assert!(fabricated_residual.validate().is_err());

        let unresolved = synthetic_unresolved_evidence_for_test(
            20,
            32,
            &[(
                StaffCropGeometry::unpadded((0, 3, 20, 12)),
                [4, 6, 8, 10, 12],
                [1, 3, 5, 7, 9],
            )],
            [16, 18, 20, 22, 25],
        );
        unresolved
            .validate()
            .expect("unresolved evidence remains structurally valid");
        assert!(
            unresolved.require_complete().is_err(),
            "the same evidence must block a complete-page publication claim"
        );
    }

    #[test]
    fn tromr_gray8_crop_png_decodes_to_exact_l8_pixels() {
        let crop = TromrGray8CropV1::from_tightly_packed(vec![0, 64, 128, 255, 7, 9], 3, 2)
            .expect("valid crop");
        let hex = |bytes: &[u8]| {
            bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        assert_eq!(
            hex(&crop.pixels_sha256()),
            "9732adb159845237b568574bb0bcb4027c98614147a5026fd7d2f98e522abcd3"
        );
        assert_eq!(
            hex(&crop.pixels_blake3()),
            "501da236ff9776d448c2027265c96b2d18e42fc9d3af78d14c0310bab911b27c"
        );
        assert_eq!(
            hex(&crop.identity_sha256()),
            "00c39d1f298abc717a25662c3c9ad1a5f5839831bfc36d5f1bf66ca571be0302"
        );
        let first = crop.to_lossless_png().expect("provider PNG");
        let second = crop.to_lossless_png().expect("deterministic provider PNG");
        assert_eq!(first, second);
        let png_sha256: [u8; 32] = Sha256::digest(&first).into();
        assert_eq!(
            hex(&png_sha256),
            "8244171084b83f18bbae7ee8ab5dede082a980086357cb07f9884538613685c2"
        );
        assert_eq!(&first[..8], b"\x89PNG\r\n\x1a\n");
        let decoded = TromrGray8CropV1::from_lossless_png(&first).expect("strict provider decode");
        assert_eq!(decoded, crop);
        assert_eq!(crop.encoding(), "gray8");
        assert_eq!(crop.row_order(), "top_to_bottom");
        assert_eq!(crop.column_order(), "left_to_right");

        let mut rgb_png = Vec::new();
        image::codecs::png::PngEncoder::new(&mut rgb_png)
            .write_image(&[1, 2, 3], 1, 1, image::ExtendedColorType::Rgb8)
            .expect("RGB fixture PNG");
        assert!(TromrGray8CropV1::from_lossless_png(&rgb_png).is_err());
        let mut corrupt = first;
        corrupt.truncate(corrupt.len() / 2);
        assert!(TromrGray8CropV1::from_lossless_png(&corrupt).is_err());
    }

    /// Draw a synthetic page: `staves` five-line groups (line thickness 2,
    /// spacing 10) at the given top offsets, on a 255 page with some noise
    /// notes (short dark runs) between lines.
    fn synth_page(w: usize, h: usize, staff_tops: &[usize]) -> DynamicImage {
        let mut gray = vec![250u8; w * h];
        for &top in staff_tops {
            for line in 0..5 {
                let y = top + line * 10;
                for dy in 0..2 {
                    for x in 20..w - 20 {
                        gray[(y + dy) * w + x] = 10;
                    }
                }
            }
            // a few "note heads" between the lines
            for k in 0..6 {
                let cx = 60 + k * 90;
                let cy = top + 14 + (k % 3) * 10;
                for dy in 0..5 {
                    for dx in 0..7 {
                        gray[(cy + dy) * w + cx + dx] = 30;
                    }
                }
            }
        }
        let img = image::GrayImage::from_raw(w as u32, h as u32, gray).unwrap();
        DynamicImage::ImageLuma8(img)
    }

    #[test]
    fn detector_retains_replayable_pixels_and_staff_rows_for_every_stage() {
        let report = detect_staves_with_evidence(&synth_page(800, 400, &[80, 250]))
            .expect("detector report");
        report
            .retained_geometry
            .validate()
            .expect("complete retained geometry replays");
        assert_eq!(report.crops.len(), 2);
        assert_eq!(report.retained_geometry.crops.len(), report.crops.len());
        assert_eq!(
            report.retained_geometry.selected_page.coordinate_space(),
            TromrGeometryCoordinateSpaceV1::SelectedPage
        );
        assert_eq!(
            report
                .retained_geometry
                .globally_deskewed_page
                .coordinate_space(),
            TromrGeometryCoordinateSpaceV1::GloballyDeskewedPage
        );
        assert_eq!(
            report.retained_geometry.selected_page.artifact_identity(),
            report.global_deskew.input_gray8
        );
        assert_eq!(
            report
                .retained_geometry
                .globally_deskewed_page
                .artifact_identity(),
            report.global_deskew.globally_deskewed_gray8
        );

        let accepted_candidates = report
            .candidates
            .iter()
            .filter(|candidate| candidate.disposition == StaffCandidateDisposition::Accepted)
            .count();
        assert_eq!(accepted_candidates, report.retained_geometry.crops.len());
        for ((crop, refinement), retained) in report
            .crops
            .iter()
            .zip(&report.crop_refinements)
            .zip(&report.retained_geometry.crops)
        {
            assert_eq!(retained.crop_transform.source_rect.as_bbox(), crop.bbox);
            assert_eq!(
                retained.globally_deskewed_staff_lines.y_rows,
                crop.globally_deskewed_raster_lines
            );
            assert_eq!(retained.review_canvas_staff_lines.y_rows, crop.lines);
            assert_eq!(
                retained.review_canvas.gray8().pixels(),
                crop.gray.as_slice()
            );
            assert_eq!(
                retained.pre_refinement_crop.artifact_identity(),
                refinement.source_crop_before_refinement_gray8
            );
            assert_eq!(
                retained.refined_unpadded_crop.artifact_identity(),
                refinement.refined_unpadded_crop_gray8
            );
            assert_eq!(
                retained.pre_refinement_staff_lines.y_rows,
                crop.globally_deskewed_raster_lines
                    .map(|row| row - crop.bbox.1)
            );
            assert_eq!(
                retained.review_canvas_staff_lines.y_rows,
                retained
                    .refined_unpadded_staff_lines
                    .y_rows
                    .map(|row| row + crop.padding.top)
            );
        }
        report.evidence().expect("pixel-free evidence cross-checks");
    }

    #[test]
    fn retained_geometry_rejects_pixel_coordinate_and_transform_mutations() {
        let report =
            detect_staves_with_evidence(&synth_page(800, 260, &[80])).expect("detector report");
        let valid = report.retained_geometry;

        let mut pixel = valid.clone();
        pixel.crops[0].pre_refinement_crop.gray8.pixels[0] ^= 1;
        assert!(pixel.validate().is_err());

        let mut coordinate_space = valid.clone();
        coordinate_space.crops[0]
            .pre_refinement_crop
            .coordinate_space = TromrGeometryCoordinateSpaceV1::ReviewCanvas;
        assert!(coordinate_space.validate().is_err());

        let mut crop_rect = valid.clone();
        crop_rect.crops[0].crop_transform.source_rect.x += 1;
        assert!(crop_rect.validate().is_err());

        let mut padding = valid.clone();
        padding.crops[0].padding_transform.padding.top += 1;
        assert!(padding.validate().is_err());

        let mut row_space = valid;
        row_space.crops[0]
            .review_canvas_staff_lines
            .coordinate_space = TromrGeometryCoordinateSpaceV1::RefinedUnpaddedCrop;
        assert!(row_space.validate().is_err());
    }

    fn draw_staff(
        gray: &mut [u8],
        width: usize,
        height: usize,
        top: usize,
        x_start: usize,
        x_end: usize,
        value: u8,
    ) {
        for line in 0..5 {
            let y = top + line * 10;
            for dy in 0..2 {
                if y + dy >= height {
                    continue;
                }
                for x in x_start..x_end.min(width) {
                    gray[(y + dy) * width + x] = value;
                }
            }
        }
    }

    fn accepted_page_lines(report: &StaffDetectionReport) -> Vec<[usize; 5]> {
        report
            .candidates
            .iter()
            .filter(|candidate| candidate.disposition == StaffCandidateDisposition::Accepted)
            .map(|candidate| candidate.lines)
            .collect()
    }

    #[test]
    fn horizontal_run_profile_ignores_fragmented_text_ink() {
        let (w, h) = (100usize, 2usize);
        let mut ink = vec![false; w * h];
        for start in [0usize, 20, 40, 60, 80] {
            ink[start..start + 8].fill(true);
        }
        ink[w + 20..w + 35].fill(true);
        let profile = horizontal_run_profile(&ink, w, h);
        assert_eq!(profile, vec![0, 15]);
    }

    #[test]
    fn interleaved_nuisance_peak_does_not_displace_a_staff_line() {
        let centers = [80usize, 90, 100, 105, 110, 120];
        let mut profile = vec![0u32; 160];
        for row in [80usize, 90, 100, 110, 120] {
            profile[row] = 100;
        }
        profile[105] = 80;

        assert!(
            !geometry_candidates(&profile, &centers, false)
                .iter()
                .any(|candidate| candidate.accepted),
            "consecutive-only geometry is the measured failure mode"
        );
        assert!(
            geometry_candidates(&profile, &centers, true)
                .iter()
                .any(|candidate| candidate.accepted && candidate.lines == [80, 90, 100, 110, 120]),
            "bounded subsequence recovery must skip the nuisance peak"
        );
    }

    #[test]
    fn noninterlacing_selection_prefers_two_rows_over_one_bridge() {
        let first = [20usize, 30, 40, 50, 60];
        let second = [100usize, 110, 120, 130, 140];
        let bridge = [50usize, 65, 80, 95, 110];
        let mut profile = vec![0u32; 200];
        for row in first.into_iter().chain(second).chain([65, 80, 95]) {
            profile[row] = 100;
        }

        let (accepted, evidence) = merge_staff_candidates(&profile);
        assert_eq!(accepted, vec![first, second]);
        assert!(accepted.windows(2).all(|pair| pair[0][4] < pair[1][0]));
        assert!(evidence.iter().any(|candidate| {
            candidate.lines == bridge
                && candidate.disposition == StaffCandidateDisposition::Rejected
                && candidate.reason == Some("overlapping_five_line_page_geometry")
        }));
    }

    #[test]
    fn local_projection_recovers_staff_suppressed_by_global_peak() {
        let (w, h) = (1_000usize, 800usize);
        let mut gray = vec![250u8; w * h];
        draw_staff(&mut gray, w, h, 100, 20, 980, 10);
        draw_staff(&mut gray, w, h, 600, 300, 700, 10);
        let image = DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(w as u32, h as u32, gray).expect("synthetic page"),
        );

        let report = detect_staves_with_evidence(&image).expect("detection report");
        assert_eq!(report.crops.len(), 2, "local pass restores the short staff");
        assert!(!report.residual.unresolved);
        let global = report
            .candidates
            .iter()
            .filter(|candidate| {
                candidate.origin == StaffCandidateOrigin::Global
                    && candidate.disposition == StaffCandidateDisposition::Accepted
            })
            .count();
        assert_eq!(global, 1, "global peak sees only the long staff");
        assert!(report.candidates.iter().any(|candidate| {
            matches!(candidate.origin, StaffCandidateOrigin::Local { .. })
                && candidate.disposition == StaffCandidateDisposition::Accepted
                && candidate.lines[0].abs_diff(600) <= 2
        }));
    }

    #[test]
    fn short_staff_below_fifteen_percent_is_explicit_rejected_evidence() {
        let (w, h) = (1_000usize, 500usize);
        let mut gray = vec![250u8; w * h];
        draw_staff(&mut gray, w, h, 100, 100, 300, 10);
        draw_staff(&mut gray, w, h, 300, 100, 200, 10);
        let image = DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(w as u32, h as u32, gray).expect("mixed extent score"),
        );

        let report = detect_staves_with_evidence(&image).expect("detection report");
        assert_eq!(report.crops.len(), 1, "only the >=15% row is crop-eligible");
        assert!(!report.residual.unresolved);
        assert!(report.candidates.iter().any(|candidate| {
            candidate.lines[0].abs_diff(300) <= 2
                && candidate.disposition == StaffCandidateDisposition::Rejected
                && candidate.reason == Some("horizontal_extent_below_15_percent")
                && candidate.minimum_horizontal_span_basis_points == AUDIT_MIN_HORIZONTAL_SPAN_BPS
        }));
    }

    #[test]
    fn overlapping_windows_deduplicate_in_stable_page_order() {
        let image = synth_page(800, 500, &[100, 300]);
        let first = detect_staves_with_evidence(&image).expect("first report");
        let second = detect_staves_with_evidence(&image).expect("second report");
        assert_eq!(first.crops.len(), 2);
        assert_eq!(accepted_page_lines(&first), accepted_page_lines(&second));
        assert!(first.candidates.iter().any(|candidate| {
            candidate.disposition == StaffCandidateDisposition::Duplicate
                && candidate.reason == Some("same_five_line_page_geometry")
        }));
        assert!(
            first
                .candidates
                .windows(2)
                .all(|pair| pair[0].lines[0] <= pair[1].lines[0]),
            "the public evidence ledger is page ordered"
        );
        assert!(first.crops[0].bbox.1 < first.crops[1].bbox.1);
    }

    #[test]
    fn broken_and_gently_curved_lines_remain_detectable() {
        let (w, h) = (800usize, 500usize);
        let mut gray = vec![250u8; w * h];

        // A central paper tear interrupts every line, but most of each staff
        // row remains measurable by the projection.
        for line in 0..5 {
            let y = 100 + line * 10;
            for dy in 0..2 {
                for x in (20..350).chain(450..780) {
                    gray[(y + dy) * w + x] = 10;
                }
            }
        }

        // One-row bow across three horizontal regions. This is deliberately
        // not independently deskewed; adjacent projection rows merge into a
        // single line band on the globally deskewed page.
        for line in 0..5 {
            let base = 300 + line * 10;
            for x in 20..780 {
                let offset = if x < 280 {
                    -1isize
                } else if x < 520 {
                    0
                } else {
                    1
                };
                let y = usize::try_from(base as isize + offset).expect("positive row");
                for dy in 0..2 {
                    gray[(y + dy) * w + x] = 10;
                }
            }
        }

        let image = DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(w as u32, h as u32, gray).expect("damaged score"),
        );
        let report = detect_staves_with_evidence(&image).expect("detection report");
        assert_eq!(report.crops.len(), 2);
        assert!(!report.residual.unresolved);
    }

    #[test]
    fn six_uniform_lines_are_rejected_as_non_score_ruling() {
        let (w, h) = (600usize, 240usize);
        let mut gray = vec![250u8; w * h];
        for line in 0..6 {
            let y = 80 + line * 10;
            for x in 20..580 {
                gray[y * w + x] = 10;
            }
        }
        let image = DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(w as u32, h as u32, gray).expect("six lines"),
        );
        let report = detect_staves_with_evidence(&image).expect("detection report");
        assert!(report.crops.is_empty());
        assert!(
            !report.residual.unresolved,
            "known non-score ruling is resolved"
        );
        assert!(
            report.candidates.iter().any(|candidate| {
                candidate.reason == Some("six_or_more_comparable_uniform_lines")
            })
        );
    }

    #[test]
    fn adjacent_short_rule_does_not_reject_a_five_line_staff() {
        let (w, h) = (600usize, 240usize);
        let mut gray = vec![250u8; w * h];
        draw_staff(&mut gray, w, h, 80, 20, 580, 10);
        // This sixth peak clears the projection threshold and is exactly one
        // staff spacing away, but its much shorter extent makes it a ledger or
        // prose rule rather than a coherent six-line ruling.
        for x in 150..450 {
            gray[130 * w + x] = 10;
        }
        let image = DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(w as u32, h as u32, gray).expect("staff plus rule"),
        );
        let report = detect_staves_with_evidence(&image).expect("detection report");
        assert_eq!(report.crops.len(), 1);
        assert!(!report.residual.unresolved);
        assert!(report.candidates.iter().any(|candidate| {
            candidate.disposition == StaffCandidateDisposition::Accepted
                && candidate.lines[0].abs_diff(80) <= 2
        }));
    }

    #[test]
    fn near_staff_residual_refuses_instead_of_silently_disappearing() {
        let (w, h) = (600usize, 260usize);
        let mut gray = vec![250u8; w * h];
        for y in [80usize, 90, 100, 114, 124] {
            for x in 20..580 {
                gray[y * w + x] = 10;
            }
        }
        let image = DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(w as u32, h as u32, gray).expect("near staff"),
        );
        let report = detect_staves_with_evidence(&image).expect("detection report");
        assert!(report.crops.is_empty());
        assert!(report.residual.unresolved);
        assert_eq!(report.residual.unresolved_candidates.len(), 1);
        let error = detect_staves(&image).expect_err("strict path refuses");
        assert!(
            error
                .to_string()
                .contains("unresolved staff-like residual ink")
        );
    }

    #[test]
    fn nonuniform_text_rules_do_not_become_a_staff_or_residual() {
        let (w, h) = (600usize, 260usize);
        let mut gray = vec![250u8; w * h];
        for y in [60usize, 70, 100, 110, 140] {
            for x in 40..560 {
                gray[y * w + x] = 10;
            }
        }
        let image = DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(w as u32, h as u32, gray).expect("text rules"),
        );
        let report = detect_staves_with_evidence(&image).expect("detection report");
        assert!(report.crops.is_empty());
        assert!(!report.residual.unresolved);
    }

    #[test]
    fn detection_is_stable_under_translation_and_uniform_contrast_change() {
        let original = synth_page(800, 700, &[100, 500]);
        let shifted = synth_page(800, 700, &[117, 517]);
        let original_report = detect_staves_with_evidence(&original).expect("original");
        let shifted_report = detect_staves_with_evidence(&shifted).expect("shifted");
        let original_lines = accepted_page_lines(&original_report);
        let shifted_lines = accepted_page_lines(&shifted_report);
        assert_eq!(original_lines.len(), shifted_lines.len());
        for (before, after) in original_lines.iter().zip(&shifted_lines) {
            assert!(before.iter().zip(after).all(|(a, b)| *b == *a + 17));
        }

        let original_gray = original.to_luma8();
        let contrasted = DynamicImage::ImageLuma8(image::GrayImage::from_fn(800, 700, |x, y| {
            let value = original_gray.get_pixel(x, y).0[0];
            image::Luma([100u8.saturating_add(value / 2)])
        }));
        let contrasted_report = detect_staves_with_evidence(&contrasted).expect("contrast");
        assert_eq!(
            accepted_page_lines(&original_report),
            accepted_page_lines(&contrasted_report)
        );
    }

    #[test]
    fn detection_is_stable_under_two_x_geometric_scale() {
        let scaled_page = |scale: usize| {
            let (w, h) = (800usize * scale, 500usize * scale);
            let mut gray = vec![250u8; w * h];
            for top in [100usize * scale, 300usize * scale] {
                for line in 0..5 {
                    let y = top + line * 10 * scale;
                    for dy in 0..scale {
                        for x in 20 * scale..w - 20 * scale {
                            gray[(y + dy) * w + x] = 10;
                        }
                    }
                }
            }
            DynamicImage::ImageLuma8(
                image::GrayImage::from_raw(w as u32, h as u32, gray).expect("scaled score"),
            )
        };

        let one_x = detect_staves_with_evidence(&scaled_page(1)).expect("1x report");
        let two_x = detect_staves_with_evidence(&scaled_page(2)).expect("2x report");
        assert!(!one_x.residual.unresolved && !two_x.residual.unresolved);
        let one_x_lines = accepted_page_lines(&one_x);
        let two_x_lines = accepted_page_lines(&two_x);
        assert_eq!(one_x_lines.len(), 2);
        assert_eq!(two_x_lines.len(), one_x_lines.len());
        for (one, two) in one_x_lines.iter().zip(two_x_lines) {
            assert_eq!(two, one.map(|row| row * 2));
        }
    }

    #[test]
    fn page_boundary_staves_and_window_arithmetic_are_bounded() {
        let (w, h) = (800usize, 400usize);
        let mut gray = vec![250u8; w * h];
        draw_staff(&mut gray, w, h, 0, 20, 780, 10);
        draw_staff(&mut gray, w, h, 358, 20, 780, 10);
        let image = DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(w as u32, h as u32, gray).expect("boundary page"),
        );
        let report = detect_staves_with_evidence(&image).expect("boundary rows");
        assert_eq!(report.crops.len(), 2);
        assert_eq!(report.crops[0].bbox.1, 0);
        assert_eq!(report.crops[1].bbox.1 + report.crops[1].bbox.3, 400);
        assert!(local_detection_windows(usize::MAX).len() <= 31);
    }

    #[test]
    fn real_spohr_page_55_recovers_all_twelve_rows() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/realscan_music/pages/spohr_p055.png");
        let image = image::open(path).expect("Spohr page 55 fixture");
        let report = detect_staves_with_evidence(&image).expect("detection report");
        assert_eq!(
            report.crops.len(),
            12,
            "verified visual truth is 12 rows; candidates={:#?}",
            report.candidates
        );
        assert!(
            !report.residual.unresolved,
            "residual={:#?}; candidates={:#?}",
            report.residual, report.candidates
        );
        assert!(report.candidates.iter().any(|candidate| {
            matches!(candidate.origin, StaffCandidateOrigin::Local { .. })
                && candidate.disposition == StaffCandidateDisposition::Accepted
        }));
        let expected_outer_lines = [
            (258usize, 292usize),
            (342, 376),
            (449, 483),
            (526, 560),
            (636, 670),
            (714, 747),
            (825, 859),
            (900, 934),
            (1014, 1050),
            (1091, 1124),
            (1205, 1239),
            (1287, 1322),
        ];
        for (actual, expected) in accepted_page_lines(&report)
            .iter()
            .zip(expected_outer_lines)
        {
            assert!(
                actual[0].abs_diff(expected.0) <= 2 && actual[4].abs_diff(expected.1) <= 2,
                "candidate {actual:?} does not match verified row span {expected:?}"
            );
        }
    }

    #[test]
    fn real_spohr_mixed_prose_page_keeps_three_full_staves() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/realscan_music/pages/spohr_p100.png");
        let image = image::open(path).expect("Spohr page 100 fixture");
        let report = detect_staves_with_evidence(&image).expect("detection report");
        assert_eq!(
            report.crops.len(),
            3,
            "three full-width embedded staves; candidates={:#?}",
            report.candidates
        );
        assert!(
            !report.residual.unresolved,
            "residual={:#?}; candidates={:#?}",
            report.residual, report.candidates
        );
    }

    /// Live public-corpus gate for the exact Breitkopf 1882 K.387 artifact.
    /// Page 1 visibly contains four quartet systems of four rows, not the
    /// stale twelve-row oracle formerly documented by the TrOMR inference
    /// test. PDF decode remains embedded through [`crate::pdf::PdfPages`].
    #[test]
    #[ignore = "requires pinned Mozart K.387 public PDF"]
    fn mozart_k387_page_one_detects_all_sixteen_visible_rows() {
        use sha2::{Digest, Sha256};

        const PDF_SHA256: &str = "64406ae67f690b32f689bb60169287d0a6d514d13437b6027ee999381a43cb01";
        let pdf_path = std::path::PathBuf::from(
            std::env::var_os("FOCR_TEST_MOZART_K387_PDF")
                .expect("FOCR_TEST_MOZART_K387_PDF must name the pinned public PDF"),
        );
        let bytes = std::fs::read(&pdf_path).expect("read pinned K.387 PDF");
        assert_eq!(format!("{:x}", Sha256::digest(bytes)), PDF_SHA256);

        let pages = crate::pdf::PdfPages::open(&pdf_path).expect("embedded PDF open");
        let page = pages.render(0).expect("embedded page-1 render");
        assert_eq!((page.width(), page.height()), (5_904, 7_558));
        let report = detect_staves_with_evidence(&page).expect("staff detection report");
        let accepted_lines = accepted_page_lines(&report);
        let accepted_candidates = report
            .candidates
            .iter()
            .filter(|candidate| candidate.disposition == StaffCandidateDisposition::Accepted)
            .count();
        let rejected_candidates = report
            .candidates
            .iter()
            .filter(|candidate| candidate.disposition == StaffCandidateDisposition::Rejected)
            .count();
        let duplicate_candidates = report
            .candidates
            .iter()
            .filter(|candidate| candidate.disposition == StaffCandidateDisposition::Duplicate)
            .count();
        let (global_candidates, local_candidates) =
            report
                .candidates
                .iter()
                .fold(
                    (0usize, 0usize),
                    |(global, local), candidate| match candidate.origin {
                        StaffCandidateOrigin::Global => (global + 1, local),
                        StaffCandidateOrigin::Local { .. } => (global, local + 1),
                    },
                );
        eprintln!(
            "[staff-detect-cert] mozart_k387 pdf_sha256={PDF_SHA256} raster={}x{} \
             candidates_total={} accepted={} rejected={} duplicate={} global={} local={} \
             accepted_lines={accepted_lines:?} coverage_basis_points={} \
             unresolved_candidates={} unresolved={}",
            page.width(),
            page.height(),
            report.candidates.len(),
            accepted_candidates,
            rejected_candidates,
            duplicate_candidates,
            global_candidates,
            local_candidates,
            report.residual.coverage_basis_points,
            report.residual.unresolved_candidates.len(),
            report.residual.unresolved,
        );
        assert_eq!(
            report.crops.len(),
            16,
            "four visible quartet systems require sixteen rows; accepted={accepted_lines:?}; residual={:#?}; candidates={:#?}",
            report.residual,
            report.candidates,
        );
        assert!(
            !report.residual.unresolved,
            "residual={:#?}; candidates={:#?}",
            report.residual, report.candidates
        );
    }

    #[test]
    fn detects_two_staves_in_order_with_sane_crops() {
        let img = synth_page(800, 400, &[80, 250]);
        let crops = detect_staves(&img).expect("detects");
        assert_eq!(crops.len(), 2, "two 5-line groups");
        // Top-to-bottom order + bboxes cover each staff with margin.
        assert!(crops[0].bbox.1 < crops[1].bbox.1);
        let (_, y0, _, ch0) = crops[0].bbox;
        assert!(y0 < 80 && y0 + ch0 > 80 + 40, "margin spans the staff");
        // Every crop is full-width and non-degenerate.
        for c in &crops {
            assert_eq!(c.w, 800);
            assert!(c.h >= 40 && c.h <= 200, "crop height {}", c.h);
            assert_eq!(c.gray.len(), c.w * c.h);
        }
    }

    #[test]
    fn deskew_recovers_a_sheared_page() {
        // Shear the synthetic page by ~2° and confirm detection still finds
        // both staves (the deskew must undo the tilt).
        let img = synth_page(800, 400, &[80, 250]);
        let gray = img.to_luma8();
        let sheared = shear_gray(gray.as_raw(), 800, 400, -2.0);
        let tilted =
            DynamicImage::ImageLuma8(image::GrayImage::from_raw(800, 400, sheared).unwrap());
        let report = detect_staves_with_evidence(&tilted).expect("detects");
        assert_eq!(report.crops.len(), 2, "deskew recovers both staves");
        assert_ne!(report.global_deskew.angle_millidegrees, 0);
        assert_ne!(
            report.global_deskew.input_gray8,
            report.global_deskew.globally_deskewed_gray8
        );
        report
            .evidence()
            .expect("pixel-free detector evidence")
            .validate()
            .expect("deskew evidence validates");
    }

    #[test]
    fn blank_and_noise_pages_yield_no_staves() {
        let blank =
            DynamicImage::ImageLuma8(image::GrayImage::from_pixel(400, 300, image::Luma([255u8])));
        assert!(detect_staves(&blank).expect("runs").is_empty());
        // 4 lines (not 5) must NOT group into a staff.
        let mut gray = vec![250u8; 400 * 300];
        for line in 0..4 {
            let y = 100 + line * 10;
            for x in 20..380 {
                gray[y * 400 + x] = 10;
            }
        }
        let four = DynamicImage::ImageLuma8(image::GrayImage::from_raw(400, 300, gray).unwrap());
        assert!(
            detect_staves(&four).expect("runs").is_empty(),
            "4 lines != a staff"
        );
    }

    /// bd-av64.4: barline detection — thin full-span verticals found at
    /// their drawn positions; stems (partial span) and beams (wide) do not
    /// qualify; speckle noise does not create false bars.
    #[test]
    fn barline_columns_finds_thin_full_span_verticals() {
        // 800x160 band: 5 lines at rows 40..80 (spacing 10), thickness 2.
        let (w, h) = (800usize, 160usize);
        let mut gray = vec![250u8; w * h];
        for line in 0..5 {
            let y = 40 + line * 10;
            for dy in 0..2 {
                for x in 20..780 {
                    gray[(y + dy) * w + x] = 10;
                }
            }
        }
        // Three true barlines (2px wide, spanning rows 40..82).
        for &bx in &[200usize, 450, 700] {
            for x in bx..bx + 2 {
                for y in 40..82 {
                    gray[y * w + x] = 10;
                }
            }
        }
        // A stem-like partial vertical (rows 52..82 only) must NOT qualify.
        for y in 52..82 {
            gray[y * w + 300] = 10;
        }
        // A beam-like WIDE dark block spanning the staff must NOT qualify.
        for x in 550..580 {
            for y in 40..82 {
                gray[y * w + x] = 10;
            }
        }
        // Speckle noise.
        for k in 0..50 {
            let (x, y) = ((k * 37) % w, (k * 53) % h);
            gray[y * w + x] = 15;
        }
        let crop = StaffCrop {
            gray,
            w,
            h,
            bbox: (0, 0, w, h),
            lines: [40, 50, 60, 70, 80],
            globally_deskewed_raster_lines: [40, 50, 60, 70, 80],
            padding: StaffPadding::default(),
        };
        let bars = barline_columns(&crop);
        assert_eq!(bars.len(), 3, "exactly the drawn barlines: {bars:?}");
        for (got, want) in bars.iter().zip([200usize, 450, 700]) {
            assert!(
                got.abs_diff(want) <= 2,
                "barline at {got} expected near {want}"
            );
        }
    }

    /// bd-av64.14 FIT-FIRST: a band that already fits the positional budget
    /// keeps the historic full-width geometry EXACTLY (recognition is
    /// margin-sensitive; geometry changes only where the old form
    /// hard-failed on the clamp).
    #[test]
    fn fitting_bands_keep_the_classic_full_width_geometry() {
        let img = synth_page(800, 260, &[80]);
        let crops = detect_staves(&img).expect("detects");
        assert_eq!(crops.len(), 1);
        let (x0, y0, cw, ch) = crops[0].bbox;
        assert_eq!((x0, cw), (0, 800), "full width kept");
        // classic band: 2*(2*spacing) margins around the 5-line span.
        assert_eq!(y0, 40, "classic top margin");
        assert_eq!(ch, 121, "classic band height");
    }

    /// bd-av64.14: horizontal ink-extent trim — on an OVER-BUDGET band
    /// (classic full width 3000 x ~120 resizes to 3200 > 1280), page
    /// margins leave the band; the ink span plus ~2-spacing pads stays.
    #[test]
    fn trim_cuts_page_margins_but_keeps_ink() {
        // Ink spans x 200..2600 on a 3000-wide page (spacing 10 => pad 20).
        let mut gray = vec![250u8; 3000 * 260];
        for line in 0..5 {
            let y = 80 + line * 10;
            for dy in 0..2 {
                for x in 200..2600 {
                    gray[(y + dy) * 3000 + x] = 10;
                }
            }
        }
        let img =
            DynamicImage::ImageLuma8(image::GrayImage::from_raw(3000, 260, gray).expect("synth"));
        let crops = detect_staves(&img).expect("detects");
        assert_eq!(crops.len(), 1);
        let (x0, _y0, cw, _ch) = crops[0].bbox;
        assert!(
            (150..=200).contains(&x0),
            "left margin trimmed to pad (x0 {x0})"
        );
        assert!(
            x0 + cw <= 2650 && x0 + cw >= 2600,
            "right margin trimmed (x0+cw {})",
            x0 + cw
        );
    }

    /// bd-av64.14: extend-to-fit — a wide staff with vertical room grows its
    /// band until the resized width fits the 1280 positional budget.
    #[test]
    fn wide_staff_with_room_fits_the_positional_budget() {
        let img = synth_page(2000, 700, &[320]);
        let crops = detect_staves(&img).expect("detects");
        assert_eq!(crops.len(), 1);
        let (_x0, _y0, cw, ch) = crops[0].bbox;
        assert!(
            128 * cw <= 1280 * ch,
            "resized width {} exceeds 1280 (cw {cw}, ch {ch})",
            128 * cw / ch
        );
        assert_eq!(
            crops[0].padding,
            StaffPadding::default(),
            "real page space is preferred when available"
        );
    }

    /// bd-av64.14: neighbor bounds — two packed wide staves may extend only
    /// to their shared midline; bands never overlap even under budget
    /// pressure, and every band keeps the whole 5-line span.
    #[test]
    fn packed_staves_stop_at_the_midline() {
        let img = synth_page(2000, 400, &[100, 240]);
        let crops = detect_staves(&img).expect("detects");
        assert_eq!(crops.len(), 2);
        let (_, ay, _, ah) = crops[0].bbox;
        let (_, by, _, _bh) = crops[1].bbox;
        assert!(ay + ah <= by, "bands must not overlap ({ay}+{ah} vs {by})");
        assert!(ay <= 100 && ay + ah >= 140 + 2, "staff A span kept");
        for crop in &crops {
            assert_eq!(
                crop.bbox.3 + crop.padding.top + crop.padding.bottom,
                crop.h,
                "source height plus padding equals canvas height"
            );
            assert!(
                fits_canvas_budget(crop.w, crop.h, 128, 1280),
                "packed staff canvas fits without overlapping its neighbor"
            );
        }
    }

    /// bd-av64.14: monotonic safety — with no budget pressure and no close
    /// neighbor, the band is never TIGHTER than the classic 12-spacing form.
    #[test]
    fn unpressured_band_keeps_the_generous_margins() {
        let img = synth_page(800, 400, &[180]);
        let crops = detect_staves(&img).expect("detects");
        let (_, _, _, ch) = crops[0].bbox;
        // spacing 10 => staff span 40 + 2 x 40 margins = 120.
        assert!(ch >= 120, "band height {ch} tighter than the classic form");
    }

    #[test]
    fn minimum_letterbox_height_uses_exact_checked_ceiling_arithmetic() {
        assert_eq!(minimum_canvas_height(1280, 128, 1280).unwrap(), 128);
        assert_eq!(minimum_canvas_height(1281, 128, 1280).unwrap(), 129);
        assert_eq!(minimum_canvas_height(2000, 128, 1280).unwrap(), 200);
        assert_eq!(minimum_canvas_height(1, 128, 1280).unwrap(), 1);

        assert!(minimum_canvas_height(0, 128, 1280).is_err());
        assert!(minimum_canvas_height(1, 0, 1280).is_err());
        assert!(minimum_canvas_height(1, 128, 0).is_err());
        assert!(minimum_canvas_height(usize::MAX, 2, 1).is_err());
    }

    #[test]
    fn over_budget_crop_is_losslessly_centered_on_a_white_canvas() {
        let source: Vec<u8> = (0..18).map(|v| v as u8).collect();
        let source_bbox = (7, 11, 9, 2);
        let mut crop = StaffCrop {
            gray: source.clone(),
            w: 9,
            h: 2,
            bbox: source_bbox,
            lines: [0, 0, 1, 1, 1],
            globally_deskewed_raster_lines: [11, 11, 12, 12, 12],
            padding: StaffPadding::default(),
        };

        // ceil(9 * 4 / 8) = 5: three rows of pad, split 1 top / 2 bottom.
        letterbox_to_budget(&mut crop, 4, 8).expect("letterbox succeeds");

        assert_eq!(crop.bbox, source_bbox, "page-space bbox is invariant");
        assert_eq!((crop.w, crop.h), (9, 5));
        assert_eq!(
            crop.padding,
            StaffPadding {
                top: 1,
                right: 0,
                bottom: 2,
                left: 0,
            }
        );
        assert_eq!(crop.lines, [1, 1, 2, 2, 2], "line rows translate");
        assert!(crop.gray[..9].iter().all(|&pixel| pixel == 255));
        assert_eq!(&crop.gray[9..27], source.as_slice(), "source bytes exact");
        assert!(crop.gray[27..].iter().all(|&pixel| pixel == 255));
        assert_eq!(
            crop.geometry(),
            StaffCropGeometry {
                source_bbox,
                canvas_width: 9,
                canvas_height: 5,
                padding: crop.padding,
            }
        );
    }

    #[test]
    fn fitting_crop_is_not_padded_or_reallocated() {
        let source = vec![42u8; 8 * 4];
        let mut crop = StaffCrop {
            gray: source.clone(),
            w: 8,
            h: 4,
            bbox: (2, 3, 8, 4),
            lines: [0, 1, 2, 3, 3],
            globally_deskewed_raster_lines: [3, 4, 5, 6, 6],
            padding: StaffPadding::default(),
        };
        letterbox_to_budget(&mut crop, 4, 8).expect("already fits");
        assert_eq!(crop.gray, source);
        assert_eq!(crop.h, 4);
        assert_eq!(crop.bbox, (2, 3, 8, 4));
        assert_eq!(crop.lines, [0, 1, 2, 3, 3]);
        assert_eq!(crop.padding, StaffPadding::default());
    }

    #[test]
    fn even_padding_is_exactly_symmetric() {
        let mut crop = StaffCrop {
            gray: vec![17u8; 8 * 2],
            w: 8,
            h: 2,
            bbox: (4, 5, 8, 2),
            lines: [0, 0, 1, 1, 1],
            globally_deskewed_raster_lines: [5, 5, 6, 6, 6],
            padding: StaffPadding::default(),
        };
        // ceil(8 * 4 / 8) = 4: one white row on each side.
        letterbox_to_budget(&mut crop, 4, 8).expect("letterbox succeeds");
        assert_eq!(crop.padding.top, 1);
        assert_eq!(crop.padding.bottom, 1);
        assert!(crop.gray[..8].iter().all(|&pixel| pixel == 255));
        assert!(crop.gray[24..].iter().all(|&pixel| pixel == 255));
    }

    #[test]
    fn letterbox_rejects_inconsistent_or_overflowing_geometry() {
        let mut malformed = StaffCrop {
            gray: vec![0; 7],
            w: 9,
            h: 2,
            bbox: (0, 0, 9, 2),
            lines: [0, 0, 1, 1, 1],
            globally_deskewed_raster_lines: [0, 0, 1, 1, 1],
            padding: StaffPadding::default(),
        };
        assert!(letterbox_to_budget(&mut malformed, 4, 8).is_err());

        let mut overflow = StaffCrop {
            gray: Vec::new(),
            w: usize::MAX,
            h: 1,
            bbox: (0, 0, usize::MAX, 1),
            lines: [0; 5],
            globally_deskewed_raster_lines: [0; 5],
            padding: StaffPadding::default(),
        };
        assert!(letterbox_to_budget(&mut overflow, 2, 1).is_err());
    }

    #[test]
    fn otsu_separates_bimodal() {
        let mut v = vec![20u8; 500];
        v.extend(vec![230u8; 500]);
        let t = otsu_threshold(&v);
        // Convention: ink = v <= t, so t must include the dark mode and
        // exclude the light one.
        assert!((20..230).contains(&t), "threshold {t} between the modes");
    }
}
