//! Native PDF page rasterization in pure, memory-safe Rust (no FFI).
//!
//! `focr ocr file.pdf` renders each PDF page to an [`image::DynamicImage`] and
//! feeds it through the same preprocess + OCR pipeline a PNG/JPG would take.
//!
//! ## Scope: bounded native PDF page paths
//!
//! A page with one full-page image XObject takes the scan fast path and decodes
//! it to RGB/gray with pure-Rust codecs: [`image`]'s JPEG via `zune-jpeg`,
//! `flate2`/`miniz_oxide` for `FlateDecode`, [`fax`] for CCITT Group 4,
//! `hayro-jpeg2000` for `JPXDecode`, and `hayro-jbig2` for `JBIG2Decode`.
//! It also handles the common layered MRC scan shape: one low-resolution opaque
//! background followed by one or more higher-resolution image layers whose
//! `/SMask` is a JBIG2 bitmap. A page with no image XObjects takes a separate
//! bounded renderer for the exact vector/path and embedded TrueType glyph dialect
//! emitted by MTDT's score-PDF writer plus the Type1C subset emitted by LilyPond.
//! [`PdfPages::selectable_text`] decodes MTDT's bounded Type0/Identity-H plus
//! complete ToUnicode representation and LilyPond's Type1C/WinAnsi + Differences
//! representation, explicitly ledgering music glyphs that have no Unicode.
//! Everything here is pure Rust with no C/C++ FFI or helper executable.
//!
//! ## Honest limits
//!
//! The exact MTDT writer dialect is not a claim to general born-digital PDF
//! rendering. Vector/text operators, fonts, CMaps, or paint semantics outside
//! that dialect, partial-page image mosaics, nested Form XObjects, skewed layers,
//! and general PDF blend modes surface as [`FocrError::InputDecode`] or a typed
//! [`PdfSelectableTextError`], naming the exact provider capability that must be
//! implemented in this module.

use std::io::Read;
use std::path::Path;

use hayro_jbig2::{Decoder as Jbig2Decoder, Image as Jbig2Image};
use hayro_jpeg2000::{ColorSpace, DecodeSettings, DecoderContext, Image as Jp2Image};
use image::{DynamicImage, GrayImage, ImageBuffer, RgbImage};
use lopdf::xobject::PdfImage;
use lopdf::{Dictionary, Document, Object, ObjectId, Stream};

use crate::error::{FocrError, FocrResult};

#[path = "pdf_vector.rs"]
mod vector;

#[path = "pdf_text.rs"]
mod selectable_text;

pub use selectable_text::{
    PDF_SELECTABLE_TEXT_SCHEMA_V2, PdfSelectableTextEncodingV2, PdfSelectableTextError,
    PdfSelectableTextErrorKind, PdfSelectableTextPageV2, PdfSelectableTextRunV2, PdfTextPositionV1,
};

/// The 5-byte header every PDF begins with (`%PDF-`).
const PDF_MAGIC: &[u8] = b"%PDF-";

/// Stable marker attached to provider-native PDF capability refusals.
///
/// Embedders may use this marker to distinguish a valid PDF that needs a
/// renderer capability from malformed encoded data. The full diagnostic
/// remains human-readable, but callers do not have to guess from incidental
/// codec wording.
pub const PDF_UNSUPPORTED_SUBSET_MARKER: &str = "franken_ocr.pdf.unsupported_subset";

/// Maximum decoded dimensions accepted from any PDF image or soft mask.
const MAX_PIXELS: u64 = 1 << 27;

/// `lopdf` parses from an in-memory buffer, so the source itself must be bounded
/// while it is read rather than checked through racy filesystem metadata. One
/// GiB leaves ample room for large archive volumes (the 266-page canonical
/// Spohr replay is 23,927,736 bytes) while placing a finite ceiling on parser
/// input and peak buffering. Larger collections should be split into volumes.
const MAX_PDF_INPUT_BYTES: u64 = 1 << 30;

const MAX_ENCODED_IMAGE_BYTES: usize = 512 * 1024 * 1024;
const MAX_PDF_CONTENT_ENCODED_BYTES: usize = 32 * 1024 * 1024;
const MAX_PDF_CONTENT_DECODED_BYTES: u64 = 64 * 1024 * 1024;
const MAX_PDF_CONTENT_OPERATIONS: usize = 1_000_000;

/// Content streams get a bounded graphics-state stack even if the PDF parser
/// accepted a hostile sequence of `q` operators.
const MAX_GRAPHICS_STATE_DEPTH: usize = 64;

/// Layered scans are a narrow page-composition path, not an unbounded image
/// stack. The estimate charges four bytes per decoded source pixel, one per mask
/// pixel, and eight bytes per canvas pixel for resize/composite working buffers.
const MAX_MRC_LAYERS: usize = 8;
const MAX_MRC_ESTIMATED_WORKING_BYTES: u128 = 768 * 1024 * 1024;

/// JBIG2 global segments are normally tiny dictionaries. This cap prevents a
/// filtered globals stream from becoming an independent inflate-bomb path.
const MAX_JBIG2_GLOBAL_BYTES: u64 = 64 * 1024 * 1024;

/// Whether `path` names a PDF: a `.pdf` extension, or a `%PDF-` magic prefix.
///
/// The magic check makes the routing robust to extension-less inputs; it reads
/// only the first few bytes and never fails the caller (an unreadable file just
/// returns `false` and is handled as a normal image path downstream).
#[must_use]
pub fn looks_like_pdf(path: &Path) -> bool {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("pdf"))
    {
        return true;
    }
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut head = [0u8; 5];
    matches!(file.read_exact(&mut head), Ok(())) && head == PDF_MAGIC
}

fn read_pdf_input<R: Read>(source: R, label: &str, max_bytes: u64) -> FocrResult<Vec<u8>> {
    let mut bounded = source.take(max_bytes.saturating_add(1));
    let mut bytes = Vec::new();
    bounded
        .read_to_end(&mut bytes)
        .map_err(|error| FocrError::InputDecode(format!("read PDF {label}: {error}")))?;

    let bytes_read = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if bytes_read > max_bytes {
        return Err(FocrError::InputDecode(format!(
            "PDF {label} exceeds native input limit of {max_bytes} bytes"
        )));
    }
    Ok(bytes)
}

fn parse_pdf_input(bytes: &[u8], label: &str) -> FocrResult<Document> {
    let document = match Document::load_mem(bytes) {
        Ok(document) => document,
        Err(lopdf::Error::InvalidPassword) => {
            return Err(FocrError::InputDecode(format!(
                "parse PDF {label}: encrypted PDFs are not supported"
            )));
        }
        Err(error) => {
            return Err(FocrError::InputDecode(format!(
                "parse PDF {label}: {error}"
            )));
        }
    };
    if document.was_encrypted() {
        return Err(FocrError::InputDecode(format!(
            "parse PDF {label}: encrypted PDFs are not supported"
        )));
    }
    Ok(document)
}

fn requires_unsupported_pdf_capability(detail: &str) -> bool {
    let normalized = detail.to_ascii_lowercase();
    [
        "outside the admitted mtdt",
        "unsupported",
        "not supported",
        "not implemented",
        "partial-page",
        "requires at least two painted image layers",
    ]
    .into_iter()
    .any(|marker| normalized.contains(marker))
}

fn pdf_page_render_error(page_one_based: usize, detail: String) -> FocrError {
    if requires_unsupported_pdf_capability(&detail) {
        FocrError::InputDecode(format!(
            "{PDF_UNSUPPORTED_SUBSET_MARKER}: PDF page {page_one_based}: {detail}"
        ))
    } else {
        FocrError::InputDecode(format!("PDF page {page_one_based}: {detail}"))
    }
}

/// A lazily-rendered PDF: the parsed document plus its page object ids in order.
///
/// Pages are rendered one at a time via [`PdfPages::render`] so a 600-page book
/// never materializes 600 rasters at once — the OCR driver pulls one page,
/// recognizes it, and drops it before the next.
pub struct PdfPages {
    /// Exact bounded parser input retained for the lifetime of every rendered
    /// page. This makes [`Self::source_bytes`] an exact-consumption surface, not
    /// a second read of a diagnostic path.
    source_bytes: std::sync::Arc<[u8]>,
    doc: Document,
    /// Page object ids in 1-based page order (the value of `get_pages`).
    pages: Vec<ObjectId>,
}

impl PdfPages {
    fn from_shared_bytes_with_label(
        source_bytes: std::sync::Arc<[u8]>,
        label: &str,
    ) -> FocrResult<Self> {
        let bytes_read = u64::try_from(source_bytes.len()).unwrap_or(u64::MAX);
        if bytes_read > MAX_PDF_INPUT_BYTES {
            return Err(FocrError::InputDecode(format!(
                "PDF {label} exceeds native input limit of {MAX_PDF_INPUT_BYTES} bytes"
            )));
        }
        let doc = parse_pdf_input(&source_bytes, label)?;
        let pages: Vec<ObjectId> = doc.get_pages().into_values().collect();
        if pages.is_empty() {
            return Err(FocrError::InputDecode(format!("PDF {label} has no pages")));
        }
        Ok(Self {
            source_bytes,
            doc,
            pages,
        })
    }

    fn from_owned_bytes_with_label(bytes: Vec<u8>, label: &str) -> FocrResult<Self> {
        Self::from_shared_bytes_with_label(bytes.into(), label)
    }

    pub(crate) fn from_shared_bytes(source_bytes: std::sync::Arc<[u8]>) -> FocrResult<Self> {
        Self::from_shared_bytes_with_label(source_bytes, "owned bytes")
    }

    /// Parse a PDF from an owned, immutable byte buffer without consulting the
    /// filesystem. The input is checked against the same native size bound as
    /// [`Self::open`] before parsing.
    ///
    /// This is the embeddable provenance-safe entry point for callers that have
    /// already pinned and hashed the exact PDF bytes they intend to render.
    /// Diagnostic path labels are intentionally absent from this API.
    ///
    /// # Errors
    /// [`FocrError::InputDecode`] if the buffer exceeds the native input bound,
    /// is not a supported PDF, or contains no pages.
    pub fn from_bytes(bytes: Vec<u8>) -> FocrResult<Self> {
        Self::from_owned_bytes_with_label(bytes, "owned bytes")
    }

    /// Read and parse a PDF from `source` without a filesystem reopen.
    ///
    /// The read is bounded to [`MAX_PDF_INPUT_BYTES`], and the exact bytes read
    /// are handed directly to the same owned-buffer parser as [`Self::from_bytes`].
    /// `label` is diagnostic only and never affects rendered content.
    ///
    /// # Errors
    /// [`FocrError::InputDecode`] if the reader fails, exceeds the native input
    /// bound, produces an unsupported PDF, or produces a document with no pages.
    pub fn from_reader<R: Read>(source: R, label: &str) -> FocrResult<Self> {
        let bytes = read_pdf_input(source, label, MAX_PDF_INPUT_BYTES)?;
        Self::from_owned_bytes_with_label(bytes, label)
    }

    /// Parse the PDF at `path`. Does not render any page yet.
    ///
    /// # Errors
    /// [`FocrError::InputDecode`] if the file cannot be parsed as a PDF.
    pub fn open(path: &Path) -> FocrResult<Self> {
        let label = path.display().to_string();
        let source = std::fs::File::open(path)
            .map_err(|error| FocrError::InputDecode(format!("open PDF {label}: {error}")))?;
        Self::from_reader(source, &label)
    }

    /// Number of pages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Whether the document has no pages (never true after [`Self::open`], which
    /// rejects empty documents — present for lint-clean `len()` ergonomics).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Exact bounded bytes parsed by this document, retained without reopening
    /// any path. The diagnostic label supplied to [`Self::from_reader`] is not
    /// part of this buffer.
    #[must_use]
    pub fn source_bytes(&self) -> &[u8] {
        &self.source_bytes
    }

    /// Decode selectable text from page `idx` (0-based) for the exact bounded
    /// Type0/Identity-H subset emitted by MTDT or the Type1C/WinAnsi +
    /// Differences subset emitted by LilyPond.
    ///
    /// The method consumes this instance's retained immutable bytes and the
    /// same bounded page-content operation stream used by [`Self::render`]. It
    /// never reopens a path or invokes a process. Type0 text requires complete
    /// embedded ToUnicode mappings; opaque Type1C music glyphs retain their
    /// exact source code and glyph name without guessed Unicode.
    ///
    /// # Errors
    /// [`PdfSelectableTextError`] with page, operation, and font context when
    /// the page is out of range, exceeds a bound, has invalid text state, or
    /// uses a font/CMap/text operator outside the admitted writer dialect.
    pub fn selectable_text(
        &self,
        idx: usize,
    ) -> Result<PdfSelectableTextPageV2, PdfSelectableTextError> {
        let page_id = *self.pages.get(idx).ok_or_else(|| PdfSelectableTextError {
            kind: PdfSelectableTextErrorKind::PageOutOfRange,
            page_index: idx,
            operation_index: None,
            font_resource: None,
            detail: format!("PDF page index {idx} out of range ({})", self.len()),
        })?;
        let content =
            bounded_page_content(&self.doc, page_id).map_err(|detail| PdfSelectableTextError {
                kind: PdfSelectableTextErrorKind::PageContent,
                page_index: idx,
                operation_index: None,
                font_resource: None,
                detail,
            })?;
        selectable_text::extract_page(&self.doc, page_id, idx, &self.source_bytes, &content)
    }

    /// Render page `idx` (0-based) to a [`DynamicImage`], applying the page's
    /// `/Rotate`.
    ///
    /// A page with no image XObjects takes the exact bounded MTDT vector/text
    /// path. A one-image page takes the original largest-image scan path. A page
    /// with multiple image XObjects is interpreted in content-stream `Do` order
    /// and accepted only when it is the bounded, full-page MRC shape described in
    /// the module docs.
    ///
    /// # Errors
    /// [`FocrError::InputDecode`] if the selected path exceeds a bound, cannot be
    /// decoded, or requires a PDF composition capability outside this native
    /// renderer.
    pub fn render(&self, idx: usize) -> FocrResult<DynamicImage> {
        let page_id = *self.pages.get(idx).ok_or_else(|| {
            FocrError::InputDecode(format!(
                "PDF page index {idx} out of range ({})",
                self.len()
            ))
        })?;

        let images = page_images(&self.doc, page_id).map_err(|e| {
            FocrError::InputDecode(format!("read images on PDF page {}: {e}", idx + 1))
        })?;

        let (decoded, content_rotation) = match images.as_slice() {
            [] => {
                let content = bounded_page_content(&self.doc, page_id).map_err(|error| {
                    FocrError::InputDecode(format!("PDF page {} vector content: {error}", idx + 1))
                })?;
                let bounds = page_bounds(&self.doc, page_id).map_err(|error| {
                    FocrError::InputDecode(format!("PDF page {} vector geometry: {error}", idx + 1))
                })?;
                let image = vector::render_mtdt_vector_page(&self.doc, page_id, &content, bounds)
                    .map_err(|error| {
                    pdf_page_render_error(idx + 1, format!("native vector/text render: {error}"))
                })?;
                Ok((image, 0))
            }
            [main] => decode_image_xobject(&self.doc, main).and_then(|image| {
                content_rotation(&self.doc, page_id).map(|rotation| (image, rotation))
            }),
            _ => render_layered_mrc(&self.doc, page_id, &images),
        }
        .map_err(|error| pdf_page_render_error(idx + 1, error))?;

        let total_rotation = (page_rotation(&self.doc, page_id) + content_rotation).rem_euclid(360);
        Ok(apply_rotation(decoded, total_rotation))
    }
}

#[derive(Clone, Copy, Debug)]
struct Matrix {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
}

impl Matrix {
    const IDENTITY: Self = Self {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        e: 0.0,
        f: 0.0,
    };

    /// PDF's `cm` concatenates the supplied matrix with the current CTM; it does
    /// not replace it. Coordinates use `x' = a*x + c*y + e` and
    /// `y' = b*x + d*y + f`.
    fn concat(self, rhs: Self) -> Self {
        Self {
            a: self.a * rhs.a + self.c * rhs.b,
            b: self.b * rhs.a + self.d * rhs.b,
            c: self.a * rhs.c + self.c * rhs.d,
            d: self.b * rhs.c + self.d * rhs.d,
            e: self.a * rhs.e + self.c * rhs.f + self.e,
            f: self.b * rhs.e + self.d * rhs.f + self.f,
        }
    }

    fn is_finite(self) -> bool {
        [self.a, self.b, self.c, self.d, self.e, self.f]
            .into_iter()
            .all(f64::is_finite)
    }

    fn approximately_equals(self, rhs: Self) -> bool {
        [
            (self.a, rhs.a),
            (self.b, rhs.b),
            (self.c, rhs.c),
            (self.d, rhs.d),
            (self.e, rhs.e),
            (self.f, rhs.f),
        ]
        .into_iter()
        .all(|(left, right)| {
            let scale = left.abs().max(right.abs()).max(1.0);
            (left - right).abs() <= scale * 1.0e-6
        })
    }

    fn point(self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.c * y + self.e,
            self.b * x + self.d * y + self.f,
        )
    }
}

#[derive(Clone, Debug)]
struct ImagePaint {
    id: ObjectId,
    name: Vec<u8>,
    matrix: Matrix,
    operation_index: usize,
}

/// Render the deliberately bounded layered-MRC subset used by scanned archives.
/// The function resolves `Do` names through page resources; resource-dictionary
/// iteration order is never treated as paint order.
fn render_layered_mrc(
    doc: &Document,
    page_id: ObjectId,
    images: &[PdfImage<'_>],
) -> Result<(DynamicImage, i64), String> {
    let (paints, content) = ordered_image_paints(doc, page_id)?;
    if paints.len() != images.len() {
        return Err(format!(
            "layered page declares {} image XObjects but paints {}; nested Form XObjects, \
             unused image resources, and repeated image invocations are not implemented in \
             franken_ocr::pdf::PdfPages",
            images.len(),
            paints.len()
        ));
    }
    if paints.len() < 2 {
        return Err("layered MRC rendering requires at least two painted image layers".to_string());
    }
    if paints.len() > MAX_MRC_LAYERS {
        return Err(format!(
            "layered page paints {} image layers, exceeding the {MAX_MRC_LAYERS}-layer MRC limit",
            paints.len()
        ));
    }
    for pair in paints.windows(2) {
        if pair[1].operation_index != pair[0].operation_index + 1 {
            return Err(format!(
                "layered image operators /{} and /{} are not consecutive; intervening PDF \
                 painting or graphics-state changes are not implemented",
                String::from_utf8_lossy(&pair[0].name),
                String::from_utf8_lossy(&pair[1].name)
            ));
        }
    }
    validate_mrc_content(
        &content,
        paints[0].operation_index,
        paints.last().expect("nonempty").operation_index,
    )?;
    if content
        .operations
        .iter()
        .skip(paints.last().expect("nonempty").operation_index + 1)
        .any(|op| op.operator != "Q")
    {
        return Err(
            "PDF paints or mutates graphics state after the layered scan; general post-image \
             content composition is not implemented in franken_ocr::pdf::PdfPages"
                .to_string(),
        );
    }

    let first_matrix = paints[0].matrix;
    if paints
        .iter()
        .skip(1)
        .any(|paint| !paint.matrix.approximately_equals(first_matrix))
    {
        return Err(
            "layered scan image XObjects use different CTMs; partial-page image mosaics are \
             not implemented in franken_ocr::pdf::PdfPages"
                .to_string(),
        );
    }
    let page_bounds = page_bounds(doc, page_id)?;
    validate_full_page_axis_aligned(first_matrix, page_bounds)?;

    let mut ordered: Vec<&PdfImage<'_>> = Vec::with_capacity(paints.len());
    for paint in &paints {
        let image = images
            .iter()
            .find(|image| image.id == paint.id)
            .ok_or_else(|| {
                format!(
                    "content stream paints /{} ({:?}), but lopdf did not expose that image XObject",
                    String::from_utf8_lossy(&paint.name),
                    paint.id
                )
            })?;
        if ordered.iter().any(|prior| prior.id == image.id) {
            return Err(format!(
                "image XObject /{} is painted more than once; repeated-layer composition is not \
                 implemented in franken_ocr::pdf::PdfPages",
                String::from_utf8_lossy(&paint.name)
            ));
        }
        ordered.push(image);
    }

    let base = ordered[0];
    if base.origin_dict.has(b"SMask")
        || base.origin_dict.has(b"Mask")
        || base.origin_dict.has(b"SMaskInData")
    {
        return Err(
            "the first MRC layer must be opaque and may not use /SMask, /Mask, or /SMaskInData"
                .to_string(),
        );
    }
    for overlay in ordered.iter().skip(1) {
        if !overlay.origin_dict.has(b"SMask")
            || overlay.origin_dict.has(b"Mask")
            || overlay.origin_dict.has(b"SMaskInData")
        {
            return Err(
                "every MRC overlay must use /SMask and may not use /Mask or /SMaskInData"
                    .to_string(),
            );
        }
    }

    let (canvas_width, canvas_height) = ordered
        .iter()
        .map(|image| checked_dimensions(image.width, image.height))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max_by_key(|(width, height)| u64::from(*width) * u64::from(*height))
        .expect("at least two layers");
    validate_mrc_budget(doc, &ordered, canvas_width, canvas_height)?;
    for image in &ordered {
        let (width, height) = checked_dimensions(image.width, image.height)?;
        let left = u128::from(width) * u128::from(canvas_height);
        let right = u128::from(height) * u128::from(canvas_width);
        let scale = left.max(right).max(1);
        if left.abs_diff(right) * 500 > scale {
            return Err(format!(
                "MRC layer {}x{} does not share the canvas aspect ratio {}x{}",
                width, height, canvas_width, canvas_height
            ));
        }
    }

    let decoded_base = decode_image_xobject(doc, base)?;
    let mut canvas = resize_layer(
        decoded_base,
        canvas_width,
        canvas_height,
        interpolation_filter(base.origin_dict),
    )
    .to_rgb8();

    for overlay in ordered.iter().skip(1) {
        let decoded = decode_image_xobject(doc, overlay)?;
        let foreground = resize_layer(
            decoded,
            canvas_width,
            canvas_height,
            interpolation_filter(overlay.origin_dict),
        )
        .to_rgb8();
        let mask_id = overlay
            .origin_dict
            .get(b"SMask")
            .and_then(Object::as_reference)
            .map_err(|e| format!("MRC /SMask must be an indirect image stream: {e}"))?;
        let mask_stream = doc
            .get_object(mask_id)
            .and_then(Object::as_stream)
            .map_err(|e| format!("read MRC soft-mask stream {mask_id:?}: {e}"))?;
        let mask = decode_jbig2_stream(doc, mask_stream)?;
        let overlay_dimensions = checked_dimensions(overlay.width, overlay.height)?;
        if mask.dimensions() != overlay_dimensions {
            return Err(format!(
                "MRC soft mask {}x{} does not match its foreground layer {}x{}",
                mask.width(),
                mask.height(),
                overlay_dimensions.0,
                overlay_dimensions.1
            ));
        }
        let mask = resize_gray(
            mask,
            canvas_width,
            canvas_height,
            interpolation_filter(&mask_stream.dict),
        );
        alpha_composite(&mut canvas, &foreground, &mask)?;
    }

    Ok((
        DynamicImage::ImageRgb8(canvas),
        matrix_rotation(first_matrix),
    ))
}

fn validate_mrc_content(
    content: &lopdf::content::Content<Vec<lopdf::content::Operation>>,
    first_image: usize,
    last_image: usize,
) -> Result<(), String> {
    let mut text_rendering_mode = 0i64;
    let mut stack = Vec::new();
    for (index, operation) in content.operations.iter().enumerate() {
        match operation.operator.as_str() {
            "q" => stack.push(text_rendering_mode),
            "Q" => {
                text_rendering_mode = stack.pop().ok_or_else(|| {
                    "unbalanced graphics state while validating MRC content".to_string()
                })?;
            }
            "gs" => {
                return Err(
                    "PDF /ExtGState application (gs), including blend and alpha state, is not \
                     implemented by the layered MRC renderer"
                        .to_string(),
                );
            }
            "Tr" if index < first_image => {
                if operation.operands.len() != 1 {
                    return Err("PDF Tr operator must have one numeric operand".to_string());
                }
                let mode_value = f64::from(
                    operation.operands[0]
                        .as_float()
                        .map_err(|e| format!("invalid PDF text rendering mode: {e}"))?,
                );
                if !mode_value.is_finite() || mode_value.fract() != 0.0 {
                    return Err(format!(
                        "PDF text rendering mode {mode_value} is not an integer"
                    ));
                }
                let mode = mode_value as i64;
                if !(0..=7).contains(&mode) {
                    return Err(format!("PDF text rendering mode {mode} is outside 0..=7"));
                }
                text_rendering_mode = mode;
            }
            "Tj" | "TJ" | "'" | "\"" if index < first_image => {
                if text_rendering_mode != 3 {
                    return Err(format!(
                        "visible pre-image text uses rendering mode {text_rendering_mode}; only \
                         invisible OCR text (Tr 3) may precede an MRC scan"
                    ));
                }
            }
            "S" | "s" | "f" | "F" | "f*" | "B" | "B*" | "b" | "b*" | "sh" | "BI" | "ID" | "EI"
            | "W" | "W*"
                if index < first_image =>
            {
                return Err(format!(
                    "visible or clipping PDF operator {} precedes the MRC image stack; general \
                     vector, inline-image, shading, and clipping composition is not implemented",
                    operation.operator
                ));
            }
            _ if index > last_image && operation.operator != "Q" => {
                return Err(format!(
                    "PDF operator {} follows the MRC image stack; post-image composition is not implemented",
                    operation.operator
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_mrc_budget(
    doc: &Document,
    layers: &[&PdfImage<'_>],
    canvas_width: u32,
    canvas_height: u32,
) -> Result<(), String> {
    let mut working_bytes = u128::from(canvas_width) * u128::from(canvas_height) * 8;
    let mut encoded_bytes = 0usize;
    for layer in layers {
        encoded_bytes = encoded_bytes
            .checked_add(layer.content.len())
            .ok_or_else(|| "MRC aggregate encoded-size overflow".to_string())?;
        let (width, height) = checked_dimensions(layer.width, layer.height)?;
        working_bytes = working_bytes
            .checked_add(u128::from(width) * u128::from(height) * 4)
            .ok_or_else(|| "MRC estimated working-set overflow".to_string())?;
        if let Ok(mask_id) = layer
            .origin_dict
            .get(b"SMask")
            .and_then(Object::as_reference)
        {
            let mask = doc
                .get_object(mask_id)
                .and_then(Object::as_stream)
                .map_err(|e| format!("read MRC soft-mask stream {mask_id:?}: {e}"))?;
            let mask_width = mask
                .dict
                .get(b"Width")
                .and_then(Object::as_i64)
                .map_err(|e| format!("MRC soft mask has no valid /Width: {e}"))?;
            let mask_height = mask
                .dict
                .get(b"Height")
                .and_then(Object::as_i64)
                .map_err(|e| format!("MRC soft mask has no valid /Height: {e}"))?;
            let (mask_width, mask_height) = checked_dimensions(mask_width, mask_height)?;
            if (mask_width, mask_height) != (width, height) {
                return Err(format!(
                    "MRC soft mask {mask_width}x{mask_height} does not match its foreground layer \
                     {width}x{height}"
                ));
            }
            encoded_bytes = encoded_bytes
                .checked_add(mask.content.len())
                .ok_or_else(|| "MRC aggregate encoded-size overflow".to_string())?;
            working_bytes = working_bytes
                .checked_add(u128::from(mask_width) * u128::from(mask_height))
                .ok_or_else(|| "MRC estimated working-set overflow".to_string())?;
        }
    }
    if working_bytes > MAX_MRC_ESTIMATED_WORKING_BYTES {
        return Err(format!(
            "layered scan has a {working_bytes}-byte estimated working set, exceeding the \
             {MAX_MRC_ESTIMATED_WORKING_BYTES}-byte MRC limit"
        ));
    }
    if encoded_bytes > MAX_ENCODED_IMAGE_BYTES {
        return Err(format!(
            "layered scan contains {encoded_bytes} encoded image bytes, exceeding the \
             {MAX_ENCODED_IMAGE_BYTES}-byte aggregate MRC limit"
        ));
    }
    Ok(())
}

fn resize_layer(
    image: DynamicImage,
    width: u32,
    height: u32,
    filter: image::imageops::FilterType,
) -> DynamicImage {
    if image.width() == width && image.height() == height {
        image
    } else {
        image.resize_exact(width, height, filter)
    }
}

fn resize_gray(
    image: GrayImage,
    width: u32,
    height: u32,
    filter: image::imageops::FilterType,
) -> GrayImage {
    if image.width() == width && image.height() == height {
        image
    } else {
        image::imageops::resize(&image, width, height, filter)
    }
}

fn interpolation_filter(dict: &Dictionary) -> image::imageops::FilterType {
    if dict
        .get(b"Interpolate")
        .and_then(Object::as_bool)
        .unwrap_or(false)
    {
        image::imageops::FilterType::Triangle
    } else {
        image::imageops::FilterType::Nearest
    }
}

fn alpha_composite(
    background: &mut RgbImage,
    foreground: &RgbImage,
    alpha: &GrayImage,
) -> Result<(), String> {
    if background.dimensions() != foreground.dimensions()
        || background.dimensions() != alpha.dimensions()
    {
        return Err("MRC foreground, background, and soft mask dimensions differ".to_string());
    }
    for ((background, foreground), alpha) in background
        .pixels_mut()
        .zip(foreground.pixels())
        .zip(alpha.pixels())
    {
        let opacity = u16::from(alpha.0[0]);
        for channel in 0..3 {
            background.0[channel] = ((u16::from(foreground.0[channel]) * opacity
                + u16::from(background.0[channel]) * (255 - opacity)
                + 127)
                / 255) as u8;
        }
    }
    Ok(())
}

fn ordered_image_paints(
    doc: &Document,
    page_id: ObjectId,
) -> Result<
    (
        Vec<ImagePaint>,
        lopdf::content::Content<Vec<lopdf::content::Operation>>,
    ),
    String,
> {
    let content = bounded_page_content(doc, page_id)?;
    let xobjects = effective_xobjects(doc, page_id)?;
    let mut ctm = Matrix::IDENTITY;
    let mut stack = Vec::new();
    let mut paints = Vec::new();

    for (operation_index, operation) in content.operations.iter().enumerate() {
        match operation.operator.as_str() {
            "q" => {
                if stack.len() >= MAX_GRAPHICS_STATE_DEPTH {
                    return Err(format!(
                        "PDF graphics-state stack exceeds {MAX_GRAPHICS_STATE_DEPTH} entries"
                    ));
                }
                stack.push(ctm);
            }
            "Q" => {
                ctm = stack
                    .pop()
                    .ok_or_else(|| "unbalanced PDF graphics-state restore (Q)".to_string())?;
            }
            "cm" => {
                if operation.operands.len() != 6 {
                    return Err(
                        "PDF cm operator must have exactly six numeric operands".to_string()
                    );
                }
                let mut values = [0.0; 6];
                for (slot, operand) in values.iter_mut().zip(&operation.operands) {
                    *slot = f64::from(
                        operand
                            .as_float()
                            .map_err(|e| format!("PDF cm operand is not numeric: {e}"))?,
                    );
                }
                let next = Matrix {
                    a: values[0],
                    b: values[1],
                    c: values[2],
                    d: values[3],
                    e: values[4],
                    f: values[5],
                };
                if !next.is_finite() {
                    return Err("PDF cm operator contains a non-finite value".to_string());
                }
                ctm = ctm.concat(next);
                if !ctm.is_finite() {
                    return Err("cumulative PDF image CTM is non-finite".to_string());
                }
            }
            "Do" => {
                let name = operation
                    .operands
                    .as_slice()
                    .first()
                    .filter(|_| operation.operands.len() == 1)
                    .ok_or_else(|| {
                        "PDF Do operator must have exactly one name operand".to_string()
                    })?
                    .as_name()
                    .map_err(|e| format!("PDF Do operand is not an XObject name: {e}"))?;
                let id = xobjects
                    .iter()
                    .find_map(|(candidate, id)| (candidate.as_slice() == name).then_some(*id))
                    .ok_or_else(|| {
                        format!(
                            "PDF content references missing XObject /{}",
                            String::from_utf8_lossy(name)
                        )
                    })?;
                let stream = doc
                    .get_object(id)
                    .and_then(Object::as_stream)
                    .map_err(|e| format!("read XObject /{}: {e}", String::from_utf8_lossy(name)))?;
                let subtype = stream
                    .dict
                    .get(b"Subtype")
                    .and_then(Object::as_name)
                    .map_err(|e| {
                        format!(
                            "XObject /{} has no subtype: {e}",
                            String::from_utf8_lossy(name)
                        )
                    })?;
                if subtype != b"Image" {
                    return Err(format!(
                        "content invokes /{} with subtype /{}; nested Form XObject rendering is \
                         not implemented in franken_ocr::pdf::PdfPages",
                        String::from_utf8_lossy(name),
                        String::from_utf8_lossy(subtype)
                    ));
                }
                paints.push(ImagePaint {
                    id,
                    name: name.to_vec(),
                    matrix: ctm,
                    operation_index,
                });
            }
            _ => {}
        }
    }
    if !stack.is_empty() {
        return Err("unbalanced PDF graphics-state save (q)".to_string());
    }
    Ok((paints, content))
}

fn effective_xobjects(
    doc: &Document,
    page_id: ObjectId,
) -> Result<Vec<(Vec<u8>, ObjectId)>, String> {
    // /Resources is inherited as one whole dictionary. Once a nearer page-tree
    // node defines it, ancestor dictionaries do not merge into or supplement it.
    let mut node_id = page_id;
    let resources = (0..64)
        .find_map(|_| {
            let node = doc.get_dictionary(node_id).ok()?;
            if let Ok(resources) = node.get(b"Resources") {
                return Some(Ok(resources));
            }
            node_id = match node.get(b"Parent").and_then(Object::as_reference) {
                Ok(parent) => parent,
                Err(_) => return Some(Err("PDF page has no inherited /Resources".to_string())),
            };
            None
        })
        .unwrap_or_else(|| Err("PDF page /Resources inheritance exceeds 64 nodes".to_string()))?;
    let (_, resources) = doc
        .dereference(resources)
        .map_err(|e| format!("dereference page /Resources: {e}"))?;
    let resources = resources
        .as_dict()
        .map_err(|e| format!("page /Resources is not a dictionary: {e}"))?;
    let mut xobjects: Vec<(Vec<u8>, ObjectId)> = Vec::new();
    if let Ok(object) = resources.get(b"XObject") {
        let (_, object) = doc
            .dereference(object)
            .map_err(|e| format!("dereference /XObject resources: {e}"))?;
        let dict = object
            .as_dict()
            .map_err(|e| format!("page /XObject resource is not a dictionary: {e}"))?;
        for (name, object) in dict.iter() {
            if xobjects
                .iter()
                .any(|(existing, _)| existing.as_slice() == name.as_slice())
            {
                continue;
            }
            let id = object.as_reference().map_err(|e| {
                format!(
                    "XObject /{} is not an indirect stream: {e}",
                    String::from_utf8_lossy(name)
                )
            })?;
            xobjects.push((name.clone(), id));
        }
    }
    Ok(xobjects)
}

fn page_images<'a>(doc: &'a Document, page_id: ObjectId) -> Result<Vec<PdfImage<'a>>, String> {
    let mut images = Vec::new();
    for (_, id) in effective_xobjects(doc, page_id)? {
        let stream = doc
            .get_object(id)
            .and_then(Object::as_stream)
            .map_err(|e| format!("read page XObject {id:?}: {e}"))?;
        let subtype = stream
            .dict
            .get(b"Subtype")
            .and_then(Object::as_name)
            .map_err(|e| format!("page XObject {id:?} has no valid /Subtype: {e}"))?;
        if subtype != b"Image" {
            continue;
        }
        let width = stream
            .dict
            .get(b"Width")
            .and_then(Object::as_i64)
            .map_err(|e| format!("image XObject {id:?} has no valid /Width: {e}"))?;
        let height = stream
            .dict
            .get(b"Height")
            .and_then(Object::as_i64)
            .map_err(|e| format!("image XObject {id:?} has no valid /Height: {e}"))?;
        let color_space = match stream.dict.get(b"ColorSpace") {
            Ok(Object::Name(name)) => Some(String::from_utf8_lossy(name).into_owned()),
            Ok(Object::Array(items)) => items
                .first()
                .and_then(|item| item.as_name().ok())
                .map(|name| String::from_utf8_lossy(name).into_owned()),
            _ => None,
        };
        let bits_per_component = stream
            .dict
            .get(b"BitsPerComponent")
            .and_then(Object::as_i64)
            .ok();
        images.push(PdfImage {
            id,
            width,
            height,
            color_space,
            filters: Some(stream_filters(&stream.dict)?),
            bits_per_component,
            content: &stream.content,
            origin_dict: &stream.dict,
        });
    }
    Ok(images)
}

fn bounded_page_content(
    doc: &Document,
    page_id: ObjectId,
) -> Result<lopdf::content::Content<Vec<lopdf::content::Operation>>, String> {
    let mut encoded_total = 0usize;
    let mut decoded = Vec::new();
    for content_id in doc.get_page_contents(page_id) {
        let stream = doc
            .get_object(content_id)
            .and_then(Object::as_stream)
            .map_err(|e| format!("read PDF page content stream {content_id:?}: {e}"))?;
        encoded_total = encoded_total
            .checked_add(stream.content.len())
            .ok_or_else(|| "PDF page content encoded-size overflow".to_string())?;
        if encoded_total > MAX_PDF_CONTENT_ENCODED_BYTES {
            return Err(format!(
                "PDF page content exceeds the {MAX_PDF_CONTENT_ENCODED_BYTES}-byte encoded limit"
            ));
        }
        let remaining = MAX_PDF_CONTENT_DECODED_BYTES
            .checked_sub(decoded.len() as u64)
            .ok_or_else(|| "PDF page content decoded-size limit exhausted".to_string())?;
        let filters = stream_filters(&stream.dict)?;
        let bytes = match filters.as_slice() {
            [] => {
                if stream.content.len() as u64 > remaining {
                    return Err(format!(
                        "PDF page content exceeds the {MAX_PDF_CONTENT_DECODED_BYTES}-byte decoded limit"
                    ));
                }
                stream.content.clone()
            }
            [filter] if filter == "FlateDecode" && stream_predictor(stream) <= 1 => {
                bounded_inflate(&stream.content, remaining)?.ok_or_else(|| {
                    "sole-Flate PDF page content is not a valid bounded zlib stream".to_string()
                })?
            }
            _ => {
                return Err(format!(
                    "PDF page content filters {filters:?} are not implemented by the bounded \
                     franken_ocr renderer; only unfiltered or sole FlateDecode content is accepted"
                ));
            }
        };
        decoded.extend_from_slice(&bytes);
        decoded.push(b'\n');
        if decoded.len() as u64 > MAX_PDF_CONTENT_DECODED_BYTES {
            return Err(format!(
                "PDF page content exceeds the {MAX_PDF_CONTENT_DECODED_BYTES}-byte decoded limit"
            ));
        }
    }
    if decoded.is_empty() {
        return Ok(lopdf::content::Content {
            operations: Vec::new(),
        });
    }
    let content = lopdf::content::Content::decode(&decoded)
        .map_err(|e| format!("parse decoded PDF page content: {e}"))?;
    if content.operations.len() > MAX_PDF_CONTENT_OPERATIONS {
        return Err(format!(
            "PDF page content has {} operations, exceeding the {MAX_PDF_CONTENT_OPERATIONS}-operation limit",
            content.operations.len()
        ));
    }
    Ok(content)
}

fn page_bounds(doc: &Document, page_id: ObjectId) -> Result<[f64; 4], String> {
    let object = inherited(doc, page_id, b"CropBox")
        .or_else(|| inherited(doc, page_id, b"MediaBox"))
        .ok_or_else(|| "PDF page has neither an inherited /CropBox nor /MediaBox".to_string())?;
    let (_, object) = doc
        .dereference(object)
        .map_err(|e| format!("dereference PDF page box: {e}"))?;
    let values = object
        .as_array()
        .map_err(|e| format!("PDF page box is not an array: {e}"))?;
    if values.len() != 4 {
        return Err("PDF page box must contain exactly four numbers".to_string());
    }
    let mut bounds = [0.0; 4];
    for (slot, value) in bounds.iter_mut().zip(values) {
        *slot = f64::from(
            value
                .as_float()
                .map_err(|e| format!("PDF page box coordinate is not numeric: {e}"))?,
        );
    }
    if !bounds.into_iter().all(f64::is_finite) || bounds[2] <= bounds[0] || bounds[3] <= bounds[1] {
        return Err(format!("invalid PDF page bounds {bounds:?}"));
    }
    Ok(bounds)
}

fn validate_full_page_axis_aligned(matrix: Matrix, bounds: [f64; 4]) -> Result<(), String> {
    let scale = [matrix.a, matrix.b, matrix.c, matrix.d]
        .into_iter()
        .map(f64::abs)
        .fold(1.0, f64::max);
    let epsilon = scale * 1.0e-7;
    let axis_aligned = (matrix.b.abs() <= epsilon && matrix.c.abs() <= epsilon)
        || (matrix.a.abs() <= epsilon && matrix.d.abs() <= epsilon);
    let determinant = matrix.a * matrix.d - matrix.b * matrix.c;
    if !axis_aligned || determinant <= epsilon {
        return Err(
            "layered scan CTM is skewed, reflected, or degenerate; only orientation-preserving \
             axis-aligned full-page placement is implemented in franken_ocr::pdf::PdfPages"
                .to_string(),
        );
    }
    let corners = [
        matrix.point(0.0, 0.0),
        matrix.point(1.0, 0.0),
        matrix.point(0.0, 1.0),
        matrix.point(1.0, 1.0),
    ];
    let rendered = [
        corners
            .iter()
            .map(|point| point.0)
            .fold(f64::INFINITY, f64::min),
        corners
            .iter()
            .map(|point| point.1)
            .fold(f64::INFINITY, f64::min),
        corners
            .iter()
            .map(|point| point.0)
            .fold(f64::NEG_INFINITY, f64::max),
        corners
            .iter()
            .map(|point| point.1)
            .fold(f64::NEG_INFINITY, f64::max),
    ];
    let page_scale = (bounds[2] - bounds[0]).max(bounds[3] - bounds[1]).max(1.0);
    let tolerance = page_scale * 1.0e-4;
    if rendered
        .into_iter()
        .zip(bounds)
        .any(|(actual, expected)| (actual - expected).abs() > tolerance)
    {
        return Err(format!(
            "layered scan CTM covers {rendered:?}, not the full PDF page {bounds:?}; partial-page \
             image placement is not implemented in franken_ocr::pdf::PdfPages"
        ));
    }
    Ok(())
}

/// Rotation (0/90/180/270 degrees, clockwise-positive like `/Rotate`) the
/// page CONTENT STREAM applies to its main image through the current
/// transformation matrix. Scanned-book PDFs often store the scan portrait
/// and place it with a rotated `cm` instead of a `/Rotate` entry — the
/// Cadwallader class: `/Rotate 0`, image 2480x3504, displayed landscape.
/// Ignoring the CTM fed the OCR model SIDEWAYS pages (bd-av64.11,
/// 2026-07-06: a spread decoded garbage until the 600s forward budget).
///
/// Only the matrix in effect at the FIRST `Do` is classified, and only
/// axis-aligned rotations are recognized (a skewed/general matrix returns
/// 0 — leave the raster as stored rather than guess).
fn content_rotation(doc: &Document, page_id: ObjectId) -> Result<i64, String> {
    let content = bounded_page_content(doc, page_id)?;
    let mut ctm = Matrix::IDENTITY;
    let mut stack = Vec::new();
    let mut painted = false;
    for op in &content.operations {
        match op.operator.as_ref() {
            "q" if stack.len() < MAX_GRAPHICS_STATE_DEPTH => stack.push(ctm),
            "q" => {
                return Err(format!(
                    "PDF graphics-state stack exceeds {MAX_GRAPHICS_STATE_DEPTH} entries"
                ));
            }
            "Q" => {
                let restored = stack
                    .pop()
                    .ok_or_else(|| "unbalanced PDF graphics-state restore (Q)".to_string())?;
                ctm = restored;
            }
            "cm" => {
                let values = op
                    .operands
                    .iter()
                    .map(|operand| operand.as_float().map(f64::from))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| format!("PDF cm operand is not numeric: {e}"))?;
                if values.len() != 6 {
                    return Err("PDF cm operator must have exactly six operands".to_string());
                }
                ctm = ctm.concat(Matrix {
                    a: values[0],
                    b: values[1],
                    c: values[2],
                    d: values[3],
                    e: values[4],
                    f: values[5],
                });
            }
            "Do" => {
                painted = true;
                break;
            }
            _ => {}
        }
    }
    Ok(if painted { matrix_rotation(ctm) } else { 0 })
}

fn matrix_rotation(matrix: Matrix) -> i64 {
    let Matrix { a, b, c, d, .. } = matrix;
    if b.abs() > a.abs() && c.abs() > d.abs() {
        // Rotated placement: the image x-axis maps to the page y-axis. For
        // b > 0 (the Cadwallader matrix [0 595 -841 0]) the stored raster
        // reads upright after a 90-degree COUNTER-clockwise turn — verified
        // empirically against a reference render (mean-abs-diff 3.6 CCW vs
        // 19.6 CW on the title page), because raster row 0 sits at page
        // LEFT under this matrix. 270 here is the image crate's CCW.
        if b > 0.0 { 270 } else { 90 }
    } else if a < 0.0 && d < 0.0 {
        180
    } else {
        0
    }
}

/// Decode one image XObject to RGB/gray, dispatching on its terminal `/Filter`.
/// Detect and split a two-page book spread into (left, right) halves
/// (bd-av64.11). A spread is a rasterized page that is (a) noticeably wider
/// than tall (w/h >= 1.25 — portrait book pages side by side) AND (b) has a
/// low-ink vertical gutter near the horizontal center. Returns `None` when
/// either condition fails — a false split is worse than none (full-bleed
/// landscape photos, single-column landscape pages, spreads with a plate
/// crossing the gutter all pass through unsplit).
///
/// The gutter search: over the middle 20% of columns, a column qualifies
/// as gutter when its dark fraction is EITHER <= 0.5% (a blank inter-page
/// gap — flat-scanned loose pages) OR >= 60% (the dark binding shadow of a
/// bound book pressed into the scanner — the Cadwallader case). Text
/// columns are mixed black-on-white and match neither. Among qualifying
/// columns the one closest to the exact center wins. Decision + geometry
/// are logged by the caller under FOCR_TIMING.
#[must_use]
pub fn split_spread(img: &DynamicImage) -> Option<(DynamicImage, DynamicImage, u32)> {
    let (w, h) = (img.width(), img.height());
    if h == 0 || (f64::from(w) / f64::from(h)) < 1.25 {
        return None;
    }
    let gray = img.to_luma8();
    let ink_threshold = 160u8; // scanned text is near-black; paper near-white
    let (lo, hi) = (w * 2 / 5, w * 3 / 5); // middle 20% of columns
    let center = i64::from(w / 2);
    let mut best: Option<(i64, u32)> = None; // (distance to center, column)
    for x in lo..hi {
        let mut dark = 0u32;
        for y in 0..h {
            if gray.get_pixel(x, y).0[0] < ink_threshold {
                dark += 1;
            }
        }
        let dark_frac_pct_x10 = u64::from(dark) * 1000 / u64::from(h);
        let is_gutter = dark_frac_pct_x10 <= 5 || dark_frac_pct_x10 >= 600;
        if is_gutter {
            let dist = (i64::from(x) - center).abs();
            if best.is_none_or(|(d, _)| dist < d) {
                best = Some((dist, x));
            }
        }
    }
    let (_, gutter_x) = best?;
    let left = img.crop_imm(0, 0, gutter_x, h);
    let right = img.crop_imm(gutter_x, 0, w - gutter_x, h);
    Some((left, right, gutter_x))
}

fn decode_image_xobject(doc: &Document, img: &PdfImage) -> Result<DynamicImage, String> {
    if img.content.len() > MAX_ENCODED_IMAGE_BYTES {
        return Err(format!(
            "encoded image stream is {} bytes, exceeding the {MAX_ENCODED_IMAGE_BYTES}-byte limit",
            img.content.len()
        ));
    }
    let (width, height) = checked_dimensions(img.width, img.height)?;
    let bpc = img.bits_per_component.unwrap_or(8);
    let color_space = img.color_space.as_deref().unwrap_or("DeviceRGB");
    let filters = img.filters.clone().unwrap_or_default();
    let terminal = filters.last().map(String::as_str).unwrap_or("");

    // The image codecs (DCT/CCITT) consume `img.content` verbatim — the RAW stream,
    // with NO filters applied (our page-resource walk does not decode). So a
    // multi-filter chain whose codec is preceded by an ASCII/Flate filter would
    // feed still-encoded bytes to the codec. Reject such chains with an accurate
    // message rather than a misleading "decode failed". (The raw-sample branch is
    // chain-safe: `decompressed_content` walks the whole filter chain.)
    let chained = filters.len() > 1;

    match terminal {
        "DCTDecode" if chained => Err(format!(
            "image filter chain {filters:?} ending in DCTDecode is unsupported (only a \
             sole DCTDecode filter); chained image-codec decoding must be implemented in \
             franken_ocr::pdf::PdfPages"
        )),
        // `content` is already the raw JPEG byte stream.
        "DCTDecode" => decode_dct(img.content, width, height),

        "JPXDecode" if chained => Err(format!(
            "image filter chain {filters:?} ending in JPXDecode is unsupported (only a sole \
             JPXDecode filter); chained image-codec decoding must be implemented in \
             franken_ocr::pdf::PdfPages"
        )),
        "JPXDecode" => decode_jpx(img.content, width, height),
        "JBIG2Decode" if chained => Err(format!(
            "image filter chain {filters:?} ending in JBIG2Decode is unsupported (only a sole \
             JBIG2Decode filter); chained image-codec decoding must be implemented in \
             franken_ocr::pdf::PdfPages"
        )),
        "JBIG2Decode" => {
            let stream = doc
                .get_object(img.id)
                .and_then(Object::as_stream)
                .map_err(|e| format!("read JBIG2 image stream: {e}"))?;
            let decoded = decode_jbig2_stream(doc, stream)?;
            if decoded.dimensions() != (width, height) {
                return Err(format!(
                    "JBIG2 dimensions {}x{} do not match PDF declaration {width}x{height}",
                    decoded.width(),
                    decoded.height()
                ));
            }
            Ok(DynamicImage::ImageLuma8(decoded))
        }

        "CCITTFaxDecode" if chained => Err(format!(
            "image filter chain {filters:?} ending in CCITTFaxDecode is unsupported (only a \
             sole CCITTFaxDecode filter); chained image-codec decoding must be implemented in \
             franken_ocr::pdf::PdfPages"
        )),
        "CCITTFaxDecode" => decode_ccitt_g4(doc, img, width, height),

        // Raw samples behind a stream-compression filter (or none): inflate and
        // pack into an image buffer per the color space / bit depth.
        // `decompressed_content` handles Flate/LZW/ASCII85; ASCIIHexDecode is NOT
        // among them, so it falls through to the honest "unsupported" arm.
        "FlateDecode" | "LZWDecode" | "ASCII85Decode" | "" => {
            // Bound the inflate at 4x the samples the (already MAX_PIXELS-bounded)
            // declared dimensions could legitimately decode to, so a highly
            // compressed "zip bomb" stream cannot inflate to GBs before any length
            // check. Only a sole FlateDecode is inflated under this cap directly
            // (see `decompressed_stream`); LZW/ASCII85/chains keep lopdf's decoder.
            let cap = expected_sample_cap(width, height, bpc, color_space);
            let sole_flate = !chained && terminal == "FlateDecode";
            let samples = decompressed_stream(doc, img.id, img.content, sole_flate, cap)?;
            raw_samples_to_image(samples, width, height, bpc, color_space)
        }
        other => Err(format!("unsupported image filter {other}")),
    }
}

fn decode_dct(content: &[u8], width: u32, height: u32) -> Result<DynamicImage, String> {
    use image::ImageDecoder as _;
    use std::io::Cursor;

    let decoder = image::codecs::jpeg::JpegDecoder::new(Cursor::new(content))
        .map_err(|e| format!("JPEG (DCTDecode) header parse failed: {e}"))?;
    let (decoded_width, decoded_height) = decoder.dimensions();
    checked_dimensions(i64::from(decoded_width), i64::from(decoded_height))?;
    if (decoded_width, decoded_height) != (width, height) {
        return Err(format!(
            "JPEG dimensions {decoded_width}x{decoded_height} do not match PDF declaration \
             {width}x{height}"
        ));
    }
    DynamicImage::from_decoder(decoder).map_err(|e| format!("JPEG (DCTDecode) decode failed: {e}"))
}

fn checked_dimensions(width: i64, height: i64) -> Result<(u32, u32), String> {
    let width = u32::try_from(width).map_err(|_| "negative image width".to_string())?;
    let height = u32::try_from(height).map_err(|_| "negative image height".to_string())?;
    if width == 0 || height == 0 {
        return Err("zero image dimension".to_string());
    }
    if u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(format!(
            "image dimensions {width}x{height} exceed the {MAX_PIXELS}-pixel maximum"
        ));
    }
    Ok((width, height))
}

fn decode_jpx(content: &[u8], width: u32, height: u32) -> Result<DynamicImage, String> {
    let image = Jp2Image::new(content, &DecodeSettings::default())
        .map_err(|e| format!("JPEG 2000 (JPXDecode) header parse failed: {e}"))?;
    checked_dimensions(i64::from(image.width()), i64::from(image.height()))?;
    if image.width() != width || image.height() != height {
        return Err(format!(
            "JPEG 2000 dimensions {}x{} do not match PDF declaration {width}x{height}",
            image.width(),
            image.height()
        ));
    }
    if image.has_alpha() {
        return Err(
            "JPEG 2000 embedded alpha is not implemented; use a PDF /SMask image layer".to_string(),
        );
    }
    enum Packing {
        Gray,
        Rgb,
        Cmyk,
    }
    let packing = match image.color_space() {
        ColorSpace::Gray => Packing::Gray,
        ColorSpace::RGB => Packing::Rgb,
        ColorSpace::CMYK => Packing::Cmyk,
        ColorSpace::Unknown { .. } | ColorSpace::Icc { .. } => {
            return Err(format!(
                "JPEG 2000 color space {:?} is not implemented in the scalar PDF path",
                image.color_space()
            ));
        }
    };
    let mut context = DecoderContext::default();
    let samples = image
        .decode(&mut context)
        .map_err(|e| format!("JPEG 2000 (JPXDecode) decode failed: {e}"))?
        .data_u8();
    match packing {
        Packing::Gray => from_raw_gray(width, height, samples),
        Packing::Rgb => from_raw_rgb(width, height, samples),
        Packing::Cmyk => Ok(DynamicImage::ImageRgb8(cmyk8_to_rgb(
            &samples, width, height,
        )?)),
    }
}

fn decode_jbig2_stream(doc: &Document, stream: &Stream) -> Result<GrayImage, String> {
    if stream.content.len() > MAX_ENCODED_IMAGE_BYTES {
        return Err(format!(
            "encoded JBIG2 stream is {} bytes, exceeding the {MAX_ENCODED_IMAGE_BYTES}-byte limit",
            stream.content.len()
        ));
    }
    if stream.dict.has(b"Matte") {
        return Err("JBIG2 soft masks with /Matte are not implemented".to_string());
    }
    let width = stream
        .dict
        .get(b"Width")
        .and_then(Object::as_i64)
        .map_err(|e| format!("JBIG2 stream has no valid /Width: {e}"))?;
    let height = stream
        .dict
        .get(b"Height")
        .and_then(Object::as_i64)
        .map_err(|e| format!("JBIG2 stream has no valid /Height: {e}"))?;
    let (width, height) = checked_dimensions(width, height)?;
    let subtype = stream
        .dict
        .get(b"Subtype")
        .and_then(Object::as_name)
        .map_err(|e| format!("JBIG2 stream has no valid /Subtype: {e}"))?;
    if subtype != b"Image" {
        return Err(format!(
            "JBIG2 soft mask has subtype /{}, expected /Image",
            String::from_utf8_lossy(subtype)
        ));
    }
    let color_space = stream
        .dict
        .get(b"ColorSpace")
        .and_then(Object::as_name)
        .map_err(|e| format!("JBIG2 stream has no valid /ColorSpace: {e}"))?;
    if color_space != b"DeviceGray" {
        return Err(format!(
            "JBIG2 soft mask uses /{}; only /DeviceGray is implemented",
            String::from_utf8_lossy(color_space)
        ));
    }
    let bits = stream
        .dict
        .get(b"BitsPerComponent")
        .and_then(Object::as_i64)
        .map_err(|e| format!("JBIG2 stream has no valid /BitsPerComponent: {e}"))?;
    if bits != 1 {
        return Err(format!(
            "JBIG2 soft mask uses {bits} bits per component; expected 1"
        ));
    }
    let filters = stream_filters(&stream.dict)?;
    if filters.as_slice() != ["JBIG2Decode"] {
        return Err(format!(
            "JBIG2 image requires exactly one /JBIG2Decode filter, found {filters:?}"
        ));
    }
    let globals = jbig2_globals(doc, &stream.dict)?;
    let image = Jbig2Image::new_embedded(&stream.content, globals.as_deref())
        .map_err(|e| format!("JBIG2Decode header parse failed: {e}"))?;
    checked_dimensions(i64::from(image.width()), i64::from(image.height()))?;
    if image.width() != width || image.height() != height {
        return Err(format!(
            "JBIG2 dimensions {}x{} do not match PDF declaration {width}x{height}",
            image.width(),
            image.height()
        ));
    }

    struct LumaDecoder {
        samples: Vec<u8>,
        expected: usize,
        overflowed: bool,
    }
    impl Jbig2Decoder for LumaDecoder {
        fn push_pixel(&mut self, black: bool) {
            if self.samples.len() >= self.expected {
                self.overflowed = true;
            } else {
                self.samples.push(if black { 0 } else { 255 });
            }
        }

        fn push_pixel_chunk(&mut self, black: bool, chunk_count: u32) {
            let Some(count) = usize::try_from(chunk_count)
                .ok()
                .and_then(|count| count.checked_mul(8))
            else {
                self.overflowed = true;
                return;
            };
            let Some(end) = self.samples.len().checked_add(count) else {
                self.overflowed = true;
                return;
            };
            if end > self.expected {
                self.overflowed = true;
                return;
            }
            self.samples.resize(end, if black { 0 } else { 255 });
        }

        fn next_line(&mut self) {}
    }

    let expected = usize::try_from(u64::from(width) * u64::from(height))
        .map_err(|_| "JBIG2 sample count does not fit usize".to_string())?;
    let mut decoder = LumaDecoder {
        samples: Vec::with_capacity(expected),
        expected,
        overflowed: false,
    };
    image
        .decode(&mut decoder)
        .map_err(|e| format!("JBIG2Decode failed: {e}"))?;
    if decoder.overflowed || decoder.samples.len() != expected {
        return Err(format!(
            "JBIG2 decoder emitted {} samples for {width}x{height}",
            decoder.samples.len()
        ));
    }
    if soft_mask_decode_is_inverted(&stream.dict)? {
        for sample in &mut decoder.samples {
            *sample = 255 - *sample;
        }
    }
    GrayImage::from_raw(width, height, decoder.samples)
        .ok_or_else(|| "JBIG2 sample count does not match image dimensions".to_string())
}

fn stream_filters(dict: &Dictionary) -> Result<Vec<String>, String> {
    let Ok(filter) = dict.get(b"Filter") else {
        return Ok(Vec::new());
    };
    match filter {
        Object::Name(name) => Ok(vec![String::from_utf8_lossy(name).into_owned()]),
        Object::Array(filters) => filters
            .iter()
            .map(|filter| {
                filter
                    .as_name()
                    .map(|name| String::from_utf8_lossy(name).into_owned())
                    .map_err(|e| format!("image /Filter array contains a non-name: {e}"))
            })
            .collect(),
        _ => Err("image /Filter must be a name or array of names".to_string()),
    }
}

fn jbig2_globals(doc: &Document, dict: &Dictionary) -> Result<Option<Vec<u8>>, String> {
    let Some(parameters) = decode_parameters(doc, dict)? else {
        return Ok(None);
    };
    let Ok(globals) = parameters.get(b"JBIG2Globals") else {
        return Ok(None);
    };
    let (_, globals) = doc
        .dereference(globals)
        .map_err(|e| format!("dereference /JBIG2Globals: {e}"))?;
    let stream = globals
        .as_stream()
        .map_err(|e| format!("/JBIG2Globals is not a stream: {e}"))?;
    let filters = stream_filters(&stream.dict)?;
    let content = match filters.as_slice() {
        [] => {
            if stream.content.len() as u64 > MAX_JBIG2_GLOBAL_BYTES {
                return Err(format!(
                    "/JBIG2Globals exceeds the {MAX_JBIG2_GLOBAL_BYTES}-byte limit"
                ));
            }
            stream.content.clone()
        }
        [filter] if filter == "FlateDecode" && stream_predictor(stream) <= 1 => {
            bounded_inflate(&stream.content, MAX_JBIG2_GLOBAL_BYTES)?.ok_or_else(|| {
                "sole-Flate /JBIG2Globals is not a valid bounded zlib stream".to_string()
            })?
        }
        _ => {
            return Err(format!(
                "filtered /JBIG2Globals {filters:?} is unsupported; only an unfiltered stream or \
                 bounded sole FlateDecode without a predictor is implemented"
            ));
        }
    };
    if content.len() as u64 > MAX_JBIG2_GLOBAL_BYTES {
        return Err(format!(
            "decoded /JBIG2Globals exceeds the {MAX_JBIG2_GLOBAL_BYTES}-byte limit"
        ));
    }
    Ok(Some(content))
}

fn decode_parameters<'a>(
    doc: &'a Document,
    dict: &'a Dictionary,
) -> Result<Option<&'a Dictionary>, String> {
    let Ok(mut parameters) = dict.get(b"DecodeParms").or_else(|_| dict.get(b"DP")) else {
        return Ok(None);
    };
    if let Object::Array(items) = parameters {
        if items.len() != 1 {
            return Err(
                "a sole JBIG2Decode filter requires exactly one /DecodeParms entry".to_string(),
            );
        }
        parameters = &items[0];
    }
    let (_, parameters) = doc
        .dereference(parameters)
        .map_err(|e| format!("dereference JBIG2 /DecodeParms: {e}"))?;
    match parameters {
        Object::Null => Ok(None),
        Object::Dictionary(parameters) => Ok(Some(parameters)),
        _ => Err("JBIG2 /DecodeParms must be a dictionary or null".to_string()),
    }
}

fn soft_mask_decode_is_inverted(dict: &Dictionary) -> Result<bool, String> {
    let Ok(decode) = dict.get(b"Decode") else {
        return Ok(false);
    };
    let values = decode
        .as_array()
        .map_err(|e| format!("soft-mask /Decode is not an array: {e}"))?;
    if values.len() != 2 {
        return Err("soft-mask /Decode must contain exactly two numbers".to_string());
    }
    let first = f64::from(
        values[0]
            .as_float()
            .map_err(|e| format!("soft-mask /Decode value is not numeric: {e}"))?,
    );
    let second = f64::from(
        values[1]
            .as_float()
            .map_err(|e| format!("soft-mask /Decode value is not numeric: {e}"))?,
    );
    if (first, second) == (0.0, 1.0) {
        Ok(false)
    } else if (first, second) == (1.0, 0.0) {
        Ok(true)
    } else {
        Err(format!(
            "soft-mask /Decode [{first} {second}] is unsupported; expected [0 1] or [1 0]"
        ))
    }
}

/// Re-fetch the image XObject as a stream and return its decompressed bytes,
/// bounding the inflate so a decompression bomb cannot OOM the process.
///
/// `raw` is `PdfImage::content`, the *raw* stream slice (still deflate/LZW/ASCII
/// encoded for those filters). lopdf's `Stream::decompressed_content` un-applies
/// the whole filter chain (including PNG/TIFF predictors) but materializes the
/// FULL inflated output before any length check — so a tiny, highly-compressed
/// FlateDecode stream (a "zip bomb", ~1000:1) inflates to GBs regardless of the
/// declared dimensions. For the common case — a SOLE FlateDecode with no predictor
/// — we inflate `raw` ourselves under `cap` and reject an overrun, allocating at
/// most `cap + 1` bytes. Everything else (LZW, ASCII85, filter chains, or a
/// `/Predictor > 1` that needs un-applying) falls back to `decompressed_content`;
/// those paths keep lopdf's residual unbounded-inflate risk.
fn decompressed_stream(
    doc: &Document,
    id: ObjectId,
    raw: &[u8],
    sole_flate: bool,
    cap: u64,
) -> Result<Vec<u8>, String> {
    let stream = doc
        .get_object(id)
        .and_then(Object::as_stream)
        .map_err(|e| format!("read image stream: {e}"))?;
    // Bounded fast path only when nothing downstream of the inflate is needed: a
    // single FlateDecode with no PNG/TIFF predictor. A predictor (>1) or any chain
    // would need lopdf's post-processing, so those keep the unbounded decoder. The
    // `?` still propagates a cap overrun (the bomb signal) before the `let Some`.
    if sole_flate
        && stream_predictor(stream) <= 1
        && let Some(out) = bounded_inflate(raw, cap)?
    {
        return Ok(out);
    }
    // The bounded path did not apply, or `raw` was not decodable as standalone zlib
    // (e.g. a headerless raw-deflate stream some producers emit); fall back to
    // lopdf's framing-tolerant decoder.
    stream
        .decompressed_content()
        .map_err(|e| format!("inflate image stream: {e}"))
}

/// The `/Predictor` in a stream's `/DecodeParms` (or its `/DP` abbreviation), or
/// `1` (no predictor) when absent. A sole-filter stream carries `DecodeParms` as a
/// single dict; the array form (filter chains) is never routed to the bounded path.
fn stream_predictor(stream: &lopdf::Stream) -> i64 {
    stream
        .dict
        .get(b"DecodeParms")
        .or_else(|_| stream.dict.get(b"DP"))
        .and_then(Object::as_dict)
        .ok()
        .and_then(|p| p.get(b"Predictor").ok())
        .and_then(|o| o.as_i64().ok())
        .unwrap_or(1)
}

/// Inflate a sole-FlateDecode (zlib) stream, refusing to allocate past `cap`.
///
/// PDF `/FlateDecode` is the zlib data format (RFC 1950), so `ZlibDecoder` is the
/// right reader. Reading just one byte past `cap` distinguishes "fits" from
/// "overruns" without buffering the whole bomb. Returns `Ok(None)` — *not* an error
/// — when `raw` is not valid standalone zlib, so the caller can fall back to lopdf's
/// framing-tolerant decoder; a clean inflate that overruns `cap` is the bomb signal
/// and is the only `Err`.
fn bounded_inflate(raw: &[u8], cap: u64) -> Result<Option<Vec<u8>>, String> {
    use std::io::Read;
    let mut out = Vec::new();
    if flate2::read::ZlibDecoder::new(raw)
        .take(cap.saturating_add(1))
        .read_to_end(&mut out)
        .is_err()
    {
        return Ok(None);
    }
    if out.len() as u64 > cap {
        return Err(format!(
            "decompressed image stream exceeds the {cap}-byte cap \
             (4x the expected sample size; possible decompression bomb)"
        ));
    }
    Ok(Some(out))
}

/// A generous cap on the inflated sample buffer: `4 × width × height × components ×
/// ceil(bpc/8)`.
///
/// The declared dimensions are already `MAX_PIXELS`-bounded, so this bounds a
/// decompression bomb to a small multiple of the bytes those dimensions could
/// legitimately decode to, instead of the GBs an adversarial stream would inflate
/// to. Unknown color spaces get the 4-component (CMYK) upper bound;
/// `raw_samples_to_image` rejects them afterward. `saturating_mul` keeps the
/// arithmetic from overflowing on hostile inputs.
///
/// `bpc` is clamped to the PDF-legal image range `1..=16` BEFORE it scales the cap:
/// a crafted `/BitsPerComponent` (e.g. `i64::MAX`) would otherwise blow the cap up
/// to `u64::MAX`, and a `cap` of `u64::MAX` makes the `take(cap + 1)` bound in
/// [`bounded_inflate`] effectively unbounded — re-opening the very bomb hole this
/// guards. The real bit-depth is validated separately in `raw_samples_to_image`.
fn expected_sample_cap(width: u32, height: u32, bpc: i64, color_space: &str) -> u64 {
    let comps: u64 = match color_space {
        "DeviceGray" | "CalGray" => 1,
        "DeviceRGB" | "CalRGB" => 3,
        _ => 4, // DeviceCMYK and any unknown: the largest plausible component count
    };
    // PDF images are 1/2/4/8/16 bpc; clamp so a hostile bit-depth cannot inflate
    // the cap past the bytes a real 16-bit image of these dimensions would need.
    let bytes_per_comp = (bpc.clamp(1, 16) as u64).div_ceil(8);
    u64::from(width)
        .saturating_mul(u64::from(height))
        .saturating_mul(comps)
        .saturating_mul(bytes_per_comp)
        .saturating_mul(4)
}

/// Build a [`DynamicImage`] from raw component samples.
fn raw_samples_to_image(
    samples: Vec<u8>,
    width: u32,
    height: u32,
    bpc: i64,
    color_space: &str,
) -> Result<DynamicImage, String> {
    let comps = match color_space {
        "DeviceRGB" | "CalRGB" => 3usize,
        "DeviceGray" | "CalGray" => 1,
        "DeviceCMYK" => 4,
        // ICCBased streams carry an /N component count; without resolving the
        // profile we cannot know it here, and Indexed/Separation need a palette.
        // Punt with a clear message rather than render garbage.
        other => return Err(format!("unsupported color space {other}")),
    };

    match bpc {
        8 => match comps {
            3 => from_raw_rgb(width, height, samples),
            1 => from_raw_gray(width, height, samples),
            4 => Ok(DynamicImage::ImageRgb8(cmyk8_to_rgb(
                &samples, width, height,
            )?)),
            _ => Err(format!("unsupported component count {comps}")),
        },
        1 => bilevel_to_gray(&samples, width, height),
        16 => {
            // Samples are big-endian; downscale to 8-bpc by keeping the high byte.
            let high: Vec<u8> = samples.as_chunks::<2>().0.iter().map(|c| c[0]).collect();
            raw_samples_to_image(high, width, height, 8, color_space)
        }
        other => Err(format!("unsupported bits-per-component {other}")),
    }
}

fn from_raw_rgb(width: u32, height: u32, samples: Vec<u8>) -> Result<DynamicImage, String> {
    let buf: RgbImage = ImageBuffer::from_raw(width, height, samples)
        .ok_or_else(|| "RGB sample count does not match image dimensions".to_string())?;
    Ok(DynamicImage::ImageRgb8(buf))
}

fn from_raw_gray(width: u32, height: u32, samples: Vec<u8>) -> Result<DynamicImage, String> {
    let buf: GrayImage = ImageBuffer::from_raw(width, height, samples)
        .ok_or_else(|| "gray sample count does not match image dimensions".to_string())?;
    Ok(DynamicImage::ImageLuma8(buf))
}

/// Expand 8-bpc CMYK to RGB (the naive `r = 255 - min(255, c + k)` conversion;
/// adequate for OCR, which only needs legible contrast, not color fidelity).
fn cmyk8_to_rgb(samples: &[u8], width: u32, height: u32) -> Result<RgbImage, String> {
    let pixels = (width as usize) * (height as usize);
    if samples.len() < pixels * 4 {
        return Err("CMYK sample count does not match image dimensions".to_string());
    }
    let mut out = Vec::with_capacity(pixels * 3);
    for px in samples.as_chunks::<4>().0.iter().take(pixels) {
        let (c, m, y, k) = (
            u16::from(px[0]),
            u16::from(px[1]),
            u16::from(px[2]),
            u16::from(px[3]),
        );
        out.push((255 - (c + k).min(255)) as u8);
        out.push((255 - (m + k).min(255)) as u8);
        out.push((255 - (y + k).min(255)) as u8);
    }
    ImageBuffer::from_raw(width, height, out).ok_or_else(|| "CMYK->RGB pack failed".to_string())
}

/// Unpack MSB-first, byte-padded 1-bpc bilevel samples to an 8-bpc gray image.
fn bilevel_to_gray(samples: &[u8], width: u32, height: u32) -> Result<DynamicImage, String> {
    let row_bytes = (width as usize).div_ceil(8);
    if samples.len() < row_bytes * height as usize {
        return Err("bilevel sample count does not match image dimensions".to_string());
    }
    let mut out = Vec::with_capacity((width as usize) * (height as usize));
    for y in 0..height as usize {
        let row = &samples[y * row_bytes..];
        for x in 0..width as usize {
            let bit = (row[x / 8] >> (7 - (x % 8))) & 1;
            out.push(if bit == 1 { 255 } else { 0 });
        }
    }
    from_raw_gray(width, height, out)
}

/// Decode a CCITT Group 4 (T.6) fax image XObject to an 8-bpc gray image.
///
/// Group 4 is `/K < 0`; G3 (`/K >= 0`) is reported unsupported. `/BlackIs1`
/// flips the 0=black / 255=white convention.
fn decode_ccitt_g4(
    doc: &Document,
    img: &PdfImage,
    width: u32,
    height: u32,
) -> Result<DynamicImage, String> {
    use fax::Color;
    use fax::decoder::{decode_g4, pels};

    let stream = doc
        .get_object(img.id)
        .and_then(Object::as_stream)
        .map_err(|e| format!("read CCITT stream: {e}"))?;

    // /DecodeParms (or the /DP abbreviation) may be a dict or an array of dicts;
    // the single-dict form is what scanners emit for a lone CCITT filter.
    let parms = stream
        .dict
        .get(b"DecodeParms")
        .or_else(|_| stream.dict.get(b"DP"))
        .and_then(Object::as_dict)
        .ok();
    let param_i64 = |key: &[u8], default: i64| -> i64 {
        parms
            .and_then(|p| p.get(key).ok())
            .and_then(|o| o.as_i64().ok())
            .unwrap_or(default)
    };
    let k = param_i64(b"K", 0);
    let columns = u16::try_from(param_i64(b"Columns", 1728)).unwrap_or(1728);
    let black_is_1 = parms
        .and_then(|p| p.get(b"BlackIs1").ok())
        .and_then(|o| o.as_bool().ok())
        .unwrap_or(false);

    if k >= 0 {
        return Err(
            "CCITTFaxDecode K>=0 (Group 3) is not supported; only Group 4 (K<0)".to_string(),
        );
    }
    let cols = if columns == 0 {
        u16::try_from(width).unwrap_or(1728)
    } else {
        columns
    };
    let (black, white) = if black_is_1 {
        (255u8, 0u8)
    } else {
        (0u8, 255u8)
    };
    let rows_hint = u16::try_from(height).ok().filter(|&h| h != 0);

    // Grow as the decode emits lines; do NOT pre-reserve from the declared
    // `/Height`, which is an attacker-controlled `u32` (`cols * height` could
    // reserve hundreds of TB and abort). The real output is bounded by the actual
    // G4 stream — `decode_g4` stops at end-of-data or `rows_hint` rows.
    let mut out: Vec<u8> = Vec::new();
    decode_g4(img.content.iter().copied(), cols, rows_hint, |line| {
        out.extend(pels(line, cols).map(|c| match c {
            Color::Black => black,
            Color::White => white,
        }));
    })
    .ok_or_else(|| "CCITT Group 4 decode failed".to_string())?;

    let decoded_rows = u32::try_from(out.len() / usize::from(cols).max(1)).unwrap_or(0);
    from_raw_gray(u32::from(cols), decoded_rows, out)
}

/// The page's `/Rotate` (an inheritable multiple of 90, clockwise), normalized
/// to `0 | 90 | 180 | 270`.
fn page_rotation(doc: &Document, page_id: ObjectId) -> i64 {
    inherited(doc, page_id, b"Rotate")
        .and_then(|o| o.as_i64().ok())
        .unwrap_or(0)
        .rem_euclid(360)
}

/// Apply a clockwise rotation (0/90/180/270) to the rendered page.
fn apply_rotation(img: DynamicImage, degrees: i64) -> DynamicImage {
    match degrees {
        90 => DynamicImage::ImageRgba8(image::imageops::rotate90(&img)),
        180 => DynamicImage::ImageRgba8(image::imageops::rotate180(&img)),
        270 => DynamicImage::ImageRgba8(image::imageops::rotate270(&img)),
        _ => img,
    }
}

/// Resolve an inheritable page attribute, walking `/Parent` (bounded against a
/// cyclic page tree).
fn inherited<'a>(doc: &'a Document, mut id: ObjectId, key: &[u8]) -> Option<&'a Object> {
    for _ in 0..64 {
        let dict = doc.get_dictionary(id).ok()?;
        if let Ok(value) = dict.get(key) {
            return Some(value);
        }
        id = dict.get(b"Parent").and_then(Object::as_reference).ok()?;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_page(w: u32, h: u32, text_cols: &[(u32, u32)]) -> DynamicImage {
        // White canvas with black "text block" columns [x0, x1).
        let mut img = image::GrayImage::from_pixel(w, h, image::Luma([255u8]));
        for &(x0, x1) in text_cols {
            for x in x0..x1 {
                for y in (10..h.saturating_sub(10)).step_by(3) {
                    img.put_pixel(x, y, image::Luma([20u8]));
                }
            }
        }
        DynamicImage::ImageLuma8(img)
    }

    /// bd-av64.11: a synthetic spread (text left + right, blank center
    /// gutter) splits at the gutter; the negatives pass through unsplit.
    #[test]
    fn split_spread_positive_and_negatives() {
        // POSITIVE: 1600x1000 spread, text at [100,700) and [900,1500).
        let spread = synth_page(1600, 1000, &[(100, 700), (900, 1500)]);
        let (left, right, gx) = split_spread(&spread).expect("spread splits");
        assert!((700..=900).contains(&gx), "gutter near center: {gx}");
        assert_eq!(left.width() + right.width(), 1600);
        assert_eq!(left.height(), 1000);
        // NEGATIVE 1: portrait page (aspect below the spread threshold).
        assert!(split_spread(&synth_page(1000, 1600, &[(100, 900)])).is_none());
        // NEGATIVE 2: landscape but ink crosses the center (a full-width
        // plate/table) — no blank gutter, no split.
        assert!(split_spread(&synth_page(1600, 1000, &[(100, 1500)])).is_none());
        // NEGATIVE 3: landscape blank page — a gutter exists but splitting a
        // blank is harmless; the heuristic DOES split it (blank center
        // qualifies). Accepting this is deliberate: both halves are blank,
        // and OCR of blank halves is cheap + correct.
        // NEGATIVE 4: single centered column (text crosses the middle).
        assert!(split_spread(&synth_page(1600, 1000, &[(600, 1000)])).is_none());
        // POSITIVE 2: a bound book's DARK binding shadow as the gutter (the
        // Cadwallader case) — a solid dark band at the center qualifies.
        let mut bound = synth_page(1600, 1000, &[(100, 700), (900, 1500)]).to_luma8();
        for x in 780..820 {
            for y in 0..1000 {
                bound.put_pixel(x, y, image::Luma([30u8]));
            }
        }
        let bound = DynamicImage::ImageLuma8(bound);
        let (_, _, gx) = split_spread(&bound).expect("binding shadow splits");
        assert!((780..=820).contains(&gx), "split inside the shadow: {gx}");
    }

    #[test]
    fn looks_like_pdf_by_extension() {
        assert!(looks_like_pdf(Path::new("/x/y/scan.pdf")));
        assert!(looks_like_pdf(Path::new("/x/y/scan.PDF")));
        assert!(!looks_like_pdf(Path::new("/x/y/page.png")));
        // Missing file, no .pdf extension -> not a PDF (no panic).
        assert!(!looks_like_pdf(Path::new("/no/such/file.bin")));
    }

    #[test]
    fn bounded_pdf_read_accepts_the_limit_and_rejects_one_byte_more() {
        let accepted = read_pdf_input(std::io::Cursor::new(b"12345"), "fixture.pdf", 5)
            .expect("input at the limit");
        assert_eq!(accepted, b"12345");

        let error = read_pdf_input(std::io::Cursor::new(b"123456"), "fixture.pdf", 5)
            .expect_err("input over the limit");
        assert_eq!(
            error.to_string(),
            "input decode error: PDF fixture.pdf exceeds native input limit of 5 bytes"
        );
    }

    #[test]
    fn bounded_pdf_read_reports_read_failure_separately() {
        struct FailingReader;

        impl std::io::Read for FailingReader {
            fn read(&mut self, _buffer: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("synthetic read failure"))
            }
        }

        let error = read_pdf_input(FailingReader, "fixture.pdf", 5)
            .expect_err("reader must propagate its failure");
        assert_eq!(
            error.to_string(),
            "input decode error: read PDF fixture.pdf: synthetic read failure"
        );
    }

    #[test]
    fn malformed_pdf_reports_parse_failure_separately() {
        let error = parse_pdf_input(b"not a PDF", "fixture.pdf").expect_err("invalid PDF");
        assert_eq!(
            error.to_string(),
            "input decode error: parse PDF fixture.pdf: couldn't parse input: invalid file header"
        );
    }

    #[test]
    fn bilevel_unpacks_msb_first() {
        // 8x1: 0b1010_0000 -> px0=255, px1=0, px2=255, rest 0.
        let img = bilevel_to_gray(&[0b1010_0000], 8, 1).expect("bilevel");
        let gray = img.to_luma8();
        assert_eq!(gray.get_pixel(0, 0).0[0], 255);
        assert_eq!(gray.get_pixel(1, 0).0[0], 0);
        assert_eq!(gray.get_pixel(2, 0).0[0], 255);
        assert_eq!(gray.get_pixel(3, 0).0[0], 0);
    }

    #[test]
    fn cmyk_pure_black_and_white() {
        // pixel0 = pure K (black), pixel1 = all-zero (white).
        let rgb = cmyk8_to_rgb(&[0, 0, 0, 255, 0, 0, 0, 0], 2, 1).expect("cmyk");
        assert_eq!(rgb.get_pixel(0, 0).0, [0, 0, 0]);
        assert_eq!(rgb.get_pixel(1, 0).0, [255, 255, 255]);
    }

    #[test]
    fn alpha_composite_obeys_zero_half_and_full_opacity() {
        let mut background = RgbImage::from_pixel(3, 1, image::Rgb([10, 20, 30]));
        let foreground = RgbImage::from_pixel(3, 1, image::Rgb([110, 120, 130]));
        let alpha = GrayImage::from_raw(3, 1, vec![0, 128, 255]).expect("mask");
        alpha_composite(&mut background, &foreground, &alpha).expect("composite");
        assert_eq!(background.get_pixel(0, 0).0, [10, 20, 30]);
        assert_eq!(background.get_pixel(1, 0).0, [60, 70, 80]);
        assert_eq!(background.get_pixel(2, 0).0, [110, 120, 130]);
    }

    #[test]
    fn soft_mask_decode_inversion_is_explicit() {
        use lopdf::dictionary;

        assert!(!soft_mask_decode_is_inverted(&dictionary! {}).expect("default"));
        assert!(
            soft_mask_decode_is_inverted(&dictionary! {
                "Decode" => vec![1_i64.into(), 0_i64.into()],
            })
            .expect("inverse")
        );
        assert!(
            !soft_mask_decode_is_inverted(&dictionary! {
                "Decode" => vec![0_i64.into(), 1_i64.into()],
            })
            .expect("normal")
        );
        assert!(
            soft_mask_decode_is_inverted(&dictionary! {
                "Decode" => vec![0_i64.into(), 2_i64.into()],
            })
            .is_err()
        );
    }

    #[test]
    fn full_page_matrix_accepts_axis_alignment_and_refuses_partial_or_skewed() {
        let bounds = [0.0, 0.0, 647.0, 942.0];
        validate_full_page_axis_aligned(
            Matrix {
                a: 647.0,
                b: 0.0,
                c: 0.0,
                d: 942.0,
                e: 0.0,
                f: 0.0,
            },
            bounds,
        )
        .expect("canonical full-page CTM");
        assert!(
            validate_full_page_axis_aligned(
                Matrix {
                    a: 600.0,
                    b: 0.0,
                    c: 0.0,
                    d: 900.0,
                    e: 0.0,
                    f: 0.0,
                },
                bounds,
            )
            .unwrap_err()
            .contains("partial-page")
        );
        assert!(
            validate_full_page_axis_aligned(
                Matrix {
                    a: 647.0,
                    b: 1.0,
                    c: 0.0,
                    d: 942.0,
                    e: 0.0,
                    f: 0.0,
                },
                bounds,
            )
            .unwrap_err()
            .contains("skewed")
        );
        assert!(
            validate_full_page_axis_aligned(
                Matrix {
                    a: -647.0,
                    b: 0.0,
                    c: 0.0,
                    d: 942.0,
                    e: 647.0,
                    f: 0.0,
                },
                bounds,
            )
            .unwrap_err()
            .contains("reflected")
        );
    }

    #[test]
    fn mrc_content_allows_only_invisible_preimage_text() {
        use lopdf::content::{Content, Operation};

        let invisible = Content {
            operations: vec![
                Operation::new("Tr", vec![3.into()]),
                Operation::new("Tj", vec![Object::string_literal("hidden OCR")]),
                Operation::new("Do", vec![Object::Name(b"Base".to_vec())]),
            ],
        };
        validate_mrc_content(&invisible, 2, 2).expect("Tr 3 text is nonpainting");

        let visible = Content {
            operations: vec![
                Operation::new("Tj", vec![Object::string_literal("visible")]),
                Operation::new("Do", vec![Object::Name(b"Base".to_vec())]),
            ],
        };
        assert!(
            validate_mrc_content(&visible, 1, 1)
                .unwrap_err()
                .contains("visible pre-image text")
        );

        for forbidden in ["W", "sh", "BI", "gs"] {
            let content = Content {
                operations: vec![
                    Operation::new(forbidden, vec![]),
                    Operation::new("Do", vec![Object::Name(b"Base".to_vec())]),
                ],
            };
            assert!(
                validate_mrc_content(&content, 1, 1).is_err(),
                "{forbidden} must be refused"
            );
        }
    }

    #[test]
    fn mrc_budget_rejects_soft_mask_dimension_mismatch_before_decode() {
        use lopdf::{Stream, dictionary};

        let mut doc = Document::with_version("1.5");
        let mask_id = doc.add_object(Stream::new(
            dictionary! {
                "Subtype" => "Image", "Width" => 9, "Height" => 10,
                "ColorSpace" => "DeviceGray", "BitsPerComponent" => 1,
                "Filter" => "JBIG2Decode",
            },
            vec![],
        ));
        let base_dict = dictionary! {
            "Subtype" => "Image", "Width" => 10, "Height" => 10,
        };
        let overlay_dict = dictionary! {
            "Subtype" => "Image", "Width" => 10, "Height" => 10,
            "SMask" => mask_id,
        };
        let base_content = vec![];
        let overlay_content = vec![];
        let base = PdfImage {
            id: (1, 0),
            width: 10,
            height: 10,
            color_space: Some("DeviceRGB".to_string()),
            filters: Some(vec!["JPXDecode".to_string()]),
            bits_per_component: Some(8),
            content: &base_content,
            origin_dict: &base_dict,
        };
        let overlay = PdfImage {
            id: (2, 0),
            width: 10,
            height: 10,
            color_space: Some("DeviceRGB".to_string()),
            filters: Some(vec!["JPXDecode".to_string()]),
            bits_per_component: Some(8),
            content: &overlay_content,
            origin_dict: &overlay_dict,
        };
        assert!(
            validate_mrc_budget(&doc, &[&base, &overlay], 10, 10)
                .unwrap_err()
                .contains("does not match")
        );
    }

    #[test]
    fn do_order_uses_resource_names_and_cumulative_ctm() {
        use lopdf::{Stream, dictionary};

        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let first_id = doc.add_object(Stream::new(
            dictionary! { "Type" => "XObject", "Subtype" => "Image", "Width" => 1, "Height" => 1 },
            vec![0],
        ));
        let second_id = doc.add_object(Stream::new(
            dictionary! { "Type" => "XObject", "Subtype" => "Image", "Width" => 1, "Height" => 1 },
            vec![0],
        ));
        // Resource order is First, Second; content paints Second, First.
        let resources_id = doc.add_object(dictionary! {
            "XObject" => dictionary! { "First" => first_id, "Second" => second_id },
        });
        let content = lopdf::content::Content {
            operations: vec![
                lopdf::content::Operation::new("q", vec![]),
                lopdf::content::Operation::new(
                    "cm",
                    vec![2.into(), 0.into(), 0.into(), 3.into(), 4.into(), 5.into()],
                ),
                lopdf::content::Operation::new(
                    "cm",
                    vec![10.into(), 0.into(), 0.into(), 20.into(), 1.into(), 2.into()],
                ),
                lopdf::content::Operation::new("Do", vec![Object::Name(b"Second".to_vec())]),
                lopdf::content::Operation::new("Do", vec![Object::Name(b"First".to_vec())]),
                lopdf::content::Operation::new("Q", vec![]),
            ],
        }
        .encode()
        .expect("encode content");
        let content_id = doc.add_object(Stream::new(Dictionary::new(), content));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Resources" => resources_id,
            "Contents" => content_id,
            "MediaBox" => vec![0.into(), 0.into(), 20.into(), 60.into()],
        });
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages", "Kids" => vec![page_id.into()], "Count" => 1,
            }),
        );

        let (paints, _) = ordered_image_paints(&doc, page_id).expect("parse paints");
        assert_eq!(paints.len(), 2);
        assert_eq!(paints[0].id, second_id);
        assert_eq!(paints[1].id, first_id);
        for paint in paints {
            assert!(paint.matrix.approximately_equals(Matrix {
                a: 20.0,
                b: 0.0,
                c: 0.0,
                d: 60.0,
                e: 6.0,
                f: 11.0,
            }));
        }
    }

    /// Live archive proof for the exact layered MRC shape that motivated this
    /// path. The PDF is intentionally not vendored; set `FOCR_TEST_SPOHR_PDF`
    /// to the Internet Archive scan and run this ignored test explicitly.
    #[test]
    #[ignore = "requires the Louis Spohr Internet Archive PDF"]
    fn real_spohr_layered_mrc_page_matches_reference_deterministically() {
        use sha2::{Digest, Sha256};

        let pdf = std::env::var_os("FOCR_TEST_SPOHR_PDF")
            .expect("FOCR_TEST_SPOHR_PDF must name the archive PDF");
        let source = std::fs::read(&pdf).expect("read archive PDF");
        let source_sha256 = format!("{:x}", Sha256::digest(&source));
        assert_eq!(
            source_sha256, "9b6b4a84400932cf5ce93bbcdc87a7041809d35ed7fecdbea9a6ebe3c8e21dac",
            "FOCR_TEST_SPOHR_PDF is not the canonical archive object"
        );
        let pages = PdfPages::open(Path::new(&pdf)).expect("open archive PDF");
        let first = pages.render(54).expect("render page 55");
        let second = pages.render(54).expect("render page 55 again");
        assert_eq!((first.width(), first.height()), (2696, 3926));
        let repeat_stable = first.as_bytes() == second.as_bytes();
        assert!(repeat_stable, "render must be stable");
        let pixel_sha256 = format!("{:x}", Sha256::digest(first.as_bytes()));
        assert_eq!(
            pixel_sha256, "c5e4237228508992d952d600e9f4c48b9784e798c6b296c21144136662e1e56a",
            "composited provider pixels changed"
        );

        let fixture = image::open("tests/fixtures/realscan_music/pages/spohr_p055.png")
            .expect("open committed reference")
            .to_luma8();
        let actual = first
            .resize_exact(
                fixture.width(),
                fixture.height(),
                image::imageops::FilterType::Triangle,
            )
            .to_luma8();
        let total_difference: u64 = actual
            .pixels()
            .zip(fixture.pixels())
            .map(|(actual, expected)| u64::from(actual.0[0].abs_diff(expected.0[0])))
            .sum();
        let mean_absolute_difference = total_difference as f64
            / (u64::from(fixture.width()) * u64::from(fixture.height())) as f64;
        eprintln!(
            "spohr_mrc_proof source_sha256={source_sha256} page_count={} page_number=55 \
             page_index=54 dimensions={}x{} pixel_sha256={pixel_sha256} \
             repeat_stable={repeat_stable} reference_mad={mean_absolute_difference:.6}",
            pages.len(),
            first.width(),
            first.height(),
        );
        assert!(
            mean_absolute_difference < 3.0,
            "provider render drifted from reference: MAD={mean_absolute_difference}"
        );
    }

    #[test]
    fn rgb_dimension_mismatch_errors() {
        // 3 bytes is not enough for a 2x2 RGB image (needs 12).
        assert!(from_raw_rgb(2, 2, vec![1, 2, 3]).is_err());
    }

    /// A minimal one-page PDF whose only object is a single image XObject — the
    /// shared scaffold for the round-trip tests. `image_xobject` is `None` for a
    /// page with no image (the vector/text case).
    fn build_single_page_pdf(image_xobject: Option<lopdf::Stream>) -> std::path::PathBuf {
        use lopdf::{Object, dictionary};

        let mut doc = lopdf::Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let resources = match image_xobject {
            Some(stream) => {
                let image_id = doc.add_object(stream);
                dictionary! { "XObject" => dictionary! { "Im0" => image_id } }
            }
            None => dictionary! {},
        };
        let resources_id = doc.add_object(resources);
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Resources" => resources_id,
            "MediaBox" => vec![0_i64.into(), 0_i64.into(), 100_i64.into(), 100_i64.into()],
        });
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => vec![page_id.into()],
                "Count" => 1,
            }),
        );
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        doc.trailer.set("Root", catalog_id);

        // Unique temp path per call (tests run on parallel threads): pid + a
        // process-wide atomic sequence, not a stack-address pointer.
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let path = std::env::temp_dir().join(format!(
            "focr_pdf_test_{}_{}.pdf",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        doc.save(&path).expect("save synthesized pdf");
        path
    }

    /// End-to-end through the real `lopdf` parser: synthesize a one-page PDF whose
    /// only XObject is a `DCTDecode` (JPEG) image, reopen it via [`PdfPages`], and
    /// confirm the page renders to an image of the JPEG's dimensions. Exercises
    /// effective-resource walk + the `DCTDecode` dispatch + the JPEG decoder — the
    /// dominant real scanned-PDF path.
    #[test]
    fn render_dctdecode_pdf_page_decodes_jpeg_xobject() {
        use image::{ImageBuffer, Rgb};
        use lopdf::{Stream, dictionary};
        use std::io::Cursor;

        let (w, h) = (16u32, 12u32);
        let src = DynamicImage::ImageRgb8(ImageBuffer::from_fn(w, h, |x, _| {
            Rgb([(x * 16) as u8, 64, 128])
        }));
        let mut jpeg = Vec::new();
        src.write_to(&mut Cursor::new(&mut jpeg), image::ImageFormat::Jpeg)
            .expect("encode jpeg");

        let image = Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Image",
                "Width" => i64::from(w),
                "Height" => i64::from(h),
                "ColorSpace" => "DeviceRGB",
                "BitsPerComponent" => 8,
                "Filter" => "DCTDecode",
            },
            jpeg,
        )
        .with_compression(false);
        let path = build_single_page_pdf(Some(image));

        let pages = PdfPages::open(&path).expect("open synthesized pdf");
        assert_eq!(pages.len(), 1);
        let page = pages.render(0).expect("render dct page");
        // The JPEG decoder reports the encoded dimensions back unchanged.
        assert_eq!((page.width(), page.height()), (w, h));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn owned_pdf_bytes_render_after_the_source_path_is_gone() {
        use image::{ImageBuffer, Rgb};
        use lopdf::{Stream, dictionary};
        use std::io::Cursor;

        let (w, h) = (11u32, 7u32);
        let src = DynamicImage::ImageRgb8(ImageBuffer::from_fn(w, h, |x, y| {
            Rgb([(x * 17) as u8, (y * 23) as u8, 91])
        }));
        let mut jpeg = Vec::new();
        src.write_to(&mut Cursor::new(&mut jpeg), image::ImageFormat::Jpeg)
            .expect("encode jpeg");
        let image = Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Image",
                "Width" => i64::from(w),
                "Height" => i64::from(h),
                "ColorSpace" => "DeviceRGB",
                "BitsPerComponent" => 8,
                "Filter" => "DCTDecode",
            },
            jpeg,
        )
        .with_compression(false);
        let path = build_single_page_pdf(Some(image));
        let bytes = std::fs::read(&path).expect("pin synthesized PDF bytes");
        std::fs::remove_file(&path).expect("remove the only filesystem copy");

        let from_bytes = PdfPages::from_bytes(bytes.clone()).expect("parse owned PDF bytes");
        let from_reader = PdfPages::from_reader(Cursor::new(bytes.clone()), "closed reader")
            .expect("parse PDF from one reader");
        assert_eq!(from_bytes.source_bytes(), bytes);
        assert_eq!(from_reader.source_bytes(), bytes);
        for pages in [&from_bytes, &from_reader] {
            assert_eq!(pages.len(), 1);
            let page = pages.render(0).expect("render from retained bytes");
            assert_eq!((page.width(), page.height()), (w, h));
        }
        assert!(
            !path.exists(),
            "neither retained-byte entry point may reopen or recreate the source path"
        );
    }

    /// A genuinely blank born-digital page is a valid vector page. It must use
    /// the declared page box and return white pixels rather than being confused
    /// with the old blanket "no image XObject" refusal.
    #[test]
    fn render_image_free_blank_page_uses_media_box() {
        let path = build_single_page_pdf(None);
        let pages = PdfPages::open(&path).expect("open synthesized pdf");
        assert_eq!(pages.len(), 1);
        let page = pages.render(0).expect("render blank vector page");
        assert_eq!((page.width(), page.height()), (100, 100));
        assert!(
            page.to_luma8().pixels().all(|pixel| pixel.0 == [255]),
            "blank vector page contains unexpected ink"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn encrypted_pdf_is_rejected_before_page_enumeration() {
        use lopdf::{
            EncryptionState, EncryptionVersion, Object, Permissions, StringFormat, dictionary,
        };

        let mut document = Document::with_version("1.5");
        let pages_id = document.new_object_id();
        let page_id = document.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "MediaBox" => vec![0_i64.into(), 0_i64.into(), 100_i64.into(), 100_i64.into()],
        });
        document.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => vec![page_id.into()],
                "Count" => 1,
            }),
        );
        let catalog_id = document.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        document.trailer.set("Root", catalog_id);
        let file_id = Object::String(b"focr-encrypted!!".to_vec(), StringFormat::Hexadecimal);
        document.trailer.set("ID", vec![file_id.clone(), file_id]);
        let encryption = EncryptionState::try_from(EncryptionVersion::V2 {
            document: &document,
            owner_password: "",
            user_password: "",
            key_length: 128,
            permissions: Permissions::all(),
        })
        .expect("construct test encryption state");
        document
            .encrypt(&encryption)
            .expect("encrypt synthesized PDF");
        let mut bytes = Vec::new();
        document
            .save_to(&mut bytes)
            .expect("serialize encrypted PDF");

        let error = match PdfPages::from_bytes(bytes) {
            Ok(_) => panic!("encrypted PDFs must refuse before exposing pages"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), "input_decode");
        assert!(
            error
                .to_string()
                .contains("encrypted PDFs are not supported"),
            "unexpected encrypted-PDF refusal: {error}"
        );
    }

    #[test]
    fn pdf_capability_marker_does_not_relabel_malformed_codestreams() {
        for detail in [
            "native vector/text render: PDF operator \"J\" is outside the admitted MTDT vector/text subset",
            "unused image resources, and repeated image invocations are not implemented in franken_ocr::pdf::PdfPages",
            "intervening PDF painting or graphics-state changes are not implemented",
            "general post-image content composition is not implemented in franken_ocr::pdf::PdfPages",
            "general vector, inline-image, shading, and clipping composition is not implemented",
            "PDF page content filters [ASCII85Decode] are not implemented by the bounded franken_ocr renderer",
            "JPEG 2000 embedded alpha is not implemented",
            "JPEG 2000 color space Cmyk is not implemented in the scalar PDF path",
            "JBIG2 soft masks with /Matte are not implemented",
            "unsupported image filter LZWDecode",
            "CCITTFaxDecode K>=0 (Group 3) is not supported; only Group 4 (K<0)",
            "layered scan CTM covers a partial-page image mosaic",
            "layered MRC rendering requires at least two painted image layers",
        ] {
            let unsupported = pdf_page_render_error(1, detail.to_owned());
            assert_eq!(unsupported.kind(), "input_decode");
            assert!(
                unsupported
                    .to_string()
                    .contains(PDF_UNSUPPORTED_SUBSET_MARKER),
                "capability detail was not marked: {detail}"
            );
        }

        for detail in [
            "parse PDF owned bytes: invalid xref",
            "decode Flate image: corrupt deflate stream",
            "decode JPEG image: truncated entropy-coded segment",
            "decode JPX image: malformed codestream",
            "decode JBIG2 image: malformed segment header",
        ] {
            let malformed = pdf_page_render_error(1, detail.to_owned());
            assert_eq!(malformed.kind(), "input_decode");
            assert!(
                !malformed
                    .to_string()
                    .contains(PDF_UNSUPPORTED_SUBSET_MARKER),
                "malformed data was mislabeled as a capability gap: {detail}"
            );
        }
    }

    /// A crafted image claiming gigapixel dimensions must be rejected by the
    /// dimension guard BEFORE any per-pixel allocation (no 280 TB reserve / no
    /// `width*height` overflow), regardless of the (never-reached) codec content.
    #[test]
    fn oversized_pdf_image_is_rejected_before_allocation() {
        use lopdf::{Stream, dictionary};

        let image = Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Image",
                "Width" => 100_000_i64,
                "Height" => 100_000_i64, // 1e10 px, far over the 1 Gpx cap
                "ColorSpace" => "DeviceRGB",
                "BitsPerComponent" => 8,
                "Filter" => "DCTDecode",
            },
            vec![0u8; 16], // dummy content; the guard fires before it is touched
        )
        .with_compression(false);
        let path = build_single_page_pdf(Some(image));
        let err = PdfPages::open(&path)
            .expect("open")
            .render(0)
            .expect_err("oversized image must error");
        assert!(err.to_string().contains("exceed"), "got: {err}");
        let _ = std::fs::remove_file(&path);
    }

    /// A multi-filter chain ending in an image codec (`[ASCII85Decode, DCTDecode]`)
    /// must be rejected with an accurate "chain ... unsupported" message rather than
    /// feeding still-ASCII-encoded bytes to the JPEG decoder.
    #[test]
    fn chained_filter_image_is_rejected() {
        use lopdf::{Object, Stream, dictionary};

        let image = Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Image",
                "Width" => 4_i64,
                "Height" => 4_i64,
                "ColorSpace" => "DeviceRGB",
                "BitsPerComponent" => 8,
                "Filter" => Object::Array(vec![
                    Object::Name(b"ASCII85Decode".to_vec()),
                    Object::Name(b"DCTDecode".to_vec()),
                ]),
            },
            vec![0u8; 16],
        )
        .with_compression(false);
        let path = build_single_page_pdf(Some(image));
        let err = PdfPages::open(&path)
            .expect("open")
            .render(0)
            .expect_err("chained filter must error");
        assert!(err.to_string().contains("chain"), "got: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn expected_sample_cap_clamps_a_hostile_bit_depth() {
        // A real 16-bit RGB image of these dims: 4 * w*h * 3 comps * 2 bytes.
        assert_eq!(
            expected_sample_cap(1024, 1024, 16, "DeviceRGB"),
            4 * 1024 * 1024 * 3 * 2
        );
        // A crafted /BitsPerComponent must NOT inflate the cap past the 16-bpc
        // figure — an unclamped i64::MAX would saturate `cap` to u64::MAX, which
        // makes bounded_inflate's `take(cap + 1)` effectively unbounded and
        // re-opens the decompression-bomb hole this guard exists to close.
        assert_eq!(
            expected_sample_cap(1024, 1024, i64::MAX, "DeviceRGB"),
            expected_sample_cap(1024, 1024, 16, "DeviceRGB")
        );
        // A zero/negative bit depth clamps UP to 1 (never a zero or giant cap).
        assert_eq!(expected_sample_cap(8, 8, -5, "DeviceGray"), 4 * 8 * 8);
        // Unknown color spaces take the 4-component (CMYK) upper bound.
        assert_eq!(expected_sample_cap(2, 2, 8, "Indexed"), 4 * 2 * 2 * 4);
    }

    #[test]
    fn bounded_inflate_passes_small_streams_and_rejects_a_bomb() {
        use std::io::Write;
        let zlib_of = |n: usize| -> Vec<u8> {
            let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::best());
            enc.write_all(&vec![0u8; n]).expect("encode");
            enc.finish().expect("finish")
        };

        // Inflates within the cap → Ok(Some(bytes)).
        let small = zlib_of(1000);
        let out = bounded_inflate(&small, 4096)
            .expect("no error")
            .expect("inflated");
        assert_eq!(out.len(), 1000);

        // A ~1000:1 "zip bomb" that inflates far past the cap → Err (the bomb signal),
        // having allocated at most cap + 1 bytes rather than the full inflation.
        let bomb = zlib_of(1_000_000);
        let err = bounded_inflate(&bomb, 4096).expect_err("bomb must be rejected");
        assert!(err.contains("cap"), "got: {err}");

        // Not standalone zlib → Ok(None) so the caller falls back to lopdf.
        assert!(
            bounded_inflate(b"not a zlib stream", 4096)
                .expect("no error")
                .is_none()
        );
    }
}
