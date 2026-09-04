//! Bounded selectable-text extraction for the exact MTDT PDF writer dialect.
//!
//! This is deliberately not a general PDF text extractor. It accepts embedded
//! Type0 `/Identity-H` fonts with one CIDFontType2 descendant and a complete
//! embedded ToUnicode CMap, plus the exact embedded Type1C/WinAnsi + Differences
//! subset used by LilyPond. Simple-font source codes and glyph names are always
//! retained; a glyph without an admitted Unicode mapping remains explicitly
//! opaque instead of being guessed or silently discarded.

use std::collections::BTreeMap;
use std::fmt;
use std::ops::RangeInclusive;

use lopdf::content::{Content, Operation};
use lopdf::{Dictionary, Document, Object, ObjectId, Stream};
use serde::Serialize;
use sha2::{Digest, Sha256};
use ttf_parser::cff;

use super::vector::{
    CidWidths, apply_lilypond_ext_gstate, load_simple_type1c_program, parse_cid_widths,
    parse_simple_type1_widths, win_ansi_glyph_name,
};
use super::{MAX_GRAPHICS_STATE_DEPTH, bounded_inflate, effective_xobjects, stream_filters};

/// Frozen schema identifier for [`PdfSelectableTextPageV2`].
pub const PDF_SELECTABLE_TEXT_SCHEMA_V2: &str = "franken_ocr.pdf.selectable_text.v2";

const MAX_PAGE_FONTS: usize = 64;
const MAX_CMAP_ENCODED_BYTES: usize = 8 * 1024 * 1024;
const MAX_CMAP_DECODED_BYTES: u64 = 16 * 1024 * 1024;
const MAX_CMAP_TOKENS: usize = 1_000_000;
const MAX_CMAP_ENTRIES: usize = 65_536;
const MAX_CMAP_RANGES: usize = 8_192;
const MAX_CMAP_TARGET_UTF16_BYTES: usize = 64 * 1024;
const MAX_CMAP_MAPPING_UTF16_BYTES: usize = 64 * 1024 * 1024;
const MAX_CMAP_MAPPING_STRING_BYTES: usize = 64 * 1024 * 1024;
const MAX_CMAP_MAPPING_UNICODE_SCALARS: usize = 4_000_000;
const MAX_TEXT_RUNS: usize = 100_000;
const MAX_TEXT_CODES: usize = 1_000_000;
const MAX_DECODED_UNICODE_SCALARS: usize = 4_000_000;
const MAX_OUTPUT_STRING_BYTES: usize = 64 * 1024 * 1024;
const MAX_RESOURCE_NAME_BYTES: usize = 127;
const MAX_TJ_OPERANDS: usize = 100_000;
const MAX_PAGE_TJ_OPERANDS: usize = 1_000_000;
const MAX_DASH_ARRAY_ENTRIES: usize = 64;
const MAX_PAGE_ANNOTATIONS: usize = 256;

/// Stable failure classification for the writer-specific selectable-text API.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PdfSelectableTextErrorKind {
    /// The requested zero-based page index does not exist.
    PageOutOfRange,
    /// The shared bounded page-content decoder refused the page.
    PageContent,
    /// A referenced page, font, or XObject resource does not exist.
    MissingResource,
    /// A font is outside the exact Type0/Identity-H or Type1C/WinAnsi subsets.
    UnsupportedFont,
    /// A ToUnicode program uses a mapping form outside the admitted subset.
    UnsupportedCMap,
    /// A ToUnicode program or text operand is malformed.
    MalformedCMap,
    /// A source code appears more than once with the same mapping.
    DuplicateMapping,
    /// A source code appears more than once with different mappings.
    ConflictingMapping,
    /// A shown Type0 CID has no ToUnicode mapping, or a Type1 code has no glyph.
    MissingMapping,
    /// A text-showing operand violates its admitted one- or two-byte form.
    MalformedTextOperand,
    /// A text-showing construct can be visible but is not admitted.
    UnsupportedTextOperator,
    /// A finite parser/state/output bound was exceeded.
    LimitExceeded,
    /// Text or graphics state is unbalanced, misplaced, or non-finite.
    InvalidState,
}

impl PdfSelectableTextErrorKind {
    /// Stable machine-readable spelling used by diagnostics and tests.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PageOutOfRange => "page_out_of_range",
            Self::PageContent => "page_content",
            Self::MissingResource => "missing_resource",
            Self::UnsupportedFont => "unsupported_font",
            Self::UnsupportedCMap => "unsupported_cmap",
            Self::MalformedCMap => "malformed_cmap",
            Self::DuplicateMapping => "duplicate_mapping",
            Self::ConflictingMapping => "conflicting_mapping",
            Self::MissingMapping => "missing_mapping",
            Self::MalformedTextOperand => "malformed_text_operand",
            Self::UnsupportedTextOperator => "unsupported_text_operator",
            Self::LimitExceeded => "limit_exceeded",
            Self::InvalidState => "invalid_state",
        }
    }
}

/// Typed, contextual refusal from [`super::PdfPages::selectable_text`].
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PdfSelectableTextError {
    /// Stable failure category.
    pub kind: PdfSelectableTextErrorKind,
    /// Zero-based page index supplied by the caller.
    pub page_index: usize,
    /// Content operation index when the failure is operation-specific.
    pub operation_index: Option<usize>,
    /// Font resource name, without the PDF `/` prefix, when known.
    pub font_resource: Option<String>,
    /// Deterministic provider diagnostic. It is not part of any output identity.
    pub detail: String,
}

impl fmt::Display for PdfSelectableTextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "PDF selectable text {} on zero-based page {}",
            self.kind.as_str(),
            self.page_index
        )?;
        if let Some(operation_index) = self.operation_index {
            write!(formatter, ", operation {operation_index}")?;
        }
        if let Some(font_resource) = &self.font_resource {
            write!(formatter, ", font /{font_resource}")?;
        }
        write!(formatter, ": {}", self.detail)
    }
}

impl std::error::Error for PdfSelectableTextError {}

/// Declared PDF text-position state at the start of a returned string operand.
///
/// These are raw PDF matrices and text-state values, not inferred baselines,
/// bounding boxes, shaping results, or semantic layout claims.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PdfTextPositionV1 {
    /// Current transformation matrix from graphics state.
    pub graphics_matrix: [f64; 6],
    /// Current text matrix at operand start, including all prior glyph and `TJ`
    /// advances since the most recent BT/Tm/Td/TD/T* positioning operation.
    pub text_matrix: [f64; 6],
    /// Current text-line matrix.
    pub line_matrix: [f64; 6],
    /// Selected font size from `Tf`.
    pub font_size: f64,
    /// Character spacing from `Tc`.
    pub character_spacing: f64,
    /// Word spacing from `Tw`.
    pub word_spacing: f64,
    /// Horizontal scaling percentage from `Tz`.
    pub horizontal_scaling_percent: f64,
    /// Text leading from `TL`.
    pub leading: f64,
    /// Text rise from `Ts`.
    pub rise: f64,
    /// Text rendering mode from `Tr` (`0..=7`).
    pub rendering_mode: u8,
}

/// Exact provider-owned decoding path used for a selectable-text run.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PdfSelectableTextEncodingV2 {
    /// Two-byte Type0 `/Identity-H` codes decoded only by embedded ToUnicode.
    Type0IdentityH,
    /// One-byte Type1C codes decoded by WinAnsi plus the declared Differences.
    Type1cWinAnsiDifferences,
}

impl PdfSelectableTextEncodingV2 {
    /// Stable machine-readable spelling used by embedding callers and hashes.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Type0IdentityH => "type0_identity_h",
            Self::Type1cWinAnsiDifferences => "type1c_win_ansi_differences",
        }
    }
}

/// One string operand decoded through its selected font's admitted mapping.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PdfSelectableTextRunV2 {
    /// Stable zero-based ordinal in this page result.
    pub run_index: usize,
    /// Stable SHA-256 over this run's canonical fields.
    pub identity_sha256: [u8; 32],
    /// Content operation index in the shared bounded operation stream.
    pub operation_index: usize,
    /// Showing-string operand index: TJ array position, zero for Tj/`'`, two for `"`.
    pub operand_index: usize,
    /// Selected page font resource, without the PDF `/` prefix.
    pub font_resource: String,
    /// Exact encoding/font subtype path used to interpret source codes.
    pub font_encoding: PdfSelectableTextEncodingV2,
    /// Fixed source-code width for this run: two for Identity-H, one for Type1C.
    pub code_width_bytes: u8,
    /// SHA-256 commitment to the exact canonical decoding map consumed.
    ///
    /// For Type0 this is the bounded decoded ToUnicode program hash. For Type1C
    /// it is the canonical 256-entry WinAnsi + Differences glyph-name map hash.
    pub font_mapping_sha256: [u8; 32],
    /// Source codes in order. One-byte Type1C codes are losslessly widened.
    pub source_codes: Vec<u16>,
    /// Exact glyph name for each source code when the encoding declares one.
    /// Type0/ToUnicode runs carry `None` because a CID is not a glyph name.
    pub glyph_names: Vec<Option<String>>,
    /// Per-source-code Unicode mappings; opaque music glyphs remain `None`.
    pub unicode_by_code: Vec<Option<String>>,
    /// Whether every source code in this run has a non-guessed Unicode mapping.
    pub unicode_complete: bool,
    /// Ordered concatenation of only the present entries in `unicode_by_code`.
    /// Consult `unicode_complete` and `unicode_by_code` before treating it as a
    /// lossless text transcription.
    pub unicode: String,
    /// Raw declared position/text-state snapshot before this string operand.
    pub position: PdfTextPositionV1,
    /// Numeric TJ operands since the prior string element.
    pub tj_adjustments_before: Vec<f64>,
    /// Trailing numeric TJ operands after the final string element.
    pub tj_adjustments_after: Vec<f64>,
}

impl PdfSelectableTextRunV2 {
    /// Compute the identity commitment for the run's current public fields.
    #[must_use]
    pub fn computed_identity_sha256(&self) -> [u8; 32] {
        run_identity(self)
    }

    /// Return whether the stored identity still commits to every public run
    /// field. Embedding libraries should call this before projecting a receipt.
    #[must_use]
    pub fn identity_is_valid(&self) -> bool {
        self.identity_sha256 == self.computed_identity_sha256()
    }
}

/// Versioned selectable-text result for exactly one PDF page.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PdfSelectableTextPageV2 {
    /// Frozen result schema identifier.
    pub schema: &'static str,
    /// Zero-based page index, matching [`super::PdfPages::render`].
    pub page_index: usize,
    /// SHA-256 of the exact immutable PDF source bytes retained by PdfPages.
    pub source_sha256: [u8; 32],
    /// Stable SHA-256 over schema, source, page index, and ordered run identities.
    pub identity_sha256: [u8; 32],
    /// Text runs in content-operation order and then TJ operand order.
    pub runs: Vec<PdfSelectableTextRunV2>,
}

impl PdfSelectableTextPageV2 {
    /// Compute the page-root commitment for the current schema, source, index,
    /// and ordered stored run identities.
    #[must_use]
    pub fn computed_identity_sha256(&self) -> [u8; 32] {
        page_identity(self)
    }

    /// Return whether every run identity and the page root identity still bind
    /// the complete decoded result.
    #[must_use]
    pub fn identity_is_valid(&self) -> bool {
        self.runs
            .iter()
            .all(PdfSelectableTextRunV2::identity_is_valid)
            && self.identity_sha256 == self.computed_identity_sha256()
    }
}

#[derive(Clone, Copy, Debug)]
struct Matrix([f64; 6]);

impl Matrix {
    const IDENTITY: Self = Self([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    fn concat(self, right: Self) -> Self {
        let [a, b, c, d, e, f] = self.0;
        let [ra, rb, rc, rd, re, rf] = right.0;
        Self([
            a * ra + c * rb,
            b * ra + d * rb,
            a * rc + c * rd,
            b * rc + d * rd,
            a * re + c * rf + e,
            b * re + d * rf + f,
        ])
    }

    fn translated(self, tx: f64, ty: f64) -> Self {
        Self([
            self.0[0],
            self.0[1],
            self.0[2],
            self.0[3],
            self.0[0] * tx + self.0[2] * ty + self.0[4],
            self.0[1] * tx + self.0[3] * ty + self.0[5],
        ])
    }

    fn is_finite(self) -> bool {
        self.0.into_iter().all(f64::is_finite)
    }
}

#[derive(Clone, Debug)]
struct GraphicsTextState {
    ctm: Matrix,
    font_name: Option<Vec<u8>>,
    font_size: f64,
    character_spacing: f64,
    word_spacing: f64,
    horizontal_scaling_percent: f64,
    leading: f64,
    rise: f64,
    rendering_mode: u8,
    nonstroking_color_space: DeviceColorSpace,
    stroking_color_space: DeviceColorSpace,
}

impl Default for GraphicsTextState {
    fn default() -> Self {
        Self {
            ctm: Matrix::IDENTITY,
            font_name: None,
            font_size: 0.0,
            character_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling_percent: 100.0,
            leading: 0.0,
            rise: 0.0,
            rendering_mode: 0,
            nonstroking_color_space: DeviceColorSpace::Gray,
            stroking_color_space: DeviceColorSpace::Gray,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeviceColorSpace {
    Gray,
    Rgb,
    Cmyk,
}

impl DeviceColorSpace {
    const fn component_count(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Rgb => 3,
            Self::Cmyk => 4,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TextObjectState {
    matrix: Matrix,
    line_matrix: Matrix,
}

#[derive(Debug)]
struct MarkedContentState {
    actual_text: Option<String>,
    first_run_index: usize,
}

impl Default for TextObjectState {
    fn default() -> Self {
        Self {
            matrix: Matrix::IDENTITY,
            line_matrix: Matrix::IDENTITY,
        }
    }
}

#[derive(Debug)]
struct FontMap {
    encoding: PdfSelectableTextEncodingV2,
    code_width_bytes: u8,
    mapping_sha256: [u8; 32],
    mappings: BTreeMap<u16, String>,
    glyph_names: Vec<Option<String>>,
    available_glyphs: Vec<bool>,
    widths: FontWidths,
}

#[derive(Debug)]
enum FontWidths {
    Type0(CidWidths),
    Type1c(Vec<Option<f64>>),
}

impl FontWidths {
    fn get(&self, code: u16) -> Option<f64> {
        match self {
            Self::Type0(widths) => Some(widths.get(code)),
            Self::Type1c(widths) => widths.get(usize::from(code)).and_then(|width| *width),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CMapToken {
    Word(String),
    Hex(Vec<u8>),
    Literal,
    ArrayStart,
    ArrayEnd,
    DictionaryStart,
    DictionaryEnd,
    ProcedureStart,
    ProcedureEnd,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CMapContainer {
    Array,
    Dictionary,
}

#[derive(Default)]
struct CMapMappingBudget {
    utf16_bytes: usize,
    string_bytes: usize,
    unicode_scalars: usize,
}

impl CMapMappingBudget {
    fn ensure_target_utf16_bytes(
        &self,
        bytes: usize,
    ) -> Result<(), (PdfSelectableTextErrorKind, String)> {
        if bytes > MAX_CMAP_TARGET_UTF16_BYTES {
            return Err((
                PdfSelectableTextErrorKind::LimitExceeded,
                format!(
                    "ToUnicode target has {bytes} UTF-16BE bytes, exceeding {MAX_CMAP_TARGET_UTF16_BYTES}"
                ),
            ));
        }
        Ok(())
    }

    fn ensure_utf16_growth(
        &self,
        additional: usize,
    ) -> Result<(), (PdfSelectableTextErrorKind, String)> {
        let total = self.utf16_bytes.checked_add(additional).ok_or_else(|| {
            (
                PdfSelectableTextErrorKind::LimitExceeded,
                "ToUnicode target-byte count overflow".to_owned(),
            )
        })?;
        if total > MAX_CMAP_MAPPING_UTF16_BYTES {
            return Err((
                PdfSelectableTextErrorKind::LimitExceeded,
                format!(
                    "ToUnicode mappings require {total} UTF-16BE target bytes, exceeding {MAX_CMAP_MAPPING_UTF16_BYTES}"
                ),
            ));
        }
        Ok(())
    }

    fn ensure_repeated_utf16_growth(
        &self,
        target_bytes: usize,
        repetitions: usize,
    ) -> Result<(), (PdfSelectableTextErrorKind, String)> {
        self.ensure_target_utf16_bytes(target_bytes)?;
        let additional = target_bytes.checked_mul(repetitions).ok_or_else(|| {
            (
                PdfSelectableTextErrorKind::LimitExceeded,
                "ToUnicode repeated target-byte count overflow".to_owned(),
            )
        })?;
        self.ensure_utf16_growth(additional)
    }

    fn charge(
        &mut self,
        target_utf16_bytes: usize,
        target: &str,
    ) -> Result<(), (PdfSelectableTextErrorKind, String)> {
        self.ensure_target_utf16_bytes(target_utf16_bytes)?;
        self.ensure_utf16_growth(target_utf16_bytes)?;
        let string_bytes = self.string_bytes.checked_add(target.len()).ok_or_else(|| {
            (
                PdfSelectableTextErrorKind::LimitExceeded,
                "ToUnicode mapped-string byte count overflow".to_owned(),
            )
        })?;
        if string_bytes > MAX_CMAP_MAPPING_STRING_BYTES {
            return Err((
                PdfSelectableTextErrorKind::LimitExceeded,
                format!(
                    "ToUnicode mapped strings require {string_bytes} bytes, exceeding {MAX_CMAP_MAPPING_STRING_BYTES}"
                ),
            ));
        }
        let target_scalars = target.chars().count();
        let unicode_scalars = self
            .unicode_scalars
            .checked_add(target_scalars)
            .ok_or_else(|| {
                (
                    PdfSelectableTextErrorKind::LimitExceeded,
                    "ToUnicode mapped scalar count overflow".to_owned(),
                )
            })?;
        if unicode_scalars > MAX_CMAP_MAPPING_UNICODE_SCALARS {
            return Err((
                PdfSelectableTextErrorKind::LimitExceeded,
                format!(
                    "ToUnicode mappings contain {unicode_scalars} Unicode scalars, exceeding {MAX_CMAP_MAPPING_UNICODE_SCALARS}"
                ),
            ));
        }
        self.utf16_bytes += target_utf16_bytes;
        self.string_bytes = string_bytes;
        self.unicode_scalars = unicode_scalars;
        Ok(())
    }
}

struct Extractor<'a> {
    doc: &'a Document,
    page_id: ObjectId,
    page_index: usize,
    fonts: BTreeMap<Vec<u8>, FontMap>,
    graphics: GraphicsTextState,
    graphics_stack: Vec<GraphicsTextState>,
    text: Option<TextObjectState>,
    marked_content: Option<MarkedContentState>,
    runs: Vec<PdfSelectableTextRunV2>,
    code_count: usize,
    unicode_scalar_count: usize,
    output_string_bytes: usize,
    tj_operand_count: usize,
}

impl<'a> Extractor<'a> {
    fn new(doc: &'a Document, page_id: ObjectId, page_index: usize) -> Self {
        Self {
            doc,
            page_id,
            page_index,
            fonts: BTreeMap::new(),
            graphics: GraphicsTextState::default(),
            graphics_stack: Vec::new(),
            text: None,
            marked_content: None,
            runs: Vec::new(),
            code_count: 0,
            unicode_scalar_count: 0,
            output_string_bytes: 0,
            tj_operand_count: 0,
        }
    }

    fn error(
        &self,
        kind: PdfSelectableTextErrorKind,
        operation_index: Option<usize>,
        font_name: Option<&[u8]>,
        detail: impl Into<String>,
    ) -> PdfSelectableTextError {
        PdfSelectableTextError {
            kind,
            page_index: self.page_index,
            operation_index,
            font_resource: font_name
                .and_then(|name| std::str::from_utf8(name).ok().map(str::to_owned)),
            detail: detail.into(),
        }
    }

    fn process(
        mut self,
        content: &Content<Vec<Operation>>,
    ) -> Result<Vec<PdfSelectableTextRunV2>, PdfSelectableTextError> {
        for (operation_index, operation) in content.operations.iter().enumerate() {
            self.operation(operation_index, operation)?;
        }
        if self.text.is_some() {
            return Err(self.error(
                PdfSelectableTextErrorKind::InvalidState,
                None,
                None,
                "unterminated text object (BT without ET)",
            ));
        }
        if self.marked_content.is_some() {
            return Err(self.error(
                PdfSelectableTextErrorKind::InvalidState,
                None,
                None,
                "unterminated marked content (BDC without EMC)",
            ));
        }
        if !self.graphics_stack.is_empty() {
            return Err(self.error(
                PdfSelectableTextErrorKind::InvalidState,
                None,
                None,
                "unbalanced graphics-state save (q without Q)",
            ));
        }
        Ok(self.runs)
    }

    fn operation(
        &mut self,
        operation_index: usize,
        operation: &Operation,
    ) -> Result<(), PdfSelectableTextError> {
        match operation.operator.as_str() {
            "q" => {
                self.expect_no_operands(operation_index, operation)?;
                if self.graphics_stack.len() >= MAX_GRAPHICS_STATE_DEPTH {
                    return Err(self.error(
                        PdfSelectableTextErrorKind::LimitExceeded,
                        Some(operation_index),
                        None,
                        format!("graphics-state stack exceeds {MAX_GRAPHICS_STATE_DEPTH} entries"),
                    ));
                }
                self.graphics_stack.push(self.graphics.clone());
            }
            "Q" => {
                self.expect_no_operands(operation_index, operation)?;
                self.graphics = self.graphics_stack.pop().ok_or_else(|| {
                    self.error(
                        PdfSelectableTextErrorKind::InvalidState,
                        Some(operation_index),
                        None,
                        "graphics-state restore Q has no matching q",
                    )
                })?;
            }
            "cm" => {
                let matrix = self.matrix_operands(operation_index, &operation.operands, "cm")?;
                self.graphics.ctm = self.graphics.ctm.concat(matrix);
                if !self.graphics.ctm.is_finite() {
                    return Err(self.error(
                        PdfSelectableTextErrorKind::InvalidState,
                        Some(operation_index),
                        None,
                        "cumulative graphics matrix is non-finite",
                    ));
                }
            }
            "BT" => {
                self.expect_no_operands(operation_index, operation)?;
                if self.text.replace(TextObjectState::default()).is_some() {
                    return Err(self.error(
                        PdfSelectableTextErrorKind::InvalidState,
                        Some(operation_index),
                        None,
                        "nested BT text object",
                    ));
                }
            }
            "ET" => {
                self.expect_no_operands(operation_index, operation)?;
                if self.text.take().is_none() {
                    return Err(self.error(
                        PdfSelectableTextErrorKind::InvalidState,
                        Some(operation_index),
                        None,
                        "ET appears outside a text object",
                    ));
                }
            }
            "BDC" => self.begin_marked_content(operation_index, &operation.operands)?,
            "EMC" => {
                self.expect_no_operands(operation_index, operation)?;
                self.end_marked_content(operation_index)?;
            }
            "Tf" => self.set_font(operation_index, &operation.operands)?,
            "Tm" => {
                self.require_text(operation_index, "Tm")?;
                let matrix = self.matrix_operands(operation_index, &operation.operands, "Tm")?;
                let text = self.text.as_mut().expect("checked text state");
                text.matrix = matrix;
                text.line_matrix = matrix;
            }
            "Td" => self.translate_text(operation_index, &operation.operands, false)?,
            "TD" => self.translate_text(operation_index, &operation.operands, true)?,
            "T*" => {
                self.expect_no_operands(operation_index, operation)?;
                self.next_text_line(operation_index, "T*")?;
            }
            "Tc" => self.set_number(
                operation_index,
                &operation.operands,
                "Tc",
                |state, value| state.character_spacing = value,
            )?,
            "Tw" => self.set_number(
                operation_index,
                &operation.operands,
                "Tw",
                |state, value| state.word_spacing = value,
            )?,
            "Tz" => self.set_number(
                operation_index,
                &operation.operands,
                "Tz",
                |state, value| state.horizontal_scaling_percent = value,
            )?,
            "TL" => self.set_number(
                operation_index,
                &operation.operands,
                "TL",
                |state, value| state.leading = value,
            )?,
            "Ts" => self.set_number(
                operation_index,
                &operation.operands,
                "Ts",
                |state, value| state.rise = value,
            )?,
            "Tr" => self.set_rendering_mode(operation_index, &operation.operands)?,
            "Tj" => self.show_tj(operation_index, &operation.operands)?,
            "TJ" => self.show_tj_array(operation_index, &operation.operands)?,
            "'" => self.show_single_quote(operation_index, &operation.operands)?,
            "\"" => self.show_double_quote(operation_index, &operation.operands)?,
            "Do" => self.validate_xobject(operation_index, &operation.operands)?,
            // Color and rendering-intent operators do not alter Unicode or the
            // raw text-position state exposed here.
            "gs" => apply_lilypond_ext_gstate(self.doc, self.page_id, &operation.operands)
                .map_err(|detail| {
                    self.error(
                        PdfSelectableTextErrorKind::UnsupportedTextOperator,
                        Some(operation_index),
                        self.graphics.font_name.as_deref(),
                        detail,
                    )
                })?,
            // These admitted graphics/path operators cannot alter text decoding
            // or text position. They are explicit so marked-content, ActualText,
            // inline-image, compatibility, and future operators fail closed.
            operator @ ("w" | "J" | "j" | "d" | "m" | "l" | "c" | "re" | "h" | "S" | "s" | "f"
            | "F" | "f*" | "n" | "g" | "G" | "rg" | "RG" | "k" | "K" | "cs" | "CS"
            | "sc" | "SC" | "scn" | "SCN" | "ri")
                if self.text.is_none() =>
            {
                self.validate_nonsemantic_graphics_operator(
                    operation_index,
                    operator,
                    &operation.operands,
                )?;
            }
            other if self.text.is_some() => {
                return Err(self.error(
                    PdfSelectableTextErrorKind::UnsupportedTextOperator,
                    Some(operation_index),
                    self.graphics.font_name.as_deref(),
                    format!("operator {other:?} inside a text object is not admitted"),
                ));
            }
            other => {
                return Err(self.error(
                    PdfSelectableTextErrorKind::UnsupportedTextOperator,
                    Some(operation_index),
                    self.graphics.font_name.as_deref(),
                    format!(
                        "operator {other:?} outside a text object is not in the explicit non-semantic graphics allowlist"
                    ),
                ));
            }
        }
        Ok(())
    }

    fn expect_no_operands(
        &self,
        operation_index: usize,
        operation: &Operation,
    ) -> Result<(), PdfSelectableTextError> {
        if operation.operands.is_empty() {
            Ok(())
        } else {
            Err(self.error(
                PdfSelectableTextErrorKind::MalformedTextOperand,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                format!("{} expects no operands", operation.operator),
            ))
        }
    }

    fn begin_marked_content(
        &mut self,
        operation_index: usize,
        operands: &[Object],
    ) -> Result<(), PdfSelectableTextError> {
        if self.marked_content.is_some() {
            return Err(self.error(
                PdfSelectableTextErrorKind::InvalidState,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                "nested BDC marked content is outside the admitted MTDT writer dialect",
            ));
        }
        let [Object::Name(tag), Object::Dictionary(properties)] = operands else {
            return Err(self.error(
                PdfSelectableTextErrorKind::MalformedTextOperand,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                "BDC expects an exact name and direct properties dictionary",
            ));
        };
        if tag.as_slice() != b"Span" {
            return Err(self.error(
                PdfSelectableTextErrorKind::UnsupportedTextOperator,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                "only the MTDT writer's /Span marked-content tag is admitted",
            ));
        }

        let mut actual_text = None;
        for (key, value) in properties.iter() {
            if key.as_slice() == b"ActualText" {
                let Object::String(bytes, _) = value else {
                    return Err(self.error(
                        PdfSelectableTextErrorKind::MalformedTextOperand,
                        Some(operation_index),
                        self.graphics.font_name.as_deref(),
                        "/ActualText must be one direct PDF string",
                    ));
                };
                if actual_text.is_some() {
                    return Err(self.error(
                        PdfSelectableTextErrorKind::MalformedTextOperand,
                        Some(operation_index),
                        self.graphics.font_name.as_deref(),
                        "BDC properties contain duplicate /ActualText",
                    ));
                }
                actual_text = Some(decode_actual_text(bytes).map_err(|detail| {
                    self.error(
                        PdfSelectableTextErrorKind::MalformedTextOperand,
                        Some(operation_index),
                        self.graphics.font_name.as_deref(),
                        detail,
                    )
                })?);
            } else if key.starts_with(b"MTDT") {
                if !matches!(value, Object::Name(_) | Object::String(_, _)) {
                    return Err(self.error(
                        PdfSelectableTextErrorKind::MalformedTextOperand,
                        Some(operation_index),
                        self.graphics.font_name.as_deref(),
                        format!(
                            "MTDT marked-content property /{} must be a direct name or string",
                            String::from_utf8_lossy(key)
                        ),
                    ));
                }
            } else {
                return Err(self.error(
                    PdfSelectableTextErrorKind::UnsupportedTextOperator,
                    Some(operation_index),
                    self.graphics.font_name.as_deref(),
                    format!(
                        "marked-content property /{} is outside the admitted MTDT writer dialect",
                        String::from_utf8_lossy(key)
                    ),
                ));
            }
        }
        if properties.is_empty() {
            return Err(self.error(
                PdfSelectableTextErrorKind::MalformedTextOperand,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                "BDC properties dictionary must not be empty",
            ));
        }
        self.marked_content = Some(MarkedContentState {
            actual_text,
            first_run_index: self.runs.len(),
        });
        Ok(())
    }

    fn end_marked_content(&mut self, operation_index: usize) -> Result<(), PdfSelectableTextError> {
        let state = self.marked_content.take().ok_or_else(|| {
            self.error(
                PdfSelectableTextErrorKind::InvalidState,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                "EMC has no matching BDC",
            )
        })?;
        let Some(actual_text) = state.actual_text else {
            return Ok(());
        };
        let enclosed_run_count = self.runs.len().saturating_sub(state.first_run_index);
        if enclosed_run_count == 0 {
            // MTDT uses the same semantic marked-content shape around vector
            // paths. Those paths have no selectable-text position or font run;
            // the PDF accessibility tree still consumes their ActualText.
            return Ok(());
        }
        if enclosed_run_count != 1 {
            return Err(self.error(
                PdfSelectableTextErrorKind::UnsupportedTextOperator,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                format!(
                    "ActualText encloses {enclosed_run_count} text runs; the admitted MTDT writer dialect requires exactly one"
                ),
            ));
        }

        let run = &self.runs[state.first_run_index];
        let old_bytes = run
            .unicode_by_code
            .iter()
            .flatten()
            .map(String::len)
            .sum::<usize>();
        let old_scalars = run.unicode.chars().count();
        let new_scalars = actual_text.chars().count();
        let prospective_output_string_bytes = self
            .output_string_bytes
            .checked_sub(old_bytes)
            .and_then(|bytes| bytes.checked_add(actual_text.len()))
            .filter(|bytes| *bytes <= MAX_OUTPUT_STRING_BYTES)
            .ok_or_else(|| {
                self.error(
                    PdfSelectableTextErrorKind::LimitExceeded,
                    Some(operation_index),
                    self.graphics.font_name.as_deref(),
                    format!(
                        "page selectable-text owned string payload exceeds {MAX_OUTPUT_STRING_BYTES} bytes after ActualText"
                    ),
                )
            })?;
        let prospective_unicode_scalar_count = self
            .unicode_scalar_count
            .checked_sub(old_scalars)
            .and_then(|count| count.checked_add(new_scalars))
            .filter(|count| *count <= MAX_DECODED_UNICODE_SCALARS)
            .ok_or_else(|| {
                self.error(
                    PdfSelectableTextErrorKind::LimitExceeded,
                    Some(operation_index),
                    self.graphics.font_name.as_deref(),
                    format!(
                        "page exceeds the {MAX_DECODED_UNICODE_SCALARS}-scalar Unicode limit after ActualText"
                    ),
                )
            })?;

        let run = &mut self.runs[state.first_run_index];
        run.unicode_by_code = (0..run.source_codes.len())
            .map(|index| {
                Some(if index == 0 {
                    actual_text.clone()
                } else {
                    String::new()
                })
            })
            .collect();
        run.unicode_complete = true;
        run.unicode = actual_text;
        run.identity_sha256 = run_identity(run);
        self.output_string_bytes = prospective_output_string_bytes;
        self.unicode_scalar_count = prospective_unicode_scalar_count;
        Ok(())
    }

    fn validate_nonsemantic_graphics_operator(
        &mut self,
        operation_index: usize,
        operator: &str,
        operands: &[Object],
    ) -> Result<(), PdfSelectableTextError> {
        let result = match operator {
            "w" => validate_exact_finite_numbers(operands, 1, "w").map(|()| None),
            "J" | "j" => validate_integer_enum(operands, operator, 0..=2).map(|()| None),
            "d" => validate_dash_operands(operands).map(|()| None),
            "m" | "l" => validate_exact_finite_numbers(operands, 2, operator).map(|()| None),
            "c" => validate_exact_finite_numbers(operands, 6, "c").map(|()| None),
            "re" => validate_exact_finite_numbers(operands, 4, "re").map(|()| None),
            "h" | "S" | "s" | "f" | "F" | "f*" | "n" => {
                validate_zero_operands(operands, operator).map(|()| None)
            }
            "g" => validate_exact_finite_numbers(operands, 1, operator)
                .map(|()| Some((false, DeviceColorSpace::Gray))),
            "G" => validate_exact_finite_numbers(operands, 1, operator)
                .map(|()| Some((true, DeviceColorSpace::Gray))),
            "rg" => validate_exact_finite_numbers(operands, 3, operator)
                .map(|()| Some((false, DeviceColorSpace::Rgb))),
            "RG" => validate_exact_finite_numbers(operands, 3, operator)
                .map(|()| Some((true, DeviceColorSpace::Rgb))),
            "k" => validate_exact_finite_numbers(operands, 4, operator)
                .map(|()| Some((false, DeviceColorSpace::Cmyk))),
            "K" => validate_exact_finite_numbers(operands, 4, operator)
                .map(|()| Some((true, DeviceColorSpace::Cmyk))),
            "cs" => validate_device_color_space_operand(operands, operator)
                .map(|space| Some((false, space))),
            "CS" => validate_device_color_space_operand(operands, operator)
                .map(|space| Some((true, space))),
            "ri" => validate_single_name_operand(operands, operator).map(|()| None),
            "sc" | "scn" => validate_color_operands(
                operands,
                operator,
                self.graphics.nonstroking_color_space.component_count(),
            )
            .map(|()| None),
            "SC" | "SCN" => validate_color_operands(
                operands,
                operator,
                self.graphics.stroking_color_space.component_count(),
            )
            .map(|()| None),
            _ => unreachable!("caller restricts the non-semantic graphics allowlist"),
        };
        let update = result.map_err(|detail| {
            self.error(
                PdfSelectableTextErrorKind::MalformedTextOperand,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                detail,
            )
        })?;
        if let Some((stroking, space)) = update {
            if stroking {
                self.graphics.stroking_color_space = space;
            } else {
                self.graphics.nonstroking_color_space = space;
            }
        }
        Ok(())
    }

    fn require_text(
        &self,
        operation_index: usize,
        operator: &str,
    ) -> Result<(), PdfSelectableTextError> {
        if self.text.is_some() {
            Ok(())
        } else {
            Err(self.error(
                PdfSelectableTextErrorKind::InvalidState,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                format!("{operator} appears outside a BT/ET text object"),
            ))
        }
    }

    fn matrix_operands(
        &self,
        operation_index: usize,
        operands: &[Object],
        operator: &str,
    ) -> Result<Matrix, PdfSelectableTextError> {
        if operands.len() != 6 {
            return Err(self.error(
                PdfSelectableTextErrorKind::MalformedTextOperand,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                format!("{operator} expects six numeric operands"),
            ));
        }
        let mut values = [0.0; 6];
        for (slot, object) in values.iter_mut().zip(operands) {
            *slot = object_number(object).ok_or_else(|| {
                self.error(
                    PdfSelectableTextErrorKind::MalformedTextOperand,
                    Some(operation_index),
                    self.graphics.font_name.as_deref(),
                    format!("{operator} contains a non-numeric operand"),
                )
            })?;
        }
        let matrix = Matrix(values);
        if matrix.is_finite() {
            Ok(matrix)
        } else {
            Err(self.error(
                PdfSelectableTextErrorKind::InvalidState,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                format!("{operator} contains a non-finite value"),
            ))
        }
    }

    fn set_font(
        &mut self,
        operation_index: usize,
        operands: &[Object],
    ) -> Result<(), PdfSelectableTextError> {
        self.require_text(operation_index, "Tf")?;
        let [Object::Name(name), size] = operands else {
            return Err(self.error(
                PdfSelectableTextErrorKind::MalformedTextOperand,
                Some(operation_index),
                None,
                "Tf expects a font resource name and numeric size",
            ));
        };
        let font_resource = resource_name(name).map_err(|detail| {
            self.error(
                PdfSelectableTextErrorKind::MissingResource,
                Some(operation_index),
                None,
                detail,
            )
        })?;
        let size = object_number(size)
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| {
                self.error(
                    PdfSelectableTextErrorKind::InvalidState,
                    Some(operation_index),
                    Some(name),
                    "Tf font size must be finite and greater than zero",
                )
            })?;
        if !self.fonts.contains_key(name) {
            if self.fonts.len() >= MAX_PAGE_FONTS {
                return Err(self.error(
                    PdfSelectableTextErrorKind::LimitExceeded,
                    Some(operation_index),
                    Some(name),
                    format!("page exceeds the {MAX_PAGE_FONTS}-font selectable-text limit"),
                ));
            }
            let map = load_font_map(
                self.doc,
                self.page_id,
                self.page_index,
                operation_index,
                name,
                &font_resource,
            )?;
            self.fonts.insert(name.clone(), map);
        }
        self.graphics.font_name = Some(name.clone());
        self.graphics.font_size = size;
        Ok(())
    }

    fn translate_text(
        &mut self,
        operation_index: usize,
        operands: &[Object],
        set_leading: bool,
    ) -> Result<(), PdfSelectableTextError> {
        self.require_text(operation_index, if set_leading { "TD" } else { "Td" })?;
        if operands.len() != 2 {
            return Err(self.error(
                PdfSelectableTextErrorKind::MalformedTextOperand,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                "Td/TD expects two numeric operands",
            ));
        }
        let tx = object_number(&operands[0]);
        let ty = object_number(&operands[1]);
        let (tx, ty) = match (tx, ty) {
            (Some(tx), Some(ty)) if tx.is_finite() && ty.is_finite() => (tx, ty),
            _ => {
                return Err(self.error(
                    PdfSelectableTextErrorKind::InvalidState,
                    Some(operation_index),
                    self.graphics.font_name.as_deref(),
                    "Td/TD operands must be finite numbers",
                ));
            }
        };
        if set_leading {
            self.graphics.leading = -ty;
        }
        let text = self.text.as_mut().expect("checked text state");
        text.line_matrix = text.line_matrix.translated(tx, ty);
        text.matrix = text.line_matrix;
        Ok(())
    }

    fn next_text_line(
        &mut self,
        operation_index: usize,
        operator: &str,
    ) -> Result<(), PdfSelectableTextError> {
        self.require_text(operation_index, operator)?;
        let leading = self.graphics.leading;
        let text = self.text.as_mut().expect("checked text state");
        text.line_matrix = text.line_matrix.translated(0.0, -leading);
        text.matrix = text.line_matrix;
        Ok(())
    }

    fn set_number(
        &mut self,
        operation_index: usize,
        operands: &[Object],
        operator: &str,
        assign: impl FnOnce(&mut GraphicsTextState, f64),
    ) -> Result<(), PdfSelectableTextError> {
        self.require_text(operation_index, operator)?;
        let [operand] = operands else {
            return Err(self.error(
                PdfSelectableTextErrorKind::MalformedTextOperand,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                format!("{operator} expects one numeric operand"),
            ));
        };
        let value = object_number(operand)
            .filter(|value| value.is_finite())
            .ok_or_else(|| {
                self.error(
                    PdfSelectableTextErrorKind::InvalidState,
                    Some(operation_index),
                    self.graphics.font_name.as_deref(),
                    format!("{operator} operand must be finite"),
                )
            })?;
        if operator == "Tz" && value <= 0.0 {
            return Err(self.error(
                PdfSelectableTextErrorKind::InvalidState,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                "Tz horizontal scaling must be greater than zero",
            ));
        }
        assign(&mut self.graphics, value);
        Ok(())
    }

    fn set_rendering_mode(
        &mut self,
        operation_index: usize,
        operands: &[Object],
    ) -> Result<(), PdfSelectableTextError> {
        self.require_text(operation_index, "Tr")?;
        let [operand] = operands else {
            return Err(self.error(
                PdfSelectableTextErrorKind::MalformedTextOperand,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                "Tr expects one integer operand",
            ));
        };
        let value = object_number(operand).ok_or_else(|| {
            self.error(
                PdfSelectableTextErrorKind::MalformedTextOperand,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                "Tr operand is not numeric",
            )
        })?;
        if !value.is_finite() || value.fract() != 0.0 || !(0.0..=7.0).contains(&value) {
            return Err(self.error(
                PdfSelectableTextErrorKind::InvalidState,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                format!("Tr rendering mode {value} is outside integer range 0..=7"),
            ));
        }
        self.graphics.rendering_mode = value as u8;
        Ok(())
    }

    fn show_tj(
        &mut self,
        operation_index: usize,
        operands: &[Object],
    ) -> Result<(), PdfSelectableTextError> {
        self.require_text(operation_index, "Tj")?;
        let [Object::String(bytes, _)] = operands else {
            return Err(self.error(
                PdfSelectableTextErrorKind::MalformedTextOperand,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                "Tj expects one string operand",
            ));
        };
        self.push_run(operation_index, 0, bytes, Vec::new(), Vec::new())
    }

    fn show_single_quote(
        &mut self,
        operation_index: usize,
        operands: &[Object],
    ) -> Result<(), PdfSelectableTextError> {
        self.require_text(operation_index, "'")?;
        let [Object::String(bytes, _)] = operands else {
            return Err(self.error(
                PdfSelectableTextErrorKind::MalformedTextOperand,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                "' expects one string operand",
            ));
        };
        self.next_text_line(operation_index, "'")?;
        self.push_run(operation_index, 0, bytes, Vec::new(), Vec::new())
    }

    fn show_double_quote(
        &mut self,
        operation_index: usize,
        operands: &[Object],
    ) -> Result<(), PdfSelectableTextError> {
        self.require_text(operation_index, "\"")?;
        let [word_spacing, character_spacing, Object::String(bytes, _)] = operands else {
            return Err(self.error(
                PdfSelectableTextErrorKind::MalformedTextOperand,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                "\" expects word spacing, character spacing, and one string operand",
            ));
        };
        let word_spacing = object_number(word_spacing)
            .filter(|value| value.is_finite())
            .ok_or_else(|| {
                self.error(
                    PdfSelectableTextErrorKind::InvalidState,
                    Some(operation_index),
                    self.graphics.font_name.as_deref(),
                    "\" word spacing must be finite",
                )
            })?;
        let character_spacing = object_number(character_spacing)
            .filter(|value| value.is_finite())
            .ok_or_else(|| {
                self.error(
                    PdfSelectableTextErrorKind::InvalidState,
                    Some(operation_index),
                    self.graphics.font_name.as_deref(),
                    "\" character spacing must be finite",
                )
            })?;
        self.graphics.word_spacing = word_spacing;
        self.graphics.character_spacing = character_spacing;
        self.next_text_line(operation_index, "\"")?;
        self.push_run(operation_index, 2, bytes, Vec::new(), Vec::new())
    }

    fn show_tj_array(
        &mut self,
        operation_index: usize,
        operands: &[Object],
    ) -> Result<(), PdfSelectableTextError> {
        self.require_text(operation_index, "TJ")?;
        let [Object::Array(items)] = operands else {
            return Err(self.error(
                PdfSelectableTextErrorKind::MalformedTextOperand,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                "TJ expects one array operand",
            ));
        };
        if items.is_empty() || items.len() > MAX_TJ_OPERANDS {
            return Err(self.error(
                if items.len() > MAX_TJ_OPERANDS {
                    PdfSelectableTextErrorKind::LimitExceeded
                } else {
                    PdfSelectableTextErrorKind::MalformedTextOperand
                },
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                format!(
                    "TJ array has {} operands; admitted range is 1..={MAX_TJ_OPERANDS}",
                    items.len()
                ),
            ));
        }
        let Some(prospective_operand_count) =
            checked_page_tj_operand_count(self.tj_operand_count, items.len())
        else {
            return Err(self.error(
                PdfSelectableTextErrorKind::LimitExceeded,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                format!(
                    "page exceeds the {MAX_PAGE_TJ_OPERANDS}-operand selectable-text limit across TJ arrays"
                ),
            ));
        };
        self.tj_operand_count = prospective_operand_count;
        let mut adjustments = Vec::new();
        let mut last_run = None;
        for (operand_index, item) in items.iter().enumerate() {
            match item {
                Object::String(bytes, _) => {
                    self.push_run(
                        operation_index,
                        operand_index,
                        bytes,
                        std::mem::take(&mut adjustments),
                        Vec::new(),
                    )?;
                    last_run = self.runs.len().checked_sub(1);
                }
                Object::Integer(_) | Object::Real(_) => {
                    let value = object_number(item)
                        .filter(|value| value.is_finite())
                        .ok_or_else(|| {
                            self.error(
                                PdfSelectableTextErrorKind::InvalidState,
                                Some(operation_index),
                                self.graphics.font_name.as_deref(),
                                format!("TJ numeric operand {operand_index} is non-finite"),
                            )
                        })?;
                    adjustments.push(value);
                }
                _ => {
                    return Err(self.error(
                        PdfSelectableTextErrorKind::MalformedTextOperand,
                        Some(operation_index),
                        self.graphics.font_name.as_deref(),
                        format!("TJ operand {operand_index} is neither a string nor a number"),
                    ));
                }
            }
        }
        let last_run = last_run.ok_or_else(|| {
            self.error(
                PdfSelectableTextErrorKind::MalformedTextOperand,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                "TJ array contains no string operand",
            )
        })?;
        if !adjustments.is_empty() {
            let font_name = self.graphics.font_name.clone().ok_or_else(|| {
                self.error(
                    PdfSelectableTextErrorKind::InvalidState,
                    Some(operation_index),
                    None,
                    "TJ trailing adjustment has no selected font",
                )
            })?;
            self.apply_tj_adjustments(operation_index, &font_name, &adjustments)?;
            self.runs[last_run].tj_adjustments_after = adjustments;
            self.runs[last_run].identity_sha256 = run_identity(&self.runs[last_run]);
        }
        Ok(())
    }

    fn apply_tj_adjustments(
        &mut self,
        operation_index: usize,
        font_name: &[u8],
        adjustments: &[f64],
    ) -> Result<(), PdfSelectableTextError> {
        let scale = self.graphics.horizontal_scaling_percent / 100.0;
        for adjustment in adjustments {
            let amount = -*adjustment * self.graphics.font_size / 1000.0 * scale;
            self.advance_text_matrix(operation_index, font_name, amount, "TJ adjustment")?;
        }
        Ok(())
    }

    fn advance_text_matrix(
        &mut self,
        operation_index: usize,
        font_name: &[u8],
        amount: f64,
        role: &str,
    ) -> Result<(), PdfSelectableTextError> {
        if !amount.is_finite() {
            return Err(self.error(
                PdfSelectableTextErrorKind::InvalidState,
                Some(operation_index),
                Some(font_name),
                format!("{role} produced a non-finite text displacement"),
            ));
        }
        let finite = {
            let text = self.text.as_mut().expect("checked text state");
            text.matrix = text.matrix.translated(amount, 0.0);
            text.matrix.is_finite()
        };
        if finite {
            Ok(())
        } else {
            Err(self.error(
                PdfSelectableTextErrorKind::InvalidState,
                Some(operation_index),
                Some(font_name),
                format!("text matrix became non-finite after {role}"),
            ))
        }
    }

    fn push_run(
        &mut self,
        operation_index: usize,
        operand_index: usize,
        bytes: &[u8],
        adjustments_before: Vec<f64>,
        adjustments_after: Vec<f64>,
    ) -> Result<(), PdfSelectableTextError> {
        if self.runs.len() >= MAX_TEXT_RUNS {
            return Err(self.error(
                PdfSelectableTextErrorKind::LimitExceeded,
                Some(operation_index),
                self.graphics.font_name.as_deref(),
                format!("page exceeds the {MAX_TEXT_RUNS}-run selectable-text limit"),
            ));
        }
        let font_name = self.graphics.font_name.clone().ok_or_else(|| {
            self.error(
                PdfSelectableTextErrorKind::InvalidState,
                Some(operation_index),
                None,
                "text string has no selected Tf font",
            )
        })?;
        let font_resource = resource_name(&font_name).map_err(|detail| {
            self.error(
                PdfSelectableTextErrorKind::MissingResource,
                Some(operation_index),
                None,
                detail,
            )
        })?;
        let font = self.fonts.get(&font_name).ok_or_else(|| {
            self.error(
                PdfSelectableTextErrorKind::MissingResource,
                Some(operation_index),
                Some(&font_name),
                "selected font has no validated decoding map",
            )
        })?;
        let code_count = match font.code_width_bytes {
            1 => bytes.len(),
            2 if bytes.len().is_multiple_of(2) => bytes.len() / 2,
            2 => {
                return Err(self.error(
                    PdfSelectableTextErrorKind::MalformedTextOperand,
                    Some(operation_index),
                    Some(&font_name),
                    format!("Identity-H string has odd byte length {}", bytes.len()),
                ));
            }
            _ => {
                return Err(self.error(
                    PdfSelectableTextErrorKind::InvalidState,
                    Some(operation_index),
                    Some(&font_name),
                    format!(
                        "validated font has unsupported {}-byte source codes",
                        font.code_width_bytes
                    ),
                ));
            }
        };
        let prospective_code_count = self.code_count.checked_add(code_count).ok_or_else(|| {
            self.error(
                PdfSelectableTextErrorKind::LimitExceeded,
                Some(operation_index),
                Some(&font_name),
                "page source-code count overflow",
            )
        })?;
        if prospective_code_count > MAX_TEXT_CODES {
            return Err(self.error(
                PdfSelectableTextErrorKind::LimitExceeded,
                Some(operation_index),
                Some(&font_name),
                format!("page exceeds the {MAX_TEXT_CODES}-code selectable-text limit"),
            ));
        }
        let mut codes = Vec::with_capacity(code_count);
        match font.code_width_bytes {
            1 => codes.extend(bytes.iter().copied().map(u16::from)),
            2 => codes.extend(
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|chunk| u16::from_be_bytes(*chunk)),
            ),
            _ => unreachable!("validated source-code width"),
        }
        let mut glyph_advances = Vec::with_capacity(codes.len());
        let mut run_glyph_name_bytes = 0usize;
        let mut run_unicode_bytes = 0usize;
        let mut run_unicode_scalar_count = 0usize;
        let font_encoding = font.encoding;
        let code_width_bytes = font.code_width_bytes;
        let font_mapping_sha256 = font.mapping_sha256;
        for code in &codes {
            let (glyph_name, mapped) = match font_encoding {
                PdfSelectableTextEncodingV2::Type0IdentityH => {
                    let mapped = font.mappings.get(code).ok_or_else(|| {
                        self.error(
                            PdfSelectableTextErrorKind::MissingMapping,
                            Some(operation_index),
                            Some(&font_name),
                            format!("CID <{code:04X}> has no complete ToUnicode mapping"),
                        )
                    })?;
                    (None, Some(mapped))
                }
                PdfSelectableTextEncodingV2::Type1cWinAnsiDifferences => {
                    let index = usize::from(*code);
                    let glyph_name = font
                        .glyph_names
                        .get(index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            self.error(
                                PdfSelectableTextErrorKind::MissingMapping,
                                Some(operation_index),
                                Some(&font_name),
                                format!(
                                    "Type1C source code <{code:02X}> has no admitted glyph name"
                                ),
                            )
                        })?;
                    if !font.available_glyphs.get(index).copied().unwrap_or(false) {
                        return Err(self.error(
                            PdfSelectableTextErrorKind::MissingMapping,
                            Some(operation_index),
                            Some(&font_name),
                            format!(
                                "Type1C source code <{code:02X}> names missing CFF glyph /{glyph_name}"
                            ),
                        ));
                    }
                    (Some(glyph_name), font.mappings.get(code))
                }
            };
            run_glyph_name_bytes = run_glyph_name_bytes
                .checked_add(glyph_name.map_or(0, String::len))
                .ok_or_else(|| {
                    self.error(
                        PdfSelectableTextErrorKind::LimitExceeded,
                        Some(operation_index),
                        Some(&font_name),
                        "glyph-name byte count overflow",
                    )
                })?;
            if checked_output_string_growth(
                self.output_string_bytes,
                font_resource.len(),
                run_glyph_name_bytes,
                run_unicode_bytes,
                MAX_OUTPUT_STRING_BYTES,
            )
            .is_none()
            {
                return Err(self.error(
                    PdfSelectableTextErrorKind::LimitExceeded,
                    Some(operation_index),
                    Some(&font_name),
                    format!(
                        "page selectable-text owned string payload exceeds {MAX_OUTPUT_STRING_BYTES} bytes"
                    ),
                ));
            }
            if let Some(mapped) = mapped {
                run_unicode_bytes =
                    run_unicode_bytes.checked_add(mapped.len()).ok_or_else(|| {
                        self.error(
                            PdfSelectableTextErrorKind::LimitExceeded,
                            Some(operation_index),
                            Some(&font_name),
                            "decoded Unicode byte count overflow",
                        )
                    })?;
                if checked_output_string_growth(
                    self.output_string_bytes,
                    font_resource.len(),
                    run_glyph_name_bytes,
                    run_unicode_bytes,
                    MAX_OUTPUT_STRING_BYTES,
                )
                .is_none()
                {
                    return Err(self.error(
                        PdfSelectableTextErrorKind::LimitExceeded,
                        Some(operation_index),
                        Some(&font_name),
                        format!(
                            "page selectable-text owned string payload exceeds {MAX_OUTPUT_STRING_BYTES} bytes"
                        ),
                    ));
                }
                run_unicode_scalar_count = run_unicode_scalar_count
                    .checked_add(mapped.chars().count())
                    .ok_or_else(|| {
                        self.error(
                            PdfSelectableTextErrorKind::LimitExceeded,
                            Some(operation_index),
                            Some(&font_name),
                            "decoded Unicode scalar count overflow",
                        )
                    })?;
                let incremental_scalar_count = self
                    .unicode_scalar_count
                    .checked_add(run_unicode_scalar_count);
                if !matches!(
                    incremental_scalar_count,
                    Some(count) if count <= MAX_DECODED_UNICODE_SCALARS
                ) {
                    return Err(self.error(
                        PdfSelectableTextErrorKind::LimitExceeded,
                        Some(operation_index),
                        Some(&font_name),
                        format!(
                            "page exceeds the {MAX_DECODED_UNICODE_SCALARS}-scalar Unicode limit"
                        ),
                    ));
                }
            }
            let width = font.widths.get(*code).ok_or_else(|| {
                self.error(
                    PdfSelectableTextErrorKind::MissingMapping,
                    Some(operation_index),
                    Some(&font_name),
                    format!("source code <{code:04X}> has no declared PDF width"),
                )
            })?;
            let word_spacing = if font_encoding
                == PdfSelectableTextEncodingV2::Type1cWinAnsiDifferences
                && *code == u16::from(b' ')
            {
                self.graphics.word_spacing
            } else {
                0.0
            };
            let advance = (width * self.graphics.font_size / 1000.0
                + self.graphics.character_spacing
                + word_spacing)
                * self.graphics.horizontal_scaling_percent
                / 100.0;
            if !advance.is_finite() {
                return Err(self.error(
                    PdfSelectableTextErrorKind::InvalidState,
                    Some(operation_index),
                    Some(&font_name),
                    format!("source code <{code:04X}> produced a non-finite text advance"),
                ));
            }
            glyph_advances.push(advance);
        }
        let prospective_unicode_scalar_count = self
            .unicode_scalar_count
            .checked_add(run_unicode_scalar_count)
            .filter(|count| *count <= MAX_DECODED_UNICODE_SCALARS)
            .ok_or_else(|| {
                self.error(
                    PdfSelectableTextErrorKind::LimitExceeded,
                    Some(operation_index),
                    Some(&font_name),
                    format!("page exceeds the {MAX_DECODED_UNICODE_SCALARS}-scalar Unicode limit"),
                )
            })?;
        let prospective_output_string_bytes = checked_output_string_growth(
            self.output_string_bytes,
            font_resource.len(),
            run_glyph_name_bytes,
            run_unicode_bytes,
            MAX_OUTPUT_STRING_BYTES,
        )
        .ok_or_else(|| {
            self.error(
                PdfSelectableTextErrorKind::LimitExceeded,
                Some(operation_index),
                Some(&font_name),
                format!(
                    "page selectable-text owned string payload exceeds {MAX_OUTPUT_STRING_BYTES} bytes"
                ),
            )
        })?;

        // Materialize only after all mappings, widths, scalar counts, and owned
        // string bytes for the complete run have passed their page budgets.
        let mut glyph_names = Vec::with_capacity(codes.len());
        let mut unicode_by_code = Vec::with_capacity(codes.len());
        let mut unicode = String::with_capacity(run_unicode_bytes);
        for code in &codes {
            let (glyph_name, mapped) = match font_encoding {
                PdfSelectableTextEncodingV2::Type0IdentityH => (None, font.mappings.get(code)),
                PdfSelectableTextEncodingV2::Type1cWinAnsiDifferences => {
                    let index = usize::from(*code);
                    (
                        font.glyph_names.get(index).and_then(Option::as_ref),
                        font.mappings.get(code),
                    )
                }
            };
            if let Some(mapped) = mapped {
                unicode.push_str(mapped);
            }
            glyph_names.push(glyph_name.cloned());
            unicode_by_code.push(mapped.cloned());
        }
        let unicode_complete = unicode_by_code.iter().all(Option::is_some);
        self.apply_tj_adjustments(operation_index, &font_name, &adjustments_before)?;
        let text = self.text.expect("checked text state");
        let position = PdfTextPositionV1 {
            graphics_matrix: self.graphics.ctm.0,
            text_matrix: text.matrix.0,
            line_matrix: text.line_matrix.0,
            font_size: self.graphics.font_size,
            character_spacing: self.graphics.character_spacing,
            word_spacing: self.graphics.word_spacing,
            horizontal_scaling_percent: self.graphics.horizontal_scaling_percent,
            leading: self.graphics.leading,
            rise: self.graphics.rise,
            rendering_mode: self.graphics.rendering_mode,
        };
        for advance in glyph_advances {
            self.advance_text_matrix(operation_index, &font_name, advance, "glyph advance")?;
        }
        self.apply_tj_adjustments(operation_index, &font_name, &adjustments_after)?;
        self.code_count = prospective_code_count;
        self.unicode_scalar_count = prospective_unicode_scalar_count;
        self.output_string_bytes = prospective_output_string_bytes;
        let mut run = PdfSelectableTextRunV2 {
            run_index: self.runs.len(),
            identity_sha256: [0; 32],
            operation_index,
            operand_index,
            font_resource,
            font_encoding,
            code_width_bytes,
            font_mapping_sha256,
            source_codes: codes,
            glyph_names,
            unicode_by_code,
            unicode_complete,
            unicode,
            position,
            tj_adjustments_before: adjustments_before,
            tj_adjustments_after: adjustments_after,
        };
        run.identity_sha256 = run_identity(&run);
        self.runs.push(run);
        Ok(())
    }

    fn validate_xobject(
        &self,
        operation_index: usize,
        operands: &[Object],
    ) -> Result<(), PdfSelectableTextError> {
        let [Object::Name(name)] = operands else {
            return Err(self.error(
                PdfSelectableTextErrorKind::MissingResource,
                Some(operation_index),
                None,
                "Do expects one XObject resource name",
            ));
        };
        let xobjects = effective_xobjects(self.doc, self.page_id).map_err(|detail| {
            self.error(
                PdfSelectableTextErrorKind::MissingResource,
                Some(operation_index),
                None,
                detail,
            )
        })?;
        let id = xobjects
            .iter()
            .find_map(|(candidate, id)| (candidate == name).then_some(*id))
            .ok_or_else(|| {
                self.error(
                    PdfSelectableTextErrorKind::MissingResource,
                    Some(operation_index),
                    None,
                    format!(
                        "content references missing XObject /{}",
                        String::from_utf8_lossy(name)
                    ),
                )
            })?;
        let stream = self
            .doc
            .get_object(id)
            .and_then(Object::as_stream)
            .map_err(|error| {
                self.error(
                    PdfSelectableTextErrorKind::MissingResource,
                    Some(operation_index),
                    None,
                    format!(
                        "XObject /{} is not a stream: {error}",
                        String::from_utf8_lossy(name)
                    ),
                )
            })?;
        let subtype = stream
            .dict
            .get(b"Subtype")
            .and_then(Object::as_name)
            .map_err(|error| {
                self.error(
                    PdfSelectableTextErrorKind::MissingResource,
                    Some(operation_index),
                    None,
                    format!(
                        "XObject /{} has no subtype: {error}",
                        String::from_utf8_lossy(name)
                    ),
                )
            })?;
        if subtype == b"Image" {
            Ok(())
        } else {
            Err(self.error(
                PdfSelectableTextErrorKind::UnsupportedTextOperator,
                Some(operation_index),
                None,
                format!(
                    "XObject /{} subtype /{} may contain nested text and is not admitted",
                    String::from_utf8_lossy(name),
                    String::from_utf8_lossy(subtype)
                ),
            ))
        }
    }
}

pub(super) fn extract_page(
    doc: &Document,
    page_id: ObjectId,
    page_index: usize,
    source_bytes: &[u8],
    content: &Content<Vec<Operation>>,
) -> Result<PdfSelectableTextPageV2, PdfSelectableTextError> {
    let source_sha256 = Sha256::digest(source_bytes).into();
    validate_page_annotations(doc, page_id).map_err(|detail| PdfSelectableTextError {
        kind: PdfSelectableTextErrorKind::UnsupportedTextOperator,
        page_index,
        operation_index: None,
        font_resource: None,
        detail,
    })?;
    let runs = Extractor::new(doc, page_id, page_index).process(content)?;
    let mut result = PdfSelectableTextPageV2 {
        schema: PDF_SELECTABLE_TEXT_SCHEMA_V2,
        page_index,
        source_sha256,
        identity_sha256: [0; 32],
        runs,
    };
    result.identity_sha256 = page_identity(&result);
    Ok(result)
}

fn validate_page_annotations(doc: &Document, page_id: ObjectId) -> Result<(), String> {
    let page = doc
        .get_object(page_id)
        .and_then(Object::as_dict)
        .map_err(|error| format!("page object is not a dictionary: {error}"))?;
    let Ok(annotations) = page.get(b"Annots") else {
        return Ok(());
    };
    let (_, annotations) = doc
        .dereference(annotations)
        .map_err(|error| format!("dereference page /Annots: {error}"))?;
    let annotations = annotations
        .as_array()
        .map_err(|error| format!("page /Annots is not an array: {error}"))?;
    if annotations.len() > MAX_PAGE_ANNOTATIONS {
        return Err(format!(
            "page has {} annotations, exceeding the {MAX_PAGE_ANNOTATIONS}-annotation selectable-text limit",
            annotations.len()
        ));
    }
    for (index, annotation) in annotations.iter().enumerate() {
        let (_, annotation) = doc
            .dereference(annotation)
            .map_err(|error| format!("dereference annotation {index}: {error}"))?;
        let annotation = annotation
            .as_dict()
            .map_err(|error| format!("annotation {index} is not a dictionary: {error}"))?;
        require_name(annotation, b"Type", b"Annot")
            .map_err(|detail| format!("annotation {index}: {detail}"))?;
        require_name(annotation, b"Subtype", b"Link").map_err(|_| {
            format!(
                "annotation {index} is not an exact /Link annotation and may have viewer-generated visible content"
            )
        })?;
        if annotation.has(b"AP") {
            return Err(format!(
                "annotation {index} declares an /AP appearance that may contain uninspected text"
            ));
        }
    }
    Ok(())
}

fn load_font_map(
    doc: &Document,
    page_id: ObjectId,
    page_index: usize,
    operation_index: usize,
    font_name: &[u8],
    font_resource: &str,
) -> Result<FontMap, PdfSelectableTextError> {
    let error = |kind, detail: String| PdfSelectableTextError {
        kind,
        page_index,
        operation_index: Some(operation_index),
        font_resource: Some(font_resource.to_owned()),
        detail,
    };
    let fonts = doc.get_page_fonts(page_id).map_err(|cause| {
        error(
            PdfSelectableTextErrorKind::MissingResource,
            format!("read page font resources: {cause}"),
        )
    })?;
    let font = fonts.get(font_name).ok_or_else(|| {
        error(
            PdfSelectableTextErrorKind::MissingResource,
            format!("page has no font resource /{font_resource}"),
        )
    })?;
    require_name(font, b"Type", b"Font")
        .map_err(|detail| error(PdfSelectableTextErrorKind::UnsupportedFont, detail))?;
    let subtype = font
        .get(b"Subtype")
        .and_then(Object::as_name)
        .map_err(|cause| {
            error(
                PdfSelectableTextErrorKind::UnsupportedFont,
                format!("font has no valid /Subtype name: {cause}"),
            )
        })?;
    match subtype {
        b"Type0" => load_type0_font_map(doc, font, error),
        b"Type1" => load_type1c_font_map(doc, font, error),
        other => Err(error(
            PdfSelectableTextErrorKind::UnsupportedFont,
            format!(
                "font subtype /{} is outside the admitted selectable-text subset",
                String::from_utf8_lossy(other)
            ),
        )),
    }
}

fn load_type0_font_map(
    doc: &Document,
    type0: &Dictionary,
    error: impl Fn(PdfSelectableTextErrorKind, String) -> PdfSelectableTextError,
) -> Result<FontMap, PdfSelectableTextError> {
    require_name(type0, b"Encoding", b"Identity-H")
        .map_err(|detail| error(PdfSelectableTextErrorKind::UnsupportedFont, detail))?;
    let descendants = type0
        .get(b"DescendantFonts")
        .and_then(Object::as_array)
        .map_err(|cause| {
            error(
                PdfSelectableTextErrorKind::UnsupportedFont,
                format!("Type0 /DescendantFonts is invalid: {cause}"),
            )
        })?;
    if descendants.len() != 1 {
        return Err(error(
            PdfSelectableTextErrorKind::UnsupportedFont,
            format!(
                "Type0 font has {} descendants; exactly one is required",
                descendants.len()
            ),
        ));
    }
    let descendant = dereference_dict(doc, &descendants[0], "CIDFont descendant")
        .map_err(|detail| error(PdfSelectableTextErrorKind::UnsupportedFont, detail))?;
    require_name(descendant, b"Type", b"Font")
        .map_err(|detail| error(PdfSelectableTextErrorKind::UnsupportedFont, detail))?;
    require_name(descendant, b"Subtype", b"CIDFontType2")
        .map_err(|detail| error(PdfSelectableTextErrorKind::UnsupportedFont, detail))?;
    require_name(descendant, b"CIDToGIDMap", b"Identity")
        .map_err(|detail| error(PdfSelectableTextErrorKind::UnsupportedFont, detail))?;

    let to_unicode = type0.get(b"ToUnicode").map_err(|cause| {
        error(
            PdfSelectableTextErrorKind::MissingResource,
            format!("Type0 font has no /ToUnicode stream: {cause}"),
        )
    })?;
    let (_, to_unicode) = doc.dereference(to_unicode).map_err(|cause| {
        error(
            PdfSelectableTextErrorKind::MissingResource,
            format!("dereference /ToUnicode: {cause}"),
        )
    })?;
    let stream = to_unicode.as_stream().map_err(|cause| {
        error(
            PdfSelectableTextErrorKind::MissingResource,
            format!("/ToUnicode is not a stream: {cause}"),
        )
    })?;
    let decoded = decode_cmap_stream(stream).map_err(|(kind, detail)| error(kind, detail))?;
    let mapping_sha256 = Sha256::digest(&decoded).into();
    let mappings = parse_cmap(&decoded).map_err(|(kind, detail)| error(kind, detail))?;
    let widths = parse_cid_widths(doc, descendant)
        .map_err(|detail| error(PdfSelectableTextErrorKind::UnsupportedFont, detail))?;
    Ok(FontMap {
        encoding: PdfSelectableTextEncodingV2::Type0IdentityH,
        code_width_bytes: 2,
        mapping_sha256,
        mappings,
        glyph_names: Vec::new(),
        available_glyphs: Vec::new(),
        widths: FontWidths::Type0(widths),
    })
}

fn load_type1c_font_map(
    doc: &Document,
    type1: &Dictionary,
    error: impl Fn(PdfSelectableTextErrorKind, String) -> PdfSelectableTextError,
) -> Result<FontMap, PdfSelectableTextError> {
    if type1.has(b"ToUnicode") {
        return Err(error(
            PdfSelectableTextErrorKind::UnsupportedFont,
            "Type1C /ToUnicode is outside the exact LilyPond subset; refusing instead of ignoring it"
                .to_owned(),
        ));
    }
    let program = load_simple_type1c_program(doc, type1)
        .map_err(|detail| error(PdfSelectableTextErrorKind::UnsupportedFont, detail))?;
    let table = cff::Table::parse(&program.bytes).ok_or_else(|| {
        error(
            PdfSelectableTextErrorKind::UnsupportedFont,
            "FontFile3 is not a valid bounded Type1C program".to_owned(),
        )
    })?;
    let available_glyphs = program
        .glyph_names
        .iter()
        .map(|name| {
            name.as_deref()
                .and_then(|name| table.glyph_index_by_name(name))
                .is_some()
        })
        .collect::<Vec<_>>();
    let mappings = program
        .glyph_names
        .iter()
        .enumerate()
        .filter_map(|(code, name)| {
            name.as_deref()
                .and_then(glyph_name_to_unicode)
                .map(|unicode| (code as u16, unicode))
        })
        .collect::<BTreeMap<_, _>>();
    let mapping_sha256 = simple_type1_encoding_identity(&program.glyph_names);
    let widths = parse_simple_type1_widths(type1)
        .map_err(|detail| error(PdfSelectableTextErrorKind::UnsupportedFont, detail))?;
    Ok(FontMap {
        encoding: PdfSelectableTextEncodingV2::Type1cWinAnsiDifferences,
        code_width_bytes: 1,
        mapping_sha256,
        mappings,
        glyph_names: program.glyph_names,
        available_glyphs,
        widths: FontWidths::Type1c(widths),
    })
}

fn simple_type1_encoding_identity(glyph_names: &[Option<String>]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash_field(
        &mut hash,
        b"domain",
        b"franken_ocr.pdf.type1c_win_ansi_differences.v2",
    );
    for (code, glyph_name) in glyph_names.iter().enumerate() {
        hash_field(&mut hash, b"code", &(code as u16).to_be_bytes());
        match glyph_name {
            Some(glyph_name) => {
                hash_field(&mut hash, b"glyph_name_present", &[1]);
                hash_field(&mut hash, b"glyph_name", glyph_name.as_bytes());
            }
            None => hash_field(&mut hash, b"glyph_name_present", &[0]),
        }
    }
    hash.finalize().into()
}

fn glyph_name_to_unicode(glyph_name: &str) -> Option<String> {
    let base_name = glyph_name
        .split_once('.')
        .map_or(glyph_name, |(base, _)| base);
    if base_name.is_empty() || base_name == ".notdef" {
        return None;
    }
    if base_name.contains('_') {
        let mut mapped = String::new();
        for component in base_name.split('_') {
            mapped.push_str(&glyph_component_to_unicode(component)?);
        }
        return (!mapped.is_empty()).then_some(mapped);
    }
    glyph_component_to_unicode(base_name)
}

fn glyph_component_to_unicode(glyph_name: &str) -> Option<String> {
    for code in 32u8..=255 {
        if win_ansi_glyph_name(code) == Some(glyph_name) {
            return win_ansi_code_to_unicode(code).map(|character| character.to_string());
        }
    }
    // Pinned Adobe Glyph List 2.0 legacy-name subset. A missing entry remains
    // opaque; production names are handled separately below.
    let common_agl = match glyph_name {
        "Delta" => '∆',
        "Lslash" => 'Ł',
        "Omega" => 'Ω',
        "dotlessi" => 'ı',
        "fi" => 'ﬁ',
        "fl" => 'ﬂ',
        "fraction" => '⁄',
        "lslash" => 'ł',
        "minus" => '−',
        "partialdiff" => '∂',
        "product" => '∏',
        "radical" => '√',
        "summation" => '∑',
        _ => return unicode_from_agl_name(glyph_name),
    };
    Some(common_agl.to_string())
}

fn unicode_from_agl_name(glyph_name: &str) -> Option<String> {
    if let Some(hex) = glyph_name.strip_prefix("uni") {
        if hex.is_empty()
            || hex.len() % 4 != 0
            || !hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'A'..=b'F'))
        {
            return None;
        }
        let mut mapped = String::new();
        for chunk in hex.as_bytes().as_chunks::<4>().0 {
            let hex = std::str::from_utf8(chunk).ok()?;
            mapped.push(char::from_u32(u32::from_str_radix(hex, 16).ok()?)?);
        }
        return (!mapped.is_empty()).then_some(mapped);
    }
    let hex = glyph_name.strip_prefix('u')?;
    if !(4..=6).contains(&hex.len())
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'A'..=b'F'))
    {
        return None;
    }
    char::from_u32(u32::from_str_radix(hex, 16).ok()?).map(|character| character.to_string())
}

fn win_ansi_code_to_unicode(code: u8) -> Option<char> {
    match code {
        32..=126 | 160..=255 => char::from_u32(u32::from(code)),
        127 | 129 | 141 | 143 | 144 | 149 | 157 => Some('•'),
        128 => Some('€'),
        130 => Some('‚'),
        131 => Some('ƒ'),
        132 => Some('„'),
        133 => Some('…'),
        134 => Some('†'),
        135 => Some('‡'),
        136 => Some('ˆ'),
        137 => Some('‰'),
        138 => Some('Š'),
        139 => Some('‹'),
        140 => Some('Œ'),
        142 => Some('Ž'),
        145 => Some('‘'),
        146 => Some('’'),
        147 => Some('“'),
        148 => Some('”'),
        150 => Some('–'),
        151 => Some('—'),
        152 => Some('˜'),
        153 => Some('™'),
        154 => Some('š'),
        155 => Some('›'),
        156 => Some('œ'),
        158 => Some('ž'),
        159 => Some('Ÿ'),
        _ => None,
    }
}

fn decode_cmap_stream(stream: &Stream) -> Result<Vec<u8>, (PdfSelectableTextErrorKind, String)> {
    if stream.dict.has(b"UseCMap") {
        return Err((
            PdfSelectableTextErrorKind::UnsupportedCMap,
            "ToUnicode stream dictionary /UseCMap inheritance is not admitted".to_owned(),
        ));
    }
    if stream.content.len() > MAX_CMAP_ENCODED_BYTES {
        return Err((
            PdfSelectableTextErrorKind::LimitExceeded,
            format!(
                "ToUnicode stream has {} encoded bytes, exceeding {MAX_CMAP_ENCODED_BYTES}",
                stream.content.len()
            ),
        ));
    }
    let filters = stream_filters(&stream.dict)
        .map_err(|detail| (PdfSelectableTextErrorKind::UnsupportedCMap, detail))?;
    let decoded = match filters.as_slice() {
        [] => {
            validate_cmap_decode_parameters(&stream.dict, false)?;
            stream.content.clone()
        }
        [filter] if filter == "FlateDecode" => {
            validate_cmap_decode_parameters(&stream.dict, true)?;
            bounded_inflate(&stream.content, MAX_CMAP_DECODED_BYTES)
                .map_err(|detail| (PdfSelectableTextErrorKind::MalformedCMap, detail))?
                .ok_or_else(|| {
                    (
                        PdfSelectableTextErrorKind::MalformedCMap,
                        "ToUnicode sole-Flate stream is not a valid bounded zlib stream".to_owned(),
                    )
                })?
        }
        _ => {
            return Err((
                PdfSelectableTextErrorKind::UnsupportedCMap,
                format!(
                    "ToUnicode filters {filters:?} are outside the admitted unfiltered/sole-Flate subset"
                ),
            ));
        }
    };
    if decoded.len() as u64 > MAX_CMAP_DECODED_BYTES {
        return Err((
            PdfSelectableTextErrorKind::LimitExceeded,
            format!(
                "ToUnicode stream has {} decoded bytes, exceeding {MAX_CMAP_DECODED_BYTES}",
                decoded.len()
            ),
        ));
    }
    Ok(decoded)
}

fn validate_cmap_decode_parameters(
    dict: &Dictionary,
    has_flate_filter: bool,
) -> Result<(), (PdfSelectableTextErrorKind, String)> {
    match (has_flate_filter, dict.get(b"Filter")) {
        (true, Ok(Object::Name(filter))) if filter == b"FlateDecode" => {}
        (true, _) => {
            return Err((
                PdfSelectableTextErrorKind::UnsupportedCMap,
                "ToUnicode requires /Filter /FlateDecode as one direct name".to_owned(),
            ));
        }
        (false, Ok(_)) => {
            return Err((
                PdfSelectableTextErrorKind::UnsupportedCMap,
                "ToUnicode has an empty or malformed /Filter declaration".to_owned(),
            ));
        }
        (false, Err(_)) => {}
    }
    let decode_parameters = dict.get(b"DecodeParms").ok();
    let abbreviated = dict.get(b"DP").ok();
    if decode_parameters.is_some() && abbreviated.is_some() {
        return Err((
            PdfSelectableTextErrorKind::UnsupportedCMap,
            "ToUnicode declares both /DecodeParms and /DP".to_owned(),
        ));
    }
    let Some(parameters) = decode_parameters.or(abbreviated) else {
        return Ok(());
    };
    if !has_flate_filter {
        return Err((
            PdfSelectableTextErrorKind::UnsupportedCMap,
            "ToUnicode decode parameters require the admitted sole /FlateDecode filter".to_owned(),
        ));
    }
    let parameters = parameters.as_dict().map_err(|cause| {
        (
            PdfSelectableTextErrorKind::UnsupportedCMap,
            format!("ToUnicode FlateDecode parameters are not a dictionary: {cause}"),
        )
    })?;
    for (key, _) in parameters.iter() {
        if key != b"Predictor" {
            return Err((
                PdfSelectableTextErrorKind::UnsupportedCMap,
                format!(
                    "ToUnicode FlateDecode parameter /{} is outside the exact subset",
                    String::from_utf8_lossy(key)
                ),
            ));
        }
    }
    let predictor = match parameters.get(b"Predictor") {
        Ok(value) => value.as_i64().map_err(|cause| {
            (
                PdfSelectableTextErrorKind::UnsupportedCMap,
                format!("ToUnicode FlateDecode /Predictor is not an integer: {cause}"),
            )
        })?,
        Err(_) => 1,
    };
    if predictor != 1 {
        return Err((
            PdfSelectableTextErrorKind::UnsupportedCMap,
            format!(
                "ToUnicode FlateDecode /Predictor {predictor} is outside the exact no-predictor subset"
            ),
        ));
    }
    Ok(())
}

fn parse_cmap(bytes: &[u8]) -> Result<BTreeMap<u16, String>, (PdfSelectableTextErrorKind, String)> {
    let tokens = tokenize_cmap(bytes)?;
    let active_top_level = cmap_top_level_flags(&tokens)?;
    let begin = tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| word_is(token, "begincmap").then_some(index))
        .collect::<Vec<_>>();
    let end = tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| word_is(token, "endcmap").then_some(index))
        .collect::<Vec<_>>();
    if begin.len() != 1 || end.len() != 1 || begin[0] >= end[0] {
        return Err((
            PdfSelectableTextErrorKind::MalformedCMap,
            format!(
                "ToUnicode requires exactly one ordered begincmap/endcmap scope; found {} begin and {} end markers",
                begin.len(),
                end.len()
            ),
        ));
    }
    if !active_top_level[begin[0]] || !active_top_level[end[0]] {
        return Err((
            PdfSelectableTextErrorKind::UnsupportedCMap,
            "begincmap/endcmap must be active at top level in the ToUnicode program".to_owned(),
        ));
    }
    if tokens.iter().any(|token| word_is(token, "usecmap")) {
        return Err((
            PdfSelectableTextErrorKind::UnsupportedCMap,
            "inherited usecmap programs are not admitted".to_owned(),
        ));
    }
    for token in tokens[..begin[0]].iter().chain(tokens[end[0] + 1..].iter()) {
        if is_cmap_section_marker(token)
            || matches!(token, CMapToken::Word(word) if word == "/CMapType")
        {
            return Err((
                PdfSelectableTextErrorKind::MalformedCMap,
                "CMap semantic token occurs outside the sole begincmap/endcmap scope".to_owned(),
            ));
        }
    }
    let tokens = &tokens[begin[0] + 1..end[0]];
    let top_level = &active_top_level[begin[0] + 1..end[0]];
    for (index, token) in tokens.iter().enumerate() {
        if (is_cmap_section_marker(token)
            || matches!(token, CMapToken::Word(word) if word == "/CMapType"))
            && !top_level[index]
        {
            return Err((
                PdfSelectableTextErrorKind::UnsupportedCMap,
                format!(
                    "ToUnicode semantic token {token:?} is not active at top level inside begincmap"
                ),
            ));
        }
    }
    let cmap_type_declarations = tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| {
            (top_level[index] && matches!(token, CMapToken::Word(word) if word == "/CMapType"))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let [cmap_type_index] = cmap_type_declarations.as_slice() else {
        return Err((
            PdfSelectableTextErrorKind::UnsupportedCMap,
            format!(
                "ToUnicode must declare exactly one active top-level /CMapType 2 def; found {} declarations",
                cmap_type_declarations.len()
            ),
        ));
    };
    if !matches!(tokens.get(*cmap_type_index + 1), Some(CMapToken::Word(word)) if word == "2")
        || !matches!(tokens.get(*cmap_type_index + 2), Some(CMapToken::Word(word)) if word == "def")
    {
        return Err((
            PdfSelectableTextErrorKind::UnsupportedCMap,
            "ToUnicode active top-level /CMapType declaration must be exactly /CMapType 2 def"
                .to_owned(),
        ));
    }

    let mut mappings = BTreeMap::new();
    let mut mapping_budget = CMapMappingBudget::default();
    let mut codespace_seen = false;
    let mut range_count = 0usize;
    let mut index = 0usize;
    while index < tokens.len() {
        if word_is(&tokens[index], "begincodespacerange") {
            if codespace_seen {
                return Err((
                    PdfSelectableTextErrorKind::UnsupportedCMap,
                    "multiple codespace sections are not admitted".to_owned(),
                ));
            }
            let count = preceding_count(tokens, index, "codespacerange")?;
            if count != 1 {
                return Err((
                    PdfSelectableTextErrorKind::UnsupportedCMap,
                    format!("Identity-H ToUnicode requires one codespace range, found {count}"),
                ));
            }
            let start = expect_hex(tokens, index + 1, "codespace start")?;
            let end = expect_hex(tokens, index + 2, "codespace end")?;
            if start != [0x00, 0x00] || end != [0xff, 0xff] {
                return Err((
                    PdfSelectableTextErrorKind::UnsupportedCMap,
                    "Identity-H ToUnicode codespace must be <0000> <FFFF>".to_owned(),
                ));
            }
            expect_word(tokens, index + 3, "endcodespacerange")?;
            codespace_seen = true;
            index += 4;
            continue;
        }
        if word_is(&tokens[index], "beginbfchar") {
            if !codespace_seen {
                return Err((
                    PdfSelectableTextErrorKind::MalformedCMap,
                    "bfchar section precedes the required codespace section".to_owned(),
                ));
            }
            let count = preceding_count(tokens, index, "bfchar")?;
            ensure_entry_budget(mappings.len(), count)?;
            let mut cursor = index + 1;
            for _ in 0..count {
                let source = source_code(expect_hex(tokens, cursor, "bfchar source")?)?;
                let encoded_target = expect_hex(tokens, cursor + 1, "bfchar target")?;
                mapping_budget.ensure_target_utf16_bytes(encoded_target.len())?;
                mapping_budget.ensure_utf16_growth(encoded_target.len())?;
                let target = decode_utf16be(encoded_target)?;
                insert_mapping(
                    &mut mappings,
                    source,
                    encoded_target.len(),
                    target,
                    &mut mapping_budget,
                )?;
                cursor += 2;
            }
            expect_word(tokens, cursor, "endbfchar")?;
            index = cursor + 1;
            continue;
        }
        if word_is(&tokens[index], "beginbfrange") {
            if !codespace_seen {
                return Err((
                    PdfSelectableTextErrorKind::MalformedCMap,
                    "bfrange section precedes the required codespace section".to_owned(),
                ));
            }
            let count = preceding_count(tokens, index, "bfrange")?;
            range_count = range_count.checked_add(count).ok_or_else(|| {
                (
                    PdfSelectableTextErrorKind::LimitExceeded,
                    "bfrange count overflow".to_owned(),
                )
            })?;
            if range_count > MAX_CMAP_RANGES {
                return Err((
                    PdfSelectableTextErrorKind::LimitExceeded,
                    format!("ToUnicode exceeds {MAX_CMAP_RANGES} bfrange mappings"),
                ));
            }
            let mut cursor = index + 1;
            for _ in 0..count {
                let first = source_code(expect_hex(tokens, cursor, "bfrange start")?)?;
                let last = source_code(expect_hex(tokens, cursor + 1, "bfrange end")?)?;
                if last < first {
                    return Err((
                        PdfSelectableTextErrorKind::MalformedCMap,
                        format!("bfrange <{first:04X}>..<{last:04X}> is descending"),
                    ));
                }
                let entry_count = usize::from(last - first) + 1;
                ensure_entry_budget(mappings.len(), entry_count)?;
                cursor += 2;
                match tokens.get(cursor) {
                    Some(CMapToken::Hex(base)) => {
                        mapping_budget.ensure_repeated_utf16_growth(base.len(), entry_count)?;
                        for offset in 0..entry_count {
                            let target = increment_be(base, offset).ok_or_else(|| {
                                (
                                    PdfSelectableTextErrorKind::MalformedCMap,
                                    "bfrange Unicode target overflows".to_owned(),
                                )
                            })?;
                            let decoded = decode_utf16be(&target)?;
                            insert_mapping(
                                &mut mappings,
                                first + offset as u16,
                                target.len(),
                                decoded,
                                &mut mapping_budget,
                            )?;
                        }
                        cursor += 1;
                    }
                    Some(CMapToken::ArrayStart) => {
                        cursor += 1;
                        for offset in 0..entry_count {
                            let encoded_target =
                                expect_hex(tokens, cursor, "bfrange array target")?;
                            mapping_budget.ensure_target_utf16_bytes(encoded_target.len())?;
                            mapping_budget.ensure_utf16_growth(encoded_target.len())?;
                            let target = decode_utf16be(encoded_target)?;
                            insert_mapping(
                                &mut mappings,
                                first + offset as u16,
                                encoded_target.len(),
                                target,
                                &mut mapping_budget,
                            )?;
                            cursor += 1;
                        }
                        if !matches!(tokens.get(cursor), Some(CMapToken::ArrayEnd)) {
                            return Err((
                                PdfSelectableTextErrorKind::MalformedCMap,
                                "bfrange target array length does not match source range"
                                    .to_owned(),
                            ));
                        }
                        cursor += 1;
                    }
                    _ => {
                        return Err((
                            PdfSelectableTextErrorKind::MalformedCMap,
                            "bfrange target is neither a hex string nor an array".to_owned(),
                        ));
                    }
                }
            }
            expect_word(tokens, cursor, "endbfrange")?;
            index = cursor + 1;
            continue;
        }
        if is_cmap_section_marker(&tokens[index]) {
            return Err((
                PdfSelectableTextErrorKind::UnsupportedCMap,
                format!(
                    "CMap mapping section {:?} is outside bfchar/bfrange",
                    tokens[index]
                ),
            ));
        }
        index += 1;
    }
    if !codespace_seen {
        return Err((
            PdfSelectableTextErrorKind::MalformedCMap,
            "ToUnicode has no codespace range".to_owned(),
        ));
    }
    if mappings.is_empty() {
        return Err((
            PdfSelectableTextErrorKind::MalformedCMap,
            "ToUnicode has no mappings".to_owned(),
        ));
    }
    Ok(mappings)
}

fn tokenize_cmap(bytes: &[u8]) -> Result<Vec<CMapToken>, (PdfSelectableTextErrorKind, String)> {
    let mut tokens = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            byte if byte.is_ascii_whitespace() => index += 1,
            b'%' => {
                index += 1;
                while index < bytes.len() && !matches!(bytes[index], b'\n' | b'\r') {
                    index += 1;
                }
            }
            b'[' => {
                tokens.push(CMapToken::ArrayStart);
                index += 1;
            }
            b']' => {
                tokens.push(CMapToken::ArrayEnd);
                index += 1;
            }
            b'<' if bytes.get(index + 1) == Some(&b'<') => {
                tokens.push(CMapToken::DictionaryStart);
                index += 2;
            }
            b'>' if bytes.get(index + 1) == Some(&b'>') => {
                tokens.push(CMapToken::DictionaryEnd);
                index += 2;
            }
            b'>' => {
                return Err((
                    PdfSelectableTextErrorKind::MalformedCMap,
                    "unmatched '>' in ToUnicode CMap".to_owned(),
                ));
            }
            b'{' => {
                tokens.push(CMapToken::ProcedureStart);
                index += 1;
            }
            b'}' => {
                tokens.push(CMapToken::ProcedureEnd);
                index += 1;
            }
            b'<' => {
                index += 1;
                let mut hex = Vec::new();
                while index < bytes.len() && bytes[index] != b'>' {
                    if !bytes[index].is_ascii_whitespace() {
                        hex.push(bytes[index]);
                    }
                    index += 1;
                }
                if index == bytes.len() {
                    return Err((
                        PdfSelectableTextErrorKind::MalformedCMap,
                        "unterminated CMap hex string".to_owned(),
                    ));
                }
                index += 1;
                if hex.is_empty() || hex.len() % 2 != 0 || !hex.iter().all(u8::is_ascii_hexdigit) {
                    return Err((
                        PdfSelectableTextErrorKind::MalformedCMap,
                        "CMap hex string must contain a non-empty even number of hex digits"
                            .to_owned(),
                    ));
                }
                let mut decoded = Vec::with_capacity(hex.len() / 2);
                for pair in hex.as_chunks::<2>().0 {
                    decoded.push((hex_value(pair[0])? << 4) | hex_value(pair[1])?);
                }
                tokens.push(CMapToken::Hex(decoded));
            }
            b'(' => {
                index += 1;
                let mut literal = Vec::new();
                let mut depth = 1usize;
                while index < bytes.len() && depth > 0 {
                    match bytes[index] {
                        b'\\' => {
                            index += 1;
                            let escaped = *bytes.get(index).ok_or_else(|| {
                                (
                                    PdfSelectableTextErrorKind::MalformedCMap,
                                    "unterminated escape in CMap literal string".to_owned(),
                                )
                            })?;
                            if escaped == b'\r' && bytes.get(index + 1) == Some(&b'\n') {
                                index += 2;
                                continue;
                            }
                            if escaped == b'\n' || escaped == b'\r' {
                                index += 1;
                                continue;
                            }
                            literal.push(escaped);
                            index += 1;
                        }
                        b'(' => {
                            depth += 1;
                            literal.push(b'(');
                            index += 1;
                        }
                        b')' => {
                            depth -= 1;
                            index += 1;
                            if depth > 0 {
                                literal.push(b')');
                            }
                        }
                        byte => {
                            literal.push(byte);
                            index += 1;
                        }
                    }
                }
                if depth != 0 {
                    return Err((
                        PdfSelectableTextErrorKind::MalformedCMap,
                        "unterminated CMap literal string".to_owned(),
                    ));
                }
                let _ = literal;
                tokens.push(CMapToken::Literal);
            }
            b')' => {
                return Err((
                    PdfSelectableTextErrorKind::MalformedCMap,
                    "unmatched ')' in CMap literal string".to_owned(),
                ));
            }
            _ => {
                let start = index;
                while index < bytes.len()
                    && !bytes[index].is_ascii_whitespace()
                    && !matches!(
                        bytes[index],
                        b'[' | b']' | b'<' | b'>' | b'(' | b')' | b'{' | b'}' | b'%'
                    )
                {
                    index += 1;
                }
                let word = std::str::from_utf8(&bytes[start..index]).map_err(|_| {
                    (
                        PdfSelectableTextErrorKind::MalformedCMap,
                        "CMap token is not ASCII/UTF-8".to_owned(),
                    )
                })?;
                tokens.push(CMapToken::Word(word.to_owned()));
            }
        }
        if tokens.len() > MAX_CMAP_TOKENS {
            return Err((
                PdfSelectableTextErrorKind::LimitExceeded,
                format!("ToUnicode exceeds {MAX_CMAP_TOKENS} lexical tokens"),
            ));
        }
    }
    Ok(tokens)
}

fn is_cmap_section_marker(token: &CMapToken) -> bool {
    matches!(token, CMapToken::Word(word) if
        (word.starts_with("begin") || word.starts_with("end"))
            && (word.contains("char") || word.contains("range")))
}

fn cmap_top_level_flags(
    tokens: &[CMapToken],
) -> Result<Vec<bool>, (PdfSelectableTextErrorKind, String)> {
    let mut stack = Vec::<CMapContainer>::new();
    let mut top_level = Vec::with_capacity(tokens.len());
    for token in tokens {
        top_level.push(stack.is_empty());
        match token {
            CMapToken::ArrayStart => stack.push(CMapContainer::Array),
            CMapToken::DictionaryStart => stack.push(CMapContainer::Dictionary),
            CMapToken::ProcedureStart | CMapToken::ProcedureEnd => {
                return Err((
                    PdfSelectableTextErrorKind::UnsupportedCMap,
                    "PostScript procedures are outside the exact active top-level ToUnicode subset"
                        .to_owned(),
                ));
            }
            CMapToken::ArrayEnd => {
                if stack.pop() != Some(CMapContainer::Array) {
                    return Err((
                        PdfSelectableTextErrorKind::MalformedCMap,
                        "unbalanced or mismatched array delimiter in ToUnicode CMap".to_owned(),
                    ));
                }
            }
            CMapToken::DictionaryEnd => {
                if stack.pop() != Some(CMapContainer::Dictionary) {
                    return Err((
                        PdfSelectableTextErrorKind::MalformedCMap,
                        "unbalanced or mismatched dictionary delimiter in ToUnicode CMap"
                            .to_owned(),
                    ));
                }
            }
            CMapToken::Word(_) | CMapToken::Hex(_) | CMapToken::Literal => {}
        }
    }
    if !stack.is_empty() {
        return Err((
            PdfSelectableTextErrorKind::MalformedCMap,
            "unterminated array or dictionary in ToUnicode CMap".to_owned(),
        ));
    }
    Ok(top_level)
}

fn object_number(object: &Object) -> Option<f64> {
    match object {
        Object::Integer(value) => Some(*value as f64),
        Object::Real(value) => Some(f64::from(*value)),
        _ => None,
    }
}

fn validate_zero_operands(operands: &[Object], operator: &str) -> Result<(), String> {
    if operands.is_empty() {
        Ok(())
    } else {
        Err(format!("{operator} expects no operands"))
    }
}

fn validate_exact_finite_numbers(
    operands: &[Object],
    expected: usize,
    operator: &str,
) -> Result<(), String> {
    if operands.len() != expected {
        return Err(format!(
            "{operator} expects exactly {expected} numeric operand{}",
            if expected == 1 { "" } else { "s" }
        ));
    }
    if operands
        .iter()
        .any(|operand| object_number(operand).is_none_or(|value| !value.is_finite()))
    {
        return Err(format!("{operator} expects only finite numeric operands"));
    }
    Ok(())
}

fn validate_integer_enum(
    operands: &[Object],
    operator: &str,
    admitted: RangeInclusive<i64>,
) -> Result<(), String> {
    let [Object::Integer(value)] = operands else {
        return Err(format!("{operator} expects exactly one integer operand"));
    };
    if admitted.contains(value) {
        Ok(())
    } else {
        Err(format!(
            "{operator} integer operand {value} is outside {}..={}",
            admitted.start(),
            admitted.end()
        ))
    }
}

fn validate_dash_operands(operands: &[Object]) -> Result<(), String> {
    let [Object::Array(pattern), phase] = operands else {
        return Err("d expects exactly one dash array and one numeric phase".to_owned());
    };
    if pattern.len() > MAX_DASH_ARRAY_ENTRIES {
        return Err(format!(
            "d dash array exceeds {MAX_DASH_ARRAY_ENTRIES} entries"
        ));
    }
    let mut all_zero = !pattern.is_empty();
    for entry in pattern {
        let value = object_number(entry)
            .ok_or_else(|| "d dash array contains a non-numeric entry".to_owned())?;
        if !value.is_finite() || value < 0.0 {
            return Err("d dash array contains a negative or non-finite entry".to_owned());
        }
        all_zero &= value == 0.0;
    }
    if all_zero {
        return Err("d dash array cannot contain only zero lengths".to_owned());
    }
    let phase = object_number(phase).ok_or_else(|| "d phase is not numeric".to_owned())?;
    if !phase.is_finite() {
        return Err("d phase is not finite".to_owned());
    }
    Ok(())
}

fn validate_single_name_operand(operands: &[Object], operator: &str) -> Result<(), String> {
    let [Object::Name(name)] = operands else {
        return Err(format!("{operator} expects exactly one name operand"));
    };
    if name.is_empty() || name.len() > MAX_RESOURCE_NAME_BYTES {
        return Err(format!(
            "{operator} name operand is outside the admitted 1..={MAX_RESOURCE_NAME_BYTES}-byte range"
        ));
    }
    Ok(())
}

fn validate_device_color_space_operand(
    operands: &[Object],
    operator: &str,
) -> Result<DeviceColorSpace, String> {
    let [Object::Name(name)] = operands else {
        return Err(format!(
            "{operator} expects exactly one device color-space name"
        ));
    };
    match name.as_slice() {
        b"DeviceGray" => Ok(DeviceColorSpace::Gray),
        b"DeviceRGB" => Ok(DeviceColorSpace::Rgb),
        b"DeviceCMYK" => Ok(DeviceColorSpace::Cmyk),
        _ => Err(format!(
            "{operator} color space /{} is outside the exact device-color subset",
            String::from_utf8_lossy(name)
        )),
    }
}

fn validate_color_operands(
    operands: &[Object],
    operator: &str,
    expected_components: usize,
) -> Result<(), String> {
    if operands.len() != expected_components {
        return Err(format!(
            "{operator} expects exactly {expected_components} numeric components for the selected device color space; pattern names are not admitted"
        ));
    }
    if operands
        .iter()
        .any(|operand| object_number(operand).is_none_or(|value| !value.is_finite()))
    {
        return Err(format!(
            "{operator} color components must be finite numbers"
        ));
    }
    Ok(())
}

fn checked_page_tj_operand_count(current: usize, additional: usize) -> Option<usize> {
    current
        .checked_add(additional)
        .filter(|count| *count <= MAX_PAGE_TJ_OPERANDS)
}

fn resource_name(bytes: &[u8]) -> Result<String, String> {
    let name =
        std::str::from_utf8(bytes).map_err(|_| "font resource name is not UTF-8".to_owned())?;
    if name.is_empty()
        || name.len() > MAX_RESOURCE_NAME_BYTES
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !matches!(byte, b'/' | b'#'))
    {
        return Err(format!(
            "font resource name is outside the admitted 1..={MAX_RESOURCE_NAME_BYTES}-byte printable ASCII subset"
        ));
    }
    Ok(name.to_owned())
}

fn checked_output_string_growth(
    current: usize,
    font_resource_bytes: usize,
    glyph_name_bytes: usize,
    unicode_bytes: usize,
    limit: usize,
) -> Option<usize> {
    // Each mapped value is owned once in unicode_by_code and once in the
    // aggregate unicode String. Glyph names and the font resource are owned
    // once in the public run.
    let duplicated_unicode_bytes = unicode_bytes.checked_mul(2)?;
    current
        .checked_add(font_resource_bytes)?
        .checked_add(glyph_name_bytes)?
        .checked_add(duplicated_unicode_bytes)
        .filter(|total| *total <= limit)
}

fn require_name(dict: &Dictionary, key: &[u8], expected: &[u8]) -> Result<(), String> {
    let actual = dict.get(key).and_then(Object::as_name).map_err(|cause| {
        format!(
            "dictionary has no valid /{} name: {cause}",
            String::from_utf8_lossy(key)
        )
    })?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "dictionary /{} is /{}, expected /{}",
            String::from_utf8_lossy(key),
            String::from_utf8_lossy(actual),
            String::from_utf8_lossy(expected)
        ))
    }
}

fn dereference_dict<'a>(
    doc: &'a Document,
    object: &'a Object,
    role: &str,
) -> Result<&'a Dictionary, String> {
    let (_, object) = doc
        .dereference(object)
        .map_err(|cause| format!("dereference {role}: {cause}"))?;
    object
        .as_dict()
        .map_err(|cause| format!("{role} is not a dictionary: {cause}"))
}

fn word_is(token: &CMapToken, expected: &str) -> bool {
    matches!(token, CMapToken::Word(word) if word == expected)
}

fn preceding_count(
    tokens: &[CMapToken],
    index: usize,
    section: &str,
) -> Result<usize, (PdfSelectableTextErrorKind, String)> {
    let count = index
        .checked_sub(1)
        .and_then(|previous| match &tokens[previous] {
            CMapToken::Word(word) => word.parse::<usize>().ok(),
            _ => None,
        })
        .ok_or_else(|| {
            (
                PdfSelectableTextErrorKind::MalformedCMap,
                format!("{section} lacks a valid preceding count"),
            )
        })?;
    if count > MAX_CMAP_ENTRIES {
        Err((
            PdfSelectableTextErrorKind::LimitExceeded,
            format!("{section} count {count} exceeds {MAX_CMAP_ENTRIES}"),
        ))
    } else {
        Ok(count)
    }
}

fn expect_hex<'a>(
    tokens: &'a [CMapToken],
    index: usize,
    role: &str,
) -> Result<&'a [u8], (PdfSelectableTextErrorKind, String)> {
    match tokens.get(index) {
        Some(CMapToken::Hex(bytes)) => Ok(bytes),
        _ => Err((
            PdfSelectableTextErrorKind::MalformedCMap,
            format!("{role} is not a hex string"),
        )),
    }
}

fn expect_word(
    tokens: &[CMapToken],
    index: usize,
    expected: &str,
) -> Result<(), (PdfSelectableTextErrorKind, String)> {
    if tokens
        .get(index)
        .is_some_and(|token| word_is(token, expected))
    {
        Ok(())
    } else {
        Err((
            PdfSelectableTextErrorKind::MalformedCMap,
            format!("CMap section is not terminated by {expected}"),
        ))
    }
}

fn source_code(bytes: &[u8]) -> Result<u16, (PdfSelectableTextErrorKind, String)> {
    let [high, low] = bytes else {
        return Err((
            PdfSelectableTextErrorKind::UnsupportedCMap,
            format!(
                "Identity-H source code has {} bytes; exactly two are required",
                bytes.len()
            ),
        ));
    };
    Ok(u16::from_be_bytes([*high, *low]))
}

fn decode_utf16be(bytes: &[u8]) -> Result<String, (PdfSelectableTextErrorKind, String)> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(2) {
        return Err((
            PdfSelectableTextErrorKind::MalformedCMap,
            "ToUnicode target must contain a non-empty even number of UTF-16BE bytes".to_owned(),
        ));
    }
    let units = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_be_bytes(*pair))
        .collect::<Vec<_>>();
    String::from_utf16(&units).map_err(|cause| {
        (
            PdfSelectableTextErrorKind::MalformedCMap,
            format!("ToUnicode target is not valid UTF-16BE: {cause}"),
        )
    })
}

fn decode_actual_text(bytes: &[u8]) -> Result<String, String> {
    if bytes.len() < 2 || !bytes.len().is_multiple_of(2) {
        return Err(
            "/ActualText must contain an even-length UTF-16BE string with a BOM".to_owned(),
        );
    }
    if bytes.len() > MAX_CMAP_TARGET_UTF16_BYTES {
        return Err(format!(
            "/ActualText has {} bytes, exceeding {MAX_CMAP_TARGET_UTF16_BYTES}",
            bytes.len()
        ));
    }
    if bytes[..2] != [0xfe, 0xff] {
        return Err("/ActualText must begin with the UTF-16BE BOM FEFF".to_owned());
    }
    let units = bytes[2..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| u16::from_be_bytes(*chunk))
        .collect::<Vec<_>>();
    String::from_utf16(&units)
        .map_err(|cause| format!("/ActualText is not valid UTF-16BE: {cause}"))
}

fn insert_mapping(
    mappings: &mut BTreeMap<u16, String>,
    source: u16,
    target_utf16_bytes: usize,
    target: String,
    budget: &mut CMapMappingBudget,
) -> Result<(), (PdfSelectableTextErrorKind, String)> {
    if let Some(existing) = mappings.get(&source) {
        let kind = if existing == &target {
            PdfSelectableTextErrorKind::DuplicateMapping
        } else {
            PdfSelectableTextErrorKind::ConflictingMapping
        };
        return Err((kind, format!("CID <{source:04X}> is mapped more than once")));
    }
    budget.charge(target_utf16_bytes, &target)?;
    mappings.insert(source, target);
    Ok(())
}

fn ensure_entry_budget(
    current: usize,
    additional: usize,
) -> Result<(), (PdfSelectableTextErrorKind, String)> {
    let total = current.checked_add(additional).ok_or_else(|| {
        (
            PdfSelectableTextErrorKind::LimitExceeded,
            "ToUnicode mapping count overflow".to_owned(),
        )
    })?;
    if total > MAX_CMAP_ENTRIES {
        Err((
            PdfSelectableTextErrorKind::LimitExceeded,
            format!("ToUnicode has {total} mappings, exceeding {MAX_CMAP_ENTRIES}"),
        ))
    } else {
        Ok(())
    }
}

fn increment_be(base: &[u8], offset: usize) -> Option<Vec<u8>> {
    let mut result = base.to_vec();
    let mut carry = offset;
    for byte in result.iter_mut().rev() {
        let sum = usize::from(*byte).checked_add(carry)?;
        *byte = (sum & 0xff) as u8;
        carry = sum >> 8;
    }
    (carry == 0).then_some(result)
}

fn hex_value(byte: u8) -> Result<u8, (PdfSelectableTextErrorKind, String)> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err((
            PdfSelectableTextErrorKind::MalformedCMap,
            "invalid hex digit in CMap".to_owned(),
        )),
    }
}

fn run_identity(run: &PdfSelectableTextRunV2) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash_field(
        &mut hash,
        b"domain",
        b"franken_ocr.pdf.selectable_text.run.v2",
    );
    hash_u64(&mut hash, b"run_index", run.run_index as u64);
    hash_u64(&mut hash, b"operation_index", run.operation_index as u64);
    hash_u64(&mut hash, b"operand_index", run.operand_index as u64);
    hash_field(&mut hash, b"font_resource", run.font_resource.as_bytes());
    hash_field(
        &mut hash,
        b"font_encoding",
        run.font_encoding.as_str().as_bytes(),
    );
    hash_field(&mut hash, b"code_width_bytes", &[run.code_width_bytes]);
    hash_field(&mut hash, b"font_mapping_sha256", &run.font_mapping_sha256);
    for code in &run.source_codes {
        hash_field(&mut hash, b"source_code", &code.to_be_bytes());
    }
    for glyph_name in &run.glyph_names {
        match glyph_name {
            Some(glyph_name) => {
                hash_field(&mut hash, b"glyph_name_present", &[1]);
                hash_field(&mut hash, b"glyph_name", glyph_name.as_bytes());
            }
            None => hash_field(&mut hash, b"glyph_name_present", &[0]),
        }
    }
    for unicode in &run.unicode_by_code {
        match unicode {
            Some(unicode) => {
                hash_field(&mut hash, b"unicode_by_code_present", &[1]);
                hash_field(&mut hash, b"unicode_by_code", unicode.as_bytes());
            }
            None => hash_field(&mut hash, b"unicode_by_code_present", &[0]),
        }
    }
    hash_field(
        &mut hash,
        b"unicode_complete",
        &[u8::from(run.unicode_complete)],
    );
    hash_field(&mut hash, b"unicode", run.unicode.as_bytes());
    for value in run
        .position
        .graphics_matrix
        .into_iter()
        .chain(run.position.text_matrix)
        .chain(run.position.line_matrix)
    {
        hash_field(&mut hash, b"matrix", &value.to_bits().to_be_bytes());
    }
    for value in [
        run.position.font_size,
        run.position.character_spacing,
        run.position.word_spacing,
        run.position.horizontal_scaling_percent,
        run.position.leading,
        run.position.rise,
    ] {
        hash_field(&mut hash, b"text_state", &value.to_bits().to_be_bytes());
    }
    hash_field(&mut hash, b"rendering_mode", &[run.position.rendering_mode]);
    for value in &run.tj_adjustments_before {
        hash_field(&mut hash, b"tj_before", &value.to_bits().to_be_bytes());
    }
    for value in &run.tj_adjustments_after {
        hash_field(&mut hash, b"tj_after", &value.to_bits().to_be_bytes());
    }
    hash.finalize().into()
}

fn page_identity(page: &PdfSelectableTextPageV2) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash_field(&mut hash, b"schema", page.schema.as_bytes());
    hash_field(&mut hash, b"source_sha256", &page.source_sha256);
    hash_u64(&mut hash, b"page_index", page.page_index as u64);
    for run in &page.runs {
        hash_field(&mut hash, b"run_identity", &run.identity_sha256);
    }
    hash.finalize().into()
}

fn hash_u64(hash: &mut Sha256, role: &[u8], value: u64) {
    hash_field(hash, role, &value.to_be_bytes());
}

fn hash_field(hash: &mut Sha256, role: &[u8], value: &[u8]) {
    hash.update((role.len() as u64).to_be_bytes());
    hash.update(role);
    hash.update((value.len() as u64).to_be_bytes());
    hash.update(value);
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::ZlibEncoder;
    use lopdf::content::{Content, Operation};
    use lopdf::{Dictionary, Document, Object, Stream, StringFormat, dictionary};
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::pdf::PdfPages;
    use crate::pdf::vector::parse_simple_type1_encoding;

    #[derive(Clone)]
    struct FontSpec<'a> {
        name: &'a str,
        cmap: Option<&'a str>,
        type0_subtype: &'a str,
        encoding: &'a str,
        descendant_subtype: &'a str,
        cid_to_gid: &'a str,
    }

    impl<'a> FontSpec<'a> {
        fn admitted(name: &'a str, cmap: &'a str) -> Self {
            Self {
                name,
                cmap: Some(cmap),
                type0_subtype: "Type0",
                encoding: "Identity-H",
                descendant_subtype: "CIDFontType2",
                cid_to_gid: "Identity",
            }
        }
    }

    fn full_cmap(mapping_sections: &str) -> String {
        format!(
            "/CIDInit /ProcSet findresource begin\n\
             12 dict begin\n\
             begincmap\n\
             /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n\
             /CMapName /Adobe-Identity-UCS def\n\
             /CMapType 2 def\n\
             1 begincodespacerange\n\
             <0000> <FFFF>\n\
             endcodespacerange\n\
             {mapping_sections}\n\
             endcmap\n\
             CMapName currentdict /CMap defineresource pop\n\
             end\n\
             end\n"
        )
    }

    fn one_mapping_cmap(unicode: &str) -> String {
        full_cmap(&format!("1 beginbfchar\n<0001> <{unicode}>\nendbfchar"))
    }

    fn zlib_bytes(bytes: &[u8]) -> Vec<u8> {
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(bytes).expect("compress CMap fixture");
        encoder.finish().expect("finish CMap fixture")
    }

    fn build_pdf(fonts: &[FontSpec<'_>], operations: Vec<Operation>) -> Vec<u8> {
        let mut doc = Document::with_version("1.7");
        let pages_id = doc.new_object_id();
        let mut font_resources = Dictionary::new();

        for spec in fonts {
            let descendant_id = doc.add_object(dictionary! {
                "Type" => "Font",
                "Subtype" => spec.descendant_subtype,
                "BaseFont" => "MTDTSubset",
                "CIDSystemInfo" => dictionary! {
                    "Registry" => Object::string_literal("Adobe"),
                    "Ordering" => Object::string_literal("Identity"),
                    "Supplement" => 0,
                },
                "CIDToGIDMap" => spec.cid_to_gid,
            });
            let mut type0 = dictionary! {
                "Type" => "Font",
                "Subtype" => spec.type0_subtype,
                "BaseFont" => "MTDTSubset",
                "Encoding" => spec.encoding,
                "DescendantFonts" => vec![descendant_id.into()],
            };
            if let Some(cmap) = spec.cmap {
                let cmap_id =
                    doc.add_object(Stream::new(Dictionary::new(), cmap.as_bytes().to_vec()));
                type0.set("ToUnicode", cmap_id);
            }
            let type0_id = doc.add_object(type0);
            font_resources.set(spec.name, type0_id);
        }

        let resources_id = doc.add_object(dictionary! { "Font" => font_resources });
        let content = Content { operations }.encode().expect("encode content");
        let content_id = doc.add_object(Stream::new(Dictionary::new(), content));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Resources" => resources_id,
            "Contents" => content_id,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        });
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => vec![page_id.into()],
                "Count" => 1,
            }),
        );
        let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        doc.trailer.set("Root", catalog_id);
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).expect("save fixture PDF");
        bytes
    }

    fn minimal_lilypond_type1c_program() -> Vec<u8> {
        const STANDARD_STRING_COUNT: u16 = 391;
        let custom_names = ["noteheads.s2", "brace189"];
        let mut string_index = Vec::new();
        string_index.extend_from_slice(&(custom_names.len() as u16).to_be_bytes());
        string_index.push(1);
        let mut offset = 1u8;
        string_index.push(offset);
        for name in custom_names {
            offset = offset
                .checked_add(name.len() as u8)
                .expect("small CFF String INDEX");
            string_index.push(offset);
        }
        for name in custom_names {
            string_index.extend_from_slice(name.as_bytes());
        }

        // Header + Name INDEX + four-byte Top DICT wrapped in its INDEX.
        let charset_offset = 4 + 6 + 9 + string_index.len() + 2;
        let charstrings_offset = charset_offset + 1 + 4 * 2;
        assert!(charset_offset <= 107 && charstrings_offset <= 107);
        let mut cff = vec![
            1,
            0,
            4,
            4, // header
            0,
            1,
            1,
            1,
            2,
            b'T', // Name INDEX
            0,
            1,
            1,
            1,
            5,
            (charset_offset + 139) as u8,
            15,
            (charstrings_offset + 139) as u8,
            17, // Top DICT INDEX
        ];
        cff.extend_from_slice(&string_index);
        cff.extend_from_slice(&[0, 0]); // Global Subr INDEX
        cff.push(0); // format-0 charset
        cff.extend_from_slice(&34u16.to_be_bytes()); // /A
        cff.extend_from_slice(&1u16.to_be_bytes()); // /space
        cff.extend_from_slice(&STANDARD_STRING_COUNT.to_be_bytes()); // /noteheads.s2
        cff.extend_from_slice(&(STANDARD_STRING_COUNT + 1).to_be_bytes()); // /brace189
        cff.extend_from_slice(&[
            0, 5, 1, 1, 2, 3, 4, 5, 6, // five one-byte CharStrings
            14, 14, 14, 14, 14, // .notdef plus the four encoded glyphs
        ]);
        let table = cff::Table::parse(&cff).expect("constructed Type1C program");
        assert_eq!(table.number_of_glyphs(), 5);
        for name in ["A", "space", "noteheads.s2", "brace189"] {
            assert!(
                table.glyph_index_by_name(name).is_some(),
                "constructed Type1C program has no /{name}"
            );
        }
        cff
    }

    fn build_lilypond_type1c_pdf(operations: Vec<Operation>) -> Vec<u8> {
        build_lilypond_type1c_pdf_with_font(
            operations,
            vec![
                0.into(),
                Object::Name(b"A".to_vec()),
                Object::Name(b"noteheads.s2".to_vec()),
                Object::Name(b"brace189".to_vec()),
            ],
            false,
        )
    }

    fn build_lilypond_type1c_pdf_with_font(
        operations: Vec<Operation>,
        differences: Vec<Object>,
        add_to_unicode: bool,
    ) -> Vec<u8> {
        let mut doc = Document::with_version("1.7");
        let pages_id = doc.new_object_id();
        let font_file_id = doc.add_object(Stream::new(
            dictionary! { "Subtype" => "Type1C" },
            minimal_lilypond_type1c_program(),
        ));
        let descriptor_id = doc.add_object(dictionary! {
            "Type" => "FontDescriptor",
            "FontName" => "FixtureMusic",
            "FontFile3" => font_file_id,
        });
        let encoding_id = doc.add_object(dictionary! {
            "Type" => "Encoding",
            "BaseEncoding" => "WinAnsiEncoding",
            "Differences" => differences,
        });
        let mut font = dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "FixtureMusic",
            "FirstChar" => 0,
            "LastChar" => 32,
            "Widths" => (0..=32).map(|_| Object::Integer(500)).collect::<Vec<_>>(),
            "Encoding" => encoding_id,
            "FontDescriptor" => descriptor_id,
        };
        if add_to_unicode {
            let to_unicode_id = doc.add_object(Stream::new(
                Dictionary::new(),
                one_mapping_cmap("0041").into_bytes(),
            ));
            font.set("ToUnicode", to_unicode_id);
        }
        let font_id = doc.add_object(font);
        let resources_id = doc.add_object(dictionary! {
            "Font" => dictionary! { "FMusic" => font_id },
        });
        let content = Content { operations }.encode().expect("encode content");
        let content_id = doc.add_object(Stream::new(Dictionary::new(), content));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Resources" => resources_id,
            "Contents" => content_id,
            "MediaBox" => vec![0.into(), 0.into(), 200.into(), 200.into()],
        });
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => vec![page_id.into()],
                "Count" => 1,
            }),
        );
        let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        doc.trailer.set("Root", catalog_id);
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).expect("save Type1C fixture PDF");
        bytes
    }

    fn parse_type1_encoding_fixture(
        base_encoding: Object,
        differences: Object,
    ) -> Result<Vec<Option<String>>, String> {
        let mut doc = Document::with_version("1.7");
        let encoding_id = doc.add_object(dictionary! {
            "Type" => "Encoding",
            "BaseEncoding" => base_encoding,
            "Differences" => differences,
        });
        let font = dictionary! { "Encoding" => encoding_id };
        parse_simple_type1_encoding(&doc, &font)
    }

    fn bt() -> Operation {
        Operation::new("BT", Vec::new())
    }

    fn et() -> Operation {
        Operation::new("ET", Vec::new())
    }

    fn tf(name: &str, size: i64) -> Operation {
        Operation::new(
            "Tf",
            vec![Object::Name(name.as_bytes().to_vec()), size.into()],
        )
    }

    fn tm(x: i64, y: i64) -> Operation {
        Operation::new(
            "Tm",
            vec![1.into(), 0.into(), 0.into(), 1.into(), x.into(), y.into()],
        )
    }

    fn hex_string(bytes: &[u8]) -> Object {
        Object::String(bytes.to_vec(), StringFormat::Hexadecimal)
    }

    fn literal_string(bytes: &[u8]) -> Object {
        Object::String(bytes.to_vec(), StringFormat::Literal)
    }

    #[test]
    fn exact_mtdt_fonts_runs_tj_tj_position_and_identities_are_stable() {
        let first_cmap = full_cmap(
            "3 beginbfchar\n\
             <0020> <0041>\n\
             <0002> <0042>\n\
             <0003> <D834DD1E>\n\
             endbfchar",
        );
        let second_cmap = one_mapping_cmap("005A");
        let operations = vec![
            Operation::new(
                "cm",
                vec![2.into(), 0.into(), 0.into(), 2.into(), 5.into(), 7.into()],
            ),
            bt(),
            tf("F0", 12),
            Operation::new("Tc", vec![1.into()]),
            Operation::new("Tw", vec![2.into()]),
            Operation::new("Tz", vec![90.into()]),
            Operation::new("TL", vec![14.into()]),
            Operation::new("Ts", vec![3.into()]),
            Operation::new("Tr", vec![1.into()]),
            tm(10, 20),
            Operation::new("Tj", vec![literal_string(&[0x00, 0x20])]),
            Operation::new(
                "TJ",
                vec![Object::Array(vec![
                    hex_string(&[0x00, 0x02]),
                    (-120).into(),
                    hex_string(&[0x00, 0x03]),
                    40.into(),
                ])],
            ),
            Operation::new("Tj", vec![hex_string(&[0x00, 0x20])]),
            tf("F1", 9),
            tm(30, 40),
            Operation::new("Tj", vec![hex_string(&[0x00, 0x01])]),
            et(),
        ];
        let bytes = build_pdf(
            &[
                FontSpec::admitted("F0", &first_cmap),
                FontSpec::admitted("F1", &second_cmap),
            ],
            operations,
        );
        let expected_source_hash: [u8; 32] = Sha256::digest(&bytes).into();
        let pages = PdfPages::from_bytes(bytes).expect("parse exact MTDT fixture");

        let first = pages.selectable_text(0).expect("extract selectable text");
        let second = pages.selectable_text(0).expect("repeat selectable text");
        assert_eq!(first, second);
        assert_eq!(first.schema, PDF_SELECTABLE_TEXT_SCHEMA_V2);
        assert_eq!(first.page_index, 0);
        assert_eq!(first.source_sha256, expected_source_hash);
        assert_ne!(first.identity_sha256, [0; 32]);
        assert_eq!(first.runs.len(), 5);
        assert_eq!(
            first
                .runs
                .iter()
                .map(|run| run.unicode.as_str())
                .collect::<Vec<_>>(),
            ["A", "B", "𝄞", "A", "Z"]
        );
        assert_eq!(
            first
                .runs
                .iter()
                .map(|run| run.operation_index)
                .collect::<Vec<_>>(),
            [10, 11, 11, 12, 15]
        );
        assert_eq!(
            first
                .runs
                .iter()
                .map(|run| run.operand_index)
                .collect::<Vec<_>>(),
            [0, 0, 2, 0, 0]
        );
        assert_eq!(first.runs[2].tj_adjustments_before, [-120.0]);
        assert_eq!(first.runs[2].tj_adjustments_after, [40.0]);
        assert_eq!(first.runs[0].font_resource, "F0");
        assert_eq!(first.runs[4].font_resource, "F1");
        assert!(first.runs.iter().all(|run| {
            run.font_encoding == PdfSelectableTextEncodingV2::Type0IdentityH
                && run.code_width_bytes == 2
                && run.glyph_names.iter().all(Option::is_none)
                && run.unicode_complete
                && run.unicode_by_code.iter().all(Option::is_some)
        }));
        assert_ne!(
            first.runs[0].font_mapping_sha256,
            first.runs[4].font_mapping_sha256
        );
        assert_eq!(
            first.runs[0].position.graphics_matrix,
            [2.0, 0.0, 0.0, 2.0, 5.0, 7.0]
        );
        assert_eq!(
            first.runs[0].position.text_matrix,
            [1.0, 0.0, 0.0, 1.0, 10.0, 20.0]
        );
        assert_eq!(first.runs[0].position.font_size, 12.0);
        assert_eq!(first.runs[0].position.character_spacing, 1.0);
        assert_eq!(first.runs[0].position.word_spacing, 2.0);
        assert_eq!(first.runs[0].position.horizontal_scaling_percent, 90.0);
        assert_eq!(first.runs[0].position.leading, 14.0);
        assert_eq!(first.runs[0].position.rise, 3.0);
        assert_eq!(first.runs[0].position.rendering_mode, 1);
        assert!((first.runs[1].position.text_matrix[4] - 21.7).abs() < 1.0e-9);
        assert!((first.runs[2].position.text_matrix[4] - 34.696).abs() < 1.0e-9);
        assert!((first.runs[3].position.text_matrix[4] - 45.964).abs() < 1.0e-9);
        assert!(first.runs.iter().all(|run| run.identity_sha256 != [0; 32]));
    }

    #[test]
    fn lilypond_type1c_codes_names_opaque_music_and_quote_operators_are_ledgered() {
        let operations = vec![
            bt(),
            tf("FMusic", 12),
            Operation::new("TL", vec![14.into()]),
            Operation::new("Tz", vec![50.into()]),
            tm(10, 100),
            Operation::new("Tj", vec![literal_string(&[0, 1, 2])]),
            Operation::new("'", vec![literal_string(&[0])]),
            Operation::new("\"", vec![2.into(), 1.into(), literal_string(&[32, 0])]),
            Operation::new("Tj", vec![literal_string(&[0])]),
            et(),
        ];
        let pages = PdfPages::from_bytes(build_lilypond_type1c_pdf(operations))
            .expect("parse Type1C fixture");
        let first = pages
            .selectable_text(0)
            .expect("extract Type1C selectable text");
        let second = pages
            .selectable_text(0)
            .expect("repeat Type1C selectable text");
        assert_eq!(first, second);
        assert_eq!(first.schema, PDF_SELECTABLE_TEXT_SCHEMA_V2);
        assert_eq!(first.runs.len(), 4);
        let mixed = &first.runs[0];
        assert_eq!(
            mixed.font_encoding,
            PdfSelectableTextEncodingV2::Type1cWinAnsiDifferences
        );
        assert_eq!(mixed.font_encoding.as_str(), "type1c_win_ansi_differences");
        assert_eq!(mixed.code_width_bytes, 1);
        assert_eq!(mixed.source_codes, [0, 1, 2]);
        assert_eq!(
            mixed.glyph_names,
            [
                Some("A".to_owned()),
                Some("noteheads.s2".to_owned()),
                Some("brace189".to_owned()),
            ]
        );
        assert_eq!(mixed.unicode_by_code, [Some("A".to_owned()), None, None]);
        assert_eq!(mixed.unicode, "A");
        assert!(!mixed.unicode_complete);
        assert_ne!(mixed.font_mapping_sha256, [0; 32]);

        assert_eq!(first.runs[1].operation_index, 6);
        assert_eq!(first.runs[1].operand_index, 0);
        assert_eq!(first.runs[1].position.text_matrix[5], 86.0);
        assert_eq!(first.runs[2].operation_index, 7);
        assert_eq!(first.runs[2].operand_index, 2);
        assert_eq!(first.runs[2].source_codes, [32, 0]);
        assert_eq!(
            first.runs[2].glyph_names,
            [Some("space".to_owned()), Some("A".to_owned())]
        );
        assert_eq!(
            first.runs[2].unicode_by_code,
            [Some(" ".to_owned()), Some("A".to_owned())]
        );
        assert_eq!(first.runs[2].unicode, " A");
        assert_eq!(first.runs[2].position.text_matrix[4], 10.0);
        assert_eq!(first.runs[2].position.text_matrix[5], 72.0);
        assert_eq!(first.runs[2].position.word_spacing, 2.0);
        assert_eq!(first.runs[2].position.character_spacing, 1.0);
        assert_eq!(first.runs[2].position.horizontal_scaling_percent, 50.0);
        assert_eq!(first.runs[3].operation_index, 8);
        assert_eq!(first.runs[3].operand_index, 0);
        assert_eq!(first.runs[3].position.text_matrix[4], 18.0);
        assert_eq!(first.runs[3].position.text_matrix[5], 72.0);
        assert_eq!(first.runs[3].position.word_spacing, 2.0);
        assert_eq!(first.runs[3].position.character_spacing, 1.0);
        assert_eq!(first.runs[3].position.horizontal_scaling_percent, 50.0);
        assert!(first.runs[1..].iter().all(|run| run.unicode_complete));

        let original_identity = mixed.identity_sha256;
        let mut changed = mixed.clone();
        changed.font_encoding = PdfSelectableTextEncodingV2::Type0IdentityH;
        assert_ne!(run_identity(&changed), original_identity);
        let mut changed = mixed.clone();
        changed.code_width_bytes = 2;
        assert_ne!(run_identity(&changed), original_identity);
        let mut changed = mixed.clone();
        changed.glyph_names[0] = None;
        assert_ne!(run_identity(&changed), original_identity);
        let mut changed = mixed.clone();
        changed.unicode_by_code[0] = None;
        assert_ne!(run_identity(&changed), original_identity);
        let mut changed = mixed.clone();
        changed.unicode_complete = true;
        assert_ne!(run_identity(&changed), original_identity);
    }

    #[test]
    fn public_identity_validation_detects_body_and_root_mutation() {
        let pages = PdfPages::from_bytes(build_lilypond_type1c_pdf(vec![
            bt(),
            tf("FMusic", 12),
            tm(10, 100),
            Operation::new(
                "TJ",
                vec![Object::Array(vec![
                    literal_string(&[0, 1]),
                    (-120).into(),
                    literal_string(&[2]),
                ])],
            ),
            et(),
        ]))
        .expect("parse identity fixture");
        let page = pages.selectable_text(0).expect("extract identity fixture");
        assert!(page.runs[0].identity_is_valid());
        assert!(page.identity_is_valid());

        let mut changed_tj = page.clone();
        changed_tj.runs[1].tj_adjustments_before[0] = -999.0;
        assert!(!changed_tj.runs[1].identity_is_valid());
        assert!(!changed_tj.identity_is_valid());

        let mut changed_position = page.clone();
        changed_position.runs[0].position.text_matrix[4] += 1.0;
        assert!(!changed_position.runs[0].identity_is_valid());
        assert!(!changed_position.identity_is_valid());

        let mut changed_page_root = page;
        changed_page_root.identity_sha256[0] ^= 1;
        assert!(changed_page_root.runs[0].identity_is_valid());
        assert!(!changed_page_root.identity_is_valid());
    }

    #[test]
    fn admitted_agl_mapping_never_guesses_music_glyph_names() {
        assert_eq!(glyph_name_to_unicode("Euro").as_deref(), Some("€"));
        assert_eq!(glyph_name_to_unicode("fi").as_deref(), Some("ﬁ"));
        assert_eq!(glyph_name_to_unicode("Delta").as_deref(), Some("∆"));
        assert_eq!(glyph_name_to_unicode("Omega").as_deref(), Some("Ω"));
        assert_eq!(glyph_name_to_unicode("dotlessj"), None);
        assert_eq!(glyph_name_to_unicode("uni00410042").as_deref(), Some("AB"));
        assert_eq!(glyph_name_to_unicode("u1D11E").as_deref(), Some("𝄞"));
        assert_eq!(glyph_name_to_unicode("uni004a"), None);
        assert_eq!(glyph_name_to_unicode("u1d11e"), None);
        assert_eq!(glyph_name_to_unicode("noteheads.s2"), None);
        assert_eq!(glyph_name_to_unicode("brace189"), None);
        assert_eq!(glyph_name_to_unicode(".notdef"), None);
    }

    #[test]
    fn bfchar_and_both_bfrange_forms_decode_ascii_ligature_and_supplementary_unicode() {
        let cmap = full_cmap(
            "1 beginbfchar\n<0001> <0041>\nendbfchar\n\
             1 beginbfrange\n<0002> <0003> <0042>\nendbfrange\n\
             1 beginbfrange\n<0004> <0005> [<D834DD1E> <00660069>]\nendbfrange",
        );
        let map = parse_cmap(cmap.as_bytes()).expect("parse admitted forms");
        assert_eq!(map.get(&1).map(String::as_str), Some("A"));
        assert_eq!(map.get(&2).map(String::as_str), Some("B"));
        assert_eq!(map.get(&3).map(String::as_str), Some("C"));
        assert_eq!(map.get(&4).map(String::as_str), Some("𝄞"));
        assert_eq!(map.get(&5).map(String::as_str), Some("fi"));
    }

    #[test]
    fn malformed_duplicate_conflicting_and_unsupported_cmaps_refuse_by_kind() {
        let cases = [
            (
                full_cmap("2 beginbfchar\n<0001> <0041>\n<0001> <0041>\nendbfchar"),
                PdfSelectableTextErrorKind::DuplicateMapping,
            ),
            (
                full_cmap("2 beginbfchar\n<0001> <0041>\n<0001> <0042>\nendbfchar"),
                PdfSelectableTextErrorKind::ConflictingMapping,
            ),
            (
                full_cmap("1 beginbfchar\n<0001> <D800>\nendbfchar"),
                PdfSelectableTextErrorKind::MalformedCMap,
            ),
            (
                full_cmap("1 beginbfchar\n<01> <0041>\nendbfchar"),
                PdfSelectableTextErrorKind::UnsupportedCMap,
            ),
            (
                full_cmap("1 begincidchar\n<0001> 1\nendcidchar"),
                PdfSelectableTextErrorKind::UnsupportedCMap,
            ),
            (
                full_cmap("65537 beginbfchar\nendbfchar"),
                PdfSelectableTextErrorKind::LimitExceeded,
            ),
        ];
        for (cmap, expected) in cases {
            let (actual, detail) = parse_cmap(cmap.as_bytes()).expect_err("CMap must refuse");
            assert_eq!(actual, expected, "wrong refusal for {detail}");
        }

        let wrong_codespace = full_cmap("1 beginbfchar\n<0001> <0041>\nendbfchar")
            .replace("<0000> <FFFF>", "<00> <FF>");
        assert_eq!(
            parse_cmap(wrong_codespace.as_bytes()).unwrap_err().0,
            PdfSelectableTextErrorKind::UnsupportedCMap
        );
        let inherited = full_cmap("/Other usecmap\n1 beginbfchar\n<0001> <0041>\nendbfchar");
        assert_eq!(
            parse_cmap(inherited.as_bytes()).unwrap_err().0,
            PdfSelectableTextErrorKind::UnsupportedCMap
        );

        let outside_before = full_cmap("1 beginbfchar\n<0001> <0041>\nendbfchar").replacen(
            "begincmap\n",
            "1 beginbfchar\n<0002> <0042>\nendbfchar\nbegincmap\n",
            1,
        );
        assert_eq!(
            parse_cmap(outside_before.as_bytes()).unwrap_err().0,
            PdfSelectableTextErrorKind::MalformedCMap
        );
        let outside_after = full_cmap("1 beginbfchar\n<0001> <0041>\nendbfchar").replacen(
            "endcmap\n",
            "endcmap\n1 beginbfchar\n<0002> <0042>\nendbfchar\n",
            1,
        );
        assert_eq!(
            parse_cmap(outside_after.as_bytes()).unwrap_err().0,
            PdfSelectableTextErrorKind::MalformedCMap
        );
        let mapping_before_codespace = full_cmap("").replacen(
            "1 begincodespacerange\n",
            "1 beginbfchar\n<0001> <0041>\nendbfchar\n1 begincodespacerange\n",
            1,
        );
        assert_eq!(
            parse_cmap(mapping_before_codespace.as_bytes())
                .unwrap_err()
                .0,
            PdfSelectableTextErrorKind::MalformedCMap
        );
        let stray_end = full_cmap("1 beginbfchar\n<0001> <0041>\nendbfchar\nendbfchar");
        assert_eq!(
            parse_cmap(stray_end.as_bytes()).unwrap_err().0,
            PdfSelectableTextErrorKind::UnsupportedCMap
        );
    }

    #[test]
    fn cmap_semantics_must_be_active_at_top_level_inside_begincmap() {
        let mapping = "1 beginbfchar\n<0001> <0041>\nendbfchar";
        for (inert, expected_detail) in [
            (format!("{{ {mapping} }}"), "PostScript procedures"),
            (format!("[ {mapping} ]"), "active at top level"),
            (
                format!("<< /Ignored [ {mapping} ] >>"),
                "active at top level",
            ),
        ] {
            let error = parse_cmap(full_cmap(&inert).as_bytes())
                .expect_err("inert mapping section must refuse");
            assert_eq!(error.0, PdfSelectableTextErrorKind::UnsupportedCMap);
            assert!(error.1.contains(expected_detail), "{}", error.1);
        }

        let inert_type = full_cmap(mapping).replace("/CMapType 2 def", "[ /CMapType 2 def ]");
        let error = parse_cmap(inert_type.as_bytes()).expect_err("inert CMapType must refuse");
        assert_eq!(error.0, PdfSelectableTextErrorKind::UnsupportedCMap);
        assert!(error.1.contains("active at top level"), "{}", error.1);

        let inert_program = format!("{{ {} }}", full_cmap(mapping));
        let error = parse_cmap(inert_program.as_bytes()).expect_err("inert begincmap must refuse");
        assert_eq!(error.0, PdfSelectableTextErrorKind::UnsupportedCMap);
        assert!(error.1.contains("PostScript procedures"), "{}", error.1);

        let redefined_type =
            full_cmap(mapping).replace("/CMapType 2 def", "/CMapType 2 def\n/CMapType 1 def");
        let error =
            parse_cmap(redefined_type.as_bytes()).expect_err("CMapType redefinition must refuse");
        assert_eq!(error.0, PdfSelectableTextErrorKind::UnsupportedCMap);
        assert!(error.1.contains("found 2 declarations"), "{}", error.1);

        let outside_type = full_cmap(mapping).replace("endcmap\n", "endcmap\n/CMapType 1 def\n");
        let error = parse_cmap(outside_type.as_bytes()).expect_err("outside CMapType must refuse");
        assert_eq!(error.0, PdfSelectableTextErrorKind::MalformedCMap);
        assert!(error.1.contains("outside the sole"), "{}", error.1);
    }

    #[test]
    fn cmap_target_and_expansion_budgets_refuse_before_range_materialization() {
        let oversized_target = "0041".repeat(MAX_CMAP_TARGET_UTF16_BYTES / 2 + 1);
        let error = parse_cmap(one_mapping_cmap(&oversized_target).as_bytes())
            .expect_err("oversized individual target must refuse");
        assert_eq!(error.0, PdfSelectableTextErrorKind::LimitExceeded);
        assert!(error.1.contains("target has"), "{}", error.1);

        let admitted_target = "0041".repeat(MAX_CMAP_TARGET_UTF16_BYTES / 2);
        let entry_count = MAX_CMAP_MAPPING_UTF16_BYTES / MAX_CMAP_TARGET_UTF16_BYTES + 1;
        let last = entry_count - 1;
        let amplified = full_cmap(&format!(
            "1 beginbfrange\n<0000> <{last:04X}> <{admitted_target}>\nendbfrange"
        ));
        let error = parse_cmap(amplified.as_bytes())
            .expect_err("bounded source must not amplify into an oversized mapping table");
        assert_eq!(error.0, PdfSelectableTextErrorKind::LimitExceeded);
        assert!(error.1.contains("target bytes"), "{}", error.1);
    }

    #[test]
    fn cmap_flate_decode_parameters_are_strict_and_fail_closed() {
        let cmap = one_mapping_cmap("0041");
        let encoded = zlib_bytes(cmap.as_bytes());

        let valid = Stream::new(
            dictionary! {
                "Filter" => "FlateDecode",
                "DecodeParms" => dictionary! { "Predictor" => 1 },
            },
            encoded.clone(),
        );
        assert_eq!(
            decode_cmap_stream(&valid).expect("admitted Flate CMap"),
            cmap.as_bytes()
        );

        let cases = [
            Stream::new(
                dictionary! { "Filter" => vec![Object::Name(b"FlateDecode".to_vec())] },
                encoded.clone(),
            ),
            Stream::new(
                dictionary! { "Filter" => Vec::<Object>::new() },
                cmap.as_bytes().to_vec(),
            ),
            Stream::new(
                dictionary! { "Filter" => "FlateDecode", "DecodeParms" => 7 },
                encoded.clone(),
            ),
            Stream::new(
                dictionary! {
                    "Filter" => "FlateDecode",
                    "DecodeParms" => dictionary! { "Predictor" => 2 },
                },
                encoded.clone(),
            ),
            Stream::new(
                dictionary! {
                    "Filter" => "FlateDecode",
                    "DecodeParms" => dictionary! { "Columns" => 1 },
                },
                encoded.clone(),
            ),
            Stream::new(
                dictionary! {
                    "Filter" => "FlateDecode",
                    "DecodeParms" => Dictionary::new(),
                    "DP" => Dictionary::new(),
                },
                encoded.clone(),
            ),
            Stream::new(
                dictionary! { "DecodeParms" => Dictionary::new() },
                cmap.as_bytes().to_vec(),
            ),
        ];
        for stream in cases {
            let error = decode_cmap_stream(&stream).expect_err("decode parameters must refuse");
            assert_eq!(error.0, PdfSelectableTextErrorKind::UnsupportedCMap);
        }
    }

    #[test]
    fn type1_encoding_and_program_contracts_refuse_ambiguous_fonts() {
        let wrong_base = parse_type1_encoding_fixture(
            Object::Name(b"MacRomanEncoding".to_vec()),
            Object::Array(Vec::new()),
        )
        .expect_err("wrong BaseEncoding must refuse");
        assert!(wrong_base.contains("WinAnsiEncoding"));

        let malformed =
            parse_type1_encoding_fixture(Object::Name(b"WinAnsiEncoding".to_vec()), 7.into())
                .expect_err("non-array Differences must refuse");
        assert!(malformed.contains("Differences"));

        let name_first = parse_type1_encoding_fixture(
            Object::Name(b"WinAnsiEncoding".to_vec()),
            Object::Array(vec![Object::Name(b"A".to_vec())]),
        )
        .expect_err("name before code must refuse");
        assert!(name_first.contains("before its first code"));

        let duplicate = parse_type1_encoding_fixture(
            Object::Name(b"WinAnsiEncoding".to_vec()),
            Object::Array(vec![
                0.into(),
                Object::Name(b"A".to_vec()),
                0.into(),
                Object::Name(b"B".to_vec()),
            ]),
        )
        .expect_err("duplicate Differences code must refuse");
        assert!(duplicate.contains("more than once"));

        let oversized_name = parse_type1_encoding_fixture(
            Object::Name(b"WinAnsiEncoding".to_vec()),
            Object::Array(vec![0.into(), Object::Name(vec![b'A'; 256])]),
        )
        .expect_err("oversized glyph name must refuse");
        assert!(oversized_name.contains("1..=255"));

        let operations = vec![
            bt(),
            tf("FMusic", 12),
            tm(0, 0),
            Operation::new("Tj", vec![literal_string(&[0])]),
            et(),
        ];
        let missing_glyph = PdfPages::from_bytes(build_lilypond_type1c_pdf_with_font(
            operations.clone(),
            vec![0.into(), Object::Name(b"missingGlyph".to_vec())],
            false,
        ))
        .expect("parse missing-glyph fixture")
        .selectable_text(0)
        .expect_err("missing CFF glyph must refuse");
        assert_eq!(
            missing_glyph.kind,
            PdfSelectableTextErrorKind::MissingMapping
        );
        assert!(missing_glyph.detail.contains("missing CFF glyph"));

        let to_unicode = PdfPages::from_bytes(build_lilypond_type1c_pdf_with_font(
            operations,
            vec![0.into(), Object::Name(b"A".to_vec())],
            true,
        ))
        .expect("parse Type1 ToUnicode fixture")
        .selectable_text(0)
        .expect_err("Type1 ToUnicode must refuse");
        assert_eq!(to_unicode.kind, PdfSelectableTextErrorKind::UnsupportedFont);
        assert!(to_unicode.detail.contains("ToUnicode"));
    }

    #[test]
    fn owned_string_and_resource_name_bounds_are_checked_without_large_allocations() {
        assert_eq!(checked_output_string_growth(3, 2, 5, 7, 24), Some(24));
        assert_eq!(checked_output_string_growth(3, 2, 5, 7, 23), None);
        assert_eq!(
            checked_output_string_growth(usize::MAX, 1, 0, 0, usize::MAX),
            None
        );
        assert_eq!(
            checked_output_string_growth(0, 0, 0, usize::MAX, usize::MAX),
            None
        );
        assert!(resource_name(b"FMusic").is_ok());
        assert!(resource_name(&[b'F'; MAX_RESOURCE_NAME_BYTES]).is_ok());
        assert!(resource_name(&[b'F'; MAX_RESOURCE_NAME_BYTES + 1]).is_err());
    }

    #[test]
    fn public_path_preflights_code_count_and_unicode_growth_before_materializing_runs() {
        let oversized_codes = vec![0; MAX_TEXT_CODES + 1];
        let type1_pages = PdfPages::from_bytes(build_lilypond_type1c_pdf(vec![
            bt(),
            tf("FMusic", 12),
            tm(0, 0),
            Operation::new("Tj", vec![literal_string(&oversized_codes)]),
            et(),
        ]))
        .expect("parse code-count fixture");
        let code_error = type1_pages
            .selectable_text(0)
            .expect_err("oversized source-code run must refuse");
        assert_eq!(code_error.kind, PdfSelectableTextErrorKind::LimitExceeded);
        assert!(code_error.detail.contains("code selectable-text limit"));

        let target_scalar_count = MAX_CMAP_TARGET_UTF16_BYTES / 2;
        let long_target = "0041".repeat(target_scalar_count);
        let cmap = one_mapping_cmap(&long_target);
        let repetitions = MAX_DECODED_UNICODE_SCALARS / target_scalar_count + 1;
        let mut shown = Vec::with_capacity(repetitions * 2);
        for _ in 0..repetitions {
            shown.extend_from_slice(&[0, 1]);
        }
        let type0_pages = PdfPages::from_bytes(build_pdf(
            &[FontSpec::admitted("F0", &cmap)],
            vec![
                bt(),
                tf("F0", 12),
                tm(0, 0),
                Operation::new("Tj", vec![hex_string(&shown)]),
                et(),
            ],
        ))
        .expect("parse Unicode-growth fixture");
        let unicode_error = type0_pages
            .selectable_text(0)
            .expect_err("oversized decoded Unicode run must refuse");
        assert_eq!(
            unicode_error.kind,
            PdfSelectableTextErrorKind::LimitExceeded
        );
        assert!(unicode_error.detail.contains("scalar Unicode limit"));
    }

    #[test]
    fn public_api_refuses_missing_mapping_odd_codes_and_unknown_text_operators_with_context() {
        let cmap = one_mapping_cmap("0041");
        let fonts = [FontSpec::admitted("F0", &cmap)];
        let cases = [
            (
                vec![
                    bt(),
                    tf("F0", 12),
                    tm(0, 0),
                    Operation::new("Tj", vec![hex_string(&[0x00, 0x02])]),
                    et(),
                ],
                PdfSelectableTextErrorKind::MissingMapping,
                3,
            ),
            (
                vec![
                    bt(),
                    tf("F0", 12),
                    tm(0, 0),
                    Operation::new("Tj", vec![hex_string(&[0x00])]),
                    et(),
                ],
                PdfSelectableTextErrorKind::MalformedTextOperand,
                3,
            ),
            (
                vec![
                    bt(),
                    tf("F0", 12),
                    tm(0, 0),
                    Operation::new("Tfoo", Vec::new()),
                    et(),
                ],
                PdfSelectableTextErrorKind::UnsupportedTextOperator,
                3,
            ),
        ];
        for (operations, expected_kind, expected_operation) in cases {
            let pages = PdfPages::from_bytes(build_pdf(&fonts, operations)).expect("parse fixture");
            let error = pages.selectable_text(0).expect_err("fixture must refuse");
            assert_eq!(error.kind, expected_kind);
            assert_eq!(error.operation_index, Some(expected_operation));
            assert_eq!(error.font_resource.as_deref(), Some("F0"));
            assert!(error.to_string().contains("zero-based page 0"));
        }
    }

    #[test]
    fn missing_tounicode_wrong_encoding_and_wrong_font_shapes_never_guess() {
        let cmap = one_mapping_cmap("0041");
        let mut cases = Vec::new();
        let mut missing = FontSpec::admitted("F0", &cmap);
        missing.cmap = None;
        cases.push((missing, PdfSelectableTextErrorKind::MissingResource));
        let mut encoding = FontSpec::admitted("F0", &cmap);
        encoding.encoding = "WinAnsiEncoding";
        cases.push((encoding, PdfSelectableTextErrorKind::UnsupportedFont));
        let mut type0 = FontSpec::admitted("F0", &cmap);
        type0.type0_subtype = "Type1";
        cases.push((type0, PdfSelectableTextErrorKind::UnsupportedFont));
        let mut descendant = FontSpec::admitted("F0", &cmap);
        descendant.descendant_subtype = "CIDFontType0";
        cases.push((descendant, PdfSelectableTextErrorKind::UnsupportedFont));
        let mut gid = FontSpec::admitted("F0", &cmap);
        gid.cid_to_gid = "CustomMap";
        cases.push((gid, PdfSelectableTextErrorKind::UnsupportedFont));

        for (font, expected_kind) in cases {
            let pages = PdfPages::from_bytes(build_pdf(&[font], vec![bt(), tf("F0", 12), et()]))
                .expect("parse fixture");
            let error = pages.selectable_text(0).expect_err("font must refuse");
            assert_eq!(error.kind, expected_kind);
            assert_eq!(error.operation_index, Some(1));
            assert_eq!(error.font_resource.as_deref(), Some("F0"));
            assert!(!error.detail.contains("replacement"));
        }
    }

    #[test]
    fn page_state_resource_and_operand_bounds_are_typed() {
        let cmap = one_mapping_cmap("0041");
        let font = FontSpec::admitted("F0", &cmap);
        let pages = PdfPages::from_bytes(build_pdf(std::slice::from_ref(&font), Vec::new()))
            .expect("parse fixture");
        let out_of_range = pages.selectable_text(1).expect_err("page bound");
        assert_eq!(
            out_of_range.kind,
            PdfSelectableTextErrorKind::PageOutOfRange
        );
        assert_eq!(out_of_range.page_index, 1);
        assert_eq!(out_of_range.operation_index, None);

        let missing_font = PdfPages::from_bytes(build_pdf(
            std::slice::from_ref(&font),
            vec![bt(), tf("Missing", 12), et()],
        ))
        .expect("parse fixture")
        .selectable_text(0)
        .expect_err("missing resource");
        assert_eq!(
            missing_font.kind,
            PdfSelectableTextErrorKind::MissingResource
        );
        assert_eq!(missing_font.font_resource.as_deref(), Some("Missing"));

        let mut deep_state = vec![Operation::new("q", Vec::new()); MAX_GRAPHICS_STATE_DEPTH + 1];
        deep_state.extend(std::iter::repeat_n(
            Operation::new("Q", Vec::new()),
            MAX_GRAPHICS_STATE_DEPTH + 1,
        ));
        let state_error = PdfPages::from_bytes(build_pdf(std::slice::from_ref(&font), deep_state))
            .expect("parse fixture")
            .selectable_text(0)
            .expect_err("state bound");
        assert_eq!(state_error.kind, PdfSelectableTextErrorKind::LimitExceeded);
        assert_eq!(state_error.operation_index, Some(MAX_GRAPHICS_STATE_DEPTH));

        let oversized_tj = vec![Object::Integer(0); MAX_TJ_OPERANDS + 1];
        let tj_error = PdfPages::from_bytes(build_pdf(
            &[font],
            vec![
                bt(),
                tf("F0", 12),
                Operation::new("TJ", vec![Object::Array(oversized_tj)]),
                et(),
            ],
        ))
        .expect("parse fixture")
        .selectable_text(0)
        .expect_err("TJ bound");
        assert_eq!(tj_error.kind, PdfSelectableTextErrorKind::LimitExceeded);
        assert_eq!(tj_error.operation_index, Some(2));

        let cmap_stream = Stream::new(Dictionary::new(), vec![0; MAX_CMAP_ENCODED_BYTES + 1]);
        assert_eq!(
            decode_cmap_stream(&cmap_stream).unwrap_err().0,
            PdfSelectableTextErrorKind::LimitExceeded
        );
        assert!(ensure_entry_budget(MAX_CMAP_ENTRIES, 1).is_err());
        assert!(increment_be(&[0xff, 0xff], 1).is_none());
        assert_eq!(
            checked_page_tj_operand_count(MAX_PAGE_TJ_OPERANDS - 1, 1),
            Some(MAX_PAGE_TJ_OPERANDS)
        );
        assert_eq!(checked_page_tj_operand_count(MAX_PAGE_TJ_OPERANDS, 1), None);
        assert_eq!(checked_page_tj_operand_count(usize::MAX, 1), None);

        let doc = Document::with_version("1.7");
        let mut extractor = Extractor::new(&doc, (1, 0), 0);
        extractor.text = Some(TextObjectState::default());
        extractor.tj_operand_count = MAX_PAGE_TJ_OPERANDS;
        let page_tj_error = extractor
            .show_tj_array(7, &[Object::Array(vec![literal_string(&[0x00, 0x01])])])
            .expect_err("a string operand must count toward the page-global TJ limit");
        assert_eq!(
            page_tj_error.kind,
            PdfSelectableTextErrorKind::LimitExceeded
        );
        assert_eq!(page_tj_error.operation_index, Some(7));
        assert!(page_tj_error.detail.contains("1000000-operand"));
        assert!(page_tj_error.detail.contains("across TJ arrays"));
        assert_eq!(extractor.tj_operand_count, MAX_PAGE_TJ_OPERANDS);

        let mut inherited_cmap_dict = Dictionary::new();
        inherited_cmap_dict.set("UseCMap", Object::Name(b"Adobe-Identity-UCS".to_vec()));
        let inherited_cmap =
            Stream::new(inherited_cmap_dict, one_mapping_cmap("0041").into_bytes());
        let inherited_error = decode_cmap_stream(&inherited_cmap).unwrap_err();
        assert_eq!(
            inherited_error.0,
            PdfSelectableTextErrorKind::UnsupportedCMap
        );
        assert!(inherited_error.1.contains("/UseCMap"));
    }

    #[test]
    fn nonsemantic_graphics_allowlist_validates_every_operand_shape() {
        let valid = vec![
            Operation::new("q", Vec::new()),
            Operation::new("w", vec![1.into()]),
            Operation::new("J", vec![1.into()]),
            Operation::new("j", vec![2.into()]),
            Operation::new("d", vec![Object::Array(vec![1.into(), 2.into()]), 0.into()]),
            Operation::new("m", vec![0.into(), 0.into()]),
            Operation::new("l", vec![1.into(), 1.into()]),
            Operation::new(
                "c",
                vec![0.into(), 0.into(), 1.into(), 1.into(), 2.into(), 2.into()],
            ),
            Operation::new("re", vec![0.into(), 0.into(), 1.into(), 1.into()]),
            Operation::new("h", Vec::new()),
            Operation::new("S", Vec::new()),
            Operation::new("s", Vec::new()),
            Operation::new("f", Vec::new()),
            Operation::new("F", Vec::new()),
            Operation::new("f*", Vec::new()),
            Operation::new("n", Vec::new()),
            Operation::new("g", vec![0.into()]),
            Operation::new("G", vec![0.into()]),
            Operation::new("rg", vec![0.into(), 0.into(), 0.into()]),
            Operation::new("RG", vec![0.into(), 0.into(), 0.into()]),
            Operation::new("k", vec![0.into(), 0.into(), 0.into(), 1.into()]),
            Operation::new("K", vec![0.into(), 0.into(), 0.into(), 1.into()]),
            Operation::new("cs", vec![Object::Name(b"DeviceGray".to_vec())]),
            Operation::new("CS", vec![Object::Name(b"DeviceRGB".to_vec())]),
            Operation::new("sc", vec![0.into()]),
            Operation::new("SC", vec![0.into(), 0.into(), 0.into()]),
            Operation::new("scn", vec![0.into()]),
            Operation::new("SCN", vec![0.into(), 0.into(), 0.into()]),
            Operation::new("ri", vec![Object::Name(b"RelativeColorimetric".to_vec())]),
            Operation::new("Q", Vec::new()),
        ];
        let page = PdfPages::from_bytes(build_pdf(&[], valid))
            .expect("parse valid graphics-operator fixture")
            .selectable_text(0)
            .expect("accept exact nonsemantic graphics operand grammar");
        assert!(page.runs.is_empty());

        let malformed = vec![
            ("q", vec![0.into()]),
            ("Q", vec![0.into()]),
            ("w", vec![Object::Name(b"wide".to_vec())]),
            ("J", vec![Object::Real(1.5)]),
            ("j", vec![3.into()]),
            ("d", vec![Object::Array(vec![0.into(), 0.into()]), 0.into()]),
            ("m", vec![0.into()]),
            ("l", vec![0.into(), Object::Name(b"y".to_vec())]),
            ("c", vec![0.into(), 0.into(), 0.into(), 0.into(), 0.into()]),
            ("re", vec![0.into(), 0.into(), 0.into()]),
            ("h", vec![0.into()]),
            ("S", vec![0.into()]),
            ("s", vec![0.into()]),
            ("f", vec![0.into()]),
            ("F", vec![0.into()]),
            ("f*", vec![0.into()]),
            ("n", vec![0.into()]),
            ("g", Vec::new()),
            ("G", vec![0.into(), 0.into()]),
            ("rg", vec![0.into(), 0.into()]),
            ("RG", vec![0.into(), 0.into(), 0.into(), 0.into()]),
            ("k", vec![0.into(), 0.into(), 0.into()]),
            ("K", vec![0.into(), 0.into(), 0.into(), 0.into(), 0.into()]),
            ("cs", vec![0.into()]),
            ("cs", vec![Object::Name(b"Pattern".to_vec())]),
            (
                "CS",
                vec![
                    Object::Name(b"DeviceGray".to_vec()),
                    Object::Name(b"DeviceRGB".to_vec()),
                ],
            ),
            ("sc", vec![0.into(), 0.into()]),
            ("SC", vec![Object::Name(b"Pattern0".to_vec())]),
            ("scn", vec![Object::Name(b"Pattern0".to_vec())]),
            ("scn", vec![0.into(), 0.into()]),
            (
                "SCN",
                vec![0.into(), 0.into(), Object::Name(b"Pattern0".to_vec())],
            ),
            ("ri", Vec::new()),
        ];
        for (operator, operands) in malformed {
            let operations = if operator == "Q" {
                vec![
                    Operation::new("q", Vec::new()),
                    Operation::new(operator, operands),
                ]
            } else {
                vec![Operation::new(operator, operands)]
            };
            let result = PdfPages::from_bytes(build_pdf(&[], operations))
                .unwrap_or_else(|cause| panic!("parse malformed {operator} fixture: {cause}"))
                .selectable_text(0);
            let error = result.expect_err("malformed graphics operands must refuse");
            assert_eq!(
                error.kind,
                PdfSelectableTextErrorKind::MalformedTextOperand,
                "operator {operator}: {error}"
            );
            assert_eq!(
                error.operation_index,
                Some(usize::from(operator == "Q")),
                "operator {operator}: {error}"
            );
            assert!(
                error.detail.contains(operator),
                "operator {operator}: {error}"
            );
        }
    }

    #[test]
    fn annotation_appearances_and_non_link_subtypes_fail_closed() {
        let mut doc = Document::with_version("1.7");
        let link_id = doc.add_object(dictionary! {
            "Type" => "Annot",
            "Subtype" => "Link",
            "Rect" => vec![0.into(), 0.into(), 10.into(), 10.into()],
        });
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Annots" => vec![link_id.into()],
        });
        validate_page_annotations(&doc, page_id).expect("exact no-appearance link is admitted");

        let appearance_id = doc.add_object(Stream::new(Dictionary::new(), b"BT ET".to_vec()));
        doc.get_object_mut(link_id)
            .and_then(Object::as_dict_mut)
            .expect("link dictionary")
            .set("AP", dictionary! { "N" => appearance_id });
        let appearance_error = validate_page_annotations(&doc, page_id).unwrap_err();
        assert!(appearance_error.contains("/AP appearance"));

        let text_id = doc.add_object(dictionary! {
            "Type" => "Annot",
            "Subtype" => "Text",
            "Rect" => vec![0.into(), 0.into(), 10.into(), 10.into()],
        });
        doc.get_object_mut(page_id)
            .and_then(Object::as_dict_mut)
            .expect("page dictionary")
            .set("Annots", vec![text_id.into()]);
        let subtype_error = validate_page_annotations(&doc, page_id).unwrap_err();
        assert!(subtype_error.contains("not an exact /Link"));
    }

    #[test]
    fn form_xobjects_and_unknown_text_object_operators_refuse_instead_of_omitting_text() {
        let cmap = one_mapping_cmap("0041");
        let font = FontSpec::admitted("F0", &cmap);
        let unknown = PdfPages::from_bytes(build_pdf(
            std::slice::from_ref(&font),
            vec![
                bt(),
                tf("F0", 12),
                Operation::new("BMC", vec![Object::Name(b"Span".to_vec())]),
                et(),
            ],
        ))
        .expect("parse fixture")
        .selectable_text(0)
        .expect_err("unknown operator");
        assert_eq!(
            unknown.kind,
            PdfSelectableTextErrorKind::UnsupportedTextOperator
        );
        assert_eq!(unknown.operation_index, Some(2));

        // A missing Do resource also fails closed with operation context. Form
        // construction itself is covered by the same subtype branch.
        let missing_do = PdfPages::from_bytes(build_pdf(
            &[font],
            vec![Operation::new("Do", vec![Object::Name(b"Fm0".to_vec())])],
        ))
        .expect("parse fixture")
        .selectable_text(0)
        .expect_err("missing XObject");
        assert_eq!(missing_do.kind, PdfSelectableTextErrorKind::MissingResource);
        assert_eq!(missing_do.operation_index, Some(0));

        let malformed_marked_content = PdfPages::from_bytes(build_pdf(
            &[FontSpec::admitted("F0", &one_mapping_cmap("0041"))],
            vec![Operation::new(
                "BDC",
                vec![
                    Object::Name(b"Span".to_vec()),
                    dictionary! { "ActualText" => Object::string_literal("fabricated") }.into(),
                ],
            )],
        ))
        .expect("parse marked-content fixture")
        .selectable_text(0)
        .expect_err("ActualText without the exact UTF-16BE shape must refuse");
        assert_eq!(
            malformed_marked_content.kind,
            PdfSelectableTextErrorKind::MalformedTextOperand
        );
        assert_eq!(malformed_marked_content.operation_index, Some(0));

        let actual_text = PdfPages::from_bytes(build_pdf(
            &[FontSpec::admitted("F0", &one_mapping_cmap("0041"))],
            vec![
                bt(),
                tf("F0", 12),
                tm(10, 20),
                Operation::new(
                    "BDC",
                    vec![
                        Object::Name(b"Span".to_vec()),
                        dictionary! {
                            "ActualText" => hex_string(&[0xfe, 0xff, 0x00, b'm', 0x00, b'f']),
                            "MTDTExpressionKind" => Object::Name(b"Dynamic".to_vec()),
                        }
                        .into(),
                    ],
                ),
                Operation::new("TJ", vec![Object::Array(vec![hex_string(&[0, 1])])]),
                Operation::new("EMC", Vec::new()),
                et(),
            ],
        ))
        .expect("parse exact MTDT ActualText fixture")
        .selectable_text(0)
        .expect("exact MTDT ActualText must be selectable");
        assert_eq!(actual_text.runs.len(), 1);
        assert_eq!(actual_text.runs[0].unicode, "mf");
        assert_eq!(
            actual_text.runs[0].unicode_by_code,
            vec![Some("mf".to_owned())]
        );
        assert!(actual_text.runs[0].unicode_complete);
        assert!(actual_text.runs[0].identity_is_valid());
    }

    fn assert_real_lilypond_pdf_contract(bytes: Vec<u8>, exact_golden: bool) {
        let pages = PdfPages::from_bytes(bytes).expect("parse retained LilyPond bytes");
        if exact_golden {
            assert_eq!(pages.len(), 1);
        }
        let mut type1c_run_count = 0usize;
        let mut opaque_code_count = 0usize;
        let mut unicode_scalar_count = 0usize;
        for page_index in 0..pages.len() {
            let first = pages
                .selectable_text(page_index)
                .expect("extract real LilyPond selectable text");
            let second = pages
                .selectable_text(page_index)
                .expect("repeat real LilyPond selectable text");
            assert_eq!(first, second, "page {page_index} replay drifted");
            assert_eq!(first.schema, PDF_SELECTABLE_TEXT_SCHEMA_V2);
            if exact_golden {
                assert_eq!(
                    first.source_sha256,
                    [
                        0x45, 0xf7, 0x61, 0x95, 0x3b, 0x23, 0x07, 0x20, 0xae, 0xa5, 0xa6, 0x86,
                        0xd6, 0x50, 0x4a, 0x55, 0x47, 0xae, 0x46, 0xd6, 0xc4, 0x0e, 0x97, 0x24,
                        0x83, 0x47, 0xd5, 0x4d, 0x12, 0x92, 0xbd, 0x68,
                    ]
                );
                assert_eq!(
                    first.identity_sha256,
                    [
                        0x55, 0x87, 0x61, 0x98, 0x7f, 0xa1, 0xb9, 0xc6, 0xe3, 0xdf, 0xde, 0x34,
                        0xcc, 0x81, 0xe4, 0xf3, 0x01, 0x2e, 0xcc, 0x2b, 0x18, 0xcf, 0x22, 0x0f,
                        0x9d, 0x0b, 0x92, 0xf5, 0x73, 0x50, 0xbd, 0x1d,
                    ]
                );
                assert_eq!(first.runs.len(), 555);
                let expected_prefix = [
                    (67, "C", "C", 84.699_203_491_210_94),
                    (108, "l", "l", 98.305_852_016_622_75),
                    (97, "a", "a", 104.428_907_253_132_42),
                    (114, "r", "r", 115.143_752_916_467_32),
                    (101, "e", "e", 124.157_817_640_246_14),
                    (169, "quotesingle", "'", 134.192_980_214_498_2),
                    (115, "s", "s", 138.444_930_611_210_54),
                    (32, "space", " ", 147.118_761_528_693_48),
                    (68, "D", "D", 152.051_180_369_682_38),
                    (114, "r", "r", 166.508_223_466_779_04),
                    (97, "a", "a", 175.522_288_190_557_87),
                    (103, "g", "g", 186.237_133_853_892_77),
                ];
                for (run, (code, glyph_name, unicode, x)) in first.runs.iter().zip(expected_prefix)
                {
                    assert_eq!(run.operation_index, 8);
                    assert_eq!(run.font_resource, "R7");
                    assert_eq!(run.source_codes, [code]);
                    assert_eq!(run.glyph_names, [Some(glyph_name.to_owned())]);
                    assert_eq!(run.unicode_by_code, [Some(unicode.to_owned())]);
                    assert!((run.position.text_matrix[4] - x).abs() < 1.0e-9);
                    assert_eq!(run.position.text_matrix[5], 809.922_973_632_812_5);
                }
                let opaque = &first.runs[139];
                assert_eq!(
                    opaque.identity_sha256,
                    [
                        0xe8, 0xb2, 0x88, 0x7f, 0xa5, 0x28, 0xdc, 0x16, 0x18, 0x78, 0xb7, 0x23,
                        0x7d, 0x6e, 0x34, 0xe6, 0x89, 0x10, 0x17, 0x26, 0xea, 0x3a, 0x5e, 0x7a,
                        0x18, 0xda, 0xaa, 0x9f, 0x1b, 0xa7, 0x92, 0xbf,
                    ]
                );
                assert_eq!(opaque.operation_index, 116);
                assert_eq!(opaque.font_resource, "R12");
                assert_eq!(
                    opaque.font_mapping_sha256,
                    [
                        0x83, 0x30, 0x1f, 0xd5, 0x05, 0xdc, 0xfc, 0xab, 0x33, 0x0b, 0xaa, 0x94,
                        0xd0, 0xc6, 0x0f, 0x00, 0x9b, 0x19, 0x6d, 0xbc, 0xae, 0x9c, 0xa3, 0x74,
                        0x08, 0x9e, 0xd3, 0xde, 0x48, 0x2a, 0x4e, 0x1f,
                    ]
                );
                assert_eq!(opaque.source_codes, [0]);
                assert_eq!(opaque.glyph_names, [Some("noteheads.s2".to_owned())]);
                assert_eq!(opaque.unicode_by_code, [None]);
                assert!(!opaque.unicode_complete);
                assert!(opaque.unicode.is_empty());
                assert_eq!(opaque.position.text_matrix[4], 441.085_998_535_156_25);
                assert_eq!(opaque.position.text_matrix[5], 715.773_986_816_406_3);
            }
            for run in &first.runs {
                assert_eq!(run.source_codes.len(), run.glyph_names.len());
                assert_eq!(run.source_codes.len(), run.unicode_by_code.len());
                assert_eq!(
                    run.unicode_complete,
                    run.unicode_by_code.iter().all(Option::is_some)
                );
                unicode_scalar_count += run.unicode.chars().count();
                if run.font_encoding == PdfSelectableTextEncodingV2::Type1cWinAnsiDifferences {
                    type1c_run_count += 1;
                    assert_eq!(run.code_width_bytes, 1);
                    assert!(run.glyph_names.iter().all(Option::is_some));
                    opaque_code_count += run
                        .unicode_by_code
                        .iter()
                        .filter(|item| item.is_none())
                        .count();
                }
            }
        }
        if exact_golden {
            assert_eq!(type1c_run_count, 555);
            assert_eq!(opaque_code_count, 322);
            assert_eq!(unicode_scalar_count, 233);
            return;
        }
        assert!(
            type1c_run_count > 0,
            "fixture exercised no Type1C text runs"
        );
        assert!(
            opaque_code_count > 0,
            "fixture exercised no ledgered opaque music glyph"
        );
        assert!(
            unicode_scalar_count > 0,
            "fixture exercised no mapped text glyph"
        );
    }

    #[test]
    fn checked_in_lilypond_pdf_has_exact_text_and_opaque_music_golden() {
        assert_real_lilypond_pdf_contract(
            include_bytes!("../tests/fixtures/lilypond_selectable_text.pdf").to_vec(),
            true,
        );
    }

    #[test]
    #[ignore = "requires FOCR_TEST_LILYPOND_PDF naming a real LilyPond-generated PDF"]
    fn real_lilypond_pdf_type1c_text_is_deterministic_and_loss_ledgered() {
        let path = std::env::var_os("FOCR_TEST_LILYPOND_PDF")
            .expect("FOCR_TEST_LILYPOND_PDF must name a LilyPond-generated PDF");
        let bytes = std::fs::read(&path).expect("read LilyPond PDF once");
        assert_real_lilypond_pdf_contract(bytes, false);
    }

    #[test]
    #[ignore = "requires FOCR_TEST_LILYPOND_WINANSI_NAME_PDF naming the exact retained regression PDF"]
    fn retained_lilypond_named_winansi_pdf_renders_and_extracts_text() {
        const SOURCE_SHA256: &str =
            "cb8a40edea42c1987a4cede3066653e25cd64c79799edb6591566788cacd1e3f";
        let path = std::env::var_os("FOCR_TEST_LILYPOND_WINANSI_NAME_PDF").expect(
            "FOCR_TEST_LILYPOND_WINANSI_NAME_PDF must name the exact retained LilyPond PDF",
        );
        let bytes = std::fs::read(&path).expect("read exact retained LilyPond PDF once");
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), SOURCE_SHA256);

        let pages = PdfPages::from_bytes(bytes).expect("parse exact retained LilyPond PDF");
        assert_eq!(pages.len(), 1);
        let first_image = pages
            .render(0)
            .expect("render exact named-WinAnsi LilyPond page natively")
            .to_luma8();
        let second_image = pages
            .render(0)
            .expect("repeat exact named-WinAnsi LilyPond render")
            .to_luma8();
        assert_eq!(first_image.dimensions(), second_image.dimensions());
        assert_eq!(first_image.as_raw(), second_image.as_raw());
        assert!(
            first_image.as_raw().iter().any(|&pixel| pixel < 250),
            "exact LilyPond page rendered blank"
        );

        let first_text = pages
            .selectable_text(0)
            .expect("extract exact named-WinAnsi LilyPond selectable text");
        let second_text = pages
            .selectable_text(0)
            .expect("repeat exact named-WinAnsi LilyPond selectable text");
        assert_eq!(first_text, second_text);
        assert_eq!(first_text.schema, PDF_SELECTABLE_TEXT_SCHEMA_V2);
        assert!(
            first_text.runs.iter().any(|run| {
                run.font_encoding == PdfSelectableTextEncodingV2::Type1cWinAnsiDifferences
            }),
            "exact LilyPond page exercised no named-WinAnsi Type1C run"
        );
    }
}
