//! Native PDF page rasterization in pure, memory-safe Rust (no FFI).
//!
//! `focr ocr file.pdf` renders each PDF page to an [`image::DynamicImage`] and
//! feeds it through the same preprocess + OCR pipeline a PNG/JPG would take, so
//! a PDF no longer has to be rasterized out of band (poppler / `pdftoppm`).
//!
//! ## Scope: the scanned-image fast path
//!
//! The overwhelming majority of OCR-input PDFs are *scans* — one full-page image
//! XObject per page. This module extracts that image and decodes it to RGB/gray
//! with the codecs the project already trusts ([`image`]'s JPEG via `zune-jpeg`,
//! `flate2`/`miniz_oxide` for `FlateDecode`, the pure-Rust [`fax`] crate for
//! CCITT Group 4). Everything here is pure Rust with no C/C++ FFI, matching the
//! project's no-FFI doctrine. `lopdf` (with `default-features` off) is the new
//! container parser; `fax` is already in the lock graph. We own the image-codec
//! decode dispatch below, so the parser is the only borrowed piece.
//!
//! ## Honest limits
//!
//! Two image codecs have **no** production-quality pure-Rust decoder and are
//! reported as a clear error rather than guessed at:
//! * `JPXDecode` (JPEG 2000) — every working decoder wraps OpenJPEG (C / FFI).
//! * `JBIG2Decode` — only C (`jbig2dec`) bindings exist.
//!
//! Born-digital PDFs whose pages are *vector / text* content (no full-page image
//! XObject) also fall outside this fast path: rasterizing arbitrary PDF vector
//! graphics needs a full content-stream interpreter + glyph rasterizer, tracked
//! separately. Such pages, and the two unsupported codecs, surface as
//! [`FocrError::InputDecode`] naming exactly what was unsupported, so the caller
//! can rasterize that PDF out of band and retry.

use std::path::Path;

use image::{DynamicImage, GrayImage, ImageBuffer, RgbImage};
use lopdf::xobject::PdfImage;
use lopdf::{Document, Object, ObjectId};

use crate::error::{FocrError, FocrResult};

/// The 5-byte header every PDF begins with (`%PDF-`).
const PDF_MAGIC: &[u8] = b"%PDF-";

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
    use std::io::Read;
    matches!(file.read_exact(&mut head), Ok(())) && head == PDF_MAGIC
}

/// A lazily-rendered PDF: the parsed document plus its page object ids in order.
///
/// Pages are rendered one at a time via [`PdfPages::render`] so a 600-page book
/// never materializes 600 rasters at once — the OCR driver pulls one page,
/// recognizes it, and drops it before the next.
pub struct PdfPages {
    doc: Document,
    /// Page object ids in 1-based page order (the value of `get_pages`).
    pages: Vec<ObjectId>,
}

impl PdfPages {
    /// Parse the PDF at `path`. Does not render any page yet.
    ///
    /// # Errors
    /// [`FocrError::InputDecode`] if the file cannot be parsed as a PDF.
    pub fn open(path: &Path) -> FocrResult<Self> {
        let doc = Document::load(path)
            .map_err(|e| FocrError::InputDecode(format!("parse PDF {}: {e}", path.display())))?;
        let pages: Vec<ObjectId> = doc.get_pages().into_values().collect();
        if pages.is_empty() {
            return Err(FocrError::InputDecode(format!(
                "PDF {} has no pages",
                path.display()
            )));
        }
        Ok(Self { doc, pages })
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

    /// Render page `idx` (0-based) to a [`DynamicImage`], applying the page's
    /// `/Rotate`.
    ///
    /// Picks the largest image XObject on the page (the main scan, ignoring small
    /// preview/thumbnail images) and decodes it. See the module docs for the
    /// supported codecs.
    ///
    /// # Errors
    /// [`FocrError::InputDecode`] if the page has no decodable full-page image
    /// (vector/text page), or its image uses an unsupported codec
    /// (`JPXDecode` / `JBIG2Decode`) or color space.
    pub fn render(&self, idx: usize) -> FocrResult<DynamicImage> {
        let page_id = *self.pages.get(idx).ok_or_else(|| {
            FocrError::InputDecode(format!(
                "PDF page index {idx} out of range ({})",
                self.len()
            ))
        })?;

        let images = self.doc.get_page_images(page_id).map_err(|e| {
            FocrError::InputDecode(format!("read images on PDF page {}: {e}", idx + 1))
        })?;

        // Fast path: the page's scan is the largest image XObject. Skip the
        // small preview thumbnails some producers embed alongside the main scan.
        let main = images
            .iter()
            .max_by_key(|im| (im.width as i128) * (im.height as i128))
            .ok_or_else(|| {
                FocrError::InputDecode(format!(
                    "PDF page {} has no image XObject (vector/text PDFs are not supported by the \
                     native fast path; rasterize the PDF out of band, e.g. with pdftoppm, and pass \
                     the page images)",
                    idx + 1
                ))
            })?;

        let decoded = decode_image_xobject(&self.doc, main)
            .map_err(|e| FocrError::InputDecode(format!("PDF page {}: {e}", idx + 1)))?;

        Ok(apply_rotation(decoded, page_rotation(&self.doc, page_id)))
    }
}

/// Decode one image XObject to RGB/gray, dispatching on its terminal `/Filter`.
fn decode_image_xobject(doc: &Document, img: &PdfImage) -> Result<DynamicImage, String> {
    let width = u32::try_from(img.width).map_err(|_| "negative image width".to_string())?;
    let height = u32::try_from(img.height).map_err(|_| "negative image height".to_string())?;
    if width == 0 || height == 0 {
        return Err("zero image dimension".to_string());
    }
    // Bound the DECLARED dimensions before any per-pixel allocation. A crafted PDF
    // can claim a gigapixel image and make a downstream `width*height` product
    // overflow `usize` (the CMYK guard) or reserve hundreds of TB (a raster
    // `Vec`). Real document scans are far below this (a 600-DPI A0 page is ~0.5
    // Gpx). `u32 * u32` is computed in `u64` so the check itself cannot overflow.
    const MAX_PIXELS: u64 = 1 << 30; // 1 Gpx
    if u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(format!(
            "image dimensions {width}x{height} exceed the {MAX_PIXELS}-pixel maximum"
        ));
    }
    let bpc = img.bits_per_component.unwrap_or(8);
    let color_space = img.color_space.as_deref().unwrap_or("DeviceRGB");
    let filters = img.filters.clone().unwrap_or_default();
    let terminal = filters.last().map(String::as_str).unwrap_or("");

    // The image codecs (DCT/CCITT) consume `img.content` verbatim — the RAW stream,
    // with NO filters applied (lopdf's `get_page_images` does not decode). So a
    // multi-filter chain whose codec is preceded by an ASCII/Flate filter would
    // feed still-encoded bytes to the codec. Reject such chains with an accurate
    // message rather than a misleading "decode failed". (The raw-sample branch is
    // chain-safe: `decompressed_content` walks the whole filter chain.)
    let chained = filters.len() > 1;

    match terminal {
        "DCTDecode" if chained => Err(format!(
            "image filter chain {filters:?} ending in DCTDecode is unsupported (only a \
             sole DCTDecode filter); rasterize this PDF out of band and retry"
        )),
        // `content` is already the raw JPEG byte stream.
        "DCTDecode" => image::load_from_memory_with_format(img.content, image::ImageFormat::Jpeg)
            .map_err(|e| format!("JPEG (DCTDecode) decode failed: {e}")),

        // No pure-Rust decoder exists for either; be honest rather than wrong.
        "JPXDecode" => Err(
            "image uses JPXDecode (JPEG 2000), which has no pure-Rust decoder; \
                            rasterize this PDF out of band and retry"
                .to_string(),
        ),
        "JBIG2Decode" => Err("image uses JBIG2Decode, which has no pure-Rust decoder; \
                              rasterize this PDF out of band and retry"
            .to_string()),

        "CCITTFaxDecode" if chained => Err(format!(
            "image filter chain {filters:?} ending in CCITTFaxDecode is unsupported (only a \
             sole CCITTFaxDecode filter); rasterize this PDF out of band and retry"
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
            let high: Vec<u8> = samples.chunks_exact(2).map(|c| c[0]).collect();
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
    for px in samples.chunks_exact(4).take(pixels) {
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

    #[test]
    fn looks_like_pdf_by_extension() {
        assert!(looks_like_pdf(Path::new("/x/y/scan.pdf")));
        assert!(looks_like_pdf(Path::new("/x/y/scan.PDF")));
        assert!(!looks_like_pdf(Path::new("/x/y/page.png")));
        // Missing file, no .pdf extension -> not a PDF (no panic).
        assert!(!looks_like_pdf(Path::new("/no/such/file.bin")));
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
    /// `get_page_images` + the `DCTDecode` dispatch + the JPEG decoder — the
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

    /// A page with no image XObject (a vector/text page) must surface the precise,
    /// actionable [`FocrError::InputDecode`] rather than rendering garbage.
    #[test]
    fn render_image_free_page_errors_clearly() {
        let path = build_single_page_pdf(None);
        let pages = PdfPages::open(&path).expect("open synthesized pdf");
        assert_eq!(pages.len(), 1);
        let err = pages.render(0).expect_err("vector page must error");
        let msg = err.to_string();
        assert!(
            msg.contains("no image XObject"),
            "expected an actionable no-image message, got: {msg}"
        );
        let _ = std::fs::remove_file(&path);
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
