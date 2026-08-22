//! Tall-capture strip routing and low-yield detection (GH #15).
//!
//! Full-page browser captures (Awesome Screenshot and friends) arrive as
//! extremely tall images — a 1:4 to 1:10 page in one PNG. A single model pass
//! over such an image squashes glyphs far below legibility (the global view
//! is aspect-preserving into a 1024² pad, so a 494×2000 page renders its text
//! at ~2px), and the layout pass then tends to classify the whole page as one
//! figure — returning near-empty markdown with exit 0. The recipe that works
//! is the one users discover by hand: cut the capture into sane-aspect
//! horizontal strips, OCR each, and concatenate. This module does that
//! automatically for extreme aspect ratios, cutting on the blankest row near
//! each nominal boundary so text lines are not severed.
//!
//! The same incident motivated the low-yield check: a page-sized input that
//! produces a handful of characters is almost certainly a capture failure,
//! and it must at minimum be *detectable* (`low_yield` on `run_complete`, a
//! stderr warning) and optionally fatal (`--fail-on-low-yield`).
//!
//! Scope: the router only fires for the plain document-OCR task on single
//! (non-PDF) images. Specialty tasks (music, VQA/describe, formula, …) keep
//! the one-pass path — their outputs are not line-concatenable.

use image::{DynamicImage, GenericImageView};

use crate::native_engine::{LayoutSpan, RecognizedDocument};

/// Height:width ratio at which a single-image run is routed through strip
/// tiling. 3:1 is comfortably beyond any normal document page (US Letter is
/// ~1.29, A4 ~1.41, even a legal-length scan stays under 2) while catching
/// the 1:4-and-taller browser captures from the report.
pub const TALL_ASPECT_TRIGGER: f64 = 3.0;

/// Nominal strip height, in multiples of the image width. Square-ish strips
/// keep each model view near the aspect the engine's certified global view
/// actually sees, so the per-strip glyph size is as large as tiling can make
/// it without cutting more often than needed.
const STRIP_HEIGHT_WIDTHS: f64 = 1.0;

/// Cut-search half-window as a fraction of the nominal strip height: the cut
/// snaps to the blankest row within ±this window of the nominal boundary.
const CUT_SEARCH_WINDOW_FRAC: f64 = 0.125;

/// Floor for a strip height so pathological widths cannot produce confetti.
const MIN_STRIP_HEIGHT: u32 = 128;

/// A run is low-yield when a sufficiently large input produced fewer than
/// this many text characters per megapixel. A dense text page yields well
/// over 1,000 chars/MP; the incident capture yielded ~15. The margin between
/// those is wide, so the constant sits near the failure end to keep sparse
/// but legitimate inputs (posters, slides with one line) from tripping it.
pub const LOW_YIELD_CHARS_PER_MEGAPIXEL: f64 = 50.0;

/// Inputs smaller than this many megapixels are never judged: a small crop
/// with a word on it is fine and cannot be a "page-sized" silent failure.
pub const LOW_YIELD_MIN_MEGAPIXELS: f64 = 0.5;

/// One planned horizontal strip: pixel rows `top..bottom` of the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Strip {
    pub top: u32,
    pub bottom: u32,
}

/// Whether an image's aspect ratio is extreme enough to route through strips.
#[must_use]
pub fn is_tall(width: u32, height: u32) -> bool {
    width > 0 && f64::from(height) / f64::from(width) >= TALL_ASPECT_TRIGGER
}

/// Row "ink" profile: per source row, the summed darkness (`255 − luma`).
/// Blank paper rows sit near zero; rows crossing text or UI chrome are large.
#[must_use]
pub fn ink_profile(img: &DynamicImage) -> Vec<u64> {
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();
    let mut profile = vec![0u64; h as usize];
    for y in 0..h {
        let mut ink = 0u64;
        for x in 0..w {
            ink += u64::from(255 - gray.get_pixel(x, y).0[0]);
        }
        profile[y as usize] = ink;
    }
    profile
}

/// The blankest row in `lo..=hi` (minimum ink; ties resolved toward
/// `nominal` so cuts stay as even as possible).
fn blankest_row(profile: &[u64], lo: u32, hi: u32, nominal: u32) -> u32 {
    let mut best = nominal;
    let mut best_ink = u64::MAX;
    for row in lo..=hi {
        let ink = profile[row as usize];
        let better =
            ink < best_ink || (ink == best_ink && row.abs_diff(nominal) < best.abs_diff(nominal));
        if better {
            best = row;
            best_ink = ink;
        }
    }
    best
}

/// Plan the strips for a `width × height` image given its row ink profile.
///
/// Nominal cuts fall every [`STRIP_HEIGHT_WIDTHS`]·width rows; each snaps to
/// the blankest row within the search window so a cut lands in an inter-line
/// gap rather than through glyphs. The final strip absorbs a short tail
/// (under half a nominal strip) instead of producing a runt. The plan always
/// tiles `0..height` exactly: no gaps, no overlap.
#[must_use]
pub fn plan_strips(width: u32, height: u32, profile: &[u64]) -> Vec<Strip> {
    debug_assert_eq!(profile.len(), height as usize);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let nominal = ((f64::from(width) * STRIP_HEIGHT_WIDTHS) as u32).max(MIN_STRIP_HEIGHT);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let window = ((f64::from(nominal) * CUT_SEARCH_WINDOW_FRAC) as u32).max(1);

    let mut strips = Vec::new();
    let mut top = 0u32;
    // Keep cutting while the remainder is tall enough that the tail after
    // this cut is still at least half a nominal strip.
    while height - top > nominal + nominal / 2 {
        let target = top + nominal;
        let lo = target.saturating_sub(window).max(top + 1);
        let hi = (target + window).min(height - 1);
        let cut = blankest_row(profile, lo, hi, target);
        strips.push(Strip { top, bottom: cut });
        top = cut;
    }
    strips.push(Strip {
        top,
        bottom: height,
    });
    strips
}

/// Crop the planned strips out of the source image, in plan order.
#[must_use]
pub fn cut_strips(img: &DynamicImage, plan: &[Strip]) -> Vec<DynamicImage> {
    let (w, _) = img.dimensions();
    plan.iter()
        .map(|s| img.crop_imm(0, s.top, w, s.bottom - s.top))
        .collect()
}

/// Merge per-strip recognitions into one document: markdown concatenated in
/// reading order (blank line between non-empty strips), layout boxes
/// translated down by each strip's `top` offset so they address the source
/// image's pixel space.
#[must_use]
pub fn merge_documents(parts: Vec<(RecognizedDocument, u32)>) -> RecognizedDocument {
    let mut markdown_parts: Vec<String> = Vec::new();
    let mut layout: Vec<LayoutSpan> = Vec::new();
    for (doc, top) in parts {
        let body = doc.markdown.trim();
        if !body.is_empty() {
            markdown_parts.push(body.to_string());
        }
        let dy = i64::from(top);
        for mut span in doc.layout {
            for b in &mut span.boxes {
                b[1] += dy;
                b[3] += dy;
            }
            layout.push(span);
        }
    }
    RecognizedDocument {
        markdown: markdown_parts.join("\n\n"),
        layout,
    }
}

/// A flagged low-yield result: the run "succeeded" but a large input
/// produced almost no text — the silent-failure signature from GH #15.
#[derive(Debug, Clone, PartialEq)]
pub struct LowYield {
    /// Characters of recognized text (markup and whitespace excluded).
    pub yield_chars: usize,
    /// Source image area in megapixels.
    pub input_megapixels: f64,
}

/// Count the characters of actual recognized text in a markdown body:
/// image placeholders (`![](…)`) contribute nothing, and whitespace is not
/// text. This is what the low-yield ratio is computed over, so a page whose
/// only output is figure chrome scores as empty.
#[must_use]
pub fn text_chars(markdown: &str) -> usize {
    let mut rest = markdown;
    let mut count = 0usize;
    while let Some(start) = rest.find("![") {
        count += rest[..start].chars().filter(|c| !c.is_whitespace()).count();
        let after = &rest[start..];
        // Skip the `![alt](target)` token when well-formed; otherwise treat
        // the `![` literally and continue after it.
        if let Some(close) = after.find(')') {
            rest = &after[close + 1..];
        } else {
            count += after.chars().filter(|c| !c.is_whitespace()).count();
            rest = "";
        }
    }
    count + rest.chars().filter(|c| !c.is_whitespace()).count()
}

/// Judge a completed single-image run: `Some(LowYield)` when the input was
/// at least [`LOW_YIELD_MIN_MEGAPIXELS`] and the recognized text density
/// fell below [`LOW_YIELD_CHARS_PER_MEGAPIXEL`].
#[must_use]
pub fn low_yield_assessment(markdown: &str, width: u32, height: u32) -> Option<LowYield> {
    let megapixels = f64::from(width) * f64::from(height) / 1_000_000.0;
    if megapixels < LOW_YIELD_MIN_MEGAPIXELS {
        return None;
    }
    let chars = text_chars(markdown);
    #[allow(clippy::cast_precision_loss)]
    let density = chars as f64 / megapixels;
    (density < LOW_YIELD_CHARS_PER_MEGAPIXEL).then_some(LowYield {
        yield_chars: chars,
        input_megapixels: megapixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma, Rgb};

    fn white_image_with_dark_bands(width: u32, height: u32, bands: &[(u32, u32)]) -> DynamicImage {
        let img = ImageBuffer::from_fn(width, height, |_, y| {
            if bands.iter().any(|&(top, bottom)| y >= top && y < bottom) {
                Rgb([0u8, 0, 0])
            } else {
                Rgb([255u8, 255, 255])
            }
        });
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn is_tall_trigger_boundary() {
        assert!(!is_tall(494, 494));
        assert!(!is_tall(500, 1499)); // 2.998 — just under
        assert!(is_tall(500, 1500)); // exactly 3.0
        assert!(is_tall(494, 2000)); // the incident capture
        assert!(!is_tall(2000, 494)); // wide is not routed
        assert!(!is_tall(0, 100)); // degenerate width never divides by zero
    }

    #[test]
    fn ink_profile_separates_text_rows_from_blank_rows() {
        let img = white_image_with_dark_bands(64, 32, &[(10, 12)]);
        let profile = ink_profile(&img);
        assert_eq!(profile.len(), 32);
        assert_eq!(profile[0], 0);
        assert_eq!(profile[10], 64 * 255);
        assert_eq!(profile[11], 64 * 255);
        assert_eq!(profile[31], 0);
    }

    #[test]
    fn plan_tiles_full_height_without_gaps_or_overlap() {
        for (w, h) in [(494u32, 2000u32), (200, 4000), (1000, 3200), (128, 12000)] {
            let profile = vec![0u64; h as usize];
            let plan = plan_strips(w, h, &profile);
            assert!(!plan.is_empty(), "{w}x{h}");
            assert_eq!(plan[0].top, 0, "{w}x{h}");
            assert_eq!(plan.last().unwrap().bottom, h, "{w}x{h}");
            for pair in plan.windows(2) {
                assert_eq!(pair[0].bottom, pair[1].top, "{w}x{h}: gap/overlap");
            }
            for s in &plan {
                assert!(s.bottom > s.top, "{w}x{h}: degenerate strip {s:?}");
            }
        }
    }

    #[test]
    fn plan_cuts_snap_to_blank_rows() {
        // 200 wide, 700 tall → nominal cut at y=200 (window ±25). Rows are
        // dark everywhere except a blank band at 190..193, so the cut must
        // land inside that band rather than at the nominal 200.
        let (w, h) = (200u32, 700u32);
        let dark = 200u64 * 255;
        let mut profile = vec![dark; h as usize];
        profile[190..193].fill(0);
        let plan = plan_strips(w, h, &profile);
        assert!(
            (190..193).contains(&(plan[0].bottom as usize)),
            "cut should land in the blank band, got {}",
            plan[0].bottom
        );
    }

    #[test]
    fn plan_absorbs_short_tail_into_last_strip() {
        // Height barely over one nominal strip: one strip, no runt tail.
        let profile = vec![0u64; 620];
        let plan = plan_strips(500, 620, &profile);
        assert_eq!(plan.len(), 1);
        assert_eq!(
            plan[0],
            Strip {
                top: 0,
                bottom: 620
            }
        );
    }

    #[test]
    fn cut_strips_match_plan_geometry() {
        let img = white_image_with_dark_bands(100, 900, &[]);
        let profile = ink_profile(&img);
        let plan = plan_strips(100, 900, &profile);
        let strips = cut_strips(&img, &plan);
        assert_eq!(strips.len(), plan.len());
        for (strip, bounds) in strips.iter().zip(&plan) {
            assert_eq!(strip.width(), 100);
            assert_eq!(strip.height(), bounds.bottom - bounds.top);
        }
    }

    #[test]
    fn merge_offsets_layout_and_joins_markdown() {
        let a = RecognizedDocument {
            markdown: "first strip\n".to_string(),
            layout: vec![LayoutSpan {
                label: "text".to_string(),
                boxes: vec![[1, 2, 3, 4]],
            }],
        };
        let b = RecognizedDocument {
            markdown: "  ".to_string(), // whitespace-only strip contributes nothing
            layout: vec![],
        };
        let c = RecognizedDocument {
            markdown: "third strip".to_string(),
            layout: vec![LayoutSpan {
                label: "text".to_string(),
                boxes: vec![[5, 6, 7, 8]],
            }],
        };
        let merged = merge_documents(vec![(a, 0), (b, 500), (c, 1000)]);
        assert_eq!(merged.markdown, "first strip\n\nthird strip");
        assert_eq!(merged.layout.len(), 2);
        assert_eq!(merged.layout[0].boxes[0], [1, 2, 3, 4]);
        assert_eq!(merged.layout[1].boxes[0], [5, 1006, 7, 1008]);
    }

    #[test]
    fn text_chars_ignores_image_placeholders_and_whitespace() {
        assert_eq!(text_chars(""), 0);
        assert_eq!(text_chars("   \n\t "), 0);
        assert_eq!(text_chars("abc def"), 6);
        // The incident output: a title and a figure placeholder.
        assert_eq!(
            text_chars("Google Messages\n![](images/0.jpg)\n"),
            "GoogleMessages".len()
        );
        assert_eq!(text_chars("before ![alt text](x.png) after"), 11);
        // Malformed placeholder (no close paren) is counted literally.
        assert!(text_chars("a ![](broken") > 1);
    }

    #[test]
    fn low_yield_flags_the_incident_shape_and_spares_dense_pages() {
        // 494x2000 ≈ 0.988 MP, ~15 chars → flagged.
        let flagged = low_yield_assessment("Google Messages\n![](images/0.jpg)\n", 494, 2000)
            .expect("incident shape must be flagged");
        assert_eq!(flagged.yield_chars, 14);
        assert!((flagged.input_megapixels - 0.988).abs() < 0.001);

        // Same output from a small crop: not judged.
        assert!(low_yield_assessment("Google Messages", 494, 700).is_none());

        // A dense page is far above the floor.
        let dense = "the quick brown fox jumps over the lazy dog ".repeat(60);
        assert!(low_yield_assessment(&dense, 1000, 1000).is_none());
    }

    #[test]
    fn ink_profile_len_matches_luma_height() {
        let img = DynamicImage::ImageLuma8(ImageBuffer::from_pixel(10, 20, Luma([128u8])));
        assert_eq!(ink_profile(&img).len(), 20);
    }
}
