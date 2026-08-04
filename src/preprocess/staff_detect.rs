//! E5 (bd-3jo6.5.5): the TrOMR staff-detection front end — full page →
//! ordered single-staff crops (tromr-spec §7, v1 scope: printed/scanned
//! pages, GLOBAL deskew only; camera dewarp and barline-split chunking are
//! filed follow-ups).
//!
//! Classical CV, pure Rust, no new dependencies:
//!
//! 1. ink gray plane (the DISC-007 rule: inverted alpha only when alpha
//!    varies; else cv2 fixed-point luma) → Otsu binarization;
//! 2. global deskew by shear: the angle in ±5° that MAXIMIZES the row-
//!    projection variance (staff lines align → sharp peaks), coarse 1° then
//!    fine 0.25°;
//! 3. row-projection profile → staff-LINE bands (rows whose ink count clears
//!    half the profile peak), merged and centered;
//! 4. groups of 5 consecutive bands with near-uniform spacing (≤ 25%
//!    deviation) become STAVES;
//! 5. each staff is cropped full-width with a vertical margin of twice the
//!    line spacing × 2 (ledger lines, dynamics), clamped to the page.
//!
//! The detector returns crops top-to-bottom with their page bboxes — the
//! `staves[]` contract E9's full-page path and the multi-staff JSON ride.

use image::DynamicImage;

use crate::error::{FocrError, FocrResult};

/// One detected staff: the cropped ink-gray plane + its page-space bbox
/// (post-deskew coordinates).
pub struct StaffCrop {
    /// Row-major gray pixels (ink dark), `h × w`.
    pub gray: Vec<u8>,
    /// Crop width (the full deskewed page width).
    pub w: usize,
    /// Crop height.
    pub h: usize,
    /// `(x, y, w, h)` on the deskewed page.
    pub bbox: (usize, usize, usize, usize),
    /// The five staff-line center rows, band-relative (top to bottom) —
    /// the anchor for barline detection (bd-av64.4).
    pub lines: [usize; 5],
}

/// The ink gray plane (shared with `tromr_staff_tensor` — DISC-007 rule).
fn ink_gray(img: &DynamicImage) -> (Vec<u8>, usize, usize) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let alpha_is_ink = img.color().has_alpha() && img.to_rgba8().pixels().any(|p| p.0[3] < 255);
    let gray = if alpha_is_ink {
        img.to_rgba8().pixels().map(|p| 255 - p.0[3]).collect()
    } else {
        img.to_rgb8()
            .pixels()
            .map(|p| {
                let [r, g, b] = p.0;
                ((4899 * u32::from(r) + 9617 * u32::from(g) + 1868 * u32::from(b) + 8192) >> 14)
                    .min(255) as u8
            })
            .collect()
    };
    (gray, w, h)
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

fn profile_variance(profile: &[u32]) -> f64 {
    let n = profile.len() as f64;
    let mean = profile.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    profile
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n
}

/// The global deskew angle (degrees) in ±5° maximizing row-profile variance:
/// coarse 1° sweep then fine 0.25° around the winner.
fn deskew_angle(ink: &[bool], w: usize, h: usize) -> f64 {
    let score = |deg: f64| -> f64 {
        profile_variance(&sheared_row_profile(ink, w, h, deg.to_radians().tan()))
    };
    let mut best = (0.0f64, score(0.0));
    for d in -5..=5 {
        let deg = f64::from(d);
        let s = score(deg);
        if s > best.1 {
            best = (deg, s);
        }
    }
    let coarse = best.0;
    let mut fine = best;
    for i in -3..=3 {
        let deg = coarse + f64::from(i) * 0.25;
        let s = score(deg);
        if s > fine.1 {
            fine = (deg, s);
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

/// Group line centers into 5-line staves with near-uniform spacing (≤ 25%
/// max deviation from the mean gap). Greedy left-to-right — staves do not
/// overlap on a page.
fn group_staves(centers: &[usize]) -> Vec<[usize; 5]> {
    let mut staves = Vec::new();
    let mut i = 0;
    while i + 4 < centers.len() {
        let five = [
            centers[i],
            centers[i + 1],
            centers[i + 2],
            centers[i + 3],
            centers[i + 4],
        ];
        let gaps: Vec<f64> = five.windows(2).map(|p| (p[1] - p[0]) as f64).collect();
        let mean = gaps.iter().sum::<f64>() / 4.0;
        let ok = mean >= 2.0 && gaps.iter().all(|g| (g - mean).abs() <= 0.25 * mean);
        if ok {
            staves.push(five);
            i += 5;
        } else {
            i += 1;
        }
    }
    staves
}

/// Detect staves on a full page (tromr-spec §7 v1). Returns crops
/// top-to-bottom; an empty result means "no 5-line staff found" (the caller
/// decides whether to fall back to whole-image recognition).
///
/// # Errors
/// A degenerate (zero-sized) image.
pub fn detect_staves(img: &DynamicImage) -> FocrResult<Vec<StaffCrop>> {
    let (gray, w, h) = ink_gray(img);
    if w == 0 || h == 0 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "staff_detect: degenerate {w}x{h} input"
        )));
    }
    let thr = otsu_threshold(&gray);
    let ink: Vec<bool> = gray.iter().map(|&v| v <= thr).collect();

    let angle = deskew_angle(&ink, w, h);
    let gray = shear_gray(&gray, w, h, angle);
    let ink: Vec<bool> = gray.iter().map(|&v| v <= thr).collect();

    let profile = sheared_row_profile(&ink, w, h, 0.0);
    let peak = profile.iter().copied().max().unwrap_or(0);
    if peak == 0 {
        return Ok(Vec::new());
    }
    let centers = line_band_centers(&profile, peak / 2);
    let staves = group_staves(&centers);

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
        let y1_classic = (five[4] + margin + 1).min(h);

        // FIT-FIRST (bd-av64.14): when the classic full-width band already
        // fits the positional budget, keep it BIT-IDENTICAL to the historic
        // geometry — recognition is knife-edge sensitive to crop margins,
        // so geometry only changes where the old geometry hard-failed.
        if img_h * w <= budget * (y1_classic - y0_classic) {
            let ch = y1_classic - y0_classic;
            let mut crop = vec![0u8; ch * w];
            crop.copy_from_slice(&gray[y0_classic * w..y1_classic * w]);
            crops.push(StaffCrop {
                gray: crop,
                w,
                h: ch,
                bbox: (0, y0_classic, w, ch),
                lines: five.map(|l| l - y0_classic),
            });
            continue;
        }

        // Over budget: (1) trim the band to its ink extent (staff lines span
        // the system, so any column inside holds >= 5 ink pixels; a >= 2
        // floor keeps specks from stretching the band to the page margins),
        // padded by ~2 line-spacings; (2) if still over, extend vertically
        // toward the neighbor midlines (measured: a staff at ~30% of the
        // frame still reads correctly, while width overflow is a hard
        // failure). A band that cannot reach the budget is emitted as-is —
        // per-staff recovery (bd-av64.2) skips it with a named reason.
        let lo_bound = if i > 0 {
            (staves[i - 1][4] + five[0]) / 2
        } else {
            0
        };
        let hi_bound = if i + 1 < staves.len() {
            (five[4] + staves[i + 1][0]).div_ceil(2)
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
        x1 = (x1 + pad).min(w);
        let step = spacing.max(1.0).round() as usize;
        while img_h * (x1 - x0) > budget * (y1 - y0) && (y0 > lo_bound || y1 < hi_bound) {
            y0 = y0.saturating_sub(step).max(lo_bound);
            y1 = (y1 + step).min(hi_bound).min(h);
        }

        let (ch, cw) = (y1 - y0, x1 - x0);
        let mut crop = vec![0u8; ch * cw];
        for (row, y) in (y0..y1).enumerate() {
            crop[row * cw..(row + 1) * cw].copy_from_slice(&gray[y * w + x0..y * w + x1]);
        }
        crops.push(StaffCrop {
            gray: crop,
            w: cw,
            h: ch,
            bbox: (x0, y0, cw, ch),
            lines: five.map(|l| l - y0),
        });
    }
    for crop in &mut crops {
        refine_band_skew(crop);
    }
    Ok(crops)
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
fn refine_band_skew(crop: &mut StaffCrop) {
    let thr = otsu_threshold(&crop.gray);
    let ink: Vec<bool> = crop.gray.iter().map(|&v| v <= thr).collect();
    let score = |deg: f64| -> f64 {
        profile_variance(&sheared_row_profile(
            &ink,
            crop.w,
            crop.h,
            deg.to_radians().tan(),
        ))
    };
    let flat = score(0.0);
    let mut best = (0.0f64, flat);
    for i in -15..=15i32 {
        let deg = f64::from(i) * 0.1;
        if deg == 0.0 {
            continue;
        }
        let sc = score(deg);
        if sc > best.1 {
            best = (deg, sc);
        }
    }
    if best.0.abs() < 0.2 {
        return;
    }
    let sheared = shear_gray(&crop.gray, crop.w, crop.h, best.0);
    let sheared_ink: Vec<bool> = sheared.iter().map(|&v| v <= thr).collect();
    let profile = sheared_row_profile(&sheared_ink, crop.w, crop.h, 0.0);
    let peak = profile.iter().copied().max().unwrap_or(0);
    if peak == 0 {
        return;
    }
    let centers = line_band_centers(&profile, peak / 2);
    let staves = group_staves(&centers);
    let Some(five) = staves.first() else { return };
    crop.gray = sheared;
    crop.lines = *five;
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
        let crops = detect_staves(&tilted).expect("detects");
        assert_eq!(crops.len(), 2, "deskew recovers both staves");
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
    fn otsu_separates_bimodal() {
        let mut v = vec![20u8; 500];
        v.extend(vec![230u8; 500]);
        let t = otsu_threshold(&v);
        // Convention: ink = v <= t, so t must include the dark mode and
        // exclude the light one.
        assert!((20..230).contains(&t), "threshold {t} between the modes");
    }
}
