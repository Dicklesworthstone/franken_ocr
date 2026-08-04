//! bd-re8.10: the metamorphic test suite — oracle-free self-consistency under
//! transformations (docs/conformance/METAMORPHIC.md is the catalog; every
//! relation below cites its MR entry). Differential asks "same as the
//! reference?"; THIS suite asks "self-consistent under transforms the model's
//! documented semantics say must not (or must predictably) matter?".
//!
//! ## ⚠️ THE NEGATIVE GUARD (MR-5 / OQ-13) — load-bearing, do not remove
//!
//! This suite deliberately contains NO "multi-page concat == sum of
//! single-page parses" assertion, and none may ever be added: under R-SWA the
//! multi-page decode is CROSS-PAGE DEPENDENT (all pages' visual+prompt
//! prefixes form one frozen reference block; page N attends to pages
//! 1..N-1 — METAMORPHIC.md §MR-5, OQ-13). The defensible property is the
//! OPPOSITE (dependence-existence), gated on OQ-13's reference-block span
//! confirmation. Until then MR-5 is counted as GATED, never silently absent.
//!
//! Relations: MR-1 identity-resize (STRICT text), MR-2 rotation bbox maps
//! (the pure coordinate math always-on; the live leg needs a det-emitting
//! corpus page and is gated), MR-3a mean-gray pad (STRICT text; the fill
//! comes from `preprocess::PAD_FILL`, never hard-coded), MR-4 determinism
//! (in-process twice + the FOCR_THREADS axis via the real binary), MR-5
//! gated. Live legs are model-gated skip-with-SUCCESS (plan §8.3).

use image::{DynamicImage, RgbImage};

#[path = "support/parity_harness.rs"]
mod parity_harness;

use franken_ocr::preprocess::PAD_FILL;

// ───────────────────────── pure transform library (§7) ─────────────────────────

/// MR-2 point map in the normalized `[0,999]²` square.
fn rot_point(rot: u32, x: i64, y: i64) -> (i64, i64) {
    match rot {
        90 => (999 - y, x),
        180 => (999 - x, 999 - y),
        270 => (y, 999 - x),
        _ => (x, y),
    }
}

/// MR-2 box map: transform both corners, re-sort to (min, max).
fn rot_box(rot: u32, b: (i64, i64, i64, i64)) -> (i64, i64, i64, i64) {
    let (x1, y1) = rot_point(rot, b.0, b.1);
    let (x2, y2) = rot_point(rot, b.2, b.3);
    (x1.min(x2), y1.min(y2), x1.max(x2), y1.max(y2))
}

/// MR-3a affine offset: a normalized coord `c` on unpadded width `w`, after a
/// left pad of `p` pixels, lands at `(c/999·w + p) / (w + p_total) · 999`.
fn pad_coord(c: i64, w: u32, pad_before: u32, pad_total: u32) -> i64 {
    let px = c as f64 / 999.0 * f64::from(w) + f64::from(pad_before);
    (px / f64::from(w + pad_total) * 999.0) as i64
}

/// Mean-gray border pad (MR-3a): the fill is THE preprocess constant.
fn pad_gray(img: &DynamicImage, left: u32, top: u32, right: u32, bottom: u32) -> DynamicImage {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let mut canvas = RgbImage::from_pixel(
        w + left + right,
        h + top + bottom,
        image::Rgb([PAD_FILL; 3]),
    );
    image::imageops::overlay(&mut canvas, &rgb, i64::from(left), i64::from(top));
    DynamicImage::ImageRgb8(canvas)
}

/// MR-1 identity resize: same dimensions, in-memory (never a lossy re-encode).
fn identity_resize(img: &DynamicImage) -> DynamicImage {
    let (w, h) = (img.width(), img.height());
    img.resize_exact(w, h, image::imageops::FilterType::Nearest)
}

// ─────────────────── always-on: the coordinate maps (MR-2/MR-3a) ───────────────────

#[test]
fn mr2_rotation_maps_hand_worked() {
    // Hand-worked: the box (100, 200, 300, 400) under each rotation.
    assert_eq!(rot_box(90, (100, 200, 300, 400)), (599, 100, 799, 300));
    assert_eq!(rot_box(180, (100, 200, 300, 400)), (699, 599, 899, 799));
    assert_eq!(rot_box(270, (100, 200, 300, 400)), (200, 699, 400, 899));
    // rot90 composed twice == rot180 (up to the re-sort, exact here).
    let b = (12, 34, 567, 890);
    assert_eq!(rot_box(90, rot_box(90, b)), rot_box(180, b));
    // rot180 is an involution.
    assert_eq!(rot_box(180, rot_box(180, b)), b);
    // Degenerate (point) boxes stay degenerate.
    let p = rot_box(90, (500, 500, 500, 500));
    assert_eq!((p.0, p.1), (p.2, p.3));
}

#[test]
fn mr3a_pad_offset_map_hand_worked() {
    // A coord at the left edge (0) of a 1000px page, padded 100px left of
    // 200 total: lands at 100/1200 of the page ⇒ 999·(100/1200) = 83.
    assert_eq!(pad_coord(0, 1000, 100, 200), 83);
    // The right edge (999) ⇒ (1000+100)/1200 ⇒ 915 (integer truncation, the
    // reference's own int() discipline).
    assert_eq!(pad_coord(999, 1000, 100, 200), 915);
    // Zero pad is the identity (the MR-1 bridge).
    for c in [0, 1, 499, 999] {
        assert_eq!(pad_coord(c, 640, 0, 0), c);
    }
}

#[test]
fn transform_generators_are_lossless_permutations() {
    // rot90 four times == identity, pixel-exact (no resampling).
    let mut img = RgbImage::new(5, 3);
    for (i, p) in img.pixels_mut().enumerate() {
        *p = image::Rgb([
            (i * 7 % 256) as u8,
            (i * 13 % 256) as u8,
            (i * 29 % 256) as u8,
        ]);
    }
    let img = DynamicImage::ImageRgb8(img);
    let r4 = img.rotate90().rotate90().rotate90().rotate90();
    assert_eq!(img.to_rgb8().as_raw(), r4.to_rgb8().as_raw());
    // The identity resize is pixel-exact.
    assert_eq!(
        img.to_rgb8().as_raw(),
        identity_resize(&img).to_rgb8().as_raw()
    );
    // The gray pad places the original intact at the offset, fill = PAD_FILL.
    let padded = pad_gray(&img, 2, 1, 3, 4);
    assert_eq!((padded.width(), padded.height()), (10, 8));
    let p = padded.to_rgb8();
    assert_eq!(
        p.get_pixel(0, 0).0,
        [PAD_FILL; 3],
        "border is the model's own fill"
    );
    assert_eq!(p.get_pixel(2, 1).0, img.to_rgb8().get_pixel(0, 0).0);
}

// ───────────────────── model-gated live relations (skip-with-SUCCESS) ─────────────────────

fn resolve_model() -> Option<std::path::PathBuf> {
    let path = franken_ocr::OcrEngine::model_path();
    path.exists().then_some(path)
}

/// The live corpus image: `FOCR_METAMORPHIC_IMAGE` (a real page) else a
/// synthetic non-square, non-640-multiple page with content (a dark block
/// pattern) — exercising the pad/tile geometry either way (§7).
fn corpus_image() -> DynamicImage {
    if let Some(p) = std::env::var_os("FOCR_METAMORPHIC_IMAGE")
        && let Ok(img) = image::open(std::path::PathBuf::from(&p))
    {
        return img;
    }
    let mut img = RgbImage::from_pixel(700, 300, image::Rgb([255, 255, 255]));
    for y in 40..60 {
        for x in 50..650 {
            if (x / 30) % 2 == 0 {
                img.put_pixel(x, y, image::Rgb([20, 20, 20]));
            }
        }
    }
    DynamicImage::ImageRgb8(img)
}

fn rollup(run: u32, gated: u32) {
    eprintln!(
        "{{\"suite\":\"metamorphic\",\"relations_total\":5,\"relations_run\":{run},\
         \"relations_gated\":{gated}}}"
    );
}

/// MR-1 + MR-3a + MR-4 (in-process): the STRICT text relations over the live
/// engine. One test so the engine/model loads once; per-relation NDJSON.
#[test]
fn mr1_mr3a_mr4_strict_text_relations() {
    let Some(model) = resolve_model() else {
        eprintln!(
            "{{\"suite\":\"metamorphic\",\"event\":\"skip\",\"result\":\"skip_no_model\",\
             \"detail\":\"MR-1/MR-3a/MR-4 live legs need the model; maps+generators ran\"}}"
        );
        rollup(0, 5);
        return;
    };
    let dir = std::env::temp_dir().join(format!("focr-metamorphic-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let engine = franken_ocr::OcrEngine::new().expect("engine builds");
    let img = corpus_image();

    let recognize = |name: &str, im: &DynamicImage| -> String {
        let path = dir.join(format!("{name}.png"));
        im.save(&path).expect("fixture saves");
        engine
            .recognize_with_model(&model, &path)
            .expect("recognize runs")
    };

    // MR-4 first (the keystone: every other relation assumes reproducibility).
    let base = recognize("base", &img);
    let again = recognize("base_again", &img);
    parity_harness::assert_outputs_deterministic(
        "metamorphic",
        "mr4_same_thread",
        1,
        base.as_bytes(),
        again.as_bytes(),
    );

    // MR-1: identity resize is a no-op on recognized text (STRICT).
    let resized = recognize("mr1_identity", &identity_resize(&img));
    parity_harness::assert_outputs_deterministic(
        "metamorphic",
        "mr1_identity_resize",
        1,
        base.as_bytes(),
        resized.as_bytes(),
    );

    // MR-3a: mean-gray pad invariance (STRICT text), symmetric + asymmetric.
    for (case, l, t, r, b) in [("mr3a_sym8", 8, 8, 8, 8), ("mr3a_asym", 32, 16, 0, 0)] {
        let padded = recognize(case, &pad_gray(&img, l, t, r, b));
        parity_harness::assert_outputs_deterministic(
            "metamorphic",
            case,
            1,
            base.as_bytes(),
            padded.as_bytes(),
        );
    }

    // MR-3b (SHOULD, existential — never a hard gate): white pad sensitivity
    // is OBSERVED and logged, not asserted (white maps to +1, visible content).
    let white = {
        let rgb = img.to_rgb8();
        let (w, h) = rgb.dimensions();
        let mut canvas = RgbImage::from_pixel(w + 64, h + 64, image::Rgb([255, 255, 255]));
        image::imageops::overlay(&mut canvas, &rgb, 32, 32);
        recognize("mr3b_white", &DynamicImage::ImageRgb8(canvas))
    };
    eprintln!(
        "{{\"suite\":\"metamorphic\",\"relation\":\"MR-3b\",\"strength\":\"SHOULD\",\
         \"text_changed\":{},\"note\":\"white pad is visible content; observed only\"}}",
        white != base
    );

    // MR-2 live leg needs a det-emitting corpus page (gated); MR-5 is gated
    // on OQ-13 (see the module-doc negative guard).
    eprintln!(
        "{{\"suite\":\"metamorphic\",\"relation\":\"MR-2-live\",\"result\":\"gated\",\
         \"detail\":\"needs a grounding-box corpus page; the coordinate maps ran always-on\"}}"
    );
    eprintln!(
        "{{\"suite\":\"metamorphic\",\"relation\":\"MR-5\",\"result\":\"gated\",\
         \"detail\":\"OQ-13 RESOLVED + recognize_multi_page SHIPPED (bd-1gv.25; the 2-page \
         real-model e2e runs one fused 227-token cross-page prefill); the dependence-EXISTENCE \
         live leg still needs a cross-referencing corpus document + the infer_multi oracle \
         (bd-1gv.26). sum-of-parts stays BANNED\"}}"
    );
    rollup(3, 2);
}

/// MR-4 thread axis: the SAME image at `FOCR_THREADS=1` vs `FOCR_THREADS=4`
/// through the REAL binary must be byte-identical on stdout (the env latches
/// per process, so this axis runs out-of-process by design — bd-3kge).
#[test]
fn mr4_thread_axis_via_binary() {
    let Some(model) = resolve_model() else {
        eprintln!(
            "{{\"suite\":\"metamorphic\",\"event\":\"skip\",\"result\":\"skip_no_model\",\
             \"detail\":\"MR-4 thread axis needs the model\"}}"
        );
        return;
    };
    let dir = std::env::temp_dir().join(format!("focr-mr4-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join("page.png");
    corpus_image().save(&path).expect("fixture saves");

    let run_at = |threads: &str| -> Vec<u8> {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_focr"))
            .args(["ocr"])
            .arg(&path)
            .arg("--model")
            .arg(&model)
            .env("FOCR_THREADS", threads)
            .output()
            .expect("focr runs");
        assert!(
            out.status.success(),
            "focr @FOCR_THREADS={threads} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        out.stdout
    };
    let one = run_at("1");
    let four = run_at("4");
    parity_harness::assert_outputs_deterministic("metamorphic", "mr4_thread_axis", 1, &one, &four);
    eprintln!(
        "{{\"suite\":\"metamorphic\",\"relation\":\"MR-4-threads\",\"result\":\"pass\",\
         \"threads\":[1,4]}}"
    );
}
