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
//! gated, MR-6 Unlimited-OCR-2 re-prefill ≡ incremental decode. Live legs are
//! model-gated skip-with-SUCCESS (plan §8.3).

use image::{DynamicImage, RgbImage};

#[path = "support/parity_harness.rs"]
mod parity_harness;
#[path = "support/uocr2.rs"]
mod uocr2;

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
         \"detail\":\"OQ-13 RESOLVED + recognize_document SHIPPED (bd-1gv.25, U2-E2; the 2-page \
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

// ─────────── MR-6: Unlimited-OCR-2 re-prefill ≡ incremental decode (U2-C2) ───────────

/// The decode steps MR-6 samples over `ids` (step t has consumed `ids[..t]`):
/// every 64th, the last, and ±2 around each generated `<PAGE>`.
fn mr6_steps(ids: &[u32]) -> Vec<usize> {
    use franken_ocr::tokenizer::special::PAGE;
    let mut steps: Vec<usize> = (64..=ids.len()).step_by(64).collect();
    steps.push(ids.len());
    for (i, _) in ids.iter().enumerate().filter(|&(_, &id)| id == PAGE) {
        // Feeding the marker is step i + 1.
        steps.extend((i + 1).saturating_sub(2)..=(i + 3).min(ids.len()));
    }
    steps.retain(|&t| t >= 1);
    steps.sort_unstable();
    steps.dedup();
    steps
}

/// MR-6 (U2-C2, plan §5.8; METAMORPHIC.md §MR-6): on Unlimited-OCR-2's cached
/// decode, incremental step t and a fresh prefill of the prompt plus the
/// first t ids agree, on the shipping int8-FFN cache and the all-f32 one.
///
/// - **Transform:** fold the first t generated ids into the prompt. The
///   re-prefill builds page coordinates from the whole id sequence
///   (`PageCoords::build`: generated markers take `bucket = cum − N`, clamped
///   at 0, and the same query-page rule), as §5.8's precondition requires.
/// - **Relation:** the same greedy token; logits within the path's
///   accumulation-order bound (prefill runs multi-row GEMMs, decode m = 1
///   GEMVs); every layer's selected blocks for the row equal (biases logged).
///   The one excuse is the router window: a prefill sets every row's M from
///   its whole length, so below the cap the re-prefill widens the window of
///   the earlier rows. A row it flags as window-pad sensitive
///   (`srswa::window_pad_sensitive`) at one layer can change the keys every
///   later layer reads, so a selection may then differ above that layer; that
///   is logged.
/// - **Retired ≡ unretired:** the incremental decode runs twice in lockstep,
///   block retirement on and off, and every step's hidden is bit-equal. The
///   unretired one is traced.
///
/// The ids are a recorded generation (`tests/fixtures/uocr2_oracle`): the
/// relation compares the model with itself, so any valid id sequence serves.
/// Cases: S1 page_0014 (one page, whose 279-row prompt has a 9-block router
/// window that the re-prefills widen) and S2 (three pages). Gated on
/// `FOCR_UOCR2_SOURCE_DIR` and the corpus pages; skips with success without.
#[test]
fn mr6_uocr2_reprefill_equals_incremental() {
    use franken_ocr::native_engine::sampler::{
        GuardVerdict, PageGuard, UOCR2_DEFAULT_MAX_PAGE_TOKENS,
    };
    use franken_ocr::native_engine::tensor::Mat;
    use franken_ocr::native_engine::{decoder, srswa};
    use franken_ocr::tokenizer::special::{EOS, PAGE};
    // The bounds on `max |Δlogit| / max |logit|`. The int8-FFN path's is
    // U2-B5's wiring bound: a last-ulp difference flips int8 activation
    // roundings in each layer's FFN (U2-C2 measured up to 4.4% on S1 and
    // 8.7% on S2). The f32 path has no such amplifier: measured at most
    // 9.0e-7 over both cases' 54 sampled steps, and held to about 11× that.
    const BOUND_MIXED: f32 = 0.10;
    const BOUND_F32: f32 = 1.0e-5;
    let skip = |detail: &str| {
        eprintln!(
            "{{\"suite\":\"metamorphic\",\"relation\":\"MR-6\",\"event\":\"skip\",\
             \"result\":\"skip_no_model\",\"detail\":\"{detail}\"}}"
        );
    };
    let Some(source) = std::env::var_os("FOCR_UOCR2_SOURCE_DIR").map(std::path::PathBuf::from)
    else {
        skip("FOCR_UOCR2_SOURCE_DIR unset");
        return;
    };
    let corpus = std::env::var_os("FOCR_CORPUS_DIR").map_or_else(
        || {
            std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME"))
                .join(".cache/franken_ocr/corpus/pages")
        },
        std::path::PathBuf::from,
    );
    let cases: [(&str, &[&str]); 2] = [
        ("S1-page_0014", &["page_0014.png"]),
        (
            "S2-p9_p14_p19",
            &["page_0009.png", "page_0014.png", "page_0019.png"],
        ),
    ];
    if let Some(missing) = cases
        .iter()
        .flat_map(|(_, pages)| pages.iter())
        .map(|p| corpus.join(p))
        .find(|p| !p.is_file())
    {
        skip(&format!("{} absent", missing.display()));
        return;
    }
    let fixtures_dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/uocr2_oracle");
    let meta: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixtures_dir.join("ORACLE.json")).expect("ORACLE.json"),
    )
    .expect("ORACLE.json parses");
    let w = uocr2::load_uocr2_source(&source);
    let table = w.mat("model.embed_tokens.weight").expect("embed table");
    let hidden = table.cols;
    let caches = [
        (
            "focr-mixed-ffn-int8",
            BOUND_MIXED,
            decoder::DecoderWeightCache::build_srswa(&w).expect("the int8-FFN cache builds"),
        ),
        (
            "f32-cached",
            BOUND_F32,
            decoder::DecoderWeightCache::build_srswa_f32(&w).expect("the f32 cache builds"),
        ),
    ];
    let domain = srswa::EntmaxDomain::TopM;
    let topm = srswa::RouterParams::uocr2().topm;
    let embed = |id: u32| Mat::from_vec(1, hidden, table.row(id as usize).to_vec());
    let last_row = |h: &Mat| Mat::from_vec(1, hidden, h.row(h.rows - 1).to_vec());
    let argmax = |logits: &[f32]| {
        (0..logits.len()).fold(0, |best, i| if logits[i] > logits[best] { i } else { best })
    };
    let started = std::time::Instant::now();
    let mut violations = 0usize;
    let mut cases_run = 0usize;

    for &(fixture, pages) in &cases {
        let entry = &meta["fixtures"][fixture];
        let pres: Vec<franken_ocr::preprocess::Preprocessed> = pages
            .iter()
            .map(|p| {
                let bytes = std::fs::read(corpus.join(p)).expect("corpus page reads");
                franken_ocr::preprocess::preprocess_bytes(
                    &bytes,
                    franken_ocr::preprocess::PreprocessMode::base(),
                    franken_ocr::preprocess::PreprocessProfile::V2,
                )
                .expect("v2 preprocess")
            })
            .collect();
        let prompt = uocr2::uocr2_document_prompt(&pres);
        assert_eq!(
            parity_harness::sha256_hex(serde_json::to_string(&prompt).unwrap().as_bytes()),
            entry["prompt_token_ids_sha256"]
                .as_str()
                .expect("prompt sha"),
            "{fixture}: the paged prompt is the oracle's"
        );
        let embeds = uocr2::uocr2_inputs_embeds(&w, &prompt, &pres);
        let ids: Vec<u32> =
            std::fs::read_to_string(fixtures_dir.join(format!("{fixture}.ids.txt")))
                .expect("recorded ids")
                .lines()
                .filter(|l| !l.starts_with('#'))
                .flat_map(str::split_whitespace)
                .map(|t| t.parse().expect("an id"))
                .collect();
        assert_eq!(
            parity_harness::sha256_hex(serde_json::to_string(&ids).unwrap().as_bytes()),
            entry["token_ids_sha256"].as_str().expect("ids sha"),
            "{fixture}: the recorded ids are the oracle's"
        );
        let steps = mr6_steps(&ids);
        let coords = srswa::PageCoords::build(&prompt).expect("page coordinates");
        // Below the cap the re-prefill's wider window reaches the prompt's
        // rows and the earlier generated ones; at the cap every row has it.
        let window_grows = prompt.len().div_ceil(srswa::BLOCK_SIZE) < topm;

        for (path, bound, wc) in &caches {
            let t_case = std::time::Instant::now();
            // The unretired decode is traced: its routes meet the re-prefill's.
            let (_, mut off) =
                decoder::prefill_with_cache_srswa_traced(wc, &embeds, &coords, domain)
                    .expect("unretired prefill");
            let (_, mut on) = decoder::prefill_with_cache_srswa(wc, &embeds, &coords, domain)
                .expect("retired prefill");
            on.set_retire(true);
            let mut guard = PageGuard::new(pages.len(), PAGE, EOS, UOCR2_DEFAULT_MAX_PAGE_TOKENS);
            let mut retire_diffs = Vec::new();
            let mut gaps = Vec::new();
            let mut excused = Vec::new();
            let mut max_bias_gap = 0.0f32;
            let mut case_violations = Vec::new();
            for t in 1..=steps.last().copied().unwrap_or(0) {
                let id = ids[t - 1];
                // The engine's flag: the guard forces the token this row predicts.
                guard.observe(id);
                let forced = matches!(guard.verdict(), GuardVerdict::Force(_));
                let row = embed(id);
                let h_on = decoder::decode_step_with_cache_srswa(wc, &mut on, id, &row, forced)
                    .expect("retired step");
                let h_off = decoder::decode_step_with_cache_srswa(wc, &mut off, id, &row, forced)
                    .expect("unretired step");
                if h_on
                    .data
                    .iter()
                    .map(|x| x.to_bits())
                    .ne(h_off.data.iter().map(|x| x.to_bits()))
                {
                    retire_diffs.push(t);
                }
                if !steps.contains(&t) {
                    continue;
                }
                let incremental = decoder::lm_head_cached(wc, &last_row(&h_off)).expect("lm_head");
                let mut grown_ids = prompt.clone();
                grown_ids.extend_from_slice(&ids[..t]);
                let mut data = embeds.data.clone();
                for &g in &ids[..t] {
                    data.extend_from_slice(table.row(g as usize));
                }
                let grown = Mat::from_vec(grown_ids.len(), hidden, data);
                let grown_coords =
                    srswa::PageCoords::build(&grown_ids).expect("grown page coordinates");
                let (h_re, re) =
                    decoder::prefill_with_cache_srswa_traced(wc, &grown, &grown_coords, domain)
                        .expect("re-prefill");
                let fresh = decoder::lm_head_cached(wc, &last_row(&h_re)).expect("lm_head");
                let gap = incremental
                    .data
                    .iter()
                    .zip(&fresh.data)
                    .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                let scale = fresh.data.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                gaps.push(gap / scale);
                if gap > *bound * scale {
                    case_violations.push(format!("step {t}: logit gap {gap} > {bound} × {scale}"));
                }
                let (ours, theirs) = (argmax(&incremental.data), argmax(&fresh.data));
                if ours != theirs {
                    case_violations.push(format!(
                        "step {t}: token {ours} vs re-prefill {theirs} (incremental margin {}, \
                         logit gap {gap})",
                        incremental.data[ours] - incremental.data[theirs]
                    ));
                }
                let (inc, rep) = (
                    off.trace.as_ref().expect("traced"),
                    re.trace.as_ref().expect("traced"),
                );
                // Rows flagged below a layer: their outputs there feed the keys
                // this layer routes on.
                let mut flagged_rows = 0;
                for layer in 0..inc.decode.len() {
                    let a = inc.decode[layer].as_ref().map(|r| &r.route);
                    let b = rep.prefill_last[layer].as_ref();
                    match (a, b) {
                        (Some(a), Some(b)) if a.blocks == b.blocks => {
                            max_bias_gap = a
                                .bias
                                .iter()
                                .zip(&b.bias)
                                .fold(max_bias_gap, |m, (x, y)| m.max((x - y).abs()));
                        }
                        _ => {
                            let detail = format!(
                                "step {t} L{layer}: blocks {:?} vs re-prefill {:?}",
                                a.map(|r| &r.blocks),
                                b.map(|r| &r.blocks)
                            );
                            if flagged_rows > 0 {
                                excused.push(serde_json::json!({
                                    "detail": detail, "flagged_rows": flagged_rows,
                                }));
                            } else {
                                case_violations.push(detail);
                            }
                        }
                    }
                    if window_grows {
                        flagged_rows += rep.pad_sensitive[layer].len();
                    }
                }
            }
            if !retire_diffs.is_empty() {
                case_violations.push(format!("retired ≠ unretired at steps {retire_diffs:?}"));
            }
            for v in &case_violations {
                eprintln!(
                    "{}",
                    serde_json::json!({
                        "suite": "metamorphic", "relation": "MR-6",
                        "case_id": format!("{path}/{fixture}"), "kind": "violation", "detail": v,
                    })
                );
            }
            eprintln!(
                "{}",
                serde_json::json!({
                    "suite": "metamorphic", "relation": "MR-6",
                    "case_id": format!("{path}/{fixture}"),
                    "strength": "STRICT tokens, selections and retirement, bounded logits",
                    "steps": steps, "decode_steps": steps.last(),
                    "violations": case_violations.len(), "bound": bound,
                    "max_rel_logit_gap": gaps.iter().copied().fold(0.0f32, f32::max),
                    "rel_logit_gaps": gaps, "max_bias_gap": max_bias_gap,
                    "window_grows": window_grows,
                    "pad_sensitive_excused": excused, "retired_blocks": on.dead_blocks(),
                    "elapsed_ms": t_case.elapsed().as_millis(),
                })
            );
            violations += case_violations.len();
            cases_run += 1;
        }
    }
    eprintln!(
        "{}",
        serde_json::json!({
            "suite": "metamorphic", "relation": "MR-6", "n_cases": cases_run,
            "violations": violations, "elapsed_ms": started.elapsed().as_millis(),
        })
    );
    assert_eq!(violations, 0, "MR-6 violated (see the NDJSON lines above)");
}
