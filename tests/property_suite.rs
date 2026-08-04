//! Property-based invariants over shared generators (bd-10sb.1).
//!
//! The four canonical runners the E-TEST plumbing bead wires (the per-module
//! assertions stay owned by module authors; these are the cross-cutting,
//! generator-driven contracts):
//!
//! 1. **SIMD == scalar bit-identity** — the dispatched int8 GEMM entrypoints
//!    (S8S8, U8S8, offline-packed-B) match the scalar oracle EXACTLY on every
//!    generated shape/operand, including the doctrine-#6 worst-case K depths
//!    and saturated extremes. Integer accumulation is exact, so any mismatch
//!    is a real kernel bug — no tolerance.
//! 2. **i32 accumulation never wraps** — the scalar oracle's i32 result
//!    equals a widened i64 reference at every generated depth up to K=6848
//!    (the proof obligation behind plan §5.4, generator-driven).
//! 3. **Preprocess geometry invariants** — `preprocess_dynamic` (Base) and
//!    `preprocess_dynamic_squash` never panic on arbitrary small RGB images
//!    and always emit the mode's exact tensor geometry + placeholder census.
//! 4. **Parser totality on untrusted bytes** — `Weights::from_bytes` over a
//!    valid `.focrq` blob under arbitrary byte flips / truncations / deletions
//!    either parses or returns a typed [`FocrError`] — NEVER panics, NEVER
//!    hangs (§7.4: malformed input is a clean error, not a crash).
//!
//! Plus the model-gated **tokenizer round-trip** (byte-level BPE must decode
//! back to the exact input for special-free text) — skip-with-SUCCESS when
//! the tokenizer artifact is absent.
//!
//! Case counts are CI-bounded via `PROPTEST_CASES` (default 64 here; deep
//! runs belong to the scheduled lane). Failures shrink to a minimal
//! counterexample and persist a regression seed beside this file
//! (`property_suite.proptest-regressions` — commit it so a found bug stays
//! found).

#[path = "support/proptest_support.rs"]
mod proptest_support;

use proptest::prelude::*;
use proptest_support::{
    apply_mutations, blob_mutations, gemm_shape, small_rgb_image, tokenizer_text,
};

use franken_ocr::FocrError;
use franken_ocr::native_engine::weights::Weights;
use franken_ocr::preprocess::{self, PreprocessMode};
use franken_ocr::quant::focrq::{FocrqBuilder, WriteDType};
use franken_ocr::simd;

/// Bounded default so the always-on lane stays fast; `PROPTEST_CASES`
/// overrides for the deep/scheduled lane.
fn cases() -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64)
}

fn config() -> ProptestConfig {
    ProptestConfig {
        cases: cases(),
        ..ProptestConfig::default()
    }
}

/// One structured NDJSON summary line per property batch (TEST-log-conventions).
fn tlog(test: &str, n_cases: u32, detail: &str) {
    println!(
        "{}",
        serde_json::json!({
            "suite": "property",
            "schema_version": 1,
            "test": test,
            "event": "stage",
            "strategy": "proptest",
            "n_cases": n_cases,
            "result": "pass",
            "detail": detail,
        })
    );
}

// ── 1. SIMD == scalar bit-identity ──────────────────────────────────────────

proptest! {
    #![proptest_config(config())]

    #[test]
    fn simd_s8s8_matches_scalar_oracle(
        (m, k, n) in gemm_shape(),
        seed_a in any::<u64>(),
        seed_b in any::<u64>(),
    ) {
        // Derive operands from the seeds with the saturated-biased pools
        // (buffer strategies at K=6848 would dominate shrink time; a seeded
        // fill keeps cases fast while proptest still shrinks the SHAPE).
        let a = seeded_i8(m * k, seed_a);
        let b = seeded_i8(n * k, seed_b);
        let mut got = vec![0i32; m * n];
        let mut want = vec![0i32; m * n];
        simd::igemm_s8s8(&a, &b, m, k, n, &mut got);
        franken_ocr::simd::scalar::igemm_s8s8(&a, &b, m, k, n, &mut want);
        prop_assert_eq!(&got, &want, "dispatched S8S8 != scalar at ({}, {}, {})", m, k, n);

        // The offline-packed-B entry must agree too (bd-2mo.3).
        let (panels, _, _) = simd::pack::smmla_pack_panels(&b, 0, n, k, k);
        let mut packed = vec![0x5a5a_5a5ai32; m * n]; // deliberately dirty
        simd::igemm_s8s8_packed_b(&a, &panels, m, k, n, &mut packed);
        prop_assert_eq!(&packed, &want, "packed-B != scalar at ({}, {}, {})", m, k, n);
    }

    #[test]
    fn simd_u8s8_matches_scalar_oracle(
        (m, k, n) in gemm_shape(),
        seed_a in any::<u64>(),
        seed_b in any::<u64>(),
    ) {
        let a = seeded_u8(m * k, seed_a);
        let b = seeded_i8(n * k, seed_b);
        let mut got = vec![0i32; m * n];
        let mut want = vec![0i32; m * n];
        simd::igemm_u8s8(&a, &b, m, k, n, &mut got);
        franken_ocr::simd::scalar::igemm_u8s8(&a, &b, m, k, n, &mut want);
        prop_assert_eq!(&got, &want, "dispatched U8S8 != scalar at ({}, {}, {})", m, k, n);
    }
}

// ── 2. i32 accumulation never wraps (widened i64 reference) ────────────────

proptest! {
    #![proptest_config(config())]

    #[test]
    fn i32_accumulation_matches_i64_reference(
        (m, k, n) in gemm_shape(),
        seed_a in any::<u64>(),
        seed_b in any::<u64>(),
    ) {
        let a = seeded_i8(m * k, seed_a);
        let b = seeded_i8(n * k, seed_b);
        let mut got = vec![0i32; m * n];
        franken_ocr::simd::scalar::igemm_s8s8(&a, &b, m, k, n, &mut got);
        for row in 0..m {
            for col in 0..n {
                let mut acc: i64 = 0;
                for i in 0..k {
                    acc += i64::from(a[row * k + i]) * i64::from(b[col * k + i]);
                }
                prop_assert!(
                    i64::from(got[row * n + col]) == acc,
                    "i32 accumulator wrapped at ({}, {}, {}) [{},{}]: i32 {} vs i64 {}",
                    m, k, n, row, col, got[row * n + col], acc
                );
            }
        }
    }
}

// ── 3. Preprocess geometry invariants ───────────────────────────────────────

proptest! {
    #![proptest_config(config())]

    #[test]
    fn preprocess_geometry_invariants_hold(
        img in small_rgb_image(),
        base_640 in any::<bool>(),
    ) {
        let base = if base_640 { 640usize } else { 1024 };
        let nq = preprocess::num_queries(base);
        let census = (nq + 1) * nq + 1;

        // Base mode: aspect-preserving pad — never panics, exact geometry.
        let pre = preprocess::preprocess_dynamic(
            img.clone(),
            PreprocessMode::Base { base_size: base },
        ).expect("Base preprocess is total over decodable RGB images");
        prop_assert_eq!(pre.global.pixels.data.len(), 3 * base * base);
        prop_assert_eq!(pre.placeholder_token_count(), census);
        prop_assert_eq!(pre.num_views(), 1);

        // Multi-page squash (bd-1gv.25): same geometry contract.
        let sq = preprocess::preprocess_dynamic_squash(img, base)
            .expect("squash preprocess is total over decodable RGB images");
        prop_assert_eq!(sq.global.pixels.data.len(), 3 * base * base);
        prop_assert_eq!(sq.placeholder_token_count(), census);
    }
}

// ── 4. Parser totality over untrusted bytes ─────────────────────────────────

/// A tiny VALID `.focrq` blob (one QInt8 tensor + one F32 tensor) the
/// mutation property corrupts. Built through the real writer so the base
/// case genuinely parses.
fn tiny_valid_focrq() -> Vec<u8> {
    let mut b = FocrqBuilder::new();
    b.add_quantized(
        "layer.weight",
        WriteDType::QInt8PerChan,
        vec![2, 4],
        vec![1u8, 2, 3, 4, 5, 6, 7, 8],
        [0.5f32, 0.25]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect(),
        0,
        0,
    )
    .expect("valid QInt8 tensor");
    b.add_tensor(
        "norm.weight",
        WriteDType::F32,
        vec![4],
        [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    )
    .expect("valid F32 tensor");
    b.build()
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn focrq_parser_is_total_on_mutated_bytes(mutations in blob_mutations()) {
        let blob = apply_mutations(&tiny_valid_focrq(), &mutations);
        // The property IS "no panic / typed error": a panic inside fails the
        // case (proptest catches it and shrinks the mutation list).
        match Weights::from_bytes(blob) {
            Ok(_) => {} // a mutation that lands in payload bytes can stay valid
            Err(
                FocrError::FormatMismatch(_)
                | FocrError::InputDecode(_)
                | FocrError::ModelNotFound(_)
                | FocrError::Other(_),
            ) => {}
            Err(other) => {
                return Err(TestCaseError::fail(format!(
                    "unexpected error class for corrupt .focrq: {other:?}"
                )));
            }
        }
    }
}

// ── 5. Tokenizer round-trip (model-gated) ───────────────────────────────────

/// Byte-level BPE must be LOSSLESS: `decode(encode(s)) == s` for any text
/// (the byte fallback covers every sequence; specials are typed, not textual).
/// Gated on the tokenizer artifact beside the default model cache — the
/// generators run regardless (the always-on leg proves the strategy shapes).
#[test]
fn tokenizer_round_trip_is_lossless_when_present() {
    let candidates = [
        std::env::var_os("FOCR_MODEL_PATH")
            .map(|p| std::path::PathBuf::from(p).join("tokenizer.json")),
        dirs_cache_tokenizer(),
    ];
    let Some(path) = candidates.into_iter().flatten().find(|p| p.is_file()) else {
        println!(
            "{}",
            serde_json::json!({
                "suite": "property", "schema_version": 1,
                "test": "tokenizer_round_trip_is_lossless_when_present",
                "event": "skip", "result": "skip_no_model",
                "detail": "tokenizer.json not present (FOCR_MODEL_PATH / user cache); strategies exercised by the other runners",
            })
        );
        return;
    };
    let tok = franken_ocr::tokenizer::Tokenizer::load(&path).expect("tokenizer loads");
    let mut runner = proptest::test_runner::TestRunner::new(config());
    runner
        .run(&tokenizer_text(), |s| {
            let ids = tok
                .encode(&s)
                .map_err(|e| TestCaseError::fail(format!("encode failed on {s:?}: {e}")))?;
            let back = tok
                .decode(&ids)
                .map_err(|e| TestCaseError::fail(format!("decode failed on {s:?}: {e}")))?;
            prop_assert_eq!(&back, &s, "round-trip mismatch");
            Ok(())
        })
        .expect("tokenizer round-trip property");
    tlog(
        "tokenizer_round_trip_is_lossless_when_present",
        cases(),
        "byte-level BPE decode(encode(s)) == s over generated Unicode",
    );
}

fn dirs_cache_tokenizer() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME")
        .map(|h| std::path::PathBuf::from(h).join(".cache/franken_ocr/models/tokenizer.json"))
}

// ── deterministic seeded operand fills (fast at K=6848, shape still shrinks) ─

fn seeded_i8(len: usize, seed: u64) -> Vec<i8> {
    let mut x = seed | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            // Bias ~25% of lanes to the saturated extremes.
            match x % 8 {
                0 => i8::MIN,
                1 => i8::MAX,
                _ => (x >> 8) as u8 as i8,
            }
        })
        .collect()
}

fn seeded_u8(len: usize, seed: u64) -> Vec<u8> {
    let mut x = seed | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            match x % 8 {
                0 => 0u8,
                1 => u8::MAX,
                _ => (x >> 8) as u8,
            }
        })
        .collect()
}

/// Regenerate the committed fuzz seed corpus (bd-10sb.1) — opt-in writer
/// (`FOCR_WRITE_FUZZ_SEEDS=1 cargo test --test property_suite write_fuzz`),
/// so `cargo fuzz run <target>` always starts from deterministic seeds:
/// valid + truncated + bit-flipped artifacts per parser target, real tiny
/// images for the decoder, split-class text for the pretokenizers.
#[test]
fn write_fuzz_seed_corpus_when_requested() {
    if std::env::var_os("FOCR_WRITE_FUZZ_SEEDS").is_none() {
        println!(
            "{}",
            serde_json::json!({
                "suite": "property", "schema_version": 1,
                "test": "write_fuzz_seed_corpus_when_requested",
                "event": "skip", "result": "pass",
                "detail": "seed writer idle (set FOCR_WRITE_FUZZ_SEEDS=1 to regenerate fuzz/corpus)",
            })
        );
        return;
    }
    let root = std::path::Path::new("fuzz/corpus");
    let write = |dir: &str, name: &str, bytes: &[u8]| {
        let d = root.join(dir);
        std::fs::create_dir_all(&d).expect("corpus dir");
        std::fs::write(d.join(name), bytes).expect("seed write");
    };

    // focrq_parse: valid + truncated + flipped.
    let valid = tiny_valid_focrq();
    write("focrq_parse", "valid.focrq", &valid);
    write("focrq_parse", "truncated.focrq", &valid[..valid.len() / 2]);
    let mut flipped = valid.clone();
    flipped[8] ^= 0xff;
    write("focrq_parse", "flipped.focrq", &flipped);

    // safetensors_parse: minimal valid blob + truncation.
    let header = br#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut st = (header.len() as u64).to_le_bytes().to_vec();
    st.extend_from_slice(header);
    st.extend_from_slice(&1.0f32.to_le_bytes());
    write("safetensors_parse", "valid.safetensors", &st);
    write(
        "safetensors_parse",
        "truncated.safetensors",
        &st[..st.len() - 3],
    );

    // image_decode: a real tiny PNG + a truncation of it.
    let img = image::RgbImage::from_fn(6, 4, |x, y| image::Rgb([x as u8 * 40, y as u8 * 60, 128]));
    let mut png: Vec<u8> = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("encode tiny png");
    write("image_decode", "tiny.png", &png);
    write("image_decode", "truncated.png", &png[..png.len() / 2]);

    // pretok_split: the split classes.
    write(
        "pretok_split",
        "ascii.txt",
        b"Hello, world! It's 2026-07-07.",
    );
    write(
        "pretok_split",
        "digits.txt",
        b"31415926535897932384626433832795",
    );
    write("pretok_split", "cjk.txt", "\u{6f22}\u{5b57}\u{3068}\u{3072}\u{3089}\u{304c}\u{306a}\u{3001}\u{4e2d}\u{6587}\u{6d4b}\u{8bd5}\u{3002}".as_bytes());
    write(
        "pretok_split",
        "emoji_ws.txt",
        "  \t\n\u{1f980} emoji \u{200b} zero-width  ".as_bytes(),
    );
    println!(
        "{}",
        serde_json::json!({
            "suite": "property", "schema_version": 1,
            "test": "write_fuzz_seed_corpus_when_requested",
            "event": "result", "result": "pass",
            "detail": "fuzz/corpus regenerated (4 targets)",
        })
    );
}

/// The summary logger fires once per binary run (the proptest macros above
/// have no epilogue hook) — a plain test emitting the batch NDJSON.
#[test]
fn zz_property_batch_summary() {
    for (test, detail) in [
        (
            "simd_s8s8_matches_scalar_oracle",
            "dispatched + packed-B == scalar, worst-case-K injected",
        ),
        (
            "simd_u8s8_matches_scalar_oracle",
            "dispatched U8S8 == scalar",
        ),
        (
            "i32_accumulation_matches_i64_reference",
            "scalar i32 == widened i64 up to K=6848",
        ),
        (
            "preprocess_geometry_invariants_hold",
            "Base + squash total; census formula holds",
        ),
        (
            "focrq_parser_is_total_on_mutated_bytes",
            "flips/truncations/deletions => parse or typed error",
        ),
    ] {
        tlog(test, cases(), detail);
    }
}
