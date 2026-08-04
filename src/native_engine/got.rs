//! GOT-OCR2.0 model assembly (bead B3): the vision front-half that turns a
//! preprocessed image + prompt id-stream into the decoder `inputs_embeds`, which
//! [`super::decoder_qwen2::forward_prefill`] then consumes. The Qwen2 dense
//! decoder itself lives in [`super::decoder_qwen2`]; this module is the
//! GOT-specific glue (SAM tower prefix, the `mm_projector_vary` connector, and the
//! `<imgpad>` splice) — none of which the Baidu path shares (GOT has no CLIP tower,
//! no `image_newline`/`view_seperator`, and its connector currency is 1024, not 1280).
//!
//! Every piece reuses an existing, parity-tested primitive:
//! * SAM-ViT-B tower → [`super::vision_sam::forward_prefix`] with the arch's
//!   `model.vision_tower_high` prefix (identical leaf names + geometry to Baidu's
//!   `model.sam_model`); returns `[1024, 256]` channel-major.
//! * connector → [`super::vision_sam::Linear::apply`] (`mm_projector_vary`, a plain
//!   `Linear(1024→1024)+bias`, no act/no norm).
//! * embed + splice → [`super::decoder::embed_tokens`] (tied HP table) +
//!   [`super::connector::masked_scatter`] over the `<imgpad>` (151859) rows.
//!
//! Certified against the bit-deterministic torch oracle: feeding the oracle's own
//! preprocessed image, the assembled `inputs_embeds` matches the oracle's
//! post-splice `hidden_0` (isolating the vision kernels + connector + splice from
//! the known CatmullRom-vs-bicubic resample tolerance, which the preprocess gate
//! covers separately). See the `#[cfg(test)]` seam gate.

use image::DynamicImage;

use super::decoder_qwen2::{self, DecoderConfig};
use super::tensor::Mat;
use super::weights::Weights;
use super::{connector, decoder, vision_sam};
use crate::error::FocrResult;
use crate::preprocess;
use crate::tokenizer::tiktoken::Tiktoken;

/// GOT generation stop id (`<|im_end|>`).
pub const EOS_ID: u32 = 151_645;

/// GOT's own generated-token ceiling (`generation_config.json` `max_new_tokens`).
/// The forward clamps any requested `--max-length` to this (bd-3j3p).
pub const MAX_NEW_TOKENS: usize = 4096;

/// Resolve the GOT global no-repeat-n-gram size (bd-ff4i kill-switch; upstream
/// `chat()` hard-codes 20, spec §12 OQ-8). Priority (fresh-eyes fix — the CLI
/// `--no-repeat-ngram` used to be silently ignored on this arm):
/// 1. the CLI decode override (`--no-repeat-ngram` / `FOCR_NO_REPEAT_NGRAM`),
/// 2. the `FOCR_GOT_NO_REPEAT_NGRAM` env (read once per process),
/// 3. the config default. `0` disables the guard at any level.
fn no_repeat_ngram_override(default: usize) -> usize {
    if let Some(n) = super::decode_overrides().no_repeat_ngram {
        return n;
    }
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    N.get_or_init(|| {
        std::env::var("FOCR_GOT_NO_REPEAT_NGRAM")
            .ok()
            .and_then(|v| v.trim().parse().ok())
    })
    .unwrap_or(default)
}

/// The GOT `<imgpad>` per-patch image token (spec §5): the prompt slot a projected
/// vision-feature row overwrites.
pub const IMG_PAD_ID: u32 = 151_859;
/// Projected vision feature rows per image view (`image_token_len`).
pub const IMAGE_TOKEN_LEN: usize = 256;

/// GOT vision features from a preprocessed `[3, side*side]` image: the SAM-ViT-B
/// tower (`prefix`, e.g. `model.vision_tower_high`) → `[1024, 256]` channel-major →
/// transpose → `[256, 1024]` → the `mm_projector_vary` `Linear(1024→1024)+bias` →
/// `[256, 1024]` token-major features. All high-precision (BF16→f32).
///
/// # Errors
/// The first vision-stage error (missing/mis-shaped tensor or kernel failure).
pub fn vision_features(statics: &GotStatics, image: &Mat) -> FocrResult<Mat> {
    let side = (image.cols as f64).sqrt() as usize;
    if side * side != image.cols || image.rows != 3 {
        return Err(crate::FocrError::Other(anyhow::anyhow!(
            "got vision: expected [3, side*side] input, got [{}, {}]",
            image.rows,
            image.cols
        )));
    }
    let sam = vision_sam::forward_with(&statics.sam, image, side, side)?; // [1024, 256]
    let sam_t = transpose(&sam); // [256, 1024] token-major
    statics.proj.apply(&sam_t) // [256, 1024]
}

/// The per-model-constant GOT tensors — the SAM tower, the `mm_projector_vary`
/// projector, and the widened `embed_tokens` table (~1 GB f32 together) —
/// hydrated ONCE and cached on the [`super::OcrModel`] (bd-av64.10: the
/// sequential page loop re-hydrated all three EVERY page, ~0.3–0.4 s/page of
/// pure bf16→f32 widening; same residency the batch path already held).
pub struct GotStatics {
    /// The hydrated SAM-ViT-B tower.
    pub sam: vision_sam::SamWeights,
    /// The `mm_projector_vary` connector (pre-transposed).
    pub proj: vision_sam::Linear,
    /// The widened `[vocab, hidden]` embed table.
    pub embed: Mat,
}

/// Hydrate a [`GotStatics`] from the artifact (`prefix` names the SAM tower,
/// e.g. `model.vision_tower_high`).
///
/// # Errors
/// A missing or mis-shaped tensor.
pub fn hydrate_statics(weights: &Weights, prefix: &str) -> FocrResult<GotStatics> {
    let th = std::time::Instant::now();
    let statics = GotStatics {
        sam: vision_sam::sam_weights_from(weights, prefix)?,
        proj: vision_sam::Linear::from_row_major(
            &weights.vec("model.mm_projector_vary.weight")?,
            weights.vec("model.mm_projector_vary.bias")?,
            1024,
            1024,
        )?,
        embed: weights.mat("model.embed_tokens.weight")?,
    };
    super::timing_log(&format!(
        "  got.hydrate(cached) {:.2}s",
        th.elapsed().as_secs_f64()
    ));
    Ok(statics)
}

/// Build the GOT decoder `inputs_embeds`: embed the prompt id-stream against the
/// tied `model.embed_tokens.weight`, then `masked_scatter` the vision features
/// into the 256 `<imgpad>` rows (in prompt order). Returns `[seq, hidden]`.
///
/// # Errors
/// A vision/embed error, or a [`connector::masked_scatter`] mismatch (the number
/// of `<imgpad>` rows must equal `vision_features.rows`).
pub fn build_inputs_embeds(
    statics: &GotStatics,
    image: &Mat,
    prompt_ids: &[u32],
) -> FocrResult<Mat> {
    let tokens = vision_features(statics, image)?; // [256, 1024]
    let embed = &statics.embed; // [vocab, hidden]
    let (vocab, hidden) = (embed.rows, embed.cols);
    let mut inputs_embeds = decoder::embed_tokens(&embed.data, vocab, hidden, prompt_ids)?;
    let mask: Vec<bool> = prompt_ids.iter().map(|&id| id == IMG_PAD_ID).collect();
    connector::masked_scatter(&mut inputs_embeds, &tokens, &mask)?;
    Ok(inputs_embeds)
}

/// Build the GOT OCR prompt id-stream (`GOTQwenForCausalLM.chat`): the MPT system
/// turn, the `<img><imgpad>×256</img>` image splice, the instruction, and the
/// assistant role marker. `format=false` → `OCR: ` (plain text); `format=true` →
/// `OCR with format: ` (the layout/LaTeX/table `.mmd` mode). Encoded with all
/// specials enabled — the plain form is token-id-EXACT to the torch oracle's 287-id
/// `l0c_prompt_ids` (proven by `tiktoken::tests::prompt_id_oracle_cross_check`).
///
/// # Errors
/// A tokenizer encode error (impossible for this fixed ASCII prompt).
pub fn ocr_prompt_ids(tk: &Tiktoken, format: bool) -> FocrResult<Vec<u32>> {
    let system = "<|im_start|>system\n        You should follow the instructions carefully and explain your answers in detail.";
    let imgpad = "<imgpad>".repeat(IMAGE_TOKEN_LEN);
    let instruction = if format { "OCR with format: " } else { "OCR: " };
    let prompt = format!(
        "{system}<|im_end|><|im_start|>user\n<img>{imgpad}</img>\n{instruction}<|im_end|><|im_start|>assistant\n"
    );
    tk.encode(&prompt)
}

/// End-to-end GOT-OCR2 recognition: squash-bicubic-1024/CLIP preprocess → SAM
/// vision + connector + `<imgpad>` splice → Qwen2 dense decoder greedy generation
/// (the O(n) KV-cache decode) → tiktoken decode (specials stripped). `prefix` is the
/// arch's vision-tower tensor prefix (`model.vision_tower_high`); `max_new` caps the
/// generated length; `format` selects plain vs `OCR with format:` (.mmd) mode. Stops
/// early at `<|im_end|>`.
///
/// # Errors
/// A preprocess, vision, decode, or tokenizer error.
pub fn recognize(
    weights: &Weights,
    statics: &GotStatics,
    tk: &Tiktoken,
    img: &DynamicImage,
    max_new: usize,
    format: bool,
) -> FocrResult<String> {
    let tv = std::time::Instant::now();
    let image = preprocess::got_view_tensor(img);
    let prompt_ids = ocr_prompt_ids(tk, format)?;
    let inputs_embeds = build_inputs_embeds(statics, &image, &prompt_ids)?;
    super::timing_log(&format!(
        "  got.vision+splice {:.2}s",
        tv.elapsed().as_secs_f64()
    ));
    let tg = std::time::Instant::now();
    let mut cfg = DecoderConfig::got_ocr2();
    cfg.no_repeat_ngram_size = no_repeat_ngram_override(cfg.no_repeat_ngram_size);
    // The O(n)-per-token KV-cache decode (B9): one seeding prefill then a full-causal
    // decode step per token, all int8 GEMMs through the n-parallel `gemv`.
    let ids =
        decoder_qwen2::generate_greedy_kvcache(weights, &cfg, &inputs_embeds, max_new, EOS_ID)?;
    super::timing_log(&format!(
        "  got.generate {} tokens {:.2}s",
        ids.len(),
        tg.elapsed().as_secs_f64()
    ));
    Ok(tk.decode_skip_special(&ids)?.trim().to_string())
}

/// Batched GOT recognition over MANY pages (A7.5, bd-3jo6.1.7.5): vision +
/// splice run SEQUENTIALLY per page (one live forward at a time, doctrine #5),
/// then ONE continuous-batch greedy decode over every page's `inputs_embeds`
/// ([`decoder_qwen2::generate_greedy_batched`] — per page byte-identical to
/// [`recognize`], the scheduler-level gate proves it). Returns one decoded
/// string per input page, in input order.
///
/// # Errors
/// As [`recognize`].
pub fn recognize_batch(
    weights: &Weights,
    statics: &GotStatics,
    tk: &Tiktoken,
    imgs: &[&DynamicImage],
    max_new: usize,
    format: bool,
) -> FocrResult<Vec<String>> {
    let prompt_ids = ocr_prompt_ids(tk, format)?;
    let tv = std::time::Instant::now();
    // The model-constant tensors arrive pre-hydrated from the OcrModel cache
    // (bd-av64.10): batches AND sequential pages share one hydration.
    let mut embeds_list: Vec<Mat> = Vec::with_capacity(imgs.len());
    for img in imgs {
        let image = preprocess::got_view_tensor(img);
        embeds_list.push(build_inputs_embeds(statics, &image, &prompt_ids)?);
    }
    super::timing_log(&format!(
        "  got.vision+splice(batch of {}) {:.2}s",
        imgs.len(),
        tv.elapsed().as_secs_f64()
    ));
    let tg = std::time::Instant::now();
    let mut cfg = DecoderConfig::got_ocr2();
    cfg.no_repeat_ngram_size = no_repeat_ngram_override(cfg.no_repeat_ngram_size);
    let caps = vec![max_new; embeds_list.len()];
    let id_streams =
        decoder_qwen2::generate_greedy_batched(weights, &cfg, &embeds_list, &caps, EOS_ID)?;
    super::timing_log(&format!(
        "  got.generate(batch of {}) {} tokens {:.2}s",
        imgs.len(),
        id_streams.iter().map(Vec::len).sum::<usize>(),
        tg.elapsed().as_secs_f64()
    ));
    id_streams
        .iter()
        .map(|ids| Ok(tk.decode_skip_special(ids)?.trim().to_string()))
        .collect()
}

/// `[r, c]` row-major → `[c, r]` row-major (channel-major SAM output → token-major).
fn transpose(m: &Mat) -> Mat {
    let (r, c) = (m.rows, m.cols);
    let mut out = vec![0.0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = m.data[i * c + j];
        }
    }
    Mat::from_vec(c, r, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **B11 — the committed GOT `focr ocr` e2e regression gate.** Runs the WHOLE
    /// pipeline (preprocess → vision → splice → KV-cache decode → tiktoken) on the
    /// committed `sample_text.png` and asserts the exact golden text (the forward is
    /// int8-bit-deterministic). Env-gated: `FOCR_GOT_MODEL` (the got-ocr2 weights) +
    /// `FOCR_GOT_TIKTOKEN` (qwen.tiktoken); skip-with-success when absent. Fast now
    /// that generation is O(n) (B9 KV cache).
    #[test]
    fn recognize_reads_the_sample_image_e2e() {
        let (Ok(model), Ok(tkp)) = (
            std::env::var("FOCR_GOT_MODEL"),
            std::env::var("FOCR_GOT_TIKTOKEN"),
        ) else {
            return;
        };
        let weights = Weights::load(std::path::Path::new(&model)).expect("load GOT weights");
        let tk = Tiktoken::from_qwen_tiktoken(&std::fs::read(&tkp).expect("qwen.tiktoken"))
            .expect("tiktoken");
        let img = image::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/got/sample_text.png"
        ))
        .expect("sample image");

        let statics = hydrate_statics(&weights, "model.vision_tower_high").expect("statics");
        let text = recognize(&weights, &statics, &tk, &img, 64, false).expect("recognize");
        eprintln!("[B11 e2e] {text:?}");
        assert_eq!(
            text,
            "HelloGOT-OCR2.0 Thequickbrownfaxjumps overthelazydog. 1234567890+=% Invoice#A-4217Total:$1,234.56",
            "GOT e2e OCR output regressed"
        );
    }

    /// **A7.5 — the dense batch-spine LOSSLESS e2e gate (bd-3jo6.1.7.5).**
    /// Model-gated (FOCR_GOT_MODEL + FOCR_GOT_TIKTOKEN, skip-with-SUCCESS):
    /// [`recognize_batch`] over TWO different pages must equal [`recognize`]
    /// run per page — string-identical, the armed twin of the in-module
    /// scheduler gates (and the durable form of the manual `ocr-batch`
    /// byte-identity proof).
    #[test]
    fn recognize_batch_matches_sequential_e2e() {
        let (Ok(model), Ok(tkp)) = (
            std::env::var("FOCR_GOT_MODEL"),
            std::env::var("FOCR_GOT_TIKTOKEN"),
        ) else {
            eprintln!(
                r#"{{"test":"got_batch_e2e","event":"result","result":"skip_no_model","reason":"FOCR_GOT_MODEL/FOCR_GOT_TIKTOKEN unset","native_path_ran":true,"fallback_target":"/nonexistent"}}"#
            );
            return;
        };
        let weights = Weights::load(std::path::Path::new(&model)).expect("load GOT weights");
        let tk = Tiktoken::from_qwen_tiktoken(&std::fs::read(&tkp).expect("qwen.tiktoken"))
            .expect("tiktoken");
        let img1 = image::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/got/sample_text.png"
        ))
        .expect("page 1");
        let img2 = image::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/got/format_corpus/table.png"
        ))
        .expect("page 2");

        let statics = hydrate_statics(&weights, "model.vision_tower_high").expect("statics");
        let solo: Vec<String> = [&img1, &img2]
            .iter()
            .map(|im| {
                recognize(&weights, &statics, &tk, im, 64, false).expect("sequential recognize")
            })
            .collect();
        let batched = recognize_batch(&weights, &statics, &tk, &[&img1, &img2], 64, false)
            .expect("batched recognize");
        assert_eq!(
            solo, batched,
            "A7.5 LOSSLESS contract broken: batched != sequential on the armed model"
        );
        eprintln!(
            r#"{{"test":"got_batch_e2e","event":"result","result":"pass","pages":2,"identical":true}}"#
        );
    }

    /// **bd-3kix phase 1 — the `--format` corpus smoke gates.** Runs the WHOLE
    /// pipeline in `OCR with format:` (.mmd) mode on one synthetic
    /// `tests/fixtures/got/format_corpus/` asset (see its README; generated by
    /// `scripts/gen_got_format_corpus.py`) and asserts non-empty structured output
    /// containing at least one LENIENT structural marker. Deliberately NOT a golden
    /// or CER gate: exact per-asset budgets are phase 2, once the real-model
    /// outputs have been eyeballed (`--nocapture` prints them). Env-gated like B11
    /// (`FOCR_GOT_MODEL` + `FOCR_GOT_TIKTOKEN`; skip-with-success when absent), and
    /// skip-with-success if the asset itself wasn't generated (molecule/music are
    /// optional at generation time).
    fn format_corpus_smoke(asset: &str, markers: &[&str]) {
        let (Ok(model), Ok(tkp)) = (
            std::env::var("FOCR_GOT_MODEL"),
            std::env::var("FOCR_GOT_TIKTOKEN"),
        ) else {
            return;
        };
        let path = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/got/format_corpus"
        ))
        .join(asset);
        if !path.exists() {
            eprintln!("[format corpus] {asset} not generated (optional asset) — skipping");
            return;
        }
        let weights = Weights::load(std::path::Path::new(&model)).expect("load GOT weights");
        let tk = Tiktoken::from_qwen_tiktoken(&std::fs::read(&tkp).expect("qwen.tiktoken"))
            .expect("tiktoken");
        let img = image::open(&path).expect("corpus image");

        let statics = hydrate_statics(&weights, "model.vision_tower_high").expect("statics");
        let text = recognize(&weights, &statics, &tk, &img, 512, true).expect("recognize --format");
        eprintln!("[format corpus {asset}] {text:?}");
        assert!(!text.is_empty(), "{asset}: `--format` output is empty");
        assert!(
            markers.iter().any(|m| text.contains(m)),
            "{asset}: `--format` output {text:?} contains none of the lenient markers {markers:?}"
        );
    }

    /// formula.png — `E = mc^2 + \frac{1}{2}\int_0^1 x^2 dx` (mathtext render).
    /// Any faithful math-LaTeX reading carries the `=` and a LaTeX escape.
    #[test]
    fn format_corpus_formula_smoke_e2e() {
        format_corpus_smoke("formula.png", &["=", "\\"]);
    }

    /// table.png — bordered Item/Qty/Price grid. Format mode emits LaTeX `tabular`
    /// (`&` column separators) or Markdown pipes; the cell digits are the fallback.
    #[test]
    fn format_corpus_table_smoke_e2e() {
        format_corpus_smoke("table.png", &["&", "|", "tabular", "17", "42"]);
    }

    /// chart.png — 4-bar chart, values 3/7/5/9 printed on the bars, title
    /// "Widget output". A chart-mode read carries a bar value or the title word.
    #[test]
    fn format_corpus_chart_smoke_e2e() {
        format_corpus_smoke("chart.png", &["7", "9", "Widget"]);
    }

    /// molecule.png — aspirin (RDKit 2D). Every SMILES spelling of aspirin
    /// contains a carbonyl `=O`.
    #[test]
    fn format_corpus_molecule_smoke_e2e() {
        format_corpus_smoke("molecule.png", &["=O"]);
    }

    /// music.png — 2-bar `**kern` staff (Verovio engraving). A kern-shaped read
    /// carries interpretation (`*`) or barline (`=`) tokens.
    #[test]
    fn format_corpus_music_smoke_e2e() {
        format_corpus_smoke("music.png", &["*", "kern", "="]);
    }

    /// **B7 — the `OCR with format:` (.mmd) prompt swaps only the instruction.** Fast
    /// (tokenizer only, env-gated on `FOCR_GOT_TIKTOKEN`). The plain form is the
    /// certified 287-id L0c stream; format adds 2 ids (`OCR: `→`OCR with format: `).
    #[test]
    fn format_prompt_swaps_the_instruction() {
        let Ok(tkp) = std::env::var("FOCR_GOT_TIKTOKEN") else {
            return;
        };
        let tk = Tiktoken::from_qwen_tiktoken(&std::fs::read(&tkp).expect("qwen.tiktoken"))
            .expect("tiktoken");
        let plain = ocr_prompt_ids(&tk, false).unwrap();
        let fmt = ocr_prompt_ids(&tk, true).unwrap();
        assert_eq!(plain.len(), 287, "plain L0c prompt is 287 ids");
        assert_eq!(
            fmt.len(),
            289,
            "format adds 2 ids (OCR: -> OCR with format: )"
        );
        assert_eq!(
            plain.iter().filter(|&&i| i == IMG_PAD_ID).count(),
            IMAGE_TOKEN_LEN
        );
        assert_eq!(
            fmt.iter().filter(|&&i| i == IMG_PAD_ID).count(),
            IMAGE_TOKEN_LEN
        );
        // the "OCR with format: " instruction tokenizes to these ids (from L0a corpus).
        assert!(
            fmt.windows(5).any(|w| w == [93495, 448, 3561, 25, 220]),
            "format instruction ids missing"
        );
    }

    /// **B3 — the GOT vision/connector/splice parity gate.** Env-gated: `FOCR_GOT_MODEL`
    /// = the got-ocr2 weights (`.focrq` or safetensors — vision is HP either way),
    /// `FOCR_ORACLE_IMAGE` = the oracle's own preprocessed image `[3,1024,1024]`
    /// (raw f32), `FOCR_ORACLE_HIDDEN0` = the oracle post-splice decoder input
    /// `[287,1024]`. Feeding the oracle's image isolates the vision kernels +
    /// connector + splice from the resample tolerance; the assembled inputs_embeds
    /// must match `hidden_0` tightly.
    #[test]
    fn vision_splice_matches_oracle_hidden0() {
        let (Ok(model), Ok(img), Ok(h0)) = (
            std::env::var("FOCR_GOT_MODEL"),
            std::env::var("FOCR_ORACLE_IMAGE"),
            std::env::var("FOCR_ORACLE_HIDDEN0"),
        ) else {
            return;
        };
        let weights = Weights::load(std::path::Path::new(&model)).expect("load GOT weights");

        // the committed 287-id GOT plain-OCR prompt (256 <imgpad>).
        const L0C: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/got/l0c_prompt.json"
        ));
        let v: serde_json::Value = serde_json::from_str(L0C).unwrap();
        let prompt_ids: Vec<u32> = v["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            prompt_ids.iter().filter(|&&i| i == IMG_PAD_ID).count(),
            IMAGE_TOKEN_LEN
        );

        let img_flat = read_f32_le(&img);
        let side = 1024usize;
        assert_eq!(img_flat.len(), 3 * side * side, "image not [3,1024,1024]");
        let image = Mat::from_vec(3, side * side, img_flat);

        let statics = hydrate_statics(&weights, "model.vision_tower_high").expect("statics");
        let embeds =
            build_inputs_embeds(&statics, &image, &prompt_ids).expect("build inputs_embeds");
        assert_eq!(embeds.rows, prompt_ids.len());
        assert_eq!(embeds.cols, 1024);

        let oracle = read_f32_le(&h0);
        assert_eq!(oracle.len(), embeds.data.len(), "hidden0 shape mismatch");
        let (cos, max_abs) = cosine_maxabs(&embeds.data, &oracle);
        eprintln!(
            "[B3 vision] inputs_embeds vs oracle hidden_0: cos={cos:.6} max_abs={max_abs:.4}"
        );
        assert!(
            cos >= 0.999,
            "inputs_embeds cosine {cos:.6} < 0.999 — vision/splice diverged"
        );
    }

    fn read_f32_le(path: &str) -> Vec<f32> {
        std::fs::read(path)
            .expect("blob")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    fn cosine_maxabs(a: &[f32], b: &[f32]) -> (f64, f32) {
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        let na: f64 = a
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            .sqrt();
        let nb: f64 = b
            .iter()
            .map(|&y| f64::from(y) * f64::from(y))
            .sum::<f64>()
            .sqrt();
        let max_abs = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0, f32::max);
        (dot / (na * nb), max_abs)
    }
}
