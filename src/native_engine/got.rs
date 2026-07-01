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
pub fn vision_features(weights: &Weights, image: &Mat, prefix: &str) -> FocrResult<Mat> {
    let sam = vision_sam::forward_prefix(weights, image, prefix)?; // [1024, 256] channel-major
    let sam_t = transpose(&sam); // [256, 1024] token-major
    let w = weights.vec("model.mm_projector_vary.weight")?; // [1024*1024] row-major [out,in]
    let b = weights.vec("model.mm_projector_vary.bias")?; // [1024]
    let proj = vision_sam::Linear {
        w,
        b,
        out: 1024,
        in_: 1024,
    };
    proj.apply(&sam_t) // [256, 1024]
}

/// Build the GOT decoder `inputs_embeds`: embed the prompt id-stream against the
/// tied `model.embed_tokens.weight`, then `masked_scatter` the vision features
/// into the 256 `<imgpad>` rows (in prompt order). Returns `[seq, hidden]`.
///
/// # Errors
/// A vision/embed error, or a [`connector::masked_scatter`] mismatch (the number
/// of `<imgpad>` rows must equal `vision_features.rows`).
pub fn build_inputs_embeds(
    weights: &Weights,
    image: &Mat,
    prompt_ids: &[u32],
    prefix: &str,
) -> FocrResult<Mat> {
    let tokens = vision_features(weights, image, prefix)?; // [256, 1024]
    let embed = weights.mat("model.embed_tokens.weight")?; // [vocab, hidden]
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
    tk: &Tiktoken,
    img: &DynamicImage,
    prefix: &str,
    max_new: usize,
    format: bool,
) -> FocrResult<String> {
    let image = preprocess::got_view_tensor(img);
    let prompt_ids = ocr_prompt_ids(tk, format)?;
    let inputs_embeds = build_inputs_embeds(weights, &image, &prompt_ids, prefix)?;
    let cfg = DecoderConfig::got_ocr2();
    // The O(n)-per-token KV-cache decode (B9) — bit-identical to the certified
    // re-prefill `generate_greedy`, but usable on real (long) documents.
    let ids =
        decoder_qwen2::generate_greedy_kvcache(weights, &cfg, &inputs_embeds, max_new, EOS_ID)?;
    Ok(tk.decode_skip_special(&ids)?.trim().to_string())
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

        let text = recognize(&weights, &tk, &img, "model.vision_tower_high", 64, false)
            .expect("recognize");
        eprintln!("[B11 e2e] {text:?}");
        assert_eq!(
            text,
            "HelloGOT-OCR2.0 Thequickbrownfaxjumps overthelazydog. 1234567890+=% Invoice#A-4217Total:$1,234.56",
            "GOT e2e OCR output regressed"
        );
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

        let embeds = build_inputs_embeds(&weights, &image, &prompt_ids, "model.vision_tower_high")
            .expect("build inputs_embeds");
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
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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
