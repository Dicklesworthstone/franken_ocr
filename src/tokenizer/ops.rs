//! The shared tokenizer surface (A6, bd-3jo6.1.6): an object-safe trait —
//! [`TokenizerOps`] — over the two concrete tokenizer engines, so the
//! multi-model decode driver (A7) can take `&dyn TokenizerOps` instead of
//! hardcoding a concrete type per model family:
//!
//! * [`super::Tokenizer`] — merge-list byte-level BPE over `tokenizer.json`
//!   (Baidu Unlimited-OCR / DeepSeek lineage; the SmolLM2 vocab (C6) is the
//!   same HF `tokenizer.json` engine).
//! * [`super::tiktoken::Tiktoken`] — raw-byte tiktoken over `qwen.tiktoken`
//!   (GOT-OCR2.0; OneChart shares the GOT/Vary lineage, D9).
//!
//! The trait is **additive**: both concrete types keep every inherent method
//! and every existing call site compiles unchanged — each trait method
//! delegates to the like-named inherent method, so routing through the trait
//! is byte-identical to calling the concrete type (zero behavior change, the
//! parity-first doctrine).
//!
//! The surface is the minimal set the engine actually calls today, derived
//! from the real call sites (do-NOT-invent-surface):
//! * `encode` — the Baidu prompt builder (`native_engine::build_prompt`) and
//!   the GOT prompt builder (`native_engine::got::ocr_prompt_ids`).
//! * `decode` — the Baidu single-page and batch-spine detokenize paths.
//! * `decode_skip_special` — the GOT recognize tail (specials stripped).
//! * `bos_id` — the Baidu prompt head (a single prepended BOS).
//! * `eos_id` / `token_to_id` — the id lookups a model-agnostic driver needs
//!   (both already inherent on both types; see the per-method contracts).
//!
//! Deliberately NOT on the trait (model-specific or engine-unused):
//! `image_id` (Baidu `<image>` 128815) vs `image_pad_id` (GOT `<imgpad>`
//! 151859) have per-model splice semantics — a generic caller resolves such
//! surfaces via [`TokenizerOps::token_to_id`]; `pad_id`/`vocab_size`/
//! `id_to_token` have no engine call site (and the two `id_to_token` inherent
//! signatures differ: `Option<&str>` vs `Option<String>`).

use crate::error::FocrResult;

use super::Tokenizer;
use super::tiktoken::Tiktoken;

/// The tokenizer operations shared by every model family's tokenizer — the
/// object-safe seam the multi-model decode driver (A7) is written against.
///
/// Implementations MUST be token-id-exact with their reference tokenizer (the
/// L0 conformance gates in [`super`] / [`super::tiktoken`]); the trait adds no
/// semantics of its own — each method's contract is the concrete engine's.
pub trait TokenizerOps {
    /// Encode `text` to token ids, with special/added-token surfaces that
    /// appear literally in `text` resolved to their single control id (HF
    /// `AddedVocabulary` splitting for the BPE engine; tiktoken
    /// `allowed_special="all"` for the Qwen engine). No BOS/EOS is
    /// auto-added — prompt builders own their own framing.
    ///
    /// # Errors
    /// Only on a corrupt vocab (a byte-level symbol / single-byte rank
    /// missing) — impossible after each engine's load-time validation.
    fn encode(&self, text: &str) -> FocrResult<Vec<u32>>;

    /// Decode ids → `String`, special tokens INCLUDED (their literal surface
    /// form). The BPE engine is strict UTF-8 (a malformed reassembly errors);
    /// the tiktoken engine is lossy (a single id can hold a partial-UTF-8
    /// fragment) — both per their inherent `decode` contracts.
    ///
    /// # Errors
    /// An unknown / out-of-range id, or (BPE engine) invalid reassembled
    /// UTF-8.
    fn decode(&self, ids: &[u32]) -> FocrResult<String>;

    /// Decode, dropping ids flagged special (`skip_special_tokens=True`) —
    /// the clean-text emit path.
    ///
    /// # Errors
    /// See [`TokenizerOps::decode`].
    fn decode_skip_special(&self, ids: &[u32]) -> FocrResult<String>;

    /// Beginning-of-sequence id (Baidu `<｜begin▁of▁sentence｜>` 0; GOT
    /// `<|endoftext|>` 151643).
    #[must_use]
    fn bos_id(&self) -> u32;

    /// End-of-sequence id per the tokenizer's own config (Baidu 1; GOT
    /// 151643). NOTE: the *generation stop id* is the ARCH's contract, not
    /// necessarily this — GOT stops at `<|im_end|>` (151645), not at its
    /// `eos_id()`.
    #[must_use]
    fn eos_id(&self) -> u32;

    /// The id for an exact token surface string (special/added tokens first,
    /// then a whole base-vocab token), or `None`. The generic form of the
    /// model-specific splice-token accessors (`image_id`/`image_pad_id`).
    #[must_use]
    fn token_to_id(&self, content: &str) -> Option<u32>;
}

/// The byte-level-BPE `tokenizer.json` engine (Baidu Unlimited-OCR) — pure
/// delegation to the inherent methods (zero behavior change).
impl TokenizerOps for Tokenizer {
    fn encode(&self, text: &str) -> FocrResult<Vec<u32>> {
        Tokenizer::encode(self, text)
    }

    fn decode(&self, ids: &[u32]) -> FocrResult<String> {
        Tokenizer::decode(self, ids)
    }

    fn decode_skip_special(&self, ids: &[u32]) -> FocrResult<String> {
        Tokenizer::decode_skip_special(self, ids)
    }

    fn bos_id(&self) -> u32 {
        Tokenizer::bos_id(self)
    }

    fn eos_id(&self) -> u32 {
        Tokenizer::eos_id(self)
    }

    fn token_to_id(&self, content: &str) -> Option<u32> {
        Tokenizer::token_to_id(self, content)
    }
}

/// The raw-byte tiktoken engine (GOT-OCR2.0 Qwen vocab) — pure delegation to
/// the inherent methods (zero behavior change).
impl TokenizerOps for Tiktoken {
    fn encode(&self, text: &str) -> FocrResult<Vec<u32>> {
        Tiktoken::encode(self, text)
    }

    fn decode(&self, ids: &[u32]) -> FocrResult<String> {
        Tiktoken::decode(self, ids)
    }

    fn decode_skip_special(&self, ids: &[u32]) -> FocrResult<String> {
        Tiktoken::decode_skip_special(self, ids)
    }

    fn bos_id(&self) -> u32 {
        Tiktoken::bos_id(self)
    }

    fn eos_id(&self) -> u32 {
        Tiktoken::eos_id(self)
    }

    fn token_to_id(&self, content: &str) -> Option<u32> {
        Tiktoken::token_to_id(self, content)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{special, tests as bpe_tests};
    use super::*;
    use crate::tokenizer::tiktoken;

    /// Compile-time object-safety guarantee: A7's driver takes
    /// `&dyn TokenizerOps`, so the trait must stay dyn-compatible.
    const _OBJECT_SAFE: fn(&dyn TokenizerOps) = |_| {};

    // ── fixtures ─────────────────────────────────────────────────────────────

    /// The byte-level-BPE engine over the SAME tiny synthetic `tokenizer.json`
    /// the unit tests in [`super::super::tests`] use (one fixture, no drift).
    fn bpe() -> Tokenizer {
        Tokenizer::from_json_bytes(bpe_tests::tiny_json().as_bytes()).expect("tiny tokenizer loads")
    }

    /// Minimal standard base64 ENCODER (test-only; the runtime only decodes).
    /// Inverse of [`tiktoken`]'s hand-rolled `b64_decode`.
    fn b64_encode(bytes: &[u8]) -> String {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let b = [
                chunk[0],
                *chunk.get(1).unwrap_or(&0),
                *chunk.get(2).unwrap_or(&0),
            ];
            let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
            out.push(ALPHABET[(n >> 18) as usize & 63] as char);
            out.push(ALPHABET[(n >> 12) as usize & 63] as char);
            out.push(if chunk.len() > 1 {
                ALPHABET[(n >> 6) as usize & 63] as char
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                ALPHABET[n as usize & 63] as char
            } else {
                '='
            });
        }
        out
    }

    /// A synthetic, FULL-SIZE `qwen.tiktoken` (the loader fail-closes on
    /// anything but 151643 dense ranks with all 256 single bytes), so the
    /// tiktoken impl is exercised through the trait WITHOUT the real 2.4 MB
    /// vocab: ranks 0..=255 are the single bytes themselves, rank 256 is
    /// `b"ab"` and 257 is `b"abc"` (to drive the merge loop — mirroring the
    /// tiny BPE fixture's `a+b→ab`, `ab+c→abc` merges), and every remaining
    /// rank is a unique 5-byte filler that no test text can contain.
    fn synthetic_qwen_tiktoken() -> Vec<u8> {
        let mut file = String::new();
        for b in 0u8..=255 {
            file.push_str(&b64_encode(&[b]));
            file.push(' ');
            file.push_str(&b.to_string());
            file.push('\n');
        }
        file.push_str(&format!("{} 256\n", b64_encode(b"ab")));
        file.push_str(&format!("{} 257\n", b64_encode(b"abc")));
        for r in 258u32..151_643 {
            let filler = [0xC0, 0xC1, (r >> 16) as u8, (r >> 8) as u8, r as u8];
            file.push_str(&format!("{} {r}\n", b64_encode(&filler)));
        }
        file.into_bytes()
    }

    fn tik() -> Tiktoken {
        Tiktoken::from_qwen_tiktoken(&synthetic_qwen_tiktoken())
            .expect("synthetic qwen.tiktoken loads")
    }

    // ── both impls through &dyn (the A7 driver shape) ────────────────────────

    /// Driver-style round trip through the vtable: encode → decode must
    /// reproduce the text on both engines.
    fn assert_round_trip(tk: &dyn TokenizerOps, text: &str) {
        let ids = tk.encode(text).expect("encode");
        assert_eq!(
            tk.decode(&ids).expect("decode"),
            text,
            "round trip {text:?}"
        );
    }

    #[test]
    fn both_impls_round_trip_through_dyn() {
        let bpe = bpe();
        let tik = tik();
        // Fixture-safe texts (the tiny BPE vocab covers a..f + space).
        for text in ["abc", "ab", " a"] {
            assert_round_trip(&bpe, text);
        }
        // The synthetic tiktoken vocab covers every byte.
        for text in ["abc", "abd", "Hello, world! 123", "café"] {
            assert_round_trip(&tik, text);
        }
    }

    #[test]
    fn dyn_dispatch_matches_inherent_exactly() {
        // Zero behavior change: the trait ids are the inherent ids, verbatim.
        let bpe = bpe();
        let dyn_bpe: &dyn TokenizerOps = &bpe;
        assert_eq!(
            dyn_bpe.encode("ab<image>c").unwrap(),
            Tokenizer::encode(&bpe, "ab<image>c").unwrap()
        );
        let tik = tik();
        let dyn_tik: &dyn TokenizerOps = &tik;
        assert_eq!(
            dyn_tik.encode("ab<|endoftext|>").unwrap(),
            Tiktoken::encode(&tik, "ab<|endoftext|>").unwrap()
        );
    }

    // ── byte-level BPE through the trait (same fixture as the unit tests) ────

    #[test]
    fn bpe_encode_and_merges_through_trait() {
        let t = bpe();
        let t: &dyn TokenizerOps = &t;
        // Same expectations as the inherent-method unit tests (mod.rs).
        assert_eq!(t.encode("abc").unwrap(), vec![8]);
        assert_eq!(t.encode("ab").unwrap(), vec![7]);
        assert_eq!(t.encode("ba").unwrap(), vec![1, 0]);
        // Added/special surfaces resolve to their single control id.
        assert_eq!(t.encode("ab<image>c").unwrap(), vec![7, 128815, 2]);
    }

    #[test]
    fn bpe_special_handling_through_trait() {
        let t = bpe();
        let t: &dyn TokenizerOps = &t;
        let ids = t.encode("ab<image>c").unwrap();
        assert_eq!(t.decode(&ids).unwrap(), "ab<image>c");
        // <image> is special:true → dropped; <|x|> (special:false) is kept.
        assert_eq!(t.decode_skip_special(&ids).unwrap(), "abc");
        let ids2 = t.encode("a<|x|>b").unwrap();
        assert_eq!(t.decode_skip_special(&ids2).unwrap(), "a<|x|>b");
    }

    #[test]
    fn bpe_id_lookups_through_trait() {
        let t = bpe();
        let t: &dyn TokenizerOps = &t;
        assert_eq!(t.bos_id(), special::BOS);
        assert_eq!(t.eos_id(), special::EOS);
        assert_eq!(t.token_to_id("<image>"), Some(special::IMAGE));
        // Base-vocab fallback after the added-token table.
        assert_eq!(t.token_to_id("abc"), Some(8));
        assert_eq!(t.token_to_id("no-such-token"), None);
    }

    // ── tiktoken through the trait (synthetic full-size vocab) ───────────────

    #[test]
    fn tiktoken_encode_and_merges_through_trait() {
        let t = tik();
        let t: &dyn TokenizerOps = &t;
        // Whole-piece fast path: b"abc" is rank 257.
        assert_eq!(t.encode("abc").unwrap(), vec![257]);
        // Merge loop: a|b|d → "ab" (rank 256) merges, "abd" is no rank →
        // [256, b'd' as rank].
        assert_eq!(t.encode("abd").unwrap(), vec![256, u32::from(b'd')]);
        // No merge: b"ba" is no rank → the two single-byte ranks.
        assert_eq!(
            t.encode("ba").unwrap(),
            vec![u32::from(b'b'), u32::from(b'a')]
        );
        // Digit-split canary survives the trait: single \p{N} pre-tokens.
        assert_eq!(
            t.encode("12").unwrap(),
            vec![u32::from(b'1'), u32::from(b'2')]
        );
    }

    #[test]
    fn tiktoken_special_handling_through_trait() {
        let t = tik();
        let t: &dyn TokenizerOps = &t;
        // allowed_special="all": the literal surface becomes the control id.
        assert_eq!(
            t.encode("ab<|endoftext|>").unwrap(),
            vec![256, tiktoken::ENDOFTEXT]
        );
        let ids = t.encode("ab<img></img>").unwrap();
        assert_eq!(ids, vec![256, tiktoken::IMG_START, tiktoken::IMG_END]);
        assert_eq!(t.decode(&ids).unwrap(), "ab<img></img>");
        assert_eq!(t.decode_skip_special(&ids).unwrap(), "ab");
    }

    #[test]
    fn tiktoken_id_lookups_through_trait() {
        let t = tik();
        let t: &dyn TokenizerOps = &t;
        assert_eq!(t.bos_id(), tiktoken::ENDOFTEXT);
        assert_eq!(t.eos_id(), tiktoken::ENDOFTEXT);
        assert_eq!(t.token_to_id("<imgpad>"), Some(tiktoken::IMG_PAD));
        assert_eq!(t.token_to_id("<|im_end|>"), Some(tiktoken::IM_END));
        // Base-rank fallback after the special table.
        assert_eq!(t.token_to_id("ab"), Some(256));
        assert_eq!(t.token_to_id("no-such-token"), None);
    }

    // ── real-vocab smoke through the trait (env-gated, model-gated pattern) ──

    #[test]
    fn real_baidu_tokenizer_through_trait() {
        // Same gate + fixture as `super::super::tests::load_real` (reused).
        let Some(t) = bpe_tests::load_real() else {
            return;
        };
        let t: &dyn TokenizerOps = &t;
        assert_eq!(t.encode("<image>").unwrap(), vec![special::IMAGE]);
        assert_eq!(t.bos_id(), special::BOS);
        assert_eq!(t.eos_id(), special::EOS);
        assert_eq!(t.token_to_id("<|grounding|>"), Some(special::GROUNDING));
        assert_round_trip(t, "The quick brown fox jumps over the lazy dog.");
    }

    /// Mirrors `tiktoken::tests::load_real` (kept local: that test module is
    /// private to the `tiktoken` file).
    fn load_real_tiktoken() -> Option<Tiktoken> {
        let p = std::env::var("FOCR_GOT_TIKTOKEN").ok()?;
        let bytes = std::fs::read(p).ok()?;
        Some(Tiktoken::from_qwen_tiktoken(&bytes).expect("real qwen.tiktoken must parse"))
    }

    #[test]
    fn real_got_tiktoken_through_trait() {
        let Some(t) = load_real_tiktoken() else {
            return;
        };
        let t: &dyn TokenizerOps = &t;
        // The digit-split canary ids, via the vtable.
        assert_eq!(
            t.encode("1234567890").unwrap(),
            vec![16, 17, 18, 19, 20, 21, 22, 23, 24, 15]
        );
        assert_eq!(t.bos_id(), tiktoken::ENDOFTEXT);
        assert_eq!(t.eos_id(), tiktoken::ENDOFTEXT);
        assert_eq!(t.token_to_id("<imgpad>"), Some(tiktoken::IMG_PAD));
        assert_round_trip(t, "Hello, world!");
    }
}
