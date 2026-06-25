//! Pure-Rust byte-level BPE tokenizer over `tokenizer.json` ([SPEC-019,
//! SPEC-035], PROPOSED_ARCHITECTURE.md §6.1). **Stub module** — bd-1gv.1.
//!
//! Encode/decode (pre-tokenizer `Sequence`, byte-fallback, merges) + the
//! special-token table (bos 0, eos 1, pad, `<image>` 128815,
//! `<|ref|>`/`<|det|>`/`<|grounding|>`/`<|User|>`/`<|Assistant|>`). Token-id-exact
//! vs `LlamaTokenizerFast` is an L0/L4 prerequisite — a mismatch corrupts every
//! downstream gate.

use std::path::Path;

use crate::error::{FocrError, FocrResult};

/// Hardcoded special-token ids ([SPEC-014/019]).
pub mod special {
    /// `<｜begin▁of▁sentence｜>` ([SPEC-014]).
    pub const BOS: u32 = 0;
    /// `<｜end▁of▁sentence｜>` ([SPEC-014]).
    pub const EOS: u32 = 1;
    /// `<image>` ([SPEC-019]); the runtime hardcodes this id.
    pub const IMAGE: u32 = 128815;
    /// `<|ref|>` ([SPEC-019]).
    pub const REF: u32 = 128816;
    /// `<|/ref|>` ([SPEC-019]).
    pub const REF_END: u32 = 128817;
    /// `<|det|>` ([SPEC-019]).
    pub const DET: u32 = 128818;
    /// `<|/det|>` ([SPEC-019]).
    pub const DET_END: u32 = 128819;
    /// `<|grounding|>` ([SPEC-019]).
    pub const GROUNDING: u32 = 128820;
    /// `<|User|>` ([SPEC-019]).
    pub const USER: u32 = 128825;
    /// `<|Assistant|>` ([SPEC-019]).
    pub const ASSISTANT: u32 = 128826;
}

/// The byte-level BPE tokenizer, loaded from a `tokenizer.json`.
///
/// Stub: the vocab/merges + pre-tokenizer state land with the tokenizer bead.
#[derive(Debug, Default)]
pub struct Tokenizer {
    _private: (),
}

impl Tokenizer {
    /// Load the tokenizer from a `tokenizer.json` at `path`.
    ///
    /// # Errors
    /// Always [`FocrError::NotImplemented`] in the skeleton.
    pub fn load(_path: &Path) -> FocrResult<Self> {
        Err(FocrError::NotImplemented(
            "tokenizer::Tokenizer::load — BPE tokenizer lands in Phase 1 (bd-1gv.1)".into(),
        ))
    }

    /// Encode `text` to token ids (no special tokens added — [SPEC-035]).
    ///
    /// # Errors
    /// Always [`FocrError::NotImplemented`] in the skeleton.
    pub fn encode(&self, _text: &str) -> FocrResult<Vec<u32>> {
        Err(FocrError::NotImplemented(
            "tokenizer::Tokenizer::encode — BPE encode lands in Phase 1 (bd-1gv.1)".into(),
        ))
    }

    /// Decode token ids back to text ([SPEC-110]).
    ///
    /// # Errors
    /// Always [`FocrError::NotImplemented`] in the skeleton.
    pub fn decode(&self, _ids: &[u32]) -> FocrResult<String> {
        Err(FocrError::NotImplemented(
            "tokenizer::Tokenizer::decode — BPE decode lands in Phase 1 (bd-1gv.1)".into(),
        ))
    }
}
