//! `pretok_split` — fuzz the pure pretokenizers over arbitrary Unicode
//! (bd-10sb.1): the hand-rolled split classes (contractions, letter/number
//! runs, CJK ranges, whitespace lookahead) are the tokenizer's
//! untrusted-text surface. Contract: total (no panic, no unbounded loop) on
//! EVERY `&str`, and the split NEVER produces an empty piece (an empty piece
//! would loop the BPE merge stage). Seeds: ASCII / CJK / emoji / digit-run /
//! whitespace samples (fuzz/corpus/pretok_split/).
#![no_main]

use franken_ocr::tokenizer::pretok;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|text: &str| {
    for piece in pretok::pretokenize(text) {
        assert!(!piece.is_empty(), "pretokenize produced an empty piece");
    }
    for piece in pretok::pretokenize_smollm2(text) {
        assert!(!piece.is_empty(), "pretokenize_smollm2 produced an empty piece");
    }
    for piece in pretok::pretokenize_gpt2(text) {
        assert!(!piece.is_empty(), "pretokenize_gpt2 produced an empty piece");
    }
});
