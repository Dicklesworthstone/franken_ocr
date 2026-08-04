//! `focrq_parse` — fuzz `Weights::from_bytes` over arbitrary bytes
//! (bd-10sb.1): the hand-written `.focrq` container parser (magic, version,
//! preamble, header JSON, directory census, payload bounds) must be TOTAL —
//! parse or typed `FocrError`, never a panic / OOB / hang (§7.4 exit-7
//! contract). Seed corpus: a tiny valid artifact + truncated/bit-flipped
//! variants (fuzz/corpus/focrq_parse/).
//!
//! (File name is the `cargo fuzz init` stub's, retained; the BINARY is
//! `focrq_parse` — see fuzz/Cargo.toml.)
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Ok and typed Err are both fine; a panic aborts the process and
    // libfuzzer records the crashing input.
    let _ = franken_ocr::native_engine::weights::Weights::from_bytes(data.to_vec());
});
