//! `safetensors_parse` — the same `Weights::from_bytes` totality contract as
//! `focrq_parse`, but seeded with SAFETENSORS-shaped inputs
//! (fuzz/corpus/safetensors_parse/): `header_len u64 LE | header_json |
//! payload`. The two containers share the entrypoint (magic-sniffed), so the
//! harness is identical — the CORPUS steers the fuzzer into the safetensors
//! index/offsets/dtype branches instead of the focrq preamble branches.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = franken_ocr::native_engine::weights::Weights::from_bytes(data.to_vec());
});
