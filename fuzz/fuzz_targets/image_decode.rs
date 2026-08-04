//! `image_decode` — fuzz the untrusted-image ingest (bd-10sb.1):
//! `preprocess_bytes` decodes arbitrary bytes through the `image` crate and
//! runs the full Base-mode preprocess (resize/pad/normalize). Contract: a
//! typed `FocrError::InputDecode` (§7.4 exit 4) on junk, a well-formed
//! `Preprocessed` on real images — never a panic or unbounded loop. Seeds:
//! tiny valid PNG/JPEG + truncations (fuzz/corpus/image_decode/).
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = franken_ocr::preprocess::preprocess_bytes(
        data,
        franken_ocr::preprocess::PreprocessMode::Base { base_size: 1024 },
    );
});
