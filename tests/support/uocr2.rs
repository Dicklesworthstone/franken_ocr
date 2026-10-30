//! Unlimited-OCR-2 decoder inputs for the model-gated tests that drive the
//! decoder directly: the parity ladder's v2 lane (`tests/parity_ladder.rs`)
//! and the metamorphic suite's MR-6 (`tests/metamorphic.rs`). The raw
//! checkpoint read into memory, a document's paged prompt, and its native
//! `inputs_embeds` (the engine's private vision tower recomposed from its
//! public parts).
//!
//! Each binary uses a subset, so unused items are allowed.
#![allow(dead_code)]

use std::path::Path;

use franken_ocr::native_engine::tensor::Mat;
use franken_ocr::native_engine::weights::Weights;
use franken_ocr::native_engine::{connector, decoder, vision_bridge, vision_clip, vision_sam};
use franken_ocr::preprocess;
use franken_ocr::tokenizer::special;

/// The raw Unlimited-OCR-2 checkpoint in `dir` (its shard index and both
/// shards), read into memory: BF16 that `Weights::mat` widens to f32.
pub fn load_uocr2_source(dir: &Path) -> Weights {
    use franken_ocr::native_engine::shards;
    let index = shards::read_index(&dir.join(shards::SHARD_INDEX_NAME)).expect("v2 shard index");
    let bytes = index
        .shards()
        .iter()
        .map(|name| std::fs::read(dir.join(name)).expect("v2 shard reads"))
        .collect();
    Weights::from_shard_bytes(&index, bytes).expect("v2 weights load")
}

/// The ids of v2's task text, "Multi page parsing.", from the committed
/// tokenizer golden.
fn task_text_ids() -> Vec<u32> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/tokenizer_baidu/expected_uocr2.json"
    );
    let golden: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).expect("the uocr2 tokenizer golden"))
            .expect("the uocr2 tokenizer golden parses");
    let row = golden["fixtures"]
        .as_array()
        .expect("golden fixtures")
        .iter()
        .find(|r| r["text"] == "<PAGE><image>Multi page parsing.")
        .expect("the one-page prompt is in the golden");
    let ids: Vec<u32> = row["ids"]
        .as_array()
        .expect("golden ids")
        .iter()
        .map(|id| u32::try_from(id.as_u64().expect("an id")).expect("a u32 id"))
        .collect();
    assert_eq!(ids[..2], [special::PAGE, special::IMAGE]);
    ids[2..].to_vec()
}

/// v2's paged prompt over `pages` (U2-B6; MU:531-580): BOS, then per page
/// `<PAGE>` and its image slots, then the task text. The engine's own builder
/// is private; `paged_prompt_reproduces_the_v2_oracle_prompts` pins the form
/// against the oracle, and callers check the result's sha256 against
/// `ORACLE.json`.
pub fn uocr2_document_prompt(pages: &[preprocess::Preprocessed]) -> Vec<u32> {
    let mut ids = vec![0];
    for page in pages {
        ids.push(special::PAGE);
        ids.extend(std::iter::repeat_n(
            special::IMAGE,
            page.placeholder_token_count(),
        ));
    }
    ids.extend(task_text_ids());
    ids
}

/// Native `inputs_embeds` for a v2 prompt: `embed_tokens` over `ids` with the
/// pages' vision features (`vision_sam` → `vision_clip` → `vision_bridge`, the
/// engine's private tower recomposed from its public parts) scattered into the
/// `<image>` rows by the connector, page by page in base mode, global after
/// tiles in gundam mode.
pub fn uocr2_inputs_embeds(w: &Weights, ids: &[u32], pages: &[preprocess::Preprocessed]) -> Mat {
    let vision = |view: &Mat| -> Mat {
        let sam = vision_sam::forward(w, view).expect("vision_sam");
        let clip = vision_clip::forward(w, view, &sam).expect("vision_clip");
        vision_bridge::forward(w, &clip, &sam).expect("vision_bridge")
    };
    let table = w.mat("model.embed_tokens.weight").expect("embed table");
    let mut embeds =
        decoder::embed_tokens(&table.data, table.rows, table.cols, ids).expect("embed_tokens");
    let mask: Vec<bool> = ids.iter().map(|&id| id == special::IMAGE).collect();
    let newline = w.vec("model.image_newline").expect("image_newline");
    let separator = w.vec("model.view_seperator").expect("view_seperator");
    let grid = preprocess::num_queries(pages[0].mode.base_size());
    if let preprocess::PreprocessMode::Gundam { tile_size, .. } = pages[0].mode {
        assert_eq!(pages.len(), 1, "a gundam fixture is one page");
        let page = &pages[0];
        let locals: Vec<Mat> = page.tiles.iter().map(|tile| vision(&tile.pixels)).collect();
        let tile = preprocess::num_queries(tile_size);
        connector::fuse_crop(
            w,
            &mut embeds,
            &locals,
            page.crop_grid.width_crop_num,
            page.crop_grid.height_crop_num,
            tile,
            tile,
            &vision(&page.global.pixels),
            grid,
            grid,
            &newline,
            &separator,
            &mask,
        )
        .expect("connector::fuse_crop");
    } else {
        let globals: Vec<Mat> = pages.iter().map(|p| vision(&p.global.pixels)).collect();
        connector::fuse_no_crop(
            w,
            &mut embeds,
            &globals,
            grid,
            grid,
            &newline,
            &separator,
            &mask,
        )
        .expect("connector::fuse_no_crop");
    }
    embeds
}
