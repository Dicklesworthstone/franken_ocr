//! Standalone end-to-end franken_ocr OCR pipeline for a single page, bypassing
//! the still-NotImplemented `native_engine::mod.rs` CLI glue.
//!
//! Closes the parity ladder by proving the *last* unproven per-stage link — the
//! vision-token SCATTER — then greedy-decodes the page from franken's OWN fused
//! `inputs_embeds` and writes the generated token ids for detokenization.
//!
//! Pipeline (no-crop / base-1024 path, matching baidu `infer(crop_mode=False,
//! base_size=1024, image_size=1024)`):
//!   1. preprocess_image(page, Base{1024})          -> global pixels [3,1024*1024]
//!   2. vision_sam -> vision_clip -> vision_bridge   -> hybrid features [256,1280]
//!   3. embed_tokens(input_ids)                      -> text embeds   [seq,1280]
//!   4. assemble_global_block(features) + masked_scatter -> inputs_embeds [seq,1280]
//!   5. compare vs baidu's dumped inputs_embeds      -> cosine + max|Δ|  (SCATTER proof)
//!   6. greedy decode (no KV cache): hidden=decoder::forward; logits=lm_head;
//!      next=argmax(last row); append embed(next); until EOS or cap.
//!
//! Usage:
//!   e2e_ocr <model.safetensors> <page.png> <input_ids.json> <images_seq_mask.json> \
//!           <baidu_inputs_embeds.f32> <out_ids.json> [max_new_tokens] [eos_id]
use anyhow::{Context, Result, bail};
use franken_ocr::native_engine::tensor::Mat;
use franken_ocr::native_engine::weights::Weights;
use franken_ocr::native_engine::{connector, decoder, vision_bridge, vision_clip, vision_sam};
use franken_ocr::preprocess::{PreprocessMode, preprocess_image};
use std::io::Read;
use std::path::Path;
use std::time::Instant;

const HIDDEN: usize = 1280;
const GRID: usize = 16; // 16x16 base-1024 hybrid grid

fn read_f32(path: &str) -> Result<Vec<f32>> {
    let mut buf = Vec::new();
    std::fs::File::open(path)
        .with_context(|| format!("open {path}"))?
        .read_to_end(&mut buf)
        .with_context(|| format!("read {path}"))?;
    if buf.len() % 4 != 0 {
        bail!(
            "{path}: f32 dump byte length {} is not divisible by 4",
            buf.len()
        );
    }
    Ok(buf
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn read_json_u32(path: &str) -> Result<Vec<u32>> {
    let s = std::fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    let v: Vec<i64> =
        serde_json::from_str(&s).with_context(|| format!("parse {path} as JSON integer array"))?;
    v.into_iter()
        .map(|x| {
            u32::try_from(x).with_context(|| format!("{path}: token id {x} is out of u32 range"))
        })
        .collect()
}

fn read_json_bool(path: &str) -> Result<Vec<bool>> {
    let s = std::fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    serde_json::from_str(&s).with_context(|| format!("parse {path} as JSON bool array"))
}

/// cosine + max|Δ| over the flattened matrices.
fn compare(a: &[f32], b: &[f32]) -> Result<(f64, f64)> {
    if a.len() != b.len() {
        bail!("compare len mismatch {} vs {}", a.len(), b.len());
    }
    let (mut dot, mut na, mut nb, mut maxd) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        maxd = maxd.max((x - y).abs());
    }
    Ok((dot / (na.sqrt() * nb.sqrt()), maxd))
}

fn argmax(row: &[f32]) -> usize {
    let (mut idx, mut best) = (0usize, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > best {
            best = v;
            idx = i;
        }
    }
    idx
}

fn main() -> Result<()> {
    let mut a = std::env::args().skip(1);
    let model = a.next().context("usage: missing model.safetensors")?;
    let page = a.next().context("usage: missing page.png")?;
    let ids_path = a.next().context("usage: missing input_ids.json")?;
    let mask_path = a.next().context("usage: missing images_seq_mask.json")?;
    let baidu_embeds_path = a.next().context("usage: missing baidu_inputs_embeds.f32")?;
    let out_ids_path = a.next().context("usage: missing out_ids.json")?;
    let max_new: usize = match a.next() {
        Some(s) => s
            .parse()
            .with_context(|| format!("parse max_new_tokens argument {s:?}"))?,
        None => 256,
    };
    let eos_id: u32 = match a.next() {
        Some(s) => s
            .parse()
            .with_context(|| format!("parse eos_id argument {s:?}"))?,
        None => 1,
    };
    // Optional: a raw [3,1024*1024] LE-f32 sam_in to feed the vision tower
    // INSTEAD of franken's own preprocess_image — used to isolate the SCATTER
    // proof from any preprocessing discrepancy (decoupled, like the vision proof).
    let sam_in_override = a.next();

    let t_load = Instant::now();
    eprintln!("[e2e] loading weights from {model} ...");
    let w = Weights::load(Path::new(&model)).context("weights load")?;
    eprintln!(
        "[e2e] weights loaded in {:.1}s",
        t_load.elapsed().as_secs_f64()
    );

    // ── Stage 1-2: preprocess + vision tower ────────────────────────────────
    let t_vis = Instant::now();
    let img: Mat = if let Some(ref p) = sam_in_override {
        let data = read_f32(p)?;
        if data.len() % 3 != 0 {
            bail!("{p}: sam_in f32 count {} is not divisible by 3", data.len());
        }
        let pixels_per_channel = data.len() / 3;
        let side = (pixels_per_channel as f64).sqrt() as usize;
        if side * side != pixels_per_channel {
            bail!("{p}: sam_in pixel count per channel {pixels_per_channel} is not a square");
        }
        eprintln!(
            "[e2e] sam_in OVERRIDE {p} -> [3,{}*{}] (preprocess BYPASSED)",
            side, side
        );
        Mat::from_vec(3, side * side, data)
    } else {
        let pre =
            preprocess_image(Path::new(&page), PreprocessMode::base()).context("preprocess")?;
        let g = pre.global.pixels;
        eprintln!("[e2e] preprocessed global [{},{}]", g.rows, g.cols);
        g
    };
    let img = &img;
    let sam = vision_sam::forward(&w, img).context("vision_sam")?;
    let clip = vision_clip::forward(&w, img, &sam).context("vision_clip")?;
    let bridge = vision_bridge::forward(&w, &clip, &sam).context("vision_bridge")?;
    eprintln!(
        "[e2e] vision tower -> hybrid features [{},{}] in {:.1}s",
        bridge.rows,
        bridge.cols,
        t_vis.elapsed().as_secs_f64()
    );
    if bridge.rows != GRID * GRID {
        bail!(
            "expected {} hybrid feature rows, got {}",
            GRID * GRID,
            bridge.rows
        );
    }
    if bridge.cols != HIDDEN {
        bail!("expected hidden width {HIDDEN}, got {}", bridge.cols);
    }

    // ── Stage 3: embed baidu's exact prompt id-stream ───────────────────────
    let ids = read_json_u32(&ids_path)?;
    let mask = read_json_bool(&mask_path)?;
    if ids.len() != mask.len() {
        bail!("ids/mask length mismatch: {} vs {}", ids.len(), mask.len());
    }
    let n_img = mask.iter().filter(|&&b| b).count();
    eprintln!(
        "[e2e] input_ids seq={} (image placeholders={}, image_token_id count={})",
        ids.len(),
        n_img,
        ids.iter().filter(|&&x| x == 128815).count()
    );

    let embed_tbl = w
        .mat("model.embed_tokens.weight")
        .context("embed_tokens.weight")?;
    let vocab = embed_tbl.rows;
    eprintln!("[e2e] embed table [{vocab},{}]", embed_tbl.cols);
    let mut inputs_embeds =
        decoder::embed_tokens(&embed_tbl.data, vocab, HIDDEN, &ids).context("embed_tokens")?;

    // ── Stage 4: assemble vision block + scatter into the placeholder slots ──
    let image_newline = w.vec("model.image_newline").context("image_newline")?;
    let view_seperator = w.vec("model.view_seperator").context("view_seperator")?;
    let block =
        connector::assemble_global_block(&bridge, GRID, GRID, &image_newline, &view_seperator)
            .context("assemble_global_block")?;
    eprintln!(
        "[e2e] vision block [{},{}] (expect 273 = 16*17 + 1); scatter into {} masked slots",
        block.rows, block.cols, n_img
    );
    connector::masked_scatter(&mut inputs_embeds, &block, &mask).context("masked_scatter")?;

    // ── Stage 5: SCATTER PARITY vs baidu's dumped inputs_embeds ─────────────
    let baidu = read_f32(&baidu_embeds_path)?;
    let (cos, maxd) = compare(&inputs_embeds.data, &baidu)?;
    println!("SCATTER_PARITY cosine={cos:.8} max_abs_delta={maxd:.6e}");
    eprintln!("[e2e] scatter parity: cosine={cos:.8} max|Δ|={maxd:.6e} (target cosine >= 0.999)");

    // ── Stage 6: greedy decode (no KV cache, O(n^2), correctness-first) ─────
    let t_dec = Instant::now();
    let mut cur = inputs_embeds; // grows by one embedded token per step
    let mut out_ids: Vec<u32> = Vec::new();
    for step in 0..max_new {
        let hidden = decoder::forward(&w, &cur).context("decoder::forward")?;
        let last = Mat::from_vec(1, hidden.cols, hidden.row(hidden.rows - 1).to_vec());
        let logits = decoder::lm_head(&w, &last).context("lm_head")?;
        let next = argmax(&logits.data) as u32;
        out_ids.push(next);
        if step < 12 {
            eprintln!("[e2e] step {step}: next_id={next}");
        }
        if next == eos_id {
            eprintln!("[e2e] EOS ({eos_id}) at step {step}");
            break;
        }
        // Append embed(next) as a new trailing row.
        let next_usize = usize::try_from(next).context("next token id does not fit usize")?;
        if next_usize >= vocab {
            bail!("decoder produced token id {next} outside embed vocab {vocab}");
        }
        let row = &embed_tbl.data[next_usize * HIDDEN..(next_usize + 1) * HIDDEN];
        let mut data = std::mem::take(&mut cur.data);
        data.extend_from_slice(row);
        cur = Mat::from_vec(cur.rows + 1, HIDDEN, data);
    }
    let dec_secs = t_dec.elapsed().as_secs_f64();
    eprintln!(
        "[e2e] decoded {} tokens in {:.1}s ({:.2}s/token)",
        out_ids.len(),
        dec_secs,
        dec_secs / out_ids.len().max(1) as f64
    );

    let preview: Vec<u32> = out_ids.iter().take(15).copied().collect();
    println!("FRANKEN_FIRST_IDS {preview:?}");
    println!("FRANKEN_NUM_IDS {}", out_ids.len());
    let out_json = serde_json::to_string(&out_ids).context("serialize out_ids")?;
    std::fs::write(&out_ids_path, out_json).with_context(|| format!("write {out_ids_path}"))?;
    eprintln!("[e2e] wrote {} ids -> {out_ids_path}", out_ids.len());
    println!("E2E_DONE");
    Ok(())
}
