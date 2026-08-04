//! Run franken_ocr's wired DeepSeek-V2 MoE decoder on baidu's EXACT prefill
//! `inputs_embeds` (the `[seq,1280]` activation entering decoder layer 0, AFTER
//! baidu's vision-token scatter) and dump the final `model.norm`-ready hidden
//! `[seq,1280]` plus the last-position `lm_head` logits `[129280]` as raw LE
//! f32. This decouples decoder parity from the prompt-scatter + KV-cache, exactly
//! as the vision dumps decoupled the tower from preprocessing.
//!
//! Usage: decoder_dump <model.safetensors> <inputs_embeds.f32> <hidden_out.f32> <logits_last.f32>
//! `inputs_embeds.f32` is row-major `[seq,1280]` LE f32; `seq` is inferred from
//! the file length (`len/4/1280`).
use anyhow::{Context, Result, bail};
use franken_ocr::native_engine::decoder;
use franken_ocr::native_engine::tensor::Mat;
use franken_ocr::native_engine::weights::Weights;
use std::io::{Read, Write};
use std::path::Path;

const HIDDEN: usize = 1280;

fn read_f32(path: &str) -> Result<Vec<f32>> {
    let mut buf = Vec::new();
    std::fs::File::open(path)
        .with_context(|| format!("open {path}"))?
        .read_to_end(&mut buf)
        .with_context(|| format!("read {path}"))?;
    if !buf.len().is_multiple_of(4) {
        bail!(
            "{path}: raw f32 byte length {} is not divisible by 4",
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

fn dump(path: &str, data: &[f32]) -> Result<()> {
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("create {path}"))?,
    );
    for v in data {
        f.write_all(&v.to_le_bytes())
            .with_context(|| format!("write {path}"))?;
    }
    f.flush().with_context(|| format!("flush {path}"))?;
    Ok(())
}

fn main() -> Result<()> {
    let mut a = std::env::args().skip(1);
    let model = a.next().context("usage: missing model shard path")?;
    let embeds_path = a.next().context("usage: missing inputs_embeds.f32 path")?;
    let hidden_out = a.next().context("usage: missing hidden_out.f32 path")?;
    let logits_out = a.next().context("usage: missing logits_last.f32 path")?;

    eprintln!("loading weights from {model} ...");
    let w = Weights::load(Path::new(&model)).context("weights load")?;

    let data = read_f32(&embeds_path)?;
    if !data.len().is_multiple_of(HIDDEN) {
        bail!(
            "{embeds_path}: inputs_embeds len {} is not a multiple of hidden {HIDDEN}",
            data.len()
        );
    }
    let seq = data.len() / HIDDEN;
    if seq == 0 {
        bail!("{embeds_path}: inputs_embeds has zero rows");
    }
    eprintln!("inputs_embeds [{seq}, {HIDDEN}] -> decoder ...");
    let embeds = Mat::from_vec(seq, HIDDEN, data);

    let hidden = decoder::forward(&w, &embeds).context("decoder::forward")?;
    eprintln!("decoder hidden [{}, {}]", hidden.rows, hidden.cols);
    if hidden.rows == 0 {
        bail!("decoder::forward returned zero hidden rows");
    }

    // lm_head on the LAST hidden row only (bit-identical to projecting all rows
    // then slicing — proved by decoder::lm_head_last_row_is_full_last_row).
    let last = Mat::from_vec(1, hidden.cols, hidden.row(hidden.rows - 1).to_vec());
    let logits = decoder::lm_head(&w, &last).context("decoder::lm_head")?;
    eprintln!("lm_head logits [{}, {}]", logits.rows, logits.cols);

    // Argmax of the last-position logits = franken's first generated token id.
    let (mut argmax, mut best) = (0usize, f32::NEG_INFINITY);
    for (i, &v) in logits.data.iter().enumerate() {
        if v > best {
            best = v;
            argmax = i;
        }
    }
    eprintln!("FRANKEN_FIRST_TOKEN_ID {argmax} (logit {best})");

    dump(&hidden_out, &hidden.data)?;
    dump(&logits_out, &logits.data)?;
    eprintln!("DECODER_DUMP_DONE seq={seq}");
    Ok(())
}
