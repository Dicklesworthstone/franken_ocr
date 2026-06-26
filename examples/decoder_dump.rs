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
use franken_ocr::native_engine::decoder;
use franken_ocr::native_engine::tensor::Mat;
use franken_ocr::native_engine::weights::Weights;
use std::io::{Read, Write};
use std::path::Path;

const HIDDEN: usize = 1280;

fn read_f32(path: &str) -> Vec<f32> {
    let mut buf = Vec::new();
    std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("open {path}: {e}"))
        .read_to_end(&mut buf)
        .unwrap();
    buf.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn dump(path: &str, data: &[f32]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for v in data {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
}

fn main() {
    let mut a = std::env::args().skip(1);
    let model = a.next().expect("model shard path");
    let embeds_path = a.next().expect("inputs_embeds.f32 path");
    let hidden_out = a.next().expect("hidden_out.f32 path");
    let logits_out = a.next().expect("logits_last.f32 path");

    eprintln!("loading weights from {model} ...");
    let w = Weights::load(Path::new(&model)).expect("weights load");

    let data = read_f32(&embeds_path);
    assert!(
        data.len().is_multiple_of(HIDDEN),
        "inputs_embeds len {} not a multiple of hidden {HIDDEN}",
        data.len()
    );
    let seq = data.len() / HIDDEN;
    eprintln!("inputs_embeds [{seq}, {HIDDEN}] -> decoder ...");
    let embeds = Mat::from_vec(seq, HIDDEN, data);

    let hidden = decoder::forward(&w, &embeds).expect("decoder::forward");
    eprintln!("decoder hidden [{}, {}]", hidden.rows, hidden.cols);

    // lm_head on the LAST hidden row only (bit-identical to projecting all rows
    // then slicing — proved by decoder::lm_head_last_row_is_full_last_row).
    let last = Mat::from_vec(1, hidden.cols, hidden.row(hidden.rows - 1).to_vec());
    let logits = decoder::lm_head(&w, &last).expect("decoder::lm_head");
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

    dump(&hidden_out, &hidden.data);
    dump(&logits_out, &logits.data);
    eprintln!("DECODER_DUMP_DONE seq={seq}");
}
