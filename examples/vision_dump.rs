//! Run franken_ocr's wired vision_sam::forward on a raw [3,1024,1024] f32 input
//! (baidu's exact sam_in) and dump the [1024,256] SAM feature as LE f32, for
//! parity vs baidu's sam_out.npy.
use anyhow::{Context, Result, bail};
use franken_ocr::native_engine::tensor::Mat;
use franken_ocr::native_engine::vision_sam;
use franken_ocr::native_engine::weights::Weights;
use std::io::{Read, Write};
use std::path::Path;

fn read_sam_input(path: &str) -> Result<Mat> {
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

    let data: Vec<f32> = buf
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    if !data.len().is_multiple_of(3) {
        bail!(
            "{path}: sam input f32 count {} is not divisible by 3 channels",
            data.len()
        );
    }
    let pixels_per_channel = data.len() / 3;
    let side = (pixels_per_channel as f64).sqrt() as usize;
    if side.checked_mul(side) != Some(pixels_per_channel) {
        bail!("{path}: sam input pixels per channel {pixels_per_channel} is not a square");
    }
    eprintln!("sam_in n={} -> [3, {}*{}]", data.len(), side, side);
    Ok(Mat::from_vec(3, side * side, data))
}

fn dump(path: &str, data: &[f32]) -> Result<()> {
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("create {path}"))?,
    );
    for v in data {
        f.write_all(&v.to_le_bytes())
            .with_context(|| format!("write {path}"))?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut a = std::env::args().skip(1);
    let model = a.next().context("usage: missing model shard path")?;
    let sam_in = a.next().context("usage: missing sam_in.f32 path")?;
    let out = a.next().context("usage: missing out.f32 path")?;

    eprintln!("loading weights from {model} ...");
    let w = Weights::load(Path::new(&model)).context("weights load")?;

    let img = read_sam_input(&sam_in)?;

    let feat = vision_sam::forward(&w, &img).context("vision_sam::forward")?;
    eprintln!(
        "sam_out rows={} cols={} (expect 1024 x 256)",
        feat.rows, feat.cols
    );
    dump(&out, &feat.data)?;
    eprintln!("wrote {} f32 -> {out}", feat.data.len());
    Ok(())
}
