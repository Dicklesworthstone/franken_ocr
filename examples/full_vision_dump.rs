//! Full franken_ocr vision tower (SAM -> CLIP -> projector bridge) on baidu's
//! exact sam_in, dumping CLIP [257,1024] and projector [256,1280] for parity.
use anyhow::{Context, Result, bail};
use franken_ocr::native_engine::tensor::Mat;
use franken_ocr::native_engine::weights::Weights;
use franken_ocr::native_engine::{vision_bridge, vision_clip, vision_sam};
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
    Ok(Mat::from_vec(3, side * side, data))
}

fn dump(path: &str, m: &Mat) -> Result<()> {
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("create {path}"))?,
    );
    for v in &m.data {
        f.write_all(&v.to_le_bytes())
            .with_context(|| format!("write {path}"))?;
    }
    eprintln!("  wrote [{}, {}] -> {path}", m.rows, m.cols);
    Ok(())
}

fn main() -> Result<()> {
    let mut a = std::env::args().skip(1);
    let model = a.next().context("usage: missing model path")?;
    let sam_in = a.next().context("usage: missing sam_in.f32 path")?;
    let clip_out = a.next().context("usage: missing clip_out.f32 path")?;
    let bridge_out = a.next().context("usage: missing bridge_out.f32 path")?;

    eprintln!("loading weights ...");
    let w = Weights::load(Path::new(&model)).context("weights")?;
    let img = read_sam_input(&sam_in)?;

    eprintln!("SAM ...");
    let sam = vision_sam::forward(&w, &img).context("sam")?;
    eprintln!("  sam [{},{}]", sam.rows, sam.cols);
    eprintln!("CLIP ...");
    let clip = vision_clip::forward(&w, &img, &sam).context("clip")?;
    eprintln!("  clip [{},{}]", clip.rows, clip.cols);
    eprintln!("bridge ...");
    let bridge = vision_bridge::forward(&w, &clip, &sam).context("bridge")?;
    eprintln!("  bridge [{},{}]", bridge.rows, bridge.cols);

    dump(&clip_out, &clip)?;
    dump(&bridge_out, &bridge)?;
    eprintln!("FULL_VISION_DONE");
    Ok(())
}
