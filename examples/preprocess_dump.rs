//! Dump franken_ocr's Base(1024) preprocessed global-view tensor as raw LE f32
//! ([3,1024,1024], channel-major) for stage-0 parity vs baidu's image_ori.
use anyhow::{Context, Result};
use franken_ocr::preprocess::{PreprocessMode, PreprocessProfile, preprocess_image};
use std::io::Write;
use std::path::Path;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let page = args
        .next()
        .context("usage: preprocess_dump <page.png> <out.f32>")?;
    let out = args
        .next()
        .context("usage: preprocess_dump <page.png> <out.f32>")?;
    let pre = preprocess_image(
        Path::new(&page),
        PreprocessMode::base(),
        PreprocessProfile::V1,
    )
    .with_context(|| format!("preprocess {page}"))?;
    let m = &pre.global.pixels;
    eprintln!(
        "global.pixels rows={} cols={} (expect 3 x 1048576)",
        m.rows, m.cols
    );
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(&out).with_context(|| format!("create {out}"))?,
    );
    for v in &m.data {
        f.write_all(&v.to_le_bytes())
            .with_context(|| format!("write {out}"))?;
    }
    eprintln!("wrote {} f32 -> {}", m.data.len(), out);
    Ok(())
}
