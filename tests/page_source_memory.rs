//! `page_source_memory` — U2-D0's memory gate: a document read through a
//! `PageSource` holds the pages in flight, never the whole document.
//!
//! The `--multi-page` input path used to rasterize every page and then clone
//! the vector, so a 300-page book held every raster twice before the model
//! read one (plan R14). This walks 100-page documents the way the document
//! pass does — rasterize, hand off, drop — and samples the process's memory
//! after every page. Three page kinds are walked, because each decodes
//! through a different allocator path: a FlateDecode PDF, a DCTDecode (JPEG)
//! PDF, and a directory of PNG files.
//!
//! The probe is the physical footprint where macOS's `footprint` exists, else
//! the resident set (`ps -o rss=`); both are read without `unsafe` platform
//! calls. Without either the test logs a skip and passes. An eager control
//! (every raster collected at once) runs last and must register, so a probe
//! that cannot see held rasters cannot pass the walk vacuously.
//!
//! The test is alone in its binary on purpose: memory is per process, and
//! other tests running beside it would allocate into the number.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use franken_ocr::page_source::PageSource;
use image::{ImageEncoder, RgbImage};

const TEST: &str = "page_source_memory";
const PAGES: usize = 100;
/// One page: 1200×1600 RGB, 5.5 MiB as a raster (a 150-dpi letter scan).
const WIDTH: u32 = 1200;
const HEIGHT: u32 = 1600;
/// A walk may grow by this many pages' worth: the page in flight plus decoder
/// scratch and allocator slack. Holding the document is 100.
const ALLOWED_GROWTH_PAGES: u64 = 16;

fn log_line(case: &str, event: &str, result: &str, fields: &str) {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    eprintln!(
        r#"{{"schema_version":1,"ts":{ts},"test":"{TEST}","case":"{case}","run_seq":0,"event":"{event}","result":"{result}",{fields}}}"#
    );
}

/// This process's resident set in bytes, or `None` without a usable `ps`.
fn rss_bytes() -> Option<u64> {
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    let kib: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    Some(kib * 1024)
}

/// This process's physical footprint in bytes (macOS `footprint`), or `None`
/// where the tool is absent.
fn footprint_bytes() -> Option<u64> {
    let out = Command::new("footprint")
        .arg(std::process::id().to_string())
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let rest = text.split("Footprint:").nth(1)?;
    let mut words = rest.split_whitespace();
    let value: f64 = words.next()?.parse().ok()?;
    let unit = match words.next()? {
        "B" => 1.0,
        "KB" => 1024.0,
        "MB" => 1024.0 * 1024.0,
        "GB" => 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    Some((value * unit) as u64)
}

/// The memory this process holds: its physical footprint where `footprint`
/// exists (macOS RSS also counts freed pages the allocator has handed back
/// but the kernel has not yet reclaimed), else its resident set.
fn held_bytes() -> Option<(u64, &'static str)> {
    footprint_bytes()
        .map(|b| (b, "footprint"))
        .or_else(|| rss_bytes().map(|b| (b, "rss")))
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// The page every document repeats: a horizontal gradient, so it compresses
/// small in every codec while still decoding to a full-size raster.
fn page_raster() -> RgbImage {
    RgbImage::from_fn(WIDTH, HEIGHT, |x, _| {
        let v = u8::try_from(x % 256).expect("fits");
        image::Rgb([v, v, 255 - v])
    })
}

/// A PDF of `PAGES` pages sharing one image XObject, stored as FlateDecode
/// samples or as a DCTDecode JPEG.
fn synthetic_book(dir: &Path, jpeg: bool) -> PathBuf {
    use lopdf::{Object, Stream, dictionary};
    let raster = page_raster();
    let mut doc = lopdf::Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let mut dict = dictionary! {
        "Type" => "XObject",
        "Subtype" => "Image",
        "Width" => i64::from(WIDTH),
        "Height" => i64::from(HEIGHT),
        "ColorSpace" => "DeviceRGB",
        "BitsPerComponent" => 8,
    };
    let stream = if jpeg {
        let mut bytes = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 85)
            .write_image(
                raster.as_raw(),
                WIDTH,
                HEIGHT,
                image::ExtendedColorType::Rgb8,
            )
            .expect("encode the page JPEG");
        dict.set("Filter", "DCTDecode");
        Stream::new(dict, bytes)
    } else {
        let mut stream = Stream::new(dict, raster.into_raw());
        stream.compress().expect("compress the page image");
        stream
    };
    let image_id = doc.add_object(stream);
    let mut kids: Vec<Object> = Vec::new();
    for _ in 0..PAGES {
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Resources" => dictionary! { "XObject" => dictionary! { "Im0" => image_id } },
            "MediaBox" => vec![0.into(), 0.into(), i64::from(WIDTH).into(), i64::from(HEIGHT).into()],
        });
        kids.push(page_id.into());
    }
    let count = i64::try_from(PAGES).expect("page count fits");
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! { "Type" => "Pages", "Kids" => kids, "Count" => count }),
    );
    let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog_id);
    let path = dir.join(if jpeg { "dct.pdf" } else { "flate.pdf" });
    doc.save(&path).expect("save the synthetic book");
    path
}

/// A directory of `PAGES` copies of one PNG page.
fn page_directory(dir: &Path) -> PathBuf {
    let pages = dir.join("pages");
    std::fs::create_dir_all(&pages).expect("create the page directory");
    let first = pages.join("p001.png");
    page_raster().save(&first).expect("write the page PNG");
    for n in 2..=PAGES {
        std::fs::copy(&first, pages.join(format!("p{n:03}.png"))).expect("copy the page PNG");
    }
    pages
}

/// Walk `source` page by page, dropping each raster, and return the peak
/// growth of the held memory over the walk.
fn walk_growth(source: PageSource) -> (usize, u64) {
    let held = || held_bytes().map_or(0, |(b, _)| b);
    let baseline = held();
    let mut peak = baseline;
    let mut rasterized = 0usize;
    for item in source {
        let raster = item.raster.expect("every synthetic page rasterizes");
        assert_eq!((raster.width(), raster.height()), (WIDTH, HEIGHT));
        rasterized += 1;
        // The pass preprocesses here; the raster is dropped either way.
        drop(raster);
        peak = peak.max(held());
    }
    (rasterized, peak.saturating_sub(baseline))
}

#[test]
fn a_hundred_page_walk_holds_the_pages_in_flight_not_the_document() {
    let Some((_, metric)) = held_bytes() else {
        log_line(
            "walk",
            "skip",
            "skip",
            r#""reason":"neither `footprint` nor `ps -o rss=` is available on this host""#,
        );
        return;
    };
    let dir = std::env::temp_dir().join(format!("focr_page_source_memory_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create the scratch directory");
    let flate = synthetic_book(&dir, false);
    let dct = synthetic_book(&dir, true);
    let pngs = page_directory(&dir);

    let page_bytes = u64::from(WIDTH) * u64::from(HEIGHT) * 3;
    let document_bytes = page_bytes * PAGES as u64;
    let allowed = page_bytes * ALLOWED_GROWTH_PAGES;
    let legs: [(&str, PageSource); 3] = [
        (
            "flate_pdf",
            PageSource::pdf(&flate).expect("open the Flate book"),
        ),
        (
            "dct_pdf",
            PageSource::pdf(&dct).expect("open the JPEG book"),
        ),
        (
            "png_directory",
            PageSource::directory(&pngs).expect("open the page directory"),
        ),
    ];
    let mut failures = Vec::new();
    for (case, source) in legs {
        let (rasterized, growth) = walk_growth(source);
        assert_eq!(rasterized, PAGES, "{case}: every page walked");
        let pass = growth < allowed;
        log_line(
            case,
            "result",
            if pass { "pass" } else { "fail" },
            &format!(
                r#""metric":"{metric}","pages":{rasterized},"page_raster_mib":{:.1},"document_raster_mib":{:.1},"growth_mib":{:.1},"growth_in_pages":{:.2},"allowed_pages":{ALLOWED_GROWTH_PAGES}"#,
                mib(page_bytes),
                mib(document_bytes),
                mib(growth),
                growth as f64 / page_bytes as f64,
            ),
        );
        if !pass {
            failures.push(format!("{case}: {:.1} MiB", mib(growth)));
        }
    }

    // The control: the old eager shape, every raster held at once.
    let before = held_bytes().map_or(0, |(b, _)| b);
    let all: Vec<_> = PageSource::pdf(&flate)
        .expect("reopen the Flate book")
        .map(|item| item.raster.expect("every synthetic page rasterizes"))
        .collect();
    let eager = held_bytes().map_or(0, |(b, _)| b).saturating_sub(before);
    drop(all);
    let _ = std::fs::remove_dir_all(&dir);
    log_line(
        "eager_control",
        "result",
        "pass",
        &format!(
            r#""metric":"{metric}","pages":{PAGES},"growth_mib":{:.1}"#,
            mib(eager)
        ),
    );
    assert!(
        eager > document_bytes / 2,
        "the {metric} probe saw only {:.1} MiB for {PAGES} held rasters of {:.1} MiB: it \
         cannot judge the walks",
        mib(eager),
        mib(page_bytes)
    );
    assert!(
        failures.is_empty(),
        "{metric} grew past {ALLOWED_GROWTH_PAGES} pages' worth ({:.1} MiB) walking {PAGES} \
         pages of {:.1} MiB each — the walk is holding memory it should have released: {}",
        mib(allowed),
        mib(page_bytes),
        failures.join(", ")
    );
}
