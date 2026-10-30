//! Lazy page sources (U2-D0): the pages of a PDF, a list of page images or a
//! directory of them, read one at a time.
//!
//! A document run used to rasterize every selected page up front and then
//! clone the whole vector, so a 300-page book held every raster twice before
//! the model read one. A [`PageSource`] rasterizes a page only when the caller
//! asks for the next one, and the caller drops the raster once it has
//! preprocessed it, so raster memory is bounded by the pages in flight.
//!
//! The source keeps the **source-page map**: one [`MappedPage`] per page it
//! has yielded, skipped pages and split-spread halves included, so a run's
//! page numbers name pages of the source however many were skipped or split.
//!
//! A page that cannot be rasterized (an unsupported codec, a vector page, an
//! unreadable image file) is yielded with its error rather than ending the
//! walk: skipping it or failing the run is the caller's decision.
//!
//! The source is `Send` and owns everything it reads, so a document pass can
//! move it onto the engine's blocking stage and rasterize there, between
//! forwards and under no lock (doctrine #5).

use std::cmp::Ordering;
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use image::DynamicImage;

use crate::error::{FocrError, FocrResult};
use crate::native_engine::{RunWarning, WarningKind, timing_log, warning_recovery};
use crate::pdf::{self, PdfPages};

/// The file extensions a directory source reads as page images
/// (case-insensitive).
pub const PAGE_IMAGE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "tif", "tiff", "webp", "bmp", "gif", "pnm", "pbm", "pgm", "ppm",
];

/// How many ignored file names an `ignored_files` warning names.
const IGNORED_EXAMPLES: usize = 3;

/// Which half of a split two-page spread a page is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Half {
    Left,
    Right,
}

impl Half {
    /// `"left"` or `"right"`: the JSON and robot `half` value.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Half::Left => "left",
            Half::Right => "right",
        }
    }
}

/// One page from a [`PageSource`].
#[derive(Debug)]
pub struct PageItem {
    /// 0-based position in the run, and in [`PageSource::page_map`].
    pub index: usize,
    /// 1-based page of the source: a PDF page, or a directory's image in
    /// natural order.
    pub source_page: usize,
    /// The half of a split spread, or `None` for a whole page.
    pub half: Option<Half>,
    /// The page's raster, or why it has none. Every raster failure is a
    /// [`FocrError::InputDecode`].
    pub raster: FocrResult<DynamicImage>,
}

/// One slot of the source-page map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MappedPage {
    /// 1-based page of the source.
    pub source_page: usize,
    /// The half of a split spread, or `None` for a whole page.
    pub half: Option<Half>,
    /// The raster's `(width, height)`; `None` for a page that was skipped.
    pub size: Option<(u32, u32)>,
}

/// The non-image files a directory source left out.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IgnoredFiles {
    /// How many there were.
    pub count: usize,
    /// The first few of their names, in natural order.
    pub examples: Vec<String>,
}

/// What a source rasterizes its pages from.
trait Rasterize: Send {
    /// Pages in the source.
    fn len(&self) -> usize;
    /// Rasterize 0-based page `idx`.
    fn render(&self, idx: usize) -> FocrResult<DynamicImage>;
}

impl Rasterize for PdfPages {
    fn len(&self) -> usize {
        PdfPages::len(self)
    }

    fn render(&self, idx: usize) -> FocrResult<DynamicImage> {
        PdfPages::render(self, idx)
    }
}

/// Page image files, decoded with the shared EXIF-aware decoder.
struct ImageFiles(Vec<PathBuf>);

impl Rasterize for ImageFiles {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn render(&self, idx: usize) -> FocrResult<DynamicImage> {
        let path = self.0.get(idx).ok_or_else(|| {
            FocrError::InputDecode(format!(
                "page index {idx} out of range ({} pages)",
                self.0.len()
            ))
        })?;
        crate::preprocess::decode_path(path)
    }
}

/// A document's pages, rasterized one at a time as the walk asks for them.
///
/// Build one with [`PageSource::open`], [`PageSource::pdf`],
/// [`PageSource::images`] or [`PageSource::directory`], narrow it with
/// [`PageSource::select`] and [`PageSource::split_spreads`] before the walk,
/// then iterate it: each [`PageItem`] carries one page's raster or its error.
pub struct PageSource {
    pages: Box<dyn Rasterize>,
    /// 0-based source pages to walk, in source order.
    selected: Vec<usize>,
    /// The next entry of `selected` to rasterize.
    next: usize,
    split_spreads: bool,
    /// A split spread's right half, yielded next: (source page, raster).
    held: Option<(usize, DynamicImage)>,
    map: Vec<MappedPage>,
    /// The directory a directory source lists, for its warning.
    dir: Option<PathBuf>,
    ignored: IgnoredFiles,
}

impl PageSource {
    /// Open `path` as a page source: a directory of page images, or else a
    /// PDF. Nothing is rasterized yet.
    ///
    /// # Errors
    /// As [`PageSource::directory`] or [`PageSource::pdf`].
    pub fn open(path: &Path) -> FocrResult<Self> {
        if path.is_dir() {
            Self::directory(path)
        } else {
            Self::pdf(path)
        }
    }

    /// The pages of the PDF at `path`.
    ///
    /// # Errors
    /// As [`PdfPages::open`].
    pub fn pdf(path: &Path) -> FocrResult<Self> {
        Ok(Self::new(Box::new(PdfPages::open(path)?)))
    }

    /// Page image files, in the order given. Each decodes with its EXIF
    /// orientation applied, as a single image does (SPEC-020).
    ///
    /// # Errors
    /// [`FocrError::Usage`] for an empty list.
    pub fn images(paths: Vec<PathBuf>) -> FocrResult<Self> {
        if paths.is_empty() {
            return Err(FocrError::Usage(
                "a page source needs at least one page image".into(),
            ));
        }
        Ok(Self::new(Box::new(ImageFiles(paths))))
    }

    /// The page images in `dir` ([`PAGE_IMAGE_EXTENSIONS`]), in natural
    /// order: `p2.png` before `p10.png`. Hidden files (`.DS_Store`) and
    /// subdirectories are passed over. Other files are left out and reported
    /// by [`PageSource::warnings`].
    ///
    /// # Errors
    /// [`FocrError::InputDecode`] when the directory cannot be read or holds
    /// no page image.
    pub fn directory(dir: &Path) -> FocrResult<Self> {
        let (pages, ignored) = list_page_images(dir)?;
        if pages.is_empty() {
            return Err(FocrError::InputDecode(format!(
                "{} has no page images (looked for .{}; subdirectories are not read)",
                dir.display(),
                PAGE_IMAGE_EXTENSIONS.join(", .")
            )));
        }
        let mut source = Self::new(Box::new(ImageFiles(pages)));
        source.dir = Some(dir.to_path_buf());
        source.ignored = ignored;
        Ok(source)
    }

    fn new(pages: Box<dyn Rasterize>) -> Self {
        let selected = (0..pages.len()).collect();
        Self {
            pages,
            selected,
            next: 0,
            split_spreads: false,
            held: None,
            map: Vec::new(),
            dir: None,
            ignored: IgnoredFiles::default(),
        }
    }

    /// Walk only the pages of a `--pages` spec ([`pdf::select_pages`]);
    /// `None` keeps every page. Call it before the walk.
    ///
    /// # Errors
    /// [`FocrError::Usage`] for a malformed or out-of-range spec.
    pub fn select(mut self, spec: Option<&str>) -> FocrResult<Self> {
        self.selected = pdf::select_pages(spec, self.pages.len())?;
        Ok(self)
    }

    /// Yield a two-page spread as its left and right halves
    /// ([`pdf::split_spread`]); a page without a detectable gutter stays
    /// whole. Call it before the walk.
    #[must_use]
    pub fn split_spreads(mut self, on: bool) -> Self {
        self.split_spreads = on;
        self
    }

    /// Pages in the source document.
    #[must_use]
    pub fn source_len(&self) -> usize {
        self.pages.len()
    }

    /// Source pages the walk reads: the selection, with a split spread
    /// counted once.
    #[must_use]
    pub fn len(&self) -> usize {
        self.selected.len()
    }

    /// Whether the walk reads no page (never true of a constructed source
    /// with its full selection).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.selected.is_empty()
    }

    /// The source-page map so far: one entry per page yielded, in run order.
    #[must_use]
    pub fn page_map(&self) -> &[MappedPage] {
        &self.map
    }

    /// The files a directory source left out (none for other sources).
    #[must_use]
    pub fn ignored_files(&self) -> &IgnoredFiles {
        &self.ignored
    }

    /// The run warnings opening the source produced: one `ignored_files`
    /// warning when a directory held files that are not page images.
    #[must_use]
    pub fn warnings(&self) -> Vec<RunWarning> {
        let IgnoredFiles { count, examples } = &self.ignored;
        if *count == 0 {
            return Vec::new();
        }
        let dir = self
            .dir
            .as_deref()
            .map_or_else(String::new, |d| d.display().to_string());
        let more = if *count > examples.len() { ", …" } else { "" };
        let mut fields = serde_json::Map::new();
        fields.insert("count".into(), (*count).into());
        fields.insert("examples".into(), examples.clone().into());
        vec![RunWarning {
            kind: "ignored_files",
            message: format!(
                "{count} file(s) in {dir} are not page images and were left out: {}{more}",
                examples.join(", ")
            ),
            recovery: warning_recovery(WarningKind::IgnoredFiles),
            fields,
        }]
    }

    /// Rasterize one page again, after the walk dropped its raster: the
    /// whole page, or one half of a spread. The result is the raster the walk
    /// yielded for that page.
    ///
    /// # Errors
    /// [`FocrError::Usage`] for a page outside the source, the page's raster
    /// error, or [`FocrError::InputDecode`] when a half is asked of a page
    /// that is not a spread.
    pub fn fetch(&self, source_page: usize, half: Option<Half>) -> FocrResult<DynamicImage> {
        let count = self.pages.len();
        if source_page == 0 || source_page > count {
            return Err(FocrError::Usage(format!(
                "page {source_page} is out of range (this document has {count} page(s))"
            )));
        }
        let image = self.pages.render(source_page - 1)?;
        let Some(half) = half else {
            return Ok(image);
        };
        let (left, right, _) = pdf::split_spread(&image).ok_or_else(|| {
            FocrError::InputDecode(format!(
                "page {source_page} is not a two-page spread, so it has no {} half",
                half.as_str()
            ))
        })?;
        Ok(match half {
            Half::Left => left,
            Half::Right => right,
        })
    }

    /// Record `source_page` (or its half) in the map and hand it out.
    fn yield_page(
        &mut self,
        source_page: usize,
        half: Option<Half>,
        raster: FocrResult<DynamicImage>,
    ) -> PageItem {
        let index = self.map.len();
        self.map.push(MappedPage {
            source_page,
            half,
            size: raster.as_ref().ok().map(|i| (i.width(), i.height())),
        });
        PageItem {
            index,
            source_page,
            half,
            raster,
        }
    }
}

impl Iterator for PageSource {
    type Item = PageItem;

    /// Rasterize the next selected page; a held right half comes first.
    fn next(&mut self) -> Option<PageItem> {
        if let Some((source_page, right)) = self.held.take() {
            return Some(self.yield_page(source_page, Some(Half::Right), Ok(right)));
        }
        let idx = *self.selected.get(self.next)?;
        self.next += 1;
        let source_page = idx + 1;
        let started = Instant::now();
        let image = match self.pages.render(idx) {
            Ok(image) => image,
            Err(e) => {
                timing_log(&format!("pagesource page {source_page}: skipped: {e}"));
                return Some(self.yield_page(source_page, None, Err(e)));
            }
        };
        let (width, height) = (image.width(), image.height());
        timing_log(&format!(
            "pagesource page {source_page}: rasterized {width}x{height}, {} bytes, {:.1} ms",
            image.as_bytes().len(),
            started.elapsed().as_secs_f64() * 1e3
        ));
        if self.split_spreads {
            if let Some((left, right, gutter_x)) = pdf::split_spread(&image) {
                timing_log(&format!(
                    "pagesource page {source_page}: spread split at x={gutter_x} ({width}x{height})"
                ));
                self.held = Some((source_page, right));
                return Some(self.yield_page(source_page, Some(Half::Left), Ok(left)));
            }
            timing_log(&format!(
                "pagesource page {source_page}: no spread detected ({width}x{height}), unsplit"
            ));
        }
        Some(self.yield_page(source_page, None, Ok(image)))
    }
}

/// The page images in `dir`, in natural order, and the other files it holds.
fn list_page_images(dir: &Path) -> FocrResult<(Vec<PathBuf>, IgnoredFiles)> {
    let unreadable = |e: std::io::Error| {
        FocrError::InputDecode(format!("read directory {}: {e}", dir.display()))
    };
    let mut pages: Vec<(String, PathBuf)> = Vec::new();
    let mut ignored: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(unreadable)? {
        let path = entry.map_err(unreadable)?.path();
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        // Hidden files (`.DS_Store`, editor swap files) are no part of a
        // document, and a page source does not descend into subdirectories.
        if name.starts_with('.') || !path.is_file() {
            continue;
        }
        if is_page_image(&path) {
            pages.push((name, path));
        } else {
            ignored.push(name);
        }
    }
    pages.sort_by(|a, b| natural_cmp(&a.0, &b.0));
    ignored.sort_by(|a, b| natural_cmp(a, b));
    let ignored = IgnoredFiles {
        count: ignored.len(),
        examples: ignored.into_iter().take(IGNORED_EXAMPLES).collect(),
    };
    Ok((pages.into_iter().map(|(_, path)| path).collect(), ignored))
}

fn is_page_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            PAGE_IMAGE_EXTENSIONS
                .iter()
                .any(|known| ext.eq_ignore_ascii_case(known))
        })
}

/// Natural order for file names: runs of ASCII digits compare as numbers
/// (`p2` < `p10`, `p01` == `p1`), other runs compare case-insensitively, and
/// names that tie compare byte-wise so the order is total.
fn natural_cmp(a: &str, b: &str) -> Ordering {
    let (mut left, mut right) = (runs(a), runs(b));
    loop {
        let ord = match (left.next(), right.next()) {
            (None, None) => return a.cmp(b),
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(x), Some(y)) => {
                let numeric = |s: &str| s.as_bytes().first().is_some_and(u8::is_ascii_digit);
                if numeric(x) && numeric(y) {
                    let (x, y) = (x.trim_start_matches('0'), y.trim_start_matches('0'));
                    x.len().cmp(&y.len()).then_with(|| x.cmp(y))
                } else {
                    x.to_lowercase().cmp(&y.to_lowercase())
                }
            }
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
}

/// `s` as alternating runs of ASCII digits and of everything else.
fn runs(s: &str) -> impl Iterator<Item = &str> {
    let mut rest = s;
    std::iter::from_fn(move || {
        let digit = rest.as_bytes().first()?.is_ascii_digit();
        let end = rest
            .find(|c: char| c.is_ascii_digit() != digit)
            .unwrap_or(rest.len());
        let (run, tail) = rest.split_at(end);
        rest = tail;
        Some(run)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdf::tests::{build_multi_page_pdf, jpeg_xobject, synth_page};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    /// A fresh, empty directory under the system temp dir.
    fn temp_dir(tag: &str) -> PathBuf {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "focr_page_source_{tag}_{}_{}",
            std::process::id(),
            SEQ.fetch_add(1, AtomicOrdering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// Write `image` to `dir/name` (the extension picks the format).
    fn write_page(dir: &Path, name: &str, image: &DynamicImage) -> PathBuf {
        let path = dir.join(name);
        image.save(&path).expect("write page image");
        path
    }

    /// A small portrait page with one text column.
    fn portrait() -> DynamicImage {
        synth_page(200, 300, &[(40, 160)])
    }

    /// A two-page spread `split_spread` cuts at its blank center gutter.
    fn spread() -> DynamicImage {
        synth_page(800, 500, &[(50, 350), (450, 750)])
    }

    fn summary(source: &mut PageSource) -> Vec<(usize, usize, Option<Half>, bool)> {
        source
            .by_ref()
            .map(|item| (item.index, item.source_page, item.half, item.raster.is_ok()))
            .collect()
    }

    #[test]
    fn natural_order_reads_numbers_as_numbers() {
        let mut names = vec![
            "p10.png",
            "p2.png",
            "P1.png",
            "p01.png",
            "p1a.png",
            "cover.png",
            "p100.png",
        ];
        names.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(
            names,
            vec![
                "cover.png",
                "P1.png",
                "p01.png",
                "p1a.png",
                "p2.png",
                "p10.png",
                "p100.png"
            ]
        );
        assert_eq!(natural_cmp("page_9", "page_10"), Ordering::Less);
        assert_eq!(natural_cmp("a", "a"), Ordering::Equal);
        assert_eq!(natural_cmp("", "a"), Ordering::Less);
    }

    #[test]
    fn a_directory_reads_its_images_in_natural_order_and_reports_the_rest() {
        let dir = temp_dir("listing");
        for name in ["p10.png", "p2.png", "p1.jpg"] {
            write_page(&dir, name, &portrait());
        }
        std::fs::write(dir.join("notes.txt"), "not a page").expect("write txt");
        std::fs::write(dir.join("README"), "not a page").expect("write readme");
        std::fs::write(dir.join(".DS_Store"), "hidden").expect("write hidden");
        std::fs::create_dir_all(dir.join("extra")).expect("subdirectory");
        write_page(&dir.join("extra"), "p0.png", &portrait());

        let (pages, ignored) = list_page_images(&dir).expect("list");
        let names: Vec<String> = pages
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["p1.jpg", "p2.png", "p10.png"]);
        assert_eq!(
            ignored,
            IgnoredFiles {
                count: 2,
                examples: vec!["notes.txt".into(), "README".into()]
            }
        );

        let source = PageSource::directory(&dir).expect("open directory");
        let warnings = source.warnings();
        assert_eq!(warnings.len(), 1, "one warning for every ignored file");
        let warning = &warnings[0];
        assert_eq!(warning.kind, "ignored_files");
        assert_eq!(warning.fields["count"], 2);
        assert_eq!(
            warning.fields["examples"],
            serde_json::json!(["notes.txt", "README"])
        );
        assert!(warning.message.contains("notes.txt"), "{}", warning.message);
        assert!(!warning.recovery.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_directory_with_only_images_warns_nothing_and_one_without_images_is_refused() {
        let dir = temp_dir("clean");
        write_page(&dir, "a.png", &portrait());
        let source = PageSource::directory(&dir).expect("open");
        assert!(source.warnings().is_empty());
        assert_eq!(source.len(), 1);

        let empty = temp_dir("empty");
        std::fs::write(empty.join("notes.txt"), "x").expect("write txt");
        let err = PageSource::directory(&empty)
            .err()
            .expect("no page images is an error");
        assert!(matches!(err, FocrError::InputDecode(_)), "got {err:?}");
        assert!(err.to_string().contains("has no page images"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&empty);
    }

    #[test]
    fn a_pdf_page_that_will_not_rasterize_is_yielded_with_its_error_and_keeps_its_slot() {
        // Page 2 has no image XObject: the vector/text page the native path
        // refuses (bd-0a7).
        let path = build_multi_page_pdf(&[Some(jpeg_xobject()), None, Some(jpeg_xobject())]);
        let mut source = PageSource::pdf(&path).expect("open");
        assert_eq!((source.source_len(), source.len()), (3, 3));

        let items: Vec<PageItem> = source.by_ref().collect();
        assert_eq!(items.len(), 3);
        assert!(items[0].raster.is_ok() && items[2].raster.is_ok());
        let err = items[1].raster.as_ref().expect_err("page 2 is skipped");
        assert!(matches!(err, FocrError::InputDecode(_)), "got {err:?}");
        assert!(
            err.to_string().contains("PDF page 2 has no image XObject"),
            "the skip names its page and reason: {err}"
        );
        assert_eq!(
            items
                .iter()
                .map(|i| (i.index, i.source_page))
                .collect::<Vec<_>>(),
            vec![(0, 1), (1, 2), (2, 3)],
            "a skip never shifts the numbering after it"
        );
        assert_eq!(
            source.page_map(),
            &[
                MappedPage {
                    source_page: 1,
                    half: None,
                    size: Some((16, 12))
                },
                MappedPage {
                    source_page: 2,
                    half: None,
                    size: None
                },
                MappedPage {
                    source_page: 3,
                    half: None,
                    size: Some((16, 12))
                },
            ]
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_map_follows_page_selections_and_split_spreads() {
        let dir = temp_dir("spreads");
        write_page(&dir, "1.png", &portrait());
        write_page(&dir, "2.png", &spread());
        write_page(&dir, "3.png", &portrait());
        write_page(&dir, "4.png", &spread());

        // Every page, spreads split: the spread on page 2 becomes two slots
        // and page 3 is still page 3.
        let mut source = PageSource::directory(&dir)
            .expect("open")
            .split_spreads(true);
        assert_eq!(
            summary(&mut source),
            vec![
                (0, 1, None, true),
                (1, 2, Some(Half::Left), true),
                (2, 2, Some(Half::Right), true),
                (3, 3, None, true),
                (4, 4, Some(Half::Left), true),
                (5, 4, Some(Half::Right), true),
            ]
        );
        let map = source.page_map();
        let (left, right) = (map[1].size.unwrap(), map[2].size.unwrap());
        assert_eq!(left.0 + right.0, 800, "the halves tile the spread");
        assert_eq!((left.1, right.1), (500, 500));

        // A `--pages` subset keeps source numbers; unsplit spreads stay whole.
        let mut subset = PageSource::directory(&dir)
            .expect("open")
            .select(Some("4,2"))
            .expect("select");
        assert_eq!(subset.len(), 2);
        assert_eq!(
            summary(&mut subset),
            vec![(0, 2, None, true), (1, 4, None, true)]
        );
        assert_eq!(subset.page_map()[1].size, Some((800, 500)));

        let err = PageSource::directory(&dir)
            .expect("open")
            .select(Some("5"))
            .err()
            .expect("page 5 of 4");
        assert!(matches!(err, FocrError::Usage(_)), "got {err:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unreadable_image_in_a_directory_is_skipped_like_a_bad_pdf_page() {
        let dir = temp_dir("corrupt");
        write_page(&dir, "p1.png", &portrait());
        std::fs::write(dir.join("p2.png"), b"not a png").expect("write corrupt page");
        write_page(&dir, "p3.png", &portrait());
        let mut source = PageSource::directory(&dir).expect("open");
        let items: Vec<PageItem> = source.by_ref().collect();
        let err = items[1].raster.as_ref().expect_err("p2 does not decode");
        assert!(matches!(err, FocrError::InputDecode(_)), "got {err:?}");
        assert!(err.to_string().contains("p2.png"), "names the file: {err}");
        assert_eq!(items[2].source_page, 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A source that counts its rasterizations.
    struct Counting {
        pages: usize,
        renders: Arc<AtomicUsize>,
    }

    impl Rasterize for Counting {
        fn len(&self) -> usize {
            self.pages
        }

        fn render(&self, _idx: usize) -> FocrResult<DynamicImage> {
            self.renders.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(portrait())
        }
    }

    #[test]
    fn a_page_is_rasterized_only_when_the_walk_reaches_it() {
        let renders = Arc::new(AtomicUsize::new(0));
        let mut source = PageSource::new(Box::new(Counting {
            pages: 1000,
            renders: Arc::clone(&renders),
        }));
        assert_eq!(
            renders.load(AtomicOrdering::SeqCst),
            0,
            "opening reads nothing"
        );
        for taken in 1..=5 {
            let item = source.next().expect("a page");
            assert_eq!(item.source_page, taken);
            assert_eq!(
                renders.load(AtomicOrdering::SeqCst),
                taken,
                "one raster per page asked for, no lookahead"
            );
            assert!(source.held.is_none(), "an unsplit walk holds no raster");
        }
        // The map is bookkeeping, not rasters: 5 slots after 5 pages.
        assert_eq!(source.page_map().len(), 5);

        // Spreads hold exactly one raster between the halves.
        let renders = Arc::new(AtomicUsize::new(0));
        struct Spreads(Arc<AtomicUsize>);
        impl Rasterize for Spreads {
            fn len(&self) -> usize {
                3
            }
            fn render(&self, _idx: usize) -> FocrResult<DynamicImage> {
                self.0.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(spread())
            }
        }
        let mut source =
            PageSource::new(Box::new(Spreads(Arc::clone(&renders)))).split_spreads(true);
        let left = source.next().expect("left");
        assert_eq!(left.half, Some(Half::Left));
        assert!(source.held.is_some(), "the right half waits");
        assert_eq!(renders.load(AtomicOrdering::SeqCst), 1);
        let right = source.next().expect("right");
        assert_eq!((right.source_page, right.half), (1, Some(Half::Right)));
        assert!(source.held.is_none());
        assert_eq!(
            renders.load(AtomicOrdering::SeqCst),
            1,
            "the right half is not re-read"
        );
    }

    #[test]
    fn fetch_reproduces_the_walks_rasters_and_the_maps_sizes() {
        let path = build_multi_page_pdf(&[Some(jpeg_xobject()), None, Some(jpeg_xobject())]);
        let mut source = PageSource::pdf(&path).expect("open");
        let walked: Vec<PageItem> = source.by_ref().collect();
        for item in &walked {
            match &item.raster {
                Ok(raster) => {
                    let again = source.fetch(item.source_page, None).expect("re-fetch");
                    assert_eq!(again.as_bytes(), raster.as_bytes());
                    assert_eq!(
                        source.page_map()[item.index].size,
                        Some((again.width(), again.height()))
                    );
                }
                Err(_) => assert!(source.fetch(item.source_page, None).is_err()),
            }
        }
        assert!(matches!(source.fetch(0, None), Err(FocrError::Usage(_))));
        assert!(matches!(source.fetch(4, None), Err(FocrError::Usage(_))));
        let _ = std::fs::remove_file(&path);

        let dir = temp_dir("fetch_halves");
        write_page(&dir, "1.png", &spread());
        write_page(&dir, "2.png", &portrait());
        let mut source = PageSource::directory(&dir)
            .expect("open")
            .split_spreads(true);
        let walked: Vec<PageItem> = source.by_ref().collect();
        for item in &walked {
            let raster = item.raster.as_ref().expect("rasterized");
            let again = source.fetch(item.source_page, item.half).expect("re-fetch");
            assert_eq!(again.as_bytes(), raster.as_bytes(), "{item:?}");
            assert_eq!(
                source.page_map()[item.index].size,
                Some((again.width(), again.height()))
            );
        }
        let err = source
            .fetch(2, Some(Half::Left))
            .expect_err("page 2 is not a spread");
        assert!(err.to_string().contains("not a two-page spread"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_empty_image_list_is_refused() {
        assert!(matches!(
            PageSource::images(Vec::new()),
            Err(FocrError::Usage(_))
        ));
    }

    #[test]
    fn a_page_source_can_move_to_the_blocking_stage() {
        fn assert_send<T: Send + 'static>() {}
        assert_send::<PageSource>();
    }
}
