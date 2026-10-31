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
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use image::DynamicImage;

use crate::error::{FocrError, FocrResult};
use crate::native_engine::{
    Half, RecognizedDocument, RunWarning, WarningKind, timing_log, warning_recovery,
};
#[cfg(feature = "pdf")]
use crate::pdf::PdfPages;

/// The file extensions a directory source reads as page images
/// (case-insensitive).
pub const PAGE_IMAGE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "tif", "tiff", "webp", "bmp", "gif", "pnm", "pbm", "pgm", "ppm",
];

/// How many ignored file names an `ignored_files` warning names.
const IGNORED_EXAMPLES: usize = 3;

/// Resolve a `--pages` spec into 0-based page indices, in source order with
/// duplicates removed. `None` selects the whole document.
///
/// Out-of-range and malformed pages are **usage errors naming the real page
/// count**, not silently dropped: asking for page 400 of a 200-page book is a
/// mistake worth reporting, and a frontend that quietly ignores it leaves the
/// user believing pages were read that never were.
///
/// # Errors
/// [`FocrError::Usage`] for an unparseable element, page 0 (pages are 1-based),
/// a page past the end, a reversed range, or an empty element.
pub fn select_pages(spec: Option<&str>, page_count: usize) -> FocrResult<Vec<usize>> {
    let Some(spec) = spec else {
        return Ok((0..page_count).collect());
    };
    let usage = |what: &str| {
        FocrError::Usage(format!(
            "--pages {spec:?}: {what} (expected 1-based pages/ranges like \"1,5-9\"; \
             this document has {page_count} page(s))"
        ))
    };
    let parse_one = |tok: &str| -> FocrResult<usize> {
        let n: usize = tok
            .trim()
            .parse()
            .map_err(|_| usage(&format!("unparseable page {tok:?}")))?;
        if n == 0 {
            return Err(usage("page 0 (pages are 1-based)"));
        }
        if n > page_count {
            return Err(usage(&format!("page {n} is out of range")));
        }
        Ok(n - 1)
    };
    let mut selected = Vec::new();
    let mut seen = vec![false; page_count];
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err(usage("empty element"));
        }
        let range = match part.split_once('-') {
            Some((a, b)) => {
                let (a, b) = (parse_one(a)?, parse_one(b)?);
                if a > b {
                    return Err(usage(&format!("reversed range {part:?}")));
                }
                a..=b
            }
            None => {
                let n = parse_one(part)?;
                n..=n
            }
        };
        for idx in range {
            if !seen[idx] {
                seen[idx] = true;
                selected.push(idx);
            }
        }
    }
    selected.sort_unstable();
    Ok(selected)
}

/// Detect and split a two-page book spread into (left, right) halves
/// (bd-av64.11). A spread is a rasterized page that is (a) noticeably wider
/// than tall (w/h >= 1.25 — portrait book pages side by side) AND (b) has a
/// low-ink vertical gutter near the horizontal center. Returns `None` when
/// either condition fails — a false split is worse than none (full-bleed
/// landscape photos, single-column landscape pages, spreads with a plate
/// crossing the gutter all pass through unsplit).
///
/// The gutter search: over the middle 20% of columns, a column qualifies
/// as gutter when its dark fraction is EITHER <= 0.5% (a blank inter-page
/// gap — flat-scanned loose pages) OR >= 60% (the dark binding shadow of a
/// bound book pressed into the scanner — the Cadwallader case). Text
/// columns are mixed black-on-white and match neither. Among qualifying
/// columns the one closest to the exact center wins. Decision + geometry
/// are logged by the caller under FOCR_TIMING.
#[must_use]
pub fn split_spread(img: &DynamicImage) -> Option<(DynamicImage, DynamicImage, u32)> {
    let (w, h) = (img.width(), img.height());
    if h == 0 || (f64::from(w) / f64::from(h)) < 1.25 {
        return None;
    }
    let gray = img.to_luma8();
    let ink_threshold = 160u8; // scanned text is near-black; paper near-white
    let (lo, hi) = (w * 2 / 5, w * 3 / 5); // middle 20% of columns
    let center = i64::from(w / 2);
    let mut best: Option<(i64, u32)> = None; // (distance to center, column)
    for x in lo..hi {
        let mut dark = 0u32;
        for y in 0..h {
            if gray.get_pixel(x, y).0[0] < ink_threshold {
                dark += 1;
            }
        }
        let dark_frac_pct_x10 = u64::from(dark) * 1000 / u64::from(h);
        let is_gutter = dark_frac_pct_x10 <= 5 || dark_frac_pct_x10 >= 600;
        if is_gutter {
            let dist = (i64::from(x) - center).abs();
            if best.is_none_or(|(d, _)| dist < d) {
                best = Some((dist, x));
            }
        }
    }
    let (_, gutter_x) = best?;
    let left = img.crop_imm(0, 0, gutter_x, h);
    let right = img.crop_imm(gutter_x, 0, w - gutter_x, h);
    Some((left, right, gutter_x))
}

/// Whether an error ends the whole document walk or just this page.
///
/// The distinction is the load-bearing part of a document run, and it is easy to
/// get wrong in the permissive direction. A page whose codec has no pure-Rust
/// decoder is *this page's* problem — one JPEG-2000 page in a 300-page scan must
/// not throw away the other 299. But a missing model, a cancelled run, or a
/// format mismatch is the *document's* problem: skipping 300 pages one at a time
/// and reporting "300 skipped" would be a slow, confusing way to say "you never
/// had a model loaded".
///
/// [`PageSource::read_pages`] applies this set to every per-page run, so it
/// decides what `focr ocr book.pdf` does. `Timeout` is deliberately NOT fatal:
/// the stage budget is per-forward, so one slow page is a skip, not a verdict on
/// the rest of the book.
#[must_use]
pub fn is_fatal_to_document(err: &FocrError) -> bool {
    matches!(
        err,
        FocrError::ModelNotFound(_) | FocrError::Cancelled | FocrError::FormatMismatch(_)
    )
}

/// One page that was refused, and why.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SkippedPage {
    /// 1-based source page number.
    pub page: usize,
    /// The engine's own message — "JPXDecode: no pure-Rust decoder", not a
    /// paraphrase.
    pub reason: String,
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
    /// Where the walk stands past this page ([`PageSource::seek`]).
    pub walk: WalkCursor,
}

/// A position in a [`PageSource`]'s walk, which a resumed run seeks to
/// without rasterizing the pages before it (U2-D4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WalkCursor {
    /// Entries of the selection already walked.
    pub next: usize,
    /// The walk holds the right half of the spread it last read.
    pub held: bool,
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

/// A page's place in a [`PageSource::read_pages`] walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageStep {
    /// 0-based position in the run, as [`PageItem::index`].
    pub index: usize,
    /// 1-based page of the source.
    pub source_page: usize,
    /// The half of a split spread, or `None` for a whole page.
    pub half: Option<Half>,
}

/// Progress across a [`PageSource::read_pages`] walk, in source order.
#[derive(Debug, Clone, Copy)]
pub enum PageEvent<'a> {
    /// About to read this page. Its raster is already made.
    Started(PageStep),
    /// This page read.
    Done(PageStep),
    /// This page failed, for this reason; the walk continues.
    Skipped(PageStep, &'a FocrError),
}

/// One page a [`PageSource::read_pages`] walk read.
#[derive(Debug, Clone)]
pub struct PageRead<T> {
    /// 1-based page of the source.
    pub source_page: usize,
    /// The half of a split spread, or `None` for a whole page.
    pub half: Option<Half>,
    /// What the reader returned for it.
    pub value: T,
    /// Wall time for this page alone, its raster included.
    pub duration: Duration,
}

/// What a [`PageSource::read_pages`] walk read and skipped.
#[derive(Debug, Clone)]
pub struct PagesRead<T> {
    /// Pages that read, in source order.
    pub pages: Vec<PageRead<T>>,
    /// Pages that failed, in source order.
    pub skipped: Vec<SkippedPage>,
    /// Pages in the source.
    pub source_pages: usize,
}

impl<T> PagesRead<T> {
    /// Total wall time across the pages that read.
    #[must_use]
    pub fn duration(&self) -> Duration {
        self.pages.iter().map(|page| page.duration).sum()
    }
}

impl PagesRead<RecognizedDocument> {
    /// The whole document as one markdown string, pages joined by a blank
    /// line: the shape `focr ocr book.pdf` writes.
    #[must_use]
    pub fn markdown(&self) -> String {
        self.pages
            .iter()
            .map(|page| page.value.markdown.trim_end())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// The pages as the model wrote them, joined as [`Self::markdown`] joins
    /// them; each page's boxes are on that page's own 0–999 grid.
    #[must_use]
    pub fn raw(&self) -> String {
        self.pages
            .iter()
            .map(|page| page.value.raw.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// What a source rasterizes its pages from.
trait Rasterize: Send {
    /// Pages in the source.
    fn len(&self) -> usize;
    /// Rasterize 0-based page `idx`.
    fn render(&self, idx: usize) -> FocrResult<DynamicImage>;
}

#[cfg(feature = "pdf")]
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

/// Pages already in memory (an app's camera captures, a test's pages): each
/// render hands out a copy, so a later [`PageSource::fetch`] still has it.
struct Rasters(Vec<DynamicImage>);

impl Rasterize for Rasters {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn render(&self, idx: usize) -> FocrResult<DynamicImage> {
        self.0.get(idx).cloned().ok_or_else(|| {
            FocrError::InputDecode(format!(
                "page index {idx} out of range ({} pages)",
                self.0.len()
            ))
        })
    }
}

/// A document's pages, rasterized one at a time as the walk asks for them.
///
/// Build one with [`PageSource::open`], [`PageSource::pdf`],
/// [`PageSource::images`], [`PageSource::directory`] or
/// [`PageSource::rasters`], narrow it with
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
    /// As [`PageSource::directory`] or `PageSource::pdf`; without the `pdf`
    /// feature, [`FocrError::NotImplemented`] for anything but a directory.
    pub fn open(path: &Path) -> FocrResult<Self> {
        if path.is_dir() {
            return Self::directory(path);
        }
        #[cfg(feature = "pdf")]
        return Self::pdf(path);
        #[cfg(not(feature = "pdf"))]
        Err(FocrError::NotImplemented(format!(
            "{}: this build has no PDF reader (the `pdf` feature)",
            path.display()
        )))
    }

    /// The pages of the PDF at `path`.
    ///
    /// # Errors
    /// As [`PdfPages::open`].
    #[cfg(feature = "pdf")]
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

    /// Pages already decoded, in the order given. A raster is used as it is:
    /// no EXIF orientation is applied, since a decoded image has none.
    ///
    /// # Errors
    /// [`FocrError::Usage`] for an empty list.
    pub fn rasters(pages: Vec<DynamicImage>) -> FocrResult<Self> {
        if pages.is_empty() {
            return Err(FocrError::Usage(
                "a page source needs at least one page".into(),
            ));
        }
        Ok(Self::new(Box::new(Rasters(pages))))
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

    /// Walk only the pages of a `--pages` spec ([`select_pages`]);
    /// `None` keeps every page. Call it before the walk.
    ///
    /// # Errors
    /// [`FocrError::Usage`] for a malformed or out-of-range spec.
    pub fn select(mut self, spec: Option<&str>) -> FocrResult<Self> {
        self.selected = select_pages(spec, self.pages.len())?;
        Ok(self)
    }

    /// Yield a two-page spread as its left and right halves
    /// ([`split_spread`]); a page without a detectable gutter stays
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

    /// Pages the walk has yet to yield: the source pages it has not reached
    /// (a spread counted once) and a held right half.
    #[must_use]
    pub fn unread(&self) -> usize {
        self.selected.len() - self.next + usize::from(self.held.is_some())
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
        let (left, right, _) = split_spread(&image).ok_or_else(|| {
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

    /// Continue the walk from `cursor`, with `map` as the pages it has
    /// yielded: the pages before it are not rasterized. A held right half
    /// is, since the walk yields it next.
    ///
    /// # Errors
    /// [`FocrError::Usage`] for a cursor past the selection, or the held
    /// page's raster error.
    pub fn seek(&mut self, cursor: WalkCursor, map: Vec<MappedPage>) -> FocrResult<()> {
        if cursor.next > self.selected.len() || (cursor.held && cursor.next == 0) {
            return Err(FocrError::Usage(format!(
                "a resumed walk at entry {} is outside this selection of {} pages",
                cursor.next,
                self.selected.len()
            )));
        }
        self.held = if cursor.held {
            let source_page = self.selected[cursor.next - 1] + 1;
            Some((source_page, self.fetch(source_page, Some(Half::Right))?))
        } else {
            None
        };
        self.next = cursor.next;
        self.map = map;
        Ok(())
    }

    /// Read the rest of the walk with `read`, one page at a time.
    ///
    /// A page that fails is skipped and reported, so one bad page cannot
    /// cost the rest of a book. A failure of the whole run
    /// ([`is_fatal_to_document`]) ends the walk on the page that met
    /// it, so a missing model is not 300 skipped pages.
    ///
    /// The walk is sequential by construction, and that is a correctness
    /// property: the engine admits one forward at a time and each forward
    /// already fans out across every core, so concurrent pages would not be
    /// faster and would multiply peak memory by the number in flight.
    ///
    /// # Errors
    /// A whole-run failure, or the first page's error when no page read, so
    /// "nothing worked" still says why.
    pub fn read_pages<T>(
        &mut self,
        mut read: impl FnMut(DynamicImage) -> FocrResult<T>,
        observer: &mut dyn FnMut(PageEvent<'_>),
    ) -> FocrResult<PagesRead<T>> {
        let mut pages = Vec::new();
        let mut skipped = Vec::new();
        let mut first_error = None;
        loop {
            let started = Instant::now();
            let Some(item) = self.next() else { break };
            let step = PageStep {
                index: item.index,
                source_page: item.source_page,
                half: item.half,
            };
            observer(PageEvent::Started(step));
            match item.raster.and_then(&mut read) {
                Ok(value) => {
                    pages.push(PageRead {
                        source_page: step.source_page,
                        half: step.half,
                        value,
                        duration: started.elapsed(),
                    });
                    observer(PageEvent::Done(step));
                }
                Err(error) if is_fatal_to_document(&error) => return Err(error),
                Err(error) => {
                    observer(PageEvent::Skipped(step, &error));
                    skipped.push(SkippedPage {
                        page: step.source_page,
                        reason: error.to_string(),
                    });
                    first_error.get_or_insert(error);
                }
            }
        }
        if pages.is_empty() {
            return Err(first_error.unwrap_or_else(|| {
                FocrError::InputDecode("the page source has no pages left to read".to_owned())
            }));
        }
        Ok(PagesRead {
            pages,
            skipped,
            source_pages: self.source_len(),
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
            walk: WalkCursor {
                next: self.next,
                held: self.held.is_some(),
            },
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
            if let Some((left, right, gutter_x)) = split_spread(&image) {
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

/// The SHA-256 a resume key names `input` by (U2-D4): a file's bytes, or for
/// a directory one line per page image, `name size sha256`, in natural order,
/// so renaming, reordering or editing a page changes it.
///
/// # Errors
/// [`FocrError::InputDecode`] when `input` cannot be read.
pub fn input_sha256(input: &Path) -> FocrResult<String> {
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;
    use std::io::Read as _;
    let file_digest = |path: &Path| {
        let unreadable =
            |e: std::io::Error| FocrError::InputDecode(format!("read {}: {e}", path.display()));
        let mut file = std::fs::File::open(path).map_err(unreadable)?;
        let (mut hasher, mut size) = (Sha256::new(), 0u64);
        let mut buffer = vec![0u8; 1 << 20];
        loop {
            let read = file.read(&mut buffer).map_err(unreadable)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            size += read as u64;
        }
        let digest = hasher.finalize().iter().fold(String::new(), |mut hex, b| {
            let _ = write!(hex, "{b:02x}");
            hex
        });
        Ok::<_, FocrError>((digest, size))
    };
    if !input.is_dir() {
        return Ok(file_digest(input)?.0);
    }
    let (pages, _) = list_page_images(input)?;
    let mut manifest = String::new();
    for page in &pages {
        let (digest, size) = file_digest(page)?;
        let name = page.file_name().map(|n| n.to_string_lossy());
        let _ = writeln!(manifest, "{} {size} {digest}", name.unwrap_or_default());
    }
    Ok(Sha256::digest(manifest.as_bytes())
        .iter()
        .fold(String::new(), |mut hex, b| {
            let _ = write!(hex, "{b:02x}");
            hex
        }))
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
pub(crate) mod tests {
    use super::*;
    #[cfg(feature = "pdf")]
    use crate::pdf::tests::{build_multi_page_pdf, jpeg_xobject};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    // ── page selection ─────────────────────────────────────────────────────

    #[test]
    fn no_spec_selects_every_page() {
        assert_eq!(select_pages(None, 4).expect("ok"), vec![0, 1, 2, 3]);
    }

    #[test]
    fn spec_is_1_based_sorted_and_deduped() {
        assert_eq!(select_pages(Some("3,1"), 5).expect("ok"), vec![0, 2]);
        assert_eq!(select_pages(Some("2-4"), 5).expect("ok"), vec![1, 2, 3]);
        // Overlapping ranges collapse rather than repeating a page.
        assert_eq!(
            select_pages(Some("1-3,2-4"), 5).expect("ok"),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn out_of_range_is_a_usage_error_naming_the_page_count() {
        // Silently dropping this is the bug: the user would believe page 9 was
        // read. The message has to name the real count.
        let err = select_pages(Some("9"), 3).expect_err("must reject");
        assert!(matches!(err, FocrError::Usage(_)), "got {err:?}");
        let text = err.to_string();
        assert!(text.contains("out of range"), "{text}");
        assert!(text.contains("3 page(s)"), "{text}");
    }

    #[test]
    fn zero_reversed_and_empty_elements_are_rejected() {
        for bad in ["0", "3-1", "1,,2", "x"] {
            let err = select_pages(Some(bad), 5).expect_err(bad);
            assert!(matches!(err, FocrError::Usage(_)), "{bad}: {err:?}");
        }
    }

    // ── fatal vs skippable ─────────────────────────────────────────────────

    #[test]
    fn only_document_level_failures_are_fatal() {
        // A page the renderer cannot decode is this page's problem.
        assert!(!is_fatal_to_document(&FocrError::InputDecode(
            "JPXDecode: no pure-Rust decoder".into()
        )));
        assert!(!is_fatal_to_document(&FocrError::NotImplemented(
            "x".into()
        )));
        // The stage budget is PER FORWARD, so one slow page is a skip, not a
        // verdict on the rest of the book. This matches what cli.rs has always
        // done; changing it would silently change `focr ocr book.pdf`.
        assert!(!is_fatal_to_document(&FocrError::Timeout("x".into())));
        // These say the run itself is over; skipping 300 pages one at a time
        // would be a slow way to report "you never had a model".
        assert!(is_fatal_to_document(&FocrError::ModelNotFound("x".into())));
        assert!(is_fatal_to_document(&FocrError::Cancelled));
        assert!(is_fatal_to_document(&FocrError::FormatMismatch("x".into())));
    }

    // ── spreads ────────────────────────────────────────────────────────────

    pub(crate) fn synth_page(w: u32, h: u32, text_cols: &[(u32, u32)]) -> DynamicImage {
        // White canvas with black "text block" columns [x0, x1).
        let mut img = image::GrayImage::from_pixel(w, h, image::Luma([255u8]));
        for &(x0, x1) in text_cols {
            for x in x0..x1 {
                for y in (10..h.saturating_sub(10)).step_by(3) {
                    img.put_pixel(x, y, image::Luma([20u8]));
                }
            }
        }
        DynamicImage::ImageLuma8(img)
    }

    /// bd-av64.11: a synthetic spread (text left + right, blank center
    /// gutter) splits at the gutter; the negatives pass through unsplit.
    #[test]
    fn split_spread_positive_and_negatives() {
        // POSITIVE: 1600x1000 spread, text at [100,700) and [900,1500).
        let spread = synth_page(1600, 1000, &[(100, 700), (900, 1500)]);
        let (left, right, gx) = split_spread(&spread).expect("spread splits");
        assert!((700..=900).contains(&gx), "gutter near center: {gx}");
        assert_eq!(left.width() + right.width(), 1600);
        assert_eq!(left.height(), 1000);
        // NEGATIVE 1: portrait page (aspect below the spread threshold).
        assert!(split_spread(&synth_page(1000, 1600, &[(100, 900)])).is_none());
        // NEGATIVE 2: landscape but ink crosses the center (a full-width
        // plate/table) — no blank gutter, no split.
        assert!(split_spread(&synth_page(1600, 1000, &[(100, 1500)])).is_none());
        // NEGATIVE 3: landscape blank page — a gutter exists but splitting a
        // blank is harmless; the heuristic DOES split it (blank center
        // qualifies). Accepting this is deliberate: both halves are blank,
        // and OCR of blank halves is cheap + correct.
        // NEGATIVE 4: single centered column (text crosses the middle).
        assert!(split_spread(&synth_page(1600, 1000, &[(600, 1000)])).is_none());
        // POSITIVE 2: a bound book's DARK binding shadow as the gutter (the
        // Cadwallader case) — a solid dark band at the center qualifies.
        let mut bound = synth_page(1600, 1000, &[(100, 700), (900, 1500)]).to_luma8();
        for x in 780..820 {
            for y in 0..1000 {
                bound.put_pixel(x, y, image::Luma([30u8]));
            }
        }
        let bound = DynamicImage::ImageLuma8(bound);
        let (_, _, gx) = split_spread(&bound).expect("binding shadow splits");
        assert!((780..=820).contains(&gx), "split inside the shadow: {gx}");
    }

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
    fn rasters_walk_in_order_split_like_any_page_and_fetch_again() {
        assert!(matches!(
            PageSource::rasters(Vec::new()),
            Err(FocrError::Usage(_))
        ));
        let pages = vec![portrait(), spread()];
        let mut source = PageSource::rasters(pages.clone())
            .expect("rasters")
            .split_spreads(true);
        assert_eq!((source.source_len(), source.len()), (2, 2));
        assert_eq!(
            summary(&mut source),
            vec![
                (0, 1, None, true),
                (1, 2, Some(Half::Left), true),
                (2, 2, Some(Half::Right), true),
            ]
        );
        assert_eq!(source.fetch(1, None).expect("page 1"), pages[0]);
        let (left, _, _) = split_spread(&pages[1]).expect("a spread");
        assert_eq!(source.fetch(2, Some(Half::Left)).expect("left"), left);
        assert!(source.warnings().is_empty());
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
    #[cfg(feature = "pdf")]
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
    fn the_input_digest_follows_the_bytes_and_a_directorys_names() {
        let dir = temp_dir("input-digest");
        let page = write_page(&dir, "p1.png", &portrait());
        let directory = input_sha256(&dir).expect("a directory");
        let file = input_sha256(&page).expect("a file");
        assert_eq!(file.len(), 64);
        assert_ne!(directory, file);
        assert_eq!(input_sha256(&dir).expect("again"), directory);
        // Text beside the pages is no page.
        std::fs::write(dir.join("notes.txt"), b"x").expect("notes");
        assert_eq!(input_sha256(&dir).expect("notes"), directory);
        std::fs::rename(&page, dir.join("p2.png")).expect("rename");
        assert_ne!(input_sha256(&dir).expect("renamed"), directory);
        assert!(matches!(
            input_sha256(&dir.join("absent.pdf")),
            Err(FocrError::InputDecode(_))
        ));
    }

    #[test]
    fn a_seek_continues_the_walk_and_rasterizes_only_a_held_half() {
        struct Spreads(Arc<AtomicUsize>);
        impl Rasterize for Spreads {
            fn len(&self) -> usize {
                5
            }
            fn render(&self, _idx: usize) -> FocrResult<DynamicImage> {
                self.0.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(spread())
            }
        }
        // Past a whole page: nothing is read until the walk reaches page 3.
        let renders = Arc::new(AtomicUsize::new(0));
        let mut source = PageSource::new(Box::new(Counting {
            pages: 5,
            renders: Arc::clone(&renders),
        }));
        let walked = vec![
            MappedPage {
                source_page: 1,
                half: None,
                size: Some((64, 96)),
            };
            2
        ];
        let cursor = WalkCursor {
            next: 2,
            held: false,
        };
        source.seek(cursor, walked).expect("a position in the walk");
        assert_eq!(renders.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(source.unread(), 3);
        let item = source.next().expect("page 3");
        assert_eq!((item.index, item.source_page), (2, 3));
        assert_eq!(
            item.walk,
            WalkCursor {
                next: 3,
                held: false
            }
        );
        assert_eq!(renders.load(AtomicOrdering::SeqCst), 1);

        // Past a left half: the right half is read again and yielded next,
        // and the cursor the walk gave out is the one a seek takes.
        let renders = Arc::new(AtomicUsize::new(0));
        let mut walk = PageSource::new(Box::new(Spreads(Arc::clone(&renders)))).split_spreads(true);
        let left = walk.next().expect("left");
        assert!(left.walk.held);
        let map = walk.page_map().to_vec();
        let mut source =
            PageSource::new(Box::new(Spreads(Arc::clone(&renders)))).split_spreads(true);
        source.seek(left.walk, map).expect("a held half");
        assert_eq!(source.unread(), walk.unread());
        let right = source.next().expect("right");
        assert_eq!(
            (right.index, right.source_page, right.half),
            (1, 1, Some(Half::Right))
        );

        // A cursor outside the selection is refused.
        let too_far = WalkCursor {
            next: 6,
            held: false,
        };
        assert!(matches!(
            source.seek(too_far, Vec::new()),
            Err(FocrError::Usage(_))
        ));
    }

    #[test]
    #[cfg(feature = "pdf")]
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

    fn recognized(markdown: &str) -> RecognizedDocument {
        RecognizedDocument {
            markdown: markdown.to_owned(),
            raw: markdown.to_owned(),
            layout: Vec::new(),
        }
    }

    #[test]
    #[cfg(feature = "pdf")]
    fn a_page_walk_reads_every_readable_page_and_skips_the_rest_with_reasons() {
        // Page 2 has no image: the renderer refuses it, and the walk must carry
        // on, because one bad page in a book cannot cost the other pages.
        let path = build_multi_page_pdf(&[Some(jpeg_xobject()), None, Some(jpeg_xobject())]);
        let mut source = PageSource::pdf(&path).expect("open");

        let mut events: Vec<String> = Vec::new();
        let mut read_count = 0;
        let outcome = source
            .read_pages(
                |image| {
                    assert!(image.width() > 0, "the page rasterized");
                    read_count += 1;
                    Ok(recognized(&format!("text {read_count}\n\n")))
                },
                &mut |event| {
                    events.push(match event {
                        PageEvent::Started(page) => format!("start {}", page.source_page),
                        PageEvent::Done(page) => format!("done {}", page.source_page),
                        PageEvent::Skipped(page, _) => format!("skip {}", page.source_page),
                    });
                },
            )
            .expect("the walk succeeds when at least one page reads");

        assert_eq!(outcome.source_pages, 3);
        let pages: Vec<usize> = outcome.pages.iter().map(|p| p.source_page).collect();
        assert_eq!(pages, vec![1, 3]);
        assert_eq!(outcome.skipped.len(), 1);
        assert_eq!(outcome.skipped[0].page, 2);
        assert!(
            !outcome.skipped[0].reason.is_empty(),
            "a skip carries the engine's reason"
        );
        assert_eq!(outcome.markdown(), "text 1\n\ntext 2");
        assert_eq!(
            events,
            [
                "start 1", "done 1", "start 2", "skip 2", "start 3", "done 3"
            ]
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn pages_read_joins_with_a_blank_line_and_sums_only_what_read() {
        let page = |source_page, markdown, millis| PageRead {
            source_page,
            half: None,
            value: recognized(markdown),
            duration: Duration::from_millis(millis),
        };
        let outcome = PagesRead {
            pages: vec![page(1, "one\n\n", 10), page(2, "two", 20)],
            skipped: vec![SkippedPage {
                page: 3,
                reason: "JBIG2Decode".into(),
            }],
            source_pages: 3,
        };
        assert_eq!(outcome.markdown(), "one\n\ntwo");
        assert_eq!(outcome.duration(), Duration::from_millis(30));
    }

    #[test]
    fn a_page_walk_ends_at_once_on_a_failure_of_the_whole_run() {
        for fatal in [
            FocrError::ModelNotFound("no model".into()),
            FocrError::Cancelled,
            FocrError::FormatMismatch("not a model".into()),
        ] {
            let kind = fatal.to_string();
            let mut fatal = Some(fatal);
            let pages = vec![portrait(), portrait(), portrait()];
            let mut source = PageSource::rasters(pages).expect("a source");
            let mut attempts = 0usize;
            let err = source
                .read_pages(
                    |_| -> FocrResult<RecognizedDocument> {
                        attempts += 1;
                        Err(fatal.take().expect("read once"))
                    },
                    &mut |_| {},
                )
                .expect_err("the run ends");
            assert_eq!(err.to_string(), kind);
            assert_eq!(attempts, 1, "no grinding through the rest of the book");
        }
    }

    #[test]
    #[cfg(feature = "pdf")]
    fn a_page_walk_with_no_readable_page_returns_the_first_error_unchanged() {
        let path = build_multi_page_pdf(&[None, None]);
        let mut source = PageSource::pdf(&path).expect("open");
        let err = source
            .read_pages(|_| Ok(recognized("")), &mut |_| {})
            .expect_err("nothing readable");
        assert!(matches!(err, FocrError::InputDecode(_)), "got {err:?}");
        assert!(!err.to_string().is_empty(), "it still says why");
        let _ = std::fs::remove_file(&path);

        // A reader's own error keeps its kind, and so the run's exit code.
        let mut source = PageSource::rasters(vec![portrait(), portrait()]).expect("a source");
        let err = source
            .read_pages(
                |_| -> FocrResult<RecognizedDocument> {
                    Err(FocrError::Timeout("the first page".into()))
                },
                &mut |_| {},
            )
            .expect_err("nothing read");
        assert!(matches!(err, FocrError::Timeout(_)), "got {err:?}");
    }

    #[test]
    fn a_page_walk_reports_each_half_of_a_spread() {
        let mut source = PageSource::rasters(vec![spread(), portrait()])
            .expect("a source")
            .split_spreads(true);
        let mut started = Vec::new();
        let outcome = source
            .read_pages(|_| Ok(()), &mut |event| {
                if let PageEvent::Started(page) = event {
                    started.push((page.index, page.source_page, page.half));
                }
            })
            .expect("every page reads");
        assert_eq!(
            started,
            [
                (0, 1, Some(Half::Left)),
                (1, 1, Some(Half::Right)),
                (2, 2, None)
            ]
        );
        assert_eq!(outcome.pages[1].half, Some(Half::Right));
        assert_eq!(outcome.source_pages, 2);
    }

    #[test]
    fn a_page_source_can_move_to_the_blocking_stage() {
        fn assert_send<T: Send + 'static>() {}
        assert_send::<PageSource>();
    }
}
