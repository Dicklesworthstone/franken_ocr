//! Image ingest front end ([SPEC-018, SPEC-020..033],
//! PROPOSED_ARCHITECTURE.md §6.2). The `infer` data pipeline, built fresh (this
//! is a frankentorch gap).
//!
//! This module turns a document image (a file path or raw bytes) into the exact
//! pixel tensors the vision tower consumes, plus the tile geometry the connector
//! needs to build the image-placeholder layout. It implements the two model
//! modes from the pinned source:
//!
//! * **Base** (`crop_mode=false`, [SPEC-029]) — a single global 1024-view:
//!   aspect-preserving resize + gray pad to `base_size × base_size`, one
//!   273-slot placeholder block.
//! * **Gundam** dynamic tiling (`crop_mode=true`, [SPEC-023..028], OQ-7) —
//!   `find_closest_aspect_ratio` over `min_num..=max_num` candidate grids, a
//!   `W×H` grid of `image_size×image_size` local tiles, PLUS the global
//!   1024-view as a thumbnail (the global view always exists; the local tiles
//!   are only emitted when the chosen grid is larger than 1×1).
//!
//! The per-tile pixel tensor is laid out exactly the way [`vision_sam::forward`]
//! reads it: a row-major `[3, H, W]` [`Mat`] (`rows = 3` channels, `cols = H*W`),
//! RGB f32, normalized by `ToTensor` -> `Normalize(0.5, 0.5)` => `[-1, 1]`
//! ([SPEC-021]). Mean/std/patch_size/downsample_ratio are the pinned
//! `processor_config` values ([SPEC-018]).
//!
//! **L0 parity = exact.** Where a Rust image-op cannot be bit-identical to PIL
//! (the resampling kernel), the divergence is named in a comment and routed to
//! the closest available filter; the *geometry* (tile counts, sizes, placeholder
//! census) is exact. The one non-geometric divergence — CatmullRom in place of
//! PIL BICUBIC — is ledgered as DISC-001 (`docs/DISCREPANCIES.md`, bd-30me) and
//! carries a kill-switch: `FOCR_RESAMPLE=pil-bicubic` swaps every resize site
//! onto [`pil_resample`], a Pillow-bit-exact fixed-point BICUBIC, for L0 EXACT
//! oracle comparison. The default stays CatmullRom (byte-identical to the
//! pre-DISC-001 pipeline, doctrine #2).

pub mod pil_resample;
pub mod staff_detect;

use std::path::Path;

use image::{DynamicImage, GenericImageView, ImageDecoder, ImageReader, imageops::FilterType};

use crate::error::{FocrError, FocrResult};

// ── pinned processor / model constants ([SPEC-018], OQ-7, OQ-18) ────────────

/// Per-channel normalization mean ([SPEC-018], `processor_config.json:11`).
pub const IMAGE_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
/// Per-channel normalization std ([SPEC-018], `processor_config.json:15`).
pub const IMAGE_STD: [f32; 3] = [0.5, 0.5, 0.5];
/// SAM/CLIP patch size ([SPEC-018/027], `processor_config.json:25`).
pub const PATCH_SIZE: usize = 16;
/// Token-compression downsample ratio ([SPEC-018/027],
/// `processor_config.json:9`).
pub const DOWNSAMPLE_RATIO: usize = 4;

/// Default base (global) view size in pixels ([SPEC-027], OQ-18,
/// `modeling_unlimitedocr.py:787` `base_size=1024`).
pub const BASE_SIZE: usize = 1024;
/// Gundam local-tile size in pixels ([SPEC-024], OQ-7,
/// `dynamic_preprocess(image_size=640)`).
pub const GUNDAM_TILE_SIZE: usize = 640;

/// Gundam `dynamic_preprocess` minimum tile count ([SPEC-024], OQ-7).
pub const MIN_NUM: usize = 2;
/// Gundam `dynamic_preprocess` maximum tile count ([SPEC-024], OQ-7).
pub const MAX_NUM: usize = 32;

/// The crop short-circuit threshold: an image `<= 640` in BOTH dims gets
/// `crop_ratio=[1,1]` (no local tiling) ([SPEC-023],
/// `modeling_unlimitedocr.py:859`).
pub const CROP_THRESHOLD: u32 = 640;

/// Gray pad fill value, `int(0.5 * 255) = 127` per channel ([SPEC-022],
/// `modeling_unlimitedocr.py:872`).
pub const PAD_FILL: u8 = 127;

// ── modes ───────────────────────────────────────────────────────────────────

/// The preprocessing mode the caller selects (the `crop_mode` flag plus its
/// associated `base_size`/`image_size`, OQ-18 census table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreprocessMode {
    /// Single global view at `base_size` ([SPEC-029], `crop_mode=false`). The
    /// README "base" single-image config (`base_size=1024, image_size=1024`).
    Base {
        /// Global-view edge length in pixels (1024 in the pinned config).
        base_size: usize,
    },
    /// Gundam dynamic tiling ([SPEC-023..028], `crop_mode=true`). A global
    /// `base_size` view plus a `W×H` grid of `tile_size` local tiles chosen by
    /// `find_closest_aspect_ratio`.
    Gundam {
        /// Global-view edge length in pixels (1024).
        base_size: usize,
        /// Local-tile edge length in pixels (640).
        tile_size: usize,
    },
}

impl Default for PreprocessMode {
    /// The pinned README "base" config: a single 1024-pixel global view
    /// ([SPEC-029]); the 273-slot census is defined at this base size.
    fn default() -> Self {
        Self::Base { base_size: 1024 }
    }
}

impl PreprocessMode {
    /// The pinned base mode: a single 1024 global view ([SPEC-029], OQ-18).
    #[must_use]
    pub fn base() -> Self {
        PreprocessMode::Base {
            base_size: BASE_SIZE,
        }
    }

    /// The pinned Gundam mode: 1024 global + 640 dynamic tiles ([SPEC-024],
    /// OQ-7).
    #[must_use]
    pub fn gundam() -> Self {
        PreprocessMode::Gundam {
            base_size: BASE_SIZE,
            tile_size: GUNDAM_TILE_SIZE,
        }
    }

    /// The global-view edge length for this mode.
    #[must_use]
    pub fn base_size(self) -> usize {
        match self {
            PreprocessMode::Base { base_size } | PreprocessMode::Gundam { base_size, .. } => {
                base_size
            }
        }
    }
}

/// `num_queries = ceil((size // patch_size) / downsample_ratio)` ([SPEC-027],
/// OQ-18, `modeling_unlimitedocr.py:904-905`). For `size=1024` this is 16; for
/// `size=640` it is 10.
#[must_use]
pub fn num_queries(size: usize) -> usize {
    (size / PATCH_SIZE).div_ceil(DOWNSAMPLE_RATIO)
}

// ── output bundle ─────────────────────────────────────────────────────────

/// A single pixel tensor for one vision view (global thumbnail or local tile).
///
/// Laid out exactly the way `vision_sam::forward` reads it ([SPEC-041]): a
/// row-major `[3, H, W]` matrix — `pixels.rows == 3` (RGB channels), `pixels.cols
/// == H*W`, element `(c, y*W + x)` is channel `c` of pixel `(x, y)`. Values are
/// normalized to `[-1, 1]` ([SPEC-021]).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ViewTensor {
    /// `[3, H*W]` normalized RGB pixels (channel-major, the SAM patch-embed
    /// NCHW layout for `batch=1`).
    pub pixels: crate::native_engine::tensor::Mat,
    /// Pixel height of this view.
    pub height: usize,
    /// Pixel width of this view.
    pub width: usize,
}

impl ViewTensor {
    /// `(height, width)` of this view in pixels.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.height, self.width)
    }
}

/// The spatial tile grid chosen for an image ([SPEC-026], `crop_ratio =
/// (width_crop_num, height_crop_num)`).
///
/// For base mode (and the Gundam no-crop short-circuit) this is `(1, 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CropGrid {
    /// Number of tiles across (columns) — `width_crop_num`.
    pub width_crop_num: usize,
    /// Number of tiles down (rows) — `height_crop_num`.
    pub height_crop_num: usize,
}

impl CropGrid {
    /// The 1×1 (no-crop) grid.
    #[must_use]
    pub fn single() -> Self {
        CropGrid {
            width_crop_num: 1,
            height_crop_num: 1,
        }
    }

    /// Total local tile count `width_crop_num * height_crop_num`.
    #[must_use]
    pub fn blocks(self) -> usize {
        self.width_crop_num
            .checked_mul(self.height_crop_num)
            .expect("CropGrid::blocks: width_crop_num*height_crop_num overflow")
    }

    /// Whether local tiles are emitted (grid larger than 1×1, [SPEC-026]).
    #[must_use]
    pub fn is_tiled(self) -> bool {
        self.width_crop_num > 1 || self.height_crop_num > 1
    }
}

/// The preprocessed image bundle handed to the vision tower + connector.
///
/// Carries the per-view pixel tensors (`global` thumbnail + the Gundam local
/// `tiles`) and the geometry the connector needs to build the image-placeholder
/// id-stream and the 273-slot layout ([SPEC-028], OQ-18).
///
/// The vision tower runs `vision_sam::forward` / `vision_clip::forward` over
/// `global.pixels` and every `tiles[i].pixels`; the connector then uses
/// `crop_grid` + [`Self::placeholder_token_count`] to assemble
/// `image_newline`/`view_seperator` and the `images_seq_mask`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Preprocessed {
    /// The mode this bundle was produced under.
    pub mode: PreprocessMode,
    /// The global (thumbnail / base) view — always present ([SPEC-022/031],
    /// `images_ori`).
    pub global: ViewTensor,
    /// The Gundam local tiles in row-major order ([SPEC-024], `images_crop`).
    /// Empty in base mode and in the Gundam no-crop short-circuit.
    pub tiles: Vec<ViewTensor>,
    /// The chosen spatial crop grid ([SPEC-026], `images_spatial_crop`).
    pub crop_grid: CropGrid,
    /// Original (post-EXIF) image size in pixels, `(width, height)`. Carried for
    /// bbox de-normalization in postprocess ([SPEC-113]).
    pub original_size: (u32, u32),
}

impl Preprocessed {
    /// Number of vision views (1 global + N local tiles).
    #[must_use]
    pub fn num_views(&self) -> usize {
        1 + self.tiles.len()
    }

    /// The image-placeholder token count for the connector's `images_seq_mask`
    /// ([SPEC-028], OQ-18 census).
    ///
    /// Global block = `(q_base + 1) * q_base + 1` (= 273 at `base_size=1024`,
    /// `q_base=16`). If the grid is tiled, add the local block
    /// `(q_local*W + 1) * (q_local*H)` (`q_local=10` at `tile_size=640`).
    #[must_use]
    pub fn placeholder_token_count(&self) -> usize {
        let q_base = num_queries(self.mode.base_size());
        // Global: 16 rows of (16 patches + 1 newline) + 1 view separator.
        let mut total = q_base
            .checked_add(1)
            .and_then(|cols| cols.checked_mul(q_base))
            .and_then(|tokens| tokens.checked_add(1))
            .expect("Preprocessed::placeholder_token_count: global placeholder count overflow");
        if let PreprocessMode::Gundam { tile_size, .. } = self.mode
            && self.crop_grid.is_tiled()
        {
            let q_local = num_queries(tile_size);
            let w = self.crop_grid.width_crop_num;
            let h = self.crop_grid.height_crop_num;
            // Local: (q_local*W patches + 1 newline) per (q_local*H) rows.
            let local_cols = q_local
                .checked_mul(w)
                .and_then(|cols| cols.checked_add(1))
                .expect("Preprocessed::placeholder_token_count: local column count overflow");
            let local_rows = q_local
                .checked_mul(h)
                .expect("Preprocessed::placeholder_token_count: local row count overflow");
            let local = local_cols
                .checked_mul(local_rows)
                .expect("Preprocessed::placeholder_token_count: local placeholder count overflow");
            total = total
                .checked_add(local)
                .expect("Preprocessed::placeholder_token_count: total placeholder count overflow");
        }
        total
    }
}

// ── public entrypoints ───────────────────────────────────────────────────

/// Decode + normalize + tile a document image at `path` for `mode`.
///
/// Decodes the image (EXIF-transposed, RGB), builds the global view (and, in
/// Gundam mode, the local tile grid), normalizes every view to `[-1, 1]`, and
/// returns the [`Preprocessed`] bundle ([SPEC-020..031]).
///
/// # Errors
/// [`FocrError::InputDecode`] if the file cannot be opened/decoded.
pub fn preprocess_image(path: &Path, mode: PreprocessMode) -> FocrResult<Preprocessed> {
    let img = decode_path(path)?;
    preprocess_dynamic(img, mode)
}

/// Decode + normalize + tile a document image from raw `bytes` for `mode`.
///
/// # Errors
/// [`FocrError::InputDecode`] if the bytes cannot be decoded.
pub fn preprocess_bytes(bytes: &[u8], mode: PreprocessMode) -> FocrResult<Preprocessed> {
    let img = decode_bytes(bytes)?;
    preprocess_dynamic(img, mode)
}

/// Core pipeline over an already-decoded (EXIF-applied, any color) image.
///
/// Split out from the I/O so the tiling/normalization math is unit-testable
/// without touching the filesystem ([SPEC-022..031]), and so an in-memory image
/// (e.g. a PDF page rasterized by [`crate::pdf`]) can enter the pipeline without
/// a temp file, identically to a decoded file.
pub fn preprocess_dynamic(img: DynamicImage, mode: PreprocessMode) -> FocrResult<Preprocessed> {
    let original_size = img.dimensions();
    let validated = validate_mode(mode)?;

    // Global view: aspect-preserving resize + gray pad to base_size² ([SPEC-022]).
    let global_img = pad_to_square(&img, validated.base_size);
    let global = view_tensor(&global_img);

    let (tiles, crop_grid) = match mode {
        PreprocessMode::Base { .. } => (Vec::new(), CropGrid::single()),
        PreprocessMode::Gundam { .. } => build_gundam_tiles(&img, validated.tile_size)?,
    };

    Ok(Preprocessed {
        mode,
        global,
        tiles,
        crop_grid,
        original_size,
    })
}

/// Multi-page per-page preprocess (bd-1gv.25/bd-1gv.26): the reference
/// `infer_multi` SQUASHES each page to `size × size` when `image_size <= 640`
/// (`image.resize((image_size, image_size))` — aspect-DESTROYING, unlike the
/// single-image Base mode's aspect-preserving `ImageOps.pad`), then the
/// already-square view passes the pad untouched. The multi-page oracle run
/// caught this divergence: padding instead of squashing garbles the glyph
/// geometry the model was trained to read in multi-page mode.
///
/// Same normalize as Base mode (the 0.5/0.5 transform in [`view_tensor`]);
/// the resample kernel follows [`resample_kind`] (CatmullRom shipped,
/// `FOCR_RESAMPLE=pil-bicubic` for oracle-exact comparison — DISC-001).
///
/// # Errors
/// Rejects a non-positive `base_size` exactly like [`preprocess_dynamic`].
pub fn preprocess_dynamic_squash(img: DynamicImage, base_size: usize) -> FocrResult<Preprocessed> {
    let mode = PreprocessMode::Base { base_size };
    let validated = validate_mode(mode)?;
    let original_size = img.dimensions();
    // PIL-faithful bicubic UNCONDITIONALLY at this site (not the
    // env-keyed [`resample_exact`]): at the multi-page 2.9x squash the
    // shipped CatmullRom kernel measurably garbles glyphs into OCR junk,
    // while the PIL kernel reproduces the reference plate text byte-exactly
    // (bd-1gv.26 oracle run, page_0009+page_0014). Parity outranks the
    // kernel-uniformity nicety (doctrine #1).
    let squashed =
        pil_resample::resize_bicubic(&img.to_rgb8(), validated.base_size, validated.base_size);
    Ok(Preprocessed {
        mode,
        global: view_tensor(&squashed.into()),
        tiles: Vec::new(),
        crop_grid: CropGrid::single(),
        original_size,
    })
}

#[derive(Debug, Clone, Copy)]
struct ValidatedMode {
    base_size: u32,
    tile_size: u32,
}

fn validate_mode(mode: PreprocessMode) -> FocrResult<ValidatedMode> {
    let base_size = validate_edge("base_size", mode.base_size(), BASE_SIZE)?;
    let tile_size = match mode {
        PreprocessMode::Base { .. } => 0,
        PreprocessMode::Gundam { tile_size, .. } => {
            validate_edge("tile_size", tile_size, GUNDAM_TILE_SIZE)?
        }
    };
    Ok(ValidatedMode {
        base_size,
        tile_size,
    })
}

fn validate_edge(name: &str, value: usize, max: usize) -> FocrResult<u32> {
    if value < PATCH_SIZE {
        return Err(FocrError::Usage(format!(
            "preprocess {name} must be at least {PATCH_SIZE} pixels, got {value}"
        )));
    }
    if value > max {
        return Err(FocrError::Usage(format!(
            "preprocess {name} must be <= {max} pixels for this model, got {value}"
        )));
    }
    if !value.is_multiple_of(PATCH_SIZE) {
        return Err(FocrError::Usage(format!(
            "preprocess {name} must be a multiple of patch size {PATCH_SIZE}, got {value}"
        )));
    }
    u32::try_from(value).map_err(|_| {
        FocrError::Usage(format!(
            "preprocess {name} exceeds u32 pixel edge limit: {value}"
        ))
    })
}

/// Decode an image file into an EXIF-transposed [`DynamicImage`] ([SPEC-020]).
///
/// `pub(crate)` so the figure-extraction path
/// ([`crate::native_engine::OcrModel::recognize_with_figures`]) can re-decode the
/// source with the EXACT same EXIF transform the forward used — the layout boxes
/// are in this image's pixel space, so the crop must come from this same decode.
pub(crate) fn decode_path(path: &Path) -> FocrResult<DynamicImage> {
    let reader = ImageReader::open(path)
        .map_err(|e| FocrError::InputDecode(format!("open {}: {e}", path.display())))?
        .with_guessed_format()
        .map_err(|e| FocrError::InputDecode(format!("sniff {}: {e}", path.display())))?;
    decode_reader(reader)
        .map_err(|e| FocrError::InputDecode(format!("decode {}: {e}", path.display())))
}

/// Decode raw image bytes into an EXIF-transposed [`DynamicImage`] ([SPEC-020]).
fn decode_bytes(bytes: &[u8]) -> FocrResult<DynamicImage> {
    let reader = ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| FocrError::InputDecode(format!("sniff bytes: {e}")))?;
    decode_reader(reader).map_err(|e| FocrError::InputDecode(format!("decode bytes: {e}")))
}

/// Decode a reader, applying its EXIF orientation in place — the equivalent of
/// PIL's `ImageOps.exif_transpose` ([SPEC-020],
/// `modeling_unlimitedocr.py:27-34`). `image::decode()` does NOT auto-apply
/// orientation, so we pull it from the decoder and `apply_orientation`.
fn decode_reader<R: std::io::BufRead + std::io::Seek>(
    reader: ImageReader<R>,
) -> image::ImageResult<DynamicImage> {
    let mut decoder = reader.into_decoder()?;
    // Decompression-bomb bound (bd-10sb.1 — found by the image_decode fuzz
    // target on its FIRST bounded smoke): a 100-byte PNG declaring a 2^31+4
    // height made `from_decoder` attempt a 38.6 GB allocation. Same 1 Gpx
    // policy as the PDF raster path (`pdf.rs` MAX_PIXELS), enforced BEFORE
    // any pixel allocation, plus the decoder's own internal limits.
    const MAX_PIXELS: u64 = 1 << 30; // 1 Gpx (matches pdf.rs)
    let (w, h) = decoder.dimensions();
    if u64::from(w) * u64::from(h) > MAX_PIXELS {
        return Err(image::ImageError::Limits(
            image::error::LimitError::from_kind(image::error::LimitErrorKind::DimensionError),
        ));
    }
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(1 << 17);
    limits.max_image_height = Some(1 << 17);
    limits.max_alloc = Some(4 << 30);
    decoder.set_limits(limits)?;
    let orientation = decoder.orientation()?;
    let mut img = DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    Ok(img)
}

// ── resample kernel selection (bd-30me, DISC-001) ───────────────────────────

/// Kill-switch for the L0 resampling kernel (DISC-001, bd-30me). Unset (the
/// default) keeps the shipped [`FilterType::CatmullRom`]; `pil-bicubic`
/// restores the reference-bit-exact Pillow BICUBIC ([`pil_resample`]) at every
/// resize site, for L0 EXACT comparison against the torch/PIL oracle.
pub const RESAMPLE_ENV: &str = "FOCR_RESAMPLE";

/// Which resampling kernel the preprocess resize sites use (see
/// [`RESAMPLE_ENV`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleKind {
    /// The shipped default: the `image` crate's Catmull-Rom cubic — the same
    /// `a = -0.5` continuous kernel as PIL BICUBIC, but with clamp-at-edge
    /// sampling and float accumulation, so NOT bit-identical (DISC-001).
    CatmullRom,
    /// The reference path: Pillow-bit-exact fixed-point BICUBIC
    /// ([`pil_resample::resize_bicubic`]), for oracle-exact preprocessing.
    PilBicubic,
}

/// Read the kernel selection from [`RESAMPLE_ENV`]. Re-read per resize (the
/// pipeline resizes a handful of times per image, so there is no `OnceLock`
/// cache to fight in tests or long-lived engine processes).
#[must_use]
pub fn resample_kind() -> ResampleKind {
    resample_kind_from(std::env::var(RESAMPLE_ENV).ok().as_deref())
}

/// [`resample_kind`] over an explicit raw value (unit-testable without
/// mutating the process environment). Unknown values keep the default — the
/// same forgiving parse as the other `FOCR_*` toggles (`env_tristate`).
fn resample_kind_from(raw: Option<&str>) -> ResampleKind {
    match raw.map(str::trim) {
        Some("pil-bicubic" | "pil_bicubic") => ResampleKind::PilBicubic,
        _ => ResampleKind::CatmullRom,
    }
}

/// Every preprocess resize funnels through here: `resize_exact` semantics
/// with the kernel chosen by [`resample_kind`] ([SPEC-022/024], spec §13b).
fn resample_exact(img: &DynamicImage, w: u32, h: u32) -> DynamicImage {
    resample_exact_with(resample_kind(), img, w, h)
}

/// [`resample_exact`] with the kernel pinned by the caller (unit-testable
/// without env mutation).
///
/// The CatmullRom arm is the pre-DISC-001 call verbatim — same filter, same
/// color-type behavior — so the default output is byte-identical (doctrine
/// #2). The PIL arm converts to RGB *first*, because the oracle does: both
/// `load_images` (`modeling_unlimitedocr.py:303`) and `GOTImageEvalProcessor`
/// `.convert("RGB")` before resizing, so reference resampling happens in RGB
/// space.
fn resample_exact_with(kind: ResampleKind, img: &DynamicImage, w: u32, h: u32) -> DynamicImage {
    match kind {
        ResampleKind::CatmullRom => img.resize_exact(w, h, FilterType::CatmullRom),
        ResampleKind::PilBicubic => {
            DynamicImage::ImageRgb8(pil_resample::resize_bicubic(&img.to_rgb8(), w, h))
        }
    }
}

// ── global view: aspect-preserving resize + gray pad ([SPEC-022]) ───────────

/// `ImageOps.pad`: resize aspect-preserving to fit inside `size × size`, then
/// center-pad the short axis with the mean gray color `(127,127,127)`
/// ([SPEC-022], `modeling_unlimitedocr.py:872-873`).
///
/// PIL `ImageOps.pad` uses `BICUBIC` resampling by default; the default kernel
/// here is `CatmullRom` (DISC-001; `FOCR_RESAMPLE=pil-bicubic` restores the
/// bit-exact reference). The pad geometry (fit + centered placement + gray
/// fill) is exact.
fn pad_to_square(img: &DynamicImage, size: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    // Aspect-preserving fit: scale so the longer side == size.
    // (w, h) -> (rw, rh) with max(rw, rh) == size, preserving ratio, >= 1px.
    let (rw, rh) = if w == 0 || h == 0 {
        (size, size)
    } else if w >= h {
        let rh = pillow_fit_edge(h, w, size);
        (size, rh)
    } else {
        let rw = pillow_fit_edge(w, h, size);
        (rw, size)
    };
    let resized = resample_exact(img, rw, rh).to_rgb8();

    // Center on a gray canvas.
    let mut canvas = image::RgbImage::from_pixel(size, size, image::Rgb([PAD_FILL; 3]));
    let ox = pillow_center_offset(size, rw);
    let oy = pillow_center_offset(size, rh);
    for y in 0..rh {
        for x in 0..rw {
            let p = *resized.get_pixel(x, y);
            canvas.put_pixel(ox + x, oy + y, p);
        }
    }
    DynamicImage::ImageRgb8(canvas)
}

fn pillow_fit_edge(short: u32, long: u32, size: u32) -> u32 {
    let scaled = f64::from(short) / f64::from(long) * f64::from(size);
    round_ties_even_positive(scaled).max(1)
}

fn pillow_center_offset(size: u32, resized: u32) -> u32 {
    round_ties_even_positive(f64::from(size - resized) * 0.5)
}

fn round_ties_even_positive(value: f64) -> u32 {
    debug_assert!(value.is_finite());
    debug_assert!(value >= 0.0);
    let floor = value.floor();
    let frac = value - floor;
    let rounded = if frac < 0.5 {
        floor
    } else if frac > 0.5 {
        floor + 1.0
    } else {
        let floor_u = floor as u64;
        if floor_u.is_multiple_of(2) {
            floor
        } else {
            floor + 1.0
        }
    };
    rounded as u32
}

// ── Gundam tiling ([SPEC-024], OQ-7) ─────────────────────────────────────────

/// Build the Gundam local tiles + chosen crop grid for an image ([SPEC-023/024],
/// OQ-7).
///
/// If the image is `<= 640` in BOTH dims, returns no tiles and a 1×1 grid (the
/// short-circuit at `modeling_unlimitedocr.py:859`). Otherwise selects the grid
/// via [`find_closest_aspect_ratio`], resizes to `(tile*W, tile*H)`, and slices
/// a row-major `W×H` grid of `tile×tile` tiles ([SPEC-024],
/// `modeling_unlimitedocr.py:192-208`).
fn build_gundam_tiles(img: &DynamicImage, tile: u32) -> FocrResult<(Vec<ViewTensor>, CropGrid)> {
    let (w, h) = img.dimensions();
    if w <= CROP_THRESHOLD && h <= CROP_THRESHOLD {
        // No crop ([SPEC-023]): crop_ratio = [1, 1], no local tiles.
        return Ok((Vec::new(), CropGrid::single()));
    }

    let ratios = candidate_ratios(MIN_NUM, MAX_NUM);
    let (wc, hc) = find_closest_aspect_ratio(w as f64 / h as f64, &ratios, w, h, tile);

    // Resize to (tile*W, tile*H), then crop a row-major W×H grid of tiles.
    let target_w = checked_tile_extent(tile, wc, "width")?;
    let target_h = checked_tile_extent(tile, hc, "height")?;
    // PIL `image.resize((W,H))` default resample is BICUBIC; the default kernel
    // here is CatmullRom (DISC-001; FOCR_RESAMPLE=pil-bicubic restores the
    // bit-exact reference). Tile geometry (crop boxes) is exact.
    let resized = resample_exact(img, target_w, target_h);

    let cols = wc; // target_width // tile
    let blocks = wc * hc;
    let mut tiles = Vec::with_capacity(blocks);
    for i in 0..blocks {
        // box = (col*tile, row*tile, (col+1)*tile, (row+1)*tile), row-major
        // ([SPEC-024], `modeling_unlimitedocr.py:199-208`).
        let col = (i % cols) as u32;
        let row = (i / cols) as u32;
        let split = resized.crop_imm(col * tile, row * tile, tile, tile);
        tiles.push(view_tensor(&split));
    }

    Ok((
        tiles,
        CropGrid {
            width_crop_num: wc,
            height_crop_num: hc,
        },
    ))
}

fn checked_tile_extent(tile: u32, count: usize, axis: &str) -> FocrResult<u32> {
    let count = u32::try_from(count).map_err(|_| {
        FocrError::Usage(format!(
            "preprocess Gundam {axis} tile count exceeds u32: {count}"
        ))
    })?;
    tile.checked_mul(count).ok_or_else(|| {
        FocrError::Usage(format!(
            "preprocess Gundam {axis} extent overflows u32: tile_size={tile}, tile_count={count}"
        ))
    })
}

/// Build the candidate `(width_tiles, height_tiles)` grids: all `(i, j)` with
/// `min_num <= i*j <= max_num`, deduplicated and sorted by tile count `i*j`
/// ([SPEC-024], OQ-7, `modeling_unlimitedocr.py:180-184`).
///
/// With defaults `(2, 32)` this yields 118 candidate grids (verified against the
/// pinned source). `(1, 1)` is NOT a candidate (`1 < min_num`).
#[must_use]
pub fn candidate_ratios(min_num: usize, max_num: usize) -> Vec<(usize, usize)> {
    let mut set = std::collections::BTreeSet::new();
    // The source iterates n in range(min_num, max_num+1) and i,j in 1..=n; the
    // membership test `min_num <= i*j <= max_num` is what actually selects pairs,
    // so iterating i,j in 1..=max_num directly yields the identical set.
    for i in 1..=max_num {
        for j in 1..=max_num {
            let prod = i * j;
            if (min_num..=max_num).contains(&prod) {
                set.insert((i, j));
            }
        }
    }
    let mut out: Vec<(usize, usize)> = set.into_iter().collect();
    // Stable sort by tile count i*j (matches the source's
    // `sorted(..., key=lambda x: x[0]*x[1])`); BTreeSet already gives a
    // deterministic (i, j) order within equal products.
    out.sort_by_key(|&(i, j)| i * j);
    out
}

/// Pick the candidate grid whose aspect ratio is closest to the image's
/// ([SPEC-025], OQ-7, `modeling_unlimitedocr.py:158-172`).
///
/// `aspect_ratio = width / height`. Minimizes `|aspect_ratio - i/j|`; on a tie,
/// prefers the larger grid only when `area > 0.5 * tile² * i * j`. Returns
/// `(width_crop_num, height_crop_num)`.
#[must_use]
pub fn find_closest_aspect_ratio(
    aspect_ratio: f64,
    target_ratios: &[(usize, usize)],
    width: u32,
    height: u32,
    tile: u32,
) -> (usize, usize) {
    let mut best_diff = f64::INFINITY;
    let mut best = (1usize, 1usize);
    let area = f64::from(width) * f64::from(height);
    let tile_f = f64::from(tile);
    for &(i, j) in target_ratios {
        let target = i as f64 / j as f64;
        let diff = (aspect_ratio - target).abs();
        if diff < best_diff {
            best_diff = diff;
            best = (i, j);
        } else if diff == best_diff && area > 0.5 * tile_f * tile_f * (i as f64) * (j as f64) {
            best = (i, j);
        }
    }
    best
}

// ── normalize: ToTensor -> Normalize(0.5, 0.5) => [-1, 1] ([SPEC-021]) ───────

/// Convert an RGB image to the `[3, H, W]` normalized pixel [`Mat`] the SAM
/// patch-embed consumes ([SPEC-021/041]).
///
/// `ToTensor` scales `u8` `[0,255]` -> `f32` `[0,1]`; `Normalize(mean=0.5,
/// std=0.5)` maps to `[-1,1]` as `(v - 0.5) / 0.5 = 2v - 1`. Layout is
/// channel-major (`rows=3`, `cols=H*W`), element `(c, y*W + x)`.
/// GOT-OCR2's OpenAI/CLIP normalization mean (`GOTImageEvalProcessor`, spec §13b).
pub const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
/// GOT-OCR2's OpenAI/CLIP normalization std.
pub const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];
/// GOT-OCR2 fixed input side (`image_size=1024`).
pub const GOT_SIZE: u32 = 1024;

/// GOT-OCR2 preprocess (`GOTImageEvalProcessor(image_size=1024)`): **squash**
/// bicubic resize to 1024×1024 (NO aspect-preserve, NO pad) + CLIP-norm →
/// `[3, 1024*1024]` channel-major, the exact tensor `vision_sam::forward` reads.
///
/// `CatmullRom ≈ PIL bicubic` is the one known sub-L0 divergence (spec §13b,
/// DISC-001): the stats match the torch oracle to ~1e-4 but the resample is not
/// bit-identical. `FOCR_RESAMPLE=pil-bicubic` restores the bit-exact kernel.
///
/// # Errors
/// [`FocrError`] if the image cannot be decoded.
pub fn preprocess_got(path: &Path) -> FocrResult<crate::native_engine::tensor::Mat> {
    Ok(got_view_tensor(&decode_path(path)?))
}

// ── SmolVLM2 preprocess (C7, bd-3jo6.3.7) ───────────────────────────────────

/// SmolVLM2 frame side (`preprocessor_config.json max_image_size.longest_edge`).
const SMOLVLM2_FRAME: u32 = 512;
/// SmolVLM2 step-1 long-side target (`size.longest_edge`) — always rescaled TO
/// this, so still images are always split (spec §6).
const SMOLVLM2_LONGEST: u32 = 2048;

/// SmolVLM2 preprocess output: `n_frames` normalized 512² frames (tiles
/// row-major, the global thumbnail LAST) + the tile grid the prompt builder
/// needs for the `<row_r_col_c>` expansion.
#[derive(Debug, Clone)]
pub struct Smolvlm2Preprocessed {
    /// `[n_frames, 3, 512, 512]` flat f32 (CHW per frame), the `x/255 → ±1`
    /// normalized rail the SigLIP tower reads.
    pub frames: Vec<f32>,
    /// Tile grid + global frame count (`rows * cols + 1`).
    pub n_frames: usize,
    /// Tile rows (`R` in the `<row_r_col_c>` markers).
    pub rows: usize,
    /// Tile cols.
    pub cols: usize,
}

/// SmolVLM2 preprocess (`SmolVLMImageProcessor`, spec §6) — an exact
/// transcription of `image_processing_smolvlm.py`, every resize
/// Pillow-bit-exact LANCZOS ([`pil_resample::resize_lanczos`]; `resample: 1`):
///
/// 1. `_resize_output_size_rescale_to_max_len`: longest edge → exactly 2048
///    (up- OR down-scale), short edge `int()`-truncated then `+1 if odd`.
/// 2. `resize_for_vision_encoder`: ceil each side to a 512 multiple (long
///    side first, short side recomputed from the aspect then ceiled).
/// 3. `split_image`: `R×C` exact 512² crops row-major, then the step-2 image
///    resized (squashed) to 512×512 appended as the global frame LAST — the
///    upstream global is the image split_image was handed, i.e. the
///    512-multiple step-2 result.
/// 4. Per frame: `u8 → f64·(1/255) → f32` (numpy `rescale` casts to f32),
///    then `(x - 0.5) / 0.5` in f32 (numpy `normalize` runs at image dtype).
///
/// # Errors
/// [`FocrError::Other`] on a degenerate (zero-sized) input image.
pub fn preprocess_smolvlm2(img: &DynamicImage) -> FocrResult<Smolvlm2Preprocessed> {
    let rgb = img.to_rgb8();
    let (w0, h0) = rgb.dimensions();
    if w0 == 0 || h0 == 0 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "smolvlm2 preprocess: degenerate {w0}x{h0} input image"
        )));
    }

    // Step 1: longest edge → exactly 2048 (aspect preserved; int() truncation
    // + `+1 if odd` on the derived edge; min clamp 1). Transcribed from
    // `_resize_output_size_rescale_to_max_len`.
    let aspect = f64::from(w0) / f64::from(h0);
    let (w2, h2) = if w0 >= h0 {
        let w = SMOLVLM2_LONGEST;
        let mut h = (f64::from(w) / aspect) as u32; // Python int() truncates
        if !h.is_multiple_of(2) {
            h += 1;
        }
        (w, h.max(1))
    } else {
        let h = SMOLVLM2_LONGEST;
        let mut w = (f64::from(h) * aspect) as u32;
        if !w.is_multiple_of(2) {
            w += 1;
        }
        (w.max(1), h)
    };
    let long2048 = pil_resample::resize_lanczos(&rgb, w2, h2);

    // Step 2: ceil to 512 multiples (`resize_for_vision_encoder` — long side
    // ceiled first, short side recomputed from the SAME aspect then ceiled).
    let ceil512 = |v: u32| v.div_ceil(SMOLVLM2_FRAME) * SMOLVLM2_FRAME;
    let aspect2 = f64::from(w2) / f64::from(h2);
    let (w3, h3) = if w2 >= h2 {
        let w = ceil512(w2);
        let h = ceil512((f64::from(w) / aspect2) as u32);
        (w, h)
    } else {
        let h = ceil512(h2);
        let w = ceil512((f64::from(h) * aspect2) as u32);
        (w, h)
    };
    let ceiled512 = if (w3, h3) == (w2, h2) {
        long2048.clone()
    } else {
        pil_resample::resize_lanczos(&long2048, w3, h3)
    };

    // Step 3: split into exact 512² tiles row-major + the global frame LAST.
    // Step 1 always makes the long side 2048 > 512, so the split always
    // engages (`split_image`'s no-split branch is unreachable here).
    let rows = (h3 / SMOLVLM2_FRAME) as usize;
    let cols = (w3 / SMOLVLM2_FRAME) as usize;
    let n_frames = rows * cols + 1;
    let side = SMOLVLM2_FRAME as usize;
    let frame_len = 3 * side * side;
    let mut frames = vec![0.0f32; n_frames * frame_len];

    // The exact numpy rescale→normalize chain (f64 mul, f32 cast, f32 affine).
    let norm = |px: u8| -> f32 {
        let r = (f64::from(px) * (1.0 / 255.0)) as f32;
        (r - 0.5) / 0.5
    };
    let mut write_frame = |idx: usize, tile: &image::RgbImage, ox: u32, oy: u32| {
        let dst = &mut frames[idx * frame_len..(idx + 1) * frame_len];
        for y in 0..side {
            for x in 0..side {
                let px = tile.get_pixel(ox + x as u32, oy + y as u32).0;
                let s = y * side + x;
                dst[s] = norm(px[0]);
                dst[side * side + s] = norm(px[1]);
                dst[2 * side * side + s] = norm(px[2]);
            }
        }
    };
    for r in 0..rows {
        for c in 0..cols {
            write_frame(
                r * cols + c,
                &ceiled512,
                c as u32 * SMOLVLM2_FRAME,
                r as u32 * SMOLVLM2_FRAME,
            );
        }
    }
    let global = pil_resample::resize_lanczos(&ceiled512, SMOLVLM2_FRAME, SMOLVLM2_FRAME);
    write_frame(n_frames - 1, &global, 0, 0);

    Ok(Smolvlm2Preprocessed {
        frames,
        n_frames,
        rows,
        cols,
    })
}

/// [`preprocess_smolvlm2`] from an image file path.
///
/// # Errors
/// [`FocrError`] if the image cannot be decoded, plus [`preprocess_smolvlm2`]'s.
pub fn preprocess_smolvlm2_path(path: &Path) -> FocrResult<Smolvlm2Preprocessed> {
    preprocess_smolvlm2(&decode_path(path)?)
}

/// OneChart preprocess (census §6, D3): the SAME squash-bicubic 1024² resize
/// as GOT, but the Normalize is a NO-OP — pixels stay raw `[0,1]` (mean 0,
/// std 1; the CLIP constants are NOT used). Returns `[3, 1024*1024]`
/// channel-major. Shares GOT's CatmullRom≈PIL-bicubic sub-L0 divergence
/// class (`FOCR_RESAMPLE=pil-bicubic` restores exactness; OQ-D3 re-derives
/// the tolerance at raw-pixel scale).
pub fn onechart_view_tensor(img: &DynamicImage) -> crate::native_engine::tensor::Mat {
    let rgb = resample_exact(img, GOT_SIZE, GOT_SIZE).to_rgb8();
    let side = GOT_SIZE as usize;
    let n = side * side;
    let mut data = vec![0.0f32; 3 * n];
    for y in 0..side {
        for x in 0..side {
            let px = rgb.get_pixel(x as u32, y as u32).0;
            let s = y * side + x;
            for c in 0..3 {
                data[c * n + s] = f32::from(px[c]) / 255.0;
            }
        }
    }
    crate::native_engine::tensor::Mat::from_vec(3, n, data)
}

/// [`preprocess_got`] over an already-decoded image (shared with the CLI/tests).
pub fn got_view_tensor(img: &DynamicImage) -> crate::native_engine::tensor::Mat {
    let rgb = resample_exact(img, GOT_SIZE, GOT_SIZE).to_rgb8();
    let side = GOT_SIZE as usize;
    let n = side * side;
    let mut data = vec![0.0f32; 3 * n];
    for y in 0..side {
        for x in 0..side {
            let px = rgb.get_pixel(x as u32, y as u32).0;
            let s = y * side + x;
            for c in 0..3 {
                let v = f32::from(px[c]) / 255.0;
                data[c * n + s] = (v - CLIP_MEAN[c]) / CLIP_STD[c];
            }
        }
    }
    crate::native_engine::tensor::Mat::from_vec(3, n, data)
}

fn view_tensor(img: &DynamicImage) -> ViewTensor {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let (wi, hi) = (w as usize, h as usize);
    let n = wi * hi;
    let mut data = vec![0.0f32; 3 * n];
    for y in 0..hi {
        for x in 0..wi {
            let px = rgb.get_pixel(x as u32, y as u32).0;
            let s = y * wi + x;
            for c in 0..3 {
                let v = f32::from(px[c]) / 255.0;
                data[c * n + s] = (v - IMAGE_MEAN[c]) / IMAGE_STD[c];
            }
        }
    }
    ViewTensor {
        pixels: crate::native_engine::tensor::Mat::from_vec(3, n, data),
        height: hi,
        width: wi,
    }
}

// ───────────────── TrOMR staff preprocess (E9, tromr-spec §6) ─────────────────

/// The `readimg` normalize constants (albumentations `Normalize(mean=0.7931,
/// std=0.1738, max_pixel_value=255)` — spec §6).
const TROMR_MEAN: f32 = 0.7931 * 255.0;
const TROMR_STD: f32 = 0.1738 * 255.0;

/// Half-pixel-center bilinear resize of one u8 plane (the cv2 `INTER_LINEAR`
/// sampling geometry: `sx = (dx+0.5)·w/nw − 0.5`, edge-clamped, NO area
/// averaging on downscale — upstream resizes staves this way, quality
/// warts and all). Float weights + round; cv2's 11-bit fixed-point
/// arithmetic can differ by ±1 LSB — a MEASURED envelope, ledgered in the
/// armed cert (the DISC-001 resample precedent).
fn bilinear_u8(src: &[u8], w: usize, h: usize, nw: usize, nh: usize) -> Vec<u8> {
    let mut out = vec![0u8; nw * nh];
    let sx_ratio = w as f32 / nw as f32;
    let sy_ratio = h as f32 / nh as f32;
    for dy in 0..nh {
        let fy = ((dy as f32 + 0.5) * sy_ratio - 0.5).max(0.0);
        let y0 = (fy as usize).min(h - 1);
        let y1 = (y0 + 1).min(h - 1);
        let wy = fy - y0 as f32;
        for dx in 0..nw {
            let fx = ((dx as f32 + 0.5) * sx_ratio - 0.5).max(0.0);
            let x0 = (fx as usize).min(w - 1);
            let x1 = (x0 + 1).min(w - 1);
            let wx = fx - x0 as f32;
            let top = f32::from(src[y0 * w + x0]) * (1.0 - wx) + f32::from(src[y0 * w + x1]) * wx;
            let bot = f32::from(src[y1 * w + x0]) * (1.0 - wx) + f32::from(src[y1 * w + x1]) * wx;
            out[dy * nw + dx] = (top * (1.0 - wy) + bot * wy).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// TrOMR staff preprocess (`staff2score.py::readimg`, spec §6): the ink gray
/// plane (RGBA ⇒ `255 − alpha`, the rendered-PNG convention; RGB ⇒ the cv2
/// fixed-point luma `(4899·R + 9617·G + 1868·B + 8192) >> 14`; gray ⇒ as-is)
/// → half-pixel-center bilinear resize to `h=128, w = ⌊(128/h)·w⌋` floored to
/// a multiple of 16 → normalize `(px − 0.7931·255)/(0.1738·255)`. Returns
/// `(pixels[128·W], W)` ready for [`crate::native_engine::tromr::encode`].
///
/// Channel-order note: upstream converts to gray AFTER the resize, but the
/// RGBA path resizes three REPLICATED gray channels (identical results) and
/// the RGB path's luma-then-resize vs resize-then-luma differ only through
/// the same ±1-LSB rounding envelope the armed cert measures.
///
/// # Errors
/// A degenerate image, or a resized width of 0 or past the 1280 position
/// clamp (the E5 front end guarantees the bound; a raw over-wide crop is a
/// clean error, never undefined crop-indexing — spec §2b).
pub fn tromr_staff_tensor(img: &DynamicImage) -> FocrResult<(Vec<f32>, usize)> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w == 0 || h == 0 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr preprocess: degenerate {w}x{h} input"
        )));
    }
    // The ink gray plane. The inverted-alpha convention applies ONLY when
    // the alpha channel varies: upstream applies 255−alpha to EVERY
    // 4-channel input, which BLANKS fully-opaque PNGs (their own demo
    // staves are opaque RGBA — measured 2026-07-06, DISC-007). A deliberate,
    // documented divergence; opaque-alpha images take the RGB luma path.
    let alpha_is_ink = img.color().has_alpha() && img.to_rgba8().pixels().any(|p| p.0[3] < 255);
    let gray: Vec<u8> = if alpha_is_ink {
        img.to_rgba8().pixels().map(|p| 255 - p.0[3]).collect()
    } else {
        img.to_rgb8()
            .pixels()
            .map(|p| {
                let [r, g, b] = p.0;
                ((4899 * u32::from(r) + 9617 * u32::from(g) + 1868 * u32::from(b) + 8192) >> 14)
                    .min(255) as u8
            })
            .collect()
    };
    let new_h = crate::native_engine::tromr::IMG_H;
    let new_w = ((new_h as f64 / h as f64 * w as f64) as usize) / 16 * 16;
    if new_w == 0 {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr preprocess: {w}x{h} resizes to zero width (image too narrow)"
        )));
    }
    if new_w > crate::native_engine::tromr::POS_COLS * crate::native_engine::tromr::PATCH {
        return Err(FocrError::Other(anyhow::anyhow!(
            "tromr preprocess: resized width {new_w} exceeds the 1280 position clamp — \
             pass a single-staff crop (aspect ≤ 10:1 at h=128; the staff-detection \
             front end enforces this)"
        )));
    }
    let resized = bilinear_u8(&gray, w, h, new_w, new_h);
    let pixels = resized
        .iter()
        .map(|&v| (f32::from(v) - TROMR_MEAN) / TROMR_STD)
        .collect();
    Ok((pixels, new_w))
}

#[cfg(test)]
mod tests {
    #[test]
    fn tromr_alpha_ink_path_fires_only_when_alpha_varies() {
        // DISC-007: the inverted-alpha ink convention applies ONLY to PNGs
        // whose alpha channel varies (rendered transparent-background
        // staves); fully-opaque RGBA takes the luma path.
        use image::{DynamicImage, Rgba, RgbaImage};
        // Varying alpha: ink strip (alpha 255) on transparent paper (alpha 0),
        // RGB deliberately garbage — the ink must come from alpha alone.
        let mut var = RgbaImage::from_pixel(64, 128, Rgba([9, 9, 9, 0]));
        for y in 60..68 {
            for x in 0..64 {
                var.put_pixel(x, y, Rgba([200, 200, 200, 255]));
            }
        }
        let (px, w) = super::tromr_staff_tensor(&DynamicImage::ImageRgba8(var))
            .expect("varying-alpha preprocess runs");
        assert_eq!(w, 64);
        // 255 − alpha: the strip (alpha 255) is INK (dark, 0), the rest paper
        // (255). Normalized: dark << 0 << light region values.
        let dark = (0.0f32 - 0.7931 * 255.0) / (0.1738 * 255.0);
        let light = (255.0f32 - 0.7931 * 255.0) / (0.1738 * 255.0);
        let mid = px[64 * 64 + 32]; // row 64 (inside the strip), col 32
        let top = px[10 * 64 + 32];
        assert!((mid - dark).abs() < 1e-4, "strip is ink: {mid} vs {dark}");
        assert!(
            (top - light).abs() < 1e-4,
            "background is paper: {top} vs {light}"
        );

        // Fully-opaque RGBA: the luma path (alpha ignored); a dark-RGB strip
        // must be the ink instead.
        let mut opaque = RgbaImage::from_pixel(64, 128, Rgba([250, 250, 250, 255]));
        for y in 60..68 {
            for x in 0..64 {
                opaque.put_pixel(x, y, Rgba([10, 10, 10, 255]));
            }
        }
        let (px, _) = super::tromr_staff_tensor(&DynamicImage::ImageRgba8(opaque))
            .expect("opaque-alpha preprocess runs");
        let mid = px[64 * 64 + 32];
        let top = px[10 * 64 + 32];
        assert!(
            mid < top,
            "opaque path reads RGB ink: strip {mid} vs paper {top}"
        );
        assert!(mid < -3.0, "the dark strip is strongly negative: {mid}");
    }

    use super::*;
    use image::{Rgb, RgbImage};

    /// **L0b — GOT preprocess vs the torch oracle.** `preprocess_got` on the shared
    /// `sample_text.png` must match `GOTImageEvalProcessor`'s output stats (from
    /// `oracle_fixtures.json` `l0b_preprocess`). The CatmullRom-vs-PIL-bicubic
    /// resample is the one known sub-L0 divergence, so aggregate stats match to a
    /// small tolerance (not bit-exact).
    #[test]
    fn got_preprocess_matches_oracle_l0b() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/got/sample_text.png"
        );
        let m = preprocess_got(std::path::Path::new(path)).expect("got preprocess");
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, (GOT_SIZE * GOT_SIZE) as usize);

        let d: Vec<f64> = m.data.iter().map(|&v| f64::from(v)).collect();
        let mean = d.iter().sum::<f64>() / d.len() as f64;
        let var = d.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / d.len() as f64;
        let std = var.sqrt();
        let min = d.iter().copied().fold(f64::INFINITY, f64::min);
        let max = d.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        eprintln!("[L0b] mean={mean:.4} std={std:.4} min={min:.4} max={max:.4}");

        // oracle l0b_preprocess (transformers 4.45.2, f32 CPU).
        let (o_mean, o_std, o_min, o_max) = (2.046_326_7, 0.138_841_3, -1.777_664_1, 2.145_897);
        assert!((mean - o_mean).abs() < 5e-3, "mean {mean} vs {o_mean}");
        assert!((std - o_std).abs() < 5e-3, "std {std} vs {o_std}");
        // CLIP-normalized extremes: white→(1-mean)/std, black→-mean/std; robust to resample.
        assert!((min - o_min).abs() < 1e-2, "min {min} vs {o_min}");
        assert!((max - o_max).abs() < 1e-2, "max {max} vs {o_max}");
    }

    /// Build a tiny synthetic RGB image (no weights/tokenizer needed).
    fn solid(w: u32, h: u32, color: [u8; 3]) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_pixel(w, h, Rgb(color)))
    }

    // ── num_queries / census ([SPEC-027], OQ-18) ────────────────────────────

    #[test]
    fn num_queries_matches_census() {
        // base 1024 -> 16; local 640 -> 10 (OQ-18).
        assert_eq!(num_queries(1024), 16);
        assert_eq!(num_queries(640), 10);
    }

    #[test]
    fn base_global_placeholder_is_273() {
        // (16+1)*16 + 1 = 273 ([SPEC-028]/OQ-18(a)).
        let p = Preprocessed {
            mode: PreprocessMode::base(),
            global: view_tensor(&solid(8, 8, [0, 0, 0])),
            tiles: Vec::new(),
            crop_grid: CropGrid::single(),
            original_size: (8, 8),
        };
        assert_eq!(p.placeholder_token_count(), 273);
        assert_eq!(p.num_views(), 1);
    }

    /// The multi-page per-page census (bd-1gv.25): `infer_multi` runs every
    /// page at Base 640 ⇒ `num_queries(640) = 10` ⇒ `(10+1)·10 + 1 = 111`
    /// placeholder slots per page — the exact per-page block the reference
    /// concatenates at the prompt's single `<image>` position.
    #[test]
    fn multi_page_base_640_placeholder_is_111() {
        assert_eq!(num_queries(640), 10);
        let p = Preprocessed {
            mode: PreprocessMode::Base { base_size: 640 },
            global: view_tensor(&solid(8, 8, [0, 0, 0])),
            tiles: Vec::new(),
            crop_grid: CropGrid::single(),
            original_size: (8, 8),
        };
        assert_eq!(p.placeholder_token_count(), 111);
        assert_eq!(p.num_views(), 1);
        println!(r#"{{"check":"multi_page_census_640","per_page":111,"result":"pass"}}"#);
    }

    /// The Gundam multi-tile totals from the OQ-18 census table:
    /// 273 global + (10W+1)(10H) local.
    #[test]
    fn gundam_placeholder_census_matches_table() {
        let cases = [
            ((2, 1), 483usize),
            ((1, 2), 493),
            ((2, 2), 693),
            ((3, 2), 893),
            ((4, 4), 1913),
        ];
        for ((w, h), expected) in cases {
            let grid = CropGrid {
                width_crop_num: w,
                height_crop_num: h,
            };
            let p = Preprocessed {
                mode: PreprocessMode::gundam(),
                global: view_tensor(&solid(4, 4, [0, 0, 0])),
                tiles: vec![view_tensor(&solid(4, 4, [0, 0, 0])); grid.blocks()],
                crop_grid: grid,
                original_size: (4, 4),
            };
            assert_eq!(
                p.placeholder_token_count(),
                expected,
                "grid {w}x{h} census mismatch"
            );
        }
    }

    #[test]
    fn gundam_no_crop_grid_is_273() {
        // A 1x1 Gundam grid emits only the 273 global block (no local tiles).
        let p = Preprocessed {
            mode: PreprocessMode::gundam(),
            global: view_tensor(&solid(4, 4, [0, 0, 0])),
            tiles: Vec::new(),
            crop_grid: CropGrid::single(),
            original_size: (4, 4),
        };
        assert_eq!(p.placeholder_token_count(), 273);
    }

    // ── candidate_ratios ([SPEC-024], OQ-7) ─────────────────────────────────

    #[test]
    fn candidate_ratios_count_and_bounds() {
        let r = candidate_ratios(MIN_NUM, MAX_NUM);
        // OQ-7: defaults (2, 32) yield exactly 118 candidate grids.
        assert_eq!(r.len(), 118);
        // (1,1) excluded (product 1 < min_num); products within [2, 32].
        assert!(!r.contains(&(1, 1)));
        for &(i, j) in &r {
            assert!((MIN_NUM..=MAX_NUM).contains(&(i * j)));
        }
        // Sorted by tile count i*j (non-decreasing).
        for pair in r.windows(2) {
            assert!(pair[0].0 * pair[0].1 <= pair[1].0 * pair[1].1);
        }
        // Spot-check the extremes are present.
        assert!(r.contains(&(1, 2)));
        assert!(r.contains(&(2, 1)));
        assert!(r.contains(&(4, 8)));
    }

    // ── find_closest_aspect_ratio ([SPEC-025], OQ-7) ────────────────────────

    #[test]
    fn closest_ratio_picks_documented_grids() {
        let ratios = candidate_ratios(MIN_NUM, MAX_NUM);
        // A 1280x640 image (aspect 2.0) -> grid (2,1): exact 2/1 match.
        let g = find_closest_aspect_ratio(1280.0 / 640.0, &ratios, 1280, 640, 640);
        assert_eq!(g, (2, 1));
        // A 640x1280 image (aspect 0.5) -> grid (1,2).
        let g = find_closest_aspect_ratio(640.0 / 1280.0, &ratios, 640, 1280, 640);
        assert_eq!(g, (1, 2));
        // A near-square large image (1300x1280, aspect ~1.016) -> (1,1) is NOT a
        // candidate, so it picks a square-ish grid with product>=2. The closest
        // ratio to ~1.0 among candidates is a k×k grid; (2,2) (ratio 1.0) is the
        // unique nearest with the smallest product among square grids.
        let g = find_closest_aspect_ratio(1300.0 / 1280.0, &ratios, 1300, 1280, 640);
        assert_eq!(g.0, g.1, "near-square should pick a square grid, got {g:?}");
    }

    #[test]
    fn closest_ratio_tie_break_prefers_larger_area() {
        // A GENUINE tie: aspect 1.0 with two square grids (2,2) and (3,3), both
        // exactly ratio 1.0 (diff 0.0). The tie-break ([SPEC-025] line 169)
        // upgrades to the later, larger grid ONLY when
        // `area > 0.5 * tile² * i * j`. Threshold for (3,3) at tile=640 is
        // 0.5*640²*9 = 1_843_200 px² (verified against the pinned source).
        let ratios = vec![(2usize, 2usize), (3usize, 3usize)];
        // Large area (4000*4000 = 16M > 1.84M): tie-break upgrades to (3,3).
        let g_big = find_closest_aspect_ratio(1.0, &ratios, 4000, 4000, 640);
        assert_eq!(g_big, (3, 3));
        // Tiny area (100*100 = 10k < threshold): no upgrade, keeps (2,2).
        let g_small = find_closest_aspect_ratio(1.0, &ratios, 100, 100, 640);
        assert_eq!(g_small, (2, 2));
    }

    // ── normalization math ([SPEC-021]) ─────────────────────────────────────

    #[test]
    fn normalize_maps_to_minus_one_one() {
        // 2x2 image: tl=black(0), tr=white(255), bl=mid(128), br=white.
        let mut img = RgbImage::new(2, 2);
        img.put_pixel(0, 0, Rgb([0, 0, 0]));
        img.put_pixel(1, 0, Rgb([255, 255, 255]));
        img.put_pixel(0, 1, Rgb([128, 128, 128]));
        img.put_pixel(1, 1, Rgb([255, 255, 255]));
        let vt = view_tensor(&DynamicImage::ImageRgb8(img));

        assert_eq!(vt.shape(), (2, 2));
        // pixels is [3, H*W] channel-major, spatial index s = y*W + x.
        assert_eq!(vt.pixels.rows, 3);
        assert_eq!(vt.pixels.cols, 4);

        // (0,0) black -> 2*0 - 1 = -1 in every channel; s=0.
        for c in 0..3 {
            assert!((vt.pixels.get(c, 0) - (-1.0)).abs() < 1e-6);
        }
        // (1,0) white -> 2*1 - 1 = 1; s = 0*2 + 1 = 1.
        for c in 0..3 {
            assert!((vt.pixels.get(c, 1) - 1.0).abs() < 1e-6);
        }
        // (0,1) mid 128 -> 2*(128/255) - 1 = 0.00392...; s = 1*2 + 0 = 2.
        let expected_mid = 2.0 * (128.0f32 / 255.0) - 1.0;
        for c in 0..3 {
            assert!((vt.pixels.get(c, 2) - expected_mid).abs() < 1e-6);
        }
    }

    // ── base-mode shape ([SPEC-029]) ────────────────────────────────────────

    #[test]
    fn base_mode_single_padded_square_view() {
        // A wide 100x40 image in base mode -> one 64x64 padded view (use a small
        // base_size so the test is fast; geometry is identical at 1024).
        let img = solid(100, 40, [200, 100, 50]);
        let p = preprocess_dynamic(img, PreprocessMode::Base { base_size: 64 }).unwrap();
        assert_eq!(p.num_views(), 1);
        assert!(p.tiles.is_empty());
        assert_eq!(p.crop_grid, CropGrid::single());
        assert_eq!(p.global.shape(), (64, 64));
        assert_eq!(p.global.pixels.rows, 3);
        assert_eq!(p.global.pixels.cols, 64 * 64);
        assert_eq!(p.original_size, (100, 40));

        // The pad rows (top/bottom gray bands) must be the normalized gray value
        // (127/255 -> 2*(127/255)-1). 100x40 -> fit width 64 =>
        // rh=round(40/100*64)=26, centered with oy=round((64-26)*0.5)=19, so
        // row 0 is pad.
        let gray = 2.0 * (f32::from(PAD_FILL) / 255.0) - 1.0;
        // s for (x=0, y=0) is 0; channel 0.
        assert!((p.global.pixels.get(0, 0) - gray).abs() < 1e-6);
    }

    /// The image_decode fuzz target's day-one find (bd-10sb.1): this exact
    /// 100-byte PNG declares 6 x 2^31+4 dimensions — pre-fix, decode
    /// attempted a 38.6 GB allocation (a DoS on any untrusted-image ingest).
    /// The decode funnel's 1 Gpx bound must reject it as a typed
    /// InputDecode error BEFORE any pixel allocation.
    #[test]
    fn decompression_bomb_png_is_rejected_before_allocation() {
        const BOMB: [u8; 100] = [
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x06, 0x80, 0x00, 0x00, 0x04, 0x08, 0x02, 0x00, 0x00,
            0x00, 0x22, 0x66, 0xd9, 0x14, 0x00, 0x00, 0x00, 0x56, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x01, 0xed, 0xc0, 0x03, 0xa0, 0x24, 0x3b, 0x6b, 0xd5, 0xaf, 0xc2, 0x67, 0x3f, 0x1a,
            0x1e, 0x0d, 0x8f, 0x86, 0x47, 0xc3, 0xa3, 0xa1, 0xf2, 0xea, 0x3c, 0x10, 0x50, 0x79,
            0x75, 0x1e, 0x08, 0xa8, 0xbc, 0x3a, 0x0f, 0x04, 0xfc, 0x23, 0x74, 0xa4, 0x02, 0x0d,
            0x6a, 0x18, 0x6a, 0x6a, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42,
            0x60, 0x82,
        ];
        let err = decode_bytes(&BOMB).expect_err("2^31-height PNG must be rejected");
        assert!(
            matches!(err, crate::FocrError::InputDecode(_)),
            "bomb must be a typed InputDecode, got {err:?}"
        );
        // And the whole preprocess entry stays total on it too.
        let err = preprocess_bytes(&BOMB, PreprocessMode::base()).expect_err("rejected");
        assert!(matches!(err, crate::FocrError::InputDecode(_)));
        println!(r#"{{"check":"decompression_bomb_bound","bytes":100,"result":"pass"}}"#);
    }

    #[test]
    fn pad_to_square_matches_pillow_rounding_geometry() {
        // Pillow ImageOps.contain rounds the fitted short edge, and ImageOps.pad
        // rounds the centered paste offset. Integer floor division shifts the
        // content/pad boundary by one pixel for both of these ordinary cases.
        let color = [200, 0, 0];
        let pad = [PAD_FILL; 3];

        // 100x40 -> 64x26, y offset 19. The old floor-based path made this
        // 64x25 and left row 44 as padding.
        let rounded_size = pad_to_square(&solid(100, 40, color), 64).to_rgb8();
        assert_eq!(rounded_size.get_pixel(0, 18).0, pad);
        assert_eq!(rounded_size.get_pixel(0, 19).0, color);
        assert_eq!(rounded_size.get_pixel(0, 44).0, color);
        assert_eq!(rounded_size.get_pixel(0, 45).0, pad);

        // 101x40 -> 64x25, y offset round(39*0.5)=20. The old floor-based
        // centering started content at row 19.
        let rounded_offset = pad_to_square(&solid(101, 40, color), 64).to_rgb8();
        assert_eq!(rounded_offset.get_pixel(0, 19).0, pad);
        assert_eq!(rounded_offset.get_pixel(0, 20).0, color);
        assert_eq!(rounded_offset.get_pixel(0, 44).0, color);
        assert_eq!(rounded_offset.get_pixel(0, 45).0, pad);
    }

    // ── resample kernel selection (bd-30me / DISC-001) ──────────────────────

    #[test]
    fn resample_kind_default_and_kill_switch_parse() {
        // Kill-switch OFF by default (doctrine #2): unset, empty, the default
        // spelled out, and unknown junk all stay CatmullRom.
        assert_eq!(resample_kind_from(None), ResampleKind::CatmullRom);
        assert_eq!(resample_kind_from(Some("")), ResampleKind::CatmullRom);
        assert_eq!(
            resample_kind_from(Some("catmullrom")),
            ResampleKind::CatmullRom
        );
        assert_eq!(resample_kind_from(Some("bogus")), ResampleKind::CatmullRom);
        // The documented value (and its underscore spelling, trimmed) arms
        // the PIL-bit-exact reference path.
        assert_eq!(
            resample_kind_from(Some("pil-bicubic")),
            ResampleKind::PilBicubic
        );
        assert_eq!(
            resample_kind_from(Some("pil_bicubic")),
            ResampleKind::PilBicubic
        );
        assert_eq!(
            resample_kind_from(Some(" pil-bicubic ")),
            ResampleKind::PilBicubic
        );
    }

    /// Doctrine #2 regression: the CatmullRom arm of the resample dispatch is
    /// byte-identical to the pre-DISC-001 direct `resize_exact` call — same
    /// bytes AND same color-type behavior (an RGBA input stays RGBA, exactly
    /// as the old inline call left it for the later `to_rgb8()`).
    #[test]
    fn default_resample_is_catmullrom_byte_identical() {
        let mut rgba = image::RgbaImage::new(13, 7);
        for (x, y, p) in rgba.enumerate_pixels_mut() {
            *p = image::Rgba([(x * 19 + y * 3) as u8, (x * 7) as u8, (y * 31) as u8, 255]);
        }
        let img = DynamicImage::ImageRgba8(rgba);
        let via_dispatch = resample_exact_with(ResampleKind::CatmullRom, &img, 8, 5);
        let direct = img.resize_exact(8, 5, FilterType::CatmullRom);
        assert_eq!(
            via_dispatch.color(),
            direct.color(),
            "default resample changed the color type"
        );
        assert_eq!(
            via_dispatch.as_bytes(),
            direct.as_bytes(),
            "default resample output moved (doctrine #2 violation)"
        );
    }

    /// The armed kill-switch routes to [`pil_resample::resize_bicubic`] over
    /// the RGB-converted image (PIL converts to RGB before resizing) and
    /// yields an RGB8 result.
    #[test]
    fn pil_kill_switch_dispatch_routes_to_pil_resampler() {
        let mut rgb = RgbImage::new(5, 4);
        for (x, y, p) in rgb.enumerate_pixels_mut() {
            *p = Rgb([(x * 40) as u8, (y * 60) as u8, (x * y * 13) as u8]);
        }
        let img = DynamicImage::ImageRgb8(rgb.clone());
        let via_dispatch = resample_exact_with(ResampleKind::PilBicubic, &img, 3, 6);
        let direct = pil_resample::resize_bicubic(&rgb, 3, 6);
        assert_eq!(via_dispatch.color(), image::ColorType::Rgb8);
        assert_eq!(via_dispatch.as_bytes(), direct.as_raw().as_slice());
    }

    // ── Gundam tiling geometry ([SPEC-023/024]) ─────────────────────────────

    #[test]
    fn gundam_small_image_short_circuits_to_no_crop() {
        // <=640 in both dims -> no tiles, 1x1 grid ([SPEC-023]).
        let img = solid(320, 200, [10, 20, 30]);
        let p = preprocess_dynamic(img, PreprocessMode::gundam()).unwrap();
        assert!(p.tiles.is_empty());
        assert_eq!(p.crop_grid, CropGrid::single());
        // Global view still at 1024.
        assert_eq!(p.global.shape(), (1024, 1024));
        assert_eq!(p.placeholder_token_count(), 273);
    }

    #[test]
    fn gundam_wide_image_tiles_into_grid() {
        // A 2:1 wide image exceeding 640 in width -> the pinned (2,1) grid at the
        // real tile_size=640 (the tie-break threshold uses tile², so the grid is
        // tile-size-dependent; we exercise the SHIPPED config here).
        let img = solid(1000, 500, [50, 60, 70]);
        // Expected grid is whatever the pinned selector yields for the shipped
        // tile_size; assert it against the public function so the test tracks the
        // spec, not a guessed constant.
        let ratios = candidate_ratios(MIN_NUM, MAX_NUM);
        let expected = find_closest_aspect_ratio(1000.0 / 500.0, &ratios, 1000, 500, 640);
        assert_eq!(expected, (2, 1), "pinned config: 2:1 wide -> (2,1)");

        let p = preprocess_dynamic(img, PreprocessMode::gundam()).unwrap();
        assert_eq!(
            p.crop_grid,
            CropGrid {
                width_crop_num: expected.0,
                height_crop_num: expected.1,
            }
        );
        assert_eq!(p.tiles.len(), expected.0 * expected.1);
        for t in &p.tiles {
            assert_eq!(t.shape(), (GUNDAM_TILE_SIZE, GUNDAM_TILE_SIZE));
            assert_eq!(t.pixels.rows, 3);
            assert_eq!(t.pixels.cols, GUNDAM_TILE_SIZE * GUNDAM_TILE_SIZE);
        }
        // Global thumbnail at base_size 1024.
        assert_eq!(p.global.shape(), (BASE_SIZE, BASE_SIZE));
        assert_eq!(p.num_views(), 1 + expected.0 * expected.1);
    }

    #[test]
    fn gundam_tile_count_equals_grid_blocks() {
        // A tall image -> some grid; tiles.len() must equal the grid block count
        // (the row-major slice loop is exact, [SPEC-024]). Uses a small base/tile
        // for speed; the census uses num_queries(tile_size), so derive q_local
        // from the mode rather than hard-coding 10.
        let img = solid(700, 2100, [1, 2, 3]); // aspect 1/3
        let mode = PreprocessMode::Gundam {
            base_size: 128,
            tile_size: 64,
        };
        let p = preprocess_dynamic(img, mode).unwrap();
        assert_eq!(p.tiles.len(), p.crop_grid.blocks());
        assert!(p.crop_grid.is_tiled());
        // census = global + (q_local*W + 1)(q_local*H), q_base/q_local per size.
        let q_base = num_queries(128);
        let q_local = num_queries(64);
        let w = p.crop_grid.width_crop_num;
        let h = p.crop_grid.height_crop_num;
        let expected = (q_base + 1) * q_base + 1 + (q_local * w + 1) * (q_local * h);
        assert_eq!(p.placeholder_token_count(), expected);
    }

    #[test]
    fn preprocess_rejects_invalid_mode_sizes() {
        let img = solid(32, 32, [1, 2, 3]);
        let cases = [
            PreprocessMode::Base { base_size: 0 },
            PreprocessMode::Base { base_size: 15 },
            PreprocessMode::Base { base_size: 2048 },
            PreprocessMode::Gundam {
                base_size: BASE_SIZE,
                tile_size: 0,
            },
            PreprocessMode::Gundam {
                base_size: BASE_SIZE,
                tile_size: 1024,
            },
            PreprocessMode::Gundam {
                base_size: BASE_SIZE,
                tile_size: 15,
            },
            PreprocessMode::Gundam {
                base_size: 2048,
                tile_size: GUNDAM_TILE_SIZE,
            },
        ];
        for mode in cases {
            let err = preprocess_dynamic(img.clone(), mode).unwrap_err();
            assert!(
                matches!(err, FocrError::Usage(_)),
                "mode {mode:?} should be a usage error, got {err:?}"
            );
        }
    }

    #[test]
    fn malformed_crop_grid_blocks_rejects_overflow() {
        let grid = CropGrid {
            width_crop_num: usize::MAX,
            height_crop_num: usize::MAX,
        };
        let panic = std::panic::catch_unwind(|| grid.blocks()).expect_err("overflow must panic");
        let message = panic_message(panic);
        assert!(message.contains("CropGrid::blocks"));
    }

    #[test]
    fn placeholder_count_rejects_malformed_grid_overflow() {
        let grid = CropGrid {
            width_crop_num: usize::MAX,
            height_crop_num: usize::MAX,
        };
        let p = Preprocessed {
            mode: PreprocessMode::gundam(),
            global: view_tensor(&solid(4, 4, [0, 0, 0])),
            tiles: Vec::new(),
            crop_grid: grid,
            original_size: (4, 4),
        };
        let panic = std::panic::catch_unwind(|| p.placeholder_token_count())
            .expect_err("overflow must panic");
        let message = panic_message(panic);
        assert!(message.contains("Preprocessed::placeholder_token_count"));
    }

    fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
        if let Some(s) = panic.downcast_ref::<&str>() {
            (*s).to_owned()
        } else if let Some(s) = panic.downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic>".to_owned()
        }
    }

    // ── decode error path ───────────────────────────────────────────────────

    #[test]
    fn preprocess_missing_file_is_input_decode_error() {
        let r = preprocess_image(
            Path::new("/definitely/not/a/real/image.png"),
            PreprocessMode::base(),
        );
        assert!(matches!(r, Err(FocrError::InputDecode(_))));
    }

    #[test]
    fn preprocess_garbage_bytes_is_input_decode_error() {
        let r = preprocess_bytes(&[0u8, 1, 2, 3, 4, 5, 6, 7], PreprocessMode::base());
        assert!(matches!(r, Err(FocrError::InputDecode(_))));
    }

    // ── round-trip through PNG bytes (decode path exercised) ─────────────────

    #[test]
    fn preprocess_bytes_decodes_real_png() {
        // Encode a tiny PNG in-memory, then preprocess it (no fixtures on disk).
        let img = solid(50, 30, [123, 45, 67]);
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let bytes = buf.into_inner();
        let p = preprocess_bytes(&bytes, PreprocessMode::Base { base_size: 32 }).unwrap();
        assert_eq!(p.global.shape(), (32, 32));
        assert_eq!(p.original_size, (50, 30));
    }

    #[test]
    fn view_tensor_layout_is_channel_major() {
        // Distinct per-channel values so we can prove channel-major ordering.
        let mut img = RgbImage::new(2, 1);
        img.put_pixel(0, 0, Rgb([0, 128, 255]));
        img.put_pixel(1, 0, Rgb([255, 128, 0]));
        let vt = view_tensor(&DynamicImage::ImageRgb8(img));
        // rows=3 channels, cols=2 spatial. Channel 0 (R): [px0=-1, px1=+1].
        assert!((vt.pixels.get(0, 0) - (-1.0)).abs() < 1e-6);
        assert!((vt.pixels.get(0, 1) - 1.0).abs() < 1e-6);
        // Channel 2 (B): [px0=+1, px1=-1].
        assert!((vt.pixels.get(2, 0) - 1.0).abs() < 1e-6);
        assert!((vt.pixels.get(2, 1) - (-1.0)).abs() < 1e-6);
    }

    // ── SmolVLM2 preprocess (C7) ─────────────────────────────────────────────

    /// Layout math across aspect ratios (spec §6 / OQ-3): landscape,
    /// portrait, square, and the odd-derived-edge `+1` bump.
    #[test]
    fn smolvlm2_layout_across_aspects() {
        let mk = |w, h| DynamicImage::ImageRgb8(RgbImage::new(w, h));
        // 1024×768 → 2048×1536 → 2048×1536 (multiples) → 3 rows × 4 cols + 1.
        let p = preprocess_smolvlm2(&mk(1024, 768)).unwrap();
        assert_eq!((p.rows, p.cols, p.n_frames), (3, 4, 13));
        assert_eq!(p.frames.len(), 13 * 3 * 512 * 512);
        // Portrait mirror: 768×1024 → 4 rows × 3 cols.
        let p = preprocess_smolvlm2(&mk(768, 1024)).unwrap();
        assert_eq!((p.rows, p.cols, p.n_frames), (4, 3, 13));
        // Square: 640×640 → 2048×2048 → 4×4 + 1 = 17 frames (the cap shape).
        let p = preprocess_smolvlm2(&mk(640, 640)).unwrap();
        assert_eq!((p.rows, p.cols, p.n_frames), (4, 4, 17));
        // Odd derived edge: 999×500 → aspect 1.998; h=int(2048/1.998)=1025
        // → odd → 1026 → ceil512 → 1536 → 3 rows.
        let p = preprocess_smolvlm2(&mk(999, 500)).unwrap();
        assert_eq!((p.rows, p.cols), (3, 4));
        // Tiny image upscales (long side ALWAYS → 2048; still splits).
        let p = preprocess_smolvlm2(&mk(10, 10)).unwrap();
        assert_eq!((p.rows, p.cols, p.n_frames), (4, 4, 17));
    }

    /// The normalize rail: a mid-gray 128 maps to (128/255 - 0.5)/0.5 and a
    /// solid image survives every resize (solid-color invariance), so every
    /// frame is exactly that constant.
    #[test]
    fn smolvlm2_normalize_rail() {
        let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(800, 600, Rgb([128, 0, 255])));
        let p = preprocess_smolvlm2(&img).unwrap();
        let want_r = ((128.0f64 * (1.0 / 255.0)) as f32 - 0.5) / 0.5;
        let n = 512 * 512;
        for f in 0..p.n_frames {
            let fr = &p.frames[f * 3 * n..(f + 1) * 3 * n];
            assert!((fr[0] - want_r).abs() < 1e-7, "frame {f} R rail");
            assert!((fr[n] - (-1.0)).abs() < 1e-7, "frame {f} G rail");
            assert!((fr[2 * n] - 1.0).abs() < 1e-7, "frame {f} B rail");
        }
    }

    #[test]
    fn smolvlm2_degenerate_image_errors() {
        // A zero-dimension image cannot come from a decoder, but the guard
        // must fail loud, not panic in the resampler.
        let img = DynamicImage::ImageRgb8(RgbImage::new(0, 5));
        assert!(preprocess_smolvlm2(&img).is_err());
    }

    /// **C7 L0b — preprocess EXACT vs the torch oracle** (skip-with-SUCCESS
    /// without `FOCR_SMOLVLM2_DIR`): our LANCZOS+split+normalize pipeline on
    /// the committed sample photo must reproduce the oracle's
    /// `pixel_values.bin` — the resample is Pillow-bit-exact by construction,
    /// so the only allowed drift is the final f32 normalize ULP.
    #[test]
    fn smolvlm2_preprocess_matches_torch_oracle() {
        let Ok(dir) = std::env::var("FOCR_SMOLVLM2_DIR") else {
            return;
        };
        let pv_path = format!("{dir}/smolvlm2_pixel_values.bin");
        if !std::path::Path::new(&pv_path).is_file() {
            eprintln!("skip-with-SUCCESS: {pv_path} absent (run the vision oracle script)");
            return;
        }
        let want: Vec<f32> = std::fs::read(&pv_path)
            .expect("oracle blob reads")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let photo = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/smolvlm2/sample_photo.png"
        );
        let p = preprocess_smolvlm2_path(std::path::Path::new(photo)).expect("preprocess");
        assert_eq!((p.rows, p.cols, p.n_frames), (3, 4, 13), "tile layout");
        assert_eq!(p.frames.len(), want.len(), "frame count/shape");
        let mut max_abs = 0.0f32;
        let mut n_diff = 0usize;
        for (a, b) in p.frames.iter().zip(&want) {
            let d = (a - b).abs();
            if d > 0.0 {
                n_diff += 1;
            }
            max_abs = max_abs.max(d);
        }
        eprintln!(
            "[C7 L0b] maxabs={max_abs:.3e} n_diff={n_diff}/{}",
            want.len()
        );
        assert!(
            max_abs <= 1e-6,
            "preprocess maxabs {max_abs:.3e} > 1e-6 — the resample or normalize drifted"
        );
    }
}
