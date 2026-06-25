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
//! census) is exact.

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
        self.width_crop_num.saturating_mul(self.height_crop_num)
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
            .saturating_add(1)
            .saturating_mul(q_base)
            .saturating_add(1);
        if let PreprocessMode::Gundam { tile_size, .. } = self.mode
            && self.crop_grid.is_tiled()
        {
            let q_local = num_queries(tile_size);
            let w = self.crop_grid.width_crop_num;
            let h = self.crop_grid.height_crop_num;
            // Local: (q_local*W patches + 1 newline) per (q_local*H) rows.
            let local_cols = q_local.saturating_mul(w).saturating_add(1);
            let local_rows = q_local.saturating_mul(h);
            total = total.saturating_add(local_cols.saturating_mul(local_rows));
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
/// without touching the filesystem ([SPEC-022..031]).
fn preprocess_dynamic(img: DynamicImage, mode: PreprocessMode) -> FocrResult<Preprocessed> {
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
fn decode_path(path: &Path) -> FocrResult<DynamicImage> {
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
    let orientation = decoder.orientation()?;
    let mut img = DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    Ok(img)
}

// ── global view: aspect-preserving resize + gray pad ([SPEC-022]) ───────────

/// `ImageOps.pad`: resize aspect-preserving to fit inside `size × size`, then
/// center-pad the short axis with the mean gray color `(127,127,127)`
/// ([SPEC-022], `modeling_unlimitedocr.py:872-873`).
///
/// PIL `ImageOps.pad` uses `BICUBIC` resampling by default; we route to
/// `CatmullRom` (the crate's cubic filter) — the closest available kernel. The
/// pad geometry (fit + centered placement + gray fill) is exact.
fn pad_to_square(img: &DynamicImage, size: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    // Aspect-preserving fit: scale so the longer side == size.
    // (w, h) -> (rw, rh) with max(rw, rh) == size, preserving ratio, >= 1px.
    let (rw, rh) = if w == 0 || h == 0 {
        (size, size)
    } else if w >= h {
        let rh = ((u64::from(h) * u64::from(size)) / u64::from(w)).max(1) as u32;
        (size, rh)
    } else {
        let rw = ((u64::from(w) * u64::from(size)) / u64::from(h)).max(1) as u32;
        (rw, size)
    };
    let resized = img.resize_exact(rw, rh, FilterType::CatmullRom).to_rgb8();

    // Center on a gray canvas.
    let mut canvas = image::RgbImage::from_pixel(size, size, image::Rgb([PAD_FILL; 3]));
    let ox = (size - rw) / 2;
    let oy = (size - rh) / 2;
    for y in 0..rh {
        for x in 0..rw {
            let p = *resized.get_pixel(x, y);
            canvas.put_pixel(ox + x, oy + y, p);
        }
    }
    DynamicImage::ImageRgb8(canvas)
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
    // PIL `image.resize((W,H))` default resample is BICUBIC; CatmullRom is the
    // closest crate cubic. Tile geometry (crop boxes) is exact.
    let resized = img.resize_exact(target_w, target_h, FilterType::CatmullRom);

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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

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
        // (127/255 -> 2*(127/255)-1). 100x40 -> fit width 64 => rh = 40*64/100 =
        // 25, centered with oy = (64-25)/2 = 19, so row 0 is pad.
        let gray = 2.0 * (f32::from(PAD_FILL) / 255.0) - 1.0;
        // s for (x=0, y=0) is 0; channel 0.
        assert!((p.global.pixels.get(0, 0) - gray).abs() < 1e-6);
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
    fn placeholder_count_saturates_for_malformed_grid() {
        let grid = CropGrid {
            width_crop_num: usize::MAX,
            height_crop_num: usize::MAX,
        };
        assert_eq!(grid.blocks(), usize::MAX);

        let p = Preprocessed {
            mode: PreprocessMode::gundam(),
            global: view_tensor(&solid(4, 4, [0, 0, 0])),
            tiles: Vec::new(),
            crop_grid: grid,
            original_size: (4, 4),
        };
        assert_eq!(p.placeholder_token_count(), usize::MAX);
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
}
