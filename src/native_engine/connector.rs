//! Connector: image_newline / view_seperator + masked_scatter
//! ([SPEC-060..066], PROPOSED_ARCHITECTURE.md §6.6).
//!
//! This module performs the *structural fusion* that turns a flat grid of
//! per-patch vision embeddings into the exact token stream the decoder expects,
//! then scatters that stream into the decoder input-embedding matrix at the
//! `<image>` placeholder positions. There is **no resampler / Q-Former** — the
//! vision features go straight into the text embedding rail (true end-to-end
//! fusion, [SPEC-064/065]).
//!
//! Two learned structural parameters live here (both `nn.Parameter(randn(1280) *
//! 1/sqrt(1280))`, [SPEC-060]):
//!
//! * `model.image_newline` — appended **once per grid row** as a trailing
//!   column, so a `16×16` global feature grid becomes `16` rows of `17`
//!   (`16 image-feature + 1 newline`) = `272` tokens.
//! * `model.view_seperator` — appended **once** at the per-image trailing token.
//!
//! For a base 1024 global view this yields exactly **`256 + 16 + 1 = 273`**
//! slots (OQ-18 / CENSUS.md §(c)). The crop ("Gundam") branch prepends a
//! `local` block ahead of the global block; the final per-image feature order is
//! `[local, global, view_seperator]` ([SPEC-062]) — and the token-side
//! `images_seq_mask` layout MUST match it ([SPEC-066 ORDERING INVARIANT]) so
//! `masked_scatter` aligns row-for-row.

use super::tensor::Mat;
use super::weights::Weights;
use crate::error::{FocrError, FocrResult};

/// Vision embedding dim (`n_embed`) — the connector currency ([SPEC-060]).
pub const N_EMBED: usize = 1280;

fn checked_add(context: &str, lhs: usize, rhs: usize, expression: &str) -> FocrResult<usize> {
    lhs.checked_add(rhs).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "{context}: usize overflow computing {expression} ({lhs} + {rhs})"
        ))
    })
}

fn checked_mul(context: &str, lhs: usize, rhs: usize, expression: &str) -> FocrResult<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        FocrError::Other(anyhow::anyhow!(
            "{context}: usize overflow computing {expression} ({lhs} * {rhs})"
        ))
    })
}

fn zeros_checked(context: &str, rows: usize, cols: usize) -> FocrResult<Mat> {
    let len = checked_mul(context, rows, cols, "rows*cols")?;
    let mut data = Vec::new();
    data.try_reserve_exact(len).map_err(|err| {
        FocrError::Other(anyhow::anyhow!(
            "{context}: could not allocate matrix [{rows}, {cols}] ({len} f32 values): {err}"
        ))
    })?;
    data.resize(len, 0.0);
    Ok(Mat { rows, cols, data })
}

fn validate_mat_len(context: &str, mat: &Mat) -> FocrResult<()> {
    let expected = checked_mul(context, mat.rows, mat.cols, "rows*cols")?;
    if mat.data.len() != expected {
        return Err(FocrError::Other(anyhow::anyhow!(
            "{context}: data len {} != rows*cols {} for shape [{}, {}]",
            mat.data.len(),
            expected,
            mat.rows,
            mat.cols
        )));
    }
    Ok(())
}

/// Append `newline` (length `dim`) as one extra trailing column to every row of
/// a `(h, w, dim)` grid laid out row-major as `[h*w, dim]`.
///
/// The grid is interpreted as `h` rows of `w` patch embeddings (each `dim`
/// wide); after this op each row holds `w + 1` embeddings — the trailing one is
/// `image_newline` ([SPEC-062]: `cat([grid, image_newline.expand(h,1,dim)],
/// dim=1)`). The result is flattened back to `[h*(w+1), dim]` in row-major
/// `(row, col)` order, which is exactly the post-`view(-1, n_dim)` layout.
///
/// # Errors
/// Returns [`FocrError::Other`] if `grid` is not `[h*w, dim]` or `newline`'s
/// length isn't `dim`.
fn append_newline_column(grid: &Mat, h: usize, w: usize, newline: &[f32]) -> FocrResult<Mat> {
    let dim = grid.cols;
    let expected_rows = checked_mul("append_newline_column", h, w, "h*w")?;
    if grid.rows != expected_rows {
        return Err(FocrError::Other(anyhow::anyhow!(
            "append_newline_column: grid rows {} != h*w {}",
            grid.rows,
            expected_rows
        )));
    }
    if newline.len() != dim {
        return Err(FocrError::Other(anyhow::anyhow!(
            "append_newline_column: newline len {} != dim {}",
            newline.len(),
            dim
        )));
    }
    let out_width = checked_add("append_newline_column", w, 1, "w+1")?;
    let out_rows = checked_mul("append_newline_column", h, out_width, "h*(w+1)")?;
    validate_mat_len("append_newline_column grid", grid)?;
    let mut out = zeros_checked("append_newline_column", out_rows, dim)?;
    for r in 0..h {
        // Copy the w real patch embeddings for this row.
        for c in 0..w {
            let src = grid.row(r * w + c);
            let dst_row = r * out_width + c;
            out.row_mut(dst_row).copy_from_slice(src);
        }
        // Trailing newline column.
        let nl_row = r * out_width + w;
        out.row_mut(nl_row).copy_from_slice(newline);
    }
    Ok(out)
}

/// Stack a list of `[*, dim]` blocks vertically into one `[sum_rows, dim]`
/// matrix (the `torch.cat(..., dim=0)` of the connector).
///
/// # Errors
/// Returns [`FocrError::Other`] if the blocks disagree on `dim`.
fn vstack(blocks: &[&Mat], dim: usize) -> FocrResult<Mat> {
    let mut total_rows = 0usize;
    for b in blocks {
        if b.cols != dim {
            return Err(FocrError::Other(anyhow::anyhow!(
                "vstack: block cols {} != dim {}",
                b.cols,
                dim
            )));
        }
        total_rows = checked_add("vstack", total_rows, b.rows, "sum_rows")?;
    }

    for b in blocks {
        validate_mat_len("vstack block", b)?;
    }

    let mut out = zeros_checked("vstack", total_rows, dim)?;
    let mut cursor = 0usize;
    for b in blocks {
        let n = checked_mul("vstack", b.rows, dim, "block_rows*dim")?;
        let start = checked_mul("vstack", cursor, dim, "cursor*dim")?;
        let end = checked_add("vstack", start, n, "copy range end")?;
        out.data[start..end].copy_from_slice(&b.data);
        cursor = checked_add("vstack", cursor, b.rows, "cursor+block_rows")?;
    }
    Ok(out)
}

/// Build the per-image vision token block for the **no-crop / single global**
/// branch ([SPEC-063], OQ-18): a single `(h, w)` global feature grid with a
/// per-row `image_newline` column, then a trailing `view_seperator`.
///
/// `global` is `[h*w, dim]` (hybrid CLIP+SAM features after the projector),
/// `image_newline` / `view_seperator` are length-`dim` learned params. At base
/// 1024 (`h=w=16`) this produces exactly `16*(16+1) + 1 = 273` rows ([SPEC-066],
/// CENSUS §(c)).
///
/// # Errors
/// Returns [`FocrError::Other`] on a shape/length mismatch.
pub fn assemble_global_block(
    global: &Mat,
    h: usize,
    w: usize,
    image_newline: &[f32],
    view_seperator: &[f32],
) -> FocrResult<Mat> {
    let dim = global.cols;
    if view_seperator.len() != dim {
        return Err(FocrError::Other(anyhow::anyhow!(
            "assemble_global_block: view_seperator len {} != dim {}",
            view_seperator.len(),
            dim
        )));
    }
    let with_nl = append_newline_column(global, h, w, image_newline)?; // [h*(w+1), dim]
    let sep = Mat::from_vec(1, dim, view_seperator.to_vec());
    // Order per [SPEC-063]: [global_features, view_seperator].
    vstack(&[&with_nl, &sep], dim)
}

/// Build the per-image vision token block for the **crop ("Gundam") branch**
/// ([SPEC-062]).
///
/// The feature order is the ORDERING INVARIANT `[local, global,
/// view_seperator]` ([SPEC-066]):
/// * `local` — the tiled local features, already spatially rearranged by the
///   caller to `[h2_total * w2_total, dim]` (the
///   `permute(0,2,1,3,4).reshape(...)` of [SPEC-062]); we append the per-row
///   `image_newline` column over its `h_local` rows.
/// * `global` — the `(h, w)` global grid with its own per-row `image_newline`
///   column.
/// * a single trailing `view_seperator`.
///
/// `h_local`/`w_local` are the local grid's *post-rearrange* row/col counts
/// (e.g. `height_crop_num*10` × `width_crop_num*10`). `h`/`w` are the global
/// grid (16×16 at base 1024).
///
/// # Errors
/// Returns [`FocrError::Other`] on a shape/length mismatch.
#[allow(clippy::too_many_arguments)]
pub fn assemble_crop_block(
    local: &Mat,
    h_local: usize,
    w_local: usize,
    global: &Mat,
    h: usize,
    w: usize,
    image_newline: &[f32],
    view_seperator: &[f32],
) -> FocrResult<Mat> {
    let dim = global.cols;
    if local.cols != dim {
        return Err(FocrError::Other(anyhow::anyhow!(
            "assemble_crop_block: local cols {} != global cols {}",
            local.cols,
            dim
        )));
    }
    if view_seperator.len() != dim {
        return Err(FocrError::Other(anyhow::anyhow!(
            "assemble_crop_block: view_seperator len {} != dim {}",
            view_seperator.len(),
            dim
        )));
    }
    let local_nl = append_newline_column(local, h_local, w_local, image_newline)?;
    let global_nl = append_newline_column(global, h, w, image_newline)?;
    let sep = Mat::from_vec(1, dim, view_seperator.to_vec());
    // Order per [SPEC-062/066]: [local, global, view_seperator].
    vstack(&[&local_nl, &global_nl, &sep], dim)
}

/// Rearrange row-major local tile feature grids into the one large local grid
/// used by the crop connector branch.
///
/// Mirrors the pinned source:
/// `local.view(height_crop_num, width_crop_num, h2, w2, dim)
///       .permute(0, 2, 1, 3, 4)
///       .reshape(height_crop_num*h2, width_crop_num*w2, dim)`.
///
/// `tiles` are row-major over the crop grid, and each tile is `[tile_h*tile_w,
/// dim]` in row-major patch order.
fn rearrange_local_tiles(
    tiles: &[Mat],
    width_crop_num: usize,
    height_crop_num: usize,
    tile_h: usize,
    tile_w: usize,
) -> FocrResult<Mat> {
    let expected_tiles = checked_mul(
        "rearrange_local_tiles",
        width_crop_num,
        height_crop_num,
        "width_crop_num*height_crop_num",
    )?;
    if tiles.len() != expected_tiles {
        return Err(FocrError::Other(anyhow::anyhow!(
            "rearrange_local_tiles: {} local tile feature blocks != width_crop_num*height_crop_num {}",
            tiles.len(),
            expected_tiles
        )));
    }
    let Some(first) = tiles.first() else {
        return Err(FocrError::Other(anyhow::anyhow!(
            "rearrange_local_tiles: crop branch needs at least one local tile feature block"
        )));
    };
    let dim = first.cols;
    let expected_tile_rows = checked_mul("rearrange_local_tiles", tile_h, tile_w, "tile_h*tile_w")?;
    for tile in tiles {
        if tile.cols != dim {
            return Err(FocrError::Other(anyhow::anyhow!(
                "rearrange_local_tiles: tile cols {} != dim {}",
                tile.cols,
                dim
            )));
        }
        if tile.rows != expected_tile_rows {
            return Err(FocrError::Other(anyhow::anyhow!(
                "rearrange_local_tiles: tile rows {} != tile_h*tile_w {}",
                tile.rows,
                expected_tile_rows
            )));
        }
        validate_mat_len("rearrange_local_tiles tile", tile)?;
    }

    let out_h = checked_mul(
        "rearrange_local_tiles",
        height_crop_num,
        tile_h,
        "height_crop_num*tile_h",
    )?;
    let out_w = checked_mul(
        "rearrange_local_tiles",
        width_crop_num,
        tile_w,
        "width_crop_num*tile_w",
    )?;
    let mut out = zeros_checked(
        "rearrange_local_tiles",
        checked_mul("rearrange_local_tiles", out_h, out_w, "out_h*out_w")?,
        dim,
    )?;

    for tile_row in 0..height_crop_num {
        for tile_col in 0..width_crop_num {
            let tile = &tiles[tile_row * width_crop_num + tile_col];
            for local_y in 0..tile_h {
                for local_x in 0..tile_w {
                    let src_row = local_y * tile_w + local_x;
                    let dst_row =
                        (tile_row * tile_h + local_y) * out_w + tile_col * tile_w + local_x;
                    out.row_mut(dst_row).copy_from_slice(tile.row(src_row));
                }
            }
        }
    }
    Ok(out)
}

/// Scatter the per-image vision feature rows into the text embedding stream at
/// the `<image>` placeholder positions ([SPEC-064]).
///
/// Mirrors `inputs_embeds[idx].masked_scatter_(images_seq_mask[idx]
/// .unsqueeze(-1), vision_features)`: each row of `inputs_embeds` whose mask bit
/// is `true` is overwritten, **in order**, with the next row of
/// `vision_features`. The number of `true` mask positions MUST equal
/// `vision_features.rows` (the ORDERING INVARIANT, [SPEC-066]).
///
/// `inputs_embeds` is `[seq_len, dim]` (the decoder `embed_tokens(input_ids)`
/// output, [SPEC-065]); `vision_features` is `[num_vision_tokens, dim]` (the
/// concatenated per-image blocks from [`assemble_global_block`] /
/// [`assemble_crop_block`]); `images_seq_mask` has length `seq_len`.
///
/// # Errors
/// Returns [`FocrError::Other`] if dims disagree, the mask length isn't
/// `seq_len`, or the `true` count doesn't match `vision_features.rows`.
pub fn masked_scatter(
    inputs_embeds: &mut Mat,
    vision_features: &Mat,
    images_seq_mask: &[bool],
) -> FocrResult<()> {
    let dim = inputs_embeds.cols;
    if vision_features.cols != dim {
        return Err(FocrError::Other(anyhow::anyhow!(
            "masked_scatter: vision_features cols {} != inputs_embeds cols {}",
            vision_features.cols,
            dim
        )));
    }
    if images_seq_mask.len() != inputs_embeds.rows {
        return Err(FocrError::Other(anyhow::anyhow!(
            "masked_scatter: mask len {} != inputs_embeds rows {}",
            images_seq_mask.len(),
            inputs_embeds.rows
        )));
    }
    let n_true = images_seq_mask.iter().filter(|&&b| b).count();
    if n_true != vision_features.rows {
        return Err(FocrError::Other(anyhow::anyhow!(
            "masked_scatter: {} masked positions != {} vision feature rows \
             (ORDERING INVARIANT [SPEC-066])",
            n_true,
            vision_features.rows
        )));
    }
    validate_mat_len("masked_scatter inputs_embeds", inputs_embeds)?;
    validate_mat_len("masked_scatter vision_features", vision_features)?;
    let mut feat = 0usize;
    for (row, &masked) in images_seq_mask.iter().enumerate() {
        if masked {
            let src = vision_features.row(feat);
            inputs_embeds.row_mut(row).copy_from_slice(src);
            feat += 1;
        }
    }
    Ok(())
}

/// Full connector entrypoint for the **no-crop** path: assemble the 273-slot
/// global block (per image), concatenate across images, and scatter into the
/// decoder embeddings.
///
/// `globals` are the per-image hybrid feature grids (each `[h*w, dim]`), in the
/// order their placeholders appear in `images_seq_mask`. `inputs_embeds` is
/// mutated in place ([SPEC-064/065]). The learned `image_newline` /
/// `view_seperator` params are passed explicitly (the `.focrq` index carries
/// them as the bare tensors `model.image_newline` / `model.view_seperator`,
/// CENSUS §(b)); `_weights` is reserved for the loaded-index handle once
/// `Weights` lands so call sites needn't thread the raw slices.
///
/// # Errors
/// Returns [`FocrError::Other`] on any shape/length/ordering mismatch.
#[allow(clippy::too_many_arguments)]
pub fn fuse_no_crop(
    _weights: &Weights,
    inputs_embeds: &mut Mat,
    globals: &[Mat],
    h: usize,
    w: usize,
    image_newline: &[f32],
    view_seperator: &[f32],
    images_seq_mask: &[bool],
) -> FocrResult<()> {
    let dim = inputs_embeds.cols;
    let mut blocks: Vec<Mat> = Vec::with_capacity(globals.len());
    for g in globals {
        blocks.push(assemble_global_block(
            g,
            h,
            w,
            image_newline,
            view_seperator,
        )?);
    }
    let refs: Vec<&Mat> = blocks.iter().collect();
    let features = vstack(&refs, dim)?;
    masked_scatter(inputs_embeds, &features, images_seq_mask)
}

/// Full connector entrypoint for the **crop / Gundam** path: rearrange the
/// local tile feature grids into the reference spatial layout, assemble
/// `[local, global, view_seperator]`, and scatter into the decoder embeddings.
///
/// `local_tiles` are row-major over crop rows, then crop columns, with
/// `width_crop_num` tiles per row; each tile is `[tile_h*tile_w, dim]`.
/// `global` is `[h*w, dim]`.
///
/// # Errors
/// Returns [`FocrError::Other`] on any tile-count, shape, dimension, or
/// placeholder-count mismatch.
#[allow(clippy::too_many_arguments)]
pub fn fuse_crop(
    _weights: &Weights,
    inputs_embeds: &mut Mat,
    local_tiles: &[Mat],
    width_crop_num: usize,
    height_crop_num: usize,
    tile_h: usize,
    tile_w: usize,
    global: &Mat,
    h: usize,
    w: usize,
    image_newline: &[f32],
    view_seperator: &[f32],
    images_seq_mask: &[bool],
) -> FocrResult<()> {
    let local =
        rearrange_local_tiles(local_tiles, width_crop_num, height_crop_num, tile_h, tile_w)?;
    let h_local = checked_mul(
        "fuse_crop",
        height_crop_num,
        tile_h,
        "height_crop_num*tile_h",
    )?;
    let w_local = checked_mul("fuse_crop", width_crop_num, tile_w, "width_crop_num*tile_w")?;
    let features = assemble_crop_block(
        &local,
        h_local,
        w_local,
        global,
        h,
        w,
        image_newline,
        view_seperator,
    )?;
    masked_scatter(inputs_embeds, &features, images_seq_mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distinct per-row sentinel values so we can assert ordering precisely.
    fn grid(h: usize, w: usize, dim: usize, base: f32) -> Mat {
        let mut m = Mat::zeros(h * w, dim);
        for r in 0..h * w {
            for c in 0..dim {
                m.set(r, c, base + r as f32 + 0.001 * c as f32);
            }
        }
        m
    }

    #[test]
    fn append_newline_inserts_one_trailing_column_per_row() {
        // 2x3 grid, dim=2. Output is 2*(3+1)=8 rows; col index 3,7 are newline.
        let g = grid(2, 3, 2, 10.0);
        let nl = vec![-1.0, -2.0];
        let out = append_newline_column(&g, 2, 3, &nl).unwrap();
        assert_eq!(out.shape(), (8, 2));
        // Row 0..2 are the 3 real patches of grid row 0, row 3 is the newline.
        assert_eq!(out.row(0), g.row(0));
        assert_eq!(out.row(1), g.row(1));
        assert_eq!(out.row(2), g.row(2));
        assert_eq!(out.row(3), &nl[..]);
        // Row 4..6 are grid row 1's patches (orig rows 3,4,5), row 7 newline.
        assert_eq!(out.row(4), g.row(3));
        assert_eq!(out.row(5), g.row(4));
        assert_eq!(out.row(6), g.row(5));
        assert_eq!(out.row(7), &nl[..]);
    }

    #[test]
    fn append_newline_rejects_bad_grid_shape() {
        let g = Mat::zeros(5, 2); // 5 != 2*3
        assert!(append_newline_column(&g, 2, 3, &[0.0, 0.0]).is_err());
    }

    #[test]
    fn append_newline_rejects_geometry_overflow_without_allocating() {
        let g = Mat::zeros(0, 1);
        assert!(matches!(
            append_newline_column(&g, usize::MAX, 2, &[0.0]),
            Err(err) if err.to_string().contains("overflow")
        ));
    }

    #[test]
    fn append_newline_rejects_output_width_overflow_without_allocating() {
        let g = Mat::zeros(0, 1);
        assert!(matches!(
            append_newline_column(&g, 0, usize::MAX, &[0.0]),
            Err(err) if err.to_string().contains("w+1")
        ));
    }

    #[test]
    fn append_newline_rejects_output_rows_overflow_without_allocating() {
        let h = usize::MAX / 2 + 1;
        let g = Mat {
            rows: h,
            cols: 1,
            data: Vec::new(),
        };
        assert!(matches!(
            append_newline_column(&g, h, 1, &[0.0]),
            Err(err) if err.to_string().contains("h*(w+1)")
        ));
    }

    #[test]
    fn append_newline_rejects_malformed_grid_data() {
        let g = Mat {
            rows: 4,
            cols: 2,
            data: vec![1.0; 7],
        };
        assert!(matches!(
            append_newline_column(&g, 2, 2, &[0.0, 0.0]),
            Err(err) if err.to_string().contains("data len 7 != rows*cols 8")
        ));
    }

    #[test]
    fn vstack_rejects_total_rows_overflow_without_allocating() {
        let huge = Mat {
            rows: usize::MAX,
            cols: 1,
            data: Vec::new(),
        };
        let one = Mat {
            rows: 1,
            cols: 1,
            data: Vec::new(),
        };
        assert!(matches!(
            vstack(&[&huge, &one], 1),
            Err(err) if err.to_string().contains("sum_rows")
        ));
    }

    #[test]
    fn vstack_rejects_element_count_overflow_without_allocating() {
        let huge = Mat {
            rows: usize::MAX,
            cols: 2,
            data: Vec::new(),
        };
        assert!(matches!(
            vstack(&[&huge], 2),
            Err(err) if err.to_string().contains("rows*cols")
        ));
    }

    #[test]
    fn vstack_rejects_malformed_block_data() {
        let malformed = Mat {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0],
        };
        assert!(matches!(
            vstack(&[&malformed], 2),
            Err(err) if err.to_string().contains("data len 3 != rows*cols 4")
        ));
    }

    /// The base-1024 invariant: a 16x16 hybrid grid + per-row newline + 1
    /// separator == exactly 273 slots (OQ-18 / CENSUS §(c)).
    #[test]
    fn assemble_global_block_is_273_at_base_1024() {
        let g = grid(16, 16, N_EMBED, 0.0);
        let nl = vec![7.0; N_EMBED];
        let sep = vec![9.0; N_EMBED];
        let block = assemble_global_block(&g, 16, 16, &nl, &sep).unwrap();
        assert_eq!(block.shape(), (273, N_EMBED));
        // Last row is the view_seperator.
        assert_eq!(block.row(272), &sep[..]);
        // The 17th token of row 0 (index 16) is the first newline.
        assert_eq!(block.row(16), &nl[..]);
        // 256 features + 16 newlines = 272 before the separator.
        let newline_count = (0..272).filter(|&r| block.row(r) == nl.as_slice()).count();
        assert_eq!(newline_count, 16);
    }

    #[test]
    fn assemble_global_block_small_geometry() {
        // h=w=2, dim=3: (2+1)*2 + 1 = 7 rows.
        let g = grid(2, 2, 3, 100.0);
        let nl = vec![-5.0, -5.0, -5.0];
        let sep = vec![-9.0, -9.0, -9.0];
        let block = assemble_global_block(&g, 2, 2, &nl, &sep).unwrap();
        assert_eq!(block.shape(), (7, 3));
        // Layout: [p00,p01,nl, p10,p11,nl, sep]
        assert_eq!(block.row(0), g.row(0));
        assert_eq!(block.row(1), g.row(1));
        assert_eq!(block.row(2), &nl[..]);
        assert_eq!(block.row(3), g.row(2));
        assert_eq!(block.row(4), g.row(3));
        assert_eq!(block.row(5), &nl[..]);
        assert_eq!(block.row(6), &sep[..]);
    }

    /// Crop branch ordering invariant: [local, global, view_seperator].
    #[test]
    fn assemble_crop_block_orders_local_then_global_then_sep() {
        // local 1x2 grid, global 1x2 grid, dim=2.
        let local = grid(1, 2, 2, 50.0);
        let global = grid(1, 2, 2, 80.0);
        let nl = vec![-1.0, -1.0];
        let sep = vec![-2.0, -2.0];
        let block = assemble_crop_block(&local, 1, 2, &global, 1, 2, &nl, &sep).unwrap();
        // local: 1*(2+1)=3, global: 1*(2+1)=3, sep: 1 => 7 rows.
        assert_eq!(block.shape(), (7, 2));
        // local block first.
        assert_eq!(block.row(0), local.row(0));
        assert_eq!(block.row(1), local.row(1));
        assert_eq!(block.row(2), &nl[..]);
        // then global block.
        assert_eq!(block.row(3), global.row(0));
        assert_eq!(block.row(4), global.row(1));
        assert_eq!(block.row(5), &nl[..]);
        // separator last.
        assert_eq!(block.row(6), &sep[..]);
    }

    #[test]
    fn rearrange_local_tiles_matches_reference_permute_layout() {
        let tiles = vec![
            Mat::from_vec(4, 1, vec![0.0, 1.0, 2.0, 3.0]),
            Mat::from_vec(4, 1, vec![10.0, 11.0, 12.0, 13.0]),
            Mat::from_vec(4, 1, vec![20.0, 21.0, 22.0, 23.0]),
            Mat::from_vec(4, 1, vec![30.0, 31.0, 32.0, 33.0]),
        ];
        let local = rearrange_local_tiles(&tiles, 2, 2, 2, 2).unwrap();
        assert_eq!(local.shape(), (16, 1));
        assert_eq!(
            local.data,
            vec![
                0.0, 1.0, 10.0, 11.0, 2.0, 3.0, 12.0, 13.0, 20.0, 21.0, 30.0, 31.0, 22.0, 23.0,
                32.0, 33.0,
            ]
        );
    }

    #[test]
    fn masked_scatter_overwrites_true_positions_in_order() {
        // 5-token text stream, dim=2; mask True at positions 1,2,4 -> 3 rows.
        let mut embeds = Mat::from_vec(
            5,
            2,
            vec![
                0.0, 0.0, // pos0 text
                1.0, 1.0, // pos1 placeholder
                2.0, 2.0, // pos2 placeholder
                3.0, 3.0, // pos3 text
                4.0, 4.0, // pos4 placeholder
            ],
        );
        let feats = Mat::from_vec(3, 2, vec![10.0, 11.0, 20.0, 21.0, 40.0, 41.0]);
        let mask = vec![false, true, true, false, true];
        masked_scatter(&mut embeds, &feats, &mask).unwrap();
        assert_eq!(embeds.row(0), &[0.0, 0.0]); // untouched
        assert_eq!(embeds.row(1), &[10.0, 11.0]); // feat row 0
        assert_eq!(embeds.row(2), &[20.0, 21.0]); // feat row 1
        assert_eq!(embeds.row(3), &[3.0, 3.0]); // untouched
        assert_eq!(embeds.row(4), &[40.0, 41.0]); // feat row 2
    }

    #[test]
    fn masked_scatter_rejects_count_mismatch() {
        let mut embeds = Mat::zeros(3, 2);
        let feats = Mat::zeros(2, 2); // 2 rows
        let mask = vec![true, false, false]; // only 1 True
        let err = masked_scatter(&mut embeds, &feats, &mask);
        assert!(err.is_err());
    }

    #[test]
    fn masked_scatter_rejects_dim_mismatch() {
        let mut embeds = Mat::zeros(2, 4);
        let feats = Mat::zeros(1, 2); // wrong dim
        let mask = vec![true, false];
        assert!(masked_scatter(&mut embeds, &feats, &mask).is_err());
    }

    #[test]
    fn masked_scatter_rejects_bad_mask_len() {
        let mut embeds = Mat::zeros(3, 2);
        let feats = Mat::zeros(1, 2);
        let mask = vec![true, false]; // len 2 != 3
        assert!(masked_scatter(&mut embeds, &feats, &mask).is_err());
    }

    #[test]
    fn masked_scatter_rejects_malformed_inputs_data() {
        let mut embeds = Mat {
            rows: 2,
            cols: 2,
            data: vec![0.0; 3],
        };
        let feats = Mat::zeros(1, 2);
        let mask = vec![true, false];
        assert!(matches!(
            masked_scatter(&mut embeds, &feats, &mask),
            Err(err) if err.to_string().contains("masked_scatter inputs_embeds")
        ));
    }

    #[test]
    fn masked_scatter_rejects_malformed_vision_data() {
        let mut embeds = Mat::zeros(2, 2);
        let feats = Mat {
            rows: 1,
            cols: 2,
            data: vec![1.0],
        };
        let mask = vec![true, false];
        assert!(matches!(
            masked_scatter(&mut embeds, &feats, &mask),
            Err(err) if err.to_string().contains("masked_scatter vision_features")
        ));
    }

    /// End-to-end no-crop fuse: a tiny 2x2 global view (7-slot block) scattered
    /// into a text stream, verifying the full assemble + scatter path and that
    /// non-placeholder text rows survive.
    #[test]
    fn fuse_no_crop_end_to_end() {
        let weights = Weights::default();
        let dim = 3;
        let g = grid(2, 2, dim, 100.0);
        let nl = vec![-5.0, -5.0, -5.0];
        let sep = vec![-9.0, -9.0, -9.0];
        // 9-token text stream: [BOS, 7 image placeholders, EOS].
        let mut embeds = Mat::zeros(9, dim);
        // mark text tokens so we can assert they survive.
        embeds.row_mut(0).copy_from_slice(&[1.0, 1.0, 1.0]);
        embeds.row_mut(8).copy_from_slice(&[2.0, 2.0, 2.0]);
        let mut mask = vec![false; 9];
        for m in mask.iter_mut().take(8).skip(1) {
            *m = true;
        }
        fuse_no_crop(
            &weights,
            &mut embeds,
            std::slice::from_ref(&g),
            2,
            2,
            &nl,
            &sep,
            &mask,
        )
        .unwrap();
        // Text rows preserved.
        assert_eq!(embeds.row(0), &[1.0, 1.0, 1.0]);
        assert_eq!(embeds.row(8), &[2.0, 2.0, 2.0]);
        // Placeholder region now holds the 7-slot block ending in the separator.
        assert_eq!(embeds.row(1), g.row(0)); // first patch
        assert_eq!(embeds.row(3), &nl[..]); // row-0 newline
        assert_eq!(embeds.row(7), &sep[..]); // view_seperator at trailing slot
    }

    #[test]
    fn fuse_no_crop_handles_multiple_images() {
        let weights = Weights::default();
        let dim = 2;
        let g0 = grid(1, 1, dim, 10.0); // (1+1)*1 + 1 = 3-slot block
        let g1 = grid(1, 1, dim, 20.0);
        let nl = vec![0.0, 0.0];
        let sep = vec![-1.0, -1.0];
        // Two 3-slot image blocks = 6 placeholders, surrounded by 2 text tokens.
        let mut embeds = Mat::zeros(8, dim);
        let mut mask = vec![false; 8];
        for m in mask.iter_mut().take(7).skip(1) {
            *m = true;
        }
        fuse_no_crop(
            &weights,
            &mut embeds,
            &[g0.clone(), g1.clone()],
            1,
            1,
            &nl,
            &sep,
            &mask,
        )
        .unwrap();
        // Image 0 block: [g0, nl, sep] at positions 1,2,3.
        assert_eq!(embeds.row(1), g0.row(0));
        assert_eq!(embeds.row(2), &nl[..]);
        assert_eq!(embeds.row(3), &sep[..]);
        // Image 1 block: [g1, nl, sep] at positions 4,5,6.
        assert_eq!(embeds.row(4), g1.row(0));
        assert_eq!(embeds.row(5), &nl[..]);
        assert_eq!(embeds.row(6), &sep[..]);
    }

    #[test]
    fn fuse_no_crop_rejects_malformed_global_data() {
        let weights = Weights::default();
        let dim = 2;
        let malformed = Mat {
            rows: 1,
            cols: dim,
            data: vec![42.0],
        };
        let mut embeds = Mat::zeros(3, dim);
        let mask = vec![true, true, true];
        let err = fuse_no_crop(
            &weights,
            &mut embeds,
            &[malformed],
            1,
            1,
            &[0.0, 0.0],
            &[1.0, 1.0],
            &mask,
        );
        assert!(matches!(
            err,
            Err(err) if err.to_string().contains("append_newline_column grid")
        ));
    }

    #[test]
    fn fuse_crop_end_to_end_rearranges_local_then_global() {
        let weights = Weights::default();
        let dim = 2;
        let local_a = Mat::from_vec(2, dim, vec![10.0, 10.1, 11.0, 11.1]);
        let local_b = Mat::from_vec(2, dim, vec![20.0, 20.1, 21.0, 21.1]);
        let global = Mat::from_vec(1, dim, vec![90.0, 90.1]);
        let nl = vec![-5.0, -5.1];
        let sep = vec![-9.0, -9.1];

        // Local grid: width_crop_num=2, height_crop_num=1, each local tile 1x2.
        // Local block has one row of 4 features + newline = 5 rows.
        // Global block has one feature + newline + separator = 3 rows.
        let mut embeds = Mat::zeros(10, dim);
        embeds.row_mut(0).copy_from_slice(&[1.0, 1.0]);
        embeds.row_mut(9).copy_from_slice(&[2.0, 2.0]);
        let mut mask = vec![false; 10];
        for m in mask.iter_mut().take(9).skip(1) {
            *m = true;
        }

        fuse_crop(
            &weights,
            &mut embeds,
            &[local_a.clone(), local_b.clone()],
            2,
            1,
            1,
            2,
            &global,
            1,
            1,
            &nl,
            &sep,
            &mask,
        )
        .unwrap();

        assert_eq!(embeds.row(0), &[1.0, 1.0]);
        assert_eq!(embeds.row(1), local_a.row(0));
        assert_eq!(embeds.row(2), local_a.row(1));
        assert_eq!(embeds.row(3), local_b.row(0));
        assert_eq!(embeds.row(4), local_b.row(1));
        assert_eq!(embeds.row(5), &nl[..]);
        assert_eq!(embeds.row(6), global.row(0));
        assert_eq!(embeds.row(7), &nl[..]);
        assert_eq!(embeds.row(8), &sep[..]);
        assert_eq!(embeds.row(9), &[2.0, 2.0]);
    }
}
