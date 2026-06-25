//! Image ingest front end ([SPEC-018, SPEC-020..033],
//! PROPOSED_ARCHITECTURE.md §6.2). **Stub module** — implemented by
//! bd-1gv.2/bd-1gv.3.
//!
//! This is a frankentorch gap, built fresh: decode (EXIF-transpose, RGB);
//! `ToTensor` -> `Normalize(0.5, 0.5)` => [-1,1]; `ImageOps.pad` gray
//! (127,127,127); bilinear/bicubic aspect-preserving resize; Base (1024,
//! crop_mode=false) vs Gundam (`dynamic_preprocess` / `find_closest_aspect_ratio`)
//! tiling; the image-token id-stream layout; BOS prepend + masks; image-tensor
//! packing `images=[(crop, ori)]`. **L0 parity = exact.**

use std::path::Path;

use crate::error::{FocrError, FocrResult};

/// The preprocessed image bundle handed to the vision tower + connector:
/// the global/local view tensors, the spatial-crop grid, and the image-token
/// id-stream + `images_seq_mask`.
///
/// Stub: the concrete tensor/mask fields land with the preprocess beads.
#[derive(Debug, Default)]
pub struct Preprocessed {
    _private: (),
}

/// Decode + normalize + tile a document image at `path` into a [`Preprocessed`]
/// bundle (the `infer` data pipeline, crop_mode default).
///
/// # Errors
/// Always [`FocrError::NotImplemented`] in the skeleton.
pub fn preprocess_image(_path: &Path) -> FocrResult<Preprocessed> {
    Err(FocrError::NotImplemented(
        "preprocess::preprocess_image — image front end lands in Phase 1 (bd-1gv.2/3)".into(),
    ))
}
