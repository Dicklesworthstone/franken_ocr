//! The native model package — the hand-written Unlimited-OCR forward over the
//! [`Mat`]/slice currency (PROPOSED_ARCHITECTURE.md §2, §6).
//!
//! This module is **plain Rust over `ft-kernel-cpu` free functions**: no tensor
//! graph, no autograd, no `ft-api` session/tape (plan §1.1 P2). Every kernel
//! call funnels through [`nn`] (the frankentorch facade, §5); every other
//! submodule implements a contiguous block of THE SPEC:
//!
//! * [`tensor`] — the `Mat` activation currency + quantized weight structs (§4).
//! * [`nn`] — the frankentorch facade (matmul / int8 linear / conv2d / sdpa /
//!   rms_norm / layer_norm / softmax / silu / gelu / quick_gelu) (§5).
//! * [`vision_sam`] / [`vision_clip`] / [`vision_bridge`] — the vision tower
//!   ([SPEC-040..052], §6.3–§6.5).
//! * [`connector`] — masked-scatter vision fusion ([SPEC-060..066], §6.6).
//! * [`decoder`] / [`rswa`] / [`moe`] — the DeepseekV2 decoder, R-SWA ring
//!   attention, and MoE block ([SPEC-070..096], §6.7–§6.9).
//! * [`sampler`] — the AR decode loop + sampler ([SPEC-100..103], §6.10).
//! * [`postprocess`] — ref/det parse, bbox /999, markdown ([SPEC-110..119], §6.11).
//! * [`weights`] — the `.focrq` reader + census (§6.12, §7).
//!
//! [`Mat`]: tensor::Mat

pub mod connector;
pub mod decoder;
pub mod moe;
pub mod nn;
pub mod postprocess;
pub mod rswa;
pub mod sampler;
pub mod tensor;
pub mod vision_bridge;
pub mod vision_clip;
pub mod vision_sam;
pub mod weights;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use crate::error::{FocrError, FocrResult};
use weights::Weights;

/// The loaded Unlimited-OCR model: weights + the fixed-shape forward.
///
/// Held behind an [`Arc`] and cached through a process-global [`Weak`] (see
/// [`OcrModel::load`]) so repeated `focr ocr` invocations in one process share
/// one weight blob — the model is a single read-only artifact; concurrent
/// forwards are serialized by the engine's sequential page loop (plan §6.5 P6),
/// not by cloning the weights.
///
/// Skeleton: construction loads (stubbed) [`Weights`]; the forward is wired in
/// Phase 1.
pub struct OcrModel {
    /// Filesystem path the model was resolved + loaded from (provenance).
    path: PathBuf,
    /// The loaded weight set (stub).
    #[allow(dead_code)]
    weights: Weights,
}

/// Process-global cache of the last-loaded model, keyed by resolved path.
///
/// A [`Weak`] so the cache never *keeps the model alive on its own*: once every
/// [`Arc<OcrModel>`] handle is dropped, the weight blob is freed; a subsequent
/// [`OcrModel::load`] of the same path re-reads it. While at least one handle is
/// live, repeat loads of the same path hand back a cheap `Arc::clone`.
fn model_cache() -> &'static Mutex<Option<(PathBuf, Weak<OcrModel>)>> {
    static CACHE: OnceLock<Mutex<Option<(PathBuf, Weak<OcrModel>)>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

impl OcrModel {
    /// Resolve `path` to a concrete model artifact (`.focrq` blob or a
    /// safetensors directory) — the header-sniff / search-path logic
    /// (`native_model_available`, bd-223.7).
    ///
    /// Skeleton: returns `path` as-is if it exists, else
    /// [`FocrError::ModelNotFound`]. The candidate-path search + magic sniff land
    /// with the resolver bead.
    pub fn resolve_model(path: &Path) -> FocrResult<PathBuf> {
        if path.exists() {
            Ok(path.to_path_buf())
        } else {
            Err(FocrError::ModelNotFound(format!(
                "no model artifact at {} (resolver lands in Phase 0/1, bd-223.7)",
                path.display()
            )))
        }
    }

    /// Load (or fetch from the global cache) the model at `path`.
    ///
    /// Resolves the path, then returns a shared [`Arc`]: if a live handle for the
    /// same resolved path is still cached, that `Arc` is cloned; otherwise the
    /// weights are loaded and a fresh handle is cached weakly.
    ///
    /// # Errors
    /// [`FocrError::ModelNotFound`] if the path doesn't resolve; otherwise
    /// whatever [`Weights::load`] returns (currently
    /// [`FocrError::NotImplemented`] — the `.focrq` reader is Phase 2).
    pub fn load(path: &Path) -> FocrResult<Arc<Self>> {
        let resolved = Self::resolve_model(path)?;

        let cache = model_cache();
        let mut guard = cache.lock().expect("model cache mutex poisoned");
        if let Some((cached_path, weak)) = guard.as_ref() {
            if *cached_path == resolved {
                if let Some(strong) = weak.upgrade() {
                    return Ok(strong);
                }
            }
        }

        let weights = Weights::load(&resolved)?;
        let model = Arc::new(Self {
            path: resolved.clone(),
            weights,
        });
        *guard = Some((resolved, Arc::downgrade(&model)));
        Ok(model)
    }

    /// The path this model was loaded from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Run the full forward for one document image and return the raw decoded
    /// model text (pre-postprocess).
    ///
    /// The vision encode -> connector -> prefill -> R-SWA/MoE decode loop. Wired
    /// in Phase 1 across the submodules above.
    ///
    /// # Errors
    /// Always [`FocrError::NotImplemented`] in the skeleton.
    pub fn forward(&self, _image_path: &Path) -> FocrResult<String> {
        Err(FocrError::NotImplemented(
            "native_engine::OcrModel::forward — the model forward lands in Phase 1 (plan §10)".into(),
        ))
    }

    /// Recognize one document image end-to-end (forward + postprocess),
    /// returning structured markdown.
    ///
    /// # Errors
    /// Always [`FocrError::NotImplemented`] in the skeleton.
    pub fn recognize(&self, image_path: &Path) -> FocrResult<String> {
        let decoded = self.forward(image_path)?;
        postprocess::finalize(&decoded, 0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_model_rejects_missing_path() {
        let missing = Path::new("/definitely/not/a/real/model/path.focrq");
        let r = OcrModel::resolve_model(missing);
        assert!(matches!(r, Err(FocrError::ModelNotFound(_))));
    }

    #[test]
    fn load_missing_path_is_model_not_found() {
        let missing = Path::new("/definitely/not/a/real/model/path.focrq");
        let r = OcrModel::load(missing);
        assert!(matches!(r, Err(FocrError::ModelNotFound(_))));
    }
}
