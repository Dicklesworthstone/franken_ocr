//! `franken_ocr` — a pure-Rust, CPU-hyper-optimized runner for the Baidu
//! Unlimited-OCR model, with no general ML framework.
//!
//! See [`COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`] for the master plan and
//! `AGENTS.md` for the engineering doctrine. The public surface is the
//! synchronous, blocking [`OcrEngine`] (plan §3.3, G6) plus the `focr` CLI; the
//! heavy model forward, the model-specific int8/int4 kernels, and the weight
//! converter land across Phases 1–4. The end-to-end pipeline is **wired** here
//! (preprocess → vision → connector → decoder → sampler → postprocess) over the
//! [`native_engine`] modules; stages whose `.focrq` tensor accessors are not yet
//! built surface a clean [`FocrError::NotImplemented`] rather than fabricating
//! output (doctrine #1).
//!
//! [`COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`]: ../COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md
#![deny(unsafe_code)]

pub mod adaptive;
pub mod cli;
pub mod error;
pub mod native_engine;
pub mod preprocess;
pub mod quant;
pub mod robot;
pub mod simd;
pub mod tokenizer;

pub use cli::cli_main;
pub use error::{FocrError, FocrResult};

use std::path::Path;
use std::sync::{Arc, Mutex};

use asupersync::runtime::{Runtime, RuntimeBuilder};
use native_engine::OcrModel;

/// Environment override for the model artifact path (`.focrq` blob or a
/// safetensors directory). When unset, [`OcrEngine`] falls back to
/// [`DEFAULT_MODEL_PATH`].
pub const MODEL_PATH_ENV: &str = "FOCR_MODEL_PATH";

/// Default model artifact location when [`MODEL_PATH_ENV`] is unset (plan §7.5).
/// A relative `models/unlimited-ocr.focrq` next to the working directory; the
/// model-gated e2e tests deliberately point this at `/nonexistent` to prove the
/// native path's clean [`FocrError::ModelNotFound`].
pub const DEFAULT_MODEL_PATH: &str = "models/unlimited-ocr.focrq";

/// The OCR engine handle.
///
/// Per the proven `franken_whisper` integration (plan §3.3) this **OWNS exactly
/// one** `asupersync` [`Runtime`] and exposes a **synchronous, blocking** API:
/// public methods run the heavy work via `runtime.block_on(...)`, so the async
/// runtime is an implementation detail never leaked to the host (satisfies G6).
/// The model forward fans out across all physical cores via the frankentorch
/// kernel's own rayon pool, driven from a **sequential** outer page loop — never
/// nest rayon under a held lock, never nest a second runtime (doctrine #5).
///
/// The loaded [`OcrModel`] is cached behind a [`Mutex<Option<Arc<…>>>`] so the
/// 6.67 GB weight blob is read once per engine and shared across calls. The
/// global weak cache in [`native_engine`] additionally de-dups across engines in
/// one process.
pub struct OcrEngine {
    /// The single owned async runtime. All public methods block on it.
    runtime: Runtime,
    /// The lazily-loaded, shared model (one read-only weight blob per engine).
    model: Mutex<Option<Arc<OcrModel>>>,
}

impl OcrEngine {
    /// Construct the engine, building the single owned `asupersync` runtime
    /// (plan §3.3: `worker_threads(2)`, `blocking_threads(1, 4)`,
    /// `thread_name_prefix("focr")`). The model is loaded lazily on the first
    /// [`OcrEngine::recognize`] so construction is cheap and never touches the
    /// 6.67 GB blob.
    ///
    /// # Errors
    /// [`FocrError::Other`] if the runtime fails to build (e.g. the OS refuses to
    /// spawn worker threads).
    pub fn new() -> FocrResult<Self> {
        // Small blocking pool is a guard, not the mechanism: exactly one live
        // forward at a time runs the N-core kernel fan-out (doctrine #5).
        let runtime = RuntimeBuilder::new()
            .worker_threads(2)
            .blocking_threads(1, 4)
            .thread_name_prefix("focr")
            .build()
            .map_err(|e| FocrError::Other(anyhow::anyhow!("asupersync runtime build: {e}")))?;
        Ok(Self {
            runtime,
            model: Mutex::new(None),
        })
    }

    /// Resolve the configured model artifact path ([`MODEL_PATH_ENV`] override,
    /// else [`DEFAULT_MODEL_PATH`]).
    #[must_use]
    pub fn model_path() -> std::path::PathBuf {
        std::env::var_os(MODEL_PATH_ENV)
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(DEFAULT_MODEL_PATH))
    }

    /// Load (or fetch the cached) [`OcrModel`] at an explicit `path`.
    ///
    /// First call for the engine reads the weights; later calls clone the cached
    /// [`Arc`]. A missing/unresolvable model yields a clean
    /// [`FocrError::ModelNotFound`] (the model-gated e2e tests rely on this path,
    /// never a panic). The cache holds at most one model per engine; if `path`
    /// differs from the cached one it is reloaded.
    ///
    /// # Errors
    /// [`FocrError::ModelNotFound`] when the artifact does not resolve; otherwise
    /// whatever [`OcrModel::load`] returns (currently [`FocrError::NotImplemented`]
    /// once a path *does* resolve — the `.focrq` reader is Phase 2).
    fn model_at(&self, path: &Path) -> FocrResult<Arc<OcrModel>> {
        let mut guard = self.model.lock().expect("OcrEngine model mutex poisoned");
        if let Some(m) = guard.as_ref() {
            if m.path() == path {
                return Ok(Arc::clone(m));
            }
        }
        let m = OcrModel::load(path)?;
        *guard = Some(Arc::clone(&m));
        Ok(m)
    }

    /// Recognize a single document image, returning structured markdown.
    ///
    /// **Synchronous and blocking** (G6): the heavy forward runs inside the
    /// engine's owned runtime via `block_on`, with a **sequential** single-page
    /// drive (doctrine #5). The model is resolved from [`OcrEngine::model_path`]
    /// (the [`MODEL_PATH_ENV`] override, else [`DEFAULT_MODEL_PATH`]) and
    /// loaded/cached on first use; when the weights are absent this returns
    /// [`FocrError::ModelNotFound`] cleanly (not a panic) so the model-gated e2e
    /// tests can skip-with-success by pointing the fallback at `/nonexistent`.
    ///
    /// # Errors
    /// * [`FocrError::ModelNotFound`] if the model artifact is absent/unresolvable.
    /// * Otherwise whatever the forward pipeline returns (today
    ///   [`FocrError::NotImplemented`] from the first stage whose `.focrq` tensor
    ///   accessor is not yet built — the pipeline is fully wired and typed).
    pub fn recognize(&self, image_path: &Path) -> FocrResult<String> {
        self.recognize_with_model(&Self::model_path(), image_path)
    }

    /// Recognize `image_path` using the model artifact at an explicit
    /// `model_path` (the path-explicit form of [`OcrEngine::recognize`]).
    ///
    /// Used by [`OcrEngine::recognize`] (with the env-resolved default) and by
    /// callers / tests that want to pin a specific artifact without setting an
    /// environment variable. Loading happens OUTSIDE `block_on` so a missing
    /// model is the clean [`FocrError::ModelNotFound`] without ever entering the
    /// runtime.
    ///
    /// # Errors
    /// As [`OcrEngine::recognize`].
    pub fn recognize_with_model(&self, model_path: &Path, image_path: &Path) -> FocrResult<String> {
        let model = self.model_at(model_path)?;
        let image_path = image_path.to_path_buf();
        // One owned runtime; the per-page forward is the only blocking work and
        // is driven sequentially (no nested runtime, no concurrent forwards).
        self.runtime
            .block_on(async move { model.recognize(&image_path) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The engine constructs (its single owned runtime builds) without touching
    /// the model blob — construction is cheap and lazy.
    #[test]
    fn engine_constructs_without_model() {
        let engine = OcrEngine::new().expect("runtime builds");
        // Constructing alone must not have loaded a model.
        assert!(
            engine.model.lock().expect("mutex").is_none(),
            "model must be loaded lazily, not at construction"
        );
    }

    /// `recognize_with_model` on a guaranteed-absent model path returns a clean
    /// `ModelNotFound` (exit code 3) — NOT a panic, NOT NotImplemented. This is
    /// the path the model-gated e2e tests pin (point the fallback at
    /// `/nonexistent`). We use the path-explicit form so the test never mutates
    /// the process environment (the crate root `#![deny(unsafe_code)]` rules out
    /// the `unsafe` `std::env::set_var`).
    #[test]
    fn recognize_missing_model_is_clean_model_not_found() {
        let engine = OcrEngine::new().expect("runtime builds");
        let err = engine
            .recognize_with_model(
                Path::new("/nonexistent/franken_ocr/model.focrq"),
                Path::new("/some/document.png"),
            )
            .expect_err("absent model must error");
        assert!(
            matches!(err, FocrError::ModelNotFound(_)),
            "expected ModelNotFound, got {err:?}"
        );
        assert_eq!(err.exit_code(), 3, "ModelNotFound must map to exit code 3");
    }

    /// The blocking `recognize` path (env-resolved default) also yields a clean
    /// `ModelNotFound` when `FOCR_MODEL_PATH` is unset and the default artifact is
    /// absent — proving the public entrypoint never panics without weights. (The
    /// default `models/unlimited-ocr.focrq` does not exist in the test CWD.)
    #[test]
    fn public_recognize_without_weights_is_model_not_found() {
        // Only assert when the env override is unset AND the default is absent —
        // the normal CI/dev condition. If a sibling test or the developer has a
        // real model, skip rather than misfire.
        if std::env::var_os(MODEL_PATH_ENV).is_none()
            && !std::path::Path::new(DEFAULT_MODEL_PATH).exists()
        {
            let engine = OcrEngine::new().expect("runtime builds");
            let err = engine
                .recognize(Path::new("/some/document.png"))
                .expect_err("absent default model must error");
            assert!(matches!(err, FocrError::ModelNotFound(_)));
        }
    }

    /// The model-path resolver falls back to the documented default when the env
    /// override is unset (read-only check; no env mutation under `deny(unsafe)`).
    #[test]
    fn model_path_falls_back_to_default_when_env_unset() {
        if std::env::var_os(MODEL_PATH_ENV).is_none() {
            assert_eq!(
                OcrEngine::model_path(),
                std::path::PathBuf::from(DEFAULT_MODEL_PATH)
            );
        }
    }

    /// Calling `recognize_with_model` twice loads the model once per distinct
    /// path (the per-engine cache); a second call with the SAME absent path still
    /// returns `ModelNotFound` (the absent model is never cached as a success).
    #[test]
    fn repeated_missing_model_stays_model_not_found() {
        let engine = OcrEngine::new().expect("runtime builds");
        let p = Path::new("/nonexistent/franken_ocr/model.focrq");
        let img = Path::new("/some/document.png");
        for _ in 0..3 {
            let err = engine.recognize_with_model(p, img).expect_err("absent");
            assert!(matches!(err, FocrError::ModelNotFound(_)));
        }
    }
}
