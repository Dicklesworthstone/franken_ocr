//! `franken_ocr` — a pure-Rust, CPU-hyper-optimized runner for the Baidu
//! Unlimited-OCR model, with no general ML framework.
//!
//! See [`COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`] for the master plan and
//! `AGENTS.md` for the engineering doctrine. **This is a pre-Phase-0 skeleton**:
//! the public surface is declared so the project compiles and the CLI runs, but
//! the model forward, the model-specific int8/int4 kernels, and the weight
//! converter are not yet implemented (Phases 1–4).
//!
//! [`COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md`]: ../COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md
#![forbid(unsafe_code)]

pub mod cli;
pub mod error;
pub mod native_engine;
pub mod preprocess;
pub mod robot;
pub mod tokenizer;

pub use cli::cli_main;
pub use error::{FocrError, FocrResult};

/// The OCR engine handle.
///
/// In the full design (plan §3.3) this OWNS exactly one `asupersync` `Runtime`
/// and exposes a **synchronous, blocking** API — the async runtime is an
/// implementation detail, never leaked to the host. The heavy model forward
/// fans out across all physical cores via the frankentorch kernel's own rayon
/// pool, from a **sequential** outer page loop (never nest rayon under a held
/// lock; never nest a second runtime — see AGENTS.md doctrine #5).
///
/// Skeleton: construction only; `recognize` is wired in Phase 1.
pub struct OcrEngine {
    _private: (),
}

impl OcrEngine {
    /// Construct the engine. Phase 0 stub (the asupersync `Runtime` + model
    /// cache land in Phase 0/1).
    pub fn new() -> FocrResult<Self> {
        Ok(Self { _private: () })
    }

    /// Recognize a single document image, returning structured markdown.
    ///
    /// Not yet implemented — the vision encoder, R-SWA decoder, MoE dispatch,
    /// and postprocessing land in Phase 1 (plan §10).
    pub fn recognize(&self, _image_path: &std::path::Path) -> FocrResult<String> {
        Err(FocrError::NotImplemented(
            "OcrEngine::recognize — the model forward lands in Phase 1 (see plan §10)".into(),
        ))
    }
}
