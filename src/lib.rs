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
#![cfg_attr(target_arch = "aarch64", allow(stable_features))]
#![cfg_attr(
    target_arch = "aarch64",
    feature(stdarch_neon_dotprod, stdarch_neon_i8mm)
)]
#![deny(unsafe_code)]

pub mod adaptive;
pub mod cli;
pub mod conformance;
pub mod dist;
pub mod doctor;
pub mod error;
pub mod native_engine;
pub mod pdf;
pub mod preprocess;
pub mod progress;
pub mod quant;
pub mod robot;
pub mod simd;
pub mod storage;
pub mod tokenizer;

pub use cli::cli_main;
pub use error::{FocrError, FocrResult};
/// Multi-model architecture descriptors + registry (the "model zoo" foundation,
/// epic bd-3jo6 / A1). Additive metadata layer; the live forward is unchanged.
pub use native_engine::model_arch;
pub use native_engine::{ExtractedFigure, LayoutSpan, RecognizedDocument};

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use asupersync::runtime::{Runtime, RuntimeBuilder};
use native_engine::OcrModel;

/// Environment override for the model artifact path (`.focrq` blob or a
/// safetensors directory). When unset, [`OcrEngine`] falls back to
/// [`DEFAULT_MODEL_PATH`].
pub const MODEL_PATH_ENV: &str = "FOCR_MODEL_PATH";

/// Source-code license notice for this crate, surfaced in the long version
/// report separately from the model-weights notice.
pub const FOCR_PROJECT_LICENSE_NOTICE: &str =
    "franken_ocr - Copyright (c) 2026 Jeffrey Emanuel, MIT License (with OpenAI/Anthropic Rider)";

/// Baidu Unlimited-OCR model-weights notice. This is the single source of truth
/// for the notice that must travel with redistributed `.focrq` artifacts and
/// agent-facing provenance surfaces (plan §2.2 / §11).
pub const FOCR_MODEL_LICENSE_NOTICE: &str =
    "Baidu Unlimited-OCR - Copyright (c) 2026 Baidu, MIT License";

/// Default model artifact location when [`MODEL_PATH_ENV`] is unset (plan §7.5).
/// A relative `models/unlimited-ocr.focrq` next to the working directory; the
/// model-gated e2e tests deliberately point this at `/nonexistent` to prove the
/// native path's clean [`FocrError::ModelNotFound`].
pub const DEFAULT_MODEL_PATH: &str = "models/unlimited-ocr.focrq";

const DEFAULT_FORWARD_STAGE_BUDGET_MS: u64 = 10 * 60 * 1000;

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
// ── Cooperative shutdown (bd-223.2) ─────────────────────────────────────────
//
// The process-global shutdown flag IS the ShutdownController: Ctrl+C (the CLI
// installs the handler in `cli_main`) or an embedder's `request_shutdown()`
// sets it; every long loop in the engine — the per-page loops and every
// per-decode-step loop — polls it via [`cancel_checkpoint`] (one relaxed
// atomic load per token: unmeasurable) and returns [`FocrError::Cancelled`]
// (exit 6) at the next boundary. Cancellation is COOPERATIVE by design:
// `spawn_blocking` closures keep running on drop, so the flag is observed
// INSIDE them (doctrine #5 / the franken_whisper pattern). Per-request tokens
// for embedders who need independent cancellation of concurrent engines are a
// documented follow-up — a single flag matches the one-live-forward discipline.
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Request cooperative shutdown: every in-flight recognition aborts with
/// [`FocrError::Cancelled`] at its next checkpoint (page boundary or decode
/// step). Idempotent; never blocks.
pub fn request_shutdown() {
    SHUTDOWN_REQUESTED.store(true, Ordering::SeqCst);
}

/// Whether cooperative shutdown has been requested.
#[must_use]
pub fn shutdown_requested() -> bool {
    SHUTDOWN_REQUESTED.load(Ordering::Relaxed)
}

/// Clear the shutdown flag (tests + long-lived embedders that survive a
/// cancelled batch and start a new one).
pub fn reset_shutdown() {
    SHUTDOWN_REQUESTED.store(false, Ordering::SeqCst);
}

/// The cooperative cancellation checkpoint (bd-223.2): call at every page
/// boundary and decode step.
///
/// # Errors
/// [`FocrError::Cancelled`] once [`request_shutdown`] has been called.
pub fn cancel_checkpoint() -> FocrResult<()> {
    if shutdown_requested() {
        return Err(FocrError::Cancelled);
    }
    Ok(())
}

// ── The one thread/CPU budget (bd-223.2 addendum; plan §7.5) ────────────────

/// The single process-wide thread budget, read ONCE: `FOCR_THREADS` (env)
/// else the PHYSICAL core count (hyperthreads oversubscribe the int8 GEMMs —
/// never `available_parallelism`). Every pool-sizing consumer (the kernel
/// rayon pool, the gauntlet fairness pins, `robot health`) reads THIS.
pub fn thread_budget() -> usize {
    static BUDGET: OnceLock<usize> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("FOCR_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(num_cpus::get_physical)
    })
}

/// The kernel rayon pool's CURRENT width — the diagnostic the capacity
/// certificate (bd-re8.18) records before/after the many-pages soak to prove
/// no second pool was spawned and the width never grew mid-run (the N×
/// oversubscription class doctrine #5 forbids). First call instantiates the
/// global pool, which is exactly what the kernels themselves use.
pub fn kernel_pool_width() -> usize {
    rayon::current_num_threads()
}

// ── Bounded per-page result streaming (bd-223.2 scaffold) ───────────────────

/// Stream page results from a SEQUENTIAL producer to a consumer through a
/// BOUNDED channel — the bd-223.2 streaming scaffold the robot/NDJSON
/// multi-page path adopts: the producer runs on its own thread and BLOCKS
/// when the consumer lags (backpressure — memory never grows unbounded);
/// the consumer loop drains with a short `recv_timeout` so it can interleave
/// its own bookkeeping. Pages are produced STRICTLY sequentially (doctrine
/// #5 — streaming the OUTPUT of sequential pages, never concurrent
/// forwards).
///
/// `produce` yields `Some(item)` per page and `None` when exhausted;
/// `consume` receives each item in order. Returns the number of items
/// streamed.
///
/// # Errors
/// The producer's first error aborts the stream and is returned after the
/// worker joins (consumers see only the items produced before it).
pub fn stream_pages<T, P, C>(capacity: usize, mut produce: P, mut consume: C) -> FocrResult<usize>
where
    T: Send + 'static,
    P: FnMut() -> FocrResult<Option<T>> + Send + 'static,
    C: FnMut(T),
{
    let (tx, rx) = std::sync::mpsc::sync_channel::<T>(capacity.max(1));
    let worker = std::thread::Builder::new()
        .name("focr-page-stream".into())
        .spawn(move || -> FocrResult<()> {
            loop {
                match produce()? {
                    Some(item) => {
                        // A closed receiver means the consumer is gone —
                        // treat as cancellation, not success.
                        if tx.send(item).is_err() {
                            return Err(FocrError::Cancelled);
                        }
                    }
                    None => return Ok(()),
                }
            }
        })
        .map_err(|e| FocrError::Other(anyhow::anyhow!("page-stream worker spawn: {e}")))?;

    let mut n = 0usize;
    loop {
        match rx.recv_timeout(Duration::from_millis(40)) {
            Ok(item) => {
                consume(item);
                n += 1;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if worker.is_finished() {
                    // Drain anything raced in between finish and the check.
                    while let Ok(item) = rx.try_recv() {
                        consume(item);
                        n += 1;
                    }
                    break;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    worker
        .join()
        .map_err(|_| FocrError::Other(anyhow::anyhow!("page-stream worker panicked")))??;
    Ok(n)
}

/// Boxed per-page streaming sink for multi-page passes (bd-2z0y): called
/// with the 1-based page index and the trimmed raw body as each `<PAGE>`
/// boundary is crossed in the token stream. `Send` because the pass runs on
/// the engine's blocking pool.
pub type PageSink = Box<dyn FnMut(usize, &str) + Send>;

pub struct OcrEngine {
    /// The single owned async runtime. All public methods block on it.
    runtime: Runtime,
    /// The lazily-loaded, shared model (one read-only weight blob per engine).
    model: Mutex<Option<Arc<OcrModel>>>,
}

impl OcrEngine {
    /// Take (consume) the staff-level metadata from the most recent TrOMR
    /// music forward on this engine's cached model, if any (bd-av64.2): the
    /// recognized staves' detection indices + page-space bboxes and any
    /// per-staff skips. Returns `None` when no model is loaded, the loaded
    /// model has run no music forward since the last take, or the last
    /// forward was not a music run. The CLI uses this to emit robot `staff`
    /// events and the `--json` `staves` array.
    #[must_use]
    pub fn take_music_page_meta(&self) -> Option<native_engine::MusicPageMeta> {
        self.model
            .lock()
            .ok()
            .and_then(|slot| slot.as_ref().map(std::sync::Arc::clone))
            .and_then(|model| model.take_music_meta())
    }

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
        {
            let guard = self.model_guard()?;
            if let Some(m) = guard.as_ref()
                && m.path() == path
            {
                return Ok(Arc::clone(m));
            }
        }

        let loaded = OcrModel::load(path)?;
        let loaded_path = loaded.path().to_path_buf();

        let mut guard = self.model_guard()?;
        if let Some(m) = guard.as_ref()
            && m.path() == loaded_path
        {
            return Ok(Arc::clone(m));
        }
        *guard = Some(Arc::clone(&loaded));
        Ok(loaded)
    }

    fn model_guard(&self) -> FocrResult<MutexGuard<'_, Option<Arc<OcrModel>>>> {
        self.model
            .lock()
            .map_err(|_| FocrError::Other(anyhow::anyhow!("OcrEngine model mutex poisoned")))
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
    /// # Choosing a model
    /// The engine runs whichever `.focrq` you point it at; pick by task:
    /// * **`unlimited-ocr`** (the default; what [`OcrEngine::recognize`] resolves) —
    ///   the **fast plain-text document OCR** model for general documents & PDFs.
    ///   This is the right default for ordinary text.
    /// * **`got-ocr2`** — a heavier, **specialized structured-output** model for the
    ///   formats the default cannot produce: math (LaTeX), tables, charts, molecular
    ///   (SMILES), geometry, and sheet music. Reach for it **only when you need that
    ///   format extraction**, not as a faster general OCR.
    ///
    /// See [`native_engine::model_arch`] for the registry (id → tasks → implemented)
    /// and `docs/zoo/` for each model's spec.
    ///
    /// # Errors
    /// As [`OcrEngine::recognize`].
    pub fn recognize_with_model(&self, model_path: &Path, image_path: &Path) -> FocrResult<String> {
        let model = self.model_at(model_path)?;
        let image_path = image_path.to_path_buf();
        // One owned runtime; the per-page forward is the only blocking work and
        // is driven sequentially on the runtime blocking pool, never inline on
        // the async polling thread (no nested runtime, no concurrent forwards).
        self.run_blocking_stage_with_budget(
            "forward",
            Self::stage_budget("FORWARD", DEFAULT_FORWARD_STAGE_BUDGET_MS),
            move || model.recognize(&image_path),
        )
    }

    /// Recognize an already-decoded in-memory [`image::DynamicImage`], returning
    /// structured markdown — the path-free form of [`OcrEngine::recognize`].
    ///
    /// This is the entry point the native PDF path uses: [`crate::pdf`] rasterizes
    /// one PDF page to a `DynamicImage` and hands it here, so a scanned PDF flows
    /// through the identical preprocess → vision → decoder → postprocess pipeline a
    /// PNG would, with no intermediate temp files. The model is resolved from
    /// [`OcrEngine::model_path`] and loaded/cached on first use.
    ///
    /// # Errors
    /// As [`OcrEngine::recognize`].
    pub fn recognize_dynamic(&self, image: image::DynamicImage) -> FocrResult<String> {
        self.recognize_dynamic_with_model(&Self::model_path(), image)
    }

    /// Recognize an in-memory [`image::DynamicImage`] against the model artifact at
    /// an explicit `model_path` (the path-explicit form of
    /// [`OcrEngine::recognize_dynamic`]).
    ///
    /// # Errors
    /// As [`OcrEngine::recognize_with_model`].
    pub fn recognize_dynamic_with_model(
        &self,
        model_path: &Path,
        image: image::DynamicImage,
    ) -> FocrResult<String> {
        let model = self.model_at(model_path)?;
        self.run_blocking_stage_with_budget(
            "forward",
            Self::stage_budget("FORWARD", DEFAULT_FORWARD_STAGE_BUDGET_MS),
            move || model.recognize_dynamic(image),
        )
    }

    /// Recognize a single document image, returning the markdown AND the
    /// structured layout (bounding boxes) — the structured form of
    /// [`OcrEngine::recognize`] that `focr ocr --json` / `-o out.json` uses.
    ///
    /// # Errors
    /// As [`OcrEngine::recognize`].
    pub fn recognize_with_layout(&self, image_path: &Path) -> FocrResult<RecognizedDocument> {
        self.recognize_with_layout_model(&Self::model_path(), image_path)
    }

    /// The path-explicit form of [`OcrEngine::recognize_with_layout`].
    ///
    /// # Errors
    /// As [`OcrEngine::recognize_with_model`].
    pub fn recognize_with_layout_model(
        &self,
        model_path: &Path,
        image_path: &Path,
    ) -> FocrResult<RecognizedDocument> {
        let model = self.model_at(model_path)?;
        let image_path = image_path.to_path_buf();
        self.run_blocking_stage_with_budget(
            "forward",
            Self::stage_budget("FORWARD", DEFAULT_FORWARD_STAGE_BUDGET_MS),
            move || model.recognize_with_layout(&image_path),
        )
    }

    /// Recognize an in-memory [`image::DynamicImage`], returning the markdown AND
    /// the structured layout — the in-memory form of
    /// [`OcrEngine::recognize_with_layout`] the native PDF JSON path uses.
    ///
    /// # Errors
    /// As [`OcrEngine::recognize_dynamic`].
    pub fn recognize_dynamic_with_layout(
        &self,
        image: image::DynamicImage,
    ) -> FocrResult<RecognizedDocument> {
        self.recognize_dynamic_with_layout_model(&Self::model_path(), image)
    }

    /// The path-explicit form of [`OcrEngine::recognize_dynamic_with_layout`].
    ///
    /// # Errors
    /// As [`OcrEngine::recognize_with_model`].
    pub fn recognize_dynamic_with_layout_model(
        &self,
        model_path: &Path,
        image: image::DynamicImage,
    ) -> FocrResult<RecognizedDocument> {
        let model = self.model_at(model_path)?;
        self.run_blocking_stage_with_budget(
            "forward",
            Self::stage_budget("FORWARD", DEFAULT_FORWARD_STAGE_BUDGET_MS),
            move || model.recognize_dynamic_with_layout(image),
        )
    }

    /// Recognize a single document image, returning the markdown + layout AND the
    /// figure regions cropped out of the source image — the regions the markdown
    /// renders as `![](images/…)` placeholders. This is what `focr ocr
    /// --extract-figures` uses to write real figure files.
    ///
    /// # Errors
    /// As [`OcrEngine::recognize_with_layout`].
    pub fn recognize_with_figures(
        &self,
        image_path: &Path,
    ) -> FocrResult<(RecognizedDocument, Vec<ExtractedFigure>)> {
        self.recognize_with_figures_model(&Self::model_path(), image_path)
    }

    /// The path-explicit form of [`OcrEngine::recognize_with_figures`].
    ///
    /// # Errors
    /// As [`OcrEngine::recognize_with_model`].
    pub fn recognize_with_figures_model(
        &self,
        model_path: &Path,
        image_path: &Path,
    ) -> FocrResult<(RecognizedDocument, Vec<ExtractedFigure>)> {
        let model = self.model_at(model_path)?;
        let image_path = image_path.to_path_buf();
        self.run_blocking_stage_with_budget(
            "forward",
            Self::stage_budget("FORWARD", DEFAULT_FORWARD_STAGE_BUDGET_MS),
            move || model.recognize_with_figures(&image_path),
        )
    }

    /// Recognize an in-memory [`image::DynamicImage`], returning the markdown +
    /// layout AND the cropped figure regions — the in-memory form the native PDF
    /// `--extract-figures` path uses (the page raster is the crop source).
    ///
    /// # Errors
    /// As [`OcrEngine::recognize_dynamic_with_layout`].
    pub fn recognize_dynamic_with_figures(
        &self,
        image: image::DynamicImage,
    ) -> FocrResult<(RecognizedDocument, Vec<ExtractedFigure>)> {
        self.recognize_dynamic_with_figures_model(&Self::model_path(), image)
    }

    /// The path-explicit form of [`OcrEngine::recognize_dynamic_with_figures`].
    ///
    /// # Errors
    /// As [`OcrEngine::recognize_with_model`].
    pub fn recognize_dynamic_with_figures_model(
        &self,
        model_path: &Path,
        image: image::DynamicImage,
    ) -> FocrResult<(RecognizedDocument, Vec<ExtractedFigure>)> {
        let model = self.model_at(model_path)?;
        self.run_blocking_stage_with_budget(
            "forward",
            Self::stage_budget("FORWARD", DEFAULT_FORWARD_STAGE_BUDGET_MS),
            move || model.recognize_dynamic_with_figures(image),
        )
    }

    /// Recognize a batch of document images in one load-once pass, returning one
    /// [`FocrResult`] per image in input order (`result[i]` ⇄ `images[i]`).
    ///
    /// The model is resolved from [`OcrEngine::model_path`] and loaded/cached on
    /// first use; see [`OcrEngine::recognize_batch_with_model`] for the
    /// path-explicit form and the spine semantics.
    ///
    /// # Errors
    /// [`FocrError::ModelNotFound`] if the model artifact is absent/unresolvable,
    /// or [`FocrError::Timeout`] if the whole batch exceeds its budget. Per-image
    /// failures surface inside the returned `Vec`, never as the outer error.
    pub fn recognize_batch(&self, images: &[&Path]) -> FocrResult<Vec<FocrResult<String>>> {
        self.recognize_batch_with_model(&Self::model_path(), images)
    }

    /// Recognize `images` against the model artifact at an explicit `model_path`
    /// (the path-explicit form of [`OcrEngine::recognize_batch`]).
    ///
    /// The model is acquired ONCE (the per-engine `Arc` cache), then the entire
    /// batch runs inside a SINGLE blocking stage on the runtime's blocking pool —
    /// the continuous-batch decode spine is the single sequential driver, with no
    /// per-step relock (Doctrine #5). The forward budget scales with the image
    /// count. When the spine is disarmed ([`native_engine`]
    /// `FOCR_BATCH_SPINE`), [`OcrModel::recognize_batch`] falls back to the proven
    /// per-image sequential path, so the spine-off result is byte-identical to
    /// today's loop.
    ///
    /// # Errors
    /// As [`OcrEngine::recognize_batch`].
    pub fn recognize_batch_with_model(
        &self,
        model_path: &Path,
        images: &[&Path],
    ) -> FocrResult<Vec<FocrResult<String>>> {
        let model = self.model_at(model_path)?;
        let owned: Vec<std::path::PathBuf> = images.iter().map(|p| p.to_path_buf()).collect();
        let count = u32::try_from(owned.len().max(1)).unwrap_or(u32::MAX);
        let per_image = Self::stage_budget("FORWARD", DEFAULT_FORWARD_STAGE_BUDGET_MS);
        let budget = per_image
            .checked_mul(count)
            .unwrap_or_else(|| Duration::from_secs(u64::MAX / 2));
        self.run_blocking_stage_with_budget("forward-batch", budget, move || {
            let refs: Vec<&Path> = owned.iter().map(std::path::PathBuf::as_path).collect();
            Ok(model.recognize_batch(&refs))
        })
    }

    /// Multi-page CROSS-PAGE document parsing (bd-1gv.25) — the reference
    /// `infer_multi` contract: one 32K pass where page N attends to pages
    /// 1..N−1 (OQ-13), returning ONE assembled markdown with `<PAGE>`
    /// separators. This is NOT [`OcrEngine::recognize_batch`] (independent
    /// pages); use this when the pages form one document whose later pages
    /// reference earlier content.
    ///
    /// # Errors
    /// As [`crate::native_engine::OcrModel::recognize_multi_page`] — notably
    /// `NotImplemented` for non-Unlimited-OCR artifacts and an actionable
    /// error when the assembled prefix exceeds the 32K position budget.
    pub fn recognize_multi_page(&self, images: &[&Path]) -> FocrResult<String> {
        self.recognize_multi_page_with_model(&Self::model_path(), images)
    }

    /// Path-explicit form of [`OcrEngine::recognize_multi_page`].
    ///
    /// # Errors
    /// As [`OcrEngine::recognize_multi_page`].
    pub fn recognize_multi_page_with_model(
        &self,
        model_path: &Path,
        images: &[&Path],
    ) -> FocrResult<String> {
        let model = self.model_at(model_path)?;
        let owned: Vec<std::path::PathBuf> = images.iter().map(|p| p.to_path_buf()).collect();
        let count = u32::try_from(owned.len().max(1)).unwrap_or(u32::MAX);
        let per_image = Self::stage_budget("FORWARD", DEFAULT_FORWARD_STAGE_BUDGET_MS);
        let budget = per_image
            .checked_mul(count)
            .unwrap_or_else(|| Duration::from_secs(u64::MAX / 2));
        self.run_blocking_stage_with_budget("forward-multi-page", budget, move || {
            let refs: Vec<&Path> = owned.iter().map(std::path::PathBuf::as_path).collect();
            model.recognize_multi_page(&refs)
        })
    }

    /// [`OcrEngine::recognize_multi_page`] over in-memory images (the PDF
    /// rasterizer's entry — pages never touch disk).
    ///
    /// # Errors
    /// As [`OcrEngine::recognize_multi_page`].
    pub fn recognize_multi_page_dynamic(
        &self,
        images: Vec<image::DynamicImage>,
    ) -> FocrResult<String> {
        self.recognize_multi_page_dynamic_with_model(&Self::model_path(), images)
    }

    /// Path-explicit form of [`OcrEngine::recognize_multi_page_dynamic`].
    ///
    /// # Errors
    /// As [`OcrEngine::recognize_multi_page`].
    pub fn recognize_multi_page_dynamic_with_model(
        &self,
        model_path: &Path,
        images: Vec<image::DynamicImage>,
    ) -> FocrResult<String> {
        let model = self.model_at(model_path)?;
        let count = u32::try_from(images.len().max(1)).unwrap_or(u32::MAX);
        let per_image = Self::stage_budget("FORWARD", DEFAULT_FORWARD_STAGE_BUDGET_MS);
        let budget = per_image
            .checked_mul(count)
            .unwrap_or_else(|| Duration::from_secs(u64::MAX / 2));
        self.run_blocking_stage_with_budget("forward-multi-page", budget, move || {
            model.recognize_multi_page_dynamic(images)
        })
    }

    /// [`OcrEngine::recognize_multi_page_dynamic`] with a PER-PAGE STREAMING
    /// sink (bd-2z0y): `on_page(k, body)` fires from the decode driver as
    /// page `k`'s `<PAGE>` boundary is crossed in the token stream (boxed +
    /// `Send` because the pass runs on the blocking pool). The returned
    /// markdown is still the full terminal assembly.
    ///
    /// # Errors
    /// As [`OcrEngine::recognize_multi_page`].
    pub fn recognize_multi_page_dynamic_streaming_with_model(
        &self,
        model_path: &Path,
        images: Vec<image::DynamicImage>,
        mut on_page: PageSink,
    ) -> FocrResult<String> {
        let model = self.model_at(model_path)?;
        let count = u32::try_from(images.len().max(1)).unwrap_or(u32::MAX);
        let per_image = Self::stage_budget("FORWARD", DEFAULT_FORWARD_STAGE_BUDGET_MS);
        let budget = per_image
            .checked_mul(count)
            .unwrap_or_else(|| Duration::from_secs(u64::MAX / 2));
        self.run_blocking_stage_with_budget("forward-multi-page", budget, move || {
            model.recognize_multi_page_dynamic_streaming(images, &mut *on_page)
        })
    }

    fn stage_budget(stage: &str, default_ms: u64) -> Duration {
        let key = format!("FOCR_STAGE_BUDGET_{stage}_MS");
        let millis = std::env::var(&key)
            .ok()
            .and_then(|raw| raw.parse::<u64>().ok())
            .filter(|&ms| ms > 0)
            .unwrap_or(default_ms);
        Duration::from_millis(millis)
    }

    fn run_blocking_stage_with_budget<T, F>(
        &self,
        stage: &'static str,
        budget: Duration,
        op: F,
    ) -> FocrResult<T>
    where
        T: Send + 'static,
        F: FnOnce() -> FocrResult<T> + Send + 'static,
    {
        self.runtime.block_on(async move {
            match asupersync::time::timeout(
                asupersync::time::wall_now(),
                budget,
                asupersync::runtime::spawn_blocking(op),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => Err(FocrError::Timeout(format!(
                    "{stage} stage exceeded {}ms budget",
                    budget.as_millis()
                ))),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log_line(test: &str, phase: &str, outcome: &str, extra: &str) {
        eprintln!(
            "{{\"test\":\"{test}\",\"phase\":\"{phase}\",\"outcome\":\"{outcome}\"{}{extra}}}",
            if extra.is_empty() { "" } else { "," }
        );
    }

    /// bd-223.2: dropping the engine shuts down its owned runtime and pools.
    #[test]
    fn engine_owns_single_runtime_and_drops_clean() {
        let engine = OcrEngine::new().expect("engine builds");
        // Capture the weak handle installed on an actual runtime worker. It
        // observes this engine's runtime lifecycle without inspecting other
        // tests' process-wide threads.
        let runtime_witness = engine
            .runtime
            .block_on(engine.runtime.handle().spawn(async {
                Runtime::current_handle().expect("spawned task has a runtime handle")
            }));

        // The runtime is live: a trivial blocking stage round-trips.
        let out = engine
            .run_blocking_stage_with_budget("drop-probe", Duration::from_secs(5), || Ok(42u8))
            .expect("stage runs");
        assert_eq!(out, 42);
        log_line(
            "engine_owns_single_runtime_and_drops_clean",
            "live",
            "pass",
            "",
        );

        drop(engine); // joins the runtime workers

        assert!(
            matches!(
                runtime_witness.try_spawn(async { 42u8 }),
                Err(asupersync::runtime::state::SpawnError::RuntimeUnavailable)
            ),
            "dropped engine runtime should reject new tasks"
        );
        assert!(
            runtime_witness.spawn_blocking(|| {}).is_none(),
            "dropped engine runtime should not accept blocking tasks"
        );
        assert!(
            runtime_witness.blocking_handle().is_none(),
            "dropped engine runtime should not expose its blocking pool"
        );
        log_line(
            "engine_owns_single_runtime_and_drops_clean",
            "dropped",
            "pass",
            "",
        );
    }

    /// bd-223.2: the checkpoint aborts a decode-style loop with Cancelled
    /// (exit 6) — the flag is observed INSIDE the spawn_blocking closure.
    #[test]
    fn cancellation_token_into_closure_aborts() {
        reset_shutdown();
        let engine = OcrEngine::new().expect("engine builds");
        let out =
            engine.run_blocking_stage_with_budget("cancel-probe", Duration::from_secs(10), || {
                for step in 0..1_000_000u64 {
                    cancel_checkpoint()?;
                    if step == 3 {
                        // The "Ctrl+C" arrives mid-loop, from inside the
                        // closure's world — exactly the cooperative contract.
                        request_shutdown();
                    }
                }
                Ok(0u64)
            });
        reset_shutdown();
        assert!(
            matches!(out, Err(FocrError::Cancelled)),
            "expected Cancelled, got {out:?}"
        );
        assert_eq!(FocrError::Cancelled.exit_code(), 6, "exit code contract");
        log_line(
            "cancellation_token_into_closure_aborts",
            "aborted",
            "pass",
            "",
        );
    }

    /// bd-223.2: the bounded channel BLOCKS the producer when the consumer
    /// lags (backpressure) — in-flight items never exceed capacity + 1.
    #[test]
    fn bounded_stream_backpressure() {
        use std::sync::atomic::{AtomicI64, Ordering};
        static IN_FLIGHT: AtomicI64 = AtomicI64::new(0);
        static MAX_SEEN: AtomicI64 = AtomicI64::new(0);
        IN_FLIGHT.store(0, Ordering::SeqCst);
        MAX_SEEN.store(0, Ordering::SeqCst);
        let total = 24u32;
        let mut produced = 0u32;
        let n = stream_pages(
            2,
            move || {
                if produced == total {
                    return Ok(None);
                }
                produced += 1;
                let now = IN_FLIGHT.fetch_add(1, Ordering::SeqCst) + 1;
                MAX_SEEN.fetch_max(now, Ordering::SeqCst);
                Ok(Some(produced))
            },
            |item: u32| {
                // Slow consumer: the producer must stall on the bounded send.
                std::thread::sleep(Duration::from_millis(5));
                IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
                let _ = item;
            },
        )
        .expect("stream completes");
        assert_eq!(n, total as usize, "every page delivered in order");
        let max = MAX_SEEN.load(Ordering::SeqCst);
        assert!(
            max <= 4,
            "in-flight items {max} exceeded capacity(2)+channel slack — backpressure broken"
        );
        log_line(
            "bounded_stream_backpressure",
            "drained",
            "pass",
            &format!("\"max_in_flight\":{max},\"n\":{n}"),
        );
    }

    /// bd-223.2 addendum: FOCR_THREADS wins, else the PHYSICAL core count.
    #[test]
    fn thread_budget_reads_env_then_physical() {
        // The OnceLock latches per process; assert the resolved value is
        // consistent with the environment THIS process started with.
        let budget = thread_budget();
        match std::env::var("FOCR_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            Some(env) if env > 0 => assert_eq!(budget, env, "env wins"),
            _ => assert_eq!(budget, num_cpus::get_physical(), "physical cores"),
        }
        assert!(budget > 0);
        assert!(
            budget
                <= std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(usize::MAX),
            "physical budget cannot exceed logical width"
        );
        log_line(
            "thread_budget_reads_env_then_physical",
            "resolved",
            "pass",
            &format!("\"threads\":{budget}"),
        );
    }

    struct TempModel(std::path::PathBuf);

    impl TempModel {
        fn write_focrq(bytes: &[u8]) -> std::io::Result<Self> {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default();
            let path = std::env::temp_dir().join(format!(
                "franken_ocr_engine_format_mismatch_{}_{}.focrq",
                std::process::id(),
                nanos
            ));
            std::fs::write(&path, bytes)?;
            Ok(Self(path))
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempModel {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn future_focrq_preamble() -> Vec<u8> {
        let mut blob = Vec::new();
        blob.extend_from_slice(native_engine::weights::FOCRQ_MAGIC);
        blob.extend_from_slice(&(native_engine::weights::FOCRQ_FORMAT_VERSION + 1).to_le_bytes());
        blob.push(0);
        blob.extend_from_slice(&[0u8; 32]);
        blob.extend_from_slice(&0u64.to_le_bytes());
        blob
    }

    /// The engine constructs (its single owned runtime builds) without touching
    /// the model blob — construction is cheap and lazy.
    #[test]
    fn engine_constructs_without_model() {
        let engine = OcrEngine::new().expect("runtime builds");
        // Constructing alone must not have loaded a model.
        assert!(
            engine.model_guard().expect("mutex").is_none(),
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
        // Only assert when the env override is unset AND the engine's own
        // resolver finds nothing — the normal CI condition. Checking the
        // repo-relative default alone is NOT enough: since bd-3u6x the
        // default spec also resolves via the user cache
        // (~/.cache/franken_ocr/models/unlimited-ocr.int8.focrq), so a dev
        // box with a pulled artifact must skip rather than misfire.
        if std::env::var_os(MODEL_PATH_ENV).is_none()
            && !std::path::Path::new(DEFAULT_MODEL_PATH).exists()
            && native_engine::OcrModel::resolve_model(Path::new(DEFAULT_MODEL_PATH)).is_err()
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

    #[test]
    fn blocking_stage_runs_on_runtime_blocking_pool() {
        let engine = OcrEngine::new().expect("runtime builds");
        let thread_name = engine
            .run_blocking_stage_with_budget("test", Duration::from_secs(1), || {
                let thread = std::thread::current();
                Ok(thread.name().unwrap_or("<unnamed>").to_string())
            })
            .expect("stage should complete");
        assert!(
            thread_name.contains("-blocking-"),
            "stage ran on {thread_name:?}, not the runtime blocking pool"
        );
    }

    #[test]
    fn blocking_stage_timeout_maps_to_stable_error() {
        let engine = OcrEngine::new().expect("runtime builds");
        let started = std::time::Instant::now();
        let err = engine
            .run_blocking_stage_with_budget("test-timeout", Duration::from_millis(10), || {
                std::thread::sleep(Duration::from_millis(100));
                Ok(())
            })
            .expect_err("slow blocking stage must time out");
        assert!(
            matches!(err, FocrError::Timeout(_)),
            "expected Timeout, got {err:?}"
        );
        assert_eq!(err.exit_code(), error::EXIT_TIMEOUT);
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "timeout wrapper waited for the whole blocking closure"
        );
    }

    /// A recognized but too-new `.focrq` must stay a public FormatMismatch
    /// through the engine entrypoint and robot event, not be collapsed into the
    /// Phase-0 generic/NotImplemented resolver path.
    #[test]
    fn public_engine_preserves_focrq_format_mismatch_robot_code()
    -> Result<(), Box<dyn std::error::Error>> {
        let model = TempModel::write_focrq(&future_focrq_preamble())?;
        let engine = OcrEngine::new()?;
        let result = engine.recognize_with_model(model.path(), Path::new("/some/document.png"));
        let Err(err) = result else {
            return Err(std::io::Error::other(
                "future .focrq version unexpectedly succeeded before forward",
            )
            .into());
        };

        assert!(
            matches!(err, FocrError::FormatMismatch(_)),
            "expected FormatMismatch, got {err:?}"
        );
        assert_eq!(err.exit_code(), error::EXIT_FORMAT_MISMATCH);

        let event = robot::run_error_event(&err);
        assert_eq!(event["event"], "run_error");
        assert_eq!(event["error_kind"], "format_mismatch");
        assert_eq!(event["code"], error::EXIT_FORMAT_MISMATCH);
        Ok(())
    }
}
