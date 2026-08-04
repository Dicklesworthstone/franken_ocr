//! The `focr` clap-derive CLI surface (plan §7.2).
//!
//! Subcommands are Phase-0 skeleton stubs: the diagnostics (`robot
//! schema/health/backends`) work today; `ocr` routes through the native model
//! resolver/engine skeleton and then fails cleanly at the first unimplemented
//! stage, while `convert` and `doctor` return clear `NotImplemented` errors
//! pointing at the plan phase that lands them. PDF input is handled natively:
//! `focr ocr file.pdf` rasterizes each page (the pure-Rust [`crate::pdf`] scanned-
//! image fast path) and feeds it through the same pipeline an image takes.
//!
//! This module lives in the **library** so the single CLI entrypoint
//! ([`cli_main`]) is shared by both binaries (`focr` and `franken_ocr`) without
//! either `src/main.rs` appearing in two build targets — each `[[bin]]` now
//! points at its own thin shim that just calls [`cli_main`]. See AGENTS.md
//! doctrine #9.

use crate::{
    FOCR_MODEL_LICENSE_NOTICE, FOCR_PROJECT_LICENSE_NOTICE, FocrError, FocrResult, OcrEngine, dist,
    native_engine, pdf, progress, quant, robot, simd,
};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

// Debug/test-only producer seam for process-level exit-code conformance while
// the Phase-1 forward is not able to naturally emit every terminal error kind.
#[cfg(debug_assertions)]
const FORCE_TEST_ERROR_ENV: &str = "FOCR_TEST_FORCE_ERROR";

const DEFAULT_BASE_SIZE: i64 = 1024;
const DEFAULT_IMAGE_SIZE: i64 = 640;
const DEFAULT_MAX_LENGTH: i64 = 32_768;
const DEFAULT_TEMPERATURE: f32 = 0.0;
const DEFAULT_NO_REPEAT_NGRAM: i64 = 35;
const DEFAULT_NGRAM_WINDOW: i64 = 128;

/// The shared process entrypoint for both binaries (`focr` and `franken_ocr`).
///
/// `fn main()` in each shim is **synchronous by design** (plan §3.3, §7.1): the
/// asupersync runtime is owned BELOW here, inside `OcrEngine`, never spanning
/// the whole process. This parses, dispatches, and maps errors to the stable
/// exit codes documented in [`crate::error`].
pub fn cli_main() -> ExitCode {
    if is_exact_long_version_request(std::env::args_os()) {
        print!("{}", long_version_report());
        return ExitCode::SUCCESS;
    }

    // Cooperative Ctrl+C (bd-223.2): the first signal requests shutdown —
    // every decode-step/page checkpoint aborts with Cancelled (exit 6) at its
    // next boundary; a second signal hard-exits 130 (the shell convention)
    // for a wedged stage. Installation failure (e.g. no signal handling in
    // odd sandboxes) is non-fatal: the engine still works, just without
    // graceful interrupt.
    let _ = ctrlc::set_handler(|| {
        if crate::shutdown_requested() {
            std::process::exit(130);
        }
        crate::request_shutdown();
        progress::suppress_for_interrupt();
        let _ = std::thread::Builder::new()
            .name("focr-interrupt-notice".into())
            .spawn(|| {
                progress::stderr_message(format_args!(
                    "focr: interrupt received — finishing the current step then aborting \
                     (Ctrl+C again to force)"
                ));
            });
    });

    let cli = Cli::parse();
    let error_mode = ErrorMode::from_cli(&cli);
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => exit_code_from_error(&err, error_mode),
    }
}

#[derive(Parser)]
#[command(
    name = "focr",
    version,
    about = "Pure-Rust, CPU-hyper-optimized runner for the Baidu Unlimited-OCR model"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

fn is_exact_long_version_request<I>(mut args: I) -> bool
where
    I: Iterator<Item = OsString>,
{
    let _program = args.next();
    matches!(
        (args.next().as_deref(), args.next()),
        (Some(arg), None) if arg == "--version"
    )
}

/// Long, attribution-bearing version report. The short `-V` surface remains
/// Clap's script-friendly `focr <semver>` output. The model-license notice is
/// read from the default model's [`crate::model_arch::ModelArch`] descriptor (the
/// source of truth as the model zoo grows); today that is byte-identical to
/// [`FOCR_MODEL_LICENSE_NOTICE`].
#[must_use]
pub fn long_version_report() -> String {
    format!(
        "focr {}\nsource_license: {}\nmodel_license: {}\n",
        env!("CARGO_PKG_VERSION"),
        FOCR_PROJECT_LICENSE_NOTICE,
        crate::model_arch::default_arch().license_notice()
    )
}

#[derive(Subcommand)]
pub enum Command {
    /// Parse a document image into structured markdown (or `--json`).
    ///
    /// `-o FILE` writes the result to a file instead of stdout: a `.json` path
    /// emits structured JSON (markdown + per-span bounding boxes), any other
    /// extension (e.g. `.md`) emits markdown; `--json` forces JSON.
    /// `--extract-figures` additionally saves figure/image regions the model does
    /// not transcribe into a subfolder, referenced from the markdown/JSON.
    Ocr(OcrArgs),
    /// OCR many images in ONE process — load the 6.2 GB weights + build the int8
    /// decoder cache ONCE, then stream a result per image (the throughput path).
    OcrBatch(OcrBatchArgs),
    /// Offline weight transformation: safetensors → `.focrq` (plan §5).
    Convert(ConvertArgs),
    /// Download the model weights (int8 `.focrq` + tokenizer) into the cache.
    Pull(PullArgs),
    /// List the models this build can run (the "model zoo"): id, tasks, status.
    Models(ModelsArgs),
    /// Agent-facing diagnostics and the machine contract.
    Robot {
        #[command(subcommand)]
        cmd: RobotCmd,
    },
    /// Query durable run history (fsqlite-backed store lands with plan §7.2).
    Runs(RunsArgs),
    /// Export/import the append-only run audit stream.
    Sync(SyncArgs),
    /// Idempotent self-check / repair.
    Doctor(DoctorArgs),
}

#[derive(Clone, Debug, Args)]
pub struct OcrArgs {
    #[command(flatten)]
    pub request: OcrRequestArgs,
    /// Emit machine-readable JSON instead of human markdown.
    #[arg(long)]
    pub json: bool,
    /// Write the result to FILE instead of stdout. The format follows the
    /// extension: `.json` emits structured JSON (markdown + bounding boxes),
    /// any other extension (e.g. `.md`) emits markdown. `--json` forces JSON.
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,
    /// Save figure/image regions the model sees but does not transcribe to a
    /// subfolder (default `<output-stem>_figures/`), referenced from the
    /// markdown/JSON. Each figure is a PNG (line-art) or JPG (photo) chosen by
    /// content. Requires `-o` (or `--figures-dir` for a stdout run).
    #[arg(long)]
    pub extract_figures: bool,
    /// Directory to save extracted figures into (implies `--extract-figures`).
    /// Relative paths are taken relative to the output file's directory (or the
    /// current directory for a stdout run) and used verbatim in references.
    #[arg(long, value_name = "DIR")]
    pub figures_dir: Option<PathBuf>,
    /// Stream NDJSON robot events as pages complete.
    #[arg(long)]
    pub robot: bool,
}

#[derive(Clone, Debug, Args)]
pub struct OcrBatchArgs {
    /// Input document image paths. The model + int8 decoder cache are built once
    /// and reused across all of them (load-once batch throughput).
    #[arg(required = true)]
    pub images: Vec<PathBuf>,
    /// Model artifact path. Default (unset) = the FAST plain-text OCR model
    /// (unlimited-ocr). Pass a `got-ocr2.int8.focrq` for SPECIALIZED structured
    /// output — math (LaTeX), tables, charts, molecular, geometry, sheet music
    /// (heavier per page). `focr pull got-ocr2` first; see `focr models`.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// Emit machine-readable JSON (one object per image + a final summary).
    #[arg(long)]
    pub json: bool,
    /// Use the experimental all-int8 decoder. This also requires
    /// `FOCR_INT8_ATTN=1` and `FOCR_INT8_LMHEAD=1`; the default follows the
    /// conservative recipe and keeps attention plus `lm_head` high precision.
    #[arg(long)]
    pub experimental_full_int8: bool,
    /// Treat the images as ONE multi-page document (the Unlimited-OCR
    /// `infer_multi` contract): a single cross-page pass where page N can
    /// reference pages 1..N-1, emitting one markdown with `<PAGE>` separators.
    /// Without this flag each image is parsed independently. Unlimited-OCR
    /// only; the whole document must fit the 32K context.
    #[arg(long)]
    pub multi_page: bool,
}

#[derive(Clone, Debug, Args)]
pub struct ModelsArgs {
    /// Emit a machine-readable JSON list instead of a human table.
    #[arg(long)]
    pub json: bool,
}

#[derive(Clone, Debug, Args)]
pub struct PullArgs {
    /// Model id to fetch (e.g. `got-ocr2`). Defaults to the manifest's primary
    /// model (`unlimited-ocr`).
    pub model: Option<String>,
    /// Quant tag to fetch (defaults to `int8`; available tags vary by model).
    #[arg(long, default_value = dist::DEFAULT_QUANT)]
    pub quant: String,
    /// Manifest source — a local path or an HTTPS URL. Defaults to
    /// `$FOCR_MANIFEST_URL`, else the release-bound embedded manifest.
    #[arg(long)]
    pub manifest: Option<String>,
    /// Emit a single JSON result object instead of human progress lines.
    #[arg(long)]
    pub json: bool,
}

#[derive(Clone, Debug, Args)]
pub struct RobotRunArgs {
    #[command(flatten)]
    pub request: OcrRequestArgs,
}

#[derive(Clone, Debug, Args)]
pub struct OcrRequestArgs {
    /// Input document path — an image (PNG/JPG/…) or a PDF (each page is
    /// rasterized natively and OCR'd as one document).
    pub image: PathBuf,
    /// Explicit model artifact path for diagnostics and model-gated runs.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// Reference global-view size from `infer(..., base_size=1024)`.
    #[arg(long, default_value_t = DEFAULT_BASE_SIZE)]
    pub base_size: i64,
    /// Reference local tile size from `infer(..., image_size=640)`.
    #[arg(long, default_value_t = DEFAULT_IMAGE_SIZE)]
    pub image_size: i64,
    /// Vision preprocessing mode (unlimited-ocr only — got-ocr2 always uses its
    /// own fixed squash-1024 preprocess). `base` (the default) is the certified
    /// single 1024-pixel global view — the mode every oracle cert and golden
    /// was produced under, and what the engine has always actually run.
    /// `gundam` selects the reference dynamic-resolution tiling (a 1024 global
    /// view plus 640 local tiles); its connector path is unit-tested but has no
    /// e2e oracle certification yet. (Until bd-1e9n this flag was parsed and
    /// silently dropped with a `gundam` default label the engine never honored;
    /// the default now states the real, certified behavior.)
    #[arg(long, value_enum, default_value_t = CropMode::Base)]
    pub crop_mode: CropMode,
    /// Maximum generated sequence length.
    #[arg(long, default_value_t = DEFAULT_MAX_LENGTH)]
    pub max_length: i64,
    /// Decode temperature; 0.0 means greedy. (unlimited-ocr only — got-ocr2
    /// decodes greedy.)
    #[arg(long, default_value_t = DEFAULT_TEMPERATURE)]
    pub temperature: f32,
    /// No-repeat n-gram size (env override: FOCR_NO_REPEAT_NGRAM). For
    /// unlimited-ocr this is the sliding-window guard (with --ngram-window);
    /// for got-ocr2 it overrides the model's global guard (default 20; 0
    /// disables).
    #[arg(
        long,
        env = "FOCR_NO_REPEAT_NGRAM",
        default_value_t = DEFAULT_NO_REPEAT_NGRAM
    )]
    pub no_repeat_ngram: i64,
    /// Sliding no-repeat n-gram lookback window. (unlimited-ocr only —
    /// got-ocr2's guard is global.)
    #[arg(long, default_value_t = DEFAULT_NGRAM_WINDOW)]
    pub ngram_window: i64,
    /// GOT-OCR2 structured output: use the model's `OCR with format:` mode instead
    /// of plain text, emitting Mathpix-Markdown (.mmd) — inline LaTeX math, Markdown
    /// tables, TikZ geometry, SMILES molecules, and `**kern` sheet music (the model
    /// auto-selects the formalism from the image). Only affects the `got-ocr2` model
    /// (`--model got-ocr2…`); a no-op for the default unlimited-ocr model.
    #[arg(long)]
    pub format: bool,
    /// Task selector — convenience routing over the model zoo (`focr models`).
    /// `ocr` (the default) is today's behavior, unchanged. The specialized tasks
    /// are served by got-ocr2's `OCR with format:` mode, so they imply `--format`
    /// (an explicit `--format` composes idempotently) and need a got-ocr2 model:
    /// `focr pull got-ocr2`, then `--model got-ocr2.int8.focrq`. `describe`
    /// (photo description / VQA) is served by smolvlm2: `--model
    /// smolvlm2.int8.focrq --task describe [--question "…"]`.
    #[arg(long, value_enum, default_value_t = OcrTask::Ocr)]
    pub task: OcrTask,
    /// The natural-language question for `--task describe` (smolvlm2 VQA) —
    /// SmolVLM2 has no instruction modes; the task IS the question. Defaults
    /// to the model-card caption prompt ("Can you describe this image?").
    /// Requires `--task describe`.
    #[arg(long)]
    pub question: Option<String>,
    /// PDF page selection: a comma list of 1-based pages and inclusive
    /// ranges ("3", "3-7", "1,5-9,218"). PDF inputs only — on an image
    /// input this is a usage error. Out-of-range pages error naming the
    /// document's page count; pages run in source order, deduplicated.
    #[arg(long)]
    pub pages: Option<String>,
    /// Split two-page book spreads (PDF inputs only): when a rasterized page
    /// is much wider than tall AND a near-blank vertical gutter sits near the
    /// center, OCR the left and right halves as separate logical pages
    /// (labelled `"half": "left"|"right"` in JSON / robot page events). Off
    /// by default; a page with no detectable gutter passes through unsplit.
    #[arg(long)]
    pub split_spreads: bool,
    /// ONE cross-page pass over the whole document (PDF inputs only; the
    /// Unlimited-OCR `infer_multi` contract): page N can reference pages
    /// 1..N-1, output is one markdown with `<PAGE>` separators. Composes
    /// with `--pages`. The document must fit the 32K context (~290 pages).
    /// Without this flag each page is parsed independently.
    #[arg(long)]
    pub multi_page: bool,
}

#[derive(Clone, Debug)]
pub struct OcrRequest {
    pub image: PathBuf,
    pub model: Option<PathBuf>,
    pub base_size: u32,
    pub image_size: u32,
    pub crop_mode: CropMode,
    pub max_length: u32,
    pub temperature: f32,
    pub no_repeat_ngram: u32,
    pub ngram_window: u32,
    pub format: bool,
    pub question: Option<String>,
    pub pages: Option<String>,
    pub split_spreads: bool,
    pub multi_page: bool,
}

impl OcrArgs {
    fn to_request(&self) -> FocrResult<OcrRequest> {
        self.request.to_request()
    }
}

impl RobotRunArgs {
    fn into_ocr_args(self) -> OcrArgs {
        OcrArgs {
            request: self.request,
            json: false,
            output: None,
            extract_figures: false,
            figures_dir: None,
            robot: true,
        }
    }
}

/// Map the request's preprocess tuning flags onto engine
/// [`native_engine::PreprocessOverrides`] (bd-1e9n), with the same
/// explicit-only rule as [`decode_overrides_from`] for the sizes. `--crop-mode`
/// is a two-value enum whose `base` default IS the engine default, so only
/// `gundam` produces an override.
fn preprocess_overrides_from(request: &OcrRequest) -> native_engine::PreprocessOverrides {
    native_engine::PreprocessOverrides {
        base_size: (i64::from(request.base_size) != DEFAULT_BASE_SIZE)
            .then_some(request.base_size as usize),
        image_size: (i64::from(request.image_size) != DEFAULT_IMAGE_SIZE)
            .then_some(request.image_size as usize),
        gundam: matches!(request.crop_mode, CropMode::Gundam).then_some(true),
    }
}

/// Map the request's decode tuning flags onto engine
/// [`native_engine::DecodeOverrides`]. A value becomes an override only when it
/// differs from the compiled default (bit-exact for the float), so an untouched
/// flag keeps the engine-side default AND leaves env overrides (e.g.
/// `FOCR_MAX_NEW_TOKENS`) in force.
fn decode_overrides_from(request: &OcrRequest) -> native_engine::DecodeOverrides {
    native_engine::DecodeOverrides {
        max_length: (i64::from(request.max_length) != DEFAULT_MAX_LENGTH)
            .then_some(request.max_length as usize),
        temperature: (request.temperature.to_bits() != DEFAULT_TEMPERATURE.to_bits())
            .then_some(request.temperature),
        no_repeat_ngram: (i64::from(request.no_repeat_ngram) != DEFAULT_NO_REPEAT_NGRAM)
            .then_some(request.no_repeat_ngram as usize),
        ngram_window: (i64::from(request.ngram_window) != DEFAULT_NGRAM_WINDOW)
            .then_some(request.ngram_window as usize),
    }
}

impl OcrRequestArgs {
    fn to_request(&self) -> FocrResult<OcrRequest> {
        validate_task_selection(self.task, self.effective_model_spec().as_deref())?;
        if self.question.is_some() && self.task != OcrTask::Describe {
            return Err(FocrError::Usage(
                "--question is the smolvlm2 VQA prompt and requires --task describe".into(),
            ));
        }
        Ok(OcrRequest {
            image: self.image.clone(),
            model: self.model.clone(),
            base_size: positive_u32("base-size", self.base_size)?,
            image_size: positive_u32("image-size", self.image_size)?,
            crop_mode: self.crop_mode,
            max_length: positive_u32("max-length", self.max_length)?,
            temperature: non_negative_finite_f32("temperature", self.temperature)?,
            no_repeat_ngram: non_negative_u32("no-repeat-ngram", self.no_repeat_ngram)?,
            ngram_window: non_negative_u32("ngram-window", self.ngram_window)?,
            // `--format` and a format-implying `--task` compose OR-wise: an
            // explicit `--format` wins / is idempotent alongside `--task`.
            format: self.format || self.task.implies_got_format(),
            question: self.question.clone(),
            pages: self.pages.clone(),
            split_spreads: self.split_spreads,
            multi_page: self.multi_page,
        })
    }

    /// The model spec this run would use, for CLI-level `--task` guidance ONLY:
    /// the explicit `--model`, else the `FOCR_MODEL_PATH` env override, else
    /// `None` — the engine's default resolution, which is always an
    /// unlimited-ocr artifact (bd-3u6x).
    fn effective_model_spec(&self) -> Option<PathBuf> {
        self.model
            .clone()
            .or_else(|| std::env::var_os(crate::MODEL_PATH_ENV).map(PathBuf::from))
    }
}

/// CLI-level `--task` feasibility check (best-effort: no model file is opened
/// here — the engine's `.focrq` arch tag stays the real dispatcher).
///
/// * `describe` (smolvlm2, C9) whose model spec is KNOWABLY not smolvlm2 gets
///   the pull/model guidance now, before any weights load.
/// * a got-only task whose model spec is KNOWABLY not got-ocr2 (see
///   [`model_spec_is_knowably_not_got`]) gets the pull/model guidance now.
/// * an ambiguous explicit spec passes through: mislabeling would reject real
///   artifacts, and the engine's arch tag makes the final call.
fn validate_task_selection(task: OcrTask, model_spec: Option<&Path>) -> FocrResult<()> {
    if task == OcrTask::ChartData && model_spec_is_knowably_not_onechart(model_spec) {
        return Err(FocrError::Usage(
            "--task chart-data (chart→dict + number-head self-verify) needs the onechart \
             model, but this run would use a different model. Re-run with \
             `--model onechart.int8.focrq` (see `focr models`)"
                .into(),
        ));
    }
    if task == OcrTask::Describe && model_spec_is_knowably_not_smolvlm2(model_spec) {
        return Err(FocrError::Usage(
            "--task describe (photo description/VQA) needs the smolvlm2 model, but this \
             run would use a different model. Re-run with `--model smolvlm2.int8.focrq` \
             (see `focr models`)"
                .into(),
        ));
    }
    if task == OcrTask::Music {
        // Music is served by TWO lanes: tromr (native OMR -> MusicXML, the
        // specialist) and got-ocr2 (sheet-music format mode). Reject only
        // when the model is knowably NEITHER.
        if model_spec_is_knowably_not_got(model_spec)
            && model_spec_is_knowably_not_tromr(model_spec)
        {
            return Err(FocrError::Usage(
                "--task music needs the tromr (native OMR -> MusicXML) or got-ocr2 \
                 (sheet-music format mode) model, but this run would use a different \
                 model. Re-run with `--model tromr.focrq` (see `focr models`)"
                    .into(),
            ));
        }
        return Ok(());
    }
    if task.implies_got_format() && model_spec_is_knowably_not_got(model_spec) {
        return Err(FocrError::Usage(format!(
            "--task {task} needs the got-ocr2 model, but this run would use the plain-text \
             unlimited-ocr model. Run `focr pull got-ocr2`, then re-run with \
             `--model got-ocr2.int8.focrq` (see `focr models`)"
        )));
    }
    Ok(())
}

/// True when the model spec is KNOWABLY not a smolvlm2 artifact: no spec at
/// all (the default resolution is always unlimited-ocr) or a file name naming
/// another family without `smolvlm`. An ambiguous name passes through to the
/// engine's arch tag.
/// True when the model spec is KNOWABLY not a onechart artifact (mirrors
/// [`model_spec_is_knowably_not_smolvlm2`]).
fn model_spec_is_knowably_not_onechart(spec: Option<&Path>) -> bool {
    let Some(path) = spec else {
        return true;
    };
    let Some(name) = path.file_name() else {
        return false;
    };
    let name = name.to_string_lossy().to_ascii_lowercase();
    !name.contains("onechart")
        && (name.contains("unlimited") || name.contains("got") || name.contains("smolvlm"))
}

/// True when the model spec is KNOWABLY not a tromr artifact (mirrors
/// [`model_spec_is_knowably_not_smolvlm2`]).
fn model_spec_is_knowably_not_tromr(spec: Option<&Path>) -> bool {
    let Some(path) = spec else {
        return true;
    };
    let Some(name) = path.file_name() else {
        return false;
    };
    let name = name.to_string_lossy().to_ascii_lowercase();
    !name.contains("tromr")
        && (name.contains("unlimited")
            || name.contains("got")
            || name.contains("smolvlm")
            || name.contains("onechart"))
}

fn model_spec_is_knowably_not_smolvlm2(spec: Option<&Path>) -> bool {
    let Some(path) = spec else {
        return true;
    };
    let Some(name) = path.file_name() else {
        return false;
    };
    let name = name.to_string_lossy().to_ascii_lowercase();
    !name.contains("smolvlm") && (name.contains("unlimited") || name.contains("got"))
}

/// True when the model spec is KNOWABLY not a got-ocr2 artifact: no spec at all
/// (the default resolution is always unlimited-ocr) or a file name carrying
/// `unlimited`. A name carrying `got` — or naming neither family — passes.
fn model_spec_is_knowably_not_got(spec: Option<&Path>) -> bool {
    let Some(path) = spec else {
        return true;
    };
    let Some(name) = path.file_name() else {
        return false;
    };
    let name = name.to_string_lossy().to_ascii_lowercase();
    name.contains("unlimited") && !name.contains("got")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum CropMode {
    /// Reference dynamic-resolution tiling (`crop_mode=true`).
    Gundam,
    /// Single global view (`crop_mode=false`).
    Base,
}

impl std::fmt::Display for CropMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Gundam => "gundam",
            Self::Base => "base",
        })
    }
}

/// `--task` selector: route a run to the model/mode serving that task (the
/// model-zoo convenience surface, bd-3jo6.1.5). Values mirror the registry's
/// task names (`focr models`); the engine's `.focrq` arch tag stays the real
/// dispatcher — this only picks the prompt mode and validates the combination.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum OcrTask {
    /// Plain document text → markdown (the default; today's behavior).
    Ocr,
    /// Math / formulas → LaTeX (got-ocr2; implies `--format`).
    Formula,
    /// Tables → structured markdown (got-ocr2; implies `--format`).
    Tables,
    /// Charts → structured output (got-ocr2; implies `--format`).
    Chart,
    /// Molecular structures → SMILES (got-ocr2; implies `--format`).
    Molecular,
    /// Geometry → TikZ (got-ocr2; implies `--format`).
    Geometry,
    /// Sheet music → `**kern` (got-ocr2; implies `--format`).
    Music,
    /// Photo description / VQA — planned (smolvlm2); errors cleanly today.
    Describe,
    /// Chart → structured python-dict data + number-head self-verify
    /// (onechart; needs `--model onechart.int8.focrq`). Distinct from
    /// `chart`, which is GOT-OCR2's format-mode rendering.
    ChartData,
}

impl OcrTask {
    /// The six GOT-OCR2 structured tasks all run the model's `OCR with format:`
    /// mode — the same engine switch as `--format` (the model auto-selects the
    /// formalism from the image, so one switch serves all six).
    fn implies_got_format(self) -> bool {
        matches!(
            self,
            Self::Formula
                | Self::Tables
                | Self::Chart
                | Self::Molecular
                | Self::Geometry
                | Self::Music
        )
    }
}

impl std::fmt::Display for OcrTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Ocr => "ocr",
            Self::Formula => "formula",
            Self::Tables => "tables",
            Self::Chart => "chart",
            Self::Molecular => "molecular",
            Self::Geometry => "geometry",
            Self::Music => "music",
            Self::Describe => "describe",
            Self::ChartData => "chart-data",
        })
    }
}

#[derive(Clone, Debug, Args)]
pub struct ConvertArgs {
    /// Source `model-00001-of-000001.safetensors`.
    pub input: PathBuf,
    /// Destination `.focrq`.
    #[arg(short, long)]
    pub output: PathBuf,
    /// Quantization target.
    #[arg(long, value_enum, default_value_t = QuantTarget::Int8)]
    pub quant: QuantTarget,
    /// Offline pre-packing target recorded in the `.focrq` header.
    #[arg(long, value_enum, default_value_t = ArchTarget::Generic)]
    pub arch: ArchTarget,
    /// Target model-architecture id the `.focrq` self-declares (the loader selects
    /// it from the registry). Default `unlimited-ocr`; e.g. `got-ocr2` (omits the
    /// tied `lm_head`, writes the Apache-2.0 notice). See `focr models`.
    #[arg(long, default_value = "unlimited-ocr")]
    pub model_id: String,
    /// Emit machine-readable scaffold JSON before the Phase-2 NotImplemented.
    #[arg(long)]
    pub json: bool,
}

#[derive(Clone, Debug, Args)]
pub struct RunsArgs {
    /// Specific run id to inspect.
    #[arg(long)]
    pub id: Option<String>,
    /// Maximum number of runs to list.
    #[arg(long, default_value_t = 20)]
    pub limit: i64,
    /// Output format for run history.
    #[arg(long, value_enum, default_value_t = OutputFormat::Plain)]
    pub format: OutputFormat,
    /// Alias for `--format json`.
    #[arg(long)]
    pub json: bool,
}

#[derive(Clone, Debug, Args)]
pub struct SyncArgs {
    /// Emit machine-readable scaffold JSON before the Phase-0 NotImplemented.
    #[arg(long, global = true)]
    pub json: bool,
    #[command(subcommand)]
    pub cmd: SyncCmd,
}

#[derive(Clone, Debug, Args)]
pub struct DoctorArgs {
    /// Emit the findings/report as one JSON object on stdout.
    #[arg(long)]
    pub json: bool,
    /// Apply the SAFE repairs (backup-first, hash-logged, reversible).
    /// Exit: 0 all fixed, 2 partial, 3 failed+rolled-back, 4 refused, 5 lock held.
    #[arg(long)]
    pub fix: bool,
    /// Disclose the worst-case blast radius without mutating anything.
    #[arg(long)]
    pub dry_run: bool,
    /// One-round-trip triage: {summary, findings, actions_planned,
    /// recommended_command} in one JSON object.
    #[arg(long)]
    pub robot_triage: bool,
    #[command(subcommand)]
    pub cmd: Option<DoctorCmd>,
}

#[derive(Clone, Debug, Eq, PartialEq, Subcommand)]
pub enum DoctorCmd {
    /// Restore a past --fix run byte-for-byte from its backups (hash-verified,
    /// fails closed on any missing backup).
    Undo {
        /// The run id printed by `doctor --fix` (also the dir name under
        /// `.doctor/runs/`).
        run_id: String,
    },
    /// The full doctor contract (detectors, fixers, exit codes, env), from
    /// the tool itself.
    Capabilities,
    /// Paste-ready agent handbook on stdout.
    RobotDocs,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum OutputFormat {
    Plain,
    Json,
    Ndjson,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Plain => "plain",
            Self::Json => "json",
            Self::Ndjson => "ndjson",
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Subcommand)]
pub enum SyncCmd {
    /// Export run-state audit records as JSONL (atomic, locked, idempotent).
    /// Export is the CANONICAL one-way direction of the audit contract
    /// (tests/fixtures/runs_schema.json): identical history re-exports
    /// byte-identically; a crash never leaves a torn file.
    ExportJsonl {
        /// Output path (default: `<run store>.jsonl`).
        #[arg(long)]
        file: Option<std::path::PathBuf>,
    },
    /// Import (replay) run-state audit records from JSONL — additive and
    /// restorative, keyed by run_id. NEVER a bidirectional merge: import
    /// restores an audit trail, it does not sync one.
    ImportJsonl {
        /// Input JSONL path.
        #[arg(long)]
        file: std::path::PathBuf,
    },
}

#[derive(Subcommand)]
pub enum RobotCmd {
    /// Stream OCR pipeline events as NDJSON.
    Run(RobotRunArgs),
    /// Self-describing event/contract schema (versioned).
    Schema,
    /// Diagnostics: model present? arch features? threads?
    Health,
    /// Detected SIMD tiers (SMMLA/SDOT/VNNI/AMX/scalar) + core count.
    Backends,
    /// Verify the dispatched int8 kernel is bit-identical to the scalar oracle
    /// on THIS host's silicon (exit 1 on any divergence). `FOCR_FORCE_ARCH`
    /// selects which available tier to verify.
    Selftest,
    /// One-round-trip agent triage: quick_ref + live health + state-aware
    /// recommendations + copy-pasteable command templates + the exit-code
    /// dictionary, in a single JSON object (the mega-command an agent runs
    /// FIRST).
    Triage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum QuantTarget {
    Int8,
    Int4,
}

impl QuantTarget {
    fn as_str(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Int4 => "int4",
        }
    }
}

impl std::fmt::Display for QuantTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum ArchTarget {
    Generic,
    Aarch64Smmla,
    X86Vnni,
    X86Amx,
}

impl ArchTarget {
    fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic",
            Self::Aarch64Smmla => "aarch64-smmla",
            Self::X86Vnni => "x86-vnni",
            Self::X86Amx => "x86-amx",
        }
    }

    /// The `.focrq` header packing byte (`0` Generic … `3` X86Amx — the order the
    /// `FocrqBuilder`/reader fix for `arch_target`).
    fn packing_byte(self) -> u8 {
        match self {
            Self::Generic => 0,
            Self::Aarch64Smmla => 1,
            Self::X86Vnni => 2,
            Self::X86Amx => 3,
        }
    }
}

impl std::fmt::Display for ArchTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Dispatch a parsed CLI invocation.
pub fn run(cli: Cli) -> FocrResult<()> {
    match cli.command {
        Command::Robot {
            cmd: RobotCmd::Run(args),
        } => {
            emit(&robot::run_start_event("ocr"));
            run_ocr(args.into_ocr_args(), true)
        }
        Command::Robot {
            cmd: RobotCmd::Schema,
        } => {
            emit(&robot::robot_schema());
            Ok(())
        }
        Command::Robot {
            cmd: RobotCmd::Health,
        } => {
            emit(&robot_health_payload());
            Ok(())
        }
        Command::Robot {
            cmd: RobotCmd::Backends,
        } => {
            emit(&robot_backends_payload());
            Ok(())
        }
        Command::Robot {
            cmd: RobotCmd::Selftest,
        } => run_robot_selftest(),
        Command::Robot {
            cmd: RobotCmd::Triage,
        } => {
            emit(&robot_triage_payload());
            Ok(())
        }
        Command::Ocr(args) if args.robot => {
            emit(&robot::run_start_event("ocr"));
            run_ocr(args, true)
        }
        Command::Ocr(args) => run_ocr(args, false),
        Command::OcrBatch(args) => run_ocr_batch(args),
        Command::Convert(args) => run_convert(&args),
        Command::Pull(args) => run_pull(&args),
        Command::Models(args) => run_models(&args),
        Command::Runs(args) => run_runs(&args),
        Command::Sync(args) => run_sync(&args),
        Command::Doctor(args) => run_doctor(&args),
    }
}

/// The result of one OCR run — a single image or a multi-page PDF — unified so the
/// output layer (markdown vs JSON-with-boxes, file vs stdout) is written once,
/// identically, regardless of which input path produced it.
enum Recognition {
    Single(native_engine::RecognizedDocument),
    Pdf(PdfRecognition),
}

impl Recognition {
    /// The rendered markdown document (PDF pages already joined by blank lines).
    fn markdown(&self) -> &str {
        match self {
            Recognition::Single(doc) => &doc.markdown,
            Recognition::Pdf(pdf) => &pdf.markdown,
        }
    }

    /// The structured JSON form. Always carries `schema_version` + `markdown`; a
    /// single image adds a top-level `layout` array, a PDF adds a `pages` array of
    /// `{page, layout}`. Every `layout` is a list of `{label, boxes}`, and each box
    /// is `[x1, y1, x2, y2]` in source-image pixels (top-left origin). When figures
    /// were extracted, a top-level `figures` array of `{label, page, bbox, path}`
    /// is appended (each `path` is the saved file, also referenced from the
    /// markdown).
    fn to_json(&self, figures: &[WrittenFigure]) -> serde_json::Value {
        let mut value = match self {
            Recognition::Single(doc) => serde_json::json!({
                "schema_version": robot::ROBOT_SCHEMA_VERSION,
                "markdown": doc.markdown,
                "layout": layout_to_json(&doc.layout),
            }),
            Recognition::Pdf(pdf) => {
                let pages: Vec<serde_json::Value> = pdf
                    .pages
                    .iter()
                    .map(|p| {
                        let mut page = serde_json::json!({
                            "page": p.page,
                            "layout": layout_to_json(&p.layout),
                        });
                        // Split-spread halves carry their side; unsplit pages
                        // keep the exact pre-bd-av64.11 shape (no key).
                        if let Some(half) = p.half {
                            page["half"] = serde_json::json!(half);
                        }
                        page
                    })
                    .collect();
                serde_json::json!({
                    "schema_version": robot::ROBOT_SCHEMA_VERSION,
                    "markdown": pdf.markdown,
                    "pages": pages,
                })
            }
        };
        if !figures.is_empty()
            && let Some(obj) = value.as_object_mut()
        {
            let arr: Vec<serde_json::Value> = figures
                .iter()
                .map(|f| {
                    serde_json::json!({
                        "label": f.label,
                        "page": f.page,
                        "bbox": f.bbox,
                        "path": f.path,
                    })
                })
                .collect();
            obj.insert("figures".to_string(), serde_json::Value::Array(arr));
        }
        value
    }
}

/// Serialize a page's layout spans as a JSON array of `{label, boxes}`, where each
/// box is the `[x1, y1, x2, y2]` pixel rectangle the model grounded that span to.
fn layout_to_json(layout: &[native_engine::LayoutSpan]) -> serde_json::Value {
    serde_json::Value::Array(
        layout
            .iter()
            .map(|span| {
                serde_json::json!({
                    "label": span.label,
                    "boxes": span.boxes,
                })
            })
            .collect(),
    )
}

/// True when the OCR result should be emitted as JSON. An output path ending in
/// `.json` selects JSON even without `--json` (a `.md`/other extension stays
/// markdown); the explicit `--json` flag is handled by the caller.
fn output_is_json(output: Option<&Path>) -> bool {
    output
        .and_then(Path::extension)
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
}

/// Write the recognition result to `path` as pretty JSON (with per-span bounding
/// boxes + any extracted `figures`) when `want_json`, else as the rendered
/// markdown. Both forms end with a trailing newline so the file is well-formed for
/// downstream tools.
fn write_ocr_output(
    path: &Path,
    rec: &Recognition,
    want_json: bool,
    figures: &[WrittenFigure],
    music_meta: Option<&native_engine::MusicPageMeta>,
) -> FocrResult<()> {
    let contents = if want_json {
        let mut value = rec.to_json(figures);
        if let Some(meta) = music_meta {
            value["staves"] = music_meta_to_json(meta);
            value["warnings"] = music_warnings_to_json(meta);
        }
        let mut s = serde_json::to_string_pretty(&value).map_err(|e| {
            FocrError::Other(anyhow::anyhow!(
                "serializing OCR JSON for {}: {e}",
                path.display()
            ))
        })?;
        s.push('\n');
        s
    } else {
        let md = rec.markdown();
        if md.ends_with('\n') {
            md.to_string()
        } else {
            format!("{md}\n")
        }
    };
    std::fs::write(path, contents).map_err(|e| {
        FocrError::Other(anyhow::anyhow!(
            "writing OCR output to {}: {e}",
            path.display()
        ))
    })
}

// ── figure extraction (`--extract-figures`) ─────────────────────────────────

/// The encoding chosen for one extracted figure by [`choose_figure_format`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FigureFormat {
    /// Lossless — for line-art / charts / screenshots (sharp edges, flat regions).
    Png,
    /// Lossy q85 — for photographic regions (smaller, ringing is imperceptible).
    Jpeg,
}

impl FigureFormat {
    fn ext(self) -> &'static str {
        match self {
            FigureFormat::Png => "png",
            FigureFormat::Jpeg => "jpg",
        }
    }
}

/// Pick a format for a cropped figure by content (the user's "auto" choice):
/// photographic regions have MANY distinct colors → JPG (small, lossy is fine);
/// line-art / charts / screenshots cluster into a few flat colors → PNG (lossless,
/// no ringing on sharp lines or embedded text). A grid sample of up to ~4096
/// pixels is quantized to 5 bits/channel and the distinct-color count + ratio
/// decide. Deterministic; defaults to PNG (the safe, lossless choice) on any
/// degenerate input.
fn choose_figure_format(img: &image::DynamicImage) -> FigureFormat {
    let rgb = img.to_rgb8();
    let total = u64::from(rgb.width()) * u64::from(rgb.height());
    if total == 0 {
        return FigureFormat::Png;
    }
    let step = total.div_ceil(4096).max(1) as usize;
    let mut seen = std::collections::HashSet::new();
    let mut sampled = 0u64;
    for px in rgb.pixels().step_by(step) {
        let [r, g, b] = px.0;
        let key = (u32::from(r >> 3) << 10) | (u32::from(g >> 3) << 5) | u32::from(b >> 3);
        seen.insert(key);
        sampled += 1;
    }
    if sampled == 0 {
        return FigureFormat::Png;
    }
    let ratio = seen.len() as f64 / sampled as f64;
    // Few distinct colors OR a low unique-color ratio ⇒ line-art ⇒ PNG.
    if seen.len() <= 64 || ratio < 0.10 {
        FigureFormat::Png
    } else {
        FigureFormat::Jpeg
    }
}

/// Encode `img` to `path` in the chosen format — JPG at quality 85, PNG lossless.
fn write_figure(img: &image::DynamicImage, path: &Path, fmt: FigureFormat) -> FocrResult<()> {
    let file = std::fs::File::create(path)
        .map_err(|e| FocrError::Other(anyhow::anyhow!("create figure {}: {e}", path.display())))?;
    let mut writer = std::io::BufWriter::new(file);
    let enc = |e: image::ImageError| {
        FocrError::Other(anyhow::anyhow!("encode figure {}: {e}", path.display()))
    };
    match fmt {
        FigureFormat::Jpeg => {
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut writer, 85)
                .encode_image(img)
                .map_err(enc)?;
        }
        FigureFormat::Png => {
            img.write_to(&mut writer, image::ImageFormat::Png)
                .map_err(enc)?;
        }
    }
    Ok(())
}

/// One figure written to disk, for the JSON `figures` array.
struct WrittenFigure {
    /// The model's ref label (`image`).
    label: String,
    /// 1-based source page (1 for a single image).
    page: usize,
    /// Source-pixel box `[x1, y1, x2, y2]` the figure was cropped from.
    bbox: [i64; 4],
    /// The reference path written into the markdown/JSON (relative to the output).
    path: String,
}

/// Where extracted figures are written and how they are referenced — resolved
/// from `--extract-figures` / `--figures-dir` + the `-o` path BEFORE any forward,
/// so a usage error fires immediately.
#[derive(Debug)]
struct FigurePlan {
    /// Filesystem directory figures are written into.
    dir: PathBuf,
    /// Prefix prepended to each figure filename in references (ends with `/`).
    ref_prefix: String,
}

impl FigurePlan {
    /// `Ok(None)` when figure extraction is off; `Ok(Some(plan))` otherwise.
    /// Usage error if `--extract-figures` is set with neither `-o` nor
    /// `--figures-dir` (no way to place the subfolder).
    fn resolve(args: &OcrArgs) -> FocrResult<Option<FigurePlan>> {
        if !args.extract_figures && args.figures_dir.is_none() {
            return Ok(None);
        }
        let output = args.output.as_deref();
        let plan = if let Some(dir_arg) = args.figures_dir.as_deref() {
            // Explicit dir: used verbatim in references; resolved against the
            // output file's dir (or the cwd for a stdout run) when relative.
            let ref_prefix = with_trailing_slash(&dir_arg.to_string_lossy());
            let dir = if dir_arg.is_absolute() {
                dir_arg.to_path_buf()
            } else {
                output_parent(output).join(dir_arg)
            };
            FigurePlan { dir, ref_prefix }
        } else {
            // `--extract-figures`: derive `<output-stem>_figures/` next to `-o`.
            let Some(out) = output else {
                return Err(FocrError::Usage(
                    "--extract-figures needs -o/--output to derive the figures \
                     subfolder; pass --figures-dir DIR for a stdout run"
                        .to_string(),
                ));
            };
            let stem = out
                .file_stem()
                .map_or_else(|| "ocr".to_string(), |s| s.to_string_lossy().into_owned());
            let dirname = format!("{stem}_figures");
            let dir = output_parent(output).join(&dirname);
            FigurePlan {
                dir,
                ref_prefix: format!("{dirname}/"),
            }
        };
        Ok(Some(plan))
    }

    fn writer(&self) -> FigureWriter {
        FigureWriter {
            dir: self.dir.clone(),
            ref_prefix: self.ref_prefix.clone(),
            created: false,
            written: Vec::new(),
        }
    }
}

/// The output file's parent directory, or `.` (cwd) for a bare filename / stdout.
fn output_parent(output: Option<&Path>) -> PathBuf {
    output
        .and_then(Path::parent)
        .filter(|p| !p.as_os_str().is_empty())
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
}

fn with_trailing_slash(s: &str) -> String {
    if s.is_empty() || s.ends_with('/') {
        s.to_string()
    } else {
        format!("{s}/")
    }
}

/// Writes a run's figures to disk (creating the dir lazily on the first one) and
/// rewrites each page's `![](images/…)` token to a real `![figure N](path)`,
/// accumulating the [`WrittenFigure`] records for the JSON output.
struct FigureWriter {
    dir: PathBuf,
    ref_prefix: String,
    created: bool,
    written: Vec<WrittenFigure>,
}

impl FigureWriter {
    fn ensure_dir(&mut self) -> FocrResult<()> {
        if !self.created {
            std::fs::create_dir_all(&self.dir).map_err(|e| {
                FocrError::Other(anyhow::anyhow!(
                    "create figures dir {}: {e}",
                    self.dir.display()
                ))
            })?;
            self.created = true;
        }
        Ok(())
    }

    /// Write one page's figures and return its markdown with each figure token
    /// rewritten to point at the saved file. `page` is 1-based (1 for a single
    /// image); figures are named `page{page}_figure_{n}.{ext}` (n 1-based, matching
    /// the page-local markdown index).
    fn process_page(
        &mut self,
        page: usize,
        markdown: &str,
        figures: Vec<native_engine::ExtractedFigure>,
    ) -> FocrResult<String> {
        let mut md = markdown.to_string();
        for fig in figures {
            let fignum = fig.index + 1;
            let fmt = choose_figure_format(&fig.image);
            let name = format!("page{page}_figure_{fignum}.{}", fmt.ext());
            self.ensure_dir()?;
            write_figure(&fig.image, &self.dir.join(&name), fmt)?;
            let rel = format!("{}{name}", self.ref_prefix);
            md = md.replace(&fig.markdown_ref, &format!("![figure {fignum}]({rel})"));
            self.written.push(WrittenFigure {
                label: fig.label,
                page,
                bbox: fig.bbox,
                path: rel,
            });
        }
        Ok(md)
    }

    fn into_written(self) -> Vec<WrittenFigure> {
        self.written
    }
}

fn run_ocr(args: OcrArgs, robot_mode: bool) -> FocrResult<()> {
    // Best-effort run telemetry (bd-223.4): capture the logging fields before
    // the args move, record the outcome after — a store failure NEVER fails
    // the user's run (stderr note only).
    let telemetry_input = args.request.image.display().to_string();
    let telemetry_model = args
        .request
        .model
        .as_ref()
        .map_or_else(|| "default".to_owned(), |p| p.display().to_string());
    let started = crate::storage::now_millis();
    let outcome = run_ocr_inner(args, robot_mode);
    let quant = if telemetry_model.contains("int8") {
        "int8"
    } else if telemetry_model.contains("int4") {
        "int4"
    } else {
        "f32-or-default"
    };
    let (status, exit_code) = match &outcome {
        Ok(()) => ("ok", 0i64),
        Err(FocrError::Cancelled) => ("cancelled", 6),
        Err(e) => ("error", i64::from(e.exit_code())),
    };
    let record = crate::storage::RunRecord {
        run_id: uuid::Uuid::new_v4().to_string(),
        started_at: started,
        finished_at: Some(crate::storage::now_millis()),
        input_path: telemetry_input,
        mode: "ocr".into(),
        quant: quant.into(),
        model_version_tag: telemetry_model,
        exit_code,
        status: status.into(),
    };
    if let Err(e) = crate::storage::RunStore::default_path()
        .and_then(|p| crate::storage::RunStore::open(&p))
        .and_then(|store| store.insert_run(&record))
    {
        let _ = progress::try_stderr_message(format_args!(
            "[focr] run-store note (telemetry only, run unaffected): {e}"
        ));
    }
    outcome
}

fn run_ocr_inner(args: OcrArgs, robot_mode: bool) -> FocrResult<()> {
    let request = args.to_request()?;
    // GOT-OCR2 `--format` (.mmd) mode and the decode tuning flags (`--max-length`,
    // `--temperature`, `--no-repeat-ngram`, `--ngram-window`) are threaded to the
    // leaf via process-globals (the shared OcrEngine/OcrModel signatures stay
    // frozen for the Baidu path). `request.format` already carries `--format` OR a
    // format-implying `--task` (folded in `to_request`). MUST precede
    // `OcrEngine::new()`: the decode params are resolved at model load.
    native_engine::force_got_format(request.format);
    native_engine::set_smolvlm2_question(request.question.clone());
    native_engine::set_decode_overrides(decode_overrides_from(&request));
    native_engine::set_preprocess_overrides(preprocess_overrides_from(&request));
    if let Some(err) = forced_test_error()? {
        return Err(err);
    }
    // Resolve the figure-extraction policy BEFORE the forward so a usage error
    // (e.g. `--extract-figures` without a place to put the subfolder) fires fast.
    let figure_plan = FigurePlan::resolve(&args)?;

    let engine = OcrEngine::new()?;
    // A `.pdf` (or `%PDF-`-magic) input rasterizes each page and OCRs them as one
    // document; everything else is a single decoded image. Both funnel through
    // `recognize_with_autodownload` so model resolution + the first-run download
    // offer behave identically. Both recognize WITH layout so the JSON output can
    // carry bounding boxes — the layout parse is negligible next to the forward
    // pass, and markdown-only consumers simply ignore it. With `--extract-figures`
    // the figure-aware variants additionally crop the `![](images/…)` regions out
    // of the source and rewrite the markdown references to the saved files.
    let is_pdf = pdf::looks_like_pdf(&request.image);
    // Human-mode progress (GH #2): a stderr-only bar for the long per-page PDF
    // drivers. NEVER in robot/`--json` mode — those stdout contracts are
    // machine-consumed — and `progress::Progress` additionally self-disables
    // when stderr is not an interactive TTY, so piped runs are byte-identical.
    let show_progress = !robot_mode && !args.json;
    if !is_pdf && (request.pages.is_some() || request.split_spreads || request.multi_page) {
        return Err(FocrError::Usage(format!(
            "--pages/--split-spreads/--multi-page operate on PDF pages, but {} is not a PDF \
             (for an image list, use `focr ocr-batch --multi-page`)",
            request.image.display()
        )));
    }
    if request.multi_page && (request.split_spreads || figure_plan.is_some()) {
        return Err(FocrError::Usage(
            "--multi-page is one cross-page pass over whole pages; it does not compose \
             with --split-spreads or --extract-figures (run those per-page passes \
             separately)"
                .into(),
        ));
    }
    if request.split_spreads && figure_plan.is_some() {
        return Err(FocrError::Usage(
            "--split-spreads does not compose with --extract-figures yet (figure \
             naming is per source page; splitting would collide the indices) — \
             run the passes separately"
                .into(),
        ));
    }
    let (recognition, figures): (Recognition, Vec<WrittenFigure>) = match (&figure_plan, is_pdf) {
        (Some(plan), true) => {
            let (pdf_rec, figs) =
                recognize_pdf_with_figures(&engine, &request, robot_mode, show_progress, plan)?;
            (Recognition::Pdf(pdf_rec), figs)
        }
        (Some(plan), false) => {
            let (mut doc, raw) =
                recognize_with_autodownload(&request, robot_mode, |model| match model {
                    Some(m) => engine.recognize_with_figures_model(m, &request.image),
                    None => engine.recognize_with_figures(&request.image),
                })?;
            let mut writer = plan.writer();
            doc.markdown = writer.process_page(1, &doc.markdown, raw)?;
            (Recognition::Single(doc), writer.into_written())
        }
        (None, true) if request.multi_page => (
            Recognition::Pdf(recognize_pdf_multi_page(
                &engine,
                &request,
                robot_mode,
                show_progress,
            )?),
            Vec::new(),
        ),
        (None, true) => (
            Recognition::Pdf(recognize_pdf(&engine, &request, robot_mode, show_progress)?),
            Vec::new(),
        ),
        (None, false) => (
            Recognition::Single(recognize_with_autodownload(
                &request,
                robot_mode,
                |model| match model {
                    Some(m) => engine.recognize_with_layout_model(m, &request.image),
                    None => engine.recognize_with_layout(&request.image),
                },
            )?),
            Vec::new(),
        ),
    };

    // Staff-level metadata from a TrOMR music forward (bd-av64.2): consumed
    // once per run; None for every non-music run.
    let music_meta = engine.take_music_page_meta();
    if robot_mode && let Some(meta) = &music_meta {
        let total = meta.staves.len() + meta.skips.len();
        for (index, bbox) in &meta.staves {
            emit(&robot::staff_event(*index, total, *bbox, "ok", None));
        }
        for skip in &meta.skips {
            emit(&robot::staff_event(
                skip.index,
                total,
                skip.bbox,
                "skipped",
                Some(&skip.reason),
            ));
        }
        for w in &meta.warnings {
            emit(&robot::music_warning_event(
                w.kind, w.part, w.measure, &w.detail,
            ));
        }
    }
    if !robot_mode
        && let Some(meta) = &music_meta
        && !meta.warnings.is_empty()
    {
        eprintln!(
            "[focr] {} musical-sanity warning(s) — annotated in the MusicXML; \
             re-run with --robot for machine-readable detail",
            meta.warnings.len()
        );
    }

    let markdown = recognition.markdown();
    // `--json` forces JSON; a `.json` output path selects it implicitly. (When no
    // `-o` is given, `output_is_json(None)` is false, so stdout behavior is exactly
    // the legacy `args.json` choice.)
    let want_json = args.json || output_is_json(args.output.as_deref());

    // An `-o/--output FILE` writes the result to disk (markdown or JSON-with-boxes)
    // regardless of mode, and is written FIRST so the file already exists when a
    // robot consumer sees the completion event below.
    if let Some(path) = args.output.as_deref() {
        write_ocr_output(path, &recognition, want_json, &figures, music_meta.as_ref())?;
    }

    if robot_mode {
        // The terminal success event carries the recognized markdown so a machine
        // consumer actually receives the OCR result on the NDJSON stream (the
        // human / `--json` modes print it below instead).
        emit(&robot::run_complete_event(markdown));
    } else if let Some(path) = args.output.as_deref() {
        // Result already went to the file; don't also echo it to stdout. Confirm
        // on stderr so stdout stays empty/clean for any wrapping pipeline.
        let figs = if figures.is_empty() {
            String::new()
        } else {
            format!(", {} figure(s)", figures.len())
        };
        eprintln!(
            "[focr] wrote {} ({}{figs})",
            path.display(),
            if want_json { "json" } else { "markdown" }
        );
    } else if args.json {
        let mut value = recognition.to_json(&figures);
        if let Some(meta) = &music_meta {
            value["staves"] = music_meta_to_json(meta);
            value["warnings"] = music_warnings_to_json(meta);
        }
        emit(&value);
    } else {
        println!("{markdown}");
    }
    Ok(())
}

/// The `--json` `staves` array for a music run (bd-av64.2): one entry per
/// DETECTED staff in detection order — recognized staves as `status: "ok"`,
/// failed ones as `status: "skipped"` with the reason. Absent entirely for
/// non-music runs, so every existing consumer's shape is unchanged.
/// The `--json` `warnings` array for a music run (bd-av64.5): annotate-only
/// musical-sanity observations, machine-stable kinds. Absent for non-music
/// runs.
fn music_warnings_to_json(meta: &native_engine::MusicPageMeta) -> serde_json::Value {
    serde_json::Value::Array(
        meta.warnings
            .iter()
            .map(|w| {
                serde_json::json!({
                    "kind": w.kind,
                    "part": w.part,
                    "measure": w.measure,
                    "detail": w.detail,
                })
            })
            .collect(),
    )
}

fn music_meta_to_json(meta: &native_engine::MusicPageMeta) -> serde_json::Value {
    let mut entries: Vec<(usize, serde_json::Value)> = meta
        .staves
        .iter()
        .map(|(index, bbox)| {
            (
                *index,
                serde_json::json!({
                    "staff": index + 1,
                    "bbox": [bbox.0, bbox.1, bbox.2, bbox.3],
                    "status": "ok",
                }),
            )
        })
        .chain(meta.skips.iter().map(|skip| {
            (
                skip.index,
                serde_json::json!({
                    "staff": skip.index + 1,
                    "bbox": [skip.bbox.0, skip.bbox.1, skip.bbox.2, skip.bbox.3],
                    "status": "skipped",
                    "reason": skip.reason,
                }),
            )
        }))
        .collect();
    entries.sort_by_key(|(index, _)| *index);
    serde_json::Value::Array(entries.into_iter().map(|(_, v)| v).collect())
}

/// Run one recognition, transparently offering the first-run model download once
/// and retrying against the freshly-fetched model.
///
/// `recog(model)` performs a full recognition: `Some(path)` pins that artifact,
/// `None` uses the engine default. The download offer fires only when the user
/// did NOT pin an explicit `--model`, we are on an interactive TTY, and not in
/// robot mode (robots never prompt/fetch — they get the clean model-not-found
/// error + pull hint). Both the single-image and PDF paths funnel through here so
/// model resolution and the auto-download behave identically.
fn recognize_with_autodownload<T, F>(
    request: &OcrRequest,
    robot_mode: bool,
    recog: F,
) -> FocrResult<T>
where
    F: Fn(Option<&Path>) -> FocrResult<T>,
{
    match recog(request.model.as_deref()) {
        Ok(md) => Ok(md),
        Err(FocrError::ModelNotFound(msg)) => {
            if request.model.is_none() && !robot_mode && is_interactive() {
                // A failed recognition retires its progress renderer without
                // waiting. Synchronize the generic terminal line before the
                // interactive download prompt is allowed to render.
                progress::clear_active_line();
                match offer_first_run_download()? {
                    Some(outcome) => recog(Some(&outcome.focrq_path)),
                    None => Err(FocrError::ModelNotFound(with_pull_hint(&msg))),
                }
            } else {
                Err(FocrError::ModelNotFound(with_pull_hint(&msg)))
            }
        }
        Err(e) => Err(e),
    }
}

/// Parse a `--pages` spec ("3", "3-7", "1,5-9,218"; 1-based, inclusive
/// ranges) against a document's page count into 0-based indices, in source
/// order, deduplicated (bd-av64.11). `None` ⇒ every page.
///
/// # Errors
/// [`FocrError::Usage`] on an empty/garbled spec, a zero page (pages are
/// 1-based), a reversed range, or a page past `page_count` (the error names
/// the document's page count).
fn parse_page_spec(spec: Option<&str>, page_count: usize) -> FocrResult<Vec<usize>> {
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

/// The logical pages of one rasterized PDF page: the page itself, or its
/// (left, right) halves when `--split-spreads` finds a book spread
/// (bd-av64.11). The split decision is logged under FOCR_TIMING.
fn logical_pages(
    image: image::DynamicImage,
    split_spreads: bool,
    source_page: usize,
) -> Vec<(image::DynamicImage, Option<&'static str>)> {
    if split_spreads {
        if let Some((left, right, gutter_x)) = pdf::split_spread(&image) {
            crate::native_engine::timing_log(&format!(
                "pdf page {source_page}: spread split at x={gutter_x} ({}x{})",
                image.width(),
                image.height()
            ));
            return vec![(left, Some("left")), (right, Some("right"))];
        }
        crate::native_engine::timing_log(&format!(
            "pdf page {source_page}: no spread detected ({}x{}), unsplit",
            image.width(),
            image.height()
        ));
    }
    vec![(image, None)]
}

/// One successfully-OCR'd PDF page's structured layout (for the JSON output).
struct PdfPageLayout {
    /// 1-based page number in the source PDF.
    page: usize,
    /// `Some("left"|"right")` when this entry is one half of a split spread.
    half: Option<&'static str>,
    /// The page's parsed layout spans (labels + pixel bounding boxes).
    layout: Vec<crate::native_engine::LayoutSpan>,
}

/// A recognized PDF: the concatenated markdown plus per-page layout for the pages
/// that decoded (skipped pages are absent from `pages`, as from the markdown).
struct PdfRecognition {
    markdown: String,
    pages: Vec<PdfPageLayout>,
}

/// `--multi-page` (bd-2z0y): rasterize the selected PDF pages and run ONE
/// cross-page pass over the whole document (the Unlimited-OCR `infer_multi`
/// contract, bd-1gv.25) — page N attends to pages 1..N−1; the output is one
/// markdown with `<PAGE>` separators. Per-page bbox layout is not produced in
/// this mode (the reference emits document-level markdown), so `pages` is
/// empty in the JSON. A page that fails to RASTER is skipped with the same
/// surfaced page event as the per-page path (the pass runs over the pages
/// that rendered); decode errors are whole-document by construction.
fn recognize_pdf_multi_page(
    engine: &OcrEngine,
    request: &OcrRequest,
    robot_mode: bool,
    show_progress: bool,
) -> FocrResult<PdfRecognition> {
    let pages = pdf::PdfPages::open(&request.image)?;
    let page_count = pages.len();
    let selected = parse_page_spec(request.pages.as_deref(), page_count)?;
    let mut images: Vec<image::DynamicImage> = Vec::new();
    let mut first_error: Option<FocrError> = None;
    // Rasterization progress (GH #2): a long book spends real time here before
    // the decode pass even starts. Stderr-only; self-disabled off-TTY.
    let raster_bar = progress::Progress::new("raster", selected.len(), show_progress);
    for idx in selected {
        raster_bar.start_item(format!("page {}/{page_count}", idx + 1));
        match pages.render(idx) {
            Ok(image) => images.push(image),
            Err(e) => {
                if robot_mode {
                    emit(&robot::page_skipped_event(idx + 1, &e));
                } else {
                    raster_bar.note(&format!("[focr] PDF page {} skipped: {e}", idx + 1));
                }
                if first_error.is_none() {
                    first_error = Some(e);
                }
            }
        }
        raster_bar.complete_item();
    }
    raster_bar.finish();
    if images.is_empty() {
        return Err(first_error.unwrap_or_else(|| {
            FocrError::Other(anyhow::anyhow!(
                "recognize_pdf_multi_page: no pages selected from {}",
                request.image.display()
            ))
        }));
    }
    let markdown = recognize_with_autodownload(request, robot_mode, |model| {
        let imgs = images.clone();
        // Robot mode streams a `page` event per crossed <PAGE> boundary
        // (bd-2z0y); the terminal run_complete still carries the assembled
        // markdown. The default-model arm resolves the same path an explicit
        // model would, so both arms stream.
        if robot_mode {
            let sink = Box::new(|page: usize, body: &str| {
                emit(&robot::page_decoded_event(page, body));
            });
            let model_path = model
                .map(std::path::Path::to_path_buf)
                .unwrap_or_else(OcrEngine::model_path);
            engine.recognize_multi_page_dynamic_streaming_with_model(&model_path, imgs, sink)
        } else {
            // Human-mode progress (GH #2) rides the SAME proven streaming
            // driver robot mode uses (identical assembled markdown, bd-2z0y);
            // the sink just advances a stderr bar per crossed <PAGE> boundary.
            // Off-TTY (or FOCR_NO_PROGRESS) the bar is disabled and the run
            // takes the original non-streaming arms, byte-for-byte.
            let bar = progress::Progress::new("ocr", imgs.len(), show_progress);
            if bar.is_enabled() {
                bar.start_item(format!("page 1/{}", imgs.len()));
                let model_path = model
                    .map(std::path::Path::to_path_buf)
                    .unwrap_or_else(OcrEngine::model_path);
                let sink = bar.page_sink();
                let out = engine.recognize_multi_page_dynamic_streaming_with_model(
                    &model_path,
                    imgs,
                    sink,
                );
                if out.is_ok() {
                    bar.finish();
                } else {
                    bar.retire();
                }
                out
            } else {
                match model {
                    Some(m) => engine.recognize_multi_page_dynamic_with_model(m, imgs),
                    None => engine.recognize_multi_page_dynamic(imgs),
                }
            }
        }
    })?;
    Ok(PdfRecognition {
        markdown,
        pages: Vec::new(),
    })
}

/// Recognize every page of a PDF, concatenating the per-page markdown into one
/// document (successful pages joined by a blank line) and collecting each decoded
/// page's layout spans for the JSON output.
///
/// Each page is rasterized in-memory by [`pdf`] and fed through the identical OCR
/// pipeline a PNG takes — no out-of-band `pdftoppm`. Pages render lazily, one at a
/// time, so a long book never holds every raster at once.
///
/// **Per-page resilience (mirrors the `ocr-batch` path):** a page that cannot be
/// rendered or recognized — an unsupported codec (`JPXDecode`/`JBIG2Decode`), a
/// vector/text page, a per-page decode or timeout error — is SKIPPED, so one bad
/// page never discards (nor wastes the compute already spent on) the OCR of every
/// other page. The skip is surfaced so it is never silent: a structured `page`
/// NDJSON event ([`robot::page_skipped_event`]) in robot mode, else a human stderr
/// warning. The **whole-run** conditions are propagated
/// immediately instead of being swallowed per-page:
/// * [`FocrError::ModelNotFound`] — lets [`recognize_with_autodownload`] offer the
///   first-run download and retry the whole document;
/// * [`FocrError::Cancelled`] — a Ctrl+C / cooperative cancel must abort the run,
///   not log a skip per remaining page and keep churning;
/// * [`FocrError::FormatMismatch`] — a bad/incompatible model artifact fails every
///   page identically, so surface it once rather than N times.
///
/// If NOT ONE page decodes, the first per-page failure is surfaced as a clean
/// error instead of an empty document.
fn recognize_pdf(
    engine: &OcrEngine,
    request: &OcrRequest,
    robot_mode: bool,
    show_progress: bool,
) -> FocrResult<PdfRecognition> {
    let pages = pdf::PdfPages::open(&request.image)?;
    let page_count = pages.len();
    recognize_with_autodownload(request, robot_mode, |model| {
        let selected = parse_page_spec(request.pages.as_deref(), page_count)?;
        // Per-page progress bar (GH #2): stderr-only, never in robot/--json
        // mode, self-disabled off-TTY (see `progress`). Built per ATTEMPT so a
        // ModelNotFound retires it without waiting; the autodownload wrapper
        // synchronizes the terminal line before its first-run prompt renders.
        let bar = progress::Progress::new("ocr", selected.len(), show_progress);
        let mut document = String::new();
        let mut page_layouts: Vec<PdfPageLayout> = Vec::new();
        let mut ok_pages = 0usize;
        let mut first_error: Option<FocrError> = None;
        for idx in selected {
            bar.start_item(format!("page {}/{page_count}", idx + 1));
            let halves = match pages.render(idx) {
                Ok(image) => logical_pages(image, request.split_spreads, idx + 1),
                Err(e) => {
                    // A raster failure skips the SOURCE page (both halves).
                    if robot_mode {
                        emit(&robot::page_skipped_event(idx + 1, &e));
                    } else {
                        bar.note(&format!("[focr] PDF page {} skipped: {e}", idx + 1));
                    }
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                    bar.complete_item();
                    continue;
                }
            };
            for (image, half) in halves {
                let page = match model {
                    Some(m) => engine.recognize_dynamic_with_layout_model(m, image),
                    None => engine.recognize_dynamic_with_layout(image),
                };
                match page {
                    Ok(doc) => {
                        if ok_pages > 0 {
                            document.push_str("\n\n");
                        }
                        document.push_str(doc.markdown.trim_end());
                        page_layouts.push(PdfPageLayout {
                            page: idx + 1,
                            half,
                            layout: doc.layout,
                        });
                        ok_pages += 1;
                    }
                    // Whole-run conditions are never per-page — abort immediately:
                    // a missing model (so the caller can offer the download + retry),
                    // a Ctrl+C / cooperative cancel, or a bad/incompatible model file
                    // (every page would fail it identically). Swallowing any of these
                    // per-page would lose the signal and waste compute on doomed pages.
                    Err(
                        e @ (FocrError::ModelNotFound(_)
                        | FocrError::Cancelled
                        | FocrError::FormatMismatch(_)),
                    ) => return Err(e),
                    // Isolate every other per-page failure: skip it, keeping the rest
                    // of the document, but SURFACE the skip on whichever stream the
                    // caller is reading — a structured `page` NDJSON event in robot
                    // mode (so a machine consumer can tell the document is missing
                    // pages), else a human stderr warning.
                    Err(e) => {
                        if robot_mode {
                            emit(&robot::page_skipped_event(idx + 1, &e));
                        } else {
                            bar.note(&format!("[focr] PDF page {} skipped: {e}", idx + 1));
                        }
                        if first_error.is_none() {
                            first_error = Some(e);
                        }
                    }
                }
            }
            bar.complete_item();
        }
        bar.finish();
        if ok_pages == 0 {
            // PdfPages::open guarantees >=1 page, and ModelNotFound returns early,
            // so first_error is always Some here; fall back defensively.
            return Err(first_error.unwrap_or_else(|| {
                FocrError::InputDecode(format!(
                    "PDF {} produced no decodable pages",
                    request.image.display()
                ))
            }));
        }
        Ok(PdfRecognition {
            markdown: document,
            pages: page_layouts,
        })
    })
}

/// [`recognize_pdf`] + figure extraction — the `--extract-figures` PDF path. Same
/// per-page resilience and whole-run abort rules, but each page is recognized WITH
/// its figure crops. The recognition pass is retryable (first-run model download),
/// so it does NO file I/O — it only collects each decodable page's number, doc, and
/// crops; the figures are written ONCE afterward and every page's markdown
/// references are rewritten to the saved files (page-namespaced so per-page
/// `images/0.jpg` tokens never collide across pages).
fn recognize_pdf_with_figures(
    engine: &OcrEngine,
    request: &OcrRequest,
    robot_mode: bool,
    show_progress: bool,
    plan: &FigurePlan,
) -> FocrResult<(PdfRecognition, Vec<WrittenFigure>)> {
    type OkPage = (
        usize,
        native_engine::RecognizedDocument,
        Vec<native_engine::ExtractedFigure>,
    );
    let pages = pdf::PdfPages::open(&request.image)?;
    let page_count = pages.len();
    let ok_pages: Vec<OkPage> = recognize_with_autodownload(request, robot_mode, |model| {
        let selected = parse_page_spec(request.pages.as_deref(), page_count)?;
        // Same per-page progress bar as `recognize_pdf` (GH #2): stderr-only,
        // disabled in robot mode and off-TTY, per-attempt lifetime.
        let bar = progress::Progress::new("ocr", selected.len(), show_progress);
        let mut out: Vec<OkPage> = Vec::new();
        let mut first_error: Option<FocrError> = None;
        for idx in selected {
            bar.start_item(format!("page {}/{page_count}", idx + 1));
            let page = pages.render(idx).and_then(|image| match model {
                Some(m) => engine.recognize_dynamic_with_figures_model(m, image),
                None => engine.recognize_dynamic_with_figures(image),
            });
            match page {
                Ok((doc, figs)) => out.push((idx + 1, doc, figs)),
                Err(
                    e @ (FocrError::ModelNotFound(_)
                    | FocrError::Cancelled
                    | FocrError::FormatMismatch(_)),
                ) => return Err(e),
                Err(e) => {
                    if robot_mode {
                        emit(&robot::page_skipped_event(idx + 1, &e));
                    } else {
                        bar.note(&format!("[focr] PDF page {} skipped: {e}", idx + 1));
                    }
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
            }
            bar.complete_item();
        }
        bar.finish();
        if out.is_empty() {
            return Err(first_error.unwrap_or_else(|| {
                FocrError::InputDecode(format!(
                    "PDF {} produced no decodable pages",
                    request.image.display()
                ))
            }));
        }
        Ok(out)
    })?;

    // Write pass — runs ONCE: write each page's figures and rewrite its markdown,
    // then concatenate exactly as `recognize_pdf` does (trim_end + blank-line join).
    let mut writer = plan.writer();
    let mut document = String::new();
    let mut page_layouts: Vec<PdfPageLayout> = Vec::new();
    for (i, (page_no, doc, figs)) in ok_pages.into_iter().enumerate() {
        let md = writer.process_page(page_no, &doc.markdown, figs)?;
        if i > 0 {
            document.push_str("\n\n");
        }
        document.push_str(md.trim_end());
        page_layouts.push(PdfPageLayout {
            page: page_no,
            half: None,
            layout: doc.layout,
        });
    }
    Ok((
        PdfRecognition {
            markdown: document,
            pages: page_layouts,
        },
        writer.into_written(),
    ))
}

/// Emit ONE batch image's outcome in the shared `ocr-batch` shape — a JSON object
/// pushed to `results` (with `--json`) or the `[focr] … =====` markdown block on
/// stdout/stderr. Factored so the sequential and spine drivers render byte-for-byte
/// identically; only the source of `outcome` differs between them.
fn emit_batch_result(
    json: bool,
    image: &std::path::Path,
    secs: f64,
    outcome: FocrResult<String>,
    results: &mut Vec<serde_json::Value>,
) {
    match outcome {
        Ok(markdown) => {
            if json {
                results.push(serde_json::json!({
                    "image": image.display().to_string(),
                    "ok": true,
                    "seconds": secs,
                    "markdown": markdown,
                }));
            } else {
                eprintln!("[focr] {} ({secs:.2}s)", image.display());
                println!("===== {} =====", image.display());
                println!("{markdown}");
            }
        }
        Err(err) => {
            if json {
                results.push(serde_json::json!({
                    "image": image.display().to_string(),
                    "ok": false,
                    "seconds": secs,
                    "error": err.to_string(),
                }));
            } else {
                eprintln!("[focr] {} FAILED ({secs:.2}s): {err}", image.display());
            }
        }
    }
}

/// Load-once batch OCR: reuse one model across every image in the process. The
/// conservative quant recipe is the default. `--experimental-full-int8` arms
/// the legacy all-int8 cache only when both independent accuracy gates are also
/// enabled. With the continuous-batch spine armed (`FOCR_BATCH_SPINE`) all pages
/// decode together through the scheduler; otherwise the proven sequential
/// per-image loop runs.
fn run_ocr_batch(args: OcrBatchArgs) -> FocrResult<()> {
    if let Some(err) = forced_test_error()? {
        return Err(err);
    }
    if args.experimental_full_int8 {
        native_engine::force_int8_decode(true)?;
    }
    native_engine::validate_experimental_full_int8_decode()?;
    let experimental_full_int8 = native_engine::experimental_full_int8_decode_requested();
    let decode_mode = if experimental_full_int8 {
        "experimental_full_int8"
    } else {
        "conservative_recipe"
    };
    // ocr-batch has no per-flag tuning surface, but the README-documented
    // FOCR_NO_REPEAT_NGRAM mitigation (int8 table-repetition) must work here
    // too (fresh-eyes fix — it previously reached only the clap env fallback
    // on `focr ocr`/`robot run` and was silently ignored by batch runs).
    if let Some(n) = std::env::var("FOCR_NO_REPEAT_NGRAM")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        native_engine::set_decode_overrides(native_engine::DecodeOverrides {
            no_repeat_ngram: Some(n),
            ..Default::default()
        });
    }
    let engine = OcrEngine::new()?;
    let model = args.model.clone();
    let count = args.images.len();
    let total = std::time::Instant::now();
    let mut results: Vec<serde_json::Value> = Vec::with_capacity(count);

    if args.multi_page {
        // ONE cross-page document pass (bd-1gv.25): page N attends to pages
        // 1..N-1; output is one assembled markdown with <PAGE> separators.
        let image_refs: Vec<&std::path::Path> = args
            .images
            .iter()
            .map(std::path::PathBuf::as_path)
            .collect();
        let markdown = match model.as_deref() {
            Some(m) => engine.recognize_multi_page_with_model(m, &image_refs),
            None => engine.recognize_multi_page(&image_refs),
        }?;
        let elapsed = total.elapsed().as_secs_f64();
        if args.json {
            emit(&serde_json::json!({
                "schema_version": robot::ROBOT_SCHEMA_VERSION,
                "command": "batch.multi_page",
                "pages": count,
                "seconds": elapsed,
                "decode_mode": decode_mode,
                "markdown": markdown,
            }));
        } else {
            println!("{markdown}");
            eprintln!("[focr] multi-page: {count} pages in one cross-page pass, {elapsed:.2}s");
        }
        return Ok(());
    }

    if native_engine::batch_scheduler::spine_enabled() {
        // Continuous-batch decode spine (FOCR_BATCH_SPINE=1): prefill + decode
        // every page TOGETHER. The per-page markdown is byte-identical to the
        // sequential loop below (bd-1azu.13), only throughput differs. A
        // batch-level failure (ModelNotFound / timeout) propagates as the run's
        // exit code rather than being folded into per-image results.
        let image_refs: Vec<&std::path::Path> = args
            .images
            .iter()
            .map(std::path::PathBuf::as_path)
            .collect();
        let batch = match model.as_deref() {
            Some(m) => engine.recognize_batch_with_model(m, &image_refs),
            None => engine.recognize_batch(&image_refs),
        }?;
        let per_image = total.elapsed().as_secs_f64() / (count.max(1) as f64);
        for (image, outcome) in args.images.iter().zip(batch) {
            emit_batch_result(args.json, image, per_image, outcome, &mut results);
        }
    } else {
        // Sequential per-image loop — the proven oracle path (FOCR_BATCH_SPINE=0),
        // byte-for-byte what it has always been. The progress bar (GH #2) is
        // stderr-only, disabled under `--json`, and self-disabled off-TTY; it
        // is suspended around each result block so the bar never fuses to the
        // per-image markdown/summary output.
        let bar = progress::Progress::new("batch", count, !args.json);
        for (i, image) in args.images.iter().enumerate() {
            let name = image.file_name().map_or_else(
                || image.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            );
            bar.start_item(format!("image {}/{count}: {name}", i + 1));
            let started = std::time::Instant::now();
            let outcome = match model.as_deref() {
                Some(m) => engine.recognize_with_model(m, image),
                None => engine.recognize(image),
            };
            let secs = started.elapsed().as_secs_f64();
            bar.complete_item();
            bar.suspend();
            emit_batch_result(args.json, image, secs, outcome, &mut results);
        }
        bar.finish();
    }

    let elapsed = total.elapsed().as_secs_f64();
    let per_image = elapsed / (count.max(1) as f64);
    if args.json {
        emit(&serde_json::json!({
            "schema_version": robot::ROBOT_SCHEMA_VERSION,
            "command": "ocr-batch",
            "count": count,
            "decode_mode": decode_mode,
            "experimental_full_int8": experimental_full_int8,
            "seconds_total": elapsed,
            "seconds_per_image": per_image,
            "results": results,
        }));
    } else {
        eprintln!(
            "[focr] batch complete: {count} images in {elapsed:.2}s \
             ({per_image:.2}s/image, decode_mode={decode_mode})"
        );
    }
    Ok(())
}

/// Offline weight transform: raw bf16 safetensors → a self-contained int8
/// `.focrq` (plan §5). Unlimited-OCR quantizes only the validated decoder
/// FFN/expert GEMMs; attention q/k/v/o, `lm_head`, vision, projector,
/// `embed_tokens`, router, and norms remain high precision. `--quant int4` is
/// not yet validated (doctrine #1) and returns `NotImplemented`.
fn run_convert(args: &ConvertArgs) -> FocrResult<()> {
    // int4: surface the machine scaffold (so robot callers still see the planned
    // shape) then refuse — BEFORE any file I/O, so the outcome is deterministic
    // regardless of whether the input exists.
    if args.quant == QuantTarget::Int4 {
        if args.json {
            emit(&serde_json::json!({
                "schema_version": robot::ROBOT_SCHEMA_VERSION,
                "command": "convert",
                "status": "scaffold",
                "implemented": false,
                "input": args.input,
                "output": args.output,
                "quant": args.quant.as_str(),
                "arch": args.arch.as_str(),
            }));
        }
        return Err(FocrError::NotImplemented(
            "focr convert --quant int4 is not yet supported; the int4 group-quantized \
             path is unvalidated (use --quant int8)"
                .into(),
        ));
    }

    // int8 — the validated path. Resolve the input the way `ocr` resolves a model
    // (a `.safetensors` file as-is, or the canonical shard inside a directory).
    let resolved = native_engine::OcrModel::resolve_model(&args.input)?;
    let bytes = std::fs::read(&resolved).map_err(|e| {
        FocrError::ModelNotFound(format!(
            "cannot read safetensors at {}: {e}",
            resolved.display()
        ))
    })?;
    let input_bytes = bytes.len();
    let source_sha256 = quant::convert::sha256_of_bytes(&bytes);
    // `from_bytes` keeps ownership of the single read; the hash above borrowed it.
    let weights = native_engine::weights::Weights::from_bytes(bytes)?;
    let tensor_count = weights.len();
    // Resolve the target model architecture (the `.focrq` self-declares its id).
    let arch = native_engine::model_arch::arch_by_id(&args.model_id).ok_or_else(|| {
        FocrError::Usage(format!(
            "unknown --model-id {:?} (see `focr models` for the registry)",
            args.model_id
        ))
    })?;
    if arch.id() == native_engine::model_arch::default_arch().id() {
        native_engine::unlimited_ocr_census::validate_conversion_source_sha256(&source_sha256)?;
    }
    let omit_lm_head = arch.tie_word_embeddings();
    let quantized = weights
        .names()
        .filter(|name| quant::convert::is_decoder_int8_tensor_for(name, arch))
        .filter(|name| !(omit_lm_head && *name == "lm_head.weight"))
        .count();

    let blob = quant::convert::safetensors_to_focrq(
        &weights,
        quant::convert::ConvertQuant::Int8,
        args.arch.packing_byte(),
        source_sha256,
        arch,
    )?;
    let output_bytes = blob.len();
    std::fs::write(&args.output, &blob).map_err(|e| {
        FocrError::Other(anyhow::anyhow!(
            "writing .focrq to {}: {e}",
            args.output.display()
        ))
    })?;

    let sha_hex = hex_encode32(&source_sha256);
    let quant_recipe = (arch.id() == native_engine::model_arch::default_arch().id())
        .then_some(quant::convert::UNLIMITED_OCR_INT8_RECIPE_ID);
    if args.json {
        emit(&serde_json::json!({
            "schema_version": robot::ROBOT_SCHEMA_VERSION,
            "command": "convert",
            "status": "ok",
            "implemented": true,
            "input": resolved,
            "output": args.output,
            "quant": args.quant.as_str(),
            "arch": args.arch.as_str(),
            "model_id": arch.id(),
            "quant_recipe": quant_recipe,
            "source_sha256": sha_hex,
            "tensors": tensor_count,
            "tensors_quantized": quantized,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
        }));
    } else {
        eprintln!(
            "[focr] convert: wrote {} ({} quant {}: {tensor_count} tensors, {quantized} int8, \
             {input_bytes} -> {output_bytes} bytes) source_sha256={sha_hex}{}",
            args.output.display(),
            args.arch.as_str(),
            args.quant.as_str(),
            quant_recipe.map_or_else(String::new, |id| format!(" quant_recipe={id}")),
        );
    }
    Ok(())
}

/// Lowercase-hex-encode the 32-byte source digest for human/robot display.
fn hex_encode32(bytes: &[u8; 32]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(64);
    for &b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn run_runs(args: &RunsArgs) -> FocrResult<()> {
    let limit = i64::from(non_negative_u32("limit", args.limit)?);
    let store = crate::storage::RunStore::open(&crate::storage::RunStore::default_path()?)?;
    let records = store.query(args.id.as_deref(), limit)?;
    let format = if args.json {
        OutputFormat::Json
    } else {
        args.format
    };
    let record_json = |r: &crate::storage::RunRecord| {
        serde_json::json!({
            "schema_version": crate::storage::SCHEMA_VERSION,
            "run_id": r.run_id,
            "started_at": r.started_at,
            "finished_at": r.finished_at,
            "input_path": r.input_path,
            "mode": r.mode,
            "quant": r.quant,
            "model_version_tag": r.model_version_tag,
            "exit_code": r.exit_code,
            "status": r.status,
        })
    };
    match format {
        OutputFormat::Json => emit(&serde_json::json!({
            "schema_version": robot::ROBOT_SCHEMA_VERSION,
            "command": "runs",
            "store": store.path(),
            "count": records.len(),
            "runs": records.iter().map(record_json).collect::<Vec<_>>(),
        })),
        OutputFormat::Ndjson => {
            for r in &records {
                emit(&record_json(r));
            }
        }
        OutputFormat::Plain => {
            if records.is_empty() {
                println!("no recorded runs ({})", store.path().display());
            }
            for r in &records {
                println!(
                    "{}  {}  {}  exit {}  {}  {}",
                    r.run_id, r.status, r.mode, r.exit_code, r.quant, r.input_path
                );
            }
        }
    }
    Ok(())
}

fn run_sync(args: &SyncArgs) -> FocrResult<()> {
    let store = crate::storage::RunStore::open(&crate::storage::RunStore::default_path()?)?;
    let (subcommand, file, n) = match &args.cmd {
        SyncCmd::ExportJsonl { file } => {
            let out = file.clone().unwrap_or_else(|| {
                let mut p = store.path().to_path_buf();
                p.set_extension("jsonl");
                p
            });
            let n = crate::storage::export_jsonl(&store, &out)?;
            ("export-jsonl", out, n)
        }
        SyncCmd::ImportJsonl { file } => {
            let n = crate::storage::import_jsonl(&store, file)?;
            ("import-jsonl", file.clone(), n)
        }
    };
    if args.json {
        emit(&serde_json::json!({
            "schema_version": robot::ROBOT_SCHEMA_VERSION,
            "command": "sync",
            "subcommand": subcommand,
            "store": store.path(),
            "file": file,
            "records": n,
        }));
    } else {
        eprintln!(
            "[focr] sync {subcommand}: {n} records via {}",
            file.display()
        );
    }
    Ok(())
}

/// A stable lowercase name for a [`crate::model_arch::Task`] (the machine
/// contract for `focr models --json`).
fn task_name(t: crate::model_arch::Task) -> &'static str {
    use crate::model_arch::Task;
    match t {
        Task::Ocr => "ocr",
        Task::Formula => "formula",
        Task::Tables => "tables",
        Task::Chart => "chart",
        Task::Molecular => "molecular",
        Task::Geometry => "geometry",
        Task::Music => "music",
        Task::Describe => "describe",
        Task::Vqa => "vqa",
        Task::Handwriting => "handwriting",
    }
}

/// Render one model-architecture descriptor as a JSON object for `focr models
/// --json`.
fn model_arch_json(a: &dyn crate::model_arch::ModelArch) -> serde_json::Value {
    serde_json::json!({
        "id": a.id(),
        "display_name": a.display_name(),
        "implemented": a.implemented(),
        "status": if a.implemented() { "ready" } else { "planned" },
        "tasks": a.tasks().iter().map(|t| task_name(*t)).collect::<Vec<_>>(),
        "vision_encoder": format!("{:?}", a.vision_encoder()),
        "decoder": format!("{:?}", a.decoder()),
        "tokenizer": format!("{:?}", a.tokenizer()),
        "default_artifact": a.default_artifact_basename(),
        "license": a.license_notice(),
    })
}

#[derive(Debug, Default)]
struct PullAvailability {
    in_manifest: bool,
    compatible: Vec<(String, String)>,
    blocked: Vec<(String, String)>,
}

fn pull_availability(manifest: Option<&dist::Manifest>, model_id: &str) -> PullAvailability {
    let Some(manifest) = manifest else {
        return PullAvailability::default();
    };
    let quants = if manifest.model == model_id {
        Some(&manifest.quants)
    } else {
        manifest.models.get(model_id).map(|entry| &entry.quants)
    };
    let Some(quants) = quants else {
        return PullAvailability::default();
    };

    let mut availability = PullAvailability {
        in_manifest: true,
        ..PullAvailability::default()
    };
    for (quant, entry) in quants {
        let item = (quant.clone(), entry.recipe.clone());
        if dist::quant_recipe_is_compatible(model_id, quant, &entry.recipe) {
            availability.compatible.push(item);
        } else {
            availability.blocked.push(item);
        }
    }
    availability
}

/// `focr models` — list the model architectures this build can run (the "model
/// zoo", epic bd-3jo6). A human table by default; `--json` for a machine-readable
/// list. Runtime readiness and artifact compatibility are separate fields so a
/// mismatched external manifest remains visible without being advertised as
/// pullable.
fn run_models(args: &ModelsArgs) -> FocrResult<()> {
    let archs = crate::model_arch::registry();
    // The embedded manifest supplies offline provenance and exact runtime recipe
    // compatibility. Keep reporting blocked entries defensively for external or
    // future manifests that drift from the runtime contract.
    let manifest = dist::builtin_manifest()?;
    if args.json {
        let models: Vec<serde_json::Value> = archs
            .iter()
            .map(|a| {
                let mut j = model_arch_json(*a);
                let pull = pull_availability(Some(&manifest), a.id());
                j["pull"] = serde_json::json!({
                    "in_manifest": pull.in_manifest,
                    "available": !pull.compatible.is_empty(),
                    "quants": pull.compatible.iter().map(|(quant, _)| quant).collect::<Vec<_>>(),
                    "recipes": pull.compatible.iter().map(|(quant, recipe)| serde_json::json!({
                        "quant": quant,
                        "recipe": recipe,
                    })).collect::<Vec<_>>(),
                    "blocked": pull.blocked.iter().map(|(quant, recipe)| serde_json::json!({
                        "quant": quant,
                        "recipe": recipe,
                        "required_recipe": dist::required_quant_recipe(a.id(), quant),
                    })).collect::<Vec<_>>(),
                });
                j
            })
            .collect();
        emit(&serde_json::json!({
            "schema_version": robot::ROBOT_SCHEMA_VERSION,
            "models": models,
            "guidance": {
                "unlimited-ocr": "default runtime; FAST plain-text OCR; `focr pull` installs the conservative exact-recipe int8 artifact",
                "got-ocr2": "specialized structured output the default can't produce — math (LaTeX), tables, charts, molecular (SMILES), geometry, sheet music; heavier per page, use when you need FORMAT not plain text; shorthand: `focr ocr --task formula|tables|chart|molecular|geometry|music` (implies `--format`)"
            },
        }));
    } else {
        // TASKS last: its width varies per model (GOT-OCR2 serves seven), so a
        // fixed-width column would misalign — keep it trailing.
        println!(
            "{:<14}  {:<8}  {:<12}  {:<22}  TASKS",
            "ID", "STATUS", "PULL", "MODEL"
        );
        for a in archs {
            let tasks = a
                .tasks()
                .iter()
                .map(|t| task_name(*t))
                .collect::<Vec<_>>()
                .join(",");
            let status = if a.implemented() { "ready" } else { "planned" };
            let availability = pull_availability(Some(&manifest), a.id());
            let pull = if !availability.compatible.is_empty() {
                availability
                    .compatible
                    .iter()
                    .map(|(quant, _)| quant.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            } else if !availability.blocked.is_empty() {
                "blocked".to_owned()
            } else if a.implemented() {
                "local".to_owned()
            } else {
                "-".to_owned()
            };
            println!(
                "{:<14}  {:<8}  {:<12}  {:<22}  {}",
                a.id(),
                status,
                pull,
                a.display_name(),
                tasks
            );
        }
        println!();
        println!("Choosing a model:");
        println!("  unlimited-ocr (default)  FAST plain-text OCR. `focr pull` installs the");
        println!("                           conservative exact-recipe int8 artifact.");
        println!(
            "  got-ocr2                 SPECIALIZED structured output the default can't produce:"
        );
        println!("                           math (LaTeX), tables, charts, molecular (SMILES),");
        println!(
            "                           geometry, sheet music. Heavier per page — use it when you"
        );
        println!(
            "                           need FORMAT, not for plain text. `focr pull got-ocr2`,"
        );
        println!("                           then `focr ocr --model got-ocr2.int8.focrq <image>`.");
        println!(
            "                           Add `--format` for structured .mmd output (LaTeX/tables/…),"
        );
        println!(
            "                           or `--task formula|tables|chart|molecular|geometry|music`"
        );
        println!(
            "                           to select the format mode by task (implies `--format`)."
        );
    }
    Ok(())
}

fn run_doctor(args: &DoctorArgs) -> FocrResult<()> {
    use crate::doctor;

    // Contract sub-surfaces first: they never touch disk.
    match &args.cmd {
        Some(DoctorCmd::Capabilities) => {
            emit(&doctor::capabilities());
            return Ok(());
        }
        Some(DoctorCmd::RobotDocs) => {
            print!("{}", doctor::robot_docs());
            return Ok(());
        }
        Some(DoctorCmd::Undo { run_id }) => {
            let root = doctor::DoctorRoot::resolve()?;
            let restored = doctor::undo(&root, run_id)?;
            emit(&serde_json::json!({
                "schema_version": doctor::DOCTOR_SCHEMA_VERSION,
                "command": "doctor.undo",
                "run_id": run_id,
                "actions_restored": restored,
                "verified": "every restored file hash-matched its recorded before_hash",
            }));
            return Ok(());
        }
        None => {}
    }

    let root = doctor::DoctorRoot::resolve()?;
    let findings = doctor::detect(&root);

    // Deterministic run id: findings-count + wall-clock seconds (unique enough
    // for a human-auditable dir name; collision just appends actions).
    let run_id = format!(
        "run-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or_default()
    );

    let planned: Vec<serde_json::Value> = findings
        .iter()
        .map(|f| {
            serde_json::json!({
                "detector": f.detector,
                "severity": f.severity,
                "path": f.path,
                "message": f.message,
                "fixability": f.fixability,
            })
        })
        .collect();

    if args.robot_triage {
        let recommended = if findings.is_empty() {
            "focr ocr <image> -o out.md".to_string()
        } else if args.fix {
            "focr doctor --fix".to_string()
        } else {
            "focr doctor --fix   # safe repairs only; see `focr doctor capabilities --json`"
                .to_string()
        };
        emit(&serde_json::json!({
            "schema_version": doctor::DOCTOR_SCHEMA_VERSION,
            "command": "doctor.robot-triage",
            "summary": {"findings": findings.len(), "healthy": findings.is_empty()},
            "findings": planned,
            "actions_planned": findings.iter().filter(|f| matches!(f.fixability, doctor::Fixability::Auto{..})).count(),
            "recommended_command": recommended,
            "capabilities_hint": "focr doctor capabilities --json",
        }));
        std::process::exit(if findings.is_empty() {
            doctor::EXIT_HEALTHY
        } else {
            doctor::EXIT_FINDINGS
        });
    }

    if args.dry_run {
        emit(&serde_json::json!({
            "schema_version": doctor::DOCTOR_SCHEMA_VERSION,
            "command": "doctor.dry-run",
            "would_mutate": findings.iter().filter(|f| matches!(f.fixability, doctor::Fixability::Auto{..})).count(),
            "blast_radius": root.cache_root.display().to_string(),
            "findings": planned,
            "note": "NO mutation performed; --fix applies the auto items backup-first",
        }));
        std::process::exit(if findings.is_empty() {
            doctor::EXIT_HEALTHY
        } else {
            doctor::EXIT_FINDINGS
        });
    }

    if args.fix {
        let report = match doctor::fix(&root, &findings, &run_id) {
            Ok(r) => r,
            Err(e) => {
                let msg = format!("{e}");
                if msg.contains("doctor lock held") {
                    eprintln!("focr doctor: {msg}");
                    std::process::exit(doctor::EXIT_CONCURRENCY_LOST);
                }
                return Err(e);
            }
        };
        emit(&serde_json::json!({
            "schema_version": doctor::DOCTOR_SCHEMA_VERSION,
            "command": "doctor.fix",
            "run_id": report.run_id,
            "fixed": report.fixed,
            "refused_unsafe": report.refused,
            "advice_only": report.advice_only,
            "failed_rolled_back": report.failed_rolled_back,
            "undo": format!("focr doctor undo {}", report.run_id),
            "findings": planned,
        }));
        std::process::exit(report.exit_code);
    }

    // Detect-only.
    if args.json {
        emit(&serde_json::json!({
            "schema_version": doctor::DOCTOR_SCHEMA_VERSION,
            "command": "doctor",
            "healthy": findings.is_empty(),
            "findings": planned,
        }));
    } else if findings.is_empty() {
        println!("focr doctor: healthy ({} checks green)", 4);
    } else {
        for f in &findings {
            println!("[{}] {}: {}", f.severity, f.detector, f.message);
        }
    }
    std::process::exit(if findings.is_empty() {
        doctor::EXIT_HEALTHY
    } else {
        doctor::EXIT_FINDINGS
    });
}

fn forced_test_error() -> FocrResult<Option<FocrError>> {
    #[cfg(debug_assertions)]
    {
        let Some(raw) = std::env::var_os(FORCE_TEST_ERROR_ENV) else {
            return Ok(None);
        };
        if raw.as_os_str().is_empty() {
            return Ok(None);
        }
        let value = raw.to_string_lossy();
        let err = match value.as_ref() {
            "input_decode" => {
                FocrError::InputDecode(format!("forced by {FORCE_TEST_ERROR_ENV}=input_decode"))
            }
            "timeout" => FocrError::Timeout(format!("forced by {FORCE_TEST_ERROR_ENV}=timeout")),
            "cancelled" => FocrError::Cancelled,
            other => {
                return Err(FocrError::Usage(format!(
                    "invalid {FORCE_TEST_ERROR_ENV}={other:?}; expected input_decode, timeout, \
                     or cancelled"
                )));
            }
        };
        Ok(Some(err))
    }

    #[cfg(not(debug_assertions))]
    {
        Ok(None)
    }
}

fn positive_u32(name: &str, value: i64) -> FocrResult<u32> {
    if value <= 0 {
        return Err(FocrError::Usage(format!("{name} must be > 0, got {value}")));
    }
    u32::try_from(value)
        .map_err(|_| FocrError::Usage(format!("{name} is too large for u32: {value}")))
}

fn non_negative_u32(name: &str, value: i64) -> FocrResult<u32> {
    if value < 0 {
        return Err(FocrError::Usage(format!(
            "{name} must be >= 0, got {value}"
        )));
    }
    u32::try_from(value)
        .map_err(|_| FocrError::Usage(format!("{name} is too large for u32: {value}")))
}

fn non_negative_finite_f32(name: &str, value: f32) -> FocrResult<f32> {
    if !value.is_finite() || value < 0.0 {
        return Err(FocrError::Usage(format!(
            "{name} must be finite and >= 0, got {value}"
        )));
    }
    Ok(value)
}

fn robot_health_payload() -> serde_json::Value {
    let model_spec = OcrEngine::model_path();
    let model_present = native_engine::native_model_available(&model_spec);
    let model_search_dirs: Vec<_> = native_engine::model_resolution_search_dirs()
        .into_iter()
        .map(|p| p.display().to_string())
        .collect();
    // Phase 0: minimal health. The expanded report (arch features, thread
    // budget) lands with the rest of plan §7.3.
    serde_json::json!({
        "schema_version": robot::ROBOT_SCHEMA_VERSION,
        "status": "scaffold",
        "ready": false,
        "phase": "pre-Phase-0 skeleton",
        "model_present": model_present,
        "model_spec": model_spec.display().to_string(),
        "model_search_dirs": model_search_dirs,
        "model_license_notice": FOCR_MODEL_LICENSE_NOTICE,
    })
}

/// The `robot triage` mega-command payload (bd-wp8.7 / agent-ergonomics
/// Axiom 0): everything an agent needs to act after ONE round-trip — what the
/// tool does (quick_ref), whether it can run right now (health), what to type
/// NEXT given that state (recommendations, copy-pasteable), the common command
/// templates, and the frozen exit-code dictionary. Composes the existing
/// health payload and schema — no duplicated facts.
fn robot_triage_payload() -> serde_json::Value {
    let health = robot_health_payload();
    let model_present = health["model_present"].as_bool().unwrap_or(false);
    let recommendations: Vec<&str> = if model_present {
        vec![
            "focr ocr <image-or-pdf> -o out.md   # primary: OCR to markdown",
            "focr ocr <image> --json             # structured JSON + bounding boxes",
            "focr models                         # which zoo models/tasks this build can run",
            "focr ocr-batch <img1> <img2> ...    # load weights once, many pages",
        ]
    } else {
        vec![
            "focr pull                           # install the default conservative Unlimited-OCR model",
            "FOCR_MODEL_PATH=/path/to/raw-or-compatible-model focr ocr <image-or-pdf> -o out.md",
            "focr models                         # inspect compatible and blocked model pulls",
        ]
    };
    serde_json::json!({
        "schema_version": robot::ROBOT_SCHEMA_VERSION,
        "command": "robot.triage",
        "quick_ref": {
            "ocr": "parse a document image/PDF into markdown (--json for boxes; -o FILE to write)",
            "ocr-batch": "many images in one process (weights load once)",
            "pull": "download a manifest-compatible model into the cache; defaults to the conservative Unlimited-OCR artifact",
            "models": "list model ids, tasks, and compatible or blocked pull status",
            "convert": "offline safetensors -> .focrq quantization",
            "runs": "query run history (--format json|ndjson; empty history = exit 0)",
            "sync": "export/import the append-only run audit JSONL (one-way contract)",
            "doctor": "self-check/repair",
            "robot": "agent surfaces: run (NDJSON stream), schema, health, backends, selftest, triage",
        },
        "health": health,
        "recommendations": recommendations,
        "commands": {
            "first_ocr": "focr ocr page.png -o page.md",
            "structured": "focr ocr page.png --json",
            "stream_events": "focr robot run page.png",
            "contract": "focr robot schema",
        },
        "exit_codes": robot::robot_schema()["exit_codes"].clone(),
    })
}

fn emit(value: &serde_json::Value) {
    // Robot-facing commands emit exactly one JSON object per line.
    println!(
        "{}",
        serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
    );
}

/// Both stdin AND stderr are TTYs — the prerequisite for an interactive
/// download prompt (stderr is where the prompt is written; stdin is the answer
/// channel; stdout is reserved for the OCR result / JSON).
fn is_interactive() -> bool {
    use std::io::IsTerminal;
    std::io::stdin().is_terminal() && std::io::stderr().is_terminal()
}

/// Append the actionable acquisition hint to a model-not-found message.
fn with_pull_hint(msg: &str) -> String {
    format!(
        "{msg} — run `focr pull` to install the compatible conservative default, point \
         FOCR_MODEL_PATH at raw BF16 weights or an exact-recipe local artifact, or inspect \
         `focr models` for other compatible pulls"
    )
}

/// Prompt on the TTY only when the selected manifest may supply a compatible
/// default. The embedded source is checked before offering its multi-gigabyte
/// download; custom sources are checked by [`dist::pull`].
fn offer_first_run_download() -> FocrResult<Option<dist::PullOutcome>> {
    use std::io::Write as _;
    let source = dist::resolve_manifest_source(None);
    if source == dist::DEFAULT_MANIFEST_SOURCE {
        let manifest = dist::builtin_manifest()?;
        let availability = pull_availability(Some(&manifest), &manifest.model);
        if availability.compatible.is_empty() {
            return Ok(None);
        }
    }
    eprint!(
        "focr: model not found. Download compatible weights from the configured manifest now? [y/N] "
    );
    std::io::stderr().flush().ok();
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(|e| FocrError::Other(anyhow::anyhow!("reading prompt response: {e}")))?;
    if !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
        return Ok(None);
    }
    let outcome = dist::pull(None, dist::DEFAULT_QUANT, &source, false, |line| {
        eprintln!("focr pull: {line}");
    })?;
    Ok(Some(outcome))
}

/// `focr pull` — download (or confirm-cached) the model weights + tokenizer.
fn run_pull(args: &PullArgs) -> FocrResult<()> {
    let source = dist::resolve_manifest_source(args.manifest.as_deref());
    let outcome = dist::pull(
        args.model.as_deref(),
        &args.quant,
        &source,
        args.json,
        |line| {
            if !args.json {
                eprintln!("focr pull: {line}");
            }
        },
    )?;
    if args.json {
        emit(&serde_json::json!({
            "schema_version": robot::ROBOT_SCHEMA_VERSION,
            "command": "pull",
            "status": "ok",
            "quant": outcome.quant,
            "focrq": outcome.focrq_path.display().to_string(),
            "tokenizer": outcome.tokenizer_path.display().to_string(),
            "sidecars": outcome
                .sidecar_paths
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>(),
            "from_cache": outcome.from_cache,
            "model_license_notice": if outcome.license_notice.is_empty() {
                FOCR_MODEL_LICENSE_NOTICE
            } else {
                &outcome.license_notice
            },
        }));
    } else {
        eprintln!(
            "focr pull: ready — model at {} ({})",
            outcome.focrq_path.display(),
            if outcome.from_cache {
                "already cached"
            } else {
                "downloaded"
            }
        );
    }
    Ok(())
}

fn robot_backends_payload() -> serde_json::Value {
    let hardware_selected = simd::detected_tier();
    let effective_route = simd::effective_dense_route();
    let available: Vec<_> = simd::available_tiers()
        .iter()
        .map(|tier| {
            serde_json::json!({
                "tag": tier.tag(),
                "feature": tier.feature_string(),
            })
        })
        .collect();

    serde_json::json!({
        "schema_version": robot::ROBOT_SCHEMA_VERSION,
        "simd_tiers": {
            "selected": effective_route.tag(),
            "selected_feature": effective_route.feature_string(),
            "hardware_selected": hardware_selected.tag(),
            "hardware_selected_feature": hardware_selected.feature_string(),
            "available": available,
            "override_env": "FOCR_FORCE_ARCH",
            "selection_scope": "ordinary_dense_int8",
            "status": "runtime capability and effective-route selection active"
        },
        "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        // The ONE process-wide budget (bd-223.2 addendum: FOCR_THREADS else
        // physical cores — pool-sizing consumers read this, never logical).
        "threads": crate::thread_budget()
    })
}

/// `focr robot selftest` — re-run the dispatched int8 GEMM against the scalar
/// oracle on this exact CPU and emit a machine-checkable verdict. Emits the
/// report JSON (always, so robots see the per-shape detail) and THEN returns a
/// generic error (exit 1) if any case diverged, so the command can gate a CI /
/// post-install check. `FOCR_FORCE_ARCH` selects which available tier runs.
fn run_robot_selftest() -> FocrResult<()> {
    let report = simd::selftest();
    let cases: Vec<_> = report
        .cases
        .iter()
        .map(|c| {
            serde_json::json!({
                "kind": c.kind,
                "label": c.label,
                "m": c.m,
                "k": c.k,
                "n": c.n,
                "ok": c.ok,
                "mismatches": c.mismatches,
                "first_bad": c.first_bad.map(|(i, got, want)| serde_json::json!({
                    "index": i, "dispatched": got, "oracle": want,
                })),
            })
        })
        .collect();
    let available: Vec<_> = report.available.iter().map(|t| t.tag()).collect();
    let executed_routes: Vec<_> = report.executed_routes.iter().map(|r| r.tag()).collect();
    let passed = report.cases.iter().filter(|c| c.ok).count();
    let oracle_independent = !matches!(
        report.effective_route,
        simd::EffectiveI8Route::Autovec | simd::EffectiveI8Route::Scalar
    );
    emit(&serde_json::json!({
        "schema_version": robot::ROBOT_SCHEMA_VERSION,
        "command": "robot.selftest",
        "selected": report.effective_route.tag(),
        "selected_feature": report.effective_route.feature_string(),
        "hardware_selected": report.hardware_selected.tag(),
        "hardware_selected_feature": report.hardware_selected.feature_string(),
        "executed_routes": executed_routes,
        "route_consistent": report.route_consistent,
        "oracle_independent": oracle_independent,
        "available": available,
        "override_env": "FOCR_FORCE_ARCH",
        "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        "threads": crate::thread_budget(),
        "cases_total": report.cases.len(),
        "cases_passed": passed,
        "all_ok": report.all_ok,
        "verdict": if report.all_ok { "pass" } else { "fail" },
        // A12 (bd-3jo6.1.12): the machine-readable PER-MODEL verdict — every
        // registered int8 decoder's real kernel shapes proven bit-identical
        // to the scalar oracle on THIS host, incl its own worst-case-K row.
        "models": report.models.iter().map(|(id, ok)| serde_json::json!({
            "id": id, "verdict": if *ok { "pass" } else { "fail" },
        })).collect::<Vec<_>>(),
        "cases": cases,
    }));
    if report.all_ok {
        Ok(())
    } else {
        let failed = report.cases.len() - passed;
        Err(FocrError::Other(anyhow::anyhow!(
            "robot selftest: {failed}/{} parity case(s) diverged; route_consistent={} \
             expected={} observed={:?}, hardware={} — the dense int8 path is not certified on this CPU",
            report.cases.len(),
            report.route_consistent,
            report.effective_route.tag(),
            report
                .executed_routes
                .iter()
                .map(|r| r.tag())
                .collect::<Vec<_>>(),
            report.hardware_selected.feature_string(),
        )))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ErrorMode {
    Human,
    Robot,
}

impl ErrorMode {
    fn from_cli(cli: &Cli) -> Self {
        match &cli.command {
            Command::Ocr(args) if args.robot => Self::Robot,
            Command::Robot {
                cmd: RobotCmd::Run(_),
            } => Self::Robot,
            _ => Self::Human,
        }
    }
}

fn exit_code_from_error(err: &FocrError, mode: ErrorMode) -> ExitCode {
    match mode {
        // Error paths retire the detached renderer without waiting for
        // terminal I/O. Do not reintroduce that wait while printing the final
        // diagnostic; a stalled renderer makes this best-effort.
        ErrorMode::Human => {
            let _ = progress::try_stderr_message(format_args!("focr: {err}"));
        }
        ErrorMode::Robot => emit(&robot::run_error_event(err)),
    }
    ExitCode::from(exit_code_byte(err))
}

fn exit_code_byte(err: &FocrError) -> u8 {
    u8::try_from(err.exit_code()).unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_error_variant_maps_to_process_exit_byte_from_error_contract() {
        let cases = [
            (FocrError::Usage("bad flag".into()), 2),
            (FocrError::ModelNotFound("missing".into()), 3),
            (FocrError::InputDecode("bad image".into()), 4),
            (FocrError::Timeout("stage".into()), 5),
            (FocrError::Cancelled, 6),
            (FocrError::FormatMismatch("bad header".into()), 7),
            (FocrError::NotImplemented("phase gap".into()), 1),
            (FocrError::Other(anyhow::anyhow!("misc")), 1),
        ];
        for (err, code) in cases {
            eprintln!(
                "{}",
                serde_json::json!({
                    "suite": "cli",
                    "test": "every_error_variant_maps_to_process_exit_byte_from_error_contract",
                    "variant": err.kind(),
                    "exit_code": code,
                    "process_exit_byte": exit_code_byte(&err),
                })
            );
            assert_eq!(exit_code_byte(&err), code);
        }
    }

    #[test]
    fn long_version_carries_project_and_model_license_notices() {
        let report = long_version_report();
        assert!(report.contains("focr "));
        assert!(report.contains(FOCR_PROJECT_LICENSE_NOTICE));
        assert!(report.contains(&format!("model_license: {FOCR_MODEL_LICENSE_NOTICE}")));
    }

    #[test]
    fn exact_long_version_detection_only_matches_top_level_long_flag() {
        assert!(is_exact_long_version_request(
            ["focr", "--version"].into_iter().map(OsString::from)
        ));
        assert!(!is_exact_long_version_request(
            ["focr", "-V"].into_iter().map(OsString::from)
        ));
        assert!(!is_exact_long_version_request(
            ["focr", "--version", "robot"]
                .into_iter()
                .map(OsString::from)
        ));
    }

    #[test]
    fn robot_health_carries_single_source_model_license_notice() {
        let payload = robot_health_payload();
        assert_eq!(
            payload["model_license_notice"],
            serde_json::json!(FOCR_MODEL_LICENSE_NOTICE)
        );
    }

    #[test]
    fn ocr_robot_flag_selects_robot_error_mode() {
        let cli = Cli {
            command: Command::Ocr(OcrArgs {
                request: OcrRequestArgs {
                    image: PathBuf::from("scan.png"),
                    model: None,
                    base_size: DEFAULT_BASE_SIZE,
                    image_size: DEFAULT_IMAGE_SIZE,
                    crop_mode: CropMode::Gundam,
                    max_length: DEFAULT_MAX_LENGTH,
                    temperature: DEFAULT_TEMPERATURE,
                    no_repeat_ngram: DEFAULT_NO_REPEAT_NGRAM,
                    ngram_window: DEFAULT_NGRAM_WINDOW,
                    format: false,
                    task: OcrTask::Ocr,
                    question: None,
                    pages: None,
                    split_spreads: false,
                    multi_page: false,
                },
                json: false,
                output: None,
                extract_figures: false,
                figures_dir: None,
                robot: true,
            }),
        };
        assert_eq!(ErrorMode::from_cli(&cli), ErrorMode::Robot);
    }

    #[test]
    fn robot_run_selects_robot_error_mode() {
        let cli = Cli {
            command: Command::Robot {
                cmd: RobotCmd::Run(RobotRunArgs {
                    request: OcrRequestArgs {
                        image: PathBuf::from("scan.png"),
                        model: None,
                        base_size: DEFAULT_BASE_SIZE,
                        image_size: DEFAULT_IMAGE_SIZE,
                        crop_mode: CropMode::Gundam,
                        max_length: DEFAULT_MAX_LENGTH,
                        temperature: DEFAULT_TEMPERATURE,
                        no_repeat_ngram: DEFAULT_NO_REPEAT_NGRAM,
                        ngram_window: DEFAULT_NGRAM_WINDOW,
                        format: false,
                        task: OcrTask::Ocr,
                        question: None,
                        pages: None,
                        split_spreads: false,
                        multi_page: false,
                    },
                }),
            },
        };
        assert_eq!(ErrorMode::from_cli(&cli), ErrorMode::Robot);
    }

    #[test]
    fn ocr_args_validate_rejects_negative_size() {
        let args = OcrArgs {
            request: OcrRequestArgs {
                image: PathBuf::from("scan.png"),
                model: None,
                base_size: -1,
                image_size: DEFAULT_IMAGE_SIZE,
                crop_mode: CropMode::Gundam,
                max_length: DEFAULT_MAX_LENGTH,
                temperature: DEFAULT_TEMPERATURE,
                no_repeat_ngram: DEFAULT_NO_REPEAT_NGRAM,
                ngram_window: DEFAULT_NGRAM_WINDOW,
                format: false,
                task: OcrTask::Ocr,
                question: None,
                pages: None,
                split_spreads: false,
                multi_page: false,
            },
            json: false,
            output: None,
            extract_figures: false,
            figures_dir: None,
            robot: false,
        };
        let err = args.to_request().expect_err("negative base-size is usage");
        assert!(matches!(err, FocrError::Usage(_)), "got {err:?}");
        assert_eq!(err.exit_code(), 2);
    }

    #[test]
    fn output_is_json_follows_extension_case_insensitively() {
        assert!(output_is_json(Some(Path::new("out.json"))));
        assert!(output_is_json(Some(Path::new("OUT.JSON"))));
        assert!(output_is_json(Some(Path::new("/tmp/a/b.Json"))));
        // A `.md` / other / missing extension stays markdown.
        assert!(!output_is_json(Some(Path::new("out.md"))));
        assert!(!output_is_json(Some(Path::new("out.txt"))));
        assert!(!output_is_json(Some(Path::new("out"))));
        assert!(!output_is_json(None));
    }

    #[test]
    fn ocr_output_flag_parses_short_and_long() {
        for flag in ["-o", "--output"] {
            let cli = Cli::try_parse_from(["focr", "ocr", "scan.png", flag, "result.json"])
                .expect("ocr -o/--output parses");
            let Command::Ocr(args) = cli.command else {
                unreachable!("expected ocr command");
            };
            assert_eq!(args.output.as_deref(), Some(Path::new("result.json")));
        }
        // No `-o` => None, i.e. the legacy stdout path.
        let cli = Cli::try_parse_from(["focr", "ocr", "scan.png"]).expect("ocr parses");
        let Command::Ocr(args) = cli.command else {
            unreachable!("expected ocr command");
        };
        assert!(args.output.is_none());
    }

    #[test]
    fn ocr_format_flag_threads_to_request() {
        // `--format` (GOT `OCR with format:` .mmd mode) parses and reaches OcrRequest;
        // absent, it defaults false (plain OCR — byte-identical to today).
        let cli = Cli::try_parse_from(["focr", "ocr", "scan.png", "--format"])
            .expect("ocr --format parses");
        let Command::Ocr(args) = cli.command else {
            unreachable!("expected ocr command");
        };
        assert!(args.to_request().expect("request builds").format);
        let cli = Cli::try_parse_from(["focr", "ocr", "scan.png"]).expect("ocr parses");
        let Command::Ocr(args) = cli.command else {
            unreachable!("expected ocr command");
        };
        assert!(!args.to_request().expect("request builds").format);
    }

    /// Parse `focr ocr <argv…>` and build the request (panics on non-ocr).
    fn ocr_request_from(argv: &[&str]) -> FocrResult<OcrRequest> {
        let full: Vec<&str> = ["focr", "ocr"].iter().chain(argv).copied().collect();
        let cli = Cli::try_parse_from(full).expect("ocr argv parses");
        let Command::Ocr(args) = cli.command else {
            unreachable!("expected ocr command");
        };
        args.to_request()
    }

    #[test]
    fn ocr_task_flag_threads_format_to_request() {
        // Each got-ocr2 structured task implies `--format` (the same engine seam
        // `--format` uses); `--task ocr` / no `--task` stay plain (format=false),
        // byte-identical to today.
        for task in [
            "formula",
            "tables",
            "chart",
            "molecular",
            "geometry",
            "music",
        ] {
            let req =
                ocr_request_from(&["scan.png", "--task", task, "--model", "got-ocr2.int8.focrq"])
                    .expect("request builds");
            assert!(req.format, "--task {task} must imply format");
        }
        let req = ocr_request_from(&["scan.png", "--task", "ocr"]).expect("request builds");
        assert!(!req.format, "--task ocr stays plain");
        let req = ocr_request_from(&["scan.png"]).expect("request builds");
        assert!(!req.format, "default task stays plain");
    }

    #[test]
    fn ocr_task_composes_with_explicit_format() {
        // Explicit `--format` wins / is idempotent: `--task ocr --format` keeps
        // format=true (the task default never masks the flag), and adding
        // `--format` to a format-implying task changes nothing.
        let req =
            ocr_request_from(&["scan.png", "--task", "ocr", "--format"]).expect("request builds");
        assert!(req.format, "--format must not be masked by --task ocr");
        let with_both = ocr_request_from(&[
            "scan.png",
            "--task",
            "tables",
            "--format",
            "--model",
            "got-ocr2.int8.focrq",
        ])
        .expect("request builds");
        let task_only = ocr_request_from(&[
            "scan.png",
            "--task",
            "tables",
            "--model",
            "got-ocr2.int8.focrq",
        ])
        .expect("request builds");
        assert!(
            with_both.format && task_only.format,
            "--format is idempotent with --task"
        );
    }

    #[test]
    fn ocr_task_describe_fails_clean_naming_smolvlm2() {
        // `describe` (C9) needs the smolvlm2 model: the default resolution is
        // knowably unlimited-ocr, so guide (Usage, exit 2) instead of silently
        // running plain OCR — the got-only-task precedent.
        let err = ocr_request_from(&["photo.jpg", "--task", "describe"])
            .expect_err("describe against the default model must guide");
        assert!(matches!(err, FocrError::Usage(_)), "got {err:?}");
        assert_eq!(err.exit_code(), 2);
        let msg = err.to_string();
        assert!(
            msg.contains("smolvlm2"),
            "must name the required model: {msg}"
        );
        // A smolvlm2 model spec passes through and carries the question.
        let req = ocr_request_from(&[
            "photo.jpg",
            "--task",
            "describe",
            "--model",
            "smolvlm2.int8.focrq",
            "--question",
            "What color is the car?",
        ])
        .expect("describe with a smolvlm2 model spec");
        assert_eq!(req.question.as_deref(), Some("What color is the car?"));
        assert!(!req.format, "describe must not imply GOT --format");
        // `--question` without `--task describe` is a usage error.
        let err = ocr_request_from(&["photo.jpg", "--question", "what?"])
            .expect_err("--question requires --task describe");
        assert!(matches!(err, FocrError::Usage(_)), "got {err:?}");
        // A got/unlimited model spec is knowably wrong for describe.
        let err = ocr_request_from(&[
            "photo.jpg",
            "--task",
            "describe",
            "--model",
            "got-ocr2.int8.focrq",
        ])
        .expect_err("describe against a got model must guide");
        assert!(matches!(err, FocrError::Usage(_)), "got {err:?}");
    }

    #[test]
    fn ocr_task_got_only_task_guides_to_got_model() {
        // A `--model` that knowably names unlimited-ocr cannot serve a got-only
        // task: Usage guidance (exit 2) pointing at `focr pull got-ocr2`.
        let err = ocr_request_from(&[
            "scan.png",
            "--task",
            "formula",
            "--model",
            "unlimited-ocr.int8.focrq",
        ])
        .expect_err("unlimited-ocr cannot serve --task formula");
        assert!(matches!(err, FocrError::Usage(_)), "got {err:?}");
        assert_eq!(err.exit_code(), 2);
        let msg = err.to_string();
        assert!(
            msg.contains("focr pull got-ocr2"),
            "must carry the pull hint: {msg}"
        );
        assert!(
            msg.contains("--task formula"),
            "must name the offending task: {msg}"
        );

        // No `--model` ⇒ the default resolution (always unlimited-ocr) — same
        // guidance. Read-only env guard (no set_var under deny(unsafe)): only
        // assert when FOCR_MODEL_PATH is not overriding the default; the pure
        // classifier below covers the None case unconditionally.
        if std::env::var_os(crate::MODEL_PATH_ENV).is_none() {
            let err = ocr_request_from(&["scan.png", "--task", "music"])
                .expect_err("default model cannot serve --task music");
            assert!(matches!(err, FocrError::Usage(_)), "got {err:?}");
        }

        // A got-named model passes and carries format (case-insensitive name).
        let req = ocr_request_from(&[
            "scan.png",
            "--task",
            "geometry",
            "--model",
            "/models/GOT-OCR2.int8.focrq",
        ])
        .expect("got model serves geometry");
        assert!(req.format);

        // The pure classifier: default/unlimited are knowably-not-got; a got or
        // ambiguous name passes through to the engine's arch-tag dispatch.
        assert!(model_spec_is_knowably_not_got(None));
        assert!(model_spec_is_knowably_not_got(Some(Path::new(
            "/m/unlimited-ocr.int8.focrq"
        ))));
        assert!(!model_spec_is_knowably_not_got(Some(Path::new(
            "got-ocr2.int8.focrq"
        ))));
        assert!(!model_spec_is_knowably_not_got(Some(Path::new(
            "/m/custom.focrq"
        ))));
    }

    #[test]
    fn ocr_task_rejects_unknown_value_and_composes_with_robot_run() {
        // Clap owns the value set: an unknown task is a parse error (usage).
        assert!(Cli::try_parse_from(["focr", "ocr", "scan.png", "--task", "poetry"]).is_err());
        // `focr robot run` flattens the same OcrRequestArgs, so `--task`
        // composes with robot mode identically.
        let cli = Cli::try_parse_from([
            "focr",
            "robot",
            "run",
            "scan.png",
            "--task",
            "chart",
            "--model",
            "got-ocr2.int8.focrq",
        ])
        .expect("robot run --task parses");
        let Command::Robot {
            cmd: RobotCmd::Run(args),
        } = cli.command
        else {
            unreachable!("expected robot run");
        };
        assert!(args.request.to_request().expect("request builds").format);
    }

    #[test]
    fn preprocess_flags_become_overrides_only_when_explicit() {
        // Defaults ⇒ NO overrides: the engine keeps its certified Base-1024.
        let cli = Cli::try_parse_from(["focr", "ocr", "scan.png"]).expect("ocr parses");
        let Command::Ocr(args) = cli.command else {
            unreachable!("expected ocr command");
        };
        let req = args.to_request().expect("request builds");
        assert_eq!(
            preprocess_overrides_from(&req),
            native_engine::PreprocessOverrides::default()
        );

        // Explicit flags ⇒ each maps onto the engine overrides; gundam is the
        // only crop-mode value that produces one (base IS the engine default).
        let cli = Cli::try_parse_from([
            "focr",
            "ocr",
            "scan.png",
            "--base-size",
            "512",
            "--image-size",
            "512",
            "--crop-mode",
            "gundam",
        ])
        .expect("ocr with preprocess flags parses");
        let Command::Ocr(args) = cli.command else {
            unreachable!("expected ocr command");
        };
        let o = preprocess_overrides_from(&args.to_request().expect("request builds"));
        assert_eq!(o.base_size, Some(512));
        assert_eq!(o.image_size, Some(512));
        assert_eq!(o.gundam, Some(true));

        let cli = Cli::try_parse_from(["focr", "ocr", "scan.png", "--crop-mode", "base"])
            .expect("ocr parses");
        let Command::Ocr(args) = cli.command else {
            unreachable!("expected ocr command");
        };
        let o = preprocess_overrides_from(&args.to_request().expect("request builds"));
        assert_eq!(o.gundam, None);
    }

    #[test]
    fn tuning_flags_become_decode_overrides_only_when_explicit() {
        // Default flags ⇒ NO overrides: engine defaults + env (FOCR_MAX_NEW_TOKENS)
        // stay in force.
        let cli = Cli::try_parse_from(["focr", "ocr", "scan.png"]).expect("ocr parses");
        let Command::Ocr(args) = cli.command else {
            unreachable!("expected ocr command");
        };
        let req = args.to_request().expect("request builds");
        assert_eq!(
            decode_overrides_from(&req),
            native_engine::DecodeOverrides::default()
        );

        // Explicit flags ⇒ each maps to Some(value) on the engine overrides.
        let cli = Cli::try_parse_from([
            "focr",
            "ocr",
            "scan.png",
            "--max-length",
            "700",
            "--temperature",
            "0.5",
            "--no-repeat-ngram",
            "20",
            "--ngram-window",
            "1024",
        ])
        .expect("ocr with tuning flags parses");
        let Command::Ocr(args) = cli.command else {
            unreachable!("expected ocr command");
        };
        let o = decode_overrides_from(&args.to_request().expect("request builds"));
        assert_eq!(o.max_length, Some(700));
        assert_eq!(o.temperature, Some(0.5));
        assert_eq!(o.no_repeat_ngram, Some(20));
        assert_eq!(o.ngram_window, Some(1024));

        // Explicitly passing the default value is indistinguishable from default
        // (and behaviorally identical), so it maps to no override.
        let cli = Cli::try_parse_from(["focr", "ocr", "scan.png", "--max-length", "32768"])
            .expect("ocr parses");
        let Command::Ocr(args) = cli.command else {
            unreachable!("expected ocr command");
        };
        let o = decode_overrides_from(&args.to_request().expect("request builds"));
        assert_eq!(o.max_length, None);
    }

    #[test]
    fn single_image_json_carries_markdown_and_bounding_boxes() {
        let rec = Recognition::Single(native_engine::RecognizedDocument {
            markdown: "# Title\n\nbody".to_string(),
            layout: vec![native_engine::LayoutSpan {
                label: "title".to_string(),
                boxes: vec![[10, 20, 110, 60]],
            }],
        });
        let json = rec.to_json(&[]);
        assert_eq!(json["schema_version"], robot::ROBOT_SCHEMA_VERSION);
        assert_eq!(json["markdown"], "# Title\n\nbody");
        assert_eq!(json["layout"][0]["label"], "title");
        assert_eq!(
            json["layout"][0]["boxes"][0],
            serde_json::json!([10, 20, 110, 60])
        );
        // A single image has no per-page `pages` array.
        assert!(json.get("pages").is_none());
    }

    /// bd-av64.11: the --pages spec parser — 1-based pages/ranges to
    /// deduplicated source-order 0-based indices, with loud usage errors.
    /// bd-av64.2: the --json staves array interleaves recognized and skipped
    /// staves in DETECTION order, 1-based, with reasons only on skips.
    #[test]
    fn music_meta_json_interleaves_in_detection_order() {
        let meta = native_engine::MusicPageMeta {
            staves: vec![(0, (0, 10, 800, 100)), (2, (0, 300, 800, 100))],
            skips: vec![native_engine::tromr::StaffSkip {
                index: 1,
                bbox: (0, 150, 800, 90),
                reason: "resized width 1296 exceeds the 1280 position clamp".into(),
            }],
            warnings: Vec::new(),
        };
        let v = music_meta_to_json(&meta);
        let arr = v.as_array().expect("array");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0]["staff"], 1);
        assert_eq!(arr[0]["status"], "ok");
        assert!(arr[0].get("reason").is_none());
        assert_eq!(arr[1]["staff"], 2);
        assert_eq!(arr[1]["status"], "skipped");
        assert!(
            arr[1]["reason"].as_str().unwrap_or("").contains("1280"),
            "skip carries the reason"
        );
        assert_eq!(arr[2]["staff"], 3);
        assert_eq!(arr[2]["bbox"], serde_json::json!([0, 300, 800, 100]));
    }

    #[test]
    fn page_spec_parses_and_rejects() {
        // None => every page.
        assert_eq!(parse_page_spec(None, 3).unwrap(), vec![0, 1, 2]);
        assert_eq!(parse_page_spec(Some("3"), 218).unwrap(), vec![2]);
        assert_eq!(
            parse_page_spec(Some("3-7"), 10).unwrap(),
            vec![2, 3, 4, 5, 6]
        );
        assert_eq!(
            parse_page_spec(Some("1,5-9,218"), 218).unwrap(),
            vec![0, 4, 5, 6, 7, 8, 217]
        );
        // Overlap/duplicates dedupe; output is source-ordered.
        assert_eq!(
            parse_page_spec(Some("5-7,6,1"), 10).unwrap(),
            vec![0, 4, 5, 6]
        );
        // Round-trip property: rendering the indices back as 1-based pages
        // and reparsing is a fixed point.
        let idx = parse_page_spec(Some("2,4-6"), 9).unwrap();
        let rendered = idx
            .iter()
            .map(|i| (i + 1).to_string())
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(parse_page_spec(Some(&rendered), 9).unwrap(), idx);
        for bad in ["", " ", ",", "0", "abc", "3-2", "1-", "-4", "300"] {
            let err = parse_page_spec(Some(bad), 218).unwrap_err();
            assert!(
                matches!(err, FocrError::Usage(_)),
                "{bad:?} must be a usage error, got {err:?}"
            );
            assert!(
                err.to_string().contains("218"),
                "{bad:?}: error names the page count: {err}"
            );
        }
    }

    #[test]
    fn pdf_json_carries_per_page_layout_with_one_based_page_numbers() {
        let rec = Recognition::Pdf(PdfRecognition {
            markdown: "p1\n\np2".to_string(),
            pages: vec![
                PdfPageLayout {
                    page: 1,
                    half: None,
                    layout: vec![native_engine::LayoutSpan {
                        label: "text".to_string(),
                        boxes: vec![[0, 0, 5, 5]],
                    }],
                },
                PdfPageLayout {
                    page: 2,
                    half: None,
                    layout: vec![],
                },
            ],
        });
        let json = rec.to_json(&[]);
        assert_eq!(json["markdown"], "p1\n\np2");
        assert_eq!(json["pages"][0]["page"], 1);
        assert_eq!(
            json["pages"][0]["layout"][0]["boxes"][0],
            serde_json::json!([0, 0, 5, 5])
        );
        assert_eq!(json["pages"][1]["page"], 2);
        assert_eq!(json["pages"][1]["layout"], serde_json::json!([]));
    }

    #[test]
    fn write_ocr_output_writes_markdown_and_json_with_boxes() {
        let dir = std::env::temp_dir().join(format!("focr_output_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let rec = Recognition::Single(native_engine::RecognizedDocument {
            markdown: "hello world".to_string(),
            layout: vec![native_engine::LayoutSpan {
                label: "text".to_string(),
                boxes: vec![[1, 2, 3, 4]],
            }],
        });

        // Markdown form: source lacks a trailing newline, so one is appended.
        let md_path = dir.join("out.md");
        write_ocr_output(&md_path, &rec, false, &[], None).expect("write md");
        assert_eq!(std::fs::read_to_string(&md_path).unwrap(), "hello world\n");

        // JSON form: valid JSON, newline-terminated, carrying the bounding boxes.
        let json_path = dir.join("out.json");
        write_ocr_output(&json_path, &rec, true, &[], None).expect("write json");
        let raw = std::fs::read_to_string(&json_path).unwrap();
        assert!(raw.ends_with('\n'), "json file should end with a newline");
        let parsed: serde_json::Value = serde_json::from_str(&raw).expect("valid json");
        assert_eq!(parsed["markdown"], "hello world");
        assert_eq!(
            parsed["layout"][0]["boxes"][0],
            serde_json::json!([1, 2, 3, 4])
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A default `OcrArgs` for the figure-plan tests, mutated by `f`.
    fn ocr_args_with(f: impl FnOnce(&mut OcrArgs)) -> OcrArgs {
        let mut args = OcrArgs {
            request: OcrRequestArgs {
                image: PathBuf::from("scan.png"),
                model: None,
                base_size: DEFAULT_BASE_SIZE,
                image_size: DEFAULT_IMAGE_SIZE,
                crop_mode: CropMode::Gundam,
                max_length: DEFAULT_MAX_LENGTH,
                temperature: DEFAULT_TEMPERATURE,
                no_repeat_ngram: DEFAULT_NO_REPEAT_NGRAM,
                ngram_window: DEFAULT_NGRAM_WINDOW,
                format: false,
                task: OcrTask::Ocr,
                question: None,
                pages: None,
                split_spreads: false,
                multi_page: false,
            },
            json: false,
            output: None,
            extract_figures: false,
            figures_dir: None,
            robot: false,
        };
        f(&mut args);
        args
    }

    #[test]
    fn extract_figures_flag_parses() {
        let cli = Cli::try_parse_from([
            "focr",
            "ocr",
            "scan.png",
            "-o",
            "out.md",
            "--extract-figures",
        ])
        .expect("--extract-figures parses");
        let Command::Ocr(args) = cli.command else {
            unreachable!("expected ocr command");
        };
        assert!(args.extract_figures);
        let cli = Cli::try_parse_from(["focr", "ocr", "scan.png", "--figures-dir", "assets"])
            .expect("--figures-dir parses");
        let Command::Ocr(args) = cli.command else {
            unreachable!("expected ocr command");
        };
        assert_eq!(args.figures_dir.as_deref(), Some(Path::new("assets")));
    }

    #[test]
    fn figure_plan_resolves_auto_subfolder_explicit_dir_and_usage_error() {
        // Auto: `<stem>_figures/` next to the `-o` file.
        let plan = FigurePlan::resolve(&ocr_args_with(|a| {
            a.extract_figures = true;
            a.output = Some(PathBuf::from("/a/b/report.md"));
        }))
        .unwrap()
        .expect("enabled");
        assert_eq!(plan.dir, PathBuf::from("/a/b/report_figures"));
        assert_eq!(plan.ref_prefix, "report_figures/");

        // Explicit relative dir: resolved under the output dir; verbatim in refs.
        let plan = FigurePlan::resolve(&ocr_args_with(|a| {
            a.figures_dir = Some(PathBuf::from("assets"));
            a.output = Some(PathBuf::from("/a/b/report.md"));
        }))
        .unwrap()
        .expect("enabled");
        assert_eq!(plan.dir, PathBuf::from("/a/b/assets"));
        assert_eq!(plan.ref_prefix, "assets/");

        // Off when neither flag is set.
        assert!(
            FigurePlan::resolve(&ocr_args_with(|_| {}))
                .unwrap()
                .is_none()
        );

        // `--extract-figures` with no `-o` and no `--figures-dir` is a usage error.
        let err = FigurePlan::resolve(&ocr_args_with(|a| a.extract_figures = true))
            .expect_err("needs a place for the subfolder");
        assert!(matches!(err, FocrError::Usage(_)), "got {err:?}");
    }

    #[test]
    fn choose_figure_format_png_for_flat_jpg_for_photo() {
        // Flat 2-color line-art ⇒ PNG (lossless).
        let mut flat = image::RgbImage::new(64, 64);
        for (i, px) in flat.pixels_mut().enumerate() {
            *px = if i % 9 == 0 {
                image::Rgb([0, 0, 0])
            } else {
                image::Rgb([255, 255, 255])
            };
        }
        assert_eq!(
            choose_figure_format(&image::DynamicImage::ImageRgb8(flat)),
            FigureFormat::Png
        );

        // Many distinct colors (photo-like) ⇒ JPG. A per-pixel `(x*4, y*4, x^y)`
        // ramp gives ~4096 distinct colors (≈1024 after 5-bit quantization), well
        // above the line-art threshold.
        let mut photo = image::RgbImage::new(64, 64);
        for (i, px) in photo.pixels_mut().enumerate() {
            let x = (i % 64) as u8;
            let y = (i / 64) as u8;
            *px = image::Rgb([x.wrapping_mul(4), y.wrapping_mul(4), x ^ (y << 1)]);
        }
        assert_eq!(
            choose_figure_format(&image::DynamicImage::ImageRgb8(photo)),
            FigureFormat::Jpeg
        );
    }

    #[test]
    fn figure_writer_writes_file_and_rewrites_markdown_reference() {
        let dir = std::env::temp_dir().join(format!("focr_figwriter_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let plan = FigurePlan {
            dir: dir.clone(),
            ref_prefix: "figs/".to_string(),
        };
        let mut writer = plan.writer();
        // A flat white image ⇒ PNG; bbox + ref carried through to the record.
        let fig = native_engine::ExtractedFigure {
            index: 0,
            label: "image".to_string(),
            bbox: [5, 6, 25, 16],
            markdown_ref: "![](images/0.jpg)".to_string(),
            image: image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                20,
                10,
                image::Rgb([255, 255, 255]),
            )),
        };
        let md = writer
            .process_page(1, "before ![](images/0.jpg)\nafter", vec![fig])
            .expect("process page");

        // The placeholder is rewritten to `![figure 1](<ref_prefix><name>)`.
        assert!(
            md.contains("![figure 1](figs/page1_figure_1.png)"),
            "md: {md}"
        );
        assert!(!md.contains("images/0.jpg"), "old token gone; md: {md}");
        // The PNG file actually exists.
        assert!(dir.join("page1_figure_1.png").is_file());
        // The JSON record carries the relative path, page, and bbox.
        let written = writer.into_written();
        assert_eq!(written.len(), 1);
        assert_eq!(written[0].path, "figs/page1_figure_1.png");
        assert_eq!(written[0].page, 1);
        assert_eq!(written[0].bbox, [5, 6, 25, 16]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn json_appends_figures_array_only_when_present() {
        let rec = Recognition::Single(native_engine::RecognizedDocument {
            markdown: "see ![figure 1](figs/page1_figure_1.png)".to_string(),
            layout: vec![],
        });
        let figures = vec![WrittenFigure {
            label: "image".to_string(),
            page: 1,
            bbox: [1, 2, 3, 4],
            path: "figs/page1_figure_1.png".to_string(),
        }];
        let json = rec.to_json(&figures);
        assert_eq!(json["figures"][0]["path"], "figs/page1_figure_1.png");
        assert_eq!(json["figures"][0]["page"], 1);
        assert_eq!(json["figures"][0]["bbox"], serde_json::json!([1, 2, 3, 4]));
        // No figures ⇒ no `figures` key.
        assert!(rec.to_json(&[]).get("figures").is_none());
    }

    #[test]
    fn models_json_describes_the_registered_archs() {
        let archs = crate::model_arch::registry();
        assert!(!archs.is_empty());
        let j = model_arch_json(archs[0]);
        assert_eq!(j["id"], "unlimited-ocr");
        assert_eq!(j["status"], "ready");
        assert_eq!(j["implemented"], true);
        assert_eq!(j["tasks"], serde_json::json!(["ocr"]));
        assert_eq!(j["decoder"], "DeepSeekV2MoeRswa");
        assert_eq!(j["vision_encoder"], "SamClip");
        assert!(j["license"].as_str().unwrap_or_default().contains("Baidu"));
    }

    /// Every runtime-ready arch has committed distribution provenance and at
    /// least one exact-recipe pull. Recipe mismatches remain independently
    /// fail-closed in the distribution layer.
    #[test]
    fn ready_archs_have_honest_committed_pull_status() {
        let m = crate::dist::builtin_manifest().expect("embedded manifest parses");
        for a in crate::model_arch::registry() {
            let pull = pull_availability(Some(&m), a.id());
            if a.implemented() {
                assert!(
                    pull.in_manifest,
                    "{} is runtime-ready but has no manifest entry — publish its \
                     artifacts (bd-av64.7 pattern) or record why not",
                    a.id()
                );
                assert!(
                    !pull.compatible.is_empty(),
                    "{} must retain at least one compatible pull",
                    a.id()
                );
                assert!(pull.blocked.is_empty());
            } else {
                assert!(
                    !pull.in_manifest,
                    "{} is planned-only but published in the manifest",
                    a.id()
                );
            }
        }
    }

    #[test]
    fn task_name_is_stable_lowercase() {
        use crate::model_arch::Task;
        assert_eq!(task_name(Task::Ocr), "ocr");
        assert_eq!(task_name(Task::Music), "music");
        assert_eq!(task_name(Task::Describe), "describe");
        assert_eq!(task_name(Task::Chart), "chart");
    }

    #[test]
    fn models_command_parses() {
        let cli = Cli::try_parse_from(["focr", "models"]).expect("focr models parses");
        assert!(matches!(cli.command, Command::Models(_)));
        let cli = Cli::try_parse_from(["focr", "models", "--json"]).expect("--json parses");
        let Command::Models(args) = cli.command else {
            unreachable!("expected models");
        };
        assert!(args.json);
    }

    #[test]
    fn convert_arch_enum_parses() {
        let parsed = Cli::try_parse_from([
            "focr",
            "convert",
            "in.safetensors",
            "-o",
            "out.focrq",
            "--arch",
            "x86-vnni",
        ]);
        let parse_error = parsed
            .as_ref()
            .err()
            .map(std::string::ToString::to_string)
            .unwrap_or_default();
        assert!(parsed.is_ok(), "convert --arch parses: {parse_error}");
        let Ok(cli) = parsed else {
            return;
        };
        let is_convert = matches!(cli.command, Command::Convert(_));
        assert!(is_convert, "expected convert command");
        if let Command::Convert(args) = cli.command {
            assert_eq!(args.quant, QuantTarget::Int8);
            assert_eq!(args.arch, ArchTarget::X86Vnni);
        };
    }

    #[test]
    fn robot_backends_reflects_simd_dispatch_snapshot() {
        let payload = robot_backends_payload();
        let tiers = &payload["simd_tiers"];
        let effective = simd::effective_dense_route();
        let hardware = simd::detected_tier();
        assert_eq!(payload["schema_version"], robot::ROBOT_SCHEMA_VERSION);
        assert_eq!(tiers["selected"], effective.tag());
        assert_eq!(tiers["selected_feature"], effective.feature_string());
        assert_eq!(tiers["hardware_selected"], hardware.tag());
        assert_eq!(
            tiers["hardware_selected_feature"],
            hardware.feature_string()
        );
        assert_eq!(tiers["override_env"], "FOCR_FORCE_ARCH");

        assert!(
            tiers["available"].as_array().is_some_and(|available| {
                !available.is_empty()
                    && available.last().and_then(|v| v["tag"].as_str())
                        == Some(simd::IsaTier::Scalar.tag())
            }),
            "available tiers must be a non-empty array ending with the scalar floor"
        );
    }
}
