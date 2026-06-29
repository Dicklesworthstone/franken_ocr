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
    native_engine, pdf, quant, robot, simd,
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
/// Clap's script-friendly `focr <semver>` output.
#[must_use]
pub fn long_version_report() -> String {
    format!(
        "focr {}\nsource_license: {}\nmodel_license: {}\n",
        env!("CARGO_PKG_VERSION"),
        FOCR_PROJECT_LICENSE_NOTICE,
        FOCR_MODEL_LICENSE_NOTICE
    )
}

#[derive(Subcommand)]
pub enum Command {
    /// Parse a document image into structured markdown (or `--json`).
    Ocr(OcrArgs),
    /// OCR many images in ONE process — load the 6.2 GB weights + build the int8
    /// decoder cache ONCE, then stream a result per image (the throughput path).
    OcrBatch(OcrBatchArgs),
    /// Offline weight transformation: safetensors → `.focrq` (plan §5).
    Convert(ConvertArgs),
    /// Download the model weights (int8 `.focrq` + tokenizer) into the cache.
    Pull(PullArgs),
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
    /// Explicit model artifact path (else the env-resolved default).
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// Emit machine-readable JSON (one object per image + a final summary).
    #[arg(long)]
    pub json: bool,
    /// Use the f32 decoder instead of the default int8 throughput path.
    #[arg(long = "f32")]
    pub no_int8: bool,
}

#[derive(Clone, Debug, Args)]
pub struct PullArgs {
    /// Quant level to fetch (only `int8` is published today).
    #[arg(long, default_value = dist::DEFAULT_QUANT)]
    pub quant: String,
    /// Manifest source — a local path or an `http(s)` URL. Defaults to
    /// `$FOCR_MANIFEST_URL`, else the built-in repo manifest.
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
    /// Vision preprocessing mode: reference Gundam tiling or base global view.
    #[arg(long, value_enum, default_value_t = CropMode::Gundam)]
    pub crop_mode: CropMode,
    /// Maximum generated sequence length.
    #[arg(long, default_value_t = DEFAULT_MAX_LENGTH)]
    pub max_length: i64,
    /// Decode temperature; 0.0 means greedy.
    #[arg(long, default_value_t = DEFAULT_TEMPERATURE)]
    pub temperature: f32,
    /// Sliding no-repeat n-gram size (env override: FOCR_NO_REPEAT_NGRAM).
    #[arg(
        long,
        env = "FOCR_NO_REPEAT_NGRAM",
        default_value_t = DEFAULT_NO_REPEAT_NGRAM
    )]
    pub no_repeat_ngram: i64,
    /// Sliding no-repeat n-gram lookback window.
    #[arg(long, default_value_t = DEFAULT_NGRAM_WINDOW)]
    pub ngram_window: i64,
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
            robot: true,
        }
    }
}

impl OcrRequestArgs {
    fn to_request(&self) -> FocrResult<OcrRequest> {
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
        })
    }
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
    /// Emit the scaffold capability/check contract as JSON.
    #[arg(long)]
    pub json: bool,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Subcommand)]
pub enum SyncCmd {
    /// Export run-state audit records as JSONL.
    ExportJsonl,
    /// Import run-state audit records from JSONL.
    ImportJsonl,
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
        Command::Ocr(args) if args.robot => {
            emit(&robot::run_start_event("ocr"));
            run_ocr(args, true)
        }
        Command::Ocr(args) => run_ocr(args, false),
        Command::OcrBatch(args) => run_ocr_batch(args),
        Command::Convert(args) => run_convert(&args),
        Command::Pull(args) => run_pull(&args),
        Command::Runs(args) => run_runs(&args),
        Command::Sync(args) => run_sync(&args),
        Command::Doctor(args) => run_doctor(&args),
    }
}

fn run_ocr(args: OcrArgs, robot_mode: bool) -> FocrResult<()> {
    let request = args.to_request()?;
    if let Some(err) = forced_test_error()? {
        return Err(err);
    }

    let engine = OcrEngine::new()?;
    // A `.pdf` (or `%PDF-`-magic) input rasterizes each page and OCRs them as one
    // document; everything else is a single decoded image. Both funnel through
    // `recognize_with_autodownload` so model resolution + the first-run download
    // offer behave identically.
    let markdown = if pdf::looks_like_pdf(&request.image) {
        recognize_pdf(&engine, &request, robot_mode)?
    } else {
        recognize_with_autodownload(&request, robot_mode, |model| match model {
            Some(m) => engine.recognize_with_model(m, &request.image),
            None => engine.recognize(&request.image),
        })?
    };
    if robot_mode {
        // The terminal success event carries the recognized markdown so a machine
        // consumer actually receives the OCR result on the NDJSON stream (the
        // human / `--json` modes print it below instead).
        emit(&robot::run_complete_event(&markdown));
    } else if args.json {
        emit(&serde_json::json!({
            "schema_version": robot::ROBOT_SCHEMA_VERSION,
            "markdown": markdown,
        }));
    } else {
        println!("{markdown}");
    }
    Ok(())
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
fn recognize_with_autodownload<F>(
    request: &OcrRequest,
    robot_mode: bool,
    recog: F,
) -> FocrResult<String>
where
    F: Fn(Option<&Path>) -> FocrResult<String>,
{
    match recog(request.model.as_deref()) {
        Ok(md) => Ok(md),
        Err(FocrError::ModelNotFound(msg)) => {
            if request.model.is_none() && !robot_mode && is_interactive() {
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

/// Recognize every page of a PDF and concatenate the per-page markdown into one
/// document (successful pages joined by a blank line).
///
/// Each page is rasterized in-memory by [`pdf`] and fed through the identical OCR
/// pipeline a PNG takes — no out-of-band `pdftoppm`. Pages render lazily, one at a
/// time, so a long book never holds every raster at once.
///
/// **Per-page resilience (mirrors the `ocr-batch` path):** a page that cannot be
/// rendered or recognized — an unsupported codec (`JPXDecode`/`JBIG2Decode`), a
/// vector/text page, a per-page decode or timeout error — is logged to stderr and
/// SKIPPED, so one bad page never discards (nor wastes the compute already spent
/// on) the OCR of every other page. The **whole-run** conditions are propagated
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
fn recognize_pdf(engine: &OcrEngine, request: &OcrRequest, robot_mode: bool) -> FocrResult<String> {
    let pages = pdf::PdfPages::open(&request.image)?;
    let page_count = pages.len();
    recognize_with_autodownload(request, robot_mode, |model| {
        let mut document = String::new();
        let mut ok_pages = 0usize;
        let mut first_error: Option<FocrError> = None;
        for idx in 0..page_count {
            let page = pages.render(idx).and_then(|image| match model {
                Some(m) => engine.recognize_dynamic_with_model(m, image),
                None => engine.recognize_dynamic(image),
            });
            match page {
                Ok(page_md) => {
                    if ok_pages > 0 {
                        document.push_str("\n\n");
                    }
                    document.push_str(page_md.trim_end());
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
                // Isolate every other per-page failure: warn and skip, keeping the
                // rest of the document. (stderr only, never in robot mode.)
                Err(e) => {
                    if !robot_mode {
                        eprintln!("[focr] PDF page {} skipped: {e}", idx + 1);
                    }
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
            }
        }
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
        Ok(document)
    })
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

/// Load-once batch OCR: build the model + int8 decoder cache ONCE (the
/// [`OcrEngine`] `Arc` cache amortizes the 6.2 GB weight read; the model's int8
/// `OnceLock` amortizes the ~1.2 s quant), then recognize every image in the same
/// process. Defaults to the int8 throughput decode path (`--f32` opts out). With
/// the continuous-batch spine armed (`FOCR_BATCH_SPINE`) all pages decode together
/// through the scheduler; otherwise the proven sequential per-image loop runs.
/// Either driver renders the SAME per-image output (bd-1azu.13).
fn run_ocr_batch(args: OcrBatchArgs) -> FocrResult<()> {
    if let Some(err) = forced_test_error()? {
        return Err(err);
    }
    if !args.no_int8 {
        native_engine::force_int8_decode(true);
    }
    let engine = OcrEngine::new()?;
    let model = args.model.clone();
    let count = args.images.len();
    let total = std::time::Instant::now();
    let mut results: Vec<serde_json::Value> = Vec::with_capacity(count);

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
        // byte-for-byte what it has always been.
        for image in &args.images {
            let started = std::time::Instant::now();
            let outcome = match model.as_deref() {
                Some(m) => engine.recognize_with_model(m, image),
                None => engine.recognize(image),
            };
            let secs = started.elapsed().as_secs_f64();
            emit_batch_result(args.json, image, secs, outcome, &mut results);
        }
    }

    let elapsed = total.elapsed().as_secs_f64();
    let per_image = elapsed / (count.max(1) as f64);
    if args.json {
        emit(&serde_json::json!({
            "schema_version": robot::ROBOT_SCHEMA_VERSION,
            "command": "ocr-batch",
            "count": count,
            "int8": !args.no_int8,
            "seconds_total": elapsed,
            "seconds_per_image": per_image,
            "results": results,
        }));
    } else {
        eprintln!(
            "[focr] batch complete: {count} images in {elapsed:.2}s ({per_image:.2}s/image, int8={})",
            !args.no_int8
        );
    }
    Ok(())
}

/// Offline weight transform: raw bf16 safetensors → a self-contained int8
/// `.focrq` (plan §5). The int8 decoder tensors are quantized with the SAME
/// [`native_engine::nn::quantize_int8`] the load-time `FOCR_DECODE_INT8` cache
/// uses, so the artifact decodes byte-for-byte like that path on the source
/// shard; everything else (vision, projector, embed_tokens, router gate, norms)
/// is copied verbatim. `--quant int4` is not yet validated (doctrine #1) and
/// returns `NotImplemented`.
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
    let quantized = weights
        .names()
        .filter(|name| quant::convert::is_decoder_int8_tensor(name))
        .count();

    let blob = quant::convert::safetensors_to_focrq(
        &weights,
        quant::convert::ConvertQuant::Int8,
        args.arch.packing_byte(),
        source_sha256,
    )?;
    let output_bytes = blob.len();
    std::fs::write(&args.output, &blob).map_err(|e| {
        FocrError::Other(anyhow::anyhow!(
            "writing .focrq to {}: {e}",
            args.output.display()
        ))
    })?;

    let sha_hex = hex_encode32(&source_sha256);
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
            "source_sha256": sha_hex,
            "tensors": tensor_count,
            "tensors_quantized": quantized,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
        }));
    } else {
        eprintln!(
            "[focr] convert: wrote {} ({} quant {}: {tensor_count} tensors, {quantized} int8, \
             {input_bytes} -> {output_bytes} bytes) source_sha256={sha_hex}",
            args.output.display(),
            args.arch.as_str(),
            args.quant.as_str(),
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
    let _limit = non_negative_u32("limit", args.limit)?;
    if args.json || args.format != OutputFormat::Plain {
        let format = if args.json {
            "json".to_owned()
        } else {
            args.format.to_string()
        };
        emit(&serde_json::json!({
            "schema_version": robot::ROBOT_SCHEMA_VERSION,
            "command": "runs",
            "status": "scaffold",
            "implemented": false,
            "landing_phase": "Phase 0",
            "plan_section": "§7.2",
            "id": args.id,
            "format": format,
        }));
    }
    Err(FocrError::NotImplemented(
        "focr runs — durable run history lands with the fsqlite RunStore in Phase 0 (see plan §7.2)".into(),
    ))
}

fn run_sync(args: &SyncArgs) -> FocrResult<()> {
    if args.json {
        emit(&serde_json::json!({
            "schema_version": robot::ROBOT_SCHEMA_VERSION,
            "command": "sync",
            "subcommand": match args.cmd {
                SyncCmd::ExportJsonl => "export-jsonl",
                SyncCmd::ImportJsonl => "import-jsonl",
            },
            "status": "scaffold",
            "implemented": false,
            "landing_phase": "Phase 0",
            "plan_section": "§7.2",
        }));
    }
    Err(FocrError::NotImplemented(
        "focr sync — JSONL audit export/import lands with the fsqlite RunStore in Phase 0 (see plan §7.2)".into(),
    ))
}

fn run_doctor(args: &DoctorArgs) -> FocrResult<()> {
    if args.json {
        emit(&doctor_scaffold_payload());
    }
    Err(FocrError::NotImplemented(
        "focr doctor — lands in Phase 5 (see plan §7)".into(),
    ))
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

fn doctor_scaffold_payload() -> serde_json::Value {
    serde_json::json!({
        "schema_version": robot::ROBOT_SCHEMA_VERSION,
        "command": "doctor",
        "status": "scaffold",
        "capabilities": [
            {
                "name": "model_resolution",
                "phase": "Phase 5",
                "idempotent": true,
                "reversible": true,
                "implemented": false
            },
            {
                "name": "format_version",
                "phase": "Phase 5",
                "idempotent": true,
                "reversible": true,
                "implemented": false
            },
            {
                "name": "permissions",
                "phase": "Phase 5",
                "idempotent": true,
                "reversible": true,
                "implemented": false
            }
        ],
        "checks": [
            {
                "name": "model_available",
                "status": "not_run",
                "landing_phase": "Phase 5"
            },
            {
                "name": "format_supported",
                "status": "not_run",
                "landing_phase": "Phase 5"
            },
            {
                "name": "paths_writable",
                "status": "not_run",
                "landing_phase": "Phase 5"
            }
        ]
    })
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
        "{msg} — run `focr pull` to download the int8 weights (~3.9 GB), or point \
         FOCR_MODEL_PATH at an existing model"
    )
}

/// Prompt on the TTY; if confirmed, download the default int8 model + tokenizer
/// and return where they landed (else `Ok(None)` when the user declines).
fn offer_first_run_download() -> FocrResult<Option<dist::PullOutcome>> {
    use std::io::Write as _;
    eprint!("focr: model not found. Download the int8 weights now (~3.9 GB)? [y/N] ");
    std::io::stderr().flush().ok();
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(|e| FocrError::Other(anyhow::anyhow!("reading prompt response: {e}")))?;
    if !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
        return Ok(None);
    }
    let source = dist::resolve_manifest_source(None);
    let outcome = dist::pull(dist::DEFAULT_QUANT, &source, false, |line| {
        eprintln!("focr pull: {line}");
    })?;
    Ok(Some(outcome))
}

/// `focr pull` — download (or confirm-cached) the model weights + tokenizer.
fn run_pull(args: &PullArgs) -> FocrResult<()> {
    let source = dist::resolve_manifest_source(args.manifest.as_deref());
    let outcome = dist::pull(&args.quant, &source, args.json, |line| {
        if !args.json {
            eprintln!("focr pull: {line}");
        }
    })?;
    if args.json {
        emit(&serde_json::json!({
            "schema_version": robot::ROBOT_SCHEMA_VERSION,
            "command": "pull",
            "status": "ok",
            "quant": outcome.quant,
            "focrq": outcome.focrq_path.display().to_string(),
            "tokenizer": outcome.tokenizer_path.display().to_string(),
            "from_cache": outcome.from_cache,
            "model_license_notice": FOCR_MODEL_LICENSE_NOTICE,
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
            "selected": simd::detected_tier().tag(),
            "selected_feature": simd::tier_string(),
            "available": available,
            "override_env": "FOCR_FORCE_ARCH",
            "status": "runtime detection active"
        },
        "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)
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
    let passed = report.cases.iter().filter(|c| c.ok).count();
    emit(&serde_json::json!({
        "schema_version": robot::ROBOT_SCHEMA_VERSION,
        "command": "robot.selftest",
        "selected": report.selected.tag(),
        "selected_feature": report.selected.feature_string(),
        "available": available,
        "override_env": "FOCR_FORCE_ARCH",
        "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        "cases_total": report.cases.len(),
        "cases_passed": passed,
        "all_ok": report.all_ok,
        "verdict": if report.all_ok { "pass" } else { "fail" },
        "cases": cases,
    }));
    if report.all_ok {
        Ok(())
    } else {
        let failed = report.cases.len() - passed;
        Err(FocrError::Other(anyhow::anyhow!(
            "robot selftest: {failed}/{} int8-kernel parity case(s) diverged from the scalar \
             oracle on tier {} — this binary's accelerated int8 path is NOT bit-exact on this CPU",
            report.cases.len(),
            report.selected.feature_string(),
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
        ErrorMode::Human => eprintln!("focr: {err}"),
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
                },
                json: false,
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
            },
            json: false,
            robot: false,
        };
        let err = args.to_request().expect_err("negative base-size is usage");
        assert!(matches!(err, FocrError::Usage(_)), "got {err:?}");
        assert_eq!(err.exit_code(), 2);
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
        assert_eq!(payload["schema_version"], robot::ROBOT_SCHEMA_VERSION);
        assert_eq!(tiers["selected"], simd::detected_tier().tag());
        assert_eq!(tiers["selected_feature"], simd::tier_string());
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
