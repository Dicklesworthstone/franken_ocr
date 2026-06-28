//! The `focr` clap-derive CLI surface (plan §7.2).
//!
//! Subcommands are Phase-0 skeleton stubs: the diagnostics (`robot
//! schema/health/backends`) work today; `ocr` routes through the native model
//! resolver/engine skeleton and then fails cleanly at the first unimplemented
//! stage, while `convert` and `doctor` return clear `NotImplemented` errors
//! pointing at the plan phase that lands them. PDF input is intentionally absent
//! — v1 is image-only (plan §7.7).
//!
//! This module lives in the **library** so the single CLI entrypoint
//! ([`cli_main`]) is shared by both binaries (`focr` and `franken_ocr`) without
//! either `src/main.rs` appearing in two build targets — each `[[bin]]` now
//! points at its own thin shim that just calls [`cli_main`]. See AGENTS.md
//! doctrine #9.

use crate::{
    FOCR_MODEL_LICENSE_NOTICE, FOCR_PROJECT_LICENSE_NOTICE, FocrError, FocrResult, OcrEngine,
    native_engine, robot, simd,
};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::ffi::OsString;
use std::path::PathBuf;
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
pub struct RobotRunArgs {
    #[command(flatten)]
    pub request: OcrRequestArgs,
}

#[derive(Clone, Debug, Args)]
pub struct OcrRequestArgs {
    /// Input document image path (v1 image-only; rasterization is out-of-band).
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
        Command::Ocr(args) if args.robot => {
            emit(&robot::run_start_event("ocr"));
            run_ocr(args, true)
        }
        Command::Ocr(args) => run_ocr(args, false),
        Command::OcrBatch(args) => run_ocr_batch(args),
        Command::Convert(args) => run_convert(&args),
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
    let markdown = match request.model.as_deref() {
        Some(model) => engine.recognize_with_model(model, &request.image)?,
        None => engine.recognize(&request.image)?,
    };
    if robot_mode {
        emit(&serde_json::json!({
            "schema_version": robot::ROBOT_SCHEMA_VERSION,
            "event": "run_complete",
        }));
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

fn run_convert(args: &ConvertArgs) -> FocrResult<()> {
    if args.json {
        emit(&serde_json::json!({
            "schema_version": robot::ROBOT_SCHEMA_VERSION,
            "command": "convert",
            "status": "scaffold",
            "implemented": false,
            "landing_phase": "Phase 2",
            "plan_section": "§5",
            "input": args.input,
            "output": args.output,
            "quant": args.quant.as_str(),
            "arch": args.arch.as_str(),
        }));
    }
    Err(FocrError::NotImplemented(
        "focr convert — the weight transformer lands in Phase 2 (see plan §5)".into(),
    ))
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
