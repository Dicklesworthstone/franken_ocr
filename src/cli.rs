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
pub struct ModelsArgs {
    /// Emit a machine-readable JSON list instead of a human table.
    #[arg(long)]
    pub json: bool,
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
            output: None,
            extract_figures: false,
            figures_dir: None,
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
                        serde_json::json!({
                            "page": p.page,
                            "layout": layout_to_json(&p.layout),
                        })
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
) -> FocrResult<()> {
    let contents = if want_json {
        let mut s = serde_json::to_string_pretty(&rec.to_json(figures)).map_err(|e| {
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
    let request = args.to_request()?;
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
    let (recognition, figures): (Recognition, Vec<WrittenFigure>) = match (&figure_plan, is_pdf) {
        (Some(plan), true) => {
            let (pdf_rec, figs) = recognize_pdf_with_figures(&engine, &request, robot_mode, plan)?;
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
        (None, true) => (
            Recognition::Pdf(recognize_pdf(&engine, &request, robot_mode)?),
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

    let markdown = recognition.markdown();
    // `--json` forces JSON; a `.json` output path selects it implicitly. (When no
    // `-o` is given, `output_is_json(None)` is false, so stdout behavior is exactly
    // the legacy `args.json` choice.)
    let want_json = args.json || output_is_json(args.output.as_deref());

    // An `-o/--output FILE` writes the result to disk (markdown or JSON-with-boxes)
    // regardless of mode, and is written FIRST so the file already exists when a
    // robot consumer sees the completion event below.
    if let Some(path) = args.output.as_deref() {
        write_ocr_output(path, &recognition, want_json, &figures)?;
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
        emit(&recognition.to_json(&figures));
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

/// One successfully-OCR'd PDF page's structured layout (for the JSON output).
struct PdfPageLayout {
    /// 1-based page number in the source PDF.
    page: usize,
    /// The page's parsed layout spans (labels + pixel bounding boxes).
    layout: Vec<crate::native_engine::LayoutSpan>,
}

/// A recognized PDF: the concatenated markdown plus per-page layout for the pages
/// that decoded (skipped pages are absent from `pages`, as from the markdown).
struct PdfRecognition {
    markdown: String,
    pages: Vec<PdfPageLayout>,
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
) -> FocrResult<PdfRecognition> {
    let pages = pdf::PdfPages::open(&request.image)?;
    let page_count = pages.len();
    recognize_with_autodownload(request, robot_mode, |model| {
        let mut document = String::new();
        let mut page_layouts: Vec<PdfPageLayout> = Vec::new();
        let mut ok_pages = 0usize;
        let mut first_error: Option<FocrError> = None;
        for idx in 0..page_count {
            let page = pages.render(idx).and_then(|image| match model {
                Some(m) => engine.recognize_dynamic_with_layout_model(m, image),
                None => engine.recognize_dynamic_with_layout(image),
            });
            match page {
                Ok(doc) => {
                    if ok_pages > 0 {
                        document.push_str("\n\n");
                    }
                    document.push_str(doc.markdown.trim_end());
                    page_layouts.push(PdfPageLayout {
                        page: idx + 1,
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
        let mut out: Vec<OkPage> = Vec::new();
        let mut first_error: Option<FocrError> = None;
        for idx in 0..page_count {
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
                        eprintln!("[focr] PDF page {} skipped: {e}", idx + 1);
                    }
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
            }
        }
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

/// `focr models` — list the model architectures this build can run (the "model
/// zoo", epic bd-3jo6). A human table by default; `--json` for a machine-readable
/// list. Today the registry holds the implemented Baidu Unlimited-OCR model; the
/// specialized zoo models appear here (as `planned`, then `ready`) once their
/// descriptors register.
fn run_models(args: &ModelsArgs) -> FocrResult<()> {
    let archs = crate::model_arch::registry();
    if args.json {
        let models: Vec<serde_json::Value> = archs.iter().map(|a| model_arch_json(*a)).collect();
        emit(&serde_json::json!({
            "schema_version": robot::ROBOT_SCHEMA_VERSION,
            "models": models,
        }));
    } else {
        println!("{:<14}  {:<8}  {:<22}  MODEL", "ID", "STATUS", "TASKS");
        for a in archs {
            let tasks = a
                .tasks()
                .iter()
                .map(|t| task_name(*t))
                .collect::<Vec<_>>()
                .join(",");
            let status = if a.implemented() { "ready" } else { "planned" };
            println!(
                "{:<14}  {:<8}  {:<22}  {}",
                a.id(),
                status,
                tasks,
                a.display_name()
            );
        }
    }
    Ok(())
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
                panic!("expected ocr command");
            };
            assert_eq!(args.output.as_deref(), Some(Path::new("result.json")));
        }
        // No `-o` => None, i.e. the legacy stdout path.
        let cli = Cli::try_parse_from(["focr", "ocr", "scan.png"]).expect("ocr parses");
        let Command::Ocr(args) = cli.command else {
            panic!("expected ocr command");
        };
        assert!(args.output.is_none());
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

    #[test]
    fn pdf_json_carries_per_page_layout_with_one_based_page_numbers() {
        let rec = Recognition::Pdf(PdfRecognition {
            markdown: "p1\n\np2".to_string(),
            pages: vec![
                PdfPageLayout {
                    page: 1,
                    layout: vec![native_engine::LayoutSpan {
                        label: "text".to_string(),
                        boxes: vec![[0, 0, 5, 5]],
                    }],
                },
                PdfPageLayout {
                    page: 2,
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
        write_ocr_output(&md_path, &rec, false, &[]).expect("write md");
        assert_eq!(std::fs::read_to_string(&md_path).unwrap(), "hello world\n");

        // JSON form: valid JSON, newline-terminated, carrying the bounding boxes.
        let json_path = dir.join("out.json");
        write_ocr_output(&json_path, &rec, true, &[]).expect("write json");
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
            panic!("expected ocr command");
        };
        assert!(args.extract_figures);
        let cli = Cli::try_parse_from(["focr", "ocr", "scan.png", "--figures-dir", "assets"])
            .expect("--figures-dir parses");
        let Command::Ocr(args) = cli.command else {
            panic!("expected ocr command");
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
            panic!("expected models");
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
