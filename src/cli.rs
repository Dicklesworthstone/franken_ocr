//! The `focr` clap-derive CLI surface (plan §7.2).
//!
//! Subcommands are Phase-0 skeleton stubs: the diagnostics (`robot
//! schema/health/backends`) work today; the real work (`ocr`, `convert`,
//! `doctor`) returns a clear `NotImplemented` pointing at the plan phase that
//! lands it. PDF input is intentionally absent — v1 is image-only (plan §7.7).

use clap::{Parser, Subcommand, ValueEnum};
use franken_ocr::{FocrError, FocrResult, robot};

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

#[derive(Subcommand)]
pub enum Command {
    /// Parse a document image into structured markdown (or `--json`).
    Ocr {
        /// Input image path (v1 is image-only; PDFs are rasterized out-of-band — plan §7.7).
        image: std::path::PathBuf,
        /// Emit machine-readable JSON instead of human markdown.
        #[arg(long)]
        json: bool,
        /// Stream NDJSON robot events as pages complete.
        #[arg(long)]
        robot: bool,
    },
    /// Offline weight transformation: safetensors → `.focrq` (plan §5).
    Convert {
        /// Source `model-00001-of-000001.safetensors`.
        input: std::path::PathBuf,
        /// Destination `.focrq`.
        #[arg(short, long)]
        output: std::path::PathBuf,
        /// Quantization target.
        #[arg(long, value_enum)]
        quant: Option<QuantTarget>,
    },
    /// Agent-facing diagnostics and the machine contract.
    Robot {
        #[command(subcommand)]
        cmd: RobotCmd,
    },
    /// Idempotent self-check / repair.
    Doctor,
}

#[derive(Subcommand)]
pub enum RobotCmd {
    /// Self-describing event/contract schema (versioned).
    Schema,
    /// Diagnostics: model present? arch features? threads?
    Health,
    /// Detected SIMD tiers (SMMLA/SDOT/VNNI/AMX/scalar) + core count.
    Backends,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum QuantTarget {
    Int8,
    Int4,
}

/// Dispatch a parsed CLI invocation.
pub fn run(cli: Cli) -> FocrResult<()> {
    match cli.command {
        Command::Robot { cmd: RobotCmd::Schema } => {
            emit(&robot::robot_schema());
            Ok(())
        }
        Command::Robot { cmd: RobotCmd::Health } => {
            // Phase 0: minimal health. The full report (model resolution, arch
            // features, thread budget) lands in Phase 5 (plan §7.3).
            emit(&serde_json::json!({
                "schema_version": robot::ROBOT_SCHEMA_VERSION,
                "status": "scaffold",
                "ready": false,
                "phase": "pre-Phase-0 skeleton",
                "model_present": false
            }));
            Ok(())
        }
        Command::Robot { cmd: RobotCmd::Backends } => {
            // Real runtime SIMD-tier detection lands in Phase 3 (plan §6.2).
            emit(&serde_json::json!({
                "schema_version": robot::ROBOT_SCHEMA_VERSION,
                "simd_tiers": {
                    "selected": null,
                    "available": [],
                    "status": "runtime detection lands in Phase 3 (plan §6.2)"
                },
                "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)
            }));
            Ok(())
        }
        Command::Ocr { .. } => Err(FocrError::NotImplemented(
            "focr ocr — the model forward lands in Phase 1 (see COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md §10)".into(),
        )),
        Command::Convert { .. } => Err(FocrError::NotImplemented(
            "focr convert — the weight transformer lands in Phase 2 (see plan §5)".into(),
        )),
        Command::Doctor => Err(FocrError::NotImplemented(
            "focr doctor — lands in Phase 5 (see plan §7)".into(),
        )),
    }
}

fn emit(value: &serde_json::Value) {
    // Robot-facing commands emit exactly one JSON object per line.
    println!(
        "{}",
        serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
    );
}
