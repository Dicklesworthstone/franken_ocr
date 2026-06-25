//! The `focr` clap-derive CLI surface (plan §7.2).
//!
//! Subcommands are Phase-0 skeleton stubs: the diagnostics (`robot
//! schema/health/backends`) work today; the real work (`ocr`, `convert`,
//! `doctor`) returns a clear `NotImplemented` pointing at the plan phase that
//! lands it. PDF input is intentionally absent — v1 is image-only (plan §7.7).
//!
//! This module lives in the **library** so the single CLI entrypoint
//! ([`cli_main`]) is shared by both binaries (`focr` and `franken_ocr`) without
//! either `src/main.rs` appearing in two build targets — each `[[bin]]` now
//! points at its own thin shim that just calls [`cli_main`]. See AGENTS.md
//! doctrine #9.

use crate::{FocrError, FocrResult, robot};
use clap::{Parser, Subcommand, ValueEnum};
use std::process::ExitCode;

/// The shared process entrypoint for both binaries (`focr` and `franken_ocr`).
///
/// `fn main()` in each shim is **synchronous by design** (plan §3.3, §7.1): the
/// asupersync runtime is owned BELOW here, inside `OcrEngine`, never spanning
/// the whole process. This parses, dispatches, and maps errors to the stable
/// exit codes documented in [`crate::error`].
pub fn cli_main() -> ExitCode {
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
        Command::Robot {
            cmd: RobotCmd::Schema,
        } => {
            emit(&robot::robot_schema());
            Ok(())
        }
        Command::Robot {
            cmd: RobotCmd::Health,
        } => {
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
        Command::Robot {
            cmd: RobotCmd::Backends,
        } => {
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
        Command::Ocr { robot: true, .. } => {
            emit(&robot::run_start_event("ocr"));
            Err(ocr_not_implemented())
        }
        Command::Ocr { .. } => Err(ocr_not_implemented()),
        Command::Convert { .. } => Err(FocrError::NotImplemented(
            "focr convert — the weight transformer lands in Phase 2 (see plan §5)".into(),
        )),
        Command::Doctor => Err(FocrError::NotImplemented(
            "focr doctor — lands in Phase 5 (see plan §7)".into(),
        )),
    }
}

fn ocr_not_implemented() -> FocrError {
    FocrError::NotImplemented(
        "focr ocr — the model forward lands in Phase 1 (see COMPREHENSIVE_PLAN_FOR_FRANKEN_OCR.md §10)".into(),
    )
}

fn emit(value: &serde_json::Value) {
    // Robot-facing commands emit exactly one JSON object per line.
    println!(
        "{}",
        serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
    );
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ErrorMode {
    Human,
    Robot,
}

impl ErrorMode {
    fn from_cli(cli: &Cli) -> Self {
        match &cli.command {
            Command::Ocr { robot: true, .. } => Self::Robot,
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
    fn ocr_robot_flag_selects_robot_error_mode() {
        let cli = Cli {
            command: Command::Ocr {
                image: std::path::PathBuf::from("scan.png"),
                json: false,
                robot: true,
            },
        };
        assert_eq!(ErrorMode::from_cli(&cli), ErrorMode::Robot);
    }
}
