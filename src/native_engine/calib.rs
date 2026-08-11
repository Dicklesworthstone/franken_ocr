//! `FOCR_CALIB_OUT`: the env-gated activation-statistics recorder that feeds the
//! calibration-aware quantizers (bd-50wo stage A).
//!
//! ## Contract
//!
//! * **Off by default, and free when off.** The env var is read ONCE into a
//!   `OnceLock`; every `record_*` entry point begins with a load of that
//!   `Option`, and does nothing else when it is `None` — no allocation, no lock,
//!   no formatting. Call sites additionally guard with [`enabled`] so even the
//!   stat-key `format!` is skipped.
//! * **When on**, each call accumulates, per GEMM input (keyed by the stat-key
//!   vocabulary of [`crate::quant::calib`]), the per-input-channel SUM OF
//!   SQUARES of the activation rows in f64, plus the row count. [`flush`]
//!   converts those to per-channel means and writes the `focr-calib-v1` JSON to
//!   `$FOCR_CALIB_OUT`.
//! * **What is covered**: the decoder GEMMs the wasm int4 recipe quantizes —
//!   attention `q/k/v` (shared `input_layernorm` output), `o_proj` (attention
//!   context), every dense/routed/shared expert SwiGLU (`gate`+`up` input and
//!   `down` input), and `lm_head` (final-norm output). The vision tower is NOT
//!   covered: it stays BF16 in this recipe, so it has no scale to calibrate.
//!
//! The recorder lives on the NATIVE side only: calibration runs the bf16
//! checkpoint through the native mixed-precision cache. Under
//! `--no-default-features` (the wasm/browser build) every entry point compiles
//! to an empty body.

#[cfg(feature = "native")]
use std::collections::BTreeMap;
#[cfg(feature = "native")]
use std::sync::{Mutex, OnceLock};

use crate::error::FocrResult;

#[cfg(feature = "native")]
struct Acc {
    rows: u64,
    sum_sq: Vec<f64>,
}

#[cfg(feature = "native")]
struct Recorder {
    path: std::path::PathBuf,
    table: Mutex<BTreeMap<String, Acc>>,
}

#[cfg(feature = "native")]
fn recorder() -> Option<&'static Recorder> {
    static STATE: OnceLock<Option<Recorder>> = OnceLock::new();
    STATE
        .get_or_init(|| {
            std::env::var_os("FOCR_CALIB_OUT").map(|path| Recorder {
                path: std::path::PathBuf::from(path),
                table: Mutex::new(BTreeMap::new()),
            })
        })
        .as_ref()
}

/// Whether activation calibration is armed (`FOCR_CALIB_OUT` set). Read once;
/// call sites use it to skip building the stat key at all.
#[must_use]
#[inline]
pub fn enabled() -> bool {
    #[cfg(feature = "native")]
    {
        recorder().is_some()
    }
    #[cfg(not(feature = "native"))]
    {
        false
    }
}

/// Accumulate a `[rows, cols]` row-major activation block under `key`.
///
/// A no-op when disabled. `data.len()` must be a multiple of `cols`; a ragged
/// block is ignored rather than panicking (the recorder is diagnostic and must
/// never be able to fail a real run).
#[inline]
pub fn record_rows(key: &str, data: &[f32], cols: usize) {
    #[cfg(not(feature = "native"))]
    {
        let _ = (key, data, cols);
    }
    #[cfg(feature = "native")]
    {
        let Some(rec) = recorder() else { return };
        if cols == 0 || !data.len().is_multiple_of(cols) {
            return;
        }
        let n_rows = data.len() / cols;
        if n_rows == 0 {
            return;
        }
        // Sum outside the lock so contended rayon callers do not serialize on
        // the whole block.
        let mut local = vec![0.0f64; cols];
        for row in data.chunks_exact(cols) {
            for (slot, &v) in local.iter_mut().zip(row.iter()) {
                let x = f64::from(v);
                *slot += x * x;
            }
        }
        let Ok(mut table) = rec.table.lock() else {
            return;
        };
        match table.get_mut(key) {
            Some(acc) if acc.sum_sq.len() == cols => {
                acc.rows = acc.rows.saturating_add(n_rows as u64);
                for (slot, &v) in acc.sum_sq.iter_mut().zip(local.iter()) {
                    *slot += v;
                }
            }
            Some(_) => {
                // A key cannot legitimately change width within a run; drop the
                // sample rather than corrupt the accumulator.
            }
            None => {
                table.insert(
                    key.to_string(),
                    Acc {
                        rows: n_rows as u64,
                        sum_sq: local,
                    },
                );
            }
        }
    }
}

/// Accumulate a single activation row under `key` (the `m = 1` decode case).
#[inline]
pub fn record_row(key: &str, row: &[f32]) {
    record_rows(key, row, row.len());
}

/// Write the accumulated statistics to `$FOCR_CALIB_OUT` as `focr-calib-v1`
/// JSON. A no-op when disabled. Safe to call repeatedly — each call rewrites the
/// file with everything accumulated so far, so a run that is killed mid-corpus
/// still leaves the statistics of the pages it finished.
///
/// # Errors
/// Propagates the write failure so a calibration run cannot silently produce
/// nothing.
pub fn flush() -> FocrResult<()> {
    #[cfg(feature = "native")]
    {
        let Some(rec) = recorder() else {
            return Ok(());
        };
        let Ok(table) = rec.table.lock() else {
            return Ok(());
        };
        let mut stats = crate::quant::calib::CalibStats::new();
        for (key, acc) in table.iter() {
            if acc.rows == 0 {
                continue;
            }
            let denom = acc.rows as f64;
            stats.insert(
                key.clone(),
                crate::quant::calib::ChannelStats {
                    rows: acc.rows,
                    mean_sq: acc.sum_sq.iter().map(|&s| s / denom).collect(),
                },
            );
        }
        std::fs::write(&rec.path, stats.to_json()).map_err(|e| {
            crate::error::FocrError::Other(anyhow::anyhow!(
                "writing FOCR_CALIB_OUT to {}: {e}",
                rec.path.display()
            ))
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_flush_are_noops_when_disabled() {
        // The test process does not set FOCR_CALIB_OUT, so this exercises the
        // zero-cost path end to end: no panic, no file, no error.
        assert!(!enabled());
        record_rows("attn.0.in", &[1.0, 2.0, 3.0, 4.0], 2);
        record_row("lm_head.in", &[1.0, 2.0]);
        flush().expect("flush is a no-op when disabled");
    }

    #[test]
    fn ragged_and_empty_blocks_are_ignored_not_panicked() {
        record_rows("attn.0.in", &[1.0, 2.0, 3.0], 2);
        record_rows("attn.0.in", &[], 2);
        record_rows("attn.0.in", &[1.0], 0);
    }
}
