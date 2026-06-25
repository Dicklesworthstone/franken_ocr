//! Adaptive / alien-artifact controllers (plan §9.7, AF-1..5).
//!
//! Each controller in this module follows the **Alien-Artifact Engineering
//! Contract** (`AGENTS.md`): every runtime/adaptive decision ships with an
//! explicit state space, explicit actions, a loss matrix, posterior/confidence
//! terms plus a calibration metric, a **deterministic fallback trigger**, and an
//! **evidence-ledger** artifact for audit. No adaptive controller ships without a
//! conservative deterministic fallback, and correctness outranks speed
//! (doctrine #1): an adaptive path that would change the OCR output is reverted.
//!
//! These are **offline / release-gate** controllers — none of them runs at
//! inference time (no Python, no network), but they live in the crate so the
//! shipping monitors are the *same* numerically-validated code the gauntlet runs.
//!
//! Members:
//! - [`tail_risk`] — AF-2: the CVaR + EVT/POT tail-risk gate (this file's owner).
//! - [`conformal`] — AF-3: conformal early-exit decode (sibling-owned).
//! - [`usl`] — AF-5: Universal-Scalability-Law pool sizing (sibling-owned).

pub mod conformal;
pub mod tail_risk;
pub mod usl;
