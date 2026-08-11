//! Activation-calibration statistics: the STAT-KEY vocabulary shared by the
//! recorder ([`crate::native_engine::calib`]) and the calibration-aware
//! converter ([`super::convert`]), plus the JSON container both sides speak.
//!
//! ## What is recorded, and why a KEY instead of a tensor name
//!
//! The calibration-aware quantizers (bd-50wo stages B/C) need, per quantized
//! GEMM, the **per-input-channel mean square of the activation rows that GEMM
//! consumes**: `E[x_i²]` for `i` in `0..k`. That is a property of the GEMM's
//! INPUT, and in this architecture several weight tensors share one input:
//!
//! * `q_proj` / `k_proj` / `v_proj` of a layer all consume the SAME
//!   `input_layernorm` output;
//! * an expert's `gate_proj` and `up_proj` both consume the SAME routed rows of
//!   the `post_attention_layernorm` output;
//! * `down_proj` alone consumes `silu(gate·x) * (up·x)`.
//!
//! Recording per *tensor name* would therefore duplicate identical vectors (and
//! double the recorder's work and the JSON size). The recorder writes one entry
//! per distinct GEMM INPUT — a **stat key** — and [`stat_key_for_tensor`] is the
//! single pure function that maps a weight tensor name onto its key. Both sides
//! call it, so the two halves cannot drift.
//!
//! ## Container
//!
//! ```json
//! { "schema": "focr-calib-v1",
//!   "keys": { "mlp.3.e17.down_in": { "rows": 1204, "mean_sq": [ ... ] } } }
//! ```
//!
//! `rows` is the number of activation rows that fed the accumulator; `mean_sq[i]`
//! is `Σ_rows x_i² / rows`. Merging two runs is the row-weighted mean (see
//! [`CalibStats::merge`]) — exactly what accumulating both runs in one process
//! would have produced, up to f64 summation order.
//!
//! Tensors with no entry (a starved MoE expert that no calibration token ever
//! routed to, or `embed_tokens`, whose "input" is a one-hot lookup rather than
//! an activation) fall back to UNIFORM importance, which is the plain
//! unweighted quantization objective.

use std::collections::BTreeMap;

use crate::error::{FocrError, FocrResult};

/// Schema tag written into (and required from) a calibration JSON.
pub const CALIB_SCHEMA: &str = "focr-calib-v1";

/// One GEMM input's accumulated per-channel statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelStats {
    /// Activation rows accumulated into `mean_sq`.
    pub rows: u64,
    /// `mean_sq[i] = Σ_rows x_i² / rows`, length = the GEMM's contraction `k`.
    pub mean_sq: Vec<f64>,
}

/// A whole calibration run: stat key → [`ChannelStats`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CalibStats {
    /// Ordered so serialization is deterministic.
    keys: BTreeMap<String, ChannelStats>,
}

impl CalibStats {
    /// An empty set of statistics (every lookup misses ⇒ uniform importance).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct GEMM inputs with recorded statistics.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether nothing was recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Iterate `(key, stats)` in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ChannelStats)> {
        self.keys.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Insert/replace one key's statistics.
    pub fn insert(&mut self, key: impl Into<String>, stats: ChannelStats) {
        self.keys.insert(key.into(), stats);
    }

    /// Raw lookup by stat key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&ChannelStats> {
        self.keys.get(key)
    }

    /// The per-input-channel importance vector for a WEIGHT TENSOR name, or
    /// `None` when this run has no statistics for that GEMM's input (⇒ the
    /// caller must fall back to uniform importance).
    ///
    /// `expect_k` is the tensor's contraction length; a recorded vector of a
    /// different length is a mismatch between the calibrated model and the
    /// checkpoint being converted, and is reported as `None` by
    /// [`Self::importance_for`] only through [`Self::importance_for_checked`]'s
    /// error — this accessor simply requires the length to match.
    #[must_use]
    pub fn importance_for(&self, tensor_name: &str, expect_k: usize) -> Option<&[f64]> {
        let key = stat_key_for_tensor(tensor_name)?;
        let stats = self.keys.get(&key)?;
        if stats.rows == 0 || stats.mean_sq.len() != expect_k {
            return None;
        }
        Some(&stats.mean_sq)
    }

    /// Like [`Self::importance_for`] but LOUD about a length disagreement: a
    /// recorded vector whose length is not `expect_k` means the calibration was
    /// captured from a different model than the one being converted, which
    /// would silently mis-weight every group.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] when a key exists with the wrong length.
    pub fn importance_for_checked(
        &self,
        tensor_name: &str,
        expect_k: usize,
    ) -> FocrResult<Option<&[f64]>> {
        let Some(key) = stat_key_for_tensor(tensor_name) else {
            return Ok(None);
        };
        let Some(stats) = self.keys.get(&key) else {
            return Ok(None);
        };
        if stats.rows == 0 {
            return Ok(None);
        }
        if stats.mean_sq.len() != expect_k {
            return Err(FocrError::FormatMismatch(format!(
                "calibration key {key:?} (for tensor {tensor_name:?}) has {} channels, but the \
                 checkpoint tensor contracts over {expect_k} — the calibration JSON was captured \
                 from a different model",
                stats.mean_sq.len()
            )));
        }
        Ok(Some(&stats.mean_sq))
    }

    /// Row-weighted merge of another run into this one: for a shared key the
    /// result is `(rows_a·mean_a + rows_b·mean_b) / (rows_a + rows_b)` — the
    /// same value accumulating both runs in one process would have produced.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if a shared key disagrees on channel count.
    pub fn merge(&mut self, other: &Self) -> FocrResult<()> {
        for (key, rhs) in &other.keys {
            match self.keys.get_mut(key) {
                None => {
                    self.keys.insert(key.clone(), rhs.clone());
                }
                Some(lhs) => {
                    if lhs.mean_sq.len() != rhs.mean_sq.len() {
                        return Err(FocrError::FormatMismatch(format!(
                            "calibration merge: key {key:?} has {} channels in one run and {} in \
                             another",
                            lhs.mean_sq.len(),
                            rhs.mean_sq.len()
                        )));
                    }
                    let total = lhs.rows.saturating_add(rhs.rows);
                    if total == 0 {
                        continue;
                    }
                    let (wa, wb) = (lhs.rows as f64, rhs.rows as f64);
                    for (a, &b) in lhs.mean_sq.iter_mut().zip(rhs.mean_sq.iter()) {
                        *a = (*a * wa + b * wb) / (wa + wb);
                    }
                    lhs.rows = total;
                }
            }
        }
        Ok(())
    }

    /// Serialize to the `focr-calib-v1` JSON container.
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(64 + self.keys.len() * 64);
        out.push_str("{\"schema\":\"");
        out.push_str(CALIB_SCHEMA);
        out.push_str("\",\"keys\":{");
        for (i, (key, stats)) in self.keys.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            // Stat keys are generated from `[A-Za-z0-9_.]` only (see
            // `stat_key_for_tensor`), so no JSON string escaping is needed —
            // asserted by `stat_keys_are_json_safe`.
            out.push('"');
            out.push_str(key);
            out.push_str("\":{\"rows\":");
            out.push_str(&stats.rows.to_string());
            out.push_str(",\"mean_sq\":[");
            for (j, v) in stats.mean_sq.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                // Finite by construction (sums of squares of finite
                // activations); a non-finite slot is written as 0 so the JSON
                // stays parseable rather than emitting bare `NaN`.
                if v.is_finite() {
                    out.push_str(&format!("{v:.6e}"));
                } else {
                    out.push('0');
                }
            }
            out.push_str("]}");
        }
        out.push_str("}}");
        out
    }

    /// Parse a `focr-calib-v1` JSON container.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] on invalid JSON, a wrong/absent schema tag,
    /// or a malformed entry.
    pub fn from_json(text: &str) -> FocrResult<Self> {
        let value: serde_json::Value = serde_json::from_str(text).map_err(|e| {
            FocrError::FormatMismatch(format!("calibration JSON is not valid JSON: {e}"))
        })?;
        let schema = value.get("schema").and_then(serde_json::Value::as_str);
        if schema != Some(CALIB_SCHEMA) {
            return Err(FocrError::FormatMismatch(format!(
                "calibration JSON schema {schema:?} != {CALIB_SCHEMA:?}"
            )));
        }
        let keys = value
            .get("keys")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                FocrError::FormatMismatch(
                    "calibration JSON has no object-valued \"keys\" member".to_string(),
                )
            })?;
        let mut out = Self::new();
        for (key, entry) in keys {
            let rows = entry
                .get("rows")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| {
                    FocrError::FormatMismatch(format!(
                        "calibration key {key:?} has no unsigned \"rows\" member"
                    ))
                })?;
            let arr = entry
                .get("mean_sq")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    FocrError::FormatMismatch(format!(
                        "calibration key {key:?} has no array \"mean_sq\" member"
                    ))
                })?;
            let mut mean_sq = Vec::with_capacity(arr.len());
            for v in arr {
                let f = v.as_f64().ok_or_else(|| {
                    FocrError::FormatMismatch(format!(
                        "calibration key {key:?} has a non-numeric mean_sq entry"
                    ))
                })?;
                if !f.is_finite() || f < 0.0 {
                    return Err(FocrError::FormatMismatch(format!(
                        "calibration key {key:?} has a negative/non-finite mean_sq entry {f}"
                    )));
                }
                mean_sq.push(f);
            }
            out.keys.insert(key.clone(), ChannelStats { rows, mean_sq });
        }
        Ok(out)
    }
}

// ── Stat-key vocabulary ─────────────────────────────────────────────────────

/// Stat key for the `input_layernorm` output of layer `l` — the input `q_proj`,
/// `k_proj` and `v_proj` all contract over.
#[must_use]
pub fn attn_in_key(layer: usize) -> String {
    format!("attn.{layer}.in")
}

/// Stat key for the attention context of layer `l` — `o_proj`'s input.
#[must_use]
pub fn o_in_key(layer: usize) -> String {
    format!("attn.{layer}.o_in")
}

/// Which MLP unit inside a decoder layer a stat key belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpUnit {
    /// The layer-0 dense SwiGLU (`mlp.{gate,up,down}_proj`).
    Dense,
    /// Routed expert `e` (`mlp.experts.{e}.…`).
    Expert(usize),
    /// The fused shared experts (`mlp.shared_experts.…`).
    Shared,
}

impl MlpUnit {
    /// The key fragment identifying this unit.
    fn tag(self) -> String {
        match self {
            MlpUnit::Dense => "dense".to_string(),
            MlpUnit::Expert(e) => format!("e{e}"),
            MlpUnit::Shared => "shared".to_string(),
        }
    }
}

/// Stat key for the SwiGLU input of `unit` in layer `l` — the input BOTH
/// `gate_proj` and `up_proj` contract over.
#[must_use]
pub fn ffn_in_key(layer: usize, unit: MlpUnit) -> String {
    format!("mlp.{layer}.{}.ffn_in", unit.tag())
}

/// Stat key for `silu(gate·x) * (up·x)` of `unit` in layer `l` — `down_proj`'s
/// input.
#[must_use]
pub fn down_in_key(layer: usize, unit: MlpUnit) -> String {
    format!("mlp.{layer}.{}.down_in", unit.tag())
}

/// Stat key for the final-norm output — `lm_head`'s input.
pub const LM_HEAD_IN_KEY: &str = "lm_head.in";

/// Map a WEIGHT TENSOR name to the stat key of the GEMM input it contracts
/// over, or `None` for a tensor with no activation input in this recipe
/// (`embed_tokens`, norms, the router gate, the whole vision tower).
///
/// **Pure** and total — the single source of truth both the recorder and the
/// converter use, so a rename can never silently de-correlate them.
#[must_use]
pub fn stat_key_for_tensor(name: &str) -> Option<String> {
    if name == "lm_head.weight" {
        return Some(LM_HEAD_IN_KEY.to_string());
    }
    let rest = name.strip_prefix("model.layers.")?;
    let (layer_str, rest) = rest.split_once('.')?;
    let layer: usize = layer_str.parse().ok()?;

    if let Some(proj) = rest.strip_prefix("self_attn.") {
        return match proj {
            "q_proj.weight" | "k_proj.weight" | "v_proj.weight" => Some(attn_in_key(layer)),
            "o_proj.weight" => Some(o_in_key(layer)),
            _ => None,
        };
    }

    let rest = rest.strip_prefix("mlp.")?;
    // Resolve the unit, leaving `leaf` = "{gate,up,down}_proj.weight".
    let (unit, leaf) = if let Some(tail) = rest.strip_prefix("experts.") {
        let (idx, leaf) = tail.split_once('.')?;
        (MlpUnit::Expert(idx.parse().ok()?), leaf)
    } else if let Some(leaf) = rest.strip_prefix("shared_experts.") {
        (MlpUnit::Shared, leaf)
    } else {
        (MlpUnit::Dense, rest)
    };
    match leaf {
        "gate_proj.weight" | "up_proj.weight" => Some(ffn_in_key(layer, unit)),
        "down_proj.weight" => Some(down_in_key(layer, unit)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_tensors_share_one_input_key() {
        for proj in ["q_proj", "k_proj", "v_proj"] {
            assert_eq!(
                stat_key_for_tensor(&format!("model.layers.7.self_attn.{proj}.weight")).as_deref(),
                Some("attn.7.in"),
                "{proj} shares the input_layernorm output"
            );
        }
        assert_eq!(
            stat_key_for_tensor("model.layers.7.self_attn.o_proj.weight").as_deref(),
            Some("attn.7.o_in"),
            "o_proj consumes the attention context, NOT the norm output"
        );
    }

    #[test]
    fn gate_and_up_share_a_key_but_down_does_not() {
        let gate = stat_key_for_tensor("model.layers.3.mlp.experts.17.gate_proj.weight");
        let up = stat_key_for_tensor("model.layers.3.mlp.experts.17.up_proj.weight");
        let down = stat_key_for_tensor("model.layers.3.mlp.experts.17.down_proj.weight");
        assert_eq!(gate, up);
        assert_eq!(gate.as_deref(), Some("mlp.3.e17.ffn_in"));
        assert_eq!(down.as_deref(), Some("mlp.3.e17.down_in"));
        assert_ne!(gate, down);
    }

    #[test]
    fn dense_shared_and_lm_head_keys() {
        assert_eq!(
            stat_key_for_tensor("model.layers.0.mlp.gate_proj.weight").as_deref(),
            Some("mlp.0.dense.ffn_in")
        );
        assert_eq!(
            stat_key_for_tensor("model.layers.0.mlp.down_proj.weight").as_deref(),
            Some("mlp.0.dense.down_in")
        );
        assert_eq!(
            stat_key_for_tensor("model.layers.5.mlp.shared_experts.up_proj.weight").as_deref(),
            Some("mlp.5.shared.ffn_in")
        );
        assert_eq!(
            stat_key_for_tensor("lm_head.weight").as_deref(),
            Some(LM_HEAD_IN_KEY)
        );
    }

    #[test]
    fn unkeyed_tensors_have_no_activation_input() {
        for name in [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.layers.3.mlp.gate.weight", // the MoE router (gate, not gate_proj)
            "model.layers.3.input_layernorm.weight",
            "model.vision_model.encoder.layers.2.self_attn.q_proj.weight",
            "model.layers.notanumber.mlp.up_proj.weight",
        ] {
            assert_eq!(stat_key_for_tensor(name), None, "{name} must have no key");
        }
    }

    #[test]
    fn key_builders_agree_with_the_tensor_name_mapping() {
        // The recorder builds keys from (layer, unit); the converter builds them
        // from tensor names. They must be the same strings.
        assert_eq!(
            ffn_in_key(3, MlpUnit::Expert(17)),
            stat_key_for_tensor("model.layers.3.mlp.experts.17.gate_proj.weight").unwrap()
        );
        assert_eq!(
            down_in_key(11, MlpUnit::Shared),
            stat_key_for_tensor("model.layers.11.mlp.shared_experts.down_proj.weight").unwrap()
        );
        assert_eq!(
            attn_in_key(0),
            stat_key_for_tensor("model.layers.0.self_attn.k_proj.weight").unwrap()
        );
        assert_eq!(
            o_in_key(2),
            stat_key_for_tensor("model.layers.2.self_attn.o_proj.weight").unwrap()
        );
    }

    #[test]
    fn stat_keys_are_json_safe() {
        // `to_json` emits keys unescaped; prove the generated vocabulary needs
        // no escaping.
        let mut keys = vec![attn_in_key(3), o_in_key(3), LM_HEAD_IN_KEY.to_string()];
        for unit in [MlpUnit::Dense, MlpUnit::Expert(63), MlpUnit::Shared] {
            keys.push(ffn_in_key(11, unit));
            keys.push(down_in_key(11, unit));
        }
        for key in keys {
            assert!(
                key.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_'),
                "{key} must be JSON-safe without escaping"
            );
        }
    }

    #[test]
    fn json_roundtrip_is_exact_enough_and_schema_checked() {
        let mut stats = CalibStats::new();
        stats.insert(
            ffn_in_key(1, MlpUnit::Expert(2)),
            ChannelStats {
                rows: 40,
                mean_sq: vec![1.5, 0.25, 1.0e-7],
            },
        );
        let text = stats.to_json();
        let back = CalibStats::from_json(&text).expect("roundtrip parses");
        let got = back.get("mlp.1.e2.ffn_in").expect("key survives");
        assert_eq!(got.rows, 40);
        for (a, b) in got.mean_sq.iter().zip([1.5, 0.25, 1.0e-7]) {
            assert!((a - b).abs() <= b.abs() * 1e-6 + 1e-30, "{a} vs {b}");
        }
        assert!(CalibStats::from_json("{\"schema\":\"nope\",\"keys\":{}}").is_err());
        assert!(CalibStats::from_json("not json").is_err());
    }

    #[test]
    fn merge_is_the_row_weighted_mean() {
        let key = down_in_key(0, MlpUnit::Dense);
        let mut a = CalibStats::new();
        a.insert(
            key.clone(),
            ChannelStats {
                rows: 3,
                mean_sq: vec![2.0, 0.0],
            },
        );
        let mut b = CalibStats::new();
        b.insert(
            key.clone(),
            ChannelStats {
                rows: 1,
                mean_sq: vec![6.0, 4.0],
            },
        );
        b.insert(
            "mlp.0.dense.ffn_in".to_string(),
            ChannelStats {
                rows: 5,
                mean_sq: vec![1.0],
            },
        );
        a.merge(&b).expect("compatible merge");
        let merged = a.get(&key).expect("merged key");
        assert_eq!(merged.rows, 4);
        // (3*2 + 1*6)/4 = 3 ; (3*0 + 1*4)/4 = 1
        assert!((merged.mean_sq[0] - 3.0).abs() < 1e-12);
        assert!((merged.mean_sq[1] - 1.0).abs() < 1e-12);
        // A key present only in `b` is carried over verbatim.
        assert_eq!(a.get("mlp.0.dense.ffn_in").map(|s| s.rows), Some(5));
    }

    #[test]
    fn merge_rejects_a_channel_count_disagreement() {
        let key = attn_in_key(0);
        let mut a = CalibStats::new();
        a.insert(
            key.clone(),
            ChannelStats {
                rows: 1,
                mean_sq: vec![1.0, 1.0],
            },
        );
        let mut b = CalibStats::new();
        b.insert(
            key,
            ChannelStats {
                rows: 1,
                mean_sq: vec![1.0],
            },
        );
        assert!(a.merge(&b).is_err());
    }

    #[test]
    fn importance_lookup_misses_are_uniform_fallbacks_and_length_is_enforced() {
        let mut stats = CalibStats::new();
        stats.insert(
            attn_in_key(0),
            ChannelStats {
                rows: 2,
                mean_sq: vec![1.0, 2.0, 3.0],
            },
        );
        stats.insert(
            o_in_key(0),
            ChannelStats {
                rows: 0,
                mean_sq: vec![1.0],
            },
        );
        let name = "model.layers.0.self_attn.q_proj.weight";
        assert_eq!(stats.importance_for(name, 3).map(<[f64]>::len), Some(3));
        // Wrong length: silent None on the lenient accessor, loud on the checked one.
        assert_eq!(stats.importance_for(name, 4), None);
        assert!(stats.importance_for_checked(name, 4).is_err());
        // rows == 0 is "no observation", not "all-zero activations".
        assert_eq!(
            stats.importance_for("model.layers.0.self_attn.o_proj.weight", 1),
            None
        );
        // A tensor with no key at all (embed_tokens) is never an error.
        assert_eq!(
            stats
                .importance_for_checked("model.embed_tokens.weight", 1280)
                .expect("no key is not an error"),
            None
        );
    }
}
