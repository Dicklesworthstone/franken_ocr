//! The quant **recipe** — doctrine #2 made executable (AGENTS.md doctrine #2,
//! PROPOSED_ARCHITECTURE.md §5.2/§7 "the quant recipe is fixed").
//!
//! This module is the single, deterministic authority that classifies **every**
//! model tensor (by its HF dotted name) into a [`QuantPolicy`]. It is the gate
//! the converter (`focr convert`) and the AF-1 bit allocator consult before
//! touching a single weight: a tensor the recipe marks [`QuantPolicy::KeepBf16`]
//! is **never** quantized, and the recipe **refuses** to quantize the entire
//! vision tower, the projector, `embed_tokens`, the MoE router gate, and all
//! norms — quantizing any of those wrecks OCR (both prior-art quants keep them;
//! the router has a measured gate-drift cliff).
//!
//! Three policies (the validated split, doctrine #2):
//!
//! * **[`QuantPolicy::KeepBf16`]** — the high-precision set, stored BF16 verbatim
//!   and widened BF16→f32 at load: the whole SAM + CLIP vision tower, the
//!   projector, `model.embed_tokens`, the MoE router `mlp.gate.weight`, and ALL
//!   norms (`*_layernorm`, `*.norm`, `model.norm`, vision `LayerNorm`s). Also the
//!   two learned connector params (`image_newline` / `view_seperator`).
//! * **[`QuantPolicy::Int8`]** — the **validated** quantizable set: the decoder
//!   FFN / expert GEMMs only. The dense layer-0 SwiGLU (`gate`/`up`/`down_proj`),
//!   the 64 routed experts × 3 proj per MoE layer, and the 2 fused shared experts
//!   × 3 proj per MoE layer. These are the NVFP4/GGUF-validated tensors.
//! * **[`QuantPolicy::Gated`]** — tensors that quantize to int8 only **behind a
//!   measured-CER kill-switch env var, default OFF** (OQ-14): attention
//!   `q/k/v/o_proj` (gated by [`FOCR_INT8_ATTN_ENV`]) and `lm_head.weight`
//!   (gated by [`FOCR_INT8_LMHEAD_ENV`]). When the switch is unset (or not a
//!   truthy value) a `Gated` tensor is treated **exactly like [`QuantPolicy::KeepBf16`]**
//!   — see [`ResolvedPolicy`] / [`resolve`].
//!
//! The classifier is a **pure function of the tensor name** (no I/O, no env read)
//! so it is unit-testable and deterministic. Env reads happen only in [`resolve`]
//! / [`Recipe::resolve`], which layer the kill-switches on top of the static
//! classification — the architecture's "additive layer behind a kill-switch"
//! discipline (plan P1).

use std::collections::BTreeMap;

/// Kill-switch env var that opts attention `q/k/v/o_proj` into int8 (OQ-14,
/// PROPOSED_ARCHITECTURE.md §5.2/§9). Default OFF. A truthy value (`1`/`true`/
/// `on`/`yes`, case-insensitive) turns it on; anything else (including unset)
/// keeps attention BF16.
pub const FOCR_INT8_ATTN_ENV: &str = "FOCR_INT8_ATTN";

/// Kill-switch env var that opts `lm_head.weight` into int8 (OQ-14). Default OFF;
/// same truthiness rule as [`FOCR_INT8_ATTN_ENV`].
pub const FOCR_INT8_LMHEAD_ENV: &str = "FOCR_INT8_LMHEAD";

/// The static (env-independent) quant policy for one tensor.
///
/// This is what [`classify`] returns — purely a function of the tensor name. The
/// runtime kill-switches are layered on by [`resolve`] into a [`ResolvedPolicy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantPolicy {
    /// Never quantize: store BF16 verbatim, widen BF16→f32 at load. The
    /// high-precision set (doctrine #2): vision tower, projector, embed_tokens,
    /// MoE router gate, all norms, connector params.
    KeepBf16,
    /// The validated int8 set: decoder FFN / expert GEMMs. Always quantizable.
    Int8,
    /// Quantizable to int8 **only** behind the named env kill-switch (default
    /// off). Until the switch is on this tensor is BF16. Carries which switch
    /// guards it so the converter/allocator can report it.
    Gated(GatedKind),
}

/// Which kill-switch guards a [`QuantPolicy::Gated`] tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatedKind {
    /// Attention `q/k/v/o_proj` — guarded by [`FOCR_INT8_ATTN_ENV`].
    Attention,
    /// `lm_head.weight` — guarded by [`FOCR_INT8_LMHEAD_ENV`].
    LmHead,
}

impl GatedKind {
    /// The env var name that opts this gated set into int8.
    #[must_use]
    pub fn env_var(self) -> &'static str {
        match self {
            GatedKind::Attention => FOCR_INT8_ATTN_ENV,
            GatedKind::LmHead => FOCR_INT8_LMHEAD_ENV,
        }
    }
}

/// The *effective* policy after the runtime kill-switches are applied: a tensor
/// is either kept high-precision or quantized to int8. (`Gated` collapses to one
/// of these once the env is read.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedPolicy {
    /// Keep BF16 (the high-precision set, or a gated tensor whose switch is off).
    KeepBf16,
    /// Quantize to int8 (the validated set, or a gated tensor whose switch is on).
    Int8,
}

impl ResolvedPolicy {
    /// Whether this resolved policy quantizes the tensor.
    #[must_use]
    pub fn is_quantized(self) -> bool {
        matches!(self, ResolvedPolicy::Int8)
    }
}

/// A classification result: the policy plus a stable, human-readable reason
/// (for the converter log, the evidence ledger, and test assertions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Classification {
    /// The static policy for this tensor.
    pub policy: QuantPolicy,
    /// A stable reason string explaining the decision (doctrine #2 trace).
    pub reason: &'static str,
}

// ── Name-pattern predicates (pure, no I/O) ──────────────────────────────────
//
// These mirror the HF dotted names line-backed in docs/truth-pack/CENSUS.md:
//   - vision:   model.vision_model.*  (CLIP-L)      and  model.sam_model.*  (SAM-ViT-B)
//   - projector:model.projector.layers.{weight,bias}
//   - embed:    model.embed_tokens.weight
//   - router:   model.layers.{1..11}.mlp.gate.weight   (the MoE gate — NOT a proj)
//   - norms:    *.input_layernorm.weight, *.post_attention_layernorm.weight,
//               model.norm.weight, and every vision *norm*/*LayerNorm* tensor
//   - dense:    model.layers.0.mlp.{gate,up,down}_proj.weight
//   - routed:   model.layers.{1..11}.mlp.experts.{0..63}.{gate,up,down}_proj.weight
//   - shared:   model.layers.{1..11}.mlp.shared_experts.{gate,up,down}_proj.weight
//   - attn:     model.layers.{0..11}.self_attn.{q,k,v,o}_proj.weight
//   - lm_head:  lm_head.weight

/// Whether `name` is a vision-tower tensor (CLIP-L or SAM). The whole tower is
/// KEEP_BF16: quantizing the vision encoder wrecks OCR (doctrine #2).
fn is_vision(name: &str) -> bool {
    name.starts_with("model.vision_model.")
        || name.starts_with("model.sam_model.")
        // Defensive: some checkpoints drop the `model.` prefix.
        || name.starts_with("vision_model.")
        || name.starts_with("sam_model.")
}

/// Whether `name` is the single linear projector (2048→1280) — KEEP_BF16.
fn is_projector(name: &str) -> bool {
    name.starts_with("model.projector.") || name.starts_with("projector.")
}

/// Whether `name` is the token embedding table — KEEP_BF16.
fn is_embed_tokens(name: &str) -> bool {
    name == "model.embed_tokens.weight" || name == "embed_tokens.weight"
}

/// Whether `name` is a learned connector parameter (`image_newline` /
/// `view_seperator`) — KEEP_BF16 (1-D learned vectors, not a GEMM weight).
fn is_connector_param(name: &str) -> bool {
    name == "model.image_newline"
        || name == "model.view_seperator"
        || name == "image_newline"
        || name == "view_seperator"
}

/// Whether `name` is the MoE router **gate** (`...mlp.gate.weight`, NOT a
/// `*_proj`) — KEEP_BF16. The router has a measured gate-drift cliff
/// ([SPEC-074..077]); it is NEVER quantized. Note `gate_proj` is an expert FFN
/// projection and is explicitly excluded here.
fn is_router_gate(name: &str) -> bool {
    // The MoE router is `model.layers.N.mlp.gate.weight`. Crucially distinct
    // from `mlp.gate_proj.weight` (an expert projection) and from
    // `mlp.experts.M.gate_proj.weight`.
    name.ends_with(".mlp.gate.weight") || name.ends_with(".mlp.gate.bias")
}

/// Whether `name` is any norm tensor — KEEP_BF16. Covers decoder
/// `input_layernorm`/`post_attention_layernorm`, the final `model.norm`, and
/// every vision LayerNorm/RMSNorm (caught broadly by the `norm`/`layernorm`
/// substrings, which only ever name norm tensors in this model).
fn is_norm(name: &str) -> bool {
    let lower_has = |needle: &str| name.contains(needle);
    lower_has("layernorm")
        || lower_has("layer_norm")
        || lower_has("LayerNorm")
        || name == "model.norm.weight"
        || name.ends_with(".norm.weight")
        || name.ends_with(".norm.bias")
        // SAM uses `norm1`/`norm2`/`neck.*.norm`; CLIP uses `ln_1`/`ln_2`/`ln_post`/`ln_pre`.
        || name.contains(".norm1.")
        || name.contains(".norm2.")
        || name.contains(".ln_")
        || name.ends_with(".ln_post.weight")
        || name.ends_with(".ln_post.bias")
}

/// Whether `name` is an attention projection (`self_attn.{q,k,v,o}_proj.weight`).
/// This is the GATED set (behind [`FOCR_INT8_ATTN_ENV`]). Vision attention
/// (`qkv_proj`/`out_proj` under `vision_model`/`sam_model`) is excluded — the
/// vision check runs first and wins.
fn is_decoder_attn_proj(name: &str) -> bool {
    name.contains(".self_attn.")
        && (name.ends_with(".q_proj.weight")
            || name.ends_with(".k_proj.weight")
            || name.ends_with(".v_proj.weight")
            || name.ends_with(".o_proj.weight"))
}

/// Whether `name` is `lm_head.weight` — the GATED set (behind
/// [`FOCR_INT8_LMHEAD_ENV`]).
fn is_lm_head(name: &str) -> bool {
    name == "lm_head.weight"
}

/// Whether `name` is a decoder FFN / expert projection — the validated INT8 set.
///
/// Matches the three SwiGLU projections (`gate_proj`/`up_proj`/`down_proj`) under
/// a decoder `mlp` (dense layer-0), a routed `mlp.experts.N`, or a fused
/// `mlp.shared_experts`. The leading `model.layers.` guard keeps this off any
/// vision MLP (which never uses `_proj` naming and is caught by `is_vision`
/// first anyway).
fn is_decoder_ffn_proj(name: &str) -> bool {
    let is_proj = name.ends_with(".gate_proj.weight")
        || name.ends_with(".up_proj.weight")
        || name.ends_with(".down_proj.weight");
    if !is_proj {
        return false;
    }
    // Must live under a language-decoder MLP.
    (name.starts_with("model.layers.") || name.starts_with("layers.")) && name.contains(".mlp.")
}

/// Classify one tensor by its HF dotted `name` into a static [`QuantPolicy`]
/// plus a stable reason. **Pure** — no env read, no I/O.
///
/// Evaluation order is load-bearing (doctrine #2): the KEEP_BF16 predicates run
/// FIRST so the vision tower / projector / router / norms can never be captured
/// by a later quantizable pattern (e.g. a vision attention `qkv_proj` must not
/// match the decoder-attn rule). The recipe **refuses** to quantize the
/// high-precision set by construction.
#[must_use]
pub fn classify(name: &str) -> Classification {
    // 1) KEEP_BF16 high-precision set — these win over everything.
    if is_vision(name) {
        return Classification {
            policy: QuantPolicy::KeepBf16,
            reason: "keep-bf16: vision tower (SAM/CLIP) — quantizing it wrecks OCR (doctrine #2)",
        };
    }
    if is_projector(name) {
        return Classification {
            policy: QuantPolicy::KeepBf16,
            reason: "keep-bf16: projector (2048->1280) — high-precision set (doctrine #2)",
        };
    }
    if is_embed_tokens(name) {
        return Classification {
            policy: QuantPolicy::KeepBf16,
            reason: "keep-bf16: embed_tokens — high-precision set (doctrine #2)",
        };
    }
    if is_connector_param(name) {
        return Classification {
            policy: QuantPolicy::KeepBf16,
            reason: "keep-bf16: connector param (image_newline/view_seperator)",
        };
    }
    if is_router_gate(name) {
        return Classification {
            policy: QuantPolicy::KeepBf16,
            reason: "keep-bf16: MoE router gate — gate-drift cliff, NEVER quantized (doctrine #2)",
        };
    }
    if is_norm(name) {
        return Classification {
            policy: QuantPolicy::KeepBf16,
            reason: "keep-bf16: norm tensor — all norms stay high precision (doctrine #2)",
        };
    }

    // 2) GATED set — int8 only behind a measured-CER kill-switch (default off).
    if is_decoder_attn_proj(name) {
        return Classification {
            policy: QuantPolicy::Gated(GatedKind::Attention),
            reason: "gated: attention q/k/v/o_proj — int8 only behind FOCR_INT8_ATTN (OQ-14)",
        };
    }
    if is_lm_head(name) {
        return Classification {
            policy: QuantPolicy::Gated(GatedKind::LmHead),
            reason: "gated: lm_head — int8 only behind FOCR_INT8_LMHEAD (OQ-14)",
        };
    }

    // 3) The validated INT8 set — decoder FFN / expert GEMMs.
    if is_decoder_ffn_proj(name) {
        return Classification {
            policy: QuantPolicy::Int8,
            reason: "int8: decoder FFN/expert GEMM — the validated quantizable set (doctrine #2)",
        };
    }

    // 4) Anything we do not recognize stays BF16 (conservative default — never
    //    quantize a tensor the recipe cannot positively identify as quantizable).
    Classification {
        policy: QuantPolicy::KeepBf16,
        reason: "keep-bf16: unclassified tensor — conservative default (refuse to quantize)",
    }
}

/// Resolve a static [`QuantPolicy`] against the runtime kill-switches, returning
/// the effective [`ResolvedPolicy`]. `attn_on` / `lmhead_on` are the (already
/// read) switch states; see [`switch_on`] for the env-truthiness rule.
#[must_use]
pub fn resolve_with(policy: QuantPolicy, attn_on: bool, lmhead_on: bool) -> ResolvedPolicy {
    match policy {
        QuantPolicy::KeepBf16 => ResolvedPolicy::KeepBf16,
        QuantPolicy::Int8 => ResolvedPolicy::Int8,
        QuantPolicy::Gated(GatedKind::Attention) => {
            if attn_on {
                ResolvedPolicy::Int8
            } else {
                ResolvedPolicy::KeepBf16
            }
        }
        QuantPolicy::Gated(GatedKind::LmHead) => {
            if lmhead_on {
                ResolvedPolicy::Int8
            } else {
                ResolvedPolicy::KeepBf16
            }
        }
    }
}

/// Resolve a tensor name end-to-end, reading the kill-switch env vars from the
/// process environment. Convenience over [`classify`] + [`resolve_with`].
#[must_use]
pub fn resolve(name: &str) -> ResolvedPolicy {
    let attn_on = switch_on(FOCR_INT8_ATTN_ENV);
    let lmhead_on = switch_on(FOCR_INT8_LMHEAD_ENV);
    resolve_with(classify(name).policy, attn_on, lmhead_on)
}

/// Whether an env-var kill-switch is truthy. A switch is ON only for
/// `1`/`true`/`on`/`yes` (case-insensitive, trimmed); unset or any other value
/// is OFF (default-off discipline, doctrine #2 / plan P1).
#[must_use]
pub fn switch_on(var: &str) -> bool {
    match std::env::var(var) {
        Ok(v) => is_truthy(&v),
        Err(_) => false,
    }
}

/// The truthiness rule for kill-switch values (split out so it is testable
/// without touching the process environment).
#[must_use]
pub fn is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "on" | "yes"
    )
}

/// A recipe handle binding the static classifier to a captured pair of
/// kill-switch states, so a whole manifest is resolved against one consistent
/// snapshot of the environment (rather than re-reading env per tensor).
#[derive(Debug, Clone, Copy)]
pub struct Recipe {
    attn_int8: bool,
    lmhead_int8: bool,
}

impl Recipe {
    /// Build a recipe with explicit kill-switch states (deterministic; for
    /// tests and the converter where the switches are decided once up front).
    #[must_use]
    pub fn new(attn_int8: bool, lmhead_int8: bool) -> Self {
        Self {
            attn_int8,
            lmhead_int8,
        }
    }

    /// Build a recipe by reading the kill-switch env vars once.
    #[must_use]
    pub fn from_env() -> Self {
        Self::new(
            switch_on(FOCR_INT8_ATTN_ENV),
            switch_on(FOCR_INT8_LMHEAD_ENV),
        )
    }

    /// The default (validated) recipe: both kill-switches OFF — attention and
    /// lm_head stay BF16, only the decoder FFN/expert set is int8.
    #[must_use]
    pub fn validated_default() -> Self {
        Self::new(false, false)
    }

    /// Whether attention int8 is enabled in this recipe.
    #[must_use]
    pub fn attn_int8(&self) -> bool {
        self.attn_int8
    }

    /// Whether lm_head int8 is enabled in this recipe.
    #[must_use]
    pub fn lmhead_int8(&self) -> bool {
        self.lmhead_int8
    }

    /// The static classification of `name` (env-independent).
    #[must_use]
    pub fn classify(&self, name: &str) -> Classification {
        classify(name)
    }

    /// The effective policy of `name` under this recipe's kill-switch states.
    #[must_use]
    pub fn resolve(&self, name: &str) -> ResolvedPolicy {
        resolve_with(classify(name).policy, self.attn_int8, self.lmhead_int8)
    }

    /// Whether `name` is quantized under this recipe.
    #[must_use]
    pub fn is_quantized(&self, name: &str) -> bool {
        self.resolve(name).is_quantized()
    }

    /// Classify a whole tensor-name set, returning a deterministic
    /// (`BTreeMap`-backed, name-sorted) map of name → [`ResolvedPolicy`]. The
    /// converter uses this to decide which tensors enter the AF-1 allocator and
    /// which are copied through BF16.
    #[must_use]
    pub fn resolve_manifest<I, S>(&self, names: I) -> BTreeMap<String, ResolvedPolicy>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        names
            .into_iter()
            .map(|n| {
                let name = n.as_ref().to_owned();
                let policy = self.resolve(&name);
                (name, policy)
            })
            .collect()
    }

    /// The sorted list of tensor names that are quantizable (resolve to int8)
    /// under this recipe — exactly the set handed to the AF-1 bit allocator.
    #[must_use]
    pub fn quantizable_names<I, S>(&self, names: I) -> Vec<String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut out: Vec<String> = names
            .into_iter()
            .filter_map(|n| {
                let name = n.as_ref();
                if self.is_quantized(name) {
                    Some(name.to_owned())
                } else {
                    None
                }
            })
            .collect();
        out.sort();
        out.dedup();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── KEEP_BF16: the recipe REFUSES the high-precision set ─────────────────

    #[test]
    fn refuses_vision_tower() {
        // CLIP-L and SAM tensors, including their attention projections, must
        // NEVER quantize — the vision-tower check wins over the attn-proj rule.
        let clip_attn = "model.vision_model.transformer.layers.15.self_attn.out_proj.weight";
        let clip_qkv = "model.vision_model.transformer.layers.3.self_attn.qkv_proj.weight";
        let sam_block = "model.sam_model.blocks.5.attn.qkv.weight";
        let clip_mlp = "model.vision_model.transformer.layers.7.mlp.fc1.weight";
        for n in [clip_attn, clip_qkv, sam_block, clip_mlp] {
            assert_eq!(
                classify(n).policy,
                QuantPolicy::KeepBf16,
                "vision tensor {n} must be KEEP_BF16"
            );
            assert_eq!(resolve(n), ResolvedPolicy::KeepBf16);
        }
    }

    #[test]
    fn refuses_projector() {
        for n in [
            "model.projector.layers.weight",
            "model.projector.layers.bias",
        ] {
            assert_eq!(classify(n).policy, QuantPolicy::KeepBf16, "{n}");
        }
    }

    #[test]
    fn refuses_embed_tokens() {
        assert_eq!(
            classify("model.embed_tokens.weight").policy,
            QuantPolicy::KeepBf16
        );
    }

    #[test]
    fn refuses_connector_params() {
        assert_eq!(
            classify("model.image_newline").policy,
            QuantPolicy::KeepBf16
        );
        assert_eq!(
            classify("model.view_seperator").policy,
            QuantPolicy::KeepBf16
        );
    }

    #[test]
    fn refuses_moe_router_gate_but_not_gate_proj() {
        // The router gate (gate-drift cliff) is KEEP_BF16...
        assert_eq!(
            classify("model.layers.5.mlp.gate.weight").policy,
            QuantPolicy::KeepBf16
        );
        // ...but `gate_proj` is an expert FFN projection — that IS int8.
        assert_eq!(
            classify("model.layers.0.mlp.gate_proj.weight").policy,
            QuantPolicy::Int8
        );
        assert_eq!(
            classify("model.layers.5.mlp.experts.10.gate_proj.weight").policy,
            QuantPolicy::Int8
        );
    }

    #[test]
    fn refuses_all_norms() {
        for n in [
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.11.post_attention_layernorm.weight",
            "model.vision_model.transformer.layers.2.norm1.weight",
            "model.sam_model.blocks.4.norm2.weight",
        ] {
            assert_eq!(
                classify(n).policy,
                QuantPolicy::KeepBf16,
                "norm tensor {n} must be KEEP_BF16"
            );
        }
    }

    // ── INT8: the validated decoder FFN/expert set ───────────────────────────

    #[test]
    fn allows_dense_layer0_ffn() {
        for n in [
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ] {
            assert_eq!(classify(n).policy, QuantPolicy::Int8, "{n}");
            assert_eq!(resolve(n), ResolvedPolicy::Int8);
        }
    }

    #[test]
    fn allows_routed_experts() {
        let n = "model.layers.10.mlp.experts.8.down_proj.weight";
        assert_eq!(classify(n).policy, QuantPolicy::Int8);
        // The CENSUS routed-expert example.
        let n2 = "model.layers.7.mlp.experts.63.up_proj.weight";
        assert_eq!(classify(n2).policy, QuantPolicy::Int8);
    }

    #[test]
    fn allows_shared_experts() {
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let n = format!("model.layers.11.mlp.shared_experts.{proj}.weight");
            assert_eq!(classify(&n).policy, QuantPolicy::Int8, "{n}");
        }
    }

    // ── GATED: attention + lm_head behind default-off kill-switches ──────────

    #[test]
    fn attention_proj_is_gated_default_off() {
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            let n = format!("model.layers.11.self_attn.{proj}.weight");
            assert_eq!(
                classify(&n).policy,
                QuantPolicy::Gated(GatedKind::Attention),
                "{n}"
            );
            // Default (switch off) => KEEP_BF16.
            assert_eq!(
                resolve_with(classify(&n).policy, false, false),
                ResolvedPolicy::KeepBf16
            );
            // Switch on => Int8.
            assert_eq!(
                resolve_with(classify(&n).policy, true, false),
                ResolvedPolicy::Int8
            );
        }
    }

    #[test]
    fn lm_head_is_gated_default_off() {
        let n = "lm_head.weight";
        assert_eq!(classify(n).policy, QuantPolicy::Gated(GatedKind::LmHead));
        assert_eq!(
            resolve_with(classify(n).policy, false, false),
            ResolvedPolicy::KeepBf16
        );
        assert_eq!(
            resolve_with(classify(n).policy, false, true),
            ResolvedPolicy::Int8
        );
        // The attn switch does NOT enable lm_head.
        assert_eq!(
            resolve_with(classify(n).policy, true, false),
            ResolvedPolicy::KeepBf16
        );
    }

    #[test]
    fn gated_kind_env_vars_are_distinct() {
        assert_eq!(GatedKind::Attention.env_var(), FOCR_INT8_ATTN_ENV);
        assert_eq!(GatedKind::LmHead.env_var(), FOCR_INT8_LMHEAD_ENV);
        assert_ne!(FOCR_INT8_ATTN_ENV, FOCR_INT8_LMHEAD_ENV);
    }

    // ── truthiness rule ──────────────────────────────────────────────────────

    #[test]
    fn truthiness_rule() {
        for on in ["1", "true", "TRUE", "On", "yes", " yes ", "Yes"] {
            assert!(is_truthy(on), "{on:?} should be truthy");
        }
        for off in ["0", "false", "", "off", "no", "2", "enabled", "ON1"] {
            assert!(!is_truthy(off), "{off:?} should be falsy");
        }
    }

    // ── Recipe handle + manifest resolution ──────────────────────────────────

    #[test]
    fn recipe_default_keeps_attn_and_lmhead_bf16() {
        let r = Recipe::validated_default();
        assert!(!r.attn_int8());
        assert!(!r.lmhead_int8());
        assert_eq!(
            r.resolve("model.layers.0.self_attn.q_proj.weight"),
            ResolvedPolicy::KeepBf16
        );
        assert_eq!(r.resolve("lm_head.weight"), ResolvedPolicy::KeepBf16);
        // The validated set is still int8.
        assert!(r.is_quantized("model.layers.3.mlp.experts.0.down_proj.weight"));
    }

    #[test]
    fn recipe_with_switches_on_quantizes_gated() {
        let r = Recipe::new(true, true);
        assert!(r.is_quantized("model.layers.0.self_attn.v_proj.weight"));
        assert!(r.is_quantized("lm_head.weight"));
        // KEEP_BF16 set is unaffected by the switches.
        assert!(!r.is_quantized("model.norm.weight"));
        assert!(
            !r.is_quantized("model.vision_model.transformer.layers.0.self_attn.qkv_proj.weight")
        );
    }

    #[test]
    fn quantizable_names_default_excludes_vision_attn_lmhead_router() {
        let names = vec![
            "model.vision_model.transformer.layers.0.self_attn.out_proj.weight",
            "model.sam_model.blocks.0.attn.qkv.weight",
            "model.projector.layers.weight",
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.layers.5.mlp.gate.weight", // router
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight", // gated off
            "lm_head.weight",                         // gated off
            "model.layers.0.mlp.down_proj.weight",    // dense ffn -> int8
            "model.layers.3.mlp.experts.7.up_proj.weight", // routed -> int8
            "model.layers.3.mlp.shared_experts.gate_proj.weight", // shared -> int8
        ];
        let r = Recipe::validated_default();
        let q = r.quantizable_names(names.iter().copied());
        assert_eq!(
            q,
            vec![
                "model.layers.0.mlp.down_proj.weight".to_string(),
                "model.layers.3.mlp.experts.7.up_proj.weight".to_string(),
                "model.layers.3.mlp.shared_experts.gate_proj.weight".to_string(),
            ]
        );
    }

    #[test]
    fn resolve_manifest_is_deterministic_and_sorted() {
        let r = Recipe::validated_default();
        let names = ["zzz.unknown", "model.norm.weight", "lm_head.weight"];
        let m = r.resolve_manifest(names);
        // BTreeMap => sorted keys.
        let keys: Vec<&String> = m.keys().collect();
        assert_eq!(
            keys,
            vec!["lm_head.weight", "model.norm.weight", "zzz.unknown"]
        );
        // Unknown tensor => conservative KEEP_BF16.
        assert_eq!(m["zzz.unknown"], ResolvedPolicy::KeepBf16);
    }

    #[test]
    fn unclassified_tensor_is_kept_bf16() {
        let c = classify("some.weird.tensor.we.do.not.know");
        assert_eq!(c.policy, QuantPolicy::KeepBf16);
        assert!(c.reason.contains("conservative"));
    }

    #[test]
    fn reasons_are_present_and_stable() {
        // Every classification carries a non-empty reason for the ledger.
        for n in [
            "model.vision_model.x.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "lm_head.weight",
            "model.norm.weight",
        ] {
            assert!(!classify(n).reason.is_empty(), "{n}");
        }
    }
}
