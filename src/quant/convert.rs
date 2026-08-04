//! `focr convert`'s offline quantizer: raw bf16 safetensors → a self-contained
//! int8 `.focrq` container.
//!
//! This is the OFFLINE half of the int8 pipeline. It wires three already-landed
//! pieces together — it invents no new math:
//!
//! 1. [`crate::native_engine::weights::Weights`] reads the raw bf16 safetensors
//!    shard and enumerates every tensor (name → dtype/shape/bytes).
//! 2. [`crate::native_engine::nn::quantize_int8`] is the **exact** per-output-
//!    channel symmetric int8 quantizer the LOAD-TIME path
//!    ([`crate::native_engine::decoder::DecoderWeightCacheI8::build`]) runs.
//! 3. [`super::focrq::FocrqBuilder`] serializes the result to the byte-exact
//!    `.focrq` layout the committed reader parses.
//!
//! ## The byte-for-byte contract
//!
//! The validated Unlimited-OCR recipe quantizes only decoder FFN/expert GEMMs
//! and leaves attention plus `lm_head` high precision. This converter classifies
//! each tensor with [`is_decoder_int8_tensor_for`], which delegates the default
//! architecture to [`super::recipe::Recipe::validated_default`] and remains keyed
//! by the target [`ModelArch`] for model-zoo layouts.
//!
//! * for a decoder int8 tensor: widens the bf16 `[n, k]` weight to f32 and calls
//!   the SAME [`nn::quantize_int8`], emitting a `QInt8PerChan` record whose int8
//!   payload + f32 inline scales are byte-identical to what `build` computes at
//!   load time;
//! * for everything else (the whole SAM+CLIP vision tower, the projector,
//!   `embed_tokens`, the MoE router `mlp.gate.weight`, and ALL norms): copies the
//!   original bf16/f32 bytes verbatim.
//!
//! The high-precision tensors are copied byte-for-byte. The existing all-int8
//! runtime cache is an explicitly gated experimental path; it is not the
//! converter's default artifact contract.

use sha2::{Digest, Sha256};

use super::focrq::{FocrqBuilder, WriteDType};
use super::recipe::Recipe;
use crate::error::{FocrError, FocrResult};
use crate::native_engine::model_arch::ModelArch;
use crate::native_engine::nn;
use crate::native_engine::weights::{DType, Weights};

/// Frozen identifier stamped into Unlimited-OCR conservative int8 artifacts.
pub const UNLIMITED_OCR_INT8_RECIPE_ID: &str = "unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1";

/// The quantization target requested on the `focr convert` command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvertQuant {
    /// Per-output-channel symmetric int8 on the decoder GEMM tensors — the
    /// validated, must-have path.
    Int8,
    /// Group-quantized int4. NOT yet shipped (AGENTS.md doctrine #1: never ship
    /// an unverified lossy path); [`safetensors_to_focrq`] returns
    /// [`FocrError::NotImplemented`].
    Int4,
}

/// SHA-256 of the raw input shard bytes, as the 32-byte digest the `.focrq`
/// preamble/header carry for provenance. Hashing the bytes the converter
/// actually read pins the artifact to its exact source checkpoint.
#[must_use]
pub fn sha256_of_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// Whether `name` is one of the decoder tensors the target architecture's
/// validated int8 recipe covers.
///
/// A **pure function of the tensor name + the arch descriptor** (no I/O, no env)
/// so it is deterministic and unit-testable. The classification is ARCH-AWARE,
/// not name-coincidence — two facts come from [`ModelArch`]:
///
/// * [`ModelArch::decoder_layers_prefix`] — WHERE the AR decoder lives.
///   Unlimited-OCR/GOT put it at `model.layers.`; SmolVLM2's Idefics3 splice
///   nests it at `model.text_model.layers.` (spec §12). The prefix is also what
///   keeps look-alike vision tensors out: SigLIP's
///   `model.vision_model.encoder.layers.{i}.self_attn.q_proj.weight` shares the
///   leaf name with a decoder GEMM but is NOT under the decoder prefix, so it
///   stays high-precision.
///
/// Unlimited-OCR delegates to [`Recipe::validated_default`]: only FFN/expert
/// projections are int8; attention and `lm_head` remain high precision. Other
/// model families retain their architecture-specific, separately certified
/// storage rules.
///
/// For non-default architectures, the prefix-based set is:
///
/// * attention `self_attn.{q,k,v,o}_proj.weight` (GQA k/v panels like
///   SmolVLM2's `[320, 960]` are just rank-2 `[n, k]` — nothing special);
/// * the dense SwiGLU and every MoE routed/shared expert
///   `mlp.…{gate,up,down}_proj.weight`.
///
/// Everything else is high-precision and returns `false`: ALL norms
/// (`*_layernorm.weight`, `model.norm.weight`), the MoE router `mlp.gate.weight`
/// (note: `gate`, NOT `gate_proj`), `embed_tokens`, the projector/connector, and
/// the entire vision tower.
#[must_use]
pub fn is_decoder_int8_tensor_for(name: &str, arch: &dyn ModelArch) -> bool {
    if arch.id() == crate::native_engine::model_arch::default_arch().id() {
        return Recipe::validated_default().is_quantized(name);
    }
    if name == "lm_head.weight" {
        return arch.lm_head_stored_int8();
    }
    let Some(rest) = name.strip_prefix(arch.decoder_layers_prefix()) else {
        return false;
    };
    // The per-layer GEMM naming is a fact of the arch's DECODER FAMILY
    // (D-census §13): OPT (OneChart) names them `self_attn.{q,k,v,out}_proj`
    // + bare `fc1`/`fc2` (all `.bias` and the two per-layer LayerNorms stay
    // high-precision); every other family keeps the historical Qwen/Llama/
    // DeepSeek rule VERBATIM — `self_attn.{q,k,v,o}_proj` plus anything under
    // the `.mlp.` subtree ending in `{gate,up,down}_proj.weight` (which is
    // what quantizes the MoE `mlp.experts.N.*` / `mlp.shared_experts.*`
    // GEMMs), with the bare router `.mlp.gate.weight` excluded.
    if arch.decoder() == crate::native_engine::model_arch::Decoder::OptDense {
        return rest.ends_with(".self_attn.q_proj.weight")
            || rest.ends_with(".self_attn.k_proj.weight")
            || rest.ends_with(".self_attn.v_proj.weight")
            || rest.ends_with(".self_attn.out_proj.weight")
            || rest.ends_with(".fc1.weight")
            || rest.ends_with(".fc2.weight");
    }
    // Seq2SeqDense (TrOMR): the 40 decoder GEMMs (`to_{q,k,v}`/`to_out.0`
    // per attn sublayer, `net.0.proj`/`net.3` per ff, 8.4 M params) are the
    // int8 set — enabled by the bd-av64.12 measurement (see the bead: the
    // corpus gate + goldens arbitrate losslessness; the encoder and every
    // norm/embedding stay high-precision per the quant doctrine).
    if arch.decoder() == crate::native_engine::model_arch::Decoder::Seq2SeqDense {
        return rest.ends_with(".to_q.weight")
            || rest.ends_with(".to_k.weight")
            || rest.ends_with(".to_v.weight")
            || rest.ends_with(".to_out.0.weight")
            || rest.ends_with(".net.0.proj.weight")
            || rest.ends_with(".net.3.weight");
    }
    if rest.contains(".self_attn.") {
        return rest.ends_with(".q_proj.weight")
            || rest.ends_with(".k_proj.weight")
            || rest.ends_with(".v_proj.weight")
            || rest.ends_with(".o_proj.weight");
    }
    if rest.contains(".mlp.") {
        return rest.ends_with(".gate_proj.weight")
            || rest.ends_with(".up_proj.weight")
            || rest.ends_with(".down_proj.weight");
    }
    false
}

/// [`is_decoder_int8_tensor_for`] instantiated at the default (Unlimited-OCR)
/// arch — the historical name-only classifier, kept so the v1 byte contract has
/// an explicit, testable anchor.
#[must_use]
pub fn is_decoder_int8_tensor(name: &str) -> bool {
    is_decoder_int8_tensor_for(name, crate::native_engine::model_arch::default_arch())
}

/// Convert a loaded raw-safetensors [`Weights`] into a self-contained `.focrq`
/// blob (preamble + header JSON + payload), ready to write to disk.
///
/// Tensors are emitted in sorted name order (the builder's `BTreeMap`), so the
/// output is byte-deterministic for a fixed input. `arch_target` is the packing
/// byte recorded in the header (`0` Generic … `3` X86Amx); `source_sha256` is the
/// 32-byte digest of the input shard ([`sha256_of_bytes`]).
///
/// `arch` is the target architecture (its `model_id` + license notice go into the
/// header, and its [`ModelArch::tie_word_embeddings`] decides whether the tied
/// `lm_head.weight` is omitted). Passing [`crate::native_engine::model_arch::default_arch`]
/// reproduces the historical Unlimited-OCR output **byte-for-byte** (default id ⇒
/// the `model_id` key is omitted, the notice is the Baidu/MIT one, and `lm_head`
/// is stored), so existing artifacts are unchanged.
///
/// # Errors
/// * [`FocrError::NotImplemented`] for [`ConvertQuant::Int4`] — the int4 group
///   path is not yet validated (doctrine #1).
/// * [`FocrError::FormatMismatch`] if a decoder int8 tensor is not rank-2
///   `[n, k]`, if a tensor's bytes disagree with its shape, or if an input tensor
///   is unexpectedly already quantized (the converter input must be raw bf16/f32).
pub fn safetensors_to_focrq(
    weights: &Weights,
    quant: ConvertQuant,
    arch_target: u8,
    source_sha256: [u8; 32],
    arch: &dyn ModelArch,
) -> FocrResult<Vec<u8>> {
    if quant == ConvertQuant::Int4 {
        return Err(FocrError::NotImplemented(
            "focr convert --quant int4 is not yet supported; the int4 group-quantized \
             path is unvalidated (use --quant int8)"
                .into(),
        ));
    }

    let mut builder = FocrqBuilder::new()
        .with_arch_target(arch_target)
        .with_source_sha256(source_sha256)
        .with_model_id(arch.id())
        .with_license_notice(arch.license_notice());
    if arch.id() == crate::native_engine::model_arch::default_arch().id() {
        builder = builder.with_packing_manifest_json(format!(
            r#"{{"quant_recipe":"{UNLIMITED_OCR_INT8_RECIPE_ID}"}}"#
        ));
    }

    // When the arch ties `lm_head` to `embed_tokens` (GOT-OCR2: proven byte-identical,
    // spec §12), omit `lm_head.weight` — it is a duplicate the loader reconstructs
    // from the stored embedding. Skips ~155 M params from the artifact.
    let omit_lm_head = arch.tie_word_embeddings();

    // When the arch instead declares an UNTIED, high-precision-stored lm_head
    // (SmolVLM2), re-verify the untie against the actual bytes — the census
    // (docs/zoo/smolvlm2-spec.md §12) demands the full-tensor inequality be
    // re-checked at convert time, so a tied checkpoint mislabeled with this
    // arch id fails loud instead of silently shipping a redundant 47 M params.
    if !omit_lm_head && !arch.lm_head_stored_int8() {
        verify_untied_lm_head(weights, arch)?;
    }
    // The SYMMETRIC guard (fresh-eyes fix): omitting `lm_head.weight` on an
    // arch's tie claim was previously taken on trust — a genuinely-untied
    // checkpoint mislabeled with a tied arch id would silently DROP its real
    // head (the loader would reconstruct from embed_tokens = wrong logits, no
    // error anywhere). Verify the tie against the actual bytes before dropping.
    if omit_lm_head {
        verify_tied_lm_head(weights, arch)?;
    }

    // `names()` is already sorted (the directory is a `BTreeMap`); collect so the
    // immutable directory borrow is released before the per-tensor accessors run.
    let names: Vec<String> = weights.names().map(str::to_owned).collect();
    for name in &names {
        if omit_lm_head && name == "lm_head.weight" {
            continue;
        }
        if is_decoder_int8_tensor_for(name, arch) {
            quantize_decoder_tensor(&mut builder, weights, name, arch_target)?;
        } else {
            if arch.id() == crate::native_engine::model_arch::default_arch().id()
                && matches!(
                    weights.record(name).map(|record| record.dtype),
                    Some(DType::F16)
                )
            {
                return Err(FocrError::FormatMismatch(format!(
                    "convert: Unlimited-OCR high-precision tensor {name:?} is F16; recipe \
                     {UNLIMITED_OCR_INT8_RECIPE_ID} permits only source BF16 or F32"
                )));
            }
            copy_high_precision_tensor(&mut builder, weights, name)?;
        }
    }
    Ok(builder.build())
}

/// Convert-time proof that an arch-declared UNTIED `lm_head` really is untied:
/// when both `lm_head.weight` and the arch's `embed_tokens` tensor exist with
/// identical shape/dtype, their raw bytes must DIFFER. Bytes-equal means the
/// checkpoint is tied and the arch descriptor (or the `--model-id`) is wrong —
/// refuse rather than store a silent duplicate. Either tensor missing is not
/// this function's problem (the load path reports missing tensors itself).
fn verify_untied_lm_head(weights: &Weights, arch: &dyn ModelArch) -> FocrResult<()> {
    let embed_name = arch.embed_tokens_name();
    let (Ok(head), Ok(embed)) = (weights.tensor("lm_head.weight"), weights.tensor(embed_name))
    else {
        return Ok(());
    };
    if head.dtype == embed.dtype && head.shape == embed.shape && head.data == embed.data {
        return Err(FocrError::FormatMismatch(format!(
            "convert: arch {:?} declares an UNTIED lm_head, but lm_head.weight is \
             byte-identical to {embed_name:?} — this checkpoint ties its embeddings; \
             the --model-id (or its descriptor) is wrong",
            arch.id()
        )));
    }
    Ok(())
}

/// The mirror of [`verify_untied_lm_head`]: an arch that DECLARES tied
/// embeddings (and therefore omits `lm_head.weight` from the artifact) must
/// actually have byte-identical head/embed tensors in the source checkpoint —
/// otherwise the omission destroys the real head. Absent `lm_head.weight` is
/// fine (already-tied checkpoints often don't store one at all).
fn verify_tied_lm_head(weights: &Weights, arch: &dyn ModelArch) -> FocrResult<()> {
    let embed_name = arch.embed_tokens_name();
    let (Ok(head), Ok(embed)) = (weights.tensor("lm_head.weight"), weights.tensor(embed_name))
    else {
        return Ok(());
    };
    if head.dtype != embed.dtype || head.shape != embed.shape || head.data != embed.data {
        return Err(FocrError::FormatMismatch(format!(
            "convert: arch {:?} declares TIED embeddings (lm_head omitted from the \
             artifact), but this checkpoint's lm_head.weight differs from \
             {embed_name:?} — omitting it would silently destroy the real head; \
             the --model-id (or its descriptor) is wrong",
            arch.id()
        )));
    }
    Ok(())
}

/// Quantize one decoder `[n, k]` weight to per-output-channel symmetric int8 with
/// the SAME [`nn::quantize_int8`] the load-time cache uses, and stage it as a
/// `QInt8PerChan` record (int8 payload + `n` f32 inline scales).
fn quantize_decoder_tensor(
    builder: &mut FocrqBuilder,
    weights: &Weights,
    name: &str,
    arch_target: u8,
) -> FocrResult<()> {
    let record = weights.record(name).ok_or_else(|| {
        FocrError::FormatMismatch(format!("convert: tensor {name:?} missing from directory"))
    })?;
    if record.shape.len() != 2 {
        return Err(FocrError::FormatMismatch(format!(
            "convert: decoder int8 tensor {name:?} must be rank-2 [n, k], got shape {:?}",
            record.shape
        )));
    }
    let (n, k) = (record.shape[0], record.shape[1]);
    // Widen bf16→f32 (exact), then the per-OC symmetric int8 quant — `out = n`
    // (shape[0]), exactly the `quant_oc(.., out)` arg the load-time builder passes.
    let mat = weights.mat(name)?;
    let q = nn::quantize_int8(&mat.data, n, k);
    // `i8 → u8` is a pure bit reinterpret (the reader does the inverse `b as i8`);
    // scales are little-endian f32 — the `.focrq` QInt8PerChan inline layout.
    //
    // `--arch aarch64-smmla` (arch_target 1, bd-2mo.3): the int8 payload is
    // stored as OFFLINE-packed SMMLA panels — the same permutation the i8mm
    // micro-kernel builds at runtime ([`crate::simd::pack::smmla_pack_panels`],
    // the single source of truth), so a matching host loads contiguous
    // register tiles with zero runtime shuffle. A pure permutation of the
    // already-quantized bytes: lossless by construction, and the loader
    // un-permutes on any non-SMMLA host (degrade to generic, never UB). The
    // quantized VALUES are identical across every arch_target.
    let weight_bytes: Vec<u8> = if arch_target == 1 {
        let (panels, _pairs, _kb) = crate::simd::pack::smmla_pack_panels(&q.w, 0, n, k, k);
        panels.iter().map(|&v| v as u8).collect()
    } else {
        q.w.iter().map(|&v| v as u8).collect()
    };
    let scale_bytes: Vec<u8> = q.scales.iter().flat_map(|&s| s.to_le_bytes()).collect();
    builder.add_quantized(
        name,
        WriteDType::QInt8PerChan,
        vec![n, k],
        weight_bytes,
        scale_bytes,
        0,
        0,
    )
}

/// Copy one high-precision tensor (the whole vision tower, projector,
/// `embed_tokens`, router gate, norms) verbatim — its on-disk bytes are emitted
/// unchanged, with the dtype mapped to the writer's tag.
fn copy_high_precision_tensor(
    builder: &mut FocrqBuilder,
    weights: &Weights,
    name: &str,
) -> FocrResult<()> {
    let view = weights.tensor(name)?;
    let dtype = match view.dtype {
        DType::F32 => WriteDType::F32,
        DType::F16 => WriteDType::F16,
        DType::BF16 => WriteDType::Bf16,
        DType::QInt8PerChan | DType::QInt4PerGroup => {
            return Err(FocrError::FormatMismatch(format!(
                "convert: tensor {name:?} is already quantized ({:?}); the converter input \
                 must be raw bf16/f32 safetensors",
                view.dtype
            )));
        }
    };
    builder.add_tensor(name, dtype, view.shape.to_vec(), view.data.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FOCR_MODEL_LICENSE_NOTICE;
    use half::bf16;

    /// Hand-assemble a minimal raw-safetensors blob from `(name, shape, f32
    /// values)` BF16 tensors laid out contiguously in directory order — the
    /// converter's input form. Mirrors the reader's own test builder.
    fn build_safetensors(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut entries = Vec::new();
        let mut payload = Vec::new();
        for (name, shape, values) in tensors {
            let beg = payload.len();
            for &v in values {
                payload.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
            }
            let end = payload.len();
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"BF16\",\"shape\":{shape:?},\
                 \"data_offsets\":[{beg},{end}]}}"
            ));
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut blob = Vec::new();
        blob.extend_from_slice(&(header.len() as u64).to_le_bytes());
        blob.extend_from_slice(header.as_bytes());
        blob.extend_from_slice(&payload);
        blob
    }

    /// A tiny synthetic checkpoint: a few decoder int8-shaped tensors (attention,
    /// dense FFN, MoE expert, lm_head) + the high-precision set (router gate,
    /// norms, a vision tensor). Values vary per row so per-OC scales differ.
    fn synthetic_safetensors() -> Vec<u8> {
        let ramp = |n: usize, k: usize, bias: f32| -> Vec<f32> {
            (0..n * k).map(|i| (i as f32) * 0.5 - bias).collect()
        };
        build_safetensors(&[
            // decoder int8 set
            ("lm_head.weight", vec![6, 8], ramp(6, 8, 11.0)),
            (
                "model.layers.0.self_attn.q_proj.weight",
                vec![4, 8],
                ramp(4, 8, 7.0),
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                vec![5, 8],
                ramp(5, 8, 9.0),
            ),
            (
                "model.layers.1.mlp.experts.0.up_proj.weight",
                vec![3, 8],
                ramp(3, 8, 5.0),
            ),
            (
                "model.layers.1.mlp.shared_experts.down_proj.weight",
                vec![8, 3],
                ramp(8, 3, 4.0),
            ),
            // high-precision set
            (
                "model.layers.1.mlp.gate.weight",
                vec![2, 8],
                ramp(2, 8, 3.0),
            ),
            (
                "model.layers.0.input_layernorm.weight",
                vec![8],
                ramp(1, 8, 1.0),
            ),
            ("model.norm.weight", vec![8], ramp(1, 8, 2.0)),
            (
                "vision_model.patch_embed.weight",
                vec![2, 3],
                ramp(2, 3, 1.0),
            ),
        ])
    }

    const INT8_NAMES: &[&str] = &[
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.experts.0.up_proj.weight",
        "model.layers.1.mlp.shared_experts.down_proj.weight",
    ];

    const KEPT_NAMES: &[&str] = &[
        "model.layers.1.mlp.gate.weight",
        "model.layers.0.input_layernorm.weight",
        "model.norm.weight",
        "vision_model.patch_embed.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ];

    #[test]
    fn classifier_matches_validated_default_recipe() {
        for name in INT8_NAMES {
            assert!(is_decoder_int8_tensor(name), "{name} must be int8");
        }
        for name in KEPT_NAMES {
            assert!(
                !is_decoder_int8_tensor(name),
                "{name} must stay high-precision"
            );
        }
        // The router gate vs the dense FFN gate projection is the subtle split;
        // attention and lm_head are also kept unless their independent gates are
        // certified and explicitly armed in a future artifact recipe.
        assert!(!is_decoder_int8_tensor("model.layers.3.mlp.gate.weight"));
        assert!(is_decoder_int8_tensor(
            "model.layers.3.mlp.gate_proj.weight"
        ));
        // A vision tensor that merely *contains* `.mlp.`/`down_proj` is excluded
        // because it is not under `model.layers.`.
        assert!(!is_decoder_int8_tensor(
            "vision_model.encoder.layers.2.mlp.fc2.weight"
        ));
    }

    #[test]
    fn unlimited_ocr_validated_recipe_has_exactly_2148_int8_tensors() {
        let mut names = Vec::new();
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            names.push(format!("model.layers.0.mlp.{proj}.weight"));
        }
        for layer in 1..12 {
            for expert in 0..64 {
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    names.push(format!(
                        "model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                    ));
                }
            }
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                names.push(format!(
                    "model.layers.{layer}.mlp.shared_experts.{proj}.weight"
                ));
            }
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                names.push(format!("model.layers.{layer}.self_attn.{proj}.weight"));
            }
        }
        names.extend([
            "lm_head.weight".to_owned(),
            "model.layers.0.self_attn.q_proj.weight".to_owned(),
            "model.layers.0.self_attn.k_proj.weight".to_owned(),
            "model.layers.0.self_attn.v_proj.weight".to_owned(),
            "model.layers.0.self_attn.o_proj.weight".to_owned(),
        ]);

        let quantized = names
            .iter()
            .filter(|name| is_decoder_int8_tensor(name))
            .count();
        assert_eq!(quantized, 2148);
        assert_eq!(names.len() - quantized, 49, "48 attention + one lm_head");
    }

    /// bd-2mo.3/.3.1: the `--arch aarch64-smmla` payload is a LOSSLESS
    /// permutation — the loader's un-permuted `qint8()` readback is
    /// int8-byte-identical (and scale-identical) to the generic artifact's,
    /// for every decoder tensor including padded shapes (odd n, k % 8 != 0).
    /// Stronger than the dequant-equivalence acceptance criterion: equality
    /// holds on the quantized integers themselves.
    #[test]
    fn smmla_arch_packing_is_a_lossless_permutation() {
        let src = synthetic_safetensors();
        let w = Weights::from_bytes(src).expect("synthetic safetensors parse");
        let arch = crate::native_engine::model_arch::default_arch();
        let generic = safetensors_to_focrq(&w, ConvertQuant::Int8, 0, [7u8; 32], arch)
            .expect("convert generic");
        let packed = safetensors_to_focrq(&w, ConvertQuant::Int8, 1, [7u8; 32], arch)
            .expect("convert smmla");
        // The reorder is real: the blobs differ (padded tensors change length,
        // multi-K-block tensors permute), yet load back logically identical.
        assert_ne!(generic, packed, "arch 1 must actually reorder the payload");
        let g = Weights::from_bytes(generic).expect("generic parse");
        let p =
            Weights::from_bytes(packed).expect("packed parse (census must accept panel lengths)");
        assert_eq!(g.arch_target(), 0);
        assert_eq!(p.arch_target(), 1);
        for name in INT8_NAMES {
            let a = g.qint8(name).expect("generic qint8");
            let b = p.qint8(name).expect("packed qint8");
            // LAYOUT-AWARE compare (the bug CI's aarch64 advisory matrix caught
            // on its first revived run): on a host whose dispatched tier is
            // SMMLA (Neoverse-class; NOT the M4, which prefers SDOT) the
            // loader CORRECTLY keeps the offline panels, so the logical
            // equality must un-permute before comparing raw bytes.
            let b_rm = match b.layout {
                crate::native_engine::tensor::WeightLayout::RowMajor => b.w.clone(),
                crate::native_engine::tensor::WeightLayout::SmmlaPanels => {
                    crate::simd::pack::smmla_unpack_panels(&b.w, b.n, b.k)
                        .expect("panel stream length is loader-validated")
                }
            };
            assert_eq!(a.w, b_rm, "{name}: int8 weights identical across packings");
            assert_eq!(
                a.scales, b.scales,
                "{name}: scales identical across packings"
            );
            assert_eq!((a.n, a.k), (b.n, b.k));
            println!(
                r#"{{"event":"prepack_equiv","arch":"aarch64-smmla","tensor":"{name}","n":{},"k":{},"ok":true}}"#,
                a.n, a.k
            );
        }
    }

    /// bd-2mo.3 acceptance: same source → same packed bytes (content-hash
    /// equal), for both the generic and the SMMLA packings.
    #[test]
    fn arch_packings_are_byte_deterministic() {
        let src = synthetic_safetensors();
        let w = Weights::from_bytes(src).expect("synthetic safetensors parse");
        let arch = crate::native_engine::model_arch::default_arch();
        for arch_target in [0u8, 1] {
            let a = safetensors_to_focrq(&w, ConvertQuant::Int8, arch_target, [7u8; 32], arch)
                .expect("convert");
            let b = safetensors_to_focrq(&w, ConvertQuant::Int8, arch_target, [7u8; 32], arch)
                .expect("convert again");
            let ha = sha256_of_bytes(&a);
            assert_eq!(
                ha,
                sha256_of_bytes(&b),
                "arch {arch_target}: nondeterministic output"
            );
            let hex: String = ha.iter().map(|x| format!("{x:02x}")).collect();
            println!(
                r#"{{"event":"prepack_equiv","arch_target":{arch_target},"ok":true,"content_hash":"{hex}"}}"#
            );
        }
    }

    #[test]
    fn int8_decoder_tensors_match_load_time_quant() {
        let src = synthetic_safetensors();
        let w = Weights::from_bytes(src).expect("synthetic safetensors parse");
        let blob = safetensors_to_focrq(
            &w,
            ConvertQuant::Int8,
            2,
            [7u8; 32],
            crate::native_engine::model_arch::default_arch(),
        )
        .expect("convert int8");
        let out = Weights::from_bytes(blob).expect("focrq parse");

        for name in INT8_NAMES {
            let rec = w.record(name).expect("record");
            let (n, k) = (rec.shape[0], rec.shape[1]);
            // The byte-for-byte oracle: the SAME nn::quantize_int8 the load-time
            // DecoderWeightCacheI8::build runs on the SAME widened f32 weight.
            let expected = nn::quantize_int8(&w.mat(name).unwrap().data, n, k);
            let got = out.qint8(name).expect("qint8 readback");
            assert_eq!(got.n, n, "{name} n");
            assert_eq!(got.k, k, "{name} k");
            assert_eq!(
                got.w, expected.w,
                "{name} int8 payload must be bit-identical"
            );
            assert_eq!(
                got.scales, expected.scales,
                "{name} f32 scales must be bit-identical"
            );
        }
    }

    #[test]
    fn high_precision_tensors_roundtrip_unchanged() {
        let src = synthetic_safetensors();
        let w = Weights::from_bytes(src).expect("synthetic safetensors parse");
        let blob = safetensors_to_focrq(
            &w,
            ConvertQuant::Int8,
            2,
            [7u8; 32],
            crate::native_engine::model_arch::default_arch(),
        )
        .expect("convert int8");
        let out = Weights::from_bytes(blob).expect("focrq parse");

        for name in KEPT_NAMES {
            let before = w.tensor(name).expect("src view");
            let after = out.tensor(name).expect("out view");
            assert_eq!(after.dtype, DType::BF16, "{name} dtype preserved");
            assert_eq!(after.shape, before.shape, "{name} shape preserved");
            assert_eq!(after.data, before.data, "{name} raw bytes verbatim");
            // And the widened f32 values are identical too.
            assert_eq!(
                out.vec(name).unwrap(),
                w.vec(name).unwrap(),
                "{name} widened values"
            );
        }
    }

    #[test]
    fn header_carries_arch_sha_and_license() {
        let src = synthetic_safetensors();
        let w = Weights::from_bytes(src).expect("synthetic safetensors parse");
        let blob = safetensors_to_focrq(
            &w,
            ConvertQuant::Int8,
            2,
            [7u8; 32],
            crate::native_engine::model_arch::default_arch(),
        )
        .expect("convert int8");
        let out = Weights::from_bytes(blob).expect("focrq parse");
        assert!(out.is_focrq());
        assert_eq!(out.arch_target(), 2);
        assert_eq!(out.source_sha256(), "07".repeat(32));
        assert_eq!(out.license_notice(), FOCR_MODEL_LICENSE_NOTICE);
        // Every source tensor survives (count preserved, names intact).
        assert_eq!(out.len(), w.len());
        for name in INT8_NAMES.iter().chain(KEPT_NAMES) {
            assert!(out.contains(name), "{name} present in converted artifact");
        }
    }

    #[test]
    fn sha256_is_the_input_digest() {
        let bytes = b"franken_ocr converter provenance";
        let a = sha256_of_bytes(bytes);
        let b = sha256_of_bytes(bytes);
        assert_eq!(a, b, "sha256 is deterministic");
        // Known SHA-256 of the empty input (e3b0c442…).
        let empty = sha256_of_bytes(&[]);
        assert_eq!(empty[0], 0xe3);
        assert_eq!(empty[1], 0xb0);
        assert_eq!(empty[2], 0xc4);
        assert_eq!(empty[3], 0x42);
    }

    #[test]
    fn int4_is_not_implemented() {
        let src = synthetic_safetensors();
        let w = Weights::from_bytes(src).expect("synthetic safetensors parse");
        let err = safetensors_to_focrq(
            &w,
            ConvertQuant::Int4,
            0,
            [0u8; 32],
            crate::native_engine::model_arch::default_arch(),
        )
        .expect_err("int4 must be NotImplemented");
        assert!(matches!(err, FocrError::NotImplemented(_)), "got {err:?}");
        assert_eq!(err.exit_code(), 1);
    }

    // ── B2: arch-aware GOT-OCR2 convert ──────────────────────────────────────

    /// A GOT-OCR2-shaped synthetic checkpoint: a tied `lm_head` (== `embed_tokens`),
    /// the Qwen2 decoder int8 GEMMs (+ a qkv bias), norms, the `mm_projector_vary`
    /// connector, and a `model.vision_tower_high.*` tensor.
    fn synthetic_got_safetensors() -> Vec<u8> {
        let ramp = |n: usize, k: usize, bias: f32| -> Vec<f32> {
            (0..n * k).map(|i| (i as f32) * 0.25 - bias).collect()
        };
        let embed = ramp(6, 8, 11.0);
        build_safetensors(&[
            ("lm_head.weight", vec![6, 8], embed.clone()), // tied -> omitted
            ("model.embed_tokens.weight", vec![6, 8], embed), // stored HP (serves both)
            (
                "model.layers.0.self_attn.q_proj.weight",
                vec![8, 8],
                ramp(8, 8, 7.0),
            ),
            (
                "model.layers.0.self_attn.q_proj.bias",
                vec![8],
                ramp(1, 8, 1.0),
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                vec![10, 8],
                ramp(10, 8, 9.0),
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                vec![8, 10],
                ramp(8, 10, 4.0),
            ),
            (
                "model.layers.0.input_layernorm.weight",
                vec![8],
                ramp(1, 8, 1.0),
            ),
            ("model.norm.weight", vec![8], ramp(1, 8, 2.0)),
            (
                "model.mm_projector_vary.weight",
                vec![8, 8],
                ramp(8, 8, 1.0),
            ),
            (
                "model.vision_tower_high.blocks.0.attn.proj.weight",
                vec![8, 8],
                ramp(8, 8, 2.0),
            ),
        ])
    }

    #[test]
    fn got_convert_omits_tied_lm_head_and_tags_arch() {
        let w = Weights::from_bytes(synthetic_got_safetensors()).expect("synthetic GOT parse");
        let got = crate::native_engine::model_arch::arch_by_id("got-ocr2").unwrap();
        let blob =
            safetensors_to_focrq(&w, ConvertQuant::Int8, 0, [3u8; 32], got).expect("got convert");
        // the bytes physically declare the arch.
        assert!(String::from_utf8_lossy(&blob).contains("\"model_id\":\"got-ocr2\""));

        let out = Weights::from_bytes(blob).expect("got .focrq loads");
        assert_eq!(out.model_id(), "got-ocr2");
        // tied lm_head is OMITTED; embed_tokens carries it (high-precision).
        assert!(
            out.tensor("lm_head.weight").is_err(),
            "lm_head must be omitted"
        );
        assert_eq!(
            out.tensor("model.embed_tokens.weight").unwrap().dtype,
            DType::BF16
        );
        // the decoder GEMMs are int8 …
        assert!(out.qint8("model.layers.0.self_attn.q_proj.weight").is_ok());
        assert!(out.qint8("model.layers.0.mlp.gate_proj.weight").is_ok());
        assert!(out.qint8("model.layers.0.mlp.down_proj.weight").is_ok());
        // … while the qkv bias, norms, connector, and vision stay high-precision.
        assert_eq!(
            out.tensor("model.layers.0.self_attn.q_proj.bias")
                .unwrap()
                .dtype,
            DType::BF16
        );
        assert_eq!(out.tensor("model.norm.weight").unwrap().dtype, DType::BF16);
        assert_eq!(
            out.tensor("model.mm_projector_vary.weight").unwrap().dtype,
            DType::BF16
        );
        assert_eq!(
            out.tensor("model.vision_tower_high.blocks.0.attn.proj.weight")
                .unwrap()
                .dtype,
            DType::BF16
        );
    }

    // ── C2: arch-aware SmolVLM2-500M convert (bd-3jo6.3.2) ───────────────────

    /// A SmolVLM2-shaped synthetic checkpoint (census names, docs/zoo/
    /// smolvlm2-spec.md §12): an UNTIED `lm_head` (≠ `embed_tokens` bytes), the
    /// Idefics3-nested SmolLM2 decoder GEMMs with GQA-shaped k/v panels
    /// (narrower than hidden — the real ones are [320,960] vs [960,960]), the
    /// SigLIP tower (whose blocks contain look-alike `self_attn.q_proj` names),
    /// the pixel-shuffle connector, and all the norms.
    fn synthetic_smolvlm2_safetensors() -> Vec<u8> {
        let ramp = |n: usize, k: usize, bias: f32| -> Vec<f32> {
            (0..n * k).map(|i| (i as f32) * 0.25 - bias).collect()
        };
        build_safetensors(&[
            // UNTIED: lm_head and embed_tokens carry DIFFERENT bytes (spec §12).
            ("lm_head.weight", vec![6, 8], ramp(6, 8, 11.0)),
            (
                "model.text_model.embed_tokens.weight",
                vec![6, 8],
                ramp(6, 8, 3.0),
            ),
            // decoder int8 set (7 GEMMs; k/v are the GQA panels).
            (
                "model.text_model.layers.0.self_attn.q_proj.weight",
                vec![8, 8],
                ramp(8, 8, 7.0),
            ),
            (
                "model.text_model.layers.0.self_attn.k_proj.weight",
                vec![4, 8],
                ramp(4, 8, 6.0),
            ),
            (
                "model.text_model.layers.0.self_attn.v_proj.weight",
                vec![4, 8],
                ramp(4, 8, 5.0),
            ),
            (
                "model.text_model.layers.0.self_attn.o_proj.weight",
                vec![8, 8],
                ramp(8, 8, 8.0),
            ),
            (
                "model.text_model.layers.0.mlp.gate_proj.weight",
                vec![10, 8],
                ramp(10, 8, 9.0),
            ),
            (
                "model.text_model.layers.0.mlp.up_proj.weight",
                vec![10, 8],
                ramp(10, 8, 2.0),
            ),
            (
                "model.text_model.layers.0.mlp.down_proj.weight",
                vec![8, 10],
                ramp(8, 10, 4.0),
            ),
            // decoder norms — high-precision.
            (
                "model.text_model.layers.0.input_layernorm.weight",
                vec![8],
                ramp(1, 8, 1.0),
            ),
            (
                "model.text_model.layers.0.post_attention_layernorm.weight",
                vec![8],
                ramp(1, 8, 1.5),
            ),
            ("model.text_model.norm.weight", vec![8], ramp(1, 8, 2.0)),
            // SigLIP tower — the arch-aware discriminator: this block's
            // `self_attn.q_proj.weight` leaf name matches a decoder GEMM's, but
            // it is NOT under `model.text_model.layers.` so it stays HP.
            (
                "model.vision_model.encoder.layers.0.self_attn.q_proj.weight",
                vec![8, 8],
                ramp(8, 8, 2.5),
            ),
            (
                "model.vision_model.encoder.layers.0.self_attn.q_proj.bias",
                vec![8],
                ramp(1, 8, 0.5),
            ),
            (
                "model.vision_model.encoder.layers.0.mlp.fc1.weight",
                vec![12, 8],
                ramp(12, 8, 1.0),
            ),
            (
                "model.vision_model.embeddings.patch_embedding.weight",
                vec![8, 3, 2, 2],
                ramp(8, 12, 1.0),
            ),
            (
                "model.vision_model.post_layernorm.weight",
                vec![8],
                ramp(1, 8, 0.25),
            ),
            // connector — one high-precision GEMM (K=12288 in the real model).
            (
                "model.connector.modality_projection.proj.weight",
                vec![8, 16],
                ramp(8, 16, 1.0),
            ),
        ])
    }

    /// The SmolVLM2 decoder int8 set of the synthetic checkpoint (7 GEMMs).
    const SMOLVLM2_INT8_NAMES: &[&str] = &[
        "model.text_model.layers.0.self_attn.q_proj.weight",
        "model.text_model.layers.0.self_attn.k_proj.weight",
        "model.text_model.layers.0.self_attn.v_proj.weight",
        "model.text_model.layers.0.self_attn.o_proj.weight",
        "model.text_model.layers.0.mlp.gate_proj.weight",
        "model.text_model.layers.0.mlp.up_proj.weight",
        "model.text_model.layers.0.mlp.down_proj.weight",
    ];

    /// Everything else in the synthetic checkpoint stays high-precision —
    /// INCLUDING the untied `lm_head` (the SmolVLM2 delta vs both GOT and the
    /// default arch).
    const SMOLVLM2_KEPT_NAMES: &[&str] = &[
        "lm_head.weight",
        "model.text_model.embed_tokens.weight",
        "model.text_model.layers.0.input_layernorm.weight",
        "model.text_model.layers.0.post_attention_layernorm.weight",
        "model.text_model.norm.weight",
        "model.vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "model.vision_model.encoder.layers.0.self_attn.q_proj.bias",
        "model.vision_model.encoder.layers.0.mlp.fc1.weight",
        "model.vision_model.embeddings.patch_embedding.weight",
        "model.vision_model.post_layernorm.weight",
        "model.connector.modality_projection.proj.weight",
    ];

    #[test]
    fn smolvlm2_classifier_is_arch_aware_not_name_coincidence() {
        let smol = crate::native_engine::model_arch::arch_by_id("smolvlm2").unwrap();
        let default = crate::native_engine::model_arch::default_arch();
        for name in SMOLVLM2_INT8_NAMES {
            assert!(
                is_decoder_int8_tensor_for(name, smol),
                "{name} must be int8 under smolvlm2"
            );
            // …and the SAME names are NOT int8 under the default arch (whose
            // decoder lives at `model.layers.`) — arch-aware, both directions.
            assert!(
                !is_decoder_int8_tensor_for(name, default),
                "{name} must not be int8 under the default arch"
            );
        }
        for name in SMOLVLM2_KEPT_NAMES {
            assert!(
                !is_decoder_int8_tensor_for(name, smol),
                "{name} must stay high-precision under smolvlm2"
            );
        }
        // Both architectures keep the untied lm_head high precision by default.
        assert!(!is_decoder_int8_tensor_for("lm_head.weight", smol));
        assert!(!is_decoder_int8_tensor_for("lm_head.weight", default));
        // A default-namespace decoder GEMM is NOT smolvlm2's decoder.
        assert!(!is_decoder_int8_tensor_for(
            "model.layers.0.self_attn.q_proj.weight",
            smol
        ));
    }

    #[test]
    fn smolvlm2_convert_keeps_untied_lm_head_and_tags_arch() {
        let w = Weights::from_bytes(synthetic_smolvlm2_safetensors())
            .expect("synthetic SmolVLM2 parse");
        let smol = crate::native_engine::model_arch::arch_by_id("smolvlm2").unwrap();
        let blob = safetensors_to_focrq(&w, ConvertQuant::Int8, 0, [5u8; 32], smol)
            .expect("smolvlm2 convert");
        // the bytes physically declare the arch.
        assert!(String::from_utf8_lossy(&blob).contains("\"model_id\":\"smolvlm2\""));

        let out = Weights::from_bytes(blob).expect("smolvlm2 .focrq loads");
        assert_eq!(out.model_id(), "smolvlm2");
        assert_eq!(out.license_notice(), smol.license_notice());
        // NOTHING is omitted: the untied head means every source tensor survives.
        assert_eq!(out.len(), w.len());
        // The UNTIED lm_head is KEPT — stored, high-precision, bytes verbatim
        // (the opposite of GOT's omit AND of the default arch's int8 head).
        let head = out.tensor("lm_head.weight").expect("lm_head stored");
        assert_eq!(head.dtype, DType::BF16, "lm_head stays high-precision");
        assert_eq!(head.data, w.tensor("lm_head.weight").unwrap().data);
        assert!(
            out.qint8("lm_head.weight").is_err(),
            "lm_head must NOT be int8 (doctrine #2 / spec §11)"
        );
        // embed_tokens is stored high-precision alongside it (dual-matrix).
        assert_eq!(
            out.tensor("model.text_model.embed_tokens.weight")
                .unwrap()
                .dtype,
            DType::BF16
        );
        // The 7 decoder GEMMs are int8, byte-identical to the load-time quant —
        // including the GQA-shaped k/v panels.
        for name in SMOLVLM2_INT8_NAMES {
            let rec = w.record(name).expect("record");
            let (n, k) = (rec.shape[0], rec.shape[1]);
            let expected = nn::quantize_int8(&w.mat(name).unwrap().data, n, k);
            let got = out.qint8(name).expect("qint8 readback");
            assert_eq!((got.n, got.k), (n, k), "{name} [n,k]");
            assert_eq!(got.w, expected.w, "{name} int8 payload bit-identical");
            assert_eq!(got.scales, expected.scales, "{name} scales bit-identical");
        }
        // The SigLIP tower (incl. the look-alike q_proj), connector, and norms
        // all stay high-precision verbatim.
        for name in SMOLVLM2_KEPT_NAMES {
            let before = w.tensor(name).expect("src view");
            let after = out.tensor(name).expect("out view");
            assert_eq!(after.dtype, DType::BF16, "{name} dtype preserved");
            assert_eq!(after.shape, before.shape, "{name} shape preserved");
            assert_eq!(after.data, before.data, "{name} raw bytes verbatim");
        }
    }

    #[test]
    fn smolvlm2_convert_rejects_a_tied_checkpoint() {
        // A checkpoint whose lm_head bytes EQUAL embed_tokens, mislabeled as
        // smolvlm2 (which is censused UNTIED): the convert-time re-verification
        // (spec §12) must refuse rather than ship a silent duplicate.
        let ramp = |n: usize, k: usize, bias: f32| -> Vec<f32> {
            (0..n * k).map(|i| (i as f32) * 0.25 - bias).collect()
        };
        let tied = ramp(6, 8, 11.0);
        let blob = build_safetensors(&[
            ("lm_head.weight", vec![6, 8], tied.clone()),
            ("model.text_model.embed_tokens.weight", vec![6, 8], tied),
            (
                "model.text_model.layers.0.self_attn.q_proj.weight",
                vec![8, 8],
                ramp(8, 8, 7.0),
            ),
        ]);
        let w = Weights::from_bytes(blob).expect("tied synthetic parse");
        let smol = crate::native_engine::model_arch::arch_by_id("smolvlm2").unwrap();
        let err = safetensors_to_focrq(&w, ConvertQuant::Int8, 0, [5u8; 32], smol)
            .expect_err("tied bytes must be refused for an untied arch");
        assert!(matches!(err, FocrError::FormatMismatch(_)), "got {err:?}");
    }

    // ── D2: arch-aware OneChart convert (bd-3jo6.4.2) ────────────────────────

    /// A OneChart-shaped synthetic checkpoint (census names,
    /// docs/zoo/onechart-spec.md §13): a TIED head (`lm_head.weight` byte-equal
    /// to `model.decoder.embed_tokens.weight` — the source stores both), the
    /// OPT decoder GEMMs (`out_proj`, bare `fc1`/`fc2` — NOT the Qwen names),
    /// all-biased linears, per-layer + model-level LayerNorms (the naming
    /// hazard: the per-layer pre-MLP norm is also called `final_layer_norm`),
    /// the learned `embed_positions`, the SAM tower under `model.vision_tower.`,
    /// the `mm_projector`, and the novel `num_decoder` number head.
    fn synthetic_onechart_safetensors() -> Vec<u8> {
        let ramp = |n: usize, k: usize, bias: f32| -> Vec<f32> {
            (0..n * k).map(|i| (i as f32) * 0.25 - bias).collect()
        };
        let tied = ramp(6, 8, 4.0);
        build_safetensors(&[
            // TIED: both stored, byte-identical (census §4 SHA-proof).
            ("lm_head.weight", vec![6, 8], tied.clone()),
            ("model.decoder.embed_tokens.weight", vec![6, 8], tied),
            (
                "model.decoder.embed_positions.weight",
                vec![10, 8],
                ramp(10, 8, 1.0),
            ),
            // decoder int8 set (6 OPT GEMMs).
            (
                "model.decoder.layers.0.self_attn.q_proj.weight",
                vec![8, 8],
                ramp(8, 8, 7.0),
            ),
            (
                "model.decoder.layers.0.self_attn.k_proj.weight",
                vec![8, 8],
                ramp(8, 8, 6.0),
            ),
            (
                "model.decoder.layers.0.self_attn.v_proj.weight",
                vec![8, 8],
                ramp(8, 8, 5.0),
            ),
            (
                "model.decoder.layers.0.self_attn.out_proj.weight",
                vec![8, 8],
                ramp(8, 8, 8.0),
            ),
            (
                "model.decoder.layers.0.fc1.weight",
                vec![16, 8],
                ramp(16, 8, 9.0),
            ),
            (
                "model.decoder.layers.0.fc2.weight",
                vec![8, 16],
                ramp(8, 16, 2.0),
            ),
            // biases + norms stay HP (enable_bias=true: EVERY linear has one).
            (
                "model.decoder.layers.0.self_attn.q_proj.bias",
                vec![8],
                ramp(1, 8, 0.1),
            ),
            (
                "model.decoder.layers.0.self_attn.out_proj.bias",
                vec![8],
                ramp(1, 8, 0.2),
            ),
            (
                "model.decoder.layers.0.fc1.bias",
                vec![16],
                ramp(1, 16, 0.3),
            ),
            ("model.decoder.layers.0.fc2.bias", vec![8], ramp(1, 8, 0.4)),
            (
                "model.decoder.layers.0.self_attn_layer_norm.weight",
                vec![8],
                ramp(1, 8, 0.5),
            ),
            (
                "model.decoder.layers.0.final_layer_norm.weight",
                vec![8],
                ramp(1, 8, 0.6),
            ),
            (
                "model.decoder.final_layer_norm.weight",
                vec![8],
                ramp(1, 8, 0.7),
            ),
            // connector + number head + SAM tower: HP.
            ("model.mm_projector.weight", vec![8, 4], ramp(8, 4, 1.5)),
            ("model.mm_projector.bias", vec![8], ramp(1, 8, 1.6)),
            ("num_decoder.0.weight", vec![4, 8], ramp(4, 8, 1.7)),
            ("num_decoder.0.bias", vec![4], ramp(1, 4, 1.8)),
            (
                "model.vision_tower.blocks.0.attn.qkv.weight",
                vec![24, 8],
                ramp(24, 8, 1.9),
            ),
        ])
    }

    #[test]
    fn onechart_classifier_matches_opt_names_only() {
        let one = crate::native_engine::model_arch::arch_by_id("onechart").unwrap();
        // The OPT GEMMs match…
        for name in [
            "model.decoder.layers.0.self_attn.q_proj.weight",
            "model.decoder.layers.11.self_attn.out_proj.weight",
            "model.decoder.layers.3.fc1.weight",
            "model.decoder.layers.3.fc2.weight",
        ] {
            assert!(is_decoder_int8_tensor_for(name, one), "{name}");
        }
        // …and biases, norms, positions, projector, number head, vision, and
        // QWEN-shaped names do NOT.
        for name in [
            "model.decoder.layers.0.self_attn.q_proj.bias",
            "model.decoder.layers.0.fc1.bias",
            "model.decoder.layers.0.self_attn_layer_norm.weight",
            "model.decoder.layers.0.final_layer_norm.weight",
            "model.decoder.final_layer_norm.weight",
            "model.decoder.embed_tokens.weight",
            "model.decoder.embed_positions.weight",
            "model.mm_projector.weight",
            "num_decoder.0.weight",
            "model.vision_tower.blocks.0.attn.qkv.weight",
            "model.layers.0.mlp.gate_proj.weight", // Qwen name, wrong prefix
            "model.decoder.layers.0.mlp.gate_proj.weight", // Qwen suffix, OPT arch
        ] {
            assert!(!is_decoder_int8_tensor_for(name, one), "{name}");
        }
        // lm_head stays high-precision for the tied OneChart head.
        assert!(!is_decoder_int8_tensor_for("lm_head.weight", one));
    }

    #[test]
    fn onechart_convert_dedups_tied_head_and_tags_arch() {
        let w = Weights::from_bytes(synthetic_onechart_safetensors())
            .expect("synthetic OneChart parse");
        let one = crate::native_engine::model_arch::arch_by_id("onechart").unwrap();
        let blob = safetensors_to_focrq(&w, ConvertQuant::Int8, 0, [7u8; 32], one)
            .expect("onechart convert");
        assert!(String::from_utf8_lossy(&blob).contains("\"model_id\":\"onechart\""));

        let out = Weights::from_bytes(blob).expect("onechart .focrq loads");
        assert_eq!(out.model_id(), "onechart");
        assert_eq!(out.license_notice(), one.license_notice());
        // TIED: lm_head is byte-verified equal then OMITTED (the GOT
        // precedent) — one copy survives as embed_tokens.
        assert_eq!(out.len(), w.len() - 1);
        assert!(out.tensor("lm_head.weight").is_err(), "tied head dropped");
        assert_eq!(
            out.tensor("model.decoder.embed_tokens.weight")
                .unwrap()
                .dtype,
            DType::BF16
        );
        // The 6 OPT GEMMs are int8, byte-identical to the load-time quant.
        for name in [
            "model.decoder.layers.0.self_attn.q_proj.weight",
            "model.decoder.layers.0.self_attn.k_proj.weight",
            "model.decoder.layers.0.self_attn.v_proj.weight",
            "model.decoder.layers.0.self_attn.out_proj.weight",
            "model.decoder.layers.0.fc1.weight",
            "model.decoder.layers.0.fc2.weight",
        ] {
            let q = out
                .qint8(name)
                .unwrap_or_else(|e| unreachable!("{name}: {e}"));
            let src = w.mat(name).unwrap();
            let expect = crate::native_engine::nn::quantize_int8(&src.data, src.rows, src.cols);
            assert_eq!(q.w, expect.w, "{name} int8 bytes");
            assert_eq!(q.scales, expect.scales, "{name} scales");
        }
        // Everything else is high-precision, bytes verbatim.
        for name in [
            "model.decoder.layers.0.self_attn.q_proj.bias",
            "model.decoder.layers.0.self_attn_layer_norm.weight",
            "model.decoder.layers.0.final_layer_norm.weight",
            "model.decoder.embed_positions.weight",
            "model.mm_projector.weight",
            "num_decoder.0.weight",
            "model.vision_tower.blocks.0.attn.qkv.weight",
        ] {
            let t = out
                .tensor(name)
                .unwrap_or_else(|e| unreachable!("{name}: {e}"));
            assert_eq!(t.dtype, DType::BF16, "{name} stays HP");
        }
    }

    #[test]
    fn onechart_convert_rejects_an_untied_checkpoint() {
        // Mutate lm_head so it no longer byte-matches embed_tokens: the tied
        // arch must refuse rather than silently dropping a REAL head.
        let mut tensors = synthetic_onechart_safetensors();
        // Rebuild with a different lm_head ramp instead of byte surgery.
        let _ = &mut tensors;
        let ramp = |n: usize, k: usize, bias: f32| -> Vec<f32> {
            (0..n * k).map(|i| (i as f32) * 0.25 - bias).collect()
        };
        let blob = build_safetensors(&[
            ("lm_head.weight", vec![6, 8], ramp(6, 8, 11.0)),
            (
                "model.decoder.embed_tokens.weight",
                vec![6, 8],
                ramp(6, 8, 4.0),
            ),
            (
                "model.decoder.layers.0.self_attn.q_proj.weight",
                vec![8, 8],
                ramp(8, 8, 7.0),
            ),
        ]);
        let w = Weights::from_bytes(blob).expect("untied synthetic parse");
        let one = crate::native_engine::model_arch::arch_by_id("onechart").unwrap();
        let err = safetensors_to_focrq(&w, ConvertQuant::Int8, 0, [7u8; 32], one)
            .expect_err("untied bytes must be refused for a tied arch");
        assert!(matches!(err, FocrError::FormatMismatch(_)), "got {err:?}");
    }

    #[test]
    fn unlimited_real_recipe_artifact_has_exact_dtype_census() {
        let Some(path) =
            std::env::var_os("FOCR_UNLIMITED_RECIPE_ARTIFACT").map(std::path::PathBuf::from)
        else {
            eprintln!("[convert-test] skip_no_model: FOCR_UNLIMITED_RECIPE_ARTIFACT unset");
            return;
        };
        let weights = Weights::load(&path).expect("recipe artifact loads through mmap");
        assert_eq!(weights.model_id(), "unlimited-ocr");
        assert_eq!(weights.len(), 2_710, "frozen real-model tensor census");
        assert_eq!(
            weights.source_sha256(),
            "2bc48a7a110061ea58fff65d3169367eebe3aee371ca6968dc2219c1b2855fc6"
        );

        let recipe = Recipe::validated_default();
        let mut int8 = 0usize;
        let mut gated_high_precision = 0usize;
        for name in weights.names() {
            let dtype = weights.record(name).expect("census name has record").dtype;
            if recipe.is_quantized(name) {
                assert_eq!(dtype, DType::QInt8PerChan, "{name}");
                int8 += 1;
            } else {
                assert!(
                    matches!(dtype, DType::BF16 | DType::F32),
                    "{name}: {dtype:?}"
                );
                if matches!(
                    recipe.classify(name).policy,
                    crate::quant::recipe::QuantPolicy::Gated(_)
                ) {
                    gated_high_precision += 1;
                }
            }
        }
        assert_eq!(int8, 2_148, "dense + routed/shared FFN matrices");
        assert_eq!(
            gated_high_precision, 49,
            "48 q/k/v/o projections plus lm_head"
        );
    }

    #[test]
    fn tromr_real_artifact_roundtrips_byte_exact() {
        // E2 byte-parity proof on the REAL export (model-gated skip-with-SUCCESS):
        // every high-precision tensor in tromr.focrq must be byte-identical to
        // the WS-folded safetensors, and any int8 record must be EXACTLY one of
        // the 40 measured decoder-GEMM candidates (bd-av64.12) with a 2-D shape
        // carried over and per-channel scales present. Accepts both artifacts:
        // the published f32 (0 int8) and an int8 convert (40 int8).
        let Some(dir) = std::env::var_os("FOCR_TROMR_DIR").map(std::path::PathBuf::from) else {
            eprintln!("[convert-test] skip_no_model: FOCR_TROMR_DIR unset (E2 real-artifact leg)");
            return;
        };
        let (st_path, fq_path) = (dir.join("model.safetensors"), dir.join("tromr.focrq"));
        if !st_path.is_file() || !fq_path.is_file() {
            eprintln!("[convert-test] skip_no_model: export/artifact absent under {dir:?}");
            return;
        }
        let st = Weights::load(&st_path).expect("safetensors loads");
        let fq = Weights::load(&fq_path).expect("focrq loads");
        assert_eq!(fq.model_id(), "tromr", "v2 header self-declares the arch");
        let st_names: Vec<String> = st.names().map(str::to_owned).collect();
        let fq_names: Vec<String> = fq.names().map(str::to_owned).collect();
        assert_eq!(
            st_names, fq_names,
            "same tensor directory (nothing dropped/added)"
        );
        assert_eq!(st_names.len(), 260, "census §12 minus note_mask");
        let tromr = crate::native_engine::model_arch::arch_by_id("tromr").unwrap();
        let mut int8 = 0usize;
        for name in &st_names {
            let a = st.tensor(name).expect("source tensor");
            let b = fq.tensor(name).expect("converted tensor");
            assert_eq!(a.shape, b.shape, "{name}: shape");
            if b.dtype == crate::native_engine::weights::DType::QInt8PerChan {
                assert!(
                    is_decoder_int8_tensor_for(name, tromr),
                    "{name}: int8 record outside the candidate set"
                );
                assert!(!b.scales.is_empty(), "{name}: int8 needs per-chan scales");
                int8 += 1;
            } else {
                assert_eq!(a.dtype, b.dtype, "{name}: HP dtype must carry over");
                assert_eq!(a.data, b.data, "{name}: HP bytes must round-trip exactly");
                assert!(
                    b.scales.is_empty(),
                    "{name}: no quant scales on an HP tensor"
                );
            }
        }
        assert!(
            int8 == 0 || int8 == 40,
            "int8 census must be the f32 artifact (0) or the full candidate set (40), got {int8}"
        );
        eprintln!(
            "[convert-test] tromr round-trip PROVEN: {} tensors ({} int8, rest byte-exact)",
            st_names.len(),
            int8
        );
    }

    #[test]
    fn tromr_classifier_marks_exactly_the_candidate_gemm_suffixes() {
        // bd-av64.12 (supersedes the pre-measurement all-HP default of
        // bd-3jo6.5.2): the 40 decoder GEMMs (`to_{q,k,v}`/`to_out.0` per attn
        // sublayer, `net.0.proj`/`net.3` per ff) classify int8 — measured
        // lossless on the truth-tier corpus (golden byte-identical, gate
        // delta 0). Everything else (embeddings, norms, heads, the WHOLE
        // encoder) stays high-precision.
        let tromr = crate::native_engine::model_arch::arch_by_id("tromr").unwrap();
        for name in [
            "decoder.net.attn_layers.layers.0.1.to_q.weight",
            "decoder.net.attn_layers.layers.0.1.to_k.weight",
            "decoder.net.attn_layers.layers.0.1.to_v.weight",
            "decoder.net.attn_layers.layers.1.1.to_out.0.weight",
            "decoder.net.attn_layers.layers.2.1.net.0.proj.weight",
            "decoder.net.attn_layers.layers.2.1.net.3.weight",
        ] {
            assert!(is_decoder_int8_tensor_for(name, tromr), "{name} is int8");
        }
        for name in [
            // embeddings / norms / heads / encoder stay HP
            "decoder.net.rhythm_emb.emb.weight",
            "decoder.net.attn_layers.layers.0.0.0.weight",
            "decoder.net.to_logits_rhythm.weight",
            "encoder.patch_embed.backbone.stem.conv.weight",
            "encoder.blocks.0.attn.qkv.weight",
            // a Qwen-shaped name under the tromr prefix must ALSO stay HP
            // (the explicit Seq2SeqDense branch, not suffix fallthrough)
            "decoder.net.attn_layers.layers.0.self_attn.q_proj.weight",
            // biases NEVER quantize, even on candidate sublayers
            "decoder.net.attn_layers.layers.2.1.net.0.proj.bias",
        ] {
            assert!(!is_decoder_int8_tensor_for(name, tromr), "{name} stays HP");
        }
    }

    #[test]
    fn default_and_got_classification_use_their_declared_recipes() {
        let got = crate::native_engine::model_arch::arch_by_id("got-ocr2").unwrap();
        let default = crate::native_engine::model_arch::default_arch();
        for name in INT8_NAMES {
            assert!(is_decoder_int8_tensor_for(name, default), "{name}");
            assert!(is_decoder_int8_tensor_for(name, got), "{name}");
            assert!(is_decoder_int8_tensor(name), "{name}");
        }
        for name in [
            "lm_head.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
        ] {
            assert!(!is_decoder_int8_tensor_for(name, default), "{name}");
            assert!(is_decoder_int8_tensor_for(name, got), "{name}");
        }
    }

    #[test]
    fn default_arch_convert_stamps_recipe_and_keeps_gated_tensors_high_precision() {
        let w = Weights::from_bytes(synthetic_safetensors()).expect("synthetic parse");
        let blob = safetensors_to_focrq(
            &w,
            ConvertQuant::Int8,
            2,
            [7u8; 32],
            crate::native_engine::model_arch::default_arch(),
        )
        .expect("default convert");
        assert!(
            !String::from_utf8_lossy(&blob).contains("model_id"),
            "default arch must omit the redundant model_id key"
        );
        assert!(
            String::from_utf8_lossy(&blob).contains(UNLIMITED_OCR_INT8_RECIPE_ID),
            "artifact must self-describe the conservative recipe"
        );
        let out = Weights::from_bytes(blob).expect("loads");
        assert_eq!(out.model_id(), "unlimited-ocr");
        assert_eq!(out.tensor("lm_head.weight").unwrap().dtype, DType::BF16);
        assert_eq!(
            out.tensor("model.layers.0.self_attn.q_proj.weight")
                .unwrap()
                .dtype,
            DType::BF16
        );
        assert_eq!(out.license_notice(), FOCR_MODEL_LICENSE_NOTICE);
    }
}
