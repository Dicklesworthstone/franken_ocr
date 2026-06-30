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
//! `DecoderWeightCacheI8::build` quantizes a fixed set of decoder GEMM tensors
//! with `quant_oc(w, out) = nn::quantize_int8(w, out, in)` and leaves everything
//! else high-precision. This converter classifies each tensor with
//! [`is_decoder_int8_tensor`] — the *same* set, derived from that builder — and:
//!
//! * for a decoder int8 tensor: widens the bf16 `[n, k]` weight to f32 and calls
//!   the SAME [`nn::quantize_int8`], emitting a `QInt8PerChan` record whose int8
//!   payload + f32 inline scales are byte-identical to what `build` computes at
//!   load time;
//! * for everything else (the whole SAM+CLIP vision tower, the projector,
//!   `embed_tokens`, the MoE router `mlp.gate.weight`, and ALL norms): copies the
//!   original bf16/f32 bytes verbatim.
//!
//! Because the offline int8 bytes equal the load-time int8 bytes, and the
//! high-precision tensors are unchanged, a converted artifact decodes
//! bit-for-bit like the `FOCR_DECODE_INT8` path on the source safetensors. The
//! runtime closes the loop by reading those pre-quantized records back through
//! [`Weights::qint8`] instead of re-quantizing (see `DecoderWeightCacheI8::build`).
//!
//! NB: this mirrors `DecoderWeightCacheI8::build` (the parity oracle the
//! end-to-end check compares against), which quantizes attention `q/k/v/o` and
//! `lm_head` UNCONDITIONALLY — a superset of the conservative kill-switched
//! [`super::recipe`] policy. The recipe stays the policy authority for the gated
//! runtime; `convert` targets the `build` byte image so the two are identical.

use sha2::{Digest, Sha256};

use super::focrq::{FocrqBuilder, WriteDType};
use crate::error::{FocrError, FocrResult};
use crate::native_engine::model_arch::ModelArch;
use crate::native_engine::nn;
use crate::native_engine::weights::{DType, Weights};

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

/// Whether `name` is one of the decoder GEMM tensors the LOAD-TIME int8 quantizer
/// [`crate::native_engine::decoder::DecoderWeightCacheI8::build`] quantizes.
///
/// A **pure function of the tensor name** (no I/O, no env) so it is deterministic
/// and unit-testable. The set is, exactly as `build` enumerates it:
///
/// * `lm_head.weight`;
/// * per decoder layer `model.layers.{L}.…`:
///   * attention `self_attn.{q,k,v,o}_proj.weight`;
///   * the dense layer-0 SwiGLU and every MoE routed/shared expert
///     `mlp.…{gate,up,down}_proj.weight`.
///
/// Everything else is high-precision and returns `false`: ALL norms
/// (`*_layernorm.weight`, `model.norm.weight`), the MoE router `mlp.gate.weight`
/// (note: `gate`, NOT `gate_proj`), `embed_tokens`, the projector, and the entire
/// SAM+CLIP vision tower (their names do not start with `model.layers.`).
#[must_use]
pub fn is_decoder_int8_tensor(name: &str) -> bool {
    if name == "lm_head.weight" {
        return true;
    }
    let Some(rest) = name.strip_prefix("model.layers.") else {
        return false;
    };
    if rest.contains(".self_attn.") {
        return rest.ends_with(".q_proj.weight")
            || rest.ends_with(".k_proj.weight")
            || rest.ends_with(".v_proj.weight")
            || rest.ends_with(".o_proj.weight");
    }
    if rest.contains(".mlp.") {
        // `.gate_proj`/`.up_proj`/`.down_proj` are the FFN/expert GEMMs (int8);
        // the bare router `.mlp.gate.weight` is excluded — it stays high-precision.
        return rest.ends_with(".gate_proj.weight")
            || rest.ends_with(".up_proj.weight")
            || rest.ends_with(".down_proj.weight");
    }
    false
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

    // When the arch ties `lm_head` to `embed_tokens` (GOT-OCR2: proven byte-identical,
    // spec §12), omit `lm_head.weight` — it is a duplicate the loader reconstructs
    // from the stored embedding. Skips ~155 M params from the artifact.
    let omit_lm_head = arch.tie_word_embeddings();

    // `names()` is already sorted (the directory is a `BTreeMap`); collect so the
    // immutable directory borrow is released before the per-tensor accessors run.
    let names: Vec<String> = weights.names().map(str::to_owned).collect();
    for name in &names {
        if omit_lm_head && name == "lm_head.weight" {
            continue;
        }
        if is_decoder_int8_tensor(name) {
            quantize_decoder_tensor(&mut builder, weights, name)?;
        } else {
            copy_high_precision_tensor(&mut builder, weights, name)?;
        }
    }
    Ok(builder.build())
}

/// Quantize one decoder `[n, k]` weight to per-output-channel symmetric int8 with
/// the SAME [`nn::quantize_int8`] the load-time cache uses, and stage it as a
/// `QInt8PerChan` record (int8 payload + `n` f32 inline scales).
fn quantize_decoder_tensor(
    builder: &mut FocrqBuilder,
    weights: &Weights,
    name: &str,
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
    let weight_bytes: Vec<u8> = q.w.iter().map(|&v| v as u8).collect();
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
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.experts.0.up_proj.weight",
        "model.layers.1.mlp.shared_experts.down_proj.weight",
    ];

    const KEPT_NAMES: &[&str] = &[
        "model.layers.1.mlp.gate.weight",
        "model.layers.0.input_layernorm.weight",
        "model.norm.weight",
        "vision_model.patch_embed.weight",
    ];

    #[test]
    fn classifier_matches_decoder_int8_set() {
        for name in INT8_NAMES {
            assert!(is_decoder_int8_tensor(name), "{name} must be int8");
        }
        for name in KEPT_NAMES {
            assert!(
                !is_decoder_int8_tensor(name),
                "{name} must stay high-precision"
            );
        }
        // The router gate vs the dense FFN gate projection is the subtle split.
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

    #[test]
    fn default_arch_convert_is_unchanged_stores_lm_head_no_model_id() {
        // Back-compat: the Unlimited-OCR (default arch) path stores lm_head as int8
        // and emits NO model_id key — byte-identical to the historical artifact.
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
            "default arch must omit the model_id key (byte-parity with v1)"
        );
        let out = Weights::from_bytes(blob).expect("loads");
        assert_eq!(out.model_id(), "unlimited-ocr");
        assert!(
            out.qint8("lm_head.weight").is_ok(),
            "default keeps lm_head int8"
        );
        assert_eq!(out.license_notice(), FOCR_MODEL_LICENSE_NOTICE);
    }
}
