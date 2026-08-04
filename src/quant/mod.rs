//! The `focr convert` quantization core (Phase 2/4) — the `.focrq` **writer**
//! plus the int8/int4 quantizers that feed it.
//!
//! This is the offline side of the weight pipeline; the runtime *reader* lives
//! in [`crate::native_engine::weights`]. The two halves cohere by a single
//! contract: every byte this writer emits, that reader parses. The round-trip
//! tests in [`focrq`] write tiny containers and read them back through
//! [`crate::native_engine::weights::Weights::from_bytes`] to prove byte-exact
//! agreement (`docs/focrq-format.md` "Writer Determinism").
//!
//! ## Submodules
//!
//! * [`focrq`] — the `.focrq` container *writer*: a [`focrq::FocrqBuilder`] that
//!   accumulates named tensors (high-precision BF16/F32 or quantized int8/int4,
//!   each with inline scales) and serializes the exact preamble + header-JSON +
//!   payload layout the committed reader consumes.
//! * [`int8`] — symmetric per-output-channel int8 weight quantization
//!   (`scale[o] = max|w_row|/127`, zero-point 0) in OUTPUT-CHANNEL-major `[N, K]`
//!   layout, plus the U8S8 dynamic activation-quant helper (asymmetric, with a
//!   zero-point). i32-accumulation overflow is safe **by construction** — proven
//!   for the global worst case K=6848 (`tests/int32_overflow_proof.rs`,
//!   AGENTS.md doctrine #6).
//! * [`int4`] — group-quantized int4 packing (two signed nibbles per byte,
//!   per-group scales) and the in-register unpack to the exact int8 values the
//!   int8 GEMM consumes (AGENTS.md doctrine #4 — the int4 *bandwidth* win).
//! * [`convert`] — the `focr convert` driver: enumerate every tensor of a raw
//!   bf16 safetensors [`crate::native_engine::weights::Weights`], int8-quantize
//!   the recipe-approved decoder FFN/expert set with
//!   [`crate::native_engine::nn::quantize_int8`], copy everything else verbatim,
//!   and serialize via [`focrq::FocrqBuilder`].
//! * `recipe` / `bit_allocator` — the per-tensor quant policy + rate-distortion
//!   bit allocator, authored by a sibling agent. Declared here so the module
//!   tree is whole; their contents are owned elsewhere.
//!
//! ## Quant recipe (AGENTS.md doctrine #2, fixed + validated)
//!
//! Only the decoder FFN/expert/dense GEMMs are quantized in the baseline recipe.
//! The entire vision tower, the projector, `embed_tokens`, the MoE router gate,
//! and ALL norms stay BF16/F32. int8 on attention `q/k/v/o` and `lm_head` go
//! beyond the validated set and ride behind measured-CER kill switches. This
//! module provides the *mechanism* (quantize/pack/write); the *policy* (which
//! tensor gets which tier) is `recipe`/`bit_allocator`.

pub mod bit_allocator;
pub mod convert;
pub mod focrq;
pub mod int4;
pub mod int8;
pub mod recipe;
