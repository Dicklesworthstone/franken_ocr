//! Browser boundary for the franken_ocr engine.
//!
//! Design rules (inherited from the frankentts wasm port, the sibling exemplar):
//!
//! * **Errors are values, never panics.** A wasm panic surfaces as an opaque
//!   `RuntimeError: unreachable`; every fallible edge here maps to
//!   `Result<_, JsValue>` with the failing stage named, and a panic hook is
//!   installed at module start so anything that still panics prints a real
//!   message to the console.
//! * **No environment variables, no filesystem.** Model weights and tokenizer
//!   sidecars arrive as bytes through [`ModelStaging`]; runtime introspection
//!   that natively lives in `focr robot backends` is exported as functions
//!   (`int8_route`, `engine_info`) so JS can ASSERT the armed route instead of
//!   trusting build flags.
//! * **Never hold a big payload twice.** [`ModelStaging::reserve`] uses
//!   `try_reserve_exact` and chunked `push`, so the weight blob exists exactly
//!   once inside wasm memory (the JS side streams fetch chunks straight in and
//!   drops them).
//! * **One code path.** This crate adds no numerics: it decodes the uploaded
//!   image with the same `image` crate the CLI uses and calls the same
//!   `OcrModel` entrypoints. On wasm32 the SIMD dispatcher lands on the scalar
//!   tier (bit-identical to every accelerated tier by the selftest contract).

use std::sync::Arc;

use franken_ocr::native_engine::weights::Weights;
use franken_ocr::native_engine::{OcrModel, SidecarBundle};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(msg: &str);
}

/// Install the panic hook. `#[wasm_bindgen(start)]` runs this once at module
/// instantiation, before any exported call.
#[wasm_bindgen(start)]
pub fn focr_wasm_start() {
    std::panic::set_hook(Box::new(|info| {
        console_error(&format!("focr-wasm panic: {info}"));
    }));
}

fn js_err(stage: &str, detail: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&format!("{stage}: {detail}"))
}

/// The effective ordinary dense-int8 route on this host. On wasm32 today this
/// reports the scalar tier; when a simd128 kernel island lands it will report
/// that route, and the site asserts whichever it expects — a silent scalar
/// fallback is invisible any other way.
#[wasm_bindgen]
#[must_use]
pub fn int8_route() -> String {
    format!("{:?}", franken_ocr::simd::effective_dense_route())
}

/// One JSON object describing this module: crate version, detected ISA tier,
/// and the licenses that must travel with the model weights.
#[wasm_bindgen]
#[must_use]
pub fn engine_info() -> String {
    serde_json::json!({
        "crate_version": env!("CARGO_PKG_VERSION"),
        "detected_tier": franken_ocr::simd::tier_string(),
        "dense_route": format!("{:?}", franken_ocr::simd::effective_dense_route()),
        "project_license": franken_ocr::FOCR_PROJECT_LICENSE_NOTICE,
    })
    .to_string()
}

/// Request cooperative cancellation of the in-flight recognition: the decode
/// loop observes the flag at its next checkpoint and returns a `Cancelled`
/// error. Call [`reset_cancel`] before the next run.
#[wasm_bindgen]
pub fn request_cancel() {
    franken_ocr::request_shutdown();
}

/// Clear the cancellation flag so the next recognition can run.
#[wasm_bindgen]
pub fn reset_cancel() {
    franken_ocr::reset_shutdown();
}

/// Staging area for a model arriving over the network as chunks: exactly one
/// copy of the weight blob lives here, reserved up front, plus the (small)
/// tokenizer sidecars keyed by their canonical zoo filenames.
#[wasm_bindgen]
pub struct ModelStaging {
    weights: Vec<u8>,
    expected: usize,
    sidecars: SidecarBundle,
    /// TrOMR WordLevel tables staged individually until all four are present.
    music: [Option<String>; 4],
}

impl Default for ModelStaging {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl ModelStaging {
    /// An empty staging area. Call [`Self::reserve`] before the first
    /// [`Self::push`].
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> ModelStaging {
        ModelStaging {
            weights: Vec::new(),
            expected: 0,
            sidecars: SidecarBundle::default(),
            music: [None, None, None, None],
        }
    }

    /// Reserve the full weight-blob size up front (`try_reserve_exact`, so a
    /// failed reservation names the byte count instead of trapping on an
    /// overcommitted grow).
    pub fn reserve(&mut self, total_bytes: f64) -> Result<(), JsValue> {
        if !(total_bytes.is_finite() && total_bytes >= 0.0 && total_bytes <= usize::MAX as f64) {
            return Err(js_err("staging.reserve", "total_bytes out of range"));
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let total = total_bytes as usize;
        if !self.weights.is_empty() || self.expected != 0 {
            return Err(js_err(
                "staging.reserve",
                "already reserved; staging is single-use",
            ));
        }
        self.weights.try_reserve_exact(total).map_err(|e| {
            js_err(
                "staging.reserve",
                format!("cannot reserve {total} bytes: {e}"),
            )
        })?;
        self.expected = total;
        Ok(())
    }

    /// Append one downloaded chunk. Refuses bytes past the reserved size —
    /// a mismatched manifest must fail loudly, not grow silently.
    pub fn push(&mut self, chunk: &[u8]) -> Result<(), JsValue> {
        if self.expected == 0 {
            return Err(js_err("staging.push", "reserve() must run first"));
        }
        if self.weights.len() + chunk.len() > self.expected {
            return Err(js_err(
                "staging.push",
                format!(
                    "overflow: {} + {} exceeds reserved {}",
                    self.weights.len(),
                    chunk.len(),
                    self.expected
                ),
            ));
        }
        self.weights.extend_from_slice(chunk);
        Ok(())
    }

    /// Bytes staged so far (for progress display).
    #[must_use]
    pub fn filled(&self) -> f64 {
        self.weights.len() as f64
    }

    /// Attach one tokenizer sidecar by its canonical zoo filename:
    /// `tokenizer.json`, `qwen.tiktoken`, or the four TrOMR tables
    /// `tokenizer_{rhythm,pitch,lift,note}.json`.
    pub fn set_sidecar(&mut self, name: &str, bytes: &[u8]) -> Result<(), JsValue> {
        let as_text = || {
            String::from_utf8(bytes.to_vec())
                .map_err(|e| js_err("staging.set_sidecar", format!("{name}: not UTF-8: {e}")))
        };
        match name {
            "tokenizer.json" => self.sidecars.tokenizer_json = Some(bytes.to_vec()),
            "qwen.tiktoken" => self.sidecars.qwen_tiktoken = Some(bytes.to_vec()),
            "tokenizer_rhythm.json" => self.music[0] = Some(as_text()?),
            "tokenizer_pitch.json" => self.music[1] = Some(as_text()?),
            "tokenizer_lift.json" => self.music[2] = Some(as_text()?),
            "tokenizer_note.json" => self.music[3] = Some(as_text()?),
            other => {
                return Err(js_err(
                    "staging.set_sidecar",
                    format!("unknown sidecar name {other:?}"),
                ));
            }
        }
        Ok(())
    }

    /// Free the staged bytes explicitly (a superseded or failed load should
    /// not wait for the JS GC's FinalizationRegistry).
    pub fn free(self) {
        drop(self);
    }
}

/// A loaded model plus the recognize entrypoints the playground calls.
#[wasm_bindgen]
pub struct WasmEngine {
    model: Arc<OcrModel>,
}

#[wasm_bindgen]
impl WasmEngine {
    /// Build the engine from a completed staging area (consumes it — the
    /// weight bytes move, they are not copied).
    ///
    /// Fails if the staging is incomplete, the blob does not parse as a
    /// `.focrq`/safetensors container, or the Unlimited-OCR recipe validation
    /// rejects the tensor dtypes.
    pub fn from_staging(staging: ModelStaging) -> Result<WasmEngine, JsValue> {
        let ModelStaging {
            weights,
            expected,
            mut sidecars,
            music,
        } = staging;
        if weights.len() != expected {
            return Err(js_err(
                "engine.from_staging",
                format!("staging incomplete: {} of {expected} bytes", weights.len()),
            ));
        }
        if music.iter().all(Option::is_some) {
            let [a, b, c, d] = music;
            sidecars.music_tables = Some([
                a.expect("checked"),
                b.expect("checked"),
                c.expect("checked"),
                d.expect("checked"),
            ]);
        } else if music.iter().any(Option::is_some) {
            return Err(js_err(
                "engine.from_staging",
                "partial TrOMR tokenizer set: all four tables are required",
            ));
        }
        let weights =
            Weights::from_bytes(weights).map_err(|e| js_err("engine.parse_weights", e))?;
        let model = OcrModel::from_weights(weights, sidecars)
            .map_err(|e| js_err("engine.validate_weights", e))?;
        Ok(WasmEngine { model })
    }

    /// The loaded model's registry id (`unlimited-ocr`, `tromr`, …).
    #[must_use]
    pub fn model_id(&self) -> String {
        self.model.arch().id().to_string()
    }

    /// Recognize one encoded image (PNG/JPEG bytes) and return the model's
    /// primary text output: markdown for the OCR models, MusicXML for TrOMR.
    pub fn recognize(&self, image_bytes: &[u8]) -> Result<String, JsValue> {
        let img =
            image::load_from_memory(image_bytes).map_err(|e| js_err("engine.decode_image", e))?;
        self.model
            .recognize_dynamic(img)
            .map_err(|e| js_err("engine.recognize", e))
    }

    /// Recognize one encoded image and return a JSON envelope:
    /// `{"model_id", "output", "layout": [{label, boxes}], "music": {...}?}`.
    /// `layout` mirrors `focr ocr --json`; `music` carries the TrOMR staff
    /// metadata (recognized bboxes, per-staff skips with reasons, annotate-only
    /// warnings) when the run produced any.
    pub fn recognize_json(&self, image_bytes: &[u8]) -> Result<String, JsValue> {
        let img =
            image::load_from_memory(image_bytes).map_err(|e| js_err("engine.decode_image", e))?;
        let doc = self
            .model
            .recognize_dynamic_with_layout(img)
            .map_err(|e| js_err("engine.recognize", e))?;
        let layout: Vec<serde_json::Value> = doc
            .layout
            .iter()
            .map(|span| {
                serde_json::json!({
                    "label": span.label,
                    "boxes": span.boxes,
                })
            })
            .collect();
        let music = self.model.take_music_meta().map(|meta| {
            serde_json::json!({
                "staves": meta
                    .staves
                    .iter()
                    .map(|(index, bbox)| serde_json::json!({
                        "index": index,
                        "bbox": [bbox.0, bbox.1, bbox.2, bbox.3],
                    }))
                    .collect::<Vec<_>>(),
                "skips": meta
                    .skips
                    .iter()
                    .map(|skip| serde_json::json!({
                        "index": skip.index,
                        "bbox": [skip.bbox.0, skip.bbox.1, skip.bbox.2, skip.bbox.3],
                        "reason": skip.reason,
                    }))
                    .collect::<Vec<_>>(),
                "warnings": meta
                    .warnings
                    .iter()
                    .map(|w| serde_json::json!({
                        "kind": w.kind,
                        "part": w.part,
                        "measure": w.measure,
                        "detail": w.detail,
                    }))
                    .collect::<Vec<_>>(),
            })
        });
        let envelope = serde_json::json!({
            "model_id": self.model.arch().id(),
            "output": doc.markdown,
            "layout": layout,
            "music": music,
        });
        Ok(envelope.to_string())
    }

    /// The model-weights license notice that must travel with the artifact.
    #[must_use]
    pub fn license_notice(&self) -> String {
        self.model.arch().license_notice().to_string()
    }

    /// Free the engine (and, if this was the last handle, the weight bytes)
    /// explicitly rather than waiting for the JS GC.
    pub fn free_engine(self) {
        drop(self);
    }
}
