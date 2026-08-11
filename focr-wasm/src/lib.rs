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
//! * **Never hold a big payload twice.** [`ModelStaging::plan`] reserves the
//!   blob's segments up front with `try_reserve_exact` and chunked `push`, so
//!   the weight bytes exist exactly once inside wasm memory (the JS side
//!   streams fetch chunks straight in and drops them).
//! * **No 2 GiB wall.** wasm32 caps ANY single Rust allocation (and any slice)
//!   at `isize::MAX` = 2 GiB, so the 3.0 GB Unlimited artifact cannot live in
//!   one `Vec<u8>` even inside the 4 GiB heap (bd-syf2). Staging is therefore
//!   two-phase: [`ModelStaging::plan`] reads the artifact's own tensor
//!   directory out of a downloaded prefix and cuts the blob into ≤1 GiB
//!   segments **at tensor boundaries only**; `push` then spills across those
//!   boundaries transparently. Every tensor still resolves to one contiguous
//!   `&[u8]`, so no kernel or accessor changes.
//! * **One code path.** This crate adds no numerics: it decodes the uploaded
//!   image with the same `image` crate the CLI uses and calls the same
//!   `OcrModel` entrypoints. On wasm32 the SIMD dispatcher lands on the scalar
//!   tier (bit-identical to every accelerated tier by the selftest contract).

use std::sync::Arc;

use franken_ocr::native_engine::weights::{self, Weights};
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

/// `initThreadPool(n)` — the rayon worker-pool constructor, present ONLY in the
/// threaded module (`site/pkg-threaded`, built with `--features threads`).
///
/// Re-exporting the crate's `#[wasm_bindgen]` function here is what puts it in
/// the generated JS surface. JS must call it exactly once, and — the rule that
/// cost the sibling project days — only AFTER the model is staged and hydrated:
/// a worker parked in `Atomics.wait` blocks a shared-memory `grow`, and staging
/// a multi-GB artifact is nothing but grows.
#[cfg(all(target_arch = "wasm32", feature = "threads"))]
pub use wasm_bindgen_rayon::init_thread_pool;

/// Rayon's *actual* worker count in this module right now.
///
/// This is the honest test of the threaded lane: the build flags, the presence
/// of `initThreadPool`, and even a `SharedArrayBuffer`-backed memory can all be
/// right while the pool silently fell back to one thread. Serial builds report
/// `1` here by construction.
///
/// Calling this initializes rayon's global pool if it is not built yet, so JS
/// must call `initThreadPool` FIRST in the threaded module.
#[wasm_bindgen]
#[must_use]
pub fn thread_count() -> usize {
    rayon::current_num_threads()
}

/// The module's `WebAssembly.Memory`, so JS can assert
/// `wasm_memory().buffer instanceof SharedArrayBuffer` after instantiation.
/// Link flags are a claim; this is the receipt.
#[wasm_bindgen]
#[must_use]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
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

/// Install (or, with `None`/`undefined`, remove) the live progress callback for
/// the SYNCHRONOUS `recognize` call — the seam that lets the playground show
/// "vision block 12/36" instead of a frozen tab for minutes.
///
/// The callback is invoked as `f(stage: string, current: number, total: number)`
/// (`total === 0` means indeterminate) from INSIDE the forward's call stack, on
/// the thread that entered `recognize_json`. That is the only way progress can
/// escape at all: the worker is blocked in one synchronous wasm call, so nothing
/// asynchronous can run until it returns.
///
/// Three rules make this safe and cheap:
///
/// * **The engine hooks sit on outer, sequential loops only** (per vision block,
///   per decoded token, never inside a rayon body), so the `js_sys::Function` —
///   which is emphatically NOT `Send`, whatever the sink signature says — is only
///   ever called from the JS-owning worker thread. The `Send`/`Sync` promise
///   below is the wasm single-threaded-JS invariant written down, not a claim
///   that a `Function` can cross a thread.
/// * **A throwing callback cannot poison a run.** The `Result` is discarded: a
///   broken progress handler must never be able to fail a recognition.
/// * **Zero cost when unset.** `set_progress_sink(None)` disarms the engine's
///   relaxed-atomic fast path, and the native build never installs anything.
#[wasm_bindgen]
pub fn set_progress_callback(f: Option<js_sys::Function>) {
    let armed = f.is_some();
    // The `Function` lives in THREAD-LOCAL storage, never in the global sink —
    // so no `unsafe impl Send` is needed anywhere (this crate denies
    // `unsafe_code`), and the invariant is enforced by the type system instead
    // of by a comment: the installed closure captures nothing, and an event
    // raised on any other thread finds an empty slot and does nothing.
    PROGRESS_CALLBACK.with(|slot| {
        if let Ok(mut slot) = slot.try_borrow_mut() {
            *slot = f;
        }
    });
    if !armed {
        franken_ocr::native_engine::progress::set_progress_sink(None);
        return;
    }
    franken_ocr::native_engine::progress::set_progress_sink(Some(Box::new(|event| {
        PROGRESS_CALLBACK.with(|slot| {
            // `try_borrow`: a callback that re-enters this seam is dropped, not
            // a panic (and a wasm panic is an opaque `unreachable`).
            let Ok(slot) = slot.try_borrow() else { return };
            let Some(f) = slot.as_ref() else { return };
            // A failed JS call (a throwing handler) is swallowed on purpose:
            // a broken progress handler must never fail a recognition.
            let _ = f.call3(
                &JsValue::NULL,
                &JsValue::from_str(event.stage),
                &JsValue::from_f64(event.current as f64),
                &JsValue::from_f64(event.total as f64),
            );
        });
    })));
}

thread_local! {
    /// The page's progress handler, owned by the thread that installed it.
    static PROGRESS_CALLBACK: std::cell::RefCell<Option<js_sys::Function>> =
        const { std::cell::RefCell::new(None) };
}

/// Set (or clear, with `0`) the sliding no-repeat n-gram decode guard for the
/// NEXT engine built by [`WasmEngine::from_staging`] — the browser analog of
/// `--no-repeat-ngram` / `FOCR_NO_REPEAT_NGRAM`. The README documents a
/// tighter guard (20 vs the default 35) as the mitigation for degenerate
/// repetition loops on hard dense pages; the measured in-browser failure on
/// such a page (an f32-drift-tipped repeat attractor) is the same class. This
/// is a mitigation the native CLI user can apply identically — not a silent
/// numerics change: the site labels the lane's guard setting.
#[wasm_bindgen]
pub fn set_no_repeat_ngram(n: u32) {
    franken_ocr::native_engine::set_decode_overrides(franken_ocr::native_engine::DecodeOverrides {
        no_repeat_ngram: if n == 0 { None } else { Some(n as usize) },
        ..Default::default()
    });
}

/// Staging area for a model arriving over the network as chunks: exactly one
/// copy of the weight blob lives here, reserved up front, plus the (small)
/// tokenizer sidecars keyed by their canonical zoo filenames.
#[wasm_bindgen]
pub struct ModelStaging {
    /// Planned segments in order: `(absolute start, bytes)`. Each `Vec` is
    /// reserved to its planned length by [`ModelStaging::plan`]; `push` fills
    /// them in sequence.
    segments: Vec<(u64, Vec<u8>)>,
    /// Planned length of each segment (parallel to `segments`).
    segment_lens: Vec<usize>,
    /// Total planned bytes across all segments.
    expected: u64,
    /// Bytes appended so far.
    filled: u64,
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
            segments: Vec::new(),
            segment_lens: Vec::new(),
            expected: 0,
            filled: 0,
            sidecars: SidecarBundle::default(),
            music: [None, None, None, None],
        }
    }

    /// PHASE 1 of staging (bd-syf2): plan the blob's segmentation from a
    /// downloaded PREFIX of the artifact, and reserve every planned segment.
    ///
    /// `header_prefix` is the artifact's leading bytes (4 MiB is ample: the
    /// 3.0 GB Unlimited artifact's header is ~0.5 MB). `total_bytes` is the
    /// artifact's full pinned size. Returns a JSON object:
    ///
    /// * `{"status":"planned","segments":[len,…],"payload_base":N}` — staging is
    ///   armed; start pushing bytes from offset 0.
    /// * `{"status":"need_prefix","need_bytes":N}` — the prefix did not contain
    ///   the whole header; refetch at least `N` leading bytes and call again
    ///   (this converges in at most two probes).
    ///
    /// A small model (TrOMR) plans to exactly one segment and behaves like the
    /// old single-buffer `reserve` did.
    ///
    /// # Errors
    /// A malformed header, an artifact whose layout offers no clean cut, or a
    /// failed reservation (named with its byte count, never an opaque trap).
    pub fn plan(&mut self, header_prefix: &[u8], total_bytes: f64) -> Result<String, JsValue> {
        if !(total_bytes.is_finite() && total_bytes >= 0.0 && total_bytes <= u64::MAX as f64) {
            return Err(js_err("staging.plan", "total_bytes out of range"));
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let total = total_bytes as u64;
        if self.expected != 0 {
            return Err(js_err(
                "staging.plan",
                "already planned; staging is single-use",
            ));
        }
        let plan = weights::focrq_segment_plan(header_prefix, total, weights::MAX_BLOB_SEGMENT)
            .map_err(|e| js_err("staging.plan", e))?;
        let (segment_lens, payload_base) = match plan {
            weights::SegmentPlan::NeedPrefix { need_bytes } => {
                return Ok(serde_json::json!({
                    "status": "need_prefix",
                    "need_bytes": need_bytes,
                })
                .to_string());
            }
            weights::SegmentPlan::Planned {
                segment_lens,
                payload_base,
            } => (segment_lens, payload_base),
        };
        let mut start = 0u64;
        for (index, &len) in segment_lens.iter().enumerate() {
            let len = usize::try_from(len).map_err(|_| {
                js_err(
                    "staging.plan",
                    format!("segment {index} of {len} bytes exceeds this target's usize"),
                )
            })?;
            let mut buf: Vec<u8> = Vec::new();
            buf.try_reserve_exact(len).map_err(|e| {
                js_err(
                    "staging.plan",
                    format!("cannot reserve segment {index} of {len} bytes: {e}"),
                )
            })?;
            self.segments.push((start, buf));
            self.segment_lens.push(len);
            start += len as u64;
        }
        self.expected = start;
        Ok(serde_json::json!({
            "status": "planned",
            "segments": segment_lens,
            "payload_base": payload_base,
        })
        .to_string())
    }

    /// PHASE 2: append one downloaded chunk. The caller streams the artifact
    /// start-to-end and never needs to know where the segment edges are — a
    /// chunk that spans a boundary is split across the two segments here.
    /// Refuses bytes past the planned total: a mismatched manifest must fail
    /// loudly, not grow silently.
    pub fn push(&mut self, chunk: &[u8]) -> Result<(), JsValue> {
        if self.expected == 0 {
            return Err(js_err("staging.push", "plan() must run first"));
        }
        if self.filled + chunk.len() as u64 > self.expected {
            return Err(js_err(
                "staging.push",
                format!(
                    "overflow: {} + {} exceeds planned {}",
                    self.filled,
                    chunk.len(),
                    self.expected
                ),
            ));
        }
        let mut rest = chunk;
        while !rest.is_empty() {
            // The first segment that still has room (segments fill in order).
            let Some(index) = self
                .segments
                .iter()
                .enumerate()
                .find(|(i, (_, bytes))| bytes.len() < self.segment_lens[*i])
                .map(|(i, _)| i)
            else {
                return Err(js_err("staging.push", "no segment has room left"));
            };
            let room = self.segment_lens[index] - self.segments[index].1.len();
            let take = room.min(rest.len());
            self.segments[index].1.extend_from_slice(&rest[..take]);
            self.filled += take as u64;
            rest = &rest[take..];
        }
        Ok(())
    }

    /// Bytes staged so far (for progress display).
    #[must_use]
    pub fn filled(&self) -> f64 {
        self.filled as f64
    }

    /// The planned segment count (1 for every model that fits one buffer).
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
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
            segments,
            segment_lens,
            expected,
            filled,
            mut sidecars,
            music,
        } = staging;
        if expected == 0 {
            return Err(js_err("engine.from_staging", "staging was never planned"));
        }
        if filled != expected {
            return Err(js_err(
                "engine.from_staging",
                format!("staging incomplete: {filled} of {expected} bytes"),
            ));
        }
        for (index, ((_, bytes), planned)) in segments.iter().zip(&segment_lens).enumerate() {
            if bytes.len() != *planned {
                return Err(js_err(
                    "engine.from_staging",
                    format!(
                        "segment {index} holds {} of {planned} planned bytes",
                        bytes.len()
                    ),
                ));
            }
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
        // One segment or several, the loader validates coverage/ordering and
        // that no tensor straddles a boundary before anything reads a weight.
        let weights =
            Weights::from_segments(segments).map_err(|e| js_err("engine.parse_weights", e))?;
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
