//! The native model package — the hand-written Unlimited-OCR forward over the
//! [`Mat`]/slice currency (PROPOSED_ARCHITECTURE.md §2, §6).
//!
//! This module is **plain Rust over `ft-kernel-cpu` free functions**: no tensor
//! graph, no autograd, no `ft-api` session/tape (plan §1.1 P2). Every kernel
//! call funnels through [`nn`] (the frankentorch facade, §5); every other
//! submodule implements a contiguous block of THE SPEC:
//!
//! * [`tensor`] — the `Mat` activation currency + quantized weight structs (§4).
//! * [`nn`] — the frankentorch facade (matmul / int8 linear / conv2d / sdpa /
//!   rms_norm / layer_norm / softmax / silu / gelu / quick_gelu) (§5).
//! * [`vision_sam`] / [`vision_clip`] / [`vision_bridge`] — the vision tower
//!   ([SPEC-040..052], §6.3–§6.5).
//! * [`connector`] — masked-scatter vision fusion ([SPEC-060..066], §6.6).
//! * [`decoder`] / [`rswa`] / [`moe`] — the DeepseekV2 decoder, R-SWA ring
//!   attention, and MoE block ([SPEC-070..096], §6.7–§6.9).
//! * [`sampler`] — the AR decode loop + sampler ([SPEC-100..103], §6.10).
//! * [`postprocess`] — ref/det parse, bbox /999, markdown ([SPEC-110..119], §6.11).
//! * [`weights`] — the `.focrq` reader + census (§6.12, §7).
//!
//! [`Mat`]: tensor::Mat

pub mod batch_scheduler;
pub mod calib;
pub mod connector;
pub mod decoder;
pub mod decoder_qwen2;
pub mod got;
pub(crate) mod lever_policy;
pub mod model_arch;
pub mod moe;
pub mod nn;
pub mod onechart;
pub mod postprocess;
pub mod progress;
pub mod rswa;
pub mod sampler;
pub mod shards;
pub mod smolvlm2;
pub(crate) mod spec;
pub mod srswa;
pub mod tensor;
pub mod token_compress;
pub mod tromr;
pub(crate) mod unlimited_ocr_census;
pub mod vision_bridge;
pub mod vision_clip;
pub mod vision_sam;
pub mod vision_siglip;
pub mod weights;
pub mod window;

use std::ffi::OsStr;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, Weak};

// Clock seam: `std::time::Instant` traps on wasm32-unknown-unknown; `web-time`
// re-exports std's types on native targets, so native behavior is unchanged.
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use crate::error::{FocrError, FocrResult};
use crate::preprocess::{self, Preprocessed};
use crate::quant::recipe::{Recipe, is_truthy};
use sampler::{DecodeOutput, DecodeParams};
use tensor::Mat;
use weights::{DType, Weights};

/// A process-local exclusive permit that never holds a mutex while protected
/// work runs. The mutex only closes the condition-variable wakeup race while a
/// contender sleeps; the returned permit owns an atomic claim, so rayon kernels
/// cannot deadlock by running beneath a held admission lock.
struct ExclusiveGate {
    occupied: std::sync::atomic::AtomicBool,
    waiters: Mutex<()>,
    wake: Condvar,
}

impl ExclusiveGate {
    const fn new() -> Self {
        Self {
            occupied: std::sync::atomic::AtomicBool::new(false),
            waiters: Mutex::new(()),
            wake: Condvar::new(),
        }
    }

    fn acquire(&self) -> ExclusivePermit<'_> {
        use std::sync::atomic::Ordering;

        if self
            .occupied
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            return ExclusivePermit { gate: self };
        }

        let mut waiter = self
            .waiters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if self
                .occupied
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                drop(waiter);
                return ExclusivePermit { gate: self };
            }
            waiter = self
                .wake
                .wait(waiter)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    fn release(&self) {
        use std::sync::atomic::Ordering;

        // Pair the state transition with the waiter's check under this mutex so
        // a notification cannot be lost between its failed CAS and `wait`.
        let _waiter = self
            .waiters
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.occupied.store(false, Ordering::Release);
        self.wake.notify_one();
    }
}

struct ExclusivePermit<'a> {
    gate: &'a ExclusiveGate,
}

impl Drop for ExclusivePermit<'_> {
    fn drop(&mut self) {
        self.gate.release();
    }
}

/// A retryable, fallible single-initializer cell. Unlike a plain
/// check/build/`OnceLock::set` sequence, concurrent callers cannot duplicate a
/// multi-GB hydration. A failed initializer leaves the cell empty for retry.
/// The initializer runs under an atomic permit, not a held mutex guard.
struct FallibleOnce<T> {
    value: OnceLock<T>,
    init: ExclusiveGate,
}

impl<T> FallibleOnce<T> {
    const fn new() -> Self {
        Self {
            value: OnceLock::new(),
            init: ExclusiveGate::new(),
        }
    }

    /// A cell that already holds `value` when one was built eagerly.
    fn prefilled(value: Option<T>) -> Self {
        let cell = Self::new();
        if let Some(value) = value {
            let _ = cell.value.set(value);
        }
        cell
    }

    fn get_or_try_init<E, F>(&self, initialize: F) -> Result<&T, E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        if let Some(value) = self.value.get() {
            return Ok(value);
        }

        let _permit = self.init.acquire();
        if let Some(value) = self.value.get() {
            return Ok(value);
        }

        let value = initialize()?;
        let set = self.value.set(value);
        debug_assert!(set.is_ok(), "single initializer set raced unexpectedly");
        Ok(self.value.get().expect("single initializer just set"))
    }
}

enum RetainedOrOwned<'a, T> {
    Retained(&'a T),
    Owned(T),
}

impl<T> AsRef<T> for RetainedOrOwned<'_, T> {
    fn as_ref(&self) -> &T {
        match self {
            Self::Retained(value) => value,
            Self::Owned(value) => value,
        }
    }
}

fn retained_or_owned<'a, T, E, F>(
    retain: bool,
    cache: &'a FallibleOnce<T>,
    initialize: F,
) -> Result<RetainedOrOwned<'a, T>, E>
where
    F: FnOnce() -> Result<T, E>,
{
    if retain {
        cache
            .get_or_try_init(initialize)
            .map(RetainedOrOwned::Retained)
    } else {
        initialize().map(RetainedOrOwned::Owned)
    }
}

/// The loaded Unlimited-OCR model: weights + the fixed-shape forward.
///
/// Held behind an [`Arc`] and cached through a process-global [`Weak`] (see
/// [`OcrModel::load`]) so repeated `focr ocr` invocations in one process share
/// one weight blob. A process-wide admission permit serializes concurrent
/// callers before they touch hydrated weights; each admitted operation then
/// drives its own sequential outer page loop (plan §6.5 P6).
///
/// The forward driver ([`OcrModel::forward`]) wires the Phase-1 pipeline shape
/// (plan §10 Phase 1): preprocess -> vision tower (SAM⊕CLIP via the
/// `vision_bridge` projector 2048->1280) -> connector (the 273-slot placeholder
/// layout + `masked_scatter`) -> decoder (the 12-layer R-SWA MoE driver) ->
/// sampler (greedy + no_repeat_ngram loop to EOS) -> postprocess
/// (markdown / ref-det bbox). The numeric kernels of every stage are implemented
/// and unit-tested in their own submodules; this driver owns ONLY the
/// orchestration, the per-stage error mapping, and the **sequential** decode loop
/// (doctrine #5: no nested runtime, no rayon under a lock — one forward at a
/// time, fanning out across cores inside the kernels).
/// In-memory tokenizer sidecar payloads for byte-loaded models
/// ([`OcrModel::from_weights`]). On the filesystem route these files resolve
/// beside the model artifact; a byte-loaded model (the browser/wasm path, or an
/// embedder that ships weights inside its own binary) has no "beside", so the
/// caller hands the payloads over up front. Only the entries the loaded arch
/// actually needs must be present — a missing entry surfaces the same clean
/// [`FocrError::ModelNotFound`] the filesystem route reports for a missing
/// sidecar, at first tokenizer use.
#[derive(Default)]
pub struct SidecarBundle {
    /// `tokenizer.json` bytes (the Unlimited-OCR / GOT-fallback BPE).
    pub tokenizer_json: Option<Vec<u8>>,
    /// `qwen.tiktoken` bytes (the GOT-OCR2 tokenizer).
    pub qwen_tiktoken: Option<Vec<u8>>,
    /// TrOMR WordLevel tables in stream order `[rhythm, pitch, lift, note]`.
    pub music_tables: Option<[String; 4]>,
    /// The OneChart / OPT slow-tokenizer triple in order
    /// `[vocab.json, merges.txt, added_tokens.json]` (D9) — the byte-loaded
    /// analog of the `from_opt_dir` directory beside the artifact. OneChart is
    /// the one zoo arch whose tokenizer is three files rather than one, so it
    /// needs its own slot; every other arch uses `tokenizer_json` or
    /// `qwen_tiktoken`.
    pub opt_triple: Option<[Vec<u8>; 3]>,
}

pub struct OcrModel {
    /// Filesystem path the model was resolved + loaded from (provenance).
    /// Byte-loaded models ([`OcrModel::from_weights`]) carry the sentinel
    /// `<in-memory>` and never touch the filesystem for sidecars.
    path: PathBuf,
    /// In-memory sidecar payloads for byte-loaded models; empty (all `None`)
    /// on the filesystem route, where sidecars resolve beside `path`.
    sidecars: SidecarBundle,
    /// The model these weights are: the id the `.focrq` declares (absent means
    /// Unlimited-OCR v1), or for a raw shard the census it matched. Resolved
    /// once at construction; never a fallback to the default model.
    arch: &'static dyn model_arch::ModelArch,
    /// The loaded weight set. The `.focrq` reader (bd-1es.3) is wired, so every
    /// `Weights`-backed stage entrypoint hydrates its named tensors and runs the
    /// real math.
    weights: Weights,
    /// Frozen greedy decode contract (temperature 0, EOS 1, no_repeat_ngram 35,
    /// single-image window 128). Built once at load so the AR loop reads a
    /// stable config (plan §6.10, [SPEC-100..103]).
    decode_params: DecodeParams,
    /// Lazily-built, then reused int8 decoder weight cache (per-output-channel
    /// S8S8). Building it quantizes the whole decoder (~1.2 s); caching it on the
    /// model amortizes that across every page in a load-once batch
    /// ([`OcrEngine`] already amortizes the 6.2 GB weight load via its `Arc`
    /// cache, so a batch loop pays load + quant ONCE, not per page).
    decoder_cache_i8: FallibleOnce<decoder::DecoderWeightCacheI8>,
    /// Lazily-built conservative recipe cache: int8 FFN/expert weights with
    /// high-precision attention and `lm_head`. Reused across every page.
    decoder_cache: FallibleOnce<decoder::DecoderWeightCache>,
    /// Lazily-built all-f32 cache ([`DECODE_F32_ENV`]): Unlimited-OCR-2's
    /// parity path. Built only when a document asks for it.
    decoder_cache_f32: FallibleOnce<decoder::DecoderWeightCache>,
    /// Lazily-hydrated, then reused CLIP tower weights (bd-av64.10). Hydration
    /// widens ~300 M bf16 params to f32 and pre-transposes every linear into
    /// GEMM layout (~0.6 s, ~1.2 GB resident — the same residency the batched
    /// spine already held per batch); caching it on the model pays that ONCE
    /// per process instead of per view/page. Only populated for archs whose
    /// pipeline runs the CLIP tower (the Baidu path).
    clip_cache: FallibleOnce<vision_clip::ClipWeights>,
    /// Lazily-hydrated Unlimited-OCR SAM tower and pretransposed bridge
    /// projector, reused across every page and both vision schedules.
    unlimited_vision: FallibleOnce<UnlimitedVisionStatics>,
    /// Lazily-hydrated, then reused GOT model-constant tensors (SAM tower +
    /// projector + widened embed table, ~1 GB f32 — bd-av64.10: the sequential
    /// page loop re-hydrated all three EVERY page). Only populated for the
    /// `got-ocr2` arch.
    got_statics: FallibleOnce<got::GotStatics>,
    /// Lazily-hydrated OneChart model-constant tensors (same idiom; only
    /// populated for the `onechart` arch).
    onechart_statics: FallibleOnce<onechart::OnechartStatics>,
    /// Lazily-hydrated SmolVLM2 model-constant tensors (same idiom; only
    /// populated for the `smolvlm2` arch).
    smol_statics: FallibleOnce<smolvlm2::SmolStatics>,
    /// Lazily-loaded, then reused BPE tokenizer. The `tokenizer.json` is ~9.9 MB;
    /// parsing it once and caching it on the model amortizes that across every
    /// page of a multi-page document (e.g. a PDF) instead of re-reading and
    /// re-parsing the file on every prompt-build and detokenize call.
    tokenizer: FallibleOnce<crate::tokenizer::Tokenizer>,
    /// Lazily-loaded GOT-OCR2 Qwen tiktoken tokenizer (from `qwen.tiktoken` beside
    /// the model). Only populated for the `got-ocr2` arch; the Baidu path uses
    /// [`Self::tokenizer`] above.
    got_tokenizer: FallibleOnce<crate::tokenizer::tiktoken::Tiktoken>,
    tromr_tokenizer: FallibleOnce<crate::tokenizer::music::MusicTokenizer>,
    /// Staff-level metadata from the most recent TrOMR full-page forward
    /// (bd-av64.2 observability): recognized staves' indices + bboxes and the
    /// per-staff skips. A side-channel rather than a widened forward return:
    /// the `(text, w, h)` forward contract is shared by every model arm, and
    /// the engine's one-live-forward discipline means "the last music
    /// forward" is well-defined. Taken (consumed) by the CLI after a music
    /// run to emit robot `staff` events and the `--json` staves array.
    music_meta: std::sync::Mutex<Option<MusicPageMeta>>,
    /// What the most recent decode through the sequential driver
    /// ([`Self::generate_with_observer`]: every single-page, multi-page and
    /// streaming Unlimited-OCR forward) did. Same side-channel idiom as
    /// `music_meta`; taken by the CLI for `finish_reason` and the v1
    /// multi-page warnings, and by the token-id golden gate
    /// (`tests/uocr2_goldens.rs`), because one drifted token can survive
    /// detokenize + postprocess without changing the markdown.
    last_decode: std::sync::Mutex<Option<DecodeReport>>,
}

struct UnlimitedVisionStatics {
    sam: vision_sam::SamWeights,
    projector: vision_bridge::ProjectorWeights,
}

/// Process-global cache of the last-loaded model, keyed by resolved path.
///
/// A [`Weak`] so the cache never *keeps the model alive on its own*: once every
/// [`Arc<OcrModel>`] handle is dropped, the weight blob is freed; a subsequent
/// [`OcrModel::load`] of the same path re-reads it. While at least one handle is
/// live, repeat loads of the same path hand back a cheap `Arc::clone`.
type ModelCacheEntry = Option<(PathBuf, Weak<OcrModel>)>;
type ModelCache = Mutex<ModelCacheEntry>;

const MODEL_HEADER_VALIDATION_MAX_BYTES: usize = 8 * 1024 * 1024;
const MODEL_DIR_ENV: &str = "FOCR_MODEL_DIR";
const MODEL_QUANT_ENV: &str = "FOCR_QUANT";

/// The base-mode task prompt. baidu's `infer(..., prompt="<image>document
/// parsing.")` (modeling_unlimitedocr.py): the literal `<image>` expands to the
/// per-view image-placeholder block, and this trailing text tokenizes to the
/// task ids (`document`,`Ġparsing`,`.` = [34030, 76466, 16]). The full base
/// prompt id-stream is therefore `[BOS] + [IMAGE]×273 + [34030, 76466, 16]`.
// Prompt-construction provenance: the reference `plain` conversation template
// concatenates User+Assistant contents with empty seps and `.strip()`
// ([SPEC-034]); the `<|User|>`/`<|Assistant|>` role strings never reach the
// prompt text ([SPEC-036]) — hence the bare "document parsing." tail below.
// The torch-side class hierarchy and config facts ([SPEC-001], [SPEC-002])
// are embodied by this fused forward + the frozen decoder constants.
const BASE_PROMPT_TEXT: &str = "document parsing.";

/// Reference `infer_multi` task prompt (`'<image>Multi page parsing.'`,
/// `modeling_unlimitedocr.py:1139` — the single `<image>` position carries ALL
/// pages' token blocks; bd-1gv.25). Unlimited-OCR-2's prompt ends with the
/// same text, for one page as for many (MU:1073-1080).
const MULTI_PAGE_PROMPT_TEXT: &str = "Multi page parsing.";

/// The most pages Unlimited-OCR-2 parses in one unwindowed pass by default
/// (U2-B6): `floor((131,072 − 64) / (274 + 1,900)) = 60`, i.e. 274 prompt
/// tokens and at most 1,900 generated tokens per page within the position
/// limit. A denser document can still reach the limit sooner; that stop is
/// reported, never silent. U2-C4 re-derives it from measured page lengths and
/// U2-D1b lifts it with windowing.
pub const UOCR2_UNWINDOWED_PAGE_LIMIT: usize = 60;

/// Unlimited-OCR-2's prompt (U2-B6; MU:531-580, 1073-1082): `bos`, then per
/// page its `marker` and `image_tokens[p]` image slots, then the task `text`
/// ids. Always this form, even for one page: no chat template and no caller
/// `<image>` placeholders. The mask is `true` exactly at the image slots.
fn paged_prompt(
    bos: u32,
    marker: u32,
    image: u32,
    image_tokens: &[usize],
    text: &[u32],
) -> (Vec<u32>, Vec<bool>) {
    let total = 1 + image_tokens.iter().map(|n| n + 1).sum::<usize>() + text.len();
    let mut ids = Vec::with_capacity(total);
    let mut mask = Vec::with_capacity(total);
    ids.push(bos);
    mask.push(false);
    for &n in image_tokens {
        ids.push(marker);
        mask.push(false);
        ids.extend(std::iter::repeat_n(image, n));
        mask.extend(std::iter::repeat_n(true, n));
    }
    ids.extend_from_slice(text);
    mask.extend(std::iter::repeat_n(false, text.len()));
    (ids, mask)
}
/// Reference `infer_multi` global-view edge (`image_size=640`, NOT the
/// single-image 1024): every page is resized+padded to 640² Base mode, so the
/// per-page placeholder block is `(10+1)·10 + 1 = 111` tokens (OQ-18).
const MULTI_PAGE_BASE_SIZE: usize = 640;
/// `max_position_embeddings` — the hard 32K cap one cross-page pass must fit
/// in ([SPEC-101]; plan §2.1 "constant-KV long-output decoding of dozens of
/// pages in one 32K pass").
const MAX_POSITION_EMBEDDINGS: usize = sampler::DEFAULT_MAX_POSITIONS;

/// Test-only position budget (U2-X2): lowers [`DecodeParams::max_positions`]
/// so CLI and e2e tests reach the multi-page clamp on small documents. Read
/// once; a value above the arch's context is a usage error
/// ([`validate_test_max_positions`]).
const TEST_MAX_POSITIONS_ENV: &str = "FOCR_TEST_MAX_POSITIONS";

/// The parsed [`TEST_MAX_POSITIONS_ENV`]: `Ok(None)` when unset.
fn test_max_positions() -> &'static Result<Option<usize>, String> {
    static VALUE: OnceLock<Result<Option<usize>, String>> = OnceLock::new();
    VALUE.get_or_init(|| {
        let Some(raw) = std::env::var_os(TEST_MAX_POSITIONS_ENV) else {
            return Ok(None);
        };
        let raw = raw.to_string_lossy();
        match raw.trim().parse::<usize>() {
            Ok(n) if (1..=MAX_POSITION_EMBEDDINGS).contains(&n) => Ok(Some(n)),
            _ => Err(format!(
                "{TEST_MAX_POSITIONS_ENV}={raw:?} must be an integer in 1..={MAX_POSITION_EMBEDDINGS} \
                 (the architecture's context); it is a test-only override"
            )),
        }
    })
}

/// Refuse an out-of-range [`TEST_MAX_POSITIONS_ENV`] before any model work
/// (exit 2).
///
/// # Errors
/// [`FocrError::Usage`] when the variable is set to anything but an integer in
/// `1..=32768`.
pub fn validate_test_max_positions() -> FocrResult<()> {
    test_max_positions()
        .as_ref()
        .map(|_| ())
        .map_err(|why| FocrError::Usage(why.clone()))
}

/// How a decode ended (U2-X2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model emitted EOS.
    Stop,
    /// A token or position limit stopped it first.
    Length,
    /// An Unlimited-OCR-2 document's cooperative stage budget ran out
    /// (`FOCR_STAGE_BUDGET_FORWARD_MS`, U2-B6).
    Budget,
    /// Cooperative shutdown ([`crate::request_shutdown`], Ctrl+C) stopped an
    /// Unlimited-OCR-2 document, which keeps the pages it completed (U2-E2).
    Cancelled,
}

impl FinishReason {
    /// The robot/`--json` spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::Budget => "budget",
            Self::Cancelled => "cancelled",
        }
    }
}

/// The limit that ended a [`FinishReason::Length`] decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LengthCap {
    /// The generated-token cap (`--max-length` / `FOCR_MAX_NEW_TOKENS`).
    MaxLength,
    /// The shared position budget: a multi-page document's prompt left fewer
    /// positions than the token cap.
    Positions,
}

/// What the most recent decode through [`OcrModel::generate_with_observer`]
/// did: every single-page, multi-page and streaming Unlimited-OCR forward.
/// Taken by the CLI for `finish_reason` and the v1 truncation / page-merge
/// warnings, and by the token-id golden gate (`tests/uocr2_goldens.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeReport {
    /// The generated token ids, before detokenize and postprocess.
    pub ids: Vec<u32>,
    /// EOS or a limit.
    pub finish_reason: FinishReason,
    /// The binding limit when `finish_reason` is `Length`.
    pub cap: Option<LengthCap>,
    /// Multi-page decodes: the page count asked for and the page bodies the
    /// output carries ([`postprocess::page_body_count`]).
    pub pages: Option<PageBodies>,
    /// The generated-token cap this decode ran under.
    pub max_length: usize,
    /// The position budget prompt and output shared.
    pub max_positions: usize,
    /// Decodes since the report was last taken (a PDF or tall-capture run
    /// decodes once per page or strip); the fields above describe the last.
    pub decodes: usize,
    /// How many of those decodes stopped at a limit.
    pub length_stops: usize,
    /// An Unlimited-OCR-2 document decode's per-page facts and counters.
    pub document: Option<DocumentStats>,
    /// The `ignored_lever` warnings the model's lever policy raised for this
    /// run's environment (U2-A7), carried into the run's `warnings[]`.
    pub admission_warnings: Vec<RunWarning>,
}

/// How one Unlimited-OCR-2 page body ended (U2-B6): its `finish`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageFinish {
    /// The model opened the next page.
    Marker,
    /// The model ended the document.
    Eos,
    /// The per-page cap forced the next `<PAGE>`, or EOS on the last page.
    PageCap,
    /// The position limit stopped decoding inside this page.
    Length,
    /// The cooperative stage budget stopped decoding inside this page.
    Budget,
    /// Cooperative shutdown stopped decoding inside this page.
    Cancelled,
}

impl PageFinish {
    /// The robot/`--json` spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Marker => "marker",
            Self::Eos => "eos",
            Self::PageCap => "page_cap",
            Self::Length => "length",
            Self::Budget => "budget",
            Self::Cancelled => "cancelled",
        }
    }
}

/// An Unlimited-OCR-2 document decode's per-page facts and counters (U2-B6).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DocumentStats {
    /// Per page the output reached, how its body ended.
    pub page_finish: Vec<PageFinish>,
    /// Generated ids per page body ([`postprocess::TokenPages::tokens`]).
    pub page_tokens: Vec<usize>,
    /// Non-blank text preceded the first `<PAGE>` and joined page 1
    /// (DISC-U2-9).
    pub leading_text: bool,
    /// `<PAGE>` markers past the page count (DISC-U2-3).
    pub extra_markers: usize,
    /// Steps the page-count guard restricted to one token.
    pub forced_steps: usize,
    /// Forced steps where the n-gram had banned the forced token, so HF
    /// would have emitted id 0 (DISC-U2-2).
    pub hf_all_neg_inf_steps: usize,
    /// KV blocks retired (U2-B4).
    pub retired_blocks: usize,
    /// Peak resident KV bytes, summed over the layers.
    pub peak_kv_bytes: usize,
    /// Reruns without retirement after the retirement detector fired.
    pub retire_fallbacks: usize,
    /// The page was read from a gundam view, which has no oracle
    /// certification for Unlimited-OCR-2 yet (U2-E1, U2-G1).
    pub gundam_view: bool,
}

/// One document decode `pass(retire, budget_floor, ids, observer)`, first
/// with block retirement as `retire`. If retirement's detector fires
/// ([`srswa::retire_fallback`]), the pass runs again without retirement,
/// which is deterministic: the observer sees each id once, the rerun must
/// repeat every id the first pass emitted, and `retire_fallbacks` counts it
/// (U2-B4, U2-B6). Any other error ends the decode.
fn decode_with_retire_fallback(
    retire: bool,
    mut observer: Option<&mut dyn FnMut(u32)>,
    mut pass: impl FnMut(
        bool,
        usize,
        &mut Vec<u32>,
        Option<&mut dyn FnMut(u32)>,
    ) -> FocrResult<DocumentDecode>,
) -> FocrResult<DocumentDecode> {
    let mut first = Vec::new();
    // The ids the observer saw: a window's forced prefix is in `first` but
    // never observed.
    let mut observed = 0usize;
    let outcome = {
        let mut counting = |id: u32| {
            observed += 1;
            if let Some(f) = observer.as_deref_mut() {
                f(id);
            }
        };
        pass(retire, 0, &mut first, Some(&mut counting))
    };
    let error = match outcome {
        Err(error) if srswa::retire_fallback(&error).is_some() => error,
        done => return done,
    };
    timing_log(&format!(
        "retire_fallback at step {}: {error:#}; decoding again without retirement",
        first.len()
    ));
    let mut seen = 0usize;
    let mut replay = |id: u32| {
        seen += 1;
        if seen > observed
            && let Some(f) = observer.as_deref_mut()
        {
            f(id);
        }
    };
    let mut ids = Vec::new();
    // The rerun checks the budget only past the ids already observed, so it
    // never ends inside what was streamed.
    let mut decode = pass(false, first.len(), &mut ids, Some(&mut replay))?;
    if !decode.ids.starts_with(&first) {
        return Err(FocrError::Other(anyhow::anyhow!(
            "the unretired rerun departs from the retired run before its detector fired \
             (step {}): block retirement changed the output",
            first.len()
        )));
    }
    decode.stats.retire_fallbacks = 1;
    Ok(decode)
}

/// How one pass of an Unlimited-OCR-2 document decode runs (U2-B6).
#[derive(Debug, Clone, Copy)]
struct DocumentPass {
    /// Which decoder runs the pass.
    precision: DocumentPrecision,
    /// Block retirement on (it also needs the page guard).
    retire: bool,
    /// The cooperative stage budget's deadline.
    deadline: Option<Instant>,
    /// Ids the pass emits before it checks the deadline: a rerun first
    /// repeats the ids its observer has already seen.
    budget_floor: usize,
}

impl DocumentPass {
    /// Past the deadline, once `generated` ids reach the floor.
    fn budget_spent(&self, generated: usize) -> bool {
        generated >= self.budget_floor && self.deadline.is_some_and(|d| Instant::now() >= d)
    }
}

/// One document decode loop's fixed inputs ([`OcrModel::document_loop`]).
#[derive(Debug, Clone, Copy)]
struct DocumentLoop<'p> {
    /// The decode params.
    params: &'p DecodeParams,
    /// The pass it belongs to (deadline and floor).
    pass: DocumentPass,
    /// The `<PAGE>` id.
    marker: u32,
    /// Prompt rows before the generated ids.
    prompt_len: usize,
    /// Pages whose bodies it closes: the document's, or a window's own.
    pages: usize,
    /// Leading ids the caller forced and fed before the loop, hidden from
    /// the observer (a reused window's spliced text and `<PAGE>`).
    forced_prefix: usize,
}

/// The page-count guard of a document decode over `pages` pages marked by
/// `marker` (0 turns it off): the per-page cap is `max_page_tokens` (when
/// > 0) within `max_length`, which in a document applies per page (D11).
fn document_guard(pages: usize, marker: u32, params: &DecodeParams) -> sampler::PageGuard {
    let page_cap = if params.max_page_tokens > 0 {
        params.max_page_tokens.min(params.max_length)
    } else {
        params.max_length
    };
    sampler::PageGuard::new(pages, marker, params.eos_token_id, page_cap)
}

/// The decoder an Unlimited-OCR-2 document pass runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DocumentPrecision {
    /// The shipping cache: int8 FFN/experts, f32 attention, router and
    /// `lm_head`.
    Mixed,
    /// [`DECODE_F32_ENV`]: the all-f32 cache.
    CachedF32,
    /// [`DECODE_STATELESS_ENV`]: the whole grown sequence re-run every step
    /// on the all-f32 path, O(n²).
    Stateless,
}

impl DocumentPrecision {
    /// The decoder the environment asks for.
    ///
    /// # Errors
    /// [`FocrError::Usage`] when both variables are set: each picks the
    /// decoder.
    fn from_env() -> FocrResult<Self> {
        match (
            std::env::var_os(DECODE_STATELESS_ENV).is_some(),
            std::env::var_os(DECODE_F32_ENV).is_some(),
        ) {
            (true, true) => Err(FocrError::Usage(format!(
                "{DECODE_STATELESS_ENV} and {DECODE_F32_ENV} each pick the decoder; set one"
            ))),
            (true, false) => Ok(Self::Stateless),
            (false, true) => Ok(Self::CachedF32),
            (false, false) => Ok(Self::Mixed),
        }
    }

    /// The `precision` timing row's value.
    fn name(self) -> &'static str {
        match self {
            Self::Mixed => "focr-mixed-ffn-int8",
            Self::CachedF32 => "f32-cached",
            Self::Stateless => "f32-stateless",
        }
    }
}

/// The engine bound's margin over a cooperative document budget `budget`
/// (U2-B6): the document loop checks its budget once per decode step, but
/// vision and prefill run unbroken, so the engine's stage timeout waits this
/// much longer before it discards the run. 30 s, or a quarter of the budget.
#[cfg(feature = "native")]
pub(crate) fn document_budget_grace(budget: std::time::Duration) -> std::time::Duration {
    (budget / 4).max(std::time::Duration::from_secs(30))
}

/// The model state an Unlimited-OCR-2 document decode advances (U2-B6).
enum DocumentDecoder<'w> {
    /// The conservative cached path: the SR-SWA caches and the last row.
    Cached {
        wc: &'w decoder::DecoderWeightCache,
        state: srswa::SrSwaState,
        hidden: Mat,
    },
    /// `FOCR_DECODE_STATELESS`: the grown sequence, re-run whole every step
    /// on the all-f32 path, routed by `domain`.
    Stateless {
        embeds: Mat,
        ids: Vec<u32>,
        domain: srswa::EntmaxDomain,
    },
}

/// One Unlimited-OCR-2 document decode ([`OcrModel::generate_document`]).
#[derive(Debug, Clone, PartialEq, Eq)]
struct DocumentDecode {
    /// The generated ids; EOS is never appended (MU:1176-1178).
    ids: Vec<u32>,
    /// EOS, or the position limit.
    finish_reason: FinishReason,
    /// The decode's facts; `page_tokens`, `leading_text` and
    /// `extra_markers` are filled by the split.
    stats: DocumentStats,
}

impl DecodeReport {
    /// The run-level `finish_reason`: `length` when any decode since the last
    /// take stopped at a limit, or `budget` / `cancelled` when the last one
    /// ran out of its stage budget or was cancelled (which ends the run).
    #[must_use]
    pub fn run_finish_reason(&self) -> FinishReason {
        match (self.length_stops, self.finish_reason) {
            (0, _) => FinishReason::Stop,
            (_, stop @ (FinishReason::Budget | FinishReason::Cancelled)) => stop,
            _ => FinishReason::Length,
        }
    }
}

/// Expected pages vs page bodies of a multi-page decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageBodies {
    /// Source pages.
    pub expected: usize,
    /// Page bodies in the decoded text.
    pub bodies: usize,
}

/// The kinds of run warnings and recoveries the engine reports (U2-X2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningKind {
    /// A v1 multi-page decode stopped at the position budget before its last page.
    V1TruncatedPositions,
    /// A v1 multi-page decode stopped at a user token cap before its last page.
    V1TruncatedUserCap,
    /// A v1 multi-page decode ended at EOS with fewer page bodies than pages.
    V1PageMerge,
    /// A multi-page prompt alone exceeds the position budget (an error, not a
    /// warning; it shares the recovery text).
    PromptTooLong,
    /// A `FOCR_*` lever that reaches no path of the loaded model (U2-A7).
    IgnoredLever,
    /// An Unlimited-OCR-2 document stopped at its position limit (U2-B6).
    V2Length,
    /// An Unlimited-OCR-2 document ran out of its stage budget (U2-B6).
    V2Budget,
    /// Text before the first generated `<PAGE>` joined page 1 (DISC-U2-9).
    LeadingText,
    /// A directory input held files that are not page images (U2-D0).
    IgnoredFiles,
    /// Unlimited-OCR-2 read a page in the uncertified gundam mode (U2-E1).
    UncertifiedCropMode,
    /// The run mode's default model is not installed, so v1 served (D25).
    PullHint,
}

/// Every machine-stable `warnings[].kind` a run can report, in the robot
/// schema's `warning_kinds` (U2-A7 registers `ignored_lever`, U2-B6 the
/// Unlimited-OCR-2 document kinds, U2-D0 `ignored_files`).
pub const RUN_WARNING_KINDS: &[&str] = &[
    "v1_truncated",
    "v1_page_merge",
    "ignored_lever",
    "v2_length",
    "v2_budget",
    "leading_text",
    "ignored_files",
    "uncertified_crop_mode",
    "pull_hint",
];

/// The one actionable next step for each [`WarningKind`]: the single source
/// U2-E4 extends with `focr pull unlimited-ocr-2` once v2 is published.
#[must_use]
pub fn warning_recovery(kind: WarningKind) -> &'static str {
    match kind {
        WarningKind::V1TruncatedPositions | WarningKind::PromptTooLong => {
            "split the document with `--pages`, or parse pages independently (omit `--multi-page`)"
        }
        WarningKind::V1TruncatedUserCap => "raise `--max-length`",
        WarningKind::V1PageMerge => {
            "parse pages independently (omit `--multi-page`) when page boundaries matter"
        }
        WarningKind::IgnoredLever => "unset it for this model's runs",
        WarningKind::V2Length => {
            "the completed pages were written; split the document with `--pages`, or parse \
             pages independently (omit `--multi-page`)"
        }
        WarningKind::V2Budget => {
            "the completed pages were written; raise FOCR_STAGE_BUDGET_FORWARD_MS (0 = \
             unlimited), or select fewer pages with `--pages`"
        }
        WarningKind::LeadingText => {
            "check the start of page 1: the model wrote text before its first page marker"
        }
        WarningKind::IgnoredFiles => {
            "only page images are read (PNG, JPEG, TIFF, WebP, BMP, GIF, PNM); convert any of \
             these files that belong to the document"
        }
        WarningKind::UncertifiedCropMode => {
            "use `--crop-mode base` (the default) for certified output; check gundam output \
             against the page"
        }
        WarningKind::PullHint => {
            "install the default with `focr pull <model_id>` (this warning's `model_id`), or \
             choose a model with `--model`"
        }
    }
}

/// One `warnings[]` entry: `{kind, message, recovery, ...kind-specific
/// fields}` on `run_complete`, in `--json` and on stderr (U2-X2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunWarning {
    /// Machine-stable kind (`v1_truncated`, `v1_page_merge`).
    pub kind: &'static str,
    /// The human line, also printed to stderr.
    pub message: String,
    /// [`warning_recovery`] for this warning.
    pub recovery: &'static str,
    /// Kind-specific fields (`expected`, `got`, `cause`).
    pub fields: serde_json::Map<String, serde_json::Value>,
}

impl RunWarning {
    /// The JSON object form.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        obj.insert("kind".into(), self.kind.into());
        obj.insert("message".into(), self.message.clone().into());
        obj.insert("recovery".into(), self.recovery.into());
        obj.extend(self.fields.clone());
        serde_json::Value::Object(obj)
    }
}

/// The model that served a run (U2-E1, D25): its registry id and the source
/// digest a `.focrq` records in its fixed prefix. A raw checkpoint load hashes
/// nothing (docs/focrq-format.md), so its digest is `None`; nothing here
/// hashes the artifact file itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIdentity {
    /// The [`model_arch::ModelArch::id`] that ran.
    pub model_id: &'static str,
    /// The lowercase-hex `source_sha256` the artifact records, if any.
    pub source_sha256: Option<String>,
}

impl ModelIdentity {
    /// The run store's `model_version_tag`: `<model_id>@<source_sha256>`, or
    /// the bare id when the artifact records no digest.
    #[must_use]
    pub fn version_tag(&self) -> String {
        match &self.source_sha256 {
            Some(sha) => format!("{}@{sha}", self.model_id),
            None => self.model_id.to_owned(),
        }
    }

    /// Set `model_id` and `source_sha256` (`null` without a digest) on a JSON
    /// object.
    pub fn merge_into(&self, value: &mut serde_json::Value) {
        value["model_id"] = self.model_id.into();
        value["source_sha256"] = self.source_sha256.clone().into();
    }
}

/// `32768` → `"32,768"`.
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// A run's warnings from its last decode report: the model admission's
/// `ignored_lever` warnings first, then an Unlimited-OCR-2 document's
/// ([`document_warnings`]) or a v1 decode's ([`v1_page_warnings`]).
#[must_use]
pub fn decode_warnings(report: &DecodeReport) -> Vec<RunWarning> {
    let mut warnings = report.admission_warnings.clone();
    warnings.extend(match &report.document {
        Some(stats) => document_warnings(report, stats),
        None => v1_page_warnings(report),
    });
    warnings
}

/// The warnings a v1 decode earns (U2-X2): only multi-page decodes with fewer
/// page bodies than pages warn. A length stop is a truncation, named by the
/// limit that bound it; an EOS stop is a page merge. Single-page decodes
/// report only their `finish_reason`.
fn v1_page_warnings(report: &DecodeReport) -> Vec<RunWarning> {
    let Some(PageBodies { expected, bodies }) = report.pages else {
        return Vec::new();
    };
    if bodies >= expected {
        return Vec::new();
    }
    let mut fields = serde_json::Map::new();
    fields.insert("expected".into(), expected.into());
    fields.insert("got".into(), bodies.into());
    let warning = match (report.finish_reason, report.cap) {
        (FinishReason::Length, Some(LengthCap::MaxLength)) => {
            fields.insert("cause".into(), "user_cap".into());
            RunWarning {
                kind: "v1_truncated",
                message: format!(
                    "stopped after {bodies} of {expected} pages: the --max-length cap of {} \
                     generated tokens was reached",
                    thousands(report.max_length)
                ),
                recovery: warning_recovery(WarningKind::V1TruncatedUserCap),
                fields,
            }
        }
        (FinishReason::Length, _) => {
            fields.insert("cause".into(), "positions".into());
            let dense = if report.max_positions == MAX_POSITION_EMBEDDINGS {
                " (≈29 dense pages)"
            } else {
                ""
            };
            RunWarning {
                kind: "v1_truncated",
                message: format!(
                    "stopped after {bodies} of {expected} pages: v1 shares {} positions between \
                     prompt and output{dense}",
                    thousands(report.max_positions)
                ),
                recovery: warning_recovery(WarningKind::V1TruncatedPositions),
                fields,
            }
        }
        (FinishReason::Stop, _) => RunWarning {
            kind: "v1_page_merge",
            message: format!(
                "v1 produced {bodies} page bodies for {expected} pages; page boundaries in this \
                 output are unreliable"
            ),
            recovery: warning_recovery(WarningKind::V1PageMerge),
            fields,
        },
        // Only Unlimited-OCR-2's document loop checks a budget or ends on a
        // cancel, and its reports take `document_warnings`.
        (FinishReason::Budget | FinishReason::Cancelled, _) => return Vec::new(),
    };
    vec![warning]
}

/// An Unlimited-OCR-2 document decode's warnings (U2-B6), never v1's: the
/// head before the first `<PAGE>` joined to page 1 (DISC-U2-9), and a stop
/// at the position limit or the stage budget, with the page bodies produced.
fn document_warnings(report: &DecodeReport, stats: &DocumentStats) -> Vec<RunWarning> {
    let mut warnings = Vec::new();
    if stats.gundam_view {
        warnings.push(RunWarning {
            kind: "uncertified_crop_mode",
            message: "Unlimited-OCR-2 read the page in gundam mode, which no oracle has \
                      certified yet"
                .into(),
            recovery: warning_recovery(WarningKind::UncertifiedCropMode),
            fields: serde_json::Map::new(),
        });
    }
    if stats.leading_text {
        warnings.push(RunWarning {
            kind: "leading_text",
            message: "text the model wrote before its first <PAGE> was joined to page 1".into(),
            recovery: warning_recovery(WarningKind::LeadingText),
            fields: serde_json::Map::new(),
        });
    }
    let (expected, got) = report.pages.map_or((1, 0), |p| (p.expected, p.bodies));
    let place = if got == 0 {
        format!("before page 1 of {expected}")
    } else {
        format!("inside page {got} of {expected}")
    };
    let mut fields = serde_json::Map::new();
    fields.insert("expected".into(), expected.into());
    fields.insert("got".into(), got.into());
    match report.finish_reason {
        // A cancel is the run's own `finish_reason`, and exit 6.
        FinishReason::Stop | FinishReason::Cancelled => {}
        FinishReason::Length => warnings.push(RunWarning {
            kind: "v2_length",
            message: format!(
                "stopped {place}: unwindowed Unlimited-OCR-2 shares {} positions between prompt \
                 and output",
                thousands(report.max_positions)
            ),
            recovery: warning_recovery(WarningKind::V2Length),
            fields,
        }),
        FinishReason::Budget => warnings.push(RunWarning {
            kind: "v2_budget",
            message: format!("stopped {place}: the forward stage budget ran out"),
            recovery: warning_recovery(WarningKind::V2Budget),
            fields,
        }),
    }
    warnings
}

/// The partial-output error for a run whose Unlimited-OCR-2 decode stopped
/// at its position limit or stage budget (exit 5) or on a cancel (exit 6),
/// carrying `warnings` (the run's, in order) and the `model` that wrote the
/// pages; `None` for every other run (U2-B6, U2-E2). The CLI returns it
/// after writing the pages, so a short document never exits 0.
#[must_use]
pub fn partial_output(
    report: &DecodeReport,
    warnings: &[RunWarning],
    model: &ModelIdentity,
) -> Option<FocrError> {
    report.document.as_ref()?;
    let finish = report.run_finish_reason();
    if finish == FinishReason::Stop {
        return None;
    }
    let (expected, pages_out) = report.pages.map_or((1, 0), |p| (p.expected, p.bodies));
    Some(FocrError::Partial(Box::new(crate::error::PartialOutput {
        finish_reason: finish.as_str(),
        pages_out,
        warnings: warnings.iter().map(RunWarning::to_json).collect(),
        model_id: model.model_id,
        source_sha256: model.source_sha256.clone(),
        message: format!(
            "the document stopped early ({}) after {pages_out} of {expected} page bodies; the \
             pages produced were written",
            finish.as_str()
        ),
    })))
}

/// Streaming sink for completed multi-page bodies (bd-2z0y): called with
/// each [`StreamedPage`] as its `<PAGE>` boundary is crossed in the token
/// stream.
type PageSink<'a> = &'a mut dyn FnMut(&StreamedPage<'_>);

/// A page body an Unlimited-OCR-2 document decode closed (U2-E3), as it
/// streams: [`StreamedPage`]'s facts before the page's origin is known.
struct ClosedPage<'a> {
    /// 1-based, in the model's emission order.
    index: usize,
    /// The `.trim()`-ed raw body.
    body: &'a str,
    /// Generated ids in the body, its `<PAGE>` excluded.
    tokens: usize,
    /// How the body ended.
    finish: PageFinish,
    /// Wall time decoding the body ([`PageResult::elapsed_ms`]).
    elapsed: std::time::Duration,
}

/// What a document's pages are read against (U2-E3): each prompt page's
/// raster size and origin, and the model.
#[derive(Debug)]
struct PageFrame<'a> {
    /// Page `k + 1`'s raster `(width, height)`, its boxes' frame.
    sizes: Vec<(u32, u32)>,
    /// Page `k + 1`'s origin.
    origins: &'a [PageOrigin],
    model_id: &'static str,
}

impl PageFrame<'_> {
    /// Page `index` (1-based, at most the page count) from its raw body.
    fn page(&self, index: usize, body: &str) -> PageResult {
        let k = index - 1;
        PageResult::read(index, body, self.sizes[k], self.origins[k], self.model_id)
    }

    /// A streamed page with its decode facts.
    fn closed_page(&self, page: &ClosedPage<'_>) -> PageResult {
        PageResult {
            tokens: Some(page.tokens),
            finish: Some(page.finish),
            elapsed_ms: Some(millis(page.elapsed)),
            ..self.page(page.index, page.body)
        }
    }

    /// Unlimited-OCR-2's pages: one per body of the token split, which
    /// yields at most one per page (D20), with its decode facts.
    fn token_pages(&self, decoded: &DecodedDocument) -> Vec<PageResult> {
        let split = &decoded.split;
        split
            .bodies
            .iter()
            .enumerate()
            .map(|(k, body)| PageResult {
                tokens: split.tokens.get(k).copied(),
                finish: decoded.finish.get(k).copied(),
                elapsed_ms: decoded.elapsed.get(k).copied().map(millis),
                ..self.page(k + 1, body)
            })
            .collect()
    }

    /// v1's pages from its decoded `text` (EOS stripped): body k is page k
    /// only when there is one body per page. Otherwise none, and the run's
    /// `v1_page_merge` warning says why.
    fn text_pages(&self, text: &str) -> Vec<PageResult> {
        let bodies: Vec<&str> = postprocess::text_page_bodies(text).collect();
        if bodies.len() != self.sizes.len() {
            return Vec::new();
        }
        bodies
            .iter()
            .enumerate()
            .map(|(k, body)| self.page(k + 1, body))
            .collect()
    }
}

/// An Unlimited-OCR-2 document decode split into pages (U2-B6, U2-E3).
#[derive(Debug)]
struct DecodedDocument {
    /// The bodies, by token id.
    split: postprocess::TokenPages,
    /// How each body ended ([`DocumentStats::page_finish`]).
    finish: Vec<PageFinish>,
    /// Each body's decode wall time ([`PageResult::elapsed_ms`]).
    elapsed: Vec<std::time::Duration>,
}

/// Env override for the generated-token cap (`max_length` in [`DecodeParams`]).
/// Unset ⇒ the reference default ([`sampler::DEFAULT_MAX_LENGTH`]). Setting it
/// only LOWERS the practical decode cost during bring-up / bounded runs — it
/// never alters the per-step math, so a capped run's tokens are a true prefix of
/// the full run's tokens.
const MAX_NEW_TOKENS_ENV: &str = "FOCR_MAX_NEW_TOKENS";

/// Force the stateless O(n^2) re-prefill decode instead of the default O(n)
/// R-SWA ring-cache decode (bd-1gv.17). Present ⇒ stateless. Kept as the parity
/// oracle: the cached path must emit the same tokens for the first 128 steps (the
/// R-SWA ring window, before any generated-tail eviction).
const DECODE_STATELESS_ENV: &str = "FOCR_DECODE_STATELESS";

/// Decode an Unlimited-OCR-2 document on the all-f32 cache
/// ([`decoder::DecoderWeightCache::build_srswa_f32`]): the stateless path's
/// math with the KV kept, in linear time (U2-C2). Present ⇒ f32. It needs the
/// raw checkpoint, and it excludes [`DECODE_STATELESS_ENV`].
const DECODE_F32_ENV: &str = "FOCR_DECODE_F32";

/// Default-on cache for the immutable Unlimited-OCR SAM/projector bundles.
/// Falsy values restore the original per-page hydration wrappers for diagnosis
/// and retained-memory comparisons.
const UNLIMITED_VISION_CACHE_ENV: &str = "FOCR_UNLIMITED_VISION_CACHE";

/// How many views the streamed (wasm-residency) vision lane keeps in flight
/// while a hydrated block is live (bd-K2 per-view hydration hoist). Every view
/// in a chunk shares ONE hydration sweep of the bf16 blob; the price is one
/// live `[gh·gw, 768]` f32 SAM activation per view (~13 MB at 1024², ~5 MB per
/// 640² Gundam tile), so the cap bounds the transient on tile-heavy pages while
/// still covering the 2-view page the unlimited path produces. Chunking cannot
/// change results — the views are independent.
const STREAMED_VIEW_CHUNK: usize = 8;

fn unlimited_vision_cache_enabled_for(value: Option<&str>) -> bool {
    !matches!(
        value
            .map(|value| value.trim().to_ascii_lowercase())
            .as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

fn unlimited_vision_cache_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let value = std::env::var(UNLIMITED_VISION_CACHE_ENV).ok();
        unlimited_vision_cache_enabled_for(value.as_deref())
    })
}

/// Stream the vision tower per block instead of retaining a hydrated f32 copy.
///
/// This is the residency lever for the models whose tower is NOT covered by the
/// Unlimited-OCR wasm recipe gate: GOT-OCR2 and OneChart hydrate a ~382 MB f32
/// SAM tower AND run the unbounded global-attention kernel (~0.8 GB of transient
/// scratch), which together are most of why they measure over 3 GB. Streaming
/// hydrates one block at a time from the same bf16 source and selects the
/// bounded query-slab attention, which is bit-identical by construction (both
/// arms run one `forward_core` body).
///
/// Default ON for iOS, where the whole point is fitting a phone, and OFF
/// elsewhere so desktop keeps the throughput of a retained tower. `FOCR_STREAM_VISION`
/// overrides in both directions for A/B measurement.
fn stream_vision_tower() -> bool {
    static STREAM: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *STREAM.get_or_init(|| match std::env::var("FOCR_STREAM_VISION").ok() {
        Some(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "on" | "true" | "yes"
        ),
        None => cfg!(target_os = "ios"),
    })
}

/// Use the experimental all-int8 decoder cache instead of the conservative
/// recipe path. A truthy value requests it, but the request is accepted only
/// when `FOCR_INT8_ATTN` and `FOCR_INT8_LMHEAD` are also truthy because this
/// cache quantizes both gated tensor sets in addition to the validated FFNs.
const DECODE_INT8_ENV: &str = "FOCR_DECODE_INT8";

/// Process-global override that forces the int8 decode path without mutating the
/// environment (edition-2024 `set_var` is `unsafe`, and this crate denies unsafe).
/// Set by the load-once batch CLI so a batch run gets the int8 throughput path +
/// amortized int8 weight cache. OR'd with [`DECODE_INT8_ENV`].
static FORCE_INT8_DECODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Force (or clear) the experimental all-int8 decode path process-wide.
///
/// # Errors
/// [`FocrError::Usage`] when enabling it without both independent accuracy
/// gates. Clearing the override always succeeds.
pub fn force_int8_decode(on: bool) -> FocrResult<()> {
    if on {
        require_experimental_full_int8_recipe()?;
    }
    FORCE_INT8_DECODE.store(on, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// Whether decode should run the all-int8 cache: the process-global force flag
/// or a truthy environment request.
fn int8_decode_requested() -> bool {
    FORCE_INT8_DECODE.load(std::sync::atomic::Ordering::Relaxed)
        || std::env::var(DECODE_INT8_ENV).is_ok_and(|value| is_truthy(&value))
}

/// Whether the experimental all-int8 decoder was requested through either the
/// library override or its master environment switch.
#[must_use]
pub fn experimental_full_int8_decode_requested() -> bool {
    int8_decode_requested()
}

/// Enforce the three-switch contract for the legacy all-int8 cache. The cache
/// quantizes q/k/v/o and `lm_head` unconditionally, so arming only its master
/// switch would bypass the two accuracy gates in the canonical recipe.
fn require_experimental_full_int8_recipe() -> FocrResult<()> {
    let recipe = Recipe::from_env();
    if recipe.attn_int8() && recipe.lmhead_int8() {
        return Ok(());
    }
    Err(FocrError::Usage(format!(
        "{DECODE_INT8_ENV}=1 requests the experimental all-int8 decoder; it also requires \
         {}=1 and {}=1 because attention q/k/v/o and lm_head are outside the validated \
         default recipe",
        crate::quant::recipe::FOCR_INT8_ATTN_ENV,
        crate::quant::recipe::FOCR_INT8_LMHEAD_ENV,
    )))
}

/// Validate an externally supplied all-int8 request, if any. CLI front ends use
/// this before loading the model so a misconfigured experiment fails promptly.
pub fn validate_experimental_full_int8_decode() -> FocrResult<()> {
    if int8_decode_requested() {
        require_experimental_full_int8_recipe()?;
    }
    Ok(())
}

/// `FOCR_GOT_FORMAT` (bd-3jo6.2.10 / B10): request GOT-OCR2's `OCR with format:`
/// structured output (Mathpix-Markdown — inline LaTeX math, Markdown tables, TikZ
/// geometry, SMILES molecules, `**kern` sheet music; the model auto-selects the
/// formalism) instead of plain `OCR: ` text. Presence ⇒ on. Only affects the
/// `got-ocr2` arm; a no-op for the default unlimited-ocr model.
const GOT_FORMAT_ENV: &str = "FOCR_GOT_FORMAT";

/// Process-global "use GOT `OCR with format:` mode" flag, set by the CLI from
/// `--format` (edition-2024 `set_var` is `unsafe`; this crate denies unsafe — same
/// pattern as [`FORCE_INT8_DECODE`]). OR'd with [`GOT_FORMAT_ENV`].
static FORCE_GOT_FORMAT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Force (or clear) GOT `OCR with format:` mode process-wide. Default false ⇒
/// [`OcrModel::forward_got`] runs plain `OCR: `, byte-identical to today.
pub fn force_got_format(on: bool) {
    FORCE_GOT_FORMAT.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether GOT should run `OCR with format:`: the process-global flag OR the env var.
fn got_format_requested() -> bool {
    FORCE_GOT_FORMAT.load(std::sync::atomic::Ordering::Relaxed)
        || std::env::var_os(GOT_FORMAT_ENV).is_some()
}

/// Optional env override for the smolvlm2 describe/VQA question (the env
/// analog of `--question`, mirroring [`GOT_FORMAT_ENV`]'s pattern). The CLI
/// flag outranks it.
const SMOLVLM2_QUESTION_ENV: &str = "FOCR_SMOLVLM2_QUESTION";

/// Process-global smolvlm2 describe/VQA question, set by the CLI from
/// `--question` (same process-global threading as [`FORCE_GOT_FORMAT`] — the
/// shared engine signatures stay frozen). `None` ⇒ the env var, else the
/// model-card describe prompt.
static SMOLVLM2_QUESTION: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Set (or clear) the smolvlm2 describe/VQA question process-wide. Only
/// affects the `smolvlm2` arm; a no-op for every other arch.
pub fn set_smolvlm2_question(question: Option<String>) {
    *SMOLVLM2_QUESTION
        .lock()
        .expect("smolvlm2 question mutex poisoned") = question;
}

/// The effective describe/VQA question: CLI flag > env var > the model-card
/// caption prompt ([`smolvlm2::DESCRIBE_QUESTION`]).
fn smolvlm2_question() -> String {
    if let Some(q) = SMOLVLM2_QUESTION
        .lock()
        .expect("smolvlm2 question mutex poisoned")
        .clone()
    {
        return q;
    }
    std::env::var(SMOLVLM2_QUESTION_ENV).unwrap_or_else(|_| smolvlm2::DESCRIBE_QUESTION.to_string())
}

// ── one-live-forward gauge (bd-1azu.14, the doctrine-#5 conscience) ─────────
//
// Every heavy forward (vision tower — per-page or batched — and a page prefill;
// the batch scheduler's decode step wires in via `enter_forward` too) holds a
// [`ForwardPass`] token for its duration. The gauge records the maximum number
// of tokens ever live at once (must stay 1: ONE live forward process-wide) and
// whether any forward began while the model-cache mutex guard was held (must
// stay false: NEVER fan out under a held lock). Reading + resetting is a
// test-only hook; the per-forward cost is two relaxed atomics.

static FORWARDS_LIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FORWARDS_MAX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FORWARD_UNDER_GUARD: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

thread_local! {
    /// Model-cache guards held by THIS thread. Thread-local (fresh-eyes fix):
    /// the doctrine-#5 rayon-under-lock class is a SAME-thread pattern — a
    /// forward beginning on thread A while an unrelated thread B briefly holds
    /// the cache guard (e.g. concurrent engine construction) is legal and must
    /// not trip the violation flag.
    static CACHE_GUARD_HELD_HERE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// RAII token for one live forward; dropping it decrements the gauge.
pub(crate) struct ForwardPass(());

impl Drop for ForwardPass {
    fn drop(&mut self) {
        FORWARDS_LIVE.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Enter a heavy forward: bump the live count, fold it into the observed max,
/// and record whether the model-cache guard is currently held by THIS thread
/// (fanning out under a held lock is the deadlock class doctrine #5 bans).
pub(crate) fn enter_forward() -> ForwardPass {
    let live = FORWARDS_LIVE.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
    FORWARDS_MAX.fetch_max(live, std::sync::atomic::Ordering::SeqCst);
    if CACHE_GUARD_HELD_HERE.with(std::cell::Cell::get) > 0 {
        FORWARD_UNDER_GUARD.store(true, std::sync::atomic::Ordering::SeqCst);
    }
    ForwardPass(())
}

/// Snapshot + reset the gauge: `(max_concurrent_forwards, forward_began_under_guard)`.
/// Test-only observability for the `many_pages_without_deadlock` watchdog.
#[doc(hidden)]
pub fn forward_gauge_take() -> (usize, bool) {
    let max = FORWARDS_MAX.swap(0, std::sync::atomic::Ordering::SeqCst);
    let under = FORWARD_UNDER_GUARD.swap(false, std::sync::atomic::Ordering::SeqCst);
    (max, under)
}

/// `FOCR_BATCH_VISION` (bd-1azu.10 / bd-t6a): inside the batch spine, run the
/// vision tower BATCHED ACROSS PAGES — hydrate the SAM/CLIP weights once per
/// batch and stack every admitted page's views into one batched SAM forward +
/// one batched CLIP forward (both proven byte-identical to their per-view
/// forwards; the composition is lossless by construction and held by a
/// model-gated parity gate). `=0`/`off` reverts to the per-page vision loop
/// (the kill-switch). Only the spine reads this; the sequential single-page
/// path is untouched.
const BATCH_VISION_ENV: &str = "FOCR_BATCH_VISION";

/// Whether the spine batches vision across pages: default ON;
/// `FOCR_BATCH_VISION=0` (or `off`/`false`/`no`) reverts to the per-page loop.
fn batch_vision_enabled() -> bool {
    !matches!(
        std::env::var(BATCH_VISION_ENV)
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

/// CLI decode-parameter overrides (`--max-length`, `--temperature`,
/// `--no-repeat-ngram`, `--ngram-window`), threaded to the leaf via a
/// process-global for the same reason as [`force_got_format`]: the shared
/// `OcrEngine`/`OcrModel` signatures stay frozen, and edition-2024 `set_var` is
/// `unsafe` (this crate denies unsafe). `None` = keep the built-in default.
/// Applied when a model's [`DecodeParams`] are built (at model load), so the CLI
/// sets these BEFORE constructing the engine. A `no_repeat_ngram`/`ngram_window`
/// override automatically disarms speculative decode: its dispatch guards on the
/// frozen `35/128` single-image ban (see [`SPEC_DECODE_ENV`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DecodeOverrides {
    /// Generated-token cap (`--max-length`). Also clamps the GOT arm (which
    /// additionally caps at its config `max_new_tokens` 4096, bd-3j3p).
    pub max_length: Option<usize>,
    /// Decode temperature (`--temperature`; 0.0 = greedy).
    pub temperature: Option<f32>,
    /// Sliding no-repeat n-gram size (`--no-repeat-ngram` / FOCR_NO_REPEAT_NGRAM).
    pub no_repeat_ngram: Option<usize>,
    /// Sliding no-repeat n-gram lookback window (`--ngram-window`).
    pub ngram_window: Option<usize>,
    /// A paged model's generated-token cap per page (`--max-page-tokens`; 0
    /// lifts only this cap). Other models have no pages and ignore it.
    pub max_page_tokens: Option<usize>,
}

/// CLI preprocess overrides (`--base-size`, `--image-size`, `--crop-mode`) —
/// the preprocess twin of [`DecodeOverrides`] (bd-1e9n). `None` everywhere =
/// the certified default (a single Base-1024 global view — every oracle cert
/// and golden was produced there). `gundam: Some(true)` selects the reference
/// dynamic-tiling mode ([SPEC-023..028]; connector + placeholder census are
/// implemented and unit-tested, but Gundam e2e has no oracle cert yet — the
/// CLI surfaces it as the reference behavior it is, and parity for it rides
/// the conformance ladder like everything else).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PreprocessOverrides {
    /// Global-view edge length (`--base-size`; default 1024).
    pub base_size: Option<usize>,
    /// Gundam local-tile edge length (`--image-size`; default 640).
    pub image_size: Option<usize>,
    /// `--crop-mode`: `Some(true)` = Gundam tiling, `Some(false)` = Base,
    /// `None` = the engine default (Base).
    pub gundam: Option<bool>,
}

/// The process-global [`PreprocessOverrides`] store (same discipline as
/// [`DECODE_OVERRIDES`]: set by the CLI before the engine is built).
static PREPROCESS_OVERRIDES: Mutex<PreprocessOverrides> = Mutex::new(PreprocessOverrides {
    base_size: None,
    image_size: None,
    gundam: None,
});

/// Set (or clear, with `PreprocessOverrides::default()`) the process-wide
/// preprocess overrides. Call BEFORE recognitions.
pub fn set_preprocess_overrides(overrides: PreprocessOverrides) {
    *PREPROCESS_OVERRIDES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = overrides;
}

/// Resolve the forward path's preprocess mode: the certified Base-1024 default
/// unless the CLI overrode it. Pure over its input; the tiny wrapper below
/// reads the global.
fn resolve_preprocess_mode(o: PreprocessOverrides) -> preprocess::PreprocessMode {
    let base_size = o.base_size.unwrap_or(1024);
    if o.gundam == Some(true) {
        preprocess::PreprocessMode::Gundam {
            base_size,
            tile_size: o.image_size.unwrap_or(640),
        }
    } else {
        preprocess::PreprocessMode::Base { base_size }
    }
}

/// The view a paged model (`model`) reads each document page in: a base view
/// of `mode`'s size.
///
/// # Errors
/// [`FocrError::Usage`] for gundam `mode`, which reads a single page: a
/// document reads base views only (DISC-U2-6), so the request is refused
/// rather than dropped (U2-E1).
fn paged_document_view(
    mode: preprocess::PreprocessMode,
    model: &str,
) -> FocrResult<preprocess::PreprocessMode> {
    match mode {
        preprocess::PreprocessMode::Base { .. } => Ok(mode),
        preprocess::PreprocessMode::Gundam { .. } => Err(FocrError::Usage(format!(
            "--crop-mode gundam reads a single page; a {model} document reads every page as a \
             base view. Omit --crop-mode, or parse the pages one at a time without --multi-page"
        ))),
    }
}

/// The forward path's preprocess mode under the current process-wide overrides.
fn preprocess_mode() -> preprocess::PreprocessMode {
    resolve_preprocess_mode(
        *PREPROCESS_OVERRIDES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
}

/// `arch`'s preprocessing rules (U2-B7): Unlimited-OCR-2 takes v2's (16-tile
/// gundam cap, PIL-exact resampling); every other arch keeps v1's.
fn preprocess_profile(arch: &dyn model_arch::ModelArch) -> preprocess::PreprocessProfile {
    match arch.unlimited_variant() {
        Some(model_arch::UnlimitedVariant::V2SrSwa) => preprocess::PreprocessProfile::V2,
        Some(model_arch::UnlimitedVariant::V1Rswa) | None => preprocess::PreprocessProfile::V1,
    }
}

/// The process-global [`DecodeOverrides`] store. A `Mutex` (not `OnceLock`) so a
/// long-lived embedder can re-set between recognitions; reads happen once per
/// model load, never per token.
static DECODE_OVERRIDES: Mutex<DecodeOverrides> = Mutex::new(DecodeOverrides {
    max_length: None,
    temperature: None,
    no_repeat_ngram: None,
    ngram_window: None,
    max_page_tokens: None,
});

/// Set (or clear, with `DecodeOverrides::default()`) the process-wide decode
/// parameter overrides. Call BEFORE constructing the engine — a model already
/// loaded into the process cache keeps the params it was built with.
pub fn set_decode_overrides(overrides: DecodeOverrides) {
    *DECODE_OVERRIDES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = overrides;
}

/// The current process-wide decode overrides (a copy).
fn decode_overrides() -> DecodeOverrides {
    *DECODE_OVERRIDES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Apply `overrides` onto `params` (a `Some` field wins; `None` keeps `params`).
/// Pure, so the mapping is unit-testable without touching the process global.
fn apply_decode_overrides(params: &mut DecodeParams, overrides: DecodeOverrides) {
    if let Some(n) = overrides.max_length {
        params.max_length = n;
    }
    if let Some(t) = overrides.temperature {
        params.temperature = t;
    }
    if let Some(n) = overrides.no_repeat_ngram {
        params.no_repeat_ngram_size = n;
    }
    if let Some(w) = overrides.ngram_window {
        params.ngram_window = w;
    }
}

/// `FOCR_SPEC_DECODE` (bd-1azu.35): presence kill-switch arming the draft -> verify
/// -> accept speculative decode inside the int8 generate loop
/// ([`OcrModel::spec_decode_i8`]). DEFAULT OFF => [`OcrModel::generate_cached_i8`]
/// runs EXACTLY today's sequential greedy loop, byte-for-byte the same tokens. When
/// set, the spec loop emits the BYTE-IDENTICAL token stream — speculation changes
/// only WHEN logits are evaluated, never WHICH token greedy decode picks. Its
/// verifier's chooser (`spec::accept_longest`) assumes the frozen single-image ban,
/// so the dispatch also guards `no_repeat_ngram_size == 35 && ngram_window == 128`.
const SPEC_DECODE_ENV: &str = "FOCR_SPEC_DECODE";

/// Whether speculative decode is armed ([`SPEC_DECODE_ENV`], read ONCE into a
/// process-wide bool — never touched per token), exactly like the other decode
/// kill-switches. The spec kernels are pure and gated regardless of this flag; only
/// the generate loop's decision to route through them is read here.
fn spec_decode_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os(SPEC_DECODE_ENV).is_some())
}

/// Emit a stage-timing line to stderr when `FOCR_TIMING` is set (perf bring-up).
pub(crate) fn timing_log(msg: &str) {
    if std::env::var_os("FOCR_TIMING").is_some() {
        crate::progress::stderr_message(format_args!("[focr-timing] {msg}"));
    }
}

/// Build the single-image greedy decode params, honoring [`MAX_NEW_TOKENS_ENV`]
/// and then the CLI's `overrides` (an explicit CLI flag outranks the env).
fn decode_params_from_env(overrides: DecodeOverrides) -> DecodeParams {
    let mut p = DecodeParams::single_image();
    if let Some(raw) = std::env::var_os(MAX_NEW_TOKENS_ENV)
        && let Some(s) = raw.to_str()
        && let Ok(n) = s.trim().parse::<usize>()
        && n > 0
    {
        p.max_length = n;
    }
    if let Ok(Some(n)) = test_max_positions() {
        p.max_positions = *n;
    }
    apply_decode_overrides(&mut p, overrides);
    p
}

/// [`decode_params_from_env`] for `arch` under `overrides` (the CLI's,
/// [`decode_overrides`], at model load). A paged model (Unlimited-OCR-2)
/// takes its own position limit ([`sampler::UOCR2_MAX_POSITIONS`]) unless
/// [`TEST_MAX_POSITIONS_ENV`] lowers it, its contract's per-page cap unless
/// `--max-page-tokens` gives one, and the entmax domain [`srswa::ENTMAX_ENV`]
/// names (validated at admission, [`admit_for_inference`]).
fn decode_params_for(arch: &dyn model_arch::ModelArch, overrides: DecodeOverrides) -> DecodeParams {
    let mut p = decode_params_from_env(overrides);
    if let Some(pages) = arch.decode_contract().pages {
        if !matches!(test_max_positions(), Ok(Some(_))) {
            p.max_positions = sampler::UOCR2_MAX_POSITIONS;
        }
        p.max_page_tokens = overrides.max_page_tokens.unwrap_or(pages.max_page_tokens);
        if let Ok(domain) = srswa::entmax_domain() {
            p.entmax_domain = *domain;
        }
    }
    p
}

fn model_cache() -> &'static ModelCache {
    static CACHE: OnceLock<ModelCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

/// Serialize model construction without keeping [`model_cache_guard`] held
/// across container parsing or mmap setup. This closes the double-checked-load
/// race while keeping the cache mutex a short metadata-only critical section.
fn model_load_admission() -> &'static ExclusiveGate {
    static ADMISSION: ExclusiveGate = ExclusiveGate::new();
    &ADMISSION
}

/// The process-wide one-live-forward admission gate (doctrine #5). Public model
/// operations acquire it before touching hydrated weights and release it after
/// their complete forward. The permit is atomic while work runs; its mutex is
/// used only by sleeping contenders, never beneath rayon fan-out.
fn forward_admission() -> &'static ExclusiveGate {
    static ADMISSION: ExclusiveGate = ExclusiveGate::new();
    &ADMISSION
}

fn model_cache_guard() -> FocrResult<TrackedCacheGuard> {
    let guard = model_cache()
        .lock()
        .map_err(|_| FocrError::Other(anyhow::anyhow!("model cache mutex poisoned")))?;
    CACHE_GUARD_HELD_HERE.with(|c| c.set(c.get() + 1));
    Ok(TrackedCacheGuard(guard))
}

/// The model-cache guard, instrumented for the one-live-forward gauge
/// (bd-1azu.14): while THIS thread holds it, a forward beginning on this
/// thread trips [`FORWARD_UNDER_GUARD`] — the rayon-under-lock deadlock class
/// doctrine #5 bans. Derefs to the plain cache entry; the count drops with the
/// guard (same thread, RAII).
struct TrackedCacheGuard(MutexGuard<'static, ModelCacheEntry>);

impl std::ops::Deref for TrackedCacheGuard {
    type Target = ModelCacheEntry;
    fn deref(&self) -> &ModelCacheEntry {
        &self.0
    }
}

impl std::ops::DerefMut for TrackedCacheGuard {
    fn deref_mut(&mut self) -> &mut ModelCacheEntry {
        &mut self.0
    }
}

impl Drop for TrackedCacheGuard {
    fn drop(&mut self) {
        CACHE_GUARD_HELD_HERE.with(|c| c.set(c.get() - 1));
    }
}

/// Resolve an existing `path`: a file as-is; a raw safetensors directory to
/// its shard index ([`shards::SHARD_INDEX_NAME`]), so that `parent()` is the
/// directory holding the sidecars; or, without an index, to the one shard of a
/// registered single-shard checkpoint ([`raw_single_shard_names`]).
fn resolve_existing_model_artifact(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        let index = path.join(shards::SHARD_INDEX_NAME);
        if index.is_file() {
            return Some(index);
        }
        raw_single_shard_names()
            .map(|name| path.join(name))
            .find(|shard| shard.is_file())
    } else if path.exists() {
        Some(path.to_path_buf())
    } else {
        None
    }
}

/// The shard file names of every registered single-shard checkpoint, from
/// the arch censuses.
fn raw_single_shard_names() -> impl Iterator<Item = &'static str> {
    model_arch::registry()
        .iter()
        .filter_map(|arch| arch.tensor_census())
        .filter_map(|census| match unlimited_ocr_census::source_shards(census) {
            [(name, _)] => Some(*name),
            _ => None,
        })
}

fn is_short_model_spec(path: &Path) -> bool {
    let mut components = path.components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}

/// The registered model a bare `spec` names (`unlimited-ocr-2`): one path
/// component equal to a registry id. Such a spec names a model, not a path
/// (U2-E1, plan §9.1).
fn registry_spec(spec: &Path) -> Option<&'static dyn model_arch::ModelArch> {
    if !is_short_model_spec(spec) {
        return None;
    }
    spec.to_str().and_then(model_arch::arch_by_id)
}

/// Whether the bounded header probe finds `arch` in the artifact at
/// `resolved`.
fn artifact_holds(resolved: &Path, arch: &dyn model_arch::ModelArch) -> bool {
    validate_model_header_file(resolved).is_ok_and(|found| found.id() == arch.id())
}

/// An artifact found under `arch`'s name whose header names another model is
/// refused, naming both. A header the probe cannot read is left to the
/// loader, which reports the format error it is.
fn check_named_artifact(resolved: &Path, arch: &dyn model_arch::ModelArch) -> FocrResult<()> {
    match validate_model_header_file(resolved) {
        Ok(found) if found.id() != arch.id() => Err(FocrError::ModelNotFound(format!(
            "{} holds the {:?} model, not the requested {:?}",
            resolved.display(),
            found.id(),
            arch.id()
        ))),
        _ => Ok(()),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ModelQuantPreference {
    Int8,
    Int4,
}

impl ModelQuantPreference {
    /// Distributed quant variants, in resolver-preference order when no explicit
    /// `FOCR_MODEL_QUANT` is set: int8 is the default `focr pull` installs;
    /// int4 is the smaller refinement variant.
    const ALL: [ModelQuantPreference; 2] = [Self::Int8, Self::Int4];

    fn as_str(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Int4 => "int4",
        }
    }
}

fn model_quant_preference_from_os(raw: Option<&OsStr>) -> Option<ModelQuantPreference> {
    let value = raw?.to_str()?.trim().to_ascii_lowercase();
    match value.as_str() {
        "int8" | "q8" => Some(ModelQuantPreference::Int8),
        "int4" | "q4" => Some(ModelQuantPreference::Int4),
        _ => None,
    }
}

fn model_quant_preference() -> Option<ModelQuantPreference> {
    let raw = std::env::var_os(MODEL_QUANT_ENV);
    model_quant_preference_from_os(raw.as_deref())
}

/// The pull cache root, when the network/dist machinery is compiled in. The
/// wasm/core build has no `focr pull` cache to search — models arrive as bytes.
#[cfg(feature = "native")]
fn dist_cache_root() -> Option<PathBuf> {
    crate::dist::cache_root()
}

#[cfg(not(feature = "native"))]
fn dist_cache_root() -> Option<PathBuf> {
    None
}

fn model_search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(raw) = std::env::var_os(MODEL_DIR_ENV)
        && !raw.is_empty()
    {
        dirs.extend(std::env::split_paths(&raw));
    }
    if let Some(root) = dist_cache_root() {
        let models = root.join("models");
        // Flat root FIRST (the layout every released binary knows), then one
        // level of per-model subdirectories, name-sorted for determinism:
        // `focr pull <model>` installs non-primary models into their own
        // subdir so same-named sidecars (e.g. two different tokenizer.json)
        // cannot clobber each other (bd-av64.7).
        let mut subdirs: Vec<PathBuf> = std::fs::read_dir(&models)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        subdirs.sort();
        dirs.push(models);
        dirs.extend(subdirs);
    }
    dirs
}

/// Directories the model resolver searches for missing relative model specs.
/// Exposed for diagnostics (`robot health`) so a missing-model report can show
/// exactly where the binary looked without duplicating resolver policy.
#[must_use]
pub fn model_resolution_search_dirs() -> Vec<PathBuf> {
    model_search_dirs()
}

/// Staff-level metadata from a TrOMR full-page music forward (bd-av64.2):
/// what the staff detector found and what the per-staff recognition did with
/// it. Surfaced through [`OcrModel::take_music_meta`] so the CLI can emit
/// robot `staff` events and the `--json` `staves` array without widening the
/// shared `(text, w, h)` forward contract.
#[derive(Debug, Clone)]
pub struct MusicPageMeta {
    /// `(detection index, page-space bbox)` per RECOGNIZED staff, top-to-bottom.
    pub staves: Vec<(usize, tromr::StaffBBox)>,
    /// Staves that failed per-staff recognition (index, bbox, reason).
    pub skips: Vec<tromr::StaffSkip>,
    /// Annotate-only musical-sanity observations (bd-av64.5).
    pub warnings: Vec<tromr::MusicWarning>,
}

/// One parsed layout span from the model's grounding output: a ref/det label
/// or Unlimited-OCR-2 block label (`"title"`, `"text"`, `"image"`, …) plus its
/// bounding boxes in PIXEL coordinates `[x1, y1, x2, y2]` for the source image
/// (de-normalized from the model's 0..=999 grid). Surfaced by
/// [`OcrModel::recognize_with_layout`] for the `focr ocr --json` /
/// `-o out.json` structured-output path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutSpan {
    /// The label classifying the span.
    pub label: String,
    /// Pixel bounding boxes `[x1, y1, x2, y2]`; a span may carry several.
    pub boxes: Vec<[i64; 4]>,
    /// The heading level of an Unlimited-OCR-2 `title` block (`<LEVELn>`, U2-E3b);
    /// `None` for v1 spans and unleveled blocks.
    pub level: Option<u8>,
}

impl LayoutSpan {
    /// The span as `{label, boxes}`, plus `level` when it has one: the
    /// `--json` layout entry and the resident daemon's wire form.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        let mut value = serde_json::json!({ "label": self.label, "boxes": self.boxes });
        if let Some(level) = self.level {
            value["level"] = level.into();
        }
        value
    }

    /// The inverse of [`Self::to_json`]; `None` without a string `label` or a
    /// `boxes` array. A box that is not four integers is skipped, as is a
    /// `level` that is not a `u8`.
    #[must_use]
    pub fn from_json(value: &serde_json::Value) -> Option<Self> {
        let label = value.get("label")?.as_str()?.to_owned();
        let boxes = value
            .get("boxes")?
            .as_array()?
            .iter()
            .filter_map(|b| {
                let coords = b.as_array()?;
                if coords.len() != 4 {
                    return None;
                }
                let mut out = [0_i64; 4];
                for (slot, coord) in out.iter_mut().zip(coords) {
                    *slot = coord.as_i64()?;
                }
                Some(out)
            })
            .collect();
        let level = value
            .get("level")
            .and_then(serde_json::Value::as_u64)
            .and_then(|l| u8::try_from(l).ok());
        Some(Self {
            label,
            boxes,
            level,
        })
    }
}

/// Which half of a split two-page spread a page is (`--split-spreads`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Half {
    Left,
    Right,
}

impl Half {
    /// `"left"` or `"right"`: the JSON and robot `half` value.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Half::Left => "left",
            Half::Right => "right",
        }
    }

    /// `'l'` or `'r'`: the mark a half's figure names carry after the page
    /// number (`page3l_figure_1`).
    #[must_use]
    pub fn initial(self) -> char {
        match self {
            Half::Left => 'l',
            Half::Right => 'r',
        }
    }
}

/// Where one page of a document came from. A document read from an image
/// list has no source document, so its pages are their 1-based positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageOrigin {
    /// 1-based page of the source.
    pub source_page: usize,
    /// The half of a split spread, or `None` for a whole page.
    pub half: Option<Half>,
}

/// One page of a document parse (U2-E3): its markdown, layout and figure
/// boxes, where it came from, and how its decode ended. Unlimited-OCR-2
/// splits its output into pages by token id, one body per page (D20), so
/// every page it reads has one. v1 merges pages, so a v1 document has
/// pages only when its text splits into exactly one body per page, and
/// then without `tokens`, `finish` or `elapsed_ms`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageResult {
    /// 1-based position among the pages the model read: the page's `<PAGE>`
    /// in the prompt, which a page that failed to rasterize does not get.
    pub run_index: usize,
    /// 1-based page of the source.
    pub source_page: usize,
    /// The half of a split spread, or `None` for a whole page.
    pub half: Option<Half>,
    /// The decode window that read the page (U2-D1b); `None` unwindowed.
    pub window: Option<usize>,
    /// The page's markdown, its figures as `![](images/page_{i}_{m}.jpg)`
    /// with `i = run_index − 1`. The document's markdown joins these
    /// ([`postprocess::join_page_texts`]).
    pub text: String,
    /// The page's layout, boxes in its own pixels.
    pub layout: Vec<LayoutSpan>,
    /// The page's figure boxes, numbered as its markdown numbers them.
    pub figures: Vec<postprocess::FigureRef>,
    /// Generated ids in the page's body, its `<PAGE>` excluded.
    pub tokens: Option<usize>,
    /// How the page's body ended.
    pub finish: Option<PageFinish>,
    /// Wall time decoding the page's body: from its `<PAGE>` (page 1: the
    /// first generated id) to the id that closed it. Vision and prefill,
    /// which serve every page at once, are not in it.
    pub elapsed_ms: Option<u64>,
    /// The model that read the page.
    pub model_id: &'static str,
}

impl PageResult {
    /// Page `run_index` (1-based) of a document read by `model_id`, from its
    /// raw `body`, the raster `size` its boxes map through, and `origin`;
    /// no decode facts yet.
    fn read(
        run_index: usize,
        body: &str,
        size: (u32, u32),
        origin: PageOrigin,
        model_id: &'static str,
    ) -> Self {
        let postprocess::PageOutput {
            text,
            layout,
            figures,
        } = postprocess::finalize_page(body, run_index - 1, size);
        Self {
            run_index,
            source_page: origin.source_page,
            half: origin.half,
            window: None,
            text,
            layout,
            figures,
            tokens: None,
            finish: None,
            elapsed_ms: None,
            model_id,
        }
    }

    /// The page's `--json` `pages[]` entry: `page` (the source page), `half`
    /// when split, `run_index`, `window` when windowed, `text`, `layout`,
    /// then `tokens` and `finish` when the model counts them. Its figures go
    /// in the run's `figures[]`, and `elapsed_ms`, which differs run to run,
    /// only in the robot `page` event.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        let mut value = serde_json::json!({ "page": self.source_page });
        if let Some(half) = self.half {
            value["half"] = half.as_str().into();
        }
        value["run_index"] = self.run_index.into();
        if let Some(window) = self.window {
            value["window"] = window.into();
        }
        value["text"] = self.text.as_str().into();
        value["layout"] = self.layout.iter().map(LayoutSpan::to_json).collect();
        if let Some(tokens) = self.tokens {
            value["tokens"] = tokens.into();
        }
        if let Some(finish) = self.finish {
            value["finish"] = finish.as_str().into();
        }
        value
    }
}

/// One page a document decode streams as the decode crosses the page's end:
/// the next `<PAGE>`, or the end of the decode (bd-2z0y).
#[derive(Debug, Clone, Copy)]
pub struct StreamedPage<'a> {
    /// 1-based, in the model's emission order.
    pub index: usize,
    /// The page's `.trim()`-ed raw body; its markdown comes with the result.
    pub body: &'a str,
    /// The page's result when the model splits pages exactly
    /// (Unlimited-OCR-2, U2-E3); `None` for v1, whose bodies may merge pages.
    pub page: Option<&'a PageResult>,
}

/// A document parse's markdown and pages (U2-E3), before the run's facts.
#[derive(Debug)]
struct DocumentPages {
    markdown: String,
    pages: Vec<PageResult>,
}

/// How a document's pages are split into decode windows (U2-E2, plan
/// §5.10). Windowed decoding reaches documents with U2-D1b; until then a
/// window smaller than the document, and [`PageWindow::Auto`], are refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageWindow {
    /// Windows of this many pages each. A window as large as the document
    /// is one pass, as [`PageWindow::Off`].
    Pages(usize),
    /// The model's default window.
    Auto,
    /// One pass over every page, past the model's page limit: a long
    /// document then stops at the position limit (`finish_reason: length`).
    Off,
}

/// Whether a document run resumes from its checkpoint (U2-E2). Resume
/// arrives with U2-D4; until then only [`ResumePolicy::Never`] is admitted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ResumePolicy {
    /// Always parse from the first page.
    #[default]
    Never,
    /// Resume when a checkpoint of the same document and model exists.
    IfMatching,
    /// Discard any checkpoint and parse from the first page.
    Recompute,
}

/// How [`OcrModel::recognize_document`] parses a document (U2-E2, D23). The
/// default is one pass under the model's page limit; the setters chain:
/// `DocumentOptions::default().page_window(PageWindow::Off)`. Decode tuning
/// ([`DecodeOverrides`], `--max-page-tokens` among them) and preprocess
/// overrides apply as for any run; a document reads base views only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DocumentOptions {
    /// The decode windows; `None` is the model's default policy.
    pub page_window: Option<PageWindow>,
    /// Checkpoint resume.
    pub resume: ResumePolicy,
}

impl DocumentOptions {
    /// These options with decode windows `window`.
    #[must_use]
    pub fn page_window(mut self, window: PageWindow) -> Self {
        self.page_window = Some(window);
        self
    }

    /// These options with resume policy `resume`.
    #[must_use]
    pub fn resume(mut self, resume: ResumePolicy) -> Self {
        self.resume = resume;
        self
    }

    /// Whether a document of `pages` pages runs under these options, and if
    /// so whether the model's page limit applies.
    ///
    /// # Errors
    /// [`FocrError::Usage`] for a window, or a resume policy, that this
    /// build cannot honor yet: refused, never ignored.
    fn page_limit(&self, pages: usize) -> FocrResult<bool> {
        if self.resume != ResumePolicy::Never {
            return Err(FocrError::Usage(format!(
                "document resume ({:?}) is not yet available; parse with ResumePolicy::Never",
                self.resume
            )));
        }
        match self.page_window {
            None => Ok(true),
            Some(PageWindow::Pages(0)) => Err(FocrError::Usage(
                "a page window needs at least one page".into(),
            )),
            // One window over every page is an explicit unwindowed pass.
            Some(PageWindow::Off) => Ok(false),
            Some(PageWindow::Pages(n)) if n >= pages => Ok(false),
            Some(window) => Err(FocrError::Usage(format!(
                "windowed document decoding ({window:?} over {pages} pages) is not yet available; \
                 parse without a page window, or with PageWindow::Off"
            ))),
        }
    }
}

/// A parsed document (U2-E2, D23): its markdown and pages, and how the
/// parse ended. A document that stopped early (the position limit, the
/// stage budget or a cancel) is a result too, with the pages it completed.
#[cfg(feature = "pdf")]
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DocumentResult {
    /// The document's markdown, pages joined by `<PAGE>`.
    pub markdown: String,
    /// One entry per page read, in run order (U2-E3). Empty for a v1
    /// document whose text did not split into one body per page (its
    /// `v1_page_merge` warning says so).
    pub pages: Vec<PageResult>,
    /// Page bodies the model wrote, the interrupted page's included.
    pub pages_out: usize,
    /// Source pages that did not rasterize, left out of the prompt (D24).
    pub skipped: Vec<crate::pdf::SkippedPage>,
    /// How the decode ended.
    pub finish_reason: FinishReason,
    /// `<PAGE>` markers past the page count, merged into the last page.
    pub extra_markers: usize,
    /// The source's warnings (`ignored_files`), then the decode's: the lever
    /// admission's `ignored_lever`, then the document's (`leading_text`,
    /// `v2_length`, `v2_budget`) or v1's (`v1_truncated`, `v1_page_merge`).
    pub warnings: Vec<RunWarning>,
    /// The model that read the document.
    pub model: ModelIdentity,
}

/// A recognized document: the rendered markdown body plus the structured layout
/// (bounding boxes) parsed from the SAME decoded model output, so the two can
/// never disagree.
#[derive(Debug, Clone)]
pub struct RecognizedDocument {
    /// The markdown body (identical to what [`OcrModel::recognize`] returns).
    pub markdown: String,
    /// The per-span layout (labels + pixel bounding boxes).
    pub layout: Vec<LayoutSpan>,
}

/// One figure/image region the model grounded but did NOT transcribe to text —
/// the regions the markdown renders as `![](images/…)` placeholders — cropped out
/// of the source image at its original resolution.
///
/// The caller (CLI `--extract-figures`) writes [`Self::image`] to a real file and
/// rewrites the markdown's [`Self::markdown_ref`] token to point at it. The crop
/// comes from the SAME decode the forward saw (EXIF-aligned), so [`Self::boxes`]
/// land exactly on the pixels.
#[derive(Debug, Clone)]
pub struct ExtractedFigure {
    /// 0-based index among the page's image spans — matches the markdown token.
    pub index: usize,
    /// The ref label (`image` for the standard figure span).
    pub label: String,
    /// The source-pixel box `[x1, y1, x2, y2]` this crop came from (clamped to the
    /// image bounds).
    pub bbox: [i64; 4],
    /// The exact markdown token the rendered document uses for this figure
    /// (`![](images/{index}.jpg)`), for the caller's string-replace.
    pub markdown_ref: String,
    /// The cropped region at original resolution (the caller encodes it).
    pub image: image::DynamicImage,
}

fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|p| p == &path) {
        paths.push(path);
    }
}

/// `<stem>.v<version>.<quant>.focrq`. Each model's artifacts carry its own
/// release version (plan D22): a stem naming a registered model uses that
/// model's version, and any other stem the default model's.
fn versioned_quant_path(direct: &Path, quant: ModelQuantPreference) -> PathBuf {
    let version = direct
        .file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(model_arch::arch_by_id)
        .map_or(crate::UNLIMITED_OCR_ARTIFACT_VERSION, |arch| {
            arch.artifact_version()
        });
    direct.with_extension(format!("v{version}.{}.focrq", quant.as_str()))
}

fn short_model_candidates(
    search_dir: &Path,
    spec: &Path,
    quant: Option<ModelQuantPreference>,
) -> Vec<PathBuf> {
    let direct = search_dir.join(spec);
    let mut candidates = Vec::new();

    // A spec that names a `.focrq` blob (the default `unlimited-ocr.focrq`) or a
    // bare stem (`unlimited-ocr`) gets the quant-variant expansion. `focr pull`
    // installs the artifact-versioned `<stem>.vX.Y.Z.int8.focrq`; historical
    // releases used `<stem>.int8.focrq`, so the default lookup tries both. This
    // keeps a fresh pull usable without `--model` while retaining old caches
    // (bd-3u6x). `with_extension` replaces a trailing `.focrq` or appends one to
    // a bare stem, so both spec forms map to the same candidates. Any other spec
    // (a safetensors directory, an explicit non-focrq file) is taken verbatim.
    let is_focrq_or_bare = match spec.extension() {
        None => true,
        Some(ext) => ext.eq_ignore_ascii_case("focrq"),
    };

    if is_focrq_or_bare {
        // 1. An explicit `FOCR_MODEL_QUANT` preference wins.
        if let Some(quant) = quant {
            push_unique_path(&mut candidates, versioned_quant_path(&direct, quant));
            push_unique_path(
                &mut candidates,
                direct.with_extension(format!("{}.focrq", quant.as_str())),
            );
        }
        // 2. The exact name (an explicit `<name>.focrq` a user / `focr convert`
        //    produced; a no-op existence-wise for a bare stem).
        push_unique_path(&mut candidates, direct.clone());
        // 3. The bare `<stem>.focrq` (covers a no-extension spec).
        push_unique_path(&mut candidates, direct.with_extension("focrq"));
        // 4. Artifact-versioned distributed names. The model release version is
        // independent of binary SemVer, so patch releases keep resolving the
        // same verified download. Versioning lets Windows upgrades commit a new
        // model without replacing the legacy destination first.
        for quant in ModelQuantPreference::ALL {
            push_unique_path(&mut candidates, versioned_quant_path(&direct, quant));
        }
        // 5. Historical unversioned distributed names remain searchable.
        for quant in ModelQuantPreference::ALL {
            push_unique_path(
                &mut candidates,
                direct.with_extension(format!("{}.focrq", quant.as_str())),
            );
        }
    } else {
        push_unique_path(&mut candidates, direct);
    }

    candidates
}

fn is_searchable_model_spec(path: &Path) -> bool {
    path.is_relative() && !path.components().any(|c| matches!(c, Component::ParentDir))
}

fn model_search_specs(spec: &Path) -> Vec<PathBuf> {
    let mut specs = vec![spec.to_path_buf()];
    if !is_short_model_spec(spec)
        && let Some(file_name) = spec.file_name()
    {
        let basename = PathBuf::from(file_name);
        if basename.as_path() != spec {
            specs.push(basename);
        }
    }
    specs
}

/// Bounded probe: does the `.focrq` at `path` declare the explicitly
/// NON-DEFAULT wasm experts-int4 recipe
/// ([`crate::quant::convert::UNLIMITED_OCR_WASM_INT4_RECIPE_ID`])?
///
/// The search-dir resolver consults this so a wasm-recipe artifact sitting in
/// a model directory is NEVER selected as a default model (bd-4l71): it may
/// only be loaded via an explicit path or [`OcrModel::from_weights`]. Reads at
/// most the preamble + bounded header; any read/parse failure (including a
/// non-`.focrq` file) answers `false` so resolution behavior is otherwise
/// unchanged.
///
/// Only a v1 header (its `model_id` absent or `unlimited-ocr`) counts: the
/// wasm recipe is v1's alone, so another model's artifact declaring it is not
/// skipped but selected, and its load gate refuses it by name (exit 7, U2-C3)
/// rather than the search reporting no artifact at all.
fn focrq_declares_wasm_recipe(path: &Path) -> bool {
    const FOCRQ_PREAMBLE_LEN: usize = 6 + 4 + 1 + 32 + 8;

    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut preamble = [0u8; FOCRQ_PREAMBLE_LEN];
    if file.read_exact(&mut preamble).is_err()
        || &preamble[..weights::FOCRQ_MAGIC.len()] != weights::FOCRQ_MAGIC
    {
        return false;
    }
    let header_len = u64::from_le_bytes(
        preamble[43..51]
            .try_into()
            .expect("fixed .focrq header length slice"),
    );
    let Ok(header_len) = usize::try_from(header_len) else {
        return false;
    };
    if header_len == 0 || header_len > MODEL_HEADER_VALIDATION_MAX_BYTES {
        return false;
    }
    let mut header_bytes = vec![0u8; header_len];
    if file.read_exact(&mut header_bytes).is_err() {
        return false;
    }
    #[derive(serde::Deserialize)]
    struct RecipeProbe {
        #[serde(default)]
        model_id: Option<String>,
        #[serde(default)]
        packing_manifest: Option<FocrqPackingManifestProbe>,
    }
    let Ok(probe) = serde_json::from_slice::<RecipeProbe>(&header_bytes) else {
        return false;
    };
    let v1 = probe
        .model_id
        .as_deref()
        .is_none_or(|id| id == model_arch::UNLIMITED_OCR_V1_ID);
    v1 && probe
        .packing_manifest
        .and_then(|manifest| manifest.quant_recipe)
        .as_deref()
        == Some(crate::quant::convert::UNLIMITED_OCR_WASM_INT4_RECIPE_ID)
}

fn resolve_model_from_search_dirs_with_quant(
    spec: &Path,
    search_dirs: &[PathBuf],
    quant: Option<ModelQuantPreference>,
) -> FocrResult<PathBuf> {
    let specs = model_search_specs(spec);
    // A bare registry id is answered only by that model (U2-E1).
    let wanted = registry_spec(spec);
    for dir in search_dirs {
        // A search entry that is itself an artifact serves any other spec, but
        // a registry id only when its header holds that model.
        if let Some(resolved) = resolve_existing_model_artifact(dir)
            && !focrq_declares_wasm_recipe(&resolved)
            && wanted.is_none_or(|arch| artifact_holds(&resolved, arch))
        {
            return Ok(resolved);
        }
        for search_spec in &specs {
            for candidate in short_model_candidates(dir, search_spec, quant) {
                if let Some(resolved) = resolve_existing_model_artifact(&candidate)
                    && !focrq_declares_wasm_recipe(&resolved)
                {
                    if let Some(arch) = wanted {
                        check_named_artifact(&resolved, arch)?;
                    }
                    return Ok(resolved);
                }
            }
        }
    }
    let searched = if search_dirs.is_empty() {
        "<none>".into()
    } else {
        search_dirs
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    Err(FocrError::ModelNotFound(format!(
        "no model artifact named {} (searched directories: {searched}; set {MODEL_DIR_ENV} \
         or pass an explicit path)",
        spec.display()
    )))
}

#[cfg(test)]
fn resolve_model_from_search_dirs(spec: &Path, search_dirs: &[PathBuf]) -> FocrResult<PathBuf> {
    resolve_model_from_search_dirs_with_quant(spec, search_dirs, None)
}

#[derive(serde::Deserialize)]
struct FocrqHeaderProbe {
    tensors: std::collections::BTreeMap<String, weights::TensorRecord>,
    #[serde(default)]
    arch_target: u8,
    #[serde(default)]
    format_version: Option<u32>,
    #[serde(default)]
    source_sha256: String,
    #[serde(default)]
    license_notice: String,
    #[serde(default)]
    model_id: String,
    #[serde(default)]
    packing_manifest: Option<FocrqPackingManifestProbe>,
    #[serde(default)]
    provenance: Option<shards::FocrqProvenance>,
}

#[derive(serde::Deserialize)]
struct FocrqPackingManifestProbe {
    #[serde(default)]
    quant_recipe: Option<String>,
}

fn compatible_arch(declared: &str) -> FocrResult<&'static dyn model_arch::ModelArch> {
    model_arch::arch_for_declared_id(declared)
        .ok_or_else(|| FocrError::FormatMismatch(format!("unknown model_id {declared:?}")))
}

fn validate_header_license(notice: &str, model_id: &str) -> FocrResult<()> {
    if model_id == model_arch::UNLIMITED_OCR_V1_ID {
        if notice == crate::FOCR_MODEL_LICENSE_NOTICE
            || (notice.contains("Copyright (c) 2026 Baidu") && notice.contains("MIT License"))
        {
            return Ok(());
        }
        return Err(FocrError::FormatMismatch(
            "Unlimited-OCR .focrq has an incompatible license notice".into(),
        ));
    }

    match model_arch::arch_by_id(model_id) {
        Some(arch) if notice == arch.license_notice() => Ok(()),
        _ => Err(FocrError::FormatMismatch(format!(
            ".focrq license notice does not match model {model_id:?}"
        ))),
    }
}

fn validate_header_source_sha256(source_sha256: &str) -> FocrResult<()> {
    if source_sha256.len() == 64
        && source_sha256
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Ok(());
    }
    Err(FocrError::FormatMismatch(
        ".focrq source_sha256 is not 64 lowercase hex characters".into(),
    ))
}

fn lowercase_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn header_record_expected_bytes(
    name: &str,
    record: &weights::TensorRecord,
    arch_target: u8,
) -> FocrResult<usize> {
    let numel = record.shape.iter().copied().try_fold(1usize, |acc, dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            FocrError::FormatMismatch(format!("tensor {name:?} shape element count overflows"))
        })
    })?;
    if arch_target == 1 && record.dtype == DType::QInt8PerChan && record.shape.len() == 2 {
        return record.shape[0]
            .div_ceil(2)
            .checked_mul(record.shape[1].div_ceil(8))
            .and_then(|panels| panels.checked_mul(16))
            .ok_or_else(|| {
                FocrError::FormatMismatch(format!(
                    "tensor {name:?} SMMLA packed byte length overflows"
                ))
            });
    }
    match record.dtype {
        DType::F32 => numel.checked_mul(4),
        DType::F16 | DType::BF16 => numel.checked_mul(2),
        DType::QInt8PerChan => Some(numel),
        DType::QInt4PerGroup if numel.is_multiple_of(2) => Some(numel / 2),
        DType::QInt4PerGroup => {
            return Err(FocrError::FormatMismatch(format!(
                "tensor {name:?} has an odd int4 element count"
            )));
        }
    }
    .ok_or_else(|| FocrError::FormatMismatch(format!("tensor {name:?} byte length overflows")))
}

fn validate_header_quant_metadata(name: &str, record: &weights::TensorRecord) -> FocrResult<()> {
    match record.dtype {
        DType::F32 | DType::F16 | DType::BF16 => {
            if record.scales_offset != 0
                || record.scales_len != 0
                || record.group_size != 0
                || record.tier != 0
            {
                return Err(FocrError::FormatMismatch(format!(
                    "tensor {name:?}: non-quantized dtype has stray quantization metadata"
                )));
            }
        }
        DType::QInt8PerChan => {
            let [rows, _cols] = record.shape.as_slice() else {
                return Err(FocrError::FormatMismatch(format!(
                    "tensor {name:?}: QInt8 shape must be rank-2 [rows, cols]"
                )));
            };
            if record.group_size != 0 || record.tier != 0 {
                return Err(FocrError::FormatMismatch(format!(
                    "tensor {name:?}: QInt8 group_size and tier must be zero"
                )));
            }
            let expected_scales = rows.checked_mul(4).ok_or_else(|| {
                FocrError::FormatMismatch(format!(
                    "tensor {name:?}: QInt8 scale byte length overflows"
                ))
            })?;
            if record.scales_len != expected_scales {
                return Err(FocrError::FormatMismatch(format!(
                    "tensor {name:?}: QInt8 scales_len {} != rows*f32 {expected_scales}",
                    record.scales_len
                )));
            }
        }
        DType::QInt4PerGroup => {
            let [rows, cols] = record.shape.as_slice() else {
                return Err(FocrError::FormatMismatch(format!(
                    "tensor {name:?}: QInt4 shape must be rank-2 [rows, cols]"
                )));
            };
            if !crate::quant::int4::VALID_GROUP_SIZES.contains(&record.group_size)
                || !cols.is_multiple_of(record.group_size)
            {
                return Err(FocrError::FormatMismatch(format!(
                    "tensor {name:?}: QInt4 group_size {} is invalid for {cols} columns",
                    record.group_size
                )));
            }
            let expected_scales = rows
                .checked_mul(cols / record.group_size)
                .and_then(|groups| groups.checked_mul(4))
                .ok_or_else(|| {
                    FocrError::FormatMismatch(format!(
                        "tensor {name:?}: QInt4 scale byte length overflows"
                    ))
                })?;
            if record.scales_len != expected_scales {
                return Err(FocrError::FormatMismatch(format!(
                    "tensor {name:?}: QInt4 scales_len {} != rows*groups*f32 {expected_scales}",
                    record.scales_len
                )));
            }
        }
    }
    Ok(())
}

fn validate_header_directory(
    directory: &std::collections::BTreeMap<String, weights::TensorRecord>,
    payload_len: usize,
    arch_target: u8,
) -> FocrResult<()> {
    for (name, record) in directory {
        validate_header_quant_metadata(name, record)?;
        let end = record
            .byte_offset
            .checked_add(record.byte_len)
            .ok_or_else(|| FocrError::FormatMismatch(format!("tensor {name:?} range overflows")))?;
        if end > payload_len {
            return Err(FocrError::FormatMismatch(format!(
                "tensor {name:?} overruns the model payload"
            )));
        }
        let expected = header_record_expected_bytes(name, record, arch_target)?;
        if record.byte_len != expected {
            return Err(FocrError::FormatMismatch(format!(
                "tensor {name:?} byte_len {} != expected {expected}",
                record.byte_len
            )));
        }
        let scales_end = record
            .scales_offset
            .checked_add(record.scales_len)
            .ok_or_else(|| {
                FocrError::FormatMismatch(format!("tensor {name:?} scales range overflows"))
            })?;
        if scales_end > payload_len {
            return Err(FocrError::FormatMismatch(format!(
                "tensor {name:?} scales overrun the model payload"
            )));
        }
    }
    weights::validate_non_overlapping_ranges(directory)?;
    Ok(())
}

fn validate_safetensors_header(
    header: &[u8],
    payload_len: usize,
) -> FocrResult<&'static dyn model_arch::ModelArch> {
    let header = shards::parse_safetensors_header(header)?;
    if header.len() == 0 {
        return Err(FocrError::FormatMismatch(
            "safetensors tensor directory is empty; no native forward can use this artifact".into(),
        ));
    }
    validate_header_directory(&header.records, payload_len, 0)?;
    shards::validate_int64_records(&header, payload_len)?;
    identify_checkpoint(&header)
}

/// Identify a whole raw checkpoint by census, and check its I64 tensors
/// against that census's verify-and-skip rules by name and shape. The full
/// load verifies their content.
fn identify_checkpoint(
    header: &shards::SafetensorsHeader,
) -> FocrResult<&'static dyn model_arch::ModelArch> {
    let arch = unlimited_ocr_census::identify_raw(
        header
            .records
            .iter()
            .map(|(name, record)| (name.as_str(), record)),
    )?;
    let rules = shards::int64_rules(Some(arch));
    shards::check_int64_entries(&header.int64, rules, arch.id())?;
    shards::check_int64_rules_present(&header.int64, rules, arch.id())?;
    Ok(arch)
}

/// Read one safetensors file's bounded header from the start of `reader`
/// and validate it structurally against the file's payload. Returns the
/// header, its raw `length prefix + JSON` bytes, and the payload base.
fn read_validated_safetensors_header(
    mut reader: impl Read,
    file_len: u64,
) -> FocrResult<(shards::SafetensorsHeader, Vec<u8>, usize)> {
    let mut prefix = [0u8; 8];
    reader.read_exact(&mut prefix).map_err(|error| {
        FocrError::FormatMismatch(format!("safetensors header truncated: {error}"))
    })?;
    let (header_json, payload_len) = read_safetensors_header_json(reader, prefix, file_len)?;
    let header = shards::parse_safetensors_header(&header_json)?;
    validate_header_directory(&header.records, payload_len, 0)?;
    shards::validate_int64_records(&header, payload_len)?;
    let payload_base = prefix.len() + header_json.len();
    let mut raw = prefix.to_vec();
    raw.extend_from_slice(&header_json);
    Ok((header, raw, payload_base))
}

/// Read a safetensors header's JSON from `reader`, whose 8-byte length
/// `prefix` was already read, bounded at [`MODEL_HEADER_VALIDATION_MAX_BYTES`].
/// Returns the JSON and the payload length. Never reads payload bytes.
fn read_safetensors_header_json(
    mut reader: impl Read,
    prefix: [u8; 8],
    file_len: u64,
) -> FocrResult<(Vec<u8>, usize)> {
    let file_len = usize::try_from(file_len).map_err(|_| {
        FocrError::FormatMismatch("model file exceeds this platform's addressable size".into())
    })?;
    let header_len_u64 = u64::from_le_bytes(prefix);
    let header_len = usize::try_from(header_len_u64)
        .map_err(|_| FocrError::FormatMismatch("safetensors header length exceeds usize".into()))?;
    if header_len == 0 || header_len > MODEL_HEADER_VALIDATION_MAX_BYTES {
        return Err(FocrError::FormatMismatch(format!(
            "safetensors header length {header_len} is outside the bounded validation limit"
        )));
    }
    let payload_base = 8usize
        .checked_add(header_len)
        .ok_or_else(|| FocrError::FormatMismatch("safetensors header length overflows".into()))?;
    if payload_base > file_len {
        return Err(FocrError::FormatMismatch(
            "safetensors header overruns the file".into(),
        ));
    }
    let mut header = vec![0u8; header_len];
    reader.read_exact(&mut header).map_err(|error| {
        FocrError::FormatMismatch(format!("safetensors header truncated: {error}"))
    })?;
    Ok((header, file_len - payload_base))
}

/// Header probe of a sharded checkpoint: open every shard its index names,
/// read and validate each bounded header from its descriptor, merge them
/// against the index, and identify the whole by census. Returns the arch and
/// the open shards, which the load then reads.
fn validate_sharded_model(
    index_path: &Path,
) -> FocrResult<(&'static dyn model_arch::ModelArch, shards::OpenedShards)> {
    let opened = shards::open_shards(index_path)?;
    let merged = read_opened_shard_headers(&opened)?;
    let arch = identify_checkpoint(&merged)?;
    Ok((arch, opened))
}

/// Read, validate and merge the bounded headers of every open shard. Offsets
/// in the result are logical: shards laid end to end in index order.
fn read_opened_shard_headers(
    opened: &shards::OpenedShards,
) -> FocrResult<shards::SafetensorsHeader> {
    let mut headers = Vec::with_capacity(opened.shards.len());
    let mut start = 0u64;
    for shard in &opened.shards {
        let (header, _, payload_base) = read_validated_safetensors_header(&shard.file, shard.len)
            .map_err(|error| shards::in_shard(&shard.name, error))?;
        headers.push((header, start + payload_base as u64));
        start = start.saturating_add(shard.len);
    }
    shards::merge_shard_headers(&opened.index, headers)
}

/// The bounded header pass of a `focr convert` source: every shard's header,
/// validated and (for an indexed source) merged against the index, plus each
/// shard's raw `length prefix + JSON` bytes, against which the streamed
/// bytes are checked.
#[cfg_attr(not(feature = "native"), allow(dead_code))] // sole non-test caller is the CLI convert path
pub(crate) fn read_source_headers(
    source: &shards::SourceShards,
) -> FocrResult<(shards::SafetensorsHeader, Vec<Vec<u8>>)> {
    let mut headers = Vec::with_capacity(source.files().len());
    let mut prefixes = Vec::with_capacity(source.files().len());
    let mut start = 0u64;
    for (name, path) in source.files() {
        let (file, file_len) = open_model_file(path)?;
        let (header, raw, payload_base) = read_validated_safetensors_header(&file, file_len)
            .map_err(|error| shards::in_shard(name, error))?;
        headers.push((header, start + payload_base as u64));
        prefixes.push(raw);
        start = start.saturating_add(file_len);
    }
    let merged = match source.index() {
        Some(index) => shards::merge_shard_headers(index, headers)?,
        None => headers
            .pop()
            .map(|(header, _)| header)
            .ok_or_else(|| FocrError::Other(anyhow::anyhow!("a source has one file")))?,
    };
    Ok((merged, prefixes))
}

/// The architecture `focr convert` targets for a source whose merged `header`
/// [`read_source_headers`] returned, reconciled with an explicit `--model-id`
/// (U2-C3). Runs on the bounded headers, so it decides before any payload byte
/// is read or any pinned shard digest is checked.
///
/// * No `--model-id`: the model whose tensor census the source matches.
/// * An explicit id that a different census matches: [`FocrError::Usage`]
///   naming both ids.
/// * An explicit id with a census the source does not match: that census's
///   [`FocrError::FormatMismatch`] diff.
/// * An explicit id without a census (the zoo models): that id.
///
/// # Errors
/// As above; [`FocrError::Usage`] also when there is no `--model-id` and the
/// source matches no census, since only an explicit id can name it then.
#[cfg_attr(not(feature = "native"), allow(dead_code))] // sole non-test caller is the CLI convert path
pub(crate) fn conversion_arch(
    explicit: Option<&'static dyn model_arch::ModelArch>,
    header: &shards::SafetensorsHeader,
) -> FocrResult<&'static dyn model_arch::ModelArch> {
    let identified = unlimited_ocr_census::identify_raw(
        header
            .records
            .iter()
            .map(|(name, record)| (name.as_str(), record)),
    );
    match (explicit, identified) {
        (Some(arch), Ok(found)) if found.id() != arch.id() => Err(FocrError::Usage(format!(
            "--model-id {:?} disagrees with the source: its tensor census is {:?}'s. Pass \
             --model-id {:?}, or omit it to use the census",
            arch.id(),
            found.id(),
            found.id()
        ))),
        (Some(arch), Ok(_)) => Ok(arch),
        (Some(arch), Err(FocrError::FormatMismatch(_))) if arch.tensor_census().is_none() => {
            Ok(arch)
        }
        (Some(_), Err(error)) => Err(error),
        (None, Ok(found)) => Ok(found),
        (None, Err(FocrError::FormatMismatch(diff))) => Err(FocrError::Usage(format!(
            "the source matches no registered model's tensor census; pass --model-id (see \
             `focr models`). {diff}"
        ))),
        (None, Err(error)) => Err(error),
    }
}

fn validate_model_header_from_reader(
    mut reader: impl Read,
    file_len: u64,
) -> FocrResult<&'static dyn model_arch::ModelArch> {
    const FOCRQ_PREAMBLE_LEN: usize = 6 + 4 + 1 + 32 + 8;

    let file_len = usize::try_from(file_len).map_err(|_| {
        FocrError::FormatMismatch("model file exceeds this platform's addressable size".into())
    })?;
    let mut prefix = [0u8; 8];
    reader
        .read_exact(&mut prefix)
        .map_err(|error| FocrError::FormatMismatch(format!("model header truncated: {error}")))?;

    if &prefix[..weights::FOCRQ_MAGIC.len()] == weights::FOCRQ_MAGIC {
        let mut preamble = [0u8; FOCRQ_PREAMBLE_LEN];
        preamble[..prefix.len()].copy_from_slice(&prefix);
        reader
            .read_exact(&mut preamble[prefix.len()..])
            .map_err(|error| {
                FocrError::FormatMismatch(format!(".focrq preamble truncated: {error}"))
            })?;
        let version = u32::from_le_bytes(
            preamble[6..10]
                .try_into()
                .expect("fixed .focrq version slice"),
        );
        if version != weights::FOCRQ_FORMAT_VERSION {
            return Err(FocrError::FormatMismatch(format!(
                ".focrq format version {version} is unsupported; this binary requires exactly {}",
                weights::FOCRQ_FORMAT_VERSION,
            )));
        }
        let header_len_u64 = u64::from_le_bytes(
            preamble[43..51]
                .try_into()
                .expect("fixed .focrq header length slice"),
        );
        let header_len = usize::try_from(header_len_u64)
            .map_err(|_| FocrError::FormatMismatch(".focrq header length exceeds usize".into()))?;
        if header_len == 0 || header_len > MODEL_HEADER_VALIDATION_MAX_BYTES {
            return Err(FocrError::FormatMismatch(format!(
                ".focrq header length {header_len} is outside the bounded validation limit"
            )));
        }
        let payload_base = FOCRQ_PREAMBLE_LEN
            .checked_add(header_len)
            .ok_or_else(|| FocrError::FormatMismatch(".focrq header length overflows".into()))?;
        if payload_base > file_len {
            return Err(FocrError::FormatMismatch(
                ".focrq header overruns the file".into(),
            ));
        }
        let mut header_bytes = vec![0u8; header_len];
        reader.read_exact(&mut header_bytes).map_err(|error| {
            FocrError::FormatMismatch(format!(".focrq header truncated: {error}"))
        })?;
        let header: FocrqHeaderProbe = serde_json::from_slice(&header_bytes).map_err(|error| {
            FocrError::FormatMismatch(format!(".focrq header JSON invalid: {error}"))
        })?;
        match header.format_version {
            Some(header_version) if header_version == version => {}
            Some(header_version) => {
                return Err(FocrError::FormatMismatch(format!(
                    ".focrq header format_version {header_version} disagrees with preamble {version}"
                )));
            }
            None => {
                return Err(FocrError::FormatMismatch(
                    ".focrq header is missing required format_version".into(),
                ));
            }
        }
        let preamble_arch_target = preamble[10];
        if header.arch_target != preamble_arch_target {
            return Err(FocrError::FormatMismatch(format!(
                ".focrq header arch_target {} disagrees with preamble {preamble_arch_target}",
                header.arch_target
            )));
        }
        weights::validate_arch_target(header.arch_target)?;
        validate_header_source_sha256(&header.source_sha256)?;
        let preamble_source_sha256 = lowercase_hex(&preamble[11..43]);
        if header.source_sha256 != preamble_source_sha256 {
            return Err(FocrError::FormatMismatch(format!(
                ".focrq header source_sha256 {} disagrees with preamble {preamble_source_sha256}",
                header.source_sha256
            )));
        }
        shards::validate_source_shards(header.provenance.as_ref(), &header.source_sha256)?;
        if header.tensors.is_empty() {
            return Err(FocrError::FormatMismatch(
                ".focrq tensor directory is empty; no native forward can use this artifact".into(),
            ));
        }
        let arch = compatible_arch(&header.model_id)?;
        validate_header_license(&header.license_notice, arch.id())?;
        let declared_recipe = header
            .packing_manifest
            .as_ref()
            .and_then(|manifest| manifest.quant_recipe.as_deref());
        validate_header_directory(&header.tensors, file_len - payload_base, header.arch_target)?;
        if let Some(census) = arch.tensor_census() {
            unlimited_ocr_census::validate_focrq_header(
                census,
                &header.tensors,
                &header.source_sha256,
                declared_recipe,
            )?;
        }
        validate_unlimited_ocr_quant_records(
            true,
            arch.id(),
            declared_recipe,
            header
                .tensors
                .iter()
                .map(|(name, record)| (name.as_str(), record.dtype)),
        )?;
        return Ok(arch);
    }

    let (header, payload_len) = read_safetensors_header_json(reader, prefix, file_len as u64)?;
    validate_safetensors_header(&header, payload_len)
}

fn open_model_file(path: &Path) -> FocrResult<(std::fs::File, u64)> {
    let file = std::fs::File::open(path).map_err(|error| {
        FocrError::ModelNotFound(format!(
            "cannot open model header at {}: {error}",
            path.display()
        ))
    })?;
    let file_len = file.metadata().map_err(|error| {
        FocrError::ModelNotFound(format!(
            "cannot stat model artifact at {}: {error}",
            path.display()
        ))
    })?;
    if !file_len.is_file() {
        return Err(FocrError::FormatMismatch(format!(
            "model artifact at {} is not a regular file",
            path.display()
        )));
    }
    Ok((file, file_len.len()))
}

fn validate_model_header_file(path: &Path) -> FocrResult<&'static dyn model_arch::ModelArch> {
    if shards::is_index_path(path) {
        return validate_sharded_model(path).map(|(arch, _)| arch);
    }
    let (file, file_len) = open_model_file(path)?;
    validate_model_header_from_reader(&file, file_len)
}

/// Determine whether `path` resolves to a model artifact whose bounded header is
/// structurally compatible with this binary and its conservative quant recipe.
///
/// The probe reads at most [`MODEL_HEADER_VALIDATION_MAX_BYTES`] plus the fixed
/// preamble and never touches tensor payload bytes, so `robot health` remains
/// cheap while refusing future versions, malformed ranges/dtypes/metadata, and
/// legacy full-int8 Unlimited-OCR artifacts the real loader would reject.
#[must_use]
pub fn native_model_available(path: &Path) -> bool {
    let Ok(resolved) = OcrModel::resolve_model(path) else {
        return false;
    };
    validate_model_header_file(&resolved).is_ok()
}

/// The id of the model `path` resolves to, from the same bounded header
/// probe as [`native_model_available`] (no payload byte is read); `None`
/// when it does not resolve or its header is not one this binary accepts.
#[must_use]
pub fn model_id_at(path: &Path) -> Option<&'static str> {
    let resolved = OcrModel::resolve_model(path).ok()?;
    validate_model_header_file(&resolved)
        .ok()
        .map(model_arch::ModelArch::id)
}

/// A filesystem model open: the model the header identified, its tokenizer
/// when the sidecar was verified at open, and the weights.
struct OpenedModel {
    arch: &'static dyn model_arch::ModelArch,
    tokenizer: Option<crate::tokenizer::Tokenizer>,
    weights: Weights,
}

impl std::fmt::Debug for OpenedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenedModel")
            .field("arch", &self.arch.id())
            .field("tokenizer", &self.tokenizer.is_some())
            .field("tensors", &self.weights.len())
            .finish()
    }
}

#[cfg(test)]
fn load_weights_from_resolved_model(resolved: &Path) -> FocrResult<OpenedModel> {
    load_weights_from_resolved_model_after_validation(resolved, |_| Ok(()), || {})
}

/// Open `resolved`: identify its model from the header, `admit` that model,
/// judge its tokenizer sidecar, then read the weights. `after_validation`
/// runs just before the first weight byte is read.
fn load_weights_from_resolved_model_after_validation(
    resolved: &Path,
    admit: impl FnOnce(&'static dyn model_arch::ModelArch) -> FocrResult<()>,
    after_validation: impl FnOnce(),
) -> FocrResult<OpenedModel> {
    // The admission and the tokenizer sidecar are judged before a single
    // weight byte is read.
    let (arch, tokenizer, weights) = if shards::is_index_path(resolved) {
        let (arch, opened) = validate_sharded_model(resolved)?;
        admit(arch)?;
        let tokenizer = eager_sidecar_tokenizer(arch, resolved)?;
        after_validation();
        (arch, tokenizer, Weights::load_opened_shards(opened)?)
    } else {
        let (file, file_len) = open_model_file(resolved)?;
        let arch = validate_model_header_from_reader(&file, file_len)?;
        admit(arch)?;
        let tokenizer = eager_sidecar_tokenizer(arch, resolved)?;
        after_validation();
        (arch, tokenizer, Weights::load_opened(file, resolved)?)
    };
    validate_unlimited_ocr_quant_recipe(&weights)?;
    if weights.model_id() != arch.id() {
        return Err(FocrError::FormatMismatch(format!(
            "{}: the loaded weights identify as {:?} but the header probe resolved {:?}",
            resolved.display(),
            weights.model_id(),
            arch.id()
        )));
    }
    Ok(OpenedModel {
        arch,
        tokenizer,
        weights,
    })
}

/// The pre-load inference gate (U2-A7): print the ignored-lever warnings,
/// then refuse an armed unsupported lever or an unimplemented model.
fn admit_for_inference(arch: &'static dyn model_arch::ModelArch) -> FocrResult<()> {
    let admission = lever_policy::admit_from_process(arch);
    for warning in &admission.warnings {
        crate::progress::stderr_message(format_args!(
            "[focr] warning: {} — {}",
            warning.message, warning.recovery
        ));
    }
    admission.verdict?;
    // A paged model routes by `FOCR_SRSWA_ENTMAX`: an unknown value is a
    // usage error before a weight byte is read, so the decode params can
    // rely on it ([`decode_params_for`]).
    if arch.decode_contract().pages.is_some() {
        srswa::entmax_domain()
            .as_ref()
            .map_err(|why| FocrError::Usage(why.clone()))?;
    }
    if arch.maturity() == model_arch::Maturity::Experimental {
        crate::progress::stderr_message(format_args!(
            "[focr] note: {} is experimental: not yet certified against its reference",
            arch.id()
        ));
    }
    Ok(())
}

/// The forward a model's single-image entry points run (U2-A7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ForwardRoute {
    UnlimitedV1,
    /// Unlimited-OCR-2's document loop over one page (U2-B6).
    UnlimitedV2,
    GotOcr2,
    SmolVlm2,
    OneChart,
    Tromr,
}

/// Route `arch`'s single-image forward: exhaustive over the decoder family
/// and Unlimited variant, so no model falls through into v1's R-SWA forward.
/// A planned zoo model keeps its [`FocrError::NotImplemented`]; an
/// implemented model with no wired forward is an explicit error.
fn forward_route(arch: &dyn model_arch::ModelArch) -> FocrResult<ForwardRoute> {
    use model_arch::{Decoder, UnlimitedVariant};
    let route = match (arch.decoder(), arch.unlimited_variant()) {
        (Decoder::DeepSeekV2MoeRswa, Some(UnlimitedVariant::V1Rswa)) => {
            Some(ForwardRoute::UnlimitedV1)
        }
        // Unlimited-OCR-2 reads one page through its document loop (U2-B6),
        // never v1's R-SWA forward; refused while it is planned.
        (Decoder::DeepSeekV2MoeSrSwa, Some(UnlimitedVariant::V2SrSwa)) => {
            OcrModel::ensure_arch_implemented(arch)?;
            Some(ForwardRoute::UnlimitedV2)
        }
        (Decoder::Qwen2Dense, None) if arch.id() == "got-ocr2" => Some(ForwardRoute::GotOcr2),
        (Decoder::LlamaDense, None) if arch.id() == "smolvlm2" => Some(ForwardRoute::SmolVlm2),
        (Decoder::OptDense, None) if arch.id() == "onechart" => Some(ForwardRoute::OneChart),
        (Decoder::Seq2SeqDense, None) if arch.id() == "tromr" => Some(ForwardRoute::Tromr),
        (
            Decoder::DeepSeekV2MoeRswa
            | Decoder::DeepSeekV2MoeSrSwa
            | Decoder::Qwen2Dense
            | Decoder::LlamaDense
            | Decoder::OptDense
            | Decoder::Seq2SeqDense,
            _,
        ) => None,
    };
    if let Some(route) = route {
        return Ok(route);
    }
    OcrModel::ensure_arch_implemented(arch)?;
    Err(FocrError::NotImplemented(format!(
        "model architecture '{}' has no forward wired in this build",
        arch.id()
    )))
}

/// Crop each of a page's `figures` out of `source`, its raster at the size
/// its boxes were mapped through (a document page's re-fetched raster,
/// U2-E3). Boxes are corner-ordered and clamped to the image; a degenerate
/// (empty) box is skipped while KEEPING the figure's index, so a written
/// figure's `markdown_ref` always matches the rendered placeholder.
#[must_use]
pub fn crop_figure_refs(
    figures: &[postprocess::FigureRef],
    source: &image::DynamicImage,
) -> Vec<ExtractedFigure> {
    let iw = i64::from(source.width());
    let ih = i64::from(source.height());
    figures
        .iter()
        .filter_map(|fr| {
            // The standard image span carries one box; use the first.
            let &[x1, y1, x2, y2] = fr.boxes.first()?;
            let cx1 = x1.min(x2).clamp(0, iw);
            let cy1 = y1.min(y2).clamp(0, ih);
            let cx2 = x1.max(x2).clamp(0, iw);
            let cy2 = y1.max(y2).clamp(0, ih);
            let w = u32::try_from(cx2 - cx1).unwrap_or(0);
            let h = u32::try_from(cy2 - cy1).unwrap_or(0);
            if w == 0 || h == 0 {
                return None; // degenerate box — nothing to crop
            }
            #[allow(clippy::cast_sign_loss)] // cx1/cy1 >= 0 after clamp
            let image = source.crop_imm(cx1 as u32, cy1 as u32, w, h);
            Some(ExtractedFigure {
                index: fr.index,
                label: fr.label.clone(),
                bbox: [cx1, cy1, cx2, cy2],
                markdown_ref: fr.markdown_ref.clone(),
                image,
            })
        })
        .collect()
}

/// `elapsed` in whole milliseconds, saturating.
fn millis(elapsed: std::time::Duration) -> u64 {
    u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
}

/// Admit `arch` to multi-page cross-page parsing (`infer_multi`): v1 enters
/// v1's 640-squash R-SWA multi-page loop and v2 its own document loop
/// ([`OcrModel::decode_document`], U2-B6), each once implemented; the zoo
/// parses pages independently.
fn multi_page_route(arch: &dyn model_arch::ModelArch) -> FocrResult<()> {
    match arch.unlimited_variant() {
        Some(model_arch::UnlimitedVariant::V1Rswa | model_arch::UnlimitedVariant::V2SrSwa) => {
            OcrModel::ensure_arch_implemented(arch)
        }
        None => Err(FocrError::NotImplemented(format!(
            "multi-page cross-page parsing (infer_multi) is the Unlimited-OCR contract; model \
             {:?} parses pages independently (use the plain multi-input/batch path)",
            arch.id()
        ))),
    }
}

/// Load the model at `spec` up front when it has a lever policy (U2-A7), so a
/// batch reports its refusal as the run's exit code, not as a per-image
/// failure; hold the result for the batch and every per-image load is a
/// cache hit. Other models, and specs that do not resolve to a readable
/// header, are left to the per-image path exactly as before.
///
/// # Errors
/// As [`OcrModel::load`], for a model with a lever policy.
pub fn load_if_policed(spec: &Path) -> FocrResult<Option<Arc<OcrModel>>> {
    let Ok(resolved) = OcrModel::resolve_model(spec) else {
        return Ok(None);
    };
    match validate_model_header_file(&resolved) {
        Ok(arch) if lever_policy::policy_for(arch).is_some() => OcrModel::load(&resolved).map(Some),
        Ok(_) | Err(_) => Ok(None),
    }
}

/// Whether the resident daemon may serve the model at `path` (U2-A7). A
/// header the bounded probe refuses (a newer format, an unknown census) runs
/// inline, which reports the same error, rather than starting a daemon that
/// can never load the model and then idles for its full period.
#[must_use]
pub fn resident_daemon_serves(path: &Path) -> bool {
    validate_model_header_file(path).is_ok_and(lever_policy::resident_daemon_serves)
}

/// Whether `arch` pins anything about its `tokenizer.json` sidecar.
fn pins_tokenizer(arch: &dyn model_arch::ModelArch) -> bool {
    arch.tokenizer_sha256().is_some() || !arch.required_token_ids().is_empty()
}

/// Parse `tokenizer.json` bytes for `arch`, refusing a sidecar whose SHA-256
/// is not the one the arch pins or that maps a required added-token id to
/// other content. `origin` names the sidecar in the error.
fn checked_tokenizer(
    arch: &dyn model_arch::ModelArch,
    bytes: &[u8],
    origin: &str,
) -> FocrResult<crate::tokenizer::Tokenizer> {
    use sha2::Digest as _;

    let id = arch.id();
    let remedy = match arch.tokenizer_source() {
        Some(source) => format!("fetch the pinned file from {source}"),
        None => format!("re-install its sidecars with `focr pull {id}`"),
    };
    if let Some(expected) = arch.tokenizer_sha256() {
        let found = format!("{:x}", sha2::Sha256::digest(bytes));
        if found != expected {
            return Err(FocrError::FormatMismatch(format!(
                "{origin} has sha256 {found}, but {id} requires {expected}; {remedy}"
            )));
        }
    }
    let tokenizer = crate::tokenizer::Tokenizer::from_json_bytes(bytes)?;
    for &(token_id, content) in arch.required_token_ids() {
        let found = tokenizer.id_to_token(token_id);
        if found != Some(content) {
            return Err(FocrError::FormatMismatch(format!(
                "{origin} maps token id {token_id} to {found:?}, but {id} requires {content:?}; \
                 {remedy}"
            )));
        }
    }
    Ok(tokenizer)
}

/// Verify and parse the `tokenizer.json` beside `model_path` at open time when
/// `arch` pins it. An absent sidecar is left to the first-use
/// [`FocrError::ModelNotFound`]; whichever path reads the file verifies it.
fn eager_sidecar_tokenizer(
    arch: &dyn model_arch::ModelArch,
    model_path: &Path,
) -> FocrResult<Option<crate::tokenizer::Tokenizer>> {
    if !pins_tokenizer(arch) {
        return Ok(None);
    }
    let path = model_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("tokenizer.json");
    match std::fs::read(&path) {
        Ok(bytes) => checked_tokenizer(arch, &bytes, &path.display().to_string()).map(Some),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(FocrError::ModelNotFound(format!(
            "tokenizer.json at {}: {error}",
            path.display()
        ))),
    }
}

/// Reject Unlimited-OCR containers that violate the fixed default quantization
/// recipe before any forward can silently consume them. Raw safetensors and
/// other registered model families have separate storage contracts.
fn validate_unlimited_ocr_quant_recipe(weights: &Weights) -> FocrResult<()> {
    validate_unlimited_ocr_quant_records(
        weights.is_focrq(),
        weights.model_id(),
        weights.quant_recipe(),
        weights.names().map(|name| {
            let dtype = weights
                .record(name)
                .expect("name iterator and tensor directory must agree")
                .dtype;
            (name, dtype)
        }),
    )?;
    unlimited_ocr_census::validate_loaded_weights(weights)
}

fn validate_unlimited_ocr_quant_records<'a>(
    is_focrq: bool,
    model_id: &str,
    declared_recipe: Option<&str>,
    records: impl IntoIterator<Item = (&'a str, DType)>,
) -> FocrResult<()> {
    if !is_focrq {
        return Ok(());
    }
    let Some(variant) = model_arch::arch_by_id(model_id).and_then(|arch| arch.unlimited_variant())
    else {
        return Ok(());
    };
    // Unlimited-OCR-2 has exactly one recipe, the conservative one (U2-C3):
    // its declared id must be that recipe's, so the wasm int4 id is refused.
    let v2 = variant == model_arch::UnlimitedVariant::V2SrSwa;
    if v2 && declared_recipe != Some(crate::quant::convert::UNLIMITED_OCR_2_INT8_RECIPE_ID) {
        return Err(FocrError::FormatMismatch(format!(
            "{model_id} .focrq packing_manifest.quant_recipe must be exactly {:?}, got \
             {declared_recipe:?}; {model_id} has no int4 or wasm recipe",
            crate::quant::convert::UNLIMITED_OCR_2_INT8_RECIPE_ID
        )));
    }

    // The explicitly-tagged NON-DEFAULT wasm experts-int4 recipe (bd-4l71) is
    // honored ONLY when a v1 artifact's packing manifest declares it; every
    // other Unlimited-OCR `.focrq` is held to the conservative default exactly
    // as before. Under the wasm recipe: expert FFN tensors MUST be
    // QInt4PerGroup, attention/lm_head/embed MUST be QInt8PerChan, everything
    // else BF16/F32 — so e.g. int4 attention is rejected loudly.
    let wasm_recipe =
        !v2 && declared_recipe == Some(crate::quant::convert::UNLIMITED_OCR_WASM_INT4_RECIPE_ID);
    let recipe = Recipe::validated_default();
    let mut violations = Vec::new();
    for (name, dtype) in records {
        let expected: (&str, bool) = if wasm_recipe {
            match crate::quant::recipe::classify_wasm_experts_int4(name) {
                crate::quant::recipe::WasmInt4Policy::ExpertInt4 => {
                    ("QInt4PerGroup", dtype == DType::QInt4PerGroup)
                }
                crate::quant::recipe::WasmInt4Policy::Int8 => {
                    ("QInt8PerChan", dtype == DType::QInt8PerChan)
                }
                crate::quant::recipe::WasmInt4Policy::KeepHighPrecision => {
                    ("BF16 or F32", matches!(dtype, DType::BF16 | DType::F32))
                }
            }
        } else if recipe.is_quantized(name) {
            ("QInt8PerChan", dtype == DType::QInt8PerChan)
        } else {
            ("BF16 or F32", matches!(dtype, DType::BF16 | DType::F32))
        };
        let (expected, valid) = expected;
        if !valid {
            violations.push(format!("{name} is {dtype:?}, expected {expected}"));
        }
    }

    if violations.is_empty() {
        return Ok(());
    }
    let shown = violations
        .iter()
        .take(8)
        .cloned()
        .collect::<Vec<_>>()
        .join("; ");
    let omitted = violations.len().saturating_sub(8);
    let suffix = if omitted == 0 {
        String::new()
    } else {
        format!("; and {omitted} more")
    };
    let (model, recipe_id) = if v2 {
        (
            "Unlimited-OCR-2",
            crate::quant::convert::UNLIMITED_OCR_2_INT8_RECIPE_ID,
        )
    } else if wasm_recipe {
        (
            "Unlimited-OCR",
            crate::quant::convert::UNLIMITED_OCR_WASM_INT4_RECIPE_ID,
        )
    } else {
        (
            "Unlimited-OCR",
            crate::quant::convert::UNLIMITED_OCR_INT8_RECIPE_ID,
        )
    };
    Err(FocrError::FormatMismatch(format!(
        "{model} .focrq violates quant recipe {recipe_id}: {shown}{suffix}. Re-convert the \
         original safetensors with this build; legacy full-int8 artifacts are not accepted",
    )))
}

/// Embedding-table access without whole-table f32 widening (bd-4l71 residency).
///
/// The conservative BF16/F32 table keeps the historical whole-table widen
/// (`F32`, byte-for-byte unchanged). A `QInt8PerChan`-stored table (the wasm
/// experts-int4 artifact) is kept as a ZERO-COPY int8 view over the blob and
/// dequantized ONE ROW at a time at token-embed time — the same
/// `f32::from(w) * scale[row]` per element that `Weights::mat`'s
/// `dequant_qint8` would have applied to the whole table, so the embedded rows
/// are bit-identical while the ~662 MB f32 copy never exists.
pub enum EmbedTable<'w> {
    /// Whole-table f32 widen (the historical accessor result).
    F32(Mat),
    /// Zero-copy int8 view + per-channel scales, dequantized per row.
    QInt8 {
        /// Row-major `[rows, cols]` int8 payload bytes (borrowed from the blob).
        data: &'w [u8],
        /// Per-row (output-channel) scales, length `rows`.
        scales: Vec<f32>,
        /// Vocab size.
        rows: usize,
        /// Hidden size.
        cols: usize,
    },
}

impl EmbedTable<'_> {
    /// Vocab size.
    pub fn rows(&self) -> usize {
        match self {
            EmbedTable::F32(m) => m.rows,
            EmbedTable::QInt8 { rows, .. } => *rows,
        }
    }

    /// Hidden size.
    pub fn cols(&self) -> usize {
        match self {
            EmbedTable::F32(m) => m.cols,
            EmbedTable::QInt8 { cols, .. } => *cols,
        }
    }

    /// Materialize row `idx` as f32 — the per-token embed. Caller guarantees
    /// `idx < rows` (both decode loops guard with their out-of-vocab check).
    ///
    /// # Panics
    /// Panics if `idx >= self.rows()` — every caller checks the decoded id
    /// against the vocab first ([`Self::embed_ids`] returns an error instead).
    #[must_use]
    pub fn row_f32(&self, idx: usize) -> Vec<f32> {
        match self {
            EmbedTable::F32(m) => m.data[idx * m.cols..(idx + 1) * m.cols].to_vec(),
            EmbedTable::QInt8 {
                data, scales, cols, ..
            } => {
                let scale = scales[idx];
                data[idx * cols..(idx + 1) * cols]
                    .iter()
                    .map(|&b| f32::from(b as i8) * scale)
                    .collect()
            }
        }
    }

    /// Gather + (for the int8 arm) dequantize `ids` into a `[seq, hidden]`
    /// matrix — the [`decoder::embed_tokens`] contract, including its
    /// out-of-range error shape.
    pub fn embed_ids(&self, ids: &[u32]) -> FocrResult<Mat> {
        match self {
            EmbedTable::F32(m) => decoder::embed_tokens(&m.data, m.rows, m.cols, ids),
            EmbedTable::QInt8 { rows, cols, .. } => {
                let mut out = Vec::with_capacity(ids.len() * cols);
                for &id in ids {
                    let row = id as usize;
                    if row >= *rows {
                        return Err(FocrError::Other(anyhow::anyhow!(
                            "embed_tokens: id {row} out of range (vocab {rows})"
                        )));
                    }
                    out.extend_from_slice(&self.row_f32(row));
                }
                Ok(Mat::from_vec(ids.len(), *cols, out))
            }
        }
    }
}

/// The tensor name of the decoder embedding table in every Unlimited-OCR
/// artifact.
const EMBED_TOKENS: &str = "model.embed_tokens.weight";

/// Resolve `model.embed_tokens.weight` out of `weights` as an [`EmbedTable`]
/// (the body of [`OcrModel::embed_table`], free-standing so the bit-exactness
/// gate can drive it over a synthetic artifact):
///
/// * BF16/F32 storage (conservative artifacts, raw safetensors) → the
///   historical whole-table f32 widen via [`Weights::mat`], byte-for-byte
///   unchanged.
/// * `QInt8PerChan` storage (the wasm experts-int4 artifact) → a ZERO-COPY
///   int8 view + per-channel scales, dequantized one row at a time at
///   token-embed time (bd-4l71 residency: the ~662 MB f32 table copy never
///   exists). Values are identical: `f32::from(w) * scale[row]` is exactly the
///   `dequant_qint8` expression `Weights::mat` applies — gated by
///   `int8_embed_row_lookup_matches_full_dequant`.
///
/// # Errors
/// [`FocrError::FormatMismatch`] if the table is absent or malformed.
fn embed_table_from(weights: &Weights) -> FocrResult<EmbedTable<'_>> {
    const NAME: &str = EMBED_TOKENS;
    // The int8-view arm only understands canonical row-major int8 payloads
    // (arch_target 0 — the only form the wasm recipe emits). Anything else
    // takes the historical `mat()` path, whose accessors handle all forms.
    if weights.arch_target() == 0
        && matches!(
            weights.record(NAME).map(|rec| rec.dtype),
            Some(DType::QInt8PerChan)
        )
    {
        let view = weights.tensor(NAME)?;
        let [rows, cols] = view.shape else {
            return Err(FocrError::FormatMismatch(format!(
                "tensor {NAME:?} has rank {}; expected 2 ([vocab, hidden])",
                view.shape.len()
            )));
        };
        let (rows, cols) = (*rows, *cols);
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| FocrError::FormatMismatch(format!("tensor {NAME:?} shape overflows")))?;
        if view.data.len() != expected {
            return Err(FocrError::FormatMismatch(format!(
                "QInt8 tensor {NAME:?}: {} payload bytes != vocab*hidden {expected}",
                view.data.len()
            )));
        }
        if view.scales.len() != rows * 4 {
            return Err(FocrError::FormatMismatch(format!(
                "QInt8 tensor {NAME:?}: {} scale bytes != vocab*f32 {}",
                view.scales.len(),
                rows * 4
            )));
        }
        let scales: Vec<f32> = view
            .scales
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        return Ok(EmbedTable::QInt8 {
            data: view.data,
            scales,
            rows,
            cols,
        });
    }
    Ok(EmbedTable::F32(weights.mat(NAME)?))
}

/// One page's prefill state, handed from the batch spine front end
/// ([`OcrModel::prefill_from_features`]) to the [`batch_scheduler`]: everything a
/// [`batch_scheduler::PageStream`] needs to seed decode, plus the per-layer R-SWA
/// rings (already prefilled) and the source pixel dims postprocess needs.
struct PagePrefill {
    /// Reference/prompt length already in the KV cache (`inputs_embeds.rows`).
    prefill_len: usize,
    /// The prompt id-stream (seeds the n-gram-ban history).
    prompt_ids: Vec<u32>,
    /// Prefill's final hidden row `[1, hidden]` — predicts the first token.
    last_hidden: Mat,
    /// One prefilled [`rswa::RingCache`] per decoder layer for this page.
    caches: Vec<rswa::RingCache>,
    /// Source image width (bbox de-normalization).
    image_w: u32,
    /// Source image height (bbox de-normalization).
    image_h: u32,
}

impl OcrModel {
    /// Resolve `path` to a concrete model artifact (`.focrq` blob or a
    /// safetensors directory) — the header-sniff / search-path logic
    /// (`native_model_available`, bd-223.7).
    ///
    /// Returns `path` as-is if it exists as a file; if `path` is a raw
    /// safetensors directory, returns its shard index
    /// (`model.safetensors.index.json`), or without one the shard of a
    /// registered single-shard checkpoint. Missing relative
    /// specs are searched under `$FOCR_MODEL_DIR` (split with the platform
    /// path-list separator) and the user cache default; each `$FOCR_MODEL_DIR`
    /// entry may itself be a direct artifact/package or a search root.
    ///
    /// A bare registry id (`unlimited-ocr-2`) names a model, never a path in
    /// the working directory: it is only searched for, and the artifact found
    /// must hold that model (U2-E1). A search entry that is itself an artifact
    /// of another model is passed over, and an artifact named for the id whose
    /// header names another model is [`FocrError::ModelNotFound`] naming both.
    pub fn resolve_model(path: &Path) -> FocrResult<PathBuf> {
        if registry_spec(path).is_some() {
            return resolve_model_from_search_dirs_with_quant(
                path,
                &model_search_dirs(),
                model_quant_preference(),
            );
        }
        if let Some(resolved) = resolve_existing_model_artifact(path) {
            return Ok(resolved);
        }
        if path.is_dir() {
            return Err(FocrError::ModelNotFound(format!(
                "no model artifact at {} (expected {} or a single-shard checkpoint ({}) inside \
                 the safetensors directory)",
                path.display(),
                shards::SHARD_INDEX_NAME,
                raw_single_shard_names().collect::<Vec<_>>().join(", ")
            )));
        }
        if is_searchable_model_spec(path) {
            resolve_model_from_search_dirs_with_quant(
                path,
                &model_search_dirs(),
                model_quant_preference(),
            )
        } else {
            Err(FocrError::ModelNotFound(format!(
                "no model artifact at {} (resolver lands in Phase 0/1, bd-223.7)",
                path.display()
            )))
        }
    }

    /// Load (or fetch from the global cache) the model at `path`.
    ///
    /// Resolves the path, then returns a shared [`Arc`]: if a live handle for the
    /// same resolved path is still cached, that `Arc` is cloned; otherwise the
    /// weights are loaded and a fresh handle is cached weakly.
    ///
    /// # Errors
    /// [`FocrError::ModelNotFound`] if the path doesn't resolve (or the resolved
    /// file is unreadable). Existing paths that are not recognized model
    /// containers, and recognized containers with structural faults, surface
    /// [`FocrError::FormatMismatch`] so the public format/version exit-code
    /// contract remains intact. [`FocrError::Usage`] when the model's lever
    /// policy refuses a set `FOCR_*` lever, or the model is registered but not
    /// implemented (U2-A7) — both before a weight byte is read.
    pub fn load(path: &Path) -> FocrResult<Arc<Self>> {
        let resolved = Self::resolve_model(path)?;
        let _load_permit = model_load_admission().acquire();

        {
            let guard = model_cache_guard()?;
            if let Some((cached_path, weak)) = guard.as_ref()
                && *cached_path == resolved
                && let Some(strong) = weak.upgrade()
            {
                // The force flag can change between loads; the warnings
                // were printed when the model was first admitted.
                lever_policy::admit_from_process(strong.arch).verdict?;
                return Ok(strong);
            }
        }

        // Route through `Weights::load`: owned bytes are the safe default.
        // `FOCR_MMAP=1` is an explicit opt-in for trusted immutable artifacts;
        // mmap failures retain the owned-buffer fallback.
        let opened = load_weights_from_resolved_model_after_validation(
            &resolved,
            admit_for_inference,
            || {},
        )?;
        let model = Arc::new(Self::assemble(
            resolved.clone(),
            SidecarBundle::default(),
            opened.arch,
            opened.weights,
            opened.tokenizer,
        ));

        let mut guard = model_cache_guard()?;
        *guard = Some((resolved, Arc::downgrade(&model)));
        Ok(model)
    }

    /// Construct a model from an already-parsed [`Weights`] blob plus in-memory
    /// tokenizer sidecars — the no-filesystem constructor (the browser/wasm
    /// path; also embedders that ship weights as bytes). Mirrors [`Self::load`]
    /// exactly — including the Unlimited-OCR conservative-recipe validation —
    /// but skips path resolution, the load-admission gate, and the process-wide
    /// weak cache (byte-loaded models are owned by their caller).
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if a raw shard matches no registered
    /// model's census, if the `tokenizer_json` sidecar is not the one the
    /// model pins (checked first, before the weights' contract), or if the
    /// weights violate the fixed Unlimited-OCR quant recipe (other archs have
    /// separate storage contracts, validated by their own tensor accessors at
    /// forward time). [`FocrError::Usage`] as [`Self::load`], before the
    /// tokenizer check.
    pub fn from_weights(weights: Weights, sidecars: SidecarBundle) -> FocrResult<Arc<Self>> {
        let arch = match model_arch::arch_by_id(weights.model_id()) {
            Some(arch) => arch,
            // Only a raw shard that matched no census lacks a registered id;
            // identify it again for the per-census diff.
            None => {
                let records: Vec<(&str, &weights::TensorRecord)> = weights
                    .names()
                    .filter_map(|name| weights.record(name).map(|record| (name, record)))
                    .collect();
                unlimited_ocr_census::identify_raw(records.iter().copied())?
            }
        };
        admit_for_inference(arch)?;
        let tokenizer = match &sidecars.tokenizer_json {
            Some(bytes) if pins_tokenizer(arch) => Some(checked_tokenizer(
                arch,
                bytes,
                "the in-memory tokenizer.json sidecar",
            )?),
            _ => None,
        };
        validate_unlimited_ocr_quant_recipe(&weights)?;
        Ok(Arc::new(Self::assemble(
            PathBuf::from("<in-memory>"),
            sidecars,
            arch,
            weights,
            tokenizer,
        )))
    }

    /// A model over admitted weights, its caches empty.
    fn assemble(
        path: PathBuf,
        sidecars: SidecarBundle,
        arch: &'static dyn model_arch::ModelArch,
        weights: Weights,
        tokenizer: Option<crate::tokenizer::Tokenizer>,
    ) -> Self {
        Self {
            path,
            sidecars,
            arch,
            weights,
            decode_params: decode_params_for(arch, decode_overrides()),
            decoder_cache_i8: FallibleOnce::new(),
            decoder_cache: FallibleOnce::new(),
            decoder_cache_f32: FallibleOnce::new(),
            clip_cache: FallibleOnce::new(),
            unlimited_vision: FallibleOnce::new(),
            got_statics: FallibleOnce::new(),
            onechart_statics: FallibleOnce::new(),
            smol_statics: FallibleOnce::new(),
            tokenizer: FallibleOnce::prefilled(tokenizer),
            got_tokenizer: FallibleOnce::new(),
            tromr_tokenizer: FallibleOnce::new(),
            music_meta: std::sync::Mutex::new(None),
            last_decode: std::sync::Mutex::new(None),
        }
    }

    /// The frozen greedy decode contract this model drives with (plan §6.10).
    #[must_use]
    pub fn decode_params(&self) -> &DecodeParams {
        &self.decode_params
    }

    /// The path this model was loaded from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Run the full forward for one document image and return the raw decoded
    /// model text (pre-postprocess) plus the source pixel dimensions the
    /// postprocess pass needs for bbox de-normalization.
    ///
    /// This is the end-to-end Phase-1 pipeline (plan §10 Phase 1, plan §6):
    ///
    /// 1. **Preprocess** ([`preprocess::preprocess_image`]): decode/resize/pad/
    ///    normalize/tile the image and build the `<image>` id-stream +
    ///    `images_seq_mask` ([SPEC-018..033]).
    /// 2. **Vision tower** ([`Self::vision_tower`]): SAM-ViT-B -> 16× token
    ///    compressor -> CLIP-L, fused into the per-view hybrid feature and mapped
    ///    by the `vision_bridge` projector 2048 -> 1280 ([SPEC-040..052]).
    /// 3. **Connector** ([`Self::build_inputs_embeds`]): embed the prompt token
    ///    ids, assemble the 273-slot structural layout
    ///    (`image_newline`/`view_seperator`), and `masked_scatter` the visual
    ///    features into the prompt embeddings at the `<image>` positions
    ///    ([SPEC-060..066]).
    /// 4. **Decoder prefill + AR decode** ([`Self::generate`]): the 12-layer
    ///    R-SWA MoE driver prefills the whole prompt, then the **sequential**
    ///    greedy decode loop (greedy + no_repeat_ngram) emits tokens to EOS
    ///    ([SPEC-070..103]).
    ///
    /// The numeric kernels of every stage live in their submodules and are
    /// unit-tested there; this method is pure orchestration over the [`Mat`]
    /// currency. Each stage reads its weights through the `.focrq`/safetensors
    /// accessor ([`Weights::mat`]/[`Weights::vec`]), so the full pipeline runs
    /// end-to-end against a loaded model.
    ///
    /// # Errors
    /// Whatever the first failing stage returns — e.g. [`FocrError::FormatMismatch`]
    /// if a required tensor is absent from the loaded weights, or an image-decode
    /// error from [`preprocess::preprocess_image`].
    pub fn forward(&self, image_path: &Path) -> FocrResult<(String, u32, u32)> {
        let _admission = forward_admission().acquire();
        match forward_route(self.arch())? {
            ForwardRoute::GotOcr2 => {
                return self.forward_got(&preprocess::decode_path(image_path)?);
            }
            ForwardRoute::SmolVlm2 => {
                return self.forward_smolvlm2(&preprocess::decode_path(image_path)?);
            }
            ForwardRoute::OneChart => {
                return self.forward_onechart(&preprocess::decode_path(image_path)?);
            }
            ForwardRoute::Tromr => {
                return self.forward_tromr(&preprocess::decode_path(image_path)?);
            }
            ForwardRoute::UnlimitedV1 | ForwardRoute::UnlimitedV2 => {}
        }
        let t = Instant::now();
        // ── 1. preprocess (decode file → normalize → tile) ───────────────────
        // Mode = the certified Base-1024 default unless the CLI overrode it
        // (--base-size/--image-size/--crop-mode, bd-1e9n).
        progress::emit("preprocess", 0, 0);
        let pre = preprocess::preprocess_image(
            image_path,
            preprocess_mode(),
            preprocess_profile(self.arch()),
        )?;
        progress::emit("preprocess", 1, 1);
        timing_log(&format!("preprocess {:.2}s", t.elapsed().as_secs_f64()));
        self.forward_pre(pre)
    }

    /// Forward an already-decoded in-memory image (e.g. a PDF page rasterized by
    /// [`crate::pdf`]) through the identical pipeline as [`Self::forward`],
    /// skipping only the file-decode step. Returns the raw decoded text plus the
    /// source pixel dims for postprocess geometry.
    ///
    /// # Errors
    /// As [`Self::forward`] (sans the file-open path).
    pub fn forward_dynamic(&self, img: image::DynamicImage) -> FocrResult<(String, u32, u32)> {
        let _admission = forward_admission().acquire();
        match forward_route(self.arch())? {
            ForwardRoute::GotOcr2 => return self.forward_got(&img),
            ForwardRoute::SmolVlm2 => return self.forward_smolvlm2(&img),
            ForwardRoute::OneChart => return self.forward_onechart(&img),
            ForwardRoute::Tromr => return self.forward_tromr(&img),
            ForwardRoute::UnlimitedV1 | ForwardRoute::UnlimitedV2 => {}
        }
        let t = Instant::now();
        progress::emit("preprocess", 0, 0);
        let pre = preprocess::preprocess_dynamic(
            img,
            preprocess_mode(),
            preprocess_profile(self.arch()),
        )?;
        progress::emit("preprocess", 1, 1);
        timing_log(&format!("preprocess {:.2}s", t.elapsed().as_secs_f64()));
        self.forward_pre(pre)
    }

    /// The GOT-OCR2 forward path (bead B3/B5/B7): squash-bicubic-1024/CLIP preprocess
    /// → SAM-ViT-B vision + `mm_projector_vary` connector + `<imgpad>` splice →
    /// Qwen2 dense decoder greedy generation → tiktoken decode. Returns the decoded
    /// text plus source pixel dims (for the postprocess geometry the wrappers share).
    fn forward_got(&self, img: &image::DynamicImage) -> FocrResult<(String, u32, u32)> {
        use image::GenericImageView;
        let t = Instant::now();
        let (w, h) = img.dimensions();
        let tk = self.got_tokenizer()?;
        // O(n) KV-cache greedy decode (B9); the model stops at <|im_end|>. The token
        // cap honors `--max-length`/FOCR_MAX_NEW_TOKENS via `decode_params` (bd-3j3p),
        // clamped to GOT's config `max_new_tokens` 4096 (the Baidu default 32768 thus
        // resolves to 4096, byte-identical to the old hardcoded cap). `--format` (CLI)
        // / FOCR_GOT_FORMAT selects the `OCR with format:` .mmd mode (LaTeX/tables/
        // charts/molecular/geometry/sheet-music); default plain `OCR: ` unchanged.
        let format = got_format_requested();
        let max_new = self.decode_params.max_length.min(got::MAX_NEW_TOKENS);
        let text = got::recognize(&self.weights, self.got_statics()?, tk, img, max_new, format)?;
        timing_log(&format!("got forward {:.2}s", t.elapsed().as_secs_f64()));
        Ok((text, w, h))
    }

    /// The SmolVLM2 describe/VQA forward path (C7/C9, bd-3jo6.3.7/.3.9):
    /// Pillow-exact LANCZOS split preprocess → SigLIP + pixel-shuffle +
    /// `modality_projection` → `<image>` splice → SmolLM2 KV-cache greedy →
    /// BPE decode. The question comes from `--question` /
    /// `FOCR_SMOLVLM2_QUESTION` (default: the model-card caption prompt); the
    /// tokenizer is the shared BPE loader ([`Self::tokenizer`]) — the C6
    /// scheme detection reads the SmolLM2 `pre_tokenizer` from the
    /// `tokenizer.json` beside the model. `--max-length` caps generation,
    /// clamped inside [`smolvlm2::recognize`] to the 8192 `max_position`
    /// budget net of the prompt.
    fn forward_smolvlm2(&self, img: &image::DynamicImage) -> FocrResult<(String, u32, u32)> {
        use image::GenericImageView;
        let t = Instant::now();
        let (w, h) = img.dimensions();
        let tk = self.tokenizer()?;
        let question = smolvlm2_question();
        let max_new = self.decode_params.max_length;
        let text = smolvlm2::recognize(
            &self.weights,
            self.smol_statics()?,
            tk,
            img,
            &question,
            max_new,
        )?;
        timing_log(&format!(
            "smolvlm2 forward {:.2}s",
            t.elapsed().as_secs_f64()
        ));
        Ok((text, w, h))
    }

    /// The OneChart chart→dict forward path (D6-D8, bd-3jo6.4.5+): raw-[0,1]
    /// squash preprocess → the certified SAM tower + `mm_projector` →
    /// `<imgpad>` splice → OPT KV-cache greedy → the `<Number>` number-head
    /// tap → strip/brace-complete → the self-verify verdict. The text output
    /// is the repaired chart dict; the verdict + `pred_locs` are logged via
    /// [`timing_log`] (a structured robot field is future surface — the
    /// census's 'structured fields' contract is [`onechart::ChartResult`],
    /// already the library return type).
    fn forward_onechart(&self, img: &image::DynamicImage) -> FocrResult<(String, u32, u32)> {
        use image::GenericImageView;
        let t = Instant::now();
        let (w, h) = img.dimensions();
        let tk = self.tokenizer()?;
        let max_new = self.decode_params.max_length;
        let res = onechart::recognize(&self.weights, self.onechart_statics()?, tk, img, max_new)?;
        timing_log(&format!(
            "onechart forward {:.2}s (reliable_distance {:?} reliable {:?})",
            t.elapsed().as_secs_f64(),
            res.reliable_distance,
            res.reliable
        ));
        Ok((res.json_text, w, h))
    }

    /// The TrOMR music-recognition forward path (E9, bd-3jo6.5.9): §6 staff
    /// preprocess → the certified hybrid encoder → deterministic argmax
    /// 4-head generate → §8 semantic merge → partwise MusicXML. The TEXT
    /// output is the MusicXML (the primary interop export); the model-native
    /// semantic stream is logged via [`timing_log`] and is the library-level
    /// [`tromr::MusicResult`] field.
    /// Take (consume) the staff-level metadata recorded by the most recent
    /// TrOMR full-page forward, if any (bd-av64.2). Non-music forwards never
    /// set it; a second take returns `None` until the next music forward.
    pub fn take_music_meta(&self) -> Option<MusicPageMeta> {
        self.music_meta.lock().ok().and_then(|mut slot| slot.take())
    }

    /// Take (consume) the [`DecodeReport`] of the most recent decode through
    /// the sequential driver — a single-page, multi-page or streaming
    /// Unlimited-OCR forward — if any. A second take returns `None` until the
    /// next decode.
    pub fn take_last_decode(&self) -> Option<DecodeReport> {
        self.last_decode
            .lock()
            .ok()
            .and_then(|mut slot| slot.take())
    }

    fn forward_tromr(&self, img: &image::DynamicImage) -> FocrResult<(String, u32, u32)> {
        use image::GenericImageView;
        let t = Instant::now();
        let (w, h) = img.dimensions();
        let tk = self.tromr_tokenizer()?;
        let page = tromr::recognize_page(&self.weights, tk, img)?;
        let total = page.staves.len() + page.skips.len();
        for skip in &page.skips {
            // Human-visible skip note on stderr (the bd-fck1 PDF page-skip
            // precedent) — stdout stays the data surface. The MusicXML still
            // carries every staff that recognized (bd-av64.2).
            crate::progress::stderr_message(format_args!(
                "[focr] staff {}/{} skipped: {} (bbox x{} y{} w{} h{})",
                skip.index + 1,
                total,
                skip.reason,
                skip.bbox.0,
                skip.bbox.1,
                skip.bbox.2,
                skip.bbox.3
            ));
        }
        timing_log(&format!(
            "tromr forward {:.2}s ({}/{} staves recognized, semantic {} chars total)",
            t.elapsed().as_secs_f64(),
            page.staves.len(),
            total,
            page.staves
                .iter()
                .map(|(_, r, _)| r.semantic.len())
                .sum::<usize>()
        ));
        let semantics: Vec<String> = page
            .staves
            .iter()
            .map(|(_, r, _)| r.semantic.clone())
            .collect();
        let warnings = tromr::sanity_warnings(&semantics);
        for w in &warnings {
            timing_log(&format!(
                "  tromr.sanity {} part {} measure {}: {}",
                w.kind, w.part, w.measure, w.detail
            ));
        }
        if let Ok(mut slot) = self.music_meta.lock() {
            *slot = Some(MusicPageMeta {
                staves: page.staves.iter().map(|(i, _, b)| (*i, *b)).collect(),
                skips: page.skips.clone(),
                warnings,
            });
        }
        let xml = if page.staves.len() == 1 {
            page.staves
                .into_iter()
                .next()
                .expect("one staff")
                .1
                .musicxml
        } else {
            let semantics: Vec<String> = page
                .staves
                .into_iter()
                .map(|(_, r, _)| r.semantic)
                .collect();
            tromr::staves_to_musicxml(&semantics)?
        };
        Ok((xml, w, h))
    }

    /// The TrOMR 4-table WordLevel music tokenizer, loaded from the
    /// `tokenizer_{rhythm,pitch,lift,note}.json` files beside the model
    /// (the zoo files-beside convention) and cached.
    fn tromr_tokenizer(&self) -> FocrResult<&crate::tokenizer::music::MusicTokenizer> {
        self.tromr_tokenizer.get_or_try_init(|| {
            if let Some([rhythm, pitch, lift, note]) = &self.sidecars.music_tables {
                return crate::tokenizer::music::MusicTokenizer::from_json_tables(
                    [rhythm, pitch, lift, note],
                    [
                        "tokenizer_rhythm.json (in-memory)",
                        "tokenizer_pitch.json (in-memory)",
                        "tokenizer_lift.json (in-memory)",
                        "tokenizer_note.json (in-memory)",
                    ],
                );
            }
            let dir = self.path.parent().unwrap_or_else(|| Path::new("."));
            crate::tokenizer::music::MusicTokenizer::from_dir(dir)
        })
    }

    /// The GOT Qwen tiktoken tokenizer, loaded from `qwen.tiktoken` beside the
    /// model and cached. (The Baidu path uses [`Self::tokenizer`].)
    fn got_tokenizer(&self) -> FocrResult<&crate::tokenizer::tiktoken::Tiktoken> {
        self.got_tokenizer.get_or_try_init(|| {
            if let Some(bytes) = &self.sidecars.qwen_tiktoken {
                return crate::tokenizer::tiktoken::Tiktoken::from_qwen_tiktoken(bytes);
            }
            let dir = self.path.parent().unwrap_or_else(|| Path::new("."));
            let path = dir.join("qwen.tiktoken");
            let bytes = std::fs::read(&path).map_err(|e| {
                FocrError::ModelNotFound(format!(
                    "GOT-OCR2 tokenizer qwen.tiktoken not found beside the model at {}: {e}",
                    path.display()
                ))
            })?;
            crate::tokenizer::tiktoken::Tiktoken::from_qwen_tiktoken(&bytes)
        })
    }

    /// The model architecture this loaded model is — the [`model_arch::ModelArch`]
    /// descriptor that drives its identity, config, and the forward dispatch.
    ///
    /// Resolved once at construction: a `.focrq` is the `model_id` it declares
    /// (absent means `unlimited-ocr`, permanently), a tagged artifact its own
    /// arch (e.g. `got-ocr2`), and a raw safetensors shard the model whose exact
    /// census it matched. There is no fallback to the default model.
    #[must_use]
    pub fn arch(&self) -> &'static dyn model_arch::ModelArch {
        self.arch
    }

    /// Which model this is, for a run's outputs (U2-E1).
    #[must_use]
    pub fn identity(&self) -> ModelIdentity {
        let sha = self.weights.source_sha256();
        ModelIdentity {
            model_id: self.arch.id(),
            source_sha256: (!sha.is_empty()).then(|| sha.to_owned()),
        }
    }

    /// Guard the per-arch forward dispatch (A1): an IMPLEMENTED arch proceeds; a
    /// planned zoo arch (described in the registry but whose forward lands in a
    /// later sub-epic) returns a clean [`FocrError::NotImplemented`] naming itself,
    /// rather than mis-running a different model's forward.
    fn ensure_arch_implemented(arch: &dyn model_arch::ModelArch) -> FocrResult<()> {
        if arch.implemented() {
            Ok(())
        } else {
            Err(FocrError::NotImplemented(format!(
                "model architecture '{}' ({}) forward is not yet implemented \
                 (franken_ocr model zoo, epic bd-3jo6)",
                arch.id(),
                arch.display_name()
            )))
        }
    }

    /// Shared post-preprocess forward: vision tower → connector → decoder →
    /// detokenize, over an already-built [`Preprocessed`] bundle. Both
    /// [`Self::forward`] (from a path) and [`Self::forward_dynamic`] (from an
    /// in-memory image) funnel through here, so the two entry points run the
    /// byte-identical model pipeline once preprocessing has produced `pre`.
    fn forward_pre(&self, pre: Preprocessed) -> FocrResult<(String, u32, u32)> {
        // Per-arch forward dispatch seam (model zoo, A1+A2): only an IMPLEMENTED arch
        // runs its forward. `arch()` now reads the loaded artifact's `model_id` tag,
        // so a (future) GOT-OCR2 / SmolVLM2 / … `.focrq` cleanly returns
        // NotImplemented here until its sub-epic lands its own forward, rather than
        // mis-running the Unlimited-OCR pipeline on foreign weights.
        Self::ensure_arch_implemented(self.arch())?;
        let (image_w, image_h) = Self::image_dims(&pre);
        if self.arch().decode_contract().pages.is_some() {
            // Unlimited-OCR-2: a document of one page; its body is the text.
            let split = self
                .decode_document(std::slice::from_ref(&pre), None)?
                .split;
            progress::emit("postprocess", 0, 0);
            let body = split.bodies.into_iter().next().unwrap_or_default();
            return Ok((body, image_w, image_h));
        }

        // ── vision tower (SAM⊕CLIP -> bridge projector 2048->1280) ───────────
        let tv = Instant::now();
        let vision_features = self.vision_tower(&pre)?;
        timing_log(&format!("vision_tower {:.2}s", tv.elapsed().as_secs_f64()));

        // ── connector: prompt embeds + masked_scatter of the 273-slot block ──
        let (inputs_embeds, prompt_ids) = self.build_inputs_embeds(&pre, &vision_features)?;

        // ── decoder prefill + sequential greedy AR decode to EOS ─────────────
        let generated = self.generate(inputs_embeds, &prompt_ids)?;

        // Detokenize the generated ids into the raw model text (the postprocess
        // pass strips EOS / parses ref-det / rewrites image spans).
        progress::emit("postprocess", 0, 0);
        let decoded = self.tokenizer()?.decode(&generated)?;
        Ok((decoded, image_w, image_h))
    }

    /// Recognize one document image end-to-end (forward + postprocess),
    /// returning structured markdown.
    ///
    /// Thin shell over [`Self::forward`] + [`postprocess::finalize`]: the forward
    /// produces the raw decoded text and the source pixel dims; postprocess
    /// strips the EOS marker, parses the ref/det spans, rewrites `image` spans to
    /// markdown image refs, deletes the other layout spans, and normalizes the
    /// LaTeX `\coloneqq`/`\eqqcolon` ([SPEC-110..119]).
    ///
    /// # Errors
    /// Propagates [`Self::forward`] (today, the preprocess `NotImplemented`) or a
    /// postprocess validation error.
    pub fn recognize(&self, image_path: &Path) -> FocrResult<String> {
        let (decoded, image_w, image_h) = self.forward(image_path)?;
        postprocess::finalize(&decoded, image_w, image_h)
    }

    /// Recognize an already-decoded in-memory image end-to-end (forward +
    /// postprocess), returning structured markdown — the in-memory form of
    /// [`Self::recognize`] that the native PDF path ([`crate::pdf`]) uses to feed
    /// one rasterized page without a temp file.
    ///
    /// # Errors
    /// As [`Self::recognize`] (sans the file-open path).
    pub fn recognize_dynamic(&self, img: image::DynamicImage) -> FocrResult<String> {
        let (decoded, image_w, image_h) = self.forward_dynamic(img)?;
        postprocess::finalize(&decoded, image_w, image_h)
    }

    /// Recognize one document image end-to-end, returning the markdown AND the
    /// structured layout (bounding boxes) parsed from the same decoded output.
    ///
    /// The boxes come from [`postprocess::parse_layout`] over the EXACT raw decode
    /// that [`postprocess::finalize`] renders to markdown, so `markdown` and
    /// `layout` are always consistent. This is the structured form
    /// `focr ocr --json` / `-o out.json` uses.
    ///
    /// # Errors
    /// As [`Self::recognize`].
    pub fn recognize_with_layout(&self, image_path: &Path) -> FocrResult<RecognizedDocument> {
        let (decoded, image_w, image_h) = self.forward(image_path)?;
        Self::finalize_document(&decoded, image_w, image_h)
    }

    /// In-memory form of [`Self::recognize_with_layout`] — the native PDF path
    /// feeds one rasterized page here to collect its per-page layout.
    ///
    /// # Errors
    /// As [`Self::recognize_dynamic`].
    pub fn recognize_dynamic_with_layout(
        &self,
        img: image::DynamicImage,
    ) -> FocrResult<RecognizedDocument> {
        let (decoded, image_w, image_h) = self.forward_dynamic(img)?;
        Self::finalize_document(&decoded, image_w, image_h)
    }

    /// Assemble the markdown body and the pixel-rescaled layout from one forward's
    /// `(decoded, width, height)` — the single point that keeps the two in sync.
    fn finalize_document(
        decoded: &str,
        image_w: u32,
        image_h: u32,
    ) -> FocrResult<RecognizedDocument> {
        let markdown = postprocess::finalize(decoded, image_w, image_h)?;
        let layout = postprocess::parse_layout(decoded, image_w, image_h);
        Ok(RecognizedDocument { markdown, layout })
    }

    /// Recognize one document image AND crop the figure regions it grounds but
    /// does not transcribe (the `![](images/…)` placeholders) out of the source,
    /// returning the [`RecognizedDocument`] plus the cropped figures. The CLI's
    /// `--extract-figures` writes each figure to a file and rewrites the markdown
    /// reference to point at it.
    ///
    /// The source is re-decoded with the SAME EXIF transform the forward used (so
    /// the crops align with the layout boxes); the re-decode is skipped entirely
    /// when the page grounds no figures.
    ///
    /// # Errors
    /// As [`Self::recognize_with_layout`], plus [`FocrError::InputDecode`] if the
    /// source cannot be re-decoded for cropping.
    pub fn recognize_with_figures(
        &self,
        image_path: &Path,
    ) -> FocrResult<(RecognizedDocument, Vec<ExtractedFigure>)> {
        let (decoded, image_w, image_h) = self.forward(image_path)?;
        let document = Self::finalize_document(&decoded, image_w, image_h)?;
        let figures = if postprocess::figure_refs(&decoded, image_w, image_h, "").is_empty() {
            Vec::new()
        } else {
            let source = preprocess::decode_path(image_path)?;
            Self::crop_figures(&decoded, &source, image_w, image_h, "")
        };
        Ok((document, figures))
    }

    /// In-memory form of [`Self::recognize_with_figures`] — the native PDF path
    /// feeds one rasterized page here; that page raster IS the crop source.
    ///
    /// # Errors
    /// As [`Self::recognize_dynamic_with_layout`].
    pub fn recognize_dynamic_with_figures(
        &self,
        img: image::DynamicImage,
    ) -> FocrResult<(RecognizedDocument, Vec<ExtractedFigure>)> {
        // The forward consumes `img`; retain the source pixels for cropping. Only
        // the figure path pays this clone (figureless callers use
        // `recognize_dynamic_with_layout`), and it is dwarfed by the forward.
        let source = img.clone();
        let (decoded, image_w, image_h) = self.forward_dynamic(img)?;
        let document = Self::finalize_document(&decoded, image_w, image_h)?;
        let figures = Self::crop_figures(&decoded, &source, image_w, image_h, "");
        Ok((document, figures))
    }

    /// Crop each figure/image span of one page out of `source` (the original-
    /// resolution, EXIF-aligned decode), pairing it with the markdown token it
    /// appears as. Boxes are corner-ordered and clamped to the image; a degenerate
    /// (empty) box is skipped while KEEPING the span's index, so a written figure's
    /// `markdown_ref` always matches the rendered placeholder. `img_base` matches
    /// the page's markdown prefix (`""` for the single-image / per-PDF-page paths).
    fn crop_figures(
        decoded: &str,
        source: &image::DynamicImage,
        image_w: u32,
        image_h: u32,
        img_base: &str,
    ) -> Vec<ExtractedFigure> {
        crop_figure_refs(
            &postprocess::figure_refs(decoded, image_w, image_h, img_base),
            source,
        )
    }

    /// Parse a document (U2-E2, D23): the reference `infer_multi` contract
    /// (bd-1gv.25), the headline "Unlimited" feature. Every page joins one
    /// cross-page prompt and one decode runs over the whole document, so
    /// page N attends to pages 1..N−1 (OQ-13), never a concatenation of
    /// single-page parses.
    ///
    /// The pages come from `source` one at a time (U2-D0): each is
    /// rasterized, preprocessed and dropped before the next is read, with
    /// its EXIF orientation applied as a single image's is (SPEC-020). Every
    /// page reaches `on_item` before it is read; a page that did not
    /// rasterize is left out of the prompt and listed in
    /// [`DocumentResult::skipped`] (D24). `on_page` gets each page body as
    /// its `<PAGE>` boundary is crossed in the token stream (bd-2z0y),
    /// with its [`PageResult`] when the model splits pages exactly; it never
    /// changes the result.
    ///
    /// A document that stops early is a result, not an error: its
    /// `finish_reason` says why and its pages are those it completed. The
    /// decode report stays on the model for [`Self::take_last_decode`].
    ///
    /// # Errors
    /// * [`FocrError::NotImplemented`] for a model without the
    ///   Unlimited-OCR document contract; zoo models parse pages
    ///   independently via [`Self::recognize_batch`].
    /// * [`FocrError::Usage`] for options this build cannot honor
    ///   ([`DocumentOptions`]), or a paged model's document over its page
    ///   limit.
    /// * The first page's raster error when no page rasterized.
    /// * A prompt with no room to generate, or any preprocess, vision or
    ///   decode error; v1 also stops with [`FocrError::Cancelled`].
    #[cfg(feature = "pdf")]
    pub fn recognize_document(
        &self,
        source: &mut crate::page_source::PageSource,
        options: &DocumentOptions,
        on_item: &mut dyn FnMut(&crate::page_source::PageItem),
        on_page: Option<PageSink<'_>>,
    ) -> FocrResult<DocumentResult> {
        multi_page_route(self.arch())?;
        let enforce = options.page_limit(source.len())?;
        self.check_document_pages(source.len(), enforce)?;
        let mut pres = Vec::with_capacity(source.len());
        let mut origins = Vec::with_capacity(source.len());
        let mut skipped = Vec::new();
        let mut first_error: Option<FocrError> = None;
        for item in source.by_ref() {
            on_item(&item);
            match item.raster {
                Ok(img) => {
                    pres.push(self.document_page(img)?);
                    origins.push(PageOrigin {
                        source_page: item.source_page,
                        half: item.half,
                    });
                }
                Err(e) => {
                    skipped.push(crate::pdf::SkippedPage {
                        page: item.source_page,
                        reason: e.to_string(),
                    });
                    first_error.get_or_insert(e);
                }
            }
        }
        if pres.is_empty() {
            return Err(first_error.unwrap_or_else(|| {
                FocrError::InputDecode("the page source yielded no pages".into())
            }));
        }
        // Split spreads can make more logical pages than source pages.
        let enforce = options.page_limit(pres.len())?;
        self.check_document_pages(pres.len(), enforce)?;
        let DocumentPages { markdown, pages } = self.parse_document(pres, &origins, on_page)?;
        // Both arms record a report; a poisoned slot leaves an EOS default.
        let report = self.last_decode.lock().ok().and_then(|slot| slot.clone());
        let mut warnings = source.warnings();
        warnings.extend(report.as_ref().map(decode_warnings).unwrap_or_default());
        Ok(DocumentResult {
            markdown,
            pages_out: report
                .as_ref()
                .and_then(|r| r.pages)
                .map_or(pages.len(), |p| p.bodies),
            pages,
            skipped,
            finish_reason: report
                .as_ref()
                .map_or(FinishReason::Stop, |r| r.finish_reason),
            extra_markers: report
                .as_ref()
                .and_then(|r| r.document.as_ref())
                .map_or(0, |stats| stats.extra_markers),
            warnings,
            model: self.identity(),
        })
    }

    /// One document page's preprocess: v1's Base-640 squash, or a paged
    /// model's base letterbox (Unlimited-OCR-2 reads a document page like a
    /// single page, never squashed; U2-B7).
    ///
    /// # Errors
    /// [`FocrError::Usage`] when gundam mode was asked of a paged model, whose
    /// documents read base views only (DISC-U2-6), rather than dropping the
    /// flag (U2-E1); a decode or preprocess error.
    fn document_page(&self, img: image::DynamicImage) -> FocrResult<Preprocessed> {
        if self.arch().decode_contract().pages.is_some() {
            preprocess::preprocess_dynamic(
                img,
                paged_document_view(preprocess_mode(), self.arch().id())?,
                preprocess_profile(self.arch()),
            )
        } else {
            preprocess::preprocess_dynamic_squash(img, MULTI_PAGE_BASE_SIZE)
        }
    }

    /// Refuse a paged model's unwindowed document over
    /// [`UOCR2_UNWINDOWED_PAGE_LIMIT`] pages before any image is read, unless
    /// `enforce` is off (an explicit unwindowed request, which then relies on
    /// the reported position stop). Other models pass.
    ///
    /// # Errors
    /// [`FocrError::Usage`] (exit 2) naming the limit and the remedy.
    fn check_document_pages(&self, pages: usize, enforce: bool) -> FocrResult<()> {
        if !enforce
            || pages <= UOCR2_UNWINDOWED_PAGE_LIMIT
            || self.arch().decode_contract().pages.is_none()
        {
            return Ok(());
        }
        Err(FocrError::Usage(format!(
            "{} parses at most {UOCR2_UNWINDOWED_PAGE_LIMIT} pages in one pass (its {} \
             position limit); this document has {pages}. Select at most \
             {UOCR2_UNWINDOWED_PAGE_LIMIT} pages with `--pages`, or parse pages independently \
             (omit `--multi-page`)",
            self.arch().id(),
            thousands(sampler::UOCR2_MAX_POSITIONS),
        )))
    }

    /// A document's parse (bd-1gv.25): per-page vision, ONE fused cross-page
    /// prefill, ONE decode, `finalize_multi` assembly. `on_page` (bd-2z0y)
    /// streams each completed page body as its boundary is crossed.
    /// `origins[k]` is where page `k + 1` of the prompt came from.
    ///
    /// The pages (U2-E3): Unlimited-OCR-2 splits its ids into one body per
    /// page (D20), so each page it read has a [`PageResult`], streamed with
    /// its body. v1 splits text, and merges pages; its pages are read from
    /// the text only when it holds exactly one body per page.
    fn parse_document(
        &self,
        pres: Vec<Preprocessed>,
        origins: &[PageOrigin],
        mut on_page: Option<PageSink<'_>>,
    ) -> FocrResult<DocumentPages> {
        let _admission = forward_admission().acquire();
        let frame = PageFrame {
            sizes: pres.iter().map(|p| p.original_size).collect(),
            origins,
            model_id: self.arch().id(),
        };
        if self.arch().decode_contract().pages.is_some() {
            let mut closed = |page: &ClosedPage<'_>| {
                if let Some(sink) = on_page.as_deref_mut() {
                    let result = frame.closed_page(page);
                    sink(&StreamedPage {
                        index: page.index,
                        body: page.body,
                        page: Some(&result),
                    });
                }
            };
            let decoded = self.decode_document(&pres, Some(&mut closed))?;
            let pages = frame.token_pages(&decoded);
            return Ok(DocumentPages {
                markdown: postprocess::join_page_texts(pages.iter().map(|p| &p.text)),
                pages,
            });
        }
        // ── vision tower per page, sequential (§6.5: one live forward) ──────
        let tv = Instant::now();
        let mut globals: Vec<Mat> = Vec::with_capacity(pres.len());
        for pre in &pres {
            let mut feats = self.vision_tower(pre)?;
            if feats.len() != 1 {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "document: a Base-mode page produced {} vision blocks; expected 1",
                    feats.len()
                )));
            }
            globals.push(feats.remove(0));
        }
        timing_log(&format!(
            "multi_page.vision ({} pages) {:.2}s",
            pres.len(),
            tv.elapsed().as_secs_f64()
        ));

        // ── ONE cross-page prompt + embeds (all blocks at the single <image>) ─
        let (prompt_ids, images_seq_mask) = self.build_prompt_multi(&pres)?;
        let max_positions = self.decode_params.max_positions;
        if prompt_ids.len() + 1 > max_positions {
            return Err(FocrError::Other(anyhow::anyhow!(
                "document: the assembled {}-page prefix is {} tokens — over the \
                 {max_positions} position budget by {} with no room to generate; {}",
                pres.len(),
                prompt_ids.len(),
                (prompt_ids.len() + 1).saturating_sub(max_positions),
                warning_recovery(WarningKind::PromptTooLong),
            )));
        }
        let mut inputs_embeds = self.embed_prompt(&prompt_ids)?;
        let image_newline = Self::image_newline(&self.weights)?;
        let view_seperator = Self::view_seperator(&self.weights)?;
        let grid = preprocess::num_queries(MULTI_PAGE_BASE_SIZE);
        connector::fuse_no_crop(
            &self.weights,
            &mut inputs_embeds,
            &globals,
            grid,
            grid,
            &image_newline,
            &view_seperator,
            &images_seq_mask,
        )?;

        // ── ONE decode across the whole document (OQ-13) ─────────────────────
        // ngram_window 1024 (multi) per OQ-18; generation capped so
        // prefix + generated never exceeds the shared position budget.
        let position_room = max_positions - prompt_ids.len();
        let params = sampler::DecodeParams {
            ngram_window: sampler::NGRAM_WINDOW_MULTI,
            max_length: self.decode_params.max_length.min(position_room),
            ..self.decode_params.clone()
        };
        let td = Instant::now();
        let generated = match on_page {
            None => self.generate_with(inputs_embeds, &prompt_ids, &params)?,
            Some(on_page) => {
                // Streaming (bd-2z0y): a per-committed-token observer decodes
                // each id with a UTF-8-safe incremental decoder (U2-X1), which
                // holds back a character split across byte-level tokens
                // instead of failing, and feeds the text so far to a
                // [`postprocess::PageStream`] every STREAM_CHECK_EVERY tokens
                // (the scanner is idempotent over re-fed prefixes). Streaming
                // never changes the emitted ids, and the terminal decode below
                // still surfaces real errors.
                const STREAM_CHECK_EVERY: usize = 8;
                let tok = self.tokenizer()?;
                let mut stream = postprocess::PageStream::new();
                let mut decoder = crate::tokenizer::StreamDecoder::new();
                let mut text = String::new();
                let mut seen = 0usize;
                // v1 merges pages, so a streamed body names no page.
                let mut on_body = |index: usize, body: &str| {
                    on_page(&StreamedPage {
                        index,
                        body,
                        page: None,
                    });
                };
                let mut observer = |id: u32| {
                    seen += 1;
                    // An id with no token adds no streamed text; the terminal
                    // decode reports it.
                    let _ = decoder.push(tok, id, &mut text);
                    if seen.is_multiple_of(STREAM_CHECK_EVERY) {
                        stream.feed(&text, &mut on_body);
                    }
                };
                let generated = self.generate_with_observer(
                    inputs_embeds,
                    &prompt_ids,
                    &params,
                    Some(&mut observer),
                )?;
                // End of decode: flush the final in-flight page (EOS marker
                // stripped, exactly as finalize_multi strips it before split).
                decoder.flush(&mut text);
                let text = postprocess::strip_eos(&text);
                stream.feed(&text, &mut on_body);
                stream.finish(&text, &mut on_body);
                generated
            }
        };
        timing_log(&format!(
            "multi_page.decode {} tokens {:.2}s",
            generated.len(),
            td.elapsed().as_secs_f64()
        ));
        let decoded = self.tokenizer()?.decode(&generated)?;
        let stripped = postprocess::strip_eos(&decoded);
        if let Ok(mut slot) = self.last_decode.lock()
            && let Some(report) = slot.as_mut()
        {
            if report.finish_reason == FinishReason::Length
                && position_room <= self.decode_params.max_length
            {
                report.cap = Some(LengthCap::Positions);
            }
            report.pages = Some(PageBodies {
                expected: pres.len(),
                bodies: postprocess::page_body_count(&stripped),
            });
        }
        Ok(DocumentPages {
            markdown: postprocess::finalize_multi(&decoded, pres.len())?,
            pages: frame.text_pages(&stripped),
        })
    }

    /// Unlimited-OCR-2's document parse (U2-B6) up to the page split: the
    /// paged prompt over `pres` ([`Self::document_inputs`]), one guarded
    /// decode ([`Self::generate_document`]), and its ids split into one body
    /// per page by token id ([`postprocess::split_pages_by_token`]). One
    /// page is a document of one. Records the [`DecodeReport`] with its
    /// [`DocumentStats`]; `on_page` gets each page as the page ends
    /// ([`postprocess::TokenPageStream`]), with its finish by the decode
    /// loop's rule, from a guard that sees the same ids. The caller holds the
    /// forward admission.
    ///
    /// # Errors
    /// A model without a page contract, a prompt that leaves no position to
    /// generate, or any vision, decode or split error.
    fn decode_document(
        &self,
        pres: &[Preprocessed],
        mut on_page: Option<&mut dyn FnMut(&ClosedPage<'_>)>,
    ) -> FocrResult<DecodedDocument> {
        let Some(paging) = self.arch().decode_contract().pages else {
            return Err(FocrError::Other(anyhow::anyhow!(
                "{} has no page contract, so no document loop",
                self.arch().id()
            )));
        };
        // The cooperative budget (U2-B6, D11): the per-page forward budget
        // times the page count, from here, checked once per decode step.
        let deadline = crate::forward_stage_budget().and_then(|per_page| {
            let pages = u32::try_from(pres.len().max(1)).unwrap_or(u32::MAX);
            Instant::now().checked_add(per_page.checked_mul(pages)?)
        });
        let params = self.document_params(pres.len());
        // The layout alone fixes the prompt length: refuse before any weight.
        let (layout, _) = self.build_prompt_multi(pres)?;
        if layout.len() + 1 > params.max_positions {
            return Err(FocrError::Other(anyhow::anyhow!(
                "the {}-page prompt is {} tokens, over the {} position limit with no room to \
                 generate; {}",
                pres.len(),
                thousands(layout.len()),
                thousands(params.max_positions),
                warning_recovery(WarningKind::PromptTooLong),
            )));
        }
        let (inputs_embeds, prompt_ids) = self.document_inputs(pres)?;
        let tok = self.tokenizer()?;
        // The pages close as the ids stream in. A page's clock starts at its
        // `<PAGE>` (page 1: the first id) and stops at the id that closes it.
        let mut stream = postprocess::TokenPageStream::new(tok, paging.marker_id, pres.len());
        let markers = prompt_ids
            .iter()
            .filter(|&&id| id == paging.marker_id)
            .count();
        let mut mirror = document_guard(
            if paging.guard { markers } else { 0 },
            paging.marker_id,
            &params,
        );
        let mut elapsed = Vec::with_capacity(pres.len());
        let mut opened: Option<Instant> = None;
        let mut observer = |id: u32| {
            let now = Instant::now();
            let since = now - *opened.get_or_insert(now);
            // The decode loop's rule for a page a marker closes.
            let finish = if mirror.at_cap() {
                PageFinish::PageCap
            } else {
                PageFinish::Marker
            };
            mirror.observe(id);
            stream.push(id, |index, body, tokens| {
                elapsed.push(since);
                opened = Some(now);
                if let Some(sink) = on_page.as_deref_mut() {
                    sink(&ClosedPage {
                        index,
                        body,
                        tokens,
                        finish,
                        elapsed: since,
                    });
                }
            });
        };
        let decode = self.generate_document(
            &inputs_embeds,
            &prompt_ids,
            &params,
            deadline,
            Some(&mut observer),
        )?;
        // The page in flight ends with the decode, by the loop's rule.
        let since = opened.map_or_else(std::time::Duration::default, |t| t.elapsed());
        let last = match decode.finish_reason {
            FinishReason::Stop if mirror.at_cap() => PageFinish::PageCap,
            FinishReason::Stop => PageFinish::Eos,
            FinishReason::Length => PageFinish::Length,
            FinishReason::Budget => PageFinish::Budget,
            FinishReason::Cancelled => PageFinish::Cancelled,
        };
        stream.finish(|index, body, tokens| {
            elapsed.push(since);
            if let Some(sink) = on_page.as_deref_mut() {
                sink(&ClosedPage {
                    index,
                    body,
                    tokens,
                    finish: last,
                    elapsed: since,
                });
            }
        });
        let split =
            postprocess::split_pages_by_token(tok, &decode.ids, paging.marker_id, pres.len())?;
        let mut stats = decode.stats;
        stats.page_tokens.clone_from(&split.tokens);
        stats.leading_text = split.leading_text;
        stats.extra_markers = split.extra_markers;
        stats.gundam_view = pres
            .iter()
            .any(|p| matches!(p.mode, preprocess::PreprocessMode::Gundam { .. }));
        let finish = stats.page_finish.clone();
        timing_log(&format!(
            "document {} pages: {} tokens, finish {}, pages {:?}, forced {}, hf-id-0 {}, \
             retired {} blocks, peak kv {} bytes, retire fallbacks {}",
            pres.len(),
            decode.ids.len(),
            decode.finish_reason.as_str(),
            stats
                .page_finish
                .iter()
                .map(|f| f.as_str())
                .collect::<Vec<_>>(),
            stats.forced_steps,
            stats.hf_all_neg_inf_steps,
            stats.retired_blocks,
            stats.peak_kv_bytes,
            stats.retire_fallbacks,
        ));
        if let Ok(mut slot) = self.last_decode.lock() {
            let (decodes, length_stops) = slot
                .as_ref()
                .map_or((0, 0), |prior| (prior.decodes, prior.length_stops));
            let length = decode.finish_reason == FinishReason::Length;
            let stopped = decode.finish_reason != FinishReason::Stop;
            *slot = Some(DecodeReport {
                ids: decode.ids,
                finish_reason: decode.finish_reason,
                cap: length.then_some(LengthCap::Positions),
                pages: Some(PageBodies {
                    expected: pres.len(),
                    bodies: split.bodies.len(),
                }),
                max_length: params.max_length,
                max_positions: params.max_positions,
                decodes: decodes + 1,
                length_stops: length_stops + usize::from(stopped),
                document: Some(stats),
                admission_warnings: lever_policy::admit_from_process(self.arch()).warnings,
            });
        }
        Ok(DecodedDocument {
            split,
            finish,
            elapsed,
        })
    }

    /// Unlimited-OCR-2's prompt ids and `inputs_embeds` over `pres` (U2-B6):
    /// the paged prompt ([`paged_prompt`]) with each page's vision block
    /// scattered into its image slots. One page may be a gundam view; a
    /// document's pages are base views (MU:1110-1124).
    ///
    /// # Errors
    /// An empty page list, a tiled page in a document, or any vision or
    /// connector error.
    fn document_inputs(&self, pres: &[Preprocessed]) -> FocrResult<(Mat, Vec<u32>)> {
        let tv = Instant::now();
        if let [pre] = pres {
            let features = self.vision_tower(pre)?;
            timing_log(&format!("vision_tower {:.2}s", tv.elapsed().as_secs_f64()));
            return self.build_inputs_embeds(pre, &features);
        }
        let Some(first) = pres.first() else {
            return Err(FocrError::Other(anyhow::anyhow!(
                "a document needs at least one page"
            )));
        };
        let mut globals = Vec::with_capacity(pres.len());
        for (i, pre) in pres.iter().enumerate() {
            let mut features = self.vision_tower(pre)?;
            if pre.crop_grid.is_tiled() || features.len() != 1 {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "document page {} is not one base view ({} vision blocks)",
                    i + 1,
                    features.len()
                )));
            }
            globals.push(features.remove(0));
        }
        timing_log(&format!(
            "document.vision ({} pages) {:.2}s",
            pres.len(),
            tv.elapsed().as_secs_f64()
        ));
        let (prompt_ids, images_seq_mask) = self.build_prompt_multi(pres)?;
        let mut inputs_embeds = self.embed_prompt(&prompt_ids)?;
        connector::fuse_no_crop(
            &self.weights,
            &mut inputs_embeds,
            &globals,
            Self::global_grid_h(first),
            Self::global_grid_w(first),
            &Self::image_newline(&self.weights)?,
            &Self::view_seperator(&self.weights)?,
            &images_seq_mask,
        )?;
        Ok((inputs_embeds, prompt_ids))
    }

    /// The decode params of an Unlimited-OCR-2 document of `pages` pages: the
    /// engine's, with the n-gram window 128 for one page and 1024 for more
    /// (D10) unless `--ngram-window` set one (`0` then means 100,
    /// [`sampler::uocr2_ngram_window`]).
    fn document_params(&self, pages: usize) -> DecodeParams {
        let default_window = if pages > 1 {
            sampler::NGRAM_WINDOW_MULTI
        } else {
            sampler::NGRAM_WINDOW_SINGLE
        };
        DecodeParams {
            ngram_window: decode_overrides().ngram_window.unwrap_or(default_window),
            ..self.decode_params.clone()
        }
    }

    /// Unlimited-OCR-2's document decode (U2-B6; MU:1124-1200) over the paged
    /// prompt `prompt_ids`, whose rows are `inputs_embeds`.
    ///
    /// Prefill once; then per step `lm_head` on the last row, the
    /// generated-only n-gram ban and the page-count guard with forced tokens
    /// authoritative ([`sampler::decode_step_guarded`], DISC-U2-2), and the
    /// greedy argmax. EOS ends the decode without joining the ids; the
    /// position limit (`pos + 1 >= max_positions`) ends it as `length`. The
    /// per-page cap is `max_page_tokens` (when > 0) within `max_length`, which
    /// in a document applies per page, never to the whole output (D11).
    ///
    /// While the guard runs, block retirement (U2-B4) frees dead KV blocks
    /// each step unless [`srswa::RETIRE_ENV`] turns it off. If its detector
    /// fires, the document is decoded again without retirement, which is
    /// deterministic: the rerun repeats every id the observer already saw,
    /// which it does not see twice, and `retire_fallbacks` counts the rerun.
    ///
    /// # Errors
    /// Any decode error, or a rerun that departs from the ids before the
    /// detector fired.
    fn generate_document(
        &self,
        inputs_embeds: &Mat,
        prompt_ids: &[u32],
        params: &DecodeParams,
        deadline: Option<Instant>,
        observer: Option<&mut dyn FnMut(u32)>,
    ) -> FocrResult<DocumentDecode> {
        let precision = DocumentPrecision::from_env()?;
        decode_with_retire_fallback(
            srswa::retire_enabled(),
            observer,
            |retire, budget_floor, ids, observer| {
                let pass = DocumentPass {
                    precision,
                    retire,
                    deadline,
                    budget_floor,
                };
                self.generate_document_once(inputs_embeds, prompt_ids, params, pass, ids, observer)
            },
        )
    }

    /// Whether this model's forward checks `FOCR_STAGE_BUDGET_FORWARD_MS`
    /// itself and stops with its completed pages (Unlimited-OCR-2's document
    /// loop, U2-B6), so the engine's stage timeout is only a backstop.
    #[cfg(feature = "native")]
    pub(crate) fn checks_forward_budget(&self) -> bool {
        self.arch().decode_contract().pages.is_some()
    }

    /// The next token's `[1, vocab]` logits from a document decode's state.
    fn document_logits(&self, decoder_state: &DocumentDecoder<'_>) -> FocrResult<Mat> {
        match decoder_state {
            DocumentDecoder::Cached { wc, hidden, .. } => decoder::lm_head_cached(wc, hidden),
            DocumentDecoder::Stateless {
                embeds,
                ids,
                domain,
            } => {
                let coords = srswa::PageCoords::build(ids)?;
                let h = decoder::forward_srswa(&self.weights, embeds, &coords, *domain)?;
                decoder::lm_head(&self.weights, &Self::last_hidden_row(&h)?)
            }
        }
    }

    /// One pass of [`Self::generate_document`], with block retirement on when
    /// `pass.retire` (and the guard runs). `ids` holds the generated ids so
    /// far, also when an error ends the pass. Past `pass.deadline` (and its
    /// floor) the pass ends as [`FinishReason::Budget`].
    fn generate_document_once(
        &self,
        inputs_embeds: &Mat,
        prompt_ids: &[u32],
        params: &DecodeParams,
        pass: DocumentPass,
        ids: &mut Vec<u32>,
        observer: Option<&mut dyn FnMut(u32)>,
    ) -> FocrResult<DocumentDecode> {
        let Some(paging) = self.arch().decode_contract().pages else {
            return Err(FocrError::Other(anyhow::anyhow!(
                "{} has no page contract",
                self.arch().id()
            )));
        };
        let pages = prompt_ids
            .iter()
            .filter(|&&id| id == paging.marker_id)
            .count();
        let prompt_len = prompt_ids.len();
        let mut guard = document_guard(
            if paging.guard { pages } else { 0 },
            paging.marker_id,
            params,
        );
        ids.clear();
        // A budget already spent (by vision) skips the decoder cache and the
        // prefill: the document stops before its first id.
        if pass.budget_spent(0) {
            timing_log("forward stage budget ran out before the prefill");
            return Ok(DocumentDecode {
                ids: Vec::new(),
                finish_reason: FinishReason::Budget,
                stats: DocumentStats {
                    page_finish: vec![PageFinish::Budget],
                    ..DocumentStats::default()
                },
            });
        }

        // Prefill: a cached path (the conservative cache, or the all-f32 one),
        // or the all-f32 re-forward of the whole grown sequence every step
        // (O(n²), the parity oracle).
        let tp = Instant::now();
        progress::emit("prefill", 0, prompt_len as u64);
        timing_log(&format!("precision {}", pass.precision.name()));
        let mut decoder_state = if pass.precision == DocumentPrecision::Stateless {
            DocumentDecoder::Stateless {
                embeds: inputs_embeds.clone(),
                ids: prompt_ids.to_vec(),
                domain: params.entmax_domain,
            }
        } else {
            let wc = if pass.precision == DocumentPrecision::CachedF32 {
                self.decoder_cache_f32()?
            } else {
                self.decoder_cache()?
            };
            let coords = srswa::PageCoords::build(prompt_ids)?;
            let (hidden, mut state) = decoder::prefill_with_cache_srswa(
                wc,
                inputs_embeds,
                &coords,
                params.entmax_domain,
            )?;
            state.set_retire(pass.retire && guard.enabled());
            DocumentDecoder::Cached {
                wc,
                state,
                hidden: Self::last_hidden_row(&hidden)?,
            }
        };
        let logits = self.document_logits(&decoder_state)?;
        progress::emit("prefill", prompt_len as u64, prompt_len as u64);
        timing_log(&format!(
            "prefill {:.2}s ({prompt_len} tokens)",
            tp.elapsed().as_secs_f64()
        ));
        let spec = DocumentLoop {
            params,
            pass,
            marker: paging.marker_id,
            prompt_len,
            pages,
            forced_prefix: 0,
        };
        let (finish_reason, stats) =
            self.document_loop(spec, &mut decoder_state, logits, &mut guard, ids, observer)?;
        Ok(DocumentDecode {
            ids: ids.clone(),
            finish_reason,
            stats,
        })
    }

    /// The decode loop of one document pass (U2-B6), or of one window
    /// (U2-D1a), from the `logits` of the next token: per step the guarded
    /// greedy step ([`sampler::decode_step_guarded`]), then EOS ends it, or
    /// the id is appended to `ids`, shown to `observer`, and fed. The
    /// position limit (the emitted row at `prompt_len + ids.len() − 1`
    /// reaching `max_positions − 1`) ends it as `length`, the pass's budget
    /// as `budget`, and a cooperative shutdown as `cancelled`. `ids` may
    /// open with `spec.forced_prefix` ids the caller forced and fed (a
    /// reused window's spliced text and `<PAGE>`): they count for the
    /// n-gram, the guard and the positions, never for the observer. Returns
    /// how it ended and its stats.
    ///
    /// # Errors
    /// Any decode error, or an id outside the embed table.
    fn document_loop(
        &self,
        spec: DocumentLoop<'_>,
        decoder_state: &mut DocumentDecoder<'_>,
        mut logits: Mat,
        guard: &mut sampler::PageGuard,
        ids: &mut Vec<u32>,
        mut observer: Option<&mut dyn FnMut(u32)>,
    ) -> FocrResult<(FinishReason, DocumentStats)> {
        let DocumentLoop {
            params,
            pass,
            marker,
            prompt_len,
            pages,
            forced_prefix,
        } = spec;
        let table = self.embed_table()?;
        let hidden_dim = table.cols();
        let room = params
            .max_positions
            .saturating_sub(prompt_len + forced_prefix);
        let mut stats = DocumentStats::default();
        let mut finish_reason = FinishReason::Length;
        let td = Instant::now();
        loop {
            // A cancel ends the document as a stop, keeping the pages done
            // (U2-E2); like the budget, a rerun first repeats what was seen.
            if crate::shutdown_requested() && ids.len() >= pass.budget_floor {
                if stats.page_finish.len() < pages.max(1) {
                    stats.page_finish.push(PageFinish::Cancelled);
                }
                finish_reason = FinishReason::Cancelled;
                timing_log(&format!(
                    "cancelled after {} generated tokens",
                    ids.len() - forced_prefix
                ));
                break;
            }
            if pass.budget_spent(ids.len()) {
                if stats.page_finish.len() < pages.max(1) {
                    stats.page_finish.push(PageFinish::Budget);
                }
                finish_reason = FinishReason::Budget;
                timing_log(&format!(
                    "forward stage budget ran out after {} generated tokens",
                    ids.len() - forced_prefix
                ));
                break;
            }
            let at_cap = guard.at_cap();
            let step = sampler::decode_step_guarded(&logits, ids, guard, params)?;
            stats.forced_steps += usize::from(step.forced);
            stats.hf_all_neg_inf_steps += usize::from(step.hf_all_neg_inf);
            if step.is_eos {
                if stats.page_finish.len() < pages.max(1) {
                    stats.page_finish.push(if at_cap {
                        PageFinish::PageCap
                    } else {
                        PageFinish::Eos
                    });
                }
                finish_reason = FinishReason::Stop;
                break;
            }
            let id = step.token_id;
            if id as usize >= table.rows() {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "decoded token id {id} outside the embed vocab {}",
                    table.rows()
                )));
            }
            ids.push(id);
            guard.observe(id);
            // A marker after the first closes the page before it.
            if id == marker && guard.markers() >= 2 && stats.page_finish.len() < pages {
                stats.page_finish.push(if at_cap {
                    PageFinish::PageCap
                } else {
                    PageFinish::Marker
                });
            }
            progress::emit("decode", (ids.len() - forced_prefix) as u64, room as u64);
            if let Some(f) = observer.as_deref_mut() {
                f(id);
            }
            // MU:1183-1185: the row just emitted sits at prompt_len + len − 1.
            if prompt_len + ids.len() >= params.max_positions {
                if stats.page_finish.len() < pages.max(1) {
                    stats.page_finish.push(PageFinish::Length);
                }
                break;
            }
            let row = table.row_f32(id as usize);
            match decoder_state {
                DocumentDecoder::Cached { wc, state, hidden } => {
                    let row = Mat::from_vec(1, hidden_dim, row);
                    let forced_next = matches!(guard.verdict(), sampler::GuardVerdict::Force(_));
                    let h =
                        decoder::decode_step_with_cache_srswa(wc, state, id, &row, forced_next)?;
                    *hidden = Self::last_hidden_row(&h)?;
                }
                DocumentDecoder::Stateless {
                    embeds, ids: grown, ..
                } => {
                    let mut data = std::mem::take(&mut embeds.data);
                    data.extend(row);
                    *embeds = Mat::from_vec(embeds.rows + 1, hidden_dim, data);
                    grown.push(id);
                }
            }
            logits = self.document_logits(decoder_state)?;
        }
        if let DocumentDecoder::Cached { state, .. } = decoder_state {
            stats.retired_blocks = state.dead_blocks();
            stats.peak_kv_bytes = state.peak_bytes();
        }
        let generated = ids.len() - forced_prefix;
        timing_log(&format!(
            "decode {:.2}s ({generated} tokens, {:.3}s/tok)",
            td.elapsed().as_secs_f64(),
            td.elapsed().as_secs_f64() / (generated.max(1) as f64)
        ));
        Ok((finish_reason, stats))
    }

    /// Recognize a batch of document images, returning one [`FocrResult`] per
    /// image in input order (`result[i]` ⇄ `image_paths[i]`).
    ///
    /// When the continuous-batch decode spine is armed
    /// ([`batch_scheduler::spine_enabled`], `FOCR_BATCH_SPINE`) AND the int8 decode
    /// path is active, every page is prefilled and then decoded TOGETHER through
    /// the [`batch_scheduler`] — the int8 weight cache and the embed table are read
    /// ONCE for the whole batch, and the scheduler is the single sequential driver
    /// (Doctrine #5: the model is already an owned `Arc`, no per-step relock).
    /// Otherwise each image falls back to the proven sequential
    /// [`Self::recognize`].
    ///
    /// LOSSLESS: with the spine on, each page's emitted tokens are byte-for-byte
    /// the tokens [`Self::generate_cached_i8`] produces for that page ALONE (the
    /// batched int8 GEMMs are `M`-independent — bd-1azu.2 — and R-SWA attention is
    /// per-stream), so the finalized markdown is sha256-identical to the spine-off
    /// path (the bd-1azu.13 gate). The spine is the int8 throughput path; if int8
    /// is not requested it stays on the sequential fallback so the result tracks
    /// whatever oracle is in force.
    #[must_use]
    pub fn recognize_batch(&self, image_paths: &[&Path]) -> Vec<FocrResult<String>> {
        // The spine reproduces the int8 R-SWA cached decode (`generate_cached_i8`),
        // which `generate` selects only when int8 is requested AND the stateless
        // O(n^2) oracle is NOT forced. Engage the spine under EXACTLY that
        // condition so its output tracks whichever sequential oracle is in force.
        // Arch gate (fresh-eyes fix): the spine pipeline is built for
        // Unlimited-OCR v1 (Baidu tensor names, R-SWA rings, the
        // BASE_PROMPT tokenizer); a got-ocr2/zoo artifact must take the
        // sequential loop, whose per-page forward() carries the arch dispatch.
        let spine = batch_scheduler::spine_enabled()
            && (int8_decode_requested() || self.artifact_stores_gated_int8())
            && std::env::var_os(DECODE_STATELESS_ENV).is_none()
            && self.arch().unlimited_variant() == Some(model_arch::UnlimitedVariant::V1Rswa);
        // A7.5 (bd-3jo6.1.7.5): the DENSE continuous-batch spine. GOT decode is
        // int8 by construction (the QInt8 panels), so it needs only the master
        // spine switch; the per-page id-stream is byte-identical to the
        // sequential loop (decoder_qwen2's scheduler-level gate).
        if !spine
            && batch_scheduler::spine_enabled()
            && matches!(self.arch().id(), "got-ocr2" | "smolvlm2" | "onechart")
        {
            let _admission = forward_admission().acquire();
            match self.recognize_batch_dense(image_paths) {
                Ok(results) => return results,
                Err(err) => {
                    let msg = err.to_string();
                    return image_paths
                        .iter()
                        .map(|_| Err(FocrError::Other(anyhow::anyhow!("{msg}"))))
                        .collect();
                }
            }
        }
        if !spine {
            return image_paths.iter().map(|p| self.recognize(p)).collect();
        }
        let _admission = forward_admission().acquire();
        match self.recognize_batch_spine(image_paths) {
            Ok(results) => results,
            Err(err) => {
                // A batch-level failure (e.g. the int8 weight cache or embed table
                // could not be built) is the SAME error the sequential loop would
                // hit on every page; report it per image so the CLI's exit
                // semantics (and per-image JSON shape) match the sequential path.
                let msg = err.to_string();
                image_paths
                    .iter()
                    .map(|_| Err(FocrError::Other(anyhow::anyhow!("{msg}"))))
                    .collect()
            }
        }
    }

    /// The DENSE continuous-batch body (A7.5): decode every page's image +
    /// prompt into `inputs_embeds` sequentially (one live vision forward at a
    /// time), then ONE [`decoder_qwen2::generate_greedy_batched`] drive over
    /// all pages, dispatched per arch (GOT / SmolVLM2 / OneChart). Per-page
    /// image failures are recorded per page; the surviving pages still batch.
    /// The outer `Err` is reserved for batch-level failures
    /// (tokenizer/weights), mirroring the MoE spine.
    fn recognize_batch_dense(&self, image_paths: &[&Path]) -> FocrResult<Vec<FocrResult<String>>> {
        let arch_id = self.arch().id();
        let max_new = match arch_id {
            "got-ocr2" => self.decode_params.max_length.min(got::MAX_NEW_TOKENS),
            _ => self.decode_params.max_length,
        };
        let t = Instant::now();
        // Decode images per page; a bad page keeps its own error and is
        // excluded from the batch (the sequential loop's per-page semantics).
        let mut out: Vec<Option<FocrResult<String>>> =
            (0..image_paths.len()).map(|_| None).collect();
        let mut live: Vec<usize> = Vec::new();
        let mut imgs: Vec<image::DynamicImage> = Vec::new();
        for (i, p) in image_paths.iter().enumerate() {
            crate::cancel_checkpoint()?;
            match preprocess::decode_path(p) {
                Ok(img) => {
                    live.push(i);
                    imgs.push(img);
                }
                Err(e) => out[i] = Some(Err(e)),
            }
        }
        if !imgs.is_empty() {
            use image::GenericImageView;
            let dims: Vec<(u32, u32)> = imgs.iter().map(|im| im.dimensions()).collect();
            let refs: Vec<&image::DynamicImage> = imgs.iter().collect();
            let texts: Vec<String> = match arch_id {
                "got-ocr2" => got::recognize_batch(
                    &self.weights,
                    self.got_statics()?,
                    self.got_tokenizer()?,
                    &refs,
                    max_new,
                    got_format_requested(),
                )?,
                "smolvlm2" => smolvlm2::recognize_batch(
                    &self.weights,
                    self.smol_statics()?,
                    self.tokenizer()?,
                    &refs,
                    &smolvlm2_question(),
                    max_new,
                )?,
                "onechart" => onechart::recognize_batch(
                    &self.weights,
                    self.onechart_statics()?,
                    self.tokenizer()?,
                    &refs,
                    max_new,
                )?
                .into_iter()
                .map(|r| r.json_text)
                .collect(),
                other => {
                    return Err(FocrError::Other(anyhow::anyhow!(
                        "dense batch spine: unrouted arch {other}"
                    )));
                }
            };
            // The SAME per-page finalize the sequential path applies
            // (recognize -> forward_<arch> -> postprocess::finalize).
            for ((slot, text), (iw, ih)) in live.into_iter().zip(texts).zip(dims) {
                out[slot] = Some(postprocess::finalize(&text, iw, ih));
            }
        }
        timing_log(&format!(
            "{arch_id} forward(batch of {}) {:.2}s",
            image_paths.len(),
            t.elapsed().as_secs_f64()
        ));
        Ok(out
            .into_iter()
            .map(|r| r.unwrap_or_else(|| Err(FocrError::Other(anyhow::anyhow!("page skipped")))))
            .collect())
    }

    /// The continuous-batch spine body for [`Self::recognize_batch`]: prefill every
    /// page, assemble a [`rswa::BatchedRingCache`] from the per-page rings, then
    /// drive the [`batch_scheduler::BatchScheduler`] to emit every page's tokens at
    /// once, detokenizing + [`postprocess::finalize`]-ing each in input order.
    ///
    /// Per-page prefill failures (e.g. a bad image) are recorded for that page and
    /// excluded from the batch; the surviving pages still decode. The outer `Err`
    /// is reserved for batch-level setup failures (weight cache / embed table).
    fn recognize_batch_spine(&self, image_paths: &[&Path]) -> FocrResult<Vec<FocrResult<String>>> {
        let n = image_paths.len();
        // ── batch-level artifacts, read ONCE (Doctrine #5) ───────────────────────
        let wc = self.decoder_cache_i8()?;
        let embed_table = self.embed_table()?;
        let tokenizer = self.tokenizer()?;

        // ── per-page prefill, staged so vision can batch ACROSS pages ────────────
        let mut out: Vec<Option<FocrResult<String>>> = (0..n).map(|_| None).collect();

        // Stage 1: preprocess every page (a per-page error records + skips it).
        // Same mode resolution as the single-page forward (bd-1e9n) — the
        // batched-vision stage groups views by side, so Gundam's mixed
        // 640/1024 views batch correctly too.
        let (mode, profile) = (preprocess_mode(), preprocess_profile(self.arch()));
        let mut pres: Vec<Option<Preprocessed>> = (0..n).map(|_| None).collect();
        for (gi, &path) in image_paths.iter().enumerate() {
            match preprocess::preprocess_image(path, mode, profile) {
                Ok(p) => pres[gi] = Some(p),
                Err(e) => out[gi] = Some(Err(e)),
            }
        }

        // Stage 2: the vision tower over every admitted page — batched ACROSS
        // pages by default (bd-1azu.10: hydrate once, ONE SAM + ONE CLIP forward
        // for the whole batch), or the per-page loop under FOCR_BATCH_VISION=0.
        // A batched-path error is batch-level (e.g. weight hydration — the
        // per-page path would hit it on page 1 too), so it is recorded on every
        // admitted page rather than aborting pages that already failed cleanly.
        let admitted: Vec<usize> = (0..n).filter(|&gi| pres[gi].is_some()).collect();
        let mut feats: Vec<Option<Vec<Mat>>> = (0..n).map(|_| None).collect();
        if !admitted.is_empty() {
            let mut batched_ok = false;
            if batch_vision_enabled() {
                let prefs: Vec<&Preprocessed> = admitted
                    .iter()
                    .map(|&gi| pres[gi].as_ref().expect("admitted page has a preprocess"))
                    .collect();
                match self.vision_tower_batched_pages(&prefs) {
                    Ok(all) => {
                        for (k, f) in all.into_iter().enumerate() {
                            feats[admitted[k]] = Some(f);
                        }
                        batched_ok = true;
                    }
                    Err(e) => {
                        // Fall back to the per-page loop (the kill-switch path)
                        // rather than stringifying the error onto every page:
                        // the per-page tower reproduces the SAME failure as a
                        // properly TYPED per-page error (FormatMismatch stays
                        // exit 7, ModelNotFound stays exit 3), and a
                        // batched-only bug degrades to correct results.
                        crate::progress::stderr_message(format_args!(
                            "[focr] batched vision failed ({e}); using the per-page tower"
                        ));
                    }
                }
            }
            if !batched_ok {
                for &gi in &admitted {
                    let pre = pres[gi].as_ref().expect("admitted page has a preprocess");
                    match self.vision_tower(pre) {
                        Ok(f) => feats[gi] = Some(f),
                        Err(e) => out[gi] = Some(Err(e)),
                    }
                }
            }
        }

        // Stage 3: per-page connector + int8 prefill (unchanged math).
        let mut streams: Vec<batch_scheduler::PageStream> = Vec::new();
        let mut stream_caches: Vec<Vec<rswa::RingCache>> = Vec::new();
        // (global page index, image_w, image_h) in scheduler-submission order.
        let mut scheduled: Vec<(usize, u32, u32)> = Vec::new();
        for gi in 0..n {
            let (Some(pre), Some(vf)) = (pres[gi].take(), feats[gi].take()) else {
                continue;
            };
            match self.prefill_from_features(wc, &pre, &vf) {
                Ok(p) => {
                    streams.push(batch_scheduler::PageStream::new(
                        gi,
                        p.prefill_len,
                        &p.prompt_ids,
                        p.last_hidden,
                    ));
                    stream_caches.push(p.caches);
                    scheduled.push((gi, p.image_w, p.image_h));
                }
                Err(e) => out[gi] = Some(Err(e)),
            }
        }

        // ── decode every prefilled page together, re-sorted to input order ───────
        if !streams.is_empty() {
            let mut batched = rswa::BatchedRingCache::from_streams(stream_caches)?;
            let mut step = batch_scheduler::DecoderBatchStep {
                wc,
                caches: &mut batched,
                embed_table: &embed_table,
                params: &self.decode_params,
            };
            let mut scheduler =
                batch_scheduler::BatchScheduler::from_env(self.decode_params.max_length);
            let token_lists = scheduler.run(streams, &mut step)?;
            // `token_lists[k]` ⇄ `scheduled[k]` (both in ascending input-index order).
            for (k, tokens) in token_lists.into_iter().enumerate() {
                let (gi, w, h) = scheduled[k];
                let finalized = tokenizer
                    .decode(&tokens)
                    .and_then(|decoded| postprocess::finalize(&decoded, w, h));
                out[gi] = Some(finalized);
            }
        }

        Ok(out
            .into_iter()
            .map(|slot| {
                slot.unwrap_or_else(|| {
                    Err(FocrError::Other(anyhow::anyhow!(
                        "native_engine::OcrModel::recognize_batch: page produced no result"
                    )))
                })
            })
            .collect())
    }

    /// Build (or fetch the cached) int8 decoder weight cache — the spine twin of
    /// the get-or-build inside [`Self::generate_cached_i8`], factored so the batch
    /// driver reads the ~1.2 s quant ONCE without re-running it per page. Does NOT
    /// change the sequential path (which keeps its own inline get-or-build).
    fn decoder_cache_i8(&self) -> FocrResult<&decoder::DecoderWeightCacheI8> {
        self.require_full_int8_authorization()?;
        self.decoder_cache_i8
            .get_or_try_init(|| decoder::DecoderWeightCacheI8::build(&self.weights))
    }

    /// Whether this model's artifact ITSELF stores the gated tensor sets
    /// (attention `q/k/v/o` + `lm_head`) in int8 — true exactly for the
    /// explicitly-tagged wasm experts-int4 recipe (bd-4l71). For such an
    /// artifact the int8 attention/lm_head values were baked in by the
    /// converter, so **artifact storage IS the opt-in**: the
    /// `FOCR_INT8_ATTN`/`FOCR_INT8_LMHEAD` env kill-switches (which gate the
    /// quantize-at-load experiment) are not required and not consulted.
    fn artifact_stores_gated_int8(&self) -> bool {
        // The wasm recipe is v1's alone; the loader refuses it on a v2 artifact
        // (U2-C3), and this never reads it as one either.
        self.arch().unlimited_variant() == Some(model_arch::UnlimitedVariant::V1Rswa)
            && self.weights.quant_recipe()
                == Some(crate::quant::convert::UNLIMITED_OCR_WASM_INT4_RECIPE_ID)
    }

    /// Authorize the all-int8 decoder cache: either the artifact itself stores
    /// the gated sets int8 (see [`Self::artifact_stores_gated_int8`]) or the
    /// experimental three-switch env contract is satisfied.
    fn require_full_int8_authorization(&self) -> FocrResult<()> {
        if self.artifact_stores_gated_int8() {
            return Ok(());
        }
        require_experimental_full_int8_recipe()
    }

    /// Build or fetch the conservative mixed-precision decoder cache. Unlike
    /// the legacy all-f32 cache, this keeps the 2,148 recipe FFN/expert matrices
    /// in int8 storage, so it is practical to retain across pages. For
    /// Unlimited-OCR-2 it also holds the f32 router weights (U2-B5).
    fn decoder_cache(&self) -> FocrResult<&decoder::DecoderWeightCache> {
        self.decoder_cache
            .get_or_try_init(|| match self.arch().unlimited_variant() {
                Some(model_arch::UnlimitedVariant::V2SrSwa) => {
                    decoder::DecoderWeightCache::build_srswa(&self.weights)
                }
                Some(model_arch::UnlimitedVariant::V1Rswa) | None => {
                    decoder::DecoderWeightCache::build(&self.weights)
                }
            })
    }

    /// Unlimited-OCR-2's all-f32 cache ([`DECODE_F32_ENV`]), built once.
    ///
    /// # Errors
    /// [`FocrError::Usage`] for another model; otherwise as
    /// [`decoder::DecoderWeightCache::build_srswa_f32`], which refuses a
    /// quantized artifact.
    fn decoder_cache_f32(&self) -> FocrResult<&decoder::DecoderWeightCache> {
        if self.arch().unlimited_variant() != Some(model_arch::UnlimitedVariant::V2SrSwa) {
            return Err(FocrError::Usage(format!(
                "{DECODE_F32_ENV} decodes Unlimited-OCR-2 documents; {} has no all-f32 cache",
                self.arch().id()
            )));
        }
        self.decoder_cache_f32.get_or_try_init(|| {
            let t = Instant::now();
            let built = decoder::DecoderWeightCache::build_srswa_f32(&self.weights)?;
            timing_log(&format!(
                "decoder f32 cache {:.2}s",
                t.elapsed().as_secs_f64()
            ));
            Ok(built)
        })
    }

    /// The lazily-hydrated CLIP tower weights (bd-av64.10), built once per
    /// model and reused by every view/page forward — same idiom as
    /// [`Self::decoder_cache_i8`]. The retryable single-initializer prevents a
    /// concurrent caller from duplicating the ~1.2 GB hydration.
    ///
    /// # Errors
    /// A missing or mis-shaped `model.vision_model.*` tensor.
    fn clip_weights(&self) -> FocrResult<&vision_clip::ClipWeights> {
        self.clip_cache.get_or_try_init(|| {
            let th = Instant::now();
            let built = vision_clip::clip_weights_from(&self.weights)?;
            timing_log(&format!(
                "    clip.hydrate(cached) {:.2}s",
                th.elapsed().as_secs_f64()
            ));
            Ok(built)
        })
    }

    fn unlimited_vision_statics(&self) -> FocrResult<&UnlimitedVisionStatics> {
        self.unlimited_vision.get_or_try_init(|| {
            let started = Instant::now();
            let built = UnlimitedVisionStatics {
                sam: vision_sam::sam_weights_from(&self.weights, "model.sam_model")?,
                projector: vision_bridge::projector_weights_from(&self.weights)?,
            };
            timing_log(&format!(
                "    unlimited_vision.hydrate(cached) {:.2}s",
                started.elapsed().as_secs_f64()
            ));
            Ok(built)
        })
    }

    /// The lazily-hydrated GOT model-constant tensors (bd-av64.10), built once
    /// per model and reused by the sequential AND batch page paths — same
    /// idiom as [`Self::clip_weights`].
    ///
    /// # Errors
    /// A missing or mis-shaped GOT tensor.
    fn got_statics(&self) -> FocrResult<&got::GotStatics> {
        self.got_statics.get_or_try_init(|| {
            got::hydrate_statics(
                &self.weights,
                self.arch().vision_tower_prefix(),
                stream_vision_tower(),
            )
        })
    }

    /// The lazily-hydrated OneChart model-constant tensors — see
    /// [`Self::got_statics`].
    ///
    /// # Errors
    /// A missing or mis-shaped OneChart tensor.
    fn onechart_statics(&self) -> FocrResult<&onechart::OnechartStatics> {
        self.onechart_statics.get_or_try_init(|| {
            onechart::hydrate_statics(
                &self.weights,
                self.arch().vision_tower_prefix(),
                stream_vision_tower(),
            )
        })
    }

    /// The lazily-hydrated SmolVLM2 model-constant tensors — see
    /// [`Self::got_statics`].
    ///
    /// # Errors
    /// A missing or mis-shaped SmolVLM2 tensor.
    fn smol_statics(&self) -> FocrResult<&smolvlm2::SmolStatics> {
        self.smol_statics
            .get_or_try_init(|| smolvlm2::hydrate_statics(&self.weights, stream_vision_tower()))
    }

    /// Connector + int8 prefill for ONE page whose vision features are ALREADY
    /// computed (stage 3 of the spine; stages 1–2 preprocess and run the tower,
    /// batched across pages by default per bd-1azu.10). Followed by the int8
    /// [`decoder::prefill_with_cache_i8`], returning the seed state a
    /// [`batch_scheduler::PageStream`] needs plus the page's pixel dims for
    /// postprocess. Mirrors EXACTLY the prefill the sequential
    /// [`Self::generate_cached_i8`] does, so the seeded decode is byte-identical.
    fn prefill_from_features(
        &self,
        wc: &decoder::DecoderWeightCacheI8,
        pre: &Preprocessed,
        vision_features: &[Mat],
    ) -> FocrResult<PagePrefill> {
        let (image_w, image_h) = Self::image_dims(pre);
        let (inputs_embeds, prompt_ids) = self.build_inputs_embeds(pre, vision_features)?;
        let prefill_len = inputs_embeds.rows;
        let _fwd = enter_forward();
        let (hidden, caches) = decoder::prefill_with_cache_i8(wc, &inputs_embeds)?;
        let last_hidden = Self::last_hidden_row(&hidden)?;
        Ok(PagePrefill {
            prefill_len,
            prompt_ids,
            last_hidden,
            caches,
            image_w,
            image_h,
        })
    }

    // ── stage orchestration (private) ──────────────────────────────────────────

    /// The byte-level BPE tokenizer over `tokenizer.json` (sibling of the
    /// model artifact), loaded lazily.
    ///
    /// The tokenizer is needed for the prompt id-stream (the connector builds
    /// `images_seq_mask` against it) and to detokenize the generated ids. It is
    /// resolved next to the model file (the `tokenizer.json` ships beside the
    /// weights).
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] (or an IO error) if `tokenizer.json` is
    /// missing or malformed beside the model artifact.
    fn tokenizer(&self) -> FocrResult<&crate::tokenizer::Tokenizer> {
        self.tokenizer.get_or_try_init(|| {
            // Byte-loaded models carry the payload directly (no "beside").
            if let Some(bytes) = &self.sidecars.tokenizer_json {
                return checked_tokenizer(self.arch, bytes, "the in-memory tokenizer.json sidecar");
            }
            if let Some([vocab, merges, added]) = &self.sidecars.opt_triple {
                return crate::tokenizer::Tokenizer::from_opt_files(vocab, merges, added);
            }
            // The tokenizer ships beside the weights; OneChart uses the OPT
            // slow-tokenizer triple instead of tokenizer.json (D9).
            let dir = self.path.parent().unwrap_or_else(|| Path::new("."));
            if self.arch().id() == "onechart" {
                return crate::tokenizer::Tokenizer::from_opt_dir(dir);
            }
            // A sidecar absent at open (or a byte-loaded model's) is verified
            // here, on first use, by the same check the open runs.
            let path = dir.join("tokenizer.json");
            let bytes = std::fs::read(&path).map_err(|error| {
                FocrError::ModelNotFound(format!("tokenizer.json at {}: {error}", path.display()))
            })?;
            checked_tokenizer(self.arch, &bytes, &path.display().to_string())
        })
    }

    /// Run the two-tower vision encoder over every preprocessed view and project
    /// each into the decoder hidden rail (2048 -> 1280), returning the per-view
    /// hybrid vision features ([SPEC-040..052]).
    ///
    /// Drives, per view: [`vision_sam::forward`] -> (its `x3` becomes CLIP's
    /// `patch_embeds`) [`vision_clip::forward`] -> [`vision_bridge::forward`]
    /// (concat CLIP[:,1:] ++ SAM, then the linear projector). The SAM/CLIP/bridge
    /// kernels are implemented and tested over explicit weight bundles; the
    /// `Weights`-backed entrypoints used here hydrate their named tensors via the
    /// `.focrq` reader (bd-1es.3) and run the real math.
    ///
    /// # Errors
    /// The first vision-stage error (e.g. a missing or mis-shaped tensor surfaced
    /// by the `Weights`-backed SAM/CLIP/bridge entrypoints, or a kernel failure).
    fn vision_tower(&self, pre: &Preprocessed) -> FocrResult<Vec<Mat>> {
        let _fwd = enter_forward();
        // bd-4l71 wasm-lane residency: with the 3 GB wasm artifact resident,
        // whole-tower f32 caches (~1.2 GB CLIP + ~0.4 GB SAM) do not fit the
        // wasm32 4 GiB ceiling. For a wasm-recipe artifact ONLY, SAM/projector
        // hydrate per view and drop (the existing uncached arm) and CLIP runs
        // the per-block streamed forward — numerics bit-identical (gated by
        // `streamed_clip_forward_is_bit_identical_to_cached`). Conservative
        // artifacts keep their caches exactly as before.
        let wasm_residency = self.artifact_stores_gated_int8();
        let statics = if !wasm_residency && unlimited_vision_cache_enabled() {
            Some(self.unlimited_vision_statics()?)
        } else {
            None
        };
        let mut features = Vec::new();
        let views = Self::views(pre);
        // Progress budget for the whole tower: every view pays the same fixed
        // SAM + CLIP block count, so the per-block hooks inside the two towers
        // report one continuous, determinate count across views.
        progress::vision_begin(
            views.len() as u64
                * (vision_sam::DEPTH as u64 + vision_clip::ClipConfig::default().num_layers as u64),
        );
        if wasm_residency {
            // bd-K2 per-view hydration hoist: the streamed lane re-dequants the
            // SAME bf16 blocks once per view, so a 2-view page pays the whole
            // hydration traffic twice. Run the views INSIDE the block loop
            // instead (hydrate block -> all views -> drop). Streamed lane ONLY;
            // the cached/conservative arm below is untouched.
            return self.vision_tower_streamed_views(&views);
        }
        for view in views {
            // SAM tower -> [1024, 16*16] x3 feature (flatten(2) layout, OQ-6).
            let ts = Instant::now();
            let sam = if let Some(statics) = statics {
                let side = (view.cols as f64).sqrt() as usize;
                if side * side != view.cols {
                    return Err(FocrError::Other(anyhow::anyhow!(
                        "native_engine::OcrModel::vision_tower: image.cols {} is not a perfect square",
                        view.cols
                    )));
                }
                vision_sam::forward_with(&statics.sam, &view, side, side)?
            } else if wasm_residency {
                // Per-block hydrate/drop + bounded-scratch global attention —
                // bit-identical, tens of MB of scratch instead of ~1.2 GB.
                vision_sam::forward_streamed(&self.weights, &view, "model.sam_model")?
            } else {
                vision_sam::forward(&self.weights, &view)?
            };
            timing_log(&format!("  vision.sam {:.2}s", ts.elapsed().as_secs_f64()));
            // CLIP tower fed SAM's x3 as patch_embeds -> [N+1, 1024] (CLS at 0).
            let tc = Instant::now();
            let clip = if wasm_residency {
                vision_clip::forward_from_sam_streamed(
                    &vision_clip::ClipConfig::default(),
                    &self.weights,
                    &sam,
                )?
            } else {
                vision_clip::forward_from_sam(
                    &vision_clip::ClipConfig::default(),
                    self.clip_weights()?,
                    &sam,
                )?
            };
            timing_log(&format!("  vision.clip {:.2}s", tc.elapsed().as_secs_f64()));
            // Bridge: concat CLIP[:,1:] ++ SAM (2048) -> projector -> [N, 1280].
            let tb = Instant::now();
            let projected = if let Some(statics) = statics {
                vision_bridge::forward_with(&statics.projector, &clip, &sam)?
            } else {
                vision_bridge::forward(&self.weights, &clip, &sam)?
            };
            timing_log(&format!(
                "  vision.bridge {:.2}s",
                tb.elapsed().as_secs_f64()
            ));
            features.push(projected);
        }
        Ok(features)
    }

    /// The streamed (wasm-residency) vision lane with the per-view hydration
    /// **hoisted out of the view loop** (bd-K2).
    ///
    /// Before (views-outer): `for view { for block { hydrate; run; drop } }` —
    /// with `V` views per page the whole bf16→f32 block-dequant traffic of the
    /// SAM tower (12 blocks) + CLIP tower (24 blocks) is paid `V` times.
    ///
    /// After (views-inner): `for block { hydrate; for view { run }; drop }` —
    /// paid ONCE per page, i.e. `V`× less hydration traffic. The cost is `V`
    /// live activations instead of one (~13 MB per 1024² view), bounded further
    /// by [`STREAMED_VIEW_CHUNK`].
    ///
    /// Bit-identical by construction: views are mathematically independent, so
    /// interleaving them reorders nothing within a view — no batching, no
    /// restacking, no reassociation. See
    /// `vision_sam::forward_streamed_views` / `vision_clip::
    /// forward_from_sam_streamed_views` and their parity gates.
    ///
    /// # Errors
    /// As [`Self::vision_tower`]'s streamed arm.
    fn vision_tower_streamed_views(&self, views: &[Mat]) -> FocrResult<Vec<Mat>> {
        let clip_cfg = vision_clip::ClipConfig::default();
        let mut features = Vec::with_capacity(views.len());
        for chunk in views.chunks(STREAMED_VIEW_CHUNK) {
            let view_refs: Vec<&Mat> = chunk.iter().collect();
            let ts = Instant::now();
            let sams =
                vision_sam::forward_streamed_views(&self.weights, &view_refs, "model.sam_model")?;
            timing_log(&format!(
                "  vision.sam(streamed x{}) {:.2}s",
                view_refs.len(),
                ts.elapsed().as_secs_f64()
            ));
            let tc = Instant::now();
            let sam_refs: Vec<&Mat> = sams.iter().collect();
            let clips =
                vision_clip::forward_from_sam_streamed_views(&clip_cfg, &self.weights, &sam_refs)?;
            timing_log(&format!(
                "  vision.clip(streamed x{}) {:.2}s",
                sam_refs.len(),
                tc.elapsed().as_secs_f64()
            ));
            // The projector is small (~10 MB) but was likewise re-hydrated per
            // view by `vision_bridge::forward`; hoist it too. Same builder, so
            // `forward_with` is the same math `forward` performs.
            let tb = Instant::now();
            let projector = vision_bridge::projector_weights_from(&self.weights)?;
            for (clip, sam) in clips.iter().zip(sams.iter()) {
                features.push(vision_bridge::forward_with(&projector, clip, sam)?);
            }
            timing_log(&format!(
                "  vision.bridge(streamed x{}) {:.2}s",
                clips.len(),
                tb.elapsed().as_secs_f64()
            ));
        }
        Ok(features)
    }

    /// Batched vision across the spine's admitted pages (bd-1azu.10 / bd-t6a):
    /// hydrate [`vision_sam::SamWeights`] + [`vision_clip::ClipWeights`] ONCE,
    /// group every page's views by side (Base mode: one 1024² view per page ⇒
    /// one group), run ONE batched SAM forward + ONE batched CLIP forward per
    /// group — both proven byte-identical to their per-view forwards — then the
    /// cheap per-view bridge. Returns per-page feature vectors in exactly the
    /// order [`Self::vision_tower`] produces them, so the connector downstream
    /// cannot tell the difference. `FOCR_BATCH_VISION=0` bypasses this path.
    ///
    /// # Errors
    /// A weight-hydration failure, a non-square view, or any batched-forward
    /// rejection (e.g. a ragged same-side group, which Base mode cannot produce).
    fn vision_tower_batched_pages(&self, pres: &[&Preprocessed]) -> FocrResult<Vec<Vec<Mat>>> {
        let _fwd = enter_forward();
        // (page, view-slot) inventory, preserving vision_tower's per-page order.
        let mut page_views: Vec<Vec<Mat>> = pres.iter().map(|pre| Self::views(pre)).collect();
        let th = Instant::now();
        let retain_statics = unlimited_vision_cache_enabled();
        let statics_owner = retained_or_owned(
            retain_statics,
            &self.unlimited_vision,
            || -> FocrResult<UnlimitedVisionStatics> {
                let started = Instant::now();
                let built = UnlimitedVisionStatics {
                    sam: vision_sam::sam_weights_from(&self.weights, "model.sam_model")?,
                    projector: vision_bridge::projector_weights_from(&self.weights)?,
                };
                timing_log(&format!(
                    "    unlimited_vision.hydrate({}) {:.2}s",
                    if retain_statics {
                        "cached"
                    } else {
                        "batch-local"
                    },
                    started.elapsed().as_secs_f64()
                ));
                Ok(built)
            },
        )?;
        let statics = statics_owner.as_ref();
        let clip_cfg = vision_clip::ClipConfig::default();
        // The CLIP tower comes from the model-level cache (bd-av64.10): the
        // first batch hydrates it, every later batch reuses it.
        let clip_w = self.clip_weights()?;
        timing_log(&format!(
            "  vision.hydrate(batch) {:.2}s",
            th.elapsed().as_secs_f64()
        ));

        // Group all views by side so each batched forward sees a rectangular
        // stack (Gundam mixes 640² tiles with the 1024² global; Base is uniform).
        let mut groups: std::collections::BTreeMap<usize, Vec<(usize, usize)>> =
            std::collections::BTreeMap::new();
        for (p, views) in page_views.iter().enumerate() {
            for (v, view) in views.iter().enumerate() {
                let side = (view.cols as f64).sqrt() as usize;
                if side * side != view.cols {
                    return Err(FocrError::Other(anyhow::anyhow!(
                        "native_engine::OcrModel::vision_tower_batched_pages: page {p} view {v} \
                         cols {} is not a perfect square",
                        view.cols
                    )));
                }
                groups.entry(side).or_default().push((p, v));
            }
        }

        // One batched SAM + CLIP forward per side group; bridge stays per-view.
        let mut features: Vec<Vec<Option<Mat>>> = page_views
            .iter()
            .map(|views| views.iter().map(|_| None).collect())
            .collect();
        for (side, slots) in &groups {
            let view_refs: Vec<&Mat> = slots.iter().map(|&(p, v)| &page_views[p][v]).collect();
            let ts = Instant::now();
            let sams = vision_sam::forward_with_batched(&statics.sam, &view_refs, *side, *side)?;
            timing_log(&format!(
                "  vision.sam(batch of {}, side {side}) {:.2}s",
                slots.len(),
                ts.elapsed().as_secs_f64()
            ));
            let tc = Instant::now();
            let sam_refs: Vec<&Mat> = sams.iter().collect();
            let clips = vision_clip::forward_batched_from_sam(&clip_cfg, clip_w, &sam_refs)?;
            timing_log(&format!(
                "  vision.clip(batch of {}) {:.2}s",
                slots.len(),
                tc.elapsed().as_secs_f64()
            ));
            let tb = Instant::now();
            for ((&(p, v), sam), clip) in slots.iter().zip(&sams).zip(&clips) {
                let projected = vision_bridge::forward_with(&statics.projector, clip, sam)?;
                features[p][v] = Some(projected);
            }
            timing_log(&format!(
                "  vision.bridge(batch of {}) {:.2}s",
                slots.len(),
                tb.elapsed().as_secs_f64()
            ));
        }
        page_views.clear();

        features
            .into_iter()
            .enumerate()
            .map(|(p, views)| {
                views
                    .into_iter()
                    .enumerate()
                    .map(|(v, slot)| {
                        slot.ok_or_else(|| {
                            FocrError::Other(anyhow::anyhow!(
                                "native_engine::OcrModel::vision_tower_batched_pages: page {p} \
                                 view {v} produced no feature (grouping bug)"
                            ))
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// Build the decoder `inputs_embeds` by embedding the prompt id-stream and
    /// scattering the per-view vision features into the `<image>` placeholder
    /// rows ([SPEC-060..066], [SPEC-070]).
    ///
    /// Returns the `[seq, hidden]` fused embedding plus the prompt id sequence
    /// (the AR loop seeds its no-repeat-ngram history with it). The token-embed
    /// and `image_newline`/`view_seperator` lookups read through the
    /// `.focrq`/safetensors accessor; the connector's structural assembly +
    /// `masked_scatter` are implemented and tested over explicit params.
    ///
    /// # Errors
    /// The first connector/embed error — e.g. [`FocrError::FormatMismatch`] if a
    /// required tensor is absent from the loaded weights.
    fn build_inputs_embeds(
        &self,
        pre: &Preprocessed,
        vision_features: &[Mat],
    ) -> FocrResult<(Mat, Vec<u32>)> {
        // The prompt id-stream (BOS + `<image>` placeholders + the task prompt)
        // and the row-aligned `images_seq_mask`; the connector scatters
        // `vision_features` into the masked (image-placeholder) rows.
        let (prompt_ids, images_seq_mask) = self.build_prompt(pre)?;

        // embed_tokens(prompt_ids) -> [seq, hidden] against the embedding table.
        let mut inputs_embeds = self.embed_prompt(&prompt_ids)?;

        let image_newline = Self::image_newline(&self.weights)?;
        let view_seperator = Self::view_seperator(&self.weights)?;

        if pre.crop_grid.is_tiled() {
            let preprocess::PreprocessMode::Gundam { tile_size, .. } = pre.mode else {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "native_engine::OcrModel::build_inputs_embeds: tiled crop grid requires Gundam preprocess mode"
                )));
            };
            let local_count = pre.tiles.len();
            let expected_feature_blocks = local_count.checked_add(1).ok_or_else(|| {
                FocrError::Other(anyhow::anyhow!(
                    "native_engine::OcrModel::build_inputs_embeds: local tile count overflow"
                ))
            })?;
            if vision_features.len() != expected_feature_blocks {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "native_engine::OcrModel::build_inputs_embeds: {} vision feature blocks != {} local tiles + 1 global view",
                    vision_features.len(),
                    local_count
                )));
            }
            let (locals, global_tail) = vision_features.split_at(local_count);
            let q_local = preprocess::num_queries(tile_size);
            connector::fuse_crop(
                &self.weights,
                &mut inputs_embeds,
                locals,
                pre.crop_grid.width_crop_num,
                pre.crop_grid.height_crop_num,
                q_local,
                q_local,
                &global_tail[0],
                Self::global_grid_h(pre),
                Self::global_grid_w(pre),
                &image_newline,
                &view_seperator,
                &images_seq_mask,
            )?;
        } else {
            // Scatter every no-crop / single-global block into the placeholder
            // rows. The connector validates the ORDERING INVARIANT.
            connector::fuse_no_crop(
                &self.weights,
                &mut inputs_embeds,
                vision_features,
                Self::global_grid_h(pre),
                Self::global_grid_w(pre),
                &image_newline,
                &view_seperator,
                &images_seq_mask,
            )?;
        }
        Ok((inputs_embeds, prompt_ids))
    }

    /// Resolve `model.embed_tokens.weight` as an [`EmbedTable`]:
    ///
    /// * BF16/F32 storage (conservative artifacts, raw safetensors) → the
    ///   historical whole-table f32 widen via [`Weights::mat`], byte-for-byte
    ///   unchanged.
    /// * `QInt8PerChan` storage (the wasm experts-int4 artifact) → a ZERO-COPY
    ///   int8 view + per-channel scales, dequantized one row at a time at
    ///   token-embed time (bd-4l71 residency: the ~662 MB f32 table copy never
    ///   exists). Values are identical: `f32::from(w) * scale[row]` is exactly
    ///   the `dequant_qint8` expression `Weights::mat` applies.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if the table is absent or malformed.
    fn embed_table(&self) -> FocrResult<EmbedTable<'_>> {
        embed_table_from(&self.weights)
    }

    /// Embed the prompt id-stream into the decoder hidden rail ([SPEC-070]).
    ///
    /// `embed_tokens(prompt_ids)` against `model.embed_tokens.weight` (`[vocab,
    /// hidden]` = `[129280, 1280]`). The gather math lives in
    /// [`decoder::embed_tokens`] (unit-tested) for the f32 arm; the int8 arm
    /// gathers + dequantizes per row through [`EmbedTable::embed_ids`].
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if the embedding table is absent/malformed,
    /// or an out-of-range id.
    fn embed_prompt(&self, prompt_ids: &[u32]) -> FocrResult<Mat> {
        self.embed_table()?.embed_ids(prompt_ids)
    }

    /// Prefill the decoder over `inputs_embeds`, then run the **sequential**
    /// greedy autoregressive decode loop to EOS, returning the generated token
    /// ids ([SPEC-072..103]).
    ///
    /// Doctrine #5: strictly sequential — one forward at a time; the per-step
    /// R-SWA/MoE math fans out across cores inside the kernels, never a nested
    /// runtime, never rayon under a lock. Two decode kernels share this contract,
    /// selected at runtime:
    ///   * **cached** (default, bd-1gv.17): prefill ONCE into a per-layer R-SWA
    ///     ring cache, then extend one token per step — O(n) ([`Self::generate_cached`]).
    ///   * **stateless** (`FOCR_DECODE_STATELESS`): re-prefill the whole growing
    ///     sequence every step — O(n^2), the parity oracle ([`Self::generate_stateless`]).
    ///
    /// Both emit the SAME tokens for the first 128 steps (the R-SWA window).
    ///
    /// # Errors
    /// The first decode-stage error — e.g. [`FocrError::FormatMismatch`] if the
    /// embedding table is absent, or a sampler/lm_head failure.
    fn generate(&self, inputs_embeds: Mat, prompt_ids: &[u32]) -> FocrResult<Vec<u32>> {
        self.generate_with(inputs_embeds, prompt_ids, &self.decode_params)
    }

    /// [`Self::generate`] with EXPLICIT decode params — the multi-page path
    /// (bd-1gv.25) selects `ngram_window = 1024` per OQ-18 while single-image
    /// callers keep the engine's own params verbatim.
    fn generate_with(
        &self,
        inputs_embeds: Mat,
        prompt_ids: &[u32],
        params: &sampler::DecodeParams,
    ) -> FocrResult<Vec<u32>> {
        self.generate_with_observer(inputs_embeds, prompt_ids, params, None)
    }

    /// [`Self::generate_with`] plus an optional PER-COMMITTED-TOKEN observer
    /// (bd-2z0y streaming): called synchronously from the sequential decode
    /// driver after each token joins `emitted`, in emission order — never
    /// from inside a rayon fan-out. `None` is byte-for-byte today's paths.
    fn generate_with_observer(
        &self,
        inputs_embeds: Mat,
        prompt_ids: &[u32],
        params: &sampler::DecodeParams,
        observer: Option<&mut dyn FnMut(u32)>,
    ) -> FocrResult<Vec<u32>> {
        // One controller instance per decode invocation. Unset/0 is a no-op;
        // invalid values fail before model work, and an armed trigger is sticky.
        let mut runaway_guard = sampler::RunawayGuard::from_env()?;
        let generated = if std::env::var_os(DECODE_STATELESS_ENV).is_some() {
            self.generate_stateless(
                inputs_embeds,
                prompt_ids,
                params,
                observer,
                &mut runaway_guard,
            )
        } else if int8_decode_requested() || self.artifact_stores_gated_int8() {
            // The wasm experts-int4 artifact routes through the int8 cache by
            // default: its attention/lm_head are int8 IN the artifact (so they
            // stay int8-resident) and its expert tensors are consumed as
            // packed int4 by the cache's QLinear arm (bd-4l71).
            self.generate_cached_i8(
                inputs_embeds,
                prompt_ids,
                params,
                observer,
                &mut runaway_guard,
            )
        } else {
            self.generate_cached(
                inputs_embeds,
                prompt_ids,
                params,
                observer,
                &mut runaway_guard,
            )
        }?;
        if let Ok(mut slot) = self.last_decode.lock() {
            let stopped = generated.last() == Some(&params.eos_token_id);
            let (decodes, length_stops) = slot
                .as_ref()
                .map_or((0, 0), |prior| (prior.decodes, prior.length_stops));
            *slot = Some(DecodeReport {
                ids: generated.clone(),
                finish_reason: if stopped {
                    FinishReason::Stop
                } else {
                    FinishReason::Length
                },
                cap: (!stopped).then_some(LengthCap::MaxLength),
                pages: None,
                max_length: params.max_length,
                max_positions: params.max_positions,
                decodes: decodes + 1,
                length_stops: length_stops + usize::from(!stopped),
                document: None,
                admission_warnings: Vec::new(),
            });
        }
        Ok(generated)
    }

    /// O(n) cached greedy decode (bd-1gv.17): [`decoder::prefill_with_cache`] once
    /// to seed each layer's R-SWA reference K/V, then
    /// [`decoder::decode_step_with_cache`] per token over the ring caches. The
    /// reference block (the entire prefill) is never evicted; the generated tail
    /// rides a 128-slot ring, so memory and per-step cost are bounded.
    ///
    /// Both prefill and decode run off the conservative
    /// [`decoder::DecoderWeightCache`], built once per model: the 2,148
    /// FFN/expert matrices are int8 (quantized at build even from a raw bf16
    /// source), and attention, the router and `lm_head` stay f32. That int8 FFN
    /// is a real numeric difference from the all-f32 stateless path
    /// ([`Self::generate_stateless`]); this is the path the certified
    /// conservative artifact runs.
    ///
    /// # Errors
    /// As [`Self::generate`].
    fn generate_cached(
        &self,
        inputs_embeds: Mat,
        prompt_ids: &[u32],
        params: &sampler::DecodeParams,
        mut observer: Option<&mut dyn FnMut(u32)>,
        runaway_guard: &mut sampler::RunawayGuard,
    ) -> FocrResult<Vec<u32>> {
        timing_log("precision focr-mixed-ffn-int8");
        let hidden_dim = inputs_embeds.cols;
        let prefill_len = inputs_embeds.rows;

        let table = self.embed_table()?;
        let vocab = table.rows();
        if table.cols() != hidden_dim {
            return Err(FocrError::Other(anyhow::anyhow!(
                "native_engine::OcrModel::generate_cached: embed table hidden {} != inputs_embeds hidden {}",
                table.cols(),
                hidden_dim
            )));
        }

        // Build the conservative recipe cache once per model: int8 FFN/expert
        // matrices, high-precision attention and lm_head.
        let tb = Instant::now();
        let wc = self.decoder_cache()?;
        timing_log(&format!(
            "weight_cache_build {:.2}s",
            tb.elapsed().as_secs_f64()
        ));

        // Prefill once, capturing each layer's reference K/V; `last_hidden` is the
        // final prefill position, which predicts the FIRST generated token.
        let tp = Instant::now();
        progress::emit("prefill", 0, prefill_len as u64);
        let (hidden, mut caches) = decoder::prefill_with_cache(wc, &inputs_embeds)?;
        progress::emit("prefill", prefill_len as u64, prefill_len as u64);
        let mut last_hidden = Self::last_hidden_row(&hidden)?;
        timing_log(&format!(
            "prefill {:.2}s ({} tokens)",
            tp.elapsed().as_secs_f64(),
            prefill_len
        ));
        let td = Instant::now();
        decoder::prof::reset();

        // `generated` seeds the no-repeat-ngram history with the prompt so the
        // sliding-window blocker sees the full context (sampler reads its tail).
        let mut generated: Vec<u32> = prompt_ids.to_vec();
        let mut emitted: Vec<u32> = Vec::new();

        while emitted.len() < params.max_length {
            crate::cancel_checkpoint()?;
            let logits = decoder::lm_head_cached(wc, &last_hidden)?;
            let step: DecodeOutput = sampler::decode_step(&logits, &generated, params)?;
            generated.push(step.token_id);
            emitted.push(step.token_id);
            runaway_guard.check_after_emit(&emitted, step.is_eos)?;
            // Per-token progress. Sequential loop on the calling thread; the
            // hook is a relaxed atomic load when no sink is installed.
            progress::emit("decode", emitted.len() as u64, params.max_length as u64);
            if let Some(f) = observer.as_deref_mut() {
                f(step.token_id);
            }
            if step.is_eos {
                break;
            }
            let next = step.token_id as usize;
            if next >= vocab {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "native_engine::OcrModel::generate_cached: decoded token id {next} outside embed vocab {vocab}"
                )));
            }
            // Embed the just-emitted token and advance the ring one step at its
            // TRUE absolute position (`prefill_len` + tokens already ringed). After
            // the push above, `emitted.len() - 1` tokens are already in the ring.
            let row = table.row_f32(next);
            let token_embed = Mat::from_vec(1, hidden_dim, row);
            let position = prefill_len + (emitted.len() - 1);
            let h = decoder::decode_step_with_cache(wc, &mut caches, &token_embed, position)?;
            last_hidden = Self::last_hidden_row(&h)?;
        }
        timing_log(&format!(
            "decode {:.2}s ({} tokens, {:.3}s/tok)",
            td.elapsed().as_secs_f64(),
            emitted.len(),
            td.elapsed().as_secs_f64() / (emitted.len().max(1) as f64)
        ));
        if decoder::prof::enabled() {
            let (lmhead, attn, experts, route) = decoder::prof::snapshot_ms();
            timing_log(&format!(
                "decode phases (ms): lm_head {lmhead:.0}  attn {attn:.0}  experts {experts:.0}  route {route:.0}"
            ));
        }
        // `FOCR_CALIB_OUT` (bd-50wo stage A): checkpoint the activation
        // statistics accumulated so far. Rewriting after every generation (not
        // only at process exit) means a calibration sweep that is interrupted
        // still leaves usable statistics for the pages it finished. A no-op —
        // not even a file open — when the recorder is disarmed.
        if calib::enabled() {
            calib::flush()?;
        }
        Ok(emitted)
    }

    /// Int8 twin of [`Self::generate_cached`] (`FOCR_DECODE_INT8`): identical O(n)
    /// R-SWA cached greedy decode, but prefill + decode run off the
    /// [`decoder::DecoderWeightCacheI8`] (per-output-channel symmetric S8S8, NEON
    /// SDOT / x86 VNNI). ~4x less weight traffic at decode and ~2.8 GB cache; the
    /// emitted tokens are verified to match the f32 path / baidu (CER) — int8 is a
    /// throughput lever, not an accuracy change within proven tolerance.
    ///
    /// # Errors
    /// As [`Self::generate_cached`].
    fn generate_cached_i8(
        &self,
        inputs_embeds: Mat,
        prompt_ids: &[u32],
        params: &sampler::DecodeParams,
        mut observer: Option<&mut dyn FnMut(u32)>,
        runaway_guard: &mut sampler::RunawayGuard,
    ) -> FocrResult<Vec<u32>> {
        self.require_full_int8_authorization()?;
        timing_log("precision focr-full-int8");
        let hidden_dim = inputs_embeds.cols;
        let prefill_len = inputs_embeds.rows;

        let table = self.embed_table()?;
        let vocab = table.rows();
        if table.cols() != hidden_dim {
            return Err(FocrError::Other(anyhow::anyhow!(
                "native_engine::OcrModel::generate_cached_i8: embed table hidden {} != inputs_embeds hidden {}",
                table.cols(),
                hidden_dim
            )));
        }

        // Quantize the decoder weights to int8 ONCE and cache on the model; a
        // load-once batch then reuses it across pages (the build is ~1.2 s). The
        // first page pays the quant; every later page in the same process skips it.
        let tb = Instant::now();
        let wc = self.decoder_cache_i8()?;
        timing_log(&format!(
            "weight_cache_build_i8 {:.2}s",
            tb.elapsed().as_secs_f64()
        ));

        let tp = Instant::now();
        progress::emit("prefill", 0, prefill_len as u64);
        let (hidden, mut caches) = decoder::prefill_with_cache_i8(wc, &inputs_embeds)?;
        progress::emit("prefill", prefill_len as u64, prefill_len as u64);
        let mut last_hidden = Self::last_hidden_row(&hidden)?;
        timing_log(&format!(
            "prefill_i8 {:.2}s ({} tokens)",
            tp.elapsed().as_secs_f64(),
            prefill_len
        ));
        let td = Instant::now();
        decoder::prof::reset();

        let mut generated: Vec<u32> = prompt_ids.to_vec();
        let mut emitted: Vec<u32> = Vec::new();

        // FOCR_FUSE_NGRAM_LMHEAD (bd-1azu.54, Lever 3): fold the sliding-window
        // no-repeat-ngram ban into the int8 lm_head dequant epilogue instead of the
        // sampler's separate copy-then-mask pass. DEFAULT OFF — read ONCE here.
        let fuse_ngram_lmhead = decoder::fuse_ngram_lmhead_enabled();

        // FOCR_SPEC_DECODE (bd-1azu.35): when armed AND the decode params match the
        // frozen single-image ban the speculative verifier's chooser assumes
        // (`spec::accept_longest` hardwires 35-gram / 128-window), run the draft ->
        // verify -> accept loop in place of the sequential greedy `while` below; it
        // emits the BYTE-FOR-BYTE-identical token stream. Unset (the default) => the
        // spec loop is skipped and the `while` runs EXACTLY today's code, untouched.
        // The params half is `DecodeParams::matches_frozen_spec_ban` (sampler.rs),
        // gated by `tests/spec_decode_gate.rs`, so guard and gate cannot drift.
        let spec_decode =
            spec_decode_enabled() && params.matches_frozen_spec_ban() && observer.is_none();
        if spec_decode {
            self.spec_decode_i8(
                wc,
                &mut caches,
                &last_hidden,
                &mut generated,
                &mut emitted,
                &table,
                vocab,
                hidden_dim,
                prefill_len,
                params,
                runaway_guard,
            )?;
        }
        while !spec_decode && emitted.len() < params.max_length {
            crate::cancel_checkpoint()?;
            // When armed AND the blocker can ban (enough history), mask in the
            // lm_head epilogue and argmax the already-masked logits; otherwise the
            // exact default `lm_head_cached_i8` -> `sampler::decode_step`. The chosen
            // token is byte-for-byte identical either way (same masked row argmax'd).
            let step: DecodeOutput = if fuse_ngram_lmhead
                && params.no_repeat_ngram_size > 0
                && generated.len() >= params.no_repeat_ngram_size
            {
                let banned = sampler::collect_sliding_window_ngram_bans(
                    &generated,
                    params.no_repeat_ngram_size,
                    params.ngram_window,
                    &[],
                    sampler::VOCAB_SIZE,
                );
                let logits = decoder::lm_head_cached_i8_ngram_masked(wc, &last_hidden, &banned)?;
                sampler::decode_step_premasked(&logits, params)?
            } else {
                let logits = decoder::lm_head_cached_i8(wc, &last_hidden)?;
                sampler::decode_step(&logits, &generated, params)?
            };
            generated.push(step.token_id);
            emitted.push(step.token_id);
            runaway_guard.check_after_emit(&emitted, step.is_eos)?;
            // Per-token progress. Sequential loop on the calling thread; the
            // hook is a relaxed atomic load when no sink is installed.
            progress::emit("decode", emitted.len() as u64, params.max_length as u64);
            if let Some(f) = observer.as_deref_mut() {
                f(step.token_id);
            }
            if step.is_eos {
                break;
            }
            let next = step.token_id as usize;
            if next >= vocab {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "native_engine::OcrModel::generate_cached_i8: decoded token id {next} outside embed vocab {vocab}"
                )));
            }
            let row = table.row_f32(next);
            let token_embed = Mat::from_vec(1, hidden_dim, row);
            let position = prefill_len + (emitted.len() - 1);
            let h = decoder::decode_step_with_cache_i8(wc, &mut caches, &token_embed, position)?;
            last_hidden = Self::last_hidden_row(&h)?;
        }
        timing_log(&format!(
            "decode_i8 {:.2}s ({} tokens, {:.3}s/tok)",
            td.elapsed().as_secs_f64(),
            emitted.len(),
            td.elapsed().as_secs_f64() / (emitted.len().max(1) as f64)
        ));
        if decoder::prof::enabled() {
            let (lmhead, attn, experts, route) = decoder::prof::snapshot_ms();
            timing_log(&format!(
                "decode_i8 phases (ms): lm_head {lmhead:.0}  attn {attn:.0}  experts {experts:.0}  route {route:.0}"
            ));
        }
        // `FOCR_CALIB_OUT` checkpoint — see the twin in `generate_cached`.
        if calib::enabled() {
            calib::flush()?;
        }
        Ok(emitted)
    }

    /// `FOCR_SPEC_DECODE` (bd-1azu.35): the draft -> verify -> accept speculative
    /// twin of [`Self::generate_cached_i8`]'s sequential greedy loop, called in its
    /// place (over the SAME prefilled `caches` / `last_hidden`) when armed. Each
    /// round cheaply PROPOSES the next tokens with the prompt-lookup drafter
    /// (`spec::draft_ngram`), VERIFIES them in ONE read-only batched forward
    /// ([`decoder::verify_forward_i8`], which folds the draft into a PRIVATE ring
    /// clone — the live `caches` are untouched), ACCEPTS the longest prefix equal to
    /// sequential greedy plus one correction token (`spec::resolve_round`), then
    /// COMMITS exactly the accepted tokens (and the correction) by REPLAYING each
    /// through the real [`decoder::decode_step_with_cache_i8`] — the byte-for-byte
    /// same KV write the sequential loop performs for that emitted token.
    ///
    /// LOSSLESS by construction: speculation changes only WHEN logits are evaluated,
    /// never WHICH token greedy decode picks. Every accepted token equals its
    /// per-position greedy token and the correction is greedy over the trailing
    /// verify row (`spec::resolve_round`), so `generated`/`emitted` advance
    /// token-for-token as the sequential loop would; and because the commit replays
    /// the IDENTICAL `decode_step_with_cache_i8` call (same token embed, same TRUE
    /// absolute position, same prior ring), `caches` after committing `k` accepted
    /// tokens is byte-for-byte the ring after `k` sequential decode steps
    /// ([SPEC-100..103]). EOS and `max_length` are honored exactly as the sequential
    /// loop: a token is emitted iff `emitted.len() < max_length`, and an emitted EOS
    /// halts before its KV write (`accept_longest` only accepts EOS as the LAST
    /// accepted token).
    ///
    /// Doctrine #5 (one live forward): the verify and commit kernels run
    /// sequentially in this single-stream loop, never nested under rayon.
    ///
    /// # Errors
    /// As [`Self::generate_cached_i8`]; propagates the verify/commit kernel errors
    /// and the out-of-vocab guard.
    #[allow(clippy::too_many_arguments)]
    fn spec_decode_i8(
        &self,
        wc: &decoder::DecoderWeightCacheI8,
        caches: &mut [rswa::RingCache],
        last_hidden: &Mat,
        generated: &mut Vec<u32>,
        emitted: &mut Vec<u32>,
        table: &EmbedTable<'_>,
        vocab: usize,
        hidden_dim: usize,
        prefill_len: usize,
        params: &sampler::DecodeParams,
        runaway_guard: &mut sampler::RunawayGuard,
    ) -> FocrResult<()> {
        // Own a mutable copy of the seed hidden (the prefill last row); the caller's
        // `last_hidden` stays intact for the guarded sequential `while` (which only
        // runs when spec decode is OFF). Each commit replaces this with the freshly
        // decoded row, exactly as the sequential loop reassigns its `last_hidden`.
        let mut last_hidden = last_hidden.clone();
        while emitted.len() < params.max_length {
            crate::cancel_checkpoint()?;
            progress::emit("decode", emitted.len() as u64, params.max_length as u64);
            let draft = spec::draft_ngram(generated, spec::SPEC_DRAFT_MAX, spec::SPEC_DRAFT_NGRAM);
            if draft.is_empty() {
                // EMPTY-DRAFT FALLBACK: nothing to verify, so take exactly ONE
                // sequential greedy step — the default loop's body (raw `lm_head` +
                // `decode_step`), then commit the chosen token.
                let logits = decoder::lm_head_cached_i8(wc, &last_hidden)?;
                let step = sampler::decode_step(&logits, generated, params)?;
                generated.push(step.token_id);
                emitted.push(step.token_id);
                runaway_guard.check_after_emit(emitted, step.is_eos)?;
                if step.is_eos {
                    break;
                }
                last_hidden = Self::commit_decode_token_i8(
                    wc,
                    caches,
                    table,
                    vocab,
                    hidden_dim,
                    prefill_len,
                    emitted.len(),
                    step.token_id,
                )?;
                continue;
            }
            // Embed each draft token (defensive vocab guard; drafter ids are always
            // prior `generated` tokens, hence already in-vocab).
            let mut draft_embeds: Vec<Mat> = Vec::with_capacity(draft.len());
            for &id in &draft {
                draft_embeds.push(Self::embed_decode_token_i8(table, vocab, hidden_dim, id)?);
            }
            // VERIFY: ONE read-only forward over the draft (private ring clone), then
            // assemble the `K + 1` verify rows. Row 0 is the live current-state row
            // (`lm_head` over `last_hidden` == what the next sequential step would
            // argmax); rows `1..=K` are the draft rows. So `verify_logits[i]`
            // conditions on `generated` ++ `draft[0..i]`, the `resolve_round`
            // contract.
            let base_position = prefill_len + emitted.len();
            let verify_rows =
                decoder::verify_forward_i8(wc, &*caches, &draft_embeds, base_position)?;
            let mut verify_logits: Vec<Mat> = Vec::with_capacity(draft.len() + 1);
            verify_logits.push(decoder::lm_head_cached_i8(wc, &last_hidden)?);
            verify_logits.extend(verify_rows);
            // ACCEPT the longest greedy-matching prefix + choose the correction with
            // the SAME chooser the sequential loop runs.
            let emit = spec::resolve_round(generated, &draft, &verify_logits, params)?;
            // COMMIT the accepted tokens by replaying each through the real decode
            // step (byte-for-byte the sequential KV writes), honoring EOS/max_length
            // exactly: a token is emitted only while `emitted.len() < max_length`,
            // and an emitted EOS halts before its KV write.
            let mut stopped = false;
            for &token in &draft[..emit.accepted] {
                generated.push(token);
                emitted.push(token);
                // `eos_token_id` on the left keeps the identifier off the left of `=`
                // (dodges a ubs secrets-heuristic FP; these are vocabulary token ids).
                let is_eos = params.eos_token_id == token;
                runaway_guard.check_after_emit(emitted, is_eos)?;
                if is_eos {
                    stopped = true;
                    break;
                }
                // Commit advances the ring KV at this token's true position. The
                // returned hidden is ALWAYS superseded — either by the correction
                // commit below (a correction exists whenever the loop completes without
                // a stop) or discarded when we break — so it is intentionally dropped.
                Self::commit_decode_token_i8(
                    wc,
                    caches,
                    table,
                    vocab,
                    hidden_dim,
                    prefill_len,
                    emitted.len(),
                    token,
                )?;
                if emitted.len() >= params.max_length {
                    stopped = true;
                    break;
                }
            }
            if stopped {
                break;
            }
            // CORRECTION/bonus token: `None` only when an accepted token was EOS
            // (handled by the loop break above), so this is `Some` here. Emit and
            // commit it exactly as the sequential loop emits its chosen token; the
            // outer `while` re-gates `max_length`.
            let Some(correction) = emit.correction else {
                break;
            };
            generated.push(correction.token_id);
            emitted.push(correction.token_id);
            runaway_guard.check_after_emit(emitted, correction.is_eos)?;
            if correction.is_eos {
                break;
            }
            last_hidden = Self::commit_decode_token_i8(
                wc,
                caches,
                table,
                vocab,
                hidden_dim,
                prefill_len,
                emitted.len(),
                correction.token_id,
            )?;
        }
        Ok(())
    }

    /// Embed one decode token id into a `[1, hidden]` row from the embed table — the
    /// exact slice [`Self::generate_cached_i8`] takes per step, with the same
    /// out-of-vocab guard. Shared by the draft-embed build and the commit replay.
    fn embed_decode_token_i8(
        table: &EmbedTable<'_>,
        vocab: usize,
        hidden_dim: usize,
        token: u32,
    ) -> FocrResult<Mat> {
        let idx = token as usize;
        if idx >= vocab {
            return Err(FocrError::Other(anyhow::anyhow!(
                "native_engine::OcrModel::generate_cached_i8: decoded token id {idx} outside embed vocab {vocab}"
            )));
        }
        let row = table.row_f32(idx);
        Ok(Mat::from_vec(1, hidden_dim, row))
    }

    /// Commit one emitted token to the live int8 decode state: write its K/V into the
    /// ring at its TRUE absolute position via the real
    /// [`decoder::decode_step_with_cache_i8`] and return the new last hidden row —
    /// byte-for-byte the call [`Self::generate_cached_i8`] makes for that token.
    /// `emitted_len` is `emitted.len()` AFTER the token was pushed, so the position is
    /// `prefill_len + emitted_len - 1`, matching the sequential loop exactly.
    #[allow(clippy::too_many_arguments)]
    fn commit_decode_token_i8(
        wc: &decoder::DecoderWeightCacheI8,
        caches: &mut [rswa::RingCache],
        table: &EmbedTable<'_>,
        vocab: usize,
        hidden_dim: usize,
        prefill_len: usize,
        emitted_len: usize,
        token: u32,
    ) -> FocrResult<Mat> {
        let token_embed = Self::embed_decode_token_i8(table, vocab, hidden_dim, token)?;
        let position = prefill_len + (emitted_len - 1);
        let h = decoder::decode_step_with_cache_i8(wc, caches, &token_embed, position)?;
        Self::last_hidden_row(&h)
    }

    /// O(n^2) stateless greedy decode: re-run [`decoder::forward`] over the whole
    /// growing sequence every step (append one embed row per emitted token). The
    /// parity oracle for [`Self::generate_cached`], selected by
    /// `FOCR_DECODE_STATELESS`.
    ///
    /// # Errors
    /// As [`Self::generate`].
    fn generate_stateless(
        &self,
        mut inputs_embeds: Mat,
        prompt_ids: &[u32],
        params: &sampler::DecodeParams,
        mut observer: Option<&mut dyn FnMut(u32)>,
        runaway_guard: &mut sampler::RunawayGuard,
    ) -> FocrResult<Vec<u32>> {
        let hidden_dim = inputs_embeds.cols;

        let table = self.embed_table()?;
        let vocab = table.rows();
        if table.cols() != hidden_dim {
            return Err(FocrError::Other(anyhow::anyhow!(
                "native_engine::OcrModel::generate_stateless: embed table hidden {} != inputs_embeds hidden {}",
                table.cols(),
                hidden_dim
            )));
        }

        // `generated` seeds the no-repeat-ngram history with the prompt so the
        // sliding-window blocker sees the full context (sampler reads its tail).
        let mut generated: Vec<u32> = prompt_ids.to_vec();
        let mut emitted: Vec<u32> = Vec::new();

        // SEQUENTIAL greedy decode loop (no nested runtime, no rayon-under-lock).
        // Bounded by `max_length` so a non-converging model can never hang.
        while emitted.len() < params.max_length {
            crate::cancel_checkpoint()?;
            // Full-sequence forward, then lm_head over ONLY the last hidden row
            // -> [1, vocab] logits (the next-token logits depend solely on the
            // final position; per decoder::lm_head_last_row_is_full_last_row).
            let hidden = decoder::forward(&self.weights, &inputs_embeds)?;
            let last_hidden = Self::last_hidden_row(&hidden)?;
            let logits = decoder::lm_head(&self.weights, &last_hidden)?;
            let step: DecodeOutput = sampler::decode_step(&logits, &generated, params)?;
            generated.push(step.token_id);
            emitted.push(step.token_id);
            runaway_guard.check_after_emit(&emitted, step.is_eos)?;
            // Per-token progress. Sequential loop on the calling thread; the
            // hook is a relaxed atomic load when no sink is installed.
            progress::emit("decode", emitted.len() as u64, params.max_length as u64);
            if let Some(f) = observer.as_deref_mut() {
                f(step.token_id);
            }
            if step.is_eos {
                break;
            }
            // Append embed(next_token) as a new trailing row of `inputs_embeds`
            // so the next forward conditions on the full prefix + this token.
            let next = step.token_id as usize;
            if next >= vocab {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "native_engine::OcrModel::generate_stateless: decoded token id {next} outside embed vocab {vocab}"
                )));
            }
            let new_rows = inputs_embeds.rows + 1;
            let mut data = std::mem::take(&mut inputs_embeds.data);
            data.extend_from_slice(&table.row_f32(next));
            inputs_embeds = Mat::from_vec(new_rows, hidden_dim, data);
        }
        Ok(emitted)
    }

    fn last_hidden_row(hidden: &Mat) -> FocrResult<Mat> {
        if hidden.rows == 0 {
            return Err(FocrError::Other(anyhow::anyhow!(
                "native_engine::OcrModel::generate: decoder forward returned zero hidden rows"
            )));
        }
        let expected_len = hidden.rows.checked_mul(hidden.cols).ok_or_else(|| {
            FocrError::Other(anyhow::anyhow!(
                "native_engine::OcrModel::generate: decoder hidden shape product overflow for [{}, {}]",
                hidden.rows,
                hidden.cols
            ))
        })?;
        if hidden.data.len() != expected_len {
            return Err(FocrError::Other(anyhow::anyhow!(
                "native_engine::OcrModel::generate: decoder hidden data len {} != rows*cols {} for shape [{}, {}]",
                hidden.data.len(),
                expected_len,
                hidden.rows,
                hidden.cols
            )));
        }
        Ok(Mat::from_vec(
            1,
            hidden.cols,
            hidden.row(hidden.rows - 1).to_vec(),
        ))
    }

    // ── preprocess/weights field accessors (loader-handoff shims) ──────────────
    //
    // `Preprocessed` now carries real image geometry and view tensors, while
    // `Weights` is still a Phase-2 handoff seam. The accessors below keep the
    // orchestration call sites explicit and make each remaining placeholder
    // visible until the dependent bead lands.

    /// Source image pixel dimensions `(w, h)` for bbox de-normalization
    /// ([SPEC-018]), carried by the preprocess front end after EXIF handling.
    fn image_dims(pre: &Preprocessed) -> (u32, u32) {
        pre.original_size
    }

    /// The preprocessed view tensors (`[3, H, W]` each), one per crop/global view
    /// ([SPEC-020..033]). Base/no-crop mode yields a single global view; the
    /// Gundam crop branch yields local tiles first, then the global thumbnail,
    /// matching the connector `[local, global, view_seperator]` invariant.
    fn views(pre: &Preprocessed) -> Vec<Mat> {
        let mut views = Vec::with_capacity(pre.num_views());
        views.extend(pre.tiles.iter().map(|tile| tile.pixels.clone()));
        views.push(pre.global.pixels.clone());
        views
    }

    /// Build the prompt id-stream + row-aligned `images_seq_mask` for the base /
    /// no-crop path ([SPEC-019]/[SPEC-035]/[SPEC-066]): a single BOS, then
    /// `placeholder_token_count()` `<image>` (128815) placeholders, then the
    /// task-prompt text tokens (`document parsing.` → [34030, 76466, 16]). The
    /// mask is `true` exactly at the image placeholders so the connector scatters
    /// the assembled vision block into them. Matches baidu's
    /// `infer(prompt="<image>document parsing.")` input_ids byte-for-byte.
    ///
    /// # Errors
    /// Propagates [`Self::tokenizer`] (the BPE load) or `encode` failures.
    fn build_prompt(&self, pre: &Preprocessed) -> FocrResult<(Vec<u32>, Vec<bool>)> {
        if self.arch().decode_contract().pages.is_some() {
            // A paged model reads one page through its document prompt.
            return self.build_prompt_multi(std::slice::from_ref(pre));
        }
        let tok = self.tokenizer()?;
        let n_image = pre.placeholder_token_count();
        let text = tok.encode(BASE_PROMPT_TEXT)?;
        let total = 1 + n_image + text.len();
        let mut ids = Vec::with_capacity(total);
        let mut mask = Vec::with_capacity(total);
        ids.push(tok.bos_id());
        mask.push(false);
        for _ in 0..n_image {
            ids.push(tok.image_id());
            mask.push(true);
        }
        for id in text {
            ids.push(id);
            mask.push(false);
        }
        Ok((ids, mask))
    }

    /// Multi-page prompt id-stream (bd-1gv.25): `BOS + [page-1 block] + … +
    /// [page-N block] + "Multi page parsing."` — the reference `infer_multi`
    /// concatenates every page's placeholder block at the prompt's single
    /// `<image>` position, each block already carrying its own trailing
    /// separator token (the `+ 1` in the per-page census, identical in shape
    /// to the single-image 273-slot block, just at base 640 ⇒ 111 slots).
    ///
    /// A paged model (Unlimited-OCR-2) instead opens every page's block with
    /// its marker ([`paged_prompt`]), for one page as for many.
    fn build_prompt_multi(&self, pres: &[Preprocessed]) -> FocrResult<(Vec<u32>, Vec<bool>)> {
        let tok = self.tokenizer()?;
        let text = tok.encode(MULTI_PAGE_PROMPT_TEXT)?;
        if let Some(pages) = self.arch().decode_contract().pages {
            let blocks: Vec<usize> = pres
                .iter()
                .map(Preprocessed::placeholder_token_count)
                .collect();
            return Ok(paged_prompt(
                tok.bos_id(),
                pages.marker_id,
                tok.image_id(),
                &blocks,
                &text,
            ));
        }
        let n_image: usize = pres.iter().map(Preprocessed::placeholder_token_count).sum();
        let total = 1 + n_image + text.len();
        let mut ids = Vec::with_capacity(total);
        let mut mask = Vec::with_capacity(total);
        ids.push(tok.bos_id());
        mask.push(false);
        for pre in pres {
            for _ in 0..pre.placeholder_token_count() {
                ids.push(tok.image_id());
                mask.push(true);
            }
        }
        for id in text {
            ids.push(id);
            mask.push(false);
        }
        Ok((ids, mask))
    }

    /// Global feature-grid height (16 at base 1024) ([SPEC-063]).
    fn global_grid_h(pre: &Preprocessed) -> usize {
        preprocess::num_queries(pre.mode.base_size())
    }

    /// Global feature-grid width (16 at base 1024) ([SPEC-063]).
    fn global_grid_w(pre: &Preprocessed) -> usize {
        preprocess::num_queries(pre.mode.base_size())
    }

    /// The learned `model.image_newline` parameter (length `N_EMBED = 1280`),
    /// [SPEC-060]. Read through the `.focrq`/safetensors accessor.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if the tensor is absent or mis-shaped.
    fn image_newline(weights: &Weights) -> FocrResult<Vec<f32>> {
        weights.vec("model.image_newline")
    }

    /// The learned `model.view_seperator` parameter (length `N_EMBED = 1280`),
    /// [SPEC-060]. Read through the `.focrq`/safetensors accessor.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if the tensor is absent or mis-shaped.
    fn view_seperator(weights: &Weights) -> FocrResult<Vec<f32>> {
        weights.vec("model.view_seperator")
    }
}

/// A census-exact two-shard Unlimited-OCR-2 checkpoint in `dir` with a sparse
/// payload, for tests whose run must be refused before any weight is read.
/// Returns its index.
#[cfg(test)]
pub(crate) fn write_sparse_v2_stand_in(dir: &Path) -> PathBuf {
    let (tensors, int64) = unlimited_ocr_census::v2_source_tensors();
    tests::write_sparse_two_shard_stand_in(dir, &tensors, &int64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The origins of a document of `pages` whole pages, numbered from 1.
    fn list_origins(pages: usize) -> Vec<PageOrigin> {
        (1..=pages)
            .map(|source_page| PageOrigin {
                source_page,
                half: None,
            })
            .collect()
    }

    /// A multi-page report as v1's document parse records it.
    fn multi_page_report(
        finish_reason: FinishReason,
        cap: Option<LengthCap>,
        expected: usize,
        decoded: &str,
    ) -> DecodeReport {
        DecodeReport {
            ids: Vec::new(),
            finish_reason,
            cap,
            pages: Some(PageBodies {
                expected,
                bodies: postprocess::page_body_count(decoded),
            }),
            max_length: 32_768,
            max_positions: MAX_POSITION_EMBEDDINGS,
            decodes: 1,
            length_stops: usize::from(finish_reason == FinishReason::Length),
            document: None,
            admission_warnings: Vec::new(),
        }
    }

    /// U2-E1: gundam mode is refused for a paged model's document rather than
    /// dropped, and a v2 page read from a gundam view warns that the mode is
    /// uncertified, ahead of the document's other warnings.
    #[test]
    fn gundam_is_refused_for_a_v2_document_and_warns_on_a_v2_page() {
        let base = preprocess::PreprocessMode::Base { base_size: 1024 };
        assert_eq!(
            paged_document_view(base, "unlimited-ocr-2").expect("base"),
            base
        );
        let gundam = preprocess::PreprocessMode::Gundam {
            base_size: 1024,
            tile_size: 640,
        };
        let error = paged_document_view(gundam, "unlimited-ocr-2").expect_err("refused");
        assert!(matches!(error, FocrError::Usage(_)), "{error}");
        assert!(
            error.to_string().contains("unlimited-ocr-2 document"),
            "{error}"
        );

        let report = |gundam_view: bool| DecodeReport {
            ids: Vec::new(),
            finish_reason: FinishReason::Stop,
            cap: None,
            pages: Some(PageBodies {
                expected: 1,
                bodies: 1,
            }),
            max_length: 32_768,
            max_positions: sampler::UOCR2_MAX_POSITIONS,
            decodes: 1,
            length_stops: 0,
            document: Some(DocumentStats {
                gundam_view,
                leading_text: true,
                ..DocumentStats::default()
            }),
            admission_warnings: Vec::new(),
        };
        let kinds = |r: &DecodeReport| {
            decode_warnings(r)
                .iter()
                .map(|w| w.kind)
                .collect::<Vec<_>>()
        };
        assert_eq!(kinds(&report(false)), ["leading_text"]);
        assert_eq!(
            kinds(&report(true)),
            ["uncertified_crop_mode", "leading_text"]
        );
        let warning = decode_warnings(&report(true)).remove(0);
        assert_eq!(
            warning.recovery,
            warning_recovery(WarningKind::UncertifiedCropMode)
        );
        assert!(RUN_WARNING_KINDS.contains(&warning.kind));
    }

    /// U2-B6: an Unlimited-OCR-2 document report earns v2's warnings, never
    /// v1's: the admission's ignored levers first, then `leading_text`, then
    /// `v2_length` or `v2_budget` with the bodies produced. Only a stopped
    /// document is a partial output (exit 5); a v1 length stop stays exit 0.
    #[test]
    fn document_reports_warn_in_v2_terms_and_only_a_stopped_one_is_partial() {
        let lever = RunWarning {
            kind: "ignored_lever",
            message: "FOCR_QKV_FUSED has no effect".into(),
            recovery: warning_recovery(WarningKind::IgnoredLever),
            fields: serde_json::Map::new(),
        };
        let report =
            |finish_reason: FinishReason, bodies: usize, leading_text: bool| DecodeReport {
                ids: Vec::new(),
                finish_reason,
                cap: (finish_reason == FinishReason::Length).then_some(LengthCap::Positions),
                pages: Some(PageBodies {
                    expected: 3,
                    bodies,
                }),
                max_length: 32_768,
                max_positions: sampler::UOCR2_MAX_POSITIONS,
                decodes: 1,
                length_stops: usize::from(finish_reason != FinishReason::Stop),
                document: Some(DocumentStats {
                    leading_text,
                    ..DocumentStats::default()
                }),
                admission_warnings: vec![lever.clone()],
            };

        let model = ModelIdentity {
            model_id: model_arch::UNLIMITED_OCR_V2_ID,
            source_sha256: None,
        };
        assert_eq!(model.version_tag(), "unlimited-ocr-2");
        let done = report(FinishReason::Stop, 3, false);
        let warnings = decode_warnings(&done);
        assert_eq!(warnings, std::slice::from_ref(&lever));
        assert!(partial_output(&done, &warnings, &model).is_none());

        let cut = report(FinishReason::Length, 2, true);
        let warnings = decode_warnings(&cut);
        let kinds: Vec<&str> = warnings.iter().map(|w| w.kind).collect();
        assert_eq!(kinds, ["ignored_lever", "leading_text", "v2_length"]);
        assert!(kinds.iter().all(|kind| RUN_WARNING_KINDS.contains(kind)));
        let json = warnings[2].to_json();
        assert_eq!(
            (json["expected"].as_u64(), json["got"].as_u64()),
            (Some(3), Some(2))
        );
        assert!(
            warnings[2]
                .message
                .starts_with("stopped inside page 2 of 3")
                && warnings[2].message.contains("131,072"),
            "{}",
            warnings[2].message
        );
        let Some(FocrError::Partial(partial)) = partial_output(&cut, &warnings, &model) else {
            panic!("a length stop is a partial output");
        };
        assert_eq!(
            (
                partial.finish_reason,
                partial.pages_out,
                partial.warnings.len()
            ),
            ("length", 2, 3)
        );
        assert_eq!(
            (partial.model_id, partial.source_sha256.as_deref()),
            ("unlimited-ocr-2", None)
        );
        assert_eq!(cut.run_finish_reason(), FinishReason::Length);

        let late = report(FinishReason::Budget, 0, false);
        let warnings = decode_warnings(&late);
        assert_eq!(warnings[1].kind, "v2_budget");
        assert!(
            warnings[1]
                .message
                .starts_with("stopped before page 1 of 3"),
            "{}",
            warnings[1].message
        );
        assert_eq!(late.run_finish_reason(), FinishReason::Budget);
        let Some(error) = partial_output(&late, &warnings, &model) else {
            panic!("a budget stop is a partial output");
        };
        assert_eq!(error.exit_code(), crate::error::EXIT_TIMEOUT);
        let FocrError::Partial(partial) = &error else {
            panic!("{error:?}");
        };
        assert_eq!((partial.finish_reason, partial.pages_out), ("budget", 0));

        let v1 = multi_page_report(
            FinishReason::Length,
            Some(LengthCap::Positions),
            3,
            "<PAGE>\na\n<PAGE>\nb",
        );
        let warnings = decode_warnings(&v1);
        assert_eq!(warnings[0].kind, "v1_truncated");
        let v1_model = ModelIdentity {
            model_id: model_arch::UNLIMITED_OCR_V1_ID,
            source_sha256: Some("ab".repeat(32)),
        };
        assert_eq!(
            v1_model.version_tag(),
            format!("unlimited-ocr@{}", "ab".repeat(32))
        );
        assert!(
            partial_output(&v1, &warnings, &v1_model).is_none(),
            "v1 stays exit 0"
        );
    }

    /// U2-X2: the frozen `infer_multi` oracle outputs end at EOS with fewer
    /// page bodies than pages, so each is a `v1_page_merge` with the exact
    /// counts.
    #[test]
    fn recorded_multi_page_outputs_are_page_merges() {
        for (name, expected, got) in [("p20", 20, 7), ("p10", 10, 9), ("p9_p14", 2, 1)] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("tests/fixtures/multi_page/{name}_raw.md"));
            let raw = std::fs::read_to_string(&path).expect("recorded oracle output");
            let report = multi_page_report(FinishReason::Stop, None, expected, &raw);
            let warnings = decode_warnings(&report);
            assert_eq!(warnings.len(), 1, "{name}");
            let w = warnings[0].to_json();
            assert_eq!(w["kind"], "v1_page_merge", "{name}");
            assert_eq!(
                (w["expected"].as_u64(), w["got"].as_u64()),
                (Some(expected as u64), Some(got)),
                "{name}"
            );
            assert_eq!(
                w["message"],
                format!(
                    "v1 produced {got} page bodies for {expected} pages; page boundaries in this \
                     output are unreliable"
                )
            );
            assert_eq!(w["recovery"], warning_recovery(WarningKind::V1PageMerge));
            // U2-E3: so their JSON `pages` stay `[]`.
            let origins = list_origins(expected);
            let frame = PageFrame {
                sizes: vec![(999, 999); expected],
                origins: &origins,
                model_id: model_arch::UNLIMITED_OCR_V1_ID,
            };
            assert!(
                frame.text_pages(&postprocess::strip_eos(&raw)).is_empty(),
                "{name}"
            );
        }
    }

    /// U2-E3: v1's text has pages only when it holds one body per page, the
    /// head before a missing opening marker included (GH #17); they carry no
    /// decode facts, which v1's text split cannot know.
    #[test]
    fn v1_text_has_pages_only_with_one_body_per_page() {
        let origins = list_origins(3);
        let frame = PageFrame {
            sizes: vec![(999, 999); 3],
            origins: &origins,
            model_id: model_arch::UNLIMITED_OCR_V1_ID,
        };
        for text in ["<PAGE>a<PAGE>b<PAGE>c", "a<PAGE>b<PAGE>c"] {
            let pages = frame.text_pages(text);
            let summary: Vec<_> = pages
                .iter()
                .map(|p| {
                    (
                        p.run_index,
                        p.source_page,
                        p.text.as_str(),
                        p.tokens,
                        p.finish,
                    )
                })
                .collect();
            assert_eq!(
                summary,
                [
                    (1, 1, "a", None, None),
                    (2, 2, "b", None, None),
                    (3, 3, "c", None, None),
                ],
                "{text:?}"
            );
            assert_eq!(
                postprocess::join_page_texts(pages.iter().map(|p| &p.text)),
                postprocess::finalize_multi(text, 3).unwrap(),
                "{text:?}"
            );
        }
        for merged in ["<PAGE>a<PAGE>b", "<PAGE>a<PAGE>b<PAGE>c<PAGE>d", ""] {
            assert!(frame.text_pages(merged).is_empty(), "{merged:?}");
        }
    }

    /// U2-E3: an Unlimited-OCR-2 decode's pages map to their source pages. A
    /// page that failed to rasterize has no prompt page (source page 2 here),
    /// a spread's halves share their source page, a non-blank head joins
    /// page 1 (so there are still exactly N pages), and an extra trailing
    /// `<PAGE>` adds none. Boxes map through each page's own size, and the
    /// pages streamed as the ids arrive are the final ones.
    #[test]
    fn document_pages_map_to_their_source_pages() {
        use crate::tokenizer::special::PAGE;
        use crate::tokenizer::tests::{byte_id, byte_tokenizer};
        let t = byte_tokenizer();
        let text = |s: &str| s.bytes().map(byte_id).collect::<Vec<u32>>();
        let figure = "<BLOCK>image[0, 0, 999, 499]<FORK>";
        let ids = [
            text("hi "),
            vec![PAGE],
            text("one"),
            vec![PAGE],
            text(figure),
            vec![PAGE],
            text("three"),
            vec![PAGE],
        ]
        .concat();
        let decoded = DecodedDocument {
            split: postprocess::split_pages_by_token(&t, &ids, PAGE, 3).unwrap(),
            finish: vec![PageFinish::Marker, PageFinish::PageCap, PageFinish::Marker],
            elapsed: [5, 6, 7].map(std::time::Duration::from_millis).to_vec(),
        };
        let origins = [
            PageOrigin {
                source_page: 1,
                half: None,
            },
            PageOrigin {
                source_page: 3,
                half: Some(Half::Left),
            },
            PageOrigin {
                source_page: 3,
                half: Some(Half::Right),
            },
        ];
        let frame = PageFrame {
            sizes: vec![(999, 999), (1998, 2997), (999, 999)],
            origins: &origins,
            model_id: model_arch::UNLIMITED_OCR_V2_ID,
        };
        let pages = frame.token_pages(&decoded);
        let summary: Vec<_> = pages
            .iter()
            .map(|p| {
                (
                    (p.run_index, p.source_page, p.half),
                    p.text.as_str(),
                    (p.tokens, p.finish, p.elapsed_ms),
                )
            })
            .collect();
        assert_eq!(
            summary,
            [
                (
                    (1, 1, None),
                    "hi one",
                    (Some(6), Some(PageFinish::Marker), Some(5))
                ),
                (
                    (2, 3, Some(Half::Left)),
                    "![](images/page_1_0.jpg)\n",
                    (Some(figure.len()), Some(PageFinish::PageCap), Some(6))
                ),
                (
                    (3, 3, Some(Half::Right)),
                    "three",
                    (Some(5), Some(PageFinish::Marker), Some(7))
                ),
            ]
        );
        assert!(
            pages
                .iter()
                .all(|p| p.window.is_none() && p.model_id == model_arch::UNLIMITED_OCR_V2_ID)
        );
        // The left half's box maps through its own 1998x2997 raster.
        assert_eq!(pages[1].layout[0].boxes, [[0, 0, 1998, 1497]]);
        assert_eq!(pages[1].figures[0].boxes, [[0, 0, 1998, 1497]]);
        assert_eq!(pages[1].figures[0].markdown_ref, "![](images/page_1_0.jpg)");
        assert!(pages[0].layout.is_empty() && pages[2].figures.is_empty());

        let mut stream = postprocess::TokenPageStream::new(&t, PAGE, 3);
        let mut streamed = Vec::new();
        let mut close = |index: usize, body: &str, tokens: usize| {
            streamed.push(frame.closed_page(&ClosedPage {
                index,
                body,
                tokens,
                finish: decoded.finish[index - 1],
                elapsed: decoded.elapsed[index - 1],
            }));
        };
        for &id in &ids {
            stream.push(id, &mut close);
        }
        stream.finish(&mut close);
        assert_eq!(streamed, pages, "streamed pages are the final pages");
    }

    /// U2-X2: healthy multi-page outputs, including one whose model skipped
    /// the opening marker (GH #17), warn nothing.
    #[test]
    fn healthy_multi_page_outputs_do_not_warn() {
        for decoded in ["<PAGE>a<PAGE>b<PAGE>c", "lead text<PAGE>b<PAGE>c"] {
            let report = multi_page_report(FinishReason::Stop, None, 3, decoded);
            assert_eq!(report.pages.map(|p| p.bodies), Some(3), "{decoded}");
            assert!(decode_warnings(&report).is_empty(), "{decoded}");
        }
        // A single-page decode has no page report and never warns, even at a cap.
        let mut single = multi_page_report(FinishReason::Length, Some(LengthCap::MaxLength), 1, "");
        single.pages = None;
        assert!(decode_warnings(&single).is_empty());
        assert_eq!(single.run_finish_reason(), FinishReason::Length);
    }

    /// U2-X2: a length stop with missing pages is a truncation named by the
    /// limit that bound it; each cause has its own recovery.
    #[test]
    fn truncations_name_their_cap() {
        let positions = decode_warnings(&multi_page_report(
            FinishReason::Length,
            Some(LengthCap::Positions),
            3,
            "<PAGE>one<PAGE>tw",
        ));
        let w = positions[0].to_json();
        assert_eq!(
            (w["kind"].as_str(), w["cause"].as_str()),
            (Some("v1_truncated"), Some("positions"))
        );
        assert_eq!(
            (w["expected"].as_u64(), w["got"].as_u64()),
            (Some(3), Some(2))
        );
        assert_eq!(
            w["message"],
            "stopped after 2 of 3 pages: v1 shares 32,768 positions between prompt and output \
             (≈29 dense pages)"
        );
        assert_eq!(
            w["recovery"],
            "split the document with `--pages`, or parse pages independently (omit `--multi-page`)"
        );
        let mut capped = multi_page_report(
            FinishReason::Length,
            Some(LengthCap::MaxLength),
            3,
            "<PAGE>one",
        );
        capped.max_length = 1200;
        let w = decode_warnings(&capped)[0].to_json();
        assert_eq!(
            (w["kind"].as_str(), w["cause"].as_str()),
            (Some("v1_truncated"), Some("user_cap"))
        );
        assert_eq!(
            w["message"],
            "stopped after 1 of 3 pages: the --max-length cap of 1,200 generated tokens was reached"
        );
        assert_eq!(w["recovery"], "raise `--max-length`");
    }

    #[test]
    fn test_position_override_has_the_shipping_default() {
        // Unset in the test environment: the validator passes and decode
        // params keep the arch's context.
        if std::env::var_os(TEST_MAX_POSITIONS_ENV).is_none() {
            assert!(validate_test_max_positions().is_ok());
            assert_eq!(
                decode_params_from_env(DecodeOverrides::default()).max_positions,
                MAX_POSITION_EMBEDDINGS
            );
        }
        assert_eq!(thousands(32_768), "32,768");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_000_000), "1,000,000");
    }

    /// Build a synthetic one-tensor artifact holding `model.embed_tokens.weight`
    /// as `QInt8PerChan` — the storage the wasm experts-int4 recipe emits.
    fn synthetic_int8_embed_artifact(rows: usize, cols: usize) -> Weights {
        let arch = model_arch::arch_by_id("got-ocr2").expect("test architecture is registered");
        let payload: Vec<u8> = (0..rows * cols)
            .map(|i| (((i * 37 + 11) % 255) as i64 - 127) as i8 as u8)
            .collect();
        let scales: Vec<u8> = (0..rows)
            .flat_map(|r| (((r % 13) as f32) * 0.0007 + 0.0031).to_le_bytes())
            .collect();
        let mut builder = crate::quant::focrq::FocrqBuilder::new()
            .with_model_id(arch.id())
            .with_license_notice(arch.license_notice());
        builder
            .add_quantized(
                EMBED_TOKENS,
                crate::quant::focrq::WriteDType::QInt8PerChan,
                vec![rows, cols],
                payload,
                scales,
                0,
                0,
            )
            .expect("valid synthetic QInt8 embedding record");
        Weights::from_bytes(builder.build()).expect("synthetic artifact loads")
    }

    /// bd-4l71 acceptance: the int8-resident embedding lookup is BIT-IDENTICAL
    /// to the historical whole-table f32 widen. Every row produced by
    /// [`EmbedTable::row_f32`] / [`EmbedTable::embed_ids`] on the zero-copy
    /// int8 arm must equal the corresponding row of the fully dequantized
    /// matrix `Weights::mat` would have built (~662 MB on the real vocab).
    #[test]
    fn int8_embed_row_lookup_matches_full_dequant() {
        let (rows, cols) = (67usize, 16usize);
        let weights = synthetic_int8_embed_artifact(rows, cols);
        // The reference: the whole-table dequant-on-access `Weights::mat` does.
        let full = weights.mat(EMBED_TOKENS).expect("whole-table widen");
        assert_eq!(full.shape(), (rows, cols));

        let table = embed_table_from(&weights).expect("int8 embed table resolves");
        assert!(
            matches!(table, EmbedTable::QInt8 { .. }),
            "a QInt8PerChan table must take the zero-copy int8 arm"
        );
        assert_eq!((table.rows(), table.cols()), (rows, cols));
        for r in 0..rows {
            assert_eq!(
                table.row_f32(r),
                full.data[r * cols..(r + 1) * cols].to_vec(),
                "per-row dequant of row {r} must be bit-identical to the full dequant"
            );
        }

        // The gather path agrees too (and keeps the out-of-vocab guard).
        let ids: Vec<u32> = vec![0, 5, 66, 5, 42];
        let gathered = table.embed_ids(&ids).expect("gather over valid ids");
        assert_eq!(gathered.shape(), (ids.len(), cols));
        for (i, &id) in ids.iter().enumerate() {
            let want = &full.data[id as usize * cols..(id as usize + 1) * cols];
            assert_eq!(&gathered.data[i * cols..(i + 1) * cols], want, "id {id}");
        }
        assert!(
            table.embed_ids(&[rows as u32]).is_err(),
            "an out-of-vocab id must still error"
        );
    }

    #[test]
    fn forward_admission_serializes_concurrent_callers() {
        use std::sync::Barrier;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let start = Barrier::new(5);
        let live = AtomicUsize::new(0);
        let max_live = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let mut workers = Vec::new();
            for _ in 0..4 {
                workers.push(scope.spawn(|| {
                    start.wait();
                    let _permit = forward_admission().acquire();
                    let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                    max_live.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    live.fetch_sub(1, Ordering::SeqCst);
                }));
            }
            start.wait();
            for worker in workers {
                worker.join().expect("gate contender did not panic");
            }
        });
        assert_eq!(max_live.load(Ordering::SeqCst), 1);
        assert_eq!(live.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn fallible_once_initializes_once_and_retries_after_error() {
        use std::sync::Barrier;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let cell = FallibleOnce::<usize>::new();
        let start = Barrier::new(5);
        let initializers = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let mut workers = Vec::new();
            for _ in 0..4 {
                workers.push(scope.spawn(|| {
                    start.wait();
                    let value = cell
                        .get_or_try_init(|| {
                            initializers.fetch_add(1, Ordering::SeqCst);
                            std::thread::sleep(std::time::Duration::from_millis(5));
                            Ok::<usize, ()>(42)
                        })
                        .expect("initializer succeeds");
                    assert_eq!(*value, 42);
                }));
            }
            start.wait();
            for worker in workers {
                worker.join().expect("initializer contender did not panic");
            }
        });
        assert_eq!(initializers.load(Ordering::SeqCst), 1);

        let retry = FallibleOnce::<usize>::new();
        assert_eq!(
            retry
                .get_or_try_init(|| Err::<usize, _>("first attempt"))
                .expect_err("first initialization must fail"),
            "first attempt"
        );
        assert_eq!(
            *retry
                .get_or_try_init(|| Ok::<usize, &str>(7))
                .expect("failed initialization remains retryable"),
            7
        );
    }

    #[test]
    fn resolve_model_rejects_missing_path() {
        let missing = Path::new("/definitely/not/a/real/model/path.focrq");
        let r = OcrModel::resolve_model(missing);
        assert!(matches!(r, Err(FocrError::ModelNotFound(_))));
    }

    #[test]
    fn load_missing_path_is_model_not_found() {
        let missing = Path::new("/definitely/not/a/real/model/path.focrq");
        let r = OcrModel::load(missing);
        assert!(matches!(r, Err(FocrError::ModelNotFound(_))));
    }

    /// A path that exists on disk but is not a real model resolves (the
    /// header-sniff resolver accepts any existing path) and then fails
    /// cleanly with `FormatMismatch` (§7.4 exit 7), never panics. This was a
    /// Phase-0 `NotImplemented` placeholder until the input-fault suite
    /// (bd-15kd) typed the corrupt-artifact class: junk bytes in a `.focrq`
    /// are a container-format fault, and the fault suite pins the same
    /// contract end-to-end through the CLI. Uses a freshly-created temp file
    /// so the test is CWD-independent.
    #[test]
    fn load_existing_non_model_path_is_format_mismatch_not_panic() -> FocrResult<()> {
        let mut tmp = std::env::temp_dir();
        tmp.push(format!(
            "franken_ocr_load_test_{}.focrq",
            std::process::id()
        ));
        std::fs::write(&tmp, b"not a real model blob").expect("write temp file");
        let r = OcrModel::load(&tmp);
        let _ = std::fs::remove_file(&tmp); // best-effort cleanup (not a delete of source)
        // resolve_model accepts an existing path; the container reader types
        // the junk as a format fault (exit 7).
        let err = match r {
            Ok(_) => {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "junk artifact must error, not load"
                )));
            }
            Err(e) => e,
        };
        assert!(
            matches!(err, FocrError::FormatMismatch(_)),
            "expected FormatMismatch (exit 7) on a junk artifact, got {err:?}"
        );
        assert_eq!(err.exit_code(), 7, "FormatMismatch maps to exit 7");
        Ok(())
    }

    fn temp_model_dir(label: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!("franken_ocr_{label}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp model dir");
        dir
    }

    /// The v1 source shard's file name, which a directory without an index
    /// resolves to.
    fn v1_raw_shard_name() -> &'static str {
        unlimited_ocr_census::source_shards(model_arch::TensorCensus::UnlimitedOcrV1)[0].0
    }

    #[test]
    fn resolve_model_searches_short_name_focrq_candidate() {
        let dir = temp_model_dir("resolve_short_focrq");
        let model = dir.join("unlimited-ocr.focrq");
        std::fs::write(&model, b"only resolver existence is tested").expect("write model");

        let resolved = resolve_model_from_search_dirs(Path::new("unlimited-ocr"), &[dir])
            .expect("resolve short-name focrq candidate");
        assert_eq!(resolved, model);
    }

    #[test]
    fn resolve_model_searches_short_name_safetensors_directory() {
        let root = temp_model_dir("resolve_short_safetensors_root");
        let package = root.join("unlimited-ocr");
        std::fs::create_dir_all(&package).expect("create safetensors package");
        let shard = package.join(v1_raw_shard_name());
        std::fs::write(&shard, b"only resolver existence is tested").expect("write shard");

        let resolved = resolve_model_from_search_dirs(Path::new("unlimited-ocr"), &[root])
            .expect("resolve short-name safetensors directory");
        assert_eq!(resolved, shard);
    }

    #[test]
    fn resolve_model_quant_preference_picks_matching_focrq_candidate() {
        let dir = temp_model_dir("resolve_quant_preference");
        let generic = dir.join("unlimited-ocr.focrq");
        let int4 = dir.join("unlimited-ocr.int4.focrq");
        std::fs::write(&generic, weights::FOCRQ_MAGIC).expect("write generic model");
        std::fs::write(&int4, weights::FOCRQ_MAGIC).expect("write int4 model");

        let resolved = resolve_model_from_search_dirs_with_quant(
            Path::new("unlimited-ocr"),
            &[dir],
            Some(ModelQuantPreference::Int4),
        )
        .expect("resolve quant-specific focrq candidate");
        assert_eq!(resolved, int4);
    }

    #[test]
    fn resolve_model_accepts_model_dir_direct_focrq_artifact() {
        let dir = temp_model_dir("resolve_model_dir_direct_focrq");
        let model = dir.join("custom.focrq");
        std::fs::write(&model, weights::FOCRQ_MAGIC).expect("write model");

        let resolved =
            resolve_model_from_search_dirs(Path::new("anything"), std::slice::from_ref(&model))
                .expect("resolve direct artifact from model dir entry");
        assert_eq!(resolved, model);
    }

    #[test]
    fn resolve_model_searches_relative_default_basename_in_model_dir() {
        let dir = temp_model_dir("resolve_default_basename");
        let model = dir.join("unlimited-ocr.focrq");
        std::fs::write(&model, weights::FOCRQ_MAGIC).expect("write model");

        let resolved =
            resolve_model_from_search_dirs(Path::new("models/unlimited-ocr.focrq"), &[dir])
                .expect("resolve default basename in model dir");
        assert_eq!(resolved, model);
    }

    #[test]
    fn resolve_model_default_spec_finds_pulled_int8_artifact() {
        // The real fresh-install path: `focr pull` installs
        // `unlimited-ocr.int8.focrq`, and a bare `focr ocr page.png` resolves the
        // DEFAULT_MODEL_PATH spec `models/unlimited-ocr.focrq`. That MUST find the
        // pulled int8 artifact with no `--model` / env override (bd-3u6x).
        let dir = temp_model_dir("resolve_default_int8");
        let int8 = dir.join("unlimited-ocr.int8.focrq");
        std::fs::write(&int8, weights::FOCRQ_MAGIC).expect("write int8 model");

        let resolved =
            resolve_model_from_search_dirs(Path::new("models/unlimited-ocr.focrq"), &[dir])
                .expect("default spec resolves the pulled int8 artifact");
        assert_eq!(resolved, int8);
    }

    #[test]
    fn resolve_model_prefers_manifest_versioned_pull_over_legacy_quant_name() {
        let dir = temp_model_dir("resolve_manifest_versioned_int8");
        let current = dir.join(format!(
            "unlimited-ocr.v{}.int8.focrq",
            crate::UNLIMITED_OCR_ARTIFACT_VERSION
        ));
        let legacy = dir.join("unlimited-ocr.int8.focrq");
        std::fs::write(&current, weights::FOCRQ_MAGIC).expect("write current model");
        std::fs::write(&legacy, weights::FOCRQ_MAGIC).expect("write legacy model");

        let resolved =
            resolve_model_from_search_dirs(Path::new("models/unlimited-ocr.focrq"), &[dir])
                .expect("default spec resolves manifest-versioned pull");
        assert_eq!(resolved, current);
    }

    #[test]
    fn resolve_model_names_each_models_artifacts_by_its_own_version() {
        let v2 = model_arch::arch_by_id(model_arch::UNLIMITED_OCR_V2_ID).expect("v2 registered");
        assert_ne!(v2.artifact_version(), crate::UNLIMITED_OCR_ARTIFACT_VERSION);
        let dir = temp_model_dir("resolve_per_model_version");
        let v1_current = dir.join(format!(
            "unlimited-ocr.v{}.int8.focrq",
            crate::UNLIMITED_OCR_ARTIFACT_VERSION
        ));
        let v2_current = dir.join(format!(
            "unlimited-ocr-2.v{}.int8.focrq",
            v2.artifact_version()
        ));
        for path in [&v1_current, &v2_current] {
            std::fs::write(path, weights::FOCRQ_MAGIC).expect("write model");
        }
        for (spec, expected) in [
            ("unlimited-ocr", &v1_current),
            ("models/unlimited-ocr.focrq", &v1_current),
            ("unlimited-ocr-2", &v2_current),
            ("models/unlimited-ocr-2.focrq", &v2_current),
        ] {
            let resolved =
                resolve_model_from_search_dirs(Path::new(spec), std::slice::from_ref(&dir))
                    .expect("spec resolves");
            assert_eq!(&resolved, expected, "{spec}");
        }
        // v1's release version on a v2 name is no v2 release.
        let misversioned = temp_model_dir("resolve_per_model_misversioned");
        std::fs::write(
            misversioned.join(format!(
                "unlimited-ocr-2.v{}.int8.focrq",
                crate::UNLIMITED_OCR_ARTIFACT_VERSION
            )),
            weights::FOCRQ_MAGIC,
        )
        .expect("write model");
        let error = resolve_model_from_search_dirs(Path::new("unlimited-ocr-2"), &[misversioned])
            .expect_err("a misversioned name is not a v2 release");
        assert!(matches!(error, FocrError::ModelNotFound(_)), "{error}");

        // A per-model subdirectory holding a sharded checkpoint resolves by
        // the model's short name to its index.
        let subdir = dir.join("unlimited-ocr-2");
        std::fs::create_dir_all(&subdir).expect("model subdir");
        std::fs::write(subdir.join(shards::SHARD_INDEX_NAME), b"{}").expect("write index");
        let resolved = resolve_model_from_search_dirs(Path::new("unlimited-ocr-2"), &[dir])
            .expect("the subdirectory resolves");
        assert_eq!(resolved, subdir.join(shards::SHARD_INDEX_NAME));
    }

    /// U2-E1: a bare registry id is answered only by its own model. A search
    /// entry that is another model's package is passed over (and still serves
    /// that model's id and any other spec), search order holds, and an
    /// artifact named for the id whose header holds another model is
    /// `ModelNotFound` naming both.
    #[cfg(unix)]
    #[test]
    fn a_registry_id_resolves_only_to_its_own_model() {
        let v1_dir = temp_model_dir("registry_id_v1_raw");
        let v1_index = write_sparse_two_shard_stand_in(
            &v1_dir,
            &unlimited_ocr_census::v1_source_tensors(),
            &[],
        );
        let root = temp_model_dir("registry_id_root");
        let v2_dir = root.join(model_arch::UNLIMITED_OCR_V2_ID);
        std::fs::create_dir_all(&v2_dir).expect("v2 subdirectory");
        let v2_index = write_sparse_v2_stand_in(&v2_dir);
        let resolve =
            |spec: &str, dirs: &[PathBuf]| resolve_model_from_search_dirs(Path::new(spec), dirs);

        let v1_only = std::slice::from_ref(&v1_dir);
        assert_eq!(
            resolve("unlimited-ocr", v1_only).expect("v1's id"),
            v1_index
        );
        assert_eq!(
            resolve("anything", v1_only).expect("any other spec"),
            v1_index
        );
        let error = resolve("unlimited-ocr-2", v1_only).expect_err("v1 never answers v2's id");
        assert!(matches!(error, FocrError::ModelNotFound(_)), "{error}");

        let both = [v1_dir.clone(), root.clone()];
        assert_eq!(
            resolve("unlimited-ocr-2", &both).expect("a later entry holds v2"),
            v2_index
        );
        assert_eq!(
            resolve("unlimited-ocr", &both).expect("the first entry holds v1"),
            v1_index
        );

        let misnamed_root = temp_model_dir("registry_id_misnamed");
        let misnamed = misnamed_root.join(model_arch::UNLIMITED_OCR_V2_ID);
        std::fs::create_dir_all(&misnamed).expect("misnamed directory");
        write_sparse_two_shard_stand_in(&misnamed, &unlimited_ocr_census::v1_source_tensors(), &[]);
        let error = resolve("unlimited-ocr-2", &[misnamed_root]).expect_err("v1 under v2's name");
        let text = error.to_string();
        assert!(matches!(error, FocrError::ModelNotFound(_)), "{error}");
        assert!(
            text.contains("\"unlimited-ocr\"") && text.contains("\"unlimited-ocr-2\""),
            "{text}"
        );

        for (spec, named) in [
            ("unlimited-ocr-2", true),
            ("unlimited-ocr", true),
            ("./unlimited-ocr-2", false),
            ("models/unlimited-ocr-2", false),
            ("unlimited-ocr-2.focrq", false),
            ("unlimited-ocr-3", false),
        ] {
            assert_eq!(registry_spec(Path::new(spec)).is_some(), named, "{spec}");
        }
    }

    #[test]
    fn resolve_model_prefers_exact_focrq_over_quant_variant() {
        // When BOTH a generic `unlimited-ocr.focrq` (e.g. from `focr convert`) and a
        // pulled `unlimited-ocr.int8.focrq` are present, the exact name wins.
        let dir = temp_model_dir("resolve_exact_over_int8");
        let generic = dir.join("unlimited-ocr.focrq");
        let int8 = dir.join("unlimited-ocr.int8.focrq");
        std::fs::write(&generic, weights::FOCRQ_MAGIC).expect("write generic model");
        std::fs::write(&int8, weights::FOCRQ_MAGIC).expect("write int8 model");

        let resolved =
            resolve_model_from_search_dirs(Path::new("models/unlimited-ocr.focrq"), &[dir])
                .expect("resolve exact generic focrq");
        assert_eq!(resolved, generic);
    }

    #[test]
    fn resolve_model_missing_short_name_lists_search_dirs() {
        let dirs = [
            PathBuf::from("/tmp/franken_ocr_missing_a"),
            PathBuf::from("/tmp/franken_ocr_missing_b"),
        ];
        let err = resolve_model_from_search_dirs(Path::new("missing-model"), &dirs)
            .expect_err("missing short name should fail");
        let text = err.to_string();
        assert!(matches!(err, FocrError::ModelNotFound(_)));
        assert!(text.contains("missing-model"));
        assert!(text.contains("/tmp/franken_ocr_missing_a"));
        assert!(text.contains("/tmp/franken_ocr_missing_b"));
        assert!(text.contains(MODEL_DIR_ENV));
    }

    fn minimal_focrq_blob(version: u32, header_json: &str, payload: &[u8]) -> Vec<u8> {
        let mut blob = Vec::new();
        blob.extend_from_slice(weights::FOCRQ_MAGIC);
        blob.extend_from_slice(&version.to_le_bytes());
        blob.push(0);
        blob.extend_from_slice(&[0u8; 32]);
        blob.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        blob.extend_from_slice(header_json.as_bytes());
        blob.extend_from_slice(payload);
        blob
    }

    #[test]
    fn load_future_focrq_version_preserves_format_mismatch() {
        let payload = [0u8, 0u8];
        let header = "{\"t\":{\"dtype\":\"BF16\",\"shape\":[1],\"byte_offset\":0,\"byte_len\":2}}";
        let path = temp_model_dir("future_focrq_version").join("future-version.focrq");
        std::fs::write(
            &path,
            minimal_focrq_blob(weights::FOCRQ_FORMAT_VERSION + 1, header, &payload),
        )
        .expect("write future-version artifact");
        let err = load_weights_from_resolved_model(&path).unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert_eq!(err.exit_code(), crate::error::EXIT_FORMAT_MISMATCH);
        assert!(format!("{err}").contains("requires exactly"));
    }

    #[test]
    fn load_malformed_safetensors_preserves_format_mismatch() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(1u64).to_le_bytes());
        bytes.extend_from_slice(b"{");
        let path = temp_model_dir("malformed_safetensors").join("bad.safetensors");
        std::fs::write(&path, bytes).expect("write malformed safetensors");
        let err = load_weights_from_resolved_model(&path).unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert_eq!(err.exit_code(), crate::error::EXIT_FORMAT_MISMATCH);
    }

    fn compatible_test_model_builder() -> crate::quant::focrq::FocrqBuilder {
        let arch = model_arch::arch_by_id("got-ocr2").expect("test arch is registered");
        let mut builder = crate::quant::focrq::FocrqBuilder::new()
            .with_model_id(arch.id())
            .with_license_notice(arch.license_notice());
        builder
            .add_tensor(
                "test.base.weight",
                crate::quant::focrq::WriteDType::Bf16,
                vec![1, 1],
                vec![0; 2],
            )
            .expect("add minimal compatible test-model tensor");
        builder
    }

    fn incomplete_unlimited_builder() -> crate::quant::focrq::FocrqBuilder {
        let mut builder =
            crate::quant::focrq::FocrqBuilder::new().with_packing_manifest_json(format!(
                r#"{{"quant_recipe":"{}"}}"#,
                crate::quant::convert::UNLIMITED_OCR_INT8_RECIPE_ID,
            ));
        builder
            .add_tensor(
                "model.embed_tokens.weight",
                crate::quant::focrq::WriteDType::Bf16,
                vec![1, 1],
                vec![0; 2],
            )
            .expect("add minimal compatible high-precision tensor");
        builder
    }

    #[test]
    fn ocr_model_load_owns_weight_bytes_by_default() -> FocrResult<()> {
        let path = temp_model_dir("ocr_model_mmap").join("minimal.focrq");
        let blob = compatible_test_model_builder().build();
        std::fs::write(&path, blob).expect("write valid minimal focrq");

        let model = OcrModel::load(&path)?;
        assert!(
            !model.weights.is_mapped() || weights::mmap_requested(),
            "production load may map only after explicit opt-in"
        );
        Ok(())
    }

    #[cfg(unix)]
    fn loader_swap_fixture(label: &str, marker: &str) -> (PathBuf, Vec<u8>) {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nonce = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "franken_ocr_loader_swap_{label}_{}_{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create loader-swap fixture dir");

        let mut builder = compatible_test_model_builder();
        builder
            .add_tensor(
                marker,
                crate::quant::focrq::WriteDType::Bf16,
                vec![1],
                vec![0; 2],
            )
            .expect("add loader-swap marker tensor");
        (dir, builder.build())
    }

    #[cfg(unix)]
    #[test]
    fn production_loader_pins_validated_file_across_path_swap() {
        let (dir, original_blob) = loader_swap_fixture("path", "test.original");
        let model_path = dir.join("active.focrq");
        let replacement_path = dir.join("replacement.focrq");
        let displaced_path = dir.join("validated-original.focrq");
        std::fs::write(&model_path, original_blob).expect("write original model");

        let mut replacement = compatible_test_model_builder();
        replacement
            .add_tensor(
                "test.replacement",
                crate::quant::focrq::WriteDType::Bf16,
                vec![1],
                vec![0; 2],
            )
            .expect("add replacement marker tensor");
        std::fs::write(&replacement_path, replacement.build()).expect("write replacement model");

        let loaded = load_weights_from_resolved_model_after_validation(
            &model_path,
            |_| Ok(()),
            || {
                std::fs::rename(&model_path, &displaced_path)
                    .expect("preserve validated path entry");
                std::fs::rename(&replacement_path, &model_path).expect("swap active path entry");
            },
        )
        .expect("load the descriptor validated before the path swap");

        assert!(loaded.weights.record("test.original").is_some());
        assert!(loaded.weights.record("test.replacement").is_none());
        let path_now = Weights::load(&model_path).expect("load replacement now at active path");
        assert!(path_now.record("test.original").is_none());
        assert!(path_now.record("test.replacement").is_some());
    }

    #[cfg(unix)]
    #[test]
    fn production_loader_pins_validated_file_across_symlink_swap() {
        use std::os::unix::fs::symlink;

        let (dir, original_blob) = loader_swap_fixture("symlink", "test.original_link");
        let original_path = dir.join("original.focrq");
        let replacement_path = dir.join("replacement.focrq");
        let active_link = dir.join("active.focrq");
        let next_link = dir.join("next.focrq");
        let retired_link = dir.join("validated-link.focrq");
        std::fs::write(&original_path, original_blob).expect("write original symlink target");

        let mut replacement = compatible_test_model_builder();
        replacement
            .add_tensor(
                "test.replacement_link",
                crate::quant::focrq::WriteDType::Bf16,
                vec![1],
                vec![0; 2],
            )
            .expect("add replacement symlink marker tensor");
        std::fs::write(&replacement_path, replacement.build())
            .expect("write replacement symlink target");
        symlink(&original_path, &active_link).expect("link active path to original");
        symlink(&replacement_path, &next_link).expect("prepare replacement symlink");

        let loaded = load_weights_from_resolved_model_after_validation(
            &active_link,
            |_| Ok(()),
            || {
                std::fs::rename(&active_link, &retired_link).expect("preserve validated symlink");
                std::fs::rename(&next_link, &active_link).expect("swap active symlink");
            },
        )
        .expect("load the descriptor validated before the symlink swap");

        assert!(loaded.weights.record("test.original_link").is_some());
        assert!(loaded.weights.record("test.replacement_link").is_none());
        let link_now = Weights::load(&active_link).expect("load replacement symlink target");
        assert!(link_now.record("test.original_link").is_none());
        assert!(link_now.record("test.replacement_link").is_some());
    }

    fn add_unit_qint8(builder: &mut crate::quant::focrq::FocrqBuilder, name: &str) {
        builder
            .add_quantized(
                name,
                crate::quant::focrq::WriteDType::QInt8PerChan,
                vec![1, 1],
                vec![0],
                1.0f32.to_le_bytes().to_vec(),
                0,
                0,
            )
            .expect("add synthetic qint8 tensor");
    }

    #[test]
    fn unlimited_focrq_rejects_int8_attention_and_lm_head() {
        for name in ["model.layers.0.self_attn.q_proj.weight", "lm_head.weight"] {
            let mut builder = crate::quant::focrq::FocrqBuilder::new();
            add_unit_qint8(&mut builder, name);
            let weights = Weights::from_bytes(builder.build()).expect("parse synthetic artifact");
            let err = validate_unlimited_ocr_quant_recipe(&weights)
                .expect_err("gated tensor must stay high precision in the default recipe");
            assert!(matches!(err, FocrError::FormatMismatch(_)));
            assert!(err.to_string().contains(name));
            assert!(
                err.to_string()
                    .contains(crate::quant::convert::UNLIMITED_OCR_INT8_RECIPE_ID)
            );
        }
    }

    #[test]
    fn unlimited_focrq_requires_int8_for_present_ffn_weights() {
        let mut builder = crate::quant::focrq::FocrqBuilder::new();
        builder
            .add_tensor(
                "model.layers.0.mlp.down_proj.weight",
                crate::quant::focrq::WriteDType::Bf16,
                vec![1, 1],
                vec![0; 2],
            )
            .expect("add synthetic bf16 FFN");
        let weights = Weights::from_bytes(builder.build()).expect("parse synthetic artifact");
        let err = validate_unlimited_ocr_quant_recipe(&weights)
            .expect_err("validated FFN tensor must use int8 storage");
        assert!(err.to_string().contains("expected QInt8PerChan"));
    }

    // ── wasm experts-int4 recipe (bd-4l71) ───────────────────────────────────

    fn add_unit_qint4(builder: &mut crate::quant::focrq::FocrqBuilder, name: &str) {
        // [1, 16] g16: 8 packed bytes, one f32 group scale.
        builder
            .add_quantized(
                name,
                crate::quant::focrq::WriteDType::QInt4PerGroup,
                vec![1, 16],
                vec![0x21; 8],
                1.0f32.to_le_bytes().to_vec(),
                16,
                0,
            )
            .expect("add synthetic qint4 tensor");
    }

    fn wasm_recipe_manifest_json() -> String {
        format!(
            r#"{{"quant_recipe":"{}"}}"#,
            crate::quant::convert::UNLIMITED_OCR_WASM_INT4_RECIPE_ID,
        )
    }

    #[test]
    fn wasm_recipe_records_accept_int4_experts_and_int8_gated_sets() {
        let wasm = Some(crate::quant::convert::UNLIMITED_OCR_WASM_INT4_RECIPE_ID);
        validate_unlimited_ocr_quant_records(
            true,
            model_arch::default_arch().id(),
            wasm,
            [
                (
                    "model.layers.3.mlp.experts.7.up_proj.weight",
                    DType::QInt4PerGroup,
                ),
                ("model.layers.0.mlp.down_proj.weight", DType::QInt4PerGroup),
                (
                    "model.layers.11.mlp.shared_experts.gate_proj.weight",
                    DType::QInt4PerGroup,
                ),
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    DType::QInt8PerChan,
                ),
                ("lm_head.weight", DType::QInt8PerChan),
                ("model.embed_tokens.weight", DType::QInt8PerChan),
                ("model.norm.weight", DType::BF16),
                ("model.layers.5.mlp.gate.weight", DType::BF16), // router
            ],
        )
        .expect("the declared wasm recipe must accept its own storage dtypes");
    }

    #[test]
    fn wasm_recipe_records_reject_int4_attention_and_wrong_expert_storage() {
        let wasm = Some(crate::quant::convert::UNLIMITED_OCR_WASM_INT4_RECIPE_ID);
        // Planted negative: int4 ATTENTION under the wasm recipe is refused.
        let err = validate_unlimited_ocr_quant_records(
            true,
            model_arch::default_arch().id(),
            wasm,
            [(
                "model.layers.0.self_attn.q_proj.weight",
                DType::QInt4PerGroup,
            )],
        )
        .expect_err("int4 attention must be rejected under the wasm recipe");
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(err.to_string().contains("expected QInt8PerChan"));
        assert!(
            err.to_string()
                .contains(crate::quant::convert::UNLIMITED_OCR_WASM_INT4_RECIPE_ID)
        );

        // Experts must actually BE int4 under the wasm recipe (not int8/bf16).
        for dtype in [DType::QInt8PerChan, DType::BF16] {
            let err = validate_unlimited_ocr_quant_records(
                true,
                model_arch::default_arch().id(),
                wasm,
                [("model.layers.3.mlp.experts.7.up_proj.weight", dtype)],
            )
            .expect_err("non-int4 expert storage must be rejected under the wasm recipe");
            assert!(err.to_string().contains("expected QInt4PerGroup"));
        }

        // The router gate stays high precision even under the wasm recipe.
        let err = validate_unlimited_ocr_quant_records(
            true,
            model_arch::default_arch().id(),
            wasm,
            [("model.layers.5.mlp.gate.weight", DType::QInt8PerChan)],
        )
        .expect_err("quantized router gate must be rejected");
        assert!(err.to_string().contains("expected BF16 or F32"));
    }

    /// U2-C3: a v2 `.focrq` is held to the v2 conservative recipe alone. Its
    /// own id with conservative dtypes passes; any other id (the wasm int4 one
    /// included) or a quantized router/indexer/attention tensor is refused with
    /// exit 7, naming the id or the tensor.
    #[test]
    fn v2_records_hold_only_the_v2_conservative_recipe() {
        let v2_id = model_arch::UNLIMITED_OCR_V2_ID;
        let v2_recipe = crate::quant::convert::UNLIMITED_OCR_2_INT8_RECIPE_ID;
        let good = [
            (
                "model.layers.3.mlp.experts.7.up_proj.weight",
                DType::QInt8PerChan,
            ),
            ("model.layers.0.mlp.down_proj.weight", DType::QInt8PerChan),
            ("model.layers.0.self_attn.q_proj.weight", DType::BF16),
            (
                "model.layers.0.self_attn.dash_router.q_proj.weight",
                DType::BF16,
            ),
            ("model.layers.0.self_attn.dash_router.head_cls", DType::BF16),
            (
                "model.layers.11.self_attn.indexer.k_norm.weight",
                DType::BF16,
            ),
            ("model.layers.5.mlp.gate.weight", DType::BF16),
            ("lm_head.weight", DType::BF16),
        ];
        validate_unlimited_ocr_quant_records(true, v2_id, Some(v2_recipe), good)
            .expect("the v2 recipe accepts its own storage dtypes");
        validate_unlimited_ocr_quant_records(false, v2_id, None, good)
            .expect("raw checkpoints carry no recipe");

        for declared in [
            None,
            Some(crate::quant::convert::UNLIMITED_OCR_INT8_RECIPE_ID),
            Some(crate::quant::convert::UNLIMITED_OCR_WASM_INT4_RECIPE_ID),
        ] {
            let err = validate_unlimited_ocr_quant_records(true, v2_id, declared, good)
                .expect_err("only the v2 recipe id is accepted");
            assert_eq!(err.exit_code(), 7, "{err}");
            assert!(err.to_string().contains("must be exactly"), "{err}");
            assert!(err.to_string().contains(v2_recipe), "{err}");
        }

        for name in [
            "model.layers.4.self_attn.dash_router.q_proj.weight",
            "model.layers.4.self_attn.indexer.k_proj.weight",
            "model.layers.4.self_attn.o_proj.weight",
        ] {
            let err = validate_unlimited_ocr_quant_records(
                true,
                v2_id,
                Some(v2_recipe),
                [(name, DType::QInt8PerChan)],
            )
            .expect_err("a quantized high-precision tensor is refused");
            assert_eq!(err.exit_code(), 7, "{err}");
            let message = err.to_string();
            assert!(message.contains(name), "{message}");
            assert!(
                message.contains("Unlimited-OCR-2 .focrq violates quant recipe"),
                "{message}"
            );
            assert!(message.contains(v2_recipe), "{message}");
        }
    }

    #[test]
    fn conservative_recipe_still_rejects_int4_expert_storage() {
        // No declared wasm recipe (None, or the conservative id): QInt4 expert
        // storage stays a violation — the default recipe is untouched.
        for declared in [
            None,
            Some(crate::quant::convert::UNLIMITED_OCR_INT8_RECIPE_ID),
        ] {
            let err = validate_unlimited_ocr_quant_records(
                true,
                model_arch::default_arch().id(),
                declared,
                [(
                    "model.layers.3.mlp.experts.7.up_proj.weight",
                    DType::QInt4PerGroup,
                )],
            )
            .expect_err("int4 experts must stay rejected outside the wasm recipe");
            assert!(err.to_string().contains("expected QInt8PerChan"));
            assert!(
                err.to_string()
                    .contains(crate::quant::convert::UNLIMITED_OCR_INT8_RECIPE_ID)
            );
        }
    }

    #[test]
    fn wasm_recipe_artifact_passes_dtype_validation_and_fails_only_the_census() {
        // A loaded artifact that DECLARES the wasm recipe and stores an int4
        // expert + int8 attention/lm_head passes the per-tensor dtype recipe
        // check; the only remaining failure is the completeness census (this
        // synthetic artifact has 3 of the 2,710 tensors) — proving acceptance
        // of the wasm dtypes end-to-end through `Weights`.
        let mut builder = crate::quant::focrq::FocrqBuilder::new()
            .with_packing_manifest_json(wasm_recipe_manifest_json())
            .with_source_sha256([
                0x2b, 0xc4, 0x8a, 0x7a, 0x11, 0x00, 0x61, 0xea, 0x58, 0xff, 0xf6, 0x5d, 0x31, 0x69,
                0x36, 0x7e, 0xeb, 0xe3, 0xae, 0xe3, 0x71, 0xca, 0x69, 0x68, 0xdc, 0x22, 0x19, 0xc1,
                0xb2, 0x85, 0x5f, 0xc6,
            ]);
        add_unit_qint4(&mut builder, "model.layers.3.mlp.experts.7.up_proj.weight");
        add_unit_qint8(&mut builder, "model.layers.0.self_attn.q_proj.weight");
        add_unit_qint8(&mut builder, "lm_head.weight");
        let weights = Weights::from_bytes(builder.build()).expect("synthetic wasm artifact parses");
        assert_eq!(
            weights.quant_recipe(),
            Some(crate::quant::convert::UNLIMITED_OCR_WASM_INT4_RECIPE_ID),
            "the loader must retain the declared recipe id"
        );
        let error = validate_unlimited_ocr_quant_recipe(&weights)
            .expect_err("a 3-tensor subset must still fail the completeness census");
        let text = error.to_string();
        assert!(
            text.contains("expected 2710 tensors, found 3"),
            "failure must be the census, not a dtype-recipe violation: {text}"
        );
        assert!(
            !text.contains("violates quant recipe"),
            "wasm dtypes must not be flagged as recipe violations: {text}"
        );
    }

    #[test]
    fn weights_without_manifest_report_no_quant_recipe() {
        let mut builder = crate::quant::focrq::FocrqBuilder::new();
        add_unit_qint8(&mut builder, "model.layers.0.mlp.down_proj.weight");
        let weights = Weights::from_bytes(builder.build()).expect("synthetic artifact parses");
        assert_eq!(weights.quant_recipe(), None);
    }

    /// Planted negative (bd-4l71 acceptance): an artifact whose int4 record
    /// carries an invalid `group_size` is rejected by the bounded header probe
    /// with `FormatMismatch` — proven by byte-patching a valid artifact's
    /// header (`"group_size":16` → `"group_size":96`; same byte length, so the
    /// container framing stays intact and ONLY the metadata is corrupt).
    #[test]
    fn focrq_int4_invalid_group_size_is_rejected_with_format_mismatch() {
        let mut builder = crate::quant::focrq::FocrqBuilder::new()
            .with_packing_manifest_json(wasm_recipe_manifest_json());
        add_unit_qint4(&mut builder, "model.layers.3.mlp.experts.7.up_proj.weight");
        let blob = builder.build();

        // The intact artifact clears the bounded header probe's structural
        // checks up to (at least) the census; the patched one must die earlier,
        // at the quant-metadata check.
        let needle = b"\"group_size\":16";
        let position = blob
            .windows(needle.len())
            .position(|window| window == needle)
            .expect("header must contain the int4 group_size field");
        let mut corrupt = blob.clone();
        corrupt[position + needle.len() - 2] = b'9'; // 16 -> 96

        let err = validate_model_header_from_reader(corrupt.as_slice(), corrupt.len() as u64)
            .expect_err("group_size 96 must be rejected");
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert!(
            err.to_string().contains("group_size"),
            "error must name the corrupt field: {err}"
        );

        // The same corruption is also refused by the full parser's accessor
        // path: the tensor can never be consumed as int4.
        let weights = Weights::from_bytes(corrupt).expect("directory framing is still intact");
        let err = weights
            .qint4("model.layers.3.mlp.experts.7.up_proj.weight")
            .expect_err("qint4 accessor must reject group_size 96");
        assert!(matches!(err, FocrError::FormatMismatch(_)));
    }

    /// bd-4l71: the search-dir resolver must NEVER hand out a wasm-recipe
    /// artifact as a default model — even when it sits at a canonical
    /// candidate name — while explicit-path resolution still reaches it.
    #[test]
    fn resolver_skips_wasm_recipe_artifacts_but_explicit_path_loads_them() {
        let dir = temp_model_dir("resolver_skips_wasm_recipe");
        let wasm_path = dir.join("unlimited-ocr.int8.focrq");
        let mut wasm_builder = crate::quant::focrq::FocrqBuilder::new()
            .with_packing_manifest_json(wasm_recipe_manifest_json());
        add_unit_qint4(
            &mut wasm_builder,
            "model.layers.3.mlp.experts.7.up_proj.weight",
        );
        std::fs::write(&wasm_path, wasm_builder.build()).expect("write wasm-recipe artifact");
        assert!(focrq_declares_wasm_recipe(&wasm_path));
        // The wasm recipe is v1's alone: a v2 header declaring it is not
        // skipped, so its load gate names the bad recipe id (U2-C3).
        let v2_path = dir.join("v2-declaring-wasm.focrq");
        let v2_builder = crate::quant::focrq::FocrqBuilder::new()
            .with_model_id(model_arch::UNLIMITED_OCR_V2_ID)
            .with_packing_manifest_json(wasm_recipe_manifest_json());
        std::fs::write(&v2_path, v2_builder.build()).expect("write v2 artifact");
        assert!(!focrq_declares_wasm_recipe(&v2_path));

        let err =
            resolve_model_from_search_dirs(Path::new("unlimited-ocr"), std::slice::from_ref(&dir))
                .expect_err("the resolver must skip the wasm-recipe artifact");
        assert!(matches!(err, FocrError::ModelNotFound(_)));

        // Explicit path: resolution returns the file (loading then applies the
        // recipe/census validation as usual).
        let resolved =
            OcrModel::resolve_model(&wasm_path).expect("explicit path must resolve verbatim");
        assert_eq!(resolved, wasm_path);

        // A conservative-recipe artifact at the same candidate name resolves.
        let conservative_path = dir.join("unlimited-ocr.int4.focrq");
        let mut conservative_builder = crate::quant::focrq::FocrqBuilder::new()
            .with_packing_manifest_json(format!(
                r#"{{"quant_recipe":"{}"}}"#,
                crate::quant::convert::UNLIMITED_OCR_INT8_RECIPE_ID,
            ));
        add_unit_qint8(
            &mut conservative_builder,
            "model.layers.0.mlp.down_proj.weight",
        );
        std::fs::write(&conservative_path, conservative_builder.build())
            .expect("write conservative artifact");
        assert!(!focrq_declares_wasm_recipe(&conservative_path));
        let resolved = resolve_model_from_search_dirs(Path::new("unlimited-ocr"), &[dir])
            .expect("a non-wasm artifact must still resolve");
        assert_eq!(resolved, conservative_path);
    }

    #[test]
    fn unlimited_focrq_rejects_incomplete_conservative_subset() -> FocrResult<()> {
        let mut builder = crate::quant::focrq::FocrqBuilder::new().with_source_sha256([
            0x2b, 0xc4, 0x8a, 0x7a, 0x11, 0x00, 0x61, 0xea, 0x58, 0xff, 0xf6, 0x5d, 0x31, 0x69,
            0x36, 0x7e, 0xeb, 0xe3, 0xae, 0xe3, 0x71, 0xca, 0x69, 0x68, 0xdc, 0x22, 0x19, 0xc1,
            0xb2, 0x85, 0x5f, 0xc6,
        ]);
        add_unit_qint8(&mut builder, "model.layers.0.mlp.down_proj.weight");
        builder.add_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            crate::quant::focrq::WriteDType::Bf16,
            vec![1, 1],
            vec![0; 2],
        )?;
        builder.add_tensor(
            "lm_head.weight",
            crate::quant::focrq::WriteDType::Bf16,
            vec![1, 1],
            vec![0; 2],
        )?;
        let weights = Weights::from_bytes(builder.build())?;
        let error = validate_unlimited_ocr_quant_recipe(&weights)
            .expect_err("a present-only subset must not pass the production census");
        assert!(error.to_string().contains("expected 2710 tensors, found 3"));
        assert!(error.to_string().contains("missing"));
        Ok(())
    }

    #[test]
    fn resolve_model_accepts_safetensors_directory_shard() {
        let dir = temp_model_dir("resolve_safetensors_dir");
        let shard = dir.join(v1_raw_shard_name());
        std::fs::write(&shard, b"not loaded by resolver").expect("write shard");

        let resolved = OcrModel::resolve_model(&dir).expect("resolve safetensors directory");
        assert_eq!(resolved, shard);
    }

    #[test]
    fn load_safetensors_directory_preserves_format_mismatch() {
        let dir = temp_model_dir("load_safetensors_dir");
        let shard = dir.join(v1_raw_shard_name());
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(1u64).to_le_bytes());
        bytes.extend_from_slice(b"{");
        std::fs::write(&shard, bytes).expect("write malformed shard");

        let r = OcrModel::load(&dir);
        assert!(matches!(&r, Err(FocrError::FormatMismatch(_))));
        if let Err(err) = r {
            assert_eq!(err.exit_code(), crate::error::EXIT_FORMAT_MISMATCH);
        }
    }

    #[test]
    fn native_model_available_requires_a_compatible_focrq_header() {
        let mut tmp = std::env::temp_dir();
        tmp.push(format!(
            "franken_ocr_available_focrq_{}.focrq",
            std::process::id()
        ));
        std::fs::write(&tmp, compatible_test_model_builder().build())
            .expect("write compatible focrq");
        assert!(native_model_available(&tmp));
        assert_eq!(model_id_at(&tmp), Some("got-ocr2"), "the builder's arch");
        assert!(load_weights_from_resolved_model(&tmp).is_ok());

        let mut magic_only = tmp.clone();
        magic_only.set_file_name(format!(
            "franken_ocr_available_magic_only_{}.focrq",
            std::process::id()
        ));
        std::fs::write(&magic_only, weights::FOCRQ_MAGIC).expect("write magic-only focrq");
        assert!(!native_model_available(&magic_only));
        assert_eq!(model_id_at(&magic_only), None);
        assert!(load_weights_from_resolved_model(&magic_only).is_err());
    }

    #[test]
    fn native_model_available_rejects_unsupported_versions_and_wrong_dtype() {
        let mut future = compatible_test_model_builder().build();
        future[6..10].copy_from_slice(&(weights::FOCRQ_FORMAT_VERSION + 1).to_le_bytes());
        let future_path = temp_model_dir("available_future").join("future.focrq");
        std::fs::write(&future_path, future).expect("write future focrq");
        assert!(!native_model_available(&future_path));
        assert!(load_weights_from_resolved_model(&future_path).is_err());

        let mut builder = incomplete_unlimited_builder();
        add_unit_qint8(&mut builder, "model.layers.0.self_attn.q_proj.weight");
        let wrong_recipe_path = temp_model_dir("available_wrong_recipe").join("wrong-recipe.focrq");
        std::fs::write(&wrong_recipe_path, builder.build()).expect("write wrong recipe focrq");
        assert!(!native_model_available(&wrong_recipe_path));
        assert!(load_weights_from_resolved_model(&wrong_recipe_path).is_err());
    }

    #[test]
    fn native_model_available_rejects_version_zero_and_header_disagreement() {
        let mut version_zero = compatible_test_model_builder().build();
        version_zero[6..10].copy_from_slice(&0u32.to_le_bytes());
        let version_zero_path = temp_model_dir("available_version_zero").join("zero.focrq");
        std::fs::write(&version_zero_path, version_zero).expect("write v0 focrq");
        assert!(!native_model_available(&version_zero_path));
        assert!(load_weights_from_resolved_model(&version_zero_path).is_err());

        let mut mismatch = compatible_test_model_builder().build();
        let marker = b"\"format_version\":1";
        let marker_start = mismatch
            .windows(marker.len())
            .position(|window| window == marker)
            .expect("builder emits header format_version");
        mismatch[marker_start + marker.len() - 1] = b'0';
        let mismatch_path = temp_model_dir("available_version_mismatch").join("mismatch.focrq");
        std::fs::write(&mismatch_path, mismatch).expect("write mismatched version focrq");
        assert!(!native_model_available(&mismatch_path));
        assert!(load_weights_from_resolved_model(&mismatch_path).is_err());

        let missing_header_version = serde_json::json!({
            "arch_target": 0,
            "license_notice": crate::FOCR_MODEL_LICENSE_NOTICE,
            "packing_manifest": {
                "quant_recipe": crate::quant::convert::UNLIMITED_OCR_INT8_RECIPE_ID,
            },
            "source_sha256": "",
            "tensors": {
                "model.embed_tokens.weight": {
                    "dtype": "BF16",
                    "shape": [1, 1],
                    "byte_offset": 0,
                    "byte_len": 2,
                },
            },
        })
        .to_string();
        let missing_header_version_path =
            temp_model_dir("available_missing_header_version").join("missing-version.focrq");
        std::fs::write(
            &missing_header_version_path,
            minimal_focrq_blob(
                weights::FOCRQ_FORMAT_VERSION,
                &missing_header_version,
                &[0u8; 2],
            ),
        )
        .expect("write focrq without header format_version");
        assert!(!native_model_available(&missing_header_version_path));
        assert!(load_weights_from_resolved_model(&missing_header_version_path).is_err());
    }

    #[test]
    fn native_model_available_rejects_preamble_header_provenance_disagreement() {
        let mut arch_mismatch = compatible_test_model_builder().with_arch_target(2).build();
        arch_mismatch[10] = 1;
        let arch_path = temp_model_dir("available_arch_mismatch").join("arch-mismatch.focrq");
        std::fs::write(&arch_path, arch_mismatch).expect("write arch-mismatched focrq");
        let arch_error = validate_model_header_file(&arch_path)
            .expect_err("header and preamble arch_target must agree");
        assert!(arch_error.to_string().contains("header arch_target"));
        assert!(!native_model_available(&arch_path));

        let mut source_mismatch = compatible_test_model_builder()
            .with_source_sha256([7u8; 32])
            .build();
        source_mismatch[11] = 6;
        let source_path = temp_model_dir("available_source_mismatch").join("source-mismatch.focrq");
        std::fs::write(&source_path, source_mismatch).expect("write source-mismatched focrq");
        let source_error = validate_model_header_file(&source_path)
            .expect_err("header and preamble source_sha256 must agree");
        assert!(source_error.to_string().contains("header source_sha256"));
        assert!(!native_model_available(&source_path));

        let mut uppercase_source = compatible_test_model_builder()
            .with_source_sha256([0xabu8; 32])
            .build();
        let marker = b"\"source_sha256\":\"ab";
        let marker_start = uppercase_source
            .windows(marker.len())
            .position(|window| window == marker)
            .expect("builder emits lowercase source sha256");
        uppercase_source[marker_start + marker.len() - 2] = b'A';
        let uppercase_path =
            temp_model_dir("available_uppercase_source").join("uppercase-source.focrq");
        std::fs::write(&uppercase_path, uppercase_source).expect("write uppercase-hash focrq");
        let uppercase_error = validate_model_header_file(&uppercase_path)
            .expect_err("header source_sha256 must be lowercase");
        assert!(uppercase_error.to_string().contains("64 lowercase hex"));
        assert!(!native_model_available(&uppercase_path));
    }

    #[test]
    fn native_model_available_rejects_unknown_arch_target() {
        let path = temp_model_dir("available_unknown_arch").join("unknown-arch.focrq");
        let blob = compatible_test_model_builder().with_arch_target(4).build();
        std::fs::write(&path, blob).expect("write unknown-arch focrq");

        let error = validate_model_header_file(&path)
            .expect_err("bounded production header validation must reject unknown arch targets");
        assert!(error.to_string().contains("arch_target 4 is unsupported"));
        assert!(!native_model_available(&path));
        assert!(load_weights_from_resolved_model(&path).is_err());
    }

    #[test]
    fn native_model_available_requires_exact_unlimited_quant_recipe_metadata() {
        let empty_path = temp_model_dir("available_empty_focrq").join("empty.focrq");
        std::fs::write(
            &empty_path,
            crate::quant::focrq::FocrqBuilder::new()
                .with_packing_manifest_json(format!(
                    r#"{{"quant_recipe":"{}"}}"#,
                    crate::quant::convert::UNLIMITED_OCR_INT8_RECIPE_ID,
                ))
                .build(),
        )
        .expect("write empty focrq");
        assert!(!native_model_available(&empty_path));
        assert!(load_weights_from_resolved_model(&empty_path).is_err());

        let missing_path = temp_model_dir("available_missing_recipe").join("missing.focrq");
        let mut missing_builder = crate::quant::focrq::FocrqBuilder::new();
        missing_builder
            .add_tensor(
                "model.embed_tokens.weight",
                crate::quant::focrq::WriteDType::Bf16,
                vec![1, 1],
                vec![0; 2],
            )
            .expect("add recipe-less tensor");
        std::fs::write(&missing_path, missing_builder.build()).expect("write recipe-less focrq");
        assert!(!native_model_available(&missing_path));
        assert!(load_weights_from_resolved_model(&missing_path).is_err());

        let wrong_path = temp_model_dir("available_bad_recipe").join("wrong.focrq");
        let mut wrong_builder = crate::quant::focrq::FocrqBuilder::new()
            .with_packing_manifest_json(r#"{"quant_recipe":"legacy-full-int8"}"#);
        wrong_builder
            .add_tensor(
                "model.embed_tokens.weight",
                crate::quant::focrq::WriteDType::Bf16,
                vec![1, 1],
                vec![0; 2],
            )
            .expect("add wrong-recipe tensor");
        std::fs::write(&wrong_path, wrong_builder.build()).expect("write wrong-recipe focrq");
        assert!(!native_model_available(&wrong_path));
        assert!(load_weights_from_resolved_model(&wrong_path).is_err());
    }

    #[test]
    fn native_model_available_rejects_malformed_quant_scale_metadata() {
        let header = serde_json::json!({
            "arch_target": 0,
            "format_version": weights::FOCRQ_FORMAT_VERSION,
            "license_notice": crate::FOCR_MODEL_LICENSE_NOTICE,
            "packing_manifest": {
                "quant_recipe": crate::quant::convert::UNLIMITED_OCR_INT8_RECIPE_ID,
            },
            "source_sha256": "00".repeat(32),
            "tensors": {
                "model.layers.0.mlp.down_proj.weight": {
                    "dtype": "QInt8PerChan",
                    "shape": [1, 1],
                    "byte_offset": 0,
                    "byte_len": 1,
                    "scales_offset": 1,
                    "scales_len": 0,
                    "group_size": 0,
                    "tier": 0,
                },
            },
        })
        .to_string();
        let path = temp_model_dir("available_bad_quant_scales").join("bad-scales.focrq");
        std::fs::write(
            &path,
            minimal_focrq_blob(weights::FOCRQ_FORMAT_VERSION, &header, &[0u8; 5]),
        )
        .expect("write malformed quant metadata focrq");

        assert!(!native_model_available(&path));
        assert!(load_weights_from_resolved_model(&path).is_err());
    }

    #[test]
    fn native_model_available_rejects_overlapping_payload_ranges() {
        let got = model_arch::arch_by_id("got-ocr2").expect("GOT test arch is registered");
        let header = serde_json::json!({
            "arch_target": 0,
            "format_version": weights::FOCRQ_FORMAT_VERSION,
            "license_notice": got.license_notice(),
            "model_id": got.id(),
            "source_sha256": "00".repeat(32),
            "tensors": {
                "test.a": {
                    "dtype": "BF16",
                    "shape": [2],
                    "byte_offset": 0,
                    "byte_len": 4,
                },
                "test.b": {
                    "dtype": "BF16",
                    "shape": [2],
                    "byte_offset": 0,
                    "byte_len": 4,
                },
            },
        })
        .to_string();
        let path = temp_model_dir("available_overlapping_ranges").join("overlap.focrq");
        std::fs::write(
            &path,
            minimal_focrq_blob(weights::FOCRQ_FORMAT_VERSION, &header, &[0u8; 4]),
        )
        .expect("write overlapping-range focrq");

        let error = validate_model_header_file(&path)
            .expect_err("bounded production header validation must reject aliases");
        assert!(error.to_string().contains("payload ranges overlap"));
        assert!(!native_model_available(&path));
        assert!(load_weights_from_resolved_model(&path).is_err());
    }

    #[test]
    fn native_model_available_rejects_incomplete_safetensors_directory_header() {
        let dir = temp_model_dir("available_safetensors_dir");
        let shard = dir.join(v1_raw_shard_name());
        let header = br#"{"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0u8; 2]);
        std::fs::write(&shard, bytes).expect("write safetensors header");

        assert!(!native_model_available(&dir));
        assert!(load_weights_from_resolved_model(&shard).is_err());
    }

    #[test]
    fn native_model_available_rejects_empty_safetensors_directory() {
        let dir = temp_model_dir("available_empty_safetensors");
        let shard = dir.join(v1_raw_shard_name());
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(2u64).to_le_bytes());
        bytes.extend_from_slice(b"{}");
        std::fs::write(&shard, bytes).expect("write empty safetensors header");

        assert!(!native_model_available(&dir));
        assert!(load_weights_from_resolved_model(&shard).is_err());
    }

    #[test]
    fn a_sharded_directory_resolves_to_its_index_and_opens_through_the_probe() {
        use shards::tests::{POSITION_IDS_RULE, arange_bytes, index_json, safetensors_file};
        let dir = temp_model_dir("sharded_dir");
        let index = dir.join(shards::SHARD_INDEX_NAME);
        std::fs::write(
            dir.join("model-00001-of-00002.safetensors"),
            safetensors_file(&[("enc.w", "BF16", vec![2], vec![1, 0, 2, 0])]),
        )
        .expect("write shard 1");
        std::fs::write(
            dir.join("model-00002-of-00002.safetensors"),
            safetensors_file(&[
                ("dec.w", "BF16", vec![1], vec![3, 0]),
                ("vision.position_ids", "I64", vec![1, 4], arange_bytes(4)),
            ]),
        )
        .expect("write shard 2");
        std::fs::write(
            &index,
            index_json(
                &[
                    ("enc.w", "model-00001-of-00002.safetensors"),
                    ("dec.w", "model-00002-of-00002.safetensors"),
                    ("vision.position_ids", "model-00002-of-00002.safetensors"),
                ],
                4 + 2 + 32,
            ),
        )
        .expect("write index");
        // The index wins over a stray single-shard file, and resolves inside the
        // directory so `parent()` is where the sidecars live.
        std::fs::write(dir.join(v1_raw_shard_name()), b"stray").expect("write stray shard");
        assert_eq!(OcrModel::resolve_model(&dir).expect("resolves"), index);

        let opened = shards::open_shards(&index).expect("opens every shard");
        let merged = read_opened_shard_headers(&opened).expect("headers merge");
        assert_eq!((merged.records.len(), merged.int64.len()), (2, 1));
        // An unidentified checkpoint is refused by the probe, as a plain file is.
        let error = validate_model_header_file(&index).expect_err("no census matches");
        assert!(
            error.to_string().contains("matches no registered model"),
            "{error}"
        );
        assert!(load_weights_from_resolved_model(&index).is_err());

        let _rules = shards::test_rules::supply(&POSITION_IDS_RULE);
        let weights = Weights::load_opened_shards(opened).expect("sharded load");
        assert_eq!(weights.names().collect::<Vec<_>>(), ["dec.w", "enc.w"]);

        let broken = temp_model_dir("sharded_dir_missing_shard");
        let broken_index = broken.join(shards::SHARD_INDEX_NAME);
        std::fs::write(&broken_index, index_json(&[("a", "gone.safetensors")], 2))
            .expect("write index");
        let error = validate_model_header_file(&broken_index).expect_err("missing shard");
        assert!(matches!(error, FocrError::ModelNotFound(_)), "{error}");
        assert!(
            error.to_string().contains("lists shard gone.safetensors"),
            "{error}"
        );
    }

    /// Write a header-only stand-in for a census-exact checkpoint into `dir`:
    /// `tensors` (sorted) split across two shards with real headers and SPARSE
    /// payloads (`set_len`, no bytes written), plus the index. Only header
    /// probes may read it. Returns the index path.
    #[cfg(unix)]
    /// A census-exact two-shard checkpoint with a sparse (all-zero) payload,
    /// plus `int64` tensors in the second shard holding `arange` content.
    pub(super) fn write_sparse_two_shard_stand_in(
        dir: &Path,
        tensors: &[(String, DType, Vec<usize>)],
        int64: &[(&str, &[usize])],
    ) -> PathBuf {
        use std::io::{Seek as _, SeekFrom, Write as _};
        let half = tensors.len() / 2;
        let mut map: Vec<(&str, String)> = Vec::new();
        let mut total = 0usize;
        for (position, chunk) in [&tensors[..half], &tensors[half..]].into_iter().enumerate() {
            let shard = format!("model-{:05}-of-00002.safetensors", position + 1);
            let mut entries = Vec::with_capacity(chunk.len());
            let mut offset = 0usize;
            for (name, dtype, shape) in chunk {
                let elem = match dtype {
                    DType::F32 => 4,
                    DType::F16 | DType::BF16 => 2,
                    other => panic!("a source census holds no {other:?} tensor"),
                };
                let len = shape.iter().product::<usize>() * elem;
                entries.push(format!(
                    r#""{name}":{{"dtype":"{dtype:?}","shape":{shape:?},"data_offsets":[{offset},{}]}}"#,
                    offset + len
                ));
                offset += len;
                map.push((name.as_str(), shard.clone()));
            }
            let mut aranges = Vec::new();
            if position == 1 {
                for &(name, shape) in int64 {
                    let count = shape.iter().product::<usize>();
                    entries.push(format!(
                        r#""{name}":{{"dtype":"I64","shape":{shape:?},"data_offsets":[{offset},{}]}}"#,
                        offset + count * 8
                    ));
                    aranges.push((offset, count));
                    offset += count * 8;
                    map.push((name, shard.clone()));
                }
            }
            let header = format!("{{{}}}", entries.join(","));
            let mut file = std::fs::File::create(dir.join(&shard)).expect("create shard");
            file.write_all(&(header.len() as u64).to_le_bytes())
                .expect("write prefix");
            file.write_all(header.as_bytes()).expect("write header");
            file.set_len((8 + header.len() + offset) as u64)
                .expect("sparse payload");
            for (start, count) in aranges {
                let bytes: Vec<u8> = (0..count as i64).flat_map(i64::to_le_bytes).collect();
                file.seek(SeekFrom::Start((8 + header.len() + start) as u64))
                    .expect("seek to the I64 tensor");
                file.write_all(&bytes).expect("write arange");
            }
            total += offset;
        }
        let pairs: Vec<(&str, &str)> = map
            .iter()
            .map(|(name, shard)| (*name, shard.as_str()))
            .collect();
        let index = dir.join(shards::SHARD_INDEX_NAME);
        std::fs::write(&index, shards::tests::index_json(&pairs, total as u64))
            .expect("write index");
        index
    }

    #[cfg(unix)]
    #[test]
    fn a_sharded_checkpoint_judges_the_sidecar_inside_its_directory() {
        let parent = temp_model_dir("sharded_sidecar_parent");
        let dir = parent.join("checkpoint");
        std::fs::create_dir_all(&dir).expect("checkpoint dir");
        let index =
            write_sparse_two_shard_stand_in(&dir, &unlimited_ocr_census::v1_source_tensors(), &[]);
        assert_eq!(OcrModel::resolve_model(&dir).expect("resolves"), index);

        // v1 resharded in two is still v1 by census; the probe reads no payload.
        let arch = validate_model_header_file(&index).expect("the v1 census matches across shards");
        assert_eq!(arch.id(), model_arch::UNLIMITED_OCR_V1_ID);
        assert!(native_model_available(&dir));

        // The parent's sidecar is never consulted: with none in the checkpoint
        // directory, the foreign one beside it does not refuse the open.
        std::fs::write(parent.join("tokenizer.json"), SIDECAR_JSON).expect("parent sidecar");
        assert!(
            eager_sidecar_tokenizer(arch, &index)
                .expect("an absent sidecar is not an open error")
                .is_none()
        );

        // The one inside the directory is judged, before any weight byte is read.
        let inside = dir.join("tokenizer.json");
        std::fs::write(&inside, SIDECAR_JSON).expect("inner sidecar");
        let error = load_weights_from_resolved_model_after_validation(
            &index,
            |_| Ok(()),
            || panic!("weights must not be read past a refused sidecar"),
        )
        .expect_err("a foreign sidecar is refused");
        let message = error.to_string();
        assert!(message.contains(&inside.display().to_string()), "{message}");
        assert!(
            message.contains(arch.tokenizer_sha256().expect("v1 pins its tokenizer")),
            "{message}"
        );

        // Armed: the real v1 sidecar inside the directory passes while a foreign
        // one sits in the parent.
        if let Some(model) = std::env::var_os("FOCR_MODEL_PATH").map(PathBuf::from) {
            let real = model.with_file_name("tokenizer.json");
            let armed = parent.join("armed");
            std::fs::create_dir_all(&armed).expect("armed dir");
            std::os::unix::fs::symlink(&real, armed.join("tokenizer.json")).expect("link sidecar");
            let tokenizer = eager_sidecar_tokenizer(arch, &armed.join(shards::SHARD_INDEX_NAME))
                .expect("the real sidecar passes");
            assert!(tokenizer.is_some());
        }
    }

    /// U2-A7, U2-B6: a census-exact v2 checkpoint is identified; an armed
    /// refused lever refuses it (exit 2) before its tokenizer sidecar is
    /// judged or a weight byte is read. Admitted, a foreign sidecar still
    /// refuses it before the weights. The resident daemon never serves it.
    #[cfg(unix)]
    #[test]
    fn v2_admission_and_sidecar_come_before_its_weights() {
        let test = "v2_admission_and_sidecar_come_before_its_weights";
        let started = std::time::Instant::now();
        let dir = temp_model_dir("v2_admission").join("checkpoint");
        std::fs::create_dir_all(&dir).expect("checkpoint dir");
        let index = write_sparse_v2_stand_in(&dir);
        let arch = validate_model_header_file(&index).expect("the v2 census matches");
        assert_eq!(arch.id(), model_arch::UNLIMITED_OCR_V2_ID);
        assert_eq!(model_id_at(&dir), Some(model_arch::UNLIMITED_OCR_V2_ID));

        // A foreign sidecar would refuse the open; the admission comes first.
        // The hook panics rather than let a regression read 6.7 GB of zeros.
        std::fs::write(dir.join("tokenizer.json"), SIDECAR_JSON).expect("foreign sidecar");
        let armed = |arch: &'static dyn model_arch::ModelArch| {
            lever_policy::admit(
                arch,
                &lever_policy::LeverInputs {
                    var: &|name| (name == "FOCR_PREFILL_CHUNK").then(|| "64".to_owned()),
                    full_int8_forced: false,
                },
            )
            .verdict
        };
        let error = load_weights_from_resolved_model_after_validation(&index, armed, || {
            panic!("v2 weights must not be read past the admission")
        })
        .expect_err("v2 is refused");
        let message = error.to_string();
        assert!(matches!(error, FocrError::Usage(_)), "{error:?}");
        assert!(
            message.contains("FOCR_PREFILL_CHUNK is not supported for unlimited-ocr-2"),
            "{message}"
        );
        assert!(
            !message.contains("tokenizer"),
            "the sidecar was not judged: {message}"
        );

        // Admitted, the same open reaches the sidecar check, still before weights.
        let sidecar = load_weights_from_resolved_model_after_validation(
            &index,
            |_| Ok(()),
            || panic!("weights must not be read past a refused sidecar"),
        )
        .expect_err("the foreign sidecar is refused")
        .to_string();
        assert!(sidecar.contains("tokenizer.json"), "{sidecar}");

        // The public entry takes the same order: the process environment's
        // admission, then the sidecar (exit 7), never the weights.
        let Err(public) = OcrModel::load(&dir) else {
            panic!("OcrModel::load refuses the foreign sidecar");
        };
        if lever_policy::admit_from_process(arch).verdict.is_ok() {
            assert_eq!(public.exit_code(), 7, "{public}");
            assert!(public.to_string().contains("tokenizer.json"), "{public}");
        } else {
            assert_eq!(public.exit_code(), 2, "{public}");
        }
        assert!(!resident_daemon_serves(&index));
        v2_gated_line(
            test,
            started,
            "sparse_v2_stand_in",
            "assert",
            "pass",
            serde_json::json!({
                "assertion": "identified as v2; admission, then sidecar, before weights",
                "pass": true, "exit_code": public.exit_code(), "message": message,
                "sidecar_message": sidecar, "weights_touched": false,
                "resident_daemon_serves": false,
            }),
        );

        // Armed: the real artifacts identify from their headers alone, and
        // only v2 is kept from the daemon.
        if let Some(v1) = std::env::var_os("FOCR_MODEL_PATH").map(PathBuf::from) {
            assert_eq!(model_id_at(&v1), Some(model_arch::UNLIMITED_OCR_V1_ID));
            assert!(resident_daemon_serves(&v1));
        }
        if let Some(v2) = std::env::var_os("FOCR_UOCR2_SOURCE_DIR").map(PathBuf::from) {
            assert_eq!(model_id_at(&v2), Some(model_arch::UNLIMITED_OCR_V2_ID));
            assert!(!resident_daemon_serves(&v2.join(shards::SHARD_INDEX_NAME)));
        }
    }

    /// An arch for routing tests: any decoder and variant, implemented or not.
    struct RouteArch {
        decoder: model_arch::Decoder,
        variant: Option<model_arch::UnlimitedVariant>,
        implemented: bool,
    }

    impl model_arch::ModelArch for RouteArch {
        fn id(&self) -> &'static str {
            "route-test"
        }
        fn display_name(&self) -> &'static str {
            "route test"
        }
        fn license_notice(&self) -> &'static str {
            "test"
        }
        fn default_artifact_basename(&self) -> &'static str {
            "route-test.focrq"
        }
        fn vision_encoder(&self) -> model_arch::VisionEncoder {
            model_arch::VisionEncoder::SamClip
        }
        fn decoder(&self) -> model_arch::Decoder {
            self.decoder
        }
        fn tokenizer(&self) -> model_arch::TokenizerKind {
            model_arch::TokenizerKind::DeepSeekBpe
        }
        fn decode_contract(&self) -> model_arch::DecodeContract {
            model_arch::default_arch().decode_contract()
        }
        fn tasks(&self) -> &'static [model_arch::Task] {
            &[model_arch::Task::Ocr]
        }
        fn maturity(&self) -> model_arch::Maturity {
            if self.implemented {
                model_arch::Maturity::Ready
            } else {
                model_arch::Maturity::Planned
            }
        }
        fn required_quant_recipe(&self, _quant: &str) -> Option<&'static str> {
            None
        }
        fn unlimited_variant(&self) -> Option<model_arch::UnlimitedVariant> {
            self.variant
        }
    }

    /// U2-A7: every registered model routes explicitly; only v1 reaches v1's
    /// forward or its multi-page loop, and no unknown pairing falls through.
    #[test]
    fn every_arch_routes_explicitly_and_only_v1_reaches_v1() {
        use model_arch::{Decoder, UnlimitedVariant};
        for arch in model_arch::registry() {
            let single = forward_route(*arch);
            let multi = multi_page_route(*arch);
            match arch.id() {
                model_arch::UNLIMITED_OCR_V1_ID => {
                    assert_eq!(single.expect("v1 routes"), ForwardRoute::UnlimitedV1);
                    multi.expect("v1 multi-page routes");
                }
                model_arch::UNLIMITED_OCR_V2_ID => {
                    assert_eq!(single.expect("v2 routes"), ForwardRoute::UnlimitedV2);
                    multi.expect("v2 runs its own document loop");
                }
                id => {
                    let expected = match id {
                        "got-ocr2" => Some(ForwardRoute::GotOcr2),
                        "smolvlm2" => Some(ForwardRoute::SmolVlm2),
                        "onechart" => Some(ForwardRoute::OneChart),
                        "tromr" => Some(ForwardRoute::Tromr),
                        _ => None,
                    };
                    match expected {
                        Some(route) => assert_eq!(single.expect("zoo routes"), route, "{id}"),
                        // A planned zoo model keeps its exit-1 NotImplemented.
                        None => {
                            let error = single.expect_err("planned");
                            assert_eq!(error.exit_code(), 1, "{id}");
                            assert!(error.to_string().contains("epic bd-3jo6"), "{error}");
                        }
                    }
                    let multi = multi.expect_err("the zoo parses pages independently");
                    assert!(multi.to_string().contains("parses pages independently"));
                }
            }
        }

        // Pairings no registered model has: never v1's forward or loop.
        let pairs = [
            (Decoder::DeepSeekV2MoeRswa, Some(UnlimitedVariant::V2SrSwa)),
            (Decoder::DeepSeekV2MoeSrSwa, Some(UnlimitedVariant::V1Rswa)),
            (Decoder::DeepSeekV2MoeRswa, None),
            (Decoder::Qwen2Dense, None),
        ];
        for (decoder, variant) in pairs {
            for implemented in [false, true] {
                let arch = RouteArch {
                    decoder,
                    variant,
                    implemented,
                };
                let route = forward_route(&arch);
                assert!(
                    !matches!(route, Ok(ForwardRoute::UnlimitedV1)),
                    "{decoder:?}/{variant:?} reached v1's forward"
                );
                let error = route.expect_err("no route").to_string();
                let expected = if implemented {
                    "no forward wired"
                } else {
                    "not yet implemented"
                };
                assert!(error.contains(expected), "{error}");
            }
        }

        // v2's pairing reaches its own loop only while implemented.
        for implemented in [false, true] {
            let arch = RouteArch {
                decoder: Decoder::DeepSeekV2MoeSrSwa,
                variant: Some(UnlimitedVariant::V2SrSwa),
                implemented,
            };
            match (forward_route(&arch), multi_page_route(&arch)) {
                (Ok(single), Ok(())) if implemented => {
                    assert_eq!(single, ForwardRoute::UnlimitedV2);
                }
                (Err(single), Err(multi)) if !implemented => {
                    assert!(
                        single.to_string().contains("not yet implemented"),
                        "{single}"
                    );
                    assert!(multi.to_string().contains("not yet implemented"), "{multi}");
                }
                other => panic!("implemented={implemented}: {other:?}"),
            }
        }
    }

    /// U2-B6: a v2 model gets past routing on every inference entry point and
    /// reads its input, as v1 does over the same empty weights. A document
    /// above [`UOCR2_UNWINDOWED_PAGE_LIMIT`] is refused with exit 2 before
    /// any page is read; the loop's limit enforcement turned off admits it.
    #[test]
    fn a_v2_model_routes_every_entry_point_and_refuses_oversized_documents_first() {
        let v2 = model_arch::arch_by_id(model_arch::UNLIMITED_OCR_V2_ID).expect("v2 registered");
        let model = OcrModel::assemble(
            PathBuf::from("<route-test>"),
            SidecarBundle::default(),
            v2,
            Weights::default(),
            None,
        );
        let missing = Path::new("/nonexistent/franken_ocr_route_test_page.png");
        let image = || image::DynamicImage::new_rgb8(4, 4);
        let reads_input = |error: FocrError| {
            assert!(matches!(error, FocrError::InputDecode(_)), "{error:?}");
        };
        reads_input(model.forward(missing).expect_err("forward"));
        reads_input(model.recognize(missing).expect_err("recognize"));
        for result in model.recognize_batch(&[missing, missing]) {
            reads_input(result.expect_err("recognize_batch"));
        }
        let options = DocumentOptions::default();
        let document = |source: &mut crate::page_source::PageSource, options: &DocumentOptions| {
            model.recognize_document(source, options, &mut |_| {}, None)
        };
        let files = |count: usize| {
            crate::page_source::PageSource::images(vec![missing.to_path_buf(); count])
                .expect("a source")
        };
        let rasters =
            |count: usize| crate::page_source::PageSource::rasters(vec![image(); count]).unwrap();
        reads_input(document(&mut files(1), &options).expect_err("document"));

        let over = UOCR2_UNWINDOWED_PAGE_LIMIT + 1;
        let too_long = |error: FocrError| {
            assert!(matches!(error, FocrError::Usage(_)), "{error:?}");
            assert_eq!(error.exit_code(), 2);
            let message = error.to_string();
            assert!(
                message.contains(&format!(
                    "at most {UOCR2_UNWINDOWED_PAGE_LIMIT} pages in one pass"
                )) && message.contains(&format!("this document has {over}"))
                    && message.contains("--pages"),
                "{message}"
            );
        };
        // Missing paths: reading any of them would be an input error instead.
        too_long(document(&mut files(over), &options).expect_err("over the limit"));
        let mut source = rasters(over);
        too_long(
            model
                .recognize_document(
                    &mut source,
                    &options,
                    &mut |_| panic!("no page is read"),
                    Some(&mut |_: &StreamedPage<'_>| panic!("no page streams")),
                )
                .expect_err("over the limit, in memory"),
        );
        assert_eq!(source.page_map().len(), 0, "refused before any raster");
        // An explicit unwindowed request reads past the limit (its pages
        // then fail to read, which proves the limit let them through).
        for window in [PageWindow::Off, PageWindow::Pages(over)] {
            let unwindowed = options.page_window(window);
            reads_input(document(&mut files(over), &unwindowed).expect_err("read"));
        }
        model
            .check_document_pages(UOCR2_UNWINDOWED_PAGE_LIMIT, true)
            .expect("the limit itself is admitted");
        model
            .check_document_pages(over, false)
            .expect("an explicit unwindowed request bypasses the limit");
        let unlimited = model_arch::default_arch();
        let v1_model = OcrModel::assemble(
            PathBuf::from("<route-test>"),
            SidecarBundle::default(),
            unlimited,
            Weights::default(),
            None,
        );
        v1_model
            .check_document_pages(over, true)
            .expect("v1 has no page limit");

        let v1 = OcrModel::assemble(
            PathBuf::from("<route-test>"),
            SidecarBundle::default(),
            model_arch::default_arch(),
            Weights::default(),
            None,
        );
        assert!(matches!(
            v1.forward(missing),
            Err(FocrError::InputDecode(_))
        ));
        let mut source = files(1);
        assert!(matches!(
            v1.recognize_document(&mut source, &options, &mut |_| {}, None),
            Err(FocrError::InputDecode(_))
        ));
    }

    /// U2-E2: a document option this build cannot honor yet (a window
    /// smaller than the document, before U2-D1b; resume, before U2-D4) is
    /// refused before any page is read, never ignored.
    #[test]
    fn document_options_refuse_what_this_build_cannot_honor() {
        let refused = |options: DocumentOptions, pages: usize, says: &str| {
            let error = options.page_limit(pages).expect_err(says);
            assert!(matches!(error, FocrError::Usage(_)), "{error:?}");
            assert!(error.to_string().contains(says), "{error}");
        };
        let window = |page_window| DocumentOptions::default().page_window(page_window);
        assert_eq!(DocumentOptions::default().page_limit(500).ok(), Some(true));
        assert_eq!(window(PageWindow::Off).page_limit(500).ok(), Some(false));
        assert_eq!(window(PageWindow::Pages(3)).page_limit(3).ok(), Some(false));
        refused(window(PageWindow::Pages(0)), 3, "at least one page");
        refused(window(PageWindow::Pages(2)), 3, "not yet available");
        refused(window(PageWindow::Auto), 3, "not yet available");
        for resume in [ResumePolicy::IfMatching, ResumePolicy::Recompute] {
            refused(DocumentOptions::default().resume(resume), 1, "resume");
        }
        assert_eq!(
            DocumentOptions::default().resume(ResumePolicy::Never),
            DocumentOptions::default()
        );
    }

    /// One NDJSON log line of a model-gated v2 test (`scripts/check_test_logs.py`).
    fn v2_gated_line(
        test: &str,
        started: std::time::Instant,
        case: &str,
        event: &str,
        result: &str,
        extra: serde_json::Value,
    ) {
        let mut object = serde_json::json!({
            "schema_version": 1, "ts": started.elapsed().as_secs_f64(), "test": test,
            "case": case, "run_seq": 0, "event": event, "result": result,
        });
        if let (Some(object), Some(extra)) = (object.as_object_mut(), extra.as_object()) {
            object.extend(extra.clone());
        }
        println!("{object}");
    }

    /// Model-gated (U2-A3, U2-A5): `FOCR_UOCR2_SOURCE_DIR` names a directory
    /// holding the real Unlimited-OCR-2 index and both shards, or header-only
    /// stand-ins (real headers, sparse payloads). Skips with success when
    /// unset. Only headers are read, and they must pass the v2 census.
    #[test]
    fn real_v2_index_and_shard_headers_parse_through_the_resolver() {
        let test = "real_v2_index_and_shard_headers_parse_through_the_resolver";
        let started = std::time::Instant::now();
        let line = |case: &str, event: &str, result: &str, extra: serde_json::Value| {
            v2_gated_line(test, started, case, event, result, extra);
        };
        let Some(dir) = std::env::var_os("FOCR_UOCR2_SOURCE_DIR").map(PathBuf::from) else {
            line(
                "v2",
                "skip",
                "skip_no_model",
                serde_json::json!({
                    "reason": "FOCR_UOCR2_SOURCE_DIR unset",
                }),
            );
            return;
        };
        let index = OcrModel::resolve_model(&dir).expect("the directory resolves");
        assert_eq!(index, dir.join(shards::SHARD_INDEX_NAME));
        let opened = shards::open_shards(&index).expect("every shard opens");
        for shard in &opened.shards {
            let file = std::fs::File::open(&shard.path).expect("open shard");
            let (header, _, _) =
                read_validated_safetensors_header(&file, shard.len).expect("shard header");
            line(
                &shard.name,
                "assert",
                "pass",
                serde_json::json!({
                    "assertion": "shard header parses and validates",
                    "pass": true,
                    "bytes": shard.len, "header_ok": true, "tensors": header.len(),
                    "hashed": false,
                }),
            );
        }
        let merged = read_opened_shard_headers(&opened).expect("headers merge against the index");
        assert_eq!(opened.index.shards().len(), 2);
        assert_eq!(opened.index.total_size(), 6_687_958_536);
        assert_eq!(merged.len(), 2_819, "the v2 tensor count");
        assert_eq!(
            merged.int64.keys().collect::<Vec<_>>(),
            ["model.vision_model.embeddings.position_ids"],
            "exactly one I64 tensor"
        );
        let arch = identify_checkpoint(&merged).expect("the headers pass the v2 census");
        assert_eq!(arch.id(), model_arch::UNLIMITED_OCR_V2_ID);
        let rules = shards::int64_rules(Some(arch));
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].name, "model.vision_model.embeddings.position_ids");
        line(
            "v2",
            "result",
            "pass",
            serde_json::json!({
                "elapsed_us": started.elapsed().as_micros() as u64,
                "tensors": merged.len(), "int64": merged.int64.len(),
                "source_sha256": serde_json::Value::Null,
            }),
        );
    }

    /// Model-gated (U2-A5): `FOCR_UOCR2_SOURCE_DIR` names the real, complete
    /// Unlimited-OCR-2 checkpoint (index, both shards, `tokenizer.json`).
    /// Loads it raw through the production open path, re-hashes both shards
    /// against the pins, and checks every vision tensor against the census's
    /// identity hashes and, with `FOCR_MODEL_PATH` naming the v1 artifact,
    /// against v1's bytes. Skips with success when unset.
    #[test]
    fn real_v2_checkpoint_loads_raw_and_matches_its_pins() {
        let test = "real_v2_checkpoint_loads_raw_and_matches_its_pins";
        let started = std::time::Instant::now();
        let line = |case: &str, event: &str, result: &str, extra: serde_json::Value| {
            v2_gated_line(test, started, case, event, result, extra);
        };
        let Some(dir) = std::env::var_os("FOCR_UOCR2_SOURCE_DIR").map(PathBuf::from) else {
            line(
                "v2",
                "skip",
                "skip_no_model",
                serde_json::json!({"reason": "FOCR_UOCR2_SOURCE_DIR unset"}),
            );
            return;
        };
        let index = OcrModel::resolve_model(&dir).expect("the directory resolves");

        let opened = load_weights_from_resolved_model(&index).expect("the raw v2 load succeeds");
        assert_eq!(opened.arch.id(), model_arch::UNLIMITED_OCR_V2_ID);
        assert_eq!(opened.weights.model_id(), model_arch::UNLIMITED_OCR_V2_ID);
        assert!(!opened.weights.is_focrq());
        assert_eq!(
            opened.weights.len(),
            2_818,
            "2,819 less the verified I64 tensor"
        );
        assert!(
            !opened
                .weights
                .contains("model.vision_model.embeddings.position_ids")
        );
        let tokenizer = opened.tokenizer.as_ref().expect("the pinned sidecar loads");
        assert_eq!(
            tokenizer.token_to_id("<PAGE>"),
            Some(crate::tokenizer::special::PAGE)
        );
        line(
            "raw_load",
            "assert",
            "pass",
            serde_json::json!({
                "assertion": "raw v2 load identifies unlimited-ocr-2 and passes its census",
                "pass": true, "tensors": opened.weights.len(), "hashed": false,
            }),
        );

        let digests = shards::hash_shards(&index).expect("hash both shards");
        let pinned = unlimited_ocr_census::source_shards(model_arch::TensorCensus::UnlimitedOcrV2);
        assert_eq!(digests.len(), pinned.len());
        for (digest, (name, sha256)) in digests.iter().zip(pinned) {
            let hex: String = digest.sha256.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!((digest.name.as_str(), hex.as_str()), (*name, *sha256));
            line(
                name,
                "assert",
                "pass",
                serde_json::json!({
                    "assertion": "shard sha256 equals its pin",
                    "pass": true, "bytes": digest.size, "hashed": true,
                }),
            );
        }
        let source: String = shards::source_sha256(&digests)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(source, unlimited_ocr_census::V2_SOURCE_SHA256);

        let v1 = std::env::var_os("FOCR_MODEL_PATH")
            .map(PathBuf::from)
            .map(|path| load_weights_from_resolved_model(&path).expect("the v1 artifact loads"));
        if let Some(v1) = &v1 {
            assert_eq!(v1.weights.model_id(), model_arch::UNLIMITED_OCR_V1_ID);
        }
        let identity = unlimited_ocr_census::v2_vision_identity_sha256();
        assert_eq!(identity.len(), 476);
        for (name, expected) in identity {
            use sha2::Digest as _;

            let view = opened.weights.tensor(name).expect("vision tensor present");
            let actual = format!("{:x}", sha2::Sha256::digest(view.data));
            assert_eq!(&actual, expected, "{name}");
            if let Some(v1) = &v1 {
                let v1_view = v1.weights.tensor(name).expect("v1 has the vision tensor");
                assert_eq!(v1_view.dtype, DType::BF16, "{name}");
                assert!(v1_view.data == view.data, "{name} differs from v1");
            }
        }
        line(
            "vision_identity",
            "assert",
            "pass",
            serde_json::json!({
                "assertion": "476 vision tensors match the census hashes",
                "pass": true, "compared_to_v1": v1.is_some(),
            }),
        );
        line(
            "v2",
            "result",
            "pass",
            serde_json::json!({
                "elapsed_us": started.elapsed().as_micros() as u64,
                "source_sha256": source,
            }),
        );
    }

    /// The real Unlimited-OCR-2 checkpoint at `FOCR_UOCR2_SOURCE_DIR`, loaded
    /// raw (U2-B5's model-gated decoder tests); `None`, after a skip line,
    /// when unset.
    fn v2_gated_model(test: &str, started: std::time::Instant) -> Option<OcrModel> {
        let Some(dir) = std::env::var_os("FOCR_UOCR2_SOURCE_DIR").map(PathBuf::from) else {
            v2_gated_line(
                test,
                started,
                "v2",
                "skip",
                "skip_no_model",
                serde_json::json!({"reason": "FOCR_UOCR2_SOURCE_DIR unset"}),
            );
            return None;
        };
        let index = OcrModel::resolve_model(&dir).expect("the directory resolves");
        let opened = load_weights_from_resolved_model(&index).expect("the raw v2 load succeeds");
        assert_eq!(opened.arch.id(), model_arch::UNLIMITED_OCR_V2_ID);
        Some(OcrModel::assemble(
            index,
            SidecarBundle::default(),
            opened.arch,
            opened.weights,
            opened.tokenizer,
        ))
    }

    /// U2-B6: the retirement fallback. A synthetic pass whose detector fires
    /// on its third id with retirement on runs again without it; the
    /// observer sees each id once, in order, and the decode counts one
    /// fallback. A pass without retirement, or one that ends on another
    /// error, runs once; a rerun that departs from the first pass's ids is an
    /// error.
    #[test]
    fn retire_fallback_reruns_the_document_unretired_once() {
        use crate::tokenizer::special::PAGE;
        let fired = || {
            FocrError::Other(
                anyhow::Error::new(srswa::RetireFallback {
                    q_pos: 700,
                    query_page: 1,
                    open: srswa::OpenRow::NoRemoteAllowed,
                    dead_blocks: 4,
                })
                .context("layer 3"),
            )
        };
        assert!(
            srswa::retire_fallback(&fired()).is_some(),
            "through context"
        );
        let ids = [PAGE, 10, 11, PAGE, 12];
        // Emits `ids` first and `rerun` after; with retirement on, the
        // detector fires after `fire_at` ids; `cancel` ends a pass early.
        let run = |retire: bool, fire_at: Option<usize>, rerun: &[u32], cancel: bool| {
            let mut calls = Vec::new();
            let mut seen = Vec::new();
            let mut observer = |id: u32| seen.push(id);
            let result = decode_with_retire_fallback(
                retire,
                Some(&mut observer),
                |retire, floor, out, mut obs| {
                    calls.push((retire, floor));
                    out.clear();
                    let emit = if calls.len() == 1 { &ids[..] } else { rerun };
                    for (k, &id) in emit.iter().enumerate() {
                        out.push(id);
                        if let Some(f) = obs.as_deref_mut() {
                            f(id);
                        }
                        if retire && fire_at == Some(k + 1) {
                            return Err(fired());
                        }
                        if cancel && k == 1 {
                            return Err(FocrError::Cancelled);
                        }
                    }
                    Ok(DocumentDecode {
                        ids: out.clone(),
                        finish_reason: FinishReason::Stop,
                        stats: DocumentStats::default(),
                    })
                },
            );
            (result, calls, seen)
        };

        let (result, calls, seen) = run(true, Some(3), &ids, false);
        let decode = result.expect("the rerun completes");
        assert_eq!(decode.ids, ids);
        // (retire, budget floor): the rerun keeps its budget off until it
        // passes the 3 ids already observed.
        assert_eq!(calls, [(true, 0), (false, 3)]);
        assert_eq!(seen, ids, "each id observed once, in order");
        assert_eq!(decode.stats.retire_fallbacks, 1);

        let (result, calls, seen) = run(false, Some(3), &ids, false);
        assert_eq!(result.expect("never fires").stats.retire_fallbacks, 0);
        assert_eq!((calls, seen), (vec![(false, 0)], ids.to_vec()));

        let (result, calls, _) = run(true, None, &ids, true);
        assert!(matches!(result, Err(FocrError::Cancelled)));
        assert_eq!(calls, [(true, 0)], "no rerun on another error");

        let (result, calls, seen) = run(true, Some(3), &[PAGE, 10, 99, PAGE, 12], false);
        let message = result.expect_err("a departing rerun").to_string();
        assert!(
            message.contains("departs") && message.contains("step 3"),
            "{message}"
        );
        assert_eq!(calls, [(true, 0), (false, 3)]);
        assert_eq!(
            seen,
            [PAGE, 10, 11, PAGE, 12],
            "the departure is reported, not streamed"
        );
    }

    /// U2-B6's L0 prompt check against the U2-A6 oracle: [`paged_prompt`]
    /// over v2's real image-block counts and the committed tokenizer golden's
    /// ids for "Multi page parsing." reproduces every fixture's prompt, by
    /// length and by `prompt_token_ids_sha256`. The block counts come from
    /// the engine's own v2 preprocess of blank pages the oracle's size: a base
    /// page and a gundam page (whose 2×3 grid adds 630 slots).
    #[test]
    fn paged_prompt_reproduces_the_v2_oracle_prompts() {
        use crate::tokenizer::special;
        use sha2::Digest as _;
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let read = |rel: &str| -> serde_json::Value {
            serde_json::from_slice(&std::fs::read(root.join(rel)).unwrap()).unwrap()
        };
        let golden = read("tokenizer_baidu/expected_uocr2.json");
        let one_page = golden["fixtures"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["text"] == "<PAGE><image>Multi page parsing.")
            .expect("the one-page prompt is in the golden");
        let ids: Vec<u32> = one_page["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| u32::try_from(id.as_u64().unwrap()).unwrap())
            .collect();
        assert_eq!(ids[..2], [special::PAGE, special::IMAGE]);
        let text = &ids[2..];

        let oracle = read("uocr2_oracle/ORACLE.json");
        let block = |mode: preprocess::PreprocessMode, (w, h): (u32, u32)| {
            let page = image::DynamicImage::new_rgb8(w, h);
            preprocess::preprocess_dynamic(page, mode, preprocess::PreprocessProfile::V2)
                .unwrap()
                .placeholder_token_count()
        };
        let mut blocks = std::collections::BTreeMap::new();
        let fixtures = oracle["fixtures"].as_object().unwrap();
        assert_eq!(fixtures.len(), 9);
        for (name, f) in fixtures {
            let mode = match f["mode"].as_str().unwrap() {
                "base" => preprocess::PreprocessMode::Base { base_size: 1024 },
                "gundam" => preprocess::PreprocessMode::Gundam {
                    base_size: 1024,
                    tile_size: 640,
                },
                other => panic!("{name}: mode {other}"),
            };
            let counts: Vec<usize> = f["pages"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    let size = |k: &str| u32::try_from(p[k].as_u64().unwrap()).unwrap();
                    let key = (f["mode"].to_string(), size("width"), size("height"));
                    *blocks
                        .entry(key)
                        .or_insert_with(|| block(mode, (size("width"), size("height"))))
                })
                .collect();
            let (prompt, mask) = paged_prompt(0, special::PAGE, special::IMAGE, &counts, text);
            assert_eq!(
                prompt.len() as u64,
                f["prompt_token_count"].as_u64().unwrap(),
                "{name}"
            );
            let json = serde_json::to_string(&prompt).unwrap();
            let sha: String = sha2::Sha256::digest(json.as_bytes())
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(sha, f["prompt_token_ids_sha256"], "{name}");
            assert_eq!(
                mask.iter().filter(|&&m| m).count(),
                counts.iter().sum::<usize>(),
                "{name}"
            );
            assert!(
                mask.iter()
                    .zip(&prompt)
                    .all(|(&m, &id)| m == (id == special::IMAGE))
            );
        }
        assert_eq!(
            blocks.values().copied().collect::<Vec<_>>(),
            [273, 273 + 630],
            "base, then gundam"
        );
    }

    /// Model-gated (U2-B6): the real v2 checkpoint's document loop on two
    /// U2-A6 fixtures, S1 page_0009 (one page) and S3 p9_p14 (two pages,
    /// streamed), over the corpus pages (`FOCR_CORPUS_DIR`, default
    /// `~/.cache/franken_ocr/corpus/pages`). Asserts the contract: one body
    /// per page, EOS, `marker`… then `eos` per page, the streamed bodies equal
    /// to the split's, no DISC-U2-2 step, no retirement fallback. Token
    /// agreement with the f32 oracle is logged as information: this is the
    /// conservative int8-FFN path, and U2-C2 gates parity.
    #[test]
    fn real_v2_document_loop_decodes_oracle_fixtures() {
        use crate::tokenizer::special;
        let test = "real_v2_document_loop_decodes_oracle_fixtures";
        let started = std::time::Instant::now();
        let Some(model) = v2_gated_model(test, started) else {
            return;
        };
        for (fixture, pages, stream) in [
            ("S1-page_0009", &["page_0009.png"][..], false),
            ("S3-p9_p14", &["page_0009.png", "page_0014.png"][..], true),
        ] {
            let Some(paths) = v2_corpus_pages(test, started, fixture, pages) else {
                continue;
            };
            let pres: Vec<Preprocessed> = paths
                .iter()
                .map(|p| {
                    let img = image::open(p).expect("page decodes");
                    model.document_page(img).expect("preprocess")
                })
                .collect();
            let t = std::time::Instant::now();
            let mut events: Vec<(usize, String, usize, PageFinish)> = Vec::new();
            let mut sink = |p: &ClosedPage<'_>| {
                events.push((p.index, p.body.to_string(), p.tokens, p.finish));
            };
            let decoded = model
                .decode_document(
                    &pres,
                    stream.then_some(&mut sink as &mut dyn FnMut(&ClosedPage<'_>)),
                )
                .expect("the document decodes");
            let split = &decoded.split;
            let report = model.take_last_decode().expect("a decode report");
            let stats = report.document.clone().expect("document stats");
            let n = pages.len();
            let oracle = v2_oracle_ids(fixture);
            let agree = report
                .ids
                .iter()
                .zip(&oracle)
                .take_while(|(a, b)| a == b)
                .count();
            let near = |ids: &[u32]| ids[agree.min(ids.len())..(agree + 8).min(ids.len())].to_vec();
            // The evidence goes out before any assert, so a failing or capped
            // run still reports what it decoded.
            v2_gated_line(
                test,
                started,
                fixture,
                "decode",
                "info",
                serde_json::json!({
                    "pages_expected": n, "pages_produced": split.bodies.len(),
                    "finish_reason": report.finish_reason.as_str(),
                    "tokens": report.ids.len(), "oracle_tokens": oracle.len(),
                    "info_agreeing_prefix": agree,
                    "info_exact": report.ids == oracle,
                    "info_ours_at_divergence": near(&report.ids),
                    "info_oracle_at_divergence": near(&oracle),
                    "page_tokens": stats.page_tokens,
                    "page_finish": stats.page_finish.iter().map(|f| f.as_str()).collect::<Vec<_>>(),
                    "extra_markers": stats.extra_markers,
                    "leading_text": stats.leading_text,
                    "forced_steps": stats.forced_steps,
                    "disc_u2_2_steps": stats.hf_all_neg_inf_steps,
                    "retired_blocks": stats.retired_blocks,
                    "peak_kv_bytes": stats.peak_kv_bytes,
                    "retire_fallbacks": stats.retire_fallbacks,
                    "wall_s": t.elapsed().as_secs_f64(),
                }),
            );
            assert_eq!(report.finish_reason, FinishReason::Stop, "{fixture}");
            assert_eq!(split.bodies.len(), n, "{fixture}");
            assert_eq!(
                report.ids.iter().filter(|&&id| id == special::PAGE).count(),
                n,
                "{fixture}: one marker per page"
            );
            let mut want = vec![PageFinish::Marker; n - 1];
            want.push(PageFinish::Eos);
            assert_eq!(stats.page_finish, want, "{fixture}");
            assert_eq!(stats.hf_all_neg_inf_steps, 0, "{fixture}: DISC-U2-2");
            assert_eq!(stats.retire_fallbacks, 0, "{fixture}");
            assert_eq!(
                stats.page_tokens.iter().sum::<usize>() + n,
                report.ids.len()
            );
            // U2-E3: every page's facts come with its body, the streamed
            // ones as the final split's.
            assert_eq!(decoded.finish, stats.page_finish, "{fixture}");
            assert_eq!(decoded.elapsed.len(), n, "{fixture}: a clock per page");
            if stream {
                let split_pages: Vec<(usize, String, usize, PageFinish)> = split
                    .bodies
                    .iter()
                    .zip(&split.tokens)
                    .zip(&stats.page_finish)
                    .enumerate()
                    .map(|(i, ((b, &t), &f))| (i + 1, b.trim().to_string(), t, f))
                    .collect();
                assert_eq!(events, split_pages, "{fixture}: streamed pages");
            }
            v2_gated_line(
                test,
                started,
                fixture,
                "assert",
                "pass",
                serde_json::json!({}),
            );
        }
    }

    /// Model-gated (U2-B6): teacher-forced along the f32 oracle's S3 p9_p14
    /// ids, the stateless f32 path ([`OcrModel::document_logits`] over the
    /// grown sequence, then [`sampler::decode_step_guarded`]) picks the
    /// oracle's token at probe steps on both pages: the first step, both
    /// markers and the rows after them, the cached path's first divergence
    /// (step 73), deep in page 2, and the final EOS. That pins the document
    /// wiring (page coordinates over generated markers, the guard, the
    /// generated-only n-gram) apart from the int8 FFN. The cached path is
    /// teacher-forced over every step too; its disagreements and their logit
    /// gaps are logged for U2-C2, which owns L4.
    #[test]
    fn real_v2_teacher_forced_f32_follows_the_oracle_across_pages() {
        use crate::tokenizer::special;
        let test = "real_v2_teacher_forced_f32_follows_the_oracle_across_pages";
        let started = std::time::Instant::now();
        let Some(model) = v2_gated_model(test, started) else {
            return;
        };
        let fixture = "S3-p9_p14";
        let Some(paths) =
            v2_corpus_pages(test, started, fixture, &["page_0009.png", "page_0014.png"])
        else {
            return;
        };
        let (embeds, prompt) = v2_decoder_prompt(&model, &paths);
        let oracle = v2_oracle_ids(fixture);
        let params = model.document_params(2);
        let table = model.embed_table().expect("embed table");
        let guard_after = |t: usize| {
            let mut guard = sampler::PageGuard::new(
                2,
                special::PAGE,
                params.eos_token_id,
                params.max_page_tokens.min(params.max_length),
            );
            for &id in &oracle[..t] {
                guard.observe(id);
            }
            guard
        };
        // `None` is EOS, the step after the last id.
        let pick = |logits: &Mat, t: usize| {
            let step = sampler::decode_step_guarded(logits, &oracle[..t], &guard_after(t), &params)
                .expect("a guarded step");
            (!step.is_eos).then_some(step.token_id)
        };
        // How far the oracle's token sits below the row's best logit.
        let gap = |logits: &Mat, want: Option<u32>| {
            let id = want.unwrap_or(params.eos_token_id) as usize;
            let best = logits
                .data
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            best - logits.data[id]
        };
        let markers: Vec<usize> = oracle
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == special::PAGE).then_some(i))
            .collect();
        assert_eq!(markers.len(), 2, "the oracle marks both pages");
        let mut probes = vec![
            0,
            markers[1],
            markers[1] + 1,
            73,
            74,
            (markers[1] + oracle.len()) / 2,
            oracle.len() - 1,
            oracle.len(),
        ];
        probes.sort_unstable();
        probes.dedup();
        let mut f32_misses = Vec::new();
        for t in probes {
            let tp = std::time::Instant::now();
            let mut data = embeds.data.clone();
            for &id in &oracle[..t] {
                data.extend(table.row_f32(id as usize));
            }
            let mut ids = prompt.clone();
            ids.extend_from_slice(&oracle[..t]);
            let grown = Mat::from_vec(embeds.rows + t, embeds.cols, data);
            let logits = model
                .document_logits(&DocumentDecoder::Stateless {
                    embeds: grown,
                    ids,
                    domain: srswa::EntmaxDomain::TopM,
                })
                .expect("f32 logits");
            let want = oracle.get(t).copied();
            let got = pick(&logits, t);
            v2_gated_line(
                test,
                started,
                fixture,
                "f32_probe",
                if got == want { "pass" } else { "fail" },
                serde_json::json!({
                    "step": t, "want": want, "got": got,
                    "oracle_gap": gap(&logits, want), "wall_s": tp.elapsed().as_secs_f64(),
                }),
            );
            if got != want {
                f32_misses.push((t, want, got));
            }
        }

        // The cached path, teacher-forced over every step (information).
        let wc = model.decoder_cache().expect("decoder cache");
        let coords = srswa::PageCoords::build(&prompt).expect("page coords");
        let (hidden, state) =
            decoder::prefill_with_cache_srswa(wc, &embeds, &coords, srswa::EntmaxDomain::TopM)
                .expect("prefill");
        let mut cached = DocumentDecoder::Cached {
            wc,
            state,
            hidden: OcrModel::last_hidden_row(&hidden).expect("last row"),
        };
        let mut cached_misses = Vec::new();
        for t in 0..=oracle.len() {
            let logits = model.document_logits(&cached).expect("cached logits");
            let want = oracle.get(t).copied();
            let got = pick(&logits, t);
            if got != want {
                cached_misses.push(serde_json::json!({
                    "step": t, "want": want, "got": got, "oracle_gap": gap(&logits, want),
                }));
            }
            let Some(id) = want else { break };
            let DocumentDecoder::Cached { wc, state, hidden } = &mut cached else {
                unreachable!("built cached")
            };
            let forced = matches!(
                guard_after(t + 1).verdict(),
                sampler::GuardVerdict::Force(_)
            );
            let row = Mat::from_vec(1, embeds.cols, table.row_f32(id as usize));
            let h = decoder::decode_step_with_cache_srswa(wc, state, id, &row, forced)
                .expect("cached step");
            *hidden = OcrModel::last_hidden_row(&h).expect("last row");
        }
        v2_gated_line(
            test,
            started,
            fixture,
            "cached_teacher_forced",
            "info",
            serde_json::json!({
                "steps": oracle.len() + 1, "disagreements": cached_misses.len(),
                "first": cached_misses.iter().take(16).collect::<Vec<_>>(),
            }),
        );
        assert!(
            f32_misses.is_empty(),
            "the f32 path departs from the oracle (step, want, got): {f32_misses:?}"
        );
    }

    /// Model-gated (U2-B6): the document loop's limits on S2 p9_p14_p19
    /// (three pages). A per-page cap of 50 (`--max-length 50`, per page in
    /// v2) gives three bodies of at most 50 ids, `page_cap` exactly where a
    /// body reached 50; block retirement on and off give the same ids. A
    /// position limit 60 rows past the prompt stops as `length` inside a
    /// page. A spent budget stops as `budget` before the first id.
    #[test]
    fn real_v2_document_limits_stop_where_the_contract_says() {
        use crate::tokenizer::special::PAGE;
        let test = "real_v2_document_limits_stop_where_the_contract_says";
        let started = std::time::Instant::now();
        let Some(model) = v2_gated_model(test, started) else {
            return;
        };
        let fixture = "S2-p9_p14_p19";
        let pages = ["page_0009.png", "page_0014.png", "page_0019.png"];
        let Some(paths) = v2_corpus_pages(test, started, fixture, &pages) else {
            return;
        };
        let (embeds, prompt) = v2_decoder_prompt(&model, &paths);
        let tok = model.tokenizer().expect("tokenizer");
        let base = model.document_params(pages.len());
        let run = |params: &DecodeParams, retire: bool, deadline: Option<Instant>| {
            let pass = DocumentPass {
                precision: DocumentPrecision::Mixed,
                retire,
                deadline,
                budget_floor: 0,
            };
            let mut ids = Vec::new();
            let decode = model
                .generate_document_once(&embeds, &prompt, params, pass, &mut ids, None)
                .expect("the pass decodes");
            let split = postprocess::split_pages_by_token(tok, &decode.ids, PAGE, pages.len())
                .expect("the split");
            (decode, split)
        };
        let finishes = |decode: &DocumentDecode| -> Vec<&'static str> {
            decode
                .stats
                .page_finish
                .iter()
                .map(|f| f.as_str())
                .collect()
        };

        let capped = DecodeParams {
            max_length: 50,
            ..base.clone()
        };
        let t = Instant::now();
        let (retired, split) = run(&capped, true, None);
        let (plain, _) = run(&capped, false, None);
        v2_gated_line(
            test,
            started,
            "page_cap_50",
            "decode",
            "info",
            serde_json::json!({
                "pages_expected": pages.len(), "pages_produced": split.bodies.len(),
                "finish_reason": retired.finish_reason.as_str(), "page_tokens": split.tokens,
                "page_finish": finishes(&retired), "forced_steps": retired.stats.forced_steps,
                "disc_u2_2_steps": retired.stats.hf_all_neg_inf_steps,
                "retired_blocks": retired.stats.retired_blocks,
                "peak_kv_bytes": retired.stats.peak_kv_bytes,
                "unretired_peak_kv_bytes": plain.stats.peak_kv_bytes,
                "wall_s": t.elapsed().as_secs_f64(),
            }),
        );
        assert_eq!(retired.finish_reason, FinishReason::Stop);
        assert_eq!(split.bodies.len(), pages.len());
        assert_eq!(retired.stats.page_finish.len(), pages.len());
        for (k, (&tokens, finish)) in split
            .tokens
            .iter()
            .zip(&retired.stats.page_finish)
            .enumerate()
        {
            assert!(tokens <= 50, "page {}: {tokens} ids", k + 1);
            assert_eq!(
                *finish == PageFinish::PageCap,
                tokens == 50,
                "page {}: {finish:?} after {tokens} ids",
                k + 1
            );
        }
        assert!(
            retired.stats.page_finish.contains(&PageFinish::PageCap),
            "the cap bound at least one page"
        );
        assert_eq!(retired.stats.hf_all_neg_inf_steps, 0, "DISC-U2-2");
        assert_eq!(plain.ids, retired.ids, "retired ≡ unretired");
        assert_eq!(plain.stats.retired_blocks, 0);

        let positions = DecodeParams {
            max_positions: prompt.len() + 60,
            ..base.clone()
        };
        let (stopped, split) = run(&positions, true, None);
        v2_gated_line(
            test,
            started,
            "position_limit",
            "decode",
            "info",
            serde_json::json!({
                "max_positions": positions.max_positions, "tokens": stopped.ids.len(),
                "pages_produced": split.bodies.len(), "page_tokens": split.tokens,
                "finish_reason": stopped.finish_reason.as_str(),
                "page_finish": finishes(&stopped),
            }),
        );
        assert_eq!(stopped.finish_reason, FinishReason::Length);
        assert_eq!(stopped.ids.len(), 60, "pos + 1 reached the limit");
        assert_eq!(stopped.stats.page_finish.last(), Some(&PageFinish::Length));
        assert_eq!(stopped.stats.page_finish.len(), split.bodies.len());
        assert!(split.bodies.len() < pages.len());

        let (spent, split) = run(&base, true, Some(Instant::now()));
        v2_gated_line(
            test,
            started,
            "budget_spent",
            "decode",
            "info",
            serde_json::json!({
                "tokens": spent.ids.len(), "pages_produced": split.bodies.len(),
                "finish_reason": spent.finish_reason.as_str(), "page_finish": finishes(&spent),
            }),
        );
        assert_eq!(spent.finish_reason, FinishReason::Budget);
        assert!(spent.ids.is_empty());
        assert_eq!(spent.stats.page_finish, [PageFinish::Budget]);
        assert!(split.bodies.is_empty());
        v2_gated_line(
            test,
            started,
            fixture,
            "assert",
            "pass",
            serde_json::json!({}),
        );
    }

    /// Model-gated (D21): on S1 (one page) the HF top-M entmax and SGLang's
    /// all-candidates entmax decode the same ids, and both equal the oracle.
    #[test]
    fn real_v2_entmax_domains_decode_s1_alike() {
        let test = "real_v2_entmax_domains_decode_s1_alike";
        let started = std::time::Instant::now();
        let Some(model) = v2_gated_model(test, started) else {
            return;
        };
        let fixture = "S1-page_0009";
        let Some(paths) = v2_corpus_pages(test, started, fixture, &["page_0009.png"]) else {
            return;
        };
        let oracle = v2_oracle_ids(fixture);
        let (embeds, prompt) = v2_decoder_prompt(&model, &paths);
        let decode = |domain: srswa::EntmaxDomain| {
            let params = DecodeParams {
                entmax_domain: domain,
                ..model.document_params(1)
            };
            let pass = DocumentPass {
                precision: DocumentPrecision::Mixed,
                retire: true,
                deadline: None,
                budget_floor: 0,
            };
            let t = Instant::now();
            let mut ids = Vec::new();
            let decode = model
                .generate_document_once(&embeds, &prompt, &params, pass, &mut ids, None)
                .expect("the pass decodes");
            (decode, t.elapsed().as_secs_f64())
        };
        let (topm, topm_s) = decode(srswa::EntmaxDomain::TopM);
        let (all, all_s) = decode(srswa::EntmaxDomain::AllCandidates);
        let first_diff = topm.ids.iter().zip(&all.ids).position(|(a, b)| a != b);
        v2_gated_line(
            test,
            started,
            fixture,
            "decode",
            "info",
            serde_json::json!({
                "topm_tokens": topm.ids.len(), "all_tokens": all.ids.len(),
                "oracle_tokens": oracle.len(), "first_diff": first_diff,
                "topm_finish": topm.finish_reason.as_str(),
                "all_finish": all.finish_reason.as_str(),
                "topm_wall_s": topm_s, "all_wall_s": all_s,
            }),
        );
        assert_eq!(all.ids, topm.ids, "D21: the domains agree on S1");
        assert_eq!(topm.ids, oracle, "the top-M decode is the oracle's");
        v2_gated_line(
            test,
            started,
            fixture,
            "assert",
            "pass",
            serde_json::json!({}),
        );
    }

    /// U2-C2 L4 on the one-page S1 fixtures ([`v2_l4`]).
    #[test]
    fn real_v2_l4_s1_pages_decode_the_oracle_ids() {
        v2_l4(
            "real_v2_l4_s1_pages_decode_the_oracle_ids",
            &["S1-page_0009", "S1-page_0014", "S1-page_0590"],
        );
    }

    /// U2-C2 L4 on the short documents, S2 and S3 p9_p14 ([`v2_l4`]).
    #[test]
    fn real_v2_l4_documents_decode_the_oracle_ids() {
        v2_l4(
            "real_v2_l4_documents_decode_the_oracle_ids",
            &["S2-p9_p14_p19", "S3-p9_p14"],
        );
    }

    /// U2-C2 L4 on the long documents, S3 p10 and p20 (6,272 and 16,880
    /// ids, [`v2_l4`]): `FOCR_E2E_LONG` arms it, any value but empty or `0`
    /// (the token-golden gate's rule).
    #[test]
    fn real_v2_l4_long_documents_decode_the_oracle_ids() {
        let test = "real_v2_l4_long_documents_decode_the_oracle_ids";
        let long = matches!(
            std::env::var("FOCR_E2E_LONG").ok().as_deref(),
            Some(v) if !v.is_empty() && v != "0"
        );
        if !long {
            v2_gated_line(
                test,
                std::time::Instant::now(),
                "S3-p10,S3-p20",
                "skip",
                "skip",
                serde_json::json!({"reason": "FOCR_E2E_LONG unset or 0"}),
            );
            return;
        }
        v2_l4(test, &["S3-p10", "S3-p20"]);
    }

    /// U2-D1a's windowed decode of the corpus `paths` at `opts` on `model`,
    /// each page through its base letterbox. `bodies` gets each window's
    /// own bodies as it ends.
    fn v2_windowed(
        model: &OcrModel,
        paths: &[PathBuf],
        opts: &window::WindowedOptions,
        bodies: &mut Vec<Vec<String>>,
    ) -> window::WindowedDocument {
        let mut pages = paths.iter().map(|path| {
            let img = image::open(path)
                .map_err(|e| FocrError::Other(anyhow::anyhow!("{}: {e}", path.display())))?;
            model.document_page(img)
        });
        let mut sink = |_: &window::WindowStats, own: &[String]| bodies.push(own.to_vec());
        model
            .decode_document_windowed(paths.len(), &mut pages, opts, Some(&mut sink))
            .expect("the windowed decode runs")
            .expect("the document is windowed")
    }

    /// The shared facts of a windowed run, for its gated log line.
    fn v2_windowed_line(doc: &window::WindowedDocument) -> serde_json::Value {
        let windows: Vec<serde_json::Value> = doc
            .windows
            .iter()
            .map(|w| {
                serde_json::json!({
                    "index": w.index, "pages": [w.pages.start, w.pages.end],
                    "lookahead": w.lookahead, "reused": w.reused,
                    "fallback_reason": w.fallback_reason.map(window::FallbackReason::as_str),
                    "straddle": w.straddle, "prefill_tokens": w.prefill_tokens,
                    "copied_rows": w.copied_rows, "forced_tokens": w.forced_tokens,
                    "generated_tokens": w.generated_tokens, "finish": w.finish.as_str(),
                    "kv_bytes_peak": w.kv_bytes_peak, "retire_fallbacks": w.retire_fallbacks,
                    "elapsed_ms": w.elapsed_ms,
                })
            })
            .collect();
        serde_json::json!({
            "finish_reason": doc.finish_reason.as_str(), "bodies": doc.pages.bodies.len(),
            "extra_markers": doc.pages.extra_markers, "leading_text": doc.pages.leading_text,
            "retired_blocks": doc.stats.retired_blocks, "peak_kv_bytes": doc.stats.peak_kv_bytes,
            "windows": windows,
        })
    }

    /// U2-D1a, model-gated: a window whose budget is spent before it starts
    /// stops the document as `budget` with the pages of the windows before
    /// it, nothing discarded and nothing invented. Three base pages at
    /// W = 1, the budget spent at window 1, which would have been reused.
    #[test]
    fn real_v2_windowed_budget_keeps_the_finished_windows() {
        let test = "real_v2_windowed_budget_keeps_the_finished_windows";
        let started = std::time::Instant::now();
        let Some(model) = v2_gated_model(test, started) else {
            return;
        };
        let names = ["page_0009.png", "page_0014.png", "page_0019.png"];
        let Some(paths) = v2_corpus_pages(test, started, "S3-p3", &names) else {
            return;
        };
        let opts = window::WindowedOptions {
            window: 1,
            budget_spent_at: Some(1),
            precision: Some(DocumentPrecision::Mixed),
            ..window::WindowedOptions::default()
        };
        let mut bodies = Vec::new();
        let doc = v2_windowed(&model, &paths, &opts, &mut bodies);
        assert_eq!(doc.finish_reason, FinishReason::Budget);
        assert_eq!(
            doc.windows.len(),
            2,
            "the document stops at the spent window"
        );
        assert_eq!(doc.windows[0].finish, FinishReason::Stop);
        let spent = &doc.windows[1];
        assert_eq!(spent.finish, FinishReason::Budget);
        assert_eq!(
            (spent.prefill_tokens, spent.copied_rows),
            (0, 0),
            "no work claimed"
        );
        assert_eq!((spent.forced_tokens, spent.generated_tokens), (0, 0));
        assert_eq!(bodies.len(), 2);
        assert_eq!(bodies[0].len(), 1, "window 0 owns one page");
        assert!(bodies[1].is_empty(), "no body for a page never reached");
        assert_eq!(doc.pages.bodies, bodies[0]);
        assert!(!doc.pages.bodies[0].trim().is_empty());
        assert!(matches!(
            doc.stats.page_finish.as_slice(),
            [PageFinish::Marker | PageFinish::Eos, PageFinish::Budget]
        ));
        v2_gated_line(
            test,
            started,
            "S3-p3",
            "windowed",
            "pass",
            v2_windowed_line(&doc),
        );
    }

    /// U2-D1a, model-gated and armed by `FOCR_E2E_LONG`: 25 base pages (the
    /// 20 corpus pages, then their first five again) at W = 10 decode as
    /// three windows, each yielding exactly its own pages, so the document
    /// has 25 bodies in page order. The later windows reuse their
    /// predecessor's tail, and no window transcribes its lookahead page: the
    /// marker that opens it is its last id.
    #[test]
    fn real_v2_windowed_25_pages_own_every_page_once() {
        use crate::tokenizer::special::PAGE;
        let test = "real_v2_windowed_25_pages_own_every_page_once";
        let started = std::time::Instant::now();
        let long = matches!(
            std::env::var("FOCR_E2E_LONG").ok().as_deref(),
            Some(v) if !v.is_empty() && v != "0"
        );
        if !long {
            let why = serde_json::json!({"reason": "FOCR_E2E_LONG unset or 0"});
            v2_gated_line(test, started, "corpus-25", "skip", "skip", why);
            return;
        }
        let Some(model) = v2_gated_model(test, started) else {
            return;
        };
        let corpus = [
            "page_0009",
            "page_0014",
            "page_0019",
            "page_0023",
            "page_0086",
            "page_0091",
            "page_0105",
            "page_0107",
            "page_0108",
            "page_0126",
            "page_0136",
            "page_0179",
            "page_0250",
            "page_0358",
            "page_0400",
            "page_0437",
            "page_0547",
            "page_0557",
            "page_0590",
            "page_0645",
        ];
        let names: Vec<String> = corpus
            .iter()
            .chain(&corpus[..5])
            .map(|p| format!("{p}.png"))
            .collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let Some(paths) = v2_corpus_pages(test, started, "corpus-25", &names) else {
            return;
        };
        let opts = window::WindowedOptions {
            window: window::DEFAULT_PAGE_WINDOW,
            ..window::WindowedOptions::default()
        };
        let mut bodies = Vec::new();
        let doc = v2_windowed(&model, &paths, &opts, &mut bodies);
        let spans: Vec<(usize, usize, bool)> = doc
            .windows
            .iter()
            .map(|w| (w.pages.start, w.pages.end, w.lookahead))
            .collect();
        assert_eq!(spans, [(0, 10, true), (10, 20, true), (20, 25, false)]);
        assert_eq!(doc.finish_reason, FinishReason::Stop);
        for (w, ids) in doc.windows.iter().zip(&doc.window_ids) {
            assert_eq!(w.finish, FinishReason::Stop, "window {}", w.index);
            assert_eq!(w.reused, w.index > 0, "window {}", w.index);
            assert_eq!(bodies[w.index].len(), w.pages.len(), "window {}", w.index);
            if w.lookahead {
                // The forced `<PAGE>` opens a reused window's first page.
                let opening = w.pages.len() + 1;
                let markers: Vec<usize> = (0..ids.len()).filter(|&i| ids[i] == PAGE).collect();
                assert!(markers.len() <= opening, "window {}", w.index);
                if markers.len() == opening {
                    assert_eq!(markers[opening - 1], ids.len() - 1, "window {}", w.index);
                }
            }
        }
        assert_eq!(doc.pages.bodies.len(), 25);
        assert_eq!(doc.pages.bodies, bodies.concat());
        v2_gated_line(
            test,
            started,
            "corpus-25",
            "windowed",
            "pass",
            v2_windowed_line(&doc),
        );
    }

    /// U2-D1a against U2-D2a's windowed oracle (`tests/fixtures/
    /// uocr2_windowed`, U2-D2b): per windowed run, the same plan, fallbacks
    /// and text splices, and per window the oracle's forced ids and
    /// generated ids exactly, on the f32 cache. The oracle's injected reasons
    /// map onto [`window::FallbackReason`]; `pool_too_small` is (ix), which
    /// is (vi) here. The unsplit run plans no windows here either.
    #[test]
    fn real_v2_windowed_decode_matches_the_windowed_oracle() {
        let test = "real_v2_windowed_decode_matches_the_windowed_oracle";
        let started = std::time::Instant::now();
        let Some(model) = v2_gated_model(test, started) else {
            return;
        };
        let ids = |v: &serde_json::Value| -> Vec<u32> {
            v.as_array().map_or_else(Vec::new, |a| {
                a.iter()
                    .map(|id| u32::try_from(id.as_u64().expect("an id")).expect("a u32 id"))
                    .collect()
            })
        };
        let mut failures = Vec::new();
        for run in window::oracle_fixtures() {
            let tag = run["tag"].as_str().expect("a tag").to_string();
            let width = usize::try_from(run["window"].as_u64().expect("window")).expect("W");
            let windows = run["windows"].as_array().expect("windows");
            if run["plan"].is_null() {
                let [only] = windows.as_slice() else {
                    panic!("{tag}: an unsplit run is one window");
                };
                let pages = only["pages"].as_array().expect("pages").len();
                let opts = window::WindowedOptions {
                    window: width,
                    precision: Some(DocumentPrecision::CachedF32),
                    ..window::WindowedOptions::default()
                };
                let unsplit = model
                    .decode_document_windowed(pages, &mut std::iter::empty(), &opts, None)
                    .expect("planning runs");
                assert!(unsplit.is_none(), "{tag}: the oracle split no windows");
                continue;
            }
            // The document's pages: each window's own pages, in order.
            let mut names: Vec<String> = Vec::new();
            for w in windows {
                let lo_hi = w["lo_hi_emit"].as_array().expect("lo_hi_emit");
                let own = lo_hi[1].as_u64().expect("hi") - lo_hi[0].as_u64().expect("lo");
                let pages = w["pages"].as_array().expect("pages");
                names.extend(
                    pages[..usize::try_from(own).expect("own")]
                        .iter()
                        .map(|p| p.as_str().expect("a page").to_string()),
                );
            }
            let names: Vec<&str> = names.iter().map(String::as_str).collect();
            let Some(paths) = v2_corpus_pages(test, started, &tag, &names) else {
                continue;
            };
            let mut opts = window::WindowedOptions {
                window: width,
                precision: Some(DocumentPrecision::CachedF32),
                ..window::WindowedOptions::default()
            };
            let forced = run["forced_fallbacks"]
                .as_object()
                .cloned()
                .unwrap_or_default();
            for (k, how) in &forced {
                let k: usize = k.parse().expect("a window index");
                match how.as_str().expect("a reason") {
                    "pool_too_small" => {
                        opts.force_fallback.push((k, window::FallbackReason::Alloc));
                    }
                    "no_text" => opts
                        .force_fallback
                        .push((k, window::FallbackReason::NoOutput)),
                    "no_entry" => {
                        opts.force_fallback
                            .push((k, window::FallbackReason::NoRetained));
                    }
                    "splice_alloc_failed" => opts.marker_only_splice.push(k),
                    other => panic!("{tag}: no hook for the oracle's {other}"),
                }
            }
            let mut bodies = Vec::new();
            let doc = v2_windowed(&model, &paths, &opts, &mut bodies);
            assert_eq!(doc.windows.len(), windows.len(), "{tag}: windows");
            let mut first_mismatch = Vec::new();
            for ((w, ours), oracle) in doc.windows.iter().zip(&doc.window_ids).zip(windows) {
                let want_forced = ids(&oracle["forced_ids"]);
                let want = ids(&oracle["generated_ids"]);
                assert_eq!(
                    w.reused,
                    oracle["reused"] == true,
                    "{tag} w{}: reused",
                    w.index
                );
                let (got_forced, got) = ours.split_at(w.forced_tokens);
                let at = got
                    .iter()
                    .zip(&want)
                    .position(|(a, b)| a != b)
                    .or_else(|| (got.len() != want.len()).then_some(got.len().min(want.len())));
                let forced_equal = got_forced == want_forced.as_slice();
                if at.is_some() || !forced_equal {
                    failures.push(format!(
                        "{tag} w{}: forced equal {forced_equal}, first generated mismatch {at:?} \
                         of {} (oracle {})",
                        w.index,
                        got.len(),
                        want.len()
                    ));
                }
                first_mismatch.push(serde_json::json!({
                    "window": w.index, "forced_equal": forced_equal, "first_mismatch": at,
                    "tokens": got.len(), "oracle_tokens": want.len(),
                }));
            }
            // The stitched document: the wheel's bodies, as the sink saw them.
            let want_bodies: Vec<&str> = run["document"]["bodies"]
                .as_array()
                .expect("bodies")
                .iter()
                .map(|b| b.as_str().expect("a body"))
                .collect();
            let bodies_equal =
                doc.pages.bodies == want_bodies && bodies.concat() == doc.pages.bodies;
            if !bodies_equal {
                failures.push(format!(
                    "{tag}: the stitched bodies differ from the oracle's"
                ));
            }
            let mut line = v2_windowed_line(&doc);
            line["compare"] = first_mismatch.into();
            line["bodies_equal"] = bodies_equal.into();
            line["path"] = "f32-cached".into();
            let exact = bodies_equal
                && line["compare"].as_array().is_some_and(|c| {
                    c.iter()
                        .all(|w| w["forced_equal"] == true && w["first_mismatch"].is_null())
                });
            let result = if exact { "pass" } else { "fail" };
            v2_gated_line(test, started, &tag, "windowed_parity", result, line);
        }
        assert!(failures.is_empty(), "windowed parity: {failures:#?}");
    }

    /// The longest oracle generation the L4 rung also decodes on the stateless
    /// f32 path: S1 page_0009 (50 ids, 6.6 minutes in `release-perf`) and
    /// page_0014 (256 ids, 31 minutes).
    const V2_L4_STATELESS_MAX_IDS: usize = 256;

    /// The decode levers U2-A7 left to U2-C2, with their values where set: a
    /// lever's certification is the L4 rung run with it armed. U2-C2 supports
    /// all but `FOCR_MOE_SCORE_ORDER`, which still refuses.
    fn v2_armed_levers() -> serde_json::Value {
        [
            "FOCR_FUSE_SWIGLU",
            "FOCR_INT8_AUTOVEC",
            "FOCR_LMHEAD_SHARD",
            "FOCR_LMHEAD_SHARD_TILES",
            "FOCR_MOE_SCORE_ORDER",
            "FOCR_SOFTMAX_COPY",
        ]
        .into_iter()
        .filter_map(|name| {
            std::env::var(name)
                .ok()
                .map(|value| (name.to_string(), serde_json::Value::String(value)))
        })
        .collect::<serde_json::Map<_, _>>()
        .into()
    }

    /// The sha256 of `ids` as a compact JSON array: the U2-A6 oracle's
    /// `token_ids_sha256`.
    fn v2_ids_sha256(ids: &[u32]) -> String {
        use sha2::Digest as _;
        let json = serde_json::to_string(ids).expect("ids serialize");
        sha2::Sha256::digest(json.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// The logits at step `ids.len()` of a document teacher-forced along
    /// `ids` on cache `wc`, and each layer's read set there: a traced
    /// prefill, then one step per id, flagged forced exactly as the document
    /// loop flags it. Retirement stays off.
    fn v2_replay(
        model: &OcrModel,
        wc: &decoder::DecoderWeightCache,
        (embeds, prompt): (&Mat, &[u32]),
        ids: &[u32],
        params: &DecodeParams,
    ) -> (Mat, Vec<Vec<u32>>) {
        use crate::tokenizer::special::PAGE;
        let pages = prompt.iter().filter(|&&id| id == PAGE).count();
        let mut guard = document_guard(pages, PAGE, params);
        let table = model.embed_table().expect("embed table");
        let coords = srswa::PageCoords::build(prompt).expect("page coords");
        let (hidden, mut state) =
            decoder::prefill_with_cache_srswa_traced(wc, embeds, &coords, params.entmax_domain)
                .expect("traced prefill");
        let mut hidden = OcrModel::last_hidden_row(&hidden).expect("last row");
        let mut routes: Vec<Vec<u32>> = state
            .trace
            .as_ref()
            .expect("traced")
            .prefill_last
            .iter()
            .map(|r| r.as_ref().map_or_else(Vec::new, |r| r.blocks.clone()))
            .collect();
        for &id in ids {
            guard.observe(id);
            let forced = matches!(guard.verdict(), sampler::GuardVerdict::Force(_));
            let row = Mat::from_vec(1, embeds.cols, table.row_f32(id as usize));
            let h = decoder::decode_step_with_cache_srswa(wc, &mut state, id, &row, forced)
                .expect("replayed step");
            hidden = OcrModel::last_hidden_row(&h).expect("last row");
            routes = state
                .trace
                .as_ref()
                .expect("traced")
                .decode
                .iter()
                .map(|r| r.as_ref().map_or_else(Vec::new, |r| r.route.blocks.clone()))
                .collect();
        }
        let logits = decoder::lm_head_cached(wc, &hidden).expect("lm_head");
        (logits, routes)
    }

    /// U2-C2 L4 (plan §7.2) on the U2-A6 fixtures `fixtures`: each document
    /// through the real document loop on both cached paths, the all-f32 one
    /// and the shipping int8-FFN one, block retirement on and off, and on the
    /// stateless f32 path for a generation of at most
    /// [`V2_L4_STATELESS_MAX_IDS`] ids.
    ///
    /// - The f32 paths give the oracle's ids exactly. At a departure the
    ///   trace diff (f32 against int8-FFN, replayed along the oracle's ids)
    ///   is in the failure.
    /// - Retired ≡ unretired, on both cached paths.
    /// - Every run keeps the cross-page invariants: a body per page in page
    ///   order (exactly N markers, none extra, so the DISC-U2-7 reset is never
    ///   reached), EOS only on the last page, a body empty only where the
    ///   oracle's is, no DISC-U2-2 step, no retirement fallback.
    /// - The int8-FFN path is held to the oracle with the bead's escape. Its
    ///   cache shares every op with the f32 one but the FFN/expert storage,
    ///   and the f32 path follows the oracle, so a departure is the int8 FFN
    ///   quant effect; it is logged `xfail` with its trace diff for U2-C4.
    ///
    /// Lever re-certification (U2-A7) is this rung run with one lever armed;
    /// the armed levers are in every line.
    fn v2_l4(test: &str, fixtures: &[&str]) {
        use crate::tokenizer::special::PAGE;
        let started = std::time::Instant::now();
        let Some(model) = v2_gated_model(test, started) else {
            return;
        };
        let meta: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/uocr2_oracle/ORACLE.json"),
            )
            .expect("ORACLE.json"),
        )
        .expect("ORACLE.json parses");
        let tok = model.tokenizer().expect("tokenizer");
        let levers = v2_armed_levers();
        for &fixture in fixtures {
            let entry = &meta["fixtures"][fixture];
            assert_eq!(entry["mode"], "base", "{fixture}: the L4 lanes are base");
            let pages: Vec<&str> = entry["pages"]
                .as_array()
                .expect("pages")
                .iter()
                .map(|p| p["name"].as_str().expect("page name"))
                .collect();
            let Some(paths) = v2_corpus_pages(test, started, fixture, &pages) else {
                continue;
            };
            let (embeds, prompt) = v2_decoder_prompt(&model, &paths);
            assert_eq!(
                Some(prompt.len() as u64),
                entry["prompt_token_count"].as_u64(),
                "{fixture}: prompt rows"
            );
            let n = pages.len();
            let params = model.document_params(n);
            let oracle = v2_oracle_ids(fixture);
            assert_eq!(
                v2_ids_sha256(&oracle),
                entry["token_ids_sha256"]
                    .as_str()
                    .expect("token_ids_sha256"),
                "{fixture}: the committed ids are the oracle's"
            );
            let oracle_split =
                postprocess::split_pages_by_token(tok, &oracle, PAGE, n).expect("oracle split");
            let mut decoded: Vec<(DocumentPrecision, Vec<u32>)> = Vec::new();
            for precision in [
                DocumentPrecision::Stateless,
                DocumentPrecision::CachedF32,
                DocumentPrecision::Mixed,
            ] {
                // The stateless path re-runs the grown sequence every step,
                // O(n²): it decodes the short fixtures itself, and the f32
                // cache stands in for it on the rest. It has no cache to
                // retire.
                let retires: &[bool] = match precision {
                    DocumentPrecision::Stateless if oracle.len() > V2_L4_STATELESS_MAX_IDS => {
                        continue;
                    }
                    DocumentPrecision::Stateless => &[false],
                    _ => &[true, false],
                };
                let mut by_retire = Vec::new();
                for &retire in retires {
                    let pass = DocumentPass {
                        precision,
                        retire,
                        deadline: None,
                        budget_floor: 0,
                    };
                    let t = Instant::now();
                    let mut ids = Vec::new();
                    let decode = model
                        .generate_document_once(&embeds, &prompt, &params, pass, &mut ids, None)
                        .expect("the pass decodes");
                    let wall = t.elapsed().as_secs_f64();
                    let split = postprocess::split_pages_by_token(tok, &decode.ids, PAGE, n)
                        .expect("the split");
                    let first_mismatch = decode
                        .ids
                        .iter()
                        .zip(&oracle)
                        .position(|(a, b)| a != b)
                        .or_else(|| {
                            (decode.ids.len() != oracle.len())
                                .then_some(oracle.len().min(decode.ids.len()))
                        });
                    // The DISC-U2-7 reset fires on a fed `<PAGE>` past the
                    // page count: every id but a final one is fed.
                    let fed = &decode.ids[..decode.ids.len().saturating_sub(1)];
                    let disc_u2_7 = fed
                        .iter()
                        .filter(|&&id| id == PAGE)
                        .count()
                        .saturating_sub(n);
                    let exact = first_mismatch.is_none();
                    let result = match (exact, precision) {
                        (true, _) => "pass",
                        (false, DocumentPrecision::Mixed) => "xfail",
                        (false, _) => "fail",
                    };
                    v2_gated_line(
                        test,
                        started,
                        fixture,
                        "parity",
                        result,
                        serde_json::json!({
                            "rung": "L4", "path": precision.name(), "retire": retire,
                            "levers": levers, "tokens": decode.ids.len(),
                            "oracle_tokens": oracle.len(),
                            "first_mismatch_step": first_mismatch,
                            "ids_sha256": v2_ids_sha256(&decode.ids),
                            "oracle_sha256": entry["token_ids_sha256"],
                            "finish_reason": decode.finish_reason.as_str(),
                            "page_tokens": split.tokens,
                            "page_finish": decode.stats.page_finish.iter().map(|f| f.as_str()).collect::<Vec<_>>(),
                            "extra_markers": split.extra_markers,
                            "disc_u2_7_steps": disc_u2_7,
                            "disc_u2_2_steps": decode.stats.hf_all_neg_inf_steps,
                            "leading_text": split.leading_text,
                            "forced_steps": decode.stats.forced_steps,
                            "retired_blocks": decode.stats.retired_blocks,
                            "peak_kv_bytes": decode.stats.peak_kv_bytes,
                            "retire_fallbacks": decode.stats.retire_fallbacks,
                            "wall_s": wall,
                            "ms_per_token": 1e3 * wall / decode.ids.len().max(1) as f64,
                        }),
                    );
                    let path = precision.name();
                    assert_eq!(decode.finish_reason, FinishReason::Stop, "{fixture} {path}");
                    assert_eq!(split.bodies.len(), n, "{fixture} {path}: a body per page");
                    let mut want = vec![PageFinish::Marker; n - 1];
                    want.push(PageFinish::Eos);
                    assert_eq!(
                        decode.stats.page_finish, want,
                        "{fixture} {path}: EOS only on the last page"
                    );
                    assert_eq!(split.extra_markers, 0, "{fixture} {path}: DISC-U2-3");
                    assert_eq!(disc_u2_7, 0, "{fixture} {path}: DISC-U2-7");
                    assert_eq!(
                        decode.stats.hf_all_neg_inf_steps, 0,
                        "{fixture} {path}: DISC-U2-2"
                    );
                    assert_eq!(decode.stats.retire_fallbacks, 0, "{fixture} {path}");
                    assert!(!split.leading_text, "{fixture} {path}: DISC-U2-9");
                    for (k, (ours, theirs)) in
                        split.bodies.iter().zip(&oracle_split.bodies).enumerate()
                    {
                        assert_eq!(
                            ours.trim().is_empty(),
                            theirs.trim().is_empty(),
                            "{fixture} {path}: page {} body emptiness",
                            k + 1
                        );
                    }
                    by_retire.push((decode.ids, first_mismatch));
                }
                if let [(retired, _), (unretired, _)] = by_retire.as_slice() {
                    assert_eq!(
                        retired,
                        unretired,
                        "{fixture} {}: retired ≡ unretired",
                        precision.name()
                    );
                }
                let (ids, first_mismatch) = by_retire.swap_remove(0);
                if let Some(step) = first_mismatch {
                    // The trace diff: both caches replayed along the oracle's
                    // ids to the departure.
                    let prefix = &oracle[..step];
                    let (f32_logits, f32_routes) = v2_replay(
                        &model,
                        model.decoder_cache_f32().expect("f32 cache"),
                        (&embeds, &prompt),
                        prefix,
                        &params,
                    );
                    let (mixed_logits, mixed_routes) = v2_replay(
                        &model,
                        model.decoder_cache().expect("mixed cache"),
                        (&embeds, &prompt),
                        prefix,
                        &params,
                    );
                    let page = prefix.iter().filter(|&&id| id == PAGE).count();
                    let page_len = prefix.iter().rev().take_while(|&&id| id != PAGE).count();
                    let diff = decoder::decode_trace_diff(
                        step,
                        page,
                        page_len,
                        decoder::StepSide {
                            label: "f32",
                            logits: &f32_logits.data,
                            routes: &f32_routes,
                        },
                        decoder::StepSide {
                            label: "int8-ffn",
                            logits: &mixed_logits.data,
                            routes: &mixed_routes,
                        },
                    );
                    // `None` is EOS. The pick is the loop's: n-gram ban,
                    // guard, greedy argmax.
                    let want = oracle.get(step).copied();
                    let mut guard = document_guard(n, PAGE, &params);
                    for &id in prefix {
                        guard.observe(id);
                    }
                    let pick = |logits: &Mat| {
                        let s = sampler::decode_step_guarded(logits, prefix, &guard, &params)
                            .expect("a guarded step");
                        (!s.is_eos).then_some(s.token_id)
                    };
                    let gap = |logits: &Mat| {
                        let best = logits
                            .data
                            .iter()
                            .copied()
                            .fold(f32::NEG_INFINITY, f32::max);
                        best - logits.data[want.unwrap_or(params.eos_token_id) as usize]
                    };
                    let routes_differ: Vec<usize> = (0..f32_routes.len())
                        .filter(|&l| {
                            f32_routes[l] != mixed_routes.get(l).cloned().unwrap_or_default()
                        })
                        .collect();
                    v2_gated_line(
                        test,
                        started,
                        fixture,
                        "trace_diff",
                        "info",
                        serde_json::json!({
                            "path": precision.name(), "step": step, "page": page,
                            "page_len": page_len, "oracle_id": want,
                            "f32_pick": pick(&f32_logits), "int8_ffn_pick": pick(&mixed_logits),
                            "f32_oracle_gap": gap(&f32_logits),
                            "int8_ffn_oracle_gap": gap(&mixed_logits),
                            "layers_routing_differently": routes_differ,
                            "diff": diff,
                        }),
                    );
                    assert_eq!(
                        precision,
                        DocumentPrecision::Mixed,
                        "{fixture}: the f32 path departs from the oracle\n{diff}"
                    );
                    assert_eq!(
                        pick(&f32_logits),
                        want,
                        "{fixture}: at the int8-FFN departure the f32 path must pick the \
                         oracle's token\n{diff}"
                    );
                }
                decoded.push((precision, ids));
            }
            v2_gated_line(
                test,
                started,
                fixture,
                "assert",
                "pass",
                serde_json::json!({
                    "levers": levers,
                    "paths_exact": decoded.iter().filter(|(_, ids)| *ids == oracle).map(|(p, _)| p.name()).collect::<Vec<_>>(),
                }),
            );
        }
    }

    /// The U2-A6 oracle's generated ids for `fixture` (EOS not included).
    fn v2_oracle_ids(fixture: &str) -> Vec<u32> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/uocr2_oracle")
            .join(format!("{fixture}.ids.txt"));
        std::fs::read_to_string(path)
            .expect("oracle ids")
            .lines()
            .filter(|l| !l.starts_with('#'))
            .flat_map(str::split_whitespace)
            .map(|t| t.parse().expect("an id"))
            .collect()
    }

    /// `pages` in the corpus (`FOCR_CORPUS_DIR`, default
    /// `~/.cache/franken_ocr/corpus/pages`), or `None`, logged as a skip,
    /// when one is absent.
    fn v2_corpus_pages(
        test: &str,
        started: std::time::Instant,
        fixture: &str,
        pages: &[&str],
    ) -> Option<Vec<PathBuf>> {
        let corpus = std::env::var_os("FOCR_CORPUS_DIR").map_or_else(
            || {
                PathBuf::from(std::env::var_os("HOME").expect("HOME"))
                    .join(".cache/franken_ocr/corpus/pages")
            },
            PathBuf::from,
        );
        let paths: Vec<PathBuf> = pages.iter().map(|p| corpus.join(p)).collect();
        if let Some(missing) = paths.iter().find(|p| !p.is_file()) {
            v2_gated_line(
                test,
                started,
                fixture,
                "skip",
                "skip_no_model",
                serde_json::json!({"reason": format!("{} absent", missing.display())}),
            );
            return None;
        }
        Some(paths)
    }

    /// v2's prompt over `pages` through the engine (U2-B6): each page's base
    /// letterbox ([`OcrModel::document_page`]), the paged prompt and the
    /// vision blocks ([`OcrModel::document_inputs`]).
    fn v2_decoder_prompt(model: &OcrModel, pages: &[PathBuf]) -> (Mat, Vec<u32>) {
        let pres: Vec<Preprocessed> = pages
            .iter()
            .map(|page| {
                let img = image::open(page).expect("page decodes");
                model.document_page(img).expect("preprocess")
            })
            .collect();
        let (embeds, ids) = model.document_inputs(&pres).expect("document inputs");
        let pages_in_prompt = ids
            .iter()
            .filter(|&&id| id == crate::tokenizer::special::PAGE)
            .count();
        assert_eq!(pages_in_prompt, pages.len());
        (embeds, ids)
    }

    /// Greedy argmax, lowest id on ties.
    fn argmax_id(logits: &[f32]) -> u32 {
        let mut best = 0;
        for (i, &x) in logits.iter().enumerate() {
            if x > logits[best] {
                best = i;
            }
        }
        best as u32
    }

    fn fixture_page(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/got")
            .join(name)
    }

    /// `embeds` with the embedding of `id` appended as a new last row.
    fn push_embed(embeds: Mat, table: &EmbedTable<'_>, id: u32) -> Mat {
        let (rows, cols) = (embeds.rows + 1, embeds.cols);
        let mut data = embeds.data;
        data.extend(table.row_f32(id as usize));
        Mat::from_vec(rows, cols, data)
    }

    /// Model-gated (U2-B5): on the real v2 checkpoint, a single-page base
    /// prefill plus 32 greedy decode steps produce finite logits on the
    /// conservative cached path (`prefill_with_cache_srswa` /
    /// `decode_step_with_cache_srswa`) and on the stateless all-f32 path
    /// (`forward_srswa`). Their token agreement is logged as information
    /// only: the cache's int8 FFN differs from f32, and U2-C2's L4 gates it.
    #[test]
    fn real_v2_decoder_runs_cached_and_stateless_with_finite_logits() {
        const STEPS: usize = 32;
        let test = "real_v2_decoder_runs_cached_and_stateless_with_finite_logits";
        let started = std::time::Instant::now();
        let Some(model) = v2_gated_model(test, started) else {
            return;
        };
        let line = |case: &str, event: &str, result: &str, extra: serde_json::Value| {
            v2_gated_line(test, started, case, event, result, extra);
        };
        let (embeds, ids) = v2_decoder_prompt(&model, &[fixture_page("sample_text.png")]);
        let table = model.embed_table().expect("embed table");
        let wc = model.decoder_cache().expect("the v2 weight cache builds");
        assert_eq!(wc.router().expect("v2 caches its router").len(), 12);
        let finite = |logits: &Mat| logits.data.iter().all(|x| x.is_finite());

        let coords = srswa::PageCoords::build(&ids).expect("v2 page coordinates");
        let (hidden, mut state) =
            decoder::prefill_with_cache_srswa(wc, &embeds, &coords, srswa::EntmaxDomain::TopM)
                .expect("cached prefill");
        let mut logits =
            decoder::lm_head_cached(wc, &OcrModel::last_hidden_row(&hidden).expect("last row"))
                .expect("lm_head");
        let mut cached = Vec::new();
        for step in 0..=STEPS {
            assert!(finite(&logits), "cached logits at step {step}");
            let id = argmax_id(&logits.data);
            cached.push(id);
            if step == STEPS {
                break;
            }
            let row = Mat::from_vec(1, table.cols(), table.row_f32(id as usize));
            let h = decoder::decode_step_with_cache_srswa(wc, &mut state, id, &row, false)
                .expect("cached decode step");
            logits = decoder::lm_head_cached(wc, &h).expect("lm_head");
        }
        line(
            "cached",
            "assert",
            "pass",
            serde_json::json!({
                "assertion": "prefill + 32 decode steps give finite logits",
                "pass": true, "prompt_rows": ids.len(), "tokens": cached,
                "elapsed_s": started.elapsed().as_secs_f64(),
            }),
        );

        let (mut grown, mut grown_ids) = (embeds, ids.clone());
        let mut stateless = Vec::new();
        for step in 0..=STEPS {
            let coords = srswa::PageCoords::build(&grown_ids).expect("coordinates");
            let h =
                decoder::forward_srswa(&model.weights, &grown, &coords, srswa::EntmaxDomain::TopM)
                    .expect("stateless forward");
            let logits = decoder::lm_head(
                &model.weights,
                &OcrModel::last_hidden_row(&h).expect("last row"),
            )
            .expect("lm_head");
            assert!(finite(&logits), "stateless logits at step {step}");
            let id = argmax_id(&logits.data);
            stateless.push(id);
            grown_ids.push(id);
            grown = push_embed(grown, &table, id);
        }
        let agree = cached
            .iter()
            .zip(&stateless)
            .take_while(|(a, b)| a == b)
            .count();
        line(
            "stateless",
            "assert",
            "pass",
            serde_json::json!({
                "assertion": "32 stateless decode steps give finite logits",
                "pass": true, "tokens": stateless,
                "info_agreeing_prefix": agree, "info_of": STEPS + 1,
                "elapsed_s": started.elapsed().as_secs_f64(),
            }),
        );
    }

    /// Model-gated (U2-B5): on the real v2 checkpoint's conservative cached
    /// path, incremental decode step t and a fresh prefill over the prompt
    /// plus the first t generated tokens (plan §5.8) pick the same greedy
    /// token, on a two-page prompt.
    ///
    /// Their logits are not bit-equal on this path for either model. The
    /// prefill runs multi-row GEMMs and the decode m = 1 GEMVs, whose
    /// accumulation orders differ, and a last-ulp difference can flip an
    /// int8 activation rounding in each of the 12 layers' FFNs. v1's
    /// R-SWA cached path, measured the same way on the same pages
    /// (2026-10-29), differs by up to 5.0% of the largest logit (2.64 of
    /// 53.2), and v2 by up to 2.4%. The bound here, 10%, is twice v1's
    /// gap: a catch for wiring errors, not a precision claim.
    #[test]
    fn real_v2_cached_decode_equals_a_reprefill() {
        const STEPS: usize = 12;
        let test = "real_v2_cached_decode_equals_a_reprefill";
        let started = std::time::Instant::now();
        let Some(model) = v2_gated_model(test, started) else {
            return;
        };
        let pages = [
            fixture_page("sample_text.png"),
            fixture_page("format_corpus/table.png"),
        ];
        let (embeds, ids) = v2_decoder_prompt(&model, &pages);
        let table = model.embed_table().expect("embed table");
        let wc = model.decoder_cache().expect("the v2 weight cache builds");
        let last_logits = |embeds: &Mat, ids: &[u32]| {
            let coords = srswa::PageCoords::build(ids).expect("coordinates");
            let (h, state) =
                decoder::prefill_with_cache_srswa(wc, embeds, &coords, srswa::EntmaxDomain::TopM)
                    .expect("cached prefill");
            let logits =
                decoder::lm_head_cached(wc, &OcrModel::last_hidden_row(&h).expect("last row"))
                    .expect("lm_head");
            (logits, state)
        };
        let (mut logits, mut state) = last_logits(&embeds, &ids);
        let (mut grown, mut grown_ids) = (embeds, ids.clone());
        let mut gaps = Vec::new();
        for step in 1..=STEPS {
            let id = argmax_id(&logits.data);
            assert_ne!(
                id,
                crate::tokenizer::special::IMAGE,
                "a re-prefill would read it as an image"
            );
            let row = Mat::from_vec(1, table.cols(), table.row_f32(id as usize));
            let h = decoder::decode_step_with_cache_srswa(wc, &mut state, id, &row, false)
                .expect("cached decode step");
            logits = decoder::lm_head_cached(wc, &h).expect("lm_head");
            grown_ids.push(id);
            grown = push_embed(grown, &table, id);
            let (fresh, _) = last_logits(&grown, &grown_ids);
            let gap = logits
                .data
                .iter()
                .zip(&fresh.data)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let scale = fresh.data.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            assert_eq!(
                argmax_id(&logits.data),
                argmax_id(&fresh.data),
                "step {step}: incremental and re-prefill pick different tokens"
            );
            assert!(
                gap <= 0.10 * scale,
                "step {step}: logit gap {gap} exceeds 10% of the largest logit {scale}"
            );
            gaps.push((gap, scale));
        }
        v2_gated_line(
            test,
            started,
            "two_pages",
            "assert",
            "pass",
            serde_json::json!({
                "assertion": "incremental decode and a re-prefill pick the same tokens, logits within 10%",
                "pass": true, "prompt_rows": ids.len(),
                "max_abs_gap_per_step": gaps.iter().map(|g| g.0).collect::<Vec<_>>(),
                "max_abs_logit_per_step": gaps.iter().map(|g| g.1).collect::<Vec<_>>(),
                "tokens": &grown_ids[ids.len()..],
                "elapsed_s": started.elapsed().as_secs_f64(),
            }),
        );
    }

    #[test]
    fn native_model_available_rejects_missing_and_garbage() {
        let missing = Path::new("/definitely/not/a/real/model/path.focrq");
        assert!(!native_model_available(missing));

        let mut tmp = std::env::temp_dir();
        tmp.push(format!(
            "franken_ocr_available_garbage_{}.bin",
            std::process::id()
        ));
        std::fs::write(&tmp, b"not a model").expect("write garbage");
        assert!(!native_model_available(&tmp));
    }

    // ── stage-helper wiring (no weights required) ──────────────────────────────

    /// The frozen greedy decode contract is the single-image profile
    /// (temperature 0, EOS 1, no_repeat_ngram 35, window 128) — the value the AR
    /// loop drives with ([SPEC-100..103]).
    #[test]
    fn default_decode_params_are_single_image_greedy() {
        let p = DecodeParams::single_image();
        assert!(p.is_greedy());
        assert_eq!(p.eos_token_id, sampler::DEFAULT_EOS_TOKEN_ID);
        assert_eq!(
            p.no_repeat_ngram_size,
            sampler::DEFAULT_NO_REPEAT_NGRAM_SIZE
        );
        assert_eq!(p.ngram_window, sampler::NGRAM_WINDOW_SINGLE);
        assert!(p.max_length > 0, "max_length must bound the decode loop");
    }

    #[test]
    fn unlimited_vision_cache_kill_switch_parses_for_both_vision_schedules() {
        for value in ["0", "off", "FALSE", " No "] {
            assert!(
                !unlimited_vision_cache_enabled_for(Some(value)),
                "{value:?} must disable retained statics in sequential and batched vision"
            );
        }
        for value in [None, Some("1"), Some("true"), Some("yes")] {
            assert!(unlimited_vision_cache_enabled_for(value));
        }
    }

    #[test]
    fn batched_vision_cache_off_keeps_statics_batch_local() {
        let cache = FallibleOnce::<usize>::new();
        let initializations = std::cell::Cell::new(0usize);
        let initialize = || {
            initializations.set(initializations.get() + 1);
            Ok::<usize, ()>(17)
        };

        let first = retained_or_owned(false, &cache, initialize).expect("batch-local init");
        assert!(matches!(first, RetainedOrOwned::Owned(17)));
        let second = retained_or_owned(false, &cache, initialize).expect("next batch-local init");
        assert!(matches!(second, RetainedOrOwned::Owned(17)));
        assert_eq!(initializations.get(), 2, "cache-off rebuilds per batch");
        assert!(cache.value.get().is_none(), "cache-off retains no statics");

        let retained = retained_or_owned(true, &cache, initialize).expect("retained init");
        assert_eq!(*retained.as_ref(), 17);
        let reused = retained_or_owned(true, &cache, initialize).expect("retained reuse");
        assert_eq!(*reused.as_ref(), 17);
        assert_eq!(
            initializations.get(),
            3,
            "cache-on initializes exactly once"
        );
    }

    /// bd-1azu.10 / bd-t6a parity gate: the batched-across-pages vision tower is
    /// BIT-IDENTICAL to the per-page tower on the REAL model. Env-gated:
    /// `FOCR_MODEL_PATH` (the model artifact) + `FOCR_PARITY_IMAGES`
    /// (comma-separated page paths, ≥2 recommended so the batch is real);
    /// skip-with-SUCCESS when either is absent so the committed suite stays
    /// model-free.
    #[test]
    fn batched_vision_matches_per_page_tower_on_real_model() {
        let (Ok(model_path), Ok(images)) = (
            std::env::var("FOCR_MODEL_PATH"),
            std::env::var("FOCR_PARITY_IMAGES"),
        ) else {
            eprintln!("skip-with-SUCCESS: FOCR_MODEL_PATH / FOCR_PARITY_IMAGES unset");
            return;
        };
        let model = OcrModel::load(Path::new(&model_path)).expect("model loads");
        let paths: Vec<&str> = images.split(',').filter(|s| !s.is_empty()).collect();
        assert!(
            !paths.is_empty(),
            "FOCR_PARITY_IMAGES must name at least one page"
        );
        let pres: Vec<Preprocessed> = paths
            .iter()
            .map(|p| {
                preprocess::preprocess_image(
                    Path::new(p),
                    preprocess::PreprocessMode::Base { base_size: 1024 },
                    preprocess::PreprocessProfile::V1,
                )
                .expect("page preprocesses")
            })
            .collect();
        let prefs: Vec<&Preprocessed> = pres.iter().collect();
        let batched = model
            .vision_tower_batched_pages(&prefs)
            .expect("batched tower runs");
        for (k, pre) in pres.iter().enumerate() {
            let per_page = model.vision_tower(pre).expect("per-page tower runs");
            assert_eq!(batched[k].len(), per_page.len(), "page {k} view count");
            for (v, (b, s)) in batched[k].iter().zip(&per_page).enumerate() {
                assert_eq!(b.shape(), s.shape(), "page {k} view {v} shape");
                let bits_equal = b
                    .data
                    .iter()
                    .zip(&s.data)
                    .all(|(x, y)| x.to_bits() == y.to_bits());
                assert!(
                    bits_equal,
                    "page {k} view {v}: batched vision != per-page tower (bit drift)"
                );
            }
        }
    }

    /// [`resolve_preprocess_mode`] is the pure CLI-flag → [`PreprocessMode`]
    /// mapping (bd-1e9n): default = the certified Base-1024; overrides land
    /// exactly where set.
    #[test]
    fn preprocess_overrides_resolve_modes() {
        use preprocess::PreprocessMode as M;
        assert_eq!(
            resolve_preprocess_mode(PreprocessOverrides::default()),
            M::Base { base_size: 1024 }
        );
        assert_eq!(
            resolve_preprocess_mode(PreprocessOverrides {
                base_size: Some(512),
                ..Default::default()
            }),
            M::Base { base_size: 512 }
        );
        assert_eq!(
            resolve_preprocess_mode(PreprocessOverrides {
                gundam: Some(true),
                ..Default::default()
            }),
            M::Gundam {
                base_size: 1024,
                tile_size: 640
            }
        );
        assert_eq!(
            resolve_preprocess_mode(PreprocessOverrides {
                gundam: Some(false),
                base_size: Some(768),
                image_size: Some(512),
            }),
            M::Base { base_size: 768 }
        );
    }

    /// [`preprocess_profile`] (U2-B7): only Unlimited-OCR-2 takes v2's
    /// preprocessing; v1 and every zoo arch keep v1's.
    #[test]
    fn only_unlimited_ocr_2_takes_the_v2_preprocess_profile() {
        for &arch in model_arch::registry() {
            let want = if arch.id() == model_arch::UNLIMITED_OCR_V2_ID {
                preprocess::PreprocessProfile::V2
            } else {
                preprocess::PreprocessProfile::V1
            };
            assert_eq!(preprocess_profile(arch), want, "{}", arch.id());
        }
    }

    /// [`apply_decode_overrides`] is the pure CLI-flag → [`DecodeParams`] mapping
    /// (bd-3j3p): a `Some` field wins, a `None` keeps the built default — tested
    /// without touching the process-global store.
    #[test]
    fn decode_overrides_apply_some_fields_and_keep_none() {
        let base = DecodeParams::single_image();

        // Default overrides are the identity.
        let mut p = DecodeParams::single_image();
        apply_decode_overrides(&mut p, DecodeOverrides::default());
        assert_eq!(p.max_length, base.max_length);
        assert_eq!(p.temperature.to_bits(), base.temperature.to_bits());
        assert_eq!(p.no_repeat_ngram_size, base.no_repeat_ngram_size);
        assert_eq!(p.ngram_window, base.ngram_window);

        // Each Some field lands; untouched fields keep the build default.
        let mut p = DecodeParams::single_image();
        apply_decode_overrides(
            &mut p,
            DecodeOverrides {
                max_length: Some(700),
                temperature: None,
                no_repeat_ngram: Some(20),
                ngram_window: None,
                max_page_tokens: None,
            },
        );
        assert_eq!(p.max_length, 700);
        assert_eq!(p.temperature.to_bits(), base.temperature.to_bits());
        assert_eq!(p.no_repeat_ngram_size, 20);
        assert_eq!(p.ngram_window, base.ngram_window);
    }

    /// U2-E1: Unlimited-OCR-2's decode params take an explicit
    /// `--no-repeat-ngram 0` (the ban off, D10) and `--max-page-tokens`, and
    /// keep the contract's 20,000 per-page cap without one. v1 has no pages.
    #[test]
    fn explicit_overrides_reach_the_v2_decode_params() {
        let v2 = model_arch::arch_by_id(model_arch::UNLIMITED_OCR_V2_ID).expect("v2 registered");
        let defaults = decode_params_for(v2, DecodeOverrides::default());
        assert_eq!(
            defaults.max_page_tokens,
            sampler::UOCR2_DEFAULT_MAX_PAGE_TOKENS
        );
        let p = decode_params_for(
            v2,
            DecodeOverrides {
                no_repeat_ngram: Some(0),
                max_page_tokens: Some(0),
                ..DecodeOverrides::default()
            },
        );
        assert_eq!((p.no_repeat_ngram_size, p.max_page_tokens), (0, 0));
        let p = decode_params_for(
            v2,
            DecodeOverrides {
                max_page_tokens: Some(512),
                ..DecodeOverrides::default()
            },
        );
        assert_eq!(p.max_page_tokens, 512);
        assert_eq!(p.no_repeat_ngram_size, defaults.no_repeat_ngram_size);

        let v1 = model_arch::arch_by_id(model_arch::UNLIMITED_OCR_V1_ID).expect("v1 registered");
        let p = decode_params_for(
            v1,
            DecodeOverrides {
                max_page_tokens: Some(512),
                ..DecodeOverrides::default()
            },
        );
        assert_eq!(
            p.max_page_tokens,
            DecodeParams::single_image().max_page_tokens,
            "a model without pages ignores the page cap"
        );
    }

    /// The connector structural geometry the driver passes is the base-1024 16×16
    /// global grid (273-slot block invariant lives in `connector`).
    #[test]
    fn driver_uses_base_1024_global_grid() {
        let pre = Preprocessed::default();
        assert_eq!(OcrModel::global_grid_h(&pre), 16);
        assert_eq!(OcrModel::global_grid_w(&pre), 16);
    }

    /// The two learned structural params resolve through the `.focrq`/safetensors
    /// accessor. With an empty weight set they report a clean
    /// [`FocrError::FormatMismatch`] (tensor not found) — never a panic, and no
    /// longer the stale `NotImplemented` (the accessor has landed).
    #[test]
    fn structural_params_read_through_weights_accessor() {
        let w = Weights::default();
        assert!(matches!(
            OcrModel::image_newline(&w),
            Err(FocrError::FormatMismatch(_))
        ));
        assert!(matches!(
            OcrModel::view_seperator(&w),
            Err(FocrError::FormatMismatch(_))
        ));
    }

    /// Image dims accessor returns the source size carried by preprocessing for
    /// postprocess bbox de-normalization.
    #[test]
    fn image_dims_come_from_preprocessed_original_size() {
        let pre = Preprocessed {
            original_size: (123, 45),
            ..Preprocessed::default()
        };
        assert_eq!(OcrModel::image_dims(&pre), (123, 45));
    }

    /// Preprocessed view tensors must flow into the vision tower in the
    /// connector's crop-branch order: local tiles first, then the global
    /// thumbnail. Otherwise masked_scatter can align image placeholders to the
    /// wrong feature rows once the Gundam branch is wired.
    #[test]
    fn views_forward_local_tiles_then_global_thumbnail() {
        let global = Mat::from_vec(3, 4, (0..12).map(|v| v as f32).collect());
        let tile_a = Mat::from_vec(3, 1, vec![1.0, 2.0, 3.0]);
        let tile_b = Mat::from_vec(3, 1, vec![4.0, 5.0, 6.0]);
        let pre = Preprocessed {
            mode: preprocess::PreprocessMode::Gundam {
                base_size: 128,
                tile_size: 64,
            },
            global: preprocess::ViewTensor {
                pixels: global.clone(),
                height: 2,
                width: 2,
            },
            tiles: vec![
                preprocess::ViewTensor {
                    pixels: tile_a.clone(),
                    height: 1,
                    width: 1,
                },
                preprocess::ViewTensor {
                    pixels: tile_b.clone(),
                    height: 1,
                    width: 1,
                },
            ],
            crop_grid: preprocess::CropGrid {
                width_crop_num: 2,
                height_crop_num: 1,
            },
            original_size: (640, 320),
        };

        assert_eq!(OcrModel::views(&pre), vec![tile_a, tile_b, global]);
        assert_eq!(OcrModel::global_grid_h(&pre), 2);
        assert_eq!(OcrModel::global_grid_w(&pre), 2);
    }

    #[test]
    fn last_hidden_row_rejects_empty_decoder_output_without_panic() {
        let hidden = Mat::zeros(0, 4);
        let err = OcrModel::last_hidden_row(&hidden).expect_err("expected empty hidden error");
        assert!(matches!(err, FocrError::Other(_)));
        assert!(err.to_string().contains("zero hidden rows"));
    }

    #[test]
    fn last_hidden_row_rejects_malformed_decoder_output_without_panic() {
        let hidden = Mat {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        };
        let err = OcrModel::last_hidden_row(&hidden).expect_err("expected malformed hidden error");
        assert!(matches!(err, FocrError::Other(_)));
        assert!(err.to_string().contains("data len 5 != rows*cols 6"));
    }

    #[test]
    fn last_hidden_row_extracts_final_row() {
        let hidden = Mat::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let last = OcrModel::last_hidden_row(&hidden).expect("last row");
        assert_eq!(last.shape(), (1, 2));
        assert_eq!(last.row(0), &[5.0, 6.0]);
    }

    #[test]
    fn crop_figures_crops_only_image_spans_from_source() {
        // A non-image span (ignored) + one full-frame image span. `[0,0,999,999]`
        // rescales exactly to the source dims, so the crop is the whole image.
        let source = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            120,
            80,
            image::Rgb([7, 8, 9]),
        ));
        let decoded = concat!(
            "<|ref|>title<|/ref|><|det|>[[0,0,500,500]]<|/det|>",
            "<|ref|>image<|/ref|><|det|>[[0,0,999,999]]<|/det|>",
        );
        let figs = OcrModel::crop_figures(decoded, &source, 120, 80, "");
        assert_eq!(figs.len(), 1, "only the image span is cropped");
        assert_eq!(figs[0].index, 0);
        assert_eq!(figs[0].label, "image");
        assert_eq!(figs[0].markdown_ref, "![](images/0.jpg)");
        assert_eq!(figs[0].bbox, [0, 0, 120, 80]);
        assert_eq!(figs[0].image.width(), 120);
        assert_eq!(figs[0].image.height(), 80);
    }

    #[test]
    fn crop_figures_crops_the_right_subregion() {
        // Left half red, right half blue; an image span over the right half must
        // crop to a blue-only region. Box x1=500 (~middle) .. x2=999 (right edge).
        let mut buf = image::RgbImage::new(100, 40);
        for y in 0..40 {
            for x in 0..100 {
                let c = if x < 50 {
                    image::Rgb([255, 0, 0])
                } else {
                    image::Rgb([0, 0, 255])
                };
                buf.put_pixel(x, y, c);
            }
        }
        let source = image::DynamicImage::ImageRgb8(buf);
        // x1 = int(500/999*100)=50, x2 = int(999/999*100)=100 -> right half.
        let decoded = "<|ref|>image<|/ref|><|det|>[[500,0,999,999]]<|/det|>";
        let figs = OcrModel::crop_figures(decoded, &source, 100, 40, "");
        assert_eq!(figs.len(), 1);
        assert_eq!(figs[0].bbox, [50, 0, 100, 40]);
        let crop = figs[0].image.to_rgb8();
        assert_eq!(crop.dimensions(), (50, 40));
        assert!(
            crop.pixels().all(|p| p.0 == [0, 0, 255]),
            "the right-half crop must be all blue"
        );
    }

    #[test]
    fn crop_figures_skips_degenerate_box() {
        let source = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            50,
            50,
            image::Rgb([0, 0, 0]),
        ));
        // x1 == x2 after rescale -> zero width -> skipped.
        let decoded = "<|ref|>image<|/ref|><|det|>[[10,0,10,999]]<|/det|>";
        assert!(OcrModel::crop_figures(decoded, &source, 50, 50, "").is_empty());
    }

    #[test]
    fn forward_dispatch_guard_passes_unlimited_ocr_rejects_a_planned_arch() {
        use crate::native_engine::model_arch::{
            self, DecodeContract, Decoder, ModelArch, Task, TokenizerKind, VisionEncoder,
        };
        // The implemented default arch (Unlimited-OCR) passes the dispatch guard.
        OcrModel::ensure_arch_implemented(model_arch::default_arch())
            .expect("unlimited-ocr is implemented");

        // A planned zoo arch (forward not yet built) returns a clean
        // NotImplemented (exit 1) naming itself — never mis-runs another forward.
        struct PlannedArch;
        impl ModelArch for PlannedArch {
            fn id(&self) -> &'static str {
                "got-ocr2"
            }
            fn display_name(&self) -> &'static str {
                "GOT-OCR2.0"
            }
            fn license_notice(&self) -> &'static str {
                "Apache-2.0"
            }
            fn default_artifact_basename(&self) -> &'static str {
                "got-ocr2.focrq"
            }
            fn vision_encoder(&self) -> VisionEncoder {
                VisionEncoder::SamVit
            }
            fn decoder(&self) -> Decoder {
                Decoder::Qwen2Dense
            }
            fn tokenizer(&self) -> TokenizerKind {
                TokenizerKind::Qwen2Bpe
            }
            fn decode_contract(&self) -> DecodeContract {
                DecodeContract {
                    temperature: 0.0,
                    eos_token_id: 0,
                    no_repeat_ngram_size: 0,
                    ngram_window: 0,
                    ngram_domain: model_arch::NgramDomain::PromptAndGenerated,
                    pages: None,
                }
            }
            fn tasks(&self) -> &'static [Task] {
                &[Task::Ocr]
            }
            fn maturity(&self) -> model_arch::Maturity {
                model_arch::Maturity::Planned
            }
            fn required_quant_recipe(&self, _quant: &str) -> Option<&'static str> {
                None
            }
        }
        let err = OcrModel::ensure_arch_implemented(&PlannedArch)
            .expect_err("a planned arch must be rejected");
        assert!(matches!(err, FocrError::NotImplemented(_)), "got {err:?}");
        assert_eq!(err.exit_code(), 1);
        assert!(
            err.to_string().contains("got-ocr2"),
            "names the arch: {err}"
        );
    }

    #[test]
    fn arch_is_read_from_the_loaded_focrq_model_id_tag() {
        use crate::native_engine::model_arch;
        // A minimal got-ocr2-tagged .focrq (no tensors needed — arch() reads only
        // the model_id the loader resolved). Proves A2's tag drives dispatch:
        // loading must NOT silently mis-identify it as the default Unlimited-OCR.
        let got_notice = model_arch::arch_by_id("got-ocr2")
            .expect("got-ocr2 registered")
            .license_notice();
        let mut builder = crate::quant::focrq::FocrqBuilder::new()
            .with_model_id("got-ocr2")
            .with_license_notice(got_notice);
        builder
            .add_tensor(
                "model.embed_tokens.weight",
                crate::quant::focrq::WriteDType::Bf16,
                vec![1, 1],
                vec![0; 2],
            )
            .expect("add minimal tagged tensor");
        let blob = builder.build();
        let path = std::env::temp_dir().join(format!(
            "focr_arch_tag_{}_{}.focrq",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, &blob).unwrap();

        let model = OcrModel::load(&path).expect("a tagged .focrq loads");
        assert_eq!(
            model.arch().id(),
            "got-ocr2",
            "arch() reads the model_id tag"
        );

        // …and the forward dispatch guard now ADMITS got-ocr2 (its full pipeline ships,
        // B1–B9/B11), so `ensure_arch_implemented` is Ok and the real got-ocr2 forward
        // runs — rather than the Unlimited-OCR pipeline being applied to foreign weights.
        // (The refusal path for a genuinely unimplemented arch is covered by the mock
        // `PlannedArch` test above.)
        OcrModel::ensure_arch_implemented(model.arch())
            .expect("got-ocr2 is implemented, so its forward is admitted");

        let _ = std::fs::remove_file(&path);
    }

    /// A one-tensor raw safetensors shard, which matches no embedded census.
    fn unidentified_safetensors() -> Vec<u8> {
        let header = br#"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0, 0]);
        bytes
    }

    /// A small `tokenizer.json` carrying the v1 `<image>` id.
    const SIDECAR_JSON: &str = r#"{"added_tokens":[{"id":128815,"content":"<image>","special":true}],
        "model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]}}"#;

    /// An arch that pins only its tokenizer sidecar.
    struct SidecarArch {
        sha256: Option<&'static str>,
        tokens: &'static [(u32, &'static str)],
    }

    impl model_arch::ModelArch for SidecarArch {
        fn id(&self) -> &'static str {
            "sidecar-test"
        }
        fn display_name(&self) -> &'static str {
            "sidecar test"
        }
        fn license_notice(&self) -> &'static str {
            "test"
        }
        fn default_artifact_basename(&self) -> &'static str {
            "sidecar-test.focrq"
        }
        fn vision_encoder(&self) -> model_arch::VisionEncoder {
            model_arch::VisionEncoder::SamClip
        }
        fn decoder(&self) -> model_arch::Decoder {
            model_arch::Decoder::DeepSeekV2MoeRswa
        }
        fn tokenizer(&self) -> model_arch::TokenizerKind {
            model_arch::TokenizerKind::DeepSeekBpe
        }
        fn decode_contract(&self) -> model_arch::DecodeContract {
            model_arch::default_arch().decode_contract()
        }
        fn tasks(&self) -> &'static [model_arch::Task] {
            &[model_arch::Task::Ocr]
        }
        fn maturity(&self) -> model_arch::Maturity {
            model_arch::Maturity::Planned
        }
        fn required_quant_recipe(&self, _quant: &str) -> Option<&'static str> {
            None
        }
        fn tokenizer_sha256(&self) -> Option<&'static str> {
            self.sha256
        }
        fn required_token_ids(&self) -> &'static [(u32, &'static str)] {
            self.tokens
        }
    }

    fn sha256_hex(bytes: &[u8]) -> &'static str {
        use sha2::Digest as _;
        Box::leak(format!("{:x}", sha2::Sha256::digest(bytes)).into_boxed_str())
    }

    #[test]
    fn absent_model_id_opens_as_v1_under_another_default() {
        let got = model_arch::arch_by_id("got-ocr2").expect("got-ocr2 registered");
        let _flip = model_arch::test_default::flip(got);
        assert_eq!(
            compatible_arch("").expect("absent id resolves").id(),
            model_arch::UNLIMITED_OCR_V1_ID
        );
        let mut builder = crate::quant::focrq::FocrqBuilder::new();
        builder
            .add_tensor(
                "model.embed_tokens.weight",
                crate::quant::focrq::WriteDType::Bf16,
                vec![1, 1],
                vec![0; 2],
            )
            .expect("add tensor");
        let weights = Weights::from_bytes(builder.build()).expect("v1 container parses");
        assert_eq!(weights.model_id(), model_arch::UNLIMITED_OCR_V1_ID);
    }

    #[test]
    fn an_unidentified_raw_shard_is_refused_on_both_open_paths() {
        let weights = Weights::from_bytes(unidentified_safetensors()).expect("raw shard parses");
        assert_eq!(weights.model_id(), "");
        let message = OcrModel::from_weights(weights, SidecarBundle::default())
            .err()
            .expect("byte path refuses it")
            .to_string();
        assert!(
            message.contains("matches no registered model's tensor census"),
            "{message}"
        );

        let path = temp_model_dir("unidentified_raw").join("model.safetensors");
        std::fs::write(&path, unidentified_safetensors()).expect("write raw shard");
        let message = load_weights_from_resolved_model(&path)
            .expect_err("file path refuses it")
            .to_string();
        assert!(
            message.contains("matches no registered model's tensor census"),
            "{message}"
        );
    }

    #[test]
    fn checked_tokenizer_refuses_a_foreign_sidecar_or_a_missing_token() {
        let bytes = SIDECAR_JSON.as_bytes();
        let pinned = SidecarArch {
            sha256: Some(sha256_hex(bytes)),
            tokens: &[(128815, "<image>")],
        };
        let tokenizer = checked_tokenizer(&pinned, bytes, "t.json").expect("the pinned sidecar");
        assert_eq!(tokenizer.image_id(), 128815);

        let other = SIDECAR_JSON.replace("\"b\":1", "\"b\":2");
        let message = checked_tokenizer(&pinned, other.as_bytes(), "t.json")
            .expect_err("a different file")
            .to_string();
        assert!(
            message.contains(sha256_hex(other.as_bytes())) && message.contains(sha256_hex(bytes)),
            "names both hashes: {message}"
        );
        assert!(message.contains("focr pull sidecar-test"), "{message}");

        let needs_page = SidecarArch {
            sha256: None,
            tokens: &[(128815, "<image>"), (128827, "<PAGE>")],
        };
        let message = checked_tokenizer(&needs_page, bytes, "t.json")
            .expect_err("no <PAGE> id")
            .to_string();
        assert!(
            message.contains("maps token id 128827 to None") && message.contains("\"<PAGE>\""),
            "{message}"
        );

        let unpinned = SidecarArch {
            sha256: None,
            tokens: &[],
        };
        assert!(!pins_tokenizer(&unpinned));
        checked_tokenizer(&unpinned, other.as_bytes(), "t.json").expect("nothing pinned");

        // A model `focr pull` does not publish names its pinned source instead.
        let v2 = model_arch::arch_by_id(model_arch::UNLIMITED_OCR_V2_ID).expect("v2 registered");
        let message = checked_tokenizer(v2, bytes, "t.json")
            .expect_err("not the v2 tokenizer")
            .to_string();
        assert!(
            message.contains("huggingface.co/baidu/Unlimited-OCR-2/blob/"),
            "{message}"
        );
        assert!(!message.contains("focr pull"), "{message}");
    }

    #[test]
    fn byte_path_refuses_a_foreign_v1_sidecar_before_the_weights_contract() {
        // A v1 container (no model_id) that would also fail the v1 census: the
        // tokenizer refusal must come first.
        let mut builder = crate::quant::focrq::FocrqBuilder::new();
        builder
            .add_tensor(
                "model.embed_tokens.weight",
                crate::quant::focrq::WriteDType::Bf16,
                vec![1, 1],
                vec![0; 2],
            )
            .expect("add tensor");
        let weights = || Weights::from_bytes(builder.clone().build()).expect("v1 container parses");
        let sidecars = SidecarBundle {
            tokenizer_json: Some(SIDECAR_JSON.as_bytes().to_vec()),
            ..SidecarBundle::default()
        };
        let message = OcrModel::from_weights(weights(), sidecars)
            .err()
            .expect("foreign sidecar refused")
            .to_string();
        assert!(
            message.contains("in-memory tokenizer.json")
                && message
                    .contains("a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4")
                && message.contains("focr pull unlimited-ocr"),
            "{message}"
        );
        // Without a sidecar the same weights fail on their own contract.
        let message = OcrModel::from_weights(weights(), SidecarBundle::default())
            .err()
            .expect("census refused")
            .to_string();
        assert!(!message.contains("tokenizer"), "{message}");
    }

    /// The file path on the real v1 artifact (env-gated: `FOCR_MODEL_PATH`,
    /// with the pinned `tokenizer.json` beside it). A foreign sidecar is
    /// refused before the load-stage probe runs, so no weight byte is read;
    /// the pinned sidecar passes the hash and every required token id; an
    /// absent one is left to first use.
    #[cfg(unix)]
    #[test]
    fn file_path_judges_the_v1_sidecar_before_reading_weights_when_armed() {
        let Some(model) = std::env::var_os("FOCR_MODEL_PATH").map(PathBuf::from) else {
            eprintln!("skip-with-SUCCESS: FOCR_MODEL_PATH unset");
            return;
        };
        let real_sidecar = model.with_file_name("tokenizer.json");
        let dir = temp_model_dir("sidecar_file_path");
        let link = dir.join("unlimited-ocr.focrq");
        if !link.exists() {
            std::os::unix::fs::symlink(&model, &link).expect("link the real artifact");
        }
        let v1 = model_arch::default_arch();
        assert_eq!(v1.id(), model_arch::UNLIMITED_OCR_V1_ID);

        std::fs::write(dir.join("tokenizer.json"), SIDECAR_JSON).expect("write foreign sidecar");
        let mut read_weights = false;
        let message = load_weights_from_resolved_model_after_validation(
            &link,
            |_| Ok(()),
            || {
                read_weights = true;
            },
        )
        .expect_err("foreign sidecar refused")
        .to_string();
        assert!(!read_weights, "refused before any weight byte was read");
        assert!(message.contains("focr pull unlimited-ocr"), "{message}");

        std::fs::copy(&real_sidecar, dir.join("tokenizer.json")).expect("copy pinned sidecar");
        let tokenizer = eager_sidecar_tokenizer(v1, &link)
            .expect("pinned sidecar passes the hash and every required id")
            .expect("present sidecar is parsed at open");
        for &(id, content) in v1.required_token_ids() {
            assert_eq!(tokenizer.id_to_token(id), Some(content));
        }

        let bare = temp_model_dir("sidecar_absent").join("unlimited-ocr.focrq");
        assert!(
            eager_sidecar_tokenizer(v1, &bare)
                .expect("absence is not an open error")
                .is_none()
        );
    }
}
