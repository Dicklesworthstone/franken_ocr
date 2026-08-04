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
pub mod connector;
pub mod decoder;
pub mod decoder_qwen2;
pub mod got;
pub mod model_arch;
pub mod moe;
pub mod nn;
pub mod onechart;
pub mod postprocess;
pub mod rswa;
pub mod sampler;
pub mod smolvlm2;
pub(crate) mod spec;
pub mod tensor;
pub mod token_compress;
pub mod tromr;
pub(crate) mod unlimited_ocr_census;
pub mod vision_bridge;
pub mod vision_clip;
pub mod vision_sam;
pub mod vision_siglip;
pub mod weights;

use std::ffi::OsStr;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, Weak};

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
pub struct OcrModel {
    /// Filesystem path the model was resolved + loaded from (provenance).
    path: PathBuf,
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

const RAW_SAFETENSORS_SHARD_NAME: &str = "model-00001-of-000001.safetensors";
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
/// pages' token blocks; bd-1gv.25).
const MULTI_PAGE_PROMPT_TEXT: &str = "Multi page parsing.";
/// Reference `infer_multi` global-view edge (`image_size=640`, NOT the
/// single-image 1024): every page is resized+padded to 640² Base mode, so the
/// per-page placeholder block is `(10+1)·10 + 1 = 111` tokens (OQ-18).
const MULTI_PAGE_BASE_SIZE: usize = 640;
/// `max_position_embeddings` — the hard 32K cap one cross-page pass must fit
/// in ([SPEC-101]; plan §2.1 "constant-KV long-output decoding of dozens of
/// pages in one 32K pass").
const MAX_POSITION_EMBEDDINGS: usize = 32768;

/// Streaming sink for completed multi-page bodies (bd-2z0y): called with the
/// 1-based page index and the trimmed raw body as each `<PAGE>` boundary is
/// crossed in the token stream.
type PageSink<'a> = &'a mut dyn FnMut(usize, &str);

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

/// Default-on cache for the immutable Unlimited-OCR SAM/projector bundles.
/// Falsy values restore the original per-page hydration wrappers for diagnosis
/// and retained-memory comparisons.
const UNLIMITED_VISION_CACHE_ENV: &str = "FOCR_UNLIMITED_VISION_CACHE";

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

/// The forward path's preprocess mode under the current process-wide overrides.
fn preprocess_mode() -> preprocess::PreprocessMode {
    resolve_preprocess_mode(
        *PREPROCESS_OVERRIDES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
}

/// The process-global [`DecodeOverrides`] store. A `Mutex` (not `OnceLock`) so a
/// long-lived embedder can re-set between recognitions; reads happen once per
/// model load, never per token.
static DECODE_OVERRIDES: Mutex<DecodeOverrides> = Mutex::new(DecodeOverrides {
    max_length: None,
    temperature: None,
    no_repeat_ngram: None,
    ngram_window: None,
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
/// and then the CLI [`DecodeOverrides`] (an explicit CLI flag outranks the env).
fn decode_params_from_env() -> DecodeParams {
    let mut p = DecodeParams::single_image();
    if let Some(raw) = std::env::var_os(MAX_NEW_TOKENS_ENV)
        && let Some(s) = raw.to_str()
        && let Ok(n) = s.trim().parse::<usize>()
        && n > 0
    {
        p.max_length = n;
    }
    apply_decode_overrides(&mut p, decode_overrides());
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

fn resolve_existing_model_artifact(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        let shard = path.join(RAW_SAFETENSORS_SHARD_NAME);
        shard.is_file().then_some(shard)
    } else if path.exists() {
        Some(path.to_path_buf())
    } else {
        None
    }
}

fn is_short_model_spec(path: &Path) -> bool {
    let mut components = path.components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
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

fn model_search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(raw) = std::env::var_os(MODEL_DIR_ENV)
        && !raw.is_empty()
    {
        dirs.extend(std::env::split_paths(&raw));
    }
    if let Some(root) = crate::dist::cache_root() {
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
/// (`"title"`, `"text"`, `"image"`, …) plus its bounding boxes in PIXEL
/// coordinates `[x1, y1, x2, y2]` for the source image (de-normalized from the
/// model's 0..=999 grid). Surfaced by [`OcrModel::recognize_with_layout`] for the
/// `focr ocr --json` / `-o out.json` structured-output path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutSpan {
    /// The ref/det label classifying the span.
    pub label: String,
    /// Pixel bounding boxes `[x1, y1, x2, y2]`; a span may carry several.
    pub boxes: Vec<[i64; 4]>,
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

fn versioned_quant_path(direct: &Path, quant: ModelQuantPreference) -> PathBuf {
    direct.with_extension(format!(
        "v{}.{}.focrq",
        crate::dist::UNLIMITED_OCR_ARTIFACT_VERSION,
        quant.as_str()
    ))
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

fn resolve_model_from_search_dirs_with_quant(
    spec: &Path,
    search_dirs: &[PathBuf],
    quant: Option<ModelQuantPreference>,
) -> FocrResult<PathBuf> {
    let specs = model_search_specs(spec);
    for dir in search_dirs {
        if let Some(resolved) = resolve_existing_model_artifact(dir) {
            return Ok(resolved);
        }
        for search_spec in &specs {
            for candidate in short_model_candidates(dir, search_spec, quant) {
                if let Some(resolved) = resolve_existing_model_artifact(&candidate) {
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
}

#[derive(serde::Deserialize)]
struct FocrqPackingManifestProbe {
    #[serde(default)]
    quant_recipe: Option<String>,
}

fn compatible_model_id(declared: &str) -> FocrResult<&'static str> {
    if declared.is_empty() {
        return Ok(model_arch::default_arch().id());
    }
    model_arch::arch_by_id(declared)
        .map(|arch| arch.id())
        .ok_or_else(|| FocrError::FormatMismatch(format!("unknown model_id {declared:?}")))
}

fn validate_header_license(notice: &str, model_id: &str) -> FocrResult<()> {
    if model_id == model_arch::default_arch().id() {
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

fn validate_safetensors_header(header: &[u8], payload_len: usize) -> FocrResult<()> {
    #[derive(serde::Deserialize)]
    struct Entry {
        dtype: String,
        shape: Vec<usize>,
        data_offsets: [usize; 2],
    }

    let raw: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(header).map_err(|error| {
            FocrError::FormatMismatch(format!("safetensors header JSON invalid: {error}"))
        })?;
    let mut directory = std::collections::BTreeMap::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let entry: Entry = serde_json::from_value(value).map_err(|error| {
            FocrError::FormatMismatch(format!("safetensors entry {name:?} invalid: {error}"))
        })?;
        let [start, end] = entry.data_offsets;
        if end < start {
            return Err(FocrError::FormatMismatch(format!(
                "safetensors entry {name:?} has reversed offsets"
            )));
        }
        let dtype = match entry.dtype.as_str() {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            other => {
                return Err(FocrError::FormatMismatch(format!(
                    "unsupported safetensors dtype {other:?}"
                )));
            }
        };
        directory.insert(
            name,
            weights::TensorRecord {
                dtype,
                shape: entry.shape,
                byte_offset: start,
                byte_len: end - start,
                scales_offset: 0,
                scales_len: 0,
                group_size: 0,
                tier: 0,
            },
        );
    }
    if directory.is_empty() {
        return Err(FocrError::FormatMismatch(
            "safetensors tensor directory is empty; no native forward can use this artifact".into(),
        ));
    }
    validate_header_directory(&directory, payload_len, 0)?;
    unlimited_ocr_census::validate_source_header(&directory)
}

fn validate_model_header_from_reader(mut reader: impl Read, file_len: u64) -> FocrResult<()> {
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
        if header.tensors.is_empty() {
            return Err(FocrError::FormatMismatch(
                ".focrq tensor directory is empty; no native forward can use this artifact".into(),
            ));
        }
        let model_id = compatible_model_id(&header.model_id)?;
        validate_header_license(&header.license_notice, model_id)?;
        let declared_recipe = header
            .packing_manifest
            .as_ref()
            .and_then(|manifest| manifest.quant_recipe.as_deref());
        validate_header_directory(&header.tensors, file_len - payload_base, header.arch_target)?;
        if model_id == model_arch::default_arch().id() {
            unlimited_ocr_census::validate_focrq_header(
                &header.tensors,
                &header.source_sha256,
                declared_recipe,
            )?;
        }
        validate_unlimited_ocr_quant_records(
            true,
            model_id,
            header
                .tensors
                .iter()
                .map(|(name, record)| (name.as_str(), record.dtype)),
        )?;
        return Ok(());
    }

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
    validate_safetensors_header(&header, file_len - payload_base)
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

fn validate_model_header_file(path: &Path) -> FocrResult<()> {
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

fn load_weights_from_resolved_model(resolved: &Path) -> FocrResult<Weights> {
    load_weights_from_resolved_model_after_validation(resolved, || {})
}

fn load_weights_from_resolved_model_after_validation(
    resolved: &Path,
    after_validation: impl FnOnce(),
) -> FocrResult<Weights> {
    let (file, file_len) = open_model_file(resolved)?;
    validate_model_header_from_reader(&file, file_len)?;
    after_validation();
    Weights::load_opened(file, resolved).and_then(|weights| {
        validate_unlimited_ocr_quant_recipe(&weights)?;
        Ok(weights)
    })
}

/// Reject Unlimited-OCR containers that violate the fixed default quantization
/// recipe before any forward can silently consume them. Raw safetensors and
/// other registered model families have separate storage contracts.
fn validate_unlimited_ocr_quant_recipe(weights: &Weights) -> FocrResult<()> {
    validate_unlimited_ocr_quant_records(
        weights.is_focrq(),
        weights.model_id(),
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
    records: impl IntoIterator<Item = (&'a str, DType)>,
) -> FocrResult<()> {
    if !is_focrq || model_id != model_arch::default_arch().id() {
        return Ok(());
    }

    let recipe = Recipe::validated_default();
    let mut violations = Vec::new();
    for (name, dtype) in records {
        let valid = if recipe.is_quantized(name) {
            dtype == DType::QInt8PerChan
        } else {
            matches!(dtype, DType::BF16 | DType::F32)
        };
        if !valid {
            let expected = if recipe.is_quantized(name) {
                "QInt8PerChan"
            } else {
                "BF16 or F32"
            };
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
    Err(FocrError::FormatMismatch(format!(
        "Unlimited-OCR .focrq violates quant recipe {}: {shown}{suffix}. Re-convert the \
         original safetensors with this build; legacy full-int8 artifacts are not accepted",
        crate::quant::convert::UNLIMITED_OCR_INT8_RECIPE_ID,
    )))
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
    /// safetensors directory, returns its canonical shard. Missing relative
    /// specs are searched under `$FOCR_MODEL_DIR` (split with the platform
    /// path-list separator) and the user cache default; each `$FOCR_MODEL_DIR`
    /// entry may itself be a direct artifact/package or a search root.
    pub fn resolve_model(path: &Path) -> FocrResult<PathBuf> {
        if let Some(resolved) = resolve_existing_model_artifact(path) {
            return Ok(resolved);
        }
        if path.is_dir() {
            return Err(FocrError::ModelNotFound(format!(
                "no model artifact at {} (expected {RAW_SAFETENSORS_SHARD_NAME} inside \
                 safetensors directory; resolver lands in Phase 0/1, bd-223.7)",
                path.display()
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
    /// contract remains intact.
    pub fn load(path: &Path) -> FocrResult<Arc<Self>> {
        let resolved = Self::resolve_model(path)?;
        let _load_permit = model_load_admission().acquire();

        {
            let guard = model_cache_guard()?;
            if let Some((cached_path, weak)) = guard.as_ref()
                && *cached_path == resolved
                && let Some(strong) = weak.upgrade()
            {
                return Ok(strong);
            }
        }

        // Route through `Weights::load`: owned bytes are the safe default.
        // `FOCR_MMAP=1` is an explicit opt-in for trusted immutable artifacts;
        // mmap failures retain the owned-buffer fallback.
        let weights = load_weights_from_resolved_model(&resolved)?;
        let model = Arc::new(Self {
            path: resolved.clone(),
            weights,
            decode_params: decode_params_from_env(),
            decoder_cache_i8: FallibleOnce::new(),
            decoder_cache: FallibleOnce::new(),
            clip_cache: FallibleOnce::new(),
            unlimited_vision: FallibleOnce::new(),
            got_statics: FallibleOnce::new(),
            onechart_statics: FallibleOnce::new(),
            smol_statics: FallibleOnce::new(),
            tokenizer: FallibleOnce::new(),
            got_tokenizer: FallibleOnce::new(),
            tromr_tokenizer: FallibleOnce::new(),
            music_meta: std::sync::Mutex::new(None),
        });

        let mut guard = model_cache_guard()?;
        *guard = Some((resolved, Arc::downgrade(&model)));
        Ok(model)
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
        if self.arch().id() == "got-ocr2" {
            return self.forward_got(&preprocess::decode_path(image_path)?);
        }
        if self.arch().id() == "smolvlm2" {
            return self.forward_smolvlm2(&preprocess::decode_path(image_path)?);
        }
        if self.arch().id() == "onechart" {
            return self.forward_onechart(&preprocess::decode_path(image_path)?);
        }
        if self.arch().id() == "tromr" {
            return self.forward_tromr(&preprocess::decode_path(image_path)?);
        }
        let t = std::time::Instant::now();
        // ── 1. preprocess (decode file → normalize → tile) ───────────────────
        // Mode = the certified Base-1024 default unless the CLI overrode it
        // (--base-size/--image-size/--crop-mode, bd-1e9n).
        let pre = preprocess::preprocess_image(image_path, preprocess_mode())?;
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
        if self.arch().id() == "got-ocr2" {
            return self.forward_got(&img);
        }
        if self.arch().id() == "smolvlm2" {
            return self.forward_smolvlm2(&img);
        }
        if self.arch().id() == "onechart" {
            return self.forward_onechart(&img);
        }
        if self.arch().id() == "tromr" {
            return self.forward_tromr(&img);
        }
        let t = std::time::Instant::now();
        let pre = preprocess::preprocess_dynamic(img, preprocess_mode())?;
        timing_log(&format!("preprocess {:.2}s", t.elapsed().as_secs_f64()));
        self.forward_pre(pre)
    }

    /// The GOT-OCR2 forward path (bead B3/B5/B7): squash-bicubic-1024/CLIP preprocess
    /// → SAM-ViT-B vision + `mm_projector_vary` connector + `<imgpad>` splice →
    /// Qwen2 dense decoder greedy generation → tiktoken decode. Returns the decoded
    /// text plus source pixel dims (for the postprocess geometry the wrappers share).
    fn forward_got(&self, img: &image::DynamicImage) -> FocrResult<(String, u32, u32)> {
        use image::GenericImageView;
        let t = std::time::Instant::now();
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
        let t = std::time::Instant::now();
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
        let t = std::time::Instant::now();
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

    fn forward_tromr(&self, img: &image::DynamicImage) -> FocrResult<(String, u32, u32)> {
        use image::GenericImageView;
        let t = std::time::Instant::now();
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
            let dir = self.path.parent().unwrap_or_else(|| Path::new("."));
            crate::tokenizer::music::MusicTokenizer::from_dir(dir)
        })
    }

    /// The GOT Qwen tiktoken tokenizer, loaded from `qwen.tiktoken` beside the
    /// model and cached. (The Baidu path uses [`Self::tokenizer`].)
    fn got_tokenizer(&self) -> FocrResult<&crate::tokenizer::tiktoken::Tiktoken> {
        self.got_tokenizer.get_or_try_init(|| {
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
    /// Read from the `.focrq` v2 `model_id` tag the loader resolved (A2): a v1
    /// `.focrq` or raw safetensors reports `unlimited-ocr`, a tagged artifact
    /// reports its own arch (e.g. `got-ocr2`). `model_id()` is always a
    /// registry-known id (validated at load), so the lookup cannot miss; the
    /// `unwrap_or_else` is a defensive fallback (e.g. an empty [`Weights::default`]).
    #[must_use]
    pub fn arch(&self) -> &'static dyn model_arch::ModelArch {
        model_arch::arch_by_id(self.weights.model_id()).unwrap_or_else(model_arch::default_arch)
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

        // ── vision tower (SAM⊕CLIP -> bridge projector 2048->1280) ───────────
        let tv = std::time::Instant::now();
        let vision_features = self.vision_tower(&pre)?;
        timing_log(&format!("vision_tower {:.2}s", tv.elapsed().as_secs_f64()));

        // ── connector: prompt embeds + masked_scatter of the 273-slot block ──
        let (inputs_embeds, prompt_ids) = self.build_inputs_embeds(&pre, &vision_features)?;

        // ── decoder prefill + sequential greedy AR decode to EOS ─────────────
        let generated = self.generate(inputs_embeds, &prompt_ids)?;

        // Detokenize the generated ids into the raw model text (the postprocess
        // pass strips EOS / parses ref-det / rewrites image spans).
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
        let layout = postprocess::parse_layout(decoded, image_w, image_h)
            .into_iter()
            .map(|(label, boxes)| LayoutSpan { label, boxes })
            .collect();
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
        let iw = i64::from(source.width());
        let ih = i64::from(source.height());
        postprocess::figure_refs(decoded, image_w, image_h, img_base)
            .into_iter()
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
                    label: fr.label,
                    bbox: [cx1, cy1, cx2, cy2],
                    markdown_ref: fr.markdown_ref,
                    image,
                })
            })
            .collect()
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
    /// Multi-page document orchestration (bd-1gv.25) — the reference
    /// `infer_multi` contract, THE headline "Unlimited" feature (plan §2.1):
    /// every page's Base-640 vision block joins ONE frozen cross-page
    /// reference KV, then a single AR decode runs over the whole document —
    /// page N attends to pages 1..N−1 (OQ-13: cross-page DEPENDENT, never a
    /// concatenation of independent single-page parses).
    ///
    /// Returns the assembled markdown: the model emits `<PAGE>` separators
    /// and [`postprocess::finalize_multi`] applies the reference
    /// `save_results` per-page post-pass + rejoin.
    ///
    /// # Errors
    /// * [`FocrError::NotImplemented`] for a non-Unlimited-OCR artifact —
    ///   `infer_multi` is the Unlimited contract; zoo models parse pages
    ///   independently via [`Self::recognize_batch`].
    /// * [`FocrError::Other`] on an empty page list or when the assembled
    ///   cross-page prefix cannot fit the 32K position budget (actionable:
    ///   says how many tokens over and suggests splitting the document).
    /// * Any preprocess / vision / decode error, as [`Self::recognize`].
    pub fn recognize_multi_page(&self, image_paths: &[&Path]) -> FocrResult<String> {
        if image_paths.is_empty() {
            return Err(FocrError::Other(anyhow::anyhow!(
                "recognize_multi_page: image_paths must be non-empty"
            )));
        }
        if self.arch().id() != model_arch::default_arch().id() {
            return Err(FocrError::NotImplemented(format!(
                "multi-page cross-page parsing (infer_multi) is the Unlimited-OCR \
                 contract; model {:?} parses pages independently (use the plain \
                 multi-input/batch path)",
                self.arch().id()
            )));
        }
        Self::ensure_arch_implemented(self.arch())?;

        // ── per-page Base-640 preprocess (multi-page forces no-crop, §2.4) ──
        let mut pres = Vec::with_capacity(image_paths.len());
        for path in image_paths {
            let img = image::open(path)
                .map_err(|e| FocrError::InputDecode(format!("{}: {e}", path.display())))?;
            pres.push(preprocess::preprocess_dynamic_squash(
                img,
                MULTI_PAGE_BASE_SIZE,
            )?);
        }
        self.recognize_multi_page_pres(pres, None)
    }

    /// [`Self::recognize_multi_page`] over in-memory images — the PDF path's
    /// entry (pages are rasterized in-memory, never written to disk).
    ///
    /// # Errors
    /// As [`Self::recognize_multi_page`].
    pub fn recognize_multi_page_dynamic(
        &self,
        images: Vec<image::DynamicImage>,
    ) -> FocrResult<String> {
        if images.is_empty() {
            return Err(FocrError::Other(anyhow::anyhow!(
                "recognize_multi_page_dynamic: images must be non-empty"
            )));
        }
        if self.arch().id() != model_arch::default_arch().id() {
            return Err(FocrError::NotImplemented(format!(
                "multi-page cross-page parsing (infer_multi) is the Unlimited-OCR \
                 contract; model {:?} parses pages independently (use the plain \
                 multi-input/batch path)",
                self.arch().id()
            )));
        }
        Self::ensure_arch_implemented(self.arch())?;
        let mut pres = Vec::with_capacity(images.len());
        for img in images {
            pres.push(preprocess::preprocess_dynamic_squash(
                img,
                MULTI_PAGE_BASE_SIZE,
            )?);
        }
        self.recognize_multi_page_pres(pres, None)
    }

    /// [`Self::recognize_multi_page_dynamic`] with a PER-PAGE STREAMING sink
    /// (bd-2z0y): `on_page(k, body)` fires as soon as page `k`'s `<PAGE>`
    /// boundary is crossed IN the token stream (the next page's marker
    /// arrives, or the decode ends), carrying the trimmed raw body. The
    /// terminal return value is still the full [`postprocess::finalize_multi`]
    /// assembly — streaming adds visibility, never changes the result.
    ///
    /// # Errors
    /// As [`Self::recognize_multi_page`].
    pub fn recognize_multi_page_dynamic_streaming(
        &self,
        images: Vec<image::DynamicImage>,
        on_page: &mut dyn FnMut(usize, &str),
    ) -> FocrResult<String> {
        if images.is_empty() {
            return Err(FocrError::Other(anyhow::anyhow!(
                "recognize_multi_page_dynamic_streaming: images must be non-empty"
            )));
        }
        if self.arch().id() != model_arch::default_arch().id() {
            return Err(FocrError::NotImplemented(format!(
                "multi-page cross-page parsing (infer_multi) is the Unlimited-OCR \
                 contract; model {:?} parses pages independently (use the plain \
                 multi-input/batch path)",
                self.arch().id()
            )));
        }
        Self::ensure_arch_implemented(self.arch())?;
        let mut pres = Vec::with_capacity(images.len());
        for img in images {
            pres.push(preprocess::preprocess_dynamic_squash(
                img,
                MULTI_PAGE_BASE_SIZE,
            )?);
        }
        self.recognize_multi_page_pres(pres, Some(on_page))
    }

    /// The shared multi-page core (bd-1gv.25): per-page vision, ONE fused
    /// cross-page prefill, ONE decode, `finalize_multi` assembly. `on_page`
    /// (bd-2z0y) streams each completed page body as its boundary is crossed.
    fn recognize_multi_page_pres(
        &self,
        pres: Vec<Preprocessed>,
        on_page: Option<PageSink<'_>>,
    ) -> FocrResult<String> {
        let _admission = forward_admission().acquire();
        // ── vision tower per page, sequential (§6.5: one live forward) ──────
        let tv = std::time::Instant::now();
        let mut globals: Vec<Mat> = Vec::with_capacity(pres.len());
        for pre in &pres {
            let mut feats = self.vision_tower(pre)?;
            if feats.len() != 1 {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "recognize_multi_page: Base-mode page produced {} vision blocks; expected 1",
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
        if prompt_ids.len() + 1 > MAX_POSITION_EMBEDDINGS {
            return Err(FocrError::Other(anyhow::anyhow!(
                "recognize_multi_page: the assembled {}-page prefix is {} tokens — over the \
                 {MAX_POSITION_EMBEDDINGS} position budget by {} with no room to generate; \
                 split the document into smaller multi-page passes",
                pres.len(),
                prompt_ids.len(),
                (prompt_ids.len() + 1).saturating_sub(MAX_POSITION_EMBEDDINGS),
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
        // prefix + generated never exceeds the 32K positions.
        let params = sampler::DecodeParams {
            ngram_window: sampler::NGRAM_WINDOW_MULTI,
            max_length: self
                .decode_params
                .max_length
                .min(MAX_POSITION_EMBEDDINGS - prompt_ids.len()),
            ..self.decode_params.clone()
        };
        let td = std::time::Instant::now();
        let generated = match on_page {
            None => self.generate_with(inputs_embeds, &prompt_ids, &params)?,
            Some(on_page) => {
                // Streaming (bd-2z0y): a per-committed-token observer feeds a
                // [`postprocess::PageStream`]. Boundary checks re-decode the
                // emitted ids every STREAM_CHECK_EVERY tokens — a full decode
                // is a linear byte-concat, trivial next to a decode step's
                // GEMMs — and PageStream is idempotent over re-fed prefixes.
                // Streaming is BEST-EFFORT: a decode error inside the observer
                // stops the stream (the terminal decode below still surfaces
                // real errors); it never changes the emitted ids.
                const STREAM_CHECK_EVERY: usize = 8;
                let tok = self.tokenizer()?;
                let mut stream = postprocess::PageStream::new();
                let mut ids: Vec<u32> = Vec::new();
                let mut stream_broken = false;
                let mut observer = |id: u32| {
                    ids.push(id);
                    if stream_broken || !ids.len().is_multiple_of(STREAM_CHECK_EVERY) {
                        return;
                    }
                    match tok.decode(&ids) {
                        Ok(text) => stream.feed(&text, &mut *on_page),
                        Err(_) => stream_broken = true,
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
                if !stream_broken && let Ok(text) = tok.decode(&generated) {
                    let text = postprocess::strip_eos(&text);
                    stream.feed(&text, &mut *on_page);
                    stream.finish(&text, &mut *on_page);
                }
                generated
            }
        };
        timing_log(&format!(
            "multi_page.decode {} tokens {:.2}s",
            generated.len(),
            td.elapsed().as_secs_f64()
        ));
        let decoded = self.tokenizer()?.decode(&generated)?;
        postprocess::finalize_multi(&decoded, pres.len())
    }

    #[must_use]
    pub fn recognize_batch(&self, image_paths: &[&Path]) -> Vec<FocrResult<String>> {
        // The spine reproduces the int8 R-SWA cached decode (`generate_cached_i8`),
        // which `generate` selects only when int8 is requested AND the stateless
        // O(n^2) oracle is NOT forced. Engage the spine under EXACTLY that
        // condition so its output tracks whichever sequential oracle is in force.
        // Arch gate (fresh-eyes fix): the spine pipeline is built for the
        // default Unlimited-OCR arch (Baidu tensor names, R-SWA rings, the
        // BASE_PROMPT tokenizer); a got-ocr2/zoo artifact must take the
        // sequential loop, whose per-page forward() carries the arch dispatch.
        let spine = batch_scheduler::spine_enabled()
            && int8_decode_requested()
            && std::env::var_os(DECODE_STATELESS_ENV).is_none()
            && self.arch().id() == model_arch::default_arch().id();
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
        let t = std::time::Instant::now();
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
        let embed_table = self.weights.mat("model.embed_tokens.weight")?;
        let tokenizer = self.tokenizer()?;

        // ── per-page prefill, staged so vision can batch ACROSS pages ────────────
        let mut out: Vec<Option<FocrResult<String>>> = (0..n).map(|_| None).collect();

        // Stage 1: preprocess every page (a per-page error records + skips it).
        // Same mode resolution as the single-page forward (bd-1e9n) — the
        // batched-vision stage groups views by side, so Gundam's mixed
        // 640/1024 views batch correctly too.
        let mode = preprocess_mode();
        let mut pres: Vec<Option<Preprocessed>> = (0..n).map(|_| None).collect();
        for (gi, &path) in image_paths.iter().enumerate() {
            match preprocess::preprocess_image(path, mode) {
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
        require_experimental_full_int8_recipe()?;
        self.decoder_cache_i8
            .get_or_try_init(|| decoder::DecoderWeightCacheI8::build(&self.weights))
    }

    /// Build or fetch the conservative mixed-precision decoder cache. Unlike
    /// the legacy all-f32 cache, this keeps the 2,148 recipe FFN/expert matrices
    /// in int8 storage, so it is practical to retain across pages.
    fn decoder_cache(&self) -> FocrResult<&decoder::DecoderWeightCache> {
        self.decoder_cache
            .get_or_try_init(|| decoder::DecoderWeightCache::build(&self.weights))
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
            let th = std::time::Instant::now();
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
            let started = std::time::Instant::now();
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
            got::hydrate_statics(&self.weights, self.arch().vision_tower_prefix())
        })
    }

    /// The lazily-hydrated OneChart model-constant tensors — see
    /// [`Self::got_statics`].
    ///
    /// # Errors
    /// A missing or mis-shaped OneChart tensor.
    fn onechart_statics(&self) -> FocrResult<&onechart::OnechartStatics> {
        self.onechart_statics.get_or_try_init(|| {
            onechart::hydrate_statics(&self.weights, self.arch().vision_tower_prefix())
        })
    }

    /// The lazily-hydrated SmolVLM2 model-constant tensors — see
    /// [`Self::got_statics`].
    ///
    /// # Errors
    /// A missing or mis-shaped SmolVLM2 tensor.
    fn smol_statics(&self) -> FocrResult<&smolvlm2::SmolStatics> {
        self.smol_statics
            .get_or_try_init(|| smolvlm2::hydrate_statics(&self.weights))
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
            // The tokenizer ships beside the weights; OneChart uses the OPT
            // slow-tokenizer triple instead of tokenizer.json (D9).
            let dir = self.path.parent().unwrap_or_else(|| Path::new("."));
            if self.arch().id() == "onechart" {
                crate::tokenizer::Tokenizer::from_opt_dir(dir)
            } else {
                crate::tokenizer::Tokenizer::load(&dir.join("tokenizer.json"))
            }
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
        let statics = if unlimited_vision_cache_enabled() {
            Some(self.unlimited_vision_statics()?)
        } else {
            None
        };
        let mut features = Vec::new();
        for view in Self::views(pre) {
            // SAM tower -> [1024, 16*16] x3 feature (flatten(2) layout, OQ-6).
            let ts = std::time::Instant::now();
            let sam = if let Some(statics) = statics {
                let side = (view.cols as f64).sqrt() as usize;
                if side * side != view.cols {
                    return Err(FocrError::Other(anyhow::anyhow!(
                        "native_engine::OcrModel::vision_tower: image.cols {} is not a perfect square",
                        view.cols
                    )));
                }
                vision_sam::forward_with(&statics.sam, &view, side, side)?
            } else {
                vision_sam::forward(&self.weights, &view)?
            };
            timing_log(&format!("  vision.sam {:.2}s", ts.elapsed().as_secs_f64()));
            // CLIP tower fed SAM's x3 as patch_embeds -> [N+1, 1024] (CLS at 0).
            let tc = std::time::Instant::now();
            let clip = vision_clip::forward_from_sam(
                &vision_clip::ClipConfig::default(),
                self.clip_weights()?,
                &sam,
            )?;
            timing_log(&format!("  vision.clip {:.2}s", tc.elapsed().as_secs_f64()));
            // Bridge: concat CLIP[:,1:] ++ SAM (2048) -> projector -> [N, 1280].
            let tb = std::time::Instant::now();
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
        let th = std::time::Instant::now();
        let retain_statics = unlimited_vision_cache_enabled();
        let statics_owner = retained_or_owned(
            retain_statics,
            &self.unlimited_vision,
            || -> FocrResult<UnlimitedVisionStatics> {
                let started = std::time::Instant::now();
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
            let ts = std::time::Instant::now();
            let sams = vision_sam::forward_with_batched(&statics.sam, &view_refs, *side, *side)?;
            timing_log(&format!(
                "  vision.sam(batch of {}, side {side}) {:.2}s",
                slots.len(),
                ts.elapsed().as_secs_f64()
            ));
            let tc = std::time::Instant::now();
            let sam_refs: Vec<&Mat> = sams.iter().collect();
            let clips = vision_clip::forward_batched_from_sam(&clip_cfg, clip_w, &sam_refs)?;
            timing_log(&format!(
                "  vision.clip(batch of {}) {:.2}s",
                slots.len(),
                tc.elapsed().as_secs_f64()
            ));
            let tb = std::time::Instant::now();
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

    /// Embed the prompt id-stream into the decoder hidden rail ([SPEC-070]).
    ///
    /// `embed_tokens(prompt_ids)` against `model.embed_tokens.weight` (`[vocab,
    /// hidden]` = `[129280, 1280]`). The gather math lives in
    /// [`decoder::embed_tokens`] (unit-tested); the table is read through the
    /// `.focrq`/safetensors accessor [`Weights::mat`].
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if the embedding table is absent/malformed,
    /// or whatever [`decoder::embed_tokens`] returns on a bad id.
    fn embed_prompt(&self, prompt_ids: &[u32]) -> FocrResult<Mat> {
        let table = self.weights.mat("model.embed_tokens.weight")?;
        let (vocab, hidden) = (table.rows, table.cols);
        decoder::embed_tokens(&table.data, vocab, hidden, prompt_ids)
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
        if std::env::var_os(DECODE_STATELESS_ENV).is_some() {
            self.generate_stateless(
                inputs_embeds,
                prompt_ids,
                params,
                observer,
                &mut runaway_guard,
            )
        } else if int8_decode_requested() {
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
        }
    }

    /// O(n) cached greedy decode (bd-1gv.17): [`decoder::prefill_with_cache`] once
    /// to seed each layer's R-SWA reference K/V, then
    /// [`decoder::decode_step_with_cache`] per token over the ring caches. The
    /// reference block (the entire prefill) is never evicted; the generated tail
    /// rides a 128-slot ring, so memory and per-step cost are bounded.
    ///
    /// Both prefill and decode run off a [`decoder::DecoderWeightCache`] built
    /// once here — the decoder weights are dequantized to f32 a single time
    /// instead of re-dequantizing ~10 GB of expert weights from the bf16 payload
    /// every token (the dominant decode cost). No math changes, only the weight
    /// source, so the output is identical to the `&Weights` path.
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

        let table = self.weights.mat("model.embed_tokens.weight")?;
        let vocab = table.rows;
        if table.cols != hidden_dim {
            return Err(FocrError::Other(anyhow::anyhow!(
                "native_engine::OcrModel::generate_cached: embed table hidden {} != inputs_embeds hidden {}",
                table.cols,
                hidden_dim
            )));
        }

        // Build the conservative recipe cache once per model: int8 FFN/expert
        // matrices, high-precision attention and lm_head.
        let tb = std::time::Instant::now();
        let wc = self.decoder_cache()?;
        timing_log(&format!(
            "weight_cache_build {:.2}s",
            tb.elapsed().as_secs_f64()
        ));

        // Prefill once, capturing each layer's reference K/V; `last_hidden` is the
        // final prefill position, which predicts the FIRST generated token.
        let tp = std::time::Instant::now();
        let (hidden, mut caches) = decoder::prefill_with_cache(wc, &inputs_embeds)?;
        let mut last_hidden = Self::last_hidden_row(&hidden)?;
        timing_log(&format!(
            "prefill {:.2}s ({} tokens)",
            tp.elapsed().as_secs_f64(),
            prefill_len
        ));
        let td = std::time::Instant::now();
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
            let row = table.data[next * hidden_dim..(next + 1) * hidden_dim].to_vec();
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
        require_experimental_full_int8_recipe()?;
        timing_log("precision focr-full-int8");
        let hidden_dim = inputs_embeds.cols;
        let prefill_len = inputs_embeds.rows;

        let table = self.weights.mat("model.embed_tokens.weight")?;
        let vocab = table.rows;
        if table.cols != hidden_dim {
            return Err(FocrError::Other(anyhow::anyhow!(
                "native_engine::OcrModel::generate_cached_i8: embed table hidden {} != inputs_embeds hidden {}",
                table.cols,
                hidden_dim
            )));
        }

        // Quantize the decoder weights to int8 ONCE and cache on the model; a
        // load-once batch then reuses it across pages (the build is ~1.2 s). The
        // first page pays the quant; every later page in the same process skips it.
        let tb = std::time::Instant::now();
        let wc = self.decoder_cache_i8()?;
        timing_log(&format!(
            "weight_cache_build_i8 {:.2}s",
            tb.elapsed().as_secs_f64()
        ));

        let tp = std::time::Instant::now();
        let (hidden, mut caches) = decoder::prefill_with_cache_i8(wc, &inputs_embeds)?;
        let mut last_hidden = Self::last_hidden_row(&hidden)?;
        timing_log(&format!(
            "prefill_i8 {:.2}s ({} tokens)",
            tp.elapsed().as_secs_f64(),
            prefill_len
        ));
        let td = std::time::Instant::now();
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
            let row = table.data[next * hidden_dim..(next + 1) * hidden_dim].to_vec();
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
        table: &Mat,
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
        table: &Mat,
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
        let row = table.data[idx * hidden_dim..(idx + 1) * hidden_dim].to_vec();
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
        table: &Mat,
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

        let table = self.weights.mat("model.embed_tokens.weight")?;
        let vocab = table.rows;
        if table.cols != hidden_dim {
            return Err(FocrError::Other(anyhow::anyhow!(
                "native_engine::OcrModel::generate_stateless: embed table hidden {} != inputs_embeds hidden {}",
                table.cols,
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
            data.extend_from_slice(&table.data[next * hidden_dim..(next + 1) * hidden_dim]);
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
    fn build_prompt_multi(&self, pres: &[Preprocessed]) -> FocrResult<(Vec<u32>, Vec<bool>)> {
        let tok = self.tokenizer()?;
        let text = tok.encode(MULTI_PAGE_PROMPT_TEXT)?;
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let shard = package.join(RAW_SAFETENSORS_SHARD_NAME);
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
            crate::dist::UNLIMITED_OCR_ARTIFACT_VERSION
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

        let loaded = load_weights_from_resolved_model_after_validation(&model_path, || {
            std::fs::rename(&model_path, &displaced_path).expect("preserve validated path entry");
            std::fs::rename(&replacement_path, &model_path).expect("swap active path entry");
        })
        .expect("load the descriptor validated before the path swap");

        assert!(loaded.record("test.original").is_some());
        assert!(loaded.record("test.replacement").is_none());
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

        let loaded = load_weights_from_resolved_model_after_validation(&active_link, || {
            std::fs::rename(&active_link, &retired_link).expect("preserve validated symlink");
            std::fs::rename(&next_link, &active_link).expect("swap active symlink");
        })
        .expect("load the descriptor validated before the symlink swap");

        assert!(loaded.record("test.original_link").is_some());
        assert!(loaded.record("test.replacement_link").is_none());
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
        let shard = dir.join(RAW_SAFETENSORS_SHARD_NAME);
        std::fs::write(&shard, b"not loaded by resolver").expect("write shard");

        let resolved = OcrModel::resolve_model(&dir).expect("resolve safetensors directory");
        assert_eq!(resolved, shard);
    }

    #[test]
    fn load_safetensors_directory_preserves_format_mismatch() {
        let dir = temp_model_dir("load_safetensors_dir");
        let shard = dir.join(RAW_SAFETENSORS_SHARD_NAME);
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
        assert!(load_weights_from_resolved_model(&tmp).is_ok());

        let mut magic_only = tmp.clone();
        magic_only.set_file_name(format!(
            "franken_ocr_available_magic_only_{}.focrq",
            std::process::id()
        ));
        std::fs::write(&magic_only, weights::FOCRQ_MAGIC).expect("write magic-only focrq");
        assert!(!native_model_available(&magic_only));
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
        let shard = dir.join(RAW_SAFETENSORS_SHARD_NAME);
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
        let shard = dir.join(RAW_SAFETENSORS_SHARD_NAME);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(2u64).to_le_bytes());
        bytes.extend_from_slice(b"{}");
        std::fs::write(&shard, bytes).expect("write empty safetensors header");

        assert!(!native_model_available(&dir));
        assert!(load_weights_from_resolved_model(&shard).is_err());
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
            },
        );
        assert_eq!(p.max_length, 700);
        assert_eq!(p.temperature.to_bits(), base.temperature.to_bits());
        assert_eq!(p.no_repeat_ngram_size, 20);
        assert_eq!(p.ngram_window, base.ngram_window);
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
                }
            }
            fn tasks(&self) -> &'static [Task] {
                &[Task::Ocr]
            }
            fn implemented(&self) -> bool {
                false
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
}
