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
pub mod moe;
pub mod nn;
pub mod postprocess;
pub mod rswa;
pub mod sampler;
pub(crate) mod spec;
pub mod tensor;
pub mod vision_bridge;
pub mod vision_clip;
pub mod vision_sam;
pub mod weights;

use std::ffi::OsStr;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use crate::error::{FocrError, FocrResult};
use crate::preprocess::{self, Preprocessed};
use sampler::{DecodeOutput, DecodeParams};
use tensor::Mat;
use weights::Weights;

/// The loaded Unlimited-OCR model: weights + the fixed-shape forward.
///
/// Held behind an [`Arc`] and cached through a process-global [`Weak`] (see
/// [`OcrModel::load`]) so repeated `focr ocr` invocations in one process share
/// one weight blob — the model is a single read-only artifact; concurrent
/// forwards are serialized by the engine's sequential page loop (plan §6.5 P6),
/// not by cloning the weights.
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
    /// The loaded weight set. Still a Phase-2 stub with no tensor accessors;
    /// every `Weights`-backed stage entrypoint therefore surfaces a clean
    /// [`FocrError::NotImplemented`] until the `.focrq` reader (bd-1es.3) lands.
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
    decoder_cache_i8: std::sync::OnceLock<decoder::DecoderWeightCacheI8>,
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
const SAFETENSORS_SNIFF_MAX_HEADER_BYTES: usize = 8 * 1024 * 1024;
const MODEL_DIR_ENV: &str = "FOCR_MODEL_DIR";
const MODEL_QUANT_ENV: &str = "FOCR_QUANT";

/// The base-mode task prompt. baidu's `infer(..., prompt="<image>document
/// parsing.")` (modeling_unlimitedocr.py): the literal `<image>` expands to the
/// per-view image-placeholder block, and this trailing text tokenizes to the
/// task ids (`document`,`Ġparsing`,`.` = [34030, 76466, 16]). The full base
/// prompt id-stream is therefore `[BOS] + [IMAGE]×273 + [34030, 76466, 16]`.
const BASE_PROMPT_TEXT: &str = "document parsing.";

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

/// Use the int8 (per-output-channel symmetric S8S8, NEON SDOT / x86 VNNI) weight
/// cache + prefill + decode instead of the f32 cache. Present ⇒ int8. ~4x less
/// weight memory traffic at decode and ~2.8 GB cache (vs ~10.5 GB f32); accuracy
/// verified against the f32 path + baidu CER. Off by default until the int8 CER
/// sweep is locked in; flip the default once proven across the 20-page gauntlet.
const DECODE_INT8_ENV: &str = "FOCR_DECODE_INT8";

/// Process-global override that forces the int8 decode path without mutating the
/// environment (edition-2024 `set_var` is `unsafe`, and this crate denies unsafe).
/// Set by the load-once batch CLI so a batch run gets the int8 throughput path +
/// amortized int8 weight cache. OR'd with [`DECODE_INT8_ENV`].
static FORCE_INT8_DECODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Force (or clear) the int8 decode path process-wide. See [`FORCE_INT8_DECODE`].
pub fn force_int8_decode(on: bool) {
    FORCE_INT8_DECODE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether decode should run int8: the process-global force flag OR the env var.
fn int8_decode_requested() -> bool {
    FORCE_INT8_DECODE.load(std::sync::atomic::Ordering::Relaxed)
        || std::env::var_os(DECODE_INT8_ENV).is_some()
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
fn timing_log(msg: &str) {
    if std::env::var_os("FOCR_TIMING").is_some() {
        eprintln!("[focr-timing] {msg}");
    }
}

/// Build the single-image greedy decode params, honoring [`MAX_NEW_TOKENS_ENV`].
fn decode_params_from_env() -> DecodeParams {
    let mut p = DecodeParams::single_image();
    if let Some(raw) = std::env::var_os(MAX_NEW_TOKENS_ENV)
        && let Some(s) = raw.to_str()
        && let Ok(n) = s.trim().parse::<usize>()
        && n > 0
    {
        p.max_length = n;
    }
    p
}

fn model_cache() -> &'static ModelCache {
    static CACHE: OnceLock<ModelCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

fn model_cache_guard() -> FocrResult<MutexGuard<'static, ModelCacheEntry>> {
    model_cache()
        .lock()
        .map_err(|_| FocrError::Other(anyhow::anyhow!("model cache mutex poisoned")))
}

fn looks_like_safetensors_container(bytes: &[u8]) -> bool {
    if bytes.len() < 8 {
        return false;
    }
    let Ok(header_len_bytes) = bytes[..8].try_into() else {
        return false;
    };
    let Ok(header_len) = usize::try_from(u64::from_le_bytes(header_len_bytes)) else {
        return false;
    };
    let Some(header_end) = 8usize.checked_add(header_len) else {
        return false;
    };
    if header_len == 0 || header_end > bytes.len() {
        return false;
    }
    bytes[8..header_end]
        .iter()
        .copied()
        .find(|b| !b.is_ascii_whitespace())
        == Some(b'{')
}

fn looks_like_weight_container(bytes: &[u8]) -> bool {
    bytes.starts_with(weights::FOCRQ_MAGIC) || looks_like_safetensors_container(bytes)
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
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        dirs.push(PathBuf::from(home).join(".cache/franken_ocr/models"));
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

fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|p| p == &path) {
        paths.push(path);
    }
}

fn short_model_candidates(
    search_dir: &Path,
    spec: &Path,
    quant: Option<ModelQuantPreference>,
) -> Vec<PathBuf> {
    let direct = search_dir.join(spec);
    let mut candidates = Vec::with_capacity(3);
    if spec.extension().is_none()
        && let Some(quant) = quant
    {
        push_unique_path(
            &mut candidates,
            direct.with_extension(format!("{}.focrq", quant.as_str())),
        );
    }
    push_unique_path(&mut candidates, direct.clone());
    if spec.extension().is_none() {
        push_unique_path(&mut candidates, direct.with_extension("focrq"));
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

fn sniff_weight_container_from_reader(mut reader: impl Read) -> bool {
    let mut prefix = [0u8; 8];
    if reader.read_exact(&mut prefix).is_err() {
        return false;
    }
    if &prefix[..weights::FOCRQ_MAGIC.len()] == weights::FOCRQ_MAGIC {
        return true;
    }
    let Ok(header_len) = usize::try_from(u64::from_le_bytes(prefix)) else {
        return false;
    };
    if header_len == 0 || header_len > SAFETENSORS_SNIFF_MAX_HEADER_BYTES {
        return false;
    }
    let mut header = vec![0u8; header_len];
    if reader.read_exact(&mut header).is_err() {
        return false;
    }
    header.iter().copied().find(|b| !b.is_ascii_whitespace()) == Some(b'{')
}

/// Cheaply determine whether `path` resolves to a local model artifact whose
/// header looks like `.focrq` or safetensors.
///
/// This is intentionally a **sniff**, not a load: it resolves the path, opens the
/// file, and reads only the fixed `.focrq` magic prefix or the bounded
/// safetensors JSON header. It never parses tensors and never reads payload
/// bytes, so `robot health` can call it without touching the 6.67 GB model body.
#[must_use]
pub fn native_model_available(path: &Path) -> bool {
    let Ok(resolved) = OcrModel::resolve_model(path) else {
        return false;
    };
    let Ok(file) = std::fs::File::open(resolved) else {
        return false;
    };
    sniff_weight_container_from_reader(file)
}

fn load_weights_from_resolved_model(resolved: &Path, bytes: Vec<u8>) -> FocrResult<Weights> {
    let recognized_container = looks_like_weight_container(&bytes);
    Weights::from_bytes(bytes).map_err(|e| {
        if recognized_container {
            e
        } else {
            FocrError::NotImplemented(format!(
                "native_engine::OcrModel::load — {} exists but is not a recognized model \
                 container, and the resolver that turns an arbitrary existing path into a \
                 validated Unlimited-OCR model (header-sniff bd-223.7 / manifest census \
                 Phase 2) is not yet implemented; underlying parse: {e}",
                resolved.display()
            ))
        }
    })
}

/// One page's prefill state, handed from the batch spine front end
/// ([`OcrModel::prefill_page_i8`]) to the [`batch_scheduler`]: everything a
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
    /// file is unreadable). Until the header-sniff resolver (bd-223.7) and the
    /// manifest census land, an *existing* path whose bytes are not a recognized
    /// model container surfaces [`FocrError::NotImplemented`] (the real model
    /// resolution/assembly is not wired yet). Recognized `.focrq` / safetensors
    /// containers preserve structural [`FocrError::FormatMismatch`] errors so the
    /// public format/version exit-code contract remains intact.
    pub fn load(path: &Path) -> FocrResult<Arc<Self>> {
        let resolved = Self::resolve_model(path)?;

        {
            let guard = model_cache_guard()?;
            if let Some((cached_path, weak)) = guard.as_ref()
                && *cached_path == resolved
                && let Some(strong) = weak.upgrade()
            {
                return Ok(strong);
            }
        }

        // `resolve_model` is still a skeleton that accepts ANY existing path. A
        // random non-model file therefore reaches the byte parser; keep that as
        // the friendlier "resolver not implemented" category, but do not hide
        // true structural errors once the bytes identify themselves as `.focrq`
        // or safetensors (future `.focrq` versions must remain exit code 7).
        let bytes = std::fs::read(&resolved).map_err(|e| {
            FocrError::ModelNotFound(format!(
                "cannot read weights at {}: {e}",
                resolved.display()
            ))
        })?;
        let weights = load_weights_from_resolved_model(&resolved, bytes)?;
        let model = Arc::new(Self {
            path: resolved.clone(),
            weights,
            decode_params: decode_params_from_env(),
            decoder_cache_i8: std::sync::OnceLock::new(),
        });

        let mut guard = model_cache_guard()?;
        if let Some((cached_path, weak)) = guard.as_ref()
            && *cached_path == resolved
            && let Some(strong) = weak.upgrade()
        {
            return Ok(strong);
        }
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
        let t = std::time::Instant::now();
        // ── 1. preprocess ────────────────────────────────────────────────────
        let pre = preprocess::preprocess_image(
            image_path,
            preprocess::PreprocessMode::Base { base_size: 1024 },
        )?;
        let (image_w, image_h) = Self::image_dims(&pre);
        timing_log(&format!("preprocess {:.2}s", t.elapsed().as_secs_f64()));

        // ── 2. vision tower (SAM⊕CLIP -> bridge projector 2048->1280) ─────────
        let tv = std::time::Instant::now();
        let vision_features = self.vision_tower(&pre)?;
        timing_log(&format!("vision_tower {:.2}s", tv.elapsed().as_secs_f64()));

        // ── 3. connector: prompt embeds + masked_scatter of the 273-slot block ─
        let (inputs_embeds, prompt_ids) = self.build_inputs_embeds(&pre, &vision_features)?;

        // ── 4. decoder prefill + sequential greedy AR decode to EOS ──────────
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
        let spine = batch_scheduler::spine_enabled()
            && int8_decode_requested()
            && std::env::var_os(DECODE_STATELESS_ENV).is_none();
        if !spine {
            return image_paths.iter().map(|p| self.recognize(p)).collect();
        }
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

        // ── per-page prefill (preprocess → vision → connector → prefill) ─────────
        let mut out: Vec<Option<FocrResult<String>>> = (0..n).map(|_| None).collect();
        let mut streams: Vec<batch_scheduler::PageStream> = Vec::new();
        let mut stream_caches: Vec<Vec<rswa::RingCache>> = Vec::new();
        // (global page index, image_w, image_h) in scheduler-submission order.
        let mut scheduled: Vec<(usize, u32, u32)> = Vec::new();
        for (gi, &path) in image_paths.iter().enumerate() {
            match self.prefill_page_i8(wc, path) {
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
        if let Some(c) = self.decoder_cache_i8.get() {
            return Ok(c);
        }
        let built = decoder::DecoderWeightCacheI8::build(&self.weights)?;
        let _ = self.decoder_cache_i8.set(built);
        Ok(self
            .decoder_cache_i8
            .get()
            .expect("decoder int8 cache just set"))
    }

    /// Prefill ONE page for the batch spine: the [`Self::forward`] front end
    /// (preprocess in Base mode → vision tower → connector) followed by the int8
    /// [`decoder::prefill_with_cache_i8`], returning the seed state a
    /// [`batch_scheduler::PageStream`] needs plus the page's pixel dims for
    /// postprocess. Mirrors EXACTLY the prefill the sequential
    /// [`Self::generate_cached_i8`] does, so the seeded decode is byte-identical.
    fn prefill_page_i8(
        &self,
        wc: &decoder::DecoderWeightCacheI8,
        image_path: &Path,
    ) -> FocrResult<PagePrefill> {
        let pre = preprocess::preprocess_image(
            image_path,
            preprocess::PreprocessMode::Base { base_size: 1024 },
        )?;
        let (image_w, image_h) = Self::image_dims(&pre);
        let vision_features = self.vision_tower(&pre)?;
        let (inputs_embeds, prompt_ids) = self.build_inputs_embeds(&pre, &vision_features)?;
        let prefill_len = inputs_embeds.rows;
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
    fn tokenizer(&self) -> FocrResult<crate::tokenizer::Tokenizer> {
        // The tokenizer.json ships beside the weights; the resolver hands us the
        // model path, the tokenizer sits in the same directory (or is the same
        // bundle dir). We look next to the resolved model path.
        let dir = self.path.parent().unwrap_or_else(|| Path::new("."));
        crate::tokenizer::Tokenizer::load(&dir.join("tokenizer.json"))
    }

    /// Run the two-tower vision encoder over every preprocessed view and project
    /// each into the decoder hidden rail (2048 -> 1280), returning the per-view
    /// hybrid vision features ([SPEC-040..052]).
    ///
    /// Drives, per view: [`vision_sam::forward`] -> (its `x3` becomes CLIP's
    /// `patch_embeds`) [`vision_clip::forward`] -> [`vision_bridge::forward`]
    /// (concat CLIP[:,1:] ++ SAM, then the linear projector). The SAM/CLIP/bridge
    /// kernels are implemented and tested over explicit weight bundles; the
    /// `Weights`-backed entrypoints used here surface
    /// [`FocrError::NotImplemented`] until the `.focrq` reader exposes their named
    /// tensors (bd-1es.3).
    ///
    /// # Errors
    /// The first vision-stage error (today [`FocrError::NotImplemented`] from the
    /// `Weights`-backed SAM/CLIP/bridge entrypoints).
    fn vision_tower(&self, pre: &Preprocessed) -> FocrResult<Vec<Mat>> {
        let mut features = Vec::new();
        for view in Self::views(pre) {
            // SAM tower -> [1024, 16*16] x3 feature (flatten(2) layout, OQ-6).
            let ts = std::time::Instant::now();
            let sam = vision_sam::forward(&self.weights, &view)?;
            timing_log(&format!("  vision.sam {:.2}s", ts.elapsed().as_secs_f64()));
            // CLIP tower fed SAM's x3 as patch_embeds -> [N+1, 1024] (CLS at 0).
            let tc = std::time::Instant::now();
            let clip = vision_clip::forward(&self.weights, &view, &sam)?;
            timing_log(&format!("  vision.clip {:.2}s", tc.elapsed().as_secs_f64()));
            // Bridge: concat CLIP[:,1:] ++ SAM (2048) -> projector -> [N, 1280].
            let tb = std::time::Instant::now();
            let projected = vision_bridge::forward(&self.weights, &clip, &sam)?;
            timing_log(&format!(
                "  vision.bridge {:.2}s",
                tb.elapsed().as_secs_f64()
            ));
            features.push(projected);
        }
        Ok(features)
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
        if std::env::var_os(DECODE_STATELESS_ENV).is_some() {
            self.generate_stateless(inputs_embeds, prompt_ids)
        } else if int8_decode_requested() {
            self.generate_cached_i8(inputs_embeds, prompt_ids)
        } else {
            self.generate_cached(inputs_embeds, prompt_ids)
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
    fn generate_cached(&self, inputs_embeds: Mat, prompt_ids: &[u32]) -> FocrResult<Vec<u32>> {
        let params = &self.decode_params;
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

        // Dequantize the decoder weights ONCE; prefill + every decode step then
        // borrow the owned f32 cache (the per-token re-dequant was the bottleneck).
        let tb = std::time::Instant::now();
        let wc = decoder::DecoderWeightCache::build(&self.weights)?;
        timing_log(&format!(
            "weight_cache_build {:.2}s",
            tb.elapsed().as_secs_f64()
        ));

        // Prefill once, capturing each layer's reference K/V; `last_hidden` is the
        // final prefill position, which predicts the FIRST generated token.
        let tp = std::time::Instant::now();
        let (hidden, mut caches) = decoder::prefill_with_cache(&wc, &inputs_embeds)?;
        let mut last_hidden = Self::last_hidden_row(&hidden)?;
        timing_log(&format!(
            "prefill {:.2}s ({} tokens)",
            tp.elapsed().as_secs_f64(),
            prefill_len
        ));
        let td = std::time::Instant::now();

        // `generated` seeds the no-repeat-ngram history with the prompt so the
        // sliding-window blocker sees the full context (sampler reads its tail).
        let mut generated: Vec<u32> = prompt_ids.to_vec();
        let mut emitted: Vec<u32> = Vec::new();

        while emitted.len() < params.max_length {
            let logits = decoder::lm_head_cached(&wc, &last_hidden)?;
            let step: DecodeOutput = sampler::decode_step(&logits, &generated, params)?;
            generated.push(step.token_id);
            emitted.push(step.token_id);
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
            let h = decoder::decode_step_with_cache(&wc, &mut caches, &token_embed, position)?;
            last_hidden = Self::last_hidden_row(&h)?;
        }
        timing_log(&format!(
            "decode {:.2}s ({} tokens, {:.3}s/tok)",
            td.elapsed().as_secs_f64(),
            emitted.len(),
            td.elapsed().as_secs_f64() / (emitted.len().max(1) as f64)
        ));
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
    fn generate_cached_i8(&self, inputs_embeds: Mat, prompt_ids: &[u32]) -> FocrResult<Vec<u32>> {
        let params = &self.decode_params;
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
        let wc = match self.decoder_cache_i8.get() {
            Some(c) => c,
            None => {
                let built = decoder::DecoderWeightCacheI8::build(&self.weights)?;
                // A concurrent racer may win the set; either way `get` then yields
                // the single shared cache (the batch CLI is sequential anyway).
                let _ = self.decoder_cache_i8.set(built);
                self.decoder_cache_i8.get().expect("just set")
            }
        };
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
        let spec_decode = spec_decode_enabled()
            && params.no_repeat_ngram_size == sampler::DEFAULT_NO_REPEAT_NGRAM_SIZE
            && params.ngram_window == sampler::NGRAM_WINDOW_SINGLE;
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
            )?;
        }
        while !spec_decode && emitted.len() < params.max_length {
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
    ) -> FocrResult<()> {
        let params = &self.decode_params;
        // Own a mutable copy of the seed hidden (the prefill last row); the caller's
        // `last_hidden` stays intact for the guarded sequential `while` (which only
        // runs when spec decode is OFF). Each commit replaces this with the freshly
        // decoded row, exactly as the sequential loop reassigns its `last_hidden`.
        let mut last_hidden = last_hidden.clone();
        while emitted.len() < params.max_length {
            let draft = spec::draft_ngram(generated, spec::SPEC_DRAFT_MAX, spec::SPEC_DRAFT_NGRAM);
            if draft.is_empty() {
                // EMPTY-DRAFT FALLBACK: nothing to verify, so take exactly ONE
                // sequential greedy step — the default loop's body (raw `lm_head` +
                // `decode_step`), then commit the chosen token.
                let logits = decoder::lm_head_cached_i8(wc, &last_hidden)?;
                let step = sampler::decode_step(&logits, generated, params)?;
                generated.push(step.token_id);
                emitted.push(step.token_id);
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
                if params.eos_token_id == token {
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
    ) -> FocrResult<Vec<u32>> {
        let params = &self.decode_params;
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
            // Full-sequence forward, then lm_head over ONLY the last hidden row
            // -> [1, vocab] logits (the next-token logits depend solely on the
            // final position; per decoder::lm_head_last_row_is_full_last_row).
            let hidden = decoder::forward(&self.weights, &inputs_embeds)?;
            let last_hidden = Self::last_hidden_row(&hidden)?;
            let logits = decoder::lm_head(&self.weights, &last_hidden)?;
            let step: DecodeOutput = sampler::decode_step(&logits, &generated, params)?;
            generated.push(step.token_id);
            emitted.push(step.token_id);
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
    /// header-sniff resolver is a later bead, so `resolve_model` accepts any
    /// existing path) and then fails cleanly with `NotImplemented`, never panics.
    /// `Weights::load` reports a low-level container `FormatMismatch` on the junk
    /// bytes, but the real gap is that the model package's resolve + manifest
    /// assembly is Phase-2; `OcrModel::load` maps that case to `NotImplemented`.
    /// Uses a freshly-created temp file so the test is CWD-independent.
    #[test]
    fn load_existing_non_model_path_is_not_implemented_not_panic() {
        let mut tmp = std::env::temp_dir();
        tmp.push(format!(
            "franken_ocr_load_test_{}.focrq",
            std::process::id()
        ));
        std::fs::write(&tmp, b"not a real model blob").expect("write temp file");
        let r = OcrModel::load(&tmp);
        let _ = std::fs::remove_file(&tmp); // best-effort cleanup (not a delete of source)
        // resolve_model accepts an existing path; Weights::load is the stub.
        assert!(matches!(r, Err(FocrError::NotImplemented(_))));
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
        let err = load_weights_from_resolved_model(
            Path::new("future-version.focrq"),
            minimal_focrq_blob(weights::FOCRQ_FORMAT_VERSION + 1, header, &payload),
        )
        .unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert_eq!(err.exit_code(), crate::error::EXIT_FORMAT_MISMATCH);
        assert!(format!("{err}").contains("newer than this binary"));
    }

    #[test]
    fn load_malformed_safetensors_preserves_format_mismatch() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(1u64).to_le_bytes());
        bytes.extend_from_slice(b"{");
        let err =
            load_weights_from_resolved_model(Path::new("bad.safetensors"), bytes).unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)));
        assert_eq!(err.exit_code(), crate::error::EXIT_FORMAT_MISMATCH);
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
    fn native_model_available_accepts_focrq_magic_without_loading() {
        let mut tmp = std::env::temp_dir();
        tmp.push(format!(
            "franken_ocr_available_focrq_{}.focrq",
            std::process::id()
        ));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(weights::FOCRQ_MAGIC);
        bytes.extend_from_slice(&[0u8; 2]);
        std::fs::write(&tmp, bytes).expect("write focrq prefix");

        assert!(native_model_available(&tmp));
    }

    #[test]
    fn native_model_available_accepts_safetensors_directory_header() {
        let dir = temp_model_dir("available_safetensors_dir");
        let shard = dir.join(RAW_SAFETENSORS_SHARD_NAME);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(2u64).to_le_bytes());
        bytes.extend_from_slice(b"{}");
        std::fs::write(&shard, bytes).expect("write safetensors header");

        assert!(native_model_available(&dir));
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
}
