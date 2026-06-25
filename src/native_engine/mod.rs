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

pub mod connector;
pub mod decoder;
pub mod moe;
pub mod nn;
pub mod postprocess;
pub mod rswa;
pub mod sampler;
pub mod tensor;
pub mod vision_bridge;
pub mod vision_clip;
pub mod vision_sam;
pub mod weights;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

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
}

/// Process-global cache of the last-loaded model, keyed by resolved path.
///
/// A [`Weak`] so the cache never *keeps the model alive on its own*: once every
/// [`Arc<OcrModel>`] handle is dropped, the weight blob is freed; a subsequent
/// [`OcrModel::load`] of the same path re-reads it. While at least one handle is
/// live, repeat loads of the same path hand back a cheap `Arc::clone`.
type ModelCache = Mutex<Option<(PathBuf, Weak<OcrModel>)>>;

fn model_cache() -> &'static ModelCache {
    static CACHE: OnceLock<ModelCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

fn looks_like_safetensors_container(bytes: &[u8]) -> bool {
    if bytes.len() < 8 {
        return false;
    }
    let Ok(header_len_bytes) = bytes[..8].try_into() else {
        return false;
    };
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;
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

impl OcrModel {
    /// Resolve `path` to a concrete model artifact (`.focrq` blob or a
    /// safetensors directory) — the header-sniff / search-path logic
    /// (`native_model_available`, bd-223.7).
    ///
    /// Skeleton: returns `path` as-is if it exists, else
    /// [`FocrError::ModelNotFound`]. The candidate-path search + magic sniff land
    /// with the resolver bead.
    pub fn resolve_model(path: &Path) -> FocrResult<PathBuf> {
        if path.exists() {
            Ok(path.to_path_buf())
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

        let cache = model_cache();
        let mut guard = cache
            .lock()
            .map_err(|_| FocrError::Other(anyhow::anyhow!("model cache mutex poisoned")))?;
        if let Some((cached_path, weak)) = guard.as_ref()
            && *cached_path == resolved
            && let Some(strong) = weak.upgrade()
        {
            return Ok(strong);
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
            decode_params: DecodeParams::single_image(),
        });
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
    /// currency. Because [`Weights`] has no tensor accessors yet (the `.focrq`
    /// reader is Phase 2, bd-1es.3), each `Weights`-backed stage entrypoint
    /// returns a clean [`FocrError::NotImplemented`] — the pipeline shape is
    /// fully wired and typed, the first such gap propagates verbatim.
    ///
    /// # Errors
    /// Whatever the first failing stage returns. Today that is
    /// [`FocrError::NotImplemented`] from [`preprocess::preprocess_image`] (the
    /// image front end is bd-1gv.2/3), surfaced through the typed pipeline.
    pub fn forward(&self, image_path: &Path) -> FocrResult<(String, u32, u32)> {
        // ── 1. preprocess ────────────────────────────────────────────────────
        let pre = preprocess::preprocess_image(
            image_path,
            preprocess::PreprocessMode::Base { base_size: 1024 },
        )?;
        let (image_w, image_h) = Self::image_dims(&pre);

        // ── 2. vision tower (SAM⊕CLIP -> bridge projector 2048->1280) ─────────
        let vision_features = self.vision_tower(&pre)?;

        // ── 3. connector: prompt embeds + masked_scatter of the 273-slot block ─
        let (mut inputs_embeds, prompt_ids) = self.build_inputs_embeds(&pre, &vision_features)?;

        // ── 4. decoder prefill + sequential greedy AR decode to EOS ──────────
        let generated = self.generate(&mut inputs_embeds, &prompt_ids)?;

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

    // ── stage orchestration (private) ──────────────────────────────────────────

    /// The byte-level BPE tokenizer over `tokenizer.json` (sibling of the
    /// model artifact), loaded lazily.
    ///
    /// The tokenizer is needed for the prompt id-stream (the connector builds
    /// `images_seq_mask` against it) and to detokenize the generated ids. It is
    /// resolved next to the model file; the loader is bd-1gv.1 and currently
    /// surfaces [`FocrError::NotImplemented`].
    ///
    /// # Errors
    /// [`FocrError::NotImplemented`] until the BPE tokenizer lands (bd-1gv.1).
    fn tokenizer(&self) -> FocrResult<crate::tokenizer::Tokenizer> {
        // The tokenizer.json ships beside the weights; the resolver hands us the
        // model path, the tokenizer sits in the same directory (or is the same
        // bundle dir). The real co-location lookup lands with the loader bead;
        // for the wiring we look next to the resolved model path.
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
            let sam = vision_sam::forward(&self.weights, &view)?;
            // CLIP tower fed SAM's x3 as patch_embeds -> [N+1, 1024] (CLS at 0).
            let clip = vision_clip::forward(&self.weights, &view, &sam)?;
            // Bridge: concat CLIP[:,1:] ++ SAM (2048) -> projector -> [N, 1280].
            let projected = vision_bridge::forward(&self.weights, &clip, &sam)?;
            features.push(projected);
        }
        Ok(features)
    }

    /// Build the decoder `inputs_embeds` by embedding the prompt id-stream and
    /// scattering the per-view vision features into the `<image>` placeholder
    /// rows ([SPEC-060..066], [SPEC-070]).
    ///
    /// Returns the `[seq, hidden]` fused embedding plus the prompt id sequence
    /// (the AR loop seeds its no-repeat-ngram history with it). The connector's
    /// structural assembly + `masked_scatter` are implemented and tested over
    /// explicit params; the `Weights`-backed token-embed + `image_newline`/
    /// `view_seperator` lookups surface [`FocrError::NotImplemented`] until the
    /// reader lands.
    ///
    /// # Errors
    /// The first connector/embed error (today [`FocrError::NotImplemented`]).
    fn build_inputs_embeds(
        &self,
        pre: &Preprocessed,
        vision_features: &[Mat],
    ) -> FocrResult<(Mat, Vec<u32>)> {
        // The prompt id-stream (BOS + `<image>` placeholders + the task prompt)
        // and the row-aligned `images_seq_mask` are produced by preprocess; the
        // connector scatters `vision_features` into the masked rows.
        let prompt_ids = Self::prompt_ids(pre);
        let images_seq_mask = Self::images_seq_mask(pre);

        // embed_tokens(prompt_ids) -> [seq, hidden]; needs the embedding table
        // from Weights (bd-1es.3). decoder::forward is the Weights-backed shim.
        let mut inputs_embeds = self.embed_prompt(&prompt_ids)?;

        // Scatter every per-view 273-slot block into the placeholder rows. The
        // no-crop / single-global path (assemble_global_block + masked_scatter)
        // is the base 1024 case; the connector validates the ORDERING INVARIANT.
        connector::fuse_no_crop(
            &self.weights,
            &mut inputs_embeds,
            vision_features,
            Self::global_grid_h(pre),
            Self::global_grid_w(pre),
            Self::image_newline(&self.weights)?,
            Self::view_seperator(&self.weights)?,
            &images_seq_mask,
        )?;
        Ok((inputs_embeds, prompt_ids))
    }

    /// Embed the prompt id-stream into the decoder hidden rail ([SPEC-070]).
    ///
    /// `embed_tokens(prompt_ids)` against `model.embed_tokens.weight`. The
    /// gather math lives in [`decoder::embed_tokens`]; the `Weights`-backed
    /// `decoder::forward` shim that would hand it the table is
    /// [`FocrError::NotImplemented`] until the reader lands, so we route through
    /// it to keep the dependency explicit.
    ///
    /// # Errors
    /// [`FocrError::NotImplemented`] until the embedding table is readable.
    fn embed_prompt(&self, _prompt_ids: &[u32]) -> FocrResult<Mat> {
        // The embedding table lives in Weights; until the reader exposes it the
        // decoder's Weights-backed entrypoint reports the gap. We surface that
        // same error here (a placeholder hidden Mat would be a fabricated value,
        // which doctrine #1 forbids).
        Err(FocrError::NotImplemented(
            "native_engine::OcrModel::embed_prompt — model.embed_tokens.weight needs the .focrq \
             tensor accessor (bd-1es.3); decoder::embed_tokens math is implemented and tested"
                .into(),
        ))
    }

    /// Prefill the decoder over `inputs_embeds`, then run the **sequential**
    /// greedy autoregressive decode loop to EOS, returning the generated token
    /// ids ([SPEC-072..103]).
    ///
    /// Doctrine #5: this loop is strictly sequential — one forward at a time, the
    /// per-step R-SWA/MoE math fans out across cores inside the kernels, never a
    /// nested runtime, never rayon under a lock. The R-SWA ring cache bounds the
    /// generated-token KV at `W = 128` while retaining the full reference block
    /// (prefill), so memory does not grow without bound during decode.
    ///
    /// # Errors
    /// The first decode-stage error (today [`FocrError::NotImplemented`] from the
    /// `Weights`-backed `decoder::forward`).
    fn generate(&self, inputs_embeds: &mut Mat, prompt_ids: &[u32]) -> FocrResult<Vec<u32>> {
        let params = &self.decode_params;

        // Prefill: run all 12 layers over the prompt, capturing each layer's
        // reference K/V into its R-SWA ring cache and returning the final hidden
        // state for the first decode step. The Weights-backed driver shim is
        // NotImplemented until the reader + ring wiring land (bd-1es.3/bd-1gv.17).
        let mut hidden = decoder::forward(&self.weights, inputs_embeds)?;

        // `generated` seeds the no-repeat-ngram history with the prompt so the
        // sliding-window blocker sees the full context (sampler reads its tail).
        let mut generated: Vec<u32> = prompt_ids.to_vec();
        let mut emitted: Vec<u32> = Vec::new();

        // SEQUENTIAL greedy decode loop (no nested runtime, no rayon-under-lock).
        // Bounded by `max_length` so a non-converging model can never hang.
        let start = generated.len();
        while generated.len() - start < params.max_length {
            // lm_head over the last hidden row -> [1, vocab] logits.
            let logits = decoder::lm_head(&self.weights, &hidden)?;
            let step: DecodeOutput = sampler::decode_step(&logits, &generated, params)?;
            generated.push(step.token);
            emitted.push(step.token);
            if step.is_eos {
                break;
            }
            // Next-step hidden: embed the just-emitted token and run one decode
            // step through the ring-cache'd decoder. The Weights-backed driver is
            // NotImplemented until the reader lands; the loop body is the wired
            // shape (one token in -> one hidden row out).
            let step_embed = self.embed_prompt(&[step.token])?;
            hidden = decoder::forward(&self.weights, &step_embed)?;
        }
        Ok(emitted)
    }

    // ── preprocess/weights field accessors (loader-handoff shims) ──────────────
    //
    // `Preprocessed` and `Weights` are still field-less Phase-1/2 stubs; the
    // accessors below name the exact data each stage needs so that when the
    // preprocess (bd-1gv.2/3) and `.focrq` reader (bd-1es.3) beads add the
    // concrete fields, the wiring is a mechanical body swap with the call sites
    // above already correct. They return the documented defaults today.

    /// Source image pixel dimensions `(w, h)` for bbox de-normalization
    /// ([SPEC-018]). Filled from `Preprocessed` once it carries the original
    /// extent (the `ori` tensor of `images=[(crop, ori)]`).
    fn image_dims(_pre: &Preprocessed) -> (u32, u32) {
        (0, 0)
    }

    /// The preprocessed view tensors (`[3, H, W]` each), one per crop/global view
    /// ([SPEC-020..033]). Base mode yields a single 1024 global view; the Gundam
    /// crop branch yields the local tiles plus the global view.
    fn views(_pre: &Preprocessed) -> Vec<Mat> {
        Vec::new()
    }

    /// The prompt token id-stream (BOS + `<image>` placeholders + task prompt),
    /// [SPEC-019]/[SPEC-035].
    fn prompt_ids(_pre: &Preprocessed) -> Vec<u32> {
        Vec::new()
    }

    /// Row-aligned `images_seq_mask` (one bool per prompt token; `true` at each
    /// `<image>` placeholder), [SPEC-066].
    fn images_seq_mask(_pre: &Preprocessed) -> Vec<bool> {
        Vec::new()
    }

    /// Global feature-grid height (16 at base 1024) ([SPEC-063]).
    fn global_grid_h(_pre: &Preprocessed) -> usize {
        16
    }

    /// Global feature-grid width (16 at base 1024) ([SPEC-063]).
    fn global_grid_w(_pre: &Preprocessed) -> usize {
        16
    }

    /// The learned `model.image_newline` parameter (length `N_EMBED = 1280`),
    /// [SPEC-060]. Needs the `.focrq` tensor accessor (bd-1es.3).
    fn image_newline(_weights: &Weights) -> FocrResult<&[f32]> {
        Err(FocrError::NotImplemented(
            "native_engine::OcrModel::image_newline — model.image_newline needs the .focrq tensor \
             accessor (bd-1es.3)"
                .into(),
        ))
    }

    /// The learned `model.view_seperator` parameter (length `N_EMBED = 1280`),
    /// [SPEC-060]. Needs the `.focrq` tensor accessor (bd-1es.3).
    fn view_seperator(_weights: &Weights) -> FocrResult<&[f32]> {
        Err(FocrError::NotImplemented(
            "native_engine::OcrModel::view_seperator — model.view_seperator needs the .focrq \
             tensor accessor (bd-1es.3)"
                .into(),
        ))
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

    /// The two learned structural params are gated on the (not-yet-built) `.focrq`
    /// tensor accessor and report a clean NotImplemented, never a panic — the
    /// connector wiring depends on them.
    #[test]
    fn structural_params_pending_weights_accessor() {
        let w = Weights::default();
        assert!(matches!(
            OcrModel::image_newline(&w),
            Err(FocrError::NotImplemented(_))
        ));
        assert!(matches!(
            OcrModel::view_seperator(&w),
            Err(FocrError::NotImplemented(_))
        ));
    }

    /// Image dims accessor returns the documented `(0, 0)` placeholder until
    /// `Preprocessed` carries the original extent; postprocess tolerates it.
    #[test]
    fn image_dims_placeholder_is_zero_until_preprocess_lands() {
        let pre = Preprocessed::default();
        assert_eq!(OcrModel::image_dims(&pre), (0, 0));
    }
}
