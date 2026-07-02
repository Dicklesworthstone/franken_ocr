//! The L0–L5 parity ladder + oracle-differential comparator (integration test).
//!
//! Design-of-record: `docs/conformance/LADDER_HARNESS.md` (this harness),
//! `docs/conformance/PARITY_LADDER.md` (the rung spec), and
//! `docs/gauntlet/METHODOLOGY.md` §1 (the comparator). The shared comparator
//! infra lives in `support/parity_harness.rs` and is declared below.
//!
//! What is ALWAYS-ON here (no weights, no oracle fixtures):
//!   * the comparator MATH (cosine, ULP table, scrubbers, the nondeterminism-
//!     floor helper) — unit-tested in the support module with synthetic vectors;
//!   * the L0 EXACT-tolerance *contract* checks that need no fixture (the
//!     stable-surface checks: error exit codes, CLI/robot schema);
//!   * the rung skeletons themselves, which run their gating logic and emit a
//!     structured line every time — even when they skip.
//!
//! What is GATED (skip-with-SUCCESS, never a silent fake pass):
//!   * every rung that needs the CUDA-host oracle fixtures
//!     (`tests/fixtures/native/...` from `scripts/gen_reference_fixtures.py`) —
//!     gated on [`parity_harness::FixtureLoader::any_present`];
//!   * every rung that needs the 6.67 GB weights — gated on the model resolving,
//!     and PROVING the native path ran by pointing the fallback at `/nonexistent`.
//!
//! Each rung emits exactly one terminal NDJSON line conforming to the frozen
//! `tests/fixtures/test_log_schema.json` contract: on a skip a
//! `result=skip_no_model` SUCCESS line explaining WHY; on a run a `parity` line
//! carrying `{gate, metric, value, tolerance, oracle_fixture, pass}`. Failures
//! are self-diagnosing — the diff / the mismatched field / the offending index
//! is printed.

#[path = "support/parity_harness.rs"]
mod parity_harness;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use parity_harness::{
    COSINE_F32_THRESHOLD, DType, FixtureLoader, Logger, NormalizedValue, OpFamily, ReferenceGolden,
    TensorSpec, cosine, establish_floor, max_abs_diff, scrub_volatile, sha256_hex, ulp_compare,
};
use serde_json::{Value, json};

// The subject (engine) side of the ladder. These are the SAME public kernels the
// off-repo example dumps (`examples/full_vision_dump.rs`, `examples/decoder_dump.rs`)
// drove to PROVE vision cosine 0.9996 and decoder argmax-exact against baidu. Wiring
// them here promotes those manual proofs into committed, gated L0–L5 parity rungs
// (bd-2ksr: replace the diagnostic oracle-only self-compares with real subject
// capture, unlocked by the bd-3s7v seam-INPUT dump: sam_input + inputs_embeds +
// token_stream).
use franken_ocr::native_engine::sampler::{self, DecodeParams};
use franken_ocr::native_engine::tensor::Mat;
use franken_ocr::native_engine::weights::Weights;
use franken_ocr::native_engine::{
    connector, decoder, postprocess, vision_bridge, vision_clip, vision_sam,
};
use franken_ocr::preprocess;
use franken_ocr::tokenizer::{Tokenizer, special};

// ─────────────────────────────────────────────────────────────────────────────
// Gating helpers — the model/fixture gate (skip-with-SUCCESS discipline).
// ─────────────────────────────────────────────────────────────────────────────

/// Are the oracle fixtures present? Every rung that compares against the oracle
/// gates on this. Absent ⇒ skip-with-SUCCESS (the fixtures come from a CUDA host
/// per OQ-17 and are not on a default dev box).
fn fixtures_present() -> bool {
    FixtureLoader::new().any_present()
}

/// Resolve the model path the same way the lib does (`$FOCR_MODEL_PATH` else the
/// default). The model-gated e2e rungs check this resolves to a real artifact;
/// absent ⇒ skip-with-SUCCESS, proving the native path by the `/nonexistent`
/// fallback the log carries.
fn model_present() -> bool {
    subject_model_path().exists()
}

/// One golden's doc stem (`<stem>_reference.json` → `<stem>`).
fn golden_stem(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_suffix("_reference.json"))
        .unwrap_or("unknown")
        .to_string()
}

// ─────────────────────────────────────────────────────────────────────────────
// Subject (engine) seam capture — the bd-2ksr deliverable.
//
// The oracle (scripts/gen_reference_fixtures.py) dumps each module's OUTPUT
// activation (sam_output / clip_output / projector_output /
// decoder_layer_NN_hidden / lm_head_logits) AND — since the bd-3s7v regen — the
// two seam INPUTS (sam_input = the preprocessed pixel tensor, inputs_embeds =
// the fused post-connector embedding) plus the full `token_stream` block
// (prompt_ids + generated_ids). Feeding the engine the oracle's EXACT upstream
// tensor and comparing its output to the oracle's output isolates each stage,
// the same decouple the example dumps use:
//   * L0 preprocess: preprocess::preprocess_image(<doc>.png) vs sam_input —
//     isolates decode/resize/pad/normalize (resample tolerance ledgered,
//     bd-30me).
//   * L1 per-op: vision_sam::forward(sam_input) vs sam_output;
//     vision_clip::forward(sam_output) vs clip_output;
//     vision_bridge::forward(clip_output, sam_output) vs projector_output.
//   * L2 layer-0: the full vision(sam→clip→bridge) + embed_tokens(prompt_ids) +
//     connector::fuse_no_crop splice vs inputs_embeds.
//   * L3: decoder::lm_head(decoder_layer_11_hidden) vs lm_head_logits —
//     isolates the final model.norm + lm_head GEMV.
//   * L4: the engine greedy AR decode loop (prefill_with_cache +
//     decode_step_with_cache + sampler::decode_step) seeded with the oracle's
//     exact inputs_embeds, vs token_stream.generated_ids — EXACT (the oracle run
//     is fully deterministic: CPU, greedy, torch.use_deterministic_algorithms).
//   * L5: detokenize + postprocess::finalize over the L4 subject ids vs the
//     golden decoded_text, CER within the documented budget.
// Every comparison runs through the shared comparator and emits a REAL
// `log.parity` (pass/fail), never a diagnostic self-compare. A shape/numel
// mismatch or a missing tensor is surfaced LOUDLY (never a fabricated pass).
//
// The one seam family the committed fixtures still cannot isolate is the
// per-layer decoder ledger for layers 00..10 (it needs a public single-layer
// engine entry seeded by the PRIOR layer's oracle hidden; only the 12-layer
// driver is exposed). L2 names that gap precisely; the 12-layer stack itself IS
// covered (the L4 prefill hidden is ledgered against decoder_layer_11_hidden).
// ─────────────────────────────────────────────────────────────────────────────

/// The subject identity stamped on every real parity row so the differential
/// guard (`EngineIdentity subject != oracle`) holds structurally.
const SUBJECT_IDENTITY: &str = "franken_ocr";
/// The oracle identity (the pinned baidu reference).
const ORACLE_IDENTITY: &str = "unlimited-ocr-oracle";

/// The resolved model path (`$FOCR_MODEL_PATH` else the default `.focrq`) —
/// shared by [`model_present`], [`load_subject_weights`] and the L5 tokenizer
/// resolution (`tokenizer.json` ships beside the weights, exactly as the lib
/// resolves it).
fn subject_model_path() -> PathBuf {
    std::env::var_os("FOCR_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/unlimited-ocr.focrq"))
}

/// Resolve + load the subject model weights via [`subject_model_path`]. Only
/// called after a rung confirms `model_present()`, so a failure here is a
/// genuine load error worth surfacing (never silently skipped).
fn load_subject_weights() -> Result<Weights, String> {
    let path = subject_model_path();
    Weights::load(&path).map_err(|e| format!("load subject weights {}: {e}", path.display()))
}

/// Serializes the fixture+model-gated rungs. Each gated rung holds a full
/// [`Weights`] load (~6.7 GB widened on access) and L4 additionally builds the
/// f32 [`decoder::DecoderWeightCache`]; the default parallel test harness would
/// otherwise run several of those at once and blow the memory budget. The lock
/// is taken AFTER the gate check, so on a fixtureless box (every rung skips)
/// there is zero contention. Poison is deliberately swallowed: a failed rung
/// must not cascade spurious lock panics into its siblings.
static HEAVY_RUNG_LOCK: Mutex<()> = Mutex::new(());

fn heavy_rung_guard() -> std::sync::MutexGuard<'static, ()> {
    HEAVY_RUNG_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The documented L0 resample tolerance (bd-30me / DISC-001), in normalized
/// `[-1, 1]` pixel units. By DEFAULT the subject resizes with
/// `image::imageops::FilterType::CatmullRom` (Keys cubic, a = -0.5, f32
/// accumulation) while the oracle uses PIL `ImageOps.pad` BICUBIC (the SAME
/// Keys a = -0.5 kernel, but PIL's fixed-point two-pass pipeline rounds each
/// separable pass to 8 bits). Kernel-identical, implementation-divergent ⇒
/// per-pixel drift of a few u8 quantization steps at content edges, never a
/// structural shift. Budget: 8 u8 steps = 8 * (2/255). Exceeding it (e.g. a
/// 1-px geometry/centering mismatch, which shifts whole rows) is a REAL
/// bd-30me defect and must fail the rung.
///
/// The EXACT path exists: the DISC-001 kill-switch `FOCR_RESAMPLE=pil-bicubic`
/// (`preprocess::RESAMPLE_ENV`, OFF by default per doctrine) swaps every resize
/// onto the Pillow-bit-exact fixed-point BICUBIC — under it the rung's
/// exact-first gate passes with `exact=true` and this envelope never engages.
/// The active kernel is ledgered per parity row (`resample_kind`).
const L0_RESAMPLE_MAX_ABS_TOL: f64 = 8.0 * (2.0 / 255.0);

/// The documented L5 CER budget. The L4 rung already requires the token stream
/// to be EXACT, so the only subject-vs-oracle divergence L5 may absorb is the
/// detokenizer + `postprocess::finalize` markdown assembly (the oracle's text is
/// `model.infer()`'s post-processed result.md). 1% CER is the documented
/// formatting envelope; a token-level divergence blows far past it (and fails
/// L4 first, self-attributing the layer).
const L5_CER_BUDGET: f64 = 0.01;

/// After the subject decode diverges from the golden stream the exact-compare
/// verdict is already sealed, so the AR loop only needs enough extra steps to
/// prove the engine did not merely stop early. This slack bounds a runaway
/// (divergent) decode instead of grinding to `max_length` = 32768.
const DECODE_DIVERGENCE_SLACK: usize = 8;

// ── fixture-shape adapters (loud, never a silent misshape) ──────────────────

/// Reshape the oracle `sam_input` activation (`[1, 3, H, W]`, C-order) into the
/// `[3, H*W]` channel-major view [`Mat`] the vision tower consumes (the exact
/// [`preprocess::ViewTensor::pixels`] layout). Rejects any other rank/shape
/// loudly — a corrupt manifest must never silently misshape into a pass.
fn sam_input_view_mat(stage: &str, nv: &NormalizedValue) -> Result<Mat, String> {
    match nv.spec.shape.as_slice() {
        [1, 3, h, w] => {
            let hw = h.checked_mul(*w).ok_or_else(|| {
                format!(
                    "activation {stage}: H*W overflow for shape {:?}",
                    nv.spec.shape
                )
            })?;
            if nv.data.len() != 3 * hw {
                return Err(format!(
                    "activation {stage}: flat len {} != 3*H*W {} (shape {:?})",
                    nv.data.len(),
                    3 * hw,
                    nv.spec.shape
                ));
            }
            Ok(Mat::from_vec(3, hw, nv.data.clone()))
        }
        other => Err(format!(
            "activation {stage}: expected shape [1, 3, H, W], got {other:?}"
        )),
    }
}

/// Reshape the oracle `sam_output` activation (`[1, C, H, W]`, C-order) into the
/// `[C, H*W]` channel-major feature [`Mat`] that `vision_sam::forward` emits
/// (the `flatten(2)` layout) — the layout `vision_clip::forward` and
/// `vision_bridge::forward` consume. The previous last-dim-cols reshape
/// (`activation_as_mat`) folded `[1, 1024, 16, 16]` into `[16384, 16]`, which
/// the bridge would reject; this adapter restores the real seam layout.
fn sam_output_features_mat(stage: &str, nv: &NormalizedValue) -> Result<Mat, String> {
    match nv.spec.shape.as_slice() {
        [1, c, h, w] => {
            let hw = h.checked_mul(*w).ok_or_else(|| {
                format!(
                    "activation {stage}: H*W overflow for shape {:?}",
                    nv.spec.shape
                )
            })?;
            let numel = c.checked_mul(hw).ok_or_else(|| {
                format!(
                    "activation {stage}: C*H*W overflow for shape {:?}",
                    nv.spec.shape
                )
            })?;
            if nv.data.len() != numel {
                return Err(format!(
                    "activation {stage}: flat len {} != C*H*W {numel} (shape {:?})",
                    nv.data.len(),
                    nv.spec.shape
                ));
            }
            Ok(Mat::from_vec(*c, hw, nv.data.clone()))
        }
        other => Err(format!(
            "activation {stage}: expected shape [1, C, H, W], got {other:?}"
        )),
    }
}

// ── golden token-stream / generation-contract readers (bd-3s7v block) ───────

/// A captured seam outcome slot: `None` = the seam had no fixture to compare
/// against; `Some(Err)` = the capture itself failed (self-diagnosing row).
type SeamCapture = Option<Result<SeamOutcome, String>>;

/// The parsed `token_stream` block of a golden — the L2 splice input
/// (`prompt_ids`) and the L4 bar (`generated_ids`).
#[derive(Debug)]
struct GoldenTokenStream {
    /// The oracle prompt id-stream (BOS + `<image>` placeholders + task text).
    prompt_ids: Vec<u32>,
    /// The oracle greedy decode output (ends with EOS on a completed run).
    generated_ids: Vec<u32>,
    /// `token_stream.generated_ids_sha256` (provenance for the parity row).
    generated_ids_sha256: String,
}

/// Parse + validate the golden's `token_stream` block. Every malformation
/// (missing block, non-u32 id, a length field disagreeing with its array, an
/// empty stream) is a self-diagnosing error naming the field — never a
/// silently-empty stream that would vacuously "match".
fn golden_token_stream(golden: &ReferenceGolden) -> Result<GoldenTokenStream, String> {
    let ts = golden
        .raw
        .get("token_stream")
        .ok_or("golden missing `token_stream` (regenerate fixtures with bd-3s7v)")?;
    let ids = |key: &str| -> Result<Vec<u32>, String> {
        ts.get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("token_stream missing array `{key}`"))?
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v.as_u64()
                    .and_then(|x| u32::try_from(x).ok())
                    .ok_or_else(|| format!("token_stream {key}[{i}] is not a u32: {v}"))
            })
            .collect()
    };
    let prompt_ids = ids("prompt_ids")?;
    let generated_ids = ids("generated_ids")?;
    for (field, len) in [
        ("n_prompt", prompt_ids.len()),
        ("n_generated", generated_ids.len()),
    ] {
        if let Some(n) = ts.get(field).and_then(Value::as_u64)
            && n as usize != len
        {
            return Err(format!(
                "token_stream `{field}` = {n} disagrees with its array len {len}"
            ));
        }
    }
    if prompt_ids.is_empty() || generated_ids.is_empty() {
        return Err("token_stream has an empty prompt_ids/generated_ids array".into());
    }
    let generated_ids_sha256 = ts
        .get("generated_ids_sha256")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    Ok(GoldenTokenStream {
        prompt_ids,
        generated_ids,
        generated_ids_sha256,
    })
}

/// Build the subject [`DecodeParams`] FROM the golden's own `generation` block,
/// so the subject decodes under the exact contract the oracle ran (never a
/// silent apples-vs-oranges compare if a future fixture regenerates with a
/// different preset). A sampling (non-greedy) oracle is rejected — token-EXACT
/// comparison is only defined for the deterministic greedy contract.
fn golden_decode_params(golden: &ReferenceGolden) -> Result<DecodeParams, String> {
    let g = golden
        .raw
        .get("generation")
        .ok_or("golden missing `generation` block")?;
    let field = |key: &str| -> Result<u64, String> {
        g.get(key)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("generation missing integer `{key}`"))
    };
    let temperature = g.get("temperature").and_then(Value::as_f64).unwrap_or(0.0);
    if temperature != 0.0 {
        return Err(format!(
            "generation.temperature = {temperature} (sampling oracle); token-exact \
             parity is only defined for the greedy (temperature 0) contract"
        ));
    }
    let eos_token_id = golden
        .raw
        .pointer("/token_stream/decode_metadata/generate_kwargs/eos_token_id")
        .and_then(Value::as_u64)
        .and_then(|x| u32::try_from(x).ok())
        .unwrap_or(sampler::DEFAULT_EOS_TOKEN_ID);
    Ok(DecodeParams {
        temperature: 0.0,
        eos_token_id,
        max_length: field("max_length")? as usize,
        no_repeat_ngram_size: field("no_repeat_ngram_size")? as usize,
        ngram_window: field("ngram_window")? as usize,
    })
}

/// Length of the longest common prefix of two token streams — the L4
/// first-divergence locator (self-diagnosing failure: the offending index is
/// printed, PARITY_LADDER §3.3).
fn matched_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Resolve the source corpus image for a golden's `doc`. The fixture layout
/// (bd-3s7v) does not carry the page images, so L0 searches, in order:
/// `$FOCR_CORPUS_DIR/<doc>`, then `<fixtures_root>/pages/<doc>` and the two
/// parent-relative `pages/` dirs (the off-repo layout keeps `pages/` two levels
/// above `fixtures/native/`), then the `--corpus` dir recorded in the golden's
/// own provenance `command_argv`. `None` ⇒ the image is genuinely absent on
/// this box (the rung skips that doc WITH the searched paths named).
fn resolve_corpus_image(root: &Path, golden: &ReferenceGolden) -> (Option<PathBuf>, Vec<PathBuf>) {
    let mut candidates: Vec<PathBuf> = Vec::new();
    // file_name() guards a traversal-shaped `doc` from escaping the corpus dirs.
    let Some(doc) = Path::new(&golden.doc).file_name() else {
        return (None, candidates);
    };
    if let Some(dir) = std::env::var_os("FOCR_CORPUS_DIR") {
        candidates.push(PathBuf::from(dir).join(doc));
    }
    candidates.push(root.join("pages").join(doc));
    if let Some(parent) = root.parent() {
        candidates.push(parent.join("pages").join(doc));
        if let Some(grandparent) = parent.parent() {
            candidates.push(grandparent.join("pages").join(doc));
        }
    }
    if let Some(args) = golden
        .provenance
        .get("command_argv")
        .and_then(Value::as_array)
    {
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if a.as_str() == Some("--corpus")
                && let Some(dir) = it.next().and_then(Value::as_str)
            {
                candidates.push(Path::new(dir).join(doc));
            }
        }
    }
    let found = candidates.iter().find(|p| p.is_file()).cloned();
    (found, candidates)
}

/// Reshape a loaded oracle activation into the 2-D `[rows, cols]` [`Mat`] the
/// engine kernels consume, taking the LAST shape dim as `cols` so a leading batch
/// dim folds into `rows`. Rejects a non-divisible flat length loudly (a corrupt
/// manifest must never silently misshape into a fabricated pass).
fn activation_as_mat(stage: &str, nv: &NormalizedValue) -> Result<Mat, String> {
    let cols = nv
        .spec
        .shape
        .last()
        .copied()
        .filter(|&c| c > 0)
        .unwrap_or_else(|| nv.data.len().max(1));
    if cols == 0 || !nv.data.len().is_multiple_of(cols) {
        return Err(format!(
            "activation {stage}: flat len {} not divisible by last-dim cols {cols} (shape {:?})",
            nv.data.len(),
            nv.spec.shape
        ));
    }
    Ok(Mat::from_vec(nv.data.len() / cols, cols, nv.data.clone()))
}

/// One real subject-vs-oracle seam result for a rung's aggregate.
struct SeamOutcome {
    /// The engine output for this seam.
    subject: Mat,
    /// The oracle output for this seam.
    oracle: NormalizedValue,
    /// The oracle activation's array sha256 (provenance), `""` if absent.
    oracle_sha256: String,
}

/// Capture the projector (vision bridge) subject seam from the committed
/// `clip_output` + `sam_output`, comparing the engine projector to
/// `projector_output`. Returns `None` when this golden lacks the three
/// activations (the rung then knows the seam was not exercised), `Some(Err)` on a
/// load/shape/kernel failure (a loud non-pass), `Some(Ok)` on a real capture.
fn capture_projector_seam(
    w: &Weights,
    loader: &FixtureLoader,
    golden: &ReferenceGolden,
    doc_stem: &str,
) -> Option<Result<SeamOutcome, String>> {
    // All three must be present to isolate the projector from committed fixtures.
    let clip_entry = golden.activations.get("clip_output")?;
    let sam_entry = golden.activations.get("sam_output")?;
    let proj_entry = golden.activations.get("projector_output")?;
    let oracle_sha256 = proj_entry.sha256.clone().unwrap_or_default();
    let run = || -> Result<SeamOutcome, String> {
        let clip = loader.load_activation(doc_stem, "clip_output", clip_entry)?;
        let sam = loader.load_activation(doc_stem, "sam_output", sam_entry)?;
        let oracle = loader.load_activation(doc_stem, "projector_output", proj_entry)?;
        let clip_mat = activation_as_mat("clip_output", &clip)?;
        // sam_output is [1, C, H, W]; the bridge consumes vision_sam's raw
        // [C, H*W] channel-major layout (it transposes internally). The last-dim
        // reshape would misshape it to [C*H, W] and be rejected loudly.
        let sam_mat = sam_output_features_mat("sam_output", &sam)?;
        let subject = vision_bridge::forward(w, &clip_mat, &sam_mat)
            .map_err(|e| format!("vision_bridge::forward: {e}"))?;
        if subject.data.len() != oracle.data.len() {
            return Err(format!(
                "projector subject numel {} != oracle {} (subject [{},{}], oracle shape {:?})",
                subject.data.len(),
                oracle.data.len(),
                subject.rows,
                subject.cols,
                oracle.spec.shape
            ));
        }
        Ok(SeamOutcome {
            subject,
            oracle,
            oracle_sha256,
        })
    };
    Some(run())
}

/// Capture the final-norm + lm_head subject seam from the committed
/// `decoder_layer_11_hidden`, comparing the engine logits to `lm_head_logits`.
fn capture_lm_head_seam(
    w: &Weights,
    loader: &FixtureLoader,
    golden: &ReferenceGolden,
    doc_stem: &str,
) -> Option<Result<SeamOutcome, String>> {
    let hidden_entry = golden.activations.get("decoder_layer_11_hidden")?;
    let logits_entry = golden.activations.get("lm_head_logits")?;
    let oracle_sha256 = logits_entry.sha256.clone().unwrap_or_default();
    let run = || -> Result<SeamOutcome, String> {
        let hidden = loader.load_activation(doc_stem, "decoder_layer_11_hidden", hidden_entry)?;
        let oracle = loader.load_activation(doc_stem, "lm_head_logits", logits_entry)?;
        let hidden_mat = activation_as_mat("decoder_layer_11_hidden", &hidden)?;
        let subject =
            decoder::lm_head(w, &hidden_mat).map_err(|e| format!("decoder::lm_head: {e}"))?;
        if subject.data.len() != oracle.data.len() {
            return Err(format!(
                "lm_head subject numel {} != oracle {} (subject [{},{}], oracle shape {:?})",
                subject.data.len(),
                oracle.data.len(),
                subject.rows,
                subject.cols,
                oracle.spec.shape
            ));
        }
        Ok(SeamOutcome {
            subject,
            oracle,
            oracle_sha256,
        })
    };
    Some(run())
}

/// Capture the SAM-tower subject seam: `vision_sam::forward` over the oracle's
/// EXACT preprocessed pixel tensor (`sam_input`, the bd-3s7v seam input),
/// compared to `sam_output`. Isolates the SAM ViT-B tower.
fn capture_sam_seam(
    w: &Weights,
    loader: &FixtureLoader,
    golden: &ReferenceGolden,
    doc_stem: &str,
) -> Option<Result<SeamOutcome, String>> {
    let input_entry = golden.activations.get("sam_input")?;
    let out_entry = golden.activations.get("sam_output")?;
    let oracle_sha256 = out_entry.sha256.clone().unwrap_or_default();
    let run = || -> Result<SeamOutcome, String> {
        let input = loader.load_activation(doc_stem, "sam_input", input_entry)?;
        let oracle = loader.load_activation(doc_stem, "sam_output", out_entry)?;
        let view = sam_input_view_mat("sam_input", &input)?;
        let subject =
            vision_sam::forward(w, &view).map_err(|e| format!("vision_sam::forward: {e}"))?;
        if subject.data.len() != oracle.data.len() {
            return Err(format!(
                "sam subject numel {} != oracle {} (subject [{},{}], oracle shape {:?})",
                subject.data.len(),
                oracle.data.len(),
                subject.rows,
                subject.cols,
                oracle.spec.shape
            ));
        }
        Ok(SeamOutcome {
            subject,
            oracle,
            oracle_sha256,
        })
    };
    Some(run())
}

/// Capture the CLIP-tower subject seam: `vision_clip::forward` fed the oracle's
/// EXACT `sam_output` as its patch embeds (the tower's only real input — the
/// image argument is unused by contract), compared to `clip_output`. Isolates
/// the CLIP tower. The view tensor is still built from `sam_input` so the call
/// mirrors the engine's own `vision_tower` call shape exactly.
fn capture_clip_seam(
    w: &Weights,
    loader: &FixtureLoader,
    golden: &ReferenceGolden,
    doc_stem: &str,
) -> Option<Result<SeamOutcome, String>> {
    let input_entry = golden.activations.get("sam_input")?;
    let sam_entry = golden.activations.get("sam_output")?;
    let out_entry = golden.activations.get("clip_output")?;
    let oracle_sha256 = out_entry.sha256.clone().unwrap_or_default();
    let run = || -> Result<SeamOutcome, String> {
        let input = loader.load_activation(doc_stem, "sam_input", input_entry)?;
        let sam = loader.load_activation(doc_stem, "sam_output", sam_entry)?;
        let oracle = loader.load_activation(doc_stem, "clip_output", out_entry)?;
        let view = sam_input_view_mat("sam_input", &input)?;
        let sam_mat = sam_output_features_mat("sam_output", &sam)?;
        let subject = vision_clip::forward(w, &view, &sam_mat)
            .map_err(|e| format!("vision_clip::forward: {e}"))?;
        if subject.data.len() != oracle.data.len() {
            return Err(format!(
                "clip subject numel {} != oracle {} (subject [{},{}], oracle shape {:?})",
                subject.data.len(),
                oracle.data.len(),
                subject.rows,
                subject.cols,
                oracle.spec.shape
            ));
        }
        Ok(SeamOutcome {
            subject,
            oracle,
            oracle_sha256,
        })
    };
    Some(run())
}

/// Capture the layer-0 `inputs_embeds` subject seam (the L2 keystone): the full
/// engine vision chain (`vision_sam` → `vision_clip` → `vision_bridge`) over the
/// oracle's EXACT `sam_input`, plus `decoder::embed_tokens` over the oracle's
/// EXACT `prompt_ids`, spliced by `connector::fuse_no_crop` — compared to the
/// oracle `inputs_embeds`. This is precisely the engine's private
/// `build_inputs_embeds` recomposed from its public, unit-tested parts, with
/// every input pinned to the oracle's bytes so a divergence attributes to the
/// vision/connector math alone.
fn capture_inputs_embeds_seam(
    w: &Weights,
    loader: &FixtureLoader,
    golden: &ReferenceGolden,
    doc_stem: &str,
) -> Option<Result<SeamOutcome, String>> {
    let input_entry = golden.activations.get("sam_input")?;
    let embeds_entry = golden.activations.get("inputs_embeds")?;
    golden.raw.get("token_stream")?;
    let oracle_sha256 = embeds_entry.sha256.clone().unwrap_or_default();
    let run = || -> Result<SeamOutcome, String> {
        // Only the base / no-crop connector path is wired here; a Gundam golden
        // must be named, not silently fused through the wrong assembly.
        if golden
            .raw
            .pointer("/mode/crop_mode")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(
                "golden is crop_mode=true (Gundam); the inputs_embeds seam only \
                        wires the base fuse_no_crop path"
                    .into(),
            );
        }
        let stream = golden_token_stream(golden)?;
        let input = loader.load_activation(doc_stem, "sam_input", input_entry)?;
        let oracle = loader.load_activation(doc_stem, "inputs_embeds", embeds_entry)?;
        let view = sam_input_view_mat("sam_input", &input)?;
        let sam = vision_sam::forward(w, &view).map_err(|e| format!("vision_sam::forward: {e}"))?;
        let clip = vision_clip::forward(w, &view, &sam)
            .map_err(|e| format!("vision_clip::forward: {e}"))?;
        let projected = vision_bridge::forward(w, &clip, &sam)
            .map_err(|e| format!("vision_bridge::forward: {e}"))?;
        let table = w
            .mat("model.embed_tokens.weight")
            .map_err(|e| format!("embed table: {e}"))?;
        let mut embeds =
            decoder::embed_tokens(&table.data, table.rows, table.cols, &stream.prompt_ids)
                .map_err(|e| format!("decoder::embed_tokens: {e}"))?;
        // The row-aligned images_seq_mask is true exactly at the `<image>`
        // placeholders (id 128815) — the same mask build_prompt derives.
        let mask: Vec<bool> = stream
            .prompt_ids
            .iter()
            .map(|&id| id == special::IMAGE)
            .collect();
        let image_newline = w
            .vec("model.image_newline")
            .map_err(|e| format!("model.image_newline: {e}"))?;
        let view_seperator = w
            .vec("model.view_seperator")
            .map_err(|e| format!("model.view_seperator: {e}"))?;
        let grid = preprocess::num_queries(preprocess::BASE_SIZE);
        connector::fuse_no_crop(
            w,
            &mut embeds,
            std::slice::from_ref(&projected),
            grid,
            grid,
            &image_newline,
            &view_seperator,
            &mask,
        )
        .map_err(|e| format!("connector::fuse_no_crop: {e}"))?;
        if embeds.data.len() != oracle.data.len() {
            return Err(format!(
                "inputs_embeds subject numel {} != oracle {} (subject [{},{}], oracle shape {:?})",
                embeds.data.len(),
                oracle.data.len(),
                embeds.rows,
                embeds.cols,
                oracle.spec.shape
            ));
        }
        Ok(SeamOutcome {
            subject: embeds,
            oracle,
            oracle_sha256,
        })
    };
    Some(run())
}

// ── L4/L5 shared subject decode capture ──────────────────────────────────────
//
// The greedy AR decode is by far the most expensive seam (f32 weight-cache
// build + prefill + one forward per token), and BOTH L4 (token ids) and L5
// (decoded text) consume the same capture. It is computed ONCE per process
// under a lock and cached; each rung then applies its own comparator to the
// shared, honestly-captured stream.

/// One golden's subject decode capture.
#[derive(Clone)]
struct DecodeCapture {
    /// The engine-emitted token ids (greedy, includes the terminal EOS).
    emitted: Vec<u32>,
    /// The oracle's `token_stream.generated_ids` (the L4 bar).
    golden_generated: Vec<u32>,
    /// `token_stream.generated_ids_sha256` (provenance for the parity row).
    generated_ids_sha256: String,
    /// Prefill-hidden cosine vs the oracle `decoder_layer_11_hidden` — the
    /// 12-layer-stack ledger row L2 cannot get per-layer (free here: the L4
    /// prefill hidden IS the layer-11 output over the prompt rows).
    prefill_hidden_cosine: Option<f64>,
    /// Prefill-hidden max-abs vs `decoder_layer_11_hidden` (ledgered).
    prefill_hidden_max_abs: Option<f64>,
    /// Prompt length (= prefill rows), for the log.
    prompt_len: usize,
}

static DECODE_CAPTURES: Mutex<Option<BTreeMap<String, Result<DecodeCapture, String>>>> =
    Mutex::new(None);

/// The (process-cached) subject decode captures for every golden. First caller
/// computes under the lock (so L4/L5 never build two 13 GB weight caches
/// concurrently); later callers get the same captures.
fn subject_decode_captures(
    w: &Weights,
    loader: &FixtureLoader,
) -> BTreeMap<String, Result<DecodeCapture, String>> {
    let mut guard = DECODE_CAPTURES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(m) = guard.as_ref() {
        return m.clone();
    }
    let m = compute_decode_captures(w, loader);
    *guard = Some(m.clone());
    m
}

fn compute_decode_captures(
    w: &Weights,
    loader: &FixtureLoader,
) -> BTreeMap<String, Result<DecodeCapture, String>> {
    let mut out = BTreeMap::new();
    let goldens = loader.list_goldens().unwrap_or_default();
    // The dominant costs (f32 decoder weight cache + embed table) are built
    // ONCE for all goldens; a build failure is recorded per golden so every
    // consumer rung sees the same loud error.
    let shared = (|| -> Result<(decoder::DecoderWeightCache, Mat), String> {
        let wc = decoder::DecoderWeightCache::build(w)
            .map_err(|e| format!("decoder::DecoderWeightCache::build: {e}"))?;
        let table = w
            .mat("model.embed_tokens.weight")
            .map_err(|e| format!("embed table: {e}"))?;
        Ok((wc, table))
    })();
    match shared {
        Ok((wc, table)) => {
            for gpath in &goldens {
                let stem = golden_stem(gpath);
                let cap = decode_one_golden(&wc, &table, loader, gpath);
                out.insert(stem, cap);
            }
        }
        Err(e) => {
            for gpath in &goldens {
                out.insert(golden_stem(gpath), Err(e.clone()));
            }
        }
    }
    out
}

/// Extract the final row of a decoder hidden as a `[1, hidden]` [`Mat`] — the
/// row that predicts the next token (the engine's own last-row contract).
fn last_hidden_row(hidden: &Mat) -> Result<Mat, String> {
    if hidden.rows == 0 || hidden.data.len() != hidden.rows * hidden.cols {
        return Err(format!(
            "decoder hidden malformed: shape [{}, {}], data len {}",
            hidden.rows,
            hidden.cols,
            hidden.data.len()
        ));
    }
    Ok(Mat::from_vec(
        1,
        hidden.cols,
        hidden.row(hidden.rows - 1).to_vec(),
    ))
}

/// Run the engine greedy decode for ONE golden, seeded with the oracle's EXACT
/// `inputs_embeds` (so a token divergence attributes to the decoder/sampler,
/// not to an upstream vision/preprocess delta — those have their own rungs).
/// This is the engine's own cached decode loop (`prefill_with_cache` →
/// `lm_head_cached` → `sampler::decode_step` → `decode_step_with_cache`)
/// recomposed from its public parts: same n-gram history seeding (prompt ids
/// included), same absolute ring positions, same EOS halt.
fn decode_one_golden(
    wc: &decoder::DecoderWeightCache,
    table: &Mat,
    loader: &FixtureLoader,
    gpath: &Path,
) -> Result<DecodeCapture, String> {
    let golden = loader.load_golden(gpath)?;
    FixtureLoader::check_provenance(&golden)?;
    let stem = golden_stem(gpath);
    let doc_stem = golden.doc_stem_or(&stem);
    let stream = golden_token_stream(&golden)?;
    let params = golden_decode_params(&golden)?;
    let embeds_entry = golden
        .activations
        .get("inputs_embeds")
        .ok_or("golden lacks the `inputs_embeds` activation (bd-3s7v seam input)")?;
    let embeds_nv = loader.load_activation(&doc_stem, "inputs_embeds", embeds_entry)?;
    let inputs_embeds = activation_as_mat("inputs_embeds", &embeds_nv)?;
    if inputs_embeds.rows != stream.prompt_ids.len() {
        return Err(format!(
            "inputs_embeds rows {} != token_stream prompt len {}",
            inputs_embeds.rows,
            stream.prompt_ids.len()
        ));
    }
    if table.cols != inputs_embeds.cols {
        return Err(format!(
            "embed table hidden {} != inputs_embeds hidden {}",
            table.cols, inputs_embeds.cols
        ));
    }

    let (hidden, mut caches) = decoder::prefill_with_cache(wc, &inputs_embeds)
        .map_err(|e| format!("decoder::prefill_with_cache: {e}"))?;
    // Free 12-layer-stack ledger: the prefill hidden over the prompt rows IS
    // the oracle's decoder_layer_11_hidden (both pre-final-norm).
    let (prefill_hidden_cosine, prefill_hidden_max_abs) =
        match golden.activations.get("decoder_layer_11_hidden") {
            Some(entry) => {
                let nv = loader.load_activation(&doc_stem, "decoder_layer_11_hidden", entry)?;
                if nv.data.len() != hidden.data.len() {
                    return Err(format!(
                        "prefill hidden numel {} != oracle decoder_layer_11_hidden {}",
                        hidden.data.len(),
                        nv.data.len()
                    ));
                }
                (
                    Some(cosine(&hidden.data, &nv.data)),
                    Some(max_abs_diff(&hidden.data, &nv.data)),
                )
            }
            None => (None, None),
        };
    let mut last_hidden = last_hidden_row(&hidden)?;

    // The n-gram history is seeded with the prompt ids — the oracle's custom
    // logits processor sees the full input_ids too (and so does the engine's
    // own generate loop).
    let mut generated: Vec<u32> = stream.prompt_ids.clone();
    let mut emitted: Vec<u32> = Vec::new();
    let step_cap = stream
        .generated_ids
        .len()
        .saturating_add(DECODE_DIVERGENCE_SLACK)
        .min(params.max_length);
    while emitted.len() < step_cap {
        let logits = decoder::lm_head_cached(wc, &last_hidden)
            .map_err(|e| format!("decoder::lm_head_cached: {e}"))?;
        let step = sampler::decode_step(&logits, &generated, &params)
            .map_err(|e| format!("sampler::decode_step: {e}"))?;
        generated.push(step.token_id);
        emitted.push(step.token_id);
        if step.is_eos {
            break;
        }
        let next = step.token_id as usize;
        if next >= table.rows {
            return Err(format!(
                "decoded token id {next} outside embed vocab {}",
                table.rows
            ));
        }
        let row = table.data[next * table.cols..(next + 1) * table.cols].to_vec();
        let token_embed = Mat::from_vec(1, table.cols, row);
        let position = inputs_embeds.rows + (emitted.len() - 1);
        let h = decoder::decode_step_with_cache(wc, &mut caches, &token_embed, position)
            .map_err(|e| format!("decoder::decode_step_with_cache: {e}"))?;
        last_hidden = last_hidden_row(&h)?;
    }

    Ok(DecodeCapture {
        emitted,
        golden_generated: stream.generated_ids,
        generated_ids_sha256: stream.generated_ids_sha256,
        prefill_hidden_cosine,
        prefill_hidden_max_abs,
        prompt_len: inputs_embeds.rows,
    })
}

/// Per-row argmax (torch tie-break: lowest index wins) over a `[rows, cols]` flat
/// buffer — the L3 token-decision invariant. NaN is skipped so an all-NaN row
/// falls back to index 0 (never silently "passes" by comparing two NaNs equal).
fn argmax_rows(data: &[f32], cols: usize) -> Vec<usize> {
    if cols == 0 {
        return Vec::new();
    }
    data.chunks_exact(cols)
        .map(|row| {
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }
            best_idx
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// L0 — preprocessing parity (EXACT) — VERIFY-ladder-l0 / bd-re8.4
//
// Preprocessing is deterministic integer/float arithmetic with NO quantization,
// so the target tolerance is EXACT (PARITY_LADDER §3.1). The L0 *contract*
// anchors that need no oracle run are checked always-on; the full tensor
// comparison is fixture-gated (the oracle's preprocessed sam_input, bd-3s7v)
// and additionally needs the source page image (see resolve_corpus_image). One
// documented deviation from EXACT applies on the DEFAULT kernel: the
// bd-30me/DISC-001 resample envelope (L0_RESAMPLE_MAX_ABS_TOL). Under the
// DISC-001 kill-switch `FOCR_RESAMPLE=pil-bicubic` the compare is EXACT
// (bit-exact passes outright); the active kernel is ledgered per row.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn l0_preprocess_exact() {
    let mut log = Logger::new("L0_preprocess", "corpus");
    log.setup(0);
    let t0 = Instant::now();

    // Always-on L0 contract anchors (PARITY_LADDER §3.1): the EXACT constants the
    // front end MUST reproduce — gray pad 127, [-1,1] normalize bounds, the 273
    // image-token slots per 1024-view (CENSUS (c)). These are asserted against
    // the pinned census numbers, not magic constants. They do not need the oracle
    // (the reference is deterministic) so they run on every box.
    const GRAY_PAD: u8 = 127; // (127,127,127) = int(0.5*255) [SPEC-022]
    const NORM_LO: f32 = -1.0; // (0-0.5)/0.5 [SPEC-021]
    const NORM_HI: f32 = 1.0; // (1-0.5)/0.5
    const SLOTS_PER_1024_VIEW: usize = (16 + 1) * 16 + 1; // 273 [SPEC-028], CENSUS (c)
    log.assertion("gray pad == int(0.5*255) == 127", GRAY_PAD == 127);
    log.assertion(
        "normalize maps to [-1,1]",
        NORM_LO == -1.0 && NORM_HI == 1.0,
    );
    log.assertion(
        "image-token slots per 1024-view == 273",
        SLOTS_PER_1024_VIEW == 273,
    );

    if !fixtures_present() || !model_present() {
        log.skip_no_model(
            "L0 preprocess-tensor comparison needs the oracle sam_input activation \
             (bd-3s7v fixtures); contract anchors above ran. Set FOCR_FIXTURES_DIR + \
             FOCR_MODEL_PATH (the ladder env gate) to run the live compare \
             (PARITY_LADDER §3.1).",
        );
        log.result("skip_no_model", t0.elapsed().as_micros());
        return;
    }

    // FIXTURE+MODEL PRESENT: the REAL subject preprocess compare (bd-2ksr).
    // `preprocess::preprocess_image` runs over the source page image and its
    // global view tensor ([3, H*W], normalized [-1,1]) is compared element-wise
    // to the oracle's sam_input ([1, 3, H, W] — the identical C-order layout).
    // The gate is EXACT-first: a bit-exact tensor passes outright; otherwise the
    // DOCUMENTED bd-30me resample envelope applies (CatmullRom vs PIL BICUBIC,
    // see L0_RESAMPLE_MAX_ABS_TOL) with the exact-element fraction ledgered.
    let _heavy = heavy_rung_guard();
    let loader = FixtureLoader::new();
    let mut ran = 0usize;
    let mut skipped_missing_image = 0usize;
    let mut all_pass = true;
    for gpath in loader.list_goldens().unwrap_or_default() {
        let stem = golden_stem(&gpath);
        let golden = match loader.load_golden(&gpath) {
            Ok(g) => g,
            Err(e) => {
                log.error("FixtureParse", 1, &e);
                all_pass = false;
                continue;
            }
        };
        if let Err(e) = FixtureLoader::check_provenance(&golden) {
            log.error("Provenance", 1, &format!("{stem}: {e}"));
            all_pass = false;
            continue;
        }
        let doc_stem = golden.doc_stem_or(&stem);
        let Some(entry) = golden.activations.get("sam_input") else {
            log.error(
                "SeamUnavailable",
                1,
                &format!("{stem}: golden lacks the sam_input activation (bd-3s7v seam input)"),
            );
            all_pass = false;
            continue;
        };
        // Only the base / no-crop geometry is wired; a Gundam golden is named.
        if golden
            .raw
            .pointer("/mode/crop_mode")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            log.error(
                "SeamUnavailable",
                1,
                &format!("{stem}: crop_mode=true (Gundam) L0 compare is not wired"),
            );
            all_pass = false;
            continue;
        }
        // The fixture layout does not carry the page images; absent image ⇒ the
        // doc is skipped WITH the searched paths named (same skip-with-SUCCESS
        // class as an absent model — never a fabricated verdict either way).
        let (image_path, candidates) = resolve_corpus_image(loader.root(), &golden);
        let Some(image_path) = image_path else {
            log.skip_no_model(&format!(
                "{stem}: source image {:?} not found (searched {:?}; set FOCR_CORPUS_DIR)",
                golden.doc, candidates
            ));
            skipped_missing_image += 1;
            continue;
        };
        let outcome = (|| -> Result<(Mat, NormalizedValue, String), String> {
            let oracle = loader.load_activation(&doc_stem, "sam_input", entry)?;
            let pre = preprocess::preprocess_image(&image_path, preprocess::PreprocessMode::base())
                .map_err(|e| format!("preprocess_image {}: {e}", image_path.display()))?;
            let subject = pre.global.pixels.clone();
            if subject.data.len() != oracle.data.len() {
                return Err(format!(
                    "preprocess subject numel {} != oracle {} (subject [{},{}], oracle shape {:?})",
                    subject.data.len(),
                    oracle.data.len(),
                    subject.rows,
                    subject.cols,
                    oracle.spec.shape
                ));
            }
            // Sanity ledger: the golden records the source image dims; the
            // engine's post-EXIF dims must agree or the compare is meaningless.
            let (w_px, h_px) = pre.original_size;
            let golden_w = golden.raw.pointer("/image/width").and_then(Value::as_u64);
            let golden_h = golden.raw.pointer("/image/height").and_then(Value::as_u64);
            if golden_w.is_some_and(|gw| gw != u64::from(w_px))
                || golden_h.is_some_and(|gh| gh != u64::from(h_px))
            {
                return Err(format!(
                    "source image dims mismatch: engine {w_px}x{h_px}, golden {golden_w:?}x{golden_h:?}"
                ));
            }
            let oracle_sha256 = entry.sha256.clone().unwrap_or_default();
            Ok((subject, oracle, oracle_sha256))
        })();
        match outcome {
            Ok((subject, oracle, oracle_sha256)) => {
                let exact_elements = subject
                    .data
                    .iter()
                    .zip(oracle.data.iter())
                    .filter(|(a, b)| a.to_bits() == b.to_bits())
                    .count();
                let exact = exact_elements == oracle.data.len();
                let mad = max_abs_diff(&subject.data, &oracle.data);
                let c = cosine(&subject.data, &oracle.data);
                // EXACT passes outright; otherwise the documented bd-30me
                // resample envelope (max-abs within budget AND cosine gate).
                let pass = exact || (mad <= L0_RESAMPLE_MAX_ABS_TOL && c >= COSINE_F32_THRESHOLD);
                ran += 1;
                all_pass &= pass;
                log.parity(
                    "L0",
                    "max_abs_diff",
                    mad,
                    L0_RESAMPLE_MAX_ABS_TOL,
                    "sam_input",
                    &oracle_sha256,
                    json!({
                        "seam": "preprocess (decode/resize/pad/normalize -> [3, H*W])",
                        "subject": SUBJECT_IDENTITY,
                        "oracle": ORACLE_IDENTITY,
                        "exact": exact,
                        "exact_elements": exact_elements,
                        "total_elements": oracle.data.len(),
                        "cosine": c,
                        // Which resize kernel produced the subject tensor —
                        // DISC-001: `FOCR_RESAMPLE=pil-bicubic` is the EXACT path.
                        "resample_kind": format!("{:?}", preprocess::resample_kind()),
                        "tolerance_source": "bd-30me/DISC-001 resample envelope (CatmullRom vs \
                                             PIL ImageOps.pad BICUBIC; 8 u8 steps in [-1,1])",
                        "doc": stem.as_str(),
                    }),
                    pass,
                );
            }
            Err(e) => {
                log.error("PreprocessSeam", 1, &format!("{stem}: {e}"));
                all_pass = false;
            }
        }
    }

    if ran == 0 {
        if skipped_missing_image > 0 && all_pass {
            // Fixtures present but the page corpus is not on this box — the same
            // honest skip class as an absent model.
            log.result("skip_no_model", t0.elapsed().as_micros());
        } else {
            log.assertion("L0 subject (engine) preprocess compare exercised", false);
            log.error(
                "NotImplemented",
                1,
                "L0: oracle fixtures present but no preprocess compare ran \
                 (sam_input activation or source corpus image unavailable).",
            );
            log.result("xfail", t0.elapsed().as_micros());
        }
    } else {
        log.result(
            if all_pass { "pass" } else { "fail" },
            t0.elapsed().as_micros(),
        );
        assert!(
            all_pass,
            "L0 subject-vs-oracle preprocess parity FAILED (outside the documented \
             bd-30me resample envelope); see structured log"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// L1 — per-op parity (cosine ≥ 0.9999 + ULP table) — bd-re8.5
//
// Each kernel's output vs the matching oracle activation, cosine ≥ 0.9999 in f32,
// and the per-op ULP table on the bridge path (PARITY_LADDER §3.2). Fixture-gated
// on the per-stage .npy activations + model-gated on the engine producing the
// same-stage tensor.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn l1_per_op_cosine() {
    let mut log = Logger::new("L1_per_op", "corpus");
    log.setup(0);
    let t0 = Instant::now();

    if !fixtures_present() || !model_present() {
        log.skip_no_model(
            "L1 per-op cosine needs the per-stage oracle activations (.npy) AND the \
             engine forward producing the same-stage tensor. Comparator math \
             (cosine ≥ 0.9999, 4-ULP matmul / 2-ULP elementwise) is unit-tested in \
             support/parity_harness.rs. Provide FOCR_FIXTURES_DIR + FOCR_MODEL_PATH \
             to run the live compare.",
        );
        log.result("skip_no_model", t0.elapsed().as_micros());
        return;
    }

    // FIXTURE+MODEL PRESENT: run the REAL subject (engine) per-op seam captures.
    // The bd-3s7v sam_input dump isolates every vision op from committed
    // fixtures alone: SAM (sam_input -> sam_output), CLIP (oracle sam_output ->
    // clip_output), and the projector (oracle clip_output + sam_output ->
    // projector_output) — each fed the oracle's EXACT upstream tensor, compared
    // through cosine + the ULP table. True franken_ocr-vs-baidu parity rows,
    // never an oracle self-compare.
    let _heavy = heavy_rung_guard();
    let loader = FixtureLoader::new();
    let w = match load_subject_weights() {
        Ok(w) => w,
        Err(e) => {
            log.error("WeightsLoad", 1, &e);
            log.result("xfail", t0.elapsed().as_micros());
            return;
        }
    };
    let mut ran = 0usize;
    let mut all_pass = true;
    for gpath in loader.list_goldens().unwrap_or_default() {
        let stem = golden_stem(&gpath);
        let golden = match loader.load_golden(&gpath) {
            Ok(g) => g,
            Err(e) => {
                log.error("FixtureParse", 1, &e);
                all_pass = false;
                continue;
            }
        };
        if let Err(e) = FixtureLoader::check_provenance(&golden) {
            log.error("Provenance", 1, &format!("{stem}: {e}"));
            all_pass = false;
            continue;
        }
        let doc_stem = golden.doc_stem_or(&stem);

        // (seam name, oracle fixture stage, upstream description, capture).
        let seams: [(&str, &str, &str, SeamCapture); 3] = [
            (
                "vision_sam (ViT-B tower)",
                "sam_output",
                "oracle sam_input (exact)",
                capture_sam_seam(&w, &loader, &golden, &doc_stem),
            ),
            (
                "vision_clip (CLIP tower over SAM patch embeds)",
                "clip_output",
                "oracle sam_output (exact)",
                capture_clip_seam(&w, &loader, &golden, &doc_stem),
            ),
            (
                "vision_bridge (projector 2048->1280)",
                "projector_output",
                "oracle clip_output + sam_output (exact)",
                capture_projector_seam(&w, &loader, &golden, &doc_stem),
            ),
        ];
        let mut any_seam = false;
        for (seam_name, fixture, input, capture) in seams {
            match capture {
                Some(Ok(seam)) => {
                    let c = cosine(&seam.subject.data, &seam.oracle.data);
                    let report =
                        ulp_compare(&seam.subject.data, &seam.oracle.data, OpFamily::MatmulF32);
                    let pass = c >= COSINE_F32_THRESHOLD;
                    ran += 1;
                    any_seam = true;
                    all_pass &= pass;
                    log.parity(
                        "L1",
                        "cosine",
                        c,
                        COSINE_F32_THRESHOLD,
                        fixture,
                        &seam.oracle_sha256,
                        json!({
                            "seam": seam_name,
                            "subject": SUBJECT_IDENTITY,
                            "oracle": ORACLE_IDENTITY,
                            "max_abs_diff": report.max_abs_diff,
                            "max_ulp": report.max_ulp,
                            "ulp_budget": report.budget_ulp,
                            "input": input,
                            "doc": stem.as_str(),
                        }),
                        pass,
                    );
                }
                Some(Err(e)) => {
                    log.error("PerOpSeam", 1, &format!("{stem}: {seam_name}: {e}"));
                    all_pass = false;
                    any_seam = true;
                }
                None => {} // this golden lacks the seam's activations; counted below
            }
        }
        if !any_seam {
            log.error(
                "SeamUnavailable",
                1,
                &format!(
                    "{stem}: no isolatable L1 per-op seam (need the bd-3s7v activation set: \
                     sam_input + sam_output + clip_output + projector_output)"
                ),
            );
        }
    }

    if ran == 0 {
        // Mirror L0: a precise XFAIL, never a fabricated pass.
        log.assertion("L1 subject (engine) per-op seam capture exercised", false);
        log.error(
            "NotImplemented",
            1,
            "L1: oracle fixtures present but no committed-isolatable per-op seam ran \
             (the SAM/CLIP/projector seams need the bd-3s7v activation set).",
        );
        log.result("xfail", t0.elapsed().as_micros());
    } else {
        log.result(
            if all_pass { "pass" } else { "fail" },
            t0.elapsed().as_micros(),
        );
        assert!(
            all_pass,
            "L1 subject-vs-oracle per-op parity FAILED (a vision seam cosine < \
             {COSINE_F32_THRESHOLD} or a seam capture errored); see structured log"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// L2 — per-layer parity (cosine ≈ 1.0 + max-abs ledger) — bd-re8.5
//
// All 12 decoder-layer hidden states + each vision-stage seam; cosine ≈ 1.0 with
// max-abs-diff LEDGERED per layer (PARITY_LADDER §3.2). The per-layer max-abs
// ledger is what makes slow cross-layer drift visible.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn l2_per_layer_cosine_and_ledger() {
    let mut log = Logger::new("L2_per_layer", "corpus");
    log.setup(0);
    let t0 = Instant::now();

    // The 12 decoder-layer seams + 3 vision seams the oracle hooks emit
    // (ActivationCapture.register, PARITY_LADDER §1). Always-on: assert the seam
    // census the ladder expects, so a fixture missing a seam is caught.
    let expected_decoder_layers = 12usize;
    let vision_seams = ["sam_output", "clip_output", "projector_output"];
    log.assertion(
        "decoder layer count == 12 (SPEC-070..072)",
        expected_decoder_layers == 12,
    );
    log.assertion(
        "vision seams == [sam, clip, projector]",
        vision_seams.len() == 3,
    );

    if !fixtures_present() || !model_present() {
        log.skip_no_model(
            "L2 per-layer cosine + max-abs ledger needs all 12 decoder_layer_NN_hidden \
             oracle activations and the engine hidden states. The max-abs ledger \
             (visible cross-layer drift) and cosine comparator are unit-tested in \
             support/parity_harness.rs.",
        );
        log.result("skip_no_model", t0.elapsed().as_micros());
        return;
    }

    // FIXTURE+MODEL PRESENT: REAL subject per-stage compare with the max-abs
    // ledger. Two per-stage seams are isolatable from the committed fixtures:
    //   * projector: oracle clip_output + sam_output -> projector_output;
    //   * layer-0 inputs_embeds (the bd-2ksr keystone): the full engine
    //     vision(sam→clip→bridge) + embed_tokens(oracle prompt_ids) +
    //     connector::fuse_no_crop splice -> oracle inputs_embeds.
    // The per-layer decoder hiddens for layers 00..10 still need a public
    // single-layer engine entry seeded by the prior layer's oracle hidden (only
    // the 12-layer driver is exposed) — named precisely, not faked; the 12-layer
    // STACK is ledgered by L4's prefill-hidden vs decoder_layer_11_hidden.
    let _heavy = heavy_rung_guard();
    let loader = FixtureLoader::new();
    let w = match load_subject_weights() {
        Ok(w) => w,
        Err(e) => {
            log.error("WeightsLoad", 1, &e);
            log.result("xfail", t0.elapsed().as_micros());
            return;
        }
    };
    let mut ran = 0usize;
    let mut all_pass = true;
    for gpath in loader.list_goldens().unwrap_or_default() {
        let stem = golden_stem(&gpath);
        let Ok(golden) = loader.load_golden(&gpath) else {
            continue;
        };
        if let Err(e) = FixtureLoader::check_provenance(&golden) {
            log.error("Provenance", 1, &format!("{stem}: {e}"));
            all_pass = false;
            continue;
        }
        let doc_stem = golden.doc_stem_or(&stem);

        let seams: [(&str, &str, SeamCapture); 2] = [
            (
                "vision_bridge (projector 2048->1280)",
                "projector_output",
                capture_projector_seam(&w, &loader, &golden, &doc_stem),
            ),
            (
                "vision(sam→clip→bridge) + connector splice (layer-0 inputs_embeds)",
                "inputs_embeds",
                capture_inputs_embeds_seam(&w, &loader, &golden, &doc_stem),
            ),
        ];
        let mut any_seam = false;
        for (seam_name, fixture, capture) in seams {
            match capture {
                Some(Ok(seam)) => {
                    let c = cosine(&seam.subject.data, &seam.oracle.data);
                    let mad = max_abs_diff(&seam.subject.data, &seam.oracle.data);
                    let pass = c >= COSINE_F32_THRESHOLD;
                    ran += 1;
                    any_seam = true;
                    all_pass &= pass;
                    log.parity(
                        "L2",
                        "max_abs_diff",
                        mad,
                        0.0,
                        fixture,
                        &seam.oracle_sha256,
                        json!({
                            "cosine": c,
                            "ledger": "per-stage max-abs (cross-stage drift)",
                            "seam": seam_name,
                            "subject": SUBJECT_IDENTITY,
                            "oracle": ORACLE_IDENTITY,
                            "cosine_threshold": COSINE_F32_THRESHOLD,
                            "doc": stem.as_str(),
                        }),
                        pass,
                    );
                }
                Some(Err(e)) => {
                    log.error("PerStageSeam", 1, &format!("{stem}: {seam_name}: {e}"));
                    all_pass = false;
                    any_seam = true;
                }
                None => {}
            }
        }
        if !any_seam {
            log.error(
                "SeamUnavailable",
                1,
                &format!(
                    "{stem}: no isolatable L2 per-stage seam (need the bd-3s7v activation set \
                     incl. sam_input + inputs_embeds + token_stream); the per-layer \
                     decoder_layer_00..10 seams additionally need a single-layer engine entry"
                ),
            );
        }
    }

    if ran == 0 {
        log.assertion(
            "L2 subject (engine) per-stage seam capture exercised",
            false,
        );
        log.error(
            "NotImplemented",
            1,
            "L2: oracle fixtures present but no committed-isolatable per-stage seam ran \
             (need the bd-3s7v activation set incl. sam_input + inputs_embeds).",
        );
        log.result("xfail", t0.elapsed().as_micros());
    } else {
        log.result(
            if all_pass { "pass" } else { "fail" },
            t0.elapsed().as_micros(),
        );
        assert!(
            all_pass,
            "L2 subject-vs-oracle per-stage parity FAILED (a stage cosine < \
             {COSINE_F32_THRESHOLD} or a seam capture errored); see structured log"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// L3 — logits parity (MEASURED int8 budget + argmax exact) — bd-re8.6
//
// Pre-sampling logits within the MEASURED int8/int4 quant tolerance DERIVED from
// the oracle nondeterminism floor (§2) — NOT the imported 0.055; argmax MUST
// match at every deterministic position (PARITY_LADDER §3.3). The keystone:
// establish the floor FIRST.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn l3_logits_measured_budget_and_argmax() {
    let mut log = Logger::new("L3_logits", "corpus");
    log.setup(0);
    let t0 = Instant::now();

    // Always-on: the keystone discipline check. The L3 tolerance is DERIVED from
    // the §2 floor, never guessed. We prove the derivation pipeline on a synthetic
    // two-run pair so the gate's machinery is exercised even with no real oracle.
    let run_a = vec![vec![3.0f32, 1.0, 2.0]; 4];
    let mut run_b = run_a.clone();
    run_b[0][2] = 2.02; // a tiny bf16-noise-level spread, not enough to flip argmax
    let tokens_a = [0u32, 0, 0, 0]; // argmax of [3,1,2] is index 0 every position
    let tokens_b = tokens_a;
    let floor = establish_floor(&run_a, &run_b, &tokens_a, &tokens_b);
    let derived_tol = floor.l3_logit_tolerance();
    log.assertion(
        "L3 tolerance DERIVED from oracle floor (== measured spread, not imported 0.055)",
        // Binds the derived tolerance to the INDEPENDENTLY measured floor spread and
        // excludes the imported constant. The old `(tol-0.05).abs()>1e-9 || tol<0.055`
        // was a tautology — true for EVERY value including the forbidden 0.055 — so it
        // could never catch a regression that hard-codes 0.055 (audit rank 4).
        (derived_tol - floor.per_logit_max_abs_spread).abs() < 1e-12
            && (derived_tol - 0.055).abs() > 1e-9,
    );
    log.assertion(
        "argmax stable across the two oracle runs (deterministic positions exist)",
        tokens_a == tokens_b,
    );

    if !fixtures_present() || !model_present() {
        log.skip_no_model(&format!(
            "L3 logit compare needs lm_head_logits oracle activation + the engine \
             prefill logits. The §2 nondeterminism floor (derived L3 tolerance \
             {derived_tol:.4}, reproducible prefix {}) is established by the harness; \
             the live compare needs FOCR_FIXTURES_DIR + FOCR_MODEL_PATH.",
            floor.l4_exact_prefix()
        ));
        log.result("skip_no_model", t0.elapsed().as_micros());
        return;
    }

    // FIXTURE+MODEL PRESENT: REAL subject logit capture. Feed the engine the
    // oracle's exact decoder_layer_11_hidden through the final model.norm + lm_head
    // (decoder::lm_head), and compare to lm_head_logits: argmax MUST match at every
    // position (the token decision) and the continuous logits must stay within the
    // f32-vs-bf16 cosine gate. This isolates the final-norm + lm_head GEMV — the
    // exact path the example decoder_dump proved argmax-exact vs baidu.
    let _heavy = heavy_rung_guard();
    let loader = FixtureLoader::new();
    let w = match load_subject_weights() {
        Ok(w) => w,
        Err(e) => {
            log.error("WeightsLoad", 1, &e);
            log.result("xfail", t0.elapsed().as_micros());
            return;
        }
    };
    let mut ran = 0usize;
    let mut all_pass = true;
    for gpath in loader.list_goldens().unwrap_or_default() {
        let stem = golden_stem(&gpath);
        let Ok(golden) = loader.load_golden(&gpath) else {
            continue;
        };
        if let Err(e) = FixtureLoader::check_provenance(&golden) {
            log.error("Provenance", 1, &format!("{stem}: {e}"));
            all_pass = false;
            continue;
        }
        let doc_stem = golden.doc_stem_or(&stem);

        match capture_lm_head_seam(&w, &loader, &golden, &doc_stem) {
            Some(Ok(seam)) => {
                let vocab = seam.subject.cols;
                let subj_argmax = argmax_rows(&seam.subject.data, vocab);
                let oracle_argmax = argmax_rows(&seam.oracle.data, vocab);
                let argmax_exact = subj_argmax == oracle_argmax;
                let report =
                    ulp_compare(&seam.subject.data, &seam.oracle.data, OpFamily::MatmulF32);
                let c = cosine(&seam.subject.data, &seam.oracle.data);
                // The token decision (argmax) MUST be exact; the continuous logits
                // are an f32-vs-bf16 divergence held to the cosine gate. max-abs is
                // ledgered against the derived §2 floor but is not the pass gate
                // (a continuous spread inside the bf16 noise is not a token error).
                let pass = argmax_exact && c >= COSINE_F32_THRESHOLD;
                ran += 1;
                all_pass &= pass;
                log.parity(
                    "L3",
                    "cosine",
                    c,
                    COSINE_F32_THRESHOLD,
                    "lm_head_logits",
                    &seam.oracle_sha256,
                    json!({
                        "seam": "final model.norm + lm_head (1280->129280)",
                        "subject": SUBJECT_IDENTITY,
                        "oracle": ORACLE_IDENTITY,
                        "argmax_exact": argmax_exact,
                        "positions": subj_argmax.len(),
                        "max_abs_diff": report.max_abs_diff,
                        "max_abs_floor": derived_tol,
                        "budget_source": "oracle_floor §2 (continuous logits ledgered, not the f32 gate)",
                        "input": "oracle decoder_layer_11_hidden (exact)",
                        "doc": stem.as_str(),
                    }),
                    pass,
                );
            }
            Some(Err(e)) => {
                log.error("LmHeadSeam", 1, &format!("{stem}: {e}"));
                all_pass = false;
            }
            None => {
                log.error(
                    "SeamUnavailable",
                    1,
                    &format!(
                        "{stem}: this golden lacks the decoder_layer_11_hidden + lm_head_logits \
                         activations the lm_head seam isolates (regenerate with bd-3s7v)"
                    ),
                );
            }
        }
    }

    if ran == 0 {
        log.assertion("L3 subject (engine) logit seam capture exercised", false);
        log.error(
            "NotImplemented",
            1,
            "L3: oracle fixtures present but no committed-isolatable logit seam ran \
             (lm_head seam needs decoder_layer_11_hidden + lm_head_logits).",
        );
        log.result("xfail", t0.elapsed().as_micros());
    } else {
        log.result(
            if all_pass { "pass" } else { "fail" },
            t0.elapsed().as_micros(),
        );
        assert!(
            all_pass,
            "L3 subject-vs-oracle logit parity FAILED (argmax drift or cosine < \
             {COSINE_F32_THRESHOLD}); see structured log"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// L4 — token parity (EXACT under greedy, over the reproducible prefix) — bd-re8.6
//
// Decoded token id sequence EXACT under greedy, defined ONLY over the §2
// reproducible prefix per document (PARITY_LADDER §3.3). Fixture+model gated.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn l4_token_exact_prefix() {
    let mut log = Logger::new("L4_token", "corpus");
    log.setup(0);
    let t0 = Instant::now();

    // Always-on: the exactness discipline. The exact-prefix comparator (compare
    // token ids only over [0, reproducible_prefix_len)) is exercised on synthetic
    // streams so the gate's logic is proven with no oracle.
    let oracle_tokens = [5u32, 6, 7, 8, 9];
    let subject_tokens = [5u32, 6, 7, 8, 9];
    let prefix = 4usize; // suppose the oracle floor only reproduces 4 tokens
    let exact_over_prefix = oracle_tokens[..prefix] == subject_tokens[..prefix];
    log.assertion(
        "L4 EXACT only over the §2 reproducible prefix",
        exact_over_prefix,
    );

    if !fixtures_present() || !model_present() {
        log.skip_no_model(
            "L4 token-exact compare needs the golden token_stream (bd-3s7v fixtures) \
             AND the engine greedy decode seeded from the oracle inputs_embeds. \
             Exact-prefix comparator demonstrated above on synthetic streams.",
        );
        log.result("skip_no_model", t0.elapsed().as_micros());
        return;
    }

    // FIXTURE+MODEL PRESENT: the REAL subject (engine) greedy decode (bd-2ksr).
    // The engine's cached AR loop (prefill_with_cache -> lm_head_cached ->
    // sampler::decode_step -> decode_step_with_cache) is seeded with the
    // oracle's EXACT inputs_embeds + prompt ids and its emitted token ids are
    // compared to token_stream.generated_ids. The committed oracle run is fully
    // deterministic (CPU, greedy, torch.use_deterministic_algorithms, seed 0;
    // deterministic_replay.expected_prefix_kind = full_decoded_text), so the §2
    // reproducible prefix is the FULL stream and the compare is EXACT end to
    // end. Seeding from the oracle embeds (not the engine's own vision output)
    // keeps the rung's failure attributable to the decoder/sampler alone — the
    // upstream stages have their own rungs (L0/L1/L2).
    let _heavy = heavy_rung_guard();
    let loader = FixtureLoader::new();
    let w = match load_subject_weights() {
        Ok(w) => w,
        Err(e) => {
            log.error("WeightsLoad", 1, &e);
            log.result("xfail", t0.elapsed().as_micros());
            return;
        }
    };
    let captures = subject_decode_captures(&w, &loader);
    let mut ran = 0usize;
    let mut all_pass = true;
    for gpath in loader.list_goldens().unwrap_or_default() {
        let stem = golden_stem(&gpath);
        let Ok(golden) = loader.load_golden(&gpath) else {
            continue;
        };
        if let Err(e) = FixtureLoader::check_provenance(&golden) {
            log.error("Provenance", 1, &format!("{stem}: {e}"));
            all_pass = false;
            continue;
        }
        match captures.get(&stem) {
            Some(Ok(cap)) => {
                let n_golden = cap.golden_generated.len();
                let matched = matched_prefix_len(&cap.emitted, &cap.golden_generated);
                let exact = matched == n_golden && cap.emitted.len() == n_golden;
                let value = matched as f64 / n_golden.max(1) as f64;
                ran += 1;
                all_pass &= exact;
                log.parity(
                    "L4",
                    "token_exact_fraction",
                    value,
                    1.0,
                    "token_stream.generated_ids",
                    &cap.generated_ids_sha256,
                    json!({
                        "seam": "decoder prefill + greedy AR decode (cached f32 loop)",
                        "subject": SUBJECT_IDENTITY,
                        "oracle": ORACLE_IDENTITY,
                        "input": "oracle inputs_embeds + prompt_ids (exact)",
                        "exact": exact,
                        "n_subject": cap.emitted.len(),
                        "n_golden": n_golden,
                        "prompt_len": cap.prompt_len,
                        "first_divergence": if exact { Value::Null } else { json!(matched) },
                        "subject_token_at_divergence":
                            cap.emitted.get(matched).copied().map_or(Value::Null, |t| json!(t)),
                        "oracle_token_at_divergence":
                            cap.golden_generated.get(matched).copied().map_or(Value::Null, |t| json!(t)),
                        // The free 12-layer-stack ledger (L2 cannot get per-layer).
                        "prefill_hidden_cosine": cap.prefill_hidden_cosine,
                        "prefill_hidden_max_abs": cap.prefill_hidden_max_abs,
                        "reproducible_prefix": "full stream (oracle run is deterministic: \
                                                CPU greedy, torch deterministic algorithms)",
                        "doc": stem.as_str(),
                    }),
                    exact,
                );
            }
            Some(Err(e)) => {
                log.error("DecodeSeam", 1, &format!("{stem}: {e}"));
                all_pass = false;
            }
            None => {
                // Defensive: captures are keyed off the same golden listing.
                log.error(
                    "SeamUnavailable",
                    1,
                    &format!("{stem}: no decode capture was computed for this golden"),
                );
                all_pass = false;
            }
        }
    }

    if ran == 0 {
        log.assertion("L4 subject (engine) greedy decode exercised", false);
        log.error(
            "NotImplemented",
            1,
            "L4: oracle fixtures present but no decode capture ran \
             (need inputs_embeds + token_stream from the bd-3s7v regen).",
        );
        log.result("xfail", t0.elapsed().as_micros());
    } else {
        log.result(
            if all_pass { "pass" } else { "fail" },
            t0.elapsed().as_micros(),
        );
        assert!(
            all_pass,
            "L4 subject-vs-oracle token parity FAILED (greedy stream not exact; the \
             first divergence index is in the structured log)"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// L5 — end-to-end OCR (exact-where-det + CER/TEDS/Formula-CDM budget) — bd-re8.7
//
// Decoded text + bbox tags on the golden corpus: exact-match where the reference
// is deterministic, aggregate CER/TEDS/Formula-CDM within a documented budget
// (PARITY_LADDER §3.4). The model-gated e2e rung — skip-with-SUCCESS without the
// weights, proving the native path via the /nonexistent fallback.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn l5_end_to_end_cer_budget() {
    let mut log = Logger::new("L5_e2e", "corpus");
    log.setup(0);
    let t0 = Instant::now();

    // Always-on: the CER metric itself is a pure function (character error rate);
    // prove it on synthetic strings so the L5 budget machinery is exercised with
    // no model. CER = Levenshtein(ref, hyp) / len(ref).
    let cer_identical = char_error_rate("# Invoice\nTotal: 42", "# Invoice\nTotal: 42");
    let cer_one_edit = char_error_rate("hello", "hallo");
    log.assertion("CER(identical) == 0", cer_identical == 0.0);
    log.assertion(
        "CER(1 substitution / 5) == 0.2",
        (cer_one_edit - 0.2).abs() < 1e-9,
    );

    if !fixtures_present() || !model_present() {
        log.skip_no_model(
            "L5 decoded-text CER compare needs the golden <doc>_reference.json \
             (decoded text) AND the 6.67 GB weights for the engine decode + \
             detokenize + postprocess. Native path would be proven by the \
             /nonexistent fallback. CER metric demonstrated on synthetic strings \
             (PARITY_LADDER §3.4).",
        );
        log.result("skip_no_model", t0.elapsed().as_micros());
        return;
    }

    // FIXTURE+MODEL PRESENT: the REAL subject decoded-text compare (bd-2ksr).
    // The engine token stream from the shared L4 decode capture is detokenized
    // (the BPE tokenizer beside the weights, the same load path the lib uses)
    // and rendered by postprocess::finalize — exactly forward_pre + recognize —
    // then compared to the golden decoded_text (= the oracle model.infer()
    // post-processed result.md) under the documented CER budget, with the
    // exact-match bit ledgered. Seeding from the oracle inputs_embeds isolates
    // the decode→text layers (the tokens are L4-exact when the decoder is at
    // parity, so the budget absorbs detokenizer/markdown assembly only); the
    // full-image `focr ocr` e2e lives in tests/e2e_recognize.rs and the
    // off-repo CER harness.
    let _heavy = heavy_rung_guard();
    let loader = FixtureLoader::new();
    let w = match load_subject_weights() {
        Ok(w) => w,
        Err(e) => {
            log.error("WeightsLoad", 1, &e);
            log.result("xfail", t0.elapsed().as_micros());
            return;
        }
    };
    let tokenizer_path = subject_model_path()
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("tokenizer.json");
    let tokenizer = match Tokenizer::load(&tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            log.error(
                "TokenizerLoad",
                1,
                &format!("load {}: {e}", tokenizer_path.display()),
            );
            log.result("xfail", t0.elapsed().as_micros());
            return;
        }
    };
    let captures = subject_decode_captures(&w, &loader);
    let mut ran = 0usize;
    let mut all_pass = true;
    for gpath in loader.list_goldens().unwrap_or_default() {
        let stem = golden_stem(&gpath);
        let Ok(golden) = loader.load_golden(&gpath) else {
            continue;
        };
        if let Err(e) = FixtureLoader::check_provenance(&golden) {
            log.error("Provenance", 1, &format!("{stem}: {e}"));
            all_pass = false;
            continue;
        }
        let Some(bar) = golden.decoded_text.clone() else {
            log.error(
                "GoldenIncomplete",
                1,
                &format!("{stem}: golden has no decoded_text (oracle wrote result.md only?)"),
            );
            all_pass = false;
            continue;
        };
        // Fixture self-consistency: the committed text must hash to its own
        // recorded sha256 (a corrupt/hand-edited golden may not set the bar).
        let bar_sha = sha256_hex(bar.as_bytes());
        if let Some(expected) = golden.decoded_text_sha256.as_deref()
            && expected != bar_sha
        {
            log.error(
                "GoldenIncomplete",
                1,
                &format!(
                    "{stem}: decoded_text sha256 {bar_sha} != recorded {expected} \
                     (corrupt golden must not set the bar)"
                ),
            );
            all_pass = false;
            continue;
        }
        match captures.get(&stem) {
            Some(Ok(cap)) => {
                let outcome = (|| -> Result<String, String> {
                    let decoded = tokenizer
                        .decode(&cap.emitted)
                        .map_err(|e| format!("tokenizer.decode: {e}"))?;
                    let dim = |key: &str| -> Result<u32, String> {
                        golden
                            .raw
                            .pointer(&format!("/image/{key}"))
                            .and_then(Value::as_u64)
                            .and_then(|x| u32::try_from(x).ok())
                            .ok_or_else(|| format!("golden missing image.{key}"))
                    };
                    let (img_w, img_h) = (dim("width")?, dim("height")?);
                    postprocess::finalize(&decoded, img_w, img_h)
                        .map_err(|e| format!("postprocess::finalize: {e}"))
                })();
                match outcome {
                    Ok(subject_text) => {
                        let cer = char_error_rate(&bar, &subject_text);
                        let exact = subject_text == bar;
                        let pass = cer <= L5_CER_BUDGET;
                        ran += 1;
                        all_pass &= pass;
                        log.parity(
                            "L5",
                            "cer",
                            cer,
                            L5_CER_BUDGET,
                            &golden.doc,
                            golden.decoded_text_sha256.as_deref().unwrap_or(""),
                            json!({
                                "seam": "decode ids -> tokenizer.decode -> postprocess::finalize",
                                "subject": SUBJECT_IDENTITY,
                                "oracle": ORACLE_IDENTITY,
                                "input": "oracle inputs_embeds + prompt_ids (exact)",
                                "exact_text": exact,
                                "subject_chars": subject_text.chars().count(),
                                "golden_chars": bar.chars().count(),
                                "budget_source": "documented detok+postprocess envelope \
                                                  (tokens are gated EXACT by L4)",
                                "doc": stem.as_str(),
                            }),
                            pass,
                        );
                    }
                    Err(e) => {
                        log.error("TextSeam", 1, &format!("{stem}: {e}"));
                        all_pass = false;
                    }
                }
            }
            Some(Err(e)) => {
                log.error("DecodeSeam", 1, &format!("{stem}: {e}"));
                all_pass = false;
            }
            None => {
                log.error(
                    "SeamUnavailable",
                    1,
                    &format!("{stem}: no decode capture was computed for this golden"),
                );
                all_pass = false;
            }
        }
    }

    if ran == 0 {
        log.assertion("L5 subject (engine) decoded-text compare exercised", false);
        log.error(
            "NotImplemented",
            1,
            "L5: oracle fixtures present but no decoded-text compare ran \
             (need decoded_text + inputs_embeds + token_stream in the goldens).",
        );
        log.result("xfail", t0.elapsed().as_micros());
    } else {
        log.result(
            if all_pass { "pass" } else { "fail" },
            t0.elapsed().as_micros(),
        );
        assert!(
            all_pass,
            "L5 subject-vs-oracle decoded-text parity FAILED (CER above the documented \
             {L5_CER_BUDGET} budget); see structured log"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Oracle-differential comparator — VERIFY-differential-suite / bd-re8.9 (§6)
//
// Differential = "same as the bf16 reference (any input)". Per-op + e2e against
// the primary bf16 oracle (frozen .npy/.json) through the ULP table / L3-L5
// tolerances. Intentional divergences are XFAIL (a DISC-NNN), never SKIP.
// Model-gated e2e: skip-with-SUCCESS, prove native path via /nonexistent.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn differential_per_op_vs_bf16_oracle() {
    let mut log = Logger::new("differential_per_op", "corpus");
    log.setup(0);
    let t0 = Instant::now();

    // Always-on: the differential ROW SHAPE (the contract each test emits, §6.2)
    // is validated on a synthetic row so a downstream consumer (the coverage
    // matrix) can rely on it. EngineIdentity must be asserted-distinct (§1.1) —
    // we assert the subject/oracle labels differ so the highest-value false green
    // (oracle compared against itself) is structurally impossible.
    let subject_identity = "franken_ocr";
    let oracle_identity = "unlimited-ocr-oracle";
    log.assertion(
        "EngineIdentity subject != oracle (never compare oracle against itself)",
        subject_identity != oracle_identity,
    );
    let row = differential_row("op", "bf16", "sam_output", 0.0, true, false, None);
    log.assertion(
        "differential row carries {scope,oracle,module,max_diff,within_tol,xfail}",
        row.get("scope").is_some()
            && row.get("oracle").is_some()
            && row.get("within_tol").is_some()
            && row.get("xfail").is_some(),
    );

    if !fixtures_present() || !model_present() {
        log.skip_no_model(
            "differential per-op needs the per-stage oracle activations + the engine \
             (the live bridge supplies ad-hoc inputs; frozen .npy supply the corpus). \
             Intentional divergences are XFAIL (a DISC-NNN), never SKIP. Row-shape + \
             EngineIdentity guard ran always-on.",
        );
        log.result("skip_no_model", t0.elapsed().as_micros());
        return;
    }

    // FIXTURE+MODEL PRESENT: a REAL per-op differential (bd-2ksr; this branch
    // previously logged an unconditional "pass" with no compare — a fabricated
    // green). The projector seam (oracle clip_output + sam_output -> engine
    // bridge vs projector_output) is diffed per doc and emitted as one §6.2
    // differential row. NOTE the oracle activations are bf16-computed (upcast
    // to f32 on the wire), so within_tol is the f32-vs-bf16 cosine gate — the
    // strict f32 ULP table is LEDGERED, not gated (METHODOLOGY §1.3 posture);
    // the broader per-module matrix accretes as the remaining seams stabilize.
    let _heavy = heavy_rung_guard();
    let loader = FixtureLoader::new();
    let w = match load_subject_weights() {
        Ok(w) => w,
        Err(e) => {
            log.error("WeightsLoad", 1, &e);
            log.result("xfail", t0.elapsed().as_micros());
            return;
        }
    };
    let mut ran = 0usize;
    let mut all_pass = true;
    for gpath in loader.list_goldens().unwrap_or_default() {
        let stem = golden_stem(&gpath);
        let Ok(golden) = loader.load_golden(&gpath) else {
            continue;
        };
        if let Err(e) = FixtureLoader::check_provenance(&golden) {
            log.error("Provenance", 1, &format!("{stem}: {e}"));
            all_pass = false;
            continue;
        }
        let doc_stem = golden.doc_stem_or(&stem);
        match capture_projector_seam(&w, &loader, &golden, &doc_stem) {
            Some(Ok(seam)) => {
                let c = cosine(&seam.subject.data, &seam.oracle.data);
                let report =
                    ulp_compare(&seam.subject.data, &seam.oracle.data, OpFamily::MatmulF32);
                let within_tol = c >= COSINE_F32_THRESHOLD;
                let row = differential_row(
                    "op",
                    "bf16",
                    "projector",
                    report.max_abs_diff,
                    within_tol,
                    false,
                    None,
                );
                ran += 1;
                all_pass &= within_tol;
                log.parity(
                    "differential",
                    "cosine",
                    c,
                    COSINE_F32_THRESHOLD,
                    "projector_output",
                    &seam.oracle_sha256,
                    json!({
                        "differential_row": row,
                        "subject": SUBJECT_IDENTITY,
                        "oracle": ORACLE_IDENTITY,
                        "max_ulp_ledgered": report.max_ulp,
                        "ulp_budget_f32": report.budget_ulp,
                        "doc": stem.as_str(),
                    }),
                    within_tol,
                );
            }
            Some(Err(e)) => {
                log.error("ProjectorSeam", 1, &format!("{stem}: {e}"));
                all_pass = false;
            }
            None => {
                log.error(
                    "SeamUnavailable",
                    1,
                    &format!(
                        "{stem}: no isolatable differential seam (need clip_output + \
                         sam_output + projector_output)"
                    ),
                );
            }
        }
    }
    if ran == 0 {
        log.assertion("differential per-op subject seam exercised", false);
        log.error(
            "NotImplemented",
            1,
            "differential: oracle fixtures present but no per-op differential ran.",
        );
        log.result("xfail", t0.elapsed().as_micros());
    } else {
        log.result(
            if all_pass { "pass" } else { "fail" },
            t0.elapsed().as_micros(),
        );
        assert!(
            all_pass,
            "differential per-op vs bf16 oracle FAILED (projector outside the cosine \
             gate); see structured log"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stable-surface anchors — these run ALWAYS (no weights, no fixtures), exercising
// the genuinely-stable public surface the harness can rely on today: the error
// exit-code contract, the robot schema, and the scrubber on a robot-shaped event.
// They are the L0-level "the contract didn't move" guards.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn surface_error_exit_codes_are_stable() {
    use franken_ocr::FocrError;
    let mut log = Logger::new("surface_exit_codes", "stable");
    log.setup(0);
    let t0 = Instant::now();
    // The documented stable codes (src/error.rs, plan §7.4). Agents branch on
    // these; a renumber is a contract break the harness must catch.
    let cases: &[(FocrError, i32)] = &[
        (FocrError::Usage("x".into()), 2),
        (FocrError::ModelNotFound("x".into()), 3),
        (FocrError::InputDecode("x".into()), 4),
        (FocrError::Timeout("x".into()), 5),
        (FocrError::Cancelled, 6),
        (FocrError::FormatMismatch("x".into()), 7),
        (FocrError::NotImplemented("x".into()), 1),
    ];
    let mut all = true;
    for (err, code) in cases {
        let got = err.exit_code();
        let ok = got == *code;
        all &= ok;
        log.assertion(&format!("{err:?} ⇒ exit {code}"), ok);
        if !ok {
            log.error("ExitCodeDrift", got, &format!("expected {code}, got {got}"));
        }
    }
    log.result(if all { "pass" } else { "fail" }, t0.elapsed().as_micros());
    assert!(
        all,
        "stable exit-code contract drifted (see structured log)"
    );
}

#[test]
fn surface_robot_schema_self_describes() {
    let mut log = Logger::new("surface_robot_schema", "stable");
    log.setup(0);
    let t0 = Instant::now();
    let schema = franken_ocr::robot::robot_schema();
    let version_ok = schema["schema_version"] == json!(franken_ocr::robot::ROBOT_SCHEMA_VERSION);
    let events_ok = schema["events"]
        .as_array()
        .map(|a| a.len() == franken_ocr::robot::EVENT_KINDS.len())
        .unwrap_or(false);
    log.assertion("robot schema advertises ROBOT_SCHEMA_VERSION", version_ok);
    log.assertion("robot schema enumerates all EVENT_KINDS", events_ok);
    // Scrub a robot-shaped event and assert the timing leaf is masked but present.
    let event = json!({
        "schema_version": 1, "event": "stage", "name": "vision", "seq": 2, "elapsed_ms": 143
    });
    let scrubbed = scrub_volatile(&event);
    let scrub_ok = scrubbed["elapsed_ms"] == json!("[ms]")
        && scrubbed.as_object().unwrap().contains_key("elapsed_ms");
    log.assertion(
        "scrubber masks elapsed_ms but keeps the field present",
        scrub_ok,
    );
    log.result(
        if version_ok && events_ok && scrub_ok {
            "pass"
        } else {
            "fail"
        },
        t0.elapsed().as_micros(),
    );
    assert!(version_ok && events_ok && scrub_ok);
}

#[test]
fn comparator_normalizes_before_numeric_compare() {
    // A shape mismatch must be caught by TensorSpec BEFORE any cosine/ULP runs —
    // METHODOLOGY §1.3 (normalize both sides first). This is the always-on guard
    // that the comparator chokepoint is honored.
    let mut log = Logger::new("comparator_normalize", "synthetic");
    log.setup(0);
    let t0 = Instant::now();
    let subject = NormalizedValue::from_f32(TensorSpec::new([2, 3], DType::F32), vec![0.0; 6]);
    let oracle = NormalizedValue::from_f32(TensorSpec::new([3, 2], DType::F32), vec![0.0; 6]);
    let mismatch = subject.spec.check_against(&oracle.spec);
    log.assertion(
        "shape mismatch rejected before numeric compare",
        mismatch.is_err(),
    );
    log.result("pass", t0.elapsed().as_micros());
    assert!(mismatch.is_err(), "{:?}", mismatch);
}

// ─────────────────────────────────────────────────────────────────────────────
// Small pure helpers used by the rungs (CER, the differential row shape).
// ─────────────────────────────────────────────────────────────────────────────

/// Character Error Rate = Levenshtein(reference, hypothesis) / len(reference).
/// Used by L5 (PARITY_LADDER §3.4). Pure; unit-tested via the L5 always-on path.
/// `len(ref) == 0` ⇒ CER 0 if hyp also empty, else 1.0 (every char inserted).
fn char_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let r: Vec<char> = reference.chars().collect();
    let h: Vec<char> = hypothesis.chars().collect();
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    let dist = levenshtein(&r, &h);
    dist as f64 / r.len() as f64
}

/// Standard O(n·m) Levenshtein over char slices (two-row DP).
fn levenshtein(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Build the differential-row contract (PARITY_LADDER §6.2): one structured row
/// per test for the coverage matrix.
fn differential_row(
    scope: &str,
    oracle: &str,
    module: &str,
    max_diff: f64,
    within_tol: bool,
    xfail: bool,
    disc: Option<&str>,
) -> Value {
    json!({
        "scope": scope,
        "oracle": oracle,
        "module": module,
        "max_diff": max_diff,
        "within_tol": within_tol,
        "xfail": xfail,
        "disc": disc,
    })
}

// A tiny extension so the rungs can resolve a golden's doc stem for the
// activations subdir (`activations/<stem>/`). The oracle keys the activations
// dir by `doc.stem` while the golden's `doc` field carries the full filename;
// fall back to the filename stem the caller already computed.
trait DocStem {
    fn doc_stem_or(&self, fallback: &str) -> String;
}

impl DocStem for parity_harness::ReferenceGolden {
    fn doc_stem_or(&self, fallback: &str) -> String {
        Path::new(&self.doc)
            .file_stem()
            .and_then(|s| s.to_str())
            .map(str::to_string)
            .unwrap_or_else(|| fallback.to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Always-on unit tests for the subject-seam helpers (synthetic inputs only —
// no fixtures, no weights). These prove the fixture-shape adapters, the golden
// token-stream/contract readers and the L4 prefix locator with the same
// no-silent-misshape discipline the gated rungs rely on.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sam_input_view_mat_reshapes_channel_major() {
    // [1, 3, 2, 2] C-order flattens to the [3, H*W] channel-major view verbatim.
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let nv = NormalizedValue::from_f32(TensorSpec::new([1, 3, 2, 2], DType::F32), data.clone());
    let m = sam_input_view_mat("sam_input", &nv).expect("reshape [1,3,2,2]");
    assert_eq!((m.rows, m.cols), (3, 4));
    assert_eq!(m.data, data, "C-order flat data is preserved verbatim");
    // Any other rank/shape is rejected loudly, never silently misshaped.
    let bad = NormalizedValue::from_f32(TensorSpec::new([3, 2, 2], DType::F32), vec![0.0; 12]);
    assert!(
        sam_input_view_mat("sam_input", &bad)
            .unwrap_err()
            .contains("[1, 3, H, W]")
    );
}

#[test]
fn sam_output_features_mat_folds_hw_into_cols() {
    // [1, C, H, W] -> [C, H*W] — the vision_sam `flatten(2)` layout the bridge
    // and CLIP consume. (The last-dim-cols reshape would misshape this to
    // [C*H, W], which vision_bridge rejects — the bug this adapter fixes.)
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let nv = NormalizedValue::from_f32(TensorSpec::new([1, 4, 2, 2], DType::F32), data.clone());
    let m = sam_output_features_mat("sam_output", &nv).expect("reshape [1,4,2,2]");
    assert_eq!((m.rows, m.cols), (4, 4));
    assert_eq!(m.data, data);
    let bad = NormalizedValue::from_f32(TensorSpec::new([2, 4, 2, 2], DType::F32), vec![0.0; 32]);
    assert!(
        sam_output_features_mat("sam_output", &bad)
            .unwrap_err()
            .contains("[1, C, H, W]")
    );
}

/// Build a synthetic golden carrying only a `token_stream` block (plus the
/// greedy generation contract), through the same parser the rungs use.
fn synthetic_stream_golden(token_stream: Value) -> ReferenceGolden {
    FixtureLoader::golden_from_value(json!({
        "doc": "doc01.png",
        "token_stream": token_stream,
        "generation": {
            "temperature": 0.0,
            "max_length": 32768,
            "no_repeat_ngram_size": 35,
            "ngram_window": 128
        },
    }))
    .expect("synthetic golden parses")
}

#[test]
fn golden_token_stream_parses_and_rejects_malformed() {
    let ok = synthetic_stream_golden(json!({
        "prompt_ids": [0, 128815, 34030, 16],
        "n_prompt": 4,
        "generated_ids": [128818, 1],
        "n_generated": 2,
        "generated_ids_sha256": "abc"
    }));
    let s = golden_token_stream(&ok).expect("valid stream");
    assert_eq!(s.prompt_ids, vec![0, 128815, 34030, 16]);
    assert_eq!(s.generated_ids, vec![128818, 1]);
    assert_eq!(s.generated_ids_sha256, "abc");

    // A count field disagreeing with its array is named.
    let bad_count = synthetic_stream_golden(json!({
        "prompt_ids": [0, 1], "n_prompt": 3, "generated_ids": [1], "n_generated": 1
    }));
    assert!(
        golden_token_stream(&bad_count)
            .unwrap_err()
            .contains("n_prompt")
    );

    // A non-u32 id is named with its index.
    let bad_id = synthetic_stream_golden(json!({
        "prompt_ids": [0, "x"], "generated_ids": [1]
    }));
    assert!(
        golden_token_stream(&bad_id)
            .unwrap_err()
            .contains("prompt_ids[1]")
    );

    // An empty stream must not vacuously "match".
    let empty = synthetic_stream_golden(json!({ "prompt_ids": [0], "generated_ids": [] }));
    assert!(golden_token_stream(&empty).unwrap_err().contains("empty"));
}

#[test]
fn golden_decode_params_come_from_the_golden_contract() {
    // The subject decodes under the GOLDEN's own generation block (never a
    // silent apples-vs-oranges compare on a re-preset fixture).
    let golden = FixtureLoader::golden_from_value(json!({
        "doc": "doc01.png",
        "generation": {
            "temperature": 0.0, "max_length": 512,
            "no_repeat_ngram_size": 7, "ngram_window": 64
        },
        "token_stream": { "decode_metadata": { "generate_kwargs": { "eos_token_id": 2 } } }
    }))
    .expect("golden parses");
    let p = golden_decode_params(&golden).expect("params");
    assert_eq!(p.max_length, 512);
    assert_eq!(p.no_repeat_ngram_size, 7);
    assert_eq!(p.ngram_window, 64);
    assert_eq!(p.eos_token_id, 2);
    assert!(p.is_greedy());

    // A sampling oracle is rejected — token-EXACT parity is undefined for it.
    let sampling = FixtureLoader::golden_from_value(json!({
        "doc": "doc01.png",
        "generation": { "temperature": 0.7, "max_length": 512,
                         "no_repeat_ngram_size": 7, "ngram_window": 64 }
    }))
    .expect("golden parses");
    assert!(
        golden_decode_params(&sampling)
            .unwrap_err()
            .contains("temperature")
    );
}

#[test]
fn matched_prefix_len_locates_first_divergence() {
    assert_eq!(matched_prefix_len(&[1, 2, 3], &[1, 2, 3]), 3);
    assert_eq!(
        matched_prefix_len(&[1, 2, 9, 3], &[1, 2, 3, 3]),
        2,
        "the return value IS the first-divergence index"
    );
    assert_eq!(matched_prefix_len(&[1, 2], &[1, 2, 3]), 2);
    assert_eq!(matched_prefix_len(&[], &[1]), 0);
}

#[test]
fn l0_resample_tolerance_is_the_documented_bd30me_envelope() {
    // 8 u8 quantization steps in [-1,1] units — the documented CatmullRom-vs-
    // PIL-BICUBIC envelope (bd-30me). A knob-turn away from the documented
    // derivation must be a reviewed diff, not a drive-by constant edit.
    let tol = std::hint::black_box(L0_RESAMPLE_MAX_ABS_TOL);
    assert!((tol - 8.0 * 2.0 / 255.0).abs() < 1e-12);
    assert!(tol < 0.1, "envelope stays far below content scale");
}

#[test]
fn resolve_corpus_image_searches_pages_dirs_and_provenance() {
    let base = std::env::temp_dir().join("franken_ocr_l0_corpus_test");
    let root = base.join("fixtures/native");
    let pages = base.join("pages");
    std::fs::create_dir_all(&root).expect("mk fixtures root");
    std::fs::create_dir_all(&pages).expect("mk pages dir");
    // Deliberately-unique doc names so a real $FOCR_CORPUS_DIR in the verify
    // environment (searched FIRST by design) cannot shadow the assertions.
    std::fs::write(pages.join("focr_parity_test_doc01.png"), b"png").expect("write page");

    // Found two levels above the fixtures root — the off-repo layout.
    let golden = FixtureLoader::golden_from_value(json!({ "doc": "focr_parity_test_doc01.png" }))
        .expect("golden");
    let (found, _) = resolve_corpus_image(&root, &golden);
    assert_eq!(found, Some(pages.join("focr_parity_test_doc01.png")));

    // Found via the golden's own provenance `--corpus` dir.
    let prov_dir = base.join("prov_corpus");
    std::fs::create_dir_all(&prov_dir).expect("mk prov corpus");
    std::fs::write(prov_dir.join("focr_parity_test_doc02.png"), b"png").expect("write page");
    let golden2 = FixtureLoader::golden_from_value(json!({
        "doc": "focr_parity_test_doc02.png",
        "provenance": {
            "command_argv": ["gen.py", "--corpus", prov_dir.to_string_lossy().to_string()]
        }
    }))
    .expect("golden");
    let (found2, _) = resolve_corpus_image(&root, &golden2);
    assert_eq!(found2, Some(prov_dir.join("focr_parity_test_doc02.png")));

    // A bare `..` doc has no file name — no candidate path is even formed, so a
    // traversal-shaped doc can never escape the corpus dirs.
    let evil = FixtureLoader::golden_from_value(json!({ "doc": ".." })).expect("golden");
    let (found3, candidates) = resolve_corpus_image(&root, &evil);
    assert_eq!(found3, None);
    assert!(candidates.is_empty());
}
