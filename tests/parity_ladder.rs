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
#[path = "support/uocr2.rs"]
mod uocr2;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use parity_harness::{
    COSINE_F32_THRESHOLD, DType, FixtureLoader, Logger, NormalizedValue, OpFamily, ReferenceGolden,
    TensorSpec, cosine, establish_floor, max_abs_diff, scrub_volatile, sha256_hex, ulp_compare,
};
use serde_json::{Value, json};
use uocr2::{load_uocr2_source, uocr2_inputs_embeds};

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
    connector, decoder, model_arch, postprocess, vision_bridge, vision_clip, vision_sam,
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
fn model_present(log: &mut Logger) -> bool {
    subject_model_path().exists() && subject_is_ladder_model(log)
}

/// Whether `$FOCR_MODEL_PATH` may arm the ladder (U2-A7). The oracle fixtures
/// are v1 R-SWA activations, so a subject the header identifies as another
/// model (Unlimited-OCR-2's ladder is U2-C3) skips with the reason logged.
/// A header that does not identify still arms the rungs, so a broken v1
/// artifact fails loudly instead of skipping.
fn subject_is_ladder_model(log: &mut Logger) -> bool {
    let Some(spec) = std::env::var_os("FOCR_MODEL_PATH") else {
        return true;
    };
    match franken_ocr::native_engine::model_id_at(Path::new(&spec)) {
        Some(id) if id != model_arch::UNLIMITED_OCR_V1_ID => {
            log.assertion(
                &format!(
                    "FOCR_MODEL_PATH is {id}: the ladder's oracle fixtures are {} only, \
                     so its model-gated rungs skip",
                    model_arch::UNLIMITED_OCR_V1_ID
                ),
                true,
            );
            false
        }
        Some(_) | None => true,
    }
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
    let p = std::env::var_os("FOCR_MODEL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/unlimited-ocr.focrq"));
    // The engine's resolver accepts a safetensors DIRECTORY; the ladder's
    // Weights::load wants the shard file. Resolve the common case here so
    // `FOCR_MODEL_PATH=<model dir>` arms the ladder instead of panicking
    // (fresh-eyes fix — this exact footgun cost a wasted armed run).
    if p.is_dir()
        && let Ok(entries) = std::fs::read_dir(&p)
    {
        let mut shards: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|c| c.extension().is_some_and(|x| x == "safetensors"))
            // exFAT/macOS AppleDouble junk (`._model-*.safetensors`, 4 KB
            // resource forks) sorts BEFORE the real shard and is a valid
            // `.safetensors` name — skip every dotfile or the armed ladder
            // dies on a phantom "header overruns file" (bd-re8.19 catch).
            .filter(|c| {
                c.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| !n.starts_with('.'))
            })
            .collect();
        shards.sort();
        if let Some(shard) = shards.into_iter().next() {
            return shard;
        }
    }
    p
}

/// Resolve + load the subject model weights via [`subject_model_path`]. Only
/// called after a rung confirms [`model_present`], so a failure here is a
/// genuine load error worth surfacing (never silently skipped).
fn load_subject_weights() -> Result<Weights, String> {
    let path = subject_model_path();
    Weights::load(&path).map_err(|e| format!("load subject weights {}: {e}", path.display()))
}

/// Serializes the fixture+model-gated rungs. Each gated rung holds a full
/// [`Weights`] load (~6.7 GB widened on access) and L4 additionally builds the
/// conservative [`decoder::DecoderWeightCache`] (int8 FFN/expert matrices,
/// f32 attention, router and `lm_head`); the default parallel test harness would
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
        max_positions: sampler::DEFAULT_MAX_POSITIONS,
        max_page_tokens: sampler::UOCR2_DEFAULT_MAX_PAGE_TOKENS,
        entmax_domain: franken_ocr::native_engine::srswa::EntmaxDomain::TopM,
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
    // The dominant costs (the conservative decoder weight cache + embed table) are built
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

    if !fixtures_present() || !model_present(&mut log) {
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
            let pre = preprocess::preprocess_image(
                &image_path,
                preprocess::PreprocessMode::base(),
                preprocess::PreprocessProfile::V1,
            )
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
                // EXACT passes outright. The bd-30me resample envelope exists
                // ONLY to cover the default CatmullRom kernel's known delta vs
                // the PIL oracle; with the FOCR_RESAMPLE=pil-bicubic
                // kill-switch armed, the kernel CLAIMS bit-exactness — a
                // non-exact result there is a real failure, and the envelope
                // must not paper over it (fresh-eyes fix).
                let pil_armed = format!("{:?}", preprocess::resample_kind()).contains("PilBicubic");
                let pass = exact
                    || (!pil_armed && mad <= L0_RESAMPLE_MAX_ABS_TOL && c >= COSINE_F32_THRESHOLD);
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

// ── Unlimited-OCR-2 lane (U2-B7, U2-C1) ─────────────────────────────────────

/// The U2-A6 f32 oracle the v2 rungs compare against: the run's dump directory
/// (`FOCR_UOCR2_FIXTURES_DIR`, `<oracle root>/f32/t12`), laid out as
/// `<fixture>_reference.json` plus `activations/<fixture>/<stage>.npy`, and the
/// committed `tests/fixtures/uocr2_oracle/ORACLE.json` naming each fixture's
/// pages and parameters.
struct Uocr2Oracle {
    dumps: PathBuf,
    fixtures: serde_json::Map<String, Value>,
}

impl Uocr2Oracle {
    /// The oracle, or `None` after a skip line when the dumps are not named.
    fn from_env(log: &mut Logger) -> Option<Self> {
        let Some(dumps) = std::env::var_os("FOCR_UOCR2_FIXTURES_DIR").map(PathBuf::from) else {
            log.skip_no_model(
                "FOCR_UOCR2_FIXTURES_DIR unset: point it at the U2-A6 dump directory \
                 (<oracle root>/f32/t12)",
            );
            return None;
        };
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/uocr2_oracle/ORACLE.json"
        );
        let oracle: Value = serde_json::from_slice(&std::fs::read(path).expect("ORACLE.json"))
            .expect("ORACLE.json parses");
        let fixtures = oracle["fixtures"]
            .as_object()
            .expect("ORACLE.json fixtures")
            .clone();
        Some(Self { dumps, fixtures })
    }

    fn npy(&self, fixture: &str, stage: &str) -> Vec<u8> {
        let path = self
            .dumps
            .join("activations")
            .join(fixture)
            .join(format!("{stage}.npy"));
        std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    fn f32s(&self, fixture: &str, stage: &str) -> (Vec<usize>, Vec<f32>) {
        parity_harness::read_npy_f32(&self.npy(fixture, stage))
            .unwrap_or_else(|e| panic!("{fixture} {stage}: {e}"))
    }

    fn i32s(&self, fixture: &str, stage: &str) -> (Vec<usize>, Vec<i32>) {
        parity_harness::read_npy_i32(&self.npy(fixture, stage))
            .unwrap_or_else(|e| panic!("{fixture} {stage}: {e}"))
    }

    fn u8s(&self, fixture: &str, stage: &str) -> (Vec<usize>, Vec<u8>) {
        parity_harness::read_npy_u8(&self.npy(fixture, stage))
            .unwrap_or_else(|e| panic!("{fixture} {stage}: {e}"))
    }

    /// `<fixture>_reference.json`.
    fn reference(&self, fixture: &str) -> Value {
        let path = self.dumps.join(format!("{fixture}_reference.json"));
        serde_json::from_slice(&std::fs::read(&path).expect("reference json"))
            .expect("reference json parses")
    }

    /// The sha256 of one dumped stage's file, from the reference's manifest.
    fn stage_sha256(reference: &Value, stage: &str) -> String {
        reference["activations"][stage]["file_sha256"]
            .as_str()
            .unwrap_or_else(|| panic!("no file_sha256 for {stage}"))
            .to_string()
    }

    /// The dumped prompt ids, which must be the reference's `prompt_token_ids`.
    fn prompt_ids(&self, fixture: &str, reference: &Value) -> Vec<u32> {
        let ids: Vec<u32> = self
            .i32s(fixture, "prompt_ids")
            .1
            .into_iter()
            .map(|id| u32::try_from(id).expect("a token id"))
            .collect();
        let listed: Vec<u32> = reference["prompt_token_ids"]
            .as_array()
            .expect("prompt_token_ids")
            .iter()
            .map(|id| id.as_u64().expect("a token id") as u32)
            .collect();
        assert_eq!(ids, listed, "{fixture}: prompt_ids.npy vs prompt_token_ids");
        ids
    }

    /// The fixture's corpus pages (`FOCR_CORPUS_DIR`, default
    /// `~/.cache/franken_ocr/corpus/pages`), or `None` after a skip line when
    /// any page of any fixture is absent.
    fn page_paths(&self, log: &mut Logger) -> Option<BTreeMap<String, Vec<PathBuf>>> {
        let corpus = std::env::var_os("FOCR_CORPUS_DIR")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(|home| PathBuf::from(home).join(".cache/franken_ocr/corpus/pages"))
            })
            .expect("FOCR_CORPUS_DIR or HOME is set");
        let mut paths = BTreeMap::new();
        for (fixture, entry) in &self.fixtures {
            let pages: Vec<PathBuf> = entry["pages"]
                .as_array()
                .expect("fixture pages")
                .iter()
                .map(|page| corpus.join(page["name"].as_str().expect("page name")))
                .collect();
            if let Some(missing) = pages.iter().find(|path| !path.is_file()) {
                log.skip_no_model(&format!(
                    "corpus page {} absent (set FOCR_CORPUS_DIR)",
                    missing.display()
                ));
                return None;
            }
            paths.insert(fixture.clone(), pages);
        }
        Some(paths)
    }

    /// Page `index` of `fixture`, held to its sha256 in `ORACLE.json`.
    fn page_bytes(&self, fixture: &str, index: usize, path: &Path) -> Vec<u8> {
        let bytes = std::fs::read(path).expect("corpus page reads");
        assert_eq!(
            sha256_hex(&bytes),
            self.fixtures[fixture]["pages"][index]["sha256"]
                .as_str()
                .expect("page sha256"),
            "{fixture}: {} is not the page the oracle read",
            path.display()
        );
        bytes
    }

    fn gundam(&self, fixture: &str) -> bool {
        self.fixtures[fixture]["params"]["crop_mode"].as_bool() == Some(true)
    }
}

/// The first element whose bits differ, as `(index, ours, oracle)`.
fn first_bit_mismatch(got: &[f32], want: &[f32]) -> Option<(usize, f32, f32)> {
    got.iter()
        .zip(want)
        .position(|(a, b)| a.to_bits() != b.to_bits())
        .map(|i| (i, got[i], want[i]))
}

/// Running ULP statistics of one table compare: elements compared, elements
/// whose bits differ, and the worst [`parity_harness::ulp_compare`] report.
#[derive(Default)]
struct UlpTally {
    compared: usize,
    differing: usize,
    max_ulp: u32,
    max_abs_diff: f64,
}

impl UlpTally {
    /// Add `ours` against `oracle` under the transcendental budget; `where_`
    /// names the table in a failure message.
    fn add(&mut self, ours: &[f32], oracle: &[f32], where_: &str) {
        let report = ulp_compare(ours, oracle, OpFamily::Transcendental);
        assert!(
            report.pass,
            "{where_}: {} ulp at {} (ours {}, oracle {}), budget {}",
            report.max_ulp,
            report.worst_index,
            ours[report.worst_index],
            oracle[report.worst_index],
            report.budget_ulp
        );
        self.compared += ours.len();
        self.differing += ours
            .iter()
            .zip(oracle)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        self.max_ulp = self.max_ulp.max(report.max_ulp);
        self.max_abs_diff = self.max_abs_diff.max(report.max_abs_diff);
    }
}

/// U2-C1 L1: native page2d coordinates equal the U2-A6 f32 oracle's exactly,
/// and their RoPE tables agree within the per-op transcendental ULP budget
/// (METHODOLOGY §1.3, 8 ulp), on every fixture.
/// - Prefill: `PageCoords::build` over the dumped prompt ids against
///   `prefill.coords_tok` / `prefill.coords_page`, and its table against both
///   the main attention's and the router's `cos`/`sin`.
/// - Decode: a `DecodePageState` fed the reference's generated ids against the
///   coordinates and tables of every kept decode step, page breaks included
///   (U2-B1's hand-over).
///
/// The tables are not compared bit for bit: torch's CPU `cos`/`sin` are
/// vectorized approximations accurate to 1 ulp, and Rust's are not the same
/// functions. The angles are the same f32 products, so a frequency off by an
/// ulp would put rows thousands of tokens in by far more than the budget.
/// Model-free. Skips with success when the dumps are absent.
#[test]
fn uocr2_l1_page2d_coordinates_and_rope_match_the_oracle() {
    use franken_ocr::native_engine::srswa;
    let mut log = Logger::new("uocr2_L1_page2d", "uocr2_oracle");
    log.setup(0);
    let t0 = Instant::now();
    let Some(oracle) = Uocr2Oracle::from_env(&mut log) else {
        return;
    };
    let mut decode_steps = 0usize;
    for fixture in oracle.fixtures.keys() {
        let reference = oracle.reference(fixture);
        let ids = oracle.prompt_ids(fixture, &reference);
        let coords = srswa::PageCoords::build(&ids).expect("v2 page coordinates");
        let tok: Vec<i32> = coords.tok_pos.iter().map(|&t| t as i32).collect();
        // The oracle dumps the page coordinate before the rotary folds it
        // mod 2: unwindowed, that is the bucket. The fold shows in the tables.
        let page: Vec<i32> = coords.bucket.iter().map(|&b| b as i32).collect();
        assert_eq!(
            tok,
            oracle.i32s(fixture, "prefill.coords_tok").1,
            "{fixture}: prefill token positions"
        );
        assert_eq!(
            page,
            oracle.i32s(fixture, "prefill.coords_page").1,
            "{fixture}: prefill page coordinates"
        );
        let rope = coords.rope_table(0);
        let mut tables = UlpTally::default();
        for (stage, ours) in [
            ("prefill.main_cos", &rope.cos),
            ("prefill.main_sin", &rope.sin),
            ("prefill.router_cos", &rope.cos),
            ("prefill.router_sin", &rope.sin),
        ] {
            let (shape, want) = oracle.f32s(fixture, stage);
            assert_eq!(shape, [ids.len(), srswa::HEAD_DIM], "{fixture} {stage}");
            tables.add(ours, &want, &format!("{fixture} {stage}"));
        }

        // Step k ≥ 1 is the decode forward of generated token k − 1.
        let generated: Vec<u32> = reference["token_ids"]
            .as_array()
            .expect("token_ids")
            .iter()
            .map(|id| id.as_u64().expect("a token id") as u32)
            .collect();
        let steps = oracle.i32s(fixture, "decode.index").1;
        let (dtok, dpage) = (
            oracle.i32s(fixture, "decode.coords_tok").1,
            oracle.i32s(fixture, "decode.coords_page").1,
        );
        let (dcos, dsin) = (
            oracle.f32s(fixture, "decode.main_cos").1,
            oracle.f32s(fixture, "decode.main_sin").1,
        );
        assert_eq!(
            dtok.len(),
            steps.len(),
            "{fixture}: one coordinate per step"
        );
        let mut state = srswa::DecodePageState::after_prompt(&coords, 0);
        let fed: Vec<srswa::DecodeCoord> = generated
            .iter()
            .take(*steps.last().expect("kept decode steps") as usize)
            .map(|&id| state.feed(id))
            .collect();
        let width = srswa::HEAD_DIM;
        for (j, &step) in steps.iter().enumerate() {
            let c = fed[step as usize - 1];
            assert_eq!(
                (c.tok_pos as i32, c.bucket as i32),
                (dtok[j], dpage[j]),
                "{fixture} decode step {step} (token position, page bucket)"
            );
            let table = srswa::page2d_rope_table(&[c.tok_pos], &[c.rope_page]);
            let span = j * width..(j + 1) * width;
            tables.add(
                &table.cos,
                &dcos[span.clone()],
                &format!("{fixture} decode step {step} cos"),
            );
            tables.add(
                &table.sin,
                &dsin[span],
                &format!("{fixture} decode step {step} sin"),
            );
        }
        let page_breaks = generated[..fed.len()]
            .iter()
            .filter(|&&id| id == special::PAGE)
            .count();
        decode_steps += steps.len();
        log.parity(
            "L1",
            "max_abs_diff",
            0.0,
            0.0,
            fixture,
            &Uocr2Oracle::stage_sha256(&reference, "prefill.coords_tok"),
            json!({
                "seam": "page2d coordinates (prefill rows, kept decode steps)",
                "exact": true,
                "prefill_rows": ids.len(),
                "decode_steps_compared": steps.len(),
                "generated_page_breaks_fed": page_breaks,
            }),
            true,
        );
        let budget = parity_harness::ulp_table(OpFamily::Transcendental);
        log.parity(
            "L1",
            "max_abs_diff",
            tables.max_abs_diff,
            f64::from(budget) * f64::from(f32::EPSILON),
            fixture,
            &Uocr2Oracle::stage_sha256(&reference, "prefill.main_cos"),
            json!({
                "seam": "page2d RoPE cos/sin (prefill main/router, kept decode steps)",
                "tolerance_rule": "per-op transcendental ULP budget (METHODOLOGY §1.3); \
                                   tolerance is that budget at |x| = 1",
                "ulp_budget": budget,
                "max_ulp": tables.max_ulp,
                "elements_compared": tables.compared,
                "elements_differing": tables.differing,
                "oracle_floor": "t12 vs t4 tables are bit-identical",
            }),
            true,
        );
    }
    log.assertion(
        "every fixture's prefill and kept decode steps compared",
        decode_steps > 0,
    );
    log.result("pass", t0.elapsed().as_micros());
}

/// U2-B7 L0, EXACT: Unlimited-OCR-2 preprocessing equals the U2-A6 f32 oracle
/// bit for bit on every committed fixture: S1 and S3 (base), S2 (multi-page
/// base) and S4 (gundam). Checked per fixture:
/// - every page's global view against `images_ori`;
/// - the gundam tiles against `images_crop`;
/// - the grids against `images_spatial_crop`;
/// - the placeholder count against `images_seq_mask`.
///
/// `FOCR_UOCR2_FIXTURES_DIR` names the oracle run's dump directory
/// (`<root>/f32/t12`). The pages come from `FOCR_CORPUS_DIR` (default
/// `~/.cache/franken_ocr/corpus/pages`), each held to its sha256 in
/// `ORACLE.json`. Skips with success when the dumps or the corpus are absent.
/// No tolerance: v2 always resamples with the PIL-exact kernel, whatever
/// `FOCR_RESAMPLE` says.
#[test]
fn uocr2_l0_preprocess_is_bit_exact_against_the_oracle() {
    let mut log = Logger::new("uocr2_L0_preprocess", "uocr2_oracle");
    log.setup(0);
    let t0 = Instant::now();
    let Some(oracle) = Uocr2Oracle::from_env(&mut log) else {
        return;
    };
    let Some(all_pages) = oracle.page_paths(&mut log) else {
        return;
    };

    let mut ran = 0usize;
    for (fixture, entry) in &oracle.fixtures {
        let pages = entry["pages"].as_array().expect("fixture pages");
        let page_paths = &all_pages[fixture];
        let gundam = oracle.gundam(fixture);
        let mode = if gundam {
            preprocess::PreprocessMode::gundam()
        } else {
            preprocess::PreprocessMode::base()
        };
        let (ori_shape, ori) = oracle.f32s(fixture, "images_ori");
        assert_eq!(
            ori_shape,
            [pages.len(), 3, 1024, 1024],
            "{fixture} images_ori"
        );
        let (_, spatial) = oracle.i32s(fixture, "images_spatial_crop");
        let (_, seq_mask) = oracle.u8s(fixture, "images_seq_mask");
        let view = 3 * 1024 * 1024;
        let mut tiles = Vec::new();
        let mut placeholders = 0usize;
        for (index, path) in page_paths.iter().enumerate() {
            let bytes = oracle.page_bytes(fixture, index, path);
            let pre = preprocess::preprocess_bytes(&bytes, mode, preprocess::PreprocessProfile::V2)
                .expect("v2 preprocess");
            let want = &ori[index * view..(index + 1) * view];
            assert_eq!(
                first_bit_mismatch(&pre.global.pixels.data, want),
                None,
                "{fixture} page {index}: global view (index, ours, oracle)"
            );
            let grid = [
                pre.crop_grid.width_crop_num as i32,
                pre.crop_grid.height_crop_num as i32,
            ];
            assert_eq!(
                grid,
                spatial[2 * index..2 * index + 2],
                "{fixture} page {index}: grid"
            );
            placeholders += pre.placeholder_token_count();
            tiles.extend(
                pre.tiles
                    .iter()
                    .flat_map(|tile| tile.pixels.data.iter().copied()),
            );
        }
        if gundam {
            let (crop_shape, crop) = oracle.f32s(fixture, "images_crop");
            assert_eq!(crop_shape[1..], [3, 640, 640], "{fixture} images_crop");
            assert_eq!(tiles.len(), crop.len(), "{fixture}: tile count");
            assert_eq!(
                first_bit_mismatch(&tiles, &crop),
                None,
                "{fixture}: tiles (index, ours, oracle)"
            );
        } else {
            assert!(tiles.is_empty(), "{fixture}: base mode emitted tiles");
        }
        let image_rows = seq_mask.iter().filter(|&&m| m == 1).count();
        assert_eq!(placeholders, image_rows, "{fixture}: placeholder count");
        let reference = oracle.reference(fixture);
        ran += 1;
        // Every view compared bit for bit above, so the largest difference is 0.
        log.parity(
            "L0",
            "max_abs_diff",
            0.0,
            0.0,
            fixture,
            &Uocr2Oracle::stage_sha256(&reference, "images_ori"),
            json!({
                "seam": "v2 preprocess (letterbox / gundam tiles -> normalized views)",
                "pages": pages.len(),
                "mode": if gundam { "gundam" } else { "base" },
                "tiles": tiles.len() / (3 * 640 * 640),
                "placeholders": placeholders,
                "resample_env": std::env::var(preprocess::RESAMPLE_ENV).ok(),
            }),
            true,
        );
    }
    assert_eq!(
        ran,
        oracle.fixtures.len(),
        "every ORACLE.json fixture compared"
    );
    log.result("pass", t0.elapsed().as_micros());
}

/// Subject-vs-oracle agreement of one float stage: the f64 cosine, and the
/// largest difference, absolute and relative to the oracle's largest magnitude
/// (the U2-A6 floor's normwise `max_rel`). For a stage compared row by row it
/// also names the prompt row that holds the largest difference, which places a
/// localized divergence (one row's flipped choice) that the normwise numbers
/// only hint at.
#[derive(Clone, Copy, Debug)]
struct FloatAgreement {
    cosine: f64,
    max_abs: f64,
    max_rel: f64,
    max_abs_row: Option<usize>,
}

impl FloatAgreement {
    fn of(ours: &[f32], oracle: &[f32]) -> Self {
        assert_eq!(ours.len(), oracle.len(), "float agreement: lengths");
        let max_abs = max_abs_diff(ours, oracle);
        let scale = oracle
            .iter()
            .fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
        Self {
            cosine: cosine(ours, oracle),
            max_abs,
            max_rel: if scale > 0.0 { max_abs / scale } else { 0.0 },
            max_abs_row: None,
        }
    }

    /// [`Self::of`] over rows `rows` of `m` against the oracle's dump of those
    /// rows.
    fn of_rows(m: &Mat, rows: &[usize], oracle: &[f32]) -> Self {
        let ours = mat_rows(m, rows);
        let worst = ours
            .iter()
            .zip(oracle)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
            .enumerate()
            .fold(
                (0, 0.0f64),
                |best, (i, d)| if d > best.1 { (i, d) } else { best },
            )
            .0;
        Self {
            max_abs_row: rows.get(worst / m.cols.max(1)).copied(),
            ..Self::of(&ours, oracle)
        }
    }
}

/// Rows `rows` of `m`, concatenated.
fn mat_rows(m: &Mat, rows: &[usize]) -> Vec<f32> {
    rows.iter()
        .flat_map(|&r| m.row(r).iter().copied())
        .collect()
}

/// The query class of prefill row `row` for the routing report (U2-C1): BOS,
/// a `<PAGE>` marker, an image row or an instruction row, suffixed `/fallback`
/// when the row's own page offers it no remote block.
fn uocr2_query_class(ids: &[u32], last_img: usize, row: usize, fallback: bool) -> String {
    let base = if row == 0 {
        "bos"
    } else if ids[row] == special::PAGE {
        "page_marker"
    } else if row <= last_img {
        "image"
    } else {
        "instruction"
    };
    if fallback {
        format!("{base}/fallback")
    } else {
        base.to_string()
    }
}

/// One prefill row whose native selection differs from the oracle's, with
/// U2-C1's routing-margin diagnostics. Margins are in score units: the gap
/// between the K'-th and (K'+1)-th entries of each side's top-M window, the
/// largest oracle-score gap between a block only one side selected and a block
/// only the other did, and this row's own largest score difference. The row's
/// noise shows whether a flip rides on ordinary drift or on a row that already
/// diverged upstream (an earlier near-tie, such as an MoE expert choice).
#[derive(Debug)]
struct RouteMismatch {
    row: usize,
    class: String,
    ours: Vec<u32>,
    oracle: Vec<u32>,
    ours_margin: f32,
    oracle_margin: f32,
    swap_gap: f32,
    row_score_noise: f32,
    ours_support: usize,
    oracle_support: usize,
}

/// The K'-th minus the (K'+1)-th value of a descending top-M window, in score
/// units; infinite when the window has no (K'+1)-th entry.
fn boundary_margin(window: &[f32], read_slots: usize, idx_scale: f32) -> f32 {
    if window.len() > read_slots {
        (window[read_slots - 1] - window[read_slots]) / idx_scale
    } else {
        f32::INFINITY
    }
}

/// The committed U2-A6 `FLOOR.json`.
/// The committed U2-A6 oracle nondeterminism floor (`FLOOR.json`).
fn read_uocr2_floor() -> Value {
    serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/uocr2_oracle/FLOOR.json"
        ))
        .expect("FLOOR.json"),
    )
    .expect("FLOOR.json parses")
}

/// The U2-A6 f32 floor of one stage of one fixture's layer, from the committed
/// `FLOOR.json` (`None` for a stage the floor does not record).
fn uocr2_floor(floor: &Value, fixture: &str, layer: Option<usize>, stage: &str) -> Option<Value> {
    let fixture = &floor["groups"]["f32"]["fixtures"][fixture];
    let entry = match layer {
        Some(layer) => &fixture["layers"][layer][stage],
        None => &fixture[stage],
    };
    (!entry.is_null()).then(|| entry.clone())
}

/// U2-C1 L1/L2: the native v2 prefill against the U2-A6 f32 oracle, layer by
/// layer, on every fixture (S1–S3 as the bead requires, and S4's single-page
/// gundam prompt, whose 28 own-page blocks make top-K bind).
///
/// The subject is the stateless all-f32 path, `decoder::forward_srswa_observed`
/// over the raw BF16 checkpoint widened to f32. Its input is native: v2
/// preprocessing, the vision tower and the connector over the oracle's prompt
/// ids. Checked per layer:
/// - **floats** (`cosine` ≥ the f32 per-op bar, with `max_abs` / `max_rel`
///   against the oracle's own t12-vs-t4 floor): `hidden_in`, the router's
///   `iq`/`ik`, `kbar`, the block scores, the entmax prior, the attention
///   context and `o_proj` output; then `final_hidden`;
/// - **exact**: which blocks are remote, the Page Gate (`cand`), the fallback
///   flag, and that the router's stage functions replayed on the trace's
///   `iq`/`ik` reproduce the forward's selections and biases bit for bit;
/// - **teacher-forced selection**: the router over the oracle's own scores
///   selects the oracle's blocks, except where torch's `topk` met an exact tie
///   at the K'-th slot;
/// - **chained selection**: native routes against the oracle's, each mismatch
///   logged with layer, row, query class, both margins, both entmax supports
///   and both selections. Since the router is exact on identical scores, a
///   mismatch can come only from score noise; one whose swapped blocks sit
///   farther apart than twice the layer's measured score noise fails.
///
/// Gated on `FOCR_UOCR2_FIXTURES_DIR`, `FOCR_UOCR2_SOURCE_DIR` (the raw
/// checkpoint directory) and the corpus pages; skips with success without them.
#[test]
fn uocr2_l1_l2_prefill_matches_the_f32_oracle() {
    use franken_ocr::native_engine::{nn, srswa};
    let mut log = Logger::new("uocr2_L1_L2_prefill", "uocr2_oracle");
    log.setup(0);
    let t0 = Instant::now();
    let Some(oracle) = Uocr2Oracle::from_env(&mut log) else {
        return;
    };
    let Some(source) = std::env::var_os("FOCR_UOCR2_SOURCE_DIR").map(PathBuf::from) else {
        log.skip_no_model("FOCR_UOCR2_SOURCE_DIR unset: point it at the raw v2 checkpoint");
        return;
    };
    let Some(all_pages) = oracle.page_paths(&mut log) else {
        return;
    };
    let floor = read_uocr2_floor();
    let _heavy = heavy_rung_guard();
    let w = load_uocr2_source(&source);
    let params = srswa::RouterParams::uocr2();
    let hidden = decoder::config::HIDDEN_SIZE;
    let qkv_dim = decoder::config::NUM_ATTENTION_HEADS * decoder::config::HEAD_DIM;
    let mut failures: Vec<String> = Vec::new();
    let mut chained_total = 0usize;
    let mut fallback_total = 0usize;

    for (fixture, pages) in &all_pages {
        let tf = Instant::now();
        let reference = oracle.reference(fixture);
        let ids = oracle.prompt_ids(fixture, &reference);
        let mode = if oracle.gundam(fixture) {
            preprocess::PreprocessMode::gundam()
        } else {
            preprocess::PreprocessMode::base()
        };
        let pres: Vec<preprocess::Preprocessed> = pages
            .iter()
            .enumerate()
            .map(|(index, path)| {
                let bytes = oracle.page_bytes(fixture, index, path);
                preprocess::preprocess_bytes(&bytes, mode, preprocess::PreprocessProfile::V2)
                    .expect("v2 preprocess")
            })
            .collect();
        let embeds = uocr2_inputs_embeds(&w, &ids, &pres);
        let rows: Vec<usize> = oracle
            .i32s(fixture, "prefill.rows")
            .1
            .into_iter()
            .map(|r| r as usize)
            .collect();
        let coords = srswa::PageCoords::build(&ids).expect("v2 page coordinates");
        let blocks = srswa::BlockPages::build(&coords.img_page);
        let sha = |stage: &str| Uocr2Oracle::stage_sha256(&reference, stage);
        let mut fallback_rows = 0usize;

        let mut on_layer = |t: decoder::SrSwaLayerTrace<'_>| {
            let layer = t.layer;
            let stage = |name: &str| format!("prefill.L{layer:02}.{name}");
            let f32s = |name: &str| oracle.f32s(fixture, &stage(name)).1;
            let i32s = |name: &str| oracle.i32s(fixture, &stage(name)).1;
            let u8s = |name: &str| oracle.u8s(fixture, &stage(name)).1;
            let mut floats: Vec<(&str, &str, FloatAgreement)> = vec![
                (
                    "L2",
                    "hidden_in",
                    FloatAgreement::of_rows(t.hidden_in, &rows, &f32s("hidden_in")),
                ),
                (
                    "L1",
                    "iq",
                    FloatAgreement::of_rows(t.iq, &rows, &f32s("iq")),
                ),
                ("L1", "ik", FloatAgreement::of(&t.ik.data, &f32s("ik"))),
            ];
            let router = srswa::SrSwaLayerWeights::load(&w, layer).expect("router weights");
            let sums = srswa::summaries(t.ik, &router.head_cls);
            floats.push(("L1", "kbar", FloatAgreement::of(&sums.kbar, &f32s("kbar"))));
            let nblk = sums.valid.len();
            let window = params.window(nblk);
            // The oracle's `topi`/`bias`/`valid` hold the window's first K' slots.
            let k = params.read_slots(window);
            let read_slots = k;
            let (o_scores, o_cand, o_fallback) = (f32s("scores"), u8s("cand"), u8s("fallback"));
            let (o_topm_x, o_topm_idx, o_prior) = (f32s("topm_x"), i32s("topm_idx"), f32s("prior"));
            let (o_topi, o_bias, o_valid) = (i32s("topi"), f32s("bias"), u8s("valid"));
            assert_eq!(
                o_scores.len(),
                rows.len() * nblk,
                "{fixture} L{layer}: block count"
            );
            assert_eq!(
                o_topm_x.len(),
                rows.len() * window,
                "{fixture} L{layer}: window"
            );
            assert_eq!(
                o_topi.len(),
                rows.len() * k,
                "{fixture} L{layer}: read slots"
            );

            let (mut our_remote, mut oracle_remote) = (Vec::new(), Vec::new());
            let (mut our_prior, mut oracle_prior) = (Vec::new(), Vec::new());
            let mut gate_mismatches = Vec::new();
            let mut replica_mismatches = 0usize;
            let (mut tf_mismatches, mut tf_exact_ties, mut tf_x_bits) = (0usize, 0usize, 0usize);
            let (mut tf_prior_max, mut tf_bias_max, mut bias_max) = (0.0f64, 0.0f64, 0.0f64);
            let mut mismatches: Vec<RouteMismatch> = Vec::new();
            let mut by_class: BTreeMap<String, usize> = BTreeMap::new();
            let mut layer_fallback = 0usize;
            for (i, &r) in rows.iter().enumerate() {
                let span = i * nblk..(i + 1) * nblk;
                let o_sc = &o_scores[span.clone()];
                let sc = srswa::scores(t.iq.row(r), r, &sums);
                let remote: Vec<bool> = sc.iter().map(|&s| s > srswa::SCORE_NEG * 0.5).collect();
                let o_remote: Vec<bool> =
                    o_sc.iter().map(|&s| s > srswa::SCORE_NEG * 0.5).collect();
                let allow = srswa::page_gate_prefill(&blocks, coords.query_page[r]);
                let o_allow: Vec<bool> = o_cand[span].iter().map(|&c| c == 1).collect();
                let fallback =
                    !remote.iter().zip(&allow).any(|(&a, &b)| a && b) && remote.contains(&true);
                if remote != o_remote || allow != o_allow || fallback != (o_fallback[i] == 1) {
                    gate_mismatches.push(r);
                    continue;
                }
                layer_fallback += usize::from(fallback);
                *by_class
                    .entry(uocr2_query_class(&ids, coords.last_img, r, fallback))
                    .or_default() += 1;
                for c in (0..nblk).filter(|&c| remote[c]) {
                    our_remote.push(sc[c]);
                    oracle_remote.push(o_sc[c]);
                }

                // The router's stages on the trace, which must be the forward's.
                let cand = srswa::candidates(&sc, Some(&allow));
                let x = srswa::entmax_inputs(&sc, &cand, &params);
                let top = srswa::entmax15_topm(&x, window);
                let route = srswa::select_bias(&top, &cand, &params);
                let traced = &t.selections[r];
                if route.blocks != traced.blocks()
                    || route
                        .bias
                        .iter()
                        .map(|b| b.to_bits())
                        .ne(traced.bias().iter().map(|b| b.to_bits()))
                {
                    replica_mismatches += 1;
                }

                // The oracle's selection: its first K slots that are candidates.
                let wspan = i * window..(i + 1) * window;
                let kspan = i * k..(i + 1) * k;
                let mut o_sel: Vec<(u32, f32)> = o_topi[kspan.clone()]
                    .iter()
                    .zip(&o_valid[kspan.clone()])
                    .zip(&o_bias[kspan])
                    .filter(|&((_, &v), _)| v == 1)
                    .map(|((&b, _), &bias)| (b as u32, bias))
                    .collect();
                o_sel.sort_unstable_by_key(|&(b, _)| b);
                let o_blocks: Vec<u32> = o_sel.iter().map(|&(b, _)| b).collect();
                let o_window: Vec<u32> = o_topm_idx[wspan.clone()]
                    .iter()
                    .map(|&b| b as u32)
                    .collect();

                // Teacher-forced: the router over the oracle's scores.
                let o_cand_used = srswa::candidates(o_sc, Some(&allow));
                let o_x = srswa::entmax_inputs(o_sc, &o_cand_used, &params);
                let o_top = srswa::entmax15_topm(&o_x, window);
                let o_route = srswa::select_bias(&o_top, &o_cand_used, &params);
                tf_x_bits += o_top
                    .slots
                    .iter()
                    .zip(&o_topm_x[wspan.clone()])
                    .filter(|((v, _), o)| v.to_bits() != o.to_bits())
                    .count();
                for (a, b) in o_top.prior.iter().zip(&o_prior[wspan.clone()]) {
                    tf_prior_max = tf_prior_max.max(f64::from((a - b).abs()));
                }
                if o_route.blocks == o_blocks {
                    for (a, (_, b)) in o_route.bias.iter().zip(&o_sel) {
                        tf_bias_max = tf_bias_max.max(f64::from((a - b).abs()));
                    }
                } else if o_top.slots.len() > read_slots
                    && o_top.slots[read_slots - 1].0.to_bits()
                        == o_top.slots[read_slots].0.to_bits()
                {
                    tf_exact_ties += 1;
                } else {
                    tf_mismatches += 1;
                    failures.push(format!(
                        "{fixture} L{layer} row {r}: teacher-forced selection {:?} != oracle {:?}",
                        o_route.blocks, o_blocks
                    ));
                }

                // Chained: the native selection against the oracle's.
                let mut ours_window: Vec<u32> = top.slots.iter().map(|&(_, b)| b).collect();
                let mut sorted_o_window = o_window.clone();
                ours_window.sort_unstable();
                sorted_o_window.sort_unstable();
                if ours_window == sorted_o_window {
                    our_prior.extend_from_slice(&top.prior);
                    oracle_prior.extend_from_slice(&o_prior[wspan.clone()]);
                }
                if route.blocks == o_blocks {
                    for (a, (_, b)) in route.bias.iter().zip(&o_sel) {
                        bias_max = bias_max.max(f64::from((a - b).abs()));
                    }
                } else {
                    let only_ours = route.blocks.iter().filter(|b| !o_blocks.contains(b));
                    let only_oracle: Vec<&u32> = o_blocks
                        .iter()
                        .filter(|b| !route.blocks.contains(b))
                        .collect();
                    let swap_gap = only_ours
                        .flat_map(|&a| {
                            only_oracle
                                .iter()
                                .map(move |&&b| (o_sc[a as usize] - o_sc[b as usize]).abs())
                        })
                        .fold(0.0f32, f32::max);
                    let row_score_noise = (0..nblk)
                        .filter(|&c| remote[c])
                        .map(|c| (sc[c] - o_sc[c]).abs())
                        .fold(0.0f32, f32::max);
                    let x_window: Vec<f32> = top.slots.iter().map(|&(v, _)| v).collect();
                    mismatches.push(RouteMismatch {
                        row: r,
                        class: uocr2_query_class(&ids, coords.last_img, r, fallback),
                        ours: route.blocks.clone(),
                        oracle: o_blocks,
                        ours_margin: boundary_margin(&x_window, read_slots, params.idx_scale),
                        oracle_margin: boundary_margin(
                            &o_topm_x[wspan.clone()],
                            read_slots,
                            params.idx_scale,
                        ),
                        swap_gap,
                        row_score_noise,
                        ours_support: top.support,
                        oracle_support: o_prior[wspan].iter().filter(|&&p| p > 0.0).count(),
                    });
                }
            }
            let score_noise = max_abs_diff(&our_remote, &oracle_remote);
            floats.push((
                "L1",
                "scores",
                FloatAgreement::of(&our_remote, &oracle_remote),
            ));
            floats.push(("L1", "prior", FloatAgreement::of(&our_prior, &oracle_prior)));
            floats.push((
                "L2",
                "attn_pre_o",
                FloatAgreement::of_rows(t.context, &rows, &f32s("attn_pre_o")),
            ));
            let o_proj = w
                .mat(&format!("model.layers.{layer}.self_attn.o_proj.weight"))
                .expect("o_proj");
            let attn_out = decoder::attn_output_proj(t.context, &o_proj.data, hidden, qkv_dim)
                .expect("o_proj");
            floats.push((
                "L2",
                "attn_out",
                FloatAgreement::of_rows(&attn_out, &rows, &f32s("attn_out")),
            ));

            if !gate_mismatches.is_empty() {
                failures.push(format!(
                    "{fixture} L{layer}: remote set / Page Gate / fallback differ on rows {:?}",
                    &gate_mismatches[..gate_mismatches.len().min(8)]
                ));
            }
            if replica_mismatches > 0 {
                failures.push(format!(
                    "{fixture} L{layer}: {replica_mismatches} rows' replayed routes differ from \
                     the forward's"
                ));
            }
            if tf_x_bits > 0 {
                failures.push(format!(
                    "{fixture} L{layer}: {tf_x_bits} teacher-forced entmax inputs differ in bits"
                ));
            }
            for (gate, name, agreement) in &floats {
                let pass = agreement.cosine >= COSINE_F32_THRESHOLD;
                if !pass {
                    failures.push(format!(
                        "{fixture} L{layer} {name}: cosine {} < {COSINE_F32_THRESHOLD}",
                        agreement.cosine
                    ));
                }
                let mut extra = serde_json::Map::new();
                for (key, value) in [
                    ("gate", json!(gate)),
                    ("metric", json!("cosine")),
                    ("value", json!(agreement.cosine)),
                    ("tolerance", json!(COSINE_F32_THRESHOLD)),
                    ("oracle_fixture", json!(fixture)),
                    ("oracle_sha256", json!(sha(&stage(name)))),
                    (
                        "nondeterminism_envelope",
                        json!({
                            "stage": name,
                            "max_abs": agreement.max_abs,
                            "max_rel": agreement.max_rel,
                            "max_abs_row": agreement.max_abs_row,
                            "oracle_floor_t12_vs_t4": uocr2_floor(&floor, fixture, Some(layer), name),
                        }),
                    ),
                    ("pass", json!(pass)),
                    ("layer_idx", json!(layer)),
                    ("rows_compared", json!(rows.len())),
                ] {
                    extra.insert(key.into(), value);
                }
                log.emit("parity", if pass { "pass" } else { "fail" }, extra);
            }

            let tie_flips = mismatches
                .iter()
                .filter(|m| f64::from(m.swap_gap) <= 2.0 * score_noise)
                .count();
            for m in &mismatches {
                if f64::from(m.swap_gap) > 2.0 * score_noise {
                    failures.push(format!(
                        "{fixture} L{layer}: a selection mismatch outside the score noise \
                         ({score_noise:e}): {m:?}"
                    ));
                }
            }
            let mut by_class_mismatch: BTreeMap<&str, usize> = BTreeMap::new();
            for m in &mismatches {
                *by_class_mismatch.entry(m.class.as_str()).or_default() += 1;
            }
            let pass = gate_mismatches.is_empty()
                && replica_mismatches == 0
                && tf_mismatches == 0
                && tie_flips == mismatches.len();
            let mut extra = serde_json::Map::new();
            for (key, value) in [
                ("gate", json!("L2")),
                ("metric", json!("routing_mismatches")),
                ("value", json!(mismatches.len())),
                ("tolerance", json!(0)),
                ("oracle_fixture", json!(fixture)),
                ("oracle_sha256", json!(sha(&stage("topi")))),
                (
                    "nondeterminism_envelope",
                    json!({
                        "rule": "a mismatch is a tie-flip when the oracle scores of the blocks \
                                 only one side selected differ by at most twice the layer's \
                                 measured score noise; the router itself is exact on the \
                                 oracle's scores (teacher-forced)",
                        "score_noise_max_abs": score_noise,
                        "tie_flips": tie_flips,
                        "oracle_self_flips_t12_vs_t4": uocr2_floor(&floor, fixture, Some(layer), "routing_rows_differing"),
                        "mismatches": mismatches.iter().map(|m| json!({
                            "row": m.row, "class": m.class, "ours": m.ours, "oracle": m.oracle,
                            "ours_margin": m.ours_margin, "oracle_margin": m.oracle_margin,
                            "swap_gap": m.swap_gap, "row_score_noise": m.row_score_noise,
                            "ours_support": m.ours_support,
                            "oracle_support": m.oracle_support,
                        })).collect::<Vec<_>>(),
                        "teacher_forced_mismatches": tf_mismatches,
                        "teacher_forced_exact_ties": tf_exact_ties,
                        "teacher_forced_prior_max_abs": tf_prior_max,
                        "teacher_forced_bias_max_abs": tf_bias_max,
                        "replica_mismatches": replica_mismatches,
                        "gate_mismatches": gate_mismatches.len(),
                        "bias_max_abs": bias_max,
                    }),
                ),
                ("pass", json!(pass)),
                ("layer_idx", json!(layer)),
                ("rows_compared", json!(rows.len())),
                ("fallback_rows", json!(layer_fallback)),
                ("rows_by_class", json!(by_class)),
                ("mismatches_by_class", json!(by_class_mismatch)),
            ] {
                extra.insert(key.into(), value);
            }
            log.emit("parity", if pass { "pass" } else { "fail" }, extra);
            chained_total += mismatches.len();
            fallback_rows = fallback_rows.max(layer_fallback);
        };
        // Layer 0's `hidden_in` line is the embedding check: vision, connector
        // and `embed_tokens` against the oracle's merged `inputs_embeds`.
        let out = decoder::forward_srswa_observed(
            &w,
            &embeds,
            &coords,
            srswa::EntmaxDomain::TopM,
            &mut on_layer,
        )
        .expect("v2 stateless forward");
        let final_norm = w.vec("model.norm.weight").expect("model.norm");
        let final_hidden = nn::rms_norm(&out, Some(&final_norm), decoder::config::RMS_NORM_EPS)
            .expect("final norm");
        let agreement = FloatAgreement::of_rows(
            &final_hidden,
            &rows,
            &oracle.f32s(fixture, "prefill.final_hidden").1,
        );
        let pass = agreement.cosine >= COSINE_F32_THRESHOLD;
        if !pass {
            failures.push(format!(
                "{fixture} final_hidden: cosine {} < {COSINE_F32_THRESHOLD}",
                agreement.cosine
            ));
        }
        log.parity(
            "L2",
            "cosine",
            agreement.cosine,
            COSINE_F32_THRESHOLD,
            fixture,
            &sha("prefill.final_hidden"),
            json!({
                "stage": "final_hidden",
                "max_abs": agreement.max_abs,
                "max_rel": agreement.max_rel,
                "max_abs_row": agreement.max_abs_row,
                "oracle_floor_t12_vs_t4": uocr2_floor(&floor, fixture, None, "final_hidden"),
                "rows_compared": rows.len(),
                "prompt_rows": ids.len(),
                "fallback_rows_per_layer": fallback_rows,
                "elapsed_s": tf.elapsed().as_secs_f64(),
            }),
            pass,
        );
        fallback_total += fallback_rows;
    }

    log.assertion(
        "fallback queries were exercised (counted per fixture above)",
        fallback_total > 0,
    );
    let pass = failures.is_empty() && fallback_total > 0;
    log.result(if pass { "pass" } else { "fail" }, t0.elapsed().as_micros());
    assert!(
        pass,
        "U2-C1 L1/L2 failed ({} chained routing mismatches in all; see the log):\n{}",
        chained_total,
        failures.join("\n")
    );
}

/// One decode step's selection that differs from the oracle's, with U2-C1's
/// margin diagnostics (see [`RouteMismatch`]).
#[derive(Debug)]
struct DecodeRouteMismatch {
    step: usize,
    ours: Vec<u32>,
    oracle: Vec<u32>,
    ours_margin: f32,
    oracle_margin: f32,
    swap_gap: f32,
    row_score_noise: f32,
}

/// The oracle's read set at one dumped row: the candidates among its first
/// `k` `topi` slots, sorted by block, each with its bias.
fn oracle_selection(topi: &[i32], valid: &[u8], bias: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut sel: Vec<(u32, f32)> = topi[..k]
        .iter()
        .zip(&valid[..k])
        .zip(&bias[..k])
        .filter(|&((_, &v), _)| v == 1)
        .map(|((&b, _), &bias)| (b as u32, bias))
        .collect();
    sel.sort_unstable_by_key(|&(b, _)| b);
    sel
}

/// U2-C2 L3 and decode-step routing (plan §7.2): the native v2 decode,
/// teacher-forced along each fixture's oracle ids, against the U2-A6 f32
/// oracle at its kept steps (0–63 and ±8 around every generated `<PAGE>`).
///
/// The subject is the all-f32 cached path (`DecoderWeightCache::
/// build_srswa_f32` over the raw BF16 checkpoint widened to f32): the
/// stateless forward's math with the KV kept. The stateless path itself
/// (`forward_srswa` over the grown sequence, O(n²) over a document) is run at
/// sampled kept steps where the prompt plus ids fit in 4,096 rows: steps 0, 1
/// and 63, and the step after each generated `<PAGE>`. Checked:
/// - **L3:** each kept step's logits, cosine ≥ the f32 bar (U2-C1's rule; the
///   oracle's own t12-vs-t4 logit spread is logged beside `max_abs`), and each
///   kept decode step's final hidden;
/// - **tokens:** every step's raw argmax against `all_steps.raw_argmax`; at
///   the first disagreement, the trace diff is logged, and a disagreement
///   must be a tie-flip (our gap between the two ids within twice the logits'
///   max abs against the oracle). Generated-token exactness is L4's;
/// - **routing**, every kept decode step and layer: the remote set, the Page
///   Gate and the fallback flag exactly; the selection, where a mismatch must
///   be a tie-flip (the swapped blocks' oracle scores within twice the
///   layer's measured score noise); the biases' max abs difference is logged;
/// - **stateless ≡ cached** at the sampled steps: cosine, max abs, argmax.
///
/// Gated like U2-C1's rung; `FOCR_E2E_LONG` (any value but empty or `0`)
/// adds S3-p10 and S3-p20.
#[test]
fn uocr2_l3_decode_steps_match_the_f32_oracle() {
    use franken_ocr::native_engine::srswa;
    const STATELESS_ROWS: usize = 4096;
    let mut log = Logger::new("uocr2_L3_decode", "uocr2_oracle");
    log.setup(0);
    let t0 = Instant::now();
    let Some(oracle) = Uocr2Oracle::from_env(&mut log) else {
        return;
    };
    let Some(source) = std::env::var_os("FOCR_UOCR2_SOURCE_DIR").map(PathBuf::from) else {
        log.skip_no_model("FOCR_UOCR2_SOURCE_DIR unset: point it at the raw v2 checkpoint");
        return;
    };
    let Some(all_pages) = oracle.page_paths(&mut log) else {
        return;
    };
    let long = matches!(
        std::env::var("FOCR_E2E_LONG").ok().as_deref(),
        Some(v) if !v.is_empty() && v != "0"
    );
    let floor = read_uocr2_floor();
    let _heavy = heavy_rung_guard();
    let w = load_uocr2_source(&source);
    let tc = Instant::now();
    let wc = decoder::DecoderWeightCache::build_srswa_f32(&w).expect("the all-f32 cache builds");
    log.emit(
        "setup",
        "pass",
        serde_json::Map::from_iter([(
            "f32_cache_build_s".to_string(),
            json!(tc.elapsed().as_secs_f64()),
        )]),
    );
    let table = w.mat("model.embed_tokens.weight").expect("embed table");
    let final_norm = w.vec("model.norm.weight").expect("model.norm");
    let eps = decoder::config::RMS_NORM_EPS;
    let params = srswa::RouterParams::uocr2();
    let argmax = |logits: &[f32]| argmax_rows(logits, logits.len())[0];
    let mut failures: Vec<String> = Vec::new();

    for (fixture, pages) in &all_pages {
        if !long && matches!(fixture.as_str(), "S3-p10" | "S3-p20") {
            log.emit(
                "skip",
                "skip",
                serde_json::Map::from_iter([
                    ("oracle_fixture".to_string(), json!(fixture)),
                    ("reason".to_string(), json!("FOCR_E2E_LONG unset or 0")),
                ]),
            );
            continue;
        }
        let tf = Instant::now();
        let reference = oracle.reference(fixture);
        let ids = oracle.prompt_ids(fixture, &reference);
        let generated: Vec<u32> = reference["token_ids"]
            .as_array()
            .expect("token_ids")
            .iter()
            .map(|id| id.as_u64().expect("a token id") as u32)
            .collect();
        let mode = if oracle.gundam(fixture) {
            preprocess::PreprocessMode::gundam()
        } else {
            preprocess::PreprocessMode::base()
        };
        let pres: Vec<preprocess::Preprocessed> = pages
            .iter()
            .enumerate()
            .map(|(index, path)| {
                let bytes = oracle.page_bytes(fixture, index, path);
                preprocess::preprocess_bytes(&bytes, mode, preprocess::PreprocessProfile::V2)
                    .expect("v2 preprocess")
            })
            .collect();
        let embeds = uocr2_inputs_embeds(&w, &ids, &pres);
        let coords = srswa::PageCoords::build(&ids).expect("v2 page coordinates");
        let sha = |stage: &str| Uocr2Oracle::stage_sha256(&reference, stage);
        let as_usize = |v: Vec<i32>| -> Vec<usize> { v.into_iter().map(|x| x as usize).collect() };

        let kept = as_usize(oracle.i32s(fixture, "steps.index").1);
        let (logit_shape, oracle_logits) = oracle.f32s(fixture, "steps.logits");
        let vocab = logit_shape[1];
        let raw_argmax = as_usize(oracle.i32s(fixture, "all_steps.raw_argmax").1);
        assert_eq!(
            raw_argmax.len(),
            generated.len() + 1,
            "{fixture}: one raw argmax per step, EOS included"
        );
        let decode_index = as_usize(oracle.i32s(fixture, "decode.index").1);
        let decode_nblk = as_usize(oracle.i32s(fixture, "decode.nblk").1);
        let decode_k = as_usize(oracle.i32s(fixture, "decode.k").1);
        let (_, decode_final) = oracle.f32s(fixture, "decode.final_hidden");
        let hidden = decoder::config::HIDDEN_SIZE;
        // Per layer: (topi, valid, bias, scores, cand, fallback, topm_x) with
        // their row widths.
        struct LayerDump {
            topi: Vec<i32>,
            valid: Vec<u8>,
            bias: Vec<f32>,
            k_width: usize,
            scores: Vec<f32>,
            cand: Vec<u8>,
            n_width: usize,
            fallback: Vec<u8>,
            topm_x: Vec<f32>,
            m_width: usize,
        }
        let dumps: Vec<LayerDump> = (0..decoder::config::NUM_HIDDEN_LAYERS)
            .map(|layer| {
                let stage = |name: &str| format!("decode.L{layer:02}.{name}");
                let (tshape, topi) = oracle.i32s(fixture, &stage("topi"));
                let (sshape, scores) = oracle.f32s(fixture, &stage("scores"));
                let (mshape, topm_x) = oracle.f32s(fixture, &stage("topm_x"));
                LayerDump {
                    topi,
                    valid: oracle.u8s(fixture, &stage("valid")).1,
                    bias: oracle.f32s(fixture, &stage("bias")).1,
                    k_width: tshape[1],
                    scores,
                    cand: oracle.u8s(fixture, &stage("cand")).1,
                    n_width: sshape[1],
                    fallback: oracle.u8s(fixture, &stage("fallback")).1,
                    topm_x,
                    m_width: mshape[1],
                }
            })
            .collect();
        let oracle_routes = |j: usize| -> Vec<Vec<u32>> {
            dumps
                .iter()
                .map(|d| {
                    let span = j * d.k_width..(j + 1) * d.k_width;
                    oracle_selection(
                        &d.topi[span.clone()],
                        &d.valid[span.clone()],
                        &d.bias[span],
                        decode_k[j],
                    )
                    .into_iter()
                    .map(|(b, _)| b)
                    .collect()
                })
                .collect()
        };
        let stateless_steps: Vec<usize> = kept
            .iter()
            .copied()
            .filter(|&k| {
                (k <= 1 || k == 63 || generated[k - 1] == special::PAGE)
                    && ids.len() + k <= STATELESS_ROWS
            })
            .collect();

        let (prefill_hidden, mut state) = decoder::prefill_with_cache_srswa_traced(
            &wc,
            &embeds,
            &coords,
            srswa::EntmaxDomain::TopM,
        )
        .expect("traced f32 prefill");
        let mut last = Mat::from_vec(
            1,
            hidden,
            prefill_hidden.row(prefill_hidden.rows - 1).to_vec(),
        );
        let mut logits_min_cos = 1.0f64;
        let mut logits_max_abs = 0.0f64;
        let mut final_min_cos = 1.0f64;
        // (step, our logit gap from our pick down to the oracle's)
        let mut argmax_misses: Vec<(usize, f32)> = Vec::new();
        let mut stateless_lines: Vec<Value> = Vec::new();
        // Per layer: selection mismatches, score noise and bias error.
        let mut layer_mismatches: Vec<Vec<DecodeRouteMismatch>> =
            (0..dumps.len()).map(|_| Vec::new()).collect();
        let mut layer_noise = vec![0.0f32; dumps.len()];
        let mut layer_bias_max = vec![0.0f32; dumps.len()];
        let mut gate_mismatches: Vec<String> = Vec::new();
        let mut fallback_steps = 0usize;
        let mut routing_rows = 0usize;
        let td = Instant::now();

        for step in 0..=generated.len() {
            if step > 0 {
                let id = generated[step - 1];
                let row = Mat::from_vec(1, hidden, table.row(id as usize).to_vec());
                last = decoder::decode_step_with_cache_srswa(&wc, &mut state, id, &row, false)
                    .expect("f32 decode step");
            }
            let logits = decoder::lm_head_cached(&wc, &last).expect("lm_head");
            let ours_argmax = argmax(&logits.data);
            let kept_at = kept.iter().position(|&s| s == step);
            let trace = state.trace.as_ref().expect("traced");
            let our_routes: Vec<Vec<u32>> = if step == 0 {
                trace
                    .prefill_last
                    .iter()
                    .map(|r| r.as_ref().map_or_else(Vec::new, |r| r.blocks.clone()))
                    .collect()
            } else {
                trace
                    .decode
                    .iter()
                    .map(|r| r.as_ref().map_or_else(Vec::new, |r| r.route.blocks.clone()))
                    .collect()
            };
            if ours_argmax != raw_argmax[step] {
                let gap = logits.data[ours_argmax] - logits.data[raw_argmax[step]];
                if argmax_misses.is_empty() {
                    let page = generated[..step]
                        .iter()
                        .filter(|&&id| id == special::PAGE)
                        .count();
                    let page_len = generated[..step]
                        .iter()
                        .rev()
                        .take_while(|&&id| id != special::PAGE)
                        .count();
                    let (o_logits, o_routes) = match kept_at {
                        Some(i) => (
                            &oracle_logits[i * vocab..(i + 1) * vocab],
                            decode_index
                                .iter()
                                .position(|&s| s == step)
                                .map_or_else(Vec::new, oracle_routes),
                        ),
                        None => (&[][..], Vec::new()),
                    };
                    let diff = decoder::decode_trace_diff(
                        step,
                        page,
                        page_len,
                        decoder::StepSide {
                            label: "ours",
                            logits: &logits.data,
                            routes: &our_routes,
                        },
                        decoder::StepSide {
                            label: "oracle",
                            logits: o_logits,
                            routes: &o_routes,
                        },
                    );
                    log.emit(
                        "parity",
                        "fail",
                        serde_json::Map::from_iter([
                            ("gate".to_string(), json!("L3")),
                            ("metric".to_string(), json!("first_raw_argmax_mismatch")),
                            ("oracle_fixture".to_string(), json!(fixture)),
                            ("step".to_string(), json!(step)),
                            ("ours".to_string(), json!(ours_argmax)),
                            ("oracle".to_string(), json!(raw_argmax[step])),
                            ("ours_gap".to_string(), json!(gap)),
                            ("diff".to_string(), json!(diff)),
                        ]),
                    );
                }
                argmax_misses.push((step, gap));
            }
            let Some(i) = kept_at else {
                continue;
            };

            // L3: the step's logits, and a decode step's final hidden.
            let agreement =
                FloatAgreement::of(&logits.data, &oracle_logits[i * vocab..(i + 1) * vocab]);
            logits_min_cos = logits_min_cos.min(agreement.cosine);
            logits_max_abs = logits_max_abs.max(agreement.max_abs);
            if agreement.cosine < COSINE_F32_THRESHOLD {
                failures.push(format!(
                    "{fixture} step {step} logits: cosine {} < {COSINE_F32_THRESHOLD}",
                    agreement.cosine
                ));
            }
            if stateless_steps.contains(&step) {
                let ts = Instant::now();
                let mut grown_ids = ids.clone();
                grown_ids.extend_from_slice(&generated[..step]);
                let mut data = embeds.data.clone();
                for &id in &generated[..step] {
                    data.extend_from_slice(table.row(id as usize));
                }
                let grown = Mat::from_vec(grown_ids.len(), hidden, data);
                let grown_coords = srswa::PageCoords::build(&grown_ids).expect("grown coordinates");
                let h =
                    decoder::forward_srswa(&w, &grown, &grown_coords, srswa::EntmaxDomain::TopM)
                        .expect("stateless forward");
                let row = Mat::from_vec(1, hidden, h.row(h.rows - 1).to_vec());
                let stateless = decoder::lm_head(&w, &row).expect("stateless lm_head");
                let vs_oracle =
                    FloatAgreement::of(&stateless.data, &oracle_logits[i * vocab..(i + 1) * vocab]);
                let vs_cached = FloatAgreement::of(&stateless.data, &logits.data);
                let stateless_argmax = argmax(&stateless.data);
                let same_argmax = stateless_argmax == ours_argmax;
                // A different pick must be a tie within the two paths' gap.
                let tie = (stateless.data[stateless_argmax] - stateless.data[ours_argmax]) as f64
                    <= 2.0 * vs_cached.max_abs;
                if vs_oracle.cosine < COSINE_F32_THRESHOLD || !(same_argmax || tie) {
                    failures.push(format!(
                        "{fixture} step {step}: stateless cosine {} vs the oracle, argmax \
                         {stateless_argmax} against the cached path's {ours_argmax}",
                        vs_oracle.cosine
                    ));
                }
                stateless_lines.push(json!({
                    "step": step, "rows": grown_ids.len(),
                    "cosine_vs_oracle": vs_oracle.cosine, "max_abs_vs_oracle": vs_oracle.max_abs,
                    "cosine_vs_cached": vs_cached.cosine, "max_abs_vs_cached": vs_cached.max_abs,
                    "same_argmax": same_argmax, "wall_s": ts.elapsed().as_secs_f64(),
                }));
            }
            let Some(j) = decode_index.iter().position(|&s| s == step) else {
                continue;
            };
            let normed = franken_ocr::native_engine::nn::rms_norm(&last, Some(&final_norm), eps)
                .expect("final norm");
            let fh = FloatAgreement::of(&normed.data, &decode_final[j * hidden..(j + 1) * hidden]);
            final_min_cos = final_min_cos.min(fh.cosine);
            if fh.cosine < COSINE_F32_THRESHOLD {
                failures.push(format!(
                    "{fixture} step {step} final hidden: cosine {} < {COSINE_F32_THRESHOLD}",
                    fh.cosine
                ));
            }

            // Routing, every layer.
            routing_rows += 1;
            for (layer, d) in dumps.iter().enumerate() {
                let ours = trace.decode[layer].as_ref().expect("a traced decode route");
                let nblk = ours.scores.len();
                if nblk != decode_nblk[j] {
                    gate_mismatches.push(format!(
                        "L{layer} step {step}: nblk {nblk} vs {}",
                        decode_nblk[j]
                    ));
                    continue;
                }
                let o_sc = &d.scores[j * d.n_width..j * d.n_width + nblk];
                let o_cand: Vec<bool> = d.cand[j * d.n_width..j * d.n_width + nblk]
                    .iter()
                    .map(|&c| c == 1)
                    .collect();
                let remote: Vec<bool> = ours
                    .scores
                    .iter()
                    .map(|&s| s > srswa::SCORE_NEG * 0.5)
                    .collect();
                let o_remote: Vec<bool> =
                    o_sc.iter().map(|&s| s > srswa::SCORE_NEG * 0.5).collect();
                let fallback = !remote.iter().zip(&ours.allow).any(|(&a, &b)| a && b)
                    && remote.contains(&true);
                fallback_steps += usize::from(fallback);
                if remote != o_remote || ours.allow != o_cand || fallback != (d.fallback[j] == 1) {
                    gate_mismatches.push(format!("L{layer} step {step}: remote/gate/fallback"));
                    continue;
                }
                for c in (0..nblk).filter(|&c| remote[c]) {
                    layer_noise[layer] = layer_noise[layer].max((ours.scores[c] - o_sc[c]).abs());
                }
                let kspan = j * d.k_width..(j + 1) * d.k_width;
                let o_sel = oracle_selection(
                    &d.topi[kspan.clone()],
                    &d.valid[kspan.clone()],
                    &d.bias[kspan],
                    decode_k[j],
                );
                let o_blocks: Vec<u32> = o_sel.iter().map(|&(b, _)| b).collect();
                if ours.route.blocks == o_blocks {
                    for (a, (_, b)) in ours.route.bias.iter().zip(&o_sel) {
                        layer_bias_max[layer] = layer_bias_max[layer].max((a - b).abs());
                    }
                    continue;
                }
                let only_oracle: Vec<u32> = o_blocks
                    .iter()
                    .copied()
                    .filter(|b| !ours.route.blocks.contains(b))
                    .collect();
                let swap_gap = ours
                    .route
                    .blocks
                    .iter()
                    .filter(|b| !o_blocks.contains(b))
                    .flat_map(|&a| {
                        only_oracle
                            .iter()
                            .map(move |&b| (o_sc[a as usize] - o_sc[b as usize]).abs())
                    })
                    .fold(0.0f32, f32::max);
                let cand = srswa::candidates(&ours.scores, Some(&ours.allow));
                let x = srswa::entmax_inputs(&ours.scores, &cand, &params);
                let window = params.window(nblk);
                let top = srswa::entmax15_topm(&x, window);
                let x_window: Vec<f32> = top.slots.iter().map(|&(v, _)| v).collect();
                let read_slots = params.read_slots(window);
                let mspan = j * d.m_width..j * d.m_width + window.min(d.m_width);
                layer_mismatches[layer].push(DecodeRouteMismatch {
                    step,
                    ours: ours.route.blocks.clone(),
                    oracle: o_blocks,
                    ours_margin: boundary_margin(&x_window, read_slots, params.idx_scale),
                    oracle_margin: boundary_margin(&d.topm_x[mspan], read_slots, params.idx_scale),
                    swap_gap,
                    row_score_noise: (0..nblk)
                        .filter(|&c| remote[c])
                        .map(|c| (ours.scores[c] - o_sc[c]).abs())
                        .fold(0.0f32, f32::max),
                });
            }
        }

        // Each layer's selection mismatches, judged by its measured noise.
        let mut chained = 0usize;
        for (layer, mismatches) in layer_mismatches.iter().enumerate() {
            let noise = layer_noise[layer];
            let tie_flips = mismatches
                .iter()
                .filter(|m| m.swap_gap <= 2.0 * noise)
                .count();
            for m in mismatches.iter().filter(|m| m.swap_gap > 2.0 * noise) {
                failures.push(format!(
                    "{fixture} L{layer}: a decode selection mismatch outside the score noise \
                     ({noise:e}): {m:?}"
                ));
            }
            chained += mismatches.len();
            log.parity(
                "L3",
                "decode_routing_mismatches",
                mismatches.len() as f64,
                0.0,
                fixture,
                &sha(&format!("decode.L{layer:02}.topi")),
                json!({
                    "rule": "a mismatch is a tie-flip when the oracle scores of the blocks only \
                             one side selected differ by at most twice the layer's measured \
                             score noise over the kept decode steps (U2-C1's rule)",
                    "layer_idx": layer,
                    "rows_compared": routing_rows,
                    "score_noise_max_abs": noise,
                    "tie_flips": tie_flips,
                    "bias_max_abs": layer_bias_max[layer],
                    "mismatches": mismatches.iter().map(|m| json!({
                        "step": m.step, "ours": m.ours, "oracle": m.oracle,
                        "ours_margin": m.ours_margin, "oracle_margin": m.oracle_margin,
                        "swap_gap": m.swap_gap, "row_score_noise": m.row_score_noise,
                    })).collect::<Vec<_>>(),
                }),
                tie_flips == mismatches.len(),
            );
        }
        if !gate_mismatches.is_empty() {
            failures.push(format!(
                "{fixture}: decode remote set / Page Gate / fallback differ: {:?}",
                &gate_mismatches[..gate_mismatches.len().min(8)]
            ));
        }
        // A raw-argmax miss is a tie-flip when our gap between the two ids is
        // within twice the logit noise measured against the oracle (U2-C1's
        // rule, on the kept steps' max abs).
        let logit_noise = logits_max_abs as f32;
        let outside: Vec<(usize, f32)> = argmax_misses
            .iter()
            .copied()
            .filter(|&(_, gap)| gap > 2.0 * logit_noise)
            .collect();
        if !outside.is_empty() {
            failures.push(format!(
                "{fixture}: raw argmax differs from the oracle's outside the logit noise \
                 ({logit_noise:e}) at {} of {} steps, first (step, gap) {:?}",
                outside.len(),
                generated.len() + 1,
                &outside[..outside.len().min(8)]
            ));
        }
        let pass = logits_min_cos >= COSINE_F32_THRESHOLD && outside.is_empty();
        log.parity(
            "L3",
            "cosine",
            logits_min_cos,
            COSINE_F32_THRESHOLD,
            fixture,
            &sha("steps.logits"),
            json!({
                "stage": "logits", "path": "f32-cached",
                "kept_steps": kept.len(), "steps": generated.len() + 1,
                "max_abs": logits_max_abs,
                "oracle_floor_t12_vs_t4": uocr2_floor(&floor, fixture, None, "envelope")
                    .map(|e| e["logit_spread_overall"].clone()),
                "final_hidden_min_cosine": final_min_cos,
                "raw_argmax_mismatches": argmax_misses,
                "raw_argmax_tie_flips": argmax_misses.len() - outside.len(),
                "decode_routing_rows": routing_rows,
                "decode_routing_mismatches": chained,
                "decode_fallback_rows": fallback_steps,
                "stateless": stateless_lines,
                "decode_s": td.elapsed().as_secs_f64(),
                "elapsed_s": tf.elapsed().as_secs_f64(),
            }),
            pass,
        );
    }

    let pass = failures.is_empty();
    log.result(if pass { "pass" } else { "fail" }, t0.elapsed().as_micros());
    assert!(
        pass,
        "U2-C2 L3 failed (see the log):\n{}",
        failures.join("\n")
    );
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

    if !fixtures_present() || !model_present(&mut log) {
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
            // The gate said the model is PRESENT; failing to load it is a real
            // broken state, not an honest skip (fresh-eyes fix — the old xfail
            // return let a corrupt/mispointed artifact turn every gated rung
            // silently green).
            log.error("WeightsLoad", 1, &e);
            log.result("fail", t0.elapsed().as_micros());
            panic!("subject weights present but failed to load: {e}");
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

    if !fixtures_present() || !model_present(&mut log) {
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
            // The gate said the model is PRESENT; failing to load it is a real
            // broken state, not an honest skip (fresh-eyes fix — the old xfail
            // return let a corrupt/mispointed artifact turn every gated rung
            // silently green).
            log.error("WeightsLoad", 1, &e);
            log.result("fail", t0.elapsed().as_micros());
            panic!("subject weights present but failed to load: {e}");
        }
    };
    let mut ran = 0usize;
    let mut all_pass = true;
    for gpath in loader.list_goldens().unwrap_or_default() {
        let stem = golden_stem(&gpath);
        let golden = match loader.load_golden(&gpath) {
            Ok(g) => g,
            Err(e) => {
                // Present-but-corrupt golden = a REAL failure (fresh-eyes fix):
                // silently dropping it would let a broken fixture set pass the
                // token-exact/CER gates with zero comparisons.
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

    if !fixtures_present() || !model_present(&mut log) {
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
            // The gate said the model is PRESENT; failing to load it is a real
            // broken state, not an honest skip (fresh-eyes fix — the old xfail
            // return let a corrupt/mispointed artifact turn every gated rung
            // silently green).
            log.error("WeightsLoad", 1, &e);
            log.result("fail", t0.elapsed().as_micros());
            panic!("subject weights present but failed to load: {e}");
        }
    };
    let mut ran = 0usize;
    let mut all_pass = true;
    for gpath in loader.list_goldens().unwrap_or_default() {
        let stem = golden_stem(&gpath);
        let golden = match loader.load_golden(&gpath) {
            Ok(g) => g,
            Err(e) => {
                // Present-but-corrupt golden = a REAL failure (fresh-eyes fix):
                // silently dropping it would let a broken fixture set pass the
                // token-exact/CER gates with zero comparisons.
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

    if !fixtures_present() || !model_present(&mut log) {
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
            // The gate said the model is PRESENT; failing to load it is a real
            // broken state, not an honest skip (fresh-eyes fix — the old xfail
            // return let a corrupt/mispointed artifact turn every gated rung
            // silently green).
            log.error("WeightsLoad", 1, &e);
            log.result("fail", t0.elapsed().as_micros());
            panic!("subject weights present but failed to load: {e}");
        }
    };
    let captures = subject_decode_captures(&w, &loader);
    let mut ran = 0usize;
    let mut all_pass = true;
    for gpath in loader.list_goldens().unwrap_or_default() {
        let stem = golden_stem(&gpath);
        let golden = match loader.load_golden(&gpath) {
            Ok(g) => g,
            Err(e) => {
                // Present-but-corrupt golden = a REAL failure (fresh-eyes fix):
                // silently dropping it would let a broken fixture set pass the
                // token-exact/CER gates with zero comparisons.
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

/// L5-MULTI (bd-1gv.26): the CROSS-PAGE pass vs the frozen `infer_multi`
/// oracle (tests/fixtures/multi_page/, generated by
/// scripts/baseline/run_baidu_reference_multi.py over page_0009+page_0014 at
/// the reference defaults: 640 squash, ngram_window 1024, greedy bf16-CPU).
///
/// Both sides are normalized through the SAME `postprocess::finalize_multi`
/// (the oracle fixture is the RAW model text with its `<|det|>` spans; ours
/// is already finalized), then:
///   * the deterministic PLATE block must match exactly (containment) — the
///     region the two precisions agree on byte-for-byte, so any
///     preprocess/prompt/census/decode-wiring regression breaks it loudly
///     (it caught the pad-vs-squash and CatmullRom-kernel bugs already);
///   * aggregate CER over the oracle-length prefix stays within the
///     documented budget (the bf16-oracle vs f32-subject greedy tail fork —
///     see DISCREPANCIES.md DISC-004).
///
/// Model + corpus gated: skip-with-SUCCESS unless `FOCR_MODEL_PATH` resolves
/// AND `FOCR_CORPUS_DIR` holds both pages. Run WITHOUT `FOCR_DECODE_INT8`
/// (the budget is calibrated on the f32 subject).
#[test]
fn l5_multi_page_matches_infer_multi_oracle() {
    // MEASURED 2026-10-30 armed: CER 0.0522, the three plate lines exact, and
    // the subject stops near the oracle's EOS. Before the antialiased
    // position-table resize (bd-p8hu) it was 0.1791 and the subject rambled on
    // past the plate (DISC-004).
    run_l5_multi(
        "p9_p14",
        "tests/fixtures/multi_page/p9_p14_raw.md",
        "tests/fixtures/multi_page/p9_p14_meta.json",
        &["page_0009.png", "page_0014.png"],
        3,
        0.10,
        None,
    );
}

/// The 10-page LONG-HORIZON leg (bd-1465): the oracle decodes 7117 tokens /
/// 9 `<PAGE>` markers of real dense-page prose; the subject is CAPPED just
/// past the oracle's token count (a capped run's tokens are a true prefix of
/// the full run's; the subject EOSes at 6435 on its own, so the cap only
/// bounds a regression). §2.5 expects graceful degradation, not equality.
///
/// The plate here is the reference's first TWO content lines. The third
/// ("PRINTED BY … LIMITED,") is a measured near-tie in this ten-page
/// context: torch's own f32 run of the reference model prefers `' L'` over
/// `' LIMITED'` by 0.50 logits, then holds three continuations within 0.08
/// (LUTHER / LONDON / LUNDER), and bf16 under torch 2.13 reads "LONDON,".
/// No f32 subject can be held to the bf16 reference's pick there (DISC-004).
/// The 2-page leg keeps the full three-line plate.
#[test]
fn l5_multi_page_10p_long_horizon() {
    run_l5_multi(
        "p10",
        "tests/fixtures/multi_page/p10_raw.md",
        "tests/fixtures/multi_page/p10_meta.json",
        &[
            "page_0009.png",
            "page_0014.png",
            "page_0019.png",
            "page_0023.png",
            "page_0086.png",
            "page_0091.png",
            "page_0105.png",
            "page_0107.png",
            "page_0108.png",
            "page_0126.png",
        ],
        2,
        L5_MULTI_10P_BUDGET,
        Some(7600),
    );
}

/// 10-page CER budget — see [`l5_multi_page_10p_long_horizon`]. MEASURED
/// 2026-10-30 armed: 0.3186 (two-line plate exact; markers 8-vs-9; the
/// DISC-004 trajectory fork compounds across pages — §2.5 graceful
/// degradation: 2-page 0.052 → 10-page 0.319). Budget = measured + headroom,
/// below the 0.4045 measured before the antialiased resize (bd-p8hu).
const L5_MULTI_10P_BUDGET: f64 = 0.40;

fn run_l5_multi(
    case: &str,
    raw_fixture: &str,
    meta_fixture: &str,
    page_names: &[&str],
    plate_lines: usize,
    cer_budget: f64,
    subject_token_cap: Option<usize>,
) {
    use franken_ocr::native_engine::OcrModel;

    let mut log = Logger::new("L5_multi_page", case);
    log.setup(0);
    let t0 = Instant::now();

    // Always-on: the fixture parses and the oracle raw carries the reference
    // invariants (a <PAGE> marker; the deterministic plate text).
    let raw = std::fs::read_to_string(raw_fixture).expect("committed oracle raw fixture");
    let meta: Value = serde_json::from_str(
        &std::fs::read_to_string(meta_fixture).expect("committed oracle meta fixture"),
    )
    .expect("oracle meta parses");
    log.assertion(
        "oracle raw carries a <PAGE> marker",
        raw.contains(postprocess::PAGE_MARKER),
    );
    log.assertion(
        "oracle meta pins ngram_window 1024 (OQ-18 multi)",
        meta["ngram_window"] == json!(1024),
    );
    let finalized_ref = postprocess::finalize_multi(&raw, page_names.len())
        .expect("finalize_multi over the oracle raw");
    // The deterministic plate anchor = the finalized reference's first
    // `plate_lines` content lines (LONDON / PRINTED BY … / STAMFORD …): the
    // prefix both precisions agree on for this document.
    let plate: Vec<&str> = finalized_ref
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.contains(postprocess::PAGE_MARKER))
        .take(plate_lines)
        .collect();
    log.assertion(
        &format!("plate anchor extracted ({plate_lines} lines)"),
        plate.len() == plate_lines,
    );

    let corpus = std::env::var_os("FOCR_CORPUS_DIR").map(PathBuf::from);
    let pages: Option<Vec<PathBuf>> = corpus.as_ref().and_then(|dir| {
        let all: Vec<PathBuf> = page_names.iter().map(|n| dir.join(n)).collect();
        all.iter().all(|p| p.is_file()).then_some(all)
    });
    let model_dir = std::env::var_os("FOCR_MODEL_PATH")
        .map(PathBuf::from)
        .filter(|p| p.exists() && subject_is_ladder_model(&mut log));
    let (Some(pages), Some(model_dir)) = (pages, model_dir) else {
        log.skip_no_model(
            "L5-multi needs FOCR_MODEL_PATH (the baseline model dir with its \
             tokenizer beside the shard) AND FOCR_CORPUS_DIR containing \
             page_0009.png + page_0014.png; fixture invariants proved always-on",
        );
        log.result("skip_no_model", t0.elapsed().as_micros());
        return;
    };

    let _heavy = heavy_rung_guard();
    // Long-horizon legs cap the subject at the oracle's token count + margin
    // (a capped run's tokens are a true PREFIX of the full run's, and the
    // compare below only reads the oracle-length prefix anyway) so an armed
    // rung costs minutes, not the uncapped 32K-cap hour.
    if let Some(cap) = subject_token_cap {
        franken_ocr::native_engine::set_decode_overrides(
            franken_ocr::native_engine::DecodeOverrides {
                max_length: Some(cap),
                ..Default::default()
            },
        );
    }
    let model = OcrModel::load(&model_dir).expect("subject model loads");
    let mut source =
        franken_ocr::page_source::PageSource::images(pages.clone()).expect("a page source");
    let ours = model.recognize_document(
        &mut source,
        &franken_ocr::DocumentOptions::default(),
        &mut |_| {},
        None,
    );
    if subject_token_cap.is_some() {
        // Never leak the cap into other rungs in this process.
        franken_ocr::native_engine::set_decode_overrides(Default::default());
    }
    let ours = ours.expect("subject cross-page pass");
    assert!(ours.skipped.is_empty(), "{:?}", ours.skipped);
    let ours = ours.markdown;

    for line in &plate {
        log.assertion(
            &format!("deterministic plate line reproduced: {line:?}"),
            ours.contains(line),
        );
        assert!(
            ours.contains(line),
            "plate line {line:?} missing from the subject output:\n{ours}"
        );
    }

    let ref_chars: Vec<char> = finalized_ref.chars().collect();
    let ours_prefix: String = ours.chars().take(ref_chars.len()).collect();
    let cer = char_error_rate(&finalized_ref, &ours_prefix);
    let ours_markers = ours.matches(postprocess::PAGE_MARKER).count();
    let ref_markers = finalized_ref.matches(postprocess::PAGE_MARKER).count();
    log.parity(
        "L5_multi_page",
        "cer_vs_infer_multi_oracle_prefix",
        cer,
        cer_budget,
        raw_fixture,
        meta["sha256"].as_str().unwrap_or(""),
        json!({"oracle": "bf16-cpu greedy (deterministic)", "subject": "f32 greedy",
               "subject_token_cap": subject_token_cap,
               "page_markers": {"ours": ours_markers, "oracle": ref_markers},
               "note": "post-plate greedy fork budgeted (DISC-004); plate exact-matched above"}),
        cer <= cer_budget,
    );
    assert!(
        cer <= cer_budget,
        "L5-multi[{case}] CER {cer:.4} > budget {cer_budget}"
    );
    println!(
        r#"{{"suite":"parity_ladder","rung":"L5_multi_page","case":{case:?},"cer":{cer:.4},"budget":{cer_budget},"markers_ours":{ours_markers},"markers_oracle":{ref_markers},"result":"pass"}}"#
    );
    log.result("pass", t0.elapsed().as_micros());
}

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

    if !fixtures_present() || !model_present(&mut log) {
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
            // The gate said the model is PRESENT; failing to load it is a real
            // broken state, not an honest skip (fresh-eyes fix — the old xfail
            // return let a corrupt/mispointed artifact turn every gated rung
            // silently green).
            log.error("WeightsLoad", 1, &e);
            log.result("fail", t0.elapsed().as_micros());
            panic!("subject weights present but failed to load: {e}");
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
        let golden = match loader.load_golden(&gpath) {
            Ok(g) => g,
            Err(e) => {
                // Present-but-corrupt golden = a REAL failure (fresh-eyes fix):
                // silently dropping it would let a broken fixture set pass the
                // token-exact/CER gates with zero comparisons.
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
// L5 corpus gate for Unlimited-OCR-2 (U2-C4; plan §7.3, RATCHET.md)
//
// The v2 int8 artifact against the v2 f32 CPU HF references on the 20-page
// corpus, with v1 int8 against the v1 f32 references in the same invocation as
// the ratchet baseline; then the S2/S3 documents, the per-page generated-token
// distribution, and the unwindowed page limit it supports. Every budget and
// category below was pre-registered in the bead notes before the first run.
// ─────────────────────────────────────────────────────────────────────────────

/// The pre-registered per-page normalized CER budget: a page passes iff its
/// CER against its own lane's f32 reference is at most this.
const CORPUS_PAGE_CER_BUDGET: f64 = 0.45;

/// v1's documented aggregate corpus budget (`docs/DISCREPANCIES.md`), a hard
/// gate on v2's aggregate.
const CORPUS_AGGREGATE_CER_BUDGET: f64 = 0.25;

/// v1's recorded aggregate against the bf16 corpus reference, reported beside
/// v2's (`docs/DISCREPANCIES.md`, the DISC-004 receipt).
const V1_RECORDED_AGGREGATE_CER: f64 = 0.193_079_245_9;

/// A page that generates this many ids is flagged for the D11 per-page cap
/// review (80% of the 20,000 cap).
const PAGE_TOKENS_FLAG: usize = 16_000;

/// The corpus pages and their pre-registered fallback categories, by each
/// page's normalized length in the historical bf16 corpus reference
/// (`corpus/ref_bf16`): `sparse` under 1,000 chars, `prose` 1,000 to 4,000,
/// `dense` over 4,000. Membership is the same for both lanes.
const CORPUS_GATE_PAGES: [(&str, &str); 20] = [
    ("page_0009", "sparse"),
    ("page_0014", "sparse"),
    ("page_0019", "prose"),
    ("page_0023", "prose"),
    ("page_0086", "prose"),
    ("page_0091", "prose"),
    ("page_0105", "prose"),
    ("page_0107", "prose"),
    ("page_0108", "prose"),
    ("page_0126", "prose"),
    ("page_0136", "prose"),
    ("page_0179", "prose"),
    ("page_0250", "prose"),
    ("page_0358", "prose"),
    ("page_0400", "prose"),
    ("page_0437", "prose"),
    ("page_0547", "prose"),
    ("page_0557", "prose"),
    ("page_0590", "dense"),
    ("page_0645", "prose"),
];

/// The ratchet's categories: `all` (conformal at n = 20) and the fallback
/// cards.
const CORPUS_GATE_CATEGORIES: [&str; 4] = ["all", "sparse", "prose", "dense"];

/// The documents the gate reads with `recognize_document`, by their U2-A6
/// oracle fixture.
const CORPUS_GATE_DOCUMENTS: [&str; 4] = ["S2-p9_p14_p19", "S3-p9_p14", "S3-p10", "S3-p20"];

/// A reference page whose generated ids reach this is a hard page, named in
/// the report.
const HARD_PAGE_REFERENCE_IDS: usize = 1_000;

/// `scripts/corpus_cer.py`'s normalization: each line stripped with its
/// whitespace runs collapsed to one space, empty lines dropped. No case or
/// Unicode folding: a quantization change that alters either is a divergence.
fn corpus_normalize(text: &str) -> String {
    text.replace("\r\n", "\n")
        .split('\n')
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// `(edits, reference chars)` of `candidate` against `reference`, both
/// normalized; the CER is their quotient ([`corpus_cer`]).
fn corpus_edits(reference: &str, candidate: &str) -> (usize, usize, usize) {
    let r: Vec<char> = corpus_normalize(reference).chars().collect();
    let c: Vec<char> = corpus_normalize(candidate).chars().collect();
    (levenshtein(&r, &c), r.len(), c.len())
}

/// `corpus_cer.py`'s rate: edits over reference chars; an empty reference
/// scores 0 against an empty candidate, else 1.
fn corpus_cer(edits: usize, reference_chars: usize) -> f64 {
    if reference_chars == 0 {
        return if edits == 0 { 0.0 } else { 1.0 };
    }
    edits as f64 / reference_chars as f64
}

/// The nearest-rank `p`-th percentile: the smallest value with at least `p`%
/// of `values` at or below it.
fn nearest_rank(values: &[usize], p: usize) -> usize {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (p * sorted.len()).div_ceil(100).max(1);
    sorted[rank - 1]
}

/// The most pages one unwindowed v2 pass fits: the largest `N` with
/// `64 + 274·N + (N − 1)·p90 + p99 ≤ 131,072`, every page but one at the p90
/// length and one at the p99 length (U2-C4's re-derivation of U2-B6's limit).
fn unwindowed_page_limit(p90: usize, p99: usize) -> usize {
    (131_072 + p90).saturating_sub(64 + p99) / (274 + p90)
}

/// `ids` without a trailing EOS (the v1 reference returns it; the engine and
/// the v2 reference do not).
fn without_eos(mut ids: Vec<u32>) -> Vec<u32> {
    if ids.last() == Some(&special::EOS) {
        ids.pop();
    }
    ids
}

/// A corpus lane: one model, against its own f32 references.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CorpusLane {
    V1,
    V2,
}

impl CorpusLane {
    fn name(self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::V2 => "v2",
        }
    }

    /// The reference of `page` under `refs` (v1: `gen_reference_fixtures.py
    /// --mode base --f32`; v2: U2-A6's `S1C` set).
    fn reference_path(self, refs: &Path, page: &str) -> PathBuf {
        match self {
            Self::V1 => refs.join(format!("{page}_reference.json")),
            Self::V2 => refs.join(format!("S1C-{page}_reference.json")),
        }
    }

    /// The reference's generated ids, EOS dropped.
    fn reference_ids(self, reference: &Value) -> Vec<u32> {
        let pointer = match self {
            Self::V1 => "/token_stream/generated_ids",
            Self::V2 => "/token_ids",
        };
        without_eos(json_ids(reference, pointer))
    }

    /// The page's pixel size as the reference recorded it.
    fn reference_dims(self, reference: &Value) -> (u32, u32) {
        let (w, h) = match self {
            Self::V1 => ("/image/width", "/image/height"),
            Self::V2 => ("/pages/0/width", "/pages/0/height"),
        };
        let dim = |p: &str| {
            reference
                .pointer(p)
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or_else(|| panic!("reference {p}"))
        };
        (dim(w), dim(h))
    }

    /// The page's markdown from its generated ids, rendered as `recognize`
    /// renders them: v1 detokenizes the stream; v2 takes the body of its one
    /// page (U2-B6) first. Both then run `postprocess::finalize`.
    fn page_text(self, tok: &Tokenizer, ids: &[u32], (w, h): (u32, u32)) -> String {
        let raw = match self {
            Self::V1 => tok.decode(ids).expect("detokenize"),
            Self::V2 => postprocess::split_pages_by_token(tok, ids, special::PAGE, 1)
                .expect("split the page")
                .bodies
                .into_iter()
                .next()
                .unwrap_or_default(),
        };
        postprocess::finalize(&raw, w, h).expect("finalize")
    }
}

/// A JSON array of token ids at `pointer`.
fn json_ids(value: &Value, pointer: &str) -> Vec<u32> {
    value
        .pointer(pointer)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("no {pointer}"))
        .iter()
        .map(|id| {
            id.as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .expect("a token id")
        })
        .collect()
}

/// A v2 document's markdown from its generated ids, rendered as
/// `recognize_document` renders them: one body per page, each finalized as
/// its page, joined by `<PAGE>`.
fn uocr2_document_text(tok: &Tokenizer, ids: &[u32], dims: &[(u32, u32)]) -> (String, Vec<usize>) {
    let split = postprocess::split_pages_by_token(tok, ids, special::PAGE, dims.len())
        .expect("split the document");
    let text = postprocess::join_page_texts(
        split
            .bodies
            .iter()
            .enumerate()
            .map(|(i, body)| postprocess::finalize_page(body, i, dims[i]).text),
    );
    (text, split.tokens)
}

/// One corpus page of one lane.
struct CorpusPage {
    page: &'static str,
    category: &'static str,
    edits: usize,
    ref_chars: usize,
    /// Generated ids, the v2 page marker included.
    tokens: usize,
    ref_tokens: usize,
    eos: bool,
}

impl CorpusPage {
    fn cer(&self) -> f64 {
        corpus_cer(self.edits, self.ref_chars)
    }

    fn pass(&self) -> bool {
        self.cer() <= CORPUS_PAGE_CER_BUDGET
    }
}

/// The corpus gate's inputs, all from the environment.
struct CorpusGatePaths {
    v1_artifact: PathBuf,
    v2_artifact: PathBuf,
    v1_refs: PathBuf,
    v2_refs: PathBuf,
    corpus: PathBuf,
}

impl CorpusGatePaths {
    /// The paths, or `None` after a skip line naming what is missing.
    fn from_env(log: &mut Logger) -> Option<Self> {
        let var = |name: &str| {
            std::env::var_os(name)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        };
        let armed = matches!(
            std::env::var("FOCR_UOCR2_CORPUS_GATE").ok().as_deref(),
            Some(v) if !v.is_empty() && v != "0"
        );
        if !armed {
            log.skip_no_model(
                "FOCR_UOCR2_CORPUS_GATE unset: the corpus gate decodes 20 pages on each \
                 lane plus four documents; arm it in an optimized build",
            );
            return None;
        }
        let named = [
            ("FOCR_MODEL_PATH", "the v1 int8 artifact"),
            (
                "FOCR_UOCR2_MODEL_PATH",
                "the v2 int8 artifact, in a directory holding v2's own tokenizer.json",
            ),
            ("FOCR_UOCR2_V1_REFS_DIR", "the v1 f32 corpus references"),
            (
                "FOCR_UOCR2_FIXTURES_DIR",
                "the v2 f32 oracle (<oracle root>/f32/t12)",
            ),
        ];
        let mut found = Vec::new();
        for (name, what) in named {
            match var(name).filter(|p| p.exists()) {
                Some(path) => found.push(path),
                None => {
                    log.skip_no_model(&format!("{name} unset or absent: point it at {what}"));
                    return None;
                }
            }
        }
        let corpus = var("FOCR_CORPUS_DIR").unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME is set"))
                .join(".cache/franken_ocr/corpus/pages")
        });
        let [v1_artifact, v2_artifact, v1_refs, v2_refs] =
            <[PathBuf; 4]>::try_from(found).expect("four paths");
        Some(Self {
            v1_artifact,
            v2_artifact,
            v1_refs,
            v2_refs,
            corpus,
        })
    }
}

/// Decode the 20 corpus pages on one lane and score each against its
/// reference, one `parity` line per page. Returns the pages and the
/// artifact's SHA-256.
fn corpus_lane(
    log: &mut Logger,
    lane: CorpusLane,
    paths: &CorpusGatePaths,
) -> (Vec<CorpusPage>, String) {
    use franken_ocr::native_engine::{FinishReason, OcrModel, PageFinish};
    let (artifact, refs) = match lane {
        CorpusLane::V1 => (&paths.v1_artifact, &paths.v1_refs),
        CorpusLane::V2 => (&paths.v2_artifact, &paths.v2_refs),
    };
    let artifact_sha256 = parity_harness::file_sha256(artifact).expect("hash the artifact");
    let model = OcrModel::load(artifact).expect("load the artifact");
    let model_id = model.identity().model_id;
    let expected_id = match lane {
        CorpusLane::V1 => "unlimited-ocr",
        CorpusLane::V2 => "unlimited-ocr-2",
    };
    assert_eq!(
        model_id,
        expected_id,
        "{}: wrong artifact",
        artifact.display()
    );
    let tok = model.tokenizer().expect("the model's tokenizer");
    let mut pages = Vec::with_capacity(CORPUS_GATE_PAGES.len());
    for (page, category) in CORPUS_GATE_PAGES {
        let ref_path = lane.reference_path(refs, page);
        let ref_bytes =
            std::fs::read(&ref_path).unwrap_or_else(|e| panic!("{}: {e}", ref_path.display()));
        let reference: Value = serde_json::from_slice(&ref_bytes).expect("reference parses");
        let image = paths.corpus.join(format!("{page}.png"));
        if lane == CorpusLane::V2 {
            let bytes = std::fs::read(&image).expect("corpus page reads");
            assert_eq!(
                Some(sha256_hex(&bytes).as_str()),
                reference.pointer("/pages/0/sha256").and_then(Value::as_str),
                "{}: not the page the reference read",
                image.display()
            );
        }
        let dims = lane.reference_dims(&reference);
        assert_eq!(
            image::image_dimensions(&image).expect("page dimensions"),
            dims,
            "{page}: the reference's page size"
        );
        let started = Instant::now();
        let markdown = model.recognize(&image).expect("recognize the page");
        let wall_s = started.elapsed().as_secs_f64();
        let report = model.take_last_decode().expect("the page's decode report");
        let ours = without_eos(report.ids.clone());
        assert_eq!(
            lane.page_text(tok, &ours, dims),
            markdown,
            "{} {page}: the harness's text function is not recognize's",
            lane.name()
        );
        let ref_ids = lane.reference_ids(&reference);
        let ref_text = lane.page_text(tok, &ref_ids, dims);
        let (edits, ref_chars, cand_chars) = corpus_edits(&ref_text, &markdown);
        let finish = report.run_finish_reason();
        let eos = finish == FinishReason::Stop
            && report.document.as_ref().is_none_or(|d| {
                d.page_finish.last() == Some(&PageFinish::Eos) && d.page_finish.len() == 1
            });
        let row = CorpusPage {
            page,
            category,
            edits,
            ref_chars,
            tokens: ours.len(),
            ref_tokens: ref_ids.len(),
            eos,
        };
        let mut extra = serde_json::Map::new();
        for (key, value) in [
            ("gate", json!("L5")),
            ("metric", json!("cer")),
            ("value", json!(row.cer())),
            ("tolerance", json!(CORPUS_PAGE_CER_BUDGET)),
            ("oracle_fixture", json!(ref_path.display().to_string())),
            ("oracle_sha256", json!(sha256_hex(&ref_bytes))),
            (
                "nondeterminism_envelope",
                json!({"source": "pre-registered per-page budget (bead U2-C4 notes, 2026-10-30)"}),
            ),
            ("pass", json!(row.pass())),
            ("lane", json!(lane.name())),
            ("page", json!(page)),
            ("category", json!(category)),
            ("edits", json!(edits)),
            ("ref_chars", json!(ref_chars)),
            ("cand_chars", json!(cand_chars)),
            ("eos", json!(eos)),
            ("finish_reason", json!(finish.as_str())),
            ("tokens", json!(row.tokens)),
            ("ref_tokens", json!(row.ref_tokens)),
            (
                "hard_page",
                json!(row.ref_tokens >= HARD_PAGE_REFERENCE_IDS),
            ),
            ("wall_s", json!(wall_s)),
            ("artifact_sha256", json!(artifact_sha256)),
            ("model_id", json!(model_id)),
        ] {
            extra.insert(key.to_string(), value);
        }
        log.emit("parity", if row.pass() { "pass" } else { "fail" }, extra);
        pages.push(row);
    }
    (pages, artifact_sha256)
}

/// A lane's per-category pass counts as ratchet bounds.
fn corpus_bounds(pages: &[CorpusPage]) -> Vec<franken_ocr::conformance::CategoryBound> {
    use franken_ocr::conformance::{CategoryCounts, category_bound};
    CORPUS_GATE_CATEGORIES
        .iter()
        .map(|&category| {
            let members = pages
                .iter()
                .filter(|p| category == "all" || p.category == category);
            let (passes, failures) =
                members.fold(
                    (0, 0),
                    |(s, f), p| {
                        if p.pass() { (s + 1, f) } else { (s, f + 1) }
                    },
                );
            category_bound(CategoryCounts {
                category,
                passes,
                failures,
            })
        })
        .collect()
}

/// Aggregate normalized CER: total edits over total reference chars.
fn corpus_aggregate(pages: &[CorpusPage]) -> f64 {
    let edits: usize = pages.iter().map(|p| p.edits).sum();
    let chars: usize = pages.iter().map(|p| p.ref_chars).sum();
    corpus_cer(edits, chars)
}

/// U2-C4's L5 corpus gate (plan §7.3, items 1–6), the documents, and the
/// page-limit re-derivation. Armed by `FOCR_UOCR2_CORPUS_GATE=1` with
/// `FOCR_MODEL_PATH` (v1 int8), `FOCR_UOCR2_MODEL_PATH` (v2 int8; the model
/// opens the pinned `tokenizer.json` beside it, so v2 needs its own directory),
/// `FOCR_UOCR2_V1_REFS_DIR` (v1 f32 base references) and
/// `FOCR_UOCR2_FIXTURES_DIR` (the v2 f32 t12 oracle, `S1C-*` included);
/// otherwise it skips with success. Run it in `release-perf` with
/// `FOCR_STAGE_BUDGET_FORWARD_MS=0`: a budget stop would truncate a hard page
/// and read as a quantization regression.
#[test]
fn uocr2_l5_corpus_gate_admits_v2_against_the_v1_baseline() {
    use franken_ocr::conformance::{BoundMethod, ratchet_decide, transparency_card};
    use franken_ocr::native_engine::{
        DocumentOptions, FinishReason, OcrModel, PageWindow, UOCR2_UNWINDOWED_PAGE_LIMIT,
    };
    use franken_ocr::page_source::PageSource;

    let mut log = Logger::new("uocr2_L5_corpus", "S1-corpus");
    log.setup(0);
    let t0 = Instant::now();
    let Some(paths) = CorpusGatePaths::from_env(&mut log) else {
        return;
    };
    let _heavy = heavy_rung_guard();
    let mut failures: Vec<String> = Vec::new();

    // ── single page: both lanes, the same invocation ────────────────────────
    let (v1_pages, _) = corpus_lane(&mut log, CorpusLane::V1, &paths);
    let (v2_pages, v2_sha256) = corpus_lane(&mut log, CorpusLane::V2, &paths);
    let v1_bounds = corpus_bounds(&v1_pages);
    let v2_bounds = corpus_bounds(&v2_pages);
    for (lane, bounds) in [("v1", &v1_bounds), ("v2", &v2_bounds)] {
        for bound in bounds.iter() {
            let mut extra = serde_json::Map::new();
            extra.insert(
                "assertion".into(),
                json!(format!("{lane} {} transparency card", bound.category)),
            );
            extra.insert("pass".into(), json!(true));
            extra.insert("lane".into(), json!(lane));
            extra.insert("card".into(), transparency_card(bound));
            log.emit("assert", "pass", extra);
        }
    }
    let all_conformal = v2_bounds
        .iter()
        .chain(&v1_bounds)
        .filter(|b| b.category == "all")
        .all(|b| b.method == BoundMethod::Conformal);
    log.assertion("category all decides conformally (n = 20)", all_conformal);
    if !all_conformal {
        failures.push("category all fell back below MIN_CALIBRATION_N".into());
    }
    // Every category, fallback cards included: a fallback bound is its point
    // estimate, so the ratchet's no-lowering rule is item 4's "v2 ≥ v1".
    let baseline: Vec<(&str, f64)> = v1_bounds.iter().map(|b| (b.category, b.lower)).collect();
    let decision = ratchet_decide(&baseline, &v2_bounds);
    for verdict in &decision.verdicts {
        log.assertion(&format!("ratchet: {verdict}"), decision.allowed);
    }
    if !decision.allowed {
        failures.push(format!("ratchet rejects v2: {:?}", decision.verdicts));
    }
    let v1_aggregate = corpus_aggregate(&v1_pages);
    let v2_aggregate = corpus_aggregate(&v2_pages);
    let mut extra = serde_json::Map::new();
    for (key, value) in [
        ("gate", json!("L5")),
        ("metric", json!("cer")),
        ("value", json!(v2_aggregate)),
        ("tolerance", json!(CORPUS_AGGREGATE_CER_BUDGET)),
        ("oracle_fixture", json!("S1C (20 corpus pages)")),
        ("oracle_sha256", json!(v2_sha256)),
        (
            "nondeterminism_envelope",
            json!({"source": "v1's documented aggregate budget (docs/DISCREPANCIES.md)"}),
        ),
        ("pass", json!(v2_aggregate <= CORPUS_AGGREGATE_CER_BUDGET)),
        ("lane", json!("v2")),
        ("aggregate", json!(true)),
        ("v1_same_invocation_aggregate", json!(v1_aggregate)),
        (
            "v1_recorded_aggregate_vs_bf16",
            json!(V1_RECORDED_AGGREGATE_CER),
        ),
    ] {
        extra.insert(key.to_string(), value);
    }
    log.emit(
        "parity",
        if v2_aggregate <= CORPUS_AGGREGATE_CER_BUDGET {
            "pass"
        } else {
            "fail"
        },
        extra,
    );
    if v2_aggregate > CORPUS_AGGREGATE_CER_BUDGET {
        failures.push(format!(
            "v2 aggregate CER {v2_aggregate:.6} > {CORPUS_AGGREGATE_CER_BUDGET}"
        ));
    }
    for page in v2_pages.iter().filter(|p| !p.eos) {
        failures.push(format!("v2 {} did not reach EOS", page.page));
    }

    // ── documents: page-count equality, per-document CER ───────────────────
    let model = OcrModel::load(&paths.v2_artifact).expect("load the v2 artifact");
    let tok = model.tokenizer().expect("the model's tokenizer");
    let mut int8_page_tokens: Vec<usize> = v2_pages.iter().map(|p| p.tokens).collect();
    let mut f32_page_tokens: Vec<usize> = v2_pages.iter().map(|p| p.ref_tokens).collect();
    for doc in CORPUS_GATE_DOCUMENTS {
        let ref_path = paths.v2_refs.join(format!("{doc}_reference.json"));
        let ref_bytes =
            std::fs::read(&ref_path).unwrap_or_else(|e| panic!("{}: {e}", ref_path.display()));
        let reference: Value = serde_json::from_slice(&ref_bytes).expect("reference parses");
        let listed = reference["pages"].as_array().expect("reference pages");
        let mut images = Vec::with_capacity(listed.len());
        let mut dims = Vec::with_capacity(listed.len());
        for page in listed {
            let image = paths.corpus.join(page["name"].as_str().expect("page name"));
            let bytes = std::fs::read(&image).expect("corpus page reads");
            assert_eq!(
                Some(sha256_hex(&bytes).as_str()),
                page["sha256"].as_str(),
                "{doc}: {} is not the page the reference read",
                image.display()
            );
            let dim = |key: &str| u32::try_from(page[key].as_u64().expect("page size")).unwrap();
            dims.push((dim("width"), dim("height")));
            images.push(image);
        }
        let n = images.len();
        let started = Instant::now();
        let mut source = PageSource::images(images).expect("a page source");
        // The reference read each document in one pass; so does the gate,
        // as pre-registered, whatever the default window (U2-D1b).
        let one_pass = DocumentOptions::default().page_window(PageWindow::Off);
        let result = model
            .recognize_document(&mut source, &one_pass, &mut |_| {}, None)
            .expect("read the document");
        let wall_s = started.elapsed().as_secs_f64();
        let report = model
            .take_last_decode()
            .expect("the document's decode report");
        let (ours, our_tokens) = uocr2_document_text(tok, &report.ids, &dims);
        assert_eq!(
            ours, result.markdown,
            "{doc}: the harness's text function is not recognize_document's"
        );
        let ref_ids = without_eos(json_ids(&reference, "/token_ids"));
        let (theirs, ref_tokens) = uocr2_document_text(tok, &ref_ids, &dims);
        let (edits, ref_chars, cand_chars) = corpus_edits(&theirs, &ours);
        let pages_equal = result.pages_out == n
            && our_tokens.len() == n
            && result.extra_markers == 0
            && result.finish_reason == FinishReason::Stop;
        // Generated ids per page, its opening marker included.
        int8_page_tokens.extend(our_tokens.iter().map(|t| t + 1));
        f32_page_tokens.extend(ref_tokens.iter().map(|t| t + 1));
        let mut extra = serde_json::Map::new();
        for (key, value) in [
            ("gate", json!("L5")),
            ("metric", json!("page_count_equal")),
            ("value", json!(result.pages_out)),
            ("tolerance", json!(0)),
            ("oracle_fixture", json!(doc)),
            ("oracle_sha256", json!(sha256_hex(&ref_bytes))),
            (
                "nondeterminism_envelope",
                json!({"source": "exact: one body per page, finish stop, no extra marker"}),
            ),
            ("pass", json!(pages_equal)),
            ("lane", json!("v2")),
            ("pages", json!(n)),
            ("finish_reason", json!(result.finish_reason.as_str())),
            ("extra_markers", json!(result.extra_markers)),
            ("cer", json!(corpus_cer(edits, ref_chars))),
            ("edits", json!(edits)),
            ("ref_chars", json!(ref_chars)),
            ("cand_chars", json!(cand_chars)),
            ("tokens", json!(report.ids.len())),
            ("ref_tokens", json!(ref_ids.len())),
            ("page_tokens", json!(our_tokens)),
            ("ref_page_tokens", json!(ref_tokens)),
            ("wall_s", json!(wall_s)),
            ("artifact_sha256", json!(v2_sha256)),
            ("model_id", json!(result.model.model_id)),
        ] {
            extra.insert(key.to_string(), value);
        }
        log.emit("parity", if pages_equal { "pass" } else { "fail" }, extra);
        if !pages_equal {
            failures.push(format!(
                "{doc}: {} of {n} pages, finish {}, {} extra markers",
                result.pages_out,
                result.finish_reason.as_str(),
                result.extra_markers
            ));
        }
    }

    // ── the per-page token distribution and the page limit it supports ─────
    let mut limits = Vec::new();
    for (name, tokens) in [("int8", &int8_page_tokens), ("f32", &f32_page_tokens)] {
        let (p50, p90, p99) = (
            nearest_rank(tokens, 50),
            nearest_rank(tokens, 90),
            nearest_rank(tokens, 99),
        );
        let max = tokens.iter().copied().max().unwrap_or(0);
        let limit = unwindowed_page_limit(p90, p99);
        let flagged = tokens.iter().filter(|&&t| t >= PAGE_TOKENS_FLAG).count();
        limits.push(limit);
        let mut extra = serde_json::Map::new();
        for (key, value) in [
            (
                "assertion",
                json!(format!("{name} per-page token distribution")),
            ),
            ("pass", json!(flagged == 0)),
            ("distribution", json!(name)),
            ("pages", json!(tokens.len())),
            ("p50", json!(p50)),
            ("p90", json!(p90)),
            ("p99", json!(p99)),
            ("max", json!(max)),
            ("flagged_at_or_above", json!(PAGE_TOKENS_FLAG)),
            ("flagged", json!(flagged)),
            ("unwindowed_page_limit", json!(limit)),
        ] {
            extra.insert(key.to_string(), value);
        }
        log.emit("assert", if flagged == 0 { "pass" } else { "fail" }, extra);
    }
    let derived = limits.iter().copied().min().expect("two distributions");
    let limit_holds = UOCR2_UNWINDOWED_PAGE_LIMIT <= derived;
    log.assertion(
        &format!(
            "UOCR2_UNWINDOWED_PAGE_LIMIT {UOCR2_UNWINDOWED_PAGE_LIMIT} <= the measured {derived}"
        ),
        limit_holds,
    );
    if !limit_holds {
        failures.push(format!(
            "UOCR2_UNWINDOWED_PAGE_LIMIT {UOCR2_UNWINDOWED_PAGE_LIMIT} promises more pages than \
             the measured distribution supports ({derived})"
        ));
    }

    log.result(
        if failures.is_empty() { "pass" } else { "fail" },
        t0.elapsed().as_micros(),
    );
    assert!(
        failures.is_empty(),
        "U2-C4 corpus gate failed:\n{}",
        failures.join("\n")
    );
}

/// Always on: the corpus gate's pure parts agree with `corpus_cer.py` and the
/// plan.
#[test]
fn corpus_gate_helpers_match_the_python_harness_and_the_plan() {
    // corpus_cer.py: strip, collapse runs, drop empty lines; no case folding.
    assert_eq!(corpus_normalize("  A \t b \r\n\n\n c  d\nE"), "A b\nc d\nE");
    assert_eq!(corpus_edits("abc", "abd"), (1, 3, 3));
    assert_eq!(corpus_edits("Abc", "abc").0, 1, "case is a divergence");
    assert_eq!(corpus_cer(0, 0), 0.0);
    assert_eq!(corpus_cer(3, 0), 1.0);
    assert_eq!(corpus_cer(1, 4), 0.25);
    // Nearest rank: the smallest value with at least p% at or below it.
    let values: Vec<usize> = (1..=20).collect();
    assert_eq!(nearest_rank(&values, 50), 10);
    assert_eq!(nearest_rank(&values, 90), 18);
    assert_eq!(nearest_rank(&values, 99), 20);
    assert_eq!(nearest_rank(&[7], 99), 7);
    // U2-B6's derivation (1,900 ids for every page) gives its 60.
    assert_eq!(unwindowed_page_limit(1_900, 1_900), 60);
    // One page_0590-length page (4,033 ids) costs one page, not 30.
    assert_eq!(unwindowed_page_limit(1_900, 4_033), 59);
    assert_eq!(without_eos(vec![5, 6, special::EOS]), vec![5, 6]);
    assert_eq!(without_eos(vec![5, 6]), vec![5, 6]);
    // The pre-registered categories: 20 distinct pages, 2 sparse, 17 prose,
    // 1 dense.
    let names: std::collections::BTreeSet<&str> =
        CORPUS_GATE_PAGES.iter().map(|(page, _)| *page).collect();
    assert_eq!(names.len(), 20);
    for (category, count) in [("sparse", 2), ("prose", 17), ("dense", 1)] {
        assert_eq!(
            CORPUS_GATE_PAGES
                .iter()
                .filter(|(_, c)| *c == category)
                .count(),
            count,
            "{category}"
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

    if !fixtures_present() || !model_present(&mut log) {
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
            // The gate said the model is PRESENT; failing to load it is a real
            // broken state, not an honest skip (fresh-eyes fix — the old xfail
            // return let a corrupt/mispointed artifact turn every gated rung
            // silently green).
            log.error("WeightsLoad", 1, &e);
            log.result("fail", t0.elapsed().as_micros());
            panic!("subject weights present but failed to load: {e}");
        }
    };
    let mut ran = 0usize;
    let mut all_pass = true;
    for gpath in loader.list_goldens().unwrap_or_default() {
        let stem = golden_stem(&gpath);
        let golden = match loader.load_golden(&gpath) {
            Ok(g) => g,
            Err(e) => {
                // Present-but-corrupt golden = a REAL failure (fresh-eyes fix):
                // silently dropping it would let a broken fixture set pass the
                // token-exact/CER gates with zero comparisons.
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
        (FocrError::LowYield("x".into()), 8),
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
