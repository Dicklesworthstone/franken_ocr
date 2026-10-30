//! Per-model `FOCR_*` lever policy (U2-A7, plan D18).
//!
//! Unlimited-OCR-2 must never run a lever built for v1's R-SWA decode or for
//! the all-int8 decoder: that would produce confident garbage with exit 0.
//! Every `FOCR_*` variable this crate reads has one row in the model's table.
//! A row is refused before a weight byte is read (exit 2, lever named),
//! supported, ignored with an `ignored_lever` warning, not applicable (another
//! model's variable), or allow-listed tooling. A lever is refused only when
//! its value arms it, matching its reader's own parse; a kill-switch value is
//! a no-op and is ignored. The census test fails on a `FOCR_*` literal in
//! `src/` without a row, so a new lever cannot reach v2 silently.
//!
//! Models without a table (v1 and the zoo) are untouched.

use super::RunWarning;
use super::model_arch::{ModelArch, UnlimitedVariant};
use crate::error::{FocrError, FocrResult};

/// When a lever's value changes behaviour from its default, matching its
/// reader's parse. `NotFalsy` compares trimmed and case-insensitively; a
/// reader with a stricter parse is over-approximated, which only refuses more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Armed {
    /// Any value, `0` and the empty string included (`var_os(..).is_some()`).
    Present,
    /// `1`, `true`, `on` or `yes` ([`crate::quant::recipe::is_truthy`]).
    Truthy,
    /// Anything but empty, `0`, `off`, `false` or `no`.
    NotFalsy,
}

impl Armed {
    /// Whether `value` arms the lever.
    pub(crate) fn by(self, value: &str) -> bool {
        let value = value.trim().to_ascii_lowercase();
        match self {
            Self::Present => true,
            Self::Truthy => crate::quant::recipe::is_truthy(&value),
            Self::NotFalsy => {
                !value.is_empty() && !matches!(value.as_str(), "0" | "off" | "false" | "no")
            }
        }
    }
}

/// What a model does with one lever. Each carries the reason, which is also
/// the user-facing explanation where one is printed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Decision {
    /// Reaches a path the model must not run, or changes its numerics
    /// unverified: refused (exit 2) when armed, ignored otherwise.
    Refuse(Armed, &'static str),
    /// Reaches the model and is exact there, or carries no numerics.
    Supported(&'static str),
    /// Reaches no path of the model: a warning when set.
    Ignored(&'static str),
    /// Another model's variable, or a transport the model does not use.
    NotApplicable(&'static str),
    /// Test, oracle, calibration or conversion tooling, or a literal that
    /// is not a variable; never consulted by an inference run.
    AllowListed(&'static str),
}

/// One `FOCR_*` variable and its decision.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Lever {
    pub(crate) name: &'static str,
    pub(crate) decision: Decision,
}

/// A model's full lever table.
#[derive(Debug)]
pub(crate) struct LeverPolicy {
    pub(crate) model_id: &'static str,
    pub(crate) levers: &'static [Lever],
}

const fn lever(name: &'static str, decision: Decision) -> Lever {
    Lever { name, decision }
}

use Decision::{AllowListed, Ignored, NotApplicable, Refuse, Supported};

const R_SWA_ONLY: &str = "it reaches only v1's R-SWA attention";
const INT8_DECODER_ONLY: &str = "it reaches only the all-int8 decoder, which this model refuses";
const UNDER_SPINE: &str = "it takes effect only under FOCR_BATCH_SPINE, which this model refuses";
const RESIDENT: &str = "the resident daemon does not serve this model until bd-fuh2";

/// Unlimited-OCR-2's table, sorted by name.
pub(crate) static UOCR2_LEVER_POLICY: LeverPolicy = LeverPolicy {
    model_id: super::model_arch::UNLIMITED_OCR_V2_ID,
    levers: &[
        lever("FOCR_ATTN_GEMM", Ignored(R_SWA_ONLY)),
        lever("FOCR_BATCH_PACK", Ignored(UNDER_SPINE)),
        lever("FOCR_BATCH_SIZE", Ignored(UNDER_SPINE)),
        lever(
            "FOCR_BATCH_SPINE",
            Refuse(
                Armed::NotFalsy,
                "the batch spine runs v1's R-SWA int8 decode (until U2-F5)",
            ),
        ),
        lever("FOCR_BATCH_VISION", Ignored(UNDER_SPINE)),
        lever("FOCR_CALIB_OUT", AllowListed("calibration tooling")),
        lever("FOCR_CORPUS_DIR", AllowListed("model-gated corpus tests")),
        lever(
            "FOCR_DECODE_F32",
            Supported("the all-f32 cached decode is a v2 parity path (U2-C2)"),
        ),
        lever(
            "FOCR_DECODE_INT8",
            Refuse(
                Armed::Truthy,
                "the experimental all-int8 decoder is v1 R-SWA only",
            ),
        ),
        lever(
            "FOCR_DECODE_STATELESS",
            Supported("the stateless f32 decode is a v2 path (U2-B5)"),
        ),
        lever("FOCR_E2E_LONG", AllowListed("long model-gated tests")),
        lever(
            "FOCR_FORCE_ARCH",
            Supported("SIMD tier override; each tier matches its scalar fallback"),
        ),
        lever("FOCR_FUSE_NGRAM_LMHEAD", Ignored(INT8_DECODER_ONLY)),
        lever("FOCR_FUSE_NORM_QUANT", Ignored(INT8_DECODER_ONLY)),
        lever(
            "FOCR_FUSE_SWIGLU",
            Supported("the fused SwiGLU gives v2's decodes their unfused ids (U2-C2)"),
        ),
        lever("FOCR_GOT_FORMAT", NotApplicable("GOT-OCR2's")),
        lever("FOCR_GOT_INT8_LMHEAD", NotApplicable("GOT-OCR2's")),
        lever("FOCR_GOT_MODEL", AllowListed("GOT-OCR2 model-gated tests")),
        lever("FOCR_GOT_NO_REPEAT_NGRAM", NotApplicable("GOT-OCR2's")),
        lever("FOCR_GOT_SEQ_ATTN", NotApplicable("GOT-OCR2's")),
        lever("FOCR_GOT_TIKTOKEN", AllowListed("GOT-OCR2 tokenizer tests")),
        lever(
            "FOCR_INT4_GROUPS",
            Ignored("it is a `focr convert --quant int4` knob, and this model has no int4 recipe"),
        ),
        lever("FOCR_INT8_ATTN", Ignored(INT8_DECODER_ONLY)),
        lever(
            "FOCR_INT8_AUTOVEC",
            Supported("either int8 kernel tier gives v2's decode the same ids (U2-C2)"),
        ),
        lever("FOCR_INT8_KV", Ignored(R_SWA_ONLY)),
        lever("FOCR_INT8_LMHEAD", Ignored(INT8_DECODER_ONLY)),
        lever(
            "FOCR_LMHEAD_SHARD",
            Supported("the vocab-tiled lm_head gives v2's f32 and int8 decodes their ids (U2-C2)"),
        ),
        lever(
            "FOCR_LMHEAD_SHARD_TILES",
            Supported("the tile count only groups lm_head columns (U2-C2)"),
        ),
        lever(
            "FOCR_MANIFEST_URL",
            Supported("only where `focr pull` reads its manifest"),
        ),
        lever(
            "FOCR_MAX_NEW_TOKENS",
            Supported("a per-page generated-token cap for this model (U2-B6)"),
        ),
        lever(
            "FOCR_MMAP",
            Supported("maps the same bytes the owned load reads"),
        ),
        lever("FOCR_MODEL_DIR", Supported("model resolution")),
        lever("FOCR_MODEL_PATH", Supported("model resolution")),
        lever(
            "FOCR_MOE_POLICY_TEST_CASE",
            AllowListed("MoE policy child-process tests"),
        ),
        lever(
            "FOCR_MOE_SCORE_ORDER",
            Refuse(
                Armed::Truthy,
                "it routes experts in score order, not the reference's torch.topk slot \
                 order, which moves v2's int8 decode off its ids (U2-C2)",
            ),
        ),
        lever("FOCR_NO_PROGRESS", Supported("progress rendering only")),
        lever(
            "FOCR_NO_REPEAT_NGRAM",
            Supported("a per-page generated-only ban for this model (U2-B6)"),
        ),
        lever("FOCR_NO_RESIDENT", NotApplicable(RESIDENT)),
        lever(
            "FOCR_ONECHART_DIR",
            AllowListed("OneChart model-gated tests"),
        ),
        lever("FOCR_ORACLE_HIDDEN0", AllowListed("GOT-OCR2 oracle tests")),
        lever("FOCR_ORACLE_IMAGE", AllowListed("GOT-OCR2 oracle tests")),
        lever("FOCR_ORACLE_LOGITS", AllowListed("GOT-OCR2 oracle tests")),
        lever("FOCR_PARITY_IMAGES", AllowListed("parity-ladder tests")),
        lever(
            "FOCR_PREFILL_CHUNK",
            Refuse(
                Armed::Present,
                "chunked prefill assumes R-SWA attention (until U2-F2)",
            ),
        ),
        lever(
            "FOCR_PROFILE_DECODE",
            Supported("decode timing counters only"),
        ),
        lever("FOCR_QKV_FUSED", Ignored(INT8_DECODER_ONLY)),
        lever("FOCR_QUANT", Supported("model resolution")),
        lever(
            "FOCR_RESAMPLE",
            Ignored("this model's preprocessing fixes its own resampling (U2-B7)"),
        ),
        lever("FOCR_RESIDENT_CLIENT_TIMEOUT_SECS", NotApplicable(RESIDENT)),
        lever("FOCR_RESIDENT_DIR", NotApplicable(RESIDENT)),
        lever("FOCR_RESIDENT_IDLE_SECS", NotApplicable(RESIDENT)),
        lever("FOCR_RESIDENT_LOG", NotApplicable(RESIDENT)),
        lever("FOCR_RESIDENT_SPAWN_WAIT_SECS", NotApplicable(RESIDENT)),
        lever("FOCR_RSWA_PARALLEL_ATTN", Ignored(R_SWA_ONLY)),
        lever(
            "FOCR_RUNAWAY_GUARD",
            Ignored("the per-page cap and page guard replace it for this model (U2-B6)"),
        ),
        lever("FOCR_RUN_STORE", Supported("run telemetry only")),
        lever(
            "FOCR_SAM_ATTN_PROBE_RUNS",
            AllowListed("SAM attention probe tests"),
        ),
        lever("FOCR_SIGLIP_SEQ", NotApplicable("SmolVLM2's")),
        lever(
            "FOCR_SMOLVLM2_CERT_FULL",
            AllowListed("SmolVLM2 certification tests"),
        ),
        lever(
            "FOCR_SMOLVLM2_DIR",
            AllowListed("SmolVLM2 model-gated tests"),
        ),
        lever(
            "FOCR_SMOLVLM2_MODEL",
            AllowListed("SmolVLM2 model-gated tests"),
        ),
        lever(
            "FOCR_SMOLVLM2_ORACLE_HIDDEN0",
            AllowListed("SmolVLM2 oracle tests"),
        ),
        lever(
            "FOCR_SMOLVLM2_ORACLE_LOGITS",
            AllowListed("SmolVLM2 oracle tests"),
        ),
        lever("FOCR_SMOLVLM2_QUESTION", NotApplicable("SmolVLM2's")),
        lever(
            "FOCR_SMOLVLM2_TOKENIZER_JSON",
            AllowListed("SmolVLM2 tokenizer tests"),
        ),
        lever(
            "FOCR_SOFTMAX_COPY",
            Supported("the copying softmax gives v2's f32 and int8 decodes their ids (U2-C2)"),
        ),
        lever(
            "FOCR_SPEC_DECODE",
            Ignored("speculative decoding runs only inside the all-int8 decoder"),
        ),
        lever(
            "FOCR_SPEC_VERIFY",
            Ignored("speculative verify has no caller in any decode path"),
        ),
        lever(
            "FOCR_SRSWA_DENSE",
            Supported("the dense SR-SWA attention is bit-identical to the gathered one"),
        ),
        lever(
            "FOCR_SRSWA_ENTMAX",
            Supported(
                "the router's entmax domain: HF's top-M window (default) or SGLang's \
                 all candidates (D21, DISC-U2-1)",
            ),
        ),
        lever(
            "FOCR_SRSWA_RETIRE",
            Supported("SR-SWA block retirement frees only blocks no query can select"),
        ),
        lever(
            "FOCR_STAGE_BUDGET",
            AllowListed("not a variable: a music test asserts it is never named"),
        ),
        lever(
            "FOCR_STAGE_BUDGET_FORWARD_MS",
            Supported("the forward stage budget (U2-B6 scales it per page)"),
        ),
        lever(
            "FOCR_STREAM_VISION",
            Ignored("it streams only the GOT-OCR2, OneChart and SmolVLM2 vision towers"),
        ),
        lever("FOCR_TEST_FORCE_ERROR", AllowListed("CLI error-path tests")),
        lever("FOCR_TEST_LILYPOND_PDF", AllowListed("PDF text tests")),
        lever(
            "FOCR_TEST_LILYPOND_WINANSI_NAME_PDF",
            AllowListed("PDF text tests"),
        ),
        lever(
            "FOCR_TEST_MAX_POSITIONS",
            AllowListed("multi-page budget tests (U2-X2)"),
        ),
        lever(
            "FOCR_TEST_UNLIMITED_2_RECIPE_MODEL",
            AllowListed("census tests"),
        ),
        lever(
            "FOCR_TEST_UNLIMITED_2_SOURCE_DIR",
            AllowListed("census tests"),
        ),
        lever(
            "FOCR_TEST_UNLIMITED_RECIPE_MODEL",
            AllowListed("census tests"),
        ),
        lever(
            "FOCR_TEST_UNLIMITED_SOURCE_MODEL",
            AllowListed("census tests"),
        ),
        lever(
            "FOCR_THREADS",
            Supported("thread count; every kernel is thread-count exact"),
        ),
        lever("FOCR_TIMING", Supported("stage timing lines only")),
        lever("FOCR_TOKENIZER_JSON", AllowListed("tokenizer tests")),
        lever(
            "FOCR_TROMR",
            AllowListed("not a variable: a music test asserts it is never named"),
        ),
        lever("FOCR_TROMR_DIR", AllowListed("TrOMR model-gated tests")),
        lever("FOCR_TROMR_SAMPLE", NotApplicable("TrOMR's")),
        lever("FOCR_TROMR_SEED", NotApplicable("TrOMR's")),
        lever("FOCR_TROMR_SPLIT", NotApplicable("TrOMR's")),
        lever(
            "FOCR_UNLIMITED_RECIPE_ARTIFACT",
            AllowListed("conversion tests"),
        ),
        lever(
            "FOCR_UNLIMITED_VISION_CACHE",
            Supported("both arms hydrate the same BF16 vision tensors, byte-identical to v1's"),
        ),
        lever("FOCR_UOCR2_SOURCE_DIR", AllowListed("v2 model-gated tests")),
        lever(
            "FOCR_UOCR2_TOKENIZER_JSON",
            AllowListed("v2 tokenizer tests"),
        ),
    ],
};

/// The lever table of `arch`, if it has one. Exhaustive over the Unlimited
/// variants, so a new variant must decide here.
pub(crate) fn policy_for(arch: &dyn ModelArch) -> Option<&'static LeverPolicy> {
    match arch.unlimited_variant() {
        None | Some(UnlimitedVariant::V1Rswa) => None,
        Some(UnlimitedVariant::V2SrSwa) => Some(&UOCR2_LEVER_POLICY),
    }
}

/// Whether the resident daemon may serve `arch`. Its protocol carries no
/// per-model flags yet (bd-fuh2), so a model with a lever table runs inline.
pub(crate) fn resident_daemon_serves(arch: &dyn ModelArch) -> bool {
    policy_for(arch).is_none()
}

/// What the check reads: one environment lookup and the process-wide
/// full-int8 force flag (`ocr-batch --experimental-full-int8`).
pub(crate) struct LeverInputs<'a> {
    pub(crate) var: &'a dyn Fn(&str) -> Option<String>,
    pub(crate) full_int8_forced: bool,
}

/// The outcome of [`admit`]: the ignored-lever warnings, which are reported
/// whether or not the run is refused, and the verdict.
#[derive(Debug)]
pub(crate) struct Admission {
    pub(crate) warnings: Vec<RunWarning>,
    pub(crate) verdict: FocrResult<()>,
}

/// Admit `arch` for an inference run, before a weight byte is read.
///
/// Refuses with [`FocrError::Usage`] (exit 2) naming every armed refused
/// lever; then refuses a model that is registered but not implemented.
pub(crate) fn admit(arch: &dyn ModelArch, inputs: &LeverInputs<'_>) -> Admission {
    let Some(policy) = policy_for(arch) else {
        return Admission {
            warnings: Vec::new(),
            verdict: Ok(()),
        };
    };
    let model = policy.model_id;
    let mut refused = Vec::new();
    let mut warnings = Vec::new();
    if inputs.full_int8_forced {
        refused.push(format!(
            "--experimental-full-int8 is not supported for {model}: the all-int8 decoder it \
             forces is v1 R-SWA only"
        ));
    }
    for lever in policy.levers {
        let Some(value) = (inputs.var)(lever.name) else {
            continue;
        };
        match lever.decision {
            Refuse(armed, why) if armed.by(&value) => {
                refused.push(format!(
                    "{} is not supported for {model}: {why}",
                    lever.name
                ));
            }
            Refuse(..) => warnings.push(ignored_lever(
                lever.name,
                model,
                &format!("{value:?} leaves it off"),
            )),
            Ignored(why) => warnings.push(ignored_lever(lever.name, model, why)),
            Supported(_) | NotApplicable(_) | AllowListed(_) => {}
        }
    }
    let verdict = if !refused.is_empty() {
        // A flag is removed from the command line; variables are unset.
        let verb = if inputs.full_int8_forced {
            "remove"
        } else {
            "unset"
        };
        let object = if refused.len() == 1 { "it" } else { "them" };
        Err(FocrError::Usage(format!(
            "{}; {verb} {object}",
            refused.join("; ")
        )))
    } else if !arch.implemented() {
        Err(FocrError::Usage(format!(
            "{model} is registered but not yet implemented in this build; use unlimited-ocr \
             meanwhile"
        )))
    } else {
        Ok(())
    };
    Admission { warnings, verdict }
}

/// [`admit`] over the process environment and force flag.
pub(crate) fn admit_from_process(arch: &dyn ModelArch) -> Admission {
    admit(
        arch,
        &LeverInputs {
            var: &|name| std::env::var_os(name).map(|value| value.to_string_lossy().into_owned()),
            full_int8_forced: super::FORCE_INT8_DECODE.load(std::sync::atomic::Ordering::Relaxed),
        },
    )
}

/// The `ignored_lever` warning for `name`.
fn ignored_lever(name: &str, model: &str, why: &str) -> RunWarning {
    let mut fields = serde_json::Map::new();
    fields.insert("env".into(), name.into());
    RunWarning {
        kind: "ignored_lever",
        message: format!("{name} has no effect on {model}: {why}"),
        recovery: super::warning_recovery(super::WarningKind::IgnoredLever),
        fields,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::native_engine::model_arch;

    fn v2() -> &'static dyn ModelArch {
        model_arch::arch_by_id(model_arch::UNLIMITED_OCR_V2_ID).expect("v2 registered")
    }

    fn line(test: &str, case: &str, event: &str, pass: bool, extra: serde_json::Value) {
        let mut object = serde_json::json!({
            "schema_version": 1, "ts": 0.0, "test": test, "case": case, "run_seq": 0,
            "event": event, "result": if pass { "pass" } else { "fail" },
        });
        if let (Some(object), Some(extra)) = (object.as_object_mut(), extra.as_object()) {
            object.extend(extra.clone());
        }
        println!("{object}");
    }

    /// Admit `arch` with exactly `env` set and the force flag as given.
    fn admit_with(arch: &dyn ModelArch, env: &[(&str, &str)], forced: bool) -> Admission {
        let env: BTreeMap<String, String> = env
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect();
        admit(
            arch,
            &LeverInputs {
                var: &|name| env.get(name).cloned(),
                full_int8_forced: forced,
            },
        )
    }

    /// A value that arms `decision` (or merely sets it).
    fn armed_sample(decision: Decision) -> &'static str {
        match decision {
            Refuse(Armed::Present, _) => "0",
            Refuse(Armed::Truthy | Armed::NotFalsy, _) => "1",
            Supported(_) | Ignored(_) | NotApplicable(_) | AllowListed(_) => "1",
        }
    }

    /// Every `.rs` file under `dir` with its text.
    fn rust_sources(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
        for entry in std::fs::read_dir(dir).expect("read src dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                rust_sources(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let text = std::fs::read_to_string(&path).expect("read source file");
                out.push((path, text));
            }
        }
    }

    /// The `[A-Z0-9_]+` runs that follow `prefix` and end at a `"`.
    fn quoted_after<'a>(text: &'a str, prefix: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        text.match_indices(prefix).filter_map(move |(at, _)| {
            let rest = &text[at + prefix.len()..];
            let end = rest
                .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
                .unwrap_or(rest.len());
            (end > 0 && rest[end..].starts_with('"')).then(|| &rest[..end])
        })
    }

    /// The census (U2-A7): every `FOCR_*` literal in `src/`, and every
    /// dynamic `FOCR_STAGE_BUDGET_{stage}_MS`, has exactly one row; every row
    /// names a variable read outside this table.
    #[test]
    fn every_focr_variable_in_src_has_exactly_one_uocr2_row() {
        let test = "every_focr_variable_in_src_has_exactly_one_uocr2_row";
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut sources = Vec::new();
        rust_sources(&src, &mut sources);
        let mut literals: BTreeMap<String, BTreeSet<PathBuf>> = BTreeMap::new();
        // The stage budgets are named through `format!`, so the literal scan
        // cannot see them; their stage names can.
        let mut dynamic = BTreeSet::new();
        for (path, text) in &sources {
            for name in quoted_after(text, "\"FOCR_") {
                literals
                    .entry(format!("FOCR_{name}"))
                    .or_default()
                    .insert(path.clone());
            }
            for stage in quoted_after(text, "stage_budget(\"") {
                dynamic.insert(format!("FOCR_STAGE_BUDGET_{stage}_MS"));
            }
        }
        assert!(
            dynamic.contains("FOCR_STAGE_BUDGET_FORWARD_MS"),
            "the scan finds the forward budget"
        );

        let names: Vec<&str> = UOCR2_LEVER_POLICY
            .levers
            .iter()
            .map(|lever| lever.name)
            .collect();
        assert!(
            names.windows(2).all(|pair| pair[0] < pair[1]),
            "rows are sorted and unique"
        );
        let rows: BTreeSet<&str> = names.iter().copied().collect();
        let missing: Vec<&String> = literals
            .keys()
            .chain(dynamic.iter())
            .filter(|name| !rows.contains(name.as_str()))
            .collect();
        let this_file = src.join("native_engine").join("lever_policy.rs");
        let stale: Vec<&str> = names
            .iter()
            .copied()
            .filter(|name| {
                let named_elsewhere = literals
                    .get(*name)
                    .is_some_and(|files| files.iter().any(|file| *file != this_file));
                !named_elsewhere && !dynamic.contains(*name)
            })
            .collect();
        let pass = missing.is_empty() && stale.is_empty();
        line(
            test,
            "census",
            "assert",
            pass,
            serde_json::json!({
                "assertion": "rows == FOCR_* literals in src + dynamic stage budgets",
                "pass": pass, "rows": rows.len(), "literals": literals.len(),
                "dynamic": dynamic, "missing": missing, "stale": stale,
            }),
        );
        assert!(
            missing.is_empty(),
            "variables without a UOCR2_LEVER_POLICY row: {missing:?}"
        );
        assert!(
            stale.is_empty(),
            "rows naming no variable read in src: {stale:?}"
        );
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Expect {
        /// Exit 2 naming the lever.
        Refused,
        /// Admitted, with one `ignored_lever` warning.
        Ignored,
        /// Admitted, no warning.
        Silent,
    }

    /// The bead's path/lever matrix, written independently of the table.
    const MATRIX: &[(&str, &str, Expect)] = &[
        ("FOCR_DECODE_INT8", "1", Expect::Refused),
        ("FOCR_DECODE_INT8", "0", Expect::Ignored),
        ("FOCR_PREFILL_CHUNK", "512", Expect::Refused),
        ("FOCR_PREFILL_CHUNK", "0", Expect::Refused),
        ("FOCR_LMHEAD_SHARD", "0", Expect::Silent),
        ("FOCR_BATCH_SPINE", "1", Expect::Refused),
        ("FOCR_BATCH_SPINE", "0", Expect::Ignored),
        ("FOCR_BATCH_SPINE", "", Expect::Ignored),
        ("FOCR_BATCH_SPINE", " OFF ", Expect::Ignored),
        ("FOCR_MOE_SCORE_ORDER", "1", Expect::Refused),
        ("FOCR_MOE_SCORE_ORDER", "0", Expect::Ignored),
        ("FOCR_FUSE_SWIGLU", "0", Expect::Silent),
        ("FOCR_SOFTMAX_COPY", "yes", Expect::Silent),
        ("FOCR_INT8_AUTOVEC", "0", Expect::Silent),
        ("FOCR_SPEC_DECODE", "1", Expect::Ignored),
        ("FOCR_SPEC_VERIFY", "1", Expect::Ignored),
        ("FOCR_INT8_ATTN", "1", Expect::Ignored),
        ("FOCR_INT8_LMHEAD", "1", Expect::Ignored),
        ("FOCR_INT8_KV", "1", Expect::Ignored),
        ("FOCR_ATTN_GEMM", "1", Expect::Ignored),
        ("FOCR_RSWA_PARALLEL_ATTN", "0", Expect::Ignored),
        ("FOCR_QKV_FUSED", "0", Expect::Ignored),
        ("FOCR_QKV_FUSED", "1", Expect::Ignored),
        ("FOCR_FUSE_NGRAM_LMHEAD", "1", Expect::Ignored),
        ("FOCR_FUSE_NORM_QUANT", "1", Expect::Ignored),
        ("FOCR_RUNAWAY_GUARD", "1", Expect::Ignored),
        ("FOCR_RUNAWAY_GUARD", "maybe", Expect::Ignored),
        ("FOCR_RESAMPLE", "pil-bicubic", Expect::Ignored),
        ("FOCR_LMHEAD_SHARD_TILES", "4", Expect::Silent),
        ("FOCR_BATCH_PACK", "1", Expect::Ignored),
        ("FOCR_BATCH_VISION", "0", Expect::Ignored),
        ("FOCR_BATCH_SIZE", "4", Expect::Ignored),
        ("FOCR_STREAM_VISION", "1", Expect::Ignored),
        ("FOCR_INT4_GROUPS", "32", Expect::Ignored),
        ("FOCR_MAX_NEW_TOKENS", "100", Expect::Silent),
        ("FOCR_NO_REPEAT_NGRAM", "20", Expect::Silent),
        ("FOCR_THREADS", "4", Expect::Silent),
        ("FOCR_FORCE_ARCH", "scalar", Expect::Silent),
        ("FOCR_PROFILE_DECODE", "0", Expect::Silent),
        ("FOCR_QUANT", "int8", Expect::Silent),
        ("FOCR_DECODE_STATELESS", "1", Expect::Silent),
        ("FOCR_DECODE_F32", "1", Expect::Silent),
        ("FOCR_STAGE_BUDGET_FORWARD_MS", "0", Expect::Silent),
        ("FOCR_UNLIMITED_VISION_CACHE", "0", Expect::Silent),
        ("FOCR_MMAP", "1", Expect::Silent),
        ("FOCR_SRSWA_RETIRE", "0", Expect::Silent),
        ("FOCR_SRSWA_DENSE", "1", Expect::Silent),
        ("FOCR_SRSWA_ENTMAX", "all", Expect::Silent),
        ("FOCR_CALIB_OUT", "/tmp/calib", Expect::Silent),
        ("FOCR_SAM_ATTN_PROBE_RUNS", "3", Expect::Silent),
        ("FOCR_TEST_MAX_POSITIONS", "100", Expect::Silent),
        ("FOCR_NO_RESIDENT", "1", Expect::Silent),
        ("FOCR_RESIDENT_DIR", "/tmp", Expect::Silent),
        ("FOCR_GOT_FORMAT", "1", Expect::Silent),
    ];

    fn observed(name: &str, admission: &Admission) -> Expect {
        let warned: Vec<&RunWarning> = admission
            .warnings
            .iter()
            .filter(|warning| warning.fields.get("env").and_then(|v| v.as_str()) == Some(name))
            .collect();
        match &admission.verdict {
            Err(error)
                if error
                    .to_string()
                    .contains(&format!("{name} is not supported for unlimited-ocr-2"))
                    && warned.is_empty() =>
            {
                Expect::Refused
            }
            Ok(()) if warned.len() == 1 && admission.warnings.len() == 1 => {
                assert_eq!(warned[0].kind, "ignored_lever");
                assert!(warned[0].message.contains(name), "{}", warned[0].message);
                Expect::Ignored
            }
            Ok(()) if admission.warnings.is_empty() => Expect::Silent,
            _ => panic!("{name}: unclassifiable admission {admission:?}"),
        }
    }

    /// The matrix (U2-A7): each lever alone, against v2 (Experimental since
    /// U2-B6, so only a refused lever stops the run).
    #[test]
    fn v2_lever_matrix_refuses_ignores_or_admits_each_lever_before_load() {
        let test = "v2_lever_matrix_refuses_ignores_or_admits_each_lever_before_load";
        for &(name, value, expect) in MATRIX {
            let admission = admit_with(v2(), &[(name, value)], false);
            let got = observed(name, &admission);
            let error = admission.verdict.as_ref().err();
            line(
                test,
                &format!("{name}={value}"),
                "assert",
                got == expect,
                serde_json::json!({
                    "assertion": "admission decision matches the bead matrix",
                    "pass": got == expect, "arch": "unlimited-ocr-2", "lever": name,
                    "value": value, "expected": format!("{expect:?}"), "got": format!("{got:?}"),
                    "exit_code": error.map(FocrError::exit_code),
                    "message": error.map(ToString::to_string),
                    "warnings": admission.warnings.len(), "weights_touched": false,
                }),
            );
            assert_eq!(got, expect, "{name}={value:?}: {admission:?}");
            if let Some(error) = error {
                assert!(matches!(error, FocrError::Usage(_)), "{error:?}");
                assert_eq!(error.exit_code(), 2);
            }
        }
    }

    /// Every row, armed, against the table's own decision; nothing panics
    /// and no row escapes the three outcomes.
    #[test]
    fn every_row_armed_alone_lands_on_its_decision() {
        for lever in UOCR2_LEVER_POLICY.levers {
            let admission = admit_with(v2(), &[(lever.name, armed_sample(lever.decision))], false);
            let expect = match lever.decision {
                Refuse(..) => Expect::Refused,
                Ignored(_) => Expect::Ignored,
                Supported(_) | NotApplicable(_) | AllowListed(_) => Expect::Silent,
            };
            assert_eq!(observed(lever.name, &admission), expect, "{}", lever.name);
        }
    }

    /// v1 and every zoo model: every lever set at once, and the force flag,
    /// change nothing (they have no table).
    #[test]
    fn v1_and_zoo_admission_is_untouched_by_every_lever() {
        let test = "v1_and_zoo_admission_is_untouched_by_every_lever";
        let env: Vec<(&str, &str)> = UOCR2_LEVER_POLICY
            .levers
            .iter()
            .map(|lever| (lever.name, armed_sample(lever.decision)))
            .collect();
        for arch in model_arch::registry() {
            if arch.id() == model_arch::UNLIMITED_OCR_V2_ID {
                continue;
            }
            for forced in [false, true] {
                let admission = admit_with(*arch, &env, forced);
                let pass = admission.verdict.is_ok() && admission.warnings.is_empty();
                line(
                    test,
                    &format!("{}:forced={forced}", arch.id()),
                    "assert",
                    pass,
                    serde_json::json!({
                        "assertion": "no lever table, admission unchanged", "pass": pass,
                        "arch": arch.id(), "levers_set": env.len(), "weights_touched": false,
                    }),
                );
                assert!(pass, "{}: {admission:?}", arch.id());
            }
            assert!(resident_daemon_serves(*arch), "{}", arch.id());
        }
        assert!(!resident_daemon_serves(v2()));
    }

    #[test]
    fn the_full_int8_force_flag_and_several_levers_are_all_named() {
        let forced = admit_with(v2(), &[], true);
        let message = forced
            .verdict
            .expect_err("forced full-int8 refused")
            .to_string();
        assert!(
            message.contains("--experimental-full-int8 is not supported for unlimited-ocr-2"),
            "{message}"
        );
        assert!(message.ends_with("; remove it"), "{message}");

        let one = admit_with(v2(), &[("FOCR_PREFILL_CHUNK", "64")], false);
        let message = one.verdict.expect_err("refused").to_string();
        assert!(message.ends_with("; unset it"), "{message}");

        let several = admit_with(
            v2(),
            &[
                ("FOCR_PREFILL_CHUNK", "64"),
                ("FOCR_BATCH_SPINE", "1"),
                ("FOCR_QKV_FUSED", "0"),
            ],
            true,
        );
        let message = several.verdict.expect_err("refused").to_string();
        for name in [
            "--experimental-full-int8",
            "FOCR_PREFILL_CHUNK",
            "FOCR_BATCH_SPINE",
        ] {
            assert!(message.contains(name), "{message}");
        }
        assert!(!message.contains("FOCR_QKV_FUSED"), "{message}");
        assert!(message.ends_with("; remove them"), "{message}");
        assert_eq!(several.warnings.len(), 1, "the ignored lever still warns");
    }

    /// v2 (Experimental since U2-B6) is admitted in an unarmed environment.
    /// A policed model that is still Planned is refused after its levers are
    /// judged: an armed refused lever is named, and ignored levers still warn.
    #[test]
    fn a_planned_policed_model_is_refused_after_its_levers() {
        let real = admit_with(v2(), &[("FOCR_THREADS", "4")], false);
        assert!(real.verdict.is_ok() && real.warnings.is_empty(), "{real:?}");

        struct PlannedV2;
        impl ModelArch for PlannedV2 {
            fn id(&self) -> &'static str {
                v2().id()
            }
            fn display_name(&self) -> &'static str {
                v2().display_name()
            }
            fn license_notice(&self) -> &'static str {
                v2().license_notice()
            }
            fn default_artifact_basename(&self) -> &'static str {
                v2().default_artifact_basename()
            }
            fn vision_encoder(&self) -> model_arch::VisionEncoder {
                v2().vision_encoder()
            }
            fn decoder(&self) -> model_arch::Decoder {
                v2().decoder()
            }
            fn tokenizer(&self) -> model_arch::TokenizerKind {
                v2().tokenizer()
            }
            fn decode_contract(&self) -> model_arch::DecodeContract {
                v2().decode_contract()
            }
            fn tasks(&self) -> &'static [model_arch::Task] {
                v2().tasks()
            }
            fn maturity(&self) -> model_arch::Maturity {
                model_arch::Maturity::Planned
            }
            fn required_quant_recipe(&self, quant: &str) -> Option<&'static str> {
                v2().required_quant_recipe(quant)
            }
            fn unlimited_variant(&self) -> Option<UnlimitedVariant> {
                v2().unlimited_variant()
            }
        }
        const NOT_IMPLEMENTED: &str =
            "unlimited-ocr-2 is registered but not yet implemented in this build";
        let clean = admit_with(&PlannedV2, &[("FOCR_THREADS", "4")], false);
        assert!(
            matches!(clean.verdict, Err(FocrError::Usage(ref m)) if m.contains(NOT_IMPLEMENTED)),
            "{clean:?}"
        );
        assert!(clean.warnings.is_empty(), "{clean:?}");
        let ignored = admit_with(&PlannedV2, &[("FOCR_QKV_FUSED", "0")], false);
        assert!(
            matches!(ignored.verdict, Err(FocrError::Usage(ref m)) if m.contains(NOT_IMPLEMENTED)),
            "{ignored:?}"
        );
        assert_eq!(ignored.warnings.len(), 1);
        let json = ignored.warnings[0].to_json();
        assert_eq!(json["kind"], "ignored_lever");
        assert_eq!(json["env"], "FOCR_QKV_FUSED");
        assert!(super::super::RUN_WARNING_KINDS.contains(&"ignored_lever"));
        let refused = admit_with(&PlannedV2, &[("FOCR_PREFILL_CHUNK", "")], false);
        assert!(
            matches!(refused.verdict, Err(FocrError::Usage(ref m))
                if m.contains("FOCR_PREFILL_CHUNK") && !m.contains(NOT_IMPLEMENTED)),
            "{refused:?}"
        );
    }

    #[test]
    fn armed_matches_each_reader_parse() {
        let cases = [
            (Armed::Present, "", true),
            (Armed::Present, "0", true),
            (Armed::Truthy, " YES ", true),
            (Armed::Truthy, "2", false),
            (Armed::Truthy, "off", false),
            (Armed::NotFalsy, "", false),
            (Armed::NotFalsy, " Off ", false),
            (Armed::NotFalsy, "anything", true),
        ];
        for (armed, value, expected) in cases {
            assert_eq!(armed.by(value), expected, "{armed:?} by {value:?}");
        }
    }
}
