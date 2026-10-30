#![allow(clippy::doc_overindented_list_items, clippy::doc_lazy_continuation)]
//! `uocr2_goldens` — the token-id golden gate under the Unlimited-OCR-2 epic
//! (bead U2-T3, `bd-uocr2-epic-cgux.1`; `docs/planning/UNLIMITED_OCR_2_PLAN.md`
//! §7.6 and §3.2 principle 1).
//!
//! Unlimited-OCR-2 shares nearly all of the engine with v1: the vision tower,
//! MoE, loader, tokenizer, sampler, multi-page page source and robot output.
//! Every shared refactor can perturb v1 silently, and one drifted token can
//! survive detokenize + postprocess without changing the markdown. So this gate
//! pins v1's real outputs at the **token-id** level, plus the final markdown and
//! the streaming robot `page` events, and fails on any change.
//!
//! **Lanes.** A lane is one model artifact, pinned by SHA-256, plus one golden
//! directory: v1 (`unlimited-ocr`, `tests/fixtures/uocr2_v1_goldens/`) and,
//! from U2-C4, v2 (`unlimited-ocr-2`, `tests/fixtures/uocr2_v2_goldens/`) on
//! the int8 artifact U2-C4 certified. Every later change on the v2 path keeps
//! the v2 lane green.
//!
//! **Cases.** v1's fast subset — `page_0009`, `page_0014`, the 2-page `p9_p14`
//! document and `p9_p14` streamed — runs whenever the gate is armed.
//! `FOCR_E2E_LONG=1` adds `page_0590` (~4k tokens, the runaway-prone table
//! page), `p10` and `p20`. v2's cases are all fast: `page_0009`, `page_0014`,
//! the 3-page S2 document (`s2`: pages 9, 14 and 19) and `s2` streamed. Inputs
//! come from `FOCR_CORPUS_DIR` (default `~/.cache/franken_ocr/corpus/pages`);
//! v1's multi-page page lists come from
//! `tests/fixtures/multi_page/{name}_meta.json`.
//!
//! **Gating** (LOGGING_AND_E2E.md §4):
//! * Always on: loading `/nonexistent` is a clean `ModelNotFound` (the native
//!   path has no silent fallback), the goldens on disk are well-formed, and the
//!   comparators fail the way they should.
//! * Armed by `FOCR_GOLDEN_GATE=1`. `scripts/token_golden_gate.sh` sets it and
//!   builds the optimized profile, because a debug-profile decode takes minutes
//!   per page. Unarmed, or armed without the pinned artifact or corpus, a lane
//!   logs `skip_no_model` with its reason and passes.
//! * Armed with the pinned artifact: every case re-runs and must match its
//!   golden byte for byte. Robot events compare on a canonical projection:
//!   only the golden's own fields, minus the shared volatile list
//!   (`parity_harness::is_scrub_key`), so additive robot fields never break it.
//!
//! **Commands** (GOLDEN.md §4a):
//! ```text
//! scripts/token_golden_gate.sh                     # fast subset (check.sh runs this)
//! FOCR_E2E_LONG=1 scripts/token_golden_gate.sh     # full set
//! UPDATE_GOLDENS=1 FOCR_E2E_LONG=1 scripts/token_golden_gate.sh   # reviewed recapture only
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[path = "support/parity_harness.rs"]
mod parity_harness;

use franken_ocr::FocrError;
use franken_ocr::native_engine::OcrModel;
use parity_harness::{
    Logger, NATIVE_PATH_FALLBACK, actual_sidecar, file_sha256, golden_diff, is_scrub_key,
    sha256_hex, update_goldens,
};
use serde_json::{Map, Value, json};

const TEST: &str = "uocr2_goldens";
/// Arms the model-backed lanes. A resolvable model alone must not: a
/// debug-profile decode is minutes per page (the e2e_recognize precedent).
const ARM_ENV: &str = "FOCR_GOLDEN_GATE";
/// Adds the long cases (plan §7.6: required before closing any bead that
/// touches the decoder, loader, tokenizer, sampler, PDF walk or CLI output).
const LONG_ENV: &str = "FOCR_E2E_LONG";
const CAPTURE_FILE: &str = "CAPTURE.json";
const IDS_PER_LINE: usize = 16;

// ───────────────────────────────────────────────────────────────────────────
// Lanes and cases.
// ───────────────────────────────────────────────────────────────────────────

struct Lane {
    model_id: &'static str,
    golden_dir: &'static str,
    /// Filenames the lane looks for in each model search dir (after
    /// `FOCR_MODEL_PATH`); the capture's SHA-256 pin picks among them.
    artifact_names: &'static [&'static str],
    cases: &'static [Case],
}

struct Case {
    name: &'static str,
    kind: Kind,
    long: bool,
}

enum Kind {
    /// One image, single-page base mode (`OcrModel::recognize`).
    Single(&'static str),
    /// One cross-page pass over a document's pages.
    Multi(Doc),
    /// The same pass through the CLI's driver — the pages read lazily from a
    /// `PageSource` (U2-D0), streaming robot `page` events. Streaming is
    /// observational, so its ids and markdown must equal the named document
    /// case's goldens (never rewritten from this case).
    Stream(Doc),
}

/// A document's pages.
#[derive(Clone, Copy)]
enum Doc {
    /// A `tests/fixtures/multi_page` fixture's page list; the fixture names
    /// its case.
    Fixture(&'static str),
    /// The named case's pages, listed here.
    Pages(&'static str, &'static [&'static str]),
}

impl Doc {
    /// The document case's name.
    fn name(self) -> &'static str {
        match self {
            Self::Fixture(name) | Self::Pages(name, _) => name,
        }
    }
}

const V1_LANE: Lane = Lane {
    model_id: "unlimited-ocr",
    golden_dir: "tests/fixtures/uocr2_v1_goldens",
    artifact_names: &[
        "unlimited-ocr.int8.focrq",
        "unlimited-ocr.v0.7.0.int8.focrq",
    ],
    cases: &[
        Case {
            name: "page_0009",
            kind: Kind::Single("page_0009.png"),
            long: false,
        },
        Case {
            name: "page_0014",
            kind: Kind::Single("page_0014.png"),
            long: false,
        },
        Case {
            name: "p9_p14",
            kind: Kind::Multi(Doc::Fixture("p9_p14")),
            long: false,
        },
        Case {
            name: "p9_p14_stream",
            kind: Kind::Stream(Doc::Fixture("p9_p14")),
            long: false,
        },
        Case {
            name: "page_0590",
            kind: Kind::Single("page_0590.png"),
            long: true,
        },
        Case {
            name: "p10",
            kind: Kind::Multi(Doc::Fixture("p10")),
            long: true,
        },
        Case {
            name: "p20",
            kind: Kind::Multi(Doc::Fixture("p20")),
            long: true,
        },
    ],
};

/// U2-A6's S2 document: three corpus pages read in one pass.
const S2: Doc = Doc::Pages("s2", &["page_0009.png", "page_0014.png", "page_0019.png"]);

/// The v2 lane (U2-C4): the regression gate for every later change on the v2
/// path. Its artifact is the conservative int8 build U2-C4 certified.
const V2_LANE: Lane = Lane {
    model_id: "unlimited-ocr-2",
    golden_dir: "tests/fixtures/uocr2_v2_goldens",
    artifact_names: &["unlimited-ocr-2.v0.1.0.int8.focrq"],
    cases: &[
        Case {
            name: "page_0009",
            kind: Kind::Single("page_0009.png"),
            long: false,
        },
        Case {
            name: "page_0014",
            kind: Kind::Single("page_0014.png"),
            long: false,
        },
        Case {
            name: "s2",
            kind: Kind::Multi(S2),
            long: false,
        },
        Case {
            name: "s2_stream",
            kind: Kind::Stream(S2),
            long: false,
        },
    ],
};

/// Every lane, in the order the always-on checks walk them.
const LANES: [&Lane; 2] = [&V1_LANE, &V2_LANE];

impl Case {
    /// The input image names, in order.
    fn inputs(&self) -> Result<Vec<String>, String> {
        match self.kind {
            Kind::Single(image) => Ok(vec![image.to_string()]),
            Kind::Multi(Doc::Fixture(fixture)) | Kind::Stream(Doc::Fixture(fixture)) => {
                multi_page_names(fixture)
            }
            Kind::Multi(Doc::Pages(_, pages)) | Kind::Stream(Doc::Pages(_, pages)) => {
                Ok(pages.iter().map(|p| (*p).to_string()).collect())
            }
        }
    }

    /// The case whose ids/markdown goldens this case is compared against.
    fn golden_stem(&self) -> &'static str {
        match self.kind {
            Kind::Stream(doc) => doc.name(),
            _ => self.name,
        }
    }
}

fn repo_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn multi_page_names(fixture: &str) -> Result<Vec<String>, String> {
    let path = repo_path(&format!("tests/fixtures/multi_page/{fixture}_meta.json"));
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let meta: Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    meta["pages"]
        .as_array()
        .and_then(|pages| {
            pages
                .iter()
                .map(|p| p.as_str().map(str::to_string))
                .collect()
        })
        .ok_or_else(|| format!("{}: `pages` is not an array of strings", path.display()))
}

fn env_flag(name: &str) -> bool {
    matches!(std::env::var(name).ok().as_deref(), Some(v) if !v.is_empty() && v != "0")
}

/// Every `FOCR_*` variable in force, for the setup line and failure hints:
/// a decode-affecting override is the first suspect when a golden moves.
fn focr_env() -> BTreeMap<String, String> {
    std::env::vars()
        .filter(|(k, _)| k.starts_with("FOCR_"))
        .collect()
}

fn corpus_dir() -> Option<PathBuf> {
    match std::env::var_os("FOCR_CORPUS_DIR") {
        Some(dir) if !dir.is_empty() => Some(PathBuf::from(dir)),
        _ => std::env::var_os("HOME")
            .map(|home| PathBuf::from(home).join(".cache/franken_ocr/corpus/pages")),
    }
}

// ───────────────────────────────────────────────────────────────────────────
// The capture manifest: what each golden was captured on.
// ───────────────────────────────────────────────────────────────────────────

/// `CAPTURE.json`: the lane's artifact pin plus, per case, the inputs'
/// SHA-256, the token count, and the artifact, binary version, git revision
/// and thread count it was captured with.
fn load_capture(lane: &Lane) -> Result<Option<Value>, String> {
    let path = repo_path(lane.golden_dir).join(CAPTURE_FILE);
    match fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text)
            .map(Some)
            .map_err(|e| format!("{}: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

struct Artifact {
    path: PathBuf,
    size: u64,
    sha256: String,
}

/// Resolve the lane's artifact. Compare mode takes only a candidate matching
/// the capture's size + SHA-256 pin, so a later default change can never
/// redirect the gate. Update mode takes the first candidate that exists
/// (`FOCR_MODEL_PATH` first), and the recapture re-pins it.
fn resolve_artifact(lane: &Lane, pin: Option<&Value>, update: bool) -> Result<Artifact, String> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(p) = std::env::var_os(franken_ocr::MODEL_PATH_ENV).filter(|p| !p.is_empty()) {
        candidates.push(PathBuf::from(p));
    }
    for dir in franken_ocr::native_engine::model_resolution_search_dirs() {
        for name in lane.artifact_names {
            candidates.push(dir.join(name));
        }
    }
    let pinned = pin.and_then(|p| Some((p["size"].as_u64()?, p["sha256"].as_str()?.to_string())));
    let mut rejected = Vec::new();
    for path in &candidates {
        let Ok(meta) = fs::metadata(path) else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        let size = meta.len();
        if !update {
            let Some((pin_size, pin_sha)) = &pinned else {
                return Err(format!(
                    "{} has no artifact pin (capture with UPDATE_GOLDENS=1 first)",
                    lane.golden_dir
                ));
            };
            if size != *pin_size {
                rejected.push(format!(
                    "{} (size {size} != pinned {pin_size})",
                    path.display()
                ));
                continue;
            }
            let sha256 = file_sha256(path)?;
            if sha256 != *pin_sha {
                rejected.push(format!(
                    "{} (sha256 {sha256} != pinned {pin_sha})",
                    path.display()
                ));
                continue;
            }
            return Ok(Artifact {
                path: path.clone(),
                size,
                sha256,
            });
        }
        let sha256 = file_sha256(path)?;
        return Ok(Artifact {
            path: path.clone(),
            size,
            sha256,
        });
    }
    Err(format!(
        "no pinned {} artifact among {} candidates{}",
        lane.model_id,
        candidates.len(),
        if rejected.is_empty() {
            String::new()
        } else {
            format!("; rejected: {}", rejected.join(", "))
        }
    ))
}

// ───────────────────────────────────────────────────────────────────────────
// Running a case.
// ───────────────────────────────────────────────────────────────────────────

struct Observed {
    ids: Vec<u32>,
    markdown: String,
    events: Option<Vec<Value>>,
}

fn run_case(model: &OcrModel, case: &Case, pages: &[PathBuf]) -> Result<Observed, String> {
    let _ = model.take_last_decode();
    let mut events = None;
    let failed = |e: franken_ocr::FocrError| format!("{} (exit {}): {e}", e.kind(), e.exit_code());
    let markdown = match case.kind {
        Kind::Single(_) => model.recognize(&pages[0]).map_err(failed)?,
        Kind::Multi(_) | Kind::Stream(_) => {
            let mut source = franken_ocr::page_source::PageSource::images(pages.to_vec())
                .map_err(|e| format!("open the page source: {e}"))?;
            let mut seen = Vec::new();
            let mut sink = |page: &franken_ocr::StreamedPage<'_>| {
                seen.push(franken_ocr::robot::page_decoded_event(
                    page.index, page.body,
                ));
            };
            let streaming = matches!(case.kind, Kind::Stream(_));
            let document = model
                .recognize_document(
                    &mut source,
                    &franken_ocr::DocumentOptions::default(),
                    &mut |_| {},
                    streaming
                        .then_some(&mut sink as &mut dyn FnMut(&franken_ocr::StreamedPage<'_>)),
                )
                .map_err(failed)?;
            if !document.skipped.is_empty() {
                return Err(format!(
                    "corpus pages failed to decode: {:?}",
                    document.skipped
                ));
            }
            if streaming {
                events = Some(seen);
            }
            document.markdown
        }
    };
    let ids = model
        .take_last_decode()
        .map(|report| report.ids)
        .ok_or("the decode recorded no generated ids")?;
    Ok(Observed {
        ids,
        markdown,
        events,
    })
}

// ───────────────────────────────────────────────────────────────────────────
// Golden formats and comparators.
// ───────────────────────────────────────────────────────────────────────────

fn ids_text(model_id: &str, case: &str, ids: &[u32]) -> String {
    let mut out = format!(
        "# {model_id} generated token ids: {case} ({} ids)\n",
        ids.len()
    );
    for chunk in ids.chunks(IDS_PER_LINE) {
        let line: Vec<String> = chunk.iter().map(u32::to_string).collect();
        out.push_str(&line.join(" "));
        out.push('\n');
    }
    out
}

fn parse_ids(text: &str) -> Result<Vec<u32>, String> {
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .flat_map(str::split_whitespace)
        .map(|t| {
            t.parse::<u32>()
                .map_err(|e| format!("bad token id {t:?}: {e}"))
        })
        .collect()
}

/// The first divergence between two token streams, readable: the index, both
/// ids, and a little context on each side. `None` when they are equal.
fn first_id_diff(golden: &[u32], actual: &[u32]) -> Option<String> {
    let common = golden
        .iter()
        .zip(actual)
        .take_while(|(g, a)| g == a)
        .count();
    if common == golden.len() && common == actual.len() {
        return None;
    }
    let ctx = |ids: &[u32]| -> String {
        let lo = common.saturating_sub(3);
        let hi = (common + 4).min(ids.len());
        format!("{:?} (from index {lo})", &ids[lo..hi])
    };
    let at = |ids: &[u32]| {
        ids.get(common)
            .map_or_else(|| "<end>".to_string(), u32::to_string)
    };
    Some(format!(
        "first differing token at index {common} (golden {} ids, actual {} ids): \
         golden id {} vs actual id {}\n  golden {}\n  actual {}",
        golden.len(),
        actual.len(),
        at(golden),
        at(actual),
        ctx(golden),
        ctx(actual),
    ))
}

/// Compare the ids against their golden, writing through `golden_diff` (so
/// `UPDATE_GOLDENS=1` and the `.actual` sidecar behave as everywhere else),
/// prefixed with the token-level diagnosis on a mismatch. `strict` never
/// rewrites (the streaming case checks against the fixture case's golden).
fn compare_ids(
    path: &Path,
    model_id: &str,
    case: &str,
    ids: &[u32],
    strict: bool,
) -> Result<(), String> {
    let golden = fs::read_to_string(path)
        .ok()
        .map(|t| parse_ids(&t))
        .transpose()?;
    if strict {
        let golden = golden.ok_or_else(|| format!("golden missing: {}", path.display()))?;
        return first_id_diff(&golden, ids).map_or(Ok(()), |d| {
            Err(format!(
                "{case}: token ids differ from {}: {d}",
                path.display()
            ))
        });
    }
    golden_diff(path, &ids_text(model_id, case, ids)).map_err(|e| match golden {
        Some(g) => format!(
            "{case}: {}\n{e}",
            first_id_diff(&g, ids).unwrap_or_default()
        ),
        None => format!("{case}: {e}"),
    })
}

fn compare_markdown(path: &Path, case: &str, markdown: &str, strict: bool) -> Result<(), String> {
    if strict {
        let golden = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        return if golden == markdown {
            Ok(())
        } else {
            Err(format!("{case}: markdown differs from {}", path.display()))
        };
    }
    golden_diff(path, markdown).map_err(|e| format!("{case}: {e}"))
}

/// Blank the volatile fields so a recaptured golden does not churn on timing.
fn blank_volatile(event: &Value) -> Value {
    match event {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| {
                    let v = if is_scrub_key(k) {
                        json!("[volatile]")
                    } else {
                        v.clone()
                    };
                    (k.clone(), v)
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

/// The canonical projection: each event compares on the GOLDEN's own field
/// set, minus the volatile fields. Fields the code adds later are ignored
/// (robot changes are additive, D15); a changed, dropped or extra event fails.
fn project_events(golden: &[Value], actual: &[Value]) -> Result<(), String> {
    for (i, (g, a)) in golden.iter().zip(actual).enumerate() {
        let g = g
            .as_object()
            .ok_or_else(|| format!("golden event {i} is not an object"))?;
        for (k, gv) in g {
            if is_scrub_key(k) {
                continue;
            }
            match a.get(k) {
                Some(av) if av == gv => {}
                Some(av) => {
                    return Err(format!("event {i} field {k:?}: golden {gv} vs actual {av}"));
                }
                None => return Err(format!("event {i} lost field {k:?} (golden {gv})")),
            }
        }
    }
    if golden.len() != actual.len() {
        return Err(format!(
            "event count differs: golden {} vs actual {}",
            golden.len(),
            actual.len()
        ));
    }
    Ok(())
}

fn events_ndjson(events: &[Value]) -> String {
    events
        .iter()
        .map(|e| format!("{}\n", blank_volatile(e)))
        .collect()
}

fn parse_ndjson(text: &str) -> Result<Vec<Value>, String> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| format!("bad NDJSON line {l:?}: {e}")))
        .collect()
}

fn compare_events(path: &Path, case: &str, events: &[Value]) -> Result<(), String> {
    let actual = events_ndjson(events);
    if update_goldens() {
        return golden_diff(path, &actual).map_err(|e| format!("{case}: {e}"));
    }
    let golden = match fs::read_to_string(path) {
        Ok(text) => parse_ndjson(&text)?,
        Err(e) => {
            let _ = fs::write(actual_sidecar(path), &actual);
            return Err(format!("{case}: golden missing: {} ({e})", path.display()));
        }
    };
    project_events(&golden, events).map_err(|d| {
        let _ = fs::write(actual_sidecar(path), &actual);
        format!(
            "{case}: page events differ from {}: {d} (wrote {})",
            path.display(),
            actual_sidecar(path).display()
        )
    })
}

// ───────────────────────────────────────────────────────────────────────────
// The lane driver.
// ───────────────────────────────────────────────────────────────────────────

fn extra(pairs: &[(&str, Value)]) -> Map<String, Value> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

fn git_revision() -> String {
    let run = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let rev = run(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".into());
    match run(&["status", "--porcelain", "--untracked-files=no"]) {
        Some(s) if !s.is_empty() => format!("{rev}+dirty"),
        _ => rev,
    }
}

fn run_lane(lane: &Lane) {
    let t0 = Instant::now();
    let mut log = Logger::new(TEST, lane.model_id);
    log.setup(0);
    let skip = |log: &mut Logger, why: &str| {
        log.skip_no_model(why);
        log.result("skip_no_model", t0.elapsed().as_micros());
        eprintln!("SUCCESS (skipped) {TEST} lane {}: {why}", lane.model_id);
    };
    if !env_flag(ARM_ENV) {
        return skip(
            &mut log,
            &format!(
                "not armed: {ARM_ENV} unset (run scripts/token_golden_gate.sh, which builds \
                 the optimized profile)"
            ),
        );
    }
    let update = update_goldens();
    let long = env_flag(LONG_ENV);
    let mut capture = load_capture(lane)
        .expect("CAPTURE.json parses")
        .unwrap_or_else(|| json!({ "model_id": lane.model_id, "artifact": null, "cases": {} }));
    let artifact = match resolve_artifact(lane, capture.get("artifact"), update) {
        Ok(a) => a,
        Err(why) => return skip(&mut log, &why),
    };
    let Some(corpus) = corpus_dir().filter(|d| d.is_dir()) else {
        return skip(
            &mut log,
            "no corpus: set FOCR_CORPUS_DIR to the 20-page corpus pages dir",
        );
    };

    let threads = franken_ocr::init_kernel_pool();
    let env = focr_env();
    log.emit(
        "assert",
        "pass",
        extra(&[
            ("assertion", json!("lane resolved its pinned artifact")),
            ("pass", json!(true)),
            ("artifact", json!(artifact.path.display().to_string())),
            ("artifact_sha256", json!(artifact.sha256)),
            ("threads", json!(threads)),
            ("update_goldens", json!(update)),
            ("long", json!(long)),
            ("focr_env", json!(env)),
        ]),
    );
    let model = OcrModel::load(&artifact.path).expect("load the pinned artifact");
    // A recapture takes the first artifact that exists, `FOCR_MODEL_PATH`
    // first: it must be this lane's model.
    assert_eq!(
        model.identity().model_id,
        lane.model_id,
        "{} is not a {} artifact (recapture one lane at a time, with FOCR_MODEL_PATH \
         naming that lane's artifact or unset)",
        artifact.path.display(),
        lane.model_id
    );
    let golden_dir = repo_path(lane.golden_dir);
    fs::create_dir_all(&golden_dir).expect("golden dir");
    let revision = git_revision();

    let mut failures: Vec<String> = Vec::new();
    for case in lane.cases.iter().filter(|c| long || !c.long) {
        let ct = Instant::now();
        let mut clog = Logger::new(TEST, case.name);
        let pages: Vec<PathBuf> = match case.inputs() {
            Ok(names) => names.iter().map(|n| corpus.join(n)).collect(),
            Err(e) => {
                failures.push(format!("{}: {e}", case.name));
                continue;
            }
        };
        let mut inputs = Map::new();
        let mut input_problem = None;
        for page in &pages {
            let name = page
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            match fs::read(page) {
                Ok(bytes) => {
                    inputs.insert(name, json!(sha256_hex(&bytes)));
                }
                Err(e) => input_problem = Some(format!("input {}: {e}", page.display())),
            }
        }
        let recorded = capture["cases"].get(case.golden_stem()).cloned();
        if input_problem.is_none()
            && !update
            && let Some(rec) = &recorded
        {
            if rec["inputs"] != Value::Object(inputs.clone()) {
                input_problem = Some(format!(
                    "inputs differ from the capture: have {}, captured {}",
                    Value::Object(inputs.clone()),
                    rec["inputs"]
                ));
            }
            if rec["artifact_sha256"] != json!(artifact.sha256) {
                input_problem = Some(format!(
                    "golden captured on artifact {} but the lane pin is {}: recapture \
                     the full lane (UPDATE_GOLDENS=1 FOCR_E2E_LONG=1)",
                    rec["artifact_sha256"], artifact.sha256
                ));
            }
        }
        if let Some(problem) = input_problem {
            clog.error("Usage", 2, &problem);
            failures.push(format!("{}: {problem}", case.name));
            continue;
        }
        clog.emit(
            "setup",
            "pass",
            extra(&[
                ("seed", json!(0)),
                ("inputs", Value::Object(inputs.clone())),
                ("artifact_sha256", json!(artifact.sha256)),
                ("threads", json!(threads)),
            ]),
        );

        let observed = match run_case(&model, case, &pages) {
            Ok(o) => o,
            Err(e) => {
                clog.error("decode", 1, &e);
                failures.push(format!("{}: {e}", case.name));
                continue;
            }
        };
        let strict = matches!(case.kind, Kind::Stream(_));
        let stem = case.golden_stem();
        let ids_path = golden_dir.join(format!("{stem}.ids.txt"));
        // The outcome is logged against the golden as it stood before any
        // recapture, so an UPDATE_GOLDENS run still reports what moved.
        let first_diff = fs::read_to_string(&ids_path)
            .ok()
            .and_then(|t| parse_ids(&t).ok())
            .and_then(|g| {
                let common = g
                    .iter()
                    .zip(&observed.ids)
                    .take_while(|(a, b)| a == b)
                    .count();
                (common != g.len() || common != observed.ids.len()).then_some(common)
            });
        let mut problems = Vec::new();
        if let Err(e) = compare_ids(&ids_path, lane.model_id, stem, &observed.ids, strict) {
            problems.push(e);
        }
        if let Err(e) = compare_markdown(
            &golden_dir.join(format!("{stem}.md")),
            stem,
            &observed.markdown,
            strict,
        ) {
            problems.push(e);
        }
        if let Some(events) = &observed.events {
            let path = golden_dir.join(format!("{}.page_events.ndjson", case.name));
            if let Err(e) = compare_events(&path, case.name, events) {
                problems.push(e);
            }
        }
        let elapsed_us = ct.elapsed().as_micros() as u64;
        let mut fields = extra(&[
            ("elapsed_us", json!(elapsed_us)),
            ("token_count", json!(observed.ids.len())),
            ("artifact_sha256", json!(artifact.sha256)),
            ("inputs", Value::Object(inputs.clone())),
            (
                "outcome",
                json!(
                    first_diff
                        .map_or_else(|| "equal".to_string(), |i| format!("first_diff_index={i}"))
                ),
            ),
            ("native_path_ran", json!(true)),
            ("fallback_target", json!(NATIVE_PATH_FALLBACK)),
        ]);
        if problems.is_empty() {
            clog.emit("result", "pass", fields);
        } else {
            let message = problems.join("\n");
            fields.insert(
                "diag".into(),
                json!({ "error_kind": "golden_mismatch", "focr_exit_code": 1, "message": message }),
            );
            clog.emit("result", "fail", fields);
            failures.push(message);
        }
        if update && !strict {
            capture["cases"][case.name] = json!({
                "kind": match case.kind { Kind::Single(_) => "single", Kind::Multi(_) => "multi", Kind::Stream(_) => "stream" },
                "long": case.long,
                "inputs": Value::Object(inputs),
                "token_count": observed.ids.len(),
                "artifact_sha256": artifact.sha256,
                "focr_version": env!("CARGO_PKG_VERSION"),
                "git_revision": revision,
                "threads": threads,
            });
        }
    }

    if update {
        capture["artifact"] = json!({
            "file_name": artifact.path.file_name().map(|n| n.to_string_lossy().into_owned()),
            "size": artifact.size,
            "sha256": artifact.sha256,
        });
        let text = serde_json::to_string_pretty(&capture).expect("serialize capture") + "\n";
        fs::write(golden_dir.join(CAPTURE_FILE), text).expect("write CAPTURE.json");
    }
    let outcome = if failures.is_empty() { "pass" } else { "fail" };
    if failures.is_empty() {
        log.result(outcome, t0.elapsed().as_micros());
    } else {
        log.emit(
            "result",
            outcome,
            extra(&[
                ("elapsed_us", json!(t0.elapsed().as_micros() as u64)),
                (
                    "diag",
                    json!({ "error_kind": "golden_mismatch", "focr_exit_code": 1,
                            "message": failures.join("\n") }),
                ),
            ]),
        );
    }
    assert!(
        failures.is_empty(),
        "{TEST} lane {}: {} case(s) failed; FOCR_* env in force: {:?}\n{}",
        lane.model_id,
        failures.len(),
        env,
        failures.join("\n\n")
    );
}

// ───────────────────────────────────────────────────────────────────────────
// Tests.
// ───────────────────────────────────────────────────────────────────────────

/// The v1 lane: the model-backed gate itself.
#[test]
fn v1_lane_matches_goldens() {
    run_lane(&V1_LANE);
}

/// The v2 lane (U2-C4): every later change on the v2 path keeps it green.
#[test]
fn v2_lane_matches_goldens() {
    run_lane(&V2_LANE);
}

/// Always on: an explicit model path that does not exist is a clean
/// `ModelNotFound` (exit 3), never a fallback to some other artifact. This is
/// what lets an armed lane log `native_path_ran: true`.
#[test]
fn native_path_has_no_fallback() {
    let mut log = Logger::new(TEST, "nonexistent_model");
    let t0 = Instant::now();
    let err = match OcrModel::load(Path::new(NATIVE_PATH_FALLBACK)) {
        Ok(_) => panic!("loading {NATIVE_PATH_FALLBACK} must fail"),
        Err(e) => e,
    };
    let ok = matches!(err, FocrError::ModelNotFound(_)) && err.exit_code() == 3;
    log.assertion("loading /nonexistent is ModelNotFound (exit 3)", ok);
    log.result(if ok { "pass" } else { "fail" }, t0.elapsed().as_micros());
    assert!(
        ok,
        "expected ModelNotFound (exit 3), got {} (exit {}): {err}",
        err.kind(),
        err.exit_code()
    );
}

/// Always on: whatever goldens are committed parse, and every captured case
/// has its files with a matching token count.
#[test]
fn committed_goldens_are_well_formed() {
    for lane in LANES {
        lane_goldens_are_well_formed(lane);
    }
}

fn lane_goldens_are_well_formed(lane: &Lane) {
    let mut log = Logger::new(TEST, lane.model_id);
    let t0 = Instant::now();
    let dir = repo_path(lane.golden_dir);
    let Some(capture) = load_capture(lane).expect("CAPTURE.json parses") else {
        // Deleting the manifest must not turn the gate into a silent skip.
        let orphans: Vec<String> = fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| !n.starts_with('.'))
            .collect();
        assert!(
            orphans.is_empty(),
            "{} has goldens but no {CAPTURE_FILE}: {orphans:?}",
            lane.golden_dir
        );
        log.assertion("no goldens captured yet", true);
        log.result("pass", t0.elapsed().as_micros());
        return;
    };
    assert_eq!(
        capture["model_id"],
        json!(lane.model_id),
        "CAPTURE.json model_id"
    );
    let pin = &capture["artifact"];
    assert!(
        pin["size"].as_u64().is_some() && pin["sha256"].as_str().is_some_and(|s| s.len() == 64),
        "CAPTURE.json artifact pin needs size + sha256: {pin}"
    );
    let cases = capture["cases"]
        .as_object()
        .expect("CAPTURE.json cases object");
    for (name, rec) in cases {
        let case = lane
            .cases
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("CAPTURE.json names unknown case {name:?}"));
        let ids = parse_ids(
            &fs::read_to_string(dir.join(format!("{name}.ids.txt"))).expect("ids golden"),
        )
        .expect("ids golden parses");
        assert_eq!(
            rec["token_count"],
            json!(ids.len()),
            "{name}: token_count vs ids file"
        );
        assert!(
            dir.join(format!("{name}.md")).is_file(),
            "{name}: markdown golden"
        );
        let mut expected = case.inputs().expect("case inputs");
        expected.sort();
        expected.dedup();
        let mut recorded: Vec<String> = rec["inputs"]
            .as_object()
            .expect("inputs")
            .keys()
            .cloned()
            .collect();
        recorded.sort();
        assert_eq!(recorded, expected, "{name}: recorded inputs");
        assert_eq!(
            rec["artifact_sha256"], pin["sha256"],
            "{name}: captured on the pinned artifact"
        );
    }
    for case in lane
        .cases
        .iter()
        .filter(|c| matches!(c.kind, Kind::Stream(_)))
    {
        let path = dir.join(format!("{}.page_events.ndjson", case.name));
        if let Ok(text) = fs::read_to_string(&path) {
            let events = parse_ndjson(&text).expect("page events parse");
            assert!(!events.is_empty(), "{}: no page events", case.name);
            assert!(
                events.iter().all(|e| e["event"] == json!("page")),
                "{}: only page events",
                case.name
            );
        }
    }
    log.assertion("committed goldens are well-formed", true);
    log.result("pass", t0.elapsed().as_micros());
}

/// Always on: the token comparator reports the first divergence, including
/// a truncated or extended stream.
#[test]
fn id_diff_names_first_divergence() {
    assert_eq!(first_id_diff(&[1, 2, 3], &[1, 2, 3]), None);
    let mid = first_id_diff(&[1, 2, 3, 4], &[1, 2, 9, 4]).expect("mid diff");
    assert!(
        mid.contains("index 2") && mid.contains("golden id 3 vs actual id 9"),
        "{mid}"
    );
    let short = first_id_diff(&[1, 2, 3], &[1, 2]).expect("truncated");
    assert!(
        short.contains("index 2") && short.contains("actual id <end>"),
        "{short}"
    );
    let long = first_id_diff(&[1, 2], &[1, 2, 7]).expect("extended");
    assert!(long.contains("golden id <end> vs actual id 7"), "{long}");
    let text = ids_text("unlimited-ocr", "x", &(0..40).collect::<Vec<u32>>());
    assert_eq!(
        parse_ids(&text).expect("round trip"),
        (0..40).collect::<Vec<u32>>()
    );
}

/// Always on: the canonical projection ignores added and volatile fields and
/// nothing else.
#[test]
fn event_projection_ignores_only_added_and_volatile_fields() {
    let golden = vec![
        json!({"schema_version": 1, "event": "page", "status": "decoded", "page": 1,
               "chars": 5, "text": "hello", "elapsed_ms": "[volatile]"}),
    ];
    let mut later = golden[0].clone();
    later["elapsed_ms"] = json!(812);
    later["model_id"] = json!("unlimited-ocr");
    assert_eq!(project_events(&golden, &[later.clone()]), Ok(()));

    let mut changed = later.clone();
    changed["text"] = json!("hellp");
    assert!(project_events(&golden, &[changed]).is_err_and(|e| e.contains("\"text\"")));
    let mut dropped = later.clone();
    dropped.as_object_mut().expect("object").remove("chars");
    assert!(project_events(&golden, &[dropped]).is_err_and(|e| e.contains("lost field")));
    assert!(project_events(&golden, &[later.clone(), later]).is_err_and(|e| e.contains("count")));
    assert!(project_events(&golden, &[]).is_err_and(|e| e.contains("count")));
    assert_eq!(
        blank_volatile(&json!({"elapsed_ms": 3, "page": 2})),
        json!({"elapsed_ms": "[volatile]", "page": 2})
    );
}
