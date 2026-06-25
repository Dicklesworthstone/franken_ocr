#![allow(clippy::doc_overindented_list_items, clippy::doc_lazy_continuation)]
//! OQ-16 — Tokenizer conformance harness (token-id-EXACT vs HF
//! `LlamaTokenizerFast`).
//!
//! This is the **prerequisite gate for every downstream conformance check**
//! (AGENTS.md Testing Policy; `docs/truth-pack/oq/tokenizer.md`): a single
//! drifted token id corrupts the decoder prompt and silently invalidates every
//! logits/hidden-state/CER comparison above it on the parity ladder. So this
//! harness holds the pure-Rust byte-level-BPE tokenizer (`franken_ocr::tokenizer`)
//! token-id-exact against golden vectors dumped from the reference tokenizer.
//!
//! HOW THE GOLDEN IS PRODUCED (provenance)
//! ---------------------------------------
//! `scripts/gen_tokenizer_fixtures.py` loads the pinned
//! `docs/truth-pack/snapshots/tokenizer.json`
//! (SHA-256 `a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4`)
//! via the HF `tokenizers` Rust crate (or `transformers.PreTrainedTokenizerFast`
//! as a fallback — both load the SAME `tokenizer.json` and yield identical ids),
//! encodes every `tests/fixtures/tokenizer/corpus.txt` case with
//! `add_special_tokens=False` (the inference path — the model, not the tokenizer,
//! owns BOS=0/EOS=1; see `modeling_unlimitedocr.py:259-268`,`:966-968`), and
//! writes one `{"text","ids","decoded"}` record per line to `expected.jsonl`,
//! preceded by a `{"_meta":true,...}` provenance line (tokenizer.json sha,
//! backend + version, bos/eos/pad/image ids). The fixture is GENERATED, never
//! hand-edited; regenerate with `python3 scripts/gen_tokenizer_fixtures.py`.
//!
//! WHAT THIS HARNESS ASSERTS, per non-`_meta` record:
//!   1. ENCODE (exact):     `Tokenizer::encode(text)            == ids`
//!   2. DECODE (round-trip): `Tokenizer::decode(ids)            == decoded`
//! Failures are self-diagnosing: they print the case index, the (escaped) text,
//! the expected ids, the actual ids, AND the first divergent index — and dump the
//! full actual stream to `expected.jsonl.actual` (the golden-artifacts `*.actual`
//! convention) so the diff is inspectable without a rerun.
//!
//! GATING (skip-with-SUCCESS) & XFAIL DISCIPLINE
//! ---------------------------------------------
//! * **Fixture-gated**: when `tokenizer.json` OR `expected.jsonl` is absent (the
//!   9.5 MB snapshot and the generated golden are not always present in a fresh
//!   checkout / CI), the conformance test prints a single explicit `SUCCESS`
//!   line stating exactly what was missing and why it skipped, then PASSES. It
//!   never silently no-ops and never fails for a missing fixture.
//! * **Impl-gated XFAIL (not SKIP)**: while `franken_ocr::tokenizer::Tokenizer`
//!   is still the Phase-1 stub (its `load`/`encode`/`decode` return
//!   `FocrError::NotImplemented`), the conformance run records an **XFAIL** with a
//!   SUCCESS line — the known, tracked divergence — rather than red-barring CI on
//!   work another agent is mid-flight on. The moment the real BPE tokenizer lands
//!   (no more `NotImplemented`), the XFAIL path disappears and the assertions go
//!   live automatically. This mirrors the conformance-harness doctrine: XFAIL a
//!   known divergence, never SKIP it, and let it auto-promote to a hard gate.
//!
//! The synthetic-vocab BPE unit tests at the bottom are **always-on** (no big
//! files, no gating) so the harness's OWN diff/round-trip/first-divergence logic
//! is exercised on every `cargo test`, even with no fixtures present.

use std::path::{Path, PathBuf};

use franken_ocr::error::FocrError;
use franken_ocr::tokenizer::{Tokenizer, special};

// ---------------------------------------------------------------------------
// Paths (resolved against CARGO_MANIFEST_DIR so the test is cwd-independent).
// ---------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    // Integration tests run with CARGO_MANIFEST_DIR = the crate root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn tokenizer_json_path() -> PathBuf {
    repo_root().join("docs/truth-pack/snapshots/tokenizer.json")
}

fn corpus_path() -> PathBuf {
    repo_root().join("tests/fixtures/tokenizer/corpus.txt")
}

fn expected_jsonl_path() -> PathBuf {
    repo_root().join("tests/fixtures/tokenizer/expected.jsonl")
}

/// Where the full actual id-stream is dumped on mismatch (golden-artifacts
/// `*.actual` convention). Lives next to the golden; should be gitignored
/// (`tests/fixtures/**/*.actual`) — recorded in the agent handoff for the
/// `.gitignore` owner.
fn actual_dump_path() -> PathBuf {
    repo_root().join("tests/fixtures/tokenizer/expected.jsonl.actual")
}

// ---------------------------------------------------------------------------
// Structured logging — every test emits a clear, greppable line on what it
// exercised, the inputs, and expected-vs-actual (a first-class requirement).
// ---------------------------------------------------------------------------

/// One structured log line. `kind` is a stable tag agents/CI can grep on
/// (RUN / SUCCESS / XFAIL / FAIL / INFO).
fn logline(kind: &str, msg: &str) {
    println!("[tokenizer_conformance][{kind}] {msg}");
}

/// Escape a string for single-line diagnostics (so embedded tabs/newlines/CR and
/// non-ASCII never break the log line or hide the real input). Mirrors the
/// JSON-encoded form the corpus uses.
fn escape_for_log(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// First index at which two id streams diverge (or `None` if one is a prefix of
/// the other AND they are equal length). When lengths differ but the shorter is a
/// prefix of the longer, the divergence index is the length of the shorter.
fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i] != b[i] {
            return Some(i);
        }
    }
    if a.len() != b.len() { Some(n) } else { None }
}

// ---------------------------------------------------------------------------
// Fixture model (mirrors the {"text","ids","decoded"} JSONL shape).
// ---------------------------------------------------------------------------

struct Case {
    /// 1-based record index (excluding the `_meta` line) for diagnostics.
    index: usize,
    text: String,
    ids: Vec<u32>,
    decoded: String,
}

struct Fixture {
    meta_summary: String,
    cases: Vec<Case>,
}

/// Parse `expected.jsonl` using `serde_json` (already a crate dependency, the same
/// JSON engine the lib uses — so embedded control chars, `\uXXXX`, surrogate
/// pairs, and non-ASCII in the corpus round-trip exactly; no hand-rolled JSON
/// string parser to get subtly wrong).
fn load_fixture(path: &Path) -> Result<Fixture, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read fixture {}: {e}", path.display()))?;

    let mut cases = Vec::new();
    let mut meta_summary = String::from("(no _meta line found)");
    let mut record_index = 0usize;

    for (lineno, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| format!("{}:{}: invalid JSON: {e}", path.display(), lineno + 1))?;

        // The provenance line carries "_meta": true — summarize, don't treat as a
        // case.
        if v.get("_meta").and_then(serde_json::Value::as_bool) == Some(true) {
            let sha = v
                .get("tokenizer_json_sha256")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("?");
            let backend = v
                .get("backend")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("?");
            let version = v
                .get("backend_version")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("?");
            let num = v
                .get("num_cases")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            meta_summary = format!(
                "tokenizer.json sha256={sha} backend={backend} v{version} num_cases={num} \
                 add_special_tokens=false (model owns bos=0/eos=1)"
            );
            continue;
        }

        let text = v
            .get("text")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                format!(
                    "{}:{}: record missing string \"text\"",
                    path.display(),
                    lineno + 1
                )
            })?
            .to_string();

        let ids_val = v
            .get("ids")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                format!(
                    "{}:{}: record missing array \"ids\"",
                    path.display(),
                    lineno + 1
                )
            })?;
        let mut ids = Vec::with_capacity(ids_val.len());
        for (i, idv) in ids_val.iter().enumerate() {
            let id = idv.as_u64().ok_or_else(|| {
                format!(
                    "{}:{}: ids[{i}] is not a non-negative integer",
                    path.display(),
                    lineno + 1
                )
            })?;
            let id = u32::try_from(id).map_err(|_| {
                format!(
                    "{}:{}: ids[{i}] = {id} overflows u32",
                    path.display(),
                    lineno + 1
                )
            })?;
            ids.push(id);
        }

        let decoded = v
            .get("decoded")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                format!(
                    "{}:{}: record missing string \"decoded\"",
                    path.display(),
                    lineno + 1
                )
            })?
            .to_string();

        record_index += 1;
        cases.push(Case {
            index: record_index,
            text,
            ids,
            decoded,
        });
    }

    if cases.is_empty() {
        return Err(format!(
            "fixture {} contained no non-_meta cases (regenerate via \
             scripts/gen_tokenizer_fixtures.py)",
            path.display()
        ));
    }

    Ok(Fixture {
        meta_summary,
        cases,
    })
}

/// Detect the Phase-1 stub: `load`/`encode`/`decode` all return
/// `FocrError::NotImplemented`. While that holds, the conformance run is an XFAIL,
/// not a hard failure (the real BPE tokenizer is being written concurrently).
fn is_not_implemented(err: &FocrError) -> bool {
    matches!(err, FocrError::NotImplemented(_))
}

// ===========================================================================
// THE CONFORMANCE GATE (fixture-gated; XFAIL while the impl is a stub).
// ===========================================================================

#[test]
fn tokenizer_conformance_vs_reference() {
    let tok_json = tokenizer_json_path();
    let expected = expected_jsonl_path();

    logline(
        "RUN",
        &format!(
            "OQ-16 token-id-exact gate | tokenizer.json={} | golden={} | corpus={}",
            tok_json.display(),
            expected.display(),
            corpus_path().display()
        ),
    );

    // ---- Gate 1: fixture presence (skip-with-SUCCESS, never a failure). ------
    if !tok_json.exists() {
        logline(
            "SUCCESS",
            &format!(
                "SKIP (fixture-gated): reference tokenizer.json absent at {}. The 9.5 MB \
                 snapshot is not in this checkout; nothing to compare against. This is an \
                 expected skip, not a failure.",
                tok_json.display()
            ),
        );
        return;
    }
    if !expected.exists() {
        logline(
            "SUCCESS",
            &format!(
                "SKIP (fixture-gated): golden {} not generated. Run \
                 `python3 scripts/gen_tokenizer_fixtures.py` (needs tokenizers>=0.15 or \
                 transformers>=4.40) to dump it from the pinned tokenizer.json. Expected \
                 skip, not a failure.",
                expected.display()
            ),
        );
        return;
    }

    // ---- Load + summarize the golden (provenance into the log). --------------
    let fixture = match load_fixture(&expected) {
        Ok(f) => f,
        Err(e) => panic!(
            "[tokenizer_conformance][FAIL] golden fixture is corrupt: {e}\n  \
             (regenerate with `python3 scripts/gen_tokenizer_fixtures.py`)"
        ),
    };
    logline(
        "INFO",
        &format!(
            "loaded golden: {} cases | provenance: {}",
            fixture.cases.len(),
            fixture.meta_summary
        ),
    );

    // ---- Gate 2: Rust tokenizer load. ---------------------------------------
    // A `NotImplemented` here = the Phase-1 stub → XFAIL-with-SUCCESS. Any OTHER
    // error from a real loader is a genuine failure (the tokenizer.json exists).
    let tokenizer = match Tokenizer::load(&tok_json) {
        Ok(t) => t,
        Err(ref e) if is_not_implemented(e) => {
            logline(
                "XFAIL",
                &format!(
                    "Tokenizer::load is the Phase-1 stub ({e}). The conformance gate is \
                     fixture-ready ({} golden cases over {}) and will go live automatically \
                     when the byte-level-BPE tokenizer lands (bd-1gv.1). Tracked divergence, \
                     not a regression.",
                    fixture.cases.len(),
                    fixture.meta_summary
                ),
            );
            logline(
                "SUCCESS",
                "XFAIL recorded (impl-gated): tokenizer stub not yet implemented.",
            );
            return;
        }
        Err(e) => panic!(
            "[tokenizer_conformance][FAIL] Tokenizer::load({}) failed with a NON-stub error \
             (the tokenizer.json IS present, so this is a real loader bug): {e:?}",
            tok_json.display()
        ),
    };

    // Cross-check the hardcoded special-token ids before the corpus sweep, so a
    // miswired special table is caught with a crisp message (these are the ids the
    // prompt-builder hardcodes; OQ-16 §6).
    assert_special_ids_match(&tokenizer);

    // ---- The sweep: encode-exact + decode-round-trip for every case. --------
    let mut failures: Vec<String> = Vec::new();
    let mut actual_dump = String::new();
    let mut first_encode_err: Option<FocrError> = None;
    let total = fixture.cases.len();

    for case in &fixture.cases {
        // ENCODE
        let actual_ids = match tokenizer.encode(&case.text) {
            Ok(ids) => ids,
            Err(ref e) if is_not_implemented(e) => {
                // load() succeeded but encode() is still a stub: XFAIL the whole
                // sweep (consistent with the load-stub branch above).
                first_encode_err = Some(FocrError::NotImplemented(format!("{e}")));
                break;
            }
            Err(e) => {
                failures.push(format!(
                    "case #{} text={}: encode() errored: {e:?}",
                    case.index,
                    escape_for_log(&case.text)
                ));
                continue;
            }
        };

        // Record the actual stream (for the *.actual dump) regardless of match.
        actual_dump.push_str(&format!(
            "{{\"index\":{},\"text\":{},\"ids\":{:?}}}\n",
            case.index,
            serde_json::to_string(&case.text).unwrap_or_else(|_| "\"<unprintable>\"".into()),
            actual_ids
        ));

        if actual_ids != case.ids {
            let div = first_divergence(&case.ids, &actual_ids);
            let div_detail = match div {
                Some(i) => {
                    let exp = case
                        .ids
                        .get(i)
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "<none>".into());
                    let act = actual_ids
                        .get(i)
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "<none>".into());
                    format!("first divergence at index {i}: expected id {exp}, got {act}")
                }
                None => {
                    "lengths/content differ but no scalar divergence found (unreachable)".into()
                }
            };
            failures.push(format!(
                "ENCODE MISMATCH case #{} text={}\n      expected ids ({}): {:?}\n      \
                 actual   ids ({}): {:?}\n      {}",
                case.index,
                escape_for_log(&case.text),
                case.ids.len(),
                case.ids,
                actual_ids.len(),
                actual_ids,
                div_detail
            ));
            // Continue scanning so the *.actual dump and failure list are complete.
        }

        // DECODE round-trip (decode the GOLDEN ids, compare to golden decoded
        // string — independent of whether encode matched).
        match tokenizer.decode(&case.ids) {
            Ok(actual_decoded) => {
                if actual_decoded != case.decoded {
                    failures.push(format!(
                        "DECODE MISMATCH case #{} ids={:?}\n      expected decoded: {}\n      \
                         actual   decoded: {}",
                        case.index,
                        case.ids,
                        escape_for_log(&case.decoded),
                        escape_for_log(&actual_decoded)
                    ));
                }
            }
            Err(ref e) if is_not_implemented(e) => {
                if first_encode_err.is_none() {
                    first_encode_err = Some(FocrError::NotImplemented(format!("decode: {e}")));
                }
                break;
            }
            Err(e) => failures.push(format!(
                "case #{} ids={:?}: decode() errored: {e:?}",
                case.index, case.ids
            )),
        }
    }

    // Encode/decode turned out to still be a stub → XFAIL the sweep.
    if let Some(e) = first_encode_err {
        logline(
            "XFAIL",
            &format!(
                "Tokenizer::load succeeded but encode/decode is still the Phase-1 stub ({e}). \
                 Gate is fixture-ready over {total} cases; goes live when bd-1gv.1 lands."
            ),
        );
        logline(
            "SUCCESS",
            "XFAIL recorded (impl-gated): encode/decode stub not yet implemented.",
        );
        return;
    }

    // ---- Verdict. -----------------------------------------------------------
    if failures.is_empty() {
        logline(
            "SUCCESS",
            &format!(
                "PASS: all {total} corpus cases are token-id-EXACT and decode round-trip \
                 against the reference LlamaTokenizerFast ({}).",
                fixture.meta_summary
            ),
        );
        // Stale *.actual from a previous failing run is misleading once green;
        // best-effort remove (RULE 0/1 forbid deleting source — this is a
        // test-emitted artifact this test owns, so removal is fine).
        let _ = std::fs::remove_file(actual_dump_path());
        return;
    }

    // Dump the full actual id-stream next to the golden for offline diffing.
    let dump_path = actual_dump_path();
    let dump_note = match std::fs::write(&dump_path, &actual_dump) {
        Ok(()) => format!("full actual id-stream written to {}", dump_path.display()),
        Err(e) => format!("(could not write {}: {e})", dump_path.display()),
    };

    let shown = failures.len().min(40);
    let mut report = String::new();
    report.push_str(&format!(
        "\n[tokenizer_conformance][FAIL] {} of {total} cases diverged from the reference \
         LlamaTokenizerFast.\n  {}\n  provenance: {}\n",
        failures.len(),
        dump_note,
        fixture.meta_summary
    ));
    for f in failures.iter().take(shown) {
        report.push_str("\n  - ");
        report.push_str(f);
        report.push('\n');
    }
    if failures.len() > shown {
        report.push_str(&format!(
            "\n  ... and {} more (see {})\n",
            failures.len() - shown,
            dump_path.display()
        ));
    }
    panic!("{report}");
}

/// Cross-check the hardcoded special-token id constants against the reference: a
/// real tokenizer MUST encode each special-token *string* to exactly its constant
/// id (single-token), since the prompt-builder and the model hardcode these
/// (OQ-16 §6). XFAIL-safe: a stub `encode` is tolerated by the caller's
/// `NotImplemented` handling, but a real encoder that gets these wrong fails loudly.
fn assert_special_ids_match(tokenizer: &Tokenizer) {
    // (string, expected id). Glyph vs ASCII-pipe distinction is load-bearing
    // (OQ-16 §6): the fullwidth-bar `｜` (U+FF5C) BOS/EOS are NOT the ASCII forms.
    let checks: &[(&str, u32)] = &[
        ("<｜begin▁of▁sentence｜>", special::BOS),
        ("<｜end▁of▁sentence｜>", special::EOS),
        ("<image>", special::IMAGE),
        ("<|ref|>", special::REF),
        ("<|/ref|>", special::REF_END),
        ("<|det|>", special::DET),
        ("<|/det|>", special::DET_END),
        ("<|grounding|>", special::GROUNDING),
        ("<|User|>", special::USER),
        ("<|Assistant|>", special::ASSISTANT),
    ];
    for (s, want) in checks {
        match tokenizer.encode(s) {
            Ok(ids) => {
                assert_eq!(
                    ids.as_slice(),
                    &[*want],
                    "[tokenizer_conformance][FAIL] special token {} must encode to exactly \
                     [{}] (OQ-16 §6), got {:?}",
                    escape_for_log(s),
                    want,
                    ids
                );
            }
            // A stub here is fine; the caller already XFAILs the whole run.
            Err(ref e) if is_not_implemented(e) => return,
            Err(e) => panic!(
                "[tokenizer_conformance][FAIL] encoding special {} errored: {e:?}",
                escape_for_log(s)
            ),
        }
    }
    logline(
        "INFO",
        "special-token id cross-check passed (bos/eos/image/ref/det/grounding/user/assistant).",
    );
}

// ===========================================================================
// ALWAYS-ON: synthetic-vocab BPE unit tests.
//
// These need NO big files and NO gating — they exercise the harness's own logic
// (first-divergence detection, escaping, round-trip comparison) AND a tiny,
// fully-specified byte-level BPE so the conformance MACHINERY is covered on every
// `cargo test`, even on a checkout with no tokenizer.json / no expected.jsonl.
//
// The mini-BPE below is a faithful, self-contained model of the real algorithm:
// byte-level symbols, rank-ordered merges (lower rank wins, applied greedily),
// and an id table. It is NOT the franken_ocr tokenizer (that lands in src/); it is
// a reference fake used to prove the comparison logic catches drift.
// ===========================================================================

/// A toy byte-level BPE over a fixed vocab + merge list, mirroring the real
/// algorithm's shape (rank = merge-list index; greedily apply the lowest-rank
/// adjacent merge until none apply). Used only to test the harness machinery.
struct ToyBpe {
    /// token string -> id
    vocab: std::collections::HashMap<String, u32>,
    /// (left, right) -> rank (lower wins)
    ranks: std::collections::HashMap<(String, String), usize>,
    /// id -> token string (for decode)
    inv: std::collections::HashMap<u32, String>,
}

impl ToyBpe {
    /// Build a tiny model. Single bytes 'a'..'d' get ids 0..3; merges build
    /// "ab"(4), "abc"(5), "cd"(6). Merge order: [(a,b)->ab, (ab,c)->abc, (c,d)->cd].
    fn new() -> Self {
        let mut vocab = std::collections::HashMap::new();
        for (i, s) in ["a", "b", "c", "d", "ab", "abc", "cd"].iter().enumerate() {
            vocab.insert((*s).to_string(), i as u32);
        }
        let merges = [("a", "b"), ("ab", "c"), ("c", "d")];
        let mut ranks = std::collections::HashMap::new();
        for (rank, (l, r)) in merges.iter().enumerate() {
            ranks.insert(((*l).to_string(), (*r).to_string()), rank);
        }
        let inv = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        ToyBpe { vocab, ranks, inv }
    }

    /// Greedy lowest-rank BPE over the byte symbols of `text` (ASCII only, for the
    /// toy). Returns ids; unknown single chars would panic (the toy corpus is
    /// closed), which is fine — this is harness machinery coverage, not the real
    /// tokenizer.
    fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut symbols: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        loop {
            // Find the adjacent pair with the lowest rank.
            let mut best: Option<(usize, usize)> = None; // (rank, position)
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(&rank) = self
                    .ranks
                    .get(&(symbols[i].clone(), symbols[i + 1].clone()))
                    && best.is_none_or(|(br, _)| rank < br)
                {
                    best = Some((rank, i));
                }
            }
            let Some((_, pos)) = best else { break };
            let merged = format!("{}{}", symbols[pos], symbols[pos + 1]);
            symbols.splice(pos..=pos + 1, std::iter::once(merged));
        }
        symbols
            .iter()
            .map(|s| {
                *self
                    .vocab
                    .get(s)
                    .unwrap_or_else(|| panic!("toy vocab miss: {s:?}"))
            })
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|id| {
                self.inv
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| panic!("toy id miss: {id}"))
            })
            .collect()
    }
}

#[test]
fn synthetic_bpe_encode_is_greedy_lowest_rank() {
    let t = ToyBpe::new();
    // "abc": (a,b)->ab rank0 applies first → [ab, c]; then (ab,c)->abc rank1 → [abc] = [5].
    assert_eq!(t.encode("abc"), vec![5], "greedy merge to 'abc'");
    // "abcd": → abc + d ... (c,d) can't fire after c is consumed by abc; so [abc, d] = [5,3].
    assert_eq!(
        t.encode("abcd"),
        vec![5, 3],
        "abc consumes c before cd can merge"
    );
    // "cd": only (c,d)->cd rank2 fires → [6].
    assert_eq!(t.encode("cd"), vec![6], "cd merge");
    // "ab": (a,b)->ab → [4].
    assert_eq!(t.encode("ab"), vec![4]);
    // empty → empty (matches add_special_tokens=false on "").
    assert_eq!(t.encode(""), Vec::<u32>::new(), "empty encodes to empty");
    logline(
        "SUCCESS",
        "synthetic BPE encode is greedy-lowest-rank (machinery covered).",
    );
}

#[test]
fn synthetic_bpe_round_trips() {
    let t = ToyBpe::new();
    for s in ["abc", "abcd", "cd", "ab", "a", "dcba", "abccd", ""] {
        let ids = t.encode(s);
        let back = t.decode(&ids);
        assert_eq!(
            back, s,
            "round-trip failed for {s:?}: ids={ids:?} -> {back:?}"
        );
    }
    logline(
        "SUCCESS",
        "synthetic BPE round-trips on the toy corpus (machinery covered).",
    );
}

#[test]
fn first_divergence_locates_the_exact_index() {
    // Equal → no divergence.
    assert_eq!(first_divergence(&[1, 2, 3], &[1, 2, 3]), None);
    // Mid-stream scalar divergence.
    assert_eq!(first_divergence(&[1, 2, 3], &[1, 9, 3]), Some(1));
    // Prefix-shorter: divergence at the length of the shorter.
    assert_eq!(first_divergence(&[1, 2], &[1, 2, 3]), Some(2));
    assert_eq!(first_divergence(&[1, 2, 3], &[1, 2]), Some(2));
    // First element differs.
    assert_eq!(first_divergence(&[9], &[1]), Some(0));
    // Both empty → no divergence.
    assert_eq!(first_divergence(&[], &[]), None);
    // One empty.
    assert_eq!(first_divergence(&[], &[7]), Some(0));
    logline(
        "SUCCESS",
        "first_divergence pinpoints the exact divergent index (diagnostic covered).",
    );
}

#[test]
fn escape_for_log_is_single_line_and_lossless_for_control_chars() {
    // Tabs/newlines/CR and a sub-0x20 control char must not break the log line.
    assert_eq!(escape_for_log("a\tb"), "\"a\\tb\"");
    assert_eq!(escape_for_log("x\ny"), "\"x\\ny\"");
    assert_eq!(escape_for_log("r\rn"), "\"r\\rn\"");
    assert_eq!(escape_for_log("q\"q"), "\"q\\\"q\"");
    assert_eq!(escape_for_log("back\\slash"), "\"back\\\\slash\"");
    assert_eq!(escape_for_log("\u{0007}"), "\"\\u0007\""); // BEL
    // Non-ASCII passes through (the corpus is full of CJK/emoji); just ensure it
    // stays one line and is quoted.
    let cjk = escape_for_log("你好");
    assert!(cjk.starts_with('"') && cjk.ends_with('"') && !cjk.contains('\n'));
    logline(
        "SUCCESS",
        "escape_for_log keeps diagnostics single-line and control-char-safe.",
    );
}

#[test]
fn special_id_constants_are_the_oq16_pinned_values() {
    // Guards against an accidental edit to the special-id table the prompt-builder
    // and embedding bead depend on (OQ-16 §6). Always-on; no fixtures needed.
    assert_eq!(special::BOS, 0, "bos id");
    assert_eq!(special::EOS, 1, "eos id");
    assert_eq!(special::IMAGE, 128815, "<image> id");
    assert_eq!(special::REF, 128816, "<|ref|> id");
    assert_eq!(special::REF_END, 128817, "<|/ref|> id");
    assert_eq!(special::DET, 128818, "<|det|> id");
    assert_eq!(special::DET_END, 128819, "<|/det|> id");
    assert_eq!(special::GROUNDING, 128820, "<|grounding|> id");
    assert_eq!(special::USER, 128825, "<|User|> id");
    assert_eq!(special::ASSISTANT, 128826, "<|Assistant|> id");
    logline(
        "SUCCESS",
        "special-id constants match the OQ-16-pinned values.",
    );
}
