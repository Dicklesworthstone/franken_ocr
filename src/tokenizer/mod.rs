//! Pure-Rust byte-level BPE tokenizer over `tokenizer.json` ([SPEC-019,
//! SPEC-035], OQ-16). Token-id-exact vs HF `LlamaTokenizerFast` — the L0/L4
//! prerequisite for every downstream conformance gate (AGENTS.md testing
//! policy): a single mismatched id corrupts the whole parity ladder.
//!
//! The algorithm is GPT-2-style **byte-level BPE** (NOT SentencePiece, NOT
//! Llama): a no-op normalizer, a four-stage byte-level pre-tokenizer
//! ([`pretok`]), merge-rank BPE over the 256-symbol byte alphabet, and an
//! added/special-token table that is split out of the text *before* BPE (HF
//! `AddedVocabulary` semantics). There is no UNK and no algorithmic
//! byte-fallback: full byte coverage comes from the byte→unicode remap
//! (OQ-16 §2).
//!
//! BOS/EOS policy (OQ-16 §5): we do **not** auto-append EOS and do **not**
//! auto-prepend BOS inside [`Tokenizer::encode`] (it encodes with
//! `add_special_tokens=False`, matching `modeling_unlimitedocr.py:260`). The
//! prompt builder prepends a single id-0 BOS at the very front of the final
//! sequence; the special-token ids it needs are exposed via [`special`] and the
//! [`Tokenizer`] accessors.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::{FocrError, FocrResult};

pub mod music;
mod ops;
/// Pure pretokenizers (public for the fuzz harness — bd-10sb.1: the
/// hand-rolled Unicode split classes are an untrusted-input surface; the
/// fuzz targets assert totality + lossless re-join over arbitrary text).
pub mod pretok;
pub mod tiktoken;
mod unicode_tables;

pub use ops::TokenizerOps;

/// SmolVLM2 / SmolLM2 special-token ids (C6, bd-3jo6.3.6) — pinned by
/// `docs/zoo/smolvlm2-spec.md` §5 (verified against the released
/// `tokenizer.json` + `added_tokens.json` + `generation_config.json`) and
/// cross-checked against the loaded added-token table under the
/// [`PretokScheme::SmolLm2`] scheme.
pub mod special_smollm2 {
    /// `<|endoftext|>` — the unk token (id 0). `generation_config.json` also
    /// labels it bos, but the chat template supplies `<|im_start|>` literally
    /// and nothing auto-prepends (spec §5/§7 — the disagreement is harmless).
    pub const UNK: u32 = 0;
    /// `<|im_start|>` — the tokenizer's bos (`tokenizer_config.json`).
    pub const BOS: u32 = 1;
    /// `<|im_end|>` — the tokenizer's pad (`text_config.pad_token_id` = 2).
    pub const PAD: u32 = 2;
    /// `<global-img>` — brackets the global (thumbnail-LAST) frame.
    pub const GLOBAL_IMG: u32 = 49152;
    /// `<fake_token_around_image>` — the image-expansion bracket token.
    pub const FAKE_AROUND_IMAGE: u32 = 49189;
    /// `<image>` — the splice slot id (`image_token_id`); 64 per 512² frame.
    pub const IMAGE: u32 = 49190;
    /// `<end_of_utterance>` — the generation stop id (eos, 49279).
    pub const END_OF_UTTERANCE: u32 = 49279;
}

/// OneChart / OPT special-token ids (D9, bd-3jo6.4.9) — pinned by
/// `docs/zoo/onechart-spec.md` §4/§7 (verified against `added_tokens.json` +
/// `config.json`) and cross-checked against the loaded table under the
/// [`PretokScheme::Gpt2`] scheme.
pub mod special_opt {
    /// `<pad>` (OPT config `pad_token_id`).
    pub const PAD: u32 = 1;
    /// `</s>` — OPT's bos AND eos (config `bos/eos_token_id` = 2). The prompt
    /// builder owns the leading `</s>`; encode adds nothing.
    pub const BOS_EOS: u32 = 2;
    /// `<imgpad>` — the 256 vision splice slots (GOT lineage).
    pub const IMG_PAD: u32 = 50_265;
    /// `<img>` opening bracket.
    pub const IMG_START: u32 = 50_266;
    /// `</img>` closing bracket.
    pub const IMG_END: u32 = 50_267;
    /// `<Number>` — the number-head trigger token (census §8).
    pub const NUMBER: u32 = 50_268;
}

/// Hardcoded special-token ids ([SPEC-014/019]). These are pinned by the model
/// runtime (`modeling_unlimitedocr.py`) and cross-checked against
/// `tokenizer.json .added_tokens`; the loader asserts the loaded table agrees.
pub mod special {
    /// `<｜begin▁of▁sentence｜>` ([SPEC-014]). Note the fullwidth bar U+FF5C and
    /// the bullet U+2581 — DeepSeek-V2 glyphs, NOT ASCII `|`/`_`.
    pub const BOS: u32 = 0;
    /// `<｜end▁of▁sentence｜>` ([SPEC-014]).
    pub const EOS: u32 = 1;
    /// `<｜▁pad▁｜>` ([SPEC-014]).
    pub const PAD: u32 = 2;
    /// `<image>` ([SPEC-019]); the runtime hardcodes this id
    /// (`modeling_unlimitedocr.py:845 image_token_id = 128815`).
    pub const IMAGE: u32 = 128815;
    /// `<|ref|>` ([SPEC-019]).
    pub const REF: u32 = 128816;
    /// `<|/ref|>` ([SPEC-019]).
    pub const REF_END: u32 = 128817;
    /// `<|det|>` ([SPEC-019]).
    pub const DET: u32 = 128818;
    /// `<|/det|>` ([SPEC-019]).
    pub const DET_END: u32 = 128819;
    /// `<|grounding|>` ([SPEC-019]).
    pub const GROUNDING: u32 = 128820;
    /// `<td>` ([SPEC-019]).
    pub const TD: u32 = 128821;
    /// `</td>` ([SPEC-019]).
    pub const TD_END: u32 = 128822;
    /// `<tr>` ([SPEC-019]).
    pub const TR: u32 = 128823;
    /// `</tr>` ([SPEC-019]).
    pub const TR_END: u32 = 128824;
    /// `<|User|>` ([SPEC-019]); ASCII-bar variant (distinct from `<｜User｜>`).
    pub const USER: u32 = 128825;
    /// `<|Assistant|>` ([SPEC-019]); ASCII-bar variant.
    pub const ASSISTANT: u32 = 128826;
}

// ── `tokenizer.json` deserialization (only the fields we need) ──────────────

/// Top-level `tokenizer.json` shape (subset). Unused sections (`normalizer`,
/// `decoder`, `post_processor`, `padding`, `truncation`, `version`) are ignored
/// — the encode path replicates them in code (no-op normalizer, byte-level
/// decoder, BOS-only post-processor that we do NOT apply, OQ-16 §3-5). The
/// `pre_tokenizer` section IS read: it selects the [`PretokScheme`].
#[derive(Debug, Deserialize)]
struct RawTokenizer {
    #[serde(default)]
    added_tokens: Vec<RawAddedToken>,
    model: RawModel,
    #[serde(default)]
    pre_tokenizer: Option<serde_json::Value>,
}

/// One entry of `.added_tokens` — an added/special token spliced out of the
/// text before BPE.
#[derive(Debug, Deserialize)]
struct RawAddedToken {
    id: u32,
    content: String,
    /// `special:true` for the OCR/role glyphs; `false` for the DeepSeek
    /// `<｜User｜>`-style glyphs. We split on BOTH (HF `AddedVocabulary` splits on
    /// every added token regardless of the `special` flag); the flag is retained
    /// only for the decoder's `skip_special_tokens` behavior.
    #[serde(default)]
    special: bool,
}

/// The `.model` section — a `BPE` with `vocab` + `merges`.
#[derive(Debug, Deserialize)]
struct RawModel {
    vocab: HashMap<String, u32>,
    /// `merges` is either the new array-of-pairs format `[["Ġ","t"], …]` or the
    /// legacy space-joined-string format `["Ġ t", …]`. We accept both.
    #[serde(default)]
    merges: Vec<RawMerge>,
}

/// A single merge rule, tolerant of both `tokenizer.json` merge encodings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawMerge {
    /// New format: `["left", "right"]`.
    Pair([String; 2]),
    /// Legacy format: `"left right"` (single space-separated string).
    Joined(String),
}

impl RawMerge {
    /// Normalize to a `(left, right)` pair. The legacy joined form splits on the
    /// FIRST space only (a byte-level token can itself be `"Ġ"`, never contains
    /// an interior space, so first-space split is unambiguous).
    fn into_pair(self) -> Option<(String, String)> {
        match self {
            RawMerge::Pair([l, r]) => Some((l, r)),
            RawMerge::Joined(s) => {
                let mut it = s.splitn(2, ' ');
                let l = it.next()?.to_string();
                let r = it.next()?.to_string();
                Some((l, r))
            }
        }
    }
}

// ── Pre-tokenizer scheme selection ──────────────────────────────────────────

/// Which pre-tokenizer sequence the loaded `tokenizer.json` declares. The BPE
/// core, added-token splitting, and byte-level decode are shared; only the
/// pre-BPE splitting (and the pinned special-id table) differ per scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PretokScheme {
    /// The DeepSeek/Llama 4-stage `Split` sequence + `ByteLevel(use_regex=false)`
    /// — Baidu Unlimited-OCR ([`pretok::pretokenize`]).
    DeepSeekV2,
    /// `Digits(individual_digits=true)` + `ByteLevel(use_regex=true)` (the
    /// GPT-2 word regex) — SmolLM2 / SmolVLM2 ([`pretok::pretokenize_smollm2`]).
    SmolLm2,
    /// The plain GPT-2 word regex, no Digits stage — OneChart / OPT over
    /// `vocab.json`+`merges.txt` ([`pretok::pretokenize_gpt2`]; selected by
    /// [`Tokenizer::from_opt_files`], never by `pre_tokenizer` detection —
    /// the slow-tokenizer files carry no such declaration).
    Gpt2,
}

/// Classify the `.pre_tokenizer` section into a supported [`PretokScheme`].
///
/// Fail-closed: an unrecognized declaration is a [`FocrError::FormatMismatch`]
/// (silently running the wrong pre-tokenizer would corrupt every downstream
/// id). A MISSING section maps to [`PretokScheme::DeepSeekV2`] — the historical
/// default, kept so the synthetic test fixtures (which carry no
/// `pre_tokenizer`) keep exercising the Baidu path.
fn classify_pretok(v: Option<&serde_json::Value>) -> FocrResult<PretokScheme> {
    let Some(v) = v else {
        return Ok(PretokScheme::DeepSeekV2);
    };
    let stages = v
        .get("pretokenizers")
        .and_then(|p| p.as_array())
        .ok_or_else(|| {
            FocrError::FormatMismatch(format!(
                "unsupported tokenizer.json pre_tokenizer (expected a Sequence): {v}"
            ))
        })?;
    fn ty(s: &serde_json::Value) -> &str {
        s.get("type").and_then(|t| t.as_str()).unwrap_or("")
    }
    // SmolLM2 / GPT-2: [Digits(individual_digits=true), ByteLevel(use_regex=true)].
    if stages.len() == 2
        && ty(&stages[0]) == "Digits"
        && stages[0].get("individual_digits").and_then(|b| b.as_bool()) == Some(true)
        && ty(&stages[1]) == "ByteLevel"
        && stages[1].get("use_regex").and_then(|b| b.as_bool()) == Some(true)
    {
        return Ok(PretokScheme::SmolLm2);
    }
    // DeepSeek-V2: [Split \p{N}{1,3}, Split CJK, Split word-regex,
    // ByteLevel(use_regex=false)] — anchor on the first Split's digit pattern
    // and the ByteLevel tail (the two load-bearing stages).
    if stages.len() == 4
        && ty(&stages[0]) == "Split"
        && stages[0].pointer("/pattern/Regex").and_then(|r| r.as_str()) == Some("\\p{N}{1,3}")
        && ty(&stages[3]) == "ByteLevel"
        && stages[3].get("use_regex").and_then(|b| b.as_bool()) == Some(false)
    {
        return Ok(PretokScheme::DeepSeekV2);
    }
    Err(FocrError::FormatMismatch(format!(
        "unsupported tokenizer.json pre_tokenizer sequence (neither the \
         DeepSeek-V2 4-stage Split nor the SmolLM2 Digits+ByteLevel shape): {v}"
    )))
}

// ── The tokenizer ───────────────────────────────────────────────────────────

/// The byte-level BPE tokenizer, loaded from a `tokenizer.json`.
///
/// Cheap to clone-by-reference (`&Tokenizer` everywhere); the big maps are owned
/// once. Construct with [`Tokenizer::from_file`] (or its alias
/// [`Tokenizer::load`]).
#[derive(Debug)]
pub struct Tokenizer {
    /// base-BPE token string (byte-level alphabet) → id.
    vocab: HashMap<String, u32>,
    /// id → token string, for decode. Covers base vocab AND added tokens.
    id_to_token: HashMap<u32, String>,
    /// merge rank: `(left, right)` → rank (lower = higher priority).
    merge_ranks: HashMap<(String, String), u32>,
    /// Added/special tokens, longest-content-first, for greedy left-to-right
    /// splitting of the input before BPE.
    added: Vec<AddedToken>,
    /// added-token content → id (exact-string match).
    added_by_content: HashMap<String, u32>,
    /// ids that are flagged `special` (for `skip_special_tokens` on decode).
    special_ids: std::collections::HashSet<u32>,
    /// The pre-tokenizer scheme the loaded `tokenizer.json` declared.
    scheme: PretokScheme,
}

/// An added token with its id and `special` flag.
#[derive(Debug, Clone)]
struct AddedToken {
    content: String,
    id: u32,
    #[allow(dead_code)]
    special: bool,
}

impl Tokenizer {
    /// Load the tokenizer from a `tokenizer.json` at `path`.
    ///
    /// The 9.9 MB file is fetched out-of-band and loaded lazily by path (never
    /// embedded). Parses `.model.vocab`, `.model.merges` (both encodings), and
    /// `.added_tokens`, then validates the pinned special-token ids (OQ-16 §6).
    ///
    /// # Errors
    /// * [`FocrError::ModelNotFound`] if the file can't be read.
    /// * [`FocrError::FormatMismatch`] if the JSON can't be parsed or a pinned
    ///   special-token id disagrees with the loaded table.
    pub fn from_file(path: &Path) -> FocrResult<Self> {
        let bytes = std::fs::read(path).map_err(|e| {
            FocrError::ModelNotFound(format!("tokenizer.json at {}: {e}", path.display()))
        })?;
        Self::from_json_bytes(&bytes)
    }

    /// Alias kept for call-site stability (older modules referenced `load`).
    ///
    /// # Errors
    /// See [`Tokenizer::from_file`].
    pub fn load(path: &Path) -> FocrResult<Self> {
        Self::from_file(path)
    }

    /// Build from in-memory `tokenizer.json` bytes (used by [`from_file`] and by
    /// tests with synthetic fixtures — no real 9.9 MB file required).
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] on parse failure or pinned-id disagreement.
    pub fn from_json_bytes(bytes: &[u8]) -> FocrResult<Self> {
        let raw: RawTokenizer = serde_json::from_slice(bytes)
            .map_err(|e| FocrError::FormatMismatch(format!("tokenizer.json parse: {e}")))?;

        let scheme = classify_pretok(raw.pre_tokenizer.as_ref())?;
        let vocab = raw.model.vocab;

        let mut merge_ranks = HashMap::with_capacity(raw.model.merges.len());
        for (rank, m) in raw.model.merges.into_iter().enumerate() {
            if let Some(pair) = m.into_pair() {
                // First occurrence wins (merges are already in priority order);
                // duplicates are ignored to keep the earliest (best) rank.
                merge_ranks.entry(pair).or_insert(rank as u32);
            }
        }

        // Build the id→token map from the base vocab.
        let mut id_to_token: HashMap<u32, String> = HashMap::with_capacity(vocab.len());
        for (tok, &id) in &vocab {
            id_to_token.insert(id, tok.clone());
        }

        // Added tokens: seed the split table + content/id maps + decode strings.
        let mut added = Vec::with_capacity(raw.added_tokens.len());
        let mut added_by_content = HashMap::with_capacity(raw.added_tokens.len());
        let mut special_ids = std::collections::HashSet::new();
        for at in raw.added_tokens {
            // Added tokens decode to their literal content (NOT byte-level
            // remapped) — they are exact UTF-8 strings (OQ-16 §6).
            id_to_token.insert(at.id, at.content.clone());
            added_by_content.insert(at.content.clone(), at.id);
            if at.special {
                special_ids.insert(at.id);
            }
            added.push(AddedToken {
                content: at.content,
                id: at.id,
                special: at.special,
            });
        }
        // Longest content first → greedy split prefers the longest match (HF
        // `AddedVocabulary` is a longest-match trie; sorting by length desc and
        // scanning left-to-right reproduces it for our non-overlapping set).
        added.sort_by(|a, b| b.content.len().cmp(&a.content.len()).then(a.id.cmp(&b.id)));

        let tk = Tokenizer {
            vocab,
            id_to_token,
            merge_ranks,
            added,
            added_by_content,
            special_ids,
            scheme,
        };
        tk.validate_pinned_ids()?;
        Ok(tk)
    }

    /// Load the OneChart / OPT slow-tokenizer file triple (D9): `vocab.json`
    /// (token→id), `merges.txt` (`#version` header + one space-joined merge
    /// per line), and `added_tokens.json` (content→id; all treated as
    /// specials, HF `AddedVocabulary` semantics). Selects
    /// [`PretokScheme::Gpt2`] and validates the census §4/§7 pins.
    ///
    /// # Errors
    /// [`FocrError::ModelNotFound`] for missing files;
    /// [`FocrError::FormatMismatch`] on parse failure or a pin disagreement.
    pub fn from_opt_dir(dir: &Path) -> FocrResult<Self> {
        let read = |name: &str| -> FocrResult<Vec<u8>> {
            std::fs::read(dir.join(name)).map_err(|e| {
                FocrError::ModelNotFound(format!("{} at {}: {e}", name, dir.display()))
            })
        };
        Self::from_opt_files(
            &read("vocab.json")?,
            &read("merges.txt")?,
            &read("added_tokens.json")?,
        )
    }

    /// [`Tokenizer::from_opt_dir`] over in-memory bytes (tests / synthetic
    /// fixtures).
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] on parse failure or pin disagreement.
    pub fn from_opt_files(
        vocab_json: &[u8],
        merges_txt: &[u8],
        added_tokens_json: &[u8],
    ) -> FocrResult<Self> {
        let vocab: HashMap<String, u32> = serde_json::from_slice(vocab_json)
            .map_err(|e| FocrError::FormatMismatch(format!("vocab.json parse: {e}")))?;
        let merges_text = std::str::from_utf8(merges_txt)
            .map_err(|e| FocrError::FormatMismatch(format!("merges.txt not UTF-8: {e}")))?;
        let mut merge_ranks = HashMap::new();
        let mut rank = 0u32;
        for line in merges_text.lines() {
            let line = line.trim_end();
            if line.is_empty() || line.starts_with("#version") {
                continue;
            }
            let Some((l, r)) = line.split_once(' ') else {
                return Err(FocrError::FormatMismatch(format!(
                    "merges.txt line without a space: {line:?}"
                )));
            };
            merge_ranks
                .entry((l.to_string(), r.to_string()))
                .or_insert(rank);
            rank += 1;
        }
        let added_map: HashMap<String, u32> = serde_json::from_slice(added_tokens_json)
            .map_err(|e| FocrError::FormatMismatch(format!("added_tokens.json parse: {e}")))?;

        let mut id_to_token: HashMap<u32, String> = HashMap::with_capacity(vocab.len());
        for (tok, &id) in &vocab {
            id_to_token.insert(id, tok.clone());
        }
        let mut added = Vec::with_capacity(added_map.len());
        let mut added_by_content = HashMap::with_capacity(added_map.len());
        let mut special_ids = std::collections::HashSet::new();
        for (content, id) in added_map {
            id_to_token.insert(id, content.clone());
            added_by_content.insert(content.clone(), id);
            special_ids.insert(id);
            added.push(AddedToken {
                content,
                id,
                special: true,
            });
        }
        // The OPT control tokens live in the BASE vocab but are splittable
        // specials (`special_tokens_map.json`: bos/eos/pad/unk) — the slow
        // `GPT2Tokenizer` resolves a literal `"</s>"` to id 2, never BPE.
        for surface in ["<s>", "<pad>", "</s>", "<unk>"] {
            if let Some(&id) = vocab.get(surface)
                && !added_by_content.contains_key(surface)
            {
                added_by_content.insert(surface.to_string(), id);
                special_ids.insert(id);
                added.push(AddedToken {
                    content: surface.to_string(),
                    id,
                    special: true,
                });
            }
        }
        added.sort_by(|a, b| b.content.len().cmp(&a.content.len()).then(a.id.cmp(&b.id)));

        let tk = Tokenizer {
            vocab,
            id_to_token,
            merge_ranks,
            added,
            added_by_content,
            special_ids,
            scheme: PretokScheme::Gpt2,
        };
        tk.validate_pinned_ids()?;
        Ok(tk)
    }

    /// Cross-check the pinned special-token ids against the loaded added-token
    /// table (per scheme: OQ-16 §6 for Baidu, spec §5 for SmolLM2, the
    /// OneChart census §4/§7 for OPT). A disagreement means the wrong
    /// tokenizer files were supplied and every downstream id would be wrong —
    /// fail loud.
    fn validate_pinned_ids(&self) -> FocrResult<()> {
        let checks: &[(&str, u32)] = match self.scheme {
            PretokScheme::DeepSeekV2 => &[
                ("<image>", special::IMAGE),
                ("<|ref|>", special::REF),
                ("<|/ref|>", special::REF_END),
                ("<|det|>", special::DET),
                ("<|/det|>", special::DET_END),
                ("<|grounding|>", special::GROUNDING),
                ("<|User|>", special::USER),
                ("<|Assistant|>", special::ASSISTANT),
            ],
            PretokScheme::SmolLm2 => &[
                ("<|endoftext|>", special_smollm2::UNK),
                ("<|im_start|>", special_smollm2::BOS),
                ("<|im_end|>", special_smollm2::PAD),
                ("<global-img>", special_smollm2::GLOBAL_IMG),
                (
                    "<fake_token_around_image>",
                    special_smollm2::FAKE_AROUND_IMAGE,
                ),
                ("<image>", special_smollm2::IMAGE),
                ("<end_of_utterance>", special_smollm2::END_OF_UTTERANCE),
            ],
            PretokScheme::Gpt2 => &[
                ("<imgpad>", special_opt::IMG_PAD),
                ("<img>", special_opt::IMG_START),
                ("</img>", special_opt::IMG_END),
                ("<Number>", special_opt::NUMBER),
            ],
        };
        for &(content, want) in checks {
            // Only validate tokens that the supplied file actually declares; a
            // tiny synthetic fixture (tests) need not carry the full OCR table.
            if let Some(&got) = self.added_by_content.get(content)
                && got != want
            {
                return Err(FocrError::FormatMismatch(format!(
                    "tokenizer.json id mismatch for {content}: file says {got}, expected {want}"
                )));
            }
        }
        Ok(())
    }

    /// Encode `text` to token ids, **without** adding any special tokens
    /// (`add_special_tokens=False`, OQ-16 §5). Added/special tokens that appear
    /// literally in `text` are still recognized and emitted as their single id
    /// (e.g. a literal `"<image>"` → one id-128815 token).
    ///
    /// # Errors
    /// Never fails for valid UTF-8 input: byte-level coverage guarantees every
    /// byte maps to a known vocab symbol, so there is no UNK path. Returns
    /// [`FocrError::FormatMismatch`] only if the loaded vocab is missing a
    /// single-byte symbol (a corrupt `tokenizer.json`).
    pub fn encode(&self, text: &str) -> FocrResult<Vec<u32>> {
        let mut ids = Vec::new();
        // Split the text on added/special tokens first (HF AddedVocabulary).
        for segment in self.split_on_added(text) {
            match segment {
                Segment::Added(id) => ids.push(id),
                Segment::Text(s) => self.encode_text_segment(s, &mut ids)?,
            }
        }
        Ok(ids)
    }

    /// BPE-encode a plain text segment (no added tokens inside) and append ids.
    fn encode_text_segment(&self, text: &str, out: &mut Vec<u32>) -> FocrResult<()> {
        let pieces = match self.scheme {
            PretokScheme::DeepSeekV2 => pretok::pretokenize(text),
            PretokScheme::SmolLm2 => pretok::pretokenize_smollm2(text),
            PretokScheme::Gpt2 => pretok::pretokenize_gpt2(text),
        };
        for piece in pieces {
            // `piece` is already byte-level remapped → its chars are vocab
            // symbols. Apply BPE merges, then map merged symbols to ids.
            let symbols = self.bpe(&piece);
            for sym in symbols {
                let id = self.vocab.get(&sym).copied().ok_or_else(|| {
                    FocrError::FormatMismatch(format!(
                        "byte-level symbol {sym:?} missing from vocab (corrupt tokenizer.json)"
                    ))
                })?;
                out.push(id);
            }
        }
        Ok(())
    }

    /// The core BPE merge loop over one pre-tokenized (byte-level) `piece`.
    /// Returns the final list of merged symbol strings. Greedily applies the
    /// lowest-rank (highest-priority) adjacent merge until none apply — the HF
    /// `BPE` word-merge algorithm.
    fn bpe(&self, piece: &str) -> Vec<String> {
        // Start: one symbol per byte-level char.
        let mut symbols: Vec<String> = piece.chars().map(|c| c.to_string()).collect();
        if symbols.len() < 2 {
            return symbols;
        }
        loop {
            // Find the adjacent pair with the lowest merge rank.
            let mut best: Option<(usize, u32)> = None;
            for i in 0..symbols.len() - 1 {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    match best {
                        Some((_, br)) if rank >= br => {}
                        _ => best = Some((i, rank)),
                    }
                }
            }
            let Some((i, _)) = best else { break };
            // Merge symbols[i] and symbols[i+1].
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols.splice(i..=i + 1, std::iter::once(merged));
            if symbols.len() < 2 {
                break;
            }
        }
        symbols
    }

    /// Split `text` into a sequence of added-token ids and plain-text runs,
    /// scanning left-to-right and greedily preferring the longest added-token
    /// content at each position (HF `AddedVocabulary`). The `added` table is
    /// pre-sorted longest-first so the first content that matches at a position
    /// is the longest.
    fn split_on_added<'a>(&self, text: &'a str) -> Vec<Segment<'a>> {
        if self.added.is_empty() {
            return vec![Segment::Text(text)];
        }
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut run_start = 0usize; // start of the current plain-text run
        let mut i = 0usize;
        while i < bytes.len() {
            let mut matched = None;
            for at in &self.added {
                let c = at.content.as_bytes();
                if !c.is_empty() && bytes[i..].starts_with(c) {
                    matched = Some((c.len(), at.id));
                    break; // longest-first ordering → first hit is longest
                }
            }
            if let Some((len, id)) = matched {
                if run_start < i {
                    out.push(Segment::Text(&text[run_start..i]));
                }
                out.push(Segment::Added(id));
                i += len;
                run_start = i;
            } else {
                // advance by one full char (stay on UTF-8 boundaries)
                let ch_len = utf8_char_len(bytes[i]);
                i += ch_len;
            }
        }
        if run_start < text.len() {
            out.push(Segment::Text(&text[run_start..]));
        }
        out
    }

    /// Decode token ids back to a `String` ([SPEC-110]). Base-vocab tokens are
    /// byte-level symbols that are un-mapped back to raw bytes and UTF-8 decoded;
    /// added/special tokens contribute their literal content. By default special
    /// tokens ARE included (the model's own decode keeps structural glyphs); use
    /// [`Tokenizer::decode_skip_special`] to drop them.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if an id is unknown or the reconstructed
    /// bytes are not valid UTF-8 (a truncated multi-byte sequence at the tail is
    /// reported rather than silently lossily replaced).
    pub fn decode(&self, ids: &[u32]) -> FocrResult<String> {
        self.decode_inner(ids, false)
    }

    /// Decode, skipping tokens flagged `special:true` (OQ-16 §6) — the
    /// `skip_special_tokens=True` path used when emitting clean markdown.
    ///
    /// # Errors
    /// See [`Tokenizer::decode`].
    pub fn decode_skip_special(&self, ids: &[u32]) -> FocrResult<String> {
        self.decode_inner(ids, true)
    }

    fn decode_inner(&self, ids: &[u32], skip_special: bool) -> FocrResult<String> {
        // Accumulate raw bytes: base tokens contribute byte-level-decoded bytes,
        // added tokens contribute their literal UTF-8 bytes. We flush to a single
        // UTF-8 string at the end so multi-byte chars split across base tokens
        // reassemble correctly.
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if skip_special && self.special_ids.contains(&id) {
                continue;
            }
            let tok = self.id_to_token.get(&id).ok_or_else(|| {
                FocrError::FormatMismatch(format!("decode: unknown token id {id}"))
            })?;
            if self.added_by_content.contains_key(tok) {
                // Added token → literal content bytes (NOT byte-level remapped).
                bytes.extend_from_slice(tok.as_bytes());
            } else {
                // Base token → each char is a byte-level symbol; invert the map.
                for c in tok.chars() {
                    let b = pretok::char_to_byte(c).ok_or_else(|| {
                        FocrError::FormatMismatch(format!(
                            "decode: token id {id} has non-byte-level symbol {c:?}"
                        ))
                    })?;
                    bytes.push(b);
                }
            }
        }
        String::from_utf8(bytes)
            .map_err(|e| FocrError::FormatMismatch(format!("decode: invalid UTF-8: {e}")))
    }

    /// The id of an added/special token by its literal content, if present.
    pub fn token_to_id(&self, content: &str) -> Option<u32> {
        self.added_by_content
            .get(content)
            .copied()
            .or_else(|| self.vocab.get(content).copied())
    }

    /// The literal string for an id (added content or byte-level symbol), if
    /// present. (Byte-level symbols are returned un-decoded — for human display
    /// prefer [`Tokenizer::decode`].)
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// Number of base-vocab entries (excludes added tokens). The embedding /
    /// LM-head shape is the model's `vocab_size` (129280), NOT this — see
    /// OQ-16 §7.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// The pre-tokenizer scheme the loaded `tokenizer.json` declared.
    pub fn scheme(&self) -> PretokScheme {
        self.scheme
    }

    /// BOS id — Baidu `<｜begin▁of▁sentence｜>` (0, OQ-16 §5) or SmolLM2
    /// `<|im_start|>` (1). Prompt builders own their own framing; nothing is
    /// auto-prepended.
    pub fn bos_id(&self) -> u32 {
        match self.scheme {
            PretokScheme::DeepSeekV2 => special::BOS,
            PretokScheme::SmolLm2 => special_smollm2::BOS,
            PretokScheme::Gpt2 => special_opt::BOS_EOS,
        }
    }
    /// EOS / generation-stop id — Baidu `<｜end▁of▁sentence｜>` (1) or SmolVLM2
    /// `<end_of_utterance>` (49279). Never auto-appended.
    pub fn eos_id(&self) -> u32 {
        match self.scheme {
            PretokScheme::DeepSeekV2 => special::EOS,
            PretokScheme::SmolLm2 => special_smollm2::END_OF_UTTERANCE,
            PretokScheme::Gpt2 => special_opt::BOS_EOS,
        }
    }
    /// PAD id — Baidu `<｜▁pad▁｜>` (2) or SmolLM2 `<|im_end|>` (2).
    pub fn pad_id(&self) -> u32 {
        match self.scheme {
            PretokScheme::DeepSeekV2 => special::PAD,
            PretokScheme::SmolLm2 => special_smollm2::PAD,
            PretokScheme::Gpt2 => special_opt::PAD,
        }
    }
    /// `<image>` splice-slot id — Baidu 128815 ([SPEC-035]) or SmolVLM2 49190
    /// (64 slots per 512² frame, spec §5).
    pub fn image_id(&self) -> u32 {
        match self.scheme {
            PretokScheme::DeepSeekV2 => special::IMAGE,
            PretokScheme::SmolLm2 => special_smollm2::IMAGE,
            PretokScheme::Gpt2 => special_opt::IMG_PAD,
        }
    }
}

/// A run produced by [`Tokenizer::split_on_added`].
enum Segment<'a> {
    /// A plain-text run to BPE-encode.
    Text(&'a str),
    /// An added/special token already resolved to its id.
    Added(u32),
}

/// Length in bytes of a UTF-8 char from its lead byte (1..=4). Used to advance
/// the added-token scanner on char boundaries without re-decoding. Shared with
/// the sibling [`tiktoken`] module's special-token scanner.
pub(super) fn utf8_char_len(lead: u8) -> usize {
    if lead < 0x80 {
        1
    } else if lead >> 5 == 0b110 {
        2
    } else if lead >> 4 == 0b1110 {
        3
    } else if lead >> 3 == 0b11110 {
        4
    } else {
        1 // continuation/invalid byte — advance one to make progress
    }
}

/// The exact Python snippet that regenerates [`unicode_tables`]. Kept here (not
/// runnable Rust) so a maintainer can reproduce the tables bit-for-bit and a
/// batch-verify can reconcile the UCD version (see [`unicode_tables::UCD_VERSION`]).
///
/// ```text
/// import unicodedata
/// for cat in 'LMNPS':
///     ranges, start, prev = [], None, None
///     for cp in range(0x110000):
///         hit = unicodedata.category(chr(cp))[0] == cat
///         if hit:
///             if start is None: start = prev = cp
///             elif cp == prev + 1: prev = cp
///             else: ranges.append((start, prev)); start = prev = cp
///         elif start is not None:
///             ranges.append((start, prev)); start = prev = None
///     if start is not None: ranges.append((start, prev))
///     # emit `pub static {CAT}: &[(u32,u32)] = &[ (lo,hi), … ];`
/// ```
#[allow(dead_code)]
const UNICODE_TABLE_REGEN: () = ();

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny synthetic `tokenizer.json` (no real 9.9 MB file). The base vocab
    /// covers the byte-level symbols for the lowercase letters used in tests
    /// plus the space glyph `Ġ`, and a couple of merges so we can prove merge
    /// ordering. Added tokens carry the pinned `<image>` id to exercise the
    /// validation + splitting paths. `pub(super)` so the [`super::ops`]
    /// trait-level tests run BOTH impls over the same fixtures (no drift).
    pub(super) fn tiny_json() -> String {
        // Byte-level: 'a'..'z' map to themselves; space → "Ġ".
        // vocab ids are arbitrary (we only test id-EXACTNESS of merge ordering
        // and round-trip, not against HF here).
        // Symbols present: a b c d e f Ġ ab abc Ġa  (merge "a"+"b"->"ab",
        // "ab"+"c"->"abc"; lower rank = applied first).
        r#"{
          "version": "1.0",
          "added_tokens": [
            {"id": 128815, "content": "<image>", "special": true},
            {"id": 100,    "content": "<|x|>",   "special": false}
          ],
          "normalizer": {"type":"Sequence","normalizers":[]},
          "model": {
            "type": "BPE",
            "vocab": {
              "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5,
              "Ġ": 6, "ab": 7, "abc": 8, "Ġa": 9, "Ġd": 10
            },
            "merges": [
              ["a", "b"],
              ["ab", "c"],
              ["Ġ", "a"],
              ["Ġ", "d"]
            ]
          }
        }"#
        .to_string()
    }

    fn tk() -> Tokenizer {
        Tokenizer::from_json_bytes(tiny_json().as_bytes()).expect("tiny tokenizer loads")
    }

    #[test]
    fn loads_and_validates_pinned_ids() {
        let t = tk();
        assert_eq!(t.token_to_id("<image>"), Some(special::IMAGE));
        assert_eq!(t.image_id(), 128815);
        assert_eq!(t.bos_id(), 0);
        assert_eq!(t.eos_id(), 1);
        assert_eq!(t.vocab_size(), 11);
    }

    #[test]
    fn pinned_id_mismatch_is_format_error() {
        // Same fixture but with the wrong <image> id → must reject.
        let bad = tiny_json().replace("\"id\": 128815", "\"id\": 999");
        let err = Tokenizer::from_json_bytes(bad.as_bytes()).unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)), "got {err:?}");
    }

    #[test]
    fn bpe_merge_ordering() {
        let t = tk();
        // "abc" → merges: "a"+"b"->"ab" (rank0), "ab"+"c"->"abc" (rank1).
        // Final single symbol "abc" = id 8.
        assert_eq!(t.encode("abc").unwrap(), vec![8]);
        // "ab" stops at "ab" = id 7.
        assert_eq!(t.encode("ab").unwrap(), vec![7]);
        // "ba" has no merge → ['b','a'] = [1,0].
        assert_eq!(t.encode("ba").unwrap(), vec![1, 0]);
    }

    #[test]
    fn merge_priority_is_rank_not_position() {
        let t = tk();
        // " a" pretokenizes to "Ġa" (leading space + letter). Merge "Ġ"+"a"
        // (rank 2) → "Ġa" = id 9.
        assert_eq!(t.encode(" a").unwrap(), vec![9]);
    }

    #[test]
    fn special_token_splitting() {
        let t = tk();
        // "ab<image>c" → BPE("ab")=[7], <image>=128815, BPE("c")=[2].
        assert_eq!(t.encode("ab<image>c").unwrap(), vec![7, 128815, 2]);
        // A non-special added token still splits.
        assert_eq!(t.encode("a<|x|>b").unwrap(), vec![0, 100, 1]);
        // Longest-match: ensure "<image>" wins as a whole (no partial BPE of '<').
        let ids = t.encode("<image>").unwrap();
        assert_eq!(ids, vec![128815]);
    }

    #[test]
    fn round_trip_encode_decode() {
        let t = tk();
        // Base-token round trip.
        let ids = t.encode("abc").unwrap();
        assert_eq!(t.decode(&ids).unwrap(), "abc");
        // With a space (byte-level Ġ must invert back to ' ').
        let ids2 = t.encode(" a").unwrap();
        assert_eq!(t.decode(&ids2).unwrap(), " a");
        // Mixed with an added token: decode includes the literal content.
        let ids3 = t.encode("ab<image>c").unwrap();
        assert_eq!(t.decode(&ids3).unwrap(), "ab<image>c");
    }

    #[test]
    fn decode_skip_special_drops_specials() {
        let t = tk();
        let ids = t.encode("ab<image>c").unwrap();
        // <image> is special:true → dropped; <|x|> would be kept (special:false).
        assert_eq!(t.decode_skip_special(&ids).unwrap(), "abc");
        let ids2 = t.encode("a<|x|>b").unwrap();
        assert_eq!(t.decode_skip_special(&ids2).unwrap(), "a<|x|>b");
    }

    #[test]
    fn byte_level_non_ascii_round_trips_through_bytes() {
        // 'é' = U+00E9 → bytes C3 A9 → byte-level chars 'Ã','©'. Neither is in
        // our tiny vocab, so encode would error — but decode of a synthetic
        // byte-split must reassemble. Use a vocab that has the two symbols.
        let json = r#"{
          "added_tokens": [],
          "model": {"type":"BPE",
            "vocab": {"Ã": 0, "©": 1},
            "merges": []
          }
        }"#;
        let t = Tokenizer::from_json_bytes(json.as_bytes()).unwrap();
        let ids = t.encode("é").unwrap(); // two byte-level symbols
        assert_eq!(ids, vec![0, 1]);
        // Decode reassembles the two bytes into the original 'é'.
        assert_eq!(t.decode(&ids).unwrap(), "é");
    }

    #[test]
    fn legacy_joined_merge_format_is_accepted() {
        // Old `tokenizer.json` encodes merges as space-joined strings.
        let json = r#"{
          "added_tokens": [],
          "model": {"type":"BPE",
            "vocab": {"a":0,"b":1,"ab":2},
            "merges": ["a b"]
          }
        }"#;
        let t = Tokenizer::from_json_bytes(json.as_bytes()).unwrap();
        assert_eq!(t.encode("ab").unwrap(), vec![2]);
    }

    #[test]
    fn decode_unknown_id_errors() {
        let t = tk();
        let err = t.decode(&[424242]).unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)), "got {err:?}");
    }

    #[test]
    fn from_file_missing_is_model_not_found() {
        let err = Tokenizer::from_file(Path::new("/nonexistent/tokenizer.json")).unwrap_err();
        assert!(matches!(err, FocrError::ModelNotFound(_)), "got {err:?}");
    }

    #[test]
    fn empty_input_encodes_empty() {
        let t = tk();
        assert_eq!(t.encode("").unwrap(), Vec::<u32>::new());
        assert_eq!(t.decode(&[]).unwrap(), "");
    }

    // ── real-vocab conformance (env-gated on the ~9.9 MB tokenizer.json) ─────
    // FOCR_TOKENIZER_JSON points at the pinned file; else the truth-pack
    // snapshot path (gitignored, fetched out-of-band by scripts/fetch_sources.sh);
    // absent ⇒ skip (the model-gated pattern, matching tiktoken.rs).

    // (`pub(super)` so the [`super::ops`] trait tests reuse the same gate.)
    pub(super) fn load_real() -> Option<Tokenizer> {
        let path = std::env::var("FOCR_TOKENIZER_JSON").unwrap_or_else(|_| {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/docs/truth-pack/snapshots/tokenizer.json"
            )
            .to_string()
        });
        let path = Path::new(&path);
        if !path.is_file() {
            eprintln!(
                "SKIP baidu tokenizer conformance: {} absent \
                 (scripts/fetch_sources.sh or FOCR_TOKENIZER_JSON)",
                path.display()
            );
            return None;
        }
        Some(Tokenizer::from_file(path).expect("pinned tokenizer.json must load"))
    }

    #[test]
    fn real_vocab_anchors() {
        let Some(t) = load_real() else {
            return;
        };
        assert_eq!(t.vocab_size(), 128000);
        // The runtime-hardcoded specials must each encode to their single id.
        assert_eq!(
            t.encode("<｜begin▁of▁sentence｜>").unwrap(),
            vec![special::BOS]
        );
        assert_eq!(
            t.encode("<｜end▁of▁sentence｜>").unwrap(),
            vec![special::EOS]
        );
        assert_eq!(t.encode("<｜▁pad▁｜>").unwrap(), vec![special::PAD]);
        assert_eq!(t.encode("<image>").unwrap(), vec![special::IMAGE]);
        assert_eq!(t.encode("<|grounding|>").unwrap(), vec![special::GROUNDING]);
        // ASCII-pipe role glyphs are distinct from the fullwidth-bar DeepSeek
        // glyphs (the glyph-vs-ASCII distinction is load-bearing, OQ-16 §6).
        assert_eq!(t.encode("<|User|>").unwrap(), vec![special::USER]);
        assert_eq!(t.encode("<|Assistant|>").unwrap(), vec![special::ASSISTANT]);
        assert_ne!(t.encode("<｜User｜>").unwrap(), vec![special::USER]);
    }

    /// **L0/L4 — the Baidu tokenizer token-id-EXACT conformance gate (OQ-16,
    /// bd-re8.8).** Parses the committed golden fixtures — generated by the
    /// reference HF `tokenizers` engine (the exact Rust crate
    /// `LlamaTokenizerFast` wraps) over the pinned `tokenizer.json` via
    /// `scripts/gen_token_id_fixtures.py` — and asserts our encoder reproduces
    /// every id stream AND our decoder every decoded string exactly. No
    /// decoder/vision parity bead may close while this is red (AGENTS.md
    /// doctrine).
    #[test]
    fn baidu_token_id_conformance_gate() {
        let Some(t) = load_real() else {
            return;
        };
        const EXPECTED: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tokenizer_baidu/expected.json"
        ));
        let v: serde_json::Value = serde_json::from_str(EXPECTED).unwrap();
        // The fixture must match the tokenizer.json it was generated from —
        // ids from a different serialization are not comparable.
        assert_eq!(
            v["_meta"]["tokenizer_json_sha256"].as_str().unwrap(),
            "a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4",
            "fixture was generated against a different tokenizer.json pin"
        );
        let cases = v["fixtures"].as_array().expect("fixtures array");
        let num_cases = v["_meta"]["num_cases"].as_u64().unwrap() as usize;
        assert_eq!(cases.len(), num_cases, "fixture _meta.num_cases drift");
        assert!(
            cases.len() >= 100,
            "conformance corpus must stay >= 100 cases"
        );
        let mut mismatches = 0usize;
        for rec in cases {
            let text = rec["text"].as_str().unwrap();
            let want: Vec<u32> = rec["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            let got = t.encode(text).unwrap();
            if got != want {
                let pos = got
                    .iter()
                    .zip(&want)
                    .position(|(a, b)| a != b)
                    .unwrap_or_else(|| got.len().min(want.len()));
                eprintln!(
                    "ENC MISMATCH {{\"case\": {text:?}, \"len\": {}, \"mismatch_pos\": {pos}}}\n  \
                     got  {got:?}\n  want {want:?}",
                    want.len()
                );
                mismatches += 1;
            }
            // Decode direction: EXACTLY the reference ids back to the reference
            // string (skip_special_tokens=false — literal round-trip).
            let want_decoded = rec["decoded"].as_str().unwrap();
            let got_decoded = t.decode(&want).unwrap();
            if got_decoded != want_decoded {
                eprintln!(
                    "DEC MISMATCH {{\"case\": {text:?}, \"mismatch_pos\": \"none\"}}\n  \
                     got  {got_decoded:?}\n  want {want_decoded:?}"
                );
                mismatches += 1;
            }
        }
        assert_eq!(
            mismatches, 0,
            "tok_id_mismatch_count must be 0 (got {mismatches})"
        );
    }

    // ── SmolLM2 scheme (C6, bd-3jo6.3.6) ─────────────────────────────────────

    /// A tiny synthetic SmolLM2-style `tokenizer.json`: the SmolLM2
    /// `pre_tokenizer` declaration (which selects [`PretokScheme::SmolLm2`]),
    /// a vocab covering letters/digits/space symbols, and added tokens
    /// carrying the pinned `<image>`/`<end_of_utterance>` ids.
    fn tiny_smollm2_json() -> String {
        r#"{
          "version": "1.0",
          "added_tokens": [
            {"id": 49190, "content": "<image>", "special": true},
            {"id": 49279, "content": "<end_of_utterance>", "special": true}
          ],
          "normalizer": null,
          "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
              {"type": "Digits", "individual_digits": true},
              {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}
            ]
          },
          "post_processor": null,
          "model": {
            "type": "BPE",
            "vocab": {
              "a": 0, "b": 1, "c": 2, "1": 3, "2": 4,
              "Ġ": 5, "ab": 6, "12": 7, "'": 8, "s": 9, "'s": 10
            },
            "merges": [
              ["a", "b"],
              ["1", "2"],
              ["'", "s"]
            ]
          }
        }"#
        .to_string()
    }

    fn tk_smollm2() -> Tokenizer {
        Tokenizer::from_json_bytes(tiny_smollm2_json().as_bytes()).expect("tiny smollm2 loads")
    }

    #[test]
    fn smollm2_scheme_is_detected_from_pre_tokenizer() {
        let t = tk_smollm2();
        assert_eq!(t.scheme(), PretokScheme::SmolLm2);
        // The synthetic Baidu fixture (no pre_tokenizer) stays DeepSeek-V2.
        assert_eq!(tk().scheme(), PretokScheme::DeepSeekV2);
    }

    #[test]
    fn smollm2_scheme_ids() {
        let t = tk_smollm2();
        assert_eq!(t.bos_id(), special_smollm2::BOS); // <|im_start|> = 1
        assert_eq!(t.eos_id(), special_smollm2::END_OF_UTTERANCE); // 49279
        assert_eq!(t.pad_id(), special_smollm2::PAD); // <|im_end|> = 2
        assert_eq!(t.image_id(), special_smollm2::IMAGE); // 49190
        // The DeepSeek scheme keeps the Baidu ids.
        let b = tk();
        assert_eq!(b.bos_id(), special::BOS);
        assert_eq!(b.eos_id(), special::EOS);
        assert_eq!(b.image_id(), special::IMAGE);
    }

    #[test]
    fn smollm2_digits_encode_individually() {
        let t = tk_smollm2();
        // The Digits stage isolates each digit into its own piece, so the
        // "1"+"2"→"12" merge can never apply across the piece boundary.
        assert_eq!(t.encode("12").unwrap(), vec![3, 4]);
        // Under the DeepSeek scheme the same vocab WOULD merge "12" (digit
        // groups of ≤3 stay one piece) — proving the scheme dispatch is live.
        // (Also rename `<image>`: the DeepSeek pin table requires 128815.)
        let baidu_style = tiny_smollm2_json()
            .replace(
                r#""pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
              {"type": "Digits", "individual_digits": true},
              {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}
            ]
          },"#,
                "",
            )
            .replace("<image>", "<img2>");
        assert!(
            !baidu_style.contains("pre_tokenizer"),
            "test fixture drifted: the pre_tokenizer block was not stripped"
        );
        let b = Tokenizer::from_json_bytes(baidu_style.as_bytes()).expect("loads");
        assert_eq!(b.scheme(), PretokScheme::DeepSeekV2);
        assert_eq!(b.encode("12").unwrap(), vec![7]);
    }

    #[test]
    fn smollm2_contraction_merges() {
        let t = tk_smollm2();
        // "ab's": alt 2 "ab", alt 1 "'s" → merge "'"+"s" applies inside the
        // "'s" piece.
        assert_eq!(t.encode("ab's").unwrap(), vec![6, 10]);
    }

    #[test]
    fn smollm2_added_tokens_split() {
        let t = tk_smollm2();
        assert_eq!(
            t.encode("ab<image>ab").unwrap(),
            vec![6, special_smollm2::IMAGE, 6]
        );
        assert_eq!(
            t.decode_skip_special(&t.encode("ab<image>ab").unwrap())
                .unwrap(),
            "abab"
        );
    }

    #[test]
    fn smollm2_wrong_image_pin_is_rejected() {
        let bad = tiny_smollm2_json().replace("\"id\": 49190", "\"id\": 999");
        let err = Tokenizer::from_json_bytes(bad.as_bytes()).unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)), "got {err:?}");
    }

    #[test]
    fn unknown_pre_tokenizer_is_rejected() {
        // A Whitespace pre-tokenizer is neither supported shape → fail closed.
        let bad = tiny_smollm2_json().replace(
            r#"{"type": "Digits", "individual_digits": true}"#,
            r#"{"type": "Whitespace"}"#,
        );
        let err = Tokenizer::from_json_bytes(bad.as_bytes()).unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)), "got {err:?}");
        // So is a non-Sequence declaration.
        let bad2 = tiny_smollm2_json().replace(
            r#"{
            "type": "Sequence",
            "pretokenizers": [
              {"type": "Digits", "individual_digits": true},
              {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}
            ]
          }"#,
            r#"{"type": "ByteLevel", "add_prefix_space": true, "use_regex": true}"#,
        );
        let err2 = Tokenizer::from_json_bytes(bad2.as_bytes()).unwrap_err();
        assert!(matches!(err2, FocrError::FormatMismatch(_)), "got {err2:?}");
    }

    // ── real-vocab conformance (env-gated on the SmolVLM2 tokenizer.json) ────
    // FOCR_SMOLVLM2_TOKENIZER_JSON points at the pinned file, else
    // $FOCR_SMOLVLM2_DIR/tokenizer.json (the zoo layout); absent ⇒ skip (the
    // model-gated pattern, matching load_real / FOCR_GOT_TIKTOKEN).

    pub(super) fn load_real_smolvlm2() -> Option<Tokenizer> {
        let path = std::env::var("FOCR_SMOLVLM2_TOKENIZER_JSON")
            .ok()
            .or_else(|| {
                std::env::var("FOCR_SMOLVLM2_DIR")
                    .ok()
                    .map(|d| format!("{d}/tokenizer.json"))
            });
        let Some(path) = path else {
            eprintln!(
                "SKIP smolvlm2 tokenizer conformance: set FOCR_SMOLVLM2_TOKENIZER_JSON \
                 (or FOCR_SMOLVLM2_DIR) to the pinned tokenizer.json"
            );
            return None;
        };
        let path = Path::new(&path);
        if !path.is_file() {
            eprintln!(
                "SKIP smolvlm2 tokenizer conformance: {} absent",
                path.display()
            );
            return None;
        }
        Some(Tokenizer::from_file(path).expect("pinned smolvlm2 tokenizer.json must load"))
    }

    #[test]
    fn smolvlm2_real_vocab_anchors() {
        let Some(t) = load_real_smolvlm2() else {
            return;
        };
        assert_eq!(t.scheme(), PretokScheme::SmolLm2);
        assert_eq!(t.vocab_size(), 49152);
        assert_eq!(t.encode("<|endoftext|>").unwrap(), vec![0]);
        assert_eq!(t.encode("<|im_start|>").unwrap(), vec![1]);
        assert_eq!(t.encode("<|im_end|>").unwrap(), vec![2]);
        assert_eq!(
            t.encode("<global-img>").unwrap(),
            vec![special_smollm2::GLOBAL_IMG]
        );
        assert_eq!(
            t.encode("<fake_token_around_image>").unwrap(),
            vec![special_smollm2::FAKE_AROUND_IMAGE]
        );
        assert_eq!(t.encode("<image>").unwrap(), vec![special_smollm2::IMAGE]);
        assert_eq!(
            t.encode("<end_of_utterance>").unwrap(),
            vec![special_smollm2::END_OF_UTTERANCE]
        );
        // The 6×6 row-marker grid brackets (spec §5).
        assert_eq!(t.encode("<row_1_col_1>").unwrap(), vec![49153]);
        assert_eq!(t.encode("<row_6_col_6>").unwrap(), vec![49188]);
    }

    /// **L0a — the SmolVLM2 tokenizer token-id-EXACT conformance gate (C6,
    /// bd-3jo6.3.6, spec §13).** Parses the committed golden fixtures —
    /// generated by the reference HF `tokenizers` engine over the pinned
    /// SmolVLM2 `tokenizer.json` via
    /// `scripts/gen_smolvlm2_token_id_fixtures.py` — and asserts our encoder
    /// reproduces every id stream AND our decoder every decoded string
    /// exactly. No SmolVLM2 vision/decoder parity bead may close while this is
    /// red (AGENTS.md doctrine).
    #[test]
    fn smolvlm2_token_id_conformance_gate() {
        let Some(t) = load_real_smolvlm2() else {
            return;
        };
        const EXPECTED: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tokenizer_smolvlm2/expected.json"
        ));
        let v: serde_json::Value = serde_json::from_str(EXPECTED).unwrap();
        assert_eq!(
            v["_meta"]["tokenizer_json_sha256"].as_str().unwrap(),
            "5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a",
            "fixture was generated against a different tokenizer.json pin"
        );
        let cases = v["fixtures"].as_array().expect("fixtures array");
        let num_cases = v["_meta"]["num_cases"].as_u64().unwrap() as usize;
        assert_eq!(cases.len(), num_cases, "fixture _meta.num_cases drift");
        assert!(
            cases.len() >= 100,
            "conformance corpus must stay >= 100 cases"
        );
        let mut mismatches = 0usize;
        for rec in cases {
            let text = rec["text"].as_str().unwrap();
            let want: Vec<u32> = rec["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            let got = t.encode(text).unwrap();
            if got != want {
                let pos = got
                    .iter()
                    .zip(&want)
                    .position(|(a, b)| a != b)
                    .unwrap_or_else(|| got.len().min(want.len()));
                eprintln!(
                    "ENC MISMATCH {{\"case\": {text:?}, \"len\": {}, \"mismatch_pos\": {pos}}}\n  \
                     got  {got:?}\n  want {want:?}",
                    want.len()
                );
                mismatches += 1;
            }
            let want_decoded = rec["decoded"].as_str().unwrap();
            let got_decoded = t.decode(&want).unwrap();
            if got_decoded != want_decoded {
                eprintln!(
                    "DEC MISMATCH {{\"case\": {text:?}}}\n  \
                     got  {got_decoded:?}\n  want {want_decoded:?}"
                );
                mismatches += 1;
            }
        }
        assert_eq!(
            mismatches, 0,
            "tok_id_mismatch_count must be 0 (got {mismatches})"
        );
    }

    // ── OneChart / OPT scheme (D9, bd-3jo6.4.9) ──────────────────────────────

    #[test]
    fn opt_files_load_and_scheme_dispatch() {
        // Tiny synthetic triple: vocab covers letters + Ġ, merges drive the
        // BPE loop, added tokens carry the census pins.
        let vocab = r#"{"a":0,"b":1,"c":2,"1":3,"2":4,"Ġ":5,"ab":6,"12":7}"#.as_bytes();
        let merges = b"#version: 0.2\na b\n1 2\n";
        let added = br#"{"<imgpad>":50265,"<img>":50266,"</img>":50267,"<Number>":50268}"#;
        let t = Tokenizer::from_opt_files(vocab, merges, added).expect("opt files load");
        assert_eq!(t.scheme(), PretokScheme::Gpt2);
        // GPT-2 scheme has NO Digits stage: "12" stays one piece and merges.
        assert_eq!(t.encode("12").unwrap(), vec![7]);
        assert_eq!(t.encode("ab").unwrap(), vec![6]);
        // Added tokens split + resolve; skip_special drops them.
        assert_eq!(
            t.encode("ab<imgpad>ab").unwrap(),
            vec![6, special_opt::IMG_PAD, 6]
        );
        assert_eq!(
            t.decode_skip_special(&t.encode("ab<Number>").unwrap())
                .unwrap(),
            "ab"
        );
        // Scheme-aware ids (census §4).
        assert_eq!(t.bos_id(), special_opt::BOS_EOS);
        assert_eq!(t.eos_id(), special_opt::BOS_EOS);
        assert_eq!(t.pad_id(), special_opt::PAD);
        assert_eq!(t.image_id(), special_opt::IMG_PAD);
        // A wrong pin fails loud.
        let bad = br#"{"<imgpad>":999}"#;
        let err = Tokenizer::from_opt_files(vocab, merges, bad).unwrap_err();
        assert!(matches!(err, FocrError::FormatMismatch(_)), "got {err:?}");
        // Missing dir → ModelNotFound.
        let err = Tokenizer::from_opt_dir(Path::new("/nonexistent")).unwrap_err();
        assert!(matches!(err, FocrError::ModelNotFound(_)), "got {err:?}");
    }

    /// **D9 — the OneChart tokenizer token-id-EXACT conformance gate**
    /// (skip-with-SUCCESS without `FOCR_ONECHART_DIR`): our OPT loader +
    /// GPT-2 scheme must reproduce every id stream of the committed golden
    /// fixtures (generated by the slow `transformers` `GPT2Tokenizer` over
    /// the pinned `vocab.json`+`merges.txt`+`added_tokens.json`).
    #[test]
    fn onechart_token_id_conformance_gate() {
        let Ok(dir) = std::env::var("FOCR_ONECHART_DIR") else {
            eprintln!("SKIP onechart tokenizer conformance: set FOCR_ONECHART_DIR");
            return;
        };
        let t = Tokenizer::from_opt_dir(Path::new(&dir)).expect("onechart tokenizer loads");
        assert_eq!(t.vocab_size(), 50_265);
        const EXPECTED: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tokenizer_onechart/expected.json"
        ));
        let v: serde_json::Value = serde_json::from_str(EXPECTED).unwrap();
        let cases = v["fixtures"].as_object().expect("fixtures object");
        assert!(cases.len() >= 25, "conformance corpus shrank");
        let mut mismatches = 0usize;
        for (text, want) in cases {
            let want: Vec<u32> = want
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            let got = t.encode(text).unwrap();
            if got != want {
                eprintln!("ENC MISMATCH {text:?}\n  got  {got:?}\n  want {want:?}");
                mismatches += 1;
            }
        }
        eprintln!("[D9] {} cases, {mismatches} mismatches", cases.len());
        assert_eq!(mismatches, 0, "tok_id_mismatch_count must be 0");
    }
}
