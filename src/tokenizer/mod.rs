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

mod pretok;
pub mod tiktoken;
mod unicode_tables;

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
/// decoder, BOS-only post-processor that we do NOT apply, OQ-16 §3-5).
#[derive(Debug, Deserialize)]
struct RawTokenizer {
    #[serde(default)]
    added_tokens: Vec<RawAddedToken>,
    model: RawModel,
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
        };
        tk.validate_pinned_ids()?;
        Ok(tk)
    }

    /// Cross-check the pinned special-token ids (OQ-16 §6) against the loaded
    /// added-token table. A disagreement means the wrong `tokenizer.json` was
    /// supplied and every downstream id would be wrong — fail loud.
    fn validate_pinned_ids(&self) -> FocrResult<()> {
        let checks: &[(&str, u32)] = &[
            ("<image>", special::IMAGE),
            ("<|ref|>", special::REF),
            ("<|/ref|>", special::REF_END),
            ("<|det|>", special::DET),
            ("<|/det|>", special::DET_END),
            ("<|grounding|>", special::GROUNDING),
            ("<|User|>", special::USER),
            ("<|Assistant|>", special::ASSISTANT),
        ];
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
        for piece in pretok::pretokenize(text) {
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

    /// BOS id (0) — the prompt builder prepends exactly one (OQ-16 §5).
    pub fn bos_id(&self) -> u32 {
        special::BOS
    }
    /// EOS id (1) — the generation stop token; never auto-appended (OQ-16 §5).
    pub fn eos_id(&self) -> u32 {
        special::EOS
    }
    /// PAD id (2).
    pub fn pad_id(&self) -> u32 {
        special::PAD
    }
    /// `<image>` id (128815) — the prompt builder splits on the literal
    /// `"<image>"` and the vision-prefix bead expands the run ([SPEC-035]).
    pub fn image_id(&self) -> u32 {
        special::IMAGE
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
    /// validation + splitting paths.
    fn tiny_json() -> String {
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
}
