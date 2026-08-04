//! Raw-byte tiktoken BPE tokenizer for the Qwen `qwen.tiktoken` vocabulary
//! (GOT-OCR2.0 decoder, beads B6/A6). Token-id-EXACT against the upstream
//! `QWenTokenizer` (`tokenization_qwen.py`) — the L0 prerequisite gate for every
//! GOT downstream parity rung (AGENTS.md doctrine: tokenizer id-exactness is a
//! precondition for all decoder/vision gates).
//!
//! Unlike the sibling merge-list BPE in [`super`] (HF `tokenizer.json`, GPT-2
//! byte→unicode-REMAPPED string symbols, an explicit `merges` priority table),
//! this engine is **raw-byte tiktoken**:
//!   * symbols are `&[u8]` slices of the piece — there is NO byte→unicode remap;
//!   * there is NO `merges` list — the merge priority of an adjacent pair is the
//!     rank of its *concatenation* in `ranks`, and that rank IS the final token id;
//!   * the pre-tokenizer is the Qwen cl100k-family pattern (single `\p{N}`,
//!     `(?i:…)` contractions, a `\s+(?!\S)` lookahead), hand-rolled in the same
//!     leftmost-first style as [`super::pretok`] because no regex engine is a
//!     dependency (Cargo.toml is owned centrally; `pretok.rs` exists for exactly
//!     this reason).
//!
//! Special tokens are built in code (the upstream `added_tokens_decoder` is
//! empty) and split out of the text BEFORE pre-tokenization (tiktoken core
//! semantics, `allowed_special="all"`).
//!
//! ## Known divergences (resolve before the GOT e2e gate, not before this one)
//! * **NFC** — `QWenTokenizer.tokenize` runs `unicodedata.normalize("NFC")` first.
//!   No NFC crate is a dependency, so this is deferred (`// TODO(NFC)` at the
//!   `encode` head). Every committed conformance fixture is NFC-stable, so the
//!   token-id gate is exact; arbitrary not-yet-normalized input is not.
//! * **decode is lossy** — a single token can be a partial-UTF-8 fragment (e.g.
//!   id 11162 = `b" \xf0\x9f"`), so `decode` accumulates bytes then
//!   `from_utf8_lossy`s, unlike the strict per-token `from_utf8` the merge-list
//!   BPE can use.

use std::collections::{HashMap, HashSet};

use crate::error::{FocrError, FocrResult};

// Reuse the existing UCD range tables + binary-search membership, and the UTF-8
// lead-byte width, VERBATIM (no duplication): `unicode_tables` + `pretok::in_ranges`
// + `utf8_char_len` are all reachable from this child module.
use super::pretok::in_ranges;
use super::unicode_tables as ucd;
use super::utf8_char_len;

// ── Special-token ids (source-verified vs tokenization_qwen.py) ──────────────
// The QWen `IMAGE_ST` block is enumerated contiguously from `len(mergeable_ranks)`
// (= 151643): the 3 control tokens, then 205 `<|extra_i|>`, then the 6 GOT
// grounding tags, then the 3 image-splice tokens — every id 0..=151859 is
// assigned, there is NO padding gap.

/// `<|endoftext|>` = bos = eos = pad (config.json / generation_config.json).
pub const ENDOFTEXT: u32 = 151_643;
/// `<|im_start|>`.
pub const IM_START: u32 = 151_644;
/// `<|im_end|>` — the GOT generation stop string.
pub const IM_END: u32 = 151_645;
/// First `<|extra_0|>`; the extras run `151646..=151850` (205 of them).
pub const EXTRA_BASE: u32 = 151_646;
/// Number of `<|extra_i|>` reserved tokens.
pub const NUM_EXTRAS: u32 = 205;
/// GOT grounding tag `<ref>` (the 6 grounding tags are NOT padding).
pub const REF: u32 = 151_851;
/// GOT grounding tag `</ref>`.
pub const REF_END: u32 = 151_852;
/// GOT grounding tag `<box>`.
pub const BOX: u32 = 151_853;
/// GOT grounding tag `</box>`.
pub const BOX_END: u32 = 151_854;
/// GOT grounding tag `<quad>`.
pub const QUAD: u32 = 151_855;
/// GOT grounding tag `</quad>`.
pub const QUAD_END: u32 = 151_856;
/// GOT image-splice open `<img>`.
pub const IMG_START: u32 = 151_857;
/// GOT image-splice close `</img>`.
pub const IMG_END: u32 = 151_858;
/// GOT per-patch image token `<imgpad>` (one per projected vision feature row).
pub const IMG_PAD: u32 = 151_859;
/// Total vocabulary: 151643 base ranks + 217 specials.
pub const N_VOCAB: usize = 151_860;
/// Number of base (mergeable) ranks parsed from `qwen.tiktoken`.
const N_BASE: usize = 151_643;
/// Number of special tokens layered on top of the base ranks.
const N_SPECIAL: usize = 217;

/// The Qwen raw-byte tiktoken tokenizer.
#[derive(Debug)]
pub struct Tiktoken {
    /// token bytes → rank (== id). [`N_BASE`] entries, ranks dense `0..=151642`.
    ranks: HashMap<Vec<u8>, u32>,
    /// id → bytes for decode. Length [`N_VOCAB`]; base ids hold their raw bytes,
    /// special ids hold the UTF-8 of their surface form.
    rev: Vec<Vec<u8>>,
    /// special surface string → id.
    special_by_content: HashMap<String, u32>,
    /// ids flagged special (for `skip_special_tokens` on decode).
    special_ids: HashSet<u32>,
    /// specials sorted LONGEST-content-first so a left-to-right scan greedily
    /// takes the longest match (same shape as `super::Tokenizer`'s added-token
    /// splitter).
    specials_sorted: Vec<(String, u32)>,
    bos: u32,
    eos: u32,
    pad: u32,
}

impl Tiktoken {
    /// Build the GOT tokenizer from the bytes of `qwen.tiktoken`. The special
    /// table is constructed in code (the upstream file's `added_tokens_decoder`
    /// is empty), NOT read from the file.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if the file is not exactly [`N_BASE`] dense
    /// ranks `0..=151642` with all 256 single bytes present (fail-closed: the
    /// tokenizer is a doctrine prerequisite gate, so a malformed vocab must be a
    /// loud load error, never a silently-degraded encoder).
    pub fn from_qwen_tiktoken(file: &[u8]) -> FocrResult<Self> {
        // ── parse base ranks ────────────────────────────────────────────────
        let mut ranks: HashMap<Vec<u8>, u32> = HashMap::with_capacity(N_BASE);
        let mut max_rank: i64 = -1;
        for (lineno, line) in file.split(|&b| b == b'\n').enumerate() {
            if line.is_empty() {
                continue;
            }
            // each line: `base64(token_bytes) SP rank_decimal`
            let sp = line.iter().position(|&b| b == b' ').ok_or_else(|| {
                FocrError::FormatMismatch(format!(
                    "qwen.tiktoken line {lineno}: no space separator"
                ))
            })?;
            let tok = b64_decode(&line[..sp]).ok_or_else(|| {
                FocrError::FormatMismatch(format!(
                    "qwen.tiktoken line {lineno}: invalid base64 token"
                ))
            })?;
            let rank_str = std::str::from_utf8(&line[sp + 1..]).map_err(|_| {
                FocrError::FormatMismatch(format!("qwen.tiktoken line {lineno}: rank not UTF-8"))
            })?;
            let rank: u32 = rank_str.trim().parse().map_err(|_| {
                FocrError::FormatMismatch(format!("qwen.tiktoken line {lineno}: rank not a u32"))
            })?;
            max_rank = max_rank.max(i64::from(rank));
            ranks.insert(tok, rank);
        }

        // ── validate (fail-closed) ──────────────────────────────────────────
        if ranks.len() != N_BASE || max_rank != (N_BASE as i64 - 1) {
            return Err(FocrError::FormatMismatch(format!(
                "qwen.tiktoken: expected {N_BASE} dense ranks 0..={}, got {} entries (max rank {max_rank})",
                N_BASE - 1,
                ranks.len()
            )));
        }
        for b in 0u8..=255 {
            if !ranks.contains_key(std::slice::from_ref(&b)) {
                return Err(FocrError::FormatMismatch(format!(
                    "qwen.tiktoken: single byte 0x{b:02x} missing (no byte fallback possible)"
                )));
            }
        }

        // ── special table (source-verified vs tokenization_qwen.py) ─────────
        let mut specials: Vec<(String, u32)> = Vec::with_capacity(N_SPECIAL);
        specials.push(("<|endoftext|>".to_string(), ENDOFTEXT));
        specials.push(("<|im_start|>".to_string(), IM_START));
        specials.push(("<|im_end|>".to_string(), IM_END));
        for i in 0..NUM_EXTRAS {
            specials.push((format!("<|extra_{i}|>"), EXTRA_BASE + i)); // 151646..=151850
        }
        for (s, id) in [
            ("<ref>", REF),
            ("</ref>", REF_END),
            ("<box>", BOX),
            ("</box>", BOX_END),
            ("<quad>", QUAD),
            ("</quad>", QUAD_END),
            ("<img>", IMG_START),
            ("</img>", IMG_END),
            ("<imgpad>", IMG_PAD),
        ] {
            specials.push((s.to_string(), id));
        }
        debug_assert_eq!(specials.len(), N_SPECIAL);

        // ── rev table + special maps ────────────────────────────────────────
        let mut rev: Vec<Vec<u8>> = vec![Vec::new(); N_VOCAB];
        for (tok, &rank) in &ranks {
            rev[rank as usize] = tok.clone();
        }
        let mut special_by_content = HashMap::with_capacity(N_SPECIAL);
        let mut special_ids = HashSet::with_capacity(N_SPECIAL);
        for (content, id) in &specials {
            rev[*id as usize] = content.as_bytes().to_vec();
            special_by_content.insert(content.clone(), *id);
            special_ids.insert(*id);
        }
        let mut specials_sorted = specials;
        // Longest-content-first → greedy longest-match on a left-to-right scan;
        // id as a stable tiebreak keeps the build deterministic.
        specials_sorted.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));

        Ok(Self {
            ranks,
            rev,
            special_by_content,
            special_ids,
            specials_sorted,
            bos: ENDOFTEXT,
            eos: ENDOFTEXT,
            pad: ENDOFTEXT,
        })
    }

    // ── encode ───────────────────────────────────────────────────────────────

    /// Encode `text` with **all specials enabled** (`allowed_special="all"`),
    /// matching `QWenTokenizer.tokenize`'s default. A literal special surface in
    /// `text` (`<|im_end|>`, `<img>`, …) becomes its control id. This is the
    /// surface the rest of the engine calls and what the GOT prompt builder (B7)
    /// relies on.
    ///
    /// NFC normalization is NOT yet applied (no crate); see the module note.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] only on a corrupt vocab (a single byte
    /// missing) — impossible after the load-time validation.
    pub fn encode(&self, text: &str) -> FocrResult<Vec<u32>> {
        // let text = &nfc(text);  // TODO(NFC): apply unicodedata.normalize("NFC")
        let mut out = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0usize;
        let mut run_start = 0usize;
        while i < bytes.len() {
            let mut matched = None;
            for (content, id) in &self.specials_sorted {
                let c = content.as_bytes();
                if bytes[i..].starts_with(c) {
                    matched = Some((c.len(), *id));
                    break; // longest-first ordering → the first hit is the longest
                }
            }
            if let Some((len, id)) = matched {
                if run_start < i {
                    self.encode_ordinary_str(&text[run_start..i], &mut out)?;
                }
                out.push(id);
                i += len;
                run_start = i;
            } else {
                i += utf8_char_len(bytes[i]);
            }
        }
        if run_start < text.len() {
            self.encode_ordinary_str(&text[run_start..], &mut out)?;
        }
        Ok(out)
    }

    /// Encode with NO special handling — special-looking substrings are treated
    /// as literal bytes (tiktoken `encode_ordinary`). Used for arbitrary/user
    /// text round-trips where `<|...|>` should be data, not a control token.
    ///
    /// # Errors
    /// See [`Tiktoken::encode`].
    pub fn encode_ordinary(&self, text: &str) -> FocrResult<Vec<u32>> {
        let mut out = Vec::new();
        self.encode_ordinary_str(text, &mut out)?;
        Ok(out)
    }

    /// Pre-tokenize one special-free segment (Qwen cl100k pattern) and BPE each
    /// piece into its token ids, appending to `out`.
    fn encode_ordinary_str(&self, text: &str, out: &mut Vec<u32>) -> FocrResult<()> {
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0usize;
        while i < chars.len() {
            // The Qwen cl100k pattern partitions the WHOLE string (every char is
            // covered by some alternative), so `match_piece` returns `Some` for
            // any non-empty input; `unwrap_or(1)` is a defensive consume-one-char
            // fallback that must never fire in practice.
            let len = match_piece(&chars, i).unwrap_or(1);
            let piece: String = chars[i..i + len].iter().collect();
            self.bpe_bytes(piece.as_bytes(), out)?;
            i += len;
        }
        Ok(())
    }

    /// Raw-byte byte-pair-merge over one pre-token piece, appending its ids.
    ///
    /// Repeatedly merges the adjacent segment pair whose concatenation has the
    /// lowest rank (leftmost on a tie), until no adjacent pair is in `ranks` — the
    /// tiktoken core. O(n²) in the (short) piece length; correctness-first per the
    /// doctrine, optimizable behind the parity gate later.
    fn bpe_bytes(&self, piece: &[u8], out: &mut Vec<u32>) -> FocrResult<()> {
        // whole-piece fast path
        if let Some(&r) = self.ranks.get(piece) {
            out.push(r);
            return Ok(());
        }
        // one segment per byte; each segment is a half-open `[start,end)` range.
        let mut parts: Vec<(usize, usize)> = (0..piece.len()).map(|k| (k, k + 1)).collect();
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..parts.len().saturating_sub(1) {
                let a = parts[i].0;
                let c = parts[i + 1].1;
                if let Some(&r) = self.ranks.get(&piece[a..c]) {
                    // `r >= br` keeps the LEFTMOST lowest-rank pair (a strictly
                    // lower rank replaces) — identical to tiktoken's `<`.
                    match best {
                        Some((_, br)) if r >= br => {}
                        _ => best = Some((i, r)),
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let a = parts[i].0;
            let c = parts[i + 1].1;
            parts[i] = (a, c);
            parts.remove(i + 1);
        }
        for (a, c) in parts {
            let r = self.ranks.get(&piece[a..c]).copied().ok_or_else(|| {
                FocrError::FormatMismatch("tiktoken: a final BPE segment was not in ranks".into())
            })?; // unreachable post-validation: every single byte is a rank
            out.push(r);
        }
        Ok(())
    }

    // ── decode ───────────────────────────────────────────────────────────────

    /// Decode ids → String. Bytes are accumulated then UTF-8-decoded **lossily**
    /// (one Unicode scalar is frequently split across several ids). Special ids
    /// yield their surface form. Specials INCLUDED by default.
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if any id is `>= N_VOCAB`.
    pub fn decode(&self, ids: &[u32]) -> FocrResult<String> {
        self.decode_inner(ids, false)
    }

    /// Decode, dropping special ids (`skip_special_tokens=True`).
    ///
    /// # Errors
    /// [`FocrError::FormatMismatch`] if any non-special id is `>= N_VOCAB`.
    pub fn decode_skip_special(&self, ids: &[u32]) -> FocrResult<String> {
        self.decode_inner(ids, true)
    }

    fn decode_inner(&self, ids: &[u32], skip_special: bool) -> FocrResult<String> {
        let mut buf: Vec<u8> = Vec::new();
        for &id in ids {
            if skip_special && self.special_ids.contains(&id) {
                continue;
            }
            let bytes = self.rev.get(id as usize).ok_or_else(|| {
                FocrError::FormatMismatch(format!("decode: token id {id} out of range"))
            })?;
            buf.extend_from_slice(bytes);
        }
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    // ── accessors (surface mirrors super::Tokenizer) ─────────────────────────

    /// Id of a token by its exact surface string (special tokens first, then a
    /// whole base token), or `None`.
    #[must_use]
    pub fn token_to_id(&self, content: &str) -> Option<u32> {
        self.special_by_content
            .get(content)
            .copied()
            .or_else(|| self.ranks.get(content.as_bytes()).copied())
    }

    /// Surface string for an id (special form, or the UTF-8 of the base bytes if
    /// valid). Prefer [`Tiktoken::decode`] for human display of an id *stream*.
    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        let b = self.rev.get(id as usize)?;
        if b.is_empty() {
            return None;
        }
        Some(String::from_utf8_lossy(b).into_owned())
    }

    /// Total vocabulary size ([`N_VOCAB`]).
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        N_VOCAB
    }

    /// Beginning-of-sequence id (`<|endoftext|>`, 151643).
    #[must_use]
    pub fn bos_id(&self) -> u32 {
        self.bos
    }

    /// End-of-sequence id (`<|endoftext|>`, 151643).
    #[must_use]
    pub fn eos_id(&self) -> u32 {
        self.eos
    }

    /// Padding id (`<|endoftext|>`, 151643).
    #[must_use]
    pub fn pad_id(&self) -> u32 {
        self.pad
    }

    /// The GOT per-patch image token `<imgpad>` (151859) — the prompt slot a
    /// projected vision feature row overwrites (B3 splice connector).
    #[must_use]
    pub fn image_pad_id(&self) -> u32 {
        IMG_PAD
    }
}

// ── hand-rolled Qwen cl100k pre-tokenizer ────────────────────────────────────
// Qwen PAT_STR (byte-for-byte):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Leftmost-first alternation (PCRE / tiktoken semantics). The pattern covers
// EVERY character, so this is a full partition (no gaps) — unlike
// `super::pretok::split_gpt_word`. The three whitespace alternatives (5/6/7) are
// the same logic as the proven `super::pretok` whitespace handling.

#[inline]
fn is_l(c: char) -> bool {
    in_ranges(c as u32, ucd::LETTER)
}
#[inline]
fn is_n(c: char) -> bool {
    in_ranges(c as u32, ucd::NUMBER)
}
#[inline]
fn is_ws(c: char) -> bool {
    c.is_whitespace() // == \p{White_Space}, matches super::pretok's is_ws
}

/// Length (in chars) of the first matching alternative of the Qwen pattern at
/// `chars[i..]`, or `None` (unreachable for non-empty input — the pattern is a
/// full partition).
fn match_piece(chars: &[char], i: usize) -> Option<usize> {
    let n = chars.len();

    // Alt 1: (?i:'s|'t|'re|'ve|'m|'ll|'d) — ASCII apostrophe + case-insensitive
    // suffix. Order is leftmost-first; no suffix is a prefix of another, so there
    // is no ambiguity. The apostrophe is the literal U+0027 only.
    if chars[i] == '\'' {
        const SUFFIXES: [&[u8]; 7] = [b"s", b"t", b"re", b"ve", b"m", b"ll", b"d"];
        for suf in SUFFIXES {
            if i + 1 + suf.len() <= n
                && suf.iter().enumerate().all(|(k, &b)| {
                    chars[i + 1 + k].is_ascii() && (chars[i + 1 + k] as u8).eq_ignore_ascii_case(&b)
                })
            {
                return Some(1 + suf.len());
            }
        }
        // no contraction → fall through (the apostrophe is claimed by alt 4)
    }

    // Alt 2: [^\r\n\p{L}\p{N}]? \p{L}+  — an optional single non-CR/LF/letter/
    // number lead, then ≥1 letters (letters only, NO \p{M}).
    {
        let c0 = chars[i];
        let mut j = i;
        if c0 != '\r'
            && c0 != '\n'
            && !is_l(c0)
            && !is_n(c0)
            && chars.get(i + 1).is_some_and(|&c| is_l(c))
        {
            j = i + 1; // consume the optional lead, only when a letter follows
        }
        let run_start = j;
        while j < n && is_l(chars[j]) {
            j += 1;
        }
        if j > run_start {
            return Some(j - i);
        }
    }

    // Alt 3: \p{N}  — a SINGLE number char (the digit-split canary).
    if is_n(chars[i]) {
        return Some(1);
    }

    // Alt 4:  ?[^\s\p{L}\p{N}]+[\r\n]*  — an optional ONE leading space, then ≥1
    // non-ws/non-letter/non-number chars, then any CR/LF.
    {
        let mut j = i;
        if chars[i] == ' ' {
            j += 1;
        }
        let run_start = j;
        while j < n {
            let c = chars[j];
            if !is_ws(c) && !is_l(c) && !is_n(c) {
                j += 1;
            } else {
                break;
            }
        }
        if j > run_start {
            while j < n && (chars[j] == '\r' || chars[j] == '\n') {
                j += 1;
            }
            return Some(j - i);
        }
        // the optional space did not lead to a run → do NOT consume it here;
        // alts 5/6/7 claim the whitespace.
    }

    // Alt 5: \s*[\r\n]+  — a whitespace run containing ≥1 CR/LF, ending right
    // after the LAST CR/LF.
    {
        let mut last_crlf_end = None;
        let mut k = i;
        while k < n && is_ws(chars[k]) {
            if chars[k] == '\r' || chars[k] == '\n' {
                last_crlf_end = Some(k + 1);
            }
            k += 1;
        }
        if let Some(end) = last_crlf_end {
            return Some(end - i);
        }
    }

    // Alt 6: \s+(?!\S)  — a greedy whitespace run minus its final char iff that
    // char is followed by a non-space.
    {
        let mut j = i;
        while j < n && is_ws(chars[j]) {
            j += 1;
        }
        let w = j - i;
        if w >= 1 {
            if j == n {
                return Some(w);
            } else if w >= 2 {
                return Some(w - 1);
            }
        }
    }

    // Alt 7: \s+  — the remaining single whitespace char.
    {
        let mut j = i;
        while j < n && is_ws(chars[j]) {
            j += 1;
        }
        if j > i {
            return Some(j - i);
        }
    }

    None // unreachable for the Qwen pattern (full partition)
}

/// Minimal standard base64 decoder (alphabet `A–Za–z0–9+/`, `=` padding).
/// Hand-rolled because no `base64` crate is a dependency. Returns `None` on any
/// invalid input. `forbid(unsafe_code)`-clean.
fn b64_decode(input: &[u8]) -> Option<Vec<u8>> {
    #[inline]
    fn val(b: u8) -> Option<u8> {
        match b {
            b'A'..=b'Z' => Some(b - b'A'),
            b'a'..=b'z' => Some(b - b'a' + 26),
            b'0'..=b'9' => Some(b - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut s = input;
    while s.last() == Some(&b'=') {
        s = &s[..s.len() - 1];
    }
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc: u32 = 0;
    let mut nbits: u32 = 0;
    for &b in s {
        let v = u32::from(val(b)?);
        acc = (acc << 6) | v;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((acc >> nbits) as u8);
        }
    }
    // Any leftover bits must be zero padding (well-formed base64).
    if nbits > 0 && (acc & ((1 << nbits) - 1)) != 0 {
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── base64 (pure, always run) ────────────────────────────────────────────

    #[test]
    fn b64_decodes_known_vectors() {
        assert_eq!(b64_decode(b"SGVsbG8=").unwrap(), b"Hello".to_vec());
        assert_eq!(b64_decode(b"IQ==").unwrap(), vec![0x21]); // '!'  (qwen rank 0)
        assert_eq!(b64_decode(b"Ig==").unwrap(), vec![0x22]); // '"'  (qwen rank 1)
        assert_eq!(b64_decode(b"").unwrap(), Vec::<u8>::new());
        // '+' and '/' alphabet members round-trip (>>> 0xFB,0xEF,0xFF == "++//").
        assert_eq!(b64_decode(b"++//").unwrap(), vec![0xfb, 0xef, 0xff]);
        // non-alphabet byte rejected.
        assert!(b64_decode(b"****").is_none());
    }

    // ── pre-tokenizer partition (no vocab needed) ────────────────────────────

    fn pieces(text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let len = match_piece(&chars, i).unwrap();
            out.push(chars[i..i + len].iter().collect());
            i += len;
        }
        out
    }

    #[test]
    fn pretokenizer_partitions_match_qwen_pattern() {
        // every char is consumed exactly once (full partition).
        for s in [
            "Hello, world!",
            "    leading and  multiple   spaces",
            "Line1\nLine2\n",
            "café 1234 ²x",
            "fn main() {}",
        ] {
            assert_eq!(
                pieces(s).concat(),
                s,
                "partition must be lossless for {s:?}"
            );
        }
        // single \p{N}: each digit is its own piece (the load-bearing canary).
        assert_eq!(pieces("1234"), vec!["1", "2", "3", "4"]);
        // a leading space attaches to the following word (cl100k).
        assert_eq!(pieces(" world"), vec![" world"]);
        // a contraction is one piece, case-insensitively.
        assert_eq!(pieces("It's"), vec!["It", "'s"]);
        assert_eq!(pieces("IT'S"), vec!["IT", "'S"]);
    }

    // ── real-vocab tests (env-gated on the ~2.4 MB qwen.tiktoken) ─────────────
    // FOCR_GOT_TIKTOKEN points at the file; absent ⇒ skip (model-gated pattern).

    fn load_real() -> Option<Tiktoken> {
        let p = std::env::var("FOCR_GOT_TIKTOKEN").ok()?;
        let bytes = std::fs::read(p).ok()?;
        Some(Tiktoken::from_qwen_tiktoken(&bytes).expect("real qwen.tiktoken must parse"))
    }

    #[test]
    fn loads_and_validates_real_vocab() {
        let Some(tk) = load_real() else {
            return;
        };
        assert_eq!(tk.vocab_size(), N_VOCAB);
        assert_eq!(tk.eos_id(), ENDOFTEXT);
        assert_eq!(tk.image_pad_id(), IMG_PAD);
    }

    #[test]
    fn digit_split_canary() {
        let Some(tk) = load_real() else {
            return;
        };
        // single \p{N} pre-token ⇒ ten single-digit ids (1..9 then 0).
        assert_eq!(
            tk.encode_ordinary("1234567890").unwrap(),
            vec![16, 17, 18, 19, 20, 21, 22, 23, 24, 15]
        );
    }

    #[test]
    fn special_vs_ordinary_split() {
        let Some(tk) = load_real() else {
            return;
        };
        // allowed_special="all": the literal surface becomes the control id.
        assert_eq!(
            tk.encode("say <|endoftext|> now").unwrap(),
            vec![36790, 220, 151643, 1431]
        );
        // encode_ordinary: the same surface is literal bytes.
        assert_eq!(
            tk.encode_ordinary("say <|endoftext|> now").unwrap(),
            vec![36790, 82639, 8691, 723, 427, 91, 29, 1431]
        );
    }

    #[test]
    fn decode_specials_and_grounding_ids() {
        let Some(tk) = load_real() else {
            return;
        };
        assert_eq!(
            tk.decode(&[151643, 151857, 151859, 151858]).unwrap(),
            "<|endoftext|><img><imgpad></img>"
        );
        assert_eq!(tk.token_to_id("<ref>"), Some(REF));
        assert_eq!(tk.token_to_id("<quad>"), Some(QUAD));
        assert_eq!(tk.token_to_id("<imgpad>"), Some(IMG_PAD));
    }

    #[test]
    fn id_bytes_spotcheck_and_lossy_decode() {
        let Some(tk) = load_real() else {
            return;
        };
        assert_eq!(tk.decode(&[9707]).unwrap(), "Hello");
        assert_eq!(tk.decode(&[1879]).unwrap(), " world");
        assert_eq!(tk.decode(&[15]).unwrap(), "0");
        assert_eq!(tk.decode(&[108386]).unwrap(), "你好");
        // a partial-UTF-8 token decodes lossily (the first half of an emoji).
        assert_eq!(tk.decode(&[11162]).unwrap(), " \u{fffd}");
    }

    /// **L0a — the GOT tokenizer token-id-EXACT conformance gate.** Parses the
    /// committed golden fixtures (generated by the upstream QWenTokenizer recipe
    /// via tiktoken) and asserts our encoder reproduces every id stream exactly.
    /// No decoder/vision bead may close while this is red (AGENTS.md doctrine).
    #[test]
    fn token_id_conformance_gate() {
        let Some(tk) = load_real() else {
            return;
        };
        const EXPECTED: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tokenizer_got/expected.json"
        ));
        let v: serde_json::Value = serde_json::from_str(EXPECTED).unwrap();
        let cases = v["fixtures"].as_object().expect("fixtures object");
        let mut mismatches = 0usize;
        for (text, ids) in cases {
            let want: Vec<u32> = ids
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            let got = tk.encode(text).unwrap();
            if got != want {
                eprintln!("MISMATCH {text:?}\n  got  {got:?}\n  want {want:?}");
                mismatches += 1;
            }
            // none of the 24 corpus strings contains a special surface that
            // changes between modes EXCEPT the explicit <|...|>/<img>... cases,
            // which the conformance ids already encode under allowed_special=all.
        }
        assert_eq!(
            mismatches, 0,
            "tok_id_mismatch_count must be 0 (got {mismatches})"
        );
        assert_eq!(cases.len(), 24, "expected the 24 golden cases");
    }

    /// **L0c — the GOT prompt-id gate, cross-validated against the torch oracle.**
    /// The committed fixture is the EXACT plain-OCR prompt string GOT builds
    /// (system + MPT conv + `<img><imgpad>×256</img>` splice) and the 287 ids the
    /// upstream `GOTQwenForCausalLM`'s own tokenizer produced for it
    /// (`scripts/gen_reference_fixtures_got.py`). Our encoder must reproduce them
    /// exactly — proving the tokenizer handles the real prompt (256 `<imgpad>`
    /// specials + role markers), not just the L0a corpus.
    #[test]
    fn prompt_id_oracle_cross_check() {
        let Some(tk) = load_real() else {
            return;
        };
        const L0C: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/got/l0c_prompt.json"
        ));
        let v: serde_json::Value = serde_json::from_str(L0C).unwrap();
        let prompt = v["prompt"].as_str().unwrap();
        let want: Vec<u32> = v["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        let got = tk.encode(prompt).unwrap();
        assert_eq!(got.len(), 287, "GOT plain-OCR prompt is 287 ids");
        assert_eq!(
            got, want,
            "Rust tiktoken must match the torch-oracle GOT prompt ids exactly"
        );
        // the 256 <imgpad> splice slots are contiguous in the stream.
        assert_eq!(
            got.iter().filter(|&&id| id == IMG_PAD).count(),
            256,
            "256 <imgpad> image slots"
        );
    }

    #[test]
    fn round_trip_byte_reconstructable_subset() {
        let Some(tk) = load_real() else {
            return;
        };
        // ASCII + whole-CJK fixtures reconstruct exactly (emoji partial-byte
        // cases are covered by the lossy spot-check instead).
        for s in [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "snake_case camelCase PascalCase kebab-case",
            "你好，世界！这是一个测试。",
            "https://example.com/path?q=1&r=2#frag-2",
        ] {
            let ids = tk.encode(s).unwrap();
            assert_eq!(tk.decode(&ids).unwrap(), s, "round-trip for {s:?}");
        }
    }
}
