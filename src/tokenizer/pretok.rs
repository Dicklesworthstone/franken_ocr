//! The byte-level pre-tokenizer `Sequence` (OQ-16 §3) — the four ordered stages
//! the HF `LlamaTokenizerFast` applies before BPE, hand-implemented because the
//! Unicode-property regex engine (`onig`/`fancy-regex`) the `tokenizers` crate
//! uses is not a dependency here and we may not add one (Cargo.toml is owned
//! centrally). Stages, verbatim from `tokenizer.json .pre_tokenizer`:
//!
//! ```text
//! 1. Split  \p{N}{1,3}                         Isolated  (digit groups of 1-3)
//! 2. Split  [一-龥぀-ゟ゠-ヿ]+                   Isolated  (CJK/Hiragana/Katakana runs)
//! 3. Split  <GPT-style word regex>             Isolated
//! 4. ByteLevel add_prefix_space=false trim_offsets=true use_regex=false
//! ```
//!
//! `Isolated` behavior means: the regex matches carve the input into pieces; the
//! matched spans become tokens AND the gaps between/around them are kept as
//! their own pieces (nothing is dropped — pre-tokenization only *splits*). Each
//! stage runs over every piece produced by the previous stage.
//!
//! The matcher honours **leftmost-first** alternation (PCRE / Oniguruma
//! semantics, the `tokenizers` default), not leftmost-longest: at each position
//! the alternatives are tried in source order and the first that matches wins.
//! The GPT-2 word regex is authored so its first matching alternative is also
//! the intended one (OQ-16 §3).

use super::unicode_tables as ucd;

/// Binary-search membership in a sorted, non-overlapping `[lo, hi]` range table.
///
/// Used by the `\p{…}` general-category predicates. `O(log n)` over the range
/// starts; the tables are generated and guaranteed sorted (UCD
/// [`ucd::UCD_VERSION`]).
pub fn in_ranges(cp: u32, ranges: &[(u32, u32)]) -> bool {
    // Find the last range whose `lo <= cp`, then check `cp <= hi`.
    match ranges.binary_search_by(|&(lo, _)| lo.cmp(&cp)) {
        Ok(_) => true, // cp is itself a range start
        Err(0) => false,
        Err(idx) => {
            let (_, hi) = ranges[idx - 1];
            cp <= hi
        }
    }
}

/// `\p{L}` — Unicode general category Letter.
#[inline]
fn is_l(c: char) -> bool {
    in_ranges(c as u32, ucd::LETTER)
}
/// `\p{M}` — Mark.
#[inline]
fn is_m(c: char) -> bool {
    in_ranges(c as u32, ucd::MARK)
}
/// `\p{N}` — Number.
#[inline]
fn is_n(c: char) -> bool {
    in_ranges(c as u32, ucd::NUMBER)
}
/// `\p{P}` — Punctuation.
#[inline]
fn is_p(c: char) -> bool {
    in_ranges(c as u32, ucd::PUNCTUATION)
}
/// `\p{S}` — Symbol.
#[inline]
fn is_s(c: char) -> bool {
    in_ranges(c as u32, ucd::SYMBOL)
}

/// `\s` — the regex whitespace class. The `tokenizers` regex engine uses the
/// Unicode-aware `\s`, which is `\p{White_Space}`. `char::is_whitespace` is
/// exactly `White_Space` in Rust's UCD, so it matches.
#[inline]
fn is_ws(c: char) -> bool {
    c.is_whitespace()
}

/// The ASCII-punctuation leading class of alternative 1 (the literal set inside
/// `[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]`).
#[inline]
fn is_ascii_punct_lead(c: char) -> bool {
    matches!(
        c,
        '!' | '"'
            | '#'
            | '$'
            | '%'
            | '&'
            | '\''
            | '('
            | ')'
            | '*'
            | '+'
            | ','
            | '-'
            | '.'
            | '/'
            | ':'
            | ';'
            | '<'
            | '='
            | '>'
            | '?'
            | '@'
            | '['
            | '\\'
            | ']'
            | '^'
            | '_'
            | '`'
            | '{'
            | '|'
            | '}'
            | '~'
    )
}

/// Stage-1 / stage-2 explicit ranges. Stage-2 isolates runs of CJK Unified
/// Ideographs `一`(U+4E00)–`龥`(U+9FA5), Hiragana `぀`(U+3040)–`ゟ`(U+309F), and
/// Katakana `゠`(U+30A0)–`ヿ`(U+30FF).
#[inline]
fn is_cjk_kana(c: char) -> bool {
    let cp = c as u32;
    (0x4E00..=0x9FA5).contains(&cp)
        || (0x3040..=0x309F).contains(&cp)
        || (0x30A0..=0x30FF).contains(&cp)
}

/// Pre-tokenize `text` into the byte-level pieces fed to BPE.
///
/// Returns each piece already mapped through the GPT-2 byte→unicode alphabet
/// (the ByteLevel stage), i.e. ready to be looked up as keys in `model.vocab`.
pub fn pretokenize(text: &str) -> Vec<String> {
    // Stage 1: split digit groups of 1-3.
    let mut pieces = vec![text.to_string()];
    pieces = split_stage(&pieces, split_digit_groups);
    // Stage 2: isolate CJK / Kana runs.
    pieces = split_stage(&pieces, split_cjk_kana);
    // Stage 3: the GPT-style word regex.
    pieces = split_stage(&pieces, split_gpt_word);
    // Stage 4: ByteLevel remap (use_regex=false → no further splitting here).
    pieces.iter().map(|p| byte_level_map(p)).collect()
}

/// Run one split stage over every input piece, concatenating the results.
fn split_stage(pieces: &[String], f: fn(&str) -> Vec<String>) -> Vec<String> {
    let mut out = Vec::with_capacity(pieces.len());
    for p in pieces {
        out.extend(f(p));
    }
    out
}

/// Stage 1: `Split \p{N}{1,3}` Isolated — isolate maximal-but-≤3 runs of Number
/// characters. A run of `k` Number chars becomes `ceil(k/3)` pieces of size 3,
/// 3, …, then the remainder (the `{1,3}` quantifier is greedy, leftmost-first).
fn split_digit_groups(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut i = 0;
    while i < chars.len() {
        if is_n(chars[i]) {
            // flush any pending non-number gap
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            // greedily take up to 3 Number chars as one isolated token
            let mut grp = String::new();
            let mut taken = 0;
            while i < chars.len() && taken < 3 && is_n(chars[i]) {
                grp.push(chars[i]);
                i += 1;
                taken += 1;
            }
            out.push(grp);
        } else {
            buf.push(chars[i]);
            i += 1;
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

/// Stage 2: `Split [一-龥぀-ゟ゠-ヿ]+` Isolated — isolate maximal runs of CJK/Kana.
fn split_cjk_kana(s: &str) -> Vec<String> {
    isolate_runs(s, is_cjk_kana)
}

/// Generic "isolate maximal runs where `pred` holds" splitter (a `[…]+`
/// Isolated stage). Gaps where `pred` is false are preserved as their own
/// pieces.
fn isolate_runs(s: &str, pred: fn(char) -> bool) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut in_run = false;
    for c in s.chars() {
        let hit = pred(c);
        if hit != in_run {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            in_run = hit;
        }
        buf.push(c);
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

/// Stage 3: the GPT-style word regex (Isolated). We scan left-to-right; at each
/// position we try the six alternatives in order and take the first non-empty
/// match (leftmost-first). Because every position is covered by alternative 2/3
/// or falls through as a single-char gap, and the alternatives never match the
/// empty string in a way that advances zero chars, the scan always progresses.
fn split_gpt_word(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut gap = String::new();
    let mut i = 0;
    while i < chars.len() {
        if let Some(len) = match_word(&chars, i) {
            // flush any pending unmatched gap (Isolated keeps gaps)
            if !gap.is_empty() {
                out.push(std::mem::take(&mut gap));
            }
            let tok: String = chars[i..i + len].iter().collect();
            out.push(tok);
            i += len;
        } else {
            gap.push(chars[i]);
            i += 1;
        }
    }
    if !gap.is_empty() {
        out.push(gap);
    }
    out
}

/// Try the six ordered alternatives of the GPT word regex at `chars[i..]`.
/// Returns the length (in chars) of the first alternative that matches, or
/// `None` if none do.
fn match_word(chars: &[char], i: usize) -> Option<usize> {
    let n = chars.len();
    let at = |k: usize| -> Option<char> { chars.get(k).copied() };

    // Alt 1: [ascii-punct][A-Za-z]+
    if let Some(c0) = at(i) {
        if is_ascii_punct_lead(c0) {
            let mut j = i + 1;
            while j < n && chars[j].is_ascii_alphabetic() {
                j += 1;
            }
            if j > i + 1 {
                return Some(j - i);
            }
        }
    }

    // Alt 2: [^\r\n\p{L}\p{P}\p{S}]? [\p{L}\p{M}]+
    {
        let mut j = i;
        // optional single leading char that is NOT CR/LF/L/P/S
        if let Some(c) = at(j) {
            if c != '\r' && c != '\n' && !is_l(c) && !is_p(c) && !is_s(c) {
                // tentatively consume it, but only if followed by ≥1 (L|M)
                if let Some(c1) = at(j + 1) {
                    if is_l(c1) || is_m(c1) {
                        j += 1;
                    }
                }
            }
        }
        let start_lm = j;
        while j < n && (is_l(chars[j]) || is_m(chars[j])) {
            j += 1;
        }
        if j > start_lm {
            return Some(j - i);
        }
    }

    // Alt 3:  ?[\p{P}\p{S}]+[\r\n]*
    {
        let mut j = i;
        let lead_space = at(j) == Some(' ');
        if lead_space {
            j += 1;
        }
        let start_ps = j;
        while j < n && (is_p(chars[j]) || is_s(chars[j])) {
            j += 1;
        }
        if j > start_ps {
            // [\r\n]* tail
            while j < n && (chars[j] == '\r' || chars[j] == '\n') {
                j += 1;
            }
            return Some(j - i);
        }
        // the optional leading space did not lead to a P/S run → alt 3 fails
    }

    // Alt 4: \s*[\r\n]+  — a whitespace run that contains ≥1 CR/LF, ending right
    // after the LAST CR/LF in the leading whitespace run. `[\r\n] ⊂ \s`, so a
    // greedy `\s*` would swallow the CR/LF; PCRE backtracks so `[\r\n]+` can
    // match. We compute it directly: scan the whitespace run and remember the
    // index just past the final CR/LF.
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

    // Alt 5: \s+(?!\S)  — a whitespace run whose match is the largest prefix
    // immediately followed by whitespace-or-end. PCRE: greedy `\s+` then the
    // lookahead `(?!\S)` fails iff the char after the run is a non-space, in
    // which case it backtracks one char (now the char after is the space it
    // gave back → lookahead holds). So for a maximal whitespace run of length
    // `w` (note `[\r\n] ⊂ \s`, but alt 4 already consumed any CR/LF-bearing run
    // above, so this run here is CR/LF-free in practice):
    //   * run reaches end-of-piece → match all `w` chars,
    //   * else (followed by non-space) → match `w-1` chars, but only if `w ≥ 2`.
    {
        let mut j = i;
        while j < n && is_ws(chars[j]) {
            j += 1;
        }
        let w = j - i;
        if w >= 1 {
            if j == n {
                return Some(w); // run reaches end → (?!\S) holds at the maximal run
            } else if w >= 2 {
                return Some(w - 1); // cede the final space; the char after it is whitespace
            }
            // w == 1 and followed by a non-space → alt 5 fails; fall to alt 6.
        }
    }

    // Alt 6: \s+  — any remaining whitespace run (the leftover single space that
    // alt 5 could not claim). PCRE matches the full maximal run here.
    {
        let mut j = i;
        while j < n && is_ws(chars[j]) {
            j += 1;
        }
        if j > i {
            return Some(j - i);
        }
    }

    None
}

/// The GPT-2 / HF ByteLevel `bytes_to_unicode` map, applied to the UTF-8 bytes
/// of `s`. 188 printable bytes map to themselves (as the corresponding
/// codepoint); the other 68 control/space bytes map into the U+0100.. region so
/// every byte becomes a single printable codepoint (no UNK, OQ-16 §2).
pub fn byte_level_map(s: &str) -> String {
    s.bytes().map(byte_to_char).collect()
}

/// Map one byte to its byte-level alphabet codepoint (the GPT-2 rule).
#[inline]
fn byte_to_char(b: u8) -> char {
    // Printable set: '!'..='~' (0x21..=0x7E), '¡'..='¬' (0xA1..=0xAC),
    // '®'..='ÿ' (0xAE..=0xFF). These map to themselves. Every other byte n maps
    // to U+0100 + (its index among the non-printable bytes, in ascending order).
    let printable =
        (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    if printable {
        // Safe: all these are valid scalar values < 0x100.
        char::from_u32(b as u32).expect("printable byte is a valid codepoint")
    } else {
        // Count how many non-printable bytes precede `b` to get its offset.
        let mut offset = 0u32;
        for x in 0u8..b {
            let x_printable = (0x21..=0x7E).contains(&x)
                || (0xA1..=0xAC).contains(&x)
                || (0xAE..=0xFF).contains(&x);
            if !x_printable {
                offset += 1;
            }
        }
        char::from_u32(0x100 + offset).expect("byte-level remap stays below 0x144")
    }
}

/// Inverse of [`byte_to_char`]: map a byte-level codepoint back to its byte.
/// Returns `None` for codepoints outside the byte-level alphabet.
#[inline]
pub fn char_to_byte(c: char) -> Option<u8> {
    let cp = c as u32;
    if cp < 0x100 {
        let b = cp as u8;
        let printable =
            (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        if printable {
            return Some(b);
        }
        return None;
    }
    // Remapped region: U+0100 + offset → the offset-th non-printable byte.
    if (0x100..0x144).contains(&cp) {
        let target = cp - 0x100;
        let mut offset = 0u32;
        for x in 0u8..=0xFF {
            let x_printable = (0x21..=0x7E).contains(&x)
                || (0xA1..=0xAC).contains(&x)
                || (0xAE..=0xFF).contains(&x);
            if !x_printable {
                if offset == target {
                    return Some(x);
                }
                offset += 1;
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_membership() {
        // 'A' is a Letter; '5' is a Number; '!' is Punctuation; '+' is Symbol(Sm).
        assert!(is_l('A'));
        assert!(is_l('é'));
        assert!(is_l('一')); // CJK ideograph, category Lo
        assert!(!is_l('5'));
        assert!(is_n('5'));
        assert!(is_n('²')); // superscript two, category No
        assert!(!is_n('A'));
        assert!(is_p('!'));
        assert!(is_p('.'));
        assert!(!is_p('+')); // '+' is Sm (Symbol), not Punctuation
        assert!(is_s('+'));
        assert!(is_s('$')); // currency symbol, Sc
        assert!(is_m('\u{0301}')); // combining acute, category Mn
        assert!(!is_m('a'));
    }

    #[test]
    fn byte_level_roundtrip_is_total() {
        // Every byte maps to a unique codepoint and back.
        let mut seen = std::collections::HashSet::new();
        for b in 0u8..=0xFF {
            let c = byte_to_char(b);
            assert!(seen.insert(c), "byte-level map not injective at {b}");
            assert_eq!(char_to_byte(c), Some(b), "roundtrip failed for byte {b}");
        }
        // Space and newline land in the remapped region.
        assert_eq!(byte_to_char(b' '), 'Ġ');
        assert_eq!(byte_to_char(b'\n'), 'Ċ');
        assert_eq!(byte_to_char(b'\t'), 'ĉ');
    }

    #[test]
    fn byte_level_maps_utf8() {
        // Non-ASCII: 'é' is U+00E9 = bytes [0xC3, 0xA9]; both are in the
        // printable Latin-1 ranges, so they map to themselves (Ã, ©).
        let s = byte_level_map("é");
        assert_eq!(s, "Ã©");
        // A leading space becomes Ġ.
        assert_eq!(byte_level_map(" a"), "Ġa");
    }

    #[test]
    fn digit_grouping_groups_of_three() {
        // \p{N}{1,3} isolates runs in greedy groups of 3.
        assert_eq!(split_digit_groups("1234567"), vec!["123", "456", "7"]);
        assert_eq!(split_digit_groups("ab12cd"), vec!["ab", "12", "cd"]);
        assert_eq!(split_digit_groups("12"), vec!["12"]);
        assert_eq!(split_digit_groups("abc"), vec!["abc"]);
    }

    #[test]
    fn cjk_isolation() {
        let out = split_cjk_kana("a日本b");
        assert_eq!(out, vec!["a", "日本", "b"]);
    }

    #[test]
    fn gpt_word_basic_split() {
        // "Hello world": "Hello" then " world" (the space is grabbed by the
        // second word's alt-2 leading [^…]? since it precedes an L char).
        let out = split_gpt_word("Hello world");
        assert_eq!(out, vec!["Hello", " world"]);
    }

    #[test]
    fn gpt_word_punct_run() {
        // "a..." → "a" then "..." (alt 3, no leading space).
        let out = split_gpt_word("a...");
        assert_eq!(out, vec!["a", "..."]);
        // " !!!" → " !!!" (alt 3 with leading space).
        let out2 = split_gpt_word(" !!!");
        assert_eq!(out2, vec![" !!!"]);
    }

    #[test]
    fn gpt_word_trailing_whitespace() {
        // "a  " → "a" then "  " (alt 5: \s+ to end).
        let out = split_gpt_word("a  ");
        assert_eq!(out, vec!["a", "  "]);
    }

    #[test]
    fn full_pretokenize_byte_mapped() {
        // End-to-end: "Hi" → ["Hi"] mapped (printable ASCII = identity).
        let p = pretokenize("Hi");
        assert_eq!(p, vec!["Hi"]);
        // " a" → [" a"] → "Ġa".
        let p2 = pretokenize(" a");
        assert_eq!(p2, vec!["Ġa"]);
    }
}
