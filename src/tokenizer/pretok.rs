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

/// Pre-tokenize `text` for the SmolLM2 / GPT-2 scheme (C6, bd-3jo6.3.6) —
/// verbatim from the SmolVLM2 `tokenizer.json .pre_tokenizer`:
///
/// ```text
/// 1. Digits    individual_digits=true                       Isolated
/// 2. ByteLevel add_prefix_space=false trim_offsets=true use_regex=true
/// ```
///
/// Stage 1 isolates every `\p{N}` char as its own piece (HF `Digits` splits on
/// Rust `char::is_numeric`, which is exactly general-category N — the same set
/// as [`is_n`]). Stage 2 is the classic GPT-2 word regex (`use_regex=true`,
/// unlike the DeepSeek sequence above which pre-splits itself and sets
/// `use_regex=false`), then the byte→unicode remap. Same leftmost-first
/// alternation semantics as [`pretokenize`].
pub fn pretokenize_smollm2(text: &str) -> Vec<String> {
    let mut pieces = vec![text.to_string()];
    pieces = split_stage(&pieces, split_digits_individual);
    pieces = split_stage(&pieces, split_gpt2_word);
    pieces.iter().map(|p| byte_level_map(p)).collect()
}

/// Pre-tokenize `text` for the plain GPT-2 scheme (OneChart / OPT, D9):
/// JUST the classic GPT-2 word regex + the byte-level remap — no Digits
/// stage (the slow `GPT2Tokenizer` over `vocab.json`+`merges.txt`; the
/// fixture `_meta.pat_str` pins the exact pattern [`split_gpt2_word`]
/// implements).
pub fn pretokenize_gpt2(text: &str) -> Vec<String> {
    split_gpt2_word(text)
        .iter()
        .map(|p| byte_level_map(p))
        .collect()
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

/// SmolLM2 stage 1: `Digits(individual_digits=true)` Isolated — every `\p{N}`
/// char becomes its own piece (HF `Digits` splits on Rust `char::is_numeric`,
/// which is general-category N — the same set as [`is_n`]).
fn split_digits_individual(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for c in s.chars() {
        if is_n(c) {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            out.push(c.to_string());
        } else {
            buf.push(c);
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
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
    if let Some(c0) = at(i)
        && is_ascii_punct_lead(c0)
    {
        let mut j = i + 1;
        while j < n && chars[j].is_ascii_alphabetic() {
            j += 1;
        }
        if j > i + 1 {
            return Some(j - i);
        }
    }

    // Alt 2: [^\r\n\p{L}\p{P}\p{S}]? [\p{L}\p{M}]+
    {
        let mut j = i;
        // optional single leading char that is NOT CR/LF/L/P/S
        if let Some(c) = at(j)
            && c != '\r'
            && c != '\n'
            && !is_l(c)
            && !is_p(c)
            && !is_s(c)
        {
            // tentatively consume it, but only if followed by ≥1 (L|M)
            if let Some(c1) = at(j + 1)
                && (is_l(c1) || is_m(c1))
            {
                j += 1;
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

/// SmolLM2 stage 2: the classic GPT-2 word regex, applied by
/// `ByteLevel(use_regex=true)` (Isolated):
///
/// ```text
/// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
/// ```
///
/// Same scan shape as [`split_gpt_word`] but with the GPT-2 alternatives: the
/// lowercase contraction suffixes come FIRST, letters are plain `\p{L}+` (no
/// `\p{M}` — unlike the DeepSeek variant), numbers are an unbounded ` ?\p{N}+`
/// (the 1-per-piece Digits stage has already run), the "other" class takes
/// everything that is not whitespace/letter/number, and there is no
/// `\s*[\r\n]+` alternative. The regex covers every char (any non-(ws|L|N)
/// char matches alternative 4), so Isolated leaves no gaps.
fn split_gpt2_word(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut gap = String::new();
    let mut i = 0;
    while i < chars.len() {
        if let Some(len) = match_gpt2_word(&chars, i) {
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

/// Try the six ordered alternatives of the GPT-2 word regex at `chars[i..]`
/// (leftmost-first). Returns the char length of the first match, or `None`.
fn match_gpt2_word(chars: &[char], i: usize) -> Option<usize> {
    let n = chars.len();

    // Alt 1: '(s|t|re|ve|m|ll|d) — ASCII apostrophe + lowercase suffix, tried
    // in source order (case-sensitive: "'S" falls through to alt 4).
    if chars[i] == '\'' {
        for suf in ["s", "t", "re", "ve", "m", "ll", "d"] {
            let sl = suf.len(); // all-ASCII → char count == byte count
            if i + 1 + sl <= n && chars[i + 1..i + 1 + sl].iter().collect::<String>() == *suf {
                return Some(1 + sl);
            }
        }
    }

    // Alt 2: ` ?\p{L}+` — optional single ASCII space, then ≥1 Letter.
    {
        let mut j = i;
        if chars[j] == ' '
            && let Some(&c1) = chars.get(j + 1)
            && is_l(c1)
        {
            j += 1;
        }
        let start = j;
        while j < n && is_l(chars[j]) {
            j += 1;
        }
        if j > start {
            return Some(j - i);
        }
    }

    // Alt 3: ` ?\p{N}+` — optional single ASCII space, then ≥1 Number.
    {
        let mut j = i;
        if chars[j] == ' '
            && let Some(&c1) = chars.get(j + 1)
            && is_n(c1)
        {
            j += 1;
        }
        let start = j;
        while j < n && is_n(chars[j]) {
            j += 1;
        }
        if j > start {
            return Some(j - i);
        }
    }

    // Alt 4: ` ?[^\s\p{L}\p{N}]+` — optional space, then ≥1 char that is not
    // whitespace/Letter/Number (punctuation, symbols, marks, …).
    {
        let other = |c: char| !is_ws(c) && !is_l(c) && !is_n(c);
        let mut j = i;
        if chars[j] == ' '
            && let Some(&c1) = chars.get(j + 1)
            && other(c1)
        {
            j += 1;
        }
        let start = j;
        while j < n && other(chars[j]) {
            j += 1;
        }
        if j > start {
            return Some(j - i);
        }
    }

    // Alt 5: `\s+(?!\S)` — same backtracking semantics as [`match_word`]'s
    // alt 5: a maximal whitespace run of length w matches all w chars when it
    // reaches end-of-piece, else w-1 chars (only if w ≥ 2).
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

    // Alt 6: `\s+` — the leftover whitespace run (a single space before a
    // non-space).
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

    // ── SmolLM2 / GPT-2 scheme (C6) ──────────────────────────────────────────

    #[test]
    fn smollm2_digits_are_individual() {
        // Digits(individual_digits=true): every \p{N} char is its own piece.
        assert_eq!(split_digits_individual("1234"), vec!["1", "2", "3", "4"]);
        assert_eq!(
            split_digits_individual("ab12cd"),
            vec!["ab", "1", "2", "cd"]
        );
        assert_eq!(split_digits_individual("²x"), vec!["²", "x"]); // No (superscript) counts
        assert_eq!(split_digits_individual("abc"), vec!["abc"]);
    }

    #[test]
    fn gpt2_contractions_match_first() {
        // "it's" → "it" (alt 2), "'s" (alt 1).
        assert_eq!(split_gpt2_word("it's"), vec!["it", "'s"]);
        // "we'll've" → "we", "'ll", "'ve".
        assert_eq!(split_gpt2_word("we'll've"), vec!["we", "'ll", "'ve"]);
        // Uppercase suffix does NOT match alt 1: "'S" → "'" (alt 4), "S" (alt 2).
        assert_eq!(split_gpt2_word("IT'S"), vec!["IT", "'", "S"]);
        // Trailing bare apostrophe → alt 4.
        assert_eq!(split_gpt2_word("dogs'"), vec!["dogs", "'"]);
    }

    #[test]
    fn gpt2_letters_have_no_mark_class() {
        // A combining mark after letters is NOT part of \p{L}+ in GPT-2 (no
        // \p{M} in alt 2, unlike the DeepSeek regex): the mark is an alt-4
        // "other" run of its own, and the following letter restarts alt 2.
        assert_eq!(split_gpt2_word("e\u{0301}x"), vec!["e", "\u{0301}", "x"]);
        assert_eq!(split_gpt2_word("e\u{0301}"), vec!["e", "\u{0301}"]);
    }

    #[test]
    fn gpt2_number_runs_after_digit_stage() {
        // Alt 3 with a leading space: " 5" is one piece when it reaches the
        // regex intact (single-digit, so the Digits stage's per-char split
        // yields the same boundary anyway for multi-digit runs).
        assert_eq!(split_gpt2_word(" 5"), vec![" 5"]);
        // Full scheme: digits isolated FIRST, so "a 12" → "a", " 1"? No — the
        // Digits stage cuts before the regex ever sees " 12": pieces are
        // ["a ", "1", "2"], and the regex runs per piece.
        assert_eq!(pretokenize_smollm2("a 12"), vec!["a", "Ġ", "1", "2"]);
    }

    #[test]
    fn gpt2_whitespace_backtracking() {
        // "a  b": maximal run of 2 spaces before 'b' → alt 5 yields w-1=1
        // space, then " b" via alt 2's optional leading space.
        assert_eq!(split_gpt2_word("a  b"), vec!["a", " ", " b"]);
        // Trailing run reaches end → alt 5 takes it whole.
        assert_eq!(split_gpt2_word("a  "), vec!["a", "  "]);
        // "\n\n" at end of piece stays ONE pretoken (BPE may merge to ĊĊ —
        // the OQ-4 image-expansion tail case).
        assert_eq!(split_gpt2_word("a\n\n"), vec!["a", "\n\n"]);
        // "\n\n" mid-piece: alt 5 cedes one char, alt 6 takes the second
        // (alt 2's optional lead is strictly ' ', so '\n' can't join 'b').
        assert_eq!(split_gpt2_word("a\n\nb"), vec!["a", "\n", "\n", "b"]);
    }

    #[test]
    fn gpt2_punct_and_symbols() {
        // " !!!" → alt 4 with leading space, one piece.
        assert_eq!(split_gpt2_word(" !!!"), vec![" !!!"]);
        // "a..." → "a", "..." (alt 4).
        assert_eq!(split_gpt2_word("a..."), vec!["a", "..."]);
        // Mixed symbol run: "+=<>" is one alt-4 run.
        assert_eq!(split_gpt2_word("+=<>"), vec!["+=<>"]);
    }

    #[test]
    fn smollm2_full_scheme_byte_mapped() {
        // "Hi 42!" → Digits: ["Hi ", "4", "2", "!"] → regex per piece →
        // ["Hi", " ", "4", "2", "!"]  (the space: alt-2 lookahead fails at
        // piece end after "Hi", so "Hi" then " " via alt 5 w=1-at-end)
        // → byte map: space becomes Ġ.
        assert_eq!(
            pretokenize_smollm2("Hi 42!"),
            vec!["Hi", "Ġ", "4", "2", "!"]
        );
        // Contraction survives the full scheme.
        assert_eq!(pretokenize_smollm2("it's"), vec!["it", "'s"]);
        // UTF-8 goes through the byte map ('é' → Ã©).
        assert_eq!(pretokenize_smollm2("é"), vec!["Ã©"]);
    }
}
