//! Postprocess: EOS strip, ref/det parse, bbox /999 rescale, markdown, `<PAGE>`
//! ([SPEC-110..119], PROPOSED_ARCHITECTURE.md §6.11).
//!
//! Faithful port of the reference Python postprocess (`modeling_unlimitedocr.py`,
//! HF commit `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`), specifically:
//!
//! * `infer(...).save_results` block (lines 1069-1095): strip a trailing EOS
//!   `<｜end▁of▁sentence｜>`, `.strip()`, run `re_match`, then rewrite every
//!   `image`-labelled ref span to `![](images/{idx}.jpg)\n` and delete every
//!   other ref/det span, applying `\coloneqq`->`:=` / `\eqqcolon`->`=:`.
//! * `infer_multi(...).save_results` block (lines 1267-1297): split on `<PAGE>`,
//!   drop the leading chunk, postprocess each page, rejoin with `<PAGE>\n` /
//!   `\n<PAGE>\n` separators (OQ-8, `oq/preprocess-infer.md`).
//! * `re_match` (lines 44-59) — the ref/det regexes — reimplemented as a
//!   hand-written scanner because the crate has no `regex` dependency (the
//!   parallel wave owns Cargo.toml; we may not add one). The scanner reproduces
//!   the two Python patterns exactly:
//!     - ref: `(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)`  (DOTALL,
//!       non-greedy)
//!     - det: `(<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>)`
//! * coordinate de-normalization (lines 107-111): `x = int(x/999*W)`,
//!   `y = int(y/999*H)` (Python `int()` = truncation toward zero).
//!
//! Out of scope for v1 (matching the stub note): box-overlay drawing, the
//! `line_type` geometry special case, image-crop file I/O (we emit the markdown
//! `![](images/{idx}.jpg)` references but do not write the JPEGs).

use crate::error::FocrResult;

/// The end-of-sentence marker the decoder emits (`modeling_unlimitedocr.py:1050`
/// / `:1071` / `:1260`). Note the full-width bar `｜` (U+FF5C) and the special
/// underscore-like `▁` (U+2581) — these are the exact codepoints in the
/// reference vocabulary, NOT ASCII `|`/`_`.
pub const EOS_MARKER: &str = "<｜end▁of▁sentence｜>";

/// The multi-page separator the model emits between page outputs
/// (`modeling_unlimitedocr.py:1269` / `:1295`).
pub const PAGE_MARKER: &str = "<PAGE>";

// ── ref/det span model ──────────────────────────────────────────────────────

/// One parsed `<|ref|>…<|/ref|><|det|>…<|/det|>` (or bare `<|det|>…<|/det|>`)
/// span found in the decoded text.
///
/// Mirrors a `re_match` tuple `(full_match, label, box_text)`: [`full`] is the
/// exact matched substring (used verbatim for `str.replace`), [`label`] is the
/// ref/det label (e.g. `"title"`, `"image"`, `"text"`), and [`boxes`] is the
/// parsed coordinate list. The reference stores `box` as raw text and `eval`s it
/// later; we parse it eagerly into integer quads.
#[derive(Debug, Clone, PartialEq)]
pub struct RefMatch {
    /// The exact matched substring, byte-for-byte (the `str.replace` key).
    pub full: String,
    /// The label between the ref (or det) tags, trimmed of surrounding
    /// whitespace for classification but stored as the captured group.
    pub label: String,
    /// Parsed coordinate quads `[x1, y1, x2, y2]` in the model's 0..=999 space.
    /// A single bare quad in the source (`[a,b,c,d]`) yields one entry; a list of
    /// quads (`[[..],[..]]`) yields several.
    pub boxes: Vec<[i64; 4]>,
}

impl RefMatch {
    /// Whether this span is an `image`-labelled region (becomes an
    /// `![](images/{idx}.jpg)` reference rather than being deleted).
    ///
    /// Matches the Python predicate `a_match[1].strip() == 'image' or
    /// '<|ref|>image<|/ref|>' in a_match[0]` (`modeling_unlimitedocr.py:55`).
    #[must_use]
    pub fn is_image(&self) -> bool {
        self.label.trim() == "image" || self.full.contains("<|ref|>image<|/ref|>")
    }

    /// Rescale this span's boxes from the model's 0..=999 grid into pixel
    /// coordinates for an `width x height` image.
    ///
    /// `x = (x / 999 * width) as i64` (truncation toward zero, matching Python
    /// `int(...)`); `y` uses `height`. See `modeling_unlimitedocr.py:107-111`.
    #[must_use]
    pub fn rescaled_boxes(&self, width: u32, height: u32) -> Vec<[i64; 4]> {
        self.boxes
            .iter()
            .map(|&[x1, y1, x2, y2]| {
                [
                    rescale(x1, width),
                    rescale(y1, height),
                    rescale(x2, width),
                    rescale(y2, height),
                ]
            })
            .collect()
    }
}

/// De-normalize one 0..=999 coordinate to a pixel coordinate for `extent`
/// pixels: `int(coord / 999 * extent)` — float divide then truncate toward zero
/// (Python `int()` on a positive float == floor; the model never emits
/// negatives, but `as i64` truncates toward zero for either sign).
#[must_use]
fn rescale(coord: i64, extent: u32) -> i64 {
    (coord as f64 / 999.0 * f64::from(extent)) as i64
}

// ── EOS / strip ─────────────────────────────────────────────────────────────

/// Strip a single trailing [`EOS_MARKER`] (only if the text ends with it) and
/// then trim surrounding ASCII/Unicode whitespace — `modeling_unlimitedocr.py:
/// 1076-1078`.
///
/// Python's `str.strip()` trims Unicode whitespace from both ends; Rust's
/// [`str::trim`] does the same (Unicode `White_Space`), so the result matches.
#[must_use]
pub fn strip_eos(text: &str) -> String {
    let body = text.strip_suffix(EOS_MARKER).unwrap_or(text);
    body.trim().to_string()
}

// ── re_match: ref/det scanner (regex-free) ──────────────────────────────────

/// Parse all ref/det spans from `text`, reproducing the reference `re_match`
/// (`modeling_unlimitedocr.py:44-59`) without a `regex` dependency.
///
/// Returns the spans in the same order the Python `re.findall` would produce:
/// first every full `<|ref|>L<|/ref|><|det|>B<|/det|>` span (in left-to-right
/// order), then every bare `<|det|>L B<|/det|>` span (left-to-right). The two
/// passes can overlap on the same `<|det|>…<|/det|>` text exactly as the Python
/// regexes do; callers downstream replace by substring so duplicates are
/// idempotent.
#[must_use]
pub fn re_match(text: &str) -> Vec<RefMatch> {
    let mut out = Vec::new();
    out.extend(scan_ref_spans(text));
    out.extend(scan_det_spans(text));
    out
}

/// Pass 1: `(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)` — DOTALL,
/// non-greedy. The ref body and det body are taken minimally; the det tags must
/// immediately follow the ref close (no text between `<|/ref|>` and `<|det|>`),
/// exactly like the literal-adjacent pattern.
fn scan_ref_spans(text: &str) -> Vec<RefMatch> {
    const REF_OPEN: &str = "<|ref|>";
    const REF_CLOSE: &str = "<|/ref|>";
    const DET_OPEN: &str = "<|det|>";
    const DET_CLOSE: &str = "<|/det|>";

    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while let Some(rel) = find_from(bytes, i, REF_OPEN) {
        let body_start = rel + REF_OPEN.len();
        // non-greedy `(.*?)` up to the first `<|/ref|>`
        let Some(ref_close) = find_from(bytes, body_start, REF_CLOSE) else {
            break;
        };
        let label = &text[body_start..ref_close];
        let after_ref = ref_close + REF_CLOSE.len();
        // The pattern requires `<|det|>` to immediately follow `<|/ref|>`.
        if !slice_starts_with(bytes, after_ref, DET_OPEN) {
            // No adjacent det: this ref-open cannot start a full span; advance
            // past this ref-open and keep scanning.
            i = body_start;
            continue;
        }
        let det_body_start = after_ref + DET_OPEN.len();
        let Some(det_close) = find_from(bytes, det_body_start, DET_CLOSE) else {
            break;
        };
        let box_text = &text[det_body_start..det_close];
        let span_end = det_close + DET_CLOSE.len();
        let full = &text[rel..span_end];
        out.push(RefMatch {
            full: full.to_string(),
            label: label.to_string(),
            boxes: parse_boxes(box_text),
        });
        i = span_end;
    }
    out
}

/// Pass 2: `(<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>)`.
///
/// A `<|det|>` whose body is `<ws><label-ident><ws><[...]><ws>` then `<|/det|>`,
/// where the label is a Python identifier-ish token (`[A-Za-z_][\w-]*`, i.e.
/// starts with an ASCII letter/underscore, continues with Unicode word chars or
/// `-`), and the box is a single bracketed group with no nested `]`. Python's
/// `\s` is Unicode-aware, so whitespace skipping is char-aware here too.
fn scan_det_spans(text: &str) -> Vec<RefMatch> {
    const DET_OPEN: &str = "<|det|>";
    const DET_CLOSE: &str = "<|/det|>";

    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while let Some(open) = find_from(bytes, i, DET_OPEN) {
        let mut p = open + DET_OPEN.len();
        i = p; // default advance: just past this `<|det|>`
        // \s*
        p = skip_ws(text, p);
        // [A-Za-z_][\w-]*
        let label_start = p;
        let Some(first) = text[p..].chars().next() else {
            continue;
        };
        if !is_ident_start(first) {
            continue;
        }
        p += first.len_utf8();
        while let Some(ch) = text[p..].chars().next() {
            if !is_ident_continue(ch) {
                break;
            }
            p += ch.len_utf8();
        }
        let label_end = p;
        // \s*
        p = skip_ws(text, p);
        // \[[^\]]+\]
        if p >= bytes.len() || bytes[p] != b'[' {
            continue;
        }
        let box_start = p;
        p += 1;
        let inner_start = p;
        while p < bytes.len() && bytes[p] != b']' {
            p += 1;
        }
        if p >= bytes.len() || p == inner_start {
            // unterminated `[` or empty `[]` (the `+` requires >=1 inner char)
            continue;
        }
        let box_end = p + 1; // include `]`
        p = box_end;
        // \s*
        p = skip_ws(text, p);
        // <|/det|>
        if !slice_starts_with(bytes, p, DET_CLOSE) {
            continue;
        }
        let span_end = p + DET_CLOSE.len();
        let full = &text[open..span_end];
        let label = &text[label_start..label_end];
        let box_text = &text[box_start..box_end];
        out.push(RefMatch {
            full: full.to_string(),
            label: label.to_string(),
            boxes: parse_boxes(box_text),
        });
        i = span_end;
    }
    out
}

/// Parse a Python-list-literal coordinate text into `[x1,y1,x2,y2]` quads,
/// reproducing `extract_coordinates_and_label`'s `eval(...)` + the
/// "single quad -> list of one quad" normalization (`:66-68`).
///
/// Accepts `[a, b, c, d]` (one quad) or `[[a,b,c,d], [e,f,g,h], ...]` (many).
/// Numbers may be int or float in the source; floats are truncated to i64 (the
/// model emits integer coords, but we tolerate floats defensively). Any quad
/// that doesn't have exactly four numbers is skipped (the Python `eval` would
/// have produced a wrongly-shaped list that the draw loop then ignores via its
/// `x1,y1,x2,y2 = points` unpack failing inside the per-points `try`).
fn parse_boxes(box_text: &str) -> Vec<[i64; 4]> {
    let nums = extract_numbers(box_text);
    let mut out = Vec::new();
    // Group flat numbers into quads. Both `[a,b,c,d]` and `[[a,b,c,d],...]`
    // flatten to the same number stream, so chunking by 4 reconstructs the
    // quads (the reference's single-quad case `[[cor_list]]` is the same data).
    for chunk in nums.chunks(4) {
        if chunk.len() == 4 {
            out.push([
                chunk[0] as i64,
                chunk[1] as i64,
                chunk[2] as i64,
                chunk[3] as i64,
            ]);
        }
    }
    out
}

/// Pull every (possibly signed/decimal) numeric literal out of `s`, in order.
fn extract_numbers(s: &str) -> Vec<f64> {
    let bytes = s.as_bytes();
    let mut nums = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        let is_num_start = c.is_ascii_digit()
            || ((c == b'-' || c == b'+' || c == b'.')
                && i + 1 < bytes.len()
                && (bytes[i + 1].is_ascii_digit() || bytes[i + 1] == b'.'));
        if is_num_start {
            let start = i;
            if c == b'-' || c == b'+' {
                i += 1;
            }
            let mut seen_dot = false;
            while i < bytes.len() {
                let d = bytes[i];
                if d.is_ascii_digit() {
                    i += 1;
                } else if d == b'.' && !seen_dot {
                    seen_dot = true;
                    i += 1;
                } else {
                    break;
                }
            }
            if let Ok(v) = s[start..i].parse::<f64>() {
                nums.push(v);
            }
        } else {
            i += 1;
        }
    }
    nums
}

// ── small byte-scanner helpers ──────────────────────────────────────────────

/// First byte index `>= from` at which `needle` occurs in `hay`, or `None`.
fn find_from(hay: &[u8], from: usize, needle: &str) -> Option<usize> {
    let n = needle.as_bytes();
    if n.is_empty() || from > hay.len() {
        return None;
    }
    let last = hay.len().checked_sub(n.len())?;
    (from..=last).find(|&i| &hay[i..i + n.len()] == n)
}

/// Whether `hay[at..]` begins with `needle`.
fn slice_starts_with(hay: &[u8], at: usize, needle: &str) -> bool {
    let n = needle.as_bytes();
    at + n.len() <= hay.len() && &hay[at..at + n.len()] == n
}

/// Advance past Unicode whitespace, matching Python regex `\s` on `str`.
fn skip_ws(text: &str, mut at: usize) -> usize {
    while let Some(ch) = text[at..].chars().next() {
        if !ch.is_whitespace() {
            break;
        }
        at += ch.len_utf8();
    }
    at
}

/// `[A-Za-z_]`.
fn is_ident_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_'
}

/// `[\w-]`: Python's Unicode word chars plus `-`.
fn is_ident_continue(ch: char) -> bool {
    ch == '_' || ch == '-' || ch.is_alphanumeric()
}

// ── markdown assembly ───────────────────────────────────────────────────────

/// Rewrite a single page's already-EOS-stripped text into final markdown,
/// reproducing the per-page replacement loop
/// (`modeling_unlimitedocr.py:1080-1089` / `:1277-1291`).
///
/// `text` is the stripped page body. `img_base` indexes the emitted image
/// references: in single-page mode it is `""` so images become
/// `![](images/0.jpg)`; in multi-page mode it is `"page_{n}_"` so they become
/// `![](images/page_{n}_0.jpg)`. The replacements run in match order, each
/// `str.replace` substituting ALL occurrences of the matched span (Python
/// semantics). `\coloneqq` / `\eqqcolon` are normalized globally after the span
/// rewrites, per SPEC-115.
fn assemble_page(text: &str, img_base: &str) -> String {
    let matches = re_match(text);
    let mut images = Vec::new();
    let mut others = Vec::new();
    for m in &matches {
        if m.is_image() {
            images.push(m);
        } else {
            others.push(m);
        }
    }

    let mut out = text.to_string();
    for (idx, m) in images.iter().enumerate() {
        let replacement = format!("![](images/{img_base}{idx}.jpg)\n");
        out = out.replace(&m.full, &replacement);
    }
    for m in &others {
        out = out.replace(&m.full, "");
    }
    normalize_colon_equals(&out)
}

fn normalize_colon_equals(text: &str) -> String {
    text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
}

/// Turn raw decoded model text + the source image dimensions into the final
/// structured markdown (ref/det parsed, image spans -> markdown image refs,
/// layout/other spans removed, LaTeX `\coloneqq`/`\eqqcolon` normalized).
///
/// This is the single-image `infer(...).save_results` path
/// (`modeling_unlimitedocr.py:1069-1095`): strip the trailing EOS, `.strip()`,
/// then run the per-page assembly with an empty image prefix.
///
/// `image_w` / `image_h` are the source pixel dimensions; they are accepted for
/// signature stability and bbox de-normalization. The reference markdown body
/// does not embed pixel coordinates (boxes are only drawn onto the overlay JPEG,
/// excluded from v1), so the dimensions do not change the returned string —
/// callers that need pixel boxes use [`parse_layout`].
///
/// # Errors
/// Infallible today (returns `Ok`); the `FocrResult` shape is kept so a future
/// validation pass (e.g. malformed-span detection) can surface errors without a
/// signature break.
pub fn finalize(decoded: &str, _image_w: u32, _image_h: u32) -> FocrResult<String> {
    let stripped = strip_eos(decoded);
    Ok(assemble_page(&stripped, ""))
}

/// Multi-page `infer_multi(...).save_results` assembly
/// (`modeling_unlimitedocr.py:1267-1295`).
///
/// Strips the trailing EOS, `.strip()`, splits on [`PAGE_MARKER`], DROPS the
/// leading chunk (`pages = outputs.split('<PAGE>')[1:]` — text before the first
/// `<PAGE>` is discarded), postprocesses each page with prefix `page_{n}_`, then
/// rejoins as `"<PAGE>\n" + "\n<PAGE>\n".join(pages)`.
///
/// `num_pages` is the count of source page images: pages with index `>= num_pages`
/// are passed through verbatim (only `.strip()`-ed), matching the reference
/// `if page_idx >= len(images): processed_pages.append(page_output); continue`.
///
/// # Errors
/// Infallible today; see [`finalize`] for why the `FocrResult` shape is kept.
pub fn finalize_multi(decoded: &str, num_pages: usize) -> FocrResult<String> {
    let stripped = strip_eos(decoded);
    // Python `str.split('<PAGE>')[1:]` — drop the chunk before the first marker.
    let mut chunks = stripped.split(PAGE_MARKER);
    let _ = chunks.next(); // discard leading chunk (the `[1:]`)
    let mut processed = Vec::new();
    for (page_idx, page) in chunks.enumerate() {
        let page = page.trim();
        if page_idx >= num_pages {
            processed.push(page.to_string());
            continue;
        }
        let prefix = format!("page_{page_idx}_");
        processed.push(assemble_page(page, &prefix));
    }
    let body = processed.join("\n<PAGE>\n");
    Ok(format!("<PAGE>\n{body}"))
}

/// Parse the layout (ref/det spans) from raw decoded text and return each span's
/// label plus its pixel-rescaled boxes, for callers that need the bounding-box
/// geometry (overlay drawing, structured layout export) rather than the markdown
/// body.
///
/// EOS is stripped first (so a trailing marker never leaks into the last span),
/// then [`re_match`] runs and every box is de-normalized to `(image_w, image_h)`
/// pixels via the `/999` rescale (`modeling_unlimitedocr.py:107-111`). Returns
/// `(label, boxes)` pairs in match order.
#[must_use]
pub fn parse_layout(decoded: &str, image_w: u32, image_h: u32) -> Vec<(String, Vec<[i64; 4]>)> {
    let stripped = strip_eos(decoded);
    re_match(&stripped)
        .into_iter()
        .map(|m| {
            let boxes = m.rescaled_boxes(image_w, image_h);
            (m.label, boxes)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_eos_removes_trailing_marker_and_trims() {
        let raw = format!("hello world{EOS_MARKER}");
        assert_eq!(strip_eos(&raw), "hello world");
        // trailing whitespace before/after the marker is trimmed too
        let raw2 = format!("  body text  {EOS_MARKER}");
        assert_eq!(strip_eos(&raw2), "body text");
        // no marker -> just trims
        assert_eq!(strip_eos("  bare  "), "bare");
        // marker only at the front is NOT stripped (only suffix)
        let raw3 = format!("{EOS_MARKER}keep");
        assert_eq!(strip_eos(&raw3), format!("{EOS_MARKER}keep"));
    }

    #[test]
    fn rescale_matches_python_int_truncation() {
        // x/999*W: 999 -> exactly W; 0 -> 0; mid truncates toward zero.
        assert_eq!(rescale(999, 1000), 1000);
        assert_eq!(rescale(0, 1000), 0);
        // 500/999*1000 = 500.5005... -> 500
        assert_eq!(rescale(500, 1000), 500);
        // 1/999*100 = 0.10010... -> 0
        assert_eq!(rescale(1, 100), 0);
    }

    #[test]
    fn extract_numbers_handles_ints_floats_signs() {
        assert_eq!(extract_numbers("[1, 2, 3, 4]"), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            extract_numbers("[10, -5, 3.5, 0]"),
            vec![10.0, -5.0, 3.5, 0.0]
        );
        assert!(extract_numbers("[]").is_empty());
    }

    #[test]
    fn parse_boxes_single_quad() {
        assert_eq!(
            parse_boxes("[100, 200, 300, 400]"),
            vec![[100, 200, 300, 400]]
        );
    }

    #[test]
    fn parse_boxes_list_of_quads() {
        assert_eq!(
            parse_boxes("[[1, 2, 3, 4], [5, 6, 7, 8]]"),
            vec![[1, 2, 3, 4], [5, 6, 7, 8]]
        );
    }

    #[test]
    fn parse_boxes_drops_incomplete_trailing() {
        // five numbers -> one full quad, trailing single dropped
        assert_eq!(parse_boxes("[1, 2, 3, 4, 5]"), vec![[1, 2, 3, 4]]);
    }

    #[test]
    fn re_match_full_ref_det_span() {
        let text = "<|ref|>title<|/ref|><|det|>[10, 20, 30, 40]<|/det|>";
        let ms = re_match(text);
        // ref pass finds the full span; det pass ALSO finds the inner det
        // (exactly like the two Python regexes overlapping). The first is the
        // full ref/det span.
        assert!(ms.iter().any(|m| m.full == text && m.label == "title"));
        let full = ms.iter().find(|m| m.label == "title").unwrap();
        assert_eq!(full.boxes, vec![[10, 20, 30, 40]]);
    }

    #[test]
    fn re_match_non_greedy_two_spans() {
        let text =
            "<|ref|>a<|/ref|><|det|>[1,2,3,4]<|/det|>X<|ref|>b<|/ref|><|det|>[5,6,7,8]<|/det|>";
        let ms = scan_ref_spans(text);
        assert_eq!(ms.len(), 2);
        assert_eq!(ms[0].label, "a");
        assert_eq!(ms[0].boxes, vec![[1, 2, 3, 4]]);
        assert_eq!(ms[1].label, "b");
        assert_eq!(ms[1].boxes, vec![[5, 6, 7, 8]]);
    }

    #[test]
    fn re_match_bare_det_span() {
        // det pattern requires an identifier label then a bracket group
        let text = "noise <|det|> figure [0, 0, 100, 100] <|/det|> tail";
        let ms = scan_det_spans(text);
        assert_eq!(ms.len(), 1);
        assert_eq!(ms[0].label, "figure");
        assert_eq!(ms[0].boxes, vec![[0, 0, 100, 100]]);
        assert!(ms[0].full.starts_with("<|det|>"));
        assert!(ms[0].full.ends_with("<|/det|>"));
    }

    #[test]
    fn re_match_det_uses_python_unicode_whitespace() {
        let text = "noise <|det|>\u{00a0}figure\u{00a0}[0, 0, 100, 100]\u{00a0}<|/det|> tail";
        let ms = scan_det_spans(text);
        assert_eq!(ms.len(), 1);
        assert_eq!(ms[0].label, "figure");
        assert_eq!(ms[0].boxes, vec![[0, 0, 100, 100]]);
    }

    #[test]
    fn re_match_det_allows_unicode_word_continuation() {
        let text = "noise <|det|> a\u{00e9}-label [1, 2, 3, 4] <|/det|> tail";
        let ms = scan_det_spans(text);
        assert_eq!(ms.len(), 1);
        assert_eq!(ms[0].label, "a\u{00e9}-label");
        assert_eq!(ms[0].boxes, vec![[1, 2, 3, 4]]);
    }

    #[test]
    fn re_match_det_rejects_bad_label() {
        // label starting with a digit is not [A-Za-z_]...
        let text = "<|det|>9bad [1,2,3,4]<|/det|>";
        assert!(scan_det_spans(text).is_empty());
    }

    #[test]
    fn is_image_predicate() {
        let m = RefMatch {
            full: "<|ref|>image<|/ref|><|det|>[1,2,3,4]<|/det|>".into(),
            label: "image".into(),
            boxes: vec![[1, 2, 3, 4]],
        };
        assert!(m.is_image());
        let m2 = RefMatch {
            full: "<|ref|>title<|/ref|><|det|>[1,2,3,4]<|/det|>".into(),
            label: "title".into(),
            boxes: vec![[1, 2, 3, 4]],
        };
        assert!(!m2.is_image());
    }

    #[test]
    fn finalize_strips_other_spans_and_keeps_text() {
        let raw =
            format!("Heading\n<|ref|>title<|/ref|><|det|>[1,2,3,4]<|/det|>\nBody text{EOS_MARKER}");
        let md = finalize(&raw, 1000, 1000).unwrap();
        // the title (other) span is removed; surrounding text remains
        assert!(md.contains("Heading"));
        assert!(md.contains("Body text"));
        assert!(!md.contains("<|ref|>"));
        assert!(!md.contains("<|det|>"));
        assert!(!md.contains(EOS_MARKER));
    }

    #[test]
    fn finalize_rewrites_image_spans_to_markdown() {
        let raw = format!(
            "Top\n<|ref|>image<|/ref|><|det|>[0,0,500,500]<|/det|>\n<|ref|>image<|/ref|><|det|>[1,1,2,2]<|/det|>End{EOS_MARKER}"
        );
        let md = finalize(&raw, 800, 600).unwrap();
        assert!(md.contains("![](images/0.jpg)"));
        assert!(md.contains("![](images/1.jpg)"));
        assert!(md.contains("Top"));
        assert!(md.contains("End"));
        assert!(!md.contains("<|ref|>"));
    }

    #[test]
    fn finalize_normalizes_latex_coloneqq() {
        // a \coloneqq inside body text gets normalized when an `other` span is
        // present.
        let raw = format!(
            "x \\coloneqq y and a \\eqqcolon b <|ref|>note<|/ref|><|det|>[1,2,3,4]<|/det|>{EOS_MARKER}"
        );
        let md = finalize(&raw, 100, 100).unwrap();
        assert!(md.contains("x := y"));
        assert!(md.contains("a =: b"));
        assert!(!md.contains("\\coloneqq"));
        assert!(!md.contains("\\eqqcolon"));
    }

    #[test]
    fn finalize_normalizes_latex_coloneqq_without_tags() {
        let raw = format!("x \\coloneqq y and a \\eqqcolon b{EOS_MARKER}");
        let md = finalize(&raw, 100, 100).unwrap();
        assert_eq!(md, "x := y and a =: b");
    }

    #[test]
    fn finalize_normalizes_latex_coloneqq_with_only_image_spans() {
        let raw =
            format!("x \\coloneqq y <|ref|>image<|/ref|><|det|>[0,0,10,10]<|/det|>{EOS_MARKER}");
        let md = finalize(&raw, 100, 100).unwrap();
        assert!(md.contains("x := y"));
        assert!(md.contains("![](images/0.jpg)"));
        assert!(!md.contains("\\coloneqq"));
    }

    #[test]
    fn finalize_multi_splits_and_rejoins_pages() {
        // leading text before first <PAGE> is dropped; two pages rejoined.
        let raw = format!("preamble<PAGE>page one text<PAGE>page two text{EOS_MARKER}");
        let md = finalize_multi(&raw, 2).unwrap();
        assert!(md.starts_with("<PAGE>\n"));
        assert!(md.contains("page one text"));
        assert!(md.contains("page two text"));
        assert!(!md.contains("preamble"));
        // exactly two page markers in the rejoined form
        assert_eq!(md.matches(PAGE_MARKER).count(), 2);
    }

    #[test]
    fn finalize_multi_per_page_image_prefix() {
        let raw = format!(
            "<PAGE>p0 <|ref|>image<|/ref|><|det|>[0,0,10,10]<|/det|><PAGE>p1 <|ref|>image<|/ref|><|det|>[0,0,10,10]<|/det|>{EOS_MARKER}"
        );
        let md = finalize_multi(&raw, 2).unwrap();
        assert!(md.contains("![](images/page_0_0.jpg)"));
        assert!(md.contains("![](images/page_1_0.jpg)"));
    }

    #[test]
    fn finalize_multi_passthrough_overflow_pages() {
        // 3 pages emitted but only 1 source image -> pages 1,2 pass through
        // verbatim (stripped only).
        let raw = format!(
            "<PAGE>real <|ref|>title<|/ref|><|det|>[1,2,3,4]<|/det|><PAGE>  extra <|ref|>x<|/ref|><|det|>[1,2,3,4]<|/det|>  {EOS_MARKER}"
        );
        let md = finalize_multi(&raw, 1).unwrap();
        // page 0 processed (title span removed)
        assert!(!md.contains("<|ref|>title"));
        // page 1 passthrough: its ref span text is preserved verbatim
        assert!(md.contains("<|ref|>x<|/ref|>"));
    }

    #[test]
    fn parse_layout_rescales_boxes() {
        let raw = format!("<|ref|>title<|/ref|><|det|>[0, 0, 999, 999]<|/det|>{EOS_MARKER}");
        let layout = parse_layout(&raw, 1920, 1080);
        // first entry is the full ref/det span labelled "title"
        let (label, boxes) = layout
            .iter()
            .find(|(l, _)| l == "title")
            .expect("title span");
        assert_eq!(label, "title");
        // 0->0, 999/999*W = W (truncated), 999/999*H = H
        assert_eq!(boxes, &vec![[0, 0, 1920, 1080]]);
    }

    #[test]
    fn re_match_empty_when_no_tags() {
        assert!(re_match("just plain markdown text\n# Heading").is_empty());
    }

    #[test]
    fn finalize_plain_text_passthrough() {
        // free-OCR style output: no ref/det tags, just prose -> unchanged body
        let raw = format!("# Title\n\nSome **markdown** body.\n{EOS_MARKER}");
        let md = finalize(&raw, 640, 480).unwrap();
        assert_eq!(md, "# Title\n\nSome **markdown** body.");
    }
}
