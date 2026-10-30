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
        let replacement = format!("{}\n", image_md_token(img_base, idx));
        out = out.replace(&m.full, &replacement);
    }
    for m in &others {
        out = out.replace(&m.full, "");
    }
    normalize_colon_equals(&out)
}

/// The exact markdown image token `assemble_page` emits for the `idx`-th image
/// span of a page (its trailing `\n` is added by the caller). The ONE place the
/// `images/{img_base}{idx}.jpg` path shape is defined, so the figure-extraction
/// enumerator ([`figure_refs`]) and the markdown writer can never disagree.
fn image_md_token(img_base: &str, idx: usize) -> String {
    format!("![](images/{img_base}{idx}.jpg)")
}

/// One image/figure span the model grounded but did NOT transcribe to text — the
/// regions [`finalize`] renders as `![](images/…)` placeholders. Surfaced so a
/// caller can crop the region out of the source image and write a real file
/// ([`OcrModel::recognize_with_figures`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FigureRef {
    /// 0-based index among the page's image spans — the `idx` in the markdown
    /// reference, matching [`assemble_page`]'s enumeration exactly.
    pub index: usize,
    /// The ref label (`image` for the standard figure span).
    pub label: String,
    /// Pixel boxes `[x1, y1, x2, y2]` rescaled to the source `(image_w, image_h)`.
    pub boxes: Vec<[i64; 4]>,
    /// The exact markdown token the rendered document uses for this figure
    /// (`![](images/{img_base}{index}.jpg)`), so a writer can string-replace it
    /// with a real `![alt](path)` reference without re-deriving the path.
    pub markdown_ref: String,
}

/// Enumerate a page's image/figure spans (the `is_image()` ref spans) in the SAME
/// order [`assemble_page`] indexes them, returning each one's pixel boxes (rescaled
/// to `(image_w, image_h)`) and the exact markdown token it appears as. `img_base`
/// must match the value passed to `assemble_page` for this page (`""` single-page,
/// `"page_{n}_"` multi-page) so `markdown_ref` lands on the right token.
///
/// This shares [`re_match`] + [`RefMatch::is_image`] with `assemble_page`, so the
/// figures a caller crops are byte-for-byte the ones the markdown references.
#[must_use]
pub fn figure_refs(decoded: &str, image_w: u32, image_h: u32, img_base: &str) -> Vec<FigureRef> {
    let stripped = strip_eos(decoded);
    re_match(&stripped)
        .into_iter()
        .filter(RefMatch::is_image)
        .enumerate()
        .map(|(index, m)| FigureRef {
            index,
            boxes: m.rescaled_boxes(image_w, image_h),
            label: m.label,
            markdown_ref: image_md_token(img_base, index),
        })
        .collect()
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
/// Strips the trailing EOS, `.strip()`, splits on [`PAGE_MARKER`],
/// postprocesses each page with prefix `page_{n}_`, then rejoins as
/// `"<PAGE>\n" + "\n<PAGE>\n".join(pages)`.
///
/// One deliberate departure from the reference (GH #17): the reference drops
/// the chunk before the first `<PAGE>` (`outputs.split('<PAGE>')[1:]`), which
/// is harmless when the model opens with a marker (the chunk is empty) but
/// silently discards the WHOLE transcription when it does not — the model
/// often omits the opening marker (a one-page pass commonly emits none at
/// all), and `focr --multi-page` then printed a bare `<PAGE>` with exit 0. A
/// non-blank leading chunk is therefore kept as the first page. When the model
/// does open with a marker the output is byte-identical to the reference.
///
/// `num_pages` is the count of source page images: pages with index `>= num_pages`
/// are passed through verbatim (only `.strip()`-ed), matching the reference
/// `if page_idx >= len(images): processed_pages.append(page_output); continue`.
///
/// # Errors
/// Infallible today; see [`finalize`] for why the `FocrResult` shape is kept.
pub fn finalize_multi(decoded: &str, num_pages: usize) -> FocrResult<String> {
    let stripped = strip_eos(decoded);
    let mut chunks = stripped.split(PAGE_MARKER);
    // The reference's `[1:]` drops the leading chunk; keep it as page 0 when
    // the model skipped the opening marker and it carries text (GH #17).
    let leading = chunks.next().filter(|c| !c.trim().is_empty());
    Ok(join_pages(leading.into_iter().chain(chunks), num_pages))
}

/// [`finalize_multi`]'s assembly over page bodies already split: each body
/// `.trim()`-ed and postprocessed with prefix `page_{n}_` (bodies at index
/// `>= num_pages` passed through), rejoined as `"<PAGE>\n" +
/// "\n<PAGE>\n".join(pages)`. Unlimited-OCR-2's bodies come from
/// [`split_pages_by_token`].
#[must_use]
pub fn finalize_pages<S: AsRef<str>>(bodies: &[S], num_pages: usize) -> String {
    join_pages(bodies.iter().map(AsRef::as_ref), num_pages)
}

fn join_pages<'a>(bodies: impl Iterator<Item = &'a str>, num_pages: usize) -> String {
    let mut processed = Vec::new();
    for (page_idx, page) in bodies.enumerate() {
        let page = page.trim();
        if page_idx >= num_pages {
            processed.push(page.to_string());
            continue;
        }
        let prefix = format!("page_{page_idx}_");
        processed.push(assemble_page(page, &prefix));
    }
    let body = processed.join("\n<PAGE>\n");
    format!("<PAGE>\n{body}")
}

/// Unlimited-OCR-2's output split into pages by token id (U2-B6, D20).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenPages {
    /// Raw body text per page the output reached, at most the page count.
    pub bodies: Vec<String>,
    /// Generated ids per body, its opening marker excluded.
    pub tokens: Vec<usize>,
    /// Non-blank text preceded the first marker and was prepended to body 1.
    pub leading_text: bool,
    /// Markers past the page count (DISC-U2-3).
    pub extra_markers: usize,
}

/// Split Unlimited-OCR-2's generated `ids` into page bodies on the marker
/// token id (D20), for a document of `pages` pages.
///
/// Each marker opens a page: body k is the text between the k-th and the
/// (k+1)-th markers, each decoded on its own with the UTF-8-safe
/// [`StreamDecoder`](crate::tokenizer::StreamDecoder), so a character split
/// just before a marker ends its body as U+FFFD, as HF's per-segment decode
/// does. Text before the first marker (a head) is prepended to body 1 and
/// flagged `leading_text` when non-blank (DISC-U2-9); HF drops it. Markers
/// past `pages` are counted in `extra_markers` (DISC-U2-3); what follows them
/// (nothing under the page-count guard, which then forces EOS) joins the last
/// body, so a complete decode yields exactly `pages` bodies.
///
/// # Errors
/// An id the tokenizer does not know ([`StreamDecoder::push`](crate::tokenizer::StreamDecoder::push)).
pub fn split_pages_by_token(
    tok: &crate::tokenizer::Tokenizer,
    ids: &[u32],
    marker: u32,
    pages: usize,
) -> FocrResult<TokenPages> {
    let (head, mut segments) = token_segments(tok, ids, marker, false)?;
    let extra_markers = segments.len().saturating_sub(pages);
    if segments.len() > pages.max(1) {
        let extras = segments.split_off(pages.max(1));
        let last = segments.last_mut().expect("pages.max(1) >= 1");
        for (text, count) in extras {
            if !text.trim().is_empty() {
                last.0.push('\n');
                last.0.push_str(&text);
            }
            last.1 += count;
        }
    }
    Ok(join_head(head, segments, extra_markers))
}

/// One Unlimited-OCR-2 window's visible `ids` split into the bodies of its
/// `own` pages (U2-D1a; plan §5.10.8): [`split_pages_by_token`]'s split, a
/// non-blank head joining page 1 (DISC-U2-9), except that every body past
/// `own` is dropped, as the wheel's front-anchored ownership drops it
/// (W/`entrypoints/openai/doc_split.py:236-257`), and no leading blank body
/// is (DISC-U2-4). `leads_with_body`: the ids open inside page 1 (a reused
/// window, whose opening `<PAGE>` is forced and hidden). `lookahead`: the
/// window reads a lookahead page and ends on the marker that opens it, which
/// is not an extra marker.
///
/// # Errors
/// As [`split_pages_by_token`].
pub fn split_window_pages(
    tok: &crate::tokenizer::Tokenizer,
    ids: &[u32],
    marker: u32,
    own: usize,
    leads_with_body: bool,
    lookahead: bool,
) -> FocrResult<TokenPages> {
    let (head, mut segments) = token_segments(tok, ids, marker, leads_with_body)?;
    let expected_markers = own + usize::from(lookahead);
    let extra_markers = segments.len().saturating_sub(expected_markers);
    segments.truncate(own);
    Ok(join_head(head, segments, extra_markers))
}

/// One page's decoded text and its id count.
type Segment = (String, usize);

/// `ids` split on `marker`: the head before the first marker, then one
/// `(text, ids)` segment per marker, each decoded on its own with the
/// UTF-8-safe [`StreamDecoder`](crate::tokenizer::StreamDecoder). With
/// `leads_with_body` the ids open the first segment and the head is empty.
fn token_segments(
    tok: &crate::tokenizer::Tokenizer,
    ids: &[u32],
    marker: u32,
    leads_with_body: bool,
) -> FocrResult<(Segment, Vec<Segment>)> {
    let mut decoder = crate::tokenizer::StreamDecoder::new();
    // Segment 0 is the head; segment k >= 1 follows the k-th marker.
    let mut segments = vec![(String::new(), 0usize)];
    if leads_with_body {
        segments.push((String::new(), 0));
    }
    for &id in ids {
        let (text, count) = segments.last_mut().expect("never empty");
        if id == marker {
            decoder.flush(text);
            segments.push((String::new(), 0));
        } else {
            decoder.push(tok, id, text)?;
            *count += 1;
        }
    }
    decoder.flush(&mut segments.last_mut().expect("never empty").0);
    let head = segments.remove(0);
    Ok((head, segments))
}

/// The bodies `segments` with a non-blank `head` prepended to the first
/// (DISC-U2-9) and its ids counted there.
fn join_head(
    (head, head_tokens): Segment,
    mut segments: Vec<Segment>,
    extra_markers: usize,
) -> TokenPages {
    let leading_text = !head.trim().is_empty();
    match segments.first_mut() {
        Some(first) => {
            if leading_text {
                first.0.insert_str(0, &head);
            }
            first.1 += head_tokens;
        }
        None if leading_text || head_tokens > 0 => segments.push((head, head_tokens)),
        None => {}
    }
    let (bodies, tokens) = segments.into_iter().unzip();
    TokenPages {
        bodies,
        tokens,
        leading_text,
        extra_markers,
    }
}

/// [`PageStream`] for Unlimited-OCR-2 (U2-B6): fed the generated ids one at a
/// time, it emits each page's `.trim()`-ed body, 1-based, once the next
/// marker arrives or the decode ends, under [`split_pages_by_token`]'s rule:
/// a non-blank head joins page 1, and nothing past the page count is emitted
/// (the guard ends the decode there). So its bodies are the split's, trimmed.
#[derive(Debug)]
pub struct TokenPageStream<'t> {
    tok: &'t crate::tokenizer::Tokenizer,
    marker: u32,
    pages: usize,
    decoder: crate::tokenizer::StreamDecoder,
    current: String,
    markers: usize,
    emitted: usize,
}

impl<'t> TokenPageStream<'t> {
    /// A stream over a `pages`-page document whose pages `marker` opens.
    #[must_use]
    pub fn new(tok: &'t crate::tokenizer::Tokenizer, marker: u32, pages: usize) -> Self {
        Self {
            tok,
            marker,
            pages,
            decoder: crate::tokenizer::StreamDecoder::new(),
            current: String::new(),
            markers: 0,
            emitted: 0,
        }
    }

    /// Take the next generated id; emit the page it completes, if any. An id
    /// the tokenizer does not know adds no text (the terminal split reports
    /// it).
    pub fn push(&mut self, id: u32, mut on_page: impl FnMut(usize, &str)) {
        if id != self.marker {
            let _ = self.decoder.push(self.tok, id, &mut self.current);
            return;
        }
        self.decoder.flush(&mut self.current);
        self.markers += 1;
        if self.markers == 1 {
            // The head joins page 1 when it has text.
            if self.current.trim().is_empty() {
                self.current.clear();
            }
        } else if self.emitted < self.pages {
            self.emitted += 1;
            on_page(self.emitted, self.current.trim());
            self.current.clear();
        }
    }

    /// End of decode: emit the page in flight.
    pub fn finish(mut self, mut on_page: impl FnMut(usize, &str)) {
        self.decoder.flush(&mut self.current);
        let started = self.markers > 0 || !self.current.trim().is_empty();
        if started && self.emitted < self.pages.max(1) {
            on_page(self.emitted + 1, self.current.trim());
        }
    }
}

/// The page bodies a multi-page decode carries under [`finalize_multi`]'s
/// split: one per `<PAGE>` marker, plus a non-blank leading chunk when the
/// model skipped the opening marker (GH #17). `text` is the decoded output
/// with its EOS stripped. The one count every multi-page report uses (U2-X2
/// warnings, U2-E3 `pages[]`).
#[must_use]
pub fn page_body_count(text: &str) -> usize {
    let mut chunks = text.split(PAGE_MARKER);
    let leading = chunks.next().is_some_and(|c| !c.trim().is_empty());
    usize::from(leading) + chunks.count()
}

/// Incremental `<PAGE>`-boundary scanner for STREAMING multi-page decodes
/// (bd-2z0y): fed the full decoded-so-far text, it emits each page body as
/// soon as the NEXT page's marker arrives (page k is complete when marker
/// k+1 appears; the final page completes at end-of-decode via
/// [`PageStream::finish`]).
///
/// Pure string logic — the token→text decode happens in the caller — so the
/// boundary semantics are unit-tested without a tokenizer. Mirrors
/// [`finalize_multi`]'s split contract: non-blank text before the FIRST
/// marker is page 1 (the model skipped the opening marker, GH #17), blank
/// pre-marker text is ignored, and streamed bodies are `.trim()`-ed raw model
/// text (the polished per-page markdown still comes
/// from the terminal [`finalize_multi`] assembly).
#[derive(Debug, Default)]
pub struct PageStream {
    /// Byte offset just past the last CONSUMED marker (the current page's
    /// body starts here). `None` until the first marker arrives.
    body_start: Option<usize>,
    /// 1-based index of the page currently being decoded.
    current_page: usize,
}

impl PageStream {
    /// Fresh scanner (no marker seen yet).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Scan `full_text` (the ENTIRE decoded text so far — the caller may call
    /// this as often or as rarely as it likes; the scanner is idempotent over
    /// already-seen prefixes) and emit every newly COMPLETED page body.
    pub fn feed(&mut self, full_text: &str, mut on_page: impl FnMut(usize, &str)) {
        loop {
            match self.body_start {
                None => {
                    // Waiting for the FIRST marker (page 1's start).
                    let Some(pos) = full_text.find(PAGE_MARKER) else {
                        return;
                    };
                    // Non-blank text before the first marker is page 1
                    // (the model skipped the opening marker, GH #17).
                    let leading = full_text[..pos].trim();
                    self.current_page = if leading.is_empty() {
                        1
                    } else {
                        on_page(1, leading);
                        2
                    };
                    self.body_start = Some(pos + PAGE_MARKER.len());
                }
                Some(start) => {
                    let Some(rel) = full_text[start..].find(PAGE_MARKER) else {
                        return;
                    };
                    let end = start + rel;
                    on_page(self.current_page, full_text[start..end].trim());
                    self.body_start = Some(end + PAGE_MARKER.len());
                    self.current_page += 1;
                }
            }
        }
    }

    /// End of decode: emit the final in-flight page (marker seen, no
    /// successor marker). A decode that never emitted a marker emits its whole
    /// text as page 1 when non-blank (matching [`finalize_multi`], GH #17).
    pub fn finish(self, full_text: &str, mut on_page: impl FnMut(usize, &str)) {
        match self.body_start {
            Some(start) => on_page(self.current_page, full_text[start..].trim()),
            None => {
                let body = full_text.trim();
                if !body.is_empty() {
                    on_page(1, body);
                }
            }
        }
    }
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

    /// U2-X2: the body count follows `finalize_multi`'s split exactly.
    #[test]
    fn page_body_count_matches_the_finalize_split() {
        for (text, bodies) in [
            ("", 0),
            ("   ", 0),
            ("<PAGE>a", 1),
            ("<PAGE>a<PAGE>b<PAGE>c", 3),
            ("lead<PAGE>b<PAGE>c", 3),
            ("  \n<PAGE>a<PAGE>b", 2),
            ("no marker at all", 1),
            ("<PAGE>a<PAGE>", 2),
        ] {
            assert_eq!(page_body_count(text), bodies, "{text:?}");
            let assembled = finalize_multi(text, 99).unwrap();
            if bodies > 0 {
                assert_eq!(assembled.matches(PAGE_MARKER).count(), bodies, "{text:?}");
            }
        }
    }

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
    fn page_stream_streams_bodies_as_markers_arrive() {
        let mut ps = PageStream::new();
        let mut got: Vec<(usize, String)> = Vec::new();
        // Incremental growth: blank text before the first marker is ignored;
        // page 1 lands only when marker 2 arrives; idempotent over re-fed
        // prefixes.
        ps.feed(" \n", |i, s| got.push((i, s.to_string())));
        assert!(got.is_empty());
        ps.feed(" \n<PAGE>\nalpha", |i, s| got.push((i, s.to_string())));
        assert!(got.is_empty(), "page 1 is still in flight");
        let text = " \n<PAGE>\nalpha\n<PAGE>\nbeta";
        ps.feed(text, |i, s| got.push((i, s.to_string())));
        assert_eq!(got, vec![(1, "alpha".to_string())]);
        ps.feed(text, |i, s| got.push((i, s.to_string())));
        assert_eq!(got.len(), 1, "re-feeding the same text must not re-emit");
        ps.finish(text, |i, s| got.push((i, s.to_string())));
        assert_eq!(
            got,
            vec![(1, "alpha".to_string()), (2, "beta".to_string())],
            "finish flushes the final in-flight page"
        );
        println!(r#"{{"check":"page_stream_boundaries","pages":2,"result":"pass"}}"#);
    }

    #[test]
    fn page_stream_without_markers_streams_whole_text_as_page_one() {
        // GH #17: a decode with no marker at all is one page, not nothing.
        let mut ps = PageStream::new();
        let mut got: Vec<(usize, String)> = Vec::new();
        ps.feed("no markers at all", |i, s| got.push((i, s.to_string())));
        assert!(got.is_empty(), "nothing streams before end of decode");
        ps.finish(" no markers at all \n", |i, s| got.push((i, s.to_string())));
        assert_eq!(got, vec![(1, "no markers at all".to_string())]);

        // A blank decode still streams nothing.
        let mut blank = 0usize;
        PageStream::new().finish(" \n ", |_, _| blank += 1);
        assert_eq!(blank, 0);
    }

    #[test]
    fn page_stream_text_before_first_marker_is_page_one() {
        // GH #17: the model skipped the opening marker; the pre-marker text is
        // page 1 and the marker opens page 2.
        let mut ps = PageStream::new();
        let mut got: Vec<(usize, String)> = Vec::new();
        ps.feed("first page", |i, s| got.push((i, s.to_string())));
        assert!(got.is_empty());
        let text = "first page\n<PAGE>\nsecond page";
        ps.feed(text, |i, s| got.push((i, s.to_string())));
        assert_eq!(got, vec![(1, "first page".to_string())]);
        ps.feed(text, |i, s| got.push((i, s.to_string())));
        assert_eq!(got.len(), 1, "re-feeding must not re-emit the leading page");
        ps.finish(text, |i, s| got.push((i, s.to_string())));
        assert_eq!(
            got,
            vec![
                (1, "first page".to_string()),
                (2, "second page".to_string())
            ]
        );
    }

    #[test]
    fn page_stream_marker_split_across_feeds_is_caught() {
        // The caller re-feeds the FULL text, so a marker that was previously
        // truncated mid-bytes ("<PA") completes on a later feed.
        let mut ps = PageStream::new();
        let mut got: Vec<usize> = Vec::new();
        ps.feed("<PAGE>one <PA", |i, _| got.push(i));
        assert!(got.is_empty());
        ps.feed("<PAGE>one <PAGE>two", |i, _| got.push(i));
        assert_eq!(got, vec![1]);
    }

    #[test]
    fn finalize_multi_splits_and_rejoins_pages() {
        // Blank text before the first <PAGE> is dropped; two pages rejoined.
        let raw = format!(" \n<PAGE>page one text<PAGE>page two text{EOS_MARKER}");
        let md = finalize_multi(&raw, 2).unwrap();
        assert_eq!(md, "<PAGE>\npage one text\n<PAGE>\npage two text");
    }

    #[test]
    fn finalize_multi_keeps_text_when_model_emits_no_marker() {
        // GH #17: a one-page pass whose decode carries no <PAGE> at all used to
        // assemble to a bare "<PAGE>\n" — the whole transcription dropped.
        let raw = format!("INVOICE #10042\nTotal due: 104.49{EOS_MARKER}");
        let md = finalize_multi(&raw, 1).unwrap();
        assert_eq!(md, "<PAGE>\nINVOICE #10042\nTotal due: 104.49");
    }

    #[test]
    fn finalize_multi_keeps_text_before_first_marker_as_page_zero() {
        // GH #17: the model skipped the opening marker but separated later
        // pages; the leading chunk is page 0 (image prefix page_0_), the
        // marker opens page 1.
        let raw = format!(
            "p0 <|ref|>image<|/ref|><|det|>[0,0,10,10]<|/det|><PAGE>p1 <|ref|>image<|/ref|><|det|>[0,0,10,10]<|/det|>{EOS_MARKER}"
        );
        let md = finalize_multi(&raw, 2).unwrap();
        assert!(md.starts_with("<PAGE>\np0"));
        assert!(md.contains("![](images/page_0_0.jpg)"));
        assert!(md.contains("![](images/page_1_0.jpg)"));
        assert_eq!(md.matches(PAGE_MARKER).count(), 2);
    }

    #[test]
    fn finalize_multi_blank_decode_is_a_bare_marker() {
        let md = finalize_multi(&format!("  {EOS_MARKER}"), 1).unwrap();
        assert_eq!(md, "<PAGE>\n");
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

    /// U2-B6's D20 split on a synthetic byte-level stream: a CJK character cut
    /// by a marker, an extra trailing marker, and a leading non-blank head.
    #[test]
    fn split_pages_by_token_keeps_n_bodies_and_reports_head_and_extras() {
        use crate::tokenizer::special::PAGE;
        use crate::tokenizer::tests::{byte_id, byte_tokenizer};
        let t = byte_tokenizer();
        let text = |s: &str| s.bytes().map(byte_id).collect::<Vec<_>>();
        let zhong = "中".as_bytes();
        let ids = [
            text("hi "),
            vec![PAGE],
            text("one "),
            vec![byte_id(zhong[0]), byte_id(zhong[1])],
            vec![PAGE],
            text("two"),
            vec![PAGE],
        ]
        .concat();
        let split = split_pages_by_token(&t, &ids, PAGE, 2).unwrap();
        assert_eq!(
            split.bodies,
            ["hi one \u{FFFD}", "two"],
            "the head joins body 1; the cut character ends it as U+FFFD"
        );
        assert_eq!(split.tokens, [3 + 4 + 2, 3]);
        assert!(split.leading_text);
        assert_eq!(split.extra_markers, 1, "the empty trailing body is dropped");

        // The same stream rendered: two pages, the v1 multi-page shape.
        let md = finalize_pages(&split.bodies, 2);
        assert_eq!(md, "<PAGE>\nhi one \u{FFFD}\n<PAGE>\ntwo");
        // finalize_multi over the text form gives the same for marker-led text.
        assert_eq!(
            finalize_multi("<PAGE>a\n<PAGE>b", 2).unwrap(),
            finalize_pages(&["a", "b"], 2)
        );

        // A marker-led, blank-headed decode: no warning, exactly N bodies.
        let ids = [text("\n"), vec![PAGE], text("x"), vec![PAGE], text("y")].concat();
        let split = split_pages_by_token(&t, &ids, PAGE, 2).unwrap();
        assert_eq!(split.bodies, ["x", "y"]);
        assert_eq!(
            split.tokens,
            [2, 1],
            "the blank head's token counts toward page 1"
        );
        assert!(!split.leading_text);
        assert_eq!(split.extra_markers, 0);

        // Text after an extra marker is kept on the last page, never dropped.
        let ids = [vec![PAGE], text("a"), vec![PAGE], text("z")].concat();
        let split = split_pages_by_token(&t, &ids, PAGE, 1).unwrap();
        assert_eq!(split.bodies, ["a\nz"]);
        assert_eq!(split.extra_markers, 1);

        // A decode stopped before its first marker still yields page 1.
        let split = split_pages_by_token(&t, &text("abc"), PAGE, 3).unwrap();
        assert_eq!(split.bodies, ["abc"]);
        assert!(split.leading_text);
        // Nothing generated: no bodies.
        assert!(
            split_pages_by_token(&t, &[], PAGE, 3)
                .unwrap()
                .bodies
                .is_empty()
        );
        // An unknown id is an error.
        assert!(split_pages_by_token(&t, &[7], PAGE, 1).is_err());
    }

    /// U2-D1a: a window keeps exactly its own pages' bodies. A recomputed
    /// window's non-blank head joins page 1 (DISC-U2-9); a reused window's
    /// ids open inside page 1; the marker opening the lookahead page and
    /// what follows are dropped without counting as extra; a blank body
    /// stays a page (DISC-U2-4); on a last window a marker past its pages is
    /// extra and its body dropped.
    #[test]
    fn split_window_pages_keeps_own_pages_only() {
        use crate::tokenizer::special::PAGE;
        use crate::tokenizer::tests::{byte_id, byte_tokenizer};
        let t = byte_tokenizer();
        let text = |s: &str| s.bytes().map(byte_id).collect::<Vec<_>>();
        let pg = || vec![PAGE];

        let ids = [
            text("h "),
            pg(),
            text("a"),
            pg(),
            text("b"),
            pg(),
            text("la"),
        ]
        .concat();
        let split = split_window_pages(&t, &ids, PAGE, 2, false, true).unwrap();
        assert_eq!(split.bodies, ["h a", "b"]);
        assert_eq!(split.tokens, [3, 1]);
        assert!(split.leading_text);
        assert_eq!(split.extra_markers, 0, "the lookahead's marker is the end");

        let ids = [text("a"), pg(), text("b"), pg()].concat();
        let split = split_window_pages(&t, &ids, PAGE, 2, true, true).unwrap();
        assert_eq!(split.bodies, ["a", "b"]);
        assert!(!split.leading_text);
        assert_eq!(split.extra_markers, 0);

        let ids = [text(" "), pg(), pg(), text("c"), pg()].concat();
        let split = split_window_pages(&t, &ids, PAGE, 3, true, true).unwrap();
        assert_eq!(split.bodies, [" ", "", "c"], "blank bodies keep alignment");

        let ids = [pg(), text(" "), pg(), text("b"), pg(), text("z")].concat();
        let split = split_window_pages(&t, &ids, PAGE, 2, false, false).unwrap();
        assert_eq!(split.bodies, [" ", "b"], "no leading-blank drop");
        assert_eq!(split.extra_markers, 1, "a last window's extra marker");

        let split = split_window_pages(&t, &text("ab"), PAGE, 2, true, true).unwrap();
        assert_eq!(split.bodies, ["ab"], "stopped inside page 1");
    }

    /// The streamed v2 page events are the split's bodies, trimmed, on every
    /// guard-shaped stream: up to one extra marker, which ends the decode.
    #[test]
    fn token_page_stream_emits_the_split_bodies() {
        use crate::tokenizer::special::PAGE;
        use crate::tokenizer::tests::{byte_id, byte_tokenizer};
        let t = byte_tokenizer();
        let alphabet: Vec<u32> = [b'a', b' ', b'\n', 0xE4, 0xB8, 0xAD]
            .into_iter()
            .map(byte_id)
            .collect();
        let mut state = 0x005e_edb6_u64;
        let mut next = |n: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % n
        };
        for _ in 0..2000 {
            let pages = 1 + next(4) as usize;
            let mut ids = Vec::new();
            let mut markers = 0;
            for _ in 0..next(40) {
                if next(6) == 0 && markers <= pages {
                    ids.push(PAGE);
                    markers += 1;
                    if markers > pages {
                        break;
                    }
                } else {
                    ids.push(alphabet[next(alphabet.len() as u64) as usize]);
                }
            }
            let split = split_pages_by_token(&t, &ids, PAGE, pages).unwrap();
            let mut stream = TokenPageStream::new(&t, PAGE, pages);
            let mut got: Vec<(usize, String)> = Vec::new();
            for &id in &ids {
                stream.push(id, |k, body| got.push((k, body.to_string())));
            }
            stream.finish(|k, body| got.push((k, body.to_string())));
            let want: Vec<(usize, String)> = split
                .bodies
                .iter()
                .enumerate()
                .filter(|(_, b)| markers > 0 || !b.trim().is_empty())
                .map(|(i, b)| (i + 1, b.trim().to_string()))
                .collect();
            assert_eq!(got, want, "{pages} pages: {ids:?}");
        }
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

    #[test]
    fn figure_refs_enumerate_image_spans_with_tokens_matching_assemble_page() {
        // A non-image span (dropped) + two image spans (the figures), with DISTINCT
        // boxes (real pages never repeat a box; identical spans would collapse under
        // assemble_page's Python `str.replace` semantics). The full-frame box
        // `[0,0,999,999]` rescales EXACTLY to the image dims.
        let decoded = concat!(
            "<|ref|>title<|/ref|><|det|>[[0,0,500,500]]<|/det|>",
            "<|ref|>image<|/ref|><|det|>[[0,0,999,999]]<|/det|>",
            "<|ref|>image<|/ref|><|det|>[[0,0,499,499]]<|/det|>",
        );
        let refs = figure_refs(decoded, 200, 100, "");
        assert_eq!(refs.len(), 2, "only the two image spans are figures");
        assert_eq!(refs[0].index, 0);
        assert_eq!(refs[0].label, "image");
        assert_eq!(refs[0].markdown_ref, "![](images/0.jpg)");
        assert_eq!(refs[0].boxes, vec![[0, 0, 200, 100]]);
        assert_eq!(refs[1].index, 1);
        assert_eq!(refs[1].markdown_ref, "![](images/1.jpg)");

        // The tokens are EXACTLY what the rendered markdown contains.
        let md = finalize(decoded, 200, 100).unwrap();
        assert!(md.contains(&refs[0].markdown_ref), "md: {md}");
        assert!(md.contains(&refs[1].markdown_ref), "md: {md}");
    }

    #[test]
    fn figure_refs_multipage_prefix_matches_assemble_page() {
        let decoded = "<|ref|>image<|/ref|><|det|>[[1,2,3,4]]<|/det|>";
        let refs = figure_refs(decoded, 999, 999, "page_0_");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].markdown_ref, "![](images/page_0_0.jpg)");
        // Matches the multi-page markdown emitter.
        let md = finalize_multi(&format!("<PAGE>\n{decoded}"), 1).unwrap();
        assert!(md.contains("![](images/page_0_0.jpg)"), "md: {md}");
    }

    #[test]
    fn figure_refs_empty_without_image_spans() {
        let decoded = "<|ref|>title<|/ref|><|det|>[[0,0,9,9]]<|/det|>plain text";
        assert!(figure_refs(decoded, 100, 100, "").is_empty());
    }
}
