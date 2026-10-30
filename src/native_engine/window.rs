//! Unlimited-OCR-2's windowed document decode (U2-D1a; plan §5.10).
//!
//! A document longer than one window decodes as a run of windows of `W` own
//! pages (D13: 10). Every window but the last also reads the next window's
//! first page, its lookahead, and ends on the `<PAGE>` that would open it.
//! Window 0, and any window whose reuse falls back, prefills its full prompt
//! at page base `lo`. Every other window is *reused*, as the SGLang wheel's
//! copy-splice does it (W/`session/uocr_doc_reuse.py`, "DR" below): its KV
//! opens with rows copied from the previous window (the tail page's marker,
//! its 128–159-row image tail, then the lookahead page whole), prefills only
//! its new pages after them, splices in the previous window's last ≤ 128
//! text rows, and decodes from a forced `<PAGE>` at page base `lo − 1`.
//! Copied keys are never re-rotated: the page base keeps every row's page
//! parity ([`srswa::rope_page`]).
//!
//! The planner, the tail plan, the text tail, the reuse checks and the
//! stitching are pure functions here; [`OcrModel::decode_document_windowed`]
//! runs them over the model, one window and one forward at a time
//! (doctrine #5).

use std::ops::Range;
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use super::tensor::Mat;
use super::{
    DecodeParams, DecodeReport, DocumentDecode, DocumentDecoder, DocumentLoop, DocumentPass,
    DocumentPrecision, DocumentStats, FinishReason, LengthCap, MULTI_PAGE_PROMPT_TEXT, OcrModel,
    PageBodies, PageFinish, UOCR2_UNWINDOWED_PAGE_LIMIT, connector, decode_with_retire_fallback,
    decoder, document_guard, lever_policy, paged_prompt, postprocess, sampler, srswa, timing_log,
};
use crate::error::{FocrError, FocrResult};
use crate::preprocess::Preprocessed;
use crate::tokenizer::special::{IMAGE, PAGE};

/// Own pages per window by default (D13; the wheel's own default is 1,
/// W/`environ.py:1354-1358`).
pub const DEFAULT_PAGE_WINDOW: usize = 10;

/// The planner's prompt rows per base page (W/`entrypoints/openai/
/// doc_split.py:28-35`): its marker and 273 image rows.
pub const PER_PAGE_EST_BASE: usize = 274;

/// The planner's prompt rows per gundam page (`doc_split.py:28-35`).
pub const PER_PAGE_EST_GUNDAM: usize = 1514;

/// The context the planner fits a window's prompt pages in.
pub const PLAN_CONTEXT: usize = 131_072;

/// Rows the planner holds back: `fits = (ctx − 65) // per_page`
/// (`doc_split.py:141-143`).
const PLAN_RESERVE: usize = 65;

/// Rows of image tail and of text tail a reused window copies: the 128-row
/// local window every query reads (DR:191-193).
pub const TAIL_TOKENS: usize = srswa::LOCAL_WINDOW;

/// Pages a reused window's prompt carries as context only: the tail page
/// (DR:195-197).
pub const CONTEXT_PAGES: usize = 1;

// ── Planning (§5.10.1) ──────────────────────────────────────────────────────

/// One window of a plan: own pages `lo..hi` (global, 0-based) and whether it
/// also reads page `hi`, the next window's first, as its lookahead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowSpan {
    /// Position in the plan.
    pub index: usize,
    /// First own page.
    pub lo: usize,
    /// One past the last own page.
    pub hi: usize,
    /// Every window but the last reads one lookahead page.
    pub lookahead: bool,
}

impl WindowSpan {
    /// Own pages.
    #[must_use]
    pub fn own(&self) -> usize {
        self.hi - self.lo
    }

    /// Pages its full prompt carries: own plus the lookahead.
    #[must_use]
    pub fn prompt_pages(&self) -> usize {
        self.own() + usize::from(self.lookahead)
    }
}

/// Own pages per window: `min((131,072 − 65) // per_page_est, window,
/// page_limit)`, at least 1. The wheel bounds only the prompt
/// (`doc_split.py:141-143`); outputs share the window's positions, so U2-B6's
/// unwindowed page limit is what keeps prompt, lookahead and output inside
/// one window.
#[must_use]
pub fn own_pages_per_window(window: usize, per_page_est: usize, page_limit: usize) -> usize {
    ((PLAN_CONTEXT - PLAN_RESERVE) / per_page_est.max(1))
        .min(window)
        .min(page_limit)
        .max(1)
}

/// The windows of a `pages`-page document, or `None` when it fits one window
/// and decodes unwindowed (the wheel never splits a document that fits,
/// `doc_split.py:150-156`). Windows are contiguous and own their pages; each
/// but the last reads the next one's first page.
#[must_use]
pub fn plan_windows(
    pages: usize,
    window: usize,
    per_page_est: usize,
    page_limit: usize,
) -> Option<Vec<WindowSpan>> {
    let own = own_pages_per_window(window, per_page_est, page_limit);
    if pages <= own {
        return None;
    }
    let spans = (0..pages)
        .step_by(own)
        .enumerate()
        .map(|(index, lo)| {
            let hi = (lo + own).min(pages);
            WindowSpan {
                index,
                lo,
                hi,
                lookahead: hi < pages,
            }
        })
        .collect();
    Some(spans)
}

/// Window `index` of a document read as its pages stream in (U2-D1b), whose
/// length is not known up front: a page that fails to rasterize is found
/// only when it is read, and the plan counts decodable pages only (D24).
/// Starting at page `lo` with `held` of its pages in hand (a carried
/// lookahead), the window pulls pages until it owns `own` or the stream
/// ends, then one more as its lookahead. `pull(page)` takes page `page`
/// (global, 0-based) in, or returns `false` once the stream has ended. Over a
/// stream of `n` pages the spans are [`plan_windows`]'s for `n`.
///
/// # Errors
/// `pull`'s.
pub fn stream_span(
    index: usize,
    lo: usize,
    own: usize,
    held: usize,
    pull: &mut dyn FnMut(usize) -> FocrResult<bool>,
) -> FocrResult<WindowSpan> {
    let mut have = held;
    while have < own && pull(lo + have)? {
        have += 1;
    }
    let hi = lo + have;
    let lookahead = have == own && pull(hi)?;
    Ok(WindowSpan {
        index,
        lo,
        hi,
        lookahead,
    })
}

// ── Prompt layout and the tail plan (§5.10.3–5.10.4) ────────────────────────

/// The rows of a window prompt `prefix (<PAGE> <image>…)… instruction`
/// (DR:152-189).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptPages {
    /// Rows before the first marker (the BOS; none in a rewritten prompt).
    pub prefix_len: usize,
    /// Each page's marker row.
    pub page_starts: Vec<usize>,
    /// One past each page's last image row.
    pub page_ends: Vec<usize>,
    /// The instruction's first row (the prompt's length when it has none).
    pub instr_start: usize,
}

impl PromptPages {
    /// Pages in the prompt.
    #[must_use]
    pub fn pages(&self) -> usize {
        self.page_starts.len()
    }
}

/// Parse a window prompt (DR:152-189): no image before the first marker,
/// every marker followed by at least one image row, and no image in the
/// instruction. `None` otherwise.
#[must_use]
pub fn parse_prompt(ids: &[u32]) -> Option<PromptPages> {
    let n = ids.len();
    let mut pos = ids.iter().position(|&t| t == PAGE || t == IMAGE)?;
    if ids[pos] == IMAGE {
        return None;
    }
    let prefix_len = pos;
    let (mut page_starts, mut page_ends) = (Vec::new(), Vec::new());
    while pos < n && ids[pos] == PAGE {
        page_starts.push(pos);
        pos += 1;
        let first_image = pos;
        while pos < n && ids[pos] == IMAGE {
            pos += 1;
        }
        if pos == first_image {
            return None;
        }
        page_ends.push(pos);
    }
    if ids[pos..].contains(&IMAGE) {
        return None;
    }
    Some(PromptPages {
        prefix_len,
        page_starts,
        page_ends,
        instr_start: pos,
    })
}

/// The first row of the image tail a successor copies of a page whose image
/// rows end at `end`: `end − 128 − end mod 32`, block-aligned, 128–159 rows
/// before `end` (DR:256-268). `None` when that is before row 0.
#[must_use]
pub fn tail_start(end: usize) -> Option<usize> {
    end.checked_sub(TAIL_TOKENS + end % srswa::BLOCK_SIZE)
}

/// How a reused window copies its predecessor's tail (DR:199-301).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TailPlan {
    /// The old window's row of the tail page's marker, copied first.
    pub marker_src: usize,
    /// The old window's rows copied after it: the image tail, then the
    /// lookahead page when it is copied.
    pub copy_src: Range<usize>,
    /// Rows copied: the marker plus `copy_src`.
    pub copy_len: usize,
    /// Where the new window's full prompt resumes after the copied rows.
    pub kept_from: usize,
    /// The rewritten prompt: `<PAGE>`, the copied rows' ids, then the full
    /// prompt's rows from `kept_from` up to its instruction. No BOS and no
    /// instruction (DR:201-214).
    pub new_ids: Vec<u32>,
    /// The old window's pages up to its tail page.
    pub old_pages: usize,
    /// The new window's prompt pages, lookahead included.
    pub new_pages: usize,
    /// Leading pages of the new window copied from the old lookahead (0 or 1).
    pub copied_pages: usize,
    /// The one-page variant: the copied lookahead page's last image row is
    /// prefilled again, since an extend needs a row to run (DR:284-287).
    pub straddle: bool,
}

/// Plan a reused window's copy of the old window's tail (DR:242-301, with
/// no overlap, which a contiguous plan never has): `old_ids` is the old
/// window's prompt as it ran, whose last `old_lookahead` pages it only read;
/// `new_ids` the new window's full prompt. With `copy_lookahead` the old
/// lookahead page, which is the new window's first page, is copied whole
/// after the tail, except its last image row when the new prompt has one
/// page. `None` when the layout cannot be reused: an unparseable prompt, no
/// own page, a tail page too short for its image tail, or a lookahead page
/// that is not the new first page (then nothing of it is copied).
#[must_use]
pub fn plan_tail_copy(
    old_ids: &[u32],
    new_ids: &[u32],
    old_lookahead: usize,
    copy_lookahead: bool,
) -> Option<TailPlan> {
    let old = parse_prompt(old_ids)?;
    let new = parse_prompt(new_ids)?;
    let p_old = old.pages().checked_sub(old_lookahead)?;
    let p_new = new.pages();
    if p_old < 1 || p_new < 1 {
        return None;
    }
    let end = old.page_ends[p_old - 1];
    let marker_src = old.page_starts[p_old - 1];
    let start = tail_start(end).filter(|&s| s > marker_src)?;
    let (mut copy_end, mut kept_from, mut copied_pages, mut straddle) =
        (end, new.prefix_len, 0, false);
    if copy_lookahead && old_lookahead > 0 {
        let lookahead = old.page_starts[p_old]..old.page_ends[p_old];
        let first = new.page_starts[0]..new.page_ends[0];
        if lookahead.start == end && old_ids[lookahead.clone()] == new_ids[first.clone()] {
            (copy_end, kept_from, copied_pages) = (lookahead.end, first.end, 1);
            if p_new == 1 {
                (copy_end, kept_from, straddle) = (lookahead.end - 1, first.end - 1, true);
            }
        }
    }
    let mut rewritten = Vec::with_capacity(1 + copy_end - start + new.instr_start - kept_from);
    rewritten.push(PAGE);
    rewritten.extend_from_slice(&old_ids[start..copy_end]);
    let copy_len = rewritten.len();
    rewritten.extend_from_slice(&new_ids[kept_from..new.instr_start]);
    Some(TailPlan {
        marker_src,
        copy_src: start..copy_end,
        copy_len,
        kept_from,
        new_ids: rewritten,
        old_pages: p_old,
        new_pages: p_new,
        copied_pages,
        straddle,
    })
}

/// Up to `n` trailing ids of `output`'s last page, never a marker
/// (DR:304-321): `(start, ids)`, cut after the last marker so a copied
/// marker never replays into the new frame. `None` for no output or `n = 0`.
#[must_use]
pub fn text_tail(output: &[u32], n: usize) -> Option<(usize, &[u32])> {
    if output.is_empty() || n == 0 {
        return None;
    }
    let after_marker = output.iter().rposition(|&t| t == PAGE).map_or(0, |i| i + 1);
    let start = after_marker.max(output.len().saturating_sub(n));
    Some((start, &output[start..]))
}

/// `output` up to the marker past its `pages` own pages (DR:808-821), which
/// a window that ends on the marker writes last.
#[must_use]
pub fn cut_at_extra_marker(output: &[u32], pages: usize) -> &[u32] {
    let mut seen = 0;
    for (i, &t) in output.iter().enumerate() {
        if t == PAGE {
            seen += 1;
            if seen > pages {
                return &output[..i];
            }
        }
    }
    output
}

/// The rows block retirement keeps in a window with a successor (§5.10.9),
/// in its own frame: the last own page's marker row and image tail, and the
/// lookahead page. Empty without a lookahead or for an unparseable prompt.
#[must_use]
pub fn pinned_rows(prompt_ids: &[u32], lookahead: usize) -> Vec<Range<usize>> {
    let Some(p) = parse_prompt(prompt_ids) else {
        return Vec::new();
    };
    if lookahead == 0 || p.pages() <= lookahead {
        return Vec::new();
    }
    let last = p.pages() - lookahead - 1;
    let (marker, end) = (p.page_starts[last], p.page_ends[last]);
    let image = tail_start(end).map_or(marker, |s| s.max(marker))..end;
    vec![
        marker..marker + 1,
        image,
        p.page_starts[last + 1]..p.page_ends[last + 1],
    ]
}

// ── Reuse and fallback (§5.10.4–5.10.5) ─────────────────────────────────────

/// Why a window after the first prefilled its full prompt instead of reusing
/// its predecessor's tail (§5.10.5). The order is the wheel's order of
/// checks (DR:956-1101).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackReason {
    /// (ii) The follow-up was retracted or already has output: SGLang
    /// scheduling only, reachable here through the test hook.
    Retracted,
    /// (i) Nothing was retained: the previous window did not end on EOS, or
    /// its prompt does not parse.
    NoRetained,
    /// (iii) The retained window ends at another page than this one starts.
    PageMismatch,
    /// (iv) The layouts do not match, or the tail page is too short for its
    /// image tail.
    Layout,
    /// (v) No output to continue from, or its text tail leaves no room.
    NoOutput,
    /// (vi) KV allocation failed (and (ix), which maps onto it with no
    /// shared pool): reachable here through the test hook.
    Alloc,
    /// (vii) An image run straddles the copy boundary.
    Straddle,
    /// (viii) The reuse planning failed.
    PlanError,
}

impl FallbackReason {
    /// The ledger and robot spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Retracted => "retracted",
            Self::NoRetained => "no_retained",
            Self::PageMismatch => "page_mismatch",
            Self::Layout => "layout",
            Self::NoOutput => "no_output",
            Self::Alloc => "alloc",
            Self::Straddle => "straddle",
            Self::PlanError => "plan_error",
        }
    }
}

/// What a finished window leaves its successor (DR:836-935).
#[derive(Debug, Clone)]
struct Retained {
    /// The window's prompt as it ran (the rewritten one when reused).
    prompt_ids: Vec<u32>,
    /// Its output, forced ids included, cut before the marker past its own
    /// pages and to the rows that hold KV.
    output: Vec<u32>,
    /// Pages of `prompt_ids` it only read.
    lookahead: usize,
    /// The global page the next window must start at.
    next_page: usize,
    /// The kept rows, when its tail page has an image tail.
    rows: Option<TailCapture>,
}

/// A finished window's kept rows (DR:323-366, row-exact; the wheel keeps the
/// whole 32-row KV pages around them, which only costs memory).
#[derive(Debug, Clone)]
struct TailCapture {
    /// The tail page's marker row.
    marker: usize,
    /// The image tail, then the lookahead page.
    image: Range<usize>,
    /// The text tail.
    text: Range<usize>,
    /// Per layer: the rows `marker`, `image`, `text`, in that order.
    layers: Vec<srswa::TailRows>,
}

/// The successor's share of a window that ended on EOS: `prompt_ids` ran
/// with `context_pages` leading pages it does not own and `lookahead`
/// trailing ones it only read; `ids` is its output, forced ids included. A
/// prompt that does not parse retains nothing (DR:884-885).
///
/// # Errors
/// [`FocrError::Other`] if a kept row is gone from the cache: the window's
/// pins and text-tail margin keep every one (§5.10.9).
fn retain(
    state: &srswa::SrSwaState,
    prompt_ids: &[u32],
    ids: &[u32],
    (context_pages, lookahead): (usize, usize),
    next_page: usize,
) -> FocrResult<Option<Retained>> {
    let Some(p) = parse_prompt(prompt_ids) else {
        return Ok(None);
    };
    let Some(own) = p
        .pages()
        .checked_sub(context_pages + lookahead)
        .filter(|&own| own > 0)
    else {
        return Ok(None);
    };
    let committed = state.rows().saturating_sub(prompt_ids.len());
    let output = cut_at_extra_marker(ids, own);
    let output = output[..output.len().min(committed)].to_vec();
    let last = p.pages() - lookahead - 1;
    let (marker, end) = (p.page_starts[last], p.page_ends[last]);
    let image_end = if lookahead > 0 {
        p.page_ends[last + 1]
    } else {
        end
    };
    let rows = match tail_start(end).filter(|&s| s > marker) {
        None => None,
        Some(start) => {
            let text = text_tail(&output, TAIL_TOKENS).map_or(0..0, |(s, t)| {
                prompt_ids.len() + s..prompt_ids.len() + s + t.len()
            });
            let ranges = [marker..marker + 1, start..image_end, text.clone()];
            let layers = state
                .layers
                .iter()
                .map(|cache| cache.tail_rows(&ranges))
                .collect::<FocrResult<_>>()?;
            Some(TailCapture {
                marker,
                image: start..image_end,
                text,
                layers,
            })
        }
    };
    Ok(Some(Retained {
        prompt_ids: prompt_ids.to_vec(),
        output,
        lookahead,
        next_page,
        rows,
    }))
}

/// A reused window's splice ([`plan_reuse`]).
#[derive(Debug, Clone)]
struct Splice {
    /// The copy and the rewritten prompt.
    plan: TailPlan,
    /// The text tail's ids; the forced `<PAGE>` follows them.
    text: Vec<u32>,
    /// Per layer, the copied prompt rows.
    copied: Vec<srswa::TailRows>,
    /// Per layer, the text tail's rows.
    text_rows: Vec<srswa::TailRows>,
}

/// Whether an image run of `prompt` straddles `kept_from` other than the one
/// a straddle plan allows (DR:986-1001): a run whose first row is before
/// `kept_from` and whose last row is at or past it.
fn straddles(prompt: &PromptPages, kept_from: usize, allow: bool) -> bool {
    let straddling: Vec<usize> = prompt
        .page_starts
        .iter()
        .zip(&prompt.page_ends)
        .filter(|&(&s, &e)| s + 1 < kept_from && kept_from < e)
        .map(|(_, &e)| e - 1)
        .collect();
    match straddling.as_slice() {
        [] => false,
        [last] => !(allow && *last == kept_from),
        _ => true,
    }
}

/// Plan window `lo`'s reuse of `retained` over its full prompt `full_ids`,
/// in the wheel's order of checks (DR:956-1101), or the reason it falls back
/// to a recompute. `room` is the positions the full prompt leaves to
/// generate. `hook` injects a reason at the point it is checked, by
/// perturbing that check's input (the fallback tests); `marker_only` splices
/// no text, as the wheel does when the text's KV allocation fails
/// (DR:1257-1285): only the forced `<PAGE>` follows the prompt.
fn plan_reuse(
    retained: Option<&Retained>,
    lo: usize,
    full_ids: &[u32],
    room: usize,
    (hook, marker_only): (Option<FallbackReason>, bool),
) -> Result<Splice, FallbackReason> {
    use FallbackReason as R;
    if hook == Some(R::Retracted) {
        return Err(R::Retracted);
    }
    let retained = retained
        .filter(|_| hook != Some(R::NoRetained))
        .ok_or(R::NoRetained)?;
    let next_page = retained.next_page + usize::from(hook == Some(R::PageMismatch));
    if next_page != lo {
        return Err(R::PageMismatch);
    }
    let old_ids: &[u32] = if hook == Some(R::Layout) {
        &[]
    } else {
        &retained.prompt_ids
    };
    let plan = plan_tail_copy(old_ids, full_ids, retained.lookahead, true).ok_or(R::Layout)?;
    let output: &[u32] = if hook == Some(R::NoOutput) {
        &[]
    } else {
        &retained.output
    };
    let text = text_tail(output, TAIL_TOKENS)
        .filter(|(_, t)| t.len() + 2 < room)
        .ok_or(R::NoOutput)?
        .1
        .to_vec();
    if hook == Some(R::Alloc) {
        return Err(R::Alloc);
    }
    let full = parse_prompt(full_ids).ok_or(R::PlanError)?;
    // The hook resumes the prompt inside its first page's image run.
    let kept_from = if hook == Some(R::Straddle) {
        full.page_starts[0] + 2
    } else {
        plan.kept_from
    };
    if straddles(&full, kept_from, plan.straddle) {
        return Err(R::Straddle);
    }
    if hook == Some(R::PlanError) {
        return Err(R::PlanError);
    }
    // The kept rows must be the ones the plan copies.
    let cap = retained.rows.as_ref().ok_or(R::PlanError)?;
    if cap.marker != plan.marker_src
        || cap.image.start != plan.copy_src.start
        || plan.copy_src.end > cap.image.end
        || cap.text.len() != text.len()
    {
        return Err(R::PlanError);
    }
    let text_at = 1 + cap.image.len();
    let text_rows = text_at..text_at + if marker_only { 0 } else { cap.text.len() };
    let text = if marker_only { Vec::new() } else { text };
    Ok(Splice {
        copied: cap
            .layers
            .iter()
            .map(|t| t.slice(0..plan.copy_len))
            .collect(),
        text_rows: cap
            .layers
            .iter()
            .map(|t| t.slice(text_rows.clone()))
            .collect(),
        plan,
        text,
    })
}

/// A window's page guard (§5.10.7): its own pages expected, so the marker
/// that opens the lookahead page forces EOS, and SGLang's last-page `<PAGE>`
/// ban unless the window ends on a marker, `end_on_marker = lookahead > 0 ||
/// reused` (CLP:227, DR:1109): only on a last window that was recomputed.
fn window_guard(span: &WindowSpan, reused: bool, params: &DecodeParams) -> sampler::PageGuard {
    let guard = document_guard(span.own(), PAGE, params);
    if span.lookahead || reused {
        guard
    } else {
        guard.ban_last_page_marker()
    }
}

// ── The driver ──────────────────────────────────────────────────────────────

/// A windowed decode's settings.
#[derive(Debug, Clone, Default)]
pub struct WindowedOptions {
    /// Own pages per window, `W ≥ 1`.
    pub window: usize,
    /// Test hook: per window index, a fallback reason injected into its
    /// reuse ([`FallbackReason`]).
    pub force_fallback: Vec<(usize, FallbackReason)>,
    /// Test hook: reused windows whose text splice fails, so only the forced
    /// `<PAGE>` is spliced (DR:1257-1285).
    pub marker_only_splice: Vec<usize>,
    /// Test hook: the window whose budget is already spent when it starts.
    pub budget_spent_at: Option<usize>,
    /// Test hook: the decoder, instead of the environment's.
    pub(super) precision: Option<DocumentPrecision>,
}

/// One window's record (U2-D1a; U2-D5 makes it the robot `window` event).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowStats {
    /// Position in the plan.
    pub index: usize,
    /// Own pages, global and 0-based.
    pub pages: Range<usize>,
    /// It read a lookahead page.
    pub lookahead: bool,
    /// It reused its predecessor's tail.
    pub reused: bool,
    /// Why a window after the first did not.
    pub fallback_reason: Option<FallbackReason>,
    /// The one-page variant: the copied lookahead page's last image row was
    /// prefilled again.
    pub straddle: bool,
    /// Rows it prefilled.
    pub prefill_tokens: usize,
    /// Rows it copied: the tail and lookahead rows, then the text tail.
    pub copied_rows: usize,
    /// Ids forced and hidden before its first decode step: the text tail
    /// and `<PAGE>`.
    pub forced_tokens: usize,
    /// Ids it generated past them.
    pub generated_tokens: usize,
    /// How it ended.
    pub finish: FinishReason,
    /// Peak resident KV bytes.
    pub kv_bytes_peak: usize,
    /// Reruns without block retirement.
    pub retire_fallbacks: usize,
    /// Wall time, vision included.
    pub elapsed_ms: u64,
}

/// A windowed document decode (U2-D1a).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowedDocument {
    /// One body per page reached, in page order.
    pub pages: postprocess::TokenPages,
    /// `stop`; or `length` (a window's position limit), `budget` or
    /// `cancelled`, each of which ends the document with the pages so far.
    pub finish_reason: FinishReason,
    /// The windows' facts and counters, summed (`peak_kv_bytes`: the most
    /// any window held).
    pub stats: DocumentStats,
    /// Per window, in order.
    pub windows: Vec<WindowStats>,
    /// Per window, its ids: the forced prefix
    /// ([`WindowStats::forced_tokens`]), then the ids it generated.
    pub window_ids: Vec<Vec<u32>>,
}

/// A window's own pages as it ends, in page order: their bodies, and per
/// body its generated ids and how it ended.
#[derive(Debug, Clone, Copy)]
pub struct WindowPages<'a> {
    /// The bodies ([`postprocess::TokenPages::bodies`]).
    pub bodies: &'a [String],
    /// Generated ids per body, its `<PAGE>` excluded.
    pub tokens: &'a [usize],
    /// How each body ended ([`DocumentStats::page_finish`]).
    pub finish: &'a [PageFinish],
}

/// Gets each window's record and its own pages as it ends.
pub type WindowSink<'a> = &'a mut dyn FnMut(&WindowStats, WindowPages<'_>);

/// How a window starts.
enum WindowStart<'s> {
    /// Its full prompt, at page base `lo`; after the first window, with the
    /// reason it could not reuse.
    Recompute(Option<FallbackReason>),
    /// Its predecessor's tail, spliced.
    Reuse(&'s Splice),
}

/// One window's run inputs.
struct WindowRun<'r> {
    span: WindowSpan,
    start: WindowStart<'r>,
    wc: &'r decoder::DecoderWeightCache,
    params: &'r DecodeParams,
    full_ids: &'r [u32],
    /// The vision block of each of its prompt pages.
    blocks: &'r [Mat],
}

/// What [`OcrModel::decode_document_windowed`] made of a document.
#[derive(Debug)]
pub enum Windowing {
    /// The document fits one window, so it decodes unwindowed (U2-B6), as
    /// the wheel never splits a document that fits (`doc_split.py:150-156`):
    /// its pages, in order, with no vision run yet.
    Fits(Vec<Preprocessed>),
    /// The windowed decode.
    Windowed(Box<WindowedDocument>),
}

impl Windowing {
    /// The windowed decode, or `None` for a document that fits one window.
    #[must_use]
    pub fn into_windowed(self) -> Option<WindowedDocument> {
        match self {
            Self::Windowed(doc) => Some(*doc),
            Self::Fits(_) => None,
        }
    }
}

impl OcrModel {
    /// Whether this model's documents decode in windows: a guarded `<PAGE>`
    /// contract (Unlimited-OCR-2).
    #[must_use]
    pub(crate) fn decodes_windows(&self) -> bool {
        self.arch()
            .decode_contract()
            .pages
            .is_some_and(|paging| paging.guard && paging.marker_id == PAGE)
    }

    /// Unlimited-OCR-2's windowed document decode (U2-D1a, U2-D1b) over the
    /// pages `pages` yields in order, read as they stream: the document's
    /// length is never needed up front. Its first own + 1 pages decide
    /// whether it fits one window ([`Windowing::Fits`], before any weight is
    /// touched). Otherwise windows run one after another ([`stream_span`]);
    /// each takes its pages' vision as it reads them, the first carried over
    /// from the previous window's lookahead, so vision runs once per page and
    /// a window's pages are dropped once encoded. `on_window` gets each
    /// window's record and its own pages' bodies as it ends. A window that
    /// stops (its position limit, a spent budget, a cancel) ends the document
    /// with the pages so far; nothing done is discarded. The run's report
    /// stays on the model ([`OcrModel::take_last_decode`]).
    ///
    /// # Errors
    /// A model without a guarded page contract, `window = 0`,
    /// `FOCR_DECODE_STATELESS` (a reused window has no embeddings for its
    /// copied rows), or any page, vision or decode error.
    pub fn decode_document_windowed(
        &self,
        pages: &mut dyn Iterator<Item = FocrResult<Preprocessed>>,
        opts: &WindowedOptions,
        mut on_window: Option<WindowSink<'_>>,
    ) -> FocrResult<Windowing> {
        if !self.decodes_windows() {
            return Err(FocrError::Other(anyhow::anyhow!(
                "{} has no guarded <PAGE> contract, so no windowed decode",
                self.arch().id()
            )));
        }
        if opts.window == 0 {
            return Err(FocrError::Usage(
                "a page window must own at least one page".into(),
            ));
        }
        let own = own_pages_per_window(opts.window, PER_PAGE_EST_BASE, UOCR2_UNWINDOWED_PAGE_LIMIT);
        // A document that fits one window has at most `own` pages.
        let mut head = Vec::with_capacity(own + 1);
        while head.len() <= own {
            match pages.next() {
                Some(pre) => head.push(pre?),
                None => return Ok(Windowing::Fits(head)),
            }
        }
        let precision = match opts.precision {
            Some(precision) => precision,
            None => DocumentPrecision::from_env()?,
        };
        let wc = match precision {
            DocumentPrecision::Mixed => self.decoder_cache()?,
            DocumentPrecision::CachedF32 => self.decoder_cache_f32()?,
            DocumentPrecision::Stateless => {
                return Err(FocrError::Usage(format!(
                    "{} re-runs the whole sequence from its embeddings, which a reused \
                     window does not have for its copied rows; windowed decoding needs a \
                     cached decoder",
                    super::DECODE_STATELESS_ENV
                )));
            }
        };
        let params = self.document_params(head.len());
        let tok = self.tokenizer()?;
        let instruction = tok.encode(MULTI_PAGE_PROMPT_TEXT)?;
        timing_log(&format!(
            "windowed document: windows of {own} own pages, precision {}",
            precision.name()
        ));
        // The head's pages first, then the rest of the stream.
        let mut head = head.into_iter();
        let mut next_page = || head.next().map(Ok).or_else(|| pages.next());
        let mut carried: Option<Mat> = None;
        let mut retained: Option<Retained> = None;
        let mut doc = WindowedDocument {
            pages: postprocess::TokenPages {
                bodies: Vec::new(),
                tokens: Vec::new(),
                leading_text: false,
                extra_markers: 0,
            },
            finish_reason: FinishReason::Stop,
            stats: DocumentStats::default(),
            windows: Vec::new(),
            window_ids: Vec::new(),
        };
        // The cooperative budget (D11): `FOCR_STAGE_BUDGET_FORWARD_MS` per
        // page, times the window's prompt pages, from the window's start.
        let page_budget = crate::forward_stage_budget();
        let mut lo = 0;
        for index in 0.. {
            let t0 = Instant::now();
            // Vision: the window's prompt pages as they are read, its first
            // carried over; each page's preprocess is dropped once encoded.
            let mut blocks: Vec<Mat> = carried.take().into_iter().collect();
            let held = blocks.len();
            let span = stream_span(index, lo, own, held, &mut |page| match next_page() {
                Some(pre) => {
                    blocks.push(self.page_block(&pre?, page)?);
                    Ok(true)
                }
                None => Ok(false),
            })?;
            if span.own() == 0 {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "window {index} has no page: the stream ended at page {lo}"
                )));
            }
            doc.stats.vision_encodes += blocks.len() - held;
            let deadline = if opts.budget_spent_at == Some(span.index) {
                Some(t0)
            } else {
                page_budget.and_then(|per_page: Duration| {
                    let pages = u32::try_from(span.prompt_pages()).unwrap_or(u32::MAX);
                    t0.checked_add(per_page.checked_mul(pages)?)
                })
            };
            let counts: Vec<usize> = blocks.iter().map(|b| b.rows).collect();
            let (full_ids, _) = paged_prompt(tok.bos_id(), PAGE, IMAGE, &counts, &instruction);
            let room = params.max_positions.saturating_sub(full_ids.len());
            if room == 0 {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "window {}'s prompt is {} tokens, over the {} position limit",
                    span.index,
                    full_ids.len(),
                    params.max_positions
                )));
            }
            let hook = opts
                .force_fallback
                .iter()
                .find(|(i, _)| *i == span.index)
                .map(|&(_, r)| r);
            let splice = if span.index == 0 {
                Err(None)
            } else {
                let marker_only = opts.marker_only_splice.contains(&span.index);
                let hooks = (hook, marker_only);
                plan_reuse(retained.take().as_ref(), span.lo, &full_ids, room, hooks).map_err(Some)
            };
            let start = match &splice {
                Ok(splice) => WindowStart::Reuse(splice),
                Err(reason) => WindowStart::Recompute(*reason),
            };
            let (reused, fallback_reason) = match start {
                WindowStart::Reuse(_) => (true, None),
                WindowStart::Recompute(reason) => (false, reason),
            };
            let (straddle, copied_rows, forced_tokens) = match &splice {
                Ok(s) => (
                    s.plan.straddle,
                    s.plan.copy_len + s.text.len(),
                    s.text.len() + 1,
                ),
                Err(_) => (false, 0, 0),
            };
            let prefill_tokens = match &splice {
                Ok(s) => s.plan.new_ids.len() - s.plan.copy_len,
                Err(_) => full_ids.len(),
            };
            // A budget its vision spent (or the hook) stops the document
            // before the window's prefill.
            if deadline.is_some_and(|d| Instant::now() >= d) {
                timing_log(&format!(
                    "forward stage budget ran out before window {}'s prefill",
                    span.index
                ));
                let stats = WindowStats {
                    index: span.index,
                    pages: span.lo..span.hi,
                    lookahead: span.lookahead,
                    reused,
                    fallback_reason,
                    straddle: false,
                    prefill_tokens: 0,
                    copied_rows: 0,
                    forced_tokens: 0,
                    generated_tokens: 0,
                    finish: FinishReason::Budget,
                    kv_bytes_peak: 0,
                    retire_fallbacks: 0,
                    elapsed_ms: u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX),
                };
                if let Some(f) = on_window.as_deref_mut() {
                    let none = WindowPages {
                        bodies: &[],
                        tokens: &[],
                        finish: &[],
                    };
                    f(&stats, none);
                }
                doc.stats.page_finish.push(PageFinish::Budget);
                doc.windows.push(stats);
                doc.window_ids.push(Vec::new());
                doc.finish_reason = FinishReason::Budget;
                break;
            }
            let run = WindowRun {
                span,
                start,
                wc,
                params: &params,
                full_ids: &full_ids,
                blocks: &blocks,
            };
            let mut next = None;
            let decode = decode_with_retire_fallback(
                srswa::retire_enabled(),
                None,
                |retire, budget_floor, ids, observer| {
                    let pass = DocumentPass {
                        precision,
                        retire,
                        deadline,
                        budget_floor,
                    };
                    self.run_window(&run, pass, ids, observer, &mut next)
                },
            )?;
            retained = next;
            if span.lookahead {
                carried = blocks.pop();
            }
            // A reused window's forced `<PAGE>` opened its first page.
            let visible = decode.ids.get(forced_tokens..).ok_or_else(|| {
                FocrError::Other(anyhow::anyhow!(
                    "window {} returned {} ids, fewer than its {forced_tokens} forced ones",
                    span.index,
                    decode.ids.len()
                ))
            })?;
            let split = postprocess::split_window_pages(
                tok,
                visible,
                PAGE,
                span.own(),
                reused,
                span.lookahead,
            )?;
            let stats = WindowStats {
                index: span.index,
                pages: span.lo..span.hi,
                lookahead: span.lookahead,
                reused,
                fallback_reason,
                straddle,
                prefill_tokens,
                copied_rows,
                forced_tokens,
                generated_tokens: visible.len(),
                finish: decode.finish_reason,
                kv_bytes_peak: decode.stats.peak_kv_bytes,
                retire_fallbacks: decode.stats.retire_fallbacks,
                elapsed_ms: u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX),
            };
            timing_log(&format!(
                "window {} pages [{}, {}){}: {}{}, prefill {}, copied {}, forced {}, \
                 generated {}, finish {}, bodies {}, extra markers {}, kv peak {} bytes, {:.2}s",
                span.index,
                span.lo,
                span.hi,
                if span.lookahead { " + lookahead" } else { "" },
                if reused { "reused" } else { "recomputed" },
                fallback_reason.map_or_else(String::new, |r| format!(" ({})", r.as_str())),
                stats.prefill_tokens,
                stats.copied_rows,
                stats.forced_tokens,
                stats.generated_tokens,
                stats.finish.as_str(),
                split.bodies.len(),
                split.extra_markers,
                stats.kv_bytes_peak,
                t0.elapsed().as_secs_f64(),
            ));
            if let Some(f) = on_window.as_deref_mut() {
                let own = WindowPages {
                    bodies: &split.bodies,
                    tokens: &split.tokens,
                    finish: &decode.stats.page_finish,
                };
                f(&stats, own);
            }
            let window_stats = decode.stats;
            let pages_out = &mut doc.pages;
            pages_out.bodies.extend(split.bodies);
            pages_out.tokens.extend(split.tokens);
            pages_out.leading_text |= split.leading_text;
            pages_out.extra_markers += split.extra_markers;
            let total = &mut doc.stats;
            total.page_finish.extend(window_stats.page_finish);
            total.forced_steps += window_stats.forced_steps;
            total.hf_all_neg_inf_steps += window_stats.hf_all_neg_inf_steps;
            total.retired_blocks += window_stats.retired_blocks;
            total.peak_kv_bytes = total.peak_kv_bytes.max(window_stats.peak_kv_bytes);
            total.retire_fallbacks += window_stats.retire_fallbacks;
            doc.window_ids.push(decode.ids);
            doc.windows.push(stats);
            // A stop inside a window ends the document with its completed
            // pages (plan §9.2); a later window would leave a gap of pages.
            if decode.finish_reason != FinishReason::Stop {
                doc.finish_reason = decode.finish_reason;
                break;
            }
            // Only the last window has no lookahead.
            if !span.lookahead {
                break;
            }
            lo = span.hi;
        }
        doc.stats.page_tokens.clone_from(&doc.pages.tokens);
        doc.stats.leading_text = doc.pages.leading_text;
        doc.stats.extra_markers = doc.pages.extra_markers;
        doc.stats.windows = doc.windows.len();
        self.record_windowed(&doc, &params);
        Ok(Windowing::Windowed(Box::new(doc)))
    }

    /// The windowed run's [`DecodeReport`], as a one-pass document leaves
    /// one: its ids are each window's generated ids in order, without the
    /// forced prefixes, and the pages it expected are those it read (each
    /// encoded once). A caller whose stream holds more pages adds them.
    fn record_windowed(&self, doc: &WindowedDocument, params: &DecodeParams) {
        let Ok(mut slot) = self.last_decode.lock() else {
            return;
        };
        let (decodes, length_stops) = slot
            .as_ref()
            .map_or((0, 0), |prior| (prior.decodes, prior.length_stops));
        let ids = doc
            .windows
            .iter()
            .zip(&doc.window_ids)
            .flat_map(|(w, ids)| ids.get(w.forced_tokens..).unwrap_or_default())
            .copied()
            .collect();
        let stopped = doc.finish_reason != FinishReason::Stop;
        *slot = Some(DecodeReport {
            ids,
            finish_reason: doc.finish_reason,
            cap: (doc.finish_reason == FinishReason::Length).then_some(LengthCap::Positions),
            pages: Some(PageBodies {
                expected: doc.stats.vision_encodes,
                bodies: doc.pages.bodies.len(),
            }),
            max_length: params.max_length,
            max_positions: params.max_positions,
            decodes: decodes + 1,
            length_stops: length_stops + usize::from(stopped),
            document: Some(doc.stats.clone()),
            admission_warnings: lever_policy::admit_from_process(self.arch()).warnings,
        });
    }

    /// One window's decode pass: start it (its full prompt, or the splice),
    /// pin the rows its successor copies, decode it with the window guard,
    /// and, when it ends on EOS with a successor, keep its tail in
    /// `retained`. `ids` gets its output, the forced prefix included.
    ///
    /// # Errors
    /// Any prefill, splice or decode error.
    fn run_window(
        &self,
        run: &WindowRun<'_>,
        pass: DocumentPass,
        ids: &mut Vec<u32>,
        observer: Option<&mut dyn FnMut(u32)>,
        retained: &mut Option<Retained>,
    ) -> FocrResult<DocumentDecode> {
        let WindowRun {
            span,
            wc,
            params,
            full_ids,
            blocks,
            ..
        } = *run;
        ids.clear();
        *retained = None;
        let router = wc.router()?;
        let table = self.embed_table()?;
        let reused = matches!(run.start, WindowStart::Reuse(_));
        let mut guard = window_guard(&span, reused, params);
        let tp = Instant::now();
        let (prompt_ids, state, hidden, context_pages) = match &run.start {
            WindowStart::Recompute(_) => {
                let coords = srswa::PageCoords::build(full_ids)?;
                let pages: Vec<Option<&Mat>> = blocks.iter().map(Some).collect();
                let embeds = self.window_embeds(full_ids, 0, &pages)?;
                let mut state = srswa::SrSwaState::new(router, &coords, page_base(span.lo)?)?;
                state.domain = params.entmax_domain;
                let hidden = decoder::extend_with_cache_srswa(wc, &mut state, &embeds, &coords)?;
                timing_log(&format!(
                    "window.recompute {} rows at page base {} {:.2}s",
                    full_ids.len(),
                    span.lo,
                    tp.elapsed().as_secs_f64()
                ));
                (full_ids.to_vec(), state, Self::last_hidden_row(&hidden)?, 0)
            }
            WindowStart::Reuse(splice) => {
                let rewritten = &splice.plan.new_ids;
                let copy_len = splice.plan.copy_len;
                let coords = srswa::PageCoords::build(rewritten)?;
                let base = page_base(span.lo - CONTEXT_PAGES)?;
                let mut state = srswa::SrSwaState::new(router, &coords, base)?;
                state.page = srswa::DecodePageState::reused_window(&coords, base)?;
                state.domain = params.entmax_domain;
                for (cache, rows) in state.layers.iter_mut().zip(&splice.copied) {
                    cache.append(&rows.k, &rows.v, &rows.ik, &coords.img_page[..copy_len])?;
                }
                // Page 0 of the rewritten prompt is the tail page, copied
                // whole; page p ≥ 1 is the window's page p − 1.
                let pages: Vec<Option<&Mat>> = std::iter::once(None)
                    .chain(blocks.iter().map(Some))
                    .collect();
                let embeds = self.window_embeds(rewritten, copy_len, &pages)?;
                // The extend's sampled token is discarded: decode starts from
                // the forced `<PAGE>` (DR:1069-1077).
                decoder::extend_with_cache_srswa(wc, &mut state, &embeds, &coords)?;
                let text_pages = vec![-1; splice.text.len()];
                for (cache, rows) in state.layers.iter_mut().zip(&splice.text_rows) {
                    cache.append(&rows.k, &rows.v, &rows.ik, &text_pages)?;
                }
                ids.extend_from_slice(&splice.text);
                ids.push(PAGE);
                for &id in ids.iter() {
                    guard.observe(id);
                }
                let row = Mat::from_vec(1, table.cols(), table.row_f32(PAGE as usize));
                let forced_next = matches!(guard.verdict(), sampler::GuardVerdict::Force(_));
                let h =
                    decoder::decode_step_with_cache_srswa(wc, &mut state, PAGE, &row, forced_next)?;
                timing_log(&format!(
                    "window.splice copied {copy_len} + {} text rows, prefilled {} rows at page \
                     base {}{} {:.2}s",
                    splice.text.len(),
                    rewritten.len() - copy_len,
                    span.lo - CONTEXT_PAGES,
                    if splice.plan.straddle {
                        " (one-page variant)"
                    } else {
                        ""
                    },
                    tp.elapsed().as_secs_f64()
                ));
                (
                    rewritten.clone(),
                    state,
                    Self::last_hidden_row(&h)?,
                    CONTEXT_PAGES,
                )
            }
        };
        let mut state = state;
        let lookahead = usize::from(span.lookahead);
        if span.lookahead {
            let pins = pinned_rows(&prompt_ids, lookahead);
            for cache in &mut state.layers {
                cache.set_pinned(&pins, true)?;
            }
        }
        state.set_retire(pass.retire && guard.enabled());
        let forced_prefix = ids.len();
        let mut decoder_state = DocumentDecoder::Cached { wc, state, hidden };
        let logits = self.document_logits(&decoder_state)?;
        let spec = DocumentLoop {
            params,
            pass,
            marker: PAGE,
            prompt_len: prompt_ids.len(),
            pages: span.own(),
            forced_prefix,
        };
        let (finish_reason, stats) =
            self.document_loop(spec, &mut decoder_state, logits, &mut guard, ids, observer)?;
        if finish_reason == FinishReason::Stop
            && span.lookahead
            && let DocumentDecoder::Cached { state, .. } = &decoder_state
        {
            *retained = retain(state, &prompt_ids, ids, (context_pages, lookahead), span.hi)?;
        }
        Ok(DocumentDecode {
            ids: ids.clone(),
            finish_reason,
            stats,
        })
    }

    /// Page `page`'s image rows `[n, hidden]`: its base view's vision block
    /// with its row breaks and trailing separator, as the unwindowed prompt
    /// scatters it ([`connector::fuse_no_crop`]).
    ///
    /// # Errors
    /// A tiled page, a vision error, or a block that does not fill the
    /// page's image slots.
    fn page_block(&self, pre: &Preprocessed, page: usize) -> FocrResult<Mat> {
        let mut features = self.vision_tower(pre)?;
        if pre.crop_grid.is_tiled() || features.len() != 1 {
            return Err(FocrError::Other(anyhow::anyhow!(
                "document page {} is not one base view ({} vision blocks)",
                page + 1,
                features.len()
            )));
        }
        let block = connector::assemble_global_block(
            &features.remove(0),
            Self::global_grid_h(pre),
            Self::global_grid_w(pre),
            &Self::image_newline(&self.weights)?,
            &Self::view_seperator(&self.weights)?,
        )?;
        if block.rows != pre.placeholder_token_count() {
            return Err(FocrError::Other(anyhow::anyhow!(
                "document page {}: a {}-row vision block for {} image slots",
                page + 1,
                block.rows,
                pre.placeholder_token_count()
            )));
        }
        Ok(block)
    }

    /// The embeddings of rows `from..` of the window prompt `ids`: the embed
    /// table's, with each image row taking its page's vision block row
    /// (`blocks[p]` for prompt page `p`; a page with `k` image rows in the
    /// prompt holds the last `k` of its block).
    ///
    /// # Errors
    /// An unparseable prompt, or an image row whose page has no block, or a
    /// block too short or of another width.
    fn window_embeds(&self, ids: &[u32], from: usize, blocks: &[Option<&Mat>]) -> FocrResult<Mat> {
        let bad =
            |why: String| FocrError::Other(anyhow::anyhow!("window prompt embeddings: {why}"));
        let prompt = parse_prompt(ids).ok_or_else(|| bad("the prompt does not parse".into()))?;
        let mut out = self.embed_prompt(&ids[from..])?;
        for (p, (&marker, &end)) in prompt.page_starts.iter().zip(&prompt.page_ends).enumerate() {
            let images = marker + 1..end;
            if images.end <= from {
                continue;
            }
            let block = blocks
                .get(p)
                .copied()
                .flatten()
                .ok_or_else(|| bad(format!("page {p} has image rows to embed and no block")))?;
            if block.rows < images.len() || block.cols != out.cols {
                return Err(bad(format!(
                    "page {p}'s block is [{}, {}] for {} image rows of width {}",
                    block.rows,
                    block.cols,
                    images.len(),
                    out.cols
                )));
            }
            let skip = block.rows - images.len();
            for (j, r) in images.enumerate().filter(|&(_, r)| r >= from) {
                out.row_mut(r - from).copy_from_slice(block.row(skip + j));
            }
        }
        Ok(out)
    }
}

/// A page base as the RoPE tables take it.
fn page_base(base: usize) -> FocrResult<u32> {
    u32::try_from(base)
        .map_err(|_| FocrError::Other(anyhow::anyhow!("page base {base} is out of range")))
}

/// The committed U2-D2a windowed-oracle runs (`tests/fixtures/uocr2_windowed`,
/// U2-D2b), in tag order.
#[cfg(test)]
pub(super) fn oracle_fixtures() -> Vec<serde_json::Value> {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/uocr2_windowed");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("the fixture directory")
        .map(|e| e.expect("an entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    assert_eq!(paths.len(), 5, "the five committed runs");
    paths
        .iter()
        .map(|p| serde_json::from_slice(&std::fs::read(p).expect("a fixture")).expect("JSON"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::special::BOS;

    const TEXT: u32 = 1000;

    /// `BOS (<PAGE> <image>×n)… instruction` over per-page image counts.
    fn full_prompt(images: &[usize]) -> Vec<u32> {
        paged_prompt(
            BOS,
            PAGE,
            IMAGE,
            images,
            &[TEXT, TEXT + 1, TEXT + 2, TEXT + 3],
        )
        .0
    }

    #[test]
    fn plan_covers_every_page_once_with_the_next_first_page_as_lookahead() {
        for pages in 1..=40 {
            for window in 1..=12 {
                let plan = plan_windows(pages, window, PER_PAGE_EST_BASE, 60);
                if pages <= window {
                    assert_eq!(plan, None, "{pages} pages at W={window} are unwindowed");
                    continue;
                }
                let plan = plan.expect("windowed");
                let mut next = 0;
                for (i, w) in plan.iter().enumerate() {
                    assert_eq!((w.index, w.lo), (i, next));
                    assert!(w.own() >= 1 && w.own() <= window);
                    let last = i + 1 == plan.len();
                    assert_eq!(
                        w.lookahead, !last,
                        "a lookahead on every window but the last"
                    );
                    if !last {
                        assert_eq!(w.own(), window);
                        assert_eq!(plan[i + 1].lo, w.hi, "the lookahead is the next first page");
                    }
                    next = w.hi;
                }
                assert_eq!(next, pages);
            }
        }
    }

    /// U2-D1b: windows found as the pages stream in are the up-front plan's,
    /// and no page is pulled twice or past the end of the stream.
    #[test]
    fn streamed_windows_are_the_up_front_plan() {
        for pages in 1..=40 {
            for window in 1..=12 {
                let own = own_pages_per_window(window, PER_PAGE_EST_BASE, 60);
                let mut pulled = 0;
                let mut ended = false;
                let mut pull = |page: usize| -> FocrResult<bool> {
                    assert!(!ended, "pulled again after the stream ended");
                    assert_eq!(page, pulled, "pages are pulled in order, once each");
                    if pulled == pages {
                        ended = true;
                        return Ok(false);
                    }
                    pulled += 1;
                    Ok(true)
                };
                let mut spans = Vec::new();
                let (mut lo, mut held) = (0, 0);
                loop {
                    let span = stream_span(spans.len(), lo, own, held, &mut pull).unwrap();
                    spans.push(span);
                    if !span.lookahead {
                        break;
                    }
                    (lo, held) = (span.hi, 1);
                }
                match plan_windows(pages, window, PER_PAGE_EST_BASE, 60) {
                    Some(plan) => assert_eq!(spans, plan, "{pages} pages at W={window}"),
                    None => assert_eq!(
                        spans,
                        [WindowSpan {
                            index: 0,
                            lo: 0,
                            hi: pages,
                            lookahead: false
                        }],
                        "{pages} pages at W={window} fit one window"
                    ),
                }
            }
        }
    }

    #[test]
    fn own_pages_take_the_tightest_bound_per_mode() {
        assert_eq!(own_pages_per_window(10, PER_PAGE_EST_BASE, 60), 10);
        assert_eq!(
            own_pages_per_window(1000, PER_PAGE_EST_BASE, 60),
            60,
            "the page limit"
        );
        assert_eq!(
            own_pages_per_window(1000, PER_PAGE_EST_BASE, usize::MAX),
            131_007 / 274,
            "the prompt bound"
        );
        assert_eq!(
            own_pages_per_window(1000, PER_PAGE_EST_GUNDAM, usize::MAX),
            86
        );
        assert_eq!(own_pages_per_window(0, PER_PAGE_EST_BASE, 60), 1);
        // With the page limit, a window's prompt and its lookahead stay well
        // inside its positions.
        for window in 1..=60 {
            let own = own_pages_per_window(window, PER_PAGE_EST_BASE, 60);
            assert!((own + 1) * PER_PAGE_EST_BASE + PLAN_RESERVE < PLAN_CONTEXT);
        }
        assert_eq!(
            plan_windows(25, DEFAULT_PAGE_WINDOW, PER_PAGE_EST_BASE, 60)
                .unwrap()
                .iter()
                .map(|w| (w.lo, w.hi, w.lookahead))
                .collect::<Vec<_>>(),
            [(0, 10, true), (10, 20, true), (20, 25, false)]
        );
    }

    #[test]
    fn prompts_parse_into_pages_and_refuse_bad_layouts() {
        let ids = full_prompt(&[273, 273]);
        let p = parse_prompt(&ids).unwrap();
        assert_eq!(p.prefix_len, 1);
        assert_eq!(p.page_starts, [1, 275]);
        assert_eq!(p.page_ends, [275, 549]);
        assert_eq!(p.instr_start, 549);
        let mut early_image = ids.clone();
        early_image[0] = IMAGE;
        assert_eq!(
            parse_prompt(&early_image),
            None,
            "an image before the first marker"
        );
        let mut empty_page = ids.clone();
        empty_page.splice(276..549, []);
        assert_eq!(parse_prompt(&empty_page), None, "a marker without images");
        let mut late_image = ids.clone();
        late_image.push(IMAGE);
        assert_eq!(
            parse_prompt(&late_image),
            None,
            "an image in the instruction"
        );
        assert_eq!(parse_prompt(&[TEXT]), None, "no marker");
        // A rewritten prompt has neither BOS nor instruction.
        let rw = parse_prompt(&[PAGE, IMAGE, IMAGE, PAGE, IMAGE]).unwrap();
        assert_eq!((rw.prefix_len, rw.instr_start), (0, 5));
    }

    /// The U2-D2a oracle's W=2 run over 7 base pages (`w2-n7-t12`): the
    /// wheel's plans for windows 1 (a lookahead copied whole) and 3 (the
    /// one-page variant), row for row.
    #[test]
    fn tail_plan_matches_the_wheel_on_the_oracle_windows() {
        let three = full_prompt(&[273, 273, 273]);
        assert_eq!(three.len(), 827);
        let plan = plan_tail_copy(&three, &three, 1, true).unwrap();
        assert_eq!(plan.marker_src, 275);
        assert_eq!(plan.copy_src, 416..823, "E = 549, 549 − 128 − 5 = 416");
        assert_eq!(
            (plan.copy_len, plan.kept_from, plan.new_ids.len()),
            (408, 275, 956)
        );
        assert_eq!(
            (plan.old_pages, plan.new_pages, plan.copied_pages),
            (2, 3, 1)
        );
        assert!(!plan.straddle);
        assert_eq!(plan.new_ids[0], PAGE);
        assert_eq!(
            plan.new_ids[134], PAGE,
            "the copied lookahead page's marker"
        );
        assert_eq!(
            parse_prompt(&plan.new_ids).unwrap().page_starts,
            [0, 134, 408, 682]
        );
        // Window 2 runs on window 1's rewritten prompt; window 3 owns one
        // page and has no lookahead: the one-page variant.
        let w1 = plan.new_ids;
        let w2 = plan_tail_copy(&w1, &three, 1, true).unwrap();
        assert_eq!((w2.marker_src, w2.copy_src.clone()), (408, 544..956));
        assert_eq!((w2.copy_len, w2.new_ids.len()), (413, 961));
        let last = full_prompt(&[273]);
        let w3 = plan_tail_copy(&w2.new_ids, &last, 1, true).unwrap();
        assert_eq!((w3.marker_src, w3.copy_src.clone()), (413, 544..960));
        assert_eq!(
            (w3.copy_len, w3.kept_from, w3.new_ids.len()),
            (417, 274, 418)
        );
        assert!(w3.straddle);
        assert_eq!(
            *w3.new_ids.last().unwrap(),
            IMAGE,
            "the re-prefilled image row"
        );
        // A W=1 window that is not the last copies its lookahead whole.
        let one_with_lookahead = full_prompt(&[273, 273]);
        let mid = plan_tail_copy(&three, &one_with_lookahead, 1, true).unwrap();
        assert!(!mid.straddle && mid.copied_pages == 1);
        // Without lookahead reuse the first page is prefilled again.
        let plain = plan_tail_copy(&three, &three, 1, false).unwrap();
        assert_eq!((plain.copy_src.clone(), plain.kept_from), (416..549, 1));
        assert_eq!(plain.copied_pages, 0);
    }

    #[test]
    fn tail_start_is_block_aligned_and_refuses_short_pages() {
        for end in 160..5000 {
            let start = tail_start(end).unwrap();
            assert_eq!(start % srswa::BLOCK_SIZE, 0);
            assert!((128..=159).contains(&(end - start)));
        }
        assert_eq!(tail_start(127), None);
        // A tail page with too few image rows has no tail past its marker.
        let short = [
            vec![BOS, PAGE],
            vec![IMAGE; 100],
            vec![PAGE],
            vec![IMAGE; 273],
        ]
        .concat();
        assert_eq!(plan_tail_copy(&short, &full_prompt(&[273]), 1, true), None);
    }

    #[test]
    fn text_tail_and_extra_marker_cut_follow_the_wheel() {
        let out = [vec![TEXT; 5], vec![PAGE], vec![TEXT + 1; 200], vec![PAGE]].concat();
        let cut = cut_at_extra_marker(&out, 1);
        assert_eq!(cut.len(), 206, "cut before the second marker");
        assert_eq!(cut_at_extra_marker(&out, 2).len(), out.len());
        let (start, tail) = text_tail(cut, TAIL_TOKENS).unwrap();
        assert_eq!((start, tail.len()), (78, 128));
        let short = [vec![TEXT; 300], vec![PAGE], vec![TEXT + 1; 7]].concat();
        let (start, tail) = text_tail(&short, TAIL_TOKENS).unwrap();
        assert_eq!(
            (start, tail),
            (301, &[TEXT + 1; 7][..]),
            "never across a marker"
        );
        assert_eq!(text_tail(&[TEXT, PAGE], TAIL_TOKENS), Some((2, &[][..])));
        assert_eq!(text_tail(&[], TAIL_TOKENS), None);
        assert_eq!(text_tail(&[TEXT], 0), None);
    }

    /// The worked coordinate table (§5.10.4), on window 1 of the W=2 oracle
    /// run, and the reused-window seed: the forced `<PAGE>` lands in bucket
    /// 2 at `tok_pos = 1 + n_img(lo)` with query page 1. The HF seed would
    /// not.
    #[test]
    fn reused_window_coordinates_follow_the_worked_table() {
        let three = full_prompt(&[273, 273, 273]);
        let plan = plan_tail_copy(&three, &three, 1, true).unwrap();
        let c = srswa::PageCoords::build(&plan.new_ids).unwrap();
        // (a) the tail marker, (b) its image tail.
        assert_eq!((c.img_page[0], c.bucket[0]), (-1, 1));
        for r in 1..134 {
            assert_eq!((c.img_page[r], c.bucket[r]), (0, 1), "row {r}");
        }
        // (c) the lookahead page: marker, then images.
        assert_eq!((c.img_page[134], c.bucket[134]), (-1, 2));
        for r in 135..408 {
            assert_eq!((c.img_page[r], c.bucket[r]), (1, 2), "row {r}");
        }
        // The first newly prefilled marker (page lo + 1).
        assert_eq!(
            (
                c.img_page[408],
                c.query_page[408],
                c.bucket[408],
                c.tok_pos[408]
            ),
            (-1, 2, 3, 0)
        );
        // The seed: pf_cnt is the oracle's seed `cnt` {1: 134, 2..4: 274}.
        assert_eq!(c.pf_cnt, [0, 134, 274, 274, 274]);
        let base = 1;
        let mut seed = srswa::DecodePageState::reused_window(&c, base).unwrap();
        let page = seed.feed(PAGE);
        assert_eq!(
            (page.bucket, page.tok_pos, page.query_page),
            (2, 1 + 273, 1),
            "the forced <PAGE> opens page lo"
        );
        assert_eq!(page.rope_page, srswa::rope_page(2, base));
        let mut hf = srswa::DecodePageState::after_prompt(&c, base);
        let wrong = hf.feed(PAGE);
        assert_ne!(
            (wrong.bucket, wrong.tok_pos, wrong.query_page),
            (page.bucket, page.tok_pos, page.query_page),
            "the HF seed fails the table"
        );
    }

    /// Every row's page parity matches its global page in every window, for
    /// both frames: local bucket + page base ≡ global page + 1 (mod 2).
    #[test]
    fn page_base_keeps_every_rows_page_parity() {
        let mut seed = 0x5EEDu64;
        let mut next = |n: u64| {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (seed >> 33) % n
        };
        for _ in 0..40 {
            let pages = 2 + next(30) as usize;
            let window = 1 + next(6) as usize;
            let Some(plan) = plan_windows(pages, window, PER_PAGE_EST_BASE, 60) else {
                continue;
            };
            for w in &plan {
                let full = full_prompt(&vec![273; w.prompt_pages()]);
                let c = srswa::PageCoords::build(&full).unwrap();
                for (r, &p) in c.img_page.iter().enumerate().filter(|&(_, &p)| p >= 0) {
                    let global = w.lo + p as usize;
                    let rope = srswa::rope_page(c.bucket[r], w.lo as u32);
                    assert_eq!(rope as usize, (global + 1) % 2, "recomputed {w:?} row {r}");
                }
                if w.index == 0 {
                    continue;
                }
                let prev = full_prompt(&vec![273; plan[w.index - 1].prompt_pages()]);
                let plan = plan_tail_copy(&prev, &full, 1, true).unwrap();
                let c = srswa::PageCoords::build(&plan.new_ids).unwrap();
                let base = (w.lo - CONTEXT_PAGES) as u32;
                for (r, &p) in c.img_page.iter().enumerate().filter(|&(_, &p)| p >= 0) {
                    // Rewritten page 0 is global page lo − 1.
                    let global = w.lo - 1 + p as usize;
                    let rope = srswa::rope_page(c.bucket[r], base);
                    assert_eq!(rope as usize, (global + 1) % 2, "reused {w:?} row {r}");
                }
            }
        }
    }

    /// §5.10.9 at W = 8: window 0's tail marker is row 1919, the last row
    /// of a block. Under retirement through a long decode, the pinned rows
    /// (a)–(c) and the text tail (d) survive bit for bit, and a reused
    /// window's frame pins its own rows.
    #[test]
    fn pinned_rows_survive_retirement_bit_for_bit() {
        let full = full_prompt(&[273; 9]);
        let pins = pinned_rows(&full, 1);
        assert_eq!(pins, [1919..1920, 2048..2193, 2193..2467]);
        assert_eq!(1919 % srswa::BLOCK_SIZE, 31);
        let c = srswa::PageCoords::build(&full).unwrap();
        let mut rng = 0x9177u64;
        let mut rand_mat = |rows: usize, cols: usize| {
            let data = (0..rows * cols)
                .map(|_| {
                    rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    ((rng >> 40) as f32 / (1u64 << 24) as f32) - 0.5
                })
                .collect();
            Mat::from_vec(rows, cols, data)
        };
        let width = srswa::NUM_HEADS * srswa::HEAD_DIM;
        let head_cls = vec![0.1f32; srswa::HEAD_DIM];
        let mut cache = srswa::SrSwaLayerCache::new(&head_cls, true).unwrap();
        let n = full.len();
        let (k, v, ik) = (
            rand_mat(n, width),
            rand_mat(n, width),
            rand_mat(n, srswa::HEAD_DIM),
        );
        cache.append(&k, &v, &ik, &c.img_page).unwrap();
        cache.set_pinned(&pins, true).unwrap();
        // Generated rows: 8 own pages of 300 text rows, each opened by a
        // marker, then the marker that opens the lookahead page.
        let mut generated = Vec::new();
        for _ in 0..8 {
            generated.push(PAGE);
            generated.extend(std::iter::repeat_n(TEXT, 300));
        }
        generated.push(PAGE);
        let mut markers = 0i32;
        for (i, &id) in generated.iter().enumerate() {
            let one = |m: &Mat| Mat::from_vec(1, m.cols, m.row(i % n).to_vec());
            cache.append(&one(&k), &one(&v), &one(&ik), &[-1]).unwrap();
            markers += i32::from(id == PAGE);
            let q = n + i;
            // The marker past the own pages forces EOS: no retirement there.
            let forced = i + 1 == generated.len();
            cache.retire_dead_blocks(markers - 1, q, forced).unwrap();
        }
        assert!(cache.dead_blocks() > 0, "retirement ran");
        let output = &generated[..];
        let cut = cut_at_extra_marker(output, 8);
        let (start, tail) = text_tail(cut, TAIL_TOKENS).unwrap();
        assert_eq!(tail.len(), TAIL_TOKENS);
        let text = n + start..n + start + tail.len();
        let kept = [pins[0].clone(), pins[1].start..pins[2].end, text];
        let rows = cache.tail_rows(&kept).expect("every kept row survives");
        let mut at = 0;
        for r in &kept {
            for (got, src) in [(&rows.k, &k), (&rows.v, &v), (&rows.ik, &ik)] {
                for (j, pos) in r.clone().enumerate() {
                    let want = src.row(if pos < n { pos } else { (pos - n) % n });
                    assert_eq!(got.row(at + j), want, "row {pos}");
                }
            }
            at += r.len();
        }
        // A reused window pins its own frame's rows.
        let three = full_prompt(&[273, 273, 273]);
        let plan = plan_tail_copy(&three, &three, 1, true).unwrap();
        assert_eq!(
            pinned_rows(&plan.new_ids, 1),
            [408..409, 544..682, 682..956]
        );
        assert!(pinned_rows(&three, 0).is_empty(), "no successor, no pins");
    }

    /// §5.10.7: a reused last window is never banned, so it may write an
    /// extra `<PAGE>`, which forces EOS (stitching drops its body); a window
    /// with a lookahead ends on the marker that opens it; only a recomputed
    /// last window bans `<PAGE>` once its pages have their markers.
    #[test]
    fn only_a_recomputed_last_window_bans_the_marker() {
        use sampler::GuardVerdict as V;
        let params = DecodeParams {
            max_page_tokens: 0,
            ..DecodeParams::single_image()
        };
        let eos = params.eos_token_id;
        let verdict = |span: &WindowSpan, reused: bool, ids: &[u32]| {
            let mut guard = window_guard(span, reused, &params);
            for &id in ids {
                guard.observe(id);
            }
            guard.verdict()
        };
        let last = WindowSpan {
            index: 2,
            lo: 20,
            hi: 22,
            lookahead: false,
        };
        let both_pages = [PAGE, TEXT, PAGE, TEXT];
        assert_eq!(verdict(&last, true, &both_pages), V::Free, "reused");
        assert_eq!(verdict(&last, false, &both_pages), V::BanPage, "recomputed");
        assert_eq!(
            verdict(&last, true, &[PAGE, TEXT]),
            V::BanEos,
            "a page to go"
        );
        let extra = [PAGE, TEXT, PAGE, TEXT, PAGE];
        assert_eq!(
            verdict(&last, true, &extra),
            V::Force(eos),
            "the extra marker"
        );
        let middle = WindowSpan {
            index: 1,
            lo: 10,
            hi: 20,
            lookahead: true,
        };
        let own: Vec<u32> = (0..10).flat_map(|_| [PAGE, TEXT]).collect();
        for reused in [false, true] {
            assert_eq!(verdict(&middle, reused, &own), V::Free, "reused {reused}");
            let opened = [own.clone(), vec![PAGE]].concat();
            assert_eq!(verdict(&middle, reused, &opened), V::Force(eos));
        }
    }

    /// A retained window over `full_prompt([273; 3])` with `text` output
    /// ids, its rows captured as `retain` keeps them (one layer).
    fn retained(text: usize) -> Retained {
        let prompt = full_prompt(&[273, 273, 273]);
        let output = [vec![PAGE], vec![TEXT; 40], vec![PAGE], vec![TEXT + 1; text]].concat();
        let rows = |n: usize| srswa::TailRows {
            k: Mat::zeros(n, srswa::NUM_HEADS * srswa::HEAD_DIM),
            v: Mat::zeros(n, srswa::NUM_HEADS * srswa::HEAD_DIM),
            ik: Mat::zeros(n, srswa::HEAD_DIM),
        };
        let kept_text = text.min(TAIL_TOKENS);
        let text_start = prompt.len() + output.len() - kept_text;
        Retained {
            rows: Some(TailCapture {
                marker: 275,
                image: 416..823,
                text: text_start..text_start + kept_text,
                layers: vec![rows(1 + 407 + kept_text)],
            }),
            prompt_ids: prompt,
            output,
            lookahead: 1,
            next_page: 2,
        }
    }

    /// §5.10.5: every reason, injected through the hook at its own check,
    /// stops the reuse with that reason; the real checks catch the same
    /// cases unhooked. A failed text splice keeps the reuse with no text.
    #[test]
    fn every_fallback_reason_is_detected_where_the_wheel_checks_it() {
        use FallbackReason as R;
        let none = (None, false);
        let full = full_prompt(&[273, 273, 273]);
        let room = 100_000;
        let ret = retained(200);
        let ok = plan_reuse(Some(&ret), 2, &full, room, none).expect("reusable");
        assert_eq!(ok.text.len(), TAIL_TOKENS);
        assert_eq!(ok.copied[0].rows(), 408);
        assert_eq!(ok.text_rows[0].rows(), TAIL_TOKENS);
        let marker_only = plan_reuse(Some(&ret), 2, &full, room, (None, true)).unwrap();
        assert!(marker_only.text.is_empty());
        assert_eq!(marker_only.text_rows[0].rows(), 0);
        assert_eq!(marker_only.copied[0].rows(), 408, "the copy is kept");
        for reason in [
            R::Retracted,
            R::NoRetained,
            R::PageMismatch,
            R::Layout,
            R::NoOutput,
            R::Alloc,
            R::Straddle,
            R::PlanError,
        ] {
            assert_eq!(
                plan_reuse(Some(&ret), 2, &full, room, (Some(reason), false)).err(),
                Some(reason),
                "{reason:?}"
            );
            assert!(!reason.as_str().is_empty());
        }
        // Unhooked: no entry, another page, a short tail page, no output, a
        // text tail with no room, and kept rows that are not the plan's.
        assert_eq!(
            plan_reuse(None, 2, &full, room, none).err(),
            Some(R::NoRetained)
        );
        assert_eq!(
            plan_reuse(Some(&ret), 3, &full, room, none).err(),
            Some(R::PageMismatch)
        );
        let mut short = retained(200);
        short.prompt_ids = [
            vec![BOS, PAGE],
            vec![IMAGE; 273],
            vec![PAGE],
            vec![IMAGE; 100],
        ]
        .concat();
        short.prompt_ids.extend(full_prompt(&[273])[1..].iter());
        assert_eq!(
            plan_reuse(Some(&short), 2, &full, room, none).err(),
            Some(R::Layout)
        );
        let mut silent = retained(0);
        silent.output.clear();
        assert_eq!(
            plan_reuse(Some(&silent), 2, &full, room, none).err(),
            Some(R::NoOutput)
        );
        assert_eq!(
            plan_reuse(Some(&ret), 2, &full, TAIL_TOKENS + 2, none).err(),
            Some(R::NoOutput)
        );
        let mut moved = retained(200);
        moved.rows.as_mut().unwrap().marker = 274;
        assert_eq!(
            plan_reuse(Some(&moved), 2, &full, room, none).err(),
            Some(R::PlanError)
        );
        // An empty last body is no fallback: only the forced <PAGE> follows.
        let empty_tail = retained(0);
        let s = plan_reuse(Some(&empty_tail), 2, &full, room, none).unwrap();
        assert!(s.text.is_empty());
        // The straddle check allows only the one-page variant's own run.
        let p = parse_prompt(&full).unwrap();
        assert!(!straddles(&p, 275, false), "at a page end");
        assert!(straddles(&p, 274, false), "inside page 1's images");
        assert!(straddles(&p, 300, true), "not at the run's last row");
        let one = parse_prompt(&full_prompt(&[273])).unwrap();
        assert!(!straddles(&one, 274, true), "the one-page variant's row");
    }

    /// One window's KV rows, as the U2-D2a dumps record them (`rows.*.npy`):
    /// per row its id, origin (0 prefilled, 1 copied, 2 spliced text, 3
    /// decode input), source row, image page, query page (−2 for a row that
    /// never queried here), bucket, tok_pos and RoPE page (a copied row's
    /// own), and its bucket and RoPE page in this window's frame.
    #[derive(Debug, Default)]
    struct OracleRows {
        arrays: [Vec<i32>; 10],
    }

    /// [`OracleRows`]' arrays, in the fixture's `rows_sha256` names.
    const ROW_ARRAYS: [&str; 10] = [
        "ids",
        "origin",
        "src_row",
        "img_page",
        "q_page",
        "bucket",
        "tok_pos",
        "rope_page",
        "frame_bucket",
        "frame_rope_page",
    ];

    impl OracleRows {
        /// Append a row: `own` is its (bucket, tok_pos, RoPE page), `frame`
        /// its (bucket, RoPE page) in this window.
        fn push(
            &mut self,
            id: u32,
            (origin, src, img, q): (i32, i32, i32, i32),
            own: [u32; 3],
            frame: [u32; 2],
        ) {
            let int = |v: u32| i32::try_from(v).expect("an i32 coordinate");
            let row = [
                int(id),
                origin,
                src,
                img,
                q,
                int(own[0]),
                int(own[1]),
                int(own[2]),
                int(frame[0]),
                int(frame[1]),
            ];
            for (array, value) in self.arrays.iter_mut().zip(row) {
                array.push(value);
            }
        }

        /// Row `r`'s own (bucket, tok_pos, RoPE page), which a copy keeps.
        fn own(&self, r: usize) -> [u32; 3] {
            let at = |a: usize| u32::try_from(self.arrays[a][r]).expect("a coordinate");
            [at(5), at(6), at(7)]
        }

        fn len(&self) -> usize {
            self.arrays[0].len()
        }

        /// The sha256 of array `a`'s little-endian int32 data, the dumps'
        /// manifest hash.
        fn sha256(&self, a: usize) -> String {
            use sha2::Digest as _;
            let mut hasher = sha2::Sha256::new();
            for v in &self.arrays[a] {
                hasher.update(v.to_le_bytes());
            }
            hasher
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect()
        }
    }

    /// The dumps' RoPE page of a row in `bucket` at page base `base`: the
    /// bucket plus the base's parity where the bucket is positive, unfolded.
    /// The rotary folds it mod 2, as [`srswa::rope_page`] already has.
    fn dump_rope(bucket: u32, base: usize) -> u32 {
        let base = u32::try_from(base).expect("a page base");
        let unfolded = if bucket == 0 { 0 } else { bucket + base % 2 };
        assert_eq!(
            unfolded % 2,
            srswa::rope_page(bucket, base),
            "the folded page"
        );
        unfolded
    }

    /// The decode rows fed from `seed` at page base `base`: each id's
    /// coordinates as SGLang records them. A `<PAGE>` past the last bucket
    /// clamps there; HF resets its tok_pos to the bucket's prompt count and
    /// SGLang keeps the running count (DISC-U2-7), which only the
    /// guard-forced EOS step after it reads.
    fn decode_rows(
        rows: &mut OracleRows,
        mut seed: srswa::DecodePageState,
        base: usize,
        fed: &[u32],
    ) {
        let mut last: Option<srswa::DecodeCoord> = None;
        for &id in fed {
            let mut c = seed.feed(id);
            assert_eq!(c.rope_page, dump_rope(c.bucket, base) % 2);
            if let Some(prev) = last
                && id == PAGE
                && c.bucket == prev.bucket
            {
                c.tok_pos = prev.tok_pos + 1;
            }
            let rope = dump_rope(c.bucket, base);
            let own = [c.bucket, c.tok_pos, rope];
            rows.push(id, (3, -1, -1, c.query_page), own, [c.bucket, rope]);
            last = Some(c);
        }
    }

    /// A finished window, as its successor reads it.
    struct OracleWindow {
        prompt_ids: Vec<u32>,
        /// Its output with its forced ids, cut before the extra marker.
        output: Vec<u32>,
        lookahead: usize,
        rows: OracleRows,
    }

    fn fixture_ids(v: &serde_json::Value) -> Vec<u32> {
        v.as_array().map_or_else(Vec::new, |a| {
            a.iter()
                .map(|id| u32::try_from(id.as_u64().expect("an id")).expect("a u32 id"))
                .collect()
        })
    }

    fn fixture_usize(v: &serde_json::Value) -> usize {
        usize::try_from(v.as_u64().expect("a count")).expect("a usize")
    }

    /// U2-D2b, model-free: per committed U2-D2a run
    /// (`tests/fixtures/uocr2_windowed`), the plan, each reused window's copy
    /// plan, text tail, splice rows and retention, and every KV row's
    /// coordinates, rebuilt here from the oracle's own generated ids, equal
    /// the wheel's: the row arrays hash to the dumps' (§5.10.3–5.10.4).
    #[test]
    fn windowed_rows_match_the_oracle_dumps() {
        let mut windows_checked = 0;
        for fx in oracle_fixtures() {
            let tag = fx["tag"].as_str().expect("a tag").to_string();
            let width = fixture_usize(&fx["window"]);
            let windows = fx["windows"].as_array().expect("windows");
            let pages = fixture_usize(&windows.last().expect("a window")["lo_hi_emit"][1]);
            let plan = plan_windows(pages, width, PER_PAGE_EST_BASE, UOCR2_UNWINDOWED_PAGE_LIMIT);
            match fx["plan"].as_array() {
                None => assert_eq!(plan, None, "{tag}: unwindowed"),
                Some(want) => {
                    let got: Vec<serde_json::Value> = plan
                        .expect("windowed")
                        .iter()
                        .map(|w| serde_json::json!([w.lo, w.hi, w.lo]))
                        .collect();
                    assert_eq!(&got, want, "{tag}: plan");
                }
            }
            let mut prev: Option<OracleWindow> = None;
            for w in windows {
                let k = fixture_usize(&w["window"]);
                let at = format!("{tag} w{k}");
                let lo = fixture_usize(&w["lo_hi_emit"][0]);
                let hi = fixture_usize(&w["lo_hi_emit"][1]);
                let lookahead = fixture_usize(&w["lookahead"]);
                let counts: Vec<usize> = w["prompt_image_counts"]
                    .as_array()
                    .expect("image counts")
                    .iter()
                    .map(fixture_usize)
                    .collect();
                let bos = u32::try_from(w["prompt_bos_id"].as_u64().expect("BOS")).expect("u32");
                let instruction = fixture_ids(&w["prompt_instruction_ids"]);
                let (full, _) = paged_prompt(bos, PAGE, IMAGE, &counts, &instruction);
                let forced = fixture_ids(&w["forced_ids"]);
                let generated = fixture_ids(&w["generated_ids"]);
                let mut rows = OracleRows::default();
                let reused = w["reused"] == true;
                let base;
                let prompt_ids = if reused {
                    let prev = prev.as_ref().expect("a reused window follows one");
                    base = lo - CONTEXT_PAGES;
                    let plan = plan_tail_copy(&prev.prompt_ids, &full, prev.lookahead, true)
                        .expect("a tail plan");
                    let want = &w["tail_copy_plan"];
                    let got = serde_json::json!({
                        "src_positions": [plan.marker_src, plan.copy_src.start, plan.copy_src.end],
                        "copy_len": plan.copy_len, "kept_from": plan.kept_from,
                        "old_pages": plan.old_pages, "new_pages": plan.new_pages,
                        "copied_pages": plan.copied_pages, "allow_straddle": plan.straddle,
                    });
                    for (key, value) in got.as_object().expect("an object") {
                        assert_eq!(&want[key], value, "{at}: tail plan {key}");
                    }
                    assert_eq!(plan.new_ids.len(), fixture_usize(&w["prompt_len"]), "{at}");
                    let c = srswa::PageCoords::build(&plan.new_ids).expect("coordinates");
                    let rope = |b: u32| dump_rope(b, base);
                    let sources = std::iter::once(plan.marker_src).chain(plan.copy_src.clone());
                    for (r, src) in sources.enumerate() {
                        let flags = (1, src as i32, c.img_page[r], -2);
                        let frame = [c.bucket[r], rope(c.bucket[r])];
                        rows.push(plan.new_ids[r], flags, prev.rows.own(src), frame);
                    }
                    for r in plan.copy_len..plan.new_ids.len() {
                        let flags = (0, -1, c.img_page[r], c.query_page[r]);
                        let own = [c.bucket[r], c.tok_pos[r], rope(c.bucket[r])];
                        rows.push(plan.new_ids[r], flags, own, [own[0], own[2]]);
                    }
                    // The text tail, which a failed text splice drops: only
                    // the forced `<PAGE>` follows (DR:1257-1285).
                    let (marker, text) = forced.split_last().expect("a forced <PAGE>");
                    assert_eq!(*marker, PAGE, "{at}");
                    if !text.is_empty() {
                        let (start, tail) =
                            text_tail(&prev.output, TAIL_TOKENS).expect("a text tail");
                        assert_eq!(tail, text, "{at}: the text tail");
                        for (j, &id) in tail.iter().enumerate() {
                            let src = prev.prompt_ids.len() + start + j;
                            // The seeded carry's response bucket is the tail page's.
                            rows.push(
                                id,
                                (2, src as i32, -1, -2),
                                prev.rows.own(src),
                                [1, rope(1)],
                            );
                        }
                    }
                    let splice = plan.new_ids.len();
                    let splice = serde_json::json!([splice, splice + text.len()]);
                    assert_eq!(w["splice_rows"], splice, "{at}: splice rows");
                    let seed = srswa::DecodePageState::reused_window(&c, base as u32)
                        .expect("a reused-window seed");
                    let fed: Vec<u32> = std::iter::once(PAGE).chain(generated.clone()).collect();
                    decode_rows(&mut rows, seed, base, &fed);
                    if text.is_empty() {
                        // With no text rows to carry it, the wheel writes the
                        // first step's carry into the prompt's last slot,
                        // whose image page then reads −1; franken_ocr keeps
                        // the row's page (DISC-U2-12).
                        rows.arrays[3][plan.new_ids.len() - 1] = -1;
                    }
                    plan.new_ids
                } else {
                    base = lo;
                    assert!(
                        forced.is_empty(),
                        "{at}: a recomputed window forces nothing"
                    );
                    assert_eq!(full.len(), fixture_usize(&w["prompt_len"]), "{at}");
                    let c = srswa::PageCoords::build(&full).expect("coordinates");
                    for r in 0..full.len() {
                        let flags = (0, -1, c.img_page[r], c.query_page[r]);
                        let rope = dump_rope(c.bucket[r], base);
                        rows.push(
                            full[r],
                            flags,
                            [c.bucket[r], c.tok_pos[r], rope],
                            [c.bucket[r], rope],
                        );
                    }
                    decode_rows(
                        &mut rows,
                        srswa::DecodePageState::after_prompt(&c, base as u32),
                        base,
                        &generated,
                    );
                    full
                };
                assert_eq!(
                    fixture_usize(&w["page_base"]["effective"]),
                    base,
                    "{at}: page base"
                );
                assert_eq!(rows.len(), fixture_usize(&w["kv_rows"]), "{at}: KV rows");
                for (a, name) in ROW_ARRAYS.iter().enumerate() {
                    assert_eq!(
                        rows.sha256(a),
                        w["rows_sha256"][name].as_str().expect("a hash"),
                        "{at}: rows.{name}"
                    );
                }
                // Retention: the output with its forced ids, cut before the
                // marker past the own pages; the kept rows widened to 32-row
                // pages within the KV rows, as the wheel keeps them.
                let own = hi - lo;
                let output: Vec<u32> = forced.iter().chain(&generated).copied().collect();
                let output = cut_at_extra_marker(&output, own).to_vec();
                let retention = &w["retention"];
                if retention["retained"] == true {
                    assert_eq!(
                        output.len(),
                        fixture_usize(&retention["output_ids_kept"]),
                        "{at}"
                    );
                    assert_eq!(
                        fixture_usize(&retention["next_page"]),
                        hi,
                        "{at}: next page"
                    );
                    let p = parse_prompt(&prompt_ids).expect("the prompt parses");
                    let last = p.pages() - lookahead - 1;
                    let marker = p.page_starts[last];
                    let image = tail_start(p.page_ends[last]).expect("an image tail")
                        ..p.page_ends[last + 1];
                    let (start, tail) = text_tail(&output, TAIL_TOKENS).expect("a text tail");
                    let text = prompt_ids.len() + start..prompt_ids.len() + start + tail.len();
                    let widen = |r: Range<usize>| {
                        let b = srswa::BLOCK_SIZE;
                        (r.start / b * b)..r.end.div_ceil(b).saturating_mul(b).min(rows.len())
                    };
                    let mut kept: Vec<usize> = [marker..marker + 1, image, text]
                        .into_iter()
                        .flat_map(widen)
                        .collect();
                    kept.sort_unstable();
                    kept.dedup();
                    let want: Vec<usize> = retention["kept_ranges"]
                        .as_array()
                        .expect("kept ranges")
                        .iter()
                        .flat_map(|r| fixture_usize(&r[0])..fixture_usize(&r[1]))
                        .collect();
                    assert_eq!(kept, want, "{at}: kept rows");
                }
                prev = Some(OracleWindow {
                    prompt_ids,
                    output,
                    lookahead,
                    rows,
                });
                windows_checked += 1;
            }
        }
        assert_eq!(windows_checked, 15, "every committed window");
    }

    /// U2-D2b, tokenizer-gated (`FOCR_UOCR2_SOURCE_DIR`): each committed
    /// run's windows, split into their own pages' bodies as the driver splits
    /// them (an unsplit run as an unwindowed document), stitch to the wheel's
    /// document bodies (W/`entrypoints/openai/doc_split.py`): the same
    /// ownership across windows, hidden forced `<PAGE>` and closing
    /// lookahead marker.
    #[test]
    fn windowed_bodies_match_the_oracle_stitching() {
        use sha2::Digest as _;
        let Some(dir) = std::env::var_os("FOCR_UOCR2_SOURCE_DIR") else {
            eprintln!("SKIP windowed stitching: FOCR_UOCR2_SOURCE_DIR unset");
            return;
        };
        let bytes = std::fs::read(std::path::Path::new(&dir).join("tokenizer.json"))
            .expect("the v2 tokenizer.json");
        assert_eq!(
            format!("{:x}", sha2::Sha256::digest(&bytes)),
            super::super::model_arch::UNLIMITED_OCR_V2_TOKENIZER_SHA256,
            "the pinned v2 tokenizer"
        );
        let tok = crate::tokenizer::Tokenizer::from_json_bytes(&bytes).expect("it loads");
        for fx in oracle_fixtures() {
            let tag = fx["tag"].as_str().expect("a tag");
            let windowed = fx["plan"].is_array();
            let mut bodies = Vec::new();
            for w in fx["windows"].as_array().expect("windows") {
                let lo = fixture_usize(&w["lo_hi_emit"][0]);
                let own = fixture_usize(&w["lo_hi_emit"][1]) - lo;
                let ids = fixture_ids(&w["generated_ids"]);
                let split = if windowed {
                    postprocess::split_window_pages(
                        &tok,
                        &ids,
                        PAGE,
                        own,
                        w["reused"] == true,
                        fixture_usize(&w["lookahead"]) > 0,
                    )
                } else {
                    postprocess::split_pages_by_token(&tok, &ids, PAGE, own)
                };
                bodies.extend(split.expect("the ids split").bodies);
            }
            let want: Vec<&str> = fx["document"]["bodies"]
                .as_array()
                .expect("bodies")
                .iter()
                .map(|b| b.as_str().expect("a body"))
                .collect();
            assert_eq!(bodies, want, "{tag}: stitched bodies");
        }
    }
}
