//! SR-SWA — Unlimited-OCR-2's page-aware sparse attention (plan
//! `docs/planning/UNLIMITED_OCR_2_PLAN.md` §5.4–§5.6). This file starts with the
//! page bookkeeping and the two-axis **page2d** RoPE that both the main
//! attention and the Token Sieve router rotate with (U2-B1).
//!
//! Sources are pinned at HF `baidu/Unlimited-OCR-2@5632329a`
//! (`docs/truth-pack/SOURCE_HASHES.md`): `SB` = `_sparse_base_impl.py`, `MU` =
//! `modeling_unlimitedocr.py`, `MD` = `modeling_deepseekv2.py`; `W/` is the
//! pinned SGLang wheel's `sglang/srt/`.
//!
//! **Coordinates.** A v2 prompt is `BOS (<PAGE> <image>×n)×N instruction`.
//! `cum[i]` counts `<PAGE>` markers up to and including row `i`, `last_img` is
//! the last `<image>` row and `N = cum[last_img]` the number of prompt pages.
//! - **Bucket** (SB:2554-2555): rows `≤ last_img` take `cum` (1..N); later rows
//!   (the instruction and every generated token) take `cum − N`, clamped at 0.
//!   Bucket 0 is the reserved "no page" bucket (BOS and the instruction).
//! - **`tok_pos`** (SB:2562-2565): the 0-based running count of earlier rows in
//!   the same bucket, so page `b`'s response continues after its prompt rows.
//! - **`pf_cnt[b]`**: bucket `b`'s rows `≤ last_img`.
//! - **Rope page** (SB:2052-2054 with `sparse_rope_page_mod = 2`): the bucket,
//!   shifted by the window's `page_base` parity and folded mod 2.
//!
//! There is **no 1-D fallback** (OQ-V2-6): HF falls back to YaRN RoPE when a
//! row has no `<PAGE>` or no image, but franken_ocr always builds `<PAGE><image>`
//! prompts, so such a prompt is an error here.
//!
//! **Token Sieve router** (U2-B2, plan §5.6): per layer and query row, the
//! remote 32-row blocks the main attention reads and the bias each one adds.
//! `DA` = `dash_attention.py`. Selection is discrete, so this is a scalar f32
//! reference that follows HF's inference path operation for operation.
//!
//! **Attention and cache** (U2-B3, U2-B4): block-sparse biased attention over
//! any [`BlockKv`], and the per-layer cache that stores K/V and index keys in
//! 32-row blocks and frees whole blocks no later query can read (plan §5.12).

use super::decoder::{RopeTable, apply_rope, linear_no_bias};
use super::tensor::Mat;
use crate::error::{FocrError, FocrResult};
use crate::tokenizer::special;
use rayon::prelude::*;
use std::ops::Range;

/// `<PAGE>`, the v2 page marker (`config.json` `page_token_id`, added token
/// 128827).
pub const PAGE_TOKEN_ID: u32 = 128_827;

/// Attention head dimension of both the main attention and the router (128).
pub const HEAD_DIM: usize = 128;

/// Frequency pairs on the page axis (`sparse_rope_page_pairs`).
const PAGE_PAIRS: usize = 1;

/// Token-axis frequency pairs: `HEAD_DIM / 2 − PAGE_PAIRS` = 63 (SB:2027).
pub const N_TOK_FREQS: usize = HEAD_DIM / 2 - PAGE_PAIRS;

/// Token-axis RoPE base (`sparse_rope_tok_theta`).
const TOK_THETA: f32 = 10_000.0;

/// The page axis is folded modulo this (`sparse_rope_page_mod`).
const PAGE_MOD: u32 = 2;

/// Page-axis frequency: `1 / page_theta^(0 / page_pairs)` = 1 for the single
/// page pair (SB:2033-2034).
const INV_PAGE: f32 = 1.0;

/// The 63 token-axis inverse frequencies in the reference's operation order,
/// "resample" mode (SB:2030-2032): `1 / (10000 ** (f32(i) / 63))`, all f32.
/// Note the `/63`, not v1's `2i/128`.
#[must_use]
pub fn inv_tok_freqs() -> [f32; N_TOK_FREQS] {
    let mut inv = [0.0f32; N_TOK_FREQS];
    for (i, slot) in inv.iter_mut().enumerate() {
        let exponent = i as f32 / N_TOK_FREQS as f32;
        *slot = 1.0 / TOK_THETA.powf(exponent);
    }
    inv
}

/// The page-axis RoPE coordinate of a row in local `bucket` of a window whose
/// first owned page sits at global page `page_base` (W/`layers/attention/
/// unlimited_ocr_v2_backend.py:1001-1031`, W/`session/uocr_doc_reuse.py:109-121`),
/// already folded mod 2 as the rotary does.
///
/// `page_base` is 0 unwindowed and for window 0, `lo` for a recomputed window
/// starting at global page `lo`, and `lo − 1` for a reused window. Bucket 0
/// stays at 0 because the base is added only where the bucket is positive.
#[must_use]
pub fn rope_page(bucket: u32, page_base: u32) -> u32 {
    if bucket == 0 {
        0
    } else {
        (bucket % PAGE_MOD + page_base % PAGE_MOD) % PAGE_MOD
    }
}

/// The 64 rotation angles of one row: 63 token-axis angles then the page
/// angle (SB:2055-2057).
fn row_angles(inv_tok: &[f32; N_TOK_FREQS], tok_pos: u32, rope_page: u32, out: &mut [f32]) {
    let t = tok_pos as f32;
    for (o, &f) in out[..N_TOK_FREQS].iter_mut().zip(inv_tok) {
        *o = t * f;
    }
    out[N_TOK_FREQS] = rope_page as f32 * INV_PAGE;
}

/// Page2d cos/sin tables for rows given as `(tok_pos, rope_page)` pairs, in the
/// layout [`super::decoder::apply_rope`] consumes.
#[must_use]
pub fn page2d_rope_table(tok_pos: &[u32], rope_pages: &[u32]) -> RopeTable {
    assert_eq!(
        tok_pos.len(),
        rope_pages.len(),
        "page2d_rope_table: row count"
    );
    let inv_tok = inv_tok_freqs();
    let half = HEAD_DIM / 2;
    let mut angles = vec![0.0f32; tok_pos.len() * half];
    for ((row, &t), &p) in angles.chunks_exact_mut(half).zip(tok_pos).zip(rope_pages) {
        row_angles(&inv_tok, t, p, row);
    }
    RopeTable::from_angles(&angles, HEAD_DIM)
}

/// Prefill-time page coordinates of a token sequence (SB:2433-2456 for the page
/// ids, SB:2511-2568 for buckets and positions).
///
/// Built over a prompt, or over a prompt plus teacher-forced generated tokens:
/// rows after `last_img` follow the response rule either way. With more than `N`
/// generated `<PAGE>` markers the bucket is not clamped at `N` here (SB:2555),
/// unlike [`DecodePageState`] (DISC-U2-7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageCoords {
    /// Page bucket per row; 0 is the reserved no-page bucket.
    pub bucket: Vec<u32>,
    /// Running 0-based count of earlier rows in the same bucket.
    pub tok_pos: Vec<u32>,
    /// 0-based page of each `<image>` row, −1 for every other row.
    pub img_page: Vec<i32>,
    /// Query page per row (`_PAGE_SCOPE_PREFILL = True`, MD:333): `cum − 1` up
    /// to `last_img`, `cum − 1 − N` after it, and −1 wherever that is negative.
    pub query_page: Vec<i32>,
    /// Prefill (`≤ last_img`) rows per bucket; its length is the bucket count.
    pub pf_cnt: Vec<u32>,
    /// `N`: `<PAGE>` markers up to the last image.
    pub prompt_pages: u32,
    /// `<PAGE>` markers in the whole sequence (MU:512's `mcum` seed).
    pub markers: u32,
    /// Index of the last `<image>` row.
    pub last_img: usize,
}

impl PageCoords {
    /// Compute the coordinates of `ids`.
    ///
    /// # Errors
    /// [`FocrError::Other`] when `ids` has no `<image>` row or no `<PAGE>`
    /// marker before its last image: v2 has no 1-D RoPE fallback (OQ-V2-6).
    pub fn build(ids: &[u32]) -> FocrResult<Self> {
        let Some(last_img) = ids.iter().rposition(|&t| t == special::IMAGE) else {
            return Err(no_fallback("the sequence has no <image> token"));
        };
        let mut cum = Vec::with_capacity(ids.len());
        let mut c = 0u32;
        for &t in ids {
            c += u32::from(t == PAGE_TOKEN_ID);
            cum.push(c);
        }
        let n = cum[last_img];
        if n == 0 {
            return Err(no_fallback("no <PAGE> marker precedes the last <image>"));
        }
        let mut bucket = Vec::with_capacity(ids.len());
        let mut query_page = Vec::with_capacity(ids.len());
        let mut img_page = Vec::with_capacity(ids.len());
        for (i, (&t, &c)) in ids.iter().zip(&cum).enumerate() {
            let c = i64::from(c);
            let n = i64::from(n);
            let in_prefill = i <= last_img;
            let b = if in_prefill { c } else { (c - n).max(0) };
            bucket.push(b as u32);
            let q = if in_prefill { c - 1 } else { c - 1 - n };
            query_page.push(if q < 0 { -1 } else { q as i32 });
            img_page.push(if t == special::IMAGE {
                (c - 1) as i32
            } else {
                -1
            });
        }
        let n_buckets = bucket.iter().copied().max().unwrap_or(0) as usize + 1;
        let mut running = vec![0u32; n_buckets];
        let mut pf_cnt = vec![0u32; n_buckets];
        let mut tok_pos = Vec::with_capacity(ids.len());
        for (i, &b) in bucket.iter().enumerate() {
            let b = b as usize;
            tok_pos.push(running[b]);
            running[b] += 1;
            if i <= last_img {
                pf_cnt[b] += 1;
            }
        }
        Ok(Self {
            bucket,
            tok_pos,
            img_page,
            query_page,
            pf_cnt,
            prompt_pages: n,
            markers: c,
            last_img,
        })
    }

    /// Rope page coordinate per row for a window at `page_base`
    /// ([`rope_page`]).
    #[must_use]
    pub fn rope_pages(&self, page_base: u32) -> Vec<u32> {
        self.bucket
            .iter()
            .map(|&b| rope_page(b, page_base))
            .collect()
    }

    /// The page2d cos/sin tables for every row.
    #[must_use]
    pub fn rope_table(&self, page_base: u32) -> RopeTable {
        page2d_rope_table(&self.tok_pos, &self.rope_pages(page_base))
    }
}

fn no_fallback(why: &str) -> FocrError {
    FocrError::Other(anyhow::anyhow!(
        "unlimited-ocr-2 page2d RoPE needs a <PAGE><image> prompt and has no 1-D \
         fallback (OQ-V2-6): {why}"
    ))
}

/// The coordinates of one decoded row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeCoord {
    /// Page bucket (0 = no page).
    pub bucket: u32,
    /// Running position inside the bucket.
    pub tok_pos: u32,
    /// Page-axis RoPE coordinate, `page_base` applied and folded mod 2.
    pub rope_page: u32,
    /// Query page for the Page Gate (MU:516-518), −1 when negative.
    pub query_page: i32,
}

/// Per-sequence decode-time page state (SB:1451-1489 for RoPE, MU:516-518
/// for the query page). Each fed token advances it once, and the coordinates
/// it returns are that token's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodePageState {
    bucket: u32,
    tok_pos: u32,
    pf_cnt: Vec<u32>,
    /// Buckets `0..n_buckets`; a generated `<PAGE>` clamps at the last one.
    n_buckets: u32,
    /// Markers seen, for the query page (`mcum`).
    markers: u32,
    /// `pp`, the prompt pages the query page is relative to.
    prompt_pages: u32,
    page_base: u32,
}

impl DecodePageState {
    /// The HF seed (SB:1467-1477, MU:512): continue from the last prompt row,
    /// whose bucket is the instruction's 0. Applies to unwindowed runs,
    /// window 0 and recomputed windows.
    #[must_use]
    pub fn after_prompt(prompt: &PageCoords, page_base: u32) -> Self {
        let last = prompt.bucket.len() - 1;
        Self {
            bucket: prompt.bucket[last],
            tok_pos: prompt.tok_pos[last],
            pf_cnt: prompt.pf_cnt.clone(),
            n_buckets: prompt.pf_cnt.len() as u32,
            markers: prompt.markers,
            prompt_pages: prompt.prompt_pages,
            page_base,
        }
    }

    /// The seed of a reused window (W/`session/uocr_doc_reuse.py:418-437,
    /// 1103-1109`). Its rewritten prompt has no BOS and no instruction: bucket
    /// 1 is the copied tail page and buckets 2..=M the newly prefilled pages.
    /// The HF seed would leave the state in bucket M and send the forced
    /// `<PAGE>` there; this one resumes in bucket 1, so the forced `<PAGE>` lands
    /// in bucket 2 with `tok_pos = pf_cnt[2]` and query page 1.
    ///
    /// # Errors
    /// [`FocrError::Other`] if the rewritten prompt does not end on its last
    /// image or has fewer than two pages.
    pub fn reused_window(rewritten: &PageCoords, page_base: u32) -> FocrResult<Self> {
        let m = rewritten.prompt_pages;
        if rewritten.last_img + 1 != rewritten.bucket.len() || m < 2 || rewritten.markers != m {
            return Err(FocrError::Other(anyhow::anyhow!(
                "reused-window seed: the rewritten prompt must be pages only (tail + \
                 >= 1 new page) ending on its last image; got {m} pages, {} markers, \
                 last image at {} of {} rows",
                rewritten.markers,
                rewritten.last_img,
                rewritten.bucket.len()
            )));
        }
        Ok(Self {
            bucket: 1,
            // A non-marker row would continue bucket 1's running count, as the
            // wheel's per-bucket `cnt` does.
            tok_pos: rewritten.pf_cnt[1] - 1,
            pf_cnt: rewritten.pf_cnt.clone(),
            n_buckets: m + 1,
            markers: m + 1,
            prompt_pages: m,
            page_base,
        })
    }

    /// Advance over the fed token `id` and return its coordinates. An extra
    /// `<PAGE>` beyond the last bucket clamps there with `tok_pos = pf_cnt[last]`
    /// (HF; SGLang keeps the running count instead, DISC-U2-7).
    pub fn feed(&mut self, id: u32) -> DecodeCoord {
        if id == PAGE_TOKEN_ID {
            self.bucket = (self.bucket + 1).min(self.n_buckets - 1);
            self.tok_pos = self.pf_cnt[self.bucket as usize];
            self.markers += 1;
        } else {
            self.tok_pos += 1;
        }
        let q = i64::from(self.markers) - 1 - i64::from(self.prompt_pages);
        DecodeCoord {
            bucket: self.bucket,
            tok_pos: self.tok_pos,
            rope_page: rope_page(self.bucket, self.page_base),
            query_page: if q < 0 { -1 } else { q as i32 },
        }
    }
}

// ── Token Sieve router (U2-B2) ──────────────────────────────────────────────
//
// Constants are the checkpoint snapshot of `sparse_attention_config`, which is
// what inference reads (MD:56-64, 296-329): `_persist_router_cfg` rewrites the
// live config at construction. `DASH_STAGE0` must be unset in the oracle.

/// Rows per router block (`sparse_block_size`).
pub const BLOCK_SIZE: usize = 32;

/// Blocks in the local window (`sparse_local_block`, 128 rows). Block `c` is
/// remote for a query in block `qb` when `c ≤ qb − LOCAL_BLOCKS`.
pub const LOCAL_BLOCKS: usize = 4;

/// `_SCORE_NEG` (DA:741): the score of a block that is not remote.
pub const SCORE_NEG: f32 = -1.0e9;

/// `_ENTMAX_NEG` (MD:65): the entmax input of a non-candidate block.
pub const ENTMAX_NEG: f32 = -1.0e4;

/// `_ACT_CLAMP` (DA:746).
const ACT_CLAMP: f32 = 1.0e4;

/// `rms_norm_eps` of the router's q/k norms (`config.json`).
const ROUTER_NORM_EPS: f32 = 1e-6;

/// `inv_sqrt_d = 1.0 / math.sqrt(128)` (MD:387). The reference multiplies by
/// it (never divides by √128), and torch rounds a Python float to f32 when it
/// scales an f32 tensor, so this is that f32.
#[must_use]
pub fn inv_sqrt_d() -> f32 {
    (1.0 / (HEAD_DIM as f64).sqrt()) as f32
}

fn finite_or_zero(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

/// Per-layer routing parameters (`router._infer_params`, MD:320-328).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouterParams {
    /// `dash_scale_per_layer[l] / √128`, the entmax input scale (as f32).
    pub idx_scale: f32,
    /// `dash_floor_per_layer[l]`: priors are floored here before the log.
    pub floor: f32,
    /// `1 / dash_sigma`.
    pub inv_sigma: f32,
    /// K, the read budget in blocks (`dash_topk_blocks`).
    pub budget: usize,
    /// The entmax window before the block count caps it (`DASH_INFER_TOPM`).
    pub topm: usize,
}

impl RouterParams {
    /// Parameters from the config values, rounded the way the reference
    /// rounds them: each is a Python float that torch casts to f32 when it
    /// meets an f32 tensor.
    #[must_use]
    pub fn new(scale: f64, floor: f64, sigma: f64, budget: usize, topm: usize) -> Self {
        Self {
            idx_scale: (scale / (HEAD_DIM as f64).sqrt()) as f32,
            floor: floor as f32,
            inv_sigma: (1.0 / sigma) as f32,
            budget,
            topm,
        }
    }

    /// Unlimited-OCR-2's snapshot, the same for all 12 layers: scale 128,
    /// floor 1e-4, σ 1, K 12, M 16.
    #[must_use]
    pub fn uocr2() -> Self {
        Self::new(128.0, 1e-4, 1.0, 12, 16)
    }

    /// M, the entmax window of a forward with `nblk` blocks (MD:352). It
    /// depends on the forward's block count, never on the query row.
    #[must_use]
    pub fn window(&self, nblk: usize) -> usize {
        if self.topm > 0 {
            self.topm.min(nblk).max(1)
        } else {
            nblk
        }
    }

    /// K', the read slots of a window of `window` blocks (MD:356).
    #[must_use]
    pub fn read_slots(&self, window: usize) -> usize {
        self.budget.min(window).max(1)
    }
}

/// One layer's router weights (`self_attn.dash_router.*`), widened to f32
/// once. The `self_attn.indexer.*` tensors are not used at inference
/// (OQ-V2-5) and are not loaded here.
#[derive(Debug, Clone, PartialEq)]
pub struct SrSwaLayerWeights {
    /// `q_proj.weight`, `[HEAD_DIM, hidden]` row-major (no bias).
    pub q_proj: Vec<f32>,
    /// `k_proj.weight`, `[HEAD_DIM, hidden]` row-major (no bias).
    pub k_proj: Vec<f32>,
    /// `q_norm.weight` (Gemma style: the scale is `1 + w`).
    pub q_norm: Vec<f32>,
    /// `k_norm.weight`.
    pub k_norm: Vec<f32>,
    /// `head_cls`, the learned block-summary query.
    pub head_cls: Vec<f32>,
    /// The decoder hidden size the projections read (1280).
    pub hidden: usize,
}

impl SrSwaLayerWeights {
    /// Shape-checked construction.
    ///
    /// # Errors
    /// [`FocrError::Other`] if a projection is not `HEAD_DIM × hidden` or a
    /// vector is not `HEAD_DIM` long.
    pub fn new(
        hidden: usize,
        q_proj: Vec<f32>,
        k_proj: Vec<f32>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        head_cls: Vec<f32>,
    ) -> FocrResult<Self> {
        let proj = HEAD_DIM.checked_mul(hidden);
        for (name, len) in [("q_proj", q_proj.len()), ("k_proj", k_proj.len())] {
            if Some(len) != proj {
                return Err(router_shape(name, len, &format!("{HEAD_DIM} x {hidden}")));
            }
        }
        for (name, len) in [
            ("q_norm", q_norm.len()),
            ("k_norm", k_norm.len()),
            ("head_cls", head_cls.len()),
        ] {
            if len != HEAD_DIM {
                return Err(router_shape(name, len, &HEAD_DIM.to_string()));
            }
        }
        Ok(Self {
            q_proj,
            k_proj,
            q_norm,
            k_norm,
            head_cls,
            hidden,
        })
    }
}

fn router_shape(name: &str, len: usize, want: &str) -> FocrError {
    FocrError::Other(anyhow::anyhow!(
        "dash_router.{name}: {len} values, want {want}"
    ))
}

/// The router's Gemma-style RMSNorm over one row, in place (DA:844-855):
/// non-finite inputs become 0, `y = x·rsqrt(mean(x²) + 1e-6)` is clamped to
/// ±1e4, then `y·(1 + w)` is clamped to ±1e4 again.
pub fn gemma_rms_norm(x: &mut [f32], weight: &[f32]) {
    assert_eq!(x.len(), weight.len(), "gemma_rms_norm: weight length");
    for v in x.iter_mut() {
        *v = finite_or_zero(*v);
    }
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let rsqrt = 1.0 / (mean + ROUTER_NORM_EPS).sqrt();
    for (v, &w) in x.iter_mut().zip(weight) {
        let y = (*v * rsqrt).clamp(-ACT_CLAMP, ACT_CLAMP);
        *v = (y * (1.0 + w)).clamp(-ACT_CLAMP, ACT_CLAMP);
    }
}

/// The router's index queries and keys for the rows of `h` (MD:365-377).
///
/// `h` (`[rows, hidden]`) is the `input_layernorm` output that also feeds the
/// main q/k/v (SB:3692-3697); `rope` holds those rows' page2d tables (the
/// router rotates with the main attention's coordinates and frequencies,
/// MD:126-143). Non-finite values become 0 before the projections and after
/// the rotation. Returns `(iq, ik)`, each `[rows, HEAD_DIM]`; the index-key
/// cache holds `ik` as returned (post-norm, post-RoPE, f32; MD:440-444).
///
/// # Errors
/// [`FocrError::Other`] on a shape mismatch.
pub fn project(h: &Mat, w: &SrSwaLayerWeights, rope: &RopeTable) -> FocrResult<(Mat, Mat)> {
    let mut hs = h.clone();
    for v in &mut hs.data {
        *v = finite_or_zero(*v);
    }
    let mut iq = linear_no_bias(&hs, &w.q_proj, w.hidden, HEAD_DIM)?;
    let mut ik = linear_no_bias(&hs, &w.k_proj, w.hidden, HEAD_DIM)?;
    for r in 0..iq.rows {
        gemma_rms_norm(iq.row_mut(r), &w.q_norm);
        gemma_rms_norm(ik.row_mut(r), &w.k_norm);
    }
    apply_rope(&mut iq, rope)?;
    apply_rope(&mut ik, rope)?;
    for v in iq.data.iter_mut().chain(ik.data.iter_mut()) {
        *v = finite_or_zero(*v);
    }
    Ok((iq, ik))
}

/// Per-block summary keys over a KV sequence's index keys (stage0 =
/// `head_cls`, MD:391-408).
#[derive(Debug, Clone, PartialEq)]
pub struct BlockSummaries {
    /// `k̄`, `[nblk, HEAD_DIM]` row-major.
    pub kbar: Vec<f32>,
    /// Whether each block holds a real row (`chunk_valid`). Only the partial
    /// tail block can be short, and it is never remote.
    pub valid: Vec<bool>,
}

impl BlockSummaries {
    /// Number of blocks.
    #[must_use]
    pub fn nblk(&self) -> usize {
        self.valid.len()
    }

    /// Block `c`'s summary key.
    #[must_use]
    pub fn kbar(&self, c: usize) -> &[f32] {
        &self.kbar[c * HEAD_DIM..(c + 1) * HEAD_DIM]
    }
}

/// Summaries of the index keys `ik` (`[rows, HEAD_DIM]`, KV order) in
/// `ceil(rows / 32)` blocks (MD:391-408). For block `c` and its rows `j`:
/// `a_j = clamp(ik_j·head_cls × inv_sqrt_d, ±1e4)`, `w_j = exp(a_j − max a) /
/// max(Σ exp(a − max a), 1e-6)`, and `k̄_c = Σ_j w_j·ik_j` with non-finite
/// values mapped to 0. A zero `head_cls` is a mean pool.
#[must_use]
pub fn summaries(ik: &Mat, head_cls: &[f32]) -> BlockSummaries {
    assert_eq!(ik.cols, HEAD_DIM, "summaries: index key width");
    assert_eq!(head_cls.len(), HEAD_DIM, "summaries: head_cls length");
    let nblk = ik.rows.div_ceil(BLOCK_SIZE);
    let mut kbar = vec![0.0f32; nblk * HEAD_DIM];
    let mut valid = vec![false; nblk];
    for (c, out) in kbar.as_chunks_mut::<HEAD_DIM>().0.iter_mut().enumerate() {
        let rows = c * BLOCK_SIZE..((c + 1) * BLOCK_SIZE).min(ik.rows);
        valid[c] = !rows.is_empty();
        block_summary(
            &ik.data[rows.start * HEAD_DIM..rows.end * HEAD_DIM],
            head_cls,
            out,
        );
    }
    BlockSummaries { kbar, valid }
}

/// One block's `k̄` from its index keys `ik` (`[n, HEAD_DIM]`, n ≤ 32), as
/// [`summaries`] defines it. The cache (U2-B4) calls this too, so its
/// incremental summaries are this computation by construction.
fn block_summary(ik: &[f32], head_cls: &[f32], out: &mut [f32; HEAD_DIM]) {
    let inv = inv_sqrt_d();
    let keys = ik.as_chunks::<HEAD_DIM>().0;
    let aw: Vec<f32> = keys
        .iter()
        .map(|k| (dot(k, head_cls) * inv).clamp(-ACT_CLAMP, ACT_CLAMP))
        .collect();
    let max = aw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = aw.iter().map(|&a| (a - max).exp()).collect();
    let denom = e.iter().sum::<f32>().max(1e-6);
    out.fill(0.0);
    for (k, &ej) in keys.iter().zip(&e) {
        let wj = ej / denom;
        for (o, &x) in out.iter_mut().zip(k) {
            *o += wj * x;
        }
    }
    for o in out.iter_mut() {
        *o = finite_or_zero(*o);
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Remote block scores of the query `iq` at KV position `q_pos` (MD:409-417):
/// `clamp(iq·k̄_c × inv_sqrt_d, ±1e4)` for every valid block `c ≤ q_pos/32 −
/// 4`, [`SCORE_NEG`] for the rest (the local window, the future, and invalid
/// blocks). `q_pos` is window-local in windowed mode.
#[must_use]
pub fn scores(iq: &[f32], q_pos: usize, sums: &BlockSummaries) -> Vec<f32> {
    let inv = inv_sqrt_d();
    let qb = q_pos / BLOCK_SIZE;
    (0..sums.nblk())
        .map(|c| {
            if c + LOCAL_BLOCKS <= qb && sums.valid[c] {
                (dot(iq, sums.kbar(c)) * inv).clamp(-ACT_CLAMP, ACT_CLAMP)
            } else {
                SCORE_NEG
            }
        })
        .collect()
}

/// The image pages each block of a KV sequence holds, from its per-row page
/// ids (`PageCoords::img_page`: −1 for non-image rows).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockPages {
    /// Distinct image pages per block, ascending.
    pages: Vec<Vec<i32>>,
    /// Lowest image page per block, −1 when it holds none (MU:510-511).
    pub bmin: Vec<i32>,
    /// Highest image page per block, −1 when it holds none (MU:509).
    pub bmax: Vec<i32>,
}

impl BlockPages {
    /// Group `img_page` into `ceil(len / 32)` blocks.
    #[must_use]
    pub fn build(img_page: &[i32]) -> Self {
        let mut pages: Vec<Vec<i32>> = img_page
            .chunks(BLOCK_SIZE)
            .map(|blk| blk.iter().copied().filter(|&p| p >= 0).collect())
            .collect();
        for p in &mut pages {
            p.sort_unstable();
            p.dedup();
        }
        let bmin = pages
            .iter()
            .map(|p| p.first().copied().unwrap_or(-1))
            .collect();
        let bmax = pages
            .iter()
            .map(|p| p.last().copied().unwrap_or(-1))
            .collect();
        Self { pages, bmin, bmax }
    }

    /// Number of blocks.
    #[must_use]
    pub fn nblk(&self) -> usize {
        self.pages.len()
    }
}

/// The prefill Page Gate of a query on page `qp` (`image_bucket_constraint =
/// true`; SB:2629-2649, 2706-2737): the blocks holding an image row of page
/// `qp`. A query with no such block, or with `qp < 0` (BOS, the instruction),
/// may read every block (the per-query fallback).
#[must_use]
pub fn page_gate_prefill(blocks: &BlockPages, qp: i32) -> Vec<bool> {
    let allow: Vec<bool> = blocks
        .pages
        .iter()
        .map(|p| qp >= 0 && p.binary_search(&qp).is_ok())
        .collect();
    if allow.contains(&true) {
        allow
    } else {
        vec![true; allow.len()]
    }
}

/// The decode Page Gate of a query on page `qp` (MU:514-528): block `c` of
/// the prompt is allowed when `bmin[c] ≥ 0` and `bmin[c] ≤ qp ≤ bmax[c]`;
/// blocks past the prompt's (generated text) never are. An open row (`qp <
/// 0`, or no block allowed) allows all `nblk` blocks.
#[must_use]
pub fn page_gate_decode(bmin: &[i32], bmax: &[i32], qp: i32, nblk: usize) -> Vec<bool> {
    assert_eq!(bmin.len(), bmax.len(), "page_gate_decode: bmin/bmax");
    let mut allow = vec![false; nblk];
    for (c, a) in allow.iter_mut().enumerate().take(bmin.len()) {
        *a = qp >= 0 && bmin[c] >= 0 && bmin[c] <= qp && qp <= bmax[c];
    }
    if allow.contains(&true) {
        allow
    } else {
        vec![true; nblk]
    }
}

/// The candidate blocks of one query (MD:346-348): remote blocks (score >
/// −5e8) that `allow` admits; when that leaves none, every remote block (an
/// open row). With no remote block the set is empty and the query reads only
/// its local window.
#[must_use]
pub fn candidates(scores: &[f32], allow: Option<&[bool]>) -> Vec<bool> {
    let remote: Vec<bool> = scores.iter().map(|&s| s > SCORE_NEG * 0.5).collect();
    let Some(allow) = allow else {
        return remote;
    };
    assert_eq!(allow.len(), scores.len(), "candidates: allow length");
    let cand: Vec<bool> = remote.iter().zip(allow).map(|(&r, &a)| r && a).collect();
    if cand.contains(&true) { cand } else { remote }
}

/// The entmax input of every block (MD:349-350): `score × idx_scale` for a
/// candidate, [`ENTMAX_NEG`] otherwise, and [`ENTMAX_NEG`] for anything
/// non-finite.
#[must_use]
pub fn entmax_inputs(scores: &[f32], cand: &[bool], p: &RouterParams) -> Vec<f32> {
    scores
        .iter()
        .zip(cand)
        .map(|(&s, &c)| {
            let x = if c { s * p.idx_scale } else { ENTMAX_NEG };
            if x.is_finite() { x } else { ENTMAX_NEG }
        })
        .collect()
}

/// 1.5-entmax over `x` (DA:772-800) in the reference's operation order:
/// `z = (x − max x)/2`, sorted descending; prefix means `mean_k` and `meansq_k`;
/// `τ_k = mean_k − √max((1 − k(meansq_k − mean_k²))/k, 0)`; the support is
/// `#{k : τ_k ≤ z_(k)}` (at least 1) and `p = max(z − τ_support, 0)²`.
///
/// The prefix sums accumulate in f64 and round to f32 per step: torch's CPU
/// `cumsum` does that for f32 (`acc_type<float, CPU>` is double), and the
/// oracle is CPU. Returns the probabilities in input order and the support.
#[must_use]
pub fn entmax15(x: &[f32]) -> (Vec<f32>, usize) {
    if x.is_empty() {
        return (Vec::new(), 0);
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let z: Vec<f32> = x.iter().map(|&v| (v - max) / 2.0).collect();
    let mut sorted = z.clone();
    sorted.sort_unstable_by(|a, b| b.total_cmp(a));
    let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
    let mut support = 0usize;
    let mut taus = Vec::with_capacity(sorted.len());
    for (k, &zk) in sorted.iter().enumerate() {
        sum += f64::from(zk);
        sum_sq += f64::from(zk * zk);
        let rho = (k + 1) as f32;
        let mean = sum as f32 / rho;
        let mean_sq = sum_sq as f32 / rho;
        let ss = rho * (mean_sq - mean * mean);
        let delta = (1.0 - ss) / rho;
        let tau = mean - delta.max(0.0).sqrt();
        support += usize::from(tau <= zk);
        taus.push(tau);
    }
    let support = support.max(1);
    let tau = taus[support - 1];
    let p = z
        .iter()
        .map(|&v| {
            let d = (v - tau).max(0.0);
            d * d
        })
        .collect();
    (p, support)
}

/// Which entmax the router runs (D21).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntmaxDomain {
    /// HF (MD:346-362), the parity default: entmax over the top-M window of
    /// all blocks, then the candidates among its first K' slots.
    TopM,
    /// SGLang (W/`layers/attention/unlimited_dash_ops/route_triton.py`):
    /// entmax over every candidate and none of the rest; read the K best
    /// candidates by score. SGLang solves τ by a 24-step bisection and the
    /// read set by a 40-step one, so a tie at K-th place can read fewer than
    /// K blocks there; this is the exact sort-based form (DISC-U2-1).
    AllCandidates,
}

/// One query's read set.
#[derive(Debug, Clone, PartialEq)]
pub struct Route {
    /// Selected remote blocks, ascending.
    pub blocks: Vec<u32>,
    /// The bias added to every logit of the block at the same position.
    pub bias: Vec<f32>,
    /// The entmax support size (a routing-margin diagnostic, U2-C1).
    pub support: usize,
}

/// The top-M window of the entmax inputs (MD:353-354).
#[derive(Debug, Clone, PartialEq)]
pub struct TopM {
    /// `(entmax input, block)` in slot order: descending, and among equal
    /// inputs the lower block first.
    pub slots: Vec<(f32, u32)>,
    /// The entmax prior of each slot.
    pub prior: Vec<f32>,
    /// The entmax support size.
    pub support: usize,
}

/// Slot order: descending input, ties by lower block index. torch's CPU
/// `topk` leaves the order of equal values to `std::nth_element` /
/// `std::partial_sort`, which is not a stable order, so no rule reproduces it
/// in general; ties matter only between equal candidate scores at the K'-th
/// slot or a candidate at exactly −1e4, and C1 classes such flips.
fn slot_order(a: &(f32, u32), b: &(f32, u32)) -> std::cmp::Ordering {
    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
}

/// The top-`window` entmax window of `x` and its prior (MD:351-354).
#[must_use]
pub fn entmax15_topm(x: &[f32], window: usize) -> TopM {
    let mut all: Vec<(f32, u32)> = x.iter().zip(0u32..).map(|(&v, c)| (v, c)).collect();
    let m = window.min(all.len());
    if m < all.len() && m > 0 {
        all.select_nth_unstable_by(m - 1, slot_order);
    }
    all.truncate(m);
    all.sort_unstable_by(slot_order);
    window_prior(all)
}

fn window_prior(slots: Vec<(f32, u32)>) -> TopM {
    let xv: Vec<f32> = slots.iter().map(|&(v, _)| v).collect();
    let (prior, support) = entmax15(&xv);
    TopM {
        slots,
        prior,
        support,
    }
}

/// The read set and bias of a [`TopM`] window (MD:355-362): the candidates
/// among its first K' slots, each with `bias = (ln max(p, floor) − μ)/σ`, μ
/// being the mean of those log-priors. A zero-prior candidate in those slots
/// is still read, with bias `ln(floor) − μ`. No clamp at inference.
#[must_use]
pub fn select_bias(top: &TopM, cand: &[bool], p: &RouterParams) -> Route {
    let k = p.read_slots(top.slots.len()).min(top.slots.len());
    let picked: Vec<(u32, f32)> = top.slots[..k]
        .iter()
        .zip(&top.prior)
        .filter(|((_, c), _)| cand[*c as usize])
        .map(|(&(_, c), &pr)| (c, pr))
        .collect();
    finish_route(picked, top.support, p)
}

/// Center the floored log-priors of the read set and sort it by block.
fn finish_route(mut picked: Vec<(u32, f32)>, support: usize, p: &RouterParams) -> Route {
    let logp: Vec<f32> = picked.iter().map(|&(_, pr)| pr.max(p.floor).ln()).collect();
    let mu = logp.iter().sum::<f32>() / logp.len().max(1) as f32;
    for ((_, b), &lp) in picked.iter_mut().zip(&logp) {
        *b = (lp - mu) * p.inv_sigma;
    }
    picked.sort_unstable_by_key(|&(c, _)| c);
    Route {
        blocks: picked.iter().map(|&(c, _)| c).collect(),
        bias: picked.iter().map(|&(_, b)| b).collect(),
        support,
    }
}

/// [`EntmaxDomain::AllCandidates`]: entmax over the candidates' inputs only,
/// then the `min(K, 16)` candidates with the highest input (all of them when
/// there are no more), μ taken over that read set. Zero-prior candidates in
/// it are read, with bias `ln(floor) − μ`.
#[must_use]
pub fn entmax15_all(x: &[f32], cand: &[bool], p: &RouterParams) -> Route {
    let mut slots: Vec<(f32, u32)> = x
        .iter()
        .zip(cand)
        .zip(0u32..)
        .filter(|&((_, &c), _)| c)
        .map(|((&v, _), b)| (v, b))
        .collect();
    slots.sort_unstable_by(slot_order);
    let xv: Vec<f32> = slots.iter().map(|&(v, _)| v).collect();
    let (prior, support) = entmax15(&xv);
    let k = p.read_slots(p.topm.max(1)).min(slots.len());
    let picked = slots[..k]
        .iter()
        .zip(&prior)
        .map(|(&(_, c), &pr)| (c, pr))
        .collect();
    finish_route(picked, support, p)
}

/// Route one query from its block scores and Page Gate (MD:338-362).
#[must_use]
pub fn route(
    scores: &[f32],
    allow: Option<&[bool]>,
    p: &RouterParams,
    domain: EntmaxDomain,
) -> Route {
    let cand = candidates(scores, allow);
    let x = entmax_inputs(scores, &cand, p);
    match domain {
        EntmaxDomain::TopM => {
            let top = entmax15_topm(&x, p.window(scores.len()));
            select_bias(&top, &cand, p)
        }
        EntmaxDomain::AllCandidates => entmax15_all(&x, &cand, p),
    }
}

/// [`EntmaxDomain::TopM`] from the candidates alone (U2-F2's lever): `cands`
/// holds each candidate's block and entmax input, `nblk` the forward's block
/// count. Every non-candidate's input is [`ENTMAX_NEG`], so the ones the full
/// window would hold are the `min(M, nblk − #cand)` lowest-indexed
/// non-candidates; they are merged into the descending order as pads, not
/// appended, because a candidate's input can sit below −1e4 (its score is
/// clamped to ±1e4 before the ×11.3 scale) and then ranks under them. Equal
/// to [`route`] over the full score row.
#[must_use]
pub fn route_topm_candidates(cands: &[(u32, f32)], nblk: usize, p: &RouterParams) -> Route {
    let mut is_cand = vec![false; nblk];
    for &(c, _) in cands {
        is_cand[c as usize] = true;
    }
    let m = p.window(nblk).min(nblk);
    let mut slots: Vec<(f32, u32)> = cands.iter().map(|&(c, v)| (v, c)).collect();
    slots.extend(
        (0u32..)
            .zip(&is_cand)
            .filter(|&(_, &c)| !c)
            .map(|(b, _)| (ENTMAX_NEG, b))
            .take(m),
    );
    slots.sort_unstable_by(slot_order);
    slots.truncate(m);
    select_bias(&window_prior(slots), &is_cand, p)
}

/// Route every row of a prefill from an empty cache (MD:433-446): row `r` sits
/// at KV position `r`, the Page Gate is [`page_gate_prefill`] on the row's
/// query page, and the forward's `nblk = ceil(S / 32)` sets the window M of
/// **every** row, early rows included (MD:437, 351-352). `iq`/`ik` come from
/// [`project`]; `coords` describes the same rows.
///
/// # Errors
/// [`FocrError::Other`] if the row counts disagree.
pub fn prefill_routes(
    iq: &Mat,
    ik: &Mat,
    coords: &PageCoords,
    head_cls: &[f32],
    p: &RouterParams,
    domain: EntmaxDomain,
) -> FocrResult<Vec<Route>> {
    if iq.rows != ik.rows || iq.rows != coords.bucket.len() {
        return Err(FocrError::Other(anyhow::anyhow!(
            "prefill_routes: {} query rows, {} key rows, {} coordinate rows",
            iq.rows,
            ik.rows,
            coords.bucket.len()
        )));
    }
    let sums = summaries(ik, head_cls);
    let blocks = BlockPages::build(&coords.img_page);
    Ok((0..iq.rows)
        .map(|r| {
            let sc = scores(iq.row(r), r, &sums);
            let allow = page_gate_prefill(&blocks, coords.query_page[r]);
            route(&sc, Some(&allow), p, domain)
        })
        .collect())
}

// ── Block-sparse biased attention (U2-B3) ───────────────────────────────────
//
// A query attends its 128-row local window without bias and the rows of its
// selected remote blocks with that block's bias (MD:454-481 decode,
// DA:1230-1261 dense prefill):
//
//   local(j)  = q − 128 < j ≤ q
//   remote(j) = j ≤ q ∧ block(j) ∈ B ∧ ¬local(j)      (local wins on overlap)
//   logit_j   = (q·k_j) × f32(1/√128) + (remote(j) ? β_block(j) : 0)
//   out       = softmax over local ∪ remote of logit, times v
//
// Every head shares the layer's selection (MD:480). franken_ocr runs it in f32
// against the f32 oracle (DISC-U2-5). Both paths below visit the attended keys
// in ascending position order and reduce in that order, so they agree bit for
// bit: the dense one only adds exact zeros for the masked keys.

/// Attention heads (`num_attention_heads` = `num_key_value_heads` = 10).
pub const NUM_HEADS: usize = 10;

/// The local window in rows (`local_window = sparse_local_block × block_size`).
pub const LOCAL_WINDOW: usize = LOCAL_BLOCKS * BLOCK_SIZE;

/// The remote blocks one query reads and the bias each adds to the logits of
/// its rows, shared by every head of the layer. A [`Route`] maps onto it one
/// to one.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BlockSelection {
    blocks: Vec<u32>,
    bias: Vec<f32>,
}

impl BlockSelection {
    /// A selection of `blocks`, each with the bias at the same position.
    ///
    /// # Errors
    /// [`FocrError::Other`] unless `blocks` is strictly ascending and as long
    /// as `bias`.
    pub fn new(blocks: Vec<u32>, bias: Vec<f32>) -> FocrResult<Self> {
        if blocks.len() != bias.len() || !blocks.windows(2).all(|w| w[0] < w[1]) {
            return Err(FocrError::Other(anyhow::anyhow!(
                "block selection: {} blocks with {} biases; blocks must be strictly \
                 ascending: {blocks:?}",
                blocks.len(),
                bias.len()
            )));
        }
        Ok(Self { blocks, bias })
    }

    /// Selected blocks, ascending.
    #[must_use]
    pub fn blocks(&self) -> &[u32] {
        &self.blocks
    }

    /// The bias of each selected block.
    #[must_use]
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }
}

impl From<Route> for BlockSelection {
    /// A route's blocks are ascending and unique by construction.
    fn from(route: Route) -> Self {
        Self {
            blocks: route.blocks,
            bias: route.bias,
        }
    }
}

/// Read access to one layer's keys and values, block by block. The gathered
/// path reads only through it; U2-B4's cache implements it as well.
pub trait BlockKv {
    /// Rows in the sequence; positions index `0..rows()`.
    fn rows(&self) -> usize;

    /// Head `h`'s keys in block `b`: contiguous `[n, HEAD_DIM]` for the
    /// block's `n` live rows.
    fn keys(&self, h: usize, b: usize) -> &[f32];

    /// Head `h`'s values in block `b`, laid out like [`BlockKv::keys`].
    fn values(&self, h: usize, b: usize) -> &[f32];
}

/// An in-memory head-major [`BlockKv`]: head `h`'s keys are one `[rows,
/// HEAD_DIM]` run.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadMajorKv {
    rows: usize,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl HeadMajorKv {
    /// From per-row K and V, each `[rows, NUM_HEADS × HEAD_DIM]` (the
    /// projection layout, RoPE already applied to K).
    ///
    /// # Errors
    /// [`FocrError::Other`] if the shapes disagree or are not that width.
    pub fn from_rows(k: &Mat, v: &Mat) -> FocrResult<Self> {
        let width = NUM_HEADS * HEAD_DIM;
        if k.shape() != v.shape() || k.cols != width {
            return Err(FocrError::Other(anyhow::anyhow!(
                "srswa kv: K {:?} and V {:?}, want [rows, {width}] each",
                k.shape(),
                v.shape()
            )));
        }
        let split = |m: &Mat| -> Vec<Vec<f32>> {
            (0..NUM_HEADS)
                .map(|h| {
                    (0..m.rows)
                        .flat_map(|r| &m.row(r)[h * HEAD_DIM..(h + 1) * HEAD_DIM])
                        .copied()
                        .collect()
                })
                .collect()
        };
        Ok(Self {
            rows: k.rows,
            k: split(k),
            v: split(v),
        })
    }

    fn block(&self, b: usize) -> std::ops::Range<usize> {
        b * BLOCK_SIZE * HEAD_DIM..((b + 1) * BLOCK_SIZE).min(self.rows) * HEAD_DIM
    }
}

impl BlockKv for HeadMajorKv {
    fn rows(&self) -> usize {
        self.rows
    }

    fn keys(&self, h: usize, b: usize) -> &[f32] {
        &self.k[h][self.block(b)]
    }

    fn values(&self, h: usize, b: usize) -> &[f32] {
        &self.v[h][self.block(b)]
    }
}

/// A contiguous run of attended rows `[from, to)` inside `block`, and the bias
/// their logits take.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Run {
    block: usize,
    from: usize,
    to: usize,
    bias: f32,
}

/// The rows the query at `q_pos` attends, as runs in ascending position order:
/// each selected block's causal rows outside the local window (with its bias),
/// then the local window (no bias). A selected block's rows inside the window
/// are read once, as local rows.
fn attended_runs(q_pos: usize, sel: &BlockSelection) -> Vec<Run> {
    let local_start = (q_pos + 1).saturating_sub(LOCAL_WINDOW);
    let mut runs = Vec::with_capacity(sel.blocks.len() + LOCAL_BLOCKS + 1);
    for (&b, &bias) in sel.blocks.iter().zip(&sel.bias) {
        let start = b as usize * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(local_start);
        if start < end {
            runs.push(Run {
                block: b as usize,
                from: 0,
                to: end - start,
                bias,
            });
        }
    }
    let mut j = local_start;
    while j <= q_pos {
        let b = j / BLOCK_SIZE;
        let end = ((b + 1) * BLOCK_SIZE).min(q_pos + 1);
        runs.push(Run {
            block: b,
            from: j - b * BLOCK_SIZE,
            to: end - b * BLOCK_SIZE,
            bias: 0.0,
        });
        j = end;
    }
    runs
}

/// Max-subtracted softmax in place, accumulated in f32.
fn softmax_in_place(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut den = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        den += *v;
    }
    for v in x.iter_mut() {
        *v /= den;
    }
}

/// One head of one query over its runs: logits in run order, softmax, then
/// `Σ p_j·v_j` in the same order. `logits` is reusable scratch.
fn attend_head(
    qh: &[f32],
    runs: &[Run],
    kv: &impl BlockKv,
    h: usize,
    logits: &mut Vec<f32>,
    out: &mut [f32],
) {
    let scale = inv_sqrt_d();
    logits.clear();
    for run in runs {
        let keys = kv.keys(h, run.block);
        for j in run.from..run.to {
            logits.push(dot(qh, &keys[j * HEAD_DIM..(j + 1) * HEAD_DIM]) * scale + run.bias);
        }
    }
    softmax_in_place(logits);
    out.fill(0.0);
    let mut probs = logits.iter();
    for run in runs {
        let values = kv.values(h, run.block);
        for j in run.from..run.to {
            let p = *probs.next().expect("one probability per attended row");
            for (o, &v) in out
                .iter_mut()
                .zip(&values[j * HEAD_DIM..(j + 1) * HEAD_DIM])
            {
                *o += p * v;
            }
        }
    }
}

fn check_attend_inputs(
    q: &Mat,
    q_pos: &[usize],
    sel: &[BlockSelection],
    kv: &impl BlockKv,
) -> FocrResult<()> {
    let width = NUM_HEADS * HEAD_DIM;
    if q.cols != width || q_pos.len() != q.rows || sel.len() != q.rows {
        return Err(FocrError::Other(anyhow::anyhow!(
            "srswa attention: q {:?} (want [rows, {width}]), {} positions, {} selections",
            q.shape(),
            q_pos.len(),
            sel.len()
        )));
    }
    if let Some(&p) = q_pos.iter().find(|&&p| p >= kv.rows()) {
        return Err(FocrError::Other(anyhow::anyhow!(
            "srswa attention: query position {p} past the {} KV rows",
            kv.rows()
        )));
    }
    Ok(())
}

/// SR-SWA attention, the gathered production path: each query reads only its
/// local window and selected blocks. `q` is `[rows, NUM_HEADS × HEAD_DIM]`
/// (RoPE applied), `q_pos[r]` row `r`'s KV position and `sel[r]` its selection;
/// `kv` holds every row up to the last query. Prefill passes all its rows,
/// decode one. Returns the concatenated heads, `[rows, NUM_HEADS × HEAD_DIM]`,
/// ready for `o_proj`.
///
/// (row, head) pairs run on the Rayon pool; each is reduced alone in a fixed
/// order, so the result does not depend on the thread count. Hold no lock
/// across this call (doctrine #5).
///
/// # Errors
/// [`FocrError::Other`] on a shape mismatch or a query past the KV rows.
pub fn attend(
    q: &Mat,
    q_pos: &[usize],
    sel: &[BlockSelection],
    kv: &(impl BlockKv + Sync),
) -> FocrResult<Mat> {
    check_attend_inputs(q, q_pos, sel, kv)?;
    let mut out = Mat::zeros(q.rows, NUM_HEADS * HEAD_DIM);
    out.data
        .par_chunks_mut(HEAD_DIM)
        .enumerate()
        .for_each_init(Vec::new, |logits, (i, dst)| {
            let (r, h) = (i / NUM_HEADS, i % NUM_HEADS);
            let runs = attended_runs(q_pos[r], &sel[r]);
            let qh = &q.row(r)[h * HEAD_DIM..(h + 1) * HEAD_DIM];
            attend_head(qh, &runs, kv, h, logits, dst);
        });
    Ok(out)
}

/// The dense oracle (DA:1230-1261's structure): every key `0..rows` gets a
/// logit, `−∞` off the attended set and the block bias on remote rows, then
/// one softmax over the full row. Equal to [`attend`] bit for bit; for tests
/// and diagnosis (U2-B5 puts it behind `FOCR_SRSWA_DENSE=1`).
///
/// # Errors
/// As [`attend`].
pub fn attend_dense(
    q: &Mat,
    q_pos: &[usize],
    sel: &[BlockSelection],
    kv: &impl BlockKv,
) -> FocrResult<Mat> {
    check_attend_inputs(q, q_pos, sel, kv)?;
    let scale = inv_sqrt_d();
    let rows = kv.rows();
    let mut out = Mat::zeros(q.rows, NUM_HEADS * HEAD_DIM);
    let mut logits = vec![0.0f32; rows];
    for (r, (&qp, s)) in q_pos.iter().zip(sel).enumerate() {
        let bias_of = |b: usize| {
            s.blocks
                .iter()
                .position(|&x| x as usize == b)
                .map(|i| s.bias[i])
        };
        for h in 0..NUM_HEADS {
            let qh = &q.row(r)[h * HEAD_DIM..(h + 1) * HEAD_DIM];
            for (j, logit) in logits.iter_mut().enumerate() {
                let (b, o) = (j / BLOCK_SIZE, j % BLOCK_SIZE);
                let local = j <= qp && j + LOCAL_WINDOW > qp;
                let remote = j <= qp && !local && bias_of(b).is_some();
                *logit = if local || remote {
                    let k = &kv.keys(h, b)[o * HEAD_DIM..(o + 1) * HEAD_DIM];
                    dot(qh, k) * scale
                        + if remote {
                            bias_of(b).unwrap_or(0.0)
                        } else {
                            0.0
                        }
                } else {
                    f32::NEG_INFINITY
                };
            }
            softmax_in_place(&mut logits);
            let dst = &mut out.row_mut(r)[h * HEAD_DIM..(h + 1) * HEAD_DIM];
            for (j, &p) in logits.iter().enumerate() {
                let (b, o) = (j / BLOCK_SIZE, j % BLOCK_SIZE);
                let v = &kv.values(h, b)[o * HEAD_DIM..(o + 1) * HEAD_DIM];
                for (d, &vd) in dst.iter_mut().zip(v) {
                    *d += p * vd;
                }
            }
        }
    }
    Ok(out)
}

// ── Layer cache and block retirement (U2-B4) ────────────────────────────────
//
// One layer's K, V and index keys live in 32-row block slabs, so one block can
// be freed on its own (plan §5.12, D9). `k̄`, `valid`, bmin and bmax are kept
// for every **logical** block, retired or not, so block indices, `nblk` and
// the router window M never change.
//
// At decode query position q on query page qp ≥ 0, an unpinned block c is dead
// once (1) `bmin[c] = −1` or `bmax[c] < qp` (no non-open query can allow it
// again: qp never decreases) and (2) its last row ≤ q − 128 (it has left the
// local window (q − 128, q], MD:472-473, for good). A non-open row never allows
// a dead block, so the block is never a candidate there and every selection,
// bias and attention output is the unretired run's. An open row reads every
// remote block. Open rows meeting retired blocks happen only on guard-forced
// steps (the proof in plan §5.12); `decode_route` checks it on every row.
//
// Page state is the same in every layer, so it lives once per sequence in
// `DecodePageState` (U2-B1) and the query page is an argument here.

/// The kill switch for block retirement.
pub const RETIRE_ENV: &str = "FOCR_SRSWA_RETIRE";

/// Whether a [`RETIRE_ENV`] value leaves retirement on: everything but
/// `0`/`off`/`false`/`no` does, unset included.
#[must_use]
pub fn retire_enabled_for(value: Option<&str>) -> bool {
    !matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "off" | "false" | "no")
    )
}

/// [`RETIRE_ENV`], read once per process.
#[must_use]
pub fn retire_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| retire_enabled_for(std::env::var(RETIRE_ENV).ok().as_deref()))
}

/// Bytes one row costs one layer: K and V (`2 × 1280` f32) and the index key
/// (128 f32). Over 12 layers that is plan §6.3's 120 KiB + 6 KiB per token.
pub const ROW_BYTES: usize = (2 * NUM_HEADS * HEAD_DIM + HEAD_DIM) * size_of::<f32>();

/// Bytes kept for every logical block, retired or not: `k̄`, bmin, bmax,
/// `valid` and the pin flag.
pub const BLOCK_META_BYTES: usize = HEAD_DIM * size_of::<f32>() + 2 * size_of::<i32>() + 2;

/// One head's rows inside a block slab.
const HEAD_STRIDE: usize = BLOCK_SIZE * HEAD_DIM;

/// One block's rows: K and V head-major (`[NUM_HEADS][32][HEAD_DIM]`, so a
/// head's rows are one contiguous run for [`BlockKv`]) and the index keys
/// (`[32][HEAD_DIM]`, the layout [`block_summary`] reads).
#[derive(Debug, Clone, PartialEq)]
struct Slab {
    k: Box<[f32]>,
    v: Box<[f32]>,
    ik: Box<[f32]>,
}

impl Slab {
    fn new() -> Self {
        let kv = || vec![0.0f32; NUM_HEADS * HEAD_STRIDE].into_boxed_slice();
        Self {
            k: kv(),
            v: kv(),
            ik: vec![0.0f32; HEAD_STRIDE].into_boxed_slice(),
        }
    }
}

/// Why a decode row is open: its candidates are every remote block (MU:520-528,
/// MD:346-348).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenRow {
    /// `qp < 0`: no generated `<PAGE>` yet.
    NoPage,
    /// No block holds an image of page qp, i.e. qp ≥ N after an extra
    /// `<PAGE>`.
    NoAllowedBlock,
    /// Blocks of page qp exist but none is remote yet.
    NoRemoteAllowed,
}

/// The detector's event: an open row met retired blocks on a step whose output
/// is not guard-forced, so its read set may differ from the unretired run's.
/// The caller re-runs the window (or the document) without retirement, the
/// deterministic fallback of plan §5.12.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetireFallback {
    /// The query's KV position.
    pub q_pos: usize,
    /// The query page.
    pub query_page: i32,
    /// Why the row was open.
    pub open: OpenRow,
    /// Retired blocks at that point.
    pub dead_blocks: usize,
}

impl std::fmt::Display for RetireFallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "srswa retirement: open decode row ({:?}) at position {} on query page {} with \
             {} retired blocks; re-run without retirement (OQ-V2-11)",
            self.open, self.q_pos, self.query_page, self.dead_blocks
        )
    }
}

impl std::error::Error for RetireFallback {}

/// A decode query's route, and whether its row was open.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeRoute {
    /// The read set and biases.
    pub route: Route,
    /// `Some` for an open row.
    pub open: Option<OpenRow>,
}

/// Whether a decode row on page `qp` is open, from the prompt's bmin/bmax and
/// the row's scores (remote iff above `SCORE_NEG / 2`).
fn decode_open_row(bmin: &[i32], bmax: &[i32], qp: i32, scores: &[f32]) -> Option<OpenRow> {
    if qp < 0 {
        return Some(OpenRow::NoPage);
    }
    let mut allowed = false;
    for (c, (&lo, &hi)) in bmin.iter().zip(bmax).enumerate() {
        if lo >= 0 && lo <= qp && qp <= hi {
            if scores[c] > SCORE_NEG * 0.5 {
                return None;
            }
            allowed = true;
        }
    }
    Some(if allowed {
        OpenRow::NoRemoteAllowed
    } else {
        OpenRow::NoAllowedBlock
    })
}

/// Rows read back by [`deserialize_tail`], in the layout
/// [`SrSwaLayerCache::append`] takes.
#[derive(Debug, Clone, PartialEq)]
pub struct TailRows {
    /// `[n, NUM_HEADS × HEAD_DIM]`, RoPE applied.
    pub k: Mat,
    /// `[n, NUM_HEADS × HEAD_DIM]`.
    pub v: Mat,
    /// `[n, HEAD_DIM]`, post-norm and post-RoPE.
    pub ik: Mat,
}

/// Tail format: this magic, then little-endian u32 version, heads and head
/// dim, a u64 row count, and per row its K, V and index key as little-endian
/// f32 (U2-D1a splices, U2-D4 checkpoints).
const TAIL_MAGIC: &[u8; 8] = b"FOCRSRSW";
const TAIL_VERSION: u32 = 1;
const TAIL_HEADER: usize = 8 + 3 * 4 + 8;

/// One layer's SR-SWA cache: K, V, index keys, block summaries, bmin/bmax,
/// the retired-block set and memory accounting (U2-B4). Reads go through
/// [`BlockKv`]; reading a retired block panics, because the detector keeps
/// every route off them.
#[derive(Debug, Clone, PartialEq)]
pub struct SrSwaLayerCache {
    head_cls: Vec<f32>,
    rows: usize,
    /// Per logical block, `None` once retired.
    slabs: Vec<Option<Slab>>,
    sums: BlockSummaries,
    bmin: Vec<i32>,
    bmax: Vec<i32>,
    pinned: Vec<bool>,
    text_tail_margin: bool,
    retire: bool,
    dead: usize,
    peak_bytes: usize,
    /// The last `(qp, q)` passed to `retire_dead_blocks`; neither may fall.
    last_step: Option<(i32, usize)>,
}

impl SrSwaLayerCache {
    /// An empty cache for a layer whose router summary vector is `head_cls`
    /// ([`SrSwaLayerWeights::head_cls`]). `retire` turns block retirement on:
    /// pass [`retire_enabled`], and `false` whenever the page-count guard is
    /// off or attention runs dense ([`attend_dense`] reads every row).
    ///
    /// # Errors
    /// [`FocrError::Other`] unless `head_cls` has [`HEAD_DIM`] values.
    pub fn new(head_cls: &[f32], retire: bool) -> FocrResult<Self> {
        if head_cls.len() != HEAD_DIM {
            return Err(router_shape(
                "dash_router.head_cls",
                head_cls.len(),
                &HEAD_DIM.to_string(),
            ));
        }
        Ok(Self {
            head_cls: head_cls.to_vec(),
            rows: 0,
            slabs: Vec::new(),
            sums: BlockSummaries {
                kbar: Vec::new(),
                valid: Vec::new(),
            },
            bmin: Vec::new(),
            bmax: Vec::new(),
            pinned: Vec::new(),
            text_tail_margin: false,
            retire,
            dead: 0,
            peak_bytes: 0,
            last_step: None,
        })
    }

    /// Logical blocks, `ceil(rows / 32)`, retired ones included.
    #[must_use]
    pub fn nblk(&self) -> usize {
        self.slabs.len()
    }

    /// Retired blocks.
    #[must_use]
    pub fn dead_blocks(&self) -> usize {
        self.dead
    }

    /// Whether block `b` still holds its rows.
    #[must_use]
    pub fn is_live(&self, b: usize) -> bool {
        self.slabs.get(b).is_some_and(Option::is_some)
    }

    /// The block summaries of every logical block.
    #[must_use]
    pub fn summaries(&self) -> &BlockSummaries {
        &self.sums
    }

    /// Lowest image page per block, −1 when it holds none.
    #[must_use]
    pub fn bmin(&self) -> &[i32] {
        &self.bmin
    }

    /// Highest image page per block, −1 when it holds none.
    #[must_use]
    pub fn bmax(&self) -> &[i32] {
        &self.bmax
    }

    /// Bytes held now: the live blocks' slabs plus every logical block's
    /// bookkeeping.
    #[must_use]
    pub fn live_bytes(&self) -> usize {
        (self.nblk() - self.dead) * BLOCK_SIZE * ROW_BYTES + self.nblk() * BLOCK_META_BYTES
    }

    /// The most [`Self::live_bytes`] has been.
    #[must_use]
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }

    /// What one layer's cache holds for `rows` rows with nothing retired: whole
    /// blocks of [`ROW_BYTES`] rows plus the per-block bookkeeping (the
    /// estimator's input, U2-D3).
    #[must_use]
    pub fn estimate_bytes(rows: usize) -> usize {
        rows.div_ceil(BLOCK_SIZE) * (BLOCK_SIZE * ROW_BYTES + BLOCK_META_BYTES)
    }

    fn block_rows(&self, b: usize) -> usize {
        (self.rows - b * BLOCK_SIZE).min(BLOCK_SIZE)
    }

    fn live_slab(&self, b: usize) -> &Slab {
        self.slabs[b].as_ref().unwrap_or_else(|| {
            panic!("srswa cache: block {b} was retired and must not be read (OQ-V2-11)")
        })
    }

    /// Append `n` rows: K and V `[n, NUM_HEADS × HEAD_DIM]` (RoPE applied to
    /// K), index keys `[n, HEAD_DIM]` from [`project`], and each row's image
    /// page: `PageCoords::img_page` for prompt rows and −1 for every generated
    /// row, which never counts as an image token even when it is id 128815
    /// (MU:514-528). Prefill appends its rows at once, decode one. The summary
    /// of each block the rows reach is recomputed from all its index keys, so
    /// a full block's `k̄` is final and the tail's equals [`summaries`].
    ///
    /// # Errors
    /// [`FocrError::Other`] on a shape mismatch.
    pub fn append(&mut self, k: &Mat, v: &Mat, ik: &Mat, img_page: &[i32]) -> FocrResult<()> {
        let (n, width) = (k.rows, NUM_HEADS * HEAD_DIM);
        if k.shape() != (n, width)
            || v.shape() != (n, width)
            || ik.shape() != (n, HEAD_DIM)
            || img_page.len() != n
        {
            return Err(FocrError::Other(anyhow::anyhow!(
                "srswa cache append: K {:?}, V {:?}, index keys {:?}, {} image pages; want \
                 [n, {width}], [n, {width}], [n, {HEAD_DIM}] and n",
                k.shape(),
                v.shape(),
                ik.shape(),
                img_page.len()
            )));
        }
        let first = self.rows / BLOCK_SIZE;
        for (r, &page) in img_page.iter().enumerate() {
            let (b, o) = (self.rows / BLOCK_SIZE, self.rows % BLOCK_SIZE);
            if o == 0 {
                self.slabs.push(Some(Slab::new()));
                self.sums.kbar.extend([0.0f32; HEAD_DIM]);
                self.sums.valid.push(false);
                self.bmin.push(-1);
                self.bmax.push(-1);
                self.pinned.push(false);
            }
            // Retirement needs the block's last row ≤ q − 128 < rows, so the
            // block being filled is live.
            let slab = self.slabs[b]
                .as_mut()
                .expect("the block being appended to is live");
            for h in 0..NUM_HEADS {
                let (dst, src) = (h * HEAD_STRIDE + o * HEAD_DIM, h * HEAD_DIM);
                slab.k[dst..dst + HEAD_DIM].copy_from_slice(&k.row(r)[src..src + HEAD_DIM]);
                slab.v[dst..dst + HEAD_DIM].copy_from_slice(&v.row(r)[src..src + HEAD_DIM]);
            }
            slab.ik[o * HEAD_DIM..(o + 1) * HEAD_DIM].copy_from_slice(ik.row(r));
            if page >= 0 {
                self.bmin[b] = if self.bmin[b] < 0 {
                    page
                } else {
                    self.bmin[b].min(page)
                };
                self.bmax[b] = self.bmax[b].max(page);
            }
            self.rows += 1;
        }
        for b in first..self.nblk() {
            let n = self.block_rows(b);
            let slab = self.slabs[b].as_ref().expect("appended blocks are live");
            let out = &mut self.sums.kbar.as_chunks_mut::<HEAD_DIM>().0[b];
            block_summary(&slab.ik[..n * HEAD_DIM], &self.head_cls, out);
            self.sums.valid[b] = n > 0;
        }
        self.peak_bytes = self.peak_bytes.max(self.live_bytes());
        Ok(())
    }

    /// Route the decode query, the newest row at `rows − 1`, from its index
    /// query `iq` and query page `qp` (MU:481-528, MD:338-362). Retired blocks
    /// score as non-remote, so no route selects one. A non-open row never
    /// allows them anyway, so its route is the unretired one. An open row
    /// meeting retired blocks fires the detector unless `forced_step` (the
    /// guard forces this step's output, so attention cannot change it).
    ///
    /// # Errors
    /// [`RetireFallback`] when the detector fires.
    ///
    /// # Panics
    /// On an empty cache.
    pub fn decode_route(
        &self,
        iq: &[f32],
        qp: i32,
        p: &RouterParams,
        domain: EntmaxDomain,
        forced_step: bool,
    ) -> Result<DecodeRoute, RetireFallback> {
        assert!(
            self.rows > 0,
            "decode_route: the query row is not in the cache"
        );
        let q_pos = self.rows - 1;
        let mut sc = scores(iq, q_pos, &self.sums);
        for (s, slab) in sc.iter_mut().zip(&self.slabs) {
            if slab.is_none() {
                *s = SCORE_NEG;
            }
        }
        let open = decode_open_row(&self.bmin, &self.bmax, qp, &sc);
        if let Some(open) = open
            && self.dead > 0
            && !forced_step
        {
            return Err(RetireFallback {
                q_pos,
                query_page: qp,
                open,
                dead_blocks: self.dead,
            });
        }
        let allow = page_gate_decode(&self.bmin, &self.bmax, qp, self.nblk());
        Ok(DecodeRoute {
            route: route(&sc, Some(&allow), p, domain),
            open,
        })
    }

    /// Free every block dead at decode query position `q` on query page `qp`:
    /// unpinned, `bmin = −1` or `bmax < qp`, and last row ≤ q − 128 (≤ q − 129
    /// for an imageless block under the text-tail margin). Frees nothing while
    /// `qp < 0`, on a guard-forced step, or with retirement off. Returns the
    /// number of blocks freed. U2-B6 calls it once per decode step, after the
    /// step's row is appended; before or after its attention, as no freed
    /// block can be in that step's window or read set.
    ///
    /// # Errors
    /// [`FocrError::Other`] if `q` is past the cache or `q` or `qp` went
    /// backwards: the rule holds only because neither does.
    pub fn retire_dead_blocks(
        &mut self,
        qp: i32,
        q: usize,
        forced_step: bool,
    ) -> FocrResult<usize> {
        if q >= self.rows {
            return Err(FocrError::Other(anyhow::anyhow!(
                "srswa retirement: query position {q} past the {} cached rows",
                self.rows
            )));
        }
        if let Some((last_qp, last_q)) = self.last_step
            && (qp < last_qp || q < last_q)
        {
            return Err(FocrError::Other(anyhow::anyhow!(
                "srswa retirement: (query page, position) went from ({last_qp}, {last_q}) to \
                 ({qp}, {q})"
            )));
        }
        self.last_step = Some((qp, q));
        if !self.retire || forced_step || qp < 0 {
            return Ok(0);
        }
        let mut freed = 0;
        for c in 0..self.nblk() {
            let last = c * BLOCK_SIZE + BLOCK_SIZE - 1;
            if last + LOCAL_WINDOW > q {
                break;
            }
            let imageless = self.bmin[c] < 0;
            let margin = LOCAL_WINDOW + usize::from(imageless && self.text_tail_margin);
            if last + margin > q || self.pinned[c] || self.slabs[c].is_none() {
                continue;
            }
            if imageless || self.bmax[c] < qp {
                self.slabs[c] = None;
                self.dead += 1;
                freed += 1;
            }
        }
        Ok(freed)
    }

    /// Pin the blocks holding any row of `rows` so retirement never frees
    /// them, replacing the previous set. `text_tail_margin` retires an
    /// imageless block only once its last row ≤ q − 129, so a 128-row text
    /// tail ending just before the final row survives. Per window with a
    /// successor, U2-D1a pins the last own page's marker and image-tail blocks
    /// and the lookahead page's rows (plan §5.10.3, §5.10.9); unwindowed runs
    /// pin nothing.
    ///
    /// # Errors
    /// [`FocrError::Other`] if a range reaches past the cache or a block it
    /// touches is already retired.
    pub fn set_pinned(&mut self, rows: &[Range<usize>], text_tail_margin: bool) -> FocrResult<()> {
        let mut pinned = vec![false; self.nblk()];
        for r in rows.iter().filter(|r| !r.is_empty()) {
            if r.end > self.rows {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "srswa pin: rows {r:?} past the {} cached rows",
                    self.rows
                )));
            }
            let blocks = r.start / BLOCK_SIZE..r.end.div_ceil(BLOCK_SIZE);
            if let Some(b) = blocks.clone().find(|&b| !self.is_live(b)) {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "srswa pin: block {b} of rows {r:?} is already retired"
                )));
            }
            pinned[blocks].fill(true);
        }
        self.pinned = pinned;
        self.text_tail_margin = text_tail_margin;
        Ok(())
    }

    /// The rows of `ranges`, in order, in the tail format: K, V and index keys
    /// only. Page state is replayed in the new frame (U2-B1's reused-window
    /// seed), never copied (W/`session/uocr_doc_reuse.py:500-516`), and `k̄` is
    /// recomputed when the rows are appended.
    ///
    /// # Errors
    /// [`FocrError::Other`] if a row is past the cache or retired.
    pub fn serialize_tail(&self, ranges: &[Range<usize>]) -> FocrResult<Vec<u8>> {
        let n: usize = ranges.iter().map(ExactSizeIterator::len).sum();
        let mut out = Vec::with_capacity(TAIL_HEADER + n * ROW_BYTES);
        out.extend_from_slice(TAIL_MAGIC);
        for word in [TAIL_VERSION, NUM_HEADS as u32, HEAD_DIM as u32] {
            out.extend_from_slice(&word.to_le_bytes());
        }
        out.extend_from_slice(&(n as u64).to_le_bytes());
        let mut put = |xs: &[f32]| {
            for x in xs {
                out.extend_from_slice(&x.to_le_bytes());
            }
        };
        for pos in ranges.iter().flat_map(Clone::clone) {
            let (b, o) = (pos / BLOCK_SIZE, pos % BLOCK_SIZE);
            if pos >= self.rows || !self.is_live(b) {
                return Err(FocrError::Other(anyhow::anyhow!(
                    "srswa tail: row {pos} is {} (block {b})",
                    if pos >= self.rows {
                        "past the cache"
                    } else {
                        "retired"
                    }
                )));
            }
            let slab = self.live_slab(b);
            for buf in [&slab.k, &slab.v] {
                for h in 0..NUM_HEADS {
                    let at = h * HEAD_STRIDE + o * HEAD_DIM;
                    put(&buf[at..at + HEAD_DIM]);
                }
            }
            put(&slab.ik[o * HEAD_DIM..(o + 1) * HEAD_DIM]);
        }
        Ok(out)
    }
}

/// Read rows written by [`SrSwaLayerCache::serialize_tail`].
///
/// # Errors
/// [`FocrError::Other`] on a bad magic, version, geometry or length.
pub fn deserialize_tail(bytes: &[u8]) -> FocrResult<TailRows> {
    let bad = |why: String| FocrError::Other(anyhow::anyhow!("srswa tail: {why}"));
    if bytes.len() < TAIL_HEADER || &bytes[..8] != TAIL_MAGIC {
        return Err(bad("not a tail (bad magic or short header)".into()));
    }
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes"));
    let (version, heads, dim) = (word(8), word(12), word(16));
    if version != TAIL_VERSION || heads as usize != NUM_HEADS || dim as usize != HEAD_DIM {
        return Err(bad(format!(
            "version {version} with {heads} heads of {dim}; want version {TAIL_VERSION} with \
             {NUM_HEADS} heads of {HEAD_DIM}"
        )));
    }
    let n = u64::from_le_bytes(bytes[20..28].try_into().expect("8 bytes"));
    let want = usize::try_from(n)
        .ok()
        .and_then(|n| n.checked_mul(ROW_BYTES))
        .and_then(|b| b.checked_add(TAIL_HEADER));
    if want != Some(bytes.len()) {
        return Err(bad(format!(
            "{n} rows need {want:?} bytes, got {}",
            bytes.len()
        )));
    }
    let n = n as usize;
    let width = NUM_HEADS * HEAD_DIM;
    let mut floats = bytes[TAIL_HEADER..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b));
    let mut take = |len: usize| -> Vec<f32> { floats.by_ref().take(len).collect() };
    let (mut k, mut v, mut ik) = (
        Vec::with_capacity(n * width),
        Vec::with_capacity(n * width),
        Vec::with_capacity(n * HEAD_DIM),
    );
    for _ in 0..n {
        k.extend(take(width));
        v.extend(take(width));
        ik.extend(take(HEAD_DIM));
    }
    Ok(TailRows {
        k: Mat::from_vec(n, width, k),
        v: Mat::from_vec(n, width, v),
        ik: Mat::from_vec(n, HEAD_DIM, ik),
    })
}

impl BlockKv for SrSwaLayerCache {
    fn rows(&self) -> usize {
        self.rows
    }

    fn keys(&self, h: usize, b: usize) -> &[f32] {
        let n = self.block_rows(b);
        &self.live_slab(b).k[h * HEAD_STRIDE..h * HEAD_STRIDE + n * HEAD_DIM]
    }

    fn values(&self, h: usize, b: usize) -> &[f32] {
        let n = self.block_rows(b);
        &self.live_slab(b).v[h * HEAD_STRIDE..h * HEAD_STRIDE + n * HEAD_DIM]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMG: u32 = special::IMAGE;
    const PG: u32 = PAGE_TOKEN_ID;
    const BOS: u32 = special::BOS;

    /// `BOS (<PAGE> <image>×n)… instruction` with the given per-page image counts.
    fn prompt(images: &[usize], instruction: usize) -> Vec<u32> {
        let mut ids = vec![BOS];
        for &n in images {
            ids.push(PG);
            ids.extend(std::iter::repeat_n(IMG, n));
        }
        ids.extend((0..instruction).map(|k| 1000 + k as u32));
        ids
    }

    /// Small deterministic generator for the property tests.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    #[test]
    fn two_page_base_prompt_hand_computed() {
        // BOS + 2 × (<PAGE> + 273 images) + 4 instruction tokens = 553 rows.
        let ids = prompt(&[273, 273], 4);
        assert_eq!(ids.len(), 553);
        let c = PageCoords::build(&ids).unwrap();
        assert_eq!((c.prompt_pages, c.markers, c.last_img), (2, 2, 548));
        assert_eq!(c.pf_cnt, vec![1, 274, 274]);
        // BOS: bucket 0, first row of it, no page anywhere.
        assert_eq!(
            (c.bucket[0], c.tok_pos[0], c.query_page[0], c.img_page[0]),
            (0, 0, -1, -1)
        );
        // Page 1: marker at 1, images at 2..=274.
        assert_eq!(
            (c.bucket[1], c.tok_pos[1], c.query_page[1], c.img_page[1]),
            (1, 0, 0, -1)
        );
        for i in 2..=274 {
            assert_eq!(
                (c.bucket[i], c.tok_pos[i], c.query_page[i], c.img_page[i]),
                (1, i as u32 - 1, 0, 0)
            );
        }
        // Page 2: marker at 275, images at 276..=548.
        assert_eq!(
            (c.bucket[275], c.tok_pos[275], c.query_page[275]),
            (2, 0, 1)
        );
        for i in 276..=548 {
            assert_eq!(
                (c.bucket[i], c.tok_pos[i], c.query_page[i], c.img_page[i]),
                (2, i as u32 - 275, 1, 1)
            );
        }
        // Instruction: back in bucket 0 after BOS, query page −1.
        for (k, i) in (549..553).enumerate() {
            assert_eq!(
                (c.bucket[i], c.tok_pos[i], c.query_page[i], c.img_page[i]),
                (0, k as u32 + 1, -1, -1)
            );
        }
        // Rope page parity, unwindowed: bucket 1 → 1, bucket 2 → 0, bucket 0 → 0.
        assert_eq!(c.rope_pages(0)[1..3], [1, 1]);
        assert_eq!(c.rope_pages(0)[275], 0);
        assert_eq!(c.rope_pages(0)[0], 0);

        // Decode: the first generated <PAGE> answers page 1 (qp 0), right after
        // its 274 prompt rows; then text; the second answers page 2.
        let mut st = DecodePageState::after_prompt(&c, 0);
        assert_eq!(
            st.feed(PG),
            DecodeCoord {
                bucket: 1,
                tok_pos: 274,
                rope_page: 1,
                query_page: 0
            }
        );
        assert_eq!(
            st.feed(7),
            DecodeCoord {
                bucket: 1,
                tok_pos: 275,
                rope_page: 1,
                query_page: 0
            }
        );
        assert_eq!(
            st.feed(PG),
            DecodeCoord {
                bucket: 2,
                tok_pos: 274,
                rope_page: 0,
                query_page: 1
            }
        );
    }

    #[test]
    fn prompts_without_pages_are_rejected_not_rotated_in_1d() {
        let no_marker = [BOS, IMG, IMG, 5];
        let err = PageCoords::build(&no_marker).unwrap_err().to_string();
        assert!(err.contains("OQ-V2-6"), "{err}");
        let no_image = [BOS, PG, 5];
        assert!(PageCoords::build(&no_image).is_err());
        // A marker only after the last image is still "no marker before it".
        assert!(PageCoords::build(&[BOS, IMG, PG, 5]).is_err());
    }

    /// A prompt plus a teacher-forced response of at most `N` page answers. The
    /// prompt ends with an instruction, as franken_ocr's always do: that is what
    /// puts the HF decode seed in bucket 0.
    fn random_case(rng: &mut Lcg) -> (Vec<u32>, usize) {
        let n_pages = 1 + rng.below(4) as usize;
        let images: Vec<usize> = (0..n_pages).map(|_| 1 + rng.below(9) as usize).collect();
        let ids = prompt(&images, 1 + rng.below(4) as usize);
        let prompt_len = ids.len();
        let mut ids = ids;
        let answers = rng.below(n_pages as u64 + 1) as usize;
        // Optional text before the first marker, then <PAGE> body per answer.
        ids.extend((0..rng.below(3)).map(|k| 2000 + k as u32));
        for _ in 0..answers {
            ids.push(PG);
            ids.extend((0..rng.below(6)).map(|k| 3000 + k as u32));
        }
        (ids, prompt_len)
    }

    #[test]
    fn decode_state_machine_matches_the_prefill_builder() {
        let mut rng = Lcg(0x5eed_0001);
        for case in 0..500 {
            let (ids, prompt_len) = random_case(&mut rng);
            let full = PageCoords::build(&ids).unwrap();
            let prompt = PageCoords::build(&ids[..prompt_len]).unwrap();
            let mut st = DecodePageState::after_prompt(&prompt, 0);
            for (i, &t) in ids.iter().enumerate().skip(prompt_len) {
                let d = st.feed(t);
                let want = (full.bucket[i], full.tok_pos[i], full.query_page[i]);
                assert_eq!(
                    (d.bucket, d.tok_pos, d.query_page),
                    want,
                    "case {case} row {i}: {ids:?}"
                );
                assert_eq!(d.rope_page, rope_page(full.bucket[i], 0));
            }
        }
    }

    #[test]
    fn coordinates_are_unique_and_parity_follows_the_global_page() {
        let mut rng = Lcg(0x5eed_0002);
        for _ in 0..500 {
            let (ids, _) = random_case(&mut rng);
            let c = PageCoords::build(&ids).unwrap();
            let mut seen = std::collections::HashSet::new();
            for (&b, &t) in c.bucket.iter().zip(&c.tok_pos) {
                assert!(
                    seen.insert((b, t)),
                    "duplicate (bucket, tok_pos) ({b}, {t}) in {ids:?}"
                );
            }
        }
        // Window layouts: a recomputed window starting at global page `lo`
        // (0-based) numbers it local bucket 1, page_base = lo; a reused window's
        // bucket 1 is the tail page lo − 1, page_base = lo − 1. Either way the
        // rope parity of a paged row is that of its global page + 1.
        for _ in 0..1000 {
            let lo = 1 + rng.below(500) as u32;
            let local = 1 + rng.below(12) as u32;
            let recomputed_global = lo + local - 1;
            assert_eq!(rope_page(local, lo), (recomputed_global + 1) % 2);
            let reused_global = lo - 1 + local - 1;
            assert_eq!(rope_page(local, lo - 1), (reused_global + 1) % 2);
            assert_eq!(rope_page(0, lo), 0, "bucket 0 never takes the base");
        }
    }

    #[test]
    fn an_extra_generated_marker_clamps_to_the_last_page() {
        // N = 2; three generated markers. The third clamps to bucket 2 and
        // restarts at pf_cnt[2], duplicating page 2's response-<PAGE>
        // coordinate (DISC-U2-7); the unclamped prefill builder disagrees there.
        let mut ids = prompt(&[3, 5], 2);
        let prompt_len = ids.len();
        ids.extend([PG, 9, PG, 9, PG]);
        let p = PageCoords::build(&ids[..prompt_len]).unwrap();
        let mut st = DecodePageState::after_prompt(&p, 0);
        let coords: Vec<DecodeCoord> = ids[prompt_len..].iter().map(|&t| st.feed(t)).collect();
        assert_eq!((coords[2].bucket, coords[2].tok_pos), (2, 6));
        assert_eq!((coords[4].bucket, coords[4].tok_pos), (2, p.pf_cnt[2]));
        assert_eq!(coords[4].tok_pos, coords[2].tok_pos);
        assert_eq!(
            coords[4].query_page, 2,
            "the query page itself is not clamped (MU:517)"
        );
        let full = PageCoords::build(&ids).unwrap();
        assert_eq!(full.bucket[ids.len() - 1], 3);
    }

    #[test]
    fn reused_window_seed_lands_the_forced_marker_on_page_lo() {
        // Rewritten prompt: tail page lo−1 (4 images) + new pages lo (6) and lo+1 (2).
        let rewritten: Vec<u32> = [4usize, 6, 2]
            .iter()
            .flat_map(|&n| std::iter::once(PG).chain(std::iter::repeat_n(IMG, n)))
            .collect();
        let c = PageCoords::build(&rewritten).unwrap();
        let lo = 7;
        let mut st = DecodePageState::reused_window(&c, lo - 1).unwrap();
        let forced = st.feed(PG);
        assert_eq!(forced.bucket, 2);
        assert_eq!(forced.tok_pos, 1 + 6, "tok_pos = 1 + n_img(lo)");
        assert_eq!(forced.query_page, 1);
        assert_eq!(
            forced.rope_page,
            (lo + 1) % 2,
            "global page lo, parity lo + 1"
        );
        // The HF seed would have clamped the forced marker into bucket M = 3.
        let mut hf = DecodePageState::after_prompt(&c, lo - 1);
        assert_eq!(hf.feed(PG).bucket, 3);
        // A prompt that is not pages-only, or has a single page, is refused.
        assert!(
            DecodePageState::reused_window(&PageCoords::build(&prompt(&[4, 6], 1)).unwrap(), 0)
                .is_err()
        );
        let one_page: Vec<u32> = std::iter::once(PG)
            .chain(std::iter::repeat_n(IMG, 3))
            .collect();
        assert!(DecodePageState::reused_window(&PageCoords::build(&one_page).unwrap(), 0).is_err());
    }

    #[test]
    fn inverse_frequencies_follow_the_resample_rule() {
        // Against an f64 evaluation of 10000^(-i/63): within f32 rounding of the
        // resample rule, and far from v1's 10000^(-2i/128) for every i > 0. The
        // bit-exact check against the oracle's dumped table is U2-C1's L1.
        let inv = inv_tok_freqs();
        assert_eq!(inv[0], 1.0);
        for (i, &f) in inv.iter().enumerate() {
            let resample = 10_000f64.powf(-(i as f64) / 63.0);
            let rel = (f64::from(f) - resample).abs() / resample;
            assert!(
                rel < 4.0 * f64::from(f32::EPSILON),
                "inv_tok[{i}] = {f}, want {resample}"
            );
            if i > 0 {
                let v1 = 10_000f64.powf(-2.0 * i as f64 / 128.0);
                assert!(
                    (f64::from(f) - v1).abs() / v1 > 1e-3,
                    "inv_tok[{i}] matches v1's rule"
                );
            }
        }
        assert!(inv.windows(2).all(|w| w[1] < w[0]));
    }

    #[test]
    fn page2d_rotation_matches_direct_cos_sin() {
        let tok = [0u32, 1, 273, 4096];
        let page = [0u32, 1, 0, 1];
        let table = page2d_rope_table(&tok, &page);
        let inv = inv_tok_freqs();
        let heads = 2;
        let mut rng = Lcg(0x5eed_0003);
        let data: Vec<f32> = (0..tok.len() * heads * HEAD_DIM)
            .map(|_| rng.below(2001) as f32 / 1000.0 - 1.0)
            .collect();
        let mut x = Mat::from_vec(tok.len(), heads * HEAD_DIM, data.clone());
        apply_rope(&mut x, &table).unwrap();
        let half = HEAD_DIM / 2;
        for r in 0..tok.len() {
            for h in 0..heads {
                let base = r * heads * HEAD_DIM + h * HEAD_DIM;
                for i in 0..half {
                    let ang = if i < N_TOK_FREQS {
                        tok[r] as f32 * inv[i]
                    } else {
                        page[r] as f32
                    };
                    let (s, c) = ang.sin_cos();
                    let (a, b) = (data[base + i], data[base + half + i]);
                    let got = (
                        x.row(r)[h * HEAD_DIM + i],
                        x.row(r)[h * HEAD_DIM + half + i],
                    );
                    assert_eq!(got.0.to_bits(), (a * c - b * s).to_bits());
                    assert_eq!(got.1.to_bits(), (b * c + a * s).to_bits());
                }
            }
        }
        // Row 0 (bucket 0, tok 0) is the identity rotation.
        assert_eq!(&x.row(0)[..heads * HEAD_DIM], &data[..heads * HEAD_DIM]);
    }

    // ── Router (U2-B2) ──────────────────────────────────────────────────────

    fn uniform(rng: &mut Lcg, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * (rng.below(1_000_001) as f32 / 1_000_000.0)
    }

    fn random_mat(rng: &mut Lcg, rows: usize, cols: usize, scale: f32) -> Mat {
        Mat::from_vec(
            rows,
            cols,
            (0..rows * cols)
                .map(|_| uniform(rng, -scale, scale))
                .collect(),
        )
    }

    /// The full HF window and selection, the reference every variant meets.
    fn full_topm(x: &[f32], cand: &[bool], p: &RouterParams) -> Route {
        select_bias(&entmax15_topm(x, p.window(x.len())), cand, p)
    }

    #[test]
    fn router_constants_follow_the_checkpoint_snapshot() {
        let p = RouterParams::uocr2();
        assert_eq!(p.idx_scale, (128.0f64 / 128f64.sqrt()) as f32);
        assert!((p.idx_scale - 11.313_708).abs() < 1e-5);
        assert_eq!(p.floor, 1e-4f64 as f32);
        assert_eq!(p.inv_sigma, 1.0);
        assert_eq!((p.budget, p.topm), (12, 16));
        assert_eq!(inv_sqrt_d(), (1.0f64 / 128f64.sqrt()) as f32);
        assert_eq!(
            [p.window(1), p.window(9), p.window(16), p.window(1000)],
            [1, 9, 16, 16]
        );
        assert_eq!(
            [p.read_slots(1), p.read_slots(9), p.read_slots(16)],
            [1, 9, 12]
        );
        let bad = SrSwaLayerWeights::new(8, vec![0.0; 3], vec![], vec![], vec![], vec![]);
        assert!(bad.unwrap_err().to_string().contains("q_proj"));
    }

    #[test]
    fn gemma_norm_matches_the_formula_with_clamps_and_non_finite_inputs() {
        let mut rng = Lcg(0x5eed_0b02);
        let w: Vec<f32> = (0..HEAD_DIM)
            .map(|_| uniform(&mut rng, -0.5, 0.5))
            .collect();
        let x: Vec<f32> = (0..HEAD_DIM)
            .map(|_| uniform(&mut rng, -3.0, 3.0))
            .collect();
        let mut y = x.clone();
        gemma_rms_norm(&mut y, &w);
        let ms = x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / HEAD_DIM as f64;
        for i in 0..HEAD_DIM {
            let want = f64::from(x[i]) / (ms + 1e-6).sqrt() * (1.0 + f64::from(w[i]));
            assert!(
                (f64::from(y[i]) - want).abs() < 1e-5,
                "{i}: {} vs {want}",
                y[i]
            );
        }
        // NaN and ±inf count as 0, in the mean as well.
        let (mut bad, mut zeroed) = (x.clone(), x.clone());
        for (i, v) in [(3, f32::NAN), (7, f32::INFINITY), (9, f32::NEG_INFINITY)] {
            bad[i] = v;
            zeroed[i] = 0.0;
        }
        gemma_rms_norm(&mut bad, &w);
        gemma_rms_norm(&mut zeroed, &w);
        assert_eq!(bad, zeroed);
        // (1 + w) can push past ±1e4, and the output clamps there.
        let mut big = vec![1.0f32; HEAD_DIM];
        big[0] = -1.0;
        gemma_rms_norm(&mut big, &[2.0e4; HEAD_DIM]);
        assert_eq!((big[0], big[1]), (-1.0e4, 1.0e4));
        // A zero row stays zero: eps keeps the rsqrt finite.
        let mut zero = vec![0.0f32; HEAD_DIM];
        gemma_rms_norm(&mut zero, &w);
        assert!(zero.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn projections_compose_linear_norm_and_page2d_rope() {
        let (hidden, rows) = (48, 3);
        let mut rng = Lcg(0x5eed_0b03);
        let mut v = |n: usize, s: f32| (0..n).map(|_| uniform(&mut rng, -s, s)).collect();
        let w = SrSwaLayerWeights::new(
            hidden,
            v(HEAD_DIM * hidden, 0.2),
            v(HEAD_DIM * hidden, 0.2),
            v(HEAD_DIM, 0.3),
            v(HEAD_DIM, 0.3),
            v(HEAD_DIM, 0.5),
        )
        .unwrap();
        let mut h: Vec<f32> = v(rows * hidden, 1.0);
        h[5] = f32::NAN;
        h[hidden + 2] = f32::INFINITY;
        let rope = page2d_rope_table(&[0, 5, 300], &[0, 1, 0]);
        let (iq, ik) = project(&Mat::from_vec(rows, hidden, h.clone()), &w, &rope).unwrap();
        for (proj, norm, got) in [(&w.q_proj, &w.q_norm, &iq), (&w.k_proj, &w.k_norm, &ik)] {
            let mut want = Mat::zeros(rows, HEAD_DIM);
            for r in 0..rows {
                let x: Vec<f32> = h[r * hidden..(r + 1) * hidden]
                    .iter()
                    .map(|&v| finite_or_zero(v))
                    .collect();
                for o in 0..HEAD_DIM {
                    want.row_mut(r)[o] = dot(&proj[o * hidden..(o + 1) * hidden], &x);
                }
                gemma_rms_norm(want.row_mut(r), norm);
            }
            apply_rope(&mut want, &rope).unwrap();
            assert_eq!(got.shape(), (rows, HEAD_DIM));
            for (a, b) in got.data.iter().zip(&want.data) {
                assert!((a - b).abs() < 1e-4, "{a} vs {b}");
            }
        }
        assert!(project(&Mat::zeros(rows, hidden + 1), &w, &rope).is_err());
    }

    #[test]
    fn summaries_and_scores_follow_the_reference_formulas() {
        let mut rng = Lcg(0x5eed_0b04);
        let rows = 9 * BLOCK_SIZE + 5; // 10 blocks, the last one partial
        let ik = random_mat(&mut rng, rows, HEAD_DIM, 1.0);
        // A zero head_cls is a mean pool.
        let mean = summaries(&ik, &[0.0; HEAD_DIM]);
        assert_eq!(mean.nblk(), 10);
        assert!(mean.valid.iter().all(|&v| v));
        // A learned head_cls weights rows by softmax(ik·hc / √128).
        let hc: Vec<f32> = (0..HEAD_DIM)
            .map(|_| uniform(&mut rng, -2.0, 2.0))
            .collect();
        let sums = summaries(&ik, &hc);
        for c in 0..10 {
            let block = c * BLOCK_SIZE..((c + 1) * BLOCK_SIZE).min(rows);
            let n = block.len() as f64;
            let a: Vec<f64> = block
                .clone()
                .map(|j| f64::from(dot(ik.row(j), &hc)) / 128f64.sqrt())
                .collect();
            let max = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = a.iter().map(|&v| (v - max).exp()).collect();
            let z: f64 = e.iter().sum();
            for d in 0..HEAD_DIM {
                let pooled = block.clone().map(|j| f64::from(ik.get(j, d))).sum::<f64>() / n;
                assert!((f64::from(mean.kbar(c)[d]) - pooled).abs() < 1e-5);
                let soft: f64 = block
                    .clone()
                    .zip(&e)
                    .map(|(j, &ej)| ej / z * f64::from(ik.get(j, d)))
                    .sum();
                assert!((f64::from(sums.kbar(c)[d]) - soft).abs() < 1e-5);
            }
        }
        // The last row's query (block 9) reads blocks 0..=5 remotely.
        let iq: Vec<f32> = (0..HEAD_DIM)
            .map(|_| uniform(&mut rng, -1.0, 1.0))
            .collect();
        let sc = scores(&iq, rows - 1, &sums);
        for (c, &s) in sc.iter().enumerate() {
            if c <= 5 {
                let want = f64::from(dot(&iq, sums.kbar(c))) / 128f64.sqrt();
                assert!((f64::from(s) - want).abs() < 1e-5, "block {c}");
            } else {
                assert_eq!(s, SCORE_NEG, "block {c} is local or future");
            }
        }
        // Nothing is remote before row 128; an invalid block never is; big
        // products clamp at ±1e4.
        assert!(scores(&iq, 127, &sums).iter().all(|&s| s == SCORE_NEG));
        let mut holed = sums.clone();
        holed.valid[2] = false;
        assert_eq!(scores(&iq, rows - 1, &holed)[2], SCORE_NEG);
        let huge: Vec<f32> = sums.kbar(0).iter().map(|&k| k * 1e6).collect();
        assert_eq!(scores(&huge, rows - 1, &sums)[0], 1.0e4);
    }

    /// 1.5-entmax in f64 by bisection on `Σ max(z − τ, 0)² = 1`.
    fn entmax15_reference(x: &[f32]) -> Vec<f64> {
        let max = f64::from(x.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let z: Vec<f64> = x.iter().map(|&v| (f64::from(v) - max) / 2.0).collect();
        let (mut lo, mut hi) = (-1.0f64, 0.0f64);
        for _ in 0..200 {
            let mid = (lo + hi) / 2.0;
            let s: f64 = z.iter().map(|&v| (v - mid).max(0.0).powi(2)).sum();
            if s > 1.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let tau = (lo + hi) / 2.0;
        z.iter().map(|&v| (v - tau).max(0.0).powi(2)).collect()
    }

    #[test]
    fn entmax15_is_a_sparse_distribution_matching_a_brute_force_solve() {
        let mut rng = Lcg(0x5eed_0b05);
        for case in 0..2000 {
            let n = 1 + rng.below(40) as usize;
            let spread = [0.5f32, 5.0, 50.0, 2.0e4][rng.below(4) as usize];
            let x: Vec<f32> = (0..n).map(|_| uniform(&mut rng, -spread, spread)).collect();
            let (p, support) = entmax15(&x);
            assert!((1..=n).contains(&support), "case {case}");
            let sum: f64 = p.iter().map(|&v| f64::from(v)).sum();
            assert!((sum - 1.0).abs() < 1e-4, "case {case}: sum {sum}");
            for (a, b) in p.iter().zip(entmax15_reference(&x)) {
                assert!((f64::from(*a) - b).abs() < 1e-4, "case {case}: {a} vs {b}");
            }
            // Raising one input never lowers its probability.
            let i = rng.below(n as u64) as usize;
            let mut up = x.clone();
            up[i] += uniform(&mut rng, 0.0, spread);
            assert!(entmax15(&up).0[i] + 1e-5 >= p[i], "case {case}: monotone");
            // Shift invariance (shifts small enough that x + s rounds exactly
            // enough to compare).
            if spread <= 50.0 {
                let s = rng.below(129) as f32 - 64.0;
                let shifted: Vec<f32> = x.iter().map(|&v| v + s).collect();
                for (a, b) in p.iter().zip(entmax15(&shifted).0) {
                    assert!((a - b).abs() < 1e-4, "case {case}: shift {s}");
                }
            }
        }
        assert_eq!(entmax15(&[3.0]), (vec![1.0], 1));
        let (p, support) = entmax15(&[ENTMAX_NEG; 16]);
        assert_eq!(support, 16);
        assert!(p.iter().all(|&v| v == 1.0 / 16.0));
    }

    #[test]
    fn top_m_slots_break_ties_by_lower_block() {
        let n = ENTMAX_NEG;
        let top = entmax15_topm(&[n, 5.0, n, 5.0, 7.0, n, n], 5);
        let order: Vec<u32> = top.slots.iter().map(|s| s.1).collect();
        assert_eq!(order, [4, 1, 3, 0, 2]);
        // The partial selection agrees with a full sort, duplicates included.
        let mut rng = Lcg(0x5eed_0b06);
        for _ in 0..300 {
            let nblk = 1 + rng.below(300) as usize;
            let x: Vec<f32> = (0..nblk).map(|_| rng.below(7) as f32 - 3.0).collect();
            let mut want: Vec<(f32, u32)> = x.iter().zip(0u32..).map(|(&v, c)| (v, c)).collect();
            want.sort_by(slot_order);
            want.truncate(16);
            assert_eq!(entmax15_topm(&x, 16).slots, want);
        }
    }

    #[test]
    fn page_gates_follow_the_reference_cases() {
        // Decode: blocks 0-1 hold page 0, block 2 straddles pages 0|1, block 3
        // page 1, block 4 page 2, block 5 is prompt text; blocks 6-7 are
        // generated text past the prompt's blocks.
        let (bmin, bmax) = ([0, 0, 0, 1, 2, -1], [0, 0, 1, 1, 2, -1]);
        let f = false;
        assert_eq!(
            page_gate_decode(&bmin, &bmax, 0, 8),
            [true, true, true, f, f, f, f, f]
        );
        assert_eq!(
            page_gate_decode(&bmin, &bmax, 1, 8),
            [f, f, true, true, f, f, f, f]
        );
        // Open rows read everything, generated text included: qp < 0, and a
        // page no block holds (an answer past the prompt's pages).
        assert_eq!(page_gate_decode(&bmin, &bmax, -1, 8), vec![true; 8]);
        assert_eq!(page_gate_decode(&bmin, &bmax, 3, 8), vec![true; 8]);

        // Prefill tests membership, not the [bmin, bmax] range: block 1 holds
        // pages 0 and 2 only (a contrived layout) and stays closed to page 1.
        let mut img_page = vec![0i32; BLOCK_SIZE];
        img_page.extend((0..BLOCK_SIZE).map(|j| if j < 16 { 0 } else { 2 }));
        img_page.extend(std::iter::repeat_n(1, BLOCK_SIZE));
        img_page.extend(std::iter::repeat_n(-1, 7));
        let blocks = BlockPages::build(&img_page);
        assert_eq!(
            (blocks.bmin.clone(), blocks.bmax.clone()),
            (vec![0, 0, 1, -1], vec![0, 2, 1, -1])
        );
        assert_eq!(page_gate_prefill(&blocks, 1), [f, f, true, f]);
        assert_eq!(page_gate_prefill(&blocks, 2), [f, true, f, f]);
        assert_eq!(page_gate_prefill(&blocks, -1), [true; 4]);
        assert_eq!(page_gate_prefill(&blocks, 7), [true; 4]);

        // Allowed blocks that are not remote fall back to every remote block.
        let sc = [3.0, SCORE_NEG, 1.0, SCORE_NEG];
        assert_eq!(
            candidates(&sc, Some(&[f, true, f, true])),
            [true, f, true, f]
        );
        assert_eq!(candidates(&sc, Some(&[f, f, true, f])), [f, f, true, f]);
        // No remote block: no candidate, and the route reads nothing remote.
        let none = route(
            &[SCORE_NEG; 4],
            None,
            &RouterParams::uocr2(),
            EntmaxDomain::TopM,
        );
        assert!(none.blocks.is_empty() && none.bias.is_empty());
    }

    /// Random entmax inputs over `nblk` blocks in the regimes the padding
    /// argument has to cover. Non-candidates sit at −1e4.
    fn random_inputs(rng: &mut Lcg, nblk: usize, max_cands: usize) -> (Vec<f32>, Vec<bool>) {
        let mut cand = vec![false; nblk];
        let n_cand = 1 + rng.below(max_cands.min(nblk) as u64) as usize;
        for _ in 0..n_cand {
            cand[rng.below(nblk as u64) as usize] = true;
        }
        let regime = rng.below(5);
        let x = cand
            .iter()
            .map(|&c| {
                if !c {
                    return ENTMAX_NEG;
                }
                match regime {
                    0 => uniform(rng, -50.0, 50.0),
                    1 => uniform(rng, -1.13e5, 1.13e5),
                    // Every candidate below −1e4 + 2: pads may enter the support.
                    2 => uniform(rng, -1.2e4, -1.0e4 + 2.0),
                    // Ties with the pads at exactly −1e4.
                    3 if rng.below(2) == 0 => ENTMAX_NEG,
                    _ => rng.below(5) as f32 - 1.0e4 + 1.0,
                }
            })
            .collect();
        (x, cand)
    }

    #[test]
    fn candidate_only_top_m_with_merged_pads_equals_the_full_window() {
        let mut rng = Lcg(0x5eed_0b07);
        let p = RouterParams::uocr2();
        let mut pads_in_support = 0;
        for case in 0..3000 {
            let nblk = 1 + rng.below(64) as usize;
            let (x, cand) = random_inputs(&mut rng, nblk, 24);
            let full = full_topm(&x, &cand, &p);
            let cands: Vec<(u32, f32)> = (0u32..)
                .zip(&x)
                .zip(&cand)
                .filter(|&(_, &c)| c)
                .map(|((b, &v), _)| (b, v))
                .collect();
            assert_eq!(
                route_topm_candidates(&cands, nblk, &p),
                full,
                "case {case}: {x:?}"
            );
            let top = entmax15_topm(&x, p.window(nblk));
            pads_in_support += usize::from(
                top.slots
                    .iter()
                    .zip(&top.prior)
                    .any(|(&(_, b), &pr)| !cand[b as usize] && pr > 0.0),
            );
        }
        assert!(
            pads_in_support > 0,
            "the pads-in-support regime was exercised"
        );
    }

    #[test]
    fn all_candidates_equals_top_m_under_its_preconditions() {
        // ≤ 16 candidates, distinct inputs (no tie at the 12th place), the best
        // at least −1e4 + 2 and none at or below −1e4.
        let mut rng = Lcg(0x5eed_0b08);
        let p = RouterParams::uocr2();
        for case in 0..2000 {
            let nblk = 1 + rng.below(80) as usize;
            let mut cand = vec![false; nblk];
            let n_cand = 1 + rng.below(16.min(nblk) as u64) as usize;
            for _ in 0..n_cand {
                cand[rng.below(nblk as u64) as usize] = true;
            }
            let spread = [1.0f32, 30.0, 9000.0][rng.below(3) as usize];
            let x: Vec<f32> = cand
                .iter()
                .zip(0u32..)
                .map(|(&c, b)| {
                    if c {
                        uniform(&mut rng, -spread, spread) + b as f32 * 1e-3
                    } else {
                        ENTMAX_NEG
                    }
                })
                .collect();
            let (all, top) = (entmax15_all(&x, &cand, &p), full_topm(&x, &cand, &p));
            assert_eq!(all.blocks, top.blocks, "case {case}");
            for (a, b) in all.bias.iter().zip(&top.bias) {
                assert!((a - b).abs() < 1e-5, "case {case}: {a} vs {b}");
            }
        }
        // Past 16 candidates the two differ by design once the support is
        // wider than the window: with 20 near-equal candidates AllCandidates
        // spreads the entmax over all 20 and TopM over its 16, so the biases
        // differ, while both still read the top 12.
        let x: Vec<f32> = (0..20).map(|i| 0.01 * i as f32).collect();
        let cand = vec![true; 20];
        let all = entmax15_all(&x, &cand, &p);
        assert_eq!(all.blocks, (8..20).collect::<Vec<u32>>());
        let top = full_topm(&x, &cand, &p);
        assert_eq!(top.blocks, all.blocks);
        assert_eq!((all.support, top.support), (20, 16));
        assert!(
            all.bias
                .iter()
                .zip(&top.bias)
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "{:?} vs {:?}",
            all.bias,
            top.bias
        );
    }

    #[test]
    fn zero_prior_candidates_in_the_read_slots_are_still_read() {
        // One dominant candidate takes all the mass; the other five have prior
        // 0 but sit in the first K' slots, so they are read at ln(floor) − μ.
        let p = RouterParams::uocr2();
        let mut x = vec![ENTMAX_NEG; 20];
        let cand: Vec<bool> = (0..20).map(|b| b < 6).collect();
        x[0] = 1000.0;
        x[1..6].fill(0.0);
        for r in [full_topm(&x, &cand, &p), entmax15_all(&x, &cand, &p)] {
            assert_eq!(r.blocks, [0, 1, 2, 3, 4, 5]);
            let floor = p.floor.ln();
            let mu = (0.0 + 5.0 * floor) / 6.0;
            assert!((r.bias[0] - (0.0 - mu)).abs() < 1e-5);
            for &b in &r.bias[1..] {
                assert!((b - (floor - mu)).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn prefill_window_is_set_by_the_forward_not_the_row() {
        // 600 rows → nblk 19 → M = 16 for every row. Row 300 sits in block 9
        // and has 6 remote blocks, all candidates (its query page is −1), and
        // every candidate input is ≈ −12800 < −1e4. In the forward's window
        // the 13 non-candidates (at −1e4) fill the first 12 slots, so nothing
        // is read. A per-row window over its 10 causal blocks would read all 6.
        let mut ids = vec![BOS, PG];
        ids.extend(std::iter::repeat_n(IMG, 100));
        ids.extend((0..498).map(|k| 1000 + k as u32));
        let coords = PageCoords::build(&ids).unwrap();
        let v = [10.0f32; HEAD_DIM];
        let ik = Mat::from_vec(600, HEAD_DIM, v.repeat(600));
        let iq = Mat::from_vec(600, HEAD_DIM, v.map(|e| -e).repeat(600));
        let p = RouterParams::uocr2();
        let zero = [0.0f32; HEAD_DIM];
        let routes = prefill_routes(&iq, &ik, &coords, &zero, &p, EntmaxDomain::TopM).unwrap();
        assert_eq!(routes.len(), 600);
        assert!(routes[300].blocks.is_empty(), "{:?}", routes[300]);
        let sc = scores(iq.row(300), 300, &summaries(&ik, &zero));
        assert_eq!(sc.len(), 19);
        let x = entmax_inputs(&sc, &candidates(&sc, None), &p);
        assert!(x[..6].iter().all(|&v| v < -1.0e4), "{x:?}");
        let per_row = route(&sc[..10], None, &p, EntmaxDomain::TopM);
        assert_eq!(per_row.blocks, [0, 1, 2, 3, 4, 5]);
        // Every row's window spans the forward (rows before 128 read nothing).
        assert!(routes[..128].iter().all(|r| r.blocks.is_empty()));
        assert!(
            prefill_routes(
                &iq,
                &ik,
                &PageCoords::build(&ids[..599]).unwrap(),
                &zero,
                &p,
                EntmaxDomain::TopM
            )
            .is_err()
        );
    }

    #[test]
    fn single_page_base_reads_every_image_block_of_its_page() {
        // BOS + <PAGE> + 273 images + 8 instruction rows, then a decode query
        // at row 700 answering page 0. The images fill blocks 0..=8, all
        // remote and all allowed, and 9 ≤ K' = 12: every one is read, with
        // biases that differ.
        let mut ids = vec![BOS, PG];
        ids.extend(std::iter::repeat_n(IMG, 273));
        ids.extend((0..8).map(|k| 1000 + k as u32));
        let coords = PageCoords::build(&ids).unwrap();
        let mut img_page = coords.img_page.clone();
        img_page.resize(701, -1);
        let blocks = BlockPages::build(&img_page);
        let mut rng = Lcg(0x5eed_0b09);
        let ik = random_mat(&mut rng, 701, HEAD_DIM, 1.0);
        let hc: Vec<f32> = (0..HEAD_DIM)
            .map(|_| uniform(&mut rng, -1.0, 1.0))
            .collect();
        let iq: Vec<f32> = (0..HEAD_DIM)
            .map(|_| uniform(&mut rng, -1.0, 1.0))
            .collect();
        let sums = summaries(&ik, &hc);
        let sc = scores(&iq, 700, &sums);
        let allow = page_gate_decode(&blocks.bmin, &blocks.bmax, 0, sums.nblk());
        assert_eq!(allow.iter().filter(|&&a| a).count(), 9);
        let p = RouterParams::uocr2();
        for domain in [EntmaxDomain::TopM, EntmaxDomain::AllCandidates] {
            let r = route(&sc, Some(&allow), &p, domain);
            assert_eq!(r.blocks, (0..9).collect::<Vec<u32>>(), "{domain:?}");
            assert!(
                r.bias.iter().any(|&b| (b - r.bias[0]).abs() > 1e-3),
                "{domain:?}"
            );
        }
    }

    // ── Attention (U2-B3) ───────────────────────────────────────────────────

    fn random_kv(rng: &mut Lcg, rows: usize) -> (Mat, Mat, HeadMajorKv) {
        let k = random_mat(rng, rows, NUM_HEADS * HEAD_DIM, 1.0);
        let v = random_mat(rng, rows, NUM_HEADS * HEAD_DIM, 1.0);
        let kv = HeadMajorKv::from_rows(&k, &v).unwrap();
        (k, v, kv)
    }

    /// A random selection over blocks `0..nblk`: any blocks at all (future,
    /// local, partially local), biases in ±3.
    fn random_selection(rng: &mut Lcg, nblk: usize) -> BlockSelection {
        let mut blocks: Vec<u32> = (0..rng.below(14))
            .map(|_| rng.below(nblk as u64) as u32)
            .collect();
        blocks.sort_unstable();
        blocks.dedup();
        let bias = blocks.iter().map(|_| uniform(rng, -3.0, 3.0)).collect();
        BlockSelection::new(blocks, bias).unwrap()
    }

    fn assert_bitwise(a: &Mat, b: &Mat, what: &str) {
        assert_eq!(a.shape(), b.shape(), "{what}");
        for (i, (x, y)) in a.data.iter().zip(&b.data).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{what}: element {i}: {x} vs {y}");
        }
    }

    /// The attention formula evaluated directly in f64 from per-row K/V.
    fn attention_f64(
        q: &[f32],
        q_pos: usize,
        sel: &BlockSelection,
        k: &Mat,
        v: &Mat,
        h: usize,
    ) -> Vec<f64> {
        let head = |m: &Mat, j: usize| m.row(j)[h * HEAD_DIM..(h + 1) * HEAD_DIM].to_vec();
        let qh = &q[h * HEAD_DIM..(h + 1) * HEAD_DIM];
        let mut logits = Vec::new();
        for j in 0..=q_pos {
            let local = j + LOCAL_WINDOW > q_pos;
            let bias = sel
                .blocks()
                .iter()
                .position(|&b| b as usize == j / BLOCK_SIZE)
                .map(|i| f64::from(sel.bias()[i]));
            if local || bias.is_some() {
                let d: f64 = qh
                    .iter()
                    .zip(head(k, j))
                    .map(|(&a, b)| f64::from(a) * f64::from(b))
                    .sum();
                let b = if local { 0.0 } else { bias.unwrap_or(0.0) };
                logits.push((j, d / 128f64.sqrt() + b));
            }
        }
        let max = logits
            .iter()
            .map(|&(_, l)| l)
            .fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = logits.iter().map(|&(_, l)| (l - max).exp()).sum();
        let mut out = vec![0.0f64; HEAD_DIM];
        for &(j, l) in &logits {
            let p = (l - max).exp() / z;
            for (o, x) in out.iter_mut().zip(head(v, j)) {
                *o += p * f64::from(x);
            }
        }
        out
    }

    #[test]
    fn gathered_attention_equals_the_dense_oracle_bit_for_bit() {
        let mut rng = Lcg(0x5eed_0b31);
        for case in 0..40 {
            let rows = 1 + rng.below(420) as usize;
            let (k, v, kv) = random_kv(&mut rng, rows);
            let n_q = 1 + rng.below(6) as usize;
            let mut q_pos: Vec<usize> = (0..n_q).map(|_| rng.below(rows as u64) as usize).collect();
            // Always cover the first rows (no remote block) and the last.
            q_pos[0] = rng.below(rows.min(BLOCK_SIZE * LOCAL_BLOCKS) as u64) as usize;
            *q_pos.last_mut().unwrap() = rows - 1;
            let sel: Vec<BlockSelection> = (0..n_q)
                .map(|i| {
                    if i == 1 {
                        BlockSelection::default()
                    } else {
                        random_selection(&mut rng, rows.div_ceil(BLOCK_SIZE) + 1)
                    }
                })
                .collect();
            let q = random_mat(&mut rng, n_q, NUM_HEADS * HEAD_DIM, 1.0);
            let gathered = attend(&q, &q_pos, &sel, &kv).unwrap();
            let dense = attend_dense(&q, &q_pos, &sel, &kv).unwrap();
            assert_bitwise(&gathered, &dense, &format!("case {case}"));
            // And both evaluate the formula itself.
            for r in 0..n_q {
                for h in [0, NUM_HEADS - 1] {
                    let want = attention_f64(q.row(r), q_pos[r], &sel[r], &k, &v, h);
                    let got = &gathered.row(r)[h * HEAD_DIM..(h + 1) * HEAD_DIM];
                    for (a, b) in got.iter().zip(&want) {
                        assert!(
                            (f64::from(*a) - b).abs() < 1e-5,
                            "case {case} row {r} head {h}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn local_rows_win_over_a_selected_block() {
        let mut rng = Lcg(0x5eed_0b32);
        let rows = 12 * BLOCK_SIZE;
        let (_, v, kv) = random_kv(&mut rng, rows);
        let q = random_mat(&mut rng, 1, NUM_HEADS * HEAD_DIM, 1.0);
        let none = BlockSelection::default();
        let one = |b: u32| BlockSelection::new(vec![b], vec![2.5]).unwrap();
        let run = |pos: usize, s: &BlockSelection| {
            attend(&q, &[pos], std::slice::from_ref(s), &kv).unwrap()
        };

        // q = 0 reads only itself.
        let first = run(0, &one(0));
        for h in 0..NUM_HEADS {
            let want = &v.row(0)[h * HEAD_DIM..(h + 1) * HEAD_DIM];
            assert_eq!(&first.row(0)[h * HEAD_DIM..(h + 1) * HEAD_DIM], want);
        }
        // q < 128: block 0 is wholly local, so selecting it changes nothing.
        assert_bitwise(&run(100, &one(0)), &run(100, &none), "q < 128");
        // q = 287 (block 8, last row): the window is rows 160..=287. Block 5
        // (160..192) is wholly local; block 4 (128..160) is just outside it.
        assert_eq!(
            attended_runs(287, &one(4))[0],
            Run {
                block: 4,
                from: 0,
                to: 32,
                bias: 2.5
            }
        );
        assert_bitwise(
            &run(287, &one(5)),
            &run(287, &none),
            "block on the boundary",
        );
        assert_ne!(run(287, &one(4)), run(287, &none));
        // q = 288: the window starts at 161, so block 5 is partially local and
        // only its row 160 takes the bias.
        assert_eq!(
            attended_runs(288, &one(5))[0],
            Run {
                block: 5,
                from: 0,
                to: 1,
                bias: 2.5
            }
        );
        assert_ne!(run(288, &one(5)), run(288, &none));
        // A future block is never read.
        assert_bitwise(&run(287, &one(11)), &run(287, &none), "future block");
    }

    #[test]
    fn prefill_attention_is_thread_count_invariant() {
        let mut rng = Lcg(0x5eed_0b33);
        let rows = 300;
        let (_, _, kv) = random_kv(&mut rng, rows);
        let q = random_mat(&mut rng, rows, NUM_HEADS * HEAD_DIM, 1.0);
        let q_pos: Vec<usize> = (0..rows).collect();
        let sel: Vec<BlockSelection> = (0..rows)
            .map(|_| random_selection(&mut rng, rows.div_ceil(BLOCK_SIZE)))
            .collect();
        let at = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| attend(&q, &q_pos, &sel, &kv).unwrap())
        };
        let one = at(1);
        assert_bitwise(&at(2), &one, "2 threads");
        let n = std::thread::available_parallelism().map_or(4, |n| n.get());
        assert_bitwise(&at(n), &one, "N threads");
    }

    #[test]
    fn attention_inputs_are_validated() {
        let mut rng = Lcg(0x5eed_0b34);
        let (k, _, kv) = random_kv(&mut rng, 40);
        let q = random_mat(&mut rng, 1, NUM_HEADS * HEAD_DIM, 1.0);
        let none = [BlockSelection::default()];
        assert!(
            attend(&q, &[40], &none, &kv).is_err(),
            "position past the rows"
        );
        assert!(
            attend(&q, &[1, 2], &none, &kv).is_err(),
            "positions per row"
        );
        assert!(attend_dense(&Mat::zeros(1, 7), &[0], &none, &kv).is_err());
        assert!(HeadMajorKv::from_rows(&k, &Mat::zeros(39, NUM_HEADS * HEAD_DIM)).is_err());
        assert!(BlockSelection::new(vec![3, 2], vec![0.0, 0.0]).is_err());
        assert!(BlockSelection::new(vec![2, 2], vec![0.0, 0.0]).is_err());
        assert!(BlockSelection::new(vec![2], vec![]).is_err());
        let route = Route {
            blocks: vec![1, 4],
            bias: vec![0.5, -0.5],
            support: 2,
        };
        let sel = BlockSelection::from(route);
        assert_eq!(
            (sel.blocks(), sel.bias()),
            (&[1u32, 4][..], &[0.5f32, -0.5][..])
        );
    }

    // ── U2-B4: layer cache and block retirement ──

    const EOS: u32 = special::EOS;
    const TEXT: u32 = 1000;

    fn random_rows(rng: &mut Lcg, n: usize) -> (Mat, Mat, Mat) {
        let k = random_mat(rng, n, NUM_HEADS * HEAD_DIM, 1.0);
        let v = random_mat(rng, n, NUM_HEADS * HEAD_DIM, 1.0);
        let ik = random_mat(rng, n, HEAD_DIM, 1.0);
        (k, v, ik)
    }

    fn random_head_cls(rng: &mut Lcg) -> Vec<f32> {
        (0..HEAD_DIM).map(|_| uniform(rng, -1.0, 1.0)).collect()
    }

    fn assert_same_summaries(a: &BlockSummaries, b: &BlockSummaries, what: &str) {
        assert_eq!(a.valid, b.valid, "{what}: valid");
        let bits = |s: &BlockSummaries| s.kbar.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(a), bits(b), "{what}: kbar");
    }

    #[test]
    fn incremental_summaries_equal_a_recompute_bit_for_bit() {
        let mut rng = Lcg(0x5eed_0b40);
        for case in 0..6 {
            let head_cls = if case == 0 {
                vec![0.0; HEAD_DIM]
            } else {
                random_head_cls(&mut rng)
            };
            let mut cache = SrSwaLayerCache::new(&head_cls, true).unwrap();
            let (mut all_k, mut all_v, mut all_ik) = (Vec::new(), Vec::new(), Vec::new());
            while cache.rows < 420 {
                // Prefill-sized and decode-sized appends, some ending mid-block.
                let n = if rng.below(3) == 0 {
                    1
                } else {
                    1 + rng.below(70) as usize
                };
                let (k, v, mut ik) = random_rows(&mut rng, n);
                if rng.below(8) == 0 {
                    let at = rng.below(ik.data.len() as u64) as usize;
                    ik.data[at] = f32::INFINITY;
                }
                cache.append(&k, &v, &ik, &vec![-1; n]).unwrap();
                all_k.extend_from_slice(&k.data);
                all_v.extend_from_slice(&v.data);
                all_ik.extend_from_slice(&ik.data);
                let rows = cache.rows;
                let ik_all = Mat::from_vec(rows, HEAD_DIM, all_ik.clone());
                assert_same_summaries(
                    cache.summaries(),
                    &summaries(&ik_all, &head_cls),
                    &format!("case {case} at {rows} rows"),
                );
            }
            // The BlockKv view is the head-major layout of the same rows.
            let rows = cache.rows;
            let width = NUM_HEADS * HEAD_DIM;
            let reference = HeadMajorKv::from_rows(
                &Mat::from_vec(rows, width, all_k),
                &Mat::from_vec(rows, width, all_v),
            )
            .unwrap();
            for b in 0..cache.nblk() {
                for h in 0..NUM_HEADS {
                    assert_eq!(cache.keys(h, b), reference.keys(h, b), "keys {h} {b}");
                    assert_eq!(cache.values(h, b), reference.values(h, b), "values {h} {b}");
                }
            }
        }
    }

    #[test]
    fn block_pages_of_a_three_page_base_prompt_hand_computed() {
        // BOS 0; page p: marker 1 + 274p, images 2 + 274p ..= 274 + 274p;
        // instruction 823..=826. 827 rows, 26 blocks.
        let ids = prompt(&[273, 273, 273], 4);
        let coords = PageCoords::build(&ids).unwrap();
        let mut want = vec![(0, 0); 26];
        want[8] = (0, 1); // rows 256..=274 page 0, marker 275, 276..=287 page 1
        for w in &mut want[9..=16] {
            *w = (1, 1);
        }
        want[17] = (1, 2); // rows 544..=548 page 1, marker 549, 550..=575 page 2
        for w in &mut want[18..=25] {
            *w = (2, 2); // block 25 also holds the instruction
        }
        let mut rng = Lcg(0x5eed_0b41);
        let mut whole = SrSwaLayerCache::new(&random_head_cls(&mut rng), true).unwrap();
        let mut chunked = whole.clone();
        let (k, v, ik) = random_rows(&mut rng, ids.len());
        whole.append(&k, &v, &ik, &coords.img_page).unwrap();
        // The same prompt in uneven chunks gives the same pages.
        let mut at = 0;
        for n in [5, 300, 27, 495] {
            let sub =
                |m: &Mat| Mat::from_vec(n, m.cols, m.data[at * m.cols..(at + n) * m.cols].to_vec());
            chunked
                .append(&sub(&k), &sub(&v), &sub(&ik), &coords.img_page[at..at + n])
                .unwrap();
            at += n;
        }
        let got: Vec<(i32, i32)> = whole
            .bmin()
            .iter()
            .copied()
            .zip(whole.bmax().iter().copied())
            .collect();
        assert_eq!(got, want);
        assert_eq!(
            (chunked.bmin(), chunked.bmax()),
            (whole.bmin(), whole.bmax())
        );
        let reference = BlockPages::build(&coords.img_page);
        assert_eq!(
            (whole.bmin(), whole.bmax()),
            (&reference.bmin[..], &reference.bmax[..])
        );
        // Generated rows never count as image rows: block 25 keeps (2, 2) and
        // a block of generated rows has none.
        let (k, v, ik) = random_rows(&mut rng, 40);
        whole.append(&k, &v, &ik, &[-1; 40]).unwrap();
        assert_eq!((whole.bmin()[25], whole.bmax()[25]), (2, 2));
        assert_eq!((whole.bmin()[26], whole.bmax()[26]), (-1, -1));
        // Page 1's gate allows exactly the blocks holding page-1 images.
        let allow = page_gate_decode(whole.bmin(), whole.bmax(), 1, whole.nblk());
        let allowed: Vec<usize> = (0..allow.len()).filter(|&c| allow[c]).collect();
        assert_eq!(allowed, (8..=17).collect::<Vec<_>>());
    }

    /// How a synthetic decode run ends after its last page body.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum End {
        /// EOS from an ordinary step.
        Eos,
        /// One more `<PAGE>`, whose step's output the guard forces to EOS: an
        /// extra page unwindowed, the lookahead marker in a window.
        ForcedMarker,
    }

    /// A synthetic one-layer decode run. Weights are random; each step's rows
    /// are nudged by the previous attention output and each token follows a
    /// hash of it, so any bit of difference changes what follows.
    #[derive(Debug, Clone)]
    struct DocPlan {
        /// Image page per prompt row.
        img_page: Vec<i32>,
        /// Text tokens before the first generated `<PAGE>` (query page −1).
        lead: usize,
        /// Query page of the first generated `<PAGE>`: 0 fresh, 1 reused.
        first_qp: i32,
        /// Per generated page, the shortest and longest body.
        bodies: Vec<(usize, usize)>,
        end: End,
        /// Pinned rows; a non-empty set also turns the text-tail margin on.
        pins: Vec<Range<usize>>,
        retire: bool,
        /// Whether the guard reports the final marker's step as forced.
        guard: bool,
        domain: EntmaxDomain,
        seed: u64,
    }

    #[derive(Debug, PartialEq)]
    struct Step {
        q: usize,
        qp: i32,
        forced: bool,
        open: Option<OpenRow>,
        blocks: Vec<u32>,
        bias: Vec<u32>,
        out: Vec<u32>,
        next: u32,
    }

    struct DocRun {
        steps: Vec<Step>,
        fallback: Option<RetireFallback>,
        cache: SrSwaLayerCache,
        most_dead: usize,
        /// Blocks the forced step would have freed had it not been forced.
        forced_would_free: usize,
        freed_on_forced: usize,
        peak_seen: usize,
    }

    fn fnv(xs: &[f32]) -> u64 {
        xs.iter().fold(0xcbf2_9ce4_8422_2325, |h, x| {
            (h ^ u64::from(x.to_bits())).wrapping_mul(0x0100_0000_01b3)
        })
    }

    fn run_doc(plan: &DocPlan) -> DocRun {
        let mut rng = Lcg(plan.seed);
        let width = NUM_HEADS * HEAD_DIM;
        let p = RouterParams::uocr2();
        let mut cache = SrSwaLayerCache::new(&random_head_cls(&mut rng), plan.retire).unwrap();
        let (k, v, ik) = random_rows(&mut rng, plan.img_page.len());
        cache.append(&k, &v, &ik, &plan.img_page).unwrap();
        cache.set_pinned(&plan.pins, !plan.pins.is_empty()).unwrap();
        let mut run = DocRun {
            steps: Vec::new(),
            fallback: None,
            cache: cache.clone(),
            most_dead: 0,
            forced_would_free: 0,
            freed_on_forced: 0,
            peak_seen: cache.live_bytes(),
        };
        let mut prev = vec![0.0f32; width];
        let mut token = if plan.lead > 0 { TEXT } else { PG };
        let (mut markers, mut lead_fed, mut body) = (0usize, 0usize, 0usize);
        loop {
            let q = cache.rows;
            let forced = token == PG && markers == plan.bodies.len();
            if token == PG {
                markers += 1;
                body = 0;
            } else if markers == 0 {
                lead_fed += 1;
            } else {
                body += 1;
            }
            let qp = if markers == 0 {
                -1
            } else {
                plan.first_qp + markers as i32 - 1
            };
            let nudge = |m: &mut Mat, from: usize| {
                for (x, &o) in m.data.iter_mut().zip(&prev[from..]) {
                    *x += 0.05 * o;
                }
            };
            let (mut k, mut v, mut ik) = random_rows(&mut rng, 1);
            let mut qm = random_mat(&mut rng, 1, width, 1.0);
            let mut iq = random_mat(&mut rng, 1, HEAD_DIM, 1.0);
            nudge(&mut k, 0);
            nudge(&mut v, 0);
            nudge(&mut qm, 0);
            nudge(&mut ik, 0);
            nudge(&mut iq, HEAD_DIM);
            cache.append(&k, &v, &ik, &[-1]).unwrap();
            run.peak_seen = run.peak_seen.max(cache.live_bytes());
            // Retire before this step's own routing and attention, the order
            // in which an off-by-one in the window rule would be read.
            if forced {
                run.forced_would_free = cache.clone().retire_dead_blocks(qp, q, false).unwrap();
            }
            let freed = cache
                .retire_dead_blocks(qp, q, forced && plan.guard)
                .unwrap();
            if forced {
                run.freed_on_forced = freed;
            }
            run.most_dead = run.most_dead.max(cache.dead_blocks());
            let dr = match cache.decode_route(&iq.data, qp, &p, plan.domain, forced && plan.guard) {
                Ok(dr) => dr,
                Err(event) => {
                    run.fallback = Some(event);
                    break;
                }
            };
            let sel = BlockSelection::from(dr.route.clone());
            let out = attend(&qm, &[q], std::slice::from_ref(&sel), &cache).unwrap();
            let next = if forced {
                EOS
            } else if markers == 0 {
                if lead_fed >= plan.lead { PG } else { TEXT }
            } else {
                let (lo, hi) = plan.bodies[markers - 1];
                let ends = body >= hi || (body >= lo && fnv(&out.data).is_multiple_of(4));
                match (ends, markers < plan.bodies.len(), plan.end) {
                    (false, _, _) => TEXT,
                    (true, true, _) | (true, false, End::ForcedMarker) => PG,
                    (true, false, End::Eos) => EOS,
                }
            };
            run.steps.push(Step {
                q,
                qp,
                forced,
                open: dr.open,
                blocks: dr.route.blocks.clone(),
                bias: dr.route.bias.iter().map(|b| b.to_bits()).collect(),
                out: out.data.iter().map(|x| x.to_bits()).collect(),
                next,
            });
            if next == EOS {
                break;
            }
            prev = out.data;
            token = next;
        }
        run.cache = cache;
        run
    }

    /// A fresh prompt's image pages (BOS, pages of `images` rows, instruction).
    fn fresh_pages(images: &[usize], instruction: usize) -> Vec<i32> {
        PageCoords::build(&prompt(images, instruction))
            .unwrap()
            .img_page
    }

    fn base_plan(img_page: Vec<i32>, bodies: Vec<(usize, usize)>, end: End, seed: u64) -> DocPlan {
        DocPlan {
            img_page,
            lead: 0,
            first_qp: 0,
            bodies,
            end,
            pins: Vec::new(),
            retire: true,
            guard: true,
            domain: EntmaxDomain::TopM,
            seed,
        }
    }

    /// Retired and unretired runs agree on every step whose output is not
    /// guard-forced: selections, biases, attention outputs and tokens.
    fn assert_same_run(retired: &DocRun, full: &DocRun, what: &str) {
        assert!(
            retired.fallback.is_none() && full.fallback.is_none(),
            "{what}: fallback"
        );
        assert_eq!(retired.steps.len(), full.steps.len(), "{what}: steps");
        for (a, b) in retired.steps.iter().zip(&full.steps) {
            if a.forced && a.open.is_some() {
                assert_eq!((a.q, a.next), (b.q, b.next), "{what}: forced step");
            } else {
                assert_eq!(a, b, "{what}: step at {}", a.q);
            }
        }
    }

    #[test]
    fn retired_runs_equal_unretired_runs_bit_for_bit() {
        let shapes: Vec<(&str, DocPlan)> = vec![
            (
                "S1 one page",
                DocPlan {
                    lead: 2,
                    ..base_plan(
                        fresh_pages(&[273], 4),
                        vec![(200, 260)],
                        End::Eos,
                        0x5eed_0b42,
                    )
                },
            ),
            (
                "S2 three pages",
                base_plan(
                    fresh_pages(&[273; 3], 4),
                    vec![(60, 120); 3],
                    End::Eos,
                    0x5eed_0b43,
                ),
            ),
            (
                "S2 three pages, all-candidates entmax",
                DocPlan {
                    domain: EntmaxDomain::AllCandidates,
                    ..base_plan(
                        fresh_pages(&[273; 3], 4),
                        vec![(60, 120); 3],
                        End::Eos,
                        0x5eed_0b44,
                    )
                },
            ),
            (
                "S3 six pages, extra <PAGE> at the end",
                base_plan(
                    fresh_pages(&[273; 6], 4),
                    vec![(20, 60); 6],
                    End::ForcedMarker,
                    0x5eed_0b45,
                ),
            ),
        ];
        for (what, plan) in shapes {
            let retired = run_doc(&plan);
            let full = run_doc(&DocPlan {
                retire: false,
                ..plan.clone()
            });
            assert_same_run(&retired, &full, what);
            assert!(retired.most_dead > 0, "{what}: nothing was retired");
            assert_eq!(full.most_dead, 0, "{what}");
            // Memory: every freed block takes its slab with it.
            let (c, f) = (&retired.cache, &full.cache);
            let slab = BLOCK_SIZE * ROW_BYTES;
            assert_eq!(
                f.live_bytes(),
                SrSwaLayerCache::estimate_bytes(f.rows),
                "{what}"
            );
            assert_eq!(
                c.live_bytes(),
                f.live_bytes() - c.dead_blocks() * slab,
                "{what}"
            );
            assert_eq!(
                (c.peak_bytes(), f.peak_bytes()),
                (retired.peak_seen, full.peak_seen),
                "{what}"
            );
            assert!(
                c.peak_bytes() < f.peak_bytes(),
                "{what}: retirement saved nothing"
            );
        }
    }

    #[test]
    fn three_page_retirement_frees_passed_pages_but_keeps_straddles_and_the_window() {
        let plan = base_plan(
            fresh_pages(&[273; 3], 4),
            vec![(150, 150), (150, 150), (40, 40)],
            End::Eos,
            0x5eed_0b46,
        );
        let run = run_doc(&plan);
        assert!(run.fallback.is_none());
        let c = &run.cache;
        let last = run.steps.last().unwrap().q;
        // Page 0's blocks 0..=7 and page 1's 9..=16 are gone, and so is the
        // 0/1 straddle (bmax 1 < qp 2); the 1/2 straddle 17 and page 2's
        // blocks 18..=25 stay, as does everything the local window touches.
        for b in 0..=16 {
            assert!(!c.is_live(b), "block {b}");
        }
        for b in 17..=25 {
            assert!(c.is_live(b), "block {b}");
        }
        for b in (last + 1 - LOCAL_WINDOW) / BLOCK_SIZE..c.nblk() {
            assert!(c.is_live(b), "window block {b}");
        }
        // Generated text behind the window is gone.
        let gen_first_full = 827usize.div_ceil(BLOCK_SIZE);
        assert!(!c.is_live(gen_first_full));
    }

    #[test]
    fn the_detector_passes_guard_forced_steps_and_fires_otherwise() {
        // An extra <PAGE> after the last page (qp = N): an open row, but the
        // guard forces the step, so no fallback, and that step frees nothing
        // although blocks are dead by the rule.
        let extra = base_plan(
            fresh_pages(&[273; 3], 4),
            vec![(60, 60); 3],
            End::ForcedMarker,
            0x5eed_0b47,
        );
        let run = run_doc(&extra);
        let last = run.steps.last().unwrap();
        assert!(run.fallback.is_none());
        assert_eq!(
            (last.qp, last.forced, last.open, last.next),
            (3, true, Some(OpenRow::NoAllowedBlock), EOS)
        );
        assert!(run.cache.dead_blocks() > 0);
        assert!(run.forced_would_free > 0);
        assert_eq!(run.freed_on_forced, 0);
        // The same row with the guard off fires the detector.
        let unguarded = run_doc(&DocPlan {
            guard: false,
            ..extra.clone()
        });
        let event = unguarded.fallback.expect("the detector fires");
        assert_eq!(
            (event.query_page, event.open, event.q_pos),
            (3, OpenRow::NoAllowedBlock, last.q)
        );
        assert!(event.dead_blocks > 0);
        assert!(event.to_string().contains("OQ-V2-11"));
        // A mid-page open row: a 20-image last page sits inside the local
        // window when its body starts, so page 2 is allowed but not remote.
        let tiny = base_plan(
            fresh_pages(&[273, 273, 20], 2),
            vec![(40, 40), (40, 40), (10, 10)],
            End::Eos,
            0x5eed_0b48,
        );
        let event = run_doc(&tiny).fallback.expect("the detector fires");
        assert_eq!(
            (event.query_page, event.open),
            (2, OpenRow::NoRemoteAllowed)
        );
        assert!(event.dead_blocks > 0);
        // Without retirement nothing fires, and the open row is reported.
        let full = run_doc(&DocPlan {
            retire: false,
            ..tiny
        });
        assert!(full.fallback.is_none());
        assert!(
            full.steps
                .iter()
                .any(|s| s.qp == 2 && s.open == Some(OpenRow::NoRemoteAllowed))
        );
        // Before the first <PAGE> rows are open (qp < 0) and nothing retires.
        let lead = run_doc(&DocPlan {
            lead: 200,
            ..base_plan(fresh_pages(&[273], 4), vec![(5, 5)], End::Eos, 0x5eed_0b49)
        });
        assert!(lead.fallback.is_none());
        let first_page = lead.steps.iter().position(|s| s.qp == 0).unwrap();
        assert!(
            lead.steps[..first_page]
                .iter()
                .all(|s| s.open == Some(OpenRow::NoPage))
        );
        assert_eq!(lead.steps[first_page - 1].qp, -1);
        assert!(
            lead.most_dead > 0,
            "imageless lead blocks retire once qp = 0"
        );
    }

    /// A window's prompt and the rows §5.10.3 retains at its end.
    struct Window {
        img_page: Vec<i32>,
        first_qp: i32,
        own_pages: usize,
        /// The last own page's marker row.
        marker: usize,
        /// `[E − 128 − (E mod 32), E)`.
        image_tail: Range<usize>,
        /// The lookahead page's marker and images.
        lookahead: Range<usize>,
    }

    /// Window 0 of a W-page window: BOS, W own pages, the lookahead page, the
    /// instruction.
    fn fresh_window(w: usize) -> Window {
        let img_page = fresh_pages(&vec![273; w + 1], 4);
        let marker = 1 + 274 * (w - 1);
        let e = marker + 274;
        Window {
            img_page,
            first_qp: 0,
            own_pages: w,
            marker,
            image_tail: e - 128 - e % 32..e,
            lookahead: e..e + 274,
        }
    }

    /// A reused window (§5.10.4): the copied tail marker and `tail_imgs` image
    /// rows of local page 0, the copied lookahead (local page 1), new pages
    /// 2..=m (m the new lookahead), then `text` copied text-tail rows.
    fn reused_window(m: usize, tail_imgs: usize, text: usize) -> Window {
        let mut img_page = vec![-1];
        img_page.extend(std::iter::repeat_n(0, tail_imgs));
        for page in 1..=m as i32 {
            img_page.push(-1);
            img_page.extend(std::iter::repeat_n(page, 273));
        }
        img_page.extend(std::iter::repeat_n(-1, text));
        let marker = 1 + tail_imgs + 274 * (m - 2);
        let e = marker + 274;
        Window {
            img_page,
            first_qp: 1,
            own_pages: m - 1,
            marker,
            image_tail: e - 128 - e % 32..e,
            lookahead: e..e + 274,
        }
    }

    #[test]
    fn pinned_rows_survive_to_the_end_of_a_window() {
        let windows = [
            ("W=8 window 0", fresh_window(8)),
            ("W=2 window 0", fresh_window(2)),
            ("reused, one own page", reused_window(2, 128, 128)),
            ("reused, two own pages", reused_window(3, 159, 37)),
            ("reused, three own pages", reused_window(4, 145, 0)),
        ];
        assert_eq!(windows[0].1.marker, 1919);
        assert_eq!(1919 % 32, 31);
        for (i, (what, win)) in windows.iter().enumerate() {
            // Short bodies, then a tail-page body long enough for a full
            // 128-row text tail.
            let mut bodies = vec![(10, 30); win.own_pages - 1];
            bodies.push((140, 170));
            let plan = DocPlan {
                first_qp: win.first_qp,
                pins: vec![
                    win.marker..win.marker + 1,
                    win.image_tail.clone(),
                    win.lookahead.clone(),
                ],
                ..base_plan(
                    win.img_page.clone(),
                    bodies,
                    End::ForcedMarker,
                    0x5eed_0b50 + i as u64,
                )
            };
            let pinned = run_doc(&plan);
            let full = run_doc(&DocPlan {
                retire: false,
                ..plan.clone()
            });
            assert_same_run(&pinned, &full, what);
            let last = pinned.steps.last().unwrap();
            assert!(
                last.forced && last.open.is_none(),
                "{what}: lookahead marker step"
            );
            let q = last.q;
            let kept = [
                win.marker..win.marker + 1,
                win.image_tail.clone(),
                win.lookahead.clone(),
                q - 128..q,
            ];
            let a = pinned
                .cache
                .serialize_tail(&kept)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(a, full.cache.serialize_tail(&kept).unwrap(), "{what}");
            assert!(pinned.most_dead > 0, "{what}: nothing was retired");
            if i == 0 {
                // Unpinned, the marker's block (page-6 images, then row 1919)
                // dies once qp = 7.
                let loose = run_doc(&DocPlan {
                    pins: Vec::new(),
                    ..plan
                });
                assert!(!loose.cache.is_live(1919 / BLOCK_SIZE));
                assert!(loose.cache.serialize_tail(&kept[..1]).is_err());
            }
        }
    }

    #[test]
    fn the_text_tail_margin_keeps_one_more_row() {
        let mut rng = Lcg(0x5eed_0b51);
        let img_page = fresh_pages(&[273], 4);
        let mut cache = SrSwaLayerCache::new(&random_head_cls(&mut rng), true).unwrap();
        let (k, v, ik) = random_rows(&mut rng, img_page.len() + 300);
        let mut pages = img_page.clone();
        pages.resize(k.rows, -1);
        cache.append(&k, &v, &ik, &pages).unwrap();
        // Block 10 (rows 320..=351) holds generated text only; block 8 (rows
        // 256..=287) ends page 0's images.
        assert_eq!((cache.bmin()[10], cache.bmax()[8]), (-1, 0));
        let mut margin = cache.clone();
        margin.set_pinned(&[], true).unwrap();
        let mut image = margin.clone();
        cache.retire_dead_blocks(0, 351 + 127, false).unwrap();
        assert!(cache.is_live(10) && !cache.is_live(9));
        cache.retire_dead_blocks(0, 351 + 128, false).unwrap();
        assert!(!cache.is_live(10));
        margin.retire_dead_blocks(0, 351 + 128, false).unwrap();
        assert!(margin.is_live(10) && !margin.is_live(9));
        margin.retire_dead_blocks(0, 351 + 129, false).unwrap();
        assert!(!margin.is_live(10));
        // The margin is for text blocks only: an image block retires at q − 128.
        image.retire_dead_blocks(1, 287 + 127, false).unwrap();
        assert!(image.is_live(8) && !image.is_live(7));
        image.retire_dead_blocks(1, 287 + 128, false).unwrap();
        assert!(!image.is_live(8));
    }

    #[test]
    fn tails_round_trip_and_reject_bad_input() {
        let mut rng = Lcg(0x5eed_0b52);
        let head_cls = random_head_cls(&mut rng);
        let mut cache = SrSwaLayerCache::new(&head_cls, true).unwrap();
        let (k, v, ik) = random_rows(&mut rng, 300);
        cache.append(&k, &v, &ik, &[-1; 300]).unwrap();
        let ranges = [3..4, 100..160, 250..300];
        let bytes = cache.serialize_tail(&ranges).unwrap();
        assert_eq!(bytes.len(), TAIL_HEADER + 111 * ROW_BYTES);
        let tail = deserialize_tail(&bytes).unwrap();
        let rows: Vec<usize> = ranges.iter().flat_map(Clone::clone).collect();
        for (i, &r) in rows.iter().enumerate() {
            assert_eq!(tail.k.row(i), k.row(r));
            assert_eq!(tail.v.row(i), v.row(r));
            assert_eq!(tail.ik.row(i), ik.row(r));
        }
        // Spliced into a new cache, the summaries are recomputed.
        let mut spliced = SrSwaLayerCache::new(&head_cls, true).unwrap();
        spliced
            .append(&tail.k, &tail.v, &tail.ik, &vec![-1; rows.len()])
            .unwrap();
        assert_same_summaries(
            spliced.summaries(),
            &summaries(&tail.ik, &head_cls),
            "spliced",
        );
        assert_eq!(cache.serialize_tail(&[]).unwrap().len(), TAIL_HEADER);
        // Malformed input.
        assert!(deserialize_tail(&bytes[..bytes.len() - 1]).is_err());
        assert!(deserialize_tail(&bytes[..10]).is_err());
        for at in [0, 8, 12, 16, 20] {
            let mut bad = bytes.clone();
            bad[at] ^= 1;
            assert!(deserialize_tail(&bad).is_err(), "byte {at}");
        }
        let past = 299..301;
        assert!(
            cache.serialize_tail(std::slice::from_ref(&past)).is_err(),
            "past the cache"
        );
        cache.retire_dead_blocks(0, 299, false).unwrap();
        assert!(!cache.is_live(0));
        let retired = 3..4;
        assert!(
            cache
                .serialize_tail(std::slice::from_ref(&retired))
                .is_err(),
            "retired"
        );
    }

    #[test]
    fn memory_accounting_matches_the_allocations_and_the_estimate() {
        // Plan §6.3: over 12 layers, K+V is 120 KiB and index keys 6 KiB per token.
        assert_eq!(12 * ROW_BYTES, (120 + 6) * 1024);
        let mut rng = Lcg(0x5eed_0b53);
        let mut cache = SrSwaLayerCache::new(&random_head_cls(&mut rng), true).unwrap();
        let measured = |c: &SrSwaLayerCache| {
            let slabs: usize = c
                .slabs
                .iter()
                .flatten()
                .map(|s| (s.k.len() + s.v.len() + s.ik.len()) * size_of::<f32>())
                .sum();
            slabs
                + c.sums.kbar.len() * size_of::<f32>()
                + (c.bmin.len() + c.bmax.len()) * size_of::<i32>()
                + c.sums.valid.len()
                + c.pinned.len()
        };
        for n in [1, 31, 1, 100, 200, 7] {
            let (k, v, ik) = random_rows(&mut rng, n);
            cache.append(&k, &v, &ik, &vec![-1; n]).unwrap();
            let rows = cache.rows;
            assert_eq!(cache.live_bytes(), measured(&cache));
            assert_eq!(cache.live_bytes(), SrSwaLayerCache::estimate_bytes(rows));
            assert_eq!(cache.peak_bytes(), cache.live_bytes());
            // Against the per-token figure: at most one partial block plus
            // 16 B of bookkeeping per row over it.
            let slack = cache.live_bytes() - rows * ROW_BYTES;
            assert!(slack < BLOCK_SIZE * ROW_BYTES + cache.nblk() * BLOCK_META_BYTES);
        }
        let peak = cache.peak_bytes();
        let freed = cache.retire_dead_blocks(0, cache.rows - 1, false).unwrap();
        assert!(freed > 0);
        assert_eq!(cache.live_bytes(), measured(&cache));
        assert_eq!(cache.live_bytes(), peak - freed * BLOCK_SIZE * ROW_BYTES);
        assert_eq!(cache.peak_bytes(), peak);
    }

    #[test]
    fn retirement_rules_and_inputs_are_checked() {
        for (value, on) in [
            (None, true),
            (Some("1"), true),
            (Some("yes"), true),
            (Some(""), true),
            (Some("0"), false),
            (Some(" OFF "), false),
            (Some("false"), false),
            (Some("no"), false),
        ] {
            assert_eq!(retire_enabled_for(value), on, "{value:?}");
        }
        let mut rng = Lcg(0x5eed_0b54);
        assert!(SrSwaLayerCache::new(&[0.0; 7], true).is_err());
        let mut cache = SrSwaLayerCache::new(&random_head_cls(&mut rng), true).unwrap();
        let (k, v, ik) = random_rows(&mut rng, 400);
        assert!(cache.append(&k, &v, &ik, &[-1; 399]).is_err());
        assert!(
            cache
                .append(&k, &v, &random_mat(&mut rng, 400, 7, 1.0), &[-1; 400])
                .is_err()
        );
        cache.append(&k, &v, &ik, &[-1; 400]).unwrap();
        assert!(
            cache.retire_dead_blocks(0, 400, false).is_err(),
            "past the cache"
        );
        // qp < 0, a forced step, or retirement off: nothing is freed.
        assert_eq!(cache.retire_dead_blocks(-1, 300, false).unwrap(), 0);
        assert_eq!(cache.retire_dead_blocks(0, 300, true).unwrap(), 0);
        let mut off = SrSwaLayerCache::new(&random_head_cls(&mut rng), false).unwrap();
        off.append(&k, &v, &ik, &[-1; 400]).unwrap();
        assert_eq!(off.retire_dead_blocks(0, 399, false).unwrap(), 0);
        // Pinned blocks survive; pinning a retired one or past the cache fails.
        cache.set_pinned(&[40..41, 70..70], false).unwrap();
        assert_eq!(cache.retire_dead_blocks(0, 300, false).unwrap(), 4);
        assert!(cache.is_live(1) && !cache.is_live(0) && !cache.is_live(2));
        let (retired, past) = (0..1, 390..401);
        assert!(cache.set_pinned(&[retired, 40..41], false).is_err());
        assert!(cache.set_pinned(&[40..41, past], false).is_err());
        assert!(cache.is_live(1), "a failed pin keeps the old set");
        assert_eq!(cache.retire_dead_blocks(0, 300, false).unwrap(), 0);
        // Neither q nor qp may go back.
        assert!(cache.retire_dead_blocks(0, 299, false).is_err());
        assert!(cache.retire_dead_blocks(-1, 301, false).is_err());
        assert!(cache.retire_dead_blocks(1, 301, false).is_ok());
    }

    #[test]
    #[should_panic(expected = "was retired")]
    fn reading_a_retired_block_panics() {
        let mut rng = Lcg(0x5eed_0b55);
        let mut cache = SrSwaLayerCache::new(&random_head_cls(&mut rng), true).unwrap();
        let (k, v, ik) = random_rows(&mut rng, 200);
        cache.append(&k, &v, &ik, &[-1; 200]).unwrap();
        cache.retire_dead_blocks(0, 199, false).unwrap();
        let _ = cache.keys(0, 0);
    }
}
