//! bd-1azu.25 — GATE: CCD-sharded / L3-tiled `lm_head` bit-identity (vocab-tiled
//! head == monolithic head, BYTE-FOR-BYTE) at the REAL decode head shape, plus
//! the *negative* self-test that proves this gate is sensitive to a vocab-order
//! bug.
//!
//! ## The claim
//! The `lm_head` projects the final decode hidden `[1, 1280]` against the
//! `[129280, 1280]` vocab weight to `[1, 129280]` logits. Each logit `o` is a
//! SELF-CONTAINED dot — int8: `(Σ_i xq[i]·w[o,i]) · a_scale · w_scale[o]` — and
//! the vocab columns never reduce into one another. So splitting the 129280
//! output columns into CONTIGUOUS tiles, computing each tile, and writing it back
//! into its own `[.., tile]` column span is byte-for-byte identical to the single
//! monolithic GEMV: same single activation quantize, same per-logit i32
//! contraction (N-independent — already relied on by `decoder::gemv_i8`'s 64-row
//! blocking), same per-channel dequant operands in the same order, and the
//! ascending tile order preserves the vocab column order — so argmax/sampling are
//! unchanged. `decoder::lmhead_shard_enabled()` (`FOCR_LMHEAD_SHARD`) is the
//! DEFAULT-OFF kill-switch that routes the head through this tiling.
//!
//! ## What proves what (no duplication)
//! - The shipped private kernels (`decoder::gemv_i8_sharded`/`gemv_sharded`) are
//!   asserted byte-for-byte equal to the monolithic `decoder::gemv_i8`/`gemv` by
//!   the crate-internal unit tests `decoder::tests::lmhead_shard_*` — exercising
//!   the REAL functions over an awkward small vocab + the rayon-chunk seams.
//! - THIS file pins the same vocab-tiling invariance, MODEL-FREE, at the actual
//!   `[129280, 1280]` head shape using the public `simd::igemm_s8s8` (the exact
//!   GEMV math the decoder runs), and gives the gate teeth with a perturbed-tile
//!   self-test (a mis-scatter MUST make the equality assertion FAIL).
//!
//! No model: synthetic deterministic weights + activations only.

use franken_ocr::native_engine::decoder;
use franken_ocr::simd::igemm_s8s8;

// The fixed DeepSeek-V2 decode head dims (PROPOSED_ARCHITECTURE §6.7): hidden
// 1280, vocab 129280 — the same constants `tests/batched_forward_parity.rs` pins
// the wide `lm_head` projection at.
const HIDDEN: usize = 1280;
const VOCAB: usize = 129280;

/// Deterministic xorshift64 PRNG — reproducible, no dev-dependency (the idiom in
/// `tests/batched_forward_parity.rs` / `tests/batched_igemm_parity.rs`).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn i8(&mut self) -> i8 {
        (self.next() & 0xff) as u8 as i8
    }
    /// An f32 in roughly `[-3, 3)` spanning negatives/positives (exercises the
    /// dynamic per-row quantize: amax, round, clamp).
    fn f32(&mut self) -> f32 {
        let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
        u * 6.0 - 3.0
    }
    fn fill_i8(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| self.i8()).collect()
    }
    fn fill_f32(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }
    /// Per-output-channel symmetric weight scales, strictly positive (the
    /// `max|W[o,:]|/127` convention's domain).
    fn scales(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| 1.0e-4 + (self.next() & 0xffff) as f32 * 1.0e-7)
            .collect()
    }
}

/// Replica of `decoder::quantize_row_i8` (private): dynamic per-row symmetric int8,
/// `a_scale = max|x|/127`, round-half-to-even via `f32::round`, clamp to `[-127,
/// 127]`. Byte-for-byte the activation quantize the decode GEMV performs.
fn quantize_row_i8(x: &[f32]) -> (Vec<i8>, f32) {
    let amax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let a_scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let inv = 1.0 / a_scale;
    let xq: Vec<i8> = x
        .iter()
        .map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (xq, a_scale)
}

/// Monolithic head exactly as `decoder::gemv_i8`: quantize the row ONCE, one
/// `igemm_s8s8` over all `n` columns, dequant `acc·a_scale·w_scale[o]` in the SAME
/// left-associative order.
fn gemv_mono(x: &[f32], w: &[i8], scales: &[f32], n: usize, k: usize) -> Vec<f32> {
    let (xq, a_scale) = quantize_row_i8(x);
    let mut acc = vec![0i32; n];
    igemm_s8s8(&xq, w, 1, k, n, &mut acc);
    (0..n)
        .map(|o| acc[o] as f32 * a_scale * scales[o])
        .collect()
}

/// Replica of `decoder::vocab_tile_ranges` (private): partition `[0, n)` into
/// `tiles` CONTIGUOUS, gap-free, ascending ranges (remainder spread one-per-tile
/// over the leading tiles); `tiles` clamped to `[1, n.max(1)]`.
fn vocab_tile_ranges(n: usize, tiles: usize) -> Vec<(usize, usize)> {
    let tiles = tiles.clamp(1, n.max(1));
    let base = n / tiles;
    let rem = n % tiles;
    let mut ranges = Vec::with_capacity(tiles);
    let mut start = 0usize;
    for t in 0..tiles {
        let len = base + usize::from(t < rem);
        let end = start + len;
        ranges.push((start, end));
        start = end;
    }
    ranges
}

/// Vocab-tiled head exactly as `decoder::gemv_i8_sharded`: quantize the row ONCE,
/// then for each CONTIGUOUS vocab tile run one `igemm_s8s8` over that column span
/// and write the dequantized slice into its absolute `y[start..end]` columns.
fn gemv_sharded(x: &[f32], w: &[i8], scales: &[f32], n: usize, k: usize, tiles: usize) -> Vec<f32> {
    let (xq, a_scale) = quantize_row_i8(x);
    let mut y = vec![0.0f32; n];
    for (start, end) in vocab_tile_ranges(n, tiles) {
        let cnt = end - start;
        if cnt == 0 {
            continue;
        }
        let mut acc = vec![0i32; cnt];
        igemm_s8s8(&xq, &w[start * k..end * k], 1, k, cnt, &mut acc);
        for (j, &a) in acc.iter().enumerate() {
            y[start + j] = a as f32 * a_scale * scales[start + j];
        }
    }
    y
}

/// True i64 dot-product oracle (`Σ_i a[i]·b[o,i]`) — absolute correctness for the
/// int8 contraction, independent of any tiling.
fn oracle_i64(xq: &[i8], w: &[i8], n: usize, k: usize) -> Vec<i64> {
    (0..n)
        .map(|o| {
            (0..k)
                .map(|i| i64::from(xq[i]) * i64::from(w[o * k + i]))
                .sum::<i64>()
        })
        .collect()
}

fn bits(s: &[f32]) -> Vec<u32> {
    s.iter().map(|f| f.to_bits()).collect()
}

/// Vocab-tiled head == monolithic head, BYTE-FOR-BYTE, at the REAL `[129280, 1280]`
/// head shape — for several contiguous-tile counts INCLUDING ones that do NOT
/// evenly divide 129280 (`129280 = 2^8·5·101`, so 7, 11 and 1000 each leave a
/// remainder). The bit-exact equality across non-dividing tile counts is the whole
/// lossless-sharding claim: tiling repartitions the column ranges fed to the
/// kernel, never a per-logit reduction or the vocab order.
#[test]
fn lmhead_shard_is_byte_identical_at_real_vocab_shape() {
    let mut rng = Rng(0x1a2b_3c4d_5e6f_7081);
    let w = rng.fill_i8(VOCAB * HIDDEN); // [vocab, hidden] int8 weight panel.
    let scales = rng.scales(VOCAB);
    let x = rng.fill_f32(HIDDEN); // the final-norm decode hidden row.

    let monolithic = gemv_mono(&x, &w, &scales, VOCAB, HIDDEN);
    assert_eq!(monolithic.len(), VOCAB);

    // 7, 11, 1000 do NOT divide 129280 (= 2^8·5·101); 1 is the trivial single
    // tile. (Even-divisor and extreme one-col/tile counts are swept cheaply on the
    // small synthetic vocab below.)
    for &tiles in &[1usize, 7, 11, 1000] {
        let sharded = gemv_sharded(&x, &w, &scales, VOCAB, HIDDEN, tiles);
        assert_eq!(
            bits(&monolithic),
            bits(&sharded),
            "lm_head: {tiles}-tile vocab shard != monolithic head (vocab={VOCAB} hidden={HIDDEN})"
        );
    }
}

/// Same vocab-tiling bit-identity on a SMALL synthetic vocab whose size is not a
/// multiple of the 64-row kernel block (so tile and chunk boundaries straddle),
/// plus an absolute i64-oracle check that the underlying int8 dot is computed
/// correctly and fits i32 (Doctrine-#6 flavor). Sweeps the extreme tile counts the
/// real-shape test is too wide to: one-column-per-tile and a count exceeding the
/// vocab (which must clamp, not panic or reorder).
#[test]
fn lmhead_shard_is_byte_identical_small_synthetic_vocab() {
    let (n, k) = (150usize, 96usize); // 150 is not a multiple of 64.
    let mut rng = Rng(0x0f1e_2d3c_4b5a_6978);
    let w = rng.fill_i8(n * k);
    let scales = rng.scales(n);
    let x = rng.fill_f32(k);

    let monolithic = gemv_mono(&x, &w, &scales, n, k);

    // 7 and 13 do not divide 150; 64 crosses block seams; 150 is one col/tile;
    // 200 > n exercises the clamp.
    for &tiles in &[1usize, 2, 7, 13, 64, 150, 200] {
        let sharded = gemv_sharded(&x, &w, &scales, n, k, tiles);
        assert_eq!(
            bits(&monolithic),
            bits(&sharded),
            "small lm_head: {tiles}-tile shard != monolithic head (n={n} k={k})"
        );
    }

    // Absolute correctness: the i32 accumulator equals the true i64 dot and fits.
    let (xq, _a_scale) = quantize_row_i8(&x);
    let oracle = oracle_i64(&xq, &w, n, k);
    let mut acc = vec![0i32; n];
    igemm_s8s8(&xq, &w, 1, k, n, &mut acc);
    for (o, (&got, &want)) in acc.iter().zip(oracle.iter()).enumerate() {
        assert!(
            i32::try_from(want).is_ok(),
            "oracle cell {o} = {want} overflows i32"
        );
        assert_eq!(
            i64::from(got),
            want,
            "kernel cell {o} = {got} != i64-oracle {want}"
        );
    }
}

/// SELF-TEST (sensitivity): a deliberately wrong tile scatter — writing a tile's
/// logits into the WRONG (swapped) column span — MUST diverge from the monolithic
/// head. Proves the positive gate above would CATCH a vocab-reorder / mis-scatter
/// bug (the bd-1waa silent-drift failure class) rather than passing vacuously.
#[test]
fn perturbed_tile_scatter_is_caught() {
    let (n, k) = (150usize, 96usize);
    let mut rng = Rng(0x7766_5544_3322_1100);
    let w = rng.fill_i8(n * k);
    let scales = rng.scales(n);
    let x = rng.fill_f32(k);

    let monolithic = gemv_mono(&x, &w, &scales, n, k);

    // Correctly sharded (3 tiles), then swap the first two tiles' output spans —
    // a classic mis-scatter that REORDERS the vocab. The tiles hold distinct
    // logits (distinct columns), so the equality gate MUST now fail.
    let correct = gemv_sharded(&x, &w, &scales, n, k, 3);
    assert_eq!(
        bits(&correct),
        bits(&monolithic),
        "baseline shard must match"
    );

    let ranges = vocab_tile_ranges(n, 3);
    let (s0, e0) = ranges[0];
    let (s1, e1) = ranges[1];
    assert_eq!(e0 - s0, e1 - s1, "swap needs equal-width leading tiles");
    let mut perturbed = correct.clone();
    for j in 0..(e0 - s0) {
        perturbed.swap(s0 + j, s1 + j);
    }
    assert_ne!(
        bits(&perturbed),
        bits(&monolithic),
        "tile-swapped (vocab-reordered) head still matched the monolithic head — the gate is NOT sensitive to a mis-scatter"
    );
}

/// The vocab-shard kill-switch is DEFAULT-OFF and the tile count defaults to a
/// positive value when the environment is unset — an additive, default-OFF lever
/// (Doctrine #3): with `FOCR_LMHEAD_SHARD` unset, the head stays the exact
/// monolithic path. Guarded behind the env so a host that exports the flag does
/// not flake.
#[test]
fn lmhead_shard_kill_switch_defaults_off() {
    if std::env::var_os("FOCR_LMHEAD_SHARD").is_none() {
        assert!(
            !decoder::lmhead_shard_enabled(),
            "FOCR_LMHEAD_SHARD must default OFF"
        );
    }
    assert!(
        decoder::lmhead_shard_tiles() >= 1,
        "vocab-tile count must always be >= 1"
    );
}
