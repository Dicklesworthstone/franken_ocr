//! Doctrine #6 PROOF OBLIGATION — int8 i32-accumulation overflow safety.
//!
//! From `AGENTS.md` doctrine #6 (and plan §5.4):
//!
//! > **int8 i32-accumulation overflow is a proof obligation, not an
//! > assumption.** The real worst case here is the dense layer-0 `down_proj`
//! > at **K = 6848** (U8S8 ≤ ~221.7M); it fits i32 but must be proven by a
//! > unit test at worst-case K on every arch. Do NOT inherit frankensearch's
//! > `k≤1536` bound.
//!
//! This file is the *test*, not a bench. It is **self-contained on purpose**:
//! it carries its own reference scalar int8 matmul with an explicit `i32`
//! accumulator and an independent `i64` oracle, so the proof holds even before
//! any real int8/int4 GEMM kernel exists in the tree (those land in Phase 2-4).
//! It references **no** crate internals — `i32` is `i32` regardless of which
//! kernel module eventually fills the slot.
//!
//! ## Why two accumulation schemes
//!
//! ONNX `MatMulInteger` / `DynamicQuantizeLinear` (which our
//! `linear_int8_dynamic` mirrors) and the SDOT/VNNI/SMMLA intrinsics each have
//! a defined operand domain:
//!
//! * **S8S8** — both operands signed int8 in `[-128, 127]`. The element product
//!   whose *magnitude* is largest is `(-128) * (-128) = +16384`, but the
//!   adversary that maximizes a single-sign running sum (no cancellation) uses
//!   `127 * 127 = 16129` on every term, because `-128` cannot be reached on
//!   both operands simultaneously without sign flips that would cancel. The
//!   strict worst case for a monotone accumulator is therefore
//!   `K * 127 * 127`. (We *also* assert the `-128 * -128` all-positive variant
//!   below, which is even larger per-term but still bounded — and we show it
//!   too fits, so the proof is conservative either way.)
//! * **U8S8** — activations are unsigned `u8` in `[0, 255]` (the
//!   `DynamicQuantizeLinear` asymmetric activation path) times signed `i8`
//!   weights in `[-128, 127]`. The worst monotone sum is `K * 255 * 127`.
//!
//! ## Arch-generality
//!
//! The proof is **arch-independent**: the accumulator is a 32-bit two's
//! complement integer on every target, so the bound `value < i32::MAX` is the
//! same for scalar, NEON SDOT, AVX-VNNI, and SMMLA/i8mm. BUT each SIMD kernel,
//! *when it lands*, must re-run this proof against **its own** accumulator
//! width and lane-reduction order (e.g. SMMLA accumulates 8 int8 MACs per
//! 32-bit lane before any horizontal add; VNNI does 4). See the clearly
//! labeled `arch_kernel_reproof_slots` test below for the scaffolded slots.

// ── Worst-case-K table for the *real* model ─────────────────────────────────
//
// These are the contraction lengths `K` that actually occur in the quantized
// decoder GEMMs. 6848 is the dense layer-0 `down_proj` and is the global worst
// case. The frankensearch prior art bounded K at 1536 and would have *rejected*
// 6848 — we explicitly do NOT inherit that bound (see
// `down_proj_6848_specifically_breaks_frankensearch_1536_bound`).
//
// `note` documents where each K comes from so a future reader can map a failing
// row back to a model tensor.
struct KCase {
    k: usize,
    note: &'static str,
}

const MODEL_KS: &[KCase] = &[
    KCase {
        k: 896,
        note: "narrow expert / projector-adjacent GEMM",
    },
    KCase {
        k: 1280,
        note: "decoder hidden size (q/k/v/o, gate/up at hidden width)",
    },
    KCase {
        k: 2048,
        note: "projector input width (2048 -> 1280)",
    },
    KCase {
        k: 4096,
        note: "wide intermediate contraction",
    },
    KCase {
        k: 6848,
        note: "DENSE LAYER-0 down_proj — GLOBAL WORST CASE (doctrine #6)",
    },
];

/// The frankensearch bound we must NOT inherit. Kept as a named constant so the
/// assertion that 6848 exceeds it is self-documenting.
const FRANKENSEARCH_K_BOUND: usize = 1536;

// ── Per-element operand extremes ────────────────────────────────────────────

/// Largest magnitude of `s8 * s8` for a *monotone* (no-cancellation) sum:
/// `127 * 127`. (`-128 * -128 = 16384` is larger per term and is checked
/// separately as an even-more-conservative variant.)
const S8S8_TERM: i64 = 127 * 127; // 16_129
/// Largest `u8 * s8` term for a monotone sum: `255 * 127`.
const U8S8_TERM: i64 = 255 * 127; // 32_385
/// The most extreme S8S8 per-term magnitude reachable at all: `(-128)*(-128)`.
const S8S8_TERM_NEG128: i64 = 128 * 128; // 16_384

/// i32 ceiling as i64, for headroom math without wraparound.
const I32_MAX_I64: i64 = i32::MAX as i64; // 2_147_483_647

// ── Self-contained reference scalar int8 matmul (the thing under proof) ──────
//
// A single-output-channel dot product is sufficient to exercise the
// accumulator: a `[1, K] x [K, 1]` GEMM IS the inner accumulation loop that
// every real kernel (scalar, SDOT, VNNI, SMMLA) must perform. We deliberately
// keep it to one accumulator so the test's `i32` value is *exactly* the kernel's
// per-output-channel accumulator, with nothing hidden.

/// Reference S8S8 dot product accumulated in **i32** — the value under proof.
/// Returns the `i32` accumulator (wrapping is *visible* because we compare it to
/// the i64 oracle; we do NOT use wrapping_add — a real overflow would panic in
/// debug and silently wrap in release, and the oracle comparison catches both).
fn dot_s8s8_i32(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    let mut acc: i32 = 0;
    for (&x, &w) in a.iter().zip(b.iter()) {
        // i32(i8) * i32(i8) is at most 16384 in magnitude; the running sum is
        // the quantity whose non-overflow we are proving.
        acc += i32::from(x) * i32::from(w);
    }
    acc
}

/// Reference U8S8 dot product accumulated in **i32** (unsigned activation,
/// signed weight) — mirrors `DynamicQuantizeLinear` + `MatMulInteger`.
fn dot_u8s8_i32(a: &[u8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    let mut acc: i32 = 0;
    for (&x, &w) in a.iter().zip(b.iter()) {
        acc += i32::from(x) * i32::from(w);
    }
    acc
}

/// Independent **i64** oracle for S8S8 — cannot overflow at any model K (i64
/// holds K up to ~5.7e14 terms). Equality with the i32 path == no i32 wrap.
fn dot_s8s8_i64(a: &[i8], b: &[i8]) -> i64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &w)| i64::from(x) * i64::from(w))
        .sum()
}

/// Independent **i64** oracle for U8S8.
fn dot_u8s8_i64(a: &[u8], b: &[i8]) -> i64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &w)| i64::from(x) * i64::from(w))
        .sum()
}

// ── Headroom logging helper ─────────────────────────────────────────────────

fn log_row(scheme: &str, k: usize, note: &str, worst: i64) {
    let headroom = I32_MAX_I64 - worst;
    let pct_used = (worst as f64) / (I32_MAX_I64 as f64) * 100.0;
    let verdict = if worst < I32_MAX_I64 { "PASS" } else { "FAIL" };
    println!(
        "[{verdict}] {scheme:<5} K={k:<6} worst_acc={worst:>13} \
         i32_headroom={headroom:>13} ({pct_used:6.3}% of i32::MAX used)  // {note}"
    );
}

// ── PROOF 1: exact closed-form worst-case values fit i32 ─────────────────────

/// The headline doctrine-#6 numbers, computed exactly (not transcribed) and
/// asserted under `i32::MAX`. Also prints the full per-K headroom table.
#[test]
fn worst_case_accumulators_fit_i32_closed_form() {
    println!("\n=== int32 overflow proof — closed-form worst-case accumulators ===");
    println!("i32::MAX = {}", i32::MAX);
    println!(
        "S8S8 per-term (monotone) = {S8S8_TERM} (127*127);  \
         U8S8 per-term = {U8S8_TERM} (255*127);  \
         S8S8 (-128*-128) = {S8S8_TERM_NEG128}"
    );

    // The two headline numbers at the global worst-case K = 6848.
    let s8s8_6848 = 6848i64 * S8S8_TERM;
    let u8s8_6848 = 6848i64 * U8S8_TERM;
    assert_eq!(
        s8s8_6848, 110_451_392,
        "S8S8 worst at K=6848 must be exactly 6848*16129"
    );
    assert_eq!(
        u8s8_6848, 221_772_480,
        "U8S8 worst at K=6848 must be exactly 6848*32385"
    );
    assert!(s8s8_6848 < I32_MAX_I64, "S8S8 @6848 overflows i32!");
    assert!(u8s8_6848 < I32_MAX_I64, "U8S8 @6848 overflows i32!");
    // Even the most extreme per-term S8S8 (-128*-128) all-positive fits.
    let s8s8_6848_neg128 = 6848i64 * S8S8_TERM_NEG128;
    assert_eq!(s8s8_6848_neg128, 112_197_632);
    assert!(
        s8s8_6848_neg128 < I32_MAX_I64,
        "S8S8(-128) @6848 overflows i32!"
    );

    println!("\n-- full per-K headroom table --");
    for case in MODEL_KS {
        let s = case.k as i64 * S8S8_TERM;
        let u = case.k as i64 * U8S8_TERM;
        log_row("S8S8", case.k, case.note, s);
        log_row("U8S8", case.k, case.note, u);
        assert!(s < I32_MAX_I64, "S8S8 K={} overflows i32", case.k);
        assert!(u < I32_MAX_I64, "U8S8 K={} overflows i32", case.k);
    }

    // The U8S8 path is the binding constraint (largest per-term). At the worst
    // model K it still uses only ~10.3% of i32::MAX — ample headroom, and the
    // reason int16 accumulation would NOT be safe (255*127 already needs > 15
    // bits at K=2 with sign).
    let pct = (u8s8_6848 as f64) / (I32_MAX_I64 as f64) * 100.0;
    println!(
        "\nU8S8 @ worst-case K=6848 uses {pct:.3}% of i32::MAX (headroom {} of {})",
        I32_MAX_I64 - u8s8_6848,
        I32_MAX_I64
    );
    assert!(pct < 100.0);
}

// ── PROOF 2: the live scalar kernel does not wrap at every model K ───────────

/// Feed each model K's adversarial all-extreme operands through the live i32
/// accumulator and assert bit-exact equality with the i64 oracle. Equality is
/// the no-wrap proof: if the i32 path overflowed (panic in debug, wrap in
/// release) the values would diverge.
#[test]
fn live_i32_accumulator_matches_i64_oracle_at_every_model_k() {
    println!("\n=== int32 overflow proof — live kernel vs i64 oracle ===");
    for case in MODEL_KS {
        let k = case.k;

        // S8S8 adversary: all +127 (monotone-max running sum).
        let a127 = vec![127i8; k];
        let b127 = vec![127i8; k];
        let acc_i32 = dot_s8s8_i32(&a127, &b127);
        let acc_i64 = dot_s8s8_i64(&a127, &b127);
        assert_eq!(
            i64::from(acc_i32),
            acc_i64,
            "S8S8(+127) i32 wrapped at K={k}"
        );
        assert_eq!(acc_i64, k as i64 * S8S8_TERM);
        log_row("S8S8", k, "all +127", acc_i64);

        // S8S8 most-extreme per-term adversary: all -128 (product +16384 each).
        let am128 = vec![-128i8; k];
        let bm128 = vec![-128i8; k];
        let acc_i32_n = dot_s8s8_i32(&am128, &bm128);
        let acc_i64_n = dot_s8s8_i64(&am128, &bm128);
        assert_eq!(
            i64::from(acc_i32_n),
            acc_i64_n,
            "S8S8(-128) i32 wrapped at K={k}"
        );
        assert_eq!(acc_i64_n, k as i64 * S8S8_TERM_NEG128);
        log_row("S8S8", k, "all -128 (max |product|)", acc_i64_n);

        // U8S8 adversary: all 255 activations * all 127 weights (monotone-max).
        let au = vec![255u8; k];
        let bu = vec![127i8; k];
        let acc_i32_u = dot_u8s8_i32(&au, &bu);
        let acc_i64_u = dot_u8s8_i64(&au, &bu);
        assert_eq!(i64::from(acc_i32_u), acc_i64_u, "U8S8 i32 wrapped at K={k}");
        assert_eq!(acc_i64_u, k as i64 * U8S8_TERM);
        log_row("U8S8", k, "255 * 127 (binding worst case)", acc_i64_u);

        // U8S8 most-negative adversary: all 255 * all -128 (largest |negative|).
        let bn = vec![-128i8; k];
        let acc_i32_un = dot_u8s8_i32(&au, &bn);
        let acc_i64_un = dot_u8s8_i64(&au, &bn);
        assert_eq!(
            i64::from(acc_i32_un),
            acc_i64_un,
            "U8S8(neg) i32 wrapped at K={k}"
        );
        assert_eq!(acc_i64_un, k as i64 * 255 * -128);
        // Negative headroom is against i32::MIN.
        assert!(
            acc_i64_un > i32::MIN as i64,
            "U8S8(neg) underflows i32 at K={k}"
        );
        println!(
            "[PASS] U8S8  K={k:<6} worst_neg_acc={acc_i64_un:>14} \
             i32::MIN headroom={:>14}  // 255 * -128",
            acc_i64_un - i32::MIN as i64
        );
    }
}

// ── PROOF 3: 6848 specifically defeats the inherited frankensearch bound ─────

/// Doctrine #6's explicit instruction: prove K=6848 is safe AND prove we did
/// not silently inherit frankensearch's `k <= 1536` cap (which would have
/// rejected the real model).
#[test]
fn down_proj_6848_specifically_breaks_frankensearch_1536_bound() {
    println!("\n=== int32 overflow proof — 6848 vs frankensearch k<=1536 ===");
    const WORST_K: usize = 6848;

    // We are *beyond* the inherited bound — that is the whole point.
    // K=6848 must exceed the frankensearch bound (1536) to be a real test.
    const _: () = assert!(WORST_K > FRANKENSEARCH_K_BOUND);
    println!(
        "WORST_K=6848 is {}x the frankensearch bound {FRANKENSEARCH_K_BOUND}; \
         we DO NOT inherit it.",
        WORST_K / FRANKENSEARCH_K_BOUND
    );

    // The bound the inherited code would have used to "prove" safety only goes
    // up to 1536. Show that 6848 still fits i32 under the binding U8S8 scheme —
    // the inherited reasoning was *unnecessarily* conservative, not wrong about
    // overflow, but it would have falsely *rejected* this model's GEMM.
    let au = vec![255u8; WORST_K];
    let bu = vec![127i8; WORST_K];
    let acc_i32 = dot_u8s8_i32(&au, &bu);
    let acc_i64 = dot_u8s8_i64(&au, &bu);
    assert_eq!(
        i64::from(acc_i32),
        acc_i64,
        "U8S8 i32 wrapped at the worst-case K=6848"
    );
    assert_eq!(acc_i64, 221_772_480);
    assert!(acc_i64 < I32_MAX_I64);
    log_row(
        "U8S8",
        WORST_K,
        "down_proj — proven safe past frankensearch",
        acc_i64,
    );

    // Same for S8S8 at 6848.
    let a127 = vec![127i8; WORST_K];
    let b127 = vec![127i8; WORST_K];
    let s_i32 = dot_s8s8_i32(&a127, &b127);
    assert_eq!(i64::from(s_i32), 110_451_392);
    const _: () = assert!(110_451_392 < I32_MAX_I64);
    log_row(
        "S8S8",
        WORST_K,
        "down_proj — proven safe past frankensearch",
        110_451_392,
    );
}

// ── PROOF 4: the actual i32 boundary — where overflow WOULD begin ────────────

/// Find and assert the exact K at the i32 boundary for the binding U8S8 scheme,
/// documenting how far the real model (6848) sits below it. This pins down the
/// true safety margin so a future kernel author knows the cliff edge.
#[test]
fn stress_k_at_the_i32_boundary() {
    println!("\n=== int32 overflow proof — boundary stress K ===");

    // Largest K with U8S8 monotone sum strictly under i32::MAX.
    let max_safe_k_u8s8 = (I32_MAX_I64 / U8S8_TERM) as usize; // 66311
    let at_boundary = max_safe_k_u8s8 as i64 * U8S8_TERM;
    let one_past = (max_safe_k_u8s8 as i64 + 1) * U8S8_TERM;
    assert_eq!(max_safe_k_u8s8, 66_311);
    assert!(at_boundary < I32_MAX_I64, "boundary K must still fit");
    assert!(
        one_past > I32_MAX_I64,
        "K just past boundary must overflow i32"
    );
    println!(
        "[PASS] U8S8 boundary: max safe K = {max_safe_k_u8s8} (acc={at_boundary}, \
         headroom={}); K+1={} overflows (acc={one_past} > {})",
        I32_MAX_I64 - at_boundary,
        max_safe_k_u8s8 + 1,
        i32::MAX
    );

    // Same for S8S8 (monotone 127*127).
    let max_safe_k_s8s8 = (I32_MAX_I64 / S8S8_TERM) as usize; // 133144
    assert_eq!(max_safe_k_s8s8, 133_144);
    assert!((max_safe_k_s8s8 as i64) * S8S8_TERM < I32_MAX_I64);
    assert!((max_safe_k_s8s8 as i64 + 1) * S8S8_TERM > I32_MAX_I64);
    println!("[PASS] S8S8 boundary: max safe K = {max_safe_k_s8s8}");

    // The real model's worst K is ~9.7x below the U8S8 cliff. Document the
    // margin explicitly: a future kernel could 9x the contraction and still be
    // safe under U8S8 — but anything wider needs i64 accumulation or tiling.
    let margin_factor = max_safe_k_u8s8 as f64 / 6848.0;
    println!(
        "Real worst-case K=6848 sits {margin_factor:.2}x below the U8S8 i32 cliff \
         (cliff K={max_safe_k_u8s8})."
    );
    assert!(
        margin_factor > 9.0,
        "expected >9x headroom below the i32 cliff"
    );

    // Run the live kernel exactly AT the boundary to prove the i32 path itself
    // (not just the closed form) survives the largest safe contraction.
    let au = vec![255u8; max_safe_k_u8s8];
    let bu = vec![127i8; max_safe_k_u8s8];
    let acc_i32 = dot_u8s8_i32(&au, &bu);
    let acc_i64 = dot_u8s8_i64(&au, &bu);
    assert_eq!(
        i64::from(acc_i32),
        acc_i64,
        "i32 path wrapped AT the boundary K"
    );
    assert_eq!(acc_i64, at_boundary);
    println!("[PASS] live i32 kernel survives boundary K={max_safe_k_u8s8} (acc={acc_i32})");
}

// ── PROOF 5: a multi-output-channel GEMM does not change the per-channel bound ─

/// A real GEMM has many output channels, but each channel has its *own* i32
/// accumulator — the bound is per-channel and independent of N (number of output
/// channels) and M (number of rows). This test runs a small `[M, K] x [K, N]`
/// with worst-case operands and asserts every output cell equals the i64 oracle,
/// proving the bound is not an artifact of the 1-channel reduction in proofs 2-4.
#[test]
fn multi_channel_gemm_each_accumulator_independent() {
    println!("\n=== int32 overflow proof — multi-channel GEMM (per-channel acc) ===");
    let m = 3usize;
    let k = 6848usize; // worst-case contraction
    let n = 4usize;

    // U8S8: activations all 255, weights all 127 (column-major [k, n] here).
    let a = vec![255u8; m * k];
    let w = vec![127i8; k * n];

    for r in 0..m {
        let arow = &a[r * k..(r + 1) * k];
        for c in 0..n {
            // gather column c of w ([k, n] row-major => stride n).
            let wcol: Vec<i8> = (0..k).map(|t| w[t * n + c]).collect();
            let acc_i32 = dot_u8s8_i32(arow, &wcol);
            let acc_i64 = dot_u8s8_i64(arow, &wcol);
            assert_eq!(i64::from(acc_i32), acc_i64, "cell ({r},{c}) i32 wrapped");
            assert_eq!(acc_i64, 221_772_480);
        }
    }
    println!(
        "[PASS] {m}x{n} output cells at K={k}, each i32 accumulator = 221,772,480 \
         (independent of M,N)"
    );
}

// ── ARCH RE-PROOF SLOTS (scaffolded, NOT silently empty) ────────────────────
//
// Doctrine #4/#6: every SIMD int8 kernel must re-run this proof against its own
// accumulator width and lane-reduction order when it lands. The slots below are
// deliberately present and *passing on the arch-independent invariant* so they
// show up in the test list as named obligations — never silently empty. When a
// kernel lands, replace the body's "drive the scalar reference" with "drive the
// real intrinsic kernel" and keep the i64-oracle equality assertion.

/// SDOT / FEAT_DotProd (ARM64): 4 int8 MACs per 32-bit lane via `sdot`.
/// Accumulator is i32 per lane — identical bound. SLOT: re-run with the real
/// `sdot`-based dot product once `linear_int8_dynamic`'s ARM path lands.
#[test]
fn arch_reproof_slot_sdot_neon() {
    // Invariant that the future sdot kernel must satisfy (proven here via the
    // arch-independent reference; the real kernel must reproduce this exactly).
    let k = 6848;
    let acc = dot_s8s8_i64(&vec![127i8; k], &vec![127i8; k]);
    assert_eq!(acc, 110_451_392);
    assert!(acc < I32_MAX_I64);
    println!(
        "[SLOT/SDOT] arch-independent invariant holds (acc={acc}); \
         WIRE THE REAL sdot KERNEL HERE when ARM int8 GEMM lands."
    );
}

/// AVX-VNNI / AVX-512-VNNI (x86-64): `vpdpbusd` does U8S8, 4 MACs per i32 lane.
/// SLOT: re-run with the real VNNI dot product (U8S8 is its native mode — the
/// binding worst case) once the x86 int8 path lands.
#[test]
fn arch_reproof_slot_vnni_x86() {
    let k = 6848;
    let acc = dot_u8s8_i64(&vec![255u8; k], &vec![127i8; k]);
    assert_eq!(acc, 221_772_480);
    assert!(acc < I32_MAX_I64);
    println!(
        "[SLOT/VNNI] arch-independent U8S8 invariant holds (acc={acc}); \
         WIRE THE REAL vpdpbusd KERNEL HERE when x86 int8 GEMM lands."
    );
}

/// SMMLA / FEAT_MATMUL_INT8 (ARM64 i8mm): 8 int8 MACs per 32-bit lane,
/// 2x2 tile. Doctrine #4 notes un-blocked SMMLA was load-bound/slower than
/// SDOT — when the *register-blocked* SMMLA GEMM lands it must re-run this
/// proof (its 8-deep per-lane reduction still terminates in an i32 lane).
/// SLOT: wire the real smmla tile dot here.
#[test]
fn arch_reproof_slot_smmla_i8mm() {
    let k = 6848;
    // SMMLA is S8S8; assert the S8S8 worst case the tiled kernel must honor.
    let acc = dot_s8s8_i64(&vec![127i8; k], &vec![127i8; k]);
    assert_eq!(acc, 110_451_392);
    assert!(acc < I32_MAX_I64);
    // The most extreme per-term variant the tile could see.
    let acc_neg = dot_s8s8_i64(&vec![-128i8; k], &vec![-128i8; k]);
    assert_eq!(acc_neg, 112_197_632);
    assert!(acc_neg < I32_MAX_I64);
    println!(
        "[SLOT/SMMLA] arch-independent S8S8 invariants hold (acc={acc}, neg={acc_neg}); \
         WIRE THE REAL register-blocked smmla TILE HERE when i8mm GEMM lands."
    );
}
