//! Runtime ISA dispatch + the public int8 GEMM entrypoint (plan §6.2 / §6.6,
//! bd-2mo.1/.2).
//!
//! This is the **single entrypoint** the rest of the engine calls
//! ([`igemm_s8s8`] / [`igemm_u8s8`]); it picks the best available int8 kernel at
//! RUNTIME and falls back to the [`scalar`] oracle. Selection is:
//!
//! * **x86-64:** `AVX-512-VNNI > AVX-VNNI > AVX2 > scalar`
//! * **aarch64 (Apple Silicon / macOS):** `SDOT (dotprod) > SMMLA (i8mm) > scalar`
//!   — i8mm issues at half-rate on every M-series core, so SDOT is the faster
//!   int8 kernel (see [`arm::detect_tier`](crate::simd::arm::detect_tier)).
//! * **aarch64 (other, e.g. Neoverse):** `SMMLA (i8mm) > SDOT (dotprod) > scalar`
//! * **everything else:** `scalar`
//!
//! `FOCR_FORCE_ARCH=<tag>` (`sdot`/`smmla`/`scalar`/`avx2`/…) overrides the
//! selection for benchmarking/debugging when the named tier is available,
//! without changing the best-first capability list. Read once per process.
//!
//! (An AMX tier is not advertised: the `x86.rs` backend implements no AMX
//! kernel, so it is not advertised as an available tier. The variant is added
//! back here when the backend grows one.)
//!
//! The chosen tier is detected **once** (cached in a [`OnceLock`]) via the
//! standard-library feature-detection macros (`is_aarch64_feature_detected!` /
//! `is_x86_feature_detected!`) so the per-call cost is a single relaxed atomic
//! load. The dispatch itself contains **no `unsafe`** — it routes by
//! `target_arch` to the per-arch backend (`arm.rs` / `x86.rs`), each of which
//! owns its own audited `unsafe` island, performs the *same* runtime feature
//! detection internally to pick its sub-tier, and falls back to the
//! bit-identical [`scalar`] oracle when no accelerated tier is present. So
//! correctness never depends on capability reporting. `robot backends` exposes
//! both that hardware tier and the effective ordinary dense-GEMM route; on Apple
//! Silicon the latter can be LLVM autovec even when the former is SDOT.
//!
//! `focr robot backends` reflects [`detected_tier`] / [`available_tiers`] /
//! [`effective_dense_route`] (bd-2mo.2 / bd-2mo.30.10).

use std::sync::OnceLock;

/// The dispatched int8-GEMM ISA tier (plan §6.6). Ordered by descending
/// throughput within an arch; the [`Ord`] derive ranks them so `max()` over the
/// available set picks the best (the variant order below IS the ranking).
///
/// Cross-arch variants coexist in one enum so a single `OnceLock<IsaTier>` and a
/// single `robot backends` surface describe every host; only the variants
/// reachable on the current arch are ever selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IsaTier {
    /// Portable scalar oracle — the floor, present on every target.
    Scalar = 0,
    /// x86-64 AVX2 — the `x86.rs` backend uses the non-saturating `vpmaddwd`
    /// (i16→i32, exact) path, NOT the saturating `vpmaddubsw` (doctrine-safe).
    Avx2 = 1,
    /// x86-64 AVX-VNNI (`vpdpbusd`, U8S8 native, 4 MACs/i32 lane).
    AvxVnni = 2,
    /// x86-64 AVX-512-VNNI (`vpdpbusd` on 512-bit lanes).
    Avx512Vnni = 3,
    /// aarch64 FEAT_DotProd SDOT (4 int8 MACs/i32 lane).
    Sdot = 4,
    /// aarch64 FEAT_MATMUL_INT8 SMMLA / i8mm (8 int8 MACs/i32 lane, 2x2 tile) —
    /// the register-blocked wedge (doctrine #4).
    Smmla = 5,
}

impl IsaTier {
    /// A stable, lowercase feature string for the dispatched tier — the value
    /// `focr robot backends`, `PERF_LEDGER.md`, and `DISCREPANCIES.md` record
    /// (e.g. `aarch64+neon+dotprod`, `aarch64+neon+i8mm`,
    /// `x86_64+avx512vnni`, `scalar`). This is the **dispatched** tier, not the
    /// host's maximum capability.
    #[must_use]
    pub fn feature_string(self) -> &'static str {
        match self {
            IsaTier::Scalar => "scalar",
            IsaTier::Avx2 => "x86_64+avx2",
            IsaTier::AvxVnni => "x86_64+avx2+avxvnni",
            IsaTier::Avx512Vnni => "x86_64+avx512vnni",
            IsaTier::Sdot => "aarch64+neon+dotprod",
            IsaTier::Smmla => "aarch64+neon+i8mm",
        }
    }

    /// A short tier tag (`"scalar"`, `"sdot"`, `"smmla"`, `"avx2"`,
    /// `"avxvnni"`, `"avx512vnni"`) for compact JSON / logs.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            IsaTier::Scalar => "scalar",
            IsaTier::Avx2 => "avx2",
            IsaTier::AvxVnni => "avxvnni",
            IsaTier::Avx512Vnni => "avx512vnni",
            IsaTier::Sdot => "sdot",
            IsaTier::Smmla => "smmla",
        }
    }
}

/// The implementation actually executed by ordinary dense int8 GEMM. Hardware
/// capability remains [`IsaTier`]; this route additionally names the measured
/// Apple LLVM-autovec path that intentionally bypasses available SDOT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EffectiveI8Route {
    Autovec,
    Scalar,
    Avx2,
    AvxVnni,
    Avx512Vnni,
    Sdot,
    Smmla,
}

impl EffectiveI8Route {
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            Self::Autovec => "autovec",
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::AvxVnni => "avxvnni",
            Self::Avx512Vnni => "avx512vnni",
            Self::Sdot => "sdot",
            Self::Smmla => "smmla",
        }
    }

    #[must_use]
    pub fn feature_string(self) -> &'static str {
        match self {
            Self::Autovec => "aarch64+llvm-autovec",
            Self::Scalar => "scalar",
            Self::Avx2 => IsaTier::Avx2.feature_string(),
            Self::AvxVnni => IsaTier::AvxVnni.feature_string(),
            Self::Avx512Vnni => IsaTier::Avx512Vnni.feature_string(),
            Self::Sdot => IsaTier::Sdot.feature_string(),
            Self::Smmla => IsaTier::Smmla.feature_string(),
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn from_isa(tier: IsaTier) -> Self {
        match tier {
            IsaTier::Scalar => Self::Scalar,
            IsaTier::Avx2 => Self::Avx2,
            IsaTier::AvxVnni => Self::AvxVnni,
            IsaTier::Avx512Vnni => Self::Avx512Vnni,
            IsaTier::Sdot => Self::Sdot,
            IsaTier::Smmla => Self::Smmla,
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn from_arm(route: super::arm::DenseI8Route) -> Self {
        match route {
            super::arm::DenseI8Route::Autovec => Self::Autovec,
            super::arm::DenseI8Route::Scalar => Self::Scalar,
            super::arm::DenseI8Route::Sdot => Self::Sdot,
            super::arm::DenseI8Route::Smmla => Self::Smmla,
        }
    }
}

/// The cached capability snapshot: the chosen hardware ISA tier plus every tier
/// this host could dispatch (for `robot backends`).
#[derive(Debug, Clone)]
pub struct Caps {
    /// The selected hardware ISA tier. See [`effective_dense_route`] for the
    /// ordinary dense-GEMM implementation selected above that capability.
    pub selected: IsaTier,
    /// All tiers detected as available on this host, best-first.
    pub available: Vec<IsaTier>,
}

static CAPS: OnceLock<Caps> = OnceLock::new();

/// Detect (once) and return the cached capability snapshot.
///
/// Feature detection runs exactly once via [`OnceLock`]; subsequent calls are a
/// cheap atomic load. Detection itself never panics (the std macros only query
/// CPUID / HWCAP). The `selected` tier is the highest-ranked `available` one.
#[must_use]
pub fn caps() -> &'static Caps {
    CAPS.get_or_init(detect)
}

/// The selected hardware ISA tier on this host.
#[must_use]
pub fn detected_tier() -> IsaTier {
    caps().selected
}

/// Every int8-GEMM tier available on this host, best-first (for `robot
/// backends`). Always contains at least [`IsaTier::Scalar`].
#[must_use]
pub fn available_tiers() -> &'static [IsaTier] {
    &caps().available
}

/// The selected hardware tier's stable feature string.
#[must_use]
pub fn tier_string() -> &'static str {
    detected_tier().feature_string()
}

/// Effective ordinary dense-int8 implementation. Unlike [`detected_tier`],
/// this reports Apple autovec when SDOT is present but deliberately bypassed.
#[must_use]
pub fn effective_dense_route() -> EffectiveI8Route {
    #[cfg(target_arch = "aarch64")]
    {
        EffectiveI8Route::from_arm(super::arm::effective_dense_route())
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        EffectiveI8Route::from_isa(detected_tier())
    }
}

/// Run the actual runtime feature detection. Builds the `available` list
/// best-first per the documented per-arch order, then selects either a valid
/// forced tier or the front (scalar is always last and always present).
fn detect() -> Caps {
    let mut available: Vec<IsaTier> = Vec::new();

    // ── aarch64 ─────────────────────────────────────────────────────────────
    // Apple Silicon (macOS/aarch64): SDOT > SMMLA. i8mm issues at half-rate on
    // every M-series core, so SMMLA's 2x MACs/instruction cancel out (measured on
    // M4: 0.994x SDOT) and it also pays a 2x2 operand repack the dot path skips —
    // so SDOT is the faster int8 kernel here. Other aarch64 (e.g. Neoverse):
    // SMMLA > SDOT, where i8mm can be full-rate. Mirrors `arm::detect_tier`.
    #[cfg(target_arch = "aarch64")]
    {
        // `is_aarch64_feature_detected!` is safe: it reads HWCAP / sysctl and is
        // the documented gate for the matching intrinsics. We only push (and
        // thus only ever select) a tier whose feature is confirmed present.
        let has_i8mm = std::arch::is_aarch64_feature_detected!("i8mm");
        let has_dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
        #[cfg(target_os = "macos")]
        {
            if has_dotprod {
                available.push(IsaTier::Sdot);
            }
            if has_i8mm {
                available.push(IsaTier::Smmla);
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            if has_i8mm {
                available.push(IsaTier::Smmla);
            }
            if has_dotprod {
                available.push(IsaTier::Sdot);
            }
        }
    }

    // ── x86-64: AVX512-VNNI > AVX-VNNI > AVX2 > scalar ──────────────────────
    //
    // This mirrors EXACTLY the sub-tiers the `x86.rs` backend actually
    // implements and selects internally (it has no AMX kernel), so the reported
    // tier never overclaims what `igemm_*` will dispatch to (doctrine #8: the
    // *dispatched* tier, not the host's max). `avx512vnni` additionally needs
    // `avx512bw`/`avx512f` for the masked-tail epilogue the backend uses.
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512f")
        {
            available.push(IsaTier::Avx512Vnni);
        }
        if std::arch::is_x86_feature_detected!("avxvnni") {
            available.push(IsaTier::AvxVnni);
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            available.push(IsaTier::Avx2);
        }
    }

    // Scalar is always available and always last (the floor).
    available.push(IsaTier::Scalar);

    // `available` remains hardware best-first regardless of overrides. A valid
    // `FOCR_FORCE_ARCH=<tag>` selects that available tier without rewriting the
    // capability order; absent, unknown, and unsupported tags are ignored.
    let selected = std::env::var("FOCR_FORCE_ARCH")
        .ok()
        .map(|force| force.trim().to_ascii_lowercase())
        .and_then(|want| available.iter().copied().find(|tier| tier.tag() == want))
        .unwrap_or(available[0]);
    Caps {
        selected,
        available,
    }
}

/// Public **int8 GEMM** entrypoint, S8S8 (signed activations · signed weights).
///
/// `C[M,N] += A[M,K] (i8, row-major) · B[N,K] (i8, output-channel-major)` into
/// the i32 buffer `out` (length `m*n`). Dispatches by architecture to the best
/// available accelerated backend, else the [`scalar`] floor; every path is
/// **bit-identical** to [`scalar::igemm_s8s8`] (i32 accumulation is exact, so
/// there is no numeric divergence between tiers — verified by each backend's
/// tests against the oracle).
///
/// The per-arch backends (`arm::igemm_s8s8`, `x86::igemm_s8s8`) own their own
/// audited `unsafe` islands and perform the *same* runtime CPU-feature detection
/// this module reflects in [`detected_tier`], selecting their sub-tier
/// (SMMLA/SDOT on ARM; AVX-512-VNNI/AVX-VNNI/AVX2 on x86) and falling back to a
/// bit-identical scalar floor internally. Routing here is therefore by
/// `target_arch` only: on a host whose accelerated tier is absent the backend
/// itself returns the scalar result, so correctness never depends on this
/// dispatcher guessing the sub-tier.
///
/// # Panics
/// As [`scalar::igemm_s8s8`] (length-contract violations).
pub fn igemm_s8s8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    let _ = igemm_s8s8_with_route(a, b, m, k, n, out);
}

fn igemm_s8s8_with_route(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [i32],
) -> EffectiveI8Route {
    #[cfg(target_arch = "aarch64")]
    {
        // ARM backend mirrors `arm::detect_tier`: SDOT > SMMLA > scalar on
        // Apple Silicon, SMMLA > SDOT > scalar on other aarch64.
        EffectiveI8Route::from_arm(super::arm::igemm_s8s8_with_route(a, b, m, k, n, out))
    }
    #[cfg(target_arch = "x86_64")]
    {
        super::x86::igemm_s8s8_with_route(a, b, m, k, n, out)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        super::scalar::igemm_s8s8(a, b, m, k, n, out);
        EffectiveI8Route::Scalar
    }
}

/// Public **int8 GEMM** entrypoint whose B operand is an OFFLINE SMMLA panel
/// stream (`focr convert --arch aarch64-smmla`, bd-2mo.3). On the SMMLA tier
/// the panels feed `vmmlaq_s32` with ZERO runtime shuffle; every other tier
/// un-permutes and runs the ordinary row-major dispatch — bit-identical
/// either way (the packing is a pure zero-padded permutation).
///
/// Unlike [`igemm_s8s8`], `out` is zeroed by this entrypoint.
///
/// # Panics
/// On length-contract violations (`a != m*k`,
/// `b_panels != ceil(n/2)*ceil(k/8)*16`, `out != m*n`).
pub fn igemm_s8s8_packed_b(
    a: &[i8],
    b_panels: &[i8],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [i32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        super::arm::igemm_s8s8_packed_b(a, b_panels, m, k, n, out);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // Portable degrade: un-permute once and run the ordinary dispatch.
        // Never the hot path — the loader only keeps panels when the SMMLA
        // tier is dispatched (aarch64-only by construction).
        let b = super::pack::smmla_unpack_panels(b_panels, n, k)
            .expect("igemm_s8s8_packed_b: panel length contract violated");
        out.fill(0);
        igemm_s8s8(a, &b, m, k, n, out);
    }
}

/// Public **int8 GEMM** entrypoint, U8S8 (unsigned activations · signed
/// weights) — the asymmetric `DynamicQuantizeLinear` activation path and the
/// native VNNI operand domain.
///
/// `C[M,N] += A[M,K] (u8, row-major) · B[N,K] (i8, output-channel-major)` into
/// the i32 buffer `out`. Dispatches as [`igemm_s8s8`]; bit-identical to
/// [`scalar::igemm_u8s8`]. The accelerated backends realize U8S8 via the +128
/// bias-correction identity (run the signed kernel on `a-128`, add
/// `128·rowsum(w)`), all in exact i32.
///
/// # Panics
/// As [`scalar::igemm_u8s8`].
pub fn igemm_u8s8(a: &[u8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    let _ = igemm_u8s8_with_route(a, b, m, k, n, out);
}

fn igemm_u8s8_with_route(
    a: &[u8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [i32],
) -> EffectiveI8Route {
    #[cfg(target_arch = "aarch64")]
    {
        EffectiveI8Route::from_arm(super::arm::igemm_u8s8_with_route(a, b, m, k, n, out))
    }
    #[cfg(target_arch = "x86_64")]
    {
        super::x86::igemm_u8s8_with_route(a, b, m, k, n, out)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        super::scalar::igemm_u8s8(a, b, m, k, n, out);
        EffectiveI8Route::Scalar
    }
}

// ── Runtime kernel self-test (`focr robot selftest`) ────────────────────────
//
// The dispatch tests below run at `cargo test` time on the BUILD host. They do
// NOT prove anything about the int8 kernels on an end user's silicon — a
// distributed binary runs on a CPU the build never saw. [`selftest`] closes
// that gap: it re-runs the dispatched int8 GEMM against the bit-identical
// [`scalar`](crate::simd::scalar) oracle, in-process, on whatever tier this
// exact CPU selected, across a battery of shapes that includes the model's real
// K dimensions and a worst-case-magnitude K=6848 case (the doctrine #6 overflow
// stress). A user on an AVX2-only Threadripper or an SDOT Apple core can run
// `focr robot selftest` and get a machine-checkable verdict that the
// accelerated kernel their binary will actually dispatch to is exact on their
// hardware. Pure safe code here — it only calls the public entrypoints.

/// One int8-GEMM parity case: the dispatched kernel vs the scalar oracle on a
/// single `(m, k, n)` shape, for one operand domain (`s8s8` or `u8s8`).
#[derive(Debug, Clone)]
pub struct SelftestCase {
    /// Operand domain: `"s8s8"` (signed·signed) or `"u8s8"` (unsigned·signed).
    pub kind: &'static str,
    /// A short human label for the case (e.g. `"model:attn_proj_gemv"`).
    pub label: &'static str,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    /// True iff the dispatched kernel matched the scalar oracle on every lane.
    pub ok: bool,
    /// Number of diverging output lanes (0 when `ok`).
    pub mismatches: usize,
    /// First diverging lane as `(index, dispatched, oracle)`, if any.
    pub first_bad: Option<(usize, i32, i32)>,
}

/// The full runtime self-test verdict: the dispatched tier, every available
/// tier, and the per-shape parity results.
#[derive(Debug, Clone)]
pub struct SelftestReport {
    /// Hardware ISA tier selected from the capability set.
    pub hardware_selected: IsaTier,
    /// Effective dense GEMM route selected before the test battery. Compare
    /// [`Self::executed_routes`] and [`Self::route_consistent`] for observation.
    pub effective_route: EffectiveI8Route,
    /// Every execution-derived route observed (normally exactly one).
    pub executed_routes: Vec<EffectiveI8Route>,
    /// True when every case executed the route predicted before the battery.
    pub route_consistent: bool,
    /// Every tier detected as available on this host, best-first.
    pub available: Vec<IsaTier>,
    /// Per-shape parity cases (both operand domains).
    pub cases: Vec<SelftestCase>,
    /// True iff every case matched the oracle and every case executed the
    /// selected effective route (the headline pass/fail).
    pub all_ok: bool,
    /// A12 per-model rollup: `(model_id, ok)` grouped on the case-label
    /// prefix (`edge:`/`ktail:`/`model:`/`overflow:` = the shared +
    /// unlimited-ocr battery, reported as "unlimited-ocr"). The
    /// machine-readable per-model verdict `focr robot selftest` renders.
    pub models: Vec<(String, bool)>,
}

/// Deterministic xorshift32 — reproducible per-case fills with no `Math::random`
/// (which is unavailable) and no run-to-run variation.
fn xs32(state: &mut u32) -> u32 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    *state = s;
    s
}

/// The shape battery: edge/tail-coverage cases, the model's real GEMV
/// dimensions, and a worst-case-K stress. `seed == 0` marks the constant-extreme
/// overflow case (filled with operand-domain max magnitudes, not the PRNG).
const SELFTEST_SHAPES: &[(&str, usize, usize, usize, u32)] = &[
    // ── correctness floor + K-tail coverage (kernels block K; the tail is scalar) ──
    ("edge:1x1x1", 1, 1, 1, 0x1111_1111),
    ("edge:1x7x3", 1, 7, 3, 0x2222_2222),
    ("edge:2x3x2", 2, 3, 2, 0x3333_3333),
    ("ktail:1x15x8", 1, 15, 8, 0x4444_4444),
    ("ktail:1x16x8", 1, 16, 8, 0x5555_5555),
    ("ktail:1x17x8", 1, 17, 8, 0x6666_6666),
    ("ktail:4x33x5", 4, 33, 5, 0x7777_7777),
    // ── the model's real decode GEMV shapes (m=1, hidden=1280) ──
    ("model:attn_proj_gemv", 1, 1280, 128, 0x0bad_c0de),
    ("model:o_proj_gemv", 1, 1280, 1280, 0x1234_5678),
    ("model:expert_down_gemv", 1, 6848, 256, 0x9abc_def0),
    ("model:prefill_tile", 4, 1280, 64, 0x0f0f_0f0f),
    // ── worst-case-K overflow stress (constant extremes; seed 0 sentinel) ──
    ("overflow:max_mag_k6848", 1, 6848, 4, 0),
    // ── A12 (bd-3jo6.1.12): EVERY registered int8 decoder's real shapes,
    //    each with its own worst-case-K overflow row (doctrine #6 per model).
    //    Labels are `<model-id>:<shape>` — the per-model rollup groups on the
    //    prefix. TrOMR is deliberately absent: its decode is f32-only until
    //    the gated int8 experiment (bd-av64.12) lands.
    // GOT-OCR2 (Qwen2-0.5B: hidden 1024, fused qkv 3072, MLP 2816).
    ("got-ocr2:qkv_fused_gemv", 1, 1024, 3072, 0x6072_0001),
    ("got-ocr2:o_proj_gemv", 1, 1024, 1024, 0x6072_0002),
    ("got-ocr2:mlp_down_gemv", 1, 2816, 1024, 0x6072_0003),
    ("got-ocr2:overflow_k2816", 1, 2816, 4, 0),
    // SmolVLM2 (SmolLM2-360M: hidden 960, GQA 15q/5kv ⇒ fused qkv 1600, MLP 2560).
    ("smolvlm2:qkv_fused_gemv", 1, 960, 1600, 0x5601_0001),
    ("smolvlm2:mlp_down_gemv", 1, 2560, 960, 0x5601_0002),
    ("smolvlm2:overflow_k2560", 1, 2560, 4, 0),
    // OneChart (OPT-125M: hidden 768, fc1/fc2 3072).
    ("onechart:fc1_gemv", 1, 768, 3072, 0x0c4a_0001),
    ("onechart:fc2_gemv", 1, 3072, 768, 0x0c4a_0002),
    ("onechart:overflow_k3072", 1, 3072, 4, 0),
];

/// Run the int8-GEMM runtime self-test (the engine behind `focr robot
/// selftest`). Re-runs the dispatched kernel against the scalar oracle on this
/// host's selected tier across [`SELFTEST_SHAPES`]; never panics or allocates
/// unboundedly (shapes are fixed and small). The result is a structured verdict
/// the CLI renders to robot JSON.
#[must_use]
pub fn selftest() -> SelftestReport {
    use super::scalar;
    use std::collections::BTreeSet;
    let mut cases = Vec::with_capacity(SELFTEST_SHAPES.len() * 2);
    let mut executed_routes = BTreeSet::new();

    for &(label, m, k, n, seed) in SELFTEST_SHAPES {
        // S8S8 domain.
        let (a_s, b_s): (Vec<i8>, Vec<i8>) = if seed == 0 {
            // Worst-case magnitude: a = i8::MAX, b = i8::MIN (largest |product|).
            (vec![i8::MAX; m * k], vec![i8::MIN; n * k])
        } else {
            let mut st = seed | 1;
            (
                (0..m * k)
                    .map(|_| (xs32(&mut st) & 0xff) as u8 as i8)
                    .collect(),
                (0..n * k)
                    .map(|_| (xs32(&mut st) & 0xff) as u8 as i8)
                    .collect(),
            )
        };
        let mut got = vec![0i32; m * n];
        let mut want = vec![0i32; m * n];
        executed_routes.insert(igemm_s8s8_with_route(&a_s, &b_s, m, k, n, &mut got));
        scalar::igemm_s8s8(&a_s, &b_s, m, k, n, &mut want);
        cases.push(compare_case("s8s8", label, m, k, n, &got, &want));

        // U8S8 domain (the DynamicQuantizeLinear activation path / VNNI domain).
        let (a_u, b_u): (Vec<u8>, Vec<i8>) = if seed == 0 {
            (vec![u8::MAX; m * k], vec![i8::MIN; n * k])
        } else {
            let mut st = seed.rotate_left(7) | 1;
            (
                (0..m * k).map(|_| (xs32(&mut st) & 0xff) as u8).collect(),
                (0..n * k)
                    .map(|_| (xs32(&mut st) & 0xff) as u8 as i8)
                    .collect(),
            )
        };
        let mut gotu = vec![0i32; m * n];
        let mut wantu = vec![0i32; m * n];
        executed_routes.insert(igemm_u8s8_with_route(&a_u, &b_u, m, k, n, &mut gotu));
        scalar::igemm_u8s8(&a_u, &b_u, m, k, n, &mut wantu);
        cases.push(compare_case("u8s8", label, m, k, n, &gotu, &wantu));
    }

    let expected_route = effective_dense_route();
    let route_consistent = executed_routes.len() == 1 && executed_routes.contains(&expected_route);
    let all_ok = cases.iter().all(|c| c.ok) && route_consistent;
    // A12 per-model rollup: zoo cases group on their `<model-id>:` label
    // prefix; the shared battery + the unlimited shapes roll up under
    // "unlimited-ocr" (they ARE its kernel set — every other model reuses it).
    let mut models: Vec<(String, bool)> = Vec::new();
    for id in ["unlimited-ocr", "got-ocr2", "smolvlm2", "onechart"] {
        let ok = cases
            .iter()
            .filter(|c| match id {
                "unlimited-ocr" => {
                    !c.label.contains(':') || {
                        let p = c.label.split(':').next().unwrap_or("");
                        matches!(p, "edge" | "ktail" | "model" | "overflow")
                    }
                }
                _ => c.label.starts_with(&format!("{id}:")),
            })
            .all(|c| c.ok);
        models.push((id.to_string(), ok));
    }
    let snapshot = caps();
    SelftestReport {
        hardware_selected: snapshot.selected,
        effective_route: expected_route,
        executed_routes: executed_routes.into_iter().collect(),
        route_consistent,
        available: snapshot.available.clone(),
        cases,
        all_ok,
        models,
    }
}

/// Element-wise compare a dispatched result against the oracle into a
/// [`SelftestCase`].
fn compare_case(
    kind: &'static str,
    label: &'static str,
    m: usize,
    k: usize,
    n: usize,
    got: &[i32],
    want: &[i32],
) -> SelftestCase {
    let mut mismatches = 0usize;
    let mut first_bad = None;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        if g != w {
            mismatches += 1;
            if first_bad.is_none() {
                first_bad = Some((i, g, w));
            }
        }
    }
    SelftestCase {
        kind,
        label,
        m,
        k,
        n,
        ok: mismatches == 0,
        mismatches,
        first_bad,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // The scalar oracle, for the bit-identical cross-checks below. Imported in
    // the test module only (the non-test lib references it solely inside the
    // generic-arch fallback arm, fully-qualified, so there is no unused import
    // on the accelerated arches).
    use super::super::scalar;

    /// Capability detection must never panic and must always offer the scalar
    /// floor as the last (always-available) tier.
    #[test]
    fn detection_does_not_panic_and_has_scalar_floor() {
        let c = caps();
        assert!(!c.available.is_empty());
        assert_eq!(
            *c.available.last().expect("non-empty"),
            IsaTier::Scalar,
            "scalar must always be the floor"
        );
        // A valid override may select a non-front tier without mutating the
        // best-first hardware capability order.
        assert!(c.available.contains(&c.selected));
    }

    /// The cached snapshot is stable across calls (OnceLock identity).
    #[test]
    fn caps_is_cached() {
        let a = caps();
        let b = caps();
        assert!(std::ptr::eq(a, b), "caps() must return the cached snapshot");
        assert_eq!(detected_tier(), a.selected);
    }

    /// The reflected feature/tag strings are stable and non-empty for every
    /// variant (the `robot backends` surface).
    #[test]
    fn tier_strings_are_stable() {
        for t in [
            IsaTier::Scalar,
            IsaTier::Avx2,
            IsaTier::AvxVnni,
            IsaTier::Avx512Vnni,
            IsaTier::Sdot,
            IsaTier::Smmla,
        ] {
            assert!(!t.feature_string().is_empty());
            assert!(!t.tag().is_empty());
        }
        assert_eq!(IsaTier::Scalar.feature_string(), "scalar");
        assert_eq!(IsaTier::Sdot.feature_string(), "aarch64+neon+dotprod");
        assert_eq!(IsaTier::Smmla.feature_string(), "aarch64+neon+i8mm");
        // The currently-dispatched tier_string() round-trips through caps().
        assert_eq!(tier_string(), detected_tier().feature_string());
    }

    /// The ranking is monotone: every accelerated tier outranks Scalar so a
    /// best-first list never leaves a faster kernel behind the floor.
    #[test]
    fn scalar_is_lowest_rank() {
        for t in [
            IsaTier::Avx2,
            IsaTier::AvxVnni,
            IsaTier::Avx512Vnni,
            IsaTier::Sdot,
            IsaTier::Smmla,
        ] {
            assert!(t > IsaTier::Scalar);
        }
    }

    /// The dispatched S8S8 entrypoint produces scalar-oracle-equal results on
    /// this machine (whatever tier was selected). Hand-computed expected value.
    #[test]
    fn dispatch_s8s8_equals_scalar_oracle() {
        let a: [i8; 6] = [1, 2, 3, 4, 5, 6];
        let b: [i8; 6] = [1, 0, 1, 0, 1, 0]; // OC-major [2,3]
        let mut got = [0i32; 4];
        let mut want = [0i32; 4];
        igemm_s8s8(&a, &b, 2, 3, 2, &mut got);
        scalar::igemm_s8s8(&a, &b, 2, 3, 2, &mut want);
        assert_eq!(got, want);
        assert_eq!(got, [4, 2, 10, 5]);
    }

    /// The dispatched U8S8 entrypoint matches the scalar oracle on a randomized
    /// case (covers the actually-selected tier on this host).
    #[test]
    fn dispatch_u8s8_equals_scalar_oracle_randomized() {
        let (m, k, n) = (3usize, 19usize, 7usize);
        let mut s = 0xc0ffee_u32 | 1;
        let mut xs = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        let a: Vec<u8> = (0..m * k).map(|_| (xs() & 0xff) as u8).collect();
        let b: Vec<i8> = (0..n * k).map(|_| (xs() & 0xff) as u8 as i8).collect();
        let mut got = vec![0i32; m * n];
        let mut want = vec![0i32; m * n];
        igemm_u8s8(&a, &b, m, k, n, &mut got);
        scalar::igemm_u8s8(&a, &b, m, k, n, &mut want);
        assert_eq!(got, want);
    }

    /// The runtime self-test passes on THIS build host (whatever tier it
    /// selected): every dispatched int8 GEMM matches the scalar oracle. This is
    /// the same routine `focr robot selftest` runs on an end user's silicon.
    #[test]
    fn selftest_passes_on_build_host() {
        let report = selftest();
        assert!(
            !report.cases.is_empty(),
            "selftest must exercise at least one shape"
        );
        // Both operand domains run for every shape.
        assert_eq!(report.cases.len(), SELFTEST_SHAPES.len() * 2);
        assert!(report.available.contains(&report.hardware_selected));
        assert!(
            report.route_consistent,
            "every case must execute the predicted route"
        );
        assert_eq!(report.executed_routes, vec![report.effective_route]);
        for case in &report.cases {
            assert!(
                case.ok,
                "tier {:?} diverged from scalar oracle on {} {} ({}x{}x{}): {} lane(s), first {:?}",
                report.effective_route,
                case.kind,
                case.label,
                case.m,
                case.k,
                case.n,
                case.mismatches,
                case.first_bad,
            );
        }
        assert!(report.all_ok, "headline verdict must reflect all-ok cases");
    }

    /// A12: every registered int8 decoder appears in the per-model rollup,
    /// its real-shape + worst-case-K rows exist, and each rollup verdict is
    /// consistent with its own cases.
    #[test]
    fn selftest_reports_a_per_model_verdict_for_every_registered_decoder() {
        let report = selftest();
        let ids: Vec<&str> = report.models.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(
            ids,
            ["unlimited-ocr", "got-ocr2", "smolvlm2", "onechart"],
            "the per-model rollup must enumerate every registered int8 decoder"
        );
        for id in ["got-ocr2", "smolvlm2", "onechart"] {
            assert!(
                report
                    .cases
                    .iter()
                    .any(|c| c.label.starts_with(&format!("{id}:overflow_k"))),
                "{id} must carry its own worst-case-K overflow row (doctrine #6 per model)"
            );
            let model_ok = report.models.iter().find(|(m, _)| m == id).unwrap().1;
            let cases_ok = report
                .cases
                .iter()
                .filter(|c| c.label.starts_with(&format!("{id}:")))
                .all(|c| c.ok);
            assert_eq!(
                model_ok, cases_ok,
                "{id}: rollup verdict must equal its cases"
            );
        }
        println!(
            r#"{{"check":"selftest_per_model_verdicts","models":{},"result":"pass"}}"#,
            report.models.len()
        );
    }

    /// The worst-case-magnitude K=6848 case actually exercises the documented
    /// extremes (so the overflow stress is real, not a degenerate zero case),
    /// and its hand-derived sum is what both kernels produce.
    #[test]
    fn selftest_overflow_case_is_worst_case_and_exact() {
        // u8s8 worst case: a = u8::MAX (255), b = i8::MIN (-128), K = 6848.
        // Σ = 255 * (-128) * 6848 = -223_518_720, comfortably inside i32 and
        // ~9.6x above the i32 floor (-2_147_483_648) — the doctrine #6 headroom,
        // proven live on this silicon.
        let (k, n) = (6848usize, 4usize);
        let a = vec![u8::MAX; k];
        let b = vec![i8::MIN; n * k];
        let mut got = vec![0i32; n];
        let mut want = vec![0i32; n];
        igemm_u8s8(&a, &b, 1, k, n, &mut got);
        scalar::igemm_u8s8(&a, &b, 1, k, n, &mut want);
        assert_eq!(got, want);
        assert!(got.iter().all(|&v| v == -223_518_720));
    }
}
