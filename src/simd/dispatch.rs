//! Runtime ISA dispatch + the public int8 GEMM entrypoint (plan §6.2 / §6.6,
//! bd-2mo.1/.2).
//!
//! This is the **single entrypoint** the rest of the engine calls
//! ([`igemm_s8s8`] / [`igemm_u8s8`]); it picks the best available int8 kernel at
//! RUNTIME and falls back to the [`scalar`] oracle. Selection is:
//!
//! * **x86-64:** `AVX-512-VNNI > AVX-VNNI > AVX2 > scalar`
//! * **aarch64:** `SMMLA (i8mm) > SDOT (dotprod) > scalar`
//! * **everything else:** `scalar`
//!
//! (An AMX tier is not advertised: the `x86.rs` backend implements no AMX
//! kernel, and `robot backends` must report the *dispatched* tier, not the
//! host's maximum capability — doctrine #8. The variant is added back here when
//! the backend grows one.)
//!
//! The chosen tier is detected **once** (cached in a [`OnceLock`]) via the
//! standard-library feature-detection macros (`is_aarch64_feature_detected!` /
//! `is_x86_feature_detected!`) so the per-call cost is a single relaxed atomic
//! load. The dispatch itself contains **no `unsafe`** — it routes by
//! `target_arch` to the per-arch backend (`arm.rs` / `x86.rs`), each of which
//! owns its own audited `unsafe` island, performs the *same* runtime feature
//! detection internally to pick its sub-tier, and falls back to the
//! bit-identical [`scalar`] oracle when no accelerated tier is present. So
//! correctness never depends on this dispatcher's reported tier; the reported
//! tier is purely the `robot backends` reflection of what the backend will run.
//!
//! `focr robot backends` reflects [`detected_tier`] / [`available_tiers`] /
//! [`tier_string`] (bd-2mo.2).

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
    /// (e.g. `aarch64+neon+dotprod+i8mm`, `x86_64+avx512vnni`, `scalar`). This
    /// is the **dispatched** tier, not the host's maximum capability.
    #[must_use]
    pub fn feature_string(self) -> &'static str {
        match self {
            IsaTier::Scalar => "scalar",
            IsaTier::Avx2 => "x86_64+avx2",
            IsaTier::AvxVnni => "x86_64+avx2+avxvnni",
            IsaTier::Avx512Vnni => "x86_64+avx512vnni",
            IsaTier::Sdot => "aarch64+neon+dotprod",
            IsaTier::Smmla => "aarch64+neon+dotprod+i8mm",
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

/// The cached capability snapshot: the chosen (best-available) tier plus every
/// tier this host could dispatch (for `robot backends`).
#[derive(Debug, Clone)]
pub struct Caps {
    /// The single tier the GEMM entrypoints dispatch to.
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

/// The single tier the int8 GEMM entrypoints dispatch to on this host.
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

/// The dispatched tier's stable feature string (the value `robot backends`
/// reports as `selected`).
#[must_use]
pub fn tier_string() -> &'static str {
    detected_tier().feature_string()
}

/// Run the actual runtime feature detection. Builds the `available` list
/// best-first per the documented per-arch order, then takes the front as
/// `selected` (scalar is always last and always present).
fn detect() -> Caps {
    let mut available: Vec<IsaTier> = Vec::new();

    // ── aarch64: SMMLA > SDOT > scalar ──────────────────────────────────────
    #[cfg(target_arch = "aarch64")]
    {
        // `is_aarch64_feature_detected!` is safe: it reads HWCAP / sysctl and is
        // the documented gate for the matching intrinsics. We only push (and
        // thus only ever select) a tier whose feature is confirmed present.
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            available.push(IsaTier::Smmla);
        }
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            available.push(IsaTier::Sdot);
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

    // `available` is already in best-first order by construction; `selected` is
    // the front. (We do not sort by the enum discriminant because the per-arch
    // push order already encodes the documented preference and is unambiguous.)
    let selected = available[0];
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
    #[cfg(target_arch = "aarch64")]
    {
        // ARM backend: picks SMMLA > SDOT > scalar internally.
        super::arm::igemm_s8s8(a, b, m, k, n, out);
    }
    #[cfg(target_arch = "x86_64")]
    {
        // x86 backend: picks AVX-512-VNNI > AVX-VNNI > AVX2 > scalar internally.
        super::x86::igemm_s8s8(a, b, m, k, n, out);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        super::scalar::igemm_s8s8(a, b, m, k, n, out);
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
    #[cfg(target_arch = "aarch64")]
    {
        super::arm::igemm_u8s8(a, b, m, k, n, out);
    }
    #[cfg(target_arch = "x86_64")]
    {
        super::x86::igemm_u8s8(a, b, m, k, n, out);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        super::scalar::igemm_u8s8(a, b, m, k, n, out);
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
        // `selected` is the best-first front and must be a member of available.
        assert_eq!(c.selected, c.available[0]);
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
}
