//! Runtime ISA dispatch + the public int8 GEMM entrypoint (plan §6.2 / §6.6,
//! bd-2mo.1/.2).
//!
//! This is the **single entrypoint** the rest of the engine calls
//! ([`igemm_s8s8`] / [`igemm_u8s8`]); it picks the best available int8 kernel at
//! RUNTIME and falls back to the [`scalar`] oracle. Selection is:
//!
//! * **x86-64:** `AMX > AVX-512-VNNI > AVX-VNNI > AVX2 > scalar`
//! * **aarch64:** `SMMLA (i8mm) > SDOT (dotprod) > scalar`
//! * **everything else:** `scalar`
//!
//! The chosen tier is detected **once** (cached in a [`OnceLock`]) via the
//! standard-library feature-detection macros (`is_aarch64_feature_detected!` /
//! `is_x86_feature_detected!`) so the per-call cost is a single relaxed atomic
//! load. The dispatch itself contains **no `unsafe`** — it only *selects* which
//! safe-wrapper kernel to call. Each accelerated wrapper (in `arm.rs` / `x86.rs`)
//! owns its own audited `unsafe` island and is only ever reached *after* this
//! module has confirmed the required CPU feature is present (the safety
//! precondition for those intrinsics). On a target whose feature is absent we
//! never construct the corresponding `IsaTier`, so the intrinsic is never
//! called — the fallback is the bit-identical [`scalar`] oracle.
//!
//! `focr robot backends` reflects [`detected_tier`] / [`available_tiers`] /
//! [`tier_string`] (bd-2mo.2).

use std::sync::OnceLock;

use super::scalar;

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
    /// x86-64 AVX2 (i16-accumulate `vpmaddubsw` path; carries its own
    /// split-accumulate overflow handling in `x86.rs`).
    Avx2 = 1,
    /// x86-64 AVX-VNNI (`vpdpbusd`, U8S8, 4 MACs/i32 lane).
    AvxVnni = 2,
    /// x86-64 AVX-512-VNNI (`vpdpbusd` on 512-bit lanes).
    Avx512Vnni = 3,
    /// x86-64 AMX int8 tiles (`tdpbssd`/`tdpbusd`).
    Amx = 4,
    /// aarch64 FEAT_DotProd SDOT (4 int8 MACs/i32 lane).
    Sdot = 5,
    /// aarch64 FEAT_MATMUL_INT8 SMMLA / i8mm (8 int8 MACs/i32 lane, 2x2 tile) —
    /// the register-blocked wedge (doctrine #4).
    Smmla = 6,
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
            IsaTier::Amx => "x86_64+amx-int8",
            IsaTier::Sdot => "aarch64+neon+dotprod",
            IsaTier::Smmla => "aarch64+neon+dotprod+i8mm",
        }
    }

    /// A short tier tag (`"scalar"`, `"sdot"`, `"smmla"`, `"avx2"`,
    /// `"avxvnni"`, `"avx512vnni"`, `"amx"`) for compact JSON / logs.
    #[must_use]
    pub fn tag(self) -> &'static str {
        match self {
            IsaTier::Scalar => "scalar",
            IsaTier::Avx2 => "avx2",
            IsaTier::AvxVnni => "avxvnni",
            IsaTier::Avx512Vnni => "avx512vnni",
            IsaTier::Amx => "amx",
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

    // ── x86-64: AMX > AVX512-VNNI > AVX-VNNI > AVX2 > scalar ─────────────────
    #[cfg(target_arch = "x86_64")]
    {
        // AMX requires both the tile config and int8 compute features. We gate
        // on both; the x86.rs AMX kernel additionally performs the OS-enable
        // (`XCR0` tile state) handshake inside its island.
        if std::arch::is_x86_feature_detected!("amx-tile")
            && std::arch::is_x86_feature_detected!("amx-int8")
        {
            available.push(IsaTier::Amx);
        }
        if std::arch::is_x86_feature_detected!("avx512vnni") {
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
/// the i32 buffer `out` (length `m*n`). Dispatches to the best available kernel;
/// every path is **bit-identical** to [`scalar::igemm_s8s8`] (i32 accumulation
/// is exact, so there is no numeric divergence between tiers — verified by each
/// backend's tests against the oracle).
///
/// # Panics
/// As [`scalar::igemm_s8s8`] (length-contract violations).
pub fn igemm_s8s8(a: &[i8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    match detected_tier() {
        #[cfg(target_arch = "aarch64")]
        IsaTier::Smmla => super::arm::igemm_s8s8_smmla(a, b, m, k, n, out),
        #[cfg(target_arch = "aarch64")]
        IsaTier::Sdot => super::arm::igemm_s8s8_sdot(a, b, m, k, n, out),
        #[cfg(target_arch = "x86_64")]
        IsaTier::Amx => super::x86::igemm_s8s8_amx(a, b, m, k, n, out),
        #[cfg(target_arch = "x86_64")]
        IsaTier::Avx512Vnni => super::x86::igemm_s8s8_avx512vnni(a, b, m, k, n, out),
        #[cfg(target_arch = "x86_64")]
        IsaTier::AvxVnni => super::x86::igemm_s8s8_avxvnni(a, b, m, k, n, out),
        #[cfg(target_arch = "x86_64")]
        IsaTier::Avx2 => super::x86::igemm_s8s8_avx2(a, b, m, k, n, out),
        // Scalar floor, and the catch-all for any tier whose arch-specific arm
        // is cfg'd out on this build (keeps the match exhaustive on every
        // target while only ever *selecting* an arch-valid tier).
        _ => scalar::igemm_s8s8(a, b, m, k, n, out),
    }
}

/// Public **int8 GEMM** entrypoint, U8S8 (unsigned activations · signed
/// weights) — the asymmetric `DynamicQuantizeLinear` activation path and the
/// native VNNI operand domain.
///
/// `C[M,N] += A[M,K] (u8, row-major) · B[N,K] (i8, output-channel-major)` into
/// the i32 buffer `out`. Dispatches as [`igemm_s8s8`]; bit-identical to
/// [`scalar::igemm_u8s8`].
///
/// # Panics
/// As [`scalar::igemm_u8s8`].
pub fn igemm_u8s8(a: &[u8], b: &[i8], m: usize, k: usize, n: usize, out: &mut [i32]) {
    match detected_tier() {
        #[cfg(target_arch = "aarch64")]
        IsaTier::Smmla => super::arm::igemm_u8s8_smmla(a, b, m, k, n, out),
        #[cfg(target_arch = "aarch64")]
        IsaTier::Sdot => super::arm::igemm_u8s8_sdot(a, b, m, k, n, out),
        #[cfg(target_arch = "x86_64")]
        IsaTier::Amx => super::x86::igemm_u8s8_amx(a, b, m, k, n, out),
        #[cfg(target_arch = "x86_64")]
        IsaTier::Avx512Vnni => super::x86::igemm_u8s8_avx512vnni(a, b, m, k, n, out),
        #[cfg(target_arch = "x86_64")]
        IsaTier::AvxVnni => super::x86::igemm_u8s8_avxvnni(a, b, m, k, n, out),
        #[cfg(target_arch = "x86_64")]
        IsaTier::Avx2 => super::x86::igemm_u8s8_avx2(a, b, m, k, n, out),
        _ => scalar::igemm_u8s8(a, b, m, k, n, out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            IsaTier::Amx,
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
            IsaTier::Amx,
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
