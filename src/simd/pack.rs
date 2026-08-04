//! Portable SMMLA panel packing — the SINGLE source of truth for the
//! `[2 rows × 8 cols]` interleaved layout the aarch64 i8mm micro-kernel walks
//! (bd-2mo.3, plan §6.6).
//!
//! Doctrine #4's measured law (NE-INH-3): "the instruction is not the lever;
//! the blocking is" — an un-blocked SMMLA is load-bound and SLOWER than SDOT.
//! The micro-kernel therefore consumes both operands as contiguous 16-byte
//! panels. This module owns that layout as a **pure, arch-independent
//! permutation** so three consumers stay byte-identical by construction:
//!
//! * the runtime kernel ([`crate::simd::arm`]) packing activations (and, when
//!   handed a row-major B, weights) per call,
//! * `focr convert --arch aarch64-smmla` pre-packing decoder int8 weights
//!   OFFLINE into the `.focrq` payload (the llama.cpp `q4_0_4x8` repack
//!   lesson: load contiguous register tiles with zero runtime shuffle),
//! * the loader un-permuting a pre-packed artifact back to row-major on a
//!   host whose dispatched tier is not SMMLA (degrade to generic, never UB).
//!
//! The packing is a permutation with zero padding — int dot-products over
//! zero lanes contribute zero, so packed GEMMs are EXACT (doctrine #1: the
//! permutation is lossless by construction; [`smmla_unpack_panels`] is its
//! byte-exact inverse over the logical region).
//!
//! ## Layout
//!
//! For a row-major `[rows, k]` int8 matrix: `row_pairs = ceil(rows/2)` and
//! `kb = ceil(k/8)`. The packed stream is walked pair-major: for row pair `p`
//! and K-block `b`, the 16-byte panel at `(p*kb + b) * 16` is
//! `[row(2p)[8b..8b+8], row(2p+1)[8b..8b+8]]`, zero-filled past `rows`/`k`.

/// Packed byte length of an SMMLA-panel `[rows, k]` matrix:
/// `ceil(rows/2) * ceil(k/8) * 16`. Equals `rows * k` exactly when
/// `rows % 2 == 0 && k % 8 == 0` (true for every registered decoder shape).
#[must_use]
pub fn smmla_packed_len(rows: usize, k: usize) -> usize {
    rows.div_ceil(2) * k.div_ceil(8) * 16
}

/// Pre-pack a region of a row-major `[rows, k]` int8 matrix (rows
/// `[base_row, base_row + rows)` of a parent with row stride `src_k`) into
/// SMMLA panels.
///
/// Returns `(packed, row_pairs, kb)`; `packed.len() == row_pairs * kb * 16`.
/// Rows beyond `rows` and columns beyond `k` are zero-filled.
///
/// # Panics
/// Panics (slice bounds) if `src` is shorter than the addressed region.
#[must_use]
pub fn smmla_pack_panels(
    src: &[i8],
    base_row: usize,
    rows: usize,
    k: usize,
    src_k: usize,
) -> (Vec<i8>, usize, usize) {
    let row_pairs = rows.div_ceil(2);
    let kb = k.div_ceil(8);
    let mut packed = vec![0i8; row_pairs * kb * 16];
    for p in 0..row_pairs {
        for block in 0..kb {
            let panel = (p * kb + block) * 16;
            let kcol = block * 8;
            let kvalid = (k - kcol).min(8); // columns present in this block
            for sub in 0..2 {
                let row = p * 2 + sub;
                if row >= rows {
                    continue; // zero-padded tail row
                }
                let src_off = (base_row + row) * src_k + kcol;
                let dst_off = panel + sub * 8;
                packed[dst_off..dst_off + kvalid].copy_from_slice(&src[src_off..src_off + kvalid]);
            }
        }
    }
    (packed, row_pairs, kb)
}

/// Invert [`smmla_pack_panels`]: reconstruct the row-major `[rows, k]` int8
/// matrix from a full-matrix panel stream (i.e. one produced with
/// `base_row == 0`, `rows`, `k`, `src_k == k`).
///
/// Lossless over the logical region by construction (the padding lanes are
/// dropped, not read back).
///
/// # Errors
/// Returns a descriptive message if `packed.len()` disagrees with
/// [`smmla_packed_len`] — a corrupt artifact must fail loudly, not slice-panic.
pub fn smmla_unpack_panels(packed: &[i8], rows: usize, k: usize) -> Result<Vec<i8>, String> {
    let row_pairs = rows.div_ceil(2);
    let kb = k.div_ceil(8);
    let want = row_pairs * kb * 16;
    if packed.len() != want {
        return Err(format!(
            "SMMLA panel stream is {} bytes; [{rows}, {k}] requires {want}",
            packed.len()
        ));
    }
    let mut out = vec![0i8; rows * k];
    for p in 0..row_pairs {
        for block in 0..kb {
            let panel = (p * kb + block) * 16;
            let kcol = block * 8;
            let kvalid = (k - kcol).min(8);
            for sub in 0..2 {
                let row = p * 2 + sub;
                if row >= rows {
                    continue;
                }
                let dst_off = row * k + kcol;
                out[dst_off..dst_off + kvalid]
                    .copy_from_slice(&packed[panel + sub * 8..panel + sub * 8 + kvalid]);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(rows: usize, k: usize) -> Vec<i8> {
        // Deterministic non-symmetric fill exercising the full i8 range.
        (0..rows * k)
            .map(|i| ((i as i64 * 73 + 11) % 255 - 127) as i8)
            .collect()
    }

    /// Round-trip is byte-exact for even/odd rows and k on/off the 8-boundary
    /// (the acceptance-criteria losslessness proof at the unit level).
    #[test]
    fn pack_unpack_round_trips_byte_exact() {
        for (rows, k) in [
            (8, 16),
            (7, 16),
            (8, 13),
            (5, 3),
            (1, 1),
            (2, 8),
            (64, 1280),
        ] {
            let src = filled(rows, k);
            let (packed, pairs, kb) = smmla_pack_panels(&src, 0, rows, k, k);
            assert_eq!(packed.len(), pairs * kb * 16);
            assert_eq!(packed.len(), smmla_packed_len(rows, k));
            let back = smmla_unpack_panels(&packed, rows, k).expect("len agrees");
            assert_eq!(back, src, "round-trip must be lossless for [{rows}, {k}]");
            println!(
                r#"{{"check":"smmla_pack_roundtrip","rows":{rows},"k":{k},"packed_len":{},"result":"pass"}}"#,
                packed.len()
            );
        }
    }

    /// Same source → same packed bytes (the determinism acceptance criterion,
    /// unit level; the converter e2e asserts the content hash).
    #[test]
    fn packing_is_deterministic() {
        let src = filled(16, 64);
        let (a, _, _) = smmla_pack_panels(&src, 0, 16, 64, 64);
        let (b, _, _) = smmla_pack_panels(&src, 0, 16, 64, 64);
        assert_eq!(a, b);
    }

    /// Padding lanes are zero (int-dot exactness depends on this).
    #[test]
    fn padding_lanes_are_zero() {
        let (rows, k) = (3, 5); // odd rows, k not a multiple of 8
        let src = vec![7i8; rows * k];
        let (packed, pairs, kb) = smmla_pack_panels(&src, 0, rows, k, k);
        assert_eq!((pairs, kb), (2, 1));
        // pair 1 holds rows 2 (real) + 3 (padded); its second sub-row is zero.
        let panel = kb * 16; // pair 1, block 0
        assert!(packed[panel + 8..panel + 16].iter().all(|&b| b == 0));
        assert!(packed[panel + k..panel + 8].iter().all(|&b| b == 0));
        // real lanes carry the fill
        assert!(packed[panel..panel + k].iter().all(|&b| b == 7));
    }

    /// The registered decoder shapes pack with ZERO padding (packed len ==
    /// n*k), so the offline artifact is byte-size-neutral for real models.
    #[test]
    fn real_decoder_shapes_pack_without_padding() {
        for (n, k) in [
            (1280usize, 1280usize),
            (6848, 1280),
            (256, 6848),
            (3072, 1024),
            (1024, 2816),
            (1600, 960),
            (960, 2560),
            (3072, 768),
            (768, 3072),
        ] {
            assert_eq!(
                smmla_packed_len(n, k),
                n * k,
                "[{n}, {k}] must tile cleanly (n even, k % 8 == 0)"
            );
        }
    }

    /// A sub-region pack (the runtime kernel's tile-at-a-time use) equals the
    /// corresponding slice of the full-matrix pack when the base row is even —
    /// the property the offline layout's block addressing relies on
    /// (`I8_GEMV_BLOCK = 64` keeps every runtime block base even).
    #[test]
    fn even_base_region_pack_is_a_slice_of_the_full_pack() {
        let (rows, k) = (32, 24);
        let src = filled(rows, k);
        let (full, _pairs, kb) = smmla_pack_panels(&src, 0, rows, k, k);
        for (base, cnt) in [(0usize, 8usize), (8, 8), (16, 16), (24, 8), (2, 30)] {
            let (region, rpairs, rkb) = smmla_pack_panels(&src, base, cnt, k, k);
            assert_eq!(rkb, kb);
            let off = (base / 2) * kb * 16;
            assert_eq!(
                region,
                full[off..off + rpairs * kb * 16],
                "even-base region [{base}, {base}+{cnt}) must alias the full pack"
            );
        }
    }
}
