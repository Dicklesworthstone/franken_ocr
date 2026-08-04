//! Shared proptest generators (bd-10sb.1) — the property-based test PLUMBING.
//!
//! E-TEST owns the generators; the invariants live in the consuming suites
//! (`tests/property_suite.rs` today; the metamorphic/unit suites may reuse
//! them via `#[path]` include). Every generator is documented with the domain
//! it models so a failing shrink reads as a real input, not noise.
//!
//! Design notes:
//! * Shapes are SMALL-biased (fast shrink, fast cases) with the doctrine-#6
//!   worst-case contraction depths (`K = 2816 / 2560 / 3072 / 6848` — the
//!   registered decoders' largest K) injected as explicit variants so every
//!   run exercises the overflow-relevant regime, not just tiny K.
//! * int8 operand pools are biased toward the saturated extremes
//!   (`i8::MIN`/`i8::MAX`, `u8::MAX`) — the values that stress i32
//!   accumulation — while keeping uniform coverage of the interior.

// Consumers select the generators they need; the rest stay available to the
// unit/metamorphic suites without a warning storm in any one includer.
#![allow(dead_code)]

use proptest::prelude::*;

/// The registered decoders' worst-case contraction depths (doctrine #6):
/// unlimited-ocr 6848, onechart 3072, got-ocr2 2816, smolvlm2 2560.
pub const WORST_CASE_KS: [usize; 4] = [2816, 2560, 3072, 6848];

/// GEMM shape `(m, k, n)`: small-biased `m`/`n`, `k` either small or one of
/// the worst-case depths (1-in-4 weighting keeps runtime bounded while every
/// run still hits the deep-K regime).
pub fn gemm_shape() -> impl Strategy<Value = (usize, usize, usize)> {
    let small_k = 1usize..=96;
    let deep_k = proptest::sample::select(WORST_CASE_KS.to_vec());
    (
        1usize..=6,
        prop_oneof![3 => small_k, 1 => deep_k],
        1usize..=12,
    )
}

/// A signed int8 operand value, biased toward the saturated extremes.
pub fn i8_extreme_biased() -> impl Strategy<Value = i8> {
    prop_oneof![
        2 => Just(i8::MIN),
        2 => Just(i8::MAX),
        6 => any::<i8>(),
    ]
}

/// An unsigned int8 activation value, biased toward the extremes
/// (`DynamicQuantizeLinear` asymmetric domain).
pub fn u8_extreme_biased() -> impl Strategy<Value = u8> {
    prop_oneof![
        2 => Just(0u8),
        2 => Just(u8::MAX),
        6 => any::<u8>(),
    ]
}

/// A `len`-element saturated-biased i8 buffer.
pub fn i8_buffer(len: usize) -> impl Strategy<Value = Vec<i8>> {
    proptest::collection::vec(i8_extreme_biased(), len)
}

/// A `len`-element saturated-biased u8 buffer.
pub fn u8_buffer(len: usize) -> impl Strategy<Value = Vec<u8>> {
    proptest::collection::vec(u8_extreme_biased(), len)
}

/// A small RGB image: dims 1..=48 (odd/even, degenerate 1-pixel edges
/// included), arbitrary pixel bytes.
pub fn small_rgb_image() -> impl Strategy<Value = image::DynamicImage> {
    (1u32..=48, 1u32..=48).prop_flat_map(|(w, h)| {
        proptest::collection::vec(any::<u8>(), (w * h * 3) as usize).prop_map(move |px| {
            image::DynamicImage::ImageRgb8(
                image::RgbImage::from_raw(w, h, px).expect("buffer sized to w*h*3"),
            )
        })
    })
}

/// One byte-level mutation applied to an (otherwise valid) artifact blob —
/// the untrusted-input model for the `.focrq` parser property: bit flips
/// anywhere (magic, version, header JSON, directory, payload), truncation,
/// and mid-blob deletion.
#[derive(Debug, Clone)]
pub enum BlobMutation {
    /// XOR one byte at `pos % len` with a non-zero mask.
    FlipByte { pos: usize, mask: u8 },
    /// Truncate to `keep % (len + 1)` bytes.
    Truncate { keep: usize },
    /// Remove one byte at `pos % len` (shifts everything after).
    DeleteByte { pos: usize },
}

/// A batch of 1..=4 mutations (compound corruption).
pub fn blob_mutations() -> impl Strategy<Value = Vec<BlobMutation>> {
    let one = prop_oneof![
        (any::<usize>(), 1u8..=255).prop_map(|(pos, mask)| BlobMutation::FlipByte { pos, mask }),
        any::<usize>().prop_map(|keep| BlobMutation::Truncate { keep }),
        any::<usize>().prop_map(|pos| BlobMutation::DeleteByte { pos }),
    ];
    proptest::collection::vec(one, 1..=4)
}

/// Apply `mutations` to a copy of `blob` (always well-defined: positions are
/// taken modulo the current length; mutating an empty blob is a no-op).
#[must_use]
pub fn apply_mutations(blob: &[u8], mutations: &[BlobMutation]) -> Vec<u8> {
    let mut b = blob.to_vec();
    for m in mutations {
        if b.is_empty() {
            break;
        }
        match *m {
            BlobMutation::FlipByte { pos, mask } => {
                let i = pos % b.len();
                b[i] ^= mask;
            }
            BlobMutation::Truncate { keep } => {
                let k = keep % (b.len() + 1);
                b.truncate(k);
            }
            BlobMutation::DeleteByte { pos } => {
                let i = pos % b.len();
                b.remove(i);
            }
        }
    }
    b
}

/// Arbitrary Unicode text for the tokenizer round-trip property: mixes plain
/// ASCII, CJK/math/emoji-heavy `any::<String>()`, and whitespace/digit runs
/// (the pretokenizer's split classes).
pub fn tokenizer_text() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => "[ -~]{0,64}",                  // printable ASCII incl empty
        3 => any::<String>(),                 // arbitrary Unicode (proptest default)
        2 => "[0-9]{1,32}",                   // digit runs (digit-split rules)
        2 => "[\\n\\t ]{1,16}[a-z]{1,8}[\\n\\t ]{1,16}", // whitespace classes
    ]
}
