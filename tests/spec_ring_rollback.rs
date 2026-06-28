//! bd-1azu.31 — GATE: R-SWA `RingCache` checkpoint / rollback (truncate-to-accepted).
//!
//! Speculative decoding writes a short *draft* block of K/V into the generated
//! tail, verifies it, and discards the rejected suffix. The roll-back MUST be
//! LOSSLESS (Doctrine #1): after [`RingCache::rollback_to`] the cache is
//! byte-for-byte the cache that NEVER wrote the discarded steps — the speculation
//! leaves no observable trace on the accepted positions, and a subsequent decode
//! continues exactly as if it had never happened.
//!
//! This file proves that property MODEL-FREE, with synthetic K/V/queries:
//! * (a) the cursors (`ring_len` / `ring_pos` / `effective_len` / `prefill_len`)
//!   are restored to the checkpoint's;
//! * (b) the K/V of the accepted positions is byte-identical to an independent
//!   *control* cache driven with the same prefill + accepted steps but NONE
//!   of the speculative ones — observed through the public, bit-exact
//!   `decode_attention` output over a battery of probe queries (the only
//!   public window onto the live K/V; equal output over many discriminating
//!   queries ⟺ identical live K/V rows). A *teeth* check first confirms the
//!   speculative writes DID change that output before the roll-back, so the
//!   equality afterwards is meaningful;
//! * (c) a fresh decode write after the roll-back lands in the SAME physical slot
//!   (and yields the same state/output) it would have without the speculation.
//!
//! Coverage: the warm-up append regime; the exact ring-fill boundary
//! (`checkpoint.ring_len + discarded == RING_WINDOW`, still lossless); a realistic
//! multi-round speculate→reject→accept loop; the per-stream `BatchedRingCache`
//! delegation (a roll-back on one stream never touches a sibling); and — in debug
//! builds — the contract guard that refuses a roll-back which would have to
//! resurrect an evicted (steady-state-overwritten) slot.

// Hot synthetic builders index parallel K/V stride-arrays by `(h, d)`; the index
// is genuinely needed across buffers, mirroring the kernel's own allow.
#![allow(clippy::needless_range_loop)]

use franken_ocr::native_engine::rswa::{
    BatchedRingCache, HEAD_DIM, NUM_HEADS, RING_WINDOW, RingCache, RingCheckpoint, decode_attention,
};

// ── deterministic synthetic data (no model, no dev-dependency) ─────────────────

/// SplitMix64 finalizer.
fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^= x >> 33;
    x
}

/// A reproducible per-`(seed, h, r, d)` value in `[-1, 1)`.
fn val(seed: u64, h: usize, r: usize, d: usize) -> f32 {
    let x = mix(seed
        ^ (h as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (r as u64).wrapping_mul(0x1656_67B1_9E37_79F9)
        ^ (d as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
    ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

/// Fold parts into one seed (stream / layer / step / kind discriminators).
fn seed(a: u64, b: u64, c: u64, d: u64) -> u64 {
    mix(a.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ c.wrapping_mul(0x1656_67B1_9E37_79F9)
        ^ d.wrapping_mul(0x27D4_EB2F_1656_67C5))
}

/// Head-major `[NUM_HEADS, rows, HEAD_DIM]` prefill K and V from a seed (V offset
/// so K != V).
fn prefill_kv(rows: usize, s: u64) -> (Vec<f32>, Vec<f32>) {
    let mut k = vec![0.0f32; NUM_HEADS * rows * HEAD_DIM];
    let mut v = vec![0.0f32; NUM_HEADS * rows * HEAD_DIM];
    for h in 0..NUM_HEADS {
        for r in 0..rows {
            for d in 0..HEAD_DIM {
                let idx = (h * rows + r) * HEAD_DIM + d;
                k[idx] = val(s ^ 0x4B, h, r, d);
                v[idx] = val(s ^ 0x56, h, r, d);
            }
        }
    }
    (k, v)
}

/// One decode token's `[NUM_HEADS, HEAD_DIM]` K/V from a seed.
fn token_kv(s: u64) -> (Vec<f32>, Vec<f32>) {
    let mut k = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    let mut v = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    for h in 0..NUM_HEADS {
        for d in 0..HEAD_DIM {
            k[h * HEAD_DIM + d] = val(s ^ 0x4B, h, 0, d);
            v[h * HEAD_DIM + d] = val(s ^ 0x56, h, 0, d);
        }
    }
    (k, v)
}

/// One probe query `[NUM_HEADS, HEAD_DIM]`.
fn query(s: u64) -> Vec<f32> {
    let mut q = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    for h in 0..NUM_HEADS {
        for d in 0..HEAD_DIM {
            q[h * HEAD_DIM + d] = val(s ^ 0x71, h, 0, d);
        }
    }
    q
}

const PREFILL_SEED: u64 = 0xA1;
const ACCEPT: u64 = 0; // kind discriminator: accepted (real) decode step
const SPEC: u64 = 1; //   kind discriminator: speculative (draft) decode step

/// Prefill `c` and drive `n` decode steps of the given `kind` for `(stream,
/// layer)` (single-stream tests use `stream = layer = 0`).
fn drive_steps(c: &mut RingCache, stream: u64, layer: u64, kind: u64, t0: usize, n: usize) {
    for t in 0..n {
        let (k, v) = token_kv(seed(stream, layer, (t0 + t) as u64, kind));
        c.write_decode_step(&k, &v).unwrap();
    }
}

fn out(c: &RingCache, q: &[f32]) -> Vec<f32> {
    decode_attention(c, q).unwrap().data
}

/// Probe battery: many discriminating queries so equal output ⟺ identical live
/// K/V rows (a single key differing would perturb some probe's softmax).
fn probes(n: u64) -> Vec<Vec<f32>> {
    (0..n).map(|i| query(0x9000 + i)).collect()
}

// ── (1) warm-up: speculate, roll back, prove byte-identity + fresh-write slot ──

#[test]
fn rollback_discards_speculative_and_is_byte_identical() {
    let cap = 512;
    let prefill_rows = 6usize;
    let accept = 10usize;
    let spec = 7usize;

    // Test cache: prefill + accepted steps + (later) speculative steps.
    let mut test = RingCache::new(cap);
    let (pk, pv) = prefill_kv(prefill_rows, PREFILL_SEED);
    test.record_prefill(&pk, &pv, prefill_rows).unwrap();
    drive_steps(&mut test, 0, 0, ACCEPT, 0, accept);

    // Control cache: identical prefill + accepted steps, NEVER the speculation.
    let mut control = RingCache::new(cap);
    control.record_prefill(&pk, &pv, prefill_rows).unwrap();
    drive_steps(&mut control, 0, 0, ACCEPT, 0, accept);

    // Checkpoint BEFORE drafting.
    let cp: RingCheckpoint = test.checkpoint();
    assert_eq!(cp.prefill_len(), Some(prefill_rows));
    assert_eq!(cp.ring_len(), accept);
    assert_eq!(cp.ring_pos(), accept); // warm-up: pos tracks len
    assert_eq!(cp.effective_len(), prefill_rows + accept);

    // Write the speculative (draft) block — distinct K/V from anything accepted.
    drive_steps(&mut test, 0, 0, SPEC, 0, spec);
    assert_eq!(test.ring_len(), accept + spec);

    // TEETH: the speculation MUST be observable before the roll-back, otherwise
    // the byte-identity assertion below would be vacuous.
    let probe = query(0xBEEF);
    assert_ne!(
        out(&test, &probe),
        out(&control, &probe),
        "speculative steps must change the attention output (else the test has no teeth)"
    );

    // Roll back to the checkpoint.
    test.rollback_to(&cp);

    // (a) cursors restored exactly.
    assert_eq!(test.prefill_len(), cp.prefill_len());
    assert_eq!(test.ring_len(), cp.ring_len());
    assert_eq!(test.ring_pos(), cp.ring_pos());
    assert_eq!(test.effective_len(), cp.effective_len());
    assert!(!test.is_warm());

    // (b) accepted-position K/V byte-identical to the control over the whole probe
    //     battery (bit-exact default `decode_attention`).
    for (i, q) in probes(16).iter().enumerate() {
        assert_eq!(
            out(&test, q),
            out(&control, q),
            "rolled-back cache diverges from the never-speculated control on probe {i}"
        );
    }

    // (c) a fresh decode write lands in the SAME slot it would have without the
    //     speculation, and keeps the caches byte-identical thereafter.
    let (fk, fv) = token_kv(seed(0, 0, accept as u64, ACCEPT));
    let test_slot = test.write_decode_step(&fk, &fv).unwrap();
    let ctrl_slot = control.write_decode_step(&fk, &fv).unwrap();
    assert_eq!(
        test_slot, ctrl_slot,
        "fresh write must land where it would have"
    );
    assert_eq!(test_slot, accept, "warm-up fresh write appends at ring_len");
    assert_eq!(test.ring_len(), accept + 1);
    for q in probes(8) {
        assert_eq!(out(&test, &q), out(&control, &q));
    }
}

// ── (2) exact ring-fill boundary: checkpoint.ring_len + discarded == W ─────────

#[test]
fn rollback_at_exact_ring_fill_boundary_is_lossless() {
    let cap = 64;
    let prefill_rows = 4usize;
    // Accept up to W-3 so the 3 draft steps fill the ring EXACTLY to W (the last
    // lossless point: every draft slot is a fresh append, none evicts a live row).
    let accept = RING_WINDOW - 3;
    let spec = 3usize;

    let (pk, pv) = prefill_kv(prefill_rows, PREFILL_SEED);

    let mut test = RingCache::new(cap);
    test.record_prefill(&pk, &pv, prefill_rows).unwrap();
    drive_steps(&mut test, 0, 0, ACCEPT, 0, accept);

    let mut control = RingCache::new(cap);
    control.record_prefill(&pk, &pv, prefill_rows).unwrap();
    drive_steps(&mut control, 0, 0, ACCEPT, 0, accept);

    let cp = test.checkpoint();
    assert_eq!(cp.ring_len(), accept);

    drive_steps(&mut test, 0, 0, SPEC, 0, spec);
    assert_eq!(test.ring_len(), RING_WINDOW, "draft fills the ring exactly");
    assert!(test.is_warm());

    test.rollback_to(&cp);
    assert_eq!(test.ring_len(), accept);
    assert_eq!(test.ring_pos(), accept);
    assert!(!test.is_warm(), "roll-back undoes the warm transition");

    for (i, q) in probes(12).iter().enumerate() {
        assert_eq!(
            out(&test, q),
            out(&control, q),
            "boundary roll-back diverges on probe {i}"
        );
    }
}

// ── (3) realistic loop: speculate → reject → accept one, many rounds ───────────

#[test]
fn multiple_speculation_rounds_match_non_speculating_control() {
    let cap = 512;
    let prefill_rows = 5usize;
    let rounds = 12usize;
    let draft = 5usize; // rejected draft length each round

    let (pk, pv) = prefill_kv(prefill_rows, PREFILL_SEED);

    let mut test = RingCache::new(cap);
    test.record_prefill(&pk, &pv, prefill_rows).unwrap();

    // Control only ever writes the single accepted token per round.
    let mut control = RingCache::new(cap);
    control.record_prefill(&pk, &pv, prefill_rows).unwrap();

    for r in 0..rounds {
        let cp = test.checkpoint();
        // Draft a speculative block, then reject ALL of it.
        drive_steps(&mut test, 0, 0, SPEC, r * draft, draft);
        test.rollback_to(&cp);
        assert_eq!(test.ring_len(), cp.ring_len());
        assert_eq!(test.ring_pos(), cp.ring_pos());

        // Accept exactly one real token (identical seed in both caches).
        let (k, v) = token_kv(seed(0, 0, r as u64, ACCEPT));
        let ts = test.write_decode_step(&k, &v).unwrap();
        let cs = control.write_decode_step(&k, &v).unwrap();
        assert_eq!(
            ts, cs,
            "post-reject accepted write must land identically (round {r})"
        );
    }

    assert_eq!(test.ring_len(), rounds);
    assert_eq!(test.effective_len(), control.effective_len());
    for (i, q) in probes(16).iter().enumerate() {
        assert_eq!(
            out(&test, q),
            out(&control, q),
            "after {rounds} speculate/reject/accept rounds the cache diverges on probe {i}"
        );
    }
}

// ── (4) BatchedRingCache: roll-back is per-stream and lossless ─────────────────

#[test]
fn batched_rollback_is_per_stream_and_lossless() {
    let caps = [16usize, 24, 32, 20];
    let n_layers = 2usize;
    let accept = 12usize;
    let spec = 6usize;
    let (ts, tl) = (2usize, 1usize); // the (stream, layer) we speculate on
    let (ss, sl) = (1usize, 0usize); // an untouched sibling

    let mut test = BatchedRingCache::new(&caps, n_layers);
    let mut control = BatchedRingCache::new(&caps, n_layers);

    for (s, &cap) in caps.iter().enumerate() {
        for l in 0..n_layers {
            let (k, v) = prefill_kv(cap, seed(s as u64, l as u64, 0, 7));
            test.record_prefill(s, l, &k, &v, cap).unwrap();
            control.record_prefill(s, l, &k, &v, cap).unwrap();
        }
    }
    // Identical accepted decode history on every (stream, layer) of both caches.
    for t in 0..accept {
        for (s, _) in caps.iter().enumerate() {
            for l in 0..n_layers {
                let (k, v) = token_kv(seed(s as u64, l as u64, t as u64, ACCEPT));
                test.write_decode_step(s, l, &k, &v).unwrap();
                control.write_decode_step(s, l, &k, &v).unwrap();
            }
        }
    }

    let cp = test.checkpoint(ts, tl);
    assert_eq!(cp.ring_len(), accept);
    let sib_len_before = test.layer(ss, sl).ring_len();

    // Speculate ONLY on (ts, tl).
    for t in 0..spec {
        let (k, v) = token_kv(seed(ts as u64, tl as u64, t as u64, SPEC));
        test.write_decode_step(ts, tl, &k, &v).unwrap();
    }
    assert_eq!(test.layer(ts, tl).ring_len(), accept + spec);

    test.rollback_to(ts, tl, &cp);

    // (a) target cursors restored.
    assert_eq!(test.layer(ts, tl).ring_len(), cp.ring_len());
    assert_eq!(test.layer(ts, tl).ring_pos(), cp.ring_pos());

    // (b) target (stream, layer) byte-identical to the never-speculated control.
    for (i, q) in probes(10).iter().enumerate() {
        assert_eq!(
            decode_attention(test.layer(ts, tl), q).unwrap().data,
            decode_attention(control.layer(ts, tl), q).unwrap().data,
            "batched target stream diverges on probe {i}"
        );
    }

    // The sibling stream/layer was never written or rolled back — still matches.
    assert_eq!(test.layer(ss, sl).ring_len(), sib_len_before);
    for (i, q) in probes(6).iter().enumerate() {
        assert_eq!(
            decode_attention(test.layer(ss, sl), q).unwrap().data,
            decode_attention(control.layer(ss, sl), q).unwrap().data,
            "sibling stream perturbed by another stream's roll-back on probe {i}"
        );
    }
}

// ── (5) contract guard (debug builds): refuse a roll-back that would resurrect ──
//        an evicted (steady-state-overwritten) slot, which indices cannot do.

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "not lossless")]
fn rollback_panics_when_speculation_evicted_a_live_slot() {
    let cap = 32;
    let prefill_rows = 4usize;
    let (pk, pv) = prefill_kv(prefill_rows, PREFILL_SEED);

    let mut c = RingCache::new(cap);
    c.record_prefill(&pk, &pv, prefill_rows).unwrap();
    // Fill the ring completely so every slot 0..W is live at the checkpoint.
    drive_steps(&mut c, 0, 0, ACCEPT, 0, RING_WINDOW);
    assert!(c.is_warm());

    let cp = c.checkpoint();
    assert_eq!(cp.ring_len(), RING_WINDOW);

    // One steady-state draft step overwrites slot 0 (a checkpoint-live row): its
    // prior K/V is physically gone, so the index-only roll-back is NOT lossless
    // and must be rejected rather than silently corrupt the cache.
    drive_steps(&mut c, 0, 0, SPEC, 0, 1);
    c.rollback_to(&cp);
}
