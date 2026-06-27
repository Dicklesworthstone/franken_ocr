//! Batched-sampler parity gate (bd-1azu.7 — the per-stream-greedy leaf of the
//! Phase-6 continuous-batch decode spine, bd-1azu).
//!
//! The batched sampler stacks `B` in-flight page-streams' `[1, vocab]` lm_head
//! rows into one `[B, vocab]` logits matrix and returns one greedy token id per
//! stream — each stream carrying its OWN generated-token history so the custom
//! sliding-window no-repeat-n-gram blocker ([SPEC-102/103]) bans DIFFERENT tokens
//! per stream off otherwise-identical logits. The whole spine rests on ONE
//! invariant (Doctrine #1, correctness > speed):
//!
//!   `batched_sample(stack(rows), [hist_0..hist_B], params)[s]`
//!       == `sample(row_s as [1, vocab], hist_s, params)`   — byte-for-byte.
//!
//! Greedy argmax + the ngram ban is a per-row reduction with no cross-stream
//! interaction, so batching is lossless by construction; this file is the
//! executing proof. The headline test builds several streams with DISTINCT
//! histories that ACTUALLY trigger the ban (and one boundary stream where the ban
//! does NOT fire), and proves each stream's batched id equals its single-stream
//! id — AND that the ngram ban genuinely changed the pick for the ban streams
//! (else the parity check would be vacuous). A randomized sweep then hammers the
//! same equality over many shapes/histories.

use franken_ocr::native_engine::sampler::{
    DecodeParams, batched_decode_step, batched_sample, decode_step, sample,
};
use franken_ocr::native_engine::tensor::Mat;

/// Deterministic xorshift64 PRNG — reproducible, no dev-dependency (the repo's
/// test idiom).
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
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// A small finite logit in [-8, 8) in 0.5 steps (ties land on lattice points,
    /// exercising the lowest-index tie-break shared by both paths).
    fn logit(&mut self) -> f32 {
        (self.below(32) as f32) * 0.5 - 8.0
    }
}

/// Stack `B` per-stream `vocab`-length rows into one `[B, vocab]` logits matrix —
/// row `s` is stream `s`'s logits, exactly what the batched sampler consumes.
fn stack(rows: &[Vec<f32>], vocab: usize) -> Mat {
    let mut data = Vec::with_capacity(rows.len() * vocab);
    for r in rows {
        assert_eq!(r.len(), vocab, "row width must equal vocab");
        data.extend_from_slice(r);
    }
    Mat::from_vec(rows.len(), vocab, data)
}

/// Build stream `s`'s logits row back out of a stacked matrix as a standalone
/// `[1, vocab]` [`Mat`] — the single-stream oracle's input.
fn row_mat(stacked: &Mat, s: usize) -> Mat {
    Mat::from_vec(1, stacked.cols, stacked.row(s).to_vec())
}

const NGRAM: usize = 35;
const PREFIX_LEN: usize = NGRAM - 1;

/// A history that, under a window ≥ `total_len`, makes the 35-gram
/// `[prefix.., banned]` repeat: the last `PREFIX_LEN` tokens equal `prefix`, so
/// the custom blocker bans `banned` from completing the n-gram. Mirrors the
/// in-crate `repeat_35gram_sequence` fixture but parameterized by `prefix_base`
/// and `banned` so each stream's history is genuinely DISTINCT and bans a
/// DIFFERENT token. Filler `99` and the `[prefix_base, prefix_base+34)` block are
/// kept disjoint from `banned`/fallback so the ONLY in-window ban is `banned`.
fn repeat_ngram_history(prefix_base: u32, banned: u32, total_len: usize) -> Vec<u32> {
    let prefix: Vec<u32> = (prefix_base..prefix_base + PREFIX_LEN as u32).collect();
    let min_len = PREFIX_LEN + 1 + PREFIX_LEN;
    assert!(
        total_len >= min_len,
        "total_len {total_len} < min {min_len}"
    );
    let mut seq = Vec::with_capacity(total_len);
    seq.extend_from_slice(&prefix);
    seq.push(banned);
    seq.extend(std::iter::repeat_n(99u32, total_len - min_len));
    seq.extend_from_slice(&prefix);
    seq
}

/// One headline stream: its history, its logits row, and whether we expect the
/// ngram ban to CHANGE the greedy pick (i.e. the ban fires in-window).
struct Case {
    history: Vec<u32>,
    logits: Vec<f32>,
    ban_changes_pick: bool,
    /// The id the single-stream path should pick under the live ngram params.
    expect: u32,
}

/// Logits where `top` is the raw argmax (10.0) and `runner_up` is the fallback
/// (9.0) once `top` is banned; everything else 0.
fn top_then_runner_up(vocab: usize, top: u32, runner_up: u32) -> Vec<f32> {
    let mut row = vec![0.0f32; vocab];
    row[top as usize] = 10.0;
    row[runner_up as usize] = 9.0;
    row
}

/// The headline mixed batch: ban-firing streams (distinct prefixes/banned tokens
/// and distinct sequence lengths), a window-boundary stream where the ban does
/// NOT fire, a too-short stream, and an EOS-picking stream.
fn headline_cases(vocab: usize) -> Vec<Case> {
    vec![
        // Stream 0: bans 7 (window 128 covers the whole len) -> picks 6.
        Case {
            history: repeat_ngram_history(20, 7, 69),
            logits: top_then_runner_up(vocab, 7, 6),
            ban_changes_pick: true,
            expect: 6,
        },
        // Stream 1: distinct prefix/banned, mid length -> bans 8, picks 7.
        Case {
            history: repeat_ngram_history(200, 8, 100),
            logits: top_then_runner_up(vocab, 8, 7),
            ban_changes_pick: true,
            expect: 7,
        },
        // Stream 2: at the exact window edge (len 128) -> bans 9, picks 8.
        Case {
            history: repeat_ngram_history(400, 9, 128),
            logits: top_then_runner_up(vocab, 9, 8),
            ban_changes_pick: true,
            expect: 8,
        },
        // Stream 3: ONE token past the window (len 129) -> ban does NOT fire under
        // window 128, raw argmax 10 stands.
        Case {
            history: repeat_ngram_history(600, 10, 129),
            logits: top_then_runner_up(vocab, 10, 9),
            ban_changes_pick: false,
            expect: 10,
        },
        // Stream 4: history far shorter than ngram_size -> no ban, raw argmax 5.
        Case {
            history: vec![0, 1, 2],
            logits: {
                let mut row = vec![0.0f32; vocab];
                row[5] = 7.0;
                row
            },
            ban_changes_pick: false,
            expect: 5,
        },
        // Stream 5: short history, logits favor EOS (id 1) -> picks 1, is_eos.
        Case {
            history: vec![3, 4, 5],
            logits: {
                let mut row = vec![0.0f32; vocab];
                row[1] = 6.0;
                row
            },
            ban_changes_pick: false,
            expect: 1,
        },
    ]
}

#[test]
fn batched_equals_single_stream_on_distinct_ban_histories() {
    let vocab = 1024usize;
    let cases = headline_cases(vocab);
    let params = DecodeParams::single_image(); // ngram 35, window 128, greedy.
    let no_ngram = DecodeParams {
        no_repeat_ngram_size: 0,
        ngram_window: 0,
        ..DecodeParams::single_image()
    };

    let rows: Vec<Vec<f32>> = cases.iter().map(|c| c.logits.clone()).collect();
    let stacked = stack(&rows, vocab);
    let histories: Vec<&[u32]> = cases.iter().map(|c| c.history.as_slice()).collect();

    // ONE batched call over all streams.
    let batched = batched_sample(&stacked, &histories, &params).expect("batched_sample");
    assert_eq!(batched.len(), cases.len());

    let mut any_ban_fired = false;
    for (s, case) in cases.iter().enumerate() {
        // Single-stream oracle: stream s's row run ALONE with stream s's history.
        let single = sample(&row_mat(&stacked, s), case.history.as_slice(), &params)
            .expect("single-stream sample");

        // The invariant: batched id == single-stream id, byte-for-byte.
        assert_eq!(
            batched[s], single,
            "stream {s}: batched id {} != single-stream id {single}",
            batched[s]
        );

        // And the explicit expected id (locks the fixture's intent).
        assert_eq!(
            batched[s], case.expect,
            "stream {s}: batched id {} != expected {}",
            batched[s], case.expect
        );

        // Non-vacuity: prove the ngram ban actually MOVED the pick for the ban
        // streams (vs the same row with the blocker disabled), and left the
        // non-ban streams untouched.
        let raw = sample(&row_mat(&stacked, s), case.history.as_slice(), &no_ngram)
            .expect("no-ngram sample");
        if case.ban_changes_pick {
            any_ban_fired = true;
            assert_ne!(
                batched[s], raw,
                "stream {s}: ngram ban was supposed to change the pick but raw==banned ({raw})"
            );
        } else {
            assert_eq!(
                batched[s], raw,
                "stream {s}: no ban expected but blocked pick {} != raw {raw}",
                batched[s]
            );
        }
    }
    assert!(
        any_ban_fired,
        "fixture must trigger the ngram ban for at least one stream (else parity is vacuous)"
    );

    // Histories must be genuinely distinct across the streams (the whole point is
    // per-stream history-dependent banning).
    for (i, ci) in cases.iter().enumerate() {
        for cj in cases.iter().skip(i + 1) {
            assert_ne!(
                ci.history, cj.history,
                "stream {i} must carry a DISTINCT history from later streams"
            );
        }
    }
}

#[test]
fn batched_decode_step_equals_single_stream_decode_step() {
    let vocab = 1024usize;
    let cases = headline_cases(vocab);
    let params = DecodeParams::single_image();

    let rows: Vec<Vec<f32>> = cases.iter().map(|c| c.logits.clone()).collect();
    let stacked = stack(&rows, vocab);
    let histories: Vec<&[u32]> = cases.iter().map(|c| c.history.as_slice()).collect();

    let batched = batched_decode_step(&stacked, &histories, &params).expect("batched_decode_step");
    assert_eq!(batched.len(), cases.len());

    let mut saw_eos = false;
    for (s, case) in cases.iter().enumerate() {
        let single = decode_step(&row_mat(&stacked, s), case.history.as_slice(), &params)
            .expect("single decode_step");
        assert_eq!(
            batched[s], single,
            "stream {s}: batched DecodeOutput != single-stream DecodeOutput"
        );
        saw_eos |= batched[s].is_eos;
    }
    // The EOS stream (id 1) must have flagged is_eos through the batched path.
    assert!(
        saw_eos,
        "batched path must classify the EOS-picking stream as is_eos"
    );
}

#[test]
fn batched_respects_multi_image_window() {
    // The window parameter flows through batching: under the multi-image window
    // (1024) a len-1024 repeat still bans, but the SAME history under the
    // single-image window (128) does not see the far-back n-gram. Two streams with
    // identical histories but the contrast proved per-window.
    let vocab = 512usize;
    let history = repeat_ngram_history(20, 7, 1024);
    let logits = top_then_runner_up(vocab, 7, 6);
    let stacked = stack(&[logits], vocab);
    let histories: [&[u32]; 1] = [history.as_slice()];

    // Multi-image window 1024: len 1024 -> idx 0 is in-window -> ban 7 -> pick 6.
    let multi = batched_sample(&stacked, &histories, &DecodeParams::multi_image()).unwrap();
    let multi_single = sample(
        &row_mat(&stacked, 0),
        &history,
        &DecodeParams::multi_image(),
    )
    .unwrap();
    assert_eq!(multi[0], multi_single);
    assert_eq!(
        multi[0], 6,
        "len-1024 repeat must ban 7 under the multi window"
    );

    // Single-image window 128: the far-back n-gram is out of window -> no ban ->
    // raw argmax 7 stands.
    let single = batched_sample(&stacked, &histories, &DecodeParams::single_image()).unwrap();
    assert_eq!(
        single[0], 7,
        "len-1024 repeat is out of the 128 window -> raw argmax"
    );
}

#[test]
fn batched_matches_per_stream_over_random_sweep() {
    // Broad parity net: many random batches of random histories + logits, compared
    // batched-vs-per-stream both WITH the ngram blocker active and disabled.
    let mut rng = Rng(0x5a4b_3c2d_1e0f_9a8b);
    let vocab = 64usize;

    let active = DecodeParams {
        no_repeat_ngram_size: 3,
        ngram_window: 8,
        ..DecodeParams::default()
    };
    let disabled = DecodeParams {
        no_repeat_ngram_size: 0,
        ngram_window: 0,
        ..DecodeParams::default()
    };

    for trial in 0..64 {
        let b = 1 + (rng.below(12) as usize); // 1..=12 streams.

        let mut histories_owned: Vec<Vec<u32>> = Vec::with_capacity(b);
        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(b);
        for _ in 0..b {
            // Short, token-collision-prone histories so the ngram-3 blocker fires
            // often: tokens drawn from a tiny alphabet [0, 6).
            let hlen = rng.below(40) as usize;
            let history: Vec<u32> = (0..hlen).map(|_| rng.below(6) as u32).collect();
            histories_owned.push(history);
            rows.push((0..vocab).map(|_| rng.logit()).collect());
        }

        let stacked = stack(&rows, vocab);
        let histories: Vec<&[u32]> = histories_owned.iter().map(Vec::as_slice).collect();

        for params in [&active, &disabled] {
            let batched =
                batched_sample(&stacked, &histories, params).expect("batched_sample failed");
            assert_eq!(batched.len(), b);
            for (s, (&token, &hist)) in batched.iter().zip(histories.iter()).enumerate() {
                let single = sample(&row_mat(&stacked, s), hist, params)
                    .expect("single-stream sample failed");
                assert_eq!(
                    token, single,
                    "trial {trial} stream {s}: batched {token} != single {single} \
                     (ngram_size={})",
                    params.no_repeat_ngram_size
                );
            }
        }
    }
}

#[test]
fn batched_b_one_equals_single() {
    // Degenerate B=1: the batched API must reduce exactly to the single path.
    let vocab = 256usize;
    let history = repeat_ngram_history(30, 11, 80);
    let logits = top_then_runner_up(vocab, 11, 10);
    let stacked = stack(&[logits], vocab);
    let histories: [&[u32]; 1] = [history.as_slice()];
    let p = DecodeParams::single_image();
    let batched = batched_sample(&stacked, &histories, &p).unwrap();
    let single = sample(&row_mat(&stacked, 0), &history, &p).unwrap();
    assert_eq!(batched, vec![single]);
}

#[test]
fn batched_sample_rejects_history_count_mismatch() {
    let vocab = 8usize;
    let stacked = stack(&[vec![0.0; vocab], vec![0.0; vocab]], vocab); // B=2
    let one: [&[u32]; 1] = [&[]]; // only 1 history for 2 rows
    let err = batched_sample(&stacked, &one, &DecodeParams::single_image()).unwrap_err();
    assert!(
        err.to_string().contains("histories"),
        "mismatch error should mention histories, got: {err}"
    );
}

#[test]
fn batched_sample_rejects_malformed_backing_data() {
    // rows*cols disagrees with data length -> defensive error, no panic.
    let bad = Mat {
        rows: 2,
        cols: 4,
        data: vec![0.0, 1.0, 2.0], // len 3 != 8
    };
    let hist: [&[u32]; 2] = [&[], &[]];
    let err = batched_sample(&bad, &hist, &DecodeParams::single_image()).unwrap_err();
    assert!(
        err.to_string().contains("data len"),
        "should report a data-length mismatch, got: {err}"
    );
}

#[test]
fn batched_sample_propagates_temperature_not_implemented() {
    let vocab = 4usize;
    let stacked = stack(&[vec![1.0, 2.0, 3.0, 0.0], vec![0.0, 0.0, 0.0, 9.0]], vocab);
    let hist: [&[u32]; 2] = [&[], &[]];
    let p = DecodeParams {
        temperature: 0.7,
        ..DecodeParams::single_image()
    };
    // temperature>0 is outside the greedy spine; the per-stream sample errors and
    // batched_sample propagates it.
    assert!(batched_sample(&stacked, &hist, &p).is_err());
}
