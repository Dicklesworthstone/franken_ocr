//! bd-1azu.8 — continuous-batch decode scheduler (Lever-A keystone).
//!
//! The single sequential driver for the Phase-6 throughput spine. It holds up
//! to `B` in-flight [`PageStream`]s and, at every step, gathers EVERY active
//! stream's current decode hidden into ONE batched forward (composing the
//! Wave-1 batched kernels: [`super::decoder::batched_lm_head_i8`] →
//! [`super::sampler::batched_decode_step`] → [`super::decoder::batched_decode_step_i8`]).
//! Streams that hit EOS / `max_length` retire and are backfilled from a pending
//! queue so the active set stays saturated; outputs are RE-SORTED to input
//! order via [`PageStream::input_index`].
//!
//! ## Doctrine #5 (the heart)
//! The scheduler is the SINGLE sequential driver — at every step EXACTLY ONE
//! batched forward is live over the active streams (NOT `N` nested forwards).
//! It instruments [`SchedulerStats::max_concurrent_forwards`] (must stay `1`)
//! and [`SchedulerStats::guard_held_during_fanout`] (must stay `false`) so the
//! bd-1azu.14 watchdog gate can assert the one-live-forward / no-lock-held-during
//! -fan-out invariants. The model-cache `Mutex` is acquired ONCE and its guard
//! dropped before any fan-out — that wiring lives in the CLI driver (bd-1azu.11);
//! this module exposes the instrumentation hooks it reports through.
//!
//! ## Losslessness & kill-switch
//! Each stream's emitted tokens are identical to running that page through the
//! sequential `generate_cached_i8` path (the int8 GEMM contraction is
//! `M`-independent and the f32 attention is per-stream, never key-batched — see
//! bd-1azu.3/.5/.6/.7). The master kill-switch `FOCR_BATCH_SPINE=0`
//! ([`spine_enabled`]) reproduces the sequential driver byte-for-byte; with the
//! spine off this module is never entered, so it changes no production output.
//!
//! ## Scope boundary
//! This bead delivers the scheduler skeleton + the lossless stream lifecycle
//! (admit/retire/backfill/re-sort/one-live-forward), proven unconditionally by
//! the [`BatchStep`]-injected unit tests below. Co-batched **chunked-prefill
//! admission** is bd-1azu.9 and the **`focr ocr-batch` CLI rewire + end-to-end
//! parity** is bd-1azu.11/.13; the production [`DecoderBatchStep`] composes the
//! real forward so the kernels are wired and type-checked, with its end-to-end
//! positional/KV correctness proven by the model-gated bd-1azu.13 gate.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::decoder::{self, DecoderWeightCacheI8};
use super::rswa::BatchedRingCache;
use super::sampler::{self, DecodeParams};
use super::tensor::Mat;
use crate::error::FocrResult;

/// Default in-flight batch width when `FOCR_BATCH_SIZE` is unset.
pub const DEFAULT_BATCH_SIZE: usize = 128;
/// Hard cap on the in-flight batch width (memory / occupancy bound).
pub const MAX_BATCH_SIZE: usize = 256;

/// Parse a `FOCR_BATCH_SIZE` value into a usable batch width, clamped to
/// `[1, MAX_BATCH_SIZE]`; `None`/blank/unparsable → [`DEFAULT_BATCH_SIZE`].
/// Factored out (pure) so it is testable without mutating the process env.
#[must_use]
pub fn parse_batch_size(val: Option<&str>) -> usize {
    match val.and_then(|s| s.trim().parse::<usize>().ok()) {
        Some(0) | None => DEFAULT_BATCH_SIZE,
        Some(n) => n.min(MAX_BATCH_SIZE),
    }
}

/// The scheduler's in-flight batch width from `FOCR_BATCH_SIZE`.
#[must_use]
pub fn scheduler_batch_size() -> usize {
    parse_batch_size(std::env::var("FOCR_BATCH_SIZE").ok().as_deref())
}

/// Master kill-switch: `FOCR_BATCH_SPINE` (shared with the decoder). When this
/// is `false` the CLI must drive the sequential path and never enter the spine.
#[must_use]
pub fn spine_enabled() -> bool {
    decoder::batch_spine_enabled()
}

/// Env kill-switch for Lever-F submodular batch-packing. When PRESENT (any
/// value, even empty) the scheduler admits pending streams in length-grouped
/// order — similar [`PageStream::prefill_len`] co-batched — so each batch window
/// has lower length variance and fewer streams idle while a longer neighbour
/// keeps decoding (cutting wasted decode steps). UNSET reproduces today's exact
/// submission-order admission, byte-for-byte.
///
/// LOSSLESS: packing is a PURE REORDERING of *which* streams share a batch. The
/// active-set cap is unchanged and [`BatchScheduler::run`] re-sorts outputs to
/// [`PageStream::input_index`]; each stream's tokens depend only on its OWN state
/// (the batched kernels are `M`-independent / per-stream — see module docs), so
/// no stream's emitted token sequence changes.
const BATCH_PACK_ENV: &str = "FOCR_BATCH_PACK";

/// Whether Lever-F batch-packing is armed ([`BATCH_PACK_ENV`], read ONCE into a
/// process-wide bool — never touched per-stream).
#[must_use]
pub fn batch_pack_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os(BATCH_PACK_ENV).is_some())
}

/// Compute the pending-admission order over `streams`.
///
/// With `pack` OFF this is submission order `0..streams.len()` — today's exact
/// behaviour. With `pack` ON (Lever-F) it is a STABLE sort of the stream indices
/// by [`PageStream::prefill_len`] (ties keep submission order), so similar-length
/// pages are admitted — and thus co-batched — together, shrinking each batch
/// window's length variance. Either way the result is a BIJECTION over
/// `0..streams.len()`: every stream appears exactly once, so admission is only
/// reordered, never adding / dropping / duplicating a stream.
#[must_use]
fn admission_order(streams: &[PageStream], pack: bool) -> Vec<usize> {
    let mut order: Vec<usize> = (0..streams.len()).collect();
    if pack {
        // Stable sort: equal `prefill_len` keep submission order, so the unset
        // path is the identity and packing is a minimal length-grouping perturbation.
        order.sort_by_key(|&i| streams[i].prefill_len);
    }
    order
}

/// One in-flight page-stream: its decode state plus the slot of identity needed
/// to restore input order. Per-layer KV lives in the shared [`BatchedRingCache`]
/// (indexed by the stream's active slot), never duplicated here.
#[derive(Debug, Clone)]
pub struct PageStream {
    /// Position of this page in the caller's input order (outputs re-sort by it).
    pub input_index: usize,
    /// Number of reference/prompt tokens already in the KV cache (prefill length).
    pub prefill_len: usize,
    /// Absolute position fed to the next decode-step advance (`prefill_len` +
    /// emitted-so-far); mirrors the sequential loop's `prefill_len + (emitted-1)`.
    pub position: usize,
    /// Full token history (prompt + emitted) — the n-gram-ban context.
    pub generated: Vec<u32>,
    /// Count of prompt tokens, so emitted = `generated[prompt_len..]`.
    pub prompt_len: usize,
    /// Current decode hidden `[1, hidden]` the next token is predicted from
    /// (seeded from prefill's last hidden, then replaced each advance).
    pub last_hidden: Mat,
    /// Retired (EOS or `max_length` reached).
    pub done: bool,
    /// Retired specifically because EOS was emitted.
    pub eos: bool,
}

impl PageStream {
    /// New stream seeded from prefill: `last_hidden` is prefill's final row,
    /// `prompt_ids` is the reference/prompt context already in the KV cache.
    #[must_use]
    pub fn new(
        input_index: usize,
        prefill_len: usize,
        prompt_ids: &[u32],
        last_hidden: Mat,
    ) -> Self {
        Self {
            input_index,
            prefill_len,
            position: prefill_len,
            generated: prompt_ids.to_vec(),
            prompt_len: prompt_ids.len(),
            last_hidden,
            done: false,
            eos: false,
        }
    }

    /// Tokens emitted by decode (excludes the prompt/reference context).
    #[must_use]
    pub fn emitted(&self) -> &[u32] {
        &self.generated[self.prompt_len..]
    }
}

/// Read-only view of one active stream handed to a [`BatchStep`].
pub struct StreamSlot<'a> {
    /// Stable index of this stream into the submitted `streams` vector — and thus
    /// into any per-stream side table the step keeps (e.g. the cache stream slot
    /// of the [`super::rswa::BatchedRingCache`]). Unlike the active-set position,
    /// this never shifts as streams retire/backfill, so the step can address the
    /// correct (full-cache) stream even after the active set shrinks (bd-1azu.11).
    pub slot_index: usize,
    /// Token history (prompt + emitted) for n-gram banning.
    pub history: &'a [u32],
    /// Current decode hidden `[1, hidden]`.
    pub hidden: &'a Mat,
    /// Absolute decode position for this step's advance.
    pub position: usize,
}

/// Per-stream result of ONE batched forward step.
pub struct StreamOut {
    /// The next token sampled for this stream.
    pub token: u32,
    /// Whether `token` is EOS (the scheduler retires the stream after appending).
    pub is_eos: bool,
    /// The stream's hidden after advancing one decode step (ignored when `is_eos`).
    pub new_hidden: Mat,
}

/// ONE batched forward over all active streams, dependency-injected so the
/// scheduler's lifecycle logic is unit-testable without the 6.67 GB weights.
/// Implementations MUST treat the call as a single live forward (the scheduler
/// accounts `max_concurrent_forwards` around it).
pub trait BatchStep {
    /// Sample the next token for each active stream from its current hidden and
    /// advance each one decode step. `slots[k]` ⇄ the returned `Vec`'s index `k`.
    ///
    /// # Errors
    /// Propagates any forward/sampling error from the underlying kernels.
    fn step(&mut self, slots: &[StreamSlot<'_>]) -> FocrResult<Vec<StreamOut>>;
}

/// One-live-forward / lock-discipline instrumentation snapshot for the
/// bd-1azu.14 watchdog gate.
#[derive(Debug, Clone, Copy)]
pub struct SchedulerStats {
    /// Peak number of simultaneously-live batched forwards (MUST be `1`).
    pub max_concurrent_forwards: usize,
    /// Whether the model-cache guard was ever reported held during fan-out
    /// (MUST be `false`).
    pub guard_held_during_fanout: bool,
    /// Peak size of the active set (MUST be `<= batch_size`).
    pub max_active: usize,
    /// Total batched forward steps executed.
    pub total_steps: usize,
}

/// Continuous-batch decode scheduler. See module docs.
pub struct BatchScheduler {
    batch_size: usize,
    max_length: usize,
    live_forwards: AtomicUsize,
    max_forwards: AtomicUsize,
    max_active: AtomicUsize,
    guard_violation: AtomicBool,
    steps: usize,
}

impl BatchScheduler {
    /// Construct with an explicit in-flight width and per-stream emission cap.
    /// `batch_size` is clamped to `[1, MAX_BATCH_SIZE]`.
    #[must_use]
    pub fn new(batch_size: usize, max_length: usize) -> Self {
        Self {
            batch_size: batch_size.clamp(1, MAX_BATCH_SIZE),
            max_length,
            live_forwards: AtomicUsize::new(0),
            max_forwards: AtomicUsize::new(0),
            max_active: AtomicUsize::new(0),
            guard_violation: AtomicBool::new(false),
            steps: 0,
        }
    }

    /// Construct from `FOCR_BATCH_SIZE` with the given per-stream emission cap.
    #[must_use]
    pub fn from_env(max_length: usize) -> Self {
        Self::new(scheduler_batch_size(), max_length)
    }

    /// The in-flight batch width.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Report (from the driver) that the model-cache guard was held during a
    /// fan-out — a Doctrine-#5 violation the watchdog gate must catch.
    pub fn note_guard_held_during_fanout(&self) {
        self.guard_violation.store(true, Ordering::SeqCst);
    }

    /// Instrumentation snapshot.
    #[must_use]
    pub fn stats(&self) -> SchedulerStats {
        SchedulerStats {
            max_concurrent_forwards: self.max_forwards.load(Ordering::SeqCst),
            guard_held_during_fanout: self.guard_violation.load(Ordering::SeqCst),
            max_active: self.max_active.load(Ordering::SeqCst),
            total_steps: self.steps,
        }
    }

    /// Drive `streams` to completion through `step`, returning each stream's
    /// emitted tokens RE-SORTED to [`PageStream::input_index`] order.
    ///
    /// Continuous batching: up to `batch_size` streams are active in any single
    /// forward; a stream retires on EOS or `max_length` and the active set is
    /// backfilled from the pending tail (admission of *new* pages mid-batch is
    /// bd-1azu.9). When the [`FOCR_BATCH_PACK`](BATCH_PACK_ENV) kill-switch is
    /// armed, pending streams are admitted in length-grouped order (Lever-F) so
    /// similar-`prefill_len` pages co-batch — a PURE reordering of admission.
    /// LOSSLESS either way: each stream's tokens equal its single-stream sequence
    /// (the step composes `M`-independent kernels) and the output is re-sorted to
    /// input order regardless of which streams shared a batch.
    ///
    /// # Errors
    /// Propagates the first [`BatchStep`] error.
    pub fn run<S: BatchStep>(
        &mut self,
        streams: Vec<PageStream>,
        step: &mut S,
    ) -> FocrResult<Vec<Vec<u32>>> {
        self.run_with_pack(streams, step, batch_pack_enabled())
    }

    /// Core of [`run`](Self::run) with the Lever-F packing decision injected, so
    /// the lossless "packing only reorders admission" property is unit-testable
    /// without mutating the process env. `pack == false` is today's exact
    /// submission-order driver, byte-for-byte.
    fn run_with_pack<S: BatchStep>(
        &mut self,
        mut streams: Vec<PageStream>,
        step: &mut S,
        pack: bool,
    ) -> FocrResult<Vec<Vec<u32>>> {
        // Indices into `streams`. Admission order is submission order, or
        // length-grouped when Lever-F packing is armed; either way it is a
        // BIJECTION over the streams (see [`admission_order`]). PURE reordering of
        // admission — the active-set cap below and the final input-order re-sort
        // are unchanged, so no stream's tokens change. Active = currently
        // decoding; pending = awaiting an open slot.
        let mut pending: VecDeque<usize> = admission_order(&streams, pack).into();
        let mut active: Vec<usize> = Vec::with_capacity(self.batch_size);
        Self::admit(&mut active, &mut pending, self.batch_size);

        while !active.is_empty() {
            self.max_active.fetch_max(active.len(), Ordering::SeqCst);

            // Build the read-only slot views for THIS single forward.
            let slots: Vec<StreamSlot<'_>> = active
                .iter()
                .map(|&i| StreamSlot {
                    slot_index: i,
                    history: &streams[i].generated,
                    hidden: &streams[i].last_hidden,
                    position: streams[i].position,
                })
                .collect();

            // ── exactly one live forward over the active set ──
            let fwd = self.enter_forward();
            let result = step.step(&slots);
            drop(fwd);
            self.exit_forward();
            let outs = result?;
            self.steps += 1;

            debug_assert_eq!(outs.len(), active.len(), "one StreamOut per active stream");

            // Apply outputs; collect retirements (active-vector positions).
            let mut retire: Vec<usize> = Vec::new();
            for (k, (&i, out)) in active.iter().zip(outs).enumerate() {
                let s = &mut streams[i];
                s.generated.push(out.token);
                s.position += 1;
                if out.is_eos || s.emitted().len() >= self.max_length {
                    s.done = true;
                    s.eos = out.is_eos;
                    retire.push(k);
                } else {
                    s.last_hidden = out.new_hidden;
                }
            }

            // Remove retired (high→low so indices stay valid), then backfill.
            for &k in retire.iter().rev() {
                active.remove(k);
            }
            Self::admit(&mut active, &mut pending, self.batch_size);
        }

        // Re-sort emitted tokens to input order.
        let mut out: Vec<(usize, Vec<u32>)> = streams
            .into_iter()
            .map(|s| (s.input_index, s.emitted().to_vec()))
            .collect();
        out.sort_by_key(|(idx, _)| *idx);
        Ok(out.into_iter().map(|(_, toks)| toks).collect())
    }

    /// Fill the active set up to `cap` from the pending front.
    fn admit(active: &mut Vec<usize>, pending: &mut VecDeque<usize>, cap: usize) {
        while active.len() < cap {
            match pending.pop_front() {
                Some(i) => active.push(i),
                None => break,
            }
        }
    }

    fn enter_forward(&self) -> super::ForwardPass {
        let n = self.live_forwards.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_forwards.fetch_max(n, Ordering::SeqCst);
        // Also joins the PROCESS-WIDE one-live-forward gauge (bd-1azu.14), so a
        // vision/prefill forward concurrent with a decode step trips the
        // watchdog even though this scheduler's local counter can't see it.
        super::enter_forward()
    }

    fn exit_forward(&self) {
        self.live_forwards.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The production [`BatchStep`]: composes the Wave-1 batched kernels over the
/// shared [`BatchedRingCache`]. It samples each active stream's next token from
/// its current hidden ([`decoder::batched_lm_head_i8`] →
/// [`sampler::batched_decode_step`]) then advances every stream one decode step
/// ([`decoder::batched_decode_step_i8`]).
///
/// Construction (weight cache + prefill-seeded rings) and the `focr ocr-batch`
/// driving land in bd-1azu.9/.11; the end-to-end positional/KV correctness of
/// this composition is proven by the model-gated bd-1azu.13 parity gate. The
/// advance addresses each active stream's OWN cache slot via
/// [`StreamSlot::slot_index`] through [`decoder::batched_decode_step_i8_streams`],
/// so once a stream retires it is pruned from the cache fan-out (bd-1azu.11) — the
/// active set is a SUBSET of the cache's streams, never required to equal it. Note
/// a stream that emits EOS on a given step is still advanced THAT step (its
/// discarded KV write is per-stream-isolated and harmless); it is excluded from
/// every subsequent step.
pub struct DecoderBatchStep<'a> {
    /// Int8 decoder weight cache (dequant-once).
    pub wc: &'a DecoderWeightCacheI8,
    /// Per-stream R-SWA KV rings, indexed by active slot.
    pub caches: &'a mut BatchedRingCache,
    /// `model.embed_tokens.weight` `[vocab, hidden]` for token embedding.
    pub embed_table: &'a Mat,
    /// Decode sampling parameters (greedy / EOS / n-gram ban).
    pub params: &'a DecodeParams,
}

impl BatchStep for DecoderBatchStep<'_> {
    fn step(&mut self, slots: &[StreamSlot<'_>]) -> FocrResult<Vec<StreamOut>> {
        let hidden_dim = self.embed_table.cols;
        let vocab = self.embed_table.rows;

        // 1. Per-stream logits from the current hiddens (M=B lm_head).
        let hiddens: Vec<Mat> = slots.iter().map(|s| s.hidden.clone()).collect();
        let logits_rows = decoder::batched_lm_head_i8(self.wc, &hiddens)?;

        // 2. Stack into a single [B, vocab] block for the batched sampler.
        let b = slots.len();
        let mut stacked = Vec::with_capacity(b * vocab);
        for row in &logits_rows {
            stacked.extend_from_slice(&row.data);
        }
        let logits = Mat::from_vec(b, vocab, stacked);
        let histories: Vec<&[u32]> = slots.iter().map(|s| s.history).collect();
        let decoded = sampler::batched_decode_step(&logits, &histories, self.params)?;

        // 3. Advance ONLY the active streams one decode step (M=active projection
        //    + attention), addressing each one's OWN cache slot via `slot_index`
        //    so retired streams are pruned from the cache fan-out (bd-1azu.11).
        let mut token_embeds: Vec<Mat> = Vec::with_capacity(b);
        let mut positions: Vec<usize> = Vec::with_capacity(b);
        let mut stream_ids: Vec<usize> = Vec::with_capacity(b);
        for (out, slot) in decoded.iter().zip(slots.iter()) {
            let t = out.token_id as usize;
            if t >= vocab {
                return Err(crate::error::FocrError::Other(anyhow::anyhow!(
                    "batch_scheduler::DecoderBatchStep: token id {t} outside embed vocab {vocab}"
                )));
            }
            let row = self.embed_table.data[t * hidden_dim..(t + 1) * hidden_dim].to_vec();
            token_embeds.push(Mat::from_vec(1, hidden_dim, row));
            positions.push(slot.position);
            stream_ids.push(slot.slot_index);
        }
        let new_hiddens = decoder::batched_decode_step_i8_streams(
            self.wc,
            self.caches,
            &stream_ids,
            &token_embeds,
            &positions,
        )?;

        // 4. Assemble per-stream results.
        Ok(decoded
            .into_iter()
            .zip(new_hiddens)
            .map(|(out, new_hidden)| StreamOut {
                token: out.token_id,
                is_eos: out.is_eos,
                new_hidden,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A weights-free [`BatchStep`] that exercises the scheduler's lifecycle.
    /// Each stream's `last_hidden` encodes `[stream_tag, eos_after]`; the mock
    /// emits `token = history_len` (so a stream emits 0,1,2,…) and signals EOS on
    /// the `eos_after`-th emitted token. `eos_after == 0` ⇒ never EOS (so the
    /// scheduler's `max_length` cap is what retires it).
    struct MockStep {
        /// Records the active-set size seen on each call (one-live-forward proof).
        batch_sizes: Vec<usize>,
    }

    impl MockStep {
        fn new() -> Self {
            Self {
                batch_sizes: Vec::new(),
            }
        }
    }

    impl BatchStep for MockStep {
        fn step(&mut self, slots: &[StreamSlot<'_>]) -> FocrResult<Vec<StreamOut>> {
            self.batch_sizes.push(slots.len());
            Ok(slots
                .iter()
                .map(|s| {
                    let tag = s.hidden.data[0];
                    let eos_after = s.hidden.data[1] as usize;
                    let emitted_before = s.history.len(); // prompt is empty in tests
                    let token = emitted_before as u32;
                    let is_eos = eos_after != 0 && emitted_before + 1 >= eos_after;
                    StreamOut {
                        token,
                        is_eos,
                        // carry the [tag, eos_after] identity forward
                        new_hidden: Mat::from_vec(1, 2, vec![tag, eos_after as f32]),
                    }
                })
                .collect())
        }
    }

    /// Build a stream whose mock will emit `eos_after` tokens (0 ⇒ unbounded).
    fn stream(input_index: usize, eos_after: usize) -> PageStream {
        let hidden = Mat::from_vec(1, 2, vec![input_index as f32, eos_after as f32]);
        PageStream::new(
            input_index,
            /*prefill_len*/ 4,
            /*prompt*/ &[],
            hidden,
        )
    }

    #[test]
    fn parse_batch_size_clamps_and_defaults() {
        assert_eq!(parse_batch_size(None), DEFAULT_BATCH_SIZE);
        assert_eq!(parse_batch_size(Some("")), DEFAULT_BATCH_SIZE);
        assert_eq!(parse_batch_size(Some("garbage")), DEFAULT_BATCH_SIZE);
        assert_eq!(parse_batch_size(Some("0")), DEFAULT_BATCH_SIZE);
        assert_eq!(parse_batch_size(Some("1")), 1);
        assert_eq!(parse_batch_size(Some("64")), 64);
        assert_eq!(parse_batch_size(Some("256")), MAX_BATCH_SIZE);
        assert_eq!(parse_batch_size(Some("100000")), MAX_BATCH_SIZE);
    }

    #[test]
    fn new_clamps_batch_size() {
        assert_eq!(BatchScheduler::new(0, 16).batch_size(), 1);
        assert_eq!(BatchScheduler::new(9999, 16).batch_size(), MAX_BATCH_SIZE);
        assert_eq!(BatchScheduler::new(8, 16).batch_size(), 8);
    }

    #[test]
    fn each_stream_emits_exactly_eos_after_tokens() {
        let mut sched = BatchScheduler::new(4, 100);
        let streams = vec![stream(0, 3), stream(1, 1), stream(2, 5)];
        let mut step = MockStep::new();
        let out = sched.run(streams, &mut step).expect("run");
        // token sequence is 0,1,2,…; EOS token IS included in emitted output.
        assert_eq!(out[0], vec![0, 1, 2]);
        assert_eq!(out[1], vec![0]);
        assert_eq!(out[2], vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn outputs_resorted_to_input_order_despite_shuffled_submission() {
        let mut sched = BatchScheduler::new(8, 100);
        // Submission order is NOT input order: input_index = 2, 0, 1.
        let streams = vec![stream(2, 2), stream(0, 4), stream(1, 1)];
        let mut step = MockStep::new();
        let out = sched.run(streams, &mut step).expect("run");
        // Re-sorted to input_index 0,1,2.
        assert_eq!(out[0], vec![0, 1, 2, 3]); // input 0 had eos_after 4
        assert_eq!(out[1], vec![0]); // input 1 had eos_after 1
        assert_eq!(out[2], vec![0, 1]); // input 2 had eos_after 2
    }

    #[test]
    fn one_live_forward_and_guard_not_held() {
        let mut sched = BatchScheduler::new(4, 100);
        let streams = vec![stream(0, 3), stream(1, 4), stream(2, 2)];
        let mut step = MockStep::new();
        let _ = sched.run(streams, &mut step).expect("run");
        let st = sched.stats();
        assert_eq!(
            st.max_concurrent_forwards, 1,
            "exactly one live forward per step"
        );
        assert!(
            !st.guard_held_during_fanout,
            "model-cache guard never held during fan-out"
        );
    }

    #[test]
    fn active_set_never_exceeds_batch_size_and_backfills() {
        // 5 streams, width 2 → the active set must stay <= 2 and still complete all.
        let mut sched = BatchScheduler::new(2, 100);
        let streams = vec![
            stream(0, 1),
            stream(1, 3),
            stream(2, 2),
            stream(3, 1),
            stream(4, 4),
        ];
        let mut step = MockStep::new();
        let out = sched.run(streams, &mut step).expect("run");
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), 1);
        assert_eq!(out[1].len(), 3);
        assert_eq!(out[2].len(), 2);
        assert_eq!(out[3].len(), 1);
        assert_eq!(out[4].len(), 4);
        let st = sched.stats();
        assert!(
            st.max_active <= 2,
            "active set bounded by batch_size (got {})",
            st.max_active
        );
        assert_eq!(st.max_concurrent_forwards, 1);
        // every mock call saw <= 2 active streams
        assert!(step.batch_sizes.iter().all(|&n| n <= 2));
    }

    #[test]
    fn batch_size_one_serializes_all_streams_correctly() {
        let mut sched = BatchScheduler::new(1, 100);
        let streams = vec![stream(0, 2), stream(1, 3)];
        let mut step = MockStep::new();
        let out = sched.run(streams, &mut step).expect("run");
        assert_eq!(out[0], vec![0, 1]);
        assert_eq!(out[1], vec![0, 1, 2]);
        let st = sched.stats();
        assert!(st.max_active <= 1);
        assert!(step.batch_sizes.iter().all(|&n| n == 1));
    }

    #[test]
    fn max_length_retires_unbounded_streams() {
        // eos_after = 0 ⇒ the mock never signals EOS; max_length must retire it.
        let mut sched = BatchScheduler::new(4, 5);
        let streams = vec![stream(0, 0), stream(1, 2)];
        let mut step = MockStep::new();
        let out = sched.run(streams, &mut step).expect("run");
        assert_eq!(out[0].len(), 5, "unbounded stream capped at max_length");
        assert_eq!(out[0], vec![0, 1, 2, 3, 4]);
        assert_eq!(out[1], vec![0, 1]);
    }

    #[test]
    fn empty_input_yields_empty_output() {
        let mut sched = BatchScheduler::new(4, 100);
        let mut step = MockStep::new();
        let out = sched.run(Vec::new(), &mut step).expect("run");
        assert!(out.is_empty());
        assert_eq!(sched.stats().total_steps, 0);
    }

    #[test]
    fn note_guard_held_is_observable() {
        let sched = BatchScheduler::new(2, 10);
        assert!(!sched.stats().guard_held_during_fanout);
        sched.note_guard_held_during_fanout();
        assert!(sched.stats().guard_held_during_fanout);
    }

    /// A [`BatchStep`] that records the `slot_index` of every active slot it sees,
    /// step by step. Behaves like [`MockStep`] (emits `0,1,2,…`, EOS on the
    /// `eos_after`-th token) but its purpose is to prove the bd-1azu.11 plumbing:
    /// `slot_index` is the STABLE stream identity (the cache slot the production
    /// `DecoderBatchStep` addresses), NOT the shifting active-set position.
    struct SlotTraceStep {
        traces: Vec<Vec<usize>>,
    }

    impl BatchStep for SlotTraceStep {
        fn step(&mut self, slots: &[StreamSlot<'_>]) -> FocrResult<Vec<StreamOut>> {
            self.traces
                .push(slots.iter().map(|s| s.slot_index).collect());
            Ok(slots
                .iter()
                .map(|s| {
                    let eos_after = s.hidden.data[1] as usize;
                    let emitted_before = s.history.len();
                    let is_eos = eos_after != 0 && emitted_before + 1 >= eos_after;
                    StreamOut {
                        token: emitted_before as u32,
                        is_eos,
                        new_hidden: s.hidden.clone(),
                    }
                })
                .collect())
        }
    }

    /// `slot_index` must identify the stream's STABLE submission slot (the cache
    /// index) across BOTH retirement (active vector shrinks) and backfill (a
    /// pending stream is admitted into a freed slot), never collapsing to the
    /// active-set position. This is the invariant the production
    /// `DecoderBatchStep` relies on to address each active stream's OWN
    /// `BatchedRingCache` slot after others retire.
    #[test]
    fn slot_index_identifies_cache_stream_through_retire_and_backfill() {
        // width 2, 4 streams → forces both retirement and mid-batch backfill.
        let mut sched = BatchScheduler::new(2, 100);
        let streams = vec![stream(0, 1), stream(1, 2), stream(2, 1), stream(3, 2)];
        let mut step = SlotTraceStep { traces: Vec::new() };
        let out = sched.run(streams, &mut step).expect("run");

        // Each stream still emits exactly its own token count, re-sorted to input.
        assert_eq!(out[0].len(), 1);
        assert_eq!(out[1].len(), 2);
        assert_eq!(out[2].len(), 1);
        assert_eq!(out[3].len(), 2);

        // The slot_index trace is the stable stream identity at each step:
        //  step 1: streams 0,1 active           → [0, 1]
        //  step 2: 0 retired, 2 backfilled       → [1, 2]
        //  step 3: 1 & 2 retired, 3 backfilled   → [3]
        //  step 4: 3's final token               → [3]
        assert_eq!(
            step.traces,
            vec![vec![0usize, 1], vec![1, 2], vec![3], vec![3]],
            "slot_index must track stable stream identity, not active position"
        );
    }

    // ── Lever-F: submodular batch-packing (FOCR_BATCH_PACK) ──────────────────

    /// Like [`stream`] but with an explicit `prefill_len` — the packing sort key.
    /// `eos_after` still drives how many tokens the mock emits.
    fn stream_len(input_index: usize, prefill_len: usize, eos_after: usize) -> PageStream {
        let hidden = Mat::from_vec(1, 2, vec![input_index as f32, eos_after as f32]);
        PageStream::new(input_index, prefill_len, /*prompt*/ &[], hidden)
    }

    /// Mean per-window variance of `prefill_len` when `order` is chunked into the
    /// consecutive co-batch windows of size `width` it would be admitted in.
    fn mean_window_len_variance(streams: &[PageStream], order: &[usize], width: usize) -> f64 {
        let mut total = 0.0_f64;
        let mut windows = 0_usize;
        for chunk in order.chunks(width) {
            let n = chunk.len() as f64;
            let mean = chunk
                .iter()
                .map(|&i| streams[i].prefill_len as f64)
                .sum::<f64>()
                / n;
            let var = chunk
                .iter()
                .map(|&i| {
                    let d = streams[i].prefill_len as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / n;
            total += var;
            windows += 1;
        }
        if windows == 0 {
            0.0
        } else {
            total / windows as f64
        }
    }

    /// (a) The packed admission order is a BIJECTION over the stream set: every
    /// stream is admitted exactly once — none lost, none duplicated — so packing
    /// can only reorder admission, never change the multiset of streams.
    #[test]
    fn packed_admission_order_is_a_bijection_over_the_streams() {
        let streams = vec![
            stream_len(0, 30, 2),
            stream_len(1, 5, 2),
            stream_len(2, 30, 2), // duplicate length → must not collapse
            stream_len(3, 12, 2),
            stream_len(4, 5, 2),
            stream_len(5, 99, 2),
        ];
        let order = admission_order(&streams, true);
        assert_eq!(order.len(), streams.len(), "no stream lost or duplicated");
        let mut seen = order;
        seen.sort_unstable();
        assert_eq!(
            seen,
            (0..streams.len()).collect::<Vec<_>>(),
            "packed order is a permutation of 0..n (every stream admitted exactly once)"
        );
    }

    /// (b) On a synthetic pool of interleaved short/long pages, the packed order's
    /// mean per-window length-variance is <= (and here strictly <) the naive
    /// order's — i.e. the packing actually GROUPS similar lengths so each
    /// co-batch window straddles a smaller length spread.
    #[test]
    fn packing_reduces_mean_window_length_variance() {
        // Interleaved so naive windows straddle the short/long gap; packed don't.
        let streams = vec![
            stream_len(0, 10, 2),
            stream_len(1, 100, 2),
            stream_len(2, 12, 2),
            stream_len(3, 98, 2),
            stream_len(4, 11, 2),
            stream_len(5, 105, 2),
            stream_len(6, 9, 2),
            stream_len(7, 101, 2),
        ];
        let width = 2;
        let naive_var =
            mean_window_len_variance(&streams, &admission_order(&streams, false), width);
        let packed_var =
            mean_window_len_variance(&streams, &admission_order(&streams, true), width);
        assert!(
            packed_var <= naive_var,
            "packing must never increase window length variance (packed {packed_var} > naive {naive_var})"
        );
        assert!(
            packed_var < naive_var,
            "on a varied pool packing should strictly group similar lengths (packed {packed_var} vs naive {naive_var})"
        );
    }

    /// (c) With `FOCR_BATCH_PACK` unset (pack == false), the admission order is
    /// byte-identical to today's exact `(0..total)` submission order.
    #[test]
    fn unpacked_admission_order_is_byte_identical_to_submission_order() {
        let streams = vec![
            stream_len(0, 30, 2),
            stream_len(1, 5, 2),
            stream_len(2, 99, 2),
            stream_len(3, 12, 2),
        ];
        assert_eq!(
            admission_order(&streams, false),
            (0..streams.len()).collect::<Vec<_>>(),
            "pack-off admission order must equal today's submission order, byte-for-byte"
        );
    }

    /// LOSSLESSNESS (the heart of Lever-F): packing changes only WHICH streams
    /// co-batch; every stream's emitted token sequence — re-sorted to input order
    /// — is byte-identical with packing ON vs OFF. Proven directly through the
    /// scheduler core with the pack flag injected (no process-env mutation), so it
    /// is deterministic and order-independent.
    #[test]
    fn packing_does_not_change_any_stream_output() {
        // Varied prefill_len AND varied emission lengths so packing genuinely
        // reshuffles the co-batch windows yet must not perturb any output.
        let make = || {
            vec![
                stream_len(0, 10, 3),
                stream_len(1, 100, 1),
                stream_len(2, 12, 5),
                stream_len(3, 98, 2),
                stream_len(4, 11, 4),
                stream_len(5, 105, 1),
            ]
        };

        let mut sched_off = BatchScheduler::new(2, 100);
        let mut step_off = MockStep::new();
        let out_off = sched_off
            .run_with_pack(make(), &mut step_off, false)
            .expect("run pack off");

        let mut sched_on = BatchScheduler::new(2, 100);
        let mut step_on = MockStep::new();
        let out_on = sched_on
            .run_with_pack(make(), &mut step_on, true)
            .expect("run pack on");

        assert_eq!(
            out_on, out_off,
            "packed admission must not change any stream's emitted tokens (lossless)"
        );
        // And the active-set cap is unchanged by packing.
        assert!(sched_on.stats().max_active <= 2);
        assert_eq!(sched_on.stats().max_concurrent_forwards, 1);
    }
}
