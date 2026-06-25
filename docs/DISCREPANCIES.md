# franken_ocr — Known Conformance Discrepancies

This document is the honest-divergence ledger: every place where `franken_ocr`'s
output or behavior **intentionally or measurably differs** from the reference
Baidu Unlimited-OCR model (the PyTorch `transformers` oracle pinned by
`scripts/gen_reference_fixtures.py`).

A discrepancy is only recorded once its impact has been **measured** against the
reference. Speculation does not belong here; the cost of a divergence must be a
real number tied to a real test before it is accepted. Every accepted divergence
carries a **kill-switch** (an environment variable that restores reference
behavior) so it can be toggled off for bit-exact comparison.

This is an **artifact-graph ledger** (plan §8.4): every entry carries the same
FrankenSuite provenance fields as `NEGATIVE_EVIDENCE.md` / `PERF_LEDGER.md`, so a
divergence is reproducible and traceable to the exact model version and command
that measured it.

## Canonical provenance source (the truth pack)

Every entry's `claim_id`/`evidence_id` and provenance fields resolve against the
**Phase −1 truth pack**:

- **Model source commit:** Hugging Face `baidu/Unlimited-OCR`
  **`3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`** (GitHub
  `7e98affeacba24e95562fbaa234ddb89b856874a`), verified 2026-06-25 — see
  `docs/truth-pack/PINNED_SOURCES.md`.
- **Source / fixture hashes:** SHA-256 of every load-bearing source in
  `docs/truth-pack/SOURCE_HASHES.md`. The `Reference behavior` of every entry
  cites the oracle code by `(file_sha256, line range)` against that table (e.g.
  `modeling_deepseekv2.py` 74e36e6b… for R-SWA semantics), and the measured impact
  cites the **fixture hash** of the parity corpus it ran against.
- **Runtime pin:** the oracle stack is `torch==2.10.0`, `transformers==4.57.1`,
  `Pillow==12.1.1` (`PINNED_SOURCES.md`). A "measured impact" produced against any
  other stack is **not comparable** and may not be recorded as ACCEPTED.

If `SOURCE_HASHES.md` fails to verify, the model moved: STOP, re-pin, and
re-confirm every entry. A DISC entry whose `Reference behavior` cannot be
resolved to a truth-pack source line is **incomplete**.

## Per-entry schema

```
## DISC-NNN: <short title>
- claim_id / evidence_id: <CLAIM-… → artifacts/perf/<bead>/ or artifacts/parity/<bead>/>
- Provenance (model commit + fixture hash): HF 3a7f4db… + <oracle file_sha256:lines from
    SOURCE_HASHES.md> + parity corpus fixture sha256 + <.focrq sha256 for the precision under test>
- CPU feature string: <dispatched SIMD tier the divergence was observed on, e.g.
    aarch64+neon+dotprod+i8mm — a divergence can be arch-specific (rounding/order)>
- Exact command + env: <gauntlet/parity invocation + FOCR_*/OMP_NUM_THREADS set>
- Reference behavior: <what the torch/transformers oracle does — quote the source line>
- Our impl: <what franken_ocr does, and where (file:fn)>
- Fallback / kill-switch state: <FOCR_* var, default value, and what the ON value restores>
- Measured impact: <real numbers vs reference — CER / token diff / TEDS / ULP / timing,
    plus the AF-2 tail figure (CVaR_0.1 / EVT_p999) for accuracy divergences>
- Resolution: ACCEPTED / INVESTIGATING / REVERT
- Tests affected: <test names / fixture corpus> (XFAIL, never SKIP — §8.6)
- Review date: <YYYY-MM-DD>
```

`Kill switch` is folded into **Fallback / kill-switch state** (the same field the
other two ledgers carry) so the three ledgers share one provenance vocabulary:
the env var name, its default, and exactly what restoring it gives back
(reference-bit-exact behavior).

Quantization-induced divergences (int8, then int4) are the expected source of
most future entries: each will record the per-bit-width measured accuracy delta
against the bf16 reference, the kill switch (e.g. forcing a layer back to higher
precision via `FOCR_INT8_ATTN=0` / `FOCR_INT8_LMHEAD=0` / dropping a tensor one
tier under AF-1), and the corpus slice (dense text / tables / formulas / numbers)
where the impact was measured — with the AF-2 tail bound, not just the mean,
since exact-token OCR fails in the tail.

---

_No discrepancies recorded yet. Nothing has been measured against the reference
oracle — the inference path does not exist. This stays empty until a real,
measured divergence appears; no placeholder or fabricated entries._

The first real entry MUST carry full truth-pack provenance. Shape to follow (a
**template**, not a measurement):

```
## DISC-001: <e.g. int8 attention q/k/v/o drifts a sub-script token on dense formulas>
- claim_id / evidence_id: CLAIM-int8-attn-qkvo → artifacts/parity/<bead>/
- Provenance (model commit + fixture hash): HF 3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5
    + modeling_deepseekv2.py sha256 74e36e6b…: <attn lines>  (SOURCE_HASHES.md)
    + parity corpus fixture sha256 <…>  +  <model>.focrq sha256 <int8-attn build>
- CPU feature string: aarch64+neon+dotprod+i8mm   (and re-checked on x86_64+avx512vnni)
- Exact command + env: cargo test -p focr --test parity -- disc_int8_attn  /
    OMP_NUM_THREADS=8  (reference torch set_num_threads(8), §9.3)
- Reference behavior: f32 Q·Kᵀ / scores·V bmm (modeling_deepseekv2.py:<lines>)
- Our impl: int8 SMMLA attention in src/decode/attention.rs::<fn>
- Fallback / kill-switch state: FOCR_INT8_ATTN (default 0 = reference f32 attention);
    =1 enables the int8 path under test
- Measured impact: CER Δ <x.xx>%, token diff <n> on the dense-formula slice;
    CVaR_0.1 <x.xx>%, EVT_p999 <x.xx>% (AF-2)
- Resolution: <ACCEPTED|INVESTIGATING|REVERT>
- Tests affected: parity::disc_int8_attn (XFAIL while kill-switch ON)
- Review date: 2026-MM-DD
```
