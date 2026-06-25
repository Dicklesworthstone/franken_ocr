# bd-1gv.10.1 projector transpose microbench evidence

This directory records the local before/after evidence for the
`vision_bridge::transpose` loop-order change in `src/native_engine/vision_bridge.rs`.

This is **not** a `docs/PERF_LEDGER.md` row: it does not compare against the
pinned Phase -1 CPU reference, and it uses deterministic synthetic inputs instead
of model fixtures. It is negative-evidence discipline for the old strided-store
transpose order, plus reproducible support for keeping the contiguous-store
replacement.

## Harness

- Scratch harness source copied here:
  - `projector_bench_Cargo.toml`
  - `projector_bench_main.rs`
- Real path under test: `franken_ocr::native_engine::vision_bridge::project`
- Shape: 256 rows x 2048 input, projector weight 1280 x 2048, bias 1280
- Per benchmark command: 16 calls to `project`
- Thread env: `RAYON_NUM_THREADS=8 OMP_NUM_THREADS=8`
- Allocator: system allocator
- Fallback / kill-switch state: no `FOCR_*` performance kill-switches set

## Results

Command:

```text
RAYON_NUM_THREADS=8 OMP_NUM_THREADS=8 /Volumes/USBNVME16TB/temp_agent_space/focr_projector_bench_target/release-perf/focr_projector_bench --iters 16
```

Baseline, old strided-destination transpose:

- Hyperfine: `187.9 ms +/- 4.2 ms` for 16 calls
- Range: `184.1 ms ... 197.6 ms`
- Smoke checksum for 4 calls: `-0.039779253`

After, contiguous-destination transpose:

- Hyperfine: `132.5 ms +/- 11.0 ms` for 16 calls
- Range: `114.3 ms ... 150.8 ms`
- Smoke checksum for 4 calls: `-0.039779253`

Local mean speedup for this harness: `187.933 / 132.451 = 1.419x`
(`29.5%` less wall time for the same 16 projector calls).

## Files

- `baseline_projector_hyperfine.{txt,json}`: old loop order.
- `after_projector_hyperfine.{txt,json}`: contiguous destination-store loop order.
- `SHA256SUMS`: hash manifest for this evidence bundle.
