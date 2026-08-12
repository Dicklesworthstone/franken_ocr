// LEVER 3 precondition: does LLVM autovectorize the f32 vision glue under
// +simd128? Two builds of the same crate, interleaved ABBA, min-of-pairs.
import fs from "node:fs";
const HERE = "/private/tmp/claude-501/-Users-jemanuel-projects-franken-ocr/1a1fb67b-0a94-4cac-a2d8-53dc6a100500/scratchpad";
function load(p) {
  const mod = new WebAssembly.Module(fs.readFileSync(p));
  const imports = {};
  for (const im of WebAssembly.Module.imports(mod)) {
    imports[im.module] ??= {};
    imports[im.module][im.name] = () => { throw new Error("unexpected import " + im.name); };
  }
  return new WebAssembly.Instance(mod, imports).exports;
}
const A = load(`${HERE}/wbench-nosimd.wasm`);
const B = load(`${HERE}/wbench-simd.wasm`);
const CASES = [
  ["bf16->f32 widen 16M elems", "bench_bf16_widen", [16 << 20], 3],
  ["layer_norm 4096x768", "bench_layer_norm", [4096, 768], 20],
  ["softmax rows 4096x196", "bench_softmax", [4096, 196], 20],
  ["sdpa 12bh 196x196 d64", "bench_sdpa", [12, 196, 196, 64], 20],
  ["sdpa 12bh 577x577 d64", "bench_sdpa", [12, 577, 577, 64], 4],
];
const PAIRS = Number(process.env.PAIRS ?? 4);
for (const [label, fn, args, rounds] of CASES) {
  const best = { A: Infinity, B: Infinity };
  const sums = {};
  A[fn](...args, 1); B[fn](...args, 1);
  for (let p = 0; p < PAIRS; p++) {
    for (const [tag, x] of p % 2 === 0 ? [["A", A], ["B", B]] : [["B", B], ["A", A]]) {
      const t0 = performance.now();
      const v = x[fn](...args, rounds);
      const ms = performance.now() - t0;
      if (ms < best[tag]) best[tag] = ms;
      sums[tag] = v;
    }
  }
  console.log(
    `${label.padEnd(30)} nosimd ${best.A.toFixed(1).padStart(8)} ms  simd128 ${best.B.toFixed(1).padStart(8)} ms  ${(best.A / best.B).toFixed(2)}x  checksum ${sums.A === sums.B ? "=" : "DIFFER"}`,
  );
}
