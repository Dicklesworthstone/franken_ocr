// LEVER-1 precondition: is matrixmultiply's wasm relaxed sgemm microkernel live
// in the +relaxed build? Same crate, same shapes, two modules — the ratio answers.
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
const A = load(`${HERE}/wbench-simd.wasm`);    // +relaxed (shipped)
const B = load(`${HERE}/wbench-relaxed.wasm`); // +relaxed-simd
console.log(`route nosimd=${A.route_id()} simd=${B.route_id()}`);
const CASES = [
  ["SAM qkv     m=4096 k=768 n=2304", [4096, 768, 2304], 1],
  ["SAM proj    m=4096 k=768 n=768", [4096, 768, 768], 2],
  ["SAM mlp up  m=4096 k=768 n=3072", [4096, 768, 3072], 1],
  ["win attn QK m=196  k=64  n=196", [196, 64, 196], 200],
  ["CLIP proj   m=577  k=1024 n=1024", [577, 1024, 1024], 4],
];
const PAIRS = Number(process.env.PAIRS ?? 3);
for (const [label, [m, k, n], rounds] of CASES) {
  const flops = 2 * m * k * n * rounds;
  const best = { A: Infinity, B: Infinity };
  const sums = {};
  A.bench_sgemm(m, k, n, 1); B.bench_sgemm(m, k, n, 1);
  for (let p = 0; p < PAIRS; p++) {
    for (const [tag, x] of p % 2 === 0 ? [["A", A], ["B", B]] : [["B", B], ["A", A]]) {
      const t0 = performance.now();
      const v = x.bench_sgemm(m, k, n, rounds);
      const ms = performance.now() - t0;
      if (ms < best[tag]) best[tag] = ms;
      sums[tag] = v;
    }
  }
  const g = (ms) => flops / (ms * 1e6);
  console.log(
    `${label.padEnd(34)} simd   ${g(best.A).toFixed(2).padStart(6)} GF/s  relaxed ${g(best.B).toFixed(2).padStart(6)} GF/s  ${(best.A / best.B).toFixed(2)}x  checksum ${sums.A === sums.B ? "=" : "DIFFER"}`,
  );
}
