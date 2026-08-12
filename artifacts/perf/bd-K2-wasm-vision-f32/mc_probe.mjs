// MATMUL_SGEMM_MC sweep: A-panel row blocking. MC changes only the m-blocking,
// never the k-accumulation order, so every variant must be BIT-IDENTICAL —
// the checksum column is the gate, the GF/s column is the payoff.
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
const ARMS = [
  ["MC=64 ", load(`${HERE}/wbench-simd.wasm`)],
  ["MC=128", load(`${HERE}/wbench-mc128.wasm`)],
  ["MC=256", load(`${HERE}/wbench-mc256.wasm`)],
];
const CASES = [
  ["SAM qkv     m=4096 k=768 n=2304", [4096, 768, 2304], 1],
  ["SAM proj    m=4096 k=768 n=768", [4096, 768, 768], 2],
  ["SAM mlp up  m=4096 k=768 n=3072", [4096, 768, 3072], 1],
  ["CLIP proj   m=577  k=1024 n=1024", [577, 1024, 1024], 4],
];
const PAIRS = Number(process.env.PAIRS ?? 5);
for (const [label, [m, k, n], rounds] of CASES) {
  const flops = 2 * m * k * n * rounds;
  const best = ARMS.map(() => Infinity);
  const wins = ARMS.map(() => 0);
  const sums = ARMS.map(() => null);
  for (const [, x] of ARMS) x.bench_sgemm(m, k, n, 1);
  for (let p = 0; p < PAIRS; p++) {
    // rotate arm order every pair so no arm is systematically thermally last
    const order = ARMS.map((_, i) => (i + p) % ARMS.length);
    const pair = ARMS.map(() => Infinity);
    for (const i of order) {
      const t0 = performance.now();
      const v = ARMS[i][1].bench_sgemm(m, k, n, rounds);
      const ms = performance.now() - t0;
      pair[i] = ms;
      if (ms < best[i]) best[i] = ms;
      sums[i] = v;
    }
    // paired sign test: within THIS adjacent triple, who was fastest?
    wins[pair.indexOf(Math.min(...pair))] += 1;
  }
  const bits = sums.every((s) => s === sums[0]) ? "BIT-EQUAL" : "DIFFER";
  const cols = ARMS.map(([tag], i) =>
    `${tag} ${(flops / (best[i] * 1e6)).toFixed(2).padStart(6)} GF/s w${wins[i]}`,
  ).join("  ");
  console.log(`${label.padEnd(34)} ${cols}  [${bits}]`);
}
