// Real-Chromium threaded-lane proof (bd-rhk7).
//
// Drives site/harness/threads.html, which talks the shipping engine-worker
// protocol directly, and prints the worker's OWN answers: which pkg dir it
// chose, whether the instantiated memory is a SharedArrayBuffer, rayon's actual
// worker count, and the recognize wall time. Also writes the full output text
// so the threaded and serial lanes can be diffed byte-for-byte — a threaded
// build that changes the OCR text is a numerics regression, not a speedup.
//
// Usage:
//   FOCR_MODEL_DIR=… [FOCR_SMOKE_MODEL=unlimited-ocr] [FOCR_FORCE_SERIAL=1]
//   [FOCR_OUT=/path/out.txt] [PORT=8933] node site/harness/threads.mjs
//
// FOCR_FORCE_SERIAL overrides the user agent to desktop Safari, which the
// worker's allow-list excludes — that is how the SERIAL lane gets exercised in
// a browser that would otherwise qualify for threads.
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

const require = (await import("node:module")).createRequire(
  "/Users/jemanuel/projects/frankentts/package.json",
);
const { chromium } = require("playwright");

const PORT = Number(process.env.PORT ?? 8933);
const modelId = process.env.FOCR_SMOKE_MODEL ?? "tromr";
const forceSerial = process.env.FOCR_FORCE_SERIAL === "1";
const outPath = process.env.FOCR_OUT ?? "";

const server = spawn("node", [new URL("./serve.mjs", import.meta.url).pathname], {
  env: { ...process.env, PORT: String(PORT) },
  stdio: ["ignore", "inherit", "inherit"],
});
await sleep(600);

let failed = null;
const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext(
    forceSerial
      ? {
          userAgent:
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        }
      : {},
  );
  const page = await context.newPage();
  page.on("pageerror", (err) => console.log(`[pageerror] ${err}`));
  page.on("crash", () => console.log("[page CRASHED]"));

  const t0 = Date.now();
  await page.goto(`http://localhost:${PORT}/harness/threads.html?model=${modelId}`);
  console.log(`[${new Date().toISOString()}] driving model=${modelId} forceSerial=${forceSerial}`);

  await page.waitForFunction(() => globalThis.__focr?.done === true, undefined, {
    timeout: 40 * 60_000,
  });
  const r = await page.evaluate(() => {
    const o = globalThis.__focr;
    return { ...o, output: o.output ?? "" };
  });
  console.log(`wall: ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  console.log(`crossOriginIsolated: ${r.crossOriginIsolated}  hardwareConcurrency: ${r.hardwareConcurrency}`);
  console.log(`init: ${JSON.stringify(r.init)}`);
  console.log(`load: ${JSON.stringify(r.load)}`);
  console.log(`stages: ${r.stages.join(" -> ")}`);
  if (r.error) throw new Error(r.error);
  console.log(`recognize: ${(r.recognize_ms / 1000).toFixed(1)}s   THREADS=${r.load.threads}  PKG=${r.load.pkg}`);
  if (!r.output.trim().length) throw new Error("empty output");
  if (outPath) {
    writeFileSync(outPath, r.output);
    console.log(`output (${r.output.length} chars) -> ${outPath}`);
  }
  console.log(`--- output ---\n${r.output.slice(0, 600)}`);
} catch (err) {
  failed = err;
} finally {
  await browser.close();
  server.kill();
}
if (failed) {
  console.error(`THREADS PROBE FAILED: ${failed}`);
  process.exit(1);
}
console.log("THREADS PROBE PASSED");
