// Real-Chromium smoke: loads the site through serve.mjs (real headers, real
// files), consents to the model download, runs the sample scan, and asserts a
// MusicXML result. This is the artifact-level gate for the whole chain:
// page → worker → wasm → model → result.
//
// Usage: FOCR_MODEL_DIR=/path/to/tromr node site/harness/browser.mjs [--headed]
import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";

const require = (await import("node:module")).createRequire(
  "/Users/jemanuel/projects/frankentts/package.json",
);
const { chromium } = require("playwright");

const PORT = Number(process.env.PORT ?? 8917);
const headed = process.argv.includes("--headed");

const server = spawn("node", [new URL("./serve.mjs", import.meta.url).pathname], {
  env: { ...process.env, PORT: String(PORT) },
  stdio: ["ignore", "inherit", "inherit"],
});
await sleep(600);

let failed = null;
const browser = await chromium.launch({ headless: !headed });
try {
  const page = await browser.newPage();
  page.on("console", (msg) => {
    if (msg.type() === "error") console.log(`[console.error] ${msg.text()}`);
  });
  page.on("pageerror", (err) => console.log(`[pageerror] ${err}`));

  await page.goto(`http://localhost:${PORT}/`);

  // 1. cross-origin isolation from the real _headers
  const coi = await page.evaluate(() => globalThis.crossOriginIsolated);
  console.log(`crossOriginIsolated: ${coi}`);

  // 2. engine init
  await page.waitForFunction(
    () => !document.getElementById("load-model").disabled,
    { timeout: 60_000 },
  );
  console.log(`route: ${await page.textContent("#route")}`);

  // 3. model download + hydrate
  await page.click("#load-model");
  await page.click("#consent-yes");
  await page.waitForFunction(
    () => document.getElementById("status").textContent.includes("Model ready"),
    { timeout: 10 * 60_000 },
  );
  console.log(`status: ${await page.textContent("#status")}`);

  // 4. sample image + recognize
  await page.click("#sample");
  await page.waitForFunction(
    () => !document.getElementById("run").disabled,
    { timeout: 30_000 },
  );
  await page.click("#run");
  await page.waitForFunction(
    () => !document.getElementById("result-wrap").hidden,
    { timeout: 10 * 60_000 },
  );
  const output = await page.textContent("#output");
  const meta = await page.textContent("#result-meta");
  console.log(`result-meta: ${meta}`);
  if (!output.includes("<score-partwise")) {
    throw new Error("output is not MusicXML");
  }
  console.log(`OK: MusicXML result (${output.length} chars) — ${await page.textContent("#status")}`);
} catch (err) {
  failed = err;
} finally {
  await browser.close();
  server.kill();
}
if (failed) {
  console.error(`SMOKE FAILED: ${failed}`);
  process.exit(1);
}
console.log("SMOKE PASSED");
