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
    if (msg.type() === "error" || msg.type() === "warning") {
      console.log(`[console.${msg.type()}] ${msg.text()}`);
    }
  });
  page.on("pageerror", (err) => console.log(`[pageerror] ${err}`));
  page.on("crash", () => console.log("[page CRASHED]"));
  const step = (name) => console.log(`[${new Date().toISOString()}] step: ${name}`);

  await page.goto(`http://localhost:${PORT}/`);

  // 1. cross-origin isolation from the real _headers
  const coi = await page.evaluate(() => globalThis.crossOriginIsolated);
  console.log(`crossOriginIsolated: ${coi}`);

  // 2. engine init
  await page.waitForFunction(
    () => !document.getElementById("load-model").disabled,
    undefined,
    { timeout: 60_000 },
  );
  console.log(`route: ${await page.textContent("#route")}`);

  // 3. model download + hydrate. FOCR_SMOKE_MODEL selects the lane:
  // "tromr" (default, 58 MB, MusicXML) or "unlimited-ocr" (2.8 GB, Markdown).
  const modelId = process.env.FOCR_SMOKE_MODEL ?? "tromr";
  step(`select ${modelId}`);
  await page.selectOption("#model-select", modelId);
  step("click load");
  await page.click("#load-model");
  step("click consent");
  await page.click("#consent-yes");
  step("wait model ready");
  await page.waitForFunction(
    () => document.getElementById("status").textContent.includes("Model ready"),
    undefined,
    { timeout: 20 * 60_000 },
  );
  console.log(`status: ${await page.textContent("#status")}`);

  // 4. sample image + recognize
  step("click sample");
  await page.click("#sample");
  await page.waitForFunction(
    () => !document.getElementById("run").disabled,
    undefined,
    { timeout: 60_000 },
  );
  step("click run");
  await page.click("#run");
  await page.waitForFunction(
    () => !document.getElementById("result-wrap").hidden,
    undefined,
    { timeout: 10 * 60_000 },
  );
  const output = await page.textContent("#output");
  const meta = await page.textContent("#result-meta");
  console.log(`result-meta: ${meta}`);
  if (modelId === "tromr") {
    if (!output.includes("<score-partwise")) {
      throw new Error("output is not MusicXML");
    }
  } else if (!output.trim().length) {
    throw new Error("empty OCR output");
  }
  console.log(`OK: result (${output.length} chars) — ${await page.textContent("#status")}`);
  console.log(`--- output head ---\n${output.slice(0, 300)}`);
} catch (err) {
  failed = err;
  try {
    const pages = browser.contexts().flatMap((c) => c.pages());
    for (const p of pages) {
      console.log(`[at-failure] status: ${await p.textContent("#status").catch(() => "?")}`);
      console.log(`[at-failure] run.disabled: ${await p.evaluate(() => document.getElementById("run")?.disabled).catch(() => "?")}`);
    }
  } catch {}
} finally {
  await browser.close();
  server.kill();
}
if (failed) {
  console.error(`SMOKE FAILED: ${failed}`);
  process.exit(1);
}
console.log("SMOKE PASSED");
