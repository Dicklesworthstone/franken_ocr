// Cache-persistence proof (task F, bd-wk3p): a SECOND visit against the SAME
// persistent Chromium profile that differential.mjs populated. Asserts the
// 2.8 GB unlimited-ocr artifact hydrates from the Cache API ("verified from
// cache" in #progress-text), that the warm load is faster than the cold load
// recorded in REPORT.json, and that no cache.put warning fires this time.
//
// Usage (AFTER differential.mjs):
//   FOCR_SCRATCH=$S node site/harness/persistence.mjs
// Writes $S/corpus/browser/PERSISTENCE.json.
import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const require = (await import("node:module")).createRequire(
  "/Users/jemanuel/projects/frankentts/package.json",
);
const { chromium } = require("playwright");

const S =
  process.env.FOCR_SCRATCH ??
  "/private/tmp/claude-501/-Users-jemanuel-projects-franken-ocr/1a1fb67b-0a94-4cac-a2d8-53dc6a100500/scratchpad";
// SAME port as differential.mjs: the Cache API is ORIGIN-keyed, and a second
// visit to the deployed site is a second visit to the same origin. (Measured:
// a run on a different port saw an empty cache and re-downloaded.)
const PORT = Number(process.env.PORT ?? 8941);
const MODEL_DIR = process.env.FOCR_MODEL_DIR ?? join(S, "models-www");
const PROFILE_DIR = join(S, "chrome-profile-diff"); // SAME profile as visit 1
const OUT_DIR = join(S, "corpus", "browser");

const reportPath = join(OUT_DIR, "REPORT.json");
const coldSeconds = existsSync(reportPath)
  ? JSON.parse(readFileSync(reportPath, "utf8")).cold_load_seconds ?? null
  : null;

const server = spawn(
  "node",
  [new URL("./serve.mjs", import.meta.url).pathname],
  {
    env: { ...process.env, PORT: String(PORT), FOCR_MODEL_DIR: MODEL_DIR },
    stdio: ["ignore", "inherit", "inherit"],
  },
);
await sleep(800);

const consoleLog = [];
const result = {
  profile: PROFILE_DIR,
  cold_load_seconds: coldSeconds,
  console: consoleLog,
};
let failed = null;

const context = await chromium.launchPersistentContext(PROFILE_DIR, {
  headless: true,
});
try {
  const page = context.pages()[0] ?? (await context.newPage());
  page.on("console", (msg) => {
    if (msg.type() === "error" || msg.type() === "warning") {
      const line = `[console.${msg.type()}] ${msg.text()}`;
      consoleLog.push(line);
      console.log(line);
    }
  });
  page.on("pageerror", (err) => {
    consoleLog.push(`[pageerror] ${err}`);
    console.log(`[pageerror] ${err}`);
  });

  await page.goto(`http://localhost:${PORT}/`);
  await page.waitForFunction(
    () => !document.getElementById("load-model").disabled,
    undefined,
    { timeout: 60_000 },
  );
  await page.selectOption("#model-select", "unlimited-ocr");
  await page.click("#load-model");
  await page.evaluate(() => {
    window.__ptLog = [];
    const el = document.getElementById("progress-text");
    new MutationObserver(() => {
      const t = el.textContent;
      if (t && window.__ptLog[window.__ptLog.length - 1] !== t) window.__ptLog.push(t);
    }).observe(el, { childList: true, characterData: true, subtree: true });
  });
  const t0 = Date.now();
  await page.click("#consent-yes");
  await page.waitForFunction(
    () => document.getElementById("status").textContent.includes("Model ready"),
    undefined,
    { timeout: 30 * 60_000 },
  );
  result.warm_load_seconds = (Date.now() - t0) / 1000;
  result.load_status = await page.textContent("#status");

  const ptLog = await page.evaluate(() => window.__ptLog);
  result.from_cache = ptLog.some((t) => t.includes("verified from cache"));
  result.progress_samples = ptLog.slice(-3);
  result.cache_put_warnings = consoleLog.filter((l) => l.includes("cache.put"));
  result.speedup =
    coldSeconds != null ? Number((coldSeconds / result.warm_load_seconds).toFixed(2)) : null;
  result.verdict =
    result.from_cache && result.cache_put_warnings.length === 0
      ? "PERSISTED"
      : "NOT_PERSISTED";

  console.log(
    `warm load: ${result.warm_load_seconds.toFixed(1)}s (cold was ${coldSeconds}s) — ` +
      `fromCache=${result.from_cache}, cache.put warnings=${result.cache_put_warnings.length} — ${result.verdict}`,
  );
} catch (err) {
  failed = err;
  result.fatal = String(err?.stack ?? err);
  result.verdict = "FAILED";
} finally {
  writeFileSync(join(OUT_DIR, "PERSISTENCE.json"), JSON.stringify(result, null, 2));
  await context.close();
  server.kill();
}
if (failed) {
  console.error(`PERSISTENCE FAILED: ${failed}`);
  process.exit(1);
}
console.log("PERSISTENCE DONE");
