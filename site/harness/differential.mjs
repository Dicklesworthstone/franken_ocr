// Browser-vs-desktop differential harness (task A3): loads the unlimited-ocr
// lane once in REAL headless Chromium (persistent context — ephemeral contexts
// fabricate storage failures, sibling NE-007), runs every corpus page through
// the real site, and diffs each browser output against the native int4
// reference (the SAME .focrq artifact) with a true Levenshtein distance.
//
// Usage:
//   FOCR_SCRATCH=$S node site/harness/differential.mjs
// Writes $S/corpus/browser/<stem>.md and $S/corpus/browser/REPORT.json.
import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";
import { readFileSync, writeFileSync, mkdirSync, statSync } from "node:fs";
import { join } from "node:path";

const require = (await import("node:module")).createRequire(
  "/Users/jemanuel/projects/frankentts/package.json",
);
const { chromium } = require("playwright");

const S =
  process.env.FOCR_SCRATCH ??
  "/private/tmp/claude-501/-Users-jemanuel-projects-franken-ocr/1a1fb67b-0a94-4cac-a2d8-53dc6a100500/scratchpad";
const PORT = Number(process.env.PORT ?? 8941);
const MODEL_DIR = process.env.FOCR_MODEL_DIR ?? join(S, "models-www");
const PROFILE_DIR = join(S, process.env.FOCR_DIFF_PROFILE ?? "chrome-profile-diff");
const OUT_DIR = join(S, "corpus", "browser");
const IMG_DIR = join(S, "corpus", "images");
const NATIVE_DIR = join(S, "corpus", process.env.FOCR_DIFF_NATIVE ?? "native_int4");
mkdirSync(OUT_DIR, { recursive: true });

const MANIFEST = JSON.parse(readFileSync(join(S, "corpus", "MANIFEST.json"), "utf8"));
const nativeSeconds = Object.fromEntries(
  MANIFEST.images.map((i) => [i.name.replace(/\.png$/, ""), i.int4_seconds]),
);

// smallest first: got_* by size, then teachers-guide, school-journal p-06 last
const PAGES = [
  "got_formula",
  "got_table",
  "got_sample_text",
  "archive_teachers-guide_1873_p-02",
  "archive_teachers-guide_1873_p-03",
  "archive_teachers-guide_1873_p-04",
  "archive_school-journal_1886_p-06",
];

// ── true Levenshtein (two Int32Array rows, full DP) ─────────────────────────
function levenshtein(a, b) {
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;
  // code-point arrays so astral chars count once
  const A = [...a].map((c) => c.codePointAt(0));
  const B = [...b].map((c) => c.codePointAt(0));
  let prev = new Int32Array(B.length + 1);
  let cur = new Int32Array(B.length + 1);
  for (let j = 0; j <= B.length; j++) prev[j] = j;
  for (let i = 1; i <= A.length; i++) {
    cur[0] = i;
    const ai = A[i - 1];
    for (let j = 1; j <= B.length; j++) {
      const sub = prev[j - 1] + (ai === B[j - 1] ? 0 : 1);
      const del = prev[j] + 1;
      const ins = cur[j - 1] + 1;
      cur[j] = sub < del ? (sub < ins ? sub : ins) : del < ins ? del : ins;
    }
    [prev, cur] = [cur, prev];
  }
  return prev[B.length];
}

// ── server ──────────────────────────────────────────────────────────────────
const server = spawn(
  "node",
  [new URL("./serve.mjs", import.meta.url).pathname],
  {
    env: { ...process.env, PORT: String(PORT), FOCR_MODEL_DIR: MODEL_DIR },
    stdio: ["ignore", "inherit", "inherit"],
  },
);
await sleep(800);

const consoleLog = []; // every warning/error, verbatim
let sawThreadedPkg = false;
let failed = null;
const report = { pages: {}, console: consoleLog };

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
  page.on("crash", () => console.log("[page CRASHED]"));
  page.on("request", (req) => {
    if (req.url().includes("/pkg-threaded/")) sawThreadedPkg = true;
  });
  const step = (name) => console.log(`[${new Date().toISOString()}] ${name}`);

  await page.goto(`http://localhost:${PORT}/`);
  report.crossOriginIsolated = await page.evaluate(() => globalThis.crossOriginIsolated);
  console.log(`crossOriginIsolated: ${report.crossOriginIsolated}`);

  await page.waitForFunction(
    () => !document.getElementById("load-model").disabled,
    undefined,
    { timeout: 60_000 },
  );
  report.route = await page.textContent("#route");
  console.log(`route: ${report.route}`);

  // ── load unlimited-ocr once ───────────────────────────────────────────────
  step("select unlimited-ocr");
  await page.selectOption("#model-select", "unlimited-ocr");
  await page.click("#load-model");
  // observe #progress-text so fromCache-vs-download is on the record
  await page.evaluate(() => {
    window.__ptLog = [];
    const el = document.getElementById("progress-text");
    new MutationObserver(() => {
      const t = el.textContent;
      if (t && window.__ptLog[window.__ptLog.length - 1] !== t) window.__ptLog.push(t);
    }).observe(el, { childList: true, characterData: true, subtree: true });
  });
  step("consent (cold load starts)");
  const tLoad0 = Date.now();
  await page.click("#consent-yes");
  await page.waitForFunction(
    () => document.getElementById("status").textContent.includes("Model ready"),
    undefined,
    { timeout: 30 * 60_000 },
  );
  report.cold_load_seconds = (Date.now() - tLoad0) / 1000;
  report.load_status = await page.textContent("#status");
  const ptLog = await page.evaluate(() => window.__ptLog);
  report.load_from_cache = ptLog.some((t) => t.includes("verified from cache"));
  report.progress_tail = ptLog.slice(-3);
  console.log(`Model ready in ${report.cold_load_seconds.toFixed(1)}s — ${report.load_status}`);
  console.log(`fromCache during load: ${report.load_from_cache}`);

  // threaded lane: which pkg was fetched + live worker count (engine worker
  // + rayon pool; app.js does not surface pkg.thread_count()).
  report.threaded = sawThreadedPkg;
  report.worker_count = page.workers().length;
  console.log(`threaded pkg fetched: ${sawThreadedPkg}; live workers: ${report.worker_count}`);

  // ── per-page runs ─────────────────────────────────────────────────────────
  for (const stem of PAGES) {
    const img = join(IMG_DIR, `${stem}.png`);
    step(`page ${stem} (${(statSync(img).size / 1024).toFixed(0)} KB)`);
    await page.setInputFiles("#file-input", img);
    await page.waitForFunction(
      () => !document.getElementById("run").disabled,
      undefined,
      { timeout: 60_000 },
    );
    // status flips to "Recognizing…" then "Done in Xs" (or "… failed"); a
    // MutationObserver flag avoids confusing this run's Done with the last one's.
    await page.evaluate(() => {
      window.__runDone = null;
      const el = document.getElementById("status");
      const mo = new MutationObserver(() => {
        const t = el.textContent;
        if (t.includes("Done in") || t.includes("failed")) {
          window.__runDone = t;
          mo.disconnect();
        }
      });
      mo.observe(el, { childList: true, characterData: true, subtree: true });
    });
    const t0 = Date.now();
    await page.click("#run");
    await page.waitForFunction(() => window.__runDone !== null, undefined, {
      timeout: 30 * 60_000,
    });
    const statusText = await page.evaluate(() => window.__runDone);
    const wall = (Date.now() - t0) / 1000;
    if (!statusText.includes("Done in")) {
      report.pages[stem] = { error: statusText, wall_seconds: wall };
      console.log(`FAILED ${stem}: ${statusText}`);
      continue;
    }
    const reported = Number(statusText.match(/Done in ([\d.]+)s/)?.[1]);
    const output = await page.textContent("#output");
    writeFileSync(join(OUT_DIR, `${stem}.md`), output);

    const native = readFileSync(join(NATIVE_DIR, `${stem}.md`), "utf8");
    const exact = output === native;
    const dist = exact ? 0 : levenshtein(output, native);
    const refLen = [...native].length;
    const cer = refLen ? dist / refLen : dist ? 1 : 0;
    report.pages[stem] = {
      browser_seconds: reported,
      wall_seconds: wall,
      native_seconds: nativeSeconds[stem] ?? null,
      exact,
      distance: dist,
      cer: Number(cer.toFixed(6)),
      bytes: Buffer.byteLength(output),
    };
    console.log(
      `${stem}: browser ${reported}s (wall ${wall.toFixed(1)}s), native ${nativeSeconds[stem]}s, ` +
        `exact=${exact}, dist=${dist}, CER=${cer.toFixed(4)}`,
    );
  }

  const done = Object.values(report.pages).filter((p) => !p.error);
  report.totals = {
    pages_run: PAGES.length,
    pages_ok: done.length,
    exact_matches: done.filter((p) => p.exact).length,
    browser_seconds_total: done.reduce((n, p) => n + (p.browser_seconds ?? 0), 0),
    native_seconds_total: done.reduce((n, p) => n + (p.native_seconds ?? 0), 0),
  };
} catch (err) {
  failed = err;
  report.fatal = String(err?.stack ?? err);
} finally {
  writeFileSync(join(OUT_DIR, process.env.FOCR_DIFF_REPORT ?? "REPORT.json"), JSON.stringify(report, null, 2));
  await context.close();
  server.kill();
}
if (failed) {
  console.error(`DIFFERENTIAL FAILED: ${failed}`);
  process.exit(1);
}
console.log(`DIFFERENTIAL DONE — ${JSON.stringify(report.totals)}`);
