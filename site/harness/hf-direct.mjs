// Proves the browser can stream model weights STRAIGHT from Hugging Face under
// this site's real headers — the thing that keeps multi-gigabyte downloads off
// the site's own bandwidth bill.
//
// This is not a reasoning exercise, it is a measurement, because three separate
// headers have to cooperate and any one of them silently kills it:
//
//   * CSP `connect-src` must list huggingface.co AND the region-specific CDN it
//     302s to (`*.hf.co`), or the fetch is blocked before it leaves.
//   * `Cross-Origin-Embedder-Policy: require-corp` (needed for SharedArrayBuffer
//     and therefore for the threaded lane) constrains cross-origin subresources;
//     a CORS-mode fetch has to actually satisfy it.
//   * Hugging Face must answer ranged requests cross-origin, since the weight
//     loader streams.
//
// Usage: node site/harness/hf-direct.mjs [--headed]
import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";

const { createRequire } = await import("node:module");
const require = createRequire("/Users/jemanuel/projects/frankentts/package.json");
const { chromium } = require("playwright");

const PORT = Number(process.env.PORT ?? 8921);
const headed = process.argv.includes("--headed");
const SMALL =
  "https://huggingface.co/Dicklesworthstone/franken_ocr-weights/resolve/main/tromr/tokenizer_note.json";
const BIG =
  "https://huggingface.co/Dicklesworthstone/franken_ocr-weights/resolve/main/tromr/tromr.int8.focrq";

const server = spawn("node", [new URL("./serve.mjs", import.meta.url).pathname], {
  env: { ...process.env, PORT: String(PORT) },
  stdio: ["ignore", "inherit", "inherit"],
});
await sleep(700);

let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures++;
};

const browser = await chromium.launch({ headless: !headed });
try {
  const page = await browser.newPage();
  const violations = [];
  page.on("console", (m) => {
    const t = m.text();
    if (/Content Security Policy|ERR_BLOCKED|CORS|Cross-Origin/i.test(t)) violations.push(t);
  });

  await page.goto(`http://localhost:${PORT}/`, { waitUntil: "load" });

  // The threaded lane's precondition. If this regressed, COEP got broken.
  const isolated = await page.evaluate(() => self.crossOriginIsolated === true);
  check("page is still cross-origin isolated (COEP/COOP intact)", isolated);

  // Whole small file, cross-origin, from inside the page's CSP.
  const small = await page.evaluate(async (url) => {
    try {
      const r = await fetch(url);
      const b = await r.arrayBuffer();
      return { ok: r.ok, status: r.status, bytes: b.byteLength };
    } catch (e) {
      return { error: String(e) };
    }
  }, SMALL);
  check(
    "cross-origin fetch of a whole file succeeds",
    small.ok === true && small.bytes === 830,
    JSON.stringify(small),
  );

  // Ranged request — the weight loader streams, so this is the load-bearing one.
  const ranged = await page.evaluate(async (url) => {
    try {
      const r = await fetch(url, { headers: { Range: "bytes=0-1023" } });
      const b = await r.arrayBuffer();
      return { status: r.status, bytes: b.byteLength };
    } catch (e) {
      return { error: String(e) };
    }
  }, BIG);
  check(
    "cross-origin RANGED fetch streams (206 + exact bytes)",
    ranged.status === 206 && ranged.bytes === 1024,
    JSON.stringify(ranged),
  );

  // Snapshot BEFORE the deliberate negative probe below, which is supposed to
  // log a CSP violation — counting that as a failure would be scoring the test's
  // own control as a bug.
  check("no CSP/CORS violations during the real fetches", violations.length === 0,
    violations.join(" | "));

  // The guarantee the page still makes: nowhere else is reachable.
  const blocked = await page.evaluate(async () => {
    try {
      await fetch("https://example.com/", { mode: "cors" });
      return "NOT BLOCKED";
    } catch {
      return "blocked";
    }
  });
  check("an unrelated origin is still blocked by CSP", blocked === "blocked", blocked);
} finally {
  await browser.close();
  server.kill("SIGTERM");
}

console.log(failures === 0 ? "\nALL CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
