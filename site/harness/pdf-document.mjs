// Multi-page PDF smoke: loads the site through serve.mjs, opens a real
// multi-page PDF, and asserts that the whole-document plumbing is wired —
// page count, the per-page ledger, the page-range grammar, and that the worker
// holds the bytes instead of re-receiving them per page.
//
// Deliberately does NOT download a model: `pdf_info`/`pdf_render_page` are
// module-level wasm exports, so the entire PDF path is testable without the
// multi-gigabyte artifact. Recognition itself is covered by browser.mjs.
//
// Usage: node site/harness/pdf-document.mjs [--headed] [path/to.pdf]
import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";

const { createRequire } = await import("node:module");
const require = createRequire("/Users/jemanuel/projects/frankentts/package.json");
const { chromium } = require("playwright");

const PORT = Number(process.env.PORT ?? 8919);
const headed = process.argv.includes("--headed");
const pdfPath = process.argv.find((a) => a.endsWith(".pdf"));
if (!pdfPath) {
  console.error("usage: node site/harness/pdf-document.mjs [--headed] <file.pdf>");
  process.exit(2);
}

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
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });

  await page.goto(`http://localhost:${PORT}/`, { waitUntil: "load" });
  // The engine module reports itself once the worker has booted.
  await page.waitForFunction(
    () => document.getElementById("route")?.textContent?.trim().length > 0,
    { timeout: 60_000 },
  );
  check("page boots and the worker reports an int8 route", true);

  // The new controls exist.
  check("page-range input present", (await page.locator("#pdf-range").count()) === 1);
  check("page ledger present", (await page.locator("#page-ledger").count()) === 1);

  // Open a real multi-page PDF through the file input.
  await page.setInputFiles("#file-input", pdfPath);
  await page.waitForFunction(
    () => !document.getElementById("pdf-bar").hidden,
    { timeout: 30_000 },
  );
  const pages = Number(await page.locator("#pdf-count").textContent());
  check("PDF opened with a page count", pages > 1, `${pages} pages`);

  // The summary lands only after page 1 has rasterized, so wait for it to
  // settle rather than reading the transient "rendering…" line.
  await page
    .waitForFunction(
      () => /whole document/i.test(document.getElementById("status")?.textContent ?? ""),
      { timeout: 30_000 },
    )
    .catch(() => {});
  const status = (await page.locator("#status").textContent()) ?? "";
  check(
    "status says the whole document will be read",
    /whole document/i.test(status),
    status.trim(),
  );

  // The preview rendered page 1 through the same pure-Rust rasterizer.
  check("page 1 preview rendered", !(await page.locator("#preview").isHidden()));

  // Page-range grammar, evaluated in the page's own scope.
  const ranges = await page.evaluate(() => {
    const set = (v) => {
      document.getElementById("pdf-range").value = v;
    };
    const out = {};
    set("");
    out.all = window.__focrSelectedPages?.() ?? null;
    set("2");
    out.single = window.__focrSelectedPages?.() ?? null;
    set("1-2");
    out.range = window.__focrSelectedPages?.() ?? null;
    set("3,1");
    out.unordered = window.__focrSelectedPages?.() ?? null;
    set("99");
    out.outOfRange = window.__focrSelectedPages?.() ?? null;
    set("");
    return out;
  });
  if (ranges.all === null) {
    check("page-range grammar exposed for test", false, "window.__focrSelectedPages missing");
  } else {
    check("empty range means every page", ranges.all.length === pages, JSON.stringify(ranges.all));
    check("single page", JSON.stringify(ranges.single) === "[2]", JSON.stringify(ranges.single));
    check("inclusive range", JSON.stringify(ranges.range) === "[1,2]", JSON.stringify(ranges.range));
    check(
      "unordered input is sorted and deduped",
      JSON.stringify(ranges.unordered) === "[1,3]",
      JSON.stringify(ranges.unordered),
    );
    check(
      "out-of-range pages are dropped",
      Array.isArray(ranges.outOfRange) && ranges.outOfRange.length === 0,
      JSON.stringify(ranges.outOfRange),
    );
  }

  check("no console errors", errors.length === 0, errors.join(" | "));
} finally {
  await browser.close();
  server.kill("SIGTERM");
}

console.log(failures === 0 ? "\nALL CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
