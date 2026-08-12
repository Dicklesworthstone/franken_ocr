// Playground orchestration (main thread): UI state, worker protocol with
// request correlation + watchdogs, crash breadcrumbs, and result rendering.
// No frameworks, no bundler, no CDN — everything is same-origin.
import { MODELS, totalBytes } from "./model-manifest.js?v=@SITEV@";

const $ = (id) => document.getElementById(id);

// ── crash breadcrumbs ───────────────────────────────────────────────────────
// The worker posts {type:"stage"} BEFORE each dangerous step; committing it
// synchronously to localStorage means a jetsam/OOM kill still leaves evidence.
// On next boot, a breadcrumb naming a hydration stage suppresses auto-load.
const CRUMB_KEY = "focr-last-stage";
const lastCrumb = localStorage.getItem(CRUMB_KEY);

// ── worker plumbing ─────────────────────────────────────────────────────────
const worker = new Worker(`./engine-worker.js?v=@SITEV@`, { type: "module" });
let nextId = 1;
const pending = new Map(); // id -> {resolve, reject, timer}

// Watchdogs catch "never", not "slow" — generous ceilings per call type.
// recognize: hard archive pages measured at up to ~15 min in-tab (int4 wasm), so the ceiling is 30 min.
const CEILINGS_MS = { init: 60_000, load: 10 * 60_000, recognize: 30 * 60_000, cancel: 5_000 };

function call(type, payload = {}, transfer = []) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`${type}: no reply within ${CEILINGS_MS[type] / 1000}s (worker hung or killed)`));
    }, CEILINGS_MS[type] ?? 60_000);
    pending.set(id, { resolve, reject, timer });
    worker.postMessage({ id, type, ...payload }, transfer);
  });
}

worker.onmessage = ({ data }) => {
  if (data.type === "stage") {
    localStorage.setItem(CRUMB_KEY, data.stage);
    return;
  }
  if (data.type === "progress") {
    onProgress(data);
    return;
  }
  if (data.type === "recognize-progress") {
    onRecognizeProgress(data);
    return;
  }
  const entry = pending.get(data.id);
  if (!entry) return;
  pending.delete(data.id);
  clearTimeout(entry.timer);
  if (data.type.endsWith(":err")) entry.reject(new Error(data.error));
  else entry.resolve(data);
};
worker.onerror = (e) => setStatus(`worker error: ${e.message}`, "err");

// ── UI state ────────────────────────────────────────────────────────────────
// Baidu Unlimited-OCR is the default lane: it is the headline capability, and
// the <select> in index.html lists it first with `selected` to match.
let modelId = "unlimited-ocr";
let model = MODELS[modelId];
let engineReady = false;
let busy = false;
let currentImage = null; // {bytes: ArrayBuffer, url: objectURL, name}
let lastResult = null;

function setStatus(text, cls = "") {
  const el = $("status");
  el.textContent = text;
  el.className = `status ${cls}`;
}

function fmtMB(n) {
  return (n / 1048576).toFixed(1);
}

// rolling throughput window for rate + ETA
const rateWindow = [];
function onProgress({ loaded, total, fromCache }) {
  $("progress-wrap").hidden = false;
  $("progress-bar").style.width = `${(loaded / total) * 100}%`;
  if (fromCache) {
    $("progress-text").textContent = `verified from cache · ${fmtMB(total)} MB`;
    return;
  }
  const now = performance.now();
  rateWindow.push([now, loaded]);
  while (rateWindow.length > 2 && now - rateWindow[0][0] > 5000) rateWindow.shift();
  let rate = 0;
  if (rateWindow.length > 1) {
    const [t0, b0] = rateWindow[0];
    rate = ((loaded - b0) / ((now - t0) / 1000)) || 0;
  }
  const eta = rate > 0 ? Math.ceil((total - loaded) / rate) : null;
  $("progress-text").textContent =
    `${fmtMB(loaded)} / ${fmtMB(total)} MB` +
    (rate > 0 ? ` · ${fmtMB(rate)} MB/s${eta !== null ? `, ~${eta}s left` : ""}` : "");
}

// ── live recognize progress ────────────────────────────────────────────────
//
// The same #progress-wrap/#progress-bar/#progress-text the download uses — the
// two never overlap in time (you cannot recognize before the model has loaded),
// and each one resets the elements when it starts.
//
// The bar is an ESTIMATE and says so: only the vision block count is a real
// denominator. Decode is the honest problem — the model stops at EOS, which is
// unknowable in advance — so token progress is mapped against a typical output
// length for the model and CAPPED at 99% until the result actually lands.
// Better a bar that finishes early than one that lies about being done.
const STAGE_WEIGHTS = { preprocess: 0.03, vision: 0.42, prefill: 0.05, decode: 0.5 };
// Typical emitted-token counts, measured on the sample pages. Undershooting is
// visible as a bar that sits at 99%; overshooting stalls it. Neither is a lie
// while the label says "estimated".
const EST_TOKENS = { "unlimited-ocr": 1600, tromr: 300, "got-ocr2": 700, smolvlm2: 300 };
const STAGE_LABELS = {
  preprocess: "Preparing the image",
  vision: "Vision encoder",
  prefill: "Reading the prompt",
  decode: "Reading the page",
  postprocess: "Assembling the result",
};

let recognizeProgress = null;

// Document-walk state. `documentIndex` is the page being read (null when the
// run is a single image), so the bar can show the DOCUMENT's progress rather
// than restarting from zero on every page.
let documentIndex = null;
let documentTotal = 0;
let documentDone = 0;
let cancelled = false;

function resetRecognizeProgress() {
  recognizeProgress = { staff: 0, staves: 1, done: 0, stage: null };
}

// Fraction (0..1) completed WITHIN one stage.
function withinStage(stage, current, total) {
  if (stage === "decode") {
    const est = EST_TOKENS[modelId] ?? 1600;
    // A determinate cap tighter than the estimate (TrOMR's 256-step ceiling)
    // is the better denominator — it cannot be exceeded.
    const denom = total > 0 ? Math.min(total, est) : est;
    return Math.min(current / denom, 1);
  }
  return total > 0 ? Math.min(current / total, 1) : 0;
}

// Fraction (0..1) of ONE staff's whole arc completed, from the stage weights.
function stageFraction(stage, current, total) {
  if (stage === "postprocess") return 1;
  let done = 0;
  for (const [name, weight] of Object.entries(STAGE_WEIGHTS)) {
    if (name === stage) break;
    done += weight;
  }
  return done + (STAGE_WEIGHTS[stage] ?? 0) * withinStage(stage, current, total);
}

function onRecognizeProgress({ stage, current, total }) {
  if (!recognizeProgress) resetRecognizeProgress();
  if (stage === "staff") {
    // TrOMR pages recognize staff by staff; each staff replays the whole
    // preprocess→vision→decode arc, so scope the estimate into its slot
    // instead of letting the bar march back to zero N times.
    recognizeProgress.staff = current;
    recognizeProgress.staves = Math.max(total, 1);
    recognizeProgress.stage = "staff";
    recognizeProgress.done = current / recognizeProgress.staves;
    paintRecognizeProgress(`Staff ${current + 1} of ${recognizeProgress.staves}`);
    return;
  }
  const { staff, staves } = recognizeProgress;
  const within = stageFraction(stage, current, total);
  // Monotonic: a stage arriving out of order must never rewind the bar.
  recognizeProgress.done = Math.max(recognizeProgress.done, (staff + within) / staves);
  recognizeProgress.stage = stage;

  const label = STAGE_LABELS[stage] ?? stage;
  let detail = label;
  if (stage === "vision" && total > 0) {
    detail = `${label} · block ${current}/${total}`;
  } else if (stage === "decode") {
    // The decode-LOCAL estimate, not the overall bar: "412 tokens (~34%)"
    // answers "how far through the reading am I".
    const pct = Math.round(withinStage(stage, current, total) * 100);
    detail = `${label} · ${current} token${current === 1 ? "" : "s"} (~${pct}%)`;
  } else if (stage === "prefill" && total > 0) {
    detail = `${label} · ${total} tokens`;
  } else if (stage === "preprocess") {
    detail = `${label}…`;
  }
  if (staves > 1) detail += ` · staff ${staff + 1}/${staves}`;
  paintRecognizeProgress(detail);
}

function paintRecognizeProgress(detail) {
  // On a document walk the headline number is the DOCUMENT's — pages already
  // finished plus how far into the current page we are. Without that second
  // term a 40-page run would sit frozen for minutes at a time, which reads as
  // a hang. Capped at 99: only the returned result may show a full bar.
  const page = recognizeProgress?.done ?? 0;
  const overall =
    documentIndex === null || documentTotal === 0
      ? page
      : (documentIndex + page) / documentTotal;
  const pct = Math.min(Math.round(overall * 100), 99);
  $("progress-wrap").hidden = false;
  $("progress-bar").style.width = `${pct}%`;
  const prefix =
    documentIndex === null
      ? ""
      : `Page ${pageLedger[documentIndex]?.page ?? documentIndex + 1} of ${documentTotal} · ${documentDone} done · `;
  $("progress-text").textContent = `${prefix}${pct}% (estimated) · ${detail}`;
}

function hideRecognizeProgress() {
  recognizeProgress = null;
  $("progress-wrap").hidden = true;
  $("progress-bar").style.width = "0%";
  $("progress-text").textContent = "";
}

// ── model selection ─────────────────────────────────────────────────────────
// Per-lane specimens. Every one of these ships in site/assets, so the sample
// button never depends on the network beyond this origin.
const SAMPLES = {
  "unlimited-ocr": "sample-doc.png",
  tromr: "sample-staff.png",
  "got-ocr2": "sample-table.png",
  smolvlm2: "sample-photo.png",
};
// Short names for the result meta line. An unknown id falls back to itself.
const SHORT_NAMES = {
  "unlimited-ocr": "Unlimited-OCR",
  tromr: "TrOMR",
  "got-ocr2": "GOT-OCR2",
  smolvlm2: "SmolVLM2",
};
// MusicXML has no useful rendered view, so the toggle is hidden for it.
const isMusic = (id) => id === "tromr";

function refreshModelUi() {
  const mb = totalBytes(model) / 1048576;
  $("model-size").textContent = mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(0)} MB`;
  $("consent-text").textContent =
    `This downloads the ${$("model-size").textContent} ${model.label} model into your ` +
    `browser's cache. After that it works offline.` +
    (model.desktopOnly
      ? " It needs several GB of memory to hydrate, so it is desktop-only and the download is large."
      : "") +
    " Continue?";
  // The question box belongs to one lane; showing it anywhere else would
  // promise something the other three models cannot do.
  $("question-wrap").hidden = modelId !== "smolvlm2";
}

$("model-select").addEventListener("change", (e) => {
  modelId = e.target.value;
  model = MODELS[modelId];
  engineReady = false; // a different model must be loaded before running
  $("load-model").disabled = false;
  $("consent").hidden = true;
  refreshModelUi();
  setStatus(`Selected ${model.label}. Load the model to begin.`);
  refreshRunButton();
});

// ── boot ────────────────────────────────────────────────────────────────────
async function boot() {
  // A soft reload restores the <select>'s previous value, which would otherwise
  // disagree with `modelId`. The control on screen is the truth; adopt it.
  const restored = $("model-select").value;
  if (MODELS[restored]) {
    modelId = restored;
    model = MODELS[modelId];
  }
  refreshModelUi();
  if (lastCrumb && ["download", "stage-weights", "hydrate"].includes(lastCrumb)) {
    setStatus(
      `The last visit ended during "${lastCrumb}". If this keeps happening, use the reset page.`,
      "warn",
    );
    $("reset-hint").hidden = false;
  }
  try {
    const { info } = await call("init");
    $("route").textContent = `${info.detected_tier} / ${info.dense_route}`;
    setStatus("Engine ready. Load the model to begin.", "ok");
    $("load-model").disabled = false;
  } catch (err) {
    setStatus(`Engine failed to start: ${err.message}`, "err");
  }
}

// ── model loading (consent-gated) ───────────────────────────────────────────
$("load-model").addEventListener("click", () => {
  $("consent").hidden = false;
  $("load-model").disabled = true;
});
$("consent-no").addEventListener("click", () => {
  $("consent").hidden = true;
  $("load-model").disabled = false;
});
$("consent-yes").addEventListener("click", async () => {
  $("consent").hidden = true;
  setStatus("Downloading + verifying model…");
  try {
    const r = await call("load", { model: modelId });
    engineReady = true;
    $("progress-wrap").hidden = true;
    $("license").textContent = r.license;
    setStatus(`Model ready (${r.model_id}, int8 route: ${r.route}). Drop an image.`, "ok");
    $("drop").classList.add("armed");
    refreshRunButton();
  } catch (err) {
    // wasm memory never shrinks, so an allocation failure usually means an
    // earlier model's pages are still held; a fresh tab starts clean.
    const oom = /memory allocation failed|cannot reserve/.test(err.message);
    setStatus(
      `Model load failed: ${err.message}` +
        (oom ? " Reload this tab and load the model first; the download is cached." : ""),
      "err",
    );
    $("load-model").disabled = false;
  }
});

$("clear-cache").addEventListener("click", async () => {
  const { clearModelCache } = await import(`./loader.js?v=@SITEV@`);
  await clearModelCache();
  localStorage.removeItem(CRUMB_KEY);
  setStatus("Model cache cleared.", "ok");
});

// ── image input: picker, drag-drop, paste, sample, PDF ─────────────────────
// A PDF keeps its bytes here; pages rasterize on demand in the worker through
// the same pure-Rust renderer the CLI uses, and the resulting PNG feeds the
// ordinary recognize path unchanged.
let currentPdf = null; // {bytes: ArrayBuffer, name, pages}

async function acceptPdf(file) {
  const bytes = await file.arrayBuffer();
  try {
    // The worker keeps the bytes for the whole document; nothing re-sends them
    // per page.
    // ONE copy, both named in the payload and listed for transfer. Passing two
    // separate `slice(0)` calls would put a buffer in the transfer list that is
    // not in the message: the payload would be structured-CLONED (a full copy of
    // the document) and the transferred buffer discarded — the exact copy this
    // is meant to avoid.
    const owned = bytes.slice(0);
    const { info } = await call("pdf-open", { bytes: owned }, [owned]);
    currentPdf = { name: file.name, pages: info.pages };
    $("pdf-count").textContent = String(info.pages);
    $("pdf-page").max = String(info.pages);
    $("pdf-page").value = "1";
    $("pdf-bar").hidden = false;
    // Render the first page BEFORE the summary, so the summary is what the
    // reader is left looking at rather than a transient "rendering…" line.
    await loadPdfPage(1);
    setStatus(
      `PDF opened: ${info.pages} page(s). Recognize reads the whole document.`,
      "ok",
    );
  } catch (err) {
    currentPdf = null;
    $("pdf-bar").hidden = true;
    setStatus(`PDF failed to open: ${err.message}`, "err");
  }
}

async function loadPdfPage(pageNo) {
  if (!currentPdf) return;
  setStatus(`Rendering PDF page ${pageNo}…`);
  try {
    const { png } = await call("pdf-render-page", { page: pageNo });
    const blob = new Blob([png], { type: "image/png" });
    if (currentImage?.url) URL.revokeObjectURL(currentImage.url);
    const stem = currentPdf.name.replace(/\.pdf$/i, "");
    currentImage = {
      bytes: await blob.arrayBuffer(),
      url: URL.createObjectURL(blob),
      name: `${stem}_p${pageNo}.png`,
    };
    showPreview();
    setStatus(`Page ${pageNo} of ${currentPdf.pages} shown.`, "ok");
    refreshRunButton();
    return true;
  } catch (err) {
    // JPXDecode/JBIG2/vector pages come back with the engine's precise
    // error string; show it verbatim rather than a vague failure.
    setStatus(`Page ${pageNo}: ${err.message}`, "warn");
    return false;
  }
}

// ── page selection ─────────────────────────────────────────────────────────
// Same grammar as the CLI's `--pages`: a comma-separated list of 1-based pages
// and inclusive ranges. Empty means the whole document.
function selectedPages() {
  const total = currentPdf?.pages ?? 0;
  if (!total) return [];
  const spec = ($("pdf-range")?.value ?? "").trim();
  if (!spec) return Array.from({ length: total }, (_, i) => i + 1);

  // Out-of-range and malformed pages THROW rather than being dropped, matching
  // `pdf::select_pages` in the engine and the documented CLI behavior. Silently
  // ignoring "9" on a 3-page document leaves the reader believing page 9 was
  // read; the error names the real count instead.
  const bad = (what) => {
    throw new Error(
      `Pages "${spec}": ${what} (expected 1-based pages/ranges like "1,5-9"; ` +
        `this document has ${total} page(s))`,
    );
  };
  const one = (tok) => {
    const t = tok.trim();
    if (!/^\d+$/.test(t)) bad(`unparseable page "${t}"`);
    const n = Number(t);
    if (n === 0) bad("page 0 (pages are 1-based)");
    if (n > total) bad(`page ${n} is out of range`);
    return n;
  };
  const wanted = new Set();
  for (const part of spec.split(",")) {
    const piece = part.trim();
    if (!piece) bad("empty element");
    const dash = piece.indexOf("-");
    if (dash > 0) {
      const lo = one(piece.slice(0, dash));
      const hi = one(piece.slice(dash + 1));
      if (lo > hi) bad(`reversed range "${piece}"`);
      for (let p = lo; p <= hi; p++) wanted.add(p);
    } else {
      wanted.add(one(piece));
    }
  }
  return [...wanted].sort((a, b) => a - b);
}

/// Mirrors `pdf::is_fatal_to_document`: does this end the whole walk, or just
/// this page? The worker prefixes engine errors with the failing stage, and a
/// cancelled run is signalled by the flag rather than the message.
function isFatalToDocument(message) {
  return /model (artifact )?(was )?not found|no model is loaded|format|version mismatch|cancell?ed|budget|timed? ?out/i.test(
    message,
  );
}

// Exposed for site/harness/pdf-document.mjs so the page-range grammar is tested
// against THIS implementation rather than a copy of it that can drift.
window.__focrSelectedPages = selectedPages;

// ── per-page ledger ────────────────────────────────────────────────────────
// One row per page: what happened, and why if it did not. A page that cannot be
// rasterized is recorded and skipped — one JPEG-2000 page in a 300-page scan
// must not throw away the other 299.
let pageLedger = []; // [{page, state, chars, ms, reason, output}]

function resetLedger(pages) {
  pageLedger = pages.map((page) => ({ page, state: "queued" }));
  paintLedger();
}

function paintLedger() {
  const host = $("page-ledger");
  if (!host) return;
  host.hidden = pageLedger.length === 0;
  host.innerHTML = "";
  for (const row of pageLedger) {
    const li = document.createElement("li");
    li.className = `ledger-row ledger-${row.state}`;
    const num = document.createElement("span");
    num.className = "ledger-num";
    num.textContent = String(row.page);
    const what = document.createElement("span");
    what.className = "ledger-what";
    if (row.state === "done") {
      what.textContent = `${row.chars.toLocaleString()} characters · ${(row.ms / 1000).toFixed(1)}s`;
    } else if (row.state === "skipped") {
      what.textContent = `skipped: ${row.reason}`;
    } else if (row.state === "running") {
      what.textContent = "reading…";
    } else {
      what.textContent = "queued";
    }
    li.append(num, what);
    host.appendChild(li);
  }
}

/// The whole document's text, with page markers — the shape the CLI writes.
function ledgerDocument() {
  return pageLedger
    .filter((r) => r.state === "done" || r.state === "skipped")
    .map((r) =>
      r.state === "done"
        ? `<!-- page ${r.page} -->\n\n${r.output}`
        : `<!-- page ${r.page} skipped: ${r.reason} -->`,
    )
    .join("\n\n");
}

$("pdf-load").addEventListener("click", () => {
  const n = Math.max(1, Math.min(Number($("pdf-page").value) || 1, currentPdf?.pages ?? 1));
  $("pdf-page").value = String(n);
  loadPdfPage(n);
});

function acceptFile(file) {
  if (!file) return;
  const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name);
  if (isPdf) {
    acceptPdf(file);
    return;
  }
  if (!/^image\/(png|jpe?g)$/.test(file.type)) {
    setStatus("PNG, JPEG, or PDF only.", "warn");
    return;
  }
  currentPdf = null;
  $("pdf-bar").hidden = true;
  file.arrayBuffer().then((bytes) => {
    if (currentImage?.url) URL.revokeObjectURL(currentImage.url);
    currentImage = { bytes, url: URL.createObjectURL(file), name: file.name };
    showPreview();
    refreshRunButton();
  });
}

$("file-input").addEventListener("change", (e) => acceptFile(e.target.files[0]));
const drop = $("drop");
drop.addEventListener("click", () => $("file-input").click());
drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  acceptFile(e.dataTransfer.files[0]);
});
document.addEventListener("paste", (e) => {
  const item = [...e.clipboardData.items].find((i) => i.type.startsWith("image/"));
  if (item) acceptFile(item.getAsFile());
});
$("sample").addEventListener("click", async () => {
  const name = SAMPLES[modelId] ?? "sample-doc.png";
  const resp = await fetch(`./assets/${name}?v=@SITEV@`);
  const blob = await resp.blob();
  acceptFile(new File([blob], name, { type: "image/png" }));
});

function showPreview() {
  const img = $("preview");
  img.src = currentImage.url;
  img.hidden = false;
  $("drop-hint").hidden = true;
  $("overlay").hidden = true;
}

function refreshRunButton() {
  $("run").disabled = !(engineReady && currentImage && !busy);
}

// ── recognition ─────────────────────────────────────────────────────────────
$("run").addEventListener("click", async () => {
  if (busy) return;
  busy = true;
  cancelled = false;
  refreshRunButton();
  $("cancel").hidden = false;
  resetRecognizeProgress();
  paintRecognizeProgress("Starting…");
  try {
    if (currentPdf) {
      await runDocument();
    } else {
      await runSinglePage();
    }
  } catch (err) {
    setStatus(`Recognition failed: ${err.message}`, "err");
  } finally {
    busy = false;
    hideRecognizeProgress();
    $("cancel").hidden = true;
    refreshRunButton();
  }
});

async function runSinglePage() {
  setStatus("Recognizing… (all compute stays in this tab)");
  pageLedger = [];
  paintLedger();
  const bytes = currentImage.bytes.slice(0);
  // Only SmolVLM2 reads a question; the worker ignores it for the other lanes.
  const question = modelId === "smolvlm2" ? $("question").value.trim() : "";
  const { result, ms } = await call("recognize", { bytes, question }, [bytes]);
  lastResult = result;
  renderResult(result, ms);
  setStatus(`Done in ${(ms / 1000).toFixed(1)}s on this device.`, "ok");
}

/// Walk every selected page of the open document.
///
/// Sequential on purpose: one wasm engine, one forward at a time. Running pages
/// concurrently would not be faster and would multiply peak linear memory by the
/// number of pages in flight — which on the lane that already peaks near the
/// wasm32 ceiling is how you lose the tab.
async function runDocument() {
  let pages;
  try {
    pages = selectedPages();
  } catch (err) {
    // A bad page spec is a usage error, reported before any work starts.
    setStatus(err.message, "warn");
    return;
  }
  if (!pages.length) {
    setStatus("That page selection matched no pages.", "warn");
    return;
  }
  resetLedger(pages);
  documentTotal = pages.length;
  documentDone = 0;

  const started = performance.now();
  const question = modelId === "smolvlm2" ? $("question").value.trim() : "";

  for (let i = 0; i < pages.length; i++) {
    if (cancelled) break;
    const page = pages[i];
    const row = pageLedger[i];
    row.state = "running";
    documentIndex = i;
    paintLedger();
    setStatus(`Page ${page} of ${currentPdf.pages} — ${documentDone} done`);

    // Render. A failure here is this page's problem, not the run's.
    let png;
    try {
      ({ png } = await call("pdf-render-page", { page }));
    } catch (err) {
      row.state = "skipped";
      row.reason = err.message;
      paintLedger();
      continue;
    }
    if (cancelled) break;

    // Show the page being read.
    const blob = new Blob([png], { type: "image/png" });
    if (currentImage?.url) URL.revokeObjectURL(currentImage.url);
    const stem = currentPdf.name.replace(/\.pdf$/i, "");
    const bytes = await blob.arrayBuffer();
    currentImage = {
      bytes,
      url: URL.createObjectURL(blob),
      name: `${stem}_p${page}.png`,
    };
    showPreview();

    const pageStarted = performance.now();
    try {
      resetRecognizeProgress();
      // One copy, transferred (see the note in acceptPdf). `bytes` itself stays
      // with `currentImage` for the preview, so it must not be the transferred
      // buffer — transferring detaches it.
      const owned = bytes.slice(0);
      const { result } = await call("recognize", { bytes: owned, question }, [owned]);
      row.state = "done";
      row.output = result.output;
      row.chars = result.output.length;
      row.ms = performance.now() - pageStarted;
      documentDone++;
      lastResult = result;
    } catch (err) {
      if (cancelled) break;
      // A document-level failure ends the run; a page-level one skips the page.
      if (isFatalToDocument(err.message)) throw err;
      row.state = "skipped";
      row.reason = err.message;
    }
    paintLedger();
  }

  documentIndex = null;
  const seconds = (performance.now() - started) / 1000;
  const skipped = pageLedger.filter((r) => r.state === "skipped").length;
  // Promote the WHOLE document to `lastResult`, not just the page that finished
  // last. The download handlers read `lastResult.output`, so leaving the final
  // page there would show the whole document on screen while saving one page of
  // it — silent data loss. Gated on a page having actually read this run:
  // `lastResult` survives across runs, so an all-skipped document would
  // otherwise redraw the previous document's layout and music payload.
  if (documentDone > 0 && lastResult) {
    lastResult = { ...lastResult, output: ledgerDocument() };
    renderResult(lastResult, seconds * 1000);
  }
  setStatus(
    cancelled
      ? `Cancelled after ${documentDone} of ${pages.length} pages.`
      : `${documentDone} of ${pages.length} pages in ${seconds.toFixed(0)}s` +
        `${skipped ? ` · ${skipped} skipped` : ""}, entirely on this device.`,
    skipped || cancelled ? "warn" : "ok",
  );
}
$("cancel").addEventListener("click", () => {
  // Stop the WHOLE document, not just the page in flight: the engine cancels
  // the current forward, and the flag stops the loop from starting the next.
  cancelled = true;
  call("cancel");
});

function renderResult(result, ms) {
  $("result-wrap").hidden = false;
  $("output").textContent = result.output;

  // staff overlay on the preview
  const music = result.music;
  const meta = [];
  if (music) {
    meta.push(`${music.staves.length} staff/staves recognized`);
    if (music.skips.length) meta.push(`${music.skips.length} skipped`);
    if (music.warnings.length) meta.push(`${music.warnings.length} sanity warning(s)`);
    drawOverlay(music);
  } else {
    // No music payload: name the lane and the size of what it wrote. Nothing
    // here assumes staves, so a document, a table or a caption all read right.
    meta.push(SHORT_NAMES[result.model_id] ?? result.model_id);
    meta.push(`${result.output.length.toLocaleString()} characters`);
  }
  $("result-meta").textContent = meta.join(" · ");

  // The staff legend and the sanity-warning footnote describe music and only
  // music; they stay hidden on every other lane.
  $("music-legend").hidden = !music;
  $("music-note").hidden = !music;

  // Downloads: raw bytes always, plus a styled self-contained page wherever a
  // rendered view exists.
  const musicLane = Boolean(music) || isMusic(result.model_id);
  $("download-xml").hidden = false;
  $("download-xml").textContent = musicLane ? "Download .musicxml" : "Download .md";
  $("download-html").hidden = musicLane;

  // Rendered / Source. MusicXML has no rendered mode, so it keeps Source only.
  $("view-toggle").hidden = musicLane;
  if (musicLane) {
    setResultView("source", { silent: true });
  } else {
    // A sandboxed frame is same-origin-less, so the page cannot measure what it
    // drew. Estimate instead: a two-line caption should not sit in a 24rem box,
    // and a long page should stop growing and scroll like the <pre> does.
    const rows = (result.output.match(/<tr\b/gi) ?? []).length;
    const textLines = result.output.split("\n").filter((l) => l.trim()).length;
    const est = Math.min(Math.max(textLines * 1.9 + rows * 2.1 + 3, 10), 26);
    $("rendered").style.height = `${est.toFixed(1)}rem`;
    $("rendered").srcdoc = renderedDocument(result.output);
    setResultView(resultView);
  }

  const warnings = music?.warnings ?? [];
  const wl = $("warnings");
  wl.innerHTML = "";
  for (const w of warnings) {
    const li = document.createElement("li");
    li.textContent = `${w.kind} (part ${w.part}, measure ${w.measure}): ${w.detail}`;
    wl.appendChild(li);
  }
  for (const s of music?.skips ?? []) {
    const li = document.createElement("li");
    li.textContent = `staff ${s.index + 1} skipped: ${s.reason}`;
    wl.appendChild(li);
  }
}

function drawOverlay(music) {
  const img = $("preview");
  const canvas = $("overlay");
  if (!img.naturalWidth) return;
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.hidden = false;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = Math.max(2, canvas.width / 400);
  for (const s of music.staves) {
    const [x, y, w, h] = s.bbox;
    ctx.strokeStyle = "rgba(52, 211, 153, 0.9)";
    ctx.strokeRect(x, y, w, h);
  }
  for (const s of music.skips) {
    const [x, y, w, h] = s.bbox;
    ctx.strokeStyle = "rgba(248, 113, 113, 0.9)";
    ctx.strokeRect(x, y, w, h);
  }
}

// ── rendered result view ────────────────────────────────────────────────────
//
// The document models emit Markdown that EMBEDS raw <table> HTML: headings,
// image references, and table blocks with rowspan/colspan. Two rules govern
// what happens to those bytes.
//
// 1. They are model output, so they are untrusted. The rendered view lives in
//    an iframe with sandbox="" — no scripts, no same-origin, no forms — and the
//    embedded HTML is rewritten through a tag/attribute ALLOWLIST first, so
//    nothing but tables survives and no on* handler can exist by construction.
// 2. The Markdown itself is rendered by the small hand-written pass below
//    rather than by a library, because a CDN is not an option here and the
//    subset the models actually emit is this short.

const BLOCK_TAGS = new Set(["table", "thead", "tbody", "tfoot", "tr", "td", "th", "caption"]);
const BLOCK_ATTRS = new Set(["rowspan", "colspan"]);

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

// Rewrite an embedded HTML span down to the table allowlist. Anything outside
// it loses its tag and keeps only its text; script/style/iframe-class elements
// lose their contents too, since their text is not prose.
function sanitizeEmbedded(html) {
  let s = html.replace(/<!--[\s\S]*?-->/g, "");
  s = s.replace(/<(script|style|iframe|object|embed|noscript|template)\b[\s\S]*?<\/\1\s*>/gi, "");
  s = s.replace(/<(script|style|iframe|object|embed|noscript|template|link|meta|base)\b[^>]*>/gi, "");
  return s.replace(
    /<(\/?)([a-zA-Z][a-zA-Z0-9-]*)((?:[^>"']|"[^"]*"|'[^']*')*)>/g,
    (_m, slash, name, attrs) => {
      const tag = name.toLowerCase();
      if (!BLOCK_TAGS.has(tag)) return ""; // unwrap: the text stays, the tag goes
      if (slash) return `</${tag}>`;
      const kept = [];
      for (const a of attrs.matchAll(
        /([a-zA-Z_:][-\w:.]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))/g,
      )) {
        const key = a[1].toLowerCase();
        if (!BLOCK_ATTRS.has(key)) continue; // this is what kills every on* handler
        const digits = (a[2] ?? a[3] ?? a[4] ?? "").replace(/\D/g, "").slice(0, 3);
        if (digits) kept.push(`${key}="${digits}"`);
      }
      return `<${tag}${kept.length ? ` ${kept.join(" ")}` : ""}>`;
    },
  );
}

function renderInline(text) {
  let t = escapeHtml(text);
  // Images never resolve client-side: the model writes references like
  // images/0.jpg into a directory that does not exist here. Say so plainly
  // instead of drawing a broken-image icon.
  t = t.replace(
    /!\[[^\]]*\]\([^)]*\)/g,
    '<span class="figure">figure (not extracted in-browser)</span>',
  );
  t = t.replace(/\[([^\]]*)\]\([^)]*\)/g, "$1");
  t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  return t;
}

function renderMarkdownBlocks(md) {
  const out = [];
  const lines = md.split("\n");
  let para = [];
  let list = null; // {tag, items}
  const flushPara = () => {
    if (para.length) out.push(`<p>${renderInline(para.join(" "))}</p>`);
    para = [];
  };
  const flushList = () => {
    if (list) out.push(`<${list.tag}>${list.items.join("")}</${list.tag}>`);
    list = null;
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^\s*```/.test(line)) {
      flushPara();
      flushList();
      const body = [];
      for (i++; i < lines.length && !/^\s*```/.test(lines[i]); i++) body.push(lines[i]);
      out.push(`<pre>${escapeHtml(body.join("\n"))}</pre>`);
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flushPara();
      flushList();
      const level = heading[1].length;
      out.push(`<h${level}>${renderInline(heading[2].trim())}</h${level}>`);
      continue;
    }
    const item = line.match(/^\s*(?:([-*+])|(\d+)[.)])\s+(.*)$/);
    if (item) {
      flushPara();
      const tag = item[1] ? "ul" : "ol";
      if (!list || list.tag !== tag) {
        flushList();
        list = { tag, items: [] };
      }
      list.items.push(`<li>${renderInline(item[3])}</li>`);
      continue;
    }
    if (/^\s*$/.test(line)) {
      flushPara();
      flushList();
      continue;
    }
    if (/^\s*([-*_])\s*\1\s*\1[\s\-*_]*$/.test(line)) {
      flushPara();
      flushList();
      out.push("<hr>");
      continue;
    }
    flushList();
    para.push(line.trim());
  }
  flushPara();
  flushList();
  return out.join("\n");
}

// Markdown outside the embedded HTML, embedded HTML passed through the
// allowlist. Splitting first keeps a <table> from being mangled by the
// paragraph pass, and keeps the Markdown from being trusted as HTML.
function renderMarkdown(source) {
  const parts = [];
  const re = /<table\b[\s\S]*?<\/table\s*>/gi;
  let cursor = 0;
  for (let m = re.exec(source); m; m = re.exec(source)) {
    parts.push(renderMarkdownBlocks(source.slice(cursor, m.index)));
    parts.push(`<div class="tbl">${sanitizeEmbedded(m[0])}</div>`);
    cursor = m.index + m[0].length;
  }
  parts.push(renderMarkdownBlocks(source.slice(cursor)));
  return parts.filter(Boolean).join("\n");
}

// One stylesheet, inlined in both the sandboxed preview and the downloadable
// page, so the two look identical and neither one fetches anything.
const RENDER_CSS = `
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 1.15rem 1.25rem;
  background: #050a08; color: #cbd5e1;
  font: 400 15px/1.7 Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}
h1, h2, h3, h4, h5, h6 { color: #e8eef2; font-weight: 800; letter-spacing: -0.02em;
  line-height: 1.22; margin: 1.6rem 0 0.6rem; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.25rem; } h3 { font-size: 1.08rem; }
h4, h5, h6 { font-size: 0.98rem; }
:is(h1, h2, h3, h4, h5, h6):first-child { margin-top: 0; }
p { margin: 0 0 0.85rem; }
ul, ol { margin: 0 0 0.9rem; padding-left: 1.3rem; }
li { margin-bottom: 0.28rem; }
strong { color: #e8eef2; font-weight: 700; }
em { color: #cbd5e1; }
hr { border: 0; border-top: 1px solid rgba(255,255,255,0.09); margin: 1.5rem 0; }
code, pre { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace; }
code { font-size: 0.86em; background: rgba(2,6,5,0.66); border: 1px solid rgba(255,255,255,0.07);
  border-radius: 5px; padding: 0.08em 0.34em; }
pre { background: rgba(2,6,5,0.66); border: 1px solid rgba(255,255,255,0.07);
  border-radius: 12px; padding: 0.9rem 1rem; overflow-x: auto; font-size: 0.8rem; }
pre code { background: none; border: 0; padding: 0; }
.tbl { overflow-x: auto; margin: 0 0 1.1rem; }
table { border-collapse: collapse; width: 100%; min-width: 380px; font-size: 0.86rem; }
th, td { border: 1px solid rgba(255,255,255,0.11); padding: 0.5rem 0.65rem;
  text-align: left; vertical-align: top; }
th { color: #34d399; font-weight: 700; background: rgba(16,185,129,0.05); }
.figure { display: inline-block; border: 1px dashed rgba(255,255,255,0.16); border-radius: 8px;
  padding: 0.7rem 1rem; color: #748496; font-size: 0.82rem; font-style: italic; }
.credit { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.09);
  color: #748496; font-size: 0.78rem; }
.credit a { color: #34d399; }
`.trim();

function renderedDocument(source) {
  return `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>${RENDER_CSS}</style></head><body>${renderMarkdown(source)}</body></html>`;
}

function styledDocument(source, title) {
  return `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(title)}</title>
<style>${RENDER_CSS}</style></head><body>
${renderMarkdown(source)}
<p class="credit">Recognized locally in the browser by franken_ocr &middot;
<a href="https://franken-ocr.com">franken-ocr.com</a></p>
</body></html>`;
}

// ── result view toggle ──────────────────────────────────────────────────────
let resultView = "rendered";

function setResultView(view, { silent = false } = {}) {
  if (!silent) resultView = view;
  const rendered = view === "rendered";
  $("output").hidden = rendered;
  $("rendered").hidden = !rendered;
  $("view-rendered").setAttribute("aria-pressed", String(rendered));
  $("view-source").setAttribute("aria-pressed", String(!rendered));
}

$("view-rendered").addEventListener("click", () => setResultView("rendered"));
$("view-source").addEventListener("click", () => setResultView("source"));

// ── downloads ───────────────────────────────────────────────────────────────
function saveBlob(blob, filename) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function baseName() {
  return currentImage?.name?.replace(/\.[^.]+$/, "") || "output";
}

// #download-xml keeps its id: the differential harness and the smoke both know
// this button, and MusicXML is still what it hands back on the music lane.
$("download-xml").addEventListener("click", () => {
  if (!lastResult) return;
  const musicLane = isMusic(lastResult.model_id) || Boolean(lastResult.music);
  saveBlob(
    new Blob([lastResult.output], {
      type: musicLane ? "application/vnd.recordare.musicxml+xml" : "text/markdown",
    }),
    baseName() + (musicLane ? ".musicxml" : ".md"),
  );
});

$("download-html").addEventListener("click", () => {
  if (!lastResult) return;
  saveBlob(
    new Blob([styledDocument(lastResult.output, baseName())], { type: "text/html" }),
    `${baseName()}.html`,
  );
});

boot();
