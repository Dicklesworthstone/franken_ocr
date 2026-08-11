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
  const entry = pending.get(data.id);
  if (!entry) return;
  pending.delete(data.id);
  clearTimeout(entry.timer);
  if (data.type.endsWith(":err")) entry.reject(new Error(data.error));
  else entry.resolve(data);
};
worker.onerror = (e) => setStatus(`worker error: ${e.message}`, "err");

// ── UI state ────────────────────────────────────────────────────────────────
let modelId = "tromr";
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
    $("progress-text").textContent = `verified from cache — ${fmtMB(total)} MB`;
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
    (rate > 0 ? ` — ${fmtMB(rate)} MB/s${eta !== null ? `, ~${eta}s left` : ""}` : "");
}

// ── model selection ─────────────────────────────────────────────────────────
function refreshModelUi() {
  const mb = totalBytes(model) / 1048576;
  $("model-size").textContent = mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(0)} MB`;
  $("consent-text").textContent =
    `This downloads the ${$("model-size").textContent} ${model.label} model into your ` +
    `browser's cache. After that it works offline.` +
    (model.desktopOnly
      ? " It needs ~3.5 GB of memory — desktop browsers only, and the download is large."
      : "") +
    " Continue?";
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
    setStatus(`Model load failed: ${err.message}`, "err");
    $("load-model").disabled = false;
  }
});

$("clear-cache").addEventListener("click", async () => {
  const { clearModelCache } = await import(`./loader.js?v=@SITEV@`);
  await clearModelCache();
  localStorage.removeItem(CRUMB_KEY);
  setStatus("Model cache cleared.", "ok");
});

// ── image input: picker, drag-drop, paste, sample ──────────────────────────
function acceptFile(file) {
  if (!file || !/^image\/(png|jpe?g)$/.test(file.type)) {
    setStatus("PNG or JPEG images only.", "warn");
    return;
  }
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
  const name = modelId === "tromr" ? "sample-staff.png" : "sample-doc.png";
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
  refreshRunButton();
  $("cancel").hidden = false;
  setStatus("Recognizing… (all compute stays in this tab)");
  try {
    const bytes = currentImage.bytes.slice(0);
    const { result, ms } = await call("recognize", { bytes }, [bytes]);
    lastResult = result;
    renderResult(result, ms);
    setStatus(`Done in ${(ms / 1000).toFixed(1)}s on this device.`, "ok");
  } catch (err) {
    setStatus(`Recognition failed: ${err.message}`, "err");
  } finally {
    busy = false;
    $("cancel").hidden = true;
    refreshRunButton();
  }
});
$("cancel").addEventListener("click", () => call("cancel"));

function renderResult(result, ms) {
  $("result-wrap").hidden = false;
  $("output").textContent = result.output;
  $("download-xml").hidden = false;

  // staff overlay on the preview
  const music = result.music;
  const meta = [];
  if (music) {
    meta.push(`${music.staves.length} staff/staves recognized`);
    if (music.skips.length) meta.push(`${music.skips.length} skipped`);
    if (music.warnings.length) meta.push(`${music.warnings.length} sanity warning(s)`);
    drawOverlay(music);
  }
  $("result-meta").textContent = meta.join(" · ") || `model: ${result.model_id}`;

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

$("download-xml").addEventListener("click", () => {
  if (!lastResult) return;
  const isMusic = lastResult.model_id === "tromr";
  const blob = new Blob([lastResult.output], {
    type: isMusic ? "application/vnd.recordare.musicxml+xml" : "text/markdown",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download =
    (currentImage?.name?.replace(/\.[^.]+$/, "") || "output") + (isMusic ? ".musicxml" : ".md");
  a.click();
  URL.revokeObjectURL(a.href);
});

boot();
