// The engine worker: owns the wasm module and the loaded model. The page
// thread never runs inference — a 2–4 s recognize would freeze the tab.
//
// Two hard-won rules from the frankentts port, copied wholesale:
//
// 1. A module worker starts its event loop while the module body is still
//    evaluating; messages posted before `self.onmessage` exists are silently
//    dropped, and any top-level `await` opens that window. So: install a
//    buffering handler FIRST, before the dynamic import.
// 2. Handlers are async; interleaved messages must not race. Every delivery
//    is chained through one promise queue — and the `.catch` is not optional
//    (one rejected link would poison the chain forever).

const inbox = [];
let deliver = (e) => inbox.push(e);
self.onmessage = (e) => deliver(e);

let queue = Promise.resolve();
const serialize = (e) => {
  queue = queue.then(() => handleMessage(e)).catch(() => {});
};

// ── threaded-lane selection ─────────────────────────────────────────────────
//
// Two modules ship (site/build.sh): `pkg` is serial with an unshared memory,
// `pkg-threaded` is linked `--shared-memory --import-memory` with atomics and
// carries wasm-bindgen-rayon's worker pool. Sharedness is a LINK-TIME property,
// so this is a choice of module, not a flag.
//
// The choice is an ALLOW-LIST, not feature detection. `SharedArrayBuffer` also
// exists on iOS/WebKit under COOP/COEP, but growing a shared memory toward 2 GB
// kills the tab there — the Unlimited artifact is 2.8 GB. So: Blink only, and
// only when actually cross-origin isolated.
const THREADED_PKG_DIR = "./pkg-threaded";
const SERIAL_PKG_DIR = "./pkg";

function blinkAllowsThreads() {
  if (typeof SharedArrayBuffer !== "function") return false;
  if (self.crossOriginIsolated !== true) return false;
  const ua = self.navigator?.userAgent ?? "";
  // Every iOS browser is WebKit under the skin, whatever the brand in the UA.
  if (/iPhone|iPad|iPod|CriOS|FxiOS|EdgiOS/.test(ua)) return false;
  // Desktop Safari: AppleWebKit with no Chrome/ token.
  return /Chrom(e|ium)\/\d/.test(ua);
}

// The rayon workers are spawned from a blob URL of wasm-bindgen-rayon's own
// helper script. A `worker-src` CSP without `blob:` kills that — and it would
// kill it AFTER the 2.8 GB model is staged, when falling back is ruinous. So
// prove the blob-module-worker path with a byte-sized canary at init time,
// before a single weight byte is downloaded.
async function blobModuleWorkerWorks() {
  let url;
  try {
    url = URL.createObjectURL(
      new Blob(["self.postMessage('ok');"], { type: "text/javascript" }),
    );
    const w = new Worker(url, { type: "module" });
    try {
      await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error("canary worker timeout")), 5000);
        w.onmessage = () => {
          clearTimeout(timer);
          resolve();
        };
        w.onerror = (e) => {
          clearTimeout(timer);
          reject(new Error(e.message ?? "canary worker error"));
        };
      });
      return true;
    } finally {
      w.terminate();
    }
  } catch {
    return false;
  } finally {
    if (url) URL.revokeObjectURL(url);
  }
}

// Leave one core for the page/worker that drives the pool, and stop at 8: past
// that the m=1 GEMV blocks get too small to pay for the dispatch.
function targetThreads() {
  return Math.max(1, Math.min((self.navigator?.hardwareConcurrency ?? 4) - 1, 8));
}

let pkgDir = SERIAL_PKG_DIR;
let threaded = false; // the loaded module is the shared-memory one
let threads = 1; // rayon's ACTUAL worker count, straight from rayon
let pkg = null;
let engine = null; // WasmEngine
let modelId = null;
// The open document's bytes, held once for the whole run (see `pdf-open`).
let pdfBytes = null;
// The manifest key the page asked for. `modelId` is the engine's own id for the
// hydrated artifact, which need not be spelled the same way.
let modelKey = null;

function post(msg) {
  self.postMessage(msg);
}
// Stage breadcrumbs: posted BEFORE each dangerous step so the page can commit
// them to durable storage — the only evidence that survives a tab kill.
function stage(name) {
  post({ type: "stage", stage: name });
}

// ── live recognize progress ─────────────────────────────────────────────────
//
// `recognize_json` is ONE synchronous wasm call: this worker's event loop is
// blocked for its whole duration, so nothing asynchronous can report anything.
// The engine therefore calls back INTO JS from inside the forward's call stack
// (per vision block, per decoded token) and we relay from there — `postMessage`
// is legal on a blocked thread; the message is queued for the page.
//
// The callback can fire per token, so throttle: at most one message every
// ~200 ms (≈5/s), except that a stage CHANGE always posts immediately (the
// text is the point) and so does the terminal count of a determinate stage.
const PROGRESS_MIN_INTERVAL_MS = 200;
let relayProgress = false;
let lastProgressPost = 0;
let lastProgressStage = null;

function installProgressRelay() {
  if (typeof pkg?.set_progress_callback !== "function") return;
  pkg.set_progress_callback((stageName, current, total) => {
    if (!relayProgress) return;
    const now = performance.now();
    const isNewStage = stageName !== lastProgressStage;
    const isStageEnd = total > 0 && current >= total;
    if (!isNewStage && !isStageEnd && now - lastProgressPost < PROGRESS_MIN_INTERVAL_MS) return;
    lastProgressPost = now;
    lastProgressStage = stageName;
    post({ type: "recognize-progress", stage: stageName, current, total });
  });
}

async function handleMessage({ data }) {
  const { id, type } = data;
  try {
    const result = await dispatch(data);
    post({ id, type: `${type}:ok`, ...result });
  } catch (err) {
    post({ id, type: `${type}:err`, error: String(err?.message ?? err) });
  }
}

async function dispatch(data) {
  switch (data.type) {
    case "init": {
      stage("init");
      const wantThreads = blinkAllowsThreads() && (await blobModuleWorkerWorks());
      pkgDir = wantThreads ? THREADED_PKG_DIR : SERIAL_PKG_DIR;
      pkg = await import(`${pkgDir}/focr_wasm.js?v=@SITEV@`);
      const module = await WebAssembly.compileStreaming(
        fetch(`${pkgDir}/focr_wasm_bg.wasm?v=@SITEV@`),
      );
      await pkg.default({ module_or_path: module });
      // The build flags are a claim; the instantiated memory is the receipt.
      // A "threaded" module whose memory is a plain ArrayBuffer would run, and
      // run single-threaded, and never say so.
      threaded =
        wantThreads && pkg.wasm_memory().buffer instanceof SharedArrayBuffer;
      if (wantThreads && !threaded) {
        throw new Error("pkg-threaded instantiated with a non-shared memory");
      }
      threads = 1; // the pool is armed after hydration, never before
      installProgressRelay();
      return { info: JSON.parse(pkg.engine_info()), pkg: pkgDir, threaded };
    }
    case "load": {
      const { MODELS } = await import(`./model-manifest.js?v=@SITEV@`);
      const { loadModel } = await import(`./loader.js?v=@SITEV@`);
      const model = MODELS[data.model];
      if (!model) throw new Error(`unknown model ${data.model}`);

      // Decode-guard mitigation (labeled in the manifest, README-documented
      // knob `FOCR_NO_REPEAT_NGRAM`): must be set BEFORE the engine is built;
      // reset to the engine default when the model declares none.
      pkg.set_no_repeat_ngram(model.decodeGuard ?? 0);

      // Per-model knobs that exist as wasm exports only after the next
      // site/build.sh run. Guarded by `typeof` so the currently shipped pkg
      // still loads both new lanes at their defaults: GOT in plain mode,
      // SmolVLM2 caption-only. The knobs light up on rebuild with no further
      // change here.
      if (data.model === "got-ocr2" && typeof pkg.set_got_format === "function") {
        pkg.set_got_format(false); // plain mode; the format lane is a UI choice we do not offer yet
      }

      // Free the PREVIOUS model BEFORE staging the next one. The 4 GiB wasm
      // heap cannot hold two multi-GB models at once, and wasm memory never
      // shrinks — a model switch that staged first and freed at hydrate time
      // failed exactly here ("cannot reserve segment 0"). The cost is honest:
      // if this load fails, the previous model is gone too (its bytes are
      // still in the cache, so reloading it is a re-verify, not a re-download).
      if (engine) {
        engine.free_engine();
        engine = null;
        modelId = null;
      }

      stage("download");
      // The weight bytes STREAM straight from fetch (or the cache) into wasm
      // staging — Chrome refuses a single multi-GB ArrayBuffer, so they are
      // never materialized JS-side (loader.js owns hashing + caching).
      //
      // Two-phase staging (bd-syf2): wasm32 caps ONE allocation at 2 GiB, so a
      // multi-GB artifact is cut into ≤1 GiB segments AT TENSOR BOUNDARIES.
      // plan() reads the artifact's own header out of the streamed 4 MiB
      // prefix to find those boundaries; push() then appends in order.
      const staging = new pkg.ModelStaging();
      const sink = {
        begin: (prefix, totalBytes) => {
          let plan = JSON.parse(staging.plan(prefix, totalBytes));
          if (plan.status === "need_prefix") {
            const need = Number(plan.need_bytes);
            if (need > prefix.byteLength) {
              throw new Error(
                `staging.plan needs a ${need}-byte header prefix but only ${prefix.byteLength} bytes streamed`,
              );
            }
            plan = JSON.parse(staging.plan(prefix.subarray(0, need), totalBytes));
          }
          if (plan.status !== "planned") throw new Error(`staging.plan: ${plan.status}`);
          post({ type: "plan", segments: plan.segments });
          staging.push(prefix);
        },
        push: (chunk) => staging.push(chunk),
      };

      let lastPct = -1;
      let sidecars;
      try {
        ({ sidecars } = await loadModel(data.model, model, sink, (loaded, total, fromCache) => {
          const pct = Math.floor((loaded / total) * 100);
          if (pct !== lastPct) {
            lastPct = pct;
            post({ type: "progress", loaded, total, fromCache });
          }
        }));
      } catch (err) {
        staging.free(); // never keep unverified bytes staged
        throw err;
      }
      for (const s of sidecars) staging.set_sidecar(s.name, s.bytes);

      stage("hydrate");
      engine = pkg.WasmEngine.from_staging(staging);
      modelId = engine.model_id();
      modelKey = data.model;

      // ARM THE POOL ONLY NOW. A rayon worker parked in `Atomics.wait` blocks a
      // shared-memory `grow`, and everything above this line — staging the
      // segments, hydrating the engine — is nothing but grows. Arming before
      // hydration deadlocks the load; arming after costs nothing.
      if (threaded && threads === 1) {
        stage("threads");
        await pkg.initThreadPool(targetThreads());
        // rayon's own count, not the number we asked for.
        threads = pkg.thread_count();
      }

      stage("ready");
      return {
        model_id: modelId,
        license: engine.license_notice(),
        route: pkg.int8_route(),
        threads,
        threaded,
        pkg: pkgDir,
      };
    }
    case "recognize": {
      if (!engine) throw new Error("no model loaded");
      stage("recognize");
      // SmolVLM2's question, when the page supplied one AND the wasm build
      // exports the setter. Before that build lands the lane still runs, it
      // just captions instead of answering.
      if (
        modelKey === "smolvlm2" &&
        data.question &&
        typeof pkg.set_smolvlm2_question === "function"
      ) {
        pkg.set_smolvlm2_question(data.question);
      }
      pkg.reset_cancel();
      const t0 = performance.now();
      relayProgress = true;
      lastProgressPost = 0;
      lastProgressStage = null;
      try {
        const json = engine.recognize_json(new Uint8Array(data.bytes));
        const ms = Math.round(performance.now() - t0);
        stage("ready");
        return { result: JSON.parse(json), ms };
      } finally {
        relayProgress = false;
      }
    }
    case "cancel": {
      // Cooperative: the decode loop observes the flag at its next token.
      pkg?.request_cancel();
      return {};
    }
    // ── PDF support ──────────────────────────────────────────────────────
    // The same pure-Rust rasterizer the CLI uses (page /Rotate + placement
    // matrix normalization included). Pages with codecs that have no pure-Rust
    // decoder (JPXDecode, JBIG2) reject with the engine's precise error string —
    // surface it verbatim so a document walk can skip that page with a reason.
    //
    // The bytes are held HERE, once, rather than crossing per page. Recognizing
    // a whole 300-page scan would otherwise structured-clone the entire document
    // three hundred times, which for a 50 MB book is 15 GB of pure copying.
    case "pdf-open": {
      if (typeof pkg?.pdf_info !== "function") {
        throw new Error("this build has no PDF support (module predates it)");
      }
      // Parse FIRST, commit second: a document that fails to open must not be
      // left resident, and must not replace a document that opened fine.
      const candidate = new Uint8Array(data.bytes);
      const info = JSON.parse(pkg.pdf_info(candidate));
      pdfBytes = candidate;
      return { info };
    }
    case "pdf-close": {
      pdfBytes = null;
      return {};
    }
    case "pdf-render-page": {
      if (typeof pkg?.pdf_render_page !== "function") {
        throw new Error("this build has no PDF support (module predates it)");
      }
      if (!pdfBytes) throw new Error("no PDF is open");
      // Structured-clone copies the PNG back (a few MB per page; fine).
      return { png: pkg.pdf_render_page(pdfBytes, data.page) };
    }
    default:
      throw new Error(`unknown message type ${data.type}`);
  }
}

// Module body done — drain the inbox, then deliver directly.
deliver = serialize;
for (const e of inbox.splice(0)) serialize(e);
