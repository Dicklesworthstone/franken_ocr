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

const PKG = "./pkg/focr_wasm.js";

let pkg = null;
let engine = null; // WasmEngine
let modelId = null;

function post(msg) {
  self.postMessage(msg);
}
// Stage breadcrumbs: posted BEFORE each dangerous step so the page can commit
// them to durable storage — the only evidence that survives a tab kill.
function stage(name) {
  post({ type: "stage", stage: name });
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
      pkg = await import(`${PKG}?v=@SITEV@`);
      const module = await WebAssembly.compileStreaming(
        fetch(`./pkg/focr_wasm_bg.wasm?v=@SITEV@`),
      );
      await pkg.default({ module_or_path: module });
      return { info: JSON.parse(pkg.engine_info()) };
    }
    case "load": {
      const { MODELS } = await import(`./model-manifest.js?v=@SITEV@`);
      const { loadModel } = await import(`./loader.js?v=@SITEV@`);
      const model = MODELS[data.model];
      if (!model) throw new Error(`unknown model ${data.model}`);

      stage("download");
      let lastPct = -1;
      const { weights, sidecars } = await loadModel(data.model, model, (loaded, total, fromCache) => {
        const pct = Math.floor((loaded / total) * 100);
        if (pct !== lastPct) {
          lastPct = pct;
          post({ type: "progress", loaded, total, fromCache });
        }
      });

      stage("stage-weights");
      const staging = new pkg.ModelStaging();
      staging.reserve(weights.byteLength);
      const SLICE = 16 * 1024 * 1024;
      for (let off = 0; off < weights.byteLength; off += SLICE) {
        staging.push(weights.subarray(off, Math.min(off + SLICE, weights.byteLength)));
      }
      for (const s of sidecars) staging.set_sidecar(s.name, s.bytes);

      stage("hydrate");
      if (engine) {
        engine.free_engine();
        engine = null;
      }
      engine = pkg.WasmEngine.from_staging(staging);
      modelId = engine.model_id();
      stage("ready");
      return {
        model_id: modelId,
        license: engine.license_notice(),
        route: pkg.int8_route(),
      };
    }
    case "recognize": {
      if (!engine) throw new Error("no model loaded");
      stage("recognize");
      pkg.reset_cancel();
      const t0 = performance.now();
      const json = engine.recognize_json(new Uint8Array(data.bytes));
      const ms = Math.round(performance.now() - t0);
      stage("ready");
      return { result: JSON.parse(json), ms };
    }
    case "cancel": {
      // Cooperative: the decode loop observes the flag at its next token.
      pkg?.request_cancel();
      return {};
    }
    default:
      throw new Error(`unknown message type ${data.type}`);
  }
}

// Module body done — drain the inbox, then deliver directly.
deliver = serialize;
for (const e of inbox.splice(0)) serialize(e);
