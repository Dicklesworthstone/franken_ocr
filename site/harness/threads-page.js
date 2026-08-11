// Threaded-lane probe page (bd-rhk7). Talks the engine-worker protocol
// DIRECTLY — no UI — because the honest evidence for a threaded build is the
// worker's own `load:ok` reply (`threads`, `threaded`, `pkg`), and the
// playground UI has no reason to surface it.
//
// It is the SHIPPING site/engine-worker.js under test here, loaded from the
// site root so its relative `./pkg…` URLs resolve exactly as they do in the
// real page. `?model=` selects the lane; `?sample=` the fixture.
//
// CSP note: this must be an external module (script-src 'self'), never inline.
const params = new URLSearchParams(location.search);
const modelId = params.get("model") ?? "tromr";
const sample = params.get("sample") ?? (modelId === "tromr" ? "sample-staff.png" : "sample-doc.png");

const out = { stages: [], done: false };
globalThis.__focr = out;

const worker = new Worker(`/engine-worker.js?v=dev`, { type: "module" });
const pending = new Map();
let nextId = 1;

worker.onmessage = ({ data }) => {
  if (data.type === "stage") {
    out.stages.push(data.stage);
    return;
  }
  if (data.type === "progress" || data.type === "plan") return;
  const p = pending.get(data.id);
  if (!p) return;
  pending.delete(data.id);
  if (data.type.endsWith(":err")) p.reject(new Error(data.error));
  else p.resolve(data);
};
worker.onerror = (e) => {
  out.error = `worker error: ${e.message}`;
  out.done = true;
};

function call(type, payload = {}, transfer = []) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, type, ...payload }, transfer);
  });
}

try {
  out.crossOriginIsolated = globalThis.crossOriginIsolated === true;
  out.hardwareConcurrency = navigator.hardwareConcurrency;
  const init = await call("init");
  out.init = { pkg: init.pkg, threaded: init.threaded, info: init.info };

  const t0 = performance.now();
  const load = await call("load", { model: modelId });
  out.load = {
    model_id: load.model_id,
    route: load.route,
    threads: load.threads,
    threaded: load.threaded,
    pkg: load.pkg,
    ms: Math.round(performance.now() - t0),
  };

  const bytes = new Uint8Array(await (await fetch(`/assets/${sample}`)).arrayBuffer());
  const rec = await call("recognize", { bytes: bytes.buffer }, [bytes.buffer]);
  out.recognize_ms = rec.ms;
  out.output = rec.result.output;
} catch (err) {
  out.error = String(err?.message ?? err);
} finally {
  out.done = true;
}
