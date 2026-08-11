// Verified model loader: fetch → hash → Cache Storage, resumable at the
// whole-asset level. Small models (the 58 MiB TrOMR artifact) download in one
// streamed pass with progress; every asset is SHA-256-verified against the
// pinned manifest BOTH on first download and on every warm start, so the
// cache can never serve silently corrupted bytes. (The frankentts OPFS +
// streaming-digest machinery is deliberately absent at this size: WebCrypto
// digests 61 MB in well under a second. When the multi-GB Unlimited lane
// lands, this file grows the endpoint-digest + OPFS discipline.)
//
// Runs inside the engine worker (Cache API is worker-visible).

const CACHE_NAME = "focr-models-v1";

async function sha256Hex(buf) {
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function assetUrl(modelId, name) {
  return `/model/${modelId}/${name}`;
}

/** Fetch one asset with streamed progress, verify size + digest, cache it. */
async function fetchVerified(modelId, spec, onProgress) {
  const url = assetUrl(modelId, spec.name);
  const cache = await caches.open(CACHE_NAME);

  const cached = await cache.match(url);
  if (cached) {
    const buf = await cached.arrayBuffer();
    if (buf.byteLength === spec.bytes && (await sha256Hex(buf)) === spec.sha256) {
      onProgress?.(spec.bytes, spec.bytes, true);
      return new Uint8Array(buf);
    }
    // Wrong bytes in cache: delete and refetch. Never trust, never patch.
    await cache.delete(url);
  }

  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`fetch ${spec.name}: HTTP ${resp.status}`);
  }
  const reader = resp.body.getReader();
  const out = new Uint8Array(spec.bytes);
  let offset = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (offset + value.byteLength > spec.bytes) {
      throw new Error(`${spec.name}: server sent more than the pinned ${spec.bytes} bytes`);
    }
    out.set(value, offset);
    offset += value.byteLength;
    onProgress?.(offset, spec.bytes, false);
  }
  if (offset !== spec.bytes) {
    throw new Error(`${spec.name}: got ${offset} of ${spec.bytes} bytes`);
  }
  const hex = await sha256Hex(out);
  if (hex !== spec.sha256) {
    throw new Error(`${spec.name}: SHA-256 mismatch (got ${hex.slice(0, 12)}…)`);
  }
  await cache.put(
    url,
    new Response(out, {
      headers: {
        "content-type": "application/octet-stream",
        "content-length": String(spec.bytes),
      },
    }),
  );
  return out;
}

/**
 * Download + verify every asset of `model` (manifest entry). Returns
 * `{ weights: Uint8Array, sidecars: [{name, bytes: Uint8Array}] }`.
 * `onProgress(loadedTotal, grandTotal, fromCache)` fires per chunk.
 */
export async function loadModel(modelId, model, onProgress) {
  const grandTotal = model.weights.bytes + model.sidecars.reduce((n, s) => n + s.bytes, 0);
  let baseLoaded = 0;
  const track = (spec) => (loaded, _total, fromCache) =>
    onProgress?.(baseLoaded + loaded, grandTotal, fromCache);

  // Sidecars first (tiny): any config/proxy failure surfaces in milliseconds,
  // before the big transfer starts.
  const sidecars = [];
  for (const spec of model.sidecars) {
    sidecars.push({ name: spec.name, bytes: await fetchVerified(modelId, spec, track(spec)) });
    baseLoaded += spec.bytes;
  }
  const weights = await fetchVerified(modelId, model.weights, track(model.weights));
  return { weights, sidecars };
}

/** Wipe the model cache (the reset page and the Clear button use this). */
export async function clearModelCache() {
  await caches.delete(CACHE_NAME);
}
