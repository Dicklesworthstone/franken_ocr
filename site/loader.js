// Verified model loader.
//
// Two lanes, one discipline (size + full SHA-256 against the pinned manifest,
// verified on first download AND on every warm start):
//
// * Sidecars (KB..MB): fetched, verified, materialized, cached — simple.
// * Weights (58 MB .. 3 GB): STREAMED. Chrome refuses a single multi-GB
//   ArrayBuffer ("Array buffer allocation failed", measured at 3.0 GB), so the
//   weight bytes are never materialized JS-side: each fetch chunk is hashed
//   incrementally (site/sha256.js — WebCrypto cannot stream), handed to the
//   wasm staging sink, and teed into the Cache API, which spools to disk.
//   Peak JS-side residency is one chunk plus the 4 MiB plan prefix.
//
// The sink contract (implemented by the engine worker over ModelStaging):
//   sink.begin(prefix, totalBytes)  — called once with the first >=PREFIX
//                                     bytes (or the whole asset if smaller);
//                                     the worker runs plan() and pushes it.
//   sink.push(chunk)                — every subsequent chunk, in order.
// Verification completes AFTER the stream; on mismatch the cache entry is
// deleted and the load throws — the worker never hydrates unverified staging.
//
// Runs inside the engine worker (Cache API is worker-visible).

import { Sha256 } from "./sha256.js";

const CACHE_NAME = "focr-models-v1";
const PREFIX_BYTES = 4 * 1024 * 1024;

function assetUrl(modelId, name) {
  return `/model/${modelId}/${name}`;
}

async function sha256HexOneShot(buf) {
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Small-asset lane: fetch, verify, materialize, cache. */
async function fetchVerifiedSmall(modelId, spec, onProgress) {
  const url = assetUrl(modelId, spec.name);
  const cache = await caches.open(CACHE_NAME);

  const cached = await cache.match(url);
  if (cached) {
    const buf = await cached.arrayBuffer();
    if (buf.byteLength === spec.bytes && (await sha256HexOneShot(buf)) === spec.sha256) {
      onProgress?.(spec.bytes, true);
      return new Uint8Array(buf);
    }
    await cache.delete(url);
  }

  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${spec.name}: HTTP ${resp.status}`);
  const buf = new Uint8Array(await resp.arrayBuffer());
  if (buf.byteLength !== spec.bytes) {
    throw new Error(`${spec.name}: got ${buf.byteLength} of ${spec.bytes} bytes`);
  }
  if ((await sha256HexOneShot(buf)) !== spec.sha256) {
    throw new Error(`${spec.name}: SHA-256 mismatch`);
  }
  await cache.put(url, new Response(buf, { headers: { "content-type": "application/octet-stream" } }));
  onProgress?.(spec.bytes, false);
  return buf;
}

/** Drive one ReadableStream through hash + sink + progress. */
async function pump(reader, spec, sink, onProgress) {
  const hash = new Sha256();
  let received = 0;
  let prefixParts = [];
  let prefixLen = 0;
  let began = false;

  const beginIfReady = (eof) => {
    if (began || (!eof && prefixLen < Math.min(PREFIX_BYTES, spec.bytes))) return;
    const prefix = new Uint8Array(prefixLen);
    let off = 0;
    for (const p of prefixParts) {
      prefix.set(p, off);
      off += p.length;
    }
    prefixParts = [];
    sink.begin(prefix, spec.bytes);
    began = true;
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (received + value.length > spec.bytes) {
      throw new Error(`${spec.name}: server sent more than the pinned ${spec.bytes} bytes`);
    }
    hash.update(value);
    received += value.length;
    if (!began) {
      prefixParts.push(value);
      prefixLen += value.length;
      beginIfReady(false);
    } else {
      sink.push(value);
    }
    onProgress?.(received, false);
  }
  beginIfReady(true); // short asset: whole thing is the prefix
  if (received !== spec.bytes) {
    throw new Error(`${spec.name}: got ${received} of ${spec.bytes} bytes`);
  }
  const hex = hash.hex();
  if (hex !== spec.sha256) {
    throw new Error(`${spec.name}: SHA-256 mismatch (got ${hex.slice(0, 12)}…)`);
  }
}

/** Big-asset lane: stream into `sink`, tee into the cache, verify by stream. */
async function streamWeights(modelId, spec, sink, onProgress) {
  const url = assetUrl(modelId, spec.name);
  const cache = await caches.open(CACHE_NAME);

  const cached = await cache.match(url);
  if (cached?.body) {
    try {
      await pump(cached.body.getReader(), spec, sink, (n) => onProgress?.(n, true));
      return;
    } catch (err) {
      // Corrupt cache: delete and fall through to a fresh download. The sink
      // may have consumed partial bytes — the caller must discard its staging.
      await cache.delete(url);
      throw new Error(`cached ${spec.name} failed verification (${err.message}); cache cleared — retry the load`);
    }
  }

  const resp = await fetch(url);
  if (!resp.ok || !resp.body) throw new Error(`fetch ${spec.name}: HTTP ${resp.status}`);
  const [toSink, toCache] = resp.body.tee();
  const cachePut = cache
    .put(url, new Response(toCache, { headers: { "content-type": "application/octet-stream" } }))
    .catch((err) => ({ cacheError: err }));
  try {
    await pump(toSink.getReader(), spec, sink, (n) => onProgress?.(n, false));
  } catch (err) {
    await cachePut;
    await cache.delete(url); // never keep bytes that failed verification
    throw err;
  }
  const putResult = await cachePut;
  if (putResult?.cacheError) {
    // Cache write failed (quota). The verified stream still reached the sink —
    // the model runs; it just won't be cached for next time.
    await cache.delete(url).catch(() => {});
    console.warn(`cache.put(${spec.name}) failed: ${putResult.cacheError} — model will re-download next visit`);
  }
}

/**
 * Load every asset of `model`: sidecars materialize (returned), weights
 * stream into `weightsSink`. `onProgress(loadedTotal, grandTotal, fromCache)`.
 */
export async function loadModel(modelId, model, weightsSink, onProgress) {
  const grandTotal = model.weights.bytes + model.sidecars.reduce((n, s) => n + s.bytes, 0);
  let base = 0;

  const sidecars = [];
  for (const spec of model.sidecars) {
    const bytes = await fetchVerifiedSmall(modelId, spec, (n, fromCache) =>
      onProgress?.(base + n, grandTotal, fromCache),
    );
    sidecars.push({ name: spec.name, bytes });
    base += spec.bytes;
  }
  await streamWeights(modelId, model.weights, weightsSink, (n, fromCache) =>
    onProgress?.(base + n, grandTotal, fromCache),
  );
  return { sidecars };
}

/** Wipe the model cache (the reset page and the Clear button use this). */
export async function clearModelCache() {
  await caches.delete(CACHE_NAME);
}
