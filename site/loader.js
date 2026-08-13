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

// Where a model file comes from, best source first.
//
// Hugging Face is fetched DIRECTLY, cross-origin. It serves ranged requests,
// sits behind a CDN, and — unlike GitHub release assets — sends CORS headers
// (`access-control-allow-origin`, reflected on the 302 and `*` on the CDN it
// redirects to), so the browser can stream from it without a same-origin hop.
// Going direct means multi-gigabyte model downloads never cross this site's own
// bandwidth bill.
//
// The same-origin Pages Function stays as the fallback: it forwards to the same
// two upstreams server-side, so a Hugging Face outage (or a CSP/extension that
// blocks the cross-origin fetch) still resolves.
//
// Every byte is SHA-256-verified against the pinned manifest no matter which
// source answered, so switching sources — even mid-asset, between parts — cannot
// yield a mixed or corrupt artifact.
const HF_BASE = "https://huggingface.co/Dicklesworthstone/franken_ocr-weights/resolve/main/";

// Per-model subdirectory inside the weights repo. Mirrors the layout in
// models/manifest-v2.json, which is the source of truth for what is published.
const HF_SUBDIR = {
  "unlimited-ocr": "",
  "got-ocr2": "",
  smolvlm2: "smolvlm2/",
  onechart: "onechart/",
  tromr: "tromr/",
};

/** Cache key. Stable across sources, so a fallback still hits a warm cache. */
function assetUrl(modelId, name) {
  return `/model/${modelId}/${name}`;
}

/** Candidate sources for one file, best first. */
function assetSources(modelId, name) {
  const subdir = HF_SUBDIR[modelId];
  const sources = [];
  if (subdir !== undefined) sources.push(`${HF_BASE}${subdir}${name}`);
  sources.push(assetUrl(modelId, name)); // same-origin proxy fallback
  return sources;
}

/** Fetch from the first source that answers; throws if none do. */
async function fetchFromAnySource(modelId, name) {
  let lastError;
  for (const url of assetSources(modelId, name)) {
    try {
      const response = await fetch(url);
      if (response.ok || response.status === 206) return response;
      lastError = new Error(`${name}: HTTP ${response.status}`);
      await response.body?.cancel();
    } catch (err) {
      // A cross-origin failure (CSP, offline, DNS) is not fatal on its own —
      // that is what the same-origin fallback is for.
      lastError = err;
    }
  }
  throw lastError ?? new Error(`${name}: no source answered`);
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

  const resp = await fetchFromAnySource(modelId, spec.name);
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

/**
 * The whole-asset streaming state: one SHA-256, one sink, one progress counter,
 * shared across every part so N parts behave as ONE logical byte stream.
 */
class WeightsPump {
  constructor(spec, sink, onProgress) {
    this.spec = spec;
    this.sink = sink;
    this.onProgress = onProgress;
    this.hash = new Sha256();
    this.received = 0;
    this.prefixChunks = [];
    this.prefixLen = 0;
    this.began = false;
  }

  beginIfReady(eof) {
    if (this.began || (!eof && this.prefixLen < Math.min(PREFIX_BYTES, this.spec.bytes))) return;
    const prefix = new Uint8Array(this.prefixLen);
    let off = 0;
    for (const p of this.prefixChunks) {
      prefix.set(p, off);
      off += p.length;
    }
    this.prefixChunks = [];
    this.sink.begin(prefix, this.spec.bytes);
    this.began = true;
  }

  /** Stream one part's reader. `partSpec` pins that part's bytes + sha256. */
  async drain(reader, partSpec, fromCache) {
    const partHash = new Sha256();
    let partReceived = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (partReceived + value.length > partSpec.bytes) {
        throw new Error(`${partSpec.name}: server sent more than the pinned ${partSpec.bytes} bytes`);
      }
      partHash.update(value);
      partReceived += value.length;
      this.hash.update(value);
      this.received += value.length;
      if (!this.began) {
        this.prefixChunks.push(value);
        this.prefixLen += value.length;
        this.beginIfReady(false);
      } else {
        this.sink.push(value);
      }
      this.onProgress?.(this.received, fromCache);
    }
    if (partReceived !== partSpec.bytes) {
      throw new Error(`${partSpec.name}: got ${partReceived} of ${partSpec.bytes} bytes`);
    }
    const hex = partHash.hex();
    if (hex !== partSpec.sha256) {
      throw new Error(`${partSpec.name}: SHA-256 mismatch (got ${hex.slice(0, 12)}…)`);
    }
  }

  finish() {
    this.beginIfReady(true); // short asset: the whole thing was the prefix
    if (this.received !== this.spec.bytes) {
      throw new Error(`${this.spec.name}: got ${this.received} of ${this.spec.bytes} bytes`);
    }
    const hex = this.hash.hex();
    if (hex !== this.spec.sha256) {
      throw new Error(`${this.spec.name}: whole-asset SHA-256 mismatch (got ${hex.slice(0, 12)}…)`);
    }
  }
}

/** Get one part's reader: cache hit, else fetch teed into the cache. */
async function partReader(cache, url, modelId, partName) {
  const cached = await cache.match(url);
  if (cached?.body) return { reader: cached.body.getReader(), fromCache: true, cachePut: null };
  const resp = await fetchFromAnySource(modelId, partName);
  if (!resp.body) throw new Error(`fetch ${partName}: no response body`);
  const [toSink, toCache] = resp.body.tee();
  const cachePut = cache
    .put(url, new Response(toCache, { headers: { "content-type": "application/octet-stream" } }))
    .catch((err) => ({ cacheError: err }));
  return { reader: toSink.getReader(), fromCache: false, cachePut };
}

/**
 * Big-asset lane: stream into `sink`, tee into the cache, verify by stream.
 * A `spec.parts` array (the GitHub 2 GiB asset-cap split) streams as one
 * logical byte stream: per-part pins verified as each part completes, the
 * whole-asset pin at the end. A spec without `parts` is a single-part asset.
 */
async function streamWeights(modelId, spec, sink, onProgress) {
  const cache = await caches.open(CACHE_NAME);
  const parts = spec.parts ?? [{ name: spec.name, bytes: spec.bytes, sha256: spec.sha256 }];
  const pump = new WeightsPump(spec, sink, onProgress);
  const puts = [];
  for (const part of parts) {
    const url = assetUrl(modelId, part.name);
    let src;
    try {
      src = await partReader(cache, url, modelId, part.name);
      await pump.drain(src.reader, part, src.fromCache);
    } catch (err) {
      if (src?.cachePut) await src.cachePut;
      await cache.delete(url).catch(() => {}); // never keep bytes that failed verification
      const flavor = src?.fromCache
        ? `cached ${part.name} failed verification (${err.message}); cache cleared — retry the load`
        : String(err.message ?? err);
      throw new Error(flavor);
    }
    if (src.cachePut) puts.push({ url, name: part.name, cachePut: src.cachePut });
  }
  pump.finish();
  for (const { url, name, cachePut } of puts) {
    const putResult = await cachePut;
    if (putResult?.cacheError) {
      // Cache write failed (quota). The verified stream still reached the sink —
      // the model runs; it just won't be cached for next time.
      await cache.delete(url).catch(() => {});
      console.warn(`cache.put(${name}) failed: ${putResult.cacheError} — model will re-download next visit`);
    }
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
