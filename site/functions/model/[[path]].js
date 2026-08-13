// Same-origin model proxy (Cloudflare Pages Function).
//
// This function forwards /model/<model>/<file> to a pinned upstream. It is a
// model mirror, not an open proxy: only the allow-listed files forward, and only
// the Range request header crosses.
//
// WHY THE PROXY STILL EXISTS, now that HuggingFace sends
// `access-control-allow-origin: *` and the browser could fetch it directly:
// the page's CSP is `connect-src 'self'`, and the site sells that as the
// enforceable form of "your document never leaves the tab". Fetching a model
// straight from huggingface.co would mean widening `connect-src` to an external
// origin, which retires exactly the guarantee a reader can verify from the CSP
// alone. Keeping the fetch same-origin costs a proxy hop and keeps the promise.
//
// UPSTREAM ORDER: HuggingFace first, GitHub second. GitHub release assets 503
// under load — which is what motivated the mirror — and cap a single asset at
// 2 GiB, which is why the 3.0 GB artifact is split into parts at all. Both hosts
// carry an identical file set, and the loader verifies each part's and the whole
// asset's SHA-256 against site/model-manifest.js regardless of which upstream
// answered, so failing over mid-download cannot yield a mixed or corrupt file.
const HF = "https://huggingface.co/Dicklesworthstone/franken_ocr-weights/resolve/main/";
const GH = "https://github.com/Dicklesworthstone/franken_ocr/releases/download/";

const RELEASES = {
  tromr: {
    bases: [`${HF}tromr/`, `${GH}models-tromr-v1/`],
    files: new Set([
      "tromr.int8.focrq",
      "tokenizer_rhythm.json",
      "tokenizer_pitch.json",
      "tokenizer_lift.json",
      "tokenizer_note.json",
    ]),
  },
  "unlimited-ocr": {
    // The wasm-only int4 artifact (v2, calibration-aware), shipped as byte-split
    // parts because of GitHub's 2 GiB asset cap. HuggingFace has no such cap, so
    // the split could eventually be retired — but only once GitHub is no longer
    // a fallback, since it cannot serve the unsplit file.
    bases: [HF, `${GH}models-unlimited-wasm-v1/`],
    files: new Set([
      "unlimited-ocr.wasm-int4.focrq.part1",
      "unlimited-ocr.wasm-int4.focrq.part2",
      "tokenizer.json",
    ]),
  },
  "got-ocr2": {
    bases: [HF, `${GH}models-got-ocr2-v1/`],
    files: new Set(["got-ocr2.int8.focrq", "qwen.tiktoken"]),
  },
  smolvlm2: {
    bases: [`${HF}smolvlm2/`, `${GH}models-smolvlm2-v1/`],
    files: new Set(["smolvlm2.int8.focrq", "tokenizer.json"]),
  },
};

export async function onRequest({ request, params }) {
  const parts = Array.isArray(params.path) ? params.path : [params.path];
  if (parts.length !== 2) return new Response("not found", { status: 404 });
  const [model, file] = parts;
  const release = RELEASES[model];
  if (!release || !release.files.has(file)) {
    return new Response("not found", { status: 404 });
  }

  // Try each upstream in order. Only the Range header crosses; nothing about the
  // requesting browser is forwarded.
  const headersOut = request.headers.has("range")
    ? { range: request.headers.get("range") }
    : {};
  let resp = null;
  let lastStatus = 502;
  for (const base of release.bases) {
    try {
      const candidate = await fetch(
        new Request(base + file, { method: "GET", headers: headersOut, redirect: "follow" }),
      );
      if (candidate.ok || candidate.status === 206) {
        resp = candidate;
        break;
      }
      lastStatus = candidate.status;
      // Drain so the failed upstream connection is not left dangling.
      await candidate.body?.cancel();
    } catch {
      lastStatus = 502;
    }
  }
  if (!resp) {
    return new Response(`upstream ${lastStatus}`, { status: 502 });
  }

  const headers = new Headers();
  for (const h of ["content-type", "content-length", "content-range", "accept-ranges", "etag"]) {
    const v = resp.headers.get(h);
    if (v) headers.set(h, v);
  }
  // Release assets are immutable by construction (tagged, hash-pinned).
  headers.set("cache-control", "public, max-age=31536000, immutable");
  return new Response(resp.body, { status: resp.status, headers });
}
