// Same-origin model proxy (Cloudflare Pages Function).
//
// GitHub release assets send no CORS headers, so the browser cannot fetch
// them cross-origin; this function forwards /model/<model>/<file> to the
// pinned release tag. It is a model mirror, not an open proxy: only the
// allow-listed files forward, and only the Range request header crosses.
const RELEASES = {
  tromr: {
    base: "https://github.com/Dicklesworthstone/franken_ocr/releases/download/models-tromr-v1/",
    files: new Set([
      "tromr.int8.focrq",
      "tokenizer_rhythm.json",
      "tokenizer_pitch.json",
      "tokenizer_lift.json",
      "tokenizer_note.json",
    ]),
  },
  // The Unlimited-OCR int4 browser artifact registers here when it ships.
};

export async function onRequest({ request, params }) {
  const parts = Array.isArray(params.path) ? params.path : [params.path];
  if (parts.length !== 2) return new Response("not found", { status: 404 });
  const [model, file] = parts;
  const release = RELEASES[model];
  if (!release || !release.files.has(file)) {
    return new Response("not found", { status: 404 });
  }

  const upstream = new Request(release.base + file, {
    method: "GET",
    headers: request.headers.has("range")
      ? { range: request.headers.get("range") }
      : {},
    redirect: "follow",
  });
  const resp = await fetch(upstream);
  if (!resp.ok && resp.status !== 206) {
    return new Response(`upstream ${resp.status}`, { status: 502 });
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
