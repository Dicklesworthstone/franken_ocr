// Local server for the real-browser harness: serves site/ and implements the
// /model/<model>/<file> route from a local directory (FOCR_MODEL_DIR), so the
// file that ships (_headers included, parsed not restated) is the file under
// test. No mocks — a stubbed fetch/Worker/OPFS harness on the sibling project
// passed 8/8 while the deployed site hung on every browser.
import { createServer } from "node:http";
import { readFileSync, existsSync, statSync, createReadStream } from "node:fs";
import { join, extname, resolve } from "node:path";

const SITE = resolve(new URL("..", import.meta.url).pathname);
const MODEL_DIR = process.env.FOCR_MODEL_DIR ?? "";
const PORT = Number(process.env.PORT ?? 8791);

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".wasm": "application/wasm",
  ".png": "image/png",
  ".json": "application/json",
  ".focrq": "application/octet-stream",
  ".tiktoken": "text/plain; charset=utf-8",
};

// Parse _headers (the shipped file) into [pattern, headers] rules.
const headerRules = [];
{
  const text = readFileSync(join(SITE, "_headers"), "utf8");
  let current = null;
  for (const raw of text.split("\n")) {
    const line = raw.replace(/#.*$/, "").trimEnd();
    if (!line.trim()) continue;
    if (!raw.startsWith(" ") && !raw.startsWith("\t")) {
      current = { pattern: line.trim(), headers: {} };
      headerRules.push(current);
    } else if (current) {
      const idx = line.indexOf(":");
      current.headers[line.slice(0, idx).trim()] = line.slice(idx + 1).trim();
    }
  }
}
function headersFor(path) {
  const out = {};
  for (const { pattern, headers } of headerRules) {
    const re = new RegExp(`^${pattern.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*")}$`);
    if (re.test(path)) Object.assign(out, headers);
  }
  return out;
}

createServer((req, res) => {
  const url = new URL(req.url, "http://localhost");
  let path = url.pathname;

  // /model/<model>/<file> → local model dir
  const m = path.match(/^\/model\/([\w-]+)\/([\w.-]+)$/);
  if (m) {
    if (!MODEL_DIR) {
      res.writeHead(503).end("FOCR_MODEL_DIR not set");
      return;
    }
    const file = join(MODEL_DIR, m[1], m[2]);
    if (!existsSync(file)) {
      res.writeHead(404).end("no such model file");
      return;
    }
    const size = statSync(file).size;
    res.writeHead(200, {
      "content-type": "application/octet-stream",
      "content-length": size,
      ...headersFor(path),
    });
    createReadStream(file).pipe(res);
    return;
  }

  if (path === "/") path = "/index.html";
  const file = join(SITE, path);
  if (!file.startsWith(SITE) || !existsSync(file) || statSync(file).isDirectory()) {
    res.writeHead(404).end("not found");
    return;
  }
  const body = readFileSync(file);
  // The harness serves unstamped files: substitute the placeholder so the
  // ?v= URLs still resolve.
  const isText = /\.(html|js|css)$/.test(file);
  const payload = isText ? body.toString().replaceAll("@SITEV@", "dev") : body;
  res.writeHead(200, {
    "content-type": TYPES[extname(file)] ?? "application/octet-stream",
    ...headersFor(path),
  });
  res.end(payload);
}).listen(PORT, () => console.log(`serving ${SITE} on http://localhost:${PORT} (models: ${MODEL_DIR || "UNSET"})`));
