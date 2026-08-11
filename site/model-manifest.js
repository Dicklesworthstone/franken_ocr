// Pinned model files for the browser playground. Mirrors models/manifest-v2.json
// (the same bytes `focr pull tromr` verifies): sizes and full SHA-256 per asset.
// The wasm engine refuses a mismatched byte count and the loader refuses a
// mismatched digest, so a truncated or tampered download can never hydrate.
//
// Assets are fetched same-origin from /model/<model>/<file>, which the Pages
// Function (functions/model/[[path]].js) proxies to the GitHub release —
// release assets send no CORS headers, so the browser cannot fetch them
// cross-origin.
export const MODELS = {
  tromr: {
    label: "Polyphonic-TrOMR (sheet music → MusicXML)",
    license: "Polyphonic-TrOMR (NetEase) - Apache-2.0",
    weights: {
      name: "tromr.int8.focrq",
      bytes: 61107485,
      sha256: "cced11c0f05656dd54cc615a15939c472dc8f916f04ae154ea4a0364839f845a",
    },
    sidecars: [
      { name: "tokenizer_rhythm.json", bytes: 10743, sha256: "603bfef760e8424f7808acba423532b4beb2d88dbf085f81add6a8e543a34035" },
      { name: "tokenizer_pitch.json", bytes: 2682, sha256: "2382e8b20c1473290e200789604656b3a06bdf4b55a0818a0f7d175e8cb64ade" },
      { name: "tokenizer_lift.json", bytes: 979, sha256: "b61ba09cecd5bc343e6a038a2e26718b54cd3c08e8f9b72013ecf80c3cac86b2" },
      { name: "tokenizer_note.json", bytes: 830, sha256: "504d886d11e3c1fe92893abd46edfc68dfbe7a8eb83e6b51646532dad8a485e1" },
    ],
  },
  "unlimited-ocr": {
    label: "Baidu Unlimited-OCR (documents → Markdown)",
    license: "Baidu Unlimited-OCR - Copyright (c) 2026 Baidu, MIT License",
    // The wasm-only int4 artifact (recipe
    // unlimited-ocr-wasm-experts-int4-attn-int8-lmhead-int8-v1): MoE experts
    // int4 g16/g32, attention/lm_head/embed int8, vision tower bf16. The
    // native CLI keeps its own conservative 4.16 GB artifact — this one is
    // NEVER a native default; corpus-level CER certification is still in
    // progress (single-page smoke matches the bf16 reference to one line).
    desktopOnly: true,
    weights: {
      name: "unlimited-ocr.wasm-int4.focrq",
      bytes: 3003988117,
      sha256: "28ac2ca296dda3b32a245652f4240f93410caad18ec23d5a2d8727f58dafe724",
    },
    sidecars: [
      { name: "tokenizer.json", bytes: 9979544, sha256: "a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4" },
    ],
  },
};

export function totalBytes(model) {
  return model.weights.bytes + model.sidecars.reduce((n, s) => n + s.bytes, 0);
}
