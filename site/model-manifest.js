// Pinned model files for the browser playground. Mirrors models/manifest-v2.json
// (the same bytes `focr pull tromr` verifies): sizes and full SHA-256 per asset.
// The wasm engine refuses a mismatched byte count and the loader refuses a
// mismatched digest, so a truncated or tampered download can never hydrate.
//
// Assets are fetched DIRECTLY from the Hugging Face weight repo, which sends
// CORS headers and serves ranged requests — so a multi-gigabyte download never
// crosses this site's own bandwidth. The same-origin Pages Function
// (functions/model/[[path]].js) remains as a fallback and forwards to Hugging
// Face first, then the GitHub release; GitHub release assets send no CORS
// headers at all, which is why they can only ever be reached through that proxy.
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
    // unlimited-ocr-wasm-experts-int4-attn-int8-lmhead-int8-v1), v2:
    // calibration-aware quantization (importance-weighted clip search + AWQ
    // down_proj fold over a 13-page activation-statistics run) — measured
    // strictly better than the plain-RTN v1 on every corpus page. MoE experts
    // int4 g16/g32, attention/lm_head/embed int8, vision tower bf16. The
    // native CLI keeps its own conservative 4.16 GB artifact — this one is
    // NEVER a native default. Corpus receipts live in docs/DISCREPANCIES.md.
    desktopOnly: true,
    // The README-documented repetition-guard mitigation (FOCR_NO_REPEAT_NGRAM
    // analog): hard dense scans can tip the wasm decode into a repeat loop
    // (measured); the tighter guard is applied at load and labeled honestly —
    // native users can set the identical knob.
    decodeGuard: 20,
    weights: {
      name: "unlimited-ocr.wasm-int4.focrq",
      bytes: 3003988117,
      sha256: "2653831ccd7f481f898f80ae5c95fa1ec7ee2a5a18005d3c927ddf64ed75e187",
      // GitHub caps release assets at 2 GiB, so the artifact ships as ordered
      // byte-split parts; the loader streams them as ONE logical byte stream
      // and verifies each part AND the whole against these pins.
      parts: [
        { name: "unlimited-ocr.wasm-int4.focrq.part1", bytes: 1677721600, sha256: "95e8bc996ef08dc9ff179dba522ee45e953823913dbf73ac710d799627a9b2c5" },
        { name: "unlimited-ocr.wasm-int4.focrq.part2", bytes: 1326266517, sha256: "1b6673345d1223f6ad4443df3f9c0760b4e401549c731c1c0d0c9e392dffda93" },
      ],
    },
    sidecars: [
      { name: "tokenizer.json", bytes: 9979544, sha256: "a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4" },
    ],
  },
  "got-ocr2": {
    label: "GOT-OCR2 (structured: formulas, tables, molecules)",
    license: "GOT-OCR2.0 - Copyright (c) 2024 Ucas-HaoranWei, Apache-2.0",
    // Measured in Node (same V8/wasm engine as Chrome): the 776 MiB artifact
    // stages into ONE segment, hydrates in <1 s, and recognizes
    // tests/fixtures/got/sample_text.png in 140-224 s. Output is byte-identical
    // to the native aarch64 CLI on the plain page AND on both format-mode
    // fixtures (formula.png, table.png) — GOT is the exact-parity lane.
    //
    // MEMORY (the number that decides the tab): wasm linear memory peaks at
    // 3456 MB during recognize — the f32 SAM-ViT-B tower hydrates whole, since
    // the streamed-vision residency mode is keyed to the unlimited wasm recipe
    // only. That is ~640 MB under the wasm32 4 GiB ceiling, so this lane is
    // desktop-Chrome-class: process RSS (723 MB) understates it by 5x because
    // macOS does not resident-back the untouched pages.
    desktopOnly: true,
    weights: {
      name: "got-ocr2.int8.focrq",
      bytes: 813877416,
      sha256: "4da43d7944d7ad6fcab85f1660ceb1a0f0cf7959d6cef0910974ec43aa0d532f",
    },
    sidecars: [
      { name: "qwen.tiktoken", bytes: 2561218, sha256: "b2b1b8dfb5cc5f024bafc373121c6aba3f66f9a5a0269e243470a1de16a33186" },
    ],
  },
  smolvlm2: {
    label: "SmolVLM2 (photo description / VQA)",
    license: "SmolVLM2-500M-Video-Instruct (HuggingFaceTB) - Apache-2.0",
    // Measured in Node: the 1.01 GiB artifact plans to TWO segments (the ≤1 GiB
    // wasm32 segmentation, bd-syf2), hydrates in <1 s, and captions
    // tests/fixtures/smolvlm2/sample_photo.png in 242-309 s. wasm linear memory
    // peaks at 2811 MB (the f32 SigLIP tower hydrates whole), so this lane is
    // desktop-Chrome-class like GOT; peak process RSS was only 704 MB, which is
    // NOT the browser-relevant figure.
    //
    // PARITY NOTE (honest, measured): the ~300-token free-form caption diverges
    // from the native aarch64 CLI's wording after the first sentence. It is not
    // the int8 kernel tier — forcing native to scalar (FOCR_FORCE_ARCH=scalar)
    // reproduces native's wording, not wasm's — so it is f32 associativity
    // drift (SigLIP tower) amplified by a low-margin free-form decode. Both
    // captions describe the same photo correctly.
    //
    // The sharper probe says the wasm lane is sound: on the three short factual
    // VQA cases of tests/fixtures/smolvlm2/vqa_fixtures.json, wasm reproduced
    // the committed PyTorch oracle answer EXACTLY on all three, while the
    // native CLI matched the oracle on only one of the three. Short constrained
    // decodes agree; long free-form ones drift. GOT, whose decode is short, is
    // byte-identical native↔wasm on all three of its fixtures.
    desktopOnly: true,
    weights: {
      name: "smolvlm2.int8.focrq",
      bytes: 1087397293,
      sha256: "4ad2ac89e47c83ad4fa3d7389ae753cbbfd190e8214707422abfaeb6439d06fc",
    },
    sidecars: [
      { name: "tokenizer.json", bytes: 3548256, sha256: "5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a" },
    ],
  },
};

export function totalBytes(model) {
  return model.weights.bytes + model.sidecars.reduce((n, s) => n + s.bytes, 0);
}
