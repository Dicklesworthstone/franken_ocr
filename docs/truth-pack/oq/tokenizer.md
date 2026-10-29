# OQ-16 — Tokenizer Parity (franken_ocr Phase -1 Truth Pack)

Model: `baidu/Unlimited-OCR` @ HF commit `3a7f4dbbbffcc6f9282712c5b0d7cc31b3812da5`
Source files read (all under `docs/truth-pack/snapshots/`):
- `tokenizer_config.json` (162 KB, 6662 lines)
- `special_tokens_map.json` (801 B)
- `processor_config.json`
- `config.json`
- `conversation.py`
- `modeling_unlimitedocr.py`
- `tokenizer.json` — **fetched** from HF (was not present locally), 9.5 MB
  - **SHA-256: `a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4`**

---

## QUESTION (verbatim)

> OQ-16 (tokenizer parity): identify the tokenizer class (LlamaTokenizerFast?), the model type (byte-level BPE?), the special tokens incl the DeepSeek glyph specials and `<image>`/`<|ref|>`/`<|det|>`, add_bos/eos behavior, and the pre-tokenizer/byte-fallback scheme. If tokenizer.json is needed for the pre-tokenizer regex/merges, fetch it, record its SHA-256, and read the "pre_tokenizer"/"model"/"added_tokens" sections. Specify exactly what a pure-Rust tokenizer must replicate.

---

## ANSWER (definitive)

### 1. Tokenizer class

`LlamaTokenizerFast` — i.e. the HF *fast* (Rust `tokenizers`) tokenizer, NOT the slow SentencePiece path. Despite the "Llama" name, the actual algorithm is configured entirely by `tokenizer.json` (a `tokenizers`-library serialization), and is **byte-level GPT‑2-style BPE**, not the Llama/SentencePiece scheme.

```
tokenizer_config.json:6658   "tokenizer_class": "LlamaTokenizerFast",
```

Supporting: `model_max_length` is the HF "very large int" sentinel (no enforced cap), `unk_token: null`, `legacy: true`, `clean_up_tokenization_spaces: false`.

```
tokenizer_config.json:6655   "legacy": true,
tokenizer_config.json:6656   "model_max_length": 1000000000000000019884624838656,
tokenizer_config.json:6659   "unk_token": null,
tokenizer_config.json:6652   "clean_up_tokenization_spaces": false,
```

### 2. Model type — byte-level BPE (GPT-2 style), NOT SentencePiece, NO algorithmic byte-fallback

From `tokenizer.json`:
```
.model = {"type":"BPE","dropout":null,"unk_token":null,
          "continuing_subword_prefix":null,"end_of_word_suffix":null,
          "fuse_unk":false,"byte_fallback":false,"ignore_merges":false}
.model.vocab  -> length 128000
.model.merges -> length 127741   (new HF format: array of [left,right] string pairs, e.g. ["Ġ","t"],["Ġ","a"],["i","n"],["Ġ","Ġ"],["h","e"])
```
(`tokenizer.json` top-level keys: `added_tokens, decoder, model, normalizer, padding, post_processor, pre_tokenizer, truncation, version`; `version: "1.0"`.)

Critical nuance for a Rust port: **`byte_fallback: false`** in the BPE model, but full byte coverage is still guaranteed because the **`ByteLevel`** pre-tokenizer maps every input byte into the 256-symbol printable byte-level alphabet (e.g. `Ġ`=id 223, `Ā`=id 191, `ĉ`=id 200 are all in the vocab). So there is no UNK and no SentencePiece-style `<0xNN>` byte tokens — byte coverage is via GPT‑2 byte→unicode remapping, exactly like GPT-2/RoBERTa. `unk_token: null` is therefore safe.

`normalizer` is a no-op:
```
.normalizer = {"type":"Sequence","normalizers":[]}
```

### 3. Pre-tokenizer / byte-fallback scheme (the load-bearing detail; only in tokenizer.json)

`.pre_tokenizer` is a `Sequence` of 4 stages (applied in order):

```
1. Split  Regex: \p{N}{1,3}                         behavior: Isolated   (split digits into groups of 1-3)
2. Split  Regex: [一-龥぀-ゟ゠-ヿ]+                  behavior: Isolated   (isolate CJK/Hiragana/Katakana runs)
3. Split  Regex (the GPT-style word regex, below)   behavior: Isolated
4. ByteLevel  add_prefix_space:false  trim_offsets:true  use_regex:false
```

Stage-3 regex (verbatim, must be replicated exactly):
```
[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

ByteLevel pre-tokenizer config: `{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}` — note `add_prefix_space:false` (no leading space inserted) and `use_regex:false` (the GPT-2 regex is **not** re-applied inside ByteLevel because the explicit Split stages above already did the splitting).

Decoder (for completeness, encode-direction is what matters for prompt building):
```
.decoder = {"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}
```

Padding/truncation are null (`"padding":null, "truncation":null`).

### 4. Post-processor: BOS template, no EOS

```
.post_processor = TemplateProcessing
  single: [ SpecialToken "<｜begin▁of▁sentence｜>" (id 0), Sequence A ]
  pair:   [ <｜BOS｜>, A, <｜BOS｜>, B ]
  special_tokens: { "<｜begin▁of▁sentence｜>": ids [0] }
```
So when `add_special_tokens=True`, the fast tokenizer prepends BOS (id 0) and adds **no** EOS — consistent with the config flags:
```
tokenizer_config.json:2   "add_bos_token": true,
tokenizer_config.json:3   "add_eos_token": false,
tokenizer_config.json:4   "add_prefix_space": null,
```

### 5. add_bos / add_eos behavior AT INFERENCE (what UnlimitedOCR actually does)

IMPORTANT: the OCR model does **NOT** rely on the post-processor. In the actual generation path it calls `tokenizer.encode(..., add_special_tokens=False)` and **hardcodes** BOS=0 / EOS=1 itself:

```
modeling_unlimitedocr.py:259  def text_encode(tokenizer, text: str, bos: bool = True, eos: bool = False):
modeling_unlimitedocr.py:260      t = tokenizer.encode(text, add_special_tokens=False)
modeling_unlimitedocr.py:261      bos_id = 0
modeling_unlimitedocr.py:262      eos_id = 1
modeling_unlimitedocr.py:263      if bos:
modeling_unlimitedocr.py:264          t = [bos_id] + t
modeling_unlimitedocr.py:265      if eos:
modeling_unlimitedocr.py:266          t = t + [eos_id]
```
And separately, after splicing image tokens, BOS is prepended exactly once at the very front of the full sequence:
```
modeling_unlimitedocr.py:966          """add the bos tokens"""
modeling_unlimitedocr.py:967          bos_id = 0
modeling_unlimitedocr.py:968          tokenized_str = [bos_id] + tokenized_str
```
So a Rust port's prompt builder: **prepend a single id 0 (BOS) at the front of the whole prompt; never auto-append EOS; encode all text segments with `add_special_tokens=False`.** BOS id = 0, EOS id = 1 (cross-checked in `config.json:26-27`/`94-95`).

`processor_config.json` confirms the processor itself does not add specials: `"add_special_token": false`.

### 6. Special tokens — IDs, normalized flag, and the OCR/grounding glyphs

Core specials (from `special_tokens_map.json` + `added_tokens_decoder`):
```
special_tokens_map.json:19  bos_token  "<｜begin▁of▁sentence｜>"   id 0
special_tokens_map.json:26  eos_token  "<｜end▁of▁sentence｜>"     id 1
special_tokens_map.json:33  pad_token  "<｜▁pad▁｜>"               id 2
```
(Note the BOS/EOS/PAD glyphs use the fullwidth bar `｜` U+FF5C and the bullet `▁` U+2581 — DeepSeek-V2 style. They are NOT ASCII `|`/`_`.)

`additional_special_tokens` (config + map): `<|User|>`, `<|Assistant|>` (these use ASCII `|`).
```
tokenizer_config.json:6647-6649  "additional_special_tokens": ["<|User|>","<|Assistant|>"]
special_tokens_map.json:2-17     additional_special_tokens: <|User|>, <|Assistant|>
```

OCR / vision / grounding / table specials — contiguous block at the top of the vocab, all `normalized:false, special:true` (cross-checked against `tokenizer.json .added_tokens`):
```
tokenizer_config.json:6550  128815  "<image>"        (also: processor_config "image_token":"<image>"; modeling_unlimitedocr.py:845 image_token_id = 128815)
tokenizer_config.json:6558  128816  "<|ref|>"
tokenizer_config.json:6566  128817  "<|/ref|>"
tokenizer_config.json:6574  128818  "<|det|>"
tokenizer_config.json:6582  128819  "<|/det|>"
tokenizer_config.json:6590  128820  "<|grounding|>"
tokenizer_config.json:6598  128821  "<td>"
tokenizer_config.json:6606  128822  "</td>"
tokenizer_config.json:6614  128823  "<tr>"
tokenizer_config.json:6622  128824  "</tr>"
tokenizer_config.json:6630  128825  "<|User|>"
tokenizer_config.json:6638  128826  "<|Assistant|>"
```

DeepSeek "glyph" specials present in the table (selected, `normalized:true, special:false` unless noted):
```
tokenizer_config.json:6430  128800  "<｜fim▁hole｜>"        (special:false)
tokenizer_config.json:6438  128801  "<｜fim▁begin｜>"
tokenizer_config.json:6446  128802  "<｜fim▁end｜>"
tokenizer_config.json:6454  128803  "<｜User｜>"            (fullwidth-bar variant, special:false — distinct from ASCII <|User|>)
tokenizer_config.json:6462  128804  "<｜Assistant｜>"       (fullwidth-bar variant, special:false)
tokenizer_config.json:6470  128805  "<|EOT|>"             (special:true)
tokenizer_config.json:6478  128806  "<｜tool▁calls▁begin｜>"
tokenizer_config.json:6486  128807  "<｜tool▁calls▁end｜>"
tokenizer_config.json:6494  128808  "<｜tool▁call▁begin｜>"
tokenizer_config.json:6502  128809  "<｜tool▁call▁end｜>"
tokenizer_config.json:6510  128810  "<｜tool▁outputs▁begin｜>"
... 128811 outputs end, 128812 output begin, 128813 output end, 128814 tool sep ...
```
IDs 128000–128799 are 800 reserved placeholder specials `<｜place▁holder▁no▁N｜>` (e.g. `tokenizer_config.json:30-37` id 128000 = `<｜place▁holder▁no▁0｜>`), all `normalized:false, special:true`.

`special_tokens_map.json` does NOT list the OCR glyphs (`<image>`, `<|ref|>`, etc.); they are only in `added_tokens` / `added_tokens_decoder`. A Rust port must seed its added-token table from `tokenizer.json .added_tokens` (or `tokenizer_config.json .added_tokens_decoder`), not from `special_tokens_map.json`.

### 7. Vocab size accounting (matters for embedding shape, not tokenization)

```
config.json:51 / :118   "vocab_size": 129280
tokenizer.json  .model.vocab length = 128000  (ids 0..127999)
tokenizer.json  .added_tokens: 830 entries; min id 0, max id 128826
  - 3 added tokens overlap the BPE region (ids 0,1,2 = BOS/EOS/PAD)
  - 827 added tokens span ids 128000..128826 contiguously (no gaps)
```
Covered ids: 0..127999 ∪ 128000..128826 = ids 0..128826. Declared `vocab_size` 129280 leaves ids **128827..129279 (453 slots) as padded/reserved embedding rows with no tokenizer entry** — never produced by tokenization, but the embedding matrix is sized 129280. Rust tokenizer max producible id = 128826; the LM head / embedding bead must size to 129280.

---

## What a pure-Rust tokenizer MUST replicate (encode direction)

1. **Algorithm**: byte-level (GPT-2) BPE. Merge ranks from `tokenizer.json .model.merges` (127741 ordered pairs, new array-of-pairs format). Vocab from `.model.vocab` (128000 entries). No dropout, `fuse_unk:false`, `ignore_merges:false`, no `unk` (don't emit UNK).
2. **Normalizer**: none (identity).
3. **Pre-tokenizer Sequence (exact order, exact regexes)**:
   - Split `\p{N}{1,3}` Isolated (digit grouping 1-3).
   - Split `[一-龥぀-ゟ゠-ヿ]+` Isolated (CJK/Kana isolation).
   - Split the GPT-style word regex (quoted verbatim in §3) Isolated.
   - ByteLevel `add_prefix_space:false`, `trim_offsets:true`, `use_regex:false` (GPT-2 byte→unicode remap of the 256 bytes; do NOT re-run the GPT-2 regex here).
   - Requires a Unicode-property-class regex engine (`\p{N}`, `\p{L}`, `\p{M}`, `\p{P}`, `\p{S}`) — e.g. the `fancy-regex`/`onig`-compatible classes the HF `tokenizers` crate uses. Plain `regex` crate Unicode classes are acceptable if behavior matches.
4. **Added/special tokens** (831 distinct contents): split them out of the text BEFORE BPE (HF `AddedVocabulary` behavior), with the exact ids in §6. All OCR/grounding glyphs are `normalized:false`. The fullwidth-bar DeepSeek glyphs (`<｜begin▁of▁sentence｜>`, `<｜User｜>`, etc.) must be matched as exact UTF-8 (U+FF5C `｜`, U+2581 `▁`) and kept distinct from the ASCII `<|User|>`/`<|Assistant|>` variants.
5. **BOS/EOS policy**: do NOT auto-append EOS. Encode all text with `add_special_tokens=False`, then prepend exactly one BOS id `0` at the front of the final prompt sequence (matching `modeling_unlimitedocr.py:259-268` and `:966-968`). EOS id = 1, PAD id = 2.
6. **Image-token splice**: `<image>` (id 128815) is split out, and the model expands it into a computed run of id-128815 tokens around the vision features (see `modeling_unlimitedocr.py:844-962`). The tokenizer itself only needs to (a) keep `<image>` as a single special token id 128815 and (b) let the prompt builder split on the literal string `"<image>"` — the count/expansion logic is the vision-prefix bead, not the tokenizer.

---

## UNBLOCKS

- **Tokenizer bead / kernel**: full spec to implement a `tokenizers`-parity pure-Rust BPE encoder (merges, vocab, 4-stage pre-tokenizer, added-token handling). Recommended: load `tokenizer.json` directly via the `tokenizers` crate, OR hand-implement matching the §"must replicate" list. Golden-vector tests should pin SHA-256 `a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4`.
- **Prompt-builder / conversation bead**: BOS=0 prepend, no EOS, `add_special_tokens=False` text encode, `<image>` id 128815, role tokens `<|User|>`/`<|Assistant|>` (ids 128825/128826), sep `"\n\n"`, sep2 `"<｜end▁of▁sentence｜>"` (id 1) from `conversation.py:198-205` (deepseek template).
- **Embedding / LM-head shape bead**: vocab_size 129280, but max tokenizer-producible id is 128826 (453 padded rows 128827..129279).
- **Vision-prefix / image-token-expansion bead**: image_token_id 128815 expansion logic in `modeling_unlimitedocr.py:844-962` (out of OQ-16 scope; flagged for the vision bead).

## Unlimited-OCR-2 (U2-A4, OQ-V2-8)

Model: `baidu/Unlimited-OCR-2` (pins in `docs/truth-pack/PINNED_SOURCES.md`).
`tokenizer.json` is 9,980,278 B, **SHA-256
`c55c165e32c5a67830362a53e1f548c8930dcc8469efa30ff3252a982799c589`**
(`model_arch::UNLIMITED_OCR_V2_TOKENIZER_SHA256`).

- **Delta from v1: four added tokens only.** `model` (vocab and merges),
  `pre_tokenizer`, `normalizer`, `post_processor` and `decoder` are identical to
  v1's, and so are v1's 830 added-token entries. The new entries are all
  `special:false`, `normalized:true`, `lstrip/rstrip/single_word:false`:

  | id | content | used by inference |
  |----|---------|-------------------|
  | 128827 | `<PAGE>` | yes: the prompt is `"<PAGE><image>" * n + "Multi page parsing."`, and the output separates pages with it |
  | 128828 | `<BLOCK>` | no |
  | 128829 | `<FORK>` | no |
  | 128830 | `<CHILD>` | no |

- **Id range.** 834 added tokens; the max producible id is **128830** (v1:
  128826). `config.json vocab_size` 129280 leaves ids 128831..129279 (449 slots)
  as rows no tokenization produces.
- **Non-special means two things.** The four are split out of text like any
  added token (HF `AddedVocabulary` matches all added tokens, not only special
  ones), and `skip_special_tokens=True` keeps them. The reference decodes model
  output with `skip_special_tokens=False` (`modeling_unlimitedocr.py:1201`) and
  splits pages on the `<PAGE>` text.
- **Required ids** (`model_arch::UNLIMITED_OCR_V2_TOKENS`, the `unlimited-ocr-2`
  arch's `required_token_ids()` since U2-A5): BOS 0, EOS 1, `<image>` 128815,
  `<PAGE>` 128827. The v1 tokenizer has no `<PAGE>`, so a v1 sidecar fails this
  check. The arch also pins the sidecar's SHA-256; a refusal names the pinned
  Hugging Face file, because `focr pull` does not publish v2 yet.

Conformance evidence (2026-10-29):
- Golden `tests/fixtures/tokenizer_baidu/expected_uocr2.json`: 227 cases (v1's
  163-case corpus plus `tests/fixtures/tokenizer/corpus_uocr2.txt`: the 1-, 3- and
  20-page prompts, each new token, `<PAGE>` beside letters, digits, CJK,
  whitespace, newlines and emoji, near misses, page-joined markdown, tables,
  LaTeX and grounding output). Generated by `FOCR_BAIDU_MODEL=unlimited-ocr-2
  scripts/gen_token_id_fixtures.py` with `tokenizers==0.21.4` (the v2 reference
  stack's pin) on Python 3.10.19. All 227 cases round-trip exactly in the
  reference, which also records each case's skip-special decode.
- `tokenizer::tests::uocr2_token_id_conformance_gate`: encode, decode and
  skip-special decode are exact on all 227 cases (0 mismatches). The anchors,
  the required ids and the id ceiling are checked too.
- `uocr2_added_tokens_stay_whole_under_fuzz` (always on, 2,000 strings) and
  `uocr2_real_tokenizer_keeps_added_tokens_whole_under_fuzz` (real v2
  tokenizer, 1,000 strings): over random interleavings of the added tokens,
  near misses and text, decode(encode(s)) == s, every token occurrence encodes
  to exactly its one id, and splitting is lossless and idempotent.
- The same generator, run with `tokenizers==0.21.4` over v1's tokenizer,
  reproduces the committed v1 golden (made with 0.23.1) exactly, so the two
  versions agree on this corpus.

## BLOCKERS

None for OQ-16. `tokenizer.json` was the only missing input and has been fetched, hashed, and read (sections: `model`, `pre_tokenizer`, `normalizer`, `post_processor`, `decoder`, `added_tokens`). No `chat_template` key exists in `tokenizer_config.json` (`grep -c chat_template` = 0) — the chat/prompt format lives in `conversation.py`, not a Jinja template; that is fully captured above and is a separate (already-answerable) prompt-builder concern, not a tokenizer gap.
