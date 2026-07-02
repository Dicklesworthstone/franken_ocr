# OneChart — architecture spec (bead bd-3jo6.4.1 / D1)

Implementation-ready census of **OneChart** (`kppkkp/OneChart`; "OneChart: Purify the
Chart Structural Extraction via One Auxiliary Token", arXiv
[2404.09987](https://arxiv.org/abs/2404.09987), ACM MM 2024 **Oral**; **Apache-2.0**)
for a from-scratch pure-Rust CPU port in franken_ocr (epic `bd-3jo6`, sub-epic D).
Every load-bearing number is cited to a source file; this doc is self-contained so the
implementer never needs to reverse-engineer the model. **Source of truth = the released
weights repo files** (`config.json`, `modeling_OneChart.py`, `sam_vision_b.py`,
`vocab.json`, `merges.txt`, `added_tokens.json`, `tokenizer_config.json`) at HF revision
`79212de2f520694e534e58240f228eff351d536c` + the GitHub repo `LingyvKong/OneChart`
(training + eval code) + the paper. Chart→structured-JSON extraction with a numeric
**self-verify head** — the precise-digitization upgrade over GOT's chart mode.

> **Headline.** Encoder = **SAM-ViT-B** + the Vary 2-conv compressor → **256 × 1024**
> (byte-for-byte the GOT-OCR2 vision graph, different weights + different preprocessing).
> Connector = **`Linear(1024→768)`**. Decoder = **OPT-125M architecture**
> (`OneChartOPTForCausalLM` extends HF `OPTForCausalLM`: 12 layers, hidden 768, 12 heads,
> FFN 3072 **ReLU non-gated**, **LayerNorm with bias — NOT RMSNorm**, **learned absolute
> positions with offset 2 — NO RoPE**, tied embeddings, vocab 50269) — **NOT Qwen-family**
> (this was the open VERIFY; it kills naive `decoder_qwen2` reuse, §12). Plus the novel
> bit: **`num_decoder`**, a 768→384→384→256 ReLU MLP off the `<Number>` token's final
> hidden state that regresses the chart's numeric values for a self-consistency check.
> 262,314,496 params bf16; `model.safetensors` 524,672,832 B single shard (verified:
> header max-offset + header == server `x-linked-size`); ~223.7 M unique after tie de-dup.

---

## 1. Top-level graph (Vary-tiny splice; direct GOT-OCR2 sibling)

```
image (1024×1024×3, RGB, [0,1] — NO mean/std normalize)
  → Vision encoder  (SAM-ViT-B backbone + neck + net_2/net_3 compressor)  → (B,1024,16,16)
  → flatten(2).permute(0,2,1)                                             → 256 × 1024
  → Connector       (mm_projector = Linear(1024→768, bias=True))          → 256 × 768
  → splice into decoder input embeddings over the 256 <imgpad> slots,
    bracketed by <img> … </img>
  → Decoder         (OPT-125M arch, 12L, hidden 768)                      → text tokens
                                                                            (a python-dict/JSON string)
  → Number head     (num_decoder MLP on the <Number> hidden state)        → 256 f32 values
                                                                            (self-verify vector)
```

Param split (computed from the safetensors header, §13): vision 95.57 M, connector
0.79 M, decoder 126.81 M (85.05 M layers + 38.61 M embed counted once + 3.15 M positions),
num head 0.54 M — **223.71 M unique** ("0.2B" headline). CLIP is **not** used. Single image, **no multi-crop**
(`modeling_OneChart.py:200` raises `NotImplementedError` for P>1 tile batches).

## 2. Vision encoder — `sam_vision_b.py::build_SAM_vit_b` (= GOT's `got_vision_b.py`)

**Geometry-identical to GOT-OCR2 spec §2** (same SAM-ViT-B + Vary compressor source
lineage). Confirmed from `sam_vision_b.py` + the safetensors header:

| field | value | same as GOT? |
|---|---|---|
| input | 1024×1024×3 RGB | yes |
| patch_size / embed_dim / depth / heads | 16 / 768 / 12 / 12 (head_dim 64) | yes |
| mlp_ratio / act | 4.0 (3072), GELU | yes |
| qkv_bias / use_rel_pos | True / True (decomposed rel-pos) | yes |
| global_attn_indexes / window_size | [2, 5, 8, 11] / 14 | yes |
| rel_pos shapes | windowed [27,64], global [127,64] | yes |
| pos embed | learned `pos_embed (1,64,64,768)` + rel-pos in attn | yes |
| LayerNorm eps | **1e-6 explicit** (`partial(nn.LayerNorm, eps=1e-6)`, `sam_vision_b.py:456`) | resolves GOT OQ-5 for this lineage |
| neck | Conv1×1(768→256,no-bias) · LN2d · Conv3×3(256→256,no-bias) · LN2d | yes |
| compressor | `net_2` Conv3×3s2(256→512) · `net_3` Conv3×3s2(512→1024) → (B,1024,16,16) | yes |

Delta vs GOT: the tower's `forward` returns the **(B,1024,16,16) conv map** (no flatten);
`OneChartModel.forward` does `flatten(2).permute(0,2,1)` → 256×1024
(`modeling_OneChart.py:196`). Tensor prefix is **`model.vision_tower.*`** (GOT:
`model.vision_tower_high.*`, Baidu: `model.sam_model.*`) — leaf names byte-identical (§13).

## 3. Connector — `mm_projector = nn.Linear(1024, 768)` (bias=True, no act, no norm; `modeling_OneChart.py:160`). Output currency **768** (OPT hidden), not 1024/1280.

**Splice** (`modeling_OneChart.py:210-231`): find `<img>` (50266) position, then
`torch.cat(embeds[:pos+1], image_features, embeds[pos+257:])` — i.e. the 256 `<imgpad>`
embedding rows are **replaced** by the projected features. Net-identical to GOT's
`<imgpad>`-row overwrite → the existing `connector::masked_scatter` pattern serves, keyed
on `<imgpad>`=50265 (validated upstream by the `</img>`-follows check at `:215`).

## 4. Language decoder — literal `config.json` (`OneChartOPTForCausalLM` = **OPT arch**)

**VERIFIED: OPT-125M architecture, NOT Qwen** (`architectures:["OneChartOPTForCausalLM"]`
extends HF `OPTForCausalLM`/`OPTModel`; `model_type:"OneChart"` over `OPTConfig`).

| field | value | | field | value |
|---|---|---|---|---|
| hidden_size | **768** | | activation_function | **relu** (non-gated fc1/fc2) |
| num_hidden_layers | **12** | | do_layer_norm_before | **true** (pre-LN) |
| num_attention_heads | **12** (MHA, head_dim 64) | | _remove_final_layer_norm | false (final LN present) |
| ffn_dim | **3072** | | enable_bias | **true** (ALL attn+MLP linears biased) |
| vocab_size | **50269** | | layer_norm_elementwise_affine | true |
| max_position_embeddings | **4096** (embed rows **4098**, offset 2) | | word_embed_proj_dim | 768 (**no** project_in/out) |
| bos/eos | **2** `</s>` | | pad | **1** `<pad>` |
| torch_dtype | bfloat16 | | dropout | 0.1 (eval ⇒ 0) |
| im_start/end/patch | **50266 / 50267 / 50265** | | number_token | **50268** `<Number>` |

Per-layer graph (HF `modeling_opt.py`, `do_layer_norm_before=true`):
`h += out_proj(attn(q,k,v))` where `q,k,v = {q,k,v}_proj(self_attn_layer_norm(h))` and
**q is pre-scaled ×(1/8)** at projection (HF OPT applies `scaling=head_dim**-0.5` to q,
then softmax(QKᵀ) with no further scale); then `h += fc2(relu(fc1(final_layer_norm(h))))`
(the per-layer pre-MLP norm is *named* `final_layer_norm` — naming hazard vs the model-level
`model.decoder.final_layer_norm`). Full-causal MHA, growing KV, **no RoPE**: positions are
**learned absolute** — `embed_positions` is `[4098,768]`, position `i` reads row `i+2`
(HF `OPTLearnedPositionalEmbedding` offset=2; ids from attention-mask cumsum−1, = 0..L−1
for our unpadded single sequence), added to `embed_tokens(ids)` before layer 0. Model-level
`final_layer_norm` after layer 11; `lm_head` bias-free and **tied** — `lm_head.weight` and
`model.decoder.embed_tokens.weight` are both stored `[50269,768]` BF16 and are
**byte-identical** (SHA-256 `1c7b2843e2…da89d0` over both tensors' bytes, range-fetch
verified 2026-07-01) → the `.focrq` stores ONE copy (GOT OQ-1 precedent).

**Hard cap:** position table = 4096 usable positions. Upstream `max_new_tokens=4096` +
prompt 309 = 4405 > 4096 would index past the table — upstream only survives because chart
JSON is short. The port MUST stop at total-seq 4096 (kill-switch-style hard stop, §11 OQ-D7).

## 5. Special tokens + prompt template (ONE fixed prompt; no modes)

Base OPT/GPT-2 vocab `vocab.json` = **50265** entries (`<s>`=0, `<pad>`=1, `</s>`=2,
`<unk>`=3, …GPT-2 byte-BPE…, `<|endoftext|>`=50260, `madeupword0000-0002`=50261-50263,
`<mask>`=50264) + `merges.txt` (50000 merges) + 4 added tokens (`added_tokens.json`):
**`<imgpad>`=50265, `<img>`=50266, `</img>`=50267, `<Number>`=50268** → vocab 50269, fully
packed. bos=eos=unk=`</s>`(2), pad=`<pad>`(1). `<Number>` never appears in the prompt — the
model *generates* it as the first answer token (§8).

**Conversation** (`conv_vicuna_v1_1`, `SeparatorStyle.TWO`, sep=`" "`, sep2=`"</s>"`;
`modeling_OneChart.py:121-131,444-449`); instruction is the single hardcoded
`query = 'Convert the key information of the chart to a python dict:'`:

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <img>{<imgpad>×256}</img>Convert the key information of the chart to a python dict:\n ASSISTANT:
```

(No space between `</img>` and `Convert`; the message-terminating sep `" "` lands between
the trailing `\n` and `ASSISTANT:`.) Tokenized with bos (`add_bos_token=true`):
**309 ids = `[2]` + 32 system/`USER: ` ids + `[50266]` + 256×`[50265]` + `[50267]` + 18
query/`ASSISTANT:` ids.** Pinned exactly (HF `GPT2Tokenizer` on the repo files) in
`tests/fixtures/tokenizer_onechart/expected.json` `prompt_fixture` — head
`[2,250,7359,227,10,10691,3018,8,41,7350,2316,3167,4,20,3167,2029,7163,6,4271,6,8,24908,5274,7,5,3018,18,1142,4,382,2076,35,1437,50266,…]`,
tail `[…,50267,9157,9942,5,762,335,9,5,5966,7,10,39825,28700,35,50118,20860,11595,11088,35]`.

## 6. Preprocessing — `OneChartImageEvalProcessor(image_size=1024)` (`modeling_OneChart.py:133-148`)

`.convert('RGB')` → bicubic `Resize((1024,1024))` (**squash, NO aspect preserve**) →
`ToTensor` [0,1] CHW → `Normalize(mean=(0,0,0), std=(1,1,1))` — **a NO-OP: pixels stay
[0,1]. The CLIP constants are NOT used** (unlike GOT §6; GitHub demo `test_transform`
confirms `(0,0,0)/(1,1,1)`). **No multi-crop, no thumbnail — exactly one 1024² tile.**
Same CatmullRom≈PIL-bicubic sub-L0 divergence class as GOT L0b.

## 7. Tokenizer — OPT GPT-2 byte-level BPE (**NOT tiktoken, NOT Qwen** — corrects D9's title)

`tokenizer_class: GPT2Tokenizer` (slow) over `vocab.json`+`merges.txt`+`added_tokens.json`;
`add_bos_token=true` (encode = `[2] + BPE(text)`), `add_prefix_space=false`, byte-level
with the classic GPT-2 regex `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.
**The existing `src/tokenizer/mod.rs` engine is the right base** (it IS GPT-2-style
byte-level BPE with added-token splitting) — deltas: (1) a **Gpt2 pretok mode** in
`pretok.rs` (single classic regex; today's pipeline is the DeepSeek 4-stage with digit-triple
+ CJK pre-splits, which is NOT id-equivalent), (2) ingest `vocab.json`+`merges.txt`
(synthesize a `tokenizer.json` at convert time (D2) OR add a small loader — the model
struct is identical), (3) the bos-prepend policy. 29 token-id-exact golden cases incl. the
full 309-id prompt: `tests/fixtures/tokenizer_onechart/expected.json` (file SHA-256s of
vocab/merges/added_tokens pinned in `_meta`).

## 8. Number head + self-verify — the novel bit (`modeling_OneChart.py:247-311,491-508`)

**Head:** `num_decoder = Linear(768→384) · ReLU · Linear(384→384) · ReLU · Linear(384→256)`
(all biased; 0.54 M params; stored top-level `num_decoder.{0,2,4}.{weight,bias}`).

**When it fires (inference):** every forward whose `input_ids` contain 50268 computes
`pred_locs = num_decoder(hidden[:, first_50268_pos, :])`. The prompt has no `<Number>`, so
during cached greedy decode this happens exactly at **the decode step whose INPUT token is
the previously-generated `<Number>`** (the model emits `<Number>` as answer token #1; fed
back at the next step, its **post-`final_layer_norm` hidden state** — the same stream that
feeds `lm_head` — goes through the MLP). Upstream keeps `pred_locs[0][:100]` (first 100 of
256) as `self.pred_locs`. Port contract: tap the post-final-norm decode hidden when
`input_id==50268`, run the MLP once, return `Option<Vec<f32>>` **per request** (upstream
keeps a stale model attribute if `<Number>` is never emitted — do NOT replicate).

**What the 256 outputs mean (training semantics, `conversation_dataset_v1_with_number.py:68-85`
+ `vary_opt_math.py:220-225`):** the GT `"Numbers"` list (the chart's values in
`values`-dict order) is padded/clipped to 256 with NaN, **min-max normalized**
`(x−min)/(max−min)` and trained with **masked mean-L1** (NaN slots excluded). So
`pred_locs[i]` ≈ the i-th numeric value of the chart normalized to [0,1].

**Self-verify (`reliable_check=True`, `chat()`):**
1. Parse the generated text (after stripping `<Number>`/`</s>`) as JSON; walk
   `out["values"]` recursively (dicts recurse — multi-series; a **list** anywhere aborts →
   unreliable); numeric leaves pass through, string leaves are normalized by
   `re.sub(r'\(\d+\)|\[\d+\]', '', v)` then `re.sub(r'[^\d.-]', '', v)` → float (skip
   `-`,`*`,`none`,`None`,``) — so `"6.12%"`→6.12, `"1,234"`→1234.
2. Min-max normalize the list: `(x−min)/(max−min+1e-9)`; len<2 ⇒ identity;
   `round(x, 4)` (python banker's rounding).
3. `reliable_distance = mean-L1(pred_locs[:n], normalized_gt)`; **< 0.1 ⇒ "reliable"**.
   Upstream appends to the response: the `<Chart>: [...]` vector, the distance, and the
   verdict line. Port: emit as structured fields (D5), not string concat.

## 9. Output schema + postprocessing

Generated text = `<Number>` + a JSON object + `</s>`. Strip `<Number>` (`chat()` uses
`replace("<Number>","")`) and the `</s>` stop string; no other normalization table
(contrast GOT §8). Schema (README training format + `eval_ChartSE.py` consumers):

```json
{"title": str|"None", "source": str|"None", "x_title": str|"None", "y_title": str|"None",
 "values": {label: "6.12%"|number, …}            // flat single-series, or
 "values": {series: {label: value, …}, …}}       // nested dict multi-series
```

Values are frequently **strings with units** (`"6.12%"`); the eval also tolerates a
`"data"` key alias and brace-completes truncated JSON (`complete_json_string`). Port the
brace-completion into the D5 renderer for robustness parity.

## 10. Generation params — `chat()` (`modeling_OneChart.py:461-483`)

`do_sample=False, num_beams=1, max_new_tokens=4096`; **`no_repeat_ngram_size` is
commented out upstream** (contrast GOT's global 20) ⇒ decode contract
`{temperature: 0.0, eos_token_id: 2, no_repeat_ngram_size: 0}`. Stop = eos 2 (HF default)
+ a `'</s>'` keyword/stop-string criterion. Greedy/deterministic. Repetition risk: the
bd-ff4i guard, if ever applied here, is a **non-parity** kill-switched mode. Hard cap
total-seq ≤ 4096 (§4).

## 11. Open questions (doctrine hard rule — no kernel ships against an unresolved OQ)

- **OQ-D1** Pretok id-equivalence: the classic GPT-2 regex vs `pretok.rs`'s DeepSeek
  4-stage pipeline (digit-triples + CJK pre-splits WILL diverge on digits/CJK). Gate: the
  29-case fixture (`tokenizer_onechart`) at 29/29 with a dedicated Gpt2 mode.
- **OQ-D2** Oracle env: `modeling_OneChart.py` was written on `transformers==4.32.1`;
  verify it loads on the GOT oracle env (4.37.2) or pin a separate venv. (Tokenizer ids
  are version-stable; fixture `_meta` records the generator version.)
- **OQ-D3** Bicubic squash: torchvision-on-PIL bicubic vs our CatmullRom — same known
  sub-L0 divergence as GOT; measure `preproc_max_abs_diff` on real charts (no CLIP
  normalize ⇒ raw-pixel scale, tolerance re-derived, not copied from GOT).
- **OQ-D4** `<Number>` protocol edge: model emits no `<Number>`, or emits it >1× (later
  fires overwrite `pred_locs` upstream). Port: first-fire wins per request; absent ⇒
  `None` + verdict "unverifiable" (never stale).
- **OQ-D5** OPT q-pre-scaling numerics: HF scales q before caching; our sdpa applies scale
  inside attention — mathematically equal, bit-different. Decide ONE placement and prove
  decode m=1 == prefill last-row (same bias-add-order discipline as GOT §13a).
- **OQ-D6** Attention-mask/position basis: upstream computes positions from cumsum(mask)−1;
  unpadded single-batch ⇒ 0..L−1 (row i+2). Assert no pad path is ever taken in our runtime.
- **OQ-D7** The 4096-position hard cap vs `max_new_tokens=4096` (§4): pin the port's stop
  condition + a loud event when hit.
- **OQ-D8** `vision_select_layer=-2` is in `config.json` but **dead code** in
  `modeling_OneChart.py` (read, never used — the tower output is always the final conv
  map). Confirm no oracle divergence implies it; do not implement.

## 12. Reuse map → franken_ocr beads (what D-lane actually builds)

**Reuse near-as-is:**
- **SAM-ViT-B tower + Vary compressor** — `vision_sam.rs` `forward_prefix(weights, image,
  "model.vision_tower")`: the prefix is ALREADY parameterized (`vision_sam.rs:274`); GOT
  needed `model.vision_tower_high`, OneChart is a third prefix value. Leaf names + geometry
  + rel-pos shapes verified identical (§13). → **D3 ≈ config, not code.**
- **Connector splice** — `vision_sam::Linear::apply` (1024→768) + `connector::masked_scatter`
  over `<imgpad>`=50265. → **D3/D4.**
- **int8 GEMM kernels** (`simd::igemm_s8s8` ladder, `gemv_i8*`, dequant-once cache, batched
  prefill driver) — arch-agnostic, drop in for the 6 decoder GEMMs/layer. → **D4.**
- **Tokenizer engine** — `src/tokenizer/mod.rs` byte-BPE + added-token trie; needs the
  Gpt2 pretok mode + vocab/merges ingestion (§7). **NOT `tiktoken.rs`.** → **D9.**
- **Sampler/driver spine** — greedy argmax + eos-stop; ngram guard OFF (contract §10).

**NEW (build):**
- **`OptDense` decoder driver** — `decoder_qwen2.rs`'s `DecoderConfig` cannot express OPT.
  Exactly what it lacks (field-by-field vs `decoder_qwen2.rs:110-133`):
  1. **No RoPE at all** — needs learned-position embedding add (`embed_positions[i+2]`)
     at the embedding seam instead of `RopeTable`/`apply_rope` on q/k (there is no
     "θ=None" escape hatch today).
  2. **LayerNorm (mean-subtract, WITH bias)** — driver only calls `nn::rms_norm`; needs
     `nn::layer_norm` w/ weight+bias at eps 1e-5 (HF OPT default — read from code, not
     config; confirm vs oracle).
  3. **Non-gated 2-GEMM ReLU MLP** (`fc1`/`fc2` + biases) — the SwiGLU gate/up/down path
     (incl. `expert_gemv_i8` fusion) does not fit; needs a `gemv→relu→gemv` layer with
     bias adds (prefill: `linear_int8_dynamic` twice).
  4. **Biases EVERYWHERE**: q/k/v (exists for GOT) **+ out_proj + fc1/fc2** (GOT's o_proj
     and MLP are bias-free — the bias-add path must generalize per-GEMM).
  5. **q pre-scale placement** (OQ-D5) with the m=1-vs-prefill byte-match cert.
  6. Growing full-causal KV cache: `Qwen2KvCache` pattern reusable as-is (MHA 12×64,
     scale 1/8, no GQA — same shape discipline, smaller dims).
  7. Tied f32 lm_head over vocab 50269 (same one-matrix pattern; int8 lm_head only behind
     the measured kill-switch).
- **Number head + self-verify** (§8) — all NEW, all HP/f32: the post-final-norm decode-step
  hidden tap keyed on input id 50268 (today's driver exposes only logits); the 3-linear
  ReLU MLP (runs ONCE per request, perf-irrelevant); the JSON numeric-leaf walker + exact
  string-normalization regexes + min-max + mean-L1 + 0.1 threshold; structured
  reliability fields in the result. → **D4 (head) + D5 (verify/render).**
- **Prompt builder** — the single fixed 309-id prompt (§5); simpler than GOT's modes.
- **model_arch registry corrections** — `model_arch.rs` currently annotates OneChart as
  `Decoder::Qwen2Dense` + `TokenizerKind::Qwen2Bpe`: **both wrong** (needs `OptDense` +
  a Gpt2/Opt BPE kind); `VisionEncoder::SamVit` + `Task::Chart` stand.

**int8 overflow (doctrine #6):** worst K = 3072 (`fc2`) ⇒ 3072·127·127 = 49,548,288 ≈
2.3% of i32::MAX (safer than the proven Baidu K=6848); other K = 768. Add
`KCase{k:3072}` + `KCase{k:768}` to `tests/int32_overflow_proof.rs`.

## 13. Conversion / quant plan (D2) — exact tensor names (safetensors header, 2026-07-01)

384 tensors, all BF16, single shard 524,672,832 B (header-verified complete). Repo also
ships a duplicate `pytorch_model.bin` — **convert from `model.safetensors` only.**

| tensor | shape | store |
|---|---|---|
| `model.decoder.layers.{i}.self_attn.{q,k,v}_proj.weight` (i∈0..12) | [768,768] | **int8** |
| `model.decoder.layers.{i}.self_attn.out_proj.weight` | [768,768] | **int8** |
| `model.decoder.layers.{i}.fc1.weight` / `fc2.weight` | [3072,768] / [768,3072] | **int8** |
| `…self_attn.{q,k,v,out}_proj.bias`, `fc1.bias` [3072], `fc2.bias` [768] | [768] etc. | HP |
| `model.decoder.layers.{i}.self_attn_layer_norm.{weight,bias}`, `…final_layer_norm.{weight,bias}` | [768] | HP |
| `model.decoder.final_layer_norm.{weight,bias}` | [768] | HP |
| `model.decoder.embed_tokens.weight` (serves embed AND lm_head; `lm_head.weight` NOT stored — §4 SHA-proof) | [50269,768] | HP |
| `model.decoder.embed_positions.weight` (offset-2 table) | [4098,768] | HP |
| `model.mm_projector.{weight,bias}` | [768,1024]/[768] | HP |
| `model.vision_tower.*` (patch_embed, pos_embed, 12 blocks w/ rel_pos_{h,w}, neck.0-3, net_2, net_3) | as GOT §12, prefix swapped | HP |
| `num_decoder.{0,2,4}.{weight,bias}` | [384,768],[384,384],[256,384] + biases | HP |

6 int8 GEMMs/layer × 12 = 84,934,656 int8 params; **`.focrq` ≈ 362 MB** (85 MB int8 +
~138.8 M HP params bf16). Tokenizer assets to bundle: `vocab.json` + `merges.txt` +
`added_tokens.json` (or the synthesized `tokenizer.json`), SHA-pinned in the fixture `_meta`.
Manifest id: `onechart`.

## 14. Oracle + parity ladder (D6) — L0→L5

Mirror the GOT ladder (§13c there); model is 262 M ⇒ a CPU-f32 torch oracle is cheap.
**Establish the oracle nondeterminism floor FIRST** (4 runs/chart × 2 thread counts →
`tolerances.toml`), then:

- **L0a tokenizer**: `tok_id_mismatch_count==0` on the 29 golden cases
  (`tests/fixtures/tokenizer_onechart/expected.json`) — the D9 gate.
- **L0b preprocess**: `preproc_max_abs_diff ≤ tol` (raw [0,1] pixels, OQ-D3) + single-tile
  layout exact.
- **L0c prompt**: id-exact vs the committed 309-id `prompt_fixture`.
- **L1 per-op**: cos ≥ 1−1e-6 (tower blocks, neck, net_2/3, projector, one OPT layer,
  layer_norm, learned-pos add, num_decoder MLP).
- **L2 per-seam**: SAM-out (256×1024) / projector-out (256×768) / 12 decoder hiddens /
  final-norm / **the `<Number>`-step hidden (768)**.
- **L3 logits + pred_locs**: prompt-step logits (f32 + int8 tracked separately) AND the
  **256-float `pred_locs` vector** (L∞ + mean-L1 vs oracle, tol from the floor) — the new
  seam class vs GOT.
- **L4 greedy ids**: id-exact to first divergence, eos-terminated, cap 4096.
- **L5 task metric**: SCRM (the official `ChartSE_eval/SCRM.py`: em, mAP strict/slight/high
  @0.5/0.75/0.9) on ChartSE if downloaded; day-1 proxy = a **synthetic matplotlib corpus**
  (bar/line/pie w/ known values — the same specialized-corpus gap flagged in
  `GOT_NEXT_STEPS.md` §2) gated on (a) valid-JSON rate, (b) per-value relative error,
  (c) `reliable_distance` agreement with the oracle verdict.

**Oracle script shape** (new `onechart` branch in `scripts/gen_reference_fixtures.py`):
`AutoTokenizer/AutoModel.from_pretrained('kppkkp/OneChart', trust_remote_code=True)` on CPU
f32, `model.eval()`; forward hooks on `model.vision_tower`, `model.mm_projector`, each
`model.decoder.layers[i]`, `model.decoder.final_layer_norm`, `num_decoder`; replicate
`chat()` minus the streamer; dump NDJSON events + `.npy` seams + `pred_locs` + the
reliability tuple; transformers pin per OQ-D2. The `reliable_check` post-math (regex walk,
min-max, round-4, L1) is pure-python — replicate as a golden unit-test vector set, no torch
needed.

## 15. Task-DAG delta for D2-D8 (returned as census output; beads not edited here)

- **D9 (bd-3jo6.4.9) — REWRITE SCOPE.** Title says "GOT/Vary lineage" tokenizer: OneChart
  is **OPT GPT-2 BPE** (vocab.json+merges.txt), NOT Qwen tiktoken. Scope = Gpt2 pretok mode
  + vocab/merges ingestion + bos policy + the 29-case fixture gate. `tiktoken.rs` is
  irrelevant. (Fixture already committed by D1.)
- **D4 (bd-3jo6.4.4) — decoder family correction.** "Wire the shared decoder (A7)" assumed
  Qwen2Dense; OneChart needs the **OptDense** driver (§12 items 1-7). The A7 dep
  (bd-3jo6.1.7) still holds but A7's planned deltas (GQA, RoPE scaling, act swap) do NOT
  cover: no-RoPE learned positions, LayerNorm+bias, non-gated ReLU MLP, out_proj/fc biases.
  Either extend A7's DecoderConfig with these four axes or add a sibling driver; the number
  head + hidden tap (§8) is D4 scope either way.
- **D2 (bd-3jo6.4.2)** — quant map = §13 (6 GEMMs/layer int8, tie de-dup, ~362 MB);
  tokenizer-asset bundling decision (synth tokenizer.json vs raw vocab/merges) lands here.
- **D3 (bd-3jo6.4.3)** — mostly config: third `vision_tower` prefix + the no-op-normalize
  preprocess key (mean 0/std 1 — do NOT reuse GOT's CLIP constants) + single-tile only.
- **D5 (bd-3jo6.4.5)** — schema §9 + brace-completion + the reliability fields
  (`pred_values`, `reliable_distance`, `reliable: bool@0.1`) in `-o` json.
- **D6 (bd-3jo6.4.6)** — ladder §14; NEW BLOCKER: no chart test corpus exists on this
  machine (GOT_NEXT_STEPS §2) — synthetic matplotlib corpus + optional ChartSE download.
- **D7 (bd-3jo6.4.7)** — CLI `focr chart` / `--task chart-data`; prompt is fixed (§5), no
  format/box/color flags; plus `--no-verify` to skip the reliability pass.
- **D8 (bd-3jo6.4.8)** — e2e NDJSON logging: add the `<Number>`-fire event, pred_locs
  digest, reliability verdict; model-gated skip-with-SUCCESS without weights.
- **model_arch.rs** (A-lane, single-owner hot cluster adjacent) — registry comment/variant
  fix: OneChart ≠ Qwen2Dense/Qwen2Bpe (§12). File as a small A-lane follow-up.

### Sources
- config.json / modeling_OneChart.py / sam_vision_b.py / vocab.json / merges.txt /
  added_tokens.json / tokenizer_config.json / generation_config.json / model.safetensors
  (header + range probes) — https://huggingface.co/kppkkp/OneChart/tree/main
  (rev `79212de2f520694e534e58240f228eff351d536c`)
- GitHub (demo `run_opt_v1.py`, training `conversation_dataset_v1_with_number.py`,
  `vary_opt_math.py::number_loss`, eval `ChartSE_eval/`) — https://github.com/LingyvKong/OneChart
- Paper — https://arxiv.org/abs/2404.09987 (ACM MM 2024 Oral)
