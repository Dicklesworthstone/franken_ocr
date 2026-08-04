# SmolVLM2-500M — architecture spec (bead bd-3jo6.3.1 / C1)

Implementation-ready census of **SmolVLM2-500M-Video-Instruct**
(`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`; SmolVLM2 blog Feb 2025; **Apache-2.0**)
for a from-scratch pure-Rust CPU port in franken_ocr (epic `bd-3jo6`, sub-epic C:
photo description + VQA). Every load-bearing number is cited to a source file; this
doc is self-contained so the implementer never needs to reverse-engineer the model.
**Source of truth = the released weights repo files** (`config.json`,
`preprocessor_config.json`, `processor_config.json`, `generation_config.json`,
`chat_template.json`, `tokenizer.json`, `added_tokens.json`) + the merged
`transformers` model code (`models/smolvlm/*.py`, in-tree since v4.50 — **no
trust_remote_code**, unlike GOT). Weight facts below were verified against the
actual `model.safetensors` header + byte-range probes (2026-07-01), §12.

> **Headline.** Encoder = **SigLIP-B/16 @ 512²** (12L, 768-dim, bidirectional) →
> 1024 patch tokens. Connector = **pixel-shuffle ×4** (1024×768 → 64×12288) + a
> single **`Linear(12288→960, bias=False)`**. Decoder = **SmolLM2-360M,
> Llama-architecture DENSE** (32 layers, hidden 960, **GQA 15q/5kv — THE delta A7
> must grow**, SwiGLU, RoPE θ=1e5, RMSNorm 1e-5, **UNTIED lm_head — verified,
> unlike GOT**). 507.5 M params total; `model.safetensors` **2.03 GB float32**
> (not bf16), single shard, 489 tensors.

---

## 1. Top-level graph (Idefics3/SmolVLM splice)

```
image (any size, RGB)
  → preprocess     (resize longest→2048, ceil to 512-multiples, split R×C 512² tiles
                    + global 512² thumbnail LAST; ≤4×4+1=17 frames)          → F×3×512×512
  → Vision encoder (SigLIP-B/16: patch-embed conv + 1024 learned pos + 12 blocks
                    + post_layernorm)                                        → F × 1024 × 768
  → Connector      (pixel_shuffle ×4 → F × 64 × 12288; Linear(12288→960))    → F × 64 × 960
  → splice into decoder embeddings at the 64·F <image> (49190) slots
    (bracketed by <fake_token_around_image> + <row_r_col_c>/<global-img> markers)
  → Decoder        (SmolLM2-360M dense, 32L, hidden 960, GQA 15/5)           → text tokens
```

Param split (safetensors, exact): vision 86.43 M, connector 11.80 M, decoder body
314.64 M, embed_tokens 47.31 M, lm_head 47.31 M (**separate — untied**), **total
507.48 M**. `image_seq_len = 64` tokens per 512² frame (`processor_config.json`).

---

## 2. Vision encoder — SigLIP-B/16 (`model_type: smolvlm_vision`)

| field | value | source |
|---|---|---|
| input | 512×512×3 per frame | `vision_config.image_size` 512 |
| patch_size | 16 (`Conv2d(3,768,k16,s16)`, bias=True) → 32×32 = 1024 tokens | config + tensor [768,3,16,16] |
| hidden_size | 768 | config |
| num_hidden_layers | **12** (config.json OMITS it; transformers default 12 — **verified: safetensors has layers 0..11**) | §12 |
| num_attention_heads | 12 (head_dim 64, scale 1/8) | config |
| intermediate_size | **3072** (config omits; default — verified fc1 [3072,768]) | §12 |
| hidden_act | `gelu_pytorch_tanh` (tanh-approx GELU; config omits, code default) | configuration_smolvlm.py |
| layer_norm_eps | 1e-6 (config omits, code default) | configuration_smolvlm.py |
| attention | q/k/v/out_proj all **with bias**, full **bidirectional** (no causal mask, no windows, no rel-pos) | tensor keys |
| pos embed | learned `nn.Embedding(1024,768)` added to patch embeds; **no CLS token** | modeling_smolvlm.py |
| block | pre-LN: `x += attn(LN1(x)); x += fc2(act(fc1(LN2(x))))`; final `post_layernorm` on the encoder output | modeling_smolvlm.py |

**Position ids (NaViT bucketize):** upstream supports variable aspect via fractional
bucketize over `patch_attention_mask`. **CORRECTION (2026-07-02, caught by the C3
embeddings-seam parity gate — this paragraph originally claimed identity ids):**
the reference computes `fractional_coords = (i/32) * (1 - 1e-6)`, and that scale
pushes every exact multiple JUST BELOW its own `i/32` boundary, so
`bucketize(·, right=True)` yields per-axis buckets **`[0, 0, 1, 2, …, 30]`** —
coordinates 0 and 1 share bucket 0 and bucket 31 is never used. For the fixed
full-mask 512² path this is exactly `max(i-1, 0)` per axis
(`pos_id = max(r-1,0)*32 + max(c-1,0)`), verified bit-level against the live
module (`vision_siglip::embed_frame`; manual identity-ids conv+pos scored only
cos 0.8416 against the true `hidden_states[0]`). Port THIS path; keep the
general variable-aspect path out of scope.

**vs SAM-ViT-B (GOT/Baidu):** same patch-embed geometry (k16 s16 — the A8 conv
leaf reuses), but SigLIP has separate q/k/v (not fused qkv), no windowed layers,
no decomposed rel-pos, no neck/compressor, tanh-GELU (not erf-GELU), and a plain
learned 1-D pos table. This is a **new tower** (C3), not a `vision_sam.rs` variant.

## 3. Connector — pixel-shuffle ×4 + `Linear(12288→960, bias=False)`

`modeling_smolvlm.py::SmolVLMConnector` (exact, `scale_factor=4` from config):

```
x: [B, 1024, 768]; h = w = 32; s = 4
x = x.view(B, h, w, d)
x = x.view(B, h, w/s, d*s)          # fold s cols into channels
x = x.permute(0, 2, 1, 3)
x = x.reshape(B, w/s, h/s, d*s*s)   # fold s rows into channels
x = x.permute(0, 2, 1, 3)
x = x.reshape(B, 1024/(s*s), d*s*s) # → [B, 64, 12288]
out = x @ proj.T                    # modality_projection.proj [960, 12288], NO bias
```

Pixel-shuffle is pure data movement (A9); one f32 GEMM K=12288 (HP). The connector
runs on the **post_layernorm** output. Both `scale_factor: 4` (top level) and
`text_config.pixel_shuffle_factor: 4` agree.

## 4. Language decoder — literal `config.json .text_config` (SmolLM2-360M, `model_type: llama`)

| field | value | | field | value |
|---|---|---|---|---|
| hidden_size | **960** | | rope_theta | **100000.0** (1e5, NOT GOT's 1e6) |
| num_hidden_layers | **32** | | rope_interleaved | false (NEOX rotate-half — same as GOT) |
| num_attention_heads | **15** | | rope_scaling | none |
| **num_key_value_heads** | **5** (**GQA 3:1** — kv width 320) | | rms_norm_eps | **1e-5** (NOT GOT's 1e-6) |
| head_dim | **64** (explicit; 15·64=960) | | hidden_act | silu / SwiGLU (llama default) |
| intermediate_size | **2560** | | attention/mlp bias | **none** (verified: zero bias tensors) |
| vocab_size | **49280** | | tie_word_embeddings | **FALSE** (top-level config; **byte-verified untied**, §12) |
| max_position_embeddings | **8192** | | qk_layer_norms | false (no q/k norm) |
| torch_dtype | **float32** | | attention | full causal, no sliding window |

Structurally **SmolLM2-360M** with the embedding resized 49152→49280 for the 128
image/control tokens. `architectures: ["VLlama3ForCausalLM"]` is a training-fork
label; the graph is stock Llama. Ignore the top-level `pad_token_id: 128002`
(an Idefics3-Llama3 leftover); real pad = 2 (`text_config.pad_token_id`, tokenizer).
The nested `perceiver_config` is dead config (`use_resampler: false`) — no resampler
tensors exist.

**Deltas vs the GOT/Qwen2 driver (`decoder_qwen2.rs`), for A7/C5:** GQA (the hard
one), untied lm_head, **no** qkv bias (`attn_qkv_bias=false`), θ=1e5, eps=1e-5,
vocab 49280, 32 layers, no_repeat_ngram **0** (upstream has no repetition guard).
Everything else (dense SwiGLU, NEOX RoPE over full head_dim 64, scale 1/8,
full-causal growing KV) is the same shape of machine.

## 5. Special tokens + prompt templates

Base BPE vocab 49152 (ids 0..49151, of which 17 are in-range specials:
`<|endoftext|>`=0 (unk), `<|im_start|>`=1 (bos), `<|im_end|>`=2 (pad), plus the
SmolLM stack tokens `<repo_name>`=3 … `<empty_output>`=16). 128 added tokens fill
49152..49279 with no gap (`added_tokens.json`): `<global-img>`=49152,
**`<row_{r}_col_{c}>`=49153..49188** (6×6 grid, row-major), `<fake_token_around_image>`=49189,
**`<image>`=49190** (`image_token_id`), `<|reserved_special_token_0..87|>`=49191..49278,
**`<end_of_utterance>`=49279 (eos)**. `generation_config.json`: bos 0, eos **49279**,
pad 2 (its bos=0 disagrees with the tokenizer's bos=1 — harmless, the template
supplies `<|im_start|>` literally and nothing auto-prepends, §7).

**Image splice (processor `get_image_prompt_string`, exact):** each `<image>`
placeholder in the user text expands to, for a split image with R rows × C cols:

```
for r in 1..=R:
  for c in 1..=C: <fake_token_around_image> <row_r_col_c> <image>×64
  "\n"
"\n" <fake_token_around_image> <global-img> <image>×64 <fake_token_around_image>
```

(unsplit/video-frame form: `<fake_token_around_image><global-img><image>×64<fake_token_around_image>`).
The `"\n"` are ordinary text — the last row's `"\n"` abuts the final `"\n"` so
`"\n\n"` may BPE-merge to one id; **pin exact counts by fixture (L0c), not formula**.
At forward, `inputs_merger` overwrites the `<image>` (49190) embedding rows with the
64·F projected vision rows, in frame order (tiles row-major, **global LAST** —
mirrors GOT's thumbnail-last, opposite bracket tokens).

**Chat template (`chat_template.json`, exact):** one literal `<|im_start|>` at the
very start (NOT per message), then per message `Capitalize(role)` + `":"` if the
first content item is an image else `": "`, the content items (text verbatim,
image → `<image>` pre-expansion), then `<end_of_utterance>\n`. Generation prompt
suffix: `Assistant:`.

**Describe (literal, single image):**
```
<|im_start|>User:{IMAGE_EXPANSION}Can you describe this image?<end_of_utterance>
Assistant:
```
**VQA:** same shape with the question text; text-first VQA renders `User: {question}<image…>`
with the space after the colon. Multi-turn appends `Assistant: {answer}<end_of_utterance>\n`
turns. There are no per-task instruction modes (no GOT-style `OCR:`/`OCR with format:`) —
task = the natural-language question. Video prompts (frame-timestamp intro text,
1 fps, ≤64 frames, no splitting) exist upstream but are **out of scope for C**.

## 6. Preprocessing — `SmolVLMImageProcessor` (`preprocessor_config.json`)

Order (all resizes **LANCZOS**, `resample: 1`):
1. `convert RGB`.
2. **Resize longest edge to exactly 2048** (`size.longest_edge`; UP- or down-scale,
   aspect preserved; the short edge is bumped +1 if odd; absolute cap 4096).
   Because this always rescales TO 2048, still images are **always split** (the
   long side always yields 4 tiles).
3. `resize_for_vision_encoder`: ceil each side to a multiple of 512
   (`max_image_size.longest_edge`) via the aspect-derived formula (long side first,
   short side recomputed from aspect then ceiled).
4. `split_image`: R=H/512 × C=W/512 grid of exact 512² crops, **row-major**, then
   append the step-2 image resized to 512×512 (the **global frame, LAST**).
   R,C ≤ 4 → ≤17 frames = **≤1088 `<image>` ids** (vs GOT's 1792).
5. Per frame: rescale ×1/255 → normalize **mean (0.5,0.5,0.5), std (0.5,0.5,0.5)**
   (x/127.5 − 1, range [−1,1]; NOT the CLIP constants GOT uses).
6. `do_pad` pads a batch to max frame count with zero-frames + mask — a no-op for
   our fixed single-image path (all frames are 512²; `get_image_features` drops
   all-zero pad frames).

## 7. Tokenizer — SmolLM2 GPT-2-style byte-level BPE (standard HF `tokenizer.json` — NOT tiktoken)

`tokenizer_class: GPT2Tokenizer`; files `tokenizer.json` (canonical; also
`vocab.json`+`merges.txt`). Model: BPE, 49152 vocab, 48900 merges, no
byte_fallback, no dropout. **Pre-tokenizer = `Sequence[ Digits(individual_digits=true),
ByteLevel(add_prefix_space=false, use_regex=true) ]`** — digits are split
one-per-token BEFORE the GPT-2 regex; this is the one stage neither existing Rust
path has (`pretok.rs` implements the DeepSeek 4-stage sequence with `use_regex=false`;
`tiktoken.rs` implements the Qwen regex, which also differs from GPT-2's).
Normalizer null, post_processor **null** → `encode(add_special_tokens=True)` adds
NOTHING; the id stream is exactly the BPE of the rendered template string, with
added-token (specials) splitting first. Decoder: ByteLevel. So C6 = the existing
HF-JSON loader spine (`src/tokenizer/mod.rs::from_json_bytes`, already proven on
the Baidu DeepSeek `tokenizer.json`) + a Digits stage + the GPT-2 `use_regex` split.

## 8. Generation / postprocessing

`generation_config.json` pins only ids (bos 0 / eos **49279** / pad 2) — no
sampling defaults, **no repetition guard** (`no_repeat_ngram_size` absent ⇒ 0; cf.
GOT's hard-coded 20 — keep the bd-ff4i guard available as an off-by-default
kill-switch). Model-card usage (all README examples): **`do_sample=False`
(greedy), `max_new_tokens=64`**; stop on `<end_of_utterance>` (49279); decode with
`skip_special_tokens=True`; the reply is everything after the final `Assistant:`.
Output is free text (captions/answers) — no markdown/LaTeX renderers, no
coordinate decoding; postprocess = strip the trailing eos.

## 9. Open questions (doctrine hard rule — no kernel ships against an unresolved OQ)

- **OQ-1** `gelu_pytorch_tanh` numerics: `nn.rs` has exact erf-GELU + CLIP
  quick_gelu but NOT the tanh approximation `0.5x(1+tanh(√(2/π)(x+0.044715x³)))` —
  new leaf, parity-gate vs torch.
- **OQ-2** LANCZOS resize parity: PIL LANCZOS vs the image-crate `Lanczos3`
  (the SmolVLM analog of GOT's known CatmullRom≈bicubic sub-L0 divergence) —
  measure at L0b, record in `tolerances.toml`.
- **OQ-3** step-2 resize rounding: `int()` truncation + `+1 if odd` on the short
  edge (transcribed from `_resize_output_size_rescale_to_max_len`) — pin with L0b
  fixtures across aspect ratios (portrait/landscape/square, up- and down-scale).
- **OQ-4** `"\n"`/`"\n\n"` BPE ids inside the image expansion (merge-dependent,
  §5) — pin via L0c prompt-id fixtures, don't hand-compute.
- **OQ-5** oracle pin: checkpoint saved with transformers 4.47.1 (pre-merge fork);
  in-tree `smolvlm` exists since v4.50 — confirm a `>=4.50,<5` CPU-f32 oracle
  reproduces the model-card outputs before trusting fixtures.
- **OQ-6** int8 policy for a VQA/caption decoder: the GOT "repetition-runs on hard
  tables" int8 failure mode has no CER-style metric here — define the L5 quality
  gate (§13) BEFORE enabling int8-by-default.

## 10. Reuse map → franken_ocr beads

**Reuse near-as-is:**
- **Dense decoder driver** (`decoder_qwen2.rs` — already `DecoderConfig`-parameterized
  for hidden/inter/layers/heads/head_dim/vocab/θ/eps/`attn_qkv_bias`/ngram) + the
  int8 GEMM stack (`simd::igemm_s8s8`, `gemv_i8*`, `linear_int8_dynamic`) +
  `RopeTable` (NEOX, param θ) + `nn::rms_norm` + `Qwen2KvCache` growth pattern +
  the sampler. → **A7/C5**, with the §4 deltas.
- **HF `tokenizer.json` BPE loader** (`src/tokenizer/mod.rs`) → **A6/C6** (+ §7 pretok stage).
- **Patch-embed conv** (16×16 s16 im2col leaf from `vision_sam.rs`) → **A8/C3**.
- `connector::masked_scatter` (the `<imgpad>` splice engine) → the `<image>`-slot
  splice, id 49190, 64·F rows. → **C7**.
- GEMM attention (`nn::sdpa`/`prefill_attention`) → SigLIP bidirectional attention
  (drop the causal mask; 1024 tokens, 12 heads). → **C3**.

**NEW (build):**
- **GQA in the shared engine** — `num_key_value_heads` field + kv-head broadcast in
  prefill + decode attention + kv-cache stride ([320] panels vs [960]); the exact
  touch-list (`qkv_dim()`, `concat_qkv`, `split_qkv_rows`, `decode_attn_head`,
  `prefill_attention`/`repeat_kv`) is in `docs/zoo/GOT_NEXT_STEPS.md` §5. → **A7**.
- **Untied lm_head** — GOT stores ONE matrix; SmolVLM2 needs both `embed_tokens`
  AND `lm_head` in `.focrq` + a driver flag. → **A7/C2** (new-found delta).
- **SigLIP tower** (12 pre-LN blocks, separate q/k/v+bias, tanh-GELU, learned pos
  table, post_layernorm; no neck). → **C3**.
- **Pixel-shuffle ×4** (§3 exact permutes) + `Linear(12288→960)` (HP). → **A9/C4**.
- **Digits pre-tokenizer + GPT-2 ByteLevel regex** in `pretok.rs`. → **C6**.
- **Prompt builder** (chat template §5 + image expansion + describe/VQA modes) +
  preprocess (§6 resize/split). → **C7**.
- `model_arch.rs` already reserves `VisionEncoder::Siglip`, `Decoder::LlamaDense`,
  `TokenizerKind::SmolLm2Bpe` — fill in the `PlannedArch` for `smolvlm2-500m`.

## 11. Conversion / quant plan (C2)

`.focrq` under the doctrine-#2 policy, keyed by arch id `smolvlm2-500m`: **int8**
the 32 decoder layers' `{q,k,v,o}_proj` + `{gate,up,down}_proj` (7 GEMMs/layer;
kv panels are [320,960]); `lm_head` [49280,960] int8 only behind the measured
quality kill-switch (OQ-6). **High-precision** everything else: the SigLIP tower,
the connector proj, `embed_tokens`, all norms (+ their biases). **Store BOTH
`embed_tokens` and `lm_head` — untied (§12), the opposite of GOT's de-dup.**
Source dtype is **F32** (2.03 GB) — convert reads f32 directly, no bf16 widening.
int8 overflow proof (doctrine #6): worst K = 2560 (`down_proj`) ⇒ 2560·127·127 =
41,290,240 ≈ 1.9% of i32::MAX (safer than the proven K=6848); add `KCase{k:2560}`
and `KCase{k:960}` (q/k/v/o + gate/up) to `tests/int32_overflow_proof.rs` — the
K=12288 connector GEMM stays HP, no proof needed.

## 12. Weight-level facts (verified from the released `model.safetensors`, 2026-07-01)

From the parsed safetensors header (489 tensors, all F32, single shard) + byte-range
probes — no full download needed:

- **UNTIED embeddings CONFIRMED.** `lm_head.weight` [49280,960] and
  `model.text_model.embed_tokens.weight` [49280,960] are both stored and their
  bytes **differ** (first-4MB and last-4MB chunks compared via ranged GETs —
  distinct SHA-256 on both). Matches top-level `tie_word_embeddings: false`. The
  convert MUST keep both (re-verify full-tensor inequality at convert time).
- **Vision depth CONFIRMED 12** (`vision_config` omits `num_hidden_layers`;
  encoder layers 0..11 exist), intermediate 3072 (`fc1` [3072,768]).
- **No decoder biases, no q/k norms, no rotary tensors, no resampler tensors** —
  the text tower is exactly the §4 table.
- Param split: vision 86.43 M / connector 11.80 M / text body 314.64 M / embed
  47.31 M / lm_head 47.31 M = **507.48 M**.

**Exact tensor names (for the C2 convert map):**
- Decoder (×32, `model.text_model.layers.{i}`): `.input_layernorm.weight` [960],
  `.post_attention_layernorm.weight` [960]; `.self_attn.q_proj.weight` [960,960],
  `.self_attn.{k,v}_proj.weight` **[320,960]**, `.self_attn.o_proj.weight` [960,960]
  (all bias-free); `.mlp.gate_proj.weight` [2560,960], `.mlp.up_proj.weight`
  [2560,960], `.mlp.down_proj.weight` [960,2560]. Top: `model.text_model.embed_tokens.weight`
  [49280,960], `model.text_model.norm.weight` [960], `lm_head.weight` [49280,960].
- Vision (×12, `model.vision_model.encoder.layers.{i}`): `.layer_norm1.{weight,bias}`
  [768], `.layer_norm2.{weight,bias}` [768]; `.self_attn.{q,k,v,out}_proj.{weight [768,768], bias [768]}`;
  `.mlp.fc1.{weight [3072,768], bias [3072]}`, `.mlp.fc2.{weight [768,3072], bias [768]}`.
  Embeddings: `model.vision_model.embeddings.patch_embedding.weight` [768,3,16,16]
  + `.bias` [768], `…embeddings.position_embedding.weight` [1024,768];
  `model.vision_model.post_layernorm.{weight,bias}` [768].
- Connector: `model.connector.modality_projection.proj.weight` [960,12288] (no bias).

## 13. Oracle / parity ladder (C8) — mirror `scripts/gen_reference_fixtures_got.py`

Build `scripts/gen_reference_fixtures_smolvlm2.py` with the same skeleton (offline
tooling; isolated venv `uv venv /private/tmp/smolvlm2_oracle_venv`, `torch` +
`transformers>=4.50,<5` + `num2words` if the processor demands it; CPU float32;
`AutoModelForImageTextToText.from_pretrained(...)` + `AutoProcessor` — NO
trust_remote_code). Same doctrine order: **nondeterminism floor FIRST** (2 runs @1
thread + 1 run @2 threads → `tolerances.toml` seeds L2/L3), then emit the committed
compact JSON (`tests/fixtures/smolvlm2/oracle_fixtures.json`) + off-repo full-tensor
`.npz`. Shared-input contract: commit BOTH a rendered describe image
(`tests/fixtures/smolvlm2/sample_photo.png` — needs a real photo-like fixture, see
follow-ups) and a VQA prompt pair.

Ladder:
- **L0a tokenizer**: `tok_id_mismatch_count==0` on a corpus that stresses Digits
  (numbers, dates, decimals), specials, UTF-8, whitespace runs (mirror
  `gen_got_token_id_fixtures.py` → new `gen_smolvlm2_token_id_fixtures.py`).
- **L0b preprocess**: `preproc_max_abs_diff ≤ tol` per frame + layout-exact
  (R, C, frame order, global-LAST) across ≥4 aspect ratios (OQ-2/OQ-3).
- **L0c prompt**: id-exact rendered describe AND VQA prompts, including the full
  image expansion (pins OQ-4).
- **L1 per-op**: tanh-GELU, pixel-shuffle, GQA broadcast, RoPE θ=1e5 (cos ≥ 1−1e-6).
- **L2 per-seam**: patch-embed+pos out → 12 vision hiddens → post_layernorm →
  pixel-shuffle out → connector out → splice check → 32 decoder hiddens → final norm.
- **L3 logits**: floor-derived tol; f32 and int8 tracked separately.
- **L4 greedy decode**: id-exact to first divergence (greedy + eos 49279, no ngram guard).
- **L5 task quality** (replaces GOT's CER): fixed VQA set scored by normalized
  exact-match + caption fixtures scored by keyword-set containment vs the oracle's
  own greedy output (NOT vs human ideal — parity, not benchmark); int8-vs-f32
  divergence gate per OQ-6.

## 14. Task-DAG delta — beads C2–C10 (exact, from `.beads` 2026-07-01)

**C1 (bd-3jo6.3.1) = this doc → close.** Current bead edges are right in shape; the
delta is which are READY vs still A-blocked, plus two scope corrections:

| bead | verdict | why |
|---|---|---|
| C6 `bd-3jo6.3.6` tokenizer | **READY NOW** | needs only C1 + the existing HF-JSON loader; its A6 (`bd-3jo6.1.6`) edge is satisfiable as "C6 IS the A6 SmolLM2 instance" — new work = §7 Digits+GPT-2-regex stage + fixtures (L0a) |
| C2 `bd-3jo6.3.2` convert | **READY after C1**, still blocked on A3/A4 (`bd-3jo6.1.3`/`.1.4` generalized convert/pull) | quant map is §11; **scope ADD: untied dual-matrix storage in `.focrq` v2** (new-found; GOT path de-dups) |
| C3 `bd-3jo6.3.3` SigLIP | **partially unblocked** | only the patch-embed conv leaf needs A8 (`bd-3jo6.1.8`); the 12 blocks + tanh-GELU (OQ-1) + bidirectional attention can start against §2/§12 immediately — recommend narrowing the A8 edge to the conv leaf |
| C4 `bd-3jo6.3.4` pixel-shuffle | blocked on A9 (`bd-3jo6.1.9`) — but §3 IS the A9 spec | recommend: A9 = the generic kernel + L1 parity, C4 = SmolVLM2 wiring; a one-day pair, do together |
| C5 `bd-3jo6.3.5` decoder | **blocked on A7-GQA** (`bd-3jo6.1.7`) — the critical path | GQA touch-list = GOT_NEXT_STEPS §5; **scope ADD to A7: untied lm_head + `attn_qkv_bias=false` path** (θ/eps already parameterized) |
| C7 `bd-3jo6.3.7` prompt/IO | blocked on C5+C6 (as filed) | template+expansion pinned here (§5/§6), so L0c fixtures + the prompt-builder unit tests can land WITH C6, before C5 |
| C8 `bd-3jo6.3.8` parity+e2e | blocked on C3/C4/C5 (+`bd-3jo6.1.10`) | **pull the §13 oracle-fixture script forward** — it only needs C1 + the upstream model, and every C3/C4/C5 seam test consumes it; file as a new early sub-bead |
| C9 `bd-3jo6.3.9` CLI | blocked on C7 + `bd-3jo6.1.5` (as filed) | `focr describe` / `--task caption\|vqa`; no change |
| C10 `bd-3jo6.3.10` tests | blocked on C8+C9 (as filed) | no change |

**Recommended order:** C6 → oracle-fixtures script (new bead) → C2 (once A3/A4
land) → C3+C4 (A8 leaf + A9) → C5 (after A7-GQA) → C7 → C8 → C9 → C10.

**Out of scope for this spec (state explicitly):** video (1 fps sampling, timestamp
intro text, 64-frame cap), multi-image interleave, the 256M/2.2B variants (256M =
same SigLIP-B + SmolLM2-135M [576h/30L/9q/3kv]; 2.2B = SigLIP-SO400M [1152/27L] +
SmolLM2-1.7B — each needs its own §2/§4 census before porting).

### Sources
- config.json / preprocessor_config.json / processor_config.json / generation_config.json /
  chat_template.json / tokenizer_config.json / added_tokens.json / special_tokens_map.json /
  tokenizer.json / model.safetensors (header + ranged probes) —
  https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/tree/main
- transformers v4.53 in-tree model code (`configuration_smolvlm.py`, `modeling_smolvlm.py`,
  `processing_smolvlm.py`, `image_processing_smolvlm.py`) —
  https://github.com/huggingface/transformers/tree/main/src/transformers/models/smolvlm
- SmolVLM2 release blog — https://huggingface.co/blog/smolvlm2 ; SmolVLM paper —
  https://arxiv.org/abs/2504.05299
