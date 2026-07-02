# Polyphonic-TrOMR — architecture spec (bead bd-3jo6.5.1 / E1)

Implementation-ready census of **Polyphonic-TrOMR** (`NetEase/Polyphonic-TrOMR`; "TrOMR:
Transformer-based Polyphonic Optical Music Recognition", arXiv
[2308.09370](https://arxiv.org/abs/2308.09370); **Apache-2.0**) for a from-scratch pure-Rust
CPU port in franken_ocr (epic `bd-3jo6`, sub-epic E). Every load-bearing number is cited to
a source file; this doc is self-contained so the implementer never needs to reverse-engineer
the model. **Source of truth = the GitHub repo files** (`tromr/workspace/config.yaml`,
`tromr/model/{encoder,decoder,tromr_arch}.py`, `tromr/staff2score.py`, `tromr/inference.py`,
the 4 tokenizer JSONs) **+ the committed checkpoint** (`tromr/workspace/checkpoints/`
`img2score_epoch47.pth`, 86,254,711 B fp32, SHA-256
`02925259ef59f5578a8c9e954ac363bb15538ea38ce73090b861c1519179f910` — the full 261-tensor
state-dict inventory in §12 was extracted from this file) **+ the exact pinned libraries**
(`timm==0.6.5`, `x-transformers==0.29.2`, per `requirements.txt` — both read at the pinned
version for every internal cited below).

> **Headline.** Input = ONE pre-cropped staff image, grayscale, height-normalized to
> **128 × W** (W ≤ 1280, multiple of 16). Encoder = **ResNetV2 [2,3,7] stem (weight-standardized
> convs + GroupNorm) → 1×1 proj → 4-layer ViT-256** → ≤ 641 context tokens. Decoder =
> **x-transformers 4-layer, dim 256, 8 heads, cross-attention every layer**, with **FOUR
> parallel output heads** (rhythm 260 / pitch 71 / lift 7 / note 2) over one shared hidden.
> **21,534,232 params fp32 (86 MB)** — the smallest model in the zoo, and the **WORST kernel
> fit in the zoo** (§10): CNN backbone + LayerNorm/abs-pos/GEGLU/cross-attn decoder — almost
> nothing rides the A7 fused Qwen2 driver; reuse is at the micro-kernel level only.

---

## 1. Top-level graph (`tromr_arch.py`)

```
staff crop (grayscale, 1×128×W, W=16k≤1280)
  → ResNetV2 backbone [2,3,7]  (StdConv2dSame + GroupNorm32 + ReLU, 16× downsample)
                                                          → (B, 1024, 8, W/16)
  → HybridEmbed proj  Conv2d(1024→256, k1, bias)          → (B, 8·W/16, 256)
  → +cls token, +pos_embed[crop-indexed], 4× ViT block, LayerNorm
                                                          → context (B, 1+8·W/16, 256)
  → AR decoder (4× [self-attn, cross-attn, GEGLU-ff], dim 256)
  → shared hidden → 4 parallel Linear heads: rhythm(260) pitch(71) lift(7) note(2)
  → 3 aligned token streams (rhythm/pitch/lift; note head UNUSED at inference, §5)
  → merge → extended-PrIMuS semantic string (§8) → MusicXML / **kern (E7)
```

Param split (measured from the checkpoint, §12): ResNetV2 backbone **9.30 M** (43.2%), ViT
blocks+norm **3.16 M** (14.7%), cls/pos/proj **0.43 M**, decoder attention GEMMs **5.24 M**
(24.3%), decoder FF GEMMs **3.15 M** (14.6%), embeddings 152 K, heads 87 K, norms/biases 16 K.

---

## 2. Vision encoder — `encoder.py::get_encoder` (timm 0.6.5 hybrid ViT)

### 2a. ResNetV2 backbone (`timm.models.resnetv2.ResNetV2`)

Constructed as `ResNetV2(layers=[2,3,7], num_classes=0, global_pool='', in_chans=1,
preact=False, stem_type='same', conv_layer=StdConv2dSame)`. All facts below read from
timm 0.6.5 `resnetv2.py` / `layers/std_conv.py` / `layers/norm_act.py` / `layers/padding.py`.

| field | value |
|---|---|
| input | (B, **1**, 128, W) — single channel (config `channels: 1`) |
| stem | `StdConv2dSame(1→64, k7, s2, bias=False)` → `GroupNormAct(64, groups=32, eps=1e-5)`+ReLU → `MaxPool2dSame(k3, s2)` |
| stages | [2, 3, 7] post-act **Bottleneck** blocks; widths 256 / 512 / 1024 (mid 64 / 128 / 256, `bottle_ratio=0.25`) |
| stage strides | stage0 **1**, stage1 **2**, stage2 **2** → total 16× (stem 4× · stages 4×) |
| block | conv1 1×1 → GN+ReLU → conv2 3×3 (stride on block 0) → GN+ReLU → conv3 1×1 → GN(**no act**) → +shortcut → ReLU |
| downsample | block 0 of each stage: `Conv2d 1×1 (stride)` → GN(**no act**); other blocks identity |
| final norm | **Identity** (preact=False ⇒ `self.norm = nn.Identity()`, confirmed absent from checkpoint) |
| conv bias | **none anywhere in the backbone** (confirmed: zero `.conv*.bias` keys in §12) |
| output | (B, 1024, 8, W/16) |

**Weight standardization (`StdConv2dSame`, std_conv.py:50):** every backbone conv
standardizes its weight **per output channel** at forward time:
`w' = (w - mean) / sqrt(var + 1e-6)` with mean/var over `(in·kh·kw)` and **population
(biased) variance** (implemented upstream as `F.batch_norm(training=True, momentum=0)`).
Deterministic function of the weights only ⇒ **fold into the stored weights at convert time
(E2)** — the runtime kernel is a plain conv. The fold must be bit-proven at L1 (§14).

**TF-'SAME' padding (padding.py):** `padding='SAME'` resolves *statically* for stride-1
convs (3×3 s1 → symmetric pad 1; 1×1 → pad 0) and *dynamically* (asymmetric, right/bottom
gets the extra) for stride-2 ops: `pad_total = max((ceil(i/s)−1)·s + k − i, 0)`, split
`left = total//2, right = total − left`. Applies to the stem 7×7 s2 (H=128 → pad_h 5 =
2 top + 3 bottom), the s2 3×3 convs (stage1/2 block 0), and the s2 3×3 **max-pool** (pads
with −∞). The s2 1×1 downsample convs resolve to pad 0 for every size. This asymmetric-pad
conv + same-pad max-pool is **NEW A8-adjacent kernel work** (§10) — `nn::conv2d` consumes
pre-padded NCHW, so a pad-then-conv wrapper is correct first, fused same-pad im2col second.

**GroupNorm:** `GroupNormAct(num_groups=32, eps=1e-5)` + ReLU (norm3/downsample norms skip
the act). 32 groups even at 64 channels (groups of 2). **No GroupNorm kernel exists in the
repo today** (Baidu/GOT are LayerNorm/RMSNorm only) — NEW (A8/E3).

### 2b. Hybrid patch embed + custom ViT (`encoder.py::CustomVisionTransformer`)

- `HybridEmbed` (timm): backbone → `proj = Conv2d(1024→256, k1, s1, bias=True)` → flatten →
  (B, 8·W/16, 256). `min_patch_size = 2^(len[2,3,7]+1) = 16`; config `patch_size: 16` ⇒
  HybridEmbed internal patch_size 1 (pure 1×1 projection, no second patching).
- **cls token** prepended; **pos embed (1, 641, 256)** = 1 + 8·80 grid (128/16 × 1280/16).
- **Crop-indexed positions** (encoder.py:23): a W-wide image (w = W/16 ≤ 80 patch columns)
  takes the **top-left w columns of each of the 8 rows** of the full 80-wide table: token
  (r, c) gets `pos_embed[1 + r·80 + c]`, cls gets `pos_embed[0]`. **W > 1280 ⇒ w > 80 ⇒ the
  index arithmetic goes negative and aliases via Python negative indexing — undefined
  upstream behavior.** The front-end MUST guarantee W ≤ 1280 (§7, OQ-T5).
- 4 × timm ViT `Block` (pre-norm): `LayerNorm(256, eps=1e-6)` → MHA **8 heads, head_dim 32,
  scale 32^-0.5**, fused `qkv = Linear(256→768, bias=True)`, `proj = Linear(256→256, bias=True)`
  → residual; `LayerNorm(eps=1e-6)` → MLP `fc1 256→1024` → **GELU (exact erf)** → `fc2
  1024→256` → residual.
- Final `LayerNorm(256, eps=1e-6)`; `num_classes=0, global_pool=""` ⇒ head is identity.
- Output context = **(B, 1+8·W/16, 256) — the cls token IS part of the cross-attn context.**

---

## 3. Connector — none. `emb_dim == dim == 256` ⇒ `project_emb = nn.Identity()` (decoder.py:31); the decoder consumes the encoder sequence directly as cross-attention K/V source.

## 4. Language decoder — `decoder.py` over x-transformers 0.29.2 (pinned; all internals below read from that exact source)

`Decoder(dim=256, depth=4, heads=8, attn_on_attn=True, cross_attend=True, ff_glu=True,
rel_pos_bias=False, use_scalenorm=False)` wrapped by `ScoreTransformerWrapper`.

**Input embedding (decoder.py:62):** at position t,
`x_t = rhythm_emb[r_t] + pitch_emb[p_t] + lift_emb[l_t] + pos_emb[t] · 256^-0.5`
— three summed token embeddings (260/71/7 × 256) plus a **learned absolute positional
embedding (256 positions × 256) scaled by dim^-0.5 = 1/16** (x_transformers.py:126). No RoPE
anywhere. The **note head has no input embedding** (output-only).

**Layer stack:** `layer_types = ('a','c','f') × 4` → a flat list of 12 sublayers
(checkpoint indices `layers.0..11`, `i%3` ⇒ 0=self-attn, 1=cross-attn, 2=ff). Each sublayer
is pre-norm + residual: `x = x + block(LayerNorm(x))`, with `LayerNorm(256, eps=1e-5 —
torch default; x-transformers passes no eps)`. Checkpoint key `layers.{i}.0.0.*` is the
pre-branch norm (norms are a 3-slot ModuleList `[pre, post_branch=None, post_main=None]`).

| sublayer | exact shape/semantics (x_transformers.py:489-724) |
|---|---|
| 'a' self-attn | causal; `to_q/to_k/to_v = Linear(256→512, bias=False)`; **8 heads × head_dim 64 (inner 512 ≠ dim 256)**; scale 64^-0.5 = **1/8**; `stable_softmax` (max-subtract); out proj = **`Linear(512→512, bias=False)` + `nn.GLU`** (`on_attn`: split 512→2×256, `a·σ(b)`) |
| 'c' cross-attn | identical shapes; K/V from the 256-dim encoder context (non-causal); query mask = the decoder padding mask (all-true at inference), context mask all-true |
| 'f' feed-forward | **GEGLU**: `proj = Linear(256→2048, bias=True)` → chunk 2×1024 → `x · GELU(gate)` → `Linear(1024→256, bias=True)` (checkpoint `net.0.proj` / `net.3`; net.1=Identity, net.2=Dropout) |

**Final norm + heads (decoder.py:33-39):** wrapper `LayerNorm(256, eps=1e-5)` → four
parallel `Linear(256→{7,71,260,2}, bias=True)` = `to_logits_{lift,pitch,rhythm,note}` — all
four applied to the SAME normed hidden every step.

**No KV cache upstream:** `generate` re-forwards the whole prefix each step and windows
inputs to the last 256 positions (`[:, -max_seq_len:]`). Our port adds a growing causal KV
cache + **cross-attn K/V computed once per staff** (context is fixed) — both bit-provable
against full re-forward at f32.

**Dead code to NOT replicate:** `select_hiddens = hiddens[0][3]` (decoder.py:65) is unused;
`decoder.note_mask` (260, marks rhythm ids 129–146) feeds only the training-time consistency
loss `calConsistencyLoss` (γ=10, L1 between note-head prob and the note-mass of each other
head) — **inference never uses it**. `min_p_pow/min_p_ratio` args: unused upstream.

**Naming-swap trap (do not "fix"):** `ScoreDecoder.generate` returns `(rhythm, pitch, lift)`;
`tromr_arch.generate` unpacks into MISNAMED locals `out_lift, out_pitch, out_rhythm` but
returns them positionally unchanged, and `staff2score.predict_img2token` unpacks
`rhythm, pitch, lift = output` — the double swap cancels; positional order is rhythm, pitch,
lift end-to-end.

---

## 5. Generation — `ScoreDecoder.generate` + `staff2score.py`

- Seeds: rhythm = `[BOS]`=1; pitch = lift = `nonote`=0 (config `bos_token: 1`,
  `nonote_token: 0`). Padding mask all-true, extended each step.
- Per step, per head **independently**: top-k filter `thres=0.9` ⇒ `k = ceil(0.1·V)` —
  rhythm k=26, pitch k=8, **lift k=1 (de-facto argmax)** — then
  `softmax(logits/T)` with **T = 0.2** (`staff2score.py` `args.get('temperature', .2)`;
  config.yaml has no key; the 0.25 default in `tromr_arch.generate` is shadowed) and
  `torch.multinomial` **sampling**. The note head's logits are computed and **discarded**.
- Stop: all batch rows have emitted rhythm `[EOS]`=2 (`cumsum ≥ 1`), or 256 steps.
- **Port decision (doctrine #8):** reference decode is *nondeterministic by construction*.
  Our default = per-head **argmax** (deterministic); the L4 gate runs argmax on BOTH sides
  (oracle patched to argmax). Sampled mode only behind a kill-switch env
  (`FOCR_TROMR_SAMPLE=1`), divergence measured and logged in `docs/DISCREPANCIES.md`.
- Batch note: upstream `torch.cat` of per-image tensors requires equal W — mixed-width
  batching is broken upstream; we run staves sequentially (doctrine #5), so no delta.

## 6. Preprocessing — `staff2score.py::readimg` (byte-exact pin for L0)

1. `cv2.imread(IMREAD_UNCHANGED)`. **RGBA input: `img = 255 − alpha`** (inverted alpha
   channel used as the grayscale ink image — rendered-PNG convention) → GRAY2RGB. BGR input
   → RGB. Other channel counts: error.
2. Resize: `new_h = 128`, `new_w = floor((128/h)·w)` **floored to a multiple of 16**;
   `cv2.resize` default **INTER_LINEAR** (half-pixel-center bilinear). No clamp at 1280 —
   the front-end must enforce it (§2b, §7).
3. Albumentations (1.2.0): `ToGray` (cv2 RGB2GRAY fixed-point luma ≈ 0.299R+0.587G+0.114B,
   rounded to **uint8 before normalize**, replicated ×3) → `Normalize(mean=0.7931,
   std=0.1738, max_pixel_value=255)` ⇒ `(px − 0.7931·255)/(0.1738·255)` → CHW, keep
   channel 0 only → (1, 128, W) f32.

L0b gate: dump reference tensors for a fixture set (RGB, RGBA, odd sizes, near-1280) and
match ≤ 1e-4 abs; the one known sub-L0 risk is our bilinear-resize vs cv2 INTER_LINEAR
fixed-point (mirror of the GOT CatmullRom note) — measure, then set the tolerance.

## 7. Front-end — staff detection / dewarp (E5; NOT in the upstream repo)

The model consumes ONE cropped staff. The paper's full-page system ("SDM") uses a
**semantic-segmentation net** for staff masks + a horizontal-projection bounding-box step —
the segmentation net is exactly the oemer-style dependency sub-epic E rejected. Scope for a
**library-free pure-Rust classical pipeline** (v1 = printed/scanned pages):

1. Grayscale → adaptive binarization (Sauvola windows; we have pure-Rust image ops precedent
   in `src/pdf.rs`/`src/preprocess`).
2. Estimate `staffline_height` / `staffspace_height` = the two modes of the vertical
   black/white run-length histograms (the classical OMR estimator).
3. Deskew: global rotation search (±5°, maximize horizontal-projection variance).
4. Staff-line rows = horizontal-projection peaks ≥ k·width; group into staves = runs of 5
   peaks equally spaced within tolerance of `staffspace_height`.
5. Crop each staff with ±3·staffspace margin (ledger lines), order top-to-bottom.
6. **Aspect guard:** after 128-height normalize, enforce W ≤ 1280 (aspect ≤ 10:1). Wider
   staves: split at barline candidates (vertical-projection minima) into overlapping chunks
   — upstream behavior above 80 patch columns is undefined (§2b) so this is a hard clamp,
   recorded in `docs/DISCREPANCIES.md`.
7. v2 (deferred, separate bead): camera-photo dewarp (per-column staffline y-offsets →
   smooth polynomial remap) for CMSD-like inputs; v1 covers MSD-like printed/scanned pages.
   Staff-detect RECALL is measured separately from model accuracy (E8, bead text).

## 8. Output semantics — extended-PrIMuS "semantic" streams (NOT **kern, NOT ABC)

The model's native output is three aligned WordLevel streams (§9), i.e. the PrIMuS-style
*semantic encoding* extended for polyphony — not **kern and not ABC. The reference merge
(`inference.py`) builds one string: tokens joined by `+`; a rhythm token `|` joins
simultaneous notes into a chord (**bottom-to-top**, per the paper); a note position renders
as `<pitch><lift?>_<duration>` (e.g. `note-F4#_whole` — accidental appended AFTER the
octave; lift attaches only if ∈ {lift_##, lift_#, lift_bb, lift_b, lift_N}); non-note rhythm
tokens (clef/keySignature/timeSignature/barline/rest/multirest) pass through.

Detokenize (staff2score.py:63): ids → tokens per stream; `[BOS]/[EOS]/[PAD]` entries deleted
**anywhere in the stream** (only the rhythm vocab has them); pitch/lift keep `nonote`
placeholders for alignment. Port rule: align by original index, strip trailing EOS, treat
mid-stream control ids as decode errors (fail loud), don't replicate the
delete-anywhere loop.

**E7 decision:** primary export = **MusicXML** (partwise; `keySignature-XM` → fifths;
clef sign+line from `clef-{G,F,C}{1..5}`; `<chord/>` for `|` groups; durations from the
rhythm duration table incl. dotted `.` variants; `multirest-N` → N measure rests) — the
interop winner and what OMR evaluation tooling consumes. `**kern` secondary (consistent
with GOT's `--format` sheet-music output) behind `--emit kern`, later bead. The raw merged
semantic string ships in `--json` as the model-native field.

## 9. Tokenizers — 4 trivial WordLevel tables (HF tokenizers JSON, committed upstream)

`tromr/workspace/tokenizers/tokenizer_{rhythm,pitch,lift,note}.json` — **WordLevel** (no
merges, no normalizer; pre-tokenizer split on `+`/newline, rhythm additionally isolates
`|`). Inference needs only the id→token tables + 3 special ids; encode is training/fixture-
side only. Conformance = table equality + detokenize goldens (a far lower bar than the GOT
tiktoken work; the `Tokenizer` trait impl in bd-3jo6.1.6 needs a decode-only WordLevel
variant).

| head | vocab | layout (exact ids) |
|---|---|---|
| rhythm | **260** | 0 `[PAD]`, 1 `[BOS]`, 2 `[EOS]`, 3 `+`, 4 `|`, 5 `barline`, 6–15 `clef-{C1..C5,F3..F5,G1,G2}`, 16–30 `keySignature-{AM..GbM}` (15 majors), 31–128 `multirest-N` (98 lexicographic), **129–146 `note-<duration>`** (breve, breve., eighth, eighth., half, half., hundred_twenty_eighth, long, quarter, quarter., sixteenth, sixteenth., sixty_fourth, sixty_fourth., thirty_second, thirty_second., whole, whole. — the `noteindexes`/note_mask set), 147–165 `rest-<duration>`, 166–259 `timeSignature-*` (incl. `C`, `C/`) |
| pitch | **71** | 0 `nonote`, 1–70 `note-{C,D,E,F,G,A,B}{0..9}` (diatonic letters only; octave-major order C0..B9 — accidentals live in the lift stream) |
| lift | **7** | 0 `nonote`, 1 `lift_null`, 2 `lift_##`, 3 `lift_#`, 4 `lift_bb`, 5 `lift_b`, 6 `lift_N` |
| note | **2** | 0 `nonote`, 1 `note` (output-only head; train-time consistency target) |

Config anchors (`config.yaml`): `pad_token: 0, bos_token: 1, eos_token: 2, nonote_token: 0,
max_seq_len: 256, num_rhythmtoken: 260, noteindexes: [129..146]`.

---

## 10. Reuse map → franken_ocr kernels (HONEST: worst fit in the zoo)

Survey verdict restated: ViT-encoder + AR-decoder = GOOD fit; CNN = POOR. TrOMR is **half
CNN by parameter count** (9.3 M ResNetV2 backbone, 43%) and its transformer half uses none
of the Qwen2 conventions. This is the price of the only Apache-2.0 polyphonic OMR model at
this size; it is still a transformer AR decode at heart, and the model is tiny enough
(21.5 M) that f32-at-peak already clears any reasonable latency bar.

**Reuse as-is (micro-kernel level):**
- `simd::igemm_s8s8` + per-row quant — IF int8 is measured-lossless (E2); worst decoder
  K = 1024 (`net.3`) ⇒ i32 accumulation worst case 1024·127·127 ≈ 16.5 M ≪ i32::MAX
  (trivially inside the proven K=6848 bound; add `KCase{k:1024}` to
  `tests/int32_overflow_proof.rs` anyway per doctrine #6).
- `nn::layer_norm` (vision f32 LayerNorm — both eps values are parameters: **1e-6 encoder,
  1e-5 decoder**, §2b/§4), `nn::gelu` (exact erf — SAM GELU, NOT `quick_gelu`), softmax,
  `nn::conv2d` (f32 direct conv over pre-padded NCHW), the GEMM-attention pattern from
  `vision_sam.rs`, embedding gather.
- Infra: `.focrq` v2 container, manifest/`focr pull`, robot/CLI surfaces, parity-harness
  scaffolding (`scripts/gen_reference_fixtures.py` grows a TrOMR branch).

**NEW — A8-adjacent conv work (called out per the bead):**
1. **GroupNorm(32, eps 1e-5)+ReLU kernel** — first GroupNorm in the repo.
2. **TF-'SAME' dynamic asymmetric padding** for s2 convs + the **−∞-padded s2 max-pool**
   (§2a) — pad-then-`nn::conv2d` wrapper first (correctness), fused same-pad im2col second.
3. **Weight-standardization fold at convert time** (E2, population variance, eps 1e-6) with
   an L1 bit-proof folded-vs-reference — runtime sees plain convs, so NO WS kernel.
4. Bottleneck residual driver (12 blocks; spatial is small — 8×80 at stage 2 for a
   full-width staff — full-core-parallel f32/bf16 per bd-3jo6.1.8 is sufficient).

**A7 NON-FIT delta (exact list — why TrOMR does NOT ride the shared dense decoder driver):**
| A7 assumption (Qwen2 dense) | TrOMR reality |
|---|---|
| RMSNorm | LayerNorm with bias (eps 1e-5) |
| RoPE | learned absolute pos embedding × dim^-0.5, added at the INPUT only |
| one `embed_tokens` | **sum of 3 embeddings** (+pos) per step |
| one `lm_head` | **4 parallel heads off one hidden**; per-step per-head argmax; streams appended independently; stop condition on the rhythm stream only |
| self-attn only | **cross-attention every layer** (encoder-decoder) — in NO current decoder path |
| o_proj Linear | `Linear(512→512, no bias)` + **GLU** gated output (`on_attn`) |
| SwiGLU (x·silu(g), no bias) | **GEGLU** (x·gelu(g), **with bias**) — same GLU family; activation+bias must be parameters if shared |
| inner attn dim == hidden | inner 512 ≠ hidden 256 (head_dim 64 × 8) |
| tied embeddings | untied; per-head Linear+bias |

**Verdict:** build a small self-contained `tromr.rs` forward (vision_sam.rs-style: explicit
graph over shared micro-kernels), NOT a parameterization of `decoder_qwen2.rs`. A7 itself
needs **zero changes** for E — the sub-epic's "E4 wires the decoder (A7)" premise is
corrected to "E4 uses the kernel layer under A7" (§15 DAG delta). Perf honesty: at dim 256
the per-call overhead of dynamic int8 quantization is a real fraction of a 256-wide GEMV;
**f32/bf16-at-peak is the default plan**, int8 only if E2 measures both losslessness AND a
speedup on M4 + AVX2 (record either way in `docs/PERF_LEDGER.md` /
`docs/NEGATIVE_EVIDENCE.md`).

## 11. Conversion / quant plan (E2)

- **Source artifact problem:** upstream ships a torch-pickle `.pth` (no safetensors, no HF
  hosting). Decision: the offline converter (Python, oracle venv) exports
  `model.safetensors` once; `focr convert` consumes safetensors as for GOT (keeps the Rust
  surface pickle-free); end users never convert — we **redistribute the converted `.focrq`
  via `focr pull`** (Apache-2.0 permits; ship the NetEase NOTICE in the manifest +
  provenance surfaces). Model id `tromr`, artifact `tromr.focrq` (~86 MB f32 / ~43 MB bf16
  — single part, no split needed).
- **Quant policy (doctrine #2 applied):** high-precision (bf16/f32) = the ENTIRE encoder
  (CNN backbone + ViT — vision stays HP, do-not-retry), all embeddings + pos tables, all
  norms (GN + LN), all biases, the 4 heads. int8 candidates = ONLY the decoder-sublayer
  GEMM weights (40 matrices: `to_{q,k,v}` + `to_out.0` ×8 attn sublayers, `net.0.proj` +
  `net.3` ×4 ff sublayers — 8.4 M params, 33.5 MB f32), gated on a measured-lossless
  L4/L5 check per E2's bead text; default f32.
- **Convert-time transforms:** WS fold (§10.3); keep `note_mask` OUT of the artifact
  (train-only); store the 4 vocab tables + special ids + preprocess constants
  (mean/std/128/1280/16) in the `.focrq` v2 arch metadata, per the A3 per-arch policy.

## 12. Exact tensor inventory (extracted from `img2score_epoch47.pth`, 2026-07-01)

261 tensors, **21,534,232 params**, fp32. Full key list archived; the complete shape map:

- **Backbone** (`encoder.patch_embed.backbone.`): `stem.conv.weight` [64,1,7,7];
  `stem.norm.{weight,bias}` [64]; stages `{0,1,2}` with {2,3,7} blocks:
  `stages.{s}.blocks.{b}.conv1.weight` [mid,in,1,1], `.conv2.weight` [mid,mid,3,3],
  `.conv3.weight` [out,mid,1,1], `.norm{1,2,3}.{weight,bias}`, block 0 only
  `.downsample.conv.weight` [out,in,1,1] + `.downsample.norm.{weight,bias}`
  (mid/out = 64/256, 128/512, 256/1024). NO conv biases, NO final backbone norm.
- **Patch proj**: `encoder.patch_embed.proj.{weight [256,1024,1,1], bias [256]}`.
- **ViT**: `encoder.cls_token` [1,1,256]; `encoder.pos_embed` [1,**641**,256]; ×4
  `encoder.blocks.{i}`: `norm1.{w,b}` [256], `attn.qkv.{weight [768,256], bias [768]}`,
  `attn.proj.{weight [256,256], bias [256]}`, `norm2.{w,b}`, `mlp.fc1.{weight [1024,256],
  bias}`, `mlp.fc2.{weight [256,1024], bias}`; `encoder.norm.{weight,bias}` [256].
- **Decoder** (`decoder.net.`): `lift_emb.emb.weight` [7,256]; `pitch_emb.emb.weight`
  [71,256]; `rhythm_emb.emb.weight` [260,256]; `pos_emb.emb.weight` [256,256]; flat
  `attn_layers.layers.{0..11}` with `i%3∈{0,1}` (attention):
  `layers.{i}.0.0.{weight,bias}` [256] (pre-LN), `layers.{i}.1.to_{q,k,v}.weight`
  [512,256] (**no bias**), `layers.{i}.1.to_out.0.weight` [512,512] (**no bias** —
  `on_attn` Linear is bias=False, GLU has no params); `i%3==2` (ff):
  `layers.{i}.0.0.{weight,bias}`, `layers.{i}.1.net.0.proj.{weight [2048,256], bias
  [2048]}`, `layers.{i}.1.net.3.{weight [256,1024], bias [256]}`; `norm.{weight,bias}`
  [256]; heads `to_logits_{lift [7,256], pitch [71,256], rhythm [260,256], note [2,256]}`
  + biases. Plus train-only `decoder.note_mask` [260].

## 13. Reference oracle (E8 prerequisite)

`requirements.txt` pins `torch==1.11.0, timm==0.6.5, x-transformers==0.29.2,
transformers==4.20.1, albumentations==1.2.0, einops==0.4.1, opencv-python, omegaconf`.
torch 1.11 has no reliable Apple-Silicon wheel → oracle venv = python 3.10 + **torch 2.x
CPU** + the OTHER pins exact (timm 0.6.5 and x-transformers 0.29.2 are the load-bearing
pins — they freeze the module graph; `strict=True` load is the guard). **OQ-T1** requires
proving torch-2.x logits ≡ torch-1.11 logits (or documenting the delta) on one x86 host
where both install, before trusting the oracle. Establish the oracle's nondeterminism floor
(4 runs × 2 thread counts) BEFORE setting L3 tolerances (doctrine #8); patch `generate` to
argmax for L4 (§5). The repo's `examples/{1..4}.{png,txt}` + `photo{1..4}.jpg` are free
committed ground truth for L5 seeds; the specialized-corpus gap (GOT_NEXT_STEPS §2, bead
bd-3kix) extends to staves — synthesize via Verovio/MuseScore renders with known encodings.

## 14. Parity ladder L0–L5 (E8, mirrors B8)

- **L0a** tokenizer: 4-table equality vs the committed JSONs + detokenize/merge goldens
  (`examples/*.txt`) — id-exact, zero mismatches.
- **L0b** preprocess: fixture tensors ≤ 1e-4 (resize-kernel divergence measured first, §6);
  includes the RGBA-inversion path + width-floor-to-16 edge cases.
- **L0c** decode seeding: (BOS,nonote,nonote) + mask semantics id-exact.
- **L1** per-op: WS-fold bit-proof; GN; same-pad conv (asymmetric cases!); max-pool −∞ pad;
  bottleneck; ViT block; GEGLU; on_attn GLU; pos-emb crop indexing (narrow widths) — cos ≥
  1−1e-6 each.
- **L2** per-seam: backbone-out → proj-out → each ViT block → encoder-out; teacher-forced
  decoder per-sublayer hiddens; the 4 head logits.
- **L3** logits within the measured oracle floor (f32 and any int8 variant tracked
  separately).
- **L4** greedy(argmax)-token-exact per stream to first divergence, teacher-forced AND
  free-running; KV-cache path proven ≡ re-forward path first.
- **L5** e2e: per-head SER + merged-string SER vs oracle on identical crops within a
  documented budget (paper anchor: merged SER 0.025 on MSD-polyphonic); music-structural
  metric (note-level F1 or music-TEDS) on the synthesized corpus; staff-detect recall
  reported separately (E5). Model-gated skip-with-SUCCESS without weights; NDJSON logging
  throughout (bead-mandated test shape).

## 15. Task-DAG delta for E2–E10 (census-driven corrections)

- **E4 (`bd-3jo6.5.4`) ⇢ A7 (`bd-3jo6.1.7`) dependency SHOULD BE RELAXED**: E4 does not
  consume the fused Qwen2 driver and needs none of the GQA/RoPE work (§10 non-fit table);
  it depends only on the micro-kernel layer (already shipped) + A8 conv kernels via E3.
  E4 can start as soon as E1+E2 land. (Bead edit — tracked as a follow-up, not run here.)
- **E3 (`bd-3jo6.5.3`) / A8 (`bd-3jo6.1.8`) scope precision**: A8's "TrOMR ResNet stem"
  line item expands to: GroupNorm32 kernel, TF-SAME asymmetric pad conv path, −∞ same-pad
  max-pool, bottleneck driver. The WS **fold** belongs to **E2** (convert-time), not A8
  (runtime) — no WS kernel exists at runtime.
- **E2 (`bd-3jo6.5.2`)**: + the `.pth`→safetensors offline export step (§11); + WS fold with
  L1 bit-proof; + manifest/NOTICE redistribution decision; int8 is opt-in behind a measured
  gate, f32/bf16 default (bead text already agrees).
- **E5 (`bd-3jo6.5.5`)**: scope pinned to §7 v1 (printed/scanned, classical CV, global
  deskew only); **NEW follow-up beads needed**: (a) camera dewarp v2, (b) barline-split
  chunking for staves wider than 10:1.
- **E6 (`bd-3jo6.5.6`)**: shrinks to a decode-only WordLevel table + specials + merge rules
  (§9); feeds the Tokenizer trait (bd-3jo6.1.6) a fourth, trivial variant.
- **E7 (`bd-3jo6.5.7`)**: format decided — MusicXML primary, `**kern` a follow-up bead
  behind `--emit kern`; raw semantic string exposed in `--json`.
- **E8 (`bd-3jo6.5.8`)**: oracle plan §13 (torch-2.x proof = OQ-T1 gate); argmax-forced L4;
  SER budgets anchored to the paper's 0.025 merged SER.
- **E9 (`bd-3jo6.5.9`)**: unchanged; multi-staff JSON = ordered staves[] with page bboxes
  from E5.
- **E10 (`bd-3jo6.5.10`)**: unchanged; add the §14 fixture inventory.
- **Cross-sub-epic synergy**: the encoder (§2) is byte-for-byte the **pix2tex / LaTeX-OCR
  hybrid backbone** (`encoder.py` is copied from that project; same [2,3,7] ResNetV2 +
  ViT recipe) — E3's kernels double-serve the opportunistic pix2tex bead, and the
  specialized-corpus work (bd-3kix) should include staves for BOTH GOT `--format` **kern
  mode and TrOMR.

## 16. Open questions (doctrine hard rule — no kernel ships against an unresolved OQ)

- **OQ-T1** oracle torch version: prove torch-2.x ≡ torch-1.11 logits for this graph (or
  pin the delta) before the parity ladder is trusted. §13.
- **OQ-T2** cv2 `INTER_LINEAR` resize parity: measure our bilinear vs cv2 fixed-point on
  the fixture set; fold into the L0b tolerance (mirror of GOT's CatmullRom note).
- **OQ-T3** albumentations 1.2.0 `ToGray` exact arithmetic (cv2 fixed-point luma
  coefficients + uint8 rounding BEFORE normalize) — pin from the albumentations source at
  E3 implementation time.
- **OQ-T4** `stable_softmax` (max-subtract) vs our softmax: confirm bitwise-benign in f32
  at L1 (expected yes; verify, don't assume).
- **OQ-T5** wide-staff policy: our 10:1 clamp + barline chunking is a documented divergence
  (upstream undefined, §2b) — needs a measured DISCREPANCIES entry once E5 exists.
- **OQ-T6** sampled-vs-argmax decode divergence rate on real staves (kill-switch
  `FOCR_TROMR_SAMPLE`) — measure for the DISCREPANCIES entry. §5.
- **RESOLVED by source this census:** `on_attn` out-proj is bias-free (x_transformers.py:577
  + checkpoint); decoder LN eps 1e-5 vs encoder 1e-6 (§4/§2b); note head inference-dead
  (§5); note_mask train-only; pos-emb scale 1/16 (§4); backbone final norm Identity;
  no conv biases; stage strides 1/2/2; GN eps 1e-5 groups 32; WS eps 1e-6 population
  variance; rhythm/pitch/lift positional-order swap benign (§4); vocab layouts (§9).

### Sources
- Code + config + tokenizers + checkpoint — https://github.com/NetEase/Polyphonic-TrOMR
  (`tromr/workspace/config.yaml`, `tromr/model/*.py`, `tromr/staff2score.py`,
  `tromr/inference.py`, `tromr/workspace/tokenizers/*.json`,
  `tromr/workspace/checkpoints/img2score_epoch47.pth`)
- Pinned library internals — `timm==0.6.5` (`models/resnetv2.py`, `models/layers/std_conv.py`,
  `models/layers/norm_act.py`, `models/layers/padding.py`, `models/vision_transformer.py`,
  `models/vision_transformer_hybrid.py`), `x-transformers==0.29.2` (`x_transformers.py`)
- Paper — https://arxiv.org/abs/2308.09370 (consistency loss λ=0.1/β=1.0 γ=10; SDM
  front-end; chords bottom-to-top; MSD/CMSD datasets; merged SER 0.025 anchor)
