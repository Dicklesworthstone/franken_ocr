# Forward-wiring intel (verified against the real checkpoint)

Reverse-engineered from `baidu/Unlimited-OCR` @ `3a7f4dbb…` (the 6.67 GB bf16
shard, 2710 tensors) and the pinned truth-pack modeling code. Captured so the
per-stage forward wiring uses the EXACT tensor names and avoids the known traps.

## You can wire + prove parity NOW — no `.focrq`, no converter
The single bf16 shard loads via `Weights::load`'s safetensors fallback
(`weights.rs` `from_safetensors_bytes`); `mat()`/`vec()` widen BF16→f32 at the
accessor boundary. So the `// NotImplemented until the .focrq reader (Phase 2)`
comments in the per-stage shims are **stale** — the accessor API already exists
and is unit-tested (`Weights::mat`/`vec`/`tensor`/`qint8`/`qint4`). Point the
loader at the raw `model-00001-of-000001.safetensors` and wire bodies.

## Canonical tensor names (EXACT — copy verbatim)
Decoder (12 layers): `model.embed_tokens.weight`, `model.norm.weight`,
`lm_head.weight` (**top-level**, NOT `model.lm_head`). Per layer L 0..11:
`model.layers.{L}.input_layernorm.weight`,
`model.layers.{L}.post_attention_layernorm.weight`,
`model.layers.{L}.self_attn.{q,k,v,o}_proj.weight` (**separate** q/k/v/o).
- Layer 0 (dense, first_k_dense_replace=1): `model.layers.0.mlp.{gate,up,down}_proj.weight` (interm 6848).
- Layers 1..11 (MoE): router `model.layers.{L}.mlp.gate.weight` [64,1280];
  experts `model.layers.{L}.mlp.experts.{0..63}.{gate,up,down}_proj.weight` (interm 896);
  shared `model.layers.{L}.mlp.shared_experts.{gate,up,down}_proj.weight`
  (**singular** `shared_experts`, fused 2×896=1792).
Projector: `model.projector.layers.weight` [1280,2048], `model.projector.layers.bias`.
Specials: `model.image_newline`, `model.view_seperator` (sic — "seperator").

SAM (`model.sam_model.`): `patch_embed.proj.{weight,bias}`, `pos_embed`; per
block 0..11: `blocks.{B}.norm1.{weight,bias}`, `blocks.{B}.norm2.{weight,bias}`,
`blocks.{B}.attn.qkv.{weight,bias}` (**fused** qkv), `blocks.{B}.attn.proj.{weight,bias}`,
`blocks.{B}.attn.rel_pos_h`, `blocks.{B}.attn.rel_pos_w`,
`blocks.{B}.mlp.lin1.{weight,bias}`, `blocks.{B}.mlp.lin2.{weight,bias}`.
Neck is IRREGULAR — exactly: `neck.0.weight`, `neck.1.weight`, `neck.1.bias`,
`neck.2.weight`, `neck.3.weight`, `neck.3.bias` (conv@0/2 no bias, LN2d@1/3 w+b),
then `net_2.weight`, `net_3.weight` (no `net_0`/`net_1`).

CLIP (`model.vision_model.`): `embeddings.class_embedding`,
`embeddings.patch_embedding.weight`, `embeddings.position_embedding.weight`,
`pre_layrnorm.{weight,bias}` (**typo "layrnorm" is REAL — preserve it**); per
transformer layer 0..23: `transformer.layers.{L}.layer_norm1.{weight,bias}`,
`transformer.layers.{L}.layer_norm2.{weight,bias}`,
`transformer.layers.{L}.self_attn.qkv_proj.{weight,bias}` (**fused**),
`transformer.layers.{L}.self_attn.out_proj.{weight,bias}`,
`transformer.layers.{L}.mlp.fc1.{weight,bias}`, `transformer.layers.{L}.mlp.fc2.{weight,bias}`.

## Vision data flow (base mode, from modeling_unlimitedocr.py)
`sam_out = sam_model(image_ori)` → `clip_out = vision_model(image_ori, sam_out)` →
`hybrid = cat(clip_out[:,1:], sam_out.flatten(2).permute(0,2,1), dim=-1)` [256,2048]
→ `projector(hybrid)` [256,1280]. (CLS dropped; SAM spatial flattened channel-major.)

## Decode KV-cache contract (the real remaining blocker)
`OcrModel::generate` is stateful-incremental: prefill calls
`decoder::forward(weights, full_prompt_embeds)`, then each step calls
`decoder::forward(weights, &single_token_embed)` expecting a persisted R-SWA ring
cache. The current `forward(&Weights,&Mat)` signature has nowhere to hold the 12
per-layer caches across calls — so correct incremental decode REQUIRES threading
`&mut [RingCache;12]` (+ absolute position_ids) through `decoder::forward` AND the
generate loop together (bd-1gv.17). A stateless full-sequence `forward` is correct
for PREFILL/parity but NOT for codex's incremental decode loop.

## Proof harness (scripts/baseline/, landed)
`run_baidu_reference.py` (CPU oracle), `dump_stage_activations.py` (per-stage
.npy fixtures), `compare_ocr.py` (CER). Reference for the 20 royalnavy02clow
pages already generating; validated correct on page 9.
