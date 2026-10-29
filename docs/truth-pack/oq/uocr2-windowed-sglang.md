# Unlimited-OCR-2 serving semantics: SGLang wheel spec (reference for franken_ocr)

## 0. Provenance and notation

* Wheel: `https://huggingface.co/baidu/Unlimited-OCR-2/resolve/5632329a32e103d5f1add270681e0f5c1233c6e4/wheel/sglang-0.5.20.dev10%2Bg8d62f5fb9c-py3-none-any.whl`
  sha256 `a3f3e2c94e09de90b61b43d01475fd5d1aa818d16eec05204b8079c09c2ba70d` (verified).
* Citation root. Every citation `W/<path>:<line>`, and every bare `<file>.py:<line>` below, is
  relative to the pinned wheel's `sglang/srt/` directory (unzip the wheel above). Line numbers are
  **dev10's**; section 5 lists what dev13 moved.
* Checkpoint config values come from the HF repo's `config.json` at the pinned commit (cited as
  `CFG`). The HF README is cited as `README`, and the HF modeling code as `HF/<file>`. Pins and
  SHA-256s: `docs/truth-pack/PINNED_SOURCES.md`, `docs/truth-pack/SOURCE_HASHES.md`.
* Nothing from the download was executed. Everything below was read with grep/sed/awk.

Checkpoint values that drive everything below (`CFG.sparse_attention_config`):

| key | value | used as |
|---|---|---|
| `sparse_block_size` | 32 | DASH block = KV page (page_size forced to 32) |
| `sparse_local_block` | 4 | local window = 4*32 = 128 tokens |
| `dash_topk_blocks` | 12 | remote read budget K |
| `sparse_topk_blocks` | 16 | buffer capacity: remote_blocks = 16-4 = 12 |
| `dash_infer_topm` | (absent) | TOPM_DEFAULT = 16 (output slot width only, see 4.3) |
| `dash_scale_per_layer` | 128.0 x 12 | idx_scale = 128/sqrt(128) = 11.3137 |
| `dash_floor_per_layer` | 1e-4 x 12 | prior floor |
| `dash_sigma` | 1.0 | bias divisor |
| `dash_stage0` | `head_cls` | block summary mode (the only one sglang implements) |
| `dash_image_only`, `image_bucket_constraint` | true, true | page-aware candidate mask on |
| `sparse_index_dim`, `sparse_num_index_heads` | 128, 1 | router dims |
| `sparse_rope_mode`, `sparse_main_rope_mode` | page2d, page2d | both branches |
| `sparse_rope_tok_theta`, `sparse_rope_page_theta` | 10000, 256 | |
| `sparse_rope_page_pairs`, `sparse_rope_page_mod` | 1, 2 | 63 tok freq pairs + 1 page pair; page axis folded mod 2 |
| `*_tok_freq_mode` | resample, resample | inv_tok[i] = theta^(-i/n_tok) |
| `page_token_id`, `image_token_id` | 128827 (`<PAGE>`), 128815 | `<PAGE>` is a non-special added token in the v2 tokenizer, so it is detokenized as literal text `<PAGE>` |
| `per_page_image_tokens` | 273 | fallback bucketing only when the prompt has no markers |
| `sparse_init_block` | 1 | **never read by the v2 serving backend** (no attention sink) |
| `eos_token_id`, `bos_token_id` | 1, 0 | |

---

## 1. Knobs and defaults (`W/environ.py:1352-1374,1423`)

| env | wheel default | README-recommended | meaning |
|---|---|---|---|
| `SGLANG_UOCR_EANABLE_DOC_LIMIT` (sic) | True | 1 | enable page-window splitting |
| `SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW` | **1** | 10 | pages a window *transcribes* (its lookahead page not counted, `W/environ.py:1353`) |
| `SGLANG_UOCR_DOC_BASE_PAGE_WINDOW` | **1** | 10 | same, for light modes (base/large/small/tiny) |
| `SGLANG_UOCR_DOC_LOOKAHEAD_PAGES` | 1 | n/a | pages a window reads past its own |
| `SGLANG_UOCR_DOC_LOOKAHEAD_KV_REUSE` | True | n/a | copy the lookahead page's KV into the next window instead of recomputing it |
| `SGLANG_UOCR_MAX_PAGE_TOKENS` | **0** (off) | 20000 | per-page output cap of the page guard. The comment says "8000 is the validated value" (`W/environ.py:1371-1374`) |
| `SGLANG_UOCR_NGRAM_OUTPUT_ONLY` | True | n/a | the n-gram domain is generated tokens only |
| `SGLANG_UOCR_DOC_PREFETCH` | True | n/a | performance only (vision-encodes window i+1 during window i) |

Mode → window limit: `page_limit_for_mode` returns the BASE limit for `{large, base, small, tiny}` and the GUNDAM limit for `gundam`, unknown, or absent modes (`W/entrypoints/openai/doc_split.py:333-343`). With `crop_mode=True` the mode is gundam, with False it is base (`W/entrypoints/openai/serving_chat.py:417-423`).

Forced server overrides for v2: `page_size = sparse_block_size`, backend `unlimited_ocr_v2`, and **radix cache disabled**, because a request resuming from a cached prefix would start its `<PAGE>`/image counters at zero (`W/arg_groups/model_overrides/unlimited_ocr_v2.py:24-52`).

Default context length: `min((W+L)*2034 + W*MAX_PAGE_TOKENS + 1024, 131072)`, where 2034 = 1+273+1760 is the costliest gundam page. It falls back to 131072 when splitting is off or the cap is 0 (`W/configs/model_config.py:297-327`). With the README settings it is 131072.

---

## 2. Prompt and output format

* Request (README): content = `[image_url x N] + [{"type":"text","text": "<PAGE><image>"*N + "Multi page parsing."}]`.
* The template `unlimited-ocr` has empty roles and separators (`W/parser/conversation.py:925-939`). Because the text already holds one `<image>` placeholder per image, no extra image token is inserted (`W/parser/conversation.py:670-679`). The prompt string is therefore exactly `<PAGE><image>...<PAGE><image>Multi page parsing.`
* Tokenization: BOS (0) is prepended (`W/configs/unlimited_ocr.py:475-476`). Each `<image>` expands to image-token ids: base (no crop) = (16+1)*16+1 = 273; gundam = 273 global + `(10*w_tiles + 1) * (10*h_tiles)` crop tokens when tiled (`W/configs/unlimited_ocr.py:445-465`). The trailing EOS that `tokenize_with_images` appends is stripped in inference mode (`W/configs/unlimited_ocr.py:503-507`).
* Canonical layout: `[BOS] ([<PAGE>] [img x n_p])_{p=1..N} [instruction tokens]`. `parse_prompt` (`W/session/uocr_doc_reuse.py:160-188`) enforces this: any non-image prefix, then a contiguous run of `<PAGE>`+image spans, then an instruction with no images.
* Output: `<PAGE>body_1<PAGE>body_2...<PAGE>body_N` then EOS. Each `<PAGE>` *opens* a page ("Markers are page prefixes", `W/sampling/custom_logit_processor.py:205-206`).

---

## 3a. Chunked ("Unlimited-Horizon") decoding as implemented

### 3a.0 Mapping to the paper's Algorithm 1 (`Unlimited-OCR-2.pdf` at the HF pin, sec. 3.4)

| Paper | Wheel |
|---|---|
| Chunks overlap by 1 boundary page | Windows are **contiguous and disjoint in ownership**; `OVERLAP_PAGES = 0` (`doc_split.py:22-25`). The 1-page overlap is a **lookahead**: window k's prompt also carries page `hi` (window k+1's first page), which it encodes but does not transcribe (`serving_chat.py:470-481,504-510`). |
| `C_overlap^img` | KV copied verbatim from window k: [marker of k's last own page] + [last 128..159 image tokens of that page] + [the whole lookahead page: its `<PAGE>` marker and all its image tokens] (`uocr_doc_reuse.py:234-300`) |
| `ENC(C_k \| C_overlap)` | prefill of window k+1's remaining own pages plus its own lookahead page, attending to the copied prefix |
| `C_tail^txt`, tail length T | up to **T = 128** trailing generated tokens of window k's last own page (after its last `<PAGE>`), KV copied verbatim (`TAIL_TOKENS = 128`, `uocr_doc_reuse.py:191-193,303-320`) |
| only the first chunk carries q | yes: the rewritten follow-up prompt drops both the BOS/prefix and the instruction (`uocr_doc_reuse.py:207-216,292`) |
| (not in paper) | a forced `<PAGE>` is the first decode input of every reused window (`uocr_doc_reuse.py:1078-1080,1296`) |
| inactive caches released | at window finish, everything except the four kept ranges is freed (sec. 3d) |

### 3a.1 Planning (`doc_split.plan_batches`, `doc_split.py:117-167`)

Planning only happens under all of these conditions: template `unlimited-ocr`, arch `UnlimitedOCRV2ForCausalLM`, `n == 1`, a positive `max_tokens`, `min_tokens == 0`, and no tools/logprobs/sessions/response_format (`serving_chat.py:374-415`).

```
per_page = {gundam:1514, large:422, base:274, small:112, tiny:74}[mode]   # estimate only (default 1514)
fits = (context_len - 64 - 1) // per_page;  fits = min(fits, window_limit)
if n_pages <= 1 or fits < 1 or overlap >= fits: unsplit
if fits >= n_pages and n_pages <= window_limit: unsplit (or one clamped batch)
batches: emit_from=0; loop: lo = max(0, emit_from-overlap) if emit_from else 0
         hi = min(lo+fits, n); append (lo, hi, emit_from); emit_from = hi
```
With overlap=0 this gives `(0,W,0), (W,2W,W), ...`.

* Lookahead of window bi = `min(LOOKAHEAD_PAGES, last_hi - hi)`. It is 0 for the last window (`serving_chat.py:470-481`).
* The child request for window bi keeps images `[lo, hi+lookahead)`, and its text is retagged to `"<PAGE><image>" * kept + tail_after_last_<image>`, so **every child request text still contains the instruction** (`doc_split.py:170-234`, `serving_chat.py:504-510`). The instruction only disappears when the scheduler rewrites the prompt for KV reuse (3a.4).
* `custom_params` per window (`serving_chat.py:511-532`): `uocr_doc_page_base = lo`, `uocr_doc_lookahead = lookahead`, `uocr_doc_id`, `uocr_doc_keep = bi < last`, `uocr_doc_reuse = bi > 0`, and `uocr_doc_overlap = emit_from - lo` (0 by default).
* Token budget: the request `max_tokens` is a **global** budget. Window bi gets `min(remaining, capacity)`, where `capacity = min(ctx - 64 - per_page*(own+lookahead), own*max_page_tokens + 256)` (`serving_chat.py:483-502,768`, `doc_split.py:64-114`). After each window, `remaining -= completion_tokens`, and forced/spliced tokens are not counted (`output_streamer.py:493-500`). Prefetch caps a follow-up's `max_new_tokens` to `budget - predecessor_used` (`uocr_doc_prefetch.py:113-122`).

### 3a.2 Window 0 (and any recomputed window)

* Prompt: `[BOS] <PAGE>img(lo) ... <PAGE>img(hi-1) [<PAGE>img(hi) lookahead] "Multi page parsing."`, with RoPE page base = `lo`.
* Page guard `expected = #<PAGE> in prompt - context_pages(0) - lookahead = own pages` (`custom_logit_processor.py:232-237`).
* Termination of a non-final window: when the model emits the `<PAGE>` that would open the lookahead page, `marks > expected` and the next sampled token is forced to EOS (`custom_logit_processor.py:250-251`). The window "ends on the marker past its own pages". `<PAGE>` is **not** masked on the last own page for such windows (`end_on_marker`, `custom_logit_processor.py:225-231,258-262`). A natural EOS on the last own page is also accepted.

### 3a.3 Retention at the end of window k (`on_cache_finished_req`, `uocr_doc_reuse.py:823-900`)

Retention happens only if `uocr_doc_keep` is set, the window finished by a matched stop token or string (EOS), was not aborted or retracted, and still holds KV. Otherwise nothing is retained and window k+1 recomputes.

1. `out = output_ids[:finished_len]`, with trailing stop ids popped. If lookahead > 0 or end_on_marker, `out` is cut at the first marker beyond `own_pages`; the extra marker and anything after it are dropped (`_cut_at_extra_marker`, `uocr_doc_reuse.py:807-820,862-870`). `out` is then limited to tokens whose KV exists (`< kv_committed_len`).
2. `next_page = page_base + overlap + (#markers in prompt - context_pages - lookahead)`, recorded so the follow-up can check continuity (`uocr_doc_reuse.py:873-883,1217-1234`).
3. `_trim_to_tails` keeps only these ranges of the KV row, each widened to whole 32-token pages, and frees the rest immediately (`uocr_doc_reuse.py:323-367,903-938`):
   * `[page_start(last own), +1)`: the `<PAGE>` marker of the last own page
   * `[E - 128 - E%32, E)` with `E = page_end(last own)` (one past its last image token): the **image tail**, 128..159 tokens starting on a block boundary
   * the whole lookahead page `[page_start(la), page_end(la))` (only if LOOKAHEAD_KV_REUSE)
   * the **text tail**: `text_tail(out, 128)` = `out[max(last_marker+1, len-128):]`. That is up to 128 trailing tokens of the last own page's body, never including a marker (`uocr_doc_reuse.py:303-320`).

### 3a.4 Window k+1 with reuse (`try_match_prefix` → `_build_reuse`, `uocr_doc_reuse.py:941-1161`)

Preconditions: a retained entry exists, `entry.next_page == lo` (otherwise "recomputing", `uocr_doc_reuse.py:1049-1057`), `plan_tail_copy` succeeds, the text tail exists and `len(text)+2 < max_new_tokens`, and there is pool room.

`plan_tail_copy(old_ids, new_ids, overlap=0, ..., old_lookahead, copy_lookahead=True)` (`uocr_doc_reuse.py:234-300`):
```
p_old = #pages(old) - old_lookahead;  E = old.page_ends[p_old-1]
start = E - 128 - E % 32            # must be > old.page_starts[p_old-1] (else no reuse)
copy_end, kept_from = E, new.prefix_len
if copy_lookahead and old_lookahead>0 and overlap==0 and old lookahead page == new first page (ids equal, la_start == E):
    copy_end, kept_from = la_end, new.page_ends[0]          # new window's first page copied whole
    if p_new == 1: copy_end -= 1; kept_from -= 1             # leave 1 image token to prefill
head = [PAGE] + old_ids[start:copy_end]
src  = (old.page_starts[p_old-1],) + range(start, copy_end)  # old KV row positions
new prompt = head + new_ids[kept_from : new.instr_start]    # no BOS, no instruction
```
The KV sequence of the reused window, in order (logical positions start at 0):

| positions | content | origin |
|---|---|---|
| 0 | `<PAGE>` of global page lo-1 | copied (old marker KV) |
| 1 .. | last 128..159 image tokens of page lo-1 | copied |
| .. | `<PAGE>` + all image tokens of page lo (own page 1) | copied from old lookahead (all but its last token if this is a 1-page final window) |
| .. | `<PAGE>`+images of pages lo+1 .. hi-1, then lookahead page hi | **prefilled** (vision runs only for these) |
| seq_len .. seq_len+n-1 | text tail (<=128 tokens) of page lo-1's transcription | copied KV (`_splice_text`, `uocr_doc_reuse.py:1237-1298`) |
| seq_len+n | forced `<PAGE>` | decode input override; its KV is computed by decode step 1 |

Details:
* The prefill's sampled token is **discarded** (`_uocr_skip_prefill_append`, `W/managers/scheduler_components/batch_result_processor.py:358-364`). `output_ids` is extended with `forced = text_tail + [<PAGE>]`, and the next decode input is forced to `<PAGE>`: "The copied KV still predicts EOS where that window stopped (18/94 follow-ups on MPDocBench); the forced marker moves decode off it" (`uocr_doc_reuse.py:1078-1080,1285-1296`).
* `max_new_tokens += len(forced)`. The forced text is hidden from the answer (`read_offset += n_forced`, `uocr_doc_reuse.py:1202-1214`), so a reused window's text **starts with the body of its first own page**.
* If the text-splice allocation fails, only the marker is forced (`forced[-1:]`) and the continuation carry is seeded on the prompt's last slot (`uocr_doc_reuse.py:1257-1285`).
* Main K/V and router index-K are copied together (`W/mem_cache/memory_pool.py:5705-5709`) with **no re-rotation** (`uocr_doc_reuse.py:500-504`, explained in 3a.6). Per-slot page-isolation state is rewritten in the new window's frame (`replay_page_isolation`), and the router block summaries `k_bar` are **recomputed** for the new block grouping, since the copied tail now starts at position 1, not on its old block boundary (`_refresh_kbar`, `uocr_doc_reuse.py:443-457,1127-1138`).
* Page-guard state for the reused window: `_uocr_context_pages = 1` (`CONTEXT_PAGES`), and `_uocr_end_on_marker = True` because there is no instruction, so the model won't stop on its own (`uocr_doc_reuse.py:1101-1109`). `expected = #<PAGE> in rewritten prompt - 1 - lookahead = own pages`. The forced `<PAGE>` counts as mark 1, opening own page 1.
* Release: the source entry is freed by an op queued after the copies (`uocr_doc_reuse.py:798-805,1297`).

Fallback (recompute the full child prompt; there is then **no cross-window context at all**): no entry, page discontinuity, layout mismatch, a page with fewer than ~129 image tokens (`start <= marker`), no text tail, a straddling multimodal item, pool pressure, or retraction (`uocr_doc_reuse.py:941-977,1038-1099,755-795`).

### 3a.5 Page-isolation numbering (window-local) across the splice

Per slot, `marker_cum` (number of markers so far), `img_run`, and `pp` (= marker_cum at the last image token so far) give the query page `q_page` (`W/layers/attention/unlimited_dash_ops/page_state.py:240-282`):
* image token: `marker_cum - 1` (0-based page)
* prompt text: `marker_cum - 1` if `marker_cum > pp` (the marker itself), else -1 (BOS/prefix and instruction)
* generated token: `marker_cum - 1 - pp` if >= 0, else -1. So the k-th generated `<PAGE>` and its body map to page k-1, which is the prompt's page k-1.

In a reused window, the tail page is local page 0, own pages are 1..n, and lookahead is n+1. The continuation carry sets the spliced text tail's slots to `marker_cum = pp+1` (page 0), so the forced `<PAGE>` lands on page 1 (`continuation_carry`, `uocr_doc_reuse.py:418-437`).

### 3a.6 2-axis RoPE coordinates (`page2d_rope.py:533-620`, backend `:1001-1031`)

Coordinates are **not** absolute positions. Both the main q/k and the router iq/ik use `(tok_pos, page_pos)`:
```
bucket = marker_cum                         if prompt token and (image or marker_cum > pp)
       = clamp(marker_cum - pp, 0, pp)      otherwise (generated tokens; also prefix/instruction -> 0)
tok_pos = running count of earlier tokens in the same (request, bucket)   # cnt[bucket]++
page_pos = bucket + (page_base % 2)  if bucket > 0 else 0;   rope uses page_pos % 2
```
* Bucket 0 holds BOS, prefix, and instruction. Bucket b >= 1 holds prompt page b's marker and images, **then** the response `<PAGE>` and body of page b. Response tok_pos continues after the page's prompt tokens: the response marker of page b gets `tok_pos = 1 + n_img(b)`. Extra generated markers are clamped to the last page.
* `page_base`: `lo` for a recomputed window, and `lo - 1` for a reused window, whose local page 1 is the tail page (`uocr_doc_reuse.py:109-122,1101-1103`). It is folded mod `page_mod` (2) and added only where `page > 0` (`backend:1001-1031`). In every window, then, a token of global 0-based page g has `page_pos ≡ g+1 (mod 2)`.
* Why no re-rotation is needed for copied KV: a copied token keeps both its page-local `tok_pos` (it was counted within the same page in the old window) and its global parity. The coordinate carry after the copied head is seeded via `replay_coords` (`uocr_doc_reuse.py:392-415,472-487,1120`). It starts `cnt` for the tail bucket at the copied count, not the old count, but no new token ever lands in that bucket.
* Frequencies (`page2d_rope.py:39-127`, `backend:1686-1717`): dim 128, half 64, 1 page pair, so n_tok = 63. `inv_tok[i] = 10000^(-i/63)` (resample), `inv_page[0] = 256^0 = 1`. `ang = [tok_pos*inv_tok, (page_pos%2)*1]` and `emb = cat(ang, ang)`. NeoX rotate-half: `x*cos + cat(-x2,x1)*sin`. cos/sin are computed in fp32 and cast to bf16. The bf16 Triton kernel rounds each product (RTNE) before the add, matching PyTorch bf16 (`page2d_rope.py:130-186`).

### 3a.7 Page guard and per-page cap across windows

See 3c. Per window, `expected` = own pages. The per-page cap counts tokens since the last `<PAGE>` in `output_ids`; in a reused window that is the count after the forced marker.

### 3a.8 Stitching (`doc_split.select_batch_output`, `doc_split.py:237-257`; `serving_chat.py:848-872,918`)

```
reused:     input_pages = hi - emit_from, context_pages = 0, leads_with_body = True
recomputed: input_pages = hi - lo,        context_pages = emit_from - lo (0), leads_with_body = False
head, *tail = text.split("<PAGE>")
bodies = [head, *tail] if leads_with_body or head.strip() else tail
while len(bodies) > input_pages and not bodies[0].strip(): bodies = bodies[1:]   # stray leading markers
owned = "".join("<PAGE>" + b for b in bodies[context_pages : input_pages])
answer = "".join(owned_k for each window k)
```
* By default nothing is deduplicated: lookahead pages are never transcribed, and spliced text is hidden. A window ended by the lookahead marker yields a trailing empty body, which is dropped because `bodies` is longer than `input_pages`.
* **Reference quirk**: if a window's first own page is blank and the window ended on the lookahead marker, the `while` loop drops the blank first body instead. The blank page then moves to the end of that window's bodies. The streaming path (`stream_owned_text`, `doc_split.py:260-275`) has no such drop.
* Finish reason: "length" if any window did not finish with "stop", fewer windows ran than planned, or any window delivered fewer bodies than it owns (`serving_chat.py:922-939`, `doc_split.py:278-286`).

### 3a.9 Port pseudo-code (single request, synchronous)

```
plan windows [(lo,hi)] of W own pages; lookahead L = 1 except the last window
for k, (lo,hi) in windows:
  if k == 0 or reuse impossible:
      prompt = BOS + pages[lo:hi+L] + instruction; page_base = lo; ctx_pages = 0; end_on_marker = (L>0)
  else:
      prompt = [PAGE] + imgtail(prev) + page(lo) [copied] + pages[lo+1:hi+L] [prefilled]
      page_base = lo-1; ctx_pages = 1; end_on_marker = True
      after prefill: drop sampled token; append KV(text_tail(prev,128)); output += text_tail; feed <PAGE>
  decode with guard(expected = #markers(prompt) - ctx_pages - L, end_on_marker) and n-gram processor
  owned text = bodies per 3a.8; retain [marker, imgtail(last own), page(hi), text_tail] if finished by EOS and k < last
```

---

## 3b. DASH sparse attention as served

### 3b.1 Confirm/correct table (HF-reference facts as given)

| claim | verdict | evidence |
|---|---|---|
| block_size 32 | **confirmed** (block = KV page; page_size forced) | `CFG`, `unlimited_ocr_v2.py(override):24-32`, `backend:155-158` |
| local window 128 tokens, no bias | **confirmed, token-level**: keys `j` with `q-127 <= j <= q`. Remote blocks contribute only keys `j <= q-128`, so the block straddling the window edge is split between passes | `attend_triton.py:196-202,230-281,369,409,448-491` |
| remote candidates = own-page image blocks, fallback all | **confirmed with precision**: remote = blocks `b <= floor(q/32) - 4` with `32b < seq_len`. `S_p` = blocks whose image-page range `[min,max]` contains `p = q_page`. cand = remote ∩ S_p if non-empty, **else all remote**. If `q_page = -1` (BOS, instruction, generated text before the first `<PAGE>`), cand = remote | `dash_stage0.py:44-58`, `route_triton.py:109-116,160-168`, `page_state.py:315-371` |
| per-layer 1-head 128-d router q/k + Gemma RMSNorm (1+w) + 2-axis RoPE | **confirmed**. No bias, eps 1e-6, and the same page2d coords as main. Index K is stored **post-norm, post-RoPE**, and k_bar pools rotated keys | `models/unlimited_ocr_v2.py:100-121,145-171`, `backend:1412-1416` |
| k_bar = softmax(head_cls·k/√128)-pooled keys per block | **confirmed**. fp32; logits clamped to ±1e4; softmax over valid tokens only (denominator clamped >= 1e-6); NaN → 0 | `dash_stage0.py:25-41`, `route_triton.py:37-72` |
| score = idx_q·k_bar/√128 | **confirmed**, clamped to ±1e4 | `route_triton.py:123-124` |
| entmax-1.5 over the **top-16** scores ×(scale/√128) | **CORRECTED for serving**: the Triton router runs entmax-1.5 over **all candidate blocks**, with an exact τ bisection. `dash_infer_topm`/16 only sets the output slot width. The torch `route_and_select` (top-M window) is never called. The HF *inference* path does use top-M=16 (`HF/modeling_deepseekv2.py:273,352-354`). Both agree whenever the entmax support has <= 16 blocks | `route_triton.py:129-189,623-678,729-847`, grep shows no caller of `route_and_select` |
| read set = top-12 | **confirmed**: the 12 highest-z candidates, including zero-prior ones. With ties at the boundary, the bisection may select fewer than 12 | `route_triton.py:191-202` |
| bias = (log(max(p,1e-4)) - mean_valid)/σ | **confirmed**: mean over the selected (active) set, σ = 1. The bias is added to the q·k/√d logits of remote keys of that block (in log2 domain as `bias*log2e`) | `route_triton.py:204-212`, `attend_triton.py:205-209,423-427` |

### 3b.2 Exact router math per layer per query token t (`route_triton.py:129-221`)

```
s_b   = clamp(iq_t · k_bar_b / sqrt(128), -1e4, 1e4)    for remote b, else -1e9
x_b   = s_b * (128/sqrt(128))   if b in cand  else -1e4
z_b   = (x_b - max_{cand} x) / 2
tau   : sum_{b in cand} max(z_b - tau, 0)^2 = 1,  tau in [-1, 0]; 24-step bisection (fp64 sum)
p_b   = max(z_b - tau, 0)^2
A     = cand ∩ {z_b > thr},  thr = 40-step bisection so |A| <= 12 (A = cand if |cand| <= 12)
beta_b = (ln max(p_b, 1e-4) - mean_{A} ln max(p, 1e-4)) / 1.0     for b in A
```
Attention per head: softmax over `{local keys: q·k/√128} ∪ {keys j <= q-128 in blocks A: q·k/√128 + beta_{b(j)}}`. If `A = ∅` (first 4 blocks), only the local window is used. Selected blocks are packed in ascending block order; the HF path packs by descending score, which only changes summation order.

### 3b.3 Serving-only behaviors / differences

* `sparse_init_block` (=1 in config) is never used. There is no sink block.
* There is no `DASH_PAGE_VISIBLE_BACK` or any previous-page visibility knob in the wheel. It appears only in an HF docstring (`HF/modeling_unlimitedocr.py:1059`, `=0`). Visibility of other pages comes only from the local window and the fallback-to-all-remote rule.
* k_bar is cached per physical page and per layer (`KBarCache`). Prefill refreshes the boundary page plus the pages in the query span, decode refreshes the last page each step (`dash_stage0.py:106-148`, `route_triton.py:575-601,681-726`, `backend:1231-1318`). Remote blocks are always complete, so caching is semantically transparent. Copied and spliced segments get recomputed summaries.
* Image-page membership per block is computed from per-slot `img_page` (`page_state.py:315-328`). Stale slots past the live tail are cleared each step.
* The candidate mask union with local blocks in decode (`backend:1111-1154`) has no effect on routing, because local blocks are not remote.
* Decode scan windowing (`SCAN_TILE = 128`, `route_triton.py:466-505`) is a performance path with identical selection.

---

## 3c. Logit processor: n-gram ban and page guard (`W/sampling/custom_logit_processor.py:194-387`)

Order per row: (1) optional periodic-tail stop, (2) page guard (a forced token short-circuits), (3) n-gram ban.

**N-gram ban** (`:343-385`). All parameters come from the request's `custom_params`; there is **no server default**. README uses `ngram_size=35`, `window_size=1024` (multi-page) / `128` (single), no whitelist, and `temperature=0`.
```
S = output_ids            (SGLANG_UOCR_NGRAM_OUTPUT_ONLY=True; else prompt+output)
if n<=0 or window<=0 or len(S) < n: skip
start = max(0, len(S)-window); end = len(S)-n+1
prefix = S[-(n-1):]
banned = { S[i+n-1] : start <= i < end, S[i:i+n-1] == prefix } - whitelist
logits[banned] = -inf
```
In a reused window, `output_ids` **includes the spliced text tail and the forced `<PAGE>`**, so they take part in n-gram matching.

**Periodic tail** (opt-in: `rep_max_pattern_size`, `rep_min_pattern_size`, `rep_min_count >= 2`, `eos_token_id`): if the output ends with a pattern repeated `rep_min_count` times, force EOS (`:169-191,292-313`).

**Page guard** (`_page_guard`, `:194-262`). It is active when `page_token_id` and `eos_token_id` are present.
```
end_on_marker = lookahead > 0 or req._uocr_end_on_marker
if stop_only and not end_on_marker: off
expected = #<PAGE>(prompt) - context_pages - lookahead;  if expected <= 0: off
marks = #<PAGE>(output_ids); last = index of last <PAGE> in output (-1 if none)
if marks > expected: force EOS
if stop_only: nothing else
if 0 < max_page_tokens <= len(output) - (last+1): force <PAGE> if marks < expected else EOS
if marks < expected: ban EOS
elif not end_on_marker: ban <PAGE>      # last page of an unsplit / instruction-bearing final window
```
Params source (`doc_split.page_guard_params`, `:294-330`; `serving_chat._apply_page_cap`, `:535-578`):
* With server cap > 0 (README 20000), full guard params are set, unless the request names its own cap <= the server's, in which case the request's own params are used. The cap is `min(request, server)`.
* With server cap = 0 (wheel default), windows with a lookahead or a reuse candidate (`bi > 0`) get `page_stop_only=True`, which only applies "force EOS once marks > expected". An unsplit request gets no guard unless the client sends the params.
* Wiring this up needs `--enable-custom-logit-processor`. The built-in processor is `DeepseekOCRNoRepeatNGramLogitProcessor`.

The README's per-request `max_page_tokens = 20000` is HF's `infer(... max_page_tokens=20000 ...)` default too (`HF/modeling_unlimitedocr.py:1043`).

---

## 3d. KV memory retirement

* **Within a window: none.** There is no eviction of a page's image KV after the decoder moves past it. grep finds no retire/visibility logic in the backend, session, or model files. All KV of the current window (own + lookahead pages + generated text) stays resident until the window finishes.
* **Across windows**: at window finish, KV outside the four kept ranges is freed at once (`_trim_to_tails`, `uocr_doc_reuse.py:903-938`). The kept ranges are the marker, the image tail, the lookahead page, and the text tail, rounded to 32-token pages. They are freed after the next window's copies run (`:798-805,1297`), on document release (`:648-654`), or under memory pressure, oldest first (`free_for_decode`/`on_no_token`, `:711-795`). Eviction makes the follow-up recompute.
* Peak KV per document is therefore about one window (own + 1 lookahead page + its outputs) plus the retained tail. The paper reports 47.00 GB → 1.77 GB peak on MarathonDocBench.

---

## 4. What cannot be determined from the wheel

* The paper's exact training-time chunking. The wheel's lookahead/tail-copy is "approximate; validated by OCR accuracy" (`uocr_doc_reuse.py:1-5`). No accuracy numbers ship in the wheel beyond inline anecdotes (18/94 EOS follow-ups, "11 bodies, all 104 pages correct").
* Whether the HF/vLLM inference paths split documents at all. HF `infer_multi` processes the whole document in one pass (README:101-108).
* Vision-tower details of gundam token ordering beyond the token counts above.

---

## 5. Upstream delta: wheel dev10 → dev13 (reviewed 2026-10-29)

On 2026-10-29 09:17-09:18 UTC the HF repo (HEAD `c14071b549f4fcab10909f3c4acc6a2edb366947`)
uploaded `wheel/sglang-0.5.20.dev13+g75b9b06b02-py3-none-any.whl` (sha256
`576c71dab3cacad843f6870739494e13b2e8169a69cc038f84395d6503e801ea`), deleted the dev10 wheel, and
changed only the wheel filename in `README.md`. Every model, code, tokenizer and weight file is
byte-identical to the pin (same git blob / LFS oids). The dev10 wheel stays retrievable at the pinned
commit. `diff -rq` of the two wheels' `sglang/` trees differs in 9 files; none changes the semantics
above:

| file | change | effect on this spec |
|---|---|---|
| `layers/attention/unlimited_ocr_v2_backend.py` (`_refresh_recorded_in_graph`, dev10 :687-692) | the recorded-refresh shortcut now requires `md.use_decode_twin`, not `not md.use_prefill_twin` | **Bug fix in dev10.** An eager decode (CUDA graphs off, or a batch past the capture sweep) skipped building page ids, coordinates and masks, so it ran "without RoPE coordinates or page isolation". dev13 restores the semantics sections 3a.5/3a.6/3b describe. Any SGLang reference run must use dev13 or later. |
| `session/uocr_doc_reuse.py` (+58 lines after dev10 :726) | new `free_for_admission` / `on_request_rows` (with `ROW_STALL_PASSES = 2`), called from the scheduler | When retained windows hold every request row and nothing is running, the newest retained window of another document is freed; that document's next window **recomputes** (section 3a.4's fallback, one more trigger). Server-load only; a single-document engine never reaches it. A reference run under load can take the recompute path. |
| `sampling/custom_logit_processor.py` | forced rows and banned cells are collected and written once (`_apply_row_edits`) | Same edits, same order constraints (a forced row has no bans). Host-sync performance only. |
| `environ.py` (+8 lines after dev10 :1374) | `SGLANG_UOCR_CHAT_BODY_PARSE_CONCURRENCY` (4), `SGLANG_UOCR_IMAGE_SPILL_DIR` ("") | Host-memory knobs; no decode effect. |
| `managers/scheduler.py`, `mem_cache/kv_cache_configurator.py`, `entrypoints/http_server.py`, new `entrypoints/openai/chat_request_body.py`, `_version.py` | serving plumbing for the above | None. |

Line numbers in this file stay dev10's: in dev13, `uocr_doc_reuse.py` citations past :726 shift by
+58, `environ.py` past :1374 by +8, and the backend past :687 by +4.
