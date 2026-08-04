# Model weights — build + upload runbook

The weights themselves are **never committed** (`*.focrq` is gitignored); only
the small [`manifest-v2.json`](./manifest-v2.json) lives in the repo. The primary
Unlimited-OCR entry names the quality-cleared conservative recipe published in
v0.7.0. Its three GitHub URLs resolve to remotely verified release assets whose
sizes and SHA-256 digests match the table below.

[`manifest.json`](./manifest.json) remains the schema-1 endpoint fetched by
already-released v0.6 binaries. New binaries do not fetch it: they embed
`manifest-v2.json`, so a later branch edit cannot silently retarget a release.

## v0.7.0 conservative artifact (published and remotely verified)

Recipe: `unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1`

| artifact | size | sha256 |
|---|---:|---|
| `unlimited-ocr.v0.7.0.int8.focrq` (reassembled install file) | 4 157 448 783 | `573340710167697891bf52dfa4cbb5d0a02a68f3011c01f8ef83fd34622fb592` |
| `unlimited-ocr.v0.7.0.int8.focrq.part00` | 1 957 046 720 | `a45aa7674f38190974a2e61bdaeb8eca0d5039a6631406c1126f6614140ec7f6` |
| `unlimited-ocr.v0.7.0.int8.focrq.part01` | 1 957 046 720 | `0081dbab8005f9bae0abae32fea6f85d20b507697ee55f2daff8d66137f9d5a8` |
| `unlimited-ocr.v0.7.0.int8.focrq.part02` | 243 355 343 | `62d34bc6acb431e0b261e8d42c0834886f3b260083c3db2ba46fde5d0d6d2eec` |

The source artifact and retained split live outside the repository:

```text
/Volumes/USBNVME16TB/temp_agent_space/franken_ocr_work/model/unlimited-ocr.recipe-v1.20260709T2304.focrq
/Volumes/USBNVME16TB/temp_agent_space/franken_ocr_release_0_7_0/model_assets/
```

The staging directory contains `MODEL_ASSET_RECEIPT.json`, `SHA256SUMS`, and
`RECONSTRUCTION.txt`. The receipt records the pinned source checkpoint SHA-256,
2,710-tensor census, recipe, split command, ordered reconstruction, and exact
whole-file result.

[GitHub's release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
requires each release asset to be under 2 GiB. The largest staged part is
1,957,046,720 bytes, safely below 2,147,483,648 bytes; GitHub places no total
release-size or bandwidth limit on these assets.

### Reverify the retained staging set

```bash
A=/Volumes/USBNVME16TB/temp_agent_space/franken_ocr_release_0_7_0/model_assets
cd "$A"
shasum -a 256 -c SHA256SUMS
cat \
  unlimited-ocr.v0.7.0.int8.focrq.part00 \
  unlimited-ocr.v0.7.0.int8.focrq.part01 \
  unlimited-ocr.v0.7.0.int8.focrq.part02 | shasum -a 256
cat \
  unlimited-ocr.v0.7.0.int8.focrq.part00 \
  unlimited-ocr.v0.7.0.int8.focrq.part01 \
  unlimited-ocr.v0.7.0.int8.focrq.part02 | wc -c
```

The final two commands must report the whole-file SHA-256 and byte count from
the table. The source split was deterministic:

```bash
split -b 1957046720 -d -a 2 SOURCE \
  unlimited-ocr.v0.7.0.int8.focrq.part
```

macOS created `._*` AppleDouble sidecars on the external volume. They are not
release assets. Never upload with a wildcard; name exactly the three files in
the table.

### v0.7.0 upload command (completed)

The release-preparation/DSR lane created the tag and release. The public v0.7.0
release contains exactly the three retained files below; this command is kept
for provenance and must not be rerun:

```bash
A=/Volumes/USBNVME16TB/temp_agent_space/franken_ocr_release_0_7_0/model_assets
gh release upload v0.7.0 \
  --repo Dicklesworthstone/franken_ocr \
  "$A/unlimited-ocr.v0.7.0.int8.focrq.part00" \
  "$A/unlimited-ocr.v0.7.0.int8.focrq.part01" \
  "$A/unlimited-ocr.v0.7.0.int8.focrq.part02"
```

Do not rerun the upload or use `--clobber`. The public release reports the three
exact names, sizes, and SHA-256 digests. On 2026-07-11, a clean-cache v0.7.0
`focr pull` verified every part hash, installed the artifact with reassembled
SHA-256 `573340710167697891bf52dfa4cbb5d0a02a68f3011c01f8ef83fd34622fb592`,
returned `from_cache:true` on a second pull, and reproduced the pinned
`page_0009` OCR golden byte-for-byte. The live manifest deliberately has only
the GitHub URLs. Add a Hugging Face fallback in a later commit only after the
identical objects have actually been uploaded and verified there.

The tokenizer remains public at `baidu/Unlimited-OCR`; its manifest identity is
9,979,544 bytes and SHA-256
`a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4`.

## Historical Unlimited-OCR artifact (blocked)

| artifact | size | sha256 |
|---|---|---|
| `unlimited-ocr.int8.focrq` (reassembled) | 3 914 093 440 | `d8c5fcf223c8e062af63f6b86964d099e2c5a5b272ae096a09433aaf5510a440` |
| `unlimited-ocr.int8.focrq.part00` | 1 957 046 720 | `e58503fb1700a56cff71d2c136c223efa5c10c38604901987904467267c815e3` |
| `unlimited-ocr.int8.focrq.part01` | 1 957 046 720 | `6a647a04fece7bced666c0e56581969c1a799e8447c64e9abf37ee10c322ec7d` |
| `tokenizer.json` | 9 979 544 | `a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4` |

The two `.focrq` parts are the already-published `models-v1` byte split. Their
concatenation has recipe
`unlimited-ocr-full-int8-attn-int8-lmhead-int8-v1`. That recipe is incompatible
with the current runtime requirement,
`unlimited-ocr-ffn-int8-attn-bf16-lmhead-bf16-v1`, so `focr pull` deliberately
rejects it. The hashes above are historical verification data, not instructions
to republish or re-enable the artifact.

The `models-v1` bytes stay available for old provenance records but must never be
restored as the current default. Their exact-recipe rejection remains covered by
`legacy_default_pull_stops_before_any_artifact_request`.

## Published v0.7.0 behavior

The three v0.7.0 model parts are published and remotely verified. A v0.7.0
release binary accepts the embedded exact recipe; `focr pull` downloads,
verifies, and installs the reassembled artifact:

```bash
focr pull
focr ocr page.png
```

`focr pull` installs `unlimited-ocr.v0.7.0.int8.focrq` only after all three part hashes,
the 4,157,448,783-byte total, and the whole-file SHA-256 match. Raw BF16 or another
exact-recipe local artifact remains available through `FOCR_MODEL_PATH`.

---

# Zoo models — smolvlm2 / onechart / tromr (bd-av64.7/.8)

Every asset is **under GitHub's 2 GB cap, so NO split parts** (the asymmetry vs
unlimited-ocr above). One GH release tag per model; the HF mirror uses a
**per-model subdirectory** (sidecar filenames are not unique across models —
smolvlm2 ships its own `tokenizer.json`). `focr pull <model>` installs each
non-primary model into its own cache subdir (`~/.cache/franken_ocr/models/<id>/`)
for the same reason.

Source of truth for every byte: the staged files on the USB zoo
(`/Volumes/USBNVME16TB/temp_agent_space/zoo/<model>/`), hashed 2026-07-06; the
committed `manifest-v2.json` carries the same sizes + sha256s (the embedded-manifest
lint test `builtin_manifest_publishes_the_zoo_and_lints_clean` cross-checks the
shape).

## What the manifest points at

| release tag | asset | size | sha256 (first 16) |
|---|---|---|---|
| `models-smolvlm2-v1` | `smolvlm2.int8.focrq` | 1 087 397 293 | `4ad2ac89e47c83ad…` |
| `models-smolvlm2-v1` | `tokenizer.json` | 3 548 256 | `5ece781dc8d2b2f3…` |
| `models-onechart-v1` | `onechart.int8.focrq` | 362 863 824 | `618189a8e975f0cf…` |
| `models-onechart-v1` | `vocab.json` | 999 355 | `32b29acf82d33334…` |
| `models-onechart-v1` | `merges.txt` | 456 318 | `1ce1664773c50f3e…` |
| `models-onechart-v1` | `added_tokens.json` | 82 | `e1b04af1435ff5b4…` |
| `models-tromr-v1` | `tromr.int8.focrq` | 61 107 485 | `cced11c0f05656dd…` |
| `models-tromr-v1` | `tromr.focrq` (f32) | 86 168 002 | `a9d41485a98534ad…` |
| `models-tromr-v1` | `tokenizer_rhythm.json` | 10 743 | `603bfef760e8424f…` |
| `models-tromr-v1` | `tokenizer_pitch.json` | 2 682 | `2382e8b20c147329…` |
| `models-tromr-v1` | `tokenizer_lift.json` | 979 | `b61ba09cecd5bc34…` |
| `models-tromr-v1` | `tokenizer_note.json` | 830 | `504d886d11e3c1fe…` |

Runtime-required file sets (verified against the loaders 2026-07-06 — a pull
missing any of these is a broken pull): smolvlm2 = focrq + `tokenizer.json`;
onechart = focrq + `vocab.json` + `merges.txt` + `added_tokens.json`
(`Tokenizer::from_opt_dir`); tromr = focrq + all four `tokenizer_*.json`
(`MusicTokenizer::from_dir`). TrOMR's `config.yaml` is **convert-time only**
(zero runtime references) — attach it to the GH release for provenance if you
like, but it is deliberately NOT in the manifest. TrOMR publishes **both
quants** since bd-av64.12 ran the lossless proof (40 decoder GEMMs int8,
committed golden byte-identical, corpus gate delta 0 — divergence ledgered as
DISC-005): the default `focr pull tromr` now fetches `tromr.int8.focrq`;
`--quant f32` fetches the bit-exact reference artifact.

## Upload — GitHub Releases (mirror 1, copy-paste)

```bash
Z=/Volumes/USBNVME16TB/temp_agent_space/zoo

gh release create models-smolvlm2-v1 --repo Dicklesworthstone/franken_ocr \
  --title "SmolVLM2-500M int8 weights (models-smolvlm2-v1)" \
  --notes "SmolVLM2-500M-Video-Instruct (HuggingFaceTB, Apache-2.0) int8 .focrq + tokenizer; see models/manifest-v2.json"
gh release upload models-smolvlm2-v1 --repo Dicklesworthstone/franken_ocr \
  "$Z/smolvlm2/smolvlm2.int8.focrq" "$Z/smolvlm2/tokenizer.json"

gh release create models-onechart-v1 --repo Dicklesworthstone/franken_ocr \
  --title "OneChart int8 weights (models-onechart-v1)" \
  --notes "OneChart (kppkkp, Apache-2.0) int8 .focrq + OPT tokenizer triple; see models/manifest-v2.json"
gh release upload models-onechart-v1 --repo Dicklesworthstone/franken_ocr \
  "$Z/onechart/onechart.int8.focrq" "$Z/onechart/vocab.json" \
  "$Z/onechart/merges.txt" "$Z/onechart/added_tokens.json"

gh release create models-tromr-v1 --repo Dicklesworthstone/franken_ocr \
  --title "Polyphonic-TrOMR f32 weights (models-tromr-v1)" \
  --notes "Polyphonic-TrOMR (NetEase, Apache-2.0) f32 .focrq + the four music tokenizer tables; see models/manifest-v2.json"
gh release upload models-tromr-v1 --repo Dicklesworthstone/franken_ocr \
  "$Z/tromr/tromr.focrq" "$Z/tromr/tokenizer_rhythm.json" \
  "$Z/tromr/tokenizer_pitch.json" "$Z/tromr/tokenizer_lift.json" \
  "$Z/tromr/tokenizer_note.json"
```

## Upload — Hugging Face mirror (per-model SUBDIRECTORIES)

```bash
Z=/Volumes/USBNVME16TB/temp_agent_space/zoo
R=Dicklesworthstone/franken_ocr-weights

huggingface-cli upload "$R" "$Z/smolvlm2/smolvlm2.int8.focrq" smolvlm2/smolvlm2.int8.focrq
huggingface-cli upload "$R" "$Z/smolvlm2/tokenizer.json"      smolvlm2/tokenizer.json
huggingface-cli upload "$R" "$Z/onechart/onechart.int8.focrq" onechart/onechart.int8.focrq
huggingface-cli upload "$R" "$Z/onechart/vocab.json"          onechart/vocab.json
huggingface-cli upload "$R" "$Z/onechart/merges.txt"          onechart/merges.txt
huggingface-cli upload "$R" "$Z/onechart/added_tokens.json"   onechart/added_tokens.json
huggingface-cli upload "$R" "$Z/tromr/tromr.focrq"            tromr/tromr.focrq
huggingface-cli upload "$R" "$Z/tromr/tokenizer_rhythm.json"  tromr/tokenizer_rhythm.json
huggingface-cli upload "$R" "$Z/tromr/tokenizer_pitch.json"   tromr/tokenizer_pitch.json
huggingface-cli upload "$R" "$Z/tromr/tokenizer_lift.json"    tromr/tokenizer_lift.json
huggingface-cli upload "$R" "$Z/tromr/tokenizer_note.json"    tromr/tokenizer_note.json
```

## Post-upload spot-check (before the full bd-av64.9 verification)

```bash
# Every GH URL in the manifest must answer 200 with the manifest's size:
python3 - <<'PY'
import json, urllib.request
m = json.load(open('models/manifest-v2.json'))
for mid, e in m['models'].items():
    files = [q['focrq'] for q in e['quants'].values()] + [e['tokenizer']] + e.get('sidecars', [])
    for f in files:
        url = f['parts'][0]['urls'][0]
        r = urllib.request.urlopen(urllib.request.Request(url, method='HEAD'))
        size = int(r.headers['Content-Length'])
        print(('OK ' if size == f['size'] else 'SIZE MISMATCH ') + f"{mid} {f['filename']} {size}")
PY
```

Then run the full bd-av64.9 verification: clean-cache `focr pull
smolvlm2|onechart|tromr` + one real inference per model.
