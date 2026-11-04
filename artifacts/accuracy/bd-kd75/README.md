# GOT and OneChart reading accuracy on clean synthetic charts and formulas

Measured 2026-10-31 on an Apple M5 Ultra, with the published artifacts
(`got-ocr2.int8.focrq`, `onechart.int8.focrq`) and the upstream GOT checkpoint
(`stepfun-ai/GOT-OCR2_0@979938bf`, sha256 `77d61440…`). Beads: bd-kd75, bd-pi74, bd-a7zs.

The images are synthetic and clean, in one drawing style. They can show a lane
failing. They cannot show a lane working on real-world charts.

## What is here

| path | contents |
|---|---|
| `gen_probe.py` | Generates the 36 charts, 8 formulas and `truth.json`. Rebuilt byte-identically for this write-up, on matplotlib 3.11.2. |
| `images.sha256` | SHA-256 of all 44 images. |
| `score_probe.py` | `score_probe.py truth.json readings/<dir>` scores one directory of chart readings. |
| `readings/int8_squashed/` | Int8, upstream squash (`--format` behaviour). |
| `readings/int8_sips_pad/` | Int8, images padded by `sips` first, the experiment that motivated the change. |
| `readings/int8_task/` | Int8 through the shipped `--task chart` and `--task formula`. |
| `readings/formula_framing/` | `formula.png` upscaled and cropped, each with and without padding. |
| `readings/f32/` | True f32 through `decoder_qwen2::generate_greedy`. Files without `_sq` are squashed; those with it are padded. `chart.out` is the corpus chart. |
| `readings/{attn,ffn,both}_high/` | Int8 artifacts with that tensor set kept BF16, through the same re-prefill decode. |
| `readings/attn_kv/` | The attention-BF16 artifact through the shipped `focr ocr --task` path, on every image `int8_task` covers. `format_*` are the five corpus assets through `--format`. |
| `readings/onechart_text/`, `readings/onechart_heads.tsv` | OneChart's text, and its head and verdict per chart. |

A chart is **right** when some series in the reading holds exactly the true A-D
values. GOT nests a series under a name it guesses, so both shapes count. Titles are
not scored: GOT often loses them.

## GOT charts

Int8, right of 12 per shape:

| shape | squashed | `--task chart` |
|---|---:|---:|
| 3:2 (600×400) | 2 | 10 |
| 16:9 (800×450) | 12 | 11 |
| 3:4 (450×600) | 6 | 8 |
| **all, of 36** | **20** | **29** |

Padding is a net win and not a clean one: 16:9 loses a chart. Where a squashed
reading fails it often invents data, such as "Biscuit and Puffed Food" with years for
labels, so a wrong reading looks like a confident one.

The shipped `--task` readings are byte-identical to the `sips` experiment on all 36
charts and all 8 formulas.

### Two separate losses

**The squash hurts f32 too.** Six squashed 3:2 charts on f32: 1 right. Int8 on the same
six: 0 right. So this is the model and its input, and it survives any change of recipe.

The f32 decodes for that row were capped at 100 ids, because re-prefill costs O(n²) per
token and a right chart needs about 75. Three hit the cap (`chart00`, `chart01`,
`chart10`) with invented series already in the output. An earlier uncapped run of
`chart00` stopped by itself at 120 ids, equally wrong.

**Int8 costs a further margin.** Three charts where shipped int8 is wrong and f32 is
right, decoded through the dtype-honouring re-prefill path:

| artifact | size | `chart.png` squashed | `chart02` padded | `chart08` padded |
|---|---:|---|---|---|
| shipped: attention + FFN int8 | 814 MB | wrong | wrong | wrong |
| FFN int8, attention BF16 | 914 MB | wrong | right | right |
| attention int8, FFN BF16 | 1,021 MB | wrong | right | right |
| no decoder int8 | 1,121 MB | right | right | right |
| upstream f32 | | right | right | right |

Vision and embeddings are BF16 in every row, so they are cleared. Neither tensor set
alone restores `chart.png` on this path, so the loss adds up across both. These rows
were decoded by re-prefill because the shipped KV-cache decode then quantized
attention at load whatever the file stored. It now keeps attention as stored (see
below).

**Sizing it at `--task chart`.** All seven charts int8 misses there were read on f32:

| chart | f32 | int8 |
|---|---|---|
| `chart02`, `chart08` | right | wrong |
| `tall01` | values right, labelled Q1-Q4 for A-D | the same reading |
| `tall02`, `tall08` | wrong | the same reading |
| `tall07` | wrong | wrong, differently |
| `wide05` | right first series, then a repeated one with a broken quote, cut at the 100-id cap | the same, complete |

So five of the seven are the model's and two are int8's, both of which attention BF16
alone fixed (table above). On this probe the shipped recipe costs 2 of 36 charts at
`--task chart`. The 29 charts int8 reads right were not decoded on f32, so this counts
what f32 gains and not anything it might lose. `wide05` and the three `tall` charts with
f32 times over 300 s were decoded at background priority beside other work.

### Attention BF16 through the shipped decode

The KV-cache decode now keeps attention projections in the precision the file stores
(it used to quantize them at load). So the attention-BF16 artifact (`attn_high`, FFN
int8, 914 MB) was read through the shipped `focr ocr --task` path on every image above,
and the published artifact was read again on the same build: byte-identical to its
frozen readings on all 51 (36 charts, 8 formulas, the table, `chart.png` through
`--task chart`, and the five corpus assets through `--format`).

| | published (all int8) | `attn_high` |
|---|---:|---:|
| `--task chart`, right of 36 | 29 | 28 |
| `--task formula`, right of 8 | 3 | 3 |
| `chart.png` through `--format` | wrong | right |

`attn_high` fixes `chart02` and `chart08`, the two charts int8 lost and f32 keeps, and
the corpus chart. It loses `chart01`, `tall09` and `wide07`, so all three were read on
f32 too (`readings/f32/*_sq.out`, padded):

| chart | f32 | published | `attn_high` |
|---|---|---|---|
| `chart01` | invents a series of years, cut at the 100-id cap | right | invents a series of years |
| `tall09` | first series 11 9 10 12 for 11 9 1 1; cut at the cap in a second series after A 11, B 9 and a lowercase `c` key | right | wrong |
| `wide07` | Q1-Q4 for A-D, values 3 2 11 11 for 3 11 2 11 | right | Q1-Q3, 3 2 11 |

f32 misses all three. Of the ten charts read at `--task chart` on f32, `attn_high` is
right on exactly the two f32 reads right and wrong on the eight it misses. The
published artifact reads three of f32's misses right. Readings this close to a tie
flip with any change of arithmetic: under `attn_high`, the re-prefill and KV-cache
decodes differ only in summation order, and they read `chart.png` differently.

`attn_high` is not simply closer to f32 everywhere. Of the corpus assets, it matches
the frozen f32 reading on the chart, the table and the molecule and departs from it on
the formula (a shorter wrong SMILES) and the music (`4dd` for f32's `4cc`), where the
published artifact matches f32 byte for byte. Both are wrong against the truth either
way.

Cost, from 24 paired runs (ABBA blocks of four, `chart02` and `wide03`, `--task
chart`, release-perf, host load 74-84 from other work):

| | published | `attn_high` |
|---|---:|---:|
| decode, median ms per token | 114.2 | 121.1 |
| decode time, median paired ratio | | 1.044 (blocks 1.033-1.160) |
| wall, median paired ratio | | 1.059 |
| peak RSS, median | 4.83 GB | 5.26 GB |
| file | 814 MB | 914 MB |

**Verdict: the published recipe stays.** On this probe, attention BF16 buys back two
charts int8 loses and gives up three it reads right, so 28 against 29 of 36. That is
no accuracy gain, at 100 MB more to download, 0.43 GB more peak memory and about 4%
slower decoding. A GOT chart reading is wrong as often because of the model as because
of int8: f32 misses 8 of the 10 charts read on it. The defence is the advice in the
top-level README to check every number, not a recipe. The chart budget in
`scripts/got_cer.py` stays where it is.

## GOT formulas

Right of 8, against the source LaTeX with spacing and braces removed:

| | squashed | `--task formula` |
|---|---:|---:|
| sans-serif, as in `formula.png` | 0 of 4 | 1 of 4 |
| Computer Modern | 1 of 4 | 2 of 4 |

The misses come back as `<smiles>` or TikZ, not as bad LaTeX. Plain mode on the same
image reads `1 E= mc 2+  1 x 2 dx`, so the model sees the glyphs and format mode picks
the wrong grammar. Cropping and upscaling `formula.png` did not help
(`readings/formula_framing/`), and its f32 reading is byte-identical to int8
(`tests/fixtures/got/format_corpus/`).

## GOT tables

One image, and it is why padding is per task: `table.png` padded by `sips` reads as
SMILES, where squashed it is exact (`readings/int8_sips_pad/table_sq.out`). The
shipped `--task tables` stays squashed and matches the frozen reading byte for byte.

## OneChart

On all 36 charts, through `onechart::recognize`:

| signal | result |
|---|---|
| Text with a parsed `values` dict | 0 of 36 |
| Number head fired | 36 of 36 |
| Head error against the chart's min-max scale | mean L1 0.052, median 0.029, worst 0.255 |
| Head under the model's 0.1 threshold | 31 of 36 |
| `reliable` verdict | `None` on 36 of 36 |

The text garbles at every precision: `onechart.rs` records f32 and int8 as
byte-identical on its corpus. The head gives relative heights only and does not say how
many bars there are. `readings/onechart_text/` stops at 34 charts, because that run was
stopped once the answer was clear; `onechart_heads.tsv` covers all 36.
