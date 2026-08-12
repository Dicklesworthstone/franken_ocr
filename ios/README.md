# FrankenOCR — the iOS/iPadOS app

The franken-ocr.com playground, rebuilt as a native app. Same engine: the Swift
here drives the identical `OcrModel` entry points the `focr` CLI and the browser
playground drive, through a small C ABI. Nothing about the model pipeline forks.

## Build

```bash
ios/build-rust.sh                 # FocrCore.xcframework (after ANY Rust change)
cd ios && xcodegen generate       # only after project.yml changes
open FrankenOCR.xcodeproj
```

Headless check:

```bash
xcodebuild -project ios/FrankenOCR.xcodeproj -scheme FrankenOCR \
  -destination "generic/platform=iOS Simulator" CODE_SIGNING_ALLOWED=NO build
```

`FocrCore.xcframework/` and `FrankenOCR.xcodeproj/` are generated and
gitignored. `project.yml`, `build-rust.sh`, `Sources/`, the entitlements, and
the privacy manifest are the source.

## Why a phone can run the flagship at all

The browser playground gates Unlimited-OCR to desktop, and the reason is not
compute — it is that WebAssembly has no mmap. A wasm module must stage the
whole 3.0 GB artifact into linear memory, so 3.0 GB of the measured ~3.6 GB
peak is the blob itself, and no residency trick can move it.

A native app maps the file instead. `focr-ios` therefore loads the engine from a
**path**, not from bytes, so `Weights::load` reaches the mmap island and the
artifact becomes clean, file-backed pages the kernel may evict — not the dirty
anonymous heap that jetsam counts and kills for. Two supporting pieces:

- **`MADV_RANDOM`.** A MoE decode routes to 6 of 64 experts per layer, so the
  default sequential read-ahead would fault in neighbouring experts the token
  never reads. The advice is what keeps the resident working set near the bytes
  actually used.
- **Streamed vision.** The per-block SAM/CLIP path (already in the engine, and
  gated bit-identical against the cached path) replaces ~1.6 GB of hydrated f32
  tower with tens of MB of scratch. It engages automatically, because it keys off
  the artifact's recipe string — and this app ships that recipe.

The `increased-memory-limit` entitlement is there for the page cache, not for the
heap.

## Engine changes this app depends on

| change | why |
|---|---|
| `mmap` split out of the `native` feature | so an app can map weights without asupersync, fsqlite, clap, ctrlc, and nix |
| mmap default-ON under `target_os = "ios"` | the sandbox supplies the enforceable immutability the desktop default could not assume |
| `MADV_RANDOM` on the mapping | stop read-ahead inflating the resident set on a scattered MoE access pattern |
| ISA gates moved from `target_os = "macos"` to `target_vendor = "apple"` | an iOS build was silently taking the Neoverse branch and preferring SMMLA, which is the measured-slower kernel on Apple cores |
| `init_kernel_pool()` + Apple QoS on workers | the documented thread budget was never actually installed, and an un-classified worker gets demoted to an E-core while the whole fork-join barrier waits for it |
| iOS thread default of `physical - 1` | leave a core for the UI that is drawing the progress bar |

## Honesty

No speed figure is claimed here. The app measures each run on the device it is
running on and shows that number; nothing else is asserted until it is measured
on A-series hardware. Same rule the ledgers use.

## Not in this version

- GOT-OCR2, SmolVLM2, and OneChart. Each hydrates its vision tower to f32 whole
  (measured wasm peaks of 3.4 GB and 2.8 GB) because the streamed residency mode
  keys off the Unlimited-OCR recipe. Extending it to their towers is what earns
  them a place in the picker; shipping them first would ship a model that gets
  the app killed.
- Figure extraction, multi-page cross-referencing (`--multi-page`), and batch
  mode.
- App Store submission artifacts beyond the privacy manifest and entitlement.
