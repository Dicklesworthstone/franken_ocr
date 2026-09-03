# FrankenOCR for iPhone and iPad

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

App Store archive and upload:

```bash
xcodebuild -project ios/FrankenOCR.xcodeproj -scheme FrankenOCR \
  -configuration Release -destination "generic/platform=iOS" \
  -archivePath "$PWD/ios/build/FrankenOCR.xcarchive" archive

xcodebuild -exportArchive \
  -archivePath "$PWD/ios/build/FrankenOCR.xcarchive" \
  -exportPath "$PWD/ios/build/export" \
  -exportOptionsPlist ios/AppStoreExportOptions.plist
```

The export configuration keeps the checked-in marketing version and build
number authoritative instead of letting App Store Connect renumber the binary.

Version 1.1 also carries the v0.9 tall-capture behavior into the native app:
extreme scrolling screenshots use the same smart-cut strip router as the CLI,
and suspiciously sparse page output is shown as a low-yield warning instead of
quietly looking successful.

The input rail also has two deliberately distinct camera paths. **Camera** takes
one still and sends it through the selected Baidu/FrankenOCR model for maximum
accuracy. **Live Camera** uses Apple's native, on-device Live Text scanner in
its accurate high-frame-rate mode, deduplicates repeated lines, and bends each
actual recognized line from its camera box into a bounded capture tray. This is
an explicit speed/accuracy tradeoff: Apple's scanner makes real-time video
possible but can be less accurate than the full Baidu model. The live surface
labels that implementation and tradeoff in-app; both paths remain offline and
upload nothing.

Finished recognition is also available from the clock button and the compact
workspace's **Library** destination. The Library keeps only the exported text
and minimal provenance in Application Support; it never copies the source image,
PDF, question, or layout coordinates. It is excluded from backup, limited to the
newest 20 results, expires results after 14 days, and provides share, copy,
per-result deletion, and Clear All controls.

`FocrCore.xcframework/` and `FrankenOCR.xcodeproj/` are generated and
gitignored. `project.yml`, `build-rust.sh`, `Sources/`, the entitlements, and
the privacy manifest are the source.

## Why a phone can run the flagship at all

The browser playground gates Unlimited-OCR to desktop for a reason that has
nothing to do with compute: WebAssembly has no mmap. A wasm module must stage
the whole 3.0 GB artifact into linear memory, so 3.0 GB of the measured ~3.6 GB
peak is the blob itself, and no residency trick can move it.

A native app maps the file instead. `focr-ios` therefore loads the engine from a
**path**, not from bytes, so `Weights::load` reaches the mmap island and the
artifact becomes clean, file-backed pages the kernel may evict, rather than the
dirty anonymous heap that jetsam counts and kills for. Two supporting pieces:

- **`MADV_RANDOM`.** A MoE decode routes to 6 of 64 experts per layer, so the
  default sequential read-ahead would fault in neighboring experts the token
  never reads. The advice is what keeps the resident working set near the bytes
  actually used.
- **Streamed vision.** The per-block SAM/CLIP path (already in the engine, and
  gated bit-identical against the cached path) replaces ~1.6 GB of hydrated f32
  tower with tens of MB of scratch. It engages automatically, because it keys off
  the artifact's recipe string, and this app ships that recipe.

The `increased-memory-limit` entitlement is there for the page cache, not for the
heap.

### Measured

One page of `site/assets/sample-doc.png` through the 3.0 GB int4 artifact, Apple
M4 Pro, warm cache, one sample per mode (`docs/NEGATIVE_EVIDENCE.md`,
CLAIM-bd-r9po-ios-mmap-residency):

| | owned bytes | mmap + `MADV_RANDOM` |
|---|---:|---:|
| peak memory footprint (`phys_footprint`) | 3,561,901,056 | **547,081,216** |
| max RSS | 3,574,841,344 | 3,391,455,232 |
| wall | 11.83 s | 11.93 s |

Peak footprint is the dirty-anonymous accounting that jetsam terminates on, and
it falls **6.5×, to 0.55 GB**. Max RSS barely moves because it still counts
clean file-backed pages, and that is the point: those are evictable. Output is
byte-identical across both modes.

This is an M4 Pro measurement, not a phone measurement. It establishes that the
residency argument holds; it says nothing about A-series wall time.

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
  (measured wasm peaks of 3.4 GB for GOT-OCR2, 2.8 GB for SmolVLM2) because the
  streamed residency mode keys off the Unlimited-OCR recipe. Extending it to
  their towers is what earns them a place in the picker; shipping them first
  would put a model on the phone that gets the app killed.
- Cross-page parsing (`--multi-page`), where page N can reference pages 1..N-1.
  The app reads every page of a document, but each page independently.
- Figure extraction and batch mode.
- Final App Store listing metadata, review notes, and device screenshots.
