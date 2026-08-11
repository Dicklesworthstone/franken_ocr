#!/usr/bin/env bash
# Build the focr-wasm browser modules: BOTH lanes, every time.
#
#   site/pkg           serial, unshared memory, +simd128       (the floor)
#   site/pkg-threaded  shared memory + atomics, rayon workers  (the fast lane)
#
# Deliberately NOT wasm-pack: wasm-pack injects its own RUSTFLAGS, which
# clobbers the target-feature list. If `+atomics` does not reach std, LLD emits
# no `__wasm_init_tls` and wasm-bindgen's threading transform fails. We drive
# cargo + wasm-bindgen directly so the flags we write are the flags that link.
#
# WHY TWO MODULES AND NOT ONE: sharedness is a LINK-TIME property. A module
# linked `--shared-memory` cannot run where `SharedArrayBuffer` is unavailable
# or unwise, and growing a shared memory toward 2 GB kills iOS Safari tabs. So
# WebKit and any non-cross-origin-isolated context get `site/pkg`; Blink with
# `crossOriginIsolated` gets `site/pkg-threaded`. engine-worker.js owns that
# allow-list decision at runtime and reports which one it loaded.
#
# THE FIVE TLS EXPORTS are the whole trick of the threaded link. The classic
# "failed to find `__wasm_init_tls`" from wasm-bindgen is a missing EXPORT, not
# a missing TLS segment: LLD keeps those symbols internal unless asked.
#
# Local serving note: `python3 -m http.server` sends no COOP/COEP headers, so it
# can only exercise the serial lane. `site/harness/serve.mjs` parses the shipped
# `_headers` and does set them (`crossOriginIsolated === true`).
set -euo pipefail

cd "$(dirname "$0")/.."

TARGET_DIR="${CARGO_TARGET_DIR:-target}"
WASM_SERIAL="$TARGET_DIR/wasm32-unknown-unknown/release/focr_wasm.wasm"
# The threaded lane links with different RUSTFLAGS, which invalidates every
# fingerprint in a shared target dir — a separate dir keeps BOTH caches warm
# instead of making each build evict the other.
THREADED_TARGET_DIR="$TARGET_DIR/wasm-threaded"
WASM_THREADED="$THREADED_TARGET_DIR/wasm32-unknown-unknown/release/focr_wasm.wasm"

# wasm-pack/wasm-bindgen may write a .gitignore that would hide the committed
# artifact from Pages' upload (the 79-commits-stale-artifact failure mode).
strip_gitignore() {
    rm -f "$1/.gitignore"
}

# $1 = pkg dir, $2 = extra wasm-opt flags
run_wasm_opt() {
    local dir="$1"
    shift
    if ! command -v wasm-opt >/dev/null 2>&1; then
        # A missing wasm-opt should cost speed/size, not the build.
        echo "wasm-opt not found; skipping (module still correct, just larger)"
        return
    fi
    echo "== wasm-opt -O2 ($dir) =="
    local before after
    before=$(wc -c <"$dir/focr_wasm_bg.wasm")
    wasm-opt -O2 --enable-simd --enable-bulk-memory --enable-mutable-globals \
        --enable-nontrapping-float-to-int "$@" \
        "$dir/focr_wasm_bg.wasm" -o "$dir/focr_wasm_bg.wasm.opt"
    mv "$dir/focr_wasm_bg.wasm.opt" "$dir/focr_wasm_bg.wasm"
    after=$(wc -c <"$dir/focr_wasm_bg.wasm")
    echo "wasm-opt: $before -> $after bytes"
}

# ── lane 1: serial ──────────────────────────────────────────────────────────
echo "== cargo build (serial, +simd128) =="
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS="-C target-feature=+simd128 -C link-arg=--max-memory=4294967296" \
    cargo build -p focr-wasm --target wasm32-unknown-unknown --release

echo "== wasm-bindgen -> site/pkg =="
wasm-bindgen "$WASM_SERIAL" --out-dir site/pkg --typescript --target web
strip_gitignore site/pkg
run_wasm_opt site/pkg

# ── lane 2: threaded ────────────────────────────────────────────────────────
# `-Z build-std=std,panic_abort` rebuilds std WITH `+atomics`; the prebuilt
# wasm32-unknown-unknown std is compiled without it, and mixing the two is
# exactly how `__wasm_init_tls` goes missing. Nightly-only, hence the pinned
# toolchain from rust-toolchain.toml.
#
# RUSTFLAGS (not CARGO_TARGET_..._RUSTFLAGS) so the flags reach the build-std
# units too. `--target` is passed, so these never touch host build scripts.
THREADED_RUSTFLAGS="-C target-feature=+simd128,+atomics,+bulk-memory,+mutable-globals"
THREADED_RUSTFLAGS="$THREADED_RUSTFLAGS -C link-arg=--shared-memory"
THREADED_RUSTFLAGS="$THREADED_RUSTFLAGS -C link-arg=--import-memory"
THREADED_RUSTFLAGS="$THREADED_RUSTFLAGS -C link-arg=--max-memory=4294967296"
for sym in __heap_base __tls_base __tls_size __tls_align __wasm_init_tls; do
    # One `-C link-arg=` per export: rustc's link-arg takes exactly ONE argument,
    # so packing several `--export=`s into one flag silently drops all but the
    # first (and then the threading transform fails with a confusing message).
    THREADED_RUSTFLAGS="$THREADED_RUSTFLAGS -C link-arg=--export=$sym"
done

echo "== cargo build (threaded, +simd128,+atomics; -Z build-std) =="
RUSTFLAGS="$THREADED_RUSTFLAGS" \
    cargo build -p focr-wasm --features threads \
    --target wasm32-unknown-unknown --release \
    --target-dir "$THREADED_TARGET_DIR" \
    -Z build-std=std,panic_abort

# Fail LOUDLY here rather than let wasm-bindgen emit its famously unhelpful
# "failed to find `__wasm_init_tls`". `node` reads the export section itself, so
# this is the module's own answer, not a flag we hoped took effect.
echo "== threaded link check (the five TLS exports) =="
node -e '
const fs = require("fs");
const mod = new WebAssembly.Module(fs.readFileSync(process.argv[1]));
const have = new Set(WebAssembly.Module.exports(mod).map((e) => e.name));
const want = ["__heap_base", "__tls_base", "__tls_size", "__tls_align", "__wasm_init_tls"];
const missing = want.filter((s) => !have.has(s));
if (missing.length) {
    console.error(`threaded link is missing exports: ${missing.join(", ")} — refusing to ship it`);
    process.exit(1);
}
console.log(`exports present: ${want.join(" ")}`);
' "$WASM_THREADED"

echo "== wasm-bindgen -> site/pkg-threaded =="
wasm-bindgen "$WASM_THREADED" --out-dir site/pkg-threaded --typescript --target web
strip_gitignore site/pkg-threaded
# `--enable-threads` is not optional here: without it wasm-opt rejects the
# shared memory outright.
run_wasm_opt site/pkg-threaded --enable-threads --enable-bulk-memory-opt

echo "== shipped sizes =="
for dir in site/pkg site/pkg-threaded; do
    ls -la "$dir/focr_wasm_bg.wasm" "$dir/focr_wasm.js"
    gzip -c "$dir/focr_wasm_bg.wasm" | wc -c |
        awk -v d="$dir" '{printf "%s gzip: %.2f MB\n", d, $1/1048576}'
done
