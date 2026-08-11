#!/usr/bin/env bash
# Build the focr-wasm browser module into site/pkg (the frankentts house style).
#
# Deliberately NOT wasm-pack: wasm-pack injects its own RUSTFLAGS, which
# clobbers the target-feature list. We drive cargo + wasm-bindgen directly.
#
# Current shape: ONE serial (unshared-memory) build with the simd128 baseline.
# simd128 is safe for the whole desktop floor + iOS 16.4+; LLVM autovectorizes
# the scalar kernels under it. There is no threaded build yet — the TrOMR
# demo model is small enough that serial decode is interactive. When the
# Unlimited-OCR browser lane lands, add the threaded variant beside this one
# (shared+imported memory, -Z build-std, and the five TLS exports:
# __heap_base/__tls_base/__tls_size/__tls_align/__wasm_init_tls — the exports,
# not a missing TLS segment, are what wasm-bindgen's threading transform needs)
# and ship BOTH: sharedness is a link-time property and growing a shared
# memory kills iOS Safari tabs at ~2 GB, so WebKit gets the serial module.
#
# Local serving note: `python3 -m http.server` sends no COOP/COEP headers.
# The serial module doesn't need them; the future threaded one will.
set -euo pipefail

cd "$(dirname "$0")/.."

TARGET_DIR="${CARGO_TARGET_DIR:-target}"
WASM="$TARGET_DIR/wasm32-unknown-unknown/release/focr_wasm.wasm"

echo "== cargo build (serial, +simd128) =="
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS="-C target-feature=+simd128 -C link-arg=--max-memory=4294967296" \
    cargo build -p focr-wasm --target wasm32-unknown-unknown --release

echo "== wasm-bindgen -> site/pkg =="
wasm-bindgen "$WASM" --out-dir site/pkg --typescript --target web

# wasm-pack/wasm-bindgen may write a .gitignore that would hide the committed
# artifact from Pages' upload (the 79-commits-stale-artifact failure mode).
rm -f site/pkg/.gitignore

if command -v wasm-opt >/dev/null 2>&1; then
    echo "== wasm-opt -O2 =="
    before=$(wc -c <site/pkg/focr_wasm_bg.wasm)
    wasm-opt -O2 --enable-simd --enable-bulk-memory --enable-mutable-globals \
        --enable-nontrapping-float-to-int \
        site/pkg/focr_wasm_bg.wasm -o site/pkg/focr_wasm_bg.wasm.opt
    mv site/pkg/focr_wasm_bg.wasm.opt site/pkg/focr_wasm_bg.wasm
    after=$(wc -c <site/pkg/focr_wasm_bg.wasm)
    echo "wasm-opt: $before -> $after bytes"
else
    # A missing wasm-opt should cost speed/size, not the build.
    echo "wasm-opt not found; skipping (module still correct, just larger)"
fi

echo "== shipped sizes =="
ls -la site/pkg/focr_wasm_bg.wasm site/pkg/focr_wasm.js
gzip -c site/pkg/focr_wasm_bg.wasm | wc -c | awk '{printf "gzip: %.2f MB\n", $1/1048576}'
