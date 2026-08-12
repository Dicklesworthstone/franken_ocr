#!/usr/bin/env bash
# Build the franken_ocr engine for iOS device + simulator and assemble
# FocrCore.xcframework.
#
# Run before the first Xcode build and after any Rust change:
#
#   ios/build-rust.sh
#
# Deliberately NOT wired into an Xcode Run Script phase: this is a fat-LTO
# release build of the whole engine, which is not something anyone wants on
# every Cmd-R. The xcframework is a build artifact and is gitignored.
#
# Two cargo targets, no lipo. `lipo` cannot hold two arm64 slices in one file,
# and device and simulator are both arm64 — which is precisely the problem the
# xcframework format exists to solve.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET_DIR="${CARGO_TARGET_DIR:-target}"
CRATE=focr-ios
LIB=libfocr_ios.a
PROFILE="${FOCR_IOS_PROFILE:-release}"

for target in aarch64-apple-ios aarch64-apple-ios-sim; do
  if ! rustup target list --installed | grep -qx "$target"; then
    echo "==> installing rust target $target"
    rustup target add "$target"
  fi
  echo "==> cargo build --$PROFILE -p $CRATE --target $target"
  cargo build --"$PROFILE" --locked -p "$CRATE" --target "$target"
done

# The header is checked in; the modulemap is generated, because its only job is
# to name the header and it has no reason to drift.
HEADERS="$(mktemp -d /tmp/focr-ios-headers.XXXXXX)"
trap 'rm -rf "$HEADERS"' EXIT
cp "$CRATE/include/focr_ios.h" "$HEADERS/"
cat > "$HEADERS/module.modulemap" <<'EOF'
module FocrCore {
    header "focr_ios.h"
    export *
}
EOF

FRAMEWORK=ios/FocrCore.xcframework
rm -rf "$FRAMEWORK"
xcodebuild -create-xcframework \
  -library "$TARGET_DIR/aarch64-apple-ios/$PROFILE/$LIB"     -headers "$HEADERS" \
  -library "$TARGET_DIR/aarch64-apple-ios-sim/$PROFILE/$LIB" -headers "$HEADERS" \
  -output "$FRAMEWORK"

echo
echo "==> $FRAMEWORK"
du -sh "$FRAMEWORK"
