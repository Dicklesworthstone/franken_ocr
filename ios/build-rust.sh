#!/usr/bin/env bash
# Build the franken_ocr engine for iOS device + simulator + Mac Catalyst and assemble
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
# Device and simulator remain separate XCFramework slices because `lipo` cannot
# hold their two arm64 libraries in one file. Catalyst, by contrast, is one
# platform slice and combines its arm64 and x86_64 libraries with `lipo`.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET_DIR="${CARGO_TARGET_DIR:-target}"
CRATE=focr-ios
LIB=libfocr_ios.a
PROFILE="${FOCR_IOS_PROFILE:-release}"
APPLE_RUST_TOOLCHAIN="${APPLE_RUST_TOOLCHAIN:-nightly-2026-08-25-aarch64-apple-darwin}"
APPLE_CARGO="${APPLE_CARGO:-$(rustup which --toolchain "$APPLE_RUST_TOOLCHAIN" cargo)}"

for target in \
  aarch64-apple-ios \
  aarch64-apple-ios-sim \
  aarch64-apple-ios-macabi \
  x86_64-apple-ios-macabi
do
  if ! rustup target list --toolchain "$APPLE_RUST_TOOLCHAIN" --installed | grep -qx "$target"; then
    echo "==> installing rust target $target"
    rustup target add --toolchain "$APPLE_RUST_TOOLCHAIN" "$target"
  fi
  echo "==> cargo build --$PROFILE -p $CRATE --target $target"
  RUSTUP_TOOLCHAIN="$APPLE_RUST_TOOLCHAIN" RCH_CARGO_WRAPPER_BYPASS=1 \
    "$APPLE_CARGO" build --"$PROFILE" --locked -p "$CRATE" --target "$target"
done

# The ABI header and module contract are checked in together.
HEADERS="$(mktemp -d /tmp/focr-ios-headers.XXXXXX)"
cp "$CRATE/include/focr_ios.h" "$CRATE/include/module.modulemap" "$HEADERS/"

CATALYST_ROOT=$(mktemp -d /tmp/focr-ios-maccatalyst.XXXXXX)
CATALYST_LIB="$CATALYST_ROOT/libfocr_ios.a"
lipo -create \
  "$TARGET_DIR/aarch64-apple-ios-macabi/$PROFILE/$LIB" \
  "$TARGET_DIR/x86_64-apple-ios-macabi/$PROFILE/$LIB" \
  -output "$CATALYST_LIB"

FRAMEWORK=ios/FocrCore.xcframework
OUTPUT_ROOT=$(mktemp -d /tmp/focr-xcframework.XXXXXX)
STAGED_FRAMEWORK="$OUTPUT_ROOT/FocrCore.xcframework"
xcodebuild -create-xcframework \
  -library "$TARGET_DIR/aarch64-apple-ios/$PROFILE/$LIB"     -headers "$HEADERS" \
  -library "$TARGET_DIR/aarch64-apple-ios-sim/$PROFILE/$LIB" -headers "$HEADERS" \
  -library "$CATALYST_LIB" -headers "$HEADERS" \
  -output "$STAGED_FRAMEWORK"

if [[ -e "$FRAMEWORK" ]]; then
  mv "$FRAMEWORK" "$FRAMEWORK.previous-$(date +%Y%m%d-%H%M%S)"
fi
mv "$STAGED_FRAMEWORK" "$FRAMEWORK"

echo
echo "==> $FRAMEWORK"
du -sh "$FRAMEWORK"
