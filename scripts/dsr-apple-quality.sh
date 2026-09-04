#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root/ios"

build_root="${FRANKEN_APPLE_BUILD_ROOT:-${DSR_QUALITY_RUN_DIR:-$repo_root/ios/build/dsr-apple-quality}}"
mkdir -p "$build_root"
sbh check --need 20G "$build_root"
command -v xcodegen >/dev/null
xcodegen generate --spec project.yml
git diff --exit-code -- FrankenOCR.xcodeproj Sources/Info.plist
git ls-files -z -- '*.swift' | xargs -0 xcrun swiftc -parse -enable-bare-slash-regex
plutil -lint Sources/Info.plist
plutil -lint Sources/PrivacyInfo.xcprivacy
/Users/jemanuel/.local/bin/ensure-simulator-audio-safe prepare
xcodebuild -project FrankenOCR.xcodeproj -scheme FrankenOCR \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath "$build_root/derived-data" \
  CODE_SIGNING_ALLOWED=NO build
xcodebuild -project FrankenOCR.xcodeproj -scheme FrankenOCR \
  -destination 'platform=macOS,variant=Mac Catalyst' \
  -derivedDataPath "$build_root/derived-data" \
  CODE_SIGNING_ALLOWED=NO test -only-testing:FrankenOCRTests

simulator_id="${FOCR_IOS_SIMULATOR_ID:-}"
if [[ -z "$simulator_id" ]]; then
  simulator_id="$({ xcrun simctl list devices available || true; } | awk -F '[()]' '
    /iPhone/ && /\(Booted\)$/ { print $2; found = 1; exit }
    /iPhone/ && fallback == "" { fallback = $2 }
    END { if (!found) print fallback }
  ')"
fi
if [[ -z "$simulator_id" ]]; then
  echo "No available iPhone Simulator for FrankenOCR UI tests" >&2
  exit 1
fi

# The gate may boot a currently shut-down device. Re-prove the audio fence
# immediately before that simulator action and preserve every existing window.
/Users/jemanuel/.local/bin/ensure-simulator-audio-safe prepare
xcodebuild -project FrankenOCR.xcodeproj -scheme FrankenOCR \
  -destination "platform=iOS Simulator,id=$simulator_id" \
  -derivedDataPath "$build_root/derived-data" \
  CODE_SIGNING_ALLOWED=NO test -only-testing:FrankenOCRUITests
