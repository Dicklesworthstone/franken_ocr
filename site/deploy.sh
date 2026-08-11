#!/usr/bin/env bash
# Deploy site/ to Cloudflare Pages with per-deploy cache stamping.
#
# Every file that CONTAINS a versioned URL must be stamped, not just the entry
# points — engine-worker.js imports ./pkg/focr_wasm.js?v=@SITEV@ and would
# otherwise ship the literal placeholder. The stamp defeats CDN zone TTL
# overrides that ignore origin no-cache headers (a measured 4-hour-stale-script
# incident on the sibling project).
set -euo pipefail

cd "$(dirname "$0")"

# BOTH lanes must ship together: engine-worker.js picks site/pkg-threaded at
# runtime on cross-origin-isolated Blink, so a deploy carrying only site/pkg
# would 404 the module for every desktop Chrome visitor.
for lane in pkg pkg-threaded; do
    if [ ! -f "$lane/focr_wasm_bg.wasm" ]; then
        echo "site/$lane is missing — run site/build.sh first" >&2
        exit 1
    fi
    # Freshness gate: refuse to ship a wasm artifact older than the crate source.
    newest_src=$(find ../focr-wasm/src ../src -name '*.rs' -newer "$lane/focr_wasm_bg.wasm" | head -1)
    if [ -n "$newest_src" ]; then
        echo "STALE ARTIFACT: $newest_src is newer than site/$lane/focr_wasm_bg.wasm — run site/build.sh" >&2
        exit 1
    fi
done

VERSION="$(git rev-parse --short HEAD)-$(date +%s)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

rsync -a --exclude build.sh --exclude deploy.sh --exclude 'harness' . "$STAGE/"

for f in "$STAGE"/index.html "$STAGE"/app.js "$STAGE"/engine-worker.js "$STAGE"/reset.html; do
    perl -pi -e "s/\\@SITEV\\@/$VERSION/g" "$f"
done

(cd "$STAGE" && wrangler pages deploy . --project-name frankenocr --branch main --commit-dirty=true)
echo "deployed with stamp $VERSION"
