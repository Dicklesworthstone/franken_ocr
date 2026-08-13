#!/usr/bin/env python3
"""Mirror every manifest asset to the HuggingFace path the manifest already names.

`models/manifest-v2.json` has listed HuggingFace URLs as a fallback for most
assets for a long time — but the repo behind them never existed, so the
advertised redundancy was fiction and a GitHub 503 took everything down with it.
This walks the manifest, downloads each asset from its GitHub URL, verifies it
against the manifest's pinned SHA-256, and uploads it to the exact HF path the
manifest already points at. After this runs, those URLs are real.

Verification is not optional and not after the fact: nothing is uploaded that did
not match its pin, so a corrupt or truncated GitHub response can never become the
mirror everyone falls back to.

    uvx --from huggingface_hub --with hf-transfer \\
        python scripts/hf_mirror_manifest.py --repo Dicklesworthstone/franken_ocr-weights

    --only tromr,onechart    mirror a subset (substring match on the HF path)
    --dry-run                list the plan, transfer nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "models" / "manifest-v2.json"
HF_PREFIX = "https://huggingface.co/"


def collect_assets(node, out):
    """Every dict with a `urls` list is an asset record."""
    if isinstance(node, dict):
        urls = node.get("urls")
        if isinstance(urls, list) and urls:
            gh = next((u for u in urls if "github.com" in u), None)
            hf = next((u for u in urls if "huggingface.co" in u), None)
            sha = node.get("sha256")
            size = node.get("bytes")
            if gh and sha:
                out.append({"gh": gh, "hf": hf, "sha256": sha, "bytes": size})
        for value in node.values():
            collect_assets(value, out)
    elif isinstance(node, list):
        for value in node:
            collect_assets(value, out)
    return out


def hf_path_in_repo(hf_url: str, repo: str) -> str | None:
    """`https://huggingface.co/<repo>/resolve/main/<path>` -> `<path>`."""
    marker = f"{HF_PREFIX}{repo}/resolve/main/"
    return hf_url[len(marker):] if hf_url and hf_url.startswith(marker) else None


def download_verified(url: str, dest: Path, expect_sha: str, expect_bytes: int | None) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and (expect_bytes is None or dest.stat().st_size == expect_bytes):
        digest = sha256_of(dest)
        if digest == expect_sha:
            print(f"    cached and verified")
            return True
        print(f"    cached copy failed digest; refetching")
        dest.unlink()

    digest = hashlib.sha256()
    total = 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "franken_ocr-mirror"})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
        while chunk := response.read(8 * 1024 * 1024):
            handle.write(chunk)
            digest.update(chunk)
            total += len(chunk)
    actual = digest.hexdigest()
    if actual != expect_sha:
        print(f"    DIGEST MISMATCH: {actual} != {expect_sha}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False
    if expect_bytes is not None and total != expect_bytes:
        print(f"    SIZE MISMATCH: {total} != {expect_bytes}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False
    tmp.rename(dest)
    print(f"    downloaded and verified ({total/1e6:.1f} MB)")
    return True


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/franken_ocr/mirror")
    parser.add_argument("--only", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    assets = collect_assets(manifest, [])

    plan = []
    skipped_no_hf = []
    for asset in assets:
        path_in_repo = hf_path_in_repo(asset["hf"] or "", args.repo)
        if not path_in_repo:
            skipped_no_hf.append(asset["gh"].rsplit("/", 1)[-1])
            continue
        if args.only and not any(f in path_in_repo for f in args.only.split(",")):
            continue
        plan.append({**asset, "path_in_repo": path_in_repo})

    print(f"{len(plan)} asset(s) to mirror; {len(skipped_no_hf)} have no HF URL in the manifest")
    for name in skipped_no_hf:
        print(f"  no-hf-url: {name}")
    if not plan:
        return 0

    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()
    try:
        info = api.repo_info(args.repo, repo_type="model", files_metadata=True)
        present = {s.rfilename: getattr(s, "size", None) for s in (info.siblings or [])}
    except RepositoryNotFoundError:
        present = {}

    failures = 0
    for i, item in enumerate(plan, 1):
        name = item["path_in_repo"]
        print(f"[{i}/{len(plan)}] {name}")
        if present.get(name) is not None and present[name] == item["bytes"]:
            print("    already on the Hub at the pinned size")
            continue
        if args.dry_run:
            print(f"    dry-run: would mirror from {item['gh']}")
            continue
        local = args.cache / name
        if not download_verified(item["gh"], local, item["sha256"], item["bytes"]):
            failures += 1
            continue
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=name,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"Mirror {name} (SHA-256 pinned in models/manifest-v2.json)",
        )
        print("    uploaded")

    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
