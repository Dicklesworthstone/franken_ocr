#!/usr/bin/env python3
"""Publish (or refresh) the franken_ocr weight mirror on Hugging Face.

GitHub release assets 503 under load and cap a single asset at 2 GiB. Hugging
Face serves the same bytes from a CDN, honors ranged requests, reflects CORS to
the page origin (so a browser can fetch a model without a proxy), and has no
per-file cap — which means the 3.0 GB artifact ships WHOLE instead of as byte
split parts.

Idempotent: it verifies every local file against its pinned SHA-256 before
uploading, skips a file already present on the Hub with the same size, and never
uploads bytes it could not verify.

    uvx --from huggingface_hub --with hf-transfer \\
        python scripts/hf_publish_weights.py --repo Dicklesworthstone/franken_ocr-weights

    # see what it would do, upload nothing
    python scripts/hf_publish_weights.py --repo ... --dry-run

Requires a token with write access (`hf auth login`).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# The pinned contract. These are the same numbers `site/model-manifest.js` and
# `models/manifest-v2.json` carry; a local file that does not match them is a
# corrupt copy and must never be published as if it were the real artifact.
PINNED: dict[str, tuple[int, str]] = {
    "unlimited-ocr.wasm-int4.focrq": (
        3_003_988_117,
        "2653831ccd7f481f898f80ae5c95fa1ec7ee2a5a18005d3c927ddf64ed75e187",
    ),
    "tokenizer.json": (
        9_979_544,
        "a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4",
    ),
}

DEFAULT_SOURCE = Path.home() / ".cache/franken_ocr/models/ios-wasm-int4"
CARD = Path(__file__).resolve().parent.parent / "models" / "HF_MODEL_CARD.md"


def sha256_of(path: Path) -> str:
    """Digest a multi-gigabyte file without reading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/name on the Hub")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--private",
        action="store_true",
        help="create the repo private (a private mirror serves 401 to everyone, "
        "so the fallback it advertises does not exist — public is the point)",
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()
    who = api.whoami()
    print(f"authenticated as {who.get('name')}")

    # ── verify locally, before anything is created or uploaded ──────────────
    staged: list[tuple[Path, str]] = []
    for name, (size, digest) in PINNED.items():
        path = args.source / name
        if not path.is_file():
            print(f"MISSING  {name}: not at {path}", file=sys.stderr)
            return 2
        actual_size = path.stat().st_size
        if actual_size != size:
            print(
                f"SIZE     {name}: {actual_size} != pinned {size}", file=sys.stderr
            )
            return 2
        print(f"hashing  {name} ({actual_size/1e9:.2f} GB)…", flush=True)
        actual = sha256_of(path)
        if actual != digest:
            print(f"DIGEST   {name}: {actual} != pinned {digest}", file=sys.stderr)
            return 2
        print(f"verified {name}")
        staged.append((path, name))

    if not CARD.is_file():
        print(f"MISSING  model card at {CARD}", file=sys.stderr)
        return 2

    # ── repo ────────────────────────────────────────────────────────────────
    try:
        info = api.repo_info(args.repo, repo_type="model")
        present = {s.rfilename: s for s in (info.siblings or [])}
        print(f"repo exists (private={info.private}), {len(present)} file(s)")
        if info.private and not args.private:
            print(
                "NOTE: repo is PRIVATE. A private mirror answers 401 to every "
                "anonymous download, so it cannot serve as a fallback. Flip it "
                "to public in the repo settings."
            )
    except RepositoryNotFoundError:
        present = {}
        print(f"repo {args.repo} does not exist yet")
        if args.dry_run:
            print("dry-run: would create it")
        else:
            api.create_repo(
                args.repo, repo_type="model", private=args.private, exist_ok=True
            )
            print(f"created {args.repo} (private={args.private})")

    # ── upload ──────────────────────────────────────────────────────────────
    plan = [(CARD, "README.md")] + staged
    for path, name in plan:
        existing = present.get(name)
        if existing is not None and getattr(existing, "size", None) == path.stat().st_size:
            print(f"skip     {name} (already on the Hub at the same size)")
            continue
        if args.dry_run:
            print(f"dry-run: would upload {name} ({path.stat().st_size} bytes)")
            continue
        print(f"upload   {name}…", flush=True)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"Add {name} (SHA-256 pinned in models/manifest-v2.json)",
        )
        print(f"uploaded {name}")

    base = f"https://huggingface.co/{args.repo}/resolve/main"
    print("\nresolve URLs:")
    for _, name in staged:
        print(f"  {base}/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
