"""Print v1 and v2 single-page markdown for one item substring."""
import json
import pathlib
import sys

runs = pathlib.Path(sys.argv[1])
needle = sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 1500


def pages(doc):
    if isinstance(doc, list):
        for d in doc:
            yield from pages(d)
    elif isinstance(doc, dict):
        if "markdown" in doc and any(isinstance(v, str) and needle in v for v in doc.values()):
            yield doc
        else:
            for v in doc.values():
                if isinstance(v, (list, dict)):
                    yield from pages(v)


for cell in ("v1", "v2"):
    for f in sorted((runs / "single" / cell).glob("chunk*.json")):
        try:
            doc = json.loads(f.read_text())
        except Exception:
            continue
        for p in pages(doc):
            md = p["markdown"]
            meta = {k: v for k, v in p.items() if k != "markdown" and not isinstance(v, (list, dict))}
            print(f"===== {cell} {f.name} {meta}")
            print(md[:limit])
            if len(md) > limit:
                print(f"... [{len(md)} chars total]")
