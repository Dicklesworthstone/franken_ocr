"""Score chart readings against truth.json.

usage: score_probe.py <truth.json> <readings dir>

A chart is right when some series in the reading holds exactly the true A-D values.
GOT nests a series under a name it guesses, so the flat and the nested shape both
count. "one series" counts the right readings that invented nothing beside the truth:
a reading with an extra series has the right numbers in it, and a reader cannot tell
them from the invented ones. A reading that is not JSON is wrong.
"""
import json
import sys
from pathlib import Path

truth = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])


def series_of(reading):
    values = reading.get("values", {})
    if values and all(isinstance(v, dict) for v in values.values()):
        return list(values.values())
    return [values]


def score(names, suffix):
    right = clean = read = 0
    wrong = []
    for name in names:
        path = out / f"{name}{suffix}.out"
        if not path.exists():
            continue
        read += 1
        try:
            reading = json.loads(path.read_text())
            series = series_of(reading) if isinstance(reading, dict) else [{}]
        except json.JSONDecodeError:
            wrong.append(name + "!")
            continue
        hit = any(s == truth["charts"][name]["values"] for s in series)
        right += hit
        clean += hit and len(series) == 1
        if not hit:
            wrong.append(name)
    return right, clean, read, wrong


groups = {
    "3:2": [n for n in truth["charts"] if n.startswith("chart")],
    "16:9": [n for n in truth["charts"] if n.startswith("wide")],
    "3:4": [n for n in truth["charts"] if n.startswith("tall")],
}
for label, names in groups.items():
    for suffix in ("", "_sq"):
        right, clean, read, wrong = score(names, suffix)
        if read:
            print(f"{label:5} right {right:2}/{read:2}  one series {clean:2}  wrong {' '.join(wrong)}")
