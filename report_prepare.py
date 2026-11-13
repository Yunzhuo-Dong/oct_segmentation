#!/usr/bin/env python
"""Print a short report of a prepared dataset (paste the output back).
python report_prepare.py            (uses $DATA)
"""
import csv
import json
import os
import sys
from collections import Counter

D = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DATA", "")
if not D or not os.path.exists(os.path.join(D, "summary.json")):
    sys.exit(f"no summary.json in {D!r}")
log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "prepare.log")
if os.path.exists(log):
    print("===== prepare.log (tail)")
    print("".join(open(log).readlines()[-12:]))
d = json.load(open(os.path.join(D, "summary.json")))
print("===== summary")
for k in ("split_counts", "n_animals", "overlap", "unparsed_animal_images", "orientation", "chain",
          "n_constraint_pairs", "experiments"):
    print(f"{k}: {d.get(k)}")
rows = list(csv.DictReader(open(os.path.join(D, "manifest.csv"))))
print("image_note:", dict(Counter(r.get("image_note", "") for r in rows if r.get("image_note"))))
multi = {}
for a in sorted({r["animal_id"] for r in rows}):
    s = Counter(r["split"] for r in rows if r["animal_id"] == a)
    if len([k for k in s if k in ("train", "val", "test")]) > 1:
        multi[a] = dict(s)
print("animals in more than one split (must be empty):", multi)
print("MA-N0xx animals:")
for a in sorted({r["animal_id"] for r in rows if r["animal_id"].startswith("MA-N0")}):
    print("  ", a, dict(Counter(r["split"] for r in rows if r["animal_id"] == a)))
print("===== skipped")
for r in csv.DictReader(open(os.path.join(D, "skipped.csv"))):
    print("  ", r["reason"][:110], "|", os.path.relpath(r["mat_path"], d["args"]["mat_root"]))
