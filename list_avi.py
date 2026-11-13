#!/usr/bin/env python
"""List AVI files, the split of their animal, and the matching training TIFF.
python list_avi.py [--split test]          (uses $DATA and searches $WS)
"""
import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from octb.common import parse_ids  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.environ.get("WS", "."))
ap.add_argument("--data-dir", default=os.environ.get("DATA", ""))
ap.add_argument("--split", default=None, help="only show AVIs whose animal is in this split")
args = ap.parse_args()

rows = list(csv.DictReader(open(os.path.join(args.data_dir, "manifest.csv"))))
split_of = {}
for r in rows:
    split_of.setdefault(r["animal_id"], Counter())[r["split"]] += 1
avis = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(args.root) for f in fs if f.lower().endswith(".avi"))
n = 0
for a in avis:
    animal, eye, _ = parse_ids(os.path.basename(a))
    sp = dict(split_of.get(animal, {}))
    if args.split and args.split not in sp:
        continue
    stem = os.path.splitext(os.path.basename(a))[0]
    tifs = [r["image_path"] for r in rows if os.path.splitext(os.path.basename(r["image_path"]))[0] == stem
            or os.path.basename(r["image_path"]).startswith(stem + "_")]
    n += 1
    print(f"[{n}] {animal or '?'} eye={eye} splits={sp or 'not in dataset'} matching_tiffs={len(tifs)}")
    print(f"     AVI={a}")
    if tifs:
        print(f"     TIF={tifs[0]}")
print(f"{n} AVI(s) listed of {len(avis)} found under {args.root}")
