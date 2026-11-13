#!/usr/bin/env python
"""Step 1: print the real structure of a few .mat files and whether their
source images can be found. Paste the output back if anything looks off.

python 01_inspect_mat.py --mat-root "$MAT_BASE" --image-roots "$MAT_BASE" --n 5
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from octb.common import parse_ids  # noqa: E402
from octb.matio import basename_any, index_images, load_annotation, read_image_gray  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat-root", required=True)
    ap.add_argument("--image-roots", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--animal-regex", default=None)
    args = ap.parse_args()

    mats = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(args.mat_root)
                  for f in fs if f.lower().endswith(".mat"))
    print(f"[mats] {len(mats)} .mat files under {args.mat_root}")
    sub = Counter(os.path.relpath(os.path.dirname(m), args.mat_root).split(os.sep)[0] for m in mats)
    print("[mats] by first sub-folder:", dict(sub))
    roots = args.image_roots or [args.mat_root]
    by_name, by_stem = index_images(roots)
    print(f"[images] {sum(len(v) for v in by_name.values())} tif/tiff files under {roots}")

    names = Counter()
    for m in mats[:args.n]:
        print("\n" + "=" * 80 + f"\n{m}")
        try:
            a = load_annotation(m)
        except ValueError as e:
            print("  LOAD FAILED:", e)
            continue
        print(f"  Source.file = {a['source_file']!r}   annotator = {a['annotator']!r}")
        kw = {"animal_re": args.animal_regex} if args.animal_regex else {}
        print("  parsed ids (animal, eye, date) =", parse_ids(a["source_file"] or m, **kw))
        cand = by_name.get(basename_any(a["source_file"]).lower(), [])
        print(f"  image candidates: {cand[:3]}{' ...' if len(cand) > 3 else ''}")
        img_shape = None
        if cand:
            img, info = read_image_gray(cand[0])
            img_shape = img.shape
            print(f"  image: shape={img.shape} info={info} mean={img.mean():.1f}")
        for ld in a["labels"]:
            msk = ld["mask"]
            if msk is None:
                print(f"   - {ld['raw_name']!r:>12} -> {ld['name']}  class={ld['cls']!r}  mask=None")
                continue
            ys, xs = np.nonzero(msk) if msk.ndim == 2 else (np.array([]), np.array([]))
            cols = np.unique(xs).size if xs.size else 0
            yinfo = f"y[{ys.min()}..{ys.max()}] median {np.median(ys):.0f}" if ys.size else "empty"
            same = "" if img_shape is None else ("  SHAPE==IMG" if msk.shape == img_shape else "  SHAPE!=IMG")
            print(f"   - {ld['raw_name']!r:>12} -> {ld['name']}  class={ld['cls']!r}  mask {msk.shape} "
                  f"{msk.dtype} px={int((msk != 0).sum())} cols={cols} {yinfo}{same}")
            names[ld["raw_name"]] += 1
    print("\n[raw label names in inspected files]", dict(names))


if __name__ == "__main__":
    main()
