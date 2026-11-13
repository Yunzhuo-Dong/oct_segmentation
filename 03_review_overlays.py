#!/usr/bin/env python
"""Step 3: draw the MATLAB-derived boundaries on the canonical images.
Picks samples spread over splits, cohorts (BL/MA...) and annotation density.
Writes to $HOME by default because HORSE PNG writes failed before (Errno 5).

python 03_review_overlays.py --data-dir "$WS/octb_data_v1" --out "$HOME/octb_review" --n 16
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from octb.common import BOUNDARIES, load_json, read_csv  # noqa: E402

COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
          "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#ffffff"]


def pick(rows, n, seed):
    rng = np.random.default_rng(seed)
    buckets = {}
    for r in rows:
        key = (r["split"], r.get("experiment", "") or r["animal_id"][:2])
        buckets.setdefault(key, []).append(r)
    chosen = []
    keys = sorted(buckets)
    while len(chosen) < n and any(buckets.values()):
        for k in keys:
            if buckets[k] and len(chosen) < n:
                b = buckets[k]
                b.sort(key=lambda r: int(r["n_boundaries"]))
                j = 0 if len(chosen) % 2 == 0 else len(b) - 1  # alternate sparse / dense
                if rng.random() < 0.3:
                    j = int(rng.integers(len(b)))
                chosen.append(b.pop(j))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/octb_review"))
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None, help="explicit sample_ids")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in read_csv(os.path.join(args.data_dir, "manifest.csv")) if r["split"] in ("train", "val", "test")]
    if args.ids:
        rows = [r for r in rows if r["sample_id"] in set(args.ids)]
    else:
        rows = pick(rows, args.n, args.seed)
    cons = load_json(os.path.join(args.data_dir, "order_constraints.json"))
    os.makedirs(args.out, exist_ok=True)
    for r in rows:
        d = np.load(os.path.join(args.data_dir, r["npz"]))
        img, runs = d["image"], d["runs"]
        H, W = img.shape
        fig, axes = plt.subplots(1, 2, figsize=(2 * W / 100 + 2.5, H / 100 + 0.5), dpi=100)
        for ax in axes:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
        handles = []
        for c, name in enumerate(BOUNDARIES):
            m = runs[:, 0] == c
            if not m.any():
                continue
            x, y = runs[m, 1], runs[m, 2]
            n_per = np.bincount(x.astype(int), minlength=W)
            multi = n_per[x.astype(int)] > 1
            axes[1].scatter(x[~multi], y[~multi], s=0.6, c=COLORS[c], linewidths=0)
            axes[1].scatter(x[multi], y[multi], s=2.5, c=COLORS[c], marker="x", linewidths=0.5)
            handles.append(plt.Line2D([], [], color=COLORS[c], lw=3, label=f"{name} ({m.sum()} runs)"))
        axes[0].set_title(f"{r['sample_id']}\n{r['animal_id']} eye={r['eye']} split={r['split']} {r.get('experiment', '')}", fontsize=8)
        axes[1].set_title(f"orientation={cons['orientation']}  x = multi-curve column", fontsize=8)
        leg = fig.legend(handles=handles, loc="center right", fontsize=7, frameon=True)
        leg.get_frame().set_facecolor("#555555")
        for t in leg.get_texts():
            t.set_color("white")
        fig.subplots_adjust(left=0.01, right=1 - 2.4 / fig.get_figwidth(), top=1 - 0.5 / fig.get_figheight(), bottom=0.01, wspace=0.02)
        p = os.path.join(args.out, f"{r['sample_id']}.png")
        fig.savefig(p)
        plt.close(fig)
        print(p)


if __name__ == "__main__":
    main()
