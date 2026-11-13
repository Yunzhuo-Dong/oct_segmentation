#!/usr/bin/env python
"""Step 5: evaluate a checkpoint on val (or test, with --confirm-test).

Reports raw predictions and, with --postprocess, E0-PP (per-column isotonic
ordering). Writes per_boundary.csv, per_pair.csv, per_animal.csv,
per_image.csv, summary.json and overlays.

python 05_evaluate.py --ckpt $WS/octb_runs/E0/best.pt --data-dir D --split val --out $WS/octb_eval/E0_val --postprocess
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from octb.common import BIDX, BOUNDARIES, load_json, read_csv, save_json, write_csv, write_test  # noqa: E402
from octb.dataset import BoundaryDataset  # noqa: E402
from octb.losses import decode  # noqa: E402
from octb.metrics import Aggregator, sample_geom, w1_columns  # noqa: E402
from octb.model import load_checkpoint  # noqa: E402
from octb.postprocess import isotonic_columns  # noqa: E402

COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
          "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#ffffff"]


def overlay(path, img, gt_runs, preds, conf, thr, titles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = img.shape
    n = 1 + len(preds)
    fig, axes = plt.subplots(1, n, figsize=(n * W / 100, H / 100 + 0.4), dpi=100)
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
    for c in range(len(BOUNDARIES)):
        m = gt_runs[:, 0] == c
        axes[0].scatter(gt_runs[m, 1], gt_runs[m, 2], s=0.6, c=COLORS[c], linewidths=0)
    axes[0].set_title("GT (MATLAB)", fontsize=8)
    xs = np.arange(W)
    for ax, y, t in zip(axes[1:], preds, titles):
        for c in range(len(BOUNDARIES)):
            yy = np.where(conf[c] > thr, y[c], np.nan)
            ax.plot(xs, yy, lw=0.8, c=COLORS[c])
        ax.set_title(t, fontsize=8)
    fig.subplots_adjust(0, 0, 1, 1 - 0.4 / fig.get_figheight(), wspace=0.01)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--confirm-test", action="store_true",
                    help="required for --split test: test is for the FINAL comparison only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--postprocess", action="store_true")
    ap.add_argument("--conf-thr", type=float, default=None, help="default: training gate threshold")
    ap.add_argument("--jump-px", type=float, default=10.0)
    ap.add_argument("--n-overlays", type=int, default=12)
    ap.add_argument("--save-pred", action="store_true")
    args = ap.parse_args()
    if args.split == "test" and not args.confirm_test:
        sys.exit("Refusing to evaluate on test without --confirm-test (do not tune on test).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(args.ckpt, device)
    targs = ck["args"]
    thr = args.conf_thr if args.conf_thr is not None else targs["gate_thr"]
    win = targs["gate_win"]
    os.makedirs(args.out, exist_ok=True)
    write_test(args.out)
    os.makedirs(os.path.join(args.out, "overlays"), exist_ok=True)

    from octb.common import sha256_file
    msha = sha256_file(os.path.join(args.data_dir, "manifest.csv"))
    if msha != ck.get("manifest_sha256"):
        print("[WARN] manifest differs from the one used for training -> results not comparable!")

    rows = [r for r in read_csv(os.path.join(args.data_dir, "manifest.csv")) if r["split"] == args.split]
    cons = load_json(os.path.join(args.data_dir, "order_constraints.json"))
    pairs = [(BIDX[a], BIDX[b]) for a, b in cons["pairs"]]
    chain = [BIDX[c] for c in cons["chain"] if c not in cons["excluded_from_constraints"]]
    ds = BoundaryDataset(rows, args.data_dir, train=False, sigma=targs["sigma"])

    use_amp = targs.get("amp") == "bf16" and device.type == "cuda"
    agg_raw, agg_pp = Aggregator(pairs), Aggregator(pairs)
    per_animal = defaultdict(lambda: (Aggregator(pairs), Aggregator(pairs)))
    per_image = []
    ov_idx = set(np.linspace(0, len(ds) - 1, min(args.n_overlays, len(ds))).astype(int).tolist()) if len(ds) else set()
    pred_dir = os.path.join(args.out, "pred")
    if args.save_pred:
        os.makedirs(pred_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            x = b["image"][None].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.autocast("cpu", enabled=False):
                logits = model(x)
            yhat, conf, p = decode(logits, win)
            tgt, val = b["target"][None].to(device), b["valid"][None].to(device)
            w1 = w1_columns(p, tgt, val)[0]
            w1s, w1n = w1.sum(-1).cpu().numpy(), val[0].float().sum(-1).cpu().numpy()
            y = yhat[0].cpu().numpy(); cf = conf[0].cpu().numpy()
            gt, sg = b["gt_y"].numpy(), b["single"].numpy()
            g = sample_geom(y, cf, gt, sg, pairs, thr, jump_px=args.jump_px)
            agg_raw.add(g, w1s, w1n)
            r = rows[i]
            pa = per_animal[r["animal_id"]]
            pa[0].add(g, w1s, w1n)
            rec = dict(sample_id=r["sample_id"], animal_id=r["animal_id"],
                       w1=float(np.nansum(w1s) / max(w1n.sum(), 1)),
                       mae=float(np.mean(np.concatenate([a for a in g["ae"] if a.size]))) if any(a.size for a in g["ae"]) else float("nan"),
                       crossings_conf=int(sum(v[1] for v in g["cross_conf"])))
            ypp = None
            if args.postprocess:
                ypp = isotonic_columns(y, cf, chain, thr)
                gp = sample_geom(ypp, cf, gt, sg, pairs, thr, jump_px=args.jump_px)
                agg_pp.add(gp)
                pa[1].add(gp)
                rec["crossings_conf_pp"] = int(sum(v[1] for v in gp["cross_conf"]))
            per_image.append(rec)
            if args.save_pred:
                np.savez_compressed(os.path.join(pred_dir, r["sample_id"] + ".npz"), y=y, conf=cf,
                                    **({"y_pp": ypp} if ypp is not None else {}))
            if i in ov_idx:
                img, runs = ds.load_raw(i)
                preds, titles = [y], ["prediction (conf>thr)"]
                if ypp is not None:
                    preds.append(ypp); titles.append("prediction + isotonic order (E0-PP)")
                overlay(os.path.join(args.out, "overlays", r["sample_id"] + ".png"), img, runs, preds, cf, thr, titles)

    s_raw = agg_raw.summary()
    s_pp = agg_pp.summary() if args.postprocess else None
    pb = []
    for n in BOUNDARIES:
        d = dict(boundary=n, **s_raw["per_boundary"][n])
        if s_pp:
            d.update({f"pp_{k}": v for k, v in s_pp["per_boundary"][n].items() if k in ("mae", "median_ae", "p90_ae", "jump_rate")})
        pb.append(d)
    write_csv(pb, os.path.join(args.out, "per_boundary.csv"))
    pp_rows = []
    for k, d in s_raw["pairs"].items():
        e = dict(pair=k, **d)
        if s_pp:
            e.update({f"pp_{kk}": vv for kk, vv in s_pp["pairs"][k].items()})
        pp_rows.append(e)
    write_csv(pp_rows, os.path.join(args.out, "per_pair.csv"))
    an = []
    for a, (ar, app) in sorted(per_animal.items()):
        sr = ar.summary()
        e = dict(animal_id=a, macro_w1=sr["macro_w1"], macro_mae=sr["macro_mae"],
                 crossing_rate_conf=sr["crossing_rate_conf"], jump_rate=sr["jump_rate"])
        if args.postprocess:
            sp = app.summary()
            e.update(pp_macro_mae=sp["macro_mae"], pp_crossing_rate_conf=sp["crossing_rate_conf"])
        an.append(e)
    write_csv(an, os.path.join(args.out, "per_animal.csv"))
    write_csv(per_image, os.path.join(args.out, "per_image.csv"))
    summary = dict(ckpt=args.ckpt, ckpt_epoch=ck["epoch"], split=args.split, n_images=len(ds), conf_thr=thr,
                   units="canonical-grid pixels (not micrometres)",
                   raw={k: v for k, v in s_raw.items() if k not in ("per_boundary", "pairs")},
                   postprocessed={k: v for k, v in s_pp.items() if k not in ("per_boundary", "pairs")} if s_pp else None,
                   animal_macro_w1_mean=float(np.nanmean([e["macro_w1"] for e in an])) if an else float("nan"),
                   animal_macro_w1_std=float(np.nanstd([e["macro_w1"] for e in an])) if an else float("nan"))
    save_json(summary, os.path.join(args.out, "summary.json"))
    print(f"[eval] {args.split}: n={len(ds)}  macro W1 {s_raw['macro_w1']:.3f}px  macro MAE {s_raw['macro_mae']:.3f}px  "
          f"cross(gt) {s_raw['crossing_rate_gt']:.4f}  cross(conf) {s_raw['crossing_rate_conf']:.4f}  "
          f"jumps {s_raw['jump_rate']:.4f}")
    if s_pp:
        print(f"[eval] E0-PP: macro MAE {s_pp['macro_mae']:.3f}px  cross(conf) {s_pp['crossing_rate_conf']:.4f}")
    print("[done]", args.out)


if __name__ == "__main__":
    main()
