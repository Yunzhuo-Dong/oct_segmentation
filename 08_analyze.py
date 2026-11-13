#!/usr/bin/env python
"""Step 8: error analysis of trained models on val (CPU is fine).

For each model (and its isotonic post-processed variant "-PP"):
  * single-curve errors per image x boundary (mean, median, P90, P99, share >10 px)
  * CHOROID evaluated as TWO curves: top-2 peaks per column, every GT curve matched
    to the nearest predicted peak (gt->pred), every confident peak matched to the
    nearest GT curve (pred->gt), and whether the number of peaks is right
  * macro MAE with and without CHOROID
  * per experiment (cohort) and per animal tables
  * tail analysis: how much of the total error comes from the worst images
  * paired bootstrap over animals for every pair of variants
  * overlays of the worst images for selected boundaries

python 08_analyze.py --data-dir "$DATA" --models E0=$RUNS/E0/best.pt E1=$RUNS/E1/best.pt --out $RUNS/analysis_val
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
from octb.model import load_checkpoint  # noqa: E402
from octb.postprocess import isotonic_columns  # noqa: E402

COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
          "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#ffffff"]
CH = BIDX["CHOROID"]


def local_peak(p, idx, win):
    """p: (H,W); idx: (W,) -> local soft-argmax y and local mass around idx."""
    H = p.shape[0]
    off = torch.arange(-win, win + 1).view(-1, 1)
    ii = idx.view(1, -1) + off
    inb = (ii >= 0) & (ii < H)
    ii_c = ii.clamp(0, H - 1)
    pw = torch.gather(p, 0, ii_c) * inb
    m = pw.sum(0)
    return (pw * ii_c.float()).sum(0) / m.clamp_min(1e-8), m


def two_peaks(p, sep=6, win=5):
    """Top-2 separated peaks per column of p (H,W)."""
    H = p.shape[0]
    i1 = p.argmax(0)
    y1, m1 = local_peak(p, i1, win)
    rows = torch.arange(H).view(-1, 1)
    p2 = p.masked_fill((rows - i1.view(1, -1)).abs() <= sep, 0.0)
    i2 = p2.argmax(0)
    y2, m2 = local_peak(p, i2, win)
    return y1.numpy(), m1.numpy(), y2.numpy(), m2.numpy()


def choroid_eval(runs, y1, m1, y2, m2, min_mass2):
    """runs: GT runs of all boundaries for one image. Returns dict of arrays."""
    r = runs[runs[:, 0] == CH]
    if len(r) == 0:
        return None
    x = r[:, 1].astype(int)
    g = r[:, 2]
    sig2 = m2 >= min_mass2
    d1 = np.abs(g - y1[x])
    d2 = np.where(sig2[x], np.abs(g - y2[x]), np.inf)
    gt2pred = np.minimum(d1, d2)
    W = y1.shape[0]
    n_gt = np.bincount(x, minlength=W)
    cols = np.flatnonzero(n_gt > 0)
    best1 = np.full(W, np.inf)
    best2 = np.full(W, np.inf)
    np.minimum.at(best1, x, np.abs(y1[x] - g))
    np.minimum.at(best2, x, np.abs(y2[x] - g))
    pred2gt = np.concatenate([best1[cols], best2[cols][sig2[cols]]])
    n_pred = 1 + sig2[cols].astype(int)
    return dict(gt2pred=gt2pred, pred2gt=pred2gt,
                count_ok=(np.minimum(n_gt[cols], 2) == n_pred), n_gt=n_gt[cols], n_pred=n_pred)


def stats(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(n=0, mae=np.nan, median=np.nan, p90=np.nan, p99=np.nan, gt10=np.nan, max=np.nan)
    return dict(n=int(a.size), mae=float(a.mean()), median=float(np.median(a)),
                p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)),
                gt10=float((a > 10).mean()), max=float(a.max()))


def fmt(v, nd=2):
    return "nan" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def bootstrap_diff(per_animal_a, per_animal_b, n_boot=10000, seed=0):
    """Paired bootstrap over animals of mean(a - b)."""
    keys = sorted(set(per_animal_a) & set(per_animal_b))
    d = np.array([per_animal_a[k] - per_animal_b[k] for k in keys])
    d = d[np.isfinite(d)]
    if d.size < 2:
        return dict(n_animals=int(d.size), mean_diff=np.nan, ci_lo=np.nan, ci_hi=np.nan, frac_a_better=np.nan)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, d.size, (n_boot, d.size))].mean(1)
    return dict(n_animals=int(d.size), mean_diff=float(d.mean()), ci_lo=float(np.percentile(bs, 2.5)),
                ci_hi=float(np.percentile(bs, 97.5)), frac_a_better=float((d < 0).mean()))


def draw(path, img, runs, preds, focus, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H, W = img.shape
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(n * W / 100, H / 100 + 0.5), dpi=100, squeeze=False)
    xs = np.arange(W)
    show = sorted({max(focus - 1, 0), focus, min(focus + 1, CH)})
    for ax, (name, y, conf, thr) in zip(axes[0], preds):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        for c in show:
            m = runs[:, 0] == c
            ax.scatter(runs[m, 1], runs[m, 2], s=3 if c == focus else 1, c=COLORS[c], linewidths=0)
            ax.plot(xs, np.where(conf[c] > thr, y[c], np.nan), lw=1.2 if c == focus else 0.6,
                    c=COLORS[c], alpha=0.9)
        ax.set_title(f"{name}: dots=GT, lines=pred ({', '.join(BOUNDARIES[c] for c in show)})", fontsize=8)
    fig.suptitle(title, fontsize=8)
    fig.subplots_adjust(0, 0, 1, 1 - 0.5 / fig.get_figheight(), wspace=0.01)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="NAME=path/to/best.pt ...")
    ap.add_argument("--split", default="val", choices=["val", "train", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--focus", nargs="+", default=["IPL", "GCL", "INL", "OLM"],
                    help="boundaries for worst-image overlays")
    ap.add_argument("--n-worst", type=int, default=8)
    ap.add_argument("--choroid-min-mass2", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    write_test(args.out)

    rows = [r for r in read_csv(os.path.join(args.data_dir, "manifest.csv")) if r["split"] == args.split]
    cons = load_json(os.path.join(args.data_dir, "order_constraints.json"))
    chain = [BIDX[c] for c in cons["chain"] if c not in cons["excluded_from_constraints"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[analyze] {len(rows)} {args.split} images, device {device}, threads {args.threads}", flush=True)

    # errors[variant][boundary][sample_id] -> abs error array
    errors = defaultdict(lambda: defaultdict(dict))
    chor = defaultdict(lambda: defaultdict(list))
    preds_cache = defaultdict(dict)  # variant -> sid -> (y, conf, thr)
    variants = []
    for spec in args.models:
        name, path = spec.split("=", 1)
        model, ck = load_checkpoint(path, device)
        targs = ck["args"]
        thr, win = targs["gate_thr"], targs["gate_win"]
        ds = BoundaryDataset(rows, args.data_dir, train=False, sigma=targs["sigma"])
        variants += [name, name + "-PP"]
        for i in range(len(ds)):
            b = ds[i]
            sid = rows[i]["sample_id"]
            with torch.no_grad():
                logits = model(b["image"][None].to(device))
                yhat, conf, p = decode(logits, win)
            y = yhat[0].cpu().numpy()
            cf = conf[0].cpu().numpy()
            ypp = isotonic_columns(y, cf, chain, thr)
            gt, sg = b["gt_y"].numpy(), b["single"].numpy()
            for c in range(len(BOUNDARIES)):
                if sg[c].any():
                    errors[name][c][sid] = np.abs(y[c][sg[c]] - gt[c][sg[c]])
                    errors[name + "-PP"][c][sid] = np.abs(ypp[c][sg[c]] - gt[c][sg[c]])
            _, runs = ds.load_raw(i)
            ce = choroid_eval(runs, *two_peaks(p[0, CH].float().cpu(), win=win), args.choroid_min_mass2)
            if ce is not None:
                for k, v in ce.items():
                    chor[name][k].append(v)
            preds_cache[name][sid] = (y, cf, thr)
            preds_cache[name + "-PP"][sid] = (ypp, cf, thr)
            if (i + 1) % 20 == 0:
                print(f"  {name}: {i + 1}/{len(ds)}", flush=True)
        del model

    meta = {r["sample_id"]: r for r in rows}
    noch = [c for c in range(len(BOUNDARIES)) if c != CH]

    # ---------- per boundary ----------
    pb = []
    macro = {}
    for v in variants:
        maes = {}
        for c, name in enumerate(BOUNDARIES):
            a = np.concatenate(list(errors[v][c].values())) if errors[v][c] else np.zeros(0)
            s = stats(a)
            pb.append(dict(variant=v, boundary=name, **s))
            maes[c] = s["mae"]
        macro[v] = dict(macro_mae_all=float(np.nanmean([maes[c] for c in maes])),
                        macro_mae_no_choroid=float(np.nanmean([maes[c] for c in noch])))
    write_csv(pb, os.path.join(args.out, "per_boundary.csv"))

    # ---------- choroid two-curve ----------
    cr = []
    for name, d in chor.items():
        g2p = np.concatenate(d["gt2pred"])
        p2g = np.concatenate(d["pred2gt"])
        ok = np.concatenate(d["count_ok"])
        ngt = np.concatenate(d["n_gt"])
        npred = np.concatenate(d["n_pred"])
        cr.append(dict(model=name, **{f"gt2pred_{k}": v for k, v in stats(g2p).items()},
                       **{f"pred2gt_{k}": v for k, v in stats(p2g).items() if k in ("mae", "median", "p90")},
                       peak_count_acc=float(ok.mean()), frac_cols_gt2=float((ngt >= 2).mean()),
                       frac_cols_pred2=float((npred == 2).mean())))
    write_csv(cr, os.path.join(args.out, "choroid_two_curve.csv"))

    # ---------- per image / experiment / animal ----------
    pi, per_exp, per_an = [], defaultdict(list), {}
    for v in variants:
        an_acc = defaultdict(lambda: defaultdict(list))
        for c, name in enumerate(BOUNDARIES):
            for sid, a in errors[v][c].items():
                m = meta[sid]
                s = stats(a)
                pi.append(dict(variant=v, sample_id=sid, animal_id=m["animal_id"],
                               experiment=m.get("experiment", ""), boundary=name, **s))
                per_exp[(v, m.get("experiment", ""), name)].append(a)
                an_acc[m["animal_id"]][c].append(a)
        per_an[v] = {a: float(np.nanmean([np.concatenate(d[c]).mean() for c in noch if d.get(c)]))
                     for a, d in an_acc.items() if any(d.get(c) for c in noch)}
    write_csv(pi, os.path.join(args.out, "per_image_boundary.csv"))
    pe = [dict(variant=v, experiment=e, boundary=b, **stats(np.concatenate(a)))
          for (v, e, b), a in sorted(per_exp.items())]
    write_csv(pe, os.path.join(args.out, "per_experiment_boundary.csv"))
    animals = sorted(set().union(*[set(d) for d in per_an.values()]))
    write_csv([dict(animal_id=a, experiment=next((r.get("experiment", "") for r in rows if r["animal_id"] == a), ""),
                    **{f"mae_no_choroid_{v}": per_an[v].get(a, np.nan) for v in variants}) for a in animals],
              os.path.join(args.out, "per_animal.csv"))

    # ---------- tail analysis ----------
    tail = []
    for v in variants:
        for c, name in enumerate(BOUNDARIES):
            if not errors[v][c]:
                continue
            tot = {sid: a.sum() for sid, a in errors[v][c].items()}
            T = sum(tot.values())
            srt = sorted(tot.items(), key=lambda kv: -kv[1])
            allerr = np.concatenate(list(errors[v][c].values()))
            k = max(1, int(round(0.01 * allerr.size)))
            top1 = np.sort(allerr)[-k:].sum()
            tail.append(dict(variant=v, boundary=name, n_images=len(tot),
                             share_worst5_images=float(sum(t for _, t in srt[:5]) / T) if T else np.nan,
                             share_worst1pct_columns=float(top1 / allerr.sum()) if allerr.sum() else np.nan,
                             worst_images="|".join(s for s, _ in srt[:5])))
    write_csv(tail, os.path.join(args.out, "tail.csv"))

    # ---------- paired bootstrap over animals ----------
    boot = []
    for i, a in enumerate(variants):
        for b2 in variants[i + 1:]:
            boot.append(dict(a=a, b=b2, metric="animal macro MAE (no CHOROID), a-b",
                             **bootstrap_diff(per_an[a], per_an[b2])))
    write_csv(boot, os.path.join(args.out, "paired_bootstrap.csv"))

    # ---------- worst-image overlays ----------
    first = args.models[0].split("=", 1)[0]
    others = [s.split("=", 1)[0] for s in args.models[1:]]
    ds0 = BoundaryDataset(rows, args.data_dir, train=False)
    idx_of = {r["sample_id"]: i for i, r in enumerate(rows)}
    for fname in args.focus:
        c = BIDX[fname]
        if not errors[first][c]:
            continue
        od = os.path.join(args.out, "worst", fname)
        os.makedirs(od, exist_ok=True)
        srt = sorted(errors[first][c].items(), key=lambda kv: -kv[1].mean())[: args.n_worst]
        for rank, (sid, a) in enumerate(srt, 1):
            img, runs = ds0.load_raw(idx_of[sid])
            pv = [(v, *preds_cache[v][sid]) for v in [first, first + "-PP"] + others]
            m = meta[sid]
            draw(os.path.join(od, f"{rank:02d}_{sid}.png"), img, runs, pv, c,
                 f"{fname} worst #{rank} in {first}: MAE {a.mean():.1f}px over {a.size} cols | "
                 f"{m['animal_id']} {m.get('experiment', '')}")

    # ---------- console report ----------
    save_json(dict(macro=macro, models=args.models, split=args.split, n_images=len(rows)),
              os.path.join(args.out, "summary.json"))
    print("\n===== macro MAE (px)")
    for v in variants:
        print(f"  {v:<8} all: {fmt(macro[v]['macro_mae_all'])}   without CHOROID: {fmt(macro[v]['macro_mae_no_choroid'])}")
    print("\n===== per boundary: MAE / median / P90 / P99 / share>10px")
    hdr = "  " + "boundary".ljust(9) + "".join(v.ljust(34) for v in variants)
    print(hdr)
    for name in BOUNDARIES:
        line = "  " + name.ljust(9)
        for v in variants:
            s = next(r for r in pb if r["variant"] == v and r["boundary"] == name)
            line += (f"{fmt(s['mae'])}/{fmt(s['median'])}/{fmt(s['p90'], 1)}/{fmt(s['p99'], 1)}/" + fmt(s['gt10'], 3) + " ").ljust(34)
        print(line)
    print("\n===== CHOROID as two curves (gt->pred: each GT curve to nearest predicted peak)")
    for r in cr:
        print(f"  {r['model']:<6} gt->pred MAE {fmt(r['gt2pred_mae'])} median {fmt(r['gt2pred_median'])} "
              f"P90 {fmt(r['gt2pred_p90'],1)} | pred->gt MAE {fmt(r['pred2gt_mae'])} | "
              f"peak count acc {fmt(r['peak_count_acc'],3)} (GT 2 curves in {fmt(r['frac_cols_gt2'],3)}, "
              f"pred 2 peaks in {fmt(r['frac_cols_pred2'],3)})")
    print("\n===== tail: share of total error from worst 5 images / worst 1% columns")
    for r in tail:
        if r["variant"] == first:
            print(f"  {r['boundary']:<8} worst5 imgs {fmt(r['share_worst5_images'],2)}  worst1% cols "
                  f"{fmt(r['share_worst1pct_columns'],2)}  ({r['n_images']} imgs)")
    print("\n===== per experiment, macro MAE without CHOROID")
    exps = sorted({k[1] for k in per_exp})
    for e in exps:
        line = f"  {e[:32]:<33}"
        for v in variants:
            vals = [np.concatenate(per_exp[(v, e, BOUNDARIES[c])]).mean() for c in noch if (v, e, BOUNDARIES[c]) in per_exp]
            line += f"{v}={fmt(float(np.mean(vals)) if vals else np.nan)}  "
        print(line)
    print("\n===== paired bootstrap over animals (mean diff a-b, 95% CI; negative = a better)")
    for r in boot:
        sig = "" if not np.isfinite(r["ci_lo"]) else ("  *" if r["ci_lo"] > 0 or r["ci_hi"] < 0 else "")
        print(f"  {r['a']:<6} vs {r['b']:<6} {fmt(r['mean_diff'],3)} [{fmt(r['ci_lo'],3)}, {fmt(r['ci_hi'],3)}] "
              f"a better in {fmt(r['frac_a_better'],2)} of {r['n_animals']} animals{sig}")
    print(f"\n[done] {args.out}  (worst-case overlays in {args.out}/worst/)")


if __name__ == "__main__":
    main()
