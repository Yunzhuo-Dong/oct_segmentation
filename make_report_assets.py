#!/usr/bin/env python
"""Copy the E1 results that belong in the repository into ./results/ .

Run on the HPC login node from the repository folder:
    source slurm/barnard_env.sh        (or slurm/env.sh on Capella)
    python make_report_assets.py --avi-name 240603_MA-N978_r_734nm_prozess

Writes:
  results/figures/   training curve, GT example, predictions, AVI frames, depth / thickness maps
  results/metrics/   per-boundary tables (val + test), CHOROID two-curve table, per-experiment table
  results/training/  config.json, history.csv, best_val_metrics.json
  results/data/      order_constraints.json, order_table.csv, boundary_inventory.csv
  results/model/     E1_weights.pt (weights only, fp16, ~26 MB)
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))


def cp(src, dst):
    if not src or not os.path.exists(src):
        print(f"  [missing] {src}")
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  {os.path.relpath(dst, HERE)}")
    return True


def filter_csv(src, dst, keep):
    if not os.path.exists(src):
        print(f"  [missing] {src}")
        return
    with open(src, newline="") as f:
        rows = [r for r in csv.DictReader(f) if keep(r)]
    if not rows:
        print(f"  [empty after filter] {src}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  {os.path.relpath(dst, HERE)}")


def plot_history(hist, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(hist, newline="") as f:
        rows = list(csv.DictReader(f))
    ep = [int(r["epoch"]) + 1 for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    ax[0].plot(ep, [float(r["train_ce"]) for r in rows], label="train CE")
    ax[0].plot(ep, [float(r["val_ce"]) for r in rows], label="val CE")
    ax[0].set_title("Cross-entropy"); ax[0].legend()
    ax[1].plot(ep, [float(r["val_macro_w1"]) for r in rows], label="val W1")
    ax[1].plot(ep, [float(r["val_macro_mae"]) for r in rows], label="val MAE")
    ax[1].set_title("Validation error (px)"); ax[1].legend()
    ax[2].plot(ep, [float(r["train_order"]) for r in rows], label="order loss")
    ax[2].plot(ep, [float(r["train_smooth"]) for r in rows], label="smooth loss")
    ax[2].set_title("Anatomical constraint losses (train)"); ax[2].legend()
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  {os.path.relpath(out, HERE)}")


def main():
    ws = os.environ.get("WS", "/data/horse/ws/yudo868h-Yunzhuo_space")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.environ.get("RUNS", os.path.join(ws, "octb_runs")))
    ap.add_argument("--data-dir", default=os.environ.get("DATA", os.path.join(ws, "octb_data_v1")))
    ap.add_argument("--review-dir", default=os.path.expanduser("~/octb_review"))
    ap.add_argument("--avi-name", required=True, help="folder name under $RUNS/avi_E1/")
    ap.add_argument("--exp", default="E1")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--no-weights", action="store_true")
    a = ap.parse_args()

    run = os.path.join(a.runs, a.exp)
    fig = os.path.join(a.out, "figures")
    met = os.path.join(a.out, "metrics")
    print("[training]")
    for f in ("config.json", "history.csv", "best_val_metrics.json"):
        cp(os.path.join(run, f), os.path.join(a.out, "training", f))
    if os.path.exists(os.path.join(run, "history.csv")):
        plot_history(os.path.join(run, "history.csv"), os.path.join(fig, "training_curves.png"))
    # config.json contains absolute paths and the host name; keep only what matters
    cfgp = os.path.join(a.out, "training", "config.json")
    if os.path.exists(cfgp):
        c = json.load(open(cfgp))
        keep = {k: c.get(k) for k in ("args", "n_params", "n_train", "n_val", "constraint_pairs",
                                      "smooth_channels", "gpu", "torch", "boundaries")}
        for k in ("data_dir", "out"):
            keep["args"].pop(k, None)
        json.dump(keep, open(cfgp, "w"), indent=2)

    print("[data rules]")
    for f in ("order_constraints.json", "order_table.csv", "boundary_inventory.csv"):
        cp(os.path.join(a.data_dir, f), os.path.join(a.out, "data", f))

    print("[metrics]")
    ev_val = os.path.join(run, "eval_val")
    cp(os.path.join(ev_val, "per_boundary.csv"), os.path.join(met, "val_per_boundary.csv"))
    an_test = os.path.join(a.runs, "analysis_test")
    an_val = os.path.join(a.runs, "analysis_val")
    for tag, d in (("test", an_test), ("val", an_val)):
        filter_csv(os.path.join(d, "per_boundary.csv"), os.path.join(met, f"{tag}_per_boundary_detailed.csv"),
                   lambda r: r.get("variant") == a.exp)
        filter_csv(os.path.join(d, "choroid_two_curve.csv"), os.path.join(met, f"{tag}_choroid_two_curve.csv"),
                   lambda r: r.get("model") == a.exp)
        filter_csv(os.path.join(d, "per_experiment_boundary.csv"),
                   os.path.join(met, f"{tag}_per_experiment_boundary.csv"), lambda r: r.get("variant") == a.exp)

    print("[figures]")
    gt = sorted(glob.glob(os.path.join(a.review_dir, "*.png")))[:2]
    for i, p in enumerate(gt, 1):
        cp(p, os.path.join(fig, f"gt_example_{i}.png"))
    ov = sorted(glob.glob(os.path.join(ev_val, "overlays", "*.png")))
    pick = [ov[i] for i in sorted({0, len(ov) // 2, len(ov) - 1})] if ov else []
    for i, p in enumerate(pick, 1):
        cp(p, os.path.join(fig, f"val_prediction_{i}.png"))
    avi = os.path.join(a.runs, "avi_E1", a.avi_name)
    frames = sorted(glob.glob(os.path.join(avi, "overlays", "frame_*.png")))
    want = [p for p in frames if os.path.basename(p) in ("frame_00050.png", "frame_00100.png", "frame_00500.png")]
    for p in (want or frames[:3]):
        cp(p, os.path.join(fig, "avi_" + os.path.basename(p)))
    cp(os.path.join(avi, "depth_maps.png"), os.path.join(fig, "avi_depth_maps.png"))
    cp(os.path.join(avi, "thickness_maps.png"), os.path.join(fig, "avi_thickness_maps.png"))
    for f in ("run.json", "match.json"):
        src = os.path.join(avi, f)
        if os.path.exists(src):
            d = json.load(open(src))
            for k in ("avi", "tiff", "ckpt"):
                if k in d:
                    d[k] = os.path.basename(str(d[k]))
            os.makedirs(os.path.join(a.out, "avi"), exist_ok=True)
            json.dump(d, open(os.path.join(a.out, "avi", f), "w"), indent=2)
            print(f"  results/avi/{f}")

    if not a.no_weights:
        print("[model]")
        ck_path = os.path.join(run, "best.pt")
        if os.path.exists(ck_path):
            import torch
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            args = dict(ck["args"])
            for k in ("data_dir", "out"):
                args.pop(k, None)
            slim = dict(model={k: v.half() if v.is_floating_point() else v for k, v in ck["model"].items()},
                        args=args, epoch=ck["epoch"], boundaries=ck.get("boundaries"),
                        manifest_sha256=ck.get("manifest_sha256"))
            dst = os.path.join(a.out, "model", f"{a.exp}_weights.pt")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            torch.save(slim, dst)
            print(f"  {os.path.relpath(dst, HERE)}  ({os.path.getsize(dst) / 2**20:.1f} MB, best epoch {ck['epoch'] + 1})")
        else:
            print(f"  [missing] {ck_path}")
    print(f"[done] -> {a.out}")


if __name__ == "__main__":
    main()
