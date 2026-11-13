#!/usr/bin/env python
"""Step 7: run a trained model on raw AVI frames.

Pipeline per frame: supervisor crop (30,26,770,1000) -> optional vertical flip
-> resample to the training pixel grid (scales from 06_avi_match.py) ->
standardise -> model -> boundaries mapped back to crop coordinates.

Optic nerve head (PDF p.3): columns inside --onh ellipse (frame, x) are
excluded from reported boundaries; layer loss there is anatomy, not pathology.

Outputs: boundaries.npz (frame, C, x) in crop pixels + confidence, overlays,
per-boundary depth maps and adjacent-layer thickness maps (frame index axis is
NOT calibrated distance).

python 07_avi_infer.py --avi X.avi --ckpt best.pt --data-dir D --match match_X.json --out out_X --step 4
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from octb.avi import crop, frame_count, parse_crop, read_frames  # noqa: E402
from octb.common import BIDX, BOUNDARIES, load_json, save_json, write_test  # noqa: E402
from octb.dataset import standardize  # noqa: E402
from octb.losses import decode  # noqa: E402
from octb.matio import resize_gray  # noqa: E402
from octb.model import load_checkpoint  # noqa: E402
from octb.postprocess import isotonic_columns  # noqa: E402

def two_peaks(p, sep=6, win=5):
    """p: (B,H,W) -> y1, m1, y2, m2 (B,W): top-2 separated peaks with local soft-argmax and mass."""
    H = p.shape[1]
    off = torch.arange(-win, win + 1, device=p.device).view(1, -1, 1)

    def local(idx):
        ii = idx.unsqueeze(1) + off
        inb = (ii >= 0) & (ii < H)
        ii = ii.clamp(0, H - 1)
        pw = torch.gather(p, 1, ii) * inb
        m = pw.sum(1)
        return (pw * ii.float()).sum(1) / m.clamp_min(1e-8), m

    i1 = p.argmax(1)
    y1, m1 = local(i1)
    rows = torch.arange(H, device=p.device).view(1, -1, 1)
    i2 = p.masked_fill((rows - i1.unsqueeze(1)).abs() <= sep, 0.0).argmax(1)
    y2, m2 = local(i2)
    return y1, m1, y2, m2


COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
          "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#ffffff"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avi", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True, help="training data dir (orientation, constraints)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", default="supervisor", help="'supervisor' or x,y,w,h")
    ap.add_argument("--match", default=None, help="json from 06_avi_match.py")
    ap.add_argument("--scale-y", type=float, default=None)
    ap.add_argument("--scale-x", type=float, default=None)
    ap.add_argument("--flip", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--frames", default=None, help="start:stop")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--conf-thr", type=float, default=None)
    ap.add_argument("--postprocess", action="store_true", help="isotonic ordering per column")
    ap.add_argument("--onh", default=None, help="frame_center,x_center,frame_radius,x_radius (crop coords)")
    ap.add_argument("--overlay-every", type=int, default=50)
    ap.add_argument("--choroid-thr", type=float, default=0.2, help="local mass threshold per CHOROID peak")
    ap.add_argument("--choroid-sep", type=int, default=6, help="min px between the two CHOROID peaks")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(args.ckpt, device)
    targs = ck["args"]
    thr = args.conf_thr if args.conf_thr is not None else targs["gate_thr"]
    cons = load_json(os.path.join(args.data_dir, "order_constraints.json"))
    chain = [BIDX[c] for c in cons["chain"] if c not in cons["excluded_from_constraints"]]

    m = load_json(args.match) if args.match else {}
    sy = args.scale_y if args.scale_y is not None else m.get("infer_scale_y", 1.0)
    sx = args.scale_x if args.scale_x is not None else m.get("infer_scale_x", 1.0)
    if args.flip == "auto":
        if "infer_flip_vertical" in m:
            flip = m["infer_flip_vertical"]
        else:  # raw AVI shows vitreous on top (PDF p.3)
            flip = cons["orientation"] == "vitreous_bottom"
    else:
        flip = args.flip == "yes"
    box = parse_crop(args.crop)
    if not args.match and args.scale_y is None:
        print("[WARN] no --match / --scale-*: assuming AVI pixel spacing == training grid. Run 06_avi_match.py first.")

    n = frame_count(args.avi)
    a, b = (0, n) if not args.frames else map(int, args.frames.split(":"))
    idx = list(range(a, min(b, n) if n > 0 else b, args.step))
    os.makedirs(os.path.join(args.out, "overlays"), exist_ok=True)
    write_test(args.out)
    hc, wc = box[3], box[2]
    hm, wm = round(hc * sy), round(wc * sx)
    print(f"[avi] {n} frames, using {len(idx)}; crop {box}; flip={flip}; model grid {hm}x{wm}")

    use_amp = targs.get("amp") == "bf16" and device.type == "cuda"
    Ys, Cs, Fs, crops_keep = [], [], [], {}
    Ch, Chm = [], []  # CHOROID: two curves (upper, lower) and their local masses
    CH = BIDX["CHOROID"]
    buf = []

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack([standardize(bb[1])[None] for bb in buf])).to(device)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.autocast("cpu", enabled=False):
                logits = model(x)
            yhat, conf, p = decode(logits, targs["gate_win"])
            ch2 = two_peaks(p[:, CH], args.choroid_sep, targs["gate_win"])  # 4 x (B, W)
        yhat, conf = yhat.cpu().numpy(), conf.cpu().numpy()
        ch2 = [t.cpu().numpy() for t in ch2]
        xm = (np.arange(wm) + 0.5) / sx - 0.5
        xc = np.arange(wc)
        for k, (fi, _) in enumerate(buf):
            y, c = yhat[k], conf[k]
            if args.postprocess:
                y = isotonic_columns(y, c, chain, thr)
            y = (y + 0.5) / sy - 0.5
            if flip:
                y = (hc - 1) - y
            yc = np.stack([np.interp(xc, xm, y[j]) for j in range(y.shape[0])])
            cc = np.stack([np.interp(xc, xm, c[j]) for j in range(c.shape[0])])
            Ys.append(yc.astype(np.float32)); Cs.append(cc.astype(np.float32)); Fs.append(fi)
            y1, m1, y2, m2 = (t[k] for t in ch2)
            lo, hi = np.minimum(y1, y2), np.maximum(y1, y2)
            mlo, mhi = np.where(y1 <= y2, m1, m2), np.where(y1 <= y2, m2, m1)
            cy, cm = [], []
            for yy, mm in ((lo, mlo), (hi, mhi)):
                yy = (yy + 0.5) / sy - 0.5
                if flip:
                    yy = (hc - 1) - yy
                cy.append(np.interp(xc, xm, yy)); cm.append(np.interp(xc, xm, mm))
            if flip:
                cy, cm = cy[::-1], cm[::-1]
            Ch.append(np.stack(cy).astype(np.float32)); Chm.append(np.stack(cm).astype(np.float32))
        buf.clear()

    for fi, fr in read_frames(args.avi, idx):
        c = crop(fr, box)
        if (fi - a) // args.step % max(1, args.overlay_every // args.step) == 0:
            crops_keep[fi] = c.copy()
        g = c[::-1] if flip else c
        g = resize_gray(np.ascontiguousarray(g), (hm, wm))
        buf.append((fi, g))
        if len(buf) >= args.batch:
            flush()
    flush()
    Y, CF, F = np.stack(Ys), np.stack(Cs), np.array(Fs)
    CHY, CHM = np.stack(Ch), np.stack(Chm)  # (N, 2, W): upper / lower CHOROID curve

    onh_mask = np.zeros((len(F), wc), bool)
    if args.onh:
        f0, x0, rf, rx = map(float, args.onh.split(","))
        onh_mask = ((F[:, None] - f0) / rf) ** 2 + ((np.arange(wc)[None] - x0) / rx) ** 2 <= 1.0
    reported = (CF > thr) & ~onh_mask[:, None, :]
    ch_rep = (CHM > args.choroid_thr) & ~onh_mask[:, None, :]
    np.savez_compressed(os.path.join(args.out, "boundaries.npz"), frames=F, y=Y, conf=CF,
                        reported=reported, onh_mask=onh_mask, boundaries=np.array(BOUNDARIES),
                        choroid_y=CHY, choroid_mass=CHM, choroid_reported=ch_rep)
    save_json(dict(avi=args.avi, ckpt=args.ckpt, crop_xywh=list(box), flip_vertical=bool(flip),
                   scale_y=sy, scale_x=sx, conf_thr=thr, postprocess=args.postprocess, onh=args.onh,
                   n_frames_used=int(len(F)), coords="y in crop pixels (0 = top of crop)",
                   frame_axis="frame index, NOT calibrated distance",
                   reported_fraction={**{n: float(reported[:, i].mean()) for i, n in enumerate(BOUNDARIES)
                                         if i != CH},
                                      "CHOROID_upper": float(ch_rep[:, 0].mean()),
                                      "CHOROID_lower": float(ch_rep[:, 1].mean())},
                   choroid="two curves from the top-2 peaks; reported where local mass > choroid_thr",
                   choroid_thr=args.choroid_thr),
              os.path.join(args.out, "run.json"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for fi, c in crops_keep.items():
        k = int(np.flatnonzero(F == fi)[0])
        fig, ax = plt.subplots(figsize=(wc / 100, hc / 100), dpi=100)
        ax.imshow(c, cmap="gray", vmin=0, vmax=255)
        for j in range(len(BOUNDARIES)):
            if j == CH:
                for q in range(2):
                    ax.plot(np.arange(wc), np.where(ch_rep[k, q], CHY[k, q], np.nan), lw=0.8, c=COLORS[j])
                continue
            ax.plot(np.arange(wc), np.where(reported[k, j], Y[k, j], np.nan), lw=0.8, c=COLORS[j])
        if onh_mask[k].any():
            xs = np.flatnonzero(onh_mask[k])
            ax.axvspan(xs.min(), xs.max(), color="cyan", alpha=0.12)
        ax.set_axis_off(); ax.set_title(f"frame {fi}", fontsize=8)
        fig.subplots_adjust(0, 0, 1, 0.97)
        fig.savefig(os.path.join(args.out, "overlays", f"frame_{fi:05d}.png"))
        plt.close(fig)
    maps = [(BOUNDARIES[j], np.where(reported[:, j], Y[:, j], np.nan))
            for j in range(len(BOUNDARIES)) if j != CH and reported[:, j].any()]
    for q, nm in enumerate(("CHOROID upper", "CHOROID lower")):
        if ch_rep[:, q].any():
            maps.append((nm, np.where(ch_rep[:, q], CHY[:, q], np.nan)))
    ncol = 4

    def grid(items, cmap, fname, unit):
        if not items:
            return
        nrow = int(np.ceil(len(items) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
        for ax, (title, m) in zip(axes.ravel(), items):
            fin = m[np.isfinite(m)]
            vmin, vmax = (np.percentile(fin, [2, 98]) if fin.size else (0, 1))
            im = ax.imshow(m, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, wc, F[-1], F[0]],
                           interpolation="nearest")
            ax.set_title(f"{title} {unit}", fontsize=9)
            ax.set_xlabel("x (crop px)"); ax.set_ylabel("frame index")
            fig.colorbar(im, ax=ax)
        for ax in axes.ravel()[len(items):]:
            ax.set_axis_off()
        fig.suptitle("colour range = 2nd-98th percentile; white = not reported (low confidence / ONH)", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, fname), dpi=110); plt.close(fig)

    grid(maps, "viridis", "depth_maps.png", "depth (crop px)")
    ch = [j for j in chain if j != CH and reported[:, j].any()]
    thick = [(f"{BOUNDARIES[u]}->{BOUNDARIES[l]}", np.where(reported[:, u] & reported[:, l], Y[:, l] - Y[:, u], np.nan))
             for u, l in zip(ch[:-1], ch[1:])]
    if BIDX["RPE"] in ch:
        r = BIDX["RPE"]
        thick.append(("RPE->CHOROID lower",
                      np.where(reported[:, r] & ch_rep[:, 1], CHY[:, 1] - Y[:, r], np.nan)))
    if ch:
        thick.append((f"{BOUNDARIES[ch[0]]}->{BOUNDARIES[ch[-1]]} (total)",
                      np.where(reported[:, ch[0]] & reported[:, ch[-1]], Y[:, ch[-1]] - Y[:, ch[0]], np.nan)))
    grid(thick, "magma", "thickness_maps.png", "(crop px)")
    print(f"[done] {args.out}  frames={len(F)}")


if __name__ == "__main__":
    main()
