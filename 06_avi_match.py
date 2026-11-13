#!/usr/bin/env python
"""Step 6: find where a training TIFF sits inside the raw AVI of the SAME scan
(frame index, vertical/horizontal scale, offset, flip). This answers, from data,
whether AVI frames need rescaling/flipping before the model sees them, and how
the supervisor crop (30,26,770,1000) relates to the TIFF field of view.

python 06_avi_match.py --avi X.avi --tiff X.tif --data-dir "$WS/octb_data_v1" --out match_X.json
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from octb.avi import SUPERVISOR_CROP, frame_count, read_frames  # noqa: E402
from octb.common import load_json, save_json  # noqa: E402
from octb.matio import read_image_gray  # noqa: E402


def prep(img, f):
    import cv2
    a = img.astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), 1.5 * f)
    if f > 1:
        a = cv2.resize(a, (max(1, round(a.shape[1] / f)), max(1, round(a.shape[0] / f))), interpolation=cv2.INTER_AREA)
    return a


def score(frame_p, tmpl, sy, sx, flip):
    import cv2
    t = tmpl[::-1] if flip else tmpl
    h = max(8, round(t.shape[0] * sy)); w = max(8, round(t.shape[1] * sx))
    if h >= frame_p.shape[0] or w >= frame_p.shape[1]:
        return -1.0, (0, 0)
    t = cv2.resize(t, (w, h), interpolation=cv2.INTER_AREA)
    r = cv2.matchTemplate(frame_p, t, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(r)
    return float(mx), loc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avi", required=True)
    ap.add_argument("--tiff", required=True)
    ap.add_argument("--data-dir", default=None, help="to convert to the canonical training grid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame-step", type=int, default=4)
    ap.add_argument("--coarse", type=int, default=4, help="downsampling factor for coarse search")
    ap.add_argument("--sy-range", default="0.6,1.6,0.05")
    ap.add_argument("--sx-range", default="0.85,1.15,0.05")
    ap.add_argument("--inner", type=float, default=0.1, help="fraction trimmed from each TIFF border for the template")
    args = ap.parse_args()

    tif, _ = read_image_gray(args.tiff)
    Ht, Wt = tif.shape
    ty0, tx0 = int(Ht * args.inner), int(Wt * args.inner)
    tmpl_full = tif[ty0:Ht - ty0, tx0:Wt - tx0]
    n = frame_count(args.avi)
    idx = list(range(0, n, args.frame_step)) if n > 0 else None
    frames = dict(read_frames(args.avi, idx))
    if not frames:
        sys.exit("no frames read")
    any_fr = next(iter(frames.values()))
    print(f"[match] tiff {tif.shape}, avi {len(frames)} sampled frames of {n}, frame {any_fr.shape}")

    f = args.coarse
    tp = prep(tmpl_full, f)
    fp = {i: prep(fr, f) for i, fr in frames.items()}
    # 1) best frames at scale 1, both orientations
    s1 = []
    for i, a in fp.items():
        for flip in (False, True):
            s, _ = score(a, tp, 1.0, 1.0, flip)
            s1.append((s, i, flip))
    s1.sort(reverse=True)
    top = s1[:6]
    # 2) scale search on top candidates
    sy0, sy1, sys_ = map(float, args.sy_range.split(","))
    sx0, sx1, sxs = map(float, args.sx_range.split(","))
    best = (-1, None)
    for _, i, flip in top:
        for sy in np.arange(sy0, sy1 + 1e-9, sys_):
            for sx in np.arange(sx0, sx1 + 1e-9, sxs):
                s, loc = score(fp[i], tp, sy, sx, flip)
                if s > best[0]:
                    best = (s, (i, flip, sy, sx))
    _, (bi, bflip, bsy, bsx) = best
    # 3) fine search at full resolution: all frames near bi, fine scales
    extra = dict(read_frames(args.avi, [j for j in range(max(0, bi - 2 * args.frame_step), bi + 2 * args.frame_step + 1)]))
    frames.update(extra)
    near = sorted(extra)
    tpf = prep(tmpl_full, 1)
    fine = (-1, None)
    for i in near:
        a = prep(frames[i], 1)
        for sy in np.arange(bsy - sys_, bsy + sys_ + 1e-9, sys_ / 5):
            for sx in np.arange(bsx - sxs, bsx + sxs + 1e-9, sxs / 5):
                s, loc = score(a, tpf, sy, sx, bflip)
                if s > fine[0]:
                    fine = (s, (i, sy, sx, loc))
    ncc, (fi, fsy, fsx, (mx, my)) = fine
    # TIFF pixel (0,0) in AVI frame coordinates
    oy = my - ty0 * fsy  # symmetric trim -> same formula with or without flip
    ox = mx - tx0 * fsx
    res = dict(avi=args.avi, tiff=args.tiff, tiff_shape=[Ht, Wt], avi_frame_shape=list(any_fr.shape),
               n_frames=n, best_frame=int(fi), ncc=ncc, flip_vertical=bool(bflip),
               scale_y_tiff_to_avi=float(fsy), scale_x_tiff_to_avi=float(fsx),
               tiff_origin_in_avi_xy=[float(ox), float(oy)],
               tiff_extent_in_avi_xywh=[float(ox), float(oy), float(Wt * fsx), float(Ht * fsy)],
               supervisor_crop_xywh=list(SUPERVISOR_CROP))
    # parameters for 07_avi_infer.py: AVI crop pixels -> canonical training grid
    ch, cw = Ht, Wt
    if args.data_dir:
        ch, cw = load_json(os.path.join(args.data_dir, "summary.json"))["canonical_hw"]
    res["infer_scale_y"] = float((1.0 / fsy) * (ch / Ht))
    res["infer_scale_x"] = float((1.0 / fsx) * (cw / Wt))
    res["infer_flip_vertical"] = bool(bflip)
    res["note"] = ("ncc < ~0.5 means the match is unreliable (different scan / processing). "
                   "Check several scans before fixing a global scale.")
    save_json(res, args.out)
    print(f"[match] frame {fi}  ncc {ncc:.3f}  flip {bflip}  scale tiff->avi y {fsy:.3f} x {fsx:.3f}")
    print(f"[match] TIFF covers AVI x,y,w,h = {[round(v, 1) for v in res['tiff_extent_in_avi_xywh']]}  "
          f"(supervisor crop {SUPERVISOR_CROP})")
    print(f"[match] for 07_avi_infer.py: --scale-y {res['infer_scale_y']:.4f} --scale-x {res['infer_scale_x']:.4f} "
          f"--flip {'yes' if bflip else 'no'}   -> {args.out}")


if __name__ == "__main__":
    main()
