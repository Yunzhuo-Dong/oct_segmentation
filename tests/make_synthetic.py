#!/usr/bin/env python
"""Generate a tiny fake dataset in the same layout as the real one:
<root>/training/*.mat + *.tif, <root>/testing/*.mat + *.tif, plus one AVI.
Only for smoke-testing the pipeline; no scientific meaning."""
import os
import sys

import numpy as np
from scipy.io import savemat

NAMES = ["RNFL", "GCL", "IPL", "INL", "OPL", "ONL", "OLM", "IS", "OS", "RPE", "chroroid"]
DEPTH = [60, 72, 95, 120, 135, 165, 172, 180, 190, 205, 222]


def make_scan(rng, H, W, onh=False):
    x = np.arange(W)
    bend = 12 * np.sin(x / W * np.pi + rng.uniform(-.3, .3)) + rng.uniform(-10, 10)
    curves = []
    for k, d in enumerate(DEPTH):
        y = d + bend + 2 * np.sin(x / rng.uniform(15, 30))
        if onh and 1 <= k <= 8:  # layers converge near an "optic nerve" region
            c = np.exp(-((x - W * 0.3) / 12) ** 2)
            y = y * (1 - c) + (DEPTH[0] + bend + 5) * c
        curves.append(np.clip(y, 0, H - 1))
    img = np.full((H, W), 20.0)
    for k in range(len(curves) - 1):
        top, bot = curves[k], curves[k + 1]
        val = 120 if k % 2 == 0 else 60
        for xi in range(W):
            img[int(top[xi]):int(bot[xi]), xi] = val
    img[int(DEPTH[-1] + 20):, :] = 10
    img = img * rng.gamma(2.0, 0.5, img.shape)  # speckle
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, curves


def label_mask(H, W, curves_list, gaps):
    m = np.zeros((H, W), bool)
    for y in curves_list:
        for xi in range(W):
            if gaps[0] <= xi < gaps[1]:
                continue
            m[int(round(y[xi])), xi] = True
    return m


def main(root):
    import tifffile
    rng = np.random.default_rng(0)
    H, W = 300, 200
    animals = [("BL", "F", 380 + i) for i in range(6)] + [("MA", "M", 830 + i) for i in range(6)]
    k = 0
    keep_for_avi = None
    for ai, (co, sx, num) in enumerate(animals):
        for rep in range(3):
            sub = "testing" if ai % 4 == 3 else "training"
            os.makedirs(os.path.join(root, sub), exist_ok=True)
            eye = "l" if rep % 2 == 0 else "r"
            stem = f"2406{10 + rep:02d}_{co}-{sx}{num}_{eye}_734nm_{k:03d}"
            if ai == 5:  # real-data style without date and with extra dash: MA-N-062_l_...
                stem = f"{co}-{sx}-{num:03d}_{eye}_734nm_prozess_{rep}"
            img, curves = make_scan(rng, H, W, onh=(k % 7 == 0))
            tifffile.imwrite(os.path.join(root, sub, stem + ".tif"), img)
            if keep_for_avi is None:
                keep_for_avi = (img, stem)
            names = list(NAMES)
            if k % 5 == 0:
                names.remove("GCL")  # not every image has every boundary
            ld = np.zeros((len(names),), dtype=[("Name", "O"), ("Class", "O"), ("labelMask", "O")])
            for j, nm in enumerate(names):
                ci = NAMES.index(nm)
                cl = [curves[ci]]
                if nm == "chroroid":
                    cl.append(curves[ci] + 15)  # two curves in one column
                g0 = int(rng.integers(0, W))
                ld[j] = (nm, "borders", label_mask(H, W, cl, (g0, g0 + int(rng.integers(0, 40)))))
            src = ("C:\\data\\oct\\" + stem + ".tif") if k % 3 == 0 else stem + ".tif"
            savemat(os.path.join(root, sub, stem + ".mat"),
                    {"grdTruthOutput": {"Source": {"file": src}, "Parameter": {"userID": "EL"}, "LabelData": ld}})
            if k == 1:  # identical copies of image+annotation in other folders (BL-F379 case)
                import shutil
                for extra in ("untitled folder", "trial_1"):
                    d = os.path.join(root, sub, extra)
                    os.makedirs(d, exist_ok=True)
                    shutil.copy(os.path.join(root, sub, stem + ".tif"), d)
                    shutil.copy(os.path.join(root, sub, stem + ".mat"), d)
            k += 1
    # broken file to exercise skipped.csv
    with open(os.path.join(root, "training", "broken.mat"), "wb") as f:
        f.write(b"not a mat file")
    # AVI containing the first TIFF at offset (x=20, y=30)
    try:
        import cv2
        img, stem = keep_for_avi
        FH, FW = 420, 232
        vw = cv2.VideoWriter(os.path.join(root, stem + ".avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (FW, FH), False)
        for fi in range(24):
            fr = np.full((FH, FW), 15, np.uint8)
            shift = abs(fi - 12)
            tile = np.roll(img, shift, axis=1) if shift else img
            fr[30:30 + H, 20:20 + W] = tile
            vw.write(fr)
        vw.release()
        print("avi:", os.path.join(root, stem + ".avi"), "tif:", stem)
    except ImportError:
        pass
    print("synthetic data in", root)


if __name__ == "__main__":
    main(sys.argv[1])
