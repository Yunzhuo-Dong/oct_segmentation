"""Sparse boundary strokes -> per-column runs -> soft vertical targets.

A *run* is a connected vertical group of annotated pixels of one boundary in
one image column. One run per (boundary, column) = single curve; two or more
runs (e.g. CHOROID) = multi-curve column, which is kept multimodal.
runs array columns: [boundary_idx, x, y_center, length_px]
"""
import numpy as np


def project_points(ys, xs, src_hw, dst_hw):
    """Map pixel coordinates from the source grid to the canonical grid."""
    sh, sw = src_hw
    dh, dw = dst_hw
    sy, sx = dh / sh, dw / sw
    yp = (ys.astype(np.float64) + 0.5) * sy - 0.5
    xp = np.round((xs.astype(np.float64) + 0.5) * sx - 0.5).astype(np.int64)
    xp = np.clip(xp, 0, dw - 1)
    return yp, xp, sy


def points_to_runs(b, ys, xs, gap_px=3.0, min_len=1.0):
    """Group points of one boundary into vertical runs per column."""
    if ys.size == 0:
        return np.zeros((0, 4), np.float32)
    o = np.lexsort((ys, xs))
    ys, xs = ys[o], xs[o]
    new = np.ones(ys.size, bool)
    new[1:] = (xs[1:] != xs[:-1]) | ((ys[1:] - ys[:-1]) > gap_px)
    starts = np.flatnonzero(new)
    rid = np.cumsum(new) - 1
    cnt = np.bincount(rid).astype(np.float64)
    yc = np.bincount(rid, ys) / cnt
    ymin = np.minimum.reduceat(ys, starts)
    ymax = np.maximum.reduceat(ys, starts)
    length = np.maximum(ymax - ymin + 1.0, min_len)
    out = np.stack([np.full(starts.size, b, np.float64), xs[starts].astype(np.float64), yc, length], 1)
    return out.astype(np.float32)


def build_target(runs, C, H, W, sigma0=1.5, kmax=120):
    """Return target (C,H,W) normalised per valid column, valid (C,W),
    nruns (C,W) int16, gt_y (C,W) float32 (nan unless exactly one run)."""
    target = np.zeros((C, H, W), np.float32)
    nruns = np.zeros((C, W), np.int16)
    gt_y = np.full((C, W), np.nan, np.float32)
    if runs is None or len(runs) == 0:
        return target, np.zeros((C, W), bool), nruns, gt_y
    b = runs[:, 0].astype(np.int64)
    x = runs[:, 1].astype(np.int64)
    yc = runs[:, 2].astype(np.float64)
    L = runs[:, 3].astype(np.float64)
    sig = np.sqrt(sigma0 ** 2 + np.maximum(L ** 2 - 1.0, 0.0) / 12.0)
    K = int(min(np.ceil(4 * sig.max()), kmax))
    off = np.arange(-K, K + 1)
    ys = np.round(yc)[:, None] + off[None, :]
    d = (ys - yc[:, None]) / sig[:, None]
    w = np.exp(-0.5 * d * d)
    keep = (ys >= 0) & (ys < H) & (np.abs(d) <= 4.0)
    bb = np.broadcast_to(b[:, None], ys.shape)[keep]
    xx = np.broadcast_to(x[:, None], ys.shape)[keep]
    np.add.at(target, (bb, ys[keep].astype(np.int64), xx), w[keep].astype(np.float32))
    np.add.at(nruns, (b, x), 1)
    s = target.sum(1)
    valid = s > 1e-6
    target /= np.where(valid, s, 1.0)[:, None, :]
    single = nruns[b, x] == 1
    gt_y[b[single], x[single]] = yc[single]
    return target, valid, nruns, gt_y
