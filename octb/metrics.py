"""Metrics. All distances in canonical-grid pixels (NOT micrometres)."""
import numpy as np
import torch

from .common import BOUNDARIES, nanstats


def w1_columns(p, target, valid):
    """1-D Wasserstein-1 per (B,C,W) column; only meaningful where valid."""
    d = (torch.cumsum(p, 2) - torch.cumsum(target, 2)).abs().sum(2)
    return d * valid.float()


def sample_geom(yhat, conf, gt_y, single, pairs, thr=0.7, tol=0.5, jump_px=10.0):
    """Geometry metrics for ONE sample (numpy arrays of shape (C, W))."""
    C, W = yhat.shape
    conf_ok = conf > thr
    ae = [np.abs(yhat[c][single[c]] - gt_y[c][single[c]]) for c in range(C)]
    det = [(int((single[c] & conf_ok[c]).sum()), int(single[c].sum())) for c in range(C)]
    cg, cc = [], []
    for a, b in pairs:
        m = single[a] & single[b]
        cg.append((int(m.sum()), int((yhat[a][m] > yhat[b][m] + tol).sum())))
        m = conf_ok[a] & conf_ok[b]
        cc.append((int(m.sum()), int((yhat[a][m] > yhat[b][m] + tol).sum())))
    jumps = []
    for c in range(C):
        m = conf_ok[c][1:] & conf_ok[c][:-1]
        dj = np.abs(np.diff(yhat[c]))[m]
        jumps.append((int(m.sum()), int((dj > jump_px).sum())))
    return dict(ae=ae, det=det, cross_gt=cg, cross_conf=cc, jumps=jumps)


class Aggregator:
    """Accumulates per-sample results; summary() gives per-boundary numbers."""

    def __init__(self, pairs):
        self.pairs = pairs
        C = len(BOUNDARIES)
        self.w1s = np.zeros(C)
        self.w1n = np.zeros(C)
        self.ae = [[] for _ in range(C)]
        self.det = np.zeros((C, 2))
        self.cg = np.zeros((len(pairs), 2))
        self.cc = np.zeros((len(pairs), 2))
        self.jp = np.zeros((C, 2))

    def add(self, geom, w1_sum=None, w1_n=None):
        if w1_sum is not None:
            self.w1s += w1_sum
            self.w1n += w1_n
        for c, a in enumerate(geom["ae"]):
            if a.size:
                self.ae[c].append(a)
        self.det += np.array(geom["det"])
        if self.pairs:
            self.cg += np.array(geom["cross_gt"])
            self.cc += np.array(geom["cross_conf"])
        self.jp += np.array(geom["jumps"])

    def summary(self):
        out = {"per_boundary": {}, "pairs": {}}
        w1_list, mae_list = [], []
        for c, name in enumerate(BOUNDARIES):
            ae = np.concatenate(self.ae[c]) if self.ae[c] else np.zeros(0)
            st = nanstats(ae)
            w1 = self.w1s[c] / self.w1n[c] if self.w1n[c] > 0 else float("nan")
            out["per_boundary"][name] = dict(
                w1=w1, w1_cols=int(self.w1n[c]),
                mae=st["mean"], median_ae=st["median"], p90_ae=st["p90"], ae_cols=st["n"],
                confident_on_gt=self.det[c, 0] / self.det[c, 1] if self.det[c, 1] else float("nan"),
                jump_rate=self.jp[c, 1] / self.jp[c, 0] if self.jp[c, 0] else float("nan"))
            if np.isfinite(w1):
                w1_list.append(w1)
            if np.isfinite(st["mean"]):
                mae_list.append(st["mean"])
        for k, (a, b) in enumerate(self.pairs):
            key = f"{BOUNDARIES[a]}<{BOUNDARIES[b]}"
            out["pairs"][key] = dict(
                gt_cols=int(self.cg[k, 0]),
                crossing_rate_gt=self.cg[k, 1] / self.cg[k, 0] if self.cg[k, 0] else float("nan"),
                conf_cols=int(self.cc[k, 0]),
                crossing_rate_conf=self.cc[k, 1] / self.cc[k, 0] if self.cc[k, 0] else float("nan"))
        tg = self.cg.sum(0) if self.pairs else np.zeros(2)
        tc = self.cc.sum(0) if self.pairs else np.zeros(2)
        tj = self.jp.sum(0)
        out["macro_w1"] = float(np.mean(w1_list)) if w1_list else float("nan")
        out["macro_mae"] = float(np.mean(mae_list)) if mae_list else float("nan")
        out["crossing_rate_gt"] = float(tg[1] / tg[0]) if tg[0] else float("nan")
        out["crossing_rate_conf"] = float(tc[1] / tc[0]) if tc[0] else float("nan")
        out["jump_rate"] = float(tj[1] / tj[0]) if tj[0] else float("nan")
        return out
