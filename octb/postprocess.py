"""E0-PP: enforce the data-derived boundary order per column after inference
(weighted isotonic regression over confident channels). Zero training cost;
the baseline E1 has to beat."""
import numpy as np


def pav(y, w):
    """Weighted pool-adjacent-violators: nondecreasing fit."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(float(yi))
        wts.append(float(wi))
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            wv = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / wv
            c = cnt[-2] + cnt[-1]
            vals[-2:], wts[-2:], cnt[-2:] = [v], [wv], [c]
    out = []
    for v, c in zip(vals, cnt):
        out.extend([v] * c)
    return np.array(out)


def isotonic_columns(yhat, conf, chain_idx, thr=0.7):
    """yhat, conf: (C, W). chain_idx: channel indices ordered top->bottom
    (increasing y). Only confident channels in a column are adjusted."""
    out = yhat.copy()
    chain = np.asarray(chain_idx)
    ok = conf[chain] > thr  # K, W
    sub_y = yhat[chain]
    # fast path: columns already ordered need no work
    for x in range(yhat.shape[1]):
        k = np.flatnonzero(ok[:, x])
        if k.size < 2:
            continue
        v = sub_y[k, x]
        if np.all(np.diff(v) >= 0):
            continue
        out[chain[k], x] = pav(v, conf[chain[k], x])
    return out
