"""Dataset: image + runs -> tensors. Augmentations transform image AND runs
together, so targets/valid masks can never drift from the image."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import BOUNDARIES
from .targets import build_target


def standardize(x):
    x = x.astype(np.float32) / 255.0
    return (x - x.mean()) / (x.std() + 1e-6)


def _shift_rows(img, dy, rng):
    """Vertical shift; exposed rows filled with background-like noise."""
    if dy == 0:
        return img
    H = img.shape[0]
    flat = img.reshape(-1)
    thr = np.percentile(flat, 30)
    bg = flat[flat <= thr].astype(np.float32)
    mu, sd = (bg.mean(), bg.std()) if bg.size else (0.0, 1.0)
    out = np.empty_like(img)
    if dy > 0:
        out[dy:] = img[:H - dy]
        out[:dy] = np.clip(rng.normal(mu, sd, (dy, img.shape[1])), 0, 255)
    else:
        k = -dy
        out[:H - k] = img[k:]
        out[H - k:] = np.clip(rng.normal(mu, sd, (k, img.shape[1])), 0, 255)
    return out


class BoundaryDataset(Dataset):
    def __init__(self, rows, data_dir, train=False, sigma=1.5, hflip=True, vshift=40,
                 intensity=True, seed=0):
        self.rows = rows
        self.data_dir = data_dir
        self.train = train
        self.sigma = sigma
        self.hflip = hflip
        self.vshift = vshift
        self.intensity = intensity
        self.seed = seed
        self.C = len(BOUNDARIES)

    def __len__(self):
        return len(self.rows)

    def load_raw(self, i):
        d = np.load(os.path.join(self.data_dir, self.rows[i]["npz"]))
        return d["image"], d["runs"].astype(np.float32)

    def __getitem__(self, i):
        img, runs = self.load_raw(i)
        H, W = img.shape
        flipped = 0
        if self.train:
            rng = np.random.default_rng()
            if self.hflip and rng.random() < 0.5:
                img = img[:, ::-1].copy()
                runs = runs.copy()
                runs[:, 1] = (W - 1) - runs[:, 1]
                flipped = 1
            if self.vshift > 0:
                dy = int(rng.integers(-self.vshift, self.vshift + 1))
                if dy:
                    img = _shift_rows(img, dy, rng)
                    runs = runs.copy()
                    runs[:, 2] += dy
                    runs = runs[(runs[:, 2] >= 0) & (runs[:, 2] <= H - 1)]
        target, valid, nruns, gt_y = build_target(runs, self.C, H, W, self.sigma)
        x = img.astype(np.float32)
        if self.train and self.intensity:
            rng = np.random.default_rng()
            g = rng.uniform(0.7, 1.4)
            x = 255.0 * (x / 255.0) ** g
            x = x * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15)
            if rng.random() < 0.5:
                x = x + rng.normal(0, rng.uniform(0, 8), x.shape)
            x = np.clip(x, 0, 255)
        x = standardize(x)
        return dict(
            image=torch.from_numpy(x[None].copy()),
            target=torch.from_numpy(target),
            valid=torch.from_numpy(valid),
            single=torch.from_numpy((nruns == 1) & valid),
            gt_y=torch.from_numpy(gt_y),
            index=i,
            hflipped=flipped,
        )
