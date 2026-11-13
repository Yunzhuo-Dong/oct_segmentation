"""Shared constants and small helpers (no torch dependency)."""
import csv
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np

# Channel order of the network output. This is ONLY a channel index, not an
# anatomical claim: the anatomical order is derived from the data by
# 02_prepare.py and written to order_constraints.json.
BOUNDARIES = ["RNFL", "GCL", "IPL", "INL", "OPL", "ONL", "OLM", "IS", "OS", "RPE", "CHOROID"]
BIDX = {n: i for i, n in enumerate(BOUNDARIES)}
NAME_ALIASES = {"CHROROID": "CHOROID", "CHORIOD": "CHOROID", "CHOROIDEA": "CHOROID"}


def norm_boundary_name(name):
    """Map a raw MATLAB label name to one of BOUNDARIES, or None if unknown."""
    s = re.sub(r"[^A-Z]", "", str(name).upper())
    s = NAME_ALIASES.get(s, s)
    return s if s in BIDX else None


# Animal IDs seen so far: BL-F388, MA-M832, MA-N578 and MA-N-062 (extra dash).
# All are normalised to XX-Y123, so MA-N-062 and MA-N062 count as the same animal.
DEFAULT_ANIMAL_RE = r"(?<![A-Za-z])([A-Za-z]{2})[-_]?([A-Za-z])[-_]?(\d{3})"
DATE_RE = re.compile(r"(?<!\d)(\d{6}|\d{8})(?!\d)")
EYE_RE = re.compile(r"(?:^|[_\-\s])([lrLR])(?=[_\-\s.]|$)")


def parse_ids(path, animal_re=DEFAULT_ANIMAL_RE):
    """Return (animal_id, eye, date) parsed from a file path; '' when not found."""
    name = os.path.basename(str(path))
    rx = re.compile(animal_re)
    animal = ""
    for text in (name, str(path)):
        m = rx.search(text)
        if m:
            animal = (f"{m.group(1)}-{m.group(2)}{m.group(3)}" if m.re.groups >= 3 else m.group(0)).upper()
            break
    m = EYE_RE.search(os.path.splitext(name)[0])
    eye = m.group(1).lower() if m else ""
    m = DATE_RE.search(name)
    date = m.group(1) if m else ""
    return animal, eye, date


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_test(directory):
    """Fail early if a directory is not writable (HORSE Errno 5 history)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f".write_test_{os.getpid()}_{int(time.time())}"
    with open(p, "wb") as f:
        f.write(os.urandom(1024 * 1024))
        f.flush()
        os.fsync(f.fileno())
    p.unlink()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def save_json(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_csv(rows, path, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def nanstats(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(n=0, mean=float("nan"), median=float("nan"), p90=float("nan"))
    return dict(n=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.percentile(a, 90)))
