"""Reading MATLAB grdTruthOutput annotations and source images."""
import os

import numpy as np

from .common import norm_boundary_name


def _get(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
        v = obj[name]
        return v.item() if v.size == 1 else v
    return getattr(obj, name, default)


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray) and x.dtype == object:
        return list(x.ravel())
    if isinstance(x, np.ndarray) and x.dtype.names:
        return list(x.ravel())
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _to_str(x):
    if x is None:
        return ""
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return ""
        if x.dtype.kind in "US":
            return "".join(str(v) for v in x.ravel())
        if x.dtype == object and x.size == 1:
            return _to_str(x.ravel()[0])
    return str(x)


def basename_any(path_str):
    """basename that also handles Windows paths stored inside .mat files."""
    return str(path_str).replace("\\", "/").rstrip("/").split("/")[-1]


def load_annotation(mat_path):
    """Return dict(source_file, annotator, labels=[dict(raw_name, name, cls, mask)]).

    Raises ValueError with a short reason on unreadable / unexpected files.
    """
    from scipy.io import loadmat
    try:
        m = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    except NotImplementedError:
        raise ValueError("mat_v7.3_hdf5_not_supported")
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"loadmat_failed:{type(e).__name__}:{e}")

    grd = m.get("grdTruthOutput")
    if grd is None:
        for k, v in m.items():
            if not k.startswith("__") and _get(v, "LabelData") is not None:
                grd = v
                break
    if grd is None:
        raise ValueError("no_grdTruthOutput")

    src = _get(grd, "Source")
    src_file = _to_str(_get(src, "file", src if isinstance(src, str) else None))
    param = _get(grd, "Parameter")
    annot = _to_str(_get(param, "userID"))

    labels = []
    for ld in _as_list(_get(grd, "LabelData")):
        raw = _to_str(_get(ld, "Name"))
        cls = _to_str(_get(ld, "Class"))
        mask = _get(ld, "labelMask")
        if mask is not None:
            try:
                import scipy.sparse as sp
                if sp.issparse(mask):
                    mask = mask.toarray()
            except ImportError:
                pass
            mask = np.asarray(mask)
        labels.append(dict(raw_name=raw, name=norm_boundary_name(raw), cls=cls, mask=mask))
    return dict(source_file=src_file, annotator=annot, labels=labels)


def read_image_gray(path):
    """Read a TIFF/PNG as 2-D uint8. Returns (img_uint8, info_dict)."""
    arr = None
    try:
        import tifffile
        if str(path).lower().endswith((".tif", ".tiff")):
            arr = tifffile.imread(path)
    except Exception:  # noqa: BLE001
        arr = None
    if arr is None:
        from PIL import Image
        with Image.open(path) as im:
            arr = np.array(im)
    arr = np.asarray(arr)
    info = dict(orig_dtype=str(arr.dtype), orig_shape="x".join(map(str, arr.shape)))
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[..., :3].astype(np.float32).mean(-1)
        elif arr.shape[0] in (3, 4) and arr.shape[-1] > 4:
            arr = arr[:3].astype(np.float32).mean(0)
        else:
            arr = arr[0]
            info["multipage_first_page_used"] = 1
    if arr.ndim != 2:
        raise ValueError(f"unsupported_image_shape:{info['orig_shape']}")
    if arr.dtype == np.uint8:
        return arr, info
    a = arr.astype(np.float32)
    if info["orig_dtype"] in ("float32", "float64") and a.max() <= 1.0 + 1e-6 and a.min() >= 0:
        out = a * 255.0
    elif info["orig_dtype"] in ("uint8",) or (a.min() >= 0 and a.max() <= 255):
        out = a
    else:
        lo, hi = np.percentile(a, [0.1, 99.9])
        out = (a - lo) / max(hi - lo, 1e-6) * 255.0
        info["rescaled_percentile"] = f"{lo:.3g}:{hi:.3g}"
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), info


def resize_gray(img, hw):
    h, w = hw
    if img.shape == (h, w):
        return img
    try:
        import cv2
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image
        return np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def index_images(roots, exts=(".tif", ".tiff")):
    """Map lowercase basename and lowercase stem -> list of full paths."""
    by_name, by_stem = {}, {}
    for root in roots:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith(exts):
                    p = os.path.join(dp, fn)
                    by_name.setdefault(fn.lower(), []).append(p)
                    by_stem.setdefault(os.path.splitext(fn)[0].lower(), []).append(p)
    return by_name, by_stem
