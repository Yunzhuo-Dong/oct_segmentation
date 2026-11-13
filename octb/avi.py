"""AVI helpers. Raw frames are ~800 (W) x 2048 (H), 8-bit.
Supervisor ROI: x=30, y=26, w=770, h=1000 (excludes top artefacts, left margin)."""
import numpy as np

SUPERVISOR_CROP = (30, 26, 770, 1000)  # x, y, w, h


def parse_crop(s):
    if s is None or s == "supervisor":
        return SUPERVISOR_CROP
    x, y, w, h = map(int, s.split(","))
    return x, y, w, h


def read_frames(path, indices=None):
    """Yield (index, gray uint8 frame). Sequential read (robust for raw AVI)."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    want = None if indices is None else set(int(i) for i in indices)
    last = -1 if want is None else max(want)
    i = 0
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if want is None or i in want:
                if fr.ndim == 3:
                    fr = fr[..., 1] if np.all(fr[..., 0] == fr[..., 1]) else cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                yield i, np.ascontiguousarray(fr)
            i += 1
            if want is not None and i > last:
                break
    finally:
        cap.release()


def frame_count(path):
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def crop(frame, box):
    x, y, w, h = box
    H, W = frame.shape
    if y + h > H or x + w > W:
        raise ValueError(f"crop {box} exceeds frame {frame.shape}")
    return frame[y:y + h, x:x + w]
