#!/usr/bin/env python
"""Step 0: check Python packages, GPU and that output dirs are writable.
python 00_check_env.py --write-dirs "$WS/octb_data_v1" "$WS/octb_runs" "$HOME/octb_review"
"""
import argparse
import importlib
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from octb.common import write_test  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--write-dirs", nargs="*", default=[])
args = ap.parse_args()
print("python", platform.python_version(), sys.executable)
need = {"numpy": True, "scipy": True, "torch": True, "matplotlib": True,
        "cv2": False, "tifffile": False, "PIL": True}
bad = False
for mod, required in need.items():
    try:
        m = importlib.import_module(mod)
        print(f"  ok   {mod:<10} {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        print(f"  {'MISSING' if required else 'missing'} {mod:<10} ({e.__class__.__name__})"
              + ("" if required else "  -> optional; pip install --user " +
                 {"cv2": "opencv-python-headless", "tifffile": "tifffile"}[mod]))
        bad |= required
try:
    import torch
    print("  cuda available:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    if torch.cuda.is_available():
        print("  bf16 supported:", torch.cuda.is_bf16_supported())
        x = torch.randn(1, 4, 64, 64, device="cuda")
        print("  fft check:", torch.fft.irfft2(torch.fft.rfft2(x), s=(64, 64)).sub(x).abs().max().item())
except Exception as e:  # noqa: BLE001
    print("  torch check failed:", e)
for d in args.write_dirs:
    try:
        write_test(d)
        print(f"  writable: {d}")
    except Exception as e:  # noqa: BLE001
        print(f"  NOT WRITABLE: {d}  ({e})")
        bad = True
sys.exit(1 if bad else 0)
