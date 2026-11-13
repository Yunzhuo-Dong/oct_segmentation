#!/bin/bash
# End-to-end test on fake data (CPU, ~2 min). Run once after copying the code:
#   bash tests/smoke_test.sh
set -eo pipefail
cd "$(dirname "$0")/.."
T="${TMPDIR:-/tmp}/octb_smoke_$$"
python tests/make_synthetic.py "$T/raw"
python 01_inspect_mat.py --mat-root "$T/raw" --n 1
python 02_prepare.py --mat-root "$T/raw" --out "$T/data" --workers 2 --order-min-support 50
python 03_review_overlays.py --data-dir "$T/data" --out "$T/review" --n 3
python 04_train.py --data-dir "$T/data" --out "$T/E1" --base 8 --depth 4 --epochs 3 --workers 0 --bs 4 --micro-bs 2 \
       --lr 2e-3 --warmup-iters 5 --lambda-order 0.1 --lambda-smooth 0.01 --constraint-start-epoch 1
python 05_evaluate.py --ckpt "$T/E1/best.pt" --data-dir "$T/data" --split val --out "$T/eval" --postprocess --n-overlays 2
AVI=$(ls "$T"/raw/*.avi 2>/dev/null | head -1 || true)
if [ -n "$AVI" ]; then
  TIF="$T/raw/training/$(basename "${AVI%.avi}").tif"
  python 06_avi_match.py --avi "$AVI" --tiff "$TIF" --data-dir "$T/data" --out "$T/match.json" --frame-step 2 --coarse 2
  python 07_avi_infer.py --avi "$AVI" --ckpt "$T/E1/best.pt" --data-dir "$T/data" --match "$T/match.json" \
         --crop 20,26,212,320 --out "$T/avi" --overlay-every 8
else
  echo "(opencv missing: AVI steps skipped)"
fi
echo "SMOKE TEST PASSED -> $T"
