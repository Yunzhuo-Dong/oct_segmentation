#!/bin/bash
#SBATCH --job-name=octb_avi
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --licenses=horse
#SBATCH --output=logs/%x_%j.log
# sbatch --export=ALL,AVI=/path/x.avi,TIF=/path/x.tif,EXP=E0,NAME=BL-F379 slurm/avi.sh
# ONH (optional): add ONH="frame,x,frame_radius,x_radius" in crop coordinates
set -eo pipefail
source slurm/env.sh
set -u
OUT="$RUNS/avi/${NAME:?}_${EXP:?}"
mkdir -p "$OUT"
python 06_avi_match.py --avi "${AVI:?}" --tiff "${TIF:?}" --data-dir "$DATA" --out "$OUT/match.json"
python 07_avi_infer.py --avi "$AVI" --ckpt "$RUNS/$EXP/best.pt" --data-dir "$DATA" --match "$OUT/match.json" \
       --out "$OUT" --step 1 --postprocess ${ONH:+--onh "$ONH"}
