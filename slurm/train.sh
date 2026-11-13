#!/bin/bash
#SBATCH --job-name=octb_train
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --licenses=horse
#SBATCH --output=logs/%x_%j.log
# usage: sbatch --export=ALL,EXP=E0,MICRO=8 slurm/train.sh     (EXP = E0 | E1 | E1_lo | E1_hi)
# Effective batch 16, 30 epochs for every experiment. MICRO only changes memory, not the result.
set -eo pipefail
source slurm/env.sh
set -u
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "CUDA NOT AVAILABLE - abort"; exit 1; }
EXP="${EXP:?set EXP=E0 or E1}"
MICRO="${MICRO:-8}"
LR="${LR:-3e-4}"
case "$EXP" in
  E0)    EXTRA="--lambda-order 0    --lambda-smooth 0" ;;
  E1)    EXTRA="--lambda-order 0.1  --lambda-smooth 0.01" ;;
  E1_lo) EXTRA="--lambda-order 0.03 --lambda-smooth 0.003" ;;
  E1_hi) EXTRA="--lambda-order 0.3  --lambda-smooth 0.03" ;;
  *) echo "unknown EXP $EXP"; exit 1 ;;
esac
RESUME=""
[ -f "$RUNS/$EXP/last.pt" ] && [ ! -f "$RUNS/$EXP/DONE.json" ] && RESUME="--resume"
python 04_train.py --data-dir "$DATA" --out "$RUNS/$EXP" --epochs 30 --bs 16 --micro-bs "$MICRO" --lr "$LR" \
       --seed 42 --workers 7 $EXTRA $RESUME
python 05_evaluate.py --ckpt "$RUNS/$EXP/best.pt" --data-dir "$DATA" --split val \
       --out "$RUNS/$EXP/eval_val" --postprocess --n-overlays 16
