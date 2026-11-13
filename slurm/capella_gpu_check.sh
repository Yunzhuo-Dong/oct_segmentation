#!/bin/bash
#SBATCH --job-name=octb_gpucheck
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --licenses=horse
#SBATCH --output=logs/%x_%j.log
# Capella. Run AFTER barnard_prepare.sh. Checks CUDA + bs16 memory for micro-batch 16/8/4.
set -eo pipefail
source slurm/env.sh
set -u
python 00_check_env.py --write-dirs "$RUNS"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE in diffusion_env'"
for MB in 16 8 4; do
  echo "=== micro-bs $MB"
  python 04_train.py --data-dir "$DATA" --out "$RUNS/_probe_mb$MB" --force --epochs 1 --bs 16 --micro-bs $MB \
         --max-train-batches 4 --max-val-batches 2 --workers 6 --lambda-order 0.1 --lambda-smooth 0.01 \
         --constraint-start-epoch 0 2>&1 | grep -E "effective|step|\[ep|Error|error" || echo "micro-bs $MB FAILED"
done
