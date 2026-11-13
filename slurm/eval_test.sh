#!/bin/bash
#SBATCH --job-name=octb_test
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --licenses=horse
#SBATCH --output=logs/%x_%j.log
# FINAL comparison only, after all choices were made on val:
# sbatch --export=ALL,EXPS="E0 E1" slurm/eval_test.sh
set -eo pipefail
source slurm/env.sh
set -u
for EXP in ${EXPS:?}; do
  python 05_evaluate.py --ckpt "$RUNS/$EXP/best.pt" --data-dir "$DATA" --split test --confirm-test \
         --out "$RUNS/$EXP/eval_test" --postprocess --n-overlays 24 --save-pred
done
