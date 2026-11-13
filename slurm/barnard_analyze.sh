#!/bin/bash
#SBATCH --job-name=octb_analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --licenses=horse
#SBATCH --output=logs/%x_%j.log
# Barnard (CPU). Submit from $HOME/py_coding/oct_bseg:  sbatch slurm/barnard_analyze.sh
# If sbatch rejects --licenses=horse, delete that line.
set -eo pipefail
source slurm/barnard_env.sh
set -u
MODELS="${MODELS:-E0 E1}"
SPEC=""
for M in $MODELS; do SPEC="$SPEC $M=$RUNS/$M/best.pt"; done
python 08_analyze.py --data-dir "$DATA" --models $SPEC --split val --out "$RUNS/analysis_val" --n-worst 8
