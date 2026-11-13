#!/bin/bash
#SBATCH --job-name=octb_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
# Barnard (CPU). Submit from $HOME/py_coding/oct_bseg:  sbatch slurm/barnard_smoke.sh
set -eo pipefail
source slurm/barnard_env.sh
set -u
bash tests/smoke_test.sh
