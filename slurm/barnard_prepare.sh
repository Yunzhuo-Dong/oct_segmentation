#!/bin/bash
#SBATCH --job-name=octb_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --licenses=horse
#SBATCH --output=logs/%x_%j.log
# Barnard (CPU). Submit from $HOME/py_coding/oct_bseg:  sbatch slurm/barnard_prepare.sh
# Protected-test variant:  sbatch --export=ALL,SPLIT_MODE=source,DATA=$WS/octb_data_src slurm/barnard_prepare.sh
# If sbatch rejects --licenses=horse on Barnard, delete that line.
set -eo pipefail
source slurm/barnard_env.sh
set -u
SPLIT_MODE="${SPLIT_MODE:-animal}"
python 02_prepare.py --mat-root "$MAT_BASE" --out "$DATA" --split-mode "$SPLIT_MODE" --workers 15
python 03_review_overlays.py --data-dir "$DATA" --out "$HOME/octb_review" --n 20
echo "== summary"; python -c "import json;d=json.load(open('$DATA/summary.json'));[print(k,':',d[k]) for k in ('split_counts','n_animals','overlap','orientation','chain','n_constraint_pairs','unparsed_animal_images','unknown_names','experiments')]"
echo "== order_table.csv"; cat "$DATA/order_table.csv"
echo "== boundary_inventory.csv"; cat "$DATA/boundary_inventory.csv"
echo "== skipped reasons"; cut -d, -f2 "$DATA/skipped.csv" | cut -d: -f1 | sort | uniq -c
