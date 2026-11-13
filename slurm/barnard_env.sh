# Barnard CPU environment (separate venv: diffusion_env was built on Capella).
OCTB_MODULES="${OCTB_MODULES:-release/24.04 GCCcore/12.3.0 Python/3.11.3}"
module purge 2>/dev/null
module load $OCTB_MODULES || { echo "module load failed; try: module spider Python"; exit 1; }
source "$HOME/octb_cpu_env/bin/activate" || { echo "run: bash slurm/setup_barnard_env.sh (login node)"; exit 1; }
export WS="/data/horse/ws/yudo868h-Yunzhuo_space"
export MAT_BASE="$WS/OCT_Retina_Segmentierung/OCT Retina Segmentierung/Segmented_images/Segmented_images"
export CODE="$HOME/py_coding/oct_bseg"
export DATA="${DATA:-$WS/octb_data_v1}"
export RUNS="$WS/octb_runs"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$CODE"
echo "== job ${SLURM_JOB_ID:-local} on $(hostname) $(date)"
python -V
