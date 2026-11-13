# Capella GPU environment. Sourced by capella job scripts.
# Only the Python base module is loaded: torch comes from diffusion_env itself.
# (Loading the PyTorch/2.1.2-CUDA-12.1.1 module on top of the venv's own torch
#  mixes two CUDA stacks.) To go back to the old setup:
#  export OCTB_MODULES="release/24.04 GCC/12.3.0 OpenMPI/4.1.5 PyTorch"
OCTB_MODULES="${OCTB_MODULES:-release/24.04 GCCcore/12.3.0 Python/3.11.3}"
module purge 2>/dev/null
module load $OCTB_MODULES
source "$HOME/diffusion_env/bin/activate"
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
python -c 'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || true
