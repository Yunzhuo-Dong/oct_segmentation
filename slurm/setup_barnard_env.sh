#!/bin/bash
# Run ONCE on a Barnard LOGIN node (needs internet):  bash slurm/setup_barnard_env.sh
set -eo pipefail
OCTB_MODULES="${OCTB_MODULES:-release/24.04 GCCcore/12.3.0 Python/3.11.3}"
module purge 2>/dev/null || true
module load $OCTB_MODULES
python -m venv "$HOME/octb_cpu_env"
source "$HOME/octb_cpu_env/bin/activate"
pip install --upgrade pip
pip install numpy scipy matplotlib pillow tifffile opencv-python-headless
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -c "import numpy, scipy, cv2, tifffile, torch; print('octb_cpu_env OK, torch', torch.__version__)"
