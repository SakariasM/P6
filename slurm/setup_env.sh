#!/bin/bash
# Sets up a virtual environment inside the AAU AI-LAB Singularity container.
# Run this ONCE after cloning the repo to the cluster.
#
# Usage:
#   srun singularity exec /ceph/container/pytorch/pytorch_25.12.sif bash slurm/setup_env.sh

set -euo pipefail

CONTAINER="/ceph/container/pytorch/pytorch_25.12.sif"
VENV_DIR="$HOME/kd_venv"

echo "=== Setting up environment ==="
echo "Node: $(hostname)"
echo "Container: ${CONTAINER}"
echo ""

# Create venv inheriting PyTorch from the container
if [ ! -d "${VENV_DIR}" ]; then
    python -m venv --system-site-packages "${VENV_DIR}"
    echo "Created virtualenv at ${VENV_DIR}"
else
    echo "Virtualenv already exists at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

# Install extra deps not in the container
pip install --no-cache-dir ultralytics tqdm pillow pytest opencv-python-headless

# Verify
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import ultralytics; print(f'Ultralytics {ultralytics.__version__}')"
python -c "import tqdm, PIL, pytest; print('tqdm, PIL, pytest: OK')"

echo ""
echo "=== Environment ready ==="
echo "Venv location: ${VENV_DIR}"
