#!/bin/bash
#SBATCH --job-name=kd-tests
#SBATCH --output=logs/tests_%j.out
#SBATCH --error=logs/tests_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00

# Run all tests on a GPU node to validate the pipeline before training.
# Usage: sbatch slurm/run_tests.slurm

set -euo pipefail

CONTAINER="/ceph/container/pytorch/pytorch_25.12.sif"
VENV_DIR="$HOME/kd_venv"
PROJECT_DIR="$(pwd)"

echo "=== Test Job ==="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo ""

mkdir -p logs

singularity exec --nv \
    -B "${VENV_DIR}:/scratch/venv" \
    -B "${PROJECT_DIR}:/scratch/project" \
    "${CONTAINER}" \
    /bin/bash -c "
        source /scratch/venv/bin/activate
        cd /scratch/project

        echo '=== Environment ==='
        python -c \"
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    free, total = torch.cuda.mem_get_info()
    print(f'Memory: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total')
\"

        echo ''
        echo '=== CPU tests ==='
        python -m pytest tests/ -v -k 'not cuda and not CUDA and not gpu and not slurm and not nccl and not num_workers' --tb=short

        echo ''
        echo '=== GPU tests ==='
        python -m pytest tests/ -v -k 'cuda or CUDA or gpu' --tb=short

        echo ''
        echo '=== All tests completed ==='
    "
