#!/bin/bash
# SLURM job script for running the external COCO benchmark.
# Submit with: sbatch slurm/run_benchmark.sh

#SBATCH --job-name=unit-benchmark
#SBATCH --output=/ceph/project/P6-Machine-Vision/P6/logs/benchmarking/benchmark_%j.out
#SBATCH --error=/ceph/project/P6-Machine-Vision/P6/logs/benchmarking/benchmark_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=l4
#SBATCH --exclude=ailab-l4-02

# Exit immediately on error, treat unset variables as errors
set -euo pipefail

# --- Paths -----------------------------------------------------
CONTAINER="/ceph/container/pytorch/pytorch_25.12.sif"
VENV_DIR="$HOME/kd_venv"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"

# --- Job Info --------------------------------------------------
echo "=== Ekstern Benchmark ==="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"

# --- Create output directories if they do not exist ------------
# --nv: enables NVIDIA GPU passthrough
# -B:   binds host paths into the container
mkdir -p /ceph/project/P6-Machine-Vision/P6/results

# --- Run benchmark inside Singularity container ----------------
singularity exec --nv \
  -B "${VENV_DIR}:/scratch/venv" \
  -B "${PROJECT_DIR}:/scratch/project" \
  "${CONTAINER}" \
  /bin/bash -c "
    cd /scratch/project
    export PYTHONPATH=/scratch/project/src
    echo '=== Kører benchmark ==='
    source /scratch/venv/bin/activate && python src/benchmark_accuracy.py --n-max 5000
    echo '===== Benchmark complete ====='
    cat /ceph/project/P6-Machine-Vision/P6/results/benchmark_accuracy_results.json
  "
