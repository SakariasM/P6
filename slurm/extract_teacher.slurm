#!/bin/bash
#SBATCH --job-name=kd-extract
#SBATCH --output=/ceph/project/P6-Machine-Vision/P6/logs/extract_%A_%a.out
#SBATCH --error=/ceph/project/P6-Machine-Vision/P6/logs/extract_%A_%a.err
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00

# Phase 1: Extract teacher predictions + intermediate features.
# Runs as a 4-worker array job — each worker processes 1/4 of the images in parallel.
# Usage: sbatch slurm/extract_teacher.slurm
#
# After all 4 workers complete, merge with:
#   python src/teacher/merge_predictions.py \
#     --input /ceph/project/P6-Machine-Vision/P6/results/hybrid_predictions \
#     --output /ceph/project/P6-Machine-Vision/P6/results/hybrid_predictions/hybrid_teacher_predictions.torch

set -euo pipefail

CONTAINER="/ceph/container/pytorch/pytorch_25.12.sif"
VENV_DIR="$HOME/kd_venv"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
NUM_WORKERS=4
WORKER_ID=${SLURM_ARRAY_TASK_ID}

# Configurable via env vars
MODEL="${TEACHER_MODEL:-/scratch/project/yolo26n-seg.pt}"
INPUT="${DATA_DIR:-/ceph/project/P6-Machine-Vision/P6/data/open-images-v7/train/data}"
OUTPUT="${OUTPUT_DIR:-/ceph/project/P6-Machine-Vision/P6/results/hybrid_predictions}"
BATCH="${BATCH_SIZE:-8}"

echo "=== Teacher Feature Extraction (worker ${WORKER_ID}/${NUM_WORKERS}) ==="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}, Array task: ${WORKER_ID}"
echo "Model: ${MODEL}"
echo "Input: ${INPUT}"
echo "Output: ${OUTPUT}"
echo "Batch size: ${BATCH}"
echo ""

singularity exec --nv \
    -B "${VENV_DIR}:/scratch/venv" \
    -B "${PROJECT_DIR}:/scratch/project" \
    -B "/ceph/project/P6-Machine-Vision/P6:/ceph/project/P6-Machine-Vision/P6" \
    "${CONTAINER}" \
    /bin/bash -c "
        source /scratch/venv/bin/activate
        cd /scratch/project
        export PYTHONPATH=/scratch/project/src

        nvidia-smi

        echo ''
        echo '=== Starting extraction (worker ${WORKER_ID}) ==='

        python -m teacher.hybrid_predictions \
            --model '${MODEL}' \
            --input '${INPUT}' \
            --output '${OUTPUT}' \
            --batch-size ${BATCH} \
            --person-only \
            --checkpoint-interval 500 \
            --worker-id ${WORKER_ID} \
            --num-workers ${NUM_WORKERS}

        echo ''
        echo '=== Worker ${WORKER_ID} complete ==='
        ls -lh '${OUTPUT}/'
    "
