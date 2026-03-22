#!/bin/bash
#SBATCH --job-name=kd-extract
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00

# Phase 1: Extract teacher predictions + intermediate features.
# Usage: sbatch slurm/extract_teacher.slurm
#
# Override defaults:
#   TEACHER_MODEL=yolo26n.pt DATA_DIR=./data/images sbatch slurm/extract_teacher.slurm

set -euo pipefail

CONTAINER="/ceph/container/pytorch/pytorch_25.08.sif"
VENV_DIR="$HOME/kd_venv"
PROJECT_DIR="$(pwd)"

# Configurable via env vars
MODEL="${TEACHER_MODEL:-yolo26n.pt}"
INPUT="${DATA_DIR:-./data/images}"
OUTPUT="${OUTPUT_DIR:-./results/hybrid_predictions}"
BATCH="${BATCH_SIZE:-8}"

echo "=== Teacher Feature Extraction ==="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Model: ${MODEL}"
echo "Input: ${INPUT}"
echo "Output: ${OUTPUT}"
echo "Batch size: ${BATCH}"
echo ""

mkdir -p logs

singularity exec --nv \
    -B "${VENV_DIR}:/scratch/venv" \
    -B "${PROJECT_DIR}:/scratch/project" \
    "${CONTAINER}" \
    /bin/bash -c "
        source /scratch/venv/bin/activate
        cd /scratch/project

        nvidia-smi

        echo ''
        echo '=== Starting extraction ==='

        python src/teacher/hybrid_predictions.py \
            --model '${MODEL}' \
            --input '${INPUT}' \
            --output '${OUTPUT}' \
            --batch-size ${BATCH} \
            --person-only \
            --checkpoint-interval 50

        echo ''
        echo '=== Extraction complete ==='
        ls -lh '${OUTPUT}/'
    "
