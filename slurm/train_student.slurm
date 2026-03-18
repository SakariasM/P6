#!/bin/bash
#SBATCH --job-name=kd-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00

# Phase 2: Train student model with hybrid distillation.
# Usage: sbatch slurm/train_student.slurm
#
# Override defaults:
#   EPOCHS=100 BATCH_SIZE=32 sbatch slurm/train_student.slurm

set -euo pipefail

CONTAINER="/ceph/container/pytorch/pytorch_25.08.sif"
VENV_DIR="$HOME/kd_venv"
PROJECT_DIR="$(pwd)"

# Configurable
PREDICTIONS="${PREDICTIONS_FILE:-./results/hybrid_predictions/hybrid_teacher_predictions.torch}"
IMAGE_ROOT="${DATA_DIR:-./data/images}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH_SIZE:-16}"
LR="${LEARNING_RATE:-1e-3}"
FEAT_W="${FEATURE_WEIGHT:-1.0}"
RESP_W="${RESPONSE_WEIGHT:-1.0}"
TEMP="${TEMPERATURE:-3.0}"
FEAT_DIST="${FEATURE_DISTANCE:-mse}"
MODEL_TYPE="${MODEL_TYPE:-standard}"
OUT_DIR="${OUTPUT_DIR:-./trained_models}"

echo "=== Student Training ==="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Model type: ${MODEL_TYPE}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH}"
echo "LR: ${LR}"
echo "Feature weight: ${FEAT_W}, Response weight: ${RESP_W}"
echo "Temperature: ${TEMP}, Distance: ${FEAT_DIST}"
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
        echo '=== Starting training ==='

        python src/training/hybrid_distillation_train.py \
            --predictions '${PREDICTIONS}' \
            --image-root '${IMAGE_ROOT}' \
            --model-type '${MODEL_TYPE}' \
            --epochs ${EPOCHS} \
            --batch-size ${BATCH} \
            --lr ${LR} \
            --feature-weight ${FEAT_W} \
            --response-weight ${RESP_W} \
            --temperature ${TEMP} \
            --feature-distance '${FEAT_DIST}' \
            --output-dir '${OUT_DIR}' \
            --save-interval 10

        echo ''
        echo '=== Training complete ==='
        ls -lh '${OUT_DIR}/'
    "
