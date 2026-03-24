#!/bin/bash
#SBATCH --job-name=kd-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00

# Phase 2: Train student model with hybrid distillation.
# Supports incremental training: automatically resumes from the latest training
# checkpoint and uses checkpoint.torch from extraction if the final predictions
# file is not yet available.
#
# Usage: sbatch slurm/train_student.slurm
#
# Override defaults:
#   EPOCHS=100 BATCH_SIZE=32 sbatch slurm/train_student.slurm

set -euo pipefail

CONTAINER="/ceph/container/pytorch/pytorch_25.12.sif"
VENV_DIR="$HOME/kd_venv"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"

# Configurable
PRED_DIR="${PRED_DIR:-/ceph/project/P6-Machine-Vision/P6/results/hybrid_predictions}"
IMAGE_ROOT="${DATA_DIR:-/ceph/project/P6-Machine-Vision/P6/data/open-images-v7/train/data}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH_SIZE:-16}"
LR="${LEARNING_RATE:-1e-3}"
FEAT_W="${FEATURE_WEIGHT:-1.0}"
RESP_W="${RESPONSE_WEIGHT:-1.0}"
TEMP="${TEMPERATURE:-3.0}"
FEAT_DIST="${FEATURE_DISTANCE:-mse}"
MODEL_TYPE="${MODEL_TYPE:-standard}"
OUT_DIR="${OUTPUT_DIR:-/ceph/project/P6-Machine-Vision/P6/trained_models}"

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

mkdir -p logs "${OUT_DIR}"

singularity exec --nv \
    -B "${VENV_DIR}:/scratch/venv" \
    -B "${PROJECT_DIR}:/scratch/project" \
    -B "/ceph/project/P6-Machine-Vision/P6/data:/ceph/project/P6-Machine-Vision/P6/data" \
    "${CONTAINER}" \
    /bin/bash -c "
        source /scratch/venv/bin/activate
        cd /scratch/project
        export PYTHONPATH=/scratch/project/src

        nvidia-smi

        # Pick predictions file: prefer final output, fall back to in-progress checkpoint
        if [ -f '${PRED_DIR}/hybrid_teacher_predictions.torch' ]; then
            PREDICTIONS='${PRED_DIR}/hybrid_teacher_predictions.torch'
        elif [ -f '${PRED_DIR}/checkpoint.torch' ]; then
            PREDICTIONS='${PRED_DIR}/checkpoint.torch'
        else
            echo 'ERROR: No predictions file found in ${PRED_DIR}'
            exit 1
        fi
        echo \"Using predictions: \${PREDICTIONS}\"

        # Resume from latest training checkpoint if one exists
        RESUME_ARG=''
        LATEST_CKPT=\$(find '${OUT_DIR}' -name 'checkpoint_epoch_*.pt' 2>/dev/null | sort -V | tail -1)
        if [ -n \"\${LATEST_CKPT}\" ]; then
            echo \"Resuming from: \${LATEST_CKPT}\"
            RESUME_ARG=\"--resume \${LATEST_CKPT}\"
        fi

        echo ''
        echo '=== Starting training ==='

        python -m training.hybrid_distillation_train \
            --predictions \"\${PREDICTIONS}\" \
            --image-root '${IMAGE_ROOT}' \
            --epochs ${EPOCHS} \
            --batch-size ${BATCH} \
            --lr ${LR} \
            --feature-weight ${FEAT_W} \
            --response-weight ${RESP_W} \
            --temperature ${TEMP} \
            --feature-distance '${FEAT_DIST}' \
            --output-dir '${OUT_DIR}' \
            --save-interval 10 \
            \${RESUME_ARG}

        echo ''
        echo '=== Training complete ==='
        ls -lh '${OUT_DIR}/'
    "
