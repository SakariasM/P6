#!/bin/bash
#SBATCH --job-name=kd-train
#SBATCH --output=/ceph/project/P6-Machine-Vision/P6/logs/train_%j.out
#SBATCH --error=/ceph/project/P6-Machine-Vision/P6/logs/train_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Phase 2: Train student model with hybrid distillation.
# Uses chunk-epoch training: loads one prediction chunk at a time to avoid OOM.
# Supports incremental training: automatically resumes from the latest checkpoint.
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
OUT_DIR="${OUTPUT_DIR:-/ceph/project/P6-Machine-Vision/P6/trained_models}"

echo "=== Student Training ==="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH}"
echo "LR: ${LR}"
echo "Feature weight: ${FEAT_W}, Response weight: ${RESP_W}"
echo "Temperature: ${TEMP}, Distance: ${FEAT_DIST}"
echo ""

mkdir -p "${OUT_DIR}"

singularity exec --nv \
    -B "${VENV_DIR}:/scratch/venv" \
    -B "${PROJECT_DIR}:/scratch/project" \
    -B "/ceph/project/P6-Machine-Vision/P6:/ceph/project/P6-Machine-Vision/P6" \
    "${CONTAINER}" \
    /bin/bash -c "
        source /scratch/venv/bin/activate
        cd /scratch/project
        export PYTHONPATH=/scratch/project/src
        export TMPDIR=/ceph/project/P6-Machine-Vision/P6/tmp/train_\${SLURM_JOB_ID}
        mkdir -p \"\${TMPDIR}\"

        nvidia-smi

        # Resume from latest training checkpoint if one exists
        RESUME_ARG=''
        LATEST_CKPT=\$(find '${OUT_DIR}' -name 'checkpoint_epoch_*.pt' 2>/dev/null | sort -V | tail -1)
        if [ -n \"\${LATEST_CKPT}\" ]; then
            echo \"Resuming from: \${LATEST_CKPT}\"
            RESUME_ARG=\"--resume \${LATEST_CKPT}\"
        fi

        echo ''
        echo '=== Starting training ==='
        echo 'Predictions dir: ${PRED_DIR}'

        python -u -m training.hybrid_distillation_train \
            --predictions '${PRED_DIR}' \
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
