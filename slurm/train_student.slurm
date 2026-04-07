#!/bin/bash
#SBATCH --job-name=kd-train
#SBATCH --output=/ceph/project/P6-Machine-Vision/P6/logs/train_%j.out
#SBATCH --error=/ceph/project/P6-Machine-Vision/P6/logs/train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Phase 2: Train student segmentation model with attention-based distillation.
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
BATCH="${BATCH_SIZE:-8}"
LR="${LEARNING_RATE:-1e-4}"
ATT_W="${ATTENTION_WEIGHT:-1.0}"
MIM_W="${MIMICRY_WEIGHT:-0.5}"
REL_W="${RELATION_WEIGHT:-0.5}"
BASE_CH="${BASE_CHANNELS:-8}"
DEPTH="${DEPTH:-4}"
IMG_SIZE="${IMG_SIZE:-640}"
WORKERS="${NUM_WORKERS:-8}"
SAVE_INT="${SAVE_INTERVAL:-2}"
OUT_DIR="${OUTPUT_DIR:-/ceph/project/P6-Machine-Vision/P6/trained_models}"

echo "=== Segmentation Distillation Training ==="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH}"
echo "LR: ${LR}"
echo "Loss weights: att=${ATT_W}, mim=${MIM_W}, rel=${REL_W}"
echo "Student: base_ch=${BASE_CH}, depth=${DEPTH}, img=${IMG_SIZE}"
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
            --attention-weight ${ATT_W} \
            --mimicry-weight ${MIM_W} \
            --relation-weight ${REL_W} \
            --base-channels ${BASE_CH} \
            --depth ${DEPTH} \
            --img-size ${IMG_SIZE} \
            --num-workers ${WORKERS} \
            --output-dir '${OUT_DIR}' \
            --save-interval ${SAVE_INT} \
            \${RESUME_ARG}

        echo ''
        echo '=== Training complete ==='
        ls -lh '${OUT_DIR}/'
    "
