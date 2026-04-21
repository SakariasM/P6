#!/bin/bash
#SBATCH --job-name=kd-online
#SBATCH --output=/ceph/project/P6-Machine-Vision/P6/logs/train_online_%j.out
#SBATCH --error=/ceph/project/P6-Machine-Vision/P6/logs/train_online_%j.err
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# Online distillation: teacher + student run simultaneously on 4 GPUs.
# Uses torchrun for DistributedDataParallel.
#
# Usage: sbatch slurm/train_online.slurm
#
# Override defaults:
#   EPOCHS=100 BATCH_SIZE=16 sbatch slurm/train_online.slurm

set -euo pipefail

CONTAINER="/ceph/container/pytorch/pytorch_25.12.sif"
VENV_DIR="$HOME/kd_venv"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"

# Configurable
TEACHER="${TEACHER_MODEL:-/ceph/project/P6-Machine-Vision/P6/yolo26n-seg.pt}"
IMAGE_DIR="${DATA_DIR:-/ceph/project/P6-Machine-Vision/P6/data/open-images-v7/train/data}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH_SIZE:-16}"
LR="${LEARNING_RATE:-2e-4}"
ATT_W="${ATTENTION_WEIGHT:-1.0}"
MIM_W="${MIMICRY_WEIGHT:-0.5}"
REL_W="${RELATION_WEIGHT:-0.5}"
SEG_W="${SEG_WEIGHT:-1.0}"
BASE_CH="${BASE_CHANNELS:-32}"
DEPTH="${DEPTH:-4}"
IMG_SIZE="${IMG_SIZE:-640}"
WORKERS="${NUM_WORKERS:-8}"
SAVE_INT="${SAVE_INTERVAL:-10}"
OUT_DIR="${OUTPUT_DIR:-/ceph/project/P6-Machine-Vision/P6/trained_models_online}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EXCLUDE_LAYERS="${EXCLUDE_LAYERS:-}"

# Exclude one layer
#EXCLUDE_LAYERS="model.9" sbatch slurm/train_online.slurm

# Exclude multiple layers
#EXCLUDE_LAYERS="model.4 model.9" sbatch slurm/train_online.slurm

# Same works for the offline script
#EXCLUDE_LAYERS="model.9" sbatch slurm/train_student.slurm

# No exclusion (default — uses all layers)
sbatch slurm/train_online.slurm

echo "=== Online Distillation Training (8 GPU DDP) ==="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Teacher: ${TEACHER}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH} x 8 GPUs = $((BATCH * 8)) effective"
echo "LR: ${LR}"
echo "Loss weights: att=${ATT_W}, mim=${MIM_W}, rel=${REL_W}, seg=${SEG_W}"
echo "Student: base_ch=${BASE_CH}, depth=${DEPTH}, img=${IMG_SIZE}"
if [ -n "${EXCLUDE_LAYERS}" ]; then
    echo "Excluding teacher layers: ${EXCLUDE_LAYERS}"
fi
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
        export TMPDIR=/ceph/project/P6-Machine-Vision/P6/tmp/train_online_\${SLURM_JOB_ID}
        mkdir -p \"\${TMPDIR}\"

        nvidia-smi

        # Exclude layers arg (conditionally added)
        EXCLUDE_ARG=''
        if [ -n '${EXCLUDE_LAYERS}' ]; then
            EXCLUDE_ARG=\"--exclude-layers ${EXCLUDE_LAYERS}\"
        fi

        # Resume from latest checkpoint
        RESUME_ARG=''
        LATEST_CKPT=\$(find '${OUT_DIR}' -name 'checkpoint_epoch_*.pt' 2>/dev/null | sort -V | tail -1)
        if [ -n \"\${LATEST_CKPT}\" ]; then
            echo \"Resuming from: \${LATEST_CKPT}\"
            RESUME_ARG=\"--resume \${LATEST_CKPT}\"
        fi

        echo ''
        echo '=== Starting online distillation training ==='

        torchrun --nproc_per_node=8 \
            -m training.online_distillation_train \
            --teacher-model '${TEACHER}' \
            --image-dir '${IMAGE_DIR}' \
            --epochs ${EPOCHS} \
            --batch-size ${BATCH} \
            --lr ${LR} \
            --attention-weight ${ATT_W} \
            --mimicry-weight ${MIM_W} \
            --relation-weight ${REL_W} \
            --seg-weight ${SEG_W} \
            --base-channels ${BASE_CH} \
            --depth ${DEPTH} \
            --img-size ${IMG_SIZE} \
            --num-workers ${WORKERS} \
            --output-dir '${OUT_DIR}' \
            --save-interval ${SAVE_INT} \
            --grad-clip ${GRAD_CLIP} \
            \${EXCLUDE_ARG} \
            \${RESUME_ARG}

        echo ''
        echo '=== Training complete ==='
        ls -lh '${OUT_DIR}/'
    "
