#!/bin/bash
#SBATCH --job-name=kd-ablation
#SBATCH --output=/ceph/project/P6-Machine-Vision/P6/logs/ablation_%A_%a.out
#SBATCH --error=/ceph/project/P6-Machine-Vision/P6/logs/ablation_%A_%a.err
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Train one student model per layer configuration.
# Each array task reads its config from configs/ablation_configs.json.
#
# Array indices map to configs:
#   0 = shallow_only     (model.4)
#   1 = mid_only         (model.6)
#   2 = deep_only        (model.9)
#   3 = backbone_3       (model.4 + model.6 + model.9)  [current default]
#   4 = backbone_plus_neck (model.9 + model.12 + model.15)
#   5 = all_5_layers     (all 5 layers)
#
# Usage:
#   sbatch slurm/train_ablation.slurm              # all 6 configs
#   sbatch --array=0,3 slurm/train_ablation.slurm  # only shallow + backbone_3

set -euo pipefail

CONTAINER="/ceph/container/pytorch/pytorch_25.12.sif"
VENV_DIR="$HOME/kd_venv"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
CONFIG_IDX=${SLURM_ARRAY_TASK_ID}

PRED_DIR="${PRED_DIR:-/ceph/project/P6-Machine-Vision/P6/results/ablation_predictions}"
IMAGE_ROOT="${DATA_DIR:-/ceph/project/P6-Machine-Vision/P6/data/open-images-v7/train/data}"
BASE_OUT="/ceph/project/P6-Machine-Vision/P6/trained_models"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH_SIZE:-8}"
LR="${LEARNING_RATE:-1e-4}"
BASE_CH="${BASE_CHANNELS:-8}"
DEPTH="${DEPTH:-4}"
IMG_SIZE="${IMG_SIZE:-640}"
WORKERS="${NUM_WORKERS:-8}"

echo "=== Ablation Training (config ${CONFIG_IDX}) ==="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}, Array task: ${CONFIG_IDX}"
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
        export TMPDIR=/ceph/project/P6-Machine-Vision/P6/tmp/ablation_\${SLURM_ARRAY_TASK_ID}
        mkdir -p \"\${TMPDIR}\"

        nvidia-smi

        CONFIG_FILE='configs/ablation_configs.json'
        CONFIG_NAME=\$(python3 -c \"
import json
with open('\${CONFIG_FILE}') as f:
    configs = json.load(f)
c = configs[${CONFIG_IDX}]
print(c['name'])
\")
        TEACHER_LAYERS=\$(python3 -c \"
import json
with open('\${CONFIG_FILE}') as f:
    configs = json.load(f)
c = configs[${CONFIG_IDX}]
print(' '.join(c['teacher_layers']))
\")
        DEPTH_OVERRIDE=\$(python3 -c \"
import json
with open('\${CONFIG_FILE}') as f:
    configs = json.load(f)
c = configs[${CONFIG_IDX}]
print(c.get('depth', '${DEPTH}'))
\")

        OUT_DIR='${BASE_OUT}/ablation_'\${CONFIG_NAME}
        mkdir -p \"\${OUT_DIR}\"

        echo \"Config: \${CONFIG_NAME}\"
        echo \"Teacher layers: \${TEACHER_LAYERS}\"
        echo \"Depth: \${DEPTH_OVERRIDE}\"
        echo \"Output: \${OUT_DIR}\"

        # Resume from latest checkpoint if exists
        RESUME_ARG=''
        LATEST_CKPT=\$(find \"\${OUT_DIR}\" -name 'checkpoint_epoch_*.pt' 2>/dev/null | sort -V | tail -1)
        MID_CKPT=\"\${OUT_DIR}/checkpoint_mid_epoch.pt\"
        if [ -f \"\${MID_CKPT}\" ]; then
            echo \"Resuming from mid-epoch checkpoint\"
            RESUME_ARG=\"--resume \${MID_CKPT}\"
        elif [ -n \"\${LATEST_CKPT}\" ]; then
            echo \"Resuming from: \${LATEST_CKPT}\"
            RESUME_ARG=\"--resume \${LATEST_CKPT}\"
        fi

        echo ''
        echo '=== Starting training ==='

        python -u -m training.hybrid_distillation_train \
            --predictions '${PRED_DIR}' \
            --image-root '${IMAGE_ROOT}' \
            --epochs ${EPOCHS} \
            --batch-size ${BATCH} \
            --lr ${LR} \
            --attention-weight 1.0 \
            --mimicry-weight 0.5 \
            --relation-weight 0.5 \
            --seg-weight 1.0 \
            --base-channels ${BASE_CH} \
            --depth \${DEPTH_OVERRIDE} \
            --img-size ${IMG_SIZE} \
            --num-workers ${WORKERS} \
            --output-dir \"\${OUT_DIR}\" \
            --save-interval 1 \
            --val-split 0.1 \
            --patience 5 \
            --augment \
            --teacher-layers \${TEACHER_LAYERS} \
            \${RESUME_ARG}

        echo ''
        echo \"=== Config \${CONFIG_NAME} complete ===\"
        ls -lh \"\${OUT_DIR}/\"
    "
