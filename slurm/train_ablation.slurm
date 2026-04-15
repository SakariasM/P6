#!/bin/bash
#SBATCH --job-name=kd-ablation
#SBATCH --output=/ceph/project/P6-Machine-Vision/P6/logs/ablation_%j.out
#SBATCH --error=/ceph/project/P6-Machine-Vision/P6/logs/ablation_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Train a student model with a specific teacher layer configuration.
# Each config trains from scratch and saves to its own directory:
#   trained_models/ablation/<config_name>/
#
# ---------------------------------------------------------------
# USAGE — pick your config name and submit:
#
#   CONFIG=shallow_only   sbatch slurm/train_ablation.slurm
#   CONFIG=backbone_3     sbatch slurm/train_ablation.slurm
#   CONFIG=backbone_plus_neck2  sbatch slurm/train_ablation.slurm
#
# Available configs (see configs/ablation_configs.json):
#
#   SINGLES:
#     shallow_only        model.4                (P3, 80x80)
#     mid_only            model.6                (P4, 40x40)
#     deep_only           model.9                (P5, 20x20)
#     neck1_only          model.12               (neck, 40x40)
#     neck2_only          model.15               (neck, 80x80)
#
#   PAIRS:
#     shallow_deep        model.4 + model.9      (P3+P5)
#     shallow_mid         model.4 + model.6      (P3+P4)
#     mid_deep            model.6 + model.9      (P4+P5)
#     deep_neck1          model.9 + model.12     (P5+neck)
#     shallow_neck2       model.4 + model.15     (extreme spread)
#
#   TRIPLES:
#     backbone_3          model.4+6+9            (full backbone, default)
#     deep_plus_neck      model.9+12+15          (deep+neck)
#     spread_3            model.4+9+15           (spread)
#
#   LARGER (4 layers):
#     backbone_plus_neck1 model.4+6+9+12         (backbone + neck1)
#     backbone_plus_neck2 model.4+6+9+15         (backbone + neck2)
#
# Supports auto-resume: re-submitting the same CONFIG will pick up
# from the latest checkpoint.
# ---------------------------------------------------------------

set -euo pipefail

# CONFIG must be set
if [ -z "${CONFIG:-}" ]; then
    echo "ERROR: CONFIG not set."
    echo "Usage: CONFIG=shallow_only sbatch slurm/train_ablation.slurm"
    echo ""
    echo "Available configs:"
    python3 -c "
import json
with open('configs/ablation_configs.json') as f:
    configs = json.load(f)
for name, cfg in configs.items():
    layers = ', '.join(cfg['teacher_layers'])
    print(f'  {name:<22} {layers}')
" 2>/dev/null || echo "  (run from project root to see configs)"
    exit 1
fi

CONTAINER="/ceph/container/pytorch/pytorch_25.12.sif"
VENV_DIR="$HOME/kd_venv"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"

PRED_DIR="${PRED_DIR:-/ceph/project/P6-Machine-Vision/P6/results/ablation_predictions}"
IMAGE_ROOT="${DATA_DIR:-/ceph/project/P6-Machine-Vision/P6/data/open-images-v7/train/data}"
BASE_OUT="/ceph/project/P6-Machine-Vision/P6/trained_models/ablation"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH_SIZE:-8}"
LR="${LEARNING_RATE:-1e-4}"
BASE_CH="${BASE_CHANNELS:-8}"
DEFAULT_DEPTH="${DEPTH:-4}"
IMG_SIZE="${IMG_SIZE:-640}"
WORKERS="${NUM_WORKERS:-8}"

echo "=== Ablation Training: ${CONFIG} ==="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
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
        export TMPDIR=/ceph/project/P6-Machine-Vision/P6/tmp/ablation_${CONFIG}_\${SLURM_JOB_ID}
        mkdir -p \"\${TMPDIR}\"

        nvidia-smi

        # Read config
        CONFIG_FILE='configs/ablation_configs.json'
        PARSED=\$(python3 -c \"
import json, sys
with open('\${CONFIG_FILE}') as f:
    configs = json.load(f)
if '${CONFIG}' not in configs:
    print('ERROR: Unknown config: ${CONFIG}', file=sys.stderr)
    print('Available:', ', '.join(configs.keys()), file=sys.stderr)
    sys.exit(1)
c = configs['${CONFIG}']
print(' '.join(c['teacher_layers']))
print(c.get('depth', '${DEFAULT_DEPTH}'))
print(c.get('description', ''))
\")

        TEACHER_LAYERS=\$(echo \"\${PARSED}\" | sed -n '1p')
        DEPTH_OVERRIDE=\$(echo \"\${PARSED}\" | sed -n '2p')
        DESCRIPTION=\$(echo \"\${PARSED}\" | sed -n '3p')

        OUT_DIR='${BASE_OUT}/${CONFIG}'
        mkdir -p \"\${OUT_DIR}\"

        echo \"Config:  ${CONFIG}\"
        echo \"Desc:    \${DESCRIPTION}\"
        echo \"Layers:  \${TEACHER_LAYERS}\"
        echo \"Depth:   \${DEPTH_OVERRIDE}\"
        echo \"Output:  \${OUT_DIR}\"
        echo \"Epochs:  ${EPOCHS}\"
        echo \"Batch:   ${BATCH}\"
        echo \"LR:      ${LR}\"
        echo \"\"

        # Resume from latest checkpoint if exists
        RESUME_ARG=''
        MID_CKPT=\"\${OUT_DIR}/checkpoint_mid_epoch.pt\"
        LATEST_CKPT=\$(find \"\${OUT_DIR}\" -name 'checkpoint_epoch_*.pt' 2>/dev/null | sort -V | tail -1)
        if [ -f \"\${MID_CKPT}\" ]; then
            echo \"Resuming from mid-epoch checkpoint\"
            RESUME_ARG=\"--resume \${MID_CKPT}\"
        elif [ -n \"\${LATEST_CKPT}\" ]; then
            echo \"Resuming from: \${LATEST_CKPT}\"
            RESUME_ARG=\"--resume \${LATEST_CKPT}\"
        else
            echo \"Starting fresh\"
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
            --mimicry-weight 2.0 \
            --relation-weight 1.0 \
            --seg-weight 0.5 \
            --grad-clip 1.0 \
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
        echo '=== ${CONFIG} training complete ==='
        ls -lh \"\${OUT_DIR}/\"
    "
