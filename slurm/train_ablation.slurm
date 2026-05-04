#!/bin/bash
#SBATCH --job-name=kd-ablation
#SBATCH --output=/ceph/project/P6-Machine-Vision/P6/logs/ablation_%j.out
#SBATCH --error=/ceph/project/P6-Machine-Vision/P6/logs/ablation_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Fine-tune a student model with CBAM disabled at encoder 0 (no_cbam_enc0
# variant) starting from the baseline checkpoint for the given teacher-layer
# CONFIG. Output goes to its own directory so the original baseline is
# preserved:
#   trained_models/ablation/<config_name>/                 <- baseline (input)
#   trained_models/ablation/<config_name>_no_cbam_enc0/    <- fine-tune (output)
#
# Defaults are tuned for fine-tuning, not from-scratch training:
#   EPOCHS=5, LEARNING_RATE=2e-5
#
# Override the baseline checkpoint path via LOAD_WEIGHTS=... if the default
# (${BASE_OUT}/${CONFIG}/best_model.pt) is not where you keep it.
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
#   SINGLES (backbone):
#     shallow_only        model.4                (C3k2, P3, 80x80)
#     mid_only            model.6                (C3k2, P4, 40x40)
#     deep_only           model.9                (SPPF, P5, 20x20)
#
#   SINGLES (neck C3k2):
#     neck1_c3k2_only     model.13               (C3k2 after concat, 40x40)
#     neck2_c3k2_only     model.16               (C3k2 after concat, 80x80)
#
#   PAIRS:
#     shallow_deep        model.4 + model.9      (P3+P5)
#     mid_deep            model.6 + model.9      (P4+P5)
#     deep_neck1          model.9 + model.12     (P5+neck concat)
#
#   TRIPLES:
#     backbone_3          model.4+6+9            (full backbone, default)
#     deep_plus_neck_c3k2 model.9+13+16          (deep+neck C3k2)
#
#   LARGER (4 layers):
#     backbone_plus_neck1_c3k2 model.4+6+9+13    (backbone + neck1 C3k2)
#     backbone_plus_neck2_c3k2 model.4+6+9+16    (backbone + neck2 C3k2)
#
#   DEPTH=5 (5 layers):
#     all_5_c3k2          model.4+6+9+13+16      (all C3k2, ~5.7M params)
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
EPOCHS="${EPOCHS:-5}"
BATCH="${BATCH_SIZE:-8}"
LR="${LEARNING_RATE:-2e-5}"
BASE_CH="${BASE_CHANNELS:-8}"
DEFAULT_DEPTH="${DEPTH:-4}"
IMG_SIZE="${IMG_SIZE:-640}"
WORKERS="${NUM_WORKERS:-8}"
# Optional override: path to baseline checkpoint to fine-tune from.
# Default resolves to ${BASE_OUT}/${CONFIG}/best_model.pt below.
LOAD_WEIGHTS_OVERRIDE="${LOAD_WEIGHTS:-}"

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
        export TMPDIR=/tmp/abl_\${SLURM_JOB_ID}
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

        BASELINE_DIR='${BASE_OUT}/${CONFIG}'
        OUT_DIR='${BASE_OUT}/${CONFIG}_no_cbam_enc0'
        mkdir -p \"\${OUT_DIR}\"

        # Resolve baseline checkpoint path for the initial fine-tune load.
        BASELINE_CKPT='${LOAD_WEIGHTS_OVERRIDE}'
        if [ -z \"\${BASELINE_CKPT}\" ]; then
            BASELINE_CKPT=\"\${BASELINE_DIR}/best_model.pt\"
        fi

        echo \"Config:    ${CONFIG}\"
        echo \"Desc:      \${DESCRIPTION}\"
        echo \"Layers:    \${TEACHER_LAYERS}\"
        echo \"Depth:     \${DEPTH_OVERRIDE}\"
        echo \"Mode:      no_cbam_enc0 fine-tune (CBAM disabled at encoder 0)\"
        echo \"Baseline:  \${BASELINE_CKPT}\"
        echo \"Output:    \${OUT_DIR}\"
        echo \"Epochs:    ${EPOCHS}\"
        echo \"Batch:     ${BATCH}\"
        echo \"LR:        ${LR}\"
        echo \"\"

        # Pick init source: resume from output dir if a fine-tune is already
        # in progress; otherwise load weights from the baseline checkpoint.
        # --resume and --load-weights are mutually exclusive in the trainer.
        INIT_ARG=''
        MID_CKPT=\"\${OUT_DIR}/checkpoint_mid_epoch.pt\"
        LATEST_CKPT=\$(find \"\${OUT_DIR}\" -name 'checkpoint_epoch_*.pt' 2>/dev/null | sort -V | tail -1)
        if [ -f \"\${MID_CKPT}\" ]; then
            echo \"Resuming fine-tune from mid-epoch checkpoint\"
            INIT_ARG=\"--resume \${MID_CKPT}\"
        elif [ -n \"\${LATEST_CKPT}\" ]; then
            echo \"Resuming fine-tune from: \${LATEST_CKPT}\"
            INIT_ARG=\"--resume \${LATEST_CKPT}\"
        elif [ -f \"\${BASELINE_CKPT}\" ]; then
            echo \"Initializing fine-tune from baseline weights (strict=False)\"
            INIT_ARG=\"--load-weights \${BASELINE_CKPT}\"
        else
            echo \"ERROR: no baseline checkpoint at \${BASELINE_CKPT} and no fine-tune in progress.\" >&2
            echo \"       Train the baseline first or pass LOAD_WEIGHTS=...\" >&2
            exit 1
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
            --cbam-levels 1 2 3 \
            \${INIT_ARG}

        echo ''
        echo '=== ${CONFIG} training complete ==='
        ls -lh \"\${OUT_DIR}/\"
    "
