#!/bin/bash
#SBATCH --job-name=kd-full
#SBATCH --output=/ceph/project/P6-Machine-Vision/P6/logs/full_%j.out
#SBATCH --error=/ceph/project/P6-Machine-Vision/P6/logs/full_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00

# Full pipeline: tests -> extract -> train (single job).
# Usage: sbatch slurm/full_pipeline.slurm

set -euo pipefail

CONTAINER="/ceph/container/pytorch/pytorch_25.12.sif"
VENV_DIR="$HOME/kd_venv"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"

MODEL="${TEACHER_MODEL:-/scratch/project/yolo26n-seg.pt}"
DATA="${DATA_DIR:-/ceph/project/P6-Machine-Vision/P6/data/open-images-v7}"
PRED_DIR="./results/hybrid_predictions"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH_SIZE:-16}"

echo "=== Full Hybrid Distillation Pipeline ==="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo ""

mkdir -p logs results/hybrid_predictions trained_models

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

        # ---- Step 1: Tests ----
        echo ''
        echo '===== STEP 1/3: Running tests ====='
        python -m pytest tests/ -v --tb=short
        echo 'Tests passed.'

        # ---- Step 2: Extract teacher knowledge ----
        echo ''
        echo '===== STEP 2/3: Teacher extraction ====='
        python -m teacher.hybrid_predictions \
            --model '${MODEL}' \
            --input '${DATA}' \
            --output '${PRED_DIR}' \
            --batch-size 8 \
            --person-only \
            --checkpoint-interval 50

        echo 'Extraction complete.'
        ls -lh '${PRED_DIR}/'

        # ---- Step 3: Train student ----
        echo ''
        echo '===== STEP 3/3: Student training ====='

        PRED_FILE=\$(find '${PRED_DIR}' -name 'hybrid_teacher_predictions.*' | head -1)
        if [ -z \"\${PRED_FILE}\" ]; then
            echo 'ERROR: No predictions file found'
            exit 1
        fi

        python -m training.hybrid_distillation_train \
            --predictions \"\${PRED_FILE}\" \
            --image-root '${DATA}' \
            --epochs ${EPOCHS} \
            --batch-size ${BATCH} \
            --lr 1e-3 \
            --feature-weight 1.0 \
            --response-weight 1.0 \
            --temperature 3.0 \
            --feature-distance mse \
            --output-dir ./trained_models \
            --save-interval 10

        echo ''
        echo '===== Pipeline complete ====='
        ls -lh ./trained_models/
    "
