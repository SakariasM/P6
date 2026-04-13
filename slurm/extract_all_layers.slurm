#!/bin/bash
#SBATCH --job-name=kd-extract-all
#SBATCH --output=/ceph/project/P6-Machine-Vision/P6/logs/extract_all_%A_%a.out
#SBATCH --error=/ceph/project/P6-Machine-Vision/P6/logs/extract_all_%A_%a.err
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# Extract teacher features from ALL 5 candidate YOLO backbone+neck layers.
# Produces chunks with model.4, model.6, model.9, model.12, model.15 features.
# These chunks are shared by all ablation training configs.
#
# Usage: sbatch slurm/extract_all_layers.slurm

set -euo pipefail

CONTAINER="/ceph/container/pytorch/pytorch_25.12.sif"
VENV_DIR="$HOME/kd_venv"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
NUM_WORKERS=4
WORKER_ID=${SLURM_ARRAY_TASK_ID}

MODEL="${TEACHER_MODEL:-/scratch/project/yolo26n-seg.pt}"
INPUT="${DATA_DIR:-/ceph/project/P6-Machine-Vision/P6/data/open-images-v7/train/data}"
OUTPUT="${OUTPUT_DIR:-/ceph/project/P6-Machine-Vision/P6/results/ablation_predictions}"
BATCH="${BATCH_SIZE:-8}"

echo "=== All-Layer Teacher Extraction (worker ${WORKER_ID}/${NUM_WORKERS}) ==="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}, Array task: ${WORKER_ID}"
echo "Model: ${MODEL}"
echo "Output: ${OUTPUT}"
echo ""

mkdir -p "${OUTPUT}"

singularity exec --nv \
    -B "${VENV_DIR}:/scratch/venv" \
    -B "${PROJECT_DIR}:/scratch/project" \
    -B "/ceph/project/P6-Machine-Vision/P6:/ceph/project/P6-Machine-Vision/P6" \
    "${CONTAINER}" \
    /bin/bash -c "
        source /scratch/venv/bin/activate
        cd /scratch/project
        export PYTHONPATH=/scratch/project/src
        export TMPDIR=/ceph/project/P6-Machine-Vision/P6/tmp/extract_all_\${SLURM_ARRAY_TASK_ID}
        mkdir -p \"\${TMPDIR}\"

        nvidia-smi

        echo ''
        echo '=== Extracting ALL layers (worker ${WORKER_ID}) ==='

        python -u -m teacher.hybrid_predictions \
            --model '${MODEL}' \
            --input '${INPUT}' \
            --output '${OUTPUT}' \
            --batch-size ${BATCH} \
            --person-only \
            --checkpoint-interval 500 \
            --feature-layers model.4 model.6 model.9 model.12 model.15 \
            --worker-id ${WORKER_ID} \
            --num-workers ${NUM_WORKERS}

        echo ''
        echo '=== Worker ${WORKER_ID} complete ==='
        ls -lh '${OUTPUT}/'
    "
