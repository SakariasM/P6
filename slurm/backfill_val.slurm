#!/bin/bash
#SBATCH --job-name=kd-backfill
#SBATCH --output=/ceph/project/P6-Machine-Vision/P6/logs/backfill_%j.out
#SBATCH --error=/ceph/project/P6-Machine-Vision/P6/logs/backfill_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

CONTAINER="/ceph/container/pytorch/pytorch_25.12.sif"
VENV_DIR="$HOME/kd_venv"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"

singularity exec --nv \
    -B "${VENV_DIR}:/scratch/venv" \
    -B "${PROJECT_DIR}:/scratch/project" \
    -B "/ceph/project/P6-Machine-Vision/P6:/ceph/project/P6-Machine-Vision/P6" \
    "${CONTAINER}" \
    /bin/bash -c "
        source /scratch/venv/bin/activate
        cd /scratch/project
        export PYTHONPATH=/scratch/project/src

        python -u -m backfill_val_metrics \
            --predictions /ceph/project/P6-Machine-Vision/P6/results/hybrid_predictions \
            --image-root /ceph/project/P6-Machine-Vision/P6/data/open-images-v7/train/data \
            --checkpoint-dir /ceph/project/P6-Machine-Vision/P6/trained_models \
            --output /ceph/project/P6-Machine-Vision/P6/trained_models/training_history.json
    "
