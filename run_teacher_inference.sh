#!/bin/bash
#SBATCH --job-name=yolo_teacher_inference
#SBATCH --gres=gpu:4               # Request 4 GPUs
#SBATCH --mem=24G                  # 24GB RAM
#SBATCH --cpus-per-task=8          # 8 CPU cores for data loading
#SBATCH --time=02:00:00            # Max 2 hours (1hr download + 1hr inference)
#SBATCH --output=/ceph/home/aau/%u/P6/logs/teacher_inference_%j.out
#SBATCH --error=/ceph/home/aau/%u/P6/logs/teacher_inference_%j.err

# Exit on any error
set -e

# Print all commands
set -x

echo "=== Job Started at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"

# Create directories
mkdir -p logs
mkdir -p results

# AAU AI Lab uses Singularity containers, not modules
# Set container path - check /ceph/container for available containers
CONTAINER="/ceph/container/pytorch-24.07-py3.sif"

# Alternative containers (if the above doesn't exist):
# CONTAINER="/ceph/container/pytorch_latest.sif"
# CONTAINER="/ceph/container/pytorch-gpu.sif"

echo "=== Container Info ==="
echo "Using container: $CONTAINER"
ls -lh "$CONTAINER"

# Print GPU info for debugging
echo ""
echo "=== GPU Information ==="
nvidia-smi

# Set data directory (adjust to your cluster home)
DATA_DIR="/ceph/home/aau/$USER/P6/data"
RESULTS_DIR="/ceph/home/aau/$USER/P6/results"
PROJECT_DIR="/ceph/home/aau/$USER/P6"

# Step 1: Install additional dependencies inside container (if needed)
echo ""
echo "=== Step 1: Installing Dependencies ==="
singularity exec --nv "$CONTAINER" \
    pip install --user ultralytics opencv-python-headless tqdm

# Step 2: Download dataset (images only, no annotations)
echo ""
echo "=== Step 2: Downloading Dataset ==="
singularity exec --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    "$CONTAINER" \
    python "$PROJECT_DIR/src/download_dataset.py" \
        --dataset coco \
        --split val2017 \
        --output "$DATA_DIR"

# Step 3: Run teacher model inference
echo ""
echo "=== Step 3: Running Teacher Model Inference ==="
singularity exec --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    "$CONTAINER" \
    python "$PROJECT_DIR/src/predictions.py" \
        --model yolo11n-seg.pt \
        --input "$DATA_DIR/val2017" \
        --output "$RESULTS_DIR" \
        --format pickle \
        --batch-size 32 \
        --person-only \
        --checkpoint-interval 500

echo ""
echo "=== Job Complete at $(date) ==="
echo "Results saved to: $RESULTS_DIR"
ls -lh "$RESULTS_DIR"
