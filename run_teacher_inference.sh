#!/bin/bash
#SBATCH --job-name=yolo_teacher_inference
#SBATCH --gres=gpu:4               # Request 4 GPUs
#SBATCH --mem=24G                  # 24GB RAM
#SBATCH --cpus-per-task=8          # 8 CPU cores for data loading
#SBATCH --time=02:00:00            # Max 2 hours (1hr download + 1hr inference)
#SBATCH --output=logs/run_teacher_inference_%j.out
#SBATCH --error=logs/run_teacher_inference_%j.err

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
CONTAINER="/ceph/container/pytorch/pytorch_25.12.sif"

echo "=== Using Container ==="
echo "Container: $CONTAINER"
# Don't try to ls the container - just verify it works when we use it

# Print GPU info for debugging
echo ""
echo "=== GPU Information ==="
nvidia-smi

# Set directories - using current working directory from job
PROJECT_DIR="$(pwd)"
DATA_DIR="$PROJECT_DIR/data"
RESULTS_DIR="$PROJECT_DIR/results"

echo ""
echo "=== Directory Setup ==="
echo "Project: $PROJECT_DIR"
echo "Data: $DATA_DIR"
echo "Results: $RESULTS_DIR"

# Create data directory
mkdir -p "$DATA_DIR" "$RESULTS_DIR"

# Step 1: Install additional dependencies inside container (if needed)
echo ""
echo "=== Step 1: Checking Dependencies ==="
singularity exec --nv "$CONTAINER" bash -c "
    # Check if ultralytics is already installed
    if python -c 'import ultralytics' 2>/dev/null; then
        echo '✓ ultralytics already installed'
    else
        echo 'Installing ultralytics...'
        pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true
        pip install --user --no-cache-dir opencv-python-headless
        pip install --user --no-cache-dir ultralytics tqdm
    fi

    echo ''
    echo 'Installed packages:'
    pip list | grep -E 'ultralytics|opencv|torch' || echo 'Package list unavailable'
"

# Step 2: Download dataset (images only, no annotations)
echo ""
echo "=== Step 2: Checking Dataset ==="

# Check if dataset already exists
DATASET_PATH="$DATA_DIR/val2017"
if [ -d "$DATASET_PATH" ]; then
    NUM_IMAGES=$(find "$DATASET_PATH" -name "*.jpg" 2>/dev/null | wc -l)
    echo "✓ Dataset already exists: $DATASET_PATH"
    echo "  Found $NUM_IMAGES images"

    if [ "$NUM_IMAGES" -lt 100 ]; then
        echo "  WARNING: Expected ~5000 images, but only found $NUM_IMAGES"
        echo "  Re-downloading dataset..."
        rm -rf "$DATASET_PATH"
    else
        echo "  Skipping download"
        SKIP_DOWNLOAD=true
    fi
fi

if [ "$SKIP_DOWNLOAD" != "true" ]; then
    echo "Downloading COCO val2017 dataset..."
    singularity exec --nv \
        --bind "$PROJECT_DIR:$PROJECT_DIR" \
        "$CONTAINER" \
        python "$PROJECT_DIR/src/download_dataset.py" \
            --dataset coco \
            --split val2017 \
            --output "$DATA_DIR"
fi

# Step 3: Run teacher model inference
echo ""
echo "=== Step 3: Running Teacher Model Inference ==="
singularity exec --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    "$CONTAINER" \
    python "$PROJECT_DIR/src/predictions.py" \
        --model yolo26n-seg.pt \
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
