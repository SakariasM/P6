#!/bin/bash
#SBATCH --job-name=yolo_teacher_inference
#SBATCH --gres=gpu:4               # Request 4 GPUs
#SBATCH --mem=24G                  # 24GB RAM
#SBATCH --cpus-per-task=8          # 8 CPU cores for data loading
#SBATCH --time=02:00:00            # Max 2 hours (1hr download + 1hr inference)
#SBATCH --output=logs/teacher_inference_%j.out
#SBATCH --error=logs/teacher_inference_%j.err

# Create directories
mkdir -p logs
mkdir -p results

# Load modules (adjust based on AAU's module system)
# module load python/3.10
# module load cuda/12.1

# Activate virtual environment if using one
# source venv/bin/activate

# Print GPU info for debugging
echo "=== GPU Information ==="
nvidia-smi

# Set data directory (adjust to your cluster home)
DATA_DIR="/ceph/home/aau/$USER/P6/data"
RESULTS_DIR="/ceph/home/aau/$USER/P6/results"

# Step 1: Download dataset (images only, no annotations)
echo ""
echo "=== Step 1: Downloading Dataset ==="
python src/download_dataset.py \
    --dataset coco \
    --split val2017 \
    --output "$DATA_DIR"

# Step 2: Run teacher model inference
echo ""
echo "=== Step 2: Running Teacher Model Inference ==="
python src/predictions.py \
    --model yolo11n-seg.pt \
    --input "$DATA_DIR/val2017" \
    --output "$RESULTS_DIR" \
    --format pickle \
    --batch-size 32 \
    --person-only \
    --checkpoint-interval 500

echo ""
echo "=== Job Complete ==="
echo "Results saved to: $RESULTS_DIR"
