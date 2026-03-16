#!/bin/bash
#SBATCH --job-name=yolo_teacher_inference
#SBATCH --partition=batch          # or 'gpu' depending on cluster config
#SBATCH --gres=gpu:4               # Request 8 GPUs
#SBATCH --mem=24G                  # 32GB RAM (adjust as needed)
#SBATCH --cpus-per-task=15         # 8 CPU cores for data loading
#SBATCH --time=01:00:00            # Max 1 hour (adjust based on dataset size)
#SBATCH --output=logs/teacher_inference_%j.out
#SBATCH --error=logs/teacher_inference_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules (adjust based on AAU's module system)
# module load python/3.10
# module load cuda/12.1



# Print GPU info for debugging
nvidia-smi

# Run teacher inference
python src/predictions.py \
    --model yolo11n-seg.pt \
    --input /path/to/your/images \
    --output results \
    --format pickle \
    --batch-size 32 \
    --person-only \
    --checkpoint-interval 1000

echo "Teacher inference completed!"
