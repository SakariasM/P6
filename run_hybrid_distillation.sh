#!/bin/bash

# Hybrid Knowledge Distillation Pipeline
# Extracts both logits and intermediate features for student training

set -e  # Exit on error

# Configuration
MODEL="yolo26n.pt"
DATA_DIR="./data/images"
OUTPUT_DIR="./hybrid_predictions"
TRAINED_DIR="./trained_models"
BATCH_SIZE=8
EPOCHS=50

echo "=========================================="
echo "Hybrid Knowledge Distillation Pipeline"
echo "=========================================="
echo ""

# Step 1: Extract teacher predictions + features
echo "[1/2] Extracting teacher predictions and features..."
echo "This will capture both logits and intermediate feature maps"
echo ""

python src/teacher/hybrid_predictions.py \
    --model ${MODEL} \
    --input ${DATA_DIR} \
    --output ${OUTPUT_DIR} \
    --batch-size ${BATCH_SIZE} \
    --person-only \
    --checkpoint-interval 50

echo ""
echo "[1/2] Complete! Teacher knowledge extracted."
echo ""

# Step 2: Train student with hybrid distillation
echo "[2/2] Training student model with hybrid distillation..."
echo "Using both feature-based and response-based losses"
echo ""

python src/training/hybrid_distillation_train.py \
    --predictions ${OUTPUT_DIR}/hybrid_teacher_predictions.pt \
    --image-root ${DATA_DIR} \
    --model-type standard \
    --epochs ${EPOCHS} \
    --batch-size 16 \
    --lr 1e-3 \
    --feature-weight 1.0 \
    --response-weight 1.0 \
    --temperature 3.0 \
    --feature-distance mse \
    --output-dir ${TRAINED_DIR}

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Teacher predictions: ${OUTPUT_DIR}/hybrid_teacher_predictions.pt"
echo "Trained model: ${TRAINED_DIR}/best_model.pt"
echo ""
