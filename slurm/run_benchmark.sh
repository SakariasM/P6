#!/bin/bash
#SBATCH --job-name=yolo-benchmark
#SBATCH --output=/ceph/project/P6-Machine-Vision/P6/logs/benchmark_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=l4

PYTHON=/ceph/project/P6-Machine-Vision/P6/.venv/bin/python
PROJECT=/ceph/project/P6-Machine-Vision/P6

mkdir -p $PROJECT/logs

$PYTHON $PROJECT/src/benchmark_accuracy.py \
    --models  $PROJECT/trained_models/best_model.pt $PROJECT/trained_models/best_model_deploy.pt \
    --types   teacher student \
    --data    $PROJECT/data/data.yaml \
    --imgsz   640 --batch 16 --device cuda --split val \
    --out     $PROJECT/results/benchmark_results.json
