#!/bin/bash
#SBATCH --job-name=coco-benchmark
#SBATCH --output=/ceph/project/P6-Machine-Vision/P6/logs/coco_benchmark_%j.log
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=l4

PYTHON=/ceph/project/P6-Machine-Vision/P6/.venv/bin/python
PROJECT=/ceph/project/P6-Machine-Vision/P6

mkdir -p $PROJECT/logs
export ULTRALYTICS_DIR=$PROJECT/tmp
export YOLO_DATASETS_DIR=$PROJECT/datasets

$PYTHON $PROJECT/coco_benchmark.py
