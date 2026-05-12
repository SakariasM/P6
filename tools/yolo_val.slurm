#!/bin/bash
#SBATCH --job-name=yolo_val
#SBATCH --output=/ceph/project/P6-Machine-Vision/benchmark/logs/yolo_val_%j.log
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

ROOT=/ceph/project/P6-Machine-Vision/benchmark

echo "Start: $(date)"

$ROOT/venv/bin/python - << 'EOF'
from ultralytics import YOLO
import os

os.environ["YOLO_DATASETS_DIR"] = "/ceph/project/P6-Machine-Vision/benchmark/datasets"

model = YOLO("/ceph/project/P6-Machine-Vision/benchmark/models/yolo26n-seg.pt")
metrics = model.val(
    data="coco.yaml",
    imgsz=640,
    split="val",
    task="segment",
    verbose=True,
    device='cpu',
)

print("\n========================================")
print(f"  mAP@50-95 (box)  : {metrics.box.map:.4f}")
print(f"  mAP@50    (box)  : {metrics.box.map50:.4f}")
print(f"  mAP@50-95 (mask) : {metrics.seg.map:.4f}")
print(f"  mAP@50    (mask) : {metrics.seg.map50:.4f}")
print("========================================\n")
EOF

echo "Done: $(date)"
