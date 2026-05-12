#!/bin/bash
#SBATCH --job-name=coco_benchmark
#SBATCH --output=/ceph/project/P6-Machine-Vision/benchmark/logs/coco_benchmark_%A_%a.log
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-23

MODELS=(
    "student_seg_all_5_c3k2_640"
    "student_seg_all_5_c3k2_no_cbam_enc0_scratch_640"
    "student_seg_backbone_3_640"
    "student_seg_backbone_3_no_cbam_enc0_scratch_640"
    "student_seg_backbone_plus_neck1_c3k2_640"
    "student_seg_backbone_plus_neck1_c3k2_no_cbam_enc0_scratch_640"
    "student_seg_backbone_plus_neck2_c3k2_640"
    "student_seg_backbone_plus_neck2_c3k2_no_cbam_enc0_scratch_640"
    "student_seg_deep_only_640"
    "student_seg_deep_only_no_cbam_enc0_scratch_640"
    "student_seg_deep_plus_neck_c3k2_640"
    "student_seg_deep_plus_neck_c3k2_no_cbam_enc0_scratch_640"
    "student_seg_mid_deep_640"
    "student_seg_mid_deep_no_cbam_enc0_scratch_640"
    "student_seg_mid_only_640"
    "student_seg_mid_only_no_cbam_enc0_scratch_640"
    "student_seg_neck1_c3k2_only_640"
    "student_seg_neck1_c3k2_only_no_cbam_enc0_scratch_640"
    "student_seg_neck2_c3k2_only_640"
    "student_seg_neck2_c3k2_only_no_cbam_enc0_scratch_640"
    "student_seg_shallow_deep_640"
    "student_seg_shallow_deep_no_cbam_enc0_scratch_640"
    "student_seg_shallow_only_640"
    "student_seg_shallow_only_no_cbam_enc0_scratch_640"
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
ROOT=/ceph/project/P6-Machine-Vision/benchmark

echo "Running COCO benchmark for: $MODEL"
echo "Job ID: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "Start: $(date)"

$ROOT/venv/bin/python $ROOT/tools/coco_benchmark.py \
    --model /ceph/project/P6-Machine-Vision/P6/models/${MODEL}.tflite \
    --coco-dir $ROOT/datasets/coco_data

echo "Done: $(date)"
