#!/bin/bash
# Runs the full practical benchmark for all 10 student models sequentially.
# Each model is streamed to the Pi, pred mask collected, and benchmark run automatically.
#
# Usage:
#   bash tools/run_all_benchmarks.sh
#   bash tools/run_all_benchmarks.sh --debug   # also fetch Pi debug video each run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DEBUG=""
[[ "$1" == "--debug" ]] && DEBUG="--debug"

GT="$ROOT_DIR/data/gt/gt_mask_room.mp4"
VIDEO="$ROOT_DIR/data/gt/raw_room.avi"

MODELS=(
    # CBAM (legacy)
    "models/legacy/student_seg_all_5_c3k2_320.tflite"
    "models/legacy/student_seg_backbone_3_320.tflite"
    "models/legacy/student_seg_backbone_plus_neck1_c3k2_320.tflite"
    "models/legacy/student_seg_backbone_plus_neck2_c3k2_320.tflite"
    "models/legacy/student_seg_deep_only_320.tflite"
    "models/legacy/student_seg_deep_plus_neck_c3k2_320.tflite"
    "models/legacy/student_seg_mid_deep_320.tflite"
    "models/legacy/student_seg_mid_only_320.tflite"
    "models/legacy/student_seg_neck1_c3k2_only_320.tflite"
    "models/legacy/student_seg_neck2_c3k2_only_320.tflite"
    "models/legacy/student_seg_shallow_deep_320.tflite"
    "models/legacy/student_seg_shallow_only_320.tflite"
    # No CBAM (scratch)
    "models/no_cbam/student_seg_all_5_c3k2_no_cbam_enc0_scratch_320.tflite"
    "models/no_cbam/student_seg_backbone_3_no_cbam_enc0_scratch_320.tflite"
    "models/no_cbam/student_seg_backbone_plus_neck1_c3k2_no_cbam_enc0_scratch_320.tflite"
    "models/no_cbam/student_seg_backbone_plus_neck2_c3k2_no_cbam_enc0_scratch_320.tflite"
    "models/no_cbam/student_seg_deep_only_no_cbam_enc0_scratch_320.tflite"
    "models/no_cbam/student_seg_deep_plus_neck_c3k2_no_cbam_enc0_scratch_320.tflite"
    "models/no_cbam/student_seg_mid_deep_no_cbam_enc0_scratch_320.tflite"
    "models/no_cbam/student_seg_mid_only_no_cbam_enc0_scratch_320.tflite"
    "models/no_cbam/student_seg_neck1_c3k2_only_no_cbam_enc0_scratch_320.tflite"
    "models/no_cbam/student_seg_neck2_c3k2_only_no_cbam_enc0_scratch_320.tflite"
    "models/no_cbam/student_seg_shallow_deep_no_cbam_enc0_scratch_320.tflite"
    "models/no_cbam/student_seg_shallow_only_no_cbam_enc0_scratch_320.tflite"
)

cd "$ROOT_DIR"

TOTAL=${#MODELS[@]}
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    NAME=$(basename "$MODEL" .tflite)
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$(( i + 1 ))/$TOTAL] $NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Stream to Pi and collect pred mask
    bash tools/stream.sh "$VIDEO" --model "$MODEL" $DEBUG
    if [[ $? -ne 0 ]]; then
        echo "[run_all] stream.sh failed for $NAME — skipping benchmark step"
        continue
    fi

    # Find the pred mask that was just written
    PRED=$(ls -t "$ROOT_DIR/data/preds/pred_mask_${NAME}_"*.mp4 2>/dev/null | head -1)
    if [[ -z "$PRED" ]]; then
        echo "[run_all] No pred mask found for $NAME — skipping benchmark step"
        continue
    fi

    # Run benchmark
    python3 tools/benchmark.py \
        --gt "$GT" \
        --pred "$PRED" \
        --pred-offset-auto

    echo "[run_all] Done: $NAME"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  All $TOTAL models done. Results in logs/benchmark_logs.txt
"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
