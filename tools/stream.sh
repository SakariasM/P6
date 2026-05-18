#!/bin/bash
# Usage:
#   ./stream.sh video.mp4                          — auto-detect FPS from the video
#   ./stream.sh video.mp4 --fps 30                 — override FPS
#   ./stream.sh video.mp4 --debug                  — also save debug overlay video to PC
#   ./stream.sh video.mp4 --model path/to/model.tflite  — upload model to Pi and switch to it

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FFMPEG="${FFMPEG:-ffmpeg}"
FFPROBE="${FFPROBE:-ffprobe}"
PI="sw6@192.168.10.3"
PI_PROJECT="~/Project/Prototype/P6"
RUN_DIR="$ROOT_DIR/runs/stream/$(date +%Y%m%d-%H%M%S)"

VIDEO=""
FPS=""
DEBUG=""
MODEL_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --fps)   FPS="$2"; shift 2 ;;
        --debug) DEBUG="1"; shift ;;
        --model) MODEL_PATH="$2"; shift 2 ;;
        *)       VIDEO="$1"; shift ;;
    esac
done

if [[ -z "$VIDEO" ]]; then
    echo "Usage: $0 <video> [--fps N] [--debug] [--model path/to/model.tflite]"
    exit 1
fi

mkdir -p "$RUN_DIR"

if [[ ! -f "$VIDEO" ]]; then
    echo "Error: file not found: $VIDEO"
    exit 1
fi

if [[ -z "$FPS" ]]; then
    RAW=$("$FFPROBE" -v error -select_streams v:0 \
        -show_entries stream=r_frame_rate -of csv=p=0 "$VIDEO")
    FPS=$(echo "$RAW" | awk -F'/' '{ if($2) printf "%.0f", $1/$2; else print $1 }')
    if [[ -z "$FPS" ]]; then
        echo "Error: could not detect FPS from video. Use --fps N to set it manually."
        exit 1
    fi
    echo "[stream] Auto-detected FPS: $FPS"
else
    echo "[stream] FPS override: $FPS"
fi

DEBUG_ARG=""
[[ -n "$DEBUG" ]] && DEBUG_ARG="--debug"

# -- model upload --
if [[ -n "$MODEL_PATH" ]]; then
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "Error: model not found: $MODEL_PATH"
        exit 1
    fi
    MODEL_FILENAME=$(basename "$MODEL_PATH")
    MODEL_NAME="${MODEL_FILENAME%.tflite}"
    MODEL_NAME="${MODEL_NAME%_float32}"
    PI_MODEL_DIR="$PI_PROJECT/models/$MODEL_NAME"

    echo "[stream] Uploading model: $MODEL_FILENAME -> Pi:$PI_MODEL_DIR/"
    ssh "$PI" "mkdir -p $PI_MODEL_DIR"
    scp "$MODEL_PATH" "$PI:$PI_MODEL_DIR/${MODEL_NAME}_float32.tflite"

    echo "[stream] Switching Pi to model: $MODEL_NAME"
    ssh "$PI" "sed -i 's/^MODEL          = .*/MODEL          = \"$MODEL_NAME\"/' $PI_PROJECT/live_mask.py"
    echo "[stream] Model set."
fi

echo "[stream] Killing any leftover processes on Pi..."
ssh "$PI" "pkill -f live_mask.py; pkill -f run_benchmark.sh; sleep 1"

MY_IP=$(ip -4 addr show | grep -oP '192\.168\.10\.\d+' | head -1)
DEST_DIR="$USER@$MY_IP:$ROOT_DIR/data/preds"

echo "[stream] Starting benchmark on Pi (--fps $FPS${DEBUG:+ --debug})..."
ssh "$PI" "nohup bash -c 'cd ~/Project/Prototype/P6 && ./run_benchmark.sh --fps $FPS --dest \"$DEST_DIR\" $DEBUG_ARG > /tmp/benchmark_out.txt 2>&1' > /dev/null 2>&1 < /dev/null &"

echo "[stream] Waiting 5s for Pi model to load..."
sleep 5

echo "[stream] Starting ffmpeg stream..."
"$FFMPEG" -re -i "$VIDEO" -f mpegts udp://192.168.10.3:1234 &
FFMPEG_PID=$!

echo "[stream] Waiting for ffmpeg to finish..."
wait $FFMPEG_PID

echo "[stream] Stream done — waiting for Pi to finish processing..."
sleep 15

echo "[stream] Fetching Pi log..."
scp "$PI:/tmp/benchmark_out.txt" "$RUN_DIR/pi_run.log" 2>/dev/null || true

echo "[stream] Fetching pred mask from Pi..."
PRED_DIR="$ROOT_DIR/data/preds"
mkdir -p "$PRED_DIR"
PRED_FILES=$(ssh "$PI" "ls /tmp/pred_mask_tmp.mp4 /tmp/pred_mask_tmp_timestamps.csv /tmp/stream_start_time.txt 2>/dev/null")
if [[ -n "$PRED_FILES" ]]; then
    MODEL_NAME=$(ssh "$PI" "grep '^MODEL ' ~/Project/Prototype/P6/live_mask.py | head -1 | sed 's/.*= *\"\(.*\)\".*/\1/'")
    RES=$(ssh "$PI" "ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 /tmp/pred_mask_tmp.mp4 2>/dev/null | tr ',' 'x'")
    [[ -z "$RES" ]] && RES="unknown"
    FILENAME="pred_mask_${MODEL_NAME}_${RES}.mp4"
    scp "$PI:/tmp/pred_mask_tmp.mp4" "$PRED_DIR/$FILENAME"
    scp "$PI:/tmp/pred_mask_tmp_timestamps.csv" "$PRED_DIR/${FILENAME%.mp4}_timestamps.csv" 2>/dev/null
    scp "$PI:/tmp/stream_start_time.txt" "$PRED_DIR/${FILENAME%.mp4}_stream_start.txt" 2>/dev/null
    scp "$PI:~/Project/Prototype/P6/run_log.txt" "$PRED_DIR/${FILENAME%.mp4}_run_log.txt" 2>/dev/null
    echo "[stream] Pred mask saved: $PRED_DIR/$FILENAME"
else
    echo "[stream] No pred mask found on Pi"
fi

if [[ -n "$DEBUG" ]]; then
    echo "[stream] Fetching debug video from Pi..."
    scp "$PI:/tmp/debug_output.mp4" "$RUN_DIR/debug_output.mp4" 2>/dev/null && \
        echo "[stream] Debug video saved to: $RUN_DIR/debug_output.mp4" || \
        echo "[stream] No debug video found on Pi"
fi

echo "[stream] Done"
