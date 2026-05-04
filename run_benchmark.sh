#!/bin/bash
cd ~/Project/Prototype/P6

MODEL=$(grep '^MODEL ' live_mask.py | head -1 | sed 's/.*= *"\(.*\)".*/\1/')
TMP_MASK=/tmp/pred_mask_tmp.mp4
rm -f /tmp/debug_output.mp4

echo "[benchmark] Model: $MODEL"
echo "[benchmark] Starting live_mask.py... (ffmpeg stream on the PC must already be running)"

FPS_ARG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --fps)   FPS_ARG="--fps $2"; FPS_ARG_VAL="$2"; shift 2 ;;
        --debug) DEBUG_ARG="--output /tmp/debug_output.mp4"; shift ;;
        *) shift ;;
    esac
done

REC_START=$(date +%s%N)
.venv/bin/python3 live_mask.py --output-mask "$TMP_MASK" ${FPS_ARG} ${DEBUG_ARG} 2>&1 | tee run_log.txt
REC_END=$(date +%s%N)
if [ -f /tmp/mask_start_ns ]; then
    REC_START=$(cat /tmp/mask_start_ns)
    rm -f /tmp/mask_start_ns
fi
REC_SECS=$(echo "$REC_START $REC_END" | awk '{printf "%.3f", ($2-$1)/1e9}')

echo "[benchmark] Stream ended. Getting resolution..."
echo "[benchmark] Fixing mask timing..."
TS_FILE_TMP="${TMP_MASK%.mp4}_timestamps.csv"
if [ -f "$TS_FILE_TMP" ]; then
    FRAMES=$(wc -l < "$TS_FILE_TMP")
else
    FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$TMP_MASK" 2>/dev/null)
fi
if [ -n "$FRAMES" ] && [ -n "$REC_SECS" ]; then
    ACTUAL_FPS=$(echo "$FRAMES $REC_SECS" | awk '{printf "%.4f", $1/$2}')
    echo "[benchmark] Recorded $FRAMES frames in ${REC_SECS}s = ${ACTUAL_FPS} fps"
    TMP_FIXED=$(mktemp /tmp/mask_fixed_XXXXXX.mp4)
    ffmpeg -r "$ACTUAL_FPS" -i "$TMP_MASK" -r "$FPS_ARG_VAL" -c:v libx264 -preset fast -crf 18 "$TMP_FIXED" -y -loglevel error
    mv "$TMP_FIXED" "$TMP_MASK"
    echo "[benchmark] Timing fixed"

    if [ -f /tmp/debug_output.mp4 ]; then
        TMP_DEBUG_FIXED=$(mktemp /tmp/debug_fixed_XXXXXX.mp4)
        ffmpeg -r "$ACTUAL_FPS" -i /tmp/debug_output.mp4 -r "$FPS_ARG_VAL" -c:v libx264 -preset fast -crf 18 "$TMP_DEBUG_FIXED" -y -loglevel error
        mv "$TMP_DEBUG_FIXED" /tmp/debug_output.mp4
        echo "[benchmark] Debug video timing fixed"
    fi
fi

if grep -q 'Cannot open source' run_log.txt; then
    echo "[benchmark] ERROR: live_mask.py could not open the stream."
    echo "[benchmark] Start ffmpeg on the PC first: ffmpeg -re -i test_footage_small.mp4 -f mpegts udp://192.168.10.3:1234"
    exit 1
fi

# Primary: read from output file
RES=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$TMP_MASK" 2>/dev/null | tr ',' 'x')

# Fallback: parse the two dimensions from the [camera]/[video] log line
if [ -z "$RES" ]; then
    RES=$(grep -m1 '\[camera\]\|\[video\]' run_log.txt | grep -oP '\d+' | head -2 | paste -sd'x')
fi

if [ -z "$RES" ]; then
    echo "[benchmark] Could not determine resolution, using 'unknown'"
    RES="unknown"
fi

FILENAME="pred_mask_${MODEL}_${RES}.mp4"
DEST_DIR="Zappars@192.168.10.5:Documents/SAM-benchmark/pred_masks"

echo "[benchmark] Sending $FILENAME to PC..."
scp -o StrictHostKeyChecking=no "$TMP_MASK" "$DEST_DIR/$FILENAME"
TS_FILE="${TMP_MASK%.mp4}_timestamps.csv"
if [ -f "$TS_FILE" ]; then
    scp -o StrictHostKeyChecking=no "$TS_FILE" "$DEST_DIR/${FILENAME%.mp4}_timestamps.csv"
    scp -o StrictHostKeyChecking=no /tmp/stream_start_time.txt "$DEST_DIR/${FILENAME%.mp4}_stream_start.txt" 2>/dev/null
fi

if [ $? -eq 0 ]; then
    echo "[benchmark] Done — saved as $FILENAME"
    rm "$TMP_MASK"
    rm -f "$TS_FILE"
else
    echo "[benchmark] SCP failed. File left at $TMP_MASK"
fi
