#!/bin/bash
# Usage:
#   ./stream.sh video.mp4              — auto-detect FPS from the video
#   ./stream.sh video.mp4 --fps 30     — override FPS
#   ./stream.sh video.mp4 --debug      — also save debug overlay video to PC

FFMPEG="/c/Users/flemm/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin/ffmpeg.exe"
FFPROBE="/c/Users/flemm/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.1-full_build/bin/ffprobe.exe"
PI="sw6@192.168.10.3"

VIDEO=""
FPS=""
DEBUG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --fps)   FPS="$2"; shift 2 ;;
        --debug) DEBUG="1"; shift ;;
        *)       VIDEO="$1"; shift ;;
    esac
done

if [[ -z "$VIDEO" ]]; then
    echo "Usage: $0 <video> [--fps N] [--debug]"
    exit 1
fi

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
[[ -n "$DEBUG" ]] && DEBUG_ARG="--output /tmp/debug_output.mp4"

echo "[stream] Killing any leftover processes on Pi..."
ssh "$PI" "pkill -f live_mask.py; pkill -f run_benchmark.sh; sleep 1"

echo "[stream] Starting benchmark on Pi (--fps $FPS${DEBUG:+ --debug})..."
ssh "$PI" "nohup bash -c 'cd ~/Project/Prototype/P6 && ./run_benchmark.sh --fps $FPS $DEBUG_ARG > /tmp/benchmark_out.txt 2>&1' > /dev/null 2>&1 < /dev/null &"

echo "[stream] Waiting 5s for Pi model to load..."
sleep 5

echo "[stream] Starting ffmpeg stream..."
"$FFMPEG" -re -i "$VIDEO" -f mpegts udp://192.168.10.3:1234 &
FFMPEG_PID=$!

echo "[stream] Waiting for ffmpeg to finish..."
wait $FFMPEG_PID

echo "[stream] Stream done — waiting for Pi to finish processing..."
sleep 15

if [[ -n "$DEBUG" ]]; then
    echo "[stream] Fetching debug video from Pi..."
    scp "$PI:/tmp/debug_output.mp4" "debug_output.mp4" 2>/dev/null && \
        echo "[stream] Debug video saved to: $(pwd)/debug_output.mp4" || \
        echo "[stream] No debug video found on Pi"
fi

echo "[stream] Done"
