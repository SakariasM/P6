# Checkpoint — 2026-04-30

## What works at this point

### stream.sh (PC-side automation)
Run a full benchmark from the PC with one command:
```bash
bash tools/stream.sh data/gt/test_footage_40s_1080p_30fps.mp4 --debug
```
- Auto-detects FPS via ffprobe
- Kills leftover Pi processes
- Starts `run_benchmark.sh` on Pi via SSH (nohup, non-blocking)
- Waits 5s for model to load, then starts ffmpeg stream
- Waits for stream to finish, then waits 0.1s for Pi to clean up
- Fetches debug video from Pi if `--debug` was passed

### run_benchmark.sh (Pi-side)
- Accepts `--fps N` and `--debug` flags
- Uses `.venv/bin/python3` (cv2 lives in the venv)
- Records wall-clock time using `mask_start_ns` written by `live_mask.py` the moment the mask writer opens — this excludes probe wait time from the stretch calculation
- Stretches raw mask video from inference FPS to target FPS using ffmpeg
- Also stretches debug video if `--debug`
- SCPs pred mask to `Zappars@192.168.10.5:Documents/SAM-benchmark/data/preds/`

### live_mask.py (Pi-side, key changes)
- Probe recreates `cv2.VideoCapture` every 20s to avoid the 30s OpenCV internal timeout — can wait up to 120s for stream
- `STREAM_TIMEOUT = 0.1s` — stops 0.1s after stream ends
- Writes `/tmp/mask_start_ns` (nanosecond unix timestamp) when mask writer opens
- No artificial pre-roll — natural black frames during model load are captured as real output

### benchmark.py (PC-side)
- `--pred-offset-seconds N` — skip first N seconds of pred before comparing
- `--stretch-gt` — resample GT to match pred frame count
- Automatically generates `runs/benchmark/<run-id>/mask_comparison.png` after every run
  - 5 sample frames at 5%, 25%, 50%, 75%, 95% of video
  - Green = GT only, Red = pred only, Yellow = overlap

---

## Known issue — temporal drift

The pred mask has non-uniform speed mid-video due to variable inference rate (thermal throttling, scene complexity). The single-ratio ffmpeg stretch corrects the overall duration but not the per-frame timing. Observed behaviour on the yolo26n-seg 1080p test:

- At ~1s: alignment correct
- At ~20s: ~2s drift (pred appears behind GT)
- At ~36.7s: alignment correct again

This causes IoU to be underestimated. The planned fix is **Option C — per-frame timestamps**:
1. `live_mask.py` writes a `.csv` sidecar file with one unix timestamp per mask frame
2. `benchmark.py` does timestamp-based nearest-frame matching instead of index-based
3. GT timestamps = `frame_index / fps` (uniform, from video file)

This was agreed on but not yet implemented. It is the next thing to build.

---

## Latest benchmark result — yolo26n-seg 1080×1920

Run: `--pred-offset-seconds 2 --stretch-gt` (2s to skip model load black frames)

| Metric | Value |
|---|---|
| Mean IoU | 0.1831 |
| Pixel Coverage | 59.9% |
| Mean Recall | 0.6156 |
| Mean Precision | 0.3082 |
| Mean F1 | 0.2737 |
| mAP@50 | 0.0000 |
| Flickering Events | 1 |
| Temporal IoU Std Dev | 0.1108 |
| MOTA | 0.8866 |
| Missed detections | 131 / 1108 (11.8%) |
| False alarms | 0 |

Low IoU and zero mAP@50 indicate the model struggles at 1080p — masks are imprecise and covered in spurious blobs. MOTA is high because it tracks continuously; the problem is mask *quality* not *presence*.

---

## Pi info

| | |
|---|---|
| IP | 192.168.10.3 |
| User | sw6 |
| Project path | `~/Project/Prototype/P6/` |
| Python | `.venv/bin/python3` |
| Model | `yolo26n-seg` (set in `live_mask.py`) |

## PC info (Windows)

| | |
|---|---|
| Username | Zappars (home: `C:\Users\flemm`) |
| SSH server | OpenSSH, auto-starts on boot |
| Pred masks | `C:\Users\flemm\Documents\SAM-benchmark\pred_masks\` |
| GT videos | `C:\Users\flemm\Documents\SAM-benchmark\GT_videos\` |
