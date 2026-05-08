# Benchmark Script Notes

---

## Features implemented in benchmark.py

### 1. On-the-fly GT resize (DONE)
GT video may be a different resolution than pred.
Each GT frame is letterbox-resized to pred resolution during load — never holds full-res GT array in memory.

### 2. Max seconds cap (DONE — `--max-seconds`)
Only compare the first N seconds of footage.
Example: `--max-seconds 32`

### 3. Temporal offset (DONE — `--pred-offset-seconds`)
GT starts at second 0 but pred has leading black frames before the matching content starts.
Example: `--pred-offset-seconds 2` skips the first 2 seconds of pred before comparing.

### 4. Temporal resampling / stretch (DONE — `--stretch-gt`)
Fallback when no timestamps file. GT and pred cover the same content but different frame counts.
For pred frame i, matching GT frame = round(i * N_gt / N_pred).
Used automatically when timestamps are unavailable.

### 5. Timestamp-based alignment (DONE)
Primary alignment method when `_timestamps.csv` and `_stream_start.txt` are present alongside the pred video.

For each raw inference frame i:
- `gt_frame = round((pred_ts[i] - stream_start) * gt_fps)`
- `stream_start` = timestamp from `_stream_start.txt` (when Pi probe saw its first stream frame)

This corrects for variable inference speed mid-video (thermal throttling, scene complexity).

The stretched pred video is sampled at `round(i * N_stretched / N_raw)` to recover one frame per raw inference timestamp.

### 6. Comparison image (DONE)
Auto-generated at `runs/benchmark/<run-id>/mask_comparison.png` on every run.
5 sample frames at 5%, 25%, 50%, 75%, 95% of the compared range.
- Green = GT only (missed by model)
- Red = pred only (false positive)
- Yellow = overlap (correct)

### 7. Output folder
All outputs go to `runs/benchmark/<run-id>/` — never `/tmp`.

---

## Current state of benchmark.py

Flags:
- `--gt` — GT mask video (required)
- `--pred` — prediction mask video (required)
- `--log` — live_mask.py log file, parsed for Seg FPS (performance section)
- `--input` — original raw video (needed for SSIM background quality)
- `--output` — model output video (needed for SSIM background quality)
- `--fps` — override FPS (auto-detected from pred if omitted)
- `--max-seconds` — only compare first N seconds
- `--pred-offset-seconds` — skip first N seconds of pred before comparing
- `--stretch-gt` — resample GT to match pred frame count (fallback)

Key functions:
- `load_mask_video(path, max_frames, resize_to)` — resizes on load
- `get_video_fps(path)` — auto-detects FPS from video

---

## TODO

- [ ] Batch mode — accept CSV/config listing (data/gt path, data/preds path, options) pairs, output summary table
- [ ] `--gt-offset-seconds` — symmetric to --pred-offset-seconds, skip first N seconds of GT
- [ ] Inference time breakdown — add `time.time()` calls to seg_worker, parse from log
