# Benchmark Plan

Compare model segmentation accuracy and performance against a SAM2-generated ground truth, with the model running on a Raspberry Pi. All heavy computation runs on the PC — the Pi only runs the model.

---

## Metrics

### Detection & Mask Quality
Computed **on PC** from `pred_mask.mp4` vs `gt_mask.mp4`.

| Metric | What it measures | How |
|---|---|---|
| IoU | Overlap between predicted and GT mask | Per-frame + mean |
| Pixel Coverage | % of actual person pixels that got masked — the privacy-critical number | Per-frame + mean |
| Recall | Frames where a person is present in GT but not detected at all | Count + % |
| Precision | Of predicted person pixels, how many are actually person | Per-frame + mean |
| F1 Score | Harmonic mean of precision and recall | Per-frame + mean |
| mAP@50 | Detection quality: correct if mask covers >= 50% of GT person | Across full video |

### Temporal & Video Stability
Computed **on PC** from the per-frame IoU series.

| Metric | What it measures | How |
|---|---|---|
| Flickering Rate | Frames where person drops out then reappears within ~0.5s | Count gaps in recall |
| Temporal IoU Variance | Frame-to-frame stability of the mask | Std dev of per-frame IoU |
| MOTA | Covers missed detections, false alarms combined | Computed offline |

> **Note:** ID Switch Rate and full MOTA require tracked IDs. `live_mask.py` produces a merged binary mask with no IDs, so these are treated as single-object tracking (one "person" label per frame).

### Hardware & Performance
Collected **on Pi** with minimal overhead (logging only), analysed on PC.

| Metric | What it measures | How to collect |
|---|---|---|
| FPS @ resolution | Throughput at 480p / 720p / 1080p | Already printed by `live_mask.py` — pipe to log file |
| Degradation Curve | FPS over 5-10 min run — flat = pass, downward slope = thermal throttle | Parse timestamps from log |
| Cold Start Time | Time from script launch to first masked frame | Timestamp log lines |
| Peak RAM | Memory headroom for future versions | Sample `free -m` via SSH from PC while Pi runs |
| Inference Time Breakdown | Preprocess -> infer -> postprocess split | Add lightweight `time.time()` calls to seg_worker |

### Background Quality
Computed **on PC** — requires recording both raw input and model output (`--output`).

| Metric | What it measures | How |
|---|---|---|
| SSIM | Structural similarity of non-person regions before/after | Compare input frame vs output frame, masked to non-person area only |

---

## What to collect where

| Collected on Pi | Collected on PC (offline) |
|---|---|
| FPS (from terminal log) | IoU, Precision, Recall, F1, mAP@50 |
| Degradation curve (from log) | Flickering rate, Temporal IoU variance, MOTA |
| Cold start time (from log) | SSIM on background regions |
| Peak RAM (sampled via SSH from PC) | Inference time breakdown (parsed from log) |

**Pi load during benchmark:** only the model itself + writing `pred_mask.mp4`. All analysis happens after.

---

## Test Runs

### Run A — Accuracy benchmark
Stream a fixed-length clip, collect pred_mask, compare against GT offline.
```bash
# [PC — Git Bash] — full automation, one command
bash stream.sh GT_videos/test_footage_40s_1080p_30fps.mp4

# [PC] — run benchmark after pred_masks/ is populated
python3 benchmark.py \
  --gt GT_videos/gt_mask_1080p_30fps.mp4 \
  --pred pred_masks/pred_mask_yolo26n-seg_1080x1920.mp4
```

### Run B — Degradation / thermal test
Run for 5-10 minutes and log FPS over time.
```bash
# [PC — Git Bash] — stream on loop for the duration
"$FFMPEG" -stream_loop -1 -re -i GT_videos/test_footage_40s_1080p_30fps.mp4 -f mpegts udp://192.168.10.3:1234
```
Pi must be started manually first: `ssh sw6@192.168.10.3 "cd ~/Project/Prototype/P6 && ./run_benchmark.sh"`

### Run C — Resolution test
Repeat Run A at different input resolutions (480p / 720p / 1080p) to get FPS vs resolution tradeoff.
GT videos for all resolutions are pre-generated in `GT_videos/`.
```bash
bash stream.sh GT_videos/test_footage_40s_480p_30fps.mp4
bash stream.sh GT_videos/test_footage_40s_720p_30fps.mp4
bash stream.sh GT_videos/test_footage_40s_1080p_30fps.mp4
```

### Run D — Background quality (SSIM)
Record both the raw input and the model output to compare non-person regions.
Pass `--output /tmp/debug_output.mp4` to live_mask.py inside run_benchmark.sh.
Then compute SSIM on PC between input frames and output frames, restricted to non-person pixels from GT.

---

## Workflow (standard accuracy run)

### 1. Generate ground truth [PC, done once]
GT videos are already generated and stored in `GT_videos/`. To regenerate:
```bat
venv\Scripts\activate
venv\Scripts\python generate_gt_sam2.py --input GT_videos\test_footage_40s.mp4 --output GT_videos\gt_mask_1080p_30fps.mp4 --debug
```

### 2. Stream video and record pred mask [PC — Git Bash]
```bash
bash stream.sh GT_videos/test_footage_40s_1080p_30fps.mp4
```
This kills leftover Pi processes, starts `run_benchmark.sh` on the Pi, streams the video, and waits for the Pi to SCP the pred mask and timestamps back to `pred_masks/`.

### 3. Run benchmark [PC]
```bash
python3 benchmark.py \
  --gt GT_videos/gt_mask_1080p_30fps.mp4 \
  --pred pred_masks/pred_mask_yolo26n-seg_1080x1920.mp4
```
Comparison image is auto-saved to `tmp/mask_comparison.png` every run.

---

## Pi Info

| | |
|---|---|
| IP | 192.168.10.3 |
| User | sw6 |
| Password | jFTYQvI88pCnsHXXWHgZ |
| Project path | `~/Project/Prototype/P6/` |

PC must be on the `192.168.10.x` subnet (static IP `192.168.10.5` on the ethernet adapter).
See HOW_TO_RUN.md for full SSH setup instructions.

---

## TODO

- [x] Write `benchmark.py` — IoU, precision, recall, F1, mAP@50, flickering rate, temporal IoU variance
- [x] Frame alignment — per-frame timestamp-based alignment corrects for variable inference speed mid-video
- [x] `--pred-offset-seconds` — skip first N seconds of pred before comparing
- [x] `--stretch-gt` — resample GT to match pred frame count (fallback when no timestamps)
- [x] `--max-seconds` — cap comparison to first N seconds
- [x] Comparison image auto-generated at `tmp/mask_comparison.png` on every run
- [x] Fix NCHW input issue in `live_mask.py` for `student_seg_320` model
- [ ] Add lightweight timing to `seg_worker` for inference time breakdown
- [ ] Test all three models and compare results:
  - `selfie_segmenter` — lightweight MediaPipe baseline
  - `yolo26n-seg` — standard YOLO model (set `MODEL = "yolo26n-seg"` in live_mask.py)
  - `student_seg_320` — custom student model
- [ ] Run C: run all three resolutions (480p / 720p / 1080p) and compare FPS + quality
- [ ] Batch mode — CSV/config listing all video pairs, summary table output
- [ ] `--gt-offset-seconds` flag (symmetric to --pred-offset-seconds)
