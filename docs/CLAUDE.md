# SAM-Benchmark — Project Context for Claude

This file captures the full state of the project so a new session can pick up exactly where we left off.

---

## What this repo does

Offline benchmark that compares a model's prediction mask video against a SAM2-generated ground truth mask video. Outputs detection quality, temporal stability, and (optionally) performance and background quality metrics.

**Main files:**
- `tools/benchmark.py` — runs the comparison
- `tools/stream.sh` — PC-side automation: streams video to Pi, waits for pred mask to arrive
- `tools/generate_gt_sam2.py` — generates GT mask video using SAM2 + Grounding DINO

---

## Repository layout

```
SAM-benchmark/
  README.md                 — entry point and folder guide
  tools/
    benchmark.py            — main benchmark script
    stream.sh               — PC-side benchmark automation (Git Bash or Linux shell)
    generate_gt_sam2.py     — GT mask generator (SAM2 + Grounding DINO)
    generate_gt.py          — older GT generator, use generate_gt_sam2.py instead
    sam_empty.bat           — Windows cleanup helper for GT generation cache
  docs/
    HOW_TO_RUN.md           — standard workflow + setup instructions
    BENCHMARK_PLAN.md       — metrics definitions, test run types
    SCRIPT_NOTES.md         — notes on planned batch script
    CHECKPOINT_2026-04-30.md — historical checkpoint notes
    CLAUDE.md               — repo map and working notes
    pi_stream_setup.txt     — Pi/PC setup notes
  data/
    raw/                    — original source footage
    gt/                     — pre-generated GT mask videos + source test footage
    preds/                  — pred mask videos SCPed from Pi after each run
  runs/
    benchmark/              — comparison images and benchmark logs
    gt-gen/                 — GT generator debug outputs
    stream/                 — streamed-run logs and optional debug video
  cache/
    gt-gen/                 — GT generation frames, masks, prompts cache
```

---

## Standard workflow

```bash
# 1. Stream video to Pi and record pred mask (Git Bash)
bash tools/stream.sh data/gt/test_footage_40s_1080p_30fps.mp4

# 2. Run benchmark (after data/preds/ is populated)
python3 tools/benchmark.py \
  --gt data/gt/gt_mask_1080p_30fps.mp4 \
  --pred data/preds/pred_mask_yolo26n-seg_1080x1920.mp4
```

---

## benchmark.py — CLI flags

```
--gt                    GT mask video (required)
--pred                  Prediction mask video (required)
--log                   live_mask.py log file — parses Seg FPS for performance section
--input                 Original raw video (needed for SSIM background quality)
--output                Model output video (needed for SSIM background quality)
--fps                   Override FPS — auto-detected from pred video if omitted
--max-seconds           Only compare first N seconds
--pred-offset-seconds   Skip first N seconds of pred before comparing
--stretch-gt            Fallback: resample GT to match pred frame count (used when no timestamps)
```

Comparison image is **always** generated under `runs/benchmark/<run-id>/mask_comparison.png` — no flag needed.

---

## Key implementation details in benchmark.py

- **GT loaded resized on-the-fly** — letterbox-resized to pred resolution frame-by-frame. Never held at full resolution (original GT was 4K portrait, ~20 GB if fully loaded).
- **Timestamp-based alignment** — if `pred_mask_..._timestamps.csv` exists next to the pred video, benchmark.py uses it for frame-accurate GT alignment. For each pred frame, GT frame = `round((pred_ts[i] - t0) * gt_fps)` where `t0` comes from `pred_mask_..._stream_start.txt` (the moment the Pi probe first saw a stream frame). This corrects for variable inference speed mid-video.
- **Fallback** — if no timestamps file, falls back to index-based alignment with optional `--stretch-gt`.
- **Comparison image** — 5 sample frames at 5/25/50/75/95% of compared frames, saved under `runs/benchmark/<run-id>/mask_comparison.png`. Green = GT only, Red = pred only, Yellow = overlap.

---

## Metrics output

**Detection & Mask Quality:** Mean IoU, Pixel Coverage (recall on person frames), Mean Recall, Mean Precision, Mean F1, mAP@50

**Temporal Stability:** Flickering Events, Temporal IoU Std Dev, MOTA (simplified), Missed detections, False alarms

**Performance** (requires `--log`): Mean/Min/Max Seg FPS, Thermal degradation curve

**Background Quality** (requires `--input` and `--output`): SSIM on non-person regions

---

## GT videos (pre-generated, Windows PC)

```
data/gt/
  gt_mask_1080p_30fps.mp4   — 1080×1920, 30fps
  gt_mask_720p_30fps.mp4    — 720×1280,  30fps
  gt_mask_480p_30fps.mp4    — 480×854,   30fps
  test_footage_40s.mp4              — source footage (3840×2160 portrait, original)
  test_footage_40s_1080p_30fps.mp4  — downscaled test video sent to Pi
  test_footage_40s_720p_30fps.mp4
  test_footage_40s_480p_30fps.mp4
```

GT generated with `generate_gt_sam2.py` (SAM2 + Grounding DINO) on the Windows 3070 Ti machine. Grounding DINO used instead of YOLO because footage has people partially cut off at door frames — YOLO misses these.

**Rule:** GT and pred must have the same aspect ratio and cover the same crop of the source. Mismatched aspect ratios will give IoU ≈ 0.

---

## stream.sh (PC-side)

Run from Git Bash in `SAM-benchmark/`:
```bash
bash tools/stream.sh data/gt/test_footage_40s_1080p_30fps.mp4 [--fps N] [--debug]
```

What it does:
1. Auto-detects FPS via ffprobe (or use `--fps` to override)
2. Kills any leftover processes on Pi
3. Starts `run_benchmark.sh` on Pi via SSH (nohup, detached)
4. Sleeps 5s for model to load
5. Streams the video via ffmpeg UDP
6. Waits for stream to finish, Pi wraps up
7. `--debug`: also SCPs the Pi's visual overlay video back as `debug_output.mp4`

---

## Pi-side files

**Pi:** `192.168.10.3`, user `sw6`, project at `~/Project/Prototype/P6/`

### run_benchmark.sh
- Accepts `--fps N` and `--debug` flags
- Uses `.venv/bin/python3` (cv2 lives in the venv, not system Python)
- Cleans up `/tmp/debug_output.mp4` at start (avoids re-stretching leftovers from previous runs)
- After live_mask.py exits: reads frame count from `_timestamps.csv` (fast `wc -l`) instead of slow ffprobe `--count_frames`
- Stretches raw mask from inference FPS to target FPS using ffmpeg
- SCPs to PC: `data/preds/pred_mask_{model}_{WxH}.mp4`, `_timestamps.csv`, `_stream_start.txt`

### live_mask.py (key changes relevant to benchmarking)
- Probe recreates `cv2.VideoCapture` every 20s to avoid OpenCV's 30s internal timeout — waits up to 120s for stream
- `STREAM_TIMEOUT = 0.1s` — stops 0.1s after stream ends
- Writes `/tmp/stream_start_time.txt` inside the probe the moment the first stream frame arrives — this is the true stream second 0 anchor for GT alignment
- Writes `{output_mask}_timestamps.csv` — one `time.time()` float per mask frame, written at the moment each frame is passed to VideoWriter
- Writes `/tmp/mask_start_ns` — nanosecond timestamp when mask writer opens, used by run_benchmark.sh to exclude probe wait time from the timing calculation

---

## Temporal alignment — how it works

The model runs at ~13fps on the Pi (variable due to thermal). The mask video is stretched to 30fps by run_benchmark.sh. Without correction, this would cause mid-video drift.

**Correction:**
1. `stream_start_time.txt`: when the Pi's probe first reads a frame from the stream = stream second 0
2. `_timestamps.csv`: when each raw inference frame was written to the mask file
3. For pred frame i: `gt_frame = round((pred_ts[i] - stream_start) * gt_fps)`
4. The stretched video is sampled at `round(i * N_stretched / N_raw)` to get the right frame for each timestamp

This corrects both the global offset (model loads after stream starts) and per-frame drift (variable inference speed).

---

## Spatial alignment — lessons learned

The first test run (`yolo26n-seg_320x320` pred vs 2160×3840 GT) gave IoU ≈ 0.07 and mAP@50 = 0.0. Visual inspection showed people in completely different pixel positions — GT was letterboxed into a narrow portrait strip, pred was square. The metrics were noise.

**Rule:** always use matching aspect ratio for GT and pred.

---

## Related repo — P6 (live mask pipeline)

**Pi project path:** `~/Project/Prototype/P6/`
**Main file:** `live_mask.py`
**Pi:** IP `192.168.10.3`, user `sw6`

Models available (set `MODEL` in `live_mask.py`):
- `yolo26n-seg` — standard YOLO model
- `selfie_segmenter` — lightweight MediaPipe baseline
- `student_seg_320` — custom student model (NCHW input layout, handled automatically)

---

## Running on Linux

The code works on Linux with two small changes:

**1. `tools/stream.sh` — uses portable `ffmpeg` / `ffprobe` defaults; override with `FFMPEG=` and `FFPROBE=` if needed.**

**2. `run_benchmark.sh` on the Pi — update SCP destination:**
The Pi SCPs to `Zappars@192.168.10.5:Documents/SAM-benchmark/data/preds/` (Windows username + path).
Change `Zappars` to the Linux username and update the path to match (e.g. `/home/<user>/SAM-benchmark/data/preds/`).

**Everything else works as-is** — `tools/benchmark.py`, `tools/generate_gt_sam2.py`, and all Pi-side files are platform-agnostic.

**Linux-specific setup notes (instead of the Windows steps in HOW_TO_RUN.md):**
- Static IP: `sudo ip addr add 192.168.10.5/24 dev <interface>` (or via NetworkManager)
- SSH server: `sudo systemctl enable --now ssh`
- No Git Bash needed — run `bash tools/stream.sh ...` in any terminal
- Python venv: use `venv/bin/python` instead of `venv\Scripts\python`

---

## TODO

- [ ] Batch mode — CSV/config listing all video pairs, summary table output
- [ ] `--pred-offset-seconds` for GT side (`--gt-offset-seconds`)
- [ ] Inference time breakdown — add `time.time()` calls to seg_worker, parse from log
- [ ] Test all three models and compare results at 480p / 720p / 1080p
- [ ] Run C: downscaled videos at 480p / 720p for resolution FPS test
