# SAM Benchmark — How to Run

---

## Running a benchmark (standard workflow)

Open **Git Bash** in `C:\Users\flemm\Documents\SAM-benchmark`.

### Step 1 — Stream the video and record the pred mask
```bash
bash tools/stream.sh data/gt/test_footage_40s_1080p_30fps.mp4
```
This will:
1. Kill any leftover processes on the Pi
2. Start `run_benchmark.sh` on the Pi via SSH
3. Wait 5 seconds for the model to load
4. Stream the video from PC to Pi over UDP
5. Wait for the Pi to stretch and SCP the pred mask + timestamps back to `data/preds/`

Optional flag: `--debug` — also records and fetches the Pi's visual overlay video.

### Step 2 — Run the benchmark
```bash
python3 tools/benchmark.py \
  --gt data/gt/gt_mask_1080p_30fps.mp4 \
  --pred data/preds/pred_mask_yolo26n-seg_1080x1920.mp4
```
This automatically uses the timestamps file (`pred_mask_..._timestamps.csv`) if present for accurate frame alignment. A comparison image is saved under `runs/benchmark/<run-id>/mask_comparison.png` every run.

#### Useful flags
| Flag | Example | Effect |
|---|---|---|
| `--pred-offset-seconds` | `--pred-offset-seconds 1` | Skip first N seconds of pred |
| `--max-seconds` | `--max-seconds 35` | Only compare first N seconds |
| `--stretch-gt` | `--stretch-gt` | Resample GT to match pred length (fallback when no timestamps) |

---

## GT videos (pre-generated)

| File | Resolution | FPS |
|---|---|---|
| `data/gt/gt_mask_1080p_30fps.mp4` | 1080×1920 | 30 |
| `data/gt/gt_mask_720p_30fps.mp4` | 720×1280 | 30 |
| `data/gt/gt_mask_480p_30fps.mp4` | 480×854 | 30 |

Source footage: `data/gt/test_footage_40s.mp4` (3840×2160, portrait, 60fps original)
Test footage (sent to Pi): `data/gt/test_footage_40s_1080p_30fps.mp4` etc.

---

## How temporal alignment works

`live_mask.py` writes two files alongside the pred mask:
- `_timestamps.csv` — one unix timestamp per mask frame (written at the moment each frame is saved)
- `_stream_start.txt` — timestamp when the probe first received a frame from the stream (= stream second 0 on the Pi)

`benchmark.py` uses these to map each pred frame to its correct GT frame:
```
gt_frame = round((pred_timestamp - stream_start) * gt_fps)
```
This corrects for variable inference speed (thermal throttling etc.) mid-video.

---

# SAM2 Ground Truth Generator

Generates a binary mask video where **white = person, black = background**.
Uses SAM2 (accurate tracker) + Grounding DINO (person detector).
Automatically uses GPU if available — strongly recommended for practical speed.

People entering mid-video are detected automatically — the entire video is scanned before tracking begins.

---

## Pi SSH Setup (new Windows PC)

Do this once whenever you move to a new Windows machine to get passwordless SSH access to the Pi.

### 1. Set a static IP on the ethernet adapter

The Pi is at `192.168.10.3` — your PC needs to be on the same subnet.

Open an **admin PowerShell** (right-click → Run as administrator) and run:
```powershell
netsh interface ip set address name="Ethernet 2" static 192.168.10.5 255.255.255.0
```
> The adapter name may differ. Run `netsh interface show interface` to find the one connected to the Pi (it will show a 169.254.x.x APIPA address if no static IP is set yet).

Alternatively via GUI: **Settings → Network & Internet → Advanced network settings → Ethernet 2 → Edit → Manual → IPv4: `192.168.10.5`, Subnet: `255.255.255.0`**

Verify the Pi is reachable:
```bash
ping -n 2 192.168.10.3
```

### 2. Generate an SSH key

In Git Bash (or any terminal):
```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
```

### 3. Copy the key to the Pi

```bash
cat ~/.ssh/id_ed25519.pub | ssh sw6@192.168.10.3 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```
Enter password `jFTYQvI88pCnsHXXWHgZ` when prompted. This is the only time you need the password.

### 4. Enable OpenSSH Server on this PC

The Pi needs to SCP pred_mask files back to this PC. This requires an SSH server running here.

Open an **admin PowerShell** and run:
```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
```

> `StartupType Automatic` means the SSH server starts on every boot — no need to re-enable it each session. To disable: `Stop-Service sshd` and `Set-Service -Name sshd -StartupType Disabled`.

### 5. Authorize the Pi to SCP to this PC

```bash
ssh sw6@192.168.10.3 "cat ~/.ssh/id_ed25519.pub" >> ~/.ssh/authorized_keys
```

### 6. Verify passwordless SSH works

```bash
ssh sw6@192.168.10.3
```
Should log in without a password prompt.

---

## Setup (Windows — fresh machine)

> Do this once on the Windows PC with the GPU.

### 1. Install Python
Download and install Python 3.11 from https://www.python.org/downloads/
Check "Add Python to PATH" during install.

### 2. Create a virtual environment
Open a terminal in the SAM2-benchmark folder:
```bat
python -m venv venv
venv\Scripts\activate
```

### 3. Install PyTorch with CUDA (for 3070 Ti)
```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Verify GPU is detected:
```bat
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Should print `True` and your GPU name. If it prints `False`, your CUDA drivers may need updating.

### 4. Install dependencies
```bat
pip install opencv-python numpy tqdm Pillow transformers
```

### 5. Install SAM2
```bat
git clone https://github.com/facebookresearch/sam2.git sam2_repo
cd sam2_repo
pip install -e .
cd ..
```
> If this fails on Windows due to CUDA compilation errors, use WSL2 instead (see bottom of this file).

### 6. Download SAM2 checkpoint
Create the checkpoints folder and download the model:
```bat
mkdir sam2_repo\checkpoints
```
Download `sam2.1_hiera_large.pt` from https://github.com/facebookresearch/sam2
and place it in `sam2_repo\checkpoints\`.

---

## Run (Windows)

```bat
venv\Scripts\activate
venv\Scripts\python tools\generate_gt_sam2.py --input data\gt\test_footage_40s.mp4 --output data\gt\gt_mask_1080p_30fps.mp4 --debug
```

Always use `generate_gt_sam2.py` — it is more accurate than `generate_gt.py` because SAM2 tracks
people through the whole video rather than detecting per-frame.

> Use the **original uncompressed video** as input for ground truth, not the downscaled test version.
> GT masks can be downscaled later to match the test resolution if needed.

---

## Run (Ubuntu / this machine)

```bash
cd /home/flemming/Documents/GitHub/P6
venv/bin/python tools/generate_gt_sam2.py --input /path/to/video.mp4 --output data/gt/gt_mask_1080p_30fps.mp4 --debug
```

---

## Pause & Resume

Press **Ctrl+C** at any time — progress is saved automatically.
To resume, re-run the exact same command.

---

## Output Files

### From `generate_gt_sam2.py` (ground truth generation)

| File | Description |
|---|---|
| `data/gt/gt_mask.mp4` | **The ground truth.** Binary mask video — white = person, black = background. This is what you compare the model against in `benchmark.py`. |
| `runs/gt-gen/<run-name>/debug.mp4` | Visual verification video — original footage with green person overlay on the left, binary mask on the right. Use this to check the GT looks correct before running a full benchmark. Only created with `--debug`. |
| `cache/gt-gen/<run-name>/frames/` | Extracted JPEG frames from the input video. Kept on disk so a resumed run skips re-extraction. Safe to delete after GT is confirmed correct — it just means re-extraction if you rerun. |
| `cache/gt-gen/<run-name>/prompts.pkl` | Cached DINO scan prompts used to resume/reuse the scan stage. |

### From `run_benchmark.sh` on the Pi (model output)

Run `./run_benchmark.sh` on the Pi instead of calling `live_mask.py` directly. The script reads the current `MODEL` from `live_mask.py`, runs the model, then automatically SCPs the result to this PC when the stream ends.

| File | Description |
|---|---|
| `pred_mask_{model}_{WxH}.mp4` | **The model prediction.** Binary mask video in the same format as `gt_mask.mp4`. Auto-named from the active model and resolution, auto-sent to `/home/flemming/Documents/GitHub/P6/data/preds/` on this PC. |
| `run_log.txt` | Terminal output from the Pi run (saved in `~/Project/Prototype/P6/`). Contains FPS readings used by `benchmark.py` for performance and thermal degradation analysis. |
| `output.mp4` | Full colour output video from the model (optional). Only needed for SSIM background quality analysis. Pass `--output output.mp4` to `live_mask.py` inside the script if needed. |

### From `benchmark.py` (analysis, runs on this PC)

`benchmark.py` prints results to the terminal — no output files. Pipe to a text file if you want to save results:
```bat
python3 tools/benchmark.py --gt data/gt/gt_mask.mp4 --pred data/preds/pred_mask.mp4 --log run_log.txt > results.txt
```

---

## Options

| Flag | Default | Description |
|---|---|---|
| `--conf` | `0.4` | YOLO detection confidence. Lower if no people are detected. |
| `--debug` | off | Write side-by-side debug video — always use this on first run to verify masks |

Example with lower confidence:
```bat
venv\Scripts\python tools\generate_gt_sam2.py --input video.mp4 --output data\gt\gt_mask.mp4 --conf 0.2
```

---

## Notes

- First run downloads Grounding DINO model weights (~700 MB) automatically from HuggingFace
- GPU (3070 Ti) will be used automatically if PyTorch CUDA is installed correctly
- If you get *"No people detected"*, try `--conf 0.2`
- Partial bodies (limbs only) may not be detected — Grounding DINO requires a recognisable person shape

---

## WSL2 fallback (if SAM2 install fails on Windows)

If `pip install -e .` in the sam2_repo step fails due to CUDA compilation errors:

1. Install WSL2: `wsl --install` in PowerShell (restart required)
2. Open a WSL2 terminal and repeat all setup steps using the Linux setup section above
3. Access Windows files from WSL2 at `/mnt/c/Users/<your name>/...`

---

---

# PART 2 — COCO mAP Benchmark

> **Linux only.** No GPU needed — runs on CPU.

Evaluates a TFLite segmentation model against the official COCO val2017 dataset
(person class only) and reports mAP@50-95 (mask) — the same metric Ultralytics
uses to publish their model numbers.

Run from the **P6 folder root** using its venv.

---

## Setup (Linux — once only)

```bash
cd /path/to/P6
venv/bin/pip install ai-edge-litert pycocotools
```

---

## Running the COCO Benchmark (Linux)

```bash
cd /path/to/P6

# Full evaluation — ~2693 person images, ~5–10 min on CPU
venv/bin/python tools/coco_benchmark.py \
  --model models/yolo26n-seg_float32.tflite

# Quick sanity check — 100 images, ~1 min
venv/bin/python tools/coco_benchmark.py \
  --model models/yolo26n-seg_float32.tflite \
  --max-images 100
```

Always use `--threshold 0.5` (the default) for benchmarking. Do not use the
`MASK_THRESHOLD = 0.7` from `live_mask.py` — that is tuned for the live pipeline
and will artificially shrink masks and hurt the score.

---

## COCO Dataset

Downloaded automatically on first run (~1.3 GB total) into `datasets/coco_data/`:
- `val2017/` — 5,000 validation images (~1 GB)
- `annotations/instances_val2017.json` — annotations (~236 MB)

Only the ~2,693 images containing at least one person annotation are evaluated.

If the dataset is already downloaded, point to its location with `--coco-dir`.

---

## Options (`coco_benchmark.py`)

| Flag | Default | Description |
|---|---|---|
| `--model` | required | Path to `.tflite` model file |
| `--coco-dir` | `./datasets/coco_data` | Where to store/find COCO data |
| `--threshold` | `0.5` | Mask probability threshold — keep at 0.5 for benchmarking |
| `--min-area` | `100` | Minimum connected-component size in pixels (filters noise) |
| `--max-images` | all | Limit number of images — useful for quick checks |

---

## Output

Results are printed to the terminal and **appended** to `logs/coco_logs.txt`.
The `logs/` folder is created automatically. Each entry includes the date, model
name, threshold, image count, and the full metric table so all runs are stored in
one place for comparison.

---

## Interpreting Results

| Metric | What it means |
|---|---|
| mAP@50-95 | Primary metric — averaged over IoU thresholds 0.50 to 0.95 |
| mAP@50 | Loose — mask just needs to cover the person roughly |
| mAP@75 | Strict — mask boundary must be fairly precise |
| mAP small / medium / large | Breakdown by person size in the original image |

**Note on resolution:** models are exported at 320×320 input. Ultralytics publishes
their numbers at 640×640. Small and medium object scores will be near zero at 320px
— this is expected. The large-object score is the most meaningful number to compare.