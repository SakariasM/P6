#!/usr/bin/env python3
"""
benchmark.py — offline benchmark: GT mask vs model prediction mask.

Usage:
    python3 tools/benchmark.py --gt data/gt/gt_mask.mp4 --pred data/preds/pred_mask.mp4
    python3 tools/benchmark.py --gt data/gt/gt_mask.mp4 --pred data/preds/pred_mask.mp4 --log run_log.txt
    python3 tools/benchmark.py --gt data/gt/gt_mask.mp4 --pred data/preds/pred_mask.mp4 --input raw.mp4 --output model_out.mp4
"""
import argparse
import datetime
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs" / "benchmark"


# ── video loading ─────────────────────────────────────────────────────────────

def get_video_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 20.0


def load_mask_video(path, max_frames=None, resize_to=None):
    """Load frames from a binary mask video as (N, H, W) bool array.

    resize_to: optional (target_h, target_w) — letterbox-resizes each frame on
               load so the full-size array is never held in memory.
    max_frames: stop after this many frames.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: cannot open {path}")
        sys.exit(1)
    frames = []
    while True:
        if max_frames and len(frames) >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        mask = gray > 127
        if resize_to:
            target_h, target_w = resize_to
            src_h, src_w = mask.shape
            scale = min(target_w / src_w, target_h / src_h)
            new_w, new_h = int(src_w * scale), int(src_h * scale)
            pad_l = (target_w - new_w) // 2
            pad_t = (target_h - new_h) // 2
            buf = np.zeros((target_h, target_w), dtype=bool)
            resized = cv2.resize(mask.astype(np.uint8) * 255, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)
            buf[pad_t:pad_t + new_h, pad_l:pad_l + new_w] = resized > 127
            mask = buf
        frames.append(mask)
    cap.release()
    return np.array(frames)


def resize_masks_to(frames, target_h, target_w):
    """Letterbox-resize a (N, H, W) bool array to (N, target_h, target_w)."""
    src_h, src_w = frames[0].shape
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    pad_left = (target_w - new_w) // 2
    pad_top  = (target_h - new_h) // 2
    out = np.zeros((len(frames), target_h, target_w), dtype=bool)
    for i, f in enumerate(frames):
        resized = cv2.resize(f.astype(np.uint8) * 255, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)
        out[i, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized > 127
    return out


def load_color_video(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and len(frames) >= max_frames):
            break
        frames.append(frame)
    cap.release()
    return frames


# ── per-frame metrics ─────────────────────────────────────────────────────────

def frame_metrics(gt, pred):
    """Return (iou, precision, recall, f1) for one frame pair."""
    tp    = np.logical_and(gt, pred).sum()
    fp    = np.logical_and(~gt, pred).sum()
    fn    = np.logical_and(gt, ~pred).sum()
    union = np.logical_or(gt, pred).sum()

    iou  = tp / union if union > 0 else (1.0 if not gt.any() else 0.0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if not gt.any() else 0.0)
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return iou, prec, rec, f1


# ── temporal metrics ──────────────────────────────────────────────────────────

def flickering_rate(gt_person, pred_person, fps, max_gap_sec=0.5):
    """Count flicker events: missed detections shorter than max_gap_sec that recover."""
    max_gap = int(max_gap_sec * fps)
    flickers, gap_len, in_gap = 0, 0, False
    for g, p in zip(gt_person, pred_person):
        if g:
            if not p:
                in_gap = True
                gap_len += 1
            else:
                if in_gap and gap_len <= max_gap:
                    flickers += 1
                in_gap, gap_len = False, 0
        else:
            in_gap, gap_len = False, 0
    return flickers


# ── log parsing ───────────────────────────────────────────────────────────────

def parse_log(log_path):
    """Extract seg FPS values from live_mask.py log output."""
    fps_vals = []
    for line in open(log_path):
        m = re.search(r'Seg:\s*([\d.]+)', line)
        if m:
            fps_vals.append(float(m.group(1)))
    return np.array(fps_vals) if fps_vals else None


# ── SSIM on background ────────────────────────────────────────────────────────

def ssim_background(input_frames, output_frames, gt_masks):
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        print("[ssim] scikit-image not installed — pip install scikit-image")
        return None
    scores = []
    for inp, out, mask in zip(input_frames, output_frames, gt_masks):
        if not (~mask).any():
            continue
        ig = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY).astype(float)
        og = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(float)
        ig[mask] = 0
        og[mask] = 0
        score, _ = ssim(ig, og, full=True, data_range=255)
        scores.append(score)
    return np.mean(scores) if scores else None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt',     required=True,            help='Ground truth mask video')
    ap.add_argument('--pred',   required=True,            help='Prediction mask video')
    ap.add_argument('--log',    default=None,             help='live_mask.py log file (for FPS)')
    ap.add_argument('--input',  default=None,             help='Original raw input video (for SSIM)')
    ap.add_argument('--output', default=None,             help='Model output video (for SSIM)')
    ap.add_argument('--fps',         type=float, default=None,  help='Override FPS (auto-detected from pred video if omitted)')
    ap.add_argument('--max-seconds', type=float, default=None,  help='Only compare first N seconds of footage')
    ap.add_argument('--stretch-gt',  action='store_true',       help='Temporally resample GT to match pred duration (use when pred is slower/longer but covers the same content)')
    ap.add_argument('--pred-offset-seconds', type=float, default=0.0, help='Skip first N seconds of pred video before comparing')
    args = ap.parse_args()

    fps = args.fps if args.fps else get_video_fps(args.pred)
    max_frames = int(fps * args.max_seconds) if args.max_seconds else None
    pred_offset_frames = int(fps * args.pred_offset_seconds)

    # Load pred first to know target resolution, then load GT resized on-the-fly
    print(f"Loading pred masks (max_frames={max_frames}, offset={pred_offset_frames} frames)...")
    pred_masks = load_mask_video(args.pred, max_frames=max_frames)
    if pred_offset_frames:
        pred_masks = pred_masks[pred_offset_frames:]
    pred_h, pred_w = pred_masks.shape[1], pred_masks.shape[2]

    print(f"Loading GT masks (resizing to {pred_w}x{pred_h} on-the-fly)...")
    gt_masks_full = load_mask_video(args.gt, resize_to=(pred_h, pred_w))
    gt_fps = get_video_fps(args.gt)

    # -- timestamp-based alignment --
    ts_path = args.pred.replace(".mp4", "_timestamps.csv")
    if os.path.exists(ts_path):
        pred_ts = np.loadtxt(ts_path)
        if pred_offset_frames:
            pred_ts = pred_ts[pred_offset_frames:]
        if max_frames:
            pred_ts = pred_ts[:max_frames]

        # The timestamps belong to the raw inference frames, but pred_masks is the
        # stretched video (more frames). Sample the stretched video at the positions
        # that correspond to each raw frame.
        n_raw = len(pred_ts)
        n_stretched = len(pred_masks)
        raw_indices = np.round(np.arange(n_raw) * n_stretched / n_raw).astype(int)
        raw_indices = np.clip(raw_indices, 0, n_stretched - 1)
        pred_masks = pred_masks[raw_indices]  # now n_raw frames, one per timestamp

        # Use stream start time as t0 so GT alignment starts at stream second 0,
        # not at the first mask frame (which may be 1-2s after stream start)
        stream_start_path = args.pred.replace(".mp4", "_stream_start.txt")
        if os.path.exists(stream_start_path):
            t0 = float(open(stream_start_path).read().strip())
            print(f"[timestamps] Stream start offset: {pred_ts[0] - t0:.2f}s before first mask frame")
        else:
            t0 = pred_ts[0]

        # For each pred frame, find the nearest GT frame by elapsed time
        gt_indices = np.round((pred_ts - t0) * gt_fps).astype(int)
        gt_indices = np.clip(gt_indices, 0, len(gt_masks_full) - 1)
        gt_masks = gt_masks_full[gt_indices]

        elapsed = pred_ts[-1] - pred_ts[0]
        print(f"[timestamps] Loaded {n_raw} pred timestamps over {elapsed:.1f}s "
              f"({n_raw/elapsed:.1f} fps avg) — sampled from {n_stretched} stretched frames")
        print(f"[timestamps] GT frame range used: {gt_indices[0]} - {gt_indices[-1]}")
    else:
        print("[timestamps] No timestamp file found — falling back to index-based alignment")
        gt_masks = gt_masks_full

        # Temporal resampling: stretch GT to match pred length (nearest-frame lookup)
        if args.stretch_gt and len(gt_masks) != len(pred_masks):
            n_pred, n_gt = len(pred_masks), len(gt_masks)
            indices = np.round(np.arange(n_pred) * n_gt / n_pred).astype(int).clip(0, n_gt - 1)
            gt_masks = gt_masks[indices]
            print(f"[stretch-gt] Resampled GT from {n_gt} -> {n_pred} frames (ratio {n_pred/n_gt:.3f}x)")

    n = min(len(gt_masks), len(pred_masks))
    if len(gt_masks) != len(pred_masks):
        print(f"[warn] Frame count mismatch — GT={len(gt_masks)}, pred={len(pred_masks)}, truncating to {n}")
    gt_masks   = gt_masks[:n]
    pred_masks = pred_masks[:n]
    print(f"Comparing {n} frames at {fps:.1f} FPS\n")

    # per-frame
    ious, precs, recs, f1s = [], [], [], []
    gt_person, pred_person = [], []
    for gt, pred in zip(gt_masks, pred_masks):
        iou, prec, rec, f1 = frame_metrics(gt, pred)
        ious.append(iou);  precs.append(prec)
        recs.append(rec);  f1s.append(f1)
        gt_person.append(gt.any())
        pred_person.append(pred.any())

    ious  = np.array(ious);  precs = np.array(precs)
    recs  = np.array(recs);  f1s   = np.array(f1s)
    gt_p  = np.array(gt_person)
    pred_p = np.array(pred_person)
    n_person = gt_p.sum()

    # mAP@50
    map50 = ((ious >= 0.5) & gt_p).sum() / n_person if n_person else 0.0

    # MOTA
    fn   = (gt_p & ~pred_p).sum()
    fp   = (~gt_p & pred_p).sum()
    mota = 1.0 - (fn + fp) / n

    # flickering
    flickers = flickering_rate(gt_person, pred_person, fps)

    # temporal variance (only on frames with a person)
    iou_std = ious[gt_p].std() if n_person else 0.0

    # pixel coverage = mean recall on person frames
    pixel_coverage = recs[gt_p].mean() if n_person else 0.0

    # ── output ────────────────────────────────────────────────────────────────
    sep = "=" * 52

    print(sep)
    print("  DETECTION & MASK QUALITY")
    print(sep)
    print(f"  Mean IoU              : {ious.mean():.4f}")
    print(f"  Pixel Coverage        : {pixel_coverage*100:.1f}%  (mean recall on person frames)")
    print(f"  Mean Recall           : {recs.mean():.4f}")
    print(f"  Mean Precision        : {precs.mean():.4f}")
    print(f"  Mean F1               : {f1s.mean():.4f}")
    print(f"  mAP@50                : {map50:.4f}")
    print()
    print(sep)
    print("  TEMPORAL STABILITY")
    print(sep)
    print(f"  Flickering Events     : {flickers}")
    print(f"  Temporal IoU Std Dev  : {iou_std:.4f}")
    print(f"  MOTA (simplified)     : {mota:.4f}")
    if n_person:
        print(f"  Missed detections     : {fn} / {n_person} person frames  ({fn/n_person*100:.1f}%)")
    print(f"  False alarms          : {fp} frames")

    if args.log:
        fps_vals = parse_log(args.log)
        print()
        print(sep)
        print("  PERFORMANCE (from log)")
        print(sep)
        if fps_vals is not None:
            split = max(1, len(fps_vals) // 10)
            start = fps_vals[:split].mean()
            end   = fps_vals[-split:].mean()
            print(f"  Mean Seg FPS          : {fps_vals.mean():.1f}")
            print(f"  Min / Max Seg FPS     : {fps_vals.min():.1f} / {fps_vals.max():.1f}")
            print(f"  Thermal degradation   : {start:.1f} -> {end:.1f} FPS  (drop: {start-end:.1f})")
        else:
            print("  No FPS values found in log")

    if args.input and args.output:
        print()
        print(sep)
        print("  BACKGROUND QUALITY")
        print(sep)
        print("  Loading videos for SSIM...")
        inp_frames = load_color_video(args.input,  max_frames=n)
        out_frames = load_color_video(args.output, max_frames=n)
        score = ssim_background(inp_frames, out_frames, gt_masks)
        if score is not None:
            print(f"  SSIM (background)     : {score:.4f}")

    print()
    print(sep)
    print()

    # ── save log entry ────────────────────────────────────────────────────────
    run_id = datetime.datetime.now().strftime("RUN-%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "benchmark_log.txt"
    with open(log_path, "a") as lf:
        lf.write(f"{'='*52}\n")
        lf.write(f"  {run_id}\n")
        lf.write(f"{'='*52}\n")
        lf.write(f"  GT   : {args.gt}\n")
        lf.write(f"  Pred : {args.pred}\n")
        if args.pred_offset_seconds:
            lf.write(f"  Pred offset : {args.pred_offset_seconds}s\n")
        if args.max_seconds:
            lf.write(f"  Max seconds : {args.max_seconds}s\n")
        lf.write(f"  Frames compared : {n}  ({n/fps:.1f}s at {fps:.1f} FPS)\n")
        lf.write("\n")
        lf.write("  DETECTION & MASK QUALITY\n")
        lf.write(f"  Mean IoU              : {ious.mean():.4f}\n")
        lf.write(f"  Pixel Coverage        : {pixel_coverage*100:.1f}%\n")
        lf.write(f"  Mean Recall           : {recs.mean():.4f}\n")
        lf.write(f"  Mean Precision        : {precs.mean():.4f}\n")
        lf.write(f"  Mean F1               : {f1s.mean():.4f}\n")
        lf.write(f"  mAP@50                : {map50:.4f}\n")
        lf.write("\n")
        lf.write("  TEMPORAL STABILITY\n")
        lf.write(f"  Flickering Events     : {flickers}\n")
        lf.write(f"  Temporal IoU Std Dev  : {iou_std:.4f}\n")
        lf.write(f"  MOTA (simplified)     : {mota:.4f}\n")
        if n_person:
            lf.write(f"  Missed detections     : {fn} / {n_person} ({fn/n_person*100:.1f}%)\n")
        lf.write(f"  False alarms          : {fp} frames\n")
        if args.log:
            fps_vals = parse_log(args.log)
            lf.write("\n")
            lf.write("  PERFORMANCE\n")
            if fps_vals is not None:
                split = max(1, len(fps_vals) // 10)
                start = fps_vals[:split].mean()
                end   = fps_vals[-split:].mean()
                lf.write(f"  Mean Seg FPS          : {fps_vals.mean():.1f}\n")
                lf.write(f"  Min / Max Seg FPS     : {fps_vals.min():.1f} / {fps_vals.max():.1f}\n")
                lf.write(f"  Thermal degradation   : {start:.1f} -> {end:.1f} FPS\n")
            else:
                lf.write("  No FPS values found in log\n")
        lf.write("\n")
    print(f"Log entry saved  -> {log_path}  (ID: {run_id})")

    # ── comparison image ──────────────────────────────────────────────────────
    out_img = run_dir / "mask_comparison.png"
    total_frames = len(gt_masks)
    sample_indices = [int(total_frames * p) for p in (0.05, 0.25, 0.5, 0.75, 0.95)]
    TH, TW = 480, 270
    panels = []
    for idx in sample_indices:
        gt_f   = gt_masks[idx]
        pred_f = pred_masks[idx]
        def _lb(mask):
            sh, sw = mask.shape
            scale = min(TW/sw, TH/sh)
            nw, nh = int(sw*scale), int(sh*scale)
            pl, pt = (TW-nw)//2, (TH-nh)//2
            buf = np.zeros((TH, TW), dtype=bool)
            r = cv2.resize(mask.astype(np.uint8)*255, (nw, nh), interpolation=cv2.INTER_AREA)
            buf[pt:pt+nh, pl:pl+nw] = r > 127
            return buf
        gt_lb   = _lb(gt_f)
        pred_lb = _lb(pred_f)
        overlap = gt_lb & pred_lb
        overlay = np.zeros((TH, TW, 3), dtype=np.uint8)
        overlay[gt_lb & ~overlap]   = (0, 180, 0)
        overlay[pred_lb & ~overlap] = (0, 0, 200)
        overlay[overlap]            = (0, 200, 200)
        def _label(img, txt):
            out = img.copy()
            cv2.putText(out, txt, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            return out
        gt_img   = (gt_lb.astype(np.uint8)*255)[:, :, None].repeat(3, axis=2)
        pred_img = (pred_lb.astype(np.uint8)*255)[:, :, None].repeat(3, axis=2)
        sec = idx / fps
        panels.append(np.hstack([
            _label(gt_img,   f"GT f{idx} ({sec:.1f}s)"),
            _label(pred_img, "pred"),
            _label(overlay,  "G=GT R=pred Y=both"),
        ]))
    divider = np.full((4, panels[0].shape[1], 3), 60, dtype=np.uint8)
    result = panels[0]
    for p in panels[1:]:
        result = np.vstack([result, divider, p])
    cv2.imwrite(str(out_img), result)
    print(f"Comparison image saved -> {out_img}")


if __name__ == "__main__":
    main()
