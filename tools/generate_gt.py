"""
Ground truth mask generator using YOLO11x-seg person segmentation.

Usage:
  python generate_gt.py --input video.mp4 --output gt_mask.mp4
  python generate_gt.py --input video.mp4 --output gt_mask.mp4 --debug  # side-by-side preview

Press Ctrl+C at any time to pause. Re-run the same command to resume.

Output is a binary mask video: person = white, background = black.
"""

import argparse
import contextlib
import os
import pickle
import signal
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent

DETECTOR_MODEL   = "yolo11x-seg.pt"  # largest/most accurate YOLO seg model
DETECTOR_CONF    = 0.4               # YOLO confidence threshold
CHECKPOINT_EVERY = 50                # save progress every N frames

# ── interrupt handling ─────────────────────────────────────────────────────────
_interrupted = False

def _handle_sigint(sig, frame):
    global _interrupted
    _interrupted = True
    print("\n[info] Ctrl+C received — will save checkpoint after this frame and exit…")

signal.signal(signal.SIGINT, _handle_sigint)


def extract_frames(video_path, frames_dir):
    """Extract all frames as JPEGs."""
    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[info] {width}×{height} @ {fps:.1f} FPS — {total} frames")
    print(f"[info] Extracting frames…")

    frames = []
    idx = 0
    pbar = tqdm(total=total, unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        path = os.path.join(frames_dir, f"{idx:06d}.jpg")
        cv2.imwrite(path, frame)
        frames.append(frame)
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return frames, fps, width, height


def load_frames_from_dir(frames_dir):
    """Re-load frames from an existing extracted directory."""
    paths = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    frames = []
    for p in tqdm(paths, unit="frame", desc="[info] Loading cached frames"):
        frames.append(cv2.imread(os.path.join(frames_dir, p)))
    return frames


def main():
    ap = argparse.ArgumentParser(description="YOLO11x-seg ground truth mask generator")
    ap.add_argument("-i", "--input",  required=True, help="Input video path")
    ap.add_argument("-o", "--output", required=True, help="Output mask video path")
    ap.add_argument("--debug", action="store_true", help="Also write side-by-side debug video")
    ap.add_argument("--conf", type=float, default=DETECTOR_CONF, help="YOLO detection confidence")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}")
        sys.exit(1)

    # ── paths ──────────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    run_name = output_path.stem
    cache_root = ROOT / "cache" / "gt-gen" / run_name
    run_root = ROOT / "runs" / "gt-gen" / run_name
    frames_dir = cache_root / "frames"
    ckpt_path = cache_root / "progress.pkl"
    debug_path = run_root / "debug.mp4"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    # ── resume from checkpoint if available ───────────────────────────────────
    mask_by_frame = {}
    start_frame   = 0
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            saved = pickle.load(f)
        mask_by_frame = saved["masks"]
        total_frames  = saved["total"]
        fps           = saved["fps"]
        w, h          = saved["size"]
        start_frame   = len(mask_by_frame)
        print(f"[resume] Loaded checkpoint: {start_frame}/{total_frames} frames processed.")

        if start_frame >= total_frames:
            print("[resume] All frames already processed — skipping to output.")
            frames = load_frames_from_dir(frames_dir)
            _write_output(frames, mask_by_frame, args, fps, w, h, debug_path)
            os.remove(ckpt_path)
            print(f"[done] Mask video saved: {output_path}")
            return
        else:
            frames = load_frames_from_dir(frames_dir)
    else:
        # ── fresh run ──────────────────────────────────────────────────────────
        if os.listdir(frames_dir):
            print(f"[info] Reusing existing frames in {frames_dir}")
            cap = cv2.VideoCapture(args.input)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            frames = load_frames_from_dir(frames_dir)
        else:
            frames, fps, w, h = extract_frames(args.input, frames_dir)

        total_frames = len(frames)

    # ── load YOLO ──────────────────────────────────────────────────────────────
    from ultralytics import YOLO
    print(f"[info] Loading {DETECTOR_MODEL}…")
    model = YOLO(DETECTOR_MODEL)

    # ── run segmentation per frame ─────────────────────────────────────────────
    print(f"[info] Segmenting {total_frames} frames… (Ctrl+C to pause)")
    for i in tqdm(range(start_frame, total_frames), unit="frame"):
        results = model(frames[i], classes=[0], conf=args.conf, verbose=False)
        mask = np.zeros((h, w), dtype=np.uint8)
        if results[0].masks is not None:
            for m in results[0].masks.data:
                seg = (m.cpu().numpy() * 255).astype(np.uint8)
                seg = cv2.resize(seg, (w, h))
                mask = np.maximum(mask, seg)
        mask_by_frame[i] = mask

        if i % CHECKPOINT_EVERY == 0:
            _save_checkpoint(ckpt_path, mask_by_frame, total_frames, fps, w, h)

        if _interrupted:
            _save_checkpoint(ckpt_path, mask_by_frame, total_frames, fps, w, h)
            print(f"[paused] Progress saved ({len(mask_by_frame)}/{total_frames} frames).")
            print(f"[paused] Re-run the same command to resume.")
            sys.exit(0)

    # ── write output ───────────────────────────────────────────────────────────
    _write_output(frames, mask_by_frame, args, fps, w, h, debug_path)

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print(f"[done] Mask video saved: {output_path}")


def _save_checkpoint(path, masks, total, fps, w, h):
    with open(path, "wb") as f:
        pickle.dump({"masks": masks, "total": total, "fps": fps, "size": (w, h)}, f)


def _write_output(frames, mask_by_frame, args, fps, w, h, debug_path):
    print("[info] Writing output video…")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_mask  = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))
    writer_debug = None
    if args.debug:
        writer_debug = cv2.VideoWriter(str(debug_path), fourcc, fps, (w * 2, h))

    for i, frame in enumerate(tqdm(frames, unit="frame")):
        mask_gray = mask_by_frame.get(i, np.zeros((h, w), dtype=np.uint8))
        mask_bgr  = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
        writer_mask.write(mask_bgr)

        if writer_debug is not None:
            overlay = frame.copy()
            overlay[mask_gray > 0] = (0, 200, 0)
            blended = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
            writer_debug.write(np.hstack([blended, mask_bgr]))

        if _interrupted:
            print("[info] Interrupted during output writing — partial video written.")
            break

    writer_mask.release()
    if writer_debug:
        writer_debug.release()
        if not _interrupted:
            print(f"[done] Debug video saved: {debug_path}")


if __name__ == "__main__":
    main()
