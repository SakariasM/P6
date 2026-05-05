"""
Ground truth mask generator using SAM 2 + Grounding DINO person detector.

Usage:
  python generate_gt_sam2.py --input video.mp4 --output gt_mask.mp4
  python generate_gt_sam2.py --input video.mp4 --output gt_mask.mp4 --debug  # side-by-side preview

SAM 2 tracks every person detected across ALL frames — people entering mid-video are caught.
Output is a binary mask video: person = white, background = black.

NOTE: Very slow on CPU (hours for a short clip). Requires CUDA for practical use.
"""

import argparse
import contextlib
import os
import pickle
import signal
import sys
import cv2
import numpy as np
import torch
from tqdm import tqdm

CHECKPOINT = os.path.join(os.path.dirname(__file__), "sam2_repo/checkpoints/sam2.1_hiera_large.pt")
CONFIG         = "configs/sam2.1/sam2.1_hiera_l.yaml"
DETECTOR_MODEL = "IDEA-Research/grounding-dino-base"
DETECTOR_TEXT  = "person."
DETECTOR_CONF  = 0.35   # box + text threshold
SCAN_EVERY     = 1      # scan every N frames to catch late arrivals
IOU_THRESHOLD  = 0.5    # overlap above this = same person already tracked

# ── interrupt handling ─────────────────────────────────────────────────────────
_interrupted = False

def _handle_sigint(sig, frame):
    global _interrupted
    _interrupted = True
    print("\n[info] Ctrl+C received — stopping after this frame…")

signal.signal(signal.SIGINT, _handle_sigint)


def box_iou(box, others):
    """Compute IoU between one box [x1,y1,x2,y2] and an array of boxes (N,4)."""
    if len(others) == 0:
        return np.zeros(0)
    xi1 = np.maximum(box[0], others[:, 0])
    yi1 = np.maximum(box[1], others[:, 1])
    xi2 = np.minimum(box[2], others[:, 2])
    yi2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)
    area_box    = (box[2] - box[0]) * (box[3] - box[1])
    area_others = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = area_box + area_others - inter
    return inter / np.maximum(union, 1e-6)


def scan_all_frames(frames_dir, total_frames, conf, device, scan_debug_path=None):
    """
    Run Grounding DINO on every Nth frame and collect (frame_idx, box) for each
    unique person appearance — deduplicating against already-known boxes.
    Returns list of (frame_idx, box) sorted by frame_idx.
    """
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from PIL import Image

    print(f"[info] Loading Grounding DINO ({DETECTOR_MODEL})…")
    processor = AutoProcessor.from_pretrained(DETECTOR_MODEL)
    detector  = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR_MODEL).to(device)
    detector.eval()

    known_boxes = []
    prompts = []
    scan_frames_with_boxes = []  # for scan debug video: (frame, [(box, is_new)])

    scan_indices = list(range(0, total_frames, SCAN_EVERY))
    for i in tqdm(scan_indices, desc="[info] Scanning frames", unit="frame"):
        if _interrupted:
            break
        frame = cv2.imread(os.path.join(frames_dir, f"{i:06d}.jpg"))
        if frame is None:
            print(f"[warning] Could not read frame {i}, skipping.")
            continue
        h, w  = frame.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        inputs = processor(images=image, text=DETECTOR_TEXT, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = detector(**inputs)

        results    = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=conf, text_threshold=conf,
            target_sizes=[(h, w)]
        )
        detections = results[0]["boxes"].cpu().numpy()

        frame_boxes = []
        for det in detections:
            if len(known_boxes) == 0:
                known_boxes.append(det)
                prompts.append((i, det))
                frame_boxes.append((det, True))
            else:
                ious = box_iou(det, np.array(known_boxes))
                is_new = ious.max() < IOU_THRESHOLD
                if is_new:
                    known_boxes.append(det)
                    prompts.append((i, det))
                frame_boxes.append((det, is_new))

        if scan_debug_path and frame_boxes:
            scan_frames_with_boxes.append((frame, frame_boxes))

    del detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not prompts:
        return []

    print(f"[info] Found {len(prompts)} unique person appearance(s) across video.")

    if scan_debug_path and scan_frames_with_boxes:
        print(f"[info] Writing scan debug video…")
        fh, fw = scan_frames_with_boxes[0][0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(scan_debug_path, fourcc, 5.0, (fw, fh))
        if not writer.isOpened():
            writer = cv2.VideoWriter(scan_debug_path, cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (fw, fh))
        for frame, boxes in scan_frames_with_boxes:
            vis = frame.copy()
            for box, is_new in boxes:
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0) if is_new else (0, 165, 255)  # green=new, orange=duplicate
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
                label = "NEW" if is_new else "DUP"
                cv2.putText(vis, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            writer.release() if False else None
            writer.write(vis)
        writer.release()
        print(f"[info] Scan debug video saved: {scan_debug_path}")

    return prompts


def extract_frames(video_path, frames_dir):
    """Extract all frames as zero-padded JPEGs for SAM 2."""
    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[info] {width}×{height} @ {fps:.1f} FPS — {total} frames")
    print(f"[info] Extracting frames…")

    idx = 0
    pbar = tqdm(total=total, unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_dir, f"{idx:06d}.jpg"), frame)
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return idx, fps, width, height


def count_frames_in_dir(frames_dir):
    return len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])


def main():
    ap = argparse.ArgumentParser(description="SAM 2 ground truth mask generator")
    ap.add_argument("-i", "--input",  required=True, help="Input video path")
    ap.add_argument("-o", "--output", required=True, help="Output mask video path")
    ap.add_argument("--debug", action="store_true", help="Also write side-by-side debug video")
    ap.add_argument("--scan-debug", action="store_true", help="Write DINO detection debug video (green=new, orange=duplicate)")
    ap.add_argument("--conf", type=float, default=DETECTOR_CONF, help="Detection confidence threshold")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}")
        sys.exit(1)
    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: SAM2 checkpoint not found: {CHECKPOINT}")
        sys.exit(1)

    # ── paths ──────────────────────────────────────────────────────────────────
    base             = os.path.splitext(args.output)[0]
    frames_dir       = base + "_frames"
    masks_dir        = base + "_masks"
    prompts_path     = base + "_prompts.pkl"
    debug_path       = base + "_debug.mp4"
    scan_debug_path  = (base + "_scan_debug.mp4") if args.scan_debug else None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── extract frames ─────────────────────────────────────────────────────────
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    if os.listdir(frames_dir):
        print(f"[info] Reusing existing frames in {frames_dir}")
        cap = cv2.VideoCapture(args.input)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        total_frames = count_frames_in_dir(frames_dir)
    else:
        total_frames, fps, w, h = extract_frames(args.input, frames_dir)

    # ── scan for people ────────────────────────────────────────────────────────
    if os.path.exists(prompts_path):
        print(f"[info] Reusing cached scan results.")
        with open(prompts_path, "rb") as f:
            prompts = pickle.load(f)
        print(f"[info] Found {len(prompts)} unique person appearance(s) across video.")
    else:
        prompts = scan_all_frames(frames_dir, total_frames, args.conf, device, scan_debug_path)
        if _interrupted:
            print(f"[paused] Interrupted during scan — re-run to restart.")
            sys.exit(0)
        if not prompts:
            print(f"[error] No people detected in video. Try lowering --conf.")
            sys.exit(1)
        with open(prompts_path, "wb") as f:
            pickle.dump(prompts, f)
        print(f"[info] Scan results cached to {prompts_path}")

    # ── load SAM 2 ─────────────────────────────────────────────────────────────
    if device == "cpu":
        print("[warning] CUDA not available — running on CPU, this will be slow.")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("[setup] Loading SAM 2…")
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT, device=device)
    if device == "cpu":
        predictor = predictor.float()
        def _cast_inputs_to_float(module, args, kwargs):
            args = tuple(a.float() if isinstance(a, torch.Tensor) else a for a in args)
            kwargs = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
            return args, kwargs
        for module in predictor.modules():
            if isinstance(module, torch.nn.Linear):
                module.register_forward_pre_hook(_cast_inputs_to_float, with_kwargs=True)

    # ── run SAM 2 propagation ──────────────────────────────────────────────────
    MEM_KEEP = 10               # keep last N frames in SAM2 memory (num_maskmem=7 + buffer)
    EMPTY_FRAMES_BEFORE_DROP = 60  # drop object after N consecutive empty frames

    with torch.inference_mode(), (torch.autocast(device, dtype=dtype) if device == "cuda" else contextlib.nullcontext()):
        inference_state = predictor.init_state(video_path=frames_dir, offload_video_to_cpu=True, offload_state_to_cpu=True)

        for obj_id, (frame_idx, box) in enumerate(prompts):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            predictor.add_new_points_or_box(
                inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box,
                points=np.array([[cx, cy]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )

        obj_start_frames = {obj_id: frame_idx for obj_id, (frame_idx, _) in enumerate(prompts)}
        obj_empty_streak = {obj_id: 0 for obj_id in range(len(prompts))}
        obj_dead = set()

        print(f"[info] Propagating {len(prompts)} tracked person(s)… (Ctrl+C to stop)")
        for frame_idx, obj_ids, mask_logits in tqdm(
            predictor.propagate_in_video(inference_state, obj_start_frames=obj_start_frames),
            total=total_frames, unit="frame"
        ):
            masks_per_obj = mask_logits.squeeze(1) > 0.0  # (N, H, W)

            for obj_idx in range(len(prompts)):
                if obj_idx in obj_dead:
                    continue
                if frame_idx < obj_start_frames.get(obj_idx, 0):
                    continue
                if obj_idx < masks_per_obj.shape[0] and masks_per_obj[obj_idx].any():
                    obj_empty_streak[obj_idx] = 0
                else:
                    obj_empty_streak[obj_idx] += 1
                    if obj_empty_streak[obj_idx] >= EMPTY_FRAMES_BEFORE_DROP:
                        obj_dead.add(obj_idx)
                        obj_start_frames[obj_idx] = total_frames

            combined = masks_per_obj.any(dim=0).cpu().numpy().astype(np.uint8) * 255
            cv2.imwrite(os.path.join(masks_dir, f"{frame_idx:06d}.png"), combined)

            if frame_idx > MEM_KEEP:
                old_idx = frame_idx - MEM_KEEP
                for obj_output in inference_state["output_dict_per_obj"].values():
                    obj_output["non_cond_frame_outputs"].pop(old_idx, None)
                for obj_tracked in inference_state["frames_tracked_per_obj"].values():
                    obj_tracked.pop(old_idx, None)

            if _interrupted:
                break

    if _interrupted:
        print(f"[stopped] Masks saved up to frame {frame_idx}. Re-run to start over.")
        sys.exit(0)

    # ── write output ───────────────────────────────────────────────────────────
    _write_output(frames_dir, masks_dir, total_frames, args, fps, w, h, debug_path)
    print(f"[done] Mask video saved: {args.output}")


def _write_output(frames_dir, masks_dir, total_frames, args, fps, w, h, debug_path):
    print("[info] Writing output video…")
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer_mask = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
    if not writer_mask.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_mask = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
    writer_debug = None
    if args.debug:
        writer_debug = cv2.VideoWriter(debug_path, fourcc, fps, (w * 2, h))

    for i in tqdm(range(total_frames), unit="frame"):
        frame = cv2.imread(os.path.join(frames_dir, f"{i:06d}.jpg"))
        mask_path = os.path.join(masks_dir, f"{i:06d}.png")
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else np.zeros((h, w), dtype=np.uint8)
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
