"""
Invisible human — TFLite selfie segmentation + background buffer
Optimised for Raspberry Pi 5 (4 GB):

Pipeline (3 processes, zero-copy shared memory):
  • FrameGrabber  — reads webcam, writes into shared-memory ring buffer
  • SegWorker     — runs selfie_segmenter.tflite via TFLite runtime
  • Main process  — compositing, display, keyboard

Dependencies: pip install ai-edge-litert opencv-python numpy

Usage:
  Live camera:        python3 live_mask.py
  Test on video:      python3 live_mask.py --input video.mp4
  Record output:      python3 live_mask.py --output recording.mp4
  Test + save:        python3 live_mask.py --input test.mp4 --output result.mp4

Keys:  q — quit   d — toggle debug mask overlay   r — toggle recording (live mode)
"""

import argparse
import cv2
import os
import sys
import time
import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory

# ── config ────────────────────────────────────────────────────────────────────
SOURCE         = 0                # webcam index, or path to a video file
MASK_THRESHOLD = 0.7              # segmentation confidence (0–1, lower = more aggressive)
MASK_DILATE    = 10               # expand mask outward to cover person edges
MASK_BLUR      = 1                # feather mask edges for smooth blending (must be odd)
BG_LEARN       = 0.02             # background learning rate (higher = adapts faster)
MASK_EDGE_PAD  = 50               # if mask is within this many px of top/bottom, extend to edge
MODEL          = "student_seg_320"    # model name — auto-downloads and exports if missing
                                  # use "selfie_segmenter" for the lightweight TFLite model
MODEL_IMGSZ    = 320              # inference resolution used when exporting YOLO models
MODEL_PATH     = f"models/{MODEL}/{MODEL}.tflite"
# ─────────────────────────────────────────────────────────────────────────────


def ensure_model():
    """Auto-download and export MODEL_PATH from the corresponding .pt if missing.
    Models are stored in models/{model_name}/ and temp export files are cleaned up."""
    if os.path.exists(MODEL_PATH):
        return

    import shutil

    model_name = os.path.basename(os.path.dirname(MODEL_PATH))
    out_dir    = os.path.dirname(MODEL_PATH)
    os.makedirs(out_dir, exist_ok=True)

    # selfie_segmenter is not a YOLO model — look for it in common locations
    if MODEL == "selfie_segmenter":
        search_paths = [
            os.path.expanduser("~/Downloads/selfie_segmenter.tflite"),
            "selfie_segmenter.tflite",
        ]
        for src in search_paths:
            if os.path.exists(src):
                shutil.copy(src, MODEL_PATH)
                print(f"[setup] Copied selfie_segmenter.tflite → {MODEL_PATH}")
                return
        print(f"\n[setup] WARNING: selfie_segmenter.tflite not found.")
        print(f"[setup] Download it from MediaPipe and place it in ~/Downloads/ or the project folder.")
        print(f"[setup] Exiting.\n")
        sys.exit(1)

    pt_path         = f"{model_name}.pt"
    tflite_filename = os.path.basename(MODEL_PATH)
    tmp_saved_model = f"{model_name}_saved_model"
    tmp_tflite      = os.path.join(tmp_saved_model, tflite_filename)
    tmp_onnx        = f"{model_name}.onnx"

    print(f"[setup] Model not found: {MODEL_PATH}")
    print(f"[setup] Downloading {pt_path}…")
    try:
        from ultralytics import YOLO
        model = YOLO(pt_path)  # triggers auto-download with progress bar if needed

        print(f"[setup] Exporting to TFLite at {MODEL_IMGSZ}×{MODEL_IMGSZ} — this takes ~1 minute…")
        model.export(format="tflite", imgsz=MODEL_IMGSZ)

        print(f"[setup] Moving model to {out_dir}/…")
        shutil.move(tmp_tflite, MODEL_PATH)

        print(f"[setup] Cleaning up temporary export files…")
        shutil.rmtree(tmp_saved_model, ignore_errors=True)
        if os.path.exists(tmp_onnx):
            os.remove(tmp_onnx)
        if os.path.exists(pt_path):
            os.remove(pt_path)
        for f in os.listdir("."):
            if f.startswith("calibration_image") and f.endswith(".npy"):
                os.remove(f)

        print(f"[setup] Ready — {MODEL_PATH}")
    except Exception as e:
        # Clean up any partial export files
        shutil.rmtree(tmp_saved_model, ignore_errors=True)
        if os.path.exists(tmp_onnx):
            os.remove(tmp_onnx)
        print(f"\n[setup] WARNING: Could not export '{model_name}'.")
        print(f"[setup] This is likely because TensorFlow is not supported on this Python version.")
        print(f"[setup] Export the model on a PC and copy models/ to this machine:")
        print(f"[setup]   scp -r models/ user@<this-machine-ip>:{os.getcwd()}/")
        print(f"[setup] Exiting.\n")
        sys.exit(1)


# ── shared-memory helpers ─────────────────────────────────────────────────────

def _create_shm(name: str, size: int):
    """Create (or recreate) a named shared-memory block."""
    try:
        old = shared_memory.SharedMemory(name=name, create=False)
        old.close()
        old.unlink()
    except FileNotFoundError:
        pass
    return shared_memory.SharedMemory(name=name, create=True, size=size)


def _np_from_shm(shm, shape, dtype=np.uint8):
    """Return a numpy array backed by the shared-memory buffer."""
    return np.ndarray(shape, dtype=dtype, buffer=shm.buf)


# ── process: frame grabber ────────────────────────────────────────────────────

def frame_grabber(source, shm_name, shape, frame_lock, stop_event, cam_fps):
    """Read camera frames into shared memory, rate-limited to camera FPS."""
    h, w = shape[:2]
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        stop_event.set()
        return

    shm = shared_memory.SharedMemory(name=shm_name, create=False)
    buf = _np_from_shm(shm, shape)

    frame_interval = 1.0 / max(cam_fps, 1.0)

    while not stop_event.is_set():
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break
        if frame.shape[:2] != shape[:2]:
            frame = cv2.resize(frame, (shape[1], shape[0]))
        with frame_lock:
            np.copyto(buf, frame)

        # Rate-limit to camera FPS
        elapsed = time.time() - t_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    shm.close()


# ── segmentation worker helpers ────────────────────────────────────────────────

def _extend_mask_edges(mask, h, w):
    """Extend mask to frame edges if the person is close to the top/bottom border."""
    if not mask.any():
        return mask
    rows = np.where(mask.max(axis=1) > 0)[0]
    if len(rows) == 0:
        return mask
    top_row, bot_row = rows[0], rows[-1]
    if top_row < MASK_EDGE_PAD:
        mask[:top_row + 1, :] = np.maximum(
            mask[:top_row + 1, :], mask[top_row:top_row + 1, :])
    if bot_row > (h - 1 - MASK_EDGE_PAD):
        mask[bot_row:, :] = np.maximum(
            mask[bot_row:, :], mask[bot_row:bot_row + 1, :])
    return mask


def _tflite_loop(model_file, frame_buf, mask_buf, frame_shape,
                 frame_lock, mask_lock, mask_ready,
                 stop_event, seg_fps_val, det_count, prob_stats, frame_interval):
    """Inference loop for TFLite models (selfie_segmenter and YOLO .tflite)."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

    print(f"[worker] loading TFLite model: {model_file}")
    interp = Interpreter(model_path=model_file, num_threads=2)
    interp.allocate_tensors()

    inp_detail  = interp.get_input_details()[0]
    out_details = interp.get_output_details()
    inp_idx = inp_detail["index"]
    inp_shape = inp_detail["shape"]

    # Detect NCHW (PyTorch-exported) vs NHWC (TFLite standard) input layout
    if inp_shape[1] <= 4 and inp_shape[2] > 4:
        # NCHW: (1, 3, H, W)
        nchw = True
        _, _, model_h, model_w = inp_shape
    else:
        # NHWC: (1, H, W, 3)
        nchw = False
        _, model_h, model_w, _ = inp_shape

    # Detect model type by output shapes:
    #   Selfie segmenter / student: single output (1, H, W, 1) or (1, 1, H, W)
    #   YOLO post-NMS:    (1, n_dets, 38) + (1, H, W, 32)
    #   YOLO raw:         (1, 116, 2100)  + (1, H, W, 32)
    det_idx, proto_idx = None, None
    selfie_idx = None
    det_shape = None
    proto_h = proto_w = n_proto = None

    if len(out_details) == 1 and len(out_details[0]["shape"]) == 4:
        # Single mask output: (1, H, W, 1) or (1, 1, H, W)
        selfie_idx = out_details[0]["index"]
        fmt = "binary mask (NCHW)" if nchw else "selfie_segmenter"
    else:
        for od in out_details:
            if len(od["shape"]) == 3:
                det_idx   = od["index"]
                det_shape = od["shape"]
            elif len(od["shape"]) == 4:
                proto_idx = od["index"]
                _, proto_h, proto_w, n_proto = od["shape"]
        if det_idx is None or proto_idx is None:
            raise RuntimeError(f"Unexpected output shapes: {[od['shape'] for od in out_details]}")
        fmt = f"post-NMS (dets×features={det_shape[1]}×{det_shape[2]})"

    if selfie_idx is not None:
        print(f"[worker] model loaded OK — input {model_w}×{model_h}, format: {fmt}")
    else:
        print(f"[worker] model loaded OK — input {model_w}×{model_h}, "
              f"prototypes {proto_w}×{proto_h}×{n_proto}, format: {fmt}")

    h, w = frame_shape[:2]
    mask_res_h = model_h if selfie_idx is not None else proto_h
    mask_res_w = model_w if selfie_idx is not None else proto_w
    scale        = mask_res_h / h
    small_dilate = max(3, int(MASK_DILATE * scale) | 1)
    small_blur   = max(3, int(MASK_BLUR   * scale) | 1)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_dilate, small_dilate))
    print(f"[worker] post-process at {mask_res_w}×{mask_res_h}: "
          f"dilate {small_dilate}×{small_dilate}, blur {small_blur}×{small_blur}")

    if nchw:
        input_data = np.empty((1, 3, model_h, model_w), dtype=np.float32)
    else:
        input_data = np.empty((1, model_h, model_w, 3), dtype=np.float32)
    proto_flat_buf = None if selfie_idx is not None else np.empty((proto_h * proto_w, n_proto), dtype=np.float32)

    print("[worker] entering inference loop")
    prev_time = time.time()
    while not stop_event.is_set():
        t_start = time.time()

        with frame_lock:
            frame = frame_buf.copy()

        # Preprocess: resize → RGB → normalise
        small = cv2.resize(frame, (model_w, model_h))
        cv2.cvtColor(small, cv2.COLOR_BGR2RGB, dst=small)
        if nchw:
            # (H, W, 3) → (3, H, W)
            np.multiply(small.transpose(2, 0, 1), 1.0 / 255.0,
                        out=input_data[0], casting="unsafe")
        else:
            np.multiply(small, 1.0 / 255.0, out=input_data[0], casting="unsafe")

        interp.set_tensor(inp_idx, input_data)
        interp.invoke()

        if selfie_idx is not None:
            # Direct mask output: (1, H, W, 1) or (1, 1, H, W)
            out = interp.get_tensor(selfie_idx)
            if nchw:
                raw = out[0, 0, :, :]          # (1, 1, H, W) → (H, W)
            else:
                raw = out[0, :, :, 0]           # (1, H, W, 1) → (H, W)
            small_mask = (raw > MASK_THRESHOLD).astype(np.uint8) * 255
        else:
            detections = interp.get_tensor(det_idx)[0]
            prototypes = interp.get_tensor(proto_idx)[0]
            small_mask = np.zeros((proto_h, proto_w), dtype=np.uint8)
            valid  = (detections[:, 4] > MASK_THRESHOLD) & (detections[:, 5].astype(int) == 0)
            coeffs = detections[valid, 6:]
            det_count.value = int(valid.sum())
            if valid.any():
                np.copyto(proto_flat_buf, prototypes.reshape(-1, n_proto))
                logits       = coeffs @ proto_flat_buf.T
                person_masks = 1.0 / (1.0 + np.exp(-logits))
                combined     = person_masks.max(axis=0).reshape(proto_h, proto_w)
                small_mask   = (combined > 0.5).astype(np.uint8) * 255

        if small_mask.any():
            small_mask = cv2.dilate(small_mask, dilate_kernel, iterations=2)
            small_mask = cv2.GaussianBlur(small_mask, (small_blur, small_blur), 0)

        mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = _extend_mask_edges(mask, h, w)

        with mask_lock:
            np.copyto(mask_buf, mask)
        mask_ready.set()

        now = time.time()
        seg_fps_val.value = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now

        elapsed = time.time() - t_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def _pt_loop(model_file, frame_buf, mask_buf, frame_shape,
             frame_lock, mask_lock, mask_ready,
             stop_event, seg_fps_val, det_count, prob_stats, frame_interval):
    """Inference loop for .pt models — auto-detects student vs ultralytics YOLO."""
    import torch, sys, os
    ckpt = torch.load(model_file, map_location='cpu', weights_only=False)
    is_student = isinstance(ckpt, dict) and 'model_state_dict' in ckpt

    h, w = frame_shape[:2]
    dilate_k      = max(3, MASK_DILATE | 1)
    blur_k        = max(3, MASK_BLUR   | 1)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))

    if is_student:
        # Add project root to path so student_model / attention can be imported
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from student.model import StudentSegmentation
        args         = ckpt['args']
        teacher_ch   = ckpt.get('teacher_channels', [128, 128, 256])
        model = StudentSegmentation(
            base_channels=args['base_channels'],
            depth=args['depth'],
            teacher_channels=teacher_ch,
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f"[worker] Student model loaded OK — base_ch={args['base_channels']}, "
              f"depth={args['depth']}, imgsz {MODEL_IMGSZ}, "
              f"dilate {dilate_k}×{dilate_k}, blur {blur_k}×{blur_k}")
    else:
        from ultralytics import YOLO
        model = YOLO(model_file)
        print(f"[worker] YOLO model loaded OK — imgsz {MODEL_IMGSZ}, "
              f"dilate {dilate_k}×{dilate_k}, blur {blur_k}×{blur_k}")

    print("[worker] entering inference loop")
    prev_time = time.time()
    while not stop_event.is_set():
        t_start = time.time()

        with frame_lock:
            frame = frame_buf.copy()

        if is_student:
            small = cv2.resize(frame, (MODEL_IMGSZ, MODEL_IMGSZ))
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp   = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
            with torch.no_grad():
                mask_out, _ = model(inp)                     # [1, 1, H, W] sigmoid
            prob = mask_out[0, 0].numpy()                    # (H, W)
            prob_stats[0] = float(prob.min())
            prob_stats[1] = float(prob.max())
            prob_stats[2] = float(prob.mean())
            small_mask = (prob > MASK_THRESHOLD).astype(np.uint8) * 255
            mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            det_count.value = 1 if small_mask.any() else 0
        else:
            results = model(frame, imgsz=MODEL_IMGSZ, verbose=False, classes=[0])
            r = results[0]
            det_count.value = len(r.boxes) if r.boxes is not None else 0
            if r.masks is not None:
                masks_np   = r.masks.data.cpu().numpy()      # (N, H, W) float32
                combined   = masks_np.max(axis=0)
                small_mask = (combined > 0.5).astype(np.uint8) * 255
                mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                mask = np.zeros((h, w), dtype=np.uint8)

        if mask.any():
            mask = cv2.dilate(mask, dilate_kernel, iterations=2)
            mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

        mask = _extend_mask_edges(mask, h, w)

        with mask_lock:
            np.copyto(mask_buf, mask)
        mask_ready.set()

        now = time.time()
        seg_fps_val.value = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now

        elapsed = time.time() - t_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# ── process: segmentation worker ──────────────────────────────────────────────

def seg_worker(frame_shm_name, mask_shm_name, frame_shape,
               frame_lock, mask_lock, mask_ready,
               stop_event, seg_fps_val, det_count, prob_stats, cam_fps):
    """Dispatch to the correct inference backend based on model file extension."""
    model_file     = os.path.join(os.path.dirname(__file__) or ".", MODEL_PATH)
    frame_interval = 1.0 / max(cam_fps, 1.0)
    h, w           = frame_shape[:2]

    shm_frame = shared_memory.SharedMemory(name=frame_shm_name, create=False)
    shm_mask  = shared_memory.SharedMemory(name=mask_shm_name, create=False)
    frame_buf = _np_from_shm(shm_frame, frame_shape)
    mask_buf  = _np_from_shm(shm_mask, (h, w))

    loop_fn = _pt_loop if model_file.endswith(".pt") else _tflite_loop

    try:
        loop_fn(model_file, frame_buf, mask_buf, frame_shape,
                frame_lock, mask_lock, mask_ready,
                stop_event, seg_fps_val, det_count, prob_stats, frame_interval)
    except Exception as exc:
        print(f"[worker] CRASHED: {exc}")
        import traceback; traceback.print_exc()
        stop_event.set()
    finally:
        shm_frame.close()
        shm_mask.close()


# ── main process: compositing + display ───────────────────────────────────────

def _parse_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        description="Invisible human — selfie segmentation + background replacement")
    ap.add_argument("-i", "--input", default=None,
                    help="Path to input video file. Omit to use live camera.")
    ap.add_argument("-o", "--output", default=None,
                    help="Path to save output video (e.g. output.mp4). "
                         "In live mode you can also press 'r' to toggle recording.")
    ap.add_argument("--output-mask", default=None,
                    help="Path to save binary mask video for benchmarking (e.g. pred_mask.mp4).")
    return ap.parse_args()


def main():
    args = _parse_args()
    if not os.path.exists(MODEL_PATH):
        print(f"[setup] Model not found: {MODEL_PATH}")
        print(f"[setup] Export it on a PC first:")
        print(f"[setup]   python src/export_model.py --checkpoint trained_models/best_model.pt --img-size {MODEL_IMGSZ}")
        print(f"[setup]   mkdir -p models/{MODEL}")
        print(f"[setup]   cp trained_models/{MODEL}_float32.tflite models/{MODEL}/")
        sys.exit(1)

    # -- determine source --
    source = args.input if args.input else SOURCE
    is_video_file = isinstance(source, str) and os.path.isfile(source)

    # -- quick probe to lock in the frame shape + FPS --
    cap_probe = cv2.VideoCapture(source)
    if not cap_probe.isOpened():
        print(f"Cannot open source: {source}")
        return
    ret, first = cap_probe.read()
    if not ret:
        cap_probe.release()
        print("Cannot read first frame")
        return
    h, w = first.shape[:2]
    frame_shape = (h, w, 3)
    cam_fps = cap_probe.get(cv2.CAP_PROP_FPS)

    if is_video_file:
        # Video files always report a reliable FPS
        if cam_fps is None or cam_fps <= 0:
            cam_fps = 30.0
        total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[video] {w}×{h} @ {cam_fps:.1f} FPS  ({total_frames} frames)")
    else:
        # Many USB webcams on Linux report 0 FPS — measure it manually
        if cam_fps is None or cam_fps <= 0 or cam_fps > 300:
            print("[camera] CAP_PROP_FPS unreliable, measuring real FPS…")
            num_test = 20
            t0 = time.time()
            for _ in range(num_test):
                cap_probe.read()
            elapsed = time.time() - t0
            cam_fps = num_test / max(elapsed, 1e-9)
        print(f"[camera] {w}×{h} @ {cam_fps:.1f} FPS")

    cap_probe.release()

    # -- shared memory blocks --
    frame_nbytes = int(np.prod(frame_shape))            # h*w*3 uint8
    mask_nbytes  = h * w                                # h*w   uint8

    shm_frame = _create_shm("livemask_frame", frame_nbytes)
    shm_mask  = _create_shm("livemask_mask",  mask_nbytes)

    # Synchronisation: locks for shared memory, event for mask availability
    frame_lock   = mp.Lock()
    mask_lock    = mp.Lock()
    mask_ready   = mp.Event()
    stop_event   = mp.Event()
    seg_fps_val  = mp.Value('d', 0.0)       # shared double for segmentation FPS
    det_count    = mp.Value('i', 0)         # shared int for last detection count
    prob_stats   = mp.Array('d', [0.0, 0.0, 0.0])  # [min, max, mean] for student model

    grabber = mp.Process(
        target=frame_grabber,
        args=(source, shm_frame.name, frame_shape,
              frame_lock, stop_event, cam_fps),
        daemon=True,
    )
    worker = mp.Process(
        target=seg_worker,
        args=(shm_frame.name, shm_mask.name, frame_shape,
              frame_lock, mask_lock, mask_ready,
              stop_event, seg_fps_val, det_count, prob_stats, cam_fps),
        daemon=True,
    )

    grabber.start()
    worker.start()

    # -- local numpy views into shared memory --
    frame_buf = _np_from_shm(shm_frame, frame_shape)
    mask_buf  = _np_from_shm(shm_mask,  (h, w))

    background  = first.copy().astype(np.float32)
    person_mask = np.zeros((h, w), dtype=np.uint8)
    debug       = False
    prev_time   = time.time()

    # Cached mask-derived arrays (recomputed only when YOLO sends a new mask)
    has_person     = False
    cached_alpha_3ch     = None
    cached_inv_alpha_3ch = None
    cached_mask_3ch      = None
    cached_inv_3ch       = None

    frame_delay_ms = max(1, int(1000.0 / cam_fps))  # waitKey delay in ms
    frame_interval = 1.0 / cam_fps                    # seconds per frame

    # -- video recording setup --
    video_writer = None
    recording    = False
    record_path  = args.output

    def _start_writer(path):
        """Create and return a VideoWriter for the given path."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(path, fourcc, cam_fps, (w, h))
        if not vw.isOpened():
            print(f"[record] ERROR — could not open writer for {path}")
            return None
        print(f"[record] ▶ saving to {path}")
        return vw

    # If --output was given, start recording immediately
    if record_path:
        video_writer = _start_writer(record_path)
        recording = video_writer is not None

    # -- mask recording setup (--output-mask) --
    mask_writer = None
    if args.output_mask:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        mask_writer = cv2.VideoWriter(args.output_mask, fourcc, cam_fps, (w, h), isColor=False)
        if mask_writer.isOpened():
            print(f"[mask] saving binary mask to {args.output_mask}")
        else:
            print(f"[mask] ERROR — could not open writer for {args.output_mask}")
            mask_writer = None

    print(f"Running — rate-limited to {cam_fps:.1f} FPS ({frame_delay_ms} ms/frame)")
    if is_video_file:
        print(f"Playing back: {source}")
    else:
        print("Stand clear for a second at start for best results.")
    keys_help = "Keys: q=quit  d=debug"
    if not record_path:
        keys_help += "  r=toggle recording"
    print(keys_help)

    try:
        while not stop_event.is_set():
            loop_start = time.time()

            # ── check worker health ──────────────────────────────────────────
            if not worker.is_alive():
                print("[main] Seg worker died — check output above for errors")
                break

            # ── grab latest frame (never blocks — just a memcpy under lock) ──
            with frame_lock:
                frame = frame_buf.copy()

            # ── grab latest mask if YOLO has produced one (non-blocking) ─────
            if mask_ready.is_set():
                with mask_lock:
                    person_mask = mask_buf.copy()
                mask_ready.clear()

                # Rebuild cached arrays only when mask changes
                has_person = person_mask.any()
                if has_person:
                    alpha = person_mask.astype(np.float32) * (1.0 / 255.0)
                    cached_alpha_3ch     = cv2.merge([alpha, alpha, alpha])
                    cached_inv_alpha_3ch = 1.0 - cached_alpha_3ch
                    cached_mask_3ch = cv2.merge(
                        [person_mask, person_mask, person_mask])
                    cached_inv_3ch  = cv2.bitwise_not(cached_mask_3ch)

            # ── compositing ──────────────────────────────────────────────────
            if has_person:
                # Background: update only where there is NO person
                diff = frame.astype(np.float32) - background
                background += cached_inv_alpha_3ch * BG_LEARN * diff
                np.clip(background, 0, 255, out=background)
                bg = background.astype(np.uint8)

                # Blend: person area → stored background, rest → live feed
                output = cv2.add(
                    cv2.multiply(bg,    cached_mask_3ch, scale=1.0/255,
                                 dtype=cv2.CV_8U),
                    cv2.multiply(frame, cached_inv_3ch,  scale=1.0/255,
                                 dtype=cv2.CV_8U))
            else:
                # No person — learn background fully, pass-through live feed
                background += BG_LEARN * (
                    frame.astype(np.float32) - background)
                np.clip(background, 0, 255, out=background)
                output = frame

            if debug:
                tint = frame.copy()
                tint[person_mask > 127] = (0, 0, 180)
                output = cv2.addWeighted(frame, 0.5, tint, 0.5, 0)

            # ── FPS overlay ──────────────────────────────────────────────────
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-9)
            prev_time = curr_time
            seg_fps = seg_fps_val.value
            n_dets  = det_count.value
            det_str = f"  Det: {n_dets}" if n_dets >= 0 else ""
            cv2.putText(output, f"Display: {fps:.0f}  Seg: {seg_fps:.1f}{det_str}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            p_min, p_max, p_mean = prob_stats[0], prob_stats[1], prob_stats[2]
            if p_max > 0:
                cv2.putText(output,
                            f"Prob  min:{p_min:.2f}  max:{p_max:.2f}  mean:{p_mean:.2f}",
                            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # ── mask recording ───────────────────────────────────────────
            if mask_writer is not None:
                mask_writer.write(person_mask)

            # ── recording indicator + write ─────────────────────────────
            if recording and video_writer is not None:
                video_writer.write(output)
                cv2.circle(output, (w - 20, 20), 8, (0, 0, 255), -1)

            cv2.imshow("Invisible Human  (q=quit, d=debug)", output)

            # Rate-limit display loop to camera FPS
            processing_ms = int((time.time() - loop_start) * 1000)
            wait_ms = max(1, frame_delay_ms - processing_ms)
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                debug = not debug
            elif key == ord('r') and not record_path:
                # Toggle recording (only when --output was NOT given)
                if not recording:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    auto_path = f"recording_{ts}.mp4"
                    video_writer = _start_writer(auto_path)
                    recording = video_writer is not None
                else:
                    print("[record] ■ stopped")
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    recording = False

    finally:
        # ── clean shutdown ────────────────────────────────────────────────
        if mask_writer is not None:
            mask_writer.release()
            print("[mask] ■ file saved")
        if video_writer is not None:
            video_writer.release()
            print("[record] ■ file saved")
        stop_event.set()
        grabber.join(timeout=2)
        worker.join(timeout=2)
        cv2.destroyAllWindows()
        shm_frame.close(); shm_frame.unlink()
        shm_mask.close();  shm_mask.unlink()


if __name__ == "__main__":
    main()
