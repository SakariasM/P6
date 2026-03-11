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
import time
import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory

# ── config ────────────────────────────────────────────────────────────────────
SOURCE         = 0       # webcam index, or path to a video file
MASK_THRESHOLD = 0.3     # segmentation confidence (0–1, lower = more aggressive)
MASK_DILATE    = 10      # expand mask outward to cover person edges
MASK_BLUR      = 1      # feather mask edges for smooth blending (must be odd)
BG_LEARN       = 0.02    # background learning rate (higher = adapts faster)
MASK_EDGE_PAD  = 50      # if mask is within this many px of top/bottom, extend to edge
MODEL_PATH     = "yolo26n-seg_saved_model/yolo26n-seg_float32.tflite"
# ─────────────────────────────────────────────────────────────────────────────


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


# ── process: TFLite segmentation worker ───────────────────────────────────────

def seg_worker(frame_shm_name, mask_shm_name, frame_shape,
               frame_lock, mask_lock, mask_ready,
               stop_event, seg_fps_val, cam_fps):
    """Run YOLO26m-seg TFLite on the latest frame, write mask to shared mem.
    Output format: (1, 300, 38) detections + (1, 32, proto_h, proto_w) prototypes.
    Each detection: [x1, y1, x2, y2, conf, class_id, coeff×32]"""

    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

    try:
        model_file = os.path.join(os.path.dirname(__file__) or ".", MODEL_PATH)
        print(f"[worker] loading TFLite model: {model_file}")
        interp = Interpreter(model_path=model_file, num_threads=2)
        interp.allocate_tensors()

        inp_detail  = interp.get_input_details()[0]
        out_details = interp.get_output_details()
        inp_idx = inp_detail["index"]
        _, model_h, model_w, _ = inp_detail["shape"]

        # Identify outputs by rank: 3-D = detections, 4-D = prototypes
        det_idx, proto_idx = None, None
        for od in out_details:
            if len(od["shape"]) == 3:
                det_idx = od["index"]
            elif len(od["shape"]) == 4:
                proto_idx = od["index"]
                _, proto_h, proto_w, n_proto = od["shape"]  # TFLite is channel-last (H, W, C)

        if det_idx is None or proto_idx is None:
            raise RuntimeError(f"Unexpected output shapes: {[od['shape'] for od in out_details]}")

        print(f"[worker] model loaded OK — input {model_w}×{model_h}, "
              f"prototypes {proto_w}×{proto_h}×{n_proto}")
    except Exception as exc:
        print(f"[worker] FAILED to load model: {exc}")
        import traceback; traceback.print_exc()
        stop_event.set()
        return

    shm_frame = shared_memory.SharedMemory(name=frame_shm_name, create=False)
    shm_mask  = shared_memory.SharedMemory(name=mask_shm_name, create=False)
    frame_buf = _np_from_shm(shm_frame, frame_shape)
    h, w = frame_shape[:2]
    mask_buf  = _np_from_shm(shm_mask, (h, w))

    # Post-process kernels scaled to prototype resolution
    scale = proto_h / h
    small_dilate = max(3, int(MASK_DILATE * scale) | 1)
    small_blur   = max(3, int(MASK_BLUR   * scale) | 1)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_dilate, small_dilate))
    print(f"[worker] post-process at {proto_w}×{proto_h}: "
          f"dilate {small_dilate}×{small_dilate}, blur {small_blur}×{small_blur}")

    input_data = np.empty((1, model_h, model_w, 3), dtype=np.float32)
    proto_flat_buf = np.empty((proto_h * proto_w, n_proto), dtype=np.float32)  # (6400, 32)

    frame_interval = 1.0 / max(cam_fps, 1.0)

    print("[worker] entering inference loop")
    prev_time = time.time()
    try:
        while not stop_event.is_set():
            t_start = time.time()

            with frame_lock:
                frame = frame_buf.copy()

            # Preprocess: resize → RGB → normalise
            small = cv2.resize(frame, (model_w, model_h))
            cv2.cvtColor(small, cv2.COLOR_BGR2RGB, dst=small)
            np.multiply(small, 1.0 / 255.0, out=input_data[0], casting="unsafe")

            interp.set_tensor(inp_idx, input_data)
            interp.invoke()

            detections = interp.get_tensor(det_idx)[0]   # (300, 38)
            prototypes = interp.get_tensor(proto_idx)[0] # (32, proto_h, proto_w)

            # Filter: confidence > threshold AND class == 0 (person)
            confs   = detections[:, 4]
            classes = detections[:, 5].astype(int)
            valid   = (confs > MASK_THRESHOLD) & (classes == 0)

            small_mask = np.zeros((proto_h, proto_w), dtype=np.uint8)

            if valid.any():
                coeffs = detections[valid, 6:]                       # (N, 32)
                np.copyto(proto_flat_buf, prototypes.reshape(-1, n_proto))  # (6400, 32)
                logits = coeffs @ proto_flat_buf.T                   # (N, 6400)
                # sigmoid
                person_masks = 1.0 / (1.0 + np.exp(-logits))    # (N, proto_h*proto_w)
                combined = person_masks.max(axis=0).reshape(proto_h, proto_w)
                small_mask = (combined > 0.5).astype(np.uint8) * 255

                if small_mask.any():
                    small_mask = cv2.dilate(small_mask, dilate_kernel, iterations=2)
                    small_mask = cv2.GaussianBlur(small_mask, (small_blur, small_blur), 0)

            # Upscale to frame resolution
            mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_LINEAR)

            # Extend mask to frame edges if close to top/bottom
            if mask.any():
                rows_with_mask = np.where(mask.max(axis=1) > 0)[0]
                if len(rows_with_mask) > 0:
                    top_row = rows_with_mask[0]
                    bot_row = rows_with_mask[-1]
                    if top_row < MASK_EDGE_PAD:
                        mask[:top_row + 1, :] = np.maximum(
                            mask[:top_row + 1, :],
                            mask[top_row:top_row + 1, :])
                    if bot_row > (h - 1 - MASK_EDGE_PAD):
                        mask[bot_row:, :] = np.maximum(
                            mask[bot_row:, :],
                            mask[bot_row:bot_row + 1, :])

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
    return ap.parse_args()


def main():
    args = _parse_args()

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
              stop_event, seg_fps_val, cam_fps),
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
            cv2.putText(output, f"Display: {fps:.0f}  Seg: {seg_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
