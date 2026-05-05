"""
Run inference with the trained student segmentation model.
Supports .pt (PyTorch), .onnx, and .tflite model formats.

Usage:
    python src/run_inference.py --model trained_models/best_model.pt --image photo.jpg
    python src/run_inference.py --model trained_models/best_model.pt --video clip.mp4
    python src/run_inference.py --model trained_models/best_model.pt --video clip.mp4 --save output.mp4
    python src/run_inference.py --model trained_models/best_model.pt --webcam
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))


def load_pytorch_model(model_path, device="cpu"):
    """Load PyTorch .pt checkpoint."""
    import torch
    from student.student_model import StudentSegmentation
    from export_model import InferenceWrapper

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    teacher_channels = checkpoint.get("teacher_channels", [256, 384, 256])
    ckpt_args = checkpoint.get("args", {})

    model = StudentSegmentation(
        in_channels=3,
        base_channels=ckpt_args.get("base_channels", 8),
        depth=ckpt_args.get("depth", 4),
        teacher_channels=teacher_channels,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    wrapper = InferenceWrapper(model).to(device)
    wrapper.eval()

    def predict(img_array):
        """img_array: [H, W, 3] float32 in [0,1]"""
        import torch
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            mask = wrapper(tensor)
        return mask.squeeze().cpu().numpy()

    return predict


def load_onnx_model(model_path):
    """Load ONNX model."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name

    def predict(img_array):
        """img_array: [H, W, 3] float32 in [0,1]"""
        tensor = img_array.transpose(2, 0, 1)[np.newaxis]  # NCHW
        outputs = session.run(None, {input_name: tensor})
        return outputs[0].squeeze()

    return predict


def load_tflite_model(model_path):
    """Load TFLite model."""
    try:
        from ai_edge_litert import interpreter as tfl
        interpreter = tfl.Interpreter(model_path=str(model_path))
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=str(model_path))
        except ImportError:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=str(model_path))

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]  # e.g. [1, 3, H, W] or [1, H, W, 3]
    is_nhwc = input_shape[-1] == 3

    def predict(img_array):
        """img_array: [H, W, 3] float32 in [0,1]"""
        if is_nhwc:
            tensor = img_array[np.newaxis]  # [1, H, W, 3]
        else:
            tensor = img_array.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]
        interpreter.set_tensor(input_details[0]["index"], tensor.astype(np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        return output.squeeze()

    return predict


def load_model(model_path, device="cpu"):
    """Load model based on file extension."""
    ext = Path(model_path).suffix.lower()
    if ext == ".pt":
        print(f"Loading PyTorch model: {model_path}")
        return load_pytorch_model(model_path, device)
    elif ext == ".onnx":
        print(f"Loading ONNX model: {model_path}")
        return load_onnx_model(model_path)
    elif ext == ".tflite":
        print(f"Loading TFLite model: {model_path}")
        return load_tflite_model(model_path)
    else:
        raise ValueError(f"Unsupported model format: {ext}")


def preprocess(image, img_size):
    """Resize and normalize image to float32 [0,1]."""
    image = image.resize((img_size, img_size), Image.BILINEAR)
    return np.array(image, dtype=np.float32) / 255.0


def visualize(original, mask, threshold=0.5, alpha=0.5):
    """Overlay segmentation mask on original image."""
    import matplotlib.pyplot as plt

    # Resize mask to original image size
    h, w = original.shape[:2]
    mask_resized = np.array(Image.fromarray(mask).resize((w, h), Image.BILINEAR))

    binary_mask = mask_resized > threshold

    # Create colored overlay
    overlay = original.copy()
    overlay[binary_mask] = (
        overlay[binary_mask] * (1 - alpha) +
        np.array([0, 255, 0], dtype=np.float64) * alpha
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask_resized, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Segmentation Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (threshold={threshold})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def run_webcam(predict_fn, img_size, threshold=0.5):
    """Run live webcam inference."""
    import cv2

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    print("Webcam running. Press 'q' to quit, '+'/'-' to adjust threshold.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = preprocess(Image.fromarray(rgb), img_size)

        # Inference
        t0 = time.time()
        mask = predict_fn(img)
        dt = time.time() - t0

        # Resize mask to frame size
        h, w = frame.shape[:2]
        mask_resized = cv2.resize(mask, (w, h))
        binary = (mask_resized > threshold).astype(np.uint8)

        # Green overlay
        overlay = frame.copy()
        overlay[binary == 1] = (
            overlay[binary == 1] * 0.5 +
            np.array([0, 255, 0]) * 0.5
        ).astype(np.uint8)

        # FPS counter
        fps = 1.0 / max(dt, 1e-6)
        cv2.putText(overlay, f"FPS: {fps:.1f} | thresh: {threshold:.2f}",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Person Segmentation", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("+") or key == ord("="):
            threshold = min(threshold + 0.05, 0.95)
        elif key == ord("-"):
            threshold = max(threshold - 0.05, 0.05)

    cap.release()
    cv2.destroyAllWindows()


def run_video(predict_fn, img_size, video_path, threshold=0.5, save_path=None):
    """Run inference on a video file."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps_in:.1f} FPS, {total} frames")

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps_in, (w, h))
        print(f"Saving output to: {save_path}")

    print("Playing. Press 'q' to quit, '+'/'-' to adjust threshold.")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = preprocess(Image.fromarray(rgb), img_size)

        t0 = time.time()
        mask = predict_fn(img)
        dt = time.time() - t0

        mask_resized = cv2.resize(mask, (w, h))
        binary = (mask_resized > threshold).astype(np.uint8)

        overlay = frame.copy()
        overlay[binary == 1] = (
            overlay[binary == 1] * 0.5 +
            np.array([0, 255, 0]) * 0.5
        ).astype(np.uint8)

        fps = 1.0 / max(dt, 1e-6)
        cv2.putText(overlay, f"FPS: {fps:.1f} | thresh: {threshold:.2f} | {frame_idx}/{total}",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if writer:
            writer.write(overlay)

        cv2.imshow("Person Segmentation", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("+") or key == ord("="):
            threshold = min(threshold + 0.05, 0.95)
        elif key == ord("-"):
            threshold = max(threshold - 0.05, 0.05)

    cap.release()
    if writer:
        writer.release()
        print(f"Saved {frame_idx} frames to: {save_path}")
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Run student segmentation inference")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model (.pt, .onnx, or .tflite)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to input video file (.mp4, .avi, etc.)")
    parser.add_argument("--webcam", action="store_true",
                        help="Use webcam for live inference")
    parser.add_argument("--img-size", type=int, default=320,
                        help="Model input size")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Segmentation threshold")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for PyTorch models (cpu/cuda)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save output to file instead of displaying")
    args = parser.parse_args()

    if not args.image and not args.webcam and not args.video:
        print("ERROR: Provide --image, --video, or --webcam")
        sys.exit(1)

    predict_fn = load_model(args.model, args.device)

    if args.webcam:
        run_webcam(predict_fn, args.img_size, args.threshold)
    elif args.video:
        run_video(predict_fn, args.img_size, args.video, args.threshold, args.save)
    else:
        # Single image inference
        original = Image.open(args.image).convert("RGB")
        img = preprocess(original, args.img_size)

        t0 = time.time()
        mask = predict_fn(img)
        dt = time.time() - t0
        print(f"Inference time: {dt*1000:.1f} ms")
        print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")

        original_np = np.array(original)

        if args.save:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            h, w = original_np.shape[:2]
            mask_resized = np.array(
                Image.fromarray(mask).resize((w, h), Image.BILINEAR)
            )
            binary = mask_resized > args.threshold
            overlay = original_np.copy()
            overlay[binary] = (
                overlay[binary] * 0.5 +
                np.array([0, 255, 0], dtype=np.float64) * 0.5
            ).astype(np.uint8)
            Image.fromarray(overlay).save(args.save)
            print(f"Saved to: {args.save}")
        else:
            visualize(original_np, mask, args.threshold)


if __name__ == "__main__":
    main()
