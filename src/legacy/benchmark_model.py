"""
Benchmark student segmentation model: params, FLOPs, and inference speed.

Usage:
    python src/benchmark_model.py --checkpoint trained_models/best_model.pt
    python src/benchmark_model.py --checkpoint trained_models/best_model.pt --onnx trained_models/student_seg_320.onnx
    python src/benchmark_model.py --checkpoint trained_models/best_model.pt --img-size 320 640
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from student.student_model import StudentSegmentation
from export_model import InferenceWrapper

# YOLO26n-seg reference values from ultralytics benchmark table
YOLO_REFERENCE = {
    "params_m": 2.7,
    "flops_b": 9.1,
    "cpu_onnx_ms": 53.3,
    "gpu_t4_ms": 2.1,
}


def load_student_model(checkpoint_path):
    """Load student model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    return model


def count_parameters(model):
    """Returns total parameter count."""
    return sum(p.numel() for p in model.parameters())


def compute_flops(model, img_size):
    """Compute FLOPs (MACs) using thop, matching YOLO's convention."""
    try:
        from thop import profile
    except ImportError:
        print("  WARNING: thop not installed (pip install ultralytics-thop)")
        return None

    wrapper = InferenceWrapper(model)
    wrapper.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    macs, _ = profile(wrapper, inputs=(dummy,), verbose=False)
    return macs  # Report MACs as "FLOPs" to match YOLO convention


def benchmark_pytorch(model, img_size, device, warmup, runs):
    """Benchmark PyTorch inference speed."""
    wrapper = InferenceWrapper(model).to(device)
    wrapper.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            wrapper(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            wrapper(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), np.std(times)


def benchmark_onnx(onnx_path, img_size, warmup, runs):
    """Benchmark ONNX inference speed on CPU."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  WARNING: onnxruntime not installed")
        return None, None, None

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    actual_size = img_size
    if isinstance(input_shape[2], int) and input_shape[2] != img_size:
        actual_size = input_shape[2]
        print(f"  NOTE: ONNX model expects {actual_size}x{actual_size}, using that instead of {img_size}")

    dummy = np.random.randn(1, 3, actual_size, actual_size).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {input_name: dummy})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), np.std(times), actual_size


def benchmark_tflite(tflite_path, img_size, warmup, runs):
    """Benchmark TFLite inference speed on CPU."""
    try:
        from ai_edge_litert import interpreter as tfl
        interpreter = tfl.Interpreter(model_path=str(tflite_path))
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=str(tflite_path))
        except ImportError:
            try:
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            except ImportError:
                print("  WARNING: No TFLite runtime found")
                return None, None, None

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    is_nhwc = input_shape[-1] == 3
    actual_size = input_shape[2] if not is_nhwc else input_shape[1]

    if actual_size != img_size:
        print(f"  NOTE: TFLite model expects {actual_size}x{actual_size}, using that instead of {img_size}")

    if is_nhwc:
        dummy = np.random.randn(1, actual_size, actual_size, 3).astype(np.float32)
    else:
        dummy = np.random.randn(1, 3, actual_size, actual_size).astype(np.float32)

    for _ in range(warmup):
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()
        times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), np.std(times), actual_size


def fmt(mean, std):
    """Format timing result."""
    return f"{mean:.1f} \u00b1 {std:.1f}"


def main():
    parser = argparse.ArgumentParser(description="Benchmark student segmentation model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--onnx", type=str, default=None,
                        help="Path to .onnx model")
    parser.add_argument("--tflite", type=str, default=None,
                        help="Path to .tflite model")
    parser.add_argument("--img-size", type=int, nargs="+", default=[640],
                        help="Input size(s) to benchmark (default: 640)")
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of timed runs (default: 100)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (default: 10)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading: {ckpt_path}")
    model = load_student_model(ckpt_path)

    total_params = count_parameters(model)
    print(f"Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    has_cuda = torch.cuda.is_available()

    for img_size in args.img_size:
        print(f"\n{'=' * 60}")
        print(f"  Benchmark @ {img_size}x{img_size}  ({args.runs} runs, {args.warmup} warmup)")
        print(f"{'=' * 60}")

        # FLOPs
        flops = compute_flops(model, img_size)
        flops_b = flops / 1e9 if flops else None
        if flops_b:
            print(f"FLOPs: {flops_b:.1f}B")

        # PyTorch CPU
        print("\nPyTorch CPU...")
        pt_mean, pt_std = benchmark_pytorch(model, img_size, "cpu", args.warmup, args.runs)

        # PyTorch GPU
        gpu_mean, gpu_std = None, None
        if has_cuda:
            print("PyTorch GPU...")
            gpu_mean, gpu_std = benchmark_pytorch(model, img_size, "cuda", args.warmup, args.runs)

        # ONNX
        onnx_mean, onnx_std, onnx_size = None, None, None
        if args.onnx:
            print("ONNX CPU...")
            onnx_mean, onnx_std, onnx_size = benchmark_onnx(args.onnx, img_size, args.warmup, args.runs)

        # TFLite
        tfl_mean, tfl_std, tfl_size = None, None, None
        if args.tflite:
            print("TFLite CPU...")
            tfl_mean, tfl_std, tfl_size = benchmark_tflite(args.tflite, img_size, args.warmup, args.runs)

        # Results table
        print(f"\n{'─' * 60}")
        print(f"{'Model':<16} {'Params':>8} {'FLOPs':>8} {'PT CPU':>14} ", end="")
        if has_cuda:
            print(f"{'PT GPU':>14} ", end="")
        if onnx_mean is not None:
            onnx_label = f"ONNX CPU" if onnx_size == img_size else f"ONNX({onnx_size})"
            print(f"{onnx_label:>14} ", end="")
        if tfl_mean is not None:
            tfl_label = f"TFLite" if tfl_size == img_size else f"TFL({tfl_size})"
            print(f"{tfl_label:>14} ", end="")
        print()

        print(f"{'':16} {'(M)':>8} {'(B)':>8} {'(ms)':>14} ", end="")
        if has_cuda:
            print(f"{'(ms)':>14} ", end="")
        if onnx_mean is not None:
            print(f"{'(ms)':>14} ", end="")
        if tfl_mean is not None:
            print(f"{'(ms)':>14} ", end="")
        print()

        print(f"{'─' * 60}")

        # Student row
        print(f"{'Student':<16} {total_params/1e6:>8.2f} {flops_b or 0:>8.1f} {fmt(pt_mean, pt_std):>14} ", end="")
        if has_cuda:
            print(f"{fmt(gpu_mean, gpu_std):>14} ", end="")
        if onnx_mean is not None:
            print(f"{fmt(onnx_mean, onnx_std):>14} ", end="")
        if tfl_mean is not None:
            print(f"{fmt(tfl_mean, tfl_std):>14} ", end="")
        print()

        # YOLO reference row
        yolo_onnx = f"{YOLO_REFERENCE['cpu_onnx_ms']}" if img_size == 640 else "--"
        print(f"{'YOLO26n-seg':<16} {YOLO_REFERENCE['params_m']:>8.2f} {YOLO_REFERENCE['flops_b']:>8.1f} {'--':>14} ", end="")
        if has_cuda:
            print(f"{YOLO_REFERENCE['gpu_t4_ms']:>14} ", end="")
        if onnx_mean is not None:
            print(f"{yolo_onnx:>14} ", end="")
        if tfl_mean is not None:
            print(f"{'--':>14} ", end="")
        print()

        print(f"{'─' * 60}")
        print("Note: FLOPs = MACs (matching YOLO convention). YOLO GPU = T4 TensorRT FP16.")


if __name__ == "__main__":
    main()
