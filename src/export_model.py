"""
Export trained student model to ONNX and TFLite formats.

Usage:
    python src/export_model.py --checkpoint trained_models/best_model.pt
    python src/export_model.py --checkpoint trained_models/best_model.pt --img-size 320
    python src/export_model.py --checkpoint trained_models/best_model.pt --quantize
    python src/export_model.py --checkpoint trained_models/best_model.pt --onnx-only
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from student.student_model import StudentSegmentation


class InferenceWrapper(nn.Module):
    """Wraps StudentSegmentation to return only the segmentation mask."""

    def __init__(self, model: StudentSegmentation):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.model(x)
        return output


def export_onnx(model, dummy_input, output_path):
    """Export model to ONNX format."""
    print(f"Exporting ONNX to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=18,
        input_names=["image"],
        output_names=["mask"],
        dynamic_axes={
            "image": {0: "batch"},
            "mask": {0: "batch"},
        },
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ONNX saved: {size_mb:.2f} MB")


def export_tflite(wrapper, dummy_input, output_path, quantize=False):
    """Convert PyTorch model directly to TFLite using litert-torch."""
    try:
        import litert_torch
    except ImportError:
        print("ERROR: litert-torch not installed. Install with:")
        print("  pip install litert-torch")
        return False

    print("Converting to TFLite via litert-torch...")

    edge_model = litert_torch.convert(wrapper, (dummy_input,))
    edge_model.export(str(output_path))

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  TFLite saved: {size_mb:.2f} MB")

    if quantize:
        quant_path = output_path.with_suffix(".int8.tflite")
        print("Applying INT8 quantization...")
        success = _quantize_tflite(output_path, quant_path)
        if success:
            size_mb = quant_path.stat().st_size / (1024 * 1024)
            print(f"  TFLite INT8 saved: {size_mb:.2f} MB")

    return True


def _quantize_tflite(input_tflite_path, output_path):
    """Apply quantization to an existing TFLite model."""
    try:
        from ai_edge_quantizer import quantizer as quant_lib
        q = quant_lib.Quantizer(str(input_tflite_path))
        q.quantize()
        q.export(str(output_path))
        return True
    except Exception as e:
        print(f"  Quantization failed: {e}")
        print("  Try: pip install ai-edge-quantizer")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export student model to ONNX/TFLite")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (best_model.pt)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as checkpoint)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Input image size for export")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply INT8 quantization to TFLite")
    parser.add_argument("--onnx-only", action="store_true",
                        help="Only export ONNX, skip TFLite conversion")
    args = parser.parse_args()

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    teacher_channels = checkpoint.get("teacher_channels", [256, 384, 256])
    ckpt_args = checkpoint.get("args", {})
    base_channels = ckpt_args.get("base_channels", 8)
    depth = ckpt_args.get("depth", 4)

    print(f"Model config: base_channels={base_channels}, depth={depth}")
    print(f"Teacher channels: {teacher_channels}")

    # Build model and load weights
    model = StudentSegmentation(
        in_channels=3,
        base_channels=base_channels,
        depth=depth,
        teacher_channels=teacher_channels,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Wrap for inference (drops distill_info)
    wrapper = InferenceWrapper(model)
    wrapper.eval()

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dummy input for tracing
    dummy = torch.randn(1, 3, args.img_size, args.img_size)

    # Verify forward pass
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"Output shape: {out.shape} (expected [1, 1, {args.img_size}, {args.img_size}])")

    # Export ONNX
    onnx_path = output_dir / f"student_seg_{args.img_size}.onnx"
    export_onnx(wrapper, dummy, onnx_path)

    if args.onnx_only:
        print("\nDone (ONNX only).")
        return

    # Export TFLite
    tflite_path = output_dir / f"student_seg_{args.img_size}.tflite"
    success = export_tflite(wrapper, dummy, tflite_path, quantize=args.quantize)

    if success:
        print(f"\nExport complete:")
        print(f"  ONNX:   {onnx_path}")
        print(f"  TFLite: {tflite_path}")
    else:
        print(f"\nONNX export complete: {onnx_path}")
        print("TFLite conversion failed — see errors above.")


if __name__ == "__main__":
    main()
