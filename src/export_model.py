"""
Export trained student model to ONNX and TFLite formats.

Single model:
    python src/export_model.py --checkpoint trained_models/best_model.pt
    python src/export_model.py --checkpoint trained_models/best_model.pt --img-size 320
    python src/export_model.py --checkpoint trained_models/best_model.pt --quantize
    python src/export_model.py --checkpoint trained_models/best_model.pt --onnx-only

All ablation models:
    python src/export_model.py --ablation-dir trained_models/ablation
    python src/export_model.py --ablation-dir trained_models/ablation --onnx-only --img-size 320
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


def export_checkpoint(ckpt_path: Path, output_dir: Path, img_size: int,
                      onnx_only: bool, quantize: bool, tag: str = ""):
    """Export a single checkpoint to ONNX (and optionally TFLite)."""
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    needs_upcast = any(v.dtype == torch.float16 for v in state_dict.values())
    if needs_upcast:
        state_dict = {k: v.float() for k, v in state_dict.items()}

    teacher_channels = checkpoint.get("teacher_channels", [256, 384, 256])
    ckpt_args = checkpoint.get("args", {})
    base_channels = ckpt_args.get("base_channels", 8)
    depth = ckpt_args.get("depth", 4)
    cbam_levels = ckpt_args.get("cbam_levels", None)

    print(f"  Model config: base_channels={base_channels}, depth={depth}")
    print(f"  Teacher channels: {teacher_channels}")
    if cbam_levels is not None:
        print(f"  CBAM levels: {cbam_levels}")

    model = StudentSegmentation(
        in_channels=3,
        base_channels=base_channels,
        depth=depth,
        teacher_channels=teacher_channels,
        cbam_levels=cbam_levels,
    )
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    wrapper = InferenceWrapper(model)
    wrapper.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, img_size, img_size)

    with torch.no_grad():
        out = wrapper(dummy)
    print(f"  Output shape: {out.shape}")

    name = f"student_seg_{tag}_{img_size}" if tag else f"student_seg_{img_size}"
    onnx_path = output_dir / f"{name}.onnx"
    export_onnx(wrapper, dummy, onnx_path)

    if onnx_only:
        return

    tflite_path = output_dir / f"{name}.tflite"
    export_tflite(wrapper, dummy, tflite_path, quantize=quantize)


def main():
    parser = argparse.ArgumentParser(description="Export student model to ONNX/TFLite")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint (best_model.pt)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as checkpoint)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Input image size for export")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply INT8 quantization to TFLite")
    parser.add_argument("--onnx-only", action="store_true",
                        help="Only export ONNX, skip TFLite conversion")
    parser.add_argument("--ablation-dir", type=str, default=None,
                        help="Export all ablation models from this directory")
    args = parser.parse_args()

    if args.ablation_dir:
        ablation_dir = Path(args.ablation_dir)
        if not ablation_dir.exists():
            print(f"ERROR: Ablation directory not found: {ablation_dir}")
            sys.exit(1)

        configs_found = 0
        for subdir in sorted(ablation_dir.iterdir()):
            if not subdir.is_dir():
                continue
            ckpt = subdir / "best_model.pt"
            deploy_ckpt = list(subdir.glob("best_model_*.pt"))
            chosen = deploy_ckpt[0] if deploy_ckpt else ckpt
            if not chosen.exists():
                print(f"Skipping {subdir.name}: no checkpoint found")
                continue

            configs_found += 1
            config_name = subdir.name
            out = Path(args.output_dir) if args.output_dir else subdir
            print(f"\n{'='*60}")
            print(f"Exporting: {config_name}")
            print(f"{'='*60}")
            export_checkpoint(chosen, out, args.img_size,
                              args.onnx_only, args.quantize, tag=config_name)

        print(f"\nDone. Exported {configs_found} ablation models.")

    elif args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"ERROR: Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
        export_checkpoint(ckpt_path, output_dir, args.img_size,
                          args.onnx_only, args.quantize)
    else:
        print("ERROR: Provide either --checkpoint or --ablation-dir")
        sys.exit(1)


if __name__ == "__main__":
    main()
