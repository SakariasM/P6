"""
Ablation study: compare student architecture variants.

Reports parameter count, FLOPs, forward-pass speed, and (optionally) trains
each variant for a few epochs to compare loss convergence.

Usage:
    # Quick comparison (no training, just params/FLOPs/speed):
    python src/ablation_study.py

    # Train each variant for 5 epochs and compare:
    python src/ablation_study.py --train --predictions ./hybrid_predictions \
        --image-root ./data/images --epochs 5
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from student.student_model import ABLATION_VARIANTS, build_variant
from export_model import InferenceWrapper


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def compute_flops(model, img_size):
    try:
        from thop import profile
    except ImportError:
        return None
    wrapper = InferenceWrapper(model)
    wrapper.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    macs, _ = profile(wrapper, inputs=(dummy,), verbose=False)
    return macs


def benchmark_speed(model, img_size, device="cpu", warmup=10, runs=50):
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


def quick_compare(img_size=640, device="cpu"):
    """Compare all variants on params, FLOPs, and inference speed."""
    print(f"{'Variant':<20} {'Params':>10} {'FLOPs (B)':>10} {'Speed (ms)':>14} {'FPS':>8}")
    print("-" * 66)

    results = {}
    for name in ABLATION_VARIANTS:
        model = build_variant(name)
        model.eval()

        params = count_params(model)
        flops = compute_flops(model, img_size)
        flops_b = flops / 1e9 if flops else 0.0
        mean_ms, std_ms = benchmark_speed(model, img_size, device=device)
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0

        print(f"{name:<20} {params:>10,} {flops_b:>10.2f} {mean_ms:>8.1f}+-{std_ms:<4.1f} {fps:>8.1f}")
        results[name] = dict(params=params, flops_b=flops_b, mean_ms=mean_ms, fps=fps)

        del model

    return results


def train_compare(args):
    """Train each variant for a few epochs and compare loss convergence."""
    from training.hybrid_distillation_train import (
        select_teacher_layers, compute_teacher_attention, ChunkDataset, collate_fn,
    )
    from training.distillation_loss import SegmentationDistillationLoss
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load chunk files
    pred_path = Path(args.predictions)
    if pred_path.is_dir():
        chunk_files = sorted(pred_path.glob("chunk_*.torch"))
    else:
        chunk_files = [pred_path]

    if not chunk_files:
        print(f"ERROR: No chunk files found in {pred_path}")
        sys.exit(1)

    # Detect teacher channels from first chunk
    first_preds = torch.load(chunk_files[0], weights_only=False)
    first_preds = [p for p in first_preds if p.features]
    teacher_layer_names, teacher_channels = select_teacher_layers(first_preds[0].features, num_scales=3)
    print(f"Teacher layers: {teacher_layer_names}")
    print(f"Teacher channels: {teacher_channels}")

    # Limit chunks for ablation speed
    max_chunks = min(len(chunk_files), args.max_chunks)
    chunk_files = chunk_files[:max_chunks]
    print(f"Using {max_chunks} chunks for ablation training\n")

    results = {}

    for variant_name in ABLATION_VARIANTS:
        print(f"\n{'='*60}")
        print(f"  Training variant: {variant_name}")
        print(f"{'='*60}")

        model = build_variant(variant_name, teacher_channels=teacher_channels).to(device)
        params = count_params(model)
        print(f"  Parameters: {params:,}")

        criterion = SegmentationDistillationLoss(
            attention_weight=1.0, mimicry_weight=0.5, relation_weight=0.5,
            seg_weight=args.seg_weight,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

        epoch_history = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_batches = 0

            for chunk_file in chunk_files:
                preds = torch.load(chunk_file, weights_only=False)
                preds = [p for p in preds if p.features]
                if not preds:
                    continue

                dataset = ChunkDataset(
                    preds, teacher_layer_names,
                    image_root=args.image_root,
                    target_size=(args.img_size, args.img_size),
                    augment=False,
                )
                loader = DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=2, pin_memory=True, collate_fn=collate_fn,
                )

                for batch in loader:
                    images = batch['image'].to(device)
                    teacher_feats = [f.to(device, dtype=torch.float32) for f in batch['teacher_features']]
                    teacher_atts = [compute_teacher_attention(f) for f in teacher_feats]

                    optimizer.zero_grad()
                    seg_output, distill_info = model(images)

                    projected = distill_info['projected']
                    student_atts = distill_info['attention_maps']
                    n = len(projected)

                    teacher_mask = batch.get('teacher_mask')
                    if teacher_mask is not None:
                        teacher_mask = teacher_mask.to(device, dtype=torch.float32)

                    loss, loss_dict = criterion(
                        student_atts=student_atts,
                        teacher_atts=teacher_atts[-n:],
                        projected_student_feats=projected,
                        teacher_feats=teacher_feats[-n:],
                        student_mask=seg_output,
                        teacher_mask=teacher_mask,
                    )
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss_dict['total']
                    epoch_batches += 1

                del preds, dataset, loader

            avg_loss = epoch_loss / max(epoch_batches, 1)
            epoch_history.append(avg_loss)
            print(f"  Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}")

        results[variant_name] = {
            "params": params,
            "final_loss": epoch_history[-1] if epoch_history else float('inf'),
            "loss_history": epoch_history,
        }
        del model, optimizer
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*60}")
    print(f"  ABLATION RESULTS ({args.epochs} epochs, {max_chunks} chunks)")
    print(f"{'='*60}")
    print(f"{'Variant':<20} {'Params':>10} {'Final Loss':>12} {'Improvement':>12}")
    print("-" * 58)

    baseline_loss = results.get("baseline", {}).get("final_loss", 1.0)
    for name, r in results.items():
        improvement = ((baseline_loss - r['final_loss']) / baseline_loss) * 100 if baseline_loss > 0 else 0
        sign = "+" if improvement >= 0 else ""
        print(f"{name:<20} {r['params']:>10,} {r['final_loss']:>12.4f} {sign}{improvement:>10.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Student architecture ablation study")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", type=str, default=None,
                        help="Device for benchmarking (default: auto)")

    # Training-based comparison
    parser.add_argument("--train", action="store_true",
                        help="Run training-based comparison (requires prediction data)")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to prediction chunks directory")
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-chunks", type=int, default=10,
                        help="Max chunks to use for ablation training (default: 10)")
    parser.add_argument("--seg-weight", type=float, default=0.0)

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 66)
    print("  ARCHITECTURE ABLATION STUDY")
    print("=" * 66)

    # Always run quick comparison
    print(f"\n--- Static comparison (img_size={args.img_size}, device={args.device}) ---\n")
    quick_compare(img_size=args.img_size, device=args.device)

    # Optionally run training comparison
    if args.train:
        if not args.predictions:
            print("\nERROR: --predictions required for training comparison")
            sys.exit(1)
        train_compare(args)


if __name__ == "__main__":
    main()
