"""
Backfill validation metrics for existing checkpoints.

Loads each checkpoint_epoch_*.pt, runs validation on the held-out chunks,
and rewrites training_history.json with train_/val_ prefixed keys.

Usage (on cluster):
    python -m backfill_val_metrics \
        --predictions /ceph/project/P6-Machine-Vision/P6/results/hybrid_predictions \
        --image-root /ceph/project/P6-Machine-Vision/P6/data/open-images-v7/train/data \
        --checkpoint-dir /ceph/project/P6-Machine-Vision/P6/trained_models \
        --output /ceph/project/P6-Machine-Vision/P6/trained_models/training_history.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from student.student_model import StudentSegmentation
from teacher.hybrid_predictions import HybridTeacherPrediction  # noqa: F401 - needed for torch.load unpickling
from training.distillation_loss import SegmentationDistillationLoss
from training.hybrid_distillation_train import (
    discover_chunk_files,
    select_teacher_layers,
    validate,
)


def main():
    parser = argparse.ArgumentParser(description="Backfill val metrics for old checkpoints")
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for updated history JSON")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    output_path = Path(args.output) if args.output else ckpt_dir / "training_history.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Discover chunks and split
    all_chunks = discover_chunk_files(args.predictions)
    n_val = max(1, int(len(all_chunks) * args.val_split))
    val_chunks = all_chunks[-n_val:]
    print(f"Total chunks: {len(all_chunks)}, Val chunks: {len(val_chunks)}")

    # Find checkpoints
    ckpt_files = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"),
                        key=lambda p: int(p.stem.split("_")[-1]))
    if not ckpt_files:
        print("No checkpoint_epoch_*.pt files found.")
        return

    print(f"Found {len(ckpt_files)} checkpoints: "
          f"epochs {[int(p.stem.split('_')[-1]) for p in ckpt_files]}")

    # Load existing history
    history_path = ckpt_dir / "training_history.json"
    old_history = {}
    if history_path.exists():
        with open(history_path) as f:
            for entry in json.load(f):
                old_history[entry["epoch"]] = entry

    # Determine teacher layout from first chunk
    first_preds = torch.load(all_chunks[0], weights_only=False)
    first_preds = [p for p in first_preds if p.features]
    teacher_layer_names, teacher_channels = select_teacher_layers(
        first_preds[0].features, num_scales=3
    )
    del first_preds
    print(f"Teacher layers: {teacher_layer_names}, channels: {teacher_channels}")

    history = []

    for ckpt_path in ckpt_files:
        epoch = int(ckpt_path.stem.split("_")[-1])
        print(f"\n--- Epoch {epoch} ({ckpt_path.name}) ---")

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_args = checkpoint.get("args", {})

        # Build model
        model = StudentSegmentation(
            in_channels=3,
            base_channels=ckpt_args.get("base_channels", 8),
            depth=ckpt_args.get("depth", 4),
            teacher_channels=teacher_channels,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Build criterion matching training config
        criterion = SegmentationDistillationLoss(
            attention_weight=ckpt_args.get("attention_weight", 1.0),
            mimicry_weight=ckpt_args.get("mimicry_weight", 0.5),
            relation_weight=ckpt_args.get("relation_weight", 0.5),
            seg_weight=ckpt_args.get("seg_weight", 1.0),
        )

        # Run validation
        val_metrics = validate(model, val_chunks, teacher_layer_names,
                               teacher_channels, criterion, device, args)

        iou_str = f"  IoU={val_metrics['iou']:.4f}  Dice={val_metrics['dice']:.4f}" if 'iou' in val_metrics else ""
        print(f"  Val — total={val_metrics['total']:.4f}"
              f"  att={val_metrics['attention']:.4f}"
              f"  mim={val_metrics['mimicry']:.4f}"
              f"  seg={val_metrics.get('segmentation', 0):.4f}{iou_str}")

        # Build history entry
        old_entry = old_history.get(epoch, {})

        # Get train metrics from old history (flat keys)
        train_keys = ["total", "attention", "mimicry", "relation", "segmentation"]
        entry = {"epoch": epoch}
        for k in train_keys:
            # Check both old format (flat) and new format (train_ prefix)
            val = old_entry.get(f"train_{k}", old_entry.get(k, 0.0))
            entry[f"train_{k}"] = val
        for k in train_keys:
            entry[f"val_{k}"] = val_metrics.get(k, 0.0)
        if 'iou' in val_metrics:
            entry["val_iou"] = val_metrics["iou"]
            entry["val_dice"] = val_metrics["dice"]
        entry["lr"] = old_entry.get("lr", 0.0)

        history.append(entry)

        del model, checkpoint
        torch.cuda.empty_cache()

    # Save updated history
    history.sort(key=lambda e: e["epoch"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved updated history ({len(history)} epochs) to {output_path}")


if __name__ == "__main__":
    main()
