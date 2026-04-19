"""
Compare training results across ablation configurations.

Reads training_history.json from each ablation output directory and produces:
1. A summary table (best val loss per config)
2. An overlay plot of val_total loss curves

Usage:
    python src/compare_ablations.py \
        --base-dir /ceph/project/P6-Machine-Vision/P6/trained_models/ablation \
        --configs configs/ablation_configs.json
    python src/compare_ablations.py \
        --base-dir trained_models/ablation \
        --configs configs/ablation_configs.json \
        --output trained_models/ablation_comparison.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_config(config_path: Path) -> dict[str, dict]:
    with open(config_path) as f:
        return json.load(f)


def load_history(history_path: Path) -> list[dict]:
    if not history_path.exists():
        return []
    with open(history_path) as f:
        history = json.load(f)
    history.sort(key=lambda e: e["epoch"])
    return history


def print_comparison_table(configs: dict[str, dict], base_dir: Path):
    """Print a summary table of best val loss and IoU/Dice for each config."""
    header = (f"{'Config':<22} {'Layers':<40} {'Epochs':>6} "
              f"{'Best IoU':>9} {'Best Dice':>10} "
              f"{'Best Val Total':>14} {'Best Val Seg':>12}")
    print(header)
    print("-" * len(header))

    for name, cfg in configs.items():
        history_path = base_dir / name / "training_history.json"
        history = load_history(history_path)

        layers_str = ", ".join(cfg["teacher_layers"])

        if not history:
            print(f"{name:<22} {layers_str:<40} {'--':>6} "
                  f"{'--':>9} {'--':>10} {'no data':>14} {'--':>12}")
            continue

        n_epochs = len(history)

        # IoU / Dice
        iou_vals = [e.get("val_iou", 0.0) for e in history]
        dice_vals = [e.get("val_dice", 0.0) for e in history]
        best_iou = max(iou_vals) if any(v > 0 for v in iou_vals) else 0.0
        best_dice = max(dice_vals) if any(v > 0 for v in dice_vals) else 0.0
        iou_str = f"{best_iou:.4f}" if best_iou > 0 else "N/A"
        dice_str = f"{best_dice:.4f}" if best_dice > 0 else "N/A"

        has_val = "val_total" in history[0]
        if has_val:
            best_entry = min(history, key=lambda e: e.get("val_total", float("inf")))
            best_val = best_entry["val_total"]
            best_seg = best_entry.get("val_segmentation", 0.0)
            print(f"{name:<22} {layers_str:<40} {n_epochs:>6} "
                  f"{iou_str:>9} {dice_str:>10} "
                  f"{best_val:>14.4f} {best_seg:>12.4f}")
        else:
            best_entry = min(history, key=lambda e: e.get("total", float("inf")))
            best_total = best_entry["total"]
            print(f"{name:<22} {layers_str:<40} {n_epochs:>6} "
                  f"{iou_str:>9} {dice_str:>10} "
                  f"{best_total:>14.4f} {'N/A':>12}")


def plot_comparison(configs: dict[str, dict], base_dir: Path, output: Path):
    """Plot IoU, Dice, and loss curves overlaid for all configs."""
    # Check which metrics are available across all configs
    any_iou = False
    for name in configs:
        history = load_history(base_dir / name / "training_history.json")
        if history and any(e.get("val_iou", 0) > 0 for e in history):
            any_iou = True
            break

    if any_iou:
        fig, (ax_iou, ax_dice, ax_total, ax_seg) = plt.subplots(
            4, 1, figsize=(14, 16))
    else:
        fig, (ax_total, ax_seg) = plt.subplots(2, 1, figsize=(14, 9))
        ax_iou = ax_dice = None

    has_any_data = False

    for name, cfg in configs.items():
        history_path = base_dir / name / "training_history.json"
        history = load_history(history_path)
        if not history:
            continue
        has_any_data = True

        epochs = [e["epoch"] for e in history]
        label = f"{name} ({', '.join(cfg['teacher_layers'])})"

        # IoU and Dice
        if ax_iou is not None:
            iou_vals = [e.get("val_iou", 0.0) for e in history]
            if any(v > 0 for v in iou_vals):
                ax_iou.plot(epochs, iou_vals, marker="o", markersize=3, label=label)

            dice_vals = [e.get("val_dice", 0.0) for e in history]
            if any(v > 0 for v in dice_vals):
                ax_dice.plot(epochs, dice_vals, marker="o", markersize=3, label=label)

        # Losses
        has_val = "val_total" in history[0]
        if has_val:
            val_totals = [e["val_total"] for e in history]
            ax_total.plot(epochs, val_totals, marker="o", markersize=3, label=label)

            val_segs = [e.get("val_segmentation", 0.0) for e in history]
            if any(v > 0 for v in val_segs):
                ax_seg.plot(epochs, val_segs, marker="s", markersize=3,
                           linestyle="--", label=label)
        else:
            totals = [e["total"] for e in history]
            ax_total.plot(epochs, totals, marker="o", markersize=3, label=label)

    if not has_any_data:
        print("No training data found for any config.")
        plt.close(fig)
        return

    if ax_iou is not None:
        ax_iou.set_xlabel("Epoch")
        ax_iou.set_ylabel("IoU")
        ax_iou.set_title("Validation IoU by Layer Configuration")
        ax_iou.legend(fontsize=7, loc="lower right")
        ax_iou.grid(True, alpha=0.3)
        ax_iou.set_ylim(0, 1)

        ax_dice.set_xlabel("Epoch")
        ax_dice.set_ylabel("Dice")
        ax_dice.set_title("Validation Dice by Layer Configuration")
        ax_dice.legend(fontsize=7, loc="lower right")
        ax_dice.grid(True, alpha=0.3)
        ax_dice.set_ylim(0, 1)

    ax_total.set_xlabel("Epoch")
    ax_total.set_ylabel("Loss")
    ax_total.set_title("Validation Total Loss by Layer Configuration")
    ax_total.legend(fontsize=7, loc="upper right")
    ax_total.grid(True, alpha=0.3)

    ax_seg.set_xlabel("Epoch")
    ax_seg.set_ylabel("Loss")
    ax_seg.set_title("Validation Segmentation Loss by Layer Configuration")
    ax_seg.legend(fontsize=7, loc="upper right")
    ax_seg.grid(True, alpha=0.3)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"\nSaved comparison plot to {output}")


def main():
    parser = argparse.ArgumentParser(description="Compare ablation training results")
    parser.add_argument("--base-dir", type=Path, required=True,
                        help="Base directory containing ablation output subdirectories "
                             "(e.g. trained_models/ablation)")
    parser.add_argument("--configs", type=Path,
                        default=Path("configs/ablation_configs.json"),
                        help="Path to ablation_configs.json")
    parser.add_argument("--output", type=Path,
                        default=Path("trained_models/ablation_comparison.png"),
                        help="Output plot path")
    args = parser.parse_args()

    configs = load_config(args.configs)
    print(f"Loaded {len(configs)} ablation configs\n")

    print_comparison_table(configs, args.base_dir)
    plot_comparison(configs, args.base_dir, args.output)


if __name__ == "__main__":
    main()
