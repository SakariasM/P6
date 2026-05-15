"""
Compare training results across ablation configurations.

Reads training_history.json from each ablation output directory and produces:
1. Separate summary tables for original and scratch (no_cbam_enc0) runs
2. Separate overlay plots for each group

Directory conventions:
  - train_ablation.slurm      -> <base_dir>/<config>/
  - train_ablation_scratch.slurm -> <base_dir>/<config>_no_cbam_enc0_scratch/

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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


EPOCH_TICKS = [1, 5, 10, 20, 30, 40, 50]


def _setup_epoch_axis(ax):
    """Use sqrt scale on x-axis with fixed epoch tick marks."""
    ax.set_xscale("function", functions=(np.sqrt, np.square))
    ax.set_xticks(EPOCH_TICKS)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())


SCRATCH_SUFFIX = "_no_cbam_enc0_scratch"


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


def get_dir_for_config(base_dir: Path, name: str, scratch: bool) -> Path:
    if scratch:
        return base_dir / f"{name}{SCRATCH_SUFFIX}"
    return base_dir / name


def print_comparison_table(configs: dict[str, dict], base_dir: Path,
                           scratch: bool = False):
    header = (f"{'Config':<22} {'Layers':<40} {'Epochs':>6} "
              f"{'Best IoU':>9} {'Best Dice':>10} "
              f"{'Best Val Total':>14} {'Best Val Seg':>12}")
    print(header)
    print("-" * len(header))

    for name, cfg in configs.items():
        dir_path = get_dir_for_config(base_dir, name, scratch)
        history = load_history(dir_path / "training_history.json")

        layers_str = ", ".join(cfg["teacher_layers"])

        if not history:
            print(f"{name:<22} {layers_str:<40} {'--':>6} "
                  f"{'--':>9} {'--':>10} {'no data':>14} {'--':>12}")
            continue

        n_epochs = len(history)

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


def plot_comparison(configs: dict[str, dict], base_dir: Path, output: Path,
                    scratch: bool = False, title_suffix: str = ""):
    any_iou = False
    for name in configs:
        dir_path = get_dir_for_config(base_dir, name, scratch)
        history = load_history(dir_path / "training_history.json")
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
        dir_path = get_dir_for_config(base_dir, name, scratch)
        history = load_history(dir_path / "training_history.json")
        if not history:
            continue
        has_any_data = True

        epochs = [e["epoch"] for e in history]
        layers_display = [l.replace("model.", "block.") for l in cfg['teacher_layers']]
        label = f"{name} ({', '.join(layers_display)})"

        if ax_iou is not None:
            iou_vals = [e.get("val_iou", 0.0) for e in history]
            if any(v > 0 for v in iou_vals):
                ax_iou.plot(epochs, iou_vals, marker="o", markersize=2, label=label)

            dice_vals = [e.get("val_dice", 0.0) for e in history]
            if any(v > 0 for v in dice_vals):
                ax_dice.plot(epochs, dice_vals, marker="o", markersize=2, label=label)

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
        print(f"No training data found for any config{title_suffix}.")
        plt.close(fig)
        return

    if ax_iou is not None:
        ax_iou.set_xlabel("Epoch")
        ax_iou.set_ylabel("IoU")
        ax_iou.set_title(f"Validation IoU by Layer Configuration{title_suffix}")
        ax_iou.legend(fontsize=10, loc="lower right")
        ax_iou.grid(True, alpha=0.3)
        _setup_epoch_axis(ax_iou)
        ax_iou.set_ylim(0.5, 0.95)

        ax_dice.set_xlabel("Epoch")
        ax_dice.set_ylabel("Dice")
        ax_dice.set_title(f"Validation Dice by Layer Configuration{title_suffix}")
        ax_dice.legend(fontsize=10, loc="lower right")
        ax_dice.grid(True, alpha=0.3)
        _setup_epoch_axis(ax_dice)
        ax_dice.set_ylim(0.7, 0.98)

    ax_total.set_xlabel("Epoch")
    ax_total.set_ylabel("Loss")
    ax_total.set_title(f"Validation Total Loss by Layer Configuration{title_suffix}")
    ax_total.legend(fontsize=10, loc="upper right")
    ax_total.grid(True, alpha=0.3)
    _setup_epoch_axis(ax_total)

    ax_seg.set_xlabel("Epoch")
    ax_seg.set_ylabel("Loss")
    ax_seg.set_title(f"Validation Segmentation Loss by Layer Configuration{title_suffix}")
    ax_seg.legend(fontsize=10, loc="upper right")
    ax_seg.grid(True, alpha=0.3)
    _setup_epoch_axis(ax_seg)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"\nSaved comparison plot to {output}")

    fig_seg, ax_seg2 = plt.subplots(1, 1, figsize=(14, 5))
    for name, cfg in configs.items():
        dir_path = get_dir_for_config(base_dir, name, scratch)
        history = load_history(dir_path / "training_history.json")
        if not history:
            continue
        epochs = [e["epoch"] for e in history]
        layers_display = [l.replace("model.", "block.") for l in cfg['teacher_layers']]
        label = f"{name} ({', '.join(layers_display)})"
        has_val = "val_total" in history[0]
        if has_val:
            val_segs = [e.get("val_segmentation", 0.0) for e in history]
            if any(v > 0 for v in val_segs):
                ax_seg2.plot(epochs, val_segs, marker="s", markersize=3,
                             linestyle="--", label=label)
    ax_seg2.set_xlabel("Epoch")
    ax_seg2.set_ylabel("Loss")
    ax_seg2.set_title(f"Validation Segmentation Loss by Layer Configuration{title_suffix}")
    ax_seg2.legend(fontsize=10, loc="upper right")
    ax_seg2.grid(True, alpha=0.3)
    _setup_epoch_axis(ax_seg2)
    fig_seg.tight_layout()
    seg_path = output.with_stem(output.stem + "_seg_loss")
    fig_seg.savefig(seg_path, dpi=150)
    plt.close(fig_seg)
    print(f"Saved segmentation loss plot to {seg_path}")


def plot_top_bottom(configs: dict[str, dict], base_dir: Path, output: Path,
                    scratch: bool = False, title_suffix: str = "",
                    n_best: int = 3, n_worst: int = 2):
    rankings = []
    for name, cfg in configs.items():
        dir_path = get_dir_for_config(base_dir, name, scratch)
        history = load_history(dir_path / "training_history.json")
        if not history:
            continue
        iou_vals = [e.get("val_iou", 0.0) for e in history]
        best_iou = max(iou_vals) if any(v > 0 for v in iou_vals) else 0.0
        rankings.append((name, cfg, history, best_iou))

    if not rankings:
        return

    rankings.sort(key=lambda x: x[3], reverse=True)
    selected = rankings[:n_best] + rankings[-n_worst:]
    selected_names = {r[0] for r in selected}
    best_names = {r[0] for r in rankings[:n_best]}

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    colors_best = ["tab:blue", "tab:green", "tab:orange"]
    colors_worst = ["tab:red", "tab:brown"]

    best_idx = 0
    worst_idx = 0
    for name, cfg, history, best_iou in rankings:
        if name not in selected_names:
            continue
        epochs = [e["epoch"] for e in history]
        val_segs = [e.get("val_segmentation", 0.0) for e in history]
        if not any(v > 0 for v in val_segs):
            continue
        layers_display = [l.replace("model.", "block.") for l in cfg['teacher_layers']]
        label = f"{name} ({', '.join(layers_display)}) [IoU: {best_iou:.4f}]"
        if name in best_names:
            color = colors_best[best_idx % len(colors_best)]
            best_idx += 1
            linestyle = "-"
        else:
            color = colors_worst[worst_idx % len(colors_worst)]
            worst_idx += 1
            linestyle = "--"
        ax.plot(epochs, val_segs, marker="o", markersize=3,
                color=color, linestyle=linestyle, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Validation Segmentation Loss — Top {n_best} vs Bottom {n_worst} by IoU{title_suffix}")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    _setup_epoch_axis(ax)
    fig.tight_layout()
    top_path = output.with_stem(output.stem + "_top_bottom")
    fig.savefig(top_path, dpi=150)
    plt.close(fig)
    print(f"Saved top/bottom plot to {top_path}")


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

    # --- Original (train_ablation.slurm) ---
    print("=" * 80)
    print("ORIGINAL (train_ablation.slurm) — full CBAM")
    print("=" * 80)
    print_comparison_table(configs, args.base_dir, scratch=False)
    original_output = args.output.with_stem(args.output.stem + "_original")
    plot_comparison(configs, args.base_dir, original_output,
                    scratch=False, title_suffix=" (Original)")
    plot_top_bottom(configs, args.base_dir, original_output,
                    scratch=False, title_suffix=" (Original)")

    # --- Scratch (train_ablation_scratch.slurm) ---
    print()
    print("=" * 80)
    print("SCRATCH (train_ablation_scratch.slurm) — no_cbam_enc0")
    print("=" * 80)
    print_comparison_table(configs, args.base_dir, scratch=True)
    scratch_output = args.output.with_stem(args.output.stem + "_scratch")
    plot_comparison(configs, args.base_dir, scratch_output,
                    scratch=True, title_suffix=" (Scratch, no_cbam_enc0)")
    plot_top_bottom(configs, args.base_dir, scratch_output,
                    scratch=True, title_suffix=" (Scratch, no_cbam_enc0)")


if __name__ == "__main__":
    main()
