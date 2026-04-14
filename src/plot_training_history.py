"""
Plot training curves from training_history.json.

Default view shows the most useful metrics: val IoU/Dice and val total loss.
Use flags to add individual loss components or training losses.

Usage:
    python src/plot_training_history.py
    python src/plot_training_history.py --history trained_models/training_history.json
    python src/plot_training_history.py --all-losses
    python src/plot_training_history.py --train-losses --log-scale
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOSS_KEYS = ["total", "attention", "mimicry", "relation", "segmentation"]
LOSS_LABELS = {
    "total": "Total",
    "attention": "Attention transfer",
    "mimicry": "Feature mimicry",
    "relation": "Relation (Gram)",
    "segmentation": "Segmentation (BCE+Dice)",
}


def load_history(path: Path) -> list[dict]:
    with open(path) as f:
        history = json.load(f)
    if not history:
        raise ValueError(f"{path} is empty — no epochs recorded yet.")
    history.sort(key=lambda e: e["epoch"])
    return history


def has_val(history: list[dict]) -> bool:
    return "val_total" in history[0]


def get_key(entry: dict, key: str, prefix: str = "") -> float:
    if prefix:
        return entry.get(f"{prefix}_{key}", 0.0)
    return entry.get(key, 0.0)


def plot(history: list[dict], output: Path, args):
    use_val = has_val(history)
    prefix = "train" if use_val else ""
    epochs = [e["epoch"] for e in history]

    has_iou = any(e.get("val_iou", 0) > 0 for e in history)
    has_seg = any(get_key(e, "segmentation", prefix) > 0 for e in history)

    # Determine layout based on what to show
    panels = []
    if has_iou:
        panels.append("iou")
    panels.append("val_loss")
    if args.all_losses or args.train_losses:
        panels.append("train_loss")
    if args.all_losses:
        panels.append("lr")

    ratios = [2 if p != "lr" else 1 for p in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3.5 * len(panels)),
                              gridspec_kw={"height_ratios": ratios})
    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        if panel == "iou":
            iou_vals = [e.get("val_iou", 0.0) for e in history]
            dice_vals = [e.get("val_dice", 0.0) for e in history]
            ax.plot(epochs, iou_vals, marker="o", markersize=4,
                    color="tab:blue", label="Val IoU")
            ax.plot(epochs, dice_vals, marker="s", markersize=4,
                    color="tab:green", label="Val Dice")
            best_idx = max(range(len(iou_vals)), key=lambda i: iou_vals[i])
            ax.annotate(f"Best IoU: {iou_vals[best_idx]:.4f} (ep {epochs[best_idx]})",
                        xy=(epochs[best_idx], iou_vals[best_idx]),
                        xytext=(10, -15), textcoords="offset points",
                        fontsize=8, color="tab:blue",
                        arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8))
            ax.set_ylabel("Score")
            ax.set_title("Validation Segmentation Quality (IoU / Dice)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

        elif panel == "val_loss":
            keys = [k for k in LOSS_KEYS if k != "segmentation" or has_seg]
            if not args.all_losses:
                keys = ["total"] + (["segmentation"] if has_seg else [])
            for key in keys:
                vals = [get_key(e, key, "val" if use_val else "") for e in history]
                ax.plot(epochs, vals, marker="o", markersize=3,
                        label=LOSS_LABELS[key])
            ax.set_ylabel("Loss")
            ax.set_title("Validation Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            if args.log_scale:
                ax.set_yscale("log")

        elif panel == "train_loss":
            keys = [k for k in LOSS_KEYS if k != "segmentation" or has_seg]
            for key in keys:
                vals = [get_key(e, key, prefix) for e in history]
                ax.plot(epochs, vals, marker="o", markersize=3,
                        label=LOSS_LABELS[key])
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            if args.log_scale:
                ax.set_yscale("log")

        elif panel == "lr":
            lrs = [e.get("lr", 0.0) for e in history]
            ax.plot(epochs, lrs, color="tab:gray", marker="o", markersize=3)
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule")
            ax.grid(True, alpha=0.3)

        ax.set_xlabel("Epoch")

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot training curves.")
    parser.add_argument(
        "--history", type=Path,
        default=Path("trained_models/training_history.json"),
        help="Path to training_history.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("trained_models/loss_curves.png"),
        help="Output image path (.png, .pdf, .svg)",
    )
    parser.add_argument("--log-scale", action="store_true",
                        help="Use log scale on loss axes")
    parser.add_argument("--all-losses", action="store_true",
                        help="Show all individual loss components, LR schedule")
    parser.add_argument("--train-losses", action="store_true",
                        help="Add training loss panel")
    args = parser.parse_args()

    if not args.history.exists():
        raise FileNotFoundError(f"History file not found: {args.history}")

    history = load_history(args.history)
    plot(history, args.output, args)

    final = history[-1]
    use_val = has_val(history)
    prefix = "train" if use_val else ""

    print(f"Saved plot to {args.output}")
    print(f"Latest epoch ({final['epoch']}):")
    if final.get("val_iou"):
        print(f"  Val IoU: {final['val_iou']:.4f}  Dice: {final['val_dice']:.4f}")
    if use_val:
        print(f"  Val   — total={get_key(final, 'total', 'val'):.4f}"
              f"  seg={get_key(final, 'segmentation', 'val'):.4f}")
    print(f"  Train — total={get_key(final, 'total', prefix):.4f}"
          f"  seg={get_key(final, 'segmentation', prefix):.4f}")


if __name__ == "__main__":
    main()
