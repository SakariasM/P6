"""
Plot training curves from training_history.json.

Single model:
    python src/plot_training_history.py
    python src/plot_training_history.py --history trained_models/training_history.json
    python src/plot_training_history.py --all-losses --weight-change 9.5

Compare ablation configs (overlay all models on one graph):
    python src/plot_training_history.py --compare \
        --ablation-dir trained_models/ablation

    python src/plot_training_history.py --compare \
        --ablation-dir /ceph/project/P6-Machine-Vision/P6/trained_models/ablation \
        --output trained_models/ablation_curves.png
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


# ---------------------------------------------------------------------------
# Single-model plot
# ---------------------------------------------------------------------------
def plot_single(history: list[dict], output: Path, args):
    use_val = has_val(history)
    prefix = "train" if use_val else ""
    epochs = [e["epoch"] for e in history]

    has_iou = any(e.get("val_iou", 0) > 0 for e in history)
    has_seg = any(get_key(e, "segmentation", prefix) > 0 for e in history)

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
            ax.set_ylim(0.5, 1)

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

        if args.weight_change:
            for wc in args.weight_change:
                ax.axvline(x=wc, color="red", linestyle="--", linewidth=1, alpha=0.7)
                if panel == panels[0]:
                    ax.text(wc + 0.15, ax.get_ylim()[1] * 0.95,
                            "weight change", fontsize=7, color="red",
                            va="top")

        ax.set_xlabel("Epoch")

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-model comparison plot
# ---------------------------------------------------------------------------
def plot_compare(ablation_dir: Path, output: Path, args):
    # Scan subdirectories for training_history.json
    model_data = {}
    for subdir in sorted(ablation_dir.iterdir()):
        if not subdir.is_dir():
            continue
        history_path = subdir / "training_history.json"
        if not history_path.exists():
            continue
        history = load_history(history_path)
        if not history:
            continue
        model_data[subdir.name] = {
            "history": history,
            "label": subdir.name,
        }

    if not model_data:
        print(f"No training_history.json found in subdirectories of {ablation_dir}")
        return

    print(f"Found data for {len(model_data)} configs: {', '.join(model_data.keys())}")

    # Check which metrics are available
    any_iou = any(
        any(e.get("val_iou", 0) > 0 for e in d["history"])
        for d in model_data.values()
    )
    any_val = any(has_val(d["history"]) for d in model_data.values())

    # Determine panels
    panels = []
    if any_iou:
        panels.append("iou")
        panels.append("dice")
    if any_val:
        panels.append("val_total")
        panels.append("val_seg")

    if not panels:
        panels = ["train_total"]

    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 4 * len(panels)))
    if len(panels) == 1:
        axes = [axes]

    # Use a colormap with enough distinct colors
    colors = plt.cm.tab10.colors
    if len(model_data) > 10:
        colors = plt.cm.tab20.colors

    for ax, panel in zip(axes, panels):
        for i, (name, data) in enumerate(model_data.items()):
            history = data["history"]
            label = data["label"]
            color = colors[i % len(colors)]
            epochs = [e["epoch"] for e in history]

            if panel == "iou":
                vals = [e.get("val_iou", 0.0) for e in history]
                if any(v > 0 for v in vals):
                    ax.plot(epochs, vals, marker="o", markersize=3,
                            color=color, label=label)
                ax.set_ylabel("IoU")
                ax.set_title("Validation IoU by Configuration")

            elif panel == "dice":
                vals = [e.get("val_dice", 0.0) for e in history]
                if any(v > 0 for v in vals):
                    ax.plot(epochs, vals, marker="o", markersize=3,
                            color=color, label=label)
                ax.set_ylabel("Dice")
                ax.set_title("Validation Dice by Configuration")

            elif panel == "val_total":
                use_val = has_val(history)
                key = "val" if use_val else ""
                vals = [get_key(e, "total", key) for e in history]
                ax.plot(epochs, vals, marker="o", markersize=3,
                        color=color, label=label)
                ax.set_ylabel("Loss")
                ax.set_title("Validation Total Loss by Configuration")

            elif panel == "val_seg":
                use_val = has_val(history)
                key = "val" if use_val else ""
                vals = [get_key(e, "segmentation", key) for e in history]
                if any(v > 0 for v in vals):
                    ax.plot(epochs, vals, marker="o", markersize=3,
                            color=color, label=label)
                ax.set_ylabel("Loss")
                ax.set_title("Validation Segmentation Loss by Configuration")

            elif panel == "train_total":
                prefix = "train" if has_val(history) else ""
                vals = [get_key(e, "total", prefix) for e in history]
                ax.plot(epochs, vals, marker="o", markersize=3,
                        color=color, label=label)
                ax.set_ylabel("Loss")
                ax.set_title("Training Total Loss by Configuration")

        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Epoch")
        if args.log_scale:
            ax.set_yscale("log")

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {output}")

    # Print summary table
    print(f"\n{'Config':<22} {'Epochs':>6} {'Best IoU':>9} {'Best Dice':>10} {'Best Val Loss':>14}")
    print("-" * 65)
    for name, data in model_data.items():
        history = data["history"]
        n_epochs = len(history)
        iou_vals = [e.get("val_iou", 0.0) for e in history]
        dice_vals = [e.get("val_dice", 0.0) for e in history]
        best_iou = max(iou_vals) if any(v > 0 for v in iou_vals) else 0.0
        best_dice = max(dice_vals) if any(v > 0 for v in dice_vals) else 0.0
        use_val = has_val(history)
        key = "val" if use_val else ""
        best_loss = min(get_key(e, "total", key) for e in history)
        iou_str = f"{best_iou:.4f}" if best_iou > 0 else "N/A"
        dice_str = f"{best_dice:.4f}" if best_dice > 0 else "N/A"
        print(f"{name:<22} {n_epochs:>6} {iou_str:>9} {dice_str:>10} {best_loss:>14.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves.")
    parser.add_argument(
        "--history", type=Path,
        default=Path("trained_models/training_history.json"),
        help="Path to training_history.json (single-model mode)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output image path (.png, .pdf, .svg)",
    )
    parser.add_argument("--log-scale", action="store_true",
                        help="Use log scale on loss axes")
    parser.add_argument("--all-losses", action="store_true",
                        help="Show all individual loss components, LR schedule")
    parser.add_argument("--train-losses", action="store_true",
                        help="Add training loss panel")
    parser.add_argument("--weight-change", type=float, nargs="+", default=None,
                        metavar="EPOCH",
                        help="Mark epoch(s) where loss weights changed "
                             "(draws vertical line, e.g. --weight-change 9.5)")

    # Compare mode
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple ablation configs on one graph")
    parser.add_argument("--ablation-dir", type=Path,
                        default=Path("trained_models/ablation"),
                        help="Base directory with ablation subdirectories")

    args = parser.parse_args()

    if args.compare:
        output = args.output or Path("trained_models/ablation_curves.png")
        plot_compare(args.ablation_dir, output, args)
    else:
        output = args.output or Path("trained_models/loss_curves.png")
        if not args.history.exists():
            raise FileNotFoundError(f"History file not found: {args.history}")

        history = load_history(args.history)
        plot_single(history, output, args)

        final = history[-1]
        use_val = has_val(history)
        prefix = "train" if use_val else ""

        print(f"Saved plot to {output}")
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
