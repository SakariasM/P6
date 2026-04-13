"""
Plot training loss curves from training_history.json.

Outputs a PNG image with train/val loss over epochs and a learning rate subplot.
Supports both old format (flat keys) and new format (train_/val_ prefixed keys).

Usage:
    python src/plot_training_history.py
    python src/plot_training_history.py --history trained_models/training_history.json
    python src/plot_training_history.py --output trained_models/loss_curves.png
    python src/plot_training_history.py --log-scale
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
    """Check if history uses train_/val_ prefixed keys."""
    return "val_total" in history[0]


def get_key(entry: dict, key: str, prefix: str = "") -> float:
    """Get a loss value, supporting both old and new key formats."""
    if prefix:
        return entry.get(f"{prefix}_{key}", 0.0)
    return entry.get(key, 0.0)


def plot(history: list[dict], output: Path, log_scale: bool):
    use_val = has_val(history)
    prefix = "train" if use_val else ""

    has_seg = any(get_key(e, "segmentation", prefix) > 0 for e in history)
    keys = [k for k in LOSS_KEYS if k != "segmentation" or has_seg]

    epochs = [e["epoch"] for e in history]

    if use_val:
        fig, (ax_train, ax_val, ax_lr) = plt.subplots(
            3, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 3, 1]})
    else:
        fig, (ax_train, ax_lr) = plt.subplots(
            2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})
        ax_val = None

    # Train losses
    for key in keys:
        values = [get_key(e, key, prefix) for e in history]
        ax_train.plot(epochs, values, marker="o", markersize=3, label=LOSS_LABELS[key])

    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Loss")
    ax_train.set_title("Training Loss")
    ax_train.legend()
    ax_train.grid(True, alpha=0.3)
    if log_scale:
        ax_train.set_yscale("log")

    # Val losses
    if use_val and ax_val is not None:
        for key in keys:
            values = [get_key(e, key, "val") for e in history]
            ax_val.plot(epochs, values, marker="s", markersize=3,
                        linestyle="--", label=f"{LOSS_LABELS[key]}")

        ax_val.set_xlabel("Epoch")
        ax_val.set_ylabel("Loss")
        ax_val.set_title("Validation Loss")
        ax_val.legend()
        ax_val.grid(True, alpha=0.3)
        if log_scale:
            ax_val.set_yscale("log")

    # Learning rate
    lrs = [e.get("lr", 0.0) for e in history]
    ax_lr.plot(epochs, lrs, color="tab:gray", marker="o", markersize=3)
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("Learning Rate")
    ax_lr.set_title("Learning Rate Schedule")
    ax_lr.grid(True, alpha=0.3)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot training loss curves.")
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
    parser.add_argument("--log-scale", action="store_true", help="Use log scale on loss axis")
    args = parser.parse_args()

    if not args.history.exists():
        raise FileNotFoundError(f"History file not found: {args.history}")

    history = load_history(args.history)
    plot(history, args.output, args.log_scale)

    final = history[-1]
    use_val = has_val(history)
    prefix = "train" if use_val else ""
    has_seg = any(get_key(e, "segmentation", prefix) > 0 for e in history)

    print(f"Saved plot to {args.output}")
    print(f"Latest epoch ({final['epoch']}):")
    print(f"  Train — total={get_key(final, 'total', prefix):.4f}"
          f"  att={get_key(final, 'attention', prefix):.4f}"
          f"  mim={get_key(final, 'mimicry', prefix):.4f}"
          f"  rel={get_key(final, 'relation', prefix):.4f}"
          + (f"  seg={get_key(final, 'segmentation', prefix):.4f}" if has_seg else ""))
    if use_val:
        print(f"  Val   — total={get_key(final, 'total', 'val'):.4f}"
              f"  att={get_key(final, 'attention', 'val'):.4f}"
              f"  mim={get_key(final, 'mimicry', 'val'):.4f}"
              f"  rel={get_key(final, 'relation', 'val'):.4f}"
              + (f"  seg={get_key(final, 'segmentation', 'val'):.4f}" if has_seg else ""))


if __name__ == "__main__":
    main()
