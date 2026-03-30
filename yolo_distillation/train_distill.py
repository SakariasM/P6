"""
Distillation Training Entry Point

Usage:
    python yolo_distillation/train_distill.py
    python yolo_distillation/train_distill.py --config yolo_distillation/config.yaml
    python yolo_distillation/train_distill.py --resume checkpoints/distillation/best.pth
"""

import argparse
import os
import sys

import torch
import yaml

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo_distillation.teacher import YOLOTeacher
from yolo_distillation.student import StudentGenerator
from yolo_distillation.trainer import DistillationTrainer
from yolo_distillation._checkpoint import load_checkpoint


def build_dataloader(config: dict, split: str):
    """Reuse the existing InpaintingDataset from data/dataset.py."""
    from data.dataset import InpaintingDataset
    from torch.utils.data import DataLoader

    data_cfg = config.get("data", {})
    dataset = InpaintingDataset(
        data_dir=data_cfg[f"{split}_dir"],
        image_size=data_cfg.get("image_size", 256),
        augment=(split == "train"),
    )
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )


def build_optimizer(student, config: dict):
    training = config["training"]
    lr = training.get("learning_rate", 1e-4)
    wd = training.get("weight_decay",  1e-5)
    name = training.get("optimizer", "adam").lower()
    if name == "adamw":
        return torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=wd)
    return torch.optim.Adam(student.parameters(), lr=lr, weight_decay=wd)


def build_scheduler(optimizer, config: dict):
    training = config["training"]
    name = training.get("scheduler", "cosine").lower()
    epochs = training.get("num_epochs", 100)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    return None   # "none"


def main():
    parser = argparse.ArgumentParser(description="YOLO Attention Distillation")
    parser.add_argument(
        "--config", default="yolo_distillation/config.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to a distillation checkpoint to resume from"
    )
    parser.add_argument(
        "--device", default=None,
        help="Force a specific device, e.g. 'cpu' or 'cuda:1'"
    )
    args = parser.parse_args()

    # ---- Config ----------------------------------------------------------
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[distill] Using device: {device}")

    # ---- Teacher ---------------------------------------------------------
    t_cfg = config["teacher"]
    print(f"[distill] Loading YOLO teacher: {t_cfg['model_name']}")
    teacher = YOLOTeacher(
        model_name=t_cfg["model_name"],
        pretrained=t_cfg.get("pretrained", True),
        hook_layers=t_cfg.get("hook_layers", None),
    ).to(device)

    # ---- Student ---------------------------------------------------------
    s_cfg = config["student"]
    print("[distill] Building student generator")
    student = StudentGenerator(
        config=s_cfg,
        teacher_channels=s_cfg["teacher_channels"],
    ).to(device)

    total_params = sum(p.numel() for p in student.parameters())
    train_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[distill] Student params: {total_params:,} total, {train_params:,} trainable")

    # ---- Optimiser + scheduler -------------------------------------------
    optimizer = build_optimizer(student, config)
    scheduler = build_scheduler(optimizer, config)

    # ---- TensorBoard (optional) ------------------------------------------
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir="runs/distillation")
    except ImportError:
        print("[distill] TensorBoard not available, skipping writer")

    # ---- Trainer ---------------------------------------------------------
    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        optimizer=optimizer,
        config=config,
        device=device,
        writer=writer,
        scheduler=scheduler,
    )

    # ---- Resume ----------------------------------------------------------
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(trainer, args.resume)

    # ---- Data ------------------------------------------------------------
    print("[distill] Building data loaders")
    train_loader = build_dataloader(config, "train")
    val_loader   = build_dataloader(config, "val")

    # ---- Train -----------------------------------------------------------
    num_epochs = config["training"]["num_epochs"]
    print(f"[distill] Starting training for {num_epochs} epochs")
    trainer.train(train_loader, val_loader, num_epochs, start_epoch)

    if writer is not None:
        writer.close()

    print("[distill] Training complete.")


if __name__ == "__main__":
    main()
