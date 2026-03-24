"""
Checkpoint utilities for the distillation trainer.

Kept separate so trainer.py stays focused on the training loop.
"""

import os
import torch


def save_checkpoint(trainer, path: str):
    """Save student weights, optimizer state, and training metadata.

    Args:
        trainer : DistillationTrainer instance
        path    : Destination .pth file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "student_state_dict":    trainer.student.state_dict(),
            "optimizer_state_dict":  trainer.optimizer.state_dict(),
            "epoch":                 trainer.current_epoch,
            "global_step":           trainer.global_step,
            "config":                trainer.config,
        },
        path,
    )
    print(f"[checkpoint] Saved → {path}")


def load_checkpoint(trainer, path: str) -> int:
    """Load student weights and optimizer state from a checkpoint.

    Args:
        trainer : DistillationTrainer instance
        path    : Source .pth file path

    Returns:
        start_epoch (int): The epoch to resume from.
    """
    ckpt = torch.load(path, map_location=trainer.device)
    trainer.student.load_state_dict(ckpt["student_state_dict"])
    trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    trainer.global_step = ckpt.get("global_step", 0)
    start_epoch = ckpt.get("epoch", 0) + 1
    print(f"[checkpoint] Loaded ← {path}  (resuming from epoch {start_epoch})")
    return start_epoch
