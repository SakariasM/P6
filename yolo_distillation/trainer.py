"""
Distillation Trainer

Orchestrates the full training loop:
  1. Teacher (frozen YOLO) extracts multi-scale features + attention maps
     from the clean ground-truth image.
  2. Student generator produces an inpainted image AND exposes intermediate
     features + CBAM attention maps from the masked input.
  3. DistillationLoss combines reconstruction + attention transfer + feature
     mimicry + relational distillation terms.
  4. Only the student (and its projection layers) are updated.

Expected batch dict keys:
    "masked_image"  : [B, 3, H, W] masked input, in [-1, 1]
    "mask"          : [B, 1, H, W] binary mask  (1 = hole)
    "ground_truth"  : [B, 3, H, W] clean target, in [-1, 1]
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from yolo_distillation.distillation_loss import DistillationLoss


def _to_01(x: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] image tensor to [0, 1] for the YOLO teacher."""
    return (x + 1.0) / 2.0


class DistillationTrainer:
    """Training loop for attention-based distillation.

    Args:
        teacher     : YOLOTeacher (already on the correct device, always frozen)
        student     : StudentGenerator
        optimizer   : torch.optim.Optimizer (for student parameters only)
        config      : dict loaded from config.yaml
        device      : torch.device
        writer      : TensorBoard SummaryWriter (or None)
    """

    def __init__(self, teacher, student, optimizer, config, device, writer=None,
                 scheduler=None):
        self.teacher   = teacher
        self.student   = student
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config    = config
        self.device    = device
        self.writer    = writer

        self.loss_fn = DistillationLoss(config).to(device)

        training_cfg = config.get("training", {})
        self.log_interval  = training_cfg.get("log_interval",  50)
        self.val_interval  = training_cfg.get("val_interval",  1)
        self.save_interval = training_cfg.get("save_interval", 5)
        self.grad_clip     = training_cfg.get("gradient_clip_max_norm", None)

        self.global_step   = 0
        self.current_epoch = 0

        # Teacher is always frozen and in eval mode
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Single step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict) -> dict:
        """One forward + backward pass.

        Args:
            batch: dict with keys "masked_image", "mask", "ground_truth"

        Returns:
            loss_dict: {"reconstruction", "attention", "mimicry", "relation", "total"}
        """
        masked_img = batch["masked_image"].to(self.device)
        mask       = batch["mask"].to(self.device)
        target     = batch["ground_truth"].to(self.device)

        # ---- Teacher: use clean target image (no mask) -------------------
        # Teacher processes [0, 1] RGB; target is in [-1, 1]
        with torch.no_grad():
            teacher_out = self.teacher(_to_01(target))
        teacher_feats = teacher_out["features"]       # list of tensors
        teacher_atts  = teacher_out["attention_maps"] # list of tensors

        # ---- Student forward ---------------------------------------------
        self.optimizer.zero_grad()
        student_out, distill_info = self.student(masked_img, mask)

        projected = distill_info["projected"]
        student_atts = distill_info["attention_maps"]

        # Align number of scales: student exposes n_align scales,
        # teacher exposes num_scales().  Use the last n_align teacher scales
        # (coarsest → most semantic, matching the deepest student encoder levels).
        n = len(projected)
        teacher_feats_aligned = teacher_feats[-n:]
        teacher_atts_aligned  = teacher_atts[-n:]

        # ---- Losses ------------------------------------------------------
        total_loss, loss_dict = self.loss_fn(
            student_out=student_out,
            target=target,
            student_atts=student_atts,
            teacher_atts=teacher_atts_aligned,
            projected_student_feats=projected,
            teacher_feats=teacher_feats_aligned,
        )

        # ---- Backward ----------------------------------------------------
        total_loss.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), self.grad_clip
            )

        self.optimizer.step()

        return loss_dict

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, val_loader) -> dict:
        """Compute average losses over the validation set.

        Returns:
            dict of averaged losses.
        """
        self.student.eval()
        accum = {}
        n_batches = 0

        for batch in tqdm(val_loader, desc="Validating", leave=False):
            masked_img = batch["masked_image"].to(self.device)
            mask       = batch["mask"].to(self.device)
            target     = batch["ground_truth"].to(self.device)

            teacher_out  = self.teacher(_to_01(target))
            teacher_feats = teacher_out["features"]
            teacher_atts  = teacher_out["attention_maps"]

            student_out, distill_info = self.student(masked_img, mask)
            projected    = distill_info["projected"]
            student_atts = distill_info["attention_maps"]

            n = len(projected)
            _, loss_dict = self.loss_fn(
                student_out=student_out,
                target=target,
                student_atts=student_atts,
                teacher_atts=teacher_atts[-n:],
                projected_student_feats=projected,
                teacher_feats=teacher_feats[-n:],
            )

            for k, v in loss_dict.items():
                accum[k] = accum.get(k, 0.0) + v
            n_batches += 1

        self.student.train()
        return {k: v / n_batches for k, v in accum.items()}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, train_loader, val_loader, num_epochs: int,
              start_epoch: int = 0):
        """Run the full training loop.

        Args:
            train_loader : DataLoader yielding the standard batch dicts
            val_loader   : DataLoader for validation
            num_epochs   : Total number of epochs
            start_epoch  : Epoch to resume from (for checkpoint restart)
        """
        import os
        from yolo_distillation._checkpoint import save_checkpoint

        self.current_epoch = start_epoch
        best_val_loss = float("inf")

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            self.student.train()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch in pbar:
                loss_dict = self.train_step(batch)
                self.global_step += 1

                if self.global_step % self.log_interval == 0:
                    if self.writer is not None:
                        for k, v in loss_dict.items():
                            self.writer.add_scalar(
                                f"distill/train_{k}", v, self.global_step
                            )
                    pbar.set_postfix({
                        "total": f"{loss_dict['total']:.4f}",
                        "att":   f"{loss_dict['attention']:.4f}",
                        "mim":   f"{loss_dict['mimicry']:.4f}",
                    })

            # Validation
            if (epoch + 1) % self.val_interval == 0:
                val_losses = self.validate(val_loader)
                if self.writer is not None:
                    for k, v in val_losses.items():
                        self.writer.add_scalar(f"distill/val_{k}", v, epoch)
                print(
                    f"Epoch {epoch + 1} | "
                    f"val_total={val_losses['total']:.4f}  "
                    f"att={val_losses['attention']:.4f}  "
                    f"mim={val_losses['mimicry']:.4f}  "
                    f"rel={val_losses['relation']:.4f}"
                )

                if val_losses["total"] < best_val_loss:
                    best_val_loss = val_losses["total"]
                    save_checkpoint(self, "checkpoints/distillation/best.pth")

            if (epoch + 1) % self.save_interval == 0:
                save_checkpoint(
                    self,
                    f"checkpoints/distillation/epoch_{epoch + 1}.pth"
                )

            if self.scheduler is not None:
                self.scheduler.step()
