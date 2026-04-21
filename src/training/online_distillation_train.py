"""
Online segmentation knowledge distillation training.

Runs teacher (YOLO-seg) and student (U-Net + CBAM) simultaneously.
The teacher is frozen (eval, no_grad, FP16) and produces features live
for each augmented batch, unlocking full spatial augmentation.

Multi-GPU via DistributedDataParallel + torchrun.

Usage (single GPU):
    python -m training.online_distillation_train \
        --teacher-model yolo26n-seg.pt --image-dir /path/to/images

Usage (8 GPU):
    torchrun --nproc_per_node=8 -m training.online_distillation_train \
        --teacher-model yolo26n-seg.pt --image-dir /path/to/images
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import argparse
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from teacher.feature_extractor import YOLOFeatureExtractor
from student.student_model import StudentSegmentation
from training.distillation_loss import SegmentationDistillationLoss
from training.hybrid_distillation_train import (
    compute_teacher_attention,
    select_teacher_layers,
)

from PIL import Image
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# DDP utilities
# ---------------------------------------------------------------------------

def setup_distributed() -> Tuple[int, int, int]:
    """Initialize distributed process group from torchrun env vars.

    Returns (rank, local_rank, world_size).  When torchrun is not used
    (single-GPU), returns (0, 0, 1) and skips init_process_group.
    """
    if "RANK" not in os.environ:
        return 0, 0, 1

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OnlineImageDataset(Dataset):
    """Image-only dataset for online distillation (no pre-extracted features).

    Discovers images from a directory or a text file listing paths.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(
        self,
        image_dir: Optional[str] = None,
        image_list: Optional[str] = None,
        target_size: int = 640,
        augment: bool = True,
    ):
        super().__init__()
        self.paths: List[Path] = []

        if image_list:
            with open(image_list) as f:
                for line in f:
                    p = Path(line.strip())
                    if p.suffix.lower() in self.EXTENSIONS:
                        self.paths.append(p)
        elif image_dir:
            root = Path(image_dir)
            for ext in self.EXTENSIONS:
                self.paths.extend(root.rglob(f"*{ext}"))
            self.paths.sort()
        else:
            raise ValueError("Provide either --image-dir or --image-list")

        if not self.paths:
            raise FileNotFoundError(
                f"No images found (dir={image_dir}, list={image_list})"
            )

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    target_size, scale=(0.5, 1.0), ratio=(0.75, 1.33),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1,
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return {"image": image, "image_path": str(path)}


# ---------------------------------------------------------------------------
# Online teacher wrapper
# ---------------------------------------------------------------------------

class OnlineTeacher:
    """Wraps a frozen YOLO-seg model for live feature extraction.

    The teacher stays in eval mode with no_grad.  Features are extracted
    via forward hooks (``YOLOFeatureExtractor``).  Segmentation masks are
    extracted on demand via the higher-level ``model.predict()`` API.
    """

    def __init__(
        self,
        model_path: str,
        feature_layers: Optional[List[str]],
        device: torch.device,
        half: bool = True,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        person_class: int = 0,
    ):
        from ultralytics import YOLO

        self.device = device
        self.half = half and device.type == "cuda"
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_class = person_class

        self.model = YOLO(model_path)
        self.model.to(device)

        self.extractor = YOLOFeatureExtractor(
            self.model,
            feature_layers=feature_layers,
            device=str(device),
        )

        # Put underlying model into half precision if requested
        if self.half:
            self.extractor.pytorch_model.half()

    # -- feature shapes (for probing) -----------------------------------

    def get_feature_shapes(
        self, input_size: Tuple[int, int] = (640, 640),
    ) -> Dict[str, Tuple]:
        return self.extractor.get_feature_shapes(input_size)

    @property
    def layer_names(self) -> List[str]:
        return list(self.extractor.hooks.keys())

    # -- live feature extraction ----------------------------------------

    @torch.no_grad()
    def forward_features(
        self, images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run teacher forward and return intermediate features (float32).

        Args:
            images: [B, 3, H, W] float32 on *this* device.
        Returns:
            dict  ``{layer_name: [B, C, h, w]}`` in float32.
        """
        inp = images.half() if self.half else images
        result = self.extractor.extract_features(inp, return_predictions=False)
        # Cast back to float32 for loss computation
        return {
            name: feat.float()
            for name, feat in result["features"].items()
        }

    # -- live segmentation mask extraction ------------------------------

    @torch.no_grad()
    def extract_seg_masks(
        self, images: torch.Tensor, target_size: Tuple[int, int] = (640, 640),
    ) -> torch.Tensor:
        """Run full YOLO predict and merge person instance masks.

        Args:
            images: [B, 3, H, W] float32, values in [0, 1].
        Returns:
            [B, 1, H, W] float32 binary masks.
        """
        import cv2

        B, _, H, W = images.shape
        masks_out = torch.zeros(B, 1, *target_size, device=images.device)

        # Ultralytics predict() accepts tensors directly (0-1 float)
        results = self.model.predict(
            source=images,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            save=False,
            verbose=False,
            half=self.half,
        )

        for i, result in enumerate(results):
            if result.masks is None or result.boxes is None:
                continue

            cls = result.boxes.cls.cpu().numpy().astype(int)
            masks_xy = result.masks.xy  # list of polygon arrays

            binary = np.zeros((H, W), dtype=np.uint8)
            for j, c in enumerate(cls):
                if c != self.person_class:
                    continue
                if j >= len(masks_xy):
                    continue
                pts = masks_xy[j].astype(np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(binary, [pts], 1)

            mask_t = torch.from_numpy(binary).float().unsqueeze(0)  # [1, H, W]
            if (H, W) != target_size:
                mask_t = F.interpolate(
                    mask_t.unsqueeze(0), size=target_size,
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
            masks_out[i] = mask_t

        return masks_out

    def close(self):
        self.extractor.close()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(args):
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_distributed = world_size > 1

    if is_main(rank):
        print(f"World size: {world_size}, device: {device}\n")

    # ---- Teacher ----------------------------------------------------------
    if is_main(rank):
        print("Loading teacher model...")
    teacher = OnlineTeacher(
        model_path=args.teacher_model,
        feature_layers=None,  # register hooks on all default layers
        device=device,
        half=True,
        person_class=0,
    )

    # Probe teacher feature shapes
    shapes = teacher.get_feature_shapes((args.img_size, args.img_size))
    if is_main(rank):
        print("Teacher feature shapes:")
        for name, shape in shapes.items():
            print(f"  {name}: {shape}")

    # Determine which layers and channel counts to use (exclude if requested)
    if args.exclude_layers:
        unknown = [l for l in args.exclude_layers if l not in shapes]
        if unknown and is_main(rank):
            print(f"Warning: --exclude-layers contains unknown layers: {unknown}")
        filtered_shapes = {k: v for k, v in shapes.items()
                          if k not in args.exclude_layers}
        if not filtered_shapes:
            raise ValueError("All teacher layers were excluded — nothing left to train on")
        if is_main(rank):
            print(f"Excluded teacher layers: {args.exclude_layers}")
    else:
        filtered_shapes = shapes

    pseudo_feats = {name: torch.empty(shape) for name, shape in filtered_shapes.items()}
    teacher_layer_names, teacher_channels = select_teacher_layers(
        pseudo_feats, num_scales=len(pseudo_feats),
    )
    if is_main(rank):
        print(f"Using teacher layers: {teacher_layer_names}")
        print(f"Teacher channels: {teacher_channels}\n")

    # ---- Student ----------------------------------------------------------
    if is_main(rank):
        print("Creating student model...")
    model = StudentSegmentation(
        in_channels=3,
        base_channels=args.base_channels,
        depth=args.depth,
        teacher_channels=teacher_channels,
    ).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    if is_main(rank):
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Student parameters: {total_params:,}\n")

    # ---- Loss, optimizer, scheduler, scaler --------------------------------
    criterion = SegmentationDistillationLoss(
        attention_weight=args.attention_weight,
        mimicry_weight=args.mimicry_weight,
        relation_weight=args.relation_weight,
        seg_weight=args.seg_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # ---- Dataset + loader --------------------------------------------------
    dataset = OnlineImageDataset(
        image_dir=args.image_dir,
        image_list=args.image_list,
        target_size=args.img_size,
        augment=not args.no_augment,
    )
    if is_main(rank):
        print(f"Dataset: {len(dataset)} images")

    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Resume ------------------------------------------------------------
    start_epoch = 1
    best_loss = float("inf")
    history: List[dict] = []

    if args.resume and Path(args.resume).exists():
        if is_main(rank):
            print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"]
        # Strip DDP "module." prefix if present
        if any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        target = model.module if isinstance(model, DDP) else model
        target.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", float("inf"))
        history = ckpt.get("history", [])
        if is_main(rank):
            print(f"Resumed at epoch {start_epoch}, best loss {best_loss:.4f}\n")
    elif args.resume:
        if is_main(rank):
            print(f"Warning: --resume path '{args.resume}' not found, starting fresh.\n")

    # ---- Output dir --------------------------------------------------------
    output_dir = Path(args.output_dir)
    if is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training loop -----------------------------------------------------
    if is_main(rank):
        print(f"Training epochs {start_epoch}..{args.epochs}, "
              f"batch {args.batch_size}x{world_size}={args.batch_size * world_size} "
              f"effective\n")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_losses = {
            "attention": 0.0, "mimicry": 0.0,
            "relation": 0.0, "segmentation": 0.0, "total": 0.0,
        }
        epoch_batches = 0

        progress = (
            tqdm(loader, desc=f"Epoch {epoch}") if is_main(rank) else loader
        )

        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)

            # --- Teacher forward (frozen, FP16) ---
            teacher_feat_dict = teacher.forward_features(images)
            teacher_feats = [teacher_feat_dict[n] for n in teacher_layer_names]
            teacher_atts = [compute_teacher_attention(f) for f in teacher_feats]

            teacher_mask = None
            if args.seg_weight > 0:
                teacher_mask = teacher.extract_seg_masks(
                    images, target_size=(args.img_size, args.img_size),
                )

            # --- Student forward (AMP) ---
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                seg_output, distill_info = model(images)
                projected = distill_info["projected"]
                student_atts = distill_info["attention_maps"]

                n = len(projected)
                t_feats = teacher_feats[-n:]
                t_atts = teacher_atts[-n:]

                loss, loss_dict = criterion(
                    student_atts=student_atts,
                    teacher_atts=t_atts,
                    projected_student_feats=projected,
                    teacher_feats=t_feats,
                    student_mask=seg_output,
                    teacher_mask=teacher_mask,
                )

            scaler.scale(loss).backward()

            if args.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            for k in epoch_losses:
                epoch_losses[k] += loss_dict.get(k, 0.0)
            epoch_batches += 1

            if is_main(rank):
                postfix = {
                    "loss": f"{loss_dict['total']:.4f}",
                    "att": f"{loss_dict['attention']:.4f}",
                    "mim": f"{loss_dict['mimicry']:.4f}",
                }
                if "segmentation" in loss_dict:
                    postfix["seg"] = f"{loss_dict['segmentation']:.4f}"
                progress.set_postfix(postfix)

        scheduler.step()

        # --- Epoch metrics (rank 0) ---
        if not is_main(rank):
            continue

        metrics = {k: v / max(epoch_batches, 1) for k, v in epoch_losses.items()}

        seg_str = (
            f"  Seg: {metrics['segmentation']:.4f}"
            if metrics.get("segmentation", 0) > 0 else ""
        )
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Total: {metrics['total']:.4f}  Att: {metrics['attention']:.4f}"
              f"  Mim: {metrics['mimicry']:.4f}  Rel: {metrics['relation']:.4f}{seg_str}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}  Batches: {epoch_batches}")

        history.append({"epoch": epoch, **metrics, "lr": optimizer.param_groups[0]["lr"]})

        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        raw_state = (
            model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        )

        if metrics["total"] < best_loss:
            best_loss = metrics["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "history": history,
                "teacher_channels": teacher_channels,
                "teacher_layer_names": teacher_layer_names,
                "args": vars(args),
            }, output_dir / "best_model.pt")
            print(f"  -> Saved best model (loss: {best_loss:.4f})")

        if epoch % args.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": metrics["total"],
                "history": history,
                "teacher_channels": teacher_channels,
                "teacher_layer_names": teacher_layer_names,
                "args": vars(args),
            }, output_dir / f"checkpoint_epoch_{epoch}.pt")

    # --- Final save (rank 0) ------------------------------------------------
    if is_main(rank):
        raw_state = (
            model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        )
        torch.save({
            "model_state_dict": raw_state,
            "teacher_channels": teacher_channels,
            "teacher_layer_names": teacher_layer_names,
            "args": vars(args),
        }, output_dir / "final_model.pt")

        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"\nTraining complete!")
        print(f"Models saved to: {output_dir}")
        print(f"Best loss: {best_loss:.4f}")

    teacher.close()
    cleanup_distributed()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Online Segmentation Knowledge Distillation Training",
    )

    # Data
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory containing training images")
    parser.add_argument("--image-list", type=str, default=None,
                        help="Text file listing image paths (one per line)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Input image size")

    # Teacher
    parser.add_argument("--teacher-model", type=str, default="yolo26n-seg.pt",
                        help="Path to YOLO teacher model weights")
    parser.add_argument("--exclude-layers", type=str, nargs="+", default=None,
                        help="Teacher layer names to EXCLUDE from distillation "
                             "(e.g. model.9 to skip the deepest layer). "
                             "All available layers are used by default.")

    # Student architecture
    parser.add_argument("--base-channels", type=int, default=32,
                        help="Base channel count for U-Net encoder")
    parser.add_argument("--depth", type=int, default=4,
                        help="Number of encoder/decoder levels")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="Gradient clipping max norm")

    # Distillation loss weights
    parser.add_argument("--attention-weight", type=float, default=1.0)
    parser.add_argument("--mimicry-weight", type=float, default=0.5)
    parser.add_argument("--relation-weight", type=float, default=0.5)
    parser.add_argument("--seg-weight", type=float, default=0.0,
                        help="Segmentation loss weight (BCE+Dice vs teacher mask). "
                             "0 = disabled (skips teacher predict() call).")

    # Augmentation
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable augmentation")

    # Output
    parser.add_argument("--output-dir", type=str, default="./trained_models")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume from")

    args = parser.parse_args()
    main(args)
