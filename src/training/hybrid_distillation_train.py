"""
Offline segmentation knowledge distillation training.

Uses pre-extracted teacher predictions (features + masks) from chunk files.
Trains a U-Net student with CBAM attention using chunk-epoch loading.
Losses: attention transfer + feature mimicry + Gram-matrix relation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import random
from tqdm import tqdm
import argparse

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from teacher.hybrid_predictions import HybridTeacherPrediction
from student.student_model import StudentSegmentation
from training.distillation_loss import SegmentationDistillationLoss
from PIL import Image
import torchvision.transforms as transforms


def select_teacher_layers(features: Dict[str, torch.Tensor],
                          num_scales: int = 3,
                          explicit_layers: Optional[List[str]] = None
                          ) -> Tuple[List[str], List[int]]:
    """Select teacher feature layers for distillation.

    Picks `num_scales` layers sorted by layer index, taking the last N (deepest).
    If `explicit_layers` is provided, those layers are used directly in the given
    order, ignoring `num_scales`.

    Args:
        features: Dict of {layer_name: tensor} from one prediction
        num_scales: How many scales to use (ignored when explicit_layers is set)
        explicit_layers: Optional list of layer names to use explicitly

    Returns:
        (layer_names, channel_counts) ordered fine to coarse
    """
    if explicit_layers is not None:
        names = []
        channels = []
        for name in explicit_layers:
            if name not in features:
                raise KeyError(
                    f"Layer '{name}' not in prediction features. "
                    f"Available: {list(features.keys())}"
                )
            names.append(name)
            channels.append(features[name].shape[0])
        return names, channels

    layers = []
    for name, tensor in features.items():
        parts = name.split(".")
        idx = int(parts[-1]) if parts[-1].isdigit() else 0
        channels = tensor.shape[0]  # stored as [C, H, W] per-image
        layers.append((idx, name, channels))

    layers.sort(key=lambda x: x[0])

    selected = layers[-num_scales:] if len(layers) > num_scales else layers
    names = [l[1] for l in selected]
    channels = [l[2] for l in selected]
    return names, channels


def compute_teacher_attention(feat: torch.Tensor) -> torch.Tensor:
    """Compute spatial attention from teacher features.

    Mean of absolute activations across channels, min-max normalized to [0,1].

    Args:
        feat: [B, C, H, W] or [C, H, W]
    Returns:
        [B, 1, H, W] attention map
    """
    if feat.dim() == 3:
        feat = feat.unsqueeze(0)
    att = feat.abs().mean(dim=1, keepdim=True)
    b = att.shape[0]
    flat = att.view(b, -1)
    mn = flat.min(1)[0].view(b, 1, 1, 1)
    mx = flat.max(1)[0].view(b, 1, 1, 1)
    return (att - mn) / (mx - mn + 1e-8)


class ChunkDataset(Dataset):
    """Dataset wrapping a single chunk of predictions."""

    def __init__(self, predictions: List[HybridTeacherPrediction],
                 teacher_layer_names: List[str],
                 image_root: Optional[str] = None,
                 target_size: Tuple[int, int] = (640, 640),
                 augment: bool = False):
        self.predictions = predictions
        self.teacher_layer_names = teacher_layer_names
        self.image_root = Path(image_root) if image_root else None
        self.target_size = target_size
        self.augment = augment
        self.base_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
        # Color augmentations only affect the image, not teacher features
        self.color_augment = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        )

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, idx):
        pred = self.predictions[idx]

        image_path = Path(pred.image_path)
        if self.image_root and not image_path.is_absolute():
            image_path = self.image_root / image_path
        pil_image = Image.open(image_path).convert('RGB')

        # Color augmentation before toTensor (student sees varied colors,
        # but must still match the same teacher features)
        if self.augment:
            pil_image = self.color_augment(pil_image)

        image = self.base_transform(pil_image)

        teacher_feats = []
        for name in self.teacher_layer_names:
            if name in pred.features:
                teacher_feats.append(pred.features[name])
            else:
                raise KeyError(f"Layer {name} not in prediction. Available: {list(pred.features.keys())}")

        # Teacher segmentation mask (if available)
        teacher_mask = None
        if pred.segmentation_mask is not None:
            teacher_mask = pred.segmentation_mask.float().unsqueeze(0)  # [1, H, W]
            # Resize to match target_size
            teacher_mask = F.interpolate(
                teacher_mask.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False
            ).squeeze(0)  # [1, H, W]

        # Random horizontal flip — applied to image, teacher features, and mask
        if self.augment and random.random() > 0.5:
            image = torch.flip(image, dims=[-1])
            teacher_feats = [torch.flip(f, dims=[-1]) for f in teacher_feats]
            if teacher_mask is not None:
                teacher_mask = torch.flip(teacher_mask, dims=[-1])

        result = {
            'image': image,
            'teacher_features': teacher_feats,
        }
        if teacher_mask is not None:
            result['teacher_mask'] = teacher_mask
        return result


def collate_fn(batch):
    """Custom collate that stacks images and teacher feature lists."""
    images = torch.stack([b['image'] for b in batch])
    num_scales = len(batch[0]['teacher_features'])
    teacher_feats = []
    for s in range(num_scales):
        teacher_feats.append(torch.stack([b['teacher_features'][s] for b in batch]))
    result = {
        'image': images,
        'teacher_features': teacher_feats,
    }
    if 'teacher_mask' in batch[0]:
        result['teacher_mask'] = torch.stack([b['teacher_mask'] for b in batch])
    return result


def discover_chunk_files(predictions_path: str) -> List[Path]:
    """Discover prediction files -- supports chunk directory or single file."""
    pred_path = Path(predictions_path)

    if pred_path.is_file():
        return [pred_path]

    if pred_path.is_dir():
        chunks = sorted(pred_path.glob("chunk_*.torch"))
        for f in sorted(pred_path.glob("hybrid_teacher_predictions_worker*.torch")):
            chunks.append(f)
        merged = pred_path / "hybrid_teacher_predictions.torch"
        if merged.exists():
            chunks.append(merged)
        if chunks:
            return chunks

    raise FileNotFoundError(f"No prediction files found at {predictions_path}")


def validate(model, val_chunks, teacher_layer_names, teacher_channels,
             criterion, device, args):
    """Run validation on held-out chunks. Returns average loss dict + IoU/Dice."""
    model.eval()
    val_losses = {"attention": 0.0, "mimicry": 0.0, "relation": 0.0,
                  "total": 0.0, "segmentation": 0.0}
    iou_sum = 0.0
    dice_sum = 0.0
    mask_batches = 0
    val_batches = 0

    with torch.no_grad():
        for chunk_path in val_chunks:
            preds = torch.load(chunk_path, weights_only=False)
            preds = [p for p in preds if p.features]
            if not preds:
                continue

            chunk_dataset = ChunkDataset(
                preds, teacher_layer_names,
                image_root=args.image_root,
                target_size=(args.img_size, args.img_size),
                augment=False,  # No augmentation for validation
            )
            chunk_loader = DataLoader(
                chunk_dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                collate_fn=collate_fn, pin_memory=True,
            )

            for batch in chunk_loader:
                images = batch['image'].to(device)
                teacher_feats = [f.to(device, dtype=torch.float32)
                                 for f in batch['teacher_features']]
                teacher_atts = [compute_teacher_attention(f) for f in teacher_feats]

                seg_output, distill_info = model(images)
                projected = distill_info['projected']
                student_atts = distill_info['attention_maps']

                n = len(projected)
                t_feats = teacher_feats[-n:]
                t_atts = teacher_atts[-n:]

                teacher_mask = batch.get('teacher_mask')
                if teacher_mask is not None:
                    teacher_mask = teacher_mask.to(device, dtype=torch.float32)

                _, loss_dict = criterion(
                    student_atts=student_atts, teacher_atts=t_atts,
                    projected_student_feats=projected, teacher_feats=t_feats,
                    student_mask=seg_output, teacher_mask=teacher_mask,
                )

                for k in val_losses:
                    val_losses[k] += loss_dict.get(k, 0.0)
                val_batches += 1

                # Compute IoU and Dice when masks are available
                if teacher_mask is not None:
                    pred_bin = (seg_output > 0.5).float()
                    intersection = (pred_bin * teacher_mask).sum()
                    union = pred_bin.sum() + teacher_mask.sum() - intersection
                    iou_sum += (intersection / (union + 1e-6)).item()
                    dice_sum += (2 * intersection / (pred_bin.sum() + teacher_mask.sum() + 1e-6)).item()
                    mask_batches += 1

            del preds, chunk_dataset, chunk_loader
            torch.cuda.empty_cache()

    model.train()
    result = {k: v / max(val_batches, 1) for k, v in val_losses.items()}
    if mask_batches > 0:
        result['iou'] = iou_sum / mask_batches
        result['dice'] = dice_sum / mask_batches
    return result


def main(args):
    """Main training function with chunk-epoch training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Using device: {device}, GPUs available: {n_gpus}\n")

    # Discover chunk files and split into train/val
    all_chunks = discover_chunk_files(args.predictions)
    print(f"Found {len(all_chunks)} prediction file(s)")

    n_val = max(1, int(len(all_chunks) * args.val_split))
    # Deterministic split: last N chunks are validation
    val_chunks = all_chunks[-n_val:]
    chunk_files = all_chunks[:-n_val]
    print(f"Train chunks: {len(chunk_files)}, Val chunks: {len(val_chunks)}")

    # Load first chunk to determine teacher feature layout
    print("Loading first chunk to determine teacher feature shapes...")
    first_preds = torch.load(chunk_files[0], weights_only=False)
    first_preds = [p for p in first_preds if p.features]
    if not first_preds:
        raise RuntimeError("First chunk has no predictions with features")

    available_features = first_preds[0].features

    if args.exclude_layers:
        # Remove excluded layers from available features before selection
        unknown = [l for l in args.exclude_layers if l not in available_features]
        if unknown:
            print(f"Warning: --exclude-layers contains unknown layers: {unknown}")
        filtered = {k: v for k, v in available_features.items()
                    if k not in args.exclude_layers}
        if not filtered:
            raise ValueError("All teacher layers were excluded — nothing left to train on")
        print(f"Excluded teacher layers: {args.exclude_layers}")
        teacher_layer_names, teacher_channels = select_teacher_layers(
            filtered, num_scales=len(filtered)
        )
    else:
        teacher_layer_names, teacher_channels = select_teacher_layers(
            available_features, num_scales=len(available_features)
        )
    print(f"Using teacher layers: {teacher_layer_names}")

    print(f"Teacher channels: {teacher_channels}")
    del first_preds

    # Create student model
    print("\nCreating student segmentation model...")
    model = StudentSegmentation(
        in_channels=3,
        base_channels=args.base_channels,
        depth=args.depth,
        teacher_channels=teacher_channels,
    )
    model = model.to(device)

    if n_gpus > 1:
        print(f"Using DataParallel across {n_gpus} GPUs")
        model = torch.nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    # Loss function
    criterion = SegmentationDistillationLoss(
        attention_weight=args.attention_weight,
        mimicry_weight=args.mimicry_weight,
        relation_weight=args.relation_weight,
        seg_weight=args.seg_weight,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Resume
    start_epoch = 1
    best_loss = float('inf')
    best_iou = 0.0
    history = []
    epochs_no_improve = 0

    resume_chunk_idx = 0
    resume_epoch_losses = None

    if args.resume and Path(args.resume).exists():
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('module.') for k in state_dict):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        target = model.module if isinstance(model, torch.nn.DataParallel) else model
        target.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
        best_iou = 0.0
        if 'history' in checkpoint:
            history = checkpoint['history']
            # Recover best IoU from history
            iou_values = [e.get('val_iou', 0.0) for e in history]
            if iou_values:
                best_iou = max(iou_values)

        if args.reset_best_loss:
            best_loss = float('inf')
            best_iou = 0.0
            epochs_no_improve = 0
            print("Reset best_loss and best_iou (--reset-best-loss)")

        # Mid-epoch resume: checkpoint saved partway through an epoch
        if checkpoint.get('chunk_idx') is not None:
            start_epoch = checkpoint['epoch']  # resume same epoch
            resume_chunk_idx = checkpoint['chunk_idx'] + 1
            resume_epoch_losses = checkpoint.get('epoch_losses')
            print(f"Resumed mid-epoch {checkpoint['epoch']} at chunk {resume_chunk_idx}/{len(chunk_files)}, best loss: {best_loss:.4f}\n")
        else:
            print(f"Resumed from epoch {checkpoint['epoch']}, best loss: {best_loss:.4f}, best IoU: {best_iou:.4f}\n")
    elif args.resume:
        print(f"Warning: --resume path '{args.resume}' not found, starting from scratch.\n")

    # Training
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting training from epoch {start_epoch} to {args.epochs}...")
    print(f"Training on {len(chunk_files)} chunks (one at a time)\n")

    mid_epoch_interval = 50  # save mid-epoch checkpoint every N chunks

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()

        # Seeded shuffle so chunk order is reproducible on resume
        rng = random.Random(epoch)
        shuffled_chunks = chunk_files.copy()
        rng.shuffle(shuffled_chunks)

        epoch_losses = {"attention": 0.0, "mimicry": 0.0, "relation": 0.0, "total": 0.0, "segmentation": 0.0}
        epoch_batches = 0

        # Restore accumulated losses when resuming mid-epoch
        start_chunk = 0
        if resume_chunk_idx > 0 and epoch == start_epoch:
            start_chunk = resume_chunk_idx
            if resume_epoch_losses:
                epoch_losses = resume_epoch_losses
                epoch_batches = resume_epoch_losses.get('_batches', 0)
            print(f"Skipping to chunk {start_chunk}/{len(shuffled_chunks)}")

        for chunk_idx, chunk_file in enumerate(shuffled_chunks):
            if chunk_idx < start_chunk:
                continue

            preds = torch.load(chunk_file, weights_only=False)
            preds = [p for p in preds if p.features]

            if not preds:
                del preds
                continue

            chunk_dataset = ChunkDataset(
                preds, teacher_layer_names,
                image_root=args.image_root,
                target_size=(args.img_size, args.img_size),
                augment=args.augment,
            )
            chunk_loader = DataLoader(
                chunk_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )

            desc = f"Epoch {epoch} chunk {chunk_idx+1}/{len(shuffled_chunks)}"
            progress_bar = tqdm(chunk_loader, desc=desc)

            for batch in progress_bar:
                images = batch['image'].to(device)
                teacher_feats = [f.to(device, dtype=torch.float32) for f in batch['teacher_features']]

                # Compute teacher attention from pre-extracted features
                teacher_atts = [compute_teacher_attention(f) for f in teacher_feats]

                # Student forward
                optimizer.zero_grad()
                seg_output, distill_info = model(images)

                projected = distill_info['projected']
                student_atts = distill_info['attention_maps']

                # Align scales
                n = len(projected)
                t_feats = teacher_feats[-n:]
                t_atts = teacher_atts[-n:]

                # Loss
                teacher_mask = batch.get('teacher_mask')
                if teacher_mask is not None:
                    teacher_mask = teacher_mask.to(device, dtype=torch.float32)
                loss, loss_dict = criterion(
                    student_atts=student_atts,
                    teacher_atts=t_atts,
                    projected_student_feats=projected,
                    teacher_feats=t_feats,
                    student_mask=seg_output,
                    teacher_mask=teacher_mask,
                )

                loss.backward()

                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()

                for k in epoch_losses:
                    epoch_losses[k] += loss_dict.get(k, 0.0)
                epoch_batches += 1

                postfix = {
                    'loss': f"{loss_dict['total']:.4f}",
                    'att': f"{loss_dict['attention']:.4f}",
                    'mim': f"{loss_dict['mimicry']:.4f}",
                }
                if 'segmentation' in loss_dict:
                    postfix['seg'] = f"{loss_dict['segmentation']:.4f}"
                progress_bar.set_postfix(postfix)

            del preds, chunk_dataset, chunk_loader
            torch.cuda.empty_cache()

            # Mid-epoch checkpoint
            if (chunk_idx + 1) % mid_epoch_interval == 0:
                raw_state = (model.module.state_dict()
                             if isinstance(model, torch.nn.DataParallel)
                             else model.state_dict())
                save_losses = {**epoch_losses, '_batches': epoch_batches}
                torch.save({
                    'epoch': epoch,
                    'chunk_idx': chunk_idx,
                    'model_state_dict': raw_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': best_loss,
                    'epoch_losses': save_losses,
                    'history': history,
                    'teacher_channels': teacher_channels,
                    'teacher_layer_names': teacher_layer_names,
                    'args': vars(args),
                }, output_dir / 'checkpoint_mid_epoch.pt')
                print(f"  [mid-epoch checkpoint saved at chunk {chunk_idx+1}]")

        # Reset mid-epoch resume state after first epoch completes
        resume_chunk_idx = 0
        resume_epoch_losses = None

        scheduler.step()

        # Epoch metrics (strip internal _batches counter from mid-epoch resume)
        epoch_losses.pop('_batches', None)
        metrics = {k: v / max(epoch_batches, 1) for k, v in epoch_losses.items()}

        print(f"\nEpoch {epoch}/{args.epochs}")
        seg_str = f"  Seg: {metrics['segmentation']:.4f}" if metrics.get('segmentation', 0) > 0 else ""
        print(f"  Train — Total: {metrics['total']:.4f}  Att: {metrics['attention']:.4f}"
              f"  Mim: {metrics['mimicry']:.4f}  Rel: {metrics['relation']:.4f}{seg_str}")

        # Validation
        val_metrics = validate(model, val_chunks, teacher_layer_names,
                               teacher_channels, criterion, device, args)
        val_seg_str = f"  Seg: {val_metrics['segmentation']:.4f}" if val_metrics.get('segmentation', 0) > 0 else ""
        val_iou_str = f"  IoU: {val_metrics['iou']:.4f}  Dice: {val_metrics['dice']:.4f}" if 'iou' in val_metrics else ""
        print(f"  Val   — Total: {val_metrics['total']:.4f}  Att: {val_metrics['attention']:.4f}"
              f"  Mim: {val_metrics['mimicry']:.4f}  Rel: {val_metrics['relation']:.4f}{val_seg_str}{val_iou_str}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}  Batches: {epoch_batches}")

        history_entry = {
            'epoch': epoch,
            **{f'train_{k}': v for k, v in metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
        }
        history.append(history_entry)

        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        raw_state = (model.module.state_dict()
                     if isinstance(model, torch.nn.DataParallel)
                     else model.state_dict())

        # Track best model by validation IoU (falls back to val_total if no masks)
        val_iou = val_metrics.get('iou', 0.0)
        improved = False
        if val_iou > 0 and val_iou > best_iou:
            best_iou = val_iou
            improved = True
        elif val_iou == 0 and val_metrics['total'] < best_loss:
            improved = True
        if improved:
            best_loss = val_metrics['total']
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'best_iou': best_iou,
                'history': history,
                'teacher_channels': teacher_channels,
                'teacher_layer_names': teacher_layer_names,
                'args': vars(args),
            }, output_dir / 'best_model.pt')
            iou_str = f", IoU: {best_iou:.4f}" if best_iou > 0 else ""
            print(f"  -> Saved best model (val_loss: {best_loss:.4f}{iou_str})")
        else:
            epochs_no_improve += 1
            print(f"  No val improvement for {epochs_no_improve}/{args.patience} epochs")

        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': metrics['total'],
                'history': history,
                'teacher_channels': teacher_channels,
                'teacher_layer_names': teacher_layer_names,
                'args': vars(args),
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')

        # Remove mid-epoch checkpoint once the full epoch is saved
        mid_ckpt = output_dir / 'checkpoint_mid_epoch.pt'
        if mid_ckpt.exists():
            mid_ckpt.unlink()

        # Early stopping
        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"\nEarly stopping: val loss did not improve for {args.patience} epochs.")
            break

    # Save final model
    raw_state = (model.module.state_dict()
                 if isinstance(model, torch.nn.DataParallel)
                 else model.state_dict())
    torch.save({
        'model_state_dict': raw_state,
        'teacher_channels': teacher_channels,
        'teacher_layer_names': teacher_layer_names,
        'args': vars(args),
    }, output_dir / 'final_model.pt')

    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Models saved to: {output_dir}")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Knowledge Distillation Training")

    # Data
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions directory or file")
    parser.add_argument("--image-root", type=str, default=None,
                        help="Root directory for images")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Input image size")

    # Teacher layer selection
    parser.add_argument("--exclude-layers", type=str, nargs="+", default=None,
                        help="Teacher layer names to EXCLUDE from distillation "
                             "(e.g. model.9 to skip the deepest layer). "
                             "All available layers are used by default.")

    # Student architecture
    parser.add_argument("--base-channels", type=int, default=32,
                        help="Base channel count for U-Net encoder")
    parser.add_argument("--depth", type=int, default=4,
                        help="Number of encoder/decoder levels")
    parser.add_argument("--teacher-layers", type=str, nargs="+", default=None,
                        help="Explicit teacher layer names to use (e.g. model.4 model.9). "
                             "Default: auto-select last 3 by layer index.")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping max norm")

    # Distillation loss weights
    parser.add_argument("--attention-weight", type=float, default=1.0)
    parser.add_argument("--mimicry-weight", type=float, default=2.0)
    parser.add_argument("--relation-weight", type=float, default=1.0)
    parser.add_argument("--seg-weight", type=float, default=0.5,
                        help="Segmentation loss weight (BCE+Dice vs teacher mask). 0 = disabled.")
    parser.add_argument("--augment", action="store_true",
                        help="Enable data augmentation (color jitter + random flip)")

    # Validation / early stopping
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of chunks to hold out for validation (default: 0.1)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (epochs without val improvement). 0 = disabled.")
    parser.add_argument("--reset-best-loss", action="store_true",
                        help="Reset best_loss on resume (use after changing loss weights)")

    # Output
    parser.add_argument("--output-dir", type=str, default="./trained_models")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume from")

    args = parser.parse_args()
    main(args)
