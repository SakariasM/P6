"""
Hybrid Knowledge Distillation Training.
Combines response-based (logit) distillation and feature-based (intermediate layer) distillation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm
import argparse

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from teacher.hybrid_predictions import load_hybrid_predictions, HybridTeacherPrediction
from student.student_model import StudentYOLO
from PIL import Image
import torchvision.transforms as transforms


class HybridDistillationDataset(Dataset):
    """
    Dataset for hybrid knowledge distillation.
    Loads images with teacher predictions and intermediate features.
    """

    def __init__(
        self,
        predictions_file: str,
        image_root: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (640, 640)
    ):
        """
        Args:
            predictions_file: Path to hybrid predictions (.pt file)
            image_root: Root directory for images
            transform: Image transforms
            target_size: Target image size (W, H)
        """
        self.predictions = load_hybrid_predictions(predictions_file)
        self.image_root = Path(image_root) if image_root else None
        self.target_size = target_size

        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # Filter out images without features
        self.predictions = [p for p in self.predictions if p.features]

        print(f"Loaded {len(self.predictions)} samples with teacher features")

    def __len__(self) -> int:
        return len(self.predictions)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dictionary with:
                - image: Input image tensor
                - teacher_features: Dict of teacher feature maps
                - teacher_logits: Teacher raw logits (if available)
                - boxes: Ground truth boxes
                - class_probs: Soft class probabilities
        """
        prediction = self.predictions[idx]

        # Load image
        image_path = Path(prediction.image_path)
        if self.image_root and not image_path.is_absolute():
            image_path = self.image_root / image_path

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Prepare teacher features
        teacher_features = {
            k: v for k, v in prediction.features.items()
        }

        sample = {
            'image': image,
            'teacher_features': teacher_features,
            'image_path': str(image_path)
        }

        # Add logits if available
        if prediction.raw_logits is not None:
            sample['teacher_logits'] = prediction.raw_logits

        # Add detection labels
        if len(prediction.boxes) > 0:
            sample['boxes'] = torch.tensor(prediction.boxes, dtype=torch.float32)
            sample['class_probs'] = torch.tensor(prediction.class_probs, dtype=torch.float32)
            sample['confidences'] = torch.tensor(prediction.confidences, dtype=torch.float32)

        return sample


class HybridDistillationLoss(nn.Module):
    """
    Combined loss for hybrid knowledge distillation.
    Includes both feature-based and response-based components.
    """

    def __init__(
        self,
        feature_weight: float = 1.0,
        response_weight: float = 1.0,
        temperature: float = 3.0,
        feature_distance: str = "mse"  # "mse", "cosine", or "at"
    ):
        """
        Args:
            feature_weight: Weight for feature-based loss
            response_weight: Weight for response-based (logit) loss
            temperature: Temperature for softening logits
            feature_distance: Distance metric for features
                - "mse": Mean squared error
                - "cosine": Cosine similarity
                - "at": Attention transfer (spatial attention maps)
        """
        super().__init__()

        self.feature_weight = feature_weight
        self.response_weight = response_weight
        self.temperature = temperature
        self.feature_distance = feature_distance

        print(f"Hybrid Distillation Loss initialized:")
        print(f"  Feature weight: {feature_weight}")
        print(f"  Response weight: {response_weight}")
        print(f"  Temperature: {temperature}")
        print(f"  Feature distance: {feature_distance}")

    def forward(
        self,
        student_output: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        teacher_logits: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hybrid distillation loss.

        Args:
            student_output: Dict with 'predictions', 'features', 'adapted_features'
            teacher_features: Dict of teacher intermediate features
            teacher_logits: Teacher output logits (optional)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        losses = {}
        total_loss = 0.0

        # 1. Feature-based distillation loss
        if 'adapted_features' in student_output and teacher_features:
            feature_loss = self._compute_feature_loss(
                student_output['adapted_features'],
                teacher_features
            )
            losses['feature_loss'] = feature_loss.item()
            total_loss += self.feature_weight * feature_loss

        # 2. Response-based distillation loss (logits)
        if teacher_logits is not None and 'predictions' in student_output:
            response_loss = self._compute_response_loss(
                student_output['predictions'],
                teacher_logits
            )
            losses['response_loss'] = response_loss.item()
            total_loss += self.response_weight * response_loss

        losses['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

        return total_loss, losses

    def _compute_feature_loss(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute feature-based distillation loss.
        Matches intermediate representations between student and teacher.
        """
        feature_losses = []

        # Match features from adapted student layers to teacher layers
        for student_name, student_feat in student_features.items():
            # Find corresponding teacher feature
            # student_name format: "stage1_to_model_4" (dots replaced with underscores for nn.ModuleDict)
            adapter_suffix = student_name.split('_to_')[-1] if '_to_' in student_name else None
            if adapter_suffix is None:
                continue

            # Try both dotted and underscored versions to find the teacher layer
            teacher_layer = None
            for key in teacher_features:
                if key == adapter_suffix or key.replace('.', '_') == adapter_suffix:
                    teacher_layer = key
                    break

            if teacher_layer and teacher_layer in teacher_features:
                teacher_feat = teacher_features[teacher_layer]

                # Ensure same device
                teacher_feat = teacher_feat.to(student_feat.device)

                # Resize if necessary
                if student_feat.shape[2:] != teacher_feat.shape[2:]:
                    teacher_feat = F.interpolate(
                        teacher_feat,
                        size=student_feat.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )

                # Compute distance based on selected metric
                if self.feature_distance == "mse":
                    loss = F.mse_loss(student_feat, teacher_feat)

                elif self.feature_distance == "cosine":
                    # Cosine similarity loss
                    student_norm = F.normalize(student_feat, p=2, dim=1)
                    teacher_norm = F.normalize(teacher_feat, p=2, dim=1)
                    loss = 1.0 - F.cosine_similarity(
                        student_norm.view(student_norm.size(0), -1),
                        teacher_norm.view(teacher_norm.size(0), -1)
                    ).mean()

                elif self.feature_distance == "at":
                    # Attention Transfer: match spatial attention maps
                    student_attn = self._compute_attention_map(student_feat)
                    teacher_attn = self._compute_attention_map(teacher_feat)
                    loss = F.mse_loss(student_attn, teacher_attn)

                else:
                    loss = F.mse_loss(student_feat, teacher_feat)

                feature_losses.append(loss)

        if feature_losses:
            return sum(feature_losses) / len(feature_losses)
        else:
            return torch.tensor(0.0)

    def _compute_attention_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial attention map from feature map.
        Used for Attention Transfer (AT) distillation.
        """
        # Sum across channel dimension and normalize
        attention = torch.sum(feature_map ** 2, dim=1, keepdim=True)
        attention = F.normalize(attention.view(attention.size(0), -1), p=2, dim=1)
        attention = attention.view(attention.size(0), 1, feature_map.size(2), feature_map.size(3))
        return attention

    def _compute_response_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute response-based distillation loss (logit matching).
        Uses KL divergence with temperature softening.
        """
        # Ensure same device
        teacher_logits = teacher_logits.to(student_logits.device)

        # Resize if necessary
        if student_logits.shape != teacher_logits.shape:
            # For detection, might need more sophisticated matching
            # Here we use simple interpolation
            if len(teacher_logits.shape) == 4:  # [B, C, H, W]
                teacher_logits = F.interpolate(
                    teacher_logits,
                    size=student_logits.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

        # Apply temperature and compute KL divergence
        T = self.temperature

        # Softmax with temperature
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)

        # KL divergence
        kl_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (T * T)  # Scale by T^2

        return kl_loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: HybridDistillationLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    feature_loss_sum = 0.0
    response_loss_sum = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        images = batch['image'].to(device)
        teacher_features = {
            k: v.to(device) for k, v in batch['teacher_features'].items()
        }
        teacher_logits = batch.get('teacher_logits')
        if teacher_logits is not None:
            teacher_logits = teacher_logits.to(device)

        # Forward pass
        optimizer.zero_grad()

        student_output = model(images, return_features=True)

        # Compute loss
        loss, loss_dict = criterion(
            student_output,
            teacher_features,
            teacher_logits
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss_dict['total_loss']
        feature_loss_sum += loss_dict.get('feature_loss', 0.0)
        response_loss_sum += loss_dict.get('response_loss', 0.0)
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss_dict['total_loss']:.4f}",
            'feat': f"{loss_dict.get('feature_loss', 0.0):.4f}",
            'resp': f"{loss_dict.get('response_loss', 0.0):.4f}"
        })

    return {
        'total_loss': total_loss / num_batches,
        'feature_loss': feature_loss_sum / num_batches,
        'response_loss': response_loss_sum / num_batches
    }


def main(args):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create dataset
    print("Loading dataset...")
    dataset = HybridDistillationDataset(
        predictions_file=args.predictions,
        image_root=args.image_root,
        target_size=(args.img_size, args.img_size)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}\n")

    # Get teacher feature shapes from first sample
    sample = dataset[0]
    teacher_shapes = {
        k: (1, *v.shape) for k, v in sample['teacher_features'].items()
    }

    # Create student model
    print("Creating student model...")
    model = StudentYOLO(
        num_classes=args.num_classes,
        teacher_feature_shapes=teacher_shapes,
        use_feature_adapters=True
    )

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    # Create loss function
    criterion = HybridDistillationLoss(
        feature_weight=args.feature_weight,
        response_weight=args.response_weight,
        temperature=args.temperature,
        feature_distance=args.feature_distance
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # Optionally resume from a previous training checkpoint
    start_epoch = 1
    best_loss = float('inf')
    history = []

    if args.resume and Path(args.resume).exists():
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
        print(f"Resumed from epoch {checkpoint['epoch']}, best loss so far: {best_loss:.4f}\n")
    elif args.resume:
        print(f"Warning: --resume path '{args.resume}' not found, starting from scratch.\n")

    # Training loop
    print(f"\nStarting training from epoch {start_epoch} to {args.epochs}...\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        metrics = train_epoch(
            model, dataloader, criterion, optimizer, device, epoch
        )

        # Update scheduler
        scheduler.step()

        # Log metrics
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Total Loss: {metrics['total_loss']:.4f}")
        print(f"  Feature Loss: {metrics['feature_loss']:.4f}")
        print(f"  Response Loss: {metrics['response_loss']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        history.append({
            'epoch': epoch,
            **metrics,
            'lr': optimizer.param_groups[0]['lr']
        })

        # Save best model
        if metrics['total_loss'] < best_loss:
            best_loss = metrics['total_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'args': vars(args)
            }, output_dir / 'best_model.pt')
            print(f"  → Saved best model (loss: {best_loss:.4f})")

        # Save checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': metrics['total_loss'],
                'args': vars(args)
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args)
    }, output_dir / 'final_model.pt')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Models saved to: {output_dir}")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid Knowledge Distillation Training"
    )

    # Data arguments
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to hybrid teacher predictions (.pt file)"
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Root directory for images"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size"
    )

    # Model arguments
    parser.add_argument(
        "--num-classes",
        type=int,
        default=80,
        help="Number of object classes"
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )

    # Distillation arguments
    parser.add_argument(
        "--feature-weight",
        type=float,
        default=1.0,
        help="Weight for feature distillation loss"
    )
    parser.add_argument(
        "--response-weight",
        type=float,
        default=1.0,
        help="Weight for response distillation loss"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=3.0,
        help="Temperature for softening logits"
    )
    parser.add_argument(
        "--feature-distance",
        type=str,
        choices=["mse", "cosine", "at"],
        default="mse",
        help="Feature distance metric"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./trained_models",
        help="Output directory for models"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a training checkpoint (.pt) to resume from"
    )

    args = parser.parse_args()

    main(args)
