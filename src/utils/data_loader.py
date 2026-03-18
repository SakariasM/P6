"""
General-purpose data loader for training.
Supports both simple datasets and distillation datasets with teacher predictions.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Tuple, Dict, List
import sys
from pathlib import Path as PathLib
sys.path.insert(0, str(PathLib(__file__).parent.parent))

from teacher.predictions import load_predictions, TeacherPrediction


class DistillationDataset(Dataset):
    """
    Dataset for loading images with teacher model predictions for distillation.
    """

    def __init__(
        self,
        predictions_file: str,
        image_root: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (640, 640),
        max_detections: int = 100
    ):
        """
        Initialize distillation dataset.

        Args:
            predictions_file: Path to saved teacher predictions (JSON or pickle)
            image_root: Root directory for images (if paths in predictions are relative)
            transform: Optional transforms to apply to images
            target_size: Target size for images (width, height)
            max_detections: Maximum number of detections to keep per image
        """
        self.predictions = load_predictions(predictions_file)
        self.image_root = Path(image_root) if image_root else None
        self.target_size = target_size
        self.max_detections = max_detections

        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

        # Filter out images with no detections
        self.predictions = [p for p in self.predictions if len(p.boxes) > 0]

        print(f"Loaded {len(self.predictions)} images with teacher predictions")

    def __len__(self) -> int:
        return len(self.predictions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Returns:
            Dictionary containing:
                - image: Transformed image tensor [C, H, W]
                - boxes: Bounding boxes tensor [N, 4] (normalized)
                - confidences: Confidence scores [N]
                - class_ids: Hard labels [N]
                - class_probs: Soft predictions [N, num_classes]
                - num_boxes: Number of actual boxes (before padding)
        """
        prediction = self.predictions[idx]

        # Load image
        image_path = Path(prediction.image_path)
        if self.image_root and not image_path.is_absolute():
            image_path = self.image_root / image_path

        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Prepare boxes and labels
        num_boxes = min(len(prediction.boxes), self.max_detections)

        # Pad or truncate to max_detections
        boxes = torch.zeros((self.max_detections, 4), dtype=torch.float32)
        confidences = torch.zeros(self.max_detections, dtype=torch.float32)
        class_ids = torch.zeros(self.max_detections, dtype=torch.long)

        # Get number of classes from first prediction
        num_classes = len(prediction.class_probs[0]) if prediction.class_probs else 80
        class_probs = torch.zeros((self.max_detections, num_classes), dtype=torch.float32)

        # Fill with actual data
        for i in range(num_boxes):
            boxes[i] = torch.tensor(prediction.boxes[i], dtype=torch.float32)
            confidences[i] = prediction.confidences[i]
            class_ids[i] = prediction.class_ids[i]
            class_probs[i] = torch.tensor(prediction.class_probs[i], dtype=torch.float32)

        return {
            'image': image,
            'boxes': boxes,
            'confidences': confidences,
            'class_ids': class_ids,
            'class_probs': class_probs,
            'num_boxes': torch.tensor(num_boxes, dtype=torch.long),
            'image_path': str(image_path)
        }


def create_distillation_dataloader(
    predictions_file: str,
    image_root: Optional[str] = None,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (640, 640)
) -> DataLoader:
    """
    Create a DataLoader for distillation training.

    Args:
        predictions_file: Path to saved teacher predictions
        image_root: Root directory for images
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        target_size: Target image size (width, height)

    Returns:
        DataLoader for distillation training
    """
    dataset = DistillationDataset(
        predictions_file=predictions_file,
        image_root=image_root,
        target_size=target_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


# Example usage and testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test data loader")
    parser.add_argument(
        "--predictions",
        type=str,
        help="Path to teacher predictions file"
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Root directory for images"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for testing"
    )

    args = parser.parse_args()

    if args.predictions:
        # Test distillation dataloader
        print("Creating distillation dataloader...")
        dataloader = create_distillation_dataloader(
            predictions_file=args.predictions,
            image_root=args.image_root,
            batch_size=args.batch_size,
            shuffle=True
        )

        print(f"\nDataset size: {len(dataloader.dataset)}")
        print(f"Number of batches: {len(dataloader)}")

        # Load one batch to test
        print("\nLoading first batch...")
        batch = next(iter(dataloader))

        print(f"Image batch shape: {batch['image'].shape}")
        print(f"Boxes batch shape: {batch['boxes'].shape}")
        print(f"Class probabilities shape: {batch['class_probs'].shape}")
        print(f"Number of boxes per image: {batch['num_boxes']}")

        # Show some statistics
        print("\nFirst image statistics:")
        num_boxes = batch['num_boxes'][0].item()
        print(f"  Number of detections: {num_boxes}")
        if num_boxes > 0:
            print(f"  Class IDs: {batch['class_ids'][0][:num_boxes]}")
            print(f"  Confidences: {batch['confidences'][0][:num_boxes]}")
            print(f"  First box: {batch['boxes'][0][0]}")
    else:
        print("Usage: python data_loader.py --predictions /path/to/predictions.json")
        print("Example:")
        print("  python data_loader.py --predictions ./teacher_predictions/teacher_predictions.json --image-root ./data/images")
