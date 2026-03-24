"""
Enhanced prediction capture for hybrid distillation.
Captures both response-based (logits) and feature-based (intermediate features) knowledge.
"""
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pickle

from .predictions import TeacherPrediction, YOLOTeacherInference
from .feature_extractor import YOLOFeatureExtractor


@dataclass
class HybridTeacherPrediction:
    """
    Enhanced structure storing both predictions and intermediate features.
    Used for hybrid knowledge distillation (response + feature based).
    """
    # Original prediction data (response-based distillation)
    image_path: str
    boxes: List[List[float]]
    confidences: List[float]
    class_ids: List[int]
    class_probs: List[List[float]]
    image_shape: Tuple[int, int, int]

    # Feature-based distillation data
    features: Dict[str, torch.Tensor] = field(default_factory=dict)
    # Raw logits before NMS (for detection head distillation)
    raw_logits: Optional[torch.Tensor] = None

    # Optional segmentation data
    masks: Optional[List[List[List[int]]]] = None

    def to_dict_without_features(self) -> Dict:
        """Convert to dictionary without heavy feature tensors (for JSON)."""
        return {
            'image_path': self.image_path,
            'boxes': self.boxes,
            'confidences': self.confidences,
            'class_ids': self.class_ids,
            'class_probs': self.class_probs,
            'image_shape': self.image_shape,
            'masks': self.masks,
            'has_features': len(self.features) > 0,
            'feature_layers': list(self.features.keys()) if self.features else [],
            'has_raw_logits': self.raw_logits is not None
        }


class HybridYOLOInference:
    """
    Captures both predictions and intermediate features from YOLO teacher model.
    Supports hybrid knowledge distillation combining response-based and feature-based methods.
    """

    def __init__(
        self,
        model_name: str = "yolo26n.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        filter_class: Optional[int] = 0,
        feature_layers: Optional[List[str]] = None,
        extract_features: bool = True
    ):
        """
        Initialize hybrid teacher inference.

        Args:
            model_name: YOLO model weights
            device: Device to run on
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            filter_class: Filter for specific class (0=person)
            feature_layers: Specific layers to extract features from
            extract_features: Whether to extract intermediate features
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package required. Install with: pip install ultralytics"
            )

        self.model = YOLO(model_name)
        self.device = device or self._get_best_device()
        self.model.to(self.device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.filter_class = filter_class
        self.extract_features = extract_features

        # Initialize feature extractor
        if extract_features:
            self.feature_extractor = YOLOFeatureExtractor(
                self.model,
                feature_layers=feature_layers,
                device=self.device
            )
            print(f"\nFeature extraction enabled")
            # Show feature shapes
            shapes = self.feature_extractor.get_feature_shapes()
            print("Feature shapes:")
            for layer, shape in shapes.items():
                print(f"  {layer}: {shape}")
        else:
            self.feature_extractor = None

        print(f"\nLoaded YOLO hybrid teacher model: {model_name}")
        print(f"Using device: {self.device}")
        print(f"Confidence threshold: {conf_threshold}")
        if filter_class is not None:
            print(f"Filtering for class ID: {filter_class}")

    def _get_best_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def run_inference(
        self,
        image_source: str,
        save_predictions: bool = True
    ) -> List[HybridTeacherPrediction]:
        """
        Run inference and extract both predictions and features.

        Args:
            image_source: Path to image file, directory, or list of paths
            save_predictions: Whether to extract structured predictions

        Returns:
            List of HybridTeacherPrediction objects
        """
        # Run standard YOLO inference
        results = self.model.predict(
            source=image_source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            save=False,
            verbose=False,
            half=True if self.device == "cuda" else False
        )

        if save_predictions:
            return self._extract_hybrid_predictions(results, image_source)
        return results

    def _extract_hybrid_predictions(
        self,
        results,
        image_sources
    ) -> List[HybridTeacherPrediction]:
        """
        Extract both standard predictions and intermediate features.

        Args:
            results: YOLO inference results
            image_sources: Original image source(s)

        Returns:
            List of HybridTeacherPrediction objects
        """
        predictions = []

        # If single source, make it a list
        if isinstance(image_sources, str):
            if Path(image_sources).is_file():
                image_sources = [image_sources]

        for idx, result in enumerate(results):
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            # Use the original source path if available, fall back to result.path
            # (Ultralytics may return generic names like 'image0.jpg' for list inputs)
            if isinstance(image_sources, list) and idx < len(image_sources):
                image_path = image_sources[idx]
            else:
                image_path = result.path
            orig_shape = result.orig_shape  # (H, W)

            # Extract box data
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            # Get soft predictions
            num_classes = len(self.model.names)
            class_probs = []
            for i in range(len(boxes)):
                prob_dist = np.zeros(num_classes)
                prob_dist[cls[i]] = conf[i]
                class_probs.append(prob_dist.tolist())

            # Normalize boxes
            h, w = orig_shape
            normalized_boxes = []
            filtered_indices = []

            for i, box in enumerate(xyxy):
                if self.filter_class is not None and cls[i] != self.filter_class:
                    continue

                x1, y1, x2, y2 = box
                normalized_box = [
                    float(x1 / w),
                    float(y1 / h),
                    float(x2 / w),
                    float(y2 / h)
                ]
                normalized_boxes.append(normalized_box)
                filtered_indices.append(i)

            if len(normalized_boxes) == 0:
                continue

            # Extract features if enabled
            features = {}
            raw_logits = None

            if self.extract_features and self.feature_extractor is not None:
                # Load and preprocess image for feature extraction
                from PIL import Image
                import torchvision.transforms as transforms

                img = Image.open(image_path).convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize((640, 640)),
                    transforms.ToTensor(),
                ])
                img_tensor = transform(img)

                # Extract features
                feature_result = self.feature_extractor.extract_features(
                    img_tensor,
                    return_predictions=True
                )

                # Store features (move to CPU to save memory)
                features = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in feature_result['features'].items()
                }

                # Store raw logits if available
                if 'logits' in feature_result:
                    logits = feature_result['logits']
                    if isinstance(logits, torch.Tensor):
                        raw_logits = logits.cpu()
                    elif isinstance(logits, (tuple, list)):
                        # YOLO may return multiple outputs, take the first
                        raw_logits = logits[0].cpu() if isinstance(logits[0], torch.Tensor) else None

            # Extract segmentation masks if available
            masks_data = None
            if hasattr(result, 'masks') and result.masks is not None:
                masks_xy = result.masks.xy
                masks_data = []
                for i in filtered_indices:
                    if i < len(masks_xy):
                        mask_polygon = masks_xy[i].tolist()
                        masks_data.append(mask_polygon)

            # Create hybrid prediction object
            prediction = HybridTeacherPrediction(
                image_path=str(image_path),
                boxes=normalized_boxes,
                confidences=[float(conf[i]) for i in filtered_indices],
                class_ids=[int(cls[i]) for i in filtered_indices],
                class_probs=[class_probs[i] for i in filtered_indices],
                image_shape=(int(h), int(w), 3),
                features=features,
                raw_logits=raw_logits,
                masks=masks_data
            )
            predictions.append(prediction)

        return predictions

    def save_predictions(
        self,
        predictions: List[HybridTeacherPrediction],
        output_path: str,
        format: str = "torch"
    ):
        """
        Save hybrid predictions to disk.

        Args:
            predictions: List of HybridTeacherPrediction objects
            output_path: Output file path
            format: Save format ('torch' recommended, 'pickle', or 'json' for metadata only)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "torch":
            # Recommended format - efficient and preserves tensors
            torch.save(predictions, output_path)

        elif format == "pickle":
            with open(output_path, 'wb') as f:
                pickle.dump(predictions, f, protocol=pickle.HIGHEST_PROTOCOL)

        elif format == "json":
            # JSON only saves metadata, not actual features/logits
            print("Warning: JSON format only saves metadata, not feature tensors")
            with open(output_path, 'w') as f:
                json.dump(
                    [pred.to_dict_without_features() for pred in predictions],
                    f,
                    indent=2
                )
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Calculate total size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Saved {len(predictions)} hybrid predictions to {output_path}")
        print(f"File size: {file_size_mb:.2f} MB")

    def process_dataset(
        self,
        image_dir: str,
        output_dir: str,
        batch_size: int = 8,  # Smaller batch for features
        save_format: str = "torch",
        checkpoint_interval: int = 50
    ):
        """
        Process entire dataset and save hybrid predictions with checkpointing.

        Args:
            image_dir: Directory containing images
            output_dir: Directory to save predictions
            batch_size: Batch size (smaller recommended when extracting features)
            save_format: Format to save ('torch' recommended)
            checkpoint_interval: Save checkpoint every N images
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = sorted([
            f for f in image_dir.rglob('*')
            if f.suffix.lower() in image_extensions
        ])

        print(f"Found {len(image_files)} images in {image_dir}")
        if self.extract_features:
            print(f"Note: Feature extraction enabled - using smaller batches")

        checkpoint_file = output_dir / f"checkpoint.{save_format}"
        progress_file = output_dir / "progress.json"
        all_predictions = []
        start_idx = 0

        # Try to resume from checkpoint
        if checkpoint_file.exists() and progress_file.exists():
            print(f"\nFound existing checkpoint, attempting to resume...")
            try:
                all_predictions = torch.load(checkpoint_file, weights_only=False)
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    start_idx = progress.get('processed_images', 0)

                print(f"Loaded {len(all_predictions)} existing predictions")
                print(f"Resuming from image {start_idx}/{len(image_files)}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting from scratch...")
                all_predictions = []
                start_idx = 0

        # Process in batches
        images_processed = start_idx
        try:
            for i in range(start_idx, len(image_files), batch_size):
                batch_files = image_files[i:i + batch_size]
                batch_paths = [str(f) for f in batch_files]

                batch_num = i // batch_size + 1
                total_batches = (len(image_files) - 1) // batch_size + 1
                print(f"\nBatch {batch_num}/{total_batches} (images {i+1}-{min(i+batch_size, len(image_files))}/{len(image_files)})")

                # Run inference
                predictions = self.run_inference(batch_paths, save_predictions=True)
                all_predictions.extend(predictions)
                images_processed += len(batch_files)

                # Checkpoint periodically
                if images_processed % checkpoint_interval < batch_size or images_processed == len(image_files):
                    print(f"  Saving checkpoint... ({len(all_predictions)} predictions)")
                    self.save_predictions(all_predictions, str(checkpoint_file), format=save_format)

                    progress = {
                        "processed_images": images_processed,
                        "total_images": len(image_files),
                        "num_predictions": len(all_predictions)
                    }
                    with open(progress_file, 'w') as f:
                        json.dump(progress, f, indent=2)

        except KeyboardInterrupt:
            print("\n\nInterrupted! Saving checkpoint...")
            self.save_predictions(all_predictions, str(checkpoint_file), format=save_format)
            return all_predictions

        except Exception as e:
            print(f"\n\nError: {e}")
            print("Saving checkpoint...")
            self.save_predictions(all_predictions, str(checkpoint_file), format=save_format)
            raise

        # Save final output
        output_file = output_dir / f"hybrid_teacher_predictions.{save_format}"
        self.save_predictions(all_predictions, str(output_file), format=save_format)

        # Save metadata
        metadata = {
            "model": str(self.model),
            "num_images": len(image_files),
            "num_predictions": len(all_predictions),
            "feature_extraction_enabled": self.extract_features,
            "feature_layers": list(all_predictions[0].features.keys()) if all_predictions and all_predictions[0].features else [],
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "filter_class": self.filter_class,
            "device": self.device
        }

        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Cleanup checkpoint
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        if progress_file.exists():
            progress_file.unlink()

        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"{'='*60}")
        print(f"Total images: {len(image_files)}")
        print(f"Total predictions: {len(all_predictions)}")
        print(f"Output: {output_file}")

        return all_predictions

    def __del__(self):
        """Cleanup feature extractor."""
        if hasattr(self, 'feature_extractor') and self.feature_extractor:
            self.feature_extractor.close()


def load_hybrid_predictions(file_path: str) -> List[HybridTeacherPrediction]:
    """
    Load hybrid predictions from disk.

    Args:
        file_path: Path to saved predictions

    Returns:
        List of HybridTeacherPrediction objects
    """
    file_path = Path(file_path)

    if file_path.suffix == '.pt' or file_path.suffix == '.pth':
        return torch.load(file_path, weights_only=False)
    elif file_path.suffix == '.pickle':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {file_path.suffix}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate hybrid teacher predictions (predictions + features)"
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="./hybrid_predictions")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--person-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--no-features", action="store_true",
                       help="Disable feature extraction")

    args = parser.parse_args()

    import traceback
    import sys

    try:
        # Initialize
        teacher = HybridYOLOInference(
            model_name=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            filter_class=0 if args.person_only else None,
            extract_features=not args.no_features
        )

        # Process dataset
        teacher.process_dataset(
            image_dir=args.input,
            output_dir=args.output,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval
        )
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        sys.exit(1)
