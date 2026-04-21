import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pickle

#The system consists of two main components:

#predictions.py - Captures teacher model predictions (soft labels + hard labels)




@dataclass
class TeacherPrediction:
    """Structure to store teacher model predictions for knowledge distillation."""
    image_path: str
    boxes: List[List[float]]  # [x1, y1, x2, y2] normalized coordinates
    confidences: List[float]  # confidence scores
    class_ids: List[int]  # hard labels
    class_probs: List[List[float]]  # soft predictions (probability distributions)
    image_shape: Tuple[int, int, int]  # original image shape (H, W, C)
    masks: Optional[List[List[List[int]]]] = None  # segmentation masks [N, H, W] binary masks

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class YOLOTeacherInference:
    """
    Handles YOLO teacher model inference and saves predictions for student training.
    Supports knowledge distillation by capturing both hard labels and soft predictions.
    """

    def __init__(
        self,
        model_name: str = "yolo26n-seg.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        filter_class: Optional[int] = 0  # 0 for person class only
    ):
        """
        Initialize YOLO teacher model for inference.

        Args:
            model_name: YOLO model weights file or model name
            device: Device to run inference on ('cuda', 'cpu', 'mps', or None for auto)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            filter_class: If set, only save predictions for this class (0=person in COCO)
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

        print(f"Loaded YOLO teacher model: {model_name}")
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
    ) -> List[Dict]:
        """
        Run inference and optionally save predictions.

        Args:
            image_source: Path to image file, directory, or video
            save_predictions: Whether to extract and return soft predictions

        Returns:
            List of prediction results
        """
        results = self.model.predict(
            source=image_source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            save=False,
            verbose=False,
            half=True if self.device == "cuda" else False  # FP16 for faster GPU inference
        )

        if save_predictions:
            return self._extract_predictions(results)
        return results

    def _extract_predictions(self, results) -> List[TeacherPrediction]:
        """
        Extract predictions including soft probabilities for distillation.

        Args:
            results: YOLO inference results

        Returns:
            List of TeacherPrediction objects
        """
        predictions = []

        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            # Get image info
            image_path = result.path
            orig_shape = result.orig_shape  # (H, W)

            # Extract box data
            xyxy = boxes.xyxy.cpu().numpy()  # boxes in xyxy format
            conf = boxes.conf.cpu().numpy()  # confidence scores
            cls = boxes.cls.cpu().numpy().astype(int)  # class IDs

            # Get soft predictions (class probabilities)
            # YOLO stores class probabilities in boxes.data
            if hasattr(boxes, 'data'):
                # Extract probabilities for all classes
                # For YOLOv8, boxes.data contains [x1, y1, x2, y2, conf, cls]
                # We need to get the full class probability distribution
                if hasattr(result, 'probs') and result.probs is not None:
                    # Classification model
                    class_probs = [result.probs.data.cpu().numpy().tolist()] * len(boxes)
                else:
                    # Detection model - reconstruct probabilities
                    # Confidence is max_prob * objectness, need to estimate distribution
                    # For now, create one-hot-like distribution centered on predicted class
                    num_classes = len(self.model.names)
                    class_probs = []
                    for i in range(len(boxes)):
                        prob_dist = np.zeros(num_classes)
                        prob_dist[cls[i]] = conf[i]
                        class_probs.append(prob_dist.tolist())
            else:
                # Fallback: create one-hot encoding
                num_classes = len(self.model.names)
                class_probs = []
                for i in range(len(boxes)):
                    prob_dist = np.zeros(num_classes)
                    prob_dist[cls[i]] = conf[i]
                    class_probs.append(prob_dist.tolist())

            # Normalize boxes to [0, 1]
            h, w = orig_shape
            normalized_boxes = []
            filtered_indices = []

            for idx, box in enumerate(xyxy):
                # Filter by class if specified
                if self.filter_class is not None and cls[idx] != self.filter_class:
                    continue

                x1, y1, x2, y2 = box
                normalized_box = [
                    float(x1 / w),
                    float(y1 / h),
                    float(x2 / w),
                    float(y2 / h)
                ]
                normalized_boxes.append(normalized_box)
                filtered_indices.append(idx)

            # Skip if no boxes after filtering
            if len(normalized_boxes) == 0:
                continue

            # Extract segmentation masks if available
            masks_data = None
            if hasattr(result, 'masks') and result.masks is not None:
                # Get masks for filtered indices only
                masks_xy = result.masks.xy  # List of [N, 2] polygon coordinates
                masks_data = []
                for i in filtered_indices:
                    if i < len(masks_xy):
                        # Convert polygon coordinates to list format
                        # Each mask is stored as polygon points [[x1,y1], [x2,y2], ...]
                        mask_polygon = masks_xy[i].tolist()
                        masks_data.append(mask_polygon)

            # Create prediction object
            prediction = TeacherPrediction(
                image_path=str(image_path),
                boxes=normalized_boxes,
                confidences=[float(conf[i]) for i in filtered_indices],
                class_ids=[int(cls[i]) for i in filtered_indices],
                class_probs=[class_probs[i] for i in filtered_indices],
                image_shape=(int(h), int(w), 3),
                masks=masks_data
            )
            predictions.append(prediction)

        return predictions

    def save_predictions(
        self,
        predictions: List[TeacherPrediction],
        output_path: str,
        format: str = "json"
    ):
        """
        Save predictions to disk.

        Args:
            predictions: List of TeacherPrediction objects
            output_path: Output file path
            format: Save format ('json', 'pickle', or 'npz')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(
                    [pred.to_dict() for pred in predictions],
                    f,
                    indent=2
                )
        elif format == "pickle":
            with open(output_path, 'wb') as f:
                pickle.dump(predictions, f)
        elif format == "npz":
            # Save as compressed numpy archive (much smaller for masks)
            np.savez_compressed(
                output_path,
                image_paths=[p.image_path for p in predictions],
                boxes=[p.boxes for p in predictions],
                confidences=[p.confidences for p in predictions],
                class_ids=[p.class_ids for p in predictions],
                class_probs=[p.class_probs for p in predictions],
                image_shapes=[p.image_shape for p in predictions],
                masks=[p.masks for p in predictions]
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Saved {len(predictions)} predictions to {output_path}")

    def process_dataset(
        self,
        image_dir: str,
        output_dir: str,
        batch_size: int = 16,
        save_format: str = "json",
        checkpoint_interval: int = 100,
        resume: bool = True
    ):
        """
        Process entire dataset and save predictions with incremental checkpointing.

        Args:
            image_dir: Directory containing images
            output_dir: Directory to save predictions
            batch_size: Number of images to process per batch
            save_format: Format to save predictions ('json' or 'pickle')
            checkpoint_interval: Save checkpoint every N images (default: 100)
            resume: Resume from last checkpoint if available (default: True)
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

        # Check for existing checkpoint
        checkpoint_file = output_dir / f"checkpoint.{save_format}"
        progress_file = output_dir / "progress.json"
        all_predictions = []
        start_idx = 0

        if resume and checkpoint_file.exists() and progress_file.exists():
            print(f"\nFound existing checkpoint, resuming...")
            try:
                # Load existing predictions
                all_predictions = list(load_predictions(str(checkpoint_file)))

                # Load progress info
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
                print(f"\nProcessing batch {batch_num}/{total_batches} (images {i+1}-{min(i+batch_size, len(image_files))}/{len(image_files)})")

                # Run inference on batch
                predictions = self.run_inference(batch_paths, save_predictions=True)
                all_predictions.extend(predictions)
                images_processed += len(batch_files)

                # Save checkpoint periodically
                if images_processed % checkpoint_interval < batch_size or images_processed == len(image_files):
                    print(f"  Saving checkpoint... ({len(all_predictions)} predictions so far)")
                    self.save_predictions(all_predictions, str(checkpoint_file), format=save_format)

                    # Save progress
                    progress = {
                        "processed_images": images_processed,
                        "total_images": len(image_files),
                        "num_predictions": len(all_predictions),
                        "last_updated": str(Path(checkpoint_file).stat().st_mtime)
                    }
                    with open(progress_file, 'w') as f:
                        json.dump(progress, f, indent=2)

        except KeyboardInterrupt:
            print("\n\nProcessing interrupted by user!")
            print(f"Saving checkpoint at {images_processed} images...")
            self.save_predictions(all_predictions, str(checkpoint_file), format=save_format)
            progress = {
                "processed_images": images_processed,
                "total_images": len(image_files),
                "num_predictions": len(all_predictions),
                "interrupted": True
            }
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
            print(f"Checkpoint saved. Resume by running the same command again.")
            return all_predictions

        except Exception as e:
            print(f"\n\nError during processing: {e}")
            print(f"Saving checkpoint at {images_processed} images...")
            self.save_predictions(all_predictions, str(checkpoint_file), format=save_format)
            progress = {
                "processed_images": images_processed,
                "total_images": len(image_files),
                "num_predictions": len(all_predictions),
                "error": str(e)
            }
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
            raise

        # Save final predictions with proper name
        output_file = output_dir / f"teacher_predictions.{save_format}"
        self.save_predictions(all_predictions, str(output_file), format=save_format)

        # Also save metadata
        metadata = {
            "model": str(self.model),
            "num_images": len(image_files),
            "num_predictions": len(all_predictions),
            "images_with_detections": sum(1 for p in all_predictions if len(p.boxes) > 0),
            "avg_detections_per_image": len(all_predictions) / len(image_files) if image_files else 0,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "filter_class": self.filter_class,
            "device": self.device
        }

        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Clean up checkpoint files
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        if progress_file.exists():
            progress_file.unlink()

        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"{'='*60}")
        print(f"Total images processed: {len(image_files)}")
        print(f"Total predictions saved: {len(all_predictions)}")
        print(f"Images with detections: {metadata['images_with_detections']}")
        print(f"Average detections per image: {metadata['avg_detections_per_image']:.2f}")
        print(f"\nOutput saved to: {output_file}")

        return all_predictions


def load_predictions(file_path: str) -> List[TeacherPrediction]:
    """
    Load saved predictions from disk.

    Args:
        file_path: Path to saved predictions file

    Returns:
        List of TeacherPrediction objects
    """
    file_path = Path(file_path)

    if file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        return [TeacherPrediction(**item) for item in data]
    elif file_path.suffix == '.pickle':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_path.suffix == '.npz':
        data = np.load(file_path, allow_pickle=True)
        predictions = []
        for i in range(len(data['image_paths'])):
            predictions.append(TeacherPrediction(
                image_path=str(data['image_paths'][i]),
                boxes=data['boxes'][i].tolist() if isinstance(data['boxes'][i], np.ndarray) else data['boxes'][i],
                confidences=data['confidences'][i].tolist() if isinstance(data['confidences'][i], np.ndarray) else data['confidences'][i],
                class_ids=data['class_ids'][i].tolist() if isinstance(data['class_ids'][i], np.ndarray) else data['class_ids'][i],
                class_probs=data['class_probs'][i].tolist() if isinstance(data['class_probs'][i], np.ndarray) else data['class_probs'][i],
                image_shape=tuple(data['image_shapes'][i]),
                masks=data['masks'][i] if 'masks' in data else None
            ))
        return predictions
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate teacher predictions from YOLO for knowledge distillation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model name or path to weights"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./teacher_predictions",
        help="Output directory for predictions"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--person-only",
        action="store_true",
        help="Only save person detections (class 0)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "pickle", "npz"],
        default="pickle",
        help="Output format (default: pickle for efficiency)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N images (default: 100)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint, start fresh"
    )

    args = parser.parse_args()

    # Initialize teacher model
    teacher = YOLOTeacherInference(
        model_name=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        filter_class=0 if args.person_only else None
    )

    # Process dataset
    teacher.process_dataset(
        image_dir=args.input,
        output_dir=args.output,
        batch_size=args.batch_size,
        save_format=args.format,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume
    )
