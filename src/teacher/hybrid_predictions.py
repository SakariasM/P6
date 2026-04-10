"""
Enhanced prediction capture for hybrid distillation.
Captures both response-based (logits) and feature-based (intermediate features) knowledge.
"""
import json
import cv2
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
    # Binary person mask [H, W] float32 (merged from all instance masks)
    segmentation_mask: Optional[torch.Tensor] = None

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
        model_name: str = "yolo26n-seg.pt",
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
        from PIL import Image
        import torchvision.transforms as transforms

        predictions = []

        # If single source, make it a list
        if isinstance(image_sources, str):
            if Path(image_sources).is_file():
                image_sources = [image_sources]

        # --- First pass: collect per-image detection data and identify which
        #     images have valid (filtered) detections ---
        per_image_data = []  # list of dicts, one per result with detections
        for idx, result in enumerate(results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            if isinstance(image_sources, list) and idx < len(image_sources):
                image_path = image_sources[idx]
            else:
                image_path = result.path
            orig_shape = result.orig_shape  # (H, W)

            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            num_classes = len(self.model.names)
            class_probs = []
            for i in range(len(boxes)):
                prob_dist = np.zeros(num_classes)
                prob_dist[cls[i]] = conf[i]
                class_probs.append(prob_dist.tolist())

            h, w = orig_shape
            normalized_boxes = []
            filtered_indices = []
            for i, box in enumerate(xyxy):
                if self.filter_class is not None and cls[i] != self.filter_class:
                    continue
                x1, y1, x2, y2 = box
                normalized_boxes.append([
                    float(x1 / w), float(y1 / h),
                    float(x2 / w), float(y2 / h)
                ])
                filtered_indices.append(i)

            if len(normalized_boxes) == 0:
                continue

            per_image_data.append({
                'image_path': image_path,
                'orig_shape': orig_shape,
                'xyxy': xyxy,
                'conf': conf,
                'cls': cls,
                'class_probs': class_probs,
                'normalized_boxes': normalized_boxes,
                'filtered_indices': filtered_indices,
                'result': result,
            })

        # --- Batch feature extraction: one forward pass for all images with detections ---
        batch_features = {}   # idx -> {layer: tensor}
        batch_logits = {}     # idx -> tensor

        if self.extract_features and self.feature_extractor is not None and per_image_data:
            transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ])
            imgs = []
            for entry in per_image_data:
                img = Image.open(entry['image_path']).convert('RGB')
                imgs.append(transform(img))

            img_batch = torch.stack(imgs)  # (N, C, H, W)
            feature_result = self.feature_extractor.extract_features(
                img_batch,
                return_predictions=True
            )

            # Split batch features back per image
            for batch_idx in range(len(per_image_data)):
                batch_features[batch_idx] = {
                    k: v[batch_idx].cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in feature_result['features'].items()
                }
                if 'logits' in feature_result:
                    logits = feature_result['logits']
                    if isinstance(logits, torch.Tensor):
                        batch_logits[batch_idx] = logits[batch_idx].cpu()
                    elif isinstance(logits, (tuple, list)) and isinstance(logits[0], torch.Tensor):
                        batch_logits[batch_idx] = logits[0][batch_idx].cpu()

        # --- Second pass: build prediction objects ---
        for batch_idx, entry in enumerate(per_image_data):
            image_path = entry['image_path']
            orig_shape = entry['orig_shape']
            conf = entry['conf']
            cls = entry['cls']
            class_probs = entry['class_probs']
            normalized_boxes = entry['normalized_boxes']
            filtered_indices = entry['filtered_indices']
            result = entry['result']
            h, w = orig_shape

            features = batch_features.get(batch_idx, {})
            raw_logits = batch_logits.get(batch_idx, None)

            masks_data = None
            seg_mask = None
            if hasattr(result, 'masks') and result.masks is not None:
                masks_xy = result.masks.xy
                masks_data = []
                for i in filtered_indices:
                    if i < len(masks_xy):
                        masks_data.append(masks_xy[i].tolist())

                # Build merged binary person mask from instance polygons
                binary_mask = np.zeros((h, w), dtype=np.uint8)
                for i in filtered_indices:
                    if i < len(masks_xy):
                        pts = masks_xy[i].astype(np.int32)
                        if len(pts) >= 3:
                            cv2.fillPoly(binary_mask, [pts], 1)
                seg_mask = torch.from_numpy(binary_mask).to(torch.uint8)

            prediction = HybridTeacherPrediction(
                image_path=str(image_path),
                boxes=normalized_boxes,
                confidences=[float(conf[i]) for i in filtered_indices],
                class_ids=[int(cls[i]) for i in filtered_indices],
                class_probs=[class_probs[i] for i in filtered_indices],
                image_shape=(int(h), int(w), 3),
                features=features,
                raw_logits=raw_logits,
                masks=masks_data,
                segmentation_mask=seg_mask,
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
        checkpoint_interval: int = 50,
        worker_id: int = 0,
        num_workers: int = 1,
    ):
        """
        Process entire dataset and save hybrid predictions with checkpointing.

        Args:
            image_dir: Directory containing images
            output_dir: Directory to save predictions
            batch_size: Batch size (smaller recommended when extracting features)
            save_format: Format to save ('torch' recommended)
            checkpoint_interval: Save checkpoint every N images
            worker_id: Index of this worker (0-based)
            num_workers: Total number of parallel workers
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

        # Shard across workers — each worker takes every num_workers-th image
        if num_workers > 1:
            image_files = image_files[worker_id::num_workers]
            print(f"Worker {worker_id}/{num_workers}: processing {len(image_files)} images")

        print(f"Found {len(image_files)} images in {image_dir}")
        if self.extract_features:
            print(f"Note: Feature extraction enabled - using smaller batches")

        suffix = f"_worker{worker_id}" if num_workers > 1 else ""
        progress_file = output_dir / f"progress{suffix}.json"
        final_output_file = output_dir / f"hybrid_teacher_predictions{suffix}.{save_format}"

        # Skip if this worker already completed (old single-file format)
        if final_output_file.exists():
            print(f"Worker {worker_id} already completed: {final_output_file} exists. Skipping.")
            return

        # Resume: count completed chunk files to find start position
        existing_chunks = sorted(output_dir.glob(f"chunk_*{suffix}.{save_format}"))
        num_completed_chunks = len(existing_chunks)
        start_idx = num_completed_chunks * checkpoint_interval

        if num_completed_chunks > 0:
            print(f"\nFound {num_completed_chunks} existing chunks, resuming from image {start_idx}/{len(image_files)}")

        # Process in batches, saving chunks and clearing RAM
        chunk_predictions = []
        images_processed = start_idx
        chunk_idx = num_completed_chunks
        total_predictions_saved = 0

        def save_chunk():
            nonlocal chunk_idx, total_predictions_saved, chunk_predictions
            chunk_file = output_dir / f"chunk_{chunk_idx:04d}{suffix}.{save_format}"
            print(f"  Saving chunk {chunk_idx} ({len(chunk_predictions)} predictions) → {chunk_file.name}")
            self.save_predictions(chunk_predictions, str(chunk_file), format=save_format)
            total_predictions_saved += len(chunk_predictions)
            chunk_predictions = []  # free RAM
            chunk_idx += 1

            progress = {
                "processed_images": images_processed,
                "total_images": len(image_files),
                "num_chunks": chunk_idx,
                "num_predictions_saved": total_predictions_saved,
            }
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)

        try:
            for i in range(start_idx, len(image_files), batch_size):
                batch_files = image_files[i:i + batch_size]
                batch_paths = [str(f) for f in batch_files]

                batch_num = i // batch_size + 1
                total_batches = (len(image_files) - 1) // batch_size + 1
                print(f"\nBatch {batch_num}/{total_batches} (images {i+1}-{min(i+batch_size, len(image_files))}/{len(image_files)})")

                predictions = self.run_inference(batch_paths, save_predictions=True)
                chunk_predictions.extend(predictions)
                images_processed += len(batch_files)

                # Save chunk and free RAM every checkpoint_interval images
                images_in_chunk = images_processed - (chunk_idx * checkpoint_interval)
                if images_in_chunk >= checkpoint_interval:
                    save_chunk()

        except KeyboardInterrupt:
            print("\n\nInterrupted! Saving partial chunk...")
            if chunk_predictions:
                save_chunk()
            return

        except Exception as e:
            print(f"\n\nError: {e}")
            print("Saving partial chunk...")
            if chunk_predictions:
                save_chunk()
            raise

        # Save any remaining predictions as the final chunk
        if chunk_predictions:
            save_chunk()

        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"{'='*60}")
        print(f"Total images processed: {images_processed}")
        print(f"Total chunks saved: {chunk_idx}")
        print(f"Total predictions saved: {total_predictions_saved}")
        print(f"Run merge_predictions.py to combine all chunks into one file.")

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
    parser.add_argument("--worker-id", type=int, default=0,
                       help="Index of this worker for parallel extraction (0-based)")
    parser.add_argument("--num-workers", type=int, default=1,
                       help="Total number of parallel workers")

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
            checkpoint_interval=args.checkpoint_interval,
            worker_id=args.worker_id,
            num_workers=args.num_workers,
        )
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        sys.exit(1)
