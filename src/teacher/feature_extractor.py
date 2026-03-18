"""
Feature extraction utilities for hybrid knowledge distillation.
Extracts intermediate feature maps and logits from teacher models.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict


class FeatureHook:
    """Hook to capture intermediate layer outputs."""

    def __init__(self, module: nn.Module, layer_name: str):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.layer_name = layer_name
        self.features = None

    def hook_fn(self, module, input, output):
        """Store the output of the layer."""
        # Detach to avoid keeping computation graph
        if isinstance(output, torch.Tensor):
            self.features = output.detach()
        elif isinstance(output, (tuple, list)):
            # Some layers return tuples
            self.features = tuple(o.detach() if isinstance(o, torch.Tensor) else o for o in output)
        else:
            self.features = output

    def close(self):
        """Remove the hook."""
        self.hook.remove()


class YOLOFeatureExtractor:
    """
    Extracts intermediate features and logits from YOLO models for distillation.
    """

    def __init__(
        self,
        model,
        feature_layers: Optional[List[str]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize feature extractor.

        Args:
            model: YOLO model (ultralytics)
            feature_layers: List of layer names to extract features from.
                          If None, will use default YOLO backbone layers.
            device: Device to run on
        """
        self.model = model
        self.device = device or self._get_best_device()
        self.model.to(self.device)
        self.model.eval()

        # Get the underlying PyTorch model
        if hasattr(model, 'model'):
            self.pytorch_model = model.model
        else:
            self.pytorch_model = model

        # Default feature layers for YOLO (backbone outputs at different scales)
        if feature_layers is None:
            self.feature_layers = self._get_default_yolo_layers()
        else:
            self.feature_layers = feature_layers

        self.hooks = {}
        self._register_hooks()

        print(f"Registered feature extraction hooks on {len(self.hooks)} layers:")
        for layer_name in self.hooks.keys():
            print(f"  - {layer_name}")

    def _get_best_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _get_default_yolo_layers(self) -> List[str]:
        """
        Get default feature extraction layers for YOLO architecture.
        These typically correspond to backbone outputs at different scales.
        """
        # For YOLOv8/v11 architecture
        # Typically we want features from the backbone at different spatial resolutions
        # Model structure: model -> model[i] where i is layer index

        default_layers = []

        # Try to find backbone output layers
        # YOLOv8 typical structure has backbone outputs at indices 4, 6, 9
        # These correspond to P3, P4, P5 feature pyramid levels
        if hasattr(self.pytorch_model, 'model'):
            model_layers = self.pytorch_model.model
            # Common feature extraction points in YOLO backbone
            # Adjust based on your specific YOLO version
            candidate_indices = [4, 6, 9, 12, 15]  # Common backbone output indices

            for idx in candidate_indices:
                if idx < len(model_layers):
                    default_layers.append(f"model.{idx}")

        # Fallback if we couldn't auto-detect
        if not default_layers:
            print("Warning: Could not auto-detect feature layers. Using generic approach.")
            default_layers = ["model.4", "model.6", "model.9"]

        return default_layers

    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        for layer_name in self.feature_layers:
            try:
                # Navigate to the layer using dot notation
                layer = self._get_layer_by_name(self.pytorch_model, layer_name)
                if layer is not None:
                    self.hooks[layer_name] = FeatureHook(layer, layer_name)
                else:
                    print(f"Warning: Could not find layer {layer_name}")
            except Exception as e:
                print(f"Warning: Could not register hook on {layer_name}: {e}")

    def _get_layer_by_name(self, model: nn.Module, layer_name: str):
        """Get a layer by its dot-notation name."""
        parts = layer_name.split('.')
        current = model

        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif part.isdigit() and isinstance(current, (nn.ModuleList, nn.Sequential)):
                idx = int(part)
                if idx >= len(current):
                    return None
                current = current[idx]
            else:
                return None
        return current

    def extract_features(
        self,
        image: torch.Tensor,
        return_predictions: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features and predictions from the model.

        Args:
            image: Input image tensor [B, C, H, W] or [C, H, W]
            return_predictions: Whether to return final predictions

        Returns:
            Dictionary containing:
                - features: Dict of intermediate features {layer_name: tensor}
                - logits: Final model output (if return_predictions)
                - predictions: Processed predictions (if return_predictions)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension

        image = image.to(self.device)

        with torch.no_grad():
            # Forward pass (this will trigger hooks)
            output = self.pytorch_model(image)

            # Collect features from hooks
            features = {}
            for layer_name, hook in self.hooks.items():
                if hook.features is not None:
                    features[layer_name] = hook.features

            result = {'features': features}

            if return_predictions:
                # Extract predictions from YOLO output
                # YOLO output is typically a tuple of (predictions, loss)
                if isinstance(output, (tuple, list)):
                    predictions = output[0]
                else:
                    predictions = output

                result['logits'] = predictions
                result['predictions'] = predictions

        return result

    def extract_features_from_path(
        self,
        image_path: str,
        img_size: int = 640
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from an image file.

        Args:
            image_path: Path to image file
            img_size: Input image size

        Returns:
            Dictionary with features and predictions
        """
        from PIL import Image
        import torchvision.transforms as transforms

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image)

        return self.extract_features(image_tensor)

    def get_feature_shapes(self, input_size: Tuple[int, int] = (640, 640)) -> Dict[str, Tuple]:
        """
        Get the shapes of extracted features for a given input size.

        Args:
            input_size: Input image size (H, W)

        Returns:
            Dictionary mapping layer names to feature shapes
        """
        dummy_input = torch.randn(1, 3, *input_size).to(self.device)
        result = self.extract_features(dummy_input, return_predictions=False)

        shapes = {}
        for layer_name, features in result['features'].items():
            if isinstance(features, torch.Tensor):
                shapes[layer_name] = features.shape
            elif isinstance(features, (tuple, list)):
                shapes[layer_name] = tuple(f.shape if isinstance(f, torch.Tensor) else None
                                          for f in features)

        return shapes

    def close(self):
        """Remove all hooks."""
        for hook in self.hooks.values():
            hook.close()
        self.hooks.clear()

    def __del__(self):
        """Cleanup hooks on deletion."""
        if hasattr(self, 'hooks'):
            self.close()


def extract_and_save_features(
    model,
    image_paths: List[str],
    output_file: str,
    batch_size: int = 8,
    img_size: int = 640,
    device: Optional[str] = None
):
    """
    Extract features from multiple images and save to file.

    Args:
        model: YOLO model
        image_paths: List of image file paths
        output_file: Output file path (.pt or .pth)
        batch_size: Batch size for processing
        img_size: Input image size
        device: Device to use
    """
    from PIL import Image
    import torchvision.transforms as transforms
    from pathlib import Path

    extractor = YOLOFeatureExtractor(model, device=device)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    all_features = []

    print(f"Extracting features from {len(image_paths)} images...")

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]

        # Load batch
        images = []
        for path in batch_paths:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)

        batch_tensor = torch.stack(images)

        # Extract features
        result = extractor.extract_features(batch_tensor)

        # Store results
        for j, path in enumerate(batch_paths):
            features_dict = {
                'image_path': str(path),
                'features': {k: v[j].cpu() for k, v in result['features'].items()},
            }
            if 'logits' in result:
                features_dict['logits'] = result['logits'][j].cpu()

            all_features.append(features_dict)

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)} images")

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(all_features, output_file)
    print(f"\nSaved features to {output_file}")

    extractor.close()

    return all_features


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract intermediate features from YOLO teacher model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model weights"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Single image to test feature extraction"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--show-shapes",
        action="store_true",
        help="Print feature shapes"
    )

    args = parser.parse_args()

    # Load model
    try:
        from ultralytics import YOLO
        model = YOLO(args.model)
        print(f"Loaded YOLO model: {args.model}")
    except ImportError:
        print("Error: ultralytics package required")
        exit(1)

    # Create feature extractor
    extractor = YOLOFeatureExtractor(model)

    # Show feature shapes
    if args.show_shapes:
        print("\nFeature shapes for input size", (args.img_size, args.img_size))
        shapes = extractor.get_feature_shapes((args.img_size, args.img_size))
        for layer_name, shape in shapes.items():
            print(f"  {layer_name}: {shape}")

    # Test on image if provided
    if args.image:
        print(f"\nExtracting features from {args.image}")
        result = extractor.extract_features_from_path(args.image, args.img_size)

        print("\nExtracted features:")
        for layer_name, features in result['features'].items():
            if isinstance(features, torch.Tensor):
                print(f"  {layer_name}: {features.shape}")

        if 'logits' in result:
            print(f"\nLogits shape: {result['logits'].shape}")

    extractor.close()
