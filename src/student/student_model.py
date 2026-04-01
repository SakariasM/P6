"""
Student model architecture for hybrid knowledge distillation.
Designed to match teacher's intermediate features and final predictions.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class FeatureMatchingLayer(nn.Module):
    """
    Adapter layer to match student features to teacher feature dimensions.
    Uses 1x1 convolutions for channel alignment.
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        use_bn: bool = True
    ):
        """
        Args:
            student_channels: Number of channels in student feature map
            teacher_channels: Number of channels in teacher feature map
            use_bn: Whether to use batch normalization
        """
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(teacher_channels) if use_bn else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


class StudentYOLO(nn.Module):
    """
    Lightweight student YOLO model for knowledge distillation.
    Includes feature matching layers at intermediate stages.
    """

    def __init__(
        self,
        num_classes: int = 1,
        teacher_feature_shapes: Optional[Dict[str, Tuple]] = None,
        use_feature_adapters: bool = True
    ):
        """
        Args:
            num_classes: Number of object classes
            teacher_feature_shapes: Dictionary mapping layer names to feature shapes
                                   from teacher model (for adapter creation)
            use_feature_adapters: Whether to use feature adapters for distillation
        """
        super().__init__()

        self.num_classes = num_classes
        self.use_feature_adapters = use_feature_adapters

        # Build lightweight backbone
        # This is a simplified version - you'd typically use MobileNet, EfficientNet, etc.
        self.backbone = self._build_backbone()

        # Feature extraction points (intermediate layers)
        self.feature_layers = ['stage1', 'stage2', 'stage3']

        # Feature adapters for matching teacher dimensions
        if use_feature_adapters and teacher_feature_shapes:
            self.feature_adapters = self._build_feature_adapters(teacher_feature_shapes)
        else:
            self.feature_adapters = None

        # Detection head (simplified)
        self.detection_head = self._build_detection_head()

    def _build_backbone(self) -> nn.ModuleDict:
        """
        Build lightweight backbone network.
        In practice, use MobileNetV3, EfficientNet, or similar.
        """
        backbone = nn.ModuleDict()

        # Stage 1: Initial convolution
        backbone['stem'] = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        # Stage 1: Early features (P2 level, stride 4)
        backbone['stage1'] = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            self._make_csp_block(64, 64, num_blocks=1),
        )

        # Stage 2: Mid features (P3 level, stride 8)
        backbone['stage2'] = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            self._make_csp_block(128, 128, num_blocks=2),
        )

        # Stage 3: Deep features (P4 level, stride 16)
        backbone['stage3'] = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            self._make_csp_block(256, 256, num_blocks=2),
        )

        # Stage 4: Deepest features (P5 level, stride 32)
        backbone['stage4'] = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
            self._make_csp_block(512, 512, num_blocks=1),
        )

        return backbone

    def _make_csp_block(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Module:
        """Create a simple CSP (Cross Stage Partial) block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def _build_feature_adapters(
        self,
        teacher_feature_shapes: Dict[str, Tuple]
    ) -> nn.ModuleDict:
        """
        Build feature adapters to match teacher feature dimensions.

        Args:
            teacher_feature_shapes: Dict of teacher feature shapes {layer_name: (B, C, H, W)}

        Returns:
            ModuleDict of adapters
        """
        adapters = nn.ModuleDict()

        # Define student feature channels at each stage (ordered small→large)
        student_stages = [
            ('stage1', 64),
            ('stage2', 128),
            ('stage3', 256),
        ]

        # Pair teacher layers with student stages by index
        # Teacher layers are sorted by name so model.4 → stage1, model.6 → stage2, etc.
        teacher_layers = [
            (name, shape) for name, shape in teacher_feature_shapes.items()
            if isinstance(shape, tuple) and len(shape) == 4
        ]

        for i, (teacher_layer, shape) in enumerate(teacher_layers):
            if i >= len(student_stages):
                break
            student_layer, s_channels = student_stages[i]
            teacher_channels = shape[1]
            adapter_name = f"{student_layer}_to_{teacher_layer.replace('.', '_')}"
            adapters[adapter_name] = FeatureMatchingLayer(
                s_channels,
                teacher_channels,
                use_bn=True
            )

        return adapters

    def _build_detection_head(self) -> nn.Module:
        """
        Build detection head for object detection.
        Simplified version - full YOLO head is more complex.
        """
        # Number of outputs per anchor: (x, y, w, h, objectness, class_probs)
        num_outputs = 5 + self.num_classes

        return nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, num_outputs, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional feature extraction.

        Args:
            x: Input tensor [B, C, H, W]
            return_features: Whether to return intermediate features

        Returns:
            Dictionary containing:
                - predictions: Detection predictions
                - features: Intermediate features (if return_features=True)
                - adapted_features: Adapted features matching teacher dims (if adapters exist)
        """
        features = {}
        adapted_features = {}

        # Forward through backbone
        x = self.backbone['stem'](x)

        for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
            x = self.backbone[stage_name](x)

            # Store intermediate features
            if return_features and stage_name in self.feature_layers:
                features[stage_name] = x

                # Apply feature adapters if available
                if self.feature_adapters:
                    for adapter_name, adapter in self.feature_adapters.items():
                        if adapter_name.startswith(stage_name):
                            adapted_features[adapter_name] = adapter(x)

        # Detection head
        predictions = self.detection_head(x)

        result = {'predictions': predictions}

        if return_features:
            result['features'] = features
            if adapted_features:
                result['adapted_features'] = adapted_features

        return result


def create_student_from_teacher(
    teacher_feature_extractor,
    num_classes: int = 1,
) -> StudentYOLO:
    """
    Create a student model matched to a teacher's feature dimensions.

    Args:
        teacher_feature_extractor: YOLOFeatureExtractor instance
        num_classes: Number of object classes

    Returns:
        Student model instance
    """
    # Get teacher feature shapes
    teacher_shapes = teacher_feature_extractor.get_feature_shapes()

    print(f"Creating student model to match teacher with {len(teacher_shapes)} feature layers")

    student = StudentYOLO(
        num_classes=num_classes,
        teacher_feature_shapes=teacher_shapes,
        use_feature_adapters=True
    )
    print("Created StudentYOLO with feature adapters")

    # Print model info
    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return student


if __name__ == "__main__":
    # Test student model creation
    print("Testing student model architectures...\n")

    # Test standard student
    print("="*60)
    print("Standard Student Model")
    print("="*60)

    # Mock teacher feature shapes (typical YOLO shapes)
    mock_teacher_shapes = {
        'model.4': (1, 128, 80, 80),
        'model.6': (1, 256, 40, 40),
        'model.9': (1, 512, 20, 20),
    }

    student = StudentYOLO(
        num_classes=1,
        teacher_feature_shapes=mock_teacher_shapes,
        use_feature_adapters=True
    )

    # Test forward pass
    dummy_input = torch.randn(2, 3, 640, 640)
    output = student(dummy_input, return_features=True)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Predictions shape: {output['predictions'].shape}")
    print(f"Number of feature maps: {len(output['features'])}")
    for name, feat in output['features'].items():
        print(f"  {name}: {feat.shape}")

    if 'adapted_features' in output:
        print(f"Number of adapted features: {len(output['adapted_features'])}")
        for name, feat in output['adapted_features'].items():
            print(f"  {name}: {feat.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in student.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print(f"\nTotal parameters: {total_params:,}")
