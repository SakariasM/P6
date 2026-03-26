"""
YOLO Teacher Model

Wraps a pretrained YOLO backbone and uses forward hooks to extract
multi-scale feature maps and spatial attention maps for distillation.

The teacher is always frozen — it is never trained.

Feature extraction strategy:
  YOLOv8 backbone layers (approximate indices in model.model):
    idx 4  → C2f after stride-8 downsampling  (fine-grained, H/8)
    idx 6  → C2f after stride-16 downsampling (mid-level,   H/16)
    idx 8  → C2f after stride-32 downsampling (semantic,    H/32)

  YOLO26 backbone layers (uses C3k2 blocks + C2PSA attention block):
    idx 4  → C3k2 after stride-8 downsampling  (fine-grained, H/8)
    idx 6  → C3k2 after stride-16 downsampling (mid-level,   H/16)
    idx 10 → C2PSA after stride-32 downsampling (semantic,   H/32)

  Attention maps are computed as the channel-mean of absolute activations,
  then min-max normalised per sample to [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Channel widths produced by each model variant at the three hook layers.
YOLO_FEATURE_CHANNELS = {
    # YOLOv8 — C2f blocks at layers 4, 6, 8
    "yolov8n": [128,  256,  512],
    "yolov8s": [256,  512,  1024],
    "yolov8m": [384,  768,  1536],
    "yolov8l": [512,  1024, 2048],
    "yolov8x": [640,  1280, 2560],
    # YOLO26 — C3k2 blocks at layers 4, 6; C2PSA block at layer 10
    "yolo26n": [128,  128,  256],
    "yolo26s": [256,  256,  512],
    "yolo26m": [384,  384,  768],
    "yolo26l": [512,  512,  1024],
    "yolo26x": [640,  640,  1280],
}

# Default layer indices inside model.model to hook.
YOLO_HOOK_LAYERS = {
    # YOLOv8: C2f blocks
    "yolov8n": [4, 6, 8],
    "yolov8s": [4, 6, 8],
    "yolov8m": [4, 6, 8],
    "yolov8l": [4, 6, 8],
    "yolov8x": [4, 6, 8],
    # YOLO26: C3k2 (4, 6) + C2PSA (10)
    "yolo26n": [4, 6, 10],
    "yolo26s": [4, 6, 10],
    "yolo26m": [4, 6, 10],
    "yolo26l": [4, 6, 10],
    "yolo26x": [4, 6, 10],
}


class YOLOTeacher(nn.Module):
    """Frozen YOLO teacher that exposes multi-scale attention maps.

    Args:
        model_name (str): One of the yolov8 variants (e.g. "yolov8s").
        pretrained (bool): Load pretrained weights from ultralytics hub.
        hook_layers (list[int] | None): Override which layer indices to hook.
            If None, uses the defaults from YOLO_HOOK_LAYERS.

    Usage::

        teacher = YOLOTeacher("yolov8s", pretrained=True)
        # image: [B, 3, H, W] normalised to [0, 1]
        out = teacher(image)
        # out["features"]      → list of [B, C, H', W'] tensors
        # out["attention_maps"] → list of [B, 1, H', W'] tensors in [0,1]
    """

    def __init__(self, model_name: str = "yolov8s", pretrained: bool = True,
                 hook_layers: list = None):
        super().__init__()

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for YOLOTeacher. "
                "Install it with: pip install ultralytics"
            ) from e

        # ---- Load backbone ------------------------------------------------
        weight_arg = f"{model_name}.pt" if pretrained else model_name
        yolo = YOLO(weight_arg)
        # Keep the full DetectionModel so its routing logic (Concat, etc.) runs
        # correctly. yolo.model.model is the raw nn.Sequential but it does not
        # handle multi-input layers; yolo.model does.
        self.yolo_model = yolo.model
        self.model_name = model_name

        # ---- Hook registration -------------------------------------------
        indices = hook_layers or YOLO_HOOK_LAYERS.get(model_name, [4, 6, 8])
        self._hook_indices = indices
        self._features: dict[str, torch.Tensor] = {}
        self._hooks = []

        for idx in indices:
            h = self.yolo_model.model[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(h)

        # ---- Freeze teacher completely ------------------------------------
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _make_hook(self, idx: int):
        def _hook(module, input, output):
            # output may be a tuple (e.g. from certain C2f variants)
            feat = output[0] if isinstance(output, (tuple, list)) else output
            self._features[idx] = feat
        return _hook

    def remove_hooks(self):
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict:
        """Extract features and attention maps.

        Args:
            x: [B, 3, H, W] image normalised to [0, 1].

        Returns:
            dict with keys:
                "features"      : list of tensors, coarse→fine (large stride first)
                "attention_maps": list of [B, 1, H', W'] tensors in [0, 1]
        """
        self._features = {}
        # Run through the full DetectionModel so that multi-input layers
        # (e.g. Concat) receive the correct layer routing.
        self.yolo_model(x)

        features = [self._features[idx] for idx in self._hook_indices]
        attention_maps = [self._spatial_attention(f) for f in features]

        return {"features": features, "attention_maps": attention_maps}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spatial_attention(feat: torch.Tensor) -> torch.Tensor:
        """Compute a normalised spatial attention map from a feature tensor.

        Strategy: channel-wise mean of absolute activations, then per-sample
        min-max normalisation → [0, 1].

        Args:
            feat: [B, C, H, W]

        Returns:
            [B, 1, H, W] in [0, 1]
        """
        att = feat.abs().mean(dim=1, keepdim=True)          # [B, 1, H, W]
        b = att.shape[0]
        flat = att.view(b, -1)
        mn = flat.min(1)[0].view(b, 1, 1, 1)
        mx = flat.max(1)[0].view(b, 1, 1, 1)
        return (att - mn) / (mx - mn + 1e-8)

    def feature_channels(self) -> list:
        """Return channel counts for each hooked scale."""
        return YOLO_FEATURE_CHANNELS.get(self.model_name, [256, 512, 1024])

    def num_scales(self) -> int:
        return len(self._hook_indices)
