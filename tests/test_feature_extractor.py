"""Tests for the feature extraction hook mechanism.
Tests hooks, layer resolution, and feature shapes without requiring a YOLO model.
"""
import pytest
import torch
import torch.nn as nn
from teacher.feature_extractor import FeatureHook, YOLOFeatureExtractor


class DummyBackbone(nn.Module):
    """Minimal model that mimics YOLO backbone structure for testing hooks."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),    # 0
            nn.ReLU(),                      # 1
            nn.Conv2d(16, 32, 3, 2, 1),    # 2
            nn.ReLU(),                      # 3
            nn.Conv2d(32, 64, 3, 2, 1),    # 4
            nn.ReLU(),                      # 5
            nn.Conv2d(64, 128, 3, 2, 1),   # 6
            nn.ReLU(),                      # 7
            nn.Conv2d(128, 256, 3, 2, 1),  # 8
            nn.ReLU(),                      # 9
        )

    def forward(self, x):
        return self.model(x)


class TestFeatureHook:
    def test_hook_captures_output(self):
        layer = nn.Conv2d(3, 16, 3, padding=1)
        hook = FeatureHook(layer, "test_layer")

        x = torch.randn(1, 3, 32, 32)
        _ = layer(x)

        assert hook.features is not None
        assert hook.features.shape == (1, 16, 32, 32)

        hook.close()

    def test_hook_detaches(self):
        layer = nn.Conv2d(3, 16, 3, padding=1)
        hook = FeatureHook(layer, "test_layer")

        x = torch.randn(1, 3, 32, 32)
        _ = layer(x)

        assert not hook.features.requires_grad
        hook.close()

    def test_hook_close(self):
        layer = nn.Conv2d(3, 16, 3, padding=1)
        hook = FeatureHook(layer, "test_layer")
        hook.close()

        x = torch.randn(1, 3, 32, 32)
        _ = layer(x)

        # After closing, the stored features should be from before close
        # (the hook no longer updates)
        assert hook.features is None

    def test_hook_handles_tuple_output(self):
        class TupleLayer(nn.Module):
            def forward(self, x):
                return x, x * 2

        layer = TupleLayer()
        hook = FeatureHook(layer, "tuple_layer")

        x = torch.randn(1, 3, 32, 32)
        _ = layer(x)

        assert isinstance(hook.features, tuple)
        assert len(hook.features) == 2
        hook.close()


class TestYOLOFeatureExtractorWithDummy:
    """Test the extractor logic using a simple dummy model instead of real YOLO."""

    def test_get_layer_by_name(self):
        model = DummyBackbone()

        # Create a minimal extractor-like object to test the method
        extractor = YOLOFeatureExtractor.__new__(YOLOFeatureExtractor)
        extractor.pytorch_model = model

        layer = extractor._get_layer_by_name(model, "model.4")
        assert isinstance(layer, nn.Conv2d)
        assert layer.in_channels == 32
        assert layer.out_channels == 64

    def test_get_layer_returns_none_for_invalid(self):
        model = DummyBackbone()
        extractor = YOLOFeatureExtractor.__new__(YOLOFeatureExtractor)
        extractor.pytorch_model = model

        assert extractor._get_layer_by_name(model, "model.999") is None
        assert extractor._get_layer_by_name(model, "nonexistent.layer") is None

    def test_hooks_register_on_valid_layers(self):
        model = DummyBackbone()
        extractor = YOLOFeatureExtractor.__new__(YOLOFeatureExtractor)
        extractor.pytorch_model = model
        extractor.model = model
        extractor.device = "cpu"
        extractor.feature_layers = ["model.4", "model.6"]
        extractor.hooks = {}
        extractor._register_hooks()

        assert "model.4" in extractor.hooks
        assert "model.6" in extractor.hooks

        extractor.close()

    def test_extract_features_captures_intermediate(self):
        model = DummyBackbone()
        extractor = YOLOFeatureExtractor.__new__(YOLOFeatureExtractor)
        extractor.pytorch_model = model
        extractor.model = model
        extractor.device = "cpu"
        extractor.feature_layers = ["model.4", "model.6"]
        extractor.hooks = {}
        extractor._register_hooks()

        x = torch.randn(1, 3, 640, 640)
        result = extractor.extract_features(x, return_predictions=True)

        assert "features" in result
        assert "model.4" in result["features"]
        assert "model.6" in result["features"]

        # model.4 is Conv2d(32, 64, ...) with stride 2 at layer 4
        # Input goes through layers 0-4 (3 stride-2 convs before, so 640/8=80, then /2=40 at layer 4)
        feat_4 = result["features"]["model.4"]
        assert feat_4.shape[1] == 64  # out_channels of layer 4

        extractor.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFeatureExtractorCUDA:
    def test_hook_on_cuda(self):
        layer = nn.Conv2d(3, 16, 3, padding=1).cuda()
        hook = FeatureHook(layer, "test")

        x = torch.randn(1, 3, 32, 32).cuda()
        _ = layer(x)

        assert hook.features.device.type == "cuda"
        hook.close()

    def test_dummy_model_on_cuda(self):
        model = DummyBackbone().cuda()
        extractor = YOLOFeatureExtractor.__new__(YOLOFeatureExtractor)
        extractor.pytorch_model = model
        extractor.model = model
        extractor.device = "cuda"
        extractor.feature_layers = ["model.4"]
        extractor.hooks = {}
        extractor._register_hooks()

        x = torch.randn(1, 3, 128, 128).cuda()
        result = extractor.extract_features(x)
        assert result["features"]["model.4"].device.type == "cuda"

        extractor.close()
