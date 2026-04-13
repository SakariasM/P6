"""Tests for explicit teacher layer selection."""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.hybrid_distillation_train import select_teacher_layers


FEATURES = {
    "model.4": torch.randn(128, 80, 80),
    "model.6": torch.randn(128, 40, 40),
    "model.9": torch.randn(256, 20, 20),
    "model.12": torch.randn(256, 40, 40),
    "model.15": torch.randn(256, 80, 80),
}


class TestSelectTeacherLayers:
    def test_default_picks_last_n(self):
        names, channels = select_teacher_layers(FEATURES, num_scales=3)
        assert names == ["model.9", "model.12", "model.15"]
        assert channels == [256, 256, 256]

    def test_explicit_single_layer(self):
        names, channels = select_teacher_layers(
            FEATURES, num_scales=1, explicit_layers=["model.4"]
        )
        assert names == ["model.4"]
        assert channels == [128]

    def test_explicit_two_layers(self):
        names, channels = select_teacher_layers(
            FEATURES, num_scales=2, explicit_layers=["model.4", "model.9"]
        )
        assert names == ["model.4", "model.9"]
        assert channels == [128, 256]

    def test_explicit_all_five_layers(self):
        all_layers = ["model.4", "model.6", "model.9", "model.12", "model.15"]
        names, channels = select_teacher_layers(
            FEATURES, num_scales=5, explicit_layers=all_layers
        )
        assert names == all_layers
        assert channels == [128, 128, 256, 256, 256]

    def test_explicit_layer_not_in_features_raises(self):
        with pytest.raises(KeyError):
            select_teacher_layers(
                FEATURES, num_scales=1, explicit_layers=["model.99"]
            )

    def test_explicit_overrides_num_scales(self):
        names, channels = select_teacher_layers(
            FEATURES, num_scales=3, explicit_layers=["model.6"]
        )
        assert len(names) == 1
        assert names == ["model.6"]
