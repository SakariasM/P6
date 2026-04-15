"""Tests for explicit teacher layer selection."""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.hybrid_distillation_train import select_teacher_layers, compute_teacher_attention
from training.distillation_loss import SegmentationDistillationLoss
from student.student_model import StudentSegmentation


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


class TestStudentLayerVariants:
    def test_single_layer_student(self):
        """Student with 1 teacher layer should work."""
        model = StudentSegmentation(
            in_channels=3, base_channels=8, depth=4,
            teacher_channels=[128],
        )
        x = torch.randn(1, 3, 64, 64)
        output, distill_info = model(x)
        assert output.shape == (1, 1, 64, 64)
        assert len(distill_info["projected"]) == 1
        assert len(distill_info["attention_maps"]) == 1

    def test_two_layer_student(self):
        """Student with 2 teacher layers should work."""
        model = StudentSegmentation(
            in_channels=3, base_channels=8, depth=4,
            teacher_channels=[128, 256],
        )
        x = torch.randn(1, 3, 64, 64)
        output, distill_info = model(x)
        assert output.shape == (1, 1, 64, 64)
        assert len(distill_info["projected"]) == 2

    def test_five_layer_student_requires_depth_5(self):
        """5 teacher layers requires depth >= 5 (assert in __init__)."""
        with pytest.raises(AssertionError):
            StudentSegmentation(
                in_channels=3, base_channels=8, depth=4,
                teacher_channels=[64, 128, 128, 256, 256],
            )

    def test_five_layer_student_with_depth_5(self):
        """5 teacher layers with depth=5 should work."""
        model = StudentSegmentation(
            in_channels=3, base_channels=8, depth=5,
            teacher_channels=[64, 128, 128, 256, 256],
        )
        x = torch.randn(1, 3, 64, 64)
        output, distill_info = model(x)
        assert output.shape == (1, 1, 64, 64)
        assert len(distill_info["projected"]) == 5


class TestAblationIntegration:
    def test_single_layer_training_step(self):
        """Full forward + backward pass with 1 teacher layer."""
        teacher_channels = [128]
        model = StudentSegmentation(
            in_channels=3, base_channels=8, depth=4,
            teacher_channels=teacher_channels,
        )
        criterion = SegmentationDistillationLoss(
            attention_weight=1.0, mimicry_weight=0.5,
            relation_weight=0.5, seg_weight=1.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        images = torch.randn(2, 3, 64, 64)
        teacher_feats = [torch.randn(2, 128, 4, 4)]
        teacher_mask = torch.randint(0, 2, (2, 1, 64, 64)).float()

        seg_output, distill_info = model(images)
        teacher_atts = [compute_teacher_attention(f) for f in teacher_feats]

        loss, loss_dict = criterion(
            student_atts=distill_info["attention_maps"],
            teacher_atts=teacher_atts,
            projected_student_feats=distill_info["projected"],
            teacher_feats=teacher_feats,
            student_mask=seg_output,
            teacher_mask=teacher_mask,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert "total" in loss_dict
        assert "segmentation" in loss_dict

    def test_five_layer_training_step(self):
        """Full forward + backward pass with 5 teacher layers (depth=5)."""
        teacher_channels = [64, 128, 128, 256, 256]
        model = StudentSegmentation(
            in_channels=3, base_channels=8, depth=5,
            teacher_channels=teacher_channels,
        )
        criterion = SegmentationDistillationLoss(
            attention_weight=1.0, mimicry_weight=0.5,
            relation_weight=0.5, seg_weight=1.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        images = torch.randn(2, 3, 64, 64)
        teacher_feats = [
            torch.randn(2, 64, 32, 32),
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 128, 8, 8),
            torch.randn(2, 256, 4, 4),
            torch.randn(2, 256, 2, 2),
        ]
        teacher_mask = torch.randint(0, 2, (2, 1, 64, 64)).float()

        seg_output, distill_info = model(images)
        teacher_atts = [compute_teacher_attention(f) for f in teacher_feats]

        loss, loss_dict = criterion(
            student_atts=distill_info["attention_maps"],
            teacher_atts=teacher_atts,
            projected_student_feats=distill_info["projected"],
            teacher_feats=teacher_feats,
            student_mask=seg_output,
            teacher_mask=teacher_mask,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
