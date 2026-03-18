"""Tests for the hybrid distillation loss function.
Validates feature loss, response loss, and combined loss computation.
"""
import pytest
import torch
import torch.nn.functional as F
from training.hybrid_distillation_train import HybridDistillationLoss


@pytest.fixture
def criterion():
    return HybridDistillationLoss(
        feature_weight=1.0,
        response_weight=1.0,
        temperature=3.0,
        feature_distance="mse",
    )


@pytest.fixture
def student_output():
    """Fake student output with features and predictions."""
    return {
        "predictions": torch.randn(2, 85, 20, 20),
        "features": {
            "stage1": torch.randn(2, 64, 80, 80),
            "stage2": torch.randn(2, 128, 40, 40),
        },
        "adapted_features": {
            "stage1_to_model_4": torch.randn(2, 128, 80, 80),
            "stage2_to_model_6": torch.randn(2, 256, 40, 40),
        },
    }


@pytest.fixture
def teacher_features():
    return {
        "model.4": torch.randn(2, 128, 80, 80),
        "model.6": torch.randn(2, 256, 40, 40),
    }


@pytest.fixture
def teacher_logits():
    return torch.randn(2, 85, 20, 20)


class TestHybridDistillationLoss:
    def test_combined_loss_runs(self, criterion, student_output, teacher_features, teacher_logits):
        loss, loss_dict = criterion(student_output, teacher_features, teacher_logits)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar
        assert "total_loss" in loss_dict
        assert "feature_loss" in loss_dict
        assert "response_loss" in loss_dict
        assert loss.item() > 0

    def test_feature_only_loss(self, student_output, teacher_features):
        criterion = HybridDistillationLoss(feature_weight=1.0, response_weight=0.0)
        loss, loss_dict = criterion(student_output, teacher_features, teacher_logits=None)

        assert loss.item() > 0
        assert loss_dict.get("response_loss", 0.0) == 0.0
        assert loss_dict["feature_loss"] > 0

    def test_response_only_loss(self, student_output, teacher_features, teacher_logits):
        # No adapted features -> no feature loss
        student_no_feat = {
            "predictions": student_output["predictions"],
        }
        criterion = HybridDistillationLoss(feature_weight=0.0, response_weight=1.0)
        loss, loss_dict = criterion(student_no_feat, {}, teacher_logits)

        assert loss.item() > 0
        assert loss_dict.get("feature_loss", 0.0) == 0.0

    def test_zero_loss_with_identical_features(self):
        criterion = HybridDistillationLoss(feature_weight=1.0, response_weight=0.0)
        shared = torch.randn(2, 128, 80, 80)
        student_output = {"adapted_features": {"stage1_to_model_4": shared}}
        teacher_features = {"model.4": shared.clone()}

        loss, loss_dict = criterion(student_output, teacher_features, teacher_logits=None)
        assert loss.item() < 1e-6

    def test_loss_is_differentiable(self, criterion, teacher_features, teacher_logits):
        # Build student output that requires grad
        pred = torch.randn(2, 85, 20, 20, requires_grad=True)
        adapted = {
            "stage1_to_model_4": torch.randn(2, 128, 80, 80, requires_grad=True),
            "stage2_to_model_6": torch.randn(2, 256, 40, 40, requires_grad=True),
        }
        student_output = {"predictions": pred, "adapted_features": adapted}

        loss, _ = criterion(student_output, teacher_features, teacher_logits)
        loss.backward()
        assert pred.grad is not None
        for v in adapted.values():
            assert v.grad is not None

    def test_feature_spatial_mismatch_handled(self):
        """Student and teacher features with different spatial dims should still work."""
        criterion = HybridDistillationLoss(feature_weight=1.0, response_weight=0.0)
        student_output = {
            "adapted_features": {"stage1_to_model_4": torch.randn(2, 128, 40, 40)},
        }
        teacher_features = {"model.4": torch.randn(2, 128, 80, 80)}

        loss, loss_dict = criterion(student_output, teacher_features, teacher_logits=None)
        assert loss.item() > 0

    def test_cosine_distance(self, student_output, teacher_features):
        criterion = HybridDistillationLoss(
            feature_weight=1.0, response_weight=0.0, feature_distance="cosine"
        )
        loss, _ = criterion(student_output, teacher_features, teacher_logits=None)
        assert loss.item() >= 0

    def test_attention_transfer_distance(self, student_output, teacher_features):
        criterion = HybridDistillationLoss(
            feature_weight=1.0, response_weight=0.0, feature_distance="at"
        )
        loss, _ = criterion(student_output, teacher_features, teacher_logits=None)
        assert loss.item() >= 0

    def test_temperature_scaling(self, student_output, teacher_features, teacher_logits):
        losses_by_temp = {}
        for temp in [1.0, 3.0, 10.0]:
            criterion = HybridDistillationLoss(
                feature_weight=0.0, response_weight=1.0, temperature=temp
            )
            loss, _ = criterion(student_output, {}, teacher_logits)
            losses_by_temp[temp] = loss.item()

        # Different temperatures should produce different losses
        assert losses_by_temp[1.0] != losses_by_temp[10.0]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_loss_on_cuda(self, criterion):
        student_output = {
            "predictions": torch.randn(2, 85, 20, 20).cuda(),
            "adapted_features": {
                "stage1_to_model_4": torch.randn(2, 128, 80, 80).cuda(),
            },
        }
        teacher_features = {"model.4": torch.randn(2, 128, 80, 80).cuda()}
        teacher_logits = torch.randn(2, 85, 20, 20).cuda()

        loss, _ = criterion(student_output, teacher_features, teacher_logits)
        assert loss.device.type == "cuda"
