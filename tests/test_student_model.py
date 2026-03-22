"""Tests for student model architecture.
Validates shapes, forward pass, feature adapters, and GPU compatibility.
"""
import pytest
import torch
from student.student_model import StudentYOLO, FeatureMatchingLayer


class TestFeatureMatchingLayer:
    def test_channel_adaptation(self):
        adapter = FeatureMatchingLayer(64, 128)
        x = torch.randn(2, 64, 80, 80)
        out = adapter(x)
        assert out.shape == (2, 128, 80, 80)

    def test_no_bn(self):
        adapter = FeatureMatchingLayer(64, 128, use_bn=False)
        x = torch.randn(2, 64, 80, 80)
        out = adapter(x)
        assert out.shape == (2, 128, 80, 80)

    def test_gradient_flows(self):
        adapter = FeatureMatchingLayer(64, 128)
        x = torch.randn(2, 64, 80, 80, requires_grad=True)
        out = adapter(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestStudentYOLO:
    def test_forward_no_features(self, mock_teacher_shapes):
        model = StudentYOLO(num_classes=80, teacher_feature_shapes=mock_teacher_shapes)
        x = torch.randn(2, 3, 640, 640)
        out = model(x, return_features=False)
        assert "predictions" in out
        assert out["predictions"].shape[0] == 2  # batch size
        assert out["predictions"].shape[1] == 85  # 5 + 80 classes

    def test_forward_with_features(self, mock_teacher_shapes):
        model = StudentYOLO(num_classes=80, teacher_feature_shapes=mock_teacher_shapes)
        x = torch.randn(2, 3, 640, 640)
        out = model(x, return_features=True)

        assert "predictions" in out
        assert "features" in out
        assert "adapted_features" in out

        # Check we get 3 feature maps (stage1, stage2, stage3)
        assert len(out["features"]) == 3

        # Check adapted features exist
        assert len(out["adapted_features"]) > 0

    def test_adapted_feature_shapes_match_teacher(self, mock_teacher_shapes):
        model = StudentYOLO(num_classes=80, teacher_feature_shapes=mock_teacher_shapes)
        x = torch.randn(1, 3, 640, 640)
        out = model(x, return_features=True)

        for name, feat in out["adapted_features"].items():
            adapter_suffix = name.split("_to_")[-1]
            # Find matching teacher layer (dots replaced with underscores)
            teacher_layer = None
            for k in mock_teacher_shapes:
                if k.replace(".", "_") == adapter_suffix:
                    teacher_layer = k
                    break
            assert teacher_layer is not None, f"No teacher match for {name}"
            expected_channels = mock_teacher_shapes[teacher_layer][1]
            assert feat.shape[1] == expected_channels, (
                f"{name}: got {feat.shape[1]} channels, expected {expected_channels}"
            )

    def test_no_adapters(self):
        model = StudentYOLO(num_classes=80, use_feature_adapters=False)
        x = torch.randn(1, 3, 640, 640)
        out = model(x, return_features=True)
        assert "adapted_features" not in out or len(out["adapted_features"]) == 0

    def test_backward_pass(self, mock_teacher_shapes):
        model = StudentYOLO(num_classes=80, teacher_feature_shapes=mock_teacher_shapes)
        x = torch.randn(1, 3, 640, 640)
        out = model(x, return_features=True)

        loss = out["predictions"].sum()
        for feat in out["adapted_features"].values():
            loss += feat.sum()

        loss.backward()

        # Check all parameters received gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_different_input_sizes(self, mock_teacher_shapes):
        model = StudentYOLO(num_classes=80, teacher_feature_shapes=mock_teacher_shapes)
        for size in [320, 416, 640]:
            x = torch.randn(1, 3, size, size)
            out = model(x, return_features=False)
            assert out["predictions"].shape[0] == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self, mock_teacher_shapes):
        model = StudentYOLO(num_classes=80, teacher_feature_shapes=mock_teacher_shapes).cuda()
        x = torch.randn(2, 3, 640, 640).cuda()
        out = model(x, return_features=True)

        assert out["predictions"].device.type == "cuda"
        for feat in out["features"].values():
            assert feat.device.type == "cuda"
        for feat in out["adapted_features"].values():
            assert feat.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_backward(self, mock_teacher_shapes):
        model = StudentYOLO(num_classes=80, teacher_feature_shapes=mock_teacher_shapes).cuda()
        x = torch.randn(1, 3, 640, 640).cuda()
        out = model(x, return_features=True)
        loss = out["predictions"].sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name} on CUDA"


