import torch
import pytest
from student.student_model import StudentSegmentation

TEACHER_CHANNELS = [128, 128, 256]


def test_student_output_shape():
    model = StudentSegmentation(teacher_channels=TEACHER_CHANNELS)
    x = torch.randn(2, 3, 256, 256)
    output, distill_info = model(x)
    assert output.shape == (2, 1, 256, 256), f"Expected (2,1,256,256), got {output.shape}"


def test_student_output_range():
    model = StudentSegmentation(teacher_channels=TEACHER_CHANNELS)
    x = torch.randn(2, 3, 256, 256)
    output, _ = model(x)
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_distill_info_keys():
    model = StudentSegmentation(teacher_channels=TEACHER_CHANNELS)
    x = torch.randn(2, 3, 256, 256)
    _, distill_info = model(x)
    assert "features" in distill_info
    assert "attention_maps" in distill_info
    assert "projected" in distill_info


def test_distill_info_scales():
    model = StudentSegmentation(teacher_channels=TEACHER_CHANNELS)
    x = torch.randn(2, 3, 256, 256)
    _, distill_info = model(x)
    assert len(distill_info["features"]) == 3
    assert len(distill_info["attention_maps"]) == 3
    assert len(distill_info["projected"]) == 3


def test_projected_channels_match_teacher():
    model = StudentSegmentation(teacher_channels=TEACHER_CHANNELS)
    x = torch.randn(2, 3, 256, 256)
    _, distill_info = model(x)
    for proj, t_ch in zip(distill_info["projected"], TEACHER_CHANNELS):
        assert proj.shape[1] == t_ch, f"Expected {t_ch} channels, got {proj.shape[1]}"


def test_attention_maps_are_spatial():
    model = StudentSegmentation(teacher_channels=TEACHER_CHANNELS)
    x = torch.randn(2, 3, 256, 256)
    _, distill_info = model(x)
    for att in distill_info["attention_maps"]:
        assert att.shape[1] == 1
        assert att.min() >= 0.0
        assert att.max() <= 1.0


def test_student_is_differentiable():
    model = StudentSegmentation(teacher_channels=TEACHER_CHANNELS)
    x = torch.randn(2, 3, 128, 128)
    output, distill_info = model(x)
    loss = output.mean() + sum(p.mean() for p in distill_info["projected"])
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            break


def test_student_param_count():
    model = StudentSegmentation(teacher_channels=TEACHER_CHANNELS)
    total = sum(p.numel() for p in model.parameters())
    assert total < 25_000_000, f"Model too large: {total:,} params"
