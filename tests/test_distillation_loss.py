import torch
import pytest
from training.distillation_loss import (
    AttentionTransferLoss,
    FeatureMimicryLoss,
    RelationDistillationLoss,
    SegmentationDistillationLoss,
)


def test_attention_transfer_loss():
    loss_fn = AttentionTransferLoss()
    student_atts = [torch.rand(2, 1, 32, 32), torch.rand(2, 1, 16, 16), torch.rand(2, 1, 8, 8)]
    teacher_atts = [torch.rand(2, 1, 32, 32), torch.rand(2, 1, 16, 16), torch.rand(2, 1, 8, 8)]
    loss = loss_fn(student_atts, teacher_atts)
    assert loss.shape == ()
    assert loss.item() >= 0.0


def test_attention_transfer_handles_size_mismatch():
    loss_fn = AttentionTransferLoss()
    student_atts = [torch.rand(2, 1, 32, 32)]
    teacher_atts = [torch.rand(2, 1, 80, 80)]
    loss = loss_fn(student_atts, teacher_atts)
    assert loss.shape == ()


def test_feature_mimicry_loss():
    loss_fn = FeatureMimicryLoss()
    projected = [torch.randn(2, 128, 32, 32), torch.randn(2, 256, 16, 16)]
    teacher = [torch.randn(2, 128, 80, 80), torch.randn(2, 256, 40, 40)]
    loss = loss_fn(projected, teacher)
    assert loss.shape == ()
    assert loss.item() >= 0.0


def test_relation_distillation_loss():
    loss_fn = RelationDistillationLoss()
    projected = [torch.randn(2, 128, 16, 16)]
    teacher = [torch.randn(2, 128, 32, 32)]
    loss = loss_fn(projected, teacher)
    assert loss.shape == ()
    assert loss.item() >= 0.0


def test_combined_loss():
    loss_fn = SegmentationDistillationLoss(
        attention_weight=1.0,
        mimicry_weight=0.5,
        relation_weight=0.5,
    )
    student_atts = [torch.rand(2, 1, 32, 32, requires_grad=True), torch.rand(2, 1, 16, 16, requires_grad=True)]
    teacher_atts = [torch.rand(2, 1, 80, 80), torch.rand(2, 1, 40, 40)]
    projected = [torch.randn(2, 128, 32, 32, requires_grad=True), torch.randn(2, 256, 16, 16, requires_grad=True)]
    teacher_feats = [torch.randn(2, 128, 80, 80), torch.randn(2, 256, 40, 40)]

    total, loss_dict = loss_fn(
        student_atts=student_atts,
        teacher_atts=teacher_atts,
        projected_student_feats=projected,
        teacher_feats=teacher_feats,
    )
    assert total.shape == ()
    assert total.requires_grad
    assert "attention" in loss_dict
    assert "mimicry" in loss_dict
    assert "relation" in loss_dict
    assert "total" in loss_dict
