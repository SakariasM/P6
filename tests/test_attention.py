import torch
import pytest
from student.attention import ChannelAttention, SpatialAttention, CBAM, AttentionProjection


def test_channel_attention_shapes():
    ca = ChannelAttention(64)
    x = torch.randn(2, 64, 32, 32)
    gate, out = ca(x)
    assert gate.shape == (2, 64, 1, 1)
    assert out.shape == (2, 64, 32, 32)


def test_spatial_attention_shapes():
    sa = SpatialAttention()
    x = torch.randn(2, 64, 32, 32)
    attn, out = sa(x)
    assert attn.shape == (2, 1, 32, 32)
    assert out.shape == (2, 64, 32, 32)
    assert attn.min() >= 0.0
    assert attn.max() <= 1.0


def test_cbam_shapes():
    cbam = CBAM(128)
    x = torch.randn(2, 128, 16, 16)
    sp_map, out = cbam(x)
    assert sp_map.shape == (2, 1, 16, 16)
    assert out.shape == (2, 128, 16, 16)


def test_attention_projection():
    proj = AttentionProjection(64, 128)
    x = torch.randn(2, 64, 32, 32)
    out = proj(x)
    assert out.shape == (2, 128, 32, 32)


def test_attention_projection_with_target_size():
    proj = AttentionProjection(64, 128)
    x = torch.randn(2, 64, 32, 32)
    out = proj(x, target_size=(16, 16))
    assert out.shape == (2, 128, 16, 16)
