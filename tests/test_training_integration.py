import torch
import pytest
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, "src")

from training.hybrid_distillation_train import (
    ChunkDataset,
    discover_chunk_files,
    select_teacher_layers,
    compute_teacher_attention,
    collate_fn,
)


def test_select_teacher_layers():
    features = {
        "model.4": torch.randn(128, 80, 80),
        "model.6": torch.randn(128, 40, 40),
        "model.9": torch.randn(256, 20, 20),
    }
    layer_names, channels = select_teacher_layers(features, num_scales=3)
    assert len(layer_names) == 3
    assert len(channels) == 3
    assert set(layer_names) == {"model.4", "model.6", "model.9"}


def test_select_teacher_layers_picks_last_n():
    features = {
        "model.4": torch.randn(128, 80, 80),
        "model.6": torch.randn(128, 40, 40),
        "model.9": torch.randn(256, 20, 20),
        "model.12": torch.randn(256, 20, 20),
        "model.15": torch.randn(512, 10, 10),
    }
    layer_names, channels = select_teacher_layers(features, num_scales=3)
    assert len(layer_names) == 3
    # Should pick the last 3 by index: model.9, model.12, model.15
    assert layer_names == ["model.9", "model.12", "model.15"]


def test_compute_teacher_attention():
    feat = torch.randn(2, 128, 40, 40)
    att = compute_teacher_attention(feat)
    assert att.shape == (2, 1, 40, 40)
    assert att.min() >= 0.0
    assert att.max() <= 1.0


def test_compute_teacher_attention_single():
    feat = torch.randn(128, 40, 40)  # no batch dim
    att = compute_teacher_attention(feat)
    assert att.shape == (1, 1, 40, 40)


def test_discover_chunk_files_from_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            torch.save([], Path(tmpdir) / f"chunk_{i:04d}_worker0.torch")
        files = discover_chunk_files(tmpdir)
        assert len(files) == 3


def test_discover_chunk_files_single_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "predictions.torch"
        torch.save([], f)
        files = discover_chunk_files(str(f))
        assert len(files) == 1
        assert files[0] == f
