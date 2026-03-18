"""Shared fixtures for all tests."""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path so imports work the same as on cluster
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def device():
    """Return best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def mock_teacher_shapes():
    """Typical YOLO teacher feature shapes."""
    return {
        "model.4": (1, 128, 80, 80),
        "model.6": (1, 256, 40, 40),
        "model.9": (1, 512, 20, 20),
    }


@pytest.fixture
def dummy_batch():
    """A minimal batch of images."""
    return torch.randn(2, 3, 640, 640)


@pytest.fixture
def small_batch():
    """Smaller batch for quick smoke tests."""
    return torch.randn(1, 3, 320, 320)
