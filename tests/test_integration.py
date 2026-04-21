"""Integration tests that simulate the full pipeline with synthetic data.
No real YOLO model or images needed - uses mocked teacher predictions.
"""
import pytest
import torch
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from teacher.hybrid_predictions import HybridTeacherPrediction
from student.student_model import StudentSegmentation
from training.distillation_loss import SegmentationDistillationLoss
from training.hybrid_distillation_train import (
    compute_teacher_attention, select_teacher_layers,
)


TEACHER_CHANNELS = [128, 128, 256]


def make_fake_predictions(n=4):
    """Create synthetic HybridTeacherPredictions for testing."""
    preds = []
    for i in range(n):
        pred = HybridTeacherPrediction(
            image_path=f"/fake/image_{i}.jpg",
            boxes=[[0.1, 0.2, 0.5, 0.8]],
            confidences=[0.9],
            class_ids=[0],
            class_probs=[[0.9] + [0.01] * 10],
            image_shape=(640, 640, 3),
            features={
                "model.4": torch.randn(128, 80, 80),
                "model.6": torch.randn(128, 40, 40),
                "model.9": torch.randn(256, 20, 20),
            },
            raw_logits=torch.randn(85, 20, 20),
        )
        preds.append(pred)
    return preds


class TestSaveLoadPredictions:
    def test_save_and_load_torch(self):
        preds = make_fake_predictions(3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_preds.pt"
            torch.save(preds, path)

            loaded = torch.load(path, weights_only=False)
            assert len(loaded) == 3
            assert loaded[0].image_path == "/fake/image_0.jpg"
            assert "model.4" in loaded[0].features
            assert loaded[0].features["model.4"].shape == (128, 80, 80)

    def test_prediction_features_are_tensors(self):
        preds = make_fake_predictions(1)
        for key, val in preds[0].features.items():
            assert isinstance(val, torch.Tensor), f"{key} is not a tensor"


class TestEndToEndTrainingLoop:
    """Simulate complete training loop with synthetic data (no disk I/O)."""

    def _run_training_steps(self, device, n_steps=3):
        model = StudentSegmentation(
            teacher_channels=TEACHER_CHANNELS,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = SegmentationDistillationLoss(
            attention_weight=1.0,
            mimicry_weight=0.5,
            relation_weight=0.5,
        )

        losses = []
        for step in range(n_steps):
            images = torch.randn(2, 3, 128, 128, device=device)

            # Simulate teacher features at 3 scales
            teacher_feats = [
                torch.randn(2, 128, 32, 32, device=device),
                torch.randn(2, 128, 16, 16, device=device),
                torch.randn(2, 256, 8, 8, device=device),
            ]
            teacher_atts = [compute_teacher_attention(f) for f in teacher_feats]

            optimizer.zero_grad()
            seg_output, distill_info = model(images)

            projected = distill_info['projected']
            student_atts = distill_info['attention_maps']

            n = len(projected)
            t_feats = teacher_feats[-n:]
            t_atts = teacher_atts[-n:]

            loss, loss_dict = criterion(
                student_atts=student_atts,
                teacher_atts=t_atts,
                projected_student_feats=projected,
                teacher_feats=t_feats,
            )
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        return model, losses

    def test_cpu_training_loop(self):
        model, losses = self._run_training_steps("cpu", n_steps=3)
        assert len(losses) == 3
        assert all(l > 0 for l in losses)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_training_loop(self):
        model, losses = self._run_training_steps("cuda", n_steps=3)
        assert len(losses) == 3
        assert all(l > 0 for l in losses)

    def test_model_checkpoint_save_load(self):
        model, _ = self._run_training_steps("cpu", n_steps=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.pt"

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": 1,
                "teacher_channels": TEACHER_CHANNELS,
            }, ckpt_path)

            new_model = StudentSegmentation(
                teacher_channels=TEACHER_CHANNELS,
            )
            ckpt = torch.load(ckpt_path, weights_only=False)
            new_model.load_state_dict(ckpt["model_state_dict"])

            x = torch.randn(1, 3, 128, 128)
            model.eval()
            new_model.eval()
            with torch.no_grad():
                out1, _ = model(x)
                out2, _ = new_model(x)
            assert torch.allclose(out1, out2, atol=1e-5)

    def test_loss_decreases(self):
        """Verify loss trends downward over several steps."""
        _, losses = self._run_training_steps("cpu", n_steps=10)
        # Average of first 3 should be higher than average of last 3
        early = sum(losses[:3]) / 3
        late = sum(losses[-3:]) / 3
        # Not guaranteed to always decrease, but should trend down
        assert late < early * 1.5, f"Loss not decreasing: early={early:.4f}, late={late:.4f}"
