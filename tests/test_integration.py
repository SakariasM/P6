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
from student.student_model import StudentYOLO
from training.hybrid_distillation_train import HybridDistillationLoss


def make_fake_predictions(n=4, num_classes=80):
    """Create synthetic HybridTeacherPredictions for testing."""
    preds = []
    for i in range(n):
        pred = HybridTeacherPrediction(
            image_path=f"/fake/image_{i}.jpg",
            boxes=[[0.1, 0.2, 0.5, 0.8]],
            confidences=[0.9],
            class_ids=[0],
            class_probs=[[0.9] + [0.1 / (num_classes - 1)] * (num_classes - 1)],
            image_shape=(640, 640, 3),
            features={
                "model.4": torch.randn(128, 80, 80),
                "model.6": torch.randn(256, 40, 40),
                "model.9": torch.randn(512, 20, 20),
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
        teacher_shapes = {
            "model.4": (1, 128, 80, 80),
            "model.6": (1, 256, 40, 40),
            "model.9": (1, 512, 20, 20),
        }

        model = StudentYOLO(
            num_classes=80, teacher_feature_shapes=teacher_shapes
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = HybridDistillationLoss(
            feature_weight=1.0,
            response_weight=1.0,
            temperature=3.0,
            feature_distance="mse",
        )

        losses = []
        for step in range(n_steps):
            images = torch.randn(2, 3, 320, 320, device=device)
            teacher_features = {
                "model.4": torch.randn(2, 128, 40, 40, device=device),
                "model.6": torch.randn(2, 256, 20, 20, device=device),
                "model.9": torch.randn(2, 512, 10, 10, device=device),
            }
            teacher_logits = torch.randn(2, 85, 10, 10, device=device)

            optimizer.zero_grad()
            out = model(images, return_features=True)
            loss, loss_dict = criterion(out, teacher_features, teacher_logits)
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

            # Save
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": 1,
            }, ckpt_path)

            # Load into new model
            teacher_shapes = {
                "model.4": (1, 128, 80, 80),
                "model.6": (1, 256, 40, 40),
                "model.9": (1, 512, 20, 20),
            }
            new_model = StudentYOLO(
                num_classes=80, teacher_feature_shapes=teacher_shapes
            )
            ckpt = torch.load(ckpt_path, weights_only=True)
            new_model.load_state_dict(ckpt["model_state_dict"])

            # Verify outputs match
            x = torch.randn(1, 3, 320, 320)
            model.eval()
            new_model.eval()
            with torch.no_grad():
                out1 = model(x)["predictions"]
                out2 = new_model(x)["predictions"]
            assert torch.allclose(out1, out2, atol=1e-5)

    def test_all_loss_distances(self):
        """Verify training works with all feature distance metrics."""
        teacher_shapes = {
            "model.4": (1, 128, 80, 80),
        }

        for distance in ["mse", "cosine", "at"]:
            model = StudentYOLO(
                num_classes=80, teacher_feature_shapes=teacher_shapes
            )
            criterion = HybridDistillationLoss(
                feature_weight=1.0,
                response_weight=0.0,
                feature_distance=distance,
            )

            images = torch.randn(1, 3, 320, 320)
            teacher_features = {"model.4": torch.randn(1, 128, 40, 40)}

            out = model(images, return_features=True)
            loss, _ = criterion(out, teacher_features, teacher_logits=None)
            loss.backward()

            assert loss.item() > 0, f"Failed with distance={distance}"
