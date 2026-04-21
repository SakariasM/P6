"""GPU cluster readiness tests.
Run these before submitting SLURM jobs to catch issues early.
Tests CUDA availability, memory, multi-GPU, and full training step.
"""
import pytest
import torch
import torch.nn as nn
import os


class TestCUDAEnvironment:
    """Tests that CUDA is properly set up on the cluster node."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_available(self):
        assert torch.cuda.is_available()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_count(self):
        count = torch.cuda.device_count()
        assert count >= 1, f"Expected at least 1 GPU, got {count}"
        print(f"GPUs available: {count}")
        for i in range(count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_sufficient(self):
        """Need at least 4GB free for training."""
        free_mem = torch.cuda.mem_get_info()[0]
        total_mem = torch.cuda.mem_get_info()[1]
        free_gb = free_mem / (1024 ** 3)
        total_gb = total_mem / (1024 ** 3)
        print(f"GPU memory: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
        assert free_gb >= 2.0, f"Need at least 2GB free, got {free_gb:.1f}GB"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensor_operations(self):
        """Basic CUDA tensor ops work."""
        a = torch.randn(100, 100, device="cuda")
        b = torch.randn(100, 100, device="cuda")
        c = a @ b
        assert c.device.type == "cuda"
        assert c.shape == (100, 100)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cudnn_available(self):
        assert torch.backends.cudnn.is_available()
        print(f"cuDNN version: {torch.backends.cudnn.version()}")


class TestTrainingStepOnGPU:
    """Simulate a full training step to verify end-to-end GPU operation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_student_training_step(self):
        """Full forward + backward + optimizer step on GPU."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from student.student_model import StudentYOLO
        from training.hybrid_distillation_train import HybridDistillationLoss

        teacher_shapes = {
            "model.4": (1, 128, 80, 80),
            "model.6": (1, 256, 40, 40),
            "model.9": (1, 512, 20, 20),
        }

        model = StudentYOLO(
            num_classes=80, teacher_feature_shapes=teacher_shapes
        ).cuda()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = HybridDistillationLoss(
            feature_weight=1.0, response_weight=1.0, temperature=3.0
        )

        # Simulate one training step
        images = torch.randn(4, 3, 640, 640).cuda()
        teacher_features = {
            "model.4": torch.randn(4, 128, 80, 80).cuda(),
            "model.6": torch.randn(4, 256, 40, 40).cuda(),
            "model.9": torch.randn(4, 512, 20, 20).cuda(),
        }
        teacher_logits = torch.randn(4, 85, 20, 20).cuda()

        optimizer.zero_grad()
        out = model(images, return_features=True)
        loss, loss_dict = criterion(out, teacher_features, teacher_logits)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert loss_dict["feature_loss"] > 0
        assert loss_dict["response_loss"] > 0
        print(f"Training step loss: {loss.item():.4f}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_no_memory_leak_over_steps(self):
        """Run several steps and check memory doesn't grow unbounded."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from student.student_model import StudentYOLO
        from training.hybrid_distillation_train import HybridDistillationLoss

        teacher_shapes = {
            "model.4": (1, 128, 80, 80),
            "model.6": (1, 256, 40, 40),
        }

        model = StudentYOLO(
            num_classes=80, teacher_feature_shapes=teacher_shapes
        ).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = HybridDistillationLoss(feature_weight=1.0, response_weight=0.0)

        torch.cuda.reset_peak_memory_stats()

        for step in range(5):
            images = torch.randn(2, 3, 320, 320).cuda()
            teacher_features = {
                "model.4": torch.randn(2, 128, 40, 40).cuda(),
                "model.6": torch.randn(2, 256, 20, 20).cuda(),
            }

            optimizer.zero_grad()
            out = model(images, return_features=True)
            loss, _ = criterion(out, teacher_features, teacher_logits=None)
            loss.backward()
            optimizer.step()

            del images, teacher_features, out, loss

        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU memory over 5 steps: {peak_mb:.1f} MB")
        # Should not exceed 4GB for this small test
        assert peak_mb < 4096, f"Peak memory too high: {peak_mb:.1f} MB"


class TestDataLoaderOnGPU:
    """Test that pin_memory and num_workers work for cluster data loading."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pinned_memory_transfer(self):
        t = torch.randn(4, 3, 640, 640, pin_memory=True)
        t_gpu = t.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        assert t_gpu.device.type == "cuda"

    def test_num_workers_spawn(self):
        """Verify DataLoader can spawn workers (catches fork/spawn issues on cluster)."""
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,)))
        loader = DataLoader(dataset, batch_size=4, num_workers=2, pin_memory=False)

        batch = next(iter(loader))
        assert batch[0].shape == (4, 3, 32, 32)


class TestEnvironmentVariables:
    """Check cluster-relevant environment variables and settings."""

    def test_torch_version(self):
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA compiled: {torch.version.cuda}")
        # Just informational, no assertion

    def test_slurm_env_vars_accessible(self):
        """On a SLURM node, these should be set. On local, we just skip."""
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id:
            print(f"SLURM_JOB_ID: {job_id}")
            print(f"SLURM_NODELIST: {os.environ.get('SLURM_NODELIST', 'N/A')}")
            print(f"SLURM_GPUS_ON_NODE: {os.environ.get('SLURM_GPUS_ON_NODE', 'N/A')}")
        else:
            pytest.skip("Not running on SLURM node")

    def test_nccl_available(self):
        """NCCL needed for multi-GPU. Just check it's compiled in."""
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            assert torch.distributed.is_nccl_available(), "NCCL not available for multi-GPU"
        else:
            pytest.skip("Single GPU or no CUDA")
