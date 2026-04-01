"""Tests specific to AAU AI-LAB cluster environment.
Validates Singularity container, NVIDIA L4 GPU, paths, and resource limits.
Run these on the cluster with: sbatch slurm/run_tests.slurm
"""
import pytest
import torch
import os
import shutil
from pathlib import Path


class TestSingularityEnvironment:
    """Verify we're running inside a Singularity container correctly."""

    def test_singularity_detected(self):
        """Check if we're inside Singularity (set by the container runtime)."""
        in_singularity = (
            os.environ.get("SINGULARITY_CONTAINER") is not None
            or os.environ.get("SINGULARITY_NAME") is not None
            or Path("/.singularity.d").exists()
        )
        if not in_singularity:
            pytest.skip("Not inside Singularity container")

        print(f"Container: {os.environ.get('SINGULARITY_CONTAINER', 'unknown')}")

    def test_nv_flag_gpu_visible(self):
        """Verify --nv flag made GPUs visible inside container."""
        if not torch.cuda.is_available():
            pytest.skip("No CUDA (may not have used --nv flag or not on GPU node)")

        assert torch.cuda.device_count() >= 1
        name = torch.cuda.get_device_name(0)
        print(f"GPU visible inside container: {name}")

    def test_scratch_mounts_exist(self):
        """If running via our SLURM scripts, /scratch/venv and /scratch/project should exist."""
        if not Path("/scratch/venv").exists():
            pytest.skip("Not running via SLURM scripts (no /scratch mounts)")

        assert Path("/scratch/venv").exists(), "/scratch/venv not mounted"
        assert Path("/scratch/project").exists(), "/scratch/project not mounted"

    def test_venv_packages_available(self):
        """Verify extra packages installed in the venv are importable."""
        import importlib
        required = ["ultralytics", "tqdm", "PIL", "pytest"]
        for pkg in required:
            try:
                importlib.import_module(pkg)
            except ImportError:
                pytest.fail(f"Package '{pkg}' not available. Run slurm/setup_env.sh first.")


class TestL4GPU:
    """Tests specific to NVIDIA L4 GPUs on AAU AI-LAB."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_gpu_is_l4(self):
        """Verify we got an L4 (the only GPU type on AI-LAB)."""
        name = torch.cuda.get_device_name(0)
        print(f"GPU: {name}")
        # Don't hard-fail if it's not L4 (could be testing locally)
        if "L4" not in name:
            pytest.skip(f"Not an L4 GPU ({name}), probably not on AI-LAB")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_l4_memory_24gb(self):
        """L4 has 24GB VRAM."""
        total = torch.cuda.get_device_properties(0).total_mem
        total_gb = total / (1024 ** 3)
        print(f"GPU memory: {total_gb:.1f} GB")
        # L4 should have ~24GB
        if total_gb < 20:
            pytest.skip(f"Only {total_gb:.1f}GB VRAM, not an L4")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_training_fits_in_l4(self):
        """Verify a realistic training batch fits within L4's 24GB."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from student.student_model import StudentYOLO

        teacher_shapes = {
            "model.4": (1, 128, 80, 80),
            "model.6": (1, 256, 40, 40),
            "model.9": (1, 512, 20, 20),
        }

        model = StudentYOLO(num_classes=80, teacher_feature_shapes=teacher_shapes).cuda()

        # Simulate realistic batch: 16 images at 640x640
        torch.cuda.reset_peak_memory_stats()

        images = torch.randn(16, 3, 640, 640, device="cuda")
        teacher_feats = {
            "model.4": torch.randn(16, 128, 80, 80, device="cuda"),
            "model.6": torch.randn(16, 256, 40, 40, device="cuda"),
            "model.9": torch.randn(16, 512, 20, 20, device="cuda"),
        }

        out = model(images, return_features=True)
        loss = out["predictions"].sum()
        for v in out["adapted_features"].values():
            loss += v.sum()
        loss.backward()

        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak memory for batch_size=16: {peak_gb:.2f} GB")

        # Must fit in L4's 24GB with headroom
        assert peak_gb < 20.0, f"Batch size 16 uses {peak_gb:.1f}GB, too much for L4"

        del images, teacher_feats, out, loss, model
        torch.cuda.empty_cache()


class TestClusterFilesystem:
    """Test filesystem expectations on AAU AI-LAB."""

    def test_ceph_home_exists(self):
        """On AI-LAB, home is under /ceph/home/."""
        home = Path.home()
        if not str(home).startswith("/ceph/"):
            pytest.skip(f"Not on AI-LAB filesystem (home={home})")
        assert home.exists()

    def test_container_dir_exists(self):
        """Pre-built containers should be at /ceph/container/."""
        container_dir = Path("/ceph/container")
        if not container_dir.exists():
            pytest.skip("Not on AI-LAB (no /ceph/container)")
        assert container_dir.exists()
        pytorch_dir = container_dir / "pytorch"
        assert pytorch_dir.exists(), "No PyTorch containers found"

    def test_write_to_home(self):
        """Verify we can write files (not a read-only filesystem)."""
        test_file = Path.home() / ".kd_write_test"
        try:
            test_file.write_text("test")
            assert test_file.read_text() == "test"
        finally:
            test_file.unlink(missing_ok=True)


class TestSLURMIntegration:
    """Verify SLURM environment variables when running as a job."""

    def test_slurm_job_vars(self):
        """On a SLURM node, job info should be available."""
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            pytest.skip("Not running as SLURM job")

        print(f"SLURM_JOB_ID: {job_id}")
        print(f"SLURM_JOB_NAME: {os.environ.get('SLURM_JOB_NAME', 'N/A')}")
        print(f"SLURM_NODELIST: {os.environ.get('SLURM_NODELIST', 'N/A')}")
        print(f"SLURM_GPUS_ON_NODE: {os.environ.get('SLURM_GPUS_ON_NODE', 'N/A')}")
        print(f"SLURM_CPUS_PER_TASK: {os.environ.get('SLURM_CPUS_PER_TASK', 'N/A')}")
        print(f"SLURM_MEM_PER_NODE: {os.environ.get('SLURM_MEM_PER_NODE', 'N/A')}")

    def test_gpu_allocation_matches_request(self):
        """Number of visible GPUs should match SLURM --gres request."""
        gpus_requested = os.environ.get("SLURM_GPUS_ON_NODE")
        if not gpus_requested:
            pytest.skip("Not running as SLURM job")

        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"Requested: {gpus_requested}, Visible: {visible}")
        assert visible >= int(gpus_requested), (
            f"Requested {gpus_requested} GPUs but only {visible} visible"
        )
