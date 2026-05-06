"""
Unit tests for TorchMlirKernel compilation_job context manager integration.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from torch_neuronx.kernels.cache_utils import CompilationJob


class TestTorchMlirKernelCompilationJob:
    """Test TorchMlirKernel's usage of compilation_job context manager."""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_kernel_uses_compilation_job_context_manager(self):
        """Test that TorchMlirKernel uses compilation_job context manager in __call__"""
        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
            },
        ):
            from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

            # Create a simple kernel
            def simple_fn(x):
                return x + 1

            kernel = TorchMlirKernel(simple_fn, "test_add")
            assert kernel.op_name == "test_add"

    def test_kernel_cleanup_on_success_preserve_false(self):
        """Test that kernel cleans up artifacts on successful execution when preserve=False"""
        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "false",
            },
        ):
            from torch_neuronx.kernels.cache_utils import compilation_job

            job_dir = None
            with compilation_job("test_op") as job:
                mock_gm = MagicMock()
                mock_gm.print_readable.return_value = "graph content"
                job.save_fx_graph("test", mock_gm)
                job_dir = job._job_dir
                assert job_dir.exists()

            # After context exit, job dir should be cleaned
            assert not job_dir.exists()

    def test_kernel_preserves_artifacts_on_success_preserve_true(self):
        """Test that kernel preserves artifacts on successful execution when preserve=True"""
        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
            },
        ):
            from torch_neuronx.kernels.cache_utils import compilation_job

            job_dir = None
            with compilation_job("test_op") as job:
                mock_gm = MagicMock()
                mock_gm.print_readable.return_value = "graph content"
                job.save_fx_graph("test", mock_gm)
                job_dir = job._job_dir
                assert job_dir.exists()

            # After context exit, job dir should still exist
            assert job_dir.exists()
            # Verify artifacts were saved
            fx_files = list(job_dir.glob("**/*.fx.txt"))
            assert len(fx_files) > 0

    def test_kernel_cleanup_on_exception_preserve_false(self):
        """Test that kernel cleans up artifacts on exception when preserve=False"""
        from torch_neuronx.kernels.cache_utils import compilation_job

        job_dir = None
        with (
            patch.dict(
                os.environ,
                {
                    "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                    "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "false",
                },
            ),
            pytest.raises(ValueError, match="Simulated compilation error"),
            compilation_job("test_op") as job,
        ):
            mock_gm = MagicMock()
            mock_gm.print_readable.return_value = "graph content"
            job.save_fx_graph("test", mock_gm)
            job_dir = job._job_dir
            assert job_dir.exists()
            raise ValueError("Simulated compilation error")

        # After exception, job dir should be cleaned
        assert job_dir is not None
        assert not job_dir.exists()

    def test_kernel_preserves_artifacts_on_exception_preserve_true(self):
        """Test that kernel preserves artifacts on exception when preserve=True for debugging"""
        from torch_neuronx.kernels.cache_utils import compilation_job

        job_dir = None
        with (
            patch.dict(
                os.environ,
                {
                    "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                    "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
                },
            ),
            pytest.raises(ValueError, match="Simulated compilation error"),
            compilation_job("test_op") as job,
        ):
            mock_gm = MagicMock()
            mock_gm.print_readable.return_value = "graph content"
            job.save_fx_graph("test", mock_gm)
            job_dir = job._job_dir
            assert job_dir.exists()
            raise ValueError("Simulated compilation error")

        # After exception, job dir should still exist for debugging
        assert job_dir is not None
        assert job_dir.exists()
        # Verify artifacts were saved
        fx_files = list(job_dir.glob("**/*.fx.txt"))
        assert len(fx_files) > 0

    def test_kernel_job_accessible_during_execution(self):
        """Test that self.job is accessible during kernel execution"""
        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
            },
        ):
            from torch_neuronx.kernels.cache_utils import compilation_job

            with compilation_job("test_op") as job:
                # Verify job attributes are accessible
                assert job.op_name == "test_op"
                assert job._job_dir.parent == self.temp_dir

                # Verify we can save artifacts
                mock_gm = MagicMock()
                mock_gm.print_readable.return_value = "fx graph"

                fx_path = job.save_fx_graph("model", mock_gm)
                assert fx_path.exists()

                transformed_path = job.save_transformed_fx("model", mock_gm)
                assert transformed_path.exists()

    def test_multiple_kernel_calls_get_unique_jobs(self):
        """Test that each kernel call gets a unique compilation job"""
        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
            },
        ):
            from torch_neuronx.kernels.cache_utils import compilation_job

            job_dirs = []

            for i in range(3):
                with compilation_job(f"test_op_{i}") as job:
                    job_dirs.append(job._job_dir)

            # All job directories should be different
            assert len(set(job_dirs)) == 3
