"""
Unit tests for kernels.cache_utils
"""

import os
import re
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from torch_neuronx.kernels.cache_utils import (
    CompilationJob,
    compilation_job,
    is_caching_disabled,
    is_preserve_artifacts_enabled,
)


class TestBooleanEnvironmentVariables:
    """Test the shared environment flag parser."""

    @pytest.mark.parametrize(
        "function,name",
        [
            (is_caching_disabled, "TORCH_NEURONX_NEFF_DISABLE_CACHE"),
            (
                is_preserve_artifacts_enabled,
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS",
            ),
        ],
    )
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "YES"])
    def test_true_values(self, monkeypatch, function, name, value):
        monkeypatch.setenv(name, value)
        assert function() is True

    @pytest.mark.parametrize(
        "function,name",
        [
            (is_caching_disabled, "TORCH_NEURONX_NEFF_DISABLE_CACHE"),
            (
                is_preserve_artifacts_enabled,
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS",
            ),
        ],
    )
    @pytest.mark.parametrize("value", ["", "0", "false", "no", "on"])
    def test_false_values(self, monkeypatch, function, name, value):
        monkeypatch.setenv(name, value)
        assert function() is False


class TestCleanupCompilationArtifacts:
    """Test CompilationJob.cleanup() behavior"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_cleanup_preserve_true(self):
        """Test cleanup with preserve=True keeps job directory"""
        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
            },
        ):
            job = CompilationJob("test_op")
            # Create job directory by saving something
            mock_gm = MagicMock()
            mock_gm.print_readable.return_value = "graph content"
            job.save_fx_graph("test", mock_gm)

            assert job._job_dir.exists()
            job.cleanup()

            # Job directory should still exist when preserved
            assert job._job_dir.exists()

    def test_cleanup_preserve_false(self):
        """Test cleanup with preserve=False removes job directory"""
        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "false",
            },
        ):
            job = CompilationJob("test_op")
            # Create job directory by saving something
            mock_gm = MagicMock()
            mock_gm.print_readable.return_value = "graph content"
            job.save_fx_graph("test", mock_gm)

            assert job._job_dir.exists()
            job.cleanup()

            # Job directory should be removed when not preserved
            assert not job._job_dir.exists()

    def test_cleanup_removes_neuronx_cc_dir(self):
        """Test that cleanup removes job dir and the neuronx-cc directory"""
        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "false",
            },
        ):
            job = CompilationJob("test_op")
            # Create job directory by saving something
            mock_gm = MagicMock()
            mock_gm.print_readable.return_value = "graph content"
            job.save_fx_graph("test", mock_gm)

            # Get the neuronx-cc working directory path (C++ creates it, we mock here)
            cc_workdir = job.get_neuronx_cc_workdir()
            cc_workdir.mkdir(parents=True, exist_ok=True)
            assert cc_workdir.exists()

            # Also create a file inside the job dir to verify it gets cleaned
            (job._job_dir / "test_file.txt").write_text("test content")

            job.cleanup()

            # Job directory should be removed
            assert not job._job_dir.exists()

            # neuronx-cc directory should not exist
            assert not cc_workdir.exists()

    def test_cleanup_exception_handling(self):
        """Test cleanup exception handling"""
        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "false",
            },
        ):
            job = CompilationJob("test_op")
            # Create job directory by saving something
            mock_gm = MagicMock()
            mock_gm.print_readable.return_value = "graph content"
            job.save_fx_graph("test", mock_gm)

            with patch("shutil.rmtree") as mock_rmtree:
                mock_rmtree.side_effect = OSError("Permission denied")

                # Should not raise an exception, just log a warning
                job.cleanup()

                # Should have attempted to remove the directory
                mock_rmtree.assert_called_once_with(job._job_dir)


class TestCompilationJob:
    """Test CompilationJob creation, path methods, and cleanup"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_job_id_format(self):
        """Test that job_id follows the expected format:
        {timestamp}_{op_name_normalized}_{random_hash}"""
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir)}):
            job = CompilationJob("my_op")

            # job_id format: YYYYMMDD_HHMMSS_ffffff_{op_name padded to 16 chars}_{16 hex chars}
            # my_op -> my_op___________ (padded to 16 chars with underscores)
            pattern = r"^\d{8}_\d{6}_\d{6}_my_op____________[0-9a-f]{16}$"
            assert re.match(
                pattern, job.job_id
            ), f"job_id '{job.job_id}' does not match expected format"

    def test_job_attributes(self):
        """Test that CompilationJob sets expected attributes"""
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir)}):
            job = CompilationJob("test_op")

            assert job.op_name == "test_op"
            assert len(job.timestamp) > 0
            assert len(job.job_id) > 0

    def test_job_directory_path(self):
        """Test that job directory path is correctly set"""
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir)}):
            job = CompilationJob("test_op")

            assert job._job_dir == self.temp_dir / job.job_id

    def test_unique_job_ids(self):
        """Test that each CompilationJob gets a unique job_id"""
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir)}):
            job1 = CompilationJob("op_a")
            job2 = CompilationJob("op_b")

            assert job1.job_id != job2.job_id

    def test_save_fx_graph(self):
        """Test FX graph save"""
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir)}):
            mock_gm = MagicMock()
            mock_gm.print_readable.return_value = "graph content"

            job = CompilationJob("test_op")
            path = job.save_fx_graph("MyModel", mock_gm)

            assert path.parent == job._job_dir / "fx_graphs"
            assert path.name == f"mymodel_{job.timestamp}.fx.txt"
            assert path.exists()

    def test_save_transformed_fx(self):
        """Test transformed FX graph save"""
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir)}):
            mock_gm = MagicMock()
            mock_gm.print_readable.return_value = "graph content"

            job = CompilationJob("test_op")
            path = job.save_transformed_fx("MyModel", mock_gm)

            assert path.parent == job._job_dir / "transformed_fx"
            assert path.name == f"mymodel_{job.timestamp}.fx.txt"
            assert path.exists()

    def test_save_raw_torch(self):
        """Test RAW Torch IR save"""
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir)}):
            mock_mlir = MagicMock()
            mock_mlir.operation.get_asm.return_value = "raw torch mlir content"

            job = CompilationJob("test_op")
            path = job.save_raw_torch("MyModel", mock_mlir)

            assert path.parent == job._job_dir / "raw"
            assert path.name == f"mymodel_{job.timestamp}.torch_raw.mlir"
            assert path.exists()
            assert path.read_text() == "raw torch mlir content"
            mock_mlir.operation.get_asm.assert_called_once_with(enable_debug_info=True)

    def test_save_stablehlo(self):
        """Test StableHLO save"""
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir)}):
            mock_mlir = MagicMock()
            mock_mlir.operation.get_asm.return_value = "stablehlo mlir content"

            job = CompilationJob("test_op")
            path = job.save_stablehlo("MyModel", mock_mlir)

            assert path.parent == job._job_dir / "stablehlo"
            assert path.name == f"mymodel_{job.timestamp}.stablehlo.mlir"
            assert path.exists()
            assert path.read_text() == "stablehlo mlir content"
            mock_mlir.operation.get_asm.assert_called_once_with(enable_debug_info=True)

    def test_get_err_mlir_path(self):
        """Test error MLIR path generation"""
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir)}):
            job = CompilationJob("test_op")
            path = job.get_err_mlir_path("MyModel")

            assert path.parent == job._job_dir / "torch_mlir_error"
            assert path.name == f"mymodel_{job.timestamp}.mlir"
            assert path.parent.exists()

    def test_get_neuronx_cc_workdir(self):
        """Test neuronx-cc working directory generation/deletion"""
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir)}):
            job = CompilationJob("test_op")
            workdir = job.get_neuronx_cc_workdir()
            assert not workdir.exists()

    def test_model_name_lowercased_in_paths(self):
        """Test that model names are lowercased in file paths"""
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir)}):
            mock_gm = MagicMock()
            mock_gm.print_readable.return_value = "graph content"

            job = CompilationJob("test_op")
            path = job.save_fx_graph("MyUpperCaseModel", mock_gm)

            assert "myuppercasemodel" in path.name
            assert "MyUpperCaseModel" not in path.name

    def test_preserve_reads_environment_variable(self):
        """Test that preserve flag reads from TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS"""
        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
            },
        ):
            job_preserve = CompilationJob("test_op")
        assert job_preserve.preserve is True

        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "false",
            },
        ):
            job_no_preserve = CompilationJob("test_op")
        assert job_no_preserve.preserve is False


class TestDebugDirEnvironmentVariables:
    """Test debug directory environment variable handling via CompilationJob"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_torch_neuronx_debug_dir_absolute_path(self):
        """Test TORCH_NEURONX_DEBUG_DIR with absolute path - CompilationJob uses it"""
        debug_path = str(self.temp_dir / "neuron_debug_test")
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": debug_path}):
            job = CompilationJob("test_op")

            # Verify job uses the debug directory
            assert job._base_dir == Path(debug_path)
            assert job._job_dir.parent == Path(debug_path)
            assert Path(debug_path).is_absolute()

    def test_torch_neuronx_debug_dir_relative_path(self):
        """Test TORCH_NEURONX_DEBUG_DIR with relative path - CompilationJob uses it"""
        debug_path = "relative/debug/path"
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": debug_path}):
            job = CompilationJob("test_op")

            # Verify job uses the relative debug directory
            assert job._base_dir == Path(debug_path)
            assert job._job_dir.parent == Path(debug_path)

    def test_torch_neuronx_debug_dir_with_spaces(self):
        """Test TORCH_NEURONX_DEBUG_DIR with spaces in path - CompilationJob uses it"""
        debug_path = str(self.temp_dir / "path with spaces" / "debug dir")
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": debug_path}):
            job = CompilationJob("test_op")

            # Verify job uses the debug directory with spaces
            assert job._base_dir == Path(debug_path)
            assert "spaces" in str(job._base_dir)

    def test_debug_dir_different_env_values(self):
        """Test that CompilationJob uses the current env value (not cached)"""
        debug_path1 = str(self.temp_dir / "debug" / "artifacts1")
        debug_path2 = str(self.temp_dir / "debug" / "artifacts2")

        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": debug_path1}):
            job1 = CompilationJob("test_op")

        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": debug_path2}):
            job2 = CompilationJob("test_op")

        # Each job should use the env value that was set when it was created
        assert job1._base_dir == Path(debug_path1)
        assert job2._base_dir == Path(debug_path2)

    def test_default_debug_dir(self):
        """Test that default debug directory is /tmp/neuron_backend when env var not set"""
        with patch.dict(os.environ, {}, clear=True):
            # Remove TORCH_NEURONX_DEBUG_DIR if it exists
            os.environ.pop("TORCH_NEURONX_DEBUG_DIR", None)
            job = CompilationJob("test_op")

            # Verify job uses the default directory
            assert job._base_dir == Path("/tmp/neuron_backend")


class TestCompilationJobContextManager:
    """Test compilation_job context manager behavior"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_context_manager_yields_compilation_job(self):
        """Test that context manager yields a CompilationJob instance"""
        with (
            patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir)}),
            compilation_job("test_op") as job,
        ):
            assert isinstance(job, CompilationJob)
            assert job.op_name == "test_op"

    def test_context_manager_cleanup_preserve_false(self):
        """Test that cleanup is called on normal exit when preserve=False"""
        with (
            patch.dict(
                os.environ,
                {
                    "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                    "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "false",
                },
            ),
            compilation_job("test_op") as job,
        ):
            # Create some artifacts
            mock_gm = MagicMock()
            mock_gm.print_readable.return_value = "graph content"
            job.save_fx_graph("test", mock_gm)
            job_dir = job._job_dir
            assert job_dir.exists()

        # After exiting context, job dir should be cleaned up
        assert not job_dir.exists()

    def test_context_manager_cleanup_preserve_true(self):
        """Test that cleanup preserves artifacts when preserve=True"""
        with (
            patch.dict(
                os.environ,
                {
                    "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                    "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
                },
            ),
            compilation_job("test_op") as job,
        ):
            # Create some artifacts
            mock_gm = MagicMock()
            mock_gm.print_readable.return_value = "graph content"
            job.save_fx_graph("test", mock_gm)
            job_dir = job._job_dir
            assert job_dir.exists()

        # After exiting context, job dir should still exist
        assert job_dir.exists()

    def test_context_manager_cleanup_on_exception_preserve_false(self):
        """Test that cleanup is called even when exception occurs (preserve=False)"""
        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "false",
            },
        ):
            job_dir = None
            try:
                with compilation_job("test_op") as job:
                    mock_gm = MagicMock()
                    mock_gm.print_readable.return_value = "graph content"
                    job.save_fx_graph("test", mock_gm)
                    job_dir = job._job_dir
                    assert job_dir.exists()
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # After exception, job dir should be cleaned up
            assert job_dir is not None
            assert not job_dir.exists()

    def test_context_manager_preserves_on_exception_preserve_true(self):
        """Test that artifacts are preserved on exception when preserve=True"""
        with patch.dict(
            os.environ,
            {
                "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
            },
        ):
            job_dir = None
            try:
                with compilation_job("test_op") as job:
                    mock_gm = MagicMock()
                    mock_gm.print_readable.return_value = "graph content"
                    job.save_fx_graph("test", mock_gm)
                    job_dir = job._job_dir
                    assert job_dir.exists()
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # After exception, job dir should still exist
            assert job_dir is not None
            assert job_dir.exists()

    def test_context_manager_artifacts_accessible_inside(self):
        """Test that artifacts can be created and accessed inside context"""
        with (
            patch.dict(
                os.environ,
                {
                    "TORCH_NEURONX_DEBUG_DIR": str(self.temp_dir),
                    "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true",
                },
            ),
            compilation_job("test_op") as job,
        ):
            mock_gm = MagicMock()
            mock_gm.print_readable.return_value = "graph content"

            # Save various artifacts
            fx_path = job.save_fx_graph("model", mock_gm)
            transformed_path = job.save_transformed_fx("model", mock_gm)

            # Verify artifacts exist
            assert fx_path.exists()
            assert transformed_path.exists()
            assert "graph content" in fx_path.read_text()
