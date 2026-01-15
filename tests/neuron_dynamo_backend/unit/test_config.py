"""
Unit tests for neuron_dynamo_backend config module and related utilities
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from torch_neuronx.neuron_dynamo_backend import config
from torch_neuronx.neuron_dynamo_backend.backend import managed_artifacts_directory
from torch_neuronx.neuron_dynamo_backend.config import (
    cleanup_compilation_artifacts,
    get_artifacts_directory,
)


class TestCleanupCompilationArtifacts:
    """Test compilation artifacts cleanup"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        (self.temp_dir / "test_file.txt").write_text("test content")
        get_artifacts_directory.cache_clear()

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_cleanup_preserve_true(self):
        """Test cleanup with preserve=True (default)"""
        assert self.temp_dir.exists()

        cleanup_compilation_artifacts(self.temp_dir, preserve=True)

        # Directory should still exist when preserved
        assert self.temp_dir.exists()
        assert (self.temp_dir / "test_file.txt").exists()

    def test_cleanup_preserve_false(self):
        """Test cleanup with preserve=False"""
        assert self.temp_dir.exists()

        cleanup_compilation_artifacts(self.temp_dir, preserve=False)

        # Directory should be removed when not preserved
        assert not self.temp_dir.exists()

    def test_cleanup_nonexistent_directory(self):
        """Test cleanup of non-existent directory"""
        nonexistent_dir = Path("/nonexistent/directory")

        # Should not raise an exception
        cleanup_compilation_artifacts(nonexistent_dir, preserve=False)

    @patch("shutil.rmtree")
    def test_cleanup_exception_handling(self, mock_rmtree):
        """Test cleanup exception handling"""
        mock_rmtree.side_effect = OSError("Permission denied")

        # Should not raise an exception, just log a warning
        cleanup_compilation_artifacts(self.temp_dir, preserve=False)

        # Should have attempted to remove the directory
        mock_rmtree.assert_called_once_with(self.temp_dir)


class TestManagedArtifactsDirectory:
    """Test managed_artifacts_directory context manager"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = None

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil

        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("torch_neuronx.neuron_dynamo_backend.config.get_artifacts_directory")
    def test_managed_artifacts_directory_preserve_true(self, mock_get_artifacts_dir):
        """Test context manager with preserve artifacts = True"""
        # Set up mocks
        self.temp_dir = Path(tempfile.mkdtemp())
        mock_get_artifacts_dir.return_value = self.temp_dir

        # Create test artifacts
        (self.temp_dir / "test_artifact.txt").write_text("test content")

        # Use context manager
        with (
            patch.dict(os.environ, {"TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true"}),
            managed_artifacts_directory() as (artifacts_dir, preserve_artifacts),
        ):
            assert preserve_artifacts
            assert artifacts_dir == self.temp_dir
            assert artifacts_dir.exists()
            assert (artifacts_dir / "test_artifact.txt").exists()

        # Directory should still exist after exiting context (preserve=True)
        assert self.temp_dir.exists()
        assert (self.temp_dir / "test_artifact.txt").exists()

    @patch("torch_neuronx.neuron_dynamo_backend.config.get_artifacts_directory")
    def test_managed_artifacts_directory_preserve_false(self, mock_get_artifacts_dir):
        """Test context manager with preserve artifacts = False"""
        # Set up mocks
        self.temp_dir = Path(tempfile.mkdtemp())
        mock_get_artifacts_dir.return_value = self.temp_dir

        # Create test artifacts
        (self.temp_dir / "test_artifact.txt").write_text("test content")

        # Use context manager
        with (
            patch.dict(os.environ, {"TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "false"}),
            managed_artifacts_directory() as (artifacts_dir, preserve_artifacts),
        ):
            assert not preserve_artifacts
            assert artifacts_dir == self.temp_dir
            assert artifacts_dir.exists()
            assert (artifacts_dir / "test_artifact.txt").exists()

        # Directory should be cleaned up after exiting context (preserve=False)
        assert not self.temp_dir.exists()

    @patch("torch_neuronx.neuron_dynamo_backend.config.get_artifacts_directory")
    def test_managed_artifacts_directory_exception_handling(self, mock_get_artifacts_dir):
        """Test context manager handles exceptions during execution"""
        # Set up mocks
        self.temp_dir = Path(tempfile.mkdtemp())
        mock_get_artifacts_dir.return_value = self.temp_dir

        # Create test artifacts
        (self.temp_dir / "test_artifact.txt").write_text("test content")

        # Test that cleanup still occurs even if exception is raised
        with (
            pytest.raises(ValueError),
            patch.dict(os.environ, {"TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "false"}),
            managed_artifacts_directory() as (artifacts_dir, preserve_artifacts),
        ):
            assert not preserve_artifacts
            assert artifacts_dir == self.temp_dir
            raise ValueError("Test exception")

        # Directory should still be cleaned up after exception
        assert not self.temp_dir.exists()

    @patch("torch_neuronx.neuron_dynamo_backend.config.cleanup_compilation_artifacts")
    @patch("torch_neuronx.neuron_dynamo_backend.config.get_artifacts_directory")
    def test_managed_artifacts_directory_cleanup_called_correctly(
        self, mock_get_artifacts_dir, mock_cleanup
    ):
        """Test context manager calls cleanup_compilation_artifacts with correct parameters"""
        # Test preserve=False case
        self.temp_dir = Path(tempfile.mkdtemp())
        mock_get_artifacts_dir.return_value = self.temp_dir

        with (
            patch.dict(os.environ, {"TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "true"}),
            managed_artifacts_directory() as _,
        ):
            pass
        mock_cleanup.assert_called_once_with(self.temp_dir, preserve=True)

        # Reset mock and test preserve=True case
        mock_cleanup.reset_mock()
        with (
            patch.dict(os.environ, {"TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "false"}),
            managed_artifacts_directory() as _,
        ):
            pass
        mock_cleanup.assert_called_once_with(self.temp_dir, preserve=False)


class TestConfigUtilities:
    """Test configuration utility functions"""

    def setup_method(self):
        """Set up test fixtures"""
        # Store original values to restore later
        self.original_model_name = config._current_model_name
        self.original_timestamp = config._current_timestamp

    def teardown_method(self):
        """Clean up test fixtures"""
        # Restore original values
        config._current_model_name = self.original_model_name
        config._current_timestamp = self.original_timestamp

    def test_model_name_management(self):
        """Test model name get/set functionality"""
        # Test default model name
        config._current_model_name = None
        model_name = config.get_model_name()
        assert model_name == "model_default"

        # Test setting custom model name
        config.set_model_name("test_model")
        assert config.get_model_name() == "test_model"

        # Test setting None resets to None
        config._current_model_name = None
        assert config._current_model_name is None

    def test_timestamp_management(self):
        """Test timestamp get/reset functionality"""
        # Test timestamp generation
        config._current_timestamp = None
        timestamp1 = config.get_current_timestamp()
        assert timestamp1 is not None
        assert len(timestamp1) > 0

        # Should return same timestamp on subsequent calls
        timestamp2 = config.get_current_timestamp()
        assert timestamp1 == timestamp2

        # Test reset functionality
        config.reset_timestamp()
        timestamp3 = config.get_current_timestamp()
        assert timestamp3 != timestamp1

    def test_timestamp_format(self):
        """Test timestamp format is correct"""
        config.reset_timestamp()
        timestamp = config.get_current_timestamp()

        # Should be in format YYYYMMDD_HHMMSS_ffffff (microsecond precision)
        parts = timestamp.split("_")
        assert len(parts) == 3
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 6  # HHMMSS
        assert len(parts[2]) == 6  # ffffff (microseconds)

    @patch("torch.distributed.get_rank")
    @patch("torch_neuronx.neuron_dynamo_backend.config.get_artifacts_directory")
    def test_get_fx_graph_path(self, mock_get_artifacts_dir, mock_get_rank):
        """Test FX graph path generation"""
        mock_get_rank.return_value = 0
        temp_dir = Path(tempfile.mkdtemp())
        mock_get_artifacts_dir.return_value = temp_dir

        try:
            config.reset_timestamp()
            path = config.get_fx_graph_path("test_model")

            # Check path structure
            assert path.parent.parent.parent == temp_dir  # base_dir
            assert path.parent.parent.name == "fx_graphs"
            assert path.parent.name == "proc_0"
            assert path.name.startswith("test_model_")
            assert path.name.endswith(".fx.txt")

            # Check directory creation
            assert path.parent.exists()
        finally:
            import shutil

            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    @patch("torch_neuronx.neuron_dynamo_backend.config.get_rank")
    @patch("torch_neuronx.neuron_dynamo_backend.config.get_artifacts_directory")
    def test_get_stablehlo_path(self, mock_get_artifacts_dir, mock_get_rank):
        """Test StableHLO path generation"""
        mock_get_rank.return_value = 1
        temp_dir = Path(tempfile.mkdtemp())
        mock_get_artifacts_dir.return_value = temp_dir

        try:
            config.reset_timestamp()
            path = config.get_stablehlo_path("test_model")

            # Check path structure
            assert path.parent.parent.parent == temp_dir  # base_dir
            assert path.parent.parent.name == "stablehlo"
            assert path.parent.name == "proc_1"
            assert path.name.startswith("test_model_")
            assert path.name.endswith(".stablehlo.mlir")

            # Check directory creation
            assert path.parent.exists()
        finally:
            import shutil

            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    @patch("torch_neuronx.neuron_dynamo_backend.config.get_rank")
    @patch("torch_neuronx.neuron_dynamo_backend.config.get_artifacts_directory")
    def test_get_neff_path(self, mock_get_artifacts_dir, mock_get_rank):
        """Test NEFF path generation"""
        mock_get_rank.return_value = 2
        temp_dir = Path(tempfile.mkdtemp())
        mock_get_artifacts_dir.return_value = temp_dir

        try:
            config.reset_timestamp()
            path = config.get_neff_path("test_model")

            # Check path structure
            assert path.parent.parent.parent == temp_dir  # base_dir
            assert path.parent.parent.name == "neff"
            assert path.parent.name == "proc_2"
            assert path.name.startswith("test_model_")
            assert path.name.endswith(".neff")

            # Check directory creation
            assert path.parent.exists()
        finally:
            import shutil

            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    @patch("torch_neuronx.neuron_dynamo_backend.config.get_rank")
    @patch("torch_neuronx.neuron_dynamo_backend.config.get_artifacts_directory")
    def test_get_neuronx_cc_working_dir(self, mock_get_artifacts_dir, mock_get_rank):
        """Test neuronx-cc working directory generation"""
        mock_get_rank.return_value = 3
        temp_dir = Path(tempfile.mkdtemp())
        mock_get_artifacts_dir.return_value = temp_dir

        try:
            path = config.get_neuronx_cc_working_dir()

            # Check path structure
            assert path.parent.parent == temp_dir  # base_dir
            assert path.parent.name == "neuronx_cc"
            assert path.name == "proc_3"

            # Check directory creation
            assert path.exists()
        finally:
            import shutil

            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    @patch("torch_neuronx.neuron_dynamo_backend.config.get_rank")
    @patch("torch_neuronx.neuron_dynamo_backend.config.get_artifacts_directory")
    def test_get_err_mlir_path(self, mock_get_artifacts_dir, mock_get_rank):
        """Test error MLIR path generation"""
        mock_get_rank.return_value = 1
        temp_dir = Path(tempfile.mkdtemp())
        mock_get_artifacts_dir.return_value = temp_dir

        try:
            config.reset_timestamp()
            config.set_model_name("test_model")
            path = config.get_err_mlir_path()

            # Check path structure
            assert path.parent.parent.parent == temp_dir  # base_dir
            assert path.parent.parent.name == "torch_mlir_error"
            assert path.parent.name == "proc_1"
            assert path.name.startswith("test_model_")
            assert path.name.endswith(".mlir")

            # Check directory creation
            assert path.parent.exists()
        finally:
            import shutil

            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def test_path_generation_without_model_name(self):
        """Test path generation when model name is None"""
        with (
            patch("torch.distributed.get_rank", return_value=0),
            patch(
                "torch_neuronx.neuron_dynamo_backend.config.get_artifacts_directory"
            ) as mock_get_dir,
        ):
            temp_dir = Path(tempfile.mkdtemp())
            mock_get_dir.return_value = temp_dir

            try:
                config.reset_timestamp()
                timestamp = config.get_current_timestamp()

                # Test FX graph path without model name
                fx_path = config.get_fx_graph_path(None)
                assert fx_path.name == f"{timestamp}.fx.txt"

                # Test StableHLO path without model name
                stablehlo_path = config.get_stablehlo_path(None)
                assert stablehlo_path.name == f"{timestamp}.stablehlo.mlir"

                # Test NEFF path without model name
                neff_path = config.get_neff_path(None)
                assert neff_path.name == f"{timestamp}.neff"

            finally:
                import shutil

                if temp_dir.exists():
                    shutil.rmtree(temp_dir)


class TestDebugDirEnvironmentVariables:
    """Test debug directory environment variable handling"""

    def setup_method(self):
        """Set up test fixtures"""
        get_artifacts_directory.cache_clear()

    def test_torch_neuronx_debug_dir_absolute_path(self):
        """Test TORCH_NEURONX_DEBUG_DIR with absolute path"""
        debug_path = "/tmp/neuron_debug_test"
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": debug_path}):
            result = config.get_artifacts_directory()
            assert result == Path(debug_path)
            assert result.is_absolute()

    def test_torch_neuronx_debug_dir_relative_path(self):
        """Test TORCH_NEURONX_DEBUG_DIR with relative path"""
        debug_path = "relative/debug/path"
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": debug_path}):
            result = config.get_artifacts_directory()
            assert result == Path(debug_path)

    def test_torch_neuronx_debug_dir_with_spaces(self):
        """Test TORCH_NEURONX_DEBUG_DIR with spaces in path"""
        debug_path = "/tmp/path with spaces/debug dir"
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": debug_path}):
            result = config.get_artifacts_directory()
            assert result == Path(debug_path)
            assert "spaces" in str(result)

    def test_debug_dir_preservation_environment_var(self):
        """Test that debug directory is preserved based on environment variable"""
        # This tests the core functionality for debug dir location + preservation environment vars
        debug_path = "/tmp/preserved/debug/artifacts"
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": debug_path}):
            result1 = config.get_artifacts_directory()
        result2 = config.get_artifacts_directory()
        new_debug_path = "/tmp/preserved/debug/artifacts2"
        with patch.dict(os.environ, {"TORCH_NEURONX_DEBUG_DIR": new_debug_path}):
            result3 = config.get_artifacts_directory()

        # Verify that the debug directory is set correctly and does not change
        assert result1 == Path(debug_path)
        assert result2 == Path(debug_path)
        assert result3 == Path(debug_path)

    @patch("tempfile.mkdtemp")
    def test_default_debug_dir_creation(self, mock_mkdtemp):
        """Test that default debug directory is created with proper prefix"""
        mock_temp_dir = "/tmp/neuron_backend_abc123"
        mock_mkdtemp.return_value = mock_temp_dir
        with patch.dict(os.environ, {}, clear=True):
            result = config.get_artifacts_directory()
            assert result == Path(mock_temp_dir)


class TestDistributedUtilities:
    """Test distributed computing utility functions"""

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    def test_get_rank(self, mock_get_rank, mock_torch_dist_init):
        """Test get_rank function"""
        mock_get_rank.return_value = 5
        mock_torch_dist_init.return_value = True
        result = config.get_rank()
        assert result == 5
        mock_get_rank.assert_called_once()

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_node_local_rank")
    def test_get_local_rank(self, mock_get_local_rank, mock_torch_dist_init):
        """Test get_local_rank function"""
        mock_get_local_rank.return_value = 2
        mock_torch_dist_init.return_value = True
        result = config.get_local_rank()
        assert result == 2
        mock_get_local_rank.assert_called_once()

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_world_size")
    def test_get_world_size(self, mock_get_world_size, mock_torch_dist_init):
        """Test get_world_size function"""
        mock_get_world_size.return_value = 8
        mock_torch_dist_init.return_value = True
        result = config.get_world_size()
        assert result == 8
        mock_get_world_size.assert_called_once()

    @patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "4"})
    def test_get_local_world_size_with_env_var(self):
        """Test get_local_world_size with environment variable set"""
        result = config.get_local_world_size()
        assert result == 4

    @patch.dict(os.environ, {}, clear=True)
    def test_get_local_world_size_without_env_var(self):
        """Test get_local_world_size without environment variable"""
        result = config.get_local_world_size()
        assert result == 1
