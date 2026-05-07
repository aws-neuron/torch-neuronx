import glob
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.kernels.test_compiler_subprocess import MockCompilerConfig
from torch_neuronx.kernels.compiler_subprocess import (
    CompilerSubprocess,
)


@pytest.fixture
def mock_config():
    """Mock compiler configuration."""
    return MockCompilerConfig()


@pytest.fixture
def sample_hlo():
    """Sample HLO protobuf data."""
    return b"mock_hlo_protobuf_data"


@pytest.fixture
def sample_neff():
    """Sample NEFF data."""
    return b"mock_compiled_neff_data"


class TestCompileHloProtobufToNeffWorkdir:
    """Test compile_hlo_protobuf_to_neff working directory handling."""

    @pytest.fixture
    def mock_run_neuronx_cc(self):
        """Mock run_neuronx_cc to avoid actual compilation."""
        with patch.object(CompilerSubprocess, "run_neuronx_cc") as mock:
            yield mock

    def test_absolute_path_workdir_nonexistent(self, mock_run_neuronx_cc, mock_config, sample_hlo):
        """Test that absolute path workdir is created if nonexistent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a path that doesn't exist yet
            workdir = Path(tmpdir) / "nonexistent" / "nested" / "workdir"
            assert not workdir.exists()

            # Mock output file reading
            expected_neff = b"compiled_neff_data"
            with (
                patch.object(Path, "read_bytes", return_value=expected_neff),
                patch.dict(
                    os.environ,
                    {
                        "TORCH_NEURONX_DEBUG_DIR": str(workdir),
                        "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "1",
                    },
                ),
            ):
                result = CompilerSubprocess.compile_hlo_protobuf_to_neff(
                    sample_hlo, mock_config, "lnc1"
                )

            # Verify workdir was created
            assert workdir.exists()
            assert workdir.is_dir()
            assert result == expected_neff

    def test_absolute_path_workdir_existent(self, mock_run_neuronx_cc, mock_config, sample_hlo):
        """Test that existing absolute path workdir is used without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "existing_workdir"
            workdir.mkdir(parents=True)
            assert workdir.exists()

            # Add a pre-existing file to verify we don't destroy content
            existing_file = workdir / "existing_file.txt"
            existing_file.write_text("pre-existing content")

            expected_neff = b"compiled_neff_data"
            with (
                patch.object(Path, "read_bytes", return_value=expected_neff),
                patch.dict(
                    os.environ,
                    {
                        "TORCH_NEURONX_DEBUG_DIR": str(workdir),
                        "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "1",
                    },
                ),
            ):
                result = CompilerSubprocess.compile_hlo_protobuf_to_neff(
                    sample_hlo, mock_config, "lnc1"
                )

            # Verify workdir still exists and pre-existing content preserved
            assert workdir.exists()
            assert existing_file.exists()
            assert existing_file.read_text() == "pre-existing content"
            assert result == expected_neff

    def test_relative_path_workdir_nonexistent(self, mock_run_neuronx_cc, mock_config, sample_hlo):
        """Test that relative path workdir is created if nonexistent."""
        # Use a relative path within a temp directory context
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # Relative path that doesn't exist
                relative_workdir = Path("relative") / "nested" / "workdir"
                assert not relative_workdir.exists()

                expected_neff = b"compiled_neff_data"
                with (
                    patch.object(Path, "read_bytes", return_value=expected_neff),
                    patch.dict(
                        os.environ,
                        {
                            "TORCH_NEURONX_DEBUG_DIR": str(relative_workdir),
                            "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "1",
                        },
                    ),
                ):
                    result = CompilerSubprocess.compile_hlo_protobuf_to_neff(
                        sample_hlo, mock_config, "lnc1"
                    )

                # Verify relative workdir was created
                assert relative_workdir.exists()
                assert relative_workdir.is_dir()
                assert result == expected_neff
            finally:
                os.chdir(original_cwd)

    def test_relative_path_workdir_existent(self, mock_run_neuronx_cc, mock_config, sample_hlo):
        """Test that existing relative path workdir is used without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                relative_workdir = Path("existing_relative_workdir")
                relative_workdir.mkdir(parents=True)
                assert relative_workdir.exists()

                expected_neff = b"compiled_neff_data"
                with (
                    patch.object(Path, "read_bytes", return_value=expected_neff),
                    patch.dict(
                        os.environ,
                        {
                            "TORCH_NEURONX_DEBUG_DIR": str(relative_workdir),
                            "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "1",
                        },
                    ),
                ):
                    result = CompilerSubprocess.compile_hlo_protobuf_to_neff(
                        sample_hlo, mock_config, "lnc1"
                    )

                assert relative_workdir.exists()
                assert result == expected_neff
            finally:
                os.chdir(original_cwd)

    def test_workdir_files_created_correctly(self, mock_run_neuronx_cc, mock_config, sample_hlo):
        """Test that HLO input file is created in workdir with correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "workdir"

            expected_neff = b"compiled_neff_data"

            # Use actual file operations to verify HLO file creation
            def mock_read_bytes(self):
                # Only return NEFF for output file
                if self.name == "module.neff":
                    return expected_neff
                raise FileNotFoundError(f"Unexpected read: {self}")

            with (
                patch.object(Path, "read_bytes", mock_read_bytes),
                patch.dict(
                    os.environ,
                    {
                        "TORCH_NEURONX_DEBUG_DIR": str(workdir),
                        "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "1",
                    },
                ),
            ):
                result = CompilerSubprocess.compile_hlo_protobuf_to_neff(
                    sample_hlo, mock_config, "lnc1"
                )

            # Verify HLO file was created with correct content
            hlo_file = glob.glob(str(workdir) + "/neuronx-cc/*/module.hlo.pb")

            assert hlo_file
            assert Path(hlo_file[0]).read_bytes() == sample_hlo
            assert result == expected_neff

    def test_workdir_stablehlo_ir_type(self, mock_run_neuronx_cc, mock_config, sample_hlo):
        """Test that StableHLO IR type creates correct input file name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "workdir"

            expected_neff = b"compiled_neff_data"

            def mock_read_bytes(self):
                if self.name == "module.neff":
                    return expected_neff
                raise FileNotFoundError(f"Unexpected read: {self}")

            with (
                patch.object(Path, "read_bytes", mock_read_bytes),
                patch.dict(
                    os.environ,
                    {
                        "TORCH_NEURONX_DEBUG_DIR": str(workdir),
                        "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "1",
                    },
                ),
            ):
                result = CompilerSubprocess.compile_hlo_protobuf_to_neff(
                    sample_hlo, mock_config, "lnc1", ir_type="StableHLO"
                )

            # Verify MLIR file was created (not HLO)
            mlir_file = glob.glob(str(workdir) + "/neuronx-cc/*/module.mlir")
            hlo_file = glob.glob(str(workdir) + "/neuronx-cc/*/module.hlo.pb")
            assert mlir_file
            assert not hlo_file
            assert Path(mlir_file[0]).read_bytes() == sample_hlo
            assert result == expected_neff

    def test_workdir_run_neuronx_cc_called_with_correct_args(
        self, mock_run_neuronx_cc, mock_config, sample_hlo
    ):
        """Test that run_neuronx_cc is called with correct file names and workdir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "workdir"

            expected_neff = b"compiled_neff_data"
            with (
                patch.object(Path, "read_bytes", return_value=expected_neff),
                patch.dict(
                    os.environ,
                    {
                        "TORCH_NEURONX_DEBUG_DIR": str(workdir),
                        "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "1",
                    },
                ),
            ):
                CompilerSubprocess.compile_hlo_protobuf_to_neff(
                    sample_hlo, mock_config, "lnc_override"
                )

            # Verify run_neuronx_cc was called with correct arguments
            args, _ = mock_run_neuronx_cc.call_args
            assert args[0] == mock_config
            assert args[1] == "module.hlo.pb"  # Input file name (relative)
            assert args[2] == "module.neff"  # Output file name (relative)
            assert args[3] == "lnc_override"
            assert re.match(str(workdir) + "/neuronx-cc/[a-f0-9]+", args[4])

    def test_workdir_deeply_nested_path(self, mock_run_neuronx_cc, mock_config, sample_hlo):
        """Test that deeply nested workdir paths are created correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a deeply nested path
            workdir = Path(tmpdir) / "a" / "b" / "c" / "d" / "e" / "workdir"
            assert not workdir.exists()

            expected_neff = b"compiled_neff_data"
            with (
                patch.object(Path, "read_bytes", return_value=expected_neff),
                patch.dict(
                    os.environ,
                    {
                        "TORCH_NEURONX_DEBUG_DIR": str(workdir),
                        "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "1",
                    },
                ),
            ):
                result = CompilerSubprocess.compile_hlo_protobuf_to_neff(
                    sample_hlo, mock_config, "lnc1"
                )

            # Verify entire path structure was created
            assert workdir.exists()
            assert workdir.is_dir()
            assert (Path(tmpdir) / "a" / "b" / "c" / "d" / "e").exists()
            assert result == expected_neff

    def test_workdir_with_special_characters_in_path(
        self, mock_run_neuronx_cc, mock_config, sample_hlo
    ):
        """Test workdir with special characters in path name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Path with spaces and special chars (common in user directories)
            workdir = Path(tmpdir) / "work dir" / "test-path_123"

            expected_neff = b"compiled_neff_data"
            with (
                patch.object(Path, "read_bytes", return_value=expected_neff),
                patch.dict(
                    os.environ,
                    {
                        "TORCH_NEURONX_DEBUG_DIR": str(workdir),
                        "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "1",
                    },
                ),
            ):
                result = CompilerSubprocess.compile_hlo_protobuf_to_neff(
                    sample_hlo, mock_config, "lnc1"
                )

            assert workdir.exists()
            assert result == expected_neff

    def test_workdir_symlink_path(self, mock_run_neuronx_cc, mock_config, sample_hlo):
        """Test that symlinked workdir path works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create actual directory and symlink to it
            actual_dir = Path(tmpdir) / "actual_workdir"
            actual_dir.mkdir()
            symlink_path = Path(tmpdir) / "symlink_workdir"
            symlink_path.symlink_to(actual_dir)

            expected_neff = b"compiled_neff_data"
            with (
                patch.object(Path, "read_bytes", return_value=expected_neff),
                patch.dict(
                    os.environ,
                    {
                        "TORCH_NEURONX_DEBUG_DIR": str(symlink_path),
                        "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "1",
                    },
                ),
            ):
                result = CompilerSubprocess.compile_hlo_protobuf_to_neff(
                    sample_hlo, mock_config, "lnc1"
                )

            # Verify symlink is followed
            assert symlink_path.exists()
            assert symlink_path.is_symlink()
            # HLO file should be created in actual directory via symlink
            hlo_file = glob.glob(str(symlink_path) + "/neuronx-cc/*/module.hlo.pb")
            assert hlo_file
            assert result == expected_neff

    def test_workdir_current_directory(self, mock_run_neuronx_cc, mock_config, sample_hlo):
        """Test using current directory (.) as workdir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                workdir = Path(".")

                expected_neff = b"compiled_neff_data"
                with (
                    patch.object(Path, "read_bytes", return_value=expected_neff),
                    patch.dict(
                        os.environ,
                        {
                            "TORCH_NEURONX_DEBUG_DIR": str(workdir),
                            "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "1",
                        },
                    ),
                ):
                    result = CompilerSubprocess.compile_hlo_protobuf_to_neff(
                        sample_hlo, mock_config, "lnc1"
                    )

                # Verify files created in current directory
                hlo_file = glob.glob(str(workdir) + "/neuronx-cc/*/module.hlo.pb")
                assert hlo_file
                assert result == expected_neff
            finally:
                os.chdir(original_cwd)
