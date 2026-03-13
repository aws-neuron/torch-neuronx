"""Comprehensive tests for NeuronxCCWrapper command-line interface."""

import tempfile
import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest

from torch_neuronx.kernels.cache_utils import atomic_write_bytes
from torch_neuronx.kernels.compiler_subprocess import NEFFCompilationError
from torch_neuronx.kernels.neuronx_cc_wrapper import (
    NeuronxCCWrapper,
    main,
)


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        nfs_dir = temp_path / "nfs_cache"
        local_dir = temp_path / "local_cache"
        nfs_dir.mkdir()
        local_dir.mkdir()

        yield {
            "temp": temp_path,
            "nfs": nfs_dir,
            "local": local_dir,
        }


@pytest.fixture
def sample_hlo():
    """Sample HLO protobuf data."""
    return b"mock_hlo_protobuf_data_for_testing"


@pytest.fixture
def sample_mlir():
    """Sample MLIR data."""
    return b"func.func @main() { return }"


@pytest.fixture
def sample_neff():
    """Sample NEFF compiled data."""
    return b"mock_compiled_neff_data_result"


@pytest.fixture
def mock_compiler_subprocess(sample_neff, monkeypatch, temp_dirs):
    """Mock CompilerSubprocess."""
    mock = Mock()
    mock_instance = Mock()
    mock.return_value = mock_instance

    # Side effect function to write to the output file
    def get_or_compile_side_effect(*args, **kwargs):
        # Write to the expected output file
        output = temp_dirs["temp"] / "test.neff"
        output.write_bytes(sample_neff)
        return sample_neff

    # Default successful behavior with side effect
    mock_instance.get_or_compile.side_effect = get_or_compile_side_effect

    # Patch the CompilerSubprocess class
    monkeypatch.setattr("torch_neuronx.kernels.neuronx_cc_wrapper.CompilerSubprocess", mock)

    return mock


class TestNeuronxCCWrapper:
    """Test NeuronxCCWrapper class functionality."""

    def test_initialization_default(self):
        """Test NeuronxCCWrapper initialization."""
        wrapper = NeuronxCCWrapper()
        assert wrapper.compiler is not None

    def test_parse_arguments(self, sample_hlo, temp_dirs):
        """Test NeuronxCCWrapper argument parsing"""
        wrapper = NeuronxCCWrapper()
        input_file = temp_dirs["temp"] / "test.hlo.pb"
        output = temp_dirs["temp"] / "test.neff"
        atomic_write_bytes(input_file, sample_hlo)

        args = [
            "compile",
            str(input_file),
            "--output",
            str(output),
            "--framework",
            "XLA",
            "--target",
            "trn1",
            "--model-type",
            "transformer",
            "--auto-cast",
            "fp16",
            "--lnc",
            "1",
            "--workdir",
            str(temp_dirs["temp"]),
            "-O2",
            "--custom-flag",
            "custom-value",
        ]

        parsed = wrapper._parse_neuronx_cc_args(args)

        # Verify I/O paths are returned separately in ParsedArgs, not in config
        assert parsed.input_file == str(input_file)
        assert parsed.output_file == str(output)
        assert parsed.workdir == str(temp_dirs["temp"])

        # Verify config contains only compilation parameters
        assert parsed.config.framework == "XLA"
        assert parsed.config.target == "trn1"
        assert parsed.config.model_type == "transformer"
        assert parsed.config.auto_cast == "fp16"
        assert parsed.lnc_override == "1"
        assert parsed.config.optimization_level == "-O2"
        assert parsed.config.extra_flags == ["--custom-flag", "custom-value"]

        # Verify config does NOT have input/output/workdir attributes
        assert not hasattr(parsed.config, "input")
        assert not hasattr(parsed.config, "output")
        assert not hasattr(parsed.config, "workdir")

    def test_successful_compilation_hlo(
        self, temp_dirs, sample_hlo, sample_neff, mock_compiler_subprocess
    ):
        """Test successful compilation with HLO input."""
        # Setup files
        input_file = temp_dirs["temp"] / "test.hlo.pb"
        output = temp_dirs["temp"] / "test.neff"
        atomic_write_bytes(input_file, sample_hlo)

        wrapper = NeuronxCCWrapper()

        args = [
            "compile",
            str(input_file),
            "--output",
            str(output),
            "--framework",
            "XLA",
            "--target",
            "trn2",
            "--workdir",
            str(temp_dirs["temp"]),
        ]

        result = wrapper.compile_with_caching(args)

        assert result == 0
        assert output.exists()
        assert output.read_bytes() == sample_neff

        # Verify CompilerSubprocess was called correctly
        mock_compiler_subprocess.return_value.get_or_compile.assert_called_once()
        call_args = mock_compiler_subprocess.return_value.get_or_compile.call_args
        assert call_args[1]["hlo_protobuf"] == sample_hlo
        assert call_args[1]["ir_type"] == "XLA"

    def test_successful_compilation_mlir(
        self, temp_dirs, sample_mlir, sample_neff, mock_compiler_subprocess
    ):
        """Test successful compilation with MLIR input."""
        # Setup files
        input_file = temp_dirs["temp"] / "test.mlir"
        output = temp_dirs["temp"] / "test.neff"
        atomic_write_bytes(input_file, sample_mlir)

        wrapper = NeuronxCCWrapper()

        args = [
            "compile",
            str(input_file),
            "--output",
            str(output),
            "--framework",
            "StableHLO",
            "--target",
            "trn2",
            "--workdir",
            str(temp_dirs["temp"]),
        ]

        result = wrapper.compile_with_caching(args)

        assert result == 0
        assert output.exists()
        assert output.read_bytes() == sample_neff

        # Verify MLIR was detected correctly
        call_args = mock_compiler_subprocess.return_value.get_or_compile.call_args
        assert call_args[1]["ir_type"] == "StableHLO"

    def test_compilation_missing_input_file(self, temp_dirs):
        """Test compilation with missing input file."""
        wrapper = NeuronxCCWrapper()

        args = [
            "compile",
            "nonexistent.hlo.pb",
            "--output",
            "/tmp/test.neff",
            "--workdir",
            str(temp_dirs["temp"]),
        ]

        result = wrapper.compile_with_caching(args)

        assert result == 1  # Should fail due to missing input

    def test_compilation_missing_output(self, temp_dirs, sample_hlo):
        """Test compilation with missing output file specification."""
        input_file = temp_dirs["temp"] / "test.hlo.pb"
        atomic_write_bytes(input_file, sample_hlo)

        wrapper = NeuronxCCWrapper()

        args = [
            "compile",
            str(input_file),
            "--framework",
            "XLA",
            "--workdir",
            str(temp_dirs["temp"]),
        ]

        result = wrapper.compile_with_caching(args)

        assert result == 1  # Should fail due to missing output

    def test_compilation_missing_workdir(self, temp_dirs, sample_hlo):
        """Test compilation with missing working directory specification."""
        input_file = temp_dirs["temp"] / "test.hlo.pb"
        atomic_write_bytes(input_file, sample_hlo)

        wrapper = NeuronxCCWrapper()

        args = ["compile", str(input_file), "--framework", "XLA", "--output", "/tmp/test.neff"]

        result = wrapper.compile_with_caching(args)

        assert result == 1  # Should fail due to missing output

    def test_compilation_neff_error(self, temp_dirs, sample_hlo, mock_compiler_subprocess):
        """Test compilation when CompilerSubprocess raises NEFFCompilationError."""
        input_file = temp_dirs["temp"] / "test.hlo.pb"
        output = temp_dirs["temp"] / "test.neff"
        atomic_write_bytes(input_file, sample_hlo)

        mock_compiler_subprocess.return_value.get_or_compile.side_effect = NEFFCompilationError(
            "Compilation failed"
        )

        wrapper = NeuronxCCWrapper()

        args = ["compile", str(input_file), "--output", str(output), "--framework", "XLA"]

        result = wrapper.compile_with_caching(args)

        assert result == 1
        assert not output.exists()

    def test_compilation_generic_exception(self, temp_dirs, sample_hlo, mock_compiler_subprocess):
        """Test compilation when generic exception is raised."""
        input_file = temp_dirs["temp"] / "test.hlo.pb"
        output = temp_dirs["temp"] / "test.neff"
        atomic_write_bytes(input_file, sample_hlo)

        mock_compiler_subprocess.return_value.get_or_compile.side_effect = RuntimeError(
            "Unknown error"
        )

        wrapper = NeuronxCCWrapper()

        args = ["compile", str(input_file), "--output", str(output), "--framework", "XLA"]

        result = wrapper.compile_with_caching(args)

        assert result == 1
        assert not output.exists()

    def test_caching_behavior(self, temp_dirs, sample_hlo, sample_neff, monkeypatch):
        """Test that caching behavior works through the wrapper"""

        # Use unique HLO content to ensure unique cache key
        unique_hlo = sample_hlo + uuid.uuid4().hex.encode()

        input_file = temp_dirs["temp"] / "test.hlo.pb"
        output = temp_dirs["temp"] / "test.neff"

        atomic_write_bytes(input_file, unique_hlo)

        # Mock the compilation method
        def write_neff_side_effect(*args, **kwargs):
            # Write to the expected output file
            output = temp_dirs["temp"] / "test.neff"
            output.write_bytes(sample_neff)
            return sample_neff

        mock_compile = Mock(side_effect=write_neff_side_effect)
        monkeypatch.setattr(
            "torch_neuronx.kernels.compiler_subprocess.CompilerSubprocess.compile_hlo_protobuf_to_neff",
            mock_compile,
        )

        # Also mock neuronxcc version for cache key generation
        monkeypatch.setattr("neuronxcc.__version__", "2.0.0")

        wrapper = NeuronxCCWrapper()

        args = [
            "compile",
            str(input_file),
            "--output",
            str(output),
            "--framework",
            "XLA",
            "--target",
            "trn1",
            "--workdir",
            str(temp_dirs["temp"]),
        ]

        # First compilation - should call compile_hlo_protobuf_to_neff
        result1 = wrapper.compile_with_caching(args)
        assert result1 == 0
        assert output.exists()
        assert output.read_bytes() == sample_neff

        # Second compilation - should use cache and NOT call compile_hlo_protobuf_to_neff again
        result2 = wrapper.compile_with_caching(args)
        assert result2 == 0

        # Verify compile_hlo_protobuf_to_neff was only called once (proving caching worked)
        assert (
            mock_compile.call_count == 1
        ), "Expected compile_hlo_protobuf_to_neff to be called once, but it was "
        f"called {mock_compile.call_count} times"

        # Verify the parameters of the single call
        call_args = mock_compile.call_args
        assert call_args[0][0] == unique_hlo  # First arg is hlo_protobuf

    def test_neuron_cc_flags_with_existing_args(self, sample_hlo, temp_dirs, monkeypatch):
        """Test that NEURON_CC_FLAGS is prepended to existing extra arguments."""
        wrapper = NeuronxCCWrapper()
        input_file = temp_dirs["temp"] / "test.hlo.pb"
        output = temp_dirs["temp"] / "test.neff"
        atomic_write_bytes(input_file, sample_hlo)

        # Set NEURON_CC_FLAGS environment variable
        monkeypatch.setenv("NEURON_CC_FLAGS", "--env-flag env-value")

        args = [
            "compile",
            str(input_file),
            "--output",
            str(output),
            "--framework",
            "XLA",
            "--existing-flag",
            "existing-value",
            "--workdir",
            str(temp_dirs["temp"]),
        ]

        parsed = wrapper._parse_neuronx_cc_args(args)

        # Verify NEURON_CC_FLAGS are prepended before existing extra args
        expected_extra_flags = ["--env-flag", "env-value", "--existing-flag", "existing-value"]
        assert parsed.config.extra_flags == expected_extra_flags

    def test_neuron_cc_flags_empty(self, sample_hlo, temp_dirs, monkeypatch):
        """Test behavior when NEURON_CC_FLAGS is empty."""
        wrapper = NeuronxCCWrapper()
        input_file = temp_dirs["temp"] / "test.hlo.pb"
        output = temp_dirs["temp"] / "test.neff"
        atomic_write_bytes(input_file, sample_hlo)

        # Test with empty string
        monkeypatch.setenv("NEURON_CC_FLAGS", "")
        args = [
            "compile",
            str(input_file),
            "--output",
            str(output),
            "--framework",
            "XLA",
            "--workdir",
            str(temp_dirs["temp"]),
        ]

        parsed = wrapper._parse_neuronx_cc_args(args)
        assert parsed.config.extra_flags == []

    def test_input_file_corrupted_data(self, temp_dirs, mock_compiler_subprocess, monkeypatch):
        """Test compilation when input file has corrupted/unreadable data."""
        wrapper = NeuronxCCWrapper()

        input_file = temp_dirs["temp"] / "input.hlo.pb"
        output = temp_dirs["temp"] / "output.neff"

        # Create input file first
        atomic_write_bytes(input_file, b"valid_data")

        # Mock the file reading to raise OSError
        original_open = open

        def mock_open(file, mode="r", **kwargs):
            if str(file).endswith("input.hlo.pb") and "rb" in mode:
                raise OSError("Input/output error")
            return original_open(file, mode, **kwargs)

        monkeypatch.setattr("builtins.open", mock_open)

        args = ["compile", str(input_file), "--output", str(output)]
        result = wrapper.compile_with_caching(args)

        assert result == 1  # Should fail due to I/O error
        assert not output.exists()

    def test_output_disk_full(
        self, temp_dirs, sample_hlo, sample_neff, mock_compiler_subprocess, monkeypatch
    ):
        """Test compilation when output file fails due to disk full."""
        wrapper = NeuronxCCWrapper()

        input_file = temp_dirs["temp"] / "input.hlo.pb"
        output = temp_dirs["temp"] / "output.neff"

        # Setup valid input file
        atomic_write_bytes(input_file, sample_hlo)

        # Mock the file writing to raise OSError (disk full)
        original_open = open

        def mock_open(file, mode="r", **kwargs):
            if str(file).endswith("output.neff") and "wb" in mode:
                raise OSError("No space left on device")
            return original_open(file, mode, **kwargs)

        monkeypatch.setattr("builtins.open", mock_open)

        args = ["compile", str(input_file), "--output", str(output)]
        result = wrapper.compile_with_caching(args)

        assert result == 1  # Should fail due to disk space error


class TestMainFunction:
    """Test main() function and command-line interface."""

    def test_main_successful_compilation(
        self, temp_dirs, sample_hlo, sample_neff, mock_compiler_subprocess, monkeypatch
    ):
        """Test main function with successful compilation."""
        input_file = temp_dirs["temp"] / "test.hlo.pb"
        output = temp_dirs["temp"] / "test.neff"
        atomic_write_bytes(input_file, sample_hlo)

        test_args = [
            "neuronx_cc_wrapper.py",
            "compile",
            str(input_file),
            "--output",
            str(output),
            "--framework",
            "XLA",
            "--workdir",
            str(temp_dirs["temp"]),
        ]
        monkeypatch.setattr("sys.argv", test_args)
        result = main()

        assert result == 0
        assert output.exists()

    def test_main_compilation_failure(
        self, temp_dirs, sample_hlo, mock_compiler_subprocess, monkeypatch
    ):
        """Test main function with compilation failure."""
        input_file = temp_dirs["temp"] / "test.hlo.pb"
        output = temp_dirs["temp"] / "test.neff"
        atomic_write_bytes(input_file, sample_hlo)

        test_args = [
            "neuron_cc_wrapper.py",
            "compile",
            str(input_file),
            "--output",
            str(output),
            "--workdir",
            str(temp_dirs["temp"]),
        ]
        monkeypatch.setattr("sys.argv", test_args)

        # Configure mock to raise an exception
        mock_compiler_subprocess.return_value.get_or_compile.side_effect = NEFFCompilationError(
            "Failed"
        )
        result = main()

        assert result == 1

    def test_main_fatal_exception(self, temp_dirs, monkeypatch):
        """Test main function with fatal exception."""
        test_args = [
            "neuronx_cc_wrapper.py",
            "compile",
            "nonexistent.hlo.pb",
            "--output",
            "/tmp/test.neff",
            "--workdir",
            str(temp_dirs["temp"]),
        ]
        monkeypatch.setattr("sys.argv", test_args)
        result = main()

        assert result == 1

    def test_main_with_explicit_args_successful_compilation(
        self, temp_dirs, sample_hlo, sample_neff, mock_compiler_subprocess
    ):
        """Test main function with explicit arguments - successful compilation."""
        input_file = temp_dirs["temp"] / "test.hlo.pb"
        output = temp_dirs["temp"] / "test.neff"
        atomic_write_bytes(input_file, sample_hlo)

        test_args = [
            "compile",
            str(input_file),
            "--output",
            str(output),
            "--framework",
            "XLA",
            "--target",
            "trn2",
            "--workdir",
            str(temp_dirs["temp"]),
        ]

        result = main(test_args)

        assert result == 0
        assert output.exists()
        assert output.read_bytes() == sample_neff
