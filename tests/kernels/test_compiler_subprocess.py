"""Comprehensive tests for CompilerSubprocess with caching, concurrency,
and race condition testing."""

import hashlib
import os
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from torch_neuronx.kernels.cache_utils import (
    atomic_write_bytes,
    get_local_cache_dir,
    get_lock_timeout,
    get_nfs_cache_dir,
    is_caching_disabled,
)
from torch_neuronx.kernels.compiler_subprocess import (
    CompilerSubprocess,
    NEFFCacheKey,
    NEFFCompilationError,
)


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean environment variables before each test and ensure caching is enabled."""
    env_vars = [
        "TORCH_NEURONX_NEFF_CACHE_DIR",
        "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR",
        "TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT",
        "TORCH_NEURONX_NEFF_DISABLE_CACHE",
    ]

    # Store original values
    original_values = {}
    for var in env_vars:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    # Ensure caching is enabled by default for tests

    yield

    # Restore original values
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as nfs_dir, tempfile.TemporaryDirectory() as local_dir:
        yield {"nfs": Path(nfs_dir), "local": Path(local_dir)}


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


class MockCompilerConfig:
    """Mock compiler configuration for testing."""

    def __init__(
        self,
        config_id="default",
        target="trn1",
        model_type="transformer",
        optimization_level="-O1",
        auto_cast="none",
        framework="XLA",
        lnc="1",
        extra_flags=None,
    ):
        self.config_id = config_id
        self.target = target
        self.model_type = model_type
        self.optimization_level = optimization_level
        self.auto_cast = auto_cast
        self.framework = framework
        self.lnc = lnc
        self.extra_flags = extra_flags or []

    def get_neuronx_cc_args(self, input_file, output_file, lnc_override=None):
        """Generate neuronx-cc command line arguments."""
        return [
            "compile",
            input_file,
            "--framework",
            self.framework,
            "--target",
            self.target,
            "--model-type",
            self.model_type,
            "--lnc",
            lnc_override or self.lnc,
            self.optimization_level,
            f"--auto-cast={self.auto_cast}",
            "--output",
            output_file,
            *self.extra_flags,
        ]

    def __str__(self):
        return f"MockConfig({self.config_id})"

    def __repr__(self):
        return f"MockConfig({self.config_id})"

    def __eq__(self, other):
        return isinstance(other, MockCompilerConfig) and self.config_id == other.config_id

    def __hash__(self):
        return hash(self.config_id)


def _write_task_for_multiprocess(args):
    """Helper function for multiprocess NFS write testing."""
    test_file, worker_id, sample_neff = args
    data = f"worker_{worker_id}_data".encode() + sample_neff
    try:
        atomic_write_bytes(test_file, data)
        return worker_id
    except Exception as e:
        return f"error_{worker_id}: {e}"


def _worker_process_for_nfs_test(args):
    """Worker process for NFS write testing."""
    test_file, worker_id, iterations = args
    results = []
    for i in range(iterations):
        try:
            data = f"worker_{worker_id}_iteration_{i}_{'x' * 100}".encode()
            atomic_write_bytes(test_file, data)
            results.append(f"success_{worker_id}_{i}")
            time.sleep(0.001)
        except Exception as e:
            results.append(f"error_{worker_id}_{i}: {e}")
    return results


def _node_process_for_multinode_test(args):
    """Node process for multi-node testing."""
    node_id, shared_nfs_dir, local_dir_base = args
    from pathlib import Path
    from unittest.mock import patch

    # Each node has its own local cache
    node_local_dir = Path(local_dir_base) / f"node_{node_id}"
    node_local_dir.mkdir(exist_ok=True)

    compiler = CompilerSubprocess(nfs_cache_dir=shared_nfs_dir, local_cache_dir=str(node_local_dir))

    results = []
    for _ in range(3):
        try:
            with (
                patch("neuronxcc.__version__", "v1.0"),
                patch(
                    "torch_neuronx.kernels.compiler_subprocess.CompilerSubprocess.compile_hlo_protobuf_to_neff"
                ) as mock,
            ):
                # mock.return_value = f"node_{node_id}_result_{i}".encode()
                mock.return_value = b"shared_cached_result"

                result = compiler.get_or_compile(
                    b"shared_hlo_data", MockCompilerConfig("shared_config"), "shared_lnc"
                )
                results.append(result.decode())
                time.sleep(0.01)

        except Exception as e:
            results.append(f"error: {e}")

    return f"node_{node_id}", results


class TestCacheUtils:
    """Test cache utility functions."""

    def test_environment_variables(self, monkeypatch):
        """Test environment variable reading."""
        monkeypatch.setenv("TORCH_NEURONX_NEFF_CACHE_DIR", "/custom/nfs")
        monkeypatch.setenv("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "/custom/local")
        monkeypatch.setenv("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT", "300")
        monkeypatch.setenv("TORCH_NEURONX_NEFF_DISABLE_CACHE", "true")

        assert get_nfs_cache_dir() == "/custom/nfs"
        assert get_local_cache_dir() == "/custom/local"
        assert get_lock_timeout() == 300
        assert is_caching_disabled() is True

    def test_default_values(self, monkeypatch):
        """Test default values when env vars not set."""
        monkeypatch.delenv("TORCH_NEURONX_NEFF_CACHE_DIR", raising=False)
        monkeypatch.delenv("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", raising=False)
        monkeypatch.delenv("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT", raising=False)
        monkeypatch.delenv("TORCH_NEURONX_NEFF_DISABLE_CACHE", raising=False)

        assert get_nfs_cache_dir() == "/tmp/neff_cache"
        assert get_local_cache_dir() == "/tmp/local_cache"
        assert get_lock_timeout() == 1200
        assert is_caching_disabled() is False


class TestNEFFCacheKey:
    """Test NEFF cache key functionality."""

    def test_cache_key_equality(self, mock_config, sample_hlo):
        """Test cache key equality and hashing."""
        key1 = NEFFCacheKey(sample_hlo, mock_config, "lnc1", "v1.0")
        key2 = NEFFCacheKey(sample_hlo, mock_config, "lnc1", "v1.0")
        key3 = NEFFCacheKey(sample_hlo, mock_config, "lnc2", "v1.0")

        assert key1 == key2
        assert key1 != key3
        assert hash(key1) == hash(key2)
        assert hash(key1) != hash(key3)

    def test_cache_key_hash_property(self, mock_config, sample_hlo):
        """Test cache key hash property returns consistent values."""
        key1 = NEFFCacheKey(sample_hlo, mock_config, "lnc1", "v1.0")
        key2 = NEFFCacheKey(sample_hlo, mock_config, "lnc1", "v1.0")
        key3 = NEFFCacheKey(sample_hlo, mock_config, "lnc2", "v1.0")

        # Hash property should return string
        assert isinstance(key1.hash, str)
        assert len(key1.hash) == 64  # SHA256 hex digest length

        # Same inputs should produce same hash
        assert key1.hash == key2.hash

        # Different inputs should produce different hash
        assert key1.hash != key3.hash

    def test_cache_key_deterministic(self, mock_config, sample_hlo):
        """Test that cache key hash is deterministic."""
        # Create multiple keys with same inputs
        keys = [NEFFCacheKey(sample_hlo, mock_config, "lnc1", "v1.0") for _ in range(10)]

        # All hashes should be identical
        hashes = [k.hash for k in keys]
        assert len(set(hashes)) == 1

    def test_cache_key_different_hlo_different_hash(self, mock_config):
        """Test that different HLO produces different hash."""
        key1 = NEFFCacheKey(b"hlo_data_1", mock_config, "lnc1", "v1.0")
        key2 = NEFFCacheKey(b"hlo_data_2", mock_config, "lnc1", "v1.0")

        assert key1.hash != key2.hash
        assert key1 != key2

    def test_cache_key_different_config_different_hash(self, sample_hlo):
        """Test that different config produces different hash."""
        config1 = MockCompilerConfig(config_id="config1", optimization_level="-O1")
        config2 = MockCompilerConfig(config_id="config2", optimization_level="-O2")

        key1 = NEFFCacheKey(sample_hlo, config1, "lnc1", "v1.0")
        key2 = NEFFCacheKey(sample_hlo, config2, "lnc1", "v1.0")

        assert key1.hash != key2.hash
        assert key1 != key2

    def test_cache_key_different_lnc_different_hash(self, mock_config, sample_hlo):
        """Test that different LNC override produces different hash."""
        key1 = NEFFCacheKey(sample_hlo, mock_config, "lnc1", "v1.0")
        key2 = NEFFCacheKey(sample_hlo, mock_config, "lnc2", "v1.0")

        assert key1.hash != key2.hash
        assert key1 != key2

    def test_cache_key_different_version_different_hash(self, mock_config, sample_hlo):
        """Test that different compiler version produces different hash."""
        key1 = NEFFCacheKey(sample_hlo, mock_config, "lnc1", "v1.0")
        key2 = NEFFCacheKey(sample_hlo, mock_config, "lnc1", "v2.0")

        assert key1.hash != key2.hash
        assert key1 != key2

    def test_cache_key_none_lnc_override(self, mock_config, sample_hlo):
        """Test cache key with None lnc_override uses config.lnc."""
        key1 = NEFFCacheKey(sample_hlo, mock_config, None, "v1.0")
        key2 = NEFFCacheKey(sample_hlo, mock_config, mock_config.lnc, "v1.0")

        # Should be equal since None falls back to config.lnc
        assert key1.hash == key2.hash
        assert key1 == key2

    def test_cache_key_extra_flags_sorted(self, sample_hlo):
        """Test that extra flags are sorted for deterministic hashing."""
        config1 = MockCompilerConfig(extra_flags=["--flag-a", "--flag-b"])
        config2 = MockCompilerConfig(extra_flags=["--flag-b", "--flag-a"])

        key1 = NEFFCacheKey(sample_hlo, config1, "lnc1", "v1.0")
        key2 = NEFFCacheKey(sample_hlo, config2, "lnc1", "v1.0")

        # Sorted flags should produce same hash
        assert key1.hash == key2.hash
        assert key1 == key2


class TestCompilerSubprocess:
    """Test CompilerSubprocess functionality."""

    def test_initialization(self, temp_dirs):
        """Test CompilerSubprocess initialization."""
        compiler = CompilerSubprocess(
            nfs_cache_dir=temp_dirs["nfs"], local_cache_dir=temp_dirs["local"]
        )

        assert compiler.nfs_cache_dir == Path(temp_dirs["nfs"])
        assert compiler.local_cache_dir == Path(temp_dirs["local"])
        assert compiler.nfs_cache_dir.exists()
        assert compiler.local_cache_dir.exists()

    def test_initialization_with_caching_disabled(self, temp_dirs, monkeypatch):
        """Test initialization when caching is disabled."""
        monkeypatch.setenv("TORCH_NEURONX_NEFF_DISABLE_CACHE", "true")

        compiler = CompilerSubprocess(
            nfs_cache_dir=temp_dirs["nfs"], local_cache_dir=temp_dirs["local"]
        )

        # Directories should still be set but may not be created
        assert compiler.nfs_cache_dir == Path(temp_dirs["nfs"])
        assert compiler.local_cache_dir == Path(temp_dirs["local"])

    def test_cache_key_generation(self, temp_dirs, mock_config, sample_hlo):
        """Test cache key generation."""
        compiler = CompilerSubprocess(
            nfs_cache_dir=temp_dirs["nfs"], local_cache_dir=temp_dirs["local"]
        )

        key = compiler._generate_cache_key(sample_hlo, mock_config, "lnc1", "v1.0")

        assert isinstance(key, NEFFCacheKey)
        # New NEFFCacheKey uses .hash property, not individual attributes
        assert isinstance(key.hash, str)
        assert len(key.hash) == 64  # SHA256 hex digest

    def test_nfs_cache_path(self, temp_dirs, mock_config, sample_hlo):
        """Test NFS cache path generation uses pre-computed hash."""
        compiler = CompilerSubprocess(
            nfs_cache_dir=temp_dirs["nfs"], local_cache_dir=temp_dirs["local"]
        )

        key = NEFFCacheKey(sample_hlo, mock_config, "lnc1", "v1.0")
        path = compiler._nfs_cache_path(key)

        # Path should use the key's .hash property
        expected_path = Path(temp_dirs["nfs"]) / key.hash / f"{key.hash}.neff"

        assert path == expected_path

    @patch(
        "torch_neuronx.kernels.compiler_subprocess.CompilerSubprocess.compile_hlo_protobuf_to_neff"
    )
    def test_caching_disabled_bypasses_cache(
        self, mock_compile, temp_dirs, mock_config, sample_hlo, sample_neff, monkeypatch
    ):
        """Test that caching disabled bypasses all cache logic."""
        monkeypatch.setenv("TORCH_NEURONX_NEFF_DISABLE_CACHE", "true")
        mock_compile.return_value = sample_neff

        compiler = CompilerSubprocess(
            nfs_cache_dir=temp_dirs["nfs"], local_cache_dir=temp_dirs["local"]
        )

        result = compiler.get_or_compile(sample_hlo, mock_config, "lnc1", "XLA", temp_dirs["local"])

        assert result == sample_neff
        mock_compile.assert_called_once_with(
            sample_hlo, mock_config, "lnc1", "XLA", temp_dirs["local"], None, None
        )

    @patch(
        "torch_neuronx.kernels.compiler_subprocess.CompilerSubprocess.compile_hlo_protobuf_to_neff"
    )
    def test_nfs_cache_hit(self, mock_compile, temp_dirs, mock_config, sample_hlo, sample_neff):
        """Test NFS cache hit scenario."""
        compiler = CompilerSubprocess(
            nfs_cache_dir=temp_dirs["nfs"], local_cache_dir=temp_dirs["local"]
        )

        # Pre-populate NFS cache
        key = compiler._generate_cache_key(sample_hlo, mock_config, "lnc1", "v1.0")
        nfs_path = compiler._nfs_cache_path(key)
        nfs_path.parent.mkdir(parents=True, exist_ok=True)
        nfs_path.write_bytes(sample_neff)

        with patch("neuronxcc.__version__", "v1.0"):
            result = compiler.get_or_compile(sample_hlo, mock_config, "lnc1")

        assert result == sample_neff
        mock_compile.assert_not_called()

    @patch(
        "torch_neuronx.kernels.compiler_subprocess.CompilerSubprocess.compile_hlo_protobuf_to_neff"
    )
    def test_nfs_cache_hit_with_output(
        self, mock_compile, temp_dirs, mock_config, sample_hlo, sample_neff
    ):
        """Test cache hit writes to output file (async mode)."""
        compiler = CompilerSubprocess(
            nfs_cache_dir=temp_dirs["nfs"], local_cache_dir=temp_dirs["local"]
        )

        # Pre-populate NFS cache to force cache hit
        with patch("neuronxcc.__version__", "v1.0"):
            key = compiler._generate_cache_key(sample_hlo, mock_config, "lnc1", "v1.0")
        nfs_path = compiler._nfs_cache_path(key)
        nfs_path.parent.mkdir(parents=True, exist_ok=True)
        nfs_path.write_bytes(sample_neff)

        # Setup async mode directories/files
        workdir = temp_dirs["local"] / "workdir"
        workdir.mkdir(parents=True, exist_ok=True)
        input_file = workdir / "input.hlo.pb"
        output_file = workdir / "output.neff"
        input_file.write_bytes(sample_hlo)

        with patch("neuronxcc.__version__", "v1.0"):
            # Call with all async mode parameters
            result = compiler.get_or_compile(
                sample_hlo,
                mock_config,
                "lnc1",
                workdir=str(workdir),
                input_file=str(input_file),
                output_file=str(output_file),
            )

        # Should return cached NEFF
        assert result == sample_neff

        # Output file should be written with cached NEFF
        assert output_file.exists()
        assert output_file.read_bytes() == sample_neff

        # No compilation because of cache hit
        mock_compile.assert_not_called()

    @patch(
        "torch_neuronx.kernels.compiler_subprocess.CompilerSubprocess.compile_hlo_protobuf_to_neff"
    )
    def test_nfs_cache_miss_compilation(
        self, mock_compile, temp_dirs, mock_config, sample_hlo, sample_neff
    ):
        """Test NFS cache miss triggers compilation."""
        mock_compile.return_value = sample_neff

        compiler = CompilerSubprocess(
            nfs_cache_dir=temp_dirs["nfs"], local_cache_dir=temp_dirs["local"]
        )

        with patch("neuronxcc.__version__", "v1.0"):
            result = compiler.get_or_compile(
                sample_hlo, mock_config, "lnc1", "XLA", temp_dirs["local"]
            )

        assert result == sample_neff
        mock_compile.assert_called_once_with(
            sample_hlo, mock_config, "lnc1", "XLA", temp_dirs["local"], None, None
        )

        # Verify NFS cache was populated
        key = compiler._generate_cache_key(sample_hlo, mock_config, "lnc1", "v1.0")
        nfs_path = compiler._nfs_cache_path(key)
        assert nfs_path.exists()
        assert nfs_path.read_bytes() == sample_neff

    @patch(
        "torch_neuronx.kernels.compiler_subprocess.CompilerSubprocess.compile_hlo_protobuf_to_neff"
    )
    def test_cache_miss_with_output_file_triggers_compilation(
        self, mock_compile, temp_dirs, mock_config, sample_hlo, sample_neff
    ):
        """Test cache miss with output_file triggers compilation."""
        mock_compile.return_value = sample_neff

        compiler = CompilerSubprocess(
            nfs_cache_dir=temp_dirs["nfs"], local_cache_dir=temp_dirs["local"]
        )

        # Setup async mode directories/files
        workdir = temp_dirs["local"] / "workdir"
        workdir.mkdir(parents=True, exist_ok=True)
        input_file = workdir / "input.hlo.pb"
        output_file = workdir / "output.neff"
        input_file.write_bytes(sample_hlo)

        with patch("neuronxcc.__version__", "v1.0"):
            # Call with all async mode parameters
            result = compiler.get_or_compile(
                sample_hlo,
                mock_config,
                "lnc1",
                workdir=str(workdir),
                input_file=str(input_file),
                output_file=str(output_file),
            )

        assert result == sample_neff

        # Compilation should have been called
        mock_compile.assert_called_once()

        # NFS cache should be populated
        with patch("neuronxcc.__version__", "v1.0"):
            key = compiler._generate_cache_key(sample_hlo, mock_config, "lnc1", "v1.0")
        nfs_path = compiler._nfs_cache_path(key)
        assert nfs_path.exists()
        assert nfs_path.read_bytes() == sample_neff


class TestConcurrency:
    """Test concurrent access and race conditions."""

    @patch(
        "torch_neuronx.kernels.compiler_subprocess.CompilerSubprocess.compile_hlo_protobuf_to_neff"
    )
    def test_concurrent_compilation_same_key(
        self, mock_compile, temp_dirs, mock_config, sample_hlo, sample_neff
    ):
        """Test concurrent compilation with same cache key."""

        # Simulate slow compilation
        def slow_compile(*args, **kwargs):
            time.sleep(0.1)
            return sample_neff

        mock_compile.side_effect = slow_compile

        compiler = CompilerSubprocess(
            nfs_cache_dir=temp_dirs["nfs"], local_cache_dir=temp_dirs["local"]
        )

        def compile_task():
            with patch("neuronxcc.__version__", "v1.0"):
                return compiler.get_or_compile(sample_hlo, mock_config, "lnc1")

        # Run multiple threads concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(compile_task) for _ in range(5)]
            results = [f.result() for f in futures]

        # All should return the same result
        assert all(r == sample_neff for r in results)

        # Compilation should only happen once due to locking
        assert mock_compile.call_count == 1

    @patch(
        "torch_neuronx.kernels.compiler_subprocess.CompilerSubprocess.compile_hlo_protobuf_to_neff"
    )
    def test_concurrent_compilation_different_keys(
        self, mock_compile, temp_dirs, mock_config, sample_hlo, sample_neff
    ):
        """Test concurrent compilation with different cache keys."""
        mock_compile.return_value = sample_neff

        compiler = CompilerSubprocess(
            nfs_cache_dir=temp_dirs["nfs"], local_cache_dir=temp_dirs["local"]
        )

        def compile_task(lnc_override):
            with patch("neuronxcc.__version__", "v1.0"):
                return compiler.get_or_compile(sample_hlo, mock_config, lnc_override)

        # Run with different LNC overrides
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(compile_task, f"lnc{i}") for i in range(3)]
            results = [f.result() for f in futures]

        # All should return the same result
        assert all(r == sample_neff for r in results)

        # Should compile 3 times (different keys)
        assert mock_compile.call_count == 3

    def test_multiprocess_nfs_write_simulation(self, temp_dirs, sample_neff):
        """Test atomic NFS writes with multiprocessing simulation."""

        nfs_dir = Path(temp_dirs["nfs"])
        test_file = nfs_dir / "test_atomic.neff"

        args_list = [(test_file, i, sample_neff) for i in range(4)]

        # Simulate multiple workers writing concurrently
        with ProcessPoolExecutor(max_workers=4) as executor:
            _ = list(executor.map(_write_task_for_multiprocess, args_list))

        # File should exist and contain data from one of the workers
        assert test_file.exists()
        content = test_file.read_bytes()

        # Verify content is from one complete write (not corrupted)
        assert content.endswith(sample_neff)
        assert any(f"worker_{i}_data".encode() in content for i in range(4))

    @patch(
        "torch_neuronx.kernels.compiler_subprocess.CompilerSubprocess.compile_hlo_protobuf_to_neff"
    )
    def test_lock_timeout_handling(
        self, mock_compile, temp_dirs, mock_config, sample_hlo, sample_neff, monkeypatch
    ):
        """Test lock timeout handling."""
        monkeypatch.setenv("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT", "1")  # 1 second timeout

        # Simulate very slow compilation
        def very_slow_compile(*args, **kwargs):
            time.sleep(2)  # Longer than timeout
            return sample_neff

        mock_compile.side_effect = very_slow_compile

        compiler = CompilerSubprocess(
            nfs_cache_dir=temp_dirs["nfs"], local_cache_dir=temp_dirs["local"]
        )

        def compile_task():
            with patch("neuronxcc.__version__", "v1.0"):
                return compiler.get_or_compile(sample_hlo, mock_config, "lnc1")

        # First task will acquire lock and start slow compilation
        # Second task should timeout waiting for lock
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(compile_task)
            time.sleep(0.1)  # Let first task acquire lock
            future2 = executor.submit(compile_task)

            # First should succeed, second should timeout
            result1 = future1.result()

            from filelock import Timeout

            with pytest.raises(Timeout):  # FileLock timeout exception
                future2.result()

        assert result1 == sample_neff


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_neuronx_cc_not_found(self):
        """Test error when neuronx-cc not found."""
        with (
            patch("shutil.which", return_value=None),
            pytest.raises(RuntimeError, match="neuronx-cc not found in PATH"),
        ):
            CompilerSubprocess.find_neuronx_cc()

    @patch("subprocess.run")
    def test_compilation_failure(self, mock_run, temp_dirs, mock_config):
        """Test compilation failure handling."""
        mock_run.return_value = Mock(returncode=1, stderr="Compilation failed")

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(NEFFCompilationError, match="NEFF compilation failed"),
        ):
            input_file = str(Path(tmpdir) / "input.hlo.pb")
            output_file = str(Path(tmpdir) / "output.neff")
            CompilerSubprocess.run_neuronx_cc(mock_config, input_file, output_file)


class TestIntegration:
    """Integration tests with real compilation simulation."""

    @patch("subprocess.run")
    @patch("shutil.which")
    def test_end_to_end_compilation_sync(
        self, mock_which, mock_run, temp_dirs, mock_config, sample_hlo, sample_neff
    ):
        """Test end-to-end compilation flow (sync mode)."""
        mock_which.return_value = "/usr/bin/neuronx-cc"
        mock_run.return_value = Mock(returncode=0, stderr="")

        # Mock the file operations in compile_hlo_protobuf_to_neff
        with (
            patch("tempfile.TemporaryDirectory") as mock_tmpdir,
            patch("pathlib.Path.write_bytes"),
            patch("pathlib.Path.read_bytes", return_value=sample_neff),
        ):
            mock_tmpdir.return_value.__enter__.return_value = "/tmp/test"
            result = CompilerSubprocess.compile_hlo_protobuf_to_neff(
                sample_hlo, mock_config, "lnc1"
            )

        assert result == sample_neff
        mock_run.assert_called_once()

    @patch("subprocess.run")
    @patch("shutil.which")
    def test_end_to_end_compilation_async(
        self, mock_which, mock_run, temp_dirs, mock_config, sample_hlo, sample_neff
    ):
        """Test end-to-end compilation flow (async mode)."""
        # Setup async mode directories/files
        workdir = temp_dirs["local"] / "workdir"
        workdir.mkdir(parents=True, exist_ok=True)
        input_file = workdir / "input.hlo.pb"
        output_file = workdir / "output.neff"
        input_file.write_bytes(sample_hlo)

        # Setup mocks
        def mock_side_effect(*args, **kwargs):
            output_file.write_bytes(sample_neff)
            return Mock(returncode=0, stderr="")

        mock_which.return_value = "/usr/bin/neuronx-cc"
        mock_run.side_effect = mock_side_effect

        result = CompilerSubprocess.compile_hlo_protobuf_to_neff(
            sample_hlo, mock_config, "lnc1", "XLA", workdir, input_file, output_file
        )

        assert output_file.exists()
        assert output_file.read_bytes() == sample_neff
        assert result == sample_neff
        mock_run.assert_called_once()


class TestRaceConditions:
    """Test race conditions in caching system."""

    def test_simultaneous_nfs_writes_different_processes(self, temp_dirs):
        """Simulate multiple nodes writing to same NFS file simultaneously."""
        nfs_dir = temp_dirs["nfs"]
        test_file = nfs_dir / "race_test.neff"

        # Prepare arguments for multiprocessing
        args_list = [(test_file, i, 5) for i in range(8)]

        # Run multiple processes simultaneously
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(_worker_process_for_nfs_test, args_list))
            all_results = []
            for result in results:
                all_results.extend(result)

        # Verify file exists and has valid content
        assert test_file.exists()
        content = test_file.read_bytes()

        # Content should be from exactly one complete write
        content_str = content.decode()
        assert content_str.startswith("worker_")
        assert content_str.count("worker_") == 1  # Only one worker's data
        assert len(content_str) > 100  # Has the padding

        # Most operations should succeed
        success_count = sum(1 for r in all_results if r.startswith("success"))
        assert success_count > 0

    def test_cache_key_collision_handling(self, temp_dirs):
        """Test handling of cache key collisions with concurrent access."""
        compiler = CompilerSubprocess(
            nfs_cache_dir=str(temp_dirs["nfs"]), local_cache_dir=str(temp_dirs["local"])
        )

        # Create scenarios that might generate same cache key
        test_data = [
            (b"hlo_data_1", MockCompilerConfig("config1"), "lnc1"),
            (b"hlo_data_1", MockCompilerConfig("config1"), "lnc1"),  # Exact duplicate
            (b"hlo_data_2", MockCompilerConfig("config1"), "lnc1"),  # Different HLO
            (
                b"hlo_data_1",
                MockCompilerConfig("config2", optimization_level="-O2"),
                "lnc1",
            ),  # Different config (id not relevant)
        ]

        def compile_task(hlo_data, config, lnc_override):
            return compiler.get_or_compile(hlo_data, config, lnc_override)

        # Apply patches at test level, not inside threads
        with (
            patch("neuronxcc.__version__", "v1.0"),
            patch(
                "torch_neuronx.kernels.compiler_subprocess.CompilerSubprocess.compile_hlo_protobuf_to_neff"
            ) as mock_compile,
        ):

            def mock_side_effect(*args, **kwargs):
                # Return different results based on input
                hlo_data = args[0]
                config = args[1]
                lnc_override = args[2] if len(args) > 2 else None
                return f"compiled_{hash((hlo_data, str(config), lnc_override))}".encode()

            mock_compile.side_effect = mock_side_effect

            # Run all scenarios concurrently
            with ThreadPoolExecutor(max_workers=len(test_data)) as executor:
                futures = [executor.submit(compile_task, *data) for data in test_data]
                results = [f.result() for f in futures]

        # First two should be identical (same cache key)
        assert results[0] == results[1]
        # Others should be different
        assert len(set(results)) == 3  # 3 unique results from 4 tasks

    def test_lock_contention_stress(self, temp_dirs, monkeypatch):
        """Stress test lock contention with many concurrent workers."""
        monkeypatch.setenv("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT", "5")  # 5 second timeout

        compiler = CompilerSubprocess(
            nfs_cache_dir=str(temp_dirs["nfs"]), local_cache_dir=str(temp_dirs["local"])
        )

        compilation_count = 0
        compilation_lock = threading.Lock()

        def slow_compile(*args, **kwargs):
            nonlocal compilation_count
            with compilation_lock:
                compilation_count += 1
            time.sleep(0.1)  # Simulate compilation time
            return b"compiled_neff_data"

        def compile_task(task_id):
            try:
                with (
                    patch("neuronxcc.__version__", "v1.0"),
                    patch(
                        "torch_neuronx.kernels.compiler_subprocess.CompilerSubprocess.compile_hlo_protobuf_to_neff",
                        side_effect=slow_compile,
                    ),
                ):
                    _ = compiler.get_or_compile(
                        b"same_hlo_data", MockCompilerConfig("same_config"), "same_lnc"
                    )
                return f"success_{task_id}"
            except Exception as e:
                return f"error_{task_id}: {e}"

        # Run many concurrent tasks with same cache key
        num_workers = 20
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(compile_task, i) for i in range(num_workers)]
            results = [f.result() for f in futures]

        # everyone should succeed
        success_count = sum(1 for r in results if r.startswith("success"))
        assert success_count == num_workers

        # Only one compilation should have occurred due to locking
        assert compilation_count == 1

    def test_process_cancellation_during_write(self, temp_dirs, sample_neff):
        """Test atomic writes prevent corruption when process is cancelled."""
        nfs_dir = temp_dirs["nfs"]
        test_file = nfs_dir / "cancellation_test.neff"

        compiler = CompilerSubprocess(
            nfs_cache_dir=str(temp_dirs["nfs"]), local_cache_dir=str(temp_dirs["local"])
        )

        def cancelled_writer():
            """Simulate process that gets cancelled during write."""
            try:
                # Start writing large data
                with open(test_file, "wb") as f:
                    f.write(b"PARTIAL_DATA_")
                    f.flush()
                    # Simulate process cancellation (SIGTERM/SIGKILL)
                    raise KeyboardInterrupt("Process cancelled")
            except KeyboardInterrupt:
                # Process was killed, file left in partial state
                pass

        cancelled_writer()

        sample_hlo_data = b"mock_hlo_protobuf_data"
        mock_cfg = MockCompilerConfig()

        with (
            patch(
                "torch_neuronx.kernels.compiler_subprocess.CompilerSubprocess.compile_hlo_protobuf_to_neff",
                return_value=sample_neff,
            ),
            patch("neuronxcc.__version__", "v1.0"),
        ):
            result = compiler.get_or_compile(sample_hlo_data, mock_cfg, "lnc1")

        # File should contain complete atomic data, not partial data
        assert result == sample_neff


class TestDeadlockPrevention:
    """Test deadlock prevention mechanisms."""

    def test_no_deadlock_with_nested_locks(self, temp_dirs):
        """Test that nested lock scenarios don't cause deadlocks."""
        compiler1 = CompilerSubprocess(
            nfs_cache_dir=str(temp_dirs["nfs"]), local_cache_dir=str(temp_dirs["local"])
        )

        compiler2 = CompilerSubprocess(
            nfs_cache_dir=str(temp_dirs["nfs"]), local_cache_dir=str(temp_dirs["local"])
        )

        def task1():
            return compiler1.get_or_compile(b"hlo1", MockCompilerConfig("config1"), "lnc1")

        def task2():
            return compiler2.get_or_compile(b"hlo2", MockCompilerConfig("config2"), "lnc2")

        # Run tasks that might compete for resources
        with (
            patch("neuronxcc.__version__", "v1.0"),
            patch(
                "torch_neuronx.kernels.compiler_subprocess.CompilerSubprocess.compile_hlo_protobuf_to_neff"
            ) as mock,
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            mock.return_value = b"result"
            future1 = executor.submit(task1)
            future2 = executor.submit(task2)

            # Both should complete without deadlock
            _ = future1.result(timeout=10)  # 10 second timeout
            _ = future2.result(timeout=10)

        nfs_dir = Path(temp_dirs["nfs"])
        cache_files = list(nfs_dir.rglob("*.neff"))
        assert len(cache_files) == 2  # Two unique cache files created

    def test_lock_timeout_prevents_deadlock(self, temp_dirs, monkeypatch):
        """Test that lock timeouts prevent indefinite blocking."""
        monkeypatch.setenv("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT", "2")  # 2 second timeout

        compiler = CompilerSubprocess(
            nfs_cache_dir=str(temp_dirs["nfs"]), local_cache_dir=str(temp_dirs["local"])
        )

        # Create a scenario where one task holds lock for too long
        def long_running_task():
            def very_slow_compile(*args, **kwargs):
                time.sleep(5)  # Longer than timeout
                return b"slow_result"

            with (
                patch("neuronxcc.__version__", "v1.0"),
                patch(
                    "torch_neuronx.kernels.compiler_subprocess.CompilerSubprocess.compile_hlo_protobuf_to_neff",
                    side_effect=very_slow_compile,
                ),
            ):
                return compiler.get_or_compile(b"hlo", MockCompilerConfig("config"), "lnc")

        def quick_task():
            with (
                patch("neuronxcc.__version__", "v1.0"),
                patch(
                    "torch_neuronx.kernels.compiler_subprocess.CompilerSubprocess.compile_hlo_protobuf_to_neff"
                ) as mock,
            ):
                mock.return_value = b"quick_result"
                return compiler.get_or_compile(b"hlo", MockCompilerConfig("config"), "lnc")

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Start long task first
            long_future = executor.submit(long_running_task)
            time.sleep(0.1)  # Let it acquire the lock

            # Start quick task that should timeout
            quick_future = executor.submit(quick_task)

            # Long task should eventually complete
            long_result = long_future.result(timeout=10)
            assert long_result == b"slow_result"

            # Quick task should timeout and raise exception
            from filelock import Timeout

            with pytest.raises(Timeout):  # FileLock timeout
                quick_future.result(timeout=5)


class TestMultiNodeSimulation:
    """Simulate multi-node scenarios with separate processes."""

    def test_multi_node_cache_sharing(self, temp_dirs):
        """Simulate multiple nodes sharing NFS cache."""
        nfs_dir = str(temp_dirs["nfs"])
        local_base = str(temp_dirs["local"])

        # Prepare arguments for multiprocessing
        args_list = [(i, nfs_dir, local_base) for i in range(4)]

        # Simulate 4 nodes
        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(_node_process_for_multinode_test, args_list))

            node_results = {}
            for node_id, node_result in results:
                node_results[node_id] = node_result

        # All nodes should get the same cached result (first one to compile wins)
        all_results = []
        for results in node_results.values():
            all_results.extend(results)

        # Should have some successful results
        successful_results = [r for r in all_results if not r.startswith("error")]
        assert len(successful_results) > 0

        # All successful results should be the same (from shared cache)
        unique_results = set(successful_results)
        assert len(unique_results) == 1  # Only one unique result due to caching

    def test_node_failure_recovery(self, temp_dirs):
        """Test recovery when a node fails during compilation."""
        nfs_dir = str(temp_dirs["nfs"])
        local_dir = str(temp_dirs["local"])

        def failing_node():
            """Node that fails during compilation."""
            compiler = CompilerSubprocess(nfs_cache_dir=nfs_dir, local_cache_dir=local_dir)

            def failing_compile(*args, **kwargs):
                time.sleep(0.1)  # Start compilation
                raise Exception("Node failure during compilation")

            with (
                patch("neuronxcc.__version__", "v1.0"),
                patch(
                    "torch_neuronx.kernels.compiler_subprocess.CompilerSubprocess.compile_hlo_protobuf_to_neff",
                    side_effect=failing_compile,
                ),
            ):
                try:
                    return compiler.get_or_compile(b"hlo", MockCompilerConfig("config"), "lnc")
                except Exception as e:
                    return f"failed: {e}"

        def recovery_node():
            """Node that should recover and complete compilation."""
            time.sleep(0.2)  # Wait for failing node to start and fail

            compiler = CompilerSubprocess(nfs_cache_dir=nfs_dir, local_cache_dir=local_dir)

            with (
                patch("neuronxcc.__version__", "v1.0"),
                patch(
                    "torch_neuronx.kernels.compiler_subprocess.CompilerSubprocess.compile_hlo_protobuf_to_neff"
                ) as mock,
            ):
                mock.return_value = b"recovery_success"
                return compiler.get_or_compile(b"hlo", MockCompilerConfig("config"), "lnc")

        with ThreadPoolExecutor(max_workers=2) as executor:
            failing_future = executor.submit(failing_node)
            recovery_future = executor.submit(recovery_node)

            failing_result = failing_future.result()
            recovery_result = recovery_future.result()

        # Failing node should report failure
        assert "failed:" in str(failing_result)

        # Recovery node should succeed
        assert recovery_result == b"recovery_success"
