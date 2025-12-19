"""Handles subprocess operations for compilation pipeline."""

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

from torch.utils._filelock import FileLock

from .cache_utils import (
    atomic_write_bytes,
    get_local_cache_dir,
    get_lock_timeout,
    get_nfs_cache_dir,
    is_caching_disabled,
)
from .compiler_config import CompilerConfig


class NEFFCacheKey:
    """Efficient cache key for NEFF compilation.

    Uses lazy hash computation to avoid re-hashing on repeated access.
    The hash combines HLO content hash with compiler configuration to
    produce a deterministic cache key.
    """

    __slots__ = ("_config_hash", "_hash", "_hlo_hash")

    def __init__(
        self,
        hlo_protobuf: bytes,
        config: CompilerConfig,
        lnc_override: str | None,
        compiler_version: str | None,
    ):
        # Pre-compute HLO hash
        self._hlo_hash = hashlib.sha256(hlo_protobuf).hexdigest()

        # Create deterministic config representation for hashing
        config_parts = [
            config.target,
            config.model_type,
            config.optimization_level,
            config.auto_cast,
            config.framework,
            lnc_override or config.lnc,
            compiler_version or "",
            *sorted(config.extra_flags),
        ]
        config_str = "|".join(config_parts)
        self._config_hash = hashlib.sha256(config_str.encode()).hexdigest()

        # Combine hashes for final key
        self._hash = hashlib.sha256(f"{self._hlo_hash}:{self._config_hash}".encode()).hexdigest()

    @property
    def hash(self) -> str:
        """Return the pre-computed hash."""
        return self._hash

    def __hash__(self) -> int:
        """Return integer hash for use in dict/set."""
        return hash(self._hash)

    def __eq__(self, other: object) -> bool:
        """Compare cache keys by their hash."""
        if not isinstance(other, NEFFCacheKey):
            return NotImplemented
        return self._hash == other._hash


class CompilationError(Exception):
    """Base exception for compilation errors."""

    pass


class NEFFCompilationError(CompilationError):
    """Exception raised when NEFF compilation fails."""

    pass


class CompilerSubprocess:
    """Handles subprocess operations for compilation pipeline."""

    def __init__(self, nfs_cache_dir: str | None = None, local_cache_dir: str | None = None):
        # Use environment variables or defaults
        self.nfs_cache_dir = Path(nfs_cache_dir or get_nfs_cache_dir())
        self.local_cache_dir = Path(local_cache_dir or get_local_cache_dir())

        # Create directories if caching is enabled
        if not is_caching_disabled():
            self.nfs_cache_dir.mkdir(parents=True, exist_ok=True)
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)

    def _nfs_cache_path(self, cache_key: NEFFCacheKey) -> Path:
        """Get NFS cache file path."""
        key_hash = cache_key.hash
        return self.nfs_cache_dir / key_hash / f"{key_hash}.neff"

    def _generate_cache_key(
        self,
        hlo_protobuf: bytes,
        config: CompilerConfig,
        lnc_override: str | None,
        compiler_version: str,
    ) -> "NEFFCacheKey":
        """Generate deterministic cache key from all compilation inputs."""
        return NEFFCacheKey(
            hlo_protobuf=hlo_protobuf,
            config=config,
            lnc_override=lnc_override,
            compiler_version=compiler_version,
        )

    def get_or_compile(
        self,
        hlo_protobuf: bytes,
        config: CompilerConfig,
        lnc_override: str | None = None,
        ir_type: str = "XLA",
        workdir: str | None = None,
        input_file: str | None = None,
        output_file: str | None = None,
    ) -> bytes:
        """Two-tier cache lookup and compilation.

        Args:
            hlo_protobuf: HLO protobuf bytes or MLIR bytes
            config: Compiler configuration
            lnc_override: Optional override for logical neuron cores
            ir_type: Type of IR ("XLA" for HLO protobuf, "StableHLO" for MLIR)
            workdir: Compilation working directory (optional, for async mode)
            input_file: Path to input file (optional, for async mode)
            output_file: Path to write output NEFF file (optional, for async mode)

        Returns:
            Compiled NEFF bytes
        """
        # Skip caching if disabled
        if is_caching_disabled():
            return CompilerSubprocess.compile_hlo_protobuf_to_neff(
                hlo_protobuf,
                config,
                lnc_override,
                ir_type,
                workdir,
                input_file,
                output_file,
            )

        import neuronxcc

        cache_key = self._generate_cache_key(
            hlo_protobuf, config, lnc_override, neuronxcc.__version__
        )

        # 1. Check NFS cache first (fastest check)
        nfs_path = self._nfs_cache_path(cache_key)
        if nfs_path.exists():
            try:
                neff_bytes = nfs_path.read_bytes()
                # For async, write to output file on cache hit
                if output_file is not None:
                    Path(output_file).write_bytes(neff_bytes)
                return neff_bytes
            except OSError:
                # NFS read failed, continue to local compilation
                pass

        # 2. Use file lock for compilation
        key_hash = cache_key.hash
        lock_dir = self.local_cache_dir / key_hash / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)

        lock = FileLock(lock_dir / "compile.lock", timeout=get_lock_timeout())

        # This locking is similar to what exist in pytorch:
        # https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codecache.py#L2104
        # Going forward with PT2.10, we can import PersistentCache class from PyTorch and
        # reuse.
        with lock:
            # Double-check NFS cache after acquiring lock
            if nfs_path.exists():
                try:
                    neff_bytes = nfs_path.read_bytes()
                    # For async, write to output file on cache hit
                    if output_file:
                        Path(output_file).write_bytes(neff_bytes)
                    return neff_bytes
                except OSError:
                    pass

            # Compile and cache
            neff_bytes = CompilerSubprocess.compile_hlo_protobuf_to_neff(
                hlo_protobuf, config, lnc_override, ir_type, workdir, input_file, output_file
            )

            # Write to NFS cache atomically (this is basically writing to user provided disk)
            # At this line, we can plug any cache of our choice - could be NFS, FSX, S3
            atomic_write_bytes(nfs_path, neff_bytes)

        return neff_bytes

    @staticmethod
    def find_neuronx_cc() -> str:
        """Find neuronx-cc compiler in PATH.

        Returns:
            Path to neuronx-cc executable

        Raises:
            RuntimeError: If neuronx-cc is not found
        """
        neuronx_cc = shutil.which("neuronx-cc")
        if not neuronx_cc:
            raise RuntimeError("neuronx-cc not found in PATH")
        return neuronx_cc

    @staticmethod
    def run_neuronx_cc(
        config: CompilerConfig,
        input_file: str,
        output_file: str,
        lnc_override: str | None = None,
        workdir: str | None = None,
    ) -> None:
        """Compile HLO protobuf to NEFF.

        Args:
            config: Compiler configuration
            input_file: Path to input HLO/MLIR file
            output_file: Path to output NEFF file
            lnc_override: Optional override for logical neuron cores
            workdir: Compilation working directory

        Raises:
            NEFFCompilationError: If compilation fails
        """
        neuronx_cc = CompilerSubprocess.find_neuronx_cc()
        cmd = [
            neuronx_cc,
            *config.get_neuronx_cc_args(input_file, output_file, lnc_override),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir)
        if result.returncode != 0:
            raise NEFFCompilationError(
                f"NEFF compilation failed with return code {result.returncode}\n"
                f"Command: {' '.join(cmd)}\n"
                f"Error: {result.stderr}"
            )

    @staticmethod
    def compile_hlo_protobuf_to_neff(
        hlo_protobuf: bytes,
        config: CompilerConfig,
        lnc_override: str | None = None,
        ir_type: str = "XLA",
        workdir: str | None = None,
        input_file: str | None = None,
        output_file: str | None = None,
    ) -> bytes:
        """Compile HLO protobuf or MLIR directly to NEFF.

        Args:
            hlo_protobuf: HLO protobuf bytes or MLIR bytes
            config: Compiler configuration
            lnc_override: Optional override for logical neuron cores
            ir_type: Type of IR ("XLA" for HLO protobuf, "StableHLO" for MLIR)
            workdir: Compilation working directory
            input_file: Path to input file (for async mode when workdir is set)
            output_file: Path to output NEFF file (for async mode)

        Returns:
            Compiled NEFF bytes

        Raises:
            NEFFCompilationError: If NEFF compilation fails
        """
        # Async mode: workdir, input_file, output_file provided
        if workdir is not None:
            assert input_file is not None, "Input file must be provided"
            assert output_file is not None, "Output file must be provided"
            CompilerSubprocess.run_neuronx_cc(
                config, input_file, output_file, lnc_override, workdir
            )
            # Read and return compiled NEFF
            return Path(output_file).read_bytes()

        # Sync mode: Create temp dir for compilation
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Determine file format based on ir_type
            # "StableHLO" uses MLIR format, "XLA" uses HLO protobuf
            hlo_file = (
                tmpdir_path / "module.mlir"
                if ir_type == "StableHLO"
                else tmpdir_path / "module.hlo.pb"
            )
            hlo_file.write_bytes(hlo_protobuf)

            # Use temp output file
            temp_output = tmpdir_path / "module.neff"

            # Passing workdir is important when we run multi-worker
            # Seeing a race condition where the artifacts get mangled
            # causing runtime errors. Passing the workdir keeps all
            # artifacts contained in their respective folder.
            CompilerSubprocess.run_neuronx_cc(
                config, str(hlo_file), str(temp_output), lnc_override, str(tmpdir_path)
            )

            # Read and return compiled NEFF
            return Path(temp_output).read_bytes()
