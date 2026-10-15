"""Handles subprocess operations for compilation pipeline."""

import hashlib
import shutil
import subprocess
from pathlib import Path

from torch.utils._filelock import FileLock

from .cache_utils import (
    atomic_write_bytes,
    compilation_job,
    get_local_cache_dir,
    get_lock_timeout,
    get_nfs_cache_dir,
    is_caching_disabled,
)
from .compiler_config import CompilerConfig


class NEFFCacheKey:
    """Create a deterministic key for a NEFF compilation.

    Args:
        hlo_protobuf (bytes): HLO protobuf or MLIR bytes to hash.
        config (CompilerConfig): Compiler configuration to hash.
        lnc_override (str | None): Logical NeuronCore override to hash; a falsey value
            uses ``config.lnc``.
        compiler_version (str | None): Compiler version to hash; ``None`` hashes
            an empty version string.

    Note:
        Precompute separate HLO and configuration hashes, then combine them so
        repeated key access does not rehash the inputs.
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
        """Return the precomputed hexadecimal cache-key hash.

        Returns:
            str: SHA-256 cache-key digest.
        """
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
    """Represent a compilation pipeline failure."""

    pass


class NEFFCompilationError(CompilationError):
    """Represent a failed NEFF compilation."""

    pass


class CompilerSubprocess:
    """Handle cached NEFF compilation through a subprocess.

    Args:
        nfs_cache_dir (str, optional): Shared cache directory override. A falsey value
            uses ``get_nfs_cache_dir()``. Defaults to ``None``.
        local_cache_dir (str, optional): Local lock directory override. A falsey value
            uses ``get_local_cache_dir()``. Defaults to ``None``.

    Note:
        Create both cache directories during initialization unless caching is
        disabled.
    """

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
        ir_type: str = "StableHLO",
        output_file: str | None = None,
        neff_cache_key: str | None = None,
    ) -> bytes:
        """Return a cached NEFF or compile and cache MLIR bytes.

        Args:
            hlo_protobuf (bytes): MLIR bytes to compile; the parameter name is retained for
                compatibility.
            config (CompilerConfig): Compiler configuration.
            lnc_override (str, optional): Logical NeuronCore override. A falsey value uses
                ``config.lnc``. Defaults to ``None``.
            ir_type (str, optional): Accepted for compatibility but ignored. Defaults to
                ``"StableHLO"``.
            output_file (str, optional): Additional NEFF destination. Defaults to ``None``.
            neff_cache_key (str, optional): External cache-key text. Defaults to ``None``.

        Returns:
            bytes: Cached or newly compiled NEFF bytes.

        Raises:
            NEFFCompilationError: If NEFF compilation fails.
            RuntimeError: If ``neuronx-cc`` is not found.
            filelock.Timeout: If the cache lock cannot be acquired before its timeout.
        """
        # Skip caching if disabled
        if is_caching_disabled():
            neff_bytes = CompilerSubprocess.compile_hlo_protobuf_to_neff(
                hlo_protobuf,
                config,
                lnc_override,
                ir_type,
            )
            if output_file is not None:
                Path(output_file).write_bytes(neff_bytes)
            return neff_bytes

        if neff_cache_key is not None:
            key_hash = hashlib.sha256(neff_cache_key.encode()).hexdigest()
        else:
            import neuronxcc

            cache_key = self._generate_cache_key(
                hlo_protobuf, config, lnc_override, neuronxcc.__version__
            )
            key_hash = cache_key.hash

        # 1. Check NFS cache first (fastest check)
        nfs_path = self.nfs_cache_dir / key_hash / f"{key_hash}.neff"
        if nfs_path.exists():
            try:
                neff_bytes = nfs_path.read_bytes()
                if output_file is not None:
                    Path(output_file).write_bytes(neff_bytes)
                return neff_bytes
            except OSError:
                pass

        # 2. Use file lock for compilation
        lock_dir = self.local_cache_dir / key_hash / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)

        lock = FileLock(lock_dir / "compile.lock", timeout=get_lock_timeout())

        # This locking is similar to what exist in pytorch:
        # https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codecache.py#L2104
        # Going forward with PT2.10, we can import PersistentCache class from PyTorch and
        # reuse.
        with lock:
            # Double-check NFS cache after acquiring lock
            if not nfs_path.exists():
                # Compile and cache
                neff_bytes = CompilerSubprocess.compile_hlo_protobuf_to_neff(
                    hlo_protobuf, config, lnc_override, ir_type
                )

                if output_file is not None:
                    Path(output_file).write_bytes(neff_bytes)

                # Write to NFS cache atomically (this is basically writing to user provided disk)
                # At this line, we can plug any cache of our choice - could be NFS, FSX, S3
                atomic_write_bytes(nfs_path, neff_bytes)

                return neff_bytes

        # Read outside lock — all waiting processes can read in parallel
        neff_bytes = nfs_path.read_bytes()
        if output_file is not None:
            Path(output_file).write_bytes(neff_bytes)
        return neff_bytes

    @staticmethod
    def find_neuronx_cc() -> str:
        """Find the ``neuronx-cc`` executable on ``PATH``.

        Returns:
            str: Path to the compiler executable.

        Raises:
            RuntimeError: If ``neuronx-cc`` is not found.
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
        """Run ``neuronx-cc`` for an input IR file.

        Args:
            config (CompilerConfig): Compiler configuration.
            input_file (str): Input HLO or MLIR filename.
            output_file (str): Output NEFF filename.
            lnc_override (str, optional): Logical NeuronCore override. A falsey value
                uses ``config.lnc``. Defaults to ``None``.
            workdir (str, optional): Compiler working directory. Defaults to ``None``.

        Raises:
            NEFFCompilationError: If the compiler exits with a nonzero status.
            RuntimeError: If ``neuronx-cc`` is not found.
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
        ir_type: str = "StableHLO",
    ) -> bytes:
        """Compile MLIR bytes into a NEFF.

        Args:
            hlo_protobuf (bytes): MLIR bytes to compile; the parameter name is retained for
                compatibility.
            config (CompilerConfig): Compiler configuration.
            lnc_override (str, optional): Logical NeuronCore override. A falsey value uses
                ``config.lnc``. Defaults to ``None``.
            ir_type (str, optional): Accepted for compatibility but ignored. Defaults to
                ``"StableHLO"``.

        Returns:
            bytes: Compiled NEFF bytes.

        Raises:
            NEFFCompilationError: If NEFF compilation fails.
            RuntimeError: If ``neuronx-cc`` is not found.
        """
        with compilation_job("eager") as job:
            workdir = job.get_neuronx_cc_workdir()
            workdir.mkdir(parents=True, exist_ok=True)

            # Determine file format based on ir_type
            # "StableHLO" uses MLIR format
            input_file_name = "module.mlir"
            output_file_name = "module.neff"
            hlo_file = workdir / input_file_name
            hlo_file.write_bytes(hlo_protobuf)
            output_file = workdir / output_file_name

            CompilerSubprocess.run_neuronx_cc(
                config, str(input_file_name), str(output_file_name), lnc_override, str(workdir)
            )

            return output_file.read_bytes()
