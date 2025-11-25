"""Handles subprocess operations for compilation pipeline."""

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

from torch.utils._filelock import FileLock

from .cache_utils import (
    atomic_write_bytes,
    get_local_cache_dir,
    get_lock_timeout,
    get_nfs_cache_dir,
    is_caching_disabled,
)
from .compiler_config import CompilerConfig


class NEFFCacheKey(NamedTuple):
    hlo_protobuf: bytes
    config: CompilerConfig
    lnc_override: str
    compiler_version: str | None


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
        key_hash = hashlib.sha256(str(cache_key).encode()).hexdigest()
        return self.nfs_cache_dir / key_hash / f"{key_hash}.neff"

    def _generate_cache_key(
        self,
        hlo_protobuf: bytes,
        config: CompilerConfig,
        lnc_override: str,
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
    ) -> bytes:
        """Two-tier cache lookup and compilation."""

        # Skip caching if disabled
        if is_caching_disabled():
            return CompilerSubprocess.compile_hlo_protobuf_to_neff(
                hlo_protobuf, config, lnc_override, ir_type
            )

        import neuronxcc

        cache_key = self._generate_cache_key(
            hlo_protobuf, config, lnc_override, neuronxcc.__version__
        )

        # 1. Check NFS cache first (fastest check)
        nfs_path = self._nfs_cache_path(cache_key)
        if nfs_path.exists():
            try:
                return nfs_path.read_bytes()
            except OSError:
                # NFS read failed, continue to local compilation
                pass

        # 2. Use file lock for compilation
        key_hash = hashlib.sha256(str(cache_key).encode()).hexdigest()
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
                    return nfs_path.read_bytes()
                except OSError:
                    pass

            # Compile and cache
            neff_bytes = CompilerSubprocess.compile_hlo_protobuf_to_neff(
                hlo_protobuf, config, lnc_override, ir_type
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
        input_file: Path,
        output_file: Path,
        config: CompilerConfig,
        lnc_override: str | None = None,
        work_dir: str | None = None,
    ) -> None:
        """Compile HLO protobuf to NEFF.

        Args:
            input_file: Path to HLO protobuf file
            output_file: Path to output NEFF file
            config: Compiler configuration
            lnc_override: Optional override for logical neuron cores

        Raises:
            NEFFCompilationError: If compilation fails
        """
        neuronx_cc = CompilerSubprocess.find_neuronx_cc()
        cmd = [
            neuronx_cc,
            *config.get_neuronx_cc_args(str(input_file), str(output_file), lnc_override),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
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
    ) -> bytes:
        """Compile HLO protobuf or MLIR directly to NEFF.

        Args:
            hlo_protobuf: HLO protobuf bytes or MLIR bytes
            config: Compiler configuration
            lnc_override: Optional override for logical neuron cores
            ir_type: Type of IR ("XLA" for HLO protobuf, "StableHLO" for MLIR)

        Returns:
            Compiled NEFF bytes

        Raises:
            NEFFCompilationError: If NEFF compilation fails
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Determine file format based on ir_type
            # "StableHLO" uses MLIR format, "XLA" uses HLO protobuf
            hlo_file = (
                tmpdir / "module.mlir" if ir_type == "StableHLO" else tmpdir / "module.hlo.pb"
            )

            hlo_file.write_bytes(hlo_protobuf)

            # Compile to NEFF
            neff_file = tmpdir / "module.neff"
            # Passing work_dir is important when we run multi-worker
            # Seeing a race condition where the artifacts get mangled
            # causing runtime errors. Passing the workdir keeps all
            # artifacts contained in their respective folder.
            CompilerSubprocess.run_neuronx_cc(
                hlo_file, neff_file, config, lnc_override, work_dir=tmpdir
            )

            # Read and return compiled NEFF
            return neff_file.read_bytes()
