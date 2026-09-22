"""Cache configuration utilities."""

import hashlib
import logging
import os
import secrets
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def get_artifacts_directory() -> Path:
    """Get the artifacts directory from environment variable or default.

    Uses TORCH_NEURONX_DEBUG_DIR if set, otherwise /tmp/neuron_backend.
    """
    debug_dir = os.getenv("TORCH_NEURONX_DEBUG_DIR")
    path = Path(debug_dir if debug_dir else "/tmp/neuron_backend")
    return path


def get_nfs_cache_dir() -> str:
    """Get NFS cache directory from environment variable."""
    return os.getenv("TORCH_NEURONX_NEFF_CACHE_DIR", "/tmp/neff_cache")


def get_local_cache_dir() -> str:
    """Get local cache directory from environment variable."""
    return os.getenv("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "/tmp/local_cache")


def is_caching_disabled() -> bool:
    """Check if caching is disabled via environment variable."""
    return os.getenv("TORCH_NEURONX_NEFF_DISABLE_CACHE", "").lower() in ("true", "1", "yes")


def get_lock_timeout() -> int:
    """Get lock timeout from environment variable."""
    return int(os.getenv("TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT", "1200"))


def is_preserve_artifacts_enabled() -> bool:
    """Get whether artifact preservation enabled"""
    return os.getenv("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", "").lower() in (
        "true",
        "1",
        "yes",
    )


def atomic_write_bytes(file_path: Path, data: bytes) -> None:
    """Atomically write bytes to file using temp file + rename."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file in same directory
    with tempfile.NamedTemporaryFile(dir=file_path.parent, delete=False, suffix=".tmp") as tmp_file:
        tmp_file.write(data)
        tmp_path = Path(tmp_file.name)

    # Atomic rename
    tmp_path.rename(file_path)


class CompilationJob:
    """Manages a single compilation job's artifact paths and lifecycle.

    Each compilation job gets a unique directory under the base artifacts directory.
    Artifacts are always written; the ``preserve`` flag only controls cleanup.

    Directory hierarchy::

        {base_dir}/
        ├── {job_id}/
        │   ├── fx_graphs/{model}_{ts}.fx.txt
        │   ├── transformed_fx/{model}_{ts}.transformed.fx.txt
        │   ├── raw/{model}_{ts}.torch_raw.mlir
        │   ├── stablehlo/{model}_{ts}.stablehlo.mlir
        │   └── torch_mlir_error/{model}_{ts}.mlir
        └── neuronx-cc/
            └── {hash}/
    """

    def __init__(self, op_name: str):
        self.op_name = op_name
        self.timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.job_id = self._generate_job_id()
        self.preserve = is_preserve_artifacts_enabled()
        self._base_dir = get_artifacts_directory()
        self._job_dir = self._base_dir / self.job_id
        self._pass_counter = 0
        working_dir_hash = hashlib.sha256(self.job_id.encode(errors="ignore")).hexdigest()
        self._neuronx_cc_directory = self._base_dir / "neuronx-cc" / working_dir_hash

    def _generate_job_id(self) -> str:
        """Generate a unique kernel ID for compilation tracking.

        The kernel ID format is: {timestamp}_{op_name}_{hash}
        where:
        - timestamp: Local timezone timestamp in format YYYYMMDD_HHMMSS_microseconds
        - op_name: Operation name truncated/padded to 16 chars, sanitized
        - hash: 16 character random hex string

        Sanitization: Replaces | and whitespace characters with _ for filesystem safety.
        """
        # Sanitize op_name: replace | and whitespace with _
        sanitized_op_name = (
            self.op_name.replace("|", "_").replace(" ", "_").replace("\t", "_").replace("\n", "_")
        )

        # Normalize op_name to 16 chars
        if len(sanitized_op_name) > 16:
            normalized_op_name = sanitized_op_name[:16]
        else:
            normalized_op_name = sanitized_op_name.ljust(16, "_")

        # Generate 16-char random hex hash
        random_hash = secrets.token_hex(8)

        return f"{self.timestamp}_{normalized_op_name}_{random_hash}"

    def _make_path(self, subdir: str, model_name: str, suffix: str) -> Path:
        """Create an artifact path, ensuring parent directory exists."""
        directory = self._job_dir / subdir
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{model_name.lower()}_{self.timestamp}{suffix}"
        return directory / filename

    def save_fx_graph(self, model_name: str, gm) -> Path:
        """Save FX graph as human-readable text.

        Args:
            model_name: Model name for the artifact filename.
            gm: FX GraphModule to save.

        Returns:
            Path: {job_dir}/fx_graphs/{model}_{ts}.fx.txt
        """
        path = self._make_path("fx_graphs", model_name, ".fx.txt")
        with open(path, "w") as f:
            f.write(gm.print_readable(print_output=False))
        logger.debug(f"Saved FX graph to {path}")
        return path

    def save_pass_fx(self, model_name: str, gm, pass_name: str) -> Path:
        """Save FX graph after an individual pass under transformed_fx/.

        Args:
            model_name: Model name for the artifact filename.
            gm: FX GraphModule to save.
            pass_name: Name of the pass (typically the class name).

        Returns:
            Path: {job_dir}/transformed_fx/{model}_{ts}.pass_{idx}_{name}.fx.txt
        """
        pass_index = self._pass_counter
        self._pass_counter += 1
        suffix = f".pass_{pass_index}_{pass_name}.fx.txt"
        path = self._make_path("transformed_fx", model_name, suffix)
        with open(path, "w") as f:
            f.write(gm.print_readable(print_output=False))
        logger.debug(f"Saved FX graph after pass {pass_index} ({pass_name}) to {path}")
        return path

    def save_transformed_fx(self, model_name: str, gm) -> Path:
        """Save transformed FX graph as human-readable text.

        Args:
            model_name: Model name for the artifact filename.
            gm: FX GraphModule to save (after collective transforms).

        Returns:
            Path: {job_dir}/transformed_fx/{model}_{ts}.fx.txt
        """
        path = self._make_path("transformed_fx", model_name, ".fx.txt")
        with open(path, "w") as f:
            f.write(gm.print_readable(print_output=False))
        logger.debug(f"Saved transformed FX graph to {path}")
        return path

    def save_raw_torch(self, model_name: str, mlir_module) -> Path:
        """Save RAW Torch IR as text.

        Args:
            model_name: Model name for the artifact filename.
            mlir_module: MLIR module to save.

        Returns:
            Path: {job_dir}/raw/{model}_{ts}.torch_raw.mlir
        """
        path = self._make_path("raw", model_name, ".torch_raw.mlir")
        with open(path, "w") as f:
            f.write(mlir_module.operation.get_asm(enable_debug_info=True))
        logger.debug(f"Saved RAW Torch module to {path}")
        return path

    def save_stablehlo(self, model_name: str, mlir_module) -> Path:
        """Save StableHLO as text.

        Args:
            model_name: Model name for the artifact filename.
            mlir_module: MLIR module to save.

        Returns:
            Path: {job_dir}/stablehlo/{model}_{ts}.stablehlo.mlir
        """
        path = self._make_path("stablehlo", model_name, ".stablehlo.mlir")
        with open(path, "w") as f:
            f.write(mlir_module.operation.get_asm(enable_debug_info=True))
        logger.debug(f"Saved StableHLO to {path}")
        return path

    def get_err_mlir_path(self, model_name: str) -> Path:
        """Get the path for saving MLIR error artifacts.

        Returns:
            Path: {job_dir}/torch_mlir_error/{model}_{ts}.mlir
        """
        return self._make_path("torch_mlir_error", model_name, ".mlir")

    def get_neuronx_cc_workdir(self) -> Path:
        """Get the working directory for neuronx-cc compiler invocation.

        Returns:
            Path: Working directory path for neuronx-cc.
        """
        return self._neuronx_cc_directory

    def cleanup(self):
        """Remove job directory if preserve is False.

        Never touches the neuronx-cc directory.
        """
        try:
            if self.preserve:
                logger.debug(
                    f"Preserving compilation artifacts: job={self._job_dir} "
                    f"neuronx-cc={self._neuronx_cc_directory}"
                )
                return
            if self._job_dir.exists():
                logger.debug(f"Cleaning up job directory: {self._job_dir}")
                shutil.rmtree(self._job_dir)
            if self._neuronx_cc_directory.exists():
                logger.debug(f"Cleaning up neuronx-cc directory: {self._neuronx_cc_directory}")
                shutil.rmtree(self._neuronx_cc_directory)

        except Exception as e:
            logger.warning(f"Failed to clean up {self._job_dir}: {e}")


@contextmanager
def compilation_job(op_name: str):
    """Context manager for CompilationJob that ensures cleanup on exit.

    Creates a CompilationJob and automatically calls cleanup() when the context exits,
    regardless of whether an exception occurred.

    Args:
        op_name: Operation name for the compilation job.

    Yields:
        CompilationJob: The compilation job instance.

    Example:
        >>> with compilation_job("my_model") as job:
        ...     job.save_fx_graph("my_model", gm)
        ...     # ... do compilation work ...
        ... # cleanup() is called automatically here
    """
    job = CompilationJob(op_name)
    try:
        yield job
    finally:
        job.cleanup()
