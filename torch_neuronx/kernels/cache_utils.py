"""Cache configuration utilities."""

import os
import tempfile
from pathlib import Path


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


def atomic_write_bytes(file_path: Path, data: bytes) -> None:
    """Atomically write bytes to file using temp file + rename."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file in same directory
    with tempfile.NamedTemporaryFile(dir=file_path.parent, delete=False, suffix=".tmp") as tmp_file:
        tmp_file.write(data)
        tmp_path = Path(tmp_file.name)

    # Atomic rename
    tmp_path.rename(file_path)
