"""NKI kernel support for torch-neuronx (free-function get_config)."""

import functools
import hashlib
import os
import warnings
from typing import Any

from nki.compiler.ncc_driver import FrameworkConfig
from nki.framework.compiled import CompileKernel
from torch.utils._filelock import FileLock

from torch_neuronx.python_ops.compilation.hlo_cache import CACHE_DIR


@functools.lru_cache(maxsize=1)
def _warn_trace_cache_once() -> None:
    """Emit a one-time (per-process) warning when the NKI trace cache is on."""
    if os.environ.get("NKI_ENABLE_TRACE_CACHE", "1") == "1":
        warnings.warn(
            "NKI_ENABLE_TRACE_CACHE=1: the NKI kernel compile cache is persisted "
            "across processes. The caching process may not capture all kernel "
            "environment dependencies or non-argument values and can yield stale "
            "cache hits; see the torch-neuronx environment variables documentation "
            "for details. Set NKI_ENABLE_TRACE_CACHE=0 to disable.",
            stacklevel=2,
        )


def return_types_of(config: FrameworkConfig) -> list[tuple]:
    """Extract output dtype and shape pairs from an NKI framework configuration.

    Args:
        config (FrameworkConfig): Compiled NKI framework configuration.

    Returns:
        list[tuple]: ``(dtype, shape)`` pair for each output specification.
    """
    return [(s.dtype, s.shape) for s in config.output_specs]


def backend_config_str(config: FrameworkConfig) -> str:
    """Convert an NKI framework configuration's ASCII payload to text.

    Args:
        config (FrameworkConfig): Configuration containing an ASCII payload.

    Returns:
        str: ASCII text stored in ``backend_config_b64``.

    Raises:
        UnicodeDecodeError: If the payload contains non-ASCII bytes.
    """
    return config.backend_config_b64.decode("ascii")


def _get_config_cache(kernel: CompileKernel) -> dict | None:
    """Return (or create) the per-kernel-func FrameworkConfig cache dict."""
    if kernel._get_compile_cache() is None:
        return None
    if not hasattr(kernel.func, "_config_cache"):
        kernel.func._config_cache = {}
    return kernel.func._config_cache


def get_config(kernel: CompileKernel, **kwargs: Any) -> FrameworkConfig:
    """Compile an NKI kernel and return its framework configuration.

    Args:
        kernel (CompileKernel): NKI kernel specialization to compile.
        **kwargs (Any): Kernel arguments and metadata forwarded to NKI.

    Returns:
        FrameworkConfig: Configuration returned by ``kernel.compile``.

    Note:
        This serves the HOP and flex-attention lowering paths. It checks an
        in-memory cache first, then uses a cache-keyed cross-rank file lock so
        waiters can reuse NKI's persistent compilation cache. However,
        ``kernel.get_cache_key`` may itself compile before the lock; the lock
        serializes only the subsequent ``kernel.compile`` call in that case.
        A process warning describes stale trace-cache risks when
        ``NKI_ENABLE_TRACE_CACHE=1``.
    """
    _warn_trace_cache_once()

    cache_key = kernel.get_cache_key(**kwargs)

    dc_cache = _get_config_cache(kernel)
    if dc_cache is not None and cache_key is not None and cache_key in dc_cache:
        return dc_cache[cache_key]

    # Cross-rank coordination: serialize compilation of a given key across
    # ranks so only one rank performs the (potentially expensive) NKI
    # compile; the others block on the lock and then hit kernel.compile()'s
    # own persistent cache.
    #
    # NOTE: kernel.get_cache_key() currently derives the key by calling
    # compile() internally, so the first compile happens before the lock is
    # taken. This lock therefore only fully serializes ranks once NKI makes
    # get_cache_key() a cheap, compile-free operation.
    if cache_key is not None:
        lock_path = os.path.join(
            CACHE_DIR, hashlib.sha256(cache_key.encode()).hexdigest() + ".lock"
        )
        with FileLock(lock_path, timeout=1800):
            # NOTE: We rely on NKI's persistent cache to handle picking up the
            # result of the kernel compile from another rank
            config, _ = kernel.compile(**kwargs)
    else:
        config, _ = kernel.compile(**kwargs)

    if dc_cache is not None and cache_key is not None:
        dc_cache[cache_key] = config
    return config
