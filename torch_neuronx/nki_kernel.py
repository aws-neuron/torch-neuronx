"""NKI kernel support for torch-neuronx (free-function get_config)."""

import hashlib
import os
from typing import Any

from nki.compiler.ncc_driver import FrameworkConfig
from nki.framework.compiled import CompileKernel
from torch.utils._filelock import FileLock

from torch_neuronx.python_ops.compilation.hlo_cache import CACHE_DIR


def return_types_of(config: FrameworkConfig) -> list[tuple]:
    """Extract (dtype, shape) tuples from FrameworkConfig.output_specs."""
    return [(s.dtype, s.shape) for s in config.output_specs]


def backend_config_str(config: FrameworkConfig) -> str:
    """Decode backend_config_b64 to a plain string."""
    return config.backend_config_b64.decode("ascii")


def _get_config_cache(kernel: CompileKernel) -> dict | None:
    """Return (or create) the per-kernel-func FrameworkConfig cache dict."""
    if kernel._get_compile_cache() is None:
        return None
    if not hasattr(kernel.func, "_config_cache"):
        kernel.func._config_cache = {}
    return kernel.func._config_cache


def get_config(kernel: CompileKernel, **kwargs: Any) -> FrameworkConfig:
    """Compile a NKI kernel and return a FrameworkConfig.

    This is the single entry-point used by the HOP dispatch layer
    and by flex-attention lowering.  Usage::

        config = get_config(kernel[grid], **meta_args)

    Caching layers (checked in order):
    1. In-memory cache keyed by NKI's cache_key.
    2. Cross-rank file lock so only one rank compiles per unique key; the
       rest wait on the lock and pick up the on-disk compilation cache that
       kernel.compile() maintains internally.
    """
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
