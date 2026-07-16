"""NKI kernel support for torch-neuronx (free-function get_config)."""

from typing import Any

from nki.compiler.ncc_driver import FrameworkConfig
from nki.framework.compiled import CompileKernel


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

    Caching: in-memory cache keyed by NKI's cache_key avoids repeated
    compile() calls for the same kernel+inputs combination.
    """
    cache_key = kernel.get_cache_key(**kwargs)

    dc_cache = _get_config_cache(kernel)
    if dc_cache is not None and cache_key is not None and cache_key in dc_cache:
        return dc_cache[cache_key]

    config, _ = kernel.compile(**kwargs)

    if dc_cache is not None and cache_key is not None:
        dc_cache[cache_key] = config
    return config
