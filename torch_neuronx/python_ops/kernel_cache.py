"""Process-local cache for small JAX kernels.

This module provides a simple, thread-safe cache to reuse JaxKernel instances
across calls that share the same configuration. It avoids repeated construction
of identical kernel objects in hot paths (e.g., copy and device_cast), while
leaving compilation caching inside JaxKernel unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock

from torch_neuronx.python_ops.jax.kernel import JaxKernel
from torch_neuronx.python_ops.processors.argument_processor import ArgumentProcessor


@dataclass(frozen=True)
class KernelKey:
    """Normalized identity for a kernel configuration."""

    fn_id: str
    op_name: str
    static_argnums: tuple[int, ...]
    static_argnames: tuple[str, ...]
    output_params: tuple[str, ...]
    argproc_id: str


_CACHE: dict[KernelKey, JaxKernel] = {}
_LOCK = RLock()


def _fn_identity(fn: Callable) -> str:
    mod = getattr(fn, "__module__", None)
    name = getattr(fn, "__name__", None)
    if mod and name:
        return f"{mod}:{name}"
    # Fallback for lambdas/closures
    return repr(fn)


def get_or_create_jax_kernel(
    *,
    jax_fn: Callable,
    op_name: str,
    static_argnums: tuple[int, ...] | None = None,
    static_argnames: tuple[str, ...] | None = None,
    output_params: tuple[str, ...] | None = None,
    argument_processor_cls: type[ArgumentProcessor] | None = None,
) -> JaxKernel:
    """Return a cached JaxKernel for the given configuration, creating if absent.

    The cache key includes function identity, op_name, static args spec, output
    params, and the requested ArgumentProcessor class. On cache miss, a new
    JaxKernel is constructed and its `arg_processor` is set to an instance of
    the given `argument_processor_cls` when provided.
    """

    key = KernelKey(
        fn_id=_fn_identity(jax_fn),
        op_name=op_name,
        static_argnums=tuple(static_argnums or ()),
        static_argnames=tuple(static_argnames or ()),
        output_params=tuple(output_params or ()),
        argproc_id=(argument_processor_cls.__name__ if argument_processor_cls else "default"),
    )
    with _LOCK:
        k = _CACHE.get(key)
        if k is not None:
            return k
        # First time: construct kernel
        kernel = JaxKernel(
            jax_fn,
            op_name=op_name,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            output_params=output_params,
        )
        if argument_processor_cls is not None:
            kernel.arg_processor = argument_processor_cls(static_argnames)  # type: ignore[assignment]
        _CACHE[key] = kernel
        return kernel


def kernel_cache_size() -> int:
    """Return the number of cached kernel instances (for tests/diagnostics)."""
    with _LOCK:
        return len(_CACHE)
