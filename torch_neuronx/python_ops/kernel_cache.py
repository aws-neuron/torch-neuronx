"""Process-local cache for small kernels.

This module provides a simple, thread-safe cache to reuse kernel instances
across calls that share the same configuration. It avoids repeated construction
of identical kernel objects in hot paths (e.g., device_cast).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock

from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel


@dataclass(frozen=True)
class KernelKey:
    """Identify a normalized MLIR kernel configuration.

    Args:
        fn_id (str): Module-and-name callable identifier used in the cache key.
        op_name (str): Operation name associated with the kernel.
        static_argnums (tuple[int, ...]): Positional arguments treated as static.
        static_argnames (tuple[str, ...]): Named arguments treated as static.
        output_params (tuple[str, ...]): Parameter names used as outputs.
        argproc_id (str): Identity of the argument processor configuration.
    """

    fn_id: str
    op_name: str
    static_argnums: tuple[int, ...]
    static_argnames: tuple[str, ...]
    output_params: tuple[str, ...]
    argproc_id: str


_MLIR_CACHE: dict[KernelKey, TorchMlirKernel] = {}
_MLIR_LOCK = RLock()


def _fn_identity(fn: Callable) -> str:
    mod = getattr(fn, "__module__", None)
    name = getattr(fn, "__name__", None)
    if mod and name:
        return f"{mod}:{name}"
    # Fallback for lambdas/closures
    return repr(fn)


def get_or_create_mlir_kernel(
    *,
    mlir_fn: Callable,
    op_name: str,
    static_argnames: tuple[str, ...] | None = None,
) -> TorchMlirKernel:
    """Return a cached MLIR kernel for a normalized configuration.

    Args:
        mlir_fn (Callable): Python function wrapped by the kernel.
        op_name (str): Operation name used for lowering.
        static_argnames (tuple[str, ...], optional): Named arguments treated as static.
            Defaults to ``None``.

    Returns:
        TorchMlirKernel: Existing or newly created kernel instance.

    Note:
        Cache lookup and creation are process-local and protected by a reentrant lock.
    """
    key = KernelKey(
        fn_id=_fn_identity(mlir_fn),
        op_name=op_name,
        static_argnums=(),
        static_argnames=tuple(static_argnames or ()),
        output_params=(),
        argproc_id="mlir",
    )
    with _MLIR_LOCK:
        k = _MLIR_CACHE.get(key)
        if k is not None:
            return k
        # Create MLIR kernel
        kernel = TorchMlirKernel(
            torch_fn=mlir_fn,
            op_name=op_name,
            static_argnames=static_argnames,
        )
        _MLIR_CACHE[key] = kernel
        return kernel


def kernel_cache_size() -> int:
    """Return the number of process-local cached kernel instances.

    Returns:
        int: Current number of entries in the MLIR kernel cache.
    """
    with _MLIR_LOCK:
        return len(_MLIR_CACHE)
