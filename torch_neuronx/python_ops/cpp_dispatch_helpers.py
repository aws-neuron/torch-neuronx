"""
C++ dispatch helpers for NeuronDispatcher.

Provides:
- make_lower_func: Creates a lower_func for any traceable Python function,
  enabling C++ dispatch via _register_neuron_op.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def make_lower_func(torch_fn, op_name, static_argnums=(), static_argnames=()):
    """Create a lowering callback for a traceable Python function.

    Args:
        torch_fn (Callable): Python function that ``make_fx`` can trace.
        op_name (str): Operation name associated with the kernel.
        static_argnums (tuple[int, ...], optional): Positional arguments treated as static.
            Defaults to ``()``.
        static_argnames (tuple[str, ...], optional): Named arguments treated as static.
            Defaults to ``()``.

    Returns:
        Callable: Dynamo-disabled callback that compiles and lowers to StableHLO.

    Note:
        Wrap ``torch_fn`` in ``TorchMlirKernel``. Use ``make_composite_lower`` for
        ``Operation`` instances.
    """
    from .torch_mlir.kernel import TorchMlirKernel

    kernel = TorchMlirKernel(
        torch_fn,
        op_name,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
    )

    @torch._dynamo.disable
    def lower_func(*args, **kwargs):
        """Lower dispatcher arguments through the captured MLIR kernel.

        Args:
            *args (Any): Positional arguments forwarded to the kernel.
            **kwargs (Any): Keyword arguments forwarded after metadata removal.

        Returns:
            Any: Lowered kernel representation.

        Note:
            ``original_args`` and ``original_kwargs`` metadata are discarded.
        """
        kwargs.pop("original_args", None)
        kwargs.pop("original_kwargs", None)
        return kernel.lower(*args, **kwargs)

    return lower_func
