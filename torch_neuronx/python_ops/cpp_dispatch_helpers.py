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
    """Create a lower_func for any traceable Python function.

    Wraps the function in a TorchMlirKernel for StableHLO compilation.
    Use this for ops that aren't Operation instances but can be traced
    by make_fx. For Operation instances, use make_composite_lower instead.

    Usage:
        _C._register_neuron_op("my_op", my_wrapper,
            make_lower_func(my_traceable_fn, "my_op", static_argnums=(1,)))
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
        kwargs.pop("original_args", None)
        kwargs.pop("original_kwargs", None)
        return kernel.lower(*args, **kwargs)

    return lower_func
