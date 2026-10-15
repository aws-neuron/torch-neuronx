"""Unified Python lower function for NKI kernel cache misses.

Called by C++ NkiDispatchCore when a custom NKI kernel has a cache miss.
Handles custom NKI kernels (kernel_idx >= 0) via NKIRegistry + NkiKernel.
"""

import logging

import torch

logger = logging.getLogger(__name__)


@torch._dynamo.disable
def nki_call_lower_func(
    tensor_args,
    kernel_idx,
    constant_args_key,
    grid,
    operand_output_aliases,
    has_collectives,
):
    """Lower a custom NKI kernel to StableHLO.

    Called by C++ NkiDispatchCore on cache miss.
    """
    prior = torch._dynamo.eval_frame.set_eval_frame(None)
    try:
        return _lower_custom_nki(
            tensor_args,
            kernel_idx,
            constant_args_key,
            grid,
        )
    finally:
        torch._dynamo.eval_frame.set_eval_frame(prior)


def _lower_custom_nki(
    tensor_args,
    kernel_idx,
    constant_args_key,
    grid,
):
    """Lower a custom NKI kernel via NKIRegistry + NkiKernel."""
    from torch_neuronx.nki_hop import get_kernel_config, kernel_registry
    from torch_neuronx.nki_kernel import backend_config_str, return_types_of
    from torch_neuronx.python_ops.torch_mlir.nki_op_impl import _global_nki_op

    arg_names = kernel_registry.get_kernel_arg_names(kernel_idx)
    constant_args = kernel_registry.get_constant_args(constant_args_key)
    tensor_arg_names = [name for name in arg_names if name not in constant_args]

    dconfig = get_kernel_config(kernel_idx, grid, tensor_args, tensor_arg_names, constant_args_key)
    backend_config = backend_config_str(dconfig)
    aliases = dconfig.operand_output_aliases

    nki_params = {
        "kernel_idx": kernel_idx,
        "grid": grid,
        "backend_config": backend_config,
        "operand_output_aliases": aliases,
        "arg_names": tensor_arg_names,
        "constant_args_key": constant_args_key,
    }

    _global_nki_op.kernel.operand_output_aliases = aliases
    result = _global_nki_op.kernel.lower(*tensor_args, **nki_params)

    # Preserve the collective flag from the config: the NKI frontend sets
    # has_collectives when the kernel emits ncc.* ops (FrameworkConfig.has_collectives)
    result.has_collectives = dconfig.has_collectives

    # Attach config so C++ can cache it in DispatchCacheEntry
    result.nki_return_types = return_types_of(dconfig)
    result.nki_operand_output_aliases = dconfig.operand_output_aliases

    return result
