"""Device-side dtype casting helper for Neuron tensors.

This module provides a simple function to cast from one Neuron tensor dtype to
another without bouncing through CPU for <=32-bit dtypes. It uses the same
JAX kernel path as the copy_ implementation to perform an on-device astype.

Note: Casting to or from 64-bit dtypes (float64/int64) is not supported on
Neuron and should be handled via CPU bounce by the caller.
"""

from __future__ import annotations

import torch

from torch_neuron.python_ops.jax.context import ExecutionContext
from torch_neuron.python_ops.jax.type_converter import JaxTypeConverter
from torch_neuron.python_ops.kernel_cache import get_or_create_jax_kernel
from torch_neuron.python_ops.processors.no_cast_argument_processor import (
    NoCastArgumentProcessor,
)


def _jax_copy_fn(x, out_dtype=None):
    if out_dtype is None:
        return x
    return x.astype(out_dtype)


def device_cast_neuron_to_neuron(src: torch.Tensor, dst: torch.Tensor, non_blocking: bool) -> None:
    """Cast `src` into `dst` on Neuron device without CPU bounce for <=32-bit dtypes.

    - If dtypes match, performs a direct device-to-device copy.
    - Otherwise, uses a tiny JAX kernel to cast on device and write into `dst`.

    Args:
        src: Source Neuron tensor
        dst: Destination Neuron tensor (already allocated with target dtype)
    """
    import torch_neuron._C as _C

    if src.device.type != "neuron" or dst.device.type != "neuron":
        raise RuntimeError("device_cast_neuron_to_neuron expects both tensors on Neuron device")

    # Fast path: same dtype -> D2D copy
    if src.dtype == dst.dtype:
        _C._nrt_copy_neuron_to_neuron_tensor(src, dst, non_blocking=non_blocking)
        return

    # Ensure contiguous inputs/outputs for kernel simplicity without redispatch
    # Use internal helper to avoid calling aten::contiguous through dispatcher
    from torch_neuron.python_ops.cast_policy import ensure_contiguous_on_device

    src_prepared = ensure_contiguous_on_device(src)
    dst_prepared = ensure_contiguous_on_device(dst)

    # Reuse a tiny JAX kernel via process-local cache
    kernel = get_or_create_jax_kernel(
        jax_fn=_jax_copy_fn,
        op_name="aten::copy_",  # reuse copy_ meta/out semantics for kernel plumbing
        static_argnames=("out_dtype",),
        argument_processor_cls=NoCastArgumentProcessor,
    )

    exec_ctx = ExecutionContext(original_inputs=(src_prepared,))
    jax_out_dtype = JaxTypeConverter().to_execution_jax_dtype(dst_prepared.dtype)

    # Execute kernel to write into dst
    kernel(src_prepared, context=exec_ctx, out=dst_prepared, out_dtype=jax_out_dtype)
