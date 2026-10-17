"""Device-side dtype casting helper for Neuron tensors.

This module provides a simple function to cast from one Neuron tensor dtype to
another without bouncing through CPU. It uses the MLIR kernel path to perform
an on-device astype.

Supports <=32-bit dtypes and int64/uint64 (the compiler handles S64 HLO
natively, and the C++ NRT layer reinterprets int64 as 2xint32 for transfers).

Note: Casting to or from float64 is not supported on device and should be
handled via CPU bounce by the caller.
"""

from __future__ import annotations

import torch

from torch_neuronx.python_ops.kernel_cache import get_or_create_mlir_kernel


def _mlir_cast_fn(x, out_dtype=None, out=None):
    return x.to(dtype=out_dtype) if out_dtype else x


class DeviceCastMLIRImpl:
    """Perform dtype casts on a Neuron device through an MLIR kernel."""

    @staticmethod
    def cast(src_prepared, dst_prepared):
        """Cast a prepared source tensor into a prepared destination tensor.

        Args:
            src_prepared (torch.Tensor): Contiguous source tensor on a Neuron device.
            dst_prepared (torch.Tensor): Contiguous destination tensor overwritten with
                the cast values.
        """
        kernel = get_or_create_mlir_kernel(
            mlir_fn=_mlir_cast_fn,
            op_name="aten::copy_",
            static_argnames=("out_dtype",),
        )
        kernel(src_prepared, out=dst_prepared, out_dtype=dst_prepared.dtype)


def device_cast_neuron_to_neuron(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Cast a tensor into a destination on a Neuron device without CPU bounce.

    Args:
        src (torch.Tensor): Source tensor on a Neuron device.
        dst (torch.Tensor): Shape-compatible destination tensor on a Neuron device.

    Raises:
        RuntimeError: If either tensor is not on a Neuron device, or a native copy
            uses different Neuron cores, a noncontiguous destination, or unequal byte counts.

    Note:
        Matching dtypes enqueue a raw native copy on the current Neuron stream and do
        not interpret source strides. Other dtype changes first make the source contiguous
        and use MLIR, including ``int64`` and ``uint64``; casts to or from ``float64``
        are unsupported. The MLIR path does not update an original noncontiguous ``dst``.
    """
    import torch_neuronx._C as _C

    if src.device.type != "neuron" or dst.device.type != "neuron":
        raise RuntimeError("device_cast_neuron_to_neuron expects both tensors on Neuron device")

    # Fast path: same dtype -> D2D copy
    if src.dtype == dst.dtype:
        _C._nrt_copy_neuron_to_neuron_tensor(src, dst)
        return

    # Ensure contiguous inputs/outputs for kernel simplicity without redispatch
    from torch_neuronx.python_ops.cast_policy import ensure_contiguous_on_device

    src_prepared = ensure_contiguous_on_device(src)
    dst_prepared = ensure_contiguous_on_device(dst)

    DeviceCastMLIRImpl.cast(src_prepared, dst_prepared)
