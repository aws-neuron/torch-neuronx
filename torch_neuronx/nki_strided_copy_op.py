"""Register neuron::nki_strided_copy_ as a torch custom op.

This allows C++ code to call torch.ops.neuron.nki_strided_copy_(src, dst)
which dispatches to the NKI stride-2 gather/scatter kernel on device.
"""

from __future__ import annotations

import torch

_LIB_DEF = torch.library.Library("neuron", "DEF")
_LIB_DEF.define("nki_strided_copy_(Tensor src, Tensor(a!) dst) -> Tensor(a!)")

_LIB_IMPL = torch.library.Library("neuron", "IMPL", "PrivateUse1")


def _nki_stride2_copy_impl(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """NKI-based stride-2 copy between Neuron tensors.

    Handles stride-2 patterns (packed int64 gather/scatter) via DMA.
    Falls back to element-wise NRT copy for other patterns.
    """
    if src.numel() == 0:
        return dst

    from torch_neuronx.python_ops.nki_kernels.stride2_gather import (
        run_stride2_gather,
        run_stride2_scatter,
    )

    src_strides = tuple(int(s) for s in src.stride())
    dst_strides = tuple(int(s) for s in dst.stride())

    # Stride-2 gather: non-contiguous src (stride 2 on last dim) → contiguous dst
    if dst.is_contiguous() and not src.is_contiguous() and src_strides[-1] == 2:
        parity = int(src.storage_offset()) % 2
        run_stride2_gather(src, dst, parity=parity)
        return dst

    # Stride-2 scatter: contiguous src → non-contiguous dst (stride 2 on last dim).
    # Writes src into the selected lanes only and preserves the other lanes,
    # mirroring the gather branch. The NKI kernel does a burst load of the
    # destination tile, overwrites the even/odd lanes on-chip, and burst-stores.
    if src.is_contiguous() and not dst.is_contiguous() and dst_strides[-1] == 2:
        parity = int(dst.storage_offset()) % 2
        run_stride2_scatter(src, dst, parity=parity)
        return dst

    # Fallback: use NRT element-wise copy
    import torch_neuronx._C as _C

    _C._nrt_copy_strided_neuron_to_neuron(src, dst)
    return dst


_LIB_IMPL.impl("nki_strided_copy_", _nki_stride2_copy_impl)
