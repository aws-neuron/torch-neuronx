"""aten::_chunk_cat for Neuron: one _foreach_copy_ over the chunk segments.

``_chunk_cat(tensors, dim, num_chunks, out=)`` pads ``size(dim)`` of each tensor to a
multiple of ``num_chunks``. It views the tensor as ``[..., num_chunks, k_i]`` and
concatenates these views on the last dim. So ``out[..., c, off_i:off_i + k_i]`` is chunk
``c`` of tensor ``i``. FSDP v2 uses the op to stage the grads of a layer into the
reduce-scatter input.

PyTorch's composite kernel builds the result with one ``aten::cat`` over the
``[num_chunks, k_i]`` views. On Neuron this concatenate of a few wide rows compiles to a
slow kernel. In the FSDP v2 case (9 grads into ``[16, 7.6M]``, bf16 to fp32) it runs at
~20 GB/s: 30 ms per Qwen3-32B layer, a quarter of the device time of a step.

Each (tensor, chunk) segment is a contiguous 1-D run in the source and in ``out``. One
``aten::_foreach_copy_`` over these segments gives the same result at ~400 GB/s (20x).
The pad runs are copied from one scratch zeros tensor in the same call. An ``out`` with a
different shape or with strides takes PyTorch's composite path, which resizes or writes
strided as ``cat`` does. So does a segment count over ``_MAX_FAST_PATH_SEGMENTS``.
"""

from __future__ import annotations

import math

import torch
from torch._decomp.decompositions import _chunk_cat as _composite_chunk_cat
from torch._decomp.decompositions import _preprocess_chunk_cat_inputs

# The fast path costs ~25 us of host time per (tensor, lead row, chunk) segment on trn2: two
# Python views plus the _foreach_copy_ dispatch. The composite path costs 4-6 ms for one FSDP
# layer (233 MB) at 24 or more chunks, 34 ms at 16 chunks. The paths cross near 220 segments.
# The composite path also holds one padded view per tensor instead of two views per segment.
_MAX_FAST_PATH_SEGMENTS = 256


def _chunk_cat_layout(tensors, dim, num_chunks):
    """Return (dim, leading rows, valid numel past dim per tensor, chunk width per tensor).

    ``_chunk_cat`` pads ``size(dim)`` to a multiple of ``num_chunks``. So a chunk holds
    ``ceil(size(dim) / num_chunks)`` rows of ``dim``, each with the full trailing extent.
    """
    dim = _preprocess_chunk_cat_inputs(tensors, dim, num_chunks)
    n_lead = math.prod(tensors[0].shape[:dim])
    valid, chunk = [], []
    for t in tensors:
        rows = t.shape[dim]
        trailing = math.prod(t.shape[dim + 1 :])
        valid.append(rows * trailing)
        chunk.append(-(-rows // num_chunks) * trailing)
    return dim, n_lead, valid, chunk


def _chunk_cat_out_neuron(tensors, dim, num_chunks, *, out):
    """Neuron implementation of ``aten::_chunk_cat.out``."""
    tensors = list(tensors)
    dim, n_lead, valid, chunk = _chunk_cat_layout(tensors, dim, num_chunks)
    # torchgen's device and out-dtype checks do not cover a hand-registered kernel (see equal.py).
    if out.device != tensors[0].device:
        raise RuntimeError(
            f"Expected all tensors to be on the same device, but got out on {out.device} "
            f"and tensors on {tensors[0].device}."
        )
    if not torch.can_cast(tensors[0].dtype, out.dtype):
        raise TypeError(
            f"result type {tensors[0].dtype} can't be cast to the desired output type {out.dtype}"
        )
    width = sum(chunk)
    shape = (*tensors[0].shape[:dim], num_chunks, width)
    if (
        n_lead == 0
        or n_lead * num_chunks * len(tensors) > _MAX_FAST_PATH_SEGMENTS
        or tuple(out.shape) != shape
        or not out.is_contiguous()
    ):
        return _composite_chunk_cat(tensors, dim, num_chunks, out=out)

    out_rows = out.view(n_lead, num_chunks, width)
    max_pad = max((k * num_chunks - n for n, k in zip(valid, chunk, strict=True)), default=0)
    zeros = out.new_zeros(max_pad) if max_pad else None
    dsts, srcs = [], []
    off = 0
    for t, n, k in zip(tensors, valid, chunk, strict=True):
        # _foreach_copy_ bug: strided sources plus the repeated zeros source read wrong data.
        rows = t.contiguous().view(n_lead, n)
        for lead in range(n_lead):
            for c in range(num_chunks):
                start = c * k
                length = max(0, min(k, n - start))
                if length:
                    dsts.append(out_rows[lead, c, off : off + length])
                    srcs.append(rows[lead, start : start + length])
                if length < k:
                    dsts.append(out_rows[lead, c, off + length : off + k])
                    srcs.append(zeros[: k - length])
        off += k
    torch._foreach_copy_(dsts, srcs)
    return out


def _chunk_cat_neuron(tensors, dim, num_chunks):
    """Neuron implementation of ``aten::_chunk_cat``: allocate ``out`` and fill it."""
    tensors = list(tensors)
    dim, _, _, chunk = _chunk_cat_layout(tensors, dim, num_chunks)
    out = tensors[0].new_empty((*tensors[0].shape[:dim], num_chunks, sum(chunk)))
    return _chunk_cat_out_neuron(tensors, dim, num_chunks, out=out)
