"""Free-dim concat (2-input) dynamic NKI adapter.

Covers ``aten::cat`` of exactly two 2-D+ tensors along the last dim:
[*lead, H1] , [*lead, H2] -> [*lead, H1+H2]. Leading dims flatten to one dynamic
row count M (padded to a 128-multiple); H1 + H2 + dtype are baked into the NEFF.
This is the MoE backward of the block's ``.chunk(2, dim=-1)`` (grad flows back as
cat([grad_gate, grad_up], dim=-1)). More than two tensors, non-last-dim concat,
and mismatched leading shapes/dtypes decline to the default lowering.
"""

import torch

from ..adapter_utils import (
    _SUPPORTED_DTYPES,
    _TILE_M,
    _ceil_to_tile,
    _compile_or_load_neff,
    _dest_buffer,
    _num_tiles_tensor,
    _pad_or_contig,
)
from ..registry import DynamicNkiEntry, register_dynamic_nki


def _cat_tensor_pair(tensors):
    """The two concat operands if `tensors` is a length-2 tensor list, else None."""
    if not isinstance(tensors, list | tuple) or len(tensors) != 2:
        return None
    a, b = tensors
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return None
    return a, b


def _is_last_dim(dim, ndim) -> bool:
    return dim in (ndim - 1, -1)


def _can_handle_cat(tensors, dim=0, out=None) -> bool:
    if out is not None:
        return False
    pair = _cat_tensor_pair(tensors)
    if pair is None:
        return False
    a, b = pair
    if a.device.type != "neuron" or a.dtype not in _SUPPORTED_DTYPES:
        return False
    if b.dtype != a.dtype or b.device.type != a.device.type:
        return False
    if a.dim() < 2 or a.dim() != b.dim():
        return False
    if not _is_last_dim(dim, a.dim()):
        return False
    # Leading dims (everything but the concatenated last dim) must match — they
    # flatten to the shared row count M.
    if a.shape[:-1] != b.shape[:-1]:
        return False
    H1, H2 = a.shape[-1], b.shape[-1]  # noqa: N806
    M = a.numel() // H1  # noqa: N806
    if M == 0 or H1 == 0 or H2 == 0:  # noqa: SIM103
        return False
    return True


def _cat_compile_key(tensors, dim=0, out=None, **_):
    """compile_key = (H1, H2, dtype) — both widths + dtype baked; M dynamic."""
    a, b = tensors[0], tensors[1]
    return (int(a.shape[-1]), int(b.shape[-1]), a.dtype)


def _compile_cat(compile_key) -> tuple[bytes, str]:
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    from ..wrappers import dynamic_cat_nki

    H1, H2, dtype = compile_key  # noqa: N806
    a = torch.empty(_TILE_M, H1, dtype=dtype, device="meta")
    b = torch.empty(_TILE_M, H2, dtype=dtype, device="meta")
    nm = torch.tensor([[1]], dtype=torch.int32, device="meta")
    out = torch.empty(_TILE_M, H1 + H2, dtype=dtype, device="meta")

    kernel = TorchMlirKernel(torch_fn=dynamic_cat_nki, op_name="dynamic_nki_cat")
    return _compile_or_load_neff(kernel, "dynamic_nki_cat", (a, b, nm), out)


def _prepare_cat_io(tensors, dim=0, out=None, dest=None, **_):
    """Flatten each operand's leading dims to [M, H], pad M to a 128-multiple;
    kernel writes the concatenated [M_ceil, H1+H2]."""
    a, b = tensors[0], tensors[1]
    H1, H2 = a.shape[-1], b.shape[-1]  # noqa: N806
    M = a.numel() // H1  # noqa: N806
    M_ceil = _ceil_to_tile(M)  # noqa: N806
    device = a.device

    a2 = _pad_or_contig(a.reshape(M, H1), M_ceil)
    b2 = _pad_or_contig(b.reshape(M, H2), M_ceil)
    nm = _num_tiles_tensor(M_ceil // _TILE_M, device)
    dest_buf = dest if M_ceil == M else None
    outbuf = _dest_buffer(dest_buf, M_ceil, H1 + H2, a.dtype, device)
    return (a2, b2, nm), (outbuf,), M


def _reshape_cat_output(outputs, original_args):
    """Slice off padded rows, restore leading dims: [*lead, H1+H2]."""
    tensors = original_args[0]
    a = tensors[0]
    H1 = a.shape[-1]  # noqa: N806
    M = a.numel() // H1  # noqa: N806
    lead = list(a.shape[:-1])
    H_out = outputs[0].shape[-1]  # noqa: N806
    return outputs[0][:M].reshape(*lead, H_out)


def register():
    register_dynamic_nki(
        "cat",
        DynamicNkiEntry(
            compile_fn=_compile_cat,
            can_handle=_can_handle_cat,
            prepare_io=_prepare_cat_io,
            reshape_output=_reshape_cat_output,
            compile_key_fn=_cat_compile_key,
        ),
    )
