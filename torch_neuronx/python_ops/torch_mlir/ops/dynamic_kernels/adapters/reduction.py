"""Free-dim sum dynamic NKI adapter.

Covers ``aten::sum`` over the last dim of a 2-D+ tensor. Leading dims flatten to
one dynamic row count M (like the elementwise adapters), so [*lead, H] reduces to
[*lead, 1] (keepdim) or [*lead] (no keepdim). M is dynamic (padded to a
128-multiple); H + dtype are baked into the NEFF.
"""

import torch

from ..adapter_utils import (
    _TILE_M,
    _can_handle_elementwise_2d,
    _ceil_to_tile,
    _compile_or_load_neff,
    _hidden_compile_key,
    _num_tiles_tensor,
    _pad_or_contig,
)
from ..registry import DynamicNkiEntry, register_dynamic_nki


def _reduces_last_dim_only(self, dim) -> bool:
    """True if `dim` selects exactly the last (free) dim of `self`."""
    dims = dim if isinstance(dim, list | tuple) else [dim]
    if len(dims) != 1:
        return False
    d = dims[0]
    last = self.dim() - 1
    return d in (last, -1)


def _can_handle_sum(self, dim=None, keepdim=False, dtype=None, out=None) -> bool:
    if dtype is not None:  # the kernel outputs the input dtype; a dtype cast would recompile
        return False
    if dim is None or not _reduces_last_dim_only(self, dim):
        return False
    # Reducing the last dim of [*lead, H] == reducing [prod(lead), H] then reshaping
    # back — so any rank >= 2 works via the leading-dim flatten in prepare_io.
    return _can_handle_elementwise_2d(self)


def _sum_compile_key(self, dim=None, keepdim=False, dtype=None, out=None, **_):
    """compile_key = (H, dtype) — the reduced width + dtype are baked; M dynamic."""
    return _hidden_compile_key(self)


def _compile_sum(compile_key) -> tuple[bytes, str]:
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    from ..wrappers import dynamic_sum_nki

    H, dtype = compile_key  # noqa: N806
    x = torch.empty(_TILE_M, H, dtype=dtype, device="meta")
    nm = torch.tensor([[1]], dtype=torch.int32, device="meta")
    out = torch.empty(_TILE_M, 1, dtype=dtype, device="meta")

    kernel = TorchMlirKernel(torch_fn=dynamic_sum_nki, op_name="dynamic_nki_sum")
    return _compile_or_load_neff(kernel, "dynamic_nki_sum", (x, nm), out)


def _prepare_sum_io(self, dim=None, keepdim=False, dtype=None, dest=None, **_):
    """Flatten leading dims to [M, H], pad M to a 128-multiple; kernel reduces the
    free dim to [M_ceil, 1]."""
    H = self.shape[-1]  # noqa: N806
    M = self.numel() // H  # noqa: N806
    M_ceil = _ceil_to_tile(M)  # noqa: N806
    device = self.device

    x = _pad_or_contig(self.reshape(M, H), M_ceil)
    nm = _num_tiles_tensor(M_ceil // _TILE_M, device)
    outbuf = torch.empty(M_ceil, 1, dtype=self.dtype, device=device)
    return (x, nm), (outbuf,), M


def _reshape_sum_output(outputs, original_args):
    """Slice off padded rows, restore leading dims: [*lead, 1] (keepdim) else [*lead]."""
    self = original_args[0]
    H = self.shape[-1]  # noqa: N806
    M = self.numel() // H  # noqa: N806
    keepdim = original_args[2] if len(original_args) > 2 else False
    reduced = outputs[0][:M]  # [M, 1]
    lead = list(self.shape[:-1])
    return reduced.reshape(*lead, 1) if keepdim else reduced.reshape(*lead)


def register():
    register_dynamic_nki(
        "sum",
        DynamicNkiEntry(
            compile_fn=_compile_sum,
            can_handle=_can_handle_sum,
            prepare_io=_prepare_sum_io,
            reshape_output=_reshape_sum_output,
            compile_key_fn=_sum_compile_key,
        ),
    )
