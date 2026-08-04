"""scatter_add / scatter_add_ dynamic NKI adapter.

In-place: self[T,H][idx[i]] += src[i], dim 0. n (scattered rows) dynamic;
T (target rows) + H + dtype baked. Data/index padded to T (padded data=0 is a
no-op add; padded idx=0 touches row 0 with a 0 add).

NOTE: correct only for UNIQUE target indices (MoE token->row is unique) — the
kernel's load-add-writeback doesn't accumulate duplicate targets in a tile, and
can_handle can't cheaply verify uniqueness. MoE-scoped assumption.
"""

import torch

from ..adapter_utils import (
    _SUPPORTED_DTYPES,
    _TILE_M,
    _ceil_to_tile,
    _compile_or_load_neff,
    _num_tiles_tensor,
    _pad_rows_to,
)
from ..registry import DynamicNkiEntry, register_dynamic_nki


def _can_handle_scatter_add(self, dim, index, src, out=None) -> bool:
    if not all(isinstance(t, torch.Tensor) for t in (self, index, src)):
        return False
    if self.device.type != "neuron" or self.dtype not in _SUPPORTED_DTYPES:
        return False
    if src.dtype != self.dtype:  # kernel adds src into self; dtypes must match
        return False
    if self.dim() != 2 or src.dim() != 2 or index.dim() != 2:
        return False
    if dim != 0:
        return False
    T, H = self.shape  # noqa: N806
    if H == 0:  # H is arbitrary — kernel tiles the free dim at trace time
        return False
    n = src.shape[0]
    # n rows scattered as ceil(n/128) whole 128-tiles; data/index padded to T, so
    # the padded count ceil(n/128)*128 must still fit within T target rows.
    if n == 0 or _ceil_to_tile(n) > T:
        return False
    # index is [n, H] expanded (stride 0 on dim1) or [n,1]; both give per-row idx.
    if index.shape[0] != n:  # noqa: SIM103
        return False
    return True


def _scatter_add_compile_key(self, dim, index, src, out=None, **_):
    """compile_key = (T, H, dtype) of the target. T is baked only under the
    current pad-to-T scheme — it must leave the key if we adopt register-count
    APs (then T becomes runtime-dynamic like n)."""
    return (int(self.shape[0]), int(self.shape[1]), self.dtype)


def _compile_scatter_add(compile_key) -> tuple[bytes, str]:
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    from ..wrappers import dynamic_scatter_add_nki

    T, H, dtype = compile_key  # noqa: N806
    data = torch.empty(T, H, dtype=dtype, device="meta")
    idx = torch.zeros(T, 1, dtype=torch.int32, device="meta")
    outbuf = torch.empty(T, H, dtype=dtype, device="meta")
    nm = torch.tensor([[1]], dtype=torch.int32, device="meta")

    kernel = TorchMlirKernel(torch_fn=dynamic_scatter_add_nki, op_name="dynamic_nki_scatter_add")
    # Aliased kernel: return_types=[<dtype>[T,H]] + operand_output_aliases={2:0}, so
    # the NEFF output set = declared output0 (fresh) + aliased input#2 (outbuf)
    # appended. Passing only (outbuf,) => runtime can't find "output1"; pass both.
    fresh = torch.empty(T, H, dtype=dtype, device="meta")
    return _compile_or_load_neff(
        kernel, "dynamic_nki_scatter_add", (data, idx, outbuf, nm), (fresh, outbuf)
    )


def _prepare_scatter_add_io(self, dim, index, src, dest=None, **_) -> tuple[tuple, tuple, int]:
    """All tensors padded to baked T; padded data=0 is a no-op add; padded
    indices=0 touches row 0 with 0 add (harmless). num_m_tiles = ceil(n / 128);
    rows n..ceil(n) fall in the zero-padded region -> no-op add too. outbuf IS
    self (in-place)."""
    T, H = self.shape  # noqa: N806
    device = self.device
    n = src.shape[0]

    idx_1d = index[:, 0].reshape(n, 1).to(torch.int32)
    idx = _pad_rows_to(idx_1d, T)
    data = _pad_rows_to(src.contiguous(), T)
    nm = _num_tiles_tensor(_ceil_to_tile(n) // _TILE_M, device)

    outbuf = self  # in-place accumulation target (alias {2:0})
    # Output set must match NEFF's 2-ref layout: declared output0 + aliased input.
    fresh = torch.empty(T, H, dtype=self.dtype, device=device)
    inputs = (data, idx, outbuf, nm)
    outputs = (fresh, outbuf)
    return inputs, outputs, n


def _reshape_scatter_add_output(outputs, original_args):
    """Return outputs[1] (the aliased in-place target `self`)."""
    return outputs[1]


def register():
    for _sa_name in ("scatter_add", "scatter_add_"):
        register_dynamic_nki(
            _sa_name,
            DynamicNkiEntry(
                compile_fn=_compile_scatter_add,
                can_handle=_can_handle_scatter_add,
                prepare_io=_prepare_scatter_add_io,
                reshape_output=_reshape_scatter_add_output,
                compile_key_fn=_scatter_add_compile_key,
            ),
        )
