"""index dynamic NKI adapter  (x[idx] row-gather: self[T,H], idx 1-D -> [n,H]).

n (num gathered rows) dynamic; T (source rows) + H + dtype baked into NEFF.
Index padded to the baked T (zero-fill => in-bounds gather of row 0, sliced off).
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


def _index_single_1d(indices):
    """Return the index tensor for the MoE row-gather pattern (single 1-D int
    index on dim 0), else None."""
    if not isinstance(indices, list | tuple) or len(indices) == 0:
        return None
    present = [t for t in indices if t is not None]
    if len(present) != 1:
        return None
    idx = present[0]
    if indices[0] is None:  # leading None => indexing a later dim
        return None
    if not isinstance(idx, torch.Tensor) or idx.dim() != 1:
        return None
    if idx.dtype not in (torch.int32, torch.int64):
        return None
    return idx


def _can_handle_index(self, indices, out=None) -> bool:
    if not isinstance(self, torch.Tensor) or self.device.type != "neuron":
        return False
    if self.dtype not in _SUPPORTED_DTYPES or self.dim() != 2:
        return False
    T, H = self.shape  # noqa: N806
    if H == 0:  # H is arbitrary — kernel tiles the free dim at trace time
        return False
    idx = _index_single_1d(indices)
    if idx is None:
        return False
    # n rows gathered as ceil(n/128) whole 128-tiles; index padded to T, so the
    # padded count ceil(n/128)*128 must still fit within T source rows.
    n = idx.numel()
    if n == 0 or _ceil_to_tile(n) > T:  # noqa: SIM103
        return False
    return True


def _index_compile_key(self, indices, out=None, **_):
    """compile_key = (T, H, dtype) of the source. T is baked only under the
    current pad-to-T scheme — it must leave the key if we adopt register-count
    APs (then T becomes runtime-dynamic like n)."""
    return (int(self.shape[0]), int(self.shape[1]), self.dtype)


def _compile_index(compile_key) -> tuple[bytes, str]:
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    from ..wrappers import dynamic_gather_nki

    T, H, dtype = compile_key  # noqa: N806
    source = torch.empty(T, H, dtype=dtype, device="meta")
    idx = torch.zeros(T, 1, dtype=torch.int32, device="meta")  # valid (0) for cold gather
    nm = torch.tensor([[1]], dtype=torch.int32, device="meta")
    out = torch.empty(T, H, dtype=dtype, device="meta")

    kernel = TorchMlirKernel(torch_fn=dynamic_gather_nki, op_name="dynamic_nki_gather")
    return _compile_or_load_neff(kernel, "dynamic_nki_gather", (source, idx, nm), out)


def _prepare_index_io(self, indices, dest=None, **_) -> tuple[tuple, tuple, int]:
    """Marshal x[idx] -> gather kernel.

    T (source rows) + H are baked, so the index is padded to the baked size
    [T, 1] (NOT n): first n real, rest ZERO (row 0 is a valid source row =>
    in-bounds gather, sliced off). num_m_tiles = ceil(n / 128) walks whole tiles;
    rows n..ceil(n) read the zero-padded index (row 0) and are sliced off too.
    """
    T, H = self.shape  # noqa: N806
    device = self.device
    idx_1d = _index_single_1d(indices)
    n = idx_1d.numel()
    idx = idx_1d.reshape(n, 1).to(torch.int32)
    idx = _pad_rows_to(idx, T)  # pad to baked source size T (zero-fill)
    nm = _num_tiles_tensor(_ceil_to_tile(n) // _TILE_M, device)
    inputs = (self.contiguous(), idx, nm)
    outputs = (torch.empty((T, H), dtype=self.dtype, device=device),)
    return inputs, outputs, n


def _reshape_index_output(outputs, original_args):
    """x[idx] output is [T, H]; slice to n valid rows."""
    self, indices = original_args[0], original_args[1]
    H = self.shape[1]  # noqa: N806
    n = _index_single_1d(indices).numel()
    return outputs[0][:n].reshape(n, H)


def register():
    register_dynamic_nki(
        "index",
        DynamicNkiEntry(
            compile_fn=_compile_index,
            can_handle=_can_handle_index,
            prepare_io=_prepare_index_io,
            reshape_output=_reshape_index_output,
            compile_key_fn=_index_compile_key,
        ),
    )
