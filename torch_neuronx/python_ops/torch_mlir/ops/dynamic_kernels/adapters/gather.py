"""Row-gather dynamic NKI adapters: ``x[idx]`` (aten::index) and ``x.gather(0, idx)``.

Both compute out[i, :] = self[idx[i], :] — n (gathered rows) dynamic; T (source
rows) + H + dtype baked into the NEFF. Index padded to the baked T (zero-fill =>
in-bounds gather of row 0, sliced off). aten::gather qualifies only when its index
is constant along the free dim, which is how the backward of a row scatter_add_
arrives.
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


def _index_row_index(self, indices, **_):
    """Row index of ``x[idx]`` for the MoE row-gather pattern (a single 1-D int
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


def _gather_row_index(self, dim, index, sparse_grad=False, **_):
    """Row index of ``gather(0, idx2d)`` when idx2d is constant along the free dim
    (stride 0, full width), else None.

    That is a row gather in disguise, and it is how the backward of
    ``scatter_add_(0, idx.unsqueeze(1).expand_as(src), src)`` arrives — the MoE
    expert-combine step. Column 0 is the row index; the expanded width is never read.
    """
    if dim != 0 or sparse_grad:
        return None
    if not isinstance(index, torch.Tensor) or index.dim() != 2:
        return None
    if index.dtype not in (torch.int32, torch.int64):
        return None
    if index.stride(1) != 0 or index.shape[1] != self.shape[1]:
        return None
    return index[:, 0]


def _row_gather_compile_key(self, *_args, **_kwargs):
    """compile_key = (T, H, dtype) of the source. T is baked only under the
    current pad-to-T scheme — it must leave the key if we adopt register-count
    APs (then T becomes runtime-dynamic like n)."""
    return (int(self.shape[0]), int(self.shape[1]), self.dtype)


def _compile_row_gather(compile_key) -> tuple[bytes, str]:
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    from ..wrappers import dynamic_gather_nki

    T, H, dtype = compile_key  # noqa: N806
    source = torch.empty(T, H, dtype=dtype, device="meta")
    idx = torch.zeros(T, 1, dtype=torch.int32, device="meta")  # valid (0) for cold gather
    nm = torch.tensor([[1]], dtype=torch.int32, device="meta")
    out = torch.empty(T, H, dtype=dtype, device="meta")

    kernel = TorchMlirKernel(torch_fn=dynamic_gather_nki, op_name="dynamic_nki_gather")
    return _compile_or_load_neff(kernel, "dynamic_nki_gather", (source, idx, nm), out)


def _row_gather_io(self, idx_1d) -> tuple[tuple, tuple, int]:
    """Marshal (source, 1-D row index) -> gather kernel.

    T (source rows) + H are baked, so the index is padded to the baked size
    [T, 1] (NOT n): first n real, rest ZERO (row 0 is a valid source row =>
    in-bounds gather, sliced off). num_m_tiles = ceil(n / 128) walks whole tiles;
    rows n..ceil(n) read the zero-padded index (row 0) and are sliced off too.
    """
    T, H = self.shape  # noqa: N806
    n = idx_1d.numel()
    idx = _pad_rows_to(idx_1d.reshape(n, 1).to(torch.int32), T)
    nm = _num_tiles_tensor(_ceil_to_tile(n) // _TILE_M, self.device)
    inputs = (self.contiguous(), idx, nm)
    outputs = (torch.empty((T, H), dtype=self.dtype, device=self.device),)
    return inputs, outputs, n


def _row_gather_entry(row_index_of) -> DynamicNkiEntry:
    """Entry for one row-gather op. ``row_index_of(*args, **kwargs)`` returns that
    op's 1-D row index, or None when its arguments aren't a row gather.

    One entry per op name (so each has its own memo), but the compile_key and the
    kernel are shared: whichever op compiles a (T, H, dtype) first, the other loads
    that same NEFF instead of compiling a second one.
    """

    def can_handle(self, *args, **kwargs) -> bool:
        """Determine whether arguments describe a supported row gather.

        Args:
            self (torch.Tensor): Source tensor containing rows to gather.
            *args (Any): Positional arguments for the gather operation.
            **kwargs (Any): Keyword arguments for the gather operation.

        Returns:
            bool: ``True`` when the row-gather kernel can serve the operation.
        """
        if not isinstance(self, torch.Tensor) or self.device.type != "neuron":
            return False
        if self.dtype not in _SUPPORTED_DTYPES or self.dim() != 2:
            return False
        if self.shape[1] == 0:  # H is arbitrary — kernel tiles the free dim at trace time
            return False
        idx = row_index_of(self, *args, **kwargs)
        if idx is None or idx.device != self.device:
            return False
        # n rows gathered as ceil(n/128) whole 128-tiles; index padded to T, so the
        # padded count ceil(n/128)*128 must still fit within T source rows.
        n = idx.numel()
        return n > 0 and _ceil_to_tile(n) <= self.shape[0]

    def prepare_io(self, *args, dest=None, **kwargs) -> tuple[tuple, tuple, int]:
        """Marshal a supported row gather for kernel execution.

        Args:
            self (torch.Tensor): Source tensor containing rows to gather.
            *args (Any): Positional arguments used to locate the row index.
            dest (torch.Tensor, optional): Unused destination buffer. Defaults to ``None``.
            **kwargs (Any): Keyword arguments used to locate the row index.

        Returns:
            tuple: Kernel inputs, output buffers, and gathered row count.
        """
        return _row_gather_io(self, row_index_of(self, *args, **kwargs))

    def reshape_output(outputs, original_args):
        """Slice the kernel output to the gathered row count.

        Args:
            outputs (tuple): Kernel output buffers with source-row capacity.
            original_args (tuple): Original operation arguments.

        Returns:
            torch.Tensor: The gathered rows with padded capacity removed.
        """
        return outputs[0][: row_index_of(*original_args).numel()]

    return DynamicNkiEntry(
        compile_fn=_compile_row_gather,
        can_handle=can_handle,
        prepare_io=prepare_io,
        reshape_output=reshape_output,
        compile_key_fn=_row_gather_compile_key,
    )


def register():
    """Register index and gather row-gather adapters."""
    register_dynamic_nki("index", _row_gather_entry(_index_row_index))
    register_dynamic_nki("gather", _row_gather_entry(_gather_row_index))
