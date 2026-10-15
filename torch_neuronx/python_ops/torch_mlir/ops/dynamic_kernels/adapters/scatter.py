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
    _index_to_int32,
    _num_tiles_tensor,
    _pad_or_contig,
    _pad_rows_to,
)
from ..registry import DynamicNkiEntry, register_dynamic_nki


def _is_combine_pattern(T, n):  # noqa: N803
    """MoE combine: many-to-one, n = T*top_k rows summed into T token rows
    (top_k >= 2). Distinct from the unique-target n <= T scatter."""
    return T > 0 and n > T and n % T == 0 and (n // T) >= 2


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
    if n == 0 or index.shape[0] != n:
        return False
    # (A) MoE combine (many-to-one) -> gather-reduce; (B) unique-target n<=T.
    if _is_combine_pattern(T, n):
        return True
    if _ceil_to_tile(n) > T:  # noqa: SIM103
        return False
    return True


def _scatter_add_compile_key(self, dim, index, src, out=None, **_):
    """Unique path: (T, H, dtype). Combine path: ("combine", top_k, H, dtype) —
    T/N are runtime-dynamic (loop count), so one NEFF serves any routing."""
    T = int(self.shape[0])  # noqa: N806
    n = int(src.shape[0])
    if _is_combine_pattern(T, n):
        # T baked: the kernel's output [T,D] and loop APs are traced at compile
        # time, so the NEFF must match the runtime T. T is a static model dim
        # (tokens), so this is still one NEFF per model in practice.
        return ("combine", T, n // T, int(self.shape[1]), self.dtype)
    return (int(self.shape[0]), int(self.shape[1]), self.dtype)


def _compile_scatter_add(compile_key) -> tuple[bytes, str]:
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    # Combine path: ("combine", T, top_k, H, dtype) -> gather-reduce kernel.
    if isinstance(compile_key, tuple) and len(compile_key) == 5 and compile_key[0] == "combine":
        from torch_neuronx.utils import get_logical_neuron_cores

        from ..wrappers import dynamic_combine_gather_reduce_nki

        _, T, top_k, H, dtype = compile_key  # noqa: N806
        # LNC row-tile sharding: pad T so every program runs equal token tiles.
        P = int(get_logical_neuron_cores())  # noqa: N806
        row_group = _TILE_M * P
        T_ceil = -(-T // row_group) * row_group  # noqa: N806
        routed = torch.empty(T * top_k, H, dtype=dtype, device="meta")
        inv_idx = torch.zeros(T_ceil, top_k, dtype=torch.int32, device="meta")
        outbuf = torch.empty(T_ceil, H, dtype=dtype, device="meta")
        nt = torch.tensor([[T_ceil // row_group]], dtype=torch.int32, device="meta")
        kernel = TorchMlirKernel(
            torch_fn=dynamic_combine_gather_reduce_nki,
            op_name="dynamic_nki_combine_gather_reduce",
        )
        # Aliased in-place kernel (out_buffer is operand#2, alias {2:0}): NEFF output
        # set = declared output0 (fresh) + aliased out_buffer. Pass both.
        fresh = torch.empty(T_ceil, H, dtype=dtype, device="meta")
        return _compile_or_load_neff(
            kernel,
            "dynamic_nki_combine_gather_reduce",
            (routed, inv_idx, outbuf, nt),
            (fresh, outbuf),
        )

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


def _prepare_scatter_add_io(self, dim, index, src, dest=None, in_place=False, **_):
    """Combine path (n = T*top_k): build inv_idx[T, top_k] (source rows per token)
    and gather-reduce. in_place (scatter_add_) writes into self; out-of-place
    (scatter_add, a torch custom-op) accumulates into a fresh clone (no-alias rule).
    Unique path (n <= T): pad-to-T in-place scatter."""
    T, H = self.shape  # noqa: N806
    device = self.device
    n = src.shape[0]

    if _is_combine_pattern(T, n):
        from torch_neuronx.utils import get_logical_neuron_cores

        top_k = n // T
        token_ids = index[:, 0].reshape(n).to(torch.int32)
        # inv_idx[t] = the source rows feeding token t. Stable argsort groups a
        # token's rows contiguously; this re-derives token_indices_experts_sorted_N
        # (already computed upstream) — reuse it if the metadata is ever plumbed here.
        # TODO: assumes every token appears exactly top_k times (dropless, EP=1,
        # non-SP). Token dropping / SP break the [T, top_k] reshape — gate or
        # generalize to a segment-sum before enabling those layouts.
        order = torch.argsort(token_ids, stable=True)
        inv_idx = order.reshape(T, top_k).to(torch.int32)
        # LNC row-tile sharding: pad token rows to row_group so every program runs
        # equal tiles. Padded inv_idx rows -> 0 (gather row 0, harmless; sliced off).
        P = int(get_logical_neuron_cores())  # noqa: N806
        row_group = _TILE_M * P
        T_ceil = -(-T // row_group) * row_group  # noqa: N806
        inv_idx = _pad_rows_to(inv_idx, T_ceil)
        num_t_tiles = _num_tiles_tensor(T_ceil // row_group, device)
        # The kernel does out_buffer += gather_reduce.
        #  - in_place scatter_add_: out_buffer IS self (written back in place) when
        #    unpadded (T is a row_group multiple, true for the MoE token count).
        #  - out-of-place scatter_add: out_buffer is a fresh clone so the result
        #    does not alias self (torch custom-op aliasing constraint).
        outbuf = self if in_place and T_ceil == T else _pad_rows_to(self.clone(), T_ceil)
        fresh = torch.empty(T_ceil, H, dtype=self.dtype, device=device)
        return (src.contiguous(), inv_idx, outbuf, num_t_tiles), (fresh, outbuf), n

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
    """Return the aliased in-place target (outputs[1]). The combine path pads it
    to a row-group multiple (LNC sharding); slice back to T = self.shape[0]."""
    result = outputs[1]
    T = original_args[0].shape[0]  # noqa: N806
    return result[:T] if result.shape[0] != T else result


# ── index_put: self[idx] += values (dim 0), same kernel as scatter_add ──
# aten::_index_put_impl_(self, indices: Tensor?[], values, accumulate, unsafe). Handles the
# MoE pattern only: single 1-D dim-0 index, accumulate=True, unique targets (see UNIQUE note
# above). Reuses _compile_scatter_add; only the arg marshaling differs.
def _index_put_index_1d(indices):
    """The single 1-D dim-0 index tensor from an index_put `indices` list, else None."""
    if not isinstance(indices, list | tuple) or len(indices) != 1:
        return None
    idx = indices[0]
    if not isinstance(idx, torch.Tensor) or idx.dim() != 1:
        return None
    if idx.dtype not in (torch.int32, torch.int64):
        return None
    return idx


def _can_handle_index_put(self, indices, values, accumulate=False, unsafe=False, **_) -> bool:
    if not accumulate:  # non-accumulate index_put is overwrite, not scatter-add
        return False
    if not isinstance(self, torch.Tensor) or not isinstance(values, torch.Tensor):
        return False
    if self.device.type != "neuron" or self.dtype not in _SUPPORTED_DTYPES:
        return False
    if values.dtype != self.dtype:
        return False
    if self.dim() != 2 or values.dim() != 2:
        return False
    idx = _index_put_index_1d(indices)
    if idx is None:
        return False
    T, H = self.shape  # noqa: N806
    if H == 0:
        return False
    n = values.shape[0]
    if n == 0 or idx.numel() != n:
        return False
    # (A) MoE combine (many-to-one, n = T*top_k > T) -> gather-reduce; the autograd
    # backward of a row gather (x[idx]) arrives here as index_put(accumulate). (B) the
    # unique-target n<=T scatter. Reject only shapes that are neither.
    if _is_combine_pattern(T, n):
        return True
    if _ceil_to_tile(n) > T:  # noqa: SIM103
        return False
    return True


def _index_put_compile_key(self, indices, values, accumulate=False, unsafe=False, **_):
    """compile_key = (T, H, dtype) — same NEFF as scatter_add."""
    T = int(self.shape[0])  # noqa: N806
    n = int(values.shape[0])
    if _is_combine_pattern(T, n):
        return ("combine", T, n // T, int(self.shape[1]), self.dtype)
    return (int(self.shape[0]), int(self.shape[1]), self.dtype)


def _prepare_index_put_io(self, indices, values, accumulate=False, unsafe=False, dest=None, **_):
    """Marshal index_put args to the scatter_add kernel I/O; self is the in-place target."""
    T, H = self.shape  # noqa: N806
    device = self.device
    n = values.shape[0]

    idx_1d = _index_to_int32(_index_put_index_1d(indices)).reshape(n, 1)
    # Combine pattern (n = T*top_k > T, e.g. row-gather backward): reuse the scatter_add
    # combine prep (inv_idx[T, top_k] gather-reduce) so it hits the fast combine kernel
    # instead of a serialized scatter. dim=0, index shaped [n,1] like scatter_add expects.
    if _is_combine_pattern(T, n):
        return _prepare_scatter_add_io(self, 0, idx_1d, values, in_place=True)
    idx = _pad_or_contig(idx_1d, T)  # contiguous even when n == T (strided int32 alias otherwise)
    data = _pad_rows_to(values.contiguous(), T)
    nm = _num_tiles_tensor(_ceil_to_tile(n) // _TILE_M, device)

    outbuf = self  # in-place target (alias {2:0})
    fresh = torch.empty(T, H, dtype=self.dtype, device=device)
    return (data, idx, outbuf, nm), (fresh, outbuf), n


def register():
    # scatter_add_ is in-place (writes self); scatter_add is out-of-place (must
    # not alias self). Same kernel/keys; the combine prepare_io differs on in_place.
    """Register dynamic scatter-add and accumulating index-put adapters."""
    import functools

    for _sa_name, _in_place in (("scatter_add_", True), ("scatter_add", False)):
        register_dynamic_nki(
            _sa_name,
            DynamicNkiEntry(
                compile_fn=_compile_scatter_add,
                can_handle=_can_handle_scatter_add,
                prepare_io=functools.partial(_prepare_scatter_add_io, in_place=_in_place),
                reshape_output=_reshape_scatter_add_output,
                compile_key_fn=_scatter_add_compile_key,
            ),
        )
    register_dynamic_nki(
        "_index_put_impl_",
        DynamicNkiEntry(
            compile_fn=_compile_scatter_add,
            can_handle=_can_handle_index_put,
            prepare_io=_prepare_index_put_io,
            reshape_output=_reshape_scatter_add_output,
            compile_key_fn=_index_put_compile_key,
        ),
    )
