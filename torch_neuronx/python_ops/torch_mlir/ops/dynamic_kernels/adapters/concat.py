"""``aten::cat`` dynamic NKI adapter — two shapes, one registry entry.

Free dim (2 inputs): [*lead, H1] , [*lead, H2] -> [*lead, H1+H2] on one NEFF (H1 + H2 +
dtype baked, M dynamic) — the MoE backward of ``.chunk(2, dim=-1)``.
Row dim, one row per source (K inputs): one fused NEFF that reads all K sources
(``cat_rows_fused_kernel``) — the DTensor reduce-scatter ``cat`` of 4 x [1, 1024, 2048].
Row dim, any other row counts (K inputs): dim-0 concat as K dynamic ``copy_`` calls, no
NEFF of its own (``_run_cat_rows``) — TorchTitan's MoE for-loop
``cat([chunk_e @ w_e ...], dim=0)``, whose per-expert row counts move every step.
Anything else (other dims, mixed dtypes, strided inputs) declines to the default lowering.
"""

import torch

from ..adapter_utils import (
    _SUPPORTED_DTYPES,
    _TILE_M,
    _ceil_to_tile,
    _compile_or_load_neff,
    _dest_buffer,
    _dest_reusable,
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


def _is_row_dim(dim, ndim) -> bool:
    return dim in (0, -ndim)


def _cat_rows_width(tensors, dim) -> int | None:
    """Row width H (trailing dims flattened) if ``tensors`` is a dim-0 concat of
    same-width contiguous neuron rows, else None. Contiguity is required: a strided input
    would take the copy kernel's strided variant, whose NEFF bakes the view's stride."""
    if not isinstance(tensors, list | tuple) or len(tensors) < 2:
        return None
    first = tensors[0]
    if not isinstance(first, torch.Tensor) or first.dim() < 2:
        return None
    if not _is_row_dim(dim, first.dim()) or first.device.type != "neuron":
        return None
    if first.dtype not in _SUPPORTED_DTYPES:
        return None
    rest = tuple(first.shape[1:])
    H = 1  # noqa: N806
    for d in rest:
        H *= int(d)  # noqa: N806
    if H == 0:
        return None
    for t in tensors:
        if not isinstance(t, torch.Tensor) or t.dtype != first.dtype:
            return None
        if t.device != first.device or t.dim() != first.dim():
            return None
        if tuple(t.shape[1:]) != rest or not t.is_contiguous():
            return None
    return H


# Arity cap for the fused NEFF: the generated kernel emits one DMA per source.
_MAX_FUSED_K = 128


def _cat_rows_fused_plan(tensors, dim) -> tuple[int, int] | None:
    """(K, H) when a dim-0 concat gives exactly one row per source, else None.

    One row per source is what lets the fused kernel keep every DMA offset static, so no
    SBUF stage and no free-dim segmentation are needed (see ``cat_rows_fused_kernel``).
    K and H are shape facts already in the compile key, so a fresh routing adds no NEFF.
    """
    H = _cat_rows_width(tensors, dim)  # noqa: N806
    if H is None or len(tensors) > _MAX_FUSED_K:
        return None
    if any(int(t.shape[0]) != 1 for t in tensors):
        return None
    return len(tensors), H


def _fused_takes_it(tensors, dim, dest) -> bool:
    """True when the fused NEFF both fits the shapes and can write ``dest`` in place. A
    dest it cannot alias would cost dispatch's stride-keyed copy-back, so leave that one
    to the composite path, which writes any destination directly."""
    plan = _cat_rows_fused_plan(tensors, dim)
    if plan is None:
        return False
    if dest is None:
        return True
    first = tensors[0]
    return _dest_reusable(dest, plan[0], plan[1], first.dtype, first.device)


def _can_handle_cat_rows(tensors, dim=0, out=None) -> bool:
    """Take the dim-0 concat only when EVERY row copy has a dynamic kernel: one that
    declines would compile per row count, i.e. the recompile this path exists to remove."""
    from .cast import _can_handle_copy

    H = _cat_rows_width(tensors, dim)  # noqa: N806
    if H is None:
        return False
    M = 0  # noqa: N806
    for t in tensors:
        n = int(t.shape[0])
        if n == 0:
            continue  # no rows, no copy
        M += n  # noqa: N806
        if not _can_handle_copy(t, t):
            return False
    if M == 0:
        return False
    first = tensors[0]
    return out is None or _dest_reusable(out, M, H, first.dtype, first.device)


def _run_cat_rows(tensors, dim=0, out=None, dest=None, **_):
    """dim-0 concat as K dynamic row copies into one buffer; None = not this shape.

    Compile-free at any K and any counts: ``torch.empty`` of a fresh row count is a pure
    allocation, and ``copy_``'s key is (H, dtype) — neither the rows nor the dst offset."""
    H = _cat_rows_width(tensors, dim)  # noqa: N806
    if H is None:
        return None
    if _fused_takes_it(tensors, dim, dest):
        return None  # the fused NEFF below covers it
    first = tensors[0]
    rest = tuple(first.shape[1:])
    M = sum(int(t.shape[0]) for t in tensors)  # noqa: N806
    buf = _dest_buffer(dest, M, H, first.dtype, first.device)
    off = 0
    for t in tensors:
        n = int(t.shape[0])
        if n == 0:
            continue
        buf[off : off + n].copy_(t.reshape(n, H))
        off += n
    return buf.reshape(M, *rest)


def _can_handle_cat(tensors, dim=0, out=None) -> bool:
    if _fused_takes_it(tensors, dim, out) or _can_handle_cat_rows(tensors, dim, out):
        return True
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
    if M == 0 or H1 == 0 or H2 == 0:
        return False
    # prepare_io aliases an out= only when no row padding is needed; one it can't alias
    # would cost a stride-keyed copy_ back, so decline that to the static lowering.
    return out is None or (
        _ceil_to_tile(M) == M and _dest_reusable(out, M, H1 + H2, a.dtype, a.device)
    )


def _cat_compile_key(tensors, dim=0, out=None, **_):
    """Fused rows: ("rows_fused", K, H, dtype). Free dim: (H1, H2, dtype) — both widths +
    dtype baked, M dynamic. ``out`` never reaches here, so the fused plan sees no dest;
    that is sound because dispatch only compiles after ``run_fn`` declined."""
    plan = _cat_rows_fused_plan(tensors, dim)
    if plan is not None:
        return ("rows_fused", plan[0], plan[1], tensors[0].dtype)
    a, b = tensors[0], tensors[1]
    return (int(a.shape[-1]), int(b.shape[-1]), a.dtype)


def _compile_cat(compile_key) -> tuple[bytes, str]:
    from torch_neuronx.python_ops.torch_mlir.kernel import TorchMlirKernel

    from ..wrappers import dynamic_cat_nki, dynamic_cat_rows_fused_nki

    if compile_key[0] == "rows_fused":
        _, k, H, dtype = compile_key  # noqa: N806
        srcs = [torch.empty(1, H, dtype=dtype, device="meta") for _ in range(k)]
        out = torch.empty(k, H, dtype=dtype, device="meta")
        graph_name = f"dynamic_nki_cat_rows_fused_k{k}_h{H}"
        kernel = TorchMlirKernel(torch_fn=dynamic_cat_rows_fused_nki, op_name=graph_name)
        return _compile_or_load_neff(kernel, graph_name, (srcs,), out)

    H1, H2, dtype = compile_key  # noqa: N806
    a = torch.empty(_TILE_M, H1, dtype=dtype, device="meta")
    b = torch.empty(_TILE_M, H2, dtype=dtype, device="meta")
    nm = torch.tensor([[1]], dtype=torch.int32, device="meta")
    out = torch.empty(_TILE_M, H1 + H2, dtype=dtype, device="meta")

    kernel = TorchMlirKernel(torch_fn=dynamic_cat_nki, op_name="dynamic_nki_cat")
    return _compile_or_load_neff(kernel, "dynamic_nki_cat", (a, b, nm), out)


def _prepare_cat_io(tensors, dim=0, out=None, dest=None, **_):
    """Fused rows: the K [1, H] source views and one [K, H] result. Free dim: flatten each
    operand's leading dims to [M, H], pad M to a 128-multiple; kernel writes the
    concatenated [M_ceil, H1+H2]."""
    plan = _cat_rows_fused_plan(tensors, dim)
    if plan is not None:
        k, H = plan  # noqa: N806
        first = tensors[0]
        srcs = tuple(t.reshape(1, H) for t in tensors)
        return srcs, (_dest_buffer(dest, k, H, first.dtype, first.device),), k

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
    """Slice off padded rows, restore leading dims: [*lead, H1+H2].

    ``original_args`` carries no kwargs, so ``dim`` may be absent. The output SHAPE tells
    the two paths apart instead: only the fused path makes a [K, H] buffer, and the free
    dim path's row count is always a 128-multiple, which K <= 2 operands cannot be.
    """
    tensors = original_args[0]
    plan = _cat_rows_fused_plan(tensors, 0)
    if plan is not None and tuple(outputs[0].shape) == plan:
        return outputs[0].reshape(plan[0], *tensors[0].shape[1:])
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
            # dim-0 concat first (no NEFF of its own); returns None for the free-dim
            # shapes, which then take the compile/execute path above.
            run_fn=_run_cat_rows,
        ),
    )
