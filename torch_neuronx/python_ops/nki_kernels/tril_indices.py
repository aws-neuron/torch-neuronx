"""
NKI kernels that produce torch.tril_indices and torch.triu_indices on device, for any offset.

Both ops use one "guess + fence" kernel. For each linear output position in the
triangular ("trapezoid") region, a float32 sqrt estimates the row index and two
integer fence steps correct it. The result is exact even when the device sqrt
rounds low or high.

triu is the reverse of tril: read the upper triangle backwards, and its rows grow
by one element per row, like the lower triangle. The kernel therefore runs the
tril arithmetic on the reversed linear index and flips the row and column at the
end. This keeps the sqrt operand a sum, which float32 handles without the
cancellation a direct upper-triangle formula would suffer.

The offset only changes the geometry the host computes: the edge row width (hence
the fence constant k), the first trapezoid row, and the full-width rectangle size.
The on-chip arithmetic is the same for every offset.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from torch_neuronx.nki_hop import wrap_nki
from torch_neuronx.python_ops.nki_kernels.packed_interleave import packed_int64_view
from torch_neuronx.utils import get_logical_neuron_cores

# The kernel forms C(r) = r*(r + k)/2 in int32. The fence product and the output
# length must stay below 2^31. This bounds the products themselves, not row and col,
# so a skewed shape with few elements stays eligible.
INT32_MAX = 2**31 - 1

P_MAX = 128  # SBUF partition dimension
F_CHUNK = 512  # free-dim chunk per partition (>= 512 elements for DMA efficiency)


def _trilu_sizes(row: int, col: int, offset: int, upper: bool) -> tuple[int, int, int, int, int]:
    """Trapezoid/rectangle geometry of the triangle. Matches torch._refs.

    Returns (n_trap, n_rect, n_trap_rows, edge, first_trap_row). ``edge`` is the width
    of the trapezoid row the fence counts from: the first row for tril, the last row
    for triu. The kernel reads triu backwards, so its rows also grow by one.
    """
    if row <= 0 or col <= 0:
        return 0, 0, 0, 0, 0
    if upper:
        m_first = max(0, col - offset) if offset > 0 else col
        rect_rows = max(0, min(row, -offset))
        n_trap_rows = max(0, min(row, col - offset) - rect_rows)
        edge = m_first - n_trap_rows + 1 if n_trap_rows > 0 else 0
        n_trap = (m_first + edge) * n_trap_rows // 2
        return n_trap, rect_rows * col, n_trap_rows, edge, rect_rows
    m_first = min(col, 1 + offset) if offset > 0 else int(row + offset > 0)
    m_last = max(0, min(col, row + offset))
    n_trap_rows = m_last - m_first + 1
    n_trap = (m_first + m_last) * n_trap_rows // 2
    if n_trap <= 0:
        n_trap, n_trap_rows = 0, 0
    n_rect = max(0, (max(0, min(row, row + offset)) - n_trap_rows) * col)
    return n_trap, n_rect, n_trap_rows, m_first, max(0, -offset)


def _trilu_fence_product(n_trap_rows: int, edge: int) -> int:
    """Largest int32 product the fence forms: r*(r + k) at r = n_trap_rows."""
    return n_trap_rows * (n_trap_rows + 2 * edge - 1)


def trilu_numel(row: int, col: int, offset: int, upper: bool) -> int:
    """Return the number of triangular index pairs.

    Args:
        row (int): Matrix row count.
        col (int): Matrix column count.
        offset (int): Diagonal offset.
        upper (bool): Whether to count the upper rather than lower triangle.

    Returns:
        int: Number of index pairs in the result.
    """
    n_trap, n_rect, _, _, _ = _trilu_sizes(row, col, offset, upper)
    return n_trap + n_rect


def trilu_nki_eligible(row: int, col: int, offset: int, upper: bool) -> bool:
    """Check whether result size and fence arithmetic fit the NKI kernel limits.

    Args:
        row (int): Matrix row count.
        col (int): Matrix column count.
        offset (int): Diagonal offset.
        upper (bool): Whether to check the upper rather than lower triangle.

    Returns:
        bool: ``True`` for a nonempty result whose length and fence product do not
            exceed ``INT32_MAX``.

    Note:
        This check does not validate whether emitted row and column coordinates fit
        in int32.
    """
    n_trap, n_rect, n_trap_rows, edge, _ = _trilu_sizes(row, col, offset, upper)
    n_total = n_trap + n_rect
    if n_total <= 0:
        return False
    return _trilu_fence_product(n_trap_rows, edge) <= INT32_MAX and n_total <= INT32_MAX


def trilu_raise_oversize(name: str, row: int, col: int, offset: int):
    """Raise when triangular-index arithmetic exceeds the int32 limit.

    Args:
        name (str): Operation name used in the error message.
        row (int): Matrix row count.
        col (int): Matrix column count.
        offset (int): Diagonal offset.

    Raises:
        NotImplementedError: Always; compute the oversized result on CPU instead.
    """
    raise NotImplementedError(
        f"{name}({row}, {col}, {offset}) needs more than {INT32_MAX} index elements. "
        "The neuron kernel computes indices in int32. Compute on CPU and move the result."
    )


def _div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


def _program_start(n_items: int) -> int:
    """First of this SPMD program's contiguous share of n_items work items."""
    return min(nl.program_id(0) * _div_ceil(n_items, nl.num_programs(0)), n_items)


def _program_stop(n_items: int) -> int:
    """One past the last of this SPMD program's share of n_items work items."""
    return min(_program_start(n_items) + _div_ceil(n_items, nl.num_programs(0)), n_items)


def _emit_tri_cumsum(c_i32, r_i32, scratch_i32, k_i32=None):
    """Emit C(r) = r*(r + k) / 2 elementwise in exact int32.

    C(r) counts the trapezoid elements before row r when the first row has
    m0 elements and each later row has one more: k = 2*m0 - 1. k_i32 is None
    for tril (m0 = 1, so C(r) = r*(r+1)/2). r*(r + k) is always even, so the
    bitvec right-shift is an exact halving.
    """
    if k_i32 is None:
        nisa.tensor_tensor(dst=scratch_i32, data1=r_i32, data2=r_i32, op=nl.multiply)
        nisa.tensor_tensor(dst=c_i32, data1=scratch_i32, data2=r_i32, op=nl.add)
    else:
        nisa.tensor_tensor(dst=scratch_i32, data1=r_i32, data2=k_i32, op=nl.add)
        nisa.tensor_tensor(dst=c_i32, data1=scratch_i32, data2=r_i32, op=nl.multiply)
    nisa.tensor_scalar(dst=c_i32, data=c_i32, op0=nl.right_shift, operand0=1)


def _emit_const_i32(dst_i32, value: int):
    """Fill an int32 tile with an exact integer constant."""
    nisa.iota(dst=dst_i32, pattern=[[0, dst_i32.shape[1]]], offset=value, channel_multiplier=0)


def _store_plane(
    output, tile_i32, p0: int, p_rows: int, f_len: int, dst_offset: int, row_stride: int
):
    """Store tile_i32[p0:p0 + p_rows, 0:f_len] into output at logical element dst_offset.

    Rows of the tile land row_stride logical elements apart.
    """
    src = tile_i32[p0 : p0 + p_rows, 0:f_len]
    nisa.dma_copy(
        dst=output.ap(pattern=[[row_stride, p_rows], [1, f_len]], offset=dst_offset), src=src
    )


@nki.jit
def trilu_indices_fence_kernel(anchor, row, col, upper, offset):
    """Compute triangular indices with int32 arithmetic on a Neuron device.

    Args:
        anchor (tensor): Unread ``[1, 1]`` tensor carrying device and fake-tensor context.
        row (int): Positive compile-time row count whose emitted indices fit int32.
        col (int): Positive compile-time column count whose emitted indices fit int32.
        upper (int): ``0`` for lower-triangle indices or ``1`` for upper-triangle indices.
        offset (int): Compile-time diagonal offset controlling trapezoid geometry.

    Returns:
        tensor: Shared-HBM int32 indices shaped ``[2, N]`` in row-major order.

    Raises:
        AssertionError: If the anchor, dimensions, selector, or checked int32 limits are invalid.

    Note:
        Callers must ensure emitted coordinates fit int32. Offset sets edge, bias, first row, and
        rectangle quotient/remainder; upper reverses/flips; eligibility caps at ``2**31``.
        Hardware lacks native int32 compare, so fences subtract in int32 before float32 comparisons.
        Down/up checks correct ``floor(sqrt(b*b+2*x)-b)`` against ``C(r)=r*(r+2*edge-1)/2``.
    """
    # NKI accepts assert but not raise in kernel-traced code.
    assert tuple(anchor.shape) == (1, 1), f"anchor must be [1, 1], got {anchor.shape}"
    assert isinstance(row, int) and isinstance(col, int), "row and col must be Python ints"
    assert row >= 1 and col >= 1, f"row={row}, col={col} must be >= 1"
    assert upper in (0, 1), f"upper must be 0 or 1, got {upper!r}"
    assert trilu_nki_eligible(
        row, col, offset, bool(upper)
    ), f"the kernel does not cover row={row}, col={col}, offset={offset}, upper={upper}"

    n_trap, n_rect, n_trap_rows, edge, first_trap_row = _trilu_sizes(row, col, offset, bool(upper))
    n_total = n_trap + n_rect
    k = 2 * edge - 1
    b = edge - 0.5
    # tril writes the trapezoid first and the rectangle after it; triu is reversed.
    trap_base = n_rect if upper else 0
    rect_base = 0 if upper else n_trap
    rect_row0 = 0 if upper else first_trap_row + n_trap_rows
    last_trap_row = first_trap_row + n_trap_rows - 1

    output = nl.ndarray((2, n_total), dtype=nl.int32, buffer=nl.shared_hbm)

    block = P_MAX * F_CHUNK
    num_blocks = _div_ceil(n_trap, block)
    # The NKI frontend rejects starred expansion, so the slice is two calls.
    for blk_idx in range(_program_start(num_blocks), _program_stop(num_blocks)):
        base = blk_idx * block
        valid = min(block, n_trap - base)
        full_p = valid // F_CHUNK
        rem = valid % F_CHUNK
        used_p = full_p + (1 if rem > 0 else 0)
        shape = (used_p, F_CHUNK)

        # Linear output positions x = base + partition*F_CHUNK + free.
        x_i32 = nl.ndarray(shape, dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=x_i32, pattern=[[1, F_CHUNK]], offset=base, channel_multiplier=F_CHUNK)

        k_i32 = None
        if upper:
            # Reverse: idx = n_trap - 1 - x, exact in int32.
            idx_i32 = nl.ndarray(shape, dtype=nl.int32, buffer=nl.sbuf)
            _emit_const_i32(idx_i32, n_trap - 1)
            nisa.tensor_tensor(dst=idx_i32, data1=idx_i32, data2=x_i32, op=nl.subtract)
        else:
            idx_i32 = x_i32
        # k == 1 is the plain triangular number C(r) = r*(r+1)/2, which needs no k tile.
        if k != 1:
            k_i32 = nl.ndarray(shape, dtype=nl.int32, buffer=nl.sbuf)
            _emit_const_i32(k_i32, k)

        # s = sqrt(2*idx + b*b) in fp32
        scratch_f32 = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=scratch_f32, data=idx_i32, op0=nl.multiply, operand0=2.0, op1=nl.add, operand1=b * b
        )
        sqrt_f32 = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sqrt_f32, data=scratch_f32, op=nl.sqrt)

        # Guess g ~= floor(s - b). The int32 cast may truncate or round to
        # nearest; either way the guess is within +/-1 and the fences fix it.
        nisa.tensor_scalar(dst=scratch_f32, data=sqrt_f32, op0=nl.subtract, operand0=b)
        g_i32 = nl.ndarray(shape, dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=g_i32, src=scratch_f32)
        nisa.tensor_scalar(dst=g_i32, data=g_i32, op0=nl.maximum, operand0=0.0)

        c_i32 = nl.ndarray(shape, dtype=nl.int32, buffer=nl.sbuf)
        flag_i32 = nl.ndarray(shape, dtype=nl.int32, buffer=nl.sbuf)
        scratch_i32 = nl.ndarray(shape, dtype=nl.int32, buffer=nl.sbuf)

        # Down fence: g -= (idx < C(g)), then clamp to >= 0.
        _emit_tri_cumsum(c_i32, g_i32, scratch_i32, k_i32)
        nisa.tensor_tensor(dst=flag_i32, data1=idx_i32, data2=c_i32, op=nl.subtract)
        nisa.tensor_scalar(dst=flag_i32, data=flag_i32, op0=nl.less, operand0=0.0)
        nisa.tensor_tensor(dst=g_i32, data1=g_i32, data2=flag_i32, op=nl.subtract)
        nisa.tensor_scalar(dst=g_i32, data=g_i32, op0=nl.maximum, operand0=0.0)

        # Up fence: g += (idx >= C(g+1)).
        gp1_i32 = nl.ndarray(shape, dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gp1_i32, data=g_i32, op0=nl.add, operand0=1.0)
        _emit_tri_cumsum(c_i32, gp1_i32, scratch_i32, k_i32)
        nisa.tensor_tensor(dst=flag_i32, data1=idx_i32, data2=c_i32, op=nl.subtract)
        nisa.tensor_scalar(dst=flag_i32, data=flag_i32, op0=nl.greater_equal, operand0=0.0)
        nisa.tensor_tensor(dst=g_i32, data1=g_i32, data2=flag_i32, op=nl.add)

        # Position inside the row: q = idx - C(g).
        _emit_tri_cumsum(c_i32, g_i32, scratch_i32, k_i32)
        q_i32 = nl.ndarray(shape, dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=q_i32, data1=idx_i32, data2=c_i32, op=nl.subtract)

        if upper:
            # rowidx = last_trap_row - g'; colidx = col - 1 - q, both exact in int32.
            rowidx_i32 = nl.ndarray(shape, dtype=nl.int32, buffer=nl.sbuf)
            _emit_const_i32(rowidx_i32, last_trap_row)
            nisa.tensor_tensor(dst=rowidx_i32, data1=rowidx_i32, data2=g_i32, op=nl.subtract)
            colidx_i32 = nl.ndarray(shape, dtype=nl.int32, buffer=nl.sbuf)
            _emit_const_i32(colidx_i32, col - 1)
            nisa.tensor_tensor(dst=colidx_i32, data1=colidx_i32, data2=q_i32, op=nl.subtract)
        else:
            # A negative offset moves the first trapezoid row down the matrix.
            if first_trap_row != 0:
                nisa.tensor_scalar(
                    dst=g_i32, data=g_i32, op0=nl.add, operand0=float(first_trap_row)
                )
            rowidx_i32 = g_i32
            colidx_i32 = q_i32

        # Store the valid region: full partitions as one 2D pattern, then the
        # trailing partial partition.
        if full_p > 0:
            _store_plane(output, rowidx_i32, 0, full_p, F_CHUNK, trap_base + base, F_CHUNK)
            _store_plane(
                output, colidx_i32, 0, full_p, F_CHUNK, n_total + trap_base + base, F_CHUNK
            )
        if rem > 0:
            tail = trap_base + base + full_p * F_CHUNK
            _store_plane(output, rowidx_i32, full_p, 1, rem, tail, rem)
            _store_plane(output, colidx_i32, full_p, 1, rem, n_total + tail, rem)

    # Rectangle region: every row is full width, so both index planes are iota.
    # tril puts it below the trapezoid; triu puts it above.
    rect_rows = n_rect // col
    if rect_rows > 0:
        num_rect_fblocks = _div_ceil(col, F_CHUNK)
        num_rect_pblocks = _div_ceil(rect_rows, P_MAX)
        for rp_idx in range(_program_start(num_rect_pblocks), _program_stop(num_rect_pblocks)):
            r_base = rp_idx * P_MAX
            p_act = min(P_MAX, rect_rows - r_base)
            for rf_idx in range(num_rect_fblocks):
                f_base = rf_idx * F_CHUNK
                f_sz = min(F_CHUNK, col - f_base)

                rect_row_i32 = nl.ndarray((p_act, f_sz), dtype=nl.int32, buffer=nl.sbuf)
                nisa.iota(
                    dst=rect_row_i32,
                    pattern=[[0, f_sz]],
                    offset=rect_row0 + r_base,
                    channel_multiplier=1,
                )
                rect_col_i32 = nl.ndarray((p_act, f_sz), dtype=nl.int32, buffer=nl.sbuf)
                nisa.iota(
                    dst=rect_col_i32, pattern=[[1, f_sz]], offset=f_base, channel_multiplier=0
                )

                dst_off = rect_base + r_base * col + f_base
                _store_plane(output, rect_row_i32, 0, p_act, f_sz, dst_off, col)
                _store_plane(output, rect_col_i32, 0, p_act, f_sz, n_total + dst_off, col)

    return output


def _trilu_grid() -> tuple[int]:
    """One kernel program per logical NeuronCore; each takes a contiguous run of output blocks."""
    return (int(get_logical_neuron_cores()),)


_wrapped_trilu_nki = wrap_nki(trilu_indices_fence_kernel)


def trilu_nki_indices(upper: bool, row: int, col: int, offset: int, dtype: torch.dtype, device):
    """Emit triangular indices through the wrapped NKI kernel.

    Args:
        upper (bool): Whether to produce upper rather than lower triangular indices.
        row (int): Matrix row count whose emitted indices fit int32.
        col (int): Matrix column count whose emitted indices fit int32.
        offset (int): Diagonal offset.
        dtype (torch.dtype): Requested result dtype.
        device (int, str, torch.device, or None): Requested device, or ``None`` during tracing.

    Returns:
        torch.Tensor: Triangular indices shaped ``[2, N]`` in the requested dtype.

    Note:
        The unread anchor carries device and fake-tensor context. Direct calls use the
        named Neuron device, and tracing uses ``meta``. Torch-MLIR lowers the NKI wrapper.
        Callers must ensure kernel eligibility and int32-representable coordinates. Int64
        results reshape before a packed sign-extended view of int32 output because the
        compiler rejects reshape after the width-changing bitcast; no numeric cast occurs.
    """
    anchor_device = (
        device if device is not None and torch.device(device).type == "neuron" else "meta"
    )
    anchor = torch.zeros((1, 1), dtype=torch.int32, device=anchor_device)
    low = _wrapped_trilu_nki[_trilu_grid()](
        anchor=anchor, row=row, col=col, upper=int(upper), offset=offset
    )
    if dtype == torch.int64:
        return packed_int64_view(low)
    return low.to(dtype)
