"""In-graph index bound checking for Neuron index ops.

Public API is written around the *purpose* (bound checking), not the mechanism,
so the implementation can change without touching callers.

Current mechanism: the row gather an index op needs is lowered as an indirect
``nisa.dma_copy`` (``vector_offset``). The dynamic DGE path range-checks each
index against the gathered tensor's extent and, with the default
``oob_mode.error``, raises a runtime error on an out-of-bounds index. This fuses
the bounds check into the gather (no separate ``torch.any(...).item()`` host-sync
precondition) and, because the check rides the lowered graph, it runs on every
execution rather than only at lowering/cache-miss.

Indices are expected in their FINAL form (in ``[0, N)`` for valid rows). Negative
index normalization is the caller's responsibility: the actual execution gather
needs the same normalized indices, so the op lowering should normalize once
(e.g. ``where(i < 0, i + N, i)`` for wrap-semantics ops) and pass the result
here. Any index outside ``[0, N)`` — including a leftover negative (reinterpreted
as a huge uint32) — goes out of bounds and raises.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from torch_neuronx.nki_hop import nki_op, wrap_nki

# Hardware partition count: indirect vector_offset gathers <=128 rows per DMA.
P_MAX = 128


# ---------------------------------------------------------------------------
# Standalone bounds check (for ops that do NOT gather: scatter, index_put,
# index_select_backward). Cost is independent of P and the check runs with no
# host-sync precondition:
#   1. burst-load indices as [128, F] and compute an OOB mask in int32
#      (idx >= N) | (idx < 0)  -> exact across the full int32 range (no fp32
#      2**24 limit), via nl.greater_equal / nl.less + nl.bitwise_or;
#   2. reduce the mask (max) to one 0/1 per partition;
#   3. turn it into an offset (mask * 128) into a fixed [128, 1] sentinel: 0
#      indexes a valid row, 128 is out of range -> DGE oob_mode.error faults.
#
# Dimension-agnostic: the bound N only enters the comparison; the gather target
# is a fixed [128, 1] HBM scratch allocated inside the kernel (no host tensor),
# independent of the indexed tensor's shape/dim. vector_offset's indirect_dim=0
# requirement is satisfied trivially by the sentinel.
# ---------------------------------------------------------------------------

# Free-dimension tile width for the bounds check. Bounds the SBUF staging
# (idx_tile + masks) to ~P_MAX * 2 * F_SIZE int32 elements per chunk.
F_SIZE = 2048


def _div_ceil(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


@nki.jit
def _mask_bounds_check_kernel(index, hi, lo):
    # index: int32 indices of ANY shape — the access patterns below view the
    #           tensor flat, so no host flatten is needed. Balanced flat tiling
    #           (the stride2_gather convention): contiguous [P_MAX, tile_w]
    #           blocks plus a [tail, 1] remainder for numel % P_MAX != 0, so the
    #           partition dim stays full for the bulk. Each tile is independent.
    # [lo, hi): inclusive lower / exclusive upper bound. lo=0 rejects negatives;
    #           lo=-hi accepts the wrap window [-hi, hi) (no host normalization).
    # Returns idx + sentinel_gather (== idx in-bounds), same shape as the input;
    # faults (oob_mode.error) iff any index is < lo or >= hi.
    numel = 1
    for d in index.shape:
        numel *= d
    # Passthrough output (== idx in-bounds, same shape): the op consumes this, so
    # the gather added into it survives DCE (which recurses into the kernel).
    out_idx = nl.ndarray(index.shape, dtype=nl.int32, buffer=nl.shared_hbm)
    # Internal HBM gather target, ZERO-initialized so an in-bounds gather returns
    # 0 (out_idx == idx); an offset of P_MAX is out of range -> oob_mode.error.
    sentinel = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.hbm)
    nisa.dma_copy(dst=sentinel, src=nl.zeros((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf))

    n_aligned = (numel // P_MAX) * P_MAX
    tile_elems = P_MAX * F_SIZE
    num_tiles = _div_ceil(n_aligned, tile_elems) if n_aligned > 0 else 0
    tail = numel - n_aligned

    # Aligned bulk: contiguous [P_MAX, tile_w] tiles. Each tile range-checks per
    # partition, gathers the (zero) sentinel — faulting on any OOB index in the
    # tile — and writes out_tile = idx_tile + gathered (== idx_tile in-bounds).
    for i in nl.affine_range(num_tiles):
        base = i * tile_elems
        tile_w = min(F_SIZE, (n_aligned - i * tile_elems) // P_MAX)
        idx_tile = nl.ndarray((P_MAX, tile_w), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx_tile, src=index.ap([[tile_w, P_MAX], [1, tile_w]], base))
        # out-of-bounds mask: (idx >= hi) | (idx < lo), reduced to one flag/partition.
        oob_mask = nl.ndarray((P_MAX, tile_w), dtype=nl.int32, buffer=nl.sbuf)
        below_lo = nl.ndarray((P_MAX, tile_w), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=oob_mask, data=idx_tile, op0=nl.greater_equal, operand0=hi)
        nisa.tensor_scalar(dst=below_lo, data=idx_tile, op0=nl.less, operand0=lo)
        nisa.tensor_tensor(dst=oob_mask, data1=oob_mask, data2=below_lo, op=nl.maximum)
        oob_flag = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=oob_flag, op=nl.max, data=oob_mask, axis=[1])
        # offset = flag * P_MAX: 0 -> sentinel row 0 (value 0), P_MAX -> OOB -> fault.
        sentinel_off = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sentinel_off, data=oob_flag, op0=nl.multiply, operand0=P_MAX)
        gathered = nl.ndarray((P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=gathered,
            src=sentinel.ap(
                pattern=[[1, P_MAX], [1, 1]], vector_offset=sentinel_off, indirect_dim=0
            ),
        )
        # out_tile = idx_tile + gathered, broadcasting gathered[P_MAX,1] over the
        # free dim (stride-0 AP). tensor_tensor (not tensor_scalar) — a tensor
        # operand to tensor_scalar must be fp32, but tensor_tensor handles int32.
        out_tile = nl.ndarray((P_MAX, tile_w), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=out_tile, data1=idx_tile, data2=gathered.ap([[1, P_MAX], [0, tile_w]]), op=nl.add
        )
        nisa.dma_copy(dst=out_idx.ap([[tile_w, P_MAX], [1, tile_w]], base), src=out_tile)

    # Tail (< P_MAX elements): one element per partition, [tail, 1].
    if tail > 0:
        base = n_aligned
        idx_tail = nl.ndarray((tail, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx_tail, src=index.ap([[1, tail], [1, 1]], base))
        oob_mask = nl.ndarray((tail, 1), dtype=nl.int32, buffer=nl.sbuf)
        below_lo = nl.ndarray((tail, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=oob_mask, data=idx_tail, op0=nl.greater_equal, operand0=hi)
        nisa.tensor_scalar(dst=below_lo, data=idx_tail, op0=nl.less, operand0=lo)
        nisa.tensor_tensor(
            dst=oob_mask, data1=oob_mask, data2=below_lo, op=nl.maximum
        )  # [tail,1] (free=1)
        sentinel_off = nl.ndarray((tail, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sentinel_off, data=oob_mask, op0=nl.multiply, operand0=P_MAX)
        gathered = nl.ndarray((tail, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=gathered,
            src=sentinel.ap(
                pattern=[[1, tail], [1, 1]], vector_offset=sentinel_off, indirect_dim=0
            ),
        )
        out_tail = nl.ndarray((tail, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=out_tail, data1=idx_tail, data2=gathered, op=nl.add)
        nisa.dma_copy(dst=out_idx.ap([[1, tail], [1, 1]], base), src=out_tail)

    return out_idx


@nki_op("nki_kernels::bounds_check_mask", mutates_args={})
def _mask_bounds_check(index: torch.Tensor, hi: int, lo: int) -> torch.Tensor:
    """Internal: return ``index`` (int32, any shape) unchanged, faulting
    (oob_mode.error) iff any index is outside ``[lo, hi)``.

    The kernel produces the output as ``idx + sentinel_gather`` (gather == 0
    in-bounds), so the consuming op MUST use the returned tensor — that data
    dependency keeps the gather alive through neuronx-cc DCE (which recurses into
    the kernel). On a fault the runtime invokes the op's registered
    ``error_handler`` (with inputs reconstructed from the OperationContext) to
    raise the exact op-specific exception (see op_error_handlers.py)."""
    return wrap_nki(_mask_bounds_check_kernel)[(1,)](index=index, hi=hi, lo=lo)


def assert_index_in_bounds(index: torch.Tensor, hi: int, lo: int = 0) -> torch.Tensor:
    """Return ``index`` (int32) unchanged, faulting on any value outside
    ``[lo, hi)``. The consuming op MUST use the returned tensor — that data
    dependency forces the in-graph check to run (and orders it before the op).
    Works for any bounds (int32-exact), any shape, any size; dimension-agnostic
    (only the scalar bounds are used). ``lo=0`` rejects negatives; ``lo=-hi``
    accepts ``[-hi, hi)``.

    The kernel tiles the index flat via access patterns and returns it in the
    SAME shape, so no host reshape/pad/slice is needed — only an int32 cast
    (a no-op when the index is already int32)."""
    hi = int(hi)
    lo = int(lo)
    if index.numel() == 0:
        return index

    idx32 = index.to(torch.int32)
    if idx32.dim() == 0:
        # The kernel can't allocate a rank-0 output (empty shape); check a [1]
        # view (a cheap reshape) and restore the original 0-d shape.
        return _mask_bounds_check(idx32.reshape(1), hi, lo).reshape(idx32.shape)
    return _mask_bounds_check(idx32, hi, lo)
