"""NKI kernel that packs an int32 tensor into the int64 pair layout on chip.

An int64 tensor lives on device as int32 pairs ``[lo, hi, ...]``. This kernel writes
that stream from a contiguous run of low words; ``hi`` is the sign extension of ``lo``.
The tiling is the balanced flat tiling of ``stride2_gather``. The kernel replaces the
torch-level shift and stack pack, which the compiler lowers per element on a kernel output.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from torch_neuronx.nki_hop import wrap_nki
from torch_neuronx.python_ops.nki_kernels.stride2_gather import MAX_STAGE_ELEMS, P_MAX, div_ceil
from torch_neuronx.utils import get_logical_neuron_cores

# Pairs per partition per tile, so the [P_MAX, 2 * PAIRS_PER_TILE] staging tile stays
# within MAX_STAGE_ELEMS.
PAIRS_PER_TILE = MAX_STAGE_ELEMS // 2


def _emit_sign_word(hi, lo):
    # hi = -1 where lo < 0 else 0. The compare runs in fp32. int32 -> fp32 keeps the sign
    # and never maps a nonzero value to zero, so the flag is exact for every int32.
    nisa.tensor_scalar(dst=hi, data=lo, op0=nl.less, operand0=0.0, op1=nl.multiply, operand1=-1.0)


def _emit_pairs(pk, lo, hi, f: int):
    # Vector-engine copies are bit-exact for int32; scalar-engine copies cast through fp32.
    nisa.tensor_copy(dst=pk[:, 0 : 2 * f : 2], src=lo, engine=nisa.engine.vector)
    nisa.tensor_copy(dst=pk[:, 1 : 2 * f : 2], src=hi, engine=nisa.engine.vector)


def _flat_interleave(low, out):
    numel = low.shape[0]

    num_progs = nl.num_programs(0)
    prog_id = nl.program_id(0)
    per_prog = div_ceil(div_ceil(numel, num_progs), P_MAX) * P_MAX
    core_base = prog_id * per_prog
    core_n = min(per_prog, numel - core_base)
    core_n = max(core_n, 0)

    n_aligned = (core_n // P_MAX) * P_MAX
    tile_elems = P_MAX * PAIRS_PER_TILE
    num_tiles = div_ceil(n_aligned, tile_elems) if n_aligned > 0 else 0
    tail = core_n - n_aligned

    for i in nl.affine_range(num_tiles):
        t0 = core_base + i * tile_elems
        f = min(PAIRS_PER_TILE, (n_aligned - i * tile_elems) // P_MAX)
        lo = nl.ndarray((P_MAX, f), dtype=low.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=lo, src=low.ap([[f, P_MAX], [1, f]], t0))
        hi = nl.ndarray((P_MAX, f), dtype=nl.int32, buffer=nl.sbuf)
        _emit_sign_word(hi, lo)
        pk = nl.ndarray((P_MAX, 2 * f), dtype=nl.int32, buffer=nl.sbuf)
        _emit_pairs(pk, lo, hi, f)
        # One contiguous store per tile; a stride-2 HBM write does not coalesce.
        nisa.dma_copy(dst=out.ap([[2 * f, P_MAX], [1, 2 * f]], 2 * t0), src=pk)

    if tail > 0:
        t0 = core_base + n_aligned
        lo = nl.ndarray((tail, 1), dtype=low.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=lo, src=low.ap([[1, tail], [1, 1]], t0))
        hi = nl.ndarray((tail, 1), dtype=nl.int32, buffer=nl.sbuf)
        _emit_sign_word(hi, lo)
        pk = nl.ndarray((tail, 2), dtype=nl.int32, buffer=nl.sbuf)
        _emit_pairs(pk, lo, hi, 1)
        nisa.dma_copy(dst=out.ap([[2, tail], [1, 2]], 2 * t0), src=pk)

    return out


@nki.jit
def packed_interleave_kernel(low_flat):
    """Pack a flat int32 stream into sign-extended int64 pairs.

    Args:
        low_flat: flat int32 [numel] contiguous low words.

    Returns:
        out: flat int32 [2*numel] contiguous ``[lo, hi, ...]`` pairs in shared_hbm,
        ``hi = -1 if lo < 0 else 0`` (the physical view of a contiguous int64
        tensor holding the same values).

    Out-of-place for the same reason as ``stride2_flat_gather_kernel``: an
    internally allocated output gives each call in a graph an independent result.
    """
    numel = low_flat.shape[0]
    out = nl.ndarray((2 * numel,), dtype=nl.int32, buffer=nl.shared_hbm)
    return _flat_interleave(low_flat, out)


packed_interleave_hop = wrap_nki(packed_interleave_kernel)


def packed_int64_view(low: torch.Tensor, lnc: int = 0) -> torch.Tensor:
    """Pack int32 low words into a logical int64 tensor.

    Args:
        low (torch.Tensor): Contiguous int32 low words to sign-extend.
        lnc (int, optional): NKI launch grid, or ``0`` to use the target LNC.
            Defaults to ``0``.

    Returns:
        torch.Tensor: Int64 view of on-device ``[low, sign-extension]`` pairs.

    Note:
        The kernel writes pairs before a width-changing bitcast because the compiler
        rejects reshape after that bitcast. The grid must match ``neuronx-cc --lnc``
        or compilation fails with ``NCC_EVRF066``.
    """
    if low.numel() == 0:
        return torch.empty(low.shape, dtype=torch.int64, device=low.device)
    pairs = packed_interleave_hop[(lnc or int(get_logical_neuron_cores()),)](low.reshape(-1))
    return pairs.reshape(*low.shape[:-1], 2 * low.shape[-1]).view(torch.int64)
