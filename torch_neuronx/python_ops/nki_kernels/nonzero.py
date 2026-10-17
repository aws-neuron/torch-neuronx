"""NKI kernel for a flattened ``torch.nonzero`` using the GpSimd
``nisa.nonzero_with_count`` ISA instruction.

Why this exists
---------------
The previous ``nonzero`` lowering decomposed into per-last-dim
``cumsum`` + ``scatter_add``.  For a flat tensor of N elements the cumsum
lowers to a ``reduce_window`` of size N, which generates an enormous number
of instructions and dominates compile time for large inputs.

``nisa.nonzero_with_count`` (NeuronCore-v3 / Trn2+) computes, for one SBUF
partition row of length T, the output ``[idx0, idx1, ..., pad, ..., count]``
of length T+1, where each ``idx`` is the position of a nonzero element plus a
caller-supplied ``index_offset``.  This file tiles a long flat input along the
free dimension, runs the ISA per tile with ``index_offset = chunk * chunk_size``
so indices are global, and **compacts** each tile's valid prefix into a single
dense output list using a running write offset (indirect DMA).

Output contract
---------------
``flat_nonzero_kernel`` returns ``(indices, count)`` where:
  * ``indices``: [1, N] int32 — the first ``count`` entries are the ascending
    flat indices of nonzero elements; entries after ``count`` are unspecified
    (the caller slices ``indices[:count]``).
  * ``count``: [1, 1] int32 — total number of nonzero elements.

The caller (``decompositions.nonzero_with_count``) unravels the flat indices
into N-D coordinates and applies any ``fill_value`` padding.

Notes / constraints
-------------------
  * Requires NeuronCore-v3+ (Trn2/Trn3). The ISA exists only there.
  * Processes 8 chunks per ISA call across the 8 GpSimd cores (partitions
    0,16,...,112) in one ISA call. Each core's result row is read straight to a
    partition-0 tile via DMA (DMA can address all 8 partitions directly), then
    compacted in ascending core order. A remainder of <8 full chunks plus a
    final partial tail tile is handled on a single partition.
  * ``chunk_size`` trades SBUF usage and instruction count.
"""

import nki
import nki.isa as nisa
import nki.language as nl

_PADDING_VAL = -1
# Default free-dim tile width. Bounds SBUF usage to ~chunk_size int32 per tile.
DEFAULT_CHUNK_SIZE = 8192

# GpSimd layout: 8 cores at partitions 0,16,...,112.
_NUM_CORES = 8
_P_STRIDE = 16  # partition gap between adjacent GpSimd cores


@nki.jit
def flat_nonzero_kernel(input_tensor, chunk_size: int = DEFAULT_CHUNK_SIZE):
    """Find ascending flat indices of nonzero elements.

    Args:
        input_tensor (nl.ndarray): Two-dimensional ``[1, N]`` ``float32`` or ``int32`` input.
        chunk_size (int, optional): Per-core free-dimension tile width. Defaults to
            ``DEFAULT_CHUNK_SIZE``.

    Returns:
        tuple[nl.ndarray, nl.ndarray]: A ``[1, N]`` index buffer whose first
            ``count`` entries are valid and whose remaining entries are unspecified,
            plus a ``[1, 1]`` nonzero count.

    Note:
        The kernel processes eight GpSimd chunks concurrently and supports any ``N``.
    """
    n = input_tensor.shape[-1]
    group = _NUM_CORES * chunk_size
    num_groups = n // group
    rem = n - num_groups * group
    num_rem_chunks = rem // chunk_size
    tail_len = rem - num_rem_chunks * chunk_size

    indices = nl.ndarray((1, n), dtype=nl.int32, buffer=nl.shared_hbm)
    count = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)

    # Running write offset (cumulative nonzero count) in SBUF.
    offset = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf, name="offset")
    nisa.memset(dst=offset, value=0)

    # Rolled loop over groups of 8 chunks. A single body instance is emitted
    # regardless of n, so instruction count / compile time stay constant.
    for g in nl.sequential_range(num_groups):
        _process_nonzero_group(input_tensor, indices, offset, g * group, chunk_size, n)

    # Remainder: the <8 full chunks of the last partial group, then a tail tile.
    rbase = num_groups * group
    for j in range(num_rem_chunks):
        _process_nonzero_tile(input_tensor, indices, offset, rbase + j * chunk_size, chunk_size, n)
    if tail_len > 0:
        _process_nonzero_tile(
            input_tensor, indices, offset, rbase + num_rem_chunks * chunk_size, tail_len, n
        )

    nisa.dma_copy(dst=count[0:1, 0:1], src=offset[0:1, 0:1])
    return indices, count


def _compact(indices, offset, src_row, length, base_correction, n):
    """Append one chunk's results to ``indices`` at the running ``offset``.

    ``src_row`` is a partition-0 (1, length+1) tile holding
    ``[idx0, ..., pad, ..., count]`` from one nonzero_with_count chunk. The
    ``length`` index/padding entries are corrected by ``+ base_correction`` and
    indirect-DMA'd into ``indices`` at the current offset; ``offset`` is then
    advanced by this chunk's count. The padding tail is overwritten by the next
    chunk's real data, so ``indices[:count]`` ends up dense.
    """
    di = nl.ndarray((1, length), dtype=nl.int32, buffer=nl.sbuf)
    if base_correction != 0:
        nisa.tensor_scalar(
            dst=di, data=src_row[0:1, 0:length], op0=nl.add, operand0=base_correction
        )
    else:
        nisa.tensor_copy(dst=di, src=src_row[0:1, 0:length])

    offset_tile = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=offset_tile, src=offset[0:1, 0:1])
    nisa.dma_copy(
        dst=indices.ap(
            pattern=[[n, 1], [1, length]], offset=0, scalar_offset=offset_tile, indirect_dim=1
        ),
        src=di,
    )

    count_tile = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=count_tile, src=src_row[0:1, length : length + 1])
    nisa.tensor_tensor(dst=offset[0:1, 0:1], data1=offset[0:1, 0:1], data2=count_tile, op=nl.add)


def _process_nonzero_group(input_tensor, indices, offset, gbase, chunk_size, n):
    """Process 8 chunks (one per GpSimd core) starting at ``gbase``.

    Loads chunk i into partition ``16*i`` (i in 0..7), runs one
    nonzero_with_count over all 8 cores (index_offset=gbase), then compacts the
    cores in ascending order. DMA can read every core's partition directly, so
    each core's result row is DMA'd to a partition-0 tile and compacted there.

    The single ``index_offset`` makes every core's indices relative to
    ``gbase``; core i additionally covers chunk i, so its indices need a static
    ``+ i*chunk_size`` correction (i is a Python int).
    """
    p = 128
    in_sb = nl.ndarray((p, chunk_size), dtype=input_tensor.dtype, buffer=nl.sbuf)
    for i in range(_NUM_CORES):
        nisa.dma_copy(
            dst=in_sb[_P_STRIDE * i, :],
            src=input_tensor[0, nl.ds(gbase + i * chunk_size, chunk_size)],
        )
    nz_sb = nl.ndarray((p, chunk_size + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nonzero_with_count(dst=nz_sb, src=in_sb, index_offset=gbase, padding_val=_PADDING_VAL)

    for i in range(_NUM_CORES):
        row = nl.ndarray((1, chunk_size + 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=row, src=nz_sb[_P_STRIDE * i : _P_STRIDE * i + 1, :])
        _compact(indices, offset, row, chunk_size, i * chunk_size, n)


def _process_nonzero_tile(input_tensor, indices, offset, base, length, n):
    """Find nonzeros in input_tensor[0, base:base+length] on a single partition.

    Used for the remainder (the <8 full chunks of the last partial group and the
    final partial tail). ``base`` may be a dynamic value; ``length`` is a Python
    int so tensor shapes stay static.
    """
    in_sb = nl.ndarray((1, length), dtype=input_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=in_sb[0, :], src=input_tensor[0, nl.ds(base, length)])

    nz_sb = nl.ndarray((1, length + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nonzero_with_count(dst=nz_sb, src=in_sb, index_offset=base, padding_val=_PADDING_VAL)

    # base is already baked in via index_offset, so no extra correction.
    _compact(indices, offset, nz_sb, length, 0, n)


try:
    import torch

    from torch_neuronx.nki_hop import nki_op, wrap_nki

    _wrapped_nonzero = wrap_nki(flat_nonzero_kernel)

    @nki_op("nki_kernels::nonzero", mutates_args={})
    def nonzero_nki(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run flattened nonzero through the wrapped NKI kernel.

        Args:
            x (torch.Tensor): Two-dimensional ``[1, N]`` ``int32`` or ``float32`` input.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A ``[1, N]`` index buffer whose first
                ``count`` entries are ascending, and a ``[1, 1]`` nonzero count.

        Note:
            Full and tail tiles are handled without host-side padding.
        """
        return _wrapped_nonzero(x, chunk_size=DEFAULT_CHUNK_SIZE)

    _has_nki_nonzero = True
except (ImportError, ModuleNotFoundError):
    _has_nki_nonzero = False


def _platform_supports_nonzero_isa() -> bool:
    """nisa.nonzero_with_count exists only on NeuronCore-v3+ (Trn2/Trn3).

    GetPlatformTarget() (PlatformUtils.cpp) only ever returns "trn1", "trn2", or
    "trn3" — v3+ are trn2 and trn3.
    """
    try:
        from torch_neuronx.utils import get_platform_target

        return get_platform_target() in ("trn2", "trn3")
    except Exception:
        return False


# Single gate used by the decomposition: nkilib importable AND a v3+ target.
# Evaluated at import time; the platform target is fixed for the process.
_has_nki_nonzero = _has_nki_nonzero and _platform_supports_nonzero_isa()
