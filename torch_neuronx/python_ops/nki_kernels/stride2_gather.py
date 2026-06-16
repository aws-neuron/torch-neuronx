"""NKI kernels for stride-2 last-dim gather and scatter.

Used for packed int64 low-/high-word extraction and materialization. An int64
tensor is stored on device as pairs of int32 lanes ``[lo, hi, lo, hi, ...]``;
these kernels move one lane (selected by ``parity``) to/from a contiguous int32
buffer.

Balanced flat tiling
--------------------
Gather/scatter is purely element-wise over a flat contiguous int32 pair-stream,
so the logical tensor shape is irrelevant to correctness. We exploit that to
flatten the stream to ``numel`` pairs and tile it as balanced ``[P_MAX, F]``
blocks via NKI access patterns (``.ap``), regardless of the logical shape. This
keeps the partition dimension full (128) for the bulk of any tensor, fixing the
shape-dependent under-utilization of a logical-shape factorization (e.g. a
``(1048576, 1)`` tensor used to run at ~14.6ms because only its trailing dim of
1 fed the free dimension; balanced flat tiling runs it at ~0.82ms, the same as a
balanced ``(1024, 1024)``). A ``<P_MAX`` remainder is handled by a separate tail
tile (the standard fused_adam pattern). Performance is now shape-independent for
a given element count.

Per tile (gather):
  1. burst-load the contiguous packed block ``[P_MAX, 2F]`` HBM -> SBUF via
     ``.ap`` (free stride 1, so HBM reads coalesce into bursts),
  2. select the parity lane on chip via an SBUF -> SBUF ``nisa.dma_copy``
     (pure data movement; bit-exact for int32),
  3. burst-store the contiguous result ``[P_MAX, F]`` SBUF -> HBM via ``.ap``.

Two constraints, learned empirically:
  - The on-chip lane select must use ``nisa.dma_copy``, NOT ``nl.copy``.
    ``nl.copy`` routes through a compute engine that rounds int32 through an fp32
    path and corrupts the low bits.
  - The lane select is done on chip (SBUF), never as a strided HBM read. A
    strided HBM read of every-other int32 cannot coalesce into bursts (~0.5 GB/s).

Naming follows the NKI partition/free convention: ``P_MAX`` is the partition
dimension (hardware-parallel, == 128) and ``F`` is the free-dimension tile width.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

from torch_neuronx.utils import get_logical_neuron_cores

# Partition dimension: the hardware partition count.
P_MAX = 128
# Free-dimension tile width, in *pairs*, for the SCATTER kernel (dma_copy-based
# lane write). Bounds the SBUF staging buffer to P_MAX * 2 * F int32 elements.
# Performance is insensitive above ~512; 2048 keeps the launch count low.
F = 2048
# Free-dimension tile width for the GATHER kernel, which selects the lane on
# chip with nisa.nc_n_gather (GpSimd) — ~3.4-4.5x faster than the dma_copy select.
#
# Must stay <= 512. nc_n_gather decomposes into n = ceil(elems_per_partition / 512)
# ISA groups. On Trn2 there is a GpSimd bug: when a kernel chains two or more
# nc_n_gather calls, the second-and-later call only produces its first ISA group
# correctly and zeros the rest. F <= 512 forces n == 1 (always correct, any number
# of chained tiles). Ticket V2245128914
F_GATHER = 512

LNC = int(get_logical_neuron_cores())


def div_ceil(numerator: int, denominator: int) -> int:
    """Ceiling division. Avoids inline ``(n + d - 1) // d`` per NKI conventions."""
    return -(-numerator // denominator)


@nki.jit
def stride2_flat_gather_kernel(packed_in, out, parity):
    """Gather one lane of a flat packed pair-stream into a contiguous buffer.

    Args:
        packed_in: flat int32 [2*numel] contiguous packed pairs ``[lo,hi,...]``,
            rebased so element ``parity`` of each pair is the requested lane.
        out: flat int32 [numel] contiguous output.
        parity: lane to gather — 0 = low word, 1 = high word.

    Returns:
        out (written in place; also returned per NKI convention).

    Notes:
        Balanced flat tiling: the stream is tiled as [P_MAX, F_GATHER] pair-blocks
        independent of logical shape, so the partition dim is always full for
        the bulk. A <P_MAX remainder is handled by a separate tail tile.

        Lane select uses nisa.nc_n_gather (GpSimd) rather than a strided
        dma_copy: it is ~3.4-4.5x faster (GpSimd does the on-chip select, so the
        DMA engines are free to overlap the HBM bursts) and bit-exact (no dtype
        conversion). The per-partition indices [parity, parity+2, ...] are built
        with nisa.iota.

        SPMD: when launched with LNC2 the stream is split across the two
        programs (P_MAX-aligned per-program slice). With LNC1 there is a single
        program over the whole stream.
    """
    numel = out.shape[0]

    # SPMD split: each program handles a contiguous P_MAX-aligned slice of pairs.
    # P_MAX alignment guarantees every program's bulk fills 128 partitions; the
    # last program absorbs whatever remains (including the <P_MAX final tail).
    num_progs = nl.num_programs(0)
    prog_id = nl.program_id(0)
    per_prog = div_ceil(div_ceil(numel, num_progs), P_MAX) * P_MAX
    core_base = prog_id * per_prog
    core_n = min(per_prog, numel - core_base)
    core_n = max(core_n, 0)

    n_aligned = (core_n // P_MAX) * P_MAX
    tile_pairs = P_MAX * F_GATHER
    num_tiles = div_ceil(n_aligned, tile_pairs) if n_aligned > 0 else 0
    tail = core_n - n_aligned

    for i in nl.affine_range(num_tiles):
        t0 = core_base + i * tile_pairs  # first pair index in this tile
        masked_f = min(F_GATHER, (n_aligned - i * tile_pairs) // P_MAX)
        # 1. contiguous burst-load [P_MAX, 2*masked_f] packed pairs (free stride 1)
        pk = nl.ndarray((P_MAX, 2 * masked_f), dtype=packed_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=pk, src=packed_in.ap([[2 * masked_f, P_MAX], [1, 2 * masked_f]], 2 * t0))
        # 2. on-chip lane select via GpSimd gather: indices [parity, parity+2, ...]
        idx = nl.ndarray((P_MAX, masked_f), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.iota(idx, pattern=[[2, masked_f]], offset=parity)
        sel = nl.ndarray((P_MAX, masked_f), dtype=packed_in.dtype, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=sel, data=pk, indices=idx)
        # 3. contiguous burst-store [P_MAX, masked_f] -> HBM
        nisa.dma_copy(dst=out.ap([[masked_f, P_MAX], [1, masked_f]], t0), src=sel)

    if tail > 0:
        t0 = core_base + n_aligned
        pk = nl.ndarray((tail, 2), dtype=packed_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=pk, src=packed_in.ap([[2, tail], [1, 2]], 2 * t0))
        idx = nl.ndarray((tail, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.iota(idx, pattern=[[2, 1]], offset=parity)
        sel = nl.ndarray((tail, 1), dtype=packed_in.dtype, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=sel, data=pk, indices=idx)
        nisa.dma_copy(dst=out.ap([[1, tail], [1, 1]], t0), src=sel)

    return out


@nki.jit
def stride2_flat_scatter_kernel(packed_io, vals, parity):
    """Scatter a contiguous buffer into one lane of a flat packed pair-stream.

    Args:
        packed_io: flat int32 [2*numel] contiguous packed pairs (read+write),
            rebased so element ``parity`` of each pair is the lane being written.
        vals: flat int32 [numel] contiguous source values.
        parity: lane to write — 0 = low word, 1 = high word.

    Returns:
        packed_io (written in place; also returned per NKI convention).

    Notes:
        Burst-loads each packed block (to preserve the lane we are not writing),
        burst-loads the source values, writes them into the parity lane on chip,
        then burst-stores the full packed block back.

        SPMD: when launched with LNC2 the stream is split across the two
        programs (P_MAX-aligned per-program slice), so both cores' DMA engines
        run in parallel. With LNC1 there is a single program over the whole
        stream.
    """
    numel = vals.shape[0]

    # SPMD split — see stride2_flat_gather_kernel for the rationale.
    num_progs = nl.num_programs(0)
    prog_id = nl.program_id(0)
    per_prog = div_ceil(div_ceil(numel, num_progs), P_MAX) * P_MAX
    core_base = prog_id * per_prog
    core_n = min(per_prog, numel - core_base)
    core_n = max(core_n, 0)

    n_aligned = (core_n // P_MAX) * P_MAX
    tile_pairs = P_MAX * F
    num_tiles = div_ceil(n_aligned, tile_pairs) if n_aligned > 0 else 0
    tail = core_n - n_aligned

    for i in nl.affine_range(num_tiles):
        t0 = core_base + i * tile_pairs
        masked_f = min(F, (n_aligned - i * tile_pairs) // P_MAX)
        # 1. burst-load the packed block (preserves the other lane)
        pk = nl.ndarray((P_MAX, 2 * masked_f), dtype=packed_io.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=pk, src=packed_io.ap([[2 * masked_f, P_MAX], [1, 2 * masked_f]], 2 * t0))
        # 2. burst-load the contiguous source values
        v = nl.ndarray((P_MAX, masked_f), dtype=vals.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=v, src=vals.ap([[masked_f, P_MAX], [1, masked_f]], t0))
        # 3. write values into the parity lane on chip (SBUF -> SBUF)
        nisa.dma_copy(dst=pk[:, parity : 2 * masked_f : 2], src=v)
        # 4. burst-store the full packed block back -> HBM
        nisa.dma_copy(dst=packed_io.ap([[2 * masked_f, P_MAX], [1, 2 * masked_f]], 2 * t0), src=pk)

    if tail > 0:
        t0 = core_base + n_aligned
        pk = nl.ndarray((tail, 2), dtype=packed_io.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=pk, src=packed_io.ap([[2, tail], [1, 2]], 2 * t0))
        v = nl.ndarray((tail, 1), dtype=vals.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=v, src=vals.ap([[1, tail], [1, 1]], t0))
        nisa.dma_copy(dst=pk[:, parity : parity + 1], src=v)
        nisa.dma_copy(dst=packed_io.ap([[2, tail], [1, 2]], 2 * t0), src=pk)

    return packed_io


def run_stride2_gather(src_tensor, dst_tensor, *, parity: int = 0):
    """Run stride-2 gather on the last dimension.

    Args:
        src_tensor: source tensor with stride-2 on last dim (a packed lane alias).
        dst_tensor: contiguous destination of the same logical shape.
        parity: 0 for low words (lane 0), 1 for high words (lane 1).
    """
    numel = int(dst_tensor.numel())
    if numel == 0:
        return

    # The lane alias shares the int32 pair-stream storage. Rebase to the even
    # (pair) boundary; the kernel re-selects lane `parity` within each pair.
    base_off = int(src_tensor.storage_offset()) - parity
    if base_off < 0:
        base_off = 0

    packed = src_tensor.as_strided((2 * numel,), (1,), storage_offset=base_off)
    out_flat = dst_tensor.reshape(numel)
    stride2_flat_gather_kernel[LNC](packed, out_flat, int(parity))


def run_stride2_scatter(src_tensor, dst_tensor, *, parity: int = 0):
    """Run stride-2 scatter on the last dimension.

    Args:
        src_tensor: contiguous source tensor.
        dst_tensor: destination tensor with stride-2 on last dim (a packed lane alias).
        parity: 0 for low word positions (lane 0), 1 for high word positions (lane 1).
    """
    numel = int(src_tensor.numel())
    if numel == 0:
        return

    base_off = int(dst_tensor.storage_offset()) - parity
    if base_off < 0:
        base_off = 0

    packed = dst_tensor.as_strided((2 * numel,), (1,), storage_offset=base_off)
    src_flat = src_tensor.reshape(numel)
    stride2_flat_scatter_kernel[LNC](packed, src_flat, int(parity))
