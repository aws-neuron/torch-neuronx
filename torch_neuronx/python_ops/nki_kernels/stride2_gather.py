"""NKI kernels for stride-2 last-dim gather and scatter.

Used for packed int64 low-/high-word extraction and materialization. An int64
tensor is stored on device as pairs of int32 lanes ``[lo, hi, lo, hi, ...]``;
these kernels move one lane (selected by ``parity``) to/from a contiguous int32
buffer.

Balanced flat tiling
--------------------
Gather/scatter is purely element-wise over a flat contiguous int32 pair-stream,
so the logical tensor shape is irrelevant to correctness. We exploit that to
flatten the stream to ``numel`` groups and tile it as balanced ``[P_MAX, groups_per_tile]``
blocks via NKI access patterns (``.ap``), regardless of the logical shape. This
keeps the partition dimension full (128) for the bulk of any tensor, fixing the
shape-dependent under-utilization of a logical-shape factorization (e.g. a
``(1048576, 1)`` tensor used to run at ~14.6ms because only its trailing dim of
1 fed the free dimension; balanced flat tiling runs it at ~0.82ms, the same as a
balanced ``(1024, 1024)``). A ``<P_MAX`` remainder is handled by a separate tail
tile (the standard fused_adam pattern). Performance is now shape-independent for
a given element count.

Per tile:
  1. burst-load the contiguous packed block ``[P_MAX, 2F]`` HBM -> SBUF via
     ``.ap`` (free stride 1, so HBM reads coalesce into bursts),
  2. select the parity lane on chip (SBUF -> SBUF), then
  3. burst-store the contiguous result ``[P_MAX, groups_per_tile]`` SBUF -> HBM via ``.ap``.

On-chip lane select engine:
  - The gather selects the lane with ``nisa.tensor_copy`` on the **Vector
    Engine** (reading the strided source ``pk[:, parity::2]``). On the Vector
    Engine tensor_copy is a bit-accurate copy when src/dst dtypes match
    (int32==int32). It is the fastest option measured (~2x faster than a GpSimd
    ``nc_n_gather`` select, ~4x faster than a strided ``dma_copy`` select),
    because the Vector Engine does the select while the DMA engines overlap the
    HBM bursts.
  - The scatter writes the lane with ``nisa.dma_copy`` (SBUF -> SBUF strided
    store into the packed block, preserving the other lane).

Constraints, learned empirically:
  - The on-chip select must NOT use ``nl.copy``: it routes through the Scalar
    Engine, which casts int32 through fp32 and corrupts the low bits. Use
    ``nisa.tensor_copy(engine=vector)`` (or ``dma_copy``), which are bit-exact
    for matching int32 dtypes.
  - The lane select is done on chip (SBUF), never as a strided HBM read. A
    strided HBM read of every-other int32 cannot coalesce into bursts (~0.5 GB/s).
  - ``nc_n_gather`` (GpSimd) also works and is bit-exact, but it decomposes into
    ceil(elems_per_partition / 512) ISA groups and corrupts later tiles when two
    or more multi-group calls are chained (ticket V2245128914), forcing a
    F<=512 cap. ``tensor_copy(vector)`` has no such limit and is faster, so it is
    preferred.

Naming follows the NKI partition/free convention: ``P_MAX`` is the partition
dimension (hardware-parallel, == 128) and the free-dimension tile width per tile
is ``groups_per_tile``, chosen so the loaded staging stays within ``MAX_STAGE_ELEMS``.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from torch_neuronx.nki_hop import nki_op, wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores, use_dynamic_nki_kernels

# Partition dimension: the hardware partition count.
P_MAX = 128
# Max elements staged in SBUF per partition per tile. The gather/scatter
# burst-loads `stride*groups` elements to select one lane, so bounding the
# LOADED size (rather than the group count) keeps SBUF use independent of
# `stride`: a large stride just yields fewer groups/tile and more tiles, never an
# overflow. 4096 int32 (16 KB/partition) is well within SBUF and matches the
# original stride-2 tile (2048 pairs); performance is insensitive above ~512, so
# the launch count stays low.
MAX_STAGE_ELEMS = 4096

LNC = int(get_logical_neuron_cores())


def div_ceil(numerator: int, denominator: int) -> int:
    """Compute the ceiling of an integer quotient.

    Args:
        numerator (int): Dividend.
        denominator (int): Nonzero divisor.

    Returns:
        int: Smallest integer greater than or equal to the quotient.
    """
    return -(-numerator // denominator)


def _flat_gather_lane(grouped_in, out, stride=2, lane=0):
    """Gather ``lane`` of a flat stride-``stride`` group-stream into ``out``.

    Shared tiling for the in-place / out-of-place gather entry points — a plain
    function that emits NKI ops into the caller's kernel trace (mirrors the
    helper pattern in ``nonzero.py``). ``numel`` is taken from ``out.shape[0]``.
    The input stream is ``stride*numel`` elements laid out as ``numel`` groups of
    ``stride``; element ``lane`` of each group is selected. ``stride=2, lane=p``
    is the packed-int64 low/high-word gather.

    Balanced flat tiling: SPMD split into P_MAX-aligned per-program slices, then
    [P_MAX, groups_per_tile] group blocks (loaded staging stride*groups bounded by
    MAX_STAGE_ELEMS) with a <P_MAX tail. Each tile burst-loads
    groups, selects ``lane`` on the Vector Engine (``tensor_copy`` is a
    bit-accurate copy for matching dtypes, unlike Scalar-Engine ``nl.copy`` which
    casts through fp32), and burst-stores contiguous words.
    """
    numel = out.shape[0]

    # A launch without a grid has no axis 0.
    if nl.program_ndim() == 0:
        num_progs, prog_id = 1, 0
    else:
        num_progs = nl.num_programs(0)
        prog_id = nl.program_id(0)
    per_prog = div_ceil(div_ceil(numel, num_progs), P_MAX) * P_MAX
    core_base = prog_id * per_prog
    core_n = min(per_prog, numel - core_base)
    core_n = max(core_n, 0)

    # Leave the last group for the tail (n_aligned <= core_n-1) so no full-group
    # main-tile read can touch a truncated last group; the tail reads only up to
    # the lane. (For a P_MAX-aligned count this moves the final P_MAX groups to
    # the tail — a small, bounded portion.)
    n_aligned = (max(core_n - 1, 0) // P_MAX) * P_MAX
    # Bound the LOADED staging (stride*groups) per partition, not the group count,
    # so SBUF use is independent of `stride` (see MAX_STAGE_ELEMS).
    groups_per_tile = max(1, MAX_STAGE_ELEMS // stride)
    tile_groups = P_MAX * groups_per_tile
    num_tiles = div_ceil(n_aligned, tile_groups) if n_aligned > 0 else 0
    tail = core_n - n_aligned

    for i in nl.affine_range(num_tiles):
        t0 = core_base + i * tile_groups  # first group index in this tile
        masked_f = min(groups_per_tile, (n_aligned - i * tile_groups) // P_MAX)
        # 1. contiguous burst-load [P_MAX, stride*masked_f] groups (free stride 1)
        pk = nl.ndarray((P_MAX, stride * masked_f), dtype=grouped_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=pk,
            src=grouped_in.ap([[stride * masked_f, P_MAX], [1, stride * masked_f]], stride * t0),
        )
        # 2. on-chip lane select on the Vector Engine: read strided src[:, lane::stride]
        #    (bit-accurate copy), write contiguous result.
        sel = nl.ndarray((P_MAX, masked_f), dtype=grouped_in.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=sel, src=pk[:, lane : stride * masked_f : stride], engine=nisa.engine.vector
        )
        # 3. contiguous burst-store [P_MAX, masked_f] -> HBM
        nisa.dma_copy(dst=out.ap([[masked_f, P_MAX], [1, masked_f]], t0), src=sel)

    if tail > 0:
        t0 = core_base + n_aligned
        # Read only elements 0..lane of each tail group, never the trailing
        # (lane+1 .. stride-1). The lane element is the last valid element of a
        # minimal-span view, so this never over-reads a truncated last group.
        pk = nl.ndarray((tail, lane + 1), dtype=grouped_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=pk, src=grouped_in.ap([[stride, tail], [1, lane + 1]], stride * t0))
        sel = nl.ndarray((tail, 1), dtype=grouped_in.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sel, src=pk[:, lane : lane + 1], engine=nisa.engine.vector)
        nisa.dma_copy(dst=out.ap([[1, tail], [1, 1]], t0), src=sel)

    return out


def _flat_scatter_lane(grouped_io, vals, stride=2, lane=0):
    """Scatter contiguous ``vals`` into ``lane`` of a flat stride-``stride`` stream.

    The inverse of :func:`_flat_gather_lane`: burst-loads each group block (to
    preserve the lanes not being written), writes ``vals`` into element ``lane``
    of each group on-chip, and burst-stores the block back. ``numel`` is taken
    from ``vals.shape[0]``. ``stride=2, lane=p`` is the packed-int64 low/high
    word scatter.
    """
    numel = vals.shape[0]

    # A launch without a grid has no axis 0.
    if nl.program_ndim() == 0:
        num_progs, prog_id = 1, 0
    else:
        num_progs = nl.num_programs(0)
        prog_id = nl.program_id(0)
    per_prog = div_ceil(div_ceil(numel, num_progs), P_MAX) * P_MAX
    core_base = prog_id * per_prog
    core_n = min(per_prog, numel - core_base)
    core_n = max(core_n, 0)

    n_aligned = (max(core_n - 1, 0) // P_MAX) * P_MAX
    # Bound the LOADED staging (stride*groups) per partition — see gather core.
    groups_per_tile = max(1, MAX_STAGE_ELEMS // stride)
    tile_groups = P_MAX * groups_per_tile
    num_tiles = div_ceil(n_aligned, tile_groups) if n_aligned > 0 else 0
    tail = core_n - n_aligned

    for i in nl.affine_range(num_tiles):
        t0 = core_base + i * tile_groups
        masked_f = min(groups_per_tile, (n_aligned - i * tile_groups) // P_MAX)
        # 1. burst-load the group block (preserves the other lanes)
        pk = nl.ndarray((P_MAX, stride * masked_f), dtype=grouped_io.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=pk,
            src=grouped_io.ap([[stride * masked_f, P_MAX], [1, stride * masked_f]], stride * t0),
        )
        # 2. burst-load the contiguous source values
        v = nl.ndarray((P_MAX, masked_f), dtype=vals.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=v, src=vals.ap([[masked_f, P_MAX], [1, masked_f]], t0))
        # 3. write values into element `lane` of each group on chip (SBUF -> SBUF)
        nisa.dma_copy(dst=pk[:, lane : stride * masked_f : stride], src=v)
        # 4. burst-store the full group block back -> HBM
        nisa.dma_copy(
            dst=grouped_io.ap([[stride * masked_f, P_MAX], [1, stride * masked_f]], stride * t0),
            src=pk,
        )

    if tail > 0:
        t0 = core_base + n_aligned
        # Read/write only elements 0..lane of each tail group; the trailing
        # (lane+1 .. stride-1) is never touched — preserved if present, and never
        # over-read/written for a truncated last group (minimal-span storage).
        pk = nl.ndarray((tail, lane + 1), dtype=grouped_io.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=pk, src=grouped_io.ap([[stride, tail], [1, lane + 1]], stride * t0))
        v = nl.ndarray((tail, 1), dtype=vals.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=v, src=vals.ap([[1, tail], [1, 1]], t0))
        nisa.dma_copy(dst=pk[:, lane : lane + 1], src=v)
        nisa.dma_copy(dst=grouped_io.ap([[stride, tail], [1, lane + 1]], stride * t0), src=pk)

    return grouped_io


@nki.jit
def strided_flat_gather_kernel_(grouped_in, out, stride, lane):
    """Gather one lane from a flat grouped stream into an existing output.

    Args:
        grouped_in (nl.ndarray): Flat source arranged in groups of ``stride`` elements.
        out (nl.ndarray): Contiguous destination receiving one value per group.
        stride (int): Number of elements in each source group.
        lane (int): Zero-based element to select from each group.

    Returns:
        nl.ndarray: ``out`` after the selected values are written.

    Note:
        A stride of two selects low or high words from packed ``int64`` storage.
    """
    return _flat_gather_lane(grouped_in, out, stride, lane)


@nki.jit
def strided_flat_scatter_kernel(grouped_io, vals, stride, lane):
    """Scatter contiguous values into one lane of a flat grouped stream.

    Args:
        grouped_io (nl.ndarray): Grouped destination whose other lanes are preserved.
        vals (nl.ndarray): Contiguous values to scatter.
        stride (int): Number of elements in each destination group.
        lane (int): Zero-based element to replace in each group.

    Returns:
        nl.ndarray: ``grouped_io`` after the in-place updates.

    Note:
        A stride of two replaces low or high words in packed ``int64`` storage.
    """
    return _flat_scatter_lane(grouped_io, vals, stride, lane)


def run_strided_gather(src_tensor, dst_tensor, *, stride: int, lane: int = 0):
    """Gather a uniformly strided lane into a contiguous destination.

    Args:
        src_tensor (torch.Tensor): Source lane view with uniform storage stride.
        dst_tensor (torch.Tensor): Contiguous destination with the same logical shape.
        stride (int): Number of storage elements in each source group.
        lane (int, optional): Element selected from each group. Defaults to ``0``.

    Note:
        ``lane`` must equal ``src_tensor.storage_offset() % stride``. This function
        writes ``dst_tensor`` and uses a dynamic kernel when the supported
        packed-``int64`` conditions are satisfied.
    """
    numel = int(dst_tensor.numel())
    if numel == 0:
        return
    base_off = int(src_tensor.storage_offset()) - lane
    if base_off < 0:
        base_off = 0
    # Span only the valid storage from base_off — a minimal-span strided view
    # (e.g. x[:, ::2] copied to device) stops right after its last element, so
    # stride*numel would overshoot. The kernel reads at most stride*(numel-1)+lane
    # (< this span) and reads the last group only up to the lane.
    span = src_tensor.untyped_storage().nbytes() // src_tensor.element_size() - base_off
    grouped = src_tensor.as_strided((span,), (1,), storage_offset=base_off)
    out_flat = dst_tensor.reshape(numel)
    if (
        use_dynamic_nki_kernels()
        and stride == 2
        and lane == 0
        and grouped.dtype == torch.int32
        and out_flat.dtype == torch.int32
    ):
        from torch_neuronx.python_ops.torch_mlir.ops.dynamic_kernels.packed_int64_gather import (
            try_run_dynamic_packed_int64_gather,
        )

        if try_run_dynamic_packed_int64_gather(grouped, out_flat, numel):
            return
    strided_flat_gather_kernel_[LNC](grouped, out_flat, int(stride), int(lane))


def run_strided_scatter(src_tensor, dst_tensor, *, stride: int, lane: int = 0):
    """Scatter a contiguous source into a uniformly strided destination lane.

    Args:
        src_tensor (torch.Tensor): Contiguous source tensor.
        dst_tensor (torch.Tensor): Destination lane view whose other lanes are preserved.
        stride (int): Number of storage elements in each destination group.
        lane (int, optional): Element replaced in each group. Defaults to ``0``.

    Note:
        ``lane`` must equal ``dst_tensor.storage_offset() % stride``. This function
        updates ``dst_tensor`` in place, preserves other lanes, and safely handles
        minimal-span storage views.
    """
    numel = int(src_tensor.numel())
    if numel == 0:
        return
    base_off = int(dst_tensor.storage_offset()) - lane
    if base_off < 0:
        base_off = 0
    # Span only the valid storage from base_off (see run_strided_gather).
    span = dst_tensor.untyped_storage().nbytes() // dst_tensor.element_size() - base_off
    grouped = dst_tensor.as_strided((span,), (1,), storage_offset=base_off)
    src_flat = src_tensor.reshape(numel)
    strided_flat_scatter_kernel[LNC](grouped, src_flat, int(stride), int(lane))


@nki.jit
def stride2_flat_gather_kernel(packed_in, parity: int = 0):
    """Gather one lane from a flat stream of packed pairs.

    Args:
        packed_in (nl.ndarray): Flat contiguous ``int32`` stream of ``[low, high]`` pairs.
        parity (int, optional): Gather ``0`` for low or ``1`` for high. Defaults to ``0``.

    Returns:
        nl.ndarray: Flat ``int32[numel]`` output for ``packed_in`` length ``2 * numel``.

    Note:
        Allocating in shared HBM avoids external ``out`` operands with
        ``output_operand_aliases`` whose identical buffers CSE could merge, causing
        gathers to alias and clobber one another. It uses balanced flat tiling from
        ``_flat_gather_lane``.
    """
    numel = packed_in.shape[0] // 2
    out = nl.ndarray((numel,), dtype=nl.int32, buffer=nl.shared_hbm)
    return _flat_gather_lane(packed_in, out, 2, parity)


_wrapped_flat_gather = wrap_nki(stride2_flat_gather_kernel)


@nki_op("nki_kernels::packed_gather", mutates_args={})
def packed_gather_nki(packed: torch.Tensor, parity: int = 0, lnc: int = 0) -> torch.Tensor:
    """Gather one lane from a packed ``int32[..., 2]`` tensor.

    Args:
        packed (torch.Tensor): Tensor whose final dimension contains low/high words.
        parity (int, optional): Final-dimension lane to gather. Defaults to ``0``.
        lnc (int, optional): Logical NeuronCore count used for the launch; ``0`` selects
            the current logical core count. Defaults to ``0``.

    Returns:
        torch.Tensor: ``int32`` values with the packed tensor's logical shape.

    Note:
        The gather co-compiles into the caller's NEFF and returns an empty tensor
        without launching NKI when the input has no elements.
    """
    logical_shape = packed.shape[:-1]
    # Empty operand (e.g. cat / foreach with a size-0 int64 tensor): the NKI
    # kernel rejects a 0-sized shape ("shape must be a tuple of positive
    # integers"). Nothing to gather — return an empty int32 of the logical
    # shape. Mirrors the numel()==0 guard in packed_int64::execution_view (C++)
    # and nki_strided_copy_ (Python).
    if packed.numel() == 0:
        return torch.empty(logical_shape, dtype=torch.int32, device=packed.device)
    flat = packed.reshape(-1)
    lane = _wrapped_flat_gather[(lnc or int(get_logical_neuron_cores()),)](flat, parity)
    return lane.reshape(logical_shape)
