"""NKI kernels for stride-2 last-dim gather and scatter.

Used for packed int64 low-/high-word extraction and materialization. An int64
tensor is stored on device as pairs of int32 lanes ``[lo, hi, lo, hi, ...]``;
these kernels move one lane (selected by ``parity``) to/from a contiguous int32
buffer.

Balanced flat tiling
--------------------
Gather/scatter is purely element-wise over a flat contiguous int32 pair-stream,
so the logical tensor shape is irrelevant to correctness. We exploit that to
flatten the stream to ``numel`` pairs and tile it as balanced ``[P_MAX, F_SIZE]``
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
  3. burst-store the contiguous result ``[P_MAX, F_SIZE]`` SBUF -> HBM via ``.ap``.

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
dimension (hardware-parallel, == 128) and ``F_SIZE`` is the free-dimension tile width.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from torch_neuronx.nki_hop import nki_op, wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

# Partition dimension: the hardware partition count.
P_MAX = 128
# Free-dimension tile width, in *pairs*, for the SCATTER kernel (dma_copy-based
# lane write). Bounds the SBUF staging buffer to P_MAX * 2 * F_SIZE int32 elements.
# Performance is insensitive above ~512; 2048 keeps the launch count low.
F_SIZE = 2048

LNC = int(get_logical_neuron_cores())


def div_ceil(numerator: int, denominator: int) -> int:
    """Ceiling division. Avoids inline ``(n + d - 1) // d`` per NKI conventions."""
    return -(-numerator // denominator)


def _flat_gather_lane(packed_in, out, parity):
    """Gather lane ``parity`` of a flat packed pair-stream into ``out``.

    Shared tiling for the in-place (``stride2_flat_gather_kernel_``) and
    out-of-place (``stride2_flat_gather_kernel``) entry points — a plain function
    that emits NKI ops into the caller's kernel trace (mirrors the helper
    pattern in ``nonzero.py``). ``numel`` is taken from ``out.shape[0]``.

    Balanced flat tiling: SPMD split into P_MAX-aligned per-program slices, then
    [P_MAX, F_SIZE] pair blocks with a <P_MAX tail. Each tile burst-loads packed
    pairs, selects lane ``parity`` on the Vector Engine (``tensor_copy`` is a
    bit-accurate int32 copy, unlike Scalar-Engine ``nl.copy`` which casts through
    fp32), and burst-stores contiguous words.
    """
    numel = out.shape[0]

    num_progs = nl.num_programs(0)
    prog_id = nl.program_id(0)
    per_prog = div_ceil(div_ceil(numel, num_progs), P_MAX) * P_MAX
    core_base = prog_id * per_prog
    core_n = min(per_prog, numel - core_base)
    core_n = max(core_n, 0)

    n_aligned = (core_n // P_MAX) * P_MAX
    tile_pairs = P_MAX * F_SIZE
    num_tiles = div_ceil(n_aligned, tile_pairs) if n_aligned > 0 else 0
    tail = core_n - n_aligned

    for i in nl.affine_range(num_tiles):
        t0 = core_base + i * tile_pairs  # first pair index in this tile
        masked_f = min(F_SIZE, (n_aligned - i * tile_pairs) // P_MAX)
        # 1. contiguous burst-load [P_MAX, 2*masked_f] packed pairs (free stride 1)
        pk = nl.ndarray((P_MAX, 2 * masked_f), dtype=packed_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=pk, src=packed_in.ap([[2 * masked_f, P_MAX], [1, 2 * masked_f]], 2 * t0))
        # 2. on-chip lane select on the Vector Engine: read strided src[:, parity::2]
        #    (bit-accurate int32 copy), write contiguous result.
        sel = nl.ndarray((P_MAX, masked_f), dtype=packed_in.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sel, src=pk[:, parity : 2 * masked_f : 2], engine=nisa.engine.vector)
        # 3. contiguous burst-store [P_MAX, masked_f] -> HBM
        nisa.dma_copy(dst=out.ap([[masked_f, P_MAX], [1, masked_f]], t0), src=sel)

    if tail > 0:
        t0 = core_base + n_aligned
        pk = nl.ndarray((tail, 2), dtype=packed_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=pk, src=packed_in.ap([[2, tail], [1, 2]], 2 * t0))
        sel = nl.ndarray((tail, 1), dtype=packed_in.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sel, src=pk[:, parity : parity + 1], engine=nisa.engine.vector)
        nisa.dma_copy(dst=out.ap([[1, tail], [1, 1]], t0), src=sel)

    return out


@nki.jit
def stride2_flat_gather_kernel_(packed_in, out, parity):
    """Gather one lane of a flat packed pair-stream into a contiguous buffer.

    Args:
        packed_in: flat int32 [2*numel] contiguous packed pairs ``[lo,hi,...]``,
            rebased so element ``parity`` of each pair is the requested lane.
        out: flat int32 [numel] contiguous output.
        parity: lane to gather — 0 = low word, 1 = high word.

    Returns:
        out (written in place; also returned per NKI convention).

    Notes:
        Balanced flat tiling: the stream is tiled as [P_MAX, F_SIZE] pair-blocks
        independent of logical shape, so the partition dim is always full for
        the bulk. A <P_MAX remainder is handled by a separate tail tile.

        Lane select uses nisa.tensor_copy on the Vector Engine, reading the
        strided SBUF source ``pk[:, parity::2]`` and writing a contiguous tile.
        On the Vector Engine tensor_copy is a bit-accurate copy when src/dst
        dtypes match (int32==int32), so unlike nl.copy (Scalar Engine, which
        casts through fp32 and corrupts int32) it preserves all 32 bits. The
        Vector Engine does the on-chip select while the DMA engines overlap the
        HBM bursts, and it is ~2x faster than the GpSimd nc_n_gather select and
        ~4x faster than a strided dma_copy select.

        (Vector tensor_copy has no multi-tile size limit — it is correct at
        F=2048 across the full shape sweep. This is unlike nc_n_gather, which
        decomposes into ceil(elems/512) ISA groups and corrupts later tiles when
        chained at >1 group; ticket V2245128914.)

        SPMD: when launched with LNC2 the stream is split across the two
        programs (P_MAX-aligned per-program slice). With LNC1 there is a single
        program over the whole stream.
    """
    return _flat_gather_lane(packed_in, out, parity)


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

    # SPMD split — see stride2_flat_gather_kernel_ for the rationale.
    num_progs = nl.num_programs(0)
    prog_id = nl.program_id(0)
    per_prog = div_ceil(div_ceil(numel, num_progs), P_MAX) * P_MAX
    core_base = prog_id * per_prog
    core_n = min(per_prog, numel - core_base)
    core_n = max(core_n, 0)

    n_aligned = (core_n // P_MAX) * P_MAX
    tile_pairs = P_MAX * F_SIZE
    num_tiles = div_ceil(n_aligned, tile_pairs) if n_aligned > 0 else 0
    tail = core_n - n_aligned

    for i in nl.affine_range(num_tiles):
        t0 = core_base + i * tile_pairs
        masked_f = min(F_SIZE, (n_aligned - i * tile_pairs) // P_MAX)
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
    stride2_flat_gather_kernel_[LNC](packed, out_flat, int(parity))


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


@nki.jit
def stride2_flat_gather_kernel(packed_in, parity: int = 0):
    """Out-of-place gather of one lane from a flat packed pair-stream.

    Args:
        packed_in: flat int32 [2*numel] contiguous ``[lo, hi, ...]`` pairs (the
            physical view of a contiguous int64 tensor).
        parity: lane to gather — 0 = low words (default), 1 = high words.

    Returns:
        out: flat int32 [numel] contiguous lane words, allocated in shared_hbm.

    Out-of-place on purpose: when several packed inputs are gathered in one
    graph, an in-place kernel's ``out`` buffer becomes a custom_call operand with
    ``output_operand_aliases``, and the identically-shaped ``torch.empty``
    buffers get CSE'd into a single constant — so the gathers alias and clobber
    each other. Allocating the output internally gives each call an independent
    result. Shares the tiling with ``stride2_flat_gather_kernel_`` via
    ``_flat_gather_lane``.
    """
    numel = packed_in.shape[0] // 2
    out = nl.ndarray((numel,), dtype=nl.int32, buffer=nl.shared_hbm)
    return _flat_gather_lane(packed_in, out, parity)


_wrapped_flat_gather = wrap_nki(stride2_flat_gather_kernel)


@nki_op("nki_kernels::packed_gather", mutates_args={})
def packed_gather_nki(packed: torch.Tensor, parity: int = 0, lnc: int = 1) -> torch.Tensor:
    """Gather one lane (low words by default) from a packed int32[..., 2] tensor.

    Called inside a traced ``torch_fn`` so the gather co-compiles into the op's
    NEFF. Returns int32 with the input's logical shape.
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
    lane = _wrapped_flat_gather[(lnc,)](flat, parity)
    return lane.reshape(logical_shape)
