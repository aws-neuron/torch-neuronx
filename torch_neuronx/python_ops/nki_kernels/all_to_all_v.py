# ruff: noqa: N806
"""Variable-count all-to-all NKI kernel (``ncc.all_to_all_v``) for the eager backend.

A pure all-to-all-v: the input is already contiguously packed by destination rank and
the metadata (send counts / displacements) is precomputed host-side by the driver in
``torch_neuronx.distributed.ops.all_to_all_v``. Unlike ``permute_a2av`` (which also
gathers/scatters tokens), this kernel only moves the packed buffer across ranks.

Authoring constraints (see the KNKL reference kernels in
``experimental/collectives/collectives.py``):
- Collective src/dst/metadata HBM buffers MUST be given ``name="..."`` or the runtime
  raises an ``NCC_IBIR440`` DRAM allocation failure.
- Collective instructions CANNOT read/write IO (kernel input/output) tensors directly,
  so we ``nisa.dma_copy`` the input and metadata into intermediate ``shared_hbm`` buffers,
  run the collective, then ``dma_copy`` the result back out.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl


@nki.jit
def _all_reduce_max_hbm(input, replica_group):
    # input: [1, N] int32 IO tensor of local capacities. Fixed shape across ranks, so this
    # kernel compiles to an identical NEFF everywhere — required for the inNEFF collective.
    # Returns the element-wise max across the replica group (globally-uniform capacities).
    src = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm, name="a2av_cap_src")
    dst = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm, name="a2av_cap_dst")
    out = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm)
    # dma_copy required: Collective instruction cannot read/write IO tensors
    nisa.dma_copy(dst=src, src=input)
    ncc.all_reduce(srcs=[src], dsts=[dst], replica_group=replica_group, op=nl.maximum)
    nisa.dma_copy(dst=out, src=dst)
    return out


@nki.jit
def _all_to_all_v_hbm(input, metadata, replica_group, dst_rows, recv_counts_known, has_rdispls):
    # input:    [send_rows, H]      IO tensor (contiguously packed by dest rank)
    # metadata: [4, EP] uint32      send/recv counts and displacements
    # dst_rows: globally-uniform packed receive capacity
    # returns:  [dst_rows, H]       received blocks at their explicit displacements
    H = input.shape[1]
    # name= required on collective src/dst/meta: NCC_IBIR440 DRAM allocation failure without it
    src = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm, name="a2av_src")
    dst = nl.ndarray((dst_rows, H), dtype=input.dtype, buffer=nl.shared_hbm, name="a2av_dst")
    meta = nl.ndarray(metadata.shape, dtype=nl.uint32, buffer=nl.shared_hbm, name="a2av_meta")
    out = nl.ndarray((dst_rows, H), dtype=input.dtype, buffer=nl.shared_hbm)
    # dma_copy required: Collective instruction cannot read/write IO tensors
    nisa.dma_copy(dst=src, src=input)
    nisa.dma_copy(dst=meta, src=metadata)
    ncc.all_to_all_v(
        srcs=[src],
        dsts=[dst],
        replica_group=replica_group,
        metadata_tensor=meta,
        recv_counts_known=recv_counts_known,
        has_rdispls=has_rdispls,
    )
    nisa.dma_copy(dst=out, src=dst)
    return out
