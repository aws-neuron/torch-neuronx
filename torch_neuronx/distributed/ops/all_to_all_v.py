# ruff: noqa: N803, N806
"""Host-side driver for native variable-count all-to-all (``all_to_all_v``).

The C++ (`NeuronBackend`) backend routes its uneven-split ``alltoall_base`` path here by
default. Set ``NEURON_USE_NKI_ALLTOALLV=0`` to use the compatibility padding fallback.
The driver builds the layout-F metadata host-side, invokes the NKI kernel via ``wrap_nki``,
then gathers the runtime's equally-spaced receive layout into a contiguous output or
directly into the list-form outputs.

Layout F (``recv_counts_known=False, has_rdispls=False``): the send side is packed
exactly, but the runtime places rank r's received chunk equally-spaced at element offset
``(dst.total_elements / rg_size) * r``. We over-allocate ``dst`` as
``[rg_size * global_recv_rows, H]`` so chunk r lands at row ``r * global_recv_rows``, then
host-gather the actual ``output_split_sizes[r]`` rows into ``output``. See
``.internal_docs/all_to_all_v_proposal.md`` (§3, §4.2) for the full rationale.

Globally-uniform buffer shapes (required for the inNEFF collective): every participating
rank MUST compile an IDENTICAL NEFF, which requires globally-uniform kernel IO tensor
SHAPES. But per-rank send counts (``sum(input_split_sizes)``) and receive capacities
(``max(output_split_sizes)``) differ across ranks, so we first ``all_reduce(MAX)`` both
to obtain a uniform send capacity and receive-rows-per-slot. We then pad ``src`` to the
uniform send capacity and size ``dst`` from the uniform recv rows, so every rank's kernel
has identical shapes → identical NEFF. Only the metadata VALUES (counts/displs) vary,
never the buffer shapes — mirroring ``permute_a2av``'s fixed ``[EP*T, H]`` capacity.
"""

from itertools import accumulate

import torch
from nki.collectives import ReplicaGroup

from torch_neuronx import wrap_nki
from torch_neuronx.python_ops.nki_kernels.all_to_all_v import (
    _all_reduce_max_hbm,
    _all_to_all_v_hbm,
)
from torch_neuronx.utils import get_logical_neuron_cores


def _build_layout_f_metadata(in_sizes, H, device):
    """Build the (3, EP) uint32 layout-F metadata tensor on ``device``.

    Only rows 0,1 are inputs; row 2 is a runtime writeback slot (left zero). Counts and
    displacements are in ELEMENTS (rows * H). Build as int64 then cast to uint32 — CPU
    ``torch.arange``/direct uint32 construction raises ``NotImplementedError`` for uint32.
    """
    ep = len(in_sizes)
    send_counts = torch.tensor(in_sizes, dtype=torch.int64) * H
    send_displs = torch.tensor(list(accumulate(in_sizes, initial=0))[:ep], dtype=torch.int64) * H
    zeros = torch.zeros(ep, dtype=torch.int64)  # row 2 (recv_counts writeback)
    meta = torch.stack([send_counts, send_displs, zeros]).to(torch.uint32)  # CPU build
    return meta.to(device)  # [3, EP] uint32 on neuron


def _ensure_nonzero_recv_rows(global_recv_rows):
    """Keep the NKI destination shape valid when every receive count is zero."""
    return max(global_recv_rows, 1)


def nki_all_to_all_v_single(output, input, output_split_sizes, input_split_sizes, replica_groups):
    """Native uneven all-to-all-single via the ``ncc.all_to_all_v`` NKI kernel (layout F).

    Args:
        output: Destination tensor or list of per-rank destination tensors. A tensor
            receives ``output_split_sizes[r]`` rows from rank r, laid out contiguously
            in rank order. A list receives each rank's rows directly in ``output[r]``.
        input: Source tensor, already contiguously packed by destination rank.
        output_split_sizes: Per-rank receive ROW counts (concrete list).
        input_split_sizes: Per-rank send ROW counts (concrete list).
        replica_groups: Nested rank-list (``list[list[int]]``) for the replica group.
    """
    # split sizes are already concrete per-rank ROW lists (None normalized by caller).
    H = 1 if input.dim() == 1 else int(input.shape[1:].numel())
    in2d = input.reshape(input.shape[0], H)
    ep = len(input_split_sizes)

    rg = ReplicaGroup(replica_groups)
    grid = (int(get_logical_neuron_cores()),)  # LNC (validated == 2 on TRN3 by caller)

    # Make send/recv buffer capacities globally uniform so every rank compiles an
    # IDENTICAL all_to_all_v NEFF. Local capacities differ per rank; all_reduce(MAX)
    # over the replica group yields uniform values. [1, 2] int32: [send_rows, recv_rows].
    send_cap_local = int(in2d.shape[0])
    recv_rows_local = max(output_split_sizes) if output_split_sizes else 0
    caps_local = torch.tensor(
        [[send_cap_local, recv_rows_local]], dtype=torch.int32, device=input.device
    )
    caps_global = wrap_nki(_all_reduce_max_hbm)[grid](caps_local, rg)  # [1, 2] int32
    caps_cpu = caps_global.to("cpu")
    global_send_cap = int(caps_cpu[0, 0])
    global_recv_rows = int(caps_cpu[0, 1])

    # Every rank observes the same reduced capacities. If the entire group is empty,
    # the capacity all-reduce has provided the required synchronization and there is
    # no payload collective to launch.
    if global_send_cap == 0 and global_recv_rows == 0:
        return output

    # Defensive shape guard: a valid exchange with any sends also has receives, but
    # retain one destination row per rank if an inconsistent caller reaches this path.
    global_recv_rows = _ensure_nonzero_recv_rows(global_recv_rows)
    dst_rows = ep * global_recv_rows

    # Pad src to the uniform send capacity (send_counts/send_displs still index within the
    # real packed prefix, so the padding rows are never read by the collective).
    if global_send_cap > send_cap_local:
        src2d = torch.zeros(global_send_cap, H, dtype=in2d.dtype, device=in2d.device)
        if send_cap_local > 0:
            src2d[:send_cap_local].copy_(in2d)
    else:
        src2d = in2d

    meta = _build_layout_f_metadata(input_split_sizes, H, input.device)
    result = wrap_nki(_all_to_all_v_hbm)[grid](
        src2d,
        meta,
        rg,
        int(dst_rows),
        recv_counts_known=False,
        has_rdispls=False,
    )  # [dst_rows, H]; chunk r at row r*global_recv_rows

    # Layout-F gather: pull actual out_sizes[r] rows from row r*global_recv_rows.
    if isinstance(output, list):
        for r, out in enumerate(output):
            n = output_split_sizes[r]
            if n > 0:
                src = result[r * global_recv_rows : r * global_recv_rows + n]
                out.copy_(src.reshape(out.shape))
    else:
        dst_off = 0
        for r in range(ep):
            n = output_split_sizes[r]
            if n > 0:
                out = output[dst_off : dst_off + n]
                src = result[r * global_recv_rows : r * global_recv_rows + n]
                out.copy_(src.reshape(out.shape))
            dst_off += n
    return output
