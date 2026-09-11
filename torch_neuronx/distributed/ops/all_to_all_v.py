# ruff: noqa: N803, N806
"""Host-side driver for native variable-count all-to-all (``all_to_all_v``).

The C++ (`NeuronBackend`) backend routes its uneven-split ``alltoall_base`` path here by
default. Set ``NEURON_USE_NKI_ALLTOALLV=0`` to use the compatibility padding fallback.
The driver builds packed send/receive metadata host-side and invokes the NKI kernel via
``wrap_nki``.

The receive counts are known from ``output_split_sizes``, so metadata supplies explicit
receive displacements and the runtime writes each source block into one contiguous packed
prefix. This avoids the fixed-slot destination-size constraint imposed when
``has_rdispls=False``.

Globally-uniform buffer shapes (required for the inNEFF collective): every participating
rank MUST compile an IDENTICAL NEFF, which requires globally-uniform kernel IO tensor
SHAPES. Per-rank send and receive totals differ, so we first ``all_reduce(MAX)`` both
totals. We pad ``src`` to the uniform send capacity and size ``dst`` to the uniform
receive capacity. Only metadata VALUES vary between ranks, never buffer shapes.
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


def _build_packed_metadata(in_sizes, out_sizes, H, device):
    """Build packed ``[send counts, send displs, recv counts, recv displs]`` metadata.

    Counts and displacements are in ELEMENTS (rows * H). Build as int64 then cast to
    uint32 because direct CPU uint32 construction is not implemented by PyTorch.
    """
    ep = len(in_sizes)
    send_counts = torch.tensor(in_sizes, dtype=torch.int64) * H
    send_displs = torch.tensor(list(accumulate(in_sizes, initial=0))[:ep], dtype=torch.int64) * H
    recv_counts = torch.tensor(out_sizes, dtype=torch.int64) * H
    recv_displs = torch.tensor(list(accumulate(out_sizes, initial=0))[:ep], dtype=torch.int64) * H
    meta = torch.stack([send_counts, send_displs, recv_counts, recv_displs]).to(torch.uint32)
    return meta.to(device)  # [4, EP] uint32 on neuron


def _ensure_nonzero_recv_capacity(global_recv_capacity):
    """Keep the NKI destination shape valid for an inconsistent nonempty exchange."""
    return max(global_recv_capacity, 1)


def _round_up_to_group_size(rows, ep):
    """Pad a collective buffer's row count to a multiple of the replica-group size."""
    return ((rows + ep - 1) // ep) * ep


def nki_all_to_all_v_single(output, input, output_split_sizes, input_split_sizes, replica_groups):
    """Native uneven all-to-all-single via packed ``ncc.all_to_all_v`` metadata.

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
    recv_capacity_local = sum(output_split_sizes)
    caps_local = torch.tensor(
        [[send_cap_local, recv_capacity_local]], dtype=torch.int32, device=input.device
    )
    caps_global = wrap_nki(_all_reduce_max_hbm)[grid](caps_local, rg)  # [1, 2] int32
    caps_cpu = caps_global.to("cpu")
    global_send_cap = int(caps_cpu[0, 0])
    global_recv_capacity = int(caps_cpu[0, 1])

    # Every rank observes the same reduced capacities. If the entire group is empty,
    # the capacity all-reduce has provided the required synchronization and there is
    # no payload collective to launch.
    if global_send_cap == 0 and global_recv_capacity == 0:
        return output

    # Trn2's all_to_all_v implementation requires each data buffer's total element count
    # to be divisible by rank_n. Padding rows to an EP multiple satisfies that constraint
    # for every H while metadata continues to address only the real packed prefixes.
    global_send_cap = _round_up_to_group_size(global_send_cap, ep)
    global_recv_capacity = _round_up_to_group_size(global_recv_capacity, ep)

    # Defensive shape guard: a valid exchange with any sends also has receives, but
    # retain one destination row if an inconsistent caller reaches this path.
    dst_rows = _ensure_nonzero_recv_capacity(global_recv_capacity)

    # Pad src to the uniform send capacity (send_counts/send_displs still index within the
    # real packed prefix, so the padding rows are never read by the collective).
    if global_send_cap > send_cap_local:
        src2d = torch.zeros(global_send_cap, H, dtype=in2d.dtype, device=in2d.device)
        if send_cap_local > 0:
            src2d[:send_cap_local].copy_(in2d)
    else:
        src2d = in2d

    meta = _build_packed_metadata(input_split_sizes, output_split_sizes, H, input.device)
    result = wrap_nki(_all_to_all_v_hbm)[grid](
        src2d,
        meta,
        rg,
        int(dst_rows),
        recv_counts_known=True,
        has_rdispls=True,
    )  # [dst_rows, H]; received blocks form a packed source-major prefix

    # Explicit receive displacements make the result a packed source-major prefix.
    src_off = 0
    if isinstance(output, list):
        for r, out in enumerate(output):
            n = output_split_sizes[r]
            if n > 0:
                src = result[src_off : src_off + n]
                out.copy_(src.reshape(out.shape))
            src_off += n
    else:
        recv_rows = sum(output_split_sizes)
        if recv_rows > 0:
            output.copy_(result[:recv_rows].reshape(output.shape))
    return output
