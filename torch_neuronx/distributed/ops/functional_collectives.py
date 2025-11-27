"""Register custom implementations for functional collectives."""

import torch
from torch.distributed.distributed_c10d import _resolve_process_group


def all_gather_into_tensor_out(input, group_size, group_name, out):
    """
    Custom implementation as Torch code registers _allgather_base_ as C10d OP in ops.cpp
    and then it requires to have getBackend method supported on the process group object.
    NeuronPG does not have this method implemented, hence we provide this custom
    implementation here.
    """
    group = _resolve_process_group(group_name)
    opts = torch._C._distributed_c10d.AllgatherOptions()
    opts.asyncOp = True

    group._allgather_base(out, input, opts)
    return out


def register_neuron_collectives():
    torch.library.impl(
        "_c10d_functional::all_gather_into_tensor_out", "PrivateUse1", all_gather_into_tensor_out
    )
