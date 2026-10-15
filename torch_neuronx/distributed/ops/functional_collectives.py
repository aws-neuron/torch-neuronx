"""Register custom implementations for functional collectives."""

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _resolve_process_group

_REDUCE_OP_MAP = {
    "sum": dist.ReduceOp.SUM,
    "avg": dist.ReduceOp.AVG,
    "product": dist.ReduceOp.PRODUCT,
    "min": dist.ReduceOp.MIN,
    "max": dist.ReduceOp.MAX,
}


def all_gather_into_tensor_out(input, group_size, group_name, out):
    """
    Custom implementation as Torch code registers _allgather_base_ as C10d OP in ops.cpp
    and then it requires to have getBackend method supported on the process group object.
    NeuronPG does not have this method implemented, hence we provide this custom
    implementation here.
    """
    group = _resolve_process_group(group_name)
    opts = torch._C._distributed_c10d.AllgatherOptions()
    opts.asyncOp = False

    group._allgather_base(out, input, opts)
    return out


def all_gather_into_tensor(input, group_size, group_name):
    """Neuron implementation of _c10d_functional::all_gather_into_tensor.

    Uses asyncOp=false to avoid the CompositeExplicitAutograd fallback retaining
    tensor references in NeuronWork's Future. Functional collective callers
    obtain results via the returned tensor and wait_tensor, not future.value().
    """
    group = _resolve_process_group(group_name)
    output = input.new_empty(input.shape[0] * group_size, *input.shape[1:])

    opts = torch._C._distributed_c10d.AllgatherOptions()
    opts.asyncOp = False

    work = group._allgather_base(output, input, opts)
    torch._C._distributed_c10d._register_work(output, work)
    return output


def reduce_scatter_tensor(input, reduce_op, group_size, group_name):
    """Neuron implementation of _c10d_functional::reduce_scatter_tensor.

    Uses asyncOp=false to avoid the CompositeExplicitAutograd fallback retaining
    tensor references in NeuronWork's Future. Functional collective callers
    obtain results via the returned tensor and wait_tensor, not future.value().
    """
    group = _resolve_process_group(group_name)
    output = input.new_empty(input.shape[0] // group_size, *input.shape[1:])

    opts = torch._C._distributed_c10d.ReduceScatterOptions()
    opts.asyncOp = False
    opts.reduceOp = _REDUCE_OP_MAP[reduce_op]

    work = group._reduce_scatter_base(output, input, opts)
    torch._C._distributed_c10d._register_work(output, work)
    return output


def reduce_scatter_tensor_coalesced(inputs, reduce_op, group_size, group_name):
    """Neuron implementation of _c10d_functional::reduce_scatter_tensor_coalesced."""
    group = _resolve_process_group(group_name)
    outputs = [t.new_empty(t.shape[0] // group_size, *t.shape[1:]) for t in inputs]

    opts = torch._C._distributed_c10d.ReduceScatterOptions()
    opts.asyncOp = False
    opts.reduceOp = _REDUCE_OP_MAP[reduce_op]

    work = group.reduce_scatter_tensor_coalesced(outputs, inputs, opts)
    for output in outputs:
        torch._C._distributed_c10d._register_work(output, work)
    return outputs


def all_reduce(input, reduce_op, group_name):
    """Neuron implementation of _c10d_functional::all_reduce.

    The CompositeExplicitAutograd fallback clones the input and does an in-place
    allreduce on the clone, but the resulting Work/Future lifecycle doesn't
    synchronize correctly with torch_neuronx.synchronize(), causing one rank to
    hang. This PrivateUse1 dispatch avoids that path entirely.
    """
    output = input.clone(memory_format=torch.contiguous_format)
    group = _resolve_process_group(group_name)

    opts = torch._C._distributed_c10d.AllreduceOptions()
    opts.asyncOp = False
    opts.reduceOp = _REDUCE_OP_MAP[reduce_op]

    work = group.allreduce([output], opts)
    torch._C._distributed_c10d._register_work(output, work)
    return output


def all_reduce_coalesced(inputs, reduce_op, group_name):
    """Neuron implementation of _c10d_functional::all_reduce_coalesced.

    Inputs are left unmodified; returns reduced clones. Same asyncOp=False
    rationale as all_reduce.
    """
    outputs = [t.clone(memory_format=torch.contiguous_format) for t in inputs]
    group = _resolve_process_group(group_name)

    opts = torch._C._distributed_c10d.AllreduceCoalescedOptions()
    opts.asyncOp = False
    opts.reduceOp = _REDUCE_OP_MAP[reduce_op]

    work = group.allreduce_coalesced(outputs, opts)
    for output in inputs:
        torch._C._distributed_c10d._register_work(output, work)
    return outputs


def register_neuron_collectives():
    torch.library.impl(
        "_c10d_functional::all_gather_into_tensor_out", "PrivateUse1", all_gather_into_tensor_out
    )
    torch.library.impl(
        "_c10d_functional::all_gather_into_tensor", "PrivateUse1", all_gather_into_tensor
    )
    torch.library.impl(
        "_c10d_functional::reduce_scatter_tensor", "PrivateUse1", reduce_scatter_tensor
    )
    torch.library.impl(
        "_c10d_functional::reduce_scatter_tensor_coalesced",
        "PrivateUse1",
        reduce_scatter_tensor_coalesced,
    )
    torch.library.impl("_c10d_functional::all_reduce", "PrivateUse1", all_reduce)
    torch.library.impl(
        "_c10d_functional::all_reduce_coalesced", "PrivateUse1", all_reduce_coalesced
    )
